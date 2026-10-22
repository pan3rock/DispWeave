#pragma once
#include "dispweave/nuts.hpp"
#include <algorithm>
namespace dw::detail {
// The endpoint test is expressed in whitened coordinates: delta.q dot M^-1 p.
inline bool continuing(const HamiltonianState& left,const HamiltonianState& right,const EuclideanMetric& metric) {
    const Eigen::VectorXd delta=right.position-left.position;
    return delta.dot(metric.velocity(left.momentum))>=0 && delta.dot(metric.velocity(right.momentum))>=0;
}
struct Tree {
    HamiltonianState left,right,candidate;
    int count=0,evaluations=0;
    bool go=true,divergent=false;
    double alpha=0,max_error=0;
    EvaluationStatus status=EvaluationStatus::valid;
};
class Builder {
public:
    const LogDensity& target;
    const EuclideanMetric& metric;
    std::mt19937_64& rng;
    double step,log_slice,joint0,threshold;
    double uniform() {return std::generate_canonical<double,53>(rng);}
    Tree build(const HamiltonianState& start,int direction,int depth) {
        if(depth==0) {
            auto integrated=leapfrog(target,start,metric,direction*step,1);
            Tree t; t.left=t.right=t.candidate=std::move(integrated.state);
            t.evaluations=integrated.evaluations;t.status=integrated.status;
            const double joint=t.status==EvaluationStatus::valid?
                t.candidate.density.value-metric.kinetic(t.candidate.momentum):-INFINITY;
            const double error=joint0-joint;
            if(!std::isfinite(joint) || !std::isfinite(error)) {
                if(t.status==EvaluationStatus::valid) t.status=EvaluationStatus::numerical_failure;
                t.go=false;t.divergent=true;t.max_error=INFINITY;return t;
            }
            t.count=log_slice<=joint;
            t.go=log_slice<joint+threshold;
            t.divergent=!t.go;t.max_error=std::abs(error);
            t.alpha=std::exp(std::min(0.,-error));return t;
        }
        auto a=build(start,direction,depth-1);
        if(!a.go) return a;
        auto b=build(direction<0?a.left:a.right,direction,depth-1);
        if(direction<0) a.left=b.left;else a.right=b.right;
        // Uniform reservoir within a subtree; a zero-size subtree has no candidate.
        if(b.count>0 && uniform()<double(b.count)/(a.count+b.count)) a.candidate=b.candidate;
        a.count+=b.count;a.evaluations+=b.evaluations;a.alpha+=b.alpha;
        a.max_error=std::max(a.max_error,b.max_error);a.divergent|=b.divergent;
        if(b.status!=EvaluationStatus::valid) a.status=b.status;
        a.go=b.go && continuing(a.left,a.right,metric);return a;
    }
};
}
