#pragma once
#include "dispweave/nuts.hpp"
#include <algorithm>
#include <cmath>
// Multinomial weights, biased progressive outer selection, and the three
// momentum-sum U-turn checks follow Stan's BSD-3-Clause base_nuts.hpp.
// Pinned source and license: third_party/stan-nuts-reference/.
namespace dw::detail {
inline double log_add_weights(double a,double b) {
    if(a==-INFINITY) return b;
    if(b==-INFINITY) return a;
    return std::max(a,b)+std::log1p(std::exp(-std::abs(a-b)));
}
struct WeightedTree {
    HamiltonianState left,right,candidate;
    Eigen::VectorXd rho;
    double log_weight=-INFINITY,alpha=0,max_error=0;
    int evaluations=0;
    bool go=true,divergent=false;
    EvaluationStatus status=EvaluationStatus::valid;
};
// a and b are in physical time order, independent of construction direction.
inline bool weighted_continuing(const WeightedTree& a,const WeightedTree& b,const EuclideanMetric& metric) {
    const auto va=metric.velocity(a.left.momentum),vb=metric.velocity(a.right.momentum);
    const auto vc=metric.velocity(b.left.momentum),vd=metric.velocity(b.right.momentum);
    const Eigen::VectorXd total=a.rho+b.rho,left=a.rho+b.left.momentum,right=b.rho+a.right.momentum;
    return va.dot(total)>0 && vd.dot(total)>0 && va.dot(left)>0 && vc.dot(left)>0
        && vb.dot(right)>0 && vd.dot(right)>0;
}
class MultinomialBuilder {
public:
    const LogDensity& target;
    const EuclideanMetric& metric;
    std::mt19937_64& rng;
    double step,joint0,threshold;
    double uniform() {return std::generate_canonical<double,53>(rng);}
    WeightedTree build(const HamiltonianState& start,int direction,int depth) {
        if(depth==0) {
            auto integrated=leapfrog(target,start,metric,direction*step,1);
            WeightedTree t;t.left=t.right=t.candidate=std::move(integrated.state);
            t.rho=t.candidate.momentum;t.evaluations=integrated.evaluations;t.status=integrated.status;
            if(t.status==EvaluationStatus::outside_support) {t.go=false;t.max_error=INFINITY;return t;}
            const double joint=t.status==EvaluationStatus::valid?
                t.candidate.density.value-metric.kinetic(t.candidate.momentum):-INFINITY;
            const double error=joint0-joint;
            if(!std::isfinite(joint) || !std::isfinite(error)) {
                if(t.status==EvaluationStatus::valid) t.status=EvaluationStatus::numerical_failure;
                t.go=false;t.divergent=true;t.max_error=INFINITY;return t;
            }
            t.log_weight=-error;t.alpha=std::exp(std::min(0.,-error));t.max_error=std::abs(error);
            t.divergent=error>threshold;t.go=!t.divergent;return t;
        }
        auto a=build(start,direction,depth-1);
        if(!a.go) return a;
        auto b=build(direction<0?a.left:a.right,direction,depth-1);
        a.evaluations+=b.evaluations;a.alpha+=b.alpha;
        a.max_error=std::max(a.max_error,b.max_error);a.divergent|=b.divergent;
        if(b.status!=EvaluationStatus::valid) a.status=b.status;
        if(!b.go) {a.go=false;return a;}
        const double weight=log_add_weights(a.log_weight,b.log_weight);
        if(uniform()<std::exp(b.log_weight-weight)) a.candidate=b.candidate;
        a.go=direction<0?weighted_continuing(b,a,metric):weighted_continuing(a,b,metric);
        if(direction<0) a.left=std::move(b.left);else a.right=std::move(b.right);
        a.rho+=b.rho;a.log_weight=weight;return a;
    }
};
struct MultinomialTransition {
    HamiltonianState state;
    NUTSTransition diagnostic;
};
inline MultinomialTransition multinomial_transition(const LogDensity& target,const HamiltonianState& start,
        const EuclideanMetric& metric,std::mt19937_64& rng,double step,int max_depth,double threshold,bool warmup) {
    const double joint0=start.density.value-metric.kinetic(start.momentum);
    MultinomialBuilder builder{target,metric,rng,step,joint0,threshold};
    WeightedTree trajectory;trajectory.left=trajectory.right=trajectory.candidate=start;
    trajectory.rho=start.momentum;trajectory.log_weight=0;
    int depth=0,evaluations=0;double alpha=0,max_error=0;
    bool go=true,divergent=false,selected=false;auto status=EvaluationStatus::valid;
    while(go && depth<max_depth) {
        const int direction=builder.uniform()>.5?1:-1;
        auto tree=builder.build(direction<0?trajectory.left:trajectory.right,direction,depth);
        ++depth;evaluations+=tree.evaluations;alpha+=tree.alpha;
        divergent|=tree.divergent;max_error=std::max(max_error,tree.max_error);
        if(tree.status!=EvaluationStatus::valid) status=tree.status;
        if(!tree.go) {go=false;break;}
        // Biased progressive transition, NOT a reservoir over old+new weight.
        if(tree.log_weight>trajectory.log_weight || builder.uniform()<std::exp(tree.log_weight-trajectory.log_weight)) {
            trajectory.candidate=tree.candidate;selected=true;
        }
        go=direction<0?weighted_continuing(tree,trajectory,metric):weighted_continuing(trajectory,tree,metric);
        trajectory.rho+=tree.rho;
        trajectory.log_weight=log_add_weights(trajectory.log_weight,tree.log_weight);
        if(direction<0) trajectory.left=std::move(tree.left);else trajectory.right=std::move(tree.right);
    }
    return {std::move(trajectory.candidate),{warmup,selected,divergent,go && depth==max_depth,
        depth,evaluations,step,alpha/evaluations,max_error,status}};
}
// Warmup-only heuristic. All probes start from the same cached valid state;
// the search changes neither position nor gradient. Attempts are bounded.
inline StepSizeSearch find_step_size(const LogDensity& target,const Eigen::VectorXd& q,const DensityEvaluation& density,
        const EuclideanMetric& metric,std::mt19937_64& rng,double step,int iteration) {
    constexpr int max_attempts=50;
    int direction=0,evaluations=0;
    for(int attempt=0;attempt<max_attempts;++attempt) {
        HamiltonianState start{q,metric.momentum(rng),density};
        const double joint0=density.value-metric.kinetic(start.momentum);
        if(!std::isfinite(joint0)) throw std::runtime_error("Nonfinite initial step-search Hamiltonian");
        const auto trial=leapfrog(target,start,metric,step,1);evaluations+=trial.evaluations;
        const double delta=trial.status==EvaluationStatus::valid?
            trial.state.density.value-metric.kinetic(trial.state.momentum)-joint0:-INFINITY;
        const bool sufficient=std::isfinite(delta) && delta>std::log(.8);
        if(direction==0) direction=sufficient?1:-1;
        else if((direction>0 && !sufficient) || (direction<0 && sufficient))
            return {iteration,evaluations,step,trial.status};
        step*=direction>0?2.:.5;
        if(!std::isfinite(step) || step<=0) break;
    }
    throw std::runtime_error("Unable to bracket a reasonable step size within 50 probes");
}
// Experimental warmup-only conservative search: four common momenta, no
// expansion, at most 12 trials (48 target calls). Never changes the position.
inline StepSizeSearch find_conservative_step(const LogDensity& target,const Eigen::VectorXd& q,const DensityEvaluation& density,
        const EuclideanMetric& metric,std::mt19937_64& rng,double step,int iteration) {
    std::vector<HamiltonianState> starts;
    for(int j=0;j<4;++j) starts.push_back({q,metric.momentum(rng),density});
    int evaluations=0;
    for(int attempt=0;attempt<12;++attempt) {
        bool acceptable=true;
        for(const auto& start:starts) {
            const double joint0=density.value-metric.kinetic(start.momentum);
            if(!std::isfinite(joint0)) throw std::runtime_error("Nonfinite conservative search Hamiltonian");
            const auto trial=leapfrog(target,start,metric,step,1);evaluations+=trial.evaluations;
            const double delta=trial.status==EvaluationStatus::valid?
                trial.state.density.value-metric.kinetic(trial.state.momentum)-joint0:-INFINITY;
            acceptable=acceptable && std::isfinite(delta) && delta>=std::log(.8);
        }
        if(acceptable) return {iteration,evaluations,step,EvaluationStatus::valid};
        step*=.5;
        if(step==0) break;
    }
    throw std::runtime_error("Unable to find a conservative step within 12 trials");
}
}
