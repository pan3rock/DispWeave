#include "dispweave/lm.hpp"
#include <Eigen/QR>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace dw {
LMResult bounded_lm(const LeastSquaresFunction& function,const Eigen::VectorXd& initial,
                    const Eigen::VectorXd& lower,const Eigen::VectorXd& upper,const LMOptions& options) {
    const auto n=initial.size();
    if(n==0 || lower.size()!=n || upper.size()!=n || !lower.allFinite() || !upper.allFinite()
       || !(upper-lower).allFinite() || (lower.array()>=upper.array()).any() || options.max_iterations<1
       || !std::isfinite(options.gradient_tolerance) || options.gradient_tolerance<=0
       || !std::isfinite(options.initial_damping) || options.initial_damping<=0)
        throw std::invalid_argument("Invalid LM bounds or options");
    LMResult result; result.q=initial;
    if(!initial.allFinite() || (initial.array()<lower.array()).any() || (initial.array()>upper.array()).any()) {
        result.reason="invalid_initial_bounds"; return result;
    }
    Eigen::Index rows=0;
    auto evaluate=[&](const Eigen::VectorXd& q) {
        ++result.evaluations;
        auto e=function(q);
        if(e.status==EvaluationStatus::valid) {
            if(e.residual.size()==0 || (rows && e.residual.size()!=rows) || e.jacobian.rows()!=e.residual.size()
               || e.jacobian.cols()!=n || !e.residual.allFinite() || !e.jacobian.allFinite()
               || !std::isfinite(e.residual.squaredNorm())) {
                e.status=EvaluationStatus::numerical_failure; e.message="Malformed least-squares evaluation";
            } else { rows=e.residual.size(); e.least_squares_cost=.5*e.residual.squaredNorm(); }
        }
        if(e.status==EvaluationStatus::forward_failure) ++result.forward_failures;
        if(e.status==EvaluationStatus::gradient_failure) ++result.gradient_failures;
        return e;
    };
    result.evaluation=evaluate(initial);
    if(result.evaluation.status!=EvaluationStatus::valid) { result.reason="invalid_initial_evaluation"; return result; }
    // Work in coordinates scaled by the parameter support width.
    const Eigen::VectorXd width=upper-lower;
    double damping=options.initial_damping, growth=2;
    for(int iteration=0;iteration<=options.max_iterations;++iteration) {
        const auto& e=result.evaluation;
        Eigen::MatrixXd j=e.jacobian*width.asDiagonal();
        Eigen::VectorXd gradient=j.transpose()*e.residual;
        if(!j.allFinite() || !gradient.allFinite() || !std::isfinite(j.squaredNorm())) {
            result.reason="numerical_scaling_failure"; break;
        }
        const Eigen::VectorXd x=(result.q-lower).cwiseQuotient(width);
        // Unit projected-gradient mapping is zero exactly at a box KKT point.
        const double pg=(x-(x-gradient).cwiseMax(0).cwiseMin(1)).lpNorm<Eigen::Infinity>();
        result.projected_gradient=pg;
        if(pg<=options.gradient_tolerance) { result.converged=true; result.reason="projected_gradient"; break; }
        if(iteration==options.max_iterations) { result.reason="max_iterations"; break; }
        for(Eigen::Index k=0;k<n;++k)
            if((x[k]<=0 && gradient[k]>0) || (x[k]>=1 && gradient[k]<0)) j.col(k).setZero();
        Eigen::VectorXd scale=j.colwise().squaredNorm().transpose().cwiseMax(1e-12);
        Eigen::MatrixXd augmented=Eigen::MatrixXd::Zero(rows+n,n);
        augmented.topRows(rows)=j;
        augmented.bottomRows(n).diagonal()=(damping*scale.array()).sqrt().matrix();
        Eigen::VectorXd rhs=Eigen::VectorXd::Zero(rows+n); rhs.head(rows)=-e.residual;
        Eigen::VectorXd step=augmented.colPivHouseholderQr().solve(rhs);
        Eigen::VectorXd trial=(result.q+width.cwiseProduct(step)).cwiseMax(lower).cwiseMin(upper);
        step=(trial-result.q).cwiseQuotient(width);
        Eigen::VectorXd linear=e.jacobian*width.cwiseProduct(step);
        double predicted=-e.residual.dot(linear)-.5*linear.squaredNorm();
        // Clipping a coupled LM step need not be a descent step. In that case
        // use a projected Cauchy step with a Lipschitz bound for the quadratic.
        if(std::isfinite(predicted) && predicted<=0) {
            ++result.cauchy_steps;
            const double lipschitz=(e.jacobian*width.asDiagonal()).squaredNorm();
            trial=(result.q-width.cwiseProduct(gradient)/lipschitz).cwiseMax(lower).cwiseMin(upper);
            step=(trial-result.q).cwiseQuotient(width);
            linear=e.jacobian*width.cwiseProduct(step);
            predicted=-e.residual.dot(linear)-.5*linear.squaredNorm();
        }
        if(!trial.allFinite() || !std::isfinite(predicted) || predicted<=0 || (trial-result.q).norm()==0) {
            result.reason="stalled_step"; break;
        }
        auto trial_e=evaluate(trial);
        double trial_cost=std::numeric_limits<double>::quiet_NaN(), gain=-std::numeric_limits<double>::infinity();
        if(trial_e.status==EvaluationStatus::valid) {
            trial_cost=trial_e.least_squares_cost;
            gain=(e.least_squares_cost-trial_cost)/predicted;
        }
        const bool accept=std::isfinite(gain) && gain>1e-4;
        result.trace.push_back({iteration,e.least_squares_cost,trial_cost,damping,gain,pg,accept,trial_e.status});
        if(accept) {
            result.q=trial; result.evaluation=std::move(trial_e); ++result.accepted;
            damping*=std::max(1./3.,1-std::pow(2*gain-1,3)); damping=std::max(damping,1e-15); growth=2;
        } else {
            ++result.rejected; damping*=growth; growth*=2;
            if(!std::isfinite(damping) || damping>1e30) { result.reason="damping_limit"; break; }
        }
    }
    return result;
}
LMResult invert_lm(const DispersionProblem& problem,const Eigen::VectorXd& initial,const LMOptions& options) {
    return bounded_lm([&](const Eigen::VectorXd& q){return problem.evaluate(q);},initial,
                      problem.parameters().lower(),problem.parameters().upper(),options);
}
}
