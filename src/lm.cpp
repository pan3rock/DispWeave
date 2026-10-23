#include "dispweave/lm.hpp"
#include <Eigen/QR>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace dw {
namespace {
struct BoxStep { Eigen::VectorXd step; int iterations=0; bool solved=false; };
// Primal feasible active-set least squares. The damping rows make the
// quadratic strictly convex. Add blocking bounds; release incorrect multipliers.
BoxStep box_step(const Eigen::MatrixXd& a,const Eigen::VectorXd& b,
                 const Eigen::VectorXd& lower,const Eigen::VectorXd& upper) {
    const Eigen::Index n=a.cols(); BoxStep out; out.step=Eigen::VectorXd::Zero(n);
    std::vector<int> active(n,0); // -1 lower, +1 upper, 0 free
    const Eigen::VectorXd linear=a.transpose()*b;
    if(!linear.allFinite()) return out;
    for(Eigen::Index iteration=0;iteration<100*(n+1);++iteration) {
        ++out.iterations;
        std::vector<Eigen::Index> free;
        Eigen::VectorXd target=out.step;
        for(Eigen::Index k=0;k<n;++k) if(active[k]==0) {free.push_back(k); target[k]=0;}
        if(!free.empty()) {
            Eigen::MatrixXd reduced(a.rows(),free.size());
            for(size_t k=0;k<free.size();++k) reduced.col(k)=a.col(free[k]);
            // Rows independent of every free variable contribute only a constant.
            // Exclude exact zeros before QR so a huge constant residual cannot
            // contaminate the transformed right-hand side through roundoff.
            const Eigen::VectorXd rhs=b-a*target;
            Eigen::MatrixXd compact(a.rows(),free.size()); Eigen::VectorXd values(a.rows());
            Eigen::Index count=0;
            for(Eigen::Index row=0;row<a.rows();++row)
                if((reduced.row(row).array()!=0).any()) {compact.row(count)=reduced.row(row); values[count]=rhs[row]; ++count;}
            const Eigen::VectorXd solution=compact.topRows(count).colPivHouseholderQr().solve(values.head(count));
            if(!solution.allFinite()) return out;
            for(size_t k=0;k<free.size();++k) target[free[k]]=solution[k];
        }
        double fraction=1; Eigen::Index blocker=-1; int side=0;
        for(auto k:free) {
            const double direction=target[k]-out.step[k];
            double ratio=1; int bound=0;
            if(target[k]<lower[k]) {ratio=(lower[k]-out.step[k])/direction; bound=-1;}
            else if(target[k]>upper[k]) {ratio=(upper[k]-out.step[k])/direction; bound=1;}
            if(bound && ratio<fraction) { fraction=std::max(0.,ratio); blocker=k; side=bound; }
        }
        out.step=(out.step+fraction*(target-out.step)).cwiseMax(lower).cwiseMin(upper);
        if(blocker>=0) {
            out.step[blocker]=side<0?lower[blocker]:upper[blocker]; active[blocker]=side; continue;
        }
        const Eigen::VectorXd product=a.transpose()*(a*out.step);
        const Eigen::VectorXd gradient=a.transpose()*(a*out.step-b);
        if(!gradient.allFinite() || !product.allFinite()) return out;
        const double tolerance=64*std::numeric_limits<double>::epsilon()*
            std::max({1.,linear.lpNorm<Eigen::Infinity>(),product.lpNorm<Eigen::Infinity>()});
        double violation=tolerance; Eigen::Index release=-1;
        for(Eigen::Index k=0;k<n;++k) {
            const double wrong=active[k]<0?-gradient[k]:(active[k]>0?gradient[k]:0.);
            if(wrong>violation) { violation=wrong; release=k; }
        }
        if(release<0) {out.solved=true; return out;}
        active[release]=0;
    }
    return out;
}
double decrease(const Eigen::VectorXd& before,const Eigen::VectorXd& after) {
    // Algebraically .5*(||before||^2-||after||^2), without subtracting two costs.
    long double value=0;
    for(Eigen::Index i=0;i<before.size();++i) {
        const long double a=before[i],b=after[i]; value+=(a-b)*(a+b)/2;
    }
    return static_cast<double>(value);
}
}
const char* to_string(LMTermination status) {
    switch(status) {
        case LMTermination::converged:return "converged";
        case LMTermination::stalled:return "stalled";
        default:return "failed";
    }
}
LMResult bounded_lm(const LeastSquaresFunction& function,const Eigen::VectorXd& initial,
                    const Eigen::VectorXd& lower,const Eigen::VectorXd& upper,const LMOptions& options) {
    const auto n=initial.size();
    if(n==0 || lower.size()!=n || upper.size()!=n || !lower.allFinite() || !upper.allFinite()
       || !(upper-lower).allFinite() || (lower.array()>=upper.array()).any() || options.max_iterations<1
       || !std::isfinite(options.gradient_tolerance) || options.gradient_tolerance<=0
       || !std::isfinite(options.step_tolerance) || options.step_tolerance<=0 || options.step_tolerance>=1
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
            } else {
                rows=e.residual.size(); e.least_squares_cost=.5*e.residual.squaredNorm();
                if(std::isnan(e.data_cost)) e.data_cost=e.least_squares_cost;
            }
        }
        if(e.status==EvaluationStatus::forward_failure) ++result.forward_failures;
        if(e.status==EvaluationStatus::gradient_failure) ++result.gradient_failures;
        return e;
    };
    result.evaluation=evaluate(initial);
    if(result.evaluation.status!=EvaluationStatus::valid) { result.reason="invalid_initial_evaluation"; return result; }
    const Eigen::VectorXd width=upper-lower;
    double damping=options.initial_damping, growth=2;
    bool rejected_valid=false, rejected_invalid=false;
    for(int iteration=0;iteration<=options.max_iterations;++iteration) {
        const auto& e=result.evaluation;
        const Eigen::MatrixXd j=e.jacobian*width.asDiagonal();
        const Eigen::VectorXd gradient=j.transpose()*e.residual;
        if(!j.allFinite() || !gradient.allFinite() || !std::isfinite(j.squaredNorm())) {
            result.reason="numerical_scaling_failure"; break;
        }
        const Eigen::VectorXd x=(result.q-lower).cwiseQuotient(width);
        const double pg=(x-(x-gradient).cwiseMax(0).cwiseMin(1)).lpNorm<Eigen::Infinity>();
        result.projected_gradient=pg;
        if(pg<=options.gradient_tolerance) {
            result.converged=true; result.termination=LMTermination::converged; result.reason="projected_gradient"; break;
        }
        if(iteration==options.max_iterations) { result.reason="max_iterations"; break; }
        const Eigen::VectorXd scale=j.colwise().squaredNorm().transpose().cwiseMax(1e-12);
        Eigen::MatrixXd augmented=Eigen::MatrixXd::Zero(rows+n,n);
        augmented.topRows(rows)=j;
        augmented.bottomRows(n).diagonal()=(damping*scale.array()).sqrt().matrix();
        if(!augmented.allFinite()) { result.reason="numerical_scaling_failure"; break; }
        Eigen::VectorXd rhs=Eigen::VectorXd::Zero(rows+n); rhs.head(rows)=-e.residual;
        const Eigen::VectorXd step_lower=(lower-result.q).cwiseQuotient(width);
        const Eigen::VectorXd step_upper=(upper-result.q).cwiseQuotient(width);
        const auto subproblem=box_step(augmented,rhs,step_lower,step_upper);
        result.box_iterations+=subproblem.iterations;
        if(!subproblem.solved) { result.reason="box_subproblem_failure"; break; }
        Eigen::VectorXd trial=(result.q+width.cwiseProduct(subproblem.step)).cwiseMax(lower).cwiseMin(upper);
        for(Eigen::Index k=0;k<n;++k) {
            if(subproblem.step[k]==step_lower[k]) trial[k]=lower[k];
            if(subproblem.step[k]==step_upper[k]) trial[k]=upper[k];
        }
        const Eigen::VectorXd step=(trial-result.q).cwiseQuotient(width);
        const double step_norm=step.lpNorm<Eigen::Infinity>();
        const Eigen::VectorXd linear=j*step;
        const double predicted=-e.residual.dot(linear)-.5*linear.squaredNorm();
        if(!trial.allFinite() || !std::isfinite(predicted)) { result.reason="numerical_step_failure"; break; }
        // A tiny step after rejected evaluations is stagnation, never success.
        // Failed physics takes priority over a numerical-stagnation classification.
        if(step_norm==0 || ((rejected_valid || rejected_invalid) && step_norm<=options.step_tolerance) || predicted<=0) {
            result.termination=rejected_invalid?LMTermination::failed:LMTermination::stalled;
            result.reason=rejected_invalid?"invalid_trial_stagnation":"step_stagnation"; break;
        }
        auto trial_e=evaluate(trial);
        double trial_cost=NAN, actual=NAN, gain=-INFINITY;
        if(trial_e.status==EvaluationStatus::valid) {
            trial_cost=trial_e.least_squares_cost; actual=decrease(e.residual,trial_e.residual); gain=actual/predicted;
        }
        const bool accept=std::isfinite(gain) && gain>1e-4;
        result.trace.push_back({iteration,e.least_squares_cost,trial_cost,damping,gain,pg,accept,trial_e.status,
                               predicted,actual,step_norm,subproblem.iterations,
                               e.data_cost,e.regularization_cost,trial_e.data_cost,trial_e.regularization_cost});
        if(accept) {
            result.q=trial; result.evaluation=std::move(trial_e); ++result.accepted;
            damping*=std::max(1./3.,1-std::pow(2*gain-1,3)); damping=std::max(damping,1e-15); growth=2;
            rejected_valid=false; rejected_invalid=false;
        } else {
            ++result.rejected;
            if(trial_e.status==EvaluationStatus::valid) rejected_valid=true; else rejected_invalid=true;
            damping*=growth; growth*=2;
            if(!std::isfinite(damping) || damping>1e30) {
                result.termination=rejected_invalid?LMTermination::failed:LMTermination::stalled;
                result.reason="damping_limit"; break;
            }
        }
    }
    return result;
}
LMResult invert_lm(const DispersionProblem& problem,const Eigen::VectorXd& initial,const LMOptions& options,
                  const std::vector<int>& fixed_vs_layers,const VsRegularization* regularization) {
    const auto& p=problem.parameters();
    const auto n=p.lower().size();
    std::vector<bool> fixed(n,false);
    for(int layer:fixed_vs_layers) {
        if(layer<0 || layer>=n || fixed[layer]) throw std::invalid_argument("Invalid/duplicate fixed Vs layer");
        fixed[layer]=true;
    }
    if(static_cast<Eigen::Index>(fixed_vs_layers.size())==n)
        throw std::invalid_argument("LM requires at least one free Vs layer");
    auto full_evaluate=[&](const Eigen::VectorXd& q) {
        auto e=problem.evaluate(q);
        if(e.status==EvaluationStatus::valid && regularization && regularization->active()) {
            const auto penalty=regularization->evaluate(q);
            const auto rows=e.residual.size(),extra=penalty.residual.size();
            e.residual.conservativeResize(rows+extra);e.residual.tail(extra)=penalty.residual;
            e.jacobian.conservativeResize(rows+extra,n);e.jacobian.bottomRows(extra)=penalty.jacobian;
            e.regularization_cost=.5*penalty.residual.squaredNorm();
            e.least_squares_cost=e.data_cost+e.regularization_cost;
        }
        return e;
    };
    // Preserve the original path and its invalid-initial diagnostics.
    if(fixed_vs_layers.empty() || !p.contains(initial))
        return bounded_lm(full_evaluate,initial,p.lower(),p.upper(),options);
    std::vector<int> free;
    for(int i=0;i<n;++i) if(!fixed[i]) free.push_back(i);
    auto select=[&](const Eigen::VectorXd& q) {
        Eigen::VectorXd x(free.size());
        for(size_t i=0;i<free.size();++i) x[i]=q[free[i]];
        return x;
    };
    auto expand=[&](const Eigen::VectorXd& x) {
        Eigen::VectorXd q=initial;
        for(size_t i=0;i<free.size();++i) q[free[i]]=x[i];
        return q;
    };
    auto evaluate=[&](const Eigen::VectorXd& x) {
        auto e=full_evaluate(expand(x));
        if(e.status==EvaluationStatus::valid) {
            Eigen::MatrixXd j(e.jacobian.rows(),free.size());
            for(size_t i=0;i<free.size();++i) j.col(i)=e.jacobian.col(free[i]);
            e.jacobian=std::move(j);
        }
        return e;
    };
    auto result=bounded_lm(evaluate,select(initial),select(p.lower()),select(p.upper()),options);
    result.q=expand(result.q);
    // Restore the full Jacobian from the retained kernels, without another
    // forward evaluation or altering evaluation/iteration counts.
    if(result.evaluation.status==EvaluationStatus::valid) {
        result.evaluation.jacobian=problem.likelihood().whiten_jacobian(
            p.phase_jacobian(result.q,result.evaluation.forward));
        if(regularization && regularization->active()) {
            const auto penalty=regularization->evaluate(result.q);
            auto& j=result.evaluation.jacobian;const auto rows=j.rows();
            j.conservativeResize(rows+penalty.jacobian.rows(),n);j.bottomRows(penalty.jacobian.rows())=penalty.jacobian;
        }
    }
    return result;
}
}
