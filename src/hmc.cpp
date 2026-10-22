#include "dispweave/hmc.hpp"
#include "sampler_adaptation.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace dw {
namespace {
bool valid(const DensityEvaluation& e,Eigen::Index n) {
    return e.status==EvaluationStatus::valid && std::isfinite(e.value) && e.gradient.size()==n && e.gradient.allFinite();
}
DensityEvaluation evaluate(const LogDensity& target,const Eigen::VectorXd& q) {
    if(!q.allFinite()) return {};
    auto e=target(q);
    if(e.status==EvaluationStatus::valid && !valid(e,q.size())) e.status=EvaluationStatus::numerical_failure;
    return e;
}
double softplus(double x) {return std::max(x,0.)+std::log1p(std::exp(-std::abs(x)));}
}
LogisticBox::LogisticBox(Eigen::VectorXd lower,Eigen::VectorXd upper)
    :lower_(std::move(lower)),upper_(std::move(upper)) {
    if(lower_.size()==0 || lower_.size()!=upper_.size() || !lower_.allFinite() || !upper_.allFinite()
       || !(upper_-lower_).allFinite() || (upper_.array()<=lower_.array()).any()) throw std::invalid_argument("Invalid logistic bounds");
    width_=upper_-lower_;
}
BoxTransformEvaluation LogisticBox::forward(const Eigen::VectorXd& z) const {
    if(z.size()!=lower_.size() || !z.allFinite()) throw std::invalid_argument("Invalid unconstrained coordinates");
    BoxTransformEvaluation t; t.q.resize(z.size()); t.derivative.resize(z.size());
    t.log_jacobian_gradient.resize(z.size()); t.log_jacobian=0;
    for(Eigen::Index i=0;i<z.size();++i) {
        const double a=std::exp(-std::abs(z[i])), small=a/(1+a);
        // Evaluate distance to the nearest endpoint without subtracting logistic values.
        t.q[i]=z[i]>=0?upper_[i]-width_[i]*small:lower_[i]+width_[i]*small;
        t.interior=t.interior && t.q[i]>lower_[i] && t.q[i]<upper_[i];
        t.derivative[i]=width_[i]*small/(1+a);
        t.log_jacobian_gradient[i]=z[i]>=0?2*small-1:1-2*small;
        t.log_jacobian+=std::log(width_[i])-softplus(z[i])-softplus(-z[i]);
    }
    return t;
}
Eigen::VectorXd LogisticBox::inverse(const Eigen::VectorXd& q) const {
    if(q.size()!=lower_.size() || !q.allFinite() || (q.array()<=lower_.array()).any() || (q.array()>=upper_.array()).any())
        throw std::invalid_argument("HMC initial point must be strictly inside bounds");
    return ((q-lower_).array().log()-(upper_-q).array().log()).matrix();
}
LogDensity transform_density(LogDensity target,LogisticBox box) {
    return [target=std::move(target),box=std::move(box)](const Eigen::VectorXd& z) {
        if(!z.allFinite()) return DensityEvaluation{};
        const auto t=box.forward(z);
        if(!t.interior) return DensityEvaluation{}; // Floating-point saturation, not a clipped proposal.
        auto e=evaluate(target,t.q);
        if(e.status==EvaluationStatus::valid) {
            e.value+=t.log_jacobian;
            e.gradient=e.gradient.cwiseProduct(t.derivative)+t.log_jacobian_gradient;
        }
        return e;
    };
}
LogDensity dispersion_density(const DispersionProblem& problem,Prior prior) {
    return [&problem,prior=std::move(prior)](const Eigen::VectorXd& q) {
        const auto p=log_posterior(problem,q,prior,true);
        return DensityEvaluation{p.status,p.value,p.gradient};
    };
}
EuclideanMetric::EuclideanMetric(Eigen::MatrixXd mass) {
    if(mass.rows()==0 || mass.rows()!=mass.cols() || !mass.allFinite() || !mass.isApprox(mass.transpose(),1e-12))
        throw std::invalid_argument("Mass must be finite symmetric positive definite");
    Eigen::LLT<Eigen::MatrixXd> factor(mass);
    if(factor.info()!=Eigen::Success) throw std::invalid_argument("Mass must be positive definite");
    cholesky_=factor.matrixL();
}
Eigen::VectorXd EuclideanMetric::velocity(const Eigen::VectorXd& p) const {
    if(p.size()!=size()) throw std::invalid_argument("Momentum dimension mismatch");
    const Eigen::VectorXd y=cholesky_.triangularView<Eigen::Lower>().solve(p);
    return cholesky_.transpose().triangularView<Eigen::Upper>().solve(y);
}
Eigen::VectorXd EuclideanMetric::momentum(std::mt19937_64& rng) const {
    std::normal_distribution<double> normal; Eigen::VectorXd z(size());
    for(Eigen::Index i=0;i<size();++i) z[i]=normal(rng);
    return cholesky_*z;
}
double EuclideanMetric::kinetic(const Eigen::VectorXd& p) const {return .5*p.dot(velocity(p));}
IntegrationResult leapfrog(const LogDensity& target,HamiltonianState state,const EuclideanMetric& metric,double step,int steps) {
    if(steps<1 || !std::isfinite(step) || step==0 || state.position.size()!=metric.size()
       || state.momentum.size()!=metric.size() || !state.position.allFinite() || !state.momentum.allFinite()
       || !valid(state.density,metric.size())) throw std::invalid_argument("Invalid leapfrog input");
    IntegrationResult r; r.state=std::move(state);
    auto& s=r.state;
    for(int k=0;k<steps;++k) {
        s.momentum+=.5*step*s.density.gradient;
        s.position+=step*metric.velocity(s.momentum);
        s.density=evaluate(target,s.position); ++r.evaluations;
        if(s.density.status!=EvaluationStatus::valid) {r.status=s.density.status; return r;}
        s.momentum+=.5*step*s.density.gradient;
        if(!s.momentum.allFinite()) {r.status=EvaluationStatus::numerical_failure; return r;}
    }
    return r;
}
HMCResult sample_hmc(const LogDensity& target,const Eigen::VectorXd& initial,const EuclideanMetric& initial_metric,const HMCOptions& o) {
    if(o.warmup<0 || o.draws<1 || o.warmup>std::numeric_limits<int>::max()-o.draws || o.leapfrog_steps<1
       || !std::isfinite(o.step_size) || o.step_size<=0 || !std::isfinite(o.target_accept) || o.target_accept<=0 || o.target_accept>=1
       || !std::isfinite(o.step_jitter) || o.step_jitter<0 || o.step_jitter>=1
       || !std::isfinite(o.divergence_threshold) || o.divergence_threshold<=0 || initial.size()!=initial_metric.size())
        throw std::invalid_argument("Invalid HMC options or dimensions");
    EuclideanMetric metric=initial_metric;
    const bool adapt_metric=o.metric_adaptation!=MetricAdaptation::none;
    if(adapt_metric && !o.adapt_step_size) throw std::invalid_argument("Metric adaptation requires step adaptation");
    const auto windows=adapt_metric?detail::warmup_windows(o.warmup):std::vector<detail::WarmupWindow>{};
    size_t window=0;
    detail::CovarianceAccumulator covariance(initial.size());
    auto current=evaluate(target,initial);
    if(!valid(current,initial.size())) throw std::invalid_argument("Invalid initial HMC density or gradient");
    // Explicit per-chain seed material; no shared/global engine or scheduling dependence.
    std::seed_seq seeds{std::uint32_t(o.seed),std::uint32_t(o.seed>>32),std::uint32_t(o.chain_id),std::uint32_t(o.chain_id>>32)};
    std::mt19937_64 rng(seeds); std::uniform_real_distribution<double> uniform(0,1);
    Eigen::VectorXd q=initial; HMCResult out; out.draws.resize(o.draws,q.size());
    double step=o.step_size;
    detail::DualAveraging adaptation(step,o.target_accept);
    for(int i=0;i<o.warmup+o.draws;++i) {
        const auto p=metric.momentum(rng); const double initial_energy=-current.value+metric.kinetic(p);
        const double used_step=step*(1+o.step_jitter*(2*uniform(rng)-1));
        auto proposal=leapfrog(target,{q,p,current},metric,used_step,o.leapfrog_steps);
        double error=INFINITY, probability=0;
        if(proposal.status==EvaluationStatus::valid) {
            error=current.value-proposal.state.density.value+metric.kinetic(proposal.state.momentum)-metric.kinetic(p);
            if(!std::isfinite(error) || !std::isfinite(initial_energy)) proposal.status=EvaluationStatus::numerical_failure;
            else probability=std::exp(std::min(0.,-error));
        }
        const bool accepted=uniform(rng)<probability;
        const bool divergent=proposal.status!=EvaluationStatus::valid || std::abs(error)>o.divergence_threshold;
        out.transitions.push_back({i<o.warmup,accepted,divergent,used_step,probability,error,proposal.status,proposal.evaluations});
        if(accepted) {q=std::move(proposal.state.position); current=std::move(proposal.state.density);}
        if(i<o.warmup && o.adapt_step_size) {
            step=adaptation.update(probability,i+1==o.warmup);
            if(window<windows.size() && i>=windows[window].begin) {
                covariance.add(q);
                if(i+1==windows[window].end) {
                    const auto mass=covariance.mass(o.metric_adaptation);
                    metric=EuclideanMetric(mass);
                    out.metric_updates.push_back({i+1,covariance.count(),mass});
                    covariance=detail::CovarianceAccumulator(initial.size());
                    ++window;
                    // Restart dual averaging after each metric update; final fast
                    // buffer tunes the new metric before sampling freezes it.
                    adaptation=detail::DualAveraging(step,o.target_accept);
                }
            }
        }
        if(i>=o.warmup) out.draws.row(i-o.warmup)=q.transpose();
    }
    out.step_size=step;out.mass=metric.mass();return out;
}
}
