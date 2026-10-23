#include "dispweave/objective.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
namespace {
void check_vectors(const Eigen::VectorXd& a,const Eigen::VectorXd& b) {
    if(a.size()==0 || a.size()!=b.size() || !a.allFinite() || !b.allFinite())
        throw std::invalid_argument("Prior vectors must be finite, nonempty and equal-sized");
}
}
void DispersionObservations::validate() const {
    const auto n=static_cast<Eigen::Index>(queries.size());
    GaussianData{phase_velocity,covariance}.validate();
    if(phase_velocity.size()!=n || (phase_velocity.array()<=0).any()) throw std::invalid_argument("Malformed dispersion observations");
    for(const auto& q:queries)
        if(!std::isfinite(q.frequency_hz) || q.frequency_hz<=0 || q.mode<0 || q.mode>1023)
            throw std::invalid_argument("Invalid observation query");
}
GaussianLikelihood::GaussianLikelihood(DispersionObservations data)
    :GaussianDataLikelihood({data.phase_velocity,data.covariance}),data_(std::move(data)) {
    data_.validate();
    data_.covariance=observations().covariance;
}
DispersionProblem::DispersionProblem(const Forward& f,FixedVs p,DispersionObservations o,SensitivityMethod method)
    :forward_(f),parameters_(std::move(p)),likelihood_(std::move(o)),derivative_(method) {
    if(method!=SensitivityMethod::energy && method!=SensitivityMethod::implicit)
        throw std::invalid_argument("Problem requires energy or implicit derivative policy");
}
DataEvaluation DispersionProblem::evaluate(const Eigen::VectorXd& q,bool gradient,double scale) const {
    DataEvaluation e;
    if(q.size()!=parameters_.lower().size() || !q.allFinite()) {e.message="Malformed q"; return e;}
    if(!parameters_.contains(q)) {e.status=EvaluationStatus::outside_support; e.log_likelihood=-INFINITY; return e;}
    validate_log_scale(scale);
    e.forward=forward_.evaluate(parameters_.decode(q),likelihood_.data().queries,gradient?derivative_:SensitivityMethod::none);
    if(e.forward.rows.size()!=likelihood_.data().queries.size() || !e.forward.all_valid()) {
        e.status=EvaluationStatus::forward_failure; e.message="Unresolved forward rows; no observations discarded"; return e;
    }
    Eigen::VectorXd prediction(e.forward.rows.size());
    for(size_t i=0;i<e.forward.rows.size();++i) prediction[i]=e.forward.rows[i].phase_velocity;
    try {
        e.residual=likelihood_.whiten(prediction-likelihood_.data().phase_velocity,scale);
        e.log_likelihood=likelihood_.log_likelihood(prediction,scale);
        e.least_squares_cost=.5*e.residual.squaredNorm();
        e.data_cost=e.least_squares_cost;
        if(gradient) {
            if(!e.forward.all_sensitivities_valid()) {e.status=EvaluationStatus::gradient_failure; e.message="Valid roots but unavailable sensitivity"; return e;}
            e.jacobian=likelihood_.whiten_jacobian(parameters_.phase_jacobian(q,e.forward),scale);
        }
        e.status=EvaluationStatus::valid;
    } catch(const std::runtime_error& error) {e.message=error.what();}
    return e;
}
Prior gaussian_prior(Eigen::VectorXd mean,Eigen::VectorXd sigma,PriorMeasure measure) {
    check_vectors(mean,sigma);
    if((sigma.array()<=0).any()) throw std::invalid_argument("Prior sigma must be positive");
    return {measure,[mean,sigma](const Eigen::VectorXd& x,bool gradient) {
        if(x.size()!=mean.size() || !x.allFinite()) throw std::invalid_argument("Malformed prior argument");
        const Eigen::VectorXd z=(x-mean).array()/sigma.array();
        PriorEvaluation p{-.5*z.squaredNorm()-sigma.array().log().sum()-.5*x.size()*std::log(2*std::acos(-1.)),{}};
        if(!std::isfinite(p.value)) p.value=NAN; // Overflow is not prior support rejection.
        if(gradient) p.gradient=-z.array()/sigma.array();
        return p;
    }};
}
Prior uniform_prior(Eigen::VectorXd lo,Eigen::VectorXd hi,PriorMeasure measure) {
    check_vectors(lo,hi);
    if((hi.array()<=lo.array()).any() || !(hi-lo).allFinite()) throw std::invalid_argument("Invalid prior bounds");
    return {measure,[lo,hi](const Eigen::VectorXd& x,bool gradient) {
        if(x.size()!=lo.size() || !x.allFinite()) throw std::invalid_argument("Malformed prior argument");
        if((x.array()<lo.array()).any() || (x.array()>hi.array()).any()) return PriorEvaluation{-INFINITY,{}};
        PriorEvaluation p{-(hi-lo).array().log().sum(),{}};
        if(gradient) p.gradient=Eigen::VectorXd::Zero(x.size());
        return p;
    }};
}
PosteriorEvaluation log_posterior(const DispersionProblem& problem,const Eigen::VectorXd& q,
                                 const Prior& prior,bool gradient,double scale) {
    PosteriorEvaluation out;
    if(q.size()!=problem.parameters().lower().size() || !q.allFinite()) {
        out.message="Malformed q";
        return out;
    }
    if(!problem.parameters().contains(q)) {
        out.status=EvaluationStatus::outside_support;
        out.value=-INFINITY;
        return out;
    }
    if(!prior.evaluate || (prior.measure!=PriorMeasure::physical_vs && prior.measure!=PriorMeasure::log_vs))
        throw std::invalid_argument("Prior must declare its measure and evaluator");
    const Eigen::VectorXd vs=q.array().exp();
    const auto p=prior.evaluate(prior.measure==PriorMeasure::physical_vs?vs:q,gradient);
    out.log_prior=p.value;
    if(p.value==-INFINITY) {
        out.status=EvaluationStatus::outside_support;
        out.value=-INFINITY;
        return out;
    }
    if(!std::isfinite(p.value)) {
        out.message="Nonfinite prior";
        return out;
    }
    if(gradient && (p.gradient.size()!=q.size() || !p.gradient.allFinite())) {
        out.status=EvaluationStatus::gradient_failure;
        out.message="Malformed prior gradient";
        return out;
    }
    out.data=problem.evaluate(q,gradient,scale);
    out.status=out.data.status;
    out.log_likelihood=out.data.log_likelihood;
    out.log_transform=prior.measure==PriorMeasure::physical_vs?problem.parameters().log_abs_det_jacobian(q):0;
    if(out.status!=EvaluationStatus::valid) {
        out.message=out.data.message;
        return out;
    }
    out.value=out.log_likelihood+out.log_prior+out.log_transform;
    if(gradient) {
        out.gradient=-out.data.jacobian.transpose()*out.data.residual;
        if(prior.measure==PriorMeasure::physical_vs)
            out.gradient+=(p.gradient.array()*vs.array()).matrix()+problem.parameters().log_abs_det_gradient(q);
        else out.gradient+=p.gradient;
        out.log_scale_gradient=out.data.residual.squaredNorm()-out.data.residual.size();
    }
    if(!std::isfinite(out.value) || (gradient && (!out.gradient.allFinite() || !std::isfinite(out.log_scale_gradient)))) {
        out.status=EvaluationStatus::numerical_failure;
        out.value=NAN;
        out.message="Posterior overflow";
    }
    return out;
}
}
