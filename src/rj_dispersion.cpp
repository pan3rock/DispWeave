#include "dispweave/rj_dispersion.hpp"
#include <stdexcept>
namespace dw {
RJDispersionProblem::RJDispersionProblem(const Forward& f,RJPrior p,DispersionObservations data,double ratio,double density)
    :forward_(f),prior_(std::move(p)),likelihood_(std::move(data)),ratio_(ratio),density_(density) {
    prior_.validate();
    if(!std::isfinite(ratio_) || ratio_<=std::sqrt(4./3.) || !std::isfinite(density_) || density_<=0
       || !std::isfinite(ratio_*prior_.vs_upper)) throw std::invalid_argument("Invalid RJ Vp/Vs ratio or density");
}
LayeredModel RJDispersionProblem::model(const RJState& s) const {
    if(!prior_.supports(s)) throw std::invalid_argument("RJ model outside prior support");
    LayeredModel m;m.depth.resize(s.vs.size());m.depth[0]=0;m.depth.tail(s.interfaces.size())=s.interfaces;
    m.vs=s.vs;m.vp=ratio_*s.vs;m.rho=Eigen::VectorXd::Constant(s.vs.size(),density_);m.validate();return m;
}
RJDispersionEvaluation RJDispersionProblem::evaluate(const RJState& s) const {
    RJDispersionEvaluation e;
    if(!prior_.supports(s)) {e.likelihood={EvaluationStatus::outside_support,-INFINITY};return e;}
    try {
        e.forward=forward_.evaluate(model(s),likelihood_.data().queries,SensitivityMethod::none);
        if(e.forward.rows.size()!=likelihood_.data().queries.size() || !e.forward.all_valid()) {
            e.likelihood.status=EvaluationStatus::forward_failure;return e;
        }
        e.prediction.resize(e.forward.rows.size());
        for(Eigen::Index i=0;i<e.prediction.size();++i) e.prediction[i]=e.forward.rows[i].phase_velocity;
        if(!e.prediction.allFinite() || (e.prediction.array()<=0).any()) return e;
        e.likelihood={EvaluationStatus::valid,likelihood_.log_likelihood(e.prediction)};
    } catch(const std::runtime_error&) {e.likelihood.status=EvaluationStatus::numerical_failure;}
    return e;
}
}
