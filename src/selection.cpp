#include "dispweave/selection.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
InformationCriteria information_criteria(double log_likelihood,int n,int k) {
    if(!std::isfinite(log_likelihood) || n<=0 || k<=0)
        throw std::invalid_argument("Require finite log likelihood and positive n,k");
    InformationCriteria out{-2*log_likelihood+2*k,-2*log_likelihood+k*std::log(static_cast<double>(n))};
    if(!std::isfinite(out.aic) || !std::isfinite(out.bic)) throw std::invalid_argument("Information criterion overflow");
    return out;
}
CandidateAssessment assess_candidate(const LMResult& fit,const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,const Eigen::VectorXd& scale,double ridge) {
    CandidateAssessment out;
    if(!fit.converged || fit.evaluation.status!=EvaluationStatus::valid) { out.reason="fit_not_converged"; return out; }
    const auto& q=fit.q; const auto& e=fit.evaluation;
    if(q.size()==0 || lower.size()!=q.size() || upper.size()!=q.size() || !q.allFinite()
       || !lower.allFinite() || !upper.allFinite() || (lower.array()>=upper.array()).any()
       || (q.array()<lower.array()).any() || (q.array()>upper.array()).any()
       || e.jacobian.cols()!=q.size() || e.jacobian.rows()!=e.residual.size())
        throw std::invalid_argument("Malformed fitted candidate");
    out.local=local_resolution(e.jacobian,scale,ridge);
    const auto ic=information_criteria(e.log_likelihood,static_cast<int>(e.residual.size()),static_cast<int>(q.size()));
    out.fitted=true; out.log_likelihood=e.log_likelihood;
    // 1e-8 of the allowed log-parameter width is considered near the boundary.
    const Eigen::VectorXd position=(q-lower).cwiseQuotient(upper-lower);
    if((position.array()<=1e-8).any() || (position.array()>=1-1e-8).any()) out.reason="boundary_solution";
    else if(out.local.numerical_rank!=q.size()) out.reason="rank_deficient";
    else if(e.residual.size()<=q.size()) out.reason="insufficient_observations";
    else { out.regular=true; out.reason="local_screen_passed"; out.aic=ic.aic; out.bic=ic.bic; }
    return out;
}
}
