#pragma once
#include "lm.hpp"
#include "resolution.hpp"
namespace dw {
struct InformationCriteria { double aic, bic; };
// Known covariance; k counts fitted continuous parameters, including any noise parameters.
InformationCriteria information_criteria(double log_likelihood,int observations,int parameters);
struct CandidateAssessment {
    bool fitted=false, regular=false;
    std::string reason;
    double log_likelihood=NAN, aic=NAN, bic=NAN;
    LocalResolution local;
};
// Regularity screen is necessary, not sufficient for asymptotic IC validity.
// No scoring or selection based on a nonconverged final iterate.
CandidateAssessment assess_candidate(const LMResult&,const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,const Eigen::VectorXd& scale,double ridge);
}
