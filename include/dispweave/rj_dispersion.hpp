#pragma once
#include "rjmcmc.hpp"
namespace dw {
struct RJDispersionEvaluation {
    RJEvaluation likelihood;
    ForwardResult forward;
    Eigen::VectorXd prediction; // Observation order; no missing-row deletion.
};
class RJDispersionProblem {
public:
    // The forward backend must outlive this object. Vp=ratio*Vs; rho constant.
    RJDispersionProblem(const Forward&,RJPrior,DispersionObservations,double vp_vs_ratio,double density);
    LayeredModel model(const RJState&) const;
    RJDispersionEvaluation evaluate(const RJState&) const;
    const RJPrior& prior() const {return prior_;}
    const GaussianLikelihood& likelihood() const {return likelihood_;}
private:
    const Forward& forward_;
    RJPrior prior_;
    GaussianLikelihood likelihood_;
    double ratio_,density_;
};
}
