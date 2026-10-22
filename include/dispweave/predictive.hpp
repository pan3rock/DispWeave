#pragma once
#include "objective.hpp"
#include "rj_model.hpp"
#include <cstdint>
namespace dw {
struct PredictiveResult {
    Eigen::MatrixXd latent,replicated; // rows = supplied draws; columns = original observation order
    Eigen::VectorXd observed_discrepancy,replicated_discrepancy;
    std::vector<EvaluationStatus> status; // failures retain their row as NaN; never silently removed
};
// Fixed observed covariance. Discrepancy = squared whitened residual, no fitted-noise approximation.
PredictiveResult posterior_predictive(const DispersionProblem&,const Eigen::MatrixXd& q_draws,std::uint64_t seed);
class RJDispersionProblem;
PredictiveResult posterior_predictive(const RJDispersionProblem&,const std::vector<RJState>&,std::uint64_t seed);
}
