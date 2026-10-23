#pragma once
#include "objective.hpp"
#include <cstdint>
namespace dw {
struct ValueEvaluation {EvaluationStatus status=EvaluationStatus::numerical_failure;double value=NAN;};
using ValueDensity=std::function<ValueEvaluation(const Eigen::VectorXd&)>;
struct MetropolisOptions {
    int warmup=1000,draws=10000;
    double scale=1,target_accept=.234;
    bool adapt_scale=true,adapt_covariance=true;
    std::uint64_t seed=2026,chain_id=0;
};
struct MetropolisTransition {
    bool warmup=false,accepted=false;
    EvaluationStatus status=EvaluationStatus::valid;
    double scale=1,acceptance_probability=0;
};
struct ProposalUpdate {int warmup_iteration,samples;Eigen::MatrixXd covariance;};
struct MetropolisResult {
    Eigen::MatrixXd draws; // In supplied coordinates, retains all rejected states.
    Eigen::VectorXd log_density;
    std::vector<MetropolisTransition> transitions; // Includes marked warmup.
    double scale=1;
    Eigen::MatrixXd covariance; // Proposal covariance is scale^2 * covariance.
    std::vector<ProposalUpdate> covariance_updates;
};
// Symmetric Gaussian random walk. All adaptation is restricted to warmup;
// production proposals are fixed, with no clipping/reflection at support bounds.
MetropolisResult sample_metropolis(const ValueDensity&,const Eigen::VectorXd& initial,
                                  const Eigen::MatrixXd& proposal_covariance,const MetropolisOptions& = {});
}
