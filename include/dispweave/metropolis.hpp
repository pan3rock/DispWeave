#pragma once
#include "objective.hpp"
#include <cstdint>
#include <optional>
namespace dw {
struct ValueEvaluation {EvaluationStatus status=EvaluationStatus::numerical_failure;double value=NAN;};
using ValueDensity=std::function<ValueEvaluation(const Eigen::VectorXd&)>;
struct GaussianProposalComponent {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
    double weight=1;
};
// Fixed BEFORE the run, e.g. from separate pilot chains. Component weights
// describe proposals, never posterior mode probabilities. Coordinates match x.
struct IndependenceProposal {
    double probability=.25; // State-independent choice between two MH kernels.
    std::vector<GaussianProposalComponent> components;
};
struct MetropolisOptions {
    int warmup=1000,draws=10000;
    double scale=1,target_accept=.234;
    bool adapt_scale=true,adapt_covariance=true;
    std::uint64_t seed=2026,chain_id=0;
    std::optional<IndependenceProposal> independence;
};
struct MetropolisTransition {
    bool warmup=false,accepted=false;
    EvaluationStatus status=EvaluationStatus::valid;
    double scale=1,acceptance_probability=0;
    bool independent=false;
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
// Symmetric Gaussian random walk, optionally mixed with a frozen Gaussian
// mixture independence MH step using the FULL proposal density ratio.
// All adaptation is restricted to warmup;
// production proposals are fixed, with no clipping/reflection at support bounds.
MetropolisResult sample_metropolis(const ValueDensity&,const Eigen::VectorXd& initial,
                                  const Eigen::MatrixXd& proposal_covariance,const MetropolisOptions& = {});
}
