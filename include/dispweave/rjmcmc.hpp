#pragma once
#include "rj_proposal.hpp"
#include "objective.hpp"
#include <cstdint>
namespace dw {
struct RJEvaluation { EvaluationStatus status=EvaluationStatus::numerical_failure; double value=NAN; };
// Data likelihood ONLY. sample_rjmcmc adds the normalized RJPrior exactly once.
using RJLikelihood=std::function<RJEvaluation(const RJState&)>;
struct RJOptions {
    int warmup=1000,draws=10000;
    double vs_step=.1,interface_step=.2; // Physical-unit Gaussian standard deviations.
    std::uint64_t seed=2026,chain_id=0;
};
struct RJTransition {
    bool warmup=false,accepted=false,evaluated=false;
    RJMove move=RJMove::birth;
    RJProposalStatus proposal_status=RJProposalStatus::unavailable;
    EvaluationStatus evaluation_status=EvaluationStatus::valid; // Meaningful if evaluated.
    int layers_before=0,layers_after=0;
    double log_accept_ratio=-INFINITY;
};
struct RJDraw { RJState state; double log_likelihood,log_prior; };
struct RJResult {
    std::vector<RJDraw> draws; // All retained states, including repeats; no thinning.
    std::vector<RJTransition> transitions; // Includes marked warmup.
};
// Four actions have fixed probability 1/4. Unavailable actions are self-transitions.
// No HMC coupling, adaptive proposals, gradients, clipping or truncated random walks.
RJResult sample_rjmcmc(const RJLikelihood&,const RJPrior&,const RJState&,const RJOptions& = {});
}
