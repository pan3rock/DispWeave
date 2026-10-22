#include "dispweave/rjmcmc.hpp"
#include <limits>
#include <random>
#include <stdexcept>
namespace dw {
namespace {
RJEvaluation evaluate(const RJLikelihood& likelihood,const RJState& state) {
    auto e=likelihood(state);
    if(e.status==EvaluationStatus::valid && !std::isfinite(e.value)) e.status=EvaluationStatus::numerical_failure;
    return e;
}
}
RJResult sample_rjmcmc(const RJLikelihood& likelihood,const RJPrior& prior,const RJState& initial,const RJOptions& o) {
    prior.validate();
    if(!prior.supports(initial) || o.warmup<0 || o.draws<1 || o.warmup>std::numeric_limits<int>::max()-o.draws
       || !std::isfinite(o.vs_step) || o.vs_step<=0 || !std::isfinite(o.interface_step) || o.interface_step<=0)
        throw std::invalid_argument("Invalid RJ initial state or sampler options");
    // Birth/death changes K by one. Do not silently explore only one disconnected class.
    if((prior.layer_weights.array()<=0).any()) throw std::invalid_argument("RJ sampler needs positive layer weights throughout the declared range");
    auto current=evaluate(likelihood,initial);
    if(current.status!=EvaluationStatus::valid) throw std::invalid_argument("Invalid initial RJ likelihood");
    RJState state=initial;double log_prior=prior.log_density(state);
    std::seed_seq seeds{std::uint32_t(o.seed),std::uint32_t(o.seed>>32),std::uint32_t(o.chain_id),std::uint32_t(o.chain_id>>32)};
    std::mt19937_64 rng(seeds);std::normal_distribution<double> normal;
    auto uniform=[&] {return std::generate_canonical<double,53>(rng);};
    auto index=[&](int n) {return std::uniform_int_distribution<int>(0,n-1)(rng);};
    RJResult result;result.draws.reserve(o.draws);result.transitions.reserve(o.warmup+o.draws);
    for(int iteration=0;iteration<o.warmup+o.draws;++iteration) {
        const int k=static_cast<int>(state.vs.size());
        const auto move=static_cast<RJMove>(index(4));
        RJProposal proposal{state,move};
        if(move==RJMove::birth) {
            if(k<prior.max_layers && rj_insertion_length(prior,state)>0) {
                const double depth=rj_insertion_position(prior,state,uniform());
                const double vs=prior.vs_lower+(prior.vs_upper-prior.vs_lower)*uniform();
                proposal=rj_birth(prior,state,depth,vs);
            }
        } else if(move==RJMove::death) {
            if(k>prior.min_layers) proposal=rj_death(prior,state,index(k-1));
        } else if(move==RJMove::interface) {
            if(k>1) {
                proposal.index=index(k-1);proposal.state.interfaces[proposal.index]+=o.interface_step*normal(rng);
                proposal.status=prior.supports(proposal.state)?RJProposalStatus::valid:RJProposalStatus::outside_support;
            }
        } else {
            proposal.index=index(k);proposal.state.vs[proposal.index]+=o.vs_step*normal(rng);
            proposal.status=prior.supports(proposal.state)?RJProposalStatus::valid:RJProposalStatus::outside_support;
        }
        RJTransition trace;trace.warmup=iteration<o.warmup;trace.move=move;trace.proposal_status=proposal.status;
        trace.layers_before=k;trace.layers_after=k;
        if(proposal.status==RJProposalStatus::valid) {
            trace.evaluated=true;
            const auto proposed=evaluate(likelihood,proposal.state);trace.evaluation_status=proposed.status;
            if(proposed.status==EvaluationStatus::valid) {
                const double proposed_prior=prior.log_density(proposal.state);
                trace.log_accept_ratio=proposed.value-current.value+proposed_prior-log_prior+proposal.log_hastings();
                if(!std::isfinite(trace.log_accept_ratio)) trace.evaluation_status=EvaluationStatus::numerical_failure;
                else if(std::log1p(-uniform())<std::min(0.,trace.log_accept_ratio)) {
                    trace.accepted=true;state=std::move(proposal.state);current=proposed;log_prior=proposed_prior;
                    trace.layers_after=static_cast<int>(state.vs.size());
                }
            }
        }
        result.transitions.push_back(trace);
        if(!trace.warmup) result.draws.push_back({state,current.value,log_prior});
    }
    return result;
}
}
