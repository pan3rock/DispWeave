#include "dispweave/rjmcmc.hpp"
#include <limits>
#include <random>
#include <stdexcept>
#include <memory>
namespace dw {
namespace {
RJEvaluation evaluate(const RJLikelihood& likelihood,const RJState& state) {
    auto e=likelihood(state);
    if(e.status==EvaluationStatus::valid && !std::isfinite(e.value)) e.status=EvaluationStatus::numerical_failure;
    return e;
}
void validate(const RJPrior& prior,const RJState& initial,const RJOptions& o) {
    prior.validate();
    if(!prior.supports(initial) || o.warmup<0 || o.draws<1 || o.warmup>std::numeric_limits<int>::max()-o.draws
       || !std::isfinite(o.velocity_refresh_probability) || o.velocity_refresh_probability<0 || o.velocity_refresh_probability>1
       || !std::isfinite(o.vs_step) || o.vs_step<=0 || !std::isfinite(o.interface_step) || o.interface_step<=0)
        throw std::invalid_argument("Invalid RJ initial state or sampler options");
    // Birth/death changes K by one. Do not silently explore only one disconnected class.
    if((prior.layer_weights.array()<=0).any()) throw std::invalid_argument("RJ sampler needs positive layer weights throughout the declared range");
}

class Chain {
    const RJLikelihood& likelihood;
    const RJPrior& prior;
    const RJOptions o;
    std::mt19937_64 rng;
    std::normal_distribution<double> normal;
    double uniform() {return std::generate_canonical<double,53>(rng);}
    int index(int n) {return std::uniform_int_distribution<int>(0,n-1)(rng);}
public:
    RJDraw current;
    Chain(const RJLikelihood& f,const RJPrior& p,const RJState& initial,const RJOptions& options)
        :likelihood(f),prior(p),o(options) {
        validate(prior,initial,o);
        const auto e=evaluate(likelihood,initial);
        if(e.status!=EvaluationStatus::valid) throw std::invalid_argument("Invalid initial RJ likelihood");
        current={initial,e.value,prior.log_density(initial)};
        std::seed_seq seeds{std::uint32_t(o.seed),std::uint32_t(o.seed>>32),std::uint32_t(o.chain_id),std::uint32_t(o.chain_id>>32)};
        rng.seed(seeds);
    }
    RJTransition step(int iteration,double beta=1) {
        const int k=static_cast<int>(current.state.vs.size());
        auto move=static_cast<RJMove>(index(4));
        if(move==RJMove::velocity && o.velocity_refresh_probability>0 && uniform()<o.velocity_refresh_probability) move=RJMove::velocity_refresh;
        RJProposal proposal{current.state,move};
        if(move==RJMove::birth) {
            if(k<prior.max_layers && rj_insertion_length(prior,current.state)>0) {
                const double depth=rj_insertion_position(prior,current.state,uniform());
                const double vs=prior.vs_lower+(prior.vs_upper-prior.vs_lower)*uniform();
                proposal=rj_birth(prior,current.state,depth,vs);
            }
        } else if(move==RJMove::death) {
            if(k>prior.min_layers) proposal=rj_death(prior,current.state,index(k-1));
        } else if(move==RJMove::interface) {
            if(k>1) {
                proposal.index=index(k-1);proposal.state.interfaces[proposal.index]+=o.interface_step*normal(rng);
                proposal.status=prior.supports(proposal.state)?RJProposalStatus::valid:RJProposalStatus::outside_support;
            }
        } else if(move==RJMove::velocity_refresh) {
            Eigen::VectorXd vs(k);
            for(int j=0;j<k;++j) vs[j]=prior.vs_lower+(prior.vs_upper-prior.vs_lower)*uniform();
            proposal=rj_velocity_refresh(prior,current.state,vs);
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
                trace.log_accept_ratio=(beta==1 ? proposed.value-current.log_likelihood : beta*(proposed.value-current.log_likelihood))+proposed_prior-current.log_prior+proposal.log_hastings();
                if(!std::isfinite(trace.log_accept_ratio)) trace.evaluation_status=EvaluationStatus::numerical_failure;
                else if(std::log1p(-uniform())<std::min(0.,trace.log_accept_ratio)) {
                    trace.accepted=true;current={std::move(proposal.state),proposed.value,proposed_prior};
                    trace.layers_after=static_cast<int>(current.state.vs.size());
                }
            }
        }
        return trace;
    }
};
}
RJResult sample_rjmcmc(const RJLikelihood& likelihood,const RJPrior& prior,const RJState& initial,const RJOptions& o) {
    Chain chain(likelihood,prior,initial,o);
    RJResult result;result.draws.reserve(o.draws);result.transitions.reserve(o.warmup+o.draws);
    for(int iteration=0;iteration<o.warmup+o.draws;++iteration) {
        result.transitions.push_back(chain.step(iteration));
        if(iteration>=o.warmup) result.draws.push_back(chain.current);
    }
    return result;
}

double rj_swap_log_ratio(double ti,double tj,double li,double lj) {
    if(!std::isfinite(ti) || !std::isfinite(tj) || ti<1 || tj<1 || !std::isfinite(li) || !std::isfinite(lj))
        throw std::invalid_argument("Invalid RJ temperature swap inputs");
    return static_cast<double>((1/static_cast<long double>(ti)-1/static_cast<long double>(tj))*(static_cast<long double>(lj)-li));
}

RJPTResult sample_rj_tempering(const std::vector<RJLikelihood>& targets,const RJPrior& prior,
                              const std::vector<RJState>& starts,const RJPTOptions& options) {
    const auto& temps=options.temperatures;const auto& o=options.local;
    if(temps.empty() || targets.size()!=temps.size() || starts.size()!=temps.size() || temps[0]!=1 || options.swap_every<1
       || options.cold_start_steps<0 || (options.cold_start_steps>0 && options.cold_start_steps>=o.warmup))
        throw std::invalid_argument("Invalid RJ tempering ladder, swap interval or cold-start warmup length");
    for(size_t t=0;t<temps.size();++t) {
        if(!std::isfinite(temps[t]) || temps[t]<1 || (t>0 && temps[t]<=temps[t-1]))
            throw std::invalid_argument("RJ temperatures must be finite, >=1 and strictly increasing");
        validate(prior,starts[t],o);
    }
    if(o.chain_id>(std::numeric_limits<std::uint64_t>::max()-(temps.size()-1))/temps.size())
        throw std::invalid_argument("RJ replica stream index overflow");
    std::vector<std::unique_ptr<Chain>> chains;
    RJPTResult result;result.temperatures.resize(temps.size());std::vector<int> ids;
    if(options.save_warmup) result.warmup_draws.resize(temps.size());
    for(size_t t=0;t<temps.size();++t) {
        RJOptions local=o;local.chain_id=o.chain_id*temps.size()+t;
        chains.push_back(std::make_unique<Chain>(targets[t],prior,starts[t],local));ids.push_back(static_cast<int>(t));
        result.initial_states.push_back(chains[t]->current);
        if(options.save_warmup) result.warmup_draws[t].reserve(o.warmup);
        result.temperatures[t].draws.reserve(o.draws);
        result.temperatures[t].transitions.reserve(o.warmup+o.draws);
    }
    std::seed_seq seeds{std::uint32_t(o.seed),std::uint32_t(o.seed>>32),std::uint32_t(o.chain_id),std::uint32_t(o.chain_id>>32),0x52535054u};
    std::mt19937_64 rng(seeds);
    int exchange=0;
    for(int iteration=0;iteration<o.warmup+o.draws;++iteration) {
        for(size_t t=0;t<temps.size();++t)
            result.temperatures[t].transitions.push_back(chains[t]->step(iteration,iteration<options.cold_start_steps?1:1/temps[t]));
        if(iteration>=options.cold_start_steps && (iteration-options.cold_start_steps+1)%options.swap_every==0) {
            const size_t first=temps.size()==2?0:exchange%2;
            for(size_t t=first;t+1<temps.size();t+=2) {
                auto& a=chains[t]->current;auto& b=chains[t+1]->current;
                RJSwap swap;swap.iteration=iteration;swap.warmup=iteration<o.warmup;
                swap.lower=t;swap.upper=t+1;swap.replica_lower=ids[t];swap.replica_upper=ids[t+1];
                swap.log_like_lower=a.log_likelihood;swap.log_like_upper=b.log_likelihood;
                swap.log_accept_ratio=rj_swap_log_ratio(temps[t],temps[t+1],a.log_likelihood,b.log_likelihood);
                swap.accepted=std::log1p(-std::generate_canonical<double,53>(rng))<std::min(0.,swap.log_accept_ratio);
                if(swap.accepted) {std::swap(a,b);std::swap(ids[t],ids[t+1]);}
                result.swaps.push_back(swap);
            }
            ++exchange;
        }
        result.replica_ids.push_back(ids);
        std::vector<int> layers;
        for(size_t t=0;t<temps.size();++t) {
            layers.push_back(static_cast<int>(chains[t]->current.state.vs.size()));
            if(iteration>=o.warmup) result.temperatures[t].draws.push_back(chains[t]->current);
            else if(options.save_warmup) result.warmup_draws[t].push_back(chains[t]->current);
        }
        result.layers_after_swaps.push_back(std::move(layers));
    }
    return result;
}
}
