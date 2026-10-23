#include "dispweave/nuts.hpp"
#include "sampler_adaptation.hpp"
#include "nuts_tree.hpp"
#include "nuts_multinomial.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace dw {
NUTSResult sample_nuts(const LogDensity& target,const Eigen::VectorXd& initial,const EuclideanMetric& initial_metric,const NUTSOptions& o) {
    if(o.warmup<0 || o.draws<1 || o.warmup>std::numeric_limits<int>::max()-o.draws
       || o.max_depth<1 || o.max_depth>20 || !std::isfinite(o.step_size) || o.step_size<=0
       || !std::isfinite(o.target_accept) || o.target_accept<=0 || o.target_accept>=1
       || !std::isfinite(o.divergence_threshold) || o.divergence_threshold<=0
       || initial.size()!=initial_metric.size() || !initial.allFinite()) throw std::invalid_argument("Invalid NUTS options or dimensions");
    if(o.kernel!=NUTSKernel::slice && o.kernel!=NUTSKernel::multinomial) throw std::invalid_argument("Unknown NUTS kernel");
    if(o.find_initial_step_size && (!o.adapt_step_size || o.warmup<1)) throw std::invalid_argument("Step-size search requires adaptive warmup");
    auto density=target(initial);
    if(density.status!=EvaluationStatus::valid || !std::isfinite(density.value)
       || density.gradient.size()!=initial.size() || !density.gradient.allFinite()) throw std::invalid_argument("Invalid initial NUTS density or gradient");
    const bool adapt_metric=o.metric_adaptation!=MetricAdaptation::none;
    if(!std::isfinite(o.metric_shrinkage) || o.metric_shrinkage<0 || (o.metric_shrinkage>0 && !adapt_metric))
        throw std::invalid_argument("metric_shrinkage must be nonnegative and requires metric adaptation");
    if(adapt_metric && !o.adapt_step_size) throw std::invalid_argument("Metric adaptation requires step adaptation");
    const auto windows=adapt_metric?detail::warmup_windows(o.warmup):std::vector<detail::WarmupWindow>{};
    std::size_t window=0;
    detail::CovarianceAccumulator covariance(initial.size());
    EuclideanMetric metric=initial_metric;
    Eigen::MatrixXd previous_covariance;
    if(o.metric_shrinkage>0) {
        previous_covariance=initial_metric.mass().llt().solve(Eigen::MatrixXd::Identity(initial.size(),initial.size()));
        previous_covariance=.5*(previous_covariance+previous_covariance.transpose()).eval();
    }
    std::seed_seq seeds{std::uint32_t(o.seed),std::uint32_t(o.seed>>32),std::uint32_t(o.chain_id),std::uint32_t(o.chain_id>>32)};
    std::mt19937_64 rng(seeds);
    Eigen::VectorXd q=initial;
    double step=o.step_size;
    NUTSResult out;out.draws.resize(o.draws,q.size());
    const auto search=[&](int iteration) {
        const auto result=detail::find_step_size(target,q,density,metric,rng,step,iteration);
        step=result.step_size;out.step_size_searches.push_back(result);
    };
    if(o.find_initial_step_size) search(0);
    detail::DualAveraging adaptation(step,o.target_accept);
    for(int i=0;i<o.warmup+o.draws;++i) {
        HamiltonianState start{q,metric.momentum(rng),density};
        const double joint0=density.value-metric.kinetic(start.momentum);
        if(!std::isfinite(joint0)) throw std::runtime_error("Nonfinite initial NUTS Hamiltonian");
        double acceptance=0;
        if(o.kernel==NUTSKernel::multinomial) {
            auto transition=detail::multinomial_transition(target,start,metric,rng,step,o.max_depth,o.divergence_threshold,i<o.warmup);
            q=std::move(transition.state.position);density=std::move(transition.state.density);
            acceptance=transition.diagnostic.acceptance_probability;
            out.transitions.push_back(transition.diagnostic);
        } else {
            // 1-U is in (0,1], so a zero RNG output cannot produce log(0).
            const double log_slice=joint0+std::log1p(-std::generate_canonical<double,53>(rng));
            detail::Builder builder{target,metric,rng,step,log_slice,joint0,o.divergence_threshold};
            auto left=start,right=start,candidate=start;
            int count=1,depth=0,evaluations=0;
            bool go=true,divergent=false,selected=false;
            double max_error=0;
            auto status=EvaluationStatus::valid;
            while(go && depth<o.max_depth) {
                const int direction=builder.uniform()<.5?-1:1;
                auto tree=builder.build(direction<0?left:right,direction,depth);
                if(direction<0) left=tree.left;else right=tree.right;
                // Algorithm 6 outer transition uses min(1,n'/n), NOT n'/(n+n').
                if(tree.go && tree.count>0 && builder.uniform()<std::min(1.,double(tree.count)/count)) {
                    candidate=tree.candidate;selected=true;
                }
                count+=tree.count;evaluations+=tree.evaluations;
                // Algorithm 6 adapts with the final doubling's average acceptance.
                acceptance=tree.alpha/tree.evaluations;
                max_error=std::max(max_error,tree.max_error);divergent|=tree.divergent;
                if(tree.status!=EvaluationStatus::valid) status=tree.status;
                go=tree.go && detail::continuing(left,right,metric);++depth;
            }
            q=candidate.position;density=candidate.density;
            out.transitions.push_back({i<o.warmup,selected,divergent,go && depth==o.max_depth,depth,evaluations,step,acceptance,max_error,status});
        }
        if(o.progress) o.progress(i,q,out.transitions.back());
        if(i<o.warmup && o.adapt_step_size) {
            step=adaptation.update(acceptance,i+1==o.warmup);
            if(window<windows.size() && i>=windows[window].begin) {
                covariance.add(q);
                if(i+1==windows[window].end) {
                    const auto mass=o.metric_shrinkage>0?
                        covariance.mass(o.metric_adaptation,previous_covariance,o.metric_shrinkage):covariance.mass(o.metric_adaptation);
                    if(o.metric_shrinkage>0)
                        previous_covariance=covariance.covariance(o.metric_adaptation,previous_covariance,o.metric_shrinkage);
                    metric=EuclideanMetric(mass);
                    out.metric_updates.push_back({i+1,covariance.count(),mass});
                    covariance=detail::CovarianceAccumulator(initial.size());++window;
                    if(o.find_initial_step_size) search(i+1);
                    adaptation=detail::DualAveraging(step,o.target_accept);
                }
            }
        }
        if(i>=o.warmup) out.draws.row(i-o.warmup)=q.transpose();
    }
    out.step_size=step;out.mass=metric.mass();return out;
}
}
