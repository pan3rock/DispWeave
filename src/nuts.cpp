#include "dispweave/nuts.hpp"
#include "sampler_adaptation.hpp"
#include "nuts_tree.hpp"
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
    auto density=target(initial);
    if(density.status!=EvaluationStatus::valid || !std::isfinite(density.value)
       || density.gradient.size()!=initial.size() || !density.gradient.allFinite()) throw std::invalid_argument("Invalid initial NUTS density or gradient");
    const bool adapt_metric=o.metric_adaptation!=MetricAdaptation::none;
    if(adapt_metric && !o.adapt_step_size) throw std::invalid_argument("Metric adaptation requires step adaptation");
    const auto windows=adapt_metric?detail::warmup_windows(o.warmup):std::vector<detail::WarmupWindow>{};
    std::size_t window=0;
    detail::CovarianceAccumulator covariance(initial.size());
    EuclideanMetric metric=initial_metric;
    std::seed_seq seeds{std::uint32_t(o.seed),std::uint32_t(o.seed>>32),std::uint32_t(o.chain_id),std::uint32_t(o.chain_id>>32)};
    std::mt19937_64 rng(seeds);
    Eigen::VectorXd q=initial;
    double step=o.step_size;
    detail::DualAveraging adaptation(step,o.target_accept);
    NUTSResult out;out.draws.resize(o.draws,q.size());
    for(int i=0;i<o.warmup+o.draws;++i) {
        HamiltonianState start{q,metric.momentum(rng),density};
        const double joint0=density.value-metric.kinetic(start.momentum);
        if(!std::isfinite(joint0)) throw std::runtime_error("Nonfinite initial NUTS Hamiltonian");
        // 1-U is in (0,1], so a zero RNG output cannot produce log(0).
        const double log_slice=joint0+std::log1p(-std::generate_canonical<double,53>(rng));
        detail::Builder builder{target,metric,rng,step,log_slice,joint0,o.divergence_threshold};
        auto left=start,right=start,candidate=start;
        int count=1,depth=0,evaluations=0;
        bool go=true,divergent=false,selected=false;
        double acceptance=0,max_error=0;
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
        if(o.progress) o.progress(i,q,out.transitions.back());
        if(i<o.warmup && o.adapt_step_size) {
            step=adaptation.update(acceptance,i+1==o.warmup);
            if(window<windows.size() && i>=windows[window].begin) {
                covariance.add(q);
                if(i+1==windows[window].end) {
                    const auto mass=covariance.mass(o.metric_adaptation);metric=EuclideanMetric(mass);
                    out.metric_updates.push_back({i+1,covariance.count(),mass});
                    covariance=detail::CovarianceAccumulator(initial.size());++window;
                    adaptation=detail::DualAveraging(step,o.target_accept);
                }
            }
        }
        if(i>=o.warmup) out.draws.row(i-o.warmup)=q.transpose();
    }
    out.step_size=step;out.mass=metric.mass();return out;
}
}
