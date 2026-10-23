#include "dispweave/metropolis.hpp"
#include "sampler_adaptation.hpp"
#include <Eigen/Cholesky>
#include <limits>
#include <random>
#include <stdexcept>
namespace dw {
namespace {
Eigen::MatrixXd factor(const Eigen::MatrixXd& c) {
    if(c.rows()==0 || c.rows()!=c.cols() || !c.allFinite() || !c.isApprox(c.transpose(),1e-12))
        throw std::invalid_argument("Proposal covariance must be finite and symmetric");
    Eigen::LLT<Eigen::MatrixXd> llt(.5*(c+c.transpose()));
    if(llt.info()!=Eigen::Success) throw std::invalid_argument("Proposal covariance must be positive definite");
    return llt.matrixL();
}
ValueEvaluation evaluate(const ValueDensity& target,const Eigen::VectorXd& x) {
    if(!x.allFinite()) return {};
    auto e=target(x);
    if((e.status==EvaluationStatus::valid && !std::isfinite(e.value)) ||
       (e.status==EvaluationStatus::outside_support && e.value!=-INFINITY)) e.status=EvaluationStatus::numerical_failure;
    return e;
}
}
MetropolisResult sample_metropolis(const ValueDensity& target,const Eigen::VectorXd& initial,
                                  const Eigen::MatrixXd& proposal,const MetropolisOptions& o) {
    if(!target || initial.size()==0 || initial.size()!=proposal.rows() || !initial.allFinite() ||
       o.warmup<0 || o.draws<1 || o.warmup>std::numeric_limits<int>::max()-o.draws ||
       !std::isfinite(o.scale) || o.scale<=0 || !std::isfinite(o.target_accept) || o.target_accept<=0 || o.target_accept>=1 ||
       (o.adapt_covariance && !o.adapt_scale)) throw std::invalid_argument("Invalid Metropolis options");
    auto current=evaluate(target,initial);
    if(current.status!=EvaluationStatus::valid) throw std::invalid_argument("Invalid initial Metropolis density");
    Eigen::MatrixXd cholesky=factor(proposal);
    Eigen::MatrixXd covariance=.5*(proposal+proposal.transpose());
    const auto windows=o.adapt_covariance && o.warmup?detail::warmup_windows(o.warmup):std::vector<detail::WarmupWindow>{};
    size_t window=0;detail::CovarianceAccumulator accumulator(initial.size());
    detail::DualAveraging adaptation(o.scale,o.target_accept);double scale=o.scale;
    std::seed_seq seeds{std::uint32_t(o.seed),std::uint32_t(o.seed>>32),std::uint32_t(o.chain_id),std::uint32_t(o.chain_id>>32)};
    std::mt19937_64 rng(seeds);std::normal_distribution<double> normal;
    MetropolisResult out;out.draws.resize(o.draws,initial.size());out.log_density.resize(o.draws);
    out.transitions.reserve(o.warmup+o.draws);Eigen::VectorXd x=initial,z(initial.size());
    for(int i=0;i<o.warmup+o.draws;++i) {
        for(Eigen::Index j=0;j<z.size();++j) z[j]=normal(rng);
        const Eigen::VectorXd proposed=x+scale*(cholesky*z);
        const auto p=evaluate(target,proposed);double probability=0;
        if(p.status==EvaluationStatus::valid) {
            const long double delta=static_cast<long double>(p.value)-current.value;
            probability=delta>=0?1:std::exp(static_cast<double>(delta));
        }
        const bool accepted=std::generate_canonical<double,53>(rng)<probability;
        if(accepted) {x=proposed;current=p;}
        out.transitions.push_back({i<o.warmup,accepted,p.status,scale,probability});
        if(i<o.warmup && o.adapt_scale) {
            scale=adaptation.update(probability,i+1==o.warmup);
            if(window<windows.size() && i>=windows[window].begin) {
                accumulator.add(x);
                if(i+1==windows[window].end) {
                    covariance=accumulator.covariance(MetricAdaptation::dense);cholesky=factor(covariance);
                    out.covariance_updates.push_back({i+1,accumulator.count(),covariance});
                    accumulator=detail::CovarianceAccumulator(initial.size());++window;
                    adaptation=detail::DualAveraging(scale,o.target_accept);
                }
            }
        }
        if(i>=o.warmup) {out.draws.row(i-o.warmup)=x.transpose();out.log_density[i-o.warmup]=current.value;}
    }
    out.scale=scale;out.covariance=covariance;return out;
}
}
