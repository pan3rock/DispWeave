#include "dispweave/metropolis.hpp"
#include "sampler_adaptation.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
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
class FrozenMixture {
public:
    FrozenMixture(const IndependenceProposal& proposal,Eigen::Index dimension) {
        if(!std::isfinite(proposal.probability) || proposal.probability<=0 || proposal.probability>=1 || proposal.components.empty())
            throw std::invalid_argument("Independence proposal needs 0<probability<1 and nonempty components");
        std::vector<double> weights;
        for(const auto& c:proposal.components) {
            if(c.mean.size()!=dimension || !c.mean.allFinite() || c.covariance.rows()!=dimension || !std::isfinite(c.weight) || c.weight<=0)
                throw std::invalid_argument("Invalid independence component dimensions/weight");
            means_.push_back(c.mean);factors_.push_back(factor(c.covariance));weights.push_back(c.weight);
        }
        const double maximum=*std::max_element(weights.begin(),weights.end());
        double sum=0;for(auto& w:weights) {w/=maximum;sum+=w;}
        for(size_t k=0;k<weights.size();++k) {
            weights[k]/=sum;
            const double constant=std::log(weights[k])-.5*dimension*std::log(2*std::acos(-1.))-
                factors_[k].diagonal().array().log().sum();
            if(!std::isfinite(constant)) throw std::invalid_argument("Independence normalization overflow/underflow");
            constants_.push_back(constant);
        }
        choice_=std::discrete_distribution<size_t>(weights.begin(),weights.end());
    }
    Eigen::VectorXd draw(std::mt19937_64& rng,const Eigen::VectorXd& z) {
        const size_t k=choice_(rng);return means_[k]+factors_[k]*z;
    }
    double log_density(const Eigen::VectorXd& x) const {
        Eigen::VectorXd terms(means_.size());
        for(size_t k=0;k<means_.size();++k) {
            const Eigen::VectorXd z=factors_[k].triangularView<Eigen::Lower>().solve(x-means_[k]);
            terms[k]=constants_[k]-.5*z.squaredNorm();
        }
        const double maximum=terms.maxCoeff();
        return std::isfinite(maximum)?maximum+std::log((terms.array()-maximum).exp().sum()):maximum;
    }
private:
    std::vector<Eigen::VectorXd> means_;
    std::vector<Eigen::MatrixXd> factors_;
    std::vector<double> constants_;
    std::discrete_distribution<size_t> choice_;
};
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
    std::optional<FrozenMixture> mixture;
    if(o.independence) {
        mixture.emplace(*o.independence,initial.size());
        if(!std::isfinite(mixture->log_density(initial))) throw std::invalid_argument("Nonfinite initial proposal density");
    }
    const auto windows=o.adapt_covariance && o.warmup?detail::warmup_windows(o.warmup):std::vector<detail::WarmupWindow>{};
    size_t window=0;detail::CovarianceAccumulator accumulator(initial.size());
    detail::DualAveraging adaptation(o.scale,o.target_accept);double scale=o.scale;
    std::seed_seq seeds{std::uint32_t(o.seed),std::uint32_t(o.seed>>32),std::uint32_t(o.chain_id),std::uint32_t(o.chain_id>>32)};
    std::mt19937_64 rng(seeds);std::normal_distribution<double> normal;
    MetropolisResult out;out.draws.resize(o.draws,initial.size());out.log_density.resize(o.draws);
    out.transitions.reserve(o.warmup+o.draws);Eigen::VectorXd x=initial,z(initial.size());
    for(int i=0;i<o.warmup+o.draws;++i) {
        const bool independent=mixture && std::generate_canonical<double,53>(rng)<o.independence->probability;
        for(Eigen::Index j=0;j<z.size();++j) z[j]=normal(rng);
        const Eigen::VectorXd proposed=independent?mixture->draw(rng,z):Eigen::VectorXd(x+scale*(cholesky*z));
        auto p=evaluate(target,proposed);double probability=0;
        if(p.status==EvaluationStatus::valid) {
            long double delta=static_cast<long double>(p.value)-current.value;
            if(independent) {
                const double before=mixture->log_density(x),after=mixture->log_density(proposed);
                if(!std::isfinite(before) || !std::isfinite(after)) p.status=EvaluationStatus::numerical_failure;
                else delta+=static_cast<long double>(before)-after;
            }
            if(p.status==EvaluationStatus::valid) probability=delta>=0?1:std::exp(static_cast<double>(delta));
        }
        const bool accepted=std::generate_canonical<double,53>(rng)<probability;
        if(accepted) {x=proposed;current=p;}
        out.transitions.push_back({i<o.warmup,accepted,p.status,scale,probability,independent});
        if(i<o.warmup && o.adapt_scale) {
            if(!independent) scale=adaptation.update(probability,i+1==o.warmup);
            else if(i+1==o.warmup) scale=adaptation.averaged_step();
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
