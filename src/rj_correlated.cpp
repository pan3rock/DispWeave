#include "dispweave/rj_correlated.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace dw {
namespace {
Eigen::VectorXd pack(const RJState& s) {
    Eigen::VectorXd x(s.interfaces.size()+s.vs.size());x<<s.interfaces,s.vs;return x;
}
}
RJGaussianMixture::RJGaussianMixture(int layers,const std::vector<RJGaussianComponent>& components):k_(layers) {
    if(k_<1 || k_>10000 || components.empty()) throw std::invalid_argument("Invalid correlated RJ dimensions/components");
    double scale=0;for(const auto& c:components) {
        if(!std::isfinite(c.weight) || c.weight<=0) throw std::invalid_argument("Invalid correlated RJ weight");
        scale=std::max(scale,c.weight);
    }
    double sum=0;for(const auto& c:components) sum+=c.weight/scale;
    for(const auto& c:components) {
        const int n=2*k_-1;
        if(c.mean.size()!=n || c.covariance.rows()!=n || c.covariance.cols()!=n || !c.mean.allFinite() || !c.covariance.allFinite()
           || !c.covariance.isApprox(c.covariance.transpose(),1e-12)) throw std::invalid_argument("Invalid correlated RJ mean/covariance");
        Eigen::LLT<Eigen::MatrixXd> llt(c.covariance);
        if(llt.info()!=Eigen::Success) throw std::invalid_argument("Correlated RJ covariance must be positive definite");
        Eigen::MatrixXd l=llt.matrixL();double w=(c.weight/scale)/sum;
        if(w<=0) throw std::invalid_argument("Correlated RJ weight underflow");
        means_.push_back(c.mean);lower_.push_back(l);weights_.push_back(w);
        log_constants_.push_back(std::log(w)-.5*n*std::log(2*std::acos(-1.))-l.diagonal().array().log().sum());
    }
}
double RJGaussianMixture::log_density(const RJState& s) const {
    if(s.vs.size()!=k_ || s.interfaces.size()!=k_-1) throw std::invalid_argument("Correlated RJ state dimension mismatch");
    auto x=pack(s);if(!x.allFinite()) return -INFINITY;
    std::vector<double> terms;double maximum=-INFINITY;
    for(size_t j=0;j<means_.size();++j) {
        Eigen::VectorXd y=lower_[j].triangularView<Eigen::Lower>().solve(x-means_[j]);
        double v=log_constants_[j]-.5*y.squaredNorm();terms.push_back(v);maximum=std::max(maximum,v);
    }
    if(!std::isfinite(maximum)) return maximum;
    double sum=0;for(double t:terms) sum+=std::exp(t-maximum);return maximum+std::log(sum);
}
RJProposal RJGaussianMixture::propose(const RJPrior& p,const RJState& current,const Eigen::VectorXd& x) const {
    if(!p.supports(current) || current.vs.size()!=k_) throw std::invalid_argument("Invalid correlated RJ source");
    RJProposal result{current,RJMove::correlated};result.status=RJProposalStatus::outside_support;
    if(x.size()!=2*k_-1 || !x.allFinite()) return result;
    result.state={x.head(k_-1),x.tail(k_)};
    if(!p.supports(result.state)) return result;
    result.status=RJProposalStatus::valid;
    result.log_forward=log_density(result.state);result.log_reverse=log_density(current);
    return result;
}
RJProposal RJGaussianMixture::draw(const RJPrior& p,const RJState& current,std::mt19937_64& rng) const {
    std::discrete_distribution<size_t> choose(weights_.begin(),weights_.end());size_t j=choose(rng);
    std::normal_distribution<double> normal;Eigen::VectorXd z(2*k_-1);
    for(int i=0;i<z.size();++i) z[i]=normal(rng);
    return propose(p,current,means_[j]+lower_[j]*z);
}
RJProposal RJGaussianMixture::propose_local(const RJPrior& p,const RJState& current,const Eigen::VectorXd& x) const {
    if(!p.supports(current) || current.vs.size()!=k_) throw std::invalid_argument("Invalid local correlated source");
    RJProposal result{current,RJMove::correlated};result.status=RJProposalStatus::outside_support;
    if(x.size()!=2*k_-1 || !x.allFinite()) return result;
    result.state={x.head(k_-1),x.tail(k_)};
    if(p.supports(result.state)) result.status=RJProposalStatus::valid;
    // Equal full-mixture forward/reverse densities cancel, including near boundaries:
    // proposals outside support are rejected once, never truncated or redrawn.
    return result;
}
RJProposal RJGaussianMixture::draw_local(const RJPrior& p,const RJState& current,std::mt19937_64& rng,double scale) const {
    if(!std::isfinite(scale) || scale<=0) throw std::invalid_argument("Local correlated scale must be positive finite");
    if(!p.supports(current) || current.vs.size()!=k_) throw std::invalid_argument("Invalid local correlated source");
    std::discrete_distribution<size_t> choose(weights_.begin(),weights_.end());size_t j=choose(rng);
    std::normal_distribution<double> normal;Eigen::VectorXd z(2*k_-1);
    for(int i=0;i<z.size();++i) z[i]=normal(rng);
    return propose_local(p,current,pack(current)+scale*(lower_[j]*z));
}
}
