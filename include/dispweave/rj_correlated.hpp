#pragma once
#include "rj_proposal.hpp"
#include <random>
#include <vector>
namespace dw {
struct RJGaussianComponent {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
    double weight=1;
};
// Frozen, normalized density on ALL physical coordinates [interfaces, Vs].
// Unsupported draws are rejected once; never truncate, sort or clip this density.
class RJGaussianMixture {
    int k_;
    std::vector<Eigen::VectorXd> means_;
    std::vector<Eigen::MatrixXd> lower_;
    std::vector<double> weights_,log_constants_;
public:
    RJGaussianMixture(int layers,const std::vector<RJGaussianComponent>&);
    int layers() const {return k_;}
    double log_density(const RJState&) const;
    RJProposal propose(const RJPrior&,const RJState&,const Eigen::VectorXd& coordinates) const;
    RJProposal draw(const RJPrior&,const RJState&,std::mt19937_64&) const;
};
}
