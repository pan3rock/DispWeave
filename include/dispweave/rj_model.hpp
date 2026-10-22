#pragma once
#include <Eigen/Core>
namespace dw {
// Physical coordinates: K total layers including the halfspace, K-1 interfaces.
struct RJState { Eigen::VectorXd interfaces,vs; };
struct RJPrior {
    int min_layers=1,max_layers=5;
    double depth_limit=10,min_spacing=.1,vs_lower=.5,vs_upper=4;
    Eigen::VectorXd layer_weights=Eigen::VectorXd::Ones(5);
    void validate() const;
    bool supports(const RJState&) const;
    double log_density(const RJState&) const;
};
}
