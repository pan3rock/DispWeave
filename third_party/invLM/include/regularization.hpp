#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

namespace inversion {

struct RegularizationEvaluation {
    Eigen::VectorXd residuals;
    Eigen::MatrixXd jacobian;
    double cost_ref = 0.0;
    double cost_smooth = 0.0;
    std::vector<std::string> term_names;
    std::vector<double> term_weights;
    std::vector<double> term_costs;
    Eigen::VectorXd interface_weights;

    double cost_total() const { return cost_ref + cost_smooth; }
};

struct RegularizationSettings {
    std::string type = "perturbation_smooth";
    double lambda_ref = 1.0;
    double lambda_smooth = 0.0;
    bool interface_adaptive = false;
    double interface_scale = 0.1;
    double tv_epsilon = 1e-3;
    int fixed_surface_layers = 0;
};

/// Deep module for an ordered set of regularization adapters. Reference-model
/// damping is first, followed by the selected smoothness adapter when enabled.
class Regularization {
public:
    Regularization(const Eigen::VectorXd& depth,
                   const Eigen::VectorXd& ref_vs,
                   const RegularizationSettings& settings);
    Regularization(const Eigen::VectorXd& depth,
                   const Eigen::VectorXd& ref_vs,
                   double lambda_ref,
                   double lambda_smooth);
    ~Regularization();
    Regularization(Regularization&&) noexcept;
    Regularization& operator=(Regularization&&) noexcept;
    Regularization(const Regularization&) = delete;
    Regularization& operator=(const Regularization&) = delete;

    bool evaluate(const Eigen::VectorXd& vs, bool with_jacobian,
                  RegularizationEvaluation& result) const;

    int num_parameters() const;
    int num_residuals() const;
    int num_reference_residuals() const;
    int num_smoothness_residuals() const;

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace inversion
