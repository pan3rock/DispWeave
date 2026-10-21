#pragma once
#include <Eigen/Core>
namespace dw {
struct LocalResolution {
    // A = whitened_jacobian * diag(parameter_scale), q = q0 + diag(scale) x.
    Eigen::MatrixXd scaled_jacobian, right_vectors, resolution;
    Eigen::VectorXd singular_values, filter_factors;
    Eigen::Index numerical_rank=0;
    double rank_threshold=0, effective_dimension=0;
};
// Ridge objective: (||A x - b||^2 + ridge^2 ||x||^2)/2, ridge > 0.
// Numerical rank is a floating-point diagnostic, not data-supported complexity.
LocalResolution local_resolution(const Eigen::MatrixXd& whitened_jacobian,
                                 const Eigen::VectorXd& parameter_scale,
                                 double ridge,double relative_rank_tolerance=1e-12);
}
