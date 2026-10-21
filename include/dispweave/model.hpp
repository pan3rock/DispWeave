#pragma once
#include <Eigen/Core>
namespace dw {
// km, km/s, g/cm^3. Final layer is an infinite halfspace.
struct LayeredModel {
    Eigen::VectorXd depth, vs, vp, rho;
    void validate() const;
};
}
