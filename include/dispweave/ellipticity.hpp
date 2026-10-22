#pragma once
#include "forward.hpp"
#include <cmath>
namespace dw {
enum class EllipticityStatus { valid, root_unavailable, eigenfunction_failure, near_pole };
const char* to_string(EllipticityStatus);
struct EllipticityRow {
    RootResult root;
    EllipticityStatus status=EllipticityStatus::root_unavailable;
    double ratio=NAN; // |horizontal surface displacement| / |vertical|; dimensionless
    double energy_residual=NAN;
    bool stabilized=false;
    std::string message;
};
// invLM fields use refined invLM roots. No analytic gradient contract.
// A tighter bracket is not a global root-completeness or field-accuracy certificate.
std::vector<EllipticityRow> rayleigh_ellipticity(const LayeredModel&,const std::vector<DispersionQuery>&,
                                               SecularMethod=SecularMethod::dunkin);
}
