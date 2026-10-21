#pragma once
#include "model.hpp"

namespace fwd {
// Solid layers with optional fixed surface water: no eigenfunctions,
// group velocity or energy-residual diagnostic is computed by this method.
struct DunkinDerivatives {
    Eigen::ArrayXd dc_dvs, dc_dvp, dc_drho;
    double root_correction = 0;
};
DunkinDerivatives dunkin_derivatives(const VelocityModel& model, double omega, double c);
// SH secular equation (shared by both forward methods); dc/dVp is exactly zero.
DunkinDerivatives love_derivatives(const VelocityModel& model, double omega, double c);
}
