#pragma once
#include <Eigen/Dense>

/// 1D isotropic velocity model. Vp/Rho are either derived from Vs or fixed
/// from a five-column reference model, depending on vs2model_type. Optional
/// surface water has Vs=0 and explicit fixed Vp/rho; internal fluids unsupported.
struct VelocityModel {
    int nl = 0;
    Eigen::VectorXd depth;   // top depth of each layer [km]
    Eigen::VectorXd vs;      // S-wave velocity [km/s]
    Eigen::VectorXd vp;      // P-wave velocity [km/s]
    Eigen::VectorXd rho;     // density [g/cm^3]
};
