#include "dispweave/forward.hpp"
#include "dispersion_kernel.hpp"
#include "dunkin_derivative.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
const char* to_string(SensitivityStatus s) {
    switch(s) {
        case SensitivityStatus::not_requested:return "not_requested";
        case SensitivityStatus::valid:return "valid";
        case SensitivityStatus::unavailable:return "unavailable";
    }
    return "unknown";
}
const char* to_string(SensitivityMethod m) {
    switch(m) {
        case SensitivityMethod::none:return "none";
        case SensitivityMethod::energy:return "energy";
        case SensitivityMethod::implicit:return "implicit";
    }
    return "unknown";
}
SensitivityResult rayleigh_sensitivity(const LayeredModel& m,double f,double c,SensitivityMethod method) {
    m.validate();
    SensitivityResult result;
    if(method==SensitivityMethod::none) return result;
    result.status=SensitivityStatus::unavailable;
    result.dc_dvs=result.dc_dvp=result.dc_drho=Eigen::VectorXd::Constant(m.vs.size(),NAN);
    if(!std::isfinite(f) || f<=0 || !std::isfinite(c) || c<=0 || c>=m.vs.tail(1)[0]) {
        result.message="Sensitivity requires a positive frequency and resolved trapped phase"; return result;
    }
    VelocityModel native;
    native.nl=static_cast<int>(m.vs.size()); native.depth=m.depth;
    native.vs=m.vs; native.vp=m.vp; native.rho=m.rho;
    try {
        Eigen::VectorXd vs,vp,rho;
        if(method==SensitivityMethod::energy) {
            fwd::RayleighKernel kernel(native.nl); kernel.set_model(native);
            const auto k=kernel.compute(2*std::acos(-1.)*f,c);
            result.energy_residual=k.energy_residual;
            result.stabilized=k.stabilized_eigenfunctions;
            if(!std::isfinite(k.energy_residual) || k.energy_residual>1e-7)
                throw std::runtime_error("Energy residual exceeds upstream 1e-7 acceptance threshold");
            vs=k.dc_dvs; vp=k.dc_dvp; rho=k.dc_drho;
        } else if(method==SensitivityMethod::implicit) {
            const auto k=fwd::dunkin_derivatives(native,2*std::acos(-1.)*f,c);
            result.root_correction=k.root_correction;
            vs=k.dc_dvs; vp=k.dc_dvp; rho=k.dc_drho;
        } else throw std::invalid_argument("Unknown sensitivity method");
        if(vs.size()!=native.nl || vp.size()!=native.nl || rho.size()!=native.nl ||
           !vs.allFinite() || !vp.allFinite() || !rho.allFinite())
            throw std::runtime_error("Nonfinite or malformed kernel");
        result.dc_dvs=vs; result.dc_dvp=vp; result.dc_drho=rho;
        result.status=SensitivityStatus::valid;
    } catch(const std::runtime_error& e) { result.message=e.what(); }
    return result;
}
}
