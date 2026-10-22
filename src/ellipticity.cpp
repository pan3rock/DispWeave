#include "dispweave/ellipticity.hpp"
#include "dispersion_kernel.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
const char* to_string(EllipticityStatus status) {
    switch(status) {
    case EllipticityStatus::valid:return "valid";
    case EllipticityStatus::root_unavailable:return "root_unavailable";
    case EllipticityStatus::eigenfunction_failure:return "eigenfunction_failure";
    case EllipticityStatus::near_pole:return "near_pole";
    }
    return "unknown";
}
std::vector<EllipticityRow> rayleigh_ellipticity(const LayeredModel& model,
                                               const std::vector<DispersionQuery>& queries,SecularMethod method) {
    const InvLMForward forward(method,RootAccuracy::refined);
    model.validate();
    const auto roots=forward.evaluate(model,queries,SensitivityMethod::none);
    if(roots.rows.size()!=queries.size()) throw std::runtime_error("Malformed ellipticity root result");
    VelocityModel native;native.nl=static_cast<int>(model.vs.size());native.depth=model.depth;
    native.vs=model.vs;native.vp=model.vp;native.rho=model.rho;
    fwd::RayleighKernel kernel(native.nl);kernel.set_model(native);
    std::vector<EllipticityRow> result(queries.size());
    for(size_t i=0;i<queries.size();++i) {
        auto& row=result[i];row.root=roots.rows[i];
        if(row.root.status!=RootStatus::valid) {row.message="Phase unavailable; no ellipticity computed";continue;}
        row.status=EllipticityStatus::eigenfunction_failure;
        try {
            const auto field=kernel.compute(2*std::acos(-1.)*queries[i].frequency_hz,row.root.phase_velocity);
            row.energy_residual=field.energy_residual;row.stabilized=field.stabilized_eigenfunctions;
            if(field.ur.size()!=native.nl || field.uz.size()!=native.nl || !field.ur.allFinite() || !field.uz.allFinite()
               || !std::isfinite(field.energy_residual) || field.energy_residual>1e-7)
                throw std::runtime_error("Invalid eigenfunction or energy residual above 1e-7");
            const double horizontal=std::abs(field.ur[0]),vertical=std::abs(field.uz[0]);
            const double scale=std::max(horizontal,vertical);
            if(scale==0) throw std::runtime_error("Zero surface displacement");
            // A relative surface-displacement criterion: invariant to normalization.
            if(vertical<=1e-7*scale) {row.status=EllipticityStatus::near_pole;row.message="Vertical surface displacement <=1e-7 of surface amplitude";continue;}
            row.ratio=horizontal/vertical;
            if(!std::isfinite(row.ratio)) throw std::runtime_error("Nonfinite ellipticity");
            row.status=EllipticityStatus::valid;
        } catch(const std::runtime_error& e) {row.message=e.what();row.ratio=NAN;}
    }
    return result;
}
}
