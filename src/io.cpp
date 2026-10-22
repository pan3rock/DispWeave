#include "dispweave/io.hpp"
#include "version.hpp"
#include <toml.hpp>
#include <fstream>
#include <iomanip>
#include <stdexcept>
namespace dw {
ForwardConfig read_forward_config(const std::filesystem::path& path) {
    const auto document=toml::parse(path.string());
    const auto model=toml::find(document,"model");
    auto vector=[&](const char* key)->Eigen::VectorXd {
        auto a=toml::find<std::vector<double>>(model,key);
        return Eigen::Map<const Eigen::VectorXd>(a.data(),a.size());
    };
    ForwardConfig config; config.model={vector("depth"),vector("vs"),vector("vp"),vector("rho")}; config.model.validate();
    const auto query=toml::find(document,"queries");
    auto frequency=toml::find<std::vector<double>>(query,"frequency_hz");
    auto mode=toml::find<std::vector<int>>(query,"mode");
    if(frequency.empty() || frequency.size()!=mode.size()) throw std::invalid_argument("Query arrays must be nonempty and equal-sized; rows are paired, not a Cartesian grid");
    for(size_t i=0;i<mode.size();++i) config.queries.push_back({frequency[i],mode[i]});
    return config;
}
void write_forward_result(const std::filesystem::path& path,const ForwardConfig& c,const ForwardResult& r) {
    std::ofstream f(path);
    if(!f) throw std::runtime_error("Cannot write "+path.string());
    f.exceptions(std::ios::badbit|std::ios::failbit);
    f<<std::setprecision(17);
    f<<"# DispWeave "<<DW_VERSION<<"; invLM "<<DW_INVLM_COMMIT<<"\n# compiler "<<DW_COMPILER<<"; build "<<DW_BUILD_TYPE
     <<"\n# Eigen 3.4.1; toml11 4.4.0; CLI11 2.4.2; backend dunkin; sensitivity "<<to_string(r.sensitivity_method)<<"\n";
    for(Eigen::Index i=0;i<c.model.vs.size();++i)
        f<<"# model depth_km vs_km_s vp_km_s rho_g_cm3: "<<c.model.depth[i]<<' '<<c.model.vs[i]<<' '<<c.model.vp[i]<<' '<<c.model.rho[i]<<'\n';
    f<<"# row frequency_hz mode phase_km_s status roots_found message";
    if(r.sensitivity_method!=SensitivityMethod::none) {
        f<<" sensitivity_status energy_residual root_correction stabilized sensitivity_message";
        for(const char* property:{"Vs","Vp","rho"})
            for(Eigen::Index layer=0;layer<c.model.vs.size();++layer) f<<" dc_d"<<property<<"_"<<layer;
    }
    f<<'\n';
    for(size_t i=0;i<r.rows.size();++i) {
        const auto& row=r.rows[i];
        f<<i<<' '<<c.queries[i].frequency_hz<<' '<<c.queries[i].mode<<' '<<row.phase_velocity<<' '
         <<to_string(row.status)<<' '<<row.roots_found<<' '<<std::quoted(row.message);
        if(r.sensitivity_method!=SensitivityMethod::none) {
            const auto& s=row.sensitivity;
            f<<' '<<to_string(s.status)<<' '<<s.energy_residual<<' '<<s.root_correction<<' '<<s.stabilized<<' '<<std::quoted(s.message);
            for(const auto* property:{&s.dc_dvs,&s.dc_dvp,&s.dc_drho})
                for(Eigen::Index layer=0;layer<property->size();++layer) f<<' '<<(*property)[layer];
        }
        f<<'\n';
    }
    f.close();
}
void write_ellipticity_result(const std::filesystem::path& path,const ForwardConfig& c,const std::vector<EllipticityRow>& rows) {
    if(rows.size()!=c.queries.size()) throw std::invalid_argument("Ellipticity output dimensions disagree");
    std::ofstream f(path);if(!f) throw std::runtime_error("Cannot write "+path.string());
    f.exceptions(std::ios::badbit|std::ios::failbit);f<<std::setprecision(17);
    f<<"# DispWeave "<<DW_VERSION<<"; invLM "<<DW_INVLM_COMMIT
     <<"\n# observable=rayleigh_modal_ellipticity; |ur(surface)|/|uz(surface)|; dimensionless; NOT ambient-noise H/V"
     <<"\n# root refinement: atol=0, rtol=8*double_epsilon; near_pole: |uz| <= 1e-7 * max(|ur|,|uz|); no analytic gradient"
     <<"\n# row frequency_hz mode phase_km_s root_status ellipticity ellipticity_status energy_residual stabilized message roots_found root_message\n";
    for(Eigen::Index i=0;i<c.model.vs.size();++i)
        f<<"# model depth_km vs_km_s vp_km_s rho_g_cm3: "<<c.model.depth[i]<<' '<<c.model.vs[i]<<' '<<c.model.vp[i]<<' '<<c.model.rho[i]<<'\n';
    for(size_t i=0;i<rows.size();++i) {
        const auto& r=rows[i];f<<i<<' '<<c.queries[i].frequency_hz<<' '<<c.queries[i].mode<<' '
            <<r.root.phase_velocity<<' '<<to_string(r.root.status)<<' '<<r.ratio<<' '<<to_string(r.status)<<' '
            <<r.energy_residual<<' '<<r.stabilized<<' '<<std::quoted(r.message)<<' '<<r.root.roots_found<<' '<<std::quoted(r.root.message)<<'\n';
    }
    f.close();
}

}
