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
     <<"\n# Eigen 3.4.1; toml11 4.4.0; CLI11 2.4.2; backend dunkin; no sensitivities\n";
    for(Eigen::Index i=0;i<c.model.vs.size();++i)
        f<<"# model depth_km vs_km_s vp_km_s rho_g_cm3: "<<c.model.depth[i]<<' '<<c.model.vs[i]<<' '<<c.model.vp[i]<<' '<<c.model.rho[i]<<'\n';
    f<<"# row frequency_hz mode phase_km_s status roots_found message\n";
    for(size_t i=0;i<r.rows.size();++i) {
        const auto& row=r.rows[i];
        f<<i<<' '<<c.queries[i].frequency_hz<<' '<<c.queries[i].mode<<' '<<row.phase_velocity<<' '
         <<to_string(row.status)<<' '<<row.roots_found<<' '<<std::quoted(row.message)<<'\n';
    }
    f.close();
}
}
