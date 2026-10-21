// Optional validation executable; MCMCLib is externally supplied, never a runtime dependency.
#include "dispweave/hmc.hpp"
#include "dispweave/io.hpp"
#include <mcmc.hpp>
#include <toml.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
Eigen::VectorXd vec(const toml::value& t,const char* k) {
    auto x=toml::find<std::vector<double>>(t,k);return Eigen::Map<const Eigen::VectorXd>(x.data(),x.size());
}
std::vector<std::vector<double>> table(const std::string& path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("Cannot read "+path);
    std::vector<std::vector<double>> rows;std::string line;
    while(std::getline(f,line)) {std::istringstream in(line.substr(0,line.find('#')));std::vector<double> r;double x;
        while(in>>x) r.push_back(x);
        if(!in.eof()) throw std::runtime_error("Invalid table");
        if(!r.empty()) rows.push_back(r);
    }
    return rows;
}
int main(int argc,char** argv) {try {
    if(argc!=8) throw std::runtime_error("Usage: hmc_reference CONFIG DATA COVARIANCE OUTPUT STEP SEED DRAWS");
    const auto config=dw::read_forward_config(argv[1]); const auto doc=toml::parse(argv[1]);
    const auto inv=toml::find(doc,"inversion"),hmc=toml::find(doc,"hmc");
    dw::FixedVs parameters(config.model,vec(inv,"lower_vs"),vec(inv,"upper_vs"),toml::find_or<double>(inv,"vp_vs_ratio",0.));
    dw::DispersionObservations data; const auto obs=table(argv[2]),cov=table(argv[3]);
    data.phase_velocity.resize(obs.size());data.covariance.resize(obs.size(),obs.size());
    if(cov.size()!=obs.size()) throw std::runtime_error("Invalid covariance rows");
    for(size_t i=0;i<obs.size();++i) {
        if(obs[i].size()!=4 || cov[i].size()!=obs.size() || obs[i][2]!=std::floor(obs[i][2])) throw std::runtime_error("Invalid observations");
        data.queries.push_back({obs[i][0],static_cast<int>(obs[i][2])});data.phase_velocity[i]=obs[i][1];
        for(size_t j=0;j<obs.size();++j) data.covariance(i,j)=cov[i][j];
    }
    dw::InvLMForward backend;
    const auto method=toml::find<std::string>(inv,"sensitivity");
    if(method!="implicit" && method!="energy") throw std::runtime_error("Invalid sensitivity");
    dw::DispersionProblem problem(backend,parameters,data,method=="implicit"?dw::SensitivityMethod::implicit:dw::SensitivityMethod::energy);
    const auto name=toml::find<std::string>(hmc,"prior");
    if(name!="uniform_vs" && name!="uniform_log_vs") throw std::runtime_error("Invalid prior");
    const auto prior=name=="uniform_vs"?dw::uniform_prior(vec(inv,"lower_vs"),vec(inv,"upper_vs"),dw::PriorMeasure::physical_vs):
        dw::uniform_prior(parameters.lower(),parameters.upper(),dw::PriorMeasure::log_vs);
    const dw::LogisticBox box(parameters.lower(),parameters.upper());
    const auto target=dw::transform_density(dw::dispersion_density(problem,prior),box);
    auto callback=[&](const Eigen::VectorXd& z,Eigen::VectorXd* g,void*) {
        const auto e=target(z);
        if(e.status!=dw::EvaluationStatus::valid) throw std::runtime_error("Reference hit invalid density/gradient; no zero substitution");
        if(g) *g=e.gradient;return e.value;
    };
    mcmc::algo_settings_t settings;settings.rng_seed_value=std::stoull(argv[6]);
    settings.hmc_settings.n_burnin_draws=toml::find<int>(hmc,"warmup");
    settings.hmc_settings.n_keep_draws=std::stoull(argv[7]);
    settings.hmc_settings.step_size=std::stod(argv[5]);
    settings.hmc_settings.n_leap_steps=toml::find<int>(hmc,"leapfrog_steps");
    const auto mass=toml::find<std::vector<std::vector<double>>>(hmc,"mass");
    settings.hmc_settings.precond_mat.resize(parameters.lower().size(),parameters.lower().size());
    if(mass.size()!=static_cast<size_t>(parameters.lower().size())) throw std::runtime_error("Invalid mass");
    for(size_t i=0;i<mass.size();++i) {
        if(mass[i].size()!=mass.size()) throw std::runtime_error("Invalid mass");
        for(size_t j=0;j<mass.size();++j) settings.hmc_settings.precond_mat(i,j)=mass[i][j];
    }
    const dw::EuclideanMetric validate(settings.hmc_settings.precond_mat);
    const auto starts=toml::find<std::vector<std::vector<double>>>(inv,"starts_vs");
    if(starts.empty()) throw std::runtime_error("Missing starts");
    Eigen::VectorXd q=Eigen::Map<const Eigen::VectorXd>(starts[0].data(),starts[0].size()).array().log();
    Eigen::MatrixXd draws;
    if(!mcmc::hmc(box.inverse(q),callback,draws,nullptr,settings)) throw std::runtime_error("Reference sampler failed");
    if(std::filesystem::exists(argv[4])) throw std::runtime_error("Refuse existing output");
    std::ofstream out(argv[4]);out.exceptions(std::ios::failbit|std::ios::badbit);out<<std::setprecision(17);
    out<<"# MCMCLib "<<MCMC_VERSION_MAJOR<<'.'<<MCMC_VERSION_MINOR<<'.'<<MCMC_VERSION_PATCH
       <<" seed="<<settings.rng_seed_value<<" step="<<settings.hmc_settings.step_size
       <<" accepted="<<settings.hmc_settings.n_accept_draws<<"; physical Vs km/s\n";
    for(Eigen::Index i=0;i<draws.rows();++i) out<<box.forward(draws.row(i).transpose()).q.array().exp().matrix().transpose()<<'\n';
    std::cout<<"accepted="<<settings.hmc_settings.n_accept_draws<<" draws="<<draws.rows()<<'\n';return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
