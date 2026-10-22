#include "dispweave/io.hpp"
#include "version.hpp"
#include "dispweave/workflow.hpp"
#include <CLI/CLI.hpp>
#include <iostream>
int main(int argc,char** argv) {
    CLI::App app{"DispWeave multimode Rayleigh forward model, fixed-layer inversion and sensitivity diagnostics"};
    app.set_version_flag("--version",DW_VERSION);
    app.require_subcommand(1,1);
    auto command=app.add_subcommand("forward","Evaluate paired frequency/mode queries using invLM Dunkin");
    std::string config,output,sensitivity="none";
    command->add_option("--config",config)->required()->check(CLI::ExistingFile);
    command->add_option("--output",output)->required();
    command->add_option("--sensitivity",sensitivity)->check(CLI::IsMember({"none","energy","implicit"}));
    auto ellipticity=app.add_subcommand("ellipticity","Evaluate modal Rayleigh surface displacement ratios; not ambient H/V");
    ellipticity->add_option("--config",config)->required()->check(CLI::ExistingFile);
    ellipticity->add_option("--output",output)->required();
    std::string inv_config,inv_output,data,covariance;
    auto receiver=app.add_subcommand("receiver","Compute incident-P receiver function with declared filtering and deconvolution");
    receiver->add_option("--config",inv_config)->required()->check(CLI::ExistingFile);
    receiver->add_option("--output",inv_output)->required();
    auto synthetic=app.add_subcommand("synthetic","Generate data and run multistart fixed-layer LM");
    auto invert=app.add_subcommand("invert","Invert observed phase velocities using fixed-layer LM");
    for(auto* cmd:{synthetic,invert}) {
        cmd->add_option("--config",inv_config)->required()->check(CLI::ExistingFile);
        cmd->add_option("--output",inv_output)->required();
    }
    invert->add_option("--data",data)->required()->check(CLI::ExistingFile);
    invert->add_option("--covariance",covariance)->check(CLI::ExistingFile);
    std::string predictive_run; int predictive_draws=200; unsigned long long predictive_seed=2026;
    auto predictive=app.add_subcommand("predict","Posterior predictive draws from saved fixed-layer chains");
    auto rj_predictive=app.add_subcommand("rj-predict","Posterior predictions from saved variable-layer chains");
    for(auto* cmd:{predictive,rj_predictive}) {
        cmd->add_option("--run",predictive_run)->required()->check(CLI::ExistingDirectory);
        cmd->add_option("--output",inv_output)->required();
        cmd->add_option("--draws-per-chain",predictive_draws)->check(CLI::PositiveNumber);
        cmd->add_option("--seed",predictive_seed);
    }
    auto hmc=app.add_subcommand("hmc","Sample fixed-layer Vs posterior with static HMC");
    auto nuts=app.add_subcommand("nuts","Sample fixed-layer Vs posterior with slice NUTS");
    auto rjmcmc=app.add_subcommand("rjmcmc","Sample layer count, interfaces and Vs with independent RJMCMC");
    for(auto* cmd:{hmc,nuts,rjmcmc}) {
        cmd->add_option("--config",inv_config)->required()->check(CLI::ExistingFile);
        cmd->add_option("--output",inv_output)->required();
        cmd->add_option("--data",data)->required()->check(CLI::ExistingFile);
        cmd->add_option("--covariance",covariance)->check(CLI::ExistingFile);
    }
    auto resolution=app.add_subcommand("resolution","Local scaled sensitivity and ridge resolution at a reference model");
    resolution->add_option("--config",inv_config)->required()->check(CLI::ExistingFile);
    resolution->add_option("--output",inv_output)->required();
    CLI11_PARSE(app,argc,argv);
    try {
        if(*receiver) return dw::run_receiver(inv_config,inv_output);
        if(*rj_predictive) return dw::run_rj_predictive(predictive_run,inv_output,predictive_draws,predictive_seed);
        if(*predictive) return dw::run_predictive(predictive_run,inv_output,predictive_draws,predictive_seed);
        if(*rjmcmc) return dw::run_rjmcmc(inv_config,inv_output,data,covariance);
        if(*nuts) return dw::run_nuts(inv_config,inv_output,data,covariance);
        if(*hmc) return dw::run_hmc(inv_config,inv_output,data,covariance);
        if(*resolution) return dw::run_resolution(inv_config,inv_output);
        if(*synthetic || *invert) return dw::run_inversion(inv_config,inv_output,static_cast<bool>(*synthetic),data,covariance);
        if(std::filesystem::weakly_canonical(config)==std::filesystem::weakly_canonical(output))
            throw std::invalid_argument("Output must not overwrite input configuration");
        const auto c=dw::read_forward_config(config);
        if(*ellipticity) {
            const auto rows=dw::rayleigh_ellipticity(c.model,c.queries);
            dw::write_ellipticity_result(output,c,rows);
            size_t invalid=0;for(const auto& row:rows) invalid+=row.status!=dw::EllipticityStatus::valid;
            std::cout<<"ellipticity_rows="<<rows.size()<<" invalid="<<invalid<<" output="<<output<<'\n';
            return invalid?2:0;
        }
        const auto method=sensitivity=="none"?dw::SensitivityMethod::none:(sensitivity=="energy"?dw::SensitivityMethod::energy:dw::SensitivityMethod::implicit);
        const auto r=dw::InvLMForward().evaluate(c.model,c.queries,method);
        dw::write_forward_result(output,c,r);
        size_t invalid=0; for(const auto& row:r.rows) if(row.status!=dw::RootStatus::valid) ++invalid;
        size_t gradient_invalid=0;
        if(method!=dw::SensitivityMethod::none)
            for(const auto& row:r.rows) if(row.sensitivity.status!=dw::SensitivityStatus::valid) ++gradient_invalid;
        std::cout<<"rows="<<r.rows.size()<<" invalid="<<invalid<<" gradient_invalid="<<gradient_invalid<<" output="<<output<<'\n';
        if(!r.all_valid()) return 2;
        if(method!=dw::SensitivityMethod::none && !r.all_sensitivities_valid()) return 3;
        return 0;
    } catch(const std::exception& e) { std::cerr<<"DispWeave: "<<e.what()<<'\n'; return 1; }
}
