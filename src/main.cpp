#include "dispweave/io.hpp"
#include "version.hpp"
#include <CLI/CLI.hpp>
#include <iostream>
int main(int argc,char** argv) {
    CLI::App app{"DispWeave multimode Rayleigh forward model and sensitivity (M2)"};
    app.set_version_flag("--version",DW_VERSION);
    app.require_subcommand(1,1);
    auto command=app.add_subcommand("forward","Evaluate paired frequency/mode queries using invLM Dunkin");
    std::string config,output,sensitivity="none";
    command->add_option("--config",config)->required()->check(CLI::ExistingFile);
    command->add_option("--output",output)->required();
    command->add_option("--sensitivity",sensitivity)->check(CLI::IsMember({"none","energy","implicit"}));
    CLI11_PARSE(app,argc,argv);
    try {
        if(std::filesystem::weakly_canonical(config)==std::filesystem::weakly_canonical(output))
            throw std::invalid_argument("Output must not overwrite input configuration");
        const auto c=dw::read_forward_config(config);
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
