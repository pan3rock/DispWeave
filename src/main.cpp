#include "dispweave/io.hpp"
#include "version.hpp"
#include <CLI/CLI.hpp>
#include <iostream>
int main(int argc,char** argv) {
    CLI::App app{"DispWeave multimode Rayleigh forward model (M1)"};
    app.set_version_flag("--version",DW_VERSION);
    app.require_subcommand(1,1);
    auto command=app.add_subcommand("forward","Evaluate paired frequency/mode queries using invLM Dunkin");
    std::string config,output;
    command->add_option("--config",config)->required()->check(CLI::ExistingFile);
    command->add_option("--output",output)->required();
    CLI11_PARSE(app,argc,argv);
    try {
        if(std::filesystem::weakly_canonical(config)==std::filesystem::weakly_canonical(output))
            throw std::invalid_argument("Output must not overwrite input configuration");
        const auto c=dw::read_forward_config(config);
        const auto r=dw::InvLMForward().evaluate(c.model,c.queries);
        dw::write_forward_result(output,c,r);
        size_t invalid=0; for(const auto& row:r.rows) if(row.status!=dw::RootStatus::valid) ++invalid;
        std::cout<<"rows="<<r.rows.size()<<" invalid="<<invalid<<" output="<<output<<'\n';
        return r.all_valid()?0:2;
    } catch(const std::exception& e) { std::cerr<<"DispWeave: "<<e.what()<<'\n'; return 1; }
}
