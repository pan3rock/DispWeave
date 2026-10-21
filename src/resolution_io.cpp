#include "dispweave/io.hpp"
#include "dispweave/objective.hpp"
#include "dispweave/resolution.hpp"
#include "version.hpp"
#include <toml.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace dw {
int run_resolution(const std::filesystem::path& config_path,const std::filesystem::path& output) {
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty");
    const auto config=read_forward_config(config_path);
    const auto document=toml::parse(config_path.string()); const auto settings=toml::find(document,"resolution");
    const auto scale_values=toml::find<std::vector<double>>(settings,"scale_log_vs");
    const Eigen::VectorXd scale=Eigen::Map<const Eigen::VectorXd>(scale_values.data(),scale_values.size());
    const double ridge=toml::find<double>(settings,"ridge");
    const double tolerance=toml::find_or<double>(settings,"relative_rank_tolerance",1e-12);
    const auto method_name=toml::find_or<std::string>(settings,"sensitivity","energy");
    if(method_name!="energy" && method_name!="implicit") throw std::invalid_argument("Require energy or implicit sensitivity");
    const auto method=method_name=="energy"?SensitivityMethod::energy:SensitivityMethod::implicit;
    const auto forward=InvLMForward().evaluate(config.model,config.queries,method);
    if(!forward.all_valid()) throw std::runtime_error("Unresolved roots; resolution requires all observation rows");
    if(!forward.all_sensitivities_valid()) throw std::runtime_error("Unavailable kernels; resolution not computed");
    const auto rows=static_cast<Eigen::Index>(config.queries.size()); const auto n=config.model.vs.size();
    Eigen::MatrixXd j(rows,n); Eigen::VectorXd prediction(rows);
    for(Eigen::Index i=0;i<rows;++i) {
        prediction[i]=forward.rows[i].phase_velocity;
        j.row(i)=(forward.rows[i].sensitivity.dc_dvs.array()*config.model.vs.array()).matrix().transpose();
    }
    Eigen::MatrixXd covariance;
    if(settings.contains("covariance") && settings.contains("sigma_km_s"))
        throw std::invalid_argument("Specify covariance OR sigma_km_s, not both");
    if(settings.contains("covariance")) {
        const auto values=toml::find<std::vector<std::vector<double>>>(settings,"covariance");
        if(static_cast<Eigen::Index>(values.size())!=rows) throw std::invalid_argument("Covariance row mismatch");
        covariance.resize(rows,rows);
        for(Eigen::Index i=0;i<rows;++i) {
            if(static_cast<Eigen::Index>(values[i].size())!=rows) throw std::invalid_argument("Covariance column mismatch");
            for(Eigen::Index k=0;k<rows;++k) covariance(i,k)=values[i][k];
        }
    } else {
        const double sigma=toml::find<double>(settings,"sigma_km_s");
        if(!std::isfinite(sigma) || sigma<=0) throw std::invalid_argument("Require positive sigma");
        covariance=Eigen::MatrixXd::Identity(rows,rows)*sigma*sigma;
    }
    GaussianLikelihood likelihood({config.queries,prediction,covariance});
    const auto result=local_resolution(likelihood.whiten_jacobian(j),scale,ridge,tolerance);
    std::filesystem::create_directories(output);
    auto file=[&](const char* name) {
        std::ofstream f(output/name); if(!f) throw std::runtime_error("Cannot write resolution output");
        f.exceptions(std::ios::failbit|std::ios::badbit); f<<std::setprecision(17); return f;
    };
    std::filesystem::copy_file(config_path,output/"config.toml");
    write_forward_result(output/"forward.txt",config,forward);
    auto spectrum=file("spectrum.txt"); spectrum<<"# component singular_value filter_factor\n";
    for(Eigen::Index i=0;i<n;++i) spectrum<<i<<' '<<result.singular_values[i]<<' '<<result.filter_factors[i]<<'\n';
    auto layers=file("layers.txt"); layers<<"# layer top_km bottom_km scale_log_vs resolution_diagonal\n";
    for(Eigen::Index i=0;i<n;++i) layers<<i<<' '<<config.model.depth[i]<<' '<<(i+1<n?config.model.depth[i+1]:INFINITY)<<' '<<scale[i]<<' '<<result.resolution(i,i)<<'\n';
    auto a=file("scaled_jacobian.txt"); a<<result.scaled_jacobian<<'\n';
    auto v=file("right_vectors.txt"); v<<result.right_vectors<<'\n';
    auto r=file("resolution_matrix.txt"); r<<result.resolution<<'\n';
    auto c=file("covariance.txt"); c<<likelihood.data().covariance<<'\n';
    auto meta=file("metadata.txt"); meta<<"version="<<DW_VERSION<<"\ncompiler="<<DW_COMPILER<<"\nbuild="<<DW_BUILD_TYPE
        <<"\ninvlm="<<DW_INVLM_COMMIT<<"\ncoordinates=q=log(Vs); fixed Vp, density and interfaces\nsensitivity="<<method_name
        <<"\nridge="<<ridge<<"\nrelative_rank_tolerance="<<tolerance<<"\nrank_threshold="<<result.rank_threshold
        <<"\nnumerical_rank="<<result.numerical_rank<<"\neffective_dimension="<<result.effective_dimension
        <<"\ninterpretation=local ridge resolution in declared scaled coordinates; not posterior or depth resolution length\n";
    std::cout<<"numerical_rank="<<result.numerical_rank<<" effective_dimension="<<result.effective_dimension<<" output="<<output<<'\n';
    return 0;
}
}
