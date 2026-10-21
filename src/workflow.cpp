#include "dispweave/workflow.hpp"
#include "dispweave/io.hpp"
#include "dispweave/lm.hpp"
#include "version.hpp"
#include <toml.hpp>
#include <Eigen/Cholesky>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
namespace dw {
namespace {
Eigen::VectorXd vector(const toml::value& table,const char* key) {
    const auto a=toml::find<std::vector<double>>(table,key);
    return Eigen::Map<const Eigen::VectorXd>(a.data(),a.size());
}
std::ofstream output_file(const std::filesystem::path& path) {
    std::ofstream f(path); if(!f) throw std::runtime_error("Cannot write "+path.string());
    f.exceptions(std::ios::badbit|std::ios::failbit); f<<std::setprecision(17); return f;
}
std::vector<std::vector<double>> read_table(const std::filesystem::path& path) {
    std::ifstream f(path); if(!f) throw std::invalid_argument("Cannot read "+path.string());
    std::vector<std::vector<double>> rows; std::string line;
    while(std::getline(f,line)) {
        line=line.substr(0,line.find('#')); std::istringstream stream(line);
        std::vector<double> row; double value;
        while(stream>>value) { if(!std::isfinite(value)) throw std::invalid_argument("Nonfinite table value"); row.push_back(value); }
        if(!stream.eof()) throw std::invalid_argument("Invalid numeric table: "+path.string());
        if(!row.empty()) rows.push_back(std::move(row));
    }
    if(f.bad() || rows.empty()) throw std::invalid_argument("Empty or unreadable table: "+path.string());
    return rows;
}
DispersionObservations read_observations(const std::filesystem::path& path,const std::filesystem::path& covpath) {
    const auto rows=read_table(path); DispersionObservations data;
    const auto n=static_cast<Eigen::Index>(rows.size()); data.phase_velocity.resize(n); data.covariance=Eigen::MatrixXd::Zero(n,n);
    for(Eigen::Index i=0;i<n;++i) {
        const auto& r=rows[i];
        if(r.size()!=4 || r[2]<0 || r[2]>1023 || r[2]!=std::floor(r[2]) || r[3]<=0)
            throw std::invalid_argument("Data columns: frequency_hz phase_km_s mode sigma_km_s; sigma > 0");
        data.queries.push_back({r[0],static_cast<int>(r[2])}); data.phase_velocity[i]=r[1]; data.covariance(i,i)=r[3]*r[3];
    }
    if(!covpath.empty()) {
        const auto covariance=read_table(covpath);
        if(covariance.size()!=rows.size()) throw std::invalid_argument("Covariance row count mismatch");
        for(Eigen::Index i=0;i<n;++i) {
            if(covariance[i].size()!=rows.size()) throw std::invalid_argument("Covariance column count mismatch");
            for(Eigen::Index j=0;j<n;++j) data.covariance(i,j)=covariance[i][j];
        }
    }
    data.validate(); return data;
}
void save_data(const std::filesystem::path& path,const DispersionObservations& data) {
    auto f=output_file(path/"observations.txt"); f<<"# frequency_hz phase_km_s mode sigma_km_s\n";
    for(Eigen::Index i=0;i<data.phase_velocity.size();++i)
        f<<data.queries[i].frequency_hz<<' '<<data.phase_velocity[i]<<' '<<data.queries[i].mode<<' '<<std::sqrt(data.covariance(i,i))<<'\n';
    auto c=output_file(path/"covariance.txt"); c<<"# covariance in (km/s)^2; authoritative over sigma column\n"<<data.covariance<<'\n';
}
const char* status(EvaluationStatus s) {
    switch(s) {
        case EvaluationStatus::valid: return "valid";
        case EvaluationStatus::outside_support: return "outside_support";
        case EvaluationStatus::forward_failure: return "forward_failure";
        case EvaluationStatus::gradient_failure: return "gradient_failure";
        default: return "numerical_failure";
    }
}
}
int run_inversion(const std::filesystem::path& config_path,const std::filesystem::path& output,
                  bool synthetic,const std::filesystem::path& data_path,const std::filesystem::path& cov_path) {
    const auto begin=std::chrono::steady_clock::now();
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty: "+output.string());
    const auto document=toml::parse(config_path.string());
    const auto table=toml::find(document,"model");
    LayeredModel model{vector(table,"depth"),vector(table,"vs"),vector(table,"vp"),vector(table,"rho")}; model.validate();
    const auto inv=toml::find(document,"inversion");
    FixedVs parameters(model,vector(inv,"lower_vs"),vector(inv,"upper_vs"),toml::find_or<double>(inv,"vp_vs_ratio",0.));
    const auto starts=toml::find<std::vector<std::vector<double>>>(inv,"starts_vs");
    if(starts.empty()) throw std::invalid_argument("At least one starts_vs row required");
    for(const auto& start:starts)
        if(static_cast<Eigen::Index>(start.size())!=model.vs.size()) throw std::invalid_argument("Start dimension mismatch");
    const auto method_name=toml::find_or<std::string>(inv,"sensitivity","energy");
    if(method_name!="energy" && method_name!="implicit") throw std::invalid_argument("Inversion sensitivity must be energy or implicit");
    const auto method=method_name=="energy"?SensitivityMethod::energy:SensitivityMethod::implicit;
    LMOptions options;
    options.max_iterations=toml::find_or<int>(inv,"max_iterations",options.max_iterations);
    options.gradient_tolerance=toml::find_or<double>(inv,"gradient_tolerance",options.gradient_tolerance);
    options.initial_damping=toml::find_or<double>(inv,"initial_damping",options.initial_damping);
    InvLMForward backend; DispersionObservations data;
    int seed=0; bool noise=false;
    if(synthetic) {
        const auto fc=read_forward_config(config_path); data.queries=fc.queries;
        const auto truth=backend.evaluate(model,data.queries);
        if(!truth.all_valid()) throw std::runtime_error("Synthetic truth has missing/unresolved roots; no data generated");
        const auto synth=toml::find(document,"synthetic");
        const double sigma=toml::find<double>(synth,"sigma_km_s");
        const double correlation=toml::find_or<double>(synth,"correlation",0.);
        seed=toml::find_or<int>(synth,"seed",2026); noise=toml::find_or<bool>(synth,"add_noise",true);
        if(!std::isfinite(sigma) || sigma<=0 || !std::isfinite(correlation) || std::abs(correlation)>=1 || seed<0)
            throw std::invalid_argument("Synthetic sigma > 0, |correlation| < 1, seed >= 0 required");
        const auto n=static_cast<Eigen::Index>(truth.rows.size()); data.phase_velocity.resize(n); data.covariance.resize(n,n);
        for(Eigen::Index i=0;i<n;++i) {
            data.phase_velocity[i]=truth.rows[i].phase_velocity;
            for(Eigen::Index j=0;j<n;++j) data.covariance(i,j)=sigma*sigma*std::pow(correlation,std::abs(i-j));
        }
        if(noise) {
            std::mt19937 generator(static_cast<unsigned>(seed)); std::normal_distribution<double> normal;
            Eigen::VectorXd z(n); for(Eigen::Index i=0;i<n;++i) z[i]=normal(generator);
            Eigen::LLT<Eigen::MatrixXd> factor(data.covariance);
            if(factor.info()!=Eigen::Success) throw std::invalid_argument("Synthetic covariance is not positive definite");
            data.phase_velocity+=factor.matrixL()*z;
        }
        data.validate();
    } else data=read_observations(data_path,cov_path);
    DispersionProblem problem(backend,parameters,data,method);
    std::vector<LMResult> runs;
    for(const auto& start:starts) {
        Eigen::VectorXd q=Eigen::Map<const Eigen::VectorXd>(start.data(),start.size()).array().log();
        runs.push_back(invert_lm(problem,q,options));
    }
    int best=-1;
    for(size_t i=0;i<runs.size();++i)
        if(runs[i].converged && (best<0 || runs[i].evaluation.least_squares_cost<runs[best].evaluation.least_squares_cost)) best=static_cast<int>(i);
    // Refuse stale mixed output directories; inputs remain intact, including on failure.
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty: "+output.string());
    std::filesystem::create_directories(output);
    std::filesystem::copy_file(config_path,output/"config.toml"); save_data(output,data);
    auto summary=output_file(output/"runs.txt");
    summary<<"# run converged cost evaluations accepted rejected forward_failures gradient_failures cauchy_steps projected_gradient bound_parameters reason\n";
    // Each start and final iterate is saved even if no run converged.
    for(size_t i=0;i<runs.size();++i) {
        const auto& r=runs[i]; const auto prefix="run_"+std::to_string(i);
        int bounds=0;
        if(parameters.contains(r.q))
            for(Eigen::Index k=0;k<r.q.size();++k)
                if(r.q[k]==parameters.lower()[k] || r.q[k]==parameters.upper()[k]) ++bounds;
        summary<<i<<' '<<r.converged<<' '<<r.evaluation.least_squares_cost<<' '<<r.evaluations<<' '<<r.accepted<<' '<<r.rejected<<' '
               <<r.forward_failures<<' '<<r.gradient_failures<<' '<<r.cauchy_steps<<' '<<r.projected_gradient<<' '<<bounds<<' '<<r.reason<<'\n';
        auto trace=output_file(output/(prefix+"_trace.txt"));
        trace<<"# iteration cost trial_cost damping gain_ratio projected_gradient accepted trial_status\n";
        for(const auto& t:r.trace) trace<<t.iteration<<' '<<t.cost<<' '<<t.trial_cost<<' '<<t.damping<<' '<<t.gain_ratio<<' '<<t.projected_gradient<<' '<<t.accepted<<' '<<status(t.trial_status)<<'\n';
        auto m=output_file(output/(prefix+"_model.txt"));
        m<<"# depth_km reference_vs initial_vs final_vs final_vp rho; reference is truth only for synthetic\n";
        for(Eigen::Index k=0;k<model.vs.size();++k) {
            const double vs=std::exp(r.q[k]); const double vp=parameters.contains(r.q)?parameters.decode(r.q).vp[k]:NAN;
            m<<model.depth[k]<<' '<<model.vs[k]<<' '<<starts[i][k]<<' '<<vs<<' '<<vp<<' '<<model.rho[k]<<'\n';
        }
        if(!r.evaluation.forward.rows.empty())
            write_forward_result(output/(prefix+"_prediction.txt"),{parameters.decode(r.q),data.queries},r.evaluation.forward);
        auto diag=output_file(output/(prefix+"_diagnostic.txt"));
        diag<<"status="<<status(r.evaluation.status)<<"\nmessage="<<r.evaluation.message<<'\n';
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    auto metadata=output_file(output/"metadata.txt");
    metadata<<"version="<<DW_VERSION<<"\ncompiler="<<DW_COMPILER<<"\nbuild="<<DW_BUILD_TYPE<<"\ninvlm="<<DW_INVLM_COMMIT
            <<"\ndependencies=Eigen 3.4.1; toml11 4.4.0; CLI11 2.4.2\nbackend=dunkin\nsensitivity="<<method_name
            <<"\nsynthetic="<<synthetic<<"\nseed="<<seed<<"\nadd_noise="<<noise<<"\nbest_run="<<best
            <<"\nseconds="<<seconds<<"\nparameterization=fixed depths/rho; log Vs; bounded data least squares\n";
    std::cout<<"runs="<<runs.size()<<" best_run="<<best<<" seconds="<<seconds<<" output="<<output<<'\n';
    if(best>=0) std::cout<<"cost="<<runs[best].evaluation.least_squares_cost<<" reason="<<runs[best].reason<<'\n';
    return best>=0?0:4;
}
}
