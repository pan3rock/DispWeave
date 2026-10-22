#include "dispweave/workflow.hpp"
#include "dispweave/io.hpp"
#include "dispweave/lm.hpp"
#include "dispweave/hmc.hpp"
#include "dispweave/nuts.hpp"
#include <type_traits>
#include "dispweave/predictive.hpp"
#include <regex>
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
    options.step_tolerance=toml::find_or<double>(inv,"step_tolerance",options.step_tolerance);
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
    summary<<"# run converged cost evaluations accepted rejected forward_failures gradient_failures box_iterations projected_gradient bound_parameters termination reason\n";
    // Each start and final iterate is saved even if no run converged.
    for(size_t i=0;i<runs.size();++i) {
        const auto& r=runs[i]; const auto prefix="run_"+std::to_string(i);
        int bounds=0;
        if(parameters.contains(r.q))
            for(Eigen::Index k=0;k<r.q.size();++k)
                if(r.q[k]==parameters.lower()[k] || r.q[k]==parameters.upper()[k]) ++bounds;
        summary<<i<<' '<<r.converged<<' '<<r.evaluation.least_squares_cost<<' '<<r.evaluations<<' '<<r.accepted<<' '<<r.rejected<<' '
               <<r.forward_failures<<' '<<r.gradient_failures<<' '<<r.box_iterations<<' '<<r.projected_gradient<<' '<<bounds<<' '<<to_string(r.termination)<<' '<<r.reason<<'\n';
        auto trace=output_file(output/(prefix+"_trace.txt"));
        trace<<"# iteration cost trial_cost damping gain_ratio projected_gradient accepted trial_status predicted_reduction actual_reduction step_norm box_iterations\n";
        for(const auto& t:r.trace) trace<<t.iteration<<' '<<t.cost<<' '<<t.trial_cost<<' '<<t.damping<<' '<<t.gain_ratio<<' '<<t.projected_gradient<<' '<<t.accepted<<' '<<status(t.trial_status)<<' '<<t.predicted_reduction<<' '<<t.actual_reduction<<' '<<t.step_norm<<' '<<t.box_iterations<<'\n';
        auto m=output_file(output/(prefix+"_model.txt"));
        m<<"# depth_km reference_vs initial_vs final_vs final_vp rho; reference is truth only for synthetic\n";
        for(Eigen::Index k=0;k<model.vs.size();++k) {
            const double vs=std::exp(r.q[k]); const double vp=parameters.contains(r.q)?parameters.decode(r.q).vp[k]:NAN;
            m<<model.depth[k]<<' '<<model.vs[k]<<' '<<starts[i][k]<<' '<<vs<<' '<<vp<<' '<<model.rho[k]<<'\n';
        }
        if(!r.evaluation.forward.rows.empty())
            write_forward_result(output/(prefix+"_prediction.txt"),{parameters.decode(r.q),data.queries},r.evaluation.forward);
        auto diag=output_file(output/(prefix+"_diagnostic.txt"));
        diag<<"termination="<<to_string(r.termination)<<"\nreason="<<r.reason<<"\nstatus="<<status(r.evaluation.status)<<"\nmessage="<<r.evaluation.message<<'\n';
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
template<class Options>
int run_sampler(const std::filesystem::path& config_path,const std::filesystem::path& output,
            const std::filesystem::path& data_path,const std::filesystem::path& cov_path) {
    constexpr bool nuts=std::is_same_v<Options,NUTSOptions>;
    const auto begin=std::chrono::steady_clock::now();
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty");
    const auto document=toml::parse(config_path.string());
    const auto table=toml::find(document,"model"),inv=toml::find(document,"inversion"),cfg=toml::find(document,nuts?"nuts":"hmc");
    LayeredModel model{vector(table,"depth"),vector(table,"vs"),vector(table,"vp"),vector(table,"rho")};
    FixedVs parameters(model,vector(inv,"lower_vs"),vector(inv,"upper_vs"),toml::find_or<double>(inv,"vp_vs_ratio",0.));
    const auto method_name=toml::find_or<std::string>(inv,"sensitivity","implicit");
    if(method_name!="energy" && method_name!="implicit") throw std::invalid_argument("Unknown sensitivity");
    InvLMForward backend; const auto data=read_observations(data_path,cov_path);
    DispersionProblem problem(backend,parameters,data,method_name=="energy"?SensitivityMethod::energy:SensitivityMethod::implicit);
    const auto prior_name=toml::find<std::string>(cfg,"prior");
    if(prior_name!="uniform_vs" && prior_name!="uniform_log_vs") throw std::invalid_argument("Sampler prior must be uniform_vs or uniform_log_vs");
    const auto prior=prior_name=="uniform_vs"?
        uniform_prior(vector(inv,"lower_vs"),vector(inv,"upper_vs"),PriorMeasure::physical_vs):
        uniform_prior(parameters.lower(),parameters.upper(),PriorMeasure::log_vs);
    LogisticBox box(parameters.lower(),parameters.upper());
    const auto target=transform_density(dispersion_density(problem,prior),box);
    const auto starts=toml::find<std::vector<std::vector<double>>>(inv,"starts_vs");
    if(starts.empty()) throw std::invalid_argument("At least one sampler start required");
    std::vector<Eigen::VectorXd> initial;
    for(const auto& start:starts) {
        const Eigen::VectorXd q=Eigen::Map<const Eigen::VectorXd>(start.data(),start.size()).array().log();
        initial.push_back(box.inverse(q));
        if(target(initial.back()).status!=EvaluationStatus::valid) throw std::invalid_argument("Invalid sampler initial density/gradient");
    }
    Options options;
    options.warmup=toml::find_or<int>(cfg,"warmup",options.warmup);
    options.draws=toml::find_or<int>(cfg,"draws",options.draws);
    if constexpr(nuts) {
        if(cfg.contains("leapfrog_steps") || cfg.contains("step_jitter")) throw std::invalid_argument("NUTS has no leapfrog_steps or step_jitter option");
        options.max_depth=toml::find_or<int>(cfg,"max_depth",options.max_depth);
    } else {
        options.leapfrog_steps=toml::find_or<int>(cfg,"leapfrog_steps",options.leapfrog_steps);
        options.step_jitter=toml::find_or<double>(cfg,"step_jitter",options.step_jitter);
    }
    options.step_size=toml::find_or<double>(cfg,"step_size",options.step_size);
    options.target_accept=toml::find_or<double>(cfg,"target_accept",options.target_accept);
    options.adapt_step_size=toml::find_or<bool>(cfg,"adapt_step_size",options.adapt_step_size);
    const auto adaptation_name=toml::find_or<std::string>(cfg,"metric_adaptation","none");
    if(adaptation_name!="none" && adaptation_name!="diagonal" && adaptation_name!="dense") throw std::invalid_argument("Unknown metric adaptation");
    options.metric_adaptation=adaptation_name=="none"?MetricAdaptation::none:(adaptation_name=="diagonal"?MetricAdaptation::diagonal:MetricAdaptation::dense);
    options.divergence_threshold=toml::find_or<double>(cfg,"divergence_threshold",options.divergence_threshold);
    const auto seed=toml::find_or<std::int64_t>(cfg,"seed",2026);
    if(seed<0) throw std::invalid_argument("Sampler seed must be nonnegative");
    options.seed=static_cast<std::uint64_t>(seed);
    Eigen::MatrixXd mass=Eigen::MatrixXd::Identity(model.vs.size(),model.vs.size());
    if(cfg.contains("mass")) {
        const auto rows=toml::find<std::vector<std::vector<double>>>(cfg,"mass");
        if(rows.size()!=static_cast<size_t>(mass.rows())) throw std::invalid_argument("Mass row count mismatch");
        for(Eigen::Index i=0;i<mass.rows();++i) {
            if(rows[i].size()!=rows.size()) throw std::invalid_argument("Mass column count mismatch");
            for(Eigen::Index j=0;j<mass.cols();++j) mass(i,j)=rows[i][j];
        }
    }
    const EuclideanMetric metric(mass);
    std::filesystem::create_directories(output);
    std::filesystem::copy_file(config_path,output/"config.toml"); save_data(output,data);
    auto mass_file=output_file(output/"mass.txt");mass_file<<mass<<'\n';
    auto summary=output_file(output/"chains.txt");
    summary<<"# chain draws accepted divergences invalid_evaluations step_size";
    if constexpr(nuts) summary<<" max_depth_reached";
    summary<<'\n';
    bool issues=false;
    for(size_t chain=0;chain<initial.size();++chain) {
        options.chain_id=chain;
        const auto r=[&] {
            if constexpr(nuts) return sample_nuts(target,initial[chain],metric,options);
            else return sample_hmc(target,initial[chain],metric,options);
        }();
        auto final_mass=output_file(output/("chain_"+std::to_string(chain)+"_mass.txt"));final_mass<<r.mass<<'\n';
        auto updates=output_file(output/("chain_"+std::to_string(chain)+"_metric_updates.txt"));
        updates<<"# each block: warmup_iteration samples; then mass matrix\n";
        for(const auto& u:r.metric_updates) updates<<"# "<<u.warmup_iteration<<' '<<u.samples<<'\n'<<u.mass<<'\n';
        auto samples=output_file(output/("chain_"+std::to_string(chain)+"_vs.txt"));
        auto positions=output_file(output/("chain_"+std::to_string(chain)+"_z.txt"));
        samples<<"# Vs km/s; one retained state per row, layer order; excludes warmup\n";
        positions<<"# unconstrained z; excludes warmup\n"<<r.draws<<'\n';
        for(Eigen::Index i=0;i<r.draws.rows();++i) samples<<box.forward(r.draws.row(i).transpose()).q.array().exp().matrix().transpose()<<'\n';
        auto trace=output_file(output/("chain_"+std::to_string(chain)+"_trace.txt"));
        trace<<"# warmup accepted divergent step_size acceptance_probability ";
        if constexpr(nuts) trace<<"max_energy_error status evaluations depth max_depth_reached\n";
        else trace<<"energy_error status evaluations\n";
        int accepted=0,divergences=0,invalid=0,depth_limits=0;
        for(const auto& t:r.transitions) {
            trace<<t.warmup<<' '<<t.accepted<<' '<<t.divergent<<' '<<t.step_size<<' '<<t.acceptance_probability<<' ';
            if constexpr(nuts) trace<<t.max_energy_error;else trace<<t.energy_error;
            trace<<' '<<status(t.status)<<' '<<t.evaluations;
            if constexpr(nuts) {
                trace<<' '<<t.depth<<' '<<t.max_depth_reached;
                if(!t.warmup) depth_limits+=t.max_depth_reached;
            }
            trace<<'\n';
            if(!t.warmup) {accepted+=t.accepted;divergences+=t.divergent;invalid+=t.status!=EvaluationStatus::valid;}
        }
        summary<<chain<<' '<<options.draws<<' '<<accepted<<' '<<divergences<<' '<<invalid<<' '<<r.step_size;
        if constexpr(nuts) summary<<' '<<depth_limits;
        summary<<'\n';
        issues=issues || divergences>0 || invalid>0 || accepted==0 || depth_limits>0;
    }
    auto metadata=output_file(output/"metadata.txt");
    metadata<<"version="<<DW_VERSION<<"\ncompiler="<<DW_COMPILER<<"\nbuild="<<DW_BUILD_TYPE
            <<"\nbackend=invLM Dunkin\ninvlm_commit="<<DW_INVLM_COMMIT
            <<"\ndependencies=Eigen 3.4.1; toml11 4.4.0; CLI11 2.4.2\nseed="<<options.seed
            <<"\nwarmup="<<options.warmup<<"\ndraws="<<options.draws;
    if constexpr(nuts) metadata<<"\nsampler=slice_nuts_hoffman_gelman_2014\nmax_depth="<<options.max_depth;
    else metadata<<"\nsampler=static_hmc\nleapfrog_steps="<<options.leapfrog_steps<<"\nstep_jitter="<<options.step_jitter;
    metadata<<"\ninitial_step_size="<<options.step_size
            <<"\nadapt_step_size="<<options.adapt_step_size<<"\ntarget_accept="<<options.target_accept
            <<"\ndivergence_threshold="<<options.divergence_threshold
            <<"\nmetric_adaptation="<<adaptation_name<<"\nprior="<<prior_name<<"\nsensitivity="<<method_name
            <<"\nparameterization=fixed layer tops/rho; log Vs; logistic box\nposterior=conditional on supplied parameterization\n"
            <<"mass_convention=momentum covariance in z coordinates\n"
            <<"diagnostic_issues="<<issues<<"\nseconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<'\n';
    std::cout<<"chains="<<initial.size()<<" diagnostic_issues="<<issues<<" output="<<output<<'\n';
    return issues?5:0;
}
int run_hmc(const std::filesystem::path& config,const std::filesystem::path& output,
            const std::filesystem::path& data,const std::filesystem::path& covariance) {
    return run_sampler<HMCOptions>(config,output,data,covariance);
}
int run_nuts(const std::filesystem::path& config,const std::filesystem::path& output,
             const std::filesystem::path& data,const std::filesystem::path& covariance) {
    return run_sampler<NUTSOptions>(config,output,data,covariance);
}
int run_predictive(const std::filesystem::path& run,const std::filesystem::path& output,int per_chain,unsigned long long seed) {
    if(per_chain<1) throw std::invalid_argument("Predictive draw count must be positive");
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty");
    const auto document=toml::parse((run/"config.toml").string());
    const auto table=toml::find(document,"model"),inv=toml::find(document,"inversion");
    LayeredModel model{vector(table,"depth"),vector(table,"vs"),vector(table,"vp"),vector(table,"rho")};
    FixedVs parameters(model,vector(inv,"lower_vs"),vector(inv,"upper_vs"),toml::find_or<double>(inv,"vp_vs_ratio",0.));
    const auto data=read_observations(run/"observations.txt",run/"covariance.txt");
    InvLMForward backend;DispersionProblem problem(backend,parameters,data);
    std::vector<std::pair<int,std::filesystem::path>> files;
    const std::regex pattern("chain_([0-9]+)_vs\\.txt");
    for(const auto& entry:std::filesystem::directory_iterator(run)) {
        std::smatch match;const auto name=entry.path().filename().string();
        if(std::regex_match(name,match,pattern)) files.emplace_back(std::stoi(match[1]),entry.path());
    }
    std::sort(files.begin(),files.end());
    if(files.empty()) throw std::invalid_argument("No saved chain Vs samples");
    std::vector<Eigen::VectorXd> samples;std::vector<std::pair<int,size_t>> indices;
    for(const auto& file:files) {
        const auto rows=read_table(file.second);
        if(static_cast<size_t>(per_chain)>rows.size()) throw std::invalid_argument("Predictive count exceeds available draws per chain");
        for(int j=0;j<per_chain;++j) {
            const size_t index=static_cast<size_t>(j)*rows.size()/per_chain;
            if(rows[index].size()!=static_cast<size_t>(model.vs.size())) throw std::invalid_argument("Predictive Vs dimensions mismatch");
            const Eigen::VectorXd q=Eigen::Map<const Eigen::VectorXd>(rows[index].data(),rows[index].size()).array().log();
            if(!q.allFinite()) throw std::invalid_argument("Nonpositive predictive Vs");
            samples.push_back(q);indices.emplace_back(file.first,index);
        }
    }
    Eigen::MatrixXd q(samples.size(),model.vs.size());
    for(size_t i=0;i<samples.size();++i) q.row(i)=samples[i].transpose();
    const auto result=posterior_predictive(problem,q,seed);
    std::filesystem::create_directories(output);save_data(output,data);
    std::filesystem::copy_file(run/"config.toml",output/"config.toml");
    auto latent=output_file(output/"latent.txt");latent<<"# rows=selected draws; columns=observation order; km/s\n"<<result.latent<<'\n';
    auto replicated=output_file(output/"replicated.txt");replicated<<"# latent + correlated Gaussian observation noise; km/s\n"<<result.replicated<<'\n';
    auto selected=output_file(output/"selected_log_vs.txt");selected<<q<<'\n';
    auto status_file=output_file(output/"draws.txt");status_file<<"# row chain source_draw_zero_based status observed_chi2 replicated_chi2\n";
    size_t failures=0;
    for(size_t i=0;i<samples.size();++i) {
        failures+=result.status[i]!=EvaluationStatus::valid;
        status_file<<i<<' '<<indices[i].first<<' '<<indices[i].second<<' '<<status(result.status[i])<<' '
                   <<result.observed_discrepancy[i]<<' '<<result.replicated_discrepancy[i]<<'\n';
    }
    auto metadata=output_file(output/"metadata.txt");
    metadata<<"version="<<DW_VERSION<<"\nseed="<<seed<<"\nsource_run="<<std::filesystem::absolute(run).string()
            <<"\nselected_per_chain="<<per_chain<<"\nselection=floor(j * chain_draws / per_chain); j starts at 0"
            <<"\nfailures="<<failures<<"\nposterior=conditional on supplied parameterization; fixed observed covariance\n";
    std::cout<<"predictive_draws="<<samples.size()<<" failures="<<failures<<'\n';return failures?5:0;
}
}
