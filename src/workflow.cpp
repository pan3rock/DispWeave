#include "dispweave/workflow.hpp"
#include "dispweave/io.hpp"
#include "dispweave/lm.hpp"
#include "dispweave/hmc.hpp"
#include "dispweave/nuts.hpp"
#include "dispweave/rj_dispersion.hpp"
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
VsBounds read_bounds(const toml::value& inv,const Eigen::VectorXd& reference) {
    const bool window=inv.contains("vs_min") || inv.contains("vs_max") || inv.contains("vs_width");
    const bool explicit_bounds=inv.contains("lower_vs") || inv.contains("upper_vs");
    if(window && explicit_bounds)
        throw std::invalid_argument("Specify vs_min/vs_max/vs_width OR lower_vs/upper_vs, not both");
    if(window) {
        if(!inv.contains("vs_min") || !inv.contains("vs_max") || !inv.contains("vs_width"))
            throw std::invalid_argument("Reference-window bounds require all of vs_min, vs_max and vs_width");
        return reference_vs_bounds(reference,toml::find<double>(inv,"vs_min"),
                                   toml::find<double>(inv,"vs_max"),toml::find<double>(inv,"vs_width"));
    }
    if(!inv.contains("lower_vs") || !inv.contains("upper_vs"))
        throw std::invalid_argument("Provide vs_min/vs_max/vs_width or both lower_vs and upper_vs");
    return {vector(inv,"lower_vs"),vector(inv,"upper_vs")};
}
ElasticRelation read_relation(const toml::value& inv) {
    ElasticRelation r;
    const double legacy_ratio=toml::find_or<double>(inv,"vp_vs_ratio",0.);
    r.kind=toml::find_or<std::string>(inv,"vs2model",legacy_ratio!=0?"constant_ratio":"fixvprho");
    r.vp_vs_ratio=toml::find_or<double>(inv,"vp_vs_ratio",r.vp_vs_ratio);
    if((r.kind=="fixvprho" && legacy_ratio!=0) || (r.kind=="brocher05" && inv.contains("vp_vs_ratio")))
        throw std::invalid_argument("vp_vs_ratio conflicts with selected vs2model");
    if(r.kind!="gardner" && (inv.contains("gardner_alpha") || inv.contains("gardner_beta")))
        throw std::invalid_argument("Gardner coefficients require vs2model=gardner");
    if(r.kind!="constant_rho" && inv.contains("constant_density"))
        throw std::invalid_argument("constant_density requires vs2model=constant_rho");
    r.gardner_alpha=toml::find_or<double>(inv,"gardner_alpha",r.gardner_alpha);
    r.gardner_beta=toml::find_or<double>(inv,"gardner_beta",r.gardner_beta);
    r.density=toml::find_or<double>(inv,"constant_density",r.density);
    return r;
}
void save_relation(std::ostream& out,const ElasticRelation& r) {
    out<<"vs2model="<<r.kind<<'\n';
    if(r.kind!="fixvprho" && r.kind!="brocher05") out<<"vp_vs_ratio="<<r.vp_vs_ratio<<'\n';
    if(r.kind=="gardner") out<<"gardner_alpha="<<r.gardner_alpha<<"\ngardner_beta="<<r.gardner_beta<<'\n';
    if(r.kind=="constant_rho") out<<"constant_density="<<r.density<<'\n';
}
VsRegularization read_regularization(const toml::value& inv,const LayeredModel& model) {
    if(!inv.contains("regularization")) return {model.depth,model.vs,"none",0};
    const auto r=toml::find(inv,"regularization");
    return {model.depth,model.vs,toml::find<std::string>(r,"type"),toml::find<double>(r,"lambda")};
}
RJPrior read_rj_prior(const toml::value& p) {
    RJPrior prior;prior.min_layers=toml::find<int>(p,"min_layers");prior.max_layers=toml::find<int>(p,"max_layers");
    prior.depth_limit=toml::find<double>(p,"depth_limit");prior.min_spacing=toml::find<double>(p,"min_spacing");
    prior.vs_lower=toml::find<double>(p,"vs_lower");prior.vs_upper=toml::find<double>(p,"vs_upper");
    prior.layer_weights=vector(p,"layer_weights");prior.validate();
    return prior;
}
std::ofstream output_file(const std::filesystem::path& path) {
    std::ofstream f(path); if(!f) throw std::runtime_error("Cannot write "+path.string());
    f.exceptions(std::ios::badbit|std::ios::failbit); f<<std::setprecision(17); return f;
}
void save_bounds(const std::filesystem::path& output,std::ostream& metadata,const toml::value& inv,
                 const FixedVs& parameters,const VsBounds& bounds,const std::vector<int>& fixed={}) {
    const auto& m=parameters.reference_model();
    const auto n=m.vs.size();std::vector<bool> is_fixed(n,false);
    for(int i:fixed) is_fixed.at(i)=true;
    auto f=output_file(output/"parameter_bounds.txt");
    f<<"# layer top_km reference_vs lower_vs upper_vs fixed halfspace; km/s; fixed means Vs=reference_vs\n";
    for(Eigen::Index i=0;i<n;++i)
        f<<i<<' '<<m.depth[i]<<' '<<m.vs[i]<<' '<<bounds.lower[i]<<' '<<bounds.upper[i]<<' '
         <<is_fixed[i]<<' '<<(i==n-1)<<'\n';
    metadata<<"bounds_mode="<<(inv.contains("vs_width")?"global_intersection_reference_window":"explicit_per_layer")<<'\n';
    if(inv.contains("vs_width"))
        metadata<<"vs_min="<<toml::find<double>(inv,"vs_min")<<"\nvs_max="<<toml::find<double>(inv,"vs_max")
                <<"\nvs_width="<<toml::find<double>(inv,"vs_width")<<'\n';
    metadata<<"halfspace_top_km="<<m.depth[n-1]<<"\nhalfspace_vs="<<(is_fixed[n-1]?"fixed":"free")
            <<"\nhalfspace_observation_floor=none\n";
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
    const auto bounds=read_bounds(inv,model.vs);
    FixedVs parameters(model,bounds.lower,bounds.upper,read_relation(inv));
    model=parameters.reference_model();
    const auto regularization=read_regularization(inv,model);
    const auto starts=toml::find<std::vector<std::vector<double>>>(inv,"starts_vs");
    if(starts.empty()) throw std::invalid_argument("At least one starts_vs row required");
    for(const auto& start:starts)
        if(static_cast<Eigen::Index>(start.size())!=model.vs.size()) throw std::invalid_argument("Start dimension mismatch");
    const auto fixed=toml::find_or<std::vector<int>>(inv,"fixed_vs_layers",{});
    std::vector<bool> is_fixed(model.vs.size(),false);
    for(int layer:fixed) {
        if(layer<0 || layer>=model.vs.size() || is_fixed[layer]) throw std::invalid_argument("Invalid/duplicate fixed Vs layer");
        is_fixed[layer]=true;
        for(const auto& start:starts)
            if(start[layer]!=model.vs[layer]) throw std::invalid_argument("Fixed Vs starts must match model.vs");
    }
    if(fixed.size()==static_cast<size_t>(model.vs.size())) throw std::invalid_argument("LM requires at least one free Vs layer");
    const auto method_name=toml::find_or<std::string>(inv,"sensitivity","energy");
    if(method_name!="energy" && method_name!="implicit") throw std::invalid_argument("Inversion sensitivity must be energy or implicit");
    const auto method=method_name=="energy"?SensitivityMethod::energy:SensitivityMethod::implicit;
    LMOptions options;
    options.max_iterations=toml::find_or<int>(inv,"max_iterations",options.max_iterations);
    options.gradient_tolerance=toml::find_or<double>(inv,"gradient_tolerance",options.gradient_tolerance);
    options.step_tolerance=toml::find_or<double>(inv,"step_tolerance",options.step_tolerance);
    options.initial_damping=toml::find_or<double>(inv,"initial_damping",options.initial_damping);
    // Extra root brackets can change the final rounding of an otherwise
    // identical phase. Refine LM evaluations to roundoff, as in joint LM,
    // so cancellation in tiny trial reductions does not cause false stalls.
    InvLMForward backend(SecularMethod::dunkin,RootAccuracy::refined); DispersionObservations data;
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
        runs.push_back(invert_lm(problem,q,options,fixed,&regularization));
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
    auto costs=output_file(output/"costs.txt");costs<<"# run data_cost regularization_cost total_cost\n";
    // Each start and final iterate is saved even if no run converged.
    for(size_t i=0;i<runs.size();++i) {
        const auto& r=runs[i]; const auto prefix="run_"+std::to_string(i);
        costs<<i<<' '<<r.evaluation.data_cost<<' '<<r.evaluation.regularization_cost<<' '<<r.evaluation.least_squares_cost<<'\n';
        int bounds=0;
        if(parameters.contains(r.q))
            for(Eigen::Index k=0;k<r.q.size();++k)
                if(!is_fixed[k] && (r.q[k]==parameters.lower()[k] || r.q[k]==parameters.upper()[k])) ++bounds;
        summary<<i<<' '<<r.converged<<' '<<r.evaluation.least_squares_cost<<' '<<r.evaluations<<' '<<r.accepted<<' '<<r.rejected<<' '
               <<r.forward_failures<<' '<<r.gradient_failures<<' '<<r.box_iterations<<' '<<r.projected_gradient<<' '<<bounds<<' '<<to_string(r.termination)<<' '<<r.reason<<'\n';
        auto trace=output_file(output/(prefix+"_trace.txt"));
        trace<<"# iteration cost trial_cost damping gain_ratio projected_gradient accepted trial_status predicted_reduction actual_reduction step_norm box_iterations\n";
        for(const auto& t:r.trace) trace<<t.iteration<<' '<<t.cost<<' '<<t.trial_cost<<' '<<t.damping<<' '<<t.gain_ratio<<' '<<t.projected_gradient<<' '<<t.accepted<<' '<<status(t.trial_status)<<' '<<t.predicted_reduction<<' '<<t.actual_reduction<<' '<<t.step_norm<<' '<<t.box_iterations<<'\n';
        auto cost_trace=output_file(output/(prefix+"_cost_trace.txt"));
        cost_trace<<"# iteration data_cost regularization_cost trial_data_cost trial_regularization_cost\n";
        for(const auto& t:r.trace) cost_trace<<t.iteration<<' '<<t.data_cost<<' '<<t.regularization_cost<<' '<<t.trial_data_cost<<' '<<t.trial_regularization_cost<<'\n';
        auto m=output_file(output/(prefix+"_model.txt"));
        m<<"# depth_km reference_vs initial_vs final_vs final_vp rho; reference is truth only for synthetic\n";
        const bool valid_model=parameters.contains(r.q);
        const auto decoded=valid_model?parameters.decode(r.q):model;
        for(Eigen::Index k=0;k<model.vs.size();++k) {
            const double vs=std::exp(r.q[k]);
            const double vp=valid_model?decoded.vp[k]:NAN,rho=valid_model?decoded.rho[k]:NAN;
            m<<model.depth[k]<<' '<<model.vs[k]<<' '<<starts[i][k]<<' '<<vs<<' '<<vp<<' '<<rho<<'\n';
        }
        if(!r.evaluation.forward.rows.empty())
            write_forward_result(output/(prefix+"_prediction.txt"),{decoded,data.queries},r.evaluation.forward);
        auto diag=output_file(output/(prefix+"_diagnostic.txt"));
        diag<<"termination="<<to_string(r.termination)<<"\nreason="<<r.reason<<"\nstatus="<<status(r.evaluation.status)<<"\nmessage="<<r.evaluation.message<<'\n';
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    auto metadata=output_file(output/"metadata.txt");
    metadata<<"version="<<DW_VERSION<<"\ncompiler="<<DW_COMPILER<<"\nbuild="<<DW_BUILD_TYPE<<"\ninvlm="<<DW_INVLM_COMMIT
            <<"\ndependencies=Eigen 3.4.1; toml11 4.4.0; CLI11 2.4.2\nbackend=dunkin\nsensitivity="<<method_name
            <<"\nroot_accuracy=refined\nsynthetic="<<synthetic<<"\nseed="<<seed<<"\nadd_noise="<<noise<<"\nbest_run="<<best
            <<"\nseconds="<<seconds<<"\nparameterization=fixed depths; log Vs; bounded penalized least squares\n";
    save_relation(metadata,parameters.relation());
    save_bounds(output,metadata,inv,parameters,bounds,fixed);
    metadata<<"regularization_type="<<regularization.type()<<"\nregularization_lambda="<<regularization.weight()
            <<"\nobjective=data_cost+regularization_cost; Gaussian log likelihood excludes regularization\n";
    metadata<<"free_vs_parameters="<<model.vs.size()-fixed.size()<<"\nfixed_vs_layers=";
    for(size_t i=0;i<fixed.size();++i) metadata<<(i?",":"")<<fixed[i];
    metadata<<"\nfixed_vs_values=model.vs; starts must match\nprojected_gradient_coordinates=free log Vs, scaled by support widths\n";
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
    if(!toml::find_or<std::vector<int>>(inv,"fixed_vs_layers",{}).empty())
        throw std::invalid_argument("fixed_vs_layers currently supported only by invert/synthetic");
    LayeredModel model{vector(table,"depth"),vector(table,"vs"),vector(table,"vp"),vector(table,"rho")};
    const auto bounds=read_bounds(inv,model.vs);
    FixedVs parameters(model,bounds.lower,bounds.upper,read_relation(inv));
    model=parameters.reference_model();
    if(read_regularization(inv,model).active())
        throw std::invalid_argument("LM regularization is not a sampler prior; declare the posterior prior explicitly");
    const auto method_name=toml::find_or<std::string>(inv,"sensitivity","implicit");
    if(method_name!="energy" && method_name!="implicit") throw std::invalid_argument("Unknown sensitivity");
    InvLMForward backend; const auto data=read_observations(data_path,cov_path);
    const auto support_name=toml::find_or<std::string>(cfg,"mode_support","strict_forward");
    if(support_name!="strict_forward" && support_name!="trapped_modes") throw std::invalid_argument("Unknown mode_support policy");
    DispersionProblem problem(backend,parameters,data,method_name=="energy"?SensitivityMethod::energy:SensitivityMethod::implicit,
        support_name=="trapped_modes"?ModeSupportPolicy::trapped_modes:ModeSupportPolicy::strict_forward);
    const auto prior_name=toml::find<std::string>(cfg,"prior");
    if(prior_name!="uniform_vs" && prior_name!="uniform_log_vs" && prior_name!="reference_gaussian_vs")
        throw std::invalid_argument("Sampler prior must be uniform_vs, uniform_log_vs or reference_gaussian_vs");
    if(prior_name!="reference_gaussian_vs" && cfg.contains("prior_sigma_vs"))
        throw std::invalid_argument("prior_sigma_vs requires reference_gaussian_vs");
    Eigen::VectorXd prior_sigma;
    const auto prior=[&]() {
        if(prior_name=="reference_gaussian_vs") {
            prior_sigma=vector(cfg,"prior_sigma_vs");
            return bounded_gaussian_prior(model.vs,prior_sigma,bounds.lower,bounds.upper,PriorMeasure::physical_vs);
        }
        return prior_name=="uniform_vs"?uniform_prior(bounds.lower,bounds.upper,PriorMeasure::physical_vs):
                                       uniform_prior(parameters.lower(),parameters.upper(),PriorMeasure::log_vs);
    }();
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
        const auto kernel=toml::find_or<std::string>(cfg,"kernel","slice");
        if(kernel!="slice" && kernel!="multinomial") throw std::invalid_argument("Unknown NUTS kernel");
        options.kernel=kernel=="slice"?NUTSKernel::slice:NUTSKernel::multinomial;
        options.find_initial_step_size=toml::find_or<bool>(cfg,"find_initial_step_size",false);
        options.metric_shrinkage=toml::find_or<double>(cfg,"metric_shrinkage",0.);
        options.conservative_warmup=toml::find_or<bool>(cfg,"conservative_warmup",false);
        options.conservative_metric_restart=toml::find_or<bool>(cfg,"conservative_metric_restart",false);
        options.metric_max_stiffness_ratio=toml::find_or<double>(cfg,"metric_max_stiffness_ratio",0.);
        if(options.metric_max_stiffness_ratio>0 && prior_name!="reference_gaussian_vs")
            throw std::invalid_argument("Local Gauss-Newton metric guard requires reference_gaussian_vs");
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
    const auto chain_offset=toml::find_or<std::int64_t>(cfg,"chain_id_offset",0);
    if(chain_offset<0) throw std::invalid_argument("Sampler chain_id_offset must be nonnegative");
    const bool record_failures=toml::find_or<bool>(cfg,"record_failures",false);
    const auto initial_metric_name=toml::find_or<std::string>(cfg,"initial_metric","supplied");
    if(initial_metric_name!="supplied" && initial_metric_name!="start_gauss_newton") throw std::invalid_argument("Unknown initial_metric");
    if(initial_metric_name=="start_gauss_newton" && (cfg.contains("mass") || prior_name!="reference_gaussian_vs"))
        throw std::invalid_argument("start_gauss_newton needs reference Gaussian prior and no supplied mass");
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
    if(prior_name=="reference_gaussian_vs") {
        auto f=output_file(output/"prior_parameters.txt");
        f<<"# layer mean_vs sigma_vs lower_vs upper_vs; independent normalized bounded Gaussian in physical Vs, km/s\n";
        for(Eigen::Index i=0;i<model.vs.size();++i)
            f<<i<<' '<<model.vs[i]<<' '<<prior_sigma[i]<<' '<<bounds.lower[i]<<' '<<bounds.upper[i]<<'\n';
    }
    auto summary=output_file(output/"chains.txt");
    summary<<"# chain draws accepted divergences invalid_evaluations step_size";
    if constexpr(nuts) summary<<" max_depth_reached";
    summary<<'\n';
    bool issues=false;
    for(size_t chain=0;chain<initial.size();++chain) {
        options.chain_id=static_cast<std::uint64_t>(chain_offset)+chain;
        const auto local_gauss_newton=[&](const Eigen::VectorXd& z) -> Eigen::MatrixXd {
            const auto t=box.forward(z);const auto e=problem.evaluate(t.q,true);
            if(e.status!=EvaluationStatus::valid) throw std::runtime_error("Unresolved Gauss-Newton metric Jacobian");
            const Eigen::MatrixXd j=e.jacobian*t.derivative.asDiagonal();
            Eigen::MatrixXd m=j.transpose()*j;
            m.diagonal().array()+=(t.q.array().exp()*t.derivative.array()/prior_sigma.array()).square();
            return m;
        };
        const EuclideanMetric chain_metric=[&] {
            if(initial_metric_name=="supplied") return metric;
            return EuclideanMetric(local_gauss_newton(initial[chain]));
        }();
        auto initial_mass_file=output_file(output/("chain_"+std::to_string(chain)+"_initial_mass.txt"));
        initial_mass_file<<chain_metric.mass()<<'\n';
        std::ofstream failed;
        std::ofstream progress;
        std::ofstream decisions;int guard_evaluations=0;
        if constexpr(nuts) if(options.metric_max_stiffness_ratio>0) {
            decisions=output_file(output/("chain_"+std::to_string(chain)+"_metric_decisions.txt"));
            decisions<<"# warmup_iteration accepted old_stiffness candidate_stiffness ratio; one additional Jacobian per decision\n";
            options.local_curvature=[&](const Eigen::VectorXd& z) {++guard_evaluations;return local_gauss_newton(z);};
            options.metric_progress=[&](const MetricDecision& d) {
                decisions<<d.warmup_iteration<<' '<<d.accepted<<' '<<d.old_stiffness<<' '<<d.candidate_stiffness
                         <<' '<<d.candidate_stiffness/d.old_stiffness<<std::endl;
            };
        }
        if constexpr(nuts) if(toml::find_or<bool>(cfg,"record_progress",false)) {
            progress=output_file(output/("chain_"+std::to_string(chain)+"_progress.txt"));
            progress<<"# iteration warmup status divergent depth evaluations step acceptance z[all parameters]\n";
            options.progress=[&](int i,const Eigen::VectorXd& z,const NUTSTransition& t) {
                progress<<i<<' '<<t.warmup<<' '<<status(t.status)<<' '<<t.divergent<<' '<<t.depth<<' '
                        <<t.evaluations<<' '<<t.step_size<<' '<<t.acceptance_probability<<' '<<z.transpose()<<std::endl;
            };
        }
        if(record_failures) {
            failed=output_file(output/("chain_"+std::to_string(chain)+"_failed_evaluations.txt"));
            failed<<"# target_call status z[all parameters]; zero-based calls including initialization\n";
        }
        std::uint64_t evaluation_index=0;
        const LogDensity recorded_target=[&](const Eigen::VectorXd& z) {
            auto e=target(z);
            if(record_failures && e.status!=EvaluationStatus::valid)
                failed<<evaluation_index<<' '<<status(e.status)<<' '<<z.transpose()<<std::endl;
            ++evaluation_index;return e;
        };
        const auto r=[&] {
            if constexpr(nuts) return sample_nuts(recorded_target,initial[chain],chain_metric,options);
            else return sample_hmc(recorded_target,initial[chain],chain_metric,options);
        }();
        if constexpr(nuts) {
            auto searches=output_file(output/("chain_"+std::to_string(chain)+"_step_size_searches.txt"));
            searches<<"# warmup_iteration evaluations step_size last_status; 0=before first transition\n";
            int probes=0;std::uint64_t transitions=0;
            for(const auto& s:r.step_size_searches) {
                searches<<s.warmup_iteration<<' '<<s.evaluations<<' '<<s.step_size<<' '<<status(s.last_status)<<'\n';probes+=s.evaluations;
            }
            for(const auto& t:r.transitions) transitions+=t.evaluations;
            auto cost=output_file(output/("chain_"+std::to_string(chain)+"_cost.txt"));
            cost<<"target_evaluations="<<evaluation_index<<"\ntransition_evaluations="<<transitions
                <<"\nstep_search_evaluations="<<probes<<"\ninitial_evaluations=1\npreflight_evaluations=1\nmetric_evaluations="
                <<(initial_metric_name=="start_gauss_newton"?1:0)<<'\n';
            if(options.metric_max_stiffness_ratio>0) cost<<"guard_evaluations="<<guard_evaluations<<'\n';
        }
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
            if(!t.warmup) {accepted+=t.accepted;divergences+=t.divergent;invalid+=t.status!=EvaluationStatus::valid && t.status!=EvaluationStatus::outside_support;}
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
            <<"\nchain_id_offset="<<chain_offset<<"\nrecord_failures="<<record_failures
            <<"\nmode_support="<<support_name
            <<"\ninitial_metric="<<initial_metric_name
            <<"\nwarmup="<<options.warmup<<"\ndraws="<<options.draws;
    if constexpr(nuts) metadata<<"\nsampler="<<(options.kernel==NUTSKernel::slice?"slice_nuts_hoffman_gelman_2014":"multinomial_nuts")
        <<"\nfind_initial_step_size="<<options.find_initial_step_size<<"\nmetric_shrinkage="<<options.metric_shrinkage<<"\nmax_depth="<<options.max_depth
        <<"\nconservative_warmup="<<options.conservative_warmup
        <<"\nconservative_metric_restart="<<options.conservative_metric_restart
        <<"\nmetric_max_stiffness_ratio="<<options.metric_max_stiffness_ratio;
    else metadata<<"\nsampler=static_hmc\nleapfrog_steps="<<options.leapfrog_steps<<"\nstep_jitter="<<options.step_jitter;
    metadata<<"\ninitial_step_size="<<options.step_size
            <<"\nadapt_step_size="<<options.adapt_step_size<<"\ntarget_accept="<<options.target_accept
            <<"\ndivergence_threshold="<<options.divergence_threshold
            <<"\nmetric_adaptation="<<adaptation_name<<"\nprior="<<prior_name<<"\nsensitivity="<<method_name
            <<"\nparameterization=fixed layer tops; log Vs; logistic box\nposterior=conditional on supplied parameterization\n"
            <<"mass_convention=momentum covariance in z coordinates\n"
            <<"diagnostic_issues="<<issues<<"\nseconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<'\n';
    save_relation(metadata,parameters.relation());
    if(prior_name=="reference_gaussian_vs")
        metadata<<"prior_measure=physical_vs\nprior_center=model.vs\nprior_scale=prior_sigma_vs_km_s\n"
                <<"prior_normalization=independent_Gaussians_conditioned_on_parameter_bounds\n";
    save_bounds(output,metadata,inv,parameters,bounds);
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
int run_rjmcmc(const std::filesystem::path& config_path,const std::filesystem::path& output,
              const std::filesystem::path& data_path,const std::filesystem::path& cov_path) {
    const auto begin=std::chrono::steady_clock::now();
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty");
    const auto document=toml::parse(config_path.string());
    const auto p=toml::find(document,"rj_prior"),physics=toml::find(document,"rj_physics"),cfg=toml::find(document,"rjmcmc");
    const auto prior=read_rj_prior(p);
    const double ratio=toml::find<double>(physics,"vp_vs_ratio"),rho=toml::find<double>(physics,"density");
    RJOptions options;
    options.warmup=toml::find_or<int>(cfg,"warmup",options.warmup);options.draws=toml::find_or<int>(cfg,"draws",options.draws);
    options.vs_step=toml::find_or<double>(cfg,"vs_step",options.vs_step);
    options.interface_step=toml::find_or<double>(cfg,"interface_step",options.interface_step);
    const auto seed=toml::find_or<std::int64_t>(cfg,"seed",2026);
    if(seed<0) throw std::invalid_argument("RJ seed must be nonnegative");
    options.seed=static_cast<std::uint64_t>(seed);
    InvLMForward backend;const auto data=read_observations(data_path,cov_path);
    RJDispersionProblem problem(backend,prior,data,ratio,rho);
    const RJLikelihood target=[&](const RJState& state) {return problem.evaluate(state).likelihood;};
    std::vector<RJState> starts;
    const auto start_table=toml::find(document,"rj_starts");
    for(const auto& item:start_table.as_array()) {
        starts.push_back({vector(item,"interfaces"),vector(item,"vs")});
        if(target(starts.back()).status!=EvaluationStatus::valid) throw std::invalid_argument("Invalid RJ initial model or forward prediction");
    }
    if(starts.empty()) throw std::invalid_argument("Need at least one RJ start");
    std::filesystem::create_directories(output);std::filesystem::copy_file(config_path,output/"config.toml");save_data(output,data);
    auto summary=output_file(output/"chains.txt");summary<<"# chain draws accepted dimension_changes forward_failures numerical_failures\n";
    bool issues=false;
    for(size_t chain=0;chain<starts.size();++chain) {
        auto failed_models=output_file(output/("chain_"+std::to_string(chain)+"_failed_models.txt"));
        auto failed_rows=output_file(output/("chain_"+std::to_string(chain)+"_failed_rows.txt"));
        failed_models<<"# evaluation status K interfaces[K-1] vs[K]; includes warmup; initial evaluation=0\n";
        failed_rows<<"# evaluation query_index root_status roots_found phase_velocity\n";
        int evaluation_index=0;
        const RJLikelihood recorded_target=[&](const RJState& state) {
            const int id=evaluation_index++;const auto e=problem.evaluate(state);
            if(e.likelihood.status!=EvaluationStatus::valid) {
                failed_models<<id<<' '<<status(e.likelihood.status)<<' '<<state.vs.size();
                for(double z:state.interfaces) failed_models<<' '<<z;
                for(double vs:state.vs) failed_models<<' '<<vs;
                failed_models<<'\n';
                for(size_t j=0;j<e.forward.rows.size();++j) {
                    const auto& row=e.forward.rows[j];
                    failed_rows<<id<<' '<<j<<' '<<to_string(row.status)<<' '<<row.roots_found<<' '<<row.phase_velocity<<'\n';
                }
            }
            return e.likelihood;
        };
        options.chain_id=chain;const auto result=sample_rjmcmc(recorded_target,prior,starts[chain],options);
        auto models=output_file(output/("chain_"+std::to_string(chain)+"_models.txt"));
        models<<"# draw K log_likelihood log_prior interfaces[K-1] vs[K]; km and km/s; excludes warmup\n";
        for(size_t i=0;i<result.draws.size();++i) {
            const auto& d=result.draws[i];models<<i<<' '<<d.state.vs.size()<<' '<<d.log_likelihood<<' '<<d.log_prior;
            for(double z:d.state.interfaces) models<<' '<<z;
            for(double vs:d.state.vs) models<<' '<<vs;
            models<<'\n';
        }
        auto trace=output_file(output/("chain_"+std::to_string(chain)+"_trace.txt"));
        trace<<"# warmup move proposal_status evaluated evaluation_status accepted layers_before layers_after log_accept_ratio evaluation_index\n";
        int accepted=0,changes=0,forward_failures=0,numerical_failures=0,trace_evaluation=0;
        for(const auto& t:result.transitions) {
            const char* move=t.move==RJMove::birth?"birth":t.move==RJMove::death?"death":t.move==RJMove::interface?"interface":"velocity";
            const char* proposal=t.proposal_status==RJProposalStatus::valid?"valid":t.proposal_status==RJProposalStatus::unavailable?"unavailable":"outside_support";
            trace<<t.warmup<<' '<<move<<' '<<proposal<<' '<<t.evaluated<<' '<<(t.evaluated?status(t.evaluation_status):"not_evaluated")<<' '<<t.accepted<<' '
                 <<t.layers_before<<' '<<t.layers_after<<' '<<t.log_accept_ratio<<' '<<(t.evaluated?++trace_evaluation:-1)<<'\n';
            if(!t.warmup) {
                accepted+=t.accepted;changes+=t.layers_before!=t.layers_after;
                forward_failures+=t.evaluated && t.evaluation_status==EvaluationStatus::forward_failure;
                numerical_failures+=t.evaluated && t.evaluation_status==EvaluationStatus::numerical_failure;
            }
        }
        summary<<chain<<' '<<options.draws<<' '<<accepted<<' '<<changes<<' '<<forward_failures<<' '<<numerical_failures<<'\n';
        issues=issues || forward_failures>0 || numerical_failures>0 || accepted==0;
    }
    auto metadata=output_file(output/"metadata.txt");
    metadata<<"version="<<DW_VERSION<<"\ncompiler="<<DW_COMPILER<<"\nbuild="<<DW_BUILD_TYPE
            <<"\nsampler=independent_birth_death_rjmcmc\nbackend=invLM Dunkin\ninvlm_commit="<<DW_INVLM_COMMIT
            <<"\nseed="<<options.seed<<"\nwarmup="<<options.warmup<<"\ndraws="<<options.draws
            <<"\nvs_step="<<options.vs_step<<"\ninterface_step="<<options.interface_step
            <<"\nvp_vs_ratio="<<ratio<<"\ndensity="<<rho
            <<"\nstate_measure=physical Vs and ordered internal interface depths\nlayer_count=includes halfspace\n"
            <<"depth_limit=interface prior support; not halfspace bottom\nmove_probabilities=.25 .25 .25 .25\n"
            <<"posterior=transdimensional, conditional on declared support and fixed Vp/Vs and density\n"
            <<"diagnostic_issues="<<issues<<"\nseconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<'\n';
    std::cout<<"RJ chains="<<starts.size()<<" diagnostic_issues="<<issues<<" output="<<output<<'\n';return issues?5:0;
}
int run_rj_predictive(const std::filesystem::path& run,const std::filesystem::path& output,int per_chain,unsigned long long seed) {
    if(per_chain<1) throw std::invalid_argument("Predictive draw count must be positive");
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty");
    const auto document=toml::parse((run/"config.toml").string());
    const auto prior=read_rj_prior(toml::find(document,"rj_prior"));const auto physics=toml::find(document,"rj_physics");
    const auto data=read_observations(run/"observations.txt",run/"covariance.txt");
    InvLMForward backend;RJDispersionProblem problem(backend,prior,data,toml::find<double>(physics,"vp_vs_ratio"),toml::find<double>(physics,"density"));
    std::vector<std::pair<int,std::filesystem::path>> files;
    const std::regex pattern("chain_([0-9]+)_models\\.txt");
    for(const auto& entry:std::filesystem::directory_iterator(run)) {
        std::smatch match;const auto name=entry.path().filename().string();
        if(std::regex_match(name,match,pattern)) files.emplace_back(std::stoi(match[1]),entry.path());
    }
    std::sort(files.begin(),files.end());if(files.empty()) throw std::invalid_argument("No saved RJ models");
    std::vector<RJState> samples;std::vector<std::pair<int,size_t>> indices;
    std::vector<std::vector<double>> selected_rows;
    for(const auto& file:files) {
        const auto rows=read_table(file.second);
        if(static_cast<size_t>(per_chain)>rows.size()) throw std::invalid_argument("Predictive count exceeds available RJ draws");
        for(size_t i=0;i<rows.size();++i) {
            const auto& row=rows[i];
            if(row.size()<5 || row[0]!=static_cast<double>(i) || row[1]<prior.min_layers || row[1]>prior.max_layers
               || row[1]!=std::floor(row[1]) || row.size()!=static_cast<size_t>(2*row[1]+3))
                throw std::invalid_argument("Malformed saved RJ model");
        }
        for(int j=0;j<per_chain;++j) {
            const size_t index=static_cast<size_t>(j)*rows.size()/per_chain;const auto& row=rows[index];const int k=static_cast<int>(row[1]);
            samples.push_back({Eigen::Map<const Eigen::VectorXd>(row.data()+4,k-1),Eigen::Map<const Eigen::VectorXd>(row.data()+3+k,k)});
            indices.emplace_back(file.first,index);selected_rows.push_back(row);
        }
    }
    const auto result=posterior_predictive(problem,samples,seed);
    std::filesystem::create_directories(output);save_data(output,data);std::filesystem::copy_file(run/"config.toml",output/"config.toml");
    auto latent=output_file(output/"latent.txt");latent<<"# rows=selected draws; columns=observation order; km/s\n"<<result.latent<<'\n';
    auto replicated=output_file(output/"replicated.txt");replicated<<"# latent + full-covariance Gaussian noise; km/s\n"<<result.replicated<<'\n';
    auto selected=output_file(output/"selected_models.txt");selected<<"# row K log_likelihood log_prior interfaces[K-1] vs[K]\n";
    auto status_file=output_file(output/"draws.txt");status_file<<"# row chain source_draw_zero_based status observed_chi2 replicated_chi2\n";
    size_t failures=0;
    for(size_t i=0;i<samples.size();++i) {
        selected<<i;for(size_t j=1;j<selected_rows[i].size();++j) selected<<' '<<selected_rows[i][j];selected<<'\n';
        failures+=result.status[i]!=EvaluationStatus::valid;
        status_file<<i<<' '<<indices[i].first<<' '<<indices[i].second<<' '<<status(result.status[i])<<' '
                   <<result.observed_discrepancy[i]<<' '<<result.replicated_discrepancy[i]<<'\n';
    }
    auto metadata=output_file(output/"metadata.txt");
    metadata<<"version="<<DW_VERSION<<"\nseed="<<seed<<"\nsource_run="<<std::filesystem::absolute(run).string()
            <<"\nselected_per_chain="<<per_chain<<"\nselection=floor(j * chain_draws / per_chain); j starts at 0"
            <<"\nfailures="<<failures<<"\nposterior=transdimensional, conditional on declared prior/support/physics; fixed observed covariance\n";
    std::cout<<"RJ predictive_draws="<<samples.size()<<" failures="<<failures<<'\n';return failures?5:0;
}
int run_predictive(const std::filesystem::path& run,const std::filesystem::path& output,int per_chain,unsigned long long seed) {
    if(per_chain<1) throw std::invalid_argument("Predictive draw count must be positive");
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty");
    const auto document=toml::parse((run/"config.toml").string());
    const auto table=toml::find(document,"model"),inv=toml::find(document,"inversion");
    LayeredModel model{vector(table,"depth"),vector(table,"vs"),vector(table,"vp"),vector(table,"rho")};
    const auto bounds=read_bounds(inv,model.vs);
    FixedVs parameters(model,bounds.lower,bounds.upper,read_relation(inv));
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
    save_relation(metadata,parameters.relation());
    save_bounds(output,metadata,inv,parameters,bounds);
    std::cout<<"predictive_draws="<<samples.size()<<" failures="<<failures<<'\n';return failures?5:0;
}
}
