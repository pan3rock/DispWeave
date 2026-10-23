#include "dispweave/workflow.hpp"
#include "dispweave/joint.hpp"
#include "dispweave/metropolis.hpp"
#include "dispweave/lm.hpp"
#include "version.hpp"
#include <toml.hpp>
#include <Eigen/Cholesky>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
namespace dw {
namespace {
Eigen::VectorXd vector(const toml::value& t,const char* key) {
    const auto a=toml::find<std::vector<double>>(t,key);
    return Eigen::Map<const Eigen::VectorXd>(a.data(),a.size());
}
Eigen::MatrixXd matrix(const toml::value& t,const char* key) {
    const auto rows=toml::find<std::vector<std::vector<double>>>(t,key);
    if(rows.empty() || rows[0].empty()) throw std::invalid_argument("Empty matrix");
    Eigen::MatrixXd a(rows.size(),rows[0].size());
    for(size_t i=0;i<rows.size();++i) {
        if(rows[i].size()!=rows[0].size()) throw std::invalid_argument("Ragged matrix");
        for(size_t j=0;j<rows[i].size();++j) a(i,j)=rows[i][j];
    }
    return a;
}
std::ofstream file(const std::filesystem::path& p) {
    std::ofstream f(p);if(!f) throw std::runtime_error("Cannot write "+p.string());
    f.exceptions(std::ios::badbit|std::ios::failbit);f<<std::setprecision(17);return f;
}
const char* status(EvaluationStatus s) {
    switch(s) {
        case EvaluationStatus::valid:return "valid";
        case EvaluationStatus::outside_support:return "outside_support";
        case EvaluationStatus::forward_failure:return "forward_failure";
        case EvaluationStatus::gradient_failure:return "gradient_failure";
        default:return "numerical_failure";
    }
}
NoiseScale noise(const toml::value& t) {
    if(!t.contains("noise")) return NoiseScale::fixed();
    const auto n=toml::find(t,"noise");const auto kind=toml::find<std::string>(n,"kind");
    if(kind=="fixed") return NoiseScale::fixed(toml::find<double>(n,"scale"));
    if(kind=="lognormal") return NoiseScale::lognormal(toml::find<double>(n,"mean_log"),toml::find<double>(n,"sd_log"));
    if(kind=="exponential") return NoiseScale::exponential(toml::find<double>(n,"rate"));
    throw std::invalid_argument("Unknown noise prior");
}
struct LoadedBlock {ObservationBlock block;Eigen::VectorXd coordinates;std::vector<int> labels;};
LoadedBlock load_block(const toml::value& t,const Forward& backend,bool synthetic) {
    const auto name=toml::find<std::string>(t,"name"),kind=toml::find<std::string>(t,"observable"),unit=toml::find<std::string>(t,"unit");
    if(!std::regex_match(name,std::regex("[A-Za-z][A-Za-z0-9_-]*"))) throw std::invalid_argument("Block name must be a safe identifier");
    Eigen::VectorXd coordinates;std::vector<int> labels;std::vector<DispersionQuery> queries;ReceiverOptions processing;
    if(kind=="rayleigh_phase_velocity" || kind=="rayleigh_modal_ellipticity") {
        if(unit!=(kind=="rayleigh_phase_velocity"?"km/s":"dimensionless")) throw std::invalid_argument("Modal observation unit mismatch");
        coordinates=vector(t,"frequency_hz");labels=toml::find<std::vector<int>>(t,"mode");
        if(labels.size()!=static_cast<size_t>(coordinates.size())) throw std::invalid_argument("Mode/frequency size mismatch");
        for(size_t j=0;j<labels.size();++j) queries.push_back({coordinates[j],labels[j]});
    } else if(kind=="p_receiver_function") {
        if(unit!="dimensionless") throw std::invalid_argument("Receiver-function unit must be dimensionless");
        const auto p=toml::find(t,"processing");
        if(toml::find_or<std::string>(p,"incident","P")!="P" || toml::find_or<std::string>(p,"deconvolution","waterlevel")!="waterlevel")
            throw std::invalid_argument("Only P incidence and waterlevel deconvolution supported");
        processing.slowness_s_km=toml::find<double>(p,"slowness_s_km");processing.samples=toml::find<int>(p,"samples");
        processing.dt=toml::find<double>(p,"dt");processing.pre_arrival=toml::find<double>(p,"pre_arrival");
        processing.time_shift=toml::find<double>(p,"time_shift");processing.gaussian_width=toml::find<double>(p,"gaussian_width");
        processing.water_level=toml::find<double>(p,"water_level");if(p.contains("source")) processing.source=vector(p,"source");
        labels=toml::find<std::vector<int>>(t,"sample_indices");coordinates.resize(labels.size());
        for(size_t j=0;j<labels.size();++j) coordinates[j]=labels[j]*processing.dt-processing.pre_arrival;
    } else throw std::invalid_argument("Unsupported observable; ambient H/V is not modal ellipticity");
    GaussianData data;
    data.values=synthetic?Eigen::VectorXd::Ones(coordinates.size()):vector(t,"values");
    if(t.contains("covariance")==t.contains("sigma")) throw std::invalid_argument("Provide exactly one of covariance or sigma");
    if(t.contains("covariance")) data.covariance=matrix(t,"covariance");
    else {
        const auto sigma=vector(t,"sigma");
        if(!sigma.allFinite() || (sigma.array()<=0).any()) throw std::invalid_argument("Sigma must be finite and positive");
        data.covariance=sigma.array().square().matrix().asDiagonal();
    }
    if(kind=="rayleigh_phase_velocity") return {dispersion_block(name,backend,{queries,data.values,data.covariance},noise(t)),coordinates,labels};
    if(kind=="rayleigh_modal_ellipticity") return {ellipticity_block(name,kind,queries,data,noise(t)),coordinates,labels};
    return {receiver_block(name,processing,labels,data,noise(t)),coordinates,labels};
}
void save_data(const std::filesystem::path& output,const std::vector<LoadedBlock>& blocks) {
    for(const auto& b:blocks) {
        auto f=file(output/("observations_"+b.block.name+".txt"));f<<"# coordinate mode_or_sample value base_sigma\n";
        const auto& data=b.block.likelihood.observations();
        for(Eigen::Index j=0;j<data.values.size();++j) f<<b.coordinates[j]<<' '<<b.labels[j]<<' '<<data.values[j]<<' '<<std::sqrt(data.covariance(j,j))<<'\n';
        file(output/("covariance_"+b.block.name+".txt"))<<data.covariance<<'\n';
    }
}
}
int run_joint(const std::filesystem::path& config,const std::filesystem::path& output,bool generate_data) {
    const auto begin=std::chrono::steady_clock::now();
    if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
        throw std::invalid_argument("Joint output must be new or empty");
    auto document=toml::parse(config.string());
    if(!toml::find_or<bool>(document,"independent_blocks",false)) throw std::invalid_argument("Declare independent_blocks=true to combine block likelihoods");
    const auto m=toml::find(document,"model"),p=toml::find(document,"parameters");
    const LayeredModel reference{vector(m,"depth"),vector(m,"vs"),vector(m,"vp"),vector(m,"rho")};
    const double ratio=toml::find_or<double>(p,"vp_vs_ratio",0.);
    const FixedVs parameters(reference,vector(p,"lower_vs"),vector(p,"upper_vs"),ratio);
    InvLMForward backend;auto tables=toml::find<std::vector<toml::value>>(document,"observations");
    std::vector<LoadedBlock> loaded;std::vector<Eigen::VectorXd> truths;
    for(const auto& t:tables) loaded.push_back(load_block(t,backend,generate_data));
    if(generate_data) {
        const auto synthetic=toml::find(document,"synthetic");const auto seed=toml::find<std::uint64_t>(synthetic,"seed");
        std::seed_seq seeds{std::uint32_t(seed),std::uint32_t(seed>>32),0x44415441u};std::mt19937_64 rng(seeds);std::normal_distribution<double> normal;
        for(size_t b=0;b<loaded.size();++b) {
            auto model=reference;model.vs=tables[b].contains("synthetic_vs")?vector(tables[b],"synthetic_vs"):vector(synthetic,"vs");
            if(ratio) model.vp=ratio*model.vs;
            model.validate();
            const auto prediction=loaded[b].block.predict(model,false);
            if(prediction.status!=EvaluationStatus::valid) throw std::invalid_argument("Synthetic "+loaded[b].block.name+": "+prediction.message);
            truths.push_back(prediction.values);
            const auto& base=loaded[b].block.likelihood.observations();const Eigen::MatrixXd l=base.covariance.llt().matrixL();
            const double scale=toml::find_or<double>(tables[b],"true_scale",1.);
            if(!std::isfinite(scale) || scale<=0) throw std::invalid_argument("Synthetic noise scale must be positive");
            Eigen::VectorXd z(base.values.size());for(double& value:z) value=normal(rng);
            const Eigen::VectorXd data=prediction.values+scale*l*z;
            tables[b]["values"]=std::vector<double>(data.data(),data.data()+data.size());
        }
        document["observations"]=tables;loaded.clear();
        for(const auto& t:tables) loaded.push_back(load_block(t,backend,false));
    }
    std::vector<ObservationBlock> blocks;for(const auto& b:loaded) blocks.push_back(b.block);
    JointProblem problem(parameters,JointLikelihood(blocks,BlockDependence::conditional_independence));
    std::filesystem::create_directories(output);std::filesystem::copy_file(config,output/"config.toml");
    file(output/"observations.toml")<<toml::format(document);save_data(output,loaded);
    auto metadata=file(output/"metadata.txt");metadata<<"version="<<DW_VERSION<<"\ncompiler="<<DW_COMPILER<<"\nbuild="<<DW_BUILD_TYPE
        <<"\ninvLM="<<DW_INVLM_COMMIT<<"\nconditional_independence=true\nparameterization=fixed_interfaces_density_logVs"
        <<"\nmode="<<(generate_data?"synthetic_data":"joint_inference")<<'\n';
    if(generate_data) {
        for(size_t b=0;b<loaded.size();++b) file(output/("truth_"+loaded[b].block.name+".txt"))<<truths[b]<<'\n';
        std::cout<<"joint_data_blocks="<<loaded.size()<<" output="<<output<<'\n';return 0;
    }
    const auto prior_table=toml::find(document,"prior");const auto measure_name=toml::find<std::string>(prior_table,"measure");
    if(measure_name!="physical_vs" && measure_name!="log_vs") throw std::invalid_argument("Unknown prior measure");
    const auto measure=measure_name=="physical_vs"?PriorMeasure::physical_vs:PriorMeasure::log_vs;
    const auto kind=toml::find<std::string>(prior_table,"kind");
    if(kind!="uniform" && kind!="gaussian") throw std::invalid_argument("Unknown model prior");
    const Prior prior=kind=="gaussian"?gaussian_prior(vector(prior_table,"mean"),vector(prior_table,"sd"),measure):
        uniform_prior(measure==PriorMeasure::physical_vs?vector(p,"lower_vs"):parameters.lower(),
                      measure==PriorMeasure::physical_vs?vector(p,"upper_vs"):parameters.upper(),measure);
    const auto sampler=toml::find(document,"sampler");MetropolisOptions options;
    options.warmup=toml::find<int>(sampler,"warmup");options.draws=toml::find<int>(sampler,"draws");options.seed=toml::find<std::uint64_t>(sampler,"seed");
    options.adapt_scale=toml::find_or<bool>(sampler,"adapt_scale",true);options.adapt_covariance=toml::find_or<bool>(sampler,"adapt_covariance",true);
    options.target_accept=toml::find_or<double>(sampler,"target_accept",.234);options.scale=toml::find_or<double>(sampler,"scale",1.);
    const int chains=toml::find<int>(sampler,"chains"),ppc=toml::find_or<int>(sampler,"predictive_per_chain",200);
    if(chains<1 || ppc<1 || ppc>options.draws) throw std::invalid_argument("Invalid chain/predictive count");
    Eigen::VectorXd proposal_sd=vector(sampler,"proposal_sd");
    if(proposal_sd.size()!=problem.size() || !proposal_sd.allFinite() || (proposal_sd.array()<=0).any()) throw std::invalid_argument("Invalid proposal standard deviations");
    const Eigen::MatrixXd proposal=proposal_sd.array().square().matrix().asDiagonal();
    const Eigen::MatrixXd starts=matrix(sampler,"initial_vs");
    if(starts.rows()!=chains || starts.cols()!=reference.vs.size()) throw std::invalid_argument("One initial Vs row per chain required");
    std::vector<Eigen::VectorXd> initial;
    for(int chain=0;chain<chains;++chain) {
        Eigen::VectorXd x=problem.initial_state(starts.row(chain).transpose().array().log().matrix());
        if(joint_log_posterior(problem,x,prior).status!=EvaluationStatus::valid) throw std::invalid_argument("Invalid initial joint posterior for chain "+std::to_string(chain));
        initial.push_back(x);
    }
    metadata<<"sampler=gaussian_random_walk_metropolis\nmodel_prior_kind="<<kind<<"\nmodel_prior_measure="<<measure_name
        <<"\nseed="<<options.seed<<"\nchains="<<chains<<"\nwarmup="<<options.warmup<<"\ndraws="<<options.draws<<"\nmodel_parameters="<<reference.vs.size()
        <<"\nnoise_parameters="<<problem.likelihood().noise_size()<<"\npredictive_per_chain="<<ppc<<"\nstate_order=logVs_then_inferred_log_scales_in_block_order\n";
    const auto reference_state=problem.initial_state(parameters.encode(reference));
    const bool lm_enabled=document.contains("lm") && toml::find_or<bool>(toml::find(document,"lm"),"enabled",false);
    if(lm_enabled) {
        // Near a least-squares minimum, standard root error can exceed the
        // predicted reduction. Refine the same brackets for BOTH values and
        // kernels during LM; leave the declared sampling target unchanged.
        const InvLMForward lm_backend(SecularMethod::dunkin,RootAccuracy::refined);
        std::vector<ObservationBlock> lm_blocks;
        for(const auto& table:tables) lm_blocks.push_back(load_block(table,lm_backend,false).block);
        const JointProblem lm_problem(parameters,JointLikelihood(std::move(lm_blocks),BlockDependence::conditional_independence));
        LMOptions lm;lm.max_iterations=toml::find_or<int>(toml::find(document,"lm"),"max_iterations",100);
        lm.gradient_tolerance=toml::find_or<double>(toml::find(document,"lm"),"gradient_tolerance",1e-5);
        const auto fit=bounded_lm([&](const Eigen::VectorXd& q){auto x=reference_state;x.head(q.size())=q;return lm_problem.least_squares(x,true).data;},
            reference_state.head(reference.vs.size()),parameters.lower(),parameters.upper(),lm);
        file(output/"lm_vs.txt")<<fit.q.array().exp().matrix()<<'\n';
        file(output/"lm_summary.txt")<<"termination="<<to_string(fit.termination)<<"\nreason="<<fit.reason<<"\ncost="<<fit.evaluation.least_squares_cost
            <<"\nevaluations="<<fit.evaluations<<"\nprojected_gradient="<<fit.projected_gradient
            <<"\naccepted="<<fit.accepted<<"\nrejected="<<fit.rejected<<"\nforward_failures="<<fit.forward_failures
            <<"\ngradient_failures="<<fit.gradient_failures
            <<"\nnoise_scales=fixed_at_prior_initial_values\nused_for_chain_initialization=false\ndispersion_root_accuracy=refined\n";
        auto trace=file(output/"lm_trace.txt");
        trace<<"# iteration cost trial_cost damping gain_ratio projected_gradient accepted trial_status predicted_reduction actual_reduction step_norm\n";
        for(const auto& t:fit.trace) trace<<t.iteration<<' '<<t.cost<<' '<<t.trial_cost<<' '<<t.damping<<' '<<t.gain_ratio<<' '
            <<t.projected_gradient<<' '<<t.accepted<<' '<<status(t.trial_status)<<' '<<t.predicted_reduction<<' '<<t.actual_reduction<<' '<<t.step_norm<<'\n';
    }
    std::vector<std::ofstream> latent,replicate,misfit;std::vector<Eigen::MatrixXd> factors;
    for(const auto& b:blocks) {
        latent.push_back(file(output/("prediction_"+b.name+".txt")));replicate.push_back(file(output/("replicate_"+b.name+".txt")));
        misfit.push_back(file(output/("predictive_misfit_"+b.name+".txt")));
        latent.back()<<"# chain draw then latent predictions in observation order\n";replicate.back()<<"# chain draw then replicated observations including covariance/noise scale\n";
        misfit.back()<<"# chain draw observed_chi_squared replicated_chi_squared scale\n";
        factors.push_back(b.likelihood.observations().covariance.llt().matrixL());
    }
    auto failed_evaluations=file(output/"failed_evaluations.txt");
    failed_evaluations<<"# evaluation status state_in_declared_order message; includes warmup\n";
    long long evaluation_count=0,failed_evaluation_count=0;
    const ValueDensity target=[&](const Eigen::VectorXd& x){
        const auto e=joint_log_posterior(problem,x,prior);++evaluation_count;
        if(e.status!=EvaluationStatus::valid && e.status!=EvaluationStatus::outside_support) {
            ++failed_evaluation_count;
            failed_evaluations<<evaluation_count<<' '<<status(e.status)<<' '<<x.transpose()<<' '<<std::quoted(e.message)<<'\n';
        }
        return ValueEvaluation{e.status,e.value};
    };
    int failures=0;
    auto predictive_status=file(output/"predictive_status.txt");predictive_status<<"# chain draw status message\n";
    for(int chain=0;chain<chains;++chain) {
        options.chain_id=chain;const auto result=sample_metropolis(target,initial[chain],proposal,options);
        const auto prefix="chain_"+std::to_string(chain);
        file(output/(prefix+"_state.txt"))<<result.draws<<'\n';file(output/(prefix+"_vs.txt"))<<result.draws.leftCols(reference.vs.size()).array().exp().matrix()<<'\n';
        file(output/(prefix+"_log_posterior.txt"))<<result.log_density<<'\n';file(output/(prefix+"_proposal_covariance.txt"))<<result.covariance*result.scale*result.scale<<'\n';
        auto trace=file(output/(prefix+"_trace.txt"));trace<<"# warmup accepted status scale acceptance_probability\n";
        int accepted=0,invalid=0,support=0;
        for(const auto& t:result.transitions) {
            trace<<t.warmup<<' '<<t.accepted<<' '<<status(t.status)<<' '<<t.scale<<' '<<t.acceptance_probability<<'\n';
            if(!t.warmup) {accepted+=t.accepted;support+=t.status==EvaluationStatus::outside_support;invalid+=t.status!=EvaluationStatus::valid && t.status!=EvaluationStatus::outside_support;}
        }
        auto adaptation=file(output/(prefix+"_adaptation.txt"));adaptation<<"# warmup_iteration samples followed by proposal covariance\n";
        for(const auto& update:result.covariance_updates) adaptation<<update.warmup_iteration<<' '<<update.samples<<'\n'<<update.covariance<<'\n';
        failures+=invalid+(accepted==0);
        std::seed_seq seeds{std::uint32_t(options.seed),std::uint32_t(options.seed>>32),std::uint32_t(chain),0x505043u};
        std::mt19937_64 rng(seeds);std::normal_distribution<double> normal;
        for(int j=0;j<ppc;++j) {
            const int draw=static_cast<int>((static_cast<long long>(2)*j+1)*options.draws/(2LL*ppc));
            const Eigen::VectorXd x=result.draws.row(draw).transpose();const auto prediction=problem.evaluate(x);
            auto predictive_state=prediction.status;auto predictive_message=prediction.message;
            if(prediction.status!=EvaluationStatus::valid) ++failures;
            for(size_t b=0;b<blocks.size();++b) {
                const auto count=blocks[b].likelihood.observations().values.size();
                Eigen::VectorXd values=Eigen::VectorXd::Constant(count,NAN),y=values;double observed=NAN,replicated=NAN,scale=NAN;
                if(prediction.status==EvaluationStatus::valid) {
                    values=prediction.blocks[b].prediction;scale=std::exp(prediction.blocks[b].log_scale);Eigen::VectorXd z(count);
                    for(double& value:z) value=normal(rng);
                    y=values+scale*factors[b]*z;observed=prediction.blocks[b].chi_squared;replicated=z.squaredNorm();
                    if(!y.allFinite()) {++failures;predictive_state=EvaluationStatus::numerical_failure;predictive_message+="Replicate overflow: "+blocks[b].name+"; ";}
                }
                latent[b]<<chain<<' '<<draw<<' '<<values.transpose()<<'\n';replicate[b]<<chain<<' '<<draw<<' '<<y.transpose()<<'\n';
                misfit[b]<<chain<<' '<<draw<<' '<<observed<<' '<<replicated<<' '<<scale<<'\n';
            }
            predictive_status<<chain<<' '<<draw<<' '<<status(predictive_state)<<' '<<std::quoted(predictive_message)<<'\n';
        }
        metadata<<prefix<<"_accepted="<<accepted<<'\n'<<prefix<<"_invalid="<<invalid<<'\n'<<prefix<<"_support_rejections="<<support<<'\n';
        std::cout<<prefix<<" accepted="<<accepted<<'/'<<options.draws<<" invalid="<<invalid<<" support="<<support<<std::endl;
    }
    metadata<<"failures="<<failures<<"\ntarget_evaluations_including_warmup="<<evaluation_count
        <<"\nfailed_target_evaluations_including_warmup="<<failed_evaluation_count
        <<"\nseconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()
        <<"\ninterpretation=conditional_on_fixed_parameterization; convergence_requires_multichain_diagnostics; latent_intervals_are_not_resolution\n";
    return failures?5:0;
}
}
