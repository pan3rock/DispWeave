#pragma once
#include "objective.hpp"
#include "receiver.hpp"
#include "hmc.hpp"
namespace dw {
enum class NoiseScaleKind { fixed, lognormal, exponential };
struct NoiseScale {
    NoiseScaleKind kind=NoiseScaleKind::fixed;
    double first=1,second=1; // fixed s; or mean/sd of log(s); or exponential rate
    static NoiseScale fixed(double s=1) {return {NoiseScaleKind::fixed,s,1};}
    static NoiseScale lognormal(double mean_log=0,double sd_log=1) {return {NoiseScaleKind::lognormal,mean_log,sd_log};}
    static NoiseScale exponential(double rate=1) {return {NoiseScaleKind::exponential,rate,1};}
    bool inferred() const {return kind!=NoiseScaleKind::fixed;}
    void validate() const;
    double initial_log_scale() const;
};
struct ModelPrediction {
    EvaluationStatus status=EvaluationStatus::numerical_failure;
    Eigen::VectorXd values;
    Eigen::MatrixXd d_vs,d_vp; // Physical partials; absent when unavailable.
    std::vector<std::string> diagnostics;
    std::string message;
};
using ModelPredictor=std::function<ModelPrediction(const LayeredModel&,bool)>;
struct ObservationBlock {
    std::string name,observable,unit;
    GaussianDataLikelihood likelihood;
    ModelPredictor predict;
    bool verified_gradient=false;
    NoiseScale noise;
};
// The forward backend must outlive its dispersion block and all copies.
ObservationBlock dispersion_block(std::string name,const Forward&,DispersionObservations,
    NoiseScale=NoiseScale::fixed(),SensitivityMethod=SensitivityMethod::energy);
// Explicit physical label is mandatory; ambient_hv is rejected.
ObservationBlock ellipticity_block(std::string name,std::string interpretation,
    std::vector<DispersionQuery>,GaussianData,NoiseScale=NoiseScale::fixed(),SecularMethod=SecularMethod::dunkin);
// Observe selected samples on the declared full RF processing grid. Missing
// samples are omitted explicitly at construction, never by inspecting a fit.
ObservationBlock receiver_block(std::string name,ReceiverOptions,std::vector<int> sample_indices,
    GaussianData,NoiseScale=NoiseScale::fixed());
struct BlockEvaluation {
    std::string name,observable,unit,message;
    EvaluationStatus status=EvaluationStatus::numerical_failure;
    Eigen::VectorXd prediction,residual;
    std::vector<std::string> diagnostics;
    double log_scale=NAN,log_likelihood=NAN,chi_squared=NAN,log_scale_gradient=NAN;
};
struct JointEvaluation {
    EvaluationStatus status=EvaluationStatus::numerical_failure;
    double log_likelihood=NAN;
    Eigen::VectorXd residual,noise_gradient;
    Eigen::MatrixXd d_vs,d_vp,jacobian; // Whitened; q Jacobian only after mapping.
    std::vector<BlockEvaluation> blocks;
    std::string message;
};
enum class BlockDependence { conditional_independence, unspecified };
class JointLikelihood {
public:
    // Independence must be explicitly asserted; no silent block-diagonalization
    // of cross-observation covariance. No model/noise priors are added here.
    JointLikelihood(std::vector<ObservationBlock>,BlockDependence);
    JointEvaluation evaluate(const LayeredModel&,const Eigen::VectorXd& noise_log_scales={},bool gradient=false) const;
    const std::vector<ObservationBlock>& blocks() const {return blocks_;}
    Eigen::Index noise_size() const {return noise_size_;}
    Eigen::VectorXd initial_noise() const;
    bool supports_gradient() const;
private:
    std::vector<ObservationBlock> blocks_;
    Eigen::Index noise_size_=0,rows_=0;
};
struct FiniteDifferenceOptions {
    double step=1e-5; // In q=log(Vs), shortened only to fit the parameter support.
    double absolute_tolerance=1e-5,relative_tolerance=5e-3; // Whitened Jacobian units.
};
struct JointLeastSquares {
    DataEvaluation data;
    int evaluations=0;
    double max_difference=0; // Between second-order stencils at h and h/2.
};
class JointProblem {
public:
    JointProblem(FixedVs,JointLikelihood);
    // x=[q=log(Vs), eta=log(s) for inferred blocks in declaration order].
    Eigen::VectorXd initial_state(const Eigen::VectorXd& q) const;
    JointEvaluation evaluate(const Eigen::VectorXd& x,bool gradient=false) const;
    // Jacobian columns are q only: noise scales MUST stay fixed throughout LM.
    // Verified blocks retain their derivatives; FD fills only undeclared model
    // gradients. A failed verified derivative is not silently replaced by FD.
    // FD is an explicit, checked local approximation, never an HMC capability.
    JointLeastSquares least_squares(const Eigen::VectorXd& x,bool finite_difference=false,
                                   const FiniteDifferenceOptions& = {}) const;
    const FixedVs& parameters() const {return parameters_;}
    const JointLikelihood& likelihood() const {return likelihood_;}
    Eigen::Index size() const {return parameters_.lower().size()+likelihood_.noise_size();}
private:
    FixedVs parameters_;
    JointLikelihood likelihood_;
};
struct JointPosteriorEvaluation {
    EvaluationStatus status=EvaluationStatus::numerical_failure;
    double value=NAN,log_likelihood=NAN,log_model_prior=NAN,log_noise_prior=NAN,log_transform=NAN;
    Eigen::VectorXd gradient;
    JointEvaluation data;
    std::string message;
};
// Density w.r.t. dq d_eta. Physical Vs and noise-scale Jacobians are explicit.
JointPosteriorEvaluation joint_log_posterior(const JointProblem&,const Eigen::VectorXd& x,
                                           const Prior&,bool gradient=false);
// Reject unsupported model gradients when constructing an HMC/NUTS target.
// The problem must outlive the callable. No finite-difference fallback.
LogDensity joint_density(const JointProblem&,Prior);
}
