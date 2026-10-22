#pragma once
#include "objective.hpp"
#include <cstdint>
#include <random>
namespace dw {
// Density with respect to the supplied coordinates. Invalid gradients are never zero-filled.
struct DensityEvaluation {
    EvaluationStatus status=EvaluationStatus::numerical_failure;
    double value=NAN;
    Eigen::VectorXd gradient;
};
using LogDensity=std::function<DensityEvaluation(const Eigen::VectorXd&)>;
struct BoxTransformEvaluation {
    Eigen::VectorXd q, derivative, log_jacobian_gradient;
    double log_jacobian;
    bool interior=true;
};
class LogisticBox {
public:
    LogisticBox(Eigen::VectorXd lower,Eigen::VectorXd upper);
    BoxTransformEvaluation forward(const Eigen::VectorXd& z) const;
    Eigen::VectorXd inverse(const Eigen::VectorXd& q) const;
private:
    Eigen::VectorXd lower_,upper_,width_;
};
LogDensity transform_density(LogDensity,LogisticBox);
// The problem must outlive the returned callable. Input/output density is in q=log(Vs).
LogDensity dispersion_density(const DispersionProblem&,Prior);
class EuclideanMetric {
public:
    // Momentum covariance M, NOT inverse mass or position covariance.
    explicit EuclideanMetric(Eigen::MatrixXd mass);
    Eigen::VectorXd velocity(const Eigen::VectorXd& p) const;
    Eigen::VectorXd momentum(std::mt19937_64&) const;
    double kinetic(const Eigen::VectorXd& p) const;
    Eigen::MatrixXd mass() const {return cholesky_*cholesky_.transpose();}
    Eigen::Index size() const {return cholesky_.rows();}
private:
    Eigen::MatrixXd cholesky_;
};
struct HamiltonianState { Eigen::VectorXd position,momentum; DensityEvaluation density; };
struct IntegrationResult {
    HamiltonianState state;
    EvaluationStatus status=EvaluationStatus::valid;
    int evaluations=0;
};
// Signed step supports reversibility checks. No clipping, reflection or root-search AD.
IntegrationResult leapfrog(const LogDensity&,HamiltonianState,const EuclideanMetric&,double step,int steps);
enum class MetricAdaptation { none,diagonal,dense };
struct MetricUpdate { int warmup_iteration,samples; Eigen::MatrixXd mass; };
struct HMCOptions {
    int warmup=500,draws=1000,leapfrog_steps=10;
    double step_size=.1,target_accept=.8,divergence_threshold=1000;
    double step_jitter=.1; // State-independent uniform multiplicative jitter in [1-j,1+j].
    bool adapt_step_size=true;
    MetricAdaptation metric_adaptation=MetricAdaptation::none;
    std::uint64_t seed=2026,chain_id=0;
};
struct HMCTransition {
    bool warmup,accepted,divergent;
    double step_size,acceptance_probability,energy_error;
    EvaluationStatus status;
    int evaluations;
};
struct HMCResult {
    Eigen::MatrixXd draws; // Retains repeated states on rejection; no thinning.
    std::vector<HMCTransition> transitions; // Includes warmup, explicitly marked.
    double step_size;
    Eigen::MatrixXd mass; // Final frozen momentum covariance.
    std::vector<MetricUpdate> metric_updates;
};
HMCResult sample_hmc(const LogDensity&,const Eigen::VectorXd& initial,const EuclideanMetric&,const HMCOptions& = {});
}
