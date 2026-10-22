#pragma once
#include "parameterization.hpp"
#include "gaussian.hpp"
#include <functional>
#include <cmath>
namespace dw {
struct DispersionObservations {
    std::vector<DispersionQuery> queries;
    Eigen::VectorXd phase_velocity;
    Eigen::MatrixXd covariance;
    void validate() const;
};
class GaussianLikelihood : public GaussianDataLikelihood {
public:
    explicit GaussianLikelihood(DispersionObservations);
    const DispersionObservations& data() const { return data_; }
private:
    DispersionObservations data_;
};
enum class EvaluationStatus { valid, outside_support, forward_failure, gradient_failure, numerical_failure };
struct DataEvaluation {
    EvaluationStatus status=EvaluationStatus::numerical_failure;
    double log_likelihood=NAN, least_squares_cost=NAN;
    Eigen::VectorXd residual;
    Eigen::MatrixXd jacobian;
    ForwardResult forward;
    std::string message;
};
class DispersionProblem {
public:
    // Backend must outlive the problem; data and parameterization are owned copies.
    DispersionProblem(const Forward&,FixedVs,DispersionObservations,SensitivityMethod=SensitivityMethod::energy);
    DataEvaluation evaluate(const Eigen::VectorXd& q,bool gradient=true,double log_scale=0) const;
    const FixedVs& parameters() const { return parameters_; }
    const GaussianLikelihood& likelihood() const { return likelihood_; }
private:
    const Forward& forward_;
    FixedVs parameters_;
    GaussianLikelihood likelihood_;
    SensitivityMethod derivative_;
};
enum class PriorMeasure { physical_vs, log_vs };
struct PriorEvaluation { double value; Eigen::VectorXd gradient; };
struct Prior {
    PriorMeasure measure;
    std::function<PriorEvaluation(const Eigen::VectorXd&,bool)> evaluate;
};
Prior gaussian_prior(Eigen::VectorXd mean,Eigen::VectorXd sigma,PriorMeasure);
Prior uniform_prior(Eigen::VectorXd lower,Eigen::VectorXd upper,PriorMeasure);
struct PosteriorEvaluation {
    EvaluationStatus status=EvaluationStatus::numerical_failure;
    double log_likelihood=NAN, log_prior=NAN, log_transform=NAN, value=NAN;
    double log_scale_gradient=NAN;
    Eigen::VectorXd gradient;
    std::string message;
    DataEvaluation data;
};
// q-gradient and optional Gaussian noise log-scale likelihood derivative.
// This legacy overload keeps noise external and adds no noise prior. Use
// joint_log_posterior (joint.hpp) for explicit noise-scale inference.
PosteriorEvaluation log_posterior(const DispersionProblem&,const Eigen::VectorXd& q,
                                 const Prior&,bool gradient=true,double log_scale=0);
}
