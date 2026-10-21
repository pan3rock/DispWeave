#pragma once
#include "parameterization.hpp"
#include <functional>
#include <cmath>
namespace dw {
struct DispersionObservations {
    std::vector<DispersionQuery> queries;
    Eigen::VectorXd phase_velocity;
    Eigen::MatrixXd covariance;
    void validate() const;
};
class GaussianLikelihood {
public:
    explicit GaussianLikelihood(DispersionObservations);
    // C(scale)=exp(2*log_scale)*C0; all normalization terms retained.
    double log_likelihood(const Eigen::VectorXd& prediction,double log_scale=0) const;
    double log_scale_gradient(const Eigen::VectorXd& prediction,double log_scale=0) const;
    Eigen::VectorXd whiten(const Eigen::VectorXd& residual,double log_scale=0) const;
    Eigen::MatrixXd whiten_jacobian(const Eigen::MatrixXd&,double log_scale=0) const;
    const DispersionObservations& data() const { return data_; }
private:
    DispersionObservations data_;
    Eigen::MatrixXd cholesky_;
    double log_normalization_;
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
// The latter excludes a noise prior: noise inference itself is not implemented.
PosteriorEvaluation log_posterior(const DispersionProblem&,const Eigen::VectorXd& q,
                                 const Prior&,bool gradient=true,double log_scale=0);
}
