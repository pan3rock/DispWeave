#include "dispweave/gaussian.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <stdexcept>
namespace dw {
void validate_log_scale(double s) {
    if(!std::isfinite(s) || !std::isfinite(std::exp(-s)) || std::exp(-s)==0)
        throw std::invalid_argument("Noise log scale outside numerical range");
}
void GaussianData::validate() const {
    const auto n=values.size();
    if(n==0 || covariance.rows()!=n || covariance.cols()!=n || !values.allFinite() ||
       !covariance.allFinite() || !covariance.isApprox(covariance.transpose(),1e-12))
        throw std::invalid_argument("Malformed Gaussian values/covariance");
    Eigen::LLT<Eigen::MatrixXd> factor(covariance);
    if(factor.info()!=Eigen::Success) throw std::invalid_argument("Covariance must be positive definite");
}
GaussianDataLikelihood::GaussianDataLikelihood(GaussianData data):data_(std::move(data)) {
    data_.validate();
    // Symmetrize only within the validated relative roundoff tolerance.
    data_.covariance=(0.5*(data_.covariance+data_.covariance.transpose())).eval();
    cholesky_=data_.covariance.llt().matrixL();
    log_normalization_=data_.values.size()*std::log(2*std::acos(-1.))+
        2*cholesky_.diagonal().array().log().sum();
    if(!std::isfinite(log_normalization_)) throw std::invalid_argument("Covariance normalization not finite");
}
Eigen::VectorXd GaussianDataLikelihood::whiten(const Eigen::VectorXd& r,double scale) const {
    validate_log_scale(scale);
    if(r.size()!=data_.values.size() || !r.allFinite()) throw std::invalid_argument("Malformed residual");
    Eigen::VectorXd out=cholesky_.triangularView<Eigen::Lower>().solve(r)*std::exp(-scale);
    if(!out.allFinite()) throw std::runtime_error("Whitened residual overflow");
    return out;
}
Eigen::MatrixXd GaussianDataLikelihood::whiten_jacobian(const Eigen::MatrixXd& j,double scale) const {
    validate_log_scale(scale);
    if(j.rows()!=data_.values.size() || !j.allFinite()) throw std::invalid_argument("Malformed Jacobian");
    Eigen::MatrixXd out=cholesky_.triangularView<Eigen::Lower>().solve(j)*std::exp(-scale);
    if(!out.allFinite()) throw std::runtime_error("Whitened Jacobian overflow");
    return out;
}
double GaussianDataLikelihood::log_likelihood(const Eigen::VectorXd& prediction,double scale) const {
    if(prediction.size()!=data_.values.size()) throw std::invalid_argument("Prediction size mismatch");
    const double value=-0.5*(whiten(prediction-data_.values,scale).squaredNorm()+log_normalization_+2*prediction.size()*scale);
    if(!std::isfinite(value)) throw std::runtime_error("Likelihood overflow");
    return value;
}
double GaussianDataLikelihood::log_scale_gradient(const Eigen::VectorXd& prediction,double scale) const {
    if(prediction.size()!=data_.values.size()) throw std::invalid_argument("Prediction size mismatch");
    const double value=whiten(prediction-data_.values,scale).squaredNorm()-prediction.size();
    if(!std::isfinite(value)) throw std::runtime_error("Scale derivative overflow");
    return value;
}
}
