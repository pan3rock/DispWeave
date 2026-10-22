#pragma once
#include <Eigen/Core>
namespace dw {
// Values/covariance are in the observable's declared units. Signed values are
// allowed; physical validation belongs to the observation adapter.
struct GaussianData {
    Eigen::VectorXd values;
    Eigen::MatrixXd covariance;
    void validate() const;
};
void validate_log_scale(double);
class GaussianDataLikelihood {
public:
    explicit GaussianDataLikelihood(GaussianData);
    // C(eta)=exp(2*eta)*C0, including log determinant and 2*pi normalization.
    double log_likelihood(const Eigen::VectorXd& prediction,double eta=0) const;
    double log_scale_gradient(const Eigen::VectorXd& prediction,double eta=0) const;
    Eigen::VectorXd whiten(const Eigen::VectorXd& residual,double eta=0) const;
    Eigen::MatrixXd whiten_jacobian(const Eigen::MatrixXd&,double eta=0) const;
    const GaussianData& observations() const { return data_; }
private:
    GaussianData data_;
    Eigen::MatrixXd cholesky_;
    double log_normalization_;
};
}
