#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
namespace inversion { class Regularization; }
namespace dw {
struct PenaltyEvaluation { Eigen::VectorXd residual; Eigen::MatrixXd jacobian; };
// invLM penalties in physical Vs (km/s), depth (km). The returned Jacobian is
// in log Vs. This is an explicit deterministic penalty, not a likelihood term.
class VsRegularization {
public:
    VsRegularization(const Eigen::VectorXd& depth,const Eigen::VectorXd& reference_vs,
                     std::string type,double lambda);
    PenaltyEvaluation evaluate(const Eigen::VectorXd& q) const;
    bool active() const { return lambda_>0 && type_!="none"; }
    const std::string& type() const { return type_; }
    double weight() const { return lambda_; }
private:
    std::string type_;
    double lambda_;
    std::shared_ptr<const inversion::Regularization> backend_;
};
}
