#include "dispweave/regularization.hpp"
#include "regularization.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
VsRegularization::VsRegularization(const Eigen::VectorXd& depth,const Eigen::VectorXd& reference,
                                   std::string type,double lambda):type_(std::move(type)),lambda_(lambda) {
    if(type_!="none" && type_!="reference_damping" && type_!="perturbation_smooth" && type_!="vs_smooth")
        throw std::invalid_argument("Unknown LM regularization type");
    if(!std::isfinite(lambda_) || lambda_<0 || (type_=="none" && lambda_!=0))
        throw std::invalid_argument("Regularization lambda must be finite/nonnegative; none requires zero");
    inversion::RegularizationSettings s;s.type=type_;s.lambda_ref=0;s.lambda_smooth=0;
    if(type_=="reference_damping") s.lambda_ref=lambda_;
    else if(type_!="none") s.lambda_smooth=lambda_;
    backend_=std::make_shared<inversion::Regularization>(depth,reference,s);
}
PenaltyEvaluation VsRegularization::evaluate(const Eigen::VectorXd& q) const {
    if(q.size()!=backend_->num_parameters() || !q.allFinite()) throw std::invalid_argument("Malformed regularization coordinates");
    if(!active()) return {Eigen::VectorXd(0),Eigen::MatrixXd(0,q.size())};
    const Eigen::VectorXd vs=q.array().exp();inversion::RegularizationEvaluation e;
    if(!backend_->evaluate(vs,true,e)) throw std::runtime_error("Regularization evaluation failed");
    // Upstream keeps zero reference rows for smooth-only penalties. Remove
    // those inert rows while preserving the exact quadratic and derivatives.
    const int offset=type_=="reference_damping"?0:backend_->num_reference_residuals();
    const int rows=e.residuals.size()-offset;
    PenaltyEvaluation result{e.residuals.tail(rows),e.jacobian.bottomRows(rows)*vs.asDiagonal()};
    if(!result.residual.allFinite() || !result.jacobian.allFinite()) throw std::runtime_error("Regularization overflow");
    return result;
}
}
