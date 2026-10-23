#pragma once
#include "forward.hpp"
#include <memory>
namespace fwd { class ModelBuilder; }
namespace dw {
struct ElasticRelation {
    std::string kind="fixvprho";
    double vp_vs_ratio=1.7321, gardner_alpha=.31, gardner_beta=.25, density=2.;
};
// Fixed interfaces, q=log(Vs). Vp/rho fixed or deterministically derived.
class FixedVs {
public:
    FixedVs(LayeredModel reference,Eigen::VectorXd lower_vs,Eigen::VectorXd upper_vs,double vp_vs_ratio=0);
    FixedVs(LayeredModel reference,Eigen::VectorXd lower_vs,Eigen::VectorXd upper_vs,ElasticRelation);
    bool contains(const Eigen::VectorXd& q) const;
    LayeredModel decode(const Eigen::VectorXd& q) const;
    Eigen::VectorXd encode(const LayeredModel&) const;
    Eigen::MatrixXd phase_jacobian(const Eigen::VectorXd& q,const ForwardResult&) const;
    // Chain rule for any observable with physical Vs/Vp partial derivatives.
    Eigen::MatrixXd model_jacobian(const Eigen::VectorXd& q,const Eigen::MatrixXd& d_vs,
                                    const Eigen::MatrixXd& d_vp,const Eigen::MatrixXd& d_rho={}) const;
    double log_abs_det_jacobian(const Eigen::VectorXd& q) const;
    Eigen::VectorXd log_abs_det_gradient(const Eigen::VectorXd& q) const;
    const Eigen::VectorXd& lower() const { return lower_; }
    const Eigen::VectorXd& upper() const { return upper_; }
    const ElasticRelation& relation() const { return relation_; }
    const LayeredModel& reference_model() const { return reference_; }
private:
    LayeredModel reference_;
    Eigen::VectorXd lower_,upper_;
    double ratio_;
    ElasticRelation relation_;
    std::shared_ptr<const fwd::ModelBuilder> builder_;
};
}
