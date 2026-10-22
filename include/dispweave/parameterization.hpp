#pragma once
#include "forward.hpp"
namespace dw {
// Fixed interfaces/density. q=log(Vs); ratio=0 means fixed Vp.
class FixedVs {
public:
    FixedVs(LayeredModel reference,Eigen::VectorXd lower_vs,Eigen::VectorXd upper_vs,double vp_vs_ratio=0);
    bool contains(const Eigen::VectorXd& q) const;
    LayeredModel decode(const Eigen::VectorXd& q) const;
    Eigen::VectorXd encode(const LayeredModel&) const;
    Eigen::MatrixXd phase_jacobian(const Eigen::VectorXd& q,const ForwardResult&) const;
    // Chain rule for any observable with physical Vs/Vp partial derivatives.
    Eigen::MatrixXd model_jacobian(const Eigen::VectorXd& q,const Eigen::MatrixXd& d_vs,
                                    const Eigen::MatrixXd& d_vp) const;
    double log_abs_det_jacobian(const Eigen::VectorXd& q) const;
    Eigen::VectorXd log_abs_det_gradient(const Eigen::VectorXd& q) const;
    const Eigen::VectorXd& lower() const { return lower_; }
    const Eigen::VectorXd& upper() const { return upper_; }
private:
    LayeredModel reference_;
    Eigen::VectorXd lower_,upper_;
    double ratio_;
};
}
