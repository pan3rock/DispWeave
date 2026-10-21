#pragma once

#include "secfunc.hpp"

namespace fwd {

class FastScalarSecularFunction final : public SecularFunction {
public:
    FastScalarSecularFunction(const Eigen::Ref<const Eigen::ArrayXXd> model,
                              bool sh);
    double evaluate(double f, double c) override;

private:
    Eigen::ArrayXd inv_vp2_;
    Eigen::ArrayXd inv_vs2_;
    Eigen::ArrayXd half_inv_vs2_;
    Eigen::ArrayXd l_;
    Eigen::ArrayXd inv_l_;
};

} // namespace fwd
