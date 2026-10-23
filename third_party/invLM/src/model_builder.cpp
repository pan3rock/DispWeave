#include "model_builder.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fwd {

using namespace Eigen;

ArrayXXd Brocher05::generate(const ArrayXd& vs) const {
  const int nl = vs.size();
  ArrayXXd model(nl, 2);
  for (int i = 0; i < nl; ++i) {
    double v = vs(i);
    double vp = 0.9409 + 2.0947 * v - 0.8206 * v * v +
                0.2683 * v * v * v - 0.0251 * v * v * v * v;
    double rho = 1.6612 * vp - 0.4721 * vp * vp +
                 0.0671 * vp * vp * vp - 0.0043 * vp * vp * vp * vp +
                 0.000106 * vp * vp * vp * vp * vp;
    model(i, 0) = vp;
    model(i, 1) = rho;
  }
  return model;
}

VsDerivatives Brocher05::derivative(const ArrayXd& vs) const {
  const int nl = vs.size();
  ArrayXd dvp(nl), drho(nl);
  for (int i = 0; i < nl; ++i) {
    double v = vs(i);
    dvp(i) = 2.0947 - 2.0 * 0.8206 * v + 3.0 * 0.2683 * v * v -
             4.0 * 0.0251 * v * v * v;
    double vp = 0.9409 + 2.0947 * v - 0.8206 * v * v +
                0.2683 * v * v * v - 0.0251 * v * v * v * v;
    drho(i) = (1.6612 - 2.0 * 0.4721 * vp + 3.0 * 0.0671 * vp * vp -
               4.0 * 0.0043 * vp * vp * vp +
               5.0 * 0.000106 * vp * vp * vp * vp) * dvp(i);
  }
  VsDerivatives res;
  res.dvp_dvs = dvp;
  res.drho_dvs = drho;
  return res;
}

ArrayXXd Gardner::generate(const ArrayXd& vs) const {
  const int nl = vs.size();
  ArrayXXd model(nl, 2);
  for (int i = 0; i < nl; ++i) {
    double vp = vp2vs_ * vs(i);
    model(i, 0) = vp;
    model(i, 1) = alpha_ * std::pow(1000.0 * vp, beta_);
  }
  return model;
}

VsDerivatives Gardner::derivative(const ArrayXd& vs) const {
  const int nl = vs.size();
  ArrayXd dvs = ArrayXd::Ones(nl);
  ArrayXd dvp = vp2vs_ * dvs;
  ArrayXd vp = vp2vs_ * vs;
  ArrayXd drho = alpha_ * beta_ * (1000.0 * vp).pow(beta_ - 1.0) * 1000.0 * dvp;

  VsDerivatives res;
  res.dvp_dvs = dvp;
  res.drho_dvs = drho;
  return res;
}

ArrayXXd NearSurface::generate(const ArrayXd& vs) const {
  const int nl = vs.size();
  ArrayXXd model(nl, 2);
  for (int i = 0; i < nl; ++i) {
    model(i, 0) = vp2vs_ * vs(i);
    model(i, 1) = a_ * vs(i) * vs(i) + b_ * vs(i) + c_;
  }
  return model;
}

VsDerivatives NearSurface::derivative(const Eigen::ArrayXd& vs) const {
  const int nl = vs.size();
  ArrayXd dvs = ArrayXd::Ones(nl);
  ArrayXd dvp = vp2vs_ * dvs;
  ArrayXd drho = (2.0 * a_ * vs + b_) * dvs;

  VsDerivatives res;
  res.dvp_dvs = dvp;
  res.drho_dvs = drho;
  return res;
}

ConstantRho::ConstantRho(double vp2vs, double rho)
    : vp2vs_(vp2vs), rho_(rho) {
  if (!std::isfinite(vp2vs_) || vp2vs_ <= 0.0 ||
      !std::isfinite(rho_) || rho_ <= 0.0)
    throw std::invalid_argument(
        "ConstantRho requires positive finite Vp/Vs and density");
}

ArrayXXd ConstantRho::generate(const ArrayXd& vs) const {
  ArrayXXd model(vs.size(), 2);
  model.col(0) = vp2vs_ * vs;
  model.col(1) = ArrayXd::Constant(vs.size(), rho_);
  return model;
}

VsDerivatives ConstantRho::derivative(const ArrayXd& vs) const {
  VsDerivatives res;
  res.dvp_dvs = ArrayXd::Constant(vs.size(), vp2vs_);
  res.drho_dvs = ArrayXd::Zero(vs.size());
  return res;
}

FixVpRho::FixVpRho(ArrayXd vp, ArrayXd rho)
    : vp_(std::move(vp)), rho_(std::move(rho)) {
  if (vp_.size() == 0 || vp_.size() != rho_.size())
    throw std::invalid_argument(
        "FixVpRho requires equally sized, non-empty reference Vp and rho");
  if (!(vp_.isFinite().all() && rho_.isFinite().all()) ||
      (vp_ <= 0.0).any() || (rho_ <= 0.0).any())
    throw std::invalid_argument(
        "FixVpRho requires positive finite Vp and rho in a five-column "
        "reference model");
}

ArrayXXd FixVpRho::generate(const ArrayXd& vs) const {
  if (vs.size() != vp_.size())
    throw std::invalid_argument(
        "FixVpRho Vs size differs from the reference model");
  ArrayXXd model(vs.size(), 2);
  model.col(0) = vp_;
  model.col(1) = rho_;
  return model;
}

VsDerivatives FixVpRho::derivative(const ArrayXd& vs) const {
  if (vs.size() != vp_.size())
    throw std::invalid_argument(
        "FixVpRho Vs size differs from the reference model");
  VsDerivatives res;
  res.dvp_dvs = ArrayXd::Zero(vs.size());
  res.drho_dvs = ArrayXd::Zero(vs.size());
  return res;
}

} // namespace fwd
