#pragma once

#include <Eigen/Dense>

namespace fwd {

/// Chain-rule derivatives dVp/dVs and dRho/dVs per layer.
struct VsDerivatives {
    Eigen::ArrayXd dvp_dvs;
    Eigen::ArrayXd drho_dvs;
};

/// Base class: Vs → (Vp, Rho) empirical relationships.
/// generate() returns nl×2: col(0)=vp, col(1)=rho.
/// derivative() returns analytical dVp/dVs, dRho/dVs.
class ModelBuilder {
public:
  virtual ~ModelBuilder() = default;

  virtual Eigen::ArrayXXd generate(const Eigen::ArrayXd& vs) const = 0;
  virtual VsDerivatives derivative(const Eigen::ArrayXd& vs) const = 0;
};

class Brocher05 : public ModelBuilder {
public:
  Eigen::ArrayXXd generate(const Eigen::ArrayXd& vs) const override;
  VsDerivatives derivative(const Eigen::ArrayXd& vs) const override;
};

class Gardner : public ModelBuilder {
public:
  Gardner(double vp2vs = 1.7321,
          double alpha = 0.31, double beta = 0.25)
      : vp2vs_(vp2vs), alpha_(alpha), beta_(beta) {}
  Eigen::ArrayXXd generate(const Eigen::ArrayXd& vs) const override;
  VsDerivatives derivative(const Eigen::ArrayXd& vs) const override;

private:
  double vp2vs_, alpha_, beta_;
};

class NearSurface : public ModelBuilder {
public:
  NearSurface(double vp2vs = 1.7321)
      : vp2vs_(vp2vs) {}
  Eigen::ArrayXXd generate(const Eigen::ArrayXd& vs) const override;
  VsDerivatives derivative(const Eigen::ArrayXd& vs) const override;

private:
  double vp2vs_;
  static constexpr double a_ = -0.22374079;
  static constexpr double b_ = 1.32248261;
  static constexpr double c_ = 1.54840433;
};

/// Keep density constant while updating Vp from a fixed Vp/Vs ratio.
class ConstantRho : public ModelBuilder {
public:
  ConstantRho(double vp2vs, double rho);
  Eigen::ArrayXXd generate(const Eigen::ArrayXd& vs) const override;
  VsDerivatives derivative(const Eigen::ArrayXd& vs) const override;

private:
  double vp2vs_;
  double rho_;
};

/// Keep Vp and density fixed at the values supplied by the reference model.
class FixVpRho : public ModelBuilder {
public:
  FixVpRho(Eigen::ArrayXd vp, Eigen::ArrayXd rho);
  Eigen::ArrayXXd generate(const Eigen::ArrayXd& vs) const override;
  VsDerivatives derivative(const Eigen::ArrayXd& vs) const override;

private:
  Eigen::ArrayXd vp_;
  Eigen::ArrayXd rho_;
};

} // namespace fwd
