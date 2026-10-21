#pragma once
#include "model.hpp"
#include <Eigen/Dense>
#include <complex>
#include <memory>

namespace fwd {

struct KernelResult {
    // Solid Rayleigh/Love waves: corrective boundary reconstruction and
    // normalized variational energy mismatch.
    bool stabilized_eigenfunctions = false;
    double energy_residual = 0.0;
    double group_velocity{};
    Eigen::ArrayXd dc_dvs;
    Eigen::ArrayXd dc_dvp;
    Eigen::ArrayXd dc_drho;
    Eigen::ArrayXd ur, uz, tr, tz;
    // Rayleigh eigenfunctions normally have uz(surface)=1. For modes with a
    // very weak surface amplitude, reconstructed fields use unit maximum
    // interface displacement instead; energy kernels are scale invariant.
    // Love uses the analogous surface/maximum displacement normalization.
    Eigen::ArrayXd uu, tt;
};

struct LayerModel {
    Eigen::ArrayXd thickness;
    Eigen::ArrayXd vp, vs, rho;
    int nlayer{};
    Eigen::ArrayXi iwat;
    bool allfluid = false;
};

class DispersionKernel {
public:
    explicit DispersionKernel(int nlayer);
    virtual ~DispersionKernel() = default;

    virtual KernelResult compute(double omega, double c) = 0;
    virtual void set_model(const VelocityModel& model);

protected:
    int nlayer_{};
    Eigen::ArrayXd thickness_, vp_, vs_, rho_;
    Eigen::ArrayXd mu_, lambda_;
    Eigen::ArrayXi iwat_;
    bool allfluid_ = false;
};

struct VarsvCacheEntry {
    double cosp, cosq, rsinp, rsinq, sinpr, sinqr, pex, svex;
};

struct RayleighState {
    Eigen::ArrayXd ur, uz, tr, tz;
    Eigen::ArrayXd dcda, dcdb, dcdr;
    Eigen::MatrixXd cd;
    Eigen::MatrixXd vv;
    Eigen::ArrayXd exe, exa;
    double sumi0 = 0.0, sumi1 = 0.0, sumi2 = 0.0, sumi3 = 0.0;
    double flagr = 0.0, are = 0.0, ugr = 0.0;
    std::complex<double> ra, rb;
    Eigen::Matrix<std::complex<double>, 4, 4> E, Einv;
    double uu0[4]{};
    std::vector<VarsvCacheEntry> varsv_cache;
    std::complex<double> FA_c, GA_c, FB_c, GB_c, H1_c, H2_c;
};

class RayleighKernel : public DispersionKernel {
public:
    using DispersionKernel::DispersionKernel;
    KernelResult compute(double omega, double c) override;
    void set_model(const VelocityModel& model) override;

private:
    RayleighState s_;
};

struct LoveState {
    Eigen::ArrayXd uu, tt;
    Eigen::ArrayXd dcdb, dcdr;
    Eigen::ArrayXd exl;
    double sumi0 = 0.0, sumi1 = 0.0, sumi2 = 0.0;
    double flagr = 0.0, ale = 0.0, ugr = 0.0;
};

class LoveKernel : public DispersionKernel {
public:
    using DispersionKernel::DispersionKernel;
    KernelResult compute(double omega, double c) override;
    void set_model(const VelocityModel& model) override;

private:
    LoveState ls_;
};

} // namespace fwd
