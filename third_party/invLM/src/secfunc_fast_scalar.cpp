#include "secfunc_fast_scalar.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace Eigen;

namespace fwd {

FastScalarSecularFunction::FastScalarSecularFunction(
    const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh)
    : SecularFunction(model, sh) {
  const int n = nl_;
  inv_vp2_.resize(n);
  inv_vs2_.resize(n);
  half_inv_vs2_.resize(n);
  l_.resize(n - 1);
  inv_l_.resize(n - 1);

  for (int i = 0; i < n; ++i) {
    const double vp2 = vp_(i) * vp_(i);
    const double vs2 = vs_(i) * vs_(i);
    inv_vp2_[i] = 1.0 / vp2;
    inv_vs2_[i] = 1.0 / vs2;
    half_inv_vs2_[i] = 0.5 / vs2;
  }

  for (int i = 0; i < n - 1; ++i) {
    const double mu_i = dns_(i) / inv_vs2_[i];
    const double mu_ip1 = dns_(i + 1) / inv_vs2_[i + 1];
    l_[i] = mu_ip1 / mu_i;
    inv_l_[i] = mu_i / mu_ip1;
  }
}

namespace {

inline void calc_branch(double eta, double k_thk,
                        double& a, double& c_over_g,
                        double& ysum) {
  if (eta == 0.0) {
    a = 1.0;
    c_over_g = k_thk;
    return;
  }

  const double ga = std::sqrt(std::abs(eta));
  const double z = ga * k_thk;
  const double inv_ga = 1.0 / ga;

  if (eta > 0.0) {
    a = std::cos(z);
    c_over_g = (std::abs(z) < 1.0e-8)
                   ? k_thk * (1.0 - z * z / 6.0)
                   : std::sin(z) * inv_ga;
    return;
  }

  ysum += z;

  if (z >= 50.0) {
    a = 0.5;
    c_over_g = 0.5 * inv_ga;
    return;
  }

  const double e2 = std::exp(-2.0 * z);
  a = 0.5 * (1.0 + e2);

  double sinh_scaled;
  if (z < 1.0e-5) {
    const double z2 = z * z;
    sinh_scaled = z - z2 + (2.0 / 3.0) * z2 * z;
  } else {
    sinh_scaled = 0.5 * (1.0 - e2);
  }

  c_over_g = sinh_scaled * inv_ga;
}

} // namespace

double FastScalarSecularFunction::evaluate(double f, double c) {
  if (sh_) {
    return evaluate_sh_impl(f, c);
  }

  if (!(f > 0.0) || !(c > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double omega = 2.0 * M_PI * f;
  const double wvno = omega / c;
  const double c2 = c * c;

  const int n = nl_ - 1;

  const double eta_p_n = c2 * inv_vp2_(n) - 1.0;
  const double eta_s_n = c2 * inv_vs2_(n) - 1.0;

  if (!(eta_p_n < 0.0) || !(eta_s_n < 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double alpha_p_n = std::sqrt(-eta_p_n);
  const double alpha_s_n = std::sqrt(-eta_s_n);
  const double t_n = 1.0 - c2 * half_inv_vs2_(n);
  const double g_n = c2 * half_inv_vs2_(n);

  const double ap_as = alpha_p_n * alpha_s_n;
  double x1 = 1.0 - ap_as;
  double x2 = t_n - ap_as;
  double x3 = -alpha_s_n * g_n;
  double x4 = alpha_p_n * g_n;
  double x5 = -t_n * t_n + ap_as;

  for (int m = nl_ - 2; m >= iwater_; --m) {
    const double t = 1.0 - c2 * half_inv_vs2_(m);
    const double g = c2 * half_inv_vs2_(m);
    const double k_thk = wvno * thk_(m);

    double ysum = 0.0;

    double a, c_c;
    calc_branch(c2 * inv_vp2_(m) - 1.0, k_thk, a, c_c, ysum);

    double b, c_d;
    calc_branch(c2 * inv_vs2_(m) - 1.0, k_thk, b, c_d, ysum);

    const double r = c2 * inv_vp2_(m) - 1.0;
    const double s_val = c2 * inv_vs2_(m) - 1.0;

    double Spq = 1.0;
    if (ysum > 0.0) {
      Spq = (ysum < 60.0) ? std::exp(-ysum) : 0.0;
    }

    x1 *= inv_l_(m);
    x5 *= l_(m);

    const double onept = 1.0 + t;
    const double twot = 2.0 * t;
    const double tt = t * t;

    const double p1 = -t * x1 + onept * x2 + x5;
    const double p2 = x1 - 2.0 * x2 - x5;
    const double p3 = g * x3;
    const double p4 = -g * x4;
    const double p5 = -tt * x1 + twot * x2 + x5;

    const double ab = a * b;
    const double cd = c_c * c_d;
    const double ad = a * c_d;
    const double bc = b * c_c;
    const double ad_s = ad * s_val;
    const double bc_r = bc * r;
    const double cd_s = cd * s_val;
    const double cd_r = cd * r;
    const double cd_rs = cd_r * s_val;

    const double q1 = ab * p2 + cd * p5 - ad * p3 - bc * p4;
    const double q2 = ad_s * p2 - bc * p5 + ab * p3 - cd_s * p4;
    const double q3 = bc_r * p2 - ad * p5 - cd_r * p3 + ab * p4;
    const double q4 = cd_rs * p2 + ab * p5 + bc_r * p3 + ad_s * p4;

    const double p1s = p1 * Spq;
    x1 = q1 - q4 + 2.0 * p1s;
    x2 = t * q1 - q4 + onept * p1s;
    x3 = g * q2;
    x4 = -g * q3;
    x5 = -tt * q1 + q4 - twot * p1s;

    // The compound state is homogeneous: positive rescaling preserves zeros
    // and signs. Without it, many thin layers introduce an enormous,
    // velocity-dependent amplitude envelope (and eventually underflow),
    // hiding close root pairs from the extremum search.
    const double scale = std::max({std::abs(x1), std::abs(x2),
        std::abs(x3), std::abs(x4), std::abs(x5)});
    if (scale > 0.0 && std::isfinite(scale)) {
      x1 /= scale; x2 /= scale; x3 /= scale; x4 /= scale; x5 /= scale;
    }
  }

  return x5;
}

} // namespace fwd
