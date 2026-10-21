#include "dispersion_kernel.hpp"

#include <cmath>
#include <complex>
#include <algorithm>
#include <Eigen/QR>
#include <Eigen/SparseLU>
#include <stdexcept>

using complex = std::complex<double>;
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Vector4d = Eigen::Matrix<double, 4, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5, Eigen::RowMajor>;
using Matrix4d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
using Matrix4c = Eigen::Matrix<complex, 4, 4, Eigen::RowMajor>;
using RowMatrix4c = Eigen::Matrix<complex, 4, 4>;

inline complex fast_csqrt(double x) {
  return (x >= 0.0) ? complex(std::sqrt(x), 0.0)
                    : complex(0.0, std::sqrt(-x));
}

namespace fwd {

DispersionKernel::DispersionKernel(int nlayer) : nlayer_(nlayer) {}

void DispersionKernel::set_model(const VelocityModel& model) {
  nlayer_ = model.nl;
  thickness_ = Eigen::ArrayXd(model.nl - 1);
  for (int i = 0; i < model.nl - 1; ++i)
    thickness_(i) = model.depth(i + 1) - model.depth(i);
  vp_ = model.vp.array();
  vs_ = model.vs.array();
  rho_ = model.rho.array();
  mu_ = rho_ * vs_.pow(2);
  lambda_ = rho_ * vp_.pow(2) - 2.0 * mu_;
  iwat_ = Eigen::ArrayXi::Zero(model.nl);
  if(model.nl>1 && vs_(0)==0)iwat_(0)=1;
  if((vs_<0).any() || (model.nl>1 && (vs_.tail(model.nl-1)<=0).any()) || vs_(model.nl-1)<=0)
    throw std::runtime_error("Only a surface water layer over solid layers is supported");
  allfluid_ = false;
}

void RayleighKernel::set_model(const VelocityModel& model) {
  DispersionKernel::set_model(model);
  const int n = nlayer_;
  s_.ur.resize(n);
  s_.uz.resize(n);
  s_.tr.resize(n);
  s_.tz.resize(n);
  s_.dcda.resize(n);
  s_.dcdb.resize(n);
  s_.dcdr.resize(n);
  s_.cd.resize(n, 5);
  s_.vv.resize(n, 4);
  s_.exe.resize(n);
  s_.exa.resize(n);
  s_.varsv_cache.resize(n);
}

void LoveKernel::set_model(const VelocityModel& model) {
  DispersionKernel::set_model(model);
  const int n = nlayer_;
  ls_.uu.resize(n);
  ls_.tt.resize(n);
  ls_.dcdb.resize(n);
  ls_.dcdr.resize(n);
  ls_.exl.resize(n);
}

namespace {

struct VarsvResult {
  double cosp, cosq, rsinp, rsinq, sinpr, sinqr, pex, svex;
};

VarsvResult varsv(complex p, complex q, complex rp, complex rsv,
                  int iwat, double zd) {
  VarsvResult r;
  r.pex = 0.0;
  r.svex = 0.0;

  double pr = p.real();
  double pi = p.imag();
  double qr = q.real();
  double qi = q.imag();
  r.pex = pr;

  if (iwat == 1) {
    complex epp = complex(std::cos(pi), std::sin(pi)) / 2.0;
    complex epm = std::conj(epp);
    double pfac = (pr < 30.0) ? std::exp(-2.0 * pr) : 0.0;
    r.cosp = (epp + pfac * epm).real();
    complex sinp = epp - pfac * epm;
    r.rsinp = (rp * sinp).real();
    if (std::abs(pr) < 1.0e-5 && std::abs(rp) < 1.0e-5) {
      r.sinpr = zd;
    } else {
      r.sinpr = (sinp / rp).real();
    }
    r.cosq = 1.0;
    r.rsinq = 0.0;
    r.sinqr = 0.0;
  } else {
    r.svex = qr;
    complex epp = complex(std::cos(pi), std::sin(pi)) / 2.0;
    complex epm = std::conj(epp);
    complex eqp = complex(std::cos(qi), std::sin(qi)) / 2.0;
    complex eqm = std::conj(eqp);

    double pfac = (pr < 30.0) ? std::exp(-2.0 * pr) : 0.0;
    r.cosp = (epp + pfac * epm).real();
    complex sinp = epp - pfac * epm;
    r.rsinp = (rp * sinp).real();
    if (std::abs(pr) < 1.0e-5 && std::abs(rp) < 1.0e-5) {
      r.sinpr = zd;
    } else {
      r.sinpr = (sinp / rp).real();
    }

    double svfac = (qr < 30.0) ? std::exp(-2.0 * qr) : 0.0;
    r.cosq = (eqp + svfac * eqm).real();
    complex sinq = eqp - svfac * eqm;
    r.rsinq = (rsv * sinq).real();
    if (std::abs(qr) < 1.0e-5 && std::abs(rsv) < 1.0e-5) {
      r.sinqr = zd;
    } else {
      r.sinqr = (sinq / rsv).real();
    }
  }
  return r;
}

template <int N>
double normc(Eigen::Matrix<double, N, 1>& ee) {
  double t1 = 0.0;
  for (int i = 0; i < N; ++i) {
    t1 = std::max(t1, std::abs(ee(i)));
  }
  if (t1 < 1.0e-40) t1 = 1.0;
  for (int i = 0; i < N; ++i) {
    ee(i) /= t1;
  }
  return std::log(t1);
}

complex ffunc(complex nub, double dm) {
  if (std::abs(nub) < 1.0e-08) return complex(dm, 0.0);
  complex argcd = nub * dm;
  complex exqq = (argcd.real() < 40.0) ? std::exp(-2.0 * argcd) : complex(0.0, 0.0);
  return (1.0 - exqq) / (2.0 * nub);
}

complex gfunc(complex nub, double dm) {
  complex argcd = nub * dm;
  return (argcd.real() < 75.0) ? std::exp(-argcd) * dm : complex(0.0, 0.0);
}

complex h1func(complex nua, complex nub, double dm) {
  if (std::abs(nub + nua) < 1.0e-08) return complex(dm, 0.0);
  complex argcd = (nua + nub) * dm;
  complex exqq = (argcd.real() < 40.0) ? std::exp(-argcd) : complex(0.0, 0.0);
  return (1.0 - exqq) / (nub + nua);
}

complex h2func(complex nua, complex nub, double dm) {
  if (std::abs(nub - nua) < 1.0e-08) return complex(dm, 0.0);
  complex argcd_p = nua * dm;
  complex exqp = (argcd_p.real() < 40.0) ? std::exp(-argcd_p) : complex(0.0, 0.0);
  complex argcd_q = nub * dm;
  complex exqq = (argcd_q.real() < 40.0) ? std::exp(-argcd_q) : complex(0.0, 0.0);
  return (exqq - exqp) / (nua - nub);
}

void dnka(Matrix5d& ca, double cosp, double rsinp, double sinpr,
          double cosq, double rsinq, double sinqr,
          double rho, double b, int iwat,
          double ex, double exa,
          double wvno, double wvno2, double om2) {
  if (iwat == 1) {
    ca.setZero();
    double dfac = (ex > 35.0) ? 0.0 : std::exp(-ex);
    ca(2, 2) = dfac;
    ca(0, 0) = cosp;
    ca(4, 4) = cosp;
    ca(0, 1) = -rsinp / (rho * om2);
    ca(1, 0) = -rho * sinpr * om2;
    ca(1, 1) = cosp;
    ca(3, 3) = cosp;
    ca(3, 4) = ca(0, 1);
    ca(4, 3) = ca(1, 0);
  } else {
    double a0 = (exa < 60.0) ? std::exp(-exa) : 0.0;
    double cpcq = cosp * cosq;
    double cpy = cosp * sinqr;
    double cpz = cosp * rsinq;
    double cqw = cosq * sinpr;
    double cqx = cosq * rsinp;
    double xy = rsinp * sinqr;
    double xz = rsinp * rsinq;
    double wy = sinpr * sinqr;
    double wz = sinpr * rsinq;
    double rho2 = rho * rho;
    double gam = 2.0 * b * b * wvno2 / om2;
    double gam2 = gam * gam;
    double gamm1 = gam - 1.0;
    double gamm2 = gamm1 * gamm1;
    double cqww2 = cqw * wvno2;
    double cqxw2 = cqx / wvno2;
    double gg1 = gam * gamm1;
    double a0c = 2.0 * (a0 - cpcq);
    double xz2 = xz / wvno2;
    double gxz2 = gam * xz2;
    double g2xz2 = gam2 * xz2;
    double a0cgg1 = a0c * (gam + gamm1);
    double wy2 = wy * wvno2;
    double g2wy2 = gamm2 * wy2;
    double g1wy2 = gamm1 * wy2;

    double temp = a0c * gg1 + g2xz2 + g2wy2;
    ca(2, 2) = a0 + temp + temp;
    ca(0, 0) = cpcq - temp;
    ca(0, 1) = (-cqx + wvno2 * cpy) / (rho * om2);
    ca(0, 2) = wvno * (0.5 * a0cgg1 + gxz2 + g1wy2) / (rho * om2);
    ca(0, 3) = (-cqww2 + cpz) / (rho * om2);
    ca(0, 4) = -(wvno2 * (a0c + wy2) + xz) / (rho2 * om2 * om2);

    ca(1, 0) = (-gamm2 * cqw + gam2 * cpz / wvno2) * rho * om2;
    ca(1, 1) = cpcq;
    ca(1, 2) = (gamm1 * cqww2 - gam * cpz) / wvno;
    ca(1, 3) = -wz;
    ca(1, 4) = ca(0, 3);

    temp = 0.5 * a0cgg1 * gg1 + gam2 * gxz2 + gamm2 * g1wy2;
    ca(2, 0) = -2.0 * temp * rho * om2 / wvno;
    ca(2, 1) = -wvno * (gam * cqxw2 - gamm1 * cpy) * 2.0;
    ca(2, 3) = -2.0 * ca(1, 2);
    ca(2, 4) = -2.0 * ca(0, 2);

    ca(3, 0) = (-gam2 * cqxw2 + gamm2 * cpy) * rho * om2;
    ca(3, 1) = -xy;
    ca(3, 2) = -ca(2, 1) / 2.0;
    ca(3, 3) = ca(1, 1);
    ca(3, 4) = ca(0, 1);

    temp = gamm2 * (a0c * gam2 + g2wy2) + gam2 * g2xz2;
    ca(4, 0) = -rho2 * om2 * om2 * temp / wvno2;
    ca(4, 1) = ca(3, 0);
    ca(4, 2) = -ca(2, 0) / 2.0;
    ca(4, 3) = ca(1, 0);
    ca(4, 4) = ca(0, 0);
  }
}

void hska(Matrix4d& aa, double cosp, double rsinp, double sinpr,
          double tcosq, double trsinq, double tsinqr,
          double rho, double b, int iwat,
          double pex, double svex,
          double wvno, double wvno2, double om2) {
  if (iwat == 1) {
    aa.setZero();
    double dfac = (pex > 35.0) ? 0.0 : std::exp(-pex);
    aa(0, 0) = dfac;
    aa(3, 3) = dfac;
    aa(1, 1) = cosp;
    aa(2, 2) = cosp;
    aa(1, 2) = -rsinp / (rho * om2);
    aa(2, 1) = -rho * om2 * sinpr;
  } else {
    double dfac = ((pex - svex) > 70.0) ? 0.0 : std::exp(svex - pex);
    double cosq = dfac * tcosq;
    double rsinsv = dfac * trsinq;
    double sinsvr = dfac * tsinqr;

    double gam = 2.0 * b * b * wvno2 / om2;
    double gamm1 = gam - 1.0;

    aa(0, 0) = cosq + gam * (cosp - cosq);
    aa(0, 1) = -wvno * gamm1 * sinpr + gam * rsinsv / wvno;
    aa(0, 2) = -wvno * (cosp - cosq) / (rho * om2);
    aa(0, 3) = (wvno2 * sinpr - rsinsv) / (rho * om2);

    aa(1, 0) = gam * rsinp / wvno - wvno * gamm1 * sinsvr;
    aa(1, 1) = cosp - gam * (cosp - cosq);
    aa(1, 2) = (-rsinp + wvno2 * sinsvr) / (rho * om2);
    aa(1, 3) = -aa(0, 2);

    aa(2, 0) = rho * om2 * gam * gamm1 * (cosp - cosq) / wvno;
    aa(2, 1) = rho * om2 * (-gamm1 * gamm1 * sinpr + gam * gam * rsinsv / wvno2);
    aa(2, 2) = aa(1, 1);
    aa(2, 3) = -aa(0, 1);

    aa(3, 0) = rho * om2 * (gam * gam * rsinp / wvno2 - gamm1 * gamm1 * sinsvr);
    aa(3, 1) = -aa(2, 0);
    aa(3, 2) = -aa(1, 0);
    aa(3, 3) = aa(0, 0);
  }
}

} // anonymous namespace

// RayleighKernel: evalg
static void evalg(RayleighState& s, int jbdry, int m,
                  const Eigen::ArrayXd& vp, const Eigen::ArrayXd& vs,
                  const Eigen::ArrayXd& rho, const Eigen::ArrayXi& iwat,
                  bool allfluid,
                  Eigen::Ref<Eigen::Matrix<complex, 2, 5>> gbr, int inp,
                  double wvno, double om, double om2, double wvno2) {
  double xka = om / vp(m);
  double xkb = (vs(m) > 0.0) ? om / vs(m) : 0.0;
  s.ra = fast_csqrt(wvno2 - xka * xka);
  s.rb = fast_csqrt(wvno2 - xkb * xkb);
  double gam = vs(m) * wvno / om;
  gam = 2.0 * gam * gam;
  double gamm1 = gam - 1.0;

  if (jbdry < 0) {
    if (vs(m) > 0.0) {
      gbr(inp, 0) = complex(1.0, 0.0);
      for (int j = 1; j < 5; ++j) gbr(inp, j) = complex(0.0, 0.0);
    } else {
      for (int j = 0; j < 5; ++j) gbr(inp, j) = complex(0.0, 0.0);
      if (allfluid)
        gbr(inp, 0) = complex(1.0, 0.0);
      else
        gbr(inp, 3) = complex(1.0, 0.0);
    }
  } else if (jbdry == 0) {
    if (iwat(m) == 0) {
      s.E(0, 0) = wvno;
      s.E(0, 1) = s.rb;
      s.E(0, 2) = wvno;
      s.E(0, 3) = -s.rb;
      s.E(1, 0) = s.ra;
      s.E(1, 1) = wvno;
      s.E(1, 2) = -s.ra;
      s.E(1, 3) = wvno;
      s.E(2, 0) = rho(m) * om2 * gamm1;
      s.E(2, 1) = rho(m) * om2 * gam * s.rb / wvno;
      s.E(2, 2) = rho(m) * om2 * gamm1;
      s.E(2, 3) = -rho(m) * om2 * gam * s.rb / wvno;
      s.E(3, 0) = rho(m) * om2 * gam * s.ra / wvno;
      s.E(3, 1) = rho(m) * om2 * gamm1;
      s.E(3, 2) = -rho(m) * om2 * gam * s.ra / wvno;
      s.E(3, 3) = rho(m) * om2 * gamm1;

      s.Einv(0, 0) = 0.5 * gam / wvno;
      s.Einv(0, 1) = -0.5 * gamm1 / s.ra;
      s.Einv(0, 2) = -0.5 / (rho(m) * om2);
      s.Einv(0, 3) = 0.5 * wvno / (rho(m) * om2 * s.ra);
      s.Einv(1, 0) = -0.5 * gamm1 / s.rb;
      s.Einv(1, 1) = 0.5 * gam / wvno;
      s.Einv(1, 2) = 0.5 * wvno / (rho(m) * om2 * s.rb);
      s.Einv(1, 3) = -0.5 / (rho(m) * om2);
      s.Einv(2, 0) = 0.5 * gam / wvno;
      s.Einv(2, 1) = 0.5 * gamm1 / s.ra;
      s.Einv(2, 2) = -0.5 / (rho(m) * om2);
      s.Einv(2, 3) = -0.5 * wvno / (rho(m) * om2 * s.ra);
      s.Einv(3, 0) = 0.5 * gamm1 / s.rb;
      s.Einv(3, 1) = 0.5 * gam / wvno;
      s.Einv(3, 2) = -0.5 * wvno / (rho(m) * om2 * s.rb);
      s.Einv(3, 3) = -0.5 / (rho(m) * om2);

      complex denom = -rho(m) * rho(m) * om2 * om2 * wvno2 * s.ra * s.rb;
      gbr(inp, 0) = 0.25 * rho(m) * rho(m) * om2 * om2 *
                     (-gam * gam * s.ra * s.rb + wvno2 * gamm1 * gamm1) / denom;
      gbr(inp, 1) = 0.25 * (-rho(m) * wvno2 * s.ra * om2) / denom;
      gbr(inp, 2) = 0.25 * (-rho(m) * (-gam * s.ra * s.rb + wvno2 * gamm1) *
                              om2 * wvno) / denom;
      gbr(inp, 3) = 0.25 * (rho(m) * wvno2 * s.rb * om2) / denom;
      gbr(inp, 4) = 0.25 * (wvno2 * (wvno2 - s.ra * s.rb)) / denom;
    } else {
      if (allfluid) {
        gbr(inp, 0) = 0.5 / s.ra;
        gbr(inp, 1) = 0.5 / (-rho(m) * om2);
        gbr(inp, 2) = complex(0.0, 0.0);
        gbr(inp, 3) = complex(0.0, 0.0);
        gbr(inp, 4) = complex(0.0, 0.0);
      } else {
        gbr(inp, 0) = complex(0.0, 0.0);
        gbr(inp, 1) = complex(0.0, 0.0);
        gbr(inp, 2) = complex(0.0, 0.0);
        gbr(inp, 3) = 0.5 * rho(m) * om2 / s.ra;
        gbr(inp, 4) = complex(-0.5, 0.0);
      }
      s.E.setZero();
      s.Einv.setZero();
      s.E(0, 0) = s.ra;
      s.E(0, 1) = -s.ra;
      s.E(1, 0) = -rho(m) * om2;
      s.E(1, 1) = -rho(m) * om2;
      s.Einv(0, 0) = 0.5 / s.ra;
      s.Einv(0, 1) = -0.5 / (rho(m) * om2);
      s.Einv(1, 0) = -0.5 / s.ra;
      s.Einv(1, 1) = -0.5 / (rho(m) * om2);
    }
  } else {
    if (vs(m) > 0.0) {
      for (int j = 0; j < 4; ++j) gbr(inp, j) = complex(0.0, 0.0);
      gbr(inp, 4) = complex(1.0, 0.0);
    } else {
      for (int j = 0; j < 5; ++j) gbr(inp, j) = complex(0.0, 0.0);
      if (allfluid)
        gbr(inp, 1) = complex(1.0, 0.0);
      else
        gbr(inp, 4) = complex(1.0, 0.0);
    }
  }
}

struct MatResult {
  double a12, a14, a21, a23;
  double ah, av, bh, bv, eta, rho;
  double TA, TC, TF, TL, TN;
};

static MatResult getmat(int m, double wvno, double om,
                        const Eigen::ArrayXd& vp,
                        const Eigen::ArrayXd& vs,
                        const Eigen::ArrayXd& rho,
                        int iwat) {
  MatResult r;
  if (iwat == 1) {
    r.ah = vp(m);
    r.av = vp(m);
    r.bh = 0.0;
    r.bv = 0.0;
    r.rho = rho(m);
    r.eta = 1.0;
    r.TL = 0.0;
    r.TN = 0.0;
    r.TC = rho(m) * vp(m) * vp(m);
    r.TA = rho(m) * vp(m) * vp(m);
    r.TF = r.TA - 2.0 * r.TN;
    r.a12 = -(wvno * wvno - om * om / (r.ah * r.ah)) / (r.rho * om * om);
    r.a14 = 0.0;
    r.a21 = 0.0;
    r.a23 = 0.0;
  } else {
    r.ah = vp(m);
    r.av = vp(m);
    r.bh = vs(m);
    r.bv = vs(m);
    r.rho = rho(m);
    r.eta = 1.0;
    r.TL = rho(m) * vs(m) * vs(m);
    r.TN = rho(m) * vs(m) * vs(m);
    r.TC = rho(m) * vp(m) * vp(m);
    r.TA = rho(m) * vp(m) * vp(m);
    r.TF = r.TA - 2.0 * r.TN;
    r.a12 = -wvno;
    r.a14 = 1.0 / r.TL;
    r.a21 = wvno * r.TF / r.TC;
    r.a23 = 1.0 / r.TC;
  }
  return r;
}

static void rayleigh_up(RayleighState& s, int nlayer,
                        const Eigen::ArrayXd& thickness,
                        const Eigen::ArrayXd& vp, const Eigen::ArrayXd& vs,
                        const Eigen::ArrayXd& rho, const Eigen::ArrayXi& iwat,
                        bool allfluid,
                        double omega, double wvno, double& fr) {
  double wvno2 = wvno * wvno;
  double om2 = omega * omega;

  Eigen::Matrix<complex, 2, 5> gbr;
  evalg(s, 0, nlayer - 1, vp, vs, rho, iwat, allfluid, gbr, 0,
        wvno, omega, om2, wvno2);

  for (int j = 0; j < 5; ++j) {
    s.cd(nlayer - 1, j) = gbr(0, j).real();
  }
  s.exe(nlayer - 1) = 0.0;

  double exsum = 0.0;
  for (int m = nlayer - 2; m >= 0; --m) {
    double xka = omega / vp(m);
    double xkb = (vs(m) > 0.0) ? omega / vs(m) : 0.0;
    complex rp = fast_csqrt(wvno2 - xka * xka);
    complex rsv = fast_csqrt(wvno2 - xkb * xkb);
    complex p = rp * thickness(m);
    complex q = rsv * thickness(m);

    auto v = varsv(p, q, rp, rsv, iwat(m), thickness(m));
    s.varsv_cache[m] = {v.cosp, v.cosq, v.rsinp, v.rsinq,
                        v.sinpr, v.sinqr, v.pex, v.svex};

    Matrix5d ca;
    dnka(ca, v.cosp, v.rsinp, v.sinpr, v.cosq, v.rsinq, v.sinqr,
         rho(m), vs(m), iwat(m), v.pex, v.pex + v.svex, wvno, wvno2, om2);

    Vector5d ee;
    for (int i = 0; i < 5; ++i) {
      double cr = 0.0;
      for (int j = 0; j < 5; ++j) {
        cr += s.cd(m + 1, j) * ca(j, i);
      }
      ee(i) = cr;
    }

    double exn = normc(ee);
    exsum += v.pex + v.svex + exn;
    s.exe(m) = exsum;

    for (int i = 0; i < 5; ++i) {
      s.cd(m, i) = ee(i);
    }
  }
  fr = s.cd(0, 0);
}

static void rayleigh_down(RayleighState& s, int nlayer,
                          const Eigen::ArrayXd& thickness,
                          const Eigen::ArrayXd& vp, const Eigen::ArrayXd& vs,
                          const Eigen::ArrayXd& rho, const Eigen::ArrayXi& iwat,
                          double omega, double wvno) {
  double om2 = omega * omega;
  double wvno2 = wvno * wvno;

  s.vv(0, 0) = 1.0;
  s.vv(0, 1) = 0.0;
  s.vv(0, 2) = 0.0;
  s.vv(0, 3) = 0.0;
  s.exa(0) = 0.0;

  double exsum = 0.0;
  for (int m = 0; m < nlayer - 1; ++m) {
    const auto& c = s.varsv_cache[m];
    VarsvResult v;
    v.cosp = c.cosp; v.cosq = c.cosq;
    v.rsinp = c.rsinp; v.rsinq = c.rsinq;
    v.sinpr = c.sinpr; v.sinqr = c.sinqr;
    v.pex = c.pex; v.svex = c.svex;

    Matrix4d aa;
    hska(aa, v.cosp, v.rsinp, v.sinpr, v.cosq, v.rsinq, v.sinqr,
         rho(m), vs(m), iwat(m), v.pex, v.svex, wvno, wvno2, om2);

    Vector4d aa0;
    for (int i = 0; i < 4; ++i) {
      double cc = 0.0;
      for (int j = 0; j < 4; ++j) {
        cc += aa(i, j) * s.vv(m, j);
      }
      aa0(i) = cc;
    }

    double ex2 = normc(aa0);
    exsum += v.pex + ex2;
    s.exa(m + 1) = exsum;

    for (int i = 0; i < 4; ++i) {
      s.vv(m + 1, i) = aa0(i);
    }
  }
}

static void rayleigh_svfunc(RayleighState& s, int nlayer,
                            const Eigen::ArrayXi& iwat, bool allfluid,
                            double omega, double wvno) {
  (void)omega; (void)wvno;
  s.ur(0) = s.cd(0, 2) / s.cd(0, 1);
  s.uz(0) = 1.0;
  s.tz(0) = 0.0;
  s.tr(0) = 0.0;
  s.uu0[0] = s.ur(0);
  s.uu0[1] = 1.0;
  // Matching the layer-recursion formulas at the free surface gives
  // uu1 = -cd(0,2), uu2 = -cd(0,1), uu3 = fr, uu4 = 0. To enforce uz(0)=1
  // consistently for all deeper layers, normalize by uu2 instead of the
  // secular function value fr, which vanishes at the dispersion root.
  s.uu0[2] = -s.cd(0, 1);
  s.uu0[3] = -s.cd(0, 1);

  for (int i = 1; i < nlayer; ++i) {
    double cd1 = s.cd(i, 0);
    double cd2 = s.cd(i, 1);
    double cd3 = s.cd(i, 2);
    double cd4 = -s.cd(i, 2);
    double cd5 = s.cd(i, 3);
    double cd6 = s.cd(i, 4);
    double tz1 = -s.vv(i, 3);
    double tz2 = -s.vv(i, 2);
    double tz3 = s.vv(i, 1);
    double tz4 = s.vv(i, 0);

    double uu1 = tz2 * cd6 - tz3 * cd5 + tz4 * cd4;
    double uu2 = -tz1 * cd6 + tz3 * cd3 - tz4 * cd2;
    double uu3 = tz1 * cd5 - tz2 * cd3 + tz4 * cd1;
    double uu4 = -tz1 * cd4 + tz2 * cd2 - tz3 * cd1;

    double ext = s.exa(i) + s.exe(i) - s.exe(0);
    if (ext > -80.0 && ext < 80.0) {
      double fact = std::exp(ext);
      s.ur(i) = uu1 * fact / s.uu0[2];
      s.uz(i) = uu2 * fact / s.uu0[2];
      s.tz(i) = uu3 * fact / s.uu0[2];
      s.tr(i) = uu4 * fact / s.uu0[2];
    } else {
      s.ur(i) = 0.0;
      s.uz(i) = 0.0;
      s.tz(i) = 0.0;
      s.tr(i) = 0.0;
    }
  }

  if (!allfluid) {
    int jwat = 0;
    for (int i = 0; i < nlayer; ++i) {
      if (iwat(i) > 0) {
        jwat = i + 1;
      } else {
        break;
      }
    }
    for (int i = 0; i < jwat; ++i) {
      s.ur(i) = 0.0;
      s.tr(i) = 0.0;
    }
  }
}

static double rayleigh_intijr(RayleighState& s, int ii, int jj, int m,
                              int typelyr,
                              int nlayer,
                              const Eigen::ArrayXd& thickness,
                              const Eigen::ArrayXd& vp, const Eigen::ArrayXd& vs,
                              const Eigen::ArrayXd& rho,
                              const Eigen::ArrayXi& iwat,
                              bool allfluid,
                              double om, double om2, double wvno, double wvno2) {
  (void)vp; (void)vs; (void)rho; (void)iwat; (void)allfluid;
  (void)om; (void)om2; (void)nlayer;

  complex cintijr;

  if (iwat(m) == 1) {
    if (typelyr < 0) {
      complex kmpu = s.Einv(0, 0) * s.uz(m + 1) + s.Einv(0, 1) * s.tz(m + 1);
      cintijr = s.E(ii, 0) * s.E(jj, 0) * kmpu * kmpu / (2.0 * s.ra);
    } else if (typelyr == 0) {
      complex km1pd = s.Einv(1, 0) * s.uz(m) + s.Einv(1, 1) * s.tz(m);
      complex kmpu = s.Einv(0, 0) * s.uz(m + 1) + s.Einv(0, 1) * s.tz(m + 1);
      cintijr = s.E(ii, 0) * s.E(jj, 0) * kmpu * kmpu * s.FA_c +
                (s.E(ii, 0) * s.E(jj, 1) + s.E(ii, 1) * s.E(jj, 0)) * kmpu * km1pd * s.GA_c +
                s.E(ii, 1) * s.E(jj, 1) * km1pd * km1pd * s.FA_c;
    } else {
      complex km1pd = s.Einv(1, 0) * s.uz(m) + s.Einv(1, 1) * s.tz(m);
      cintijr = s.E(ii, 1) * s.E(jj, 1) * km1pd * km1pd / (2.0 * s.ra);
    }
  } else {
    if (typelyr < 0) {
      complex kmpu = s.Einv(0, 0) * s.ur(m) + s.Einv(0, 1) * s.uz(m) +
                     s.Einv(0, 2) * s.tz(m) + s.Einv(0, 3) * s.tr(m);
      complex kmsu = s.Einv(1, 0) * s.ur(m) + s.Einv(1, 1) * s.uz(m) +
                     s.Einv(1, 2) * s.tz(m) + s.Einv(1, 3) * s.tr(m);
      cintijr = s.E(ii, 0) * s.E(jj, 0) * kmpu * kmpu / (2.0 * s.ra) +
                (s.E(ii, 0) * s.E(jj, 1) + s.E(ii, 1) * s.E(jj, 0)) * kmpu * kmsu / (s.ra + s.rb) +
                s.E(ii, 1) * s.E(jj, 1) * kmsu * kmsu / (2.0 * s.rb);
    } else if (typelyr == 0) {
      complex km1pd = s.Einv(2, 0) * s.ur(m) + s.Einv(2, 1) * s.uz(m) +
                      s.Einv(2, 2) * s.tz(m) + s.Einv(2, 3) * s.tr(m);
      complex km1sd = s.Einv(3, 0) * s.ur(m) + s.Einv(3, 1) * s.uz(m) +
                      s.Einv(3, 2) * s.tz(m) + s.Einv(3, 3) * s.tr(m);
      complex kmpu = s.Einv(0, 0) * s.ur(m + 1) + s.Einv(0, 1) * s.uz(m + 1) +
                     s.Einv(0, 2) * s.tz(m + 1) + s.Einv(0, 3) * s.tr(m + 1);
      complex kmsu = s.Einv(1, 0) * s.ur(m + 1) + s.Einv(1, 1) * s.uz(m + 1) +
                     s.Einv(1, 2) * s.tz(m + 1) + s.Einv(1, 3) * s.tr(m + 1);
      cintijr = s.E(ii, 0) * s.E(jj, 0) * kmpu * kmpu * s.FA_c +
                s.E(ii, 2) * s.E(jj, 2) * km1pd * km1pd * s.FA_c +
                s.E(ii, 1) * s.E(jj, 1) * kmsu * kmsu * s.FB_c +
                s.E(ii, 3) * s.E(jj, 3) * km1sd * km1sd * s.FB_c +
                s.H1_c * ((s.E(ii, 0) * s.E(jj, 1) + s.E(ii, 1) * s.E(jj, 0)) * kmpu * kmsu +
                          (s.E(ii, 2) * s.E(jj, 3) + s.E(ii, 3) * s.E(jj, 2)) * km1pd * km1sd) +
                s.H2_c * ((s.E(ii, 0) * s.E(jj, 3) + s.E(ii, 3) * s.E(jj, 0)) * kmpu * km1sd +
                          (s.E(ii, 1) * s.E(jj, 2) + s.E(ii, 2) * s.E(jj, 1)) * km1pd * kmsu) +
                s.GA_c * (s.E(ii, 0) * s.E(jj, 2) + s.E(ii, 2) * s.E(jj, 0)) * kmpu * km1pd +
                s.GB_c * (s.E(ii, 1) * s.E(jj, 3) + s.E(ii, 3) * s.E(jj, 1)) * kmsu * km1sd;
    } else {
      complex km1pd = s.Einv(2, 0) * s.ur(m) + s.Einv(2, 1) * s.uz(m) +
                      s.Einv(2, 2) * s.tz(m) + s.Einv(2, 3) * s.tr(m);
      complex km1sd = s.Einv(3, 0) * s.ur(m) + s.Einv(3, 1) * s.uz(m) +
                      s.Einv(3, 2) * s.tz(m) + s.Einv(3, 3) * s.tr(m);
      cintijr = s.E(ii, 2) * s.E(jj, 2) * km1pd * km1pd / (2.0 * s.ra) +
                (s.E(ii, 2) * s.E(jj, 3) + s.E(ii, 3) * s.E(jj, 2)) * km1pd * km1sd / (s.ra + s.rb) +
                s.E(ii, 3) * s.E(jj, 3) * km1sd * km1sd / (2.0 * s.rb);
    }
  }

  return cintijr.real();
}

static void rayleigh_energy(RayleighState& s, int nlayer,
                            const Eigen::ArrayXd& thickness,
                            const Eigen::ArrayXd& vp, const Eigen::ArrayXd& vs,
                            const Eigen::ArrayXd& rho,
                            const Eigen::ArrayXi& iwat,
                            bool allfluid,
                            double omega, double wvno) {
  double om2 = omega * omega;
  double wvno2 = wvno * wvno;
  double c = omega / wvno;

  s.sumi0 = 0.0;
  s.sumi1 = 0.0;
  s.sumi2 = 0.0;
  s.sumi3 = 0.0;

  for (int m = 0; m < nlayer; ++m) {
    auto mat = getmat(m, wvno, omega, vp, vs, rho, iwat(m));
    int typelyr = (m == nlayer - 1) ? 1 : 0;

    Eigen::Matrix<complex, 2, 5> gbr;
    evalg(s, 0, m, vp, vs, rho, iwat, allfluid, gbr, 0,
          wvno, omega, om2, wvno2);

    if (typelyr == 0) {
      s.FA_c = ffunc(s.ra, thickness(m));
      s.GA_c = gfunc(s.ra, thickness(m));
      s.FB_c = ffunc(s.rb, thickness(m));
      s.GB_c = gfunc(s.rb, thickness(m));
      s.H1_c = h1func(s.ra, s.rb, thickness(m));
      s.H2_c = h2func(s.ra, s.rb, thickness(m));
    }

    double INT11 = rayleigh_intijr(s, 0, 0, m, typelyr, nlayer,
                                   thickness, vp, vs, rho, iwat, allfluid,
                                   omega, om2, wvno, wvno2);
    double INT13 = rayleigh_intijr(s, 0, 2, m, typelyr, nlayer,
                                   thickness, vp, vs, rho, iwat, allfluid,
                                   omega, om2, wvno, wvno2);
    double INT22 = rayleigh_intijr(s, 1, 1, m, typelyr, nlayer,
                                   thickness, vp, vs, rho, iwat, allfluid,
                                   omega, om2, wvno, wvno2);
    double INT24 = rayleigh_intijr(s, 1, 3, m, typelyr, nlayer,
                                   thickness, vp, vs, rho, iwat, allfluid,
                                   omega, om2, wvno, wvno2);
    double INT33 = rayleigh_intijr(s, 2, 2, m, typelyr, nlayer,
                                   thickness, vp, vs, rho, iwat, allfluid,
                                   omega, om2, wvno, wvno2);
    double INT44 = rayleigh_intijr(s, 3, 3, m, typelyr, nlayer,
                                   thickness, vp, vs, rho, iwat, allfluid,
                                   omega, om2, wvno, wvno2);

    if (iwat(m) == 1) {
      double URUR = INT22 * std::pow(wvno / (rho(m) * om2), 2);
      double UZUZ = INT11;
      double URDUZ = -(wvno / (rho(m) * om2)) * mat.a12 * INT22;
      double DUZDUZ = mat.a12 * mat.a12 * INT22;
      s.sumi0 += rho(m) * (URUR + UZUZ);
      s.sumi1 += mat.TA * URUR;
      s.sumi2 -= mat.TF * URDUZ;
      s.sumi3 += mat.TC * DUZDUZ;

      double facah = rho(m) * mat.ah * (URUR - 2.0 * mat.eta * URDUZ / wvno);
      double facav = rho(m) * mat.av * DUZDUZ / wvno2;
      s.dcda(m) = facah + facav;
      s.dcdb(m) = 0.0; // Fluid Vs is fixed at zero, never an uninitialized kernel entry.

      double facr = -0.5 * c * c * (URUR + UZUZ);
      s.dcdr(m) = 0.5 * (mat.av * facav + mat.ah * facah) / rho(m) + facr;
      s.ur(m) = -wvno * s.tz(m) / (rho(m) * om2);
    } else {
      double URUR = INT11;
      double UZUZ = INT22;
      double DURDUR = mat.a12 * mat.a12 * INT22 + 2.0 * mat.a12 * mat.a14 * INT24 +
                      mat.a14 * mat.a14 * INT44;
      double DUZDUZ = mat.a21 * mat.a21 * INT11 + 2.0 * mat.a21 * mat.a23 * INT13 +
                      mat.a23 * mat.a23 * INT33;
      double URDUZ = mat.a21 * INT11 + mat.a23 * INT13;
      double UZDUR = mat.a12 * INT22 + mat.a14 * INT24;

      s.sumi0 += rho(m) * (URUR + UZUZ);
      s.sumi1 += mat.TL * UZUZ + mat.TA * URUR;
      s.sumi2 += mat.TL * UZDUR - mat.TF * URDUZ;
      s.sumi3 += mat.TL * DURDUR + mat.TC * DUZDUZ;

      double facah = rho(m) * mat.ah * (URUR - 2.0 * mat.eta * URDUZ / wvno);
      double facav = rho(m) * mat.av * DUZDUZ / wvno2;
      double facbv = rho(m) * mat.bv * (UZUZ + 2.0 * UZDUR / wvno + DURDUR / wvno2 +
                                          4.0 * mat.eta * URDUZ / wvno);
      s.dcda(m) = facah + facav;
      s.dcdb(m) = facbv;

      double facr = -0.5 * c * c * (URUR + UZUZ);
      s.dcdr(m) = 0.5 * (mat.av * facav + mat.ah * facah + mat.bv * facbv) / rho(m) + facr;
    }
  }

  s.flagr = om2 * s.sumi0 - wvno2 * s.sumi1 - 2.0 * wvno * s.sumi2 - s.sumi3;
  s.ugr = (wvno * s.sumi1 + s.sumi2) / (omega * s.sumi0);
  s.are = wvno / (2.0 * omega * s.ugr * s.sumi0);

  for (int m = 0; m < nlayer; ++m) {
    s.dcda(m) /= (s.ugr * s.sumi0);
    s.dcdb(m) /= (s.ugr * s.sumi0);
    s.dcdr(m) /= (s.ugr * s.sumi0);
  }
}

// A two-point boundary solve avoids propagating a tiny surface amplitude across
// an evanescent cap. Upgoing amplitudes are referenced to the layer bottom,
// downgoing amplitudes to its top: all exponential magnitudes are <= 1.
static void rayleigh_boundary_eigenfunctions(
    RayleighState& s, int n, const Eigen::ArrayXd& thickness,
    const Eigen::ArrayXd& vp, const Eigen::ArrayXd& vs,
    const Eigen::ArrayXd& rho, const Eigen::ArrayXi& iwat,
    double omega, double k, bool dense_qr = false) {
  if(n<2) throw std::runtime_error("Rayleigh boundary reconstruction needs finite layers");
  const bool water=iwat(0)!=0;
  const int count=4*(n-1)+2-(water?2:0);
  auto offset=[&](int m){return 4*m-(water && m>0?2:0);};
  std::vector<Matrix4c> tops(n), bottoms(n);
  for(int m=0;m<n;++m) {
    Eigen::Matrix<complex,2,5> unused;
    evalg(s,0,m,vp,vs,rho,iwat,false,unused,0,k,omega,omega*omega,k*k);
    tops[m]=s.E;bottoms[m]=s.E;
    if(iwat(m)) {
      // Acoustic P waves only. No tangential-displacement continuity is
      // imposed at the water/solid boundary (slip is physically allowed).
      tops[m].setZero();
      tops[m](0,0)=k;tops[m](0,1)=k;
      tops[m](1,0)=s.ra;tops[m](1,1)=-s.ra;
      tops[m](2,0)=-rho(m)*omega*omega;tops[m](2,1)=-rho(m)*omega*omega;
      bottoms[m]=tops[m];
      const complex ep=std::exp(-s.ra*thickness(m));
      tops[m].col(0)*=ep;bottoms[m].col(1)*=ep;
      continue;
    }
    if(m<n-1) {
      const complex ep=std::exp(-s.ra*thickness(m));
      const complex es=std::exp(-s.rb*thickness(m));
      tops[m].col(0)*=ep;tops[m].col(1)*=es;
      bottoms[m].col(2)*=ep;bottoms[m].col(3)*=es;
    }
  }
  // Only adjacent layers couple. Assemble and equilibrate in sparse form;
  // allocating a dense matrix here would retain quadratic work/storage.
  std::vector<Eigen::Triplet<complex>> entries;
  entries.reserve(32*n);
  auto block=[&](int row,int col,const auto& values) {
    for(int j=0;j<values.cols();++j) for(int i=0;i<values.rows();++i) {
      const complex value=values(i,j);
      if(!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::runtime_error("Non-finite Rayleigh boundary matrix");
      if(value!=complex(0)) entries.emplace_back(row+i,col+j,value);
    }
  };
  if(water)block(0,0,tops[0].block(2,0,1,2));
  else block(0,0,tops[0].bottomRows(2));
  for(int m=0;m<n-1;++m) {
    if(water && m==0) {
      const int first=n==2?2:0,cols=n==2?2:4;
      block(1,0,bottoms[0].block(1,0,2,2));
      block(1,offset(1),-tops[1].block(1,first,2,cols));
      block(3,offset(1),tops[1].block(3,first,1,cols)); // solid shear traction = 0
    } else {
      const int row=2+4*m-(water?2:0);
      block(row,offset(m),bottoms[m]);
      if(m+1<n-1) block(row,offset(m+1),-tops[m+1]);
      else block(row,offset(m+1),-tops[m+1].rightCols(2));
    }
  }
  Eigen::SparseMatrix<complex> a(count,count);
  a.setFromTriplets(entries.begin(),entries.end());
  Eigen::VectorXd row_norm=Eigen::VectorXd::Zero(count);
  for(int j=0;j<count;++j)
    for(Eigen::SparseMatrix<complex>::InnerIterator it(a,j);it;++it)
      row_norm(it.row())+=std::norm(it.value());
  row_norm=row_norm.array().sqrt();
  if(!row_norm.allFinite() || (row_norm.array()<=0).any())
    throw std::runtime_error("Degenerate Rayleigh boundary row");
  for(int j=0;j<count;++j) {
    for(Eigen::SparseMatrix<complex>::InnerIterator it(a,j);it;++it)
      it.valueRef()/=row_norm(it.row());
  }
  Eigen::VectorXd scale(count);
  for(int i=0;i<count;++i) {
    scale(i)=a.col(i).norm();
    if(!(scale(i)>0) || !std::isfinite(scale(i)))
      throw std::runtime_error("Degenerate Rayleigh boundary column");
    for(Eigen::SparseMatrix<complex>::InnerIterator it(a,i);it;++it)
      it.valueRef()/=scale(i);
  }
  Eigen::VectorXcd v;
  if(dense_qr) {
    // Preserve the validated rank-revealing solve as a last-resort path.
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> qr{Eigen::MatrixXcd(a)};
    Eigen::VectorXcd permuted=Eigen::VectorXcd::Zero(count);
    permuted(count-1)=1.0;
    permuted.head(count-1)=qr.matrixR().topLeftCorner(count-1,count-1)
        .triangularView<Eigen::Upper>().solve(-qr.matrixR().col(count-1).head(count-1));
    v=(qr.colsPermutation()*permuted).normalized();
  } else {
    // At an isolated dispersion root A has a near-null direction. Fixed-count
    // inverse iteration extracts it with one pivoted sparse LU factorization;
    // no secular evaluations or changes to the root are involved.
    Eigen::SparseLU<Eigen::SparseMatrix<complex>> lu;
    lu.compute(a);
    if(lu.info()!=Eigen::Success)
      throw std::runtime_error("Rayleigh boundary LU factorization failed");
    v=Eigen::VectorXcd::Ones(count).normalized();
    for(int iteration=0;iteration<3;++iteration) {
      v=lu.solve(v).eval().normalized();
      if(lu.info()!=Eigen::Success || !v.allFinite())
        throw std::runtime_error("Rayleigh boundary inverse iteration failed");
    }
  }
  if((a*v).norm()>1e-7 || !v.allFinite())
    throw std::runtime_error("Rayleigh boundary eigenfunction residual is too large");
  const Eigen::VectorXcd coefficients=v.array()/scale.array();
  Eigen::MatrixXcd fields(n,4);
  for(int m=0;m<n;++m) {
    if(water && m==0)fields.row(m)=(tops[m].leftCols(2)*coefficients.head(2)).transpose();
    else if(m<n-1) fields.row(m)=(tops[m]*coefficients.segment(offset(m),4)).transpose();
    else fields.row(m)=(tops[m].rightCols(2)*coefficients.tail(2)).transpose();
  }
  Eigen::Index row,col;
  fields.leftCols(2).cwiseAbs().maxCoeff(&row,&col);
  complex reference=fields(row,col);
  if(!(std::abs(reference)>0)) throw std::runtime_error("Zero Rayleigh boundary displacement");
  if(std::abs(fields(0,1))>1e-6*std::abs(reference))reference=fields(0,1);
  fields/=reference;
  if(fields.imag().norm()>1e-6*fields.real().norm())
    throw std::runtime_error("Rayleigh boundary eigenfunction is not real");
  s.ur=fields.col(0).real().array();s.uz=fields.col(1).real().array();
  s.tz=fields.col(2).real().array();s.tr=fields.col(3).real().array();
}

KernelResult RayleighKernel::compute(double omega, double c) {
  double wvno = omega / c;
  const int nl = nlayer_;

  double fr;
  rayleigh_up(s_, nl, thickness_, vp_, vs_, rho_, iwat_, allfluid_,
              omega, wvno, fr);
  rayleigh_down(s_, nl, thickness_, vp_, vs_, rho_, iwat_,
                omega, wvno);
  rayleigh_svfunc(s_, nl, iwat_, allfluid_, omega, wvno);
  rayleigh_energy(s_, nl, thickness_, vp_, vs_, rho_, iwat_, allfluid_,
                  omega, wvno);

  auto energy_residual = [&]() {
    const double scale=std::abs(omega*omega*s_.sumi0)
        +std::abs(wvno*wvno*s_.sumi1)+std::abs(2*wvno*s_.sumi2)+std::abs(s_.sumi3);
    return std::abs(s_.flagr)/scale;
  };
  bool stabilized=false;
  if ((!std::isfinite(energy_residual()) || energy_residual()>1e-7)
      && ((vs_>0.0).all() || (iwat_(0) && (vs_.tail(nl-1)>0).all()))) {
    auto reconstruct=[&](bool dense_qr) {
      rayleigh_boundary_eigenfunctions(s_,nl,thickness_,vp_,vs_,rho_,iwat_,omega,wvno,dense_qr);
      rayleigh_energy(s_,nl,thickness_,vp_,vs_,rho_,iwat_,allfluid_,omega,wvno);
      if(!std::isfinite(energy_residual()) || energy_residual()>1e-7)
        throw std::runtime_error("Rayleigh reconstructed eigenfunction fails energy balance");
    };
    try { reconstruct(false); }
    catch(const std::runtime_error&) { reconstruct(true); }
    stabilized=true;
  }

  KernelResult result;
  result.stabilized_eigenfunctions=stabilized;
  result.energy_residual=energy_residual();
  result.group_velocity = s_.ugr;
  result.dc_dvs = s_.dcdb;
  result.dc_dvp = s_.dcda;
  result.dc_drho = s_.dcdr;
  result.ur = s_.ur;
  result.uz = s_.uz;
  result.tr = s_.tr;
  result.tz = s_.tz;
  return result;
}

// LoveKernel
using Matrix2c = Eigen::Matrix<complex, 2, 2>;

namespace {

struct VarlResult {
  double cosq, sinq, yl, zl, mu, rb, xkb, eexl;
};

VarlResult varl(int m, double omega, double wvno, double dpth,
                const Eigen::ArrayXd& vs, const Eigen::ArrayXd& rho) {
  VarlResult r;
  r.xkb = omega / vs(m);
  double wvnop = wvno + r.xkb;
  double wvnom = std::abs(wvno - r.xkb);
  r.rb = std::sqrt(wvnop * wvnom);
  double q = r.rb * dpth;
  r.mu = rho(m) * vs(m) * vs(m);
  r.eexl = 0.0;

  if (wvno < r.xkb) {
    r.sinq = std::sin(q);
    r.yl = r.sinq / r.rb;
    r.zl = -r.rb * r.sinq;
    r.cosq = std::cos(q);
  } else if (wvno == r.xkb) {
    r.cosq = 1.0;
    r.yl = dpth;
    r.zl = 0.0;
    r.sinq = 0.0;
  } else {
    r.eexl = q;
    double fac = (q < 18.0) ? std::exp(-2.0 * q) : 0.0;
    r.cosq = (1.0 + fac) * 0.5;
    r.sinq = (1.0 - fac) * 0.5;
    r.yl = r.sinq / r.rb;
    r.zl = r.rb * r.sinq;
  }
  return r;
}

void hskl(double hl[2][2], double cosq, double yl, double zl, double mu,
          int iwat) {
  if (iwat == 0) {
    hl[0][0] = cosq;
    hl[0][1] = yl / mu;
    hl[1][0] = zl * mu;
    hl[1][1] = cosq;
  } else {
    hl[0][0] = 1.0;
    hl[0][1] = 0.0;
    hl[1][0] = 0.0;
    hl[1][1] = 1.0;
  }
}

void emat(complex xnub, double wvno, Matrix2c& el, Matrix2c& einvl) {
  einvl(0, 0) = 0.5 / wvno;
  einvl(0, 1) = 0.5 / (wvno * xnub);
  einvl(1, 0) = 0.5 / wvno;
  einvl(1, 1) = -0.5 / (wvno * xnub);
  el(0, 0) = wvno;
  el(0, 1) = wvno;
  el(1, 0) = wvno * xnub;
  el(1, 1) = -wvno * xnub;
}

} // anonymous namespace

static void love_up(LoveState& s, int nlayer,
                    const Eigen::ArrayXd& thickness,
                    const Eigen::ArrayXd& vs, const Eigen::ArrayXd& rho,
                    const Eigen::ArrayXi& iwat,
                    const Eigen::ArrayXd& mu,
                    double omega, double wvno, double& fl) {
  if (vs(nlayer - 1) > 0.01) {
    double dpth = 0.0;
    auto v = varl(nlayer - 1, omega, wvno, dpth, vs, rho);
    s.uu(nlayer - 1) = 1.0;
    s.tt(nlayer - 1) = -mu(nlayer - 1) * v.rb;
  } else {
    s.uu(nlayer - 1) = 1.0;
    s.tt(nlayer - 1) = 0.0;
  }
  s.exl(nlayer - 1) = 0.0;

  for (int k = nlayer - 2; k >= 0; --k) {
    if (iwat(k) == 0) {
      double dpth = thickness(k);
      auto v = varl(k, omega, wvno, dpth, vs, rho);

      double hl[2][2];
      hskl(hl, v.cosq, v.yl, v.zl, v.mu, iwat(k));

      double a11 = hl[0][0];
      double a22 = hl[1][1];
      double a12 = -hl[0][1];
      double a21 = -hl[1][0];

      double amp0 = a11 * s.uu(k + 1) + a12 * s.tt(k + 1);
      double str0 = a21 * s.uu(k + 1) + a22 * s.tt(k + 1);

      double rr = std::max(std::abs(amp0), std::abs(str0));
      if (rr < 1.0e-30) rr = 1.0;
      s.exl(k) = std::log(rr) + v.eexl;
      s.uu(k) = amp0 / rr;
      s.tt(k) = str0 / rr;
    }
  }
  fl = s.tt(0);
}

static void love_shfunc(LoveState& s, int nlayer,
                        const Eigen::ArrayXd& vs, const Eigen::ArrayXd& rho,
                        const Eigen::ArrayXd& mu,
                        const Eigen::ArrayXd& thickness,
                        const Eigen::ArrayXi& iwat,
                        double omega, double wvno) {
  double fl;
  love_up(s, nlayer, thickness, vs, rho, iwat, mu, omega, wvno, fl);

  double umax = s.uu(0);
  s.tt(0) = 0.0;
  for (int k = 1; k < nlayer; ++k) {
    if (iwat(k) == 0) {
      double ext = 0.0;
      for (int j = 0; j < k; ++j) ext += s.exl(j);
      double fact = (ext < 80.0) ? 1.0 / std::exp(ext) : 0.0;
      s.uu(k) *= fact;
      s.tt(k) *= fact;
    } else {
      s.uu(k) = 0.0;
      s.tt(k) = 0.0;
    }
    if (std::abs(s.uu(k)) > std::abs(umax)) {
      umax = s.uu(k);
    }
  }
  if (s.uu(0) != 0.0) umax = s.uu(0);
  if (std::abs(umax) > 0.0) {
    for (int k = 0; k < nlayer; ++k) {
      if (iwat(k) == 0) {
        s.uu(k) /= umax;
        s.tt(k) /= umax;
      }
    }
  }
}

static void love_energy(LoveState& s, int nlayer,
                        const Eigen::ArrayXd& thickness,
                        const Eigen::ArrayXd& vs, const Eigen::ArrayXd& rho,
                        const Eigen::ArrayXi& iwat,
                        const Eigen::ArrayXd& mu,
                        double omega, double wvno) {
  double c = omega / wvno;
  double omega2 = omega * omega;
  double wvno2 = wvno * wvno;

  s.sumi0 = 0.0;
  s.sumi1 = 0.0;
  s.sumi2 = 0.0;

  for (int k = 0; k < nlayer; ++k) {
    if (iwat(k) == 0) {
      double TN = rho(k) * vs(k) * vs(k);
      double TL = rho(k) * vs(k) * vs(k);
      double VSHH = vs(k);
      double VSHV = vs(k);

      double dmu = mu(k);

      double upup, dupdup;

      if (k == nlayer - 1) {
        auto v = varl(k, omega, wvno, 0.0, vs, rho);
        double rb = v.rb;
        if (rb < 1.0e-10) rb = 1.0e-10;
        upup = 0.5 / rb * s.uu(nlayer - 1) * s.uu(nlayer - 1);
        dupdup = 0.5 * rb * s.uu(nlayer - 1) * s.uu(nlayer - 1);
      } else {
        double dpth = thickness(k);
        auto v = varl(k, omega, wvno, dpth, vs, rho);
        double rb = v.rb;
        if (rb < 1.0e-10) rb = 1.0e-10;
        complex nub(rb, 0.0);
        if (wvno < v.xkb) nub = complex(0.0, rb);
        complex xnub = dmu * nub;

        Matrix2c el, einvl;
        emat(xnub, wvno, el, einvl);

        complex km1dn = einvl(1, 0) * s.uu(k) + einvl(1, 1) * s.tt(k);
        complex kmup = einvl(0, 0) * s.uu(k + 1) + einvl(0, 1) * s.tt(k + 1);

        complex f3 = nub * dpth;
        complex exqq = (f3.real() < 40.0) ? std::exp(-2.0 * f3) : complex(0.0, 0.0);
        complex f = (1.0 - exqq) / (2.0 * nub);

        complex exqq2 = (f3.real() < 75.0) ? std::exp(-f3) : complex(0.0, 0.0);
        complex g = dpth * exqq2;

        complex f1 = f * (el(0, 0) * el(0, 0) * kmup * kmup +
                          el(0, 1) * el(0, 1) * km1dn * km1dn);
        complex f2 = g * (el(0, 0) * el(0, 1) + el(0, 0) * el(0, 1)) * kmup * km1dn;

        upup = (f1 + f2).real();
        dupdup = (nub * nub * (f1 - f2)).real();
      }

      s.sumi0 += rho(k) * upup;
      s.sumi1 += TN * upup;
      s.sumi2 += TL * dupdup;

      double DCDBH = c * rho(k) * VSHH * upup;
      double DCDBV = c * rho(k) * VSHV * dupdup / wvno2;
      s.dcdb(k) = DCDBH + DCDBV;

      double DCDRSH = 0.5 * c * (-c * c * upup + VSHH * VSHH * upup +
                                   VSHV * VSHV * dupdup / wvno2);
      s.dcdr(k) = DCDRSH;
    } else {
      s.dcdb(k) = 0.0;
      s.dcdr(k) = 0.0;
    }
  }

  for (int k = 0; k < nlayer; ++k) {
    if (iwat(k) == 0) {
      s.dcdb(k) /= s.sumi1;
      s.dcdr(k) /= s.sumi1;
    }
  }

  s.flagr = omega2 * s.sumi0 - wvno2 * s.sumi1 - s.sumi2;
  s.ugr = s.sumi1 / (c * s.sumi0);
  s.ale = 0.5 / s.sumi1;
}

// SH counterpart of the solid Rayleigh two-point reconstruction. Referencing
// upgoing waves to the bottom and downgoing waves to the top prevents growth
// across an evanescent cap; only neighboring layers enter the sparse system.
static void love_boundary_eigenfunctions(
    LoveState& s, int n, const Eigen::ArrayXd& thickness,
    const Eigen::ArrayXd& vs, const Eigen::ArrayXd& mu,
    double omega, double k, bool dense_qr) {
  if(n<2 || !(k>omega/vs(n-1)))
    throw std::runtime_error("Love reconstruction requires a trapped solid mode");
  const int count=2*n-1;
  std::vector<Matrix2c> tops(n), bottoms(n);
  for(int m=0;m<n;++m) {
    const complex q=std::sqrt(complex(k*k-omega*omega/(vs(m)*vs(m)),0));
    tops[m]<<1.0,1.0,mu(m)*q,-mu(m)*q;
    bottoms[m]=tops[m];
    if(m<n-1) {
      const complex e=std::exp(-q*thickness(m));
      tops[m].col(0)*=e;
      bottoms[m].col(1)*=e;
    }
  }
  std::vector<Eigen::Triplet<complex>> entries;
  entries.reserve(8*n);
  auto block=[&](int row,int col,const auto& values) {
    for(int j=0;j<values.cols();++j) for(int i=0;i<values.rows();++i) {
      const complex value=values(i,j);
      if(!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::runtime_error("Non-finite Love boundary matrix");
      if(value!=complex(0))entries.emplace_back(row+i,col+j,value);
    }
  };
  block(0,0,tops[0].bottomRows(1));
  for(int m=0;m<n-1;++m) {
    block(1+2*m,2*m,bottoms[m]);
    if(m+1<n-1)block(1+2*m,2*(m+1),-tops[m+1]);
    else block(1+2*m,2*(m+1),-tops[m+1].rightCols(1));
  }
  Eigen::SparseMatrix<complex> a(count,count);
  a.setFromTriplets(entries.begin(),entries.end());
  Eigen::VectorXd rows=Eigen::VectorXd::Zero(count),scale(count);
  for(int j=0;j<count;++j)
    for(Eigen::SparseMatrix<complex>::InnerIterator it(a,j);it;++it)
      rows(it.row())+=std::norm(it.value());
  rows=rows.array().sqrt();
  if(!rows.allFinite() || (rows.array()<=0).any())
    throw std::runtime_error("Degenerate Love boundary row");
  for(int j=0;j<count;++j) {
    for(Eigen::SparseMatrix<complex>::InnerIterator it(a,j);it;++it)
      it.valueRef()/=rows(it.row());
    scale(j)=a.col(j).norm();
    if(!(scale(j)>0) || !std::isfinite(scale(j)))
      throw std::runtime_error("Degenerate Love boundary column");
    for(Eigen::SparseMatrix<complex>::InnerIterator it(a,j);it;++it)
      it.valueRef()/=scale(j);
  }
  Eigen::VectorXcd v;
  if(dense_qr) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> qr{Eigen::MatrixXcd(a)};
    Eigen::VectorXcd p=Eigen::VectorXcd::Zero(count);p(count-1)=1;
    p.head(count-1)=qr.matrixR().topLeftCorner(count-1,count-1)
      .triangularView<Eigen::Upper>().solve(-qr.matrixR().col(count-1).head(count-1));
    v=(qr.colsPermutation()*p).normalized();
  } else {
    Eigen::SparseLU<Eigen::SparseMatrix<complex>> lu;lu.compute(a);
    if(lu.info()!=Eigen::Success)throw std::runtime_error("Love boundary LU failed");
    v=Eigen::VectorXcd::Ones(count).normalized();
    for(int i=0;i<3;++i) {
      v=lu.solve(v).eval().normalized();
      if(lu.info()!=Eigen::Success || !v.allFinite())
        throw std::runtime_error("Love boundary inverse iteration failed");
    }
  }
  if(!v.allFinite() || (a*v).norm()>1e-7)
    throw std::runtime_error("Love boundary residual is too large");
  const Eigen::VectorXcd coeff=v.array()/scale.array();
  Eigen::MatrixXcd fields(n,2);
  for(int m=0;m<n-1;++m)fields.row(m)=(tops[m]*coeff.segment(2*m,2)).transpose();
  fields.row(n-1)=(tops[n-1].col(1)*coeff(count-1)).transpose();
  Eigen::Index row;fields.col(0).cwiseAbs().maxCoeff(&row);
  complex reference=fields(row,0);
  if(!(std::abs(reference)>0))throw std::runtime_error("Zero Love displacement");
  if(std::abs(fields(0,0))>1e-6*std::abs(reference))reference=fields(0,0);
  fields/=reference;
  if(!fields.allFinite() || fields.imag().norm()>1e-6*fields.real().norm())
    throw std::runtime_error("Love boundary eigenfunction is not real");
  s.uu=fields.col(0).real().array();s.tt=fields.col(1).real().array();
}

KernelResult LoveKernel::compute(double omega, double c) {
  double wvno = omega / c;
  const int nl = nlayer_;
  if(iwat_(0)) {
    // Ideal water carries no SH traction: use the solid subsystem, including
    // its stabilized reconstruction, then pad the inactive fluid entries.
    VelocityModel solid;solid.nl=nl-1;
    solid.depth=Eigen::VectorXd::Zero(nl-1);
    for(int i=1;i<nl-1;++i)solid.depth(i)=solid.depth(i-1)+thickness_(i);
    solid.vs=vs_.tail(nl-1);solid.vp=vp_.tail(nl-1);solid.rho=rho_.tail(nl-1);
    LoveKernel kernel(nl-1);kernel.set_model(solid);
    auto result=kernel.compute(omega,c);
    auto pad=[&](Eigen::ArrayXd& a){Eigen::ArrayXd full=Eigen::ArrayXd::Zero(nl);full.tail(nl-1)=a;a=full;};
    pad(result.dc_dvs);pad(result.dc_dvp);pad(result.dc_drho);pad(result.uu);pad(result.tt);
    return result;
  }

  love_shfunc(ls_, nl, vs_, rho_, mu_, thickness_, iwat_, omega, wvno);
  love_energy(ls_, nl, thickness_, vs_, rho_, iwat_, mu_, omega, wvno);

  auto energy_residual=[&]() {
    const double scale=std::abs(omega*omega*ls_.sumi0)
      +std::abs(wvno*wvno*ls_.sumi1)+std::abs(ls_.sumi2);
    return std::abs(ls_.flagr)/scale;
  };
  bool stabilized=false;
  if((vs_>0).all() && (!std::isfinite(energy_residual()) || energy_residual()>1e-7)) {
    auto reconstruct=[&](bool dense_qr) {
      love_boundary_eigenfunctions(ls_,nl,thickness_,vs_,mu_,omega,wvno,dense_qr);
      love_energy(ls_,nl,thickness_,vs_,rho_,iwat_,mu_,omega,wvno);
      if(!std::isfinite(energy_residual()) || energy_residual()>1e-7)
        throw std::runtime_error("Love reconstructed eigenfunction fails energy balance");
    };
    try {reconstruct(false);}catch(const std::runtime_error&) {reconstruct(true);}
    stabilized=true;
  }
  KernelResult result;
  result.stabilized_eigenfunctions=stabilized;
  result.energy_residual=energy_residual();
  result.group_velocity = ls_.ugr;
  result.dc_dvs = ls_.dcdb;
  result.dc_dvp = Eigen::ArrayXd::Zero(nl);
  result.dc_drho = ls_.dcdr;
  result.uu = ls_.uu;
  result.tt = ls_.tt;
  return result;
}

} // namespace fwd
