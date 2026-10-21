#include "secfunc.hpp"
#include "secfunc_fast_scalar.hpp"

#include <Eigen/Dense>
#include <complex>

using namespace Eigen;
using namespace std::complex_literals;
using Cmplx = std::complex<double>;

namespace fwd {

SecularFunction::SecularFunction(const Eigen::Ref<const Eigen::ArrayXXd> model,
                                 bool sh)
    : nl_(model.rows()), thk_(nl_ - 1), dns_(model.col(2)), vs_(model.col(3)),
      vp_(model.col(4)), sh_(sh), is_water_(vs_(0) == 0.0) {
  ArrayXd z = model.col(1);
  if(nl_<1 || !vs_.allFinite() || (vs_<0).any()
     || vs_(nl_-1)<=0 || (nl_>1 && (vs_.tail(nl_-1)<=0).any()))
    throw std::invalid_argument("Only a surface water layer over solid layers is supported");
  if(is_water_ && (!std::isfinite(vp_(0)) || vp_(0)<=0 || !std::isfinite(dns_(0)) || dns_(0)<=0))
    throw std::invalid_argument("Surface water requires positive Vp and density");
  thk_ = z.tail(nl_ - 1) - z.head(nl_ - 1);
  if (is_water_) {
    vs_(0) = 1.0e-8;
    iwater_ = 1;
  } else {
    iwater_ = 0;
  }
}

std::unique_ptr<SecularFunction>
SecularFunction::create(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh,
                        DispMethod method) {
  switch (method) {
  case DispMethod::FAST_SCALAR:
    if(!sh && model(0,3)==0)
      throw std::invalid_argument("Surface-water Rayleigh requires forward_method=dunkin");
    return std::make_unique<FastScalarSecularFunction>(model, sh);
  case DispMethod::DUNKIN:
  default:
    return std::make_unique<DunkinSecularFunction>(model, sh);
  }
}

double SecularFunction::evaluate_sh_impl(double f, double c) const {
  double omega = 2.0 * M_PI * f;
  double wvno = omega / c;
  double beta1 = vs_(nl_ - 1);
  double rho1 = dns_(nl_ - 1);
  double xkb = omega / beta1;
  double wvnop = wvno + xkb;
  double wvnom = abs(wvno - xkb);
  double rb = sqrt(wvnop * wvnom);
  double e1 = rho1 * rb;
  double e2 = 1.0 / (beta1 * beta1);

  for (int m = nl_ - 2; m >= iwater_; --m) {
    beta1 = vs_(m);
    rho1 = dns_(m);
    double xmu = rho1 * beta1 * beta1;
    xkb = omega / beta1;
    wvnop = wvno + xkb;
    wvnom = abs(wvno - xkb);
    rb = sqrt(wvnop * wvnom);
    double q = thk_(m) * rb;

    double sinq, y, z, cosq;
    if (wvno < xkb) {
      sinq = sin(q);
      y = sinq / rb;
      z = -rb * sinq;
      cosq = cos(q);
    } else if (wvno == xkb) {
      cosq = 1.0;
      y = thk_(m);
      z = 0.0;
    } else {
      double fac = (q < 16.0) ? exp(-2.0 * q) : 0.0;
      cosq = (1.0 + fac) * 0.5;
      sinq = (1.0 - fac) * 0.5;
      y = sinq / rb;
      z = rb * sinq;
    }

    double e10 = e1 * cosq + e2 * xmu * z;
    double e20 = e1 * y / xmu + e2 * cosq;
    e1 = e10;
    e2 = e20;
  }
  return e1;
}

int SecularFunction::love_root_count(double f, double c) const {
  if (!sh_ || !std::isfinite(f) || f <= 0 || !std::isfinite(c) || c <= 0
      || c >= vs_(nl_-1)) return -1;
  const double omega=2*M_PI*f, k=omega/c;
  if(!std::isfinite(omega) || !std::isfinite(k)) return -1;
  const double kh=omega/vs_(nl_-1);
  // U and E=-traction propagated upward from the decaying halfspace.
  double u=1, e=dns_(nl_-1)*vs_(nl_-1)*vs_(nl_-1)*std::sqrt((k+kh)*(k-kh));
  if(!std::isfinite(e)) return -1;
  int nodes=0;
  for(int i=nl_-2;i>=iwater_;--i) {
    const double mu=dns_(i)*vs_(i)*vs_(i), kb=omega/vs_(i);
    const double r=std::sqrt((k+kb)*std::abs(k-kb)), q=r*thk_(i);
    if(!std::isfinite(q) || q<0 || !(mu>0)) return -1;
    double C,S,Z;
    if(k<kb) {
      // U(t)=A sin(theta+r*t); each multiple of pi is one displacement node.
      double theta=std::atan2(u,e/(mu*r));
      if(theta<0) theta+=M_PI;
      if(theta>=M_PI) theta-=M_PI;
      const double crossings=std::floor((theta+q)/M_PI);
      if(crossings>2000 || nodes+crossings>2000) return -1;
      nodes+=static_cast<int>(crossings);
      C=std::cos(q); S=std::sin(q)/r; Z=-r*std::sin(q);
    } else if(r==0) {
      C=1; S=thk_(i); Z=0;
    } else {
      const double decay=std::exp(-2*q);
      C=.5*(1+decay); S=-.5*std::expm1(-2*q)/r; Z=r*r*S;
    }
    const double next_u=C*u+S*e/mu, next_e=C*e+mu*Z*u;
    // A nonoscillatory layer can have at most one zero of displacement.
    if(k>=kb && u!=0 && next_u!=0 && std::signbit(u)!=std::signbit(next_u)) ++nodes;
    const double scale=std::max(std::abs(next_u),std::abs(next_e));
    if(!(scale>0) || !std::isfinite(scale)) return -1;
    u=next_u/scale; e=next_e/scale;
  }
  // A free-surface eigenvalue is crossed halfway between displacement nodes.
  return nodes+(u!=0 && e!=0 && std::signbit(u)!=std::signbit(e));
}

double DunkinSecularFunction::evaluate(double f, double c) {
  if (sh_) {
    return evaluate_sh_impl(f, c);
  } else {
    return evaluate_psv(f, c);
  }
}

double DunkinSecularFunction::evaluate_psv(double f, double c) {
  double omega = 2.0 * M_PI * f;
  double wvno = omega / c;
  Array<double, 5, 1> e = Array<double, 5, 1>::Zero();
  Array<double, 5, 1> ee;

  double wvno2 = wvno * wvno;
  double xka = omega / vp_[nl_ - 1];
  double xkb = omega / vs_[nl_ - 1];
  double wvnop = wvno + xka;
  double wvnom = abs(wvno - xka);
  double ra = sqrt(wvnop * wvnom);
  wvnop = wvno + xkb;
  wvnom = abs(wvno - xkb);
  double rb = sqrt(wvnop * wvnom);
  double t = vs_[nl_ - 1] / omega;

  double gammk = 2.0 * t * t;
  double gam = gammk * wvno2;
  double gamm1 = gam - 1.0;
  double rho1 = dns_[nl_ - 1];
  e[0] = rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb);
  e[1] = -rho1 * ra;
  e[2] = rho1 * (gamm1 - gammk * ra * rb);
  e[3] = rho1 * rb;
  e[4] = wvno2 - ra * rb;

  for (int m = nl_ - 2; m >= iwater_; --m) {
    xka = omega / vp_[m];
    xkb = omega / vs_[m];
    t = vs_[m] / omega;
    gammk = 2.0 * t * t;
    gam = gammk * wvno2;
    wvnop = wvno + xka;
    wvnom = abs(wvno - xka);
    ra = sqrt(wvnop * wvnom);
    wvnop = wvno + xkb;
    wvnom = abs(wvno - xkb);
    rb = sqrt(wvnop * wvnom);

    double dpth = thk_[m];
    rho1 = dns_[m];
    double p = ra * dpth;
    double q = rb * dpth;

    double w, cosp, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz;
    var(p, q, ra, rb, wvno, xka, xkb, dpth, w, cosp, a0, cpcq, cpy, cpz, cqw,
        cqx, xy, xz, wy, wz);

    Array<double, 5, 5> ca;
    dnka(wvno2, gam, gammk, rho1, a0, cpcq, cpy, cpz, cqw, cqx,
         xy, xz, wy, wz, ca);

    ee = (ca.matrix().transpose() * e.matrix()).array();
    e = ee;
  }

  double dlt;
  if (iwater_ == 1) {
    xka = omega / vp_[0];
    wvnop = wvno + xka;
    wvnom = abs(wvno - xka);
    ra = sqrt(wvnop * wvnom);
    double dpth = thk_[0];
    rho1 = dns_[0];
    double p = ra * dpth;
    double zul = 1.0e-5;
    double w, cosp, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz;
    var(p, zul, ra, zul, wvno, xka, xkb, dpth, w, cosp, a0, cpcq, cpy, cpz, cqw,
        cqx, xy, xz, wy, wz);
    dlt = cosp * e[0] - rho1 * w * e[1];
  } else {
    dlt = e[0];
  }
  return dlt;
}

void DunkinSecularFunction::var(double p, double q, double ra, double rb,
                                double wvno, double xka, double xkb,
                                double dpth, double &w, double &cosp,
                                double &a0, double &cpcq, double &cpy,
                                double &cpz, double &cqw, double &cqx,
                                double &xy, double &xz, double &wy,
                                double &wz) {
  double pex = 0.0;
  double x = 0.0;
  if (wvno < xka) {
    double sinp = sin(p);
    w = sinp / ra;
    x = -ra * sinp;
    cosp = cos(p);
  } else if (wvno == xka) {
    cosp = 1.0;
    w = dpth;
    x = 0.0;
  } else {
    pex = p;
    double fac = (p < 16.0) ? exp(-2.0 * p) : 0.0;
    cosp = (1.0 + fac) * 0.5;
    double sinp = (1.0 - fac) * 0.5;
    w = sinp / ra;
    x = ra * sinp;
  }

  double sex = 0.0;
  double sinq = 0.0, y = 0.0, z = 0.0, cosq = 0.0;
  if (wvno < xkb) {
    sinq = sin(q);
    y = sinq / rb;
    z = -rb * sinq;
    cosq = cos(q);
  } else if (wvno == xkb) {
    cosq = 1.0;
    y = dpth;
    z = 0.0;
  } else {
    sex = q;
    double fac = (q < 16.0) ? exp(-2.0 * q) : 0.0;
    cosq = (1.0 + fac) * 0.5;
    sinq = (1.0 - fac) * 0.5;
    y = sinq / rb;
    z = rb * sinq;
  }

  double exa = pex + sex;
  a0 = (exa < 60.0) ? exp(-exa) : 0.0;
  cpcq = cosp * cosq;
  cpy = cosp * y;
  cpz = cosp * z;
  cqw = cosq * w;
  cqx = cosq * x;
  xy = x * y;
  xz = x * z;
  wy = w * y;
  wz = w * z;
}

void DunkinSecularFunction::dnka(double wvno2, double gam, double gammk,
                                 double rho, double a0, double cpcq,
                                 double cpy, double cpz, double cqw,
                                 double cqx, double xy, double xz, double wy,
                                 double wz,
                                 Array<double, 5, 5> &ca) {
  ca.setZero();
  double gamm1 = gam - 1.0;
  double twgm1 = gam + gamm1;
  double gmgmk = gam * gammk;
  double gmgm1 = gam * gamm1;
  double gm1sq = gamm1 * gamm1;
  double rho2 = rho * rho;
  double a0pq = a0 - cpcq;

  ca(0, 0) = cpcq - 2.0 * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy;
  ca(0, 1) = (wvno2 * cpy - cqx) / rho;
  ca(0, 2) = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho;
  ca(0, 3) = (cpz - wvno2 * cqw) / rho;
  ca(0, 4) = -(2.0 * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2;

  ca(1, 0) = (gmgmk * cpz - gm1sq * cqw) * rho;
  ca(1, 1) = cpcq;
  ca(1, 2) = gammk * cpz - gamm1 * cqw;
  ca(1, 3) = -wz;
  ca(1, 4) = ca(0, 3);

  ca(3, 0) = (gm1sq * cpy - gmgmk * cqx) * rho;
  ca(3, 1) = -xy;
  ca(3, 2) = gamm1 * cpy - gammk * cqx;
  ca(3, 3) = ca(1, 1);
  ca(3, 4) = ca(0, 1);

  ca(4, 0) = -(2.0 * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz
               + gm1sq * gm1sq * wy) * rho2;
  ca(4, 1) = ca(3, 0);
  ca(4, 2) = -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz
               + gamm1 * gm1sq * wy) * rho;
  ca(4, 3) = ca(1, 0);
  ca(4, 4) = ca(0, 0);

  ca(2, 0) = -2.0 * wvno2 * ca(4, 2);
  ca(2, 1) = -2.0 * wvno2 * ca(3, 2);
  ca(2, 2) = a0 + 2.0 * (cpcq - ca(0, 0));
  ca(2, 3) = -2.0 * wvno2 * ca(1, 2);
  ca(2, 4) = -2.0 * wvno2 * ca(0, 2);
}

} // namespace fwd
