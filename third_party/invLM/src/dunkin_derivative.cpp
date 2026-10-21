#include "dunkin_derivative.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fwd {
namespace {
// Local derivative slots: c, Vs, Vp, rho. These are exact chain-rule
// derivatives of the arithmetic, not finite differences of model/root values.
// Fixed-size dual arithmetic: scalar operations must not propagate an
// artificial zero derivative vector through the general product/quotient rule.
struct Jet {
    double v;
    Eigen::Vector4d d;
    Jet(double value=0):v(value),d(Eigen::Vector4d::Zero()) {}
    Jet(double value,const Eigen::Vector4d& derivative):v(value),d(derivative) {}
    double value() const { return v; }
    Eigen::Vector4d& derivatives() { return d; }
    const Eigen::Vector4d& derivatives() const { return d; }
};
Jet operator+(const Jet& a,const Jet& b){return {a.v+b.v,a.d+b.d};}
Jet operator-(const Jet& a,const Jet& b){return {a.v-b.v,a.d-b.d};}
Jet operator-(const Jet& a){return {-a.v,-a.d};}
Jet operator*(const Jet& a,const Jet& b){return {a.v*b.v,a.d*b.v+b.d*a.v};}
Jet operator/(const Jet& a,const Jet& b){const double inv=1/b.v,r=a.v*inv;return {r,(a.d-r*b.d)*inv};}
Jet operator+(const Jet& a,double b){return {a.v+b,a.d};}
Jet operator+(double a,const Jet& b){return b+a;}
Jet operator-(const Jet& a,double b){return {a.v-b,a.d};}
Jet operator-(double a,const Jet& b){return {a-b.v,-b.d};}
Jet operator*(const Jet& a,double b){return {a.v*b,a.d*b};}
Jet operator*(double a,const Jet& b){return b*a;}
Jet operator/(const Jet& a,double b){return a*(1/b);}
Jet operator/(double a,const Jet& b){const double inv=1/b.v,r=a*inv;return {r,b.d*(-r*inv)};}
Jet sqrt(const Jet& a){const double r=std::sqrt(a.v);return {r,a.d*(.5/r)};}
Jet sin(const Jet& a){return {std::sin(a.v),a.d*std::cos(a.v)};}
Jet cos(const Jet& a){return {std::cos(a.v),a.d*(-std::sin(a.v))};}
Jet exp(const Jet& a){const double r=std::exp(a.v);return {r,a.d*r};}
using V5=Eigen::Matrix<double,5,1>;
using M5=Eigen::Matrix<double,5,5>;
using JM5=Eigen::Matrix<Jet,5,5>;
Jet seed(double value,int index) {
    Jet x(value);x.derivatives()(index)=1;return x;
}
struct Wave { Jet cosine,w,x,exponent; };
Wave wave(const Jet& z,double h) {
    Wave t;
    const Jet u=z*(h*h);
    if(std::abs(u.value())<1e-6) {
        // Analytic limits in vertical wavenumber squared; differentiating
        // sqrt(abs(z)) at z=0 would produce a spurious singular derivative.
        t.cosine=1+u/2+u*u/24+u*u*u/720;
        t.w=h*(1+u/6+u*u/120+u*u*u/5040);
        t.x=z*t.w;t.exponent=0;
    } else if(z.value()<0) {
        const Jet r=sqrt(Jet(-z)),q=r*h;
        t.cosine=cos(q);t.w=sin(q)/r;t.x=z*t.w;t.exponent=0;
    } else {
        const Jet r=sqrt(z),q=r*h;
        const Jet fac=q.value()<16 ? Jet(exp(Jet(-2*q))) : Jet(0);
        t.cosine=(1+fac)*0.5;t.w=(1-fac)*0.5/r;t.x=z*t.w;t.exponent=q;
    }
    return t;
}
JM5 layer(double omega,const Jet& c,const Jet& vs,const Jet& vp,
          const Jet& rho,double h) {
    const Jet k=omega/c,wvno2=k*k;
    const Jet ka=omega/vp,kb=omega/vs;
    const auto p=wave(Jet(wvno2-ka*ka),h);
    const auto q=wave(Jet(wvno2-kb*kb),h);
    const Jet gammk=2*(vs/omega)*(vs/omega),gam=gammk*wvno2;
    const Jet exponent=p.exponent+q.exponent;
    const Jet a0=exponent.value()<60 ? Jet(exp(Jet(-exponent))) : Jet(0);
    const Jet cpcq=p.cosine*q.cosine,cpy=p.cosine*q.w,cpz=p.cosine*q.x;
    const Jet cqw=q.cosine*p.w,cqx=q.cosine*p.x;
    const Jet xy=p.x*q.w,xz=p.x*q.x,wy=p.w*q.w,wz=p.w*q.x;
    // Dunkin compound-matrix entries mirror secfunc.cpp::dnka. Kept isolated
    // from the root evaluator; parity/critical-speed/subdivision tests guard them.
    JM5 ca;
  Jet gamm1 = gam - 1.0;
  Jet twgm1 = gam + gamm1;
  Jet gmgmk = gam * gammk;
  Jet gmgm1 = gam * gamm1;
  Jet gm1sq = gamm1 * gamm1;
  Jet rho2 = rho * rho;
  Jet a0pq = a0 - cpcq;

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
    return ca;
}
struct LayerTape { M5 a;std::array<M5,4> d;V5 below;double scale; };
}

DunkinDerivatives dunkin_derivatives(const VelocityModel& model,double omega,double c) {
    const int n=model.nl;
    const int top=n>1 && model.vs.size()==n && model.vs(0)==0 ? 1:0;
    if(n<1 || !(omega>0) || !(c>0) || !std::isfinite(omega) || !std::isfinite(c)
       || model.vs.size()!=n || model.vp.size()!=n || model.rho.size()!=n || model.depth.size()!=n
       || !model.vs.allFinite() || !model.vp.allFinite() || !model.rho.allFinite()
       || !model.depth.allFinite() || (model.vs.tail(n-top).array()<=0).any() || (model.vp.array()<=model.vs.array()).any()
       || (model.rho.array()<=0).any() || c>=model.vs(n-1))
        throw std::runtime_error("Implicit Dunkin derivative requires solid layers with at most one surface water layer and a trapped mode");
    for(int i=1;i<n;++i) if(!(model.depth(i)>model.depth(i-1)))
        throw std::runtime_error("Implicit Dunkin derivative requires increasing depth");
    const Jet cj=seed(c,0),s=seed(model.vs(n-1),1),p=seed(model.vp(n-1),2),rho=seed(model.rho(n-1),3);
    const Jet k=omega/cj,k2=k*k,ka=omega/p,kb=omega/s;
    const Jet ra=sqrt(Jet(k2-ka*ka)),rb=sqrt(Jet(k2-kb*kb));
    const Jet gk=2*(s/omega)*(s/omega),g=gk*k2,gm=g-1;
    std::array<Jet,5> ej={rho*rho*(gm*gm-g*gk*ra*rb),-rho*ra,
        rho*(gm-gk*ra*rb),rho*rb,k2-ra*rb};
    V5 e;Eigen::Matrix<double,5,4> bottom_d;
    for(int i=0;i<5;++i) {e(i)=ej[i].value();bottom_d.row(i)=ej[i].derivatives().transpose();}
    const double bottom_scale=e.cwiseAbs().maxCoeff();
    if(!(bottom_scale>0) || !std::isfinite(bottom_scale)) throw std::runtime_error("Invalid Dunkin halfspace");
    e/=bottom_scale;bottom_d/=bottom_scale;
    std::vector<LayerTape> tape(n-1);
    for(int m=n-2;m>=top;--m) {
        const auto ca=layer(omega,cj,seed(model.vs(m),1),seed(model.vp(m),2),
                            seed(model.rho(m),3),model.depth(m+1)-model.depth(m));
        auto& t=tape[m];t.below=e;
        for(int i=0;i<5;++i) for(int j=0;j<5;++j) {
            t.a(i,j)=ca(j,i).value();
            for(int d=0;d<4;++d)t.d[d](i,j)=ca(j,i).derivatives()(d);
        }
        e=t.a*e;t.scale=e.cwiseAbs().maxCoeff();
        if(!(t.scale>0) || !std::isfinite(t.scale)) throw std::runtime_error("Invalid Dunkin recursion");
        e/=t.scale;
    }
    // Scaling constants are frozen at this model. At a root this leaves the
    // implicit derivative ratio invariant and avoids differentiating max/abs.
    Eigen::MatrixXd partial=Eigen::MatrixXd::Zero(n,3);
    V5 adj=V5::Unit(0);double fc=0, secular=e(0);
    if(top) {
        // Acoustic free-surface load: no S-wave or 1/Vs terms in the water.
        const Jet p0=seed(model.vp(0),2),r0=seed(model.rho(0),3),ka0=omega/p0;
        const auto w=wave(Jet(k*k-ka0*ka0),model.depth(1)-model.depth(0));
        const Jet f=w.cosine*e(0)-r0*w.w*e(1);
        secular=f.v;fc=f.d(0);partial.row(0)=f.d.tail<3>().transpose();
        adj.setZero();adj(0)=w.cosine.v;adj(1)=-(r0*w.w).v;
    }
    for(int m=top;m<n-1;++m) {
        const auto& t=tape[m];const V5 a=adj/t.scale;
        fc+=a.dot(t.d[0]*t.below);
        for(int d=1;d<4;++d)partial(m,d-1)=a.dot(t.d[d]*t.below);
        adj=t.a.transpose()*a;
    }
    const Eigen::Vector4d last=bottom_d.transpose()*adj;
    fc+=last(0);partial.row(n-1)=last.tail<3>().transpose();
    if(!std::isfinite(fc) || fc==0 || !partial.allFinite())
        throw std::runtime_error("Singular implicit Dunkin derivative");
    DunkinDerivatives result;
    result.root_correction=std::abs(secular/fc);
    if(!std::isfinite(result.root_correction) || result.root_correction>1e-8)
        throw std::runtime_error("Implicit Dunkin derivative was not evaluated at a resolved root");
    result.dc_dvs=-partial.col(0).array()/fc;
    result.dc_dvp=-partial.col(1).array()/fc;
    result.dc_drho=-partial.col(2).array()/fc;
    if(!result.dc_dvs.allFinite() || !result.dc_dvp.allFinite() || !result.dc_drho.allFinite())
        throw std::runtime_error("Non-finite implicit Dunkin phase derivative");
    return result;
}

DunkinDerivatives love_derivatives(const VelocityModel& model,double omega,double c) {
    const int n=model.nl;
    if(n>1 && model.vs.size()==n && model.vs(0)==0) {
        if(model.depth.size()!=n || model.rho.size()!=n || model.vp.size()!=n
           || !std::isfinite(model.depth(0)) || !(model.depth(1)>model.depth(0))
           || !std::isfinite(model.vp(0)) || !(model.vp(0)>0)
           || !std::isfinite(model.rho(0)) || !(model.rho(0)>0)
           || (model.vs.tail(n-1).array()<=0).any())
            throw std::runtime_error("Invalid surface-water Love model");
        VelocityModel solid{n-1,(model.depth.tail(n-1).array()-model.depth(1)).matrix(),
            model.vs.tail(n-1),model.vp.tail(n-1),model.rho.tail(n-1)};
        const auto d=love_derivatives(solid,omega,c);
        DunkinDerivatives out;out.root_correction=d.root_correction;
        out.dc_dvs=Eigen::ArrayXd::Zero(n);out.dc_dvp=Eigen::ArrayXd::Zero(n);out.dc_drho=Eigen::ArrayXd::Zero(n);
        out.dc_dvs.tail(n-1)=d.dc_dvs;out.dc_drho.tail(n-1)=d.dc_drho;
        return out;
    }
    if(n<2 || !(omega>0) || !(c>0) || !std::isfinite(omega) || !std::isfinite(c)
       || model.vs.size()!=n || model.rho.size()!=n || model.depth.size()!=n
       || !model.vs.allFinite() || !model.rho.allFinite() || !model.depth.allFinite()
       || (model.vs.array()<=0).any() || (model.rho.array()<=0).any() || c>=model.vs(n-1))
        throw std::runtime_error("Implicit Love derivative requires a solid, trapped SH mode");
    for(int i=1;i<n;++i)if(!(model.depth(i)>model.depth(i-1)))
        throw std::runtime_error("Implicit Love derivative requires increasing depth");
    using V2=Eigen::Vector2d;
    using M2=Eigen::Matrix2d;
    struct Tape {M2 a;std::array<M2,4> d;V2 below;double scale;};
    const Jet cj=seed(c,0),s=seed(model.vs(n-1),1),rho=seed(model.rho(n-1),3);
    const Jet k=omega/cj,kb=omega/s;
    // Same two-component halfspace vector as SecularFunction::evaluate_sh_impl.
    const std::array<Jet,2> bottom={rho*sqrt(Jet(k*k-kb*kb)),1/(s*s)};
    V2 e;Eigen::Matrix<double,2,4> bottom_d;
    for(int i=0;i<2;++i){e(i)=bottom[i].v;bottom_d.row(i)=bottom[i].d.transpose();}
    const double scale=e.cwiseAbs().maxCoeff();
    if(!(scale>0) || !std::isfinite(scale))throw std::runtime_error("Invalid Love halfspace");
    e/=scale;bottom_d/=scale;
    std::vector<Tape> tape(n-1);
    for(int m=n-2;m>=0;--m) {
        const Jet vs=seed(model.vs(m),1),r=seed(model.rho(m),3),mu=r*vs*vs,b=omega/vs;
        const auto w=wave(Jet(k*k-b*b),model.depth(m+1)-model.depth(m));
        const std::array<Jet,4> a={w.cosine,mu*w.x,w.w/mu,w.cosine};
        auto& t=tape[m];t.below=e;
        for(int i=0;i<2;++i)for(int j=0;j<2;++j) {
            t.a(i,j)=a[2*i+j].v;
            for(int d=0;d<4;++d)t.d[d](i,j)=a[2*i+j].d(d);
        }
        e=t.a*e;t.scale=e.cwiseAbs().maxCoeff();
        if(!(t.scale>0) || !std::isfinite(t.scale))throw std::runtime_error("Invalid Love recursion");
        e/=t.scale;
    }
    // Freeze normalization, as in the Rayleigh adjoint: at F=0 the common
    // scale cancels from -F_m/F_c. All layer derivatives take O(N) work/storage.
    Eigen::MatrixXd partial=Eigen::MatrixXd::Zero(n,3);
    V2 adj=V2::Unit(0);double fc=0;
    for(int m=0;m<n-1;++m) {
        const auto& t=tape[m];const V2 a=adj/t.scale;
        fc+=a.dot(t.d[0]*t.below);
        for(int d=1;d<4;++d)partial(m,d-1)=a.dot(t.d[d]*t.below);
        adj=t.a.transpose()*a;
    }
    const Eigen::Vector4d last=bottom_d.transpose()*adj;
    fc+=last(0);partial.row(n-1)=last.tail<3>().transpose();
    if(!std::isfinite(fc) || fc==0 || !partial.allFinite())
        throw std::runtime_error("Singular implicit Love derivative");
    DunkinDerivatives result;result.root_correction=std::abs(e(0)/fc);
    if(!std::isfinite(result.root_correction) || result.root_correction>1e-8)
        throw std::runtime_error("Implicit Love derivative was not evaluated at a resolved root");
    result.dc_dvs=-partial.col(0).array()/fc;
    result.dc_dvp=Eigen::ArrayXd::Zero(n);
    result.dc_drho=-partial.col(2).array()/fc;
    if(!result.dc_dvs.allFinite() || !result.dc_drho.allFinite())
        throw std::runtime_error("Non-finite implicit Love derivative");
    return result;
}
}
