#pragma once
// Test-only unscaled P/SV plane-wave propagation. Independent of invLM code.
// Valid only for these moderate optical-thickness fixtures, not a production solver.
#include "dispweave/model.hpp"
#include <Eigen/LU>
#include <complex>
#include <stdexcept>
#include <cmath>
using Complex=std::complex<double>;
using C4=Eigen::Matrix<Complex,4,4>;
struct Basis { C4 field; Eigen::Matrix<Complex,4,1> vertical; };
inline Basis basis(double vp,double vs,double rho,double omega,double k) {
    const Complex I(0,1);
    const double mu=rho*vs*vs, lambda=rho*vp*vp-2*mu;
    const Complex p=std::sqrt(Complex(k*k-omega*omega/(vp*vp),0));
    const Complex s=std::sqrt(Complex(k*k-omega*omega/(vs*vs),0));
    Basis b; b.vertical<<-p,-s,p,s;
    for(int j=0;j<4;++j) {
        const Complex z=b.vertical[j];
        // state [ux, uz, txz, tzz], horizontal exp(i*k*x), depth positive down.
        if(j%2==0) b.field.col(j)<<I*k,z,2.0*I*mu*k*z,(lambda+2*mu)*z*z-lambda*k*k;
        else b.field.col(j)<<-z,I*k,-mu*(z*z+k*k),2.0*I*mu*k*z;
    }
    return b;
}
inline double independent_secular(const dw::LayeredModel& m,double frequency,double c) {
    const double omega=2*std::acos(-1.)*frequency,k=omega/c;
    int last=static_cast<int>(m.vs.size())-1;
    // Only downward-decaying P/SV waves in the halfspace.
    Eigen::Matrix<Complex,4,2> state=basis(m.vp[last],m.vs[last],m.rho[last],omega,k).field.leftCols<2>();
    for(int i=last-1;i>=0;--i) {
        const auto b=basis(m.vp[i],m.vs[i],m.rho[i],omega,k);
        auto solver=b.field.fullPivLu();
        if(!solver.isInvertible()) throw std::runtime_error("Reference basis at critical speed");
        Eigen::Matrix<Complex,4,2> amplitudes=solver.solve(state);
        const double h=m.depth[i+1]-m.depth[i];
        for(int j=0;j<4;++j) amplitudes.row(j)*=std::exp(-b.vertical[j]*h);
        state=b.field*amplitudes;
        for(int j=0;j<2;++j) state.col(j)/=state.col(j).norm();
    }
    const Complex determinant=state.bottomRows<2>().determinant();
    if(!std::isfinite(determinant.real()) || std::abs(determinant.imag())>1e-8)
        throw std::runtime_error("Independent reference ill-conditioned");
    return determinant.real();
}
inline double reference_root(const dw::LayeredModel& m,double f,double lo,double hi) {
    double a=independent_secular(m,f,lo),b=independent_secular(m,f,hi);
    if(a*b>=0) throw std::runtime_error("Reference root not bracketed");
    for(int it=0;it<60;++it) {
        const double mid=(lo+hi)/2, y=independent_secular(m,f,mid);
        if(std::signbit(a)!=std::signbit(y)) hi=mid;
        else { lo=mid; a=y; }
    }
    return (lo+hi)/2;
}
