#include "dispweave/receiver.hpp"
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace dw {
namespace {
using C=std::complex<double>;
using Matrix4=Eigen::Matrix<C,4,4>;
struct Waves {Matrix4 field;C qp,qs;};
Waves waves(double vp,double vs,double rho,double p) {
    const double ip=1/vp,is=1/vs,eps=32*std::numeric_limits<double>::epsilon();
    if(std::abs(ip-p)<=eps*std::max(ip,p) || std::abs(is-p)<=eps*std::max(is,p))
        throw std::runtime_error("Critical vertical slowness within 32 epsilon: degenerate P/SV basis");
    // Factoring the difference of squares avoids cancellation at criticality.
    Waves w;w.qp=std::sqrt(C((ip-p)*(ip+p),0));w.qs=std::sqrt(C((is-p)*(is+p),0));
    const double mu=rho*vs*vs,lambda=rho*vp*vp-2*mu;
    for(int j=0;j<4;++j) {
        const double sign=j<2?1.:-1.;
        if(j%2==0) {
            const C q=sign*w.qp;
            w.field.col(j)<<vp*p,vp*q,vp*2*mu*p*q,vp*(lambda/(vp*vp)+2*mu*q*q);
        } else {
            const C q=sign*w.qs;
            w.field.col(j)<<-vs*q,vs*p,vs*mu*(p*p-q*q),vs*2*mu*p*q;
        }
    }
    return w;
}
}
const char* to_string(ReceiverStatus s) {return s==ReceiverStatus::valid?"valid":"numerical_failure";}
std::vector<ReceiverSpectrumRow> receiver_spectrum(const LayeredModel& model,
    const std::vector<double>& frequencies,double p) {
    model.validate();
    const int n=static_cast<int>(model.vs.size());
    if(n>128) throw std::invalid_argument("Receiver boundary solver supports at most 128 total layers");
    if(!std::isfinite(p) || p<0 || p>=1/model.vp[n-1])
        throw std::invalid_argument("Receiver requires 0 <= slowness < 1/Vp_halfspace (propagating incident P)");
    if(frequencies.empty()) throw std::invalid_argument("Empty receiver frequency list");
    for(double f:frequencies) if(!std::isfinite(f) || f<0) throw std::invalid_argument("Receiver frequencies must be finite and nonnegative");
    std::vector<ReceiverSpectrumRow> result(frequencies.size());
    std::vector<Waves> basis;
    try {for(int i=0;i<n;++i) basis.push_back(waves(model.vp[i],model.vs[i],model.rho[i],p));}
    catch(const std::runtime_error& e) {for(auto& row:result) row.message=e.what();return result;}
    const int unknowns=4*(n-1)+2;
    for(size_t index=0;index<frequencies.size();++index) {
        auto& row=result[index];
        const double omega=2*std::acos(-1.)*frequencies[index];
        std::vector<Matrix4> top(n),bottom(n);
        for(int i=0;i<n;++i) {
            top[i]=bottom[i]=basis[i].field;
            if(i==n-1) continue;
            const double h=model.depth[i+1]-model.depth[i];
            const C ep=std::exp(C(0,omega)*basis[i].qp*h),es=std::exp(C(0,omega)*basis[i].qs*h);
            // Anchor each travelling amplitude at its incident boundary. Only
            // decaying evanescent factors enter; no growing transfer matrices.
            bottom[i].col(0)*=ep;bottom[i].col(1)*=es;
            top[i].col(2)*=ep;top[i].col(3)*=es;
        }
        Eigen::MatrixXcd a=Eigen::MatrixXcd::Zero(unknowns,unknowns);
        Eigen::VectorXcd b=Eigen::VectorXcd::Zero(unknowns);
        a.block(0,0,2,n==1?2:4)=top[0].block(2,0,2,n==1?2:4);
        if(n==1) b.head(2)=-top[0].block(2,2,2,1);
        for(int i=0;i<n-1;++i) {
            const int r=2+4*i;
            a.block(r,4*i,4,4)=bottom[i];
            const int cols=i==n-2?2:4;
            a.block(r,4*(i+1),4,cols)=-top[i+1].leftCols(cols);
            if(i==n-2) b.segment(r,4)=top[n-1].col(2);
        }
        if(!a.allFinite() || !b.allFinite()) {row.message="Nonfinite receiver boundary system";continue;}
        // Row equilibration accounts for different displacement/traction units.
        for(int r=0;r<unknowns;++r) {
            const double scale=a.row(r).cwiseAbs().maxCoeff();
            if(scale>0) {a.row(r)/=scale;b[r]/=scale;}
        }
        const Eigen::FullPivLU<Eigen::MatrixXcd> lu(a);
        row.reciprocal_condition=lu.rcond();
        if(!lu.isInvertible() || !std::isfinite(row.reciprocal_condition) || row.reciprocal_condition<1e-12) {
            row.message="Receiver boundary system singular or rcond < 1e-12";continue;
        }
        const Eigen::VectorXcd solution=lu.solve(b);
        row.backward_error=(a*solution-b).norm()/(a.norm()*solution.norm()+b.norm());
        if(!solution.allFinite() || !std::isfinite(row.backward_error) || row.backward_error>1e-10) {
            row.message="Receiver boundary solve failed backward-error check";continue;
        }
        Eigen::Matrix<C,4,1> surface=top[0].leftCols(n==1?2:4)*solution.head(n==1?2:4);
        if(n==1) surface+=top[0].col(2);
        if(!surface.allFinite()) {row.message="Nonfinite receiver surface displacement";continue;}
        row.radial=surface[0];row.vertical=-surface[1];row.status=ReceiverStatus::valid;
    }
    return result;
}
void ReceiverOptions::validate() const {
    if(samples<8 || samples>4096 || samples%2!=0 || !std::isfinite(dt) || dt<=0)
        throw std::invalid_argument("Receiver samples must be even in [8,4096], with positive finite dt");
    if(!std::isfinite(gaussian_width) || gaussian_width<=0 || !std::isfinite(water_level) || water_level<=0 || water_level>1)
        throw std::invalid_argument("Receiver Gaussian width must be positive and water level in (0,1]");
    const double window=samples*dt;
    if(!std::isfinite(window) || !std::isfinite(pre_arrival) || pre_arrival<0 || pre_arrival>=window
       || !std::isfinite(time_shift) || pre_arrival+time_shift<0 || pre_arrival+time_shift>=window)
        throw std::invalid_argument("Shifted receiver arrival must lie inside the finite sample window");
    if(source.size()!=0 && (source.size()!=samples || !source.allFinite() || source.cwiseAbs().maxCoeff()==0))
        throw std::invalid_argument("Source must be empty (impulse) or a finite nonzero vector of sample length");
}
ReceiverResult receiver_function(const LayeredModel& model,const ReceiverOptions& options) {
    options.validate();
    const int n=options.samples,half=n/2;
    const double pi=std::acos(-1.);
    ReceiverResult result;
    result.time.resize(n);result.radial_rf=Eigen::VectorXd::Constant(n,NAN);
    for(int j=0;j<n;++j) result.time[j]=j*options.dt-options.pre_arrival;
    for(int k=0;k<=half;++k) result.frequency_hz.push_back(k/(n*options.dt));
    result.spectrum=receiver_spectrum(model,result.frequency_hz,options.slowness_s_km);
    result.source_spectrum=Eigen::VectorXcd::Ones(half+1);
    result.radial_transfer=Eigen::VectorXcd::Constant(half+1,C(NAN,NAN));
    result.vertical_transfer=result.radial_transfer;
    // Direct DFT: explicit sign/normalization and no added FFT dependency.
    // Source is transformed with +i omega t; synthesis below uses -i omega t.
    if(options.source.size()) for(int k=0;k<=half;++k) {
        C sum=0,phase=1;const C step=std::polar(1.,2*pi*k/n);
        for(int j=0;j<n;++j) {sum+=options.source[j]*phase;phase*=step;}
        result.source_spectrum[k]=sum;
    }
    for(const auto& row:result.spectrum) if(row.status!=ReceiverStatus::valid) {
        result.message="Receiver response has failed frequency bins; inspect spectrum diagnostics";return result;
    }
    Eigen::VectorXcd radial(half+1),vertical(half+1);
    double max_power=0;
    for(int k=0;k<=half;++k) {
        radial[k]=result.spectrum[k].radial*result.source_spectrum[k];
        vertical[k]=result.spectrum[k].vertical*result.source_spectrum[k];
        max_power=std::max(max_power,std::norm(vertical[k]));
    }
    if(!(max_power>0) || !std::isfinite(max_power) || !radial.allFinite() || !vertical.allFinite()) {
        result.message="Source/vertical response has zero or nonfinite spectral power";return result;
    }
    const double floor=options.water_level*max_power;
    double normalization=0;
    for(int k=0;k<=half;++k) {
        const double omega=2*pi*result.frequency_hz[k];
        const double gaussian=std::exp(-std::pow(omega/(2*options.gaussian_width),2));
        const double power=std::norm(vertical[k]),denominator=std::max(power,floor);
        result.radial_transfer[k]=radial[k]*std::conj(vertical[k])/denominator*gaussian;
        result.vertical_transfer[k]=power/denominator*gaussian;
        normalization+=(k==0 || k==half?1:2)*result.vertical_transfer[k].real()/n;
    }
    if(!(normalization>0) || !std::isfinite(normalization) || !result.radial_transfer.allFinite()) {
        result.message="Invalid receiver deconvolution normalization";return result;
    }
    result.normalization=normalization;
    result.radial_transfer/=normalization;result.vertical_transfer/=normalization;
    const double shift=options.pre_arrival+options.time_shift;
    for(int j=0;j<n;++j) {
        C phase=1;const C step=std::polar(1.,-2*pi*(j*options.dt-shift)/(n*options.dt));
        double sum=0;
        for(int k=0;k<=half;++k) {
            sum+=(k==0 || k==half?1:2)*(result.radial_transfer[k]*phase).real();
            phase*=step;
        }
        result.radial_rf[j]=sum/n;
    }
    if(!result.radial_rf.allFinite()) {result.message="Nonfinite receiver time series";result.radial_rf.setConstant(NAN);return result;}
    result.status=ReceiverStatus::valid;return result;
}

}
