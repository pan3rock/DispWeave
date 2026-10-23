#include "dispweave/mode_support.hpp"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <algorithm>
#include <complex>
#include <map>
#include <stdexcept>
#include <vector>
namespace dw {
namespace {
using C=std::complex<double>;
using Matrix=Eigen::Matrix<C,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
constexpr double pi=3.14159265358979323846;
// Global displacement/traction matching, independently of the production secular
// recurrence. Positive scalings preserve phase. The original bandwidth is 5;
// partial pivoting adds at most 5 upper diagonals. Only U is needed for det phase.
C determinant_phase(Matrix a) {
    const int n=a.rows();C phase=1.;
    for(int j=0;j<n;++j) {
        int pivot=j;const int end=std::min(n,j+6),right=std::min(n,j+11);
        for(int i=j+1;i<end;++i) if(std::abs(a(i,j))>std::abs(a(pivot,j))) pivot=i;
        const double magnitude=std::abs(a(pivot,j));
        if(!std::isfinite(magnitude) || magnitude<1e-13) throw std::runtime_error("Singular/ill-conditioned contour matrix");
        if(pivot!=j) {for(int k=j;k<right;++k) std::swap(a(j,k),a(pivot,k));phase=-phase;}
        phase*=a(j,j)/magnitude;
        for(int i=j+1;i<end;++i) {
            const C factor=a(i,j)/a(j,j);
            for(int k=j+1;k<right;++k) a(i,k)-=factor*a(j,k);
        }
    }
    return phase/std::abs(phase);
}
struct Boundary {
    const LayeredModel& m;
    double omega;
    Eigen::VectorXd mu;
    std::map<std::pair<double,double>,C> cache;
    C phase(C t,bool check_endpoint=false) {
        auto key=std::make_pair(t.real(),t.imag());auto found=cache.find(key);
        if(found!=cache.end()) return found->second;
        const int n=m.vs.size();const C c=m.vs[n-1]*std::sqrt(1.-t*t),k=omega/c;
        Matrix a=Matrix::Zero(4*n-2,4*n-2);
        for(int i=0;i<n;++i) {
            const C p=std::sqrt(k*k-std::pow(omega/m.vp[i],2));
            // Uniformizes the halfspace S branch point exactly at t=0.
            const C s=i==n-1?k*t:std::sqrt(k*k-std::pow(omega/m.vs[i],2));
            const C d=2.*mu[i]*k*k-m.rho[i]*omega*omega;
            Matrix top,bottom;
            if(i==n-1) {
                top.resize(4,2);top<<k,s,-p,-k,-2.*mu[i]*k*p,-d,d,2.*mu[i]*k*s;
            } else {
                top.resize(4,4);bottom.resize(4,4);
                const double h=(m.depth[i+1]-m.depth[i])/2;
                for(int shear=0;shear<2;++shear) {
                    const C q=shear?s:p,v=q*h;
                    const double scale=std::abs(q.real())*h;
                    const C ep=std::exp(v-scale),em=std::exp(-v-scale),ch=(ep+em)/2.;
                    const C sh=std::abs(v)<1e-4?h*(1.+v*v/6.+std::pow(v,4)/120.+std::pow(v,6)/5040.)*std::exp(-scale):(ep-em)/(2.*q);
                    for(int side:{-1,1}) for(int j=0;j<2;++j) {
                        const C f=j?double(side)*sh:ch,fp=j?ch:double(side)*q*q*sh;
                        auto& b=side<0?top:bottom;const int col=2*shear+j;
                        if(shear) b.col(col)<<-fp,-k*f,-d*f,-2.*mu[i]*k*fp;
                        else b.col(col)<<k*f,fp,2.*mu[i]*k*fp,d*f;
                    }
                }
            }
            if(i==0) a.block(0,0,2,top.cols())=top.bottomRows(2);
            else a.block(4*i-2,4*i,4,top.cols())=-top;
            if(i<n-1) a.block(4*i+2,4*i,4,4)=bottom;
        }
        for(int j=0;j<a.cols();++j) {
            const double scale=a.col(j).cwiseAbs().maxCoeff();
            if(!std::isfinite(scale) || scale<=0) throw std::runtime_error("Nonfinite boundary matrix");
            a.col(j)/=scale;
        }
        for(int i=0;i<a.rows();++i) {
            const double scale=a.row(i).cwiseAbs().maxCoeff();
            if(!std::isfinite(scale) || scale<=0) throw std::runtime_error("Nonfinite boundary matrix");
            a.row(i)/=scale;
        }
        if(check_endpoint) {
            // Pivot sizes alone cannot detect a globally ill-conditioned endpoint.
            const Eigen::MatrixXd real=a.real();
            const Eigen::VectorXd singular=real.jacobiSvd().singularValues();
            if(a.imag().cwiseAbs().maxCoeff()>1e-12 || singular.tail(1)[0]/singular[0]<1e-12)
                throw std::runtime_error("Unresolved threshold/lower endpoint");
        }
        C result=determinant_phase(std::move(a))*std::exp(C(0,-(2.*k*m.depth[n-1]).imag()));
        cache.emplace(key,result);return result;
    }
};
struct Contour {int count;double maximum,error;};
Contour contour(Boundary& b,double high,double height,int n) {
    std::vector<double> x;
    for(int i=0;i<=n;++i) x.push_back(high*i/n);
    for(int i=0;i<=n/2;++i) x.push_back(std::exp(std::log(1e-12)+(std::log(high)-std::log(1e-12))*i/(n/2)));
    std::sort(x.begin(),x.end());x.erase(std::unique(x.begin(),x.end()),x.end());
    // Taper away from propagating/leaky continuations on the imaginary t axis.
    // Includes the exact threshold; no real interval is omitted.
    constexpr double tip=1e-10;
    const double right=std::min(height,.25*high+tip);
    std::vector<C> points;
    for(double v:x) points.emplace_back(v,-std::min(height,.25*v+tip));
    for(int i=0;i<=n/2;++i) points.emplace_back(high,-right+2*right*i/(n/2));
    for(auto it=x.rbegin();it!=x.rend();++it) points.emplace_back(*it,std::min(height,.25**it+tip));
    for(int i=0;i<=16;++i) points.emplace_back(0,tip-2*tip*i/16);
    double winding=0,maximum=0;const C first=b.phase(points.front());C previous=first;
    for(std::size_t i=1;i<=points.size();++i) {
        const C next=i==points.size()?first:b.phase(points[i]);
        const double delta=std::arg(next/previous);previous=next;
        winding+=delta;maximum=std::max(maximum,std::abs(delta));
    }
    winding/=2*pi;const int count=std::lround(winding);
    return {count,maximum,std::abs(winding-count)};
}
}
ModeCount trapped_mode_count(const LayeredModel& m,double frequency,int budget) {
    m.validate();ModeCount out;
    if(!std::isfinite(frequency) || frequency<=0 || budget<64 || budget>16384) throw std::invalid_argument("Invalid mode-count frequency/budget");
    const Eigen::VectorXd mu=m.rho.array()*m.vs.array().square();
    const Eigen::VectorXd lambda_mu=m.rho.array()*(m.vp.array().square()-m.vs.array().square());
    // Plane-strain energy >= homogeneous (lambda=0, mu=mu0) energy,
    // with mass <= rho_max. Its Rayleigh c^2=(3-sqrt(5))*mu0/rho_max.
    out.lower_speed=.95*std::sqrt((3-std::sqrt(5.))*mu.cwiseMin(lambda_mu).minCoeff()/m.rho.maxCoeff());
    const double ratio=out.lower_speed/m.vs.tail(1)[0],high=std::sqrt(1-ratio*ratio);
    Boundary b{m,2*pi*frequency,mu,{}};
    try {
        // Check the real endpoint too; a threshold zero is unresolved, not absent.
        b.phase(0.,true);b.phase(high,true);
        int agreed=-1;
        for(double height:{.02,.01}) {
            Contour previous{-1,INFINITY,INFINITY};bool resolved=false;
            for(int n=64;n<=budget;n*=2) {
                const auto current=contour(b,high,height,n);
                if(current.count>=0 && current.count==previous.count &&
                   std::max(current.maximum,previous.maximum)<pi/4 &&
                   std::max(current.error,previous.error)<1e-8) {
                    if(agreed>=0 && agreed!=current.count) throw std::runtime_error("Contour heights disagree");
                    agreed=current.count;out.maximum_phase_increment=std::max(out.maximum_phase_increment,current.maximum);
                    resolved=true;break;
                }
                previous=current;
            }
            if(!resolved) throw std::runtime_error("Mode-count refinement budget exhausted");
        }
        out.count=agreed;out.resolved=true;out.message="Two tapered uniformized contours converged over full trapped interval";
    } catch(const std::runtime_error& e) {out.message=e.what();}
    out.evaluations=b.cache.size();return out;
}
}
