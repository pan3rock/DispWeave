#pragma once
#include "dispweave/hmc.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <stdexcept>
namespace dw::detail {
class DualAveraging {
public:
    DualAveraging(double step,double target):target_(target),average_(std::log(step)),mu_(std::log(10.)+std::log(step)) {}
    double update(double acceptance,bool finish) {
        const double t=++count_,eta=1/(t+10.);
        hbar_=(1-eta)*hbar_+eta*(target_-acceptance);
        const double log_step=mu_-std::sqrt(t)*hbar_/.05,weight=std::pow(t,-.75);
        average_=weight*log_step+(1-weight)*average_;
        const double step=std::exp(finish?average_:log_step);
        if(!std::isfinite(step) || step<=0) throw std::runtime_error("Step adaptation overflow/underflow");
        return step;
    }
private:
    double target_,average_,mu_,hbar_=0;
    int count_=0;
};
struct WarmupWindow {int begin,end;}; // zero-based half-open iteration range
inline std::vector<WarmupWindow> warmup_windows(int warmup) {
    if(warmup<20) throw std::invalid_argument("Metric adaptation needs at least 20 warmup iterations");
    const int initial=warmup>=150?75:static_cast<int>(.15*warmup);
    const int terminal=warmup>=150?50:static_cast<int>(.1*warmup);
    const int slow_end=warmup-terminal;
    int width=warmup>=150?25:slow_end-initial;
    std::vector<WarmupWindow> windows;
    for(int start=initial;start<slow_end;) {
        const int remaining=slow_end-start;
        const int end=width>remaining/3?slow_end:start+width;
        windows.push_back({start,end});start=end;
        if(start<slow_end) width*=2;
    }
    return windows;
}
class CovarianceAccumulator {
public:
    explicit CovarianceAccumulator(Eigen::Index n):mean_(Eigen::VectorXd::Zero(n)),sum_(Eigen::MatrixXd::Zero(n,n)) {}
    void add(const Eigen::VectorXd& x) {
        ++count_;const Eigen::VectorXd delta=x-mean_;mean_+=delta/count_;
        sum_.noalias()+=delta*(x-mean_).transpose();
    }
    int count() const {return count_;}
    Eigen::MatrixXd mass(MetricAdaptation type) const {
        if(count_<2) throw std::runtime_error("Insufficient covariance samples");
        Eigen::MatrixXd covariance=(sum_+sum_.transpose()).eval()/(2.*(count_-1));
        if(type==MetricAdaptation::diagonal) covariance=covariance.diagonal().asDiagonal().toDenseMatrix();
        const double weight=count_/(count_+5.);
        covariance*=weight;covariance.diagonal().array()+=(1-weight)*1e-3;
        if(!covariance.allFinite()) throw std::runtime_error("Nonfinite adapted covariance");
        Eigen::LLT<Eigen::MatrixXd> factor(covariance);
        if(factor.info()!=Eigen::Success) throw std::runtime_error("Adapted covariance not positive definite");
        Eigen::MatrixXd mass=factor.solve(Eigen::MatrixXd::Identity(mean_.size(),mean_.size()));
        return .5*(mass+mass.transpose()).eval();
    }
private:
    int count_=0;
    Eigen::VectorXd mean_;
    Eigen::MatrixXd sum_;
};
}
