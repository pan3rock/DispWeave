#pragma once
#include "dispweave/hmc.hpp"
namespace stan::mcmc {
struct ps_point {
    Eigen::VectorXd q,p;
    dw::DensityEvaluation density;
    double V=0;
    explicit ps_point(int n):q(Eigen::VectorXd::Zero(n)),p(q) {}
    template<class T> void set_metric(const T&) {} // Metric supplied through test Model.
};
class sample {
    Eigen::VectorXd q_;
    double log_prob_,accept_;
public:
    sample(Eigen::VectorXd q,double lp,double accept):q_(std::move(q)),log_prob_(lp),accept_(accept) {}
    const Eigen::VectorXd& cont_params() const {return q_;}
    double log_prob() const {return log_prob_;}
    double accept_stat() const {return accept_;}
};
}
