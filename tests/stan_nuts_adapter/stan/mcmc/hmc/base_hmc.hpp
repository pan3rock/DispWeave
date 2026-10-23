#pragma once
#include "stan/mcmc/hmc/hamiltonians/ps_point.hpp"
// Minimal interface adapter for executing the UNMODIFIED upstream NUTS header.
// Shares the independently tested target, metric and leapfrog with DispWeave;
// this verifies trajectory construction/selection, not Stan's AD or adaptation.
namespace stan::mcmc {
template<class Model,template<class,class> class Hamiltonian,template<class> class Integrator,class BaseRNG>
class base_hmc {
    struct Uniform {
        BaseRNG& rng;
        double operator()() {return std::generate_canonical<double,53>(rng);}
    };
protected:
    ps_point z_;
    Hamiltonian<Model,BaseRNG> hamiltonian_;
    Integrator<Hamiltonian<Model,BaseRNG>> integrator_;
    BaseRNG& rand_int_;
    Uniform rand_uniform_;
    double epsilon_=.1;
    void sample_stepsize() {}
    void seed(const Eigen::VectorXd& q) {z_.q=q;}
public:
    base_hmc(const Model& model,BaseRNG& rng):z_(model.dimension),hamiltonian_(model),rand_int_(rng),rand_uniform_{rng} {}
    template<class Metric> base_hmc(const Model& model,BaseRNG& rng,const Metric&):base_hmc(model,rng) {}
    void set_nominal_stepsize(double step) {epsilon_=step;}
};
}
