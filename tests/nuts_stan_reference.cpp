#include "../src/nuts_multinomial.hpp"
#include "stan/callbacks/logger.hpp"
#include "../third_party/stan-nuts-reference/upstream/base_nuts.hpp"
#include <Eigen/Cholesky>
#include <iostream>
struct Model {dw::LogDensity target;dw::EuclideanMetric metric;int dimension=2;};
template<class M,class RNG> struct Hamiltonian {
    const M& model;
    explicit Hamiltonian(const M& m):model(m) {}
    void sample_p(stan::mcmc::ps_point& z,RNG& rng) {z.p=model.metric.momentum(rng);}
    void init(stan::mcmc::ps_point& z,stan::callbacks::logger&) {z.density=model.target(z.q);z.V=-z.density.value;}
    double H(const stan::mcmc::ps_point& z) {return z.V+model.metric.kinetic(z.p);}
    Eigen::VectorXd dtau_dp(const stan::mcmc::ps_point& z) {return model.metric.velocity(z.p);}
};
template<class H> struct Integrator {
    void evolve(stan::mcmc::ps_point& z,H& h,double step,stan::callbacks::logger&) {
        auto e=dw::leapfrog(h.model.target,{z.q,z.p,z.density},h.model.metric,step,1);
        z.q=std::move(e.state.position);z.p=std::move(e.state.momentum);z.density=std::move(e.state.density);
        z.V=e.status==dw::EvaluationStatus::valid?-z.density.value:INFINITY;
    }
};
int main() {try {
    using namespace dw;
    for(int kind=0;kind<7;++kind) for(int seed:{17,2026,99173}) {
        Eigen::Matrix2d cov;cov<<1,.99,.99,1;
        const Eigen::MatrixXd precision=cov.llt().solve(Eigen::Matrix2d::Identity());
        const Eigen::MatrixXd mass=kind==1?precision:Eigen::MatrixXd::Identity(2,2);
        LogDensity target=[=](const Eigen::VectorXd& q)->DensityEvaluation {
            if(kind==3 && q[0]<=0) return {EvaluationStatus::outside_support,-INFINITY,{}};
            if(kind==4 && q.squaredNorm()>=1) return {EvaluationStatus::outside_support,-INFINITY,{}};
            if(kind==5 && q[0]>.1) return {};
            if(kind<2) return {EvaluationStatus::valid,-.5*q.dot(precision*q),-precision*q};
            if(kind==2) {
                const double y=q[1]-.5*(q[0]*q[0]-1);Eigen::Vector2d g(-q[0]+q[0]*y,-y);
                return {EvaluationStatus::valid,-.5*(q[0]*q[0]+y*y),g};
            }
            if(kind==4) return {EvaluationStatus::valid,0,Eigen::Vector2d::Zero()};
            return {EvaluationStatus::valid,-.5*q.squaredNorm(),-q};
        };
        Model model{target,EuclideanMetric(mass),2};
        std::mt19937_64 ours_rng(seed),reference_rng(seed);
        stan::mcmc::base_nuts<Model,Hamiltonian,Integrator,std::mt19937_64> reference(model,reference_rng);
        reference.set_max_depth(8);reference.set_max_delta(1000);
        const double step=kind==0?.12:(kind==6?4.:.4);reference.set_nominal_stepsize(step);
        Eigen::VectorXd q=Eigen::Vector2d::Zero();if(kind==3) q[0]=.5;
        stan::mcmc::sample previous(q,target(q).value,0);stan::callbacks::logger logger;
        for(int i=0;i<1000;++i) {
            HamiltonianState initial{q,model.metric.momentum(ours_rng),target(q)};
            auto result=detail::multinomial_transition(target,initial,model.metric,ours_rng,step,8,1000,false);
            auto expected=reference.transition(previous,logger);std::vector<double> diagnostic;reference.get_sampler_params(diagnostic);
            if(!result.state.position.isApprox(expected.cont_params(),1e-12) ||
               result.diagnostic.evaluations!=static_cast<int>(diagnostic[2]) ||
               std::abs(result.diagnostic.acceptance_probability-expected.accept_stat())>1e-12 ||
               (kind!=3 && kind!=4 && result.diagnostic.divergent!=bool(diagnostic[3])))
                throw std::runtime_error("Stan transition disagreement: kind="+std::to_string(kind)+", seed="+std::to_string(seed)+", iteration="+std::to_string(i));
            q=result.state.position;previous=expected;
        }
        std::cout<<"Stan fixed-kernel replay: kind="<<kind<<" seed="<<seed<<" 1000 transitions agree\n";
    }
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
