#include "dispweave/metropolis.hpp"
#include "models.hpp"
#include <Eigen/Cholesky>
#include <iostream>
#include <stdexcept>
void check(bool ok,const char* msg) {if(!ok) throw std::runtime_error(msg);}
void rejects(const std::function<void()>& f) {bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}check(caught,"Invalid sampler input accepted");}
int main() {try {
    Eigen::Matrix2d c;c<<1,.97,.97,1;const Eigen::Matrix2d inverse=c.llt().solve(Eigen::Matrix2d::Identity());
    const dw::ValueDensity gaussian=[inverse](const Eigen::VectorXd& x) {return dw::ValueEvaluation{dw::EvaluationStatus::valid,-.5*x.dot(inverse*x)};};
    for(unsigned seed:{43,731,2026}) {
        dw::MetropolisOptions o;o.seed=seed;o.warmup=2000;o.draws=30000;
        const auto r=dw::sample_metropolis(gaussian,v({1,-1}),Eigen::Matrix2d::Identity()*.1,o);
        const Eigen::VectorXd mean=r.draws.colwise().mean();
        const Eigen::MatrixXd centered=r.draws.rowwise()-mean.transpose();
        const Eigen::Matrix2d covariance=centered.transpose()*centered/(r.draws.rows()-1.);
        check(mean.cwiseAbs().maxCoeff()<.07,"Correlated Gaussian mean");
        check((covariance-c).cwiseAbs().maxCoeff()<.12,"Correlated Gaussian covariance");
        check(!r.covariance_updates.empty(),"Covariance was not adapted");
        for(const auto& t:r.transitions) if(!t.warmup) check(t.scale==r.scale,"Adaptation leaked into retained sampling");
        for(const auto& update:r.covariance_updates) check(update.warmup_iteration<=o.warmup,"Production covariance adaptation");
        std::cout<<"seed="<<seed<<" mean="<<mean.transpose()<<" max_cov_error="<<(covariance-c).cwiseAbs().maxCoeff()<<'\n';
    }
    const dw::ValueDensity uniform=[](const Eigen::VectorXd& x) {
        return x[0]>=0 && x[0]<=1?dw::ValueEvaluation{dw::EvaluationStatus::valid,0}:dw::ValueEvaluation{dw::EvaluationStatus::outside_support,-INFINITY};
    };
    dw::MetropolisOptions o;o.warmup=0;o.draws=30000;o.adapt_scale=false;o.adapt_covariance=false;
    const auto bounded=dw::sample_metropolis(uniform,v({.5}),Eigen::MatrixXd::Constant(1,1,.16),o);
    check(std::abs(bounded.draws.mean()-.5)<.02 && std::abs(bounded.draws.array().square().mean()-1./3)<.02,"Support-boundary bias");
    check((bounded.draws.array()>=0).all() && (bounded.draws.array()<=1).all(),"Out-of-support draw retained");
    int rejected=0;
    for(int i=1;i<o.draws;++i) if(!bounded.transitions[i].accepted) {++rejected;check(bounded.draws(i,0)==bounded.draws(i-1,0),"Rejected state was dropped");}
    check(rejected>1000,"Boundary rejection test ineffective");
    const dw::ValueDensity exponential=[](const Eigen::VectorXd& x) {
        return x[0]>=0?dw::ValueEvaluation{dw::EvaluationStatus::valid,-x[0]}:dw::ValueEvaluation{dw::EvaluationStatus::outside_support,-INFINITY};
    };
    const auto skew=dw::sample_metropolis(exponential,v({1}),Eigen::MatrixXd::Constant(1,1,1),o);
    check(std::abs(skew.draws.mean()-1)<.07 && std::abs(skew.draws.array().square().mean()-2)<.2,"Exponential distribution mismatch");
    o.draws=100;
    const auto a=dw::sample_metropolis(uniform,v({.5}),Eigen::MatrixXd::Constant(1,1,.16),o);
    const auto b=dw::sample_metropolis(uniform,v({.5}),Eigen::MatrixXd::Constant(1,1,.16),o);
    check(a.draws==b.draws,"Fixed seed replay differs");o.chain_id=1;
    const auto other=dw::sample_metropolis(uniform,v({.5}),Eigen::MatrixXd::Constant(1,1,.16),o);
    check(!a.draws.isApprox(other.draws),"Chains share a random stream");
    rejects([&]{dw::sample_metropolis(uniform,v({-1}),Eigen::MatrixXd::Constant(1,1,.16),o);});
    rejects([&]{dw::sample_metropolis(gaussian,v({0,0}),Eigen::Matrix2d::Ones(),o);});
    rejects([&]{dw::sample_metropolis(gaussian,v({0,0}),Eigen::MatrixXd::Ones(2,3),o);});
    const dw::ValueDensity broken=[](const Eigen::VectorXd&){return dw::ValueEvaluation{dw::EvaluationStatus::valid,NAN};};
    rejects([&]{dw::sample_metropolis(broken,v({0}),Eigen::MatrixXd::Identity(1,1),o);});
    int calls=0;const dw::ValueDensity failures=[&](const Eigen::VectorXd&) {
        ++calls;return calls==1?dw::ValueEvaluation{dw::EvaluationStatus::valid,0}:dw::ValueEvaluation{dw::EvaluationStatus::forward_failure,NAN};
    };
    const auto failed=dw::sample_metropolis(failures,v({0}),Eigen::MatrixXd::Identity(1,1),o);
    check(failed.draws.isZero() && calls==101,"Failed proposals corrupted chain");
    for(const auto& t:failed.transitions) check(!t.accepted && t.status==dw::EvaluationStatus::forward_failure,"Failure diagnostic lost");
    std::cout<<"PASS Metropolis distributions, adaptation, support, replay and failures\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
