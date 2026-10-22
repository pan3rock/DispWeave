#include "dispweave/predictive.hpp"
#include "dispweave/rj_dispersion.hpp"
#include <iostream>
#include <stdexcept>
void check(bool b,const char* message) {if(!b) throw std::runtime_error(message);}
int main() {try {
    dw::LayeredModel m{Eigen::VectorXd::Zero(1),Eigen::VectorXd::Constant(1,2),Eigen::VectorXd::Constant(1,3.6),Eigen::VectorXd::Constant(1,2.5)};
    dw::FixedVs params(m,Eigen::VectorXd::Constant(1,1),Eigen::VectorXd::Constant(1,3),1.8);
    dw::InvLMForward backend;
    Eigen::Matrix2d cov;cov<<.04,.03,.03,.09;
    dw::DispersionProblem problem(backend,params,{{{.3,0},{.8,0}},Eigen::Vector2d(1.8,1.9),cov});
    Eigen::MatrixXd q=Eigen::MatrixXd::Constant(12000,1,std::log(2.));
    auto r=dw::posterior_predictive(problem,q,2026);
    check(r.status.size()==12000 && r.latent.rows()==12000,"row preservation");
    for(auto status:r.status) check(status==dw::EvaluationStatus::valid,"valid predictive status");
    auto noise=(r.replicated-r.latent).eval();
    check(noise.colwise().mean().norm()<.01,"predictive noise mean");
    check((noise.transpose()*noise/noise.rows()-cov).norm()<.004,"full predictive covariance");
    check(std::abs(r.replicated_discrepancy.mean()-2)<.06,"replicate chi-square calibration");
    check((r.latent.row(0)-r.latent.row(11999)).norm()==0,"latent draws are noise-free");
    auto replay=dw::posterior_predictive(problem,q.topRows(2),2026);
    check((replay.replicated-r.replicated.topRows(2)).norm()==0,"predictive reproducibility");
    q(0,0)=std::log(4.);auto invalid=dw::posterior_predictive(problem,q.topRows(2),2026);
    check(invalid.status[0]==dw::EvaluationStatus::outside_support && !invalid.latent.row(0).allFinite(),"invalid draw retained");
    check((invalid.replicated.row(1)-r.replicated.row(1)).norm()==0,"failure does not shift noise stream");
    dw::DispersionProblem missing(backend,params,{{{.3,1},{.8,1}},Eigen::Vector2d(1.8,1.9),cov});
    auto fail=dw::posterior_predictive(missing,q.bottomRows(1),2026);
    check(fail.status[0]==dw::EvaluationStatus::forward_failure && !fail.latent.allFinite(),"missing mode not dropped");
    dw::RJPrior prior;prior.max_layers=2;prior.layer_weights=Eigen::Vector2d::Ones();prior.vs_lower=1;prior.vs_upper=3;
    dw::RJDispersionProblem rj(backend,prior,problem.likelihood().data(),1.8,2.5);
    std::vector<dw::RJState> states(12000,{Eigen::VectorXd(),Eigen::VectorXd::Constant(1,2)});
    const auto rp=dw::posterior_predictive(rj,states,2026);
    check((rp.replicated-r.replicated).norm()<1e-10,"RJ/fixed predictive covariance or stream mismatch");
    states.resize(3);states[0].vs[0]=4;states[2]={Eigen::VectorXd::Constant(1,1),Eigen::Vector2d(2,2)};
    const auto mixed=dw::posterior_predictive(rj,states,2026);
    check(mixed.status[0]==dw::EvaluationStatus::outside_support && !mixed.latent.row(0).allFinite(),"RJ invalid draw deleted");
    check((mixed.replicated.row(1)-r.replicated.row(1)).norm()<1e-12,"RJ failure shifted RNG stream");
    check(mixed.status[2]==dw::EvaluationStatus::valid && (mixed.latent.row(2)-r.latent.row(2)).norm()<1e-7,"Mixed-dimension predictive mapping");
    dw::RJDispersionProblem rj_missing(backend,prior,missing.likelihood().data(),1.8,2.5);
    const auto rm=dw::posterior_predictive(rj_missing,std::vector<dw::RJState>{states[1]},2026);
    check(rm.status[0]==dw::EvaluationStatus::forward_failure && !rm.latent.allFinite(),"RJ missing mode dropped");
    std::cout<<"predictive covariance, discrepancy, replay and explicit failures passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
