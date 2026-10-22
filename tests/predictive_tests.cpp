#include "dispweave/predictive.hpp"
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
    std::cout<<"predictive covariance, discrepancy, replay and explicit failures passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
