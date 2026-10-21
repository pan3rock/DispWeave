#include "dispweave/objective.hpp"
#include <iostream>
#include <iomanip>
int main() {
    dw::LayeredModel model;
    model.depth=Eigen::Vector3d(0,2,4); model.vs=Eigen::Vector3d(1,2,3);
    model.vp=Eigen::Vector3d(1.8,3.6,5.2); model.rho=Eigen::Vector3d(2,2.2,2.5);
    dw::InvLMForward backend;
    const std::vector<dw::DispersionQuery> queries{{.5,0},{.5,1},{.2,0}};
    auto prediction=backend.evaluate(model,queries);
    if(!prediction.all_valid()) return 1;
    Eigen::Vector3d observed;
    for(int i=0;i<3;++i) observed[i]=prediction.rows[i].phase_velocity;
    observed+=Eigen::Vector3d(.001,-.002,.003);
    Eigen::Matrix3d covariance; covariance<<4,1,.5,1,9,1,.5,1,16; covariance*=1e-6;
    dw::FixedVs parameters(model,.8*model.vs,1.1*model.vs);
    dw::DispersionProblem problem(backend,parameters,{queries,observed,covariance},dw::SensitivityMethod::implicit);
    auto prior=dw::uniform_prior(.8*model.vs,1.1*model.vs,dw::PriorMeasure::physical_vs);
    auto result=dw::log_posterior(problem,parameters.encode(model),prior);
    if(result.status!=dw::EvaluationStatus::valid) {std::cerr<<result.message<<'\n';return 1;}
    std::cout<<std::setprecision(12)<<"log_likelihood="<<result.log_likelihood
             <<"\nlog_prior_dVs="<<result.log_prior<<"\nlog_transform="<<result.log_transform
             <<"\nlog_posterior_dq="<<result.value<<"\ngradient_q="<<result.gradient.transpose()
             <<"\nlikelihood_log_noise_scale_gradient="<<result.log_scale_gradient<<'\n';
}
