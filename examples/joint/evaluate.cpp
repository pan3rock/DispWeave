#include "dispweave/joint.hpp"
#include "dispweave/ellipticity.hpp"
#include <iomanip>
#include <iostream>
int main() {try {
    dw::LayeredModel model;
    model.depth=Eigen::Vector3d(0,2,4);model.vs=Eigen::Vector3d(1,2,3);
    model.vp=Eigen::Vector3d(1.8,3.6,5.2);model.rho=Eigen::Vector3d(2,2.2,2.5);
    dw::InvLMForward backend;
    dw::DispersionObservations dispersion;
    dispersion.queries={{.2,0},{.5,0},{.5,1},{.8,1}};
    const auto f=backend.evaluate(model,dispersion.queries);
    if(!f.all_valid()) throw std::runtime_error("Synthetic dispersion unavailable");
    dispersion.phase_velocity.resize(4);
    for(int j=0;j<4;++j) dispersion.phase_velocity[j]=f.rows[j].phase_velocity+(j%2?.003:-.003);
    dispersion.covariance=Eigen::Matrix4d::Identity()*1e-4;
    dispersion.covariance(0,1)=dispersion.covariance(1,0)=2e-5;
    const std::vector<dw::DispersionQuery> queries={{.3,0},{.8,0},{.5,1}};
    const auto ellipses=dw::rayleigh_ellipticity(model,queries);
    Eigen::VectorXd ellipse(3);
    for(int j=0;j<3;++j) {
        if(ellipses[j].status!=dw::EllipticityStatus::valid) throw std::runtime_error("Synthetic ellipticity unavailable");
        ellipse[j]=ellipses[j].ratio+.01;
    }
    dw::ReceiverOptions options;options.samples=1024;options.pre_arrival=2;
    const auto rf=dw::receiver_function(model,options);
    if(rf.status!=dw::ReceiverStatus::valid) throw std::runtime_error("Synthetic receiver response unavailable");
    const std::vector<int> samples={20,31,42,55,70,90,110};Eigen::VectorXd observed_rf(samples.size());
    for(size_t j=0;j<samples.size();++j) observed_rf[j]=rf.radial_rf[samples[j]]+(j%2?.002:-.002);
    const std::vector<dw::ObservationBlock> blocks={
        dw::dispersion_block("dispersion",backend,dispersion,dw::NoiseScale::lognormal(0,.5)),
        dw::ellipticity_block("ellipticity","rayleigh_modal_ellipticity",queries,
            {ellipse,Eigen::Matrix3d::Identity()*.0009},dw::NoiseScale::fixed()),
        dw::receiver_block("receiver",options,samples,
            {observed_rf,Eigen::MatrixXd::Identity(samples.size(),samples.size())*.0001},dw::NoiseScale::exponential(1))};
    dw::FixedVs parameters(model,.8*model.vs,1.1*model.vs);
    dw::JointProblem problem(parameters,dw::JointLikelihood(blocks,dw::BlockDependence::conditional_independence));
    const auto x=problem.initial_state(parameters.encode(model));
    const auto prior=dw::uniform_prior(.8*model.vs,1.1*model.vs,dw::PriorMeasure::physical_vs);
    const auto p=dw::joint_log_posterior(problem,x,prior);
    if(p.status!=dw::EvaluationStatus::valid) throw std::runtime_error(p.message);
    std::cout<<std::setprecision(12)<<"state=[log(Vs[0:3]),log(scale_dispersion),log(scale_receiver)]\nx="<<x.transpose()
        <<"\nlog_likelihood="<<p.log_likelihood<<"\nlog_model_prior="<<p.log_model_prior
        <<"\nlog_noise_prior="<<p.log_noise_prior<<"\nlog_transform="<<p.log_transform<<"\nlog_posterior="<<p.value<<'\n';
    for(const auto& block:p.data.blocks) std::cout<<"block="<<block.name<<" observable="<<block.observable<<" unit="<<block.unit
        <<" rows="<<block.prediction.size()<<" chi_squared="<<block.chi_squared<<" log_likelihood="<<block.log_likelihood<<'\n';
    const auto derivatives=problem.least_squares(x,true);
    if(derivatives.data.status!=dw::EvaluationStatus::valid) throw std::runtime_error(derivatives.data.message);
    std::cout<<"LM_noise_scales=fixed_at_supplied_state\nLM_jacobian="<<derivatives.data.jacobian.rows()<<"x"<<derivatives.data.jacobian.cols()
        <<" evaluations="<<derivatives.evaluations<<" max_FD_h_h2_difference="<<derivatives.max_difference
        <<"\nHMC_gradient_available="<<problem.likelihood().supports_gradient()<<'\n';
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
