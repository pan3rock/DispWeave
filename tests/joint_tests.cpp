#include "dispweave/joint.hpp"
#include "dispweave/ellipticity.hpp"
#include "dispweave/lm.hpp"
#include "models.hpp"
#include <iostream>
#include <map>
#include <stdexcept>
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
void near(double a,double b,double tolerance) {
    if(!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b)>tolerance)
        throw std::runtime_error("Mismatch "+std::to_string(a)+" vs "+std::to_string(b));
}
void rejects(const std::function<void()>& f) {bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}check(rejected,"Invalid joint input accepted");}
constexpr auto independent=dw::BlockDependence::conditional_independence;
dw::LayeredModel toy_model() {return {v({0,2}),v({1,2}),v({2,4}),v({2,2.5})};}
dw::ObservationBlock linear_block(std::string name,Eigen::MatrixXd a,Eigen::MatrixXd b,dw::GaussianData data,
                                   dw::NoiseScale noise=dw::NoiseScale::fixed()) {
    return {name,"test_linear","test_unit",dw::GaussianDataLikelihood(std::move(data)),
        [a,b](const dw::LayeredModel& m,bool gradient) {
            dw::ModelPrediction p;p.status=dw::EvaluationStatus::valid;p.values=a*m.vs+b*m.vp;
            if(gradient) {p.d_vs=a;p.d_vp=b;}return p;
        },true,noise};
}
std::vector<dw::ObservationBlock> manual_blocks() {
    Eigen::Matrix2d a,c;a<<1,1,1,2;c<<4,1,1,9;
    Eigen::Matrix<double,1,2> b;b<<-1,1;
    return {linear_block("first",a,Eigen::Matrix2d::Zero(),{v({2,3}),c},dw::NoiseScale::lognormal(0,.7)),
            linear_block("second",b,Eigen::Matrix<double,1,2>::Zero(),{v({-1}),Eigen::MatrixXd::Constant(1,1,.25)},dw::NoiseScale::exponential(2))};
}
void manual() {
    const auto model=toy_model();dw::FixedVs parameters(model,.8*model.vs,1.2*model.vs,2);
    dw::JointProblem problem(parameters,dw::JointLikelihood(manual_blocks(),independent));
    auto x=problem.initial_state(parameters.encode(model));x.tail(2)=v({.4,-.2});
    const auto value=problem.evaluate(x,true);check(value.status==dw::EvaluationStatus::valid,"Manual joint likelihood");
    const double log2pi=std::log(2*std::acos(-1.));
    const double expected=-.5*(21./35*std::exp(-.8)+std::log(35.)+2*log2pi+4*.4)
                          -.5*(16*std::exp(.4)+std::log(.25)+log2pi-2*.2);
    near(value.log_likelihood,expected,1e-12);
    near(value.residual.squaredNorm(),21./35*std::exp(-.8)+16*std::exp(.4),1e-12);
    near(value.noise_gradient[0],21./35*std::exp(-.8)-2,1e-12);
    near(value.noise_gradient[1],16*std::exp(.4)-1,1e-12);
    check(value.blocks.size()==2 && value.residual.size()==3 && value.jacobian.cols()==2,"Block dimensions");
    // Unit conversion changes density by the expected measure factor only.
    Eigen::Matrix2d c;c<<4,1,1,9;
    dw::GaussianDataLikelihood original({v({2,3}),c}),converted({1000*v({2,3}),1e6*c});
    near(converted.log_likelihood(1000*v({3,5}))-original.log_likelihood(v({3,5})),-2*std::log(1000.),1e-12);
    check(converted.whiten(1000*v({1,2})).isApprox(original.whiten(v({1,2})),1e-12),"Unit conversion residual");
    auto blocks=manual_blocks();std::reverse(blocks.begin(),blocks.end());
    const auto reversed=dw::JointLikelihood(blocks,independent).evaluate(model,v({-.2,.4}),true);
    near(reversed.log_likelihood,expected,1e-12);
    for(auto measure:{dw::PriorMeasure::physical_vs,dw::PriorMeasure::log_vs}) {
        int calls=0;
        const auto inner=dw::gaussian_prior(measure==dw::PriorMeasure::physical_vs?model.vs:parameters.encode(model),v({.3,.4}),measure);
        dw::Prior prior{measure,[&](const Eigen::VectorXd& q,bool gradient){++calls;return inner.evaluate(q,gradient);}};
        const auto out=dw::joint_log_posterior(problem,x,prior,true);
        check(calls==1 && out.status==dw::EvaluationStatus::valid,"Model prior repeated or posterior failed");
        const double noise_prior=-.5*std::pow(.4/.7,2)-std::log(.7)-.5*log2pi-.4+std::log(2.)-2*std::exp(-.2);
        near(out.log_noise_prior,noise_prior,1e-12);
        near(out.log_transform,(measure==dw::PriorMeasure::physical_vs?x.head(2).sum():0)+.2,1e-12);
        near(out.value,expected+out.log_model_prior+noise_prior+out.log_transform,1e-12);
        for(int j=0;j<x.size();++j) {
            auto plus=x,minus=x;plus[j]+=1e-6;minus[j]-=1e-6;
            near(out.gradient[j],(dw::joint_log_posterior(problem,plus,prior).value-dw::joint_log_posterior(problem,minus,prior).value)/2e-6,1e-7);
        }
        const auto density=dw::joint_density(problem,prior)(x);near(density.value,out.value,1e-12);
        check(density.gradient.isApprox(out.gradient,1e-12),"Joint HMC density gradient");
    }
    // Shared Vp/Vs mapping contributes the Vp partials, not just Vs.
    const auto block=linear_block("vp",Eigen::Matrix2d::Identity(),2*Eigen::Matrix2d::Identity(),{model.vs+2*model.vp,Eigen::Matrix2d::Identity()});
    dw::JointProblem ratio(parameters,dw::JointLikelihood({block},independent));
    const auto e=ratio.evaluate(parameters.encode(model),true);
    check(e.jacobian.isApprox((5*model.vs).asDiagonal().toDenseMatrix(),1e-12),"Joint parameter chain rule omitted Vp");
    auto outside=x;outside[0]=parameters.lower()[0]-.1;
    const auto p=dw::joint_log_posterior(problem,outside,dw::uniform_prior(.8*model.vs,1.2*model.vs,dw::PriorMeasure::physical_vs));
    check(p.status==dw::EvaluationStatus::outside_support && p.value==-INFINITY,"Joint support rejection");
}
void single_block() {
    dw::InvLMForward backend;
    for(double ratio:{0.,1.8}) {
        auto model=normal();if(ratio) model.vp=ratio*model.vs;
        dw::FixedVs parameters(model,.8*model.vs,1.1*model.vs,ratio);
        dw::DispersionObservations data;data.queries={{.5,0},{.5,1},{.2,0}};
        const auto f=backend.evaluate(model,data.queries);check(f.all_valid(),"Single-block reference roots");
        data.phase_velocity.resize(3);for(int j=0;j<3;++j) data.phase_velocity[j]=f.rows[j].phase_velocity;
        data.phase_velocity+=v({.001,-.002,.003});
        data.covariance.resize(3,3);data.covariance<<4,1,.5,1,9,1,.5,1,16;data.covariance*=1e-6;
        dw::DispersionProblem old(backend,parameters,data,dw::SensitivityMethod::implicit);
        dw::JointProblem joint(parameters,dw::JointLikelihood({dw::dispersion_block("phase",backend,data,dw::NoiseScale::fixed(std::exp(.2)),dw::SensitivityMethod::implicit)},independent));
        const auto q=parameters.encode(model),a=old.evaluate(q,true,.2).residual;
        const auto b=joint.evaluate(q,true);check(b.status==dw::EvaluationStatus::valid,"Single-block adapter");
        check(a.isApprox(b.residual,1e-13),"Single-block residual changed");
        check(old.evaluate(q,true,.2).jacobian.isApprox(b.jacobian,1e-12),"Single-block Jacobian changed");
        for(auto measure:{dw::PriorMeasure::physical_vs,dw::PriorMeasure::log_vs}) {
            const auto prior=dw::gaussian_prior(measure==dw::PriorMeasure::physical_vs?model.vs:q,Eigen::Vector3d::Constant(.2),measure);
            const auto previous=dw::log_posterior(old,q,prior,true,.2);
            const auto current=dw::joint_log_posterior(joint,q,prior,true);
            near(current.value,previous.value,1e-12);check(current.gradient.isApprox(previous.gradient,1e-11),"Single-block posterior gradient changed");
        }
    }
}
std::vector<dw::ObservationBlock> physical_blocks(const dw::Forward& backend,const dw::LayeredModel& model) {
    dw::DispersionObservations phase;phase.queries={{.2,0},{.5,0},{.5,1},{.8,1}};
    const auto roots=backend.evaluate(model,phase.queries);check(roots.all_valid(),"Physical reference roots");
    phase.phase_velocity.resize(4);for(int i=0;i<4;++i)phase.phase_velocity[i]=roots.rows[i].phase_velocity;
    phase.covariance=Eigen::Matrix4d::Identity()*1e-4;
    const std::vector<dw::DispersionQuery> eq={{.3,0},{.8,0},{.5,1}};
    const auto er=dw::rayleigh_ellipticity(model,eq);Eigen::VectorXd e(3);
    for(int i=0;i<3;++i){check(er[i].status==dw::EllipticityStatus::valid,"Physical reference ellipticity");e[i]=er[i].ratio;}
    dw::ReceiverOptions options;options.samples=256;options.dt=.1;options.pre_arrival=2;
    const auto rf=dw::receiver_function(model,options);check(rf.status==dw::ReceiverStatus::valid,"Physical reference RF");
    const std::vector<int> indices={90,20,31,42,55,70,110};Eigen::VectorXd r(indices.size());
    for(size_t i=0;i<indices.size();++i) r[i]=rf.radial_rf[indices[i]];
    return {dw::dispersion_block("phase",backend,phase),
        dw::ellipticity_block("ellipse","rayleigh_modal_ellipticity",eq,{e,Eigen::Matrix3d::Identity()*.0009}),
        dw::receiver_block("rf",options,indices,{r,Eigen::MatrixXd::Identity(r.size(),r.size())*.0001})};
}
void physical() {
    dw::InvLMForward backend;const auto model=normal();dw::FixedVs parameters(model,.8*model.vs,1.1*model.vs);
    auto blocks=physical_blocks(backend,model);
    dw::JointProblem problem(parameters,dw::JointLikelihood(blocks,independent));const auto q=parameters.encode(model);
    const auto value=problem.evaluate(q);check(value.status==dw::EvaluationStatus::valid,"Three-observable evaluation");
    near(value.residual.squaredNorm(),0,1e-20);
    check(value.blocks[0].prediction.size()==4 && value.blocks[1].prediction.size()==3 && value.blocks[2].prediction.size()==7,"Different sampling grids lost");
    const auto unavailable=problem.evaluate(q,true);
    check(unavailable.status==dw::EvaluationStatus::gradient_failure && unavailable.jacobian.size()==0,"Unsupported model gradient was invented");
    near(unavailable.log_likelihood,value.log_likelihood,1e-12);
    const auto prior=dw::uniform_prior(.8*model.vs,1.1*model.vs,dw::PriorMeasure::physical_vs);
    rejects([&]{dw::joint_density(problem,prior);});
    const auto requested=dw::joint_log_posterior(problem,q,prior,true);
    check(requested.status==dw::EvaluationStatus::gradient_failure && std::isfinite(requested.value) && requested.gradient.size()==0,"Preserve values on gradient failure");
    for(double step:{1e-3,1e-4,1e-5,1e-6}) {
        dw::FiniteDifferenceOptions o;o.step=step;const auto report=problem.least_squares(q,true,o);
        std::cout<<"FD step="<<step<<" status="<<static_cast<int>(report.data.status)<<" max_difference="<<report.max_difference<<" "<<report.data.message<<'\n';
    }
    const auto fd=problem.least_squares(q,true);
    if(fd.data.status!=dw::EvaluationStatus::valid) std::cerr<<fd.data.message<<" evaluations="<<fd.evaluations<<" max_difference="<<fd.max_difference<<'\n';
    check(fd.data.status==dw::EvaluationStatus::valid && fd.evaluations==13,"Checked physical finite differences failed");
    dw::FiniteDifferenceOptions half;half.step=5e-6;
    const auto refined=problem.least_squares(q,true,half);check(refined.data.status==dw::EvaluationStatus::valid,"Physical finite difference refinement");
    const double change=(fd.data.jacobian-refined.data.jacobian).cwiseAbs().maxCoeff();
    check(change<.003,"Physical Jacobian did not stabilize");
    std::cout<<"physical_fd_h_h2="<<fd.max_difference<<" refined_difference="<<change<<'\n';
    const auto fit=dw::bounded_lm([&](const Eigen::VectorXd& trial){return problem.least_squares(trial,true).data;},
        (model.vs*.97).array().log().matrix(),parameters.lower(),parameters.upper(),{100,1e-5,1e-3,1e-12});
    check(fit.converged && fit.evaluation.least_squares_cost<1e-6,"Joint fixed-scale LM failed");
    check((fit.q-q).cwiseAbs().maxCoeff()<1e-5,"Joint LM did not recover synthetic model");
    std::cout<<"joint_LM_cost="<<fit.evaluation.least_squares_cost<<" evaluations="<<fit.evaluations<<'\n';
}
void finite_difference() {
    const auto model=toy_model();dw::FixedVs parameters(model,.8*model.vs,1.2*model.vs,2);
    auto block=manual_blocks()[0];block.noise=dw::NoiseScale::fixed();
    dw::JointProblem exact_problem(parameters,dw::JointLikelihood({block},independent));
    block.verified_gradient=false;
    dw::JointProblem problem(parameters,dw::JointLikelihood({block},independent));
    const auto q=parameters.encode(model);
    for(Eigen::VectorXd point:{q,parameters.lower(),parameters.upper()}) {
        const auto fd=problem.least_squares(point,true);const auto exact=exact_problem.least_squares(point,false);
        check(fd.data.status==dw::EvaluationStatus::valid && exact.data.status==dw::EvaluationStatus::valid,"Boundary finite differences");
        check((fd.data.jacobian-exact.data.jacobian).cwiseAbs().maxCoeff()<1e-7,"Second-order one-sided/central stencil");
    }
    block.verified_gradient=false;
    block.predict=[](const dw::LayeredModel& m,bool) {dw::ModelPrediction p;p.status=dw::EvaluationStatus::valid;p.values=v({std::sin(1e5*std::log(m.vs[0])),0});return p;};
    dw::JointProblem oscillatory(parameters,dw::JointLikelihood({block},independent));
    const auto bad=oscillatory.least_squares(q,true);
    check(bad.data.status==dw::EvaluationStatus::gradient_failure && bad.data.jacobian.size()==0,"Unstable finite difference accepted");
    block.predict=[model](const dw::LayeredModel& m,bool) {dw::ModelPrediction p;p.values=v({1,2});p.status=m.vs.isApprox(model.vs,1e-14)?dw::EvaluationStatus::valid:dw::EvaluationStatus::forward_failure;return p;};
    dw::JointProblem branches(parameters,dw::JointLikelihood({block},independent));
    check(branches.evaluate(q).status==dw::EvaluationStatus::valid,"Reference branch failed");
    check(branches.least_squares(q,true).data.status==dw::EvaluationStatus::gradient_failure,"Failed perturbation silently ignored");
    block.verified_gradient=true;
    block.predict=[](const auto&,bool gradient) {dw::ModelPrediction p;p.values=v({1,2});p.status=gradient?dw::EvaluationStatus::gradient_failure:dw::EvaluationStatus::valid;return p;};
    dw::JointProblem kernel_failure(parameters,dw::JointLikelihood({block},independent));
    check(kernel_failure.evaluate(q).status==dw::EvaluationStatus::valid,"Gradient failure erased valid prediction");
    check(kernel_failure.least_squares(q,true).data.status==dw::EvaluationStatus::gradient_failure,"Failed verified derivative silently replaced by FD");
    block.predict=[](const auto&,bool gradient) {
        dw::ModelPrediction p;p.values=v({1,2});p.status=dw::EvaluationStatus::valid;
        if(gradient) {p.d_vs=Eigen::Matrix2d::Constant(1e308);p.d_vp=Eigen::Matrix2d::Constant(1e308);}return p;
    };
    dw::JointProblem overflow(parameters,dw::JointLikelihood({block},independent));
    check(overflow.least_squares(q,true).data.status==dw::EvaluationStatus::gradient_failure,"Chain-rule overflow accepted by LM");
}
void failures() {
    auto blocks=manual_blocks();rejects([&]{dw::JointLikelihood bad(blocks,dw::BlockDependence::unspecified);});
    rejects([&]{dw::JointLikelihood bad({},independent);});
    rejects([&]{dw::JointLikelihood bad({blocks[0],blocks[0]},independent);});
    auto model=toy_model();dw::FixedVs parameters(model,.8*model.vs,1.2*model.vs,2);
    blocks[1].predict=[](const auto&,bool){dw::ModelPrediction p;p.status=dw::EvaluationStatus::valid;p.values.resize(0);return p;};
    dw::JointProblem missing(parameters,dw::JointLikelihood(blocks,independent));auto x=missing.initial_state(parameters.encode(model));
    const auto m=missing.evaluate(x,true);
    check(m.status==dw::EvaluationStatus::forward_failure && std::isnan(m.log_likelihood) && m.blocks.size()==2 && m.blocks[0].prediction.size()==2,"Missing prediction silently removed");
    const auto prior=dw::uniform_prior(.8*model.vs,1.2*model.vs,dw::PriorMeasure::physical_vs);
    check(std::isnan(dw::joint_log_posterior(missing,x,prior).value),"Numerical forward failure became zero probability");
    x[2]=NAN;check(missing.evaluate(x).status==dw::EvaluationStatus::numerical_failure,"Nonfinite hyperparameter accepted");
    x[2]=1000;check(missing.evaluate(x).status==dw::EvaluationStatus::numerical_failure,"Scale overflow accepted");
    auto b=manual_blocks()[0];b.noise=dw::NoiseScale::lognormal(0,0);rejects([&]{dw::JointLikelihood bad({b},independent);});
    dw::GaussianData data{v({1}),Eigen::MatrixXd::Identity(1,1)};
    rejects([&]{dw::ellipticity_block("hv","ambient_hv",{{1,0}},data);});
    dw::ReceiverOptions options;options.samples=64;options.pre_arrival=1;
    rejects([&]{dw::receiver_block("rf",options,{64},data);});
    rejects([&]{dw::receiver_block("rf",options,{2,2},{v({1,1}),Eigen::Matrix2d::Identity()});});
    data.values[0]=NAN;rejects([&]{dw::receiver_block("rf",options,{1},data);});
    rejects([&]{dw::GaussianDataLikelihood invalid({v({1,2}),Eigen::Matrix2d::Ones()});});
    // A real missing mode must stay a forward failure, even when another block
    // first reported unavailable gradients. Preserve all requested rows.
    dw::InvLMForward backend;model=normal();dw::FixedVs p(model,.8*model.vs,1.1*model.vs);
    auto physical=physical_blocks(backend,model);
    physical.push_back(dw::dispersion_block("missing",backend,{{{.2,999}},v({1}),Eigen::MatrixXd::Identity(1,1)}));
    dw::JointProblem failed(p,dw::JointLikelihood(physical,independent));const auto result=failed.evaluate(p.encode(model),true);
    check(result.status==dw::EvaluationStatus::forward_failure && std::isnan(result.log_likelihood) && result.blocks.size()==4,"Failure precedence lost");
    check(result.blocks.back().prediction.size()==1 && !result.blocks.back().diagnostics.empty(),"Missing mode diagnostics lost");
}
int main(int argc,char** argv) {
    const std::map<std::string,std::function<void()>> tests={{"manual",manual},{"single",single_block},{"physical",physical},{"finite_difference",finite_difference},{"failures",failures}};
    try {if(argc!=2 || !tests.count(argv[1])) throw std::invalid_argument("Test case required");tests.at(argv[1])();std::cout<<"PASS "<<argv[1]<<'\n';return 0;}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
