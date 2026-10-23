#include "dispweave/objective.hpp"
#include "models.hpp"
#include <iostream>
#include <map>
#include <functional>
#include <cmath>
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
void near(double a,double b,double tolerance) {if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tolerance) throw std::runtime_error("Mismatch "+std::to_string(a)+" vs "+std::to_string(b));}
void rejects(const std::function<void()>& fn) {bool caught=false;try{fn();}catch(const std::invalid_argument&){caught=true;}check(caught,"Invalid input accepted");}
Eigen::VectorXd phases(const dw::ForwardResult& f) {
    check(f.all_valid(),"Invalid roots"); Eigen::VectorXd c(f.rows.size());
    for(size_t i=0;i<f.rows.size();++i)c[i]=f.rows[i].phase_velocity; return c;
}
dw::DispersionObservations observations() {
    auto c=phases(dw::InvLMForward().evaluate(normal(),{{.5,0},{.5,1},{.2,0}}));
    c+=v({.001,-.002,.003});
    Eigen::Matrix3d covariance; covariance<<4,1,.5,1,9,1,.5,1,16; covariance*=1e-6;
    return {{{.5,0},{.5,1},{.2,0}},c,covariance};
}
void covariance() {
    Eigen::Matrix2d c; c<<4,1,1,9;
    dw::DispersionObservations o{{{1,0},{2,0}},v({2,3}),c}; dw::GaussianLikelihood l(o);
    const auto predicted=v({3,5}); // residual [1,2], determinant 35, quadratic 21/35
    const double expected=-.5*(21./35+std::log(35.)+2*std::log(2*std::acos(-1.)));
    near(l.log_likelihood(predicted),expected,1e-13);
    near(l.whiten(v({1,2})).squaredNorm(),21./35,1e-13);
    const double scale=.4;
    near(l.log_likelihood(predicted,scale),-.5*(21./35*std::exp(-2*scale)+std::log(35.)+4*scale+2*std::log(2*std::acos(-1.))),1e-13);
    near(l.log_scale_gradient(predicted,scale),(l.log_likelihood(predicted,scale+1e-5)-l.log_likelihood(predicted,scale-1e-5))/2e-5,1e-9);
    Eigen::Matrix2d j; j<<1,2,3,4;
    const auto white=l.whiten_jacobian(j);
    near(white.col(0).squaredNorm(),39./35,1e-13);
    for(int kind=0;kind<6;++kind) {auto bad=o;
        if(kind==0)bad.covariance(0,1)=2;
        if(kind==1)bad.covariance(0,0)=-1;
        if(kind==2)bad.covariance.setOnes();
        if(kind==3)bad.covariance(0,0)=NAN;
        if(kind==4)bad.phase_velocity.resize(1);
        if(kind==5)bad.queries[0].mode=-1;
        rejects([&]{dw::GaussianLikelihood invalid(bad);});
    }
    rejects([&]{l.log_likelihood(v({1}));});
    rejects([&]{l.log_likelihood(predicted,NAN);});
}
void parameterization() {
    const auto bounds=dw::reference_vs_bounds(v({.7,3.,4.6}),.5,5.,1.2);
    check(bounds.lower.isApprox(v({.5,2.4,4.0}),1e-14) &&
          bounds.upper.isApprox(v({1.3,3.6,5.0}),1e-14),"Global/window intersection, including halfspace");
    for(const auto& values:std::vector<std::vector<double>>{{0,5,1},{1,1,1},{2,1,1},{.5,5,0},{.5,5,-1},
                                                        {NAN,5,1},{.5,INFINITY,1},{.5,5,NAN}})
        rejects([&]{dw::reference_vs_bounds(v({1,3}),values[0],values[1],values[2]);});
    rejects([&]{dw::reference_vs_bounds(v({1,6}),.5,5,1);}); // empty intersection
    rejects([&]{dw::reference_vs_bounds(v({1,5.2}),.5,5,1);}); // reference outside intersection
    rejects([&]{dw::reference_vs_bounds(v({1,NAN}),.5,5,1);});
    rejects([&]{dw::reference_vs_bounds(Eigen::VectorXd(),.5,5,1);});
    dw::InvLMForward backend; const std::vector<dw::DispersionQuery> query{{.5,0},{.5,1},{.2,0}};
    for(double ratio:{0.,1.8}) {
        auto m=normal(); if(ratio>0)m.vp=ratio*m.vs;
        dw::FixedVs p(m,.8*m.vs,1.1*m.vs,ratio); const auto q=p.encode(m);
        check(p.decode(q).vs.isApprox(m.vs),"Roundtrip");
        const auto f=backend.evaluate(m,query,dw::SensitivityMethod::implicit);
        const auto jac=p.phase_jacobian(q,f);
        for(int col=0;col<q.size();++col)for(double h:{1e-3,1e-4,1e-5}) {
            Eigen::VectorXd plus=q,minus=q; plus[col]+=h;minus[col]-=h;
            const Eigen::VectorXd fd=(phases(backend.evaluate(p.decode(plus),query))-phases(backend.evaluate(p.decode(minus),query)))/(2*h);
            check((jac.col(col)-fd).cwiseAbs().maxCoeff()<3e-4,"Chain rule central difference");
        }
        near(p.log_abs_det_jacobian(q),q.sum(),0);
        for(int i=0;i<q.size();++i){Eigen::VectorXd plus=q,minus=q;plus[i]+=1e-5;minus[i]-=1e-5;
            near(p.log_abs_det_gradient(q)[i],(p.log_abs_det_jacobian(plus)-p.log_abs_det_jacobian(minus))/2e-5,1e-10);}
        auto wrong=m;wrong.depth[1]+=.1;rejects([&]{p.encode(wrong);});
        wrong=m;wrong.rho[0]+=.1;rejects([&]{p.encode(wrong);});
        wrong=m;wrong.vp[0]+=.1;rejects([&]{p.encode(wrong);});
        rejects([&]{p.decode(v({1}));});
    }
    rejects([&]{dw::FixedVs p(normal(),v({0,1,2}),v({1,2,3}));});
    rejects([&]{dw::FixedVs p(normal(),v({.5,1,2}),v({2,3,4}));});
}
void posterior() {
    dw::InvLMForward backend;auto m=normal();dw::FixedVs parameters(m,.8*m.vs,1.1*m.vs);
    dw::DispersionProblem problem(backend,parameters,observations(),dw::SensitivityMethod::implicit);
    const auto q=parameters.encode(m);
    for(auto measure:{dw::PriorMeasure::physical_vs,dw::PriorMeasure::log_vs}) {
        const Eigen::VectorXd mean=measure==dw::PriorMeasure::physical_vs?m.vs:q;
        const auto prior=dw::gaussian_prior(mean,Eigen::VectorXd::Constant(3,.2),measure);
        const auto p=dw::log_posterior(problem,q,prior,true,.2);
        check(p.status==dw::EvaluationStatus::valid,"Posterior valid");
        near(p.log_prior,-1.5*std::log(2*std::acos(-1.))-3*std::log(.2),1e-12);
        near(p.log_transform,measure==dw::PriorMeasure::physical_vs?q.sum():0,0);
        near(p.value,p.log_likelihood+p.log_prior+p.log_transform,1e-12);
        for(int i=0;i<q.size();++i) {
            Eigen::VectorXd plus=q,minus=q;plus[i]+=1e-6;minus[i]-=1e-6;
            near(p.gradient[i],(dw::log_posterior(problem,plus,prior,false,.2).value-dw::log_posterior(problem,minus,prior,false,.2).value)/2e-6,1e-3);
        }
        near(p.log_scale_gradient,(dw::log_posterior(problem,q,prior,false,.200001).value-dw::log_posterior(problem,q,prior,false,.199999).value)/2e-6,1e-7);
        near(problem.evaluate(q,false).least_squares_cost,.5*problem.evaluate(q,false).residual.squaredNorm(),0);
        auto outside=q;outside[0]=parameters.lower()[0]-.01;
        const auto rejected=dw::log_posterior(problem,outside,prior);
        check(rejected.status==dw::EvaluationStatus::outside_support && rejected.value==-INFINITY,"Support rejection");
    }
    auto prior=dw::uniform_prior(.8*m.vs,1.1*m.vs,dw::PriorMeasure::physical_vs);
    near(prior.evaluate(m.vs,true).value,-(.3*m.vs).array().log().sum(),1e-14);
    const auto uniform=dw::log_posterior(problem,q,prior);
    near((uniform.gradient+uniform.data.jacobian.transpose()*uniform.data.residual-Eigen::VectorXd::Ones(3)).norm(),0,1e-12);
    auto too_narrow=dw::uniform_prior(.8*m.vs,.9*m.vs,dw::PriorMeasure::physical_vs);
    check(dw::log_posterior(problem,q,too_narrow).status==dw::EvaluationStatus::outside_support,"Prior support");
    dw::Prior broken{dw::PriorMeasure::log_vs,[](const auto&,bool){return dw::PriorEvaluation{0,{}};}};
    check(dw::log_posterior(problem,q,broken).status==dw::EvaluationStatus::gradient_failure,"Missing prior gradient");
}
class FailingGradient final : public dw::Forward {
public:
    dw::ForwardResult evaluate(const dw::LayeredModel& m,const std::vector<dw::DispersionQuery>& q,dw::SensitivityMethod method) const override {
        auto f=dw::InvLMForward().evaluate(m,q);
        if(method!=dw::SensitivityMethod::none) for(auto& r:f.rows)r.sensitivity.status=dw::SensitivityStatus::unavailable;
        return f;
    }
};
void failures() {
    const auto m=normal();dw::FixedVs parameters(m,.8*m.vs,1.1*m.vs);const auto q=parameters.encode(m);
    FailingGradient backend;dw::DispersionProblem problem(backend,parameters,observations());
    const auto e=problem.evaluate(q);
    check(e.status==dw::EvaluationStatus::gradient_failure,"Gradient failure status");
    check(e.forward.all_valid() && std::isfinite(e.log_likelihood),"Roots/likelihood preserved on gradient failure");
    check(problem.evaluate(q,false).status==dw::EvaluationStatus::valid,"Value-only still works");
    auto o=observations();o.queries[0].mode=999;dw::InvLMForward physics;
    dw::DispersionProblem bad(physics,parameters,o);
    const auto prior=dw::gaussian_prior(q,Eigen::VectorXd::Ones(3),dw::PriorMeasure::log_vs);
    const auto p=dw::log_posterior(bad,q,prior);
    check(p.status==dw::EvaluationStatus::forward_failure && std::isnan(p.value),"Unresolved root is not zero probability");
    check(p.data.forward.rows.size()==3,"No discarded observations");
    auto malformed=q;malformed[0]=NAN;
    check(dw::log_posterior(problem,malformed,prior).status==dw::EvaluationStatus::numerical_failure,"NaN q is numerical failure");
}
int main(int argc,char** argv) {
    const std::map<std::string,std::function<void()>> tests={{"covariance",covariance},{"parameterization",parameterization},{"posterior",posterior},{"objective_failures",failures}};
    try{if(argc!=2||!tests.count(argv[1]))throw std::invalid_argument("Test name");tests.at(argv[1])();std::cout<<"PASS "<<argv[1]<<'\n';return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
