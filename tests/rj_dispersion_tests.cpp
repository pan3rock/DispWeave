#include "dispweave/rj_dispersion.hpp"
#include <iostream>
#include <stdexcept>
using namespace dw;
Eigen::VectorXd v(std::initializer_list<double> x) {return Eigen::Map<const Eigen::VectorXd>(x.begin(),x.size());}
void check(bool x,const char* m) {if(!x) throw std::runtime_error(m);}
int main() {try {
    RJPrior p;p.max_layers=3;p.layer_weights=v({1,1,1});p.vs_lower=.7;p.vs_upper=3.5;p.min_spacing=.1;p.depth_limit=8;
    InvLMForward backend;
    DispersionObservations data{{{.2,0},{.5,0},{.2,1}},v({1.3,1.,2.5}),Eigen::MatrixXd::Identity(3,3)*.01};
    data.covariance(0,1)=data.covariance(1,0)=.003;
    RJDispersionProblem rj(backend,p,data,1.8,2.5);
    RJState state{v({2,4}),v({1,2,3})};
    const auto m=rj.model(state);
    check(m.depth==v({0,2,4}) && m.vp.isApprox(1.8*m.vs) && m.rho==v({2.5,2.5,2.5}),"Physical mapping");
    FixedVs fixed(m,v({.7,.7,.7}),v({3.5,3.5,3.5}),1.8);
    DispersionProblem old(backend,fixed,data,SensitivityMethod::implicit);
    const auto a=rj.evaluate(state);const auto b=old.evaluate(state.vs.array().log().matrix(),false);
    check(a.likelihood.status==EvaluationStatus::valid && b.status==EvaluationStatus::valid,"Layered predictions failed");
    check(std::abs(a.likelihood.value-b.log_likelihood)<1e-12,"RJ/fixed likelihood mismatch");
    check(a.forward.rows.size()==3 && a.forward.sensitivity_method==SensitivityMethod::none,"Unexpected gradient or deleted query");
    RJState halfspace{v({}),v({2})};
    const auto missing=rj.evaluate(halfspace);
    check(missing.likelihood.status==EvaluationStatus::forward_failure && missing.forward.rows.size()==3,"Missing mode hidden");
    const auto outside=rj.evaluate({v({.01}),v({1,2})});
    check(outside.likelihood.status==EvaluationStatus::outside_support && outside.forward.rows.empty(),"Outside support evaluated");
    // Independent halfspace Rayleigh cubic; fixed Vp/Vs yields frequency-independent c/Vs.
    const double beta=1/3.24;double lo=0,hi=1;
    for(int i=0;i<60;++i) {const double x=(lo+hi)/2;const double d=x*x*x-8*x*x+(24-16*beta)*x-16*(1-beta);if(d>0) hi=x;else lo=x;}
    const double expected=2*std::sqrt((lo+hi)/2);
    DispersionObservations simple{{{.1,0},{1.,0}},v({expected+.02,expected-.01}),Eigen::MatrixXd::Identity(2,2)*.0025};
    RJDispersionProblem analytic(backend,p,simple,1.8,2.5);
    const auto result=analytic.evaluate(halfspace);
    check(result.likelihood.status==EvaluationStatus::valid,"Halfspace forward failed");
    check((result.prediction.array()-expected).abs().maxCoeff()<1e-7,"Independent cubic mismatch");
    const double exact=-.5*(.0005/.0025+2*std::log(2*std::acos(-1.)*.0025));
    check(std::abs(result.likelihood.value-exact)<1e-6,"Gaussian normalization mismatch");
    bool caught=false;try {RJDispersionProblem invalid(backend,p,data,1.,2.5);} catch(const std::invalid_argument&) {caught=true;}
    check(caught,"Unphysical Vp/Vs accepted");
    std::cout<<"RJ physical model, independent halfspace, covariance likelihood and missing modes passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
