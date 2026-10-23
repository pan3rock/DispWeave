#include "dispweave/nuts.hpp"
#include "dispweave/regularization.hpp"
#include <iostream>
#include <stdexcept>
using Eigen::VectorXd;
VectorXd v(std::initializer_list<double> x) {return Eigen::Map<const VectorXd>(x.begin(),x.size());}
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
void near(double a,double b,double tolerance) {check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance,"Numerical mismatch");}
void rejects(const std::function<void()>& f) {bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}check(caught,"Malformed prior accepted");}

void normalization() {
    for(auto measure:{dw::PriorMeasure::physical_vs,dw::PriorMeasure::log_vs})
    for(const auto& bounds:std::vector<std::pair<double,double>>{{1.7,2.4},{2.,2.4},{1.999999,2.000002},{.1,20.}}) {
        const double lo=bounds.first,hi=bounds.second;
        const auto prior=dw::bounded_gaussian_prior(v({2}),v({.3}),v({lo}),v({hi}),measure);
        // Composite Simpson quadrature of the actual density, independent of its CDF normalizer.
        const int n=20000;double integral=0;
        for(int i=0;i<=n;++i) integral+=(i==0||i==n?1:i%2?4:2)*std::exp(prior.evaluate(v({lo+(hi-lo)*i/n}),false).value);
        near(integral*(hi-lo)/(3*n),1.,2e-9);
        check(prior.evaluate(v({lo-.01}),true).value==-INFINITY,"Lower support");
        check(prior.evaluate(v({hi+.01}),true).value==-INFINITY,"Upper support");
        const double x=(lo+hi)/2,h=(hi-lo)*1e-5;
        near(prior.evaluate(v({x}),true).gradient[0],
             (prior.evaluate(v({x+h}),false).value-prior.evaluate(v({x-h}),false).value)/(2*h),1e-4);
    }
    const auto mean=v({1.,2.,3.}),sd=v({.1,.1,.1}),x=v({1.05,1.9,3.2});
    const auto prior=dw::bounded_gaussian_prior(mean,sd,mean*.7,mean*1.3,dw::PriorMeasure::physical_vs);
    // Physical-density difference exactly agrees with explicit reference damping.
    dw::VsRegularization penalty(v({0,1,3}),mean,"reference_damping",100.);
    near(prior.evaluate(mean,false).value-prior.evaluate(x,false).value,
         .5*penalty.evaluate(x.array().log()).residual.squaredNorm(),1e-12);
    rejects([&]{dw::bounded_gaussian_prior(mean,v({.1}),mean*.7,mean*1.3,dw::PriorMeasure::physical_vs);});
    for(double bad:std::vector<double>{0.,-1.,NAN,INFINITY}) rejects([&]{dw::bounded_gaussian_prior(v({2}),v({bad}),v({1}),v({3}),dw::PriorMeasure::physical_vs);});
    rejects([&]{dw::bounded_gaussian_prior(v({0}),v({1}),v({1}),v({3}),dw::PriorMeasure::physical_vs);});
    rejects([&]{dw::bounded_gaussian_prior(v({2}),v({1}),v({3}),v({1}),dw::PriorMeasure::physical_vs);});
    rejects([&]{prior.evaluate(v({1}),true);});
    rejects([&]{prior.evaluate(v({1,NAN,3}),true);});
}

void physical() {
    // Homogeneous Rayleigh cubic gives c=k*Vs independently of invLM.
    const double ratio=1.8,beta=1/(ratio*ratio);
    auto polynomial=[&](double x) {return x*x*x-8*x*x+(24-16*beta)*x-16*(1-beta);};
    double left=0,right=1;
    for(int i=0;i<60;++i) {const double mid=(left+right)/2;if(polynomial(mid)>0)right=mid;else left=mid;}
    const double k=std::sqrt((left+right)/2),sigma=.15,observed=2.45*k,prior_sd=.3,lo=1.5,hi=2.5;
    dw::LayeredModel model{v({0}),v({2}),v({3.6}),v({2.5})};
    dw::FixedVs parameters(model,v({lo}),v({hi}),ratio);
    dw::InvLMForward backend;
    dw::DispersionProblem problem(backend,parameters,{{{.5,0}},v({observed}),Eigen::MatrixXd::Constant(1,1,sigma*sigma)},dw::SensitivityMethod::implicit);
    const auto prior=dw::bounded_gaussian_prior(v({2}),v({prior_sd}),v({lo}),v({hi}),dw::PriorMeasure::physical_vs);
    dw::LogisticBox box(parameters.lower(),parameters.upper());
    const auto target=dw::transform_density(dw::dispersion_density(problem,prior),box);
    for(double vs:{1.6,2.1,2.4}) {
        const auto q=v({std::log(vs)}),z=box.inverse(q);const auto transformed=box.forward(z);
        const auto e=target(z);const auto physical=dw::log_posterior(problem,q,prior);
        const double normal_mass=.5*(std::erf((hi-2)/(prior_sd*std::sqrt(2.)))-std::erf((lo-2)/(prior_sd*std::sqrt(2.))));
        const double expected=-.5*std::pow((k*vs-observed)/sigma,2)-std::log(sigma*std::sqrt(2*std::acos(-1.)))
                              -.5*std::pow((vs-2)/prior_sd,2)-std::log(prior_sd*std::sqrt(2*std::acos(-1.))*normal_mass);
        near(physical.value,expected+std::log(vs),1e-8);
        near(e.value,expected+std::log(vs)+transformed.log_jacobian,1e-8);
        for(double h:{1e-4,1e-5}) {auto a=z,b=z;a[0]+=h;b[0]-=h;near(e.gradient[0],(target(a).value-target(b).value)/(2*h),1e-6);}
    }
    const double precision=k*k/(sigma*sigma)+1/(prior_sd*prior_sd),sd=1/std::sqrt(precision);
    const double mean=(k*observed/(sigma*sigma)+2/(prior_sd*prior_sd))/precision;
    auto phi=[](double x) {return std::exp(-.5*x*x)/std::sqrt(2*std::acos(-1.));};
    const double a=(lo-mean)/sd,b=(hi-mean)/sd,mass=.5*(std::erf(b/std::sqrt(2.))-std::erf(a/std::sqrt(2.)));
    const double shift=(phi(a)-phi(b))/mass,expected_mean=mean+sd*shift;
    const double variance=sd*sd*(1+(a*phi(a)-b*phi(b))/mass-shift*shift);
    for(int seed:{17,2026,99173}) {
        dw::NUTSOptions options;options.seed=seed;options.warmup=600;options.draws=12000;options.target_accept=.95;
        auto sample=dw::sample_nuts(target,box.inverse(v({std::log(1.8)})),dw::EuclideanMetric(Eigen::MatrixXd::Identity(1,1)),options);
        VectorXd values(options.draws);
        for(int i=0;i<options.draws;++i) values[i]=std::exp(box.forward(sample.draws.row(i).transpose()).q[0]);
        const double got_variance=(values.array()-values.mean()).square().sum()/(options.draws-1);
        // Fixed bounds based on analytical posterior scale, checked over three independent seeds.
        near(values.mean(),expected_mean,.06*std::sqrt(variance));near(got_variance,variance,.10*variance);
        for(const auto& transition:sample.transitions) if(!transition.warmup)
            check(!transition.divergent && !transition.max_depth_reached && transition.status==dw::EvaluationStatus::valid,"Analytical posterior sampling failed");
        std::cout<<"seed="<<seed<<" mean="<<values.mean()<<" expected="<<expected_mean
                 <<" variance="<<got_variance<<" expected="<<variance<<'\n';
    }
}
int main(int argc,char** argv) {try {
    if(argc!=2)throw std::invalid_argument("Test case required");
    const std::string name=argv[1];if(name=="normalization")normalization();else if(name=="physical")physical();else throw std::invalid_argument("Unknown case");
    std::cout<<"PASS reference prior "<<name<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
