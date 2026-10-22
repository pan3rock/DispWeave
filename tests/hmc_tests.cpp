#include "dispweave/hmc.hpp"
#include "../src/sampler_adaptation.hpp"
#include <Eigen/Cholesky>
#include <iostream>
#include <stdexcept>
using Eigen::VectorXd;
using Eigen::MatrixXd;
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
VectorXd v(std::initializer_list<double> x) {return Eigen::Map<const VectorXd>(x.begin(),x.size());}
dw::LogDensity gaussian(MatrixXd precision) {
    return [precision=std::move(precision)](const VectorXd& x) {
        return dw::DensityEvaluation{dw::EvaluationStatus::valid,-.5*x.dot(precision*x),-precision*x};
    };
}
// Fixed before experiments: 60 batch means; each mean and raw second moment
// within max(0.04, 6 batch MCSE), MCSE < 0.10. No iid SE assumptions.
void moment(const VectorXd& values,double truth) {
    const int batches=60,n=values.size()/batches; VectorXd b(batches);
    for(int i=0;i<batches;++i) b[i]=values.segment(i*n,n).mean();
    const double se=std::sqrt((b.array()-b.mean()).square().sum()/(batches*(batches-1)));
    std::cout<<" moment="<<values.mean()<<" truth="<<truth<<" mcse="<<se;
    check(se<.10,"moment MCSE too large");
    check(std::abs(values.mean()-truth)<std::max(.04,6*se),"moment outside statistical tolerance");
}
int main(int argc,char** argv) {try {
    const std::string test=argc>1?argv[1]:"";
    const MatrixXd identity=MatrixXd::Identity(2,2);
    const auto target=gaussian(identity); const dw::EuclideanMetric metric(identity);
    if(test=="mechanics") {
        MatrixXd mass(2,2); mass<<2,.4,.4,1;
        dw::EuclideanMetric dense(mass);
        dw::HamiltonianState s{v({.7,-1.1}),v({.3,.8}),target(v({.7,-1.1}))};
        auto a=dw::leapfrog(target,s,dense,.03,37);
        auto b=dw::leapfrog(target,a.state,dense,-.03,37);
        check((b.state.position-s.position).norm()<2e-13 && (b.state.momentum-s.momentum).norm()<2e-13,"leapfrog reversibility");
        double previous=0;
        for(int n:{10,20,40,80}) {
            auto r=dw::leapfrog(target,s,dense,1./n,n);
            const double error=std::abs(-r.state.density.value+dense.kinetic(r.state.momentum)+s.density.value-dense.kinetic(s.momentum));
            if(previous) check(previous/error>3.8 && previous/error<4.2,"second order energy convergence");
            previous=error;
        }
        std::mt19937_64 rng(5); MatrixXd samples(50000,2);
        for(int i=0;i<samples.rows();++i) samples.row(i)=dense.momentum(rng);
        check((samples.transpose()*samples/samples.rows()-mass).norm()<.04,"momentum mass convention");
        check((mass*dense.velocity(v({.3,.8}))-v({.3,.8})).norm()<1e-14,"inverse mass action");
    } else if(test=="transform") {
        dw::LogisticBox box(v({-2,1}),v({3,4})); const auto z=v({-.8,1.3}); auto t=box.forward(z);
        check((box.inverse(t.q)-z).norm()<1e-14,"box roundtrip");
        auto f=dw::transform_density(target,box); auto e=f(z);
        for(int k=0;k<2;++k) {auto plus=z,minus=z; plus[k]+=1e-5; minus[k]-=1e-5;
            check(std::abs((f(plus).value-f(minus).value)/2e-5-e.gradient[k])<1e-8,"transformed gradient");}
        auto tails=box.forward(v({-1000,1000}));
        check(std::isfinite(tails.log_jacobian) && tails.derivative.allFinite(),"stable transform tails");
        check(f(v({-1000,1000})).status==dw::EvaluationStatus::numerical_failure,"saturated transform diagnosed");
        auto uniform=dw::transform_density([](const VectorXd& q) {
            return dw::DensityEvaluation{dw::EvaluationStatus::valid,-std::log(15.),VectorXd::Zero(q.size())};
        },box);
        dw::HMCOptions o; o.draws=12000;o.leapfrog_steps=7;
        auto r=dw::sample_hmc(uniform,v({0,0}),metric,o);
        VectorXd scaled(o.draws);
        for(int i=0;i<o.draws;++i) scaled[i]=(box.forward(r.draws.row(i).transpose()).q[0]+2)/5;
        moment(scaled,.5); moment(scaled.array().square().matrix(),1./3.); std::cout<<'\n';
    } else if(test=="distributions") {
        for(const auto seed:{17,2026,99173}) for(int kind=0;kind<3;++kind) {
            MatrixXd cov=identity; if(kind==1) cov<<1,.995,.995,1;
            MatrixXd precision=cov.llt().solve(identity);
            dw::LogDensity f=gaussian(precision);
            if(kind==2) f=[](const VectorXd& q) {
                const double y=q[1]-.5*(q[0]*q[0]-1);
                return dw::DensityEvaluation{dw::EvaluationStatus::valid,-.5*(q[0]*q[0]+y*y),v({-q[0]+q[0]*y,-y})};
            };
            dw::HMCOptions o; o.seed=seed;o.draws=12000;o.warmup=800;o.leapfrog_steps=7;o.target_accept=kind==2?.99:.95;
            auto r=dw::sample_hmc(f,v({.4,-.3}),dw::EuclideanMetric(precision),o);
            int accepted=0,divergent=0;
            for(const auto& t:r.transitions) if(!t.warmup) {
                accepted+=t.accepted;divergent+=t.divergent;
                check(t.step_size>=r.step_size*(1-o.step_jitter) && t.step_size<=r.step_size*(1+o.step_jitter),"frozen nominal sampling step");
            }
            std::cout<<"seed="<<seed<<" kind="<<kind<<" accept="<<double(accepted)/o.draws<<" divergences="<<divergent;
            check(accepted>.55*o.draws,"acceptance too low"); check(divergent==0,"analytic sampling divergence");
            MatrixXd x=r.draws;
            if(kind==1) x=cov.llt().matrixL().solve(x.transpose()).transpose().eval();
            for(int k=0;k<2;++k) {moment(x.col(k),0);moment(x.col(k).array().square().matrix(),kind==2 && k==1?1.5:1.);}
            moment((x.col(0).array()*x.col(1).array()).matrix(),0); std::cout<<'\n';
        }
    } else if(test=="adaptation") {
        const auto windows=dw::detail::warmup_windows(1000);
        check(windows.front().begin==75 && windows.back().end==950,"warmup fast buffers");
        for(size_t i=1;i<windows.size();++i) check(windows[i].begin==windows[i-1].end,"contiguous covariance windows");
        dw::detail::CovarianceAccumulator accumulator(2);
        for(auto x:{v({1,2}),v({3,4}),v({5,8})}) accumulator.add(x);
        MatrixXd points(3,2);points<<1,2,3,4,5,8;
        points.rowwise()-=points.colwise().mean().eval();
        MatrixXd covariance=3./8.*points.transpose()*points/2;
        covariance.diagonal().array()+=5./8.*1e-3;
        check((accumulator.mass(dw::MetricAdaptation::dense)*covariance-identity).norm()<1e-11,"Welford covariance and mass orientation");
        for(int kind=0;kind<2;++kind) for(auto seed:{17,2026,99173}) {
            MatrixXd cov=MatrixXd::Zero(2,2);cov.diagonal()=v({.01,100});
            if(kind==1) {MatrixXd rotation(2,2);rotation<<.8,-.6,.6,.8;cov=(rotation*cov*rotation.transpose()).eval();}
            const MatrixXd precision=cov.llt().solve(identity);
            dw::HMCOptions o;o.seed=seed;o.warmup=2000;o.draws=12000;o.leapfrog_steps=7;o.target_accept=.95;
            o.metric_adaptation=kind==0?dw::MetricAdaptation::diagonal:dw::MetricAdaptation::dense;
            const auto r=dw::sample_hmc(gaussian(precision),v({0,0}),metric,o);
            check(!r.metric_updates.empty() && r.metric_updates.back().warmup_iteration==1950,"metric frozen before final buffer");
            check(r.mass.isApprox(r.metric_updates.back().mass,1e-12),"final mass frozen");
            int divergent=0;for(auto t:r.transitions) if(!t.warmup) divergent+=t.divergent;
            check(divergent==0,"adapted Gaussian divergences");
            MatrixXd x=cov.llt().matrixL().solve(r.draws.transpose()).transpose();
            std::cout<<"adapt kind="<<kind<<" seed="<<seed<<" step="<<r.step_size;
            for(int k=0;k<2;++k) {moment(x.col(k),0);moment(x.col(k).array().square().matrix(),1);}
            moment((x.col(0).array()*x.col(1).array()).matrix(),0);std::cout<<'\n';
        }
        dw::HMCOptions short_run;short_run.warmup=200;short_run.draws=20;short_run.metric_adaptation=dw::MetricAdaptation::diagonal;
        const auto first=dw::sample_hmc(target,v({0,0}),metric,short_run);
        short_run.draws=40;const auto second=dw::sample_hmc(target,v({0,0}),metric,short_run);
        check(first.mass==second.mass && first.draws==second.draws.topRows(20),"draw budget cannot change warmup or mass");
        short_run.warmup=10;bool rejected=false;
        try {dw::sample_hmc(target,v({0,0}),metric,short_run);} catch(const std::invalid_argument&) {rejected=true;}
        check(rejected,"too short metric adaptation rejected");
    } else if(test=="physical") {
        // Independent homogeneous Rayleigh cubic, not an invLM golden value.
        const double beta=1/(1.8*1.8);
        auto secular=[&](double x) {return x*x*x-8*x*x+(24-16*beta)*x-16*(1-beta);};
        double lo=0,hi=1;
        for(int i=0;i<60;++i) {double mid=(lo+hi)/2;if(secular(mid)>0) hi=mid;else lo=mid;}
        const double k=std::sqrt((lo+hi)/2),sigma=.05,observed=2*k+.02;
        dw::LayeredModel model{v({0}),v({2}),v({3.6}),v({2.5})};
        dw::FixedVs parameters(model,v({1}),v({3}),1.8);
        dw::InvLMForward backend;
        dw::DispersionProblem problem(backend,parameters,{{{.5,0}},v({observed}),MatrixXd::Constant(1,1,sigma*sigma)},dw::SensitivityMethod::implicit);
        dw::LogisticBox box(parameters.lower(),parameters.upper());
        auto f=dw::transform_density(dw::dispersion_density(problem,dw::uniform_prior(v({1}),v({3}),dw::PriorMeasure::physical_vs)),box);
        auto z=box.inverse(v({std::log(2.)}));auto plus=z,minus=z;plus[0]+=1e-5;minus[0]-=1e-5;
        check(std::abs((f(plus).value-f(minus).value)/2e-5-f(z).gradient[0])<1e-5,"physical posterior transformed gradient");
        dw::HMCOptions o;o.draws=12000;o.target_accept=.95;o.leapfrog_steps=7;
        const auto r=dw::sample_hmc(f,z,dw::EuclideanMetric(MatrixXd::Identity(1,1)),o);
        VectorXd standardized(o.draws);
        for(int i=0;i<o.draws;++i) standardized[i]=(std::exp(box.forward(r.draws.row(i).transpose()).q[0])-observed/k)/(sigma/k);
        moment(standardized,0);moment(standardized.array().square().matrix(),1);std::cout<<'\n';
        for(auto t:r.transitions) if(!t.warmup) check(!t.divergent,"physical posterior divergence");
    } else if(test=="failures") {
        dw::HMCOptions o;o.warmup=0;o.draws=100;o.adapt_step_size=false;o.step_size=.2;
        auto a=dw::sample_hmc(target,v({0,0}),metric,o),b=dw::sample_hmc(target,v({0,0}),metric,o);
        check(a.draws==b.draws,"seed reproducibility");
        o.chain_id=1; auto c=dw::sample_hmc(target,v({0,0}),metric,o);
        check(a.draws!=c.draws,"independent chain stream");
        auto broken=[](const VectorXd& q) {return dw::DensityEvaluation{q.norm()==0?dw::EvaluationStatus::valid:dw::EvaluationStatus::gradient_failure,0,VectorXd::Zero(q.size())};};
        auto failed=dw::sample_hmc(broken,v({0,0}),metric,o);
        check(failed.draws.norm()==0,"rejected states retained");
        for(auto t:failed.transitions) check(!t.accepted && t.divergent && t.status==dw::EvaluationStatus::gradient_failure,"failure diagnosis");
        int caught=0;
        try {dw::LogisticBox bad(v({0}),v({1,2}));} catch(const std::invalid_argument&) {++caught;}
        try {dw::EuclideanMetric bad(-identity);} catch(const std::invalid_argument&) {++caught;}
        try {dw::LogisticBox(v({0}),v({1})).inverse(v({0}));} catch(const std::invalid_argument&) {++caught;}
        try {dw::sample_hmc(broken,v({1,0}),metric,o);} catch(const std::invalid_argument&) {++caught;}
        try {dw::sample_hmc([](const VectorXd&) {return dw::DensityEvaluation{dw::EvaluationStatus::valid,0,VectorXd()};},v({0,0}),metric,o);} catch(const std::invalid_argument&) {++caught;}
        check(caught==5,"invalid inputs rejected");
        dw::HMCOptions adapted;adapted.warmup=50;adapted.draws=20;adapted.step_jitter=0;
        auto frozen=dw::sample_hmc(target,v({0,0}),metric,adapted);
        for(auto t:frozen.transitions) if(!t.warmup) check(t.step_size==frozen.step_size,"no adaptation after warmup");
        o.step_size=10; auto unstable=dw::sample_hmc(target,v({1,1}),metric,o);
        for(auto t:unstable.transitions) check(t.divergent && !t.accepted,"unstable dynamics diagnosed");
    } else throw std::runtime_error("Unknown test");
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
