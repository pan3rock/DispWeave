#include "dispweave/nuts.hpp"
#include "../src/nuts_multinomial.hpp"
#include "../src/sampler_adaptation.hpp"
#include <Eigen/LU>
#include <iostream>
#include <stdexcept>
using namespace dw;
void require(bool b,const char* msg) {if(!b) throw std::runtime_error(msg);}
void moment(const Eigen::VectorXd& x,double expected) {
    Eigen::VectorXd means(60);const int width=x.size()/60;
    for(int i=0;i<60;++i) means[i]=x.segment(i*width,width).mean();
    const double se=std::sqrt((means.array()-means.mean()).square().sum()/(60.*59));
    require(se<.1 && std::abs(x.mean()-expected)<std::max(.04,6*se),"Posterior moment changed");
}
int main() {try {
    const Eigen::MatrixXd eye=Eigen::Matrix2d::Identity();EuclideanMetric identity(eye);
    const LogDensity normal=[](const Eigen::VectorXd& q) {return DensityEvaluation{EvaluationStatus::valid,-.5*q.squaredNorm(),-q};};
    const Eigen::VectorXd zero=Eigen::Vector2d::Zero();std::mt19937_64 rng(17);
    int calls=0;const LogDensity counted=[&](const Eigen::VectorXd& q) {++calls;return normal(q);};
    const auto search=detail::find_conservative_step(counted,zero,normal(zero),identity,rng,100.,0);
    require(search.step_size<=100 && search.step_size>0 && search.evaluations==calls && calls%4==0 && calls<=48,"Conservative search budget");
    detail::DualAveraging gentle(.1,.99,1),standard(.1,.99);
    require(std::abs(gentle.update(.99,false)-.1)<1e-14 && std::abs(standard.update(.99,false)-1)<1e-14,"DA restart center");
    Eigen::MatrixXd cov(2,2);cov<<1,.98,.98,1;const Eigen::MatrixXd precision=cov.inverse();
    require(std::abs(detail::local_stiffness(EuclideanMetric(precision),precision)-1)<1e-12,"Stiffness mass convention");
    require(std::abs(detail::local_stiffness(EuclideanMetric(precision/9),precision)-9)<1e-11,"Stiffness scaling");

    // Reject every proposed update; continuing DA and the RNG stream must agree
    // with a run that has no metric adaptation at all.
    NUTSOptions reject;reject.kernel=NUTSKernel::multinomial;reject.conservative_warmup=true;
    reject.find_initial_step_size=true;reject.step_size=1e-6;reject.target_accept=.99;
    reject.warmup=300;reject.draws=20;reject.max_depth=1;reject.metric_shrinkage=25;
    reject.metric_adaptation=MetricAdaptation::dense;reject.metric_max_stiffness_ratio=1;
    int guards=0,decisions=0;
    reject.local_curvature=[&](const Eigen::VectorXd&) {++guards;return eye;};
    reject.metric_progress=[&](const MetricDecision& d) {++decisions;require(!d.accepted,"Expected rejected update");};
    EuclideanMetric huge(1e8*eye);const auto rejected=sample_nuts(normal,zero,huge,reject);
    require(rejected.metric_updates.empty() && guards==3 && decisions==3 && rejected.step_size_searches.size()==1,"Rejected update restarted adaptation");
    auto fixed=reject;fixed.metric_adaptation=MetricAdaptation::none;fixed.metric_shrinkage=0;fixed.metric_max_stiffness_ratio=0;
    const auto unchanged=sample_nuts(normal,zero,huge,fixed);
    require(rejected.draws==unchanged.draws && rejected.step_size==unchanged.step_size && rejected.mass==unchanged.mass,"Rejected update changed sampling path");
    auto restart=reject;restart.conservative_warmup=false;restart.conservative_metric_restart=true;
    guards=decisions=0;const auto restart_rejected=sample_nuts(normal,zero,huge,restart);
    require(restart_rejected.metric_updates.empty() && guards==3 && decisions==3 && restart_rejected.step_size_searches.size()==1,"Rejected restart-only update changed adaptation");
    fixed=restart;fixed.conservative_metric_restart=false;fixed.metric_adaptation=MetricAdaptation::none;
    fixed.metric_shrinkage=0;fixed.metric_max_stiffness_ratio=0;
    const auto restart_fixed=sample_nuts(normal,zero,huge,fixed);
    require(restart_rejected.draws==restart_fixed.draws && restart_rejected.step_size==restart_fixed.step_size,"Restart-only rejection changed RNG or DA");
    // Ordinary startup is exactly preserved through transition 99, including
    // both tree kernels and configurations with/without initial step search.
    for(auto kernel:{NUTSKernel::slice,NUTSKernel::multinomial}) for(bool search_enabled:{false,true}) {
        NUTSOptions o;o.kernel=kernel;o.find_initial_step_size=search_enabled;o.warmup=300;o.draws=20;
        o.metric_adaptation=MetricAdaptation::dense;o.metric_shrinkage=25;
        std::vector<Eigen::VectorXd> before,after;
        o.progress=[&](int,const Eigen::VectorXd& q,const NUTSTransition&) {before.push_back(q);};
        const auto original=sample_nuts(normal,zero,identity,o);
        o.conservative_metric_restart=true;
        o.progress=[&](int,const Eigen::VectorXd& q,const NUTSTransition&) {after.push_back(q);};
        const auto changed=sample_nuts(normal,zero,identity,o);
        for(int i=0;i<100;++i) {
            const auto& a=original.transitions[i];const auto& b=changed.transitions[i];
            require(before[i]==after[i] && a.step_size==b.step_size && a.evaluations==b.evaluations
                && a.acceptance_probability==b.acceptance_probability && a.depth==b.depth
                && a.status==b.status && a.divergent==b.divergent,"Restart-only changed initial warmup");
        }
        require(original.metric_updates[0].mass==changed.metric_updates[0].mass,"First candidate metric changed");
    }
    // Actual distribution checks for separate and combined options.
    for(int policy=0;policy<5;++policy) for(int seed:{17,2026,99173}) {
        const LogDensity target=[&](const Eigen::VectorXd& q) {return DensityEvaluation{EvaluationStatus::valid,-.5*q.dot(precision*q),-precision*q};};
        NUTSOptions o;o.kernel=NUTSKernel::multinomial;o.find_initial_step_size=true;o.warmup=300;o.draws=12000;
        o.target_accept=.95;o.seed=seed;o.metric_adaptation=MetricAdaptation::dense;o.metric_shrinkage=25;
        o.conservative_warmup=policy==0 || policy==2;o.conservative_metric_restart=policy>=3;
        o.metric_max_stiffness_ratio=(policy==1 || policy==2 || policy==4)?4:0;
        int curvature_calls=0;o.local_curvature=[&](const Eigen::VectorXd&) {++curvature_calls;return precision;};
        const auto r=sample_nuts(target,zero,EuclideanMetric(precision),o);
        require(curvature_calls==(o.metric_max_stiffness_ratio>0?3:0),"Guard ran outside slow windows");
        const Eigen::MatrixXd white=cov.llt().matrixL().solve(r.draws.transpose()).transpose();
        for(int j=0;j<2;++j) {moment(white.col(j),0);moment(white.col(j).array().square().matrix(),1);}
        moment((white.col(0).array()*white.col(1).array()).matrix(),0);
        for(int i=o.warmup;i<o.warmup+o.draws;++i)
            require(r.transitions[i].step_size==r.step_size && !r.transitions[i].divergent && r.transitions[i].status==EvaluationStatus::valid,"Production adaptation or failure");
        o.draws=20;const auto prefix=sample_nuts(target,zero,EuclideanMetric(precision),o);
        require(prefix.draws==r.draws.topRows(20) && prefix.mass==r.mass,"Draw budget changed warmup");
    }
    auto invalid=reject;invalid.local_curvature=[](const Eigen::VectorXd&) {return Eigen::MatrixXd::Zero(2,2).eval();};
    bool failed=false;try {sample_nuts(normal,zero,huge,invalid);}catch(const std::invalid_argument&) {failed=true;}
    require(failed,"Invalid curvature silently ignored");
    for(double ratio:{-.1,.5,double(NAN),double(INFINITY)}) {
        invalid=reject;invalid.metric_max_stiffness_ratio=ratio;failed=false;
        try {sample_nuts(normal,zero,huge,invalid);}catch(const std::invalid_argument&) {failed=true;}
        require(failed,"Invalid guard threshold accepted");
    }
    invalid=reject;invalid.local_curvature={};failed=false;
    try {sample_nuts(normal,zero,huge,invalid);}catch(const std::invalid_argument&) {failed=true;}
    require(failed,"Missing curvature accepted");
    invalid=fixed;invalid.conservative_metric_restart=true;failed=false;
    try {sample_nuts(normal,zero,huge,invalid);}catch(const std::invalid_argument&) {failed=true;}
    require(failed,"Restart-only option accepted without metric adaptation");
    const LogDensity point=[](const Eigen::VectorXd& q) {return q.squaredNorm()>0?DensityEvaluation{}:DensityEvaluation{EvaluationStatus::valid,0,Eigen::VectorXd::Zero(q.size())};};
    failed=false;try {detail::find_conservative_step(point,zero,point(zero),identity,rng,.1,0);}catch(const std::runtime_error&) {failed=true;}
    require(failed,"Unbounded conservative search");
    std::cout<<"Conservative warmup and metric guard: law, costs, rejected updates, freezing and replay passed\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
