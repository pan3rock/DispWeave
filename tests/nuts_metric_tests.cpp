#include "dispweave/nuts.hpp"
#include "../src/sampler_adaptation.hpp"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <iostream>
#include <stdexcept>
using namespace dw;
void require(bool b,const char* message) {if(!b) throw std::runtime_error(message);}
void moment(const Eigen::VectorXd& x,double expected) {
    const int batches=60,width=x.size()/batches;Eigen::VectorXd means(batches);
    for(int i=0;i<batches;++i) means[i]=x.segment(i*width,width).mean();
    const double se=std::sqrt((means.array()-means.mean()).square().sum()/(batches*(batches-1)));
    require(se<.1 && std::abs(x.mean()-expected)<std::max(.04,6*se),"Whitened posterior moment");
}
int main() {try {
    // 25 draws cannot identify a dense covariance in 45 dimensions. Persistence
    // must retain known variance in directions the window has not explored.
    detail::CovarianceAccumulator accumulator(45);
    Eigen::MatrixXd previous=Eigen::VectorXd::LinSpaced(45,.1,2.).asDiagonal();
    for(int i=0;i<25;++i) {Eigen::VectorXd x=Eigen::VectorXd::Zero(45);x[0]=i-12;accumulator.add(x);}
    Eigen::MatrixXd sample=Eigen::MatrixXd::Zero(45,45);sample(0,0)=25.*26./12.;
    const Eigen::MatrixXd expected=(25.*sample+25.*previous+.005*Eigen::MatrixXd::Identity(45,45))/55.;
    const auto adapted=accumulator.covariance(MetricAdaptation::dense,previous,25);
    require(adapted.isApprox(expected,1e-13),"Covariance blend or pseudo-count convention");
    require(accumulator.mass(MetricAdaptation::dense,previous,25).isApprox(expected.inverse(),1e-12),"Blended mass must invert position covariance");
    require(accumulator.mass(MetricAdaptation::dense,previous,0)==accumulator.mass(MetricAdaptation::dense),"Zero shrinkage changed old results");
    previous(0,1)=previous(1,0)=.01;
    require(accumulator.covariance(MetricAdaptation::diagonal,previous,25).isApprox(expected,1e-13),"Diagonal adaptation retained cross terms");

    Eigen::MatrixXd cov(2,2);cov<<1,.98,.98,1;
    const Eigen::MatrixXd precision=cov.inverse();
    const LogDensity target=[&](const Eigen::VectorXd& q) {return DensityEvaluation{EvaluationStatus::valid,-.5*q.dot(precision*q),-precision*q};};
    for(auto seed:{17,2026,99173}) {
        NUTSOptions o;o.kernel=NUTSKernel::multinomial;o.find_initial_step_size=true;
        o.metric_adaptation=MetricAdaptation::dense;o.metric_shrinkage=25;
        o.warmup=300;o.draws=12000;o.target_accept=.95;o.seed=seed;
        Eigen::MatrixXd window(25,2);
        o.progress=[&](int i,const Eigen::VectorXd& q,const NUTSTransition&) {if(i>=75 && i<100) window.row(i-75)=q.transpose();};
        const auto r=sample_nuts(target,Eigen::VectorXd::Zero(2),EuclideanMetric(precision),o);
        const Eigen::MatrixXd centered=window.rowwise()-window.colwise().mean();
        const Eigen::MatrixXd first=(25.*centered.transpose()*centered/24.+25.*cov+.005*Eigen::Matrix2d::Identity())/55.;
        require(r.metric_updates.front().mass.isApprox(first.inverse(),1e-10),"First update failed to retain supplied metric");
        require(r.step_size_searches.size()==r.metric_updates.size()+1,"Step searches lost after blended updates");
        const Eigen::MatrixXd white=cov.llt().matrixL().solve(r.draws.transpose()).transpose();
        for(int j=0;j<2;++j) {moment(white.col(j),0);moment(white.col(j).array().square().matrix(),1);}
        moment((white.col(0).array()*white.col(1).array()).matrix(),0);
        for(int i=o.warmup;i<o.warmup+o.draws;++i)
            require(!r.transitions[i].divergent && r.transitions[i].status==EvaluationStatus::valid && r.transitions[i].step_size==r.step_size,"Invalid or adapting production");
        o.draws=20;o.progress={};
        const auto short_run=sample_nuts(target,Eigen::VectorXd::Zero(2),EuclideanMetric(precision),o);
        require(short_run.draws==r.draws.topRows(20) && short_run.mass==r.mass,"Shrinkage depends on production draw budget");
    }
    for(double k:{-1.,double(INFINITY),double(NAN),25.}) {
        NUTSOptions o;o.metric_shrinkage=k;bool rejected=false;
        try {sample_nuts(target,Eigen::VectorXd::Zero(2),EuclideanMetric(precision),o);}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"Invalid or unused shrinkage accepted");
    }
    std::cout<<"Metric persistence: rank-deficient window, mass convention, replay, and posterior law passed\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
