// Analytic geometry experiments, not geophysical observations.
#include "dispweave/nuts.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
using namespace dw;
LogDensity funnel(bool noncentered) {
    return [noncentered](const Eigen::VectorXd& q) {
        const double v=q[0],scale=noncentered?1.:std::exp(-v);
        const double sum=q.tail(q.size()-1).squaredNorm()*scale;
        Eigen::VectorXd g=-scale*q;g[0]=-v/9+(noncentered?0:.5*(sum-(q.size()-1)));
        return DensityEvaluation{EvaluationStatus::valid,-v*v/18-.5*sum-(noncentered?0:.5*(q.size()-1)*v),g};
    };
}
LogDensity mixture(double separation) {
    return [separation](const Eigen::VectorXd& q) {
        const double a=-.5*std::pow(q[0]-separation,2),b=-.5*std::pow(q[0]+separation,2),m=std::max(a,b);
        const double w=std::exp(a-m)/(std::exp(a-m)+std::exp(b-m));
        Eigen::VectorXd g=-q;g[0]=w*separation-(1-w)*separation-q[0];
        return DensityEvaluation{EvaluationStatus::valid,m+std::log(std::exp(a-m)+std::exp(b-m))-std::log(2.)-.5*q[1]*q[1],g};
    };
}
void check_gradient(const LogDensity& f,int n) {
    Eigen::VectorXd q=Eigen::VectorXd::LinSpaced(n,-.7,.9);const auto e=f(q);
    for(int j=0;j<n;++j) {
        auto plus=q,minus=q;plus[j]+=1e-5;minus[j]-=1e-5;
        if(std::abs((f(plus).value-f(minus).value)/2e-5-e.gradient[j])>1e-7)
            throw std::runtime_error("Analytic benchmark gradient failed finite difference check");
    }
}
int main(int argc,char** argv) {try {
    if(argc!=2) throw std::invalid_argument("Usage: nuts_geometry OUTPUT_DIRECTORY");
    const std::filesystem::path output=argv[1];
    if(std::filesystem::exists(output)) throw std::invalid_argument("Output already exists");
    for(int kind=0;kind<4;++kind) {
        const bool is_funnel=kind<2,noncentered=kind==1;const int n=is_funnel?10:2;
        const auto target=is_funnel?funnel(noncentered):mixture(kind==2?1.:6.);
        check_gradient(target,n);
        const auto dir=output/(kind==0?"funnel_centered":kind==1?"funnel_noncentered":kind==2?"mixture_overlap":"mixture_separated");
        std::filesystem::create_directories(dir);
        std::ofstream stats(dir/"chains.txt");stats<<std::setprecision(17)<<"# chain divergences invalid depth_limits evaluations seconds step_size\n";
        for(int chain=0;chain<4;++chain) {
            NUTSOptions options;options.warmup=1500;options.draws=6000;options.target_accept=.9;
            options.seed=2026;options.chain_id=chain;options.metric_adaptation=MetricAdaptation::diagonal;
            Eigen::VectorXd initial=Eigen::VectorXd::Zero(n);
            initial[0]=is_funnel?(chain-1.5):((chain%2?1.:-1.)*(kind==2?1.:6.));
            const auto begin=std::chrono::steady_clock::now();
            const auto result=sample_nuts(target,initial,EuclideanMetric(Eigen::MatrixXd::Identity(n,n)),options);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            std::ofstream samples(dir/("chain_"+std::to_string(chain)+".txt"));samples<<std::setprecision(17);
            samples<<"# "<<(is_funnel?"v x0 ... x8 (physical funnel coordinates)":"x y")<<"; retained draws\n";
            for(int i=0;i<options.draws;++i) {
                Eigen::VectorXd q=result.draws.row(i);
                if(noncentered) q.tail(n-1)*=std::exp(.5*q[0]);
                samples<<q.transpose()<<'\n';
            }
            int divergences=0,invalid=0,limits=0,evaluations=0;
            for(const auto& t:result.transitions) if(!t.warmup) {
                divergences+=t.divergent;invalid+=t.status!=EvaluationStatus::valid;limits+=t.max_depth_reached;evaluations+=t.evaluations;
            }
            stats<<chain<<' '<<divergences<<' '<<invalid<<' '<<limits<<' '<<evaluations<<' '<<seconds<<' '<<result.step_size<<'\n';
            std::cout<<dir.filename()<<" chain="<<chain<<" divergences="<<divergences<<" limits="<<limits<<std::endl;
        }
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
