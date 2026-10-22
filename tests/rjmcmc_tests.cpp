#include "dispweave/rjmcmc.hpp"
#include <iostream>
#include <stdexcept>
#include <array>
using namespace dw;
Eigen::VectorXd v(std::initializer_list<double> x) {return Eigen::Map<const Eigen::VectorXd>(x.begin(),x.size());}
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
// 50 long batches, predefined before runs: max(.005,6 MCSE); MCSE < .025.
void moment(const Eigen::VectorXd& batch,double truth,const char* label) {
    const double se=std::sqrt((batch.array()-batch.mean()).square().sum()/(batch.size()*(batch.size()-1)));
    std::cout<<label<<" estimate="<<batch.mean()<<" truth="<<truth<<" mcse="<<se<<'\n';
    check(se<.025,"Insufficient Monte Carlo precision");
    check(std::abs(batch.mean()-truth)<std::max(.005,6*se),"Analytic posterior moment mismatch");
}
int main(int argc,char** argv) {try {
    const std::string name=argc>1?argv[1]:"";
    RJPrior prior;prior.max_layers=4;prior.layer_weights=v({1,2,1,1});prior.depth_limit=5;prior.min_spacing=.2;
    prior.vs_lower=1;prior.vs_upper=3;
    const RJState initial{v({}),v({2})};
    RJLikelihood flat=[](const RJState&) {return RJEvaluation{EvaluationStatus::valid,0};};
    if(name=="distribution") {
        for(bool tilted:{false,true}) for(auto seed:{17,2026,99173}) {
            const double lambda=tilted?.8:0,penalty=tilted?.6:0;
            RJLikelihood likelihood=[=](const RJState& x) {return RJEvaluation{EvaluationStatus::valid,-penalty*x.vs.size()+lambda*((x.vs.array()-1)/2).sum()};};
            // Independent midpoint integration on physical Vs standardized to [0,1].
            double norm=0,mean=0,second=0;
            for(int i=0;i<20000;++i) {double u=(i+.5)/20000.,w=std::exp(lambda*u)/20000.;norm+=w;mean+=w*u;second+=w*u*u;}
            mean/=norm;second/=norm;
            Eigen::VectorXd pk(4);
            for(int k=1;k<=4;++k) pk[k-1]=prior.layer_weights[k-1]*std::pow(norm*std::exp(-penalty),k);
            pk/=pk.sum();
            RJOptions o;o.warmup=5000;o.draws=100000;o.seed=seed;o.vs_step=.5;o.interface_step=1;
            const auto result=sample_rjmcmc(likelihood,prior,initial,o);
            std::cout<<"tilted="<<tilted<<" seed="<<seed<<'\n';
            int roundtrips=0,phase=0;std::array<int,4> actions{},accepted{};
            for(const auto& t:result.transitions) if(!t.warmup) {
                ++actions[static_cast<int>(t.move)];accepted[static_cast<int>(t.move)]+=t.accepted;
                if(t.layers_after==1) {if(phase==2) ++roundtrips;phase=1;}
                if(t.layers_after==4 && phase==1) phase=2;
                check(!t.evaluated || t.evaluation_status==EvaluationStatus::valid,"Analytic target evaluation failed");
            }
            check(roundtrips>100,"Insufficient dimension round trips");
            for(int move=0;move<4;++move) {
                check(std::abs(actions[move]-o.draws/4)<6*std::sqrt(o.draws*.25*.75),"Move probabilities changed at boundary");
                check(accepted[move]>100,"Move never mixes");
            }
            std::cout<<"roundtrips="<<roundtrips<<'\n';
            for(int k=1;k<=4;++k) {
                Eigen::VectorXd probability(50),first(50),square(50),interface(50),cross(50);
                for(int b=0;b<50;++b) {
                    int count=0;double s1=0,s2=0,z1=0,s12=0;
                    for(int j=0;j<2000;++j) {
                        const auto& draw=result.draws[b*2000+j];const auto& x=draw.state;
                        check(prior.supports(x),"Retained unsupported model");
                        if(x.vs.size()!=k) continue;
                        ++count;const Eigen::ArrayXd u=(x.vs.array()-1)/2;s1+=u.mean();s2+=u.square().mean();
                        if(k>1) {z1+=(x.interfaces[0]-.2)/(5-k*.2);s12+=u[0]*u[k-1];}
                    }
                    check(count>20,"Too few conditional samples in batch");
                    probability[b]=count/2000.;first[b]=s1/count;square[b]=s2/count;interface[b]=z1/count;cross[b]=s12/count;
                }
                std::cout<<"K="<<k<<'\n';
                moment(probability,pk[k-1],"P(K)");moment(first,mean,"E(U|K)");moment(square,second,"E(U^2|K)");
                if(k>1) {moment(interface,1./k,"E(first order statistic|K)");moment(cross,mean*mean,"E(U0*Ulast|K)");}
            }
        }
    } else if(name=="failures") {
        RJOptions o;o.warmup=20;o.draws=200;
        const auto a=sample_rjmcmc(flat,prior,initial,o),b=sample_rjmcmc(flat,prior,initial,o);
        for(int i=0;i<o.draws;++i) check(a.draws[i].state.vs==b.draws[i].state.vs && a.draws[i].state.interfaces==b.draws[i].state.interfaces,"Seed replay");
        o.draws=300;const auto longer=sample_rjmcmc(flat,prior,initial,o);
        for(int i=0;i<200;++i) check(a.draws[i].state.vs==longer.draws[i].state.vs && a.draws[i].state.interfaces==longer.draws[i].state.interfaces,"Draw prefix changed");
        o.chain_id=1;const auto other=sample_rjmcmc(flat,prior,initial,o);
        bool differs=false;for(int i=0;i<200;++i) if(a.draws[i].state.vs.size()!=other.draws[i].state.vs.size() || !a.draws[i].state.vs.isApprox(other.draws[i].state.vs)) differs=true;
        check(differs,"Chain streams identical");
        for(auto failure:{EvaluationStatus::forward_failure,EvaluationStatus::numerical_failure,EvaluationStatus::outside_support}) {
            RJLikelihood broken=[=](const RJState& x) {
                if(x.vs.size()==1 && x.vs[0]==2) return RJEvaluation{EvaluationStatus::valid,0};
                return failure==EvaluationStatus::numerical_failure?RJEvaluation{EvaluationStatus::valid,NAN}:RJEvaluation{failure,0};
            };
            const auto rejected=sample_rjmcmc(broken,prior,initial,o);int failed=0,unavailable=0;
            for(const auto& d:rejected.draws) check(d.state.vs==initial.vs && d.state.interfaces.size()==0,"Failed proposal changed state");
            for(const auto& t:rejected.transitions) {check(!t.accepted,"Invalid evaluation accepted");if(t.evaluated) {++failed;check(t.evaluation_status==failure,"Failure classification");}unavailable+=t.proposal_status==RJProposalStatus::unavailable;}
            check(failed>0 && unavailable>0,"Failure/unavailable paths untested");
        }
        int caught=0;
        auto gap=prior;gap.layer_weights[1]=0;
        try {sample_rjmcmc(flat,gap,initial,o);} catch(const std::invalid_argument&) {++caught;}
        auto bad=o;bad.vs_step=0;
        try {sample_rjmcmc(flat,prior,initial,bad);} catch(const std::invalid_argument&) {++caught;}
        try {sample_rjmcmc([](const RJState&) {return RJEvaluation{};},prior,initial,o);} catch(const std::invalid_argument&) {++caught;}
        check(caught==3,"Invalid sampler inputs accepted");
    } else throw std::invalid_argument("Unknown test case");
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
