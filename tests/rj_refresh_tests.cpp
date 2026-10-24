#include "dispweave/rjmcmc.hpp"
#include <iostream>
#include <stdexcept>
using namespace dw;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
Eigen::VectorXd v(std::initializer_list<double> x){return Eigen::Map<const Eigen::VectorXd>(x.begin(),x.size());}
int main(){try {
 RJPrior p;p.min_layers=p.max_layers=2;p.layer_weights=v({1});p.depth_limit=5;p.min_spacing=.3;p.vs_lower=1;p.vs_upper=3;
 RJState a{v({1.5}),v({1.4,2.6})},b{v({1.5}),v({2.6,1.4})};
 auto proposal=rj_velocity_refresh(p,a,b.vs),reverse=rj_velocity_refresh(p,b,a.vs);
 check(proposal.status==RJProposalStatus::valid && proposal.state.interfaces==a.interfaces,"Refresh changed interfaces");
 check(proposal.log_forward==reverse.log_reverse && proposal.log_hastings()==0 && std::abs(proposal.log_forward+2*std::log(2.))<1e-14,"Refresh density mismatch");
 for(auto bad:{v({.9,2}),v({2,3.1}),v({NAN,2}),v({2})})check(rj_velocity_refresh(p,a,bad).status==RJProposalStatus::outside_support,"Refresh accepted invalid Vs");
 auto logmix=[](const RJState& s){
  double x=s.vs[0],y=s.vs[1];double l=std::log(.25)-.5*(std::pow((x-1.4)/.12,2)+std::pow((y-2.6)/.12,2));
  double r=std::log(.75)-.5*(std::pow((x-2.6)/.12,2)+std::pow((y-1.4)/.12,2));double m=std::max(l,r);return m+std::log(std::exp(l-m)+std::exp(r-m));
 };
 // Direct detailed balance flux for the nonuniform two-mode target.
 double la=logmix(a)+p.log_density(a),lb=logmix(b)+p.log_density(b);
 check(std::abs((la+proposal.log_forward+std::min(0.,lb-la))-(lb+reverse.log_forward+std::min(0.,la-lb)))<1e-14,"Refresh detailed balance");
 RJLikelihood f=[&](const RJState& s){return RJEvaluation{EvaluationStatus::valid,logmix(s)};};
 for(int seed:{17,2026,99173}){
  RJOptions o;o.warmup=5000;o.draws=400000;o.seed=seed;o.velocity_refresh_probability=.4;
  const auto run=sample_rjmcmc(f,p,a,o);int upper=0,refreshes=0,jumps=0;double mean=0;
  Eigen::VectorXd batches=Eigen::VectorXd::Zero(40);
  for(size_t i=0;i<run.draws.size();++i)batches[i/10000]+=(run.draws[i].state.vs[0]>run.draws[i].state.vs[1])/10000.;
  for(const auto& d:run.draws){upper+=d.state.vs[0]>d.state.vs[1];mean+=d.state.vs[0];}
  for(size_t i=1;i<run.draws.size();++i){const auto& tr=run.transitions[o.warmup+i];if(tr.move==RJMove::velocity_refresh){++refreshes;
   if(tr.accepted){const auto& x=run.draws[i-1].state;const auto& y=run.draws[i].state;check(x.interfaces==y.interfaces && x.vs[0]!=y.vs[0] && x.vs[1]!=y.vs[1],"Not a joint Vs update");jumps+=(x.vs[0]>x.vs[1])!=(y.vs[0]>y.vs[1]);}
  }}
  double se=std::sqrt((batches.array()-batches.mean()).square().sum()/(40*39.));
  std::cout<<"seed="<<seed<<" upper_mass="<<double(upper)/o.draws<<" batch_mcse="<<se<<" mean="<<mean/o.draws<<" jumps="<<jumps<<"\n";
  check(se<.03 && std::abs(double(upper)/o.draws-.75)<std::max(.005,6*se) && std::abs(mean/o.draws-2.3)<std::max(.01,7.2*se),"Bimodal mixture mass/mean mismatch");
  check(std::abs(refreshes-.1*o.draws)<6*std::sqrt(o.draws*.1*.9) && jumps>100,"Refresh frequency or mode transitions insufficient");
 }
 // Nonuniform K prior under a tempered likelihood must remain unheated.
 p.min_layers=1;p.max_layers=3;p.layer_weights=v({1,3,2});RJPTOptions pt;pt.local.warmup=5000;pt.local.draws=100000;pt.local.velocity_refresh_probability=.4;
 RJLikelihood tilt=[](const RJState& s){return RJEvaluation{EvaluationStatus::valid,-.7*s.vs.size()};};
 auto run=sample_rj_tempering(std::vector<RJLikelihood>(4,tilt),p,std::vector<RJState>(4,a),pt);
 for(size_t t=0;t<4;++t){Eigen::VectorXd truth(3),count=Eigen::VectorXd::Zero(3);for(int k=1;k<=3;++k)truth[k-1]=p.layer_weights[k-1]*std::exp(-.7*k/pt.temperatures[t]);truth/=truth.sum();
  for(const auto& d:run.temperatures[t].draws)++count[d.state.vs.size()-1];check((count/pt.local.draws-truth).cwiseAbs().maxCoeff()<.02,"Tempered K distribution mismatch");}
 for(double value:{-.1,1.1,double(NAN)}){RJOptions o;o.velocity_refresh_probability=value;bool threw=false;try{sample_rjmcmc(tilt,p,a,o);}catch(const std::invalid_argument&){threw=true;}check(threw,"Bad refresh probability accepted");}
 RJOptions o;o.draws=500;o.warmup=0;o.velocity_refresh_probability=1;
 RJLikelihood bad=[&](const RJState& s){return RJEvaluation{(s.vs.size()==a.vs.size() && s.vs==a.vs)?EvaluationStatus::valid:EvaluationStatus::forward_failure,0};};
 auto failed=sample_rjmcmc(bad,p,a,o);for(const auto& d:failed.draws)check(d.state.vs==a.vs,"Failed refreshed model accepted");
 std::cout<<"Joint refresh balance, bimodal distribution, variable K tempering and failures passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
