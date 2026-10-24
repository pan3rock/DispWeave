#include "dispweave/rjmcmc.hpp"
#include <iostream>
#include <stdexcept>
using namespace dw;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
Eigen::VectorXd v(std::initializer_list<double> x){return Eigen::Map<const Eigen::VectorXd>(x.begin(),x.size());}
int main(){try {
 RJPrior p;p.max_layers=3;p.layer_weights=v({1,3,2});p.depth_limit=5;p.min_spacing=.3;p.vs_lower=1;p.vs_upper=3;
 RJState start{v({}),v({2})};
 RJLikelihood f=[](const RJState& x){return RJEvaluation{EvaluationStatus::valid,-.7*x.vs.size()};};
 RJPTOptions o;o.local.warmup=5000;o.local.draws=100000;o.local.vs_step=.5;o.local.interface_step=.8;
 o.temperatures={1};const auto a=sample_rjmcmc(f,p,start,o.local),b=sample_rj_tempering({f},p,{start},o).temperatures[0];
 for(size_t i=0;i<a.draws.size();++i)check(a.draws[i].state.vs==b.draws[i].state.vs && a.draws[i].state.interfaces==b.draws[i].state.interfaces && a.draws[i].log_likelihood==b.draws[i].log_likelihood,"T=1 replay");
 o.temperatures={1,2,4,8};size_t calls=0;
 RJLikelihood counted=[&](const RJState& x){++calls;return f(x);};
 auto run=sample_rj_tempering(std::vector<RJLikelihood>(4,counted),p,std::vector<RJState>(4,start),o);
 size_t expected=4;
 for(size_t t=0;t<4;++t){
  Eigen::VectorXd truth(3),counts=Eigen::VectorXd::Zero(3);for(int k=1;k<=3;++k)truth[k-1]=p.layer_weights[k-1]*std::exp(-.7*k/o.temperatures[t]);truth/=truth.sum();
  for(const auto& d:run.temperatures[t].draws){++counts[d.state.vs.size()-1];check(d.log_likelihood==f(d.state).value && d.log_prior==p.log_density(d.state),"Swapped cache stale");}
  for(const auto& tr:run.temperatures[t].transitions)expected+=tr.evaluated;
  check((counts/o.local.draws-truth).cwiseAbs().maxCoeff()<.02,"Tempered analytic K distribution / unheated prior");
 }
 check(calls==expected,"Swaps reevaluate target");
 for(const auto& s:run.swaps){double before=s.log_like_lower/o.temperatures[s.lower]+s.log_like_upper/o.temperatures[s.upper];double after=s.log_like_upper/o.temperatures[s.lower]+s.log_like_lower/o.temperatures[s.upper];check(std::abs(after-before-s.log_accept_ratio)<1e-12,"Swap detailed balance");}
 // Continuous bimodal target: independently integrate each tempered marginal.
 RJPrior one=p;one.min_layers=one.max_layers=1;one.layer_weights=v({1});
 auto logmix=[](double x){double a=-.5*std::pow((x-1.4)/.08,2),b=std::log(3.)-.5*std::pow((x-2.6)/.08,2);double m=std::max(a,b);return m+std::log(std::exp(a-m)+std::exp(b-m));};
 RJLikelihood mixture=[&](const RJState& x){return RJEvaluation{EvaluationStatus::valid,logmix(x.vs[0])};};
 auto bimodal=sample_rj_tempering(std::vector<RJLikelihood>(4,mixture),one,std::vector<RJState>(4,start),o);
 for(size_t t=0;t<4;++t){double norm=0,upper=0;for(int j=0;j<20000;++j){double x=1+(j+.5)/10000.,w=std::exp(logmix(x)/o.temperatures[t]);norm+=w;if(x>2)upper+=w;}
  int count=0;for(const auto& d:bimodal.temperatures[t].draws){count+=d.state.vs[0]>2;check(d.log_likelihood==mixture(d.state).value,"Continuous cache stale");}
  check(std::abs(double(count)/o.local.draws-upper/norm)<.025,"Bimodal tempered mass mismatch");
 }
 o.local.warmup=20;o.local.draws=100;
 auto replay=sample_rj_tempering(std::vector<RJLikelihood>(4,f),p,std::vector<RJState>(4,start),o);
 for(size_t i=0;i<replay.replica_ids.size();++i)check(replay.replica_ids[i]==run.replica_ids[i],"Replica reproducibility");
 RJLikelihood bad=[](const RJState& x){return RJEvaluation{x.vs[0]==2?EvaluationStatus::valid:EvaluationStatus::forward_failure,0};};
 auto failed=sample_rj_tempering(std::vector<RJLikelihood>(4,bad),p,std::vector<RJState>(4,start),o);
 for(const auto& temp:failed.temperatures)for(const auto& d:temp.draws)check(d.state.vs[0]==2,"Heat rescued invalid model");
 for(auto ts:std::vector<std::vector<double>>{{},{2},{1,1},{1,NAN},{1,.5}}){o.temperatures=ts;bool threw=false;try{sample_rj_tempering(std::vector<RJLikelihood>(ts.size(),f),p,std::vector<RJState>(ts.size(),start),o);}catch(const std::invalid_argument&){threw=true;}check(threw,"Bad ladder accepted");}
 std::cout<<"PT analytic distributions, caches, balance, costs, replay and invalid states passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
