#include "dispweave/rjmcmc.hpp"
#include <iostream>
#include <stdexcept>
using namespace dw;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
Eigen::VectorXd v(std::initializer_list<double> x){return Eigen::Map<const Eigen::VectorXd>(x.begin(),x.size());}
RJGaussianMixture mixture(int k) {
 int n=2*k-1;RJGaussianComponent a,b;a.mean=Eigen::VectorXd::Constant(n,1.5);b.mean=Eigen::VectorXd::Constant(n,2.5);
 for(int i=0;i<k-1;++i){a.mean[i]=5.*(i+1)/k-.3;b.mean[i]=5.*(i+1)/k+.3;}
 a.covariance=Eigen::MatrixXd::Identity(n,n);b.covariance=a.covariance;
 if(n>1){a.covariance(0,n-1)=a.covariance(n-1,0)=.55;b.covariance(0,n-1)=b.covariance(n-1,0)=-.4;}
 a.weight=3;b.weight=1;return RJGaussianMixture(k,{a,b});
}
double mean_exp(double lo,double hi,double rate){double width=hi-lo;return lo+width/(1-std::exp(-rate*width))-1/rate;}
int main(){try {
 RJPrior p;p.min_layers=p.max_layers=2;p.layer_weights=v({1});p.depth_limit=5;p.min_spacing=.3;p.vs_lower=1;p.vs_upper=3;
 RJState a{v({1}),v({1.4,2.6})},b{v({3}),v({2.4,1.3})};auto q=mixture(2);
 auto ab=q.propose(p,a,v({3,2.4,1.3})),ba=q.propose(p,b,v({1,1.4,2.6}));
 check(ab.status==RJProposalStatus::valid && ba.status==RJProposalStatus::valid,"Valid correlated proposal rejected");
 check(std::abs(ab.log_hastings())>.1 && ab.log_forward==ba.log_reverse && ab.log_reverse==ba.log_forward,"Mixture Hastings omitted");
 double la=.7,lb=-.4;
 check(std::abs(la+ab.log_forward+std::min(0.,lb-la+ab.log_hastings())-(lb+ba.log_forward+std::min(0.,la-lb+ba.log_hastings())))<1e-12,"Correlated detailed balance");
 RJGaussianComponent c;c.mean=v({2});c.covariance=Eigen::MatrixXd::Constant(1,1,.25);RJGaussianComponent d=c;d.mean=v({1});d.weight=3;
 RJGaussianMixture one(1,{c,d});RJState s{v({}),v({2})};
 double normalizer=1/std::sqrt(2*std::acos(-1.)*.25);double expected=std::log(.25*normalizer+.75*normalizer*std::exp(-2));
 check(std::abs(one.log_density(s)-expected)<1e-14,"Mixture normalization/logsumexp");
 std::mt19937_64 rng(17);int outside=0;for(int i=0;i<1000;++i)outside+=q.draw(p,a,rng).status==RJProposalStatus::outside_support;
 check(outside>100,"Proposal silently truncated or redrawn");
 auto bad=c;bad.covariance(0,0)=0;bool threw=false;try{RJGaussianMixture unused(1,{bad});}catch(const std::invalid_argument&){threw=true;}check(threw,"Singular covariance accepted");
 RJLikelihood target=[](const RJState& s){return RJEvaluation{EvaluationStatus::valid,.5*s.interfaces[0]+.8*(s.vs[0]-1)-.4*(s.vs[1]-1)};};
 RJPTOptions options;options.local.warmup=5000;options.local.draws=150000;options.local.correlated_probability=.8;options.local.correlated_models={q};
 auto result=sample_rj_tempering(std::vector<RJLikelihood>(4,target),p,std::vector<RJState>(4,a),options);
 for(int t=0;t<4;++t){double temperature=options.temperatures[t];Eigen::VectorXd truth=v({mean_exp(.3,4.7,.5/temperature),mean_exp(1,3,.8/temperature),mean_exp(1,3,-.4/temperature)});
  Eigen::MatrixXd batches=Eigen::MatrixXd::Zero(50,3);
  for(size_t i=0;i<result.temperatures[t].draws.size();++i){const auto& state=result.temperatures[t].draws[i].state;batches.row(i/3000)+=v({state.interfaces[0],state.vs[0],state.vs[1]}).transpose()/3000.;}
  for(int j=0;j<3;++j){double mean=batches.col(j).mean(),se=std::sqrt((batches.col(j).array()-mean).square().sum()/(50*49.));check(se<.03 && std::abs(mean-truth[j])<std::max(.01,6*se),"Tempered analytic mean / proposal correction mismatch");}
 }
 p.min_layers=1;p.max_layers=3;p.layer_weights=v({1,3,2});options.local.correlated_models={mixture(1),mixture(2),mixture(3)};options.local.draws=100000;
 RJLikelihood tilt=[](const RJState& s){return RJEvaluation{EvaluationStatus::valid,-.7*s.vs.size()};};
 auto variable=sample_rj_tempering(std::vector<RJLikelihood>(4,tilt),p,std::vector<RJState>(4,a),options);
 for(int t=0;t<4;++t){Eigen::VectorXd truth(3),count=Eigen::VectorXd::Zero(3);for(int k=1;k<=3;++k)truth[k-1]=p.layer_weights[k-1]*std::exp(-.7*k/options.temperatures[t]);truth/=truth.sum();for(const auto& draw:variable.temperatures[t].draws)++count[draw.state.vs.size()-1];check((count/options.local.draws-truth).cwiseAbs().maxCoeff()<.025,"Correlated proposal changed layer prior");}
 auto invalid=options.local;invalid.correlated_models.pop_back();threw=false;try{sample_rjmcmc(tilt,p,a,invalid);}catch(const std::invalid_argument&){threw=true;}check(threw,"Missing K density accepted");
 invalid=options.local;invalid.velocity_refresh_probability=.5;threw=false;try{sample_rjmcmc(tilt,p,a,invalid);}catch(const std::invalid_argument&){threw=true;}check(threw,"Invalid mixture weights accepted");
 std::cout<<"Correlated Gaussian mixture: density, balance, full-temperature moments and K probabilities passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
