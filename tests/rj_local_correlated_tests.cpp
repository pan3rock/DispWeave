#include "dispweave/rjmcmc.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace dw;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Eigen::VectorXd v(std::initializer_list<double> x){return Eigen::Map<const Eigen::VectorXd>(x.begin(),x.size());}
RJGaussianMixture mixture(int k) {
    int n=2*k-1;RJGaussianComponent a,b;a.mean=Eigen::VectorXd::Constant(n,-100.);b.mean=Eigen::VectorXd::Constant(n,100.);
    a.covariance=Eigen::MatrixXd::Identity(n,n);b.covariance=2*a.covariance;
    if(n>1){a.covariance(0,n-1)=a.covariance(n-1,0)=.6;b.covariance(0,n-1)=b.covariance(n-1,0)=-.5;}
    a.weight=3;b.weight=1;return RJGaussianMixture(k,{a,b});
}
double mean_exp(double lo,double hi,double rate){return lo+(hi-lo)/(1-std::exp(-rate*(hi-lo)))-1/rate;}
int main(){try {
    RJPrior p;p.min_layers=p.max_layers=2;p.layer_weights=v({1});p.depth_limit=5;p.min_spacing=.3;p.vs_lower=1;p.vs_upper=3;
    RJState a{v({1}),v({1.4,2.6})},b{v({3}),v({2.4,1.3})};auto q=mixture(2);
    auto ab=q.propose_local(p,a,v({3,2.4,1.3})),ba=q.propose_local(p,b,v({1,1.4,2.6}));
    check(ab.status==RJProposalStatus::valid&&ba.status==RJProposalStatus::valid&&ab.log_hastings()==0&&ba.log_hastings()==0,"Local symmetry violated");
    std::mt19937_64 rng(123);int outside=0;Eigen::VectorXd displacement=Eigen::VectorXd::Zero(3);Eigen::MatrixXd second=Eigen::MatrixXd::Zero(3,3);
    for(int i=0;i<60000;++i){auto proposal=q.draw_local(p,a,rng,.3);outside+=proposal.status==RJProposalStatus::outside_support;
        auto x=v({proposal.state.interfaces[0]-1,proposal.state.vs[0]-1.4,proposal.state.vs[1]-2.6});displacement+=x;second+=x*x.transpose();}
    check(outside>1000,"Boundary draws were clipped/redrawn");
    check(displacement.cwiseAbs().maxCoeff()/60000<.006,"Local increments depend on component means");
    Eigen::MatrixXd expected=Eigen::MatrixXd::Identity(3,3)*1.25*.09;expected(0,2)=expected(2,0)=(.75*.6-.25*.5)*.09;
    check((second/60000-expected).cwiseAbs().maxCoeff()<.004,"Mixture increment covariance mismatch");
    bool threw=false;try{q.draw_local(p,a,rng,0);}catch(const std::invalid_argument&){threw=true;}check(threw,"Invalid scale accepted");
    RJLikelihood target=[](const RJState& s){return RJEvaluation{EvaluationStatus::valid,.5*s.interfaces[0]+.8*(s.vs[0]-1)-.4*(s.vs[1]-1)};};
    RJPTOptions o;o.local.warmup=5000;o.local.draws=150000;o.local.correlated_probability=.8;o.local.correlated_local_scale=.35;o.local.correlated_models={q};
    auto result=sample_rj_tempering(std::vector<RJLikelihood>(4,target),p,std::vector<RJState>(4,a),o);
    for(int t=0;t<4;++t){double temp=o.temperatures[t];auto truth=v({mean_exp(.3,4.7,.5/temp),mean_exp(1,3,.8/temp),mean_exp(1,3,-.4/temp)});
        Eigen::MatrixXd batches=Eigen::MatrixXd::Zero(50,3);
        for(size_t i=0;i<result.temperatures[t].draws.size();++i){auto s=result.temperatures[t].draws[i].state;batches.row(i/3000)+=v({s.interfaces[0],s.vs[0],s.vs[1]}).transpose()/3000.;}
        for(int j=0;j<3;++j){double mean=batches.col(j).mean(),se=std::sqrt((batches.col(j).array()-mean).square().sum()/(50*49.));check(se<.04&&std::abs(mean-truth[j])<std::max(.015,6*se),"Tempered analytic mean mismatch");}}
    p.min_layers=1;p.max_layers=3;p.layer_weights=v({1,3,2});o.local.correlated_models={mixture(1),mixture(2),mixture(3)};o.local.draws=100000;
    RJLikelihood tilt=[](const RJState& s){return RJEvaluation{EvaluationStatus::valid,-.7*s.vs.size()};};
    auto varying=sample_rj_tempering(std::vector<RJLikelihood>(4,tilt),p,std::vector<RJState>(4,a),o);
    for(int t=0;t<4;++t){auto count=Eigen::VectorXd::Zero(3).eval();Eigen::VectorXd truth(3);for(int k=1;k<=3;++k)truth[k-1]=p.layer_weights[k-1]*std::exp(-.7*k/o.temperatures[t]);truth/=truth.sum();for(const auto& d:varying.temperatures[t].draws)++count[d.state.vs.size()-1];check((count/o.local.draws-truth).cwiseAbs().maxCoeff()<.025,"Local move altered K prior");}
    for(double scale:{-1.,std::numeric_limits<double>::infinity()}){auto invalid=o.local;invalid.correlated_local_scale=scale;threw=false;try{sample_rjmcmc(tilt,p,a,invalid);}catch(const std::invalid_argument&){threw=true;}check(threw,"Invalid sampler scale accepted");}
    std::cout<<"Local correlated increments: symmetry, boundaries, covariance, tempered moments and K probabilities passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
