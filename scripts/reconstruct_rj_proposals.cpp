// Offline reconstruction of an existing run's proposals, never calls a target.
// RNG consumption mirrors src/rjmcmc.cpp; every move/status/decision is checked.
#include "dispweave/rj_correlated.hpp"
#include <toml.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
using Row=std::vector<std::string>;
using Table=std::vector<Row>;
namespace fs=std::filesystem;
void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
Table table(const fs::path& path) {
    std::ifstream f(path);require(bool(f),"Cannot read "+path.string());Table result;std::string line;
    while(std::getline(f,line)) {if(line.empty()||line[0]=='#')continue;std::istringstream s(line);Row row;std::string x;while(s>>x)row.push_back(x);result.push_back(row);}
    return result;
}
dw::RJState unpack(const Row& row,int k_column,int coordinates) {
    int k=std::stoi(row[k_column]);dw::RJState state{Eigen::VectorXd(k-1),Eigen::VectorXd(k)};
    require(int(row.size())==coordinates+2*k-1,"Malformed state");
    for(int j=0;j<k-1;++j)state.interfaces[j]=std::stod(row[coordinates+j]);
    for(int j=0;j<k;++j)state.vs[j]=std::stod(row[coordinates+k-1+j]);
    return state;
}
double difference(const dw::RJState& a,const dw::RJState& b) {
    require(a.vs.size()==b.vs.size(),"Dimension mismatch");double result=(a.vs-b.vs).cwiseAbs().maxCoeff();
    if(a.interfaces.size())result=std::max(result,(a.interfaces-b.interfaces).cwiseAbs().maxCoeff());return result;
}
int main(int argc,char** argv) try {
    require(argc==3,"Usage: reconstruct_rj_proposals RUN_DIRECTORY OUTPUT_TSV");fs::path source=argv[1],output=argv[2];
    require(!fs::exists(output),"Output exists");auto config=toml::parse((source/"rj.toml").string());
    const auto& prior_cfg=toml::find(config,"rj_prior");const auto& opts=toml::find(config,"rjmcmc");
    dw::RJPrior prior;prior.min_layers=toml::find<int>(prior_cfg,"min_layers");prior.max_layers=toml::find<int>(prior_cfg,"max_layers");
    prior.depth_limit=toml::find<double>(prior_cfg,"depth_limit");prior.min_spacing=toml::find<double>(prior_cfg,"min_spacing");
    prior.vs_lower=toml::find<double>(prior_cfg,"vs_lower");prior.vs_upper=toml::find<double>(prior_cfg,"vs_upper");
    auto weights=toml::find<std::vector<double>>(prior_cfg,"layer_weights");prior.layer_weights=Eigen::Map<Eigen::VectorXd>(weights.data(),weights.size());prior.validate();
    const auto warm=toml::find<int>(opts,"warmup"),draws=toml::find<int>(opts,"draws");
    const auto seed=toml::find<std::uint64_t>(opts,"seed");const auto& pt=toml::find(opts,"tempering");
    auto temperatures=toml::find<std::vector<double>>(pt,"temperatures");
    require(toml::find_or<int>(pt,"cold_start_steps",0)==0,"Cold start not supported by reconstruction");
    require(toml::find_or<double>(opts,"velocity_refresh_probability",0.)==0,"Refresh not supported");
    double cp=toml::find_or<double>(opts,"correlated_probability",0.),vs_step=toml::find<double>(opts,"vs_step"),z_step=toml::find<double>(opts,"interface_step");
    double local_scale=toml::find_or<double>(opts,"correlated_local_scale",0.);
    std::map<int,dw::RJGaussianMixture> mixtures;
    if(config.contains("rj_correlated_models"))for(const auto& m:toml::find(config,"rj_correlated_models").as_array()) {
        int k=toml::find<int>(m,"layers");std::vector<dw::RJGaussianComponent> parts;
        for(const auto& part:toml::find(m,"components").as_array()) {
            auto mean=toml::find<std::vector<double>>(part,"mean");auto covariance=toml::find<std::vector<std::vector<double>>>(part,"covariance");
            dw::RJGaussianComponent p;p.mean=Eigen::Map<Eigen::VectorXd>(mean.data(),mean.size());p.covariance.resize(mean.size(),mean.size());
            for(size_t i=0;i<mean.size();++i)for(size_t j=0;j<mean.size();++j)p.covariance(i,j)=covariance.at(i).at(j);
            p.weight=toml::find<double>(part,"weight");parts.push_back(p);
        }
        mixtures.emplace(k,dw::RJGaussianMixture(k,parts));
    }
    std::ofstream out(output);out<<std::setprecision(17)<<"# chain temperature iteration warmup move proposal_status evaluation_status accepted K log_ratio delta_logL_over_T log_prior_ratio log_q_ratio logq_old logq_new\n";
    const char* move_names[]={"birth","death","interface","velocity","velocity_refresh","correlated"};
    const char* status_names[]={"valid","unavailable","outside_support"};
    size_t checks=0,accepted_checks=0,failed_checks=0;double max_state_error=0,max_ll_error=0;
    int ladders=int(toml::find(config,"rj_starts").as_array().size()),nt=int(temperatures.size());
    for(int c=0;c<ladders;++c) {
        auto folder=source/"rj"/("ladder_"+std::to_string(c));std::vector<Table> state(nt);
        for(int t=0;t<nt;++t) {auto prefix="temperature_"+std::to_string(t);state[t]=table(folder/(prefix+"_warmup_models.txt"));auto formal=table(folder/(prefix+"_models.txt"));state[t].insert(state[t].end(),formal.begin(),formal.end());require(int(state[t].size())==warm+draws+1,"Wrong state count");}
        std::vector<std::vector<int>> destination(warm+draws,std::vector<int>(nt));
        for(auto& v:destination)for(int t=0;t<nt;++t)v[t]=t;
        for(const auto& s:table(folder/"swaps.txt"))if(s[9]=="1")std::swap(destination.at(std::stoi(s[0])).at(std::stoi(s[2])),destination.at(std::stoi(s[0])).at(std::stoi(s[3])));
        for(int t=0;t<nt;++t) {
            auto prefix="temperature_"+std::to_string(t);auto trace=table(folder/(prefix+"_local_trace.txt"));require(int(trace.size())==warm+draws,"Wrong trace count");
            std::map<int,dw::RJState> failed;for(const auto& r:table(folder/(prefix+"_failed_models.txt")))failed.emplace(std::stoi(r[0]),unpack(r,2,3));
            std::uint64_t id=c*nt+t;std::seed_seq seeds{std::uint32_t(seed),std::uint32_t(seed>>32),std::uint32_t(id),std::uint32_t(id>>32)};
            std::mt19937_64 rng(seeds);std::normal_distribution<double> normal;
            auto uniform=[&](){return std::generate_canonical<double,53>(rng);};auto index=[&](int n){return std::uniform_int_distribution<int>(0,n-1)(rng);};
            for(int i=0;i<warm+draws;++i) {
                const auto& tr=trace[i];const auto& before_row=state[t][i];auto before=unpack(before_row,1,4);int k=int(before.vs.size());
                auto move=static_cast<dw::RJMove>(index(4));if(move==dw::RJMove::velocity&&cp>0&&uniform()<cp)move=dw::RJMove::correlated;
                require(tr[1]==move_names[int(move)],"Move RNG mismatch");dw::RJProposal p{before,move};
                if(move==dw::RJMove::birth) {if(k<prior.max_layers&&dw::rj_insertion_length(prior,before)>0) {double z=dw::rj_insertion_position(prior,before,uniform());double v=prior.vs_lower+(prior.vs_upper-prior.vs_lower)*uniform();p=dw::rj_birth(prior,before,z,v);}}
                else if(move==dw::RJMove::death) {if(k>prior.min_layers)p=dw::rj_death(prior,before,index(k-1));}
                else if(move==dw::RJMove::interface) {if(k>1) {int j=index(k-1);p.state.interfaces[j]+=z_step*normal(rng);p.status=prior.supports(p.state)?dw::RJProposalStatus::valid:dw::RJProposalStatus::outside_support;}}
                else if(move==dw::RJMove::correlated)p=local_scale>0?mixtures.at(k).draw_local(prior,before,rng,local_scale):mixtures.at(k).draw(prior,before,rng);
                else {int j=index(k);p.state.vs[j]+=vs_step*normal(rng);p.status=prior.supports(p.state)?dw::RJProposalStatus::valid:dw::RJProposalStatus::outside_support;}
                require(tr[2]==status_names[int(p.status)],"Proposal status mismatch");require((tr[3]=="1")== (p.status==dw::RJProposalStatus::valid),"Evaluation mismatch");
                double ratio=std::stod(tr[8]),lp=NAN,h=NAN,dll=NAN,qo=NAN,qn=NAN;bool accepted=false;
                if(p.status==dw::RJProposalStatus::valid) {
                    lp=prior.log_density(p.state)-std::stod(before_row[3]);h=p.log_hastings();
                    if(move==dw::RJMove::correlated) {qo=p.log_reverse;qn=p.log_forward;}
                    if(tr[4]=="valid") {require(std::isfinite(ratio),"Nonfinite ratio");dll=ratio-lp-h;accepted=std::log1p(-uniform())<std::min(0.,ratio);}
                    else {auto it=failed.find(std::stoi(tr[9]));require(it!=failed.end(),"Missing failed model");double error=difference(p.state,it->second);max_state_error=std::max(max_state_error,error);require(error<1e-12,"Failed candidate mismatch");++failed_checks;}
                }
                require(accepted==(tr[5]=="1"),"Acceptance RNG mismatch");const auto& after_row=state[destination[i][t]][i+1];auto after=unpack(after_row,1,4);
                double error=difference(accepted?p.state:before,after);max_state_error=std::max(max_state_error,error);require(error<1e-12,"Local state mismatch");
                if(accepted) {double e=std::abs(dll-(std::stod(after_row[2])-std::stod(before_row[2]))/temperatures[t]);max_ll_error=std::max(max_ll_error,e);require(e<1e-8,"Accepted likelihood decomposition mismatch");++accepted_checks;}
                ++checks;
                if(move==dw::RJMove::correlated||move==dw::RJMove::birth||move==dw::RJMove::death)
                    out<<c<<' '<<temperatures[t]<<' '<<i<<' '<<(i<warm)<<' '<<tr[1]<<' '<<tr[2]<<' '<<tr[4]<<' '<<accepted<<' '<<k<<' '<<ratio<<' '<<dll<<' '<<lp<<' '<<h<<' '<<qo<<' '<<qn<<'\n';
            }
        }
    }
    std::cout<<std::setprecision(17)<<"{\"transitions_verified\":"<<checks<<",\"accepted_states_verified\":"<<accepted_checks<<",\"failed_candidates_verified\":"<<failed_checks<<",\"max_state_error\":"<<max_state_error<<",\"max_accepted_likelihood_error\":"<<max_ll_error<<",\"forward_calls\":0,\"new_samples\":0}\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
