#include "dispweave/rj_proposal.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace dw {
namespace {
void source_valid(const RJPrior& p,const RJState& s) {
    if(!p.supports(s)) throw std::invalid_argument("RJ proposal source outside prior support");
}
void probabilities_valid(double birth,double death) {
    if(!std::isfinite(birth) || !std::isfinite(death) || birth<=0 || death<=0 || birth+death>1)
        throw std::invalid_argument("Invalid RJ birth/death probabilities");
}
Eigen::VectorXd insert(const Eigen::VectorXd& x,Eigen::Index i,double value) {
    Eigen::VectorXd y(x.size()+1);y.head(i)=x.head(i);y[i]=value;y.tail(x.size()-i)=x.tail(x.size()-i);return y;
}
Eigen::VectorXd remove(const Eigen::VectorXd& x,Eigen::Index i) {
    Eigen::VectorXd y(x.size()-1);y.head(i)=x.head(i);y.tail(x.size()-i-1)=x.tail(x.size()-i-1);return y;
}
}
double rj_insertion_length(const RJPrior& p,const RJState& s) {
    source_valid(p,s);double sum=0,left=0;
    for(Eigen::Index i=0;i<s.vs.size();++i) {
        const double right=i<s.interfaces.size()?s.interfaces[i]:p.depth_limit;
        sum+=std::max(0.,right-left-2*p.min_spacing);left=right;
    }
    return sum;
}
double rj_insertion_position(const RJPrior& p,const RJState& s,double u) {
    if(!std::isfinite(u) || u<0 || u>=1) throw std::invalid_argument("Insertion coordinate must be in [0,1)");
    const double length=rj_insertion_length(p,s);
    if(length<=0) throw std::invalid_argument("No admissible insertion interval");
    double offset=u*length,left=0;
    for(Eigen::Index i=0;i<s.vs.size();++i) {
        const double right=i<s.interfaces.size()?s.interfaces[i]:p.depth_limit;
        const double width=std::max(0.,right-left-2*p.min_spacing);
        if(offset<width) return left+p.min_spacing+offset;
        offset-=width;left=right;
    }
    // Floating-point endpoint saturation is an invalid proposal, not a clipped one.
    return NAN;
}
RJProposal rj_birth(const RJPrior& p,const RJState& s,double depth,double new_vs,double pb,double pd) {
    source_valid(p,s);probabilities_valid(pb,pd);
    RJProposal out{s,RJMove::birth};
    const int k=static_cast<int>(s.vs.size());
    if(k>=p.max_layers) return out;
    const double length=rj_insertion_length(p,s);
    if(length<=0) return out;
    out.status=RJProposalStatus::outside_support;
    if(!std::isfinite(depth) || !std::isfinite(new_vs) || new_vs<p.vs_lower || new_vs>p.vs_upper) return out;
    int i=0;while(i<k-1 && s.interfaces[i]<depth) ++i;
    const double left=i==0?0:s.interfaces[i-1],right=i==k-1?p.depth_limit:s.interfaces[i];
    if(right-left<=2*p.min_spacing || depth-left<p.min_spacing || right-depth<p.min_spacing) return out;
    // x plus auxiliaries (depth,new_vs) -> permutation of the enlarged state: |J|=1.
    out.state={insert(s.interfaces,i,depth),insert(s.vs,i+1,new_vs)};out.index=i;
    out.log_forward=std::log(pb)-std::log(length)-std::log(p.vs_upper-p.vs_lower);
    out.log_reverse=std::log(pd)-std::log(double(k));
    if(p.supports(out.state)) out.status=RJProposalStatus::valid;
    return out;
}
RJProposal rj_death(const RJPrior& p,const RJState& s,int i,double pb,double pd) {
    source_valid(p,s);probabilities_valid(pb,pd);
    RJProposal out{s,RJMove::death};
    const int k=static_cast<int>(s.vs.size());
    if(k<=p.min_layers) return out;
    if(i<0 || i>=k-1) throw std::invalid_argument("Invalid death interface index");
    const double left=i==0?0:s.interfaces[i-1],right=i==k-2?p.depth_limit:s.interfaces[i+1];
    if(right-left<=2*p.min_spacing) {out.status=RJProposalStatus::outside_support;return out;}
    out.state={remove(s.interfaces,i),remove(s.vs,i+1)};out.index=i;
    out.status=RJProposalStatus::outside_support;
    if(!p.supports(out.state)) return out;
    const double length=rj_insertion_length(p,out.state);
    // Merging two minimum-width layers may have zero insertion measure at an exact boundary.
    if(length<=0) return out;
    out.log_forward=std::log(pd)-std::log(double(k-1));
    out.log_reverse=std::log(pb)-std::log(length)-std::log(p.vs_upper-p.vs_lower);
    out.status=RJProposalStatus::valid;return out;
}
}
