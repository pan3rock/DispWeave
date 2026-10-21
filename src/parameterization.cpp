#include "dispweave/parameterization.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
FixedVs::FixedVs(LayeredModel m,Eigen::VectorXd lo,Eigen::VectorXd hi,double ratio)
    :reference_(std::move(m)),lower_(std::move(lo)),upper_(std::move(hi)),ratio_(ratio) {
    reference_.validate();
    if(lower_.size()!=reference_.vs.size() || upper_.size()!=lower_.size() ||
       !lower_.allFinite() || !upper_.allFinite() || (lower_.array()<=0).any() ||
       (upper_.array()<=lower_.array()).any() || !std::isfinite(ratio_) ||
       (ratio_!=0 && ratio_<=std::sqrt(4./3.))) throw std::invalid_argument("Invalid parameter bounds or Vp/Vs ratio");
    if(ratio_==0 && (upper_.array()*std::sqrt(4./3.)>=reference_.vp.array()).any())
        throw std::invalid_argument("Vs bounds violate fixed-Vp elastic stability");
    if(ratio_>0 && !reference_.vp.isApprox(ratio_*reference_.vs,1e-12))
        throw std::invalid_argument("Reference Vp inconsistent with ratio");
    lower_=lower_.array().log(); upper_=upper_.array().log();
    if(!contains(reference_.vs.array().log())) throw std::invalid_argument("Reference outside bounds");
}
bool FixedVs::contains(const Eigen::VectorXd& q) const {
    return q.size()==lower_.size() && q.allFinite() && (q.array()>=lower_.array()).all() && (q.array()<=upper_.array()).all();
}
LayeredModel FixedVs::decode(const Eigen::VectorXd& q) const {
    if(!contains(q)) throw std::invalid_argument("Parameters outside bounds");
    auto m=reference_; m.vs=q.array().exp(); if(ratio_>0) m.vp=ratio_*m.vs; m.validate(); return m;
}
Eigen::VectorXd FixedVs::encode(const LayeredModel& m) const {
    m.validate();
    if(m.vs.size()!=reference_.vs.size() || !m.depth.isApprox(reference_.depth,1e-12) ||
       !m.rho.isApprox(reference_.rho,1e-12) ||
       !(ratio_>0?m.vp.isApprox(ratio_*m.vs,1e-12):m.vp.isApprox(reference_.vp,1e-12)))
        throw std::invalid_argument("Model is not in this parameterization");
    Eigen::VectorXd q=m.vs.array().log(); if(!contains(q)) throw std::invalid_argument("Model outside bounds"); return q;
}
Eigen::MatrixXd FixedVs::phase_jacobian(const Eigen::VectorXd& q,const ForwardResult& f) const {
    const auto m=decode(q);
    if(!f.all_sensitivities_valid()) throw std::invalid_argument("Unavailable sensitivity");
    Eigen::MatrixXd j(f.rows.size(),q.size());
    for(size_t i=0;i<f.rows.size();++i) {
        const auto& s=f.rows[i].sensitivity;
        if(s.dc_dvs.size()!=q.size() || s.dc_dvp.size()!=q.size() || !s.dc_dvs.allFinite() || !s.dc_dvp.allFinite())
            throw std::invalid_argument("Malformed sensitivity");
        j.row(i)=((s.dc_dvs+ratio_*s.dc_dvp).array()*m.vs.array()).matrix().transpose();
    }
    return j;
}
double FixedVs::log_abs_det_jacobian(const Eigen::VectorXd& q) const {
    if(!contains(q)) throw std::invalid_argument("Parameters outside bounds");
    return q.sum();
}
Eigen::VectorXd FixedVs::log_abs_det_gradient(const Eigen::VectorXd& q) const {
    if(!contains(q)) throw std::invalid_argument("Parameters outside bounds");
    return Eigen::VectorXd::Ones(q.size());
}
}
