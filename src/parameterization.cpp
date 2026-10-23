#include "dispweave/parameterization.hpp"
#include "model_builder.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
VsBounds reference_vs_bounds(const Eigen::VectorXd& reference,double vs_min,double vs_max,double vs_width) {
    if(!std::isfinite(vs_min) || !std::isfinite(vs_max) || !std::isfinite(vs_width) ||
       vs_min<=0 || vs_min>=vs_max || vs_width<=0)
        throw std::invalid_argument("Require finite 0 < vs_min < vs_max and vs_width > 0 (full width, km/s)");
    if(reference.size()==0 || !reference.allFinite() || (reference.array()<=0).any())
        throw std::invalid_argument("Reference Vs must be nonempty, finite and positive");
    VsBounds bounds{reference,reference};
    for(Eigen::Index i=0;i<reference.size();++i) {
        bounds.lower[i]=std::max(vs_min,reference[i]-.5*vs_width);
        bounds.upper[i]=std::min(vs_max,reference[i]+.5*vs_width);
        if(bounds.lower[i]>=bounds.upper[i])
            throw std::invalid_argument("Empty or zero-width Vs intersection at layer "+std::to_string(i));
        if(reference[i]<bounds.lower[i] || reference[i]>bounds.upper[i])
            throw std::invalid_argument("Reference Vs outside global bounds at layer "+std::to_string(i));
    }
    return bounds;
}
FixedVs::FixedVs(LayeredModel m,Eigen::VectorXd lo,Eigen::VectorXd hi,double ratio)
    :FixedVs(std::move(m),std::move(lo),std::move(hi),ElasticRelation{ratio==0?"fixvprho":"constant_ratio",ratio}) {}
FixedVs::FixedVs(LayeredModel m,Eigen::VectorXd lo,Eigen::VectorXd hi,ElasticRelation relation)
    :reference_(std::move(m)),lower_(std::move(lo)),upper_(std::move(hi)),
     ratio_(relation.kind=="constant_ratio"?relation.vp_vs_ratio:0),relation_(std::move(relation)) {
    reference_.validate();
    const auto& r=relation_;
    if(r.kind!="fixvprho" && r.kind!="brocher05" &&
       (!std::isfinite(r.vp_vs_ratio) || r.vp_vs_ratio<=std::sqrt(4./3.)))
        throw std::invalid_argument("Empirical Vp/Vs must exceed sqrt(4/3)");
    if(r.kind=="brocher05") builder_=std::make_shared<fwd::Brocher05>();
    else if(r.kind=="gardner") {
        if(!std::isfinite(r.gardner_alpha) || r.gardner_alpha<=0 || !std::isfinite(r.gardner_beta) || r.gardner_beta<=0)
            throw std::invalid_argument("Positive finite Gardner coefficients required");
        builder_=std::make_shared<fwd::Gardner>(r.vp_vs_ratio,r.gardner_alpha,r.gardner_beta);
    } else if(r.kind=="nearsurface") builder_=std::make_shared<fwd::NearSurface>(r.vp_vs_ratio);
    else if(r.kind=="constant_rho") builder_=std::make_shared<fwd::ConstantRho>(r.vp_vs_ratio,r.density);
    else if(r.kind!="fixvprho" && r.kind!="constant_ratio") throw std::invalid_argument("Unknown elastic relation");
    if(builder_) {
        const auto values=builder_->generate(reference_.vs.array());
        reference_.vp=values.col(0);reference_.rho=values.col(1);reference_.validate();
    }
    if(lower_.size()!=reference_.vs.size() || upper_.size()!=lower_.size() ||
       !lower_.allFinite() || !upper_.allFinite() || (lower_.array()<=0).any() ||
       (upper_.array()<=lower_.array()).any() || !std::isfinite(ratio_) ||
       (ratio_!=0 && ratio_<=std::sqrt(4./3.))) throw std::invalid_argument("Invalid parameter bounds or Vp/Vs ratio");
    if(!builder_ && ratio_==0)
        for(Eigen::Index i=0;i<upper_.size();++i)
            if(upper_[i]*std::sqrt(4./3.)>=reference_.vp[i])
                throw std::invalid_argument("Vs bounds violate fixed-Vp elastic stability at layer "+std::to_string(i));
    if(ratio_>0 && !reference_.vp.isApprox(ratio_*reference_.vs,1e-12))
        throw std::invalid_argument("Reference Vp inconsistent with ratio");
    lower_=lower_.array().log(); upper_=upper_.array().log();
    if(!contains(reference_.vs.array().log())) throw std::invalid_argument("Reference outside bounds");
}
bool FixedVs::contains(const Eigen::VectorXd& q) const {
    if(q.size()!=lower_.size() || !q.allFinite() || (q.array()<lower_.array()).any() || (q.array()>upper_.array()).any()) return false;
    if(builder_) {
        auto m=reference_;m.vs=q.array().exp();const auto values=builder_->generate(m.vs.array());
        m.vp=values.col(0);m.rho=values.col(1);
        try {m.validate();} catch(const std::invalid_argument&) {return false;}
    }
    return true;
}
LayeredModel FixedVs::decode(const Eigen::VectorXd& q) const {
    if(!contains(q)) throw std::invalid_argument("Parameters outside bounds");
    auto m=reference_; m.vs=q.array().exp(); if(ratio_>0) m.vp=ratio_*m.vs;
    if(builder_) {const auto values=builder_->generate(m.vs.array());m.vp=values.col(0);m.rho=values.col(1);}
    m.validate(); return m;
}
Eigen::VectorXd FixedVs::encode(const LayeredModel& m) const {
    m.validate();
    if(m.vs.size()!=reference_.vs.size() || !m.depth.isApprox(reference_.depth,1e-12))
        throw std::invalid_argument("Model is not in this parameterization");
    Eigen::VectorXd q=m.vs.array().log(); if(!contains(q)) throw std::invalid_argument("Model outside bounds");
    const auto expected=decode(q);
    if(!m.rho.isApprox(expected.rho,1e-12) || !m.vp.isApprox(expected.vp,1e-12))
        throw std::invalid_argument("Model properties disagree with elastic relation");
    return q;
}
Eigen::MatrixXd FixedVs::phase_jacobian(const Eigen::VectorXd& q,const ForwardResult& f) const {
    if(!f.all_sensitivities_valid()) throw std::invalid_argument("Unavailable sensitivity");
    Eigen::MatrixXd d_vs(f.rows.size(),q.size()),d_vp(f.rows.size(),q.size()),d_rho(f.rows.size(),q.size());
    for(size_t i=0;i<f.rows.size();++i) {
        const auto& s=f.rows[i].sensitivity;
        if(s.dc_dvs.size()!=q.size() || s.dc_dvp.size()!=q.size() || s.dc_drho.size()!=q.size() ||
           !s.dc_dvs.allFinite() || !s.dc_dvp.allFinite() || !s.dc_drho.allFinite())
            throw std::invalid_argument("Malformed sensitivity");
        d_vs.row(i)=s.dc_dvs.transpose();d_vp.row(i)=s.dc_dvp.transpose();d_rho.row(i)=s.dc_drho.transpose();
    }
    return model_jacobian(q,d_vs,d_vp,d_rho);
}
double FixedVs::log_abs_det_jacobian(const Eigen::VectorXd& q) const {
    if(!contains(q)) throw std::invalid_argument("Parameters outside bounds");
    return q.sum();
}
Eigen::MatrixXd FixedVs::model_jacobian(const Eigen::VectorXd& q,const Eigen::MatrixXd& d_vs,
                                       const Eigen::MatrixXd& d_vp,const Eigen::MatrixXd& d_rho) const {
    const auto m=decode(q);
    if(d_vs.cols()!=q.size() || d_vp.cols()!=q.size() || d_vs.rows()!=d_vp.rows() ||
       !d_vs.allFinite() || !d_vp.allFinite()) throw std::invalid_argument("Malformed model sensitivity");
    if(!builder_) return (d_vs+ratio_*d_vp)*m.vs.asDiagonal();
    const auto derivatives=builder_->derivative(m.vs.array());
    Eigen::MatrixXd j=d_vs+d_vp*derivatives.dvp_dvs.matrix().asDiagonal();
    if(d_rho.size()==0) {
        if((derivatives.drho_dvs!=0).any()) throw std::invalid_argument("Density derivative required by empirical relation");
    } else {
        if(d_rho.rows()!=d_vs.rows() || d_rho.cols()!=q.size() || !d_rho.allFinite())
            throw std::invalid_argument("Malformed density sensitivity");
        j+=d_rho*derivatives.drho_dvs.matrix().asDiagonal();
    }
    if(!j.allFinite()) throw std::runtime_error("Nonfinite empirical chain rule");
    return j*m.vs.asDiagonal();
}
Eigen::VectorXd FixedVs::log_abs_det_gradient(const Eigen::VectorXd& q) const {
    if(!contains(q)) throw std::invalid_argument("Parameters outside bounds");
    return Eigen::VectorXd::Ones(q.size());
}
}
