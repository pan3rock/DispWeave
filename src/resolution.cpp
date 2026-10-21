#include "dispweave/resolution.hpp"
#include <Eigen/SVD>
#include <cmath>
#include <stdexcept>
namespace dw {
LocalResolution local_resolution(const Eigen::MatrixXd& j,const Eigen::VectorXd& scale,
                                 double ridge,double tolerance) {
    if(j.rows()==0 || j.cols()==0 || scale.size()!=j.cols() || !j.allFinite() || !scale.allFinite()
       || (scale.array()<=0).any() || !std::isfinite(ridge) || ridge<=0
       || !std::isfinite(tolerance) || tolerance<=0 || tolerance>=1)
        throw std::invalid_argument("Require finite nonempty Jacobian, positive scales/ridge and rank tolerance in (0,1)");
    LocalResolution out; out.scaled_jacobian=j*scale.asDiagonal();
    if(!out.scaled_jacobian.allFinite()) throw std::invalid_argument("Scaled Jacobian overflow");
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(out.scaled_jacobian,Eigen::ComputeFullV);
    if(svd.info()!=Eigen::Success || !svd.singularValues().allFinite()) throw std::runtime_error("Resolution SVD failed");
    const auto n=j.cols(); out.singular_values=Eigen::VectorXd::Zero(n);
    out.singular_values.head(svd.singularValues().size())=svd.singularValues();
    out.right_vectors=svd.matrixV(); out.filter_factors.resize(n);
    out.rank_threshold=tolerance*out.singular_values[0];
    for(Eigen::Index i=0;i<n;++i) {
        const double s=out.singular_values[i];
        if(s>out.rank_threshold) ++out.numerical_rank;
        // Ratio branches avoid squaring large singular values or ridge parameters.
        if(s>=ridge) { const double t=ridge/s; out.filter_factors[i]=1/(1+t*t); }
        else { const double t=s/ridge; out.filter_factors[i]=t*t/(1+t*t); }
    }
    out.effective_dimension=out.filter_factors.sum();
    out.resolution=out.right_vectors*out.filter_factors.asDiagonal()*out.right_vectors.transpose();
    return out;
}
}
