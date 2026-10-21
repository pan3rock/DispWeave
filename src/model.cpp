#include "dispweave/model.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
void LayeredModel::validate() const {
    const auto n=vs.size();
    if(n<1 || depth.size()!=n || vp.size()!=n || rho.size()!=n ||
       !depth.allFinite() || !vs.allFinite() || !vp.allFinite() || !rho.allFinite())
        throw std::invalid_argument("Model arrays must be finite, nonempty and equal-sized");
    if(depth[0]!=0 || (vs.array()<=0).any() || (rho.array()<=0).any() ||
       (vp.array()<=std::sqrt(4.0/3.0)*vs.array()).any())
        throw std::invalid_argument("Require depth[0]=0, Vs/rho>0 and Vp>sqrt(4/3)*Vs; fluids unsupported");
    for(Eigen::Index i=1;i<n;++i)
        if(depth[i]<=depth[i-1]) throw std::invalid_argument("Layer tops must strictly increase");
}
}
