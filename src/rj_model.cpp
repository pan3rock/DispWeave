#include "dispweave/rj_model.hpp"
#include <cmath>
#include <stdexcept>
namespace dw {
void RJPrior::validate() const {
    if(min_layers<1 || max_layers<min_layers || max_layers>10000
       || !std::isfinite(depth_limit) || !std::isfinite(min_spacing) || min_spacing<=0
       || depth_limit<=max_layers*min_spacing || !std::isfinite(vs_lower) || !std::isfinite(vs_upper)
       || vs_lower<=0 || vs_upper<=vs_lower || !std::isfinite(vs_upper-vs_lower)
       || layer_weights.size()!=max_layers-min_layers+1 || !layer_weights.allFinite()
       || (layer_weights.array()<0).any() || layer_weights.maxCoeff()<=0)
        throw std::invalid_argument("Invalid RJ prior support or layer weights");
}
bool RJPrior::supports(const RJState& s) const {
    validate();
    const auto k=s.vs.size();
    if(k<min_layers || k>max_layers || s.interfaces.size()!=k-1 || !s.vs.allFinite()
       || !s.interfaces.allFinite() || (s.vs.array()<vs_lower).any() || (s.vs.array()>vs_upper).any()
       || layer_weights[k-min_layers]==0) return false;
    double previous=0;
    for(Eigen::Index j=0;j<s.interfaces.size();++j) {
        if(s.interfaces[j]-previous<min_spacing) return false;
        previous=s.interfaces[j];
    }
    return depth_limit-previous>=min_spacing;
}
double RJPrior::log_density(const RJState& s) const {
    if(!supports(s)) return -INFINITY;
    const auto k=s.vs.size();
    // Scale weights before summation to avoid overflow from large common factors.
    const double scale=layer_weights.maxCoeff();
    const double log_pk=std::log(layer_weights[k-min_layers])-std::log(scale)-std::log((layer_weights/scale).sum());
    return log_pk+std::lgamma(double(k))-(k-1)*std::log(depth_limit-k*min_spacing)-k*std::log(vs_upper-vs_lower);
}
}
