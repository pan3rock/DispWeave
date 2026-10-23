#pragma once
#include <Eigen/Core>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>
namespace stan::math {
inline double log_sum_exp(double x,double y) {
    if(x==-INFINITY) return y;
    if(y==-INFINITY) return x;
    const double largest=std::max(x,y);
    return largest+std::log1p(std::exp(std::min(x,y)-largest));
}
}
