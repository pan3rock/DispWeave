#pragma once
#include "dispweave/model.hpp"
#include <initializer_list>
inline Eigen::VectorXd v(std::initializer_list<double> a) { return Eigen::Map<const Eigen::VectorXd>(a.begin(),a.size()); }
inline dw::LayeredModel normal() { return {v({0,2,4}),v({1,2,3}),v({1.8,3.6,5.2}),v({2,2.2,2.5})}; }
inline dw::LayeredModel low_velocity() { return {v({0,.01,.02,.04}),v({.18,.35,.25,.60}),v({1.5,1.7,1.6,1.9}),v({1.78,1.85,1.80,1.93})}; }
inline dw::LayeredModel dense() { return {v({0,.008,.016,.024,.032,.04}),v({.119,.201,.446,.111,.272,.5}),v({.205,.348,.773,.192,.470,.866}),v({1.174,1.339,1.634,1.153,1.444,1.682})}; }
