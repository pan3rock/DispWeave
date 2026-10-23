#pragma once
#include "model.hpp"
#include <string>
namespace dw {
// Numerically checked full trapped interval, not an interval-arithmetic proof.
// Unresolved counts must NEVER be interpreted as absent modes.
struct ModeCount {
    bool resolved=false;
    int count=-1,evaluations=0;
    double lower_speed=0,maximum_phase_increment=0;
    std::string message;
};
ModeCount trapped_mode_count(const LayeredModel&,double frequency,int max_segments=2048);
}
