#pragma once
#include "rj_model.hpp"
namespace dw {
enum class RJMove { birth,death,interface,velocity,velocity_refresh,correlated };
enum class RJProposalStatus { valid,unavailable,outside_support };
struct RJProposal {
    RJState state;
    RJMove move;
    RJProposalStatus status=RJProposalStatus::unavailable;
    int index=-1; // Inserted/deleted interface, or updated continuous coordinate.
    double log_forward=0,log_reverse=0,log_jacobian=0;
    double log_hastings() const {return log_reverse-log_forward+log_jacobian;}
};
// Joint independent Uniform(Vs lower,upper) proposal at fixed K/interfaces.
// Conditional proposal densities are equal; state-independent mixture weights cancel.
RJProposal rj_velocity_refresh(const RJPrior&,const RJState&,const Eigen::VectorXd& proposed_vs);
// Geometry only: length of the union of admissible insertion intervals.
// Source must be supported by the prior. Does not impose max_layers on the length.
double rj_insertion_length(const RJPrior&,const RJState&);
// Uniform union coordinate u in [0,1); throws when the insertion union is empty.
double rj_insertion_position(const RJPrior&,const RJState&,double u);
// Deterministic, inspectable proposals. No RNG or target likelihood evaluation.
// Probabilities include move selection; unavailable moves retain source state.
RJProposal rj_birth(const RJPrior&,const RJState&,double depth,double new_vs,
                    double birth_probability=.25,double death_probability=.25);
RJProposal rj_death(const RJPrior&,const RJState&,int interface_index,
                    double birth_probability=.25,double death_probability=.25);
}
