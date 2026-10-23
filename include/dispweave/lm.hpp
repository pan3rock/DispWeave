#pragma once
#include "objective.hpp"
namespace dw {
struct LMOptions {
    int max_iterations=150;
    double gradient_tolerance=1e-7;
    double initial_damping=1e-3;
    double step_tolerance=1e-12; // Fraction of parameter support width; stagnation only.
};
struct LMIteration {
    int iteration;
    double cost, trial_cost, damping, gain_ratio, projected_gradient;
    bool accepted;
    EvaluationStatus trial_status;
    double predicted_reduction, actual_reduction, step_norm;
    int box_iterations;
};
enum class LMTermination { converged, stalled, failed };
const char* to_string(LMTermination);
struct LMResult {
    Eigen::VectorXd q;
    DataEvaluation evaluation;
    bool converged=false;
    LMTermination termination=LMTermination::failed;
    std::string reason;
    int evaluations=0, accepted=0, rejected=0, forward_failures=0, gradient_failures=0, box_iterations=0;
    double projected_gradient=NAN;
    std::vector<LMIteration> trace;
};
using LeastSquaresFunction=std::function<DataEvaluation(const Eigen::VectorXd&)>;
// Residual and Jacobian must use identical weighting; cost is recomputed from residual.
// Bounds are finite, strictly ordered, and inclusive. No prior/regularizer is added.
LMResult bounded_lm(const LeastSquaresFunction&,const Eigen::VectorXd& initial,
                    const Eigen::VectorXd& lower,const Eigen::VectorXd& upper,
                    const LMOptions& options={});
// Freeze the listed zero-based layers at their initial log-Vs values. Solve
// only the remaining coordinates; q and the returned Jacobian stay full-sized.
// Projected gradient and step norms refer to the free coordinates only.
LMResult invert_lm(const DispersionProblem&,const Eigen::VectorXd& initial,const LMOptions& options={},
                  const std::vector<int>& fixed_vs_layers={});
}
