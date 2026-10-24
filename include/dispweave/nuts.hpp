#pragma once
#include "hmc.hpp"
namespace dw {
struct NUTSTransition;
struct MetricDecision {
    int warmup_iteration;
    bool accepted;
    double old_stiffness,candidate_stiffness;
};
enum class NUTSKernel { slice, multinomial };
// The legacy slice kernel remains the default for reproducible existing runs.
struct NUTSOptions {
    int warmup=500,draws=1000,max_depth=10;
    double step_size=.1,target_accept=.8,divergence_threshold=1000;
    bool adapt_step_size=true;
    MetricAdaptation metric_adaptation=MetricAdaptation::none;
    std::uint64_t seed=2026,chain_id=0;
    NUTSKernel kernel=NUTSKernel::slice;
    bool find_initial_step_size=false; // Also search after each metric update.
    // Pseudo-observations retaining the previous window's POSITION covariance.
    double metric_shrinkage=0; // Zero preserves the original adaptation path.
    bool conservative_warmup=false; // Shrink-only search; DA centered on the starting step.
    bool conservative_metric_restart=false; // Keep ordinary startup; use conservative restarts after accepted metric updates.
    double metric_max_stiffness_ratio=0; // 0 disables the local-curvature guard; otherwise >=1.
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> local_curvature;
    std::function<void(const MetricDecision&)> metric_progress;
    // Optional observation only; called after every transition, including warmup.
    std::function<void(int,const Eigen::VectorXd&,const NUTSTransition&)> progress;
};
struct StepSizeSearch {
    int warmup_iteration,evaluations; // 0 before warmup; otherwise after this iteration.
    double step_size;
    EvaluationStatus last_status;
};
struct NUTSTransition {
    bool warmup,accepted,divergent,max_depth_reached;
    int depth,evaluations;
    double step_size,acceptance_probability,max_energy_error;
    EvaluationStatus status;
};
struct NUTSResult {
    Eigen::MatrixXd draws,mass;
    std::vector<NUTSTransition> transitions;
    std::vector<MetricUpdate> metric_updates;
    std::vector<StepSizeSearch> step_size_searches;
    std::vector<MetricDecision> metric_decisions;
    double step_size;
};
NUTSResult sample_nuts(const LogDensity&,const Eigen::VectorXd&,const EuclideanMetric&,const NUTSOptions& = {});
}
