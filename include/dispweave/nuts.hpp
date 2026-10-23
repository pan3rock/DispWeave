#pragma once
#include "hmc.hpp"
namespace dw {
struct NUTSTransition;
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
    double step_size;
};
NUTSResult sample_nuts(const LogDensity&,const Eigen::VectorXd&,const EuclideanMetric&,const NUTSOptions& = {});
}
