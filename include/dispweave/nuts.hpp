#pragma once
#include "hmc.hpp"
namespace dw {
// Slice NUTS: Hoffman & Gelman (2014), Algorithm 6, with a fixed
// Euclidean metric and a finite tree-depth budget. No multinomial selection.
struct NUTSOptions {
    int warmup=500,draws=1000,max_depth=10;
    double step_size=.1,target_accept=.8,divergence_threshold=1000;
    bool adapt_step_size=true;
    MetricAdaptation metric_adaptation=MetricAdaptation::none;
    std::uint64_t seed=2026,chain_id=0;
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
    double step_size;
};
NUTSResult sample_nuts(const LogDensity&,const Eigen::VectorXd&,const EuclideanMetric&,const NUTSOptions& = {});
}
