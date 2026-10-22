#include "dispweave/predictive.hpp"
#include "dispweave/rj_dispersion.hpp"
#include <Eigen/Cholesky>
#include <random>
#include <stdexcept>
namespace dw {
PredictiveResult posterior_predictive(const DispersionProblem& problem,const Eigen::MatrixXd& q,std::uint64_t seed) {
    const auto& data=problem.likelihood().data();
    if(q.rows()<1 || q.cols()!=problem.parameters().lower().size() || !q.allFinite())
        throw std::invalid_argument("Invalid posterior predictive parameter draws");
    const auto n=data.phase_velocity.size();
    PredictiveResult r;
    r.latent=Eigen::MatrixXd::Constant(q.rows(),n,NAN);r.replicated=r.latent;
    r.observed_discrepancy=Eigen::VectorXd::Constant(q.rows(),NAN);r.replicated_discrepancy=r.observed_discrepancy;
    std::mt19937_64 rng(seed);std::normal_distribution<double> normal;
    Eigen::LLT<Eigen::MatrixXd> factor(data.covariance);
    if(factor.info()!=Eigen::Success) throw std::invalid_argument("Invalid predictive covariance");
    for(Eigen::Index i=0;i<q.rows();++i) {
        Eigen::VectorXd z(n);for(Eigen::Index j=0;j<n;++j) z[j]=normal(rng);
        const auto e=problem.evaluate(q.row(i).transpose(),false);
        r.status.push_back(e.status);
        if(e.status!=EvaluationStatus::valid) continue;
        Eigen::VectorXd prediction(n);
        for(Eigen::Index j=0;j<n;++j) prediction[j]=e.forward.rows[j].phase_velocity;
        const Eigen::VectorXd replicated=prediction+factor.matrixL()*z;
        if(!replicated.allFinite()) {r.status.back()=EvaluationStatus::numerical_failure;continue;}
        r.latent.row(i)=prediction.transpose();r.replicated.row(i)=replicated.transpose();
        r.observed_discrepancy[i]=e.residual.squaredNorm();
        r.replicated_discrepancy[i]=z.squaredNorm();
    }
    return r;
}
PredictiveResult posterior_predictive(const RJDispersionProblem& problem,const std::vector<RJState>& draws,std::uint64_t seed) {
    if(draws.empty()) throw std::invalid_argument("No RJ predictive draws");
    const auto& data=problem.likelihood().data();const auto n=data.phase_velocity.size();
    PredictiveResult r;r.latent=Eigen::MatrixXd::Constant(draws.size(),n,NAN);r.replicated=r.latent;
    r.observed_discrepancy=Eigen::VectorXd::Constant(draws.size(),NAN);r.replicated_discrepancy=r.observed_discrepancy;
    Eigen::LLT<Eigen::MatrixXd> factor(data.covariance);
    if(factor.info()!=Eigen::Success) throw std::invalid_argument("Invalid predictive covariance");
    std::mt19937_64 rng(seed);std::normal_distribution<double> normal;
    for(size_t i=0;i<draws.size();++i) {
        Eigen::VectorXd z(n);for(Eigen::Index j=0;j<n;++j) z[j]=normal(rng);
        const auto e=problem.evaluate(draws[i]);r.status.push_back(e.likelihood.status);
        if(e.likelihood.status!=EvaluationStatus::valid) continue;
        const Eigen::VectorXd replicated=e.prediction+factor.matrixL()*z;
        if(!replicated.allFinite()) {r.status.back()=EvaluationStatus::numerical_failure;continue;}
        r.latent.row(i)=e.prediction.transpose();r.replicated.row(i)=replicated.transpose();
        r.observed_discrepancy[i]=problem.likelihood().whiten(e.prediction-data.phase_velocity).squaredNorm();
        r.replicated_discrepancy[i]=z.squaredNorm();
    }
    return r;
}

}
