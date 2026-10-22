#include "dispweave/predictive.hpp"
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
}
