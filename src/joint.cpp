#include "dispweave/joint.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
namespace dw {
void NoiseScale::validate() const {
    if(!std::isfinite(first) || !std::isfinite(second) ||
       (kind==NoiseScaleKind::fixed && first<=0) ||
       (kind==NoiseScaleKind::lognormal && second<=0) ||
       (kind==NoiseScaleKind::exponential && first<=0) ||
       (kind!=NoiseScaleKind::fixed && kind!=NoiseScaleKind::lognormal && kind!=NoiseScaleKind::exponential))
        throw std::invalid_argument("Invalid noise scale or normalized scale prior");
    validate_log_scale(initial_log_scale());
}
double NoiseScale::initial_log_scale() const {
    if(kind==NoiseScaleKind::fixed) return std::log(first);
    return kind==NoiseScaleKind::lognormal?first:-std::log(first);
}
JointLikelihood::JointLikelihood(std::vector<ObservationBlock> blocks,BlockDependence assumption):blocks_(std::move(blocks)) {
    if(assumption!=BlockDependence::conditional_independence)
        throw std::invalid_argument("Joint likelihood requires explicit conditional independence of observation blocks");
    if(blocks_.empty()) throw std::invalid_argument("Joint likelihood needs at least one observation block");
    std::set<std::string> names;
    for(const auto& block:blocks_) {
        if(block.name.empty() || block.observable.empty() || block.unit.empty() || !block.predict || !names.insert(block.name).second)
            throw std::invalid_argument("Observation blocks require unique names, observable, unit and predictor");
        block.noise.validate();
        noise_size_+=block.noise.inferred();rows_+=block.likelihood.observations().values.size();
    }
}
bool JointLikelihood::supports_gradient() const {
    return std::all_of(blocks_.begin(),blocks_.end(),[](const auto& b){return b.verified_gradient;});
}
Eigen::VectorXd JointLikelihood::initial_noise() const {
    Eigen::VectorXd noise(noise_size_);Eigen::Index k=0;
    for(const auto& b:blocks_) if(b.noise.inferred()) noise[k++]=b.noise.initial_log_scale();
    return noise;
}
JointEvaluation JointLikelihood::evaluate(const LayeredModel& model,const Eigen::VectorXd& noise,bool gradient) const {
    model.validate();JointEvaluation out;
    if(noise.size()!=noise_size_ || !noise.allFinite()) {out.message="Malformed noise coordinates";return out;}
    out.status=EvaluationStatus::valid;out.log_likelihood=0;
    out.residual=Eigen::VectorXd::Constant(rows_,NAN);out.noise_gradient=Eigen::VectorXd::Constant(noise_size_,NAN);
    if(gradient) {
        out.d_vs=Eigen::MatrixXd::Constant(rows_,model.vs.size(),NAN);out.d_vp=out.d_vs;
    }
    Eigen::Index offset=0,k=0;
    for(const auto& block:blocks_) {
        BlockEvaluation b;b.name=block.name;b.observable=block.observable;b.unit=block.unit;
        b.log_scale=block.noise.inferred()?noise[k]:block.noise.initial_log_scale();
        const auto count=block.likelihood.observations().values.size();
        try {
            validate_log_scale(b.log_scale);
            const auto p=block.predict(model,gradient && block.verified_gradient);
            b.prediction=p.values;b.diagnostics=p.diagnostics;b.message=p.message;b.status=p.status;
            if(b.status==EvaluationStatus::outside_support) {
                b.status=EvaluationStatus::forward_failure;b.message="Predictors cannot turn unavailable physics into prior support rejection: "+p.message;
            }
            const bool value_ok=p.status==EvaluationStatus::valid || p.status==EvaluationStatus::gradient_failure;
            if(value_ok && (p.values.size()!=count || !p.values.allFinite())) {
                b.status=EvaluationStatus::forward_failure;b.message="Malformed prediction; observation rows cannot be discarded";
            } else if(value_ok) {
                b.residual=block.likelihood.whiten(p.values-block.likelihood.observations().values,b.log_scale);
                b.log_likelihood=block.likelihood.log_likelihood(p.values,b.log_scale);
                b.chi_squared=b.residual.squaredNorm();b.log_scale_gradient=b.chi_squared-count;
                out.residual.segment(offset,count)=b.residual;
                out.log_likelihood+=b.log_likelihood;
                if(block.noise.inferred()) out.noise_gradient[k]=b.log_scale_gradient;
                if(gradient) {
                    if(!block.verified_gradient || p.status==EvaluationStatus::gradient_failure ||
                       p.d_vs.rows()!=count || p.d_vp.rows()!=count || p.d_vs.cols()!=model.vs.size() ||
                       p.d_vp.cols()!=model.vs.size() || !p.d_vs.allFinite() || !p.d_vp.allFinite()) {
                        b.status=EvaluationStatus::gradient_failure;
                        if(b.message.empty()) b.message="No verified model gradient for this observation block";
                    } else {
                        out.d_vs.middleRows(offset,count)=block.likelihood.whiten_jacobian(p.d_vs,b.log_scale);
                        out.d_vp.middleRows(offset,count)=block.likelihood.whiten_jacobian(p.d_vp,b.log_scale);
                    }
                }
            }
        } catch(const std::exception& error) {
            b.status=EvaluationStatus::numerical_failure;b.message=error.what();
        }
        if(b.status!=EvaluationStatus::valid) {
            if(out.status==EvaluationStatus::valid || out.status==EvaluationStatus::gradient_failure) {
                out.status=b.status;out.message=block.name+": "+b.message;
            }
            if(b.status!=EvaluationStatus::gradient_failure) out.log_likelihood=NAN;
        }
        out.blocks.push_back(std::move(b));offset+=count;if(block.noise.inferred()) ++k;
    }
    if(out.status==EvaluationStatus::valid && (!std::isfinite(out.log_likelihood) || !out.noise_gradient.allFinite())) {
        out.status=EvaluationStatus::numerical_failure;out.message="Joint likelihood overflow";
    }
    return out;
}
JointProblem::JointProblem(FixedVs parameters,JointLikelihood likelihood)
    :parameters_(std::move(parameters)),likelihood_(std::move(likelihood)) {}
Eigen::VectorXd JointProblem::initial_state(const Eigen::VectorXd& q) const {
    if(!parameters_.contains(q)) throw std::invalid_argument("Initial model is outside parameter support");
    Eigen::VectorXd x(size());x<<q,likelihood_.initial_noise();return x;
}
JointEvaluation JointProblem::evaluate(const Eigen::VectorXd& x,bool gradient) const {
    JointEvaluation out;
    if(x.size()!=size() || !x.allFinite()) {out.message="Malformed joint state";return out;}
    const auto n=parameters_.lower().size();const Eigen::VectorXd q=x.head(n);
    if(!parameters_.contains(q)) {out.status=EvaluationStatus::outside_support;out.log_likelihood=-INFINITY;return out;}
    out=likelihood_.evaluate(parameters_.decode(q),x.tail(likelihood_.noise_size()),gradient);
    if(gradient && out.status==EvaluationStatus::valid) {
        out.jacobian=parameters_.model_jacobian(q,out.d_vs,out.d_vp);
        if(!out.jacobian.allFinite()) {
            out.status=EvaluationStatus::gradient_failure;out.message="Nonfinite parameter Jacobian";out.jacobian.resize(0,0);
        }
    }
    return out;
}
namespace {
DataEvaluation least_squares_data(const JointEvaluation& e) {
    DataEvaluation d;d.status=e.status;d.log_likelihood=e.log_likelihood;d.residual=e.residual;
    d.least_squares_cost=e.residual.size()?.5*e.residual.squaredNorm():NAN;
    d.jacobian=e.jacobian;d.message=e.message;return d;
}
}
JointLeastSquares JointProblem::least_squares(const Eigen::VectorXd& x,bool finite_difference,const FiniteDifferenceOptions& options) const {
    JointLeastSquares out;
    const auto base=evaluate(x,true);out.evaluations=1;out.data=least_squares_data(base);
    if(!finite_difference || (base.status!=EvaluationStatus::valid && base.status!=EvaluationStatus::gradient_failure)) return out;
    if(!std::isfinite(options.step) || options.step<=0 || !std::isfinite(options.absolute_tolerance) ||
       options.absolute_tolerance<0 || !std::isfinite(options.relative_tolerance) || options.relative_tolerance<=0)
        throw std::invalid_argument("Invalid finite-difference step/tolerances");
    const auto n=parameters_.lower().size();Eigen::MatrixXd jacobian(base.residual.size(),n);
    std::vector<bool> approximate(base.residual.size(),false);Eigen::Index offset=0;bool any_approximate=false;
    for(size_t b=0;b<base.blocks.size();++b) {
        const auto count=likelihood_.blocks()[b].likelihood.observations().values.size();
        if(likelihood_.blocks()[b].verified_gradient) {
            if(base.blocks[b].status!=EvaluationStatus::valid) return out;
            jacobian.middleRows(offset,count)=parameters_.model_jacobian(x.head(n),base.d_vs.middleRows(offset,count),base.d_vp.middleRows(offset,count));
            if(!jacobian.middleRows(offset,count).allFinite()) {
                out.data.status=EvaluationStatus::gradient_failure;out.data.message="Nonfinite parameter Jacobian";return out;
            }
        } else {
            for(Eigen::Index row=offset;row<offset+count;++row) approximate[row]=true;
            any_approximate=true;
        }
        offset+=count;
    }
    out.data.status=EvaluationStatus::valid;out.data.message.clear();
    if(!any_approximate) {out.data.jacobian=std::move(jacobian);return out;}
    for(Eigen::Index j=0;j<n;++j) {
        const double lo=parameters_.lower()[j],hi=parameters_.upper()[j];
        const double h=std::min(options.step,(hi-lo)/4);
        bool failed=false;
        auto residual=[&](double delta) -> Eigen::VectorXd {
            auto trial=x;trial[j]+=delta;
            const auto e=evaluate(trial,false);++out.evaluations;
            if(e.status!=EvaluationStatus::valid || e.residual.size()!=base.residual.size() || trial[j]==x[j]) {
                failed=true;return Eigen::VectorXd::Constant(base.residual.size(),NAN);
            }
            return e.residual;
        };
        Eigen::VectorXd coarse,fine;
        if(x[j]-h>=lo && x[j]+h<=hi) {
            const auto plus=residual(h),minus=residual(-h),half_plus=residual(h/2),half_minus=residual(-h/2);
            coarse=(plus-minus)/(2*h);fine=(half_plus-half_minus)/h;
        } else {
            const double sign=x[j]+2*h<=hi?1.:-1.;
            const auto one=residual(sign*h),two=residual(sign*2*h),half=residual(sign*h/2);
            coarse=(-3*base.residual+4*one-two)/(sign*2*h);
            fine=(-3*base.residual+4*half-one)/(sign*h);
        }
        if(failed || !coarse.allFinite() || !fine.allFinite()) {
            out.data.status=EvaluationStatus::gradient_failure;out.data.message="Finite-difference model evaluation failed";return out;
        }
        Eigen::VectorXd difference=(fine-coarse).cwiseAbs();
        for(Eigen::Index row=0;row<difference.size();++row) if(!approximate[row]) difference[row]=0;
        out.max_difference=std::max(out.max_difference,difference.maxCoeff());
        const Eigen::VectorXd tolerance=options.absolute_tolerance+options.relative_tolerance*coarse.cwiseAbs().cwiseMax(fine.cwiseAbs()).array();
        if(!difference.allFinite() || !tolerance.allFinite()) {
            out.data.status=EvaluationStatus::gradient_failure;out.data.message="Finite-difference comparison overflow";return out;
        }
        if((difference.array()>tolerance.array()).any()) {
            Eigen::Index row=0;while(difference[row]<=tolerance[row]) ++row;
            out.data.status=EvaluationStatus::gradient_failure;
            out.data.message="Finite-difference h/h2 consistency check failed: parameter="+std::to_string(j)+
                " row="+std::to_string(row)+" coarse="+std::to_string(coarse[row])+" fine="+std::to_string(fine[row]);return out;
        }
        for(Eigen::Index row=0;row<fine.size();++row) if(approximate[row]) jacobian(row,j)=fine[row];
    }
    out.data.jacobian=std::move(jacobian);return out;
}
JointPosteriorEvaluation joint_log_posterior(const JointProblem& problem,const Eigen::VectorXd& x,const Prior& prior,bool gradient) {
    JointPosteriorEvaluation out;
    const auto n=problem.parameters().lower().size();
    if(x.size()!=problem.size() || !x.allFinite()) {out.message="Malformed joint state";return out;}
    const Eigen::VectorXd q=x.head(n),vs=q.array().exp();
    if(!problem.parameters().contains(q)) {out.status=EvaluationStatus::outside_support;out.value=-INFINITY;return out;}
    if(!prior.evaluate || (prior.measure!=PriorMeasure::physical_vs && prior.measure!=PriorMeasure::log_vs))
        throw std::invalid_argument("Prior must declare its measure and evaluator");
    const auto model_prior=prior.evaluate(prior.measure==PriorMeasure::physical_vs?vs:q,gradient);
    out.log_model_prior=model_prior.value;
    if(model_prior.value==-INFINITY) {out.status=EvaluationStatus::outside_support;out.value=-INFINITY;return out;}
    if(!std::isfinite(model_prior.value)) {out.message="Nonfinite model prior";return out;}
    out.log_transform=prior.measure==PriorMeasure::physical_vs?problem.parameters().log_abs_det_jacobian(q):0;
    out.log_noise_prior=0;Eigen::VectorXd noise_gradient(problem.likelihood().noise_size());Eigen::Index k=0;
    for(const auto& block:problem.likelihood().blocks()) if(block.noise.inferred()) {
        const double eta=x[n+k];double value,derivative;
        if(block.noise.kind==NoiseScaleKind::lognormal) {
            const double z=(eta-block.noise.first)/block.noise.second;
            value=-.5*z*z-std::log(block.noise.second)-.5*std::log(2*std::acos(-1.))-eta;
            derivative=-z/block.noise.second;
        } else {
            const double scaled=block.noise.first*std::exp(eta);
            value=std::log(block.noise.first)-scaled;derivative=1-scaled;
        }
        out.log_noise_prior+=value;out.log_transform+=eta;noise_gradient[k++]=derivative;
    }
    if(!std::isfinite(out.log_noise_prior) || !std::isfinite(out.log_transform) || !noise_gradient.allFinite()) {
        out.message="Noise prior/transform overflow";return out;
    }
    out.data=problem.evaluate(x,gradient);out.status=out.data.status;out.message=out.data.message;
    out.log_likelihood=out.data.log_likelihood;
    if(out.status!=EvaluationStatus::valid && out.status!=EvaluationStatus::gradient_failure) return out;
    out.value=out.log_likelihood+out.log_model_prior+out.log_noise_prior+out.log_transform;
    if(gradient && out.status==EvaluationStatus::valid) {
        if(model_prior.gradient.size()!=n || !model_prior.gradient.allFinite()) {
            out.status=EvaluationStatus::gradient_failure;out.message="Malformed model prior gradient";return out;
        }
        out.gradient.resize(x.size());
        out.gradient.head(n)=-out.data.jacobian.transpose()*out.data.residual;
        if(prior.measure==PriorMeasure::physical_vs)
            out.gradient.head(n)+=(model_prior.gradient.array()*vs.array()).matrix()+problem.parameters().log_abs_det_gradient(q);
        else out.gradient.head(n)+=model_prior.gradient;
        out.gradient.tail(k)=out.data.noise_gradient+noise_gradient;
    }
    if(!std::isfinite(out.value) || (out.gradient.size() && !out.gradient.allFinite())) {
        out.status=EvaluationStatus::numerical_failure;out.value=NAN;out.gradient.resize(0);out.message="Joint posterior overflow";
    }
    return out;
}
LogDensity joint_density(const JointProblem& problem,Prior prior) {
    if(!problem.likelihood().supports_gradient()) throw std::invalid_argument("Joint HMC/NUTS requires verified gradients for every observation block; no finite-difference fallback");
    return [&problem,prior=std::move(prior)](const Eigen::VectorXd& x) {
        const auto out=joint_log_posterior(problem,x,prior,true);
        return DensityEvaluation{out.status,out.value,out.gradient};
    };
}
}
