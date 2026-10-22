#include "dispweave/joint.hpp"
#include "dispweave/ellipticity.hpp"
#include <set>
#include <stdexcept>
namespace dw {
namespace {
void queries_valid(const std::vector<DispersionQuery>& queries,Eigen::Index size) {
    if(queries.size()!=static_cast<size_t>(size)) throw std::invalid_argument("Observation/query size mismatch");
    for(const auto& q:queries) if(!std::isfinite(q.frequency_hz) || q.frequency_hz<=0 || q.mode<0 || q.mode>1023)
        throw std::invalid_argument("Invalid modal observation query");
}
}
ObservationBlock dispersion_block(std::string name,const Forward& backend,DispersionObservations data,
                                  NoiseScale noise,SensitivityMethod method) {
    data.validate();noise.validate();
    if(method!=SensitivityMethod::energy && method!=SensitivityMethod::implicit)
        throw std::invalid_argument("Dispersion block requires an explicit verified derivative method");
    return {std::move(name),"rayleigh_phase_velocity","km/s",GaussianDataLikelihood({data.phase_velocity,data.covariance}),
        [&backend,queries=data.queries,method](const LayeredModel& model,bool gradient) {
            ModelPrediction out;
            const auto f=backend.evaluate(model,queries,gradient?method:SensitivityMethod::none);
            out.values=Eigen::VectorXd::Constant(f.rows.size(),NAN);
            for(size_t i=0;i<f.rows.size();++i) {
                const auto& row=f.rows[i];out.values[i]=row.phase_velocity;
                out.diagnostics.push_back(std::string(to_string(row.status))+": "+row.message+
                    (gradient?"; sensitivity="+std::string(to_string(row.sensitivity.status))+": "+row.sensitivity.message:""));
            }
            if(f.rows.size()!=queries.size() || !f.all_valid()) {
                out.status=EvaluationStatus::forward_failure;out.message="Unresolved dispersion rows; none discarded";return out;
            }
            if(gradient) {
                if(!f.all_sensitivities_valid()) {
                    out.status=EvaluationStatus::gradient_failure;out.message="Valid roots but unavailable sensitivity";return out;
                }
                out.d_vs.resize(f.rows.size(),model.vs.size());out.d_vp.resize(f.rows.size(),model.vs.size());
                for(size_t i=0;i<f.rows.size();++i) {
                    if(f.rows[i].sensitivity.dc_dvs.size()!=model.vs.size() || f.rows[i].sensitivity.dc_dvp.size()!=model.vs.size()) {
                        out.status=EvaluationStatus::gradient_failure;out.message="Malformed physical sensitivity";return out;
                    }
                    out.d_vs.row(i)=f.rows[i].sensitivity.dc_dvs.transpose();
                    out.d_vp.row(i)=f.rows[i].sensitivity.dc_dvp.transpose();
                }
            }
            out.status=EvaluationStatus::valid;return out;
        },true,noise};
}
ObservationBlock ellipticity_block(std::string name,std::string interpretation,std::vector<DispersionQuery> queries,
                                   GaussianData data,NoiseScale noise,SecularMethod method) {
    if(interpretation!="rayleigh_modal_ellipticity")
        throw std::invalid_argument("Ellipticity data must explicitly declare rayleigh_modal_ellipticity; ambient H/V requires a separate wavefield model");
    data.validate();noise.validate();queries_valid(queries,data.values.size());
    if((data.values.array()<0).any()) throw std::invalid_argument("Unsigned modal ellipticity cannot be negative");
    return {std::move(name),std::move(interpretation),"dimensionless",GaussianDataLikelihood(std::move(data)),
        [queries=std::move(queries),method](const LayeredModel& model,bool) {
            ModelPrediction out;const auto rows=rayleigh_ellipticity(model,queries,method);
            out.values=Eigen::VectorXd::Constant(rows.size(),NAN);out.status=EvaluationStatus::valid;
            for(size_t i=0;i<rows.size();++i) {
                out.values[i]=rows[i].ratio;
                out.diagnostics.push_back(std::string(to_string(rows[i].status))+": "+rows[i].message);
                if(rows[i].status!=EllipticityStatus::valid) out.status=EvaluationStatus::forward_failure;
            }
            if(out.status!=EvaluationStatus::valid) out.message="Unavailable modal ellipticity; none discarded";
            return out;
        },false,noise};
}
ObservationBlock receiver_block(std::string name,ReceiverOptions options,std::vector<int> indices,GaussianData data,NoiseScale noise) {
    data.validate();noise.validate();options.validate();
    if(!std::isfinite(options.slowness_s_km) || options.slowness_s_km<0) throw std::invalid_argument("Invalid receiver slowness");
    if(indices.size()!=static_cast<size_t>(data.values.size())) throw std::invalid_argument("Receiver sample/value size mismatch");
    std::set<int> seen;
    for(int i:indices) if(i<0 || i>=options.samples || !seen.insert(i).second)
        throw std::invalid_argument("Receiver sample indices must be unique and within the processing grid");
    return {std::move(name),"p_receiver_function","dimensionless",GaussianDataLikelihood(std::move(data)),
        [options=std::move(options),indices=std::move(indices)](const LayeredModel& model,bool) {
            ModelPrediction out;const auto rf=receiver_function(model,options);
            out.values=Eigen::VectorXd::Constant(indices.size(),NAN);
            if(rf.status!=ReceiverStatus::valid) {
                out.status=EvaluationStatus::forward_failure;out.message=rf.message;
                for(size_t k=0;k<rf.spectrum.size();++k) if(rf.spectrum[k].status!=ReceiverStatus::valid)
                    out.diagnostics.push_back("f="+std::to_string(rf.frequency_hz[k])+": "+rf.spectrum[k].message);
                return out;
            }
            for(size_t j=0;j<indices.size();++j) out.values[j]=rf.radial_rf[indices[j]];
            out.status=EvaluationStatus::valid;return out;
        },false,noise};
}
}
