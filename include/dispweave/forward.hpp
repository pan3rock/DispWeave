#pragma once
#include "model.hpp"
#include <limits>
#include <string>
#include <vector>
namespace dw {
struct DispersionQuery { double frequency_hz; int mode; };
enum class RootStatus { valid, missing_or_unresolved, numerical_failure };
const char* to_string(RootStatus);
enum class SensitivityMethod { none, energy, implicit };
enum class SensitivityStatus { not_requested, valid, unavailable };
const char* to_string(SensitivityStatus);
const char* to_string(SensitivityMethod);
struct SensitivityResult {
    SensitivityStatus status=SensitivityStatus::not_requested;
    Eigen::VectorXd dc_dvs, dc_dvp, dc_drho;
    double energy_residual=std::numeric_limits<double>::quiet_NaN();
    double root_correction=std::numeric_limits<double>::quiet_NaN();
    bool stabilized=false;
    std::string message;
};
// Evaluate at an already resolved phase; never changes or replaces that phase.
SensitivityResult rayleigh_sensitivity(const LayeredModel&,double frequency_hz,
                                      double phase_velocity,SensitivityMethod);
struct RootResult {
    double phase_velocity=std::numeric_limits<double>::quiet_NaN();
    RootStatus status=RootStatus::missing_or_unresolved;
    SensitivityResult sensitivity;
    int roots_found=0; // Found requested prefix, NOT a total-mode count or certificate.
    std::string message;
};
struct ForwardResult {
    SensitivityMethod sensitivity_method=SensitivityMethod::none;
    std::vector<RootResult> rows;
    bool all_valid() const;
    bool all_sensitivities_valid() const;
};
class Forward {
public:
    virtual ~Forward()=default;
    virtual ForwardResult evaluate(const LayeredModel&,const std::vector<DispersionQuery>&,
                                   SensitivityMethod=SensitivityMethod::none) const=0;
};
enum class SecularMethod { dunkin, fast_scalar };
class InvLMForward final : public Forward {
public:
    explicit InvLMForward(SecularMethod method=SecularMethod::dunkin):method_(method) {}
    ForwardResult evaluate(const LayeredModel&,const std::vector<DispersionQuery>&,
                           SensitivityMethod=SensitivityMethod::none) const override;
private:
    SecularMethod method_;
};
}
