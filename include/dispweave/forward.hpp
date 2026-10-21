#pragma once
#include "model.hpp"
#include <limits>
#include <string>
#include <vector>
namespace dw {
struct DispersionQuery { double frequency_hz; int mode; };
enum class RootStatus { valid, missing_or_unresolved, numerical_failure };
const char* to_string(RootStatus);
struct RootResult {
    double phase_velocity=std::numeric_limits<double>::quiet_NaN();
    RootStatus status=RootStatus::missing_or_unresolved;
    int roots_found=0; // Found requested prefix, NOT a total-mode count or certificate.
    std::string message;
};
struct ForwardResult {
    std::vector<RootResult> rows;
    bool all_valid() const;
};
class Forward {
public:
    virtual ~Forward()=default;
    virtual ForwardResult evaluate(const LayeredModel&,const std::vector<DispersionQuery>&) const=0;
};
enum class SecularMethod { dunkin, fast_scalar };
class InvLMForward final : public Forward {
public:
    explicit InvLMForward(SecularMethod method=SecularMethod::dunkin):method_(method) {}
    ForwardResult evaluate(const LayeredModel&,const std::vector<DispersionQuery>&) const override;
private:
    SecularMethod method_;
};
}
