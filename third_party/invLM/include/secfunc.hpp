#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <stdexcept>

namespace fwd {

/// Dispersion secular-function evaluation method.
enum class DispMethod { DUNKIN, FAST_SCALAR };

/// Parse a DispMethod from a config string.
inline DispMethod disp_method_from_string(const std::string& s) {
    if (s == "dunkin") return DispMethod::DUNKIN;
    if (s == "fast_scalar") return DispMethod::FAST_SCALAR;
    throw std::invalid_argument("forward_method must be dunkin or fast_scalar");
}

/// Abstract base class for secular function evaluation.
///
/// Subclasses implement different algorithms for evaluating the Rayleigh
/// (PSV) secular equation at a given (frequency, phase-velocity) pair.
/// Love (SH) waves use a shared Haskell 2×2 implementation in the base class.
class SecularFunction {
public:
    SecularFunction(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh);
    virtual ~SecularFunction() = default;

    /// Evaluate the secular function D(f, c).
    /// Roots in c (for a fixed f) are Rayleigh/Love wave phase velocities.
    virtual double evaluate(double f, double c) = 0;

    // Number of trapped SH modes below c, from displacement nodes and surface
    // traction sign (Sturm oscillation count). -1 means invalid/unresolved.
    int love_root_count(double f, double c) const;

    /// Factory: create a SecularFunction of the requested method.
    static std::unique_ptr<SecularFunction>
    create(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh,
           DispMethod method);

protected:
    /// Common SH (Love wave) secular function implementation.
    /// Used by all methods — the SH case is simple enough that a single
    /// Haskell 2×2 propagator suffices regardless of the PSV method.
    double evaluate_sh_impl(double f, double c) const;

    // Model data
    const int nl_;
    Eigen::ArrayXd thk_, dns_, vs_, vp_;
    const bool sh_;
    const bool is_water_;
    int iwater_;
};

/// Dunkin (1965) 5×5 compound-matrix method for PSV (Rayleigh) waves.
class DunkinSecularFunction final : public SecularFunction {
public:
    using SecularFunction::SecularFunction;
    double evaluate(double f, double c) override;

private:
    double evaluate_psv(double f, double c);

    void var(double p, double q, double ra, double rb, double wvno, double xka,
             double xkb, double dpth, double &w, double &cosp, double &a0,
             double &cpcq, double &cpy, double &cpz, double &cqw, double &cqx,
             double &xy, double &xz, double &wy, double &wz);

    void dnka(double wvno2, double gam, double gammk, double rho,
              double a0, double cpcq, double cpy, double cpz,
              double cqw, double cqx, double xy, double xz, double wy,
              double wz, Eigen::Array<double, 5, 5> &ca);
};

} // namespace fwd
