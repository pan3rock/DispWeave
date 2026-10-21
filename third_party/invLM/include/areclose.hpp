#pragma once

#include <cfloat>
#include <cmath>

struct AreClose {
public:
  AreClose() : abs_tol{0}, rel_tol{1e-09} {}

  AreClose(double absTol, double relTol) : abs_tol{absTol}, rel_tol{relTol} {
    if (invalidTolerance(abs_tol) || invalidTolerance(rel_tol))
      abs_tol = rel_tol = NAN;
  }

  bool isValid() const {
    return abs_tol == abs_tol && rel_tol == rel_tol;
  }

  bool operator()(double a, double b) const {
    if (a == b)
      return true;
    if (std::isinf(a) || std::isinf(b))
      return false;
    double diff = std::fabs(a - b);
    return diff <= abs_tol || diff <= std::fabs(rel_tol * a) ||
           diff <= std::fabs(rel_tol * b);
  }

  double absoluteTolerance() const { return abs_tol; }

  bool setAbsoluteTolerance(double absTol) {
    if (invalidTolerance(absTol))
      return false;
    abs_tol = absTol;
    return true;
  }

  double relativeTolerance() const { return rel_tol; }

  bool setRelativeTolerance(double relTol) {
    if (invalidTolerance(relTol))
      return false;
    rel_tol = relTol;
    return true;
  }

  static bool invalidTolerance(double tol) {
    return !(tol == 0.0 || (tol >= 4 * DBL_EPSILON && std::isfinite(tol)));
  }

private:
  double abs_tol, rel_tol;
};
