#include "toms748.h"
#include "areclose.hpp"
#include <cfloat>
#include <cmath>
using namespace std;

static const int defaultInterpolationsPerIteration = 2,
    defaultMaximumIterations = 100;
static const double eps = DBL_EPSILON, mu = 0.5,
    defaultRelativeTolerance = eps * 4, defaultAbsoluteTolerance = 1e-8;

struct Toms748Solver {
private:
  Toms748InputFunction f;
  void *args;
  double a, fa, b, fb, c, fc, d, fd, e, fe, u, fu;
  Toms748ResultStatus *stat;
  AreClose areClose;
  int maxiter, remiter, fncount, k;
  double root;

public:
  Toms748Solver(Toms748InputFunction function, void *otherInput,
                double intervalStart, double intervalEnd,
                Toms748ResultStatus *resultStatus, double absoluteTolerance,
                double relativeTolerance, int maximumIterations,
                int interpolationsPerIteration)
      : f{function}, args{otherInput}, a{intervalStart}, fa{0.0},
        b{intervalEnd}, fb{0.0}, c{0.0}, fc{0.0}, d{0.0}, fd{0.0}, e{NAN},
        fe{0.0}, u{0.0}, fu{0.0}, stat{resultStatus},
        areClose{absoluteTolerance, relativeTolerance},
        maxiter{maximumIterations}, remiter{maximumIterations}, fncount{0},
        k{interpolationsPerIteration}
  {
    fe = e = fd = 1e5F;
    if (checkBracketForEnd()) return;
    if (callFunctionAndCheckEnd(a, fa)) return;
    if (callFunctionAndCheckEnd(b, fb)) return;
    if (dontBracketARoot(fa, fb)) {
      finalize(NAN, TOMS748_INTERVAL_DOES_NOT_BRACKET_A_ROOT, a, b);
      return;
    }
    interpolateSecant();
    if (evalRebracketAndCheckEnd()) return;
    --remiter;
    for (; remiter; --remiter) {
      double initBracketWidth = b - a;
      for (int stepCount = 2; stepCount < k + 2; ++stepCount) {
        if (cantDoCubic())
          interpolateNewtonQuadratic(stepCount);
        else
          interpolateCubic();
        if (evalRebracketAndCheckEnd()) return;
      }
      interpolateDoubleLengthSecant();
      if (evalRebracketAndCheckEnd()) return;
      if (b - a > mu * initBracketWidth)
        interpolateBisection();
      if (evalRebracketAndCheckEnd()) return;
    }
    finalize(NAN, TOMS748_MAXIMUM_ITERATIONS_REACHED, a, b);
  }

  double result() const { return root; }

private:
  void finalize(double val, int errorCode, double start, double end) {
    root = val;
    if (stat) {
      stat->iterations = maxiter - remiter;
      stat->functionCalls = fncount;
      stat->errorCode = errorCode;
      stat->bracketStart = start;
      stat->bracketEnd = end;
    }
  }

  bool checkBracketForEnd() {
    if (!areClose(a, b)) return false;
    interpolateBisection();
    finalize(c, TOMS748_NO_ERROR, a, b);
    return true;
  }

  bool callFunctionAndCheckEnd(double x, double &fx) {
    fx = f(x, args);
    fncount += 1;
    if (fx == 0) { finalize(x, TOMS748_NO_ERROR, x, x); return true; }
    if (!isfinite(fx)) {
      finalize(NAN, TOMS748_INVALID_FUNCTION_VALUE, a, b);
      return true;
    }
    return false;
  }

  static bool dontBracketARoot(double p, double q) { return p * q > 0; }
  bool outOfBracket(double val) { return val <= a || val >= b; }
  void interpolateBisection() { c = a + (b - a) / 2; }

  void interpolateSecant() {
    if (fabs(fb) > fabs(fa))
      c = (a - fa / fb * b) / (1 - fa / fb);
    else
      c = (b - fb / fa * a) / (1 - fb / fa);
    if (c < a + fabs(a) * eps * 5 || c > b - fabs(b) * eps * 5)
      interpolateBisection();
  }

  void interpolateDoubleLengthSecant() {
    if (fabs(fa) < fabs(fb)) { u = a; fu = fa; }
    else { u = b; fu = fb; }
    c = u - 2 * (fu / (fb - fa)) * (b - a);
    double sepCU = fabs(c - u);
    if (sepCU > (b - a) / 2)
      interpolateBisection();
    else if (sepCU < eps * u) {
      int aExp, bExp, largeExp, smallExp;
      double nonU;
      frexp(a, &aExp); frexp(b, &bExp);
      if (aExp < bExp) { smallExp = aExp; largeExp = bExp; nonU = b; }
      else { smallExp = bExp; largeExp = aExp; nonU = a; }
      if (smallExp < largeExp - 50)
        c = (31 * u + nonU) / 32;
      else {
        double adj = fmax(areClose.absoluteTolerance(),
                          areClose.relativeTolerance() * fabs(c));
        if (u == a) c = u + adj;
        else c = u - adj;
        if (outOfBracket(c)) interpolateBisection();
      }
    }
  }

  static double safeDiv(double n, double d, double r) {
    if (fabs(d) < 1 && fabs(d) * DBL_MAX <= fabs(n)) return r;
    return n / d;
  }

  void interpolateNewtonQuadratic(int stepCount) {
    double A, B;
    A = safeDiv(fd - fb, d - b, DBL_MAX);
    B = safeDiv(fb - fa, b - a, DBL_MAX);
    A = safeDiv(A - B, d - a, 0);
    if (A == 0) return interpolateSecant();
    if (dontBracketARoot(A, fa)) c = a;
    else c = b;
    int step;
    for (step = 0; step < stepCount; ++step) {
      double c1 = c - safeDiv(fa + (B + A * (c - b)) * (c - a),
                              B + A * (2 * c - a - b), 1 + c - a);
      if (outOfBracket(c1)) break;
      c = c1;
    }
    if (step == 0) interpolateSecant();
  }

  void interpolateCubic() {
    double q11 = (d - e) * fd / (fe - fd);
    double q21 = (b - d) * fb / (fd - fb);
    double q31 = (a - b) * fa / (fb - fa);
    double d21 = (b - d) * fd / (fd - fb);
    double d31 = (a - b) * fb / (fb - fa);
    double q22 = (d21 - q11) * fb / (fe - fb);
    double q32 = (d31 - q21) * fa / (fd - fa);
    double d32 = (d31 - q21) * fd / (fd - fa);
    double q33 = (d32 - q22) * fa / (fe - fa);
    c = q31 + q32 + q33 + a;
    if (outOfBracket(c)) interpolateNewtonQuadratic(3);
  }

  bool cantDoCubic() {
    static const double tol = eps * 32;
    return e != e ||
           fabs(fa - fb) < tol || fabs(fa - fd) < tol || fabs(fa - fe) < tol ||
           fabs(fb - fd) < tol || fabs(fb - fe) < tol || fabs(fd - fe) < tol;
  }

  bool evalRebracketAndCheckEnd() {
    {
      double closestAllowedCNearA = a + 2 * eps * fabs(a);
      if (c < closestAllowedCNearA) c = closestAllowedCNearA;
      else {
        double closestAllowedCNearB = b - 2 * eps * fabs(b);
        if (c > closestAllowedCNearB) c = closestAllowedCNearB;
      }
    }
    if (callFunctionAndCheckEnd(c, fc)) return true;
    e = d; fe = fd;
    if (dontBracketARoot(fa, fc)) { d = a; fd = fa; a = c; fa = fc; }
    else { d = b; fd = fb; b = c; fb = fc; }
    return checkBracketForEnd();
  }
};

extern "C" double toms748Custom(Toms748InputFunction function, void *otherInput,
                                double intervalStart, double intervalEnd,
                                Toms748ResultStatus *resultStatus,
                                double absoluteTolerance,
                                double relativeTolerance, int maximumIterations,
                                int interpolationsPerIteration) {
  int errorCode = TOMS748_NO_ERROR;
  if (!isfinite(intervalStart)) errorCode |= TOMS748_INVALID_INTERVAL_START;
  if (!isfinite(intervalEnd))   errorCode |= TOMS748_INVALID_INTERVAL_END;
  if (intervalStart >= intervalEnd) errorCode |= TOMS748_INVALID_INTERVAL;
  if (AreClose::invalidTolerance(absoluteTolerance)) errorCode |= TOMS748_INVALID_ABSOLUTE_TOLERANCE;
  if (AreClose::invalidTolerance(relativeTolerance)) errorCode |= TOMS748_INVALID_RELATIVE_TOLERANCE;
  if (maximumIterations < 1) errorCode |= TOMS748_INVALID_MAXIMUM_ITERATIONS;
  if (interpolationsPerIteration < 1) errorCode |= TOMS748_INVALID_INTERPOLATIONS_PER_ITERATION;
  if (errorCode != TOMS748_NO_ERROR) {
    if (resultStatus) resultStatus->errorCode = errorCode;
    return NAN;
  }
  return Toms748Solver(function, otherInput, intervalStart, intervalEnd,
                       resultStatus, absoluteTolerance, relativeTolerance,
                       maximumIterations, interpolationsPerIteration).result();
}

extern "C" double toms748(Toms748InputFunction function, void *otherInput,
                          double intervalStart, double intervalEnd,
                          Toms748ResultStatus *resultStatus) {
  return toms748Custom(function, otherInput, intervalStart, intervalEnd,
                       resultStatus, defaultAbsoluteTolerance,
                       defaultRelativeTolerance, defaultMaximumIterations,
                       defaultInterpolationsPerIteration);
}
