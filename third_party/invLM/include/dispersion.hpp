#pragma once

#include "secfunc.hpp"

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace fwd {

// Evidence from the last global enumeration, not a completeness certificate.
struct RootSearchDiagnostics {
  int roots_found = 0;
  bool numerical_issue = false;
};

class Dispersion {
public:
  Dispersion(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh,
             DispMethod method = DispMethod::DUNKIN);
  ~Dispersion();

  std::vector<double> search(double f, int num_mode) const;
  double search_mode(double f, int mode, bool* from_fallback = nullptr) const;
  bool search_mode_near(double f, int mode, double guess, double& root) const;
  RootSearchDiagnostics search_diagnostics() const { return search_diagnostics_; }

private:
  double evaluate_secular(double f, double c) const;
  mutable RootSearchDiagnostics search_diagnostics_;
  double approx(double f, double c) const;
  const std::vector<double>& get_samples(double f) const;
  std::vector<std::pair<double, double>> find_coarse_intv(double f,
                                                           int num_mode) const;
  void locate_extremum(double f,
                       std::vector<std::pair<double, double>>& pts) const;
  std::vector<std::pair<double, double>> certify_love_intervals(double f, int num_mode,
      std::vector<std::pair<double, double>> intervals) const;
  double evaluate_rayleigh_velocity();

  const double ednn_ = 0.50;
  const int nfine_ = 2;
  const double ctol_ = 1.0e-5;

  const int nbias_ = 2;
  const int niter_ext_ = 3;

  const int nl_;
  Eigen::ArrayXd thk_, vs_, vp_, inv_vs2_, inv_vp2_;
  const bool sh_;
  const int itop_;
  std::unique_ptr<SecularFunction> sf_;
  DispMethod method_;

  mutable std::vector<double> pred_cache_, fine_cache_;

  double vs0_, vp0_, vs_min_, vs_max_, vs_hf_, rayv_;
};

} // namespace fwd
