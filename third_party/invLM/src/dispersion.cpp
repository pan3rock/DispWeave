// DispWeave modification (2026-10-21): bracket auxetic solid Rayleigh roots.
// See third_party/invLM/PATCHES.md; remaining implementation is from invLM.
#include "dispersion.hpp"
#include "secfunc.hpp"
#include "toms748.h"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <functional>

using namespace Eigen;

namespace fwd {

namespace {
double calculate_newton_step(const double root, const double vp,
                             const double vs) {
  double p = pow(root, -1);
  double p2 = pow(root, -2);
  double vs_2 = pow(vs, -2);
  double vp_2 = pow(vp, -2);
  double xi = sqrt(p2 - vs_2);
  double eta = sqrt(p2 - vp_2);
  double func = pow(vs_2 - 2.0 * p2, 2) - 4.0 * xi * eta * p2;
  double deri = p2 * (8.0 * p * (vs_2 - 2 * p2) + 8.0 * p * xi * eta +
                      4.0 * p2 * p * (xi / eta + eta / xi));
  return func / deri;
}

double find_extremum(double x1, double x2, double x3, double f1, double f2,
                     double f3) {
  double B1 = (pow(x2, 2) - pow(x3, 2)) * f1;
  double B2 = (pow(x3, 2) - pow(x1, 2)) * f2;
  double B3 = (pow(x1, 2) - pow(x2, 2)) * f3;
  double C1 = (x2 - x3) * f1;
  double C2 = (x3 - x1) * f2;
  double C3 = (x1 - x2) * f3;
  double D = (x1 - x2) * (x2 - x3) * (x3 - x1);
  double b = (B1 + B2 + B3) / D;
  double a = -(C1 + C2 + C3) / D;
  return -b / (2.0 * a);
}

inline void insert_sorted(std::vector<double>& vec, double val) {
    auto it = std::lower_bound(vec.begin(), vec.end(), val);
    if (it == vec.end() || *it != val) {
        vec.insert(it, val);
    }
}
} // namespace

Dispersion::Dispersion(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh,
                       DispMethod method)
    : nl_(model.rows()),
      thk_(model.col(1).tail(nl_ - 1) - model.col(1).head(nl_ - 1)),
      vs_(model.col(3)), vp_(model.col(4)),
      inv_vs2_(1.0 / vs_.square()), inv_vp2_(1.0 / vp_.square()),
      sh_(sh),
      itop_(model(0, 3) == 0 ? 1 : 0),
      sf_(SecularFunction::create(model, sh, method)),
      method_(method) {
  vs0_ = vs_(itop_);
  vp0_ = vp_(itop_);
  vs_min_ = vs_.tail(nl_ - itop_).minCoeff();
  if (itop_ == 1 && !sh_) {
    vs_min_ = std::min(vs_min_, vp_(0));
  }
  vs_max_ = vs_.maxCoeff();
  vs_hf_ = vs_(nl_ - 1);
  rayv_ = evaluate_rayleigh_velocity();
}

Dispersion::~Dispersion() = default;

double Dispersion::evaluate_secular(double f, double c) const {
  const double value = sf_->evaluate(f, c);
  if (!std::isfinite(value)) search_diagnostics_.numerical_issue = true;
  return value;
}

double Dispersion::evaluate_rayleigh_velocity() {
  double root = 0.8 * vs0_;
  int count = 0;
  while (true) {
    double update_step = calculate_newton_step(root, vp0_, vs0_);
    root -= update_step;
    if (count > 10 || std::abs(update_step) < ctol_)
      break;
    ++count;
  }
  return root;
}

double Dispersion::approx(double f, double c) const {
  double sum = 0.0;
  double c_2 = 1.0 / (c * c);
  for (int i = itop_; i < nl_ - 1; ++i) {
    double diff = inv_vs2_(i) - c_2;
    if (diff > 0.0) {
      sum += std::sqrt(diff) * thk_(i);
    }
  }
  if (!sh_) {
    for (int i = 0; i < nl_ - 1; ++i) {
      double diff = inv_vp2_(i) - c_2;
      if (diff > 0.0) {
        sum += std::sqrt(diff) * thk_(i);
      }
    }
  }
  sum *= 2.0 * f;
  return sum;
}

const std::vector<double>& Dispersion::get_samples(double f) const {
  pred_cache_.clear();
  fine_cache_.clear();
  auto& pred = pred_cache_;
  auto& fine = fine_cache_;

  // Equal solid velocities still admit a Rayleigh/Scholte surface mode below
  // Vs (notably a fixed water layer over a single solid halfspace).
  if (!(vs_hf_ > 0) || vs_hf_ < vs_min_) {
    return pred;
  }

  double a_hf = approx(f, vs_hf_);
  if (!std::isfinite(a_hf) || a_hf < 0.0) a_hf = 0.0;
  int nmax = static_cast<int>(std::floor(a_hf)) + 1;
  constexpr int NMAX_CAP = 2000;
  if (nmax > NMAX_CAP) { nmax = NMAX_CAP; search_diagnostics_.numerical_issue = true; }
  if (nmax < 1) nmax = 1;
  double dc = (vs_hf_ - vs_min_) / nmax;
  double c1 = vs_min_;
  double e1 = approx(f, c1);
  constexpr int MAX_OUTER = 4000;
  int outer = 0;
  while (c1 < vs_hf_ && outer++ < MAX_OUTER) {
    double c2 = c1 + dc;
    double e2 = approx(f, c2);
    int inner = 0;
    while (std::abs(e2 - e1) > ednn_) {
      c2 = c1 + (c2 - c1) * 0.618;
      e2 = approx(f, c2);
      if (++inner > 50) { search_diagnostics_.numerical_issue = true; break; }
    }
    c1 = c2;
    e1 = e2;
    if (c2 < vs_hf_) {
      pred.push_back(c2);
    }
  }

  if (outer > MAX_OUTER) search_diagnostics_.numerical_issue = true;
  // Positive bulk modulus permits negative Poisson ratios, whose Rayleigh
  // speed can be below 0.8 Vs. Start below the elastic halfspace lower
  // limit (~0.689 Vs), rather than relying on a Newton sample at the root.
  insert_sorted(pred, (sh_ ? 0.8 : 0.5) * vs_min_);
  insert_sorted(pred, vs_hf_ - ctol_);

  int seg = 1 << nfine_;
  fine.reserve((pred.size() - 1) * seg + 1);
  for (size_t i = 0; i < pred.size() - 1; ++i) {
    fine.push_back(pred[i]);
    double step = (pred[i + 1] - pred[i]) / seg;
    for (int j = 1; j < seg; ++j) {
      fine.push_back(pred[i] + j * step);
    }
  }
  fine.push_back(pred.back());

  for (double eps = ctol_ * 0.1; eps >= 1e-10; eps *= 0.1) {
    const double x = vs_hf_ - eps;
    if (x > fine.back() + std::numeric_limits<double>::epsilon()) {
      fine.push_back(x);
    }
  }

  insert_sorted(fine, vs_min_ - ctol_ * 10);
  insert_sorted(fine, vs_min_ + ctol_ * 10);

  if (!sh_) {
    insert_sorted(fine, rayv_);
  }

  return fine;
}

std::vector<std::pair<double, double>>
Dispersion::find_coarse_intv(double f, int num_mode) const {
  const auto& samples = get_samples(f);
  if (samples.empty()) return {};

  std::vector<std::pair<double, double>> pts;
  pts.reserve(samples.size() + 64);
  pts.push_back({samples[0], evaluate_secular(f, samples[0])});

  int count_root = 0;
  for (size_t i = 1; i < samples.size(); ++i) {
    double x = samples[i];
    double y = evaluate_secular(f, x);
    pts.push_back({x, y});
    if (std::isfinite(y) && std::isfinite(pts[i-1].second) &&
        (y == 0 || (pts[i-1].second != 0 &&
                    std::signbit(pts[i-1].second) != std::signbit(y))))
      ++count_root;
    if (count_root >= num_mode + nbias_)
      break;
  }

  if (pts.size() >= 3) {
    locate_extremum(f, pts);
  }

  std::sort(pts.begin(), pts.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  pts.erase(std::unique(pts.begin(),pts.end(),
                       [](const auto& a,const auto& b){return a.first==b.first;}),pts.end());

  std::vector<std::pair<double, double>> find_intv;
  for (size_t i = 0; i < pts.size(); ++i) {
    // An exactly-zero sample is a root too. In a homogeneous cap the inserted
    // surface Rayleigh speed can land exactly on it after floating-point rounding.
    // Strict sign-change-only enumeration silently discarded that mode.
    if (pts[i].second == 0) {
      find_intv.push_back({pts[i].first, pts[i].first});
    } else if (i + 1 < pts.size() && std::isfinite(pts[i].second) &&
               std::isfinite(pts[i+1].second) && pts[i+1].second != 0 &&
               std::signbit(pts[i].second) != std::signbit(pts[i+1].second)) {
      find_intv.push_back({pts[i].first, pts[i + 1].first});
    }
  }

  if (sh_) return certify_love_intervals(f,num_mode,std::move(find_intv));
  if (int(find_intv.size()) > num_mode) {
    return std::vector<std::pair<double, double>>(find_intv.data(),
                                                   find_intv.data() + num_mode);
  }
  return find_intv;
}

std::vector<std::pair<double,double>> Dispersion::certify_love_intervals(
    double f,int num_mode,std::vector<std::pair<double,double>> intervals) const {
  if(num_mode<=0) return {};
  // Keep this repair independent of the halfspace-endpoint margin experiment.
  const double lo=vs_min_, hi=vs_hf_-1e-10;
  if(!(hi>lo)) return {};
  auto count=[&](double c){return sf_->love_root_count(f,c);};
  if(int(intervals.size())>num_mode) intervals.resize(num_mode);
  bool valid=true;
  for(int i=0;valid && i<int(intervals.size());++i) {
    valid=intervals[i].first>=lo && intervals[i].second<=hi
        && count(intervals[i].first)==i && count(intervals[i].second)==i+1;
  }
  // A certified requested prefix needs no full-spectrum count at the upper end.
  if(valid && int(intervals.size())==num_mode) return intervals;
  const int total=count(hi);
  if(total<0) {search_diagnostics_.numerical_issue=true; return {};}
  const int wanted=std::min(num_mode,total);
  if(valid && int(intervals.size())==wanted) return intervals;

  std::vector<std::pair<double,double>> certified;
  bool failed=false;
  std::function<void(double,double,int,int,int)> split;
  split=[&](double a,double b,int na,int nb,int depth) {
    if(failed || na>=wanted || nb==na) return;
    if(nb==na+1) {
      const double fa=evaluate_secular(f,a), fb=evaluate_secular(f,b);
      if(std::isfinite(fa) && std::isfinite(fb)
          && (fa==0 || fb==0 || std::signbit(fa)!=std::signbit(fb))) {
        certified.emplace_back(a,b); return;
      }
    }
    const double mid=a+(b-a)*.5;
    if(depth>=80 || mid==a || mid==b) {failed=true;return;}
    const int nm=count(mid);
    if(nm<na || nm>nb) {failed=true;return;}
    split(a,mid,na,nm,depth+1);
    split(mid,b,nm,nb,depth+1);
  };
  const int lower_count=count(lo);
  if(lower_count!=0) failed=true;
  else split(lo,hi,0,total,0);
  if(failed || int(certified.size())!=wanted) {
    search_diagnostics_.numerical_issue=true;
    return {};
  }
  return certified;
}

std::vector<double> Dispersion::search(double f, int num_mode) const {
  search_diagnostics_ = {};
  auto func = [&](double c) { return evaluate_secular(f, c); };
  auto find_intv = find_coarse_intv(f, num_mode);

  const double dedup_tol = 1.0e-7;

  std::vector<double> find;
  for (auto it = find_intv.cbegin(); it != find_intv.end(); ++it) {
    double c1 = it->first;
    double c2 = it->second;
    Toms748ResultStatus status{};
    double root = c1 == c2 ? c1 : toms748(func, c1, c2, sh_ ? &status : nullptr, 1e-12, 1e-14, 100, 2);
    if(sh_ && (status.errorCode!=TOMS748_NO_ERROR || !std::isfinite(root)
        || (!find.empty() && root<=find.back()))) {
      // Counted intervals each own a different mode. Never merge a close pair
      // using the legacy 1e-7 dedup threshold and silently renumber later roots.
      search_diagnostics_.numerical_issue=true;
      return {};
    }
    if (!std::isfinite(root)) search_diagnostics_.numerical_issue = true;
    if (!std::isnan(root))
      if (find.empty() || sh_ || std::abs(root - find.back()) > dedup_tol) {
        find.push_back(root);
      }
  }

  search_diagnostics_.roots_found = static_cast<int>(find.size());
  if (int(find.size()) > num_mode) {
    return std::vector<double>(find.data(), find.data() + num_mode);
  }
  return find;
}

double Dispersion::search_mode(double f, int mode,
                                bool* from_fallback) const {
  if (from_fallback) *from_fallback = false;
  auto cs = search(f, mode + 1);
  if (static_cast<int>(cs.size()) >= mode + 1) {
    return cs[mode];
  }

  if (static_cast<int>(cs.size()) == mode && mode < nl_) {
    if (from_fallback) *from_fallback = true;
    return vs_hf_ - ctol_;
  }

  return std::numeric_limits<double>::quiet_NaN();
}

bool Dispersion::search_mode_near(double f, int mode, double guess,
                                  double& root) const {
  if (!std::isfinite(guess)) return false;

  // Proximity alone cannot establish a mode's ordinal number. In particular,
  // rejected LM trials can leave hints on a different branch. Verify against
  // the ordered global roots and return the canonical value so the objective
  // is independent of the history of trial evaluations.
  auto accept = [&](double candidate, double width) {
    if (!std::isfinite(candidate) || std::abs(candidate - guess) > width)
      return false;
    const auto roots = search(f, mode + 1);
    if (mode < 0 || mode >= static_cast<int>(roots.size())
        || std::abs(candidate - roots[mode]) > 1e-7)
      return false;
    root = roots[mode];
    return true;
  };

  const double lo = vs_min_ + ctol_ * 10.0;
  const double hi = vs_hf_ - ctol_;
  if (!(hi > lo)) return false;

  double center = std::clamp(guess, lo, hi);
  double span = hi - lo;
  double width = std::max(10.0 * ctol_, 0.002 * span);

  const double y0 = sf_->evaluate(f, center);
  if (!std::isfinite(y0)) return false;

  auto func = [&](double c) { return sf_->evaluate(f, c); };
  double left = std::max(lo, center - width);
  double right = std::min(hi, center + width);
  if (!(right > left)) return false;

  double yl = func(left);
  double yr = func(right);
  if (std::isfinite(yl) && yl * y0 < 0.0) {
    root = toms748(func, left, center);
    return accept(root, width);
  }
  if (std::isfinite(yr) && y0 * yr < 0.0) {
    root = toms748(func, center, right);
    return accept(root, width);
  }
  if (std::isfinite(yl) && std::isfinite(yr) && yl * yr < 0.0) {
    root = toms748(func, left, right);
    return accept(root, width);
  }
  return false;
}

void Dispersion::locate_extremum(double f,
                                 std::vector<std::pair<double, double>>& pts) const {
  std::vector<std::pair<double, double>> new_pts;

  for (int it = 0; it < niter_ext_; ++it) {
    new_pts.clear();
    for (size_t i = 0; i < pts.size() - 2; ++i) {
      double x1 = pts[i].first;
      double x2 = pts[i + 1].first;
      double x3 = pts[i + 2].first;
      double f1 = pts[i].second;
      double f2 = pts[i + 1].second;
      double f3 = pts[i + 2].second;
      if (!((f2 > f1 && f2 > f3) || (f2 < f1 && f2 < f3)))
        continue;
      double xe_i = find_extremum(x1, x2, x3, f1, f2, f3);
      if (xe_i < x3 && xe_i > x1) {
        new_pts.push_back({xe_i, evaluate_secular(f, xe_i)});
      }
    }
    // Even one extremum can reveal a pair of roots between same-sign samples.
    // Do not discard the last pass or require three new extrema: the next
    // pass operates on the entire accumulated sample set, not just new_pts.
    if (new_pts.empty())
      break;

    pts.insert(pts.end(), new_pts.begin(), new_pts.end());
    std::sort(pts.begin(), pts.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }
}

} // namespace fwd
