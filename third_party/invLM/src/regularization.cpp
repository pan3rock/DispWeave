#include "regularization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace inversion {
namespace {

class TermAdapter {
public:
    virtual ~TermAdapter() = default;
    virtual int rows() const = 0;
    virtual const std::string& name() const = 0;
    virtual double weight() const = 0;
    virtual void evaluate(const Eigen::VectorXd& vs,
                          Eigen::VectorXd& residuals,
                          Eigen::MatrixXd& jacobian,
                          bool with_jacobian) const = 0;
};

class ReferenceDampingAdapter final : public TermAdapter {
public:
    ReferenceDampingAdapter(Eigen::VectorXd reference, double lambda,int fixed)
        : reference_(std::move(reference)), lambda_(lambda),
          scale_(std::sqrt(lambda)),fixed_(fixed) {}

    int rows() const override { return static_cast<int>(reference_.size()); }
    const std::string& name() const override {
        static const std::string value = "reference_damping";
        return value;
    }
    double weight() const override { return lambda_; }

    void evaluate(const Eigen::VectorXd& vs,
                  Eigen::VectorXd& residuals,
                  Eigen::MatrixXd& jacobian,
                  bool with_jacobian) const override {
        residuals = scale_ * (vs - reference_);
        residuals.head(fixed_).setZero();
        if (with_jacobian) {
            jacobian.diagonal().setConstant(scale_);
            jacobian.topRows(fixed_).setZero();
        }
    }

private:
    Eigen::VectorXd reference_;
    double lambda_;
    double scale_;
    int fixed_;
};

class FirstDifferenceAdapter final : public TermAdapter {
public:
    FirstDifferenceAdapter(Eigen::VectorXd target,
                           const Eigen::VectorXd& depth,
                           const Eigen::VectorXd& interface_weights,
                           double lambda, std::string adapter_name)
        : target_(std::move(target)), lambda_(lambda),
          name_(std::move(adapter_name)), scales_(depth.size() - 1) {
        for (int row = 0; row < scales_.size(); ++row) {
            const double dz = depth(row + 1) - depth(row);
            scales_(row) = std::sqrt(
                lambda_ * interface_weights(row) / dz);
        }
    }

    int rows() const override { return static_cast<int>(scales_.size()); }
    const std::string& name() const override { return name_; }
    double weight() const override { return lambda_; }

    void evaluate(const Eigen::VectorXd& vs,
                  Eigen::VectorXd& residuals,
                  Eigen::MatrixXd& jacobian,
                  bool with_jacobian) const override {
        const Eigen::VectorXd values = vs - target_;
        for (int row = 0; row < rows(); ++row) {
            residuals(row) = scales_(row)
                * (values(row + 1) - values(row));
            if (with_jacobian) {
                jacobian(row, row) = -scales_(row);
                jacobian(row, row + 1) = scales_(row);
            }
        }
    }

private:
    Eigen::VectorXd target_;
    double lambda_;
    std::string name_;
    Eigen::VectorXd scales_;
};

class TotalVariationAdapter final : public TermAdapter {
public:
    TotalVariationAdapter(int parameter_count,
                          Eigen::VectorXd interface_weights,
                          double lambda, double epsilon)
        : rows_(parameter_count - 1),
          interface_weights_(std::move(interface_weights)),
          lambda_(lambda), epsilon_(epsilon) {}

    int rows() const override { return rows_; }
    const std::string& name() const override {
        static const std::string value = "total_variation";
        return value;
    }
    double weight() const override { return lambda_; }

    void evaluate(const Eigen::VectorXd& vs,
                  Eigen::VectorXd& residuals,
                  Eigen::MatrixXd& jacobian,
                  bool with_jacobian) const override {
        for (int row = 0; row < rows_; ++row) {
            const double difference = vs(row + 1) - vs(row);
            const double q = std::sqrt(
                difference * difference + epsilon_ * epsilon_);
            const double scale = std::sqrt(
                2.0 * lambda_ * interface_weights_(row)
                / (q + epsilon_));
            residuals(row) = scale * difference;
            if (with_jacobian) {
                const double derivative = scale * (
                    1.0 - difference * difference
                    / (2.0 * q * (q + epsilon_)));
                jacobian(row, row) = -derivative;
                jacobian(row, row + 1) = derivative;
            }
        }
    }

private:
    int rows_;
    Eigen::VectorXd interface_weights_;
    double lambda_;
    double epsilon_;
};

} // namespace

struct Regularization::Implementation {
    Eigen::VectorXd ref_vs;
    Eigen::VectorXd interface_weights;
    std::vector<std::unique_ptr<TermAdapter>> terms;
    int residual_count = 0;
    int smoothness_rows = 0;
};

Regularization::Regularization(
    const Eigen::VectorXd& depth, const Eigen::VectorXd& ref_vs,
    const RegularizationSettings& settings)
    : implementation_(std::make_unique<Implementation>()) {
    auto& impl = *implementation_;
    impl.ref_vs = ref_vs;
    if (depth.size() != ref_vs.size())
        throw std::invalid_argument(
            "regularization depth and reference Vs sizes differ");
    if (ref_vs.size() == 0)
        throw std::invalid_argument("regularization requires at least one layer");
    if (!depth.allFinite())
        throw std::invalid_argument("regularization depths must be finite");
    if (!ref_vs.allFinite())
        throw std::invalid_argument("regularization reference Vs must be finite");
    if (!std::isfinite(settings.lambda_ref) || settings.lambda_ref < 0.0
        || !std::isfinite(settings.lambda_smooth)
        || settings.lambda_smooth < 0.0) {
        throw std::invalid_argument(
            "regularization weights must be non-negative and finite");
    }
    if (!std::isfinite(settings.interface_scale)
        || settings.interface_scale <= 0.0) {
        throw std::invalid_argument(
            "regularization interface scale must be positive and finite");
    }
    if (!std::isfinite(settings.tv_epsilon) || settings.tv_epsilon <= 0.0)
        throw std::invalid_argument(
            "regularization TV epsilon must be positive and finite");
    if (settings.type != "perturbation_smooth"
        && settings.type != "vs_smooth" && settings.type != "tv"
        && settings.type != "reference_damping" && settings.type != "none") {
        throw std::invalid_argument(
            "unsupported regularization type: " + settings.type);
    }

    if ((settings.type == "reference_damping" && settings.lambda_smooth != 0.0)
        || (settings.type == "none" && (settings.lambda_ref != 0.0 || settings.lambda_smooth != 0.0)))
        throw std::invalid_argument("regularization type conflicts with active weights");

    impl.interface_weights = Eigen::VectorXd::Ones(
        std::max<Eigen::Index>(0, ref_vs.size() - 1));
    for (int row = 0; row < impl.interface_weights.size(); ++row) {
        const double dz = depth(row + 1) - depth(row);
        if (!std::isfinite(dz) || dz <= 0.0)
            throw std::invalid_argument(
                "regularization depths must be strictly increasing");
        if (settings.interface_adaptive) {
            const double relative_jump =
                (ref_vs(row + 1) - ref_vs(row)) / settings.interface_scale;
            impl.interface_weights(row) =
                1.0 / (1.0 + relative_jump * relative_jump);
        }
    }
    if(settings.fixed_surface_layers<0 || settings.fixed_surface_layers>=ref_vs.size())
        throw std::invalid_argument("Invalid fixed surface-layer count for regularization");
    impl.interface_weights.head(settings.fixed_surface_layers).setZero();

    impl.terms.push_back(std::make_unique<ReferenceDampingAdapter>(
        ref_vs, settings.lambda_ref,settings.fixed_surface_layers));
    impl.residual_count = static_cast<int>(ref_vs.size());
    if (settings.lambda_smooth > 0.0 && ref_vs.size() > 1) {
        impl.smoothness_rows = static_cast<int>(ref_vs.size()) - 1;
        if (settings.type == "tv") {
            impl.terms.push_back(std::make_unique<TotalVariationAdapter>(
                static_cast<int>(ref_vs.size()), impl.interface_weights,
                settings.lambda_smooth, settings.tv_epsilon));
        } else {
            const Eigen::VectorXd target = settings.type == "vs_smooth"
                ? Eigen::VectorXd::Zero(ref_vs.size()) : ref_vs;
            impl.terms.push_back(std::make_unique<FirstDifferenceAdapter>(
                target, depth, impl.interface_weights,
                settings.lambda_smooth, settings.type));
        }
        impl.residual_count += impl.smoothness_rows;
    }
}

Regularization::Regularization(const Eigen::VectorXd& depth,
                               const Eigen::VectorXd& ref_vs,
                               double lambda_ref,
                               double lambda_smooth)
    : Regularization(
          depth, ref_vs,
          RegularizationSettings{
              "perturbation_smooth", lambda_ref, lambda_smooth,
              false, 0.1, 1e-3}) {}

Regularization::~Regularization() = default;
Regularization::Regularization(Regularization&&) noexcept = default;
Regularization& Regularization::operator=(Regularization&&) noexcept = default;

int Regularization::num_parameters() const {
    return static_cast<int>(implementation_->ref_vs.size());
}

int Regularization::num_residuals() const {
    return implementation_->residual_count;
}

int Regularization::num_reference_residuals() const {
    return num_parameters();
}

int Regularization::num_smoothness_residuals() const {
    return implementation_->smoothness_rows;
}

bool Regularization::evaluate(const Eigen::VectorXd& vs,
                              bool with_jacobian,
                              RegularizationEvaluation& result) const {
    result = RegularizationEvaluation{};
    if (vs.size() != num_parameters() || !vs.allFinite())
        return false;

    result.residuals = Eigen::VectorXd::Zero(num_residuals());
    result.jacobian = with_jacobian
        ? Eigen::MatrixXd::Zero(num_residuals(), num_parameters())
        : Eigen::MatrixXd(0, 0);
    result.interface_weights = implementation_->interface_weights;
    int offset = 0;
    for (const auto& term : implementation_->terms) {
        const int rows = term->rows();
        Eigen::VectorXd term_residuals = Eigen::VectorXd::Zero(rows);
        Eigen::MatrixXd term_jacobian = with_jacobian
            ? Eigen::MatrixXd::Zero(rows, num_parameters())
            : Eigen::MatrixXd(0, 0);
        term->evaluate(vs, term_residuals, term_jacobian, with_jacobian);
        result.residuals.segment(offset, rows) = term_residuals;
        if (with_jacobian)
            result.jacobian.middleRows(offset, rows) = term_jacobian;
        const double cost = 0.5 * term_residuals.squaredNorm();
        result.term_names.push_back(term->name());
        result.term_weights.push_back(term->weight());
        result.term_costs.push_back(cost);
        if (offset == 0)
            result.cost_ref = cost;
        else
            result.cost_smooth += cost;
        offset += rows;
    }
    return true;
}

} // namespace inversion
