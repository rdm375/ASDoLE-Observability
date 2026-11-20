// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/differentiation.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace asdole {

using VectorAction = std::function<Vector(std::span<const double>)>;

using InitialParameterAction = std::function<Vector(std::span<const double>)>;
using ParameterStepAction = std::function<Vector(
    std::size_t, std::span<const double>, double, std::span<const double>)>;
using ObservationParameterAction = std::function<Vector(
    std::size_t, std::span<const double>, std::span<const double>)>;

// Linear map from state perturbations to observation perturbations. Supplying
// actions instead of a matrix lets callers later use sparse or matrix-free
// sensor operators.
struct LinearObservationOperator {
    std::size_t state_dimension{};
    std::size_t observation_dimension{};
    VectorAction apply;
    VectorAction transpose_apply;

    [[nodiscard]] Vector operator()(std::span<const double> state) const {
        if (state.size() != state_dimension || !apply)
            throw std::invalid_argument("observation operator input dimension mismatch");
        Vector value = apply(state);
        if (value.size() != observation_dimension)
            throw std::invalid_argument("observation operator returned wrong dimension");
        return value;
    }

    [[nodiscard]] Vector transpose(std::span<const double> observation) const {
        if (observation.size() != observation_dimension || !transpose_apply)
            throw std::invalid_argument("transposed observation input dimension mismatch");
        Vector value = transpose_apply(observation);
        if (value.size() != state_dimension)
            throw std::invalid_argument("transposed observation operator returned wrong dimension");
        return value;
    }
};

inline LinearObservationOperator dense_observation_operator(DenseMatrix matrix) {
    const auto rows = matrix.rows();
    const auto cols = matrix.cols();
    const auto shared_matrix = std::make_shared<const DenseMatrix>(std::move(matrix));
    return {cols, rows,
        [shared_matrix](std::span<const double> x) { return detail::matvec(*shared_matrix, x); },
        [shared_matrix](std::span<const double> x) { return detail::transpose_matvec(*shared_matrix, x); }};
}

// Reusable linearization of one stored discrete trajectory. step_sizes[k]
// maps states[k] to states[k+1]. Only TLM and adjoint actions are formed; no
// trajectory-wide Jacobian or propagator is assembled.
class DiscreteTrajectoryLinearization {
public:
    DiscreteTrajectoryLinearization(const Stepper& stepper,
                                    HessianFunction hessian,
                                    const std::vector<Vector>& states,
                                    std::vector<double> step_sizes)
        : stepper_(stepper), hessian_(std::move(hessian)), states_(states),
          step_sizes_(std::move(step_sizes)) {
        if (!hessian_ || states_.empty() || step_sizes_.size() + 1 != states_.size())
            throw std::invalid_argument("trajectory linearization needs states, Hessians, and matching steps");
        const auto n = states_.front().size();
        if (n == 0) throw std::invalid_argument("trajectory state dimension must be positive");
        for (const auto& state : states_)
            if (state.size() != n) throw std::invalid_argument("trajectory state dimensions differ");
        for (double h : step_sizes_)
            if (!(h > 0.0) || !std::isfinite(h)) throw std::invalid_argument("trajectory step sizes must be finite and positive");
    }

    [[nodiscard]] std::size_t state_dimension() const noexcept { return states_.front().size(); }
    [[nodiscard]] std::size_t node_count() const noexcept { return states_.size(); }
    [[nodiscard]] std::size_t step_count() const noexcept { return step_sizes_.size(); }
    [[nodiscard]] const std::vector<Vector>& states() const noexcept { return states_; }
    [[nodiscard]] double step_size(std::size_t step_index) const {
        if (step_index >= step_count()) throw std::out_of_range("trajectory step index out of range");
        return step_sizes_[step_index];
    }

    [[nodiscard]] Vector apply(std::span<const double> initial_direction,
                               std::size_t end_node) const {
        check_node(end_node);
        if (initial_direction.size() != state_dimension())
            throw std::invalid_argument("initial TLM direction has wrong dimension");
        Vector value(initial_direction.begin(), initial_direction.end());
        for (std::size_t k = 0; k < end_node; ++k)
            value = apply_step(value, k);
        return value;
    }

    [[nodiscard]] Vector apply_step(std::span<const double> direction,
                                    std::size_t step_index) const {
        if (step_index >= step_count()) throw std::out_of_range("trajectory step index out of range");
        if (direction.size() != state_dimension())
            throw std::invalid_argument("TLM direction has wrong dimension");
        return tangent_step(stepper_, hessian_, states_[step_index], step_sizes_[step_index], direction);
    }

    [[nodiscard]] Vector transpose_apply(std::span<const double> node_cotangent,
                                         std::size_t node) const {
        return pullback_to_node(node_cotangent, 0, node);
    }

    // Pull a covector at end_node backward to start_node, with
    // 0 <= start_node <= end_node.
    [[nodiscard]] Vector pullback_to_node(std::span<const double> end_cotangent,
                                          std::size_t start_node,
                                          std::size_t end_node) const {
        check_node(start_node);
        check_node(end_node);
        if (start_node > end_node) throw std::invalid_argument("pullback start node exceeds end node");
        if (end_cotangent.size() != state_dimension())
            throw std::invalid_argument("node cotangent has wrong dimension");
        Vector value(end_cotangent.begin(), end_cotangent.end());
        for (std::size_t k = end_node; k > start_node; --k)
            value = adjoint_step(stepper_, hessian_, states_[k - 1], step_sizes_[k - 1], value);
        return value;
    }

    [[nodiscard]] Vector pullback(std::span<const Vector> node_cotangents) const {
        if (node_cotangents.size() != node_count())
            throw std::invalid_argument("node cotangent history has wrong length");
        Vector value(node_cotangents.back());
        if (value.size() != state_dimension()) throw std::invalid_argument("node cotangent has wrong dimension");
        for (std::size_t reverse = step_count(); reverse > 0; --reverse) {
            const std::size_t k = reverse - 1;
            if (node_cotangents[k].size() != state_dimension())
                throw std::invalid_argument("node cotangent has wrong dimension");
            value = adjoint_step(stepper_, hessian_, states_[k], step_sizes_[k], value);
            for (std::size_t i = 0; i < value.size(); ++i) value[i] += node_cotangents[k][i];
        }
        return value;
    }

private:
    void check_node(std::size_t node) const {
        if (node >= node_count()) throw std::out_of_range("trajectory node index out of range");
    }

    const Stepper& stepper_;
    HessianFunction hessian_;
    const std::vector<Vector>& states_;
    std::vector<double> step_sizes_;
};

// One observation block for a parameter-to-observation sensitivity map.
// direct_parameter_apply/transpose_apply are optional terms for observations
// that depend explicitly on the parameters, in addition to depending on the
// simulated state.
struct ParameterObservationSample {
    std::size_t node{};
    LinearObservationOperator observation;
    double quadrature_weight{1.0};
    VectorAction precision_apply;
    ObservationParameterAction direct_parameter_apply;
    ObservationParameterAction direct_parameter_transpose_apply;
};

// Matrix-free derivative of stacked observations with respect to a parameter
// vector. Callbacks provide D_p y_0, the direct D_p Phi_k contribution for
// each ASDoLE step, and (when needed) direct parameter dependence of each
// observation. The trajectory supplies the state TLM and discrete adjoint.
class ParameterObservationSensitivityOperator {
public:
    ParameterObservationSensitivityOperator(
        const DiscreteTrajectoryLinearization& trajectory,
        std::size_t parameter_dimension,
        InitialParameterAction initial_parameter_apply,
        InitialParameterAction initial_parameter_transpose_apply,
        ParameterStepAction step_parameter_apply,
        ParameterStepAction step_parameter_transpose_apply,
        std::vector<ParameterObservationSample> samples)
        : trajectory_(trajectory), parameter_dimension_(parameter_dimension),
          initial_parameter_apply_(std::move(initial_parameter_apply)),
          initial_parameter_transpose_apply_(std::move(initial_parameter_transpose_apply)),
          step_parameter_apply_(std::move(step_parameter_apply)),
          step_parameter_transpose_apply_(std::move(step_parameter_transpose_apply)),
          samples_(std::move(samples)) {
        if (parameter_dimension_ == 0 || !initial_parameter_apply_
            || !initial_parameter_transpose_apply_ || !step_parameter_apply_
            || !step_parameter_transpose_apply_ || samples_.empty())
            throw std::invalid_argument("parameter sensitivity needs dimensions, derivative actions, and samples");
        for (const auto& sample : samples_) {
            if (sample.node >= trajectory_.node_count()
                || sample.observation.state_dimension != trajectory_.state_dimension()
                || sample.observation.observation_dimension == 0
                || !sample.observation.apply || !sample.observation.transpose_apply
                || !(sample.quadrature_weight > 0.0) || !std::isfinite(sample.quadrature_weight)
                || static_cast<bool>(sample.direct_parameter_apply)
                   != static_cast<bool>(sample.direct_parameter_transpose_apply))
                throw std::invalid_argument("invalid parameter observation sample");
        }
    }

    [[nodiscard]] std::size_t input_dimension() const noexcept { return parameter_dimension_; }
    [[nodiscard]] std::size_t parameter_dimension() const noexcept { return parameter_dimension_; }
    [[nodiscard]] std::size_t output_dimension() const noexcept {
        std::size_t size = 0;
        for (const auto& sample : samples_) size += sample.observation.observation_dimension;
        return size;
    }

    [[nodiscard]] Vector apply(std::span<const double> parameter_direction) const {
        if (parameter_direction.size() != parameter_dimension_)
            throw std::invalid_argument("parameter sensitivity input has wrong dimension");
        const auto& states = trajectory_.states();
        Vector state_direction = initial_parameter_apply_(parameter_direction);
        if (state_direction.size() != trajectory_.state_dimension())
            throw std::invalid_argument("initial parameter action returned wrong dimension");
        std::vector<Vector> state_directions(trajectory_.node_count());
        state_directions[0] = std::move(state_direction);
        for (std::size_t k = 0; k < trajectory_.step_count(); ++k) {
            state_direction = trajectory_.apply_step(state_directions[k], k);
            const Vector direct = step_parameter_apply_(k, states[k], trajectory_.step_size(k), parameter_direction);
            if (direct.size() != trajectory_.state_dimension())
                throw std::invalid_argument("step parameter action returned wrong dimension");
            for (std::size_t i = 0; i < state_direction.size(); ++i) state_direction[i] += direct[i];
            state_directions[k + 1] = std::move(state_direction);
        }

        Vector result;
        result.reserve(output_dimension());
        for (const auto& sample : samples_) {
            Vector observed = sample.observation(state_directions[sample.node]);
            if (sample.direct_parameter_apply) {
                const Vector direct = sample.direct_parameter_apply(
                    sample.node, states[sample.node], parameter_direction);
                if (direct.size() != observed.size())
                    throw std::invalid_argument("direct observation parameter action returned wrong dimension");
                for (std::size_t i = 0; i < observed.size(); ++i) observed[i] += direct[i];
            }
            result.insert(result.end(), observed.begin(), observed.end());
        }
        return result;
    }

    [[nodiscard]] Vector transpose_apply(std::span<const double> stacked_cotangent) const {
        if (stacked_cotangent.size() != output_dimension())
            throw std::invalid_argument("parameter observation cotangent has wrong dimension");
        const auto n = trajectory_.state_dimension();
        const auto& states = trajectory_.states();
        std::vector<Vector> node_loads(trajectory_.node_count(), Vector(n, 0.0));
        Vector parameter_gradient(parameter_dimension_, 0.0);
        std::size_t offset = 0;
        for (const auto& sample : samples_) {
            const auto m = sample.observation.observation_dimension;
            const auto block = stacked_cotangent.subspan(offset, m);
            const Vector state_load = sample.observation.transpose(block);
            for (std::size_t i = 0; i < n; ++i) node_loads[sample.node][i] += state_load[i];
            if (sample.direct_parameter_transpose_apply) {
                const Vector direct = sample.direct_parameter_transpose_apply(
                    sample.node, states[sample.node], block);
                add_parameter_vector(parameter_gradient, direct);
            }
            offset += m;
        }

        Vector costate = node_loads.back();
        for (std::size_t reverse = trajectory_.step_count(); reverse > 0; --reverse) {
            const std::size_t k = reverse - 1;
            const Vector direct = step_parameter_transpose_apply_(
                k, states[k], trajectory_.step_size(k), costate);
            add_parameter_vector(parameter_gradient, direct);
            costate = trajectory_.pullback_to_node(costate, k, k + 1);
            for (std::size_t i = 0; i < n; ++i) costate[i] += node_loads[k][i];
        }
        add_parameter_vector(parameter_gradient, initial_parameter_transpose_apply_(costate));
        return parameter_gradient;
    }

    [[nodiscard]] Vector weighted_transpose_apply(std::span<const double> stacked_value) const {
        if (stacked_value.size() != output_dimension())
            throw std::invalid_argument("weighted parameter observation vector has wrong dimension");
        Vector weighted(stacked_value.begin(), stacked_value.end());
        std::size_t offset = 0;
        for (const auto& sample : samples_) {
            const auto m = sample.observation.observation_dimension;
            const auto block = stacked_value.subspan(offset, m);
            Vector value = sample.precision_apply ? sample.precision_apply(block)
                                                  : Vector(block.begin(), block.end());
            if (value.size() != m) throw std::invalid_argument("precision operator returned wrong dimension");
            for (std::size_t i = 0; i < m; ++i) weighted[offset + i] = sample.quadrature_weight * value[i];
            offset += m;
        }
        return transpose_apply(weighted);
    }

    [[nodiscard]] Vector weighted_gram_action(std::span<const double> parameter_direction) const {
        return weighted_transpose_apply(apply(parameter_direction));
    }

private:
    void add_parameter_vector(Vector& target, std::span<const double> value) const {
        if (value.size() != parameter_dimension_)
            throw std::invalid_argument("parameter transpose action returned wrong dimension");
        for (std::size_t i = 0; i < parameter_dimension_; ++i) target[i] += value[i];
    }

    const DiscreteTrajectoryLinearization& trajectory_;
    std::size_t parameter_dimension_{};
    InitialParameterAction initial_parameter_apply_;
    InitialParameterAction initial_parameter_transpose_apply_;
    ParameterStepAction step_parameter_apply_;
    ParameterStepAction step_parameter_transpose_apply_;
    std::vector<ParameterObservationSample> samples_;
};

struct SensitivityDiagnostics {
    // Singular values of the observation Jacobian after applying the supplied
    // quadrature and precision weights, in descending order.
    std::vector<double> singular_values;
    std::size_t numerical_rank{};
    double condition_number{};
    double relative_rank_threshold{};
};

inline std::vector<double> symmetric_eigenvalues(DenseMatrix matrix) {
    if (matrix.rows() != matrix.cols()) throw std::invalid_argument("symmetric eigensolver requires a square matrix");
    const std::size_t n = matrix.rows();
    if (n == 0) return {};
    const std::size_t max_iterations = 100 * n * n;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        std::size_t p = 0, q = 0;
        double largest = 0.0;
        double scale = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            scale = std::max(scale, std::abs(matrix(i, i)));
            for (std::size_t j = i + 1; j < n; ++j) {
                if (std::abs(matrix(i, j)) > largest) {
                    largest = std::abs(matrix(i, j)); p = i; q = j;
                }
            }
        }
        if (largest <= 1.0e-14 * std::max(scale, std::numeric_limits<double>::min())) break;
        const double angle = 0.5 * std::atan2(2.0 * matrix(p, q), matrix(q, q) - matrix(p, p));
        const double c = std::cos(angle), s = std::sin(angle);
        const double app = matrix(p, p), aqq = matrix(q, q), apq = matrix(p, q);
        matrix(p, p) = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix(q, q) = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix(p, q) = matrix(q, p) = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            if (r == p || r == q) continue;
            const double arp = matrix(r, p), arq = matrix(r, q);
            matrix(r, p) = matrix(p, r) = c * arp - s * arq;
            matrix(r, q) = matrix(q, r) = s * arp + c * arq;
        }
    }
    std::vector<double> values(n);
    for (std::size_t i = 0; i < n; ++i) values[i] = matrix(i, i);
    std::sort(values.begin(), values.end(), std::greater<>());
    return values;
}

template<class SensitivityOperator>
inline SensitivityDiagnostics analyze_sensitivity(
    const SensitivityOperator& sensitivity,
    double relative_rank_threshold = 1.0e-8) {
    if (!(relative_rank_threshold > 0.0) || !std::isfinite(relative_rank_threshold))
        throw std::invalid_argument("relative rank threshold must be finite and positive");
    const std::size_t n = sensitivity.input_dimension();
    if (n == 0) throw std::invalid_argument("sensitivity input dimension must be positive");
    DenseMatrix gram(n, n);
    Vector basis(n, 0.0);
    for (std::size_t column = 0; column < n; ++column) {
        std::fill(basis.begin(), basis.end(), 0.0);
        basis[column] = 1.0;
        const Vector action = sensitivity.weighted_gram_action(basis);
        if (action.size() != n) throw std::invalid_argument("weighted Gram action returned wrong dimension");
        for (std::size_t row = 0; row < n; ++row) gram(row, column) = action[row];
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const double symmetric = 0.5 * (gram(i, j) + gram(j, i));
            gram(i, j) = gram(j, i) = symmetric;
        }
    const auto eigenvalues = symmetric_eigenvalues(std::move(gram));
    SensitivityDiagnostics result;
    result.relative_rank_threshold = relative_rank_threshold;
    result.singular_values.reserve(eigenvalues.size());
    for (double value : eigenvalues) result.singular_values.push_back(std::sqrt(std::max(0.0, value)));
    const double threshold = result.singular_values.front() * relative_rank_threshold;
    for (double sigma : result.singular_values)
        if (sigma > threshold) ++result.numerical_rank;
    result.condition_number = result.numerical_rank == n && result.singular_values.back() > 0.0
        ? result.singular_values.front() / result.singular_values.back()
        : std::numeric_limits<double>::infinity();
    return result;
}

struct ObservationSample {
    std::size_t node{};
    LinearObservationOperator observation;
    double quadrature_weight{1.0};
    // Applies the observation precision matrix R^{-1}; empty means identity.
    VectorAction precision_apply;
};

// Matrix-free Jacobian from the initial state to a stacked collection of
// observations. The output is concatenated in sample order. Its transpose
// action exactly reverses the same discrete TLM trajectory.
class ObservationSensitivityOperator {
public:
    ObservationSensitivityOperator(const DiscreteTrajectoryLinearization& trajectory,
                                   std::vector<ObservationSample> samples)
        : trajectory_(trajectory), samples_(std::move(samples)) {
        if (samples_.empty()) throw std::invalid_argument("at least one observation sample is required");
        for (const auto& sample : samples_) {
            if (sample.node >= trajectory_.node_count()
                || sample.observation.state_dimension != trajectory_.state_dimension()
                || sample.observation.observation_dimension == 0
                || !sample.observation.apply || !sample.observation.transpose_apply
                || !(sample.quadrature_weight > 0.0) || !std::isfinite(sample.quadrature_weight))
                throw std::invalid_argument("invalid observation sample");
        }
    }

    [[nodiscard]] std::size_t input_dimension() const noexcept { return trajectory_.state_dimension(); }
    [[nodiscard]] std::size_t output_dimension() const noexcept {
        std::size_t size = 0;
        for (const auto& sample : samples_) size += sample.observation.observation_dimension;
        return size;
    }

    [[nodiscard]] Vector apply(std::span<const double> initial_direction) const {
        if (initial_direction.size() != input_dimension())
            throw std::invalid_argument("observation sensitivity input has wrong dimension");
        std::vector<Vector> perturbations(trajectory_.node_count());
        perturbations[0].assign(initial_direction.begin(), initial_direction.end());
        for (std::size_t k = 0; k < trajectory_.step_count(); ++k)
            perturbations[k + 1] = trajectory_.apply_step(perturbations[k], k);
        Vector result;
        result.reserve(output_dimension());
        for (const auto& sample : samples_) {
            const Vector value = sample.observation(perturbations[sample.node]);
            result.insert(result.end(), value.begin(), value.end());
        }
        return result;
    }

    [[nodiscard]] Vector transpose_apply(std::span<const double> stacked_cotangent) const {
        if (stacked_cotangent.size() != output_dimension())
            throw std::invalid_argument("stacked observation cotangent has wrong dimension");
        std::vector<Vector> node_loads(trajectory_.node_count(), Vector(input_dimension(), 0.0));
        std::size_t offset = 0;
        for (const auto& sample : samples_) {
            const auto m = sample.observation.observation_dimension;
            const Vector load = sample.observation.transpose(stacked_cotangent.subspan(offset, m));
            for (std::size_t i = 0; i < input_dimension(); ++i)
                node_loads[sample.node][i] += load[i];
            offset += m;
        }
        return trajectory_.pullback(node_loads);
    }

    // Apply J^T Q to a stacked observation vector, where Q contains each
    // sample's trapezoidal weight and precision operator.
    [[nodiscard]] Vector weighted_transpose_apply(std::span<const double> stacked_value) const {
        if (stacked_value.size() != output_dimension())
            throw std::invalid_argument("stacked observation vector has wrong dimension");
        Vector weighted(stacked_value.begin(), stacked_value.end());
        std::size_t offset = 0;
        for (const auto& sample : samples_) {
            const auto m = sample.observation.observation_dimension;
            const auto block = std::span<const double>(stacked_value).subspan(offset, m);
            Vector value = sample.precision_apply ? sample.precision_apply(block)
                                                   : Vector(block.begin(), block.end());
            if (value.size() != m) throw std::invalid_argument("precision operator returned wrong dimension");
            for (std::size_t i = 0; i < m; ++i) weighted[offset + i] = sample.quadrature_weight * value[i];
            offset += m;
        }
        return transpose_apply(weighted);
    }

    [[nodiscard]] Vector weighted_gram_action(std::span<const double> initial_direction) const {
        const Vector observed_direction = apply(initial_direction);
        return weighted_transpose_apply(observed_direction);
    }

private:
    const DiscreteTrajectoryLinearization& trajectory_;
    std::vector<ObservationSample> samples_;
};

inline SensitivityDiagnostics analyze_observability(
    const ObservationSensitivityOperator& sensitivity,
    double relative_rank_threshold = 1.0e-8) {
    return analyze_sensitivity(sensitivity, relative_rank_threshold);
}

inline SensitivityDiagnostics analyze_parameter_identifiability(
    const ParameterObservationSensitivityOperator& sensitivity,
    double relative_rank_threshold = 1.0e-8) {
    return analyze_sensitivity(sensitivity, relative_rank_threshold);
}

struct StateMetric {
    // For a W norm, apply computes W*x and solve computes W^{-1}*x.
    VectorAction apply;
    VectorAction solve;
};

struct AmplificationEstimate {
    double gain{};
    Vector initial_direction;
    Vector final_perturbation;
    int iterations{};
    bool converged{};
};

inline double inner_product(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size()) throw std::invalid_argument("inner-product dimensions differ");
    double value = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) value += x[i] * y[i];
    return value;
}

inline double metric_norm(std::span<const double> x, const StateMetric& metric = {}) {
    const Vector weighted = metric.apply ? metric.apply(x) : Vector(x.begin(), x.end());
    if (weighted.size() != x.size()) throw std::invalid_argument("state metric returned wrong dimension");
    const double square = inner_product(x, weighted);
    if (square < -1.0e-12 || !std::isfinite(square))
        throw std::invalid_argument("state metric must be positive definite");
    return std::sqrt(std::max(0.0, square));
}

inline AmplificationEstimate estimate_transient_amplification(
    const DiscreteTrajectoryLinearization& trajectory,
    std::size_t end_node,
    const StateMetric& metric = {},
    std::span<const double> initial_guess = {},
    int max_iterations = 200,
    double relative_tolerance = 1.0e-10) {
    if (end_node >= trajectory.node_count()) throw std::out_of_range("amplification end node out of range");
    if (max_iterations <= 0 || !(relative_tolerance > 0.0))
        throw std::invalid_argument("power iteration controls must be positive");
    if (static_cast<bool>(metric.apply) != static_cast<bool>(metric.solve))
        throw std::invalid_argument("a state metric must provide both apply and solve actions");
    const auto n = trajectory.state_dimension();
    Vector direction(n);
    for (std::size_t i = 0; i < n; ++i)
        direction[i] = std::sin(0.731 * static_cast<double>(i + 1))
                     + std::cos(0.213 * static_cast<double>(i + 1));
    if (!initial_guess.empty()) {
        if (initial_guess.size() != n) throw std::invalid_argument("amplification initial guess has wrong dimension");
        direction.assign(initial_guess.begin(), initial_guess.end());
    }
    double norm = metric_norm(direction, metric);
    if (norm == 0.0) throw std::invalid_argument("amplification initial guess must be nonzero");
    for (double& value : direction) value /= norm;

    AmplificationEstimate result;
    double previous_gain = -1.0;
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        const Vector final = trajectory.apply(direction, end_node);
        const Vector metric_final = metric.apply ? metric.apply(final) : final;
        const Vector pulled = trajectory.transpose_apply(metric_final, end_node);
        const Vector next = metric.solve ? metric.solve(pulled) : pulled;
        if (next.size() != n) throw std::invalid_argument("state metric solve returned wrong dimension");
        const double next_norm = metric_norm(next, metric);
        result.iterations = iteration;
        if (next_norm <= 1.0e-30) {
            result.gain = 0.0;
            result.initial_direction = direction;
            result.final_perturbation = final;
            result.converged = true;
            return result;
        }
        const double gain = metric_norm(final, metric);
        Vector next_direction = next;
        for (double& value : next_direction) value /= next_norm;
        const Vector metric_next = metric.apply ? metric.apply(next_direction) : next_direction;
        const double alignment = std::min(1.0, std::abs(inner_product(direction, metric_next)));
        const double direction_defect = std::sqrt(std::max(0.0, 1.0 - alignment * alignment));
        result.gain = gain;
        if (previous_gain >= 0.0
            && std::abs(gain - previous_gain) <= relative_tolerance * std::max(1.0e-30, gain)
            && direction_defect <= std::sqrt(relative_tolerance)) {
            result.converged = true;
            break;
        }
        previous_gain = gain;
        direction = std::move(next_direction);
    }
    result.initial_direction = direction;
    result.final_perturbation = trajectory.apply(direction, end_node);
    result.gain = metric_norm(result.final_perturbation, metric);
    return result;
}

inline std::vector<AmplificationEstimate> transient_amplification_history(
    const DiscreteTrajectoryLinearization& trajectory,
    const StateMetric& metric = {},
    int max_iterations = 200,
    double relative_tolerance = 1.0e-10) {
    std::vector<AmplificationEstimate> history(trajectory.node_count());
    Vector guess(trajectory.state_dimension());
    for (std::size_t i = 0; i < guess.size(); ++i)
        guess[i] = std::sin(0.731 * static_cast<double>(i + 1))
                 + std::cos(0.213 * static_cast<double>(i + 1));
    const double initial_norm = metric_norm(guess, metric);
    for (double& value : guess) value /= initial_norm;
    history[0] = {1.0, guess, guess, 0, true};
    for (std::size_t node = 1; node < trajectory.node_count(); ++node) {
        history[node] = estimate_transient_amplification(
            trajectory, node, metric, guess, max_iterations, relative_tolerance);
        guess = history[node].initial_direction;
    }
    return history;
}

} // namespace asdole
