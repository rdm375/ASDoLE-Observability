// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/matrix_free.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace asdole {

using StateJacobianDirectionalAction = std::function<void(
    std::span<const double>, std::span<const double>, std::span<const double>, std::span<double>)>;
using HessianBilinearPullback = std::function<Vector(
    std::span<const double>, std::span<const double>, std::span<const double>)>;
using ParameterRhsAction = std::function<Vector(
    std::span<const double>, std::span<const double>)>;
using ParameterJacobianAction = std::function<void(
    std::span<const double>, std::span<const double>, std::span<const double>, std::span<double>)>;
using ParameterRhsTransposeAction = std::function<Vector(
    std::span<const double>, std::span<const double>)>;
using ParameterJacobianBilinearPullback = std::function<Vector(
    std::span<const double>, std::span<const double>, std::span<const double>)>;

// These actions define the derivatives needed by a matrix-free discrete
// tangent/adjoint. The last callback returns g satisfying
//   g dot v = left^T (D J(y)[v]) right.
// It is the Hessian contraction used by reverse mode without forming D^2 f.
struct MatrixFreeDerivativeActions {
    StateJacobianAction jacobian_transpose;
    StateJacobianDirectionalAction jacobian_directional;
    HessianBilinearPullback hessian_bilinear_pullback;
};

// Parameter derivatives for f(y,p) and J(y,p). The Jacobian pullback returns
// g satisfying g dot dp = left^T (D_p J(y)[dp]) right.
struct MatrixFreeParameterDerivativeActions {
    ParameterRhsAction rhs_parameter;
    ParameterJacobianAction jacobian_parameter;
    ParameterRhsTransposeAction rhs_parameter_transpose;
    ParameterJacobianBilinearPullback jacobian_parameter_bilinear_pullback;
};

class MatrixFreeTrajectoryLinearization {
public:
    MatrixFreeTrajectoryLinearization(const MatrixFreeStepper& stepper,
                                      MatrixFreeDerivativeActions derivatives,
                                      const std::vector<Vector>& states,
                                      std::vector<double> step_sizes)
        : stepper_(stepper), derivatives_(std::move(derivatives)), states_(states),
          step_sizes_(std::move(step_sizes)) {
        if (!derivatives_.jacobian_transpose || !derivatives_.jacobian_directional
            || !derivatives_.hessian_bilinear_pullback || states_.empty()
            || step_sizes_.size() + 1 != states_.size())
            throw std::invalid_argument("matrix-free trajectory requires states, steps, and derivative actions");
        const std::size_t n = states_.front().size();
        if (n == 0) throw std::invalid_argument("trajectory state dimension must be positive");
        for (const auto& state : states_)
            if (state.size() != n) throw std::invalid_argument("trajectory state dimensions differ");
        for (double h : step_sizes_)
            if (!(h > 0.0) || !std::isfinite(h))
                throw std::invalid_argument("trajectory step sizes must be finite and positive");
    }

    [[nodiscard]] std::size_t state_dimension() const noexcept { return states_.front().size(); }
    [[nodiscard]] std::size_t node_count() const noexcept { return states_.size(); }
    [[nodiscard]] std::size_t step_count() const noexcept { return step_sizes_.size(); }
    [[nodiscard]] const std::vector<Vector>& states() const noexcept { return states_; }

    [[nodiscard]] Vector apply_step(std::span<const double> direction,
                                    std::size_t step_index) const {
        check_step(step_index);
        if (direction.size() != state_dimension()) throw std::invalid_argument("TLM direction has wrong size");
        const auto& y = states_[step_index];
        const double h = step_sizes_[step_index];
        const double b = 0.5 * h;
        const double theta = stepper_.theta();
        const auto stages = stepper_.step_stages(y, h);
        Vector jv(state_dimension(), 0.0);
        stepper_.apply_jacobian(y, direction, jv);

        Vector dj_full(state_dimension(), 0.0), dj_first(state_dimension(), 0.0), dj_second(state_dimension(), 0.0);
        derivatives_.jacobian_directional(y, direction, stages.full_increment, dj_full);
        derivatives_.jacobian_directional(y, direction, stages.first_half_increment, dj_first);
        derivatives_.jacobian_directional(y, direction, stages.second_half_increment, dj_second);
        const auto solve = [&](double coefficient, std::span<const double> rhs) {
            return stepper_.solve_shifted(y, coefficient, rhs);
        };

        Vector rhs_full(state_dimension()), rhs_first(state_dimension());
        for (std::size_t i = 0; i < state_dimension(); ++i) {
            rhs_full[i] = h * jv[i] + theta * h * dj_full[i];
            rhs_first[i] = b * jv[i] + theta * b * dj_first[i];
        }
        const Vector delta_full = solve(theta * h, rhs_full);
        const Vector delta_first = solve(theta * b, rhs_first);
        Vector j_delta_first(state_dimension(), 0.0);
        stepper_.apply_jacobian(y, delta_first, j_delta_first);
        Vector rhs_second(state_dimension());
        for (std::size_t i = 0; i < state_dimension(); ++i)
            rhs_second[i] = b * (jv[i] + dj_first[i] + j_delta_first[i] + theta * dj_second[i]);
        const Vector delta_second = solve(theta * b, rhs_second);

        Vector result(direction.begin(), direction.end());
        for (std::size_t i = 0; i < state_dimension(); ++i)
            result[i] += 2.0 * delta_first[i] + 2.0 * delta_second[i] - delta_full[i];
        return result;
    }

    [[nodiscard]] Vector adjoint_step(std::span<const double> output_cotangent,
                                      std::size_t step_index) const {
        check_step(step_index);
        if (output_cotangent.size() != state_dimension())
            throw std::invalid_argument("adjoint cotangent has wrong size");
        const auto& y = states_[step_index];
        const double h = step_sizes_[step_index];
        const double b = 0.5 * h;
        const double theta = stepper_.theta();
        const auto stages = stepper_.step_stages(y, h);
        const auto solve_transpose = [&](double coefficient, std::span<const double> rhs) {
            return stepper_.solve_shifted_transpose(y, coefficient, rhs, derivatives_.jacobian_transpose);
        };

        Vector bar_full(state_dimension()), bar_first(state_dimension()), bar_second(state_dimension());
        for (std::size_t i = 0; i < state_dimension(); ++i) {
            bar_full[i] = -output_cotangent[i];
            bar_first[i] = 2.0 * output_cotangent[i];
            bar_second[i] = 2.0 * output_cotangent[i];
        }
        Vector bar_f(state_dimension(), 0.0);
        Vector hessian_term(state_dimension(), 0.0);

        const Vector lambda_second = solve_transpose(theta * b, bar_second);
        Vector jt_lambda_second(state_dimension(), 0.0);
        derivatives_.jacobian_transpose(y, lambda_second, jt_lambda_second);
        for (std::size_t i = 0; i < state_dimension(); ++i) {
            bar_f[i] += b * lambda_second[i];
            bar_first[i] += b * jt_lambda_second[i];
        }
        add_hessian_term(hessian_term, lambda_second, stages.first_half_increment, b, y);
        add_hessian_term(hessian_term, lambda_second, stages.second_half_increment, theta * b, y);

        const Vector lambda_first = solve_transpose(theta * b, bar_first);
        const Vector lambda_full = solve_transpose(theta * h, bar_full);
        for (std::size_t i = 0; i < state_dimension(); ++i)
            bar_f[i] += b * lambda_first[i] + h * lambda_full[i];
        add_hessian_term(hessian_term, lambda_first, stages.first_half_increment, theta * b, y);
        add_hessian_term(hessian_term, lambda_full, stages.full_increment, theta * h, y);

        Vector jt_bar_f(state_dimension(), 0.0);
        derivatives_.jacobian_transpose(y, bar_f, jt_bar_f);
        Vector result(output_cotangent.begin(), output_cotangent.end());
        for (std::size_t i = 0; i < state_dimension(); ++i)
            result[i] += jt_bar_f[i] + hessian_term[i];
        return result;
    }

    [[nodiscard]] Vector parameter_tangent_step(
        std::span<const double> parameter_direction, std::size_t step_index,
        const MatrixFreeParameterDerivativeActions& parameter_actions) const {
        check_step(step_index);
        validate_parameter_actions(parameter_actions);
        const auto& y = states_[step_index];
        const double h = step_sizes_[step_index];
        const double b = 0.5 * h;
        const double theta = stepper_.theta();
        const auto stages = stepper_.step_stages(y, h);
        const Vector f_parameter = parameter_actions.rhs_parameter(y, parameter_direction);
        if (f_parameter.size() != state_dimension())
            throw std::invalid_argument("parameter RHS action returned wrong dimension");

        const auto j_parameter_on = [&](std::span<const double> vector) {
            Vector value(state_dimension(), 0.0);
            parameter_actions.jacobian_parameter(y, parameter_direction, vector, value);
            return value;
        };
        const Vector jp_full = j_parameter_on(stages.full_increment);
        const Vector jp_first = j_parameter_on(stages.first_half_increment);
        const Vector jp_second = j_parameter_on(stages.second_half_increment);
        Vector rhs_full(state_dimension()), rhs_first(state_dimension());
        for (std::size_t i = 0; i < state_dimension(); ++i) {
            rhs_full[i] = h * f_parameter[i] + theta * h * jp_full[i];
            rhs_first[i] = b * f_parameter[i] + theta * b * jp_first[i];
        }
        const Vector delta_full = stepper_.solve_shifted(y, theta * h, rhs_full);
        const Vector delta_first = stepper_.solve_shifted(y, theta * b, rhs_first);
        Vector j_delta_first(state_dimension(), 0.0);
        stepper_.apply_jacobian(y, delta_first, j_delta_first);
        Vector rhs_second(state_dimension());
        for (std::size_t i = 0; i < state_dimension(); ++i)
            rhs_second[i] = b * (f_parameter[i] + jp_first[i] + j_delta_first[i]
                                 + theta * jp_second[i]);
        const Vector delta_second = stepper_.solve_shifted(y, theta * b, rhs_second);
        Vector result(state_dimension());
        for (std::size_t i = 0; i < state_dimension(); ++i)
            result[i] = 2.0 * delta_first[i] + 2.0 * delta_second[i] - delta_full[i];
        return result;
    }

    [[nodiscard]] Vector parameter_step_pullback(
        std::span<const double> output_cotangent, std::size_t step_index,
        const MatrixFreeParameterDerivativeActions& parameter_actions,
        std::size_t parameter_dimension) const {
        check_step(step_index);
        validate_parameter_actions(parameter_actions);
        if (output_cotangent.size() != state_dimension() || parameter_dimension == 0)
            throw std::invalid_argument("parameter step pullback dimensions are invalid");
        const auto& y = states_[step_index];
        const double h = step_sizes_[step_index];
        const double b = 0.5 * h;
        const double theta = stepper_.theta();
        const auto stages = stepper_.step_stages(y, h);
        Vector bar_first(state_dimension()), bar_second(state_dimension()), bar_full(state_dimension());
        for (std::size_t i = 0; i < state_dimension(); ++i) {
            bar_full[i] = -output_cotangent[i];
            bar_first[i] = 2.0 * output_cotangent[i];
            bar_second[i] = 2.0 * output_cotangent[i];
        }
        Vector parameter_gradient(parameter_dimension, 0.0);
        const auto add_parameter = [&](std::span<const double> value) {
            if (value.size() != parameter_dimension)
                throw std::invalid_argument("parameter pullback returned wrong dimension");
            for (std::size_t i = 0; i < parameter_dimension; ++i) parameter_gradient[i] += value[i];
        };
        const auto add_jacobian_parameter = [&](std::span<const double> left,
                                                std::span<const double> right,
                                                double scale) {
            const Vector value = parameter_actions.jacobian_parameter_bilinear_pullback(y, left, right);
            if (value.size() != parameter_dimension)
                throw std::invalid_argument("Jacobian parameter pullback returned wrong dimension");
            for (std::size_t i = 0; i < parameter_dimension; ++i)
                parameter_gradient[i] += scale * value[i];
        };

        const Vector lambda_second = stepper_.solve_shifted_transpose(
            y, theta * b, bar_second, derivatives_.jacobian_transpose);
        Vector jt_lambda_second(state_dimension(), 0.0);
        derivatives_.jacobian_transpose(y, lambda_second, jt_lambda_second);
        Vector bar_f(state_dimension());
        for (std::size_t i = 0; i < state_dimension(); ++i) {
            bar_f[i] = b * lambda_second[i];
            bar_first[i] += b * jt_lambda_second[i];
        }
        add_jacobian_parameter(lambda_second, stages.first_half_increment, b);
        add_jacobian_parameter(lambda_second, stages.second_half_increment, theta * b);

        const Vector lambda_first = stepper_.solve_shifted_transpose(
            y, theta * b, bar_first, derivatives_.jacobian_transpose);
        const Vector lambda_full = stepper_.solve_shifted_transpose(
            y, theta * h, bar_full, derivatives_.jacobian_transpose);
        for (std::size_t i = 0; i < state_dimension(); ++i)
            bar_f[i] += b * lambda_first[i] + h * lambda_full[i];
        add_parameter(parameter_actions.rhs_parameter_transpose(y, bar_f));
        add_jacobian_parameter(lambda_first, stages.first_half_increment, theta * b);
        add_jacobian_parameter(lambda_full, stages.full_increment, theta * h);
        return parameter_gradient;
    }

    [[nodiscard]] Vector apply(std::span<const double> initial_direction,
                               std::size_t end_node) const {
        check_node(end_node);
        if (initial_direction.size() != state_dimension())
            throw std::invalid_argument("initial TLM direction has wrong size");
        Vector value(initial_direction.begin(), initial_direction.end());
        for (std::size_t k = 0; k < end_node; ++k) value = apply_step(value, k);
        return value;
    }

    [[nodiscard]] Vector pullback_to_initial(std::span<const double> terminal_cotangent) const {
        if (terminal_cotangent.size() != state_dimension())
            throw std::invalid_argument("terminal cotangent has wrong size");
        Vector value(terminal_cotangent.begin(), terminal_cotangent.end());
        for (std::size_t k = step_count(); k > 0; --k) value = adjoint_step(value, k - 1);
        return value;
    }

    [[nodiscard]] Vector pullback_to_node(std::span<const double> end_cotangent,
                                          std::size_t start_node,
                                          std::size_t end_node) const {
        check_node(start_node);
        check_node(end_node);
        if (start_node > end_node || end_cotangent.size() != state_dimension())
            throw std::invalid_argument("matrix-free pullback node range or dimension is invalid");
        Vector value(end_cotangent.begin(), end_cotangent.end());
        for (std::size_t k = end_node; k > start_node; --k)
            value = adjoint_step(value, k - 1);
        return value;
    }

private:
    static void validate_parameter_actions(const MatrixFreeParameterDerivativeActions& actions) {
        if (!actions.rhs_parameter || !actions.jacobian_parameter
            || !actions.rhs_parameter_transpose || !actions.jacobian_parameter_bilinear_pullback)
            throw std::invalid_argument("all matrix-free parameter derivative actions are required");
    }

    void add_hessian_term(Vector& sum, std::span<const double> left,
                          std::span<const double> right, double scale,
                          std::span<const double> state) const {
        const Vector value = derivatives_.hessian_bilinear_pullback(state, left, right);
        if (value.size() != state_dimension())
            throw std::invalid_argument("Hessian bilinear pullback returned wrong dimension");
        for (std::size_t i = 0; i < sum.size(); ++i) sum[i] += scale * value[i];
    }

    void check_node(std::size_t node) const {
        if (node >= node_count()) throw std::out_of_range("trajectory node index out of range");
    }
    void check_step(std::size_t step) const {
        if (step >= step_count()) throw std::out_of_range("trajectory step index out of range");
    }

    const MatrixFreeStepper& stepper_;
    MatrixFreeDerivativeActions derivatives_;
    const std::vector<Vector>& states_;
    std::vector<double> step_sizes_;
};

} // namespace asdole
