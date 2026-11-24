// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/stepper.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace asdole {

using JacobianAction = std::function<void(std::span<const double>, std::span<double>)>;
using StateJacobianAction = std::function<void(
    std::span<const double>, std::span<const double>, std::span<double>)>;

struct GmresOptions {
    double relative_tolerance{1.0e-10};
    double absolute_tolerance{1.0e-13};
    std::size_t restart{40};
    std::size_t maximum_iterations{400};
};

struct LinearSolveResult {
    Vector solution;
    std::size_t iterations{};
    double residual_norm{};
};

namespace matrix_free_detail {

inline double dot(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) throw std::invalid_argument("vector dimensions differ");
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}

inline double norm(std::span<const double> x) {
    return std::sqrt(dot(x, x));
}

inline Vector apply_shifted_operator(const JacobianAction& jacobian_action,
                                     std::span<const double> x,
                                     double coefficient) {
    Vector jx(x.size(), 0.0);
    jacobian_action(x, jx);
    if (jx.size() != x.size()) throw std::invalid_argument("Jacobian action returned wrong dimension");
    Vector result(x.begin(), x.end());
    for (std::size_t i = 0; i < result.size(); ++i) result[i] -= coefficient * jx[i];
    return result;
}

} // namespace matrix_free_detail

// Restarted, unrestarted-at-the-outer-level GMRES for A x = b, where
// A = I - coefficient * J is applied without assembling J or A. This is a
// dependency-free reference backend; production applications can substitute
// a sparse direct or preconditioned Krylov solver through the same operator.
inline LinearSolveResult gmres_solve_shifted_jacobian(
    const JacobianAction& jacobian_action,
    double coefficient,
    std::span<const double> rhs,
    const GmresOptions& options = {}) {
    if (!jacobian_action || rhs.empty()) throw std::invalid_argument("GMRES requires an operator and nonempty RHS");
    if (!(options.relative_tolerance > 0.0) || !(options.absolute_tolerance >= 0.0)
        || options.restart == 0 || options.maximum_iterations == 0)
        throw std::invalid_argument("invalid GMRES options");

    const std::size_t n = rhs.size();
    const double target = std::max(options.absolute_tolerance,
                                   options.relative_tolerance * matrix_free_detail::norm(rhs));
    LinearSolveResult result{Vector(n, 0.0), 0, matrix_free_detail::norm(rhs)};
    if (result.residual_norm <= target) return result;

    const std::size_t restart = std::min(options.restart, n);
    while (result.iterations < options.maximum_iterations) {
        Vector residual(rhs.begin(), rhs.end());
        const Vector ax = matrix_free_detail::apply_shifted_operator(
            jacobian_action, result.solution, coefficient);
        for (std::size_t i = 0; i < n; ++i) residual[i] -= ax[i];
        const double beta = matrix_free_detail::norm(residual);
        result.residual_norm = beta;
        if (beta <= target) return result;

        const std::size_t m = std::min(restart, options.maximum_iterations - result.iterations);
        std::vector<Vector> basis(m + 1, Vector(n, 0.0));
        for (std::size_t i = 0; i < n; ++i) basis[0][i] = residual[i] / beta;
        std::vector<std::vector<double>> hessenberg(m + 1, std::vector<double>(m, 0.0));
        std::vector<double> cosine(m, 0.0), sine(m, 0.0), g(m + 1, 0.0);
        g[0] = beta;
        std::size_t used = 0;
        bool converged = false;

        for (std::size_t j = 0; j < m; ++j) {
            Vector work = matrix_free_detail::apply_shifted_operator(
                jacobian_action, basis[j], coefficient);
            for (std::size_t i = 0; i <= j; ++i) {
                hessenberg[i][j] = matrix_free_detail::dot(work, basis[i]);
                for (std::size_t k = 0; k < n; ++k) work[k] -= hessenberg[i][j] * basis[i][k];
            }
            hessenberg[j + 1][j] = matrix_free_detail::norm(work);
            if (hessenberg[j + 1][j] > 0.0)
                for (std::size_t k = 0; k < n; ++k) basis[j + 1][k] = work[k] / hessenberg[j + 1][j];

            for (std::size_t i = 0; i < j; ++i) {
                const double first = cosine[i] * hessenberg[i][j] + sine[i] * hessenberg[i + 1][j];
                const double second = -sine[i] * hessenberg[i][j] + cosine[i] * hessenberg[i + 1][j];
                hessenberg[i][j] = first;
                hessenberg[i + 1][j] = second;
            }
            const double denominator = std::hypot(hessenberg[j][j], hessenberg[j + 1][j]);
            cosine[j] = denominator == 0.0 ? 1.0 : hessenberg[j][j] / denominator;
            sine[j] = denominator == 0.0 ? 0.0 : hessenberg[j + 1][j] / denominator;
            hessenberg[j][j] = cosine[j] * hessenberg[j][j] + sine[j] * hessenberg[j + 1][j];
            hessenberg[j + 1][j] = 0.0;
            g[j + 1] = -sine[j] * g[j];
            g[j] *= cosine[j];
            ++result.iterations;
            used = j + 1;
            if (std::abs(g[j + 1]) <= target || denominator == 0.0) {
                converged = true;
                break;
            }
        }

        Vector coefficients(used, 0.0);
        for (std::size_t reverse = used; reverse > 0; --reverse) {
            const std::size_t i = reverse - 1;
            double value = g[i];
            for (std::size_t j = i + 1; j < used; ++j) value -= hessenberg[i][j] * coefficients[j];
            if (std::abs(hessenberg[i][i]) <= std::numeric_limits<double>::min())
                throw std::runtime_error("GMRES encountered a singular Krylov factor");
            coefficients[i] = value / hessenberg[i][i];
        }
        for (std::size_t j = 0; j < used; ++j)
            for (std::size_t i = 0; i < n; ++i)
                result.solution[i] += coefficients[j] * basis[j][i];

        result.residual_norm = std::abs(g[used]);
        if (converged && result.residual_norm <= target) return result;
    }
    throw std::runtime_error("GMRES failed to converge within maximum_iterations");
}

struct MatrixFreeStepResult {
    StepResult step;
    std::size_t linear_iterations{};
    double maximum_linear_residual{};
};

struct MatrixFreeStepStages {
    Vector initial;
    Vector rhs;
    Vector full_increment;
    Vector first_half_increment;
    Vector second_half_increment;
    Vector state;
    std::size_t linear_iterations{};
    double maximum_linear_residual{};
};

// Matrix-free ASDoLE state step. The RHS/Jacobian products and shifted solves
// never form an n-by-n matrix. This state-only path is intended for large
// sparse/structured models; the dense Stepper remains the reference path for
// discrete TLM and adjoint analyses.
class MatrixFreeStepper {
public:
    MatrixFreeStepper(RhsFunction rhs, StateJacobianAction jacobian_action,
                      double theta = 1.0, GmresOptions options = {})
        : rhs_(std::move(rhs)), jacobian_action_(std::move(jacobian_action)),
          theta_(theta), options_(options) {
        if (!rhs_ || !jacobian_action_) throw std::invalid_argument("matrix-free ASDoLE requires RHS and Jacobian action");
        if (!std::isfinite(theta_)) throw std::invalid_argument("theta must be finite");
    }

    [[nodiscard]] double theta() const noexcept { return theta_; }

    void apply_jacobian(std::span<const double> state,
                        std::span<const double> direction,
                        std::span<double> image) const {
        if (state.size() != direction.size() || image.size() != direction.size())
            throw std::invalid_argument("Jacobian-vector dimensions differ");
        jacobian_action_(state, direction, image);
    }

    [[nodiscard]] Vector solve_shifted(std::span<const double> state, double coefficient,
                                       std::span<const double> rhs) const {
        const JacobianAction action = [this, state](std::span<const double> direction,
                                                    std::span<double> image) {
            jacobian_action_(state, direction, image);
        };
        return gmres_solve_shifted_jacobian(action, coefficient, rhs, options_).solution;
    }

    [[nodiscard]] Vector solve_shifted_transpose(
        std::span<const double> state, double coefficient, std::span<const double> rhs,
        const StateJacobianAction& transpose_action) const {
        if (!transpose_action) throw std::invalid_argument("transpose Jacobian action is required");
        const JacobianAction action = [state, &transpose_action](
            std::span<const double> direction, std::span<double> image) {
            transpose_action(state, direction, image);
        };
        return gmres_solve_shifted_jacobian(action, coefficient, rhs, options_).solution;
    }

    [[nodiscard]] MatrixFreeStepStages step_stages(std::span<const double> initial, double h) const {
        if (initial.empty()) throw std::invalid_argument("ASDoLE state must not be empty");
        if (!(h > 0.0) || !std::isfinite(h)) throw std::invalid_argument("step size must be finite and positive");
        const std::size_t n = initial.size();
        Vector f(n, 0.0);
        rhs_(initial, f);
        if (f.size() != n) throw std::invalid_argument("RHS callback changed the state dimension");

        const JacobianAction jacobian_at_initial = [this, initial](
            std::span<const double> direction, std::span<double> image) {
            jacobian_action_(initial, direction, image);
        };

        const auto solve = [&](double coefficient, std::span<const double> right_hand_side) {
            return gmres_solve_shifted_jacobian(jacobian_at_initial, coefficient, right_hand_side, options_);
        };
        Vector full_rhs(n), first_rhs(n);
        for (std::size_t i = 0; i < n; ++i) {
            full_rhs[i] = h * f[i];
            first_rhs[i] = 0.5 * h * f[i];
        }
        const auto full = solve(theta_ * h, full_rhs);
        const auto first = solve(theta_ * 0.5 * h, first_rhs);
        Vector j_first(n, 0.0);
        jacobian_action_(initial, first.solution, j_first);
        if (j_first.size() != n) throw std::invalid_argument("Jacobian action returned wrong dimension");
        Vector second_rhs(n);
        for (std::size_t i = 0; i < n; ++i) second_rhs[i] = 0.5 * h * (f[i] + j_first[i]);
        const auto second = solve(theta_ * 0.5 * h, second_rhs);

        MatrixFreeStepStages result;
        result.initial.assign(initial.begin(), initial.end());
        result.rhs = std::move(f);
        result.full_increment = full.solution;
        result.first_half_increment = first.solution;
        result.second_half_increment = second.solution;
        result.state.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double full_state = initial[i] + full.solution[i];
            const double doubled_state = initial[i] + first.solution[i] + second.solution[i];
            result.state[i] = 2.0 * doubled_state - full_state;
        }
        result.linear_iterations = full.iterations + first.iterations + second.iterations;
        result.maximum_linear_residual = std::max({full.residual_norm, first.residual_norm, second.residual_norm});
        return result;
    }

    [[nodiscard]] MatrixFreeStepResult step(std::span<const double> initial, double h) const {
        const MatrixFreeStepStages stages = step_stages(initial, h);
        const std::size_t n = stages.initial.size();
        MatrixFreeStepResult result;
        result.step.state = stages.state;
        result.step.full_step.resize(n);
        result.step.doubled_step.resize(n);
        result.step.error_estimate.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            result.step.full_step[i] = stages.initial[i] + stages.full_increment[i];
            result.step.doubled_step[i] = stages.initial[i] + stages.first_half_increment[i]
                                        + stages.second_half_increment[i];
            result.step.error_estimate[i] = result.step.doubled_step[i] - result.step.full_step[i];
        }
        result.linear_iterations = stages.linear_iterations;
        result.maximum_linear_residual = stages.maximum_linear_residual;
        return result;
    }

private:
    RhsFunction rhs_;
    StateJacobianAction jacobian_action_;
    double theta_;
    GmresOptions options_;
};

} // namespace asdole
