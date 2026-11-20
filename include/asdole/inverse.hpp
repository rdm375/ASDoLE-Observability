// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/stepper.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>

namespace asdole {

struct LeastSquaresLinearization {
    double objective{};
    Vector gradient;
    // Gauss-Newton approximation to the objective Hessian.
    DenseMatrix gauss_newton_matrix;
};

using LeastSquaresLinearizer = std::function<LeastSquaresLinearization(std::span<const double>)>;
using LeastSquaresObjective = std::function<double(std::span<const double>)>;

struct GaussNewtonOptions {
    int maximum_iterations{50};
    int maximum_line_search_steps{16};
    double gradient_tolerance{1.0e-9};
    double step_tolerance{1.0e-10};
    double initial_damping{1.0e-3};
    double minimum_damping{1.0e-12};
    double maximum_damping{1.0e12};
    double armijo_constant{1.0e-4};
};

struct GaussNewtonResult {
    Vector parameters;
    double objective{};
    double gradient_norm{};
    double damping{};
    int iterations{};
    bool converged{};
};

inline double euclidean_norm(std::span<const double> value) {
    double square = 0.0;
    for (double component : value) square += component * component;
    return std::sqrt(square);
}

// Damped Gauss-Newton with diagonal Levenberg damping and an Armijo
// backtracking line search. The caller supplies the objective and its
// Gauss-Newton linearization, which may be built from matrix-free sensitivity
// actions. This routine is agnostic to the forward model and observation type.
inline GaussNewtonResult damped_gauss_newton(
    LeastSquaresLinearizer linearize,
    LeastSquaresObjective objective,
    Vector initial_parameters,
    const GaussNewtonOptions& options = {}) {
    if (!linearize || !objective || initial_parameters.empty()
        || options.maximum_iterations <= 0 || options.maximum_line_search_steps <= 0
        || !(options.gradient_tolerance > 0.0) || !(options.step_tolerance > 0.0)
        || !(options.initial_damping > 0.0) || !(options.minimum_damping > 0.0)
        || !(options.maximum_damping >= options.initial_damping)
        || !(options.armijo_constant > 0.0 && options.armijo_constant < 1.0))
        throw std::invalid_argument("invalid damped Gauss-Newton configuration");

    GaussNewtonResult result;
    result.parameters = std::move(initial_parameters);
    result.damping = options.initial_damping;
    LeastSquaresLinearization current = linearize(result.parameters);
    const std::size_t n = result.parameters.size();
    auto validate = [n](const LeastSquaresLinearization& value) {
        if (value.gradient.size() != n || value.gauss_newton_matrix.rows() != n
            || value.gauss_newton_matrix.cols() != n || !std::isfinite(value.objective))
            throw std::invalid_argument("least-squares linearization has invalid dimensions or objective");
        for (double component : value.gradient)
            if (!std::isfinite(component)) throw std::invalid_argument("least-squares gradient is not finite");
    };
    validate(current);
    const double initial_gradient_norm = euclidean_norm(current.gradient);
    const double gradient_scale = std::max(1.0, initial_gradient_norm);
    result.objective = current.objective;
    result.gradient_norm = euclidean_norm(current.gradient);
    if (result.gradient_norm <= options.gradient_tolerance * gradient_scale) {
        result.converged = true;
        return result;
    }

    for (int iteration = 0; iteration < options.maximum_iterations; ++iteration) {
        bool accepted = false;
        Vector accepted_parameters;
        double accepted_objective = current.objective;
        double accepted_step_norm = 0.0;

        for (int damping_attempt = 0; damping_attempt < 24 && !accepted; ++damping_attempt) {
            DenseMatrix system = current.gauss_newton_matrix;
            for (std::size_t i = 0; i < n; ++i) {
                const double diagonal_scale = std::max(std::abs(system(i, i)), 1.0e-12);
                system(i, i) += result.damping * diagonal_scale;
            }
            Vector rhs(current.gradient.size());
            for (std::size_t i = 0; i < n; ++i) rhs[i] = -current.gradient[i];

            Vector direction;
            try {
                direction = detail::DenseLU(std::move(system)).solve(rhs);
            } catch (const std::runtime_error&) {
                result.damping = std::min(options.maximum_damping, result.damping * 10.0);
                continue;
            }
            double directional_derivative = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                directional_derivative += current.gradient[i] * direction[i];
            if (!(directional_derivative < 0.0)) {
                result.damping = std::min(options.maximum_damping, result.damping * 10.0);
                continue;
            }

            double alpha = 1.0;
            for (int line = 0; line < options.maximum_line_search_steps; ++line) {
                Vector candidate = result.parameters;
                for (std::size_t i = 0; i < n; ++i) candidate[i] += alpha * direction[i];
                const double candidate_objective = objective(candidate);
                const double required_decrease = options.armijo_constant * alpha * -directional_derivative;
                if (std::isfinite(candidate_objective)
                    && candidate_objective <= current.objective - required_decrease) {
                    accepted = true;
                    accepted_parameters = std::move(candidate);
                    accepted_objective = candidate_objective;
                    accepted_step_norm = alpha * euclidean_norm(direction);
                    break;
                }
                alpha *= 0.5;
            }
            if (!accepted) {
                if (result.damping >= options.maximum_damping) break;
                result.damping = std::min(options.maximum_damping, result.damping * 10.0);
            }
        }

        if (!accepted) break;
        result.parameters = std::move(accepted_parameters);
        current = linearize(result.parameters);
        validate(current);
        result.objective = accepted_objective;
        result.gradient_norm = euclidean_norm(current.gradient);
        result.iterations = iteration + 1;
        result.damping = std::max(options.minimum_damping, result.damping / 3.0);

        if (result.gradient_norm <= options.gradient_tolerance * gradient_scale
            || accepted_step_norm <= options.step_tolerance * (1.0 + euclidean_norm(result.parameters))) {
            result.converged = true;
            break;
        }
    }
    return result;
}

} // namespace asdole
