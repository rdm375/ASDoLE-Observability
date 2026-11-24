// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/covariance.hpp"
#include "asdole/matrix_free_differentiation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using asdole::DenseMatrix;
using asdole::Vector;

constexpr std::size_t grid_size = 48;
constexpr std::size_t time_steps = 100;
constexpr double final_time = 0.5;
constexpr double diffusivity = 0.005;
constexpr double growth_rate = 2.0;
constexpr double theta = 0.7;

double spacing() { return 1.0 / static_cast<double>(grid_size + 1); }

void rhs(std::span<const double> u, std::span<double> f) {
    const double inverse_dx2 = 1.0 / (spacing() * spacing());
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double left = i == 0 ? 0.0 : u[i - 1];
        const double right = i + 1 == grid_size ? 0.0 : u[i + 1];
        const double laplacian = (left - 2.0 * u[i] + right) * inverse_dx2;
        f[i] = diffusivity * laplacian + growth_rate * u[i] * (1.0 - u[i]);
    }
}

void jacobian_action(std::span<const double> u,
                     std::span<const double> v,
                     std::span<double> image) {
    const double inverse_dx2 = 1.0 / (spacing() * spacing());
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double left = i == 0 ? 0.0 : v[i - 1];
        const double right = i + 1 == grid_size ? 0.0 : v[i + 1];
        const double laplacian = (left - 2.0 * v[i] + right) * inverse_dx2;
        image[i] = diffusivity * laplacian + growth_rate * (1.0 - 2.0 * u[i]) * v[i];
    }
}

void jacobian_transpose_action(std::span<const double> u,
                               std::span<const double> v,
                               std::span<double> image) {
    // The centered Dirichlet Laplacian and pointwise reaction Jacobian are symmetric.
    jacobian_action(u, v, image);
}

void jacobian_directional_action(std::span<const double>,
                                 std::span<const double> direction,
                                 std::span<const double> vector,
                                 std::span<double> image) {
    for (std::size_t i = 0; i < grid_size; ++i)
        image[i] = -2.0 * growth_rate * direction[i] * vector[i];
}

Vector hessian_bilinear_pullback(std::span<const double>,
                                 std::span<const double> left,
                                 std::span<const double> right) {
    Vector result(grid_size);
    for (std::size_t i = 0; i < grid_size; ++i)
        result[i] = -2.0 * growth_rate * left[i] * right[i];
    return result;
}

double dot(std::span<const double> a, std::span<const double> b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

double relative_error(std::span<const double> a, std::span<const double> b) {
    double numerator = 0.0, denominator = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        numerator += (a[i] - b[i]) * (a[i] - b[i]);
        denominator += b[i] * b[i];
    }
    return std::sqrt(numerator) / std::max(1.0e-30, std::sqrt(denominator));
}

double factor_trace(const DenseMatrix& factor) {
    double trace = 0.0;
    for (std::size_t i = 0; i < factor.rows(); ++i)
        for (std::size_t j = 0; j < factor.cols(); ++j)
            trace += factor(i, j) * factor(i, j);
    return trace;
}

int main_impl(int argc, char** argv) {
    const bool check = argc > 1 && std::string(argv[1]) == "--check";
    asdole::GmresOptions options;
    options.relative_tolerance = 1.0e-12;
    options.absolute_tolerance = 1.0e-14;
    options.restart = 40;
    options.maximum_iterations = 400;
    const asdole::MatrixFreeStepper stepper(rhs, jacobian_action, theta, options);

    const double dx = spacing();
    std::vector<Vector> states(time_steps + 1, Vector(grid_size));
    std::vector<double> step_sizes(time_steps, final_time / static_cast<double>(time_steps));
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double x = static_cast<double>(i + 1) * dx;
        states[0][i] = std::sin(3.14159265358979323846 * x)
            * (0.2 + 0.7 * std::exp(-90.0 * (x - 0.5) * (x - 0.5)));
    }
    std::size_t total_gmres_iterations = 0;
    for (std::size_t k = 0; k < time_steps; ++k) {
        const auto result = stepper.step(states[k], step_sizes[k]);
        states[k + 1] = result.step.state;
        total_gmres_iterations += result.linear_iterations;
    }

    asdole::MatrixFreeDerivativeActions derivatives{
        jacobian_transpose_action, jacobian_directional_action, hessian_bilinear_pullback};
    const asdole::MatrixFreeTrajectoryLinearization linearization(
        stepper, std::move(derivatives), states, step_sizes);

    Vector direction(grid_size), terminal_cotangent(grid_size);
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double x = static_cast<double>(i + 1) * dx;
        direction[i] = std::sin(3.14159265358979323846 * x);
        terminal_cotangent[i] = std::cos(2.0 * 3.14159265358979323846 * x);
    }
    const Vector tangent = linearization.apply(direction, time_steps);
    const Vector pulled_back = linearization.pullback_to_initial(terminal_cotangent);
    const double dot_defect = std::abs(dot(tangent, terminal_cotangent) - dot(direction, pulled_back))
        / std::max({1.0, std::abs(dot(tangent, terminal_cotangent)),
                    std::abs(dot(direction, pulled_back))});

    DenseMatrix initial_factor(grid_size, 2);
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double x = static_cast<double>(i + 1) * dx;
        initial_factor(i, 0) = 1.0e-3 * std::sin(3.14159265358979323846 * x);
        initial_factor(i, 1) = 5.0e-4 * std::sin(4.0 * 3.14159265358979323846 * x);
    }
    const auto covariance_factors = asdole::propagate_covariance_factor(linearization, initial_factor);
    const double initial_trace = factor_trace(covariance_factors.front());
    const double final_trace = factor_trace(covariance_factors.back());

    constexpr double epsilon = 1.0e-6;
    Vector perturbed_initial = states.front();
    for (std::size_t i = 0; i < grid_size; ++i) perturbed_initial[i] += epsilon * direction[i];
    for (std::size_t k = 0; k < time_steps; ++k)
        perturbed_initial = stepper.step(perturbed_initial, step_sizes[k]).step.state;
    Vector finite_difference(grid_size);
    for (std::size_t i = 0; i < grid_size; ++i)
        finite_difference[i] = (perturbed_initial[i] - states.back()[i]) / epsilon;
    const double tlm_fd_error = relative_error(tangent, finite_difference);

    std::cout << std::setprecision(12)
              << "pde=Fisher-KPP 1d: u_t=0.005*u_xx+2*u*(1-u), Dirichlet boundaries\n"
              << "grid_points=" << grid_size << " dx=" << dx << " theta=" << theta
              << " steps=" << time_steps << " dt=" << step_sizes.front()
              << " final_time=" << final_time << '\n'
              << "mean_gmres_iterations_per_asdole_step="
              << static_cast<double>(total_gmres_iterations) / static_cast<double>(time_steps) << '\n'
              << "final_state_min=" << *std::min_element(states.back().begin(), states.back().end())
              << " final_state_max=" << *std::max_element(states.back().begin(), states.back().end()) << '\n'
              << "tlm_vs_forward_difference_relative_error=" << tlm_fd_error << '\n'
              << "tlm_adjoint_dot_defect=" << dot_defect << '\n'
              << "rank2_covariance_trace_initial=" << initial_trace
              << " final=" << final_trace
              << " ratio=" << final_trace / initial_trace << '\n';

    if (check && (!(tlm_fd_error < 2.0e-5) || !(dot_defect < 2.0e-8)
                  || !(std::isfinite(final_trace) && final_trace > 0.0)))
        throw std::runtime_error("matrix-free PDE derivative or covariance check failed");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return main_impl(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "matrix_free_reaction_diffusion: " << error.what() << '\n';
        return 2;
    }
}
