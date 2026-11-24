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
constexpr std::size_t step_count = 1500;
constexpr double final_time = 15.0;
constexpr double diffusivity = 0.005;
constexpr double growth_rate = 2.0;
constexpr double theta = 0.7;

double dx() { return 1.0 / static_cast<double>(grid_size + 1); }

void rhs(std::span<const double> u, std::span<double> f) {
    const double inv_dx2 = 1.0 / (dx() * dx());
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double left = i == 0 ? 0.0 : u[i - 1];
        const double right = i + 1 == grid_size ? 0.0 : u[i + 1];
        f[i] = diffusivity * (left - 2.0 * u[i] + right) * inv_dx2
             + growth_rate * u[i] * (1.0 - u[i]);
    }
}

void jacobian_action(std::span<const double> u, std::span<const double> v,
                     std::span<double> image) {
    const double inv_dx2 = 1.0 / (dx() * dx());
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double left = i == 0 ? 0.0 : v[i - 1];
        const double right = i + 1 == grid_size ? 0.0 : v[i + 1];
        image[i] = diffusivity * (left - 2.0 * v[i] + right) * inv_dx2
                 + growth_rate * (1.0 - 2.0 * u[i]) * v[i];
    }
}

Vector initial_profile_a() {
    Vector u(grid_size);
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double x = static_cast<double>(i + 1) * dx();
        u[i] = std::sin(3.14159265358979323846 * x)
             * (0.12 + 0.82 * std::exp(-110.0 * (x - 0.30) * (x - 0.30)));
    }
    return u;
}

Vector initial_profile_b() {
    Vector u(grid_size);
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double x = static_cast<double>(i + 1) * dx();
        u[i] = std::sin(3.14159265358979323846 * x)
             * (0.72 + 0.08 * std::cos(2.0 * 3.14159265358979323846 * x));
    }
    return u;
}

double l2_distance(std::span<const double> a, std::span<const double> b) {
    double square = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double difference = a[i] - b[i];
        square += dx() * difference * difference;
    }
    return std::sqrt(square);
}

double factor_trace(const DenseMatrix& factor) {
    double result = 0.0;
    for (std::size_t i = 0; i < factor.rows(); ++i)
        for (std::size_t j = 0; j < factor.cols(); ++j)
            result += factor(i, j) * factor(i, j);
    return result;
}

int main_impl(int argc, char** argv) {
    const bool check = argc > 1 && std::string(argv[1]) == "--check";
    asdole::GmresOptions options;
    options.relative_tolerance = 1.0e-12;
    options.absolute_tolerance = 1.0e-14;
    options.restart = 40;
    options.maximum_iterations = 400;
    const asdole::MatrixFreeStepper stepper(rhs, jacobian_action, theta, options);

    std::vector<Vector> states(step_count + 1, Vector(grid_size));
    std::vector<Vector> second_trajectory(step_count + 1, Vector(grid_size));
    std::vector<double> steps(step_count, final_time / static_cast<double>(step_count));
    states[0] = initial_profile_a();
    second_trajectory[0] = initial_profile_b();
    const std::vector<std::size_t> report_steps{0, 50, 100, 200, 500, 1000, step_count};
    std::vector<double> gaps;
    gaps.reserve(report_steps.size());
    gaps.push_back(l2_distance(states[0], second_trajectory[0]));
    std::size_t report_index = 1;
    for (std::size_t k = 0; k < step_count; ++k) {
        states[k + 1] = stepper.step(states[k], steps[k]).step.state;
        second_trajectory[k + 1] = stepper.step(second_trajectory[k], steps[k]).step.state;
        if (report_index < report_steps.size() && k + 1 == report_steps[report_index]) {
            gaps.push_back(l2_distance(states[k + 1], second_trajectory[k + 1]));
            ++report_index;
        }
    }

    asdole::MatrixFreeDerivativeActions derivatives{
        jacobian_action,
        [](std::span<const double>, std::span<const double> direction,
           std::span<const double> vector, std::span<double> image) {
            for (std::size_t i = 0; i < grid_size; ++i)
                image[i] = -2.0 * growth_rate * direction[i] * vector[i];
        },
        [](std::span<const double>, std::span<const double> left,
           std::span<const double> right) {
            Vector image(grid_size);
            for (std::size_t i = 0; i < grid_size; ++i)
                image[i] = -2.0 * growth_rate * left[i] * right[i];
            return image;
        }};
    const asdole::MatrixFreeTrajectoryLinearization linearization(
        stepper, std::move(derivatives), states, steps);
    DenseMatrix initial_factor(grid_size, 2);
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double x = static_cast<double>(i + 1) * dx();
        initial_factor(i, 0) = 1.0e-2 * std::sin(3.14159265358979323846 * x);
        initial_factor(i, 1) = 5.0e-3 * std::sin(3.0 * 3.14159265358979323846 * x);
    }
    const auto factors = asdole::propagate_covariance_factor(linearization, initial_factor);
    const double covariance_ratio = factor_trace(factors.back()) / factor_trace(factors.front());
    const double gap_ratio = gaps.back() / gaps.front();
    Vector terminal_rhs(grid_size, 0.0);
    rhs(states.back(), terminal_rhs);
    double steady_residual_inf = 0.0;
    for (double value : terminal_rhs) steady_residual_inf = std::max(steady_residual_inf, std::abs(value));

    std::cout << std::setprecision(12)
              << "pde=Fisher-KPP: u_t=0.005*u_xx+2*u*(1-u), homogeneous Dirichlet boundaries\n"
              << "grid_points=" << grid_size << " theta=" << theta << " steps=" << step_count
              << " dt=" << steps.front() << " final_time=" << final_time << '\n'
              << "positive_initial_profiles=distinct; both approach the same positive steady state\n"
              << "time,l2_profile_difference\n";
    for (std::size_t i = 0; i < report_steps.size(); ++i)
        std::cout << steps.front() * static_cast<double>(report_steps[i]) << ',' << gaps[i] << '\n';
    std::cout << "initial_condition_gap_ratio=" << gap_ratio
              << " initial_covariance_trace=" << factor_trace(factors.front())
              << " final_covariance_trace=" << factor_trace(factors.back())
              << " covariance_trace_ratio=" << covariance_ratio
              << " final_steady_residual_inf=" << steady_residual_inf << '\n';

    if (check && (!(gap_ratio < 0.02) || !(covariance_ratio < 0.05)
                  || !(steady_residual_inf < 1.0e-5)))
        throw std::runtime_error("Fisher-KPP initial-condition forgetting check failed");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return main_impl(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "matrix_free_reaction_diffusion_forgetting: " << error.what() << '\n';
        return 2;
    }
}
