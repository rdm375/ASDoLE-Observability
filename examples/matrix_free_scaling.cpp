// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/matrix_free.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using asdole::DenseMatrix;
using asdole::Vector;
using Clock = std::chrono::steady_clock;

void diffusion_action(std::span<const double> x, std::span<double> y) {
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i)
        y[i] = (i == 0 ? 0.0 : x[i - 1]) - 2.0 * x[i] + (i + 1 == n ? 0.0 : x[i + 1]);
}

void rhs(std::span<const double> x, std::span<double> y) { diffusion_action(x, y); }

void diffusion_jacobian_action(std::span<const double>,
                               std::span<const double> direction,
                               std::span<double> image) {
    diffusion_action(direction, image);
}

void dense_jacobian(std::span<const double>, DenseMatrix& j) {
    for (std::size_t i = 0; i < j.rows(); ++i) {
        j(i, i) = -2.0;
        if (i > 0) j(i, i - 1) = 1.0;
        if (i + 1 < j.rows()) j(i, i + 1) = 1.0;
    }
}

double relative_error(std::span<const double> a, std::span<const double> b) {
    double difference = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        difference += (a[i] - b[i]) * (a[i] - b[i]);
        scale += b[i] * b[i];
    }
    return std::sqrt(difference) / std::max(1.0e-30, std::sqrt(scale));
}

Vector initial_state(std::size_t n) {
    Vector state(n);
    for (std::size_t i = 0; i < n; ++i)
        // A deterministic broadband profile excites many Laplacian modes.
        // A single sine mode is an eigenvector and makes GMRES converge in
        // one iteration, which is not a useful scaling workload.
        state[i] = std::sin(12.9898 * static_cast<double>(i + 1))
                 * std::cos(78.233 * static_cast<double>(i + 1));
    return state;
}

void run_case(std::size_t n, bool compare_dense, bool check) {
    constexpr double h = 0.5;
    constexpr double theta = 0.7;
    asdole::GmresOptions options;
    options.relative_tolerance = 1.0e-11;
    options.absolute_tolerance = 1.0e-13;
    options.restart = std::min<std::size_t>(40, n);
    options.maximum_iterations = std::max<std::size_t>(100, 2 * n);
    const asdole::MatrixFreeStepper matrix_free(rhs, diffusion_jacobian_action, theta, options);
    const Vector initial = initial_state(n);

    const auto matrix_start = Clock::now();
    const auto matrix_result = matrix_free.step(initial, h);
    const auto matrix_stop = Clock::now();
    const double matrix_ms = std::chrono::duration<double, std::milli>(matrix_stop - matrix_start).count();

    double dense_ms = std::numeric_limits<double>::quiet_NaN();
    double error = std::numeric_limits<double>::quiet_NaN();
    if (compare_dense) {
        const asdole::Stepper dense(rhs, dense_jacobian, theta);
        const auto dense_start = Clock::now();
        const auto dense_result = dense.step(initial, h);
        const auto dense_stop = Clock::now();
        dense_ms = std::chrono::duration<double, std::milli>(dense_stop - dense_start).count();
        error = relative_error(matrix_result.step.state, dense_result.state);
        if (check && !(error < 2.0e-9))
            throw std::runtime_error("matrix-free and dense ASDoLE states disagree");
    }
    std::cout << n << ',' << matrix_ms << ',' << matrix_result.linear_iterations << ','
              << matrix_result.maximum_linear_residual << ',' << dense_ms << ',' << error << '\n';
}

int main_impl(int argc, char** argv) {
    const bool check = argc > 1 && std::string(argv[1]) == "--check";
    std::cout << "model=1d_discrete_laplacian theta=0.7 h=0.5\n"
              << "dimension,matrix_free_ms,gmres_iterations,max_linear_residual,dense_ms,relative_state_difference\n"
              << std::setprecision(8);
    if (check) {
        run_case(20, true, true);
        run_case(40, true, true);
    } else {
        run_case(32, true, false);
        run_case(64, true, false);
        run_case(128, true, false);
        run_case(256, true, false);
        run_case(512, false, false);
        run_case(1024, false, false);
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return main_impl(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "matrix_free_scaling: " << error.what() << '\n';
        return 2;
    }
}
