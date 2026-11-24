// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/covariance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using asdole::DenseMatrix;
using asdole::HessianTensor;
using asdole::Vector;

constexpr std::size_t ensemble_size = 30000;
constexpr int step_count = 60;
constexpr double final_time = 3.0;

void rhs(std::span<const double> y, std::span<double> f) {
    f[0] = -y[0] + 0.15 * y[1] * y[1];
    f[1] = y[0] - 1.2 * y[1] - 0.1 * y[1] * y[1] * y[1];
}

void jacobian(std::span<const double> y, DenseMatrix& j) {
    j(0, 0) = -1.0; j(0, 1) = 0.3 * y[1];
    j(1, 0) = 1.0;  j(1, 1) = -1.2 - 0.3 * y[1] * y[1];
}

void hessian(std::span<const double> y, HessianTensor& h) {
    h[0](1, 1) = 0.3;
    h[1](1, 1) = -0.6 * y[1];
}

DenseMatrix sample_covariance(const std::vector<Vector>& samples, std::span<const double> mean) {
    DenseMatrix result(mean.size());
    for (const auto& sample : samples)
        for (std::size_t i = 0; i < mean.size(); ++i)
            for (std::size_t j = 0; j < mean.size(); ++j)
                result(i, j) += (sample[i] - mean[i]) * (sample[j] - mean[j]);
    for (std::size_t i = 0; i < mean.size(); ++i)
        for (std::size_t j = 0; j < mean.size(); ++j)
            result(i, j) /= static_cast<double>(samples.size() - 1);
    return result;
}

double relative_error(const DenseMatrix& a, const DenseMatrix& b) {
    double numerator = 0.0, denominator = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) {
            numerator += (a(i, j) - b(i, j)) * (a(i, j) - b(i, j));
            denominator += b(i, j) * b(i, j);
        }
    return std::sqrt(numerator) / std::max(1.0e-30, std::sqrt(denominator));
}

void run_case(double sigma, std::uint64_t seed, bool check) {
    if (!(sigma > 0.0) || !std::isfinite(sigma))
        throw std::invalid_argument("initial standard deviation must be finite and positive");
    constexpr double theta = 0.7;
    const double dt = final_time / step_count;
    const asdole::Stepper stepper(rhs, jacobian, theta);
    std::vector<Vector> states(step_count + 1);
    std::vector<double> steps(step_count, dt);
    states[0] = {1.0, 0.6};
    for (int k = 0; k < step_count; ++k)
        states[static_cast<std::size_t>(k + 1)] = stepper.step(states[static_cast<std::size_t>(k)], dt).state;
    const asdole::DiscreteTrajectoryLinearization trajectory(stepper, hessian, states, std::move(steps));

    DenseMatrix initial_covariance(2);
    initial_covariance(0, 0) = sigma * sigma;
    initial_covariance(1, 1) = sigma * sigma;
    const DenseMatrix predicted = asdole::propagate_covariance(trajectory, initial_covariance).back();

    std::mt19937_64 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Vector> terminal_states;
    terminal_states.reserve(ensemble_size);
    Vector ensemble_mean(2, 0.0);
    for (std::size_t sample = 0; sample < ensemble_size; ++sample) {
        Vector y{states.front()[0] + sigma * normal(generator),
                 states.front()[1] + sigma * normal(generator)};
        for (int k = 0; k < step_count; ++k) y = stepper.step(y, dt).state;
        for (std::size_t i = 0; i < 2; ++i) ensemble_mean[i] += y[i];
        terminal_states.push_back(std::move(y));
    }
    for (double& value : ensemble_mean) value /= static_cast<double>(ensemble_size);
    const DenseMatrix empirical = sample_covariance(terminal_states, ensemble_mean);
    const double error = relative_error(empirical, predicted);
    const double mean_bias = std::hypot(ensemble_mean[0] - states.back()[0],
                                        ensemble_mean[1] - states.back()[1]);

    std::cout << std::setprecision(12)
              << "model=x'=-x+0.15y^2, y'=x-1.2y-0.1y^3\n"
              << "theta=" << theta << " steps=" << step_count << " dt=" << dt
              << " sigma_initial=" << sigma << " ensemble_size=" << ensemble_size
              << " seed=" << seed << '\n'
              << "linearized_covariance=[[" << predicted(0, 0) << ',' << predicted(0, 1)
              << "],[" << predicted(1, 0) << ',' << predicted(1, 1) << "]]\n"
              << "ensemble_covariance=[[" << empirical(0, 0) << ',' << empirical(0, 1)
              << "],[" << empirical(1, 0) << ',' << empirical(1, 1) << "]]\n"
              << "relative_covariance_error=" << error << " ensemble_mean_bias=" << mean_bias << '\n';
    if (check && !(error < 0.04)) throw std::runtime_error("nonlinear ensemble covariance differs from TLM prediction");
}

int main_impl(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--sweep") {
        for (double sigma : {0.01, 0.1, 0.5, 1.0})
            run_case(sigma, 20261123, false);
    } else if (argc > 1 && std::string(argv[1]) == "--sigma") {
        if (argc < 3) throw std::invalid_argument("--sigma requires a standard deviation");
        run_case(std::stod(argv[2]), 20261123, false);
    } else {
        const bool check = argc > 1 && std::string(argv[1]) == "--check";
        run_case(0.01, 20261123, check);
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return main_impl(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "nonlinear_covariance_ensemble: " << error.what() << '\n';
        return 2;
    }
}
