// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/covariance.hpp"

#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using asdole::DenseMatrix;
using asdole::HessianTensor;
using asdole::Vector;

constexpr double final_time = 4.0;
constexpr int steps = 160;
constexpr int observation_stride = 8;
constexpr double dt = final_time / steps;
constexpr double sigma_x0 = 0.08;
constexpr double sigma_y0 = 0.05;
constexpr double measurement_sigma = 0.01;
constexpr double true_bias = 0.015;

void rhs(std::span<const double> state, std::span<double> value) {
    value[0] = -state[0];
    value[1] = state[0] - state[1];
}

void jacobian(std::span<const double>, DenseMatrix& matrix) {
    matrix(0, 0) = -1.0; matrix(0, 1) = 0.0;
    matrix(1, 0) = 1.0; matrix(1, 1) = -1.0;
}

void zero_hessian(std::span<const double>, HessianTensor&) {}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string output_path = argc > 1
            ? argv[1] : "results/observation-gls/observation_gls.csv";
        const std::filesystem::path output_file(output_path);
        if (output_file.has_parent_path())
            std::filesystem::create_directories(output_file.parent_path());
        const asdole::Stepper stepper(rhs, jacobian, 0.7);
        std::vector<Vector> states(static_cast<std::size_t>(steps + 1));
        std::vector<double> step_sizes(static_cast<std::size_t>(steps), dt);
        states.front() = {1.0, 0.7};
        for (int k = 0; k < steps; ++k)
            states[static_cast<std::size_t>(k + 1)] = stepper.step(
                states[static_cast<std::size_t>(k)], dt).state;
        const asdole::DiscreteTrajectoryLinearization trajectory(
            stepper, zero_hessian, states, std::move(step_sizes));

        DenseMatrix initial_covariance(2);
        initial_covariance(0, 0) = sigma_x0 * sigma_x0;
        initial_covariance(1, 1) = sigma_y0 * sigma_y0;
        const auto state_covariance = asdole::propagate_covariance(
            trajectory, initial_covariance);

        DenseMatrix sensor_matrix(1, 2);
        sensor_matrix(0, 1) = 1.0;
        const auto sensor = asdole::dense_observation_operator(sensor_matrix);
        std::vector<asdole::ObservationCovarianceSample> samples;
        std::vector<double> times;
        for (int k = 0; k <= steps; k += observation_stride) {
            samples.push_back({static_cast<std::size_t>(k), sensor});
            times.push_back(k * dt);
        }
        const std::size_t count = samples.size();
        DenseMatrix measurement_covariance(count);
        for (std::size_t i = 0; i < count; ++i)
            measurement_covariance(i, i) = measurement_sigma * measurement_sigma;
        const DenseMatrix observation_covariance = asdole::stacked_observation_covariance(
            trajectory, state_covariance, samples, measurement_covariance);

        // Synthetic residuals contain a constant sensor bias, one realization
        // of the uncertain initial state, and a small deterministic sensor-noise
        // pattern. The bias is independent of the dynamical state uncertainty.
        const Vector initial_error{0.06, -0.035};
        DenseMatrix design(count, 1);
        Vector residual_data(count);
        for (std::size_t i = 0; i < count; ++i) {
            design(i, 0) = 1.0;
            const auto state_error = trajectory.apply(initial_error, samples[i].node);
            const double measurement_error = measurement_sigma *
                (0.7 * std::sin(0.83 * static_cast<double>(i))
                 + 0.3 * std::cos(0.37 * static_cast<double>(i)));
            residual_data[i] = true_bias + state_error[1] + measurement_error;
        }

        const auto gls = asdole::generalized_least_squares(
            design, residual_data, observation_covariance);
        DenseMatrix white_covariance(count);
        for (std::size_t i = 0; i < count; ++i)
            white_covariance(i, i) = measurement_sigma * measurement_sigma;
        const auto ordinary = asdole::generalized_least_squares(
            design, residual_data, white_covariance);

        std::ofstream csv(output_path);
        if (!csv) throw std::runtime_error("could not open output CSV: " + output_path);
        csv << std::setprecision(12)
            << "time,residual,state_variance,observation_variance,covariance_with_first,"
               "correlation_with_first\n";
        for (std::size_t i = 0; i < count; ++i) {
            const double covariance_with_first = observation_covariance(0, i);
            const double denominator = std::sqrt(observation_covariance(0, 0)
                                                  * observation_covariance(i, i));
            csv << times[i] << ',' << residual_data[i] << ','
                << state_covariance[samples[i].node](1, 1) << ','
                << observation_covariance(i, i) << ',' << covariance_with_first << ','
                << (denominator > 0.0 ? covariance_with_first / denominator : 0.0) << '\n';
        }

        std::cout << std::setprecision(12)
                  << "model=x'=-x, y'=x-y; observation= y + sensor_bias"
                  << "\nobservations=" << count
                  << " state_uncertainty=independent_initial_x_y"
                  << " measurement_noise=independent_white"
                  << "\ntrue_bias=" << true_bias
                  << "\nordinary_least_squares_bias=" << ordinary.parameters[0]
                  << " ordinary_formal_sigma="
                  << std::sqrt(ordinary.parameter_covariance(0, 0))
                  << " ordinary_objective=" << ordinary.objective
                  << "\ngls_bias=" << gls.parameters[0]
                  << " gls_formal_sigma="
                  << std::sqrt(gls.parameter_covariance(0, 0))
                  << " gls_objective=" << gls.objective
                  << "\nobservation_covariance_shape=" << count << 'x' << count
                  << "\nwithin_state_covariance_range=" << observation_covariance(0, count - 1)
                  << "\noutput_csv=" << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "observation_gls: " << error.what() << '\n';
        return 2;
    }
}
