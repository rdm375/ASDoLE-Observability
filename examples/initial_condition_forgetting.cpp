// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/covariance.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace asdole;
constexpr double final_time = 12.0;
constexpr int steps = 480;
constexpr double dt = final_time / steps;
constexpr double sigma_x0 = 0.10;
constexpr double sigma_y0 = 0.08;
constexpr double observation_sigma = 0.01;
constexpr double theta = 0.7;

void rhs(std::span<const double> state, std::span<double> value) {
    value[0] = -state[0];
    value[1] = state[0] - state[1];
}

void jacobian(std::span<const double>, DenseMatrix& matrix) {
    matrix(0, 0) = -1.0; matrix(0, 1) = 0.0;
    matrix(1, 0) = 1.0; matrix(1, 1) = -1.0;
}

void zero_hessian(std::span<const double>, HessianTensor&) {}

double fitted_decay_rate(const std::vector<double>& values,
                         const std::vector<double>& times,
                         double first_time,
                         double last_time) {
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (times[i] < first_time || times[i] > last_time || !(values[i] > 0.0)) continue;
        const double x = times[i], y = std::log(values[i]);
        sx += x; sy += y; sxx += x*x; sxy += x*y; ++count;
    }
    const double n = static_cast<double>(count);
    const double slope = (n*sxy - sx*sy) / (n*sxx - sx*sx);
    return -slope;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string output = argc > 1
            ? argv[1] : "results/initial-condition-forgetting/forgetting.csv";
        const std::filesystem::path output_path(output);
        if (output_path.has_parent_path())
            std::filesystem::create_directories(output_path.parent_path());

        const Stepper stepper(rhs, jacobian, theta);
        std::vector<Vector> states(static_cast<std::size_t>(steps + 1));
        std::vector<double> step_sizes(static_cast<std::size_t>(steps), dt);
        states.front() = {1.0, 0.7};
        for (int k = 0; k < steps; ++k)
            states[static_cast<std::size_t>(k + 1)] = stepper.step(
                states[static_cast<std::size_t>(k)], dt).state;
        const DiscreteTrajectoryLinearization trajectory(
            stepper, zero_hessian, states, std::move(step_sizes));

        DenseMatrix initial_covariance(2);
        initial_covariance(0, 0) = sigma_x0 * sigma_x0;
        initial_covariance(1, 1) = sigma_y0 * sigma_y0;
        const auto covariance = propagate_covariance(trajectory, initial_covariance);
        const auto gain = transient_amplification_history(trajectory);

        // A single initial perturbation illustrates component transfer. The
        // covariance uses independent random initial errors with the sigmas above.
        const Vector initial_error{0.10, 0.04};
        const Vector basis_x{1.0, 0.0};
        const Vector basis_y{0.0, 1.0};
        DenseMatrix observation(1, 2);
        observation(0, 1) = 1.0;
        const auto sensor = dense_observation_operator(observation);
        DenseMatrix information(2);
        std::vector<double> times(static_cast<std::size_t>(steps + 1));
        std::vector<double> error_norms(times.size()), covariance_traces(times.size());
        std::vector<double> local_error_rates(times.size(), 0.0);
        std::vector<double> local_covariance_rates(times.size(), 0.0);

        std::ofstream csv(output);
        if (!csv) throw std::runtime_error("could not open output CSV: " + output);
        csv << std::setprecision(12)
            << "time,error_x,error_y,error_norm,tlm_gain_2norm,var_x,cov_xy,var_y,"
               "trace_covariance,sigma_x,sigma_y,correlation_xy,"
               "local_error_decay_rate,local_covariance_decay_rate,"
               "information_xx,information_xy,information_yy,"
               "information_eigen_min,information_eigen_max,information_rate_y\n";

        for (int k = 0; k <= steps; ++k) {
            const auto node = static_cast<std::size_t>(k);
            const double time = node * dt;
            times[node] = time;
            const Vector error = trajectory.apply(initial_error, node);
            const auto& p = covariance[node];
            const double trace = p(0,0) + p(1,1);
            error_norms[node] = metric_norm(error);
            covariance_traces[node] = trace;

            if (k > 0) {
                local_error_rates[node] = -std::log(error_norms[node] / error_norms[node-1]) / dt;
                local_covariance_rates[node] = -std::log(trace / covariance_traces[node-1]) / dt;
            }

            const Vector row = trajectory.apply(basis_x, node);
            const Vector col = trajectory.apply(basis_y, node);
            const double quadrature = dt * ((k == 0 || k == steps) ? 0.5 : 1.0);
            const double sensitivity_x = sensor(row)[0];
            const double sensitivity_y = sensor(col)[0];
            const double precision_weight = quadrature / (observation_sigma * observation_sigma);
            information(0,0) += precision_weight * sensitivity_x * sensitivity_x;
            information(0,1) += precision_weight * sensitivity_x * sensitivity_y;
            information(1,0) = information(0,1);
            information(1,1) += precision_weight * sensitivity_y * sensitivity_y;
            const auto information_eigenvalues = symmetric_eigenvalues(information);
            const double rho_denominator = std::sqrt(std::max(0.0, p(0,0)*p(1,1)));
            const double rho = rho_denominator > 0.0 ? p(0,1)/rho_denominator : 0.0;
            const double information_rate =
                (sensitivity_x*sensitivity_x + sensitivity_y*sensitivity_y)
                / (observation_sigma*observation_sigma);

            csv << time << ',' << error[0] << ',' << error[1] << ','
                << error_norms[node] << ',' << gain[node].gain << ','
                << p(0,0) << ',' << p(0,1) << ',' << p(1,1) << ',' << trace << ','
                << std::sqrt(std::max(0.0,p(0,0))) << ','
                << std::sqrt(std::max(0.0,p(1,1))) << ',' << rho << ','
                << local_error_rates[node] << ',' << local_covariance_rates[node] << ','
                << information(0,0) << ',' << information(0,1) << ','
                << information(1,1) << ',' << information_eigenvalues.back() << ','
                << information_eigenvalues.front() << ',' << information_rate << '\n';
        }

        const double fit_start = 6.0;
        const double error_rate = fitted_decay_rate(error_norms, times, fit_start, final_time);
        const double covariance_rate = fitted_decay_rate(
            covariance_traces, times, fit_start, final_time);
        const auto& final_covariance = covariance.back();
        const double final_sigma_x = std::sqrt(final_covariance(0,0));
        const double final_sigma_y = std::sqrt(final_covariance(1,1));
        const double final_correlation = final_covariance(0,1)
            / (final_sigma_x * final_sigma_y);
        std::cout << std::setprecision(12)
                  << "model=x'=-x, y'=x-y theta=" << theta
                  << " steps=" << steps << " dt=" << dt
                  << " final_time=" << final_time
                  << "\ninitial_error=[0.1,0.04]"
                  << " initial_sigmas=[" << sigma_x0 << ',' << sigma_y0 << ']'
                  << "\nlate_window=" << fit_start << ".." << final_time
                  << " fitted_error_decay_rate=" << error_rate
                  << " fitted_covariance_trace_decay_rate=" << covariance_rate
                  << "\nfinal_error_norm=" << error_norms.back()
                  << " final_covariance_trace=" << covariance_traces.back()
                  << " final_sigma_x=" << final_sigma_x
                  << " final_sigma_y=" << final_sigma_y
                  << " final_correlation_xy=" << final_correlation
                  << "\nfinal_information_eigenvalues=["
                  << symmetric_eigenvalues(information).back() << ','
                  << symmetric_eigenvalues(information).front() << ']'
                  << "\noutput_csv=" << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "initial_condition_forgetting: " << error.what() << '\n';
        return 2;
    }
}
