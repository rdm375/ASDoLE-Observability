// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/covariance.hpp"

#include <cmath>
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

void rhs(std::span<const double> state, std::span<double> value) {
    value[0] = -state[0];
    value[1] = state[0] - state[1];
}

void jacobian(std::span<const double>, DenseMatrix& matrix) {
    matrix(0, 0) = -1.0;
    matrix(0, 1) = 0.0;
    matrix(1, 0) = 1.0;
    matrix(1, 1) = -1.0;
}

void hessian(std::span<const double>, HessianTensor& tensor) {
    // The vector field is linear, so all second derivatives vanish.
    (void)tensor;
}

int main_impl(int argc, char** argv) {
    const double theta = argc > 1 ? std::stod(argv[1]) : 0.7;
    const int steps = argc > 2 ? std::stoi(argv[2]) : 80;
    constexpr double final_time = 2.0;
    constexpr double sigma_x0 = 0.1;
    constexpr double sigma_y0 = 0.1;
    if (steps <= 0) throw std::invalid_argument("step count must be positive");

    const double dt = final_time / static_cast<double>(steps);
    const asdole::Stepper stepper(rhs, jacobian, theta);
    std::vector<Vector> states(static_cast<std::size_t>(steps + 1));
    std::vector<double> step_sizes(static_cast<std::size_t>(steps), dt);
    states.front() = {1.0, 0.0};
    for (int k = 0; k < steps; ++k)
        states[static_cast<std::size_t>(k + 1)] = stepper.step(states[static_cast<std::size_t>(k)], dt).state;

    const asdole::DiscreteTrajectoryLinearization trajectory(
        stepper, hessian, states, std::move(step_sizes));
    DenseMatrix initial_covariance(2);
    initial_covariance(0, 0) = sigma_x0 * sigma_x0;
    initial_covariance(1, 1) = sigma_y0 * sigma_y0;
    const auto covariance = asdole::propagate_covariance(trajectory, initial_covariance);

    std::cout << std::setprecision(12)
              << "theta,steps,dt,time,x,y,var_x,cov_xy,var_y,sigma_x,sigma_y,correlation_xy,"
                 "exact_cov_xy,state_error,covariance_frobenius_error\n";
    for (int k = 0; k <= steps; ++k) {
        const auto index = static_cast<std::size_t>(k);
        const double time = dt * static_cast<double>(k);
        const auto& state = states[index];
        const auto& p = covariance[index];
        const auto diagnostic = asdole::diagnose_covariance(p);
        const double exact_cross = sigma_x0 * sigma_x0 * time * std::exp(-2.0 * time);
        const double decay2 = std::exp(-2.0 * time);
        const double exact_xx = sigma_x0 * sigma_x0 * decay2;
        const double exact_yy = decay2 * (time * time * sigma_x0 * sigma_x0 + sigma_y0 * sigma_y0);
        const double d00 = p(0, 0) - exact_xx;
        const double d01 = p(0, 1) - exact_cross;
        const double d10 = p(1, 0) - exact_cross;
        const double d11 = p(1, 1) - exact_yy;
        const double covariance_error = std::sqrt(d00*d00 + d01*d01 + d10*d10 + d11*d11);
        const double exact_x = std::exp(-time);
        const double exact_y = time * exact_x;
        const double state_error = std::hypot(state[0] - exact_x, state[1] - exact_y);
        std::cout << theta << ',' << steps << ',' << dt << ',' << time << ','
                  << state[0] << ',' << state[1] << ','
                  << p(0, 0) << ',' << p(0, 1) << ',' << p(1, 1) << ','
                  << diagnostic.standard_deviations[0] << ','
                  << diagnostic.standard_deviations[1] << ','
                  << diagnostic.correlation(0, 1) << ',' << exact_cross << ','
                  << state_error << ',' << covariance_error << '\n';
    }

    const auto cross = asdole::state_cross_covariance(
        trajectory, covariance, 0, static_cast<std::size_t>(steps));
    const auto obs_x = asdole::dense_observation_operator([] {
        DenseMatrix h(1, 2);
        h(0, 0) = 1.0;
        return h;
    }());
    const auto obs_y = asdole::dense_observation_operator([] {
        DenseMatrix h(1, 2);
        h(0, 1) = 1.0;
        return h;
    }());
    const auto cross_observation = asdole::observation_cross_covariance(obs_x, cross, obs_y);
    std::cerr << "# final_time=" << final_time << " theta=" << theta
              << " steps=" << steps
              << " Cov(x(0),y(T))=" << cross_observation(0, 0) << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "covariance_transport: " << error.what() << '\n';
        return 2;
    }
}
