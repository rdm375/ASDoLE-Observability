// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/covariance.hpp"

#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using asdole::DenseMatrix;
using asdole::HessianTensor;
using asdole::Vector;

constexpr double coupling = 0.8;

void rhs(std::span<const double> y, std::span<double> f) {
    f[0] = -y[0] + coupling * y[1];
    f[1] = -2.0 * y[1];
}

void jacobian(std::span<const double>, DenseMatrix& j) {
    j(0, 0) = -1.0;
    j(0, 1) = coupling;
    j(1, 0) = 0.0;
    j(1, 1) = -2.0;
}

void hessian(std::span<const double>, HessianTensor&) {}

int main_impl(int argc, char** argv) {
    const double theta = argc > 1 ? std::stod(argv[1]) : 0.7;
    constexpr int steps = 80;
    constexpr double final_time = 4.0;
    constexpr double dt = final_time / steps;
    const asdole::Stepper stepper(rhs, jacobian, theta);

    std::vector<Vector> states(steps + 1);
    std::vector<double> step_sizes(steps, dt);
    states[0] = {1.0, 0.7};
    for (int k = 0; k < steps; ++k)
        states[static_cast<std::size_t>(k + 1)] = stepper.step(states[static_cast<std::size_t>(k)], dt).state;
    const asdole::DiscreteTrajectoryLinearization trajectory(stepper, hessian, states, std::move(step_sizes));

    DenseMatrix jacobian_parameter_derivative(2);
    jacobian_parameter_derivative(0, 1) = 1.0;
    std::vector<DenseMatrix> parameter_steps;
    parameter_steps.reserve(steps);
    for (int k = 0; k < steps; ++k) {
        const auto& state = states[static_cast<std::size_t>(k)];
        const Vector rhs_parameter_derivative{state[1], 0.0};
        const Vector b = asdole::parameter_tangent_step(
            stepper, state, dt, rhs_parameter_derivative, jacobian_parameter_derivative);
        DenseMatrix step_sensitivity(2, 1);
        step_sensitivity(0, 0) = b[0];
        step_sensitivity(1, 0) = b[1];
        parameter_steps.push_back(std::move(step_sensitivity));
    }

    // Joint random vector is [x(0), y(0), k]. These priors are independent.
    DenseMatrix initial_joint_covariance(3);
    initial_joint_covariance(0, 0) = 0.05 * 0.05;
    initial_joint_covariance(1, 1) = 0.05 * 0.05;
    initial_joint_covariance(2, 2) = 0.04 * 0.04;
    const auto joint_covariances = asdole::propagate_joint_state_parameter_covariance(
        trajectory, initial_joint_covariance, parameter_steps);

    std::cout << std::setprecision(12)
              << "theta,steps,dt,time,x,y,var_x,cov_xy,var_y,cov_xk,cov_yk,var_k\n";
    for (int k = 0; k <= steps; ++k) {
        const auto index = static_cast<std::size_t>(k);
        const auto& p = joint_covariances[index];
        std::cout << theta << ',' << steps << ',' << dt << ',' << dt * k << ','
                  << states[index][0] << ',' << states[index][1] << ','
                  << p(0, 0) << ',' << p(0, 1) << ',' << p(1, 1) << ','
                  << p(0, 2) << ',' << p(1, 2) << ',' << p(2, 2) << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "parameter_covariance_transport: " << error.what() << '\n';
        return 2;
    }
}
