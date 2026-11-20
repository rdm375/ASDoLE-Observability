// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/differentiation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {
constexpr double initial_state = 0.3;
constexpr double observation = 0.15;
constexpr double final_time = 1.0;
constexpr int steps = 80;
constexpr double finite_difference_step = 1.0e-6;

asdole::Stepper make_stepper(double mu, double theta) {
    const auto rhs = [mu](std::span<const double> y, std::span<double> f) {
        f[0] = mu * y[0] * y[0];
    };
    const auto jacobian = [mu](std::span<const double> y, asdole::DenseMatrix& j) {
        j(0, 0) = 2.0 * mu * y[0];
    };
    return asdole::Stepper(rhs, jacobian, theta);
}

void hessian(double mu, std::span<const double>, asdole::HessianTensor& h) {
    h[0](0, 0) = 2.0 * mu;
}

asdole::Vector rhs_parameter_derivative(std::span<const double> y) {
    return {y[0] * y[0]};
}

asdole::DenseMatrix jacobian_parameter_derivative(std::span<const double> y) {
    asdole::DenseMatrix result(1);
    result(0, 0) = 2.0 * y[0];
    return result;
}

struct Trajectory {
    std::vector<double> state;
    double objective{};
};

Trajectory evaluate(double mu, double theta) {
    const asdole::Stepper stepper = make_stepper(mu, theta);
    const double dt = final_time / steps;
    Trajectory result;
    result.state.resize(steps + 1);
    result.state[0] = initial_state;
    for (int k = 0; k < steps; ++k) {
        result.state[k + 1] = stepper.step(asdole::Vector{result.state[k]}, dt).state[0];
    }
    for (int k = 0; k <= steps; ++k) {
        const double weight = (k == 0 || k == steps) ? 0.5 : 1.0;
        const double residual = result.state[k] - observation;
        result.objective += 0.5 * dt * weight * residual * residual;
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    const double theta = argc > 1 ? std::stod(argv[1]) : 0.7;
    constexpr double mu = 0.8;
    const double dt = final_time / steps;
    const asdole::Stepper stepper = make_stepper(mu, theta);
    std::vector<double> state(steps + 1), tangent(steps + 1), adjoint(steps + 1);
    std::vector<double> step_derivative(steps), parameter_derivative(steps);
    state[0] = initial_state;
    tangent[0] = 0.0;

    double tangent_objective_gradient = 0.0;
    for (int k = 0; k < steps; ++k) {
        const asdole::Vector y{state[k]};
        const asdole::Vector f_mu = rhs_parameter_derivative(y);
        const asdole::DenseMatrix j_mu = jacobian_parameter_derivative(y);
        const asdole::HessianFunction hessian_function = [mu](std::span<const double> x,
                                                               asdole::HessianTensor& h) {
            hessian(mu, x, h);
        };
        state[k + 1] = stepper.step(y, dt).state[0];
        const double propagated = asdole::tangent_step(stepper, hessian_function, y, dt,
                                                       asdole::Vector{tangent[k]})[0];
        tangent[k + 1] = propagated
            + asdole::parameter_tangent_step(stepper, y, dt, f_mu, j_mu)[0];
        step_derivative[k] = asdole::tangent_step(stepper, hessian_function, y, dt,
                                                  asdole::Vector{1.0})[0];
        parameter_derivative[k] = asdole::parameter_tangent_step(stepper, y, dt, f_mu, j_mu)[0];
    }

    double objective = 0.0;
    for (int k = 0; k <= steps; ++k) {
        const double weight = (k == 0 || k == steps) ? 0.5 : 1.0;
        const double residual = state[k] - observation;
        objective += 0.5 * dt * weight * residual * residual;
        tangent_objective_gradient += dt * weight * residual * tangent[k];
    }

    // Discrete adjoint for the same trapezoidal objective and ASDoLE step map.
    adjoint[steps] = 0.5 * dt * (state[steps] - observation);
    double adjoint_parameter_gradient = 0.0;
    for (int k = steps - 1; k >= 0; --k) {
        adjoint_parameter_gradient += adjoint[k + 1] * parameter_derivative[k];
        const double weight = (k == 0) ? 0.5 : 1.0;
        adjoint[k] = dt * weight * (state[k] - observation)
                   + step_derivative[k] * adjoint[k + 1];
    }

    const double objective_plus = evaluate(mu + finite_difference_step, theta).objective;
    const double objective_minus = evaluate(mu - finite_difference_step, theta).objective;
    const double finite_difference_gradient =
        (objective_plus - objective_minus) / (2.0 * finite_difference_step);
    const double gradient_fd_error = std::abs(adjoint_parameter_gradient - finite_difference_gradient)
                                   / std::max(1.0, std::abs(finite_difference_gradient));
    const double tlm_fd_error = std::abs(tangent_objective_gradient - finite_difference_gradient)
                              / std::max(1.0, std::abs(finite_difference_gradient));
    const double tlm_adjoint_defect = std::abs(tangent_objective_gradient - adjoint_parameter_gradient);

    constexpr double finite_difference_tolerance = 2.0e-8;
    constexpr double adjoint_tolerance = 2.0e-12;
    const bool passed = gradient_fd_error < finite_difference_tolerance
        && tlm_fd_error < finite_difference_tolerance
        && tlm_adjoint_defect < adjoint_tolerance;
    std::cout << std::setprecision(12)
              << "theta=" << theta << " mu=" << mu << " steps=" << steps << '\n'
              << "objective=" << objective << '\n'
              << "parameter_tlm_gradient=" << tangent_objective_gradient << '\n'
              << "parameter_adjoint_gradient=" << adjoint_parameter_gradient << '\n'
              << "parameter_finite_difference_gradient=" << finite_difference_gradient << '\n'
              << "parameter_tlm_adjoint_defect=" << tlm_adjoint_defect << '\n'
              << "parameter_adjoint_vs_fd_relative_error=" << gradient_fd_error << '\n'
              << "parameter_tlm_vs_fd_relative_error=" << tlm_fd_error << '\n'
              << (passed ? "result=PASS\n" : "result=FAIL\n");
    return passed ? 0 : 1;
}
