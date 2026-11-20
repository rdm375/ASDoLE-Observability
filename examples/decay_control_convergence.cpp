// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/stepper.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

struct ControlErrors {
    double state{};
    double tangent{};
    double costate{};
    double objective{};
    double adjoint_identity{};
};

struct ControlTrajectory {
    std::vector<double> state;
    std::vector<double> tangent;
    std::vector<double> costate;
    ControlErrors errors;
};

ControlTrajectory solve_control(double theta, int steps) {
    constexpr double decay_rate = 0.8;
    constexpr double initial_state = 1.25;
    constexpr double observation_initial_state = 0.8;
    constexpr double final_time = 4.0;
    const double h = final_time / steps;

    const auto rhs = [](std::span<const double> y, std::span<double> f) {
        f[0] = -decay_rate * y[0];
    };
    const auto jacobian = [](std::span<const double>, asdole::DenseMatrix& j) {
        j(0, 0) = -decay_rate;
    };
    const asdole::Stepper stepper(rhs, jacobian, theta);

    ControlTrajectory result;
    result.state.resize(steps + 1);
    result.tangent.resize(steps + 1);
    result.costate.resize(steps + 1);
    std::vector<double> step_jacobian(steps);
    result.state[0] = initial_state;
    result.tangent[0] = 1.0;

    for (int k = 0; k < steps; ++k) {
        const asdole::Vector y{result.state[k]};
        const asdole::Vector perturbation{result.tangent[k]};
        const auto state_step = stepper.step(y, h);
        const auto tangent_step = stepper.step(perturbation, h);
        result.state[k + 1] = state_step.state[0];
        result.tangent[k + 1] = tangent_step.state[0];
        step_jacobian[k] = stepper.step(asdole::Vector{1.0}, h).state[0];
    }

    const auto observation = [=](double t) {
        return observation_initial_state * std::exp(-decay_rate * t);
    };
    const auto running_cost_gradient = [&](int k) {
        const double t = h * k;
        return 2.0 * (result.state[k] - observation(t));
    };

    // Composite trapezoidal quadrature on the stored discrete state nodes.
    // The exact discrete adjoint below differentiates this same functional.
    double objective = 0.0;
    double tangent_gradient = 0.0;
    for (int k = 0; k <= steps; ++k) {
        const double weight = (k == 0 || k == steps) ? 0.5 : 1.0;
        const double difference = result.state[k] - observation(h * k);
        objective += h * weight * difference * difference;
        tangent_gradient += h * weight * running_cost_gradient(k) * result.tangent[k];
    }

    // Reverse the discrete state map. The step Jacobian is the derivative of
    // the ASDoLE map, and the quadrature gradient enters at the same nodes.
    result.costate[steps] = 0.5 * h * running_cost_gradient(steps);
    for (int k = steps - 1; k >= 0; --k) {
        const double weight = k == 0 ? 0.5 : 1.0;
        result.costate[k] = h * weight * running_cost_gradient(k)
                          + step_jacobian[k] * result.costate[k + 1];
    }

    // The raw discrete multiplier at an interior node includes the full
    // objective load h*ell_y(y_k), while the continuous costate is a
    // one-sided-in-time quantity. Remove the half-load belonging to the
    // preceding trapezoid interval. At k=0, keep the multiplier: it is the
    // derivative with respect to the initial-state control.
    for (int k = 1; k <= steps; ++k) {
        result.costate[k] -= 0.5 * h * running_cost_gradient(k);
    }

    const double difference_initial = initial_state - observation_initial_state;
    const double exact_objective = difference_initial * difference_initial
        * (1.0 - std::exp(-2.0 * decay_rate * final_time)) / (2.0 * decay_rate);

    for (int k = 0; k <= steps; ++k) {
        const double t = h * k;
        const double exact_state = initial_state * std::exp(-decay_rate * t);
        const double exact_tangent = std::exp(-decay_rate * t);
        const double exact_costate = difference_initial / decay_rate
            * (std::exp(-decay_rate * t)
               - std::exp(decay_rate * t - 2.0 * decay_rate * final_time));
        result.errors.state = std::max(result.errors.state, std::abs(result.state[k] - exact_state));
        result.errors.tangent = std::max(result.errors.tangent, std::abs(result.tangent[k] - exact_tangent));
        result.errors.costate = std::max(result.errors.costate, std::abs(result.costate[k] - exact_costate));
    }
    result.errors.objective = std::abs(objective - exact_objective);
    result.errors.adjoint_identity = std::abs(tangent_gradient - result.costate[0]);
    return result;
}

int main() {
    constexpr int refinement_levels = 8;
    constexpr std::array theta_values{0.5, 0.6, 0.7, 1.0};
    std::cout << "theta,dt,steps,state_error,tlm_error,costate_error,objective_error,"
                 "state_order,tlm_order,costate_order,objective_order,adjoint_identity_error\n";
    std::cout << std::setprecision(12);

    for (const double theta : theta_values) {
        ControlErrors previous{};
        for (int level = 0; level < refinement_levels; ++level) {
            const int steps = 4 * (1 << level);
            const double dt = 4.0 / steps;
            const ControlTrajectory trajectory = solve_control(theta, steps);
            const ControlErrors& e = trajectory.errors;
            std::cout << theta << ',' << dt << ',' << steps << ','
                      << e.state << ',' << e.tangent << ',' << e.costate << ',' << e.objective << ',';
            if (level == 0) {
                std::cout << "NA,NA,NA,NA,";
            } else {
                std::cout << std::log2(previous.state / e.state) << ','
                          << std::log2(previous.tangent / e.tangent) << ','
                          << std::log2(previous.costate / e.costate) << ','
                          << std::log2(previous.objective / e.objective) << ',';
            }
            std::cout << e.adjoint_identity << '\n';
            previous = e;
        }
    }
}
