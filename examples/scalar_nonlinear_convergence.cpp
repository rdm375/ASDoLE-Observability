// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/differentiation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {
constexpr double y0 = 0.3;
constexpr double observation = 0.15;
constexpr double final_time = 1.0;

void rhs(std::span<const double> y, std::span<double> f) {
    f[0] = y[0] * y[0];
}

void jacobian(std::span<const double> y, asdole::DenseMatrix& j) {
    j(0, 0) = 2.0 * y[0];
}

void hessian(std::span<const double>, asdole::HessianTensor& h) {
    h[0](0, 0) = 2.0;
}

double exact_state(double t) {
    return y0 / (1.0 - y0 * t);
}

double exact_tangent(double t) {
    const double denominator = 1.0 - y0 * t;
    return 1.0 / (denominator * denominator);
}

double exact_costate(double t) {
    const double denominator = 1.0 - y0 * t;
    const double terminal_denominator = 1.0 - y0 * final_time;
    return -0.5 + (observation / y0) * denominator
        + 0.5 * std::pow(denominator / terminal_denominator, 2)
        - (observation / y0) * denominator * denominator / terminal_denominator;
}

double exact_objective() {
    const double denominator = 1.0 - y0 * final_time;
    const double integral = y0 * (1.0 / denominator - 1.0)
        + 2.0 * observation * std::log(denominator)
        + observation * observation * final_time;
    return 0.5 * integral;
}

struct Errors {
    double state{};
    double tangent{};
    double costate{};
    double objective{};
    double adjoint_identity{};
};

Errors solve(double theta, int steps) {
    const double h = final_time / steps;
    const asdole::Stepper stepper(rhs, jacobian, theta);
    std::vector<double> state(steps + 1);
    std::vector<double> tangent(steps + 1);
    std::vector<double> costate(steps + 1);
    std::vector<double> step_derivative(steps);
    state[0] = y0;
    tangent[0] = 1.0;

    for (int k = 0; k < steps; ++k) {
        const asdole::Vector y{state[k]};
        state[k + 1] = stepper.step(y, h).state[0];
        tangent[k + 1] = asdole::tangent_step(stepper, hessian, y, h,
                                               asdole::Vector{tangent[k]})[0];
        step_derivative[k] = asdole::tangent_step(stepper, hessian, y, h,
                                                   asdole::Vector{1.0})[0];
    }

    double objective = 0.0;
    double tlm_objective_gradient = 0.0;
    for (int k = 0; k <= steps; ++k) {
        const double weight = (k == 0 || k == steps) ? 0.5 : 1.0;
        const double residual = state[k] - observation;
        objective += 0.5 * h * weight * residual * residual;
        tlm_objective_gradient += h * weight * residual * tangent[k];
    }

    // Reverse the discrete trapezoidal objective through the exact derivative
    // of each nonlinear ASDoLE state map.
    const double terminal_gradient = state[steps] - observation;
    costate[steps] = 0.5 * h * terminal_gradient;
    for (int k = steps - 1; k >= 0; --k) {
        const double weight = (k == 0) ? 0.5 : 1.0;
        costate[k] = h * weight * (state[k] - observation)
                   + step_derivative[k] * costate[k + 1];
    }
    // Convert the discrete multiplier to the nodal continuous-costate convention.
    for (int k = 1; k <= steps; ++k) {
        costate[k] -= 0.5 * h * (state[k] - observation);
    }

    Errors e;
    for (int k = 0; k <= steps; ++k) {
        const double t = h * k;
        e.state = std::max(e.state, std::abs(state[k] - exact_state(t)));
        e.tangent = std::max(e.tangent, std::abs(tangent[k] - exact_tangent(t)));
        e.costate = std::max(e.costate, std::abs(costate[k] - exact_costate(t)));
    }
    e.objective = std::abs(objective - exact_objective());
    e.adjoint_identity = std::abs(tlm_objective_gradient - costate[0]);
    return e;
}
} // namespace

int main() {
    constexpr std::array theta_values{0.5, 0.6, 0.7, 1.0};
    constexpr int levels = 8;
    constexpr double minimum_order = 1.90;
    constexpr double adjoint_identity_tolerance = 1.0e-11;
    bool passed = true;
    std::cout << std::setprecision(12)
              << "theta,dt,steps,state_error,tlm_error,costate_error,objective_error,"
                 "state_order,tlm_order,costate_order,objective_order,adjoint_identity_error\n";
    for (const double theta : theta_values) {
        Errors previous{};
        for (int level = 0; level < levels; ++level) {
            const int steps = 4 * (1 << level);
            const double dt = final_time / steps;
            const Errors e = solve(theta, steps);
            std::cout << theta << ',' << dt << ',' << steps << ',' << e.state << ','
                      << e.tangent << ',' << e.costate << ',' << e.objective << ',';
            if (level == 0) {
                std::cout << "NA,NA,NA,NA,";
            } else {
                const double state_order = std::log2(previous.state / e.state);
                const double tangent_order = std::log2(previous.tangent / e.tangent);
                const double costate_order = std::log2(previous.costate / e.costate);
                const double objective_order = std::log2(previous.objective / e.objective);
                std::cout << state_order << ',' << tangent_order << ',' << costate_order << ','
                          << objective_order << ',';
                if (level == levels - 1) {
                    passed = passed && state_order > minimum_order
                        && tangent_order > minimum_order && costate_order > minimum_order
                        && objective_order > minimum_order;
                }
            }
            std::cout << e.adjoint_identity << '\n';
            passed = passed && e.adjoint_identity < adjoint_identity_tolerance;
            previous = e;
        }
    }
    return passed ? 0 : 1;
}
