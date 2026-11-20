// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/differentiation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using asdole::DenseMatrix;
using asdole::HessianTensor;
using asdole::Vector;

constexpr double test_h = 0.07;
constexpr double final_time = 1.0;
constexpr int objective_steps = 40;
constexpr double fd_epsilon = 1.0e-6;

void rhs(std::span<const double> y, std::span<double> f) {
    f[0] = -0.7 * y[0] + 0.15 * y[1] + 0.04 * y[0] * y[0] + 0.03 * y[0] * y[1];
    f[1] = -0.2 * y[0] - 1.1 * y[1] + 0.02 * y[0] * y[1] + 0.05 * y[1] * y[1];
}

void jacobian(std::span<const double> y, DenseMatrix& j) {
    j(0, 0) = -0.7 + 0.08 * y[0] + 0.03 * y[1];
    j(0, 1) = 0.15 + 0.03 * y[0];
    j(1, 0) = -0.2 + 0.02 * y[1];
    j(1, 1) = -1.1 + 0.02 * y[0] + 0.10 * y[1];
}

void hessian(std::span<const double>, HessianTensor& h) {
    h[0](0, 0) = 0.08;
    h[0](0, 1) = 0.03;
    h[0](1, 0) = 0.03;
    h[1](0, 1) = 0.02;
    h[1](1, 0) = 0.02;
    h[1](1, 1) = 0.10;
}

double dot(std::span<const double> x, std::span<const double> y) {
    double value = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) value += x[i] * y[i];
    return value;
}

double norm(std::span<const double> x) {
    return std::sqrt(dot(x, x));
}

Vector shifted(std::span<const double> x, std::span<const double> direction, double scale) {
    Vector result(x.begin(), x.end());
    for (std::size_t i = 0; i < x.size(); ++i) result[i] += scale * direction[i];
    return result;
}

double relative_error(std::span<const double> actual, std::span<const double> reference) {
    Vector difference(actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i) difference[i] = actual[i] - reference[i];
    return norm(difference) / std::max(1.0, norm(reference));
}

Vector observation(double t) {
    return {0.65 * std::exp(-0.3 * t), -0.25 * std::exp(-0.5 * t)};
}

Vector loss_gradient(std::span<const double> y, double t) {
    const Vector obs = observation(t);
    return {y[0] - obs[0], y[1] - obs[1]};
}

struct Trajectory {
    std::vector<Vector> state;
    double objective{};
};

Trajectory evaluate_objective(const asdole::Stepper& stepper, std::span<const double> initial) {
    const double h = final_time / objective_steps;
    Trajectory trajectory;
    trajectory.state.resize(objective_steps + 1);
    trajectory.state[0].assign(initial.begin(), initial.end());
    for (int k = 0; k < objective_steps; ++k) {
        trajectory.state[k + 1] = stepper.step(trajectory.state[k], h).state;
    }
    for (int k = 0; k <= objective_steps; ++k) {
        const Vector gradient = loss_gradient(trajectory.state[k], h * k);
        const double weight = (k == 0 || k == objective_steps) ? 0.5 : 1.0;
        trajectory.objective += 0.5 * h * weight * dot(gradient, gradient);
    }
    return trajectory;
}

Vector objective_adjoint(const asdole::Stepper& stepper, const Trajectory& trajectory) {
    const double h = final_time / objective_steps;
    Vector p = loss_gradient(trajectory.state[objective_steps], final_time);
    for (double& value : p) value *= 0.5 * h;

    for (int k = objective_steps - 1; k >= 1; --k) {
        const Vector running = loss_gradient(trajectory.state[k], h * k);
        Vector previous = asdole::adjoint_step(stepper, hessian, trajectory.state[k], h, p);
        for (std::size_t i = 0; i < p.size(); ++i) previous[i] += h * running[i];
        p = std::move(previous);
    }

    const Vector initial_running = loss_gradient(trajectory.state[0], 0.0);
    Vector gradient = asdole::adjoint_step(stepper, hessian, trajectory.state[0], h, p);
    for (std::size_t i = 0; i < p.size(); ++i) gradient[i] += 0.5 * h * initial_running[i];
    return gradient;
}

double objective_tlm_directional(const asdole::Stepper& stepper,
                                 const Trajectory& trajectory,
                                 std::span<const double> initial_direction) {
    const double h = final_time / objective_steps;
    Vector tangent(initial_direction.begin(), initial_direction.end());
    double derivative = 0.0;
    for (int k = 0; k <= objective_steps; ++k) {
        const Vector running = loss_gradient(trajectory.state[k], h * k);
        const double weight = (k == 0 || k == objective_steps) ? 0.5 : 1.0;
        derivative += h * weight * dot(running, tangent);
        if (k < objective_steps) {
            tangent = asdole::tangent_step(stepper, hessian, trajectory.state[k], h, tangent);
        }
    }
    return derivative;
}

} // namespace

int main(int argc, char** argv) {
    const double theta = argc > 1 ? std::stod(argv[1]) : 0.7;
    const asdole::Stepper stepper(rhs, jacobian, theta);
    const Vector y{0.8, -0.35};
    const Vector direction{0.6, -0.8};
    const Vector cotangent{-0.4, 0.9};

    const Vector tangent = asdole::tangent_step(stepper, hessian, y, test_h, direction);
    const Vector y_plus = stepper.step(shifted(y, direction, fd_epsilon), test_h).state;
    const Vector y_minus = stepper.step(shifted(y, direction, -fd_epsilon), test_h).state;
    Vector tangent_fd(2);
    for (std::size_t i = 0; i < tangent.size(); ++i) {
        tangent_fd[i] = (y_plus[i] - y_minus[i]) / (2.0 * fd_epsilon);
    }

    const Vector pulled_back = asdole::adjoint_step(stepper, hessian, y, test_h, cotangent);
    const double adjoint_dot_defect = std::abs(dot(cotangent, tangent) - dot(pulled_back, direction));
    Vector adjoint_fd(2);
    for (std::size_t i = 0; i < 2; ++i) {
        Vector coordinate(2, 0.0);
        coordinate[i] = 1.0;
        const Vector plus = stepper.step(shifted(y, coordinate, fd_epsilon), test_h).state;
        const Vector minus = stepper.step(shifted(y, coordinate, -fd_epsilon), test_h).state;
        adjoint_fd[i] = (dot(cotangent, plus) - dot(cotangent, minus)) / (2.0 * fd_epsilon);
    }

    const Vector initial{0.8, -0.35};
    const Trajectory trajectory = evaluate_objective(stepper, initial);
    const Vector objective_gradient = objective_adjoint(stepper, trajectory);
    const double tlm_objective_directional = objective_tlm_directional(stepper, trajectory, direction);
    const double adjoint_objective_directional = dot(objective_gradient, direction);
    const double objective_plus = evaluate_objective(stepper, shifted(initial, direction, fd_epsilon)).objective;
    const double objective_minus = evaluate_objective(stepper, shifted(initial, direction, -fd_epsilon)).objective;
    const double objective_fd_directional = (objective_plus - objective_minus) / (2.0 * fd_epsilon);

    Vector objective_gradient_fd(2);
    for (std::size_t i = 0; i < 2; ++i) {
        Vector coordinate(2, 0.0);
        coordinate[i] = 1.0;
        objective_gradient_fd[i] =
            (evaluate_objective(stepper, shifted(initial, coordinate, fd_epsilon)).objective
             - evaluate_objective(stepper, shifted(initial, coordinate, -fd_epsilon)).objective)
            / (2.0 * fd_epsilon);
    }

    const double tlm_fd_error = relative_error(tangent, tangent_fd);
    const double adjoint_fd_error = relative_error(pulled_back, adjoint_fd);
    const double objective_gradient_fd_error = relative_error(objective_gradient, objective_gradient_fd);
    const double objective_tlm_adjoint_defect =
        std::abs(tlm_objective_directional - adjoint_objective_directional);
    const double objective_fd_error =
        std::abs(objective_fd_directional - adjoint_objective_directional)
        / std::max(1.0, std::abs(objective_fd_directional));

    constexpr double finite_difference_tolerance = 2.0e-8;
    constexpr double adjoint_tolerance = 2.0e-12;
    const bool passed = tlm_fd_error < finite_difference_tolerance
        && adjoint_fd_error < finite_difference_tolerance
        && objective_gradient_fd_error < finite_difference_tolerance
        && objective_fd_error < finite_difference_tolerance
        && adjoint_dot_defect < adjoint_tolerance
        && objective_tlm_adjoint_defect < adjoint_tolerance;

    std::cout << std::setprecision(12)
              << "theta=" << theta << " h=" << test_h << '\n'
              << "step_tlm_vs_fd_relative_error=" << tlm_fd_error << '\n'
              << "step_adjoint_dot_defect=" << adjoint_dot_defect << '\n'
              << "step_adjoint_vs_fd_relative_error=" << adjoint_fd_error << '\n'
              << "objective_tlm_vs_adjoint_defect=" << objective_tlm_adjoint_defect << '\n'
              << "objective_gradient_vs_fd_relative_error=" << objective_gradient_fd_error << '\n'
              << "objective_directional_vs_fd_relative_error=" << objective_fd_error << '\n'
              << (passed ? "result=PASS\n" : "result=FAIL\n");
    return passed ? 0 : 1;
}
