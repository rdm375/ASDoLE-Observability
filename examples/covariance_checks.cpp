// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/covariance.hpp"

#include <algorithm>
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

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void coupled_rhs(std::span<const double> y, std::span<double> f) {
    f[0] = -y[0];
    f[1] = y[0] - y[1];
}

void coupled_jacobian(std::span<const double>, DenseMatrix& j) {
    j(0, 0) = -1.0; j(0, 1) = 0.0;
    j(1, 0) = 1.0;  j(1, 1) = -1.0;
}

void zero_hessian(std::span<const double>, HessianTensor&) {}

// A simpler trajectory holder avoids moving an object whose linearization
// stores references into its state history.
struct LinearRun {
    asdole::Stepper stepper;
    std::vector<Vector> states;
    std::vector<double> steps;
    asdole::DiscreteTrajectoryLinearization linearization;

    LinearRun(double theta, int count, double final_time = 2.0)
        : stepper(coupled_rhs, coupled_jacobian, theta),
          states(static_cast<std::size_t>(count + 1)),
          steps(static_cast<std::size_t>(count), final_time / count),
          linearization([&]() {
              states[0] = {1.0, 0.0};
              for (int k = 0; k < count; ++k)
                  states[static_cast<std::size_t>(k + 1)] = stepper.step(
                      states[static_cast<std::size_t>(k)], steps[static_cast<std::size_t>(k)]).state;
              return asdole::DiscreteTrajectoryLinearization(stepper, zero_hessian, states, steps);
          }()) {}
};

DenseMatrix initial_state_covariance() {
    DenseMatrix p(2);
    p(0, 0) = 0.01;
    p(1, 1) = 0.01;
    return p;
}

DenseMatrix transpose(const DenseMatrix& a) {
    DenseMatrix result(a.cols(), a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) result(j, i) = a(i, j);
    return result;
}

DenseMatrix multiply(const DenseMatrix& a, const DenseMatrix& b) {
    require(a.cols() == b.rows(), "test matrix dimensions differ");
    DenseMatrix result(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t k = 0; k < a.cols(); ++k)
            for (std::size_t j = 0; j < b.cols(); ++j)
                result(i, j) += a(i, k) * b(k, j);
    return result;
}

double relative_matrix_error(const DenseMatrix& a, const DenseMatrix& b) {
    require(a.rows() == b.rows() && a.cols() == b.cols(), "matrix comparison dimensions differ");
    double difference = 0.0, reference = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) {
            difference += (a(i, j) - b(i, j)) * (a(i, j) - b(i, j));
            reference += b(i, j) * b(i, j);
        }
    return std::sqrt(difference) / std::max(1.0e-30, std::sqrt(reference));
}

void exact_covariance_convergence() {
    for (double theta : {0.5, 0.6, 0.7, 1.0}) {
        double previous_error = 0.0;
        double finest_order = 0.0;
        for (int steps : {20, 40, 80, 160}) {
            LinearRun run(theta, steps);
            const auto p = asdole::propagate_covariance(run.linearization, initial_state_covariance()).back();
            const double t = 2.0;
            const double e2 = std::exp(-2.0 * t);
            DenseMatrix exact(2);
            exact(0, 0) = 0.01 * e2;
            exact(0, 1) = exact(1, 0) = 0.01 * t * e2;
            exact(1, 1) = 0.01 * (t * t + 1.0) * e2;
            const double error = relative_matrix_error(p, exact);
            if (previous_error > 0.0) finest_order = std::log2(previous_error / error);
            previous_error = error;
        }
        std::cout << "covariance_exact_convergence theta=" << theta
                  << " finest_order=" << finest_order << '\n';
        require(finest_order > 1.8, "state covariance did not converge at second order");
    }
}

void process_noise_psd_and_symmetry() {
    constexpr int steps = 40;
    LinearRun run(0.7, steps);
    const auto p0 = initial_state_covariance();
    const auto without_noise = asdole::propagate_covariance(run.linearization, p0);
    std::vector<DenseMatrix> q(static_cast<std::size_t>(steps), DenseMatrix(2));
    for (auto& noise : q) {
        noise(0, 0) = 2.0e-8;
        noise(0, 1) = noise(1, 0) = 1.0e-8;
        noise(1, 1) = 2.0e-8;
    }
    const auto with_noise = asdole::propagate_covariance(run.linearization, p0, q);
    for (const auto& covariance : with_noise) {
        const auto diagnostic = asdole::diagnose_covariance(covariance);
        require(diagnostic.symmetry_defect < 1.0e-13, "propagated covariance lost symmetry");
        require(diagnostic.smallest_eigenvalue > -1.0e-12,
                "propagated covariance is not positive semidefinite");
    }
    const double trace_no_noise = without_noise.back()(0, 0) + without_noise.back()(1, 1);
    const double trace_noise = with_noise.back()(0, 0) + with_noise.back()(1, 1);
    require(trace_noise > trace_no_noise, "independent process noise did not increase final variance");
    std::cout << "process_noise_psd symmetry_defect="
              << asdole::diagnose_covariance(with_noise.back()).symmetry_defect
              << " minimum_eigenvalue="
              << asdole::diagnose_covariance(with_noise.back()).smallest_eigenvalue << '\n';
}

void cross_time_and_observation_covariance() {
    LinearRun run(0.7, 80);
    const auto covariance = asdole::propagate_covariance(run.linearization, initial_state_covariance());
    const std::size_t last = run.linearization.node_count() - 1;
    const DenseMatrix cross = asdole::state_cross_covariance(run.linearization, covariance, 0, last);
    const DenseMatrix reverse = asdole::state_cross_covariance(run.linearization, covariance, last, 0);
    require(relative_matrix_error(reverse, transpose(cross)) < 1.0e-13,
            "reverse-time cross-covariance is not the transpose");

    // Independently form the numerical state-transition matrix by applying
    // the TLM to coordinate directions over the entire stored trajectory.
    DenseMatrix phi(2);
    for (std::size_t j = 0; j < 2; ++j) {
        Vector basis(2, 0.0); basis[j] = 1.0;
        const Vector image = run.linearization.apply(basis, last);
        for (std::size_t i = 0; i < 2; ++i) phi(i, j) = image[i];
    }
    const DenseMatrix expected = multiply(initial_state_covariance(), transpose(phi));
    require(relative_matrix_error(cross, expected) < 1.0e-13,
            "cross-time state covariance does not match the TLM transition");

    DenseMatrix hx(1, 2), hy(1, 2);
    hx(0, 0) = 1.0; hy(0, 1) = 1.0;
    const auto observe_x = asdole::dense_observation_operator(hx);
    const auto observe_y = asdole::dense_observation_operator(hy);
    DenseMatrix measurement_cross(1, 1);
    measurement_cross(0, 0) = 3.0e-4;
    const auto observed = asdole::observation_cross_covariance(
        observe_x, cross, observe_y, &measurement_cross);
    const double expected_observed = expected(0, 1) + measurement_cross(0, 0);
    require(std::abs(observed(0, 0) - expected_observed) < 1.0e-13,
            "observation cross-covariance projection or measurement term is wrong");
    std::cout << "cross_time_observation_covariance value=" << observed(0, 0) << '\n';
}

struct ParameterRun {
    asdole::Stepper stepper;
    std::vector<Vector> states;
    std::vector<double> steps;
    asdole::DiscreteTrajectoryLinearization linearization;
    std::vector<DenseMatrix> parameter_steps;

    ParameterRun(double coupling, int count)
        : stepper(
              [coupling](std::span<const double> y, std::span<double> f) {
                  f[0] = -y[0] + coupling * y[1]; f[1] = -2.0 * y[1];
              },
              [coupling](std::span<const double>, DenseMatrix& j) {
                  j(0, 0) = -1.0; j(0, 1) = coupling;
                  j(1, 0) = 0.0;  j(1, 1) = -2.0;
              }, 0.7),
          states(static_cast<std::size_t>(count + 1)),
          steps(static_cast<std::size_t>(count), 2.0 / count),
          linearization([&]() {
              states[0] = {1.0, 0.7};
              for (int k = 0; k < count; ++k)
                  states[static_cast<std::size_t>(k + 1)] = stepper.step(
                      states[static_cast<std::size_t>(k)], steps[static_cast<std::size_t>(k)]).state;
              return asdole::DiscreteTrajectoryLinearization(stepper, zero_hessian, states, steps);
          }()),
          parameter_steps(make_parameter_steps(coupling, count)) {}

private:
    std::vector<DenseMatrix> make_parameter_steps(double coupling, int count) {
        DenseMatrix j_parameter(2);
        j_parameter(0, 1) = 1.0;
        std::vector<DenseMatrix> result;
        result.reserve(static_cast<std::size_t>(count));
        for (int k = 0; k < count; ++k) {
            const auto& y = states[static_cast<std::size_t>(k)];
            const Vector f_parameter{y[1], 0.0};
            const Vector b = asdole::parameter_tangent_step(
                stepper, y, steps[static_cast<std::size_t>(k)], f_parameter, j_parameter);
            DenseMatrix column(2, 1);
            column(0, 0) = b[0]; column(1, 0) = b[1];
            result.push_back(std::move(column));
        }
        (void)coupling;
        return result;
    }
};

Vector evaluate_parameter_model(double x0, double y0, double coupling, int steps) {
    ParameterRun run(coupling, steps);
    // Recompute with perturbed initial data; the stepper captures the requested k.
    run.states[0] = {x0, y0};
    for (int k = 0; k < steps; ++k)
        run.states[static_cast<std::size_t>(k + 1)] = run.stepper.step(
            run.states[static_cast<std::size_t>(k)], run.steps[static_cast<std::size_t>(k)]).state;
    return {run.states.back()[0], run.states.back()[1], coupling};
}

void joint_parameter_covariance_matches_finite_differences() {
    constexpr int steps = 40;
    constexpr double k = 0.8;
    ParameterRun run(k, steps);
    DenseMatrix p0(3);
    p0(0, 0) = 0.05 * 0.05;
    p0(1, 1) = 0.05 * 0.05;
    p0(2, 2) = 0.04 * 0.04;
    const auto propagated = asdole::propagate_joint_state_parameter_covariance(
        run.linearization, p0, run.parameter_steps).back();

    constexpr double epsilon = 1.0e-6;
    DenseMatrix jacobian(3);
    for (std::size_t column = 0; column < 3; ++column) {
        double xp = 1.0, xm = 1.0, yp = 0.7, ym = 0.7, kp = k, km = k;
        if (column == 0) { xp += epsilon; xm -= epsilon; }
        if (column == 1) { yp += epsilon; ym -= epsilon; }
        if (column == 2) { kp += epsilon; km -= epsilon; }
        const Vector plus = evaluate_parameter_model(xp, yp, kp, steps);
        const Vector minus = evaluate_parameter_model(xm, ym, km, steps);
        for (std::size_t row = 0; row < 3; ++row)
            jacobian(row, column) = (plus[row] - minus[row]) / (2.0 * epsilon);
    }
    const DenseMatrix fd_covariance = multiply(multiply(jacobian, p0), transpose(jacobian));
    const double error = relative_matrix_error(propagated, fd_covariance);
    std::cout << "joint_parameter_covariance_fd_relative_error=" << error << '\n';
    require(error < 2.0e-7,
            "joint state/parameter covariance disagrees with finite-difference sensitivities");
}

void low_rank_factor_matches_dense_covariance() {
    LinearRun run(0.7, 40);
    DenseMatrix factor(2, 2);
    factor(0,0) = 0.1;
    factor(1,0) = 0.03;
    factor(1,1) = 0.08;
    const DenseMatrix p0 = multiply(factor, transpose(factor));
    const auto dense_history = asdole::propagate_covariance(run.linearization, p0);
    const auto factor_history = asdole::propagate_covariance_factor(run.linearization, factor);
    require(factor_history.size() == dense_history.size(),
            "low-rank covariance history has the wrong node count");
    double maximum_error = 0.0;
    for (std::size_t k = 0; k < factor_history.size(); ++k) {
        const DenseMatrix reconstructed = multiply(factor_history[k], transpose(factor_history[k]));
        maximum_error = std::max(maximum_error, relative_matrix_error(reconstructed, dense_history[k]));
    }
    const Vector direction{0.7, -0.4};
    const Vector matrix_free = asdole::apply_covariance_factor(factor_history.back(), direction);
    const Vector dense_action = {
        dense_history.back()(0,0)*direction[0] + dense_history.back()(0,1)*direction[1],
        dense_history.back()(1,0)*direction[0] + dense_history.back()(1,1)*direction[1]};
    const double action_error = std::hypot(matrix_free[0]-dense_action[0],
                                            matrix_free[1]-dense_action[1]);
    std::cout << "low_rank_covariance_factor_max_relative_error=" << maximum_error
              << " covariance_action_error=" << action_error << '\n';
    require(maximum_error < 1.0e-11 && action_error < 1.0e-13,
            "matrix-free low-rank covariance disagrees with dense propagation");
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(12);
        exact_covariance_convergence();
        process_noise_psd_and_symmetry();
        cross_time_and_observation_covariance();
        joint_parameter_covariance_matches_finite_differences();
        low_rank_factor_matches_dense_covariance();
        std::cout << "result=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "covariance checks failed: " << error.what() << '\n';
        std::cout << "result=FAIL\n";
        return 1;
    }
}
