// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/analysis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

using asdole::DenseMatrix;
using asdole::Vector;
constexpr std::size_t steps = 80;
constexpr double final_time = 4.0;
constexpr double h = final_time / static_cast<double>(steps);
constexpr double theta = 0.7;
constexpr double coupling = 1.0;
constexpr double rank_threshold = 1.0e-8;
const asdole::HessianFunction zero_hessian = [](std::span<const double>, asdole::HessianTensor& hessian) {
    for (auto& component : hessian) component.fill(0.0);
};

asdole::Stepper make_stepper(double k) {
    const auto rhs = [k](std::span<const double> y, std::span<double> f) {
        f[0] = -y[0] + k * y[1];
        f[1] = -2.0 * y[1];
    };
    const auto jacobian = [k](std::span<const double>, DenseMatrix& j) {
        j(0, 0) = -1.0; j(0, 1) = k;
        j(1, 0) = 0.0;  j(1, 1) = -2.0;
    };
    return asdole::Stepper(rhs, jacobian, theta);
}

std::vector<Vector> integrate_states(const Vector& initial, double k) {
    const auto stepper = make_stepper(k);
    std::vector<Vector> states(steps + 1, Vector(2));
    states[0] = initial;
    for (std::size_t n = 0; n < steps; ++n)
        states[n + 1] = stepper.step(states[n], h).state;
    return states;
}

double trapezoid_weight(std::size_t node) {
    if (node == 0 || node == steps) return 0.5 * h;
    return h;
}

Vector parameter_step_action(const asdole::Stepper& stepper,
                             std::span<const double> state,
                             double step_size,
                             std::span<const double> direction,
                             std::span<const int> selected_unknowns) {
    double dk = 0.0;
    for (std::size_t j = 0; j < selected_unknowns.size(); ++j)
        if (selected_unknowns[j] == 2) dk += direction[j];
    Vector rhs_parameter{state[1] * dk, 0.0};
    DenseMatrix jacobian_parameter(2);
    jacobian_parameter(0, 1) = dk;
    return asdole::parameter_tangent_step(
        stepper, state, step_size, rhs_parameter, jacobian_parameter);
}

struct CaseResult {
    std::string name;
    std::vector<int> columns;
    asdole::SensitivityDiagnostics diagnostics;
    double finite_difference_relative_error{};
    double adjoint_dot_defect{};
};

CaseResult analyze(std::string name, const Vector& initial, const std::vector<int>& columns) {
    const auto stepper = make_stepper(coupling);
    const auto states = integrate_states(initial, coupling);
    const asdole::DiscreteTrajectoryLinearization trajectory(
        stepper, zero_hessian, states, std::vector<double>(steps, h));

    DenseMatrix sensor_matrix(1, 2);
    sensor_matrix(0, 0) = 1.0;
    const auto sensor = asdole::dense_observation_operator(std::move(sensor_matrix));
    std::vector<asdole::ParameterObservationSample> samples;
    samples.reserve(steps + 1);
    for (std::size_t node = 0; node <= steps; ++node)
        samples.push_back({node, sensor, trapezoid_weight(node), {}, {}, {}});

    const auto initial_apply = [columns](std::span<const double> parameter_direction) {
        Vector result(2, 0.0);
        for (std::size_t j = 0; j < columns.size(); ++j) {
            if (columns[j] == 0) result[0] += parameter_direction[j];
            if (columns[j] == 1) result[1] += parameter_direction[j];
        }
        return result;
    };
    const auto initial_transpose = [columns](std::span<const double> state_cotangent) {
        Vector result(columns.size(), 0.0);
        for (std::size_t j = 0; j < columns.size(); ++j) {
            if (columns[j] == 0) result[j] = state_cotangent[0];
            if (columns[j] == 1) result[j] = state_cotangent[1];
        }
        return result;
    };
    const auto step_apply = [stepper, columns](std::size_t, std::span<const double> state,
                                                double step_size, std::span<const double> direction) {
        return parameter_step_action(stepper, state, step_size, direction, columns);
    };
    const auto step_transpose = [step_apply, columns](std::size_t step,
                                                       std::span<const double> state,
                                                       double step_size,
                                                       std::span<const double> state_cotangent) {
        Vector result(columns.size(), 0.0);
        Vector basis(columns.size(), 0.0);
        for (std::size_t j = 0; j < columns.size(); ++j) {
            basis[j] = 1.0;
            const Vector column = step_apply(step, state, step_size, basis);
            result[j] = asdole::inner_product(column, state_cotangent);
            basis[j] = 0.0;
        }
        return result;
    };

    const asdole::ParameterObservationSensitivityOperator sensitivity(
        trajectory, columns.size(), initial_apply, initial_transpose,
        step_apply, step_transpose, std::move(samples));
    const auto diagnostics = asdole::analyze_parameter_identifiability(sensitivity, rank_threshold);

    constexpr double epsilon = 1.0e-6;
    double fd_error_sq = 0.0, sensitivity_sq = 0.0;
    for (std::size_t j = 0; j < columns.size(); ++j) {
        Vector basis(columns.size(), 0.0);
        basis[j] = 1.0;
        const Vector predicted = sensitivity.apply(basis);
        Vector plus_initial = initial, minus_initial = initial;
        double plus_k = coupling, minus_k = coupling;
        if (columns[j] == 0) { plus_initial[0] += epsilon; minus_initial[0] -= epsilon; }
        else if (columns[j] == 1) { plus_initial[1] += epsilon; minus_initial[1] -= epsilon; }
        else { plus_k += epsilon; minus_k -= epsilon; }
        const auto plus = integrate_states(plus_initial, plus_k);
        const auto minus = integrate_states(minus_initial, minus_k);
        for (std::size_t node = 0; node <= steps; ++node) {
            const double fd = (plus[node][0] - minus[node][0]) / (2.0 * epsilon);
            const double weight = trapezoid_weight(node);
            const double error = predicted[node] - fd;
            fd_error_sq += weight * error * error;
            sensitivity_sq += weight * predicted[node] * predicted[node];
        }
    }

    Vector parameter_direction(columns.size()), observation_cotangent(sensitivity.output_dimension());
    for (std::size_t i = 0; i < parameter_direction.size(); ++i)
        parameter_direction[i] = std::sin(0.37 * static_cast<double>(i + 1));
    for (std::size_t i = 0; i < observation_cotangent.size(); ++i)
        observation_cotangent[i] = std::cos(0.19 * static_cast<double>(i + 1));
    const Vector forward = sensitivity.apply(parameter_direction);
    const Vector reverse = sensitivity.transpose_apply(observation_cotangent);
    const double lhs = asdole::inner_product(forward, observation_cotangent);
    const double rhs = asdole::inner_product(parameter_direction, reverse);

    return {std::move(name), columns, diagnostics,
            std::sqrt(fd_error_sq) / std::max(1.0e-30, std::sqrt(sensitivity_sq)),
            std::abs(lhs - rhs) / std::max({1.0, std::abs(lhs), std::abs(rhs)})};
}

void print(const CaseResult& result) {
    std::cout << "case=" << result.name << " parameters=" << result.columns.size()
              << " rank=" << result.diagnostics.numerical_rank << " singular_values=[";
    for (std::size_t i = 0; i < result.diagnostics.singular_values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << result.diagnostics.singular_values[i];
    }
    std::cout << "] condition=" << result.diagnostics.condition_number
              << " sensitivity_fd_relative_error=" << result.finite_difference_relative_error
              << " tlm_adjoint_dot_defect=" << result.adjoint_dot_defect << '\n';
}

} // namespace

int main() {
    std::cout << std::setprecision(12)
              << "model=y1'=-y1+k*y2, y2'=-2*y2 observation=H[ y1;y2 ]=y1\n"
              << "theta=" << theta << " steps=" << steps << " dt=" << h
              << " quadrature=trapezoidal\n"
              << "unknown identifiers: 0=initial_y1, 1=initial_y2, 2=coupling_k\n"
              << "rank threshold=" << rank_threshold << " times largest singular value\n";
    print(analyze("known_initial_estimate_k", {1.0, 0.7}, {2}));
    print(analyze("known_y2_estimate_initial_y1_and_k", {1.0, 0.7}, {0, 2}));
    print(analyze("jointly_estimate_initial_state_and_k", {1.0, 0.7}, {0, 1, 2}));
    print(analyze("no_excitation_y2_zero_estimate_k", {1.0, 0.0}, {2}));
}
