// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/covariance.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using asdole::DenseMatrix;
using asdole::HessianTensor;
using asdole::Vector;

constexpr double coupling_mean = 1.0;
constexpr double sigma_coupling = 0.1;
constexpr double final_time = 4.0;
constexpr int steps = 160;
constexpr double dt = final_time / steps;

void rhs(std::span<const double> y, std::span<double> f) {
    f[0] = -y[0];
    f[1] = coupling_mean * y[0] - y[1];
}

void jacobian(std::span<const double>, DenseMatrix& j) {
    j(0, 0) = -1.0; j(0, 1) = 0.0;
    j(1, 0) = coupling_mean; j(1, 1) = -1.0;
}

void zero_hessian(std::span<const double>, HessianTensor&) {}

struct Results {
    double noise_intensity{};
    std::vector<DenseMatrix> parameter;
    std::vector<DenseMatrix> forcing;
};

Results compute() {
    const asdole::Stepper stepper(rhs, jacobian, 0.7);
    std::vector<Vector> states(static_cast<std::size_t>(steps + 1));
    std::vector<double> step_sizes(static_cast<std::size_t>(steps), dt);
    states[0] = {1.0, 0.0};
    for (int k = 0; k < steps; ++k)
        states[static_cast<std::size_t>(k + 1)] = stepper.step(
            states[static_cast<std::size_t>(k)], dt).state;
    const asdole::DiscreteTrajectoryLinearization trajectory(
        stepper, zero_hessian, states, step_sizes);

    // Scenario A: only the persistent coupling parameter k is uncertain.
    DenseMatrix jacobian_k(2);
    jacobian_k(1, 0) = 1.0;
    std::vector<DenseMatrix> b_k;
    b_k.reserve(steps);
    for (int k = 0; k < steps; ++k) {
        const auto& y = states[static_cast<std::size_t>(k)];
        const Vector f_k{0.0, y[0]};
        const Vector b = asdole::parameter_tangent_step(
            stepper, y, dt, f_k, jacobian_k);
        DenseMatrix block(2, 1);
        block(0, 0) = b[0]; block(1, 0) = b[1];
        b_k.push_back(std::move(block));
    }
    DenseMatrix parameter_prior(3);
    parameter_prior(2, 2) = sigma_coupling * sigma_coupling;
    const auto joint = asdole::propagate_joint_state_parameter_covariance(
        trajectory, parameter_prior, b_k);
    std::vector<DenseMatrix> parameter_state;
    parameter_state.reserve(joint.size());
    for (const auto& p : joint) {
        DenseMatrix state_covariance(2);
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 2; ++j)
                state_covariance(i, j) = p(i, j);
        parameter_state.push_back(std::move(state_covariance));
    }

    // Scenario B: independent white-noise-like increments enter x at each
    // step, with Q_k = q * dt * e_x e_x^T. First calculate the response for
    // q=1, then choose q to match the parameter case's final Var(y).
    std::vector<DenseMatrix> unit_noise;
    unit_noise.reserve(steps);
    for (int k = 0; k < steps; ++k) {
        DenseMatrix q(2);
        q(0, 0) = dt;
        unit_noise.push_back(std::move(q));
    }
    DenseMatrix zero_initial(2);
    const auto unit_response = asdole::propagate_covariance(trajectory, zero_initial, unit_noise);
    const double parameter_final_var_y = parameter_state.back()(1, 1);
    const double unit_final_var_y = unit_response.back()(1, 1);
    if (!(parameter_final_var_y > 0.0) || !(unit_final_var_y > 0.0))
        throw std::runtime_error("source comparison requires positive final y variance");
    const double q_intensity = parameter_final_var_y / unit_final_var_y;

    std::vector<DenseMatrix> scaled_noise;
    scaled_noise.reserve(steps);
    for (int k = 0; k < steps; ++k) {
        DenseMatrix q(2);
        q(0, 0) = q_intensity * dt;
        scaled_noise.push_back(std::move(q));
    }
    auto forcing_state = asdole::propagate_covariance(trajectory, zero_initial, scaled_noise);

    // Assertions make the executable a useful regression as well as a data
    // generator: the parameter affects y only, while x-forcing spreads into
    // both states, and the selected scale aligns final Var(y).
    if (std::abs(parameter_state.back()(0, 0)) > 1.0e-14
        || std::abs(parameter_state.back()(0, 1)) > 1.0e-14)
        throw std::runtime_error("coupling parameter must not perturb x in this model");
    if (!(forcing_state.back()(0, 0) > 0.0)
        || !(forcing_state.back()(0, 1) > 0.0)
        || !(forcing_state.back()(1, 1) > 0.0))
        throw std::runtime_error("forcing perturbations did not propagate through the coupling");
    const double final_match = std::abs(forcing_state.back()(1, 1) - parameter_final_var_y)
                             / parameter_final_var_y;
    if (final_match > 1.0e-12)
        throw std::runtime_error("source calibration failed to match final y variance");
    for (std::size_t k = 1; k < forcing_state.size(); ++k) {
        if (forcing_state[k](1, 1) + 1.0e-14 < forcing_state[k - 1](1, 1))
            throw std::runtime_error("stepwise forcing variance in y should accumulate monotonically here");
    }
    return {q_intensity, std::move(parameter_state), std::move(forcing_state)};
}

} // namespace

int main() {
    try {
        const Results result = compute();
        std::cout << std::setprecision(12)
                  << "time,param_var_x,param_cov_xy,param_var_y,forcing_var_x,"
                     "forcing_cov_xy,forcing_var_y,forcing_intensity\n";
        for (int k = 0; k <= steps; ++k) {
            const auto i = static_cast<std::size_t>(k);
            const auto& p = result.parameter[i];
            const auto& q = result.forcing[i];
            std::cout << dt * k << ',' << p(0, 0) << ',' << p(0, 1) << ',' << p(1, 1) << ','
                      << q(0, 0) << ',' << q(0, 1) << ',' << q(1, 1) << ','
                      << result.noise_intensity << '\n';
        }
        std::cerr << "# theta=0.7 steps=" << steps << " final_time=" << final_time
                  << " sigma_k=" << sigma_coupling
                  << " calibrated_white_noise_intensity=" << result.noise_intensity
                  << " final_var_y=" << result.parameter.back()(1, 1) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error_source_comparison: " << error.what() << '\n';
        return 1;
    }
}
