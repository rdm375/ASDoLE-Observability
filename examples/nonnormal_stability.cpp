// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/analysis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {
constexpr double coupling = 20.0;
constexpr double final_time = 1.0;
constexpr int steps = 200;

void rhs(std::span<const double> y, std::span<double> f) {
    f[0] = -y[0] + coupling * y[1];
    f[1] = -2.0 * y[1];
}

void jacobian(std::span<const double>, asdole::DenseMatrix& j) {
    j(0, 0) = -1.0;
    j(0, 1) = coupling;
    j(1, 0) = 0.0;
    j(1, 1) = -2.0;
}

double norm(std::span<const double> x) {
    return std::sqrt(x[0] * x[0] + x[1] * x[1]);
}

double dot(std::span<const double> x, std::span<const double> y) {
    return x[0] * y[0] + x[1] * y[1];
}

} // namespace

int main() {
    constexpr std::array theta_values{0.5, 0.6, 0.7, 1.0};
    constexpr double expected_state = std::exp(-final_time);
    std::cout << std::setprecision(12)
              << "theta,time,state_x1,state_x2,state_norm,tlm_x1,tlm_x2,tlm_norm,"
                 "adjoint_x1,adjoint_x2,adjoint_norm,optimal_gain,"
                 "optimal_initial_x1,optimal_initial_x2,optimal_final_x1,"
                 "optimal_final_x2,optimal_iterations,optimal_converged\n";
    bool passed = true;
    for (double theta : theta_values) {
        const double h = final_time / steps;
        const asdole::Stepper stepper(rhs, jacobian, theta);
        const asdole::HessianFunction zero_hessian = [](std::span<const double>,
                                                         asdole::HessianTensor&) {};
        std::vector<asdole::Vector> state(steps + 1, asdole::Vector(2));
        std::vector<asdole::Vector> tangent(steps + 1, asdole::Vector(2));
        std::vector<asdole::Vector> adjoint(steps + 1, asdole::Vector(2));
        state[0] = {1.0, 0.0};
        tangent[0] = {0.0, 1.0};
        for (int k = 0; k < steps; ++k) {
            state[k + 1] = stepper.step(state[k], h).state;
            tangent[k + 1] = asdole::tangent_step(stepper, zero_hessian, state[k], h, tangent[k]);
        }
        adjoint[steps] = {1.0, 0.0};
        for (int k = steps - 1; k >= 0; --k)
            adjoint[k] = asdole::adjoint_step(stepper, zero_hessian, state[k], h, adjoint[k + 1]);

        const asdole::DiscreteTrajectoryLinearization linearization(
            stepper, zero_hessian, state, std::vector<double>(steps, h));
        const auto optimal_gain = asdole::transient_amplification_history(linearization);

        double max_tlm = 0.0;
        double max_adjoint = 0.0;
        double max_optimal_gain = 0.0;
        for (int k = 0; k <= steps; ++k) {
            const double t = h * k;
            std::cout << theta << ',' << t << ','
                      << state[k][0] << ',' << state[k][1] << ',' << norm(state[k]) << ','
                      << tangent[k][0] << ',' << tangent[k][1] << ',' << norm(tangent[k]) << ','
                      << adjoint[k][0] << ',' << adjoint[k][1] << ',' << norm(adjoint[k]) << ','
                      << optimal_gain[k].gain << ','
                      << optimal_gain[k].initial_direction[0] << ','
                      << optimal_gain[k].initial_direction[1] << ','
                      << optimal_gain[k].final_perturbation[0] << ','
                      << optimal_gain[k].final_perturbation[1] << ','
                      << optimal_gain[k].iterations << ',' << (optimal_gain[k].converged ? 1 : 0) << '\n';
            max_tlm = std::max(max_tlm, norm(tangent[k]));
            max_adjoint = std::max(max_adjoint, norm(adjoint[k]));
            max_optimal_gain = std::max(max_optimal_gain, optimal_gain[k].gain);
            passed = passed && optimal_gain[k].converged
                && optimal_gain[k].gain + 1.0e-9 >= norm(tangent[k]);
        }
        const double defect = std::abs(dot(adjoint[steps], tangent[steps])
                                     - dot(adjoint[0], tangent[0]));
        passed = passed && std::abs(norm(state[steps]) - expected_state) < 2.0e-3
            && max_tlm > 3.0 && max_adjoint > 3.0 && max_optimal_gain > 3.0
            && defect < 2.0e-10;
        std::cerr << "summary theta=" << theta << " final_state_norm=" << norm(state[steps])
                  << " max_tlm_norm=" << max_tlm << " max_adjoint_norm=" << max_adjoint
                  << " max_optimal_gain=" << max_optimal_gain << " dot_defect=" << defect << '\n';
    }
    return passed ? 0 : 1;
}
