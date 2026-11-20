// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/differentiation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>

namespace {
double theta_amplification(double theta, double z) {
    const auto base = [theta](double value) {
        return (1.0 + (1.0 - theta) * value) / (1.0 - theta * value);
    };
    return 2.0 * base(0.5 * z) * base(0.5 * z) - base(z);
}

double measured_amplification(double theta, double z, int mode) {
    const auto rhs = [z](std::span<const double> y, std::span<double> f) {
        f[0] = z * y[0]; // h is set to one, so z=lambda*h.
    };
    const auto jacobian = [z](std::span<const double>, asdole::DenseMatrix& j) {
        j(0, 0) = z;
    };
    const asdole::Stepper stepper(rhs, jacobian, theta);
    const asdole::Vector initial{1.0};
    if (mode == 0) return stepper.step(initial, 1.0).state[0];
    const asdole::HessianFunction hessian = [](std::span<const double>, asdole::HessianTensor&) {};
    if (mode == 1)
        return asdole::tangent_step(stepper, hessian, initial, 1.0, asdole::Vector{1.0})[0];
    return asdole::adjoint_step(stepper, hessian, initial, 1.0, asdole::Vector{1.0})[0];
}

bool close(double a, double b) {
    return std::abs(a - b) <= 2.0e-10 * std::max({1.0, std::abs(a), std::abs(b)});
}
} // namespace

int main() {
    constexpr std::array theta_values{0.5, 0.6, 0.7, 1.0};
    constexpr int points = 301;
    constexpr double minimum_x = 1.0e-5;
    constexpr double maximum_x = 1.0e3;
    std::cout << std::setprecision(12)
              << "theta,z,amplification,state_gain,tlm_gain,adjoint_gain,stable\n";
    bool passed = true;
    for (double theta : theta_values) {
        for (int i = 0; i < points; ++i) {
            const double fraction = static_cast<double>(i) / (points - 1);
            const double x = std::pow(10.0, std::log10(minimum_x)
                                      + fraction * (std::log10(maximum_x) - std::log10(minimum_x)));
            const double z = -x;
            const double expected = theta_amplification(theta, z);
            const double state = measured_amplification(theta, z, 0);
            const double tlm = measured_amplification(theta, z, 1);
            const double adjoint = measured_amplification(theta, z, 2);
            passed = passed && close(state, expected) && close(tlm, expected) && close(adjoint, expected);
            std::cout << theta << ',' << z << ',' << expected << ',' << state << ',' << tlm << ','
                      << adjoint << ',' << (std::abs(expected) <= 1.0 ? 1 : 0) << '\n';
        }
    }

    // Sanity checks for the stability regions seen in the sweep.
    passed = passed && std::abs(theta_amplification(0.5, -1.0)) < 1.0
        && std::abs(theta_amplification(0.5, -10.0)) > 1.0
        && std::abs(theta_amplification(0.6, -30.0)) > 1.0
        && std::abs(theta_amplification(0.7, -1000.0)) < 1.0
        && std::abs(theta_amplification(1.0, -1000.0)) < 1.0e-3;
    std::cerr << (passed ? "stability_sweep=PASS\n" : "stability_sweep=FAIL\n");
    return passed ? 0 : 1;
}
