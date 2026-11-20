// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/stepper.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

struct Errors {
    double doubled_step;
    double extrapolated;
};

Errors integrate_asdole(double h, int step_count, const asdole::Stepper& integrator) {
    asdole::Vector extrapolated{1.0};
    asdole::Vector doubled_step{1.0};

    for (int step = 0; step < step_count; ++step) {
        const auto corrected = integrator.step(extrapolated, h);
        extrapolated = corrected.state;

        const auto uncorrected = integrator.step(doubled_step, h);
        doubled_step = uncorrected.doubled_step;
    }

    constexpr double decay_rate = 0.8;
    const double exact = std::exp(-decay_rate * h * step_count);
    return {std::abs(doubled_step[0] - exact), std::abs(extrapolated[0] - exact)};
}

double integrate_theta_method(double theta, double h, int step_count) {
    constexpr double decay_rate = 0.8;
    const double amplification =
        (1.0 - (1.0 - theta) * decay_rate * h) / (1.0 + theta * decay_rate * h);
    double y = 1.0;
    for (int step = 0; step < step_count; ++step) y *= amplification;

    const double exact = std::exp(-decay_rate * h * step_count);
    return std::abs(y - exact);
}

int main() {
    constexpr double decay_rate = 0.8;
    constexpr double final_time = 4.0;
    constexpr int refinement_levels = 8;
    constexpr std::array theta_values{0.5, 0.6, 0.7, 1.0};

    const auto rhs = [](std::span<const double> y, std::span<double> f) {
        f[0] = -decay_rate * y[0];
    };
    const auto jacobian = [](std::span<const double>, asdole::DenseMatrix& j) {
        j(0, 0) = -decay_rate;
    };

    std::cout << "method,theta,dt,steps,global_error,observed_order\n";
    std::cout << std::setprecision(12);

    for (const double theta : theta_values) {
        const asdole::Stepper asdole_integrator(rhs, jacobian, theta);
        double previous_theta_error = 0.0;
        double previous_doubled_error = 0.0;
        double previous_asdole_error = 0.0;

        for (int level = 0; level < refinement_levels; ++level) {
            const int steps = 4 * (1 << level);
            const double h = final_time / steps;
            const double theta_error = integrate_theta_method(theta, h, steps);
            const Errors asdole_errors = integrate_asdole(h, steps, asdole_integrator);

            std::cout << "theta_method," << theta << ',' << h << ',' << steps << ','
                      << theta_error << ',';
            if (level == 0) std::cout << "NA\n";
            else std::cout << std::log2(previous_theta_error / theta_error) << '\n';

            std::cout << "affine_doubled_step," << theta << ',' << h << ',' << steps << ','
                      << asdole_errors.doubled_step << ',';
            if (level == 0) std::cout << "NA\n";
            else std::cout << std::log2(previous_doubled_error / asdole_errors.doubled_step) << '\n';

            std::cout << "asdole_extrapolated," << theta << ',' << h << ',' << steps << ','
                      << asdole_errors.extrapolated << ',';
            if (level == 0) std::cout << "NA\n";
            else std::cout << std::log2(previous_asdole_error / asdole_errors.extrapolated) << '\n';

            previous_theta_error = theta_error;
            previous_doubled_error = asdole_errors.doubled_step;
            previous_asdole_error = asdole_errors.extrapolated;
        }
    }
}
