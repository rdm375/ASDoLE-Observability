// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/adaptive.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>

namespace {
constexpr double final_time = 1.0;
constexpr double pulse_center = 0.5;
constexpr double pulse_width = 0.08;
constexpr double pulse_amplitude = 40.0;

void rhs(std::span<const double> y, std::span<double> f) {
    const double offset = (y[0] - pulse_center) / pulse_width;
    const double rate = 0.5 + pulse_amplitude * std::exp(-offset * offset);
    f[0] = 1.0;
    f[1] = -rate * y[1];
}

void jacobian(std::span<const double> y, asdole::DenseMatrix& j) {
    const double offset = (y[0] - pulse_center) / pulse_width;
    const double pulse = pulse_amplitude * std::exp(-offset * offset);
    const double rate = 0.5 + pulse;
    const double rate_derivative = -2.0 * offset * pulse / pulse_width;
    j(0, 0) = 0.0;
    j(0, 1) = 0.0;
    j(1, 0) = -rate_derivative * y[1];
    j(1, 1) = -rate;
}

double exact_final_state() {
    const double scaled_start = (0.0 - pulse_center) / pulse_width;
    const double scaled_end = (final_time - pulse_center) / pulse_width;
    const double integrated_rate = 0.5 * final_time
        + pulse_amplitude * pulse_width * std::sqrt(std::acos(-1.0)) * 0.5
          * (std::erf(scaled_end) - std::erf(scaled_start));
    return std::exp(-integrated_rate);
}

struct Summary {
    int accepted{};
    int rejected{};
    double final_state{};
    double global_error{};
    double minimum_pulse_step{std::numeric_limits<double>::infinity()};
    double maximum_far_step{};
    double maximum_accepted_estimate{};
};

Summary integrate(double theta) {
    asdole::AdaptiveOptions options;
    options.absolute_tolerance = 1.0e-9;
    options.relative_tolerance = 1.0e-5;
    options.maximum_step = 0.25;
    const asdole::Stepper stepper(rhs, jacobian, theta);
    const asdole::StepSizeController controller(theta, options);

    asdole::Vector state{0.0, 1.0};
    double h = 0.2;
    Summary summary;
    int attempts = 0;
    while (state[0] < final_time) {
        const double remaining = final_time - state[0];
        if (remaining <= 4.0 * std::numeric_limits<double>::epsilon() * final_time) {
            state[0] = final_time;
            break;
        }
        if (++attempts > 100000) throw std::runtime_error("adaptive transient exceeded attempt limit");
        h = std::min(h, remaining);
        const asdole::StepResult trial = stepper.step(state, h);
        const asdole::StepSizeDecision decision = controller.assess(state, trial, h);
        std::cout << theta << ',' << state[0] << ',' << h << ',' << decision.scaled_error << ','
                  << (decision.accept ? 1 : 0) << ',' << decision.next_step << ',' << state[1] << ','
                  << trial.state[1] << '\n';
        if (decision.accept) {
            const double t0 = state[0];
            state = trial.state;
            if (t0 >= 0.3 && t0 <= 0.7)
                summary.minimum_pulse_step = std::min(summary.minimum_pulse_step, h);
            if (t0 <= 0.2 || t0 >= 0.8)
                summary.maximum_far_step = std::max(summary.maximum_far_step, h);
            summary.maximum_accepted_estimate = std::max(summary.maximum_accepted_estimate,
                                                         decision.scaled_error);
            ++summary.accepted;
        } else {
            ++summary.rejected;
        }
        if (!decision.accept && h <= options.minimum_step * (1.0 + 1.0e-12))
            throw std::runtime_error("adaptive transient reached minimum step after rejection");
        h = decision.next_step;
    }
    summary.final_state = state[1];
    summary.global_error = std::abs(summary.final_state - exact_final_state());
    return summary;
}
} // namespace

int main() {
    constexpr std::array theta_values{0.5, 0.6, 0.7, 1.0};
    std::cout << std::setprecision(12)
              << "theta,t_start,h,error_norm,accepted,h_next,y_start,y_trial\n";
    bool passed = true;
    for (double theta : theta_values) {
        const Summary s = integrate(theta);
        std::cerr << "summary theta=" << theta << " accepted=" << s.accepted
                  << " rejected=" << s.rejected << " final_error=" << s.global_error
                  << " min_pulse_h=" << s.minimum_pulse_step
                  << " max_far_h=" << s.maximum_far_step
                  << " max_accepted_estimate=" << s.maximum_accepted_estimate << '\n';
        passed = passed && s.accepted > 0 && s.rejected > 0 && s.global_error < 2.0e-4
            && s.minimum_pulse_step < 0.5 * s.maximum_far_step
            && s.maximum_accepted_estimate <= 1.0;
    }
    return passed ? 0 : 1;
}
