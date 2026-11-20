// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/adaptive.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
const char* norm_name(asdole::AdaptiveErrorNorm norm) {
    return norm == asdole::AdaptiveErrorNorm::weighted_infinity ? "inf" : "rms";
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [--atol VALUE] [--rtol VALUE] [--norm rms|inf]\n";
}
} // namespace

int main(int argc, char** argv) {
    constexpr double decay_rate = 0.8;
    constexpr double initial_state = 1.25;
    constexpr double final_time = 4.0;
    constexpr std::array theta_values{0.5, 0.6, 0.7, 1.0};

    const auto rhs = [](std::span<const double> y, std::span<double> f) {
        f[0] = -decay_rate * y[0];
    };
    const auto jacobian = [](std::span<const double>, asdole::DenseMatrix& j) {
        j(0, 0) = -decay_rate;
    };

    asdole::AdaptiveOptions options;
    options.absolute_tolerance = 1.0e-7;
    options.relative_tolerance = 1.0e-5;
    options.maximum_step = 1.0;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (argument != "--atol" && argument != "--rtol" && argument != "--norm") {
            std::cerr << "Unknown option: " << argument << '\n';
            print_usage(argv[0]);
            return 2;
        }
        if (++i >= argc) {
            std::cerr << "Missing value for " << argument << '\n';
            print_usage(argv[0]);
            return 2;
        }
        const std::string value(argv[i]);
        try {
            if (argument == "--atol") options.absolute_tolerance = std::stod(value);
            else if (argument == "--rtol") options.relative_tolerance = std::stod(value);
            else if (value == "rms") options.error_norm = asdole::AdaptiveErrorNorm::weighted_rms;
            else if (value == "inf") options.error_norm = asdole::AdaptiveErrorNorm::weighted_infinity;
            else throw std::invalid_argument("norm must be rms or inf");
        } catch (const std::exception& error) {
            std::cerr << "Invalid value for " << argument << ": " << error.what() << '\n';
            return 2;
        }
    }

    std::cout << std::setprecision(12)
              << "atol,rtol,norm,theta,accepted_steps,rejected_steps,final_dt,final_state,exact_state,global_error,max_accepted_estimate\n";
    for (double theta : theta_values) {
        const asdole::Stepper stepper(rhs, jacobian, theta);
        const asdole::StepSizeController controller(theta, options);

        asdole::Vector state{initial_state};
        double time = 0.0;
        double h = 0.5;
        double last_step = h;
        double max_accepted_estimate = 0.0;
        int accepted = 0;
        int rejected = 0;
        int attempts = 0;
        while (time < final_time) {
            if (++attempts > 100000) throw std::runtime_error("adaptive solve exceeded attempt limit");
            h = std::min(h, final_time - time);
            const asdole::StepResult trial = stepper.step(state, h);
            const asdole::StepSizeDecision decision = controller.assess(state, trial, h);
            if (decision.accept) {
                state = trial.state;
                time += h;
                last_step = h;
                max_accepted_estimate = std::max(max_accepted_estimate, decision.scaled_error);
                ++accepted;
            } else {
                ++rejected;
                if (h <= options.minimum_step * (1.0 + 1.0e-12))
                    throw std::runtime_error("adaptive solve reached minimum step after rejection");
            }
            h = decision.next_step;
        }

        const double exact = initial_state * std::exp(-decay_rate * final_time);
        const double global_error = std::abs(state[0] - exact);
        std::cout << options.absolute_tolerance << ',' << options.relative_tolerance << ','
                  << norm_name(options.error_norm) << ',' << theta << ',' << accepted << ','
                  << rejected << ',' << last_step << ','
                  << state[0] << ',' << exact << ',' << global_error << ','
                  << max_accepted_estimate << '\n';
        if (!std::isfinite(global_error) || global_error > 5.0e-5
            || accepted == 0 || rejected == 0 || max_accepted_estimate > 1.0)
            return 1;
    }
}
