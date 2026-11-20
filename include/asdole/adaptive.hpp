// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/stepper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace asdole {

enum class AdaptiveErrorNorm {
    weighted_rms,
    weighted_infinity
};

struct AdaptiveOptions {
    double absolute_tolerance{1.0e-8};
    double relative_tolerance{1.0e-6};
    double safety_factor{0.9};
    double minimum_factor{0.2};
    double maximum_factor{5.0};
    double minimum_step{1.0e-12};
    double maximum_step{1.0};
    AdaptiveErrorNorm error_norm{AdaptiveErrorNorm::weighted_infinity};
};

struct StepSizeDecision {
    bool accept{};
    double scaled_error{};
    double next_step{};
};

// Controls the step using StepResult::error_estimate = D-S. The estimator is
// O(h^2) for theta != 0.5 and O(h^3) at theta == 0.5, so the corresponding
// powers are used in the proportional step-size update.
class StepSizeController {
public:
    explicit StepSizeController(double theta, AdaptiveOptions options = {})
        : theta_(theta), options_(options) {
        if (!std::isfinite(theta_)) throw std::invalid_argument("theta must be finite");
        if (!(options_.absolute_tolerance > 0.0) || !std::isfinite(options_.absolute_tolerance))
            throw std::invalid_argument("absolute tolerance must be finite and positive");
        if (!(options_.relative_tolerance >= 0.0) || !std::isfinite(options_.relative_tolerance))
            throw std::invalid_argument("relative tolerance must be finite and nonnegative");
        if (!(options_.safety_factor > 0.0 && options_.safety_factor <= 1.0))
            throw std::invalid_argument("safety factor must be in (0, 1]");
        if (!(options_.minimum_factor > 0.0 && options_.minimum_factor < 1.0)
            || !(options_.maximum_factor >= 1.0) || !std::isfinite(options_.maximum_factor))
            throw std::invalid_argument("step factors must satisfy 0 < min < 1 <= max");
        if (!(options_.minimum_step > 0.0) || !std::isfinite(options_.minimum_step)
            || !(options_.maximum_step >= options_.minimum_step)
            || !std::isfinite(options_.maximum_step))
            throw std::invalid_argument("step bounds must be finite, positive, and ordered");
    }

    [[nodiscard]] StepSizeDecision assess(std::span<const double> initial,
                                          const StepResult& result,
                                          double h) const {
        if (initial.empty() || initial.size() != result.state.size()
            || initial.size() != result.error_estimate.size())
            throw std::invalid_argument("adaptive step state dimensions do not match");
        if (!(h > 0.0) || !std::isfinite(h))
            throw std::invalid_argument("step size must be finite and positive");

        double sum_squared = 0.0;
        double maximum = 0.0;
        for (std::size_t i = 0; i < initial.size(); ++i) {
            const double scale = options_.absolute_tolerance
                + options_.relative_tolerance * std::max(std::abs(initial[i]), std::abs(result.state[i]));
            const double normalized = result.error_estimate[i] / scale;
            if (!std::isfinite(normalized)) {
                maximum = std::numeric_limits<double>::infinity();
                sum_squared = std::numeric_limits<double>::infinity();
                break;
            }
            sum_squared += normalized * normalized;
            maximum = std::max(maximum, std::abs(normalized));
        }
        const double error = options_.error_norm == AdaptiveErrorNorm::weighted_infinity
            ? maximum
            : std::sqrt(sum_squared / static_cast<double>(initial.size()));
        const bool accept = std::isfinite(error) && error <= 1.0;
        const double factor = error == 0.0
            ? options_.maximum_factor
            : (!std::isfinite(error)
                ? options_.minimum_factor
                : options_.safety_factor * std::pow(1.0 / error, 1.0 / estimator_power()));
        const double bounded_factor = std::clamp(factor, options_.minimum_factor,
                                                 options_.maximum_factor);
        const double next = std::clamp(h * bounded_factor, options_.minimum_step,
                                       options_.maximum_step);
        return {accept, error, next};
    }

    [[nodiscard]] int estimator_power() const noexcept {
        return theta_ == 0.5 ? 3 : 2;
    }

private:
    double theta_;
    AdaptiveOptions options_;
};

} // namespace asdole
