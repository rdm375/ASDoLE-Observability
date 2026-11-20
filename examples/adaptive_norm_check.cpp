// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/adaptive.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    asdole::AdaptiveOptions options;
    options.absolute_tolerance = 1.0;
    options.relative_tolerance = 0.0;

    asdole::StepResult trial;
    trial.state = {0.0, 0.0, 0.0, 0.0};
    trial.error_estimate = {2.0, 0.0, 0.0, 0.0};
    const asdole::Vector initial(4, 0.0);

    options.error_norm = asdole::AdaptiveErrorNorm::weighted_rms;
    const auto rms = asdole::StepSizeController(0.6, options).assess(initial, trial, 0.1);
    options.error_norm = asdole::AdaptiveErrorNorm::weighted_infinity;
    const auto infinity = asdole::StepSizeController(0.6, options).assess(initial, trial, 0.1);

    const bool passed = rms.accept && std::abs(rms.scaled_error - 1.0) < 1.0e-14
        && !infinity.accept && std::abs(infinity.scaled_error - 2.0) < 1.0e-14;
    std::cout << std::setprecision(12)
              << "rms_scaled_error=" << rms.scaled_error << " rms_accept=" << rms.accept << '\n'
              << "inf_scaled_error=" << infinity.scaled_error << " inf_accept=" << infinity.accept << '\n'
              << (passed ? "result=PASS\n" : "result=FAIL\n");
    return passed ? 0 : 1;
}
