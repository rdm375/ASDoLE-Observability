// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    // A decaying state x drives y, which also decays:
    //     x' = -x
    //     y' =  x - y
    // For x(0)=1, y(0)=0, the exact state is (e^-t, t e^-t).
    // Independent initial errors in x and y have standard deviation 0.1.
    // The x error drives y, inducing cross-covariance as the system evolves.
    constexpr double sigma_initial_x = 0.1;
    constexpr double sigma_initial_y = 0.1;
    constexpr double final_time = 2.0;
    constexpr int output_intervals = 4;

    std::cout << std::setprecision(10)
              << "model=x'=-x, y'=x-y\n"
              << "nominal_initial=[1,0], initial_error_sigmas=["
              << sigma_initial_x << ',' << sigma_initial_y << "] (independent)\n"
              << "time,x,y,sigma_x,sigma_y,induced_delta_y_from_x_1sigma,"
                 "covariance_xy,correlation_xy\n";

    for (int k = 0; k <= output_intervals; ++k) {
        const double time = final_time * k / output_intervals;
        const double decay = std::exp(-time);
        const double x = decay;
        const double y = time * decay;
        const double variance_x = sigma_initial_x * sigma_initial_x * decay * decay;
        const double variance_y = decay * decay
            * (time * time * sigma_initial_x * sigma_initial_x
               + sigma_initial_y * sigma_initial_y);
        const double covariance_xy =
            sigma_initial_x * sigma_initial_x * time * decay * decay;
        const double sigma_x = std::sqrt(variance_x);
        const double sigma_y = std::sqrt(variance_y);
        const double induced_delta_y_from_x =
            sigma_initial_x * time * decay;
        const double correlation = covariance_xy / (sigma_x * sigma_y);

        std::cout << time << ',' << x << ',' << y << ','
                  << sigma_x << ',' << sigma_y << ','
                  << induced_delta_y_from_x << ',' << covariance_xy << ','
                  << correlation << '\n';
    }
}
