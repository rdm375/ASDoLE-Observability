// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/contraction.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace asdole;
constexpr double theta = 0.7;
constexpr double final_time = 12.0;
constexpr int steps = 480;
constexpr double dt = final_time / steps;
constexpr double rate_safety_fraction = 0.8;

void rhs(std::span<const double> x, std::span<double> f) {
    f[0] = -x[0];
    f[1] = x[0] - x[1];
}
void jacobian(std::span<const double>, DenseMatrix& a) {
    a(0,0) = -1.0; a(0,1) = 0.0;
    a(1,0) = 1.0;  a(1,1) = -1.0;
}
void zero_hessian(std::span<const double>, HessianTensor&) {}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string output = argc > 1 ? argv[1]
            : "results/contraction-analysis/contraction.csv";
        const std::filesystem::path path(output);
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());

        DenseMatrix a(2);
        a(0,0) = -1.0; a(0,1) = 0.0;
        a(1,0) = 1.0;  a(1,1) = -1.0;
        const auto selected_metric = estimate_lyapunov_metric(a, rate_safety_fraction);
        const DenseMatrix& w = selected_metric.weight;
        const double certified_rate = selected_metric.requested_rate;
        const auto continuous = certify_continuous_contraction(a, w, certified_rate);
        if (!continuous.certified)
            throw std::runtime_error("expected Lyapunov contraction certificate did not pass");

        const Stepper stepper(rhs, jacobian, theta);
        std::vector<Vector> states(static_cast<std::size_t>(steps + 1));
        std::vector<double> hs(static_cast<std::size_t>(steps), dt);
        states[0] = {1.0, 0.7};
        for (int k = 0; k < steps; ++k)
            states[static_cast<std::size_t>(k + 1)] = stepper.step(
                states[static_cast<std::size_t>(k)], dt).state;
        const DiscreteTrajectoryLinearization trajectory(stepper, zero_hessian, states, hs);
        const auto discrete = discrete_contraction_history(trajectory, w);
        const double determinant = w(0,0)*w(1,1) - w(0,1)*w(1,0);
        const StateMetric metric{
            [&w](std::span<const double> x) {
                Vector y(2, 0.0);
                for (std::size_t i = 0; i < 2; ++i)
                    for (std::size_t j = 0; j < 2; ++j) y[i] += w(i,j)*x[j];
                return y;
            },
            [&w, determinant](std::span<const double> x) {
            return Vector{(w(1,1)*x[0]-w(0,1)*x[1])/determinant,
                          (-w(1,0)*x[0]+w(0,0)*x[1])/determinant};
            }
        };
        const auto gains = transient_amplification_history(trajectory, metric);

        std::ofstream csv(output);
        if (!csv) throw std::runtime_error("could not open output CSV: " + output);
        csv << std::setprecision(12)
            << "time,step_induced_factor,step_certified_rate,cumulative_contraction_bound,"
               "estimated_worst_case_gain\n";
        double cumulative_bound = 1.0;
        csv << 0.0 << ",NA,NA," << cumulative_bound << ',' << gains[0].gain << '\n';
        for (int k = 0; k < steps; ++k) {
            const auto& item = discrete[static_cast<std::size_t>(k)];
            cumulative_bound *= item.induced_factor;
            csv << (k + 1)*dt << ',' << item.induced_factor << ','
                << item.certified_rate << ',' << cumulative_bound << ','
                << gains[static_cast<std::size_t>(k + 1)].gain << '\n';
        }

        DenseMatrix symmetric_a(2);
        symmetric_a(0,0) = -1.0; symmetric_a(1,1) = -1.0;
        symmetric_a(0,1) = symmetric_a(1,0) = 0.5;
        const double euclidean_rate = -symmetric_eigenvalues(symmetric_a).front();
        const double largest_factor = std::max_element(discrete.begin(), discrete.end(),
            [](const auto& l, const auto& r) { return l.induced_factor < r.induced_factor; })
            ->induced_factor;
        std::cout << std::setprecision(12)
                  << "model=x'=-x, y'=x-y theta=" << theta
                  << " dt=" << dt << " steps=" << steps
                  << " final_time=" << final_time
                  << "\ncontinuous_lyapunov_certificate=" << (continuous.certified ? "PASS" : "FAIL")
                  << " requested_rate=" << certified_rate
                  << " estimated_rate_limit=" << selected_metric.estimated_rate_limit
                  << " residual_max_eigenvalue=" << continuous.largest_residual_eigenvalue
                  << " W_eigenvalues=[" << continuous.smallest_weight_eigenvalue << ','
                  << continuous.largest_weight_eigenvalue << ']'
                  << "\neuclidean_contraction_rate=" << euclidean_rate
                  << "\ndiscrete_all_steps_contract="
                  << (std::all_of(discrete.begin(), discrete.end(),
                      [](const auto& s) { return s.contracts; }) ? "yes" : "no")
                  << " max_step_factor=" << largest_factor
                  << " minimum_step_rate=" << -std::log(largest_factor)/dt
                  << "\nfinal_estimated_W_norm_gain=" << gains.back().gain
                  << " final_product_of_step_factors=" << cumulative_bound
                  << "\noutput_csv=" << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "contraction_analysis: " << error.what() << '\n';
        return 2;
    }
}
