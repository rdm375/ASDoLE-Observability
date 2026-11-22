// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/contraction.hpp"
#include "asdole/covariance.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace asdole;
constexpr double theta = 0.7;
constexpr double final_time = 12.0;
constexpr int steps = 1200;
constexpr double dt = final_time / steps;

void rhs(std::span<const double> y, std::span<double> f) {
    f[0] = -0.7*y[0] + 0.15*y[1] + 0.04*y[0]*y[0] + 0.03*y[0]*y[1];
    f[1] = -0.2*y[0] - 1.1*y[1] + 0.02*y[0]*y[1] + 0.05*y[1]*y[1];
}

void jacobian(std::span<const double> y, DenseMatrix& j) {
    j(0,0) = -0.7 + 0.08*y[0] + 0.03*y[1];
    j(0,1) = 0.15 + 0.03*y[0];
    j(1,0) = -0.2 + 0.02*y[1];
    j(1,1) = -1.1 + 0.02*y[0] + 0.10*y[1];
}

void hessian(std::span<const double>, HessianTensor& h) {
    h[0](0,0) = 0.08;
    h[0](0,1) = h[0](1,0) = 0.03;
    h[1](0,1) = h[1](1,0) = 0.02;
    h[1](1,1) = 0.10;
}

double norm2(std::span<const double> v) {
    double value = 0.0;
    for (double x : v) value += x*x;
    return std::sqrt(value);
}

double box_symmetric_jacobian_max_eigenvalue() {
    double maximum = -std::numeric_limits<double>::infinity();
    for (double x : {-1.0, 1.0}) {
        for (double y : {-1.0, 1.0}) {
            DenseMatrix j(2), symmetric(2);
            const Vector point{x,y};
            jacobian(point, j);
            for (std::size_t i = 0; i < 2; ++i)
                for (std::size_t k = 0; k < 2; ++k)
                    symmetric(i,k) = 0.5*(j(i,k)+j(k,i));
            maximum = std::max(maximum, symmetric_eigenvalues(symmetric).front());
        }
    }
    return maximum;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string output = argc > 1 ? argv[1]
            : "results/contraction-analysis/nonlinear_contraction.csv";
        const std::filesystem::path output_path(output);
        if (output_path.has_parent_path())
            std::filesystem::create_directories(output_path.parent_path());

        // The box [-1,1]^2 is forward invariant: each component of f points
        // strictly inward on its two corresponding faces. The symmetric part
        // of J is affine in (x,y), so convexity of lambda_max bounds its maximum
        // over this box by the maximum at its four vertices.
        const double max_symmetric_eigenvalue = box_symmetric_jacobian_max_eigenvalue();
        const double certified_rate = -max_symmetric_eigenvalue;
        if (!(certified_rate > 0.0))
            throw std::runtime_error("nonlinear vector field is not contractive on the chosen box");

        const Stepper stepper(rhs, jacobian, theta);
        std::vector<Vector> states(static_cast<std::size_t>(steps + 1));
        std::vector<Vector> perturbed_states(static_cast<std::size_t>(steps + 1));
        std::vector<double> step_sizes(static_cast<std::size_t>(steps), dt);
        states[0] = {0.8, -0.35};
        const Vector initial_difference{0.04, -0.03};
        perturbed_states[0] = {states[0][0] + initial_difference[0],
                               states[0][1] + initial_difference[1]};
        for (int k = 0; k < steps; ++k) {
            states[static_cast<std::size_t>(k+1)] = stepper.step(
                states[static_cast<std::size_t>(k)], dt).state;
            perturbed_states[static_cast<std::size_t>(k+1)] = stepper.step(
                perturbed_states[static_cast<std::size_t>(k)], dt).state;
        }
        const DiscreteTrajectoryLinearization trajectory(stepper, hessian, states, step_sizes);

        DenseMatrix identity(2);
        identity(0,0) = identity(1,1) = 1.0;
        const auto discrete_steps = discrete_contraction_history(trajectory, identity);
        const auto gains = transient_amplification_history(trajectory);

        DenseMatrix initial_covariance(2);
        initial_covariance(0,0) = 0.01*0.01;
        initial_covariance(1,1) = 0.008*0.008;
        const auto covariance = propagate_covariance(trajectory, initial_covariance);

        std::ofstream csv(output);
        if (!csv) throw std::runtime_error("could not open output CSV: " + output);
        csv << std::setprecision(12)
            << "time,finite_pair_error,tlm_error,finite_to_tlm_defect,"
               "estimated_tlm_gain,covariance_trace,step_induced_factor,"
               "cumulative_discrete_bound\n";
        double cumulative_bound = 1.0;
        double maximum_finite_tlm_defect = 0.0;
        csv << 0.0 << ',' << norm2(initial_difference) << ',' << norm2(initial_difference)
            << ",0,1," << covariance[0](0,0)+covariance[0](1,1) << ",NA,1\n";
        double max_discrete_factor = 0.0;
        for (int k = 0; k < steps; ++k) {
            const auto node = static_cast<std::size_t>(k+1);
            const Vector finite_difference{
                perturbed_states[node][0]-states[node][0],
                perturbed_states[node][1]-states[node][1]};
            const Vector tangent_difference = trajectory.apply(initial_difference, node);
            const double finite_norm = norm2(finite_difference);
            const double tangent_norm = norm2(tangent_difference);
            const double pair_defect = norm2(Vector{
                finite_difference[0]-tangent_difference[0],
                finite_difference[1]-tangent_difference[1]});
            maximum_finite_tlm_defect = std::max(maximum_finite_tlm_defect, pair_defect);
            const auto& step = discrete_steps[static_cast<std::size_t>(k)];
            max_discrete_factor = std::max(max_discrete_factor, step.induced_factor);
            cumulative_bound *= step.induced_factor;
            csv << node*dt << ',' << finite_norm << ',' << tangent_norm << ','
                << pair_defect << ',' << gains[node].gain << ','
                << covariance[node](0,0)+covariance[node](1,1) << ','
                << step.induced_factor << ',' << cumulative_bound << '\n';
        }

        const double initial_norm = norm2(initial_difference);
        const double final_pair_norm = norm2(Vector{
            perturbed_states.back()[0]-states.back()[0],
            perturbed_states.back()[1]-states.back()[1]});
        const double final_tlm_norm = norm2(trajectory.apply(initial_difference, steps));
        const double initial_trace = initial_covariance(0,0)+initial_covariance(1,1);
        const double final_trace = covariance.back()(0,0)+covariance.back()(1,1);
        const bool all_discrete_steps_contract = std::all_of(
            discrete_steps.begin(), discrete_steps.end(),
            [](const auto& item) { return item.contracts; });

        std::cout << std::setprecision(12)
                  << "model=polynomial_two_state theta=" << theta
                  << " dt=" << dt << " steps=" << steps
                  << " final_time=" << final_time
                  << "\nregion=[-1,1]^2 forward_invariant=yes"
                  << "\ncontinuous_euclidean_contraction_rate=" << certified_rate
                  << "\nfinite_pair_error_initial=" << initial_norm
                  << " finite_pair_error_final=" << final_pair_norm
                  << " finite_pair_ratio=" << final_pair_norm/initial_norm
                  << "\nmaximum_finite_to_tlm_defect=" << maximum_finite_tlm_defect
                  << "\ntlm_error_final=" << final_tlm_norm
                  << " tlm_gain_final=" << gains.back().gain
                  << "\ncovariance_trace_initial=" << initial_trace
                  << " covariance_trace_final=" << final_trace
                  << " covariance_trace_ratio=" << final_trace/initial_trace
                  << "\ndiscrete_all_steps_contract="
                  << (all_discrete_steps_contract ? "yes" : "no")
                  << " max_local_discrete_factor=" << max_discrete_factor
                  << " discrete_product_bound=" << cumulative_bound
                  << "\noutput_csv=" << output << '\n';
        return all_discrete_steps_contract ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "nonlinear_contraction_analysis: " << error.what() << '\n';
        return 2;
    }
}
