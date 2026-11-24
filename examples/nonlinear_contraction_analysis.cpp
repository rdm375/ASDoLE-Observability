// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/contraction.hpp"
#include "asdole/covariance.hpp"
#include "asdole/interval_dual.hpp"

#include <algorithm>
#include <array>
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

std::vector<Interval> interval_rhs(const IntervalBox& y) {
    return {-0.7*y[0] + 0.15*y[1] + 0.04*y[0]*y[0] + 0.03*y[0]*y[1],
            -0.2*y[0] - 1.1*y[1] + 0.02*y[0]*y[1] + 0.05*y[1]*y[1]};
}

std::vector<std::vector<Interval>> interval_jacobian(const IntervalBox& y) {
    return {{-0.7 + 0.08*y[0] + 0.03*y[1], 0.15 + 0.03*y[0]},
            {-0.2 + 0.02*y[1], -1.1 + 0.02*y[0] + 0.10*y[1]}};
}

struct IntervalStepEvaluation {
    std::array<Interval, 2> state;
    std::array<std::array<Interval, 2>, 2> jacobian;
};

IntervalStepEvaluation interval_asdole_step(const IntervalBox& box,
                                           double h, double step_theta) {
    using Dual = IntervalDual<2>;
    if (box.size() != 2) throw std::invalid_argument("two-state ASDoLE enclosure needs a 2D box");
    const std::array<Dual, 2> y{
        Dual::variable(box[0], 0), Dual::variable(box[1], 1)};
    const std::array<Dual, 2> f{
        -0.7*y[0] + 0.15*y[1] + 0.04*square(y[0]) + 0.03*y[0]*y[1],
        -0.2*y[0] - 1.1*y[1] + 0.02*y[0]*y[1] + 0.05*square(y[1])};
    const std::array<std::array<Dual, 2>, 2> j{{
        {{-0.7 + 0.08*y[0] + 0.03*y[1], 0.15 + 0.03*y[0]}},
        {{-0.2 + 0.02*y[1], -1.1 + 0.02*y[0] + 0.10*y[1]}}}};

    const auto solve = [](const std::array<std::array<Dual, 2>, 2>& a,
                          const std::array<Dual, 2>& rhs) {
        const Dual determinant = a[0][0]*a[1][1] - a[0][1]*a[1][0];
        return std::array<Dual, 2>{
            (a[1][1]*rhs[0] - a[0][1]*rhs[1])/determinant,
            (a[0][0]*rhs[1] - a[1][0]*rhs[0])/determinant};
    };
    const auto matrix = [&](double scale) {
        return std::array<std::array<Dual, 2>, 2>{{
            {{1.0-step_theta*scale*j[0][0], -step_theta*scale*j[0][1]}},
            {{-step_theta*scale*j[1][0], 1.0-step_theta*scale*j[1][1]}}}};
    };

    const auto full = solve(matrix(h), {h*f[0], h*f[1]});
    const double half_h = 0.5*h;
    const auto half_matrix = matrix(half_h);
    const auto first = solve(half_matrix, {half_h*f[0], half_h*f[1]});
    const std::array<Dual, 2> second_rhs{
        half_h*(f[0]+j[0][0]*first[0]+j[0][1]*first[1]),
        half_h*(f[1]+j[1][0]*first[0]+j[1][1]*first[1])};
    const auto second = solve(half_matrix, second_rhs);
    const std::array<Dual, 2> next{
        y[0]+2.0*first[0]+2.0*second[0]-full[0],
        y[1]+2.0*first[1]+2.0*second[1]-full[1]};

    IntervalStepEvaluation result;
    for (std::size_t i = 0; i < 2; ++i) {
        result.state[i] = next[i].value;
        for (std::size_t k = 0; k < 2; ++k) result.jacobian[i][k] = next[i].gradient[k];
    }
    return result;
}

IntervalDiscreteMap interval_asdole_map_enclosure(double h, double step_theta) {
    return [h, step_theta](const IntervalBox& box) {
        const auto jacobian = interval_asdole_step(box, h, step_theta).jacobian;
        IntervalBox center_box;
        center_box.reserve(box.size());
        for (const auto& coordinate : box)
            center_box.emplace_back(0.5*(coordinate.lower+coordinate.upper));
        const auto at_center = interval_asdole_step(center_box, h, step_theta).state;
        std::vector<Interval> image(2, Interval(0.0));
        for (std::size_t i = 0; i < 2; ++i) {
            image[i] = at_center[i];
            for (std::size_t k = 0; k < 2; ++k) {
                const double center = center_box[k].lower;
                image[i] = image[i] + jacobian[i][k]*(box[k]-center);
            }
        }
        return image;
    };
}

IntervalDiscreteJacobian interval_asdole_jacobian_enclosure(double h, double step_theta) {
    return [h, step_theta](const IntervalBox& box) {
        const auto j = interval_asdole_step(box, h, step_theta).jacobian;
        return std::vector<std::vector<Interval>>{{j[0][0], j[0][1]}, {j[1][0], j[1][1]}};
    };
}

bool contains_with_roundoff(const Interval& enclosure, double value) {
    constexpr double allowance = 2.0e-12;
    return value >= enclosure.lower-allowance && value <= enclosure.upper+allowance;
}

void check_interval_asdole_enclosures(const IntervalBox& region, double h, double step_theta) {
    const auto map_enclosure = interval_asdole_map_enclosure(h, step_theta)(region);
    const auto jacobian_enclosure = interval_asdole_jacobian_enclosure(h, step_theta)(region);
    const Stepper stepper(rhs, jacobian, step_theta);
    std::vector<Vector> samples;
    for (double x : {region[0].lower, 0.0, region[0].upper})
        for (double y : {region[1].lower, 0.0, region[1].upper}) samples.push_back({x,y});
    for (const auto& point : samples) {
        const auto stepped = stepper.step(point, h).state;
        for (std::size_t i = 0; i < 2; ++i)
            if (!contains_with_roundoff(map_enclosure[i], stepped[i]))
                throw std::runtime_error("interval discrete-map enclosure missed an ASDoLE sample");
        for (std::size_t column = 0; column < 2; ++column) {
            Vector basis(2, 0.0);
            basis[column] = 1.0;
            const auto tangent = tangent_step(stepper, hessian, point, h, basis);
            for (std::size_t row = 0; row < 2; ++row)
                if (!contains_with_roundoff(jacobian_enclosure[row][column], tangent[row]))
                    throw std::runtime_error("interval TLM enclosure missed an ASDoLE tangent sample");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string output = argc > 1 ? argv[1]
            : "results/contraction-analysis/nonlinear_contraction.csv";
        const std::filesystem::path output_path(output);
        if (output_path.has_parent_path())
            std::filesystem::create_directories(output_path.parent_path());

        // The interval callbacks enclose the field and Jacobian over the full
        // box. The reusable certificate checks inward-pointing faces and a
        // uniform Euclidean logarithmic-norm bound.
        const IntervalBox region{Interval(-1.0, 1.0), Interval(-1.0, 1.0)};
        const auto regional = certify_nonlinear_box(
            region, interval_rhs, interval_jacobian);
        if (!regional.forward_invariant)
            throw std::runtime_error("nonlinear vector field is not certified forward invariant on the chosen box");
        if (!regional.contraction_certified)
            throw std::runtime_error("nonlinear vector field is not certified contractive on the chosen box");
        const double certified_rate = regional.contraction_rate;

        const auto discrete_regional = certify_discrete_box_contraction(
            region, interval_asdole_map_enclosure(dt, theta),
            interval_asdole_jacobian_enclosure(dt, theta));
        if (!discrete_regional.maps_into_box || !discrete_regional.contraction_certified)
            throw std::runtime_error("discrete ASDoLE step is not certified invariant and contractive on the box");
        if (argc > 1 && std::string(argv[1]) == "--check-discrete") {
            check_interval_asdole_enclosures(region, dt, theta);
            std::cout << std::setprecision(12)
                      << "two_state_discrete_box_invariant=yes induced_factor_bound="
                      << discrete_regional.induced_factor_bound
                      << " sampled_state_tlm_enclosures=PASS\n";
            return 0;
        }
        if (argc > 1 && std::string(argv[1]).starts_with("--"))
            throw std::invalid_argument("unknown option: " + std::string(argv[1]));

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
                  << "\ndiscrete_regional_map_into_box="
                  << (discrete_regional.maps_into_box ? "yes" : "no")
                  << " induced_factor_bound=" << discrete_regional.induced_factor_bound
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
