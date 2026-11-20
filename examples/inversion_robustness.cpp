// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/analysis.hpp"
#include "asdole/inverse.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using asdole::DenseMatrix;
using asdole::Vector;
constexpr double final_time = 4.0;
constexpr double truth_y10 = 1.0;
constexpr double known_y20 = 0.7;
constexpr double truth_k = 1.0;
constexpr double theta = 0.7;
constexpr std::array<std::size_t, 3> step_counts{20, 40, 80};
constexpr std::array<double, 4> relative_noise_levels{0.0, 0.01, 0.05, 0.20};
constexpr std::array<double, 3> regularization_strengths{0.0, 1.0, 10.0};
constexpr std::size_t noisy_realizations = 12;
const Vector prior_mean{0.9, 0.8};
const Vector prior_scale{0.5, 0.5};
const Vector initial_guess{0.65, 0.55};
const asdole::HessianFunction zero_hessian = [](std::span<const double>, asdole::HessianTensor& hessian) {
    for (auto& component : hessian) component.fill(0.0);
};

asdole::Stepper make_stepper(double k) {
    const auto rhs = [k](std::span<const double> y, std::span<double> f) {
        f[0] = -y[0] + k * y[1];
        f[1] = -2.0 * y[1];
    };
    const auto jacobian = [k](std::span<const double>, DenseMatrix& j) {
        j(0, 0) = -1.0; j(0, 1) = k;
        j(1, 0) = 0.0;  j(1, 1) = -2.0;
    };
    return asdole::Stepper(rhs, jacobian, theta);
}

double exact_observation(double time) {
    return truth_y10 * std::exp(-time)
         + truth_k * known_y20 * (std::exp(-time) - std::exp(-2.0 * time));
}

std::vector<Vector> integrate(const Vector& parameters, std::size_t steps) {
    const auto stepper = make_stepper(parameters[1]);
    const double h = final_time / static_cast<double>(steps);
    std::vector<Vector> states(steps + 1, Vector(2));
    states[0] = {parameters[0], known_y20};
    for (std::size_t k = 0; k < steps; ++k) states[k + 1] = stepper.step(states[k], h).state;
    return states;
}

struct ObservationData {
    std::string schedule;
    std::size_t steps{};
    double h{};
    double relative_noise{};
    double noise_standard_deviation{};
    double precision{};
    std::vector<std::size_t> nodes;
    Vector times;
    Vector values;
    Vector quadrature_weights;
};

ObservationData make_observations(std::string schedule,
                                  std::size_t steps,
                                  double relative_noise,
                                  std::size_t realization) {
    ObservationData result;
    result.schedule = std::move(schedule);
    result.steps = steps;
    result.h = final_time / static_cast<double>(steps);
    result.relative_noise = relative_noise;
    const std::size_t last_node = result.schedule == "full_window" ? steps : steps / 4;
    for (std::size_t node = 0; node <= last_node; ++node) result.nodes.push_back(node);
    result.times.reserve(result.nodes.size());
    result.values.reserve(result.nodes.size());
    result.quadrature_weights.resize(result.nodes.size());
    double weighted_signal_square = 0.0;
    double interval = 0.0;
    for (std::size_t i = 0; i < result.nodes.size(); ++i) {
        const double time = result.nodes[i] * result.h;
        result.times.push_back(time);
        result.quadrature_weights[i] = (i == 0 || i + 1 == result.nodes.size())
            ? 0.5 * result.h : result.h;
        const double value = exact_observation(time);
        weighted_signal_square += result.quadrature_weights[i] * value * value;
        interval += result.quadrature_weights[i];
    }
    const double signal_rms = std::sqrt(weighted_signal_square / interval);
    result.noise_standard_deviation = relative_noise * signal_rms;
    result.precision = result.noise_standard_deviation > 0.0
        ? 1.0 / (result.noise_standard_deviation * result.noise_standard_deviation) : 1.0;

    const std::uint64_t seed = 0x4153444f4c45ULL
        + static_cast<std::uint64_t>(steps) * 100003ULL
        + static_cast<std::uint64_t>(std::llround(relative_noise * 1000.0)) * 1009ULL
        + static_cast<std::uint64_t>(realization);
    std::mt19937_64 generator(seed);
    if (result.noise_standard_deviation > 0.0) {
        std::normal_distribution<double> normal(0.0, result.noise_standard_deviation);
        for (double time : result.times)
            result.values.push_back(exact_observation(time) + normal(generator));
    } else {
        for (double time : result.times) result.values.push_back(exact_observation(time));
    }
    return result;
}

struct FitLinearization {
    double objective{};
    Vector gradient;
    DenseMatrix gauss_newton;
    double unweighted_residual_rms{};
};

double prior_objective(std::span<const double> parameters, double strength) {
    double value = 0.0;
    for (std::size_t j = 0; j < parameters.size(); ++j) {
        const double scaled = (parameters[j] - prior_mean[j]) / prior_scale[j];
        value += 0.5 * strength * scaled * scaled;
    }
    return value;
}

double objective_only(std::span<const double> parameters,
                      const ObservationData& data,
                      double regularization) {
    const auto states = integrate(Vector(parameters.begin(), parameters.end()), data.steps);
    double objective = prior_objective(parameters, regularization);
    for (std::size_t i = 0; i < data.nodes.size(); ++i) {
        const double residual = states[data.nodes[i]][0] - data.values[i];
        objective += 0.5 * data.quadrature_weights[i] * data.precision * residual * residual;
    }
    return objective;
}

FitLinearization linearize(std::span<const double> parameters,
                           const ObservationData& data,
                           double regularization) {
    const Vector p(parameters.begin(), parameters.end());
    const auto stepper = make_stepper(p[1]);
    const double h = data.h;
    auto states = std::make_shared<std::vector<Vector>>(integrate(p, data.steps));
    auto trajectory = std::make_shared<asdole::DiscreteTrajectoryLinearization>(
        stepper, zero_hessian, *states, std::vector<double>(data.steps, h));

    DenseMatrix sensor_matrix(1, 2);
    sensor_matrix(0, 0) = 1.0;
    const auto sensor = asdole::dense_observation_operator(std::move(sensor_matrix));
    std::vector<asdole::ParameterObservationSample> samples;
    samples.reserve(data.nodes.size());
    const auto precision = data.precision;
    for (std::size_t i = 0; i < data.nodes.size(); ++i) {
        const auto node = data.nodes[i];
        samples.push_back({node, sensor, data.quadrature_weights[i],
            [precision](std::span<const double> value) {
                return Vector{precision * value[0]};
            }, {}, {}});
    }

    const auto initial_apply = [](std::span<const double> direction) {
        return Vector{direction[0], 0.0};
    };
    const auto initial_transpose = [](std::span<const double> state_cotangent) {
        return Vector{state_cotangent[0], 0.0};
    };
    const auto step_apply = [stepper](std::size_t, std::span<const double> state,
                                      double step_size, std::span<const double> direction) {
        const double dk = direction[1];
        const Vector rhs_parameter{dk * state[1], 0.0};
        DenseMatrix jacobian_parameter(2);
        jacobian_parameter(0, 1) = dk;
        return asdole::parameter_tangent_step(
            stepper, state, step_size, rhs_parameter, jacobian_parameter);
    };
    const auto step_transpose = [step_apply](std::size_t step, std::span<const double> state,
                                              double step_size, std::span<const double> cotangent) {
        Vector result(2, 0.0);
        for (std::size_t j = 0; j < 2; ++j) {
            Vector basis(2, 0.0);
            basis[j] = 1.0;
            result[j] = asdole::inner_product(step_apply(step, state, step_size, basis), cotangent);
        }
        return result;
    };
    const asdole::ParameterObservationSensitivityOperator sensitivity(
        *trajectory, 2, initial_apply, initial_transpose, step_apply,
        step_transpose, std::move(samples));

    Vector residual(data.nodes.size());
    double objective = prior_objective(parameters, regularization);
    double residual_square = 0.0;
    for (std::size_t i = 0; i < data.nodes.size(); ++i) {
        residual[i] = (*states)[data.nodes[i]][0] - data.values[i];
        residual_square += residual[i] * residual[i];
        objective += 0.5 * data.quadrature_weights[i] * data.precision
                   * residual[i] * residual[i];
    }

    Vector gradient = sensitivity.weighted_transpose_apply(residual);
    DenseMatrix gauss_newton(2, 2);
    for (std::size_t column = 0; column < 2; ++column) {
        Vector basis(2, 0.0);
        basis[column] = 1.0;
        const Vector action = sensitivity.weighted_gram_action(basis);
        for (std::size_t row = 0; row < 2; ++row) gauss_newton(row, column) = action[row];
    }
    for (std::size_t j = 0; j < 2; ++j) {
        const double prior_precision = regularization / (prior_scale[j] * prior_scale[j]);
        gradient[j] += prior_precision * (parameters[j] - prior_mean[j]);
        gauss_newton(j, j) += prior_precision;
    }
    return {objective, std::move(gradient), std::move(gauss_newton),
            std::sqrt(residual_square / static_cast<double>(residual.size()))};
}

struct FitSummary {
    asdole::GaussNewtonResult optimization;
    double parameter_error_y10{};
    double parameter_error_k{};
    double scaled_parameter_error{};
    double residual_rms{};
    asdole::SensitivityDiagnostics identifiability;
};

FitSummary fit(const ObservationData& data, double regularization) {
    const auto make_linearization = [&data, regularization](std::span<const double> p) {
        const auto value = linearize(p, data, regularization);
        return asdole::LeastSquaresLinearization{
            value.objective, value.gradient, value.gauss_newton};
    };
    const auto objective = [&data, regularization](std::span<const double> p) {
        return objective_only(p, data, regularization);
    };
    asdole::GaussNewtonOptions options;
    options.maximum_iterations = 60;
    options.gradient_tolerance = 1.0e-8;
    const auto result = asdole::damped_gauss_newton(
        make_linearization, objective, initial_guess, options);
    const auto final = linearize(result.parameters, data, regularization);

    const double y10_error = result.parameters[0] - truth_y10;
    const double k_error = result.parameters[1] - truth_k;
    // Recompute the local data identifiability without including the prior.
    const auto unregularized = linearize(result.parameters, data, 0.0);
    DenseMatrix data_gram = unregularized.gauss_newton;
    // Remove no regularization here: this evaluation used strength zero.
    const auto eigenvalues = asdole::symmetric_eigenvalues(std::move(data_gram));
    asdole::SensitivityDiagnostics diagnostics;
    diagnostics.relative_rank_threshold = 1.0e-8;
    for (double value : eigenvalues)
        diagnostics.singular_values.push_back(std::sqrt(std::max(0.0, value)));
    const double threshold = diagnostics.singular_values.front() * diagnostics.relative_rank_threshold;
    for (double sigma : diagnostics.singular_values)
        if (sigma > threshold) ++diagnostics.numerical_rank;
    diagnostics.condition_number = diagnostics.numerical_rank == 2
        ? diagnostics.singular_values.front() / diagnostics.singular_values.back()
        : std::numeric_limits<double>::infinity();

    return {result, y10_error, k_error,
        std::hypot(y10_error / prior_scale[0], k_error / prior_scale[1]),
        final.unweighted_residual_rms, std::move(diagnostics)};
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path output_directory = argc > 1
        ? std::filesystem::path(argv[1]) : std::filesystem::path("results/inversion-robustness");
    std::filesystem::create_directories(output_directory);
    std::ofstream csv(output_directory / "inversion_robustness.csv");
    if (!csv) throw std::runtime_error("could not open inversion robustness CSV output");
    csv << "schedule,steps,dt,relative_noise,noise_sd,regularization,realization,"
           "estimate_y10,estimate_k,error_y10,error_k,scaled_parameter_error,"
           "residual_rms,objective,rank,condition,iterations,gradient_norm,converged\n";
    csv << std::setprecision(16);

    bool all_converged = true;
    for (const std::size_t steps : step_counts) {
        for (const std::string schedule : {"full_window", "early_window"}) {
            for (const double relative_noise : relative_noise_levels) {
                const std::size_t realizations = relative_noise == 0.0 ? 1 : noisy_realizations;
                for (std::size_t realization = 0; realization < realizations; ++realization) {
                    const ObservationData data = make_observations(
                        schedule, steps, relative_noise, realization);
                    for (const double regularization : regularization_strengths) {
                        const FitSummary summary = fit(data, regularization);
                        const auto& result = summary.optimization;
                        all_converged = all_converged && result.converged;
                        csv << schedule << ',' << steps << ',' << data.h << ',' << relative_noise << ','
                            << data.noise_standard_deviation << ',' << regularization << ',' << realization << ','
                            << result.parameters[0] << ',' << result.parameters[1] << ','
                            << summary.parameter_error_y10 << ',' << summary.parameter_error_k << ','
                            << summary.scaled_parameter_error << ',' << summary.residual_rms << ','
                            << result.objective << ',' << summary.identifiability.numerical_rank << ','
                            << summary.identifiability.condition_number << ',' << result.iterations << ','
                            << result.gradient_norm << ',' << (result.converged ? 1 : 0) << '\n';
                    }
                }
            }
        }
    }
    std::cout << "truth_y10=" << truth_y10 << " known_y20=" << known_y20
              << " truth_k=" << truth_k << " theta=" << theta << '\n'
              << "noise levels are relative to the observation RMS; each noisy case uses "
              << noisy_realizations << " seeded Gaussian realizations\n"
              << "schedules=full_window,early_window regularization=0,1,10\n"
              << "wrote=" << (output_directory / "inversion_robustness.csv") << '\n'
              << "all_fits_converged=" << (all_converged ? "yes" : "no") << '\n';
    return all_converged ? 0 : 1;
}
