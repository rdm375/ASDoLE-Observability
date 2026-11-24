// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/inverse.hpp"
#include "asdole/matrix_free_observations.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using asdole::DenseMatrix;
using asdole::Vector;

constexpr std::size_t grid_size = 48;
constexpr std::size_t time_steps = 100;
constexpr double final_time = 0.5;
constexpr double theta = 0.7;
constexpr std::size_t parameter_dimension = 2;
constexpr std::size_t sensor_count = 3;
constexpr std::size_t samples_per_sensor = 11;
constexpr std::size_t total_observations = sensor_count * samples_per_sensor;

double dx() { return 1.0 / static_cast<double>(grid_size + 1); }

void evaluate_rhs(double diffusivity, double rate,
                  std::span<const double> u, std::span<double> f) {
    const double inverse_dx2 = 1.0 / (dx() * dx());
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double left = i == 0 ? 0.0 : u[i - 1];
        const double right = i + 1 == grid_size ? 0.0 : u[i + 1];
        f[i] = diffusivity * (left - 2.0 * u[i] + right) * inverse_dx2
             + rate * u[i] * (1.0 - u[i]);
    }
}

void evaluate_jacobian(double diffusivity, double rate, std::span<const double> u,
                       std::span<const double> v, std::span<double> image) {
    const double inverse_dx2 = 1.0 / (dx() * dx());
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double left = i == 0 ? 0.0 : v[i - 1];
        const double right = i + 1 == grid_size ? 0.0 : v[i + 1];
        image[i] = diffusivity * (left - 2.0 * v[i] + right) * inverse_dx2
                 + rate * (1.0 - 2.0 * u[i]) * v[i];
    }
}

asdole::GmresOptions solver_options() {
    asdole::GmresOptions options;
    options.relative_tolerance = 1.0e-12;
    options.absolute_tolerance = 1.0e-14;
    options.restart = 40;
    options.maximum_iterations = 400;
    return options;
}

asdole::MatrixFreeStepper make_stepper(double diffusivity, double rate) {
    return asdole::MatrixFreeStepper(
        [diffusivity, rate](std::span<const double> u, std::span<double> f) {
            evaluate_rhs(diffusivity, rate, u, f);
        },
        [diffusivity, rate](std::span<const double> u, std::span<const double> v, std::span<double> out) {
            evaluate_jacobian(diffusivity, rate, u, v, out);
        }, theta, solver_options());
}

asdole::MatrixFreeDerivativeActions make_state_derivatives(double diffusivity, double rate) {
    return {
        [diffusivity, rate](std::span<const double> u, std::span<const double> v, std::span<double> out) {
            evaluate_jacobian(diffusivity, rate, u, v, out);
        },
        [rate](std::span<const double>, std::span<const double> direction,
               std::span<const double> vector, std::span<double> out) {
            for (std::size_t i = 0; i < grid_size; ++i)
                out[i] = -2.0 * rate * direction[i] * vector[i];
        },
        [rate](std::span<const double>, std::span<const double> left,
               std::span<const double> right) {
            Vector out(grid_size);
            for (std::size_t i = 0; i < grid_size; ++i)
                out[i] = -2.0 * rate * left[i] * right[i];
            return out;
        }};
}

asdole::MatrixFreeParameterDerivativeActions make_parameter_derivatives(
    double diffusivity, double rate) {
    return {
        [diffusivity, rate](std::span<const double> u, std::span<const double> dp) {
            if (dp.size() != parameter_dimension) throw std::invalid_argument("parameter direction must have dimension two");
            Vector out(grid_size);
            for (std::size_t i = 0; i < grid_size; ++i) {
                const double left = i == 0 ? 0.0 : u[i - 1];
                const double right = i + 1 == grid_size ? 0.0 : u[i + 1];
                const double lap = (left - 2.0 * u[i] + right) / (dx() * dx());
                out[i] = dp[0] * diffusivity * lap + dp[1] * rate * u[i] * (1.0 - u[i]);
            }
            return out;
        },
        [diffusivity, rate](std::span<const double> u, std::span<const double> dp,
           std::span<const double> v, std::span<double> out) {
            if (dp.size() != parameter_dimension) throw std::invalid_argument("parameter direction must have dimension two");
            for (std::size_t i = 0; i < grid_size; ++i) {
                const double left = i == 0 ? 0.0 : v[i - 1];
                const double right = i + 1 == grid_size ? 0.0 : v[i + 1];
                const double lap = (left - 2.0 * v[i] + right) / (dx() * dx());
                out[i] = dp[0] * diffusivity * lap + dp[1] * rate * (1.0 - 2.0 * u[i]) * v[i];
            }
        },
        [diffusivity, rate](std::span<const double> u, std::span<const double> bar_f) {
            Vector gradient(parameter_dimension, 0.0);
            for (std::size_t i = 0; i < grid_size; ++i) {
                const double left = i == 0 ? 0.0 : u[i - 1];
                const double right = i + 1 == grid_size ? 0.0 : u[i + 1];
                const double lap = (left - 2.0 * u[i] + right) / (dx() * dx());
                gradient[0] += diffusivity * lap * bar_f[i];
                gradient[1] += rate * u[i] * (1.0 - u[i]) * bar_f[i];
            }
            return gradient;
        },
        [diffusivity, rate](std::span<const double> u, std::span<const double> left,
           std::span<const double> right) {
            Vector gradient(parameter_dimension, 0.0);
            for (std::size_t i = 0; i < grid_size; ++i) {
                const double lap = ((i == 0 ? 0.0 : right[i - 1]) - 2.0 * right[i]
                    + (i + 1 == grid_size ? 0.0 : right[i + 1])) / (dx() * dx());
                gradient[0] += diffusivity * left[i] * lap;
                gradient[1] += rate * left[i] * (1.0 - 2.0 * u[i]) * right[i];
            }
            return gradient;
        }};
}

Vector initial_state() {
    Vector u(grid_size);
    for (std::size_t i = 0; i < grid_size; ++i) {
        const double x = static_cast<double>(i + 1) * dx();
        u[i] = std::sin(3.14159265358979323846 * x)
             * (0.2 + 0.7 * std::exp(-90.0 * (x - 0.5) * (x - 0.5)));
    }
    return u;
}

struct PdeRun {
    double diffusivity;
    double rate;
    asdole::MatrixFreeStepper stepper;
    std::vector<Vector> states;
    std::vector<double> steps;
    asdole::MatrixFreeDerivativeActions state_derivatives;
    std::unique_ptr<asdole::MatrixFreeTrajectoryLinearization> trajectory;

    explicit PdeRun(double diffusion, double growth_rate)
        : diffusivity(diffusion), rate(growth_rate), stepper(make_stepper(diffusion, growth_rate)),
          states(time_steps + 1, Vector(grid_size)),
          steps(time_steps, final_time / static_cast<double>(time_steps)),
          state_derivatives(make_state_derivatives(diffusion, growth_rate)) {
        states[0] = initial_state();
        for (std::size_t k = 0; k < time_steps; ++k)
            states[k + 1] = stepper.step(states[k], steps[k]).step.state;
        trajectory = std::make_unique<asdole::MatrixFreeTrajectoryLinearization>(
            stepper, state_derivatives, states, steps);
    }
};

std::vector<asdole::ParameterObservationSample> observation_samples() {
    std::vector<asdole::ParameterObservationSample> samples;
    samples.reserve(total_observations);
    const std::vector<std::size_t> sensor_nodes{
        static_cast<std::size_t>(std::round(0.22 / dx())) - 1,
        static_cast<std::size_t>(std::round(0.50 / dx())) - 1,
        static_cast<std::size_t>(std::round(0.78 / dx())) - 1};
    constexpr std::size_t time_stride = time_steps / (samples_per_sensor - 1);
    for (std::size_t time_sample = 0; time_sample < samples_per_sensor; ++time_sample) {
        const std::size_t node = time_sample * time_stride;
        const double time_weight = (time_sample == 0 || time_sample + 1 == samples_per_sensor)
            ? 0.5 * final_time / static_cast<double>(samples_per_sensor - 1)
            : final_time / static_cast<double>(samples_per_sensor - 1);
        for (const std::size_t sensor : sensor_nodes) {
            asdole::LinearObservationOperator observation{
                grid_size, 1,
                [sensor](std::span<const double> u) { return Vector{u[sensor]}; },
                [sensor](std::span<const double> z) {
                    Vector out(grid_size, 0.0); out[sensor] = z[0]; return out;
                }};
            samples.push_back({node, std::move(observation), time_weight, {}, {}, {}});
        }
    }
    return samples;
}

Vector observe(const PdeRun& run, const std::vector<asdole::ParameterObservationSample>& samples) {
    Vector result;
    result.reserve(samples.size());
    for (const auto& sample : samples)
        result.push_back(sample.observation(run.states[sample.node])[0]);
    return result;
}

double weighted_objective(std::span<const double> residual,
                          const std::vector<asdole::ParameterObservationSample>& samples) {
    double value = 0.0;
    for (std::size_t i = 0; i < residual.size(); ++i)
        value += 0.5 * samples[i].quadrature_weight * residual[i] * residual[i];
    return value;
}

asdole::MatrixFreeParameterObservationSensitivityOperator make_sensitivity(
    const PdeRun& run, const std::vector<asdole::ParameterObservationSample>& samples) {
    return asdole::MatrixFreeParameterObservationSensitivityOperator(
        *run.trajectory, parameter_dimension,
        [](std::span<const double>) { return Vector(grid_size, 0.0); },
        [](std::span<const double>) { return Vector(parameter_dimension, 0.0); },
        make_parameter_derivatives(run.diffusivity, run.rate), samples);
}

asdole::LeastSquaresLinearization linearize(
    std::span<const double> parameters,
    std::span<const double> data,
    const std::vector<asdole::ParameterObservationSample>& samples) {
    if (parameters.size() != parameter_dimension) throw std::invalid_argument("log-diffusivity/log-growth parameter vector must have dimension two");
    PdeRun run(std::exp(parameters[0]), std::exp(parameters[1]));
    const auto sensitivity = make_sensitivity(run, samples);
    const Vector prediction = observe(run, samples);
    Vector residual(prediction.size());
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = prediction[i] - data[i];

    asdole::LeastSquaresLinearization result;
    result.objective = weighted_objective(residual, samples);
    result.gradient = sensitivity.weighted_transpose_apply(residual);
    result.gauss_newton_matrix = DenseMatrix(parameter_dimension);
    for (std::size_t column = 0; column < parameter_dimension; ++column) {
        Vector basis(parameter_dimension, 0.0); basis[column] = 1.0;
        const Vector image = sensitivity.weighted_gram_action(basis);
        for (std::size_t row = 0; row < parameter_dimension; ++row)
            result.gauss_newton_matrix(row, column) = image[row];
    }
    return result;
}

double evaluate_objective(std::span<const double> parameters,
                          std::span<const double> data,
                          const std::vector<asdole::ParameterObservationSample>& samples) {
    PdeRun run(std::exp(parameters[0]), std::exp(parameters[1]));
    const Vector prediction = observe(run, samples);
    Vector residual(prediction.size());
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = prediction[i] - data[i];
    return weighted_objective(residual, samples);
}

double dot(std::span<const double> a, std::span<const double> b) {
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}

struct IdentifiabilityResult {
    double sigma_max{};
    double sigma_min{};
    double condition{};
    std::size_t rank{};
};

IdentifiabilityResult identifiability(const DenseMatrix& gram,
                                      double relative_threshold = 1.0e-6) {
    const double a = gram(0, 0), b = 0.5 * (gram(0, 1) + gram(1, 0)), c = gram(1, 1);
    const double discriminant = std::sqrt(std::max(0.0, (a - c) * (a - c) + 4.0 * b * b));
    const double lambda_max = std::max(0.0, 0.5 * (a + c + discriminant));
    const double lambda_min = std::max(0.0, 0.5 * (a + c - discriminant));
    IdentifiabilityResult result;
    result.sigma_max = std::sqrt(lambda_max);
    result.sigma_min = std::sqrt(lambda_min);
    result.rank = (result.sigma_max > 0.0 ? 1u : 0u)
        + (result.sigma_min > relative_threshold * result.sigma_max ? 1u : 0u);
    result.condition = result.rank == 2 && result.sigma_min > 0.0
        ? result.sigma_max / result.sigma_min
        : std::numeric_limits<double>::infinity();
    return result;
}

int main_impl(int argc, char** argv) {
    const bool check = argc > 1 && std::string(argv[1]) == "--check";
    constexpr double true_diffusivity = 0.005;
    constexpr double true_rate = 2.0;
    constexpr double initial_diffusivity = 0.009;
    constexpr double initial_rate = 1.3;
    const auto samples = observation_samples();
    PdeRun truth(true_diffusivity, true_rate);
    Vector data = observe(truth, samples);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] += 2.0e-4 * std::sin(1.7 * static_cast<double>(i + 1));

    const Vector initial_parameter{std::log(initial_diffusivity), std::log(initial_rate)};
    const auto initial_linearization = linearize(initial_parameter, data, samples);
    constexpr double epsilon = 1.0e-5;
    double gradient_fd_error = 0.0;
    for (std::size_t parameter = 0; parameter < parameter_dimension; ++parameter) {
        Vector plus = initial_parameter, minus = initial_parameter;
        plus[parameter] += epsilon;
        minus[parameter] -= epsilon;
        const double fd_gradient = (evaluate_objective(plus, data, samples)
                                  - evaluate_objective(minus, data, samples)) / (2.0 * epsilon);
        const double error = std::abs(initial_linearization.gradient[parameter] - fd_gradient)
            / std::max({1.0, std::abs(initial_linearization.gradient[parameter]), std::abs(fd_gradient)});
        gradient_fd_error = std::max(gradient_fd_error, error);
    }

    PdeRun check_run(initial_diffusivity, initial_rate);
    const auto sensitivity = make_sensitivity(check_run, samples);
    Vector observation_cotangent(samples.size());
    for (std::size_t i = 0; i < observation_cotangent.size(); ++i)
        observation_cotangent[i] = std::cos(0.37 * static_cast<double>(i + 1));
    const Vector parameter_direction{0.35, -0.6};
    const Vector observation_direction = sensitivity.apply(parameter_direction);
    const Vector adjoint_parameter = sensitivity.transpose_apply(observation_cotangent);
    const double sensitivity_adjoint_defect = std::abs(
        dot(observation_direction, observation_cotangent) - dot(parameter_direction, adjoint_parameter))
        / std::max({1.0, std::abs(dot(observation_direction, observation_cotangent)),
                    std::abs(dot(parameter_direction, adjoint_parameter))});

    const auto full_identifiability = identifiability(initial_linearization.gauss_newton_matrix);
    std::vector<asdole::ParameterObservationSample> single_sample{samples[samples.size() - 2]};
    single_sample[0].quadrature_weight = 1.0;
    const auto sparse_sensitivity = make_sensitivity(check_run, single_sample);
    DenseMatrix sparse_gram(parameter_dimension);
    for (std::size_t column = 0; column < parameter_dimension; ++column) {
        Vector basis(parameter_dimension, 0.0); basis[column] = 1.0;
        const Vector image = sparse_sensitivity.weighted_gram_action(basis);
        for (std::size_t row = 0; row < parameter_dimension; ++row)
            sparse_gram(row, column) = image[row];
    }
    const auto sparse_identifiability = identifiability(sparse_gram);

    asdole::GaussNewtonOptions options;
    options.maximum_iterations = 20;
    options.gradient_tolerance = 1.0e-10;
    options.initial_damping = 1.0e-4;
    const auto result = asdole::damped_gauss_newton(
        [&](std::span<const double> p) { return linearize(p, data, samples); },
        [&](std::span<const double> p) { return evaluate_objective(p, data, samples); },
        initial_parameter, options);

    std::cout << std::setprecision(12)
              << "inverse_problem=Fisher-KPP_joint_log_diffusivity_and_growth_rate_fit\n"
              << "grid_points=" << grid_size << " observations=" << samples.size()
              << " theta=" << theta << " steps=" << time_steps << " final_time=" << final_time << '\n'
              << "true_diffusivity=" << true_diffusivity << " initial_diffusivity=" << initial_diffusivity
              << " recovered_diffusivity=" << std::exp(result.parameters[0]) << '\n'
              << "true_growth_rate=" << true_rate << " initial_growth_rate=" << initial_rate
              << " recovered_growth_rate=" << std::exp(result.parameters[1]) << '\n'
              << "objective_initial=" << initial_linearization.objective
              << " objective_final=" << result.objective
              << " gradient_norm_final=" << result.gradient_norm
              << " iterations=" << result.iterations
              << " converged=" << (result.converged ? "yes" : "no") << '\n'
              << "initial_gradient_fd_relative_error=" << gradient_fd_error
              << " sensitivity_adjoint_dot_defect=" << sensitivity_adjoint_defect << '\n'
              << "all_observations_identifiability_rank=" << full_identifiability.rank
              << " singular_values=[" << full_identifiability.sigma_max << ','
              << full_identifiability.sigma_min << "] condition=" << full_identifiability.condition << '\n'
              << "single_final_sensor_identifiability_rank=" << sparse_identifiability.rank
              << " singular_values=[" << sparse_identifiability.sigma_max << ','
              << sparse_identifiability.sigma_min << "] condition=" << sparse_identifiability.condition << '\n';

    if (check && (!result.converged
                  || std::abs(std::exp(result.parameters[0]) / true_diffusivity - 1.0) > 0.03
                  || std::abs(std::exp(result.parameters[1]) / true_rate - 1.0) > 0.02
                  || !(result.objective < initial_linearization.objective)
                  || !(gradient_fd_error < 2.0e-5)
                  || !(sensitivity_adjoint_defect < 1.0e-8)
                  || full_identifiability.rank != 2 || sparse_identifiability.rank != 1))
        throw std::runtime_error("matrix-free reaction-diffusion inverse check failed");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return main_impl(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "matrix_free_reaction_diffusion_inverse: " << error.what() << '\n';
        return 2;
    }
}
