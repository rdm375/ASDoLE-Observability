// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace asdole {

namespace covariance_detail {

inline DenseMatrix transpose(const DenseMatrix& matrix) {
    DenseMatrix result(matrix.cols(), matrix.rows());
    for (std::size_t i = 0; i < matrix.rows(); ++i)
        for (std::size_t j = 0; j < matrix.cols(); ++j)
            result(j, i) = matrix(i, j);
    return result;
}

inline DenseMatrix multiply(const DenseMatrix& left, const DenseMatrix& right) {
    if (left.cols() != right.rows())
        throw std::invalid_argument("matrix product dimensions differ");
    DenseMatrix result(left.rows(), right.cols());
    for (std::size_t i = 0; i < left.rows(); ++i)
        for (std::size_t k = 0; k < left.cols(); ++k)
            for (std::size_t j = 0; j < right.cols(); ++j)
                result(i, j) += left(i, k) * right(k, j);
    return result;
}

inline DenseMatrix symmetrized(const DenseMatrix& matrix) {
    if (matrix.rows() != matrix.cols())
        throw std::invalid_argument("covariance matrix must be square");
    DenseMatrix result(matrix.rows());
    for (std::size_t i = 0; i < matrix.rows(); ++i)
        for (std::size_t j = 0; j < matrix.cols(); ++j)
            result(i, j) = 0.5 * (matrix(i, j) + matrix(j, i));
    return result;
}

inline void validate_covariance(const DenseMatrix& covariance,
                                std::size_t dimension,
                                const char* label,
                                double tolerance = 1.0e-10) {
    if (covariance.rows() != dimension || covariance.cols() != dimension)
        throw std::invalid_argument(std::string(label) + " has the wrong dimensions");
    double scale = 0.0;
    double asymmetry = 0.0;
    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t j = 0; j < dimension; ++j) {
            if (!std::isfinite(covariance(i, j)))
                throw std::invalid_argument(std::string(label) + " contains a non-finite value");
            scale = std::max(scale, std::abs(covariance(i, j)));
            asymmetry = std::max(asymmetry, std::abs(covariance(i, j) - covariance(j, i)));
        }
    }
    const double comparison_scale = std::max(scale, std::numeric_limits<double>::min());
    if (asymmetry > tolerance * comparison_scale)
        throw std::invalid_argument(std::string(label) + " must be symmetric");

    const auto eigenvalues = symmetric_eigenvalues(symmetrized(covariance));
    const double psd_tolerance = tolerance * comparison_scale;
    for (double eigenvalue : eigenvalues)
        if (eigenvalue < -psd_tolerance)
            throw std::invalid_argument(std::string(label) + " must be positive semidefinite");
}

inline DenseMatrix step_matrix(const DiscreteTrajectoryLinearization& trajectory,
                               std::size_t step) {
    const auto n = trajectory.state_dimension();
    DenseMatrix result(n);
    Vector basis(n, 0.0);
    for (std::size_t column = 0; column < n; ++column) {
        std::fill(basis.begin(), basis.end(), 0.0);
        basis[column] = 1.0;
        const Vector image = trajectory.apply_step(basis, step);
        for (std::size_t row = 0; row < n; ++row) result(row, column) = image[row];
    }
    return result;
}

inline DenseMatrix observation_matrix(const LinearObservationOperator& observation) {
    DenseMatrix result(observation.observation_dimension, observation.state_dimension);
    Vector basis(observation.state_dimension, 0.0);
    for (std::size_t column = 0; column < observation.state_dimension; ++column) {
        std::fill(basis.begin(), basis.end(), 0.0);
        basis[column] = 1.0;
        const Vector image = observation(basis);
        for (std::size_t row = 0; row < observation.observation_dimension; ++row)
            result(row, column) = image[row];
    }
    return result;
}

} // namespace covariance_detail

struct CovarianceDiagnostics {
    Vector standard_deviations;
    DenseMatrix correlation;
    double symmetry_defect{};
    double smallest_eigenvalue{};
};

inline CovarianceDiagnostics diagnose_covariance(const DenseMatrix& covariance) {
    covariance_detail::validate_covariance(covariance, covariance.rows(), "covariance");
    CovarianceDiagnostics result;
    const auto n = covariance.rows();
    result.standard_deviations.resize(n);
    result.correlation = DenseMatrix(n);
    const auto symmetric = covariance_detail::symmetrized(covariance);
    for (std::size_t i = 0; i < n; ++i) {
        result.standard_deviations[i] = std::sqrt(std::max(0.0, symmetric(i, i)));
        for (std::size_t j = 0; j < n; ++j) {
            result.symmetry_defect = std::max(result.symmetry_defect,
                std::abs(covariance(i, j) - covariance(j, i)));
            const double denominator = result.standard_deviations[i] * result.standard_deviations[j];
            result.correlation(i, j) = denominator > 0.0
                ? symmetric(i, j) / denominator
                : std::numeric_limits<double>::quiet_NaN();
        }
    }
    const auto eigenvalues = symmetric_eigenvalues(symmetric);
    result.smallest_eigenvalue = eigenvalues.empty() ? 0.0 : eigenvalues.back();
    return result;
}

// Return covariance matrices at every trajectory node. process_covariances[k]
// is an independent additive state perturbation introduced at the end of step
// k. An empty list means deterministic propagation (Q_k = 0).
inline std::vector<DenseMatrix> propagate_covariance(
    const DiscreteTrajectoryLinearization& trajectory,
    const DenseMatrix& initial_covariance,
    std::span<const DenseMatrix> process_covariances = {}) {
    const auto n = trajectory.state_dimension();
    covariance_detail::validate_covariance(initial_covariance, n, "initial covariance");
    if (!process_covariances.empty() && process_covariances.size() != trajectory.step_count())
        throw std::invalid_argument("process covariance history must have one matrix per step");

    std::vector<DenseMatrix> result;
    result.reserve(trajectory.node_count());
    result.push_back(covariance_detail::symmetrized(initial_covariance));
    for (std::size_t k = 0; k < trajectory.step_count(); ++k) {
        const DenseMatrix transition = covariance_detail::step_matrix(trajectory, k);
        DenseMatrix next = covariance_detail::multiply(
            covariance_detail::multiply(transition, result.back()),
            covariance_detail::transpose(transition));
        if (!process_covariances.empty()) {
            covariance_detail::validate_covariance(process_covariances[k], n, "process covariance");
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    next(i, j) += process_covariances[k](i, j);
        }
        next = covariance_detail::symmetrized(next);
        covariance_detail::validate_covariance(next, n, "propagated covariance", 1.0e-8);
        result.push_back(std::move(next));
    }
    return result;
}

// Propagate a joint covariance for [state; parameter] when the parameter is
// constant in time. parameter_step_sensitivities[k] is B_k in
//   dx_{k+1} = M_k dx_k + B_k dp + w_k,  dp_{k+1} = dp_k.
// Each B_k has shape state_dimension x parameter_dimension. This also models
// any fixed finite-dimensional control coefficients; per-step independent
// input uncertainty can be supplied through process_covariances.
inline std::vector<DenseMatrix> propagate_joint_state_parameter_covariance(
    const DiscreteTrajectoryLinearization& trajectory,
    const DenseMatrix& initial_joint_covariance,
    std::span<const DenseMatrix> parameter_step_sensitivities,
    std::span<const DenseMatrix> process_covariances = {}) {
    const auto n = trajectory.state_dimension();
    if (parameter_step_sensitivities.size() != trajectory.step_count())
        throw std::invalid_argument("parameter sensitivity history must have one matrix per step");
    if (!process_covariances.empty() && process_covariances.size() != trajectory.step_count())
        throw std::invalid_argument("process covariance history must have one matrix per step");
    if (initial_joint_covariance.rows() != initial_joint_covariance.cols()
        || initial_joint_covariance.rows() <= n)
        throw std::invalid_argument("joint covariance must include state and at least one parameter");
    const auto joint_dimension = initial_joint_covariance.rows();
    const auto p = joint_dimension - n;
    covariance_detail::validate_covariance(initial_joint_covariance, joint_dimension,
                                           "initial joint covariance");

    std::vector<DenseMatrix> result;
    result.reserve(trajectory.node_count());
    result.push_back(covariance_detail::symmetrized(initial_joint_covariance));
    for (std::size_t k = 0; k < trajectory.step_count(); ++k) {
        const auto& b = parameter_step_sensitivities[k];
        if (b.rows() != n || b.cols() != p)
            throw std::invalid_argument("parameter step sensitivity has the wrong dimensions");
        DenseMatrix augmented(joint_dimension);
        const DenseMatrix transition = covariance_detail::step_matrix(trajectory, k);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) augmented(i, j) = transition(i, j);
            for (std::size_t j = 0; j < p; ++j) augmented(i, n + j) = b(i, j);
        }
        for (std::size_t i = 0; i < p; ++i) augmented(n + i, n + i) = 1.0;

        DenseMatrix next = covariance_detail::multiply(
            covariance_detail::multiply(augmented, result.back()),
            covariance_detail::transpose(augmented));
        if (!process_covariances.empty()) {
            covariance_detail::validate_covariance(process_covariances[k], n, "process covariance");
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    next(i, j) += process_covariances[k](i, j);
        }
        next = covariance_detail::symmetrized(next);
        covariance_detail::validate_covariance(next, joint_dimension, "propagated joint covariance", 1.0e-8);
        result.push_back(std::move(next));
    }
    return result;
}

// Cov(x_i, x_j), with rows indexed by x_i and columns by x_j. The returned
// cross covariance uses the same discrete TLM maps as the marginal history.
inline DenseMatrix state_cross_covariance(
    const DiscreteTrajectoryLinearization& trajectory,
    std::span<const DenseMatrix> marginal_covariances,
    std::size_t node_i,
    std::size_t node_j) {
    if (marginal_covariances.size() != trajectory.node_count())
        throw std::invalid_argument("marginal covariance history has wrong length");
    if (node_i >= trajectory.node_count() || node_j >= trajectory.node_count())
        throw std::out_of_range("cross-covariance node is out of range");
    const auto n = trajectory.state_dimension();
    if (node_i == node_j) {
        covariance_detail::validate_covariance(
            marginal_covariances[node_i], trajectory.state_dimension(), "marginal covariance");
        return marginal_covariances[node_i];
    }
    const std::size_t first = std::min(node_i, node_j);
    const std::size_t last = std::max(node_i, node_j);
    covariance_detail::validate_covariance(marginal_covariances[first], n, "marginal covariance");

    DenseMatrix transition(n);
    for (std::size_t column = 0; column < n; ++column) {
        Vector value(n, 0.0);
        value[column] = 1.0;
        for (std::size_t k = first; k < last; ++k) value = trajectory.apply_step(value, k);
        for (std::size_t row = 0; row < n; ++row) transition(row, column) = value[row];
    }
    // For first <= last, Cov(x_first, x_last) = P_first Phi(last,first)^T.
    const DenseMatrix forward = covariance_detail::multiply(
        marginal_covariances[first], covariance_detail::transpose(transition));
    return node_i <= node_j ? forward : covariance_detail::transpose(forward);
}

inline DenseMatrix observation_cross_covariance(
    const LinearObservationOperator& observation_i,
    const DenseMatrix& state_cross,
    const LinearObservationOperator& observation_j,
    const DenseMatrix* measurement_cross_covariance = nullptr) {
    if (observation_i.state_dimension != state_cross.rows()
        || observation_j.state_dimension != state_cross.cols())
        throw std::invalid_argument("state and observation cross-covariance dimensions differ");
    const DenseMatrix hi = covariance_detail::observation_matrix(observation_i);
    const DenseMatrix hj = covariance_detail::observation_matrix(observation_j);
    DenseMatrix result = covariance_detail::multiply(
        covariance_detail::multiply(hi, state_cross), covariance_detail::transpose(hj));
    if (measurement_cross_covariance != nullptr) {
        if (measurement_cross_covariance->rows() != result.rows()
            || measurement_cross_covariance->cols() != result.cols())
            throw std::invalid_argument("measurement cross-covariance dimensions differ");
        for (std::size_t i = 0; i < result.rows(); ++i)
            for (std::size_t j = 0; j < result.cols(); ++j)
                result(i, j) += (*measurement_cross_covariance)(i, j);
    }
    return result;
}

// One linear observation block in a stacked time history. Samples may refer
// to the same trajectory node and may have different rectangular operators.
struct ObservationCovarianceSample {
    std::size_t node{};
    LinearObservationOperator observation;
};

// Construct Cov([H_0 x_{i_0}; ...; H_s x_{i_s}] + measurement_error).
// measurement_covariance is the full covariance of the stacked measurement
// errors, so it can represent white, temporally correlated, or cross-sensor
// noise. An empty matrix (0 x 0) means no measurement noise.
inline DenseMatrix stacked_observation_covariance(
    const DiscreteTrajectoryLinearization& trajectory,
    std::span<const DenseMatrix> marginal_covariances,
    std::span<const ObservationCovarianceSample> samples,
    const DenseMatrix& measurement_covariance = DenseMatrix{}) {
    if (marginal_covariances.size() != trajectory.node_count())
        throw std::invalid_argument("marginal covariance history has wrong length");
    if (samples.empty()) throw std::invalid_argument("at least one observation sample is required");

    std::vector<std::size_t> offsets(samples.size() + 1, 0);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        if (sample.node >= trajectory.node_count()
            || sample.observation.state_dimension != trajectory.state_dimension()
            || sample.observation.observation_dimension == 0
            || !sample.observation.apply || !sample.observation.transpose_apply)
            throw std::invalid_argument("invalid observation covariance sample");
        offsets[i + 1] = offsets[i] + sample.observation.observation_dimension;
    }
    const std::size_t total_dimension = offsets.back();
    const bool has_measurement_covariance = measurement_covariance.rows() != 0
        || measurement_covariance.cols() != 0;
    if (has_measurement_covariance)
        covariance_detail::validate_covariance(measurement_covariance, total_dimension,
                                               "stacked measurement covariance");

    DenseMatrix result(total_dimension);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        for (std::size_t j = i; j < samples.size(); ++j) {
            const DenseMatrix state_cross = state_cross_covariance(
                trajectory, marginal_covariances, samples[i].node, samples[j].node);
            const DenseMatrix block = observation_cross_covariance(
                samples[i].observation, state_cross, samples[j].observation);
            for (std::size_t row = 0; row < block.rows(); ++row) {
                for (std::size_t col = 0; col < block.cols(); ++col) {
                    result(offsets[i] + row, offsets[j] + col) = block(row, col);
                    result(offsets[j] + col, offsets[i] + row) = block(row, col);
                }
            }
        }
    }
    if (has_measurement_covariance)
        for (std::size_t i = 0; i < total_dimension; ++i)
            for (std::size_t j = 0; j < total_dimension; ++j)
                result(i, j) += measurement_covariance(i, j);
    return covariance_detail::symmetrized(result);
}

namespace covariance_detail {

inline DenseMatrix cholesky_factor(const DenseMatrix& matrix) {
    if (matrix.rows() != matrix.cols() || matrix.rows() == 0)
        throw std::invalid_argument("Cholesky factorization requires a nonempty square matrix");
    const auto n = matrix.rows();
    DenseMatrix lower(n);
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        scale = std::max(scale, std::abs(matrix(i, i)));
    const double tolerance = std::numeric_limits<double>::epsilon()
        * std::max(scale, std::numeric_limits<double>::min()) * static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = matrix(i, j);
            for (std::size_t k = 0; k < j; ++k) value -= lower(i, k) * lower(j, k);
            if (i == j) {
                if (!(value > tolerance) || !std::isfinite(value))
                    throw std::invalid_argument("GLS covariance or normal matrix must be positive definite");
                lower(i, j) = std::sqrt(value);
            } else {
                lower(i, j) = value / lower(j, j);
            }
        }
    }
    return lower;
}

inline Vector cholesky_solve(const DenseMatrix& lower, std::span<const double> rhs) {
    const auto n = lower.rows();
    if (lower.cols() != n || rhs.size() != n)
        throw std::invalid_argument("Cholesky solve dimension mismatch");
    Vector result(rhs.begin(), rhs.end());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) result[i] -= lower(i, j) * result[j];
        result[i] /= lower(i, i);
    }
    for (std::size_t reverse = n; reverse > 0; --reverse) {
        const auto i = reverse - 1;
        for (std::size_t j = i + 1; j < n; ++j) result[i] -= lower(j, i) * result[j];
        result[i] /= lower(i, i);
    }
    return result;
}

} // namespace covariance_detail

struct GeneralizedLeastSquaresResult {
    Vector parameters;
    DenseMatrix parameter_covariance;
    Vector residual;
    Vector weighted_residual; // C^{-1} residual
    double objective{};        // 1/2 residual^T C^{-1} residual
};

// Solve the full-column-rank linear model data = design * parameters + error
// by generalized least squares, minimizing 1/2 r^T C^{-1} r. C must be
// symmetric positive definite. The reported parameter covariance is
// (A^T C^{-1} A)^{-1}, the usual GLS covariance when C is known.
inline GeneralizedLeastSquaresResult generalized_least_squares(
    const DenseMatrix& design,
    std::span<const double> data,
    const DenseMatrix& observation_covariance) {
    const std::size_t m = design.rows();
    const std::size_t p = design.cols();
    if (m == 0 || p == 0 || p > m || data.size() != m
        || observation_covariance.rows() != m || observation_covariance.cols() != m)
        throw std::invalid_argument("GLS design, data, and covariance dimensions are inconsistent");
    for (double value : data)
        if (!std::isfinite(value)) throw std::invalid_argument("GLS data contains a non-finite value");
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < p; ++j)
            if (!std::isfinite(design(i, j)))
                throw std::invalid_argument("GLS design contains a non-finite value");
    covariance_detail::validate_covariance(observation_covariance, m,
                                           "GLS observation covariance");

    const DenseMatrix lower = covariance_detail::cholesky_factor(observation_covariance);
    DenseMatrix weighted_design(m, p);
    for (std::size_t j = 0; j < p; ++j) {
        Vector column(m);
        for (std::size_t i = 0; i < m; ++i) column[i] = design(i, j);
        const Vector solved = covariance_detail::cholesky_solve(lower, column);
        for (std::size_t i = 0; i < m; ++i) weighted_design(i, j) = solved[i];
    }
    const Vector weighted_data = covariance_detail::cholesky_solve(lower, data);
    DenseMatrix normal(p);
    Vector rhs(p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t row = 0; row < m; ++row) {
            rhs[i] += design(row, i) * weighted_data[row];
            for (std::size_t j = 0; j < p; ++j)
                normal(i, j) += design(row, i) * weighted_design(row, j);
        }
    }
    const DenseMatrix normal_factor = covariance_detail::cholesky_factor(normal);
    GeneralizedLeastSquaresResult result;
    result.parameters = covariance_detail::cholesky_solve(normal_factor, rhs);
    result.parameter_covariance = DenseMatrix(p);
    for (std::size_t j = 0; j < p; ++j) {
        Vector basis(p, 0.0);
        basis[j] = 1.0;
        const Vector column = covariance_detail::cholesky_solve(normal_factor, basis);
        for (std::size_t i = 0; i < p; ++i) result.parameter_covariance(i, j) = column[i];
    }
    result.residual.resize(m);
    for (std::size_t i = 0; i < m; ++i) {
        result.residual[i] = data[i];
        for (std::size_t j = 0; j < p; ++j)
            result.residual[i] -= design(i, j) * result.parameters[j];
    }
    result.weighted_residual = covariance_detail::cholesky_solve(lower, result.residual);
    for (std::size_t i = 0; i < m; ++i)
        result.objective += 0.5 * result.residual[i] * result.weighted_residual[i];
    return result;
}

// Push a covariance in any finite-dimensional input space through an existing
// matrix-free sensitivity operator (for example, parameters to observations).
template<class SensitivityOperator>
inline DenseMatrix covariance_from_sensitivity(
    const SensitivityOperator& sensitivity,
    const DenseMatrix& input_covariance) {
    const auto n = sensitivity.input_dimension();
    const auto m = sensitivity.output_dimension();
    covariance_detail::validate_covariance(input_covariance, n, "sensitivity input covariance");
    DenseMatrix jacobian(m, n);
    Vector basis(n, 0.0);
    for (std::size_t column = 0; column < n; ++column) {
        std::fill(basis.begin(), basis.end(), 0.0);
        basis[column] = 1.0;
        const Vector image = sensitivity.apply(basis);
        if (image.size() != m)
            throw std::invalid_argument("sensitivity operator returned wrong dimension");
        for (std::size_t row = 0; row < m; ++row) jacobian(row, column) = image[row];
    }
    return covariance_detail::multiply(
        covariance_detail::multiply(jacobian, input_covariance),
        covariance_detail::transpose(jacobian));
}

} // namespace asdole
