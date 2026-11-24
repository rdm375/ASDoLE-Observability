// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace asdole {

namespace contraction_detail {

inline DenseMatrix transpose(const DenseMatrix& a) {
    DenseMatrix result(a.cols(), a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) result(j, i) = a(i, j);
    return result;
}

inline DenseMatrix multiply(const DenseMatrix& a, const DenseMatrix& b) {
    if (a.cols() != b.rows()) throw std::invalid_argument("matrix product dimensions differ");
    DenseMatrix result(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t k = 0; k < a.cols(); ++k)
            for (std::size_t j = 0; j < b.cols(); ++j)
                result(i, j) += a(i, k) * b(k, j);
    return result;
}

inline DenseMatrix cholesky_lower(const DenseMatrix& w) {
    if (w.rows() == 0 || w.rows() != w.cols())
        throw std::invalid_argument("weighting matrix must be nonempty and square");
    const auto n = w.rows();
    DenseMatrix l(n);
    double scale = 0.0, asymmetry = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(w(i,j))) throw std::invalid_argument("weighting matrix must be finite");
            scale = std::max(scale, std::abs(w(i,j)));
            asymmetry = std::max(asymmetry, std::abs(w(i,j) - w(j,i)));
        }
    if (asymmetry > 1.0e-12 * std::max(scale, std::numeric_limits<double>::min()))
        throw std::invalid_argument("weighting matrix must be symmetric");
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = w(i,j);
            for (std::size_t k = 0; k < j; ++k) value -= l(i,k) * l(j,k);
            if (i == j) {
                if (!(value > 0.0)) throw std::invalid_argument("weighting matrix must be positive definite");
                l(i,j) = std::sqrt(value);
            } else {
                l(i,j) = value / l(j,j);
            }
        }
    }
    return l;
}

inline DenseMatrix inverse_lower(const DenseMatrix& l) {
    const auto n = l.rows();
    DenseMatrix inv(n);
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            double value = row == col ? 1.0 : 0.0;
            for (std::size_t k = 0; k < row; ++k) value -= l(row,k) * inv(k,col);
            inv(row,col) = value / l(row,row);
        }
    }
    return inv;
}

inline DenseMatrix validate_square_map(const DenseMatrix& map, std::size_t n) {
    if (map.rows() != n || map.cols() != n)
        throw std::invalid_argument("state transition and weighting dimensions differ");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!std::isfinite(map(i,j))) throw std::invalid_argument("state transition must be finite");
    return map;
}

// Express a state-space map in Euclidean coordinates for the W norm,
// where W = L L^T and z = L^T x. The transformed map is L^T M L^{-T}.
inline DenseMatrix metric_coordinate_map(const DenseMatrix& map, const DenseMatrix& lower) {
    const auto n = lower.rows();
    (void)validate_square_map(map, n);
    const auto lower_t = transpose(lower);
    const auto inverse_lower_t = transpose(inverse_lower(lower));
    return multiply(multiply(lower_t, map), inverse_lower_t);
}

inline DenseMatrix step_matrix(const DiscreteTrajectoryLinearization& trajectory, std::size_t k) {
    const auto n = trajectory.state_dimension();
    DenseMatrix m(n);
    Vector basis(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        std::fill(basis.begin(), basis.end(), 0.0);
        basis[j] = 1.0;
        const Vector image = trajectory.apply_step(basis, k);
        for (std::size_t i = 0; i < n; ++i) m(i,j) = image[i];
    }
    return m;
}

} // namespace contraction_detail

struct ContinuousContractionCertificate {
    double requested_rate{};
    double largest_residual_eigenvalue{};
    double smallest_weight_eigenvalue{};
    double largest_weight_eigenvalue{};
    bool certified{};
};

struct LyapunovMetricResult {
    DenseMatrix weight;
    double requested_rate{};
    double estimated_rate_limit{};
    ContinuousContractionCertificate certificate;
};

inline ContinuousContractionCertificate certify_continuous_contraction(
    const DenseMatrix& a, const DenseMatrix& w, double rate, double tolerance);

namespace contraction_detail {

inline Vector solve_dense(Vector matrix, Vector rhs, std::size_t n) {
    if (matrix.size() != n*n || rhs.size() != n)
        throw std::invalid_argument("dense linear solve dimensions differ");
    double scale = 0.0;
    for (double value : matrix) {
        if (!std::isfinite(value)) throw std::invalid_argument("linear system must be finite");
        scale = std::max(scale, std::abs(value));
    }
    for (double value : rhs)
        if (!std::isfinite(value)) throw std::invalid_argument("linear system right-hand side must be finite");
    if (!(scale > 0.0)) throw std::runtime_error("Lyapunov operator is singular");
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(matrix[i*n+k]) > std::abs(matrix[pivot*n+k])) pivot = i;
        if (std::abs(matrix[pivot*n+k]) <= 64.0 * std::numeric_limits<double>::epsilon() * scale)
            throw std::runtime_error("Lyapunov operator is singular or ill-conditioned");
        if (pivot != k) {
            for (std::size_t j = k; j < n; ++j) std::swap(matrix[k*n+j], matrix[pivot*n+j]);
            std::swap(rhs[k], rhs[pivot]);
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = matrix[i*n+k] / matrix[k*n+k];
            matrix[i*n+k] = 0.0;
            for (std::size_t j = k + 1; j < n; ++j) matrix[i*n+j] -= factor * matrix[k*n+j];
            rhs[i] -= factor * rhs[k];
        }
    }
    Vector solution(n, 0.0);
    for (std::size_t reverse = n; reverse > 0; --reverse) {
        const std::size_t i = reverse - 1;
        double value = rhs[i];
        for (std::size_t j = i + 1; j < n; ++j) value -= matrix[i*n+j] * solution[j];
        solution[i] = value / matrix[i*n+i];
    }
    return solution;
}

inline DenseMatrix solve_lyapunov(const DenseMatrix& a, const DenseMatrix& q, double rate) {
    const auto n = a.rows();
    if (n == 0 || a.cols() != n || q.rows() != n || q.cols() != n)
        throw std::invalid_argument("A and Q must be nonempty square matrices of equal size");
    if (!(rate >= 0.0) || !std::isfinite(rate))
        throw std::invalid_argument("Lyapunov rate must be finite and nonnegative");
    (void)cholesky_lower(q);
    DenseMatrix shifted = a;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(a(i,j))) throw std::invalid_argument("A must be finite");
            if (i == j) shifted(i,j) += rate;
        }
    // Row-major vectorization of C^T W + W C = -Q.
    const std::size_t dimension = n*n;
    Vector operator_matrix(dimension*dimension, 0.0), rhs(dimension, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const auto row = i*n+j;
            rhs[row] = -q(i,j);
            for (std::size_t k = 0; k < n; ++k) {
                operator_matrix[row*dimension + k*n+j] += shifted(k,i);
                operator_matrix[row*dimension + i*n+k] += shifted(k,j);
            }
        }
    const Vector solution = solve_dense(std::move(operator_matrix), std::move(rhs), dimension);
    DenseMatrix w(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            w(i,j) = 0.5 * (solution[i*n+j] + solution[j*n+i]);
    (void)cholesky_lower(w);
    return w;
}

} // namespace contraction_detail

// Construct W by solving (A + rate I)^T W + W(A + rate I) = -Q.
// A positive-definite W certifies the requested rate. This routine is dense
// and intended for small/medium analysis problems.
inline LyapunovMetricResult lyapunov_metric_for_rate(
    const DenseMatrix& a, double rate, const DenseMatrix& q) {
    DenseMatrix w = contraction_detail::solve_lyapunov(a, q, rate);
    const auto certificate = certify_continuous_contraction(a, w, rate, 1.0e-10);
    if (!certificate.certified)
        throw std::runtime_error("computed Lyapunov metric failed its residual check");
    return {std::move(w), rate, rate, certificate};
}

// Estimate the largest certifiable rate by bisection, then back away from the
// boundary by safety_fraction to avoid the ill-conditioned limiting metric.
inline LyapunovMetricResult estimate_lyapunov_metric(
    const DenseMatrix& a, double safety_fraction = 0.8,
    std::size_t bisection_iterations = 60) {
    if (!(safety_fraction > 0.0 && safety_fraction < 1.0)
        || !std::isfinite(safety_fraction) || bisection_iterations == 0)
        throw std::invalid_argument("safety fraction must be in (0,1) and iterations positive");
    if (a.rows() == 0 || a.rows() != a.cols())
        throw std::invalid_argument("A must be a nonempty square matrix");
    DenseMatrix q(a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i) q(i,i) = 1.0;
    DenseMatrix base = contraction_detail::solve_lyapunov(a, q, 0.0);
    (void)base;
    double upper = 1.0;
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) {
            if (!std::isfinite(a(i,j))) throw std::invalid_argument("A must be finite");
            upper += a(i,j)*a(i,j);
        }
    upper = std::sqrt(upper) + 1.0;
    double lower = 0.0;
    for (std::size_t iteration = 0; iteration < bisection_iterations; ++iteration) {
        const double candidate = 0.5 * (lower + upper);
        bool feasible = false;
        try {
            (void)contraction_detail::solve_lyapunov(a, q, candidate);
            feasible = true;
        } catch (const std::exception&) {
            feasible = false;
        }
        if (feasible) lower = candidate;
        else upper = candidate;
    }
    if (!(lower > 0.0)) throw std::runtime_error("A is not certified exponentially stable");
    const double rate = safety_fraction * lower;
    auto result = lyapunov_metric_for_rate(a, rate, q);
    result.estimated_rate_limit = lower;
    return result;
}

// Check A^T W + W A <= -2 lambda W for a supplied symmetric positive
// definite W. This is a certificate for the linear system, or for a nonlinear
// Jacobian only when the inequality is established throughout the region of
// interest (a single trajectory sample is not a global certificate).
inline ContinuousContractionCertificate certify_continuous_contraction(
    const DenseMatrix& a, const DenseMatrix& w, double rate, double tolerance = 1.0e-10) {
    if (a.rows() == 0 || a.rows() != a.cols() || w.rows() != a.rows() || w.cols() != a.cols())
        throw std::invalid_argument("A and W must be nonempty square matrices of the same size");
    if (!(rate > 0.0) || !std::isfinite(rate) || !(tolerance >= 0.0) || !std::isfinite(tolerance))
        throw std::invalid_argument("rate must be positive and tolerance nonnegative");
    const auto lower = contraction_detail::cholesky_lower(w);
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j)
            if (!std::isfinite(a(i,j))) throw std::invalid_argument("system matrix must be finite");
    // Congruence by L^{-1} preserves negative semidefiniteness and gives
    // C = B^T + B + 2 rate I, with B = L^T A L^{-T}. This removes the
    // arbitrary overall scale of W from both the residual and tolerance.
    const DenseMatrix b = contraction_detail::metric_coordinate_map(a, lower);
    DenseMatrix residual(a.rows());
    double scale = 2.0 * rate;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            residual(i,j) = b(j,i) + b(i,j) + (i == j ? 2.0 * rate : 0.0);
            scale = std::max(scale, std::abs(b(i,j)) + std::abs(b(j,i))
                                      + (i == j ? 2.0 * rate : 0.0));
        }
    }
    const auto eig = symmetric_eigenvalues(residual);
    const auto we = symmetric_eigenvalues(w);
    ContinuousContractionCertificate result;
    result.requested_rate = rate;
    // This is the largest eigenvalue of the W-normalized Lyapunov residual.
    result.largest_residual_eigenvalue = eig.front();
    result.smallest_weight_eigenvalue = we.back();
    result.largest_weight_eigenvalue = we.front();
    result.certified = eig.front() <= tolerance
        * std::max(scale, std::numeric_limits<double>::min());
    return result;
}

// Exact induced operator norm in the W norm for a dense finite-dimensional
// linear map M: sqrt(lambda_max(W^{-1/2} M^T W M W^{-1/2})).
inline double weighted_induced_norm(const DenseMatrix& map, const DenseMatrix& w) {
    const auto l = contraction_detail::cholesky_lower(w);
    const auto n = w.rows();
    (void)contraction_detail::validate_square_map(map, n);
    const auto whitened = contraction_detail::metric_coordinate_map(map, l);
    const auto gram = contraction_detail::multiply(
        contraction_detail::transpose(whitened), whitened);
    const auto eig = symmetric_eigenvalues(gram);
    return std::sqrt(std::max(0.0, eig.front()));
}

struct DiscreteContractionStep {
    double step_size{};
    double induced_factor{};
    double certified_rate{};
    bool contracts{};
};

inline std::vector<DiscreteContractionStep> discrete_contraction_history(
    const DiscreteTrajectoryLinearization& trajectory, const DenseMatrix& w,
    double tolerance = 1.0e-12) {
    if (w.rows() != trajectory.state_dimension() || w.cols() != trajectory.state_dimension())
        throw std::invalid_argument("weighting matrix dimension does not match trajectory");
    (void)contraction_detail::cholesky_lower(w);
    if (!(tolerance >= 0.0) || !std::isfinite(tolerance))
        throw std::invalid_argument("tolerance must be finite and nonnegative");
    std::vector<DiscreteContractionStep> result;
    result.reserve(trajectory.step_count());
    for (std::size_t k = 0; k < trajectory.step_count(); ++k) {
        const double h = trajectory.step_size(k);
        const double factor = weighted_induced_norm(contraction_detail::step_matrix(trajectory, k), w);
        result.push_back({h, factor, factor > 0.0 ? -std::log(factor)/h
                                                    : std::numeric_limits<double>::infinity(),
                          factor <= 1.0 + tolerance});
    }
    return result;
}

} // namespace asdole
