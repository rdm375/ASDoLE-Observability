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
    (void)contraction_detail::cholesky_lower(w);
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j)
            if (!std::isfinite(a(i,j))) throw std::invalid_argument("system matrix must be finite");
    DenseMatrix residual(a.rows());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            for (std::size_t k = 0; k < a.rows(); ++k)
                residual(i,j) += a(k,i)*w(k,j) + w(i,k)*a(k,j);
            residual(i,j) += 2.0*rate*w(i,j);
        }
    }
    const auto eig = symmetric_eigenvalues(residual);
    const auto we = symmetric_eigenvalues(w);
    double scale = 0.0;
    for (std::size_t i = 0; i < residual.rows(); ++i)
        for (std::size_t j = 0; j < residual.cols(); ++j) scale = std::max(scale, std::abs(residual(i,j)));
    ContinuousContractionCertificate result;
    result.requested_rate = rate;
    result.largest_residual_eigenvalue = eig.front();
    result.smallest_weight_eigenvalue = we.back();
    result.largest_weight_eigenvalue = we.front();
    result.certified = eig.front() <= tolerance * std::max(1.0, scale);
    return result;
}

// Exact induced operator norm in the W norm for a dense finite-dimensional
// linear map M: sqrt(lambda_max(W^{-1/2} M^T W M W^{-1/2})).
inline double weighted_induced_norm(const DenseMatrix& map, const DenseMatrix& w) {
    const auto l = contraction_detail::cholesky_lower(w);
    const auto n = w.rows();
    (void)contraction_detail::validate_square_map(map, n);
    const auto lt = contraction_detail::transpose(l);
    const auto inv_lt = contraction_detail::transpose(contraction_detail::inverse_lower(l));
    const auto whitened = contraction_detail::multiply(
        contraction_detail::multiply(lt, map), inv_lt);
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
