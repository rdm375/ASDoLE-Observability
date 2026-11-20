// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace asdole {

using Vector = std::vector<double>;

// Dense row-major matrix. This small implementation keeps the algorithm
// dependency-free; large pipe meshes should supply a sparse linear solver.
class DenseMatrix {
public:
    explicit DenseMatrix(std::size_t size = 0)
        : rows_(size), cols_(size), values_(size * size, 0.0) {}

    DenseMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), values_(rows * cols, 0.0) {}

    // size() remains the row count for compatibility with existing square
    // matrix callers. New code should use rows() and cols() explicitly.
    [[nodiscard]] std::size_t size() const noexcept { return rows_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    double& operator()(std::size_t row, std::size_t col) {
        return values_.at(row * cols_ + col);
    }
    double operator()(std::size_t row, std::size_t col) const {
        return values_.at(row * cols_ + col);
    }
    void fill(double value) { std::fill(values_.begin(), values_.end(), value); }

private:
    std::size_t rows_{};
    std::size_t cols_{};
    std::vector<double> values_;
};

using RhsFunction = std::function<void(std::span<const double>, std::span<double>)>;
using JacobianFunction = std::function<void(std::span<const double>, DenseMatrix&)>;

struct StepResult {
    // Second-order local extrapolation: y_next = 2 D - S.
    Vector state;
    // One full affine theta step, S.
    Vector full_step;
    // Two affine half steps, D.
    Vector doubled_step;
    // Difference between the doubled step and full step, D - S. This serves
    // as an error indicator; its asymptotic role depends on theta.
    Vector error_estimate;
};

struct StepStages {
    Vector initial;
    Vector rhs;
    DenseMatrix jacobian;
    Vector full_increment;
    Vector first_half_increment;
    Vector second_half_increment;
    Vector state;
};

namespace detail {

class DenseLU {
public:
    explicit DenseLU(DenseMatrix matrix) : lu_(std::move(matrix)), pivots_(lu_.rows()) {
        if (lu_.rows() != lu_.cols()) throw std::invalid_argument("linear system matrix must be square");
        const auto n = lu_.size();
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t pivot = k;
            double largest = std::abs(lu_(k, k));
            for (std::size_t row = k + 1; row < n; ++row) {
                const double candidate = std::abs(lu_(row, k));
                if (candidate > largest) {
                    largest = candidate;
                    pivot = row;
                }
            }
            if (!std::isfinite(largest) || largest <= std::numeric_limits<double>::epsilon()) {
                throw std::runtime_error("ASDoLE linear system is singular or ill-conditioned");
            }
            pivots_[k] = pivot;
            if (pivot != k) {
                for (std::size_t col = 0; col < n; ++col) {
                    std::swap(lu_(k, col), lu_(pivot, col));
                }
            }
            for (std::size_t row = k + 1; row < n; ++row) {
                lu_(row, k) /= lu_(k, k);
                for (std::size_t col = k + 1; col < n; ++col) {
                    lu_(row, col) -= lu_(row, k) * lu_(k, col);
                }
            }
        }
    }

    [[nodiscard]] Vector solve(std::span<const double> rhs) const {
        const auto n = lu_.size();
        if (rhs.size() != n) throw std::invalid_argument("linear-system RHS has wrong size");
        Vector result(rhs.begin(), rhs.end());
        for (std::size_t k = 0; k < n; ++k) {
            if (pivots_[k] != k) std::swap(result[k], result[pivots_[k]]);
            for (std::size_t row = k + 1; row < n; ++row) {
                result[row] -= lu_(row, k) * result[k];
            }
        }
        for (std::size_t reverse = n; reverse > 0; --reverse) {
            const auto row = reverse - 1;
            for (std::size_t col = row + 1; col < n; ++col) {
                result[row] -= lu_(row, col) * result[col];
            }
            result[row] /= lu_(row, row);
        }
        return result;
    }

private:
    DenseMatrix lu_;
    std::vector<std::size_t> pivots_;
};

inline DenseMatrix affine_theta_matrix(const DenseMatrix& jacobian, double theta_h) {
    if (jacobian.rows() != jacobian.cols()) throw std::invalid_argument("Jacobian must be square");
    const auto n = jacobian.size();
    DenseMatrix matrix(n);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            matrix(row, col) = (row == col ? 1.0 : 0.0) - theta_h * jacobian(row, col);
        }
    }
    return matrix;
}

inline DenseMatrix transpose(const DenseMatrix& matrix) {
    DenseMatrix result(matrix.cols(), matrix.rows());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            result(col, row) = matrix(row, col);
        }
    }
    return result;
}

inline Vector matvec(const DenseMatrix& matrix, std::span<const double> x) {
    if (matrix.cols() != x.size()) throw std::invalid_argument("matrix-vector size mismatch");
    Vector result(matrix.rows(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            result[row] += matrix(row, col) * x[col];
        }
    }
    return result;
}

} // namespace detail

// Affine Step Doubling with Local Extrapolation (ASDoLE), generalized to the
// affine theta method. theta=1 is the backward-Euler ASDoLE in the paper.
//
// The RHS and Jacobian are evaluated once at the beginning of the full step.
// The same Jacobian is then used for the full step and both half steps. The
// half-step matrix is factored once and reused for both half-step solves.
class Stepper {
public:
    Stepper(RhsFunction rhs, JacobianFunction jacobian, double theta = 1.0)
        : rhs_(std::move(rhs)), jacobian_(std::move(jacobian)), theta_(theta) {
        if (!rhs_ || !jacobian_) throw std::invalid_argument("ASDoLE requires RHS and Jacobian callbacks");
        if (!std::isfinite(theta_)) throw std::invalid_argument("theta must be finite");
    }

    [[nodiscard]] double theta() const noexcept { return theta_; }

    [[nodiscard]] StepStages step_stages(std::span<const double> initial, double h) const {
        if (initial.empty()) throw std::invalid_argument("ASDoLE state must not be empty");
        if (!(h > 0.0) || !std::isfinite(h)) throw std::invalid_argument("step size must be finite and positive");

        const std::size_t n = initial.size();
        Vector f(n, 0.0);
        DenseMatrix jacobian(n);
        rhs_(initial, f);
        jacobian_(initial, jacobian);
        if (f.size() != n || jacobian.rows() != n || jacobian.cols() != n) {
            throw std::invalid_argument("RHS or Jacobian callback changed the state dimension");
        }

        // Affine theta step: deltaS = h (F + theta J deltaS), so
        // (I - theta h J) deltaS = h F.
        const detail::DenseLU full_lu(detail::affine_theta_matrix(jacobian, theta_ * h));
        Vector full_rhs(n);
        for (std::size_t i = 0; i < n; ++i) full_rhs[i] = h * f[i];
        const Vector delta_s = full_lu.solve(full_rhs);

        // First half step: (I - theta h/2 J) deltaD1 = h/2 F.
        const double half_h = 0.5 * h;
        const detail::DenseLU half_lu(detail::affine_theta_matrix(jacobian, theta_ * half_h));
        Vector first_half_rhs(n);
        for (std::size_t i = 0; i < n; ++i) first_half_rhs[i] = half_h * f[i];
        const Vector delta_d1 = half_lu.solve(first_half_rhs);

        // Second half step uses the same affine expansion about the original
        // state: (I - theta h/2 J) deltaD2 = h/2 (F + J deltaD1).
        const Vector j_delta_d1 = detail::matvec(jacobian, delta_d1);
        Vector second_half_rhs(n);
        for (std::size_t i = 0; i < n; ++i) {
            second_half_rhs[i] = half_h * (f[i] + j_delta_d1[i]);
        }
        const Vector delta_d2 = half_lu.solve(second_half_rhs);

        StepStages result;
        result.initial.assign(initial.begin(), initial.end());
        result.rhs = std::move(f);
        result.jacobian = std::move(jacobian);
        result.full_increment = delta_s;
        result.first_half_increment = delta_d1;
        result.second_half_increment = delta_d2;
        result.state.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            result.state[i] = initial[i] + 2.0 * delta_d1[i] + 2.0 * delta_d2[i] - delta_s[i];
        }
        return result;
    }

    [[nodiscard]] StepResult step(std::span<const double> initial, double h) const {
        const StepStages stages = step_stages(initial, h);
        const auto n = stages.initial.size();
        StepResult result;
        result.full_step.resize(n);
        result.doubled_step.resize(n);
        result.state = stages.state;
        result.error_estimate.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            result.full_step[i] = stages.initial[i] + stages.full_increment[i];
            result.doubled_step[i] = stages.initial[i] + stages.first_half_increment[i]
                                   + stages.second_half_increment[i];
            result.error_estimate[i] = result.doubled_step[i] - result.full_step[i];
        }
        return result;
    }

private:
    RhsFunction rhs_;
    JacobianFunction jacobian_;
    double theta_;
};

} // namespace asdole
