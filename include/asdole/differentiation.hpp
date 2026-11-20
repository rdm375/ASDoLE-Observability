// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/stepper.hpp"

#include <span>
#include <stdexcept>
#include <vector>

namespace asdole {

// hessian[i](j,k) = d^2 f_i / (dy_j dy_k) = dJ_ij / dy_k.
using HessianTensor = std::vector<DenseMatrix>;
using HessianFunction = std::function<void(std::span<const double>, HessianTensor&)>;

namespace detail {

inline void validate_hessian(const HessianTensor& hessian, std::size_t n) {
    if (hessian.size() != n) throw std::invalid_argument("Hessian output dimension mismatch");
    for (const auto& component : hessian) {
        if (component.rows() != n || component.cols() != n)
            throw std::invalid_argument("Hessian matrix dimension mismatch");
    }
}

inline Vector transpose_matvec(const DenseMatrix& matrix, std::span<const double> x) {
    if (matrix.rows() != x.size()) throw std::invalid_argument("transpose matrix-vector size mismatch");
    Vector result(matrix.cols(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            result[col] += matrix(row, col) * x[row];
        }
    }
    return result;
}

inline void add_outer(DenseMatrix& destination,
                      std::span<const double> left,
                      std::span<const double> right,
                      double scale) {
    if (destination.rows() != left.size() || destination.cols() != right.size()) {
        throw std::invalid_argument("outer-product size mismatch");
    }
    for (std::size_t row = 0; row < left.size(); ++row) {
        for (std::size_t col = 0; col < right.size(); ++col) {
            destination(row, col) += scale * left[row] * right[col];
        }
    }
}

inline DenseMatrix jacobian_direction(const HessianTensor& hessian,
                                      std::span<const double> direction) {
    const auto n = direction.size();
    DenseMatrix result(n);
    for (std::size_t output = 0; output < n; ++output) {
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                result(output, row) += hessian[output](row, col) * direction[col];
            }
        }
    }
    return result;
}

inline Vector hessian_contraction(const HessianTensor& hessian,
                                 const DenseMatrix& jacobian_bar) {
    const auto n = jacobian_bar.rows();
    if (jacobian_bar.cols() != n) throw std::invalid_argument("Jacobian cotangent must be square");
    Vector result(n, 0.0);
    for (std::size_t output = 0; output < n; ++output) {
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                result[col] += jacobian_bar(output, row) * hessian[output](row, col);
            }
        }
    }
    return result;
}

} // namespace detail

// Exact tangent-linear action of one generalized ASDoLE step. The Hessian is
// needed because the step freezes J=DF(y) at the start state, but J itself
// changes when the start state is perturbed.
inline Vector tangent_step(const Stepper& stepper,
                          const HessianFunction& hessian_function,
                          std::span<const double> initial,
                          double h,
                          std::span<const double> direction) {
    if (initial.size() != direction.size()) throw std::invalid_argument("TLM direction has wrong size");
    const StepStages stages = stepper.step_stages(initial, h);
    const auto n = initial.size();
    HessianTensor hessian(n, DenseMatrix(n));
    hessian_function(initial, hessian);
    detail::validate_hessian(hessian, n);

    const DenseMatrix d_j = detail::jacobian_direction(hessian, direction);
    const double b = 0.5 * h;
    const double theta = stepper.theta();
    const detail::DenseLU full_lu(detail::affine_theta_matrix(stages.jacobian, theta * h));
    const detail::DenseLU half_lu(detail::affine_theta_matrix(stages.jacobian, theta * b));

    const Vector j_direction = detail::matvec(stages.jacobian, direction);
    const Vector dj_s = detail::matvec(d_j, stages.full_increment);
    Vector rhs_s(n);
    for (std::size_t i = 0; i < n; ++i) {
        rhs_s[i] = h * j_direction[i] + theta * h * dj_s[i];
    }
    const Vector delta_s = full_lu.solve(rhs_s);

    const Vector dj_d1 = detail::matvec(d_j, stages.first_half_increment);
    Vector rhs_d1(n);
    for (std::size_t i = 0; i < n; ++i) {
        rhs_d1[i] = b * j_direction[i] + theta * b * dj_d1[i];
    }
    const Vector delta_d1 = half_lu.solve(rhs_d1);

    const Vector j_delta_d1 = detail::matvec(stages.jacobian, delta_d1);
    const Vector dj_d1_again = detail::matvec(d_j, stages.first_half_increment);
    const Vector dj_d2 = detail::matvec(d_j, stages.second_half_increment);
    Vector rhs_d2(n);
    for (std::size_t i = 0; i < n; ++i) {
        rhs_d2[i] = b * (j_direction[i] + dj_d1_again[i] + j_delta_d1[i]
                         + theta * dj_d2[i]);
    }
    const Vector delta_d2 = half_lu.solve(rhs_d2);

    Vector result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = direction[i] + 2.0 * delta_d1[i] + 2.0 * delta_d2[i] - delta_s[i];
    }
    return result;
}

// Direct parameter derivative of one ASDoLE step, holding its initial state
// fixed. The caller supplies f_mu and J_mu; the result is the B_k term in
// d y_{k+1} = A_k d y_k + B_k d mu.
inline Vector parameter_tangent_step(const Stepper& stepper,
                                     std::span<const double> initial,
                                     double h,
                                     std::span<const double> rhs_parameter_derivative,
                                     const DenseMatrix& jacobian_parameter_derivative) {
    const auto n = initial.size();
    if (rhs_parameter_derivative.size() != n || jacobian_parameter_derivative.rows() != n
        || jacobian_parameter_derivative.cols() != n)
        throw std::invalid_argument("parameter derivative dimension mismatch");
    const StepStages stages = stepper.step_stages(initial, h);
    const double b = 0.5 * h;
    const double theta = stepper.theta();
    const detail::DenseLU full_lu(detail::affine_theta_matrix(stages.jacobian, theta * h));
    const detail::DenseLU half_lu(detail::affine_theta_matrix(stages.jacobian, theta * b));

    const Vector jmu_s = detail::matvec(jacobian_parameter_derivative, stages.full_increment);
    Vector rhs_s(n);
    for (std::size_t i = 0; i < n; ++i)
        rhs_s[i] = h * rhs_parameter_derivative[i] + theta * h * jmu_s[i];
    const Vector delta_s = full_lu.solve(rhs_s);

    const Vector jmu_d1 = detail::matvec(jacobian_parameter_derivative, stages.first_half_increment);
    Vector rhs_d1(n);
    for (std::size_t i = 0; i < n; ++i)
        rhs_d1[i] = b * rhs_parameter_derivative[i] + theta * b * jmu_d1[i];
    const Vector delta_d1 = half_lu.solve(rhs_d1);

    const Vector j_delta_d1 = detail::matvec(stages.jacobian, delta_d1);
    const Vector jmu_d2 = detail::matvec(jacobian_parameter_derivative, stages.second_half_increment);
    Vector rhs_d2(n);
    for (std::size_t i = 0; i < n; ++i)
        rhs_d2[i] = b * (rhs_parameter_derivative[i] + jmu_d1[i] + j_delta_d1[i]
                         + theta * jmu_d2[i]);
    const Vector delta_d2 = half_lu.solve(rhs_d2);

    Vector result(n);
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 2.0 * delta_d1[i] + 2.0 * delta_d2[i] - delta_s[i];
    return result;
}

// Reverse-mode action of the same ASDoLE step. Given a cotangent at the step
// output, return its pullback to the input. This differentiates through the
// implicit full/half solves and through the state-dependent Jacobian.
inline Vector adjoint_step(const Stepper& stepper,
                          const HessianFunction& hessian_function,
                          std::span<const double> initial,
                          double h,
                          std::span<const double> output_cotangent) {
    if (initial.size() != output_cotangent.size()) {
        throw std::invalid_argument("adjoint cotangent has wrong size");
    }
    const StepStages stages = stepper.step_stages(initial, h);
    const auto n = initial.size();
    HessianTensor hessian(n, DenseMatrix(n));
    hessian_function(initial, hessian);
    detail::validate_hessian(hessian, n);

    const double b = 0.5 * h;
    const double theta = stepper.theta();
    const DenseMatrix full_matrix = detail::affine_theta_matrix(stages.jacobian, theta * h);
    const DenseMatrix half_matrix = detail::affine_theta_matrix(stages.jacobian, theta * b);
    const detail::DenseLU full_transpose_lu(detail::transpose(full_matrix));
    const detail::DenseLU half_transpose_lu(detail::transpose(half_matrix));

    Vector bar_full(n), bar_first(n), bar_second(n);
    for (std::size_t i = 0; i < n; ++i) {
        bar_full[i] = -output_cotangent[i];
        bar_first[i] = 2.0 * output_cotangent[i];
        bar_second[i] = 2.0 * output_cotangent[i];
    }

    Vector bar_f(n, 0.0);
    DenseMatrix bar_j(n);

    // Reverse the second half-step solve:
    // M_b d2 = b (f + J d1), M_b = I - theta b J.
    const Vector lambda_second = half_transpose_lu.solve(bar_second);
    const Vector jt_lambda_second = detail::transpose_matvec(stages.jacobian, lambda_second);
    for (std::size_t i = 0; i < n; ++i) {
        bar_f[i] += b * lambda_second[i];
        bar_first[i] += b * jt_lambda_second[i];
    }
    detail::add_outer(bar_j, lambda_second, stages.first_half_increment, b);
    detail::add_outer(bar_j, lambda_second, stages.second_half_increment, theta * b);

    // Reverse the first half-step and the full-step solves.
    const Vector lambda_first = half_transpose_lu.solve(bar_first);
    const Vector lambda_full = full_transpose_lu.solve(bar_full);
    for (std::size_t i = 0; i < n; ++i) {
        bar_f[i] += b * lambda_first[i] + h * lambda_full[i];
    }
    detail::add_outer(bar_j, lambda_first, stages.first_half_increment, theta * b);
    detail::add_outer(bar_j, lambda_full, stages.full_increment, theta * h);

    Vector result(output_cotangent.begin(), output_cotangent.end());
    const Vector jt_bar_f = detail::transpose_matvec(stages.jacobian, bar_f);
    const Vector hessian_bar_j = detail::hessian_contraction(hessian, bar_j);
    for (std::size_t i = 0; i < n; ++i) result[i] += jt_bar_f[i] + hessian_bar_j[i];
    return result;
}

} // namespace asdole
