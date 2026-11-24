// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/analysis.hpp"
#include "asdole/matrix_free_differentiation.hpp"

#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace asdole {

// Matrix-free parameter-to-observation derivative. The state trajectory,
// step TLM/adjoint, and parameter step derivatives are all applied as actions;
// no observation sensitivity matrix or state transition matrix is assembled.
class MatrixFreeParameterObservationSensitivityOperator {
public:
    MatrixFreeParameterObservationSensitivityOperator(
        const MatrixFreeTrajectoryLinearization& trajectory,
        std::size_t parameter_dimension,
        InitialParameterAction initial_parameter_apply,
        InitialParameterAction initial_parameter_transpose_apply,
        MatrixFreeParameterDerivativeActions parameter_actions,
        std::vector<ParameterObservationSample> samples)
        : trajectory_(trajectory), parameter_dimension_(parameter_dimension),
          initial_parameter_apply_(std::move(initial_parameter_apply)),
          initial_parameter_transpose_apply_(std::move(initial_parameter_transpose_apply)),
          parameter_actions_(std::move(parameter_actions)), samples_(std::move(samples)) {
        if (parameter_dimension_ == 0 || !initial_parameter_apply_
            || !initial_parameter_transpose_apply_ || samples_.empty())
            throw std::invalid_argument("matrix-free parameter sensitivity needs dimensions, actions, and observations");
        for (const auto& sample : samples_) {
            if (sample.node >= trajectory_.node_count()
                || sample.observation.state_dimension != trajectory_.state_dimension()
                || sample.observation.observation_dimension == 0
                || !sample.observation.apply || !sample.observation.transpose_apply
                || !(sample.quadrature_weight > 0.0) || !std::isfinite(sample.quadrature_weight)
                || static_cast<bool>(sample.direct_parameter_apply)
                   != static_cast<bool>(sample.direct_parameter_transpose_apply))
                throw std::invalid_argument("invalid matrix-free parameter observation sample");
        }
    }

    [[nodiscard]] std::size_t input_dimension() const noexcept { return parameter_dimension_; }
    [[nodiscard]] std::size_t parameter_dimension() const noexcept { return parameter_dimension_; }
    [[nodiscard]] std::size_t output_dimension() const noexcept {
        std::size_t result = 0;
        for (const auto& sample : samples_) result += sample.observation.observation_dimension;
        return result;
    }

    [[nodiscard]] Vector apply(std::span<const double> parameter_direction) const {
        if (parameter_direction.size() != parameter_dimension_)
            throw std::invalid_argument("parameter sensitivity direction has wrong dimension");
        const auto& states = trajectory_.states();
        std::vector<Vector> state_directions(trajectory_.node_count());
        state_directions[0] = initial_parameter_apply_(parameter_direction);
        if (state_directions[0].size() != trajectory_.state_dimension())
            throw std::invalid_argument("initial parameter action returned wrong dimension");
        for (std::size_t k = 0; k < trajectory_.step_count(); ++k) {
            Vector next = trajectory_.apply_step(state_directions[k], k);
            const Vector direct = trajectory_.parameter_tangent_step(
                parameter_direction, k, parameter_actions_);
            add_state(next, direct);
            state_directions[k + 1] = std::move(next);
        }

        Vector result;
        result.reserve(output_dimension());
        for (const auto& sample : samples_) {
            Vector observation = sample.observation(state_directions[sample.node]);
            if (sample.direct_parameter_apply) {
                const Vector direct = sample.direct_parameter_apply(
                    sample.node, states[sample.node], parameter_direction);
                if (direct.size() != observation.size())
                    throw std::invalid_argument("direct observation parameter action returned wrong dimension");
                add_state(observation, direct);
            }
            result.insert(result.end(), observation.begin(), observation.end());
        }
        return result;
    }

    [[nodiscard]] Vector transpose_apply(std::span<const double> stacked_cotangent) const {
        if (stacked_cotangent.size() != output_dimension())
            throw std::invalid_argument("stacked observation cotangent has wrong dimension");
        const std::size_t n = trajectory_.state_dimension();
        const auto& states = trajectory_.states();
        std::vector<Vector> node_loads(trajectory_.node_count(), Vector(n, 0.0));
        Vector parameter_gradient(parameter_dimension_, 0.0);
        std::size_t offset = 0;
        for (const auto& sample : samples_) {
            const std::size_t m = sample.observation.observation_dimension;
            const auto block = stacked_cotangent.subspan(offset, m);
            const Vector state_load = sample.observation.transpose(block);
            add_state(node_loads[sample.node], state_load);
            if (sample.direct_parameter_transpose_apply)
                add_parameter(parameter_gradient, sample.direct_parameter_transpose_apply(
                    sample.node, states[sample.node], block));
            offset += m;
        }

        Vector costate = node_loads.back();
        for (std::size_t reverse = trajectory_.step_count(); reverse > 0; --reverse) {
            const std::size_t k = reverse - 1;
            add_parameter(parameter_gradient, trajectory_.parameter_step_pullback(
                costate, k, parameter_actions_, parameter_dimension_));
            costate = trajectory_.adjoint_step(costate, k);
            add_state(costate, node_loads[k]);
        }
        add_parameter(parameter_gradient, initial_parameter_transpose_apply_(costate));
        return parameter_gradient;
    }

    [[nodiscard]] Vector weighted_transpose_apply(std::span<const double> stacked_value) const {
        if (stacked_value.size() != output_dimension())
            throw std::invalid_argument("weighted observation vector has wrong dimension");
        Vector weighted(stacked_value.begin(), stacked_value.end());
        std::size_t offset = 0;
        for (const auto& sample : samples_) {
            const std::size_t m = sample.observation.observation_dimension;
            const auto block = stacked_value.subspan(offset, m);
            Vector value = sample.precision_apply ? sample.precision_apply(block)
                                                  : Vector(block.begin(), block.end());
            if (value.size() != m) throw std::invalid_argument("precision action returned wrong dimension");
            for (std::size_t i = 0; i < m; ++i)
                weighted[offset + i] = sample.quadrature_weight * value[i];
            offset += m;
        }
        return transpose_apply(weighted);
    }

    [[nodiscard]] Vector weighted_gram_action(std::span<const double> parameter_direction) const {
        return weighted_transpose_apply(apply(parameter_direction));
    }

private:
    static void add_state(Vector& target, std::span<const double> value) {
        if (target.size() != value.size()) throw std::invalid_argument("state vector dimensions differ");
        for (std::size_t i = 0; i < target.size(); ++i) target[i] += value[i];
    }
    void add_parameter(Vector& target, std::span<const double> value) const {
        if (target.size() != parameter_dimension_ || value.size() != parameter_dimension_)
            throw std::invalid_argument("parameter vector dimensions differ");
        for (std::size_t i = 0; i < parameter_dimension_; ++i) target[i] += value[i];
    }

    const MatrixFreeTrajectoryLinearization& trajectory_;
    std::size_t parameter_dimension_{};
    InitialParameterAction initial_parameter_apply_;
    InitialParameterAction initial_parameter_transpose_apply_;
    MatrixFreeParameterDerivativeActions parameter_actions_;
    std::vector<ParameterObservationSample> samples_;
};

} // namespace asdole
