// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/stepper.hpp"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace asdole {

struct ObservationObjective {
    double value{};
    // Discrete derivative of the objective with respect to each state node.
    std::vector<Vector> state_gradient;
};

// Evaluate 1/2 integral r(t)^T W r(t) dt by composite trapezoidal
// quadrature on the actual trajectory times. H maps n states to m sensors;
// precision is the m-by-m observation weight matrix W.
inline ObservationObjective trapezoidal_observation_objective(
    std::span<const double> times,
    const std::vector<Vector>& states,
    const std::vector<Vector>& data,
    const DenseMatrix& observation,
    const DenseMatrix& precision) {
    const std::size_t count = times.size();
    if (count < 2 || states.size() != count || data.size() != count)
        throw std::invalid_argument("observation objective needs matching trajectory, data, and at least two times");
    const std::size_t n = states.front().size();
    const std::size_t m = observation.rows();
    if (n == 0 || m == 0 || observation.cols() != n
        || precision.rows() != m || precision.cols() != m)
        throw std::invalid_argument("observation or precision matrix has inconsistent dimensions");
    for (std::size_t k = 0; k < count; ++k) {
        if (!(times[k] >= 0.0) || !std::isfinite(times[k])
            || (k > 0 && !(times[k] > times[k - 1])))
            throw std::invalid_argument("trajectory times must be finite and strictly increasing");
        if (states[k].size() != n || data[k].size() != m)
            throw std::invalid_argument("state or observation data dimension mismatch");
    }

    ObservationObjective result;
    result.state_gradient.assign(count, Vector(n, 0.0));
    for (std::size_t k = 0; k < count; ++k) {
        const double weight = k == 0 ? 0.5 * (times[1] - times[0])
            : (k + 1 == count ? 0.5 * (times[k] - times[k - 1])
                              : 0.5 * (times[k + 1] - times[k - 1]));
        const Vector prediction = detail::matvec(observation, states[k]);
        Vector residual(m);
        for (std::size_t i = 0; i < m; ++i) residual[i] = prediction[i] - data[k][i];
        const Vector weighted_residual = detail::matvec(precision, residual);
        for (std::size_t i = 0; i < m; ++i)
            result.value += 0.5 * weight * residual[i] * weighted_residual[i];
        result.state_gradient[k] = detail::transpose_matvec(observation, weighted_residual);
        for (double& component : result.state_gradient[k]) component *= weight;
    }
    return result;
}

} // namespace asdole
