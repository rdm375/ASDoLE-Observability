// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace asdole {

// An interval enclosure [lower, upper]. Callback implementations must return
// outward enclosures of the exact range over the supplied box for the result
// to be a mathematical certificate.
struct Interval {
    double lower{};
    double upper{};

    Interval() = default;
    explicit Interval(double value) : lower(value), upper(value) {
        if (!std::isfinite(value)) throw std::invalid_argument("interval endpoint must be finite");
    }
    Interval(double lo, double hi) : lower(lo), upper(hi) {
        if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
            throw std::invalid_argument("invalid interval bounds");
    }
};

inline Interval operator+(Interval a, Interval b) {
    return {std::nextafter(a.lower+b.lower, -std::numeric_limits<double>::infinity()),
            std::nextafter(a.upper+b.upper,  std::numeric_limits<double>::infinity())};
}
inline Interval operator-(Interval a, Interval b) {
    return {std::nextafter(a.lower-b.upper, -std::numeric_limits<double>::infinity()),
            std::nextafter(a.upper-b.lower,  std::numeric_limits<double>::infinity())};
}
inline Interval operator*(Interval a, Interval b) {
    const double p[] = {a.lower*b.lower, a.lower*b.upper,
                        a.upper*b.lower, a.upper*b.upper};
    return {std::nextafter(*std::min_element(std::begin(p), std::end(p)),
                           -std::numeric_limits<double>::infinity()),
            std::nextafter(*std::max_element(std::begin(p), std::end(p)),
                            std::numeric_limits<double>::infinity())};
}
inline Interval operator*(double a, Interval b) {
    return Interval(a) * b;
}
inline Interval operator*(Interval a, double b) {
    return a * Interval(b);
}
inline Interval operator+(double a, Interval b) { return Interval(a) + b; }
inline Interval operator+(Interval a, double b) { return a + Interval(b); }
inline Interval operator-(double a, Interval b) { return Interval(a) - b; }
inline Interval operator-(Interval a, double b) { return a - Interval(b); }
inline Interval operator-(Interval a) { return Interval(0.0) - a; }
inline Interval operator/(Interval a, Interval b) {
    if (b.lower <= 0.0 && b.upper >= 0.0)
        throw std::domain_error("interval division by an interval containing zero");
    const double r0 = 1.0 / b.lower;
    const double r1 = 1.0 / b.upper;
    const Interval reciprocal(
        std::nextafter(std::min(r0, r1), -std::numeric_limits<double>::infinity()),
        std::nextafter(std::max(r0, r1), std::numeric_limits<double>::infinity()));
    return a * reciprocal;
}
inline Interval square(Interval a) {
    const double lo = a.lower <= 0.0 && a.upper >= 0.0
        ? 0.0 : std::min(a.lower*a.lower, a.upper*a.upper);
    const double hi = std::max(a.lower*a.lower, a.upper*a.upper);
    return {std::nextafter(lo, -std::numeric_limits<double>::infinity()),
            std::nextafter(hi, std::numeric_limits<double>::infinity())};
}

using IntervalBox = std::vector<Interval>;
using IntervalVectorField = std::function<std::vector<Interval>(const IntervalBox&)>;
using IntervalJacobian = std::function<std::vector<std::vector<Interval>>(const IntervalBox&)>;
using IntervalDiscreteMap = IntervalVectorField;
using IntervalDiscreteJacobian = IntervalJacobian;

struct RegionalContractionCertificate {
    bool forward_invariant{};
    bool contraction_certified{};
    double contraction_rate{};
    double upper_logarithmic_norm{};
};

namespace regional_contraction_detail {

inline void validate_interval(const Interval& interval) {
    if (!std::isfinite(interval.lower) || !std::isfinite(interval.upper)
        || interval.lower > interval.upper)
        throw std::invalid_argument("interval callback returned invalid bounds");
}

inline void validate_box(const IntervalBox& box) {
    if (box.empty()) throw std::invalid_argument("regional box must be nonempty");
    for (const auto& interval : box) validate_interval(interval);
}

inline std::vector<Interval> evaluate_field(const IntervalVectorField& field,
                                            const IntervalBox& box) {
    if (!field) throw std::invalid_argument("field enclosure callback is required");
    auto result = field(box);
    if (result.size() != box.size())
        throw std::invalid_argument("field enclosure has wrong dimension");
    for (const auto& interval : result) validate_interval(interval);
    return result;
}

inline std::vector<std::vector<Interval>> evaluate_jacobian(
    const IntervalJacobian& jacobian, const IntervalBox& box) {
    if (!jacobian) throw std::invalid_argument("Jacobian enclosure callback is required");
    auto result = jacobian(box);
    if (result.size() != box.size())
        throw std::invalid_argument("Jacobian enclosure has wrong row count");
    for (const auto& row : result) {
        if (row.size() != box.size()) throw std::invalid_argument("Jacobian enclosure has wrong column count");
        for (const auto& interval : row) validate_interval(interval);
    }
    return result;
}

} // namespace regional_contraction_detail

// Certify forward invariance of an axis-aligned box and Euclidean contraction.
// A symmetric Gershgorin bound encloses the largest eigenvalue of (J+J^T)/2.
// This can be conservative, but never relies on point sampling.
inline RegionalContractionCertificate certify_nonlinear_box(
    const IntervalBox& box,
    const IntervalVectorField& field_enclosure,
    const IntervalJacobian& jacobian_enclosure) {
    regional_contraction_detail::validate_box(box);

    bool invariant = true;
    for (std::size_t axis = 0; axis < box.size(); ++axis) {
        IntervalBox face = box;
        face[axis] = Interval(box[axis].lower);
        const auto at_lower = regional_contraction_detail::evaluate_field(field_enclosure, face);
        invariant = invariant && at_lower[axis].lower >= 0.0;
        face[axis] = Interval(box[axis].upper);
        const auto at_upper = regional_contraction_detail::evaluate_field(field_enclosure, face);
        invariant = invariant && at_upper[axis].upper <= 0.0;
    }

    const auto j = regional_contraction_detail::evaluate_jacobian(jacobian_enclosure, box);
    const auto n = box.size();
    double upper_log_norm = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        double row_bound = j[i][i].upper;
        for (std::size_t k = 0; k < n; ++k) {
            if (k == i) continue;
            const Interval symmetric = 0.5 * (j[i][k] + j[k][i]);
            row_bound = std::nextafter(
                row_bound + std::max(std::abs(symmetric.lower), std::abs(symmetric.upper)),
                std::numeric_limits<double>::infinity());
        }
        upper_log_norm = std::max(upper_log_norm, row_bound);
    }
    return {invariant, upper_log_norm < 0.0,
            upper_log_norm < 0.0 ? -upper_log_norm : 0.0,
            upper_log_norm};
}

struct DiscreteRegionalContractionCertificate {
    bool maps_into_box{};
    bool contraction_certified{};
    double induced_factor_bound{};
    double gram_eigenvalue_upper_bound{};
};

// Certify that a discrete map sends the whole box into itself and has induced
// Euclidean Jacobian norm below one throughout the box. The callbacks must
// enclose the complete one-step map and Jacobian ranges. The Gershgorin bound
// is applied to an interval enclosure of J^T J.
inline DiscreteRegionalContractionCertificate certify_discrete_box_contraction(
    const IntervalBox& box,
    const IntervalDiscreteMap& map_enclosure,
    const IntervalDiscreteJacobian& jacobian_enclosure) {
    regional_contraction_detail::validate_box(box);
    const auto image = regional_contraction_detail::evaluate_field(map_enclosure, box);
    bool maps_into_box = true;
    for (std::size_t i = 0; i < box.size(); ++i)
        maps_into_box = maps_into_box && image[i].lower >= box[i].lower
                                     && image[i].upper <= box[i].upper;

    const auto jacobian = regional_contraction_detail::evaluate_jacobian(jacobian_enclosure, box);
    const auto n = box.size();
    double gram_upper = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        Interval diagonal(0.0);
        for (std::size_t k = 0; k < n; ++k)
            diagonal = diagonal + jacobian[k][i] * jacobian[k][i];
        double row_bound = diagonal.upper;
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            Interval off_diagonal(0.0);
            for (std::size_t k = 0; k < n; ++k)
                off_diagonal = off_diagonal + jacobian[k][i] * jacobian[k][j];
            row_bound = std::nextafter(
                row_bound + std::max(std::abs(off_diagonal.lower), std::abs(off_diagonal.upper)),
                std::numeric_limits<double>::infinity());
        }
        gram_upper = std::max(gram_upper, row_bound);
    }
    gram_upper = std::max(0.0, gram_upper);
    const double factor = std::sqrt(gram_upper);
    return {maps_into_box, factor < 1.0, factor, gram_upper};
}

} // namespace asdole
