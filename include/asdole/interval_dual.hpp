// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "asdole/regional_contraction.hpp"

#include <array>
#include <cstddef>

namespace asdole {

// Forward-mode automatic differentiation over interval values. This encloses
// both a scalar expression and its gradient when evaluated on an interval box.
template <std::size_t Dimension>
struct IntervalDual {
    Interval value;
    std::array<Interval, Dimension> gradient{};

    static IntervalDual variable(Interval value, std::size_t index) {
        if (index >= Dimension) throw std::out_of_range("interval-dual variable index out of range");
        IntervalDual result;
        result.value = value;
        result.gradient[index] = Interval(1.0);
        return result;
    }

    static IntervalDual constant(double value) {
        IntervalDual result;
        result.value = Interval(value);
        return result;
    }
};

template <std::size_t N>
inline IntervalDual<N> operator+(const IntervalDual<N>& a, const IntervalDual<N>& b) {
    IntervalDual<N> result;
    result.value = a.value + b.value;
    for (std::size_t i = 0; i < N; ++i) result.gradient[i] = a.gradient[i] + b.gradient[i];
    return result;
}

template <std::size_t N>
inline IntervalDual<N> operator-(const IntervalDual<N>& a, const IntervalDual<N>& b) {
    IntervalDual<N> result;
    result.value = a.value - b.value;
    for (std::size_t i = 0; i < N; ++i) result.gradient[i] = a.gradient[i] - b.gradient[i];
    return result;
}

template <std::size_t N>
inline IntervalDual<N> operator*(const IntervalDual<N>& a, const IntervalDual<N>& b) {
    IntervalDual<N> result;
    result.value = a.value * b.value;
    for (std::size_t i = 0; i < N; ++i)
        result.gradient[i] = a.gradient[i] * b.value + a.value * b.gradient[i];
    return result;
}

template <std::size_t N>
inline IntervalDual<N> operator/(const IntervalDual<N>& a, const IntervalDual<N>& b) {
    IntervalDual<N> result;
    result.value = a.value / b.value;
    for (std::size_t i = 0; i < N; ++i)
        result.gradient[i] = (a.gradient[i] * b.value - a.value * b.gradient[i])
                           / square(b.value);
    return result;
}

template <std::size_t N>
inline IntervalDual<N> operator+(const IntervalDual<N>& a, double b) {
    return a + IntervalDual<N>::constant(b);
}
template <std::size_t N>
inline IntervalDual<N> operator+(double a, const IntervalDual<N>& b) { return b + a; }
template <std::size_t N>
inline IntervalDual<N> operator-(const IntervalDual<N>& a, double b) {
    return a - IntervalDual<N>::constant(b);
}
template <std::size_t N>
inline IntervalDual<N> operator-(double a, const IntervalDual<N>& b) {
    return IntervalDual<N>::constant(a) - b;
}
template <std::size_t N>
inline IntervalDual<N> operator*(const IntervalDual<N>& a, double b) {
    return a * IntervalDual<N>::constant(b);
}
template <std::size_t N>
inline IntervalDual<N> operator*(double a, const IntervalDual<N>& b) { return b * a; }
template <std::size_t N>
inline IntervalDual<N> operator/(const IntervalDual<N>& a, double b) {
    return a / IntervalDual<N>::constant(b);
}

template <std::size_t N>
inline IntervalDual<N> square(const IntervalDual<N>& a) {
    IntervalDual<N> result;
    result.value = square(a.value);
    for (std::size_t i = 0; i < N; ++i)
        result.gradient[i] = 2.0 * a.value * a.gradient[i];
    return result;
}

} // namespace asdole
