// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/contraction.hpp"
#include "asdole/regional_contraction.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace asdole;

struct IntervalDual {
    Interval value;
    Interval tangent;
};

IntervalDual operator+(IntervalDual a, IntervalDual b) {
    return {a.value+b.value, a.tangent+b.tangent};
}
IntervalDual operator-(IntervalDual a, IntervalDual b) {
    return {a.value-b.value, a.tangent-b.tangent};
}
IntervalDual operator*(IntervalDual a, IntervalDual b) {
    return {a.value*b.value, a.tangent*b.value+a.value*b.tangent};
}
IntervalDual operator/(IntervalDual a, IntervalDual b) {
    return {a.value/b.value,
            (a.tangent*b.value-a.value*b.tangent)/(b.value*b.value)};
}
IntervalDual square(IntervalDual a) {
    return {asdole::square(a.value), 2.0*a.value*a.tangent};
}
IntervalDual operator+(IntervalDual a, double b) { return a+IntervalDual{Interval(b),Interval(0.0)}; }
IntervalDual operator+(double a, IntervalDual b) { return b+a; }
IntervalDual operator-(double a, IntervalDual b) { return IntervalDual{Interval(a),Interval(0.0)}-b; }
IntervalDual operator*(IntervalDual a, double b) { return a*IntervalDual{Interval(b),Interval(0.0)}; }
IntervalDual operator*(double a, IntervalDual b) { return b*a; }

IntervalDual scalar_nonlinear_asdole_step(const Interval& box, double h, double theta) {
    const IntervalDual y{box, Interval(1.0)};
    const IntervalDual f = -0.8*y + 0.1*square(y);
    const IntervalDual j = -0.8 + 0.2*y;
    const IntervalDual full = h*f/(1.0-theta*h*j);
    const double half_h = 0.5*h;
    const IntervalDual first_half = half_h*f/(1.0-theta*half_h*j);
    const IntervalDual second_half = half_h*(f+j*first_half)/(1.0-theta*half_h*j);
    return y+2.0*first_half+2.0*second_half-full;
}

void check_continuous_certificate_scaling() {
    DenseMatrix unstable(1), tiny_weight(1);
    unstable(0,0) = 1.0;
    tiny_weight(0,0) = 1.0e-12;
    const auto false_positive = certify_continuous_contraction(
        unstable, tiny_weight, 0.1);
    if (false_positive.certified)
        throw std::runtime_error("unstable scalar system passed after scaling W down");

    DenseMatrix stable(1), unit_weight(1), scaled_weight(1);
    stable(0,0) = -2.0;
    unit_weight(0,0) = 1.0;
    scaled_weight(0,0) = 1.0e-24;
    const auto unit = certify_continuous_contraction(stable, unit_weight, 0.5);
    const auto scaled = certify_continuous_contraction(stable, scaled_weight, 0.5);
    if (!unit.certified || !scaled.certified)
        throw std::runtime_error("stable system's certificate changed under W rescaling");
    if (std::abs(unit.largest_residual_eigenvalue
                 - scaled.largest_residual_eigenvalue) > 1.0e-12)
        throw std::runtime_error("normalized residual changed under W rescaling");

    std::cout << "unstable_small_W_rejected=yes stable_W_rescaling_invariant=yes\n";
}

void check_weighted_induced_norm_scaling() {
    DenseMatrix map(2), weight(2), scaled_weight(2);
    map(0,0) = 0.5;
    map(1,1) = 0.25;
    weight(0,0) = 1.0;
    weight(1,1) = 4.0;
    scaled_weight(0,0) = 1.0e-24;
    scaled_weight(1,1) = 4.0e-24;
    const double reference = weighted_induced_norm(map, weight);
    const double scaled = weighted_induced_norm(map, scaled_weight);
    if (std::abs(reference - 0.5) > 1.0e-12
        || std::abs(reference - scaled) > 1.0e-12)
        throw std::runtime_error("weighted induced norm changed under W rescaling");
    std::cout << "weighted_norm=" << reference << " W_rescaling_invariant=yes\n";
}

void check_automatic_lyapunov_metric() {
    DenseMatrix a(2);
    a(0,0) = -1.0;
    a(1,0) = 1.0;
    a(1,1) = -1.0;
    const auto metric = estimate_lyapunov_metric(a, 0.8);
    if (!metric.certificate.certified || !(metric.estimated_rate_limit > 0.95)
        || std::abs(metric.requested_rate - 0.8*metric.estimated_rate_limit) > 1.0e-12)
        throw std::runtime_error("automatic Lyapunov metric did not recover the expected stable rate");
    const auto verified = certify_continuous_contraction(a, metric.weight, metric.requested_rate);
    if (!verified.certified)
        throw std::runtime_error("automatically selected W failed the public contraction check");

    DenseMatrix unstable(1);
    unstable(0,0) = 0.25;
    bool rejected = false;
    try { (void)estimate_lyapunov_metric(unstable); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error("unstable system received an automatic Lyapunov metric");
    std::cout << "automatic_metric_rate=" << metric.requested_rate
              << " estimated_limit=" << metric.estimated_rate_limit
              << " unstable_rejected=yes\n";
}

IntervalVectorField polynomial_field_enclosure() {
    return [](const IntervalBox& x) {
        return std::vector<Interval>{
            -0.7*x[0] + 0.15*x[1] + 0.04*x[0]*x[0] + 0.03*x[0]*x[1],
            -0.2*x[0] - 1.1*x[1] + 0.02*x[0]*x[1] + 0.05*x[1]*x[1]};
    };
}

IntervalJacobian polynomial_jacobian_enclosure() {
    return [](const IntervalBox& x) {
        return std::vector<std::vector<Interval>>{
            {-0.7 + 0.08*x[0] + 0.03*x[1], 0.15 + 0.03*x[0]},
            {-0.2 + 0.02*x[1], -1.1 + 0.02*x[0] + 0.10*x[1]}};
    };
}

void check_regional_nonlinear_certificate() {
    const IntervalBox box{Interval(-1.0, 1.0), Interval(-1.0, 1.0)};
    const auto certificate = certify_nonlinear_box(
        box, polynomial_field_enclosure(), polynomial_jacobian_enclosure());
    if (!certificate.forward_invariant || !certificate.contraction_certified
        || !(certificate.contraction_rate > 0.0))
        throw std::runtime_error("polynomial regional certificate did not prove invariance and contraction");

    const IntervalVectorField shifted_decay = [](const IntervalBox& x) {
        return std::vector<Interval>{-x[0] + 2.0, -x[1]};
    };
    const IntervalJacobian decay_jacobian = [](const IntervalBox&) {
        return std::vector<std::vector<Interval>>{{Interval(-1.0), Interval(0.0)},
                                                  {Interval(0.0), Interval(-1.0)}};
    };
    const auto noninvariant = certify_nonlinear_box(
        box, shifted_decay, decay_jacobian);
    if (noninvariant.forward_invariant || !noninvariant.contraction_certified)
        throw std::runtime_error("regional certificate failed to separate invariance from contraction");
    std::cout << "box_invariant=yes contraction_rate=" << certificate.contraction_rate
              << " noninvariant_box_rejected=yes\n";
}

void check_discrete_asdole_box_certificate() {
    constexpr double h = 0.1;
    constexpr double theta = 0.7;
    const IntervalBox box{Interval(-0.3, 0.3)};
    const IntervalDiscreteMap map = [](const IntervalBox& x) {
        // The range callback sharpens the natural interval extension by using
        // the positive derivative over the box: the scalar map is monotone,
        // so its extrema occur at the two endpoints.
        const auto derivative = scalar_nonlinear_asdole_step(x[0], h, theta).tangent;
        if (!(derivative.lower > 0.0))
            throw std::runtime_error("cannot use monotonic endpoint enclosure on this box");
        const auto left = scalar_nonlinear_asdole_step(Interval(x[0].lower), h, theta).value;
        const auto right = scalar_nonlinear_asdole_step(Interval(x[0].upper), h, theta).value;
        return std::vector<Interval>{Interval(std::min(left.lower, right.lower),
                                              std::max(left.upper, right.upper))};
    };
    const IntervalDiscreteJacobian jacobian = [](const IntervalBox& x) {
        return std::vector<std::vector<Interval>>{
            {scalar_nonlinear_asdole_step(x[0], h, theta).tangent}};
    };
    const auto result = certify_discrete_box_contraction(box, map, jacobian);
    if (!result.maps_into_box || !result.contraction_certified
        || !(result.induced_factor_bound < 1.0))
        throw std::runtime_error("nonlinear ASDoLE map was not certified contractive on its box");
    std::cout << "discrete_asdole_box_invariant=yes induced_factor_bound="
              << result.induced_factor_bound << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected a named contraction check");
        const std::string check = argv[1];
        if (check == "continuous") check_continuous_certificate_scaling();
        else if (check == "discrete_norm") check_weighted_induced_norm_scaling();
        else if (check == "metric") check_automatic_lyapunov_metric();
        else if (check == "regional") check_regional_nonlinear_certificate();
        else if (check == "discrete_box") check_discrete_asdole_box_certificate();
        else throw std::invalid_argument("unknown check name: " + check);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "contraction_checks: " << error.what() << '\n';
        return 1;
    }
}
