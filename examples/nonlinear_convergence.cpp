// Copyright (c) 2026 Richard Douglas Myers
// SPDX-License-Identifier: AGPL-3.0-only

#include "asdole/differentiation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {
using asdole::DenseMatrix;
using asdole::HessianTensor;
using asdole::Vector;
constexpr double final_time = 1.0;
constexpr int reference_steps = 131072;
constexpr std::size_t dimension = 2;

void rhs(std::span<const double> y, std::span<double> f) {
    f[0] = -0.7 * y[0] + 0.15 * y[1] + 0.04 * y[0] * y[0] + 0.03 * y[0] * y[1];
    f[1] = -0.2 * y[0] - 1.1 * y[1] + 0.02 * y[0] * y[1] + 0.05 * y[1] * y[1];
}

void jacobian(std::span<const double> y, DenseMatrix& j) {
    j(0, 0) = -0.7 + 0.08 * y[0] + 0.03 * y[1];
    j(0, 1) = 0.15 + 0.03 * y[0];
    j(1, 0) = -0.2 + 0.02 * y[1];
    j(1, 1) = -1.1 + 0.02 * y[0] + 0.10 * y[1];
}

void hessian(std::span<const double>, HessianTensor& h) {
    h[0](0, 0) = 0.08;
    h[0](0, 1) = h[0](1, 0) = 0.03;
    h[1](0, 1) = h[1](1, 0) = 0.02;
    h[1](1, 1) = 0.10;
}

Vector observation(double t) {
    return {0.65 * std::exp(-0.3 * t), -0.25 * std::exp(-0.5 * t)};
}
Vector loss_gradient(std::span<const double> y, double t) {
    const auto obs = observation(t);
    return {y[0] - obs[0], y[1] - obs[1]};
}

double norm(std::span<const double> x) {
    double sum = 0.0;
    for (double v : x) sum += v * v;
    return std::sqrt(sum);
}

// Reference integration state: y[0:2], column-major sensitivity [2:6],
// and the integral objective in component 6.
using RefState = std::array<double, 7>;
RefState reference_rhs(const RefState& z, double t) {
    const Vector y{z[0], z[1]};
    Vector f(2);
    rhs(y, f);
    DenseMatrix j(2);
    jacobian(y, j);
    RefState dz{};
    dz[0] = f[0]; dz[1] = f[1];
    for (std::size_t col = 0; col < 2; ++col) {
        for (std::size_t row = 0; row < 2; ++row) {
            dz[2 + 2 * col + row] = j(row, 0) * z[2 + 2 * col]
                                        + j(row, 1) * z[2 + 2 * col + 1];
        }
    }
    const auto g = loss_gradient(y, t);
    dz[6] = 0.5 * (g[0] * g[0] + g[1] * g[1]);
    return dz;
}

RefState add_scaled(const RefState& x, const RefState& k, double scale) {
    RefState out{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = x[i] + scale * k[i];
    return out;
}

std::vector<RefState> make_reference() {
    const double dt = final_time / reference_steps;
    std::vector<RefState> ref(reference_steps + 1);
    ref[0][0] = 0.8; ref[0][1] = -0.35;
    ref[0][2] = 1.0; ref[0][5] = 1.0; // identity initial sensitivity
    for (int n = 0; n < reference_steps; ++n) {
        const auto& z = ref[n];
        const auto k1 = reference_rhs(z, n * dt);
        const auto k2 = reference_rhs(add_scaled(z, k1, 0.5 * dt), (n + 0.5) * dt);
        const auto k3 = reference_rhs(add_scaled(z, k2, 0.5 * dt), (n + 0.5) * dt);
        const auto k4 = reference_rhs(add_scaled(z, k3, dt), (n + 1.0) * dt);
        for (std::size_t i = 0; i < z.size(); ++i)
            ref[n + 1][i] = z[i] + (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }
    return ref;
}

Vector reference_state_at(const std::vector<RefState>& ref, double t) {
    const double coordinate = std::clamp(t * reference_steps / final_time, 0.0,
                                         static_cast<double>(reference_steps));
    const auto lo = static_cast<int>(coordinate);
    const auto hi = std::min(lo + 1, reference_steps);
    const double alpha = coordinate - lo;
    return {(1.0-alpha)*ref[lo][0] + alpha*ref[hi][0],
            (1.0-alpha)*ref[lo][1] + alpha*ref[hi][1]};
}

struct Errors { double state{}, tlm{}, costate{}, objective{}; };
struct Run { Errors errors; };

Run solve(double theta, int steps, const std::vector<RefState>& ref,
          const std::vector<Vector>& lambda_ref) {
    const double h = final_time / steps;
    const asdole::Stepper stepper(rhs, jacobian, theta);
    std::vector<Vector> y(steps + 1, Vector(2));
    std::vector<std::array<double, 4>> tangent(steps + 1);
    std::vector<Vector> costate(steps + 1, Vector(2));
    y[0] = {0.8, -0.35}; tangent[0] = {1.0, 0.0, 0.0, 1.0};
    double objective = 0.0;
    for (int k = 0; k < steps; ++k) {
        y[k+1] = stepper.step(y[k], h).state;
        for (std::size_t col=0; col<2; ++col) {
            Vector v{tangent[k][2*col], tangent[k][2*col+1]};
            const Vector next = asdole::tangent_step(stepper, hessian, y[k], h, v);
            tangent[k+1][2*col] = next[0]; tangent[k+1][2*col+1] = next[1];
        }
    }
    for (int k=0; k<=steps; ++k) {
        const auto g=loss_gradient(y[k], h*k);
        objective += h*((k==0 || k==steps) ? 0.25 : 0.5)*(g[0]*g[0]+g[1]*g[1]);
    }
    auto gN=loss_gradient(y[steps], final_time);
    costate[steps]={0.5*h*gN[0], 0.5*h*gN[1]};
    for (int k=steps-1; k>=0; --k) {
        const Vector pulled=asdole::adjoint_step(stepper, hessian, y[k], h, costate[k+1]);
        const Vector g=loss_gradient(y[k], h*k);
        const double weight=(k==0)?0.5:1.0;
        costate[k]={pulled[0]+h*weight*g[0], pulled[1]+h*weight*g[1]};
    }
    // Convert the discrete multiplier to the nodal continuous-costate convention.
    for (int k=1; k<=steps; ++k) {
        const Vector g=loss_gradient(y[k], h*k);
        costate[k][0]-=0.5*h*g[0]; costate[k][1]-=0.5*h*g[1];
    }

    Run run;
    const int stride=reference_steps/steps;
    for (int k=0; k<=steps; ++k) {
        const int ri=k*stride;
        const Vector yr{ref[ri][0], ref[ri][1]};
        run.errors.state=std::max(run.errors.state,norm(Vector{y[k][0]-yr[0],y[k][1]-yr[1]}));
        double td=0.0;
        for (std::size_t i=0;i<4;++i) td+=std::pow(tangent[k][i]-ref[ri][2+i],2);
        run.errors.tlm=std::max(run.errors.tlm,std::sqrt(td));
        run.errors.costate=std::max(run.errors.costate,
            norm(Vector{costate[k][0]-lambda_ref[ri][0],costate[k][1]-lambda_ref[ri][1]}));
    }
    run.errors.objective=std::abs(objective-ref.back()[6]);
    return run;
}
} // namespace

int main() {
    constexpr std::array theta_values{0.5, 0.6, 0.7, 1.0};
    constexpr int levels=8;
    constexpr double minimum_order=1.85;
    const auto ref=make_reference();
    // Continuous costate reference is integrated backward on the same fine RK4 grid.
    const double ref_dt=final_time/reference_steps;
    std::vector<Vector> lambda_ref(reference_steps+1, Vector(2,0.0));
    auto costate_rhs=[&](std::span<const double> p,double t) {
        const Vector y=reference_state_at(ref,t); DenseMatrix j(2); jacobian(y,j);
        const Vector g=loss_gradient(y,t);
        return Vector{-j(0,0)*p[0]-j(1,0)*p[1]-g[0],-j(0,1)*p[0]-j(1,1)*p[1]-g[1]};
    };
    for(int n=reference_steps;n>0;--n) {
        const double t=n*ref_dt; const auto p=lambda_ref[n];
        const auto k1=costate_rhs(p,t);
        const Vector p2{p[0]-0.5*ref_dt*k1[0],p[1]-0.5*ref_dt*k1[1]};
        const auto k2=costate_rhs(p2,t-0.5*ref_dt);
        const Vector p3{p[0]-0.5*ref_dt*k2[0],p[1]-0.5*ref_dt*k2[1]};
        const auto k3=costate_rhs(p3,t-0.5*ref_dt);
        const Vector p4{p[0]-ref_dt*k3[0],p[1]-ref_dt*k3[1]};
        const auto k4=costate_rhs(p4,t-ref_dt);
        lambda_ref[n-1]={p[0]-(ref_dt/6.0)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
                         p[1]-(ref_dt/6.0)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])};
    }
    std::cout<<std::setprecision(12)
             <<"theta,dt,steps,state_error,tlm_error,costate_error,objective_error,state_order,tlm_order,costate_order,objective_order\n";
    bool passed=true;
    for(double theta:theta_values) {
        Errors previous{};
        for(int level=0;level<levels;++level) {
            const int steps=4*(1<<level); const double dt=final_time/steps;
            const auto run=solve(theta,steps,ref,lambda_ref); const auto e=run.errors;
            std::cout<<theta<<','<<dt<<','<<steps<<','<<e.state<<','<<e.tlm<<','<<e.costate<<','<<e.objective<<',';
            if(level==0) std::cout<<"NA,NA,NA,NA\n";
            else {
                const double state_order=std::log2(previous.state/e.state);
                const double tlm_order=std::log2(previous.tlm/e.tlm);
                const double costate_order=std::log2(previous.costate/e.costate);
                const double objective_order=std::log2(previous.objective/e.objective);
                std::cout<<state_order<<','<<tlm_order<<','<<costate_order<<','<<objective_order<<'\n';
                if(level==levels-1)
                    passed=passed && state_order>minimum_order && tlm_order>minimum_order
                        && costate_order>minimum_order && objective_order>minimum_order;
            }
            previous=e;
        }
    }
    return passed ? 0 : 1;
}
