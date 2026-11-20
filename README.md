# ASDoLE: affine step doubling for pipe-flow models

This C++20 starter implements the **Affine Step Doubling with Local
Extrapolation (ASDoLE)** time step described by Dupont and Myers (PSIG 2019),
with a configurable affine-theta generalization. The paper's named ASDoLE
method uses backward Euler, corresponding to `theta = 1`. The implementation
is problem-independent: a pipe discretization supplies its semidiscrete
right-hand side and Jacobian, while the stepper applies the time-integration
formula.

## Method

At the start state `Y`, evaluate `F = F(Y)` and `J = DF(Y)` once. For a step
of length `h` and affine-theta parameter `theta`, compute

```text
(I - theta h J)       deltaS  = h F
(I - theta h J/2)     deltaD1 = h F/2
(I - theta h J/2)     deltaD2 = h/2 (F + J deltaD1)
S = Y + deltaS
D = Y + deltaD1 + deltaD2
Y_next = 2 D - S
```

`S` is one full step and `D` is the result of two half steps. Their
extrapolation cancels the leading first-order error when `theta != 0.5`.
`D - S` is returned as an error indicator. The half-step matrix is factored
once and reused for both half-step solves. Both half steps use the Jacobian
evaluated at the start of the full interval, as in the paper's affine
linearization.

The paper also describes SDoLE, which evaluates the nonlinear backward-Euler
equations at each substep. This code implements ASDoLE, the affine variant
that avoids those nonlinear solves and evaluates the RHS and Jacobian once.

## Scope

The paper's pipe examples use an isothermal gas-flow PDE with a box
discretization in space and time. This project supplies the temporal
integrator, not a calibrated gas-pipeline model, equation of state, pipe
network solver, or boundary-condition treatment. A concrete pipe model must
provide the spatially discretized `F` and `DF` callbacks. Its network
coupling can then impose end pressures at the full-step and intermediate
time levels described in the paper.

The included dense LU implementation is intended for the small standalone
example and early algorithm work. Pipeline meshes lead to sparse systems;
for production-sized models, replace the dense matrix/factorization backend
with a sparse factorization or a suitable iterative solver while preserving
the same three linear systems.

## Build the example

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/asdole_decay
```

## Plot convergence

The plotting script reads the example's CSV output and produces a two-row
figure: global error versus time-step size, and observed order versus
time-step size, with one column per `theta` value.

```sh
python3 -m pip install matplotlib
./scripts/run_convergence.sh
```

The runner writes `results/convergence/convergence.csv` and
`results/convergence/convergence.png`. To plot an existing CSV directly:

```sh
python3 scripts/plot_convergence.py path/to/convergence.csv \
  --output path/to/convergence.png
```

The runner accepts an optional executable path and output directory:
`./scripts/run_convergence.sh ./build/asdole_decay ./results/my-run`.

The example integrates `y' = -0.8 y` to `t = 4` with eight successively
halved time steps. For `theta = 0.5, 0.6, 0.7, 1.0`, it reports convergence
for the theta method, the unextrapolated doubled step `D`, and the extrapolated
ASDoLE result `2D - S`. The expected asymptotic rate for the theta method is
two at `theta = 0.5` and one for the other values. The extrapolated ASDoLE
variant is expected to converge at order two for all four values. At
`theta = 0.5`, the base affine theta method is already second order, so the
paper's `2D - S` extrapolation remains second order rather than increasing
the order further.

## Discrete state-control convergence

`asdole_decay_control_convergence` treats the initial state `u = y(0)` as the
control for

```text
y' = -a y,  y(0) = u
J(u) = integral_0^T (y(t) - y_obs(t))^2 dt
y_obs(t) = y_obs(0) exp(-a t)
```

The example fixes `a = 0.8`, `T = 4`, `u = 1.25`, and `y_obs(0) = 0.8`.
It uses a composite trapezoidal rule on the numerical state nodes for `J_h`.
The tangent-linear model propagates `delta y` through the derivative `A_k` of
the ASDoLE step map. The costate is the reverse recurrence for that same
discrete objective and state map; its initial value equals the discrete
objective gradient. A separate tangent-linear gradient sum is computed and
compared with the costate value as a discrete-adjoint identity check.

The plotted errors are maximum nodal errors for the state, tangent-linear
state, and costate, plus the absolute objective error against the analytic
integral. The exact continuous solutions used for comparison are

```text
y(t) = u exp(-a t)
delta y(t) / delta u = exp(-a t)
lambda(t) = (u - y_obs(0))/a * (exp(-a t) - exp(a t - 2 a T))
J(u) = (u - y_obs(0))^2 * (1 - exp(-2 a T)) / (2 a)
```

Build and run the control convergence study with:

```sh
cmake --build build --target asdole_decay_control_convergence
./scripts/run_control_convergence.sh
```

This writes `results/control-convergence/control_convergence.csv` and
`results/control-convergence/control_convergence.png`. The figure compares
the four errors and their observed orders for `theta = 0.5, 0.6, 0.7, 1.0`.
The discrete-adjoint identity defect should remain near floating-point
roundoff. At the finest step, the measured orders for `(state, TLM, costate,
objective)` were approximately `(2.0000, 2.0000, 2.0000, 2.0000)` for
`theta = 0.5` and `(1.9866, 1.9866, 1.9936, 1.9923)` for `theta = 1.0`.

## Nonlinear stage-adjoint check

`asdole_nonlinear_adjoint_check` implements the stage-wise TLM and reverse
adjoint for a nonlinear two-state ODE. The derivation, including the Hessian
terms from the state-dependent Jacobian, is in
[`docs/asdole-stage-adjoint.md`](docs/asdole-stage-adjoint.md). Build and run
the check with:

```sh
cmake --build build --target asdole_nonlinear_adjoint_check
./build/asdole_nonlinear_adjoint_check
```

It compares the step TLM to central finite differences, the step adjoint to
both the TLM dot-product identity and finite differences, and the full
trapezoidal-objective gradient to TLM and finite differences. Pass a theta
value as an optional argument, for example `./build/asdole_nonlinear_adjoint_check 0.5`.

## Nonlinear convergence study

`asdole_nonlinear_convergence` runs the same nonlinear two-state problem while
halving the step size from `dt = 0.25` to `dt = 1/512`. For each theta it
reports maximum nodal errors in the state, the full two-column TLM, and the
continuous-time-style costate, plus the objective error. The reference state,
TLM, costate, and objective are integrated with RK4 using 131072 reference
steps; the ASDoLE objective uses composite trapezoidal quadrature on its own
state nodes, with its discrete gradient propagated by the derived stage
adjoint.

Build and reproduce the CSV and plot with:

```sh
cmake --build build --target asdole_nonlinear_convergence
./scripts/run_nonlinear_convergence.sh
```

The output is written to `results/nonlinear-convergence/`. In this nonlinear
trial, all four quantities approach second-order convergence for
`theta = 0.5, 0.6, 0.7, 1.0`. The finest-grid observed orders are in the CSV;
the state, TLM, costate, and objective rates are approximately 2 for every
theta.

## Scalar nonlinear exact-solution check

`asdole_scalar_nonlinear_convergence` provides a second, independent nonlinear
check using

```text
y' = y^2,  y(0) = 0.3,  T = 1
J = (1/2) integral_0^T (y(t) - 0.15)^2 dt
```

The exact state is `y(t) = y(0)/(1 - y(0)t)` and the exact TLM is
`dy(t)/dy(0) = 1/(1 - y(0)t)^2`. The analytic objective and continuous
costate are used as references. As in the vector nonlinear study, the
numerical objective is trapezoidal on the computed ASDoLE nodes, and its
discrete adjoint uses the stage-wise derivative. The output also reports the
TLM-adjoint identity defect, which should remain near roundoff.

Build and run it with:

```sh
cmake --build build --target asdole_scalar_nonlinear_convergence
./scripts/run_scalar_nonlinear_convergence.sh
```

This writes a CSV and plot under `results/scalar-nonlinear-convergence/`.
The measured state, TLM, costate, and objective orders approach two for all
four tested theta values.

## Nonlinear parameter-gradient check

`asdole_nonlinear_parameter_gradient_check` treats `mu` as a scalar model
parameter in `y' = mu*y^2`. It propagates the direct stage sensitivity using
the parameter derivatives `f_mu` and `J_mu`, then forms the discrete objective
gradient by summing each step's direct parameter contribution against the
downstream adjoint. The forward sensitivity and adjoint gradient are compared
with each other and with centered finite differences of the same discrete
objective, for each tested theta.

Build and run one case with:

```sh
cmake --build build --target asdole_nonlinear_parameter_gradient_check
./build/asdole_nonlinear_parameter_gradient_check 0.7
```

## Automated numerical regressions

CTest runs the nonlinear stage-adjoint and parameter-gradient finite-difference
checks for all four theta values, plus both nonlinear convergence studies. The convergence
programs return failure if a finest-level observed order falls below their
threshold or, for the scalar case, if the TLM-adjoint identity defect exceeds
its tolerance. Run them with:

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Adaptive step-size control

`include/asdole/adaptive.hpp` provides a proportional accept/reject controller
for the stepper's `D - S` estimate. It scales each component by
`atol + rtol * max(abs(y_old), abs(y_trial))`. The weighted RMS norm controls
the typical scaled component error and is less sensitive to state dimension;
the weighted infinity norm enforces the tolerance against the largest scaled
component error. Infinity norm is the default, so every scaled component
estimate must satisfy the tolerance; select RMS when average scaled error is
the intended criterion. The controller uses estimator power 2 for
`theta != 0.5` and 3 for `theta = 0.5`, reflecting the respective asymptotic
orders of `D - S`. The accepted state remains the local-extrapolation result
`2D - S`; the estimate is an error indicator for step control, not a rigorous
bound on the true error in that corrected value.

The adaptive decay example demonstrates step rejection, acceptance, and
step-size updates for all four theta values:

```sh
cmake --build build --target asdole_adaptive_decay
./build/asdole_adaptive_decay
```

Set the tolerances and select the norm from the command line:

```sh
./build/asdole_adaptive_decay --atol 1e-8 --rtol 1e-6
```

Infinity norm is the default; pass `--norm rms` for the RMS criterion. The
example runs with both supported norms as CTest regressions, and
`adaptive_error_norm_semantics` checks that RMS and infinity norms make the
expected different decision when one component is an outlier.

For a more informative controller exercise, `asdole_adaptive_transient`
integrates a decay problem with a localized high-rate pulse. It writes a CSV
row for every attempted step, including rejected attempts, and the runner
plots step size and scaled error against time:

```sh
cmake --build build --target asdole_adaptive_transient
./scripts/run_adaptive_transient.sh
```

The test requires rejections, smaller accepted steps through the pulse,
step-size recovery away from it, and a bounded final-state error.

## Linear stability of the state and sensitivities

`asdole_stability_sweep` evaluates the scalar test equation `y' = lambda*y`
over negative `z = lambda*h`. Its amplification factor is

```text
R_ASDoLE(z) = 2 R_theta(z/2)^2 - R_theta(z)
R_theta(z)  = (1 + (1-theta) z) / (1 - theta z)
```

For this linear problem, the state, TLM, and discrete adjoint have the same
amplification factor. The sweep checks all three against the formula and plots
where `abs(R_ASDoLE)` exceeds one. It also exposes that `theta = 0.5` and
`theta = 0.6` are not stable for arbitrarily stiff negative-real modes, while
`theta = 0.7` remains stable over the swept range and `theta = 1` damps the
stiff limit. Generate the CSV and plot with:

```sh
cmake --build build --target asdole_stability_sweep
./scripts/run_stability_sweep.sh
```

The results go to `results/stability/`; the sweep is also a CTest regression.

The second stability case uses the stable but non-normal matrix
`[[-1, 20], [0, -2]]`. Starting from `(1, 0)`, the state decays, while a
TLM perturbation initialized in the second coordinate develops a transient
first-coordinate component through the coupling. The TLM is the forward
evolution of an initial perturbation. The adjoint is different: it is a
covector initialized at the terminal time and propagated backward, showing
how sensitive a chosen terminal quantity is to perturbations at earlier
times. For this test, the terminal covector is `(1, 0)`. The discrete identity
`lambda(T)^T delta_y(T) = lambda(0)^T delta_y(0)` checks that the adjoint
propagation is the transpose of the TLM map. Run the case and generate both
norm and signed-component plots with:

```sh
cmake --build build --target asdole_nonnormal_stability
./scripts/run_nonnormal_stability.sh
```

This writes CSV histories, a norm plot, and a component plot to
`results/nonnormal-stability/`. The norm plot includes the worst-case
finite-time TLM gain, estimated by matrix-free power iteration using TLM and
adjoint actions; the component plot shows the selected perturbation's signed
coordinates. The case runs as the
`nonnormal_tlm_adjoint_transient_growth` CTest regression.

## Perturbation growth and inverse-problem analysis

[`docs/inverse-problem-analysis.md`](docs/inverse-problem-analysis.md)
defines transient amplification, observability and identifiability, observation
sensitivity, and inversion robustness mathematically. It connects these ideas
to the discrete ASDoLE TLM and stage-wise adjoint, works through the current
non-normal two-state example, and outlines a sequence of inverse-problem
experiments for this repository.

The first observation experiment uses a rectangular `1 x 2` sensor matrix
and a fixed-grid, trapezoidal least-squares objective. Build and run it with:

```sh
cmake --build build --target asdole_observation_analysis
./build/asdole_observation_analysis
```

It prints the quadrature-weighted observability singular values and compares
the objective gradient from TLM propagation, the discrete adjoint, and finite
differences. It also writes
`results/observation-analysis/observation_sensitivity.csv` with initial-state
sensitivity curves and terminal/objective adjoint histories. Plot those
histories with:

```sh
python3 scripts/plot_observation_sensitivity.py \
  results/observation-analysis/observation_sensitivity.csv \
  --output results/observation-analysis/observation_sensitivity.png
```

To compare parameter-identifiability scenarios, build and run:

```sh
cmake --build build --target asdole_parameter_identifiability --parallel
./build/asdole_parameter_identifiability
```

This reports the rank and singular values for estimating the coupling with a
known initial state, estimating it jointly with part of the initial state, and
estimating all three quantities together. It also shows the loss of parameter
information when the coupling state is unexcited. Both analysis examples use
the reusable matrix-free workflows in `include/asdole/analysis.hpp`:
`analyze_observability` and `analyze_parameter_identifiability`. The parameter
workflow accepts callbacks for initial-state, per-step, and direct observation
parameter derivatives, including matching transpose actions. The diagnostics
report weighted singular values, numerical rank, and condition number without
assembling the full sensitivity matrix. They do form the smaller unknown-space
Gram matrix, so very large unknown vectors will need an iterative spectral
diagnostic. The default relative rank threshold is `1e-8`; set it explicitly
for a particular scale and noise model when using the C++ analysis calls.

## Inversion robustness study

`asdole_inversion_robustness` fits the initial value $y_1(0)$ and coupling
$k$ from noisy observations of $y_1(t)$, with $y_2(0)$ known. It uses the
matrix-free parameter-to-observation sensitivities for Gauss-Newton gradient
and curvature actions, with a reusable damped Gauss-Newton solver in
`include/asdole/inverse.hpp`.

Build and run the full study with:

```sh
cmake --build build --target asdole_inversion_robustness --parallel
bash scripts/run_inversion_robustness.sh
```

The full seeded recovery sweep is also registered with CTest as
`inverse_problem_noisy_parameter_recovery`.

The run compares observations over $[0,4]$ with observations over the early
window $[0,1]$. It sweeps relative noise levels $0\%,1\%,5\%,20\%$, three
time steps ($0.2,0.1,0.05$), and diagonal-prior strengths $0,1,10$. Each noisy
case uses 12 seeded Gaussian realizations. The noiseless observations come
from the analytic continuous solution, so their parameter error measures
time-discretization bias. Results and the plot are written to
`results/inversion-robustness/`.

The first two plot panels show parameter-recovery RMSE versus observation
noise for the two sensor windows. The third shows noiseless parameter bias
versus time step. The CSV also reports residuals, local sensitivity rank and
condition number, optimizer iterations, and convergence status. The solver is
unconstrained; this example does not impose physical parameter bounds.

## Source

Dupont, T. F. and Myers, R., “Step Doubling for Pipeline Flow,” PSIG 1923,
2019.

## License

Copyright (c) 2026 Richard Douglas Myers. This project is licensed under the
GNU Affero General Public License v3.0 only; see [`LICENSE`](LICENSE).
The SPDX identifier is `AGPL-3.0-only`.
