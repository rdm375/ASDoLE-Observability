# Dynamical error propagation and cross-covariance analysis

This document explains the numerical machinery in `include/asdole/covariance.hpp`,
connects each routine to its mathematical model, and gives reproducible examples.
It builds on [the cross-covariance derivation](cross-covariance-theory.md),
[the error-dynamics primer](error-dynamics-control-theory.md), and the exact
[two-state demonstration](cross-coupled-error.md).

The analysis asks two related questions:

1. Given an uncertainty in the initial state, parameters, or forcing, how does
   the dynamics damp, preserve, or amplify it?
2. How do shared dynamical pathways make errors in different state variables
   statistically dependent?

The first is about propagation and gain. The second is about covariance and
correlation. They are related, but neither can replace the other.

## 1. The numerical map is the object being differentiated

Write one numerical ASDoLE step as

$$
x_{k+1}=\Psi_k(x_k).
$$

The map depends on the step size and on the stepper's `theta`; those details
are implicit in the subscript. Along a computed nominal trajectory
\(\bar{x}_k\), the discrete tangent-linear transition is

$$
M_k=D\Psi_k(\bar{x}_k),\qquad
\delta x_{k+1}\simeq M_k\delta x_k.
$$

This is the same derivative used by the repository's TLM and discrete
adjoint. For ASDoLE, the TLM must account for the state dependence of the
Jacobian frozen within each affine step, which is why
`DiscreteTrajectoryLinearization` receives the model Hessian callback. A
linear model has zero Hessian. The covariance routines do not introduce a
separate approximation for the state transition.

For a linear model, the recurrence is exact for any zero-mean initial error
with finite covariance. For a nonlinear model, it is a first-order
approximation around the nominal numerical trajectory; accuracy depends on
the perturbation staying in a region where the step map is well represented
by its derivative.

## 2. State covariance transport

Let \(e_k=x_k-\bar{x}_k\) be a zero-mean state perturbation, and define

$$
P_k=\mathbb{E}[e_ke_k^\mathsf{T}].
$$

With no new uncertainty introduced during a step, the linearized recurrence
implies

$$
P_{k+1}=M_kP_kM_k^\mathsf{T}.
$$

If an independent, zero-mean additive perturbation \(w_k\) enters at the end
of step \(k\), with covariance \(Q_k\), then

$$
e_{k+1}=M_ke_k+w_k,\qquad
P_{k+1}=M_kP_kM_k^\mathsf{T}+Q_k.
$$

The independence assumption removes the cross terms. If \(w_k\) is
correlated with \(e_k\), or with earlier forcing, those cross-covariances
must also be propagated; they are not silently assumed to be zero by the
API.

`propagate_covariance(trajectory, P0, Q)` returns one dense covariance for
each node of the stored trajectory. `Q` may be empty, or contain one
end-of-step covariance per numerical step. It obtains each \(M_k\) by
applying the existing discrete TLM to coordinate directions. This keeps
the implementation tied to the differentiated ASDoLE map, at the cost of
materializing a dense transition matrix for each step.

The output is symmetrized against roundoff. Inputs are checked for symmetry
and positive semidefiniteness with a scale-relative eigenvalue tolerance.
`diagnose_covariance` reports marginal standard deviations, the correlation
matrix, symmetry defect, and smallest eigenvalue. A correlation is returned
as NaN if either corresponding variance is zero: its denominator is then
zero and the correlation is undefined.

## 3. State, parameter, and control uncertainty

For a parameter vector \(p\) that is constant in time, linearizing the
discrete map gives

$$
\delta x_{k+1}=M_k\delta x_k+B_k\delta p+w_k,\qquad
\delta p_{k+1}=\delta p_k,
$$

where \(B_k=\partial\Psi_k/\partial p\), holding the input state fixed.
This can be written as one augmented transition:

$$
\begin{bmatrix}\delta x_{k+1}\\\delta p_{k+1}\end{bmatrix}
=
\underbrace{\begin{bmatrix}M_k&B_k\\0&I\end{bmatrix}}_{\widetilde M_k}
\begin{bmatrix}\delta x_k\\\delta p_k\end{bmatrix}
+\begin{bmatrix}w_k\\0\end{bmatrix}.
$$

If the joint state/parameter covariance is

$$
\widetilde P_k=
\begin{bmatrix}P_{xx,k}&P_{xp,k}\\P_{px,k}&P_{pp,k}\end{bmatrix},
$$

then \(\widetilde P_{k+1}=\widetilde M_k\widetilde P_k\widetilde
M_k^\mathsf{T}+\widetilde Q_k\). The off-diagonal block \(P_{xp}\) records
state/parameter cross-covariance. It can be nonzero even when the initial
state and parameter priors are independent, because the parameter affects
the state evolution.

`propagate_joint_state_parameter_covariance` implements this recurrence.
The initial joint covariance is ordered as `[state; parameter]`, and the
`parameter_step_sensitivities[k]` matrix is the `state_dimension ×
parameter_dimension` matrix \(B_k\). The existing
`parameter_tangent_step` function can be applied to parameter basis vectors
to construct \(B_k\). This is also a model for uncertainty in fixed
finite-dimensional control coefficients, such as a constant bias or a
prescribed input basis.

Independent control or model disturbances that vary from step to step can
instead be represented by \(Q_k\) when their state-equivalent effect is an
additive perturbation at the step endpoint. Correlated forcing histories
need an augmented state or an explicit temporal covariance model. Assigning
a covariance to a parameter or input is a modeling choice: deterministic
unknowns do not become random merely because a covariance routine is used.

## 4. Cross-covariance across state variables and times

For state components \(i,j\), the entry \(P_{ij}\) is their cross-covariance.
It measures shared signed variation, in the variables' physical units.
Correlation normalizes it:

$$
\rho_{ij}=\frac{P_{ij}}{\sqrt{P_{ii}P_{jj}}},\qquad -1\leq\rho_{ij}\leq1,
$$

when both variances are positive. Covariance magnitude and correlation can
move in different directions. The exact two-state example has declining
variances and a cross-covariance that eventually decays, while its
correlation rises toward one because the independent part becomes small
relative to the shared part.

Cross-time covariance is also useful. For \(i\leq j\), and with independent
step forcing after node \(i\),

$$
\operatorname{Cov}(x_i,x_j)=P_i\Phi_{j,i}^\mathsf{T},\qquad
\Phi_{j,i}=M_{j-1}\cdots M_i.
$$

`state_cross_covariance(trajectory, marginal_history, i, j)` returns this
block, with rows associated with node \(i\) and columns with node \(j\).
For the reverse node order, it returns the transpose. It uses only the
state transition between those nodes and the earlier marginal covariance;
independent future process noise is correctly uncorrelated with the earlier
state.

For linear observations \(y_i=H_ix_i+\epsilon_i\),

$$
\operatorname{Cov}(y_i,y_j)
=H_i\operatorname{Cov}(x_i,x_j)H_j^\mathsf{T}
+\operatorname{Cov}(\epsilon_i,\epsilon_j).
$$

`observation_cross_covariance` applies possibly rectangular observation
operators and accepts an optional measurement cross-covariance block. For
one node, this reduces to \(H_iP_iH_i^\mathsf{T}+R_i\). This connects state
uncertainty to correlated observation errors and generalized least squares.

### Stacking observations and generalized least squares

For observations at nodes \(i_1,\ldots,i_s\), stack the errors as

$$
z=\begin{bmatrix}H_1e_{i_1}+\epsilon_1\\\vdots\\H_se_{i_s}+\epsilon_s\end{bmatrix}.
$$

The full observation covariance \(C=\operatorname{Cov}(z)\) is block
structured. Its \((a,b)\) block is

$$
C_{ab}=H_a\operatorname{Cov}(x_{i_a},x_{i_b})H_b^\mathsf{T}+R_{ab},
$$

where \(R_{ab}=\operatorname{Cov}(\epsilon_a,\epsilon_b)\). Even when
measurement errors are independent so that \(R\) is block diagonal, shared
uncertain initial conditions and process history generally make \(C\)
non-diagonal. The errors at two times then carry partly the same dynamical
uncertainty. `stacked_observation_covariance` constructs this complete matrix
from observation samples and a full optional measurement covariance. Its
sample order defines the order of stacked rows and columns.

Suppose a residual vector is modeled by \(d=A\beta+z\), with known
positive-definite covariance \(C\). Generalized least squares (GLS) chooses

$$
\widehat\beta=\underset{\beta}{\operatorname{argmin}}\;
\frac12(d-A\beta)^\mathsf{T}C^{-1}(d-A\beta)
= (A^\mathsf{T}C^{-1}A)^{-1}A^\mathsf{T}C^{-1}d,
$$

provided \(A\) has full column rank. Under this linear model, unbiasedness,
and the stated covariance, the estimator covariance is

$$
\operatorname{Cov}(\widehat\beta)=(A^\mathsf{T}C^{-1}A)^{-1}.
$$

Ordinary least squares is the special case \(C=\sigma^2I\). It remains
unbiased for many correlated-error models, but generally is less efficient
and its usual uncertainty estimate is wrong. The implementation
`generalized_least_squares` uses Cholesky solves rather than forming
\(C^{-1}\); a singular covariance or rank-deficient design needs separate
regularization or a reduced parameterization and is rejected by this dense
routine.

`asdole_observation_gls` demonstrates the distinction. It propagates
independent uncertainty in \(x(0),y(0)\) through the coupled decay system,
observes \(y\) at 21 nodes, adds independent sensor noise, and fits an
independent constant sensor bias. It reports the fit from GLS using the full
time-correlated observation covariance and from ordinary least squares using
only the white sensor-noise covariance. The synthetic residual is assembled
from one deterministic initial-state perturbation, the known bias, and a
small deterministic sensor-noise sequence; it is an illustration, not a
Monte Carlo performance claim. The reported formal standard errors expose
how treating each observation as independent can overstate the amount of
information in a time series.

Build and generate the CSV, summary, and plot with:

```sh
cmake --build build --target asdole_observation_gls --parallel
bash scripts/run_observation_gls.sh
```

Outputs go to `results/observation-gls/`. The plot shows the state-driven and
total variance over time, correlation with the first observation, and the
GLS and ordinary least-squares bias fits.

## 5. Covariance induced by parameter sensitivities

Suppose the linearized map from a finite-dimensional uncertain input \(q\)
to a vector of outputs \(z\) is

$$
\delta z=J_q\delta q,\qquad \operatorname{Cov}(q)=\Sigma_q.
$$

Then

$$
\operatorname{Cov}(z)=J_q\Sigma_qJ_q^\mathsf{T}.
$$

`covariance_from_sensitivity` pushes a covariance through any existing
matrix-free sensitivity operator exposing `input_dimension`,
`output_dimension`, and `apply`. It can be used with
`ObservationSensitivityOperator` or
`ParameterObservationSensitivityOperator` to obtain observation covariance
due to initial-state or parameter uncertainty. It materializes the dense
input-to-output Jacobian, so it is intended for modest dimensions. It does
not include measurement noise unless that covariance is added separately.

## 6. Example A: coupled initial-state uncertainty

The executable `asdole_covariance_transport` integrates

$$
\dot{x}=-x,\qquad \dot{y}=x-y,
$$

from nominal initial state \((1,0)\). The initial errors are independent,
with standard deviations \(0.1\). For this linear system,

$$
\Phi(t)=e^{-t}\begin{bmatrix}1&0\\t&1\end{bmatrix},\qquad
P(t)=e^{-2t}
\begin{bmatrix}
\sigma_{x0}^2&t\sigma_{x0}^2\\
t\sigma_{x0}^2&t^2\sigma_{x0}^2+\sigma_{y0}^2
\end{bmatrix}.
$$

Unlike the earlier exact-only display, this executable advances the nominal
state with ASDoLE and propagates covariance using its discrete TLM. It writes
every node to CSV, including state and covariance errors against the exact
continuous solution. The runner sweeps `theta = 0.5, 0.6, 0.7, 1.0` and
`10, 20, 40, 80, 160` steps, then plots convergence and the time histories
of cross-covariance and correlation.

Build and run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target asdole_covariance_transport --parallel
./scripts/run_covariance_transport.sh
```

The script writes `results/covariance-transport/covariance_transport.csv`
and `covariance_transport.png`. To run one configuration directly, pass
optional `theta` and step-count arguments, for example:

```sh
./build/asdole_covariance_transport 0.7 80
```

The final diagnostic printed to standard error is
\(\operatorname{Cov}(x(0),y(T))\), computed via the discrete cross-time
covariance and observation operators.

## 9. Covariance regression checks

`asdole_covariance_checks` exercises the public covariance APIs in four ways:

1. It compares final state covariance against the exact coupled-system
   solution while refining the ASDoLE step, for all four standard theta
   values. The finest observed covariance order must exceed 1.8.
2. It compares propagated joint state/parameter covariance with a centered
   finite-difference Jacobian of the full numerical map.
3. It checks symmetry and positive semidefiniteness while adding a positive
   semidefinite per-step process covariance, and verifies that the added noise
   increases the final total variance.
4. It checks cross-time covariance against the full discrete TLM transition,
   verifies transpose orientation when node order is reversed, and validates
   rectangular observation projection with a measurement cross-covariance.

Build and run the focused regression directly:

```sh
cmake --build build --target asdole_covariance_checks --parallel
./build/asdole_covariance_checks
```

It is also registered with CTest as `covariance_transport_regressions`.

## 10. Persistent parameter uncertainty versus repeated forcing error

`asdole_error_source_comparison` distinguishes two sources that can produce
similar errors at one selected time but have different temporal behavior. It
uses

$$
\dot{x}=-x,\qquad \dot{y}=k x-y,
$$

with nominal \(k=1\), initial state \((1,0)\), \(\theta=0.7\), and final
time 4. The first case assigns standard deviation 0.1 to the one fixed
parameter \(k\). The second case holds \(k\) fixed and injects independent
zero-mean errors into \(x\) after each step, using

$$
Q_k=q\,\Delta t\begin{bmatrix}1&0\\0&0\end{bmatrix}.
$$

The program selects \(q\) so both cases have the same added \(y\)-variance at
the final time. This scale matching makes the time histories easy to compare;
it is not a statistical claim that the two error sources are equivalent.
Uncertainty in \(k\) affects \(y\) but not \(x\), so its contribution to
\(P_{yy}\) grows while the nominal \(x\) signal is present and then decays.
Independent forcing increments contribute to both state variances, and the
\(x\)-to-\(y\) coupling induces \(P_{xy}\). Because fresh forcing continues
to enter, its covariance accumulates toward a forced steady regime.

The executable checks these properties and is registered as the CTest
`persistent_parameter_vs_step_noise`. Generate its CSV and plot with:

```sh
cmake --build build --target asdole_error_source_comparison --parallel
./scripts/run_error_source_comparison.sh
```

The outputs are written under `results/error-source-comparison/`.

## 7. Example B: a persistent parameter error

`asdole_parameter_covariance_transport` uses

$$
\dot{x}=-x+k y,\qquad \dot{y}=-2y,
$$

with uncertain initial state and uncertain constant coupling \(k\). Its
initial joint random vector is `[x(0), y(0), k]`; the example starts with
independent prior errors. Each \(B_k\) is computed from the discrete
parameter derivative of the ASDoLE step, and the augmented covariance
recurrence produces \(\operatorname{Cov}(x,k)\) and
\(\operatorname{Cov}(y,k)\) over time.

```sh
cmake --build build --target asdole_parameter_covariance_transport --parallel
./scripts/run_parameter_covariance.sh
```

This writes a CSV and plot under `results/parameter-covariance/`. Pass a
different `theta` as the third runner argument, for example
`./scripts/run_parameter_covariance.sh ./build/asdole_parameter_covariance_transport results/parameter-covariance 1.0`.

## 8. Interpretation and limits

- Covariance describes the spread of an assumed uncertainty distribution; it
  is not a worst-case bound. Worst-case finite-time amplification is the
  separate operator-gain analysis already in `analysis.hpp`.
- The API propagates covariance of uncertainty in the model's discrete
  numerical map. It does **not** estimate ASDoLE truncation error. Treating a
  local error estimate as random process noise requires a separate model.
- In a nonlinear model, first-order covariance may miss skewness,
  multimodality, and curvature effects. An ensemble comparison is a useful
  later validation step, not a guarantee that the linearized covariance is
  accurate for arbitrary uncertainty size.
- Current propagation is dense: storing one covariance costs \(O(n^2)\), and
  each step forms a dense transition matrix using \(n\) TLM applications.
  This is appropriate for the current low-dimensional analyses. Large pipe
  networks will need low-rank factors, sparse/structured transitions, or
  matrix-free covariance actions.
- Cross-time covariance can require substantial storage if all node pairs
  are retained. The API computes a requested pair on demand instead.

## Initial-condition forgetting and information saturation

For a stable linear system, perturbations of the initial state evolve as

$$
e(t)=\Phi(t,0)e(0),\qquad P(t)=\Phi(t,0)P(0)\Phi(t,0)^\mathsf{T}.
$$

If the state-transition matrix \(\Phi(t,0)\) decays, both individual
perturbations and their absolute covariance fade. This example uses

$$
\dot{x}=-x,\qquad \dot{y}=x-y,
$$

for which

$$
\Phi(t,0)=e^{-t}\begin{bmatrix}1&0\\t&1\end{bmatrix}.
$$

Thus \(e_x(t)=e^{-t}e_x(0)\) and
\(e_y(t)=e^{-t}(e_y(0)+t e_x(0))\). The polynomial factor comes from the
coupling and repeated eigenvalue; the asymptotic exponential decay rate is
still 1. With independent initial variances \(\sigma_x^2,\sigma_y^2\),

$$
P(t)=e^{-2t}\begin{bmatrix}
\sigma_x^2&t\sigma_x^2\\
t\sigma_x^2&t^2\sigma_x^2+\sigma_y^2
\end{bmatrix}.
$$

Every covariance entry tends to zero, with asymptotic exponential rate 2.
The normalized correlation \(P_{xy}/\sqrt{P_{xx}P_{yy}}\), however, tends to
one: what remains becomes aligned even while its absolute magnitude vanishes.

`asdole_initial_condition_forgetting` computes these quantities using the
ASDoLE discrete TLM and covariance transport. It also accumulates the
quadrature-weighted initial-state information matrix for observations of
\(y\),

$$
\mathcal I(T)=\int_0^T \Phi(t,0)^\mathsf{T}H^\mathsf{T}R^{-1}H\Phi(t,0)\,dt,
\qquad H=[0\;1].
$$

Since the initial-state sensitivities decay, the information rate fades and
\(\mathcal I(T)\) approaches a finite limit. This means extending the
observation window far into the decayed regime contributes little additional
information about the initial state. A single sensor observing \(y\) still
has two independent sensitivity directions over a finite time interval, so
the accumulated information matrix becomes full rank.

Build and run the study with:

```sh
cmake --build build --target asdole_initial_condition_forgetting --parallel
bash scripts/run_initial_condition_forgetting.sh
```

The CSV records a selected perturbation, the worst-case 2-norm TLM gain,
covariance entries, local decay rates, and accumulated information eigenvalues.
The fitted rates use the late-time window and are finite-time estimates; the
analytic asymptotic rates for this model are 1 for state errors and 2 for
covariance.

## References

- R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,”
  *Journal of Basic Engineering*, 82(1), 35–45, 1960.
  [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552).
- The TLM and discrete adjoint formulas used here are developed for this
  implementation in [`asdole-stage-adjoint.md`](asdole-stage-adjoint.md).
