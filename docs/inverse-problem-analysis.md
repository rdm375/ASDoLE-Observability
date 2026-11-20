# Perturbation Growth and Inverse-Problem Analysis

This note develops four related analyses for ASDoLE: transient amplification,
observability and identifiability, observation sensitivity, and inversion
robustness. They use the same discrete state map, tangent-linear model (TLM),
and discrete adjoint, but answer different questions. In particular, large
perturbation growth does not by itself imply that a parameter can be inferred
from the available observations.

## The discrete ASDoLE model

Write one ASDoLE step as

$$
y_{k+1}=\Phi_k(y_k,p),
$$

where $y_k\in\mathbb R^n$ is the state at time $t_k$, $p\in\mathbb R^q$
is a vector of unknown parameters, and $\Phi_k$ is the actual numerical step
map, including the full-step / doubled-step combination. Define its local
derivatives along a computed trajectory by

$$
A_k=D_y\Phi_k(y_k,p),\qquad B_k=D_p\Phi_k(y_k,p).
$$

For perturbations $\delta y_k$ and $\delta p$, the discrete TLM is

$$
\delta y_{k+1}=A_k\delta y_k+B_k\delta p.
$$

For fixed parameters, the state propagator from step 0 to step $m$ is

$$
P_{m,0}=A_{m-1}A_{m-2}\cdots A_0,\qquad \delta y_m=P_{m,0}\delta y_0.
$$

These are derivatives of the discrete ASDoLE trajectory, not derivatives of a
separate continuous-time approximation. This is the right map for analyzing
the numerical experiment and for making its TLM and adjoint consistent with
the forward calculation.

## Four analyses at a glance

| Analysis | Question | Mathematical object | Typical diagnostics |
|---|---|---|---|
| **Transient amplification** | Which initial perturbations grow most, and when? | State propagator $P_{m,0}$ | Maximum singular-value gain, gain versus time, most-amplified initial direction |
| **Observability and identifiability** | Can the observations distinguish the unknown initial state or parameters? | Observation sensitivity matrix | Rank, singular values, condition number, weighted information matrix |
| **Observation sensitivity** | Which state changes and parameters most affect a specified measurement? | TLM sensitivities and adjoint covectors | Sensitivity curves, adjoint histories, parameter-gradient components |
| **Inversion robustness** | Do inferred values remain reliable under noise and numerical changes? | The observation map and its inverse/regularized least-squares problem | Parameter error, residual, noise amplification, conditioning, convergence across resolutions |

The four analyses are connected: the TLM describes perturbation evolution,
observations sample that evolution, and the inverse problem tries to recover
unknowns from those samples. A well-conditioned inversion requires both
informative observations and a sufficiently stable mapping from unknowns to
those observations.

## 1. Transient amplification

For a chosen state norm $\|v\|_W=(v^TWv)^{1/2}$, where $W$ is symmetric
positive definite, the maximum amplification from step 0 to step $m$ is

$$
G_m=\max_{\delta y_0\ne0}
\frac{\|P_{m,0}\delta y_0\|_W}{\|\delta y_0\|_W}
=\left\|W^{1/2}P_{m,0}W^{-1/2}\right\|_2.
$$

Thus $G_m$ is the largest singular value of the propagator in the selected
inner product. The right singular vector gives the initial perturbation that
achieves this gain; the corresponding left singular vector gives its final
direction. The choice of $W$ matters when state components have different
units or scales. For small systems, the propagator and its singular values can
be formed directly. For large systems, one can estimate the leading singular
value using repeated TLM and adjoint actions without storing a dense
propagator.

The reusable implementation in `include/asdole/analysis.hpp` estimates this
gain without forming $P_{m,0}$. It applies the TLM forward and the discrete
adjoint backward in a power iteration on $P_{m,0}^TWP_{m,0}$. The result
contains the estimated gain, an initial maximizing direction, its final
perturbation, and convergence information. The default metric is Euclidean;
for a weighted norm, callers supply actions for $v\mapsto Wv$ and
$v\mapsto W^{-1}v$.

This is a finite-time question. Stable eigenvalues only describe asymptotic
behavior; a non-normal operator can temporarily amplify perturbations before
they decay. Singular-value analysis of the propagator is the standard way to
measure that worst-case finite-time gain.

### The current two-state example

The non-normal test uses

$$
A=\begin{bmatrix}-1&20\\0&-2\end{bmatrix},\qquad
y'=Ay.
$$

Both eigenvalues are negative. Its exact propagator is

$$
e^{At}=\begin{bmatrix}
e^{-t}&20(e^{-t}-e^{-2t})\\
0&e^{-2t}
\end{bmatrix}.
$$

The reference state $y(0)=(1,0)^T$ is simply $(e^{-t},0)^T$, so it
decays. The existing TLM perturbation starts at $\delta y(0)=(0,1)^T$:

$$
\delta y(t)=\begin{bmatrix}20(e^{-t}-e^{-2t})\\e^{-2t}\end{bmatrix}.
$$

Its first component peaks at $t=\log 2$, where it equals 5; its Euclidean
norm there is $\sqrt{25+1/16}\approx5.006$. The worst possible perturbation
direction is obtained from the largest singular value of $e^{At}$. The
implementation now estimates the discrete ASDoLE maximum gain at every output
node and reports its maximizing initial direction, alongside the selected
direction $(0,1)^T$. For this run the discrete maximum is about $5.03$ near
$t=0.685$, with maximizing initial direction approximately $(0.10,0.995)^T$.
The selected-vector and worst-case curves answer related but distinct
questions.

The adjoint history in the current example has terminal covector
$q_T=(1,0)^T$, corresponding to the terminal quantity $q_T^Ty(T)=y_1(T)$.
At an earlier time $t$,

$$
q(t)=e^{A^T(T-t)}q_T,
$$

and $q(t)^T\delta y(t)=q_T^T\delta y(T)$. The adjoint is a covector that
measures the effect of a perturbation at time $t$ on the selected terminal
quantity; it is not another forward state perturbation.

## 2. Observability and identifiability

Suppose observations are made at selected steps $k_i$:

$$
d_i=H_i y_{k_i}+\varepsilon_i,
$$

where $H_i$ is the observation operator and $\varepsilon_i$ is
measurement error. For fixed parameters, a small change in the initial state
produces

$$
\delta d_i=H_iP_{k_i,0}\delta y_0.
$$

Stacking all observation perturbations gives

$$
\delta d=\mathcal O\,\delta y_0,\qquad
\mathcal O=\begin{bmatrix}
H_1P_{k_1,0}\\
H_2P_{k_2,0}\\
\vdots
\end{bmatrix}.
$$

The matrix $\mathcal O$ is the finite-observation observability matrix for
this discrete trajectory. If it has deficient column rank, some initial-state
directions have no effect on the observations and cannot be recovered locally.
If it has full rank but a very small singular value, those directions are
formally observable but practically difficult to estimate in the presence of
noise. With observation-error covariance $R$, the weighted information
matrix is

$$
W_o=\mathcal O^T R^{-1}\mathcal O.
$$

For continuously sampled linear observations $H y(t)$, the analogous
finite-time observability Gramian is

$$
W_o(T)=\int_0^T P(t,0)^T H^T R^{-1}H P(t,0)\,dt.
$$

### Fixed-grid observation objective and first example

The initial implementation uses the actual time nodes emitted by the
trajectory. If $d_k$ is the observed sensor vector at node $t_k$, its discrete
least-squares objective is the composite trapezoidal approximation

$$
J_h=\frac12\sum_{k=0}^{N}w_k(Hy_k-d_k)^TR^{-1}(Hy_k-d_k),
$$

where $w_0=(t_1-t_0)/2$, $w_N=(t_N-t_{N-1})/2$, and
$w_k=(t_{k+1}-t_{k-1})/2$ for interior nodes. This uses the same sampling
times as the state simulation, including nonuniform times if the integrator
later supplies them. Its derivative with respect to a state node is

$$
\nabla_{y_k}J_h=w_kH^TR^{-1}(Hy_k-d_k).
$$

Those node derivatives are accumulated by the discrete adjoint through the
same ASDoLE step map. The quadrature-weighted initial-state information matrix
is the matching discrete approximation

$$
W_{o,h}=\sum_{k=0}^{N}w_kP_{k,0}^TH^TR^{-1}HP_{k,0}.
$$

The example `asdole_observation_analysis` demonstrates this with
$y_1'=-y_1+y_2$, $y_2'=-2y_2$ and a single sensor $H=[1\;0]$. Thus $H$ is
$1\times2$: one measured quantity observes a two-component state. A single
instant cannot determine both state components, but the coupled dynamics
allow the second component to influence the measured first component later.
The example forms the stacked sensitivity matrix with TLM actions, reports
the singular values of the quadrature-weighted observation map, and compares
the objective gradient from TLM accumulation, the discrete adjoint, and
centered finite differences.

The general interfaces in `analysis.hpp` expose a trajectory linearization
with forward and transpose actions, and an `ObservationSensitivityOperator`
that maps an initial perturbation to stacked sensor perturbations and pulls
observation covectors back to the initial state. Observation maps and
precision weights are supplied as actions, so sparse sensor layouts need not
be stored as dense matrices. `analyze_observability` obtains the weighted
singular values, numerical rank, and condition number from the operator's Gram
action; it does not assemble the full observation-sensitivity matrix. The
example uses that workflow for its $81\times2$ map and writes TLM sensitivity
and adjoint histories to a CSV for plotting.

The initial-state workflow is constructed from a stored trajectory and a list
of observation samples. Each sample supplies its node, rectangular sensor
operator, quadrature weight, and optional precision action:

```cpp
const asdole::ObservationSensitivityOperator sensitivity(trajectory, samples);
const auto report = asdole::analyze_observability(sensitivity, 1.0e-8);
// report.singular_values, report.numerical_rank, report.condition_number
```

This lets the caller choose a different sensor operator or weight at each
observation node without constructing a dense stacked sensitivity matrix.

Trapezoidal time weights model a time-integrated squared residual when the
observations represent a time series over the integration interval. For
isolated measurement events with no continuous-in-time interpretation, use a
sum over those events with their measurement covariance weights instead of
assigning time-quadrature weights.

These formulas concern recovery of the initial state. **Parameter
identifiability** instead asks whether observations distinguish entries of
$p$. Let $S_k=D_p y_k$. The parameter sensitivity satisfies

$$
S_0=D_p y_0,\qquad S_{k+1}=A_kS_k+B_k.
$$

For observations $g_i(y_{k_i},p)$, the total derivative of the predicted
observation with respect to the parameters is

$$
\frac{d}{dp}g_i(y_{k_i}(p),p)
=D_y g_i\,S_{k_i}+D_p g_i.
$$

Stack these total derivatives into the parameter sensitivity matrix

$$
J_p=\begin{bmatrix}
D_y g_1S_{k_1}+D_p g_1\\
D_y g_2S_{k_2}+D_p g_2\\
\vdots
\end{bmatrix}.
$$

Local parameter identifiability requires $J_p$ to have full column rank.
Its singular values reveal weakly constrained parameter combinations. The
weighted matrix $J_p^TR^{-1}J_p$ is the local information matrix for
independent Gaussian observation errors with covariance $R$. This is a
local test: it does not rule out distant parameter values that produce the
same observations.

### A coupling parameter can be unidentifiable despite TLM growth

Write the coupling as an unknown $k$:

$$
y_1'=-y_1+k y_2,\qquad y_2'=-2y_2.
$$

For initial state $(x_{10},x_{20})^T$,

$$
y_2(t)=x_{20}e^{-2t},\qquad
y_1(t)=x_{10}e^{-t}+k x_{20}(e^{-t}-e^{-2t}).
$$

In the current reference run, $x_{20}=0$, hence $\partial y_1(t)/\partial
k=0$ at every observation time. No amount of noise-free observation of that
trajectory can identify $k$. Yet the TLM initialized in the second
coordinate grows strongly because it asks what happens after a perturbation
away from that reference trajectory. To estimate $k$, choose a reference
with $x_{20}\ne0$, or otherwise excite the coupled state. If both initial
components and $k$ are unknown while only $y_1$ is observed, the data
depend on only the two combinations $x_{10}+kx_{20}$ and $kx_{20}$; an
additional independent observation or known initial condition is needed to
resolve all three unknowns.

### Parameter-identifiability combinations

The reusable `ParameterObservationSensitivityOperator` accepts callback
actions for the initial-state derivative $D_p y_0$, each step's direct
parameter derivative $D_p\Phi_k$, and optional direct parameter dependence of
the observation function. It combines those actions with the stored
trajectory's TLM and adjoint to apply the parameter-to-observation Jacobian and
its transpose without assembling the sensitivity matrix. A model supplies
matching forward and transpose callback pairs, so the parameter and
observation dimensions are not fixed by this library. The
`analyze_parameter_identifiability` workflow forms the weighted information
matrix through operator actions and reports singular values, numerical rank,
and condition number. The default numerical-rank cutoff is $10^{-8}$ times
the largest singular value. It is configurable because practical rank also
depends on parameter scaling, observation noise, and the intended application.
The callbacks represent, respectively,
$v\mapsto D_p y_0v$, $(k,y_k,h,v)\mapsto D_p\Phi_kv$, and
$(i,y_i,v)\mapsto D_p g_i v$; each transpose callback must implement the
corresponding Euclidean transpose action. Check these pairs with the dot
product identity before relying on rank diagnostics.

The example `asdole_parameter_identifiability` exercises that workflow on the
coupling model. It constructs candidate unknown vectors from initial-state
components and the coupling, supplies the direct ASDoLE step derivative for
the coupling, and analyzes each candidate unknown set. Its weighted Gram
matrix is

$$
G_p=J_p^TQJ_p,\qquad
Q=\operatorname{diag}(w_0R^{-1},\ldots,w_NR^{-1}).
$$

Its eigenvalues are the squared singular values of the weighted sensitivity
matrix $Q^{1/2}J_p$. The reported numerical rank uses a relative singular-value
threshold, so it should be interpreted with the displayed singular values
and the model's parameter scaling, not as a universal cutoff.

The four cases make the assumptions explicit:

| Unknown quantities | Initial/excitation assumption | Expected result |
|---|---|---|
| $k$ | Both initial components known and $y_2(0)\ne0$ | $k$ affects observed $y_1$ and is locally identifiable. |
| $y_1(0),k$ | $y_2(0)$ known and nonzero | The time history can distinguish the initial measured level from the coupling effect. |
| $y_1(0),y_2(0),k$ | Only $y_1(t)$ observed | Rank deficient: the continuous model reveals only two independent combinations of these three unknowns. |
| $k$ | Known $y_2(0)=0$ | No excitation reaches the coupling, so the sensitivity to $k$ vanishes. |

The example also compares each discrete sensitivity column with centered
finite differences of the corresponding initial value or parameter and checks
the forward/transpose dot-product identity. These checks exercise the callback
contract for the actual ASDoLE map used in the analysis.

## 3. Observation sensitivity and the adjoint

For a scalar observation $g(y_m,p)$, its state derivative is a covector
$q_m=\nabla_y g(y_m,p)$. Pulling it back through the discrete dynamics
gives

$$
q_k=A_k^Tq_{k+1},\qquad k=m-1,\ldots,0.
$$

Then $q_k^T\delta y_k=q_m^T\delta y_m$. A large component of $q_k$
means that a perturbation in that state direction at time $t_k$ can strongly
affect the selected observation. The TLM propagates a chosen perturbation
forward; the adjoint propagates an observation's sensitivity backward.

For the discrete least-squares objective

$$
J_h(p)=\frac12\sum_i r_i^TR_i^{-1}r_i
+\frac12(p-p_0)^T\Gamma^{-1}(p-p_0),\qquad
r_i=g_i(y_{k_i},p)-d_i,
$$

the adjoint recurrence accumulates observation contributions:

$$
\lambda_k=A_k^T\lambda_{k+1}
+\sum_{i:k_i=k}(D_y g_i)^TR_i^{-1}r_i.
$$

With terminal value zero where there is no terminal observation, the parameter
gradient is

$$
\nabla_pJ_h=\Gamma^{-1}(p-p_0)
+\sum_k B_k^T\lambda_{k+1}
+\sum_i(D_p g_i)^TR_i^{-1}r_i.
$$

The last term accounts for observations that depend directly on parameters.
This reverse calculation costs roughly one adjoint trajectory per scalar
objective, independent of the number of parameters. If there are only a few
parameters, the TLM can instead build $J_p$ column by column. The existing
stage-wise ASDoLE adjoint differentiates the actual discrete step, including
its state-dependent Jacobian; the derivation is in
[`asdole-stage-adjoint.md`](asdole-stage-adjoint.md).

In the coupling example with known $x_{20}$,

$$
\frac{\partial y_1(t)}{\partial k}
=x_{20}(e^{-t}-e^{-2t}),
$$

which is largest at $t=\log 2$. This suggests that observations near that
time are most informative about $k$ in this idealized model. With multiple
unknowns and noisy data, sensor times should be chosen using the whole weighted
sensitivity matrix, not a single derivative in isolation.

## 4. Inversion robustness

Let $\mathcal G(p)$ denote the vector of all predicted observations and
$d$ the measured data. A regularized weighted least-squares estimate is

$$
\widehat p=\operatorname*{argmin}_p
\left\{\frac12\|R^{-1/2}(\mathcal G(p)-d)\|_2^2
+\frac12\|\Gamma^{-1/2}(p-p_0)\|_2^2\right\}.
$$

Here $R$ represents observation-error covariance and $\Gamma$ encodes
prior parameter scales or regularization. Near a solution, a data perturbation
$\delta d$ produces approximately

$$
\delta p\approx
\left(J_p^TR^{-1}J_p+\Gamma^{-1}\right)^{-1}
J_p^TR^{-1}\delta d.
$$

This equation explains why a small observation residual alone is not enough:
weak singular directions in $J_p$ can permit large parameter changes with
little change in fit. With a locally linear model and Gaussian errors, the
inverse of $J_p^TR^{-1}J_p+\Gamma^{-1}$ is also the usual local covariance
approximation for the regularized estimate.

A useful robustness study should therefore vary the sources of uncertainty
separately:

1. Generate synthetic observations from a known parameter vector using a
   finer reference calculation.
2. Add noise with a stated covariance $R$, and repeat over noise
   realizations.
3. Recover parameters from several initial guesses and regularization
   strengths.
4. Repeat with different observation times or observed state components.
5. Refine the ASDoLE step size and compare recovered parameters, rather than
   only comparing state trajectories.
6. Report parameter error, weighted residual, sensitivity singular values,
   and TLM-adjoint/finite-difference gradient checks.

For a first study, use a fixed time grid while optimizing parameters. An
adaptive controller can change the accepted step sequence as parameters
change, making the discrete observation map only piecewise smooth unless the
step-selection logic is included in the derivative. Adaptive inversion should
be treated as a separate follow-up after the fixed-grid sensitivities and
gradients are verified.

## Scope of the reusable workflow

Both observability and identifiability accept rectangular observation maps,
arbitrary state and unknown-vector dimensions, nonuniform node selection, and
per-sample quadrature and precision actions. The sensitivity matrix itself is
never assembled. The current rank diagnostic does assemble the smaller
unknown-space Gram matrix, which costs $O(q^2)$ storage for $q$ unknowns and
can lose relative accuracy for extremely ill-conditioned maps because small
singular values are obtained by taking square roots of Gram eigenvalues. Very large
unknown spaces will need an iterative singular-value diagnostic using the
same forward and transpose actions without forming that Gram matrix.

The current stepper still uses dense linear solves, and the included examples
use two-state models. Applying the analysis layer to larger systems therefore
also requires a scalable ASDoLE solve backend.

## Inversion-robustness experiment

The `asdole_inversion_robustness` example fits $p=(y_1(0),k)$ while holding
$y_2(0)=0.7$ fixed. It uses analytic continuous observations as synthetic
truth, then fits them with the discrete ASDoLE model. Noiseless recovery error
therefore measures time-discretization bias. For noisy cases, the observation
standard deviation is set as a fraction of the signal RMS, and the objective
uses the corresponding inverse variance. Twelve deterministic Gaussian noise
realizations are used at each nonzero noise level.

The study compares two observation windows, $[0,4]$ and $[0,1]$, three time
steps, four noise levels, and three prior strengths. The prior is diagonal in
the scaled parameter deviations from $(0.9,0.8)$, with scales $(0.5,0.5)$.
Each nonlinear least-squares iteration obtains the gradient and
Gauss-Newton matrix from the parameter sensitivity operator; a damped
Gauss-Newton solver with an Armijo line search updates the parameters. This
isolates the first-order effect of observation design, noise, prior strength,
and time discretization on parameter recovery. The example leaves parameter
bounds and more general optimizers for later work.

The output CSV includes each recovery, residual RMS, local sensitivity rank
and condition number, iteration count, and convergence flag. The plotting
script aggregates seeded realizations into parameter RMSE curves and a
noiseless time-step bias curve. A larger residual does not necessarily mean a
more accurate parameter estimate; the local sensitivity conditioning and
regularization must be considered together.

The existing convergence studies establish discretization convergence for
state, TLM, adjoint, and objective quantities. They do not by themselves
establish observability, identifiability, or robustness of an inverse problem;
those depend on the observation design and on how unknowns affect the data.

## References

- Y.-M. Ducimetière, E. Boujo, and F. Gallaire, “Weak nonlinearity for strong
  non-normality,” *Journal of Fluid Mechanics*, vol. 947 (2022), article A37.
  [https://doi.org/10.1017/jfm.2022.664](https://doi.org/10.1017/jfm.2022.664).
