# Initial-condition forgetting and incremental exponential stability

This note turns the idea of “forgetting the initial condition” into
quantitative stability criteria and describes what those criteria say about
state and covariance errors. The small C++ interface in
`include/asdole/contraction.hpp` checks linear Lyapunov inequalities and
weighted contraction factors of the actual discrete ASDoLE tangent map.

## 1. From forgetting to incremental stability

Let \(\varphi(t,t_0,x_0)\) denote the solution at time \(t\) that starts at
\(x_0\) at time \(t_0\). A system forgets its initial condition if solutions
that start from different initial states approach one another. A useful
uniform, quantitative definition is **incremental exponential stability**:

$$
\|\varphi(t,t_0,x_0)-\varphi(t,t_0,\widetilde x_0)\|
\le K e^{-\lambda(t-t_0)}\|x_0-\widetilde x_0\|,
\qquad t\ge t_0,
$$

for every pair of initial states in the region under study, with constants
\(K\ge1\) and \(\lambda>0\) independent of the pair and initial time.
\(\lambda\) is the guaranteed asymptotic forgetting rate. \(K\) allows
transient amplification before decay. The bound is about differences between
solutions; it does not require that either solution approach a particular
equilibrium. A driven system can therefore forget its initial state while
following a nonconstant reference trajectory.

This is stronger than asymptotic stability of one equilibrium. An equilibrium
result only compares nearby solutions with that equilibrium, while incremental
stability compares any two trajectories in the specified region.

## 2. Linear systems: exact spectral criterion and Lyapunov certificate

For the homogeneous linear error equation

$$
\dot e=Ae,
$$

incremental stability is exponential stability. In finite dimensions this
holds exactly when \(A\) is Hurwitz: every eigenvalue has negative real part.
The eigenvalues determine asymptotic stability, but they do not give a
well-conditioned transient bound for a nonnormal matrix.

A constructive sufficient certificate chooses a symmetric positive-definite
matrix \(W\) and verifies, in Loewner (quadratic-form) order,

$$
A^\mathsf{T}W+WA\preceq -2\lambda W.
$$

For the weighted norm \(\|e\|_W=(e^\mathsf{T}We)^{1/2}\), this gives

$$
\frac{d}{dt}\|e\|_W^2\le -2\lambda\|e\|_W^2,
\qquad
\|e(t)\|_W\le e^{-\lambda(t-t_0)}\|e(t_0)\|_W.
$$

To express the same guarantee in the Euclidean norm, let \(mI\preceq W\preceq
MI\), where \(m=\lambda_{\min}(W)\), \(M=\lambda_{\max}(W)\). Then

$$
\|e(t)\|_2\le \sqrt{M/m}\,e^{-\lambda(t-t_0)}\|e(t_0)\|_2.
$$

The condition number of \(W\) therefore matters: a strong weighted rate can
come with a large Euclidean transient prefactor. For Hurwitz \(A\), the
Lyapunov equation \(A^\mathsf{T}W+WA=-Q\), with any \(Q\succ0\), has a unique
solution \(W\succ0\). It certifies a positive rate, though not necessarily
the best rate or the smallest transient bound.

### Euclidean one-sided Lipschitz test

Taking \(W=I\), a convenient sufficient condition is

$$
\frac{A+A^\mathsf{T}}{2}\preceq-\lambda I.
$$

For a nonlinear vector field \(\dot x=f(x,t)\), the corresponding condition
is

$$
\frac{J(x,t)+J(x,t)^\mathsf{T}}{2}\preceq-\lambda I,
\qquad J=f_x,
$$

uniformly over a forward-invariant region containing the trajectories. By the
mean-value integral along the segment between two states, this implies
incremental contraction there. Checking the Jacobian only at sampled points
of one trajectory is useful diagnostic evidence, but is not a certificate for
the whole region or for every pair of trajectories.

For a state-dependent metric \(W(x,t)\), the differential condition becomes

$$
\dot W+J^\mathsf{T}W+WJ\preceq-2\lambda W.
$$

This extends contraction tests to systems that are not contractive in the
ordinary Euclidean metric. In practice, proving bounds on the full region can
require interval methods, symbolic inequalities, or optimization over that
region; this repository currently provides dense finite-dimensional numeric
checks, not a formal verified-arithmetic proof.

## 3. What the certificate says about forcing and parameter errors

For a perturbed error equation

$$
\dot e=A(t)e+w(t),
$$

suppose the transition matrix satisfies
\(\|\Phi(t,s)\|\le K e^{-\lambda(t-s)}\). Variation of constants yields

$$
\begin{aligned}
\|e(t)\|\le{}&K e^{-\lambda(t-t_0)}\|e(t_0)\|\\
&+K\int_{t_0}^{t}e^{-\lambda(t-s)}\|w(s)\|\,ds.
\end{aligned}
$$

The first term forgets its initial error. A continuing forcing or a fixed
parameter error generally leaves a residual offset. If \(\|w(t)\|\le b\),
the forced contribution is at most \(Kb(1-e^{-\lambda(t-t_0)})/\lambda\),
so the system is input-to-state stable with respect to that disturbance. This
separates initial-condition forgetting from robustness to persistent model
error.

For a parameter perturbation, the linearized dynamics are
\(\dot e=Ae+B_p\delta p\), where \(\delta p\) is constant. If \(A\) is
Hurwitz, the state approaches the offset \(-A^{-1}B_p\delta p\); the
parameter error itself does not disappear. In a nonlinear model the offset
and its existence depend on the local dynamics.

## 4. Consequences for covariance

For a deterministic linear transition \(e(t)=\Phi(t,t_0)e_0\) and initial
covariance \(P_0=\mathbb{E}[e_0e_0^\mathsf{T}]\),

$$
P(t)=\Phi(t,t_0)P_0\Phi(t,t_0)^\mathsf{T}.
$$

If the transition contracts in the same \(W\) norm at rate \(\lambda\),
then the weighted mean-square error obeys

$$
\mathbb{E}[e(t)^\mathsf{T}We(t)]
\le e^{-2\lambda(t-t_0)}\mathbb{E}[e_0^\mathsf{T}We_0].
$$

Thus standard deviations decay at least at rate \(\lambda\) in the weighted
norm and covariance energy at rate \(2\lambda\). In Euclidean coordinates,
the metric conversion can introduce the same condition-number prefactor.
Independent process noise changes the recurrence to
\(P_{k+1}=M_kP_kM_k^\mathsf{T}+Q_k\); persistent noise produces a nonzero
uncertainty floor even when the unforced system forgets initial errors.

For nonlinear systems this covariance law is a first-order, local
approximation around the nominal trajectory. Incremental contraction is a
stronger property: if it is proven throughout a region, it bounds the
separation of actual nonlinear trajectories, not merely infinitesimal TLM
perturbations.

## 5. The actual discrete ASDoLE map

The continuous ODE inequality does not automatically certify a numerical
integrator at a finite step size. Let the numerical update be
\(x_{k+1}=\Psi_k(x_k)\), and let
\(M_k=D\Psi_k(\bar x_k)\) be the discrete TLM matrix. The corresponding
one-step condition is

$$
M_k^\mathsf{T}W M_k\preceq\rho_k^2W.
$$

It implies \(\|e_{k+1}\|_W\le\rho_k\|e_k\|_W\) for the linearized
discrete perturbation. If \(\rho_k<1\), that step contracts. Over many
steps, the bound is the product \(\prod_k\rho_k\). For constant step size
\(h\), a uniform \(\rho<1\) gives a discrete exponential rate
\(-\log(\rho)/h\). With variable steps, report each \(\rho_k\) and the
product; do not infer a common rate without a uniform bound relative to each
step size.

`discrete_contraction_history` constructs the step matrices by applying the
existing differentiated step map to coordinate directions, then computes
the induced operator norm in a supplied \(W\) norm. This is an exact dense
linear-algebra calculation up to floating-point roundoff for each stored
step map. On a nonlinear trajectory it is a **local** TLM diagnostic. To
claim nonlinear incremental stability, the inequality must hold uniformly
for every relevant state (and, for time-varying systems, time) in the region.

## 6. Demonstration: coupled decay

The example uses

$$
\dot x=-x,\qquad \dot y=x-y,
\qquad
A=\begin{bmatrix}-1&0\\1&-1\end{bmatrix}.
$$

Its symmetric part has largest eigenvalue \(-1/2\), so the Euclidean test
certifies rate \(1/2\). A supplied positive-definite Lyapunov matrix

$$
W=\begin{bmatrix}33.75&6.25\\6.25&2.5\end{bmatrix}
$$

satisfies \((A+0.8I)^\mathsf{T}W+W(A+0.8I)=-I\), hence certifies rate
\(0.8\) in its weighted norm. The large eigenvalue ratio of this \(W\)
warns that the Euclidean bound derived from it has a sizable prefactor.
The exact solution contains \(t e^{-t}\) in the coupled component, so its
asymptotic exponential rate is 1 even though it can exhibit transient
component growth. The certificate is a guaranteed bound, not necessarily the
sharp observed rate.

The example also checks the per-step contraction factor for the actual ASDoLE
map at \(\theta=0.7\), compares the product of local bounds to the power-iteration
estimate of worst-case TLM gain in the same metric, and writes a CSV history. The nominal
trajectory is only for evaluating this linear model's discrete tangent map;
the continuous certificate applies to all states because \(A\) is constant.

Build and run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target asdole_contraction_analysis --parallel
./build/asdole_contraction_analysis
```

The summary reports the continuous Lyapunov check, the discrete one-step
factors and rate, and the estimated final weighted TLM gain against the
product bound.
CSV output is written to `results/contraction-analysis/contraction.csv`.

## 7. Nonlinear example: the polynomial two-state model

`asdole_nonlinear_contraction_analysis` reuses the polynomial model from the
nonlinear convergence study:

$$
\begin{aligned}
\dot x&=-0.7x+0.15y+0.04x^2+0.03xy,\\
\dot y&=-0.2x-1.1y+0.02xy+0.05y^2.
\end{aligned}
$$

The box \(\mathcal B=[-1,1]^2\) is forward invariant. On its four faces the
normal components point inward:

$$
\begin{array}{ll}
x=1: & f_1=-0.66+0.18y\le-0.48,\\
x=-1: & f_1=0.74+0.12y\ge0.62,\\
y=1: & f_2=-1.05-0.18x\le-0.87,\\
y=-1: & f_2=1.15-0.22x\ge0.93.
\end{array}
$$

The symmetric Jacobian \(S=(J+J^\mathsf{T})/2\) is affine in \(x,y\).
Since \(\lambda_{\max}(S)\) is a convex function of a symmetric matrix,
its maximum on this rectangle is bounded by the maximum of its values at
the four vertices. That maximum is \(-0.59\), giving the global Euclidean
incremental contraction guarantee

$$
\|x(t)-\widetilde x(t)\|_2
\le e^{-0.59(t-t_0)}\|x(t_0)-\widetilde x(t_0)\|_2
$$

for any two exact solution trajectories that start in the box. This is a
genuine regional nonlinear certificate, not a fit to observed decay rates.
Its assumptions are explicit: the box must contain the initial states, be
forward invariant, and the stated Jacobian bound must hold throughout it.

The executable integrates two nearby initial conditions with ASDoLE, compares
their difference with the TLM prediction, propagates an initial covariance,
and computes per-step Euclidean induced factors for the discrete TLM along
the nominal trajectory. The continuous-time regional result applies to the
nonlinear ODE. The discrete factors apply to the local numerical map on this
trajectory; they do not by themselves prove global contraction of the
nonlinear ASDoLE map. The covariance is also a first-order approximation.

Build and run:

```sh
cmake --build build --target asdole_nonlinear_contraction_analysis --parallel
./build/asdole_nonlinear_contraction_analysis
```

It writes `results/contraction-analysis/nonlinear_contraction.csv`. With the
included \(\theta=0.7\), \(h=0.01\) run, the continuous regional rate is
0.59, every sampled discrete TLM step contracts, and the finite initial-state
separation falls from 0.05 to about \(3.32\times10^{-6}\) by \(t=12\).
