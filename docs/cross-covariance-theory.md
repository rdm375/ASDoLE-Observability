# Cross-covariance in coupled dynamical systems

This note develops the mathematics behind the
[cross-coupled error example](cross-coupled-error.md). The central idea is
that initially independent state errors can become statistically dependent
when the governing dynamics transmit one state into another.
For an introduction to the underlying error dynamics, stability, Lyapunov
weighting, and parameter/control sensitivities, see
[Error dynamics: states, parameters, controls, and stability](error-dynamics-control-theory.md).

The covariance transport and state-transition viewpoint is classical in
linear estimation theory; Kalman's original state-space treatment derives
covariance evolution for prediction and filtering problems
([Kalman, 1960](https://doi.org/10.1115/1.3662552)).

## 1. Linear perturbation dynamics

Let a nominal state satisfy a dynamical model, and let $e(t)$ denote a small
perturbation around it. For a linear model, or a first-order linearization
about a nominal trajectory,

$$
\dot e(t)=A(t)e(t).
$$

The matrix $A(t)$ is the system matrix for a linear model and the Jacobian of
the vector field along the nominal trajectory for a nonlinear model. Let
$\Phi(t,t_0)$ be the state-transition matrix:

$$
\frac{\partial \Phi(t,t_0)}{\partial t}
=A(t)\Phi(t,t_0),\qquad \Phi(t_0,t_0)=I.
$$

Then

$$
e(t)=\Phi(t,t_0)e(t_0).
$$

If the initial perturbation has zero mean and covariance
$P_0=\mathbb{E}[e(t_0)e(t_0)^\mathsf{T}]$, its propagated covariance is

$$
P(t)=\mathbb{E}[e(t)e(t)^\mathsf{T}]
=\Phi(t,t_0)P_0\Phi(t,t_0)^\mathsf{T}.
$$

This formula transports both marginal variances and cross-covariances.

## 2. Differential Lyapunov equation

For time-invariant $A$, differentiating the covariance transport formula
gives

$$
\dot P=AP+PA^\mathsf{T}.
$$

This is the homogeneous differential Lyapunov equation. If the perturbation
dynamics also include independent, zero-mean white process noise with
covariance intensity $Q_c$,

$$
\dot e=Ae+w,
$$

then the covariance equation becomes

$$
\dot P=AP+PA^\mathsf{T}+Q_c.
$$

For noise entering through a matrix $G$, the added term is
$GQ_cG^\mathsf{T}$. The process-noise term can create cross-covariance
directly when its covariance matrix has nonzero off-diagonal entries; the
dynamics can then further transport that covariance.

## 3. Exact result for the two-state example

The example uses

$$
\dot x=-x,\qquad \dot y=x-y,
\qquad
A=\begin{bmatrix}-1&0\\1&-1\end{bmatrix}.
$$

Its state-transition matrix is

$$
\Phi(t)=e^{-t}
\begin{bmatrix}1&0\\t&1\end{bmatrix}.
$$

Suppose the initial errors in $x$ and $y$ are independent, with standard
deviations $\sigma_x$ and $\sigma_y$. Then

$$
P_0=
\begin{bmatrix}\sigma_x^2&0\\0&\sigma_y^2\end{bmatrix}
$$

and covariance propagation gives

$$
P(t)=e^{-2t}
\begin{bmatrix}
\sigma_x^2&t\sigma_x^2\\
t\sigma_x^2&t^2\sigma_x^2+\sigma_y^2
\end{bmatrix}.
$$

In particular,

$$
\begin{aligned}
\operatorname{Var}(x(t))&=e^{-2t}\sigma_x^2,\\
\operatorname{Cov}(x(t),y(t))&=t e^{-2t}\sigma_x^2,\\
\operatorname{Var}(y(t))&=e^{-2t}(t^2\sigma_x^2+\sigma_y^2).
\end{aligned}
$$

The initial errors are independent, but the later states share the
contribution from the uncertain $x(0)$. That shared source creates positive
cross-covariance. The independent error in $y(0)$ contributes to
$\operatorname{Var}(y)$ but not to $\operatorname{Cov}(x,y)$.

The example sets $\sigma_x=\sigma_y=0.1$. The resulting correlation is

$$
\rho_{xy}(t)
=\frac{P_{xy}(t)}{\sqrt{P_{xx}(t)P_{yy}(t)}}
=\frac{t\sigma_x}{\sqrt{t^2\sigma_x^2+\sigma_y^2}}
=\frac{t}{\sqrt{t^2+1}}.
$$

Thus $\rho_{xy}(0)=0$, $\rho_{xy}(1)=1/\sqrt{2}\approx0.7071$, and
$\rho_{xy}(2)=2/\sqrt{5}\approx0.8944$.

### Covariance and correlation have different meanings

The cross-covariance magnitude is

$$
P_{xy}(t)=\sigma_x^2 t e^{-2t}.
$$

For $t\geq0$, it reaches its maximum at $t=1/2$ and then decays to zero.
By contrast, $\rho_{xy}(t)=t/\sqrt{t^2+1}$ increases toward $1$. The
independent initial $y$ contribution becomes small relative to the shared
contribution from $x(0)$, even though both state uncertainties are shrinking.

Correlation describes normalized alignment, not error magnitude. A high
correlation at late time does not imply large errors; inspect the variances
or standard deviations as well.

## 4. A local criterion for the initial covariance growth

For a two-state linear system with

$$
A=\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{bmatrix},
$$

the homogeneous covariance equation gives

$$
\dot P_{xy}
=(a_{11}+a_{22})P_{xy}+a_{12}P_{yy}+a_{21}P_{xx}.
$$

If the initial errors are independent, $P_{xy}(0)=0$, so

$$
\dot P_{xy}(0)
=a_{12}P_{yy}(0)+a_{21}P_{xx}(0).
$$

With cross-correlated process noise, add its cross-covariance intensity
$Q_{c,xy}$ to this initial growth rate. This gives a local sign criterion:
the direction and strength of the state couplings, the initial variances,
and any direct cross-correlated forcing determine whether cross-covariance
starts positive or negative.

For the example, $a_{21}=1$, $a_{12}=0$, and $P_{xy}(0)=0$, hence
$\dot P_{xy}(0)=P_{xx}(0)>0$. Reversing the $x$-to-$y$ coupling sign reverses
the initial sign of the induced cross-covariance.

## 5. Positive semidefiniteness and perfect correlation

Covariance matrices are symmetric positive semidefinite. For a two-state
covariance this implies

$$
P_{xx}\geq0,\qquad P_{yy}\geq0,\qquad
P_{xy}^2\leq P_{xx}P_{yy}.
$$

Whenever both variances are positive, this guarantees
$-1\leq\rho_{xy}\leq1$. Equality occurs when one centered error is an exact
linear multiple of the other.

If only $x(0)$ is uncertain and $y(0)$ is exact, both later errors are
multiples of the same scalar uncertainty. The covariance matrix is rank one,
and the correlation is $+1$ for $t>0$. Adding an independent uncertainty in
$y(0)$ raises the covariance rank and makes the correlation less than one at
finite times.

## 6. Discrete propagation with a numerical time step

For a discrete state map

$$
z_{k+1}=\Psi_k(z_k),
$$

the tangent-linear matrix along the nominal trajectory is

$$
M_k=D\Psi_k(\bar z_k).
$$

First-order covariance propagation consistent with that discrete map is

$$
P_{k+1}=M_kP_kM_k^\mathsf{T}+Q_k,
$$

where $Q_k$ is the covariance of an independent additive perturbation
introduced during step $k$. With $Q_k=0$, this is exactly the linearized
covariance transport for the numerical trajectory. For ASDoLE, $M_k$ should
be the Jacobian of the discrete ASDoLE step map, the same map differentiated
by its tangent-linear model.

This matters when studying the numerical method: propagating covariance with
the continuous equation while integrating the state with ASDoLE generally
does not give the covariance of the discrete numerical solution. The two
should use matching transition maps when numerical consistency is the goal.

## 7. What uncertainty is being propagated?

The current example propagates uncertainty in the initial state through an
exact deterministic model. It does not estimate numerical truncation error.
Those are different questions:

- **Initial-state uncertainty** describes uncertainty in the state supplied
  to the model.
- **Process uncertainty** describes uncertain forcing or unresolved model
  effects, represented here by $Q_c$ or $Q_k$.
- **Numerical error** is the difference between a numerical trajectory and
  the exact solution of the chosen model. Treating it as random process noise
  requires a separate probabilistic model; it should not be assumed
  automatically.

For nonlinear models, covariance propagation with the Jacobian is a
first-order approximation. Strong nonlinearities or non-Gaussian
uncertainties may require higher-order methods or ensembles.

## 8. Connection to observations and inverse problems

For an observation model

$$
y_k=H_k z_k+\epsilon_k,
$$

with independent measurement noise covariance $R_k$, the predicted
observation covariance at one time is

$$
\operatorname{Cov}(y_k)=H_kP_kH_k^\mathsf{T}+R_k.
$$

Across times, the propagated state uncertainty also creates cross-time
observation covariance. This can be used in covariance-weighted least squares
and generalized least squares. In parameter estimation, the observation
sensitivity matrix and observation covariance determine which parameter
combinations the data constrain. State correlation alone is not an
identifiability test; it describes uncertainty dependence, while
identifiability depends on the observation sensitivities and experimental
design.

## References

- R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,”
  *Journal of Basic Engineering*, 82(1), 35–45, 1960.
  [https://doi.org/10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
