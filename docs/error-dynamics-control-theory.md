# Error dynamics: states, parameters, controls, and stability

This note introduces the control-theory language behind the error-propagation
examples. It connects perturbations in initial states, model parameters, and
time-varying inputs to tangent-linear models, stability criteria, covariance,
and observations.

We use **dynamical error propagation and cross-covariance analysis** as the
working term for this combined study: first tracking how perturbations evolve
through the dynamics, then quantifying statistical dependence among the
resulting errors when uncertainty models are specified.

The main distinction to keep in mind is this:

- A **state error** is a difference in the current state, often caused by an
  uncertain initial condition.
- A **parameter error** is a difference in a model coefficient. A parameter
  is usually constant in time, so its error acts as a persistent source in
  the state equations.
- A **control or input error** is a difference in a time-varying forcing
  $u(t)$. It may be a pulse, a sustained bias, or a more general time history.

In inverse problems, all three may be unknown quantities being estimated.
Mathematically they enter the dynamics in different ways, so it helps to keep
their perturbations separate.

## 1. Linearizing the model gives the error dynamics

Consider a nonlinear model

$$
\dot{x}=f(x,p,u,t),\qquad y=h(x,p,u,t),
$$

where $x(t)$ is the state, $p$ is a vector of constant parameters, $u(t)$ is a
time-varying input, and $y(t)$ is an observed quantity.

Let $(\bar{x},\bar{p},\bar{u})$ be a nominal solution. Define perturbations
$\delta x=x-\bar{x}$, $\delta p=p-\bar{p}$, and
$\delta u=u-\bar{u}$. For sufficiently small perturbations, first-order
Taylor expansion gives

$$
\delta\dot{x}
=A(t)\delta x+B_p(t)\delta p+B_u(t)\delta u(t),
$$

with

$$
A=f_x(\bar{x},\bar{p},\bar{u},t),\qquad
B_p=f_p(\bar{x},\bar{p},\bar{u},t),\qquad
B_u=f_u(\bar{x},\bar{p},\bar{u},t).
$$

This is the **tangent-linear model (TLM)**. It describes the first-order
effect of small changes in state, parameters, and inputs along the nominal
trajectory. The parameter perturbation is constant when the parameter is
constant: $\delta\dot p=0$.
For a nonlinear model this is a local approximation; if perturbations become
large, the perturbed trajectory may leave the region where this linearization
is accurate.

The corresponding first-order observation perturbation is

$$
\delta y=H(t)\delta x+D_p(t)\delta p+D_u(t)\delta u(t),
$$

where $H=h_x$, $D_p=h_p$, and $D_u=h_u$, evaluated on the nominal trajectory.
An error can affect an observation directly through $D_p$ or $D_u$, as well
as indirectly through the state.

## 2. State transition and parameter sensitivity

Let $\Phi(t,s)$ be the state-transition matrix for
$\delta\dot{x}=A(t)\delta x$:

$$
\frac{\partial\Phi(t,s)}{\partial t}=A(t)\Phi(t,s),\qquad \Phi(s,s)=I.
$$

Variation of constants separates the sources of state error:

$$
\delta x(t)
=\Phi(t,t_0)\delta x(t_0)
+\int_{t_0}^{t}\Phi(t,s)B_p(s)\delta p\,ds
+\int_{t_0}^{t}\Phi(t,s)B_u(s)\delta u(s)\,ds.
$$

The three terms are, respectively, propagated initial-state error,
parameter-induced error, and input-induced error.

For a constant parameter, the sensitivity matrix
$S_p(t)=\partial x(t)/\partial p$ satisfies

$$
\dot S_p=A S_p+B_p,\qquad
S_p(t_0)=\frac{\partial x(t_0)}{\partial p}.
$$

Thus $\delta x(t)\approx S_p(t)\delta p$ for a small parameter error when
other perturbations are zero. A time-varying input instead acts through the
integral kernel $\Phi(t,s)B_u(s)$: an input error at time $s$ is propagated
from $s$ to the later time $t$.

## 3. A scalar example: decay, persistence, and equilibrium shift

Take

$$
\dot{x}=-a x+b u,\qquad a>0,
$$

and suppose the nominal constant input $\bar u$ holds the state at its
equilibrium $\bar x=b\bar u/a$. Introduce an initial-state error
$\delta x_0$ and a constant input bias $\delta u$. The exact error is

$$
\delta x(t)=e^{-at}\delta x_0
+\frac{b}{a}(1-e^{-at})\delta u.
$$

The initial-state error decays. The constant input error does not decay away:
it moves the equilibrium, leaving the persistent state bias
$b\delta u/a$.

A temporary input error behaves differently. If an input perturbation stops,
the stable dynamics subsequently damp its state effect. A periodic input
error generally produces a periodic state response. If the dynamics are
unstable instead, state and input errors can grow.

Parameters can have either transient or persistent effects, depending on how
they enter. For example, with $\dot{x}=-p x$ and $p>0$, a small error
$\delta p$ gives

$$
\delta x(t)\approx -t x_0 e^{-pt}\delta p.
$$

Here the parameter itself remains wrong, but its effect on the decaying state
eventually becomes small. In a model whose parameter shifts a stable
equilibrium, the state effect can instead persist as a bias.

## 4. What “stable,” “damped,” and “amplified” mean

For an equilibrium, **Lyapunov stable** means every sufficiently small
initial perturbation stays small for all future time. **Asymptotically
stable** means it is Lyapunov stable and the perturbation tends to zero.
**Exponentially stable** adds a rate bound of the form
$\|e(t)\|\leq K e^{-\alpha(t-t_0)}\|e(t_0)\|$ for some $K,\alpha>0$.

For a constant linear error system $\dot e=Ae$, the matrix $A$ is **Hurwitz**
when all its eigenvalues have negative real parts. This guarantees
exponential decay of every unforced state perturbation. Conversely, for a
finite-dimensional continuous-time LTI system, exponential stability implies
that $A$ is Hurwitz. It does not by itself say that every component decreases
monotonically at every instant.

For a time-varying system, the relevant object is $\Phi(t,s)$. A bound

$$
\|\Phi(t,s)\|\leq K e^{-\alpha(t-s)},\qquad K\geq1,\quad\alpha>0,
$$

expresses uniform exponential decay in the chosen norm. A value $K>1$ allows
finite-time growth before eventual decay. More generally, the maximum
finite-time gain

$$
G(T)=\max_{0\leq s\leq t\leq T}\|\Phi(t,s)\|
$$

measures worst-case amplification over the time interval and norm of
interest. This can reveal transient growth in nonnormal systems whose
eigenvalues are all stable.

With a bounded forcing error in an exponentially stable system, the state
response remains bounded. For constant matrices and a constant parameter
perturbation, if $A$ is Hurwitz then the steady state error is

$$
\delta x_\infty=-A^{-1}B_p\delta p,
$$

provided the perturbation enters as the constant forcing $B_p\delta p$.
Stability damps the transient toward that shifted equilibrium; it does not
remove the parameter bias.

More quantitatively, if
$\|\Phi(t,s)\|\leq K e^{-\alpha(t-s)}$ and $\|B_u(t)\|\leq b$, variation of
constants gives

$$
\|\delta x(t)\|
\leq K e^{-\alpha(t-t_0)}\|\delta x(t_0)\|
+\frac{Kb}{\alpha}
  \sup_{t_0\leq s\leq t}\|\delta u(s)\|.
$$

This is a basic **input-to-state stability (ISS)** estimate: the initial
error decays, while a bounded continuing input error produces a bounded
state response proportional to its size. The estimate does not say that the
response vanishes while the forcing error remains nonzero.

## 5. Lyapunov functions and the weighting matrix

Lyapunov's direct method studies stability by finding a scalar function that
is positive for nonzero errors and decreases along error trajectories. For
linear dynamics, a common choice is the quadratic function

$$
V(e)=e^\mathsf{T}We,\qquad W=W^\mathsf{T}\succ0.
$$

The matrix $W$ defines a weighted norm
$\|e\|_W=\sqrt{e^\mathsf{T}We}$. Along $\dot e=Ae$,

$$
\dot V=e^\mathsf{T}(A^\mathsf{T}W+WA)e.
$$

If

$$
A^\mathsf{T}W+WA=-Q_L,\qquad Q_L=Q_L^\mathsf{T}\succ0,
$$

then $\dot V=-e^\mathsf{T}Q_Le<0$ for every nonzero error. The notation
$Q_L\succ0$ means positive definite: $v^\mathsf{T}Q_Lv>0$ for all
$v\ne0$. The order is a matrix (Loewner) order, not entry-by-entry
comparison.

For a Hurwitz $A$, every chosen $Q_L\succ0$ gives a unique $W\succ0$ through
the integral

$$
W=\int_0^\infty e^{A^\mathsf{T}t}Q_Le^{At}\,dt.
$$

Consequently, $e_0^\mathsf{T}We_0$ is the accumulated future
$Q_L$-weighted squared error starting from $e_0$. The choice of $Q_L$
specifies which state directions matter more. A practical choice is to scale
state components by representative allowable errors $s_i$ and use
$Q_L=D^{-\mathsf{T}}D^{-1}$ for $D=\operatorname{diag}(s_i)$. This avoids
comparing unlike units as though they had equal numerical meaning.

The same equation has an observation-energy interpretation. If
$Q_L=C^\mathsf{T}C$, then $e_0^\mathsf{T}We_0$ is the total future squared
output energy $\int_0^\infty\|Ce^{At}e_0\|^2\,dt$. This $W$ is the
observability Gramian when the integral is finite.

## 6. Random uncertainty and deterministic amplification

The TLM can propagate a particular deterministic perturbation or a random
uncertainty distribution. For zero-mean initial uncertainty with covariance
$P_0$, and no process noise,

$$
P(t)=\Phi(t,t_0)P_0\Phi(t,t_0)^\mathsf{T}.
$$

For white process noise of covariance intensity $Q_c$, the covariance
satisfies the differential Lyapunov equation

$$
\dot P=AP+PA^\mathsf{T}+Q_c.
$$

Covariance answers statistical questions, such as the variance of each state
and correlation between states. A worst-case gain answers a different
question: how large can an output perturbation become for a perturbation of a
given norm? Neither quantity is automatically the numerical integration
error. The distinction between initial-state uncertainty, model/process
uncertainty, and numerical truncation error is discussed in
[the cross-covariance note](cross-covariance-theory.md).

## 7. Discrete-time propagation and ASDoLE

For a numerical one-step map

$$
x_{k+1}=\Psi_k(x_k,p,u_k),
$$

linearization gives

$$
\delta x_{k+1}=M_k\delta x_k+N_{p,k}\delta p+N_{u,k}\delta u_k,
$$

where $M_k=\partial\Psi_k/\partial x$, $N_{p,k}=\partial\Psi_k/\partial p$,
and $N_{u,k}=\partial\Psi_k/\partial u_k$ along the nominal solution. For a
constant parameter perturbation, its sensitivity follows

$$
S_{p,k+1}=M_k S_{p,k}+N_{p,k}.
$$

This recurrence is the discrete counterpart of $\dot S_p=A S_p+B_p$. For
ASDoLE, $M_k$ must be the derivative of the discrete ASDoLE step map if the
goal is to study perturbations of the computed trajectory.

If $P_k$ is the covariance of state uncertainty and an independent additive
error with covariance $Q_k$ enters at the step, then

$$
P_{k+1}=M_kP_kM_k^\mathsf{T}+Q_k.
$$

For a fixed linear step $e_{k+1}=Me_k$, the discrete Lyapunov condition is

$$
M^\mathsf{T}WM-W\prec0.
$$

It guarantees that $V_k=e_k^\mathsf{T}We_k$ decreases at every step. For a
time-varying sequence of step matrices, one may instead use matrices $W_k$
and check $M_k^\mathsf{T}W_{k+1}M_k-W_k\prec0$.

The TLM pushes perturbations forward. The discrete adjoint pulls the
derivative of a chosen scalar objective backward. The adjoint is efficient
for gradients with many controls and few objectives; it is not itself a
covariance or a measure of worst-case state growth.

## 8. Stability, controllability, observability, and identifiability

These are related questions, but they are not synonyms:

| Concept | Mathematical object | Question |
|---|---|---|
| Stability | $\Phi(t,s)$ or a Lyapunov function | Do unforced state errors decay? |
| Controllability | $(A,B)$ or a controllability Gramian | Can inputs move the state in the desired directions? |
| Observability | $(A,C)$ or an observability Gramian | Can measurements reveal the state directions? |
| Parameter identifiability | $S_y=\partial y/\partial p$ | Can the observations distinguish parameter changes? |

For the linear model $\dot{x}=Ax+Bu$, the finite-horizon controllability
Gramian is

$$
W_c(T)=\int_0^T e^{A\tau}BB^\mathsf{T}e^{A^\mathsf{T}\tau}\,d\tau.
$$

It measures how input energy can move the state over the horizon. A positive
definite Gramian means every state direction is reachable over that horizon.
The observability Gramian instead measures how much output energy is produced
by each initial-state direction. These are dual questions: controllability
concerns what inputs can do to the state; observability concerns what outputs
can reveal about it.

For a finite-dimensional, time-invariant system, the equivalent algebraic
tests are

$$
\operatorname{rank}\begin{bmatrix}B&AB&\cdots&A^{n-1}B\end{bmatrix}=n
$$

for controllability, and

$$
\operatorname{rank}\begin{bmatrix}
C\\CA\\\vdots\\CA^{n-1}
\end{bmatrix}=n
$$

for observability. Rank answers whether a direction is reachable or visible
at all. The Gramian eigenvalues add a quantitative measure: a very small
eigenvalue means that direction is technically reachable or visible but
poorly conditioned in the chosen coordinate and input/output scaling.

## 9. State amplification is not the same as observability

An observation perturbation is approximately

$$
\delta y_k=H_k\delta x_k+D_{p,k}\delta p+D_{u,k}\delta u_k.
$$

A state direction can be strongly amplified and still be nearly invisible if
the observation operator does not see it. Conversely, a modest state error
can matter greatly if it projects strongly onto a measured output.

For parameter estimation, the sensitivity matrix
$S_y=\partial y/\partial p$ describes how observations change with
parameters. Its rank and singular values reveal which parameter
combinations the data can distinguish. Stability concerns how perturbations
evolve; observability concerns what the measurements reveal; identifiability
concerns whether unknown parameters can be separated from those measurements.
These questions are related but not interchangeable.

## References

- A. M. Lyapunov, *The General Problem of the Stability of Motion* (1892;
  later English translation). [Original-work publication record](https://doi.org/10.1080/00207179208934253).
- R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,”
  *Journal of Basic Engineering*, 82(1), 35–45, 1960.
  [https://doi.org/10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
