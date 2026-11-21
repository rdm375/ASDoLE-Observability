# Cross-coupled error propagation

This small example shows how an error in one component of a dynamical state
can induce an error in another component, even when there is no direct
measurement or forcing error in the second component.
For the general covariance transport results and criteria, see
[Cross-covariance in coupled dynamical systems](cross-covariance-theory.md).
For the broader distinction between state, parameter, and input errors, see
[Error dynamics: states, parameters, controls, and stability](error-dynamics-control-theory.md).

Consider

$$
\dot{x}=-x,\qquad \dot{y}=x-y,
$$

with nominal initial condition $x(0)=1$, $y(0)=0$. The exact trajectory is

$$
x(t)=e^{-t},\qquad y(t)=t e^{-t}.
$$

Now introduce independent, zero-mean initial errors $\eta_x$ and $\eta_y$:
$x(0)=1+\eta_x$ and $y(0)=\eta_y$. Because the system is linear, the state
error obeys the same equations. Its exact solution is

$$
\delta x(t)=e^{-t}\eta_x,\qquad
\delta y(t)=e^{-t}(t\eta_x+\eta_y).
$$

The coupling term $x$ in the equation for $y$ transfers part of the initial
$x$ error into $y$. The factor $t$ shows that this induced contribution first
grows from zero; the exponential factor eventually damps both state errors.

The state-transition matrix is

$$
\begin{bmatrix}\delta x(t)\\\delta y(t)\end{bmatrix}
=
\underbrace{\begin{bmatrix}
e^{-t} & 0\\
t e^{-t} & e^{-t}
\end{bmatrix}}_{\Phi(t)}
\begin{bmatrix}\delta x(0)\\\delta y(0)\end{bmatrix}.
$$

Let the initial errors be independent with standard deviations
$\sigma_x=\sigma_y=0.1$, so $\Sigma_0=\operatorname{diag}(\sigma_x^2,\sigma_y^2)$.
Propagating the covariance gives

$$
\Sigma(t)=\Phi(t)\Sigma_0\Phi(t)^{\mathsf T}
=e^{-2t}
\begin{bmatrix}
\sigma_x^2 & t\sigma_x^2\\
t\sigma_x^2 & t^2\sigma_x^2+\sigma_y^2
\end{bmatrix},
\qquad
\operatorname{Cov}(\delta x,\delta y)
=t e^{-2t}\sigma_x^2.
$$

The $y$ uncertainty at the initial time contributes to $\operatorname{Var}(y)$
but not to the cross-covariance, because it is independent of the initial
$x$ error. Thus, independent initial errors develop positive correlation
through the $x$-to-$y$ coupling. With equal initial standard deviations, the
correlation at $t=1$ is $1/\sqrt{2}\approx0.7071$; at $t=0$ it is zero.
This avoids the perfect correlation of a one-error-source example.

The covariance and correlation tell different parts of the story. Here
$\operatorname{Cov}(\delta x,\delta y)$ reaches its maximum at $t=1/2$ and
then decays toward zero. The correlation keeps increasing toward $1$, because
the independent contribution from $\eta_y$ becomes small relative to the
shared contribution from $\eta_x$. Thus, high correlation at late time does
not mean large errors: both marginal standard deviations are also decaying
toward zero.

Run the example from the repository root:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target asdole_cross_coupled_error --parallel
    ./build/asdole_cross_coupled_error

The program prints the exact nominal trajectory, propagated standard
deviations, the portion of $y$ uncertainty induced by $x(0)$, and the
cross-covariance at five times between $0$ and $2$. Because it uses the
analytic solution, the table demonstrates dynamical coupling without mixing
in time-discretization error. A pipe model adds more states and constitutive
couplings; thermodynamic property propagation is deferred to a separate
project.
