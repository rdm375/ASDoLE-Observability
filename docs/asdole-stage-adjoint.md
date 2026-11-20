# Stage-wise TLM and adjoint for ASDoLE

This note derives the derivative of the implemented affine-theta step-doubling
map. The paper's ASDoLE is the special case `theta = 1`. Let `y` be the input
state, `f = F(y)`, `J = DF(y)`, `h` the full step, and `b = h/2`. Define the
three increments by

```text
M_h s = h f,              M_h = I - theta h J
M_b d1 = b f,             M_b = I - theta b J
M_b d2 = b (f + J d1)
Phi_h(y) = y + 2 d1 + 2 d2 - s
```

Thus `s` is the single full step, `d1,d2` are the two increments of the
doubled step, and `Phi_h = 2D - S` is the local extrapolation.

## Tangent-linear step

For a state perturbation `v`, write `df = J v` and `dJ = DJ(y)[v]`. With
`H[i](j,k) = d^2 F_i / (dy_j dy_k)`,

```text
dJ[i,j] = sum_k H[i](j,k) v[k]
```

Differentiating each implicit stage gives

```text
M_h ds = h (J v) + theta h (dJ) s
M_b dd1 = b (J v) + theta b (dJ) d1
M_b dd2 = b ((J v) + (dJ) d1 + J dd1 + theta (dJ) d2)
dPhi = v + 2 dd1 + 2 dd2 - ds
```

The same `M_h` and `M_b` factors used by the primal step can be reused for
these solves. Terms involving `dJ` are essential for nonlinear `F`; omitting
them gives the wrong derivative of the actual ASDoLE map.

## Reverse-mode step

Given an output cotangent `q`, initialize

```text
bar_y = q
bar_s = -q
bar_d1 = 2 q
bar_d2 = 2 q
bar_f = 0
bar_J = 0
```

Reverse the second half-step solve first. Solve
`M_b^T mu2 = bar_d2`, then accumulate

```text
bar_f  += b mu2
bar_J  += b mu2 d1^T + theta b mu2 d2^T
bar_d1 += b J^T mu2
```

Next solve `M_b^T mu1 = bar_d1` and `M_h^T mus = bar_s`, accumulating

```text
bar_f += b mu1 + h mus
bar_J += theta b mu1 d1^T + theta h mus s^T
```

Finally pull the `f(y)` and `J(y)` bars back to the input:

```text
bar_y += J^T bar_f + H^*[bar_J]
(H^*[bar_J])[k] = sum_(i,j) bar_J[i,j] H[i](j,k)
```

This reverse recurrence is the transpose of the stage-wise TLM above. The
implementation is in `include/asdole/differentiation.hpp` as
`tangent_step` and `adjoint_step`.

## Objective and trajectory adjoint

The nonlinear check uses a trapezoidal objective on the discrete ASDoLE nodes:

```text
J_h = h (1/2 ell(y0,t0) + sum_(k=1)^(N-1) ell(yk,tk) + 1/2 ell(yN,tN))
yk+1 = Phi_h(yk)
```

For the Lagrangian

```text
L_h = J_h + sum_(k=0)^(N-1) p_(k+1)^T (Phi_h(yk) - y_(k+1))
```

stationarity yields the discrete adjoint and initial-control gradient

```text
pN = (h/2) ell_y(yN,tN)
pk = h ell_y(yk,tk) + A_k^T p_(k+1),  k = N-1,...,1
grad_y0 J_h = (h/2) ell_y(y0,t0) + A_0^T p1
A_k = D Phi_h(yk)
```

The reported continuous-time-style nodal costate is `lambda_0 = grad_y0 J_h`
and `lambda_k = p_k - (h/2) ell_y(yk,tk)` for `k >= 1`. This removes the
half-node load that belongs to the preceding trapezoid interval; in particular
`lambda_N = 0`. The TLM objective derivative and this adjoint gradient are two
ways of differentiating the same `J_h`.

## Nonlinear finite-difference check

`examples/nonlinear_adjoint_check.cpp` uses a two-state nonlinear polynomial
vector field with nonzero Hessians. It checks the single-step TLM against
central finite differences, the adjoint against both the TLM dot-product
identity and finite differences, and the full trajectory objective gradient
against TLM and finite differences. The printed checks should pass at the
configured `theta = 0.7` and step sizes.
