# Handoff: ASDoLE analysis to transient pipe-state finding

## Purpose and boundary

`ASDoLE-Observability` is the numerical-method and analysis sandbox. The next
repository should apply the verified machinery to a particular task:
recovering a physically consistent transient hydraulic state in a pipe
network from sparse measurements, boundary data, and operating history.

The handoff point is a tested example in which the discrete ASDoLE trajectory,
initial-state sensitivities, physical-parameter sensitivities, observation
operators, adjoints, priors, and identifiability diagnostics work together.
The Fisher–KPP examples provide that small, reproducible capstone. They do not
claim to solve pipe hydraulics.

## What this repository now provides

- The ASDoLE step and discrete TLM/adjoint for nonlinear state trajectories.
- Matrix-free state, tangent, adjoint, and parameter-to-observation actions.
- Weighted least-squares and damped Gauss–Newton machinery for small control
  vectors.
- Observation sensitivity, rank, singular-value, and parameter-identifiability
  diagnostics.
- Covariance propagation, parameter uncertainty, error-source comparison,
  transient amplification, and contraction analyses.
- A PDE example that jointly estimates diffusivity, growth rate, and two
  smooth initial-profile coefficients with an explicit prior.
- Checks against finite differences and TLM/adjoint dot identities, plus
  CTest regressions for the principal numerical behaviors.

The joint-initial-profile test is the immediate handoff evidence: all four
unknowns are recovered from synthetic measurements, the four-dimensional
data sensitivity has full rank, and the physical-parameter information is
reported after accounting for initial-profile nuisance directions.

## What belongs in the pipe-state-finding repository

The new repository should own the domain model and the estimator choices that
depend on pipe physics:

1. **State definition.** Specify pipe-node pressures, pipe-face flows, stored
   linepack or other thermodynamic variables, and any compressor, valve, or
   boundary states. Record units and admissible ranges for every component.
2. **Discrete hydraulic model.** Implement the transient mass and momentum
   equations, constitutive relations, network topology, boundary conditions,
   and event/controller models. The inverse problem must differentiate the
   actual discrete time-step map that is run in production.
3. **State-finding objective.** Define which unknowns are estimated: an initial
   hydraulic state, time-varying states, uncertain boundary inputs, model
   parameters, or a selected combination. Specify the observation and
   background-error covariances and any hard physical constraints.
4. **Scalable linear algebra.** The current reference GMRES has no
   application-specific preconditioner, and the included damped Gauss–Newton
   routine forms a dense normal matrix in the control dimension. Those are
   suitable for demonstrations and small controls, not a large pipe network
   whose entire initial state is unknown. The pipe repository must supply a
   sparse/preconditioned step solve and a scalable least-squares optimizer
   (for example, a Krylov method using matrix-free $Jv$ and $J^\mathsf{T}w$).
5. **Operational validation.** Check mass conservation, pressure/flow bounds,
   boundary consistency, sensitivity/adjoint identities, recovery on synthetic
   transients, robustness to sensor loss and noise, and performance on the
   intended network size.

## Initial-condition uncertainty

The PDE capstone represents initial-profile uncertainty with two smooth
coefficients and a Gaussian prior. This is deliberately finite dimensional.
For pipe state finding, a useful first formulation is often a discretized
initial hydraulic state $x_0$ with background mean $x_b$ and covariance

$$
\Phi(x_0,p)=
\frac12\|\mathcal H(\mathcal M(p,x_0))-d\|_{R^{-1}}^2+
\frac12\|x_0-x_b\|_{B^{-1}}^2+\Phi_p(p),
$$

where $\mathcal M$ is the complete discrete transient model, $p$
contains any estimated physical parameters, $R$ is the observation-error
covariance, and $B$ describes plausible initial-state error. The choice of
$B$ is a modeling assumption and should be documented; conservation or
network constraints may require a structured covariance or a constrained
parameterization. Report data-only identifiability separately from
prior-conditioned uncertainty.

If the model error over the assimilation window is material, add an explicit
model-error representation (weak-constraint formulation) rather than forcing
the initial state or physical parameters to absorb it. Keep that as a later
milestone after a strong-constraint transient estimator is verified.

## Closeout checklist for this repository

This repository is ready to hand off once the joint-initial-profile example
is included in CMake/CTest, its checks pass with the rest of the suite, the
README links the new example, and this document is reviewed. After that,
additional pipe-specific equations, network components, production
preconditioners, and full-state optimization should be developed in the new
pipe-state-finding repository rather than expanding this demonstration repo.

Suggested closeout commands:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
