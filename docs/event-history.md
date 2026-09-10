# Event histories

`gam_event_history` fits shared latent marked counting processes. The Rust
API is also available as `gam::event_history`; Python exposes
`gamfit.fit_event_history`.

## Probability model

For subject `i`, mark `d`, and an at-risk indicator `Y_id`,

```text
lambda_id(t) = Y_id(t) exp(eta0_id(t) + a_d' z_i(t) - m_d(t)).
```

The latent coordinates are independent stationary unit-variance OU processes,
shared across marks through their loading vectors. Each mark has its own GAM
baseline predictor. Smooth genetic-score effects such as `s(time, by=prs)`
enter that predictor.

This is an exponential-of-linear Gaussian latent model. It is not a positive
sum of signature intensities, and supplied covariates are not joint measurement
channels. Missing covariates, genetic effects on latent dynamics, and learned
disease-triggered state transitions require additional probability-model
components; this API does not implement those components. Their mathematical
target is specified in [latent-signatures.md](latent-signatures.md).

Marks can be recurrent, once-only, or terminal. A once-only event removes its
own mark from the risk set. A terminal event ends follow-up. Covariate changes
are predictable: an event uses the covariate row in force immediately before
its time.

The recorded follow-up is the observation window. Pre-entry diagnoses establish
risk status, but do not by themselves supply a fitted latent entry-state law.
In particular, they do not justify inventing event-free recurrent exposure
before entry.

## Reference centring

Without a reference population, `m_d = |a_d|^2 / 2`. Then `exp(eta0)` is
the mean intensity under the stationary latent prior. It is not generally
the incidence among the survivors.

With `ReferenceStrata`, the continuous-time model instead uses

```text
m_d(t) = log E[exp(a_d' z(t)) | alive and still at risk for mark d at t].
```

For the specified reference covariate profile, the mean intensity among those
at risk is then `exp(eta0_d(t))`. For one first-occurrence disease without
competing death, the continuous-time identity is

```text
S(t) = exp(-integral exp(eta0(s)) ds).
```

This identity describes the continuous model. Finite numerical steps approximate
it. Reference evolution uses symmetric OU splitting and a midpoint killing
equation, evaluated on a grid that includes both reference endpoints. The
driver compares log normalisers and log risk masses on a twice-finer grid at
the **same coefficients**. It refines until the discrepancy is below
`reference_tolerance` (default `1e-4` nats), or returns an unresolved numerical
error. A discrepancy above tolerance is not a successful certificate.

The reference law is a differentiable part of the objective. Its sensitivities
propagate through evolution and interpolation into the likelihood gradient,
Hessian, and directional Hessian derivatives. There is no solve under a frozen
normaliser and no compensator argument used to discard its observed score.

The normaliser and risk masses exported with a fit are evaluated at the returned
coefficient state using that fit's reference grid. Forecasting reads those
values and uses the same interpolation. `reference_refinements` records
fixed-parameter grid discrepancies; `reference_certificate` records the
accepted final discrepancy.

The reference interval is the cohort's overall follow-up interval. A reference
forecast outside that interval is rejected. Endpoint clamping does not extend
the population law. Supporting a longer horizon requires fitting a reference law
over that horizon.

Reference profiles are positional. `reference_profiles[s]` is the covariate
row for stratum `s`; training and prediction both use that integer index.
Strings, booleans, fractional indices, empty profile requests, and out-of-range
indices are rejected. NumPy arrays are accepted without testing their truth
value. One profile without explicit strata assigns every subject to stratum
zero. Unused reference profiles do not reorder assignments.

The baseline's incidence interpretation belongs to the specified reference
profile and risk set. It does not establish calibration in another population.

## Automatic complexity and numerical inference

The residual covariance statistic is a **proposal heuristic** for an additional
atom's rate. Products of filtered residual means are not claimed to be the
exact curvature of the latent-marginal likelihood.

At the proposed rate, the loading boundary curvature is obtained by
differentiating the computed augmented-factor likelihood. It integrates the
existing latent process and includes derivatives of reference centring.
Sampled likelihood profiles along the resulting eigen-directions propose a
loading prior. Their product is an approximation used to initialise the joint
fit. Final candidate acceptance compares the joint solver's LAML criterion;
the reported accepted gain comes from that comparison.

LAML is a Laplace approximation, not exact Bayesian evidence. The rate search,
the prior choices, and numerical quadrature also affect the comparison.
Automatic selection is not a guarantee of globally optimal rank. If a proposed
model cannot be represented or fitted, selection is unresolved: computational
failure is not evidence against an additional factor.

The latent integral uses adaptive product Gauss-Hermite filtering. Subject
meshes refine time integration, separately from reference evolution. The
stationarity checks compare gradients at fixed coefficients, converting their
discrepancy to a coefficient shift through the fitted posterior covariance.
The default acceptable shift is `0.05` posterior standard deviations.

The state grid still has `G^K` points. Backward interpolation streams one
source row at a time, so it no longer allocates the `G^(2K)` all-source kernel.
This removes that quadratic memory allocation; it does not remove exponential
state growth or the backward computation's arithmetic cost. This solver is
appropriate as a small-state numerical reference, not a claim of arbitrary-rank
biobank scalability.

## Forecasting

`forecast` filters a training history; `forecast_history` takes an independent
history and covariate table. A cutoff discards later records and never invents
follow-up beyond a recorded exit. Appending records after a cutoff cannot
change the forecast at that cutoff.

For reference-centred models, `population_forecast` at a later time conditions
the reference population on being alive and free of all once-only diagnoses
at that time. Recurrent events are unobserved and integrated out. At the
reference origin it begins at the stationary law. Without reference centring,
the population forecast begins with the stationary prior at the requested start.

Conditioning on absence of *all* once-only diseases differs from conditioning
only on remaining at risk for one disease. A short-window population forecast
therefore is not a way to read that disease's marginal baseline.
`baseline_rates` evaluates `exp(eta0)` directly.

Future terminal survival and mark counts integrate the killed latent process.
Terminal counts are cumulative incidences, once-only counts are first-event
probabilities before termination, and recurrent counts are expected numbers of
events before termination.

## Python example

```python
model = gamfit.fit_event_history(
    subjects, events, covariates, ["s(time, by=prs)", "s(time)"],
    marks={"disease": "once", "death": "terminal"},
    reference_profiles=np.array([0]),
)
model.reference_refinements
model.reference_certificate
model.baseline_rates({"prs": 0.0}, times=[1.0, 2.0])
model.forecast_history(
    entry=0.0, exit=2.0, events=[],
    covariates={"prs": 0.0}, horizons=[3.0], stratum=0,
)
```

The reference interval must include the requested horizon.
Tables have columns `subjects: id, entry, exit`, `events: id, time, mark`,
and `covariates: id, start, ...`. Categorical covariates retain their labels.
All declared covariates are required by this conditional model.

## Validation and interpretation

The reference regressions compare survival with an independent continuous-time
solution obtained by inverting a frailty Laplace transform. They also compare
total derivatives with finite differences of the recomputed objective, including
the single-disease case whose exact frailty score is zero. A static recurrent
event regression checks that added-factor curvature is independent of event time.

Predictive PIT spells include censored tails. `pit_uniform_distance` uses
Kaplan-Meier over event and censored spells on the covered range. It requires
conditionally independent censoring. A training-data PIT distance is a
diagnostic, not an externally calibrated goodness-of-fit test.

Numerical agreement does not establish external calibration, a uniquely best
architecture, or superiority over another model. Those require appropriate
held-out cohorts and explicit evaluation.
