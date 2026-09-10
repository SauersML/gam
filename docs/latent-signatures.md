# Joint latent signatures: mathematical target

This document specifies the complete model being built. The `joint` module in
`gam_event_history` implements its complete-path density, structured Laplace
posterior, and importance integration. It does not yet implement the complete
reference evolution, parameter-fitting workflow, structure search, or serving
interface specified below. The older log-linear Gaussian event model and its
numerical limits are documented in `event-history.md`.

The implemented state has independent OU innovations with stationary variance
one, a genetic mean linear in supplied predictable basis rows, and constant
learned jumps for nonterminal marks. Entry means depend on supplied context,
genetics, and recorded once-only prevalence. This conditional entry regression
does not yet provide the reference-law conditioning required below. Decoder
weights are constant per mark; observation intercepts and slopes are constant
per channel. These restrictions are explicit parts of the present density.

## State, events, and a positive decoder

Let `x_i(t)` be a shared K-dimensional state, `g_i` a vector of genetic scores,
`c_i` baseline context, and `H_i(t-)` the event history. Use

```text
dx_i = Kappa [B(t,c_i) g_i + u(t,c_i) - x_i] dt
       + L dW_i + sum_d J_d(x_i(t-),c_i) dN_id(t).

R_d(x,c) = pi_d0(c) + sum_k pi_dk(c) softplus(x_k),
pi_dk >= 0, sum_{k=0}^K pi_dk = 1, pi_d0 > 0.

lambda_id(t) = Y_id(t) exp(eta0_d(t,c_i)) R_d(x_i(t-),c_i) / M_d(t,c_i).
```

The decoder is a positive sum of signature contributions. Softplus gives
nonnegative activity with linear growth. A positive background channel keeps
the rate defined when signature activity is small. Loading weights describe
how signatures contribute to marks; genetic coefficients shape the common
trajectory. They have different roles and need separate identifiability checks.

Use positive mean-reversion rates and bounded jump functions. Over a finite
follow-up with bounded baseline predictors, rates then grow at most linearly
in the state. Exponential activity combined with positive recurrent jumps would
require additional non-explosion conditions and is not this specification.

Event jumps are predictable state transitions following an observed event;
that event's intensity uses the state before its jump. Once-only and terminal
marks retain their risk-set semantics. The default jump prior concentrates
around zero. A disease-associated transition is a statistical association;
it does not identify an intervention effect.

Fix diffusion scales or impose an equivalent state-scale constraint. Otherwise
measurement slopes, genetic effects, and decoder coordinates can exchange
scale. Permutations remain a reporting gauge. Nearly coincident signatures
must be reported with their joint uncertainty, not as distinct discoveries.

## Observation channels and missing values

Measurements are observations of the common state. Each channel declares its
support and likelihood:

* continuous laboratories: Student-t location `alpha_j(t,c) + h_j' x`,
  positive scale, and degrees of freedom greater than two;
* binary responses: probit likelihood;
* ordered survey responses: cumulative probit with ordered thresholds;
* counts: negative-binomial likelihood with an appropriate exposure offset.

These are defaults to evaluate against channel-specific residual diagnostics.
The model must not silently turn an unsupported channel into a Gaussian one.

An unobserved measurement contributes its integrated likelihood, which is one.
It is not replaced by a single imputed value. Missing genetic scores require
an explicit joint baseline distribution for those scores, conditional on the
declared reference context. Future measurements and states are integrated in
the same predictive distribution.

Measurement times need an observation contract. Exogenous scheduled visits may
be conditioned on. Informative visits require a jointly modeled visit process.
Nonattendance, an unmeasured channel at a visit, and a recorded negative answer
are different observations. Missingness assumptions cannot be identified or
validated solely from the missing values themselves.

## Reference population and entry

A reference object declares the origin, supported horizon, baseline covariate
and genetic distribution, initial state law, and ascertainment/eligibility law.
It also declares the conditioning variables indexing each normalization.

```text
M_d(t,c) = E_reference[R_d(x(t-),c) | Y_d(t)=1, c].
```

This expectation is taken under the model's complete reference evolution,
including other diseases, their state jumps, measurements when they affect
ascertainment, and mortality. With learned disease jumps, independently killing
one filter per disease is no longer that evolution: prior diseases affect
subsequent state dynamics. The reference solver must retain that information.

The conditional mean intensity among the specified risk set is
`exp(eta0_d(t,c))`. For one first-occurrence disease without death this implies
`S(t|c)=exp(-integral exp(eta0_d(s,c)) ds)`. With competing mortality, cause-specific
hazard and cumulative incidence remain different quantities.

Entry at a late age conditions this same law on survival, eligibility, and the
records actually observed before entry. A known disease-free interval is not
equivalent to an interval with no available records. Unknown pre-entry event
paths are integrated out. Neither a fresh stationary draw at a late age nor
invented event-free exposure represents that conditioning.

## One estimation objective

The objective is the integrated likelihood of events, measurements, observation
times when modeled, and entry/ascertainment, under the joint law above and its
declared parameter priors. Its derivative includes the derivative of the
reference evolution. A frozen-normalizer score is a different estimating
procedure and must not substitute for this derivative.

Training may use a structured variational approximation with local Gaussian
state factors and temporal precision blocks, plus shared parameter factors.
This avoids a Cartesian latent grid; it does not make the posterior Gaussian
or provide exact uncertainty. A temporal Gaussian precision solve has linear
storage in the number of time blocks at fixed K; dense within-state blocks
still cost quadratically in K to store and cubically in K to factor.

Retain the existing small-state quadrature implementation as an independently
testable numerical oracle where it resolves. Validate the structured solver
against that oracle and closed-form limits. Correct forecast approximations
using guided particles or importance sampling only when their diagnostics and
Monte Carlo uncertainty meet the stated accuracy requirement. Report an
unresolved calculation if correction degenerates.

The implemented importance bank draws from an equal mixture of the structured
Laplace Gaussian and the normalized Gaussian path prior conditional on observed
genetic scores and event jumps. The complete observation factors remain in the
importance numerator. Including the prior protects against a local Gaussian
proposal whose tails are too light; it does not ensure efficient sampling in a
large or multimodal problem. This is defensive mixture sampling, as described
by [Hesterberg](https://statistics.stanford.edu/technical-reports/weighted-average-importance-sampling-and-defensive-mixture-distributions).

An integration bank belongs to one immutable model specification and history.
Its nodes and normalized proposal density stay fixed during coefficient and
reference-sensitivity evaluations. Thus its returned jets differentiate the
same finite sampled objective as its value. The integral, gradient, and
curvature are Monte Carlo approximations to their population counterparts.
The bank reports an estimated log-integral standard error, effective sample
count, and largest normalized weight, and refuses evaluations outside the
requested error and effective-sample limits. Those diagnostics are not a
deterministic certificate or a bound on derivative and forecast errors.
Independent banks must assess each fitted or served quantity. Replacing a bank
inside an optimization line search would change the sampled objective and is
not permitted by this contract.

At a moved parameter state, the bank checks the sufficient tail condition
`2 Q(theta) - Q(anchor) > 0` on the Gaussian prior precisions. Since the proposal
contains half the anchor prior and the observation factors have at most
polynomial growth in the path, this establishes finite second moments of the
importance weights and polynomial state summaries. It is a sufficient condition,
not a necessary one; failure requires a new proposal. Weight diagnostics alone
cannot establish this tail property. Posterior means and selected covariance
blocks also have their own estimated standardized-error acceptance limit.

Sampling uses the block precision factors directly: a genetic Schur draw and
backward conditional state draws. It does not form a dense trajectory covariance
or a Cartesian state grid. Retaining S importance paths costs O(S(NK+G)) storage;
the API checks that additional memory budget before allocating them. A modest
state factorization alone does not guarantee that importance sampling remains
effective as the number of observations grows.

Signature capacity grows through proposed splits/additions and shrinks through
hierarchical priors. The number of provisioned coordinates is a computational
capacity, not an evidence decision. A variational bound difference is not an
exact Bayes factor. Final structure comparisons need a common probabilistic
objective and an integration-error assessment; a failed larger fit cannot count
as evidence against its extra signature. Rate-search and structural priors must
be included in any claim to marginal evidence.

## Acceptance requirements

The implementation is complete only when the following agree:

1. Analytic special cases, the evaluated likelihood, and its derivatives.
2. Reference evolution at fixed parameters under independent refinements.
3. Fitting, saved model restoration, history filtering, and forecasting.
4. Observation/entry conditioning with fully observed and missing channels.
5. Structure comparisons and the objective that their evidence names.
6. Runtime, peak memory, and forecast error at increasing K and cohort size.

Calibration is an empirical property as well as a model property. Assess it on
held-out subjects, calendar periods, and relevant populations, with censoring
and competing risks handled consistently. Numerical agreement establishes none
of those external calibration results by itself.
