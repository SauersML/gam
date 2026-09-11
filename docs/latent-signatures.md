# Joint latent signatures: mathematical target

This document specifies the complete model being built. The `joint` module in
`gam_event_history` implements its complete-path density, structured Laplace
posterior, importance integration, and differentiated reference evolution with
disease histories, independent replication, adaptive time/particle
refinement, and a cohort observation objective sharing these reference strata.
The parameter-fitting workflow, structure search, and serving
interface are still unfinished. The older log-linear Gaussian event model and its
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
ascertainment, and mortality. With learned disease jumps, a killed no-event
filter that omits other diagnoses is insufficient: prior diseases affect
subsequent state dynamics. The reference solver must retain that information.

The implemented `JointReferenceBank` retains each proposed disease history and
Gaussian state innovation. It draws missing genetic scores from their Gaussian
law conditional on the profile's observed scores. OU half steps surround a
competing-event update, and a nonterminal event applies its learned jump before
the next state step. Each required risk set has a conditional particle
population. Mortality and that population's focal once-only disease enter
survival weights rather than randomly deleting particles. Other diagnoses are
still simulated, retain their once-only status, and apply their state jumps.
Terminal/recurrent marks share the alive population; an all-once model does not
allocate an unused alive population. This representation is linear in the
number of risk sets, not a Cartesian product of all diagnosis combinations.

For risk set H, the conditional reference equation is a state/event evolution
with killing rate `kappa_H`, minus its conditional mean killing rate to preserve
unit conditional mass. The survival mass is retained separately. The numerical
step normalizes the non-killing transition (whose mass is one), applies its
survival factors, and saves their normalizing constant. This reduces sampling
noise in the risk mass while preserving disease histories relevant to future
rates. It does not make a finite time step exact.

Baseline basis rows are interpolated within each reference interval, while the
supplied drive basis holds constant over that interval. Refinement preserves
these declared functions. A reference interval without a representable interior
midpoint is rejected. Midpoint risk masses interpolate endpoint log masses;
they are not mislabeled interval-start masses.

At its anchor parameters the event proposal follows the particle population's
rates. Later evaluations keep the proposed histories and proposal probabilities
fixed and differentiate the target/proposal weights, the genetic state drive,
the jumps, and the risk-set moments. Freezing the proposal is an integration
device; it does not freeze the normalizer's score. `normalized_log_marginal` and
`normalized_posterior` require a `ResolvedReference`, compute the reference and
observation likelihood at the same coefficient state, and return the reference
object actually used. Its
coefficient vector and derivative channels travel with its moments. Requests
outside the reference origin and horizon are rejected.

`resolve_reference` compares three ensembles at fixed coefficients: coarse
time, half-size time intervals at the same particle count, and the fine grid
with twice the particles. Each ensemble contains independent reference
populations. Pooling averages their risk masses and risk-weighted activities,
then takes the conditional-moment ratio. It does not average conditional
moments without their risk weights. Entire populations, rather than interacting
particles within them, provide the replicate uncertainty estimate.

The acceptance estimate combines observed time and particle discrepancies with
a configured multiple of the corresponding Monte Carlo standard errors. The
controller refines time or particles until both moment and risk-mass estimates
meet their tolerances; excessive event-step hazards trigger time refinement.
Round, particle, and total retained-bank memory limits produce explicit failures.
All three ensembles remain fixed during later coefficient and jet evaluations,
which recheck the diagnostics. A failed check requires resolution outside the
objective evaluation, never silent replacement of the samples during a line
search.

These are estimated numerical-error controls, not deterministic certificates.
A finite event step admits at most one simulated non-killing event, so midpoint
rate normalization does not establish the continuous-time survival identity.
Two-level differences can underestimate discretization error, and a finite
replicate sample can miss rare behavior. The standard-error multiplier has no
claimed simultaneous confidence coverage. Value diagnostics do not certify
derivative error or external calibration. Broader validation and performance
assessment remain necessary before release.

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

`JointLikelihood::cohort_integration` binds independently sampled subject banks
to resolved reference populations by positional stratum index. Each coefficient
evaluation evolves each reference once, then uses that same curve and its
sensitivities in every assigned subject. The result retains the coefficient
state, stratum mapping, reference curves and reports, and individual integrals.
It is a likelihood evaluation, not a converged fit or an evidence estimate.
Pairwise summation combines the subject likelihoods and their derivative
channels. Its integration tolerance applies to the aggregate estimated
log-likelihood standard error conditional on the reference curves. Shared
reference uncertainty is not independent between subjects and is not included
in that conditional standard error. The stronger `resolved_score` checkpoint
below assesses it at the cohort level; final fitting and structure comparisons
must use those checks as well as resolve their higher derivatives and subject
time discretization.

The cohort `posterior` method returns integrated state/genetic means and
selected covariances at the supplied global coefficients. It does not integrate
uncertainty in those coefficients. A Laplace mode is an internal proposal
construction point, not the default estimate of an individual's latent state.

The fitting implementation must follow `SPEC.md`: REML/LAML must learn
function-level penalty strengths; global posterior means must be the default
reporting target; and a fit object requires converged optimization. General
outer optimization belongs in the existing `opt` dependency. A quadratic
coefficient representation of a function penalty is permitted only with its
equivalence established: for a declared function `f = B beta`, measure `W`,
and linear function operator `L`,

```text
||L f||_W^2 = beta' (L B)' W (L B) beta.
```

This identity does not justify an arbitrary identity ridge on nonlinear
decoder logits, rate coordinates, or an intercept. Nonlinear function priors
also need their normalization and coordinate Jacobians in any evidence
calculation; substituting a Gaussian penalty determinant would change the
objective. The function priors, fixed-bank coefficient integral and assessed
interior strength optimization below are implemented, together with an
automatic prior-based starting proposal. Posterior proposal adaptation,
refinement and null-boundary inference are unfinished.

### Decoder function prior

At fixed time and reference stratum write the final intensity as
`F(x) = c [pi_0 + sum_k pi_k softplus(x_k)]`, where `c` includes the
baseline and reference normalizer. Its lower asymptote is `L = c pi_0`,
and its slope as axis `k` tends to positive infinity is `s_k = c pi_k`.
Thus `b(F) = L / (L + sum_k s_k) = pi_0` is determined by the final
function, independently of `c`. The penalty `J(F) = -log b(F)` shrinks
latent disease dependence without penalizing the baseline rate.

With simplex volume as the base measure, the normalized prior is

```text
p(pi | lambda) = C_K(lambda) pi_0^lambda,
C_K(lambda) = product_{j=1}^K (lambda + j), lambda > 0.
```

This is `Dirichlet(1+lambda, 1, ..., 1)`. In decoder coordinates
`w_k = log(pi_k/pi_0)`, the simplex Jacobian is `pi_0 product_k pi_k`, so
the log density used in coefficient inference is

```text
log C_K(lambda) + sum_k w_k
    - (lambda + K + 1) log(1 + sum_k exp(w_k)).
```

`JointLikelihood::decoder_prior` includes this normalizer and Jacobian,
analytic coefficient/hyperparameter derivatives, and a Hessian-vector
product with linear work and storage in K. Its implementation retains
curvature even when a simplex probability rounds to one. The exact prior
means are `(1+lambda)/(K+lambda+1)` for the background and
`1/(K+lambda+1)` for each signature. They are prior means, not fitted
posterior estimates. The no-latent-effect function is recovered as
`lambda` tends to infinity; at K=0 the prior is a point mass with no
strength coordinate.

`JointCohortIntegration::score_with_function_priors` combines this density
and the function priors below with the shared-reference observation
integral and its total coefficient score. This is an integrand for coefficient
inference. It does not integrate global coefficients, learn strengths, or
produce a fit. Maximizing its joint density would not be REML/LAML.
A Laplace approximation must also be checked in the strong
shrinkage limit: exact prior normalization alone does not make an
approximation to its coefficient integral exact.

### Gaussian function measures

`JointLikelihood::function_priors` freezes function measures from the supplied
cohort design. Subjects receive equal weight; time within each subject receives
normalized exposure measure. The genetic measure is the joint Gaussian law in
the model specification, including its nonzero mean and covariance. The current
Gaussian penalties are:

* variance of each baseline log-rate surface, excluding its constant level;
* squared energy of the common genetic-drive mean function;
* squared energy of the entry mean with no prevalent diagnosis, plus a separate
  squared function contrast for each once-only prevalence indicator;
* squared magnitude of each constant nonterminal disease-jump function;
* squared measurement effect relative to its intercept, integrated over a
  standard Gaussian state. For Student-t channels this effect is divided by
  the channel's residual scale.

The entry contrasts do not invent pre-entry exposure or a prevalence
distribution. They regularize the declared entry-mean regression; a properly
selected entry law remains separate work. The baseline basis must carry a unit
constant in column zero. Its remaining columns are centered under the function
measure. An aliased function basis is rejected for removal before fitting;
the code does not add a numerical ridge to make it identifiable.

Each function norm has a root `R` satisfying `||f||^2 = ||R beta||^2`.
For rank q and strength `lambda = exp(rho)`, the coordinate density includes

```text
(q/2) rho + (1/2) log det(R'R) - (q/2) log(2 pi)
    - (lambda/2) ||R beta||^2.
```

For genetics, with precision `L L'`, the root implements
`E[(a+b'g)^2] = (a+b'mu)^2 + ||L^{-1}b||^2` directly. This avoids subtracting
large genetic means to recover a small variance. Design and genetic roots
combine by a Kronecker product and are shared across signatures.

The Student-t measurement-effect prior uses effective log precision
`rho - 2 log sigma`; its normalization therefore contributes `-K log sigma`.
The coefficient score, Hessian products, and coefficient/strength cross
derivatives include that scale dependence. Omitting it would change both the
likelihood integrand and subsequent strength learning. Precision products are
scaled before multiplication so `exp(rho)` need not itself be representable.

`FunctionPriorEvaluation` supplies analytic coefficient gradients,
coefficient Hessian products, strength derivatives, and mixed products without
a full coefficients-by-strengths matrix. Roots and evaluation workspace are
bounded before construction. These are normalized densities on their penalized
blocks. GAM operator-specific smoothness penalties and the automatic
coefficient-integration/boundary driver still need completion.

### Physical rates and observation measures

The constant baseline level has a separate positive-rate prior. Let `T` be
the mean observed follow-up span and let `bar eta_d` be the mean baseline
log rate under the same equal-subject, normalized-exposure measure. The
dimensionless function `v_d = T exp(bar eta_d)` has an exponential law with
one shared strength across marks. Its density in baseline coordinates includes
the Jacobian `v_d`. Together with the centered variation prior this is a
normalized joint density. At fixed strength, an intercept-only mark with zero
events has a positive posterior mean; the Poisson/Gamma limit is checked by
independent integration. Learning that strength, including its boundary when
all marks are empty, is still a fitting responsibility.

The dynamics prior is exponential on `T r`, the temporal variation functional
`2 T integral_0^infinity [d exp(-r t)/dt]^2 dt`, with one strength shared by
the signatures. Measurement priors use physical scale and variance functions:

* Student-t residual precision `sigma^-2` has a Gamma shape-three law.
  This is the smallest integer shape with a finite prior fourth moment of
  `sigma`.
* Student-t variance inflation `2/(nu-2)` and negative-binomial excess
  variance `(Var(Y)-mu)/mu^2 = 1/size` have Gamma shape-five laws. This is
  the smallest integer shape with four finite inverse-functional moments.
* The negative-binomial baseline mean has an exponential law.
* Binary and ordinal baseline category probabilities have a uniform
  Dirichlet law. In the probit intercept/threshold chart the density includes
  the normal-CDF and positive-gap Jacobians; it has no strength coordinate.

Each Gamma/exponential strength is represented by its logarithm. Normalizers,
chart Jacobians, analytic coefficient/strength derivatives and Hessian
products are included. These choices establish normalized function priors,
not calibration or finite posterior moments for every possible dataset.
Student-t location intercepts retain a flat nuisance measure; their
integrability must be established during coefficient inference. The
static-process, Gaussian-observation, Poisson-count and zero-effect limits
still need explicit handling in the eventual strength optimizer.

### Fixed-bank coefficient evidence

`JointCohortIntegration::guided_coefficient_proposal` first optimizes the
normalized cohort log likelihood plus the declared function log priors at
fixed strengths, using the total analytic score through `opt::Bfgs`. Subject
draws and reference banks remain fixed throughout that search. A recomputed
final gradient must meet the requested stationarity tolerance; a stalled or
failed solve does not return a proposal.

The proposal is an equal mixture of the normalized prior proposal and a
Gaussian centered at that stationary point. Its covariance is the BFGS
inverse search metric retained by `opt::Bfgs::run_with_metric`, checked by
Cholesky factorization. This matrix is an approximation used for sampling;
it is not claimed to be the observed Hessian, exact evidence curvature, or
posterior covariance. The mixture density includes both components regardless
of which component generated a draw. Allocation budgets include the dense
optimizer matrices, prior proposal workspace, and retained draws. Numerical
draw failures are reported rather than silently resampled.

Fresh coefficient draws from the completed proposal enter the resolved
coefficient integral below. Fitting a proposal does not learn its strengths
or establish posterior coverage, tail integrability, reference accuracy, or
global optimality. Inference still averages final functions using the
coefficient evidence weights. The Poisson–Gamma regression checks that the
returned coefficient mean differs from the proposal mode and agrees with
the analytic posterior mean, and checks the full mixture density and evidence
against independent formulas.

`JointCohortIntegration::coefficient_integral` evaluates the cohort at supplied
independent draws from a normalized coefficient proposal. Every draw first
passes the whole-cohort reference/value/score resolution assessment. Its
likelihood is then cached, so a strength evaluation does not rerun subjects
or reference populations. For draw `theta_s` from density `q`, it computes

```text
Z_hat(rho) = (1/S) sum_s exp(ell(theta_s) + log p(theta_s|rho) - log q(theta_s)).
g(rho) = sum_s w_s partial_rho log p(theta_s|rho).
H(rho) = sum_s w_s partial_rho^2 log p(theta_s|rho)
         + Cov_w(partial_rho log p(theta_s|rho)).
```

These are analytic derivatives of that same sampled integral. The covariance
term includes cross-strength curvature even when every conditional prior has
diagonal strength curvature. Hessian-vector products require `O(S H)` work
and storage for `H` strengths, without a dense `H` by `H` matrix. Coefficient
means and marginal variances use the same normalized weights. Predictions
must average their final functions over these weights; substituting a
function evaluated at the mean coefficient generally changes the answer.

The result reports conditional Monte Carlo errors for log evidence, means,
strength gradients and Hessian products. Cached inner likelihood errors are
correlated across coefficient draws and are reported separately as the
largest per-draw resolution estimate, without a spurious `1/sqrt(S)` reduction.
Neither this diagnostic nor a large empirical ESS proves proposal coverage,
finite importance variance or simultaneous error control. The proposal's
normalization, independence and chart Jacobians are a caller contract; MCMC
draws cannot be substituted while retaining these standard-error formulas.

This is a coefficient-integration component, not a fit or a finished REML
driver. Posterior proposal adaptation/refinement, null boundaries and
integrated serving remain unfinished.
Tests compare the Poisson/Gamma integral and posterior means with closed
forms, and the complete sampled gradient/Hessian with direct differentiation.

### Normalized starting coefficient proposals

`JointCohortIntegration::coefficient_proposal` constructs a
`PriorCoefficientProposal` from the cohort's function priors and measurements.
Its independent draws carry their complete normalized proposal densities and
can be supplied to `coefficient_integral`. Caller-created proposal draws are
no longer the only way to enter coefficient integration.

The proposal samples Dirichlet decoder weights, Gaussian function coordinates,
positive rate levels, temporal variation and measurement shapes under their
declared laws. Gaussian roots are inverted once with pivoted QR and shared
across repeated function blocks; the draw transform inverts the complete root,
including the nonzero genetic mean and covariance. Residual scales are sampled
before the measurement slopes conditional on them. Baseline levels are sampled
after the centered variation coefficients, so their geometric-rate prior uses
the same function measure as its density. Category probabilities are sampled
on their simplex and converted to probit coordinates using the smaller tail.
Structural Gamma draws are transformed in the log domain.

Student-t location intercepts retain their flat fitted-prior measure. For
sampling only, each gets a proper Cauchy law centered on that channel's observed
mean, with conditional scale `sigma/sqrt(number observed)`. Its normalizer is
included in the proposal density and removed by the importance ratio; this
does not introduce a location penalty. A completely unobserved Student-t
channel is rejected because its flat location integral is undefined. Missing
measurements for individual subjects continue to contribute likelihood one.

Root workspace and draw storage are checked before allocation. An
unrepresentable draw raises an error instead of being discarded and replaced,
which would change the proposal's law. Tests check Gaussian function-space
moments, normalized prior-score identities, category probabilities, the
conditional Cauchy transform, reproducibility and chart/error behavior.

This is a starting or defensive proposal, not a claim that prior sampling
adequately covers a concentrated posterior. The guided proposal adds a
data-informed component; independent refinement is still required. The latent
path proposal uses a Student/Gaussian mixture to retain finite variance as
coefficients move, but can still fail its finite-sample accuracy checks.
An automatic fitting driver must resolve those errors and coefficient
coverage rather than treating proposal construction as finished inference.

### Assessed interior strength optimization

`JointCoefficientIntegral::optimize_strengths` maximizes the integrated
evidence with the workspace's pinned `opt` BFGS implementation. It supplies
analytic first derivatives, uses no strength bounds, and disables the generic
relative-stall exit. The final evidence score is recomputed and must satisfy
the absolute tolerance in dimensionless log-strength coordinates. A failed
solve never yields a strength optimum.

Acceptance also requires a separate independent coefficient bank from the
same cohort and the same frozen function measure. Those validation draws
must not have selected the fitting bank or its optimum. Cohort identities are
retained by both banks; unrelated datasets or duplicated banks are rejected.
The negative evidence Hessian must admit a Cholesky factor without clipping
or jitter. That factor defines the units for comparing strength scores and
curvatures between banks. The report includes the log-evidence discrepancy,
whitened score/curvature errors, and the largest posterior-mean error in
coefficient posterior-SD units. Effective sample counts and all resolution
targets must pass.

Inner likelihood errors are propagated through normalized weights. If every
cached log likelihood changes by at most `d`, weight ratios lie between
`exp(-2d)` and `exp(2d)`; this gives score, curvature and mean perturbation
bounds from weighted centered moments. Here `d` is an estimated numerical
error, so the resulting diagnostics remain estimates rather than deterministic
or simultaneous confidence certificates. Correlated transformed Hessian
errors use sums of absolute transformation coefficients. A small optimization
gradient cannot suppress any of these error terms.

The returned `JointStrengthOptimum` retains its coefficient draws, weights and
posterior means. It is an assessed interior optimum of the supplied sampled
integral, not a completed joint-model fit or a guarantee of global optimality.
Posterior proposal coverage/integrability, exact null-boundary
comparisons, the selected entry law and serving still need completion. The
dense optimizer/curvature workspace is checked together with both banks
before optimization; a memory rejection is a computational limitation, not
evidence against an additional signature. A conjugate Poisson/Gamma test
checks the learned strength against its analytic optimum and requires
underresolved or nonstationary results to fail.

### Adaptive coefficient inference

`JointCohortIntegration::infer_coefficients` connects proposal fitting,
coefficient integration, strength optimization, and independent validation.
For a constant-rate zero-signature model without measurement channels, it
uses the exact route below. Otherwise each round fits the guided proposal against the same normalized cohort
likelihood and function priors, draws a fresh fitting bank, caches its resolved
likelihoods, and optimizes integrated evidence. Only then does it draw the
separate validation bank. Acceptance requires the value, whitened strength
score and curvature, coefficient means, and effective sample counts to pass
the assessment above.

The initial bank size follows the required effective sample count and the
coefficient/strength dimensions. If the assessment fails, the driver doubles
the sample count and re-anchors the proposal at the preceding posterior mean
and learned strengths. It discards both old banks before allocating their
replacements. A combined memory check covers both banks, evidence arrays,
proposal, optimizer, and assessment workspace before sampling. Exhausting
that budget returns an unresolved error. An inner likelihood error floor
requires refinement of the subject/reference banks; drawing more coefficients
cannot eliminate it. Optimizer and inner-integration failures propagate.

For sampled inference, `JointCoefficientInference` owns the final fitting draws and
their normalized proposal densities together with their posterior weights,
means, evidence, learned strengths, and refinement reports. It never replaces
posterior means with the proposal mode. The conjugate end-to-end test checks
the production cohort likelihood at every retained draw, the analytic evidence
optimum, and the analytic posterior mean, and exercises unresolved-memory
rejection.

These are conditional Monte Carlo error estimates, not confidence sequences
for the sequential stopping rule or a proof of tail coverage. This driver
handles interior strength inference for a declared structure and supplied
latent/reference banks. Null boundaries outside the constant-rate case, automatic rank selection,
subject/reference re-anchoring, integrated forecasts, and a standalone model
shared by Rust, Python, and the CLI still need implementation.

### Exact constant-rate inference and the zero-rate boundary

With zero signatures, a unit intercept as the only baseline column, and no
measurement channels, the counting-process likelihood factorizes by mark:

```text
L(r) = p(observed genetics) product_d r_d^y_d exp(-E_d r_d).
T r_d ~ Exponential(lambda), independently across marks at shared lambda.
c = lambda T.
p(data | lambda) = p(observed genetics)
                  product_d c Gamma(y_d+1) / (E_d+c)^(y_d+1).
r_d | data, lambda ~ Gamma(shape=y_d+1, rate=E_d+c).
```

`E_d` sums the supplied quadrature exposure while the subject is at risk for
that mark. Once-only events stop their own exposure; prevalent once-only
marks enter outside that risk set. Terminal events end the history. The
genetic likelihood uses the same analytic marginal over missing scores as
the subject integrator. `T` is precisely the frozen function prior's mean
follow-up span, so this route integrates the same model as coefficient
importance sampling.

Each exposed mark contributes score `E_d/(E_d+c)-y_d*c/(E_d+c)` in log `c`
and strictly negative curvature `-(y_d+1) E_d*c/(E_d+c)^2`. When events are
observed with positive risk exposure, the total score changes from positive
to negative, giving a unique evidence maximum. Equal exposed durations give
`c = sum_d E_d / sum_d y_d` in closed form; differing exposures use analytic
scores in `opt`, with a recomputed stationarity check and no search bounds.
No events in an exposed cohort instead give the exact global boundary
`c -> infinity`, with all rate posteriors concentrated at zero. An entirely
unexposed cohort has unidentified evidence and returns an error; an event
with zero supplied risk exposure has no finite optimum and is also rejected.

`JointCoefficientInference::constant_rates` exposes the exact rate means,
variances, and posterior no-event transform. The latter returns
`product_d (1 + u_d/(E_d+c))^(-(y_d+1))` for requested nonnegative exposures
`u_d`, integrating coefficient uncertainty rather than inserting mean rates.
Finite log-rate means are `digamma(y_d+1)-log(E_d+c)` and variances are
`trigamma(y_d+1)`. At the zero-rate boundary, finite log-coefficient moments
and log-strength coordinates do not exist: their accessors return `None`,
`is_zero_rate()` is true, physical rate means/variances are zero, and no-event
probabilities are one. Sampled evidence and draws are absent on the analytic
route, which consumes no coefficient RNG draws.

Tests compare the sampled and automatic analytic routes, check the unequal
risk-exposure optimum against its independent algebraic root, retain the
observed genetic density with a missing correlated score, and change time
units by factors of `1e100` and `1e-100`. These are model/numerical checks;
they do not establish external calibration or complete signature selection.

### Coefficient-integrated predictive history densities

`JointCohortIntegration::predictive_history_density` evaluates the joint
observation density of an additional cohort under the fitted coefficient
distribution:

```text
p(H_new | D, learned strengths)
    = integral p(H_new | theta) p(theta | D, learned strengths) d theta.
```

The additional subjects are conditionally independent of training subjects
given the model parameters. Training histories must not be supplied again.
The method checks the inference's training-cohort identity, the additional
cohort's model, and reuse of the training reference populations. At each
coefficient draw it regenerates the corresponding reference moments and
resolves the complete additional-cohort likelihood under those moments.
Thus each latent integral uses the same coefficient/reference state. The
sampled route rejects unsupported reference horizons. The analytic
constant-rate route needs no finite reference horizon because its normalizer
is identically one.

Several additional subjects share uncertain coefficients. Their joint
predictive density is an average of their joint likelihood, not the product
of separately averaged subject predictions. The test demonstrates the
difference using an independently known Gamma posterior. For constant rates,
Gamma integration handles both new events and event-free exposure exactly,
including the marginal density of partially observed genetics. An event
under the zero-rate posterior has log predictive density negative infinity.

For a sampled coefficient law, let `w_i` be normalized training weights and
`v_i` their normalized values after multiplying by the additional likelihood.
The conditional coefficient Monte Carlo standard error of the log predictive
density is `sqrt(n/(n-1) sum_i (v_i-w_i)^2)`. This retains the shared numerator
and denominator error: a constant added likelihood has zero coefficient
sampling error. Log training weights are retained even when their ordinary
weights underflow, since a later likelihood can make those draws relevant.
Training and new-history likelihood errors are not independent over
coefficient draws. The reported error estimate adds twice the training
log-likelihood error estimate and the maximum additional-cohort error to the
requested multiple of the coefficient standard error. The numerical error
target and effective coefficient sample requirement must both pass.

This returns a density in the model's observation measure. It is not a
terminal-survival curve, a cumulative-incidence forecast, or a conditional
forecast obtained by averaging coefficient-specific likelihood ratios.
Conditioning on a new subject's earlier outcomes uses the corresponding
joint predictive numerator and denominator described next. General future-path integration,
the reference-conditioned entry law, standalone serialization, and unified
Python/CLI serving remain unfinished.

### Conditioning on earlier histories

`JointCohortIntegration::conditional_history_density` pairs each earlier
history with an extension and computes

```text
p(new outcomes | earlier histories, D)
    = integral L(extended | theta) p(theta | D) dtheta
      / integral L(earlier | theta) p(theta | D) dtheta.
```

Earlier outcomes therefore update coefficient weights as well as the latent
state law. Averaging coefficient-specific conditional likelihood ratios under
the unchanged training weights would give a different answer. Both integrals
retain the person's original entry and latent trajectory. This method never
restarts a stationary prior at the assessment cutoff.

The paired subjects must retain their order and reference population. The
extension must preserve the exact prefix mesh, quadrature exposures, event
records, baseline and drive designs, entry context, genetics, and measurement
records, including before/after-event timing. New measurement records cannot
be inserted before the cutoff. Current genotype values carry no acquisition
times, so this interface does not permit changing them in a continuation.
These are checks on the model's numerical observation histories; creating
those histories from raw records remains an interface responsibility.

The sampled ratio uses the same coefficient bank for both integrals. If
`u_i` and `v_i` are its normalized weights after the earlier and extended
histories respectively, its conditional coefficient log-standard-error is
`sqrt(n/(n-1) sum_i (v_i-u_i)^2)`. Both integrals must pass the effective
sample requirement. The combined error estimate includes twice the training
likelihood error and the earlier/extended likelihood errors, without dividing
their correlated contributions by the number of coefficient draws. An
unchanged history uses the exact conditional identity of one. With no
appended events or observed measurements, the result is the probability of
no event of any mark over the new windows. A numerical value above one is
rejected for refinement, never clipped. It is not terminal survival with
nonterminal events integrated out.

The constant-rate route conditions the Gamma posterior analytically:
`shape += earlier event count`, `rate += earlier risk exposure`, then evaluates
the appended events/exposure under that updated distribution. It accumulates
appended exposure directly, avoiding subtraction of large totals. Earlier
once-only diagnoses still remove their marks from future risk, and the
unchanged genetic density cancels. Conditioning on an event impossible under
the zero-rate law is rejected. Tests check independent Gamma formulas,
coefficient reweighting, unchanged-history normalization, all past-data
checks, and once-only risk removal across extreme time-unit changes.

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

The latent solver now uses analytic location scores and curvatures for all
four measurement families. They avoid differentiating unused shape channels
and avoid gamma-function evaluations in Student-t and count location updates.
Tests compare them with derivatives of the complete observation density,
including extreme tails and measurements after disease jumps. This removes
one production AD hot path. It does not establish the performance exception
in `SPEC.md` for all remaining generic coefficient/reference derivative paths;
those still need analytic replacements or corresponding speed evidence.

`log_density_score` now supplies analytic complete-path coefficient scores for
the baseline surfaces, positive decoder, OU rates, genetic drive and entry
regressions, disease jumps, measurement locations/loadings, and measurement
shape parameters. The genetic distribution is fixed by the specification, so
its density has no fitted-coefficient score. Missing genetic values remain
latent path coordinates. Measurements after an event contribute to that event's
jump score as well as to their observation parameters.

The reference score is retained separately: a node contributes
`exposure * lambda - event` to its log-moment derivative. `JointPathScore::pullback`
requires the node-major reference Jacobian and adds its contribution to the
coefficient score. `JointIntegration::log_marginal_score` averages these total
scores under the same normalized importance weights as the likelihood value.
These low-level APIs require the supplied reference values and Jacobian to
describe the same coefficient state. They do not generate or certify that
Jacobian, and must not be wired to fitting with frozen or missing reference
sensitivities. `JointCohortIntegration::score` now constructs those sensitivities
from its own reference banks and uses this analytic kernel throughout.

`JointReferenceBank::sensitivity` propagates state, log-weight, and survival-mass
Jacobians through the retained population. Its analytic updates include OU
decay and innovation spread, genetic drive and entry regression, event
probabilities, disease jumps, killing, and conditional moment normalization.
The scalar value evolution is authoritative: every replayed conditional moment
and endpoint risk mass must agree exactly with it before a Jacobian is returned.
Moment interpolation uses the same grid and linear interpolation as the value.

`ResolvedReference::sensitivity` first rechecks all value-resolution ensembles
at the requested coefficients. It then differentiates the pooled fine ensemble,
including each replicate's risk-mass derivative. For normalized activity and
risk weights `a_r` and `m_r`, respectively, the pooled log-moment derivative is

```text
sum_r a_r d log M_r + sum_r (a_r - m_r) d log risk_mass_r.
```

An unweighted average of conditional Jacobians would omit that second term.
The result owns its coefficient vector, value curves, Jacobians, and resolution
report. That reference report assesses values, not derivative sampling or
discretization error. `JointCohortIntegration::resolved_score` separately
assesses the reference's effect on the entire likelihood and its total score.

The cohort processes one stratum's Jacobian workspace at a time, sums its
subject coefficient scores with compensated summation, and retains aggregate
conditional score errors. It does not retain a subjects-by-coefficients matrix.
Reference derivative workspace is bounded separately from the already retained
bank storage; interpolation also checks its requested Jacobian allocation.

The integrated score reports a delta-method sampling error per coefficient,
conditional on the supplied reference. A second pass over the fixed paths
accumulates `weight * (path_score - mean_score)` with scaled Euclidean norms.
It avoids squaring a tiny weight before multiplying a large score, subtracting
raw second moments, and retaining a samples-by-coefficients array. These errors
do not include reference uncertainty, finite-sample bias, or derivative
discretization error. Coefficient Hessians and the higher derivatives required
by LAML remain to be connected to the same objective.

### Shared-reference error in the cohort objective

`resolved_score` evaluates each stratum's summed log likelihood and total
coefficient score under the coarse-time, fine-time, and increased-particle
reference ensembles. It also evaluates these same functions after deleting
each independent reference population from an ensemble. Its jackknife measures
reference uncertainty after combining all subjects sharing that population.
Treating those subjects' reference errors as independent would miss a common
normalizer error that can grow linearly with cohort size.

For R populations and deletion values `F_(-r)`, the reference standard error
estimate is `sqrt((R-1)/R sum_r (F_(-r)-mean(F_(-r)))^2)`. The corresponding
bias estimate is `(R-1)(mean(F_(-r))-F)`. These calculations apply separately
to the likelihood and every total coefficient score. A deletion differentiates
the remaining ratio of risk-weighted activity to risk mass; it does not freeze
the normalizer or average conditional Jacobians without their mass derivatives.
The implementation subtracts a population from the pooled value/Jacobian,
then checks against direct re-pooling in tests. It uses two sequential passes
over the populations instead of R-squared reference replays or retaining all
population Jacobians.

The combined estimate includes absolute stratum refinement discrepancies,
the estimated final reference bias, and sampling-error margins. Independent
reference ensembles use root-sum-square errors. The same subject importance
banks are reused across refinements, so the conditional standard errors of a
difference are conservatively added rather than assumed independent. Stratum
discrepancies are added in absolute value to prevent cancellation. Replication
and importance banks must be independently generated where these calculations
assume independence.

`CohortScoreTolerance` supplies a likelihood-error budget and an absolute budget
for every coefficient score; an optimizer must derive those budgets from its
coefficient geometry and stationarity requirement. The method refuses an
unresolved evaluation and returns the original fine-ensemble coefficient,
normalizer, and score state only when the combined estimates pass. It never
changes a mesh, proposal, or random draw inside the objective. Refinement and
optimization restarts must happen outside that fixed objective.

This is a potentially expensive resolution checkpoint, with downstream
evaluations for every population deletion. It is not required on every line
search trial. Its finite-replicate error estimates are not deterministic bounds
or simultaneous confidence guarantees, and they do not assess subject time-mesh
error, global coefficient curvature, or external calibration. A final fitting
driver and structure-learning implementation remain unfinished.

The implemented importance bank draws from an equal mixture of the structured
Laplace Gaussian and a multivariate Student t with three degrees of freedom,
centered at the conditional path-prior mean with the same covariance. The
complete joint law, including its Gaussian path prior, remains in the importance
numerator. Three is the smallest integer degree of freedom with a finite covariance, allowing that
covariance match while providing polynomial tails. The Student displacement
is a structured Gaussian draw divided by the square root of a chi-squared
draw with three degrees of freedom. Both mixture densities are normalized.

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

For every fixed finite coefficient state with a proper Gaussian path law,
the observation factors grow at most polynomially in that path. The Student
component therefore gives finite second moments of importance weights and
polynomial state summaries, without restricting coefficient changes to
`2 Q(theta) - Q(anchor) > 0`. Finite variance does not ensure adequate coverage
by a finite sample or efficient sampling in a large or multimodal problem.
Posterior means and selected covariance blocks retain their own estimated
standardized-error acceptance limit.

When there are zero signatures, or all marks are already out of their risk
sets and no measurements are observed, the latent integral is Gaussian by
model structure. These cases use analytic integration, including the marginal
density of observed genetic scores and the conditional law of missing scores.
They return exact Gaussian means and selected covariance blocks, zero sampling
errors, zero samples, and absent effective-sample/weight diagnostics. The
`LatentIntegrationMethod` distinguishes them from importance estimates.
The rank-zero route retains all observation and reference derivatives; the
fully unobserved route has zero coefficient score. Small rates or loadings
at a particular coefficient value do not trigger this structural shortcut.

The rank-zero reference population also has an analytic route. For a profile
interval with linear log baseline, its integrated hazard is
`dt * exp(eta_left) * exprel(eta_right - eta_left)`, evaluated in a scaled
log form. Log risk mass is minus the cumulative terminal hazard, with the
mark's own cumulative hazard added for a once-only mark. Log normalizers and
their coefficient derivatives are zero. Values at the stored endpoints and
midpoints, and their total derivatives, therefore do not require event-step
refinement or a hazard cap. Interpolation between stored points retains its
separate resolution requirement. Positive-rank references still use the
controlled numerical evolution.

The relative-exponential derivative tower uses moments of a tilted uniform
law, with a positive series near zero and a contracting recurrence away from
zero. Normalized moments avoid underflow in ratios. Tests compare through
fourth derivatives with independent quadrature and check competing-risk masses
and first derivatives across time refinements. This analytic route also lets
coefficient pilots cross rate values that the former finite-event reference
step incorrectly made numerically inaccessible at rank zero.

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
