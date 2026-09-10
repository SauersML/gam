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
objective. The function priors below are implemented; coefficient integration
and hyperparameter learning are unfinished.

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
blocks. GAM operator-specific smoothness penalties, coefficient integration,
and REML/LAML optimization still need completion.

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
