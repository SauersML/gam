# Marginal-slope models

A marginal-slope model fits a standardised risk score `z` whose effect on
the outcome varies across covariate space. The baseline risk surface and
the score's slope surface live in two separate formulas, so the
baseline does not absorb score-specific signal and vice versa.

![two-surface marginal-slope viz over a joint Duchon smooth](images/marginal_slope_3d.png)

The vertical gap between the two probability surfaces is the risk
difference for a unit contrast in `z`. The modelled score effect lives on
the probit/slope scale and varies smoothly with covariates.

Two families:

- Bernoulli marginal-slope for binary outcomes. In Python, pass
  `family="bernoulli-marginal-slope"` with `slope_formula=`; in the
  CLI, `--z-column` and `--slope-formula` route to this fit.
- Survival marginal-slope (`survival_likelihood="marginal-slope"`) for
  time-to-event outcomes.

Both are identified around a latent `z` scale that should be approximately
`N(0, 1)` conditional on the covariates. Pass
`transformation_normal_stage1=gamfit.CtnStage1(...)` to condition the score
on covariates inside the fit (the calibrated chain below), or a raw
`z_column=` when the score is already conditionally `N(0, 1)` from outside
this pipeline.

## When to use it

You have:

1. A binary or time-to-event outcome.
2. A continuous risk score that is, or can be made, conditionally
   `N(0, 1)`.
3. Reason to believe the score's effect size varies across covariates
   (e.g. across age, grouping PCs).

A single-coefficient logistic or Cox fit on `outcome ~ score + ...`
forces one slope on the score. Marginal-slope makes the slope itself a
smooth function of covariates while leaving the baseline as a separate
smooth.

## Calibrated marginal slope (CTN-conditioned score)

When the score must be conditioned on covariates to reach the latent
`N(0, 1)` scale, supply a Stage-1 transformation-normal recipe with
`transformation_normal_stage1=`. This is the single calibrated
marginal-slope entry: it fits the conditional transformation
`h(score | covariates) ~ N(0, 1)` in each fold's training complement. An
ordinary penalized marginal-slope outcome uses those OOF scores, with no
influence absorber or second normalization. A separate full-training CTN is
saved together with the outcome in the native GAM `Model`. Rust owns the
composition, folds and persistence for library, CLI and Python. Prediction replays
that frozen transform on raw scores. Cross-fitting alone establishes neither
Neyman orthogonality nor outcome calibration.

Supply explicit fold labels or a group column. Group assignment is seeded and
independent of row order; supplied folds are checked for group separation when
both columns are present. All fitting data must belong to the outer training
sample. Fold-local knots and geometry never use the outer test sample.

```python
import numpy as np
import pandas as pd
import gamfit
from scipy.stats import norm

rng = np.random.default_rng(0)
n = 600
df = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=["pc1", "pc2", "pc3", "pc4"])
df["age"], df["family_id"] = rng.uniform(40, 70, n), np.arange(n) // 2
z = rng.normal(0, 1, n)                              # latent score, N(0, 1) given the PCs
df["raw_score"] = 0.5 * df["pc1"] + np.exp(0.3 * z)  # observed score: shifted and skewed
df["case"] = (rng.uniform(size=n) < norm.cdf(-0.5 + 0.03 * (df["age"] - 55) + (0.6 + 0.3 * df["pc2"]) * z)).astype(float)
test_df = df.iloc[:100]

model = gamfit.fit(
    df,
    "case ~ s(age) + matern(pc1, pc2, pc3)",
    family="bernoulli-marginal-slope",
    slope_formula="matern(pc1, pc2, pc3)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="raw_score",
        covariates="duchon(pc1, pc2, pc3, pc4, centers=20)",
        group_column="family_id", folds=2, seed=20260910,
    ),
    scale_dimensions=True,
)

probs = model.predict(test_df)
model.save("predictor.gamfit")
probs_reloaded = gamfit.load("predictor.gamfit").predict(test_df)
```

By default, Bernoulli marginal-slope prediction returns a 1-D NumPy
array of probabilities. Each probability is the posterior-predictive
probability of the anchored model, `E[Φ(η(θ)) | data]` over the coefficient
posterior: the linear predictor is `η = c(b)·q + b·z` (or `a(q, b) + b·z`
under a declared empirical latent law), so a coefficient draw moves the
marginal index `q`, the slope `b` *and* the anchor `c(b)·q` / `a(q, b)`.
Because `q` and `b` are affine in the coefficients, the integral is taken
exactly over their bivariate Gaussian law with the anchor re-solved at every
quadrature node — not by inserting posterior-mean coefficients into `Φ`, and
not by the Gaussian shortcut `Φ(η̂/√(1 + v))`, which is exact only when `η`
itself is Gaussian. The shortcuts remain reachable by name in Rust
(`gam_predict::bernoulli_marginal_slope::AnchoredPosteriorIntegration`) for
comparison; with a score-warp or link-deviation runtime the anchor depends on
those coefficient vectors as well and the point is the first-order
(linearised-anchor) integration. Passing `return_type=` asks for a table.
Passing `interval=0.95` asks for the interval table with `linear_predictor`,
`mean`, `std_error`, `mean_lower`, and `mean_upper`; probability-scale
values are clipped to `[0, 1]`. `std_error` is the probability-scale
posterior standard error (the same response-scale quantity every class's
`std_error` / `posterior_mean_standard_error` column carries); the bounds
are the inverse probit of the η-scale endpoints `η ± z·se_η`, so they are
symmetric about `linear_predictor` on the link scale.

- `family="bernoulli-marginal-slope"` names the likelihood;
  `slope_formula=` is the slope surface as a function of covariates.
- `transformation_normal_stage1=gamfit.CtnStage1(response=..., covariates=...)`
  is the Stage-1 recipe: `response` is the raw score column to condition,
  `covariates` is the covariate-side formula right-hand side used to fit
  `h(score | covariates) ~ N(0, 1)`. `fold_column` supplies explicit labels;
  `group_column` assigns entire families together. Failed folds raise an error.
- `model.transformation_score(data)` replays the embedded transform using raw PGS
  and its fitted covariates. The native payload stores the transform and outcome.
  Outcome intervals are conditional on the fitted CTN, not full two-stage
  uncertainty estimates. The predictive chain does not claim Neyman orthogonality.
- The base link is fixed to probit. The Python `link=` keyword is not
  needed for marginal-slope fits.

The main formula controls the baseline risk; `slope_formula` controls
the strength of the score effect at each point in covariate space.

The same recipe drives the survival likelihood:

Use a prospective prediction frame with entry fixed at baseline and exit set
to the requested prediction horizon, rather than each person's observed outcome
time. Evaluate the returned survival surface at explicit times. For delayed
entry, condition on surviving to entry. Competing-risk incidence needs separate
cause components and a CIF, not one minus disease-only net survival.

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 600
df = pd.DataFrame({"bmi": rng.normal(27, 4, n), "hba1c": rng.normal(6, 0.8, n), "family_id": np.arange(n) // 2})
z = rng.normal(0, 1, n)
df["raw_score"] = 0.1 * (df["bmi"] - 27) + np.exp(0.3 * z)
t = rng.exponential(10 * np.exp(-0.05 * (df["bmi"] - 27) - 0.3 * (df["hba1c"] - 6) - 0.5 * z))
c = rng.uniform(2, 25, n)
df["entry"], df["exit"], df["event"] = 0.0, np.minimum(t, c), (t <= c).astype(float)
test_df = df.iloc[:100].assign(exit=10.0)   # prospective frame: exit = prediction horizon

model = gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(bmi) + s(hba1c)",
    survival_likelihood="marginal-slope",
    slope_formula="s(bmi) + s(hba1c)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="raw_score",
        covariates="s(bmi) + s(hba1c)",
        group_column="family_id", folds=2,
    ),
)

pred = model.predict(test_df)
S = pred.survival_at([1, 5, 10])
```

An already fitted external CTN can be passed through the same native composition:

```python no-exec
reference = gamfit.load("reference.gamfit")
model = gamfit.fit(
    df, "Surv(entry, exit, event) ~ s(age) + sex + duchon(pc1, pc2)",
    survival_likelihood="marginal-slope", slope_formula="1 + duchon(pc1, pc2)",
    transformation_normal_stage1=reference,
)
```

This applies the supplied CTN unchanged to both fitting and prediction. No
target-data refit, additional normalization, or influence matrix is performed.
Matching score and covariate definitions remains the caller's responsibility;
an external transform does not certify target-population conditional normality.

The main formula specifies the baseline survival surface; the score's
slope on the marginal-calibrated probit survival scale is a smooth
function of covariates given by `slope_formula`. In the Python API,
omitting `slope_formula` reuses the main covariate formula for the
slope surface.

### A note on the name: `slope_formula` is the SLOPE surface

The block used to be called `logslope`; the map is the identity. `slope_formula=`
gives the surface `b(x)` that the latent score enters the probit index
with directly, not its logarithm, and that is deliberate rather than an
oversight:

- **The slope is signed.** `b(x) < 0` is a score that is protective at
  that point in covariate space, and with several scores each carries its
  own sign. A genuine log link `b = exp(g)` cannot express either. Where
  the surface crosses zero is not a pathology; it is the covariate value
  at which the score stops predicting.
- **A log penalty would not change the fit, only `λ`'s units.** Rescaling
  the score `z → z/κ` sends `b → κb` and the score covariance
  `Σ → Σ/κ²`, and the index `η` and the preserving scale `c` are
  *pointwise* unchanged under that — measured, in
  `survival_multi_z_slope_scale_equivariance_2764`. What is left is the
  penalty `λβᵀSβ`, which needs `λ → λ/κ²` to match, and REML supplies
  exactly that on its own: its criterion shifts by a constant in `λ` under
  the reparameterisation, so `λ̂` moves and the fitted surface does not. A
  penalty on `log b` would additionally pin the numeric value of `λ̂`; that
  is a statement about units, and it would cost the sign.

The keyword, the CLI flag and the saved-model fields were renamed to `slope`
as a deliberately breaking change: no `logslope` alias or persisted fallback
is retained, so a model saved under the old field names is not read as a
slope model and must be refitted. Inside the library the function that states
the map is called `rigid_observed_slope`, because that is what it computes.

## Letting the slope vary along follow-up (survival)

`slope_formula` makes the slope a surface in *covariates*. On the survival
likelihood it can also be a surface in *follow-up time*, which is the natural
question for a score whose effect is thought to attenuate with age:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
n = 500
bmi, z = rng.normal(27, 4, n), rng.normal(0, 1, n)
t = rng.weibull(1.5, n) * 10 * np.exp(-0.05 * (bmi - 27) - 0.6 * z)
c = rng.uniform(2, 25, n)
df = {"entry": np.zeros(n), "exit": np.minimum(t, c), "event": (t <= c).astype(float), "bmi": bmi, "z": z}

model = gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(bmi)",
    survival_likelihood="marginal-slope",
    z_column="z",
    slope_formula="s(bmi)",
    config={"slope_time_k": 6},   # B-spline margin in log(time)
)
```

`slope_time_k` tensors the slope covariate design against a B-spline
margin in `log t`, exactly as `threshold_time_k` and `sigma_time_k` do for the
location-scale family, so `b` becomes a fitted surface `b(x, t)` with
independent smoothing parameters for the covariate and time directions.
`slope_time_degree` (default `3`) sets the margin's polynomial degree; the
same `k >= degree + 1` rule applies. The CLI spelling is `--slope-time-k`.

Why this needs family support rather than data reshaping: the marginal-slope
likelihood is a *transformation* model, `S(t | x, z) = Φ(−η(t))`, not a hazard
model. Splitting each subject into intervals with a piecewise-constant slope —
the usual Cox `tt()` workaround — gives per-row contributions
`log S(t₁) − log S(t₀)` that do not telescope into any survival function. The
slope has to move inside the row likelihood, which also means the event density
picks up the two terms a constant slope zeroes out:

```
η′(t) = q′(t)·c(t) + q(t)·c′(t) + b′(t)·z
```

Current boundaries, all of which are refused with a message rather than
silently reinterpreted:

- a per-score slope topology (a vector latent score with one slope surface
  per coordinate) cannot take a single time margin;
- a non-zero smooth anchor, coefficient bounds, or linear constraints on the
  slope surface are stated in the covariate coordinate chart, and the time
  tensor product is a different chart.

Saving and prediction carry the margin. The resolved knots ride on the saved
model next to the threshold and log-σ margins, and predict rebuilds the block as
`X_cov ⊗ᵣ B(log t)` against *those* knots, so a prediction sample can never move
the basis by re-estimating quantiles. Two consequences worth stating, because
they are what makes the replay faithful rather than merely well-typed:

- a predicted survival curve evaluates `b(t)` **at each time on the curve**, not
  at the row's observed exit time. `S(t) = Φ(−η(t))` with `η(t) = q(t)c(t) +
  b(t)z`, so freezing `b` at the exit time would return a different model at
  every point of the curve but one;
- the leave-one-out (`--alo`) replay rebuilds all three follow-up channels —
  entry, exit, and exit-rate — because that is what the row program reads. With
  only the exit channel it would report the influence of a time-constant slope.

## Externally calibrated score (raw `z_column`)

If the score was produced outside this pipeline, pass it directly with
`z_column=` and omit the Stage-1 recipe. This raw-`z` path uses the
free-warp `score_warp` fallback for shape miscalibration, and anchors the
index on the estimated law of the score described below — the score does
not have to be standard normal, conditionally or at all. Use
`config={"frozen_score": True}` when an external transformation already
supplies a conditionally standard-normal score and the fit should use the
Gaussian closed form. That is a declaration about the score, and the fit
checks it: it refuses the declaration when the score's conditional law moves,
and warns with what the declaration costs when the pooled score only looks
non-normal. Prefer the CTN
chain when the fitted transformation should travel with the outcome
predictor.

```python
import numpy as np
import gamfit
from scipy.stats import norm

rng = np.random.default_rng(0)
n = 500
pc, age, z = rng.normal(0, 1, (n, 3)), rng.uniform(40, 70, n), rng.normal(0, 1, n)
p = norm.cdf(-0.5 + 0.03 * (age - 55) + 0.3 * pc[:, 0] + (0.6 + 0.3 * pc[:, 1]) * z)
df = {"age": age, "pc1": pc[:, 0], "pc2": pc[:, 1], "pc3": pc[:, 2], "z": z,
      "case": (rng.uniform(size=n) < p).astype(float)}

model = gamfit.fit(
    df,
    "case ~ s(age) + matern(pc1, pc2, pc3)",
    family="bernoulli-marginal-slope",
    z_column="z",
    slope_formula="matern(pc1, pc2, pc3)",
    scale_dimensions=True,
)
```

- `z_column="z"`: name of the conditional z-score column in both the
  training and prediction tables.

CLI equivalent:

```bash
gam fit data.csv 'case ~ s(age) + matern(pc1, pc2, pc3)' \
    --slope-formula 'matern(pc1, pc2, pc3)' --z-column z \
    --scale-dimensions --out model.gam
```

Bernoulli marginal-slope currently consumes a single `z_column`.

### Which latent law the index is anchored on

You do not have to make the raw score look normal. The parameterisation's
whole point is that `q` is the **marginal** index, and that is an identity
about the law of the score in context:

```text
E[Φ(α(a) + b(a)·z) | a] = Φ(q(a))
```

On any finite law of `z | a` this equation has exactly one solution
`α(a)`, and the familiar closed form `α = q·√(1+b²)` is its
`z | a ~ N(0, 1)` case. The model does not need the score to be Gaussian;
it needs the fit to anchor on the law the score has. So by default both
marginal-slope families estimate that law and anchor on it, with the score
on its own axis:

1. a robust Rao score test of `E[z|a]`, `Var(z|a)` and the third
   standardised moment over the marginal-index span;
2. if none moves and the score passes the standard-normal adequacy screen
   (mean, SD, skewness, kurtosis, KS distance, tail mass, largest `|z|`),
   the closed form, provisionally. Each of the screen's eight bounds is the
   null quantile of its own statistic at the sample's Kish effective size,
   at level 0.05 split over the eight, so an exactly Gaussian score fails the
   screen at most 5% of the time at every `n`, and a departure fails it once
   `n` resolves it: the KS bound is Kolmogorov's critical value, about
   `1.70/√n`. The screen only decides which route the fit starts on and says
   nothing directly about the anchoring error, so at the converged fit one
   pass over the rows measures each row's residual `r = Σ_k w_k Φ(a_cf + h_k) − π` under the
   estimated law at the closed-form intercept, and its sampling standard
   error `se` under that law. `r² − 2·se²` estimates without bias how much
   less accurate the closed form is than the estimated law's own anchor on
   that row, and the fit records `D̂ = Σ w (r² − 2·se²)/(π(1−π))`. It keeps
   the closed form (`estimated-gaussian-adequate`) unless the residual
   energy `Σ w r²/(π(1−π))` is beyond what the estimated law's own sampling
   error gives an exactly Gaussian score: its exact null law is a weighted
   chi-square over the anchors' shared noise, and the closed form is kept
   unless the energy is in that law's upper 5%. Otherwise the fit re-solves on
   the estimated law from the closed-form coefficients
   (`estimated-global-by-residual`). On exactly Gaussian scores the
   certificate fired on 1 of 40 fits at 2 000 rows and 3 of 40 at 100 000
   (design 5%), where the sign of `D̂` fired on 4 and 8 of the same 40;
3. if none moves and the score fails that check, one finite law of the
   score — a 65-node equal-mass compression that keeps the score's own
   location and scale;
4. if any moves, local finite laws by context: the training rows are
   partitioned over the covariates the marginal formula reads, each context
   gets its own law, and every row anchors on a kernel mixture of its four
   nearest contexts plus a small fixed share (1e-3 of the kernel's peak) of
   the pooled law. A context's weight falls to zero exactly where it stops
   being one of the four nearest, and the pooled share keeps the mixture
   defined where contexts tie, so the law, the anchor and the prediction are
   continuous in the covariates everywhere.

   Where a moment moves the law is chosen among nested arms, simplest first:
   the Gaussian law, the location-scale law `m(a) + √v(a)·ε` with a Gaussian
   `ε`, the same with `ε` on its estimated law, and the local laws. An arm
   that anchors on a Gaussian residual (the score for the Gaussian arm, `ε`
   for the location-scale Gaussian arm) is a candidate only if that residual
   passes the standard-normal adequacy screen. The fit is solved on the
   simplest candidate of the location-scale structure, and at the converged
   fit the certificate takes the simplest candidate whose cross-fitted
   excess anchoring loss is within one paired standard error of the lowest
   candidate's, re-solving on it when it is another arm. A heavy-tailed or
   skewed `ε` therefore anchors on its estimated law even where the
   cross-fitted loss does not resolve the Gaussian arm from it.

The conditional test comes first because a score can be exactly `N(0, 1)`
overall while every conditional law `z | a` is shifted. One pooled law then
puts `b(a)·E[z|a]` into `q` and the marginal coefficients are wrong, and no
transform of the score's marginal distribution can fix that, because the
marginal distribution is already correct. A law estimated by context does.

Every other law is a declaration, set with `config={"latent_measure": ...}`:

| `latent_measure` | Law anchored on | When the score contradicts it |
|---|---|---|
| `"auto"` (default) | the closed form when the estimated law passes the adequacy check, else the estimated law, global or local by context | — |
| `"gaussian"` | the closed form, `N(0, 1)` | refused when the conditional law moves; otherwise fitted with a warning and its `D̂` |
| `"global-empirical"` | the pooled estimated law, whatever the span shows | — |
| `"conditional-location-scale"` | `z = m(a) + √v(a)·ε` with `ε` on its estimated law; the slope lives on `ε`'s axis | — |

`config={"declared_latent_law": {"nodes": [...], "weights": [...]}}` anchors
on exactly the finite law given. `frozen_score=True` and the CTN chain
declare the Gaussian law.

A Gaussian declaration is checked, never assumed. When `E[z|a]` or
`Var(z|a)` moves on the span, no single declared law can be right, and the
fit is refused with the p-values. When the pooled score fails the
standard-normal adequacy screen (mean, SD, skewness, kurtosis, KS distance,
tail mass, largest `|z|`), the declaration is fitted and the fit warns with
the adequacy ledger and the declaration's estimated excess anchoring loss
`D̂` at the converged fit, both recorded with the model. The screen is a
level-0.05 test of the standard normal, and a failed test is not grounds to
refuse a declaration: at large `n` it detects departures that cost no
anchoring accuracy, and `D̂` measures what the departure costs. To fit the closed form
on purpose on a score that is not normal without the warning, declare a
Gauss–Hermite law: it is the Gaussian case to quadrature tolerance.

Which law the fit consumed is **persisted with the model** as
`latent_law_consumed` — `estimated-gaussian-adequate` (with every adequacy
statistic beside its bound and the anchoring certificate),
`estimated-global-by-residual`, `gaussian-uncertified` (with what is
missing), `estimated-global`, `estimated-local`,
`requested-global-empirical`, `declared-finite-law`,
`conditional-location-scale` or `declared-gaussian`, with the test
evidence — beside the law itself (`latent_measure`). Prediction and
leave-one-out diagnostics replay that law by the same anchoring equation,
because the fitted coefficients are defined against its anchor and mean
nothing under another; a local law is replayed from the prediction table's
own covariates. Models saved before the field existed replay their old
calibration unchanged.

When the declared location-scale law calibrates the score, the score
becomes a generated regressor: the coefficient covariance carries a
Murphy–Topel correction for the first stage's estimation error, or is
withheld with a typed reason if the fit's shape cannot supply the
correction. It is never published uncorrected. The estimated laws are built
from the score as given and have no first stage to correct for.

On a Gaussian score the default is the closed form, the pooled estimated law
(`global-empirical`) agrees with it to sampling tolerance, and a declared Gauss–Hermite law agrees with the closed form to
quadrature tolerance once its node count is tied to the drive scale
`b·sd(z)`: at drive SD 2–4 a 64-node law is within about 1.5e-4 and a
128-node law within about 4e-8.

### The survival kernel anchors on a declared law

The identity above is the closed-form standard-normal lowering: with
`z | a ~ N(0, 1)` the row index is `η = q·√(1 + b²) + b·z` and `q` is the
marginal index. That lowering is exact only for a Gaussian score. On a
declared finite law `F_a` — nodes `u_k` with weights `w_k` — the survival
kernel instead solves the defining equation itself, per row and per time:

```text
Σ_k w_k Φ(−(α(t, a) + b(a)·u_k)) = Φ(−q(t, a)) ,        η = α + b·z .
```

The left side is continuous and strictly decreasing in `α` with limits
`1` and `0`, so `α` exists and is unique; it is found by a bracketed
Newton–Halley solve, and its derivatives for the score, Hessian and the
higher-order towers come from implicit differentiation of the same
equation (`α_θ = −Σ_k w_k φ(η_k) ∂_θ(b·u_k) / Σ_k w_k φ(η_k)`, and so on).
Gaussian is the special case: on a Gauss–Hermite law the anchored fit
reproduces the closed-form fit — coefficients, log-likelihood and the
fitted survival index — to quadrature tolerance, which the acceptance
test `declared_latent_law_2923` pins. On a skewed law the two are
different models: the closed form still fits the conditional law (a
flexible baseline absorbs `α`), but its `q̂` stops being the marginal
index, so `Φ(−q̂(t))` is off from the marginal survival by a measurable
amount; the anchored `q̂` is the marginal index, as the identity says.

Three ways to get there:

- the default: the fit anchors on the estimated law of the score, global
  or local by context, exactly as the Bernoulli family does (the closed form
  when the score passes the adequacy check);
- `config={"latent_measure": "global-empirical"}`: always anchor on the
  pooled estimated law of the score;
- `config={"declared_latent_law": {"nodes": [...], "weights": [...]}}`:
  anchor on exactly this law. The score is then taken as supplied — nothing
  is estimated or checked, because the law is your statement about that
  very score.

`config={"latent_measure": "gaussian"}` reaches the closed form, checked as
described above.

Whichever way, the law is **persisted with the model** as its latent
measure and replayed at prediction and in leave-one-out diagnostics by
the same anchoring equation; the saved coefficients are defined against
that law's anchor and mean nothing under another.

Current boundaries, refused with a message rather than silently
reinterpreted: a declared law is a law of one score (several scores anchor
on their joint law, below); no score-warp or
link-deviation flex block, no CTN Stage-1 influence absorber, no
time-wiggle baseline, and a time-constant slope. On those configurations an
explicitly requested finite law is refused by name, and the default keeps the
closed form and certifies it by `D̂` as above: a flex block through its own
de-nested index at each node of the estimated law, a follow-up-varying slope on
each anchor's own slope, and an influence absorber on the offset-free anchor
the fit solves, because the absorber's offset is added only after the anchor.
Where the certificate prefers the estimated law, which nothing there can
re-solve on yet (gam#2948), the fit keeps the closed form, recorded
`gaussian-uncertified` with that certificate and why nothing re-solves on it,
which `require_certified` refuses by name. A fit on any of
them whose law departs or moves records `gaussian-uncertified` with a warning
naming what is missing. The Jeffreys/Firth arming's closed-form fifth
and sixth derivatives are the Gaussian lowering's and are not served on a
declared law; the fit runs without them.

### Several scores at once, and the covariance between them

The survival family accepts more than one latent score — one `z(...)`
surface per score in `slope_formula=`, each with its own slope
surface. With `K` scores the row index is

```text
η = c(a)·q(t, a) + Σ_k r_k(a)·z_k ,
```

and the identity above generalises to

```text
E_z[Φ(−η) | a] = Φ(−q(t, a))     ⟺     c(a) = √(1 + r(a)ᵀ Σ(a) r(a)) ,
```

with `Σ(a) = Var(z | a)` the **conditional** covariance of the score
vector. Only the diagonal of that is reachable by the per-coordinate gate
above: it standardises each `z_k` given `C`, which forces
`Var(z_k | a) = 1`, and leaves `Cov(z_j, z_k | a)` untouched. Two scores
can each be conditionally standard normal while their correlation moves
across the covariate space — different ancestries, different genotyping
arrays, different assay batches all do this — and then one pooled `Σ̄`
gives the wrong `c` at every row. The consequence is not subtle: the
realised marginal index becomes `q·c̄/c(a)`, so every marginal coefficient
is multiplied by a covariate-dependent factor. Measured on a two-score
sample whose conditional correlation moves over `±0.8`, that factor
reaches **1.46**.

So the fit tests for it. One robust Rao score test per score pair, on the
same conditioning span and at the same level as the gate above, asks
whether `Cov(z_j, z_k | a)` is constant. If no pair says otherwise, the
pooled covariance is used unchanged. If one does, the fit estimates
`Σ(a)` and every row gets its own `c(a)`.

The estimated object is a **modified Cholesky** regression —
`T(a)Σ(a)T(a)ᵀ = D(a)`, with `T` unit lower triangular and `log D(a)`
linear in the conditioning span. That parameterisation is unconstrained,
so `Σ(a)` is positive definite at every `a` including rows far outside
the training data; each fitted linear predictor is additionally held to
the range it took over the training rows, because a linear predictor is
only identified on the range the sample explored.

Those predictors are **linear in the conditioning span**, which is the
marginal design. So the model removes the part of the covariance's
variation that the marginal design can express, and no more — if the
correlation moves smoothly with a covariate, put a smooth of that
covariate in the marginal formula and the span will carry it. A bare
linear column leaves a real residual when the truth is curved, and that
residual grows with the slope rather than sitting at the noise floor.

Two limits are worth stating plainly:

- **`K = 1` is unaffected.** There is no off-diagonal, and `Var(z | a)` is
  already the per-coordinate gate's business. A single-score fit is
  bit-for-bit what it was.
- **A closed-form fit that used a conditional `Σ(a)` is refused at save
  time.** The closed-form contract carries one score column and one score
  covariance, so saving is refused at the point of loss with the reason
  attached.
- **The declared-law path saves `K ≥ 2`.** With one `slope(z_k, ...)`
  surface per score and `config={"latent_measure": "global-empirical"}`,
  the anchor reads only the law of the drive `rᵀz`, so the fit declares the
  joint law of the score vector: the pooled law of the whitened training
  residuals `ε = L(a)⁻¹(z − μ)`, compressed to 128 nodes with its mean and
  covariance kept exactly, and transported to each row's context as
  `μ + L(a)ε` — `L(a)` the factor of the conditional `Σ(a)` above when the
  pair gate fires, of the pooled `Σ̄` otherwise. On Gaussian scores this is
  the conditional closed form; on any other residual shape it is that
  shape's anchor. The law, its transport, one score column and one slope
  surface per score are persisted, and prediction replays the same anchor
  per row. Refused by name on this path: a shared slope over several scores,
  a spatial length scale on a slope surface or in the marginal formula, a
  learned frailty, per-score pre-transforms, uncertainty bands, the
  posterior-mean estimand, and leave-one-out replay.
- **The default on several scores.** Under `"auto"` each score is tested as
  above. When some score departs from the standard normal without moving on
  the span, the fit anchors on the joint law, and the scores the screen
  passed are recorded as `estimated-global`, the law they are anchored on.
  When some score's law moves, the transport `μ + L(a)ε` follows a moving
  covariance but not a moving mean or shape, so every score keeps the closed
  form, recorded as `gaussian-uncertified` naming the moving score (gam#2949).
  When every score passes the screen, the closed form at `Σ(a)` is
  provisional, and the converged fit certifies it by `D̂` under the joint
  law, each row's residual `Σ_m w_m Φ(−(q·√(1 + rᵀΣ(a)r) + rᵀu_m)) − Φ(−q)`
  on the row's transported nodes. Where the certificate fires, the fit
  re-solves on the joint law; where nothing can re-solve on it, as with a
  slope shared across the scores, it keeps the closed form, recorded
  `gaussian-uncertified` with the certificate.

## Residual genetic repair: reading what the score discarded

A polygenic score keeps **one** direction of the genome. A varying slope
`b(a)` rescales that direction across covariate space; it cannot turn it
into a different predictive direction, and no calibration of the same
scalar — affine or not — recovers variation in `E[Y | genome, a]` that
`(S, A)` does not determine. What can is a block of genetic features the
score threw away, entered next to the score and shrunk like everything
else:

```text
η_i = c(a_i)·q(a_i) + b(a_i)·z_i + βᵀ r_i ,        r_i = φ_i − E_ref[φ | S_i, A_i]
```

`r` is a block of `K` **conditionally centred** residual features — block
partial scores, local-ancestry contrasts, selected dosages — and `β` is a
constant coefficient block under one ridge penalty whose smoothing
parameter is estimated by REML/LAML like every other penalty. Under
`r ⟂ Y` the marginal likelihood drives that penalty up and `β → 0`,
leaving the score-only fit unchanged. The population value a linear read
of `r` removes from squared risk is exactly `c_rᵀ Σ_r⁺ c_r` with
`c_r = E[rY]`, `Σ_r = E[rrᵀ]`, whatever the rank
(`Descent.Portability.ResidualGeneticRepair.residual_repair_law`).

```python
import numpy as np
import gamfit
from scipy.stats import norm

rng = np.random.default_rng(0)
n = 500
pc, age, z = rng.normal(0, 1, (n, 3)), rng.uniform(40, 70, n), rng.normal(0, 1, n)
r = rng.normal(0, 1, (n, 3))                      # conditionally centred residual features
p = norm.cdf(-0.5 + 0.03 * (age - 55) + (0.6 + 0.3 * pc[:, 1]) * z + 0.4 * r[:, 0])
df = {"age": age, "pc1": pc[:, 0], "pc2": pc[:, 1], "pc3": pc[:, 2], "z": z,
      "chr6_partial": r[:, 0], "chr11_partial": r[:, 1], "afr_contrast": r[:, 2],
      "case": (rng.uniform(size=n) < p).astype(float)}

model = gamfit.fit(
    df,
    "case ~ s(age) + matern(pc1, pc2, pc3)",
    family="bernoulli-marginal-slope",
    z_column="z",
    slope_formula="matern(pc1, pc2, pc3)",
    residual_columns=["chr6_partial", "chr11_partial", "afr_contrast"],
)
```

CLI: repeat `--residual-column NAME` for every column. The prediction table
must carry the same columns, centred on the same reference law.

### The anchor moves with the block

The marginal interpretation `E[p | a] = Φ(q(a))` is what keeps the
baseline surface meaningful, and it needs the anchor to integrate the
**whole** genetic drive. Under the conditionally Gaussian joint law of
`(z, r)` that is

```text
c(a) = √(1 + b̃(a)ᵀ Σ(a) b̃(a)) ,      b̃(a) = (b(a), β) ,      Σ(a) = Var((z, r) | a) ,
```

the several-scores identity above with the residual block appended to the
score vector. The score's coordinate keeps its declared unit variance, so
at `β = 0` the anchor is exactly the score-only `√(1 + b²)`; its couplings
to `r` and `Var(r)` are estimated. `Σ(a)` is the pooled joint covariance
unless the pairwise Rao gate of the several-scores section escalates it to
a conditional `Σ(a)`; either way it is persisted with the model and
replayed at prediction, plug-in and posterior-mean alike. The posterior
mean integrates the residual coefficients' uncertainty jointly with the
surfaces to first order (the complete gradient of `η`, the anchor moved
through `b̃ᵀΣb̃`): the anchor now reads the whole coefficient vector, not
`(q, b)` alone, so the exact bivariate integration of the score-only model
does not apply. Adding `βᵀr` without moving
the anchor would leak `Var(βᵀr | a)` into the baseline — the unconditioned-
score defect in another coat. With the anchor holding along every
parameter path the baseline and predictor-shape directions are
Fisher-orthogonal under the declared law
(`Descent.Portability.MarginalAnchor.crossInformation_baseline_shape_zero`),
which is what makes `β` estimable without corrupting `q`.

The default latent-law certificates read this joint anchor through the score
(gam#2985). Given the score, the fit models `βᵀr | z ~ N(u z, v − u²)` with
`u = βᵀγ(a)`, `v = βᵀΣ_rr(a)β`, so at the row's index intercept `α` the
residual integrates out: `E_{r|z}[Φ(α + s(g z + βᵀr))] = Φ(ã + B z)` with
`B = s(g + u)/τ`, `ã = α/τ`, `τ = √(1 + s²(v − u²))`. Under any law of the
score the joint anchor is the score-only anchor at the row's `(ã, B)`, so the
closed form's excess anchoring loss and the moving-law certificate run on it
unchanged, and a fit with the block is certified like one without it.

### What the fit checks and refuses

- Every residual column is tested for `E_w[r | marginal-index span] = 0`
  with the same robust Rao score test the score's conditional gate uses, at
  the same level, **including the level**. A column that fails is refused
  with a typed reason; the fit never centres a feature for you, because a
  level absorbed into the baseline is a different model.
- The block is lowered through the rigid row kernel with the coefficients
  as row primaries, so it cannot be combined with `linkwiggle(...)`
  score-warp / link-deviation blocks, a learned frailty scale, or an
  absorbed CTN influence block. Each of those is a typed refusal, not a
  silent reinterpretation. Spatial length scales are held at their seeded
  values in the presence of the block; pass `length_scale=` to choose them.
- When the score's conditional law moves and the conditional location-scale
  calibration fires, the calibrated score is a generated regressor, and the
  block reads it three ways: through each row's own `ζ_i`, through the pooled
  joint covariance (`γ = c/a`, `Σ_rr` and the anchor's `u`, `v` are weighted
  moments of `ζ`, so every `ζ_j` moves every row's anchor), and, on a
  global-empirical law, through the grid built from `ζ`. The Murphy–Topel
  correction carries all three (gam#2985) under the standard-normal and
  global-empirical laws. When the joint covariance escalates to the
  conditional `Σ(a)`, its regressions are an M-estimate fitted on the
  calibrated score whose implicit derivative is not implemented, so the fit
  publishes its point estimates and certificate and withholds the
  coefficient covariance, recording why
  (`CovarianceDeclined::BmsGeneratedRegressorResidualRepairChannelUnavailable`).
  An uncorrected covariance would be too narrow.

### On a declared finite law of the score

When the score's law is declared as a finite law (`latent_measure=
"global-empirical"`, or the gate's empirical fallback, global or local), the
residual block keeps its Gaussian law given the score, `r | z, a ~ N(γz,
Σ_{r·z})`, read off the same persisted joint covariance (its first column and
Schur complement). Given node `z_k` the drive `s(g z + βᵀr)` is then
`N(m z_k, v)` with `m = s(g + βᵀγ)` and `v = s²βᵀΣ_{r·z}β` — a finite mixture
of Gaussians — and the anchor is the finite-law probit anchor on the scaled
coordinates:

```text
Σ_k w_k Φ(ã + B·z_k) = Φ(q) ,      α = τ·ã ,   B = m/τ ,   τ = √(1 + v) .
```

On a Gauss–Hermite law it is the closed form above to quadrature tolerance.
The fit differentiates the root through fourth order, so the REML/LAML outer
derivatives read the same anchor the plug-in prediction replays.
- There is no width ceiling. The row likelihood reads `β` only through
  `t = βᵀr`, `u = βᵀγ` and `v = βᵀΣ_rrβ`, so the row program has five
  primaries whatever the width, and the Hessian channels add the constant
  curvature `2Σ_rr` of `v` to the design pullback: `O(nK²)` assembly.

## Fixed external baseline (slope-only fit)

Both marginal-slope families route the two offset columns onto their two
predictors: `offset=` is added to the marginal (baseline) predictor and
`noise_offset=` to the slope predictor. Under `bernoulli-marginal-slope`
the marginal predictor is the probit index of the marginal prevalence, so
a baseline known from outside the data — a published prevalence
`prev(x)` — enters as `Φ⁻¹(prev)` in an offset column against an
intercept-only marginal formula, leaving the slope surface free:

```python
import numpy as np
import pandas as pd
import gamfit
from scipy.stats import norm

rng = np.random.default_rng(0)
n = 500
df = pd.DataFrame({"x": rng.uniform(0, 1, n), "z": rng.normal(0, 1, n)})
df["prev"] = 0.1 + 0.2 * df["x"]                                  # published prevalence prev(x)
b = 0.5 + 0.5 * df["x"]                                           # true slope surface
p = norm.cdf(norm.ppf(df["prev"]) * np.sqrt(1 + b**2) + b * df["z"])
df["case"] = (rng.uniform(size=n) < p).astype(float)

df["baseline"] = norm.ppf(df["prev"])
model = gamfit.fit(
    df,
    "case ~ 1",
    family="bernoulli-marginal-slope",
    z_column="z",
    slope_formula="s(x)",
    offset="baseline",
)
```

The intercept absorbs any constant miscalibration of the external
baseline. `noise_offset=` plays the same role on the slope side: a known
slope component held fixed while the rest of `slope_formula=` is
estimated. Under `survival-marginal-slope` the same two columns offset the
threshold predictor and the slope predictor respectively. CLI equivalents
are `--offset-column` and `--noise-offset-column`.

## Frailty in marginal-slope survival

Survival marginal-slope supports no frailty, or
`frailty_kind="gaussian-shift"` with a fixed `frailty_sd`.
`"hazard-multiplier"` and a learnable gaussian-shift sigma are rejected
at fit time. With the default slope (an intercept in every slope surface and
no offset, or a constant one), a learnable sigma is rejected because the
likelihood does not identify it: it reads the frailty only as the observed
slope `s(σ)·g`, `s(σ) = 1/√(1+σ²)`, so rescaling the slope undoes any change of
σ and the data cannot tell two values apart (gam#2938). A fixed `frailty_sd`
only rescales the reported slope.

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 600
df = pd.DataFrame({"age": rng.uniform(40, 70, n), "family_id": np.arange(n) // 2})
z = rng.normal(0, 1, n)
df["raw_score"] = 0.02 * (df["age"] - 55) + np.exp(0.3 * z)
t = rng.exponential(10 * np.exp(-0.04 * (df["age"] - 55) - 0.5 * z + rng.normal(0, 0.3, n)))
c = rng.uniform(2, 25, n)
df["entry"], df["exit"], df["event"] = 0.0, np.minimum(t, c), (t <= c).astype(float)

gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age)",
    survival_likelihood="marginal-slope",
    slope_formula="s(age)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="raw_score", covariates="s(age)",
        group_column="family_id", folds=2,
    ),
    frailty_kind="gaussian-shift",
    frailty_sd=0.3,
)
```

## Detecting marginal-slope models after loading

```python
import numpy as np
import gamfit
from scipy.stats import norm

rng = np.random.default_rng(0)
age, z = rng.uniform(40, 70, 400), rng.normal(0, 1, 400)
df = {"age": age, "z": z, "case": (rng.uniform(size=400) < norm.cdf(0.03 * (age - 55) + 0.7 * z)).astype(float)}
model = gamfit.fit(df, "case ~ s(age)", family="bernoulli-marginal-slope", z_column="z", slope_formula="s(age)")

model.save("model.gam")            # a model saved earlier
model = gamfit.load("model.gam")
model.is_marginal_slope            # True if a marginal-slope family
model.is_survival                  # True if survival
model.is_transformation_normal     # True if a Stage 1 calibration model
model.model_class                  # full class string
```

## Notes

- Supply `transformation_normal_stage1=` to condition the score on
  covariates inside the fit (the calibrated chain). Use a raw `z_column=`
  only for a score already conditionally `N(0, 1)` from outside this
  pipeline.
- Set `scale_dimensions=True` when calibrating on a handful of PCs so
  anisotropic length scales are learned per axis.
- Posterior sampling: Bernoulli marginal-slope and transformation-normal
  models use the Gaussian Laplace approximation in `Model.sample(...)`;
  survival marginal-slope uses NUTS over the joint coefficient vector.
  See [posterior-sampling.md](posterior-sampling.md).
- Predict output: Bernoulli marginal-slope returns a 1-D probability
  array by default. Pass `id_column=` or `return_type="dict"` for a
  table. The same applies to transformation-normal models.

## End-to-end example

```python
import gamfit
import numpy as np
import pandas as pd

n = 1000
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "PGS":  rng.normal(0, 1, n) + 0.3 * rng.normal(0, 1, n),
    "pc1":  rng.normal(0, 1, n),
    "pc2":  rng.normal(0, 1, n),
    "pc3":  rng.normal(0, 1, n),
})
risk = -1.1 + 0.8 * df["PGS"] + 0.4 * df["pc1"]
df["disease"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-risk))).astype(float)
df["family_id"] = np.arange(n)  # This illustrative sample is unrelated.

# Condition the score on the PCs and fit the slope surface in one
# cross-fitted predictive call.
model = gamfit.fit(
    df,
    "disease ~ matern(pc1, pc2, pc3, centers=20)",
    family="bernoulli-marginal-slope",
    slope_formula="matern(pc1, pc2, pc3, centers=20)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="PGS",
        covariates="matern(pc1, pc2, pc3, centers=20)",
        group_column="family_id", folds=2,
    ),
    scale_dimensions=True,
)

test = df.head(50).copy()
probs = model.predict(test)
```
