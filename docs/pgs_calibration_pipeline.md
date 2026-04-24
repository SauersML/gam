# Polygenic score calibration pipeline

A two-stage fitting protocol for polygenic scores (PGS) that separates
ancestry-conditional distributional drift from outcome-conditional risk
miscalibration. Every layer carries an identifiable parametric anchor, so zeroing
the corresponding coefficient block collapses the model to a familiar textbook
baseline. The library composes: conditional Gaussianization on principal
components (PCs), then a disease model whose score slope, inverse link, and
baseline hazard are each learnable perturbations around an anchored null.

## 1. Motivation

Polygenic scores deployed across diverse cohorts fail in two distinct ways, and
conflating them is the source of most published calibration debates.

**Score-distribution drift** is the marginal failure. The raw PGS distribution
is not invariant across ancestry: mean, variance, and higher moments shift with
principal components, so a quantile in one subpopulation does not correspond to
the same relative risk position in another. Naive rank transforms within a
pooled sample discard ancestry-conditional structure; naive within-stratum
standardisation requires discrete ancestry labels that are a poor proxy for a
continuous genetic background. The first failure is distributional.

**Disease-risk miscalibration** is the conditional failure. Even if the PGS is
perfectly standardised conditional on ancestry, the map from a standardised
score to disease probability, hazard, or survival may itself depend on
ancestry, age, and nonlinearities in the link. A slope learned on one cohort
may under- or over-transmit risk in another. A proportional-hazards assumption
can fit the average population while missing a late-life divergence that
matters clinically. The second failure is about the outcome model.

The pipeline treats these as separable problems. Stage 1 produces
`PGS_cal`, a polygenic score that is standard normal conditional on a
continuous PC coordinate by construction. Stage 2 uses `PGS_cal` as an anchor
and learns an outcome model whose identity element is a fully parametric,
reproducible null: linear probit slope for binary outcomes and a Gompertz-Makeham
proportional-hazards baseline for survival. Departures from each anchor are
penalised perturbations rather than unconstrained flexibility, so clinical
interpretability survives non-linear fitting.

## 2. Mathematical framework

The pipeline is built from five ingredients. Each one has a null coefficient
value at which the fitted object reduces to an identifiable parametric object.

**Conditional transformation model.** For a scalar PGS `Y` and PC covariate `x`,
fit a monotone map `h(Y | x)` such that `h(Y | x) ~ N(0, 1)`. The
response-direction basis is `[1, Y, anchored B-spline deviations]`, tensored
with a covariate design operator over PC space. The deviation B-splines are
projected so that their value and first derivative vanish at the response
median; zero deviation coefficients therefore recover an exact affine
(location-scale) Gaussianisation. Monotonicity is enforced by explicit
derivative lower-bound constraints on a response grid plus the natural
`log(h'(Y))` barrier in the change-of-variables density
$$
\ell_i = -\tfrac{1}{2} h_i^2 + \log h'_i.
$$
**Anchor.** Zero transformation-normal deviation coefficients give
$h(Y \mid x) = a(x) + b(x)\,Y$, i.e. affine Gaussianisation.

**Duchon radial basis on PC space.** Multivariate radial-basis spline with
triple operator regularization on the final function. This is not the native
spectral Duchon seminorm. For the hybrid Duchon-Matern kernel, the radial
kernel spectrum and partial-fraction coefficients are
$$
\frac{1}{\rho^{2p}(\kappa^2 + \rho^2)^{s}}
= \sum_{m=1}^{p} \frac{a_m}{\rho^{2m}} + \sum_{n=1}^{s} \frac{b_n}{(\kappa^2 + \rho^2)^{n}},
$$
implemented in `duchon_partial_fraction_coeffs`. The null space is the
polynomial block of total order `m` over the PC dimensions, constructed by
`polynomial_block_from_order`. **Anchor.** Zero Duchon deviation coefficients
leave only the polynomial null space, i.e. a linear (or low-order polynomial)
function of PCs.

**linkwiggle inverse-link deviation.** Anchored monotone deviation from the
base link $g^{-1}$. Internally a cubic I-spline in the standardised linear
predictor `z = q + h` whose coefficients are constrained to a moment-anchor
null space so that the first two moments of the fitted deviation against the
standard normal reference density vanish. **Anchor.** Zero linkwiggle
coefficients recover the chosen base link exactly (probit in the
marginal-slope path).

**Score-warp logslope deviation.** In marginal-slope families, the
`--logslope-formula` carries a secondary Duchon + linkwiggle block that is
routed into an anchored monotone warp of the exposure `z`. **Anchor.** Zero
score-warp coefficients reduce the warp to the identity, producing a linear
log-slope in `z` with PC-dependent offset but no nonlinearity in the score.

**timewiggle time-axis deviation.** For survival outcomes, a cubic
I-spline in time adds an anchored monotone deviation on top of a parametric
baseline cumulative hazard $H_0(t)$. **Anchor.** Zero timewiggle coefficients
leave only the parametric baseline $H_0(t)$, chosen via `--baseline-target`
(`linear`, `weibull`, `gompertz`, or `gompertz-makeham`).

The composition is identifiability-preserving: each block's null is a distinct
parametric object, the deviation bases are linearly independent of their
anchors, and the penalised likelihood collapses to a fully parametric model
when all deviation coefficients are zero.

## 3. CLI walkthrough

Input data `biobank.csv` has columns `PGS`, `pc1`..`pc4`, `disease`,
`age_entry`, `age_exit`, `event`.

**Stage 1: conditional Gaussianisation.** Fit `h(PGS | PCs) ~ N(0, 1)` with a
Duchon smooth over the four PCs, then predict to append `pgs_ctn_z`:

```bash
gam fit biobank.csv \
  'PGS ~ duchon(pc1, pc2, pc3, pc4, centers=24, order=1, power=1, length_scale=1.0)' \
  --transformation-normal \
  --scale-dimensions \
  --out stage1.gam

gam predict stage1.gam biobank.csv --out biobank_cal.csv
```

The produced CSV adds a calibrated z-score column corresponding to
`h(PGS | PCs)`. The Duchon `centers=24` picks the radial-basis centers; `order`
and `power` are the polyharmonic exponents (`p`, `s` in the spectral
decomposition above); `length_scale=1.0` opts into the hybrid
Duchon-Matern kernel so anisotropic PC scaling from `--scale-dimensions` is
identifiable.

**Stage 2a: binary marginal-slope.** With `pgs_ctn_z` as the exposure anchor,
fit a probit marginal-slope model. The Bernoulli marginal-slope path is
triggered when both `--logslope-formula` and `--z-column` are supplied:

```bash
gam fit biobank_cal.csv \
  'disease ~ z + duchon(pc1, pc2, pc3, pc4, centers=24, order=1, power=1) + linkwiggle(degree=3, internal_knots=10)' \
  --logslope-formula 'duchon(pc1, pc2, pc3, pc4, centers=24, order=1, power=1) + linkwiggle(degree=3, internal_knots=10)' \
  --z-column pgs_ctn_z \
  --out stage2a.gam
```

Inside the marginal formula, `linkwiggle(...)` routes to the anchored
inverse-link deviation block. Inside `--logslope-formula`, `linkwiggle(...)`
routes to the anchored score-warp block. The base link is probit by default
for this family.

**Stage 2b: survival marginal-slope.** `Surv(...)` on the left selects the
survival path; `--survival-likelihood marginal-slope` requests the anchored
probit-slope formulation; `--baseline-target gompertz-makeham` picks the
parametric anchor:

```bash
gam fit biobank_cal.csv \
  'Surv(age_entry, age_exit, event) ~ z + duchon(pc1, pc2, pc3, pc4, centers=24, order=1, power=1) + linkwiggle(degree=3, internal_knots=10) + timewiggle(degree=3, internal_knots=8)' \
  --survival-likelihood marginal-slope \
  --baseline-target gompertz-makeham \
  --baseline-rate 0.08 \
  --baseline-makeham 0.015 \
  --logslope-formula 'duchon(pc1, pc2, pc3, pc4, centers=24, order=1, power=1) + linkwiggle(degree=3, internal_knots=10)' \
  --z-column pgs_ctn_z \
  --time-basis ispline --time-degree 3 --time-num-internal-knots 8 \
  --out stage2b.gam
```

`timewiggle(...)` adds an anchored monotone deviation to the cumulative
hazard; with zero coefficients the fit reduces to the Gompertz-Makeham
parametric baseline supplied by `--baseline-rate` and `--baseline-makeham`.

## 4. Python walkthrough

The Python surface mirrors the CLI. Stage 1 has a dedicated helper
`gam.pgs.PgsCalibration` that encodes the Duchon defaults; Stages 2a and 2b
call `gam.fit` directly with the same kwargs the CLI exposes.

```python
import pandas as pd
import gam
from gam.pgs import PgsCalibration

df = pd.read_csv("biobank.csv")
pc_columns = ["pc1", "pc2", "pc3", "pc4"]
```

**Stage 1.** `PgsCalibration` builds the formula
`PGS ~ duchon(pc1, ..., pc4, centers=..., order=1, power=1, length_scale=1.0)`
and forwards `transformation_normal=True` and `scale_dimensions="auto"` to
`gam.fit`:

```python
calibration = PgsCalibration(
    pc_columns=pc_columns,
    pgs_column="PGS",
    duchon_centers=len(pc_columns) + 20,
    out_column="pgs_ctn_z",
).fit(df)

df_cal = calibration.transform(df)   # adds pgs_ctn_z column
```

`calibration.predict(df)` returns the raw numpy array; `transform(df)` returns
a copy of the input with the calibrated column appended, preserving
DataFrame / pyarrow / dict input types.

**Stage 2a.** Bernoulli marginal-slope via `gam.fit`. The probit link is the
default for this family, and the main/logslope `linkwiggle(...)` terms are
routed internally:

```python
pc = "duchon(pc1, pc2, pc3, pc4, centers=24, order=1, power=1)"
disease_model = gam.fit(
    df_cal,
    f"disease ~ z + {pc} + linkwiggle(degree=3, internal_knots=10)",
    family="bernoulli-marginal-slope",
    z_column="pgs_ctn_z",
    logslope_formula=f"{pc} + linkwiggle(degree=3, internal_knots=10)",
)

probs = disease_model.predict(df_cal, return_type="dict")["mean"]
```

**Stage 2b.** Survival marginal-slope. `family="survival"` with
`survival_likelihood="marginal-slope"` plus `baseline_target="gompertz-makeham"`
picks the anchored hazard formulation. The predict call returns a
`SurvivalPrediction` whose `hazard_at`, `cumulative_hazard_at`, and
`survival_at` methods evaluate the fitted surface on a caller-supplied time
grid:

```python
import numpy as np

surv_main = (
    "Surv(age_entry, age_exit, event) ~ z + "
    f"{pc} + linkwiggle(degree=3, internal_knots=10) + "
    "timewiggle(degree=3, internal_knots=8)"
)
surv_model = gam.fit(
    df_cal,
    surv_main,
    family="survival",
    survival_likelihood="marginal-slope",
    baseline_target="gompertz-makeham",
    baseline_rate=0.08,
    baseline_makeham=0.015,
    z_column="pgs_ctn_z",
    logslope_formula=f"{pc} + linkwiggle(degree=3, internal_knots=10)",
)

pred = surv_model.predict(df_cal)
ages = np.linspace(50.0, 85.0, 36)
S = pred.survival_at(ages)            # shape (n, 36)
```

Fitted models serialise with `model.save(path)` / `gam.load(path)`.

## 5. Interpretation guide

A calibrated z-score near zero means the PGS is at the ancestry-conditional
median; $\pm 1$ means one ancestry-conditional standard deviation above or
below. Because Stage 1 enforces standard-normal residuals given PCs,
`pgs_ctn_z` is directly comparable across subpopulations without
stratification.

For binary outcomes, the Stage 2a fitted probability is the posterior mean
under the anchored probit slope. Zeroing the linkwiggle coefficients reduces
the fitted function to the pure probit curve; zeroing the score-warp reduces
the log-slope to a PC-dependent constant; zeroing the Duchon main-effect
removes ancestry-dependent intercept. Reading the three deviation blocks in
isolation localises where ancestry-conditional nonlinearity lives.

For survival, the returned `SurvivalPrediction` exposes hazard, cumulative
hazard, and survival on any caller-chosen time grid. With zero timewiggle the
hazard is exactly Gompertz-Makeham at the configured rate and Makeham
additive floor. Nonzero timewiggle coefficients quantify the departure from
proportional hazards; the anchor means every post-hoc diagnostic still has a
closed-form parametric reference. A useful summary for clinical reporting is
to evaluate survival at a fixed horizon (say 10 years past `age_entry`) both
under the fitted model and under the parametric anchor with deviations held
at zero; the difference decomposes the risk attributable to ancestry-conditional
nonlinearity, score-warp, and time-axis departure from proportionality.

## 6. Current limitations

The Python FFI currently plumbs `transformation_normal`, binary and survival
marginal-slope fits (via `family="bernoulli-marginal-slope"` and
`family="survival"` with `survival_likelihood="marginal-slope"`), and
standard GLM fits. Survival location-scale fits are not yet exposed through
Python; use the CLI for that path. Frailty kwargs
(`frailty_kind`, `frailty_sd`, `hazard_loading`) and the
`predict_noise` / `noise_offset_column` secondary formulas are supported on
the CLI and exposed through the CLI flags documented above, but do not yet
have dedicated Python kwargs. Uncertainty intervals from `gam predict
--uncertainty` are CLI-only pending the Python predict-intervals surface.

Biobank-scale continuous integration is in progress: the `biobank_sim.py`
400 K-sample synthetic cohort is exercised in `bench/`, but end-to-end
400 K-row calibration + disease-fit timing gates are not yet part of the
required CI suite. Planned next: FFI coverage for
location-scale survival fitting, native Python `predict` interval support,
and a packaged biobank simulation entry point.

## 7. References

- Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in
  Sobolev spaces. *Constructive Theory of Functions of Several Variables*,
  Springer Lecture Notes in Mathematics 571, 85-100.
- Rigby, R. A. and Stasinopoulos, D. M. (2005). Generalized additive models
  for location, scale and shape (with discussion). *Journal of the Royal
  Statistical Society: Series C*, 54(3), 507-554.
- Hothorn, T., Mo\"ost, L. and B\"uhlmann, P. (2014). Most likely
  transformations. *Scandinavian Journal of Statistics*, 45(1), 110-134.
- Royston, P. and Parmar, M. K. B. (2002). Flexible parametric
  proportional-hazards and proportional-odds models for censored survival
  data, with application to prognostic modelling and estimation of treatment
  effects. *Statistics in Medicine*, 21(15), 2175-2197.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*,
  2nd edition. Chapman and Hall/CRC.
- Ramsay, J. O. (1988). Monotone regression splines in action. *Statistical
  Science*, 3(4), 425-461.
