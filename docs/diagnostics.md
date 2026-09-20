# Diagnostics, summaries, plots, reports

A fitted `Model` exposes six inspection methods:

| Method | Returns | Contents |
| --- | --- | --- |
| `summary()` | `Summary` | Formula, family/link name, model class, deviance, REML/LAML score (in the `reml_score` field), per-coefficient table, smoothing parameters (`lambdas`), group metadata, and deployment extensions. |
| `basis_check(data)` | `list[dict]` | Per-smooth basis-adequacy report: is each smooth's basis rich enough for the function it was asked to represent? |
| `diagnose(data)` | `Diagnostics` | Observed values, predicted columns, residuals, and aggregate metrics for point-payload models (the point column is the one the model's class publishes: `posterior_mean`, or `mean` for the transformation-normal and Bernoulli marginal-slope classes). |
| `check(data)` | `SchemaCheck` | Schema validation result with structured issues. |
| `plot(data, x=, kind=)` | `matplotlib.axes.Axes` | Prediction / residual / observed-vs-predicted plot. |
| `report(path=None)` | `str` | Self-contained HTML report (string, or written path). |

`reml_score` and `raw_reml_score` are `None` when the fit has **no**
criterion, which is a different statement from "not recorded". A Gaussian
fit whose fitted mean reproduces the response to floating-point resolution
has `sigma_hat = 0`, so its restricted likelihood is unbounded and every
score derived from it — the comparable REML/LAML headline, `Summary.aic_corrected`,
`Model.evidence_ratio_vs`, `gamfit.compare_models` — is undefined rather than
large. `Summary.reml_score_unavailable` then carries the explanation, and
those ranking surfaces raise it instead of ranking a stand-in value. Compare
such a model on predictive accuracy, or refit on data whose response is not
an exact function of the design.

`gamfit.validate_formula(...)` validates a formula and data against the
parser and schema without fitting.

Research/inference instruments used by the SAE and structure-discovery
workflows live in submodules: `gamfit.sae` has `split_likelihood_log_e`,
`e_bh_dictionary_certificate`, `log_e_from_p_value`,
`select_probe_by_expected_evidence`, `expected_resolution_budget`, and
`plan_probe_for_contested_claim`; `gamfit.inference` has
`lawley_bartlett_factor` and `glm_full_conformal`. These are low-level
building blocks rather than `Model` methods; see the
[API reference](api-reference.md) for signatures. (`debiased_functional`, by
contrast, is a `Model` method.)

## summary()

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train_df, "y ~ s(x)")

s = model.summary()
print(s)                       # text repr; HTML in notebooks
s["formula"]
s["family_name"]
s["model_class"]
s["deviance"]
s["reml_score"]
s["scale"]                     # dispersion phi-hat (Gaussian sigma^2)
s["convergence"]               # certificate incl. outer/inner iteration counts
s["coefficients"]              # list of dicts (per-term records)
s.coefficients                 # same list via property
s.to_dict()                    # full payload as a dict
s.coefficients_frame()         # pandas.DataFrame; requires pandas
```

`Summary` supports `__getitem__` and `get(key, default)` like a dict.

`model.smoothing_parameters()` returns a `{penalty_index: lambda}` dict of
the fitted smoothing/precision parameters by penalty index (via a dedicated
FFI call), the same values surfaced under `summary()["lambdas"]`.

### Fit notes, warnings, and solver logs

A fit records two kinds of notes, both listed in `model.notes` and
`summary().notes` (and printed under `Notes:` in the text summary):

- **advisories** — the fitted model differs from the literal request (a `k`
  capped to the covariate's distinct values, a basis too small for the
  residuals). These are also raised as `gamfit.errors.GamInferenceWarning`, attributed
  to your calling line; the CLI prints them to stderr.
- **informational notes** — a default the engine chose for you, such as the
  internal-knot count of a default `s(x)`. These are never warned.

A default fit writes nothing to stdout or stderr. The engine's solver trace
(`[OUTER …]`, `[PIRLS …]`, …) goes to the `gamfit` Python logger at `DEBUG`
(finer records below that), which is silent unless you opt in:

```python
import logging
logging.basicConfig()
logging.getLogger("gamfit").setLevel(logging.DEBUG)
```

The CLI equivalent is `gam -v …` (`-vv` for the finer records).

### Shape-constrained smooths have no significance p-value

A smooth with `shape=monotone_increasing` (or `monotone_decreasing`, `convex`,
`concave`) reports `edf` and `ref_df` in `summary().smooth_terms` but no
`chi_sq` or `p_value`. The row carries `p_value_unavailable =
"shape_constrained"` instead, and `model.smooth_significance(data)` returns the
same reason in place of an LR row. The printed summary (Python and CLI) names
the reason under the smooth table.

Why the number is withheld:

- The null `f = 0` is the apex of the constraint cone. Under a flat truth the
  estimator sits on the cone boundary, so neither the Wald χ² nor the LR's
  spectral reference describes the statistic's null law.
- Chi-bar-square (a mixture of χ² laws weighted by the cone's face
  probabilities) and tests conditional on the active set are the textbook fixes.
  Both are the null law of the **cone projection** with a fixed cone. The
  coefficients here are the **truncated posterior mean**, which lies strictly
  inside the cone and has no active set. Its λ is selected by REML on the same
  data. Neither reference applies.

`basis_check` still reports for these terms. It tests structure outside the
term's column span, which the cone does not restrict: its score is built from
enrichment columns made orthogonal (in the working weights) to the whole
design, so for a Gaussian identity fit the score equals the enrichment
projection of `y` and does not depend on the shape term's coefficients at all.
The fit enters only through the dispersion estimate, as it does for an
unconstrained term. For other families the score uses the fitted mean, which
is consistent under the null for the truncated posterior mean as it is for the
unconstrained one.

## basis_check() — is the basis big enough?

A converged, `certified` fit says nothing about whether the basis it was given
can represent the function it was asked to model. Both a smooth whose basis
spans the truth and a smooth that cannot reach it converge, certify, and report
a per-term EDF that is some fraction of the term's column count. What separates
them is whether the **residuals still carry structure in that smooth's own
covariates**.

That is what `basis_check` measures, and what every fit now measures for itself:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
pcs = rng.normal(0, 1, (400, 4))
dose = rng.uniform(0, 2, 400)
eta = -0.5 + 0.8 * dose + np.sin(pcs[:, 0]) + 0.5 * pcs[:, 1] * pcs[:, 2]
data = {"dose": dose, **{f"pc{j + 1}": pcs[:, j] for j in range(4)},
        "case": (rng.uniform(size=400) < 1 / (1 + np.exp(-eta))).astype(float)}

model = gamfit.fit(data, "case ~ dose + duchon(pc1, pc2, pc3, pc4, centers=24)",
                   family="binomial")
# 1. when the basis is too small, the fit already told you, as a GamInferenceWarning:
#    "basis adequacy: smooth 'duchon(...)' has <k> coefficient columns
#     (<m> unpenalized), and the fit's residuals still carry structure in its
#     covariates that this basis cannot represent (lack-of-fit p = <p> ...)"

model.summary().basis_checks     # the same evidence, no data, no refit
model.basis_check(data)          # recomputed from the training rows
```

Each row carries

| Field | Meaning |
| --- | --- |
| `basis_dim` | The realized coefficient width `k'` of the term. |
| `nullspace_dim` | The dimension of its **joint** penalty null space — the directions no penalty touches — so `basis_dim - nullspace_dim` is the penalizable capacity. `0` for every double-penalized smooth, which includes the whole radial family. |
| `edf` | The term's effective degrees of freedom, carried here so it can be read against the two above. |
| `enrichment_dim` / `enrichment_rank` | Width of the higher-resolution alternative the residuals were tested against, and how many of its directions survived projecting the fitted design out. The rank is the test's reference degrees of freedom. |
| `statistic` / `p_value` | The penalized score (Rao) lack-of-fit test. A small `p_value` means the basis is too small. |
| `provenance` | `"radial_enrichment"` when a test ran, else the NAME of the evidence that was missing. |

`p_value` is present exactly when a test ran, so "adequate" and "not measured"
are never confusable.

!!! warning "EDF-vs-`k'` misleads on a radial smooth"

    A `d`-dimensional radial smooth is *double penalized*: an RKHS curvature
    penalty plus a complementary trend ridge on its `d + 1`-column polynomial
    block. Nothing in it is completely unshrunk, so `nullspace_dim` is `0` — but
    the polynomial block is only WEAKLY penalized, and on a 16-D
    `duchon(..., centers=24)` those 17 of 23 columns carry most of the term's
    effective dimension. The EDF column then reads near-saturated on a fit whose
    problem is the *span* of its basis, not its rank. That is the reading
    `p_value` exists to replace: it asks whether the residuals carry structure
    the column space cannot reach, which no EDF number can answer.

### What the test is, and what it deliberately ignores

The alternative is a Duchon kernel at data-driven centers over the term's
standardized covariates, orthogonalized against the fitted design **in the fit's
own IRLS weight metric**. That projection is the whole design of the statistic:
a penalized fit is biased, and its shrinkage bias lives entirely inside the span
of the fitted design, so projecting it out makes the test blind to "λ is large"
and sensitive only to structure the design *cannot represent at all*.

For a Gaussian response the score is exactly normal and the `χ²`/`F` reference
is exact. For a canonical binomial (logit) or Poisson (log) fit it is only first
order, and at small `n` it is visibly off (on `n = 200` binomial rows it was
conservative: size `0.032` at `0.05`, which is as much a miscalibration as an
anti-conservative test). There the score is evaluated at the unpenalized null
MLE and referred to its law **conditional on the sufficient statistic**
`Xᵀ(w∘y)`, which does not depend on the nuisance coefficients at all; its mean,
covariance and fourth cumulant are corrected to `O(1/n)`. Where that expansion
leaves its range of validity — high-leverage rows at an extreme fitted mean —
the row reports `provenance = "conditional_reference_unavailable"` and no
`p_value` (about 7% of null replicates of a default `s(x)` binomial fit at
`n = 200`; none at `n = 2000`). `"null_fit_unavailable"` means the null MLE
itself could not be certified. The refused share grows as events get rarer:
15–19% of `n = 200` rows at Poisson means or success probabilities around
`0.1–1`, and two-thirds with 28–34 expected events in 200 rows. In that
last regime the p-values that are reported are **not calibrated** (Poisson:
size `0.030` at `0.05`, KS `p = 2e-4` against uniformity); treat them as
unmeasured. The conditional law reads prior weights as frequency weights (as
the likelihood does), and its calibration was measured on designs of about 10
columns; much wider designs are tested on a row sample capped by the
reference's cost, a regime that has not been measured.

Asking whether a direction the basis HAS is being over-smoothed is a
smoothing-parameter question, and this report declines to answer it. It is also
conditional on the fitted `λ̂` and on the alternative, exactly as the `summary()`
Wald column is conditional on `λ̂`. A rejection says there is signal outside the
term's column span; it does not say how much of *your* estimand that signal
moves. **Refit with a larger basis for the flagged term and compare.**

### `certified` is about the optimizer, not about the model

`summary().convergence["certified"]` says the inner P-IRLS solve and the outer
smoothing-parameter search reached a certified stationary point at the
tolerances the mint gate applies. It makes **no** claim that the basis those
iterations converged on is rich enough, that the family is right, or that a
fitted adjustment removes the confounding it was given. A fit can be `certified`
and inadequate at the same time, and on a high-dimensional confounder adjustment
that combination produces a confidently significant false association. Read
`basis_checks` alongside it.

## diagnose()

```text
diag = model.diagnose(data, *, y=None, interval=0.95)
```

`diagnose` calls `predict` on the feature columns of `data` with the
given interval, then packages the result against the observed
response.

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Table-like input that includes the response column. |
| `y` | `None` | Response column name. Defaults to `model.response_name`. Required when the formula does not name a single response (e.g. survival `Surv(...)` formulas). |
| `interval` | `0.95` | Coverage probability forwarded to `predict`. Set `None` to skip interval columns. |

Returns a frozen `Diagnostics` dataclass:

| Field | Type | Meaning |
| --- | --- | --- |
| `formula` | `str` | Fitted formula. |
| `response_name` | `str` | Resolved response column. |
| `observed` | `list[float]` | Observed values. |
| `residuals` | `list[float]` | `observed - predicted[point_column]`. |
| `predicted` | `dict[str, list[float]]` | The predict-table columns (at least the point column; with `interval` also its `_lower` / `_upper` bands). |
| `point_column` | `str` | Name of the response-scale point series in `predicted`: `"posterior_mean"` for standard and location-scale fits, `"mean"` for the transformation-normal and Bernoulli marginal-slope classes. |
| `metrics` | `dict[str, int \| float]` | `n_obs`, `mae`, `rmse`, `bias`; adds `r_squared` when the response variance is positive. |
| `interval_lower`, `interval_upper` | `list[float] \| None` | Aliases for the `f"{point_column}_lower"` / `f"{point_column}_upper"` columns. |

`diagnose` raises `ValueError` when the response column cannot be
inferred or is missing from `data`. The point column it reads is the one
the model's class publishes through `predict(..., return_type="dict")`.

## check()

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train_df, "y ~ s(x)")
test_df = {"x": np.linspace(0.5, 9.5, 20)}

check = model.check(test_df)

if check.ok:
    preds = model.predict(test_df)
else:
    for issue in check.issues:
        print(issue.kind, issue.column, issue.message)
    check.raise_for_error()       # raises ValueError
```

`SchemaCheck.ok` is `True` when no issues were reported.
`bool(check)` returns `check.ok`. `SchemaIssue` has three fields:
`kind`, `column`, `message`; typical kinds include `missing_column`
and `schema_error`.

## validate_formula()

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
site = rng.choice(["a", "b", "c"], 300)
data = {"x": x, "site": site, "y": np.sin(x) + (site == "b") + rng.normal(0, 0.3, 300)}

v = gamfit.validate_formula(
    data,
    "y ~ s(x) + group(site)",
    family="auto",
    # additional parser/materialization kwargs are accepted
)
v["formula"]
v["model_class"]
v["family_name"]
v["response_column"]
v.supported_by_python      # bool
```

Returns a `FormulaValidation` dataclass that wraps the parsed payload.

Accepts these parser/materialization keyword arguments from `gamfit.fit`,
with the same semantics, and does no fitting:
`family`, `negative_binomial_theta`, `expectile_tau`, `offset`, `weights`,
`transformation_normal`, `transformation_normal_stage1`,
`survival_likelihood`, `survival_time_anchor`, `baseline_target`,
`baseline_scale`, `baseline_shape`, `baseline_rate`, `baseline_makeham`,
`z_column`, `residual_columns`, `link`, `slope_formula`, `frailty_kind`, `frailty_sd`,
`hazard_loading`, `scale_dimensions`, `firth`, `noise_formula`,
`noise_offset`, `flexible_link`, `config`.

The list is exact: `validate_formula_docs_list_is_the_signature` in
`tests/test_validate_formula_docs_kwargs_match_signature.py` fails if it
drifts from the signature in either direction. It deliberately excludes the
fit-only objects `constraints`, `latents`, `penalties`, `smooths`,
`precision_hyperpriors` and `response_geometry`, which validation refuses.

## plot()

```python
import matplotlib.pyplot as plt
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train_df, "y ~ s(x)")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
model.plot(train_df, kind="prediction",            ax=axes[0])
model.plot(train_df, kind="residuals",             ax=axes[1])
model.plot(train_df, kind="observed_vs_predicted", ax=axes[2])
plt.tight_layout()
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Held-out data, with the response column present (same requirements as `diagnose`). |
| `y` | `None` | Response column override. |
| `interval` | `0.95` | Wald-band coverage for `"prediction"`. Ignored for the other kinds. |
| `kind` | `"prediction"` | One of `"prediction"`, `"residuals"`, `"observed_vs_predicted"`. |
| `ax` | `None` | Existing axes; a fresh one is created via `plt.subplots()` when omitted. |

| `kind` | Contents |
| --- | --- |
| `"prediction"` | For a model with one feature column: the mean curve over it, shaded Wald band when `interval` is set, observed scatter overlay. |
| `"residuals"` | Residuals vs predicted mean with a horizontal zero line. |
| `"observed_vs_predicted"` | Observed vs predicted with a `y = x` reference line. |

Returns the `matplotlib.axes.Axes` drawn on. Raises `ValueError` for
unknown `kind`, or for `"prediction"` on data with more than one feature
column: the full model's prediction sorted by one column zigzags through the
others' values. Per-term curves are partial effects; draw them with
`model.plot_terms()` (see [partial-effects.md](partial-effects.md)). Requires
matplotlib (install `gamfit[plot]`).

## report()

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train_df, "y ~ s(x)")

model.report("report.html")       # writes the file and returns its path
html = model.report()             # returns the HTML string
```

The report is a self-contained HTML document containing the summary
table, coefficient table, and EDF-by-block table rendered by the
`gam-report` crate (`crates/gam-report/src/lib.rs`). Python `Model.report()` currently omits data-dependent
diagnostics and smooth plots. The HTML is also used as `Model._repr_html_`
for notebook display.

## Inspecting the model object

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train_df, "y ~ s(x)")

model.formula                   # str
model.family_name               # str, e.g. "Gaussian Identity"
model.model_class               # str, e.g. "standard", "survival marginal-slope"
model.is_survival               # bool
model.is_marginal_slope         # bool
model.is_transformation_normal  # bool
model.response_name             # str | None (None for Surv(...) or non-simple lhs formulas)
model.training_table_kind       # "pandas" | "polars" | "pyarrow" | "numpy" | "mapping" | "records" | "rows" | "unknown"
model.group_metadata            # dict | None, persisted per-group metadata
model.deployment_extensions     # tuple of dicts, no-refit group extensions
```

These are read-only properties.

## When something looks wrong

| Symptom | Try this |
| --- | --- |
| `diag.metrics["r_squared"]` low on training | A fixed basis may be too small for the function. A default `s(x)` grows its own basis, but an explicit `k`, a `by=` smooth and the tensor-product, cyclic, factor-smooth and Matérn bases do not. Run `basis_check(data)`; where it reports a fixed basis is inadequate, drop the `k=` on an `s(x)` or give the other bases a larger `k`. Or add interactions via `te(...)` / multi-d smooths. See [Choosing k](formulas.md#choosing-k). |
| `rmse` low on training, high on test | Lowering `k` is not the fix: REML already penalizes wiggliness the data do not support. Check for leakage between training and test rows, for a shift between them, and for terms that should not be in the model. |
| `diagnose()` raises about the response column | Pass `y="column_name"` explicitly. |
| `check()` reports `missing_column` | The prediction data is missing a required feature. |
| `predict` raises `SchemaMismatchError` | Run `check()` first to identify the offending column. |
| `predict` raises `PredictionError` | Prediction failed for a non-schema reason; inspect the exception and confirm the fitted model class and prediction mode are supported. |

See [exceptions.md](exceptions.md) for the full exception hierarchy.
