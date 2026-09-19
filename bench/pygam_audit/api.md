# Audit: Python API ergonomics and scikit-learn integration (gamfit vs pyGAM)

Environment: scratchpad/bvenv (python, sklearn 1.9.1, numpy 2.4.6, pandas 3.0.6, pyGAM 0.12.0, gamfit 0.1.267 wheel).
pyarrow is NOT installed in the venv (it is an optional extra); a private copy was installed in
`scratchpad/audit/api/pa_site` and used via `PYTHONPATH=pa_site` only for the DataFrame tests that need it.
All scripts: `scratchpad/audit/api/w1_basic.py ... w10.py`. check_estimator output: `check_reg.txt`, `check_clf.txt`
(stderr in `.err`). Run from `/tmp` (repo cwd shadows the wheel with a gamfit lacking `_rust`).

## Headline

* gamfit's sklearn wrapper actually works with modern sklearn (Pipeline, clone, cross_val_score, GridSearchCV,
  ColumnTransformer). pyGAM 0.12.0 is **broken** under sklearn 1.9 (no `__sklearn_tags__`), so on integration
  gamfit is already ahead.
* But three blockers stop it from being a drop-in: fitted models cannot be pickled, pandas input raises
  ImportError without pyarrow, and ~1/3 of fits with an irrelevant covariate die with an IntegrationError
  (the smoothing-parameter cubature fails when a smooth collapses to its null space).
* check_estimator: GAMRegressor 29 pass / 22 fail / 1 skip; GAMClassifier 20 pass / 34 fail / 1 skip.

## check_estimator summary (formula "s(x0)+s(x1)", `on_fail=None`)

| Cause | Checks | Class |
|---|---|---|
| pickle fails (`cannot pickle 'builtins._FittedModel'`) | check_estimators_pickle (x2) | F1 |
| null smooth leads to cubature IntegrationError | check_regressors_train (x3), check_regressors_int, check_estimators_dtypes, check_fit_idempotent, check_n_features_in (reg); check_estimators_dtypes, check_fit_idempotent (clf) | F3 |
| binary-only classifier | 14 clf checks (fit_score_takes_y, overwrite_params, dont_overwrite_parameters, fit_returns_self, readonly_memmap, n_features_in_after_fitting, dtype_object, f_contiguous, classifiers_classes, supervised_y_2d, sample_order_invariance, subset_invariance, dict_unchanged, fit2d_predict1d) | F9 |
| predict does not validate n_features_in_ (extra columns silently ignored) | check_n_features_in_after_fitting (reg), check_classifiers_train (x2) | F16 |
| sparse input: message does not say sparse unsupported | sparse_tag, sparse_array, sparse_matrix (both) | F17 |
| objects with `__array__` rejected | check_regressor_data_not_an_array, check_classifier_data_not_an_array | F17 |
| y shape (n,1) rejected | check_supervised_y_2d (reg) | F14 |
| unfitted GAMClassifier.predict raises AttributeError not NotFittedError | check_estimators_unfitted (clf) | F17 |
| message wording (continuous target, y=None, 0 features, 1 feature, 1 sample, complex) | classifiers_regression_target, requires_y_none, empty_data_messages, fit2d_1feature, fit2d_1sample, complex_data | F17 |
| tiny n=10 data fails outer certification | check_estimators_nan_inf (clean fit of 10x3 before NaN test), check_classifiers_train (blobs) | F3 (robustness axis) |

Many of the message-wording failures come from a fixed formula meeting sklearn's variable-width test data;
a `formula=None` default (F6) would clear most of them.

## Findings

### F1 BLOCKER bug: fitted models are not picklable
`pickle.dumps(GAMRegressor(formula="s(x0)").fit(X,y))` gives `TypeError: cannot pickle 'builtins._FittedModel' object`;
the same for `gamfit.fit(...)` Models and joblib.dump. pyGAM pickles fine. This breaks joblib model persistence,
`GridSearchCV(n_jobs>1)` result caching, mlflow, and dask.
Cause: `gamfit/_model.py:119-124` has `__slots__` holding `_prediction_model = rust_module().compile_model(bytes)`, and
`crates/gam-pyffi/src/model/model_ffi.rs:79` `#[pyclass(name="_FittedModel", frozen)]` has no pickle support.
Fix: `Model.__reduce__` returns `(Model._from_bytes, (self._model_bytes, self._training_table_kind))`; do the same for
MultinomialModel (`_model.py:1405`). The model already serialises to bytes (`dumps()` at `_model.py:1068`), so this
adds no new format. Size S.

### F2 BLOCKER bug: pandas DataFrame input fails without pyarrow
With pandas 3.0.6 and no pyarrow (pyarrow only appears in the `pandas` extra, `pyproject.toml:72`; README says
`pip install gamfit`), `gamfit.fit(df, "y ~ s(x)")`, `Model.predict(df)`, and `GAMRegressor.predict(df)`/`score(df)` raise
`ImportError: Import pyarrow failed`, even for an all-float DataFrame. (GAMRegressor.fit(df) happens to work.)
Cause: `gamfit/_tables.py:108-124` takes the `__arrow_c_stream__` path whenever the attribute exists. pandas 3 always
has the attribute but needs pyarrow to run it.
Fix: take the arrow path only when the import succeeds, and otherwise fall back to the existing column path
(`table_columns`, `_tables.py:354`). This is an external-software boundary, so SPEC 8 allows it. The alternative is
to make pyarrow a hard dependency. Size S.

### F3 BLOCKER bug: null (irrelevant) smooths fail the fit about 1/3 of the time
`w9_null.py`: y is pure noise, X is 200x2 normal, `gamfit.fit(d, "y ~ s(x0)+s(x1)")` gives
`IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain` for 4 of 12 seeds (0, 5, 7, 10).
pyGAM fits all 12, though it overfits (edf about 19-20 on pure noise). sklearn's `_regression_dataset`
(1 informative of 10 features) fails the same way, and that accounts for 7 check_estimator failures.
The stderr just before shows "fit collapsed to its penalty null space ... rho=27.07" (a smoothing parameter at the
domain upper edge). The error comes from `crates/gam-solve/src/reml/eval.rs:1377-1380`: the proposal-ellipsoid scale
reaches 0 because the centre sits on the rho-domain boundary.
Irrelevant covariates are the most common real-data case, and the correct REML answer ("this smooth is zero") is
exactly where the posterior-mean integration breaks.
Fix: in the cubature, treat directions pinned at the rho boundary (a certified null collapse) as degenerate. Either
integrate them one-sided or drop them from the cubature (their contribution to the posterior mean of f is 0 to
first order), so the error is never returned. Size M. The robustness axis should cross-check this.

### F4 HIGH bug: `s()` on a categorical or string column silently fits a smooth over level codes
`gamfit.fit(df, "y ~ s(age)+s(region)")` with `region` in {N,S,E} succeeds: s(region) edf 1.98, p=3e-47, plus stderr
"[thin-plate] requested 96 distinct knots but the data contain only 3". A numpy object array with one "abc" in x1
makes x1 categorical and s(x1) fits (edf 8.27); predict with numeric x1 then fails
"column 'x1' is numeric in prediction data but categorical in the training schema". pyGAM raises clearly.
Fix: in Rust term materialisation, reject a smooth on a categorical column with a message that suggests
`factor(region)`/`group(region)`. For object arrays, report the first non-numeric value with its row. Size M.

### F5 MED bug: inconsistent handling of bad strings
DataFrame object column containing 'n/a' gives a raw, unmapped `pyarrow.lib.ArrowInvalid`. A numeric column stored as
strings is silently accepted. `None` in a categorical column at predict gives "unsupported Arrow column type Null for
column 'region'". Fix: catch these at the table boundary (`_tables.py normalize_table`) and map them to DataError with
column and row. Size S-M.

### F6 HIGH gap: formula is required, with no `terms='auto'` equivalent
`GAMRegressor()` gives TypeError (missing formula). `fit_array` also requires a formula. pyGAM's `LinearGAM().fit(X,y)`
auto-builds `s(0)+...+s(k)`. Fix: `formula=None` means Rust builds a default RHS from the schema (smooth for numeric,
factor for categorical) inside `sklearn_fit_metadata` (`crates/gam-pyffi/src/sklearn/sklearn_metadata.rs:23-74`,
but it belongs in the shared formula layer so the CLI gets the same default). This also clears many
check_estimator message failures. Size M.

### F7 HIGH gap: no `sample_weight`
`GAMRegressor(...).fit(X, y, sample_weight=w)` gives TypeError. Weights only work as a column name
(`weights="w"`), which does not fit numpy or Pipeline usage. `cross_val_score(..., params={"sample_weight": w})` and
`Pipeline.fit(..., gam__sample_weight=w)` fail. pyGAM has `fit(X, y, weights=)`.
Fix: `fit(X, y=None, sample_weight=None)` in `gamfit/sklearn.py:138`, forwarded as an array to the Rust fit (add an
array-weights path to `_api.fit`, `_api.py:668`). Size S-M.

### F8 MED bug: `GAMClassifier.score` returns AUC, not accuracy
`gamfit/sklearn.py:458-508`. `cross_val_score(GAMClassifier(...))` returns 0.955 (AUC) while accuracy is 0.87. The
sklearn ClassifierMixin contract is accuracy. Users comparing against LogisticRegression or pyGAM get inflated
numbers silently. Fix: return accuracy and expose AUC via `scoring="roc_auc"` (which already works via
predict_proba). Size S.

### F9 LOW gap: GAMClassifier is binary-only although the engine has a multinomial family
`sklearn.py:229-243` raises for `classes.size != 2`. pyGAM is also binary-only, so this is an opportunity rather than
a gap. It accounts for 14 check_estimator failures. Fix: route classes >2 to `family="multinomial"` (MultinomialModel
already exists). Size M.

### F10 MED gap: pyGAM result accessors missing
There is no `deviance_residuals`, `loglikelihood(X,y)`, `predict_mu`, `statistics_` (AIC, pseudo-R2, edof), `coef_`,
or `n_iter_` on the estimator or Model (all hasattr False). Partial substitutes: `summary().deviance`,
`.log_likelihood`, `.edf_total`, `diagnose(data)`, `evidence()`. Fix: expose `residuals(data, kind=...)`,
`log_likelihood(data)` computed in Rust, and fitted attributes (`coef_`, `edf_`, `n_iter_`) on the estimator. Do NOT
copy pyGAM's GCV/UBRE in statistics_ (SPEC 11). Size M.

### F11 MED gap: `print(model.summary())` has no per-term table
`gamfit/_summary.py:462-496` prints header statistics only. The per-term edf/ref_df/chi_sq/p is only in
`smooth_terms_frame()` (442-458) and in HTML. pyGAM `summary()` prints a term table. Fix: render the text summary in
Rust (like `summary_html`) including the term table, and delete the Python assembly (SPEC 8). Size S-M.

### F12 MED-HIGH bug: stderr log spam on routine fits, with no quiet or verbose control
A logistic CV run emits dozens of `[GAM COST] -> P-IRLS INNER LOOP FAILED ... min eigenvalue -6.6e174`
(`crates/gam-solve/src/reml/gradient_hessian.rs:7058`, log::warn!), plus `[OUTER] ARC cost-stall STUCK`,
`[INDEF-HESS] ...` dumps, and `[HGB] target_mse below mandatory floors`. check_reg.err is 813 KB for 52 checks. The
logger is installed by `crates/gam-solve/src/progress_log.rs` (set_logger ~187, DEFAULT_LOG_LEVEL ~224/232).
pyGAM is silent by default. Fix: demote internal trial or rejected-step diagnostics to debug/trace, keep the default
level for user-actionable messages only, and add one shared `verbose` control (Rust config, CLI and Python parity).
Size S-M.

### F13 LOW-MED: GamInferenceWarning on every default `s()` fit
"Automatically set 8 internal knots ... clamp(unique/4, 4..8)" appears on every default fit, plus basis-adequacy
warnings. Informational messages should not be warnings on the default path. The clamp is also a magic constant
(SPEC 23). Fix: report it in `summary()`/`report()`, not as a warning. Size S.

### F14 LOW: exception and shape inconsistencies
NaN raises DataError at fit but GamError at predict. y of shape (n,1) gives "target arrays must be 1D" (pyGAM and
sklearn accept it with ravel). Predicting with numpy after a DataFrame fit gives SchemaMismatchError. Numpy interval
predict returns 6 unlabeled columns. Fix: one exception hierarchy mapped in pyffi, `np.ravel` for (n,1) y in the
wrapper, and named columns or a structured return. Size S.

### F15 LOW-MED: formula parser rejects common syntax
Backtick names fail (`y ~ s(\`patient age\`)` gives FormulaError), so columns with spaces are unusable.
Patsy-style `C(x2)` is rejected (only `factor()` works). Fix: support backtick quoting in the Rust formula lexer and
accept `C()` as an alias, or give an error that suggests `factor()`. Size S.

### F16 MED bug: numpy predict silently ignores extra or misaligned columns
With a model fitted on 2 columns, `predict(X[:, :5])` returns values without error (`n_features_in_ = 2`), which fails
check_n_features_in_after_fitting and check_classifiers_train. For numpy input, a wrong column order or count is a
silent-wrong-answer risk. Fix: when the input is an unnamed array, the wrapper checks `X.shape[1] == n_features_in_`
(sklearn contract, allowed as external-interop logic). Size S.

### F17 LOW bug: minor sklearn-contract gaps
* `GAMClassifier().predict(X)` before fit raises `AttributeError: no attribute 'classes_'`, not NotFittedError.
* Sparse input gives "unsupported table input" (it should say sparse is unsupported, or densify).
* `_NotAnArray` (objects with `__array__`) are rejected. Use `np.asarray` fallback.
* A continuous target on the classifier gives "requires exactly two observed classes; got 60: array([...])" instead
  of "Unknown label type: continuous".
* `y=None` gives GamError, not the sklearn "requires y to be passed" ValueError.
* Complex input gives TypeError, not ValueError.

Fix all of these in `gamfit/sklearn.py` (wrapper boundary). Size S.

### F18 LOW bug: `@dataclass` estimator is unhashable and overrides BaseEstimator repr/eq
`hash(GAMRegressor(formula="s(x0)"))` gives TypeError (dataclass eq=True sets `__hash__=None`, `sklearn.py:58`). It
breaks use as a dict/set key and some caching tools. Fix: `@dataclass(eq=False, repr=False)` or a plain `__init__`.
Size S.

## pyGAM slop to avoid (do not copy)
* pyGAM's default `lam=0.6` with no REML gives edf about 19-20 on pure noise (`w9_null.py`), while gamfit gives
  about 1-5 when it succeeds. Keep REML defaults.
* pyGAM `gridsearch()` and GCV/UBRE in `statistics_` go against SPEC 11 and 18. Do not add them to reach parity.
* pyGAM p-values are documented as biased. Do not mirror its summary statistics.

## Already better than pyGAM
* sklearn 1.9 integration: pyGAM 0.12.0 fails Pipeline, cross_val_score and GridSearchCV
  (`AttributeError: __sklearn_tags__`). gamfit passes get_params, set_params, clone,
  Pipeline(StandardScaler) (R2 0.92), cross_val_score (about 0.91), GridSearchCV over formula (n_jobs=2 as well), and
  ColumnTransformer(set_output="pandas") with named formulas (0.88). String class labels work.
* pyGAM `clone()` loses terms (repr shows `terms=,`). gamfit's clone is correct.
* Error messages are excellent: a typo gives "Did you mean one of [age]?"; you get a family list; binomial
  out-of-range rows; "n too small" with a fix; constant column; an unseen level suggests `group(x2)`.
* Accuracy parity at lower complexity: gamfit R2 0.9239 at edf 17.8 vs pyGAM 0.9253 at edf 27.85.
* The numpy one-liner is the same length (`GAMRegressor(formula="s(x0)+s(x1)").fit(X,y)`), and DataFrame plus named
  formulas beat pyGAM's integer term indices.
* Warm start: neither library has sklearn `warm_start`, and gamfit has `persistent_warm_start_root`, so this is not
  a gap.
* Performance note (for the speed axis): GAMClassifier fit takes about 2.6 s vs pyGAM 1.5 s at n=400.
