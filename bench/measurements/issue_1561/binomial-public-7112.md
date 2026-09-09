# Partial public binomial replay — 2026-09-08

Issue #1561 remains open. This was a bounded diagnostic, not a completed quality
population or a five-fold pass/fail result.

The public Python `gamfit.fit` / `model.predict` path replayed the original
654-row prostate fixture, formula `y ~ s(pc1, k=5) + s(pc2, k=5)`, binomial logit,
and deterministic folds `row % 5`, visited in order 3, 0, 1, 2, 4. Both plugin and
posterior probabilities were retained. MSI acn112 used CPUs 108–109, two Rayon
and OpenBLAS threads, and `.venv-python312-issues` (the clean Python environment).
The loaded Linux extension was the previously verified artifact with SHA-256
`7112c1fffe857fb5eaf709c93bbc6de6b899925329f287b53ef66ba506c7509c`.
Its build and source provenance are retained in
`../issue_triage_20260908/issue2833-verification.md` and associated receipts.

| Fold | Seconds | Plugin AUC | Posterior AUC |
| --- | ---: | ---: | ---: |
| 3 | 21.1632 | 0.6428571 | 0.6430906 |
| 0 | 3.0177 | 0.6989272 | 0.6989272 |
| 1 | 6.5762 | 0.7290112 | 0.7290112 |
| 2 | 9.2392 | 0.7451026 | 0.7448694 |
| 4 | incomplete | — | — |

The process reached the 180-second diagnostic limit with exit status 124 while
fitting fold 4. The log records smoothing searches rejecting seeds with gradient
norms above their convergence thresholds. A complete five-fold result was not
produced. This diagnostic limit is shorter than the original 360-second case
limit, so it does not establish failure of that original deadline. No worker
owned by this audit remains running.

`binomial-public-7112.log` is the raw process log;
`binomial-public-7112.csv` contains only the four completed held-out folds. Do not
aggregate the partial population as if it were a completed quality test.
`public_binomial_holdout_audit.py` preserves the original absolute/reference AUC
and NLL bars and scores both public prediction modes. After the diagnostic
timeout, it was updated to persist partial JSON provenance after each fold; that
incremental-persistence edit has not been rerun. The first run produced no JSON
because its original script wrote JSON only after all folds completed.

The retained `binomial-reference.jsonl` already contains a separate completed
reference audit: mean AUC 0.707298 for GAM, 0.699185 for EBM, and 0.706406 for
pyGAM; fold-zero NLL 0.621532, 0.621059, and 0.621257 respectively. Its original
four bars pass. These reference models were not recomputed in this replay, and
their scores do not establish completion of the current public API run.

No original quality threshold, model specification, fold assignment, or
convergence tolerance was changed. No global significance claim follows from
this partial diagnostic.

## Follow-up source audit

The later issue update reports all five native fits in 3.340 seconds. Its raw
MSI log, `$MSI_HOME/issue1561-binomial-holdout.log`,
records each fold's successful fit, EDF, and rho. This does not supersede the
partial Python run or establish a measured Python-overhead ratio: an exact
native executable hash and its complete dependency/source provenance were not
present in the retained seven-file `source-sha256.txt` receipt.

The configuration review found no different statistical option to explain the
gap. Python's omitted link resolves to binomial logit, matching the native
example's explicit `link="logit"`. Both leave Firth, flexible links, dimension
scaling, and adaptive regularization at their common defaults. JSON parsing
starts from `FitConfig::default()`. Python's table-kind field changes saved
metadata only. `fit_formula_to_payload` and `fit_from_formula` share the standard
fit driver.

The reviewed family resolution, configuration resolution, and payload service
files match the actual 7112 build manifest exactly:

| File under `crates/gam-models/src/` | SHA-256 |
| --- | --- |
| `fit_orchestration/materialize/family.rs` | `e8a126f9d40c53b932a4393085700766a5ea2e7f1a41828f5d0964def885c059` |
| `fit_orchestration/fit_config.rs` | `a3bc5b691e02e1b714c46af72fc0590ce7a29621b13e831e4f1743248a94fe6b` |
| `inference/model_payload_builders.rs` | `43dbce57c689ad733efcaa2adf63491cf323d5186c75f2f43b6cb4e1e6bb2f9a` |

The Python log shows repeated BFGS seed searches and stationarity refusals;
the elapsed time cannot be assigned to table conversion or serialization from
these observations. A useful next comparison must run both entry points
against the same compiled dependency graph, identical encoded rows and thread
allocation, and retain terminal rho/EDF and source provenance. No speculative
configuration change was made. MSI compute nodes were unavailable during this
follow-up; no builds or test computations ran on the login node.
