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
