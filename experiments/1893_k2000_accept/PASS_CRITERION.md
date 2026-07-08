# #1893 K=2000 acceptance — exact pass criterion

Reusable acceptance harness for the #1893 bar: does the curved-manifold SAE at
K=2000 (the K≫P regime) recover a real activation set as well as an external TopK
SAE, at usable throughput? PASS iff ALL THREE legs hold (the driver prints
`[#1893] VERDICT=PASS|FAIL ...` and exits 0 only on PASS):

1. **Completion — no co-collapse / no silent-linear reroute.**
   `gamfit.sae_manifold_fit(X_train, K=2000, assignment='topk', top_k=32)` returns a
   `ManifoldSAE` without raising. A `GamError` / infeasible-sentinel refusal (or any
   exception) is a FAIL (`completed=false`). This is the leg the acceptance-deflation
   fixes (dbdb20d27 / 37e4d2eae / 36bec0e29) had to land for.

2. **Quality — same-data match-or-beat the external TopK baseline.**
   `ev_ours >= ev_external`. Both EVs use the identical formula on the identical
   held-out split, `EV = 1 − ||X_test − recon||²_F / ||X_test − mean_train||²_F`,
   `recon = model.reconstruct(X_test)` (exact OOS latent solve). In the production
   sbatch mode `--external-live` trains a Gao-et-al. TopK SAE on the SAME
   train/test split at the SAME K and top_k — no cross-dataset literature number is
   borrowed. (The fixed `--external-ev`, e.g. 0.878 @ pythia-70m p=512, is only a
   fallback for an internet-capable pythia harvest run.) Match-or-beat, never
   "gam ≈ baseline", never a weakened threshold.

3. **Usable throughput (the #1995 leg).**
   `fit_seconds <= --max-fit-seconds` (default 3600 s). #1995 (compute-bound
   ~70–130 s/iter at K=96–128) was CLOSED in code by **a215a7345** ("Optimize sparse
   SAE Schur block GEMM (#1995)", PR #2041, ancestor of HEAD) + the 50aae88fb
   criterion-cost restructure, but its K=2000 EFFECT was never measured — this leg
   produces it. A correct-but-too-slow fit is FAIL on throughput and reopens #1995
   with the measured K=2000 s/iter.

## How it runs on MSI (from `run_1893_k2000_accept.sbatch`)

- `amdsmall`, 64 cpu, 220 GB, 8 h; root `/projects/standard/hsiehph/sauer354`;
  MSI OpenBLAS on `LD_LIBRARY_PATH` (the code_space_manifold precedent).
- **Venv is PINNED per-job**: a fresh `python3 -m venv` + `pip install
  gamfit==$GAMFIT_VERSION numpy scikit-learn torch` — never trusts a shared venv's
  wheel. Submit with `GAMFIT_VERSION=<the ≥0.1.252 wheel>`.
- **Data is MSI-resident + fail-loud**: `NPZ` must point at a real-activation file
  on `/projects`; the sbatch `test -f`s it and aborts (rc 2) rather than degrading to
  a synthetic fallback. Do NOT rely on in-job pythia harvesting (compute-node
  internet is not guaranteed); `--data-mode harvest_pythia70m` stays available for a
  later internet-capable run.

## Operator-confirm items (not resolvable from the repo)

- The exact MSI activation file + npz key (the #1942 map names emotions p=512 N=60k
  and OLMo L18 PCA-64; pick the p≈512-or-more set that exists on `/projects`).
- `GAMFIT_VERSION` — the published wheel to pin (lead sets it when PyPI serves it).
