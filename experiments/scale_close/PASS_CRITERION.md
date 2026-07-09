# Scale-closure runtime evidence — #2132, #1893, #2134

The mechanistic code for all three issues is landed (chunked-seed streaming driver
`31d13d651`, `AssignmentMode::TopK` support-sparse build `15fd89c17+`, co-collapse
fixes `73dd39009`/`f46b1bccf`/`71f06cce1`). What each issue's own text still demands
is the MEASURED closure number. These three harnesses produce those numbers, each
printing a single `VERDICT=PASS|FAIL` line so the coordinator can close from the job
log alone. Every driver `py_compile`s clean and uses gamfit's PUBLIC API only.

Outputs land under `/projects/standard/hsiehph/sauer354/scale_close/` (the #1893
harness keeps its own landed output path; see below).

---

## #2132 — held-out EV vs K on planted circle mixtures

**Driver:** `driver_2132_planted_circle_ev_vs_k.py`  ·  **Sbatch:**
`run_2132_planted_circle.sbatch` (`msismall`, 16c/48g/3h, pinned wheel).

Pure synthetic truth-recovery bar (no data file, no network): M unit circles in
mutually-orthogonal 2-planes, concentric at the origin, plus isotropic noise; each
token drawn on one circle. The curved fit is `gamfit.sae_manifold_fit(assignment=
'topk', d_atom=1, atom_topology='circle', top_k=1)`; the linear bar is affine PCA at
rank == K (the EV-optimal linear rank-K reconstruction — a strong, un-weakened
same-rank baseline). Both scored with the identical held-out EV on the identical
split.

**PASS (all three legs):**
1. **Above linear PCA at every K** — `ev_curved(K) >= ev_pca(rank=K) + 0.05`. A 1-D
   circle atom captures a whole planted circle from one coordinate; affine needs a
   2-plane per circle, so at matched rank K curved wins (~K/M vs K/2M).
2. **Increasing in K** — `ev_curved(K_max) - ev_curved(K_min) >= 0.10` (more atoms
   capture more circles).
3. **Monotone up to fit noise** — no `ev_curved` step drops by more than `0.02`.

A fail on leg 1 means the curved lane is not beating linear (reopens #2132 on
quality); a fail on legs 2/3 means EV does not scale with K.

## #1893 — K=2000 curved-beats-linear on real activations (REUSE)

**Driver + sbatch:** `experiments/1893_k2000_accept/` (`dcea9ef9b`, audited — no
changes needed). Output: `/projects/standard/hsiehph/sauer354/sae_1893_k2000_%j.out`.

Audit result: submission-ready. The wheel is pinned per job (fresh venv +
`gamfit==$GAMFIT_VERSION`); the creditscope Qwen3.5-35B-A3B layer-30 residual set is
MSI-resident and `test -f`-guarded (no synthetic fallback); the pass criterion
matches the issue exactly — three legs: (1) `sae_manifold_fit(K=2000,
assignment='topk')` COMPLETES (no co-collapse / silent-linear reroute), (2)
`ev_ours >= ev_external` where the external TopK SAE is trained live on the SAME
train/test split at the SAME K/top_k (`--external-live`, match-or-beat, no borrowed
literature number), (3) `fit_seconds <= 3600` (the #1995 throughput leg). Driver
uses the public API only and prints `[#1893] VERDICT=...`.

## #2134 — refused (N,K,P) region runs with peak-RSS discipline

**Generator:** `gen_2134_planted_input.py`  ·  **Sbatch:**
`run_2134_shape_matrix.sbatch` (`msismall`, 32c/96g/12h, fresh-clone cargo build).

Drives the compiled `crates/gam-sae/examples/scale_k.rs` — the public front-door +
block-sparse streaming lane whose peak-RSS invariants are ENFORCED inside the
example (it exits non-zero if peak RSS ever implies a dense `(N,K)` or a second full
`(N,P)` allocation). The sbatch sweeps the overcomplete shape matrix (K > P in every
cell): N ∈ {2e5, 5e5}, K ∈ {2048, 8192}, P ∈ {256, 1024}. One synthetic planted
`(N,P)` input per cell (reused across K).

**PASS (per cell, and overall = all cells):**
- `example_rc == 0` (no front-door refusal, no RSS-invariant abort);
- `invariants.sparse_front_door` — admitted to the sparse/streaming lane, not a
  dense build;
- `no_dense_nxk_by_peak_rss` — measured peak RSS below the dense `N×K` f32 payload;
- `no_second_full_nxp_by_payload_bound` — peak below one `N×P` + streaming payload;
- `fit.final_ev >= 0.10` (EV-sanity floor: the fit actually reconstructed planted
  structure).

The overall `[#2134] VERDICT=PASS` line prints only if every overcomplete cell
passes all five. `summary_2134.jsonl` carries the per-cell records.

> Note on the vehicle: #2134's own bar is "the refused region RUNS with peak-RSS
> discipline", which `scale_k` already asserts in-process at exactly these shapes.
> A bespoke new example was deliberately NOT added — a fresh Rust compile target is
> unverifiable from this seat and would risk the examples build; the public
> `sae_manifold_fit(assignment='topk')` at K>P is the python-side twin of the same
> curved streaming lane if a curved (rather than block-sparse) run is later wanted.

---

## Submission commands (dependency order)

`$WHEEL` is the published wheel the coordinator pins (current source tree is
`0.3.148`; the wheel must be built from ≥ the co-collapse + block-GEMM fixes).

1. **Push these files to `origin/main` FIRST.** The #2134 job clones `origin/main`
   and refuses to run unless the fresh clone == `origin/main`, so the generator and
   this directory must be on `origin/main` before submission. (#2132 and #1893
   resolve their driver from the submitted sbatch's directory, so they also need the
   checkout present.)

2. **#2134 — shape matrix (no wheel needed; builds at HEAD):**
   ```
   sbatch experiments/scale_close/run_2134_shape_matrix.sbatch
   ```
   Quick smoke first if desired:
   ```
   NS="200000" KS="2048" PS="256" EPOCHS=1 sbatch experiments/scale_close/run_2134_shape_matrix.sbatch
   ```

3. **#2132 — planted-circle EV-vs-K (pinned wheel):**
   ```
   GAMFIT_VERSION=$WHEEL sbatch experiments/scale_close/run_2132_planted_circle.sbatch
   ```

4. **#1893 — K=2000 real-activation acceptance (pinned wheel; creditscope data
   resident):**
   ```
   GAMFIT_VERSION=$WHEEL sbatch experiments/1893_k2000_accept/run_1893_k2000_accept.sbatch
   ```

Steps 2–4 are independent of each other and may be submitted in parallel once step 1
has landed on `origin/main`. Close each issue from its `VERDICT=` / per-cell
`RESULT` lines in the job log.
