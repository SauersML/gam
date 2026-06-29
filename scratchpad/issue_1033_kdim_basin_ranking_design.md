# #1033 — k-dim sufficient-statistic basin ranking (REML outer loop)

Design doc for the REML/#1575 facet of #1033. Owner lane: reml-perf. Shared
surface with the sae-cluster #1033 lane (kappa/psi design-interpolation) — see
"Overlap with the SAE lane" at the end.

Status of the umbrella issue: #1033 = "n-dependent work happens ONCE per fit;
the rho/kappa/psi outer loop manipulates only k×k objects". Acceptance is a
MEASURED n-independent per-trial eval cost (every theory-only magnitude estimate
in this arc has been wrong — so this design GATES on a local profile before any
rewrite).

---

## 1. Current full-n probe flow (binomial logit, the #1575 repro)

`estimate/optimizer.rs` drives, per fit:

1. **Seed-grid prepass** — `seeding.rs:507 select_objective_seed_on_log_lambda_grid`,
   called from `optimizer.rs:811/826` with the eval closure
   `|rho| reml_state.compute_cost(rho)`. For k smoothing params it scores:
   baseline + 9 isotropic shifts (±3…±12) + per-axis ±3 refinement + the
   `#1266` over-smoothing saturation corner per axis + the `#1548` keep
   ("lower saturation") corner for null-space axes + (k≤6) pairwise corners.
   ⇒ ~20–32 **distinct, value-only `compute_cost` calls**, each a full-n inner
   P-IRLS solve to the adaptive-KKT tol. mgcv does ZERO pre-grid solves.
2. **Seed screening + multistart** — a handful more full solves; the 2nd start
   is load-bearing when the 1st converges to an under-penalized basin
   (`3aa783052` waiver deliberately does NOT fire there, #1373).
3. **ARC outer Newton** — ~8 textbook-quadratic steps, each: ONE inner P-IRLS
   solve + analytic k-dim gradient + analytic/Fisher k×k Hessian
   (`rho_optimizer/bridges.rs:980+`). This phase is already mgcv-shaped and fine.
4. **Finalize + inference** — a few more full solves (post-fit covariance, EDF).

Measured on `main` (prior runs, R-free 2k-row 3-smooth fixture): outer cost evals
~26, actual cache-missing inner P-IRLS solves **53** (21 grid + 7 screen + 18
multistart/Newton + 7 finalize). The headline ~150 of the issue is already gone;
the residual full-n multiplier is the **grid prepass + multistart count** (≈40 of
the 53), NOT a slow individual solve and NOT outer-Newton step count.

### What `compute_cost(rho)` actually pays per call (n-dependent components)
Each call (`reml/mod.rs` cost path) runs:
- **Inner P-IRLS** to convergence: per iter forms `XᵀW(β̂)X` (O(n·p²) → p×p),
  builds working response `z` and weights `W` (n-vectors), sums deviance.
  Warm-started from `warm_start_beta` / the IFT predictor
  (`predict_warm_start_beta_ift_with_outcome`) — ~2 iters near the optimum,
  more on far grid corners.
- **Stable reparameterization** (`ReparamResult`): eigen/QR of penalty-vs-Gram
  structure, p×p (microseconds at p≈28) but rebuilt per call unless the
  penalty-subspace cache hits.
- **Firth/Jeffreys bias term** (binomial, n≤`FIRTH_MAX_OBSERVATIONS`=20_000):
  Tierney-Kadane / Jeffreys log-det of the Fisher info and its derivative — the
  `jeffreys_subspace.rs` path. mgcv never pays this. It is a deliberate #1426
  separation guard and CHANGES β/λ/EDF, so it is not removable.
- **LAML/REML cost assembly**: `½log|H_pen| − ½log|S|₊` + the dispersion/Mp term.

> ⚠️ PROFILE GATE (do FIRST, locally, bounded): the 2.3 s/solve at n=16k is NOT
> explained by the p×p linear algebra (sub-ms) or one Gram form (≈1.2e7 flops).
> The dominant per-call n-cost is most likely (i) the number of inner P-IRLS
> iterations on far grid corners × the n-sized W/z/deviance passes, and/or (ii)
> the Firth/Jeffreys n-sized derivative assembly. Confirm with a `perf`/sampling
> profile of `examples/binomial_pspline_perf_1575.rs` (restore it — it was
> dropped by `051397fab`) at n=2k and n=8k before choosing a mechanism. Do not
> optimize an unprofiled hotspot.

---

## 2. The objects that must become k-dim sufficient statistics

The architectural invariant (#1033): n-dependent work ONCE per fit; the ρ search
touches only k×k. For the **rho-only binomial** case the design matrix `X` does
NOT move with ρ — only the IRLS weight diagonal `W=W(β̂(ρ))` does. So:

| object | today | target |
|---|---|---|
| `X` (n×p) | fixed, re-read each solve | read once; never re-touched in ρ loop |
| `XᵀW(β̂)X` (p×p) | re-formed per IRLS iter per probe | reference Gram + structured update across ρ (the hard part — W moves with the fitted mean; this is the #1033(2) "non-Gaussian W" case, NOT the rho-only Gram-cache case #1033(a)) |
| `β̂(ρ)` | full inner solve per probe | reference + IFT/Newton prediction (already partially done) |
| grid basin RANK | full-n cost per grid point | **k-dim surrogate cost** (this doc's core proposal) |
| Firth/Jeffreys logdet | n-sized per probe | reference + k-dim derivative extrapolation, exact recompute only at adopted candidates |

The key reframing: the grid prepass does NOT need the EXACT cost at each probe —
it needs a correct **ranking** to pick the starting basin. The exact cost is
recomputed (full-n) only for the ≤1 candidate the optimizer adopts. So the
target is a cheap, monotone-faithful surrogate `Ṽ(ρ)` that:
- ranks nearby interior points by the analytic quadratic model (free: g, H at the
  reference are already k-dim), and
- ranks the far load-bearing CORNERS (#1266 shrink-out λ→∞, #1548 keep λ_null→0,
  #1464 collapse) by their **analytic limiting form**, not a full re-solve.

---

## 3. Candidate mechanisms (increasing ambition; aligned to #1033 (a)/(b)/(c))

**M0 — restore the perf harness + profile (prerequisite, zero-risk).**
Restore `examples/binomial_pspline_perf_1575.rs`; profile to attribute the 2.3 s
between inner-iters / Firth / reparam. Gates every later choice.

**M1 — quadratic surrogate for the INTERIOR grid points (low risk).**
At the reference solve compute V₀, k-dim g, k×k H (already available in phase 3).
Replace the isotropic ±3/±6 *interior* grid `compute_cost` calls with
`Ṽ(ρ)=V₀+gᵀΔ+½ΔᵀHΔ`. Only the argmin candidate is verified with a real full-n
solve before adoption. Bit-contract: the FINAL fit still comes from the outer
Newton seeded by the adopted candidate, so β̂/λ̂/EDF are unchanged whenever the
surrogate picks the same basin as the full-n grid would. RISK: a sub-tol basin
tie could flip; mitigate by verifying the top-2 surrogate candidates full-n and
keeping the existing criterion-rank comparison. Expected: removes the interior
grid solves (~half of the 21), keeps all corner guards.

**M2 — analytic limiting surrogate for the load-bearing CORNERS (medium risk).**
The corners are structured limits, not arbitrary ρ:
- #1266 shrink-out (λ_i→∞): term i collapses to its penalty null space → the
  cost is that of the model with term i's range columns dropped. Evaluate via a
  **dimension-reduced** solve on the reference factorization (downdate the
  dropped columns) rather than a full cold re-solve.
- #1548 keep (λ_null→0): the null columns become unpenalized → a low-rank
  *freeing* update of the reference factor.
- These are rank-`(block size)` updates to `XᵀW X + S`, NOT the full-rank ΔW that
  made Woodbury impossible for the inner solve (see
  `project_reml_trace_pcg_perf_first_principles`). The W change between the
  reference and the corner is the obstruction — bound it / re-linearize W at the
  reference (one extra IRLS half-step) so the corner cost is a structured p×p
  update. RISK: W-motion across a far corner may be too large for a single
  re-linearization → fall back to the full solve for that corner (criterion-rank
  still correct, just not cheaper). Must preserve #1266/#1464/#1548 exactly:
  every corner the full grid would have FOUND must still be found.

**M3 — #1033(c) endgame (defer):** precision/SPDE representation where the
data-fit operator is θ-free and only the k×k prior precision moves. This is the
sae-cluster's 1-D-scan production instance; for the general binomial GAM it is a
research rewrite, out of scope for the #1575 mitigation.

---

## 4. Bit-contract & quality-guard constraints (HARD)

- **#1426 λ→0 trap:** a capped/loose/surrogate cost MUST NOT report spuriously
  low cost as λ→0. The quadratic model is safe iff H carries the correct
  log|S|₊ curvature; a corner surrogate must include the penalty-logdet limit.
  Any surrogate that can be fooled here is rejected — this is the single most
  important guard (it is WHY the grid runs full-precision today).
- **#1266 / #1548 / #1464:** every basin the full-n grid currently discovers must
  still be discovered. Surrogate ranking is only allowed to CHANGE which point is
  probed first; the adopted candidate is always verified full-n, and the existing
  release-and-rerank lower-bound guard (`release_rerank_seed`, optimizer.rs:842,
  #1371) must still hold (`certified optimum ≤ grid best`).
- **#1373:** the multistart 2nd seed must still fire when start-1 lands in an
  under-penalized basin.
- **Final fit invariance:** β̂/λ̂/EDF must be bit-identical whenever the surrogate
  selects the same basin the full grid would. The regression guard
  `binomial_logit_reml_outer_work_bounded_1575` (records `inner_pirls_solves`,
  bounds it <150, asserts score-relative stationarity + EDF band) is the gate;
  tighten its bound as solves drop, never loosen.

---

## 5. Staged migration plan

0. **M0**: restore harness, local bounded profile (n=2k/8k), attribute the cost.
   Land harness (additive). Pick the real hotspot.
1. **M1**: add the quadratic interior surrogate behind the existing grid; verify
   top-2 candidates full-n; assert (locally + via the #1575 guard) byte-identical
   final β̂/λ̂/EDF on the binomial/3-smooth fixtures + the #1266/#1548/#1464
   quality tests. Tighten the solve-count bound. Land in one increment.
2. **M2**: structured corner surrogate with full-n fallback; same verification
   matrix; must not regress any corner guard. Land per-corner-type incrementally.
3. **M3**: defer to the SPDE/precision rewrite (sae-cluster + research).

Each stage is independently landable and independently revertible; none ships
without the local byte-identity check on the guard fixtures AND the corner
quality tests.

---

## 5b. Concrete per-trial λ-invariant waste found by inspection (ready edits)

Confirmed by reading `pirls/loop_driver.rs::fit_model_for_fixed_rho_with_adaptive_kkt`
(the sole full-n inner-solve entry, called once per cost eval from
`gradient_hessian.rs:6241/6643`). These run EVERY trial though they depend only
on the fixed design/penalty structure, not on λ=exp(ρ):

1. **Fresh `PirlsWorkspace` per call** — `loop_driver.rs:906`
   `PirlsWorkspace::new(n, p, ebrows, erows)` allocates+zeros ~6 O(n) buffers +
   several p×p matrices on every solve (~53/fit). Bit-identical fix: pool one
   workspace on `RemlState`, reset/resize per solve. Removes the literal
   #1033(3) "trial loop owns n-shaped allocation".

2. **Full design sparse RE-detection per call** — `loop_driver.rs:889-894`
   `x.into()` + `sparse_from_denseview(dense.view())` scans the entire n×p design
   for sparsity EVERY call. Purely a function of X (λ-invariant). Fix: detect
   once per fit, pass the resolved `DesignMatrix` in. Bit-identical (same design).

3. (Already done, noted) kronecker reparam memoized via `invariant_structure()`
   (`:854`); `balanced_penalty_root` precomputed (`:822`).

> These are real and bit-identical but individually <~1% of the measured
> 2.3 s/solve (alloc/scan are ms-scale at n=16k). The DOMINANT 2.3 s is almost
> certainly legitimate compute: inner P-IRLS iteration COUNT on the extreme grid
> corners (ρ=±12 ⇒ near-unpenalized, ill-conditioned ⇒ many IRLS iters), each
> iter an O(n·p²) BLAS Gram, plus the Firth/Jeffreys n-sized assembly and the
> reparam eigendecomp. Avoiding THAT needs M1/M2 (cut the COUNT of full-n corner
> solves), not the hoists above. ⇒ M0 profile is required to split "per-solve
> overhead" (fix via hoists 1-2) from "solve count × real compute" (fix via
> M1/M2). Do not land a surrogate before the profile attributes the cost.

### Verification protocol for these edits (when RAM frees)
- `cargo check -p gam-solve` (single crate) for compile.
- The #1575 guard `binomial_logit_reml_outer_work_bounded_1575` for bit-identity
  of reml_score/EDF + the `inner_pirls_solves` bound (must not rise).
- The #1266/#1464/#1548 quality fixtures stay green (hoists 1-2 don't touch
  ranking; only M1/M2 do).

## 6. Overlap with the SAE / sae-cluster #1033 lane

- The **sae-cluster** owns #1033 mechanism **(b)** Chebyshev-in-θ design
  interpolation + **(c)** the precision/SPDE 1-D scan, and the large-n SAE
  per-row Jacobian resident-memory floor (`project_1033_arc_share_per_row_jacobian`).
  Those touch design-dependent hyperparameters (κ/ψ) and the SAE kron-Jacobian
  operator.
- **This doc** owns the **rho-only binomial REML** facet: the n-sized inner
  P-IRLS re-solve per outer trial (#1033 violation (2), "non-Gaussian W
  re-pays O(n k²)") and the seed-grid basin-ranking count.
- Shared surface: the k-dim quadratic-model machinery (g, H reuse) and the
  reference-Gram + structured-update primitive could be a common crate-level
  utility. Coordinate the API for "structured low-rank update to a cached
  factorization" so both lanes use one implementation, not two.
- Boundary to confirm with the lead: who owns the shared structured-update
  primitive. Proposal: reml-perf prototypes it for M2 (rho-only, simplest W
  case); sae-cluster generalizes it for the κ/ψ design-motion case.
