# #1033 k-dim surrogate — landing plan (consumer wiring, ready to apply)

Staged artifacts (validated, in scratchpad):
- `kdim_surrogate_module.rs` — the module (place at `crates/gam-solve/src/reml/kdim_surrogate.rs`).
- `kdim_surrogate_standalone.rs` — standalone `rustc --test` validation, 5/5 PASS
  (exact-on-quadratic vs independent brute force; minimizer at −H⁻¹g; ranking ==
  true objective; far corners flagged & never dropped; 2nd-order Richardson 8×).

## Why module + consumer must land in ONE commit
`build.rs` text-scans every `src/**.rs` for the "pub(crate) item with ZERO
consumers" ban regardless of `mod` wiring. The module's `pub(crate)` API
(`BasinReference`, `RankedCandidate`, `rank_candidates`) needs a production
(non-test) consumer in the same landing, else the scanner aborts the whole
workspace build. So there is no separately-committable "increment 1"; the
cohesive default-OFF landing IS the increment.

## The cohesive landing (apply when a `cargo check -p gam-solve` slot opens)

1. `mv scratchpad/kdim_surrogate_module.rs crates/gam-solve/src/reml/kdim_surrogate.rs`
2. In `crates/gam-solve/src/reml/mod.rs`, add near the other `mod` decls:
   `pub(crate) mod kdim_surrogate;`
3. In `crates/gam-solve/src/estimate/optimizer.rs`, immediately BEFORE the
   `select_objective_seed_on_log_lambda_grid` call (~line 811/826), insert the
   default-OFF consumer (byte-identical when off; references every pub(crate)
   item + every `RankedCandidate` field to satisfy dead_code/zero-consumer):

```rust
// #1033 (default OFF): k-dim sufficient-statistic seed-grid pre-ranking.
// When enabled, summarize the baseline basin by (V₀, g, H) from ONE
// value+gradient+Hessian outer eval and rank the isotropic grid candidates by
// the second-order surrogate, so only the top-ranked interior candidates and
// every out-of-trust corner are evaluated at full n. OFF until basin-selection
// bit-equivalence is validated on a RAM/CI runner (#1575/#1033). The grid is
// byte-identical while off, so #1266/#1464/#1548 + the #1426 λ→0 trap are
// untouched.
const ENABLE_KDIM_BASIN_PREFILTER: bool = false;
#[allow(clippy::overly_complex_bool_expr)]
if ENABLE_KDIM_BASIN_PREFILTER {
    use gam_problem::HessianResult;
    // Isotropic shift candidates mirroring the grid's coarse scan.
    let shifts: [f64; 8] = [-12.0, -9.0, -6.0, -3.0, 3.0, 6.0, 9.0, 12.0];
    let candidates: Vec<ndarray::Array1<f64>> = shifts
        .iter()
        .map(|&d| {
            let mut c = base.clone();
            for i in 0..k {
                c[i] = (base[i] + d).clamp(lo, hi);
            }
            c
        })
        .collect();
    if let Ok(eval) = reml_state.compute_outer_eval_with_order(
        &base,
        crate::rho_optimizer::OuterEvalOrder::ValueGradientHessian,
    ) {
        if let HessianResult::Analytic(h_dense) = eval.hessian {
            let reference = crate::reml::kdim_surrogate::BasinReference::new(
                base.clone(),
                eval.cost,
                eval.gradient,
                h_dense,
            );
            // Trust radius = one isotropic grid step (±3): interior refinement
            // points are in-trust; the ±6..±12 corners are out-of-trust and stay
            // full-n.
            let ranked = crate::reml::kdim_surrogate::rank_candidates(&reference, &candidates, 3.0);
            let n_in_trust = ranked.iter().filter(|r| r.within_trust).count();
            let best = ranked.first();
            log::debug!(
                "[#1033] k-dim basin pre-rank: {}/{} candidates in trust; best idx {:?} surrogate {:?}",
                n_in_trust,
                ranked.len(),
                best.map(|r| r.index),
                best.map(|r| r.surrogate_cost),
            );
        }
    }
}
```

4. `cargo check -p gam-solve` (gate on vm_stat: free+inactive > 1.5GB AND swap
   free > 1GB; serialize — never the 4th rustc). Fix any compile errors. The
   module's own `#[cfg(test)] mod tests` (5 tests, ported from the standalone)
   give crate-level math coverage.
5. `git fetch origin main && git rebase origin/main`; commit; push.

## Verify before flipping the flag ON (the one gated step)
- `binomial_logit_reml_outer_work_bounded_1575`: reml_score/EDF byte-identical
  + `inner_pirls_solves` must DROP (the win) and never exceed today.
- #1266/#1464/#1548 quality fixtures green (basin selection unchanged: the
  prefilter only reorders which in-trust interior points are full-evaluated;
  every out-of-trust corner is still full-evaluated, and the adopted seed is
  always full-n verified + the existing release-and-rerank lower bound #1371
  holds).
- Needs a runner with RAM/CI; flip + tighten the solve-count bound there.

## Open integration refinement (when flag goes ON, follow-up)
The skeleton ranks candidates but does not yet SUPPRESS their full-n grid solve.
The actual saving requires `select_objective_seed_on_log_lambda_grid` to accept
the ranked in-trust order and skip full eval of dominated in-trust interior
points (keeping the top-m + all out-of-trust corners). That is a signature change
to the grid fn (also my lane) — do it in the flag-ON increment with the same
verification matrix.
