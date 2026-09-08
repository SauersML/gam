# Pending coefficient-seed builder verification for #2817

Reviewed against published HEAD `07f46dec1`. MSI compute nodes were unavailable
during this continuation; no compilation or test was run on the login node or
locally. This note does not claim that #2817 is resolved.

The pending criterion-resolution API was wired incorrectly:
`ClosureObjective::with_seed_inner_state` initialized `criterion_resolution_fn`
to `None` when moving the existing objective into its new closure type. Both
standard REML builders install `with_criterion_resolution` before calling
`with_seed_inner_state`, so that construction discarded the publisher.

The local correction moves `self.criterion_resolution_fn` into the rebuilt
objective, preserving its existing numerical contract. The public regression
`crates/gam-solve/tests/outer_decrement_2817.rs` now exercises all four
combinations of coefficient-seed hook present/absent and explicit gradient
requirement present/absent. The criterion-only case must retain the initial
point whose Newton decrement is unresolved; the stricter caller control must
reach the optimum. No earlier case was removed.

Source pins:

- Preserved optimized build's objective source snapshot:
  `385b011d8ace4367aa9937b058eb355893ccb81ee80b0c01a1bf79ee2ad36bce`.
- Corrected local objective:
  `3c2dc5bacf5edae4323b856d1aaaf030901b2ebc643b74bb11ecbd0e4ce53cdd`.
- Expanded local regression:
  `d2ad3042ff19ca9e0376a74f6ddf06906fa0638a77f920acd473c8a707ec6ab4`.

The previous, smaller regression passed in 0.04 seconds. That result predates
the added coefficient-seed cases and does not validate the new correction.
The expanded test is expected to fail against the preserved pre-correction
library; this expectation is based on source inspection, not a completed run.

Once compute access returns, upload the expanded canonical test and compile
it against the immutable baseline graph from the canonical source directory:

```sh
reml_snapshot=/scratch.global/sauer354/gam-deslop-a2-target/debug/deps/.reml-optimized-snapshot-20260908-1421
timeout 90s taskset -c 112-115 rustc --edition=2024 --test \
  crates/gam-solve/tests/outer_decrement_2817.rs \
  --crate-name outer_decrement_2817 --deny=warnings \
  -Copt-level=1 -Cdebuginfo=0 -Ccodegen-units=16 -Clto=off \
  -L "dependency=$reml_snapshot" \
  -L native=/scratch.global/sauer354/gam-deslop-a2-target/debug/build/zstd-sys-3cd5a87001ab0920/out \
  --extern "gam_solve=$reml_snapshot/libgam_solve-7f5b8ec14241822d.rlib" \
  --extern "gam_problem=$reml_snapshot/libgam_problem-a7ac6b5118c5d832.rlib" \
  --extern "ndarray=$reml_snapshot/libndarray-3e2fc606731d0706.rlib" \
  --extern "opt=$reml_snapshot/libopt-5dde3eb0c3e8fe95.rlib" \
  -o .buildd/outer-decrement-2817-builder-baseline
timeout 30s taskset -c 112-115 \
  .buildd/outer-decrement-2817-builder-baseline --nocapture
```

Then rebuild the canonical solver leaf with the preserved dependency graph
and repeat against its fresh library. The original 50,000-row acceptance still
requires a coherent model graph and remains pending independently of this test.

## Publication dependency review

The correction cannot be committed alone against HEAD: HEAD has no
`criterion_resolution_fn` field or `with_criterion_resolution` API. A coherent
future publication needs these selected changes, with fresh verification:

1. `rho_optimizer/objective.rs`: the criterion-resolution trait hook, forwarding
   wrappers, closure field/builder, and the corrected seed-builder transfer.
2. `rho_optimizer/run.rs`: the three closure-field initializers, the
   `CriterionResolution` helper, and certificate consumers that use the same
   resolution as the search. Preserve unrelated retry, chart, and stationarity
   changes already published in HEAD.
3. `rho_optimizer/run_plan.rs`: the matrix-free decrement callback using that
   owner, with the stricter caller-gradient requirement retained. The upstream
   opt pin and termination mapping are already published; no new Cargo change
   is required.
4. `reml/mod.rs`, `reml/gradient_hessian.rs`, `reml/objective.rs`, and
   `estimate/optimizer.rs`: the stored resolution, its initializer, additive
   criterion publication, and both standard REML callback installations.
5. The public regression above. Its standalone publication would otherwise
   introduce a test that cannot build against HEAD.

The separate #2834 Cholesky roundoff-admission patch is not a dependency of
this correction and remains unverified by fresh public acceptance.
