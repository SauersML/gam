# Marginal-slope inner PIRLS scaling notes

## FLEX biobank-shape local reproducer (May 2026)

A focused ignored Rust integration test now exercises the same cycle-0 path as the
biobank-scale `rust_margslope_aniso_duchon16d_linkwiggle_scorewarp_fast` lane:

```text
cargo test --release --test margslope_flex_biobank_repro \
    -- --ignored --nocapture margslope_flex_biobank_repro
```

The test uses probit bernoulli marginal slope with both FLEX deviation blocks
(`score_warp` and `link_dev`), a joint 16D PC Duchon smooth with
`centers=24`, `nullspace_order=Linear`, `power=8`, a separate
`smooth(age_entry_std)`, and standard-normal latent `z`. It caps the optimizer at
one outer evaluation and one inner cycle so the printed elapsed time isolates the
cycle-0 exact-Newton row-kernel cost.

Validation status in this container after the per-cycle denested-cell moment cache fix:

| command | n | elapsed | notes |
| --- | ---: | ---: | --- |
| `cargo test --test margslope_flex_biobank_repro --no-run` | n/a | 38.17 s | compiled the ignored repro test successfully in dev/test profile |
| `GAM_MARGSLOPE_REPRO_N=100 cargo test --test margslope_flex_biobank_repro -- --ignored --nocapture margslope_flex_biobank_repro` | 100 | >60 s in dev/test profile | debug build is not representative and was stopped after the harness warning |

Release benchmark timing should be collected with the manual command above on a machine with an already-built release test binary. The test intentionally remains ignored so normal CI does not pay this cost.

The implemented fix caches each row's converged denested partition and degree-9
cell moments in the exact-evaluation cache. Those moments depend on the current
cycle's β (through the row intercept and FLEX coefficients) but not on the CG or
Hessian-vector direction, so gradient/diagonal/matvec passes reuse identical
floating-point inputs instead of repeatedly calling `evaluate_cell_moments` /
`bivariate_normal_cdf` for the same row and cycle.
