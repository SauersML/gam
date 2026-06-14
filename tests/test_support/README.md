# Test Support

Shared Rust modules for integration tests and Criterion benches.

`margslope_flex_equivalence.rs` builds a deterministic Bernoulli
marginal-slope FLEX problem with 16 PC inputs, an age smooth, Duchon smooths,
score-warp/link-deviation blocks, and cycle-capped fit helpers. It is included
by large-scale marginal-slope repro tests and
`bench/cargo_benches/margslope_flex_large_scale_hv.rs`.

Keep this directory for reusable test harness code that is compiled with
`#[path = ...]`; ordinary one-off fixtures should stay next to the test that
owns them.
