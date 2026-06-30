//! Bug hunt: the workspace does not compile from a clean checkout at HEAD.
//!
//! The squash-merge `c8c3192fa` ("feat(#580): periodic period derivation for
//! radial builders — boolean periodic= ... on duchon/tps/matern") rewrote the
//! tensor-margin periodic branch in
//! `crates/gam-terms/src/term_builder.rs`. The previous branch consumed a local
//!
//!     let margin_is_cc = matches!(
//!         canonicalize_smooth_type(per_axis_bs[axis]...), "cc" | "cp" | "cyclic");
//!
//! to drive a `cc`-margin "wrap on the observed data range" fallback. The
//! refactor replaced that fallback with an unconditional
//! `periods_opt[axis].ok_or_else(...)` (periodic tensor margins now REQUIRE an
//! explicit `period=`), but it left the `margin_is_cc` binding behind, now
//! never read (term_builder.rs ~3245). The doc comment directly above it still
//! describes the deleted "wrap each margin on its own [min, max] span" fallback.
//!
//! The workspace builds under `[lints.rust] warnings = "deny"` (every crate is
//! compiled with `--deny=warnings`, visible in the rustc invocation), so the
//! dead binding fires `-D unused-variables`:
//!
//!     error: unused variable: `margin_is_cc`
//!       --> crates/gam-terms/src/term_builder.rs:3245:21
//!       = note: `-D unused-variables` implied by `-D warnings`
//!     error: could not compile `gam-terms` (lib) due to 1 previous error
//!
//! and `cargo build`, `cargo test`, and the `gamfit` maturin wheel all abort
//! with exit 101. Note the repository's own `build.rs` ban-scanner forbids
//! silencing this with either `#[allow(unused_variables)]` *or* an
//! underscore-prefixed `let _margin_is_cc` (the "`let _...` ban", build.rs):
//! the only sanctioned fix is to remove the dead binding (or wire it back into
//! the cyclic-margin path it was meant to drive).
//!
//! While the bug is present `gam-terms` — and therefore the `gam` crate this
//! integration test links against — does not compile, so this test target
//! cannot be produced and `cargo test` fails at the build step. Once the dead
//! binding is removed the workspace compiles and the assertion below runs: it
//! exercises the very feature the breaking commit shipped (a 1-D periodic
//! Duchon radial smooth, `duchon(x, periodic=true)`) and asserts the fit
//! succeeds and yields finite predictions — i.e. the periodic radial builder is
//! reachable rather than a compile error. No further edits are needed.

use gam::{
    FitConfig, FitResult, data::EncodedDataset, encode_recordswith_inferred_schema,
    fit_from_formula,
};
use csv::StringRecord;

/// Deterministic SplitMix64 — no external RNG crate, reproducible across runs.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit().max(1.0e-12), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A clean periodic signal sampled on `[0, 2π)`: `sin(x) + 0.5 cos(2x)`.
fn gen_periodic(n: usize, seed: u64) -> EncodedDataset {
    let mut rng = SplitMix64::new(seed);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = std::f64::consts::TAU * rng.unit();
        let f = x.sin() + 0.5 * (2.0 * x).cos();
        let y = f + 0.1 * rng.normal();
        rows.push(StringRecord::from(vec![x.to_string(), y.to_string()]));
    }
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

#[test]
fn periodic_duchon_radial_smooth_is_fittable_and_finite() {
    let data = gen_periodic(400, 0xC0FFEE);
    let cfg = FitConfig::default();

    // The feature shipped by the breaking commit: a boolean-periodic 1-D radial
    // (Duchon) smooth. The fit must succeed (the periodic radial path is wired)
    // and produce a finite coefficient vector — i.e. the periodic radial builder
    // is reachable rather than a compile error.
    let FitResult::Standard(fit) = fit_from_formula("y ~ duchon(x, periodic=true)", &data, &cfg)
        .expect("periodic Duchon radial fit should succeed")
    else {
        panic!("expected a standard Gaussian GAM fit");
    };

    assert!(
        fit.fit.beta.len() > 1,
        "periodic Duchon fit should carry a non-trivial coefficient vector, got len {}",
        fit.fit.beta.len()
    );
    for (i, b) in fit.fit.beta.iter().enumerate() {
        assert!(b.is_finite(), "periodic Duchon coefficient {i} is non-finite: {b}");
    }
}
