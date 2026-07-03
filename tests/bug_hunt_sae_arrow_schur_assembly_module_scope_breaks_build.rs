//! Bug hunt: the workspace does not compile from a clean checkout at HEAD —
//! `gam-sae` fails with two `E0425 cannot find function ...` errors, so every
//! target that links it (the `gam` crate this integration test depends on, the
//! whole `cargo build`/`cargo test`, and the `gamfit` maturin wheel) aborts at
//! the build step.
//!
//! The squash-merge `cd0639e4a` ("refactor(#780): split arrow-Schur assembly
//! out of construction.rs (10k-line gate)") moved the Arrow-Schur bordered-
//! Hessian assembly out of `crates/gam-sae/src/manifold/construction.rs` into a
//! new sibling file `construction_arrow_schur_assembly.rs`, and wired it into
//! the parent module with a plain module declaration:
//!
//!     // crates/gam-sae/src/manifold/mod.rs:139
//!     mod construction_arrow_schur_assembly;
//!
//! Every OTHER cohesive split-out of `construction.rs` is wired with `include!`
//! precisely so the extracted code keeps `construction`'s module scope:
//!
//!     // crates/gam-sae/src/manifold/construction.rs
//!     include!("softmax_entropy_majorizer.rs");        // line 91
//!     include!("construction_exact_hessian.rs");       // line 97
//!     include!("construction_row_jet_logdet_channels.rs");
//!     include!("construction_smoothness_dof.rs");
//!     include!("construction_tests.rs");
//!
//! The two private leaf helpers the assembly needs —
//!
//!     fn softmax_majorizer_log_mean(a: &[f64]) -> f64
//!     fn active_softmax_gershgorin_majorizer_entry(a: &[f64], kk, m, scale) -> f64
//!
//! live in `softmax_entropy_majorizer.rs`, whose own header comment states they
//! are "Included via `include!` from `construction.rs` so they keep the SAME
//! module scope (`use super::*`), visibility" — i.e. they are module-private
//! `fn`s of the `construction` module, not re-exported and not `pub(crate)`.
//!
//! Because `construction_arrow_schur_assembly.rs` is a SEPARATE module (a bare
//! `mod`, whose only import is `use super::*;` bringing in the parent
//! `manifold` module's items), those two `construction`-private helpers are out
//! of scope at its call sites:
//!
//!     // construction_arrow_schur_assembly.rs:1057
//!     .map(|_| softmax_majorizer_log_mean(assignments_slice));
//!     // construction_arrow_schur_assembly.rs:1063
//!     active_softmax_gershgorin_majorizer_entry(assignments_slice, k, m, *scale)
//!
//! so `cargo build -p gam-sae` fails:
//!
//!     error[E0425]: cannot find function `softmax_majorizer_log_mean` in this scope
//!        --> crates/gam-sae/src/manifold/construction_arrow_schur_assembly.rs:1057:42
//!     error[E0425]: cannot find function `active_softmax_gershgorin_majorizer_entry` in this scope
//!        --> crates/gam-sae/src/manifold/construction_arrow_schur_assembly.rs:1063:45
//!     error: could not compile `gam-sae` (lib) due to 2 previous errors
//!
//! The sanctioned fix mirrors the sibling splits: `include!` the extracted file
//! back into `construction.rs` (so it shares the module scope the two helpers
//! and the `impl SaeManifoldTerm` block live in), or otherwise make the two
//! helpers reachable from the new module. Either way the workspace must build.
//!
//! While the bug is present `gam-sae` — and therefore the `gam` crate this
//! integration test links against — does not compile, so this test target
//! cannot be produced and `cargo test` fails at the build step. Once the module
//! is wired so the workspace compiles, the assertion below runs: it drives a
//! real `gam-sae` SAE primitive (`sparse_dict::fit_sparse_dictionary`, reached
//! through the public `gam::terms::sae` re-export) on data that lies exactly in
//! a `K`-atom, single-active sparse dictionary and asserts the trainer recovers
//! that structure — i.e. the SAE subsystem is reachable and functional rather
//! than a compile error. No further edits are needed.

use gam::terms::sae::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
use ndarray::Array2;

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
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// Build data that lies EXACTLY in a `K`-atom, single-active sparse dictionary:
/// each of the `n` rows equals `code · atom_k` for one atom `k`, where the `K`
/// atoms have disjoint 2-column support (so they are mutually orthogonal, unit
/// norm, and trivially separable by routing). A sparse-dictionary trainer with
/// `K` atoms and an active budget of 1 must reconstruct such data essentially
/// perfectly.
fn build_single_active_dictionary_data(
    n: usize,
    k: usize,
    seed: u64,
) -> (Array2<f32>, usize) {
    let p = 2 * k; // each atom occupies its own disjoint 2-column block
    let inv_sqrt2 = (0.5_f64).sqrt();
    let mut rng = SplitMix64::new(seed);
    let mut x = Array2::<f32>::zeros((n, p));
    for i in 0..n {
        // Assign row i to atom (i % k) so every atom is well populated.
        let atom = i % k;
        // Positive, well-scaled code in [1, 3).
        let code = 1.0 + 2.0 * rng.unit();
        // Atom `atom` is the unit vector 1/√2 · (e_{2a} + e_{2a+1}).
        x[[i, 2 * atom]] = (code * inv_sqrt2) as f32;
        x[[i, 2 * atom + 1]] = (code * inv_sqrt2) as f32;
    }
    (x, p)
}

#[test]
fn sae_sparse_dictionary_recovers_single_active_structure() {
    let n = 300usize;
    let k = 3usize;
    let (x, p) = build_single_active_dictionary_data(n, k, 0x5AE_5EED);

    let mut config = SparseDictConfig::new(k);
    config.active = 1;
    config.minibatch = 64;
    config.max_epochs = 50;

    let fit = fit_sparse_dictionary(x.view(), &config)
        .expect("sparse dictionary fit should succeed on clean single-active data");

    // Structural invariants: decoder is K×P and finite; codes/indices well shaped.
    assert_eq!(
        fit.decoder.dim(),
        (k, p),
        "decoder should be K×P, got {:?}",
        fit.decoder.dim()
    );
    for &d in fit.decoder.iter() {
        assert!(d.is_finite(), "decoder entry must be finite, got {d}");
    }
    assert_eq!(fit.active, 1, "active budget should be 1");

    // The data lies exactly in a K-atom single-active dictionary, so the trainer
    // must recover almost all of the variance. A no-op / degenerate trainer
    // returns EV ~ 0 (or non-finite); require a comfortably-above-trivial floor.
    assert!(
        fit.explained_variance.is_finite(),
        "explained variance must be finite, got {}",
        fit.explained_variance
    );
    assert!(
        fit.explained_variance > 0.9,
        "single-active dictionary data should be reconstructed nearly perfectly; \
         got explained_variance = {}",
        fit.explained_variance
    );

    // The dense reconstruction from the sparse routing must track the input.
    let recon = fit.reconstruct();
    assert_eq!(recon.dim(), (n, p));
    let mut sse = 0.0_f64;
    let mut sst = 0.0_f64;
    let mean = x.mean().expect("non-empty data has a mean");
    for i in 0..n {
        for c in 0..p {
            let e = (x[[i, c]] - recon[[i, c]]) as f64;
            sse += e * e;
            let d = (x[[i, c]] - mean) as f64;
            sst += d * d;
        }
    }
    let r2 = 1.0 - sse / sst;
    assert!(
        r2 > 0.9,
        "reconstruction R² should be near 1 on single-active data, got {r2}"
    );
}
