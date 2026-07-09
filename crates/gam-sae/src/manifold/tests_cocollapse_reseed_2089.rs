//! #2089 — a tiny featureless SAE fit that TOTALLY co-collapses must not abort /
//! SIGKILL the outer alpha="auto" REML search: the co-collapse refusal is a
//! RECOVERABLE infeasible-ρ probe the outer optimizer steers around, not a fatal
//! error.
//!
//! Repro (from the issue): `gamfit.sae_manifold_fit` on
//! `X = rng.normal(size=(120, 32))`, `K = 6`, `d_atom = 1`,
//! `atom_topology = "circle"`, `assignment = "ibp_map"`, `alpha = "auto"`
//! terminated the Python PROCESS with exit 137 (SIGKILL) — uncatchable by
//! `try/except`.
//!
//! White noise carries no low-rank circle structure, so at any non-trivial
//! smoothing ρ the six decoders co-vanish together and the inner joint fit, after
//! its bounded co-collapse reseed multi-start, refuses with the typed "dictionary
//! did not escape total co-collapse" wall (rather than spiralling into a
//! catastrophic / non-finite EV — see the numeric keep-best guard in
//! `enforce_decoder_norm_guard`). The BUG: that refusal was classified
//! NON-recoverable, so when the outer alpha="auto" ρ-search line-searches into a
//! co-collapsing ρ (which, on featureless data, is most of the ρ neighbourhood)
//! the whole fit aborts — the SAME infeasible-ρ pathology #1782 / #2080 fixed for
//! the non-PD Hessian probes, and the thrash that terminates the host process.
//!
//! The fix classifies the co-collapse refusal as a RECOVERABLE value-probe refusal
//! (`is_recoverable_value_probe_refusal`), so every eval lane maps it to the finite
//! collapse wall and the outer optimizer STEERS ρ back toward the feasible region
//! (or ships best-so-far) instead of aborting.
//!
//! This drives the inner joint fit on the EXACT repro geometry (120×32, K=6,
//! circle, ibp_map) at a co-collapsing ρ, and asserts (a) it terminates with the
//! typed co-collapse refusal — never a panic/abort or a non-finite blow-up — and
//! (b) that refusal is classified RECOVERABLE. Pre-fix (b) FAILS (the message is
//! not in the recoverable set); post-fix it PASSES.

use super::*;
use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use ndarray::{Array2, ArrayView2, array, s};
use std::sync::Arc;

/// Deterministic `splitmix64` bit-mixer — a structureless uniform PRNG (no
/// periodic / low-rank pattern a circle chart could latch onto), so the fixture is
/// a faithful, byte-reproducible stand-in for `rng.normal(size=(120, 32))`.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Featureless standard-normal target (Box–Muller over `splitmix64`) — the exact
/// `X = rng.normal(size=(n, p))` regime from the #2089 repro.
fn white_noise_gaussian_target(n: usize, p: usize) -> Array2<f64> {
    let mut state: u64 = 0x2089_2089_2089_2089;
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        for col in 0..p {
            let u1 = uniform01(&mut state).max(1.0e-300);
            let u2 = uniform01(&mut state);
            z[[row, col]] = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        }
    }
    z
}

/// K-atom, d=1 periodic (circle) SAE term with IBP-MAP assignment, seeded the way
/// the production cold path does (PCA-seed the coordinates, ridge-LSQ each decoder).
/// `harmonics` sets the basis size `m = 1 + 2·harmonics`. Mirrors the #2080 fixture.
fn white_noise_circle_term(z: ArrayView2<'_, f64>, k: usize, harmonics: usize) -> SaeManifoldTerm {
    let n = z.nrows();
    let p = z.ncols();
    let dim = 1usize;
    let num_basis = 1 + 2 * harmonics;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(num_basis).unwrap());
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let atom_dims = vec![dim; k];
    let seed_coords = sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims).unwrap();
    let mut atoms = Vec::with_capacity(k);
    let mut coords_blocks = Vec::with_capacity(k);
    let mut manifolds = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let coords = seed_coords.slice(s![atom_idx, .., 0..dim]).to_owned();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mm = phi.ncols();
        // Cold (zero) decoders: on featureless data the joint fit cannot grow K=6
        // circle decoders out of the null floor at a non-trivial smoothing ρ, so
        // the dictionary totally co-collapses and the reseed multi-start fires.
        let decoder = Array2::<f64>::zeros((mm, p));
        let atom = SaeManifoldAtom::new(
            "circle",
            SaeAtomBasisKind::Periodic,
            dim,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(mm),
        )
        .unwrap()
        .with_basis_evaluator(evaluator.clone());
        atoms.push(atom);
        coords_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let logits = Array2::<f64>::from_elem((n, k), 6.0);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let assignment =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coords_blocks, manifolds, mode)
            .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// #2089 — the featureless K=6 circle / ibp_map inner fit at a co-collapsing ρ must
/// TERMINATE cleanly (a finite result or the typed co-collapse refusal, never a
/// panic/abort or a non-finite blow-up), AND that refusal must be classified as a
/// RECOVERABLE infeasible-ρ probe so the outer alpha="auto" search steers around it
/// instead of aborting the whole fit. Pre-fix the co-collapse refusal was
/// non-recoverable (the outer search aborted / thrashed the host to the exit-137
/// SIGKILL); post-fix it is the finite collapse wall the search steers around.
#[test]
pub(crate) fn cocollapse_refusal_is_recoverable_infeasible_rho_2089() {
    // Mirror the issue's repro dimensions: X = rng.normal(size=(120, 32)), K = 6.
    let n = 120usize;
    let p = 32usize;
    let k = 6usize;
    let harmonics = 2usize; // m = 5: [1, sin2πt, cos2πt, sin4πt, cos4πt]

    let z = white_noise_gaussian_target(n, p);
    let mut term = white_noise_circle_term(z.view(), k, harmonics);

    // A non-trivial smoothing ρ: featureless data cannot support K=6 circle charts
    // there, so the dictionary totally co-collapses and the bounded reseed
    // multi-start fires. `log_lambda_smooth = 2.0` sits well inside the co-collapse
    // regime (a healthy fit only survives at near-zero smoothing on this input).
    let mut rho = SaeManifoldRho::new(0.0, 2.0, (0..k).map(|_| array![0.0]).collect());

    // The inner fit must terminate cleanly — never panic/abort. It either returns a
    // finite result (the reseed recovered) or the typed co-collapse refusal.
    let result = term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 0.05, 1.0e-3, 1.0e-3);

    match result {
        Ok(loss) => {
            // If the reseed managed to recover, the loss and EV must at least be
            // finite and non-catastrophic — never the −625 blow-up the numeric
            // keep-best guard prevents.
            let ev = term.dictionary_reconstruction_ev(z.view(), &rho).unwrap();
            eprintln!(
                "[#2089 repro] featureless K=6 circle recovered at this ρ: EV={ev:.4}, loss={:.4e}",
                loss.total()
            );
            assert!(
                loss.total().is_finite() && ev.is_finite() && ev > -1.0,
                "#2089: co-collapse fit blew up instead of keeping the incumbent \
                 (loss={:?}, EV={ev})",
                loss.total()
            );
        }
        Err(err) => {
            eprintln!("[#2089 repro] featureless K=6 circle co-collapse refusal: {err}");
            // The reseed path fired and refused with the typed total-co-collapse
            // wall (not a panic, not a generic defect).
            assert!(
                err.contains("did not escape total co-collapse"),
                "#2089: expected the typed total-co-collapse refusal from the reseed path, \
                 got a different error: {err}"
            );
            // THE FIX: that refusal must be a RECOVERABLE infeasible-ρ probe so the
            // outer alpha=\"auto\" optimizer reads it as the finite collapse wall and
            // steers ρ back toward the feasible region — NOT a fatal abort that (with
            // the pre-guard reseed thrash) SIGKILLs the host. Pre-fix this assertion
            // FAILS; post-fix it PASSES.
            assert!(
                SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(&err),
                "#2089: the total-co-collapse refusal must be classified as a RECOVERABLE \
                 infeasible-ρ probe (the outer search must steer around a co-collapsing ρ, \
                 not abort the whole alpha=\"auto\" fit). Refusal: {err}"
            );
        }
    }
}
