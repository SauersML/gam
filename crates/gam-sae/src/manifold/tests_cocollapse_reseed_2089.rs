//! #2089 — a tiny featureless SAE fit that TOTALLY co-collapses must not abort /
//! SIGKILL the outer alpha="auto" REML search: the co-collapse refusal is a
//! RECOVERABLE infeasible-ρ probe the outer optimizer steers around, not a fatal
//! error.
//!
//! Repro (from the issue): `gamfit.sae_manifold_fit` on
//! `X = rng.normal(size=(120, 32))`, `K = 6`, `d_atom = 1`,
//! `atom_topology = "circle"`, `assignment = "ordered_beta_bernoulli"`, `alpha = "auto"`
//! terminated the Python PROCESS with exit 137 (SIGKILL) — uncatchable by
//! `try/except`.
//!
//! White noise carries no low-rank circle structure. When the smoothing ρ is large enough that
//! every seeded decoder's gated signal becomes numerically unobservable, the inner joint fit,
//! after its bounded co-collapse reseed multi-start, refuses with the typed "dictionary did not
//! escape total co-collapse" error (rather than spiralling into a catastrophic / non-finite EV —
//! see the numeric keep-best guard in `enforce_decoder_norm_guard`). The BUG: that refusal was classified
//! NON-recoverable, so when the outer alpha="auto" ρ-search line-searches into a
//! co-collapsing ρ (which, on featureless data, is most of the ρ neighbourhood)
//! the whole fit aborts — the SAME infeasible-ρ pathology #1782 / #2080 fixed for
//! the non-PD Hessian probes, and the thrash that terminates the host process.
//!
//! The fix classifies the co-collapse refusal as a RECOVERABLE value-probe refusal
//! (`is_recoverable_value_probe_refusal`), so every eval lane returns an infeasible
//! trial and the outer optimizer STEERS ρ back toward the feasible region instead
//! of treating a made-up finite cost as evidence.
//!
//! Both tests drive the inner joint fit on the EXACT repro geometry (120×32, K=6, circle,
//! ordered_beta_bernoulli), with each decoder seeded by least squares at the PCA-seeded charts, as
//! the production cold path does. An entry refuses an identically zero decoder (#2822).
//! - At `log_lambda_smooth = 2.0` the seeded fit recovers. The test pins that it terminates finite:
//!   never a panic/abort or a non-finite blow-up.
//! - At `log_lambda_smooth = 40.0` every atom's gated decoder signal sits at its computed roundoff
//!   floor, far below the vanished-atom certificate's derived boundary, so the fit refuses with the
//!   typed co-collapse error. The test pins that the refusal is classified RECOVERABLE. Pre-fix
//!   that assertion FAILS (the message is not in the recoverable set); post-fix it PASSES.

#![cfg(test)]

use super::*;
use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use ndarray::{Array2, ArrayView2, array, s};
use std::sync::Arc;

// Deterministic `splitmix64` bit-mixer — a structureless uniform PRNG (no
// periodic / low-rank pattern a circle chart could latch onto), so the fixture is
// a faithful, byte-reproducible stand-in for `rng.normal(size=(120, 32))`.
use gam_linalg::utils::splitmix64;

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

/// K-atom, d=1 periodic (circle) SAE term with ordered Beta--Bernoulli assignment and PCA-seeded
/// coordinates. The decoders are placeholders: [`seeded_featureless_circle_fixture`] seeds each by
/// least squares before any entry, as the production cold path does. `harmonics` sets the basis
/// size `m = 1 + 2·harmonics`. Mirrors the #2080 fixture.
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
        // A placeholder decoder, seeded by least squares before any entry refuses a zero decoder.
        let decoder = Array2::<f64>::zeros((mm, p));
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let assignment =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coords_blocks, manifolds, mode)
            .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// The #2089 repro geometry at a given smoothing ρ. The target is `X = rng.normal(size=(120, 32))`,
/// with K = 6 circles of two harmonics (m = 5: [1, sin2πt, cos2πt, sin4πt, cos4πt]), and each
/// decoder is seeded by least squares at the PCA-seeded charts.
fn seeded_featureless_circle_fixture(
    log_lambda_smooth: f64,
) -> (Array2<f64>, SaeManifoldTerm, SaeManifoldRho) {
    let n = 120usize;
    let p = 32usize;
    let k = 6usize;
    let harmonics = 2usize;
    let z = white_noise_gaussian_target(n, p);
    let mut term = white_noise_circle_term(z.view(), k, harmonics);
    let rho = SaeManifoldRho::new(0.0, log_lambda_smooth, (0..k).map(|_| array![0.0]).collect());
    // #2822 — an entry refuses an identically zero decoder; seed each from the data.
    term.refit_decoder_least_squares_at_current_state(z.view(), Some(&rho))
        .expect("featureless data still spans nonzero least-squares decoders");
    (z, term, rho)
}

/// #2089 — the featureless K=6 circle / ordered_beta_bernoulli inner fit must TERMINATE cleanly (a
/// finite result or the typed co-collapse refusal, never a panic/abort or a non-finite blow-up),
/// AND a co-collapse refusal must be classified as a RECOVERABLE infeasible-ρ probe, so the outer
/// alpha="auto" search steers around it instead of aborting the whole fit. At
/// `log_lambda_smooth = 2.0` the seeded fit recovers (lane probe 1205540: EV 0.7009), so this pins
/// the finite arm. `total_co_collapse_refusal_is_recoverable_at_an_unobservable_rho_2089` pins the
/// refusal arm.
#[test]
pub(crate) fn cocollapse_refusal_is_recoverable_infeasible_rho_2089() {
    let (z, mut term, mut rho) = seeded_featureless_circle_fixture(2.0);

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
            // error (not a panic, not a generic defect).
            assert!(
                err.contains("did not escape total co-collapse"),
                "#2089: expected the typed total-co-collapse refusal from the reseed path, \
                 got a different error: {err}"
            );
            // THE FIX: that refusal must be a RECOVERABLE infeasible-ρ probe so the
            // outer alpha=\"auto\" optimizer reads it as an infeasible trial and
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

/// #2089 — the refusal arm. At `log_lambda_smooth = 40.0` the seeded fit refuses on the certified
/// vanishing arm of the total co-collapse refusal. On lane sweep 1207899 the refusal fired at 37,
/// 40 and 45, and the nearest recovering cell was 30. At 40 every atom's gated decoder signal sits
/// at its computed roundoff floor (largest signal upper bound 5.387e-29, close to 45's 5.701e-29),
/// 7.7e3× below the derived signal boundary 4.152e-25. The margin is measured, not bounded. The
/// test pins that the refusal is typed and classified RECOVERABLE, and it prints the outcome
/// unconditionally.
#[test]
pub(crate) fn total_co_collapse_refusal_is_recoverable_at_an_unobservable_rho_2089() {
    let (z, mut term, mut rho) = seeded_featureless_circle_fixture(40.0);
    let result = term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 0.05, 1.0e-3, 1.0e-3);
    match &result {
        Ok(loss) => eprintln!("[#2089 refusal] recovered, loss={:.4e}", loss.total()),
        Err(err) => eprintln!("[#2089 refusal] {err}"),
    }
    let err = match result {
        Ok(loss) => panic!(
            "#2089: at log_lambda_smooth = 40 every seeded decoder's gated signal must vanish and \
             the fit must refuse; it recovered with loss {:.4e}",
            loss.total()
        ),
        Err(err) => err,
    };
    assert!(
        err.contains("did not escape total co-collapse")
            && err.contains("every atom's actual gated decoder signal vanished"),
        "#2089: expected the typed total co-collapse refusal on its certified-vanishing arm, \
         got: {err}"
    );
    assert!(
        SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(&err),
        "#2089: the total co-collapse refusal must be classified as a RECOVERABLE infeasible-ρ \
         probe. Refusal: {err}"
    );
}
