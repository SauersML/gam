#![cfg(test)]
//! #4611 — documenting tests for the mid-fit #1117 contraction under a
//! PRODUCTION-shaped roughness (follow-up to #4610's finding 1).
//!
//! `penalized_quasi_laplace_criterion`'s refine loop re-enters
//! `run_joint_fit_arrow_schur_for_quasi_laplace` per round
//! (construction_undamped_inner.rs:1385), and each entry re-runs
//! `prepare_entry_stages` → `reduce_atoms_to_data_supported_rank`
//! (fit_drivers.rs:6244) on the LIVE, MOVED term — the #1117 doc's "the
//! depth decision is made once here and held FIXED across the inner Newton
//! walk" (fit_drivers.rs:6870) does not hold for the evidence lane. When the
//! walk passes through a transient coordinate collapse, the reduction
//! snapshots a numerically rank-deficient data Gram and PERMANENTLY
//! contracts the atom (`reduce_basis_to_subspace` wraps the evaluator).
//!
//! Measured on this fixture family (issue #4611, probe branch, entry
//! ordinals from an instrumented reduction):
//!
//! - TPS roughness `S = diag(0, 0, 1e6)` on atom B — the exact
//!   second-derivative energy on the `[1, s, s²]` patch basis; production
//!   `Polynomial`-resolution atoms use the same `EuclideanPatchEvaluator`
//!   (geometry_plan.rs:650) with derivative-energy Grams — CONTRACTS
//!   mid-fit even with channel-separated decoders (B: 3→2 at the 3rd
//!   entry), and with blurred decoders contracts BOTH atoms (A: 3→2 at the
//!   2nd entry on gram evals `[2.379e-13, 8.823e-7, 2.067e0]` against
//!   cutoff `2.067e-9`, then 2→1 at the 3rd entry on an EXACT zero;
//!   B: 3→2 at `2.106e-10` = 0.53×cutoff). Atom A carries an IDENTITY
//!   roughness — the contraction propagates through the mixture coupling,
//!   not through the atom's own penalty. κ ∈ {1e4, 1e6, 1e8} all end at
//!   basis sizes `[1, 2]`.
//! - The planted rank-1 projector `I − vvᵀ` that triggered the 11-12
//!   fixture observation now REFUSES (objective stalled 3 consecutive
//!   refine rounds → off-optimum refusal) without contracting: the
//!   synthetic projector was never the trigger, and channel separation is
//!   NOT load-bearing against the production-shaped one.
//! - That refusal is BUDGET-INDEPENDENT: inner budgets 60 / 240 / 960 all
//!   refuse with the same stall kind, the KKT gradient norm frozen to six
//!   digits (‖g‖ = 2.357e-2, ~525× over tolerance), the objective frozen
//!   at 9.717682, and sizes `[3, 3]` at every rung. The walk is at a fixed
//!   point of its own dynamics (both line-search families find zero
//!   band-material drop on all 20 finite trials) — "raise the budget" is
//!   not a remedy for this arm; the fail-closed refusal is the stable
//!   behavior.
//!
//! These tests PIN the current behavior so any change (#4611's remedies:
//! per-evaluation freeze, entry-state-derived rank, or a contraction
//! marker) turns them red and forces the update — the fixture story, the
//! entry ordinals, and the dose-response live in the issue.

use super::*;
use crate::basis::EuclideanPatchEvaluator;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

const LO: f64 = -1.0;
const HI: f64 = 1.0;

/// The two-atom interval fixture with a caller-provided DECODER for atom B
/// (the channel-mixing knob) and a caller-provided roughness Gram for atom B
/// (the penalty-shape knob). Mirrors `two_atom_interval_fixture_with_grams`
/// in `tests_half_line_channels_4077.rs`, with decoder B opened up.
#[allow(clippy::type_complexity)]
fn two_atom_fixture_open_decoder_b(
    decoder_b: &Array2<f64>,
    gram_b: impl FnOnce(usize) -> Array2<f64>,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 8usize;
    let p = 2usize;
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("patch basis"));
    let truth_a: [f64; 8] = [-0.7, -0.3, 0.1, 0.4, 1.6, 1.8, 2.0, 0.8];
    let truth_b: [f64; 8] = [0.8, 2.0, 1.8, 1.6, 0.4, 0.1, -0.3, -0.7];
    let coords_a =
        Array2::<f64>::from_shape_fn((n, 1), |(row, _)| 0.8 * truth_a[row].clamp(-0.9, 0.9));
    let coords_b =
        Array2::<f64>::from_shape_fn((n, 1), |(row, _)| 0.8 * truth_b[row].clamp(-0.9, 0.9));
    let (phi_a, jet_a) = evaluator
        .evaluate(coords_a.view())
        .expect("coords evaluate");
    let (phi_b, jet_b) = evaluator
        .evaluate(coords_b.view())
        .expect("coords evaluate");
    let width_a = phi_a.ncols();
    let width_b = phi_b.ncols();
    assert_eq!(
        decoder_b.dim(),
        (width_b, p),
        "decoder B must be (width_b, p)"
    );
    let decoder_a = Array2::<f64>::from_shape_fn((width_a, p), |(b, o)| {
        [[0.12, 0.01], [1.0, 0.03], [0.25, -0.02]][b][o]
    });
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let (s_a, s_b) = (truth_a[row], truth_b[row]);
        for o in 0..p {
            let decode = |d: &Array2<f64>, s: f64| d[[0, o]] + d[[1, o]] * s + d[[2, o]] * s * s;
            target[[row, o]] = 0.5 * decode(&decoder_a, s_a)
                + 0.5 * decode(decoder_b, s_b)
                + 0.03 * (1.7 * row as f64 + 0.9 * o as f64).sin();
        }
    }
    let atom_a = SaeManifoldAtom::new_with_provided_function_gram(
        "interval_a".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi_a,
        jet_a,
        decoder_a,
        Array2::<f64>::eye(width_a),
    )
    .expect("atom A shapes agree")
    .with_basis_second_jet(evaluator.clone());
    let atom_b = SaeManifoldAtom::new_with_provided_function_gram(
        "interval_b".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi_b,
        jet_b,
        decoder_b.clone(),
        gram_b(width_b),
    )
    .expect("atom B shapes agree")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 2)),
        vec![coords_a, coords_b],
        vec![
            LatentManifold::Interval { lo: LO, hi: HI },
            LatentManifold::Interval { lo: LO, hi: HI },
        ],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom_a, atom_b], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 1.0, vec![array![-3.0], array![-3.0]]);
    (term, target, rho)
}

/// The 11-12 "blurred" decoder B: comparable mass on BOTH outputs, so atom B
/// chases the ½/½ mixture on both channels instead of owning output 1.
fn blurred_decoder_b() -> Array2<f64> {
    Array2::<f64>::from_shape_fn((3, 2), |(b, o)| {
        [[0.30, 0.26], [0.42, 0.38], [0.14, 0.10]][b][o]
    })
}

/// The shipped fixture's channel-separated decoder B (dominant on output 1).
fn separated_decoder_b() -> Array2<f64> {
    Array2::<f64>::from_shape_fn((3, 2), |(b, o)| {
        [[0.02, -0.15], [-0.03, 0.7], [0.01, -0.3]][b][o]
    })
}

/// The production thin-plate shape on the `[1, s, s²]` patch basis: the
/// second-derivative roughness `∫ (f″)² ds` penalizes ONLY the quadratic
/// coefficient. `κ` stiff; the `{1, s}` border null is the frame's own null
/// space, not a planted one.
fn tps_penalty(kappa: f64) -> impl FnOnce(usize) -> Array2<f64> {
    move |width: usize| {
        let mut s = Array2::<f64>::zeros((width, width));
        for i in 0..width {
            s[[i, i]] = if i + 1 == width { kappa } else { 0.0 };
        }
        s
    }
}

/// Run the evidence-lane criterion (refine rounds re-enter the entry stages
/// on the live term). `Ok`/`Err` is returned so callers can pin the exact
/// lane outcome (a refusal is the fail-closed discipline, not a failure of
/// this helper).
fn run_criterion(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Result<(), String> {
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        rho,
        None,
        60,
        0.4,
        1.0e-8,
        1.0e-8,
    )
    .map(|_| ())
    .map_err(|err| err.to_string())
}

/// The planting discipline of #4609/#4610: converge an identity-gram probe
/// first, then return the top eigenvector of atom B's FITTED decoder gram —
/// the direction the data itself supports best, so the planted projector
/// `I − vvᵀ` removes exactly the data-favorite direction from the penalty.
fn planted_projector_direction() -> Array1<f64> {
    let eye = |width: usize| Array2::<f64>::eye(width);
    let (mut probe, probe_target, probe_rho) =
        two_atom_fixture_open_decoder_b(&blurred_decoder_b(), eye);
    probe
        .penalized_quasi_laplace_criterion_with_cache(
            probe_target.view(),
            &probe_rho,
            None,
            60,
            0.4,
            1.0e-8,
            1.0e-8,
        )
        .expect("the identity-gram blurred probe converges before planting");
    let mut grams = probe.empty_decoder_gram_accumulator();
    probe
        .accumulate_decoder_gram(&mut grams)
        .expect("the probe term accumulates both atoms' decoder grams");
    let mut gram_b = grams[1].clone();
    for i in 0..gram_b.nrows() {
        for j in 0..i {
            let value = 0.5 * (gram_b[[i, j]] + gram_b[[j, i]]);
            gram_b[[i, j]] = value;
            gram_b[[j, i]] = value;
        }
    }
    let (evals, evecs) = gram_b
        .eigh(Side::Lower)
        .expect("atom B's probe gram eigendecomposes");
    let top = evals
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).expect("finite eigenvalues"))
        .expect("the gram has a spectrum")
        .0;
    evecs.column(top).to_owned()
}

/// The planted-projector fixture: atom B's roughness Gram is `I − vvᵀ`
/// with `v` the fitted gram's top direction.
fn planted_projector_fixture(v: &Array1<f64>) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    two_atom_fixture_open_decoder_b(&blurred_decoder_b(), |width| {
        let mut planted = Array2::<f64>::eye(width);
        for i in 0..width {
            for j in 0..width {
                planted[[i, j]] -= v[i] * v[j];
            }
        }
        planted
    })
}

/// #4611 — the 11-12 configuration (planted rank-1 projector on atom B's
/// top data-Gram direction, blurred decoder B) currently REFUSES at this
/// budget without contracting: the objective stalls for 3 consecutive
/// refine rounds and the lane declines to rank an off-optimum criterion.
/// The synthetic projector is NOT the mid-fit contraction trigger on
/// current bytes.
#[test]
fn planted_projector_blurred_refuses_without_contracting_4611() {
    let v = planted_projector_direction();
    let (mut term, target, rho) = planted_projector_fixture(&v);
    let outcome = run_criterion(&mut term, &target, &rho);
    let refusal = outcome.expect_err(
        "the planted-projector arm currently refuses (objective stalled across refine rounds); \
         if it now converges, re-measure the contraction verdict and update #4611",
    );
    assert!(
        refusal.contains("did not converge"),
        "unexpected refusal kind: {refusal}"
    );
    assert_eq!(
        (term.atoms[0].basis_size(), term.atoms[1].basis_size()),
        (3, 3),
        "the planted-projector arm must not contract while refusing"
    );
}

/// #4611 — the production-shaped TPS roughness `diag(0, 0, κ)` on atom B
/// contracts atom B mid-fit EVEN with the shipped fixture's channel-SEPARATED
/// decoder (the #4610 fixture doc's load-bearing choice — not load-bearing
/// here). The criterion then converges on the contracted model.
#[test]
fn tps_stiff_contracted_midfit_channel_separated_4611() {
    let (mut term, target, rho) =
        two_atom_fixture_open_decoder_b(&separated_decoder_b(), tps_penalty(1.0e6));
    assert_eq!(
        (term.atoms[0].basis_size(), term.atoms[1].basis_size()),
        (3, 3),
        "the seed state is full-width"
    );
    run_criterion(&mut term, &target, &rho)
        .expect("the channel-separated TPS arm currently converges (on the contracted model)");
    assert_eq!(
        (term.atoms[0].basis_size(), term.atoms[1].basis_size()),
        (3, 2),
        "current #4611 behavior: atom B contracts 3->2 mid-fit under the TPS roughness \
         while atom A keeps its width; a change here means one of #4611's remedies landed"
    );
}

/// #4611 — the same TPS roughness with the BLURRED decoder contracts BOTH
/// atoms mid-fit, including the identity-roughness partner atom A
/// (3→2→1 across two consecutive entries in the probe trace), and the
/// dose-response is flat across the stiff regime: κ ∈ {1e4, 1e6, 1e8} all
/// end at `[1, 2]`.
#[test]
fn tps_stiff_contracted_midfit_blurred_dose_response_4611() {
    for kappa in [1.0e4, 1.0e6, 1.0e8] {
        let (mut term, target, rho) =
            two_atom_fixture_open_decoder_b(&blurred_decoder_b(), tps_penalty(kappa));
        run_criterion(&mut term, &target, &rho)
            .unwrap_or_else(|err| panic!("kappa={kappa:.0e}: unexpected refusal: {err}"));
        let sizes: Vec<usize> = term.atoms.iter().map(|atom| atom.basis_size()).collect();
        assert_eq!(
            sizes,
            vec![1, 2],
            "kappa={kappa:.0e}: current #4611 behavior contracts BOTH atoms (A through the \
             mixture coupling, not its own penalty); a change here means one of #4611's \
             remedies landed"
        );
        // The contraction is permanent for the evaluator: a SECOND criterion
        // call at the same ρ starts from the contracted widths (the wrapped
        // evaluator survives the refresh), so the evidence this lane returns
        // is priced on the contracted model from the first contraction on.
        run_criterion(&mut term, &target, &rho)
            .unwrap_or_else(|err| panic!("kappa={kappa:.0e}: second call refused: {err}"));
        let sizes_again: Vec<usize> = term.atoms.iter().map(|atom| atom.basis_size()).collect();
        assert_eq!(
            sizes_again,
            vec![1, 2],
            "kappa={kappa:.0e}: the contraction must persist across criterion calls"
        );
    }
}

/// #4611 — the planted-projector arm's refusal is BUDGET-INDEPENDENT: at
/// inner budgets 60 / 240 / 960 (16× the shipped one) the walk refuses with
/// the same stall kind ("objective stalled for 3 consecutive refine
/// rounds"), the KKT gradient norm is frozen to six digits (‖g‖ = 2.357e-2
/// → 2.3567e-2, still ~525× over tolerance), the objective is frozen at
/// 9.717682, and neither atom contracts at any rung. The walk is at a fixed
/// point of its own dynamics — both line-search families (orbit gauge and
/// ambient) find zero band-material objective drop on all 20 finite trials
/// — NOT a slow grind that more iterations would resolve. "Raise the
/// budget" is not a remedy for this arm; the fail-closed refusal is the
/// stable behavior on current bytes.
#[test]
fn planted_projector_refusal_budget_independent_4611() {
    let v = planted_projector_direction();
    for budget in [60usize, 240, 960] {
        let (mut term, target, rho) = planted_projector_fixture(&v);
        let outcome = term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            budget,
            0.4,
            1.0e-8,
            1.0e-8,
        );
        let refusal = outcome.map_err(|err| err.to_string()).expect_err(&format!(
            "budget={budget}: the planted-projector arm refuses at every inner budget; \
                 if it now converges, re-measure the verdict and update #4611"
        ));
        assert!(
            refusal.contains("did not converge"),
            "budget={budget}: unexpected refusal kind: {refusal}"
        );
        assert!(
            refusal.contains("objective stalled"),
            "budget={budget}: expected the stall refusal kind: {refusal}"
        );
        assert_eq!(
            (term.atoms[0].basis_size(), term.atoms[1].basis_size()),
            (3, 3),
            "budget={budget}: the planted-projector arm must not contract at any budget"
        );
    }
}
