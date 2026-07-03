//! Owed-work regression gate for issue #993: drive the #975/#984 within-atom
//! carve instrument END TO END on a REAL fitted SAE term — not a synthetic
//! `CarveInput` stub, and not a direct call to `carve_input_from_fitted_atom`,
//! but through the production discovery seam `harvest_move_proposals`, which is
//! what a real fit's structure search actually calls.
//!
//! The companion gate `tests/owed_993.rs` plants decoder coefficients and calls
//! the producer (`carve_input_from_fitted_atom`) directly. This gate closes the
//! remaining gap: it builds a genuine `d = 2` torus PRODUCT atom into a real
//! [`SaeManifoldTerm`] (real basis evaluator, real fitted decoder, real
//! softmax routing), runs the harvest pass, and asserts the within-atom carve
//! actually RAN on that atom's own fitted decoder-coefficient covariance —
//! `fission_carve_ran_count > 0` — and that its binding verdict flows through to
//! the fission-proposal stream. The pre-existing in-crate test only exercises
//! the *negative* loud-skip path (1-D periodic atoms, where the product carve
//! is undefined and `fission_carve_ran_count == 0`); this is its positive twin.
//!
//! Wiring under test (all already on the fitted object, nothing forked):
//!   real atom `basis_values` (Φ, the fused torus design) +
//!   `decoder_coefficients` (B, the fitted decoder) +
//!   `basis_evaluator.factor_basis_sizes()` (the Kronecker split)
//!     → `run_within_atom_carve` → `carve_input_from_fitted_atom`
//!     (REML re-fit → scale-included `coeff_covariance`, the mgcv-`Vb` contract)
//!     → `carve` → `FissionDecision` → fission proposal gate.
//!
//! Two arms pin the binding test's two outcomes on a real fit:
//!   1. SEPARABLE parent decoder  → carve permits the split → a fission on the
//!      parent atom is proposed (and `fission_carve_blocked_count == 0`).
//!   2. BOUND parent decoder (real θ₁·θ₂ interaction) → carve proves binding →
//!      `fission_carve_blocked_count > 0` and NO fission on the parent.
//! Both arms require `fission_carve_ran_count > 0`: the instrument is genuinely
//! running on the real fitted atom, not degrading to the energy-fraction stub.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::solver::structure_search::StructureMove;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::structure_harvest::{HarvestParams, harvest_move_proposals};
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
    SaeManifoldRho, SaeManifoldTerm, TorusHarmonicEvaluator,
};

const N_HARMONICS: usize = 2;
const N: usize = 240;
const ON: f64 = 6.0;
const OFF: f64 = -6.0;

/// Deterministic interleaved torus code sample on `[0,1)²` (coprime strides give
/// full 2-D coverage, no degenerate lattice).
fn torus_coords() -> Array2<f64> {
    let mut coords = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        coords[[i, 0]] = (i as f64 + 0.5) / N as f64;
        coords[[i, 1]] = (((i * 137) % N) as f64 + 0.5) / N as f64;
    }
    coords
}

/// A SEPARABLE decoder `B` (`m·m × p`): purely additive in the centered factor
/// bases (no interaction block), column order `flat = j·m + k` — the constant-
/// leading convention the producer reads. Mirrors `owed_993.rs::separable_decoder`.
fn separable_decoder(m: usize, p: usize) -> Array2<f64> {
    let mut b = Array2::<f64>::zeros((m * m, p));
    for d in 0..p {
        let alpha = 1.0 + d as f64;
        let beta = 0.5 - 0.3 * d as f64;
        for j in 0..m {
            let a_j = ((j + 1) as f64).recip() * alpha;
            for k in 0..m {
                let b_k = ((k + 1) as f64).recip() * beta;
                let coeff = if k == 0 { a_j } else { 0.0 } + if j == 0 { b_k } else { 0.0 };
                b[[j * m + k, d]] = coeff;
            }
        }
    }
    b
}

/// A BOUND decoder: the separable part PLUS a genuine bilinear interaction block
/// on the curved×curved columns (a real `θ₁·θ₂` coupling).
fn bound_decoder(m: usize, p: usize) -> Array2<f64> {
    let mut b = separable_decoder(m, p);
    for d in 0..p {
        for j in 1..m {
            for k in 1..m {
                b[[j * m + k, d]] += 0.8 / ((j * k) as f64);
            }
        }
    }
    b
}

/// Build a real 2-atom torus SAE term. Atom 0 (the parent) is a genuine `d = 2`
/// torus PRODUCT atom carrying `parent_decoder`; atom 1 (the child) is a second
/// torus atom whose routing support nests strictly inside atom 0's, so the
/// harvest's absorption-asymmetry audit flags atom 0 for fission — and the
/// within-atom carve then runs on atom 0's own fitted decoder/covariance.
fn nested_torus_term(parent_decoder: Array2<f64>) -> (SaeManifoldTerm, SaeManifoldRho, usize) {
    let coords = torus_coords();
    let evaluator = Arc::new(TorusHarmonicEvaluator::new(2, N_HARMONICS).unwrap());
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let (m_a, m_b) = evaluator.factor_basis_sizes().unwrap();
    assert_eq!(m_a, m_b);
    let m = m_a;
    assert_eq!(phi.ncols(), m * m);
    let p = parent_decoder.ncols();

    // Child decoder: a distinct, non-degenerate (separable) direction so the
    // child reconstructs something of its own; its carve outcome is irrelevant
    // (only the parent is the audited fission candidate), but it must be a valid
    // atom. Give it a different column footprint than the parent.
    let mut child_decoder = Array2::<f64>::zeros((m * m, p));
    for d in 0..p {
        child_decoder[[1, d]] = 0.4 + 0.1 * d as f64;
        child_decoder[[m, d]] = -0.3 + 0.05 * d as f64;
    }

    let make_atom = |name: &str, decoder: Array2<f64>| -> SaeManifoldAtom {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Torus,
            2,
            phi.clone(),
            jet.clone(),
            decoder,
            Array2::<f64>::eye(m * m),
        )
        .unwrap()
        .with_basis_evaluator(evaluator.clone() as Arc<dyn SaeBasisEvaluator>)
    };

    let atoms = vec![
        make_atom("torus_parent", parent_decoder),
        make_atom("torus_child", child_decoder),
    ];

    // Routing: parent active on rows ≡ 0 mod 2 PLUS rows ≡ 1 mod 4; child active
    // only on rows ≡ 0 mod 4 — strictly nested in the parent's support, so
    // P(parent|child) = 1, P(child|parent) < 1 ⇒ absorption asymmetry on atom 0.
    let mut logits = Array2::<f64>::zeros((N, 2));
    for row in 0..N {
        let parent = row % 2 == 0 || row % 4 == 1;
        let child = row % 4 == 0;
        logits[[row, 0]] = if parent { ON } else { OFF };
        logits[[row, 1]] = if child { ON } else { OFF };
    }

    let manifold = LatentManifold::Product(vec![
        LatentManifold::Circle { period: 1.0 },
        LatentManifold::Circle { period: 1.0 },
    ]);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone(), coords.clone()],
        vec![manifold.clone(), manifold],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();

    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(2); 2]);
    (term, rho, m)
}

fn residuals_of(term: &SaeManifoldTerm) -> Array2<f64> {
    let fitted = term.try_fitted().unwrap();
    -&fitted
}

fn harvest_params() -> HarvestParams {
    HarvestParams {
        max_fusions: 4,
        max_fissions: 4,
        max_births: 0,
    }
}

/// A real fitted torus atom with a SEPARABLE decoder: the within-atom carve RUNS
/// on the atom's own fitted decoder-coefficient covariance, permits the split,
/// and a fission on the parent atom is proposed.
#[test]
fn real_fitted_separable_torus_atom_runs_carve_and_proposes_fission() {
    let separable = {
        let evaluator = TorusHarmonicEvaluator::new(2, N_HARMONICS).unwrap();
        let (m, _) = evaluator.factor_basis_sizes().unwrap();
        separable_decoder(m, 3)
    };
    let (term, rho, _m) = nested_torus_term(separable);
    let residuals = residuals_of(&term);
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &harvest_params()).unwrap();

    assert!(
        report.fission_carve_ran_count > 0,
        "the within-atom carve must RUN on the real fitted torus parent atom (a genuine d=2 \
         product); instead it skipped (ran={}, unavailable={})",
        report.fission_carve_ran_count,
        report.fission_carve_unavailable_count
    );
    assert_eq!(
        report.fission_carve_blocked_count, 0,
        "a separable decoder must NOT have its fission blocked by the binding test"
    );
    assert!(
        !report.fission_carve_results.is_empty(),
        "a carve that ran must record its result for the ledger"
    );
    // The carve permitted the split ⇒ the parent fission rides as a proposal.
    let parent_fissioned = report
        .proposals
        .iter()
        .any(|pr| matches!(pr.mv, StructureMove::Fission { atom: 0 }));
    assert!(
        parent_fissioned,
        "the separable parent atom must be proposed for fission once the carve permits it; \
         proposals = {:?}",
        report.proposals.iter().map(|pr| &pr.mv).collect::<Vec<_>>()
    );
}

/// A real fitted torus atom with a BOUND decoder (a genuine θ₁·θ₂ interaction):
/// the within-atom carve RUNS on the atom's own fitted covariance, PROVES
/// binding, blocks the fission, and the parent is NOT proposed for fission.
#[test]
fn real_fitted_bound_torus_atom_runs_carve_and_blocks_fission() {
    let bound = {
        let evaluator = TorusHarmonicEvaluator::new(2, N_HARMONICS).unwrap();
        let (m, _) = evaluator.factor_basis_sizes().unwrap();
        bound_decoder(m, 3)
    };
    let (term, rho, _m) = nested_torus_term(bound);
    let residuals = residuals_of(&term);
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &harvest_params()).unwrap();

    assert!(
        report.fission_carve_ran_count > 0,
        "the within-atom carve must RUN on the real fitted bound torus parent atom; \
         instead it skipped (ran={}, unavailable={})",
        report.fission_carve_ran_count,
        report.fission_carve_unavailable_count
    );
    assert!(
        report.fission_carve_blocked_count > 0,
        "a bound decoder (real θ₁·θ₂ interaction) must have its fission BLOCKED by the \
         binding test; blocked={}",
        report.fission_carve_blocked_count
    );
    // Binding proven ⇒ the parent must NOT be proposed for fission.
    let parent_fissioned = report
        .proposals
        .iter()
        .any(|pr| matches!(pr.mv, StructureMove::Fission { atom: 0 }));
    assert!(
        !parent_fissioned,
        "a bound parent atom whose carve proves binding must NOT be proposed for fission; \
         proposals = {:?}",
        report.proposals.iter().map(|pr| &pr.mv).collect::<Vec<_>>()
    );

    // The carve also recorded a Keep decision for the parent.
    let parent_kept = report.fission_carve_results.iter().any(|r| {
        r.atom == 0
            && matches!(
                r.decision,
                gam::terms::structure::anova_atom::FissionDecision::Keep
            )
    });
    assert!(
        parent_kept,
        "the bound parent's carve result must record a Keep decision; results = {:?}",
        report.fission_carve_results
    );
}
