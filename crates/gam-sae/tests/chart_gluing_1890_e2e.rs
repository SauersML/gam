//! #1890 — END-TO-END chart-gluing verification.
//!
//! `chart_gluing_1890.rs` pins the glue lane at the PROPOSAL level only: it
//! calls `harvest_move_proposals` and asserts a `Glue` proposal with a
//! certifying seam e-value. It never drives the engine, so the
//! bank → certify → apply chain (gam-solve `search`,
//! `structure_search.rs:374-397`, then `apply_structure_move`'s `Glue` arm)
//! is UNVERIFIED there.
//!
//! This suite closes that gap: it drives an over-tiled single circle through
//! the PRODUCTION driver [`run_production_structure_search`] and asserts the
//! glue actually FIRES end-to-end — a `Glue` move is Accepted in the round
//! ledger, the effective (active) atom count drops, the surviving atom absorbs
//! the co-tiled arcs, and the reconstruction is preserved. A negative arm on
//! two genuinely distinct (orthogonal-plane) circles must NOT glue.
//!
//! NOTE on the observable: a glue (like a fusion) applies via `fold_atom_into`,
//! which DEMOTES the folded atom's routing to ~0 (logit → `DEMOTE_LOGIT`) and
//! folds its mass into the survivor — it does NOT shrink `term.k_atoms()` (the
//! column stays, carrying zero mass). So "K drops" is asserted on the ACTIVE
//! atom count (atoms carrying routing mass above the support floor), not on the
//! raw `k_atoms()`.

use gam_sae::assignment::{AssignmentMode, SaeAssignment};
use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::manifold::{SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm};
use gam_sae::structure_harvest::{
    run_production_structure_search, HarvestParams, ProductionRefitParams, RoundDriverConfig,
};
use gam_solve::structure_search::{MoveBudget, MoveVerdict, StructureMove};
use gam_terms::inference::structure_evidence::StructureLedger;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;

const ON: f64 = 6.0;
const OFF: f64 = -6.0;

/// Support floor for "active" atom counting — mirrors the harvest lane's
/// `ACTIVE_SUPPORT_REL_FLOOR / k` (structure_harvest.rs). A demoted atom
/// (`logit = DEMOTE_LOGIT = -40`) carries ~`e^-40` softmax mass, far below this.
const ACTIVE_SUPPORT_REL_FLOOR: f64 = 0.5;

/// Build a K-atom periodic SAE term over `n` rows. Atom `j` owns the contiguous
/// arc `arcs[j] = (start, end)` (disjoint supports) and decodes through
/// `decoders[j]` (a `3 × p` periodic-harmonic decoder). Every atom shares the
/// full-circle coordinate `t = row / n` and a `Circle { period: 1.0 }` manifold.
/// (Copied from `chart_gluing_1890.rs::build_term` so the two suites pin the
/// SAME fixture at the proposal and end-to-end levels.)
fn build_term(n: usize, arcs: &[(usize, usize)], decoders: &[Array2<f64>]) -> SaeManifoldTerm {
    let k = arcs.len();
    assert_eq!(decoders.len(), k);
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    let mut atoms = Vec::with_capacity(k);
    let mut coord_blocks = Vec::with_capacity(k);
    for decoder in decoders.iter() {
        let atom = SaeManifoldAtom::new(
            "arc",
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords.clone());
    }

    let mut logits = Array2::<f64>::zeros((n, k));
    for (atom, &(start, end)) in arcs.iter().enumerate() {
        for row in 0..n {
            logits[[row, atom]] = if row >= start && row < end { ON } else { OFF };
        }
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

fn rho_for(k: usize) -> SaeManifoldRho {
    SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k])
}

/// A decoder that traces a unit circle in the ambient plane `(ax_sin, ax_cos)`.
fn circle_decoder(p: usize, ax_sin: usize, ax_cos: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((3, p));
    d[[1, ax_sin]] = 1.0;
    d[[2, ax_cos]] = 1.0;
    d
}

/// Count atoms carrying routing mass above the support floor on at least one
/// row — the EFFECTIVE dictionary size. A glued (folded) atom is demoted to
/// ~0 mass everywhere and drops out of this count.
fn active_atom_count(term: &SaeManifoldTerm) -> usize {
    let asg = term.assignment.assignments();
    let (n, k) = asg.dim();
    let floor = ACTIVE_SUPPORT_REL_FLOOR / k as f64;
    (0..k)
        .filter(|&a| (0..n).any(|r| asg[[r, a]] > floor))
        .count()
}

/// Rows atom `a` routes above the support floor (its active support).
fn active_rows(term: &SaeManifoldTerm, a: usize) -> Vec<usize> {
    let asg = term.assignment.assignments();
    let (n, k) = asg.dim();
    let floor = ACTIVE_SUPPORT_REL_FLOOR / k as f64;
    (0..n).filter(|&r| asg[[r, a]] > floor).collect()
}

/// Number of `Glue` moves Accepted across all rounds — the direct witness that
/// the bank → certify → apply chain fired.
fn accepted_glues(result: &gam_sae::structure_harvest::StructureSearchResult) -> usize {
    result
        .rounds
        .iter()
        .flat_map(|r| r.moves.iter())
        .filter(|m| {
            matches!(m.mv, StructureMove::Glue { .. })
                && matches!(m.verdict, MoveVerdict::Accepted { .. })
        })
        .count()
}

/// Driver config with ONLY the glue lane able to change the effective size:
/// births/fissions off, `max_fusions > 0` supplies the glue budget (the
/// co-activation fusion lane provably proposes nothing on disjoint arcs), curl
/// off. `max_rounds = 3` lets three co-tiled arcs collapse over rounds.
fn glue_only_config() -> RoundDriverConfig {
    RoundDriverConfig {
        n_shards: 4,
        budget: MoveBudget {
            max_moves: 4,
            alpha: 0.05,
        },
        max_rounds: 3,
        harvest_params: HarvestParams {
            max_fusions: 8,
            max_fissions: 0,
            max_births: 0,
        },
        curl: None,
    }
}

fn refit_params() -> ProductionRefitParams {
    ProductionRefitParams {
        inner_max_iter: 24,
        scoring_inner_max_iter: 8,
        learning_rate: 1.0,
        ridge_ext_coord: 1e-6,
        ridge_beta: 1e-6,
    }
}

/// The reconstruction target: the term's own decoded image (points on the
/// shared circle) plus a small deterministic isometry-band residual — well
/// above the curve-grid quantization floor, well below the ambient radius.
fn target_from_term(term: &SaeManifoldTerm, p: usize) -> Array2<f64> {
    let mut target = term.try_fitted().unwrap();
    let (n, _) = target.dim();
    for row in 0..n {
        for col in 0..p {
            target[[row, col]] += 0.01 * ((row * 7 + col * 3) as f64).sin();
        }
    }
    target
}

#[test]
fn over_tiled_circle_glues_end_to_end() {
    // One unit circle over-tiled into K=3 disjoint arc atoms sharing ONE decoder
    // (the co-activation lane is silent on their anti-correlated codes; only the
    // #1890 glue lane can collapse them).
    let n = 120;
    let k = 3;
    let p = 4;
    let arc = n / k;
    let arcs: Vec<(usize, usize)> = (0..k).map(|j| (j * arc, (j + 1) * arc)).collect();
    let decoders: Vec<Array2<f64>> = (0..k).map(|_| circle_decoder(p, 0, 1)).collect();
    let term = build_term(n, &arcs, &decoders);
    let rho = rho_for(k);
    let target = target_from_term(&term, p);

    assert_eq!(active_atom_count(&term), k, "fixture starts with K active arcs");

    let mut ledger = StructureLedger::new();
    let result = run_production_structure_search(
        term,
        rho,
        target.view(),
        glue_only_config(),
        refit_params(),
        &mut ledger,
    )
    .unwrap();

    let glues = accepted_glues(&result);
    let active_after = active_atom_count(&result.term);
    eprintln!(
        "[1890-e2e] accepted_glues={glues} active_after={active_after} \
         structure_changed={} k_atoms={}",
        result.structure_changed(),
        result.term.k_atoms(),
    );

    // (core) The bank → certify → apply chain fired: at least one Glue Accepted.
    assert!(
        glues >= 1,
        "no Glue move certified end-to-end through the production driver; the \
         proposal-level test passes but bank→certify→apply did not fire"
    );
    // The engine recorded a structural change.
    assert!(result.structure_changed(), "a glue applied but structure_changed() is false");

    // (i) Effective size strictly drops: the co-tiled arcs collapse (folded atoms
    // are demoted to ~0 mass, so this is the ACTIVE count, not raw k_atoms()).
    assert!(
        active_after < k,
        "active atom count did not drop ({active_after} of {k}); the glue demoted no arc"
    );

    // (ii) The surviving atom(s) cover MORE than a single original arc — the
    // fold absorbed at least one co-tiled arc's rows into the survivor.
    let survivor = (0..result.term.k_atoms())
        .max_by_key(|&a| active_rows(&result.term, a).len())
        .unwrap();
    let survivor_rows = active_rows(&result.term, survivor).len();
    assert!(
        survivor_rows > arc,
        "surviving atom routes {survivor_rows} rows, not more than one arc ({arc}); \
         its chart did not absorb the co-tiled arc(s)"
    );

    // (iii) Reconstruction preserved within the isometry band: the merged atom
    // still decodes the whole circle.
    let fitted = result.term.try_fitted().unwrap();
    let mut max_abs = 0.0_f64;
    for (a, b) in fitted.iter().zip(target.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 0.15,
        "post-glue reconstruction diverged from the circle target by {max_abs:.3e} (> 0.15)"
    );
}

#[test]
fn distinct_circles_do_not_glue_end_to_end() {
    // Two DIFFERENT circles (orthogonal ambient planes) with adjacent disjoint
    // supports: the disjoint-support signature is present, but the geometry is
    // not — the driver must NOT glue them.
    let n = 40;
    let p = 4;
    let arcs = [(0usize, 20usize), (20usize, 40usize)];
    let decoders = [circle_decoder(p, 0, 1), circle_decoder(p, 2, 3)];
    let term = build_term(n, &arcs, &decoders);
    let rho = rho_for(2);
    let target = target_from_term(&term, p);

    assert_eq!(active_atom_count(&term), 2);

    let mut ledger = StructureLedger::new();
    let result = run_production_structure_search(
        term,
        rho,
        target.view(),
        glue_only_config(),
        refit_params(),
        &mut ledger,
    )
    .unwrap();

    eprintln!(
        "[1890-e2e-neg] accepted_glues={} active_after={}",
        accepted_glues(&result),
        active_atom_count(&result.term),
    );

    assert_eq!(
        accepted_glues(&result),
        0,
        "distinct orthogonal-plane circles must NOT certify a glue"
    );
    assert_eq!(
        active_atom_count(&result.term),
        2,
        "both distinct circles must remain active (no spurious fold)"
    );
}
