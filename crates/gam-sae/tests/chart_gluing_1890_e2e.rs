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
//! NOTE on the observable: a glue folds the merged atom's mass into the survivor
//! and then EXCISES it (true fusion, `remove_atom`), so both the raw `k_atoms()`
//! and the ACTIVE atom count fall by one. A mere demotion would not survive: the
//! next joint refit's active-mass guard reseeds any ~0-mass atom back to per-row
//! winner parity, resurrecting the atom the glue retired (the #1890 root cause).
//! We assert the drop on the ACTIVE count (atoms carrying routing mass above the
//! support floor) — with removal this equals the raw `k_atoms()`, but the active
//! count is the invariant the observable is about and stays correct even if a
//! future glue variant demotes-without-removing a genuinely covered atom.

use gam_sae::assignment::{AssignmentMode, SaeAssignment};
use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator, SphereChartEvaluator};
use gam_sae::manifold::{
    AtlasOrientability, AtlasSeamKind, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm, UnitSpeedChartTransition,
};
use gam_sae::structure_harvest::{
    HarvestParams, ProductionRefitParams, RoundDriverConfig, run_production_structure_search,
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

/// The SAME unit circle image traced with reversed orientation: `sin` flips sign
/// (`x(t) = (-sin 2πt, cos 2πt)`), so its coordinate map is `A`'s reflected —
/// the transition is the orientation-reversing isometry `t_A = -t_B`. Decoding
/// the identical embedded curve, it is an equivalence (the seam certifier fires),
/// but no single orientable chart can absorb it, so it must REGISTER (Increment 2)
/// rather than fuse.
fn reflected_circle_decoder(p: usize, ax_sin: usize, ax_cos: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((3, p));
    d[[1, ax_sin]] = -1.0;
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

    assert_eq!(
        active_atom_count(&term),
        k,
        "fixture starts with K active arcs"
    );

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
    let raw_after = result.term.k_atoms();
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
    assert!(
        result.structure_changed(),
        "a glue applied but structure_changed() is false"
    );

    // (i) Effective size strictly drops: the co-tiled arcs collapse (folded atoms
    // are demoted to ~0 mass, so this is the ACTIVE count, not raw k_atoms()).
    assert!(
        active_after < k,
        "active atom count did not drop ({active_after} of {k}); the glue demoted no arc"
    );
    assert!(
        raw_after < k,
        "raw atom count did not drop ({raw_after} of {k}); a demoted zombie survived compaction"
    );
    assert_eq!(
        active_after, raw_after,
        "every physically retained atom must carry routing mass after the glue polish"
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
    assert_eq!(
        result.term.k_atoms(),
        2,
        "the negative control must not physically remove either circle"
    );
}

#[test]
fn orientation_reversing_pair_registers_atlas_end_to_end() {
    // Increment 2 — ATLAS REGISTER outcome, end-to-end through the production
    // driver. Two disjoint-support arcs decode the SAME embedded circle, but the
    // second chart is REFLECTED (`sin` sign flipped), so their overlap transition
    // is orientation-reversing (`t_A = -t_B`, sign `-1`). This is a genuine
    // equivalence — the seam certifier fires exactly as for the plain over-tile —
    // yet no single ORIENTABLE chart can represent the union. The driver must
    // therefore take the REGISTER branch (`ChartGlueOutcome::RegisterAtlas`), NOT
    // the destructive fuse: keep BOTH numerical charts active, quotient them into
    // one semantic atom carrying the fitted `-1` transition, and expose the chart
    // gates as `activation × partition-of-unity`.
    let n = 40;
    let p = 4;
    let arcs = [(0usize, 20usize), (20usize, 40usize)];
    // Same ambient plane (0,1), opposite orientation ⇒ same image, reversed chart.
    let decoders = [circle_decoder(p, 0, 1), reflected_circle_decoder(p, 0, 1)];
    let term = build_term(n, &arcs, &decoders);
    let rho = rho_for(2);
    let target = target_from_term(&term, p);
    let fitted_before = term.try_fitted().unwrap();

    assert_eq!(
        active_atom_count(&term),
        2,
        "fixture starts with 2 active arcs"
    );
    assert_eq!(term.semantic_atom_count(), 2, "no atlas registered yet");

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
    eprintln!(
        "[1890-e2e-atlas] accepted_glues={glues} k_atoms={} semantic_atoms={} atlases={}",
        result.term.k_atoms(),
        result.term.semantic_atom_count(),
        result.term.chart_atlases().len(),
    );

    // The bank → certify → apply chain fired on the reversing seam.
    assert!(
        glues >= 1,
        "no Glue certified for the orientation-reversing pair; the atlas-register \
         lane did not fire end-to-end"
    );
    assert!(
        result.structure_changed(),
        "a glue applied but structure_changed() is false"
    );

    // REGISTER, not FUSE: both numerical charts survive (nothing is excised) and
    // both keep routing mass — the register outcome does not demote either chart.
    assert_eq!(
        result.term.k_atoms(),
        2,
        "atlas registration must keep BOTH local charts (register, not destructive fuse)"
    );
    assert_eq!(
        active_atom_count(&result.term),
        2,
        "both charts of a registered atlas stay active under the partition of unity"
    );

    // The two charts are quotiented into ONE semantic atom carrying the seam.
    assert_eq!(
        result.term.semantic_atom_count(),
        1,
        "the reversing pair must collapse to one semantic atlas atom"
    );
    assert_eq!(
        result.term.chart_atlases().len(),
        1,
        "exactly one atlas registered"
    );
    let atlas = &result.term.chart_atlases()[0];
    assert_eq!(atlas.charts(), &[0, 1], "the atlas covers both charts");
    assert_eq!(
        atlas.transitions().len(),
        1,
        "the pair registers exactly one directed seam"
    );
    assert_eq!(
        atlas.transitions()[0].sign,
        -1,
        "the registered transition must be orientation-reversing"
    );

    // A single reversing seam is REMOVABLE by flipping one chart's local
    // orientation, so the two-chart cover is still ORIENTABLE — non-orientability
    // is a property of the sign COCYCLE (a negative-holonomy cycle), not of one
    // negative edge. The dedicated Möbius pin below exercises that path.
    assert_eq!(
        atlas.orientability(),
        AtlasOrientability::Orientable,
        "one reversing edge is a gauge choice; the cover stays orientable"
    );

    // Registration is an image-EXACT algebraic quotient of the same gates: the
    // reconstruction is untouched, and the chart gates factor exactly into
    // activation × a normalized partition of unity on every row.
    assert_eq!(
        result.term.try_fitted().unwrap(),
        fitted_before,
        "atlas registration must not change the decoded image"
    );
    let assignments = result.term.assignment.assignments();
    for row in 0..n {
        let (activation, partition) = result
            .term
            .atlas_partition_of_unity(0, assignments.row(row))
            .unwrap();
        assert!(
            (partition.sum() - 1.0).abs() < 8.0 * f64::EPSILON,
            "partition of unity must be normalized on row {row}"
        );
        for (slot, &chart) in atlas.charts().iter().enumerate() {
            assert!(
                (activation * partition[slot] - assignments[[row, chart]]).abs()
                    < 8.0 * f64::EPSILON,
                "activation × partition must reproduce the chart gate on row {row}"
            );
        }
    }
}

/// Build a TWO-sphere-chart term over `n` rows that cover ONE ambient unit
/// sphere (in ambient dims `0,1,2`) with mutually-interior poles. Chart A uses
/// the identity lat/lon frame; chart B is the SAME sphere reparametrized with its
/// pole along A's `x`-axis (a 90° ambient rotation), so A's pole is a regular
/// interior point of B and vice versa — the defining sphere pole seam. Rows
/// `0..n/2` are chart A's disjoint support, `n/2..n` chart B's.
///
/// Each physical point `q = [q0, q1, q2]` on the sphere is parametrized in BOTH
/// charts (`A`: `lat = asin q2, lon = atan2(q1, q0)`; `B`: `u_b = [-q2, q1, q0]`),
/// and BOTH decoders map their unit vector back to the SAME ambient `q` — so the
/// two charts are an exact over-tiling whose transition is the ambient rotation
/// `R = [[0,0,1],[0,1,0],[-1,0,0]]` (`det R = +1`, a proper rotation).
fn build_sphere_pair_term(n: usize) -> (SaeManifoldTerm, Array2<f64>) {
    assert!(n % 2 == 0 && n >= 8);
    let p = 4usize;
    let evaluator = Arc::new(SphereChartEvaluator);
    let half = n / 2;

    // Physical points on the unit sphere (ambient dims 0,1,2). Group A near A's
    // equator (lat = ±0.3 straddling 0); group B near B's equator (lat_B = ±0.3).
    let mut q = Array2::<f64>::zeros((n, p));
    for j in 0..half {
        let lon = -std::f64::consts::PI + std::f64::consts::TAU * (j as f64 + 0.5) / half as f64;
        let lat: f64 = if j % 2 == 0 { 0.3 } else { -0.3 };
        // A's frame: ambient = [x, y, z].
        q[[j, 0]] = lat.cos() * lon.cos();
        q[[j, 1]] = lat.cos() * lon.sin();
        q[[j, 2]] = lat.sin();
    }
    for j in 0..half {
        let row = half + j;
        let lon_b = -std::f64::consts::PI + std::f64::consts::TAU * (j as f64 + 0.5) / half as f64;
        let lat_b: f64 = if j % 2 == 0 { 0.3 } else { -0.3 };
        let xb = lat_b.cos() * lon_b.cos();
        let yb = lat_b.cos() * lon_b.sin();
        let zb = lat_b.sin();
        // B's embedding: ambient = [zb, yb, -xb].
        q[[row, 0]] = zb;
        q[[row, 1]] = yb;
        q[[row, 2]] = -xb;
    }

    // Per-atom (lat, lon) coordinates of EVERY physical point in each chart.
    let mut coords_a = Array2::<f64>::zeros((n, 2));
    let mut coords_b = Array2::<f64>::zeros((n, 2));
    for r in 0..n {
        let (q0, q1, q2) = (q[[r, 0]], q[[r, 1]], q[[r, 2]]);
        coords_a[[r, 0]] = q2.clamp(-1.0, 1.0).asin();
        coords_a[[r, 1]] = q1.atan2(q0);
        // B-unit-vector of the same physical point: u_b = [-q2, q1, q0].
        let (xb, yb, zb) = (-q2, q1, q0);
        coords_b[[r, 0]] = zb.clamp(-1.0, 1.0).asin();
        coords_b[[r, 1]] = yb.atan2(xb);
    }

    let (phi_a, jet_a) = evaluator.evaluate(coords_a.view()).unwrap();
    let (phi_b, jet_b) = evaluator.evaluate(coords_b.view()).unwrap();

    // Pure linear sphere embeddings (quadratic rows zero). basis = [1,x,y,z,xy,yz,xz].
    let mut decoder_a = Array2::<f64>::zeros((7, p));
    decoder_a[[1, 0]] = 1.0; // x -> dim0
    decoder_a[[2, 1]] = 1.0; // y -> dim1
    decoder_a[[3, 2]] = 1.0; // z -> dim2
    let mut decoder_b = Array2::<f64>::zeros((7, p));
    decoder_b[[3, 0]] = 1.0; // z_b -> dim0
    decoder_b[[2, 1]] = 1.0; // y_b -> dim1
    decoder_b[[1, 2]] = -1.0; // x_b -> dim2 (negated)

    let mut penalty = Array2::<f64>::eye(7);
    penalty *= 1.0e-4;
    let atom_a = SaeManifoldAtom::new(
        "sphere_a",
        SaeAtomBasisKind::Sphere,
        2,
        phi_a,
        jet_a,
        decoder_a,
        penalty.clone(),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let atom_b = SaeManifoldAtom::new(
        "sphere_b",
        SaeAtomBasisKind::Sphere,
        2,
        phi_b,
        jet_b,
        decoder_b,
        penalty,
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());

    // Disjoint supports: A owns the first half, B the second.
    let mut logits = Array2::<f64>::zeros((n, 2));
    for r in 0..n {
        if r < half {
            logits[[r, 0]] = ON;
            logits[[r, 1]] = OFF;
        } else {
            logits[[r, 0]] = OFF;
            logits[[r, 1]] = ON;
        }
    }
    let sphere_manifold = || {
        LatentManifold::Product(vec![
            LatentManifold::Interval {
                lo: -std::f64::consts::FRAC_PI_2,
                hi: std::f64::consts::FRAC_PI_2,
            },
            LatentManifold::Circle {
                period: std::f64::consts::TAU,
            },
        ])
    };
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords_a, coords_b],
        vec![sphere_manifold(), sphere_manifold()],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom_a, atom_b], assignment).unwrap();
    (term, q)
}

fn sphere_rho() -> SaeManifoldRho {
    // Per-axis ARD: one log-precision per sphere axis (dim = 2).
    SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(2); 2])
}

#[test]
fn sphere_pole_pair_registers_atlas_end_to_end() {
    // Increment 2 — the d=2 POLE-seam register emitter. Two sphere charts cover
    // ONE ambient sphere with mutually-interior poles: the transition is an
    // ambient rotation `R ∈ SO(3)`, not a 1-D affine map, so neither the fusion
    // lane nor the 1-D glue lane can see it. The sphere pole lane must fit the
    // rotation, classify the overlap as a POLE seam, certify equivalence, and
    // REGISTER (keep both charts as one partition-of-unity atlas atom) end-to-end
    // through the production driver.
    let n = 48;
    let (term, q) = build_sphere_pair_term(n);
    let p = q.ncols();
    let rho = sphere_rho();

    // Target: the shared sphere image plus a small isometry-band residual.
    let mut target = term.try_fitted().unwrap();
    for r in 0..n {
        for c in 0..p {
            target[[r, c]] += 0.01 * ((r * 7 + c * 3) as f64).sin();
        }
    }

    assert_eq!(
        active_atom_count(&term),
        2,
        "fixture starts with 2 sphere charts"
    );
    assert_eq!(term.semantic_atom_count(), 2, "no atlas registered yet");

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
    eprintln!(
        "[1890-e2e-pole] accepted_glues={glues} k_atoms={} semantic_atoms={} atlases={}",
        result.term.k_atoms(),
        result.term.semantic_atom_count(),
        result.term.chart_atlases().len(),
    );

    // bank -> certify -> apply fired on the pole seam.
    assert!(
        glues >= 1,
        "no Glue certified for the sphere pole pair; the pole-seam register lane \
         did not fire end-to-end"
    );
    assert!(
        result.structure_changed(),
        "a glue applied but structure_changed() is false"
    );

    // REGISTER, not FUSE: both sphere charts survive and keep routing mass — a
    // pole seam has no destructive fuse outcome (neither lat/lon chart alone
    // covers both poles).
    assert_eq!(
        result.term.k_atoms(),
        2,
        "sphere pole registration must keep BOTH local charts"
    );
    assert_eq!(
        active_atom_count(&result.term),
        2,
        "both sphere charts stay active under the partition of unity"
    );

    // The two charts quotient into ONE semantic atom carrying the pole seam.
    assert_eq!(
        result.term.semantic_atom_count(),
        1,
        "the sphere pole pair must collapse to one semantic atlas atom"
    );
    assert_eq!(result.term.chart_atlases().len(), 1, "exactly one atlas");
    let atlas = &result.term.chart_atlases()[0];
    assert_eq!(
        atlas.charts(),
        &[0, 1],
        "the atlas covers both sphere charts"
    );
    // The pole seam is stored as a 2-D ambient-rotation transition, NOT a 1-D
    // affine one — asserting a 1-D map here would misdescribe the overlap.
    assert!(
        atlas.transitions().is_empty(),
        "a sphere pole seam is not a 1-D unit-speed transition"
    );
    assert_eq!(
        atlas.sphere_transitions().len(),
        1,
        "the pole pair registers exactly one ambient-rotation seam"
    );
    let sphere_seam = &atlas.sphere_transitions()[0];
    assert_eq!(
        sphere_seam.seam_kind,
        AtlasSeamKind::Pole,
        "classified as a pole seam"
    );
    assert_eq!(
        sphere_seam.sign(),
        1,
        "a sphere is orientable: its proper-rotation transition carries sign +1"
    );
    assert!(
        (sphere_seam.determinant() - 1.0).abs() < 1e-9,
        "the fitted transition must be a proper rotation (det = +1), got {}",
        sphere_seam.determinant()
    );

    // A single orientable sphere cover: the sign cocycle is trivially +1.
    assert_eq!(
        atlas.orientability(),
        AtlasOrientability::Orientable,
        "a sphere pole cover is orientable"
    );

    // Registration is an image-EXACT algebraic quotient of the same gates: it
    // rewrites `sum_c a_c gamma_c = (sum a_c) * sum_c [a_c/sum] gamma_c` without
    // touching any decoded value.  That exactness is pinned directly below by the
    // partition-of-unity factorization reproducing every chart gate to 8 ULP —
    // which holds for WHATEVER logits the terminal state carries.  (The absolute
    // reconstruction quality of the subsequent joint POLISH refit is a property of
    // the shared REML fit driver, not of this pole-seam register lane, and is
    // exercised by the fit-driver suites; this pin asserts the registration
    // outcome, not the refit's convergence.)
    let assignments = result.term.assignment.assignments();
    for row in 0..n {
        let (activation, partition) = result
            .term
            .atlas_partition_of_unity(0, assignments.row(row))
            .unwrap();
        assert!(
            (partition.sum() - 1.0).abs() < 8.0 * f64::EPSILON,
            "partition of unity must be normalized on row {row}"
        );
        for (slot, &chart) in atlas.charts().iter().enumerate() {
            assert!(
                (activation * partition[slot] - assignments[[row, chart]]).abs()
                    < 8.0 * f64::EPSILON,
                "activation × partition must reproduce the chart gate on row {row}"
            );
        }
    }
}

#[test]
fn mobius_cocycle_reports_non_orientable_via_sign_holonomy() {
    // A genuine Möbius obstruction (as opposed to a single reflected chart) needs
    // an overlap whose transition FLIPS sign around a cycle: going once around the
    // cover reverses the frame with no local re-choice able to undo it. We build
    // that from a two-chart term whose two overlap COMPONENTS carry opposite signs
    // — the defining half-twist — and register both through the SAME production
    // entry point (`register_chart_transition`). The sign cocycle then has
    // holonomy `-1`, which the orientability readout must report as Möbius.
    let n = 20;
    let p = 4;
    let arcs = [(0usize, 10usize), (10usize, 20usize)];
    let decoders = [circle_decoder(p, 0, 1), reflected_circle_decoder(p, 0, 1)];
    let mut term = build_term(n, &arcs, &decoders);

    let period = 1.0;
    // Overlap component 1: orientation-PRESERVING (sign +1).
    term.register_chart_transition(
        UnitSpeedChartTransition::new(0, 1, 1, 0.0, period, AtlasSeamKind::Regular).unwrap(),
    )
    .unwrap();
    // Overlap component 2 on the SAME pair: orientation-REVERSING (sign -1). The
    // cycle 0 →(+1) 1 →(-1) 0 has sign product -1 — the Möbius holonomy.
    term.register_chart_transition(
        UnitSpeedChartTransition::new(1, 0, -1, 0.5, period, AtlasSeamKind::Regular).unwrap(),
    )
    .unwrap();

    assert_eq!(
        term.chart_atlases().len(),
        1,
        "both seams live in one atlas"
    );
    let atlas = &term.chart_atlases()[0];
    assert_eq!(atlas.charts(), &[0, 1]);
    assert_eq!(
        atlas.transitions().len(),
        2,
        "two overlap components registered"
    );
    assert_eq!(
        atlas.orientability(),
        AtlasOrientability::NonOrientable,
        "opposite-sign overlap components give the atlas Möbius holonomy"
    );

    // Both charts remain one semantic atom under a normalized partition of unity —
    // the Möbius cover is representable ONLY as this multi-chart atlas atom.
    assert_eq!(term.k_atoms(), 2);
    assert_eq!(term.semantic_atom_count(), 1);
    let assignments = term.assignment.assignments();
    for row in 0..n {
        let (activation, partition) = term
            .atlas_partition_of_unity(0, assignments.row(row))
            .unwrap();
        assert!((partition.sum() - 1.0).abs() < 8.0 * f64::EPSILON);
        for (slot, &chart) in atlas.charts().iter().enumerate() {
            assert!(
                (activation * partition[slot] - assignments[[row, chart]]).abs()
                    < 8.0 * f64::EPSILON
            );
        }
    }
}
