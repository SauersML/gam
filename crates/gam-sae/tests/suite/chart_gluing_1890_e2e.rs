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
use gam_sae::basis::{
    AmbientSphereHarmonicEvaluator, PeriodicHarmonicEvaluator, SaeBasisEvaluator,
};
use gam_sae::manifold::{
    AtlasOrientability, AtlasSeamKind, SAE_AMBIENT_SPHERE_DEFAULT_DEGREE, SaeAtomBasisKind,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, UnitSpeedChartTransition,
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

/// Coordinate width of an ambient sphere atom: `S²` carries THREE ambient
/// coordinates for its two intrinsic dimensions, which is exactly what buys it a
/// global chart. This is the `latent_dim` `SaeAtomGeometryPlan::new` accepts for
/// `(Sphere, AmbientSphereHarmonics, RoundSphere)` and refuses anything else at.
const AMBIENT_SPHERE_LATENT_DIM: usize = 3;

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
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
/// off. The driver continues until its no-move fixpoint.
fn glue_only_config() -> RoundDriverConfig {
    RoundDriverConfig {
        n_shards: 4,
        budget: MoveBudget {
            max_moves: 4,
            alpha: 0.05,
        },
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

/// The ambient rotation carrying B's unit vector into A's (`u_a = R u_b`) for
/// the POLE fixture: B is the same sphere framed with its axis along A's
/// `x`-axis (a 90° rotation about `y`), so B's axis sits on A's equator and A's
/// on B's — each atom's frame axis is interior to the OTHER's active band, which
/// is the defining pole seam. `det R = +1`, a proper rotation.
const POLE_FRAME_ROTATION: [[f64; 3]; 3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];

/// The identity ambient frame — atom A's frame, and, used as B's frame too, the
/// NEGATIVE control: both atoms then carry the SAME frame, so each atom's axis
/// lands at the other's pole (`lat = π/2`), far OUTSIDE the other's active band.
/// That overlap is regular, not a pole seam, and must not certify.
const IDENTITY_FRAME: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Build a TWO-sphere-atom term over `n` rows that cover ONE ambient unit sphere
/// (in ambient dims `0,1,2`) through two frames related by `rotation`
/// (`u_a = R u_b`). Atom A uses the identity ambient frame; atom B's frame is
/// `R`. Rows `0..n/2` are A's disjoint support, `n/2..n` B's, and each group sits
/// in a narrow band around ITS OWN frame's equator — so the two active bands, and
/// therefore whether each frame axis is interior to the other, are decided
/// entirely by `R`.
///
/// Each physical point `q` on the sphere is parametrized in BOTH frames by its
/// own UNIT VECTOR (`A`: `u_a = q`; `B`: `u_b = Rᵀ q`), and BOTH decoders map
/// that unit vector back to the SAME ambient `q` — so the two atoms are an exact
/// over-tiling whose transition is `R`.
///
/// #2698 — every declaration below is the AMBIENT form, because that is the only
/// sphere the engine can build: `SaeAtomGeometryPlan::new` accepts
/// `(Sphere, latent_dim = 3, AmbientSphereHarmonics, RoundSphere)` and refuses
/// `(Sphere, 2, ..)`, and `SphereChartEvaluator` — the `(lat, lon)` chart this
/// fixture used to declare while already building ambient data — no longer
/// exists. So `SaeAtomBasisKind::Sphere` is declared at `latent_dim = 3`, the
/// latent manifold is `LatentManifold::Sphere { dim: 3 }`, and the ARD block is
/// 3 wide, all matching the ambient unit-vector coordinates and the
/// `AmbientSphereHarmonicEvaluator` this builder actually uses. The seam under
/// test is unchanged by that: the transition between two ambient frames is still
/// a rotation in `SO(3)`, which is exactly what `SphereChartTransition` stores.
fn build_sphere_pair_term(n: usize, rotation: [[f64; 3]; 3]) -> (SaeManifoldTerm, Array2<f64>) {
    assert!(n % 2 == 0 && n >= 8);
    let p = 4usize;
    let evaluator =
        Arc::new(AmbientSphereHarmonicEvaluator::new(SAE_AMBIENT_SPHERE_DEFAULT_DEGREE).unwrap());
    let half = n / 2;
    let apply = |m: [[f64; 3]; 3], u: [f64; 3]| -> [f64; 3] {
        [
            m[0][0] * u[0] + m[0][1] * u[1] + m[0][2] * u[2],
            m[1][0] * u[0] + m[1][1] * u[1] + m[1][2] * u[2],
            m[2][0] * u[0] + m[2][1] * u[1] + m[2][2] * u[2],
        ]
    };
    let mut inverse = [[0.0_f64; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            inverse[row][column] = rotation[column][row];
        }
    }
    // A band around a frame's own equator: latitudes straddle zero, longitudes
    // sweep the full circle, so the band is a genuine annulus and its latitude
    // span excludes the frame's own axis.
    let band_unit = |j: usize| -> [f64; 3] {
        let lon = -std::f64::consts::PI + std::f64::consts::TAU * (j as f64 + 0.5) / half as f64;
        let lat: f64 = if j % 2 == 0 { 0.3 } else { -0.3 };
        [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
    };

    // Physical points on the unit sphere (ambient dims 0,1,2). Group A is a band
    // around A's equator, group B a band around B's — the latter written in B's
    // frame and carried into the ambient by `R`.
    let mut q = Array2::<f64>::zeros((n, p));
    for j in 0..half {
        let u = band_unit(j);
        for axis in 0..3 {
            q[[j, axis]] = u[axis];
        }
    }
    for j in 0..half {
        let u = apply(rotation, band_unit(j));
        for axis in 0..3 {
            q[[half + j, axis]] = u[axis];
        }
    }

    // Per-atom AMBIENT coordinates of every physical point in each frame. The
    // asin/atan2 round-trip the chart needed is gone: each atom's coordinate is
    // the unit vector itself, in that atom's own frame.
    let mut coords_a = Array2::<f64>::zeros((n, 3));
    let mut coords_b = Array2::<f64>::zeros((n, 3));
    for r in 0..n {
        let point = [q[[r, 0]], q[[r, 1]], q[[r, 2]]];
        let u_b = apply(inverse, point);
        for axis in 0..3 {
            coords_a[[r, axis]] = point[axis];
            coords_b[[r, axis]] = u_b[axis];
        }
    }

    let (phi_a, jet_a) = evaluator.evaluate(coords_a.view()).unwrap();
    let (phi_b, jet_b) = evaluator.evaluate(coords_b.view()).unwrap();

    // Pure linear sphere embeddings: only the degree-1 (dipole) block is used,
    // every higher harmonic row left at zero. The columns are located by their
    // (degree, order) rather than by a hardcoded layout, so this cannot silently
    // decode the wrong harmonic if the column order ever changes. Column `(1, m)`
    // is `N_{1,m}` times ONE ambient coordinate (`m = +1 -> Re[w] = x`,
    // `m = -1 -> Im[w] = y`, `m = 0 -> z`), and each `N_{1,m}` is read back OUT
    // of the evaluator as `Φ(e_axis)` on that column rather than restated as a
    // literal here — so a decoder row of `frame[out][axis] / N_axis` decodes the
    // unit vector to exactly `frame · u`.
    let modes = evaluator.spectral_modes();
    let dipole = |order: i64| -> usize {
        modes
            .iter()
            .position(|mode| mode.degree == 1 && mode.order == order)
            .expect("the degree-1 block is present at every degree >= 1")
    };
    let dipole_column = [dipole(1), dipole(-1), dipole(0)];
    let width = modes.len();
    let (phi_axes, _) = evaluator.evaluate(Array2::<f64>::eye(3).view()).unwrap();
    let dipole_norm: Vec<f64> = (0..3)
        .map(|axis| phi_axes[[axis, dipole_column[axis]]])
        .collect();

    // A decoder whose degree-1 block realizes the linear map `frame`: the atom
    // decodes its unit vector `u` to the ambient point `frame · u`.
    let linear_decoder = |frame: [[f64; 3]; 3]| -> Array2<f64> {
        let mut decoder = Array2::<f64>::zeros((width, p));
        for axis in 0..3 {
            for output in 0..3 {
                decoder[[dipole_column[axis], output]] = frame[output][axis] / dipole_norm[axis];
            }
        }
        decoder
    };
    // A's frame is the identity embedding; B's is `R`, so both decode the SAME
    // physical point from their own unit vector.
    let decoder_a = linear_decoder(IDENTITY_FRAME);
    let decoder_b = linear_decoder(rotation);

    let mut penalty = Array2::<f64>::eye(width);
    penalty *= 1.0e-4;
    let atom_a = SaeManifoldAtom::new_with_provided_function_gram(
        "sphere_a",
        SaeAtomBasisKind::Sphere,
        AMBIENT_SPHERE_LATENT_DIM,
        phi_a,
        jet_a,
        decoder_a,
        penalty.clone(),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let atom_b = SaeManifoldAtom::new_with_provided_function_gram(
        "sphere_b",
        SaeAtomBasisKind::Sphere,
        AMBIENT_SPHERE_LATENT_DIM,
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
    // The coordinate is the ambient unit vector, so the manifold that owns its
    // retraction and its tangent projection is the embedded sphere itself — not
    // the `Interval × Circle` product a `(lat, lon)` chart would need.
    let sphere_manifold = || LatentManifold::Sphere {
        dim: AMBIENT_SPHERE_LATENT_DIM,
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
    // Per-axis ARD: one log-precision per AMBIENT sphere axis, so the block is as
    // wide as the atom's coordinate (#2698).
    SaeManifoldRho::new(
        0.0,
        -4.0,
        vec![Array1::<f64>::zeros(AMBIENT_SPHERE_LATENT_DIM); 2],
    )
}

