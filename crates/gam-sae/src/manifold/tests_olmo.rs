//! Real-OLMo-activation manifold tests.
//!
//! The only declaration of this module (`crate::manifold`) is
//! `#[cfg(test)] mod tests_olmo;`, so the inner attribute below is a no-op — it
//! just makes the test-only scope a claim the compiler checks in this file
//! rather than one carried by the filename.
#![cfg(test)]

use gam_linalg::faer_ndarray::fast_ata;

use super::*;
use ndarray::array;

/// Build a production-style K-atom, d=2 periodic (torus = Circle×Circle) SAE
/// manifold term seeded from REAL activations `z` exactly the way the
/// production cold path does: PCA-seed the per-atom chart, fit a per-atom
/// decoder by ridge LSQ on the gated basis, install the analytic torus
/// evaluator, and assemble the multi-atom assignment with the curved product
/// manifold on every atom. This is the d>=2 atom regime the #1019 canonical
/// charts gauge and the #1007 curvature anchor have to identify on real data.
pub(crate) fn real_data_torus_seed_term(
    z: ArrayView2<'_, f64>,
    k: usize,
    num_harmonics: usize,
) -> SaeManifoldTerm {
    let n = z.nrows();
    let evaluator = Arc::new(TorusHarmonicEvaluator::new(2, num_harmonics).unwrap());
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let atom_dims = vec![2usize; k];
    let seed_coords = sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims).unwrap();
    let mut atoms = Vec::with_capacity(k);
    let mut coords_blocks = Vec::with_capacity(k);
    let mut manifolds = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let coords = seed_coords.slice(s![atom_idx, .., 0..2]).to_owned();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        // Per-atom decoder by ridge LSQ on the gated basis (gate = 1 at seed).
        let mut xtx = fast_ata(&phi);
        for i in 0..m {
            xtx[[i, i]] += 1.0e-8;
        }
        let xtz = fast_atb(&phi, &z.to_owned());
        let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz);
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "torus",
            SaeAtomBasisKind::Periodic,
            2,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(evaluator.clone());
        atoms.push(atom);
        coords_blocks.push(coords);
        manifolds.push(LatentManifold::Product(vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ]));
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n, k), 0.0),
        coords_blocks,
        manifolds,
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// #1190 — REAL-data curvature-anchor positive-definiteness.
///
/// On genuine OLMo-3-32B residual-stream activations the manifold-SAE
/// curvature anchor (the undamped evidence Hessian assembled at the #1007
/// homotopy `η = 1` basis) must be positive-definite on the gauge quotient so
/// the d=2 atoms are IDENTIFIED. The pre-fix failure mode: on the long-tailed
/// real spectrum the undamped per-row `H_tt` blocks carry a near-null /
/// negative direction that is NOT a closed-form chart-gauge direction, so the
/// smallest undamped pivot collapses below the safe-SPD floor and the atoms
/// are under-identified. This test pins the anchor PD-ness on the committed
/// real fixture.
#[test]
pub(crate) fn olmo_real_curvature_anchor_is_positive_definite() {
    let path = olmo_fixture_path("olmo_mixedlayer_pca64_768.npy");
    let z = read_npy_f32_2d(&path);
    assert_eq!(z.dim(), (768, 64), "real OLMo fixture shape");
    // Small REAL slice (K=2 d=2 torus, 160 rows) so the per-row curvature-anchor
    // assembly + eigendecomposition completes in seconds. The PD property under
    // test is a per-row block property of the genuine assembled evidence Hessian,
    // so a representative real-data slice exercises it without the full-N inner
    // joint Newton fit (which is the slow path; we don't need a fit to read the
    // raw anchor). #1190.
    let z_train = z.slice(s![..160, ..]).to_owned();
    let k = 2usize;

    let mut term = real_data_torus_seed_term(z_train.view(), k, 3);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0, 0.0]; k]);
    let registry = SaeManifoldOuterObjective::new(
        term.clone(),
        z_train.clone(),
        None,
        rho.clone(),
        0,
        0.04,
        1.0e-6,
        1.0e-6,
    )
    .registry;

    // GENUINE curvature anchor = the RAW assembled per-row evidence Hessian
    // blocks BEFORE factorization/deflation, evaluated at the real-data PCA seed.
    // This is what actually pins the atoms; if a block is genuinely indefinite (a
    // negative eigenvalue OFF the closed-form gauge orbit), the spectral deflation
    // would silently flatten that direction to unit stiffness — the factor stays
    // PD but the atom coordinate along it is UNIDENTIFIED. Reading the raw anchor
    // needs only ONE assembly (no inner fit), so it is fast and deterministic.
    // The #1190 fix makes the softmax curvature block the PSD Fisher metric, so
    // every per-row block is PD up to round-off on this real slice.
    use gam_linalg::faer_ndarray::FaerEigh;
    let sys = term
        .assemble_arrow_schur(z_train.view(), &rho, registry.as_ref())
        .expect("assemble raw curvature anchor");
    let mut min_raw_eig = f64::INFINITY;
    let mut max_raw_eig = 0.0_f64;
    let mut indefinite_rows = 0usize;
    let mut total_neg_dirs = 0usize;
    for block in &sys.rows {
        let d = block.htt.nrows();
        if d == 0 {
            continue;
        }
        let mut sym = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                sym[[i, j]] = 0.5 * (block.htt[[i, j]] + block.htt[[j, i]]);
            }
        }
        let (evals, _) = sym.eigh(faer::Side::Lower).unwrap();
        let max_abs = evals.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let neg_floor = -1.0e-8 * max_abs;
        let row_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        let row_neg = evals.iter().filter(|&&v| v < neg_floor).count();
        min_raw_eig = min_raw_eig.min(row_min);
        max_raw_eig = max_raw_eig.max(max_abs);
        if row_neg > 0 {
            indefinite_rows += 1;
            total_neg_dirs += row_neg;
        }
    }
    let rel_min = min_raw_eig / max_raw_eig.max(1.0);
    eprintln!(
        "[#1190] real-data curvature anchor (K={k}, N={}): RAW assembled H_tt \
         min_eig={min_raw_eig:.6e} (rel={rel_min:.3e}) indefinite_rows={indefinite_rows}/{} \
         total_neg_dirs={total_neg_dirs}",
        z_train.nrows(),
        sys.rows.len()
    );

    // The curvature anchor is IDENTIFIED iff the genuine assembled per-row
    // evidence Hessian is positive-semidefinite up to a relative floor on EVERY
    // row: no row may carry a data-supported negative-curvature direction that
    // the deflation would have to flatten (which would leave that atom
    // coordinate unpinned). A relative floor of -1e-8 admits only round-off
    // negatives; a genuine indefinite block sits orders of magnitude below it.
    assert!(
        rel_min >= -1.0e-8,
        "real-data curvature anchor is genuinely indefinite: raw assembled H_tt \
         min eigenvalue {min_raw_eig:.6e} (relative {rel_min:.3e}) is negative on \
         {indefinite_rows}/{} rows ({total_neg_dirs} negative directions) — the \
         d=2 atoms are under-identified on real OLMo activations (#1190). The \
         curvature anchor must be PD (or its negative directions must be genuine \
         closed-form gauge nulls, not data-supported directions).",
        sys.rows.len()
    );
}

/// Resolve a committed OLMo `.npy` fixture by file name.
///
/// The fixtures are tracked at the WORKSPACE root (`tests/data/<name>`), but
/// `CARGO_MANIFEST_DIR` for this crate is `crates/gam-sae`, so the naive
/// `MANIFEST_DIR/tests/data` join misses them. Some call sites historically
/// open-coded a `../../tests/data` fallback and some forgot it (the latter then
/// panicked with a bare ENOENT — `olmo_real_curvature_anchor_is_positive_definite`
/// and `olmo_real_arrival_floor_tracks_data_ceiling`). Route every
/// fixture lookup through this resolver so the two layouts are tried in ONE
/// place: the per-crate path first (for a crate-local checkout), then the
/// workspace-root path. A clear panic names both probed paths if neither exists.
pub(crate) fn olmo_fixture_path(name: &str) -> std::path::PathBuf {
    let mani = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let crate_local = mani.join("tests/data").join(name);
    if crate_local.exists() {
        return crate_local;
    }
    let workspace_root = mani.join("../../tests/data").join(name);
    if workspace_root.exists() {
        return workspace_root;
    }
    panic!(
        "OLMo fixture {name} not found at {} or {}",
        crate_local.display(),
        workspace_root.display()
    );
}

/// Read a 2-D float32 (`<f4`) C-contiguous `.npy` into an `Array2<f64>`.
/// The committed OLMo activation fixtures are float32; the production smooth
/// loader only parses `<f8`, so this test-local reader covers the `<f4` case
/// for the real-data curvature-anchor probe.
pub(crate) fn read_npy_f32_2d(path: &std::path::Path) -> Array2<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len() > 10 && &bytes[0..6] == b"\x93NUMPY",
        "not a .npy"
    );
    let major = bytes[6];
    let (hdr_start, hdr_len) = if major == 1 {
        (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize)
    } else {
        (
            12usize,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        )
    };
    let data_off = hdr_start + hdr_len;
    let header = std::str::from_utf8(&bytes[hdr_start..data_off]).unwrap();
    assert!(
        header.contains("'<f4'") || header.contains("\"<f4\""),
        "fixture must be little-endian float32; header: {header}"
    );
    assert!(!header.contains("True"), "fixture must be C-contiguous");
    let open = header.find('(').unwrap();
    let close = header[open..].find(')').unwrap() + open;
    let dims: Vec<usize> = header[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().unwrap())
        .collect();
    assert_eq!(dims.len(), 2, "fixture must be 2-D");
    let (n, p) = (dims[0], dims[1]);
    let mut out = Array2::<f64>::zeros((n, p));
    let payload = &bytes[data_off..];
    assert!(payload.len() >= n * p * 4, "truncated payload");
    for r in 0..n {
        for c in 0..p {
            let i = (r * p + c) * 4;
            let v =
                f32::from_le_bytes([payload[i], payload[i + 1], payload[i + 2], payload[i + 3]]);
            out[[r, c]] = v as f64;
        }
    }
    out
}

/// #2498 — the final fitted-data verdict accepts no externally fabricated
/// reconstruction or assignments. It derives all evidence from the live term,
/// and records an event exactly when that same state's decoder disappears.
#[test]
pub(crate) fn fit_data_collapse_verdict_uses_one_self_consistent_state_s1() {
    let coords = array![[0.0_f64], [0.25], [0.5], [0.75]];
    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 3));
    let mut jet = Array3::<f64>::zeros((n, 3, 1));
    for row in 0..n {
        let angle = 2.0 * std::f64::consts::PI * coords[[row, 0]];
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = angle.sin();
        phi[[row, 2]] = angle.cos();
        jet[[row, 1, 0]] = 2.0 * std::f64::consts::PI * angle.cos();
        jet[[row, 2, 0]] = -2.0 * std::f64::consts::PI * angle.sin();
    }
    let mut live_decoder = Array2::<f64>::zeros((3, 2));
    live_decoder[[2, 0]] = 1.0;
    live_decoder[[1, 1]] = 1.0;
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        live_decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut live_term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    let target = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
    let rho = SaeManifoldRho::new(-0.3, 0.0, vec![array![0.0]]);

    let live_verdict = live_term
        .dictionary_collapse_verdict(target.view(), &rho, None)
        .expect("live verdict");
    assert!(!live_verdict.all_decoders_vanished(1));
    assert!(!live_verdict.degenerate(1));
    let recorded_live = live_term
        .record_fit_data_collapse_if_needed(target.view(), &rho, 3)
        .unwrap();
    assert!(
        !recorded_live,
        "a live decoder must not produce a terminal collapse event"
    );
    assert!(
        !live_term
            .collapse_events()
            .iter()
            .any(|e| e.action == CollapseAction::Terminal),
        "no terminal event may be recorded for a live decoder"
    );

    let mut vanished_term = live_term.clone();
    vanished_term.atoms[0].decoder_coefficients_mut().fill(0.0);
    let vanished_verdict = vanished_term
        .dictionary_collapse_verdict(target.view(), &rho, None)
        .expect("vanished verdict");
    assert!(vanished_verdict.all_decoders_vanished(1));
    assert!(vanished_verdict.degenerate(1));
    let recorded_vanished = vanished_term
        .record_fit_data_collapse_if_needed(target.view(), &rho, 7)
        .unwrap();
    assert!(
        recorded_vanished,
        "an exactly vanished decoder is a genuine #853/#976 co-collapse"
    );
    let terminal = vanished_term
        .collapse_events()
        .iter()
        .find(|e| e.action == CollapseAction::Terminal)
        .expect("a terminal collapse event must be recorded for the vanished dictionary");
    assert!(
        terminal.floor.is_finite() && terminal.floor >= 0.0,
        "the event carries the derived signal boundary, never a NaN sentinel"
    );
}

// ── Out-of-sample evaluation helpers (real OLMo l18) ────────────────────────
// All SAE reconstruction-QUALITY claims below are evaluated OUT-OF-SAMPLE: an
// atom is fit on a train split and its reconstruction is measured on a held-out
// test split via the fast encode→decode. In-sample EV is optimistic (capacity
// can memorise the train manifold); the held-out number is the honest one.

/// Run the production circle readout (`build_sae_minimal_seed` →
/// `build_sae_fit_seed` → `run_sae_manifold_fit`, single periodic atom, the
/// unbundled direct path) at a given `random_state` and return the converged
/// per-row circle coordinate `t_i ∈ [0,1)`. Mirrors the #2023 tier-0 primary path
/// test's `run_primary`, threading the seed everywhere it is consumed (minimal
/// seed jitter + seed-refine routing) so different seeds are genuinely different
/// starts.
fn production_circle_coords_at_seed(
    target: &Array2<f64>,
    random_state: u64,
) -> ndarray::Array1<f64> {
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitRequest, SaeFitSeedReport, SaeFitSeedRequest,
        SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed, build_sae_minimal_seed,
        run_sae_manifold_fit,
    };
    let assignment_kind = SaeFitAssignmentKind::Softmax;
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: target.view(),
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![1],
        assignment_kind,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let SaeMinimalSeedReport {
        geometry_plans,
        basis_values,
        basis_jacobian,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        refine_routing,
    } = minimal;
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: target.view(),
        geometry_plans: &geometry_plans,
        basis_values: basis_values.view(),
        basis_jacobian: basis_jacobian.view(),
        decoder_coefficients: decoder_coefficients.view(),
        smooth_penalties: smooth_penalties.view(),
        initial_logits: initial_logits.view(),
        initial_coords: initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 12,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: refine_routing,
        seed_refine_random_state: random_state,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    let SaeFitSeedReport {
        base_term,
        initial_rho,
        isometry_pin_active,
        metric_provenance,
    } = seed;
    let report = run_sae_manifold_fit(SaeFitRequest {
            reconstruction_optimism_folds: None,
        base_term,
        target: target.clone(),
        registry,
        initial_rho,
        max_iter: 12,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        alpha: 1.0,
        isometry_pin_active,
        metric_provenance,
        promote_from_residual: false,
        run_structure_search: false,
        run_outer_rho_search: false,
        // Unbundled direct path: seed -> single certified fit on the iid
        // likelihood (no structured-residual re-whitening), the deterministic
        // "exactly one fit" contract.
        structured_residual_passes: 0,
        cancel: None,
    })
    .expect("production circle fit")
    .manifold_or_error()
    .expect("planted circle must retain a manifold atom");
    let coords = report.term.assignment.coords[0].as_matrix();
    ndarray::Array1::from_iter((0..coords.nrows()).map(|i| coords[[i, 0]]))
}

/// #2260 — MEASURE cross-seed stability of the production circle readout on a
/// LARGER-model activation cloud (Qwen-3.5-9B layer-21, PCA-64).
///
/// #2260 reported the gradient (torch-lane) circle fit's calendar orderings
/// spanning 0.67–0.97 across seeds at EV≈0.98 on OLMo-2-7B — i.e. high
/// reconstruction did not certify a stable latent parameterization. That lane
/// (torch-Adam, random cold init) has since been deleted; the surviving
/// production path seeds the circle coordinate DETERMINISTICALLY as
/// `atan2(PC2, PC1)` off the data (`sae_pca_seed_initial_coords` takes no
/// `random_state`), so `random_state` only perturbs a 1e-3 logit jitter and the
/// seed-refine routing — not the coordinate itself. The empirical question this
/// pins: does that leave any residual cross-seed spread in the CONVERGED readout?
///
/// It runs the production fit at five seeds (42–46) on the closest in-tree
/// larger-model cloud and reports the O(2)-aligned cross-seed circular
/// concordance (the exact statistic #2260 asks to be reported alongside EV). This
/// is a MEASUREMENT: it prints the min/mean aligned score and asserts only that
/// the readout is well-posed and finite, so the close-vs-keep-open verdict for
/// #2260 is read off the printed numbers rather than baked into an a-priori
/// threshold.
#[test]
fn production_circle_readout_cross_seed_concordance_2260() {
    let path = olmo_fixture_path("qwen35_9b_actsL21_pca64_2000.npy");
    let full = read_npy_f32_2d(&path);
    // First 800 rows keep the fit bounded while preserving the 64-dim ambient
    // subspace that gives a circle room to wander out-of-plane (#2260's mechanism).
    let n = 800.min(full.nrows());
    let z = full.slice(s![..n, ..]).to_owned();
    let seeds = [42u64, 43, 44, 45, 46];
    let mut coord_rows = Array2::<f64>::zeros((seeds.len(), n));
    for (r, &seed) in seeds.iter().enumerate() {
        let coords = production_circle_coords_at_seed(&z, seed);
        assert_eq!(coords.len(), n, "seed {seed}: one coordinate per row");
        assert!(
            coords.iter().all(|v| v.is_finite()),
            "seed {seed}: converged circle coordinate must be finite"
        );
        coord_rows.row_mut(r).assign(&coords);
    }
    let report = crate::circular_concordance::circular_concordance(coord_rows.view(), 1.0)
        .expect("circular concordance over the five seed replicates");
    let min_aligned = report.minimum_aligned_score;
    let mean_aligned = report.mean_aligned_score;
    eprintln!(
        "[#2260] Qwen-9B L21 production circle readout, seeds 42-46 (N={n}): \
         cross-seed circular concordance min={min_aligned:?} mean={mean_aligned:?}  \
         (torch-lane reported 0.67-0.97 spread; 1.0 = seed-identical ordering)"
    );
    for pair in &report.pairs {
        eprintln!(
            "[#2260]   pair ({},{}) aligned={:?} reflected={:?}",
            pair.left, pair.right, pair.aligned_score, pair.reflected
        );
    }
    // Well-posedness only — every replicate must span a 2-D circle embedding so
    // the aligned score is meaningful (not a degenerate collapse), and the score
    // must be reported. The numeric verdict lives in the printed line above.
    assert!(
        report.coverage.iter().all(|c| c.well_posed),
        "every seed's circle embedding must be well-posed (2-D span) for the \
         concordance to be meaningful"
    );
    assert!(
        min_aligned.is_some() && mean_aligned.is_some(),
        "cross-seed aligned concordance must be computable across the five seeds"
    );
    // Hardened guard (#2260): the production circle readout is seed-STABLE. The
    // coordinate seed is a deterministic atan2(PC2,PC1) read (no random_state), so
    // the converged ordering must not wander across seeds — measured min aligned
    // concordance is 1.0 across all pairs (vs the deleted torch lane's 0.67-0.97).
    // 0.99 leaves a wide margin over that 1.0 while still tripping hard if a
    // regression reintroduces seed-dependent basin selection into the readout.
    let min_aligned = min_aligned.expect("min aligned concordance");
    assert!(
        min_aligned >= 0.99,
        "production circle readout must be seed-stable: min cross-seed aligned \
         concordance {min_aligned:.4} must be >= 0.99 (deterministic atan2 seed); \
         a lower value means seed-dependent basin selection has regressed (#2260)"
    );
}

