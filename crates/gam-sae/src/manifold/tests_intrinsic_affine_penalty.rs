//! Split out of the oversized `tests` module (build.sh 10k-line hygiene gate,
//! sibling of #2130): the intrinsic/affine smooth-penalty and factored
//! β-penalty test cluster. These tests are self-contained — they define their
//! own fixtures (`intrinsic_test_atom`, `affine_canonicalization_test_term`)
//! and otherwise reuse the shared helpers that remain in the parent `tests`
//! module.
use super::*;
use super::tests::*;
use approx::assert_abs_diff_eq;
use gam_solve::arrow_schur::{
    ArrowFactorSlab, ArrowHtbetaCache, ArrowSolverMode, ArrowUndampedFactors, PcgDiagnostics,
};
use gam_terms::analytic_penalties::ARDPenalty;
use ndarray::{Array5, array};

pub(crate) fn intrinsic_test_atom(jacobian_scale: f64) -> SaeManifoldAtom {
    let m = 5usize;
    let n = m;
    let p = 1usize;
    let mut phi = Array2::<f64>::zeros((n, m));
    let mut jet = Array3::<f64>::zeros((n, m, 1));
    let mut decoder = Array2::<f64>::zeros((m, p));
    for mu in 0..m {
        // Localized basis: Φ_μ(t_n) ≈ δ_{nμ}.
        phi[[mu, mu]] = 1.0;
        // Per-sample basis derivative (axis 0) grows with μ — a
        // non-constant-speed curve — scaled by `jacobian_scale` to emulate
        // a global linear reparameterization t -> t / jacobian_scale.
        jet[[mu, mu, 0]] = jacobian_scale * (1.0 + mu as f64);
        decoder[[mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    SaeManifoldAtom::new(
        "intrinsic-1d",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap()
}

/// The roughness operator order is recovered from the raw Gram's null
/// space: an order-2 difference penalty annihilates the affine functions,
/// so `nullity = 2` and the arc-length exponent is `β = ½ − 2 = −3/2`.
#[test]
pub(crate) fn intrinsic_penalty_recovers_order_two_from_nullity() {
    let atom = intrinsic_test_atom(1.0);
    assert_eq!(atom.smooth_penalty_order, 2);
}

#[test]
pub(crate) fn line_search_snapshot_restores_intrinsic_smooth_penalty() {
    let atom = intrinsic_test_atom(1.0);
    let n = atom.n_obs();
    let logits = Array2::<f64>::zeros((n, 1));
    let coords = vec![Array2::<f64>::zeros((n, 1))];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let original = term.atoms[0].smooth_penalty.clone();
    let snapshot = term.snapshot_mutable_state();

    term.atoms[0].decoder_coefficients[[0, 0]] *= 3.0;
    term.atoms[0].refresh_intrinsic_smooth_penalty();
    let changed = (&term.atoms[0].smooth_penalty - &original)
        .mapv(f64::abs)
        .sum();
    assert!(
        changed > 1e-6,
        "test setup must perturb the live intrinsic smoothness Gram"
    );

    term.restore_mutable_state(&snapshot);
    let restored = (&term.atoms[0].smooth_penalty - &original)
        .mapv(f64::abs)
        .sum();
    assert!(
        restored < 1e-12,
        "line-search restore left a stale intrinsic smoothness Gram: {restored}"
    );
}

/// Gauge invariance (issue #673): a global reparameterization of the latent
/// coordinate scales every per-sample speed by a common factor, which
/// cancels in the centered reweighting — so the intrinsic Gram `S̃` (and
/// hence the topology evidence `tr(BᵀS̃B)`) is identical across the two
/// reparameterizations, even though the basis Jacobian (the metric) differs.
#[test]
pub(crate) fn intrinsic_penalty_is_invariant_to_speed_rescaling() {
    let a1 = intrinsic_test_atom(1.0);
    let a2 = intrinsic_test_atom(7.5);
    // Same raw Gram and decoder; only the basis Jacobian (speed) differs.
    assert_abs_diff_eq!(
        (&a1.smooth_penalty_raw - &a2.smooth_penalty_raw)
            .mapv(f64::abs)
            .sum(),
        0.0,
        epsilon = 1e-12
    );
    // The intrinsic (reweighted) Gram is identical despite the 7.5x speed
    // rescale: the centered ratios are invariant to a global speed factor.
    let diff = (&a1.smooth_penalty - &a2.smooth_penalty)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff < 1e-9,
        "intrinsic Gram changed under a global speed rescale (gauge leak): {diff}"
    );
}

pub(crate) fn affine_canonicalization_test_term() -> SaeManifoldTerm {
    let n = 80usize;
    let p = 2usize;
    let evaluator = EuclideanPatchEvaluator::new(1, 2).unwrap();
    let mut coords = Array2::<f64>::zeros((n, 1));
    for row in 0..n {
        coords[[row, 0]] = -4.0 + 12.0 * row as f64 / (n as f64 - 1.0);
    }
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[0, 0]] = 0.8;
    decoder[[1, 0]] = -0.4;
    decoder[[2, 0]] = 0.15;
    decoder[[0, 1]] = -0.2;
    decoder[[1, 1]] = 0.9;
    decoder[[2, 1]] = -0.08;
    let smooth_penalty = gam_terms::basis::create_difference_penalty_matrix(3, 2, None).unwrap();
    let atom = SaeManifoldAtom::new(
        "affine-canonicalization",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        smooth_penalty,
    )
    .unwrap()
    .with_basis_second_jet(Arc::new(evaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

#[test]
pub(crate) fn affine_canonicalization_transports_live_penalty_instead_of_recomputing() {
    let mut term = affine_canonicalization_test_term();
    let before: f64 = term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let old_smooth_penalty = term.atoms[0].smooth_penalty.clone();
    let old_decoder = term.atoms[0].decoder_coefficients.clone();

    term.canonicalize_atom_affine_gauge(0, None).unwrap();
    let after: f64 = term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let invariant_gap = (after - before).abs() / before.abs().max(1.0);
    assert!(
        invariant_gap < 1.0e-9,
        "canonicalization changed fixed-rho smoothness energy: before={before:.12e}, after={after:.12e}"
    );

    let mut recomputed_atom = term.atoms[0].clone();
    recomputed_atom.refresh_intrinsic_smooth_penalty();
    let recomputed_term = SaeManifoldTerm::new(
        vec![recomputed_atom],
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((term.n_obs(), 1)),
            vec![term.assignment.coords[0].as_matrix()],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap(),
    )
    .unwrap();
    let recomputed: f64 = recomputed_term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let recompute_jump = (recomputed - before).abs() / before.abs().max(1.0);
    assert!(
        recompute_jump > 1.0e-2,
        "test fixture failed to expose the intrinsic recompute energy jump: before={before:.12e}, recomputed={recomputed:.12e}"
    );

    let transport =
        solve_basis_transport(term.atoms[0].basis_values.view(), old_smooth_penalty.view())
            .expect_err("shape mismatch must reject invalid transport solve");
    assert!(
        transport.contains("row mismatch") || transport.contains("SVD failed"),
        "unexpected transport-shape diagnostic: {transport}"
    );
    let roundtrip = transport_smooth_penalty_for_decoder(
        solve_design_least_squares(
            term.atoms[0].decoder_coefficients.view(),
            old_decoder.view(),
        )
        .unwrap_or_else(|err| panic!("decoder transport fixture became singular: {err}"))
        .view(),
        old_smooth_penalty.view(),
    );
    assert!(
        roundtrip.is_err(),
        "non-square decoder transport must not be accepted as a penalty congruence"
    );
}

/// Non-constant speed genuinely reshapes the penalty: the intrinsic Gram
/// must differ from the raw Gram when the decoder curve is not
/// constant-speed, otherwise the reweighting is a no-op and the gauge fix
/// would be vacuous. The congruence preserves symmetry.
#[test]
pub(crate) fn intrinsic_penalty_differs_from_raw_under_varying_speed() {
    let atom = intrinsic_test_atom(1.0);
    let diff = (&atom.smooth_penalty - &atom.smooth_penalty_raw)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff > 1e-6,
        "intrinsic reweighting was a no-op on a non-constant-speed curve: {diff}"
    );
    for i in 0..atom.basis_size() {
        for j in 0..atom.basis_size() {
            assert_abs_diff_eq!(
                atom.smooth_penalty[[i, j]],
                atom.smooth_penalty[[j, i]],
                epsilon = 1e-12
            );
        }
    }
}

/// Constant-speed atoms are untouched: when every sample shares one speed
/// (the periodic sin/cos limit), the centered weights are all `1`, so
/// `S̃ = S_raw` exactly and the topology comparison among constant-speed
/// atoms is unaffected.
#[test]
pub(crate) fn intrinsic_penalty_leaves_constant_speed_atom_unchanged() {
    let m = 6usize;
    let n = m;
    let mut phi = Array2::<f64>::zeros((n, m));
    let mut jet = Array3::<f64>::zeros((n, m, 1));
    let mut decoder = Array2::<f64>::zeros((m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        // Identical derivative magnitude at every sample => constant speed.
        jet[[mu, mu, 0]] = 2.0;
        decoder[[mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let atom = SaeManifoldAtom::new(
        "constant-speed",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap();
    let diff = (&atom.smooth_penalty - &atom.smooth_penalty_raw)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff < 1e-9,
        "constant-speed atom's penalty was reweighted (should be identity): {diff}"
    );
}

/// Build a low-rank decoder atom (`p` large, true column rank `r ≪ p`) and
/// verify the auto-activation installs a frame, the factored border holds
/// exactly `Σ M_k·r_k`, and reconstruction recovers `B_k` to machine
/// precision.
#[test]
pub(crate) fn factored_border_dim_invariant_and_reconstruction() {
    let m = 6usize;
    let p = 16usize;
    let r = 2usize;
    // B = C0 · Frameᵀ with a planted rank-`r` column span.
    let mut frame = Array2::<f64>::zeros((p, r));
    frame[[0, 0]] = 1.0;
    frame[[1, 1]] = 1.0;
    let mut c0 = Array2::<f64>::zeros((m, r));
    for mu in 0..m {
        c0[[mu, 0]] = 1.0 + mu as f64;
        c0[[mu, 1]] = 0.5 * mu as f64 - 1.0;
    }
    let decoder = fast_abt(&c0, &frame);
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        jet[[mu, mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new(
        "lowrank",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder.clone(),
        s_raw,
    )
    .unwrap();
    let activated = atom.maybe_activate_decoder_frame().expect("activate");
    assert_eq!(
        activated,
        Some(r),
        "rank-{r} decoder should profile to r={r}"
    );
    assert_eq!(atom.border_frame_rank(), r);
    assert_eq!(atom.frame_manifold_dimension(), r * (p - r));

    // Reconstruction recovers B_k to machine precision.
    let coords = atom.factored_coordinates().unwrap().expect("coords");
    assert_eq!(coords.dim(), (m, r));
    let reconstructed = atom
        .reconstruct_decoder_coefficients(coords.view())
        .unwrap();
    for mu in 0..m {
        for j in 0..p {
            assert_abs_diff_eq!(reconstructed[[mu, j]], decoder[[mu, j]], epsilon = 1.0e-9);
        }
    }

    let term = SaeManifoldTerm::new(
        vec![atom],
        SaeAssignment::from_blocks_with_mode(
            Array2::<f64>::zeros((m, 1)),
            vec![Array2::<f64>::zeros((m, 1))],
            AssignmentMode::softmax(0.7),
        )
        .unwrap(),
    )
    .unwrap();
    // Border-size invariant: factored border == Σ M_k·r_k.
    grassmann_assert_border_dim_invariant(&term).expect("border invariant");
    assert_eq!(term.factored_border_dim(), m * r);
    assert_eq!(term.grassmann_evidence_dimension(), r * (p - r));
    // Round-trip flatten/scatter of the factored border preserves B_k.
    let mut term = term;
    let border = term.flatten_factored_border().unwrap();
    assert_eq!(border.len(), m * r);
    let saved = term.atoms[0].decoder_coefficients.clone();
    term.scatter_factored_border(border.view()).unwrap();
    for mu in 0..m {
        for j in 0..p {
            assert_abs_diff_eq!(
                term.atoms[0].decoder_coefficients[[mu, j]],
                saved[[mu, j]],
                epsilon = 1.0e-9
            );
        }
    }
}

#[test]
pub(crate) fn factored_beta_penalty_probing_matches_projected_dense_curvature() {
    let k_atoms = 2usize;
    let m = 4usize;
    let p = 24usize;
    let r = 2usize;
    let n_obs = 5usize;
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let mut frame = Array2::<f64>::zeros((p, r));
        frame[[atom_idx * r, 0]] = 1.0;
        frame[[atom_idx * r + 1, 1]] = 1.0;
        let mut coords = Array2::<f64>::zeros((n_obs, 1));
        for row in 0..n_obs {
            coords[[row, 0]] = row as f64;
        }
        let mut phi = Array2::<f64>::zeros((n_obs, m));
        let mut jet = Array3::<f64>::zeros((n_obs, m, 1));
        for row in 0..n_obs {
            for basis_col in 0..m {
                let x = (row + 1) as f64 * (basis_col + 1) as f64;
                phi[[row, basis_col]] = 0.05 * x + if row == basis_col { 1.0 } else { 0.0 };
                jet[[row, basis_col, 0]] = 0.01 * x;
            }
        }
        let mut c = Array2::<f64>::zeros((m, r));
        for basis_col in 0..m {
            c[[basis_col, 0]] = 0.3 + 0.07 * (basis_col + atom_idx) as f64;
            c[[basis_col, 1]] = -0.2 + 0.05 * (basis_col * 2 + atom_idx) as f64;
        }
        let decoder = fast_abt(&c, &frame);
        let mut atom = SaeManifoldAtom::new(
            "factored_probe",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap();
        atom.maybe_activate_decoder_frame()
            .expect("frame activation")
            .expect("rank-2 atom should activate a frame");
        atoms.push(atom);
        coord_blocks.push(coords);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n_obs, k_atoms), 0.25),
        coord_blocks,
        vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    assert!(term.frames_active());
    assert_eq!(term.factored_border_dim(), k_atoms * m * r);

    let beta_len = term.beta_dim();
    let mut registry = AnalyticPenaltyRegistry::new();
    let nuclear = NuclearNormPenalty::new(
        PsiSlice {
            range: 0..beta_len,
            latent_dim: Some(beta_len / p),
        },
        0.7,
        p,
        1.0e-4,
        None,
        false,
    )
    .unwrap();
    registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(nuclear)));
    let incoherence = DecoderIncoherencePenalty::new(
        PsiSlice {
            range: 0..beta_len,
            latent_dim: Some(beta_len / p),
        },
        vec![m, m],
        p,
        Array2::<f64>::from_elem((k_atoms, k_atoms), 0.5),
        0.6,
        false,
    )
    .unwrap();
    registry.push(AnalyticPenaltyKind::DecoderIncoherence(Arc::new(
        incoherence,
    )));

    let mut dense_sys = ArrowSchurSystem::new(0, 0, beta_len);
    let dense_assembly = term
        .add_sae_analytic_penalty_contributions(&mut dense_sys, &registry, 1.0, None, true, None)
        .unwrap();
    assert!(dense_assembly.dense_written);
    assert!(!dense_assembly.deferred_factored);

    let projection = FrameProjection::new(&term);
    let border_dim = term.factored_border_dim();
    let projected = term.project_dense_penalty_to_factored(dense_sys.hbb.view(), &projection);
    let direct = term.build_factored_beta_penalty_curvature(&registry, 1.0, &projection);
    for row in 0..border_dim {
        for col in 0..border_dim {
            assert_abs_diff_eq!(direct[[row, col]], projected[[row, col]], epsilon = 1.0e-10);
        }
    }

    let mut deferred_term = term.clone();
    let rho = SaeManifoldRho::new(
        0.0,
        -20.0,
        vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
    );
    let target = Array2::<f64>::zeros((n_obs, p));
    let sys = deferred_term
        .assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
            target.view(),
            &rho,
            Some(&registry),
            1.0,
            1,
        )
        .unwrap();
    assert_eq!(sys.k, border_dim);
    assert!(sys.hbb.is_empty());
}

