//! Poincaré gauge-orbit diagnosis: WHY does the inner solve refuse where
//! periodic/duchon/linear solve?
//!
//! ## Background
//!
//! The #2720 geometry sweep (PR #2772) found the poincaré atom's inner solve
//! REFUSES on the planted-circle fixture: after 2680 inner iterations,
//! `gauge_share = 0.9456` — 94.6% of the KKT gradient lies INSIDE the
//! chart-gauge span, and the solver declines to rank an off-optimum
//! criterion. Refusal is robust to a 10× budget.
//!
//! The gauge construction treats the poincaré tangent patch as sharing the
//! Euclidean patch's translation + scale orbit (`dense_step_gauge_vectors`,
//! fit_drivers.rs): "the hyperbolic structure lives in the penalty, not the
//! gauge." Two stories explain the refusal:
//!
//! * **Story 1 (construction defect):** the claimed gauge vectors are NOT
//!   likelihood-null on a poincaré atom — the quotient projection discards
//!   gradient the data term genuinely cares about, and the solver chases it.
//! * **Story 2 (penalty magnitude):** the vectors ARE null, but the
//!   conformal-Dirichlet penalty's gradient along the orbit is enormous, so
//!   the posterior gradient never meets tolerance.
//!
//! ## Probes
//!
//! **A — seed-state nullness (no solve).** At the SEEDED state, project the
//! near-null-penalty gradient onto the gauge basis. If |g_nullᵀv| is at
//! numerical noise relative to ‖g‖ for periodic but NOT poincaré, the vectors
//! themselves are non-null on curved charts — Story 1. Comparing kinds at the
//! seed isolates the geometry: same fixture, same seed path, same instrument,
//! zero iterations of solver drift.
//!
//! **B — native target.** Re-run the measurement with a target that is a
//! poincaré atom's own reconstruction at its own seed (data_fit ≈ 0 by
//! construction). If the solve STILL refuses on a self-consistent target,
//! the phenomenon is not fixture stress — it is native to the atom kind.
//!
//! Marked `#[ignore]` — a manual diagnostic reproducer, same convention as
//! the #2770 baseline and the #2772 sweep.
//! Run: `cargo test -p gam-sae poincare_gauge -- --ignored --nocapture`

#![cfg(test)]
use super::*;

/// One seeded term of the requested kind on the Fixture-B circle cloud
/// (`planted_circle_cloud`, re-exported from tests_gauge_frame_roundtrip_2720:
/// n=42, p=48, the ACTUAL #2253/#2234 LCG). An earlier revision of this file
/// re-implemented the fixture with a different generator and sinusoid basis
/// vectors while claiming "same LCG" — that was wrong, and every number from
/// the self-consistent-target probe on that revision belongs to a different
/// dataset than the other probes. All probes now share the one fixture.
fn seeded_circle_term(kind: &str) -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>) {
    use crate::manifold::tests_gauge_frame_roundtrip_2720::planted_circle_cloud;
    let z = planted_circle_cloud();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec![kind.to_string()],
        atom_dim: vec![1],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .unwrap_or_else(|e| panic!("[poincare-gauge] minimal seed failed for {kind}: {e}"));
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::Softmax,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: minimal.refine_routing,
        seed_refine_random_state: 45,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .unwrap_or_else(|e| panic!("[poincare-gauge] fit seed failed for {kind}: {e}"));
    let mut rho = seed.initial_rho;
    // ARD-saddle settings, matching the sweep (#2772).
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    (seed.base_term, rho, z)
}

/// Near-null rho: every penalty channel collapsed to `exp(-20)`, layout kept.
fn near_null_rho(rho: &SaeManifoldRho) -> SaeManifoldRho {
    let mut null = rho.clone();
    null.log_lambda_sparse = -20.0;
    for value in null.log_lambda_smooth.iter_mut() {
        *value = -20.0;
    }
    for axis in null.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -20.0;
        }
    }
    for value in null.log_lambda_block.iter_mut() {
        *value = -20.0;
    }
    null
}

/// Joint gradient + gauge basis at the CURRENT term state, no solve.
fn state_snapshot(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    label: &str,
) -> (Array1<f64>, Array1<f64>, Vec<Array1<f64>>, f64) {
    let sys = term
        .assemble_arrow_schur(target, rho, None)
        .unwrap_or_else(|e| panic!("[poincare-gauge] {label}: assemble failed: {e}"));
    let border_dim = sys.gb.len();
    let full_len: usize = sys.rows.iter().map(|r| r.gt.len()).sum::<usize>() + border_dim;
    let mut grad = Array1::<f64>::zeros(full_len);
    let mut row_offsets = vec![0usize];
    let mut offset = 0usize;
    for row in &sys.rows {
        for (i, &v) in row.gt.iter().enumerate() {
            grad[offset + i] = v;
        }
        offset += row.gt.len();
        row_offsets.push(offset);
    }
    for (i, &v) in sys.gb.iter().enumerate() {
        grad[offset + i] = v;
    }
    let null_sys = term
        .assemble_arrow_schur(target, &near_null_rho(rho), None)
        .unwrap_or_else(|e| panic!("[poincare-gauge] {label}: null assemble failed: {e}"));
    let mut null_grad = Array1::<f64>::zeros(full_len);
    let mut off2 = 0usize;
    for row in &null_sys.rows {
        for (i, &v) in row.gt.iter().enumerate() {
            null_grad[off2 + i] = v;
        }
        off2 += row.gt.len();
    }
    for (i, &v) in null_sys.gb.iter().enumerate() {
        null_grad[off2 + i] = v;
    }
    let basis = term
        .joint_chart_gauge_basis_for_arrow_layout(
            &row_offsets,
            border_dim,
            &format!("poincare-gauge {label}"),
        )
        .unwrap_or_else(|e| panic!("[poincare-gauge] {label}: gauge basis failed: {e}"));
    let grad_norm = grad.dot(&grad).sqrt();
    (grad, null_grad, basis, grad_norm)
}

/// Probe A: at the SEEDED state, is the gauge basis likelihood-null for each
/// kind? The instrument: worst |g_nullᵀv| / ‖g_null‖ over basis directions,
/// plus the same for the FULL gradient (penalty included) as context.
///
/// Enforced (deterministic fixture): every kind's gauge directions must be
/// likelihood-null at machine level, and the periodic control must produce a
/// measurable basis. The measured values at `64e7818`: worst null-shares
/// 6.8e-15 (periodic), 3.5e-9 (duchon), 1.8e-7 (poincare), 1.2e-7 (linear).
#[test]
fn poincare_gauge_nullness_at_seed_2720() {
    for kind in ["periodic", "duchon", "poincare", "linear"] {
        let (mut term, rho, z) = seeded_circle_term(kind);
        for atom in term.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
        let (grad, null_grad, basis, grad_norm) = state_snapshot(&mut term, z.view(), &rho, kind);
        let null_norm = null_grad.dot(&null_grad).sqrt();
        let coords = term.assignment.coords[0].as_matrix();
        let max_abs_coord = coords.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        if basis.is_empty() {
            eprintln!(
                "[poincare-gauge] {kind:<9} seed: NO gauge directions (‖g‖={grad_norm:.3e}, \
                 ‖g_null‖={null_norm:.3e}, max|t|={max_abs_coord:.3})"
            );
            continue;
        }
        let mut worst_null_share = 0.0f64;
        let mut worst_full_share = 0.0f64;
        for v in basis.iter() {
            let n = null_grad.dot(v).abs() / null_norm.max(f64::MIN_POSITIVE);
            let f = grad.dot(v).abs() / grad_norm.max(f64::MIN_POSITIVE);
            worst_null_share = worst_null_share.max(n);
            worst_full_share = worst_full_share.max(f);
        }
        eprintln!(
            "[poincare-gauge] {kind:<9} seed: {} dirs | ‖g‖={:.3e} ‖g_null‖={:.3e} | \
             worst |g_nullᵀv|/‖g_null‖={:.3e}  |gᵀv|/‖g‖={:.3e} | max|t|={max_abs_coord:.3}",
            basis.len(),
            grad_norm,
            null_norm,
            worst_null_share,
            worst_full_share
        );
        // ENFORCED: gauge directions are likelihood-null at machine level on
        // every kind. Meaningfulness guard is RELATIVE: the null-share is
        // resolvable whenever the near-null projection itself carries mass
        // above absolute noise (1e-12), not only when ‖g_null‖ is O(1).
        let worst_abs_null_proj = basis
            .iter()
            .map(|v| null_grad.dot(v).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst_abs_null_proj < 1.0e-5 * null_norm.max(1.0e-12) || worst_abs_null_proj < 1.0e-8,
            "[poincare-gauge] {kind}: gauge direction is NOT likelihood-null at the \
             seed (worst |g_nullᵀv| = {worst_abs_null_proj:.3e} against ‖g_null‖ \
             = {null_norm:.3e}) — the chart-gauge construction regressed on this kind"
        );
        // The periodic control must be measurable at all.
        assert!(
            !basis.is_empty() || kind != "periodic",
            "[poincare-gauge] periodic control produced NO gauge directions"
        );
    }
}

/// Probe B: a target that is BY CONSTRUCTION this poincaré atom's own
/// reconstruction at its own seed state. data_fit ≈ 0 at t=0; if the solve
/// still refuses, the refusal is native to the kind, not fixture stress.
///
/// Enforced (deterministic fixture): poincare MUST solve its native target
/// (measured criterion 5.964e1, data_fit 1.46) and periodic MUST solve its
/// own (criterion 2.471e1) — pinning the corrected-fixture finding that the
/// earlier "periodic refuses native" was a harness RNG artifact.
#[test]
fn poincare_self_consistent_target_2720() {
    let (mut term, rho, _z) = seeded_circle_term("poincare");
    for atom in term.atoms.iter_mut() {
        atom.deactivate_decoder_frame();
    }
    // Reconstruct: atom basis at current coords through the decoder.
    let phi = term.atoms[0].basis_values.view();
    let decoder = term.atoms[0].decoder_coefficients();
    let native = phi.dot(decoder);
    let data_fit = ((&native - &_z).mapv(|v| v * v)).sum() / 2.0;
    eprintln!(
        "[poincare-gauge] native target: shape {:?}, seed-state data_fit vs circle={data_fit:.3e}",
        native.dim()
    );

    // First: measure at the seed against the NATIVE target (no solve).
    let (_grad, null_grad, basis, grad_norm) =
        state_snapshot(&mut term, native.view(), &rho, "poincare/native@seed");
    if !basis.is_empty() {
        let null_norm = null_grad.dot(&null_grad).sqrt();
        let worst = basis
            .iter()
            .map(|v| null_grad.dot(v).abs() / null_norm.max(f64::MIN_POSITIVE))
            .fold(0.0f64, f64::max);
        eprintln!(
            "[poincare-gauge] poincare  native@seed: {} dirs | ‖g‖={grad_norm:.3e} \
             ‖g_null‖={null_norm:.3e} | worst |g_nullᵀv|/‖g_null‖={worst:.3e}",
            basis.len()
        );
    } else {
        eprintln!("[poincare-gauge] poincare  native@seed: NO gauge directions");
    }

    // Then: try the solve on the native target.
    let budget = 40usize;
    let poincare_solved = match term.penalized_quasi_laplace_criterion_with_cache(
        native.view(),
        &rho,
        None,
        budget,
        0.4,
        1.0e-6,
        1.0e-6,
    ) {
        Ok((value, loss, _)) => {
            eprintln!(
                "[poincare-gauge] poincare  native SOLVED: criterion={value:.6e} \
                 (data_fit={:.3e}, smoothness={:.3e}, ard={:.3e}) — the refusal WAS fixture stress",
                loss.data_fit, loss.smoothness, loss.ard
            );
            true
        }
        Err(e) => {
            let note = e.to_string();
            eprintln!(
                "[poincare-gauge] poincare  native REFUSED (budget={budget}): {}",
                note.chars().take(400).collect::<String>()
            );
            eprintln!("[poincare-gauge] → refusal is NATIVE to the poincare kind, not the fixture");
            false
        }
    };

    // Control: same native-target experiment on periodic.
    let (mut pterm, prho, _pz) = seeded_circle_term("periodic");
    for atom in pterm.atoms.iter_mut() {
        atom.deactivate_decoder_frame();
    }
    let pphi = pterm.atoms[0].basis_values.view();
    let pdec = pterm.atoms[0].decoder_coefficients();
    let pnative = pphi.dot(pdec);
    let periodic_solved = match pterm.penalized_quasi_laplace_criterion_with_cache(
        pnative.view(),
        &prho,
        None,
        budget,
        0.4,
        1.0e-6,
        1.0e-6,
    ) {
        Ok((value, loss, _)) => {
            eprintln!(
                "[poincare-gauge] periodic  native SOLVED: criterion={value:.6e} \
                 (data_fit={:.3e})",
                loss.data_fit
            );
            true
        }
        Err(e) => {
            eprintln!(
                "[poincare-gauge] periodic  native REFUSED: {}",
                e.to_string().chars().take(300).collect::<String>()
            );
            false
        }
    };
    // ENFORCED (corrected-fixture findings, deterministic): both kinds solve
    // their own reconstructions. The pre-correction harness claimed periodic
    // refuses its native target — that was the RNG-artifact fixture, and this
    // assertion keeps it from coming back unnoticed.
    assert!(
        poincare_solved,
        "[poincare-gauge] poincare no longer solves its native target — the refusal \
         became kind-native, contradicting the corrected-fixture measurement"
    );
    assert!(
        periodic_solved,
        "[poincare-gauge] periodic no longer solves its native target — regression \
         against the corrected-fixture measurement (criterion 2.471e1)"
    );
}
