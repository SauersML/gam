//! #1783 regression gate — the `d_atom = 1` (1-D curve / circle) manifold-SAE
//! fit must ROUTE to the device-resident SAE PCG at realistic scale, not run the
//! inner arrow-Schur on one CPU core with the GPU idle at 0%.
//!
//! ROOT CAUSE (issue #1783): the primary manifold-SAE regime is a dictionary of
//! 1-D curve atoms fitted with the isometry gauge ON (the `#795` `t`↔`B` coupling
//! the circle fit needs to reach KKT stationarity). Curved atoms take the
//! FACTORED (Grassmann-frame) β-tier, whose device data is only installed when
//! `has_dense_beta_penalty == false` (the device framed kernel cannot model an
//! extra dense `H_ββ` term). The isometry contribution, however, is NOT a dense
//! `H_ββ` term: `add_sae_isometry_beta_penalty` writes only the per-row `H_tt`
//! curvature and the `H_tβ` cross-block, and `build_factored_beta_penalty_
//! curvature` ignores Isometry entirely (it adds ZERO to the factored `hbb_c`).
//! Yet the assembly used to call `beta_assembly.record_curvature(...)` for the
//! isometry, spuriously setting `has_dense_beta_penalty == true` → the framed
//! path SKIPPED `set_device_sae_pcg_data` → `device_sae_pcg == None` → the
//! Direct/InexactPCG device seams both declined at the very first guard, so every
//! curved (`d_atom = 1`) fit was CPU-bound with the GPU at 0% (the reporter's
//! B200 observation).
//!
//! FIX: the isometry no longer records a dense/deferred β-tier curvature, so the
//! framed device SAE PCG data is installed for a curved fit. The CPU dense
//! reference and the device framed kernel already both carry the isometry `H_tt`
//! (per-row Cholesky factors) and `H_tβ` (device `row_htbeta`), so the numerics
//! are unchanged — only WHERE the reduced-Schur matvec runs changes.
//!
//! WHAT IS VERIFIED HERE (CPU-observable, no GPU needed):
//!   * `device_sae_pcg.is_some()` for a framed circle + isometry fit — the
//!     routing DATA the device seam requires is now built (was `None`).
//!   * The `reduced_schur_matvec_should_offload` policy admits the reporter's
//!     exact `d_atom = 1` realistic shape, so the device WOULD be selected at
//!     scale on a CUDA host.
//! The remaining `0% → engaged` GPU-utilisation observation needs real CUDA
//! hardware (the device kernels only launch on Linux + a present runtime); that
//! is the GPU-gated remainder, pinned on hardware by the `#1551` engagement tests
//! (`owed_1551_sae_direct_device_engage.rs`).

use super::tests::{TestPeriodicEvaluator, periodic_basis};
use super::tests_factored_htbeta::{factored_htbeta_rho, low_rank_factored_htbeta_term};
use super::*;
use crate::assignment::{AssignmentMode, SaeAssignment};
use gam_terms::latent::LatentManifold;
use ndarray::array;
use std::sync::Arc;

/// Build a framed (`p ≥ SAE_FRAME_MIN_AUTO_OUTPUT_DIM`) single-circle
/// (`d_atom = 1`) SAE term whose low-rank decoder auto-activates a Grassmann
/// frame, plus the exact decoded target that makes the isometry gauge live.
/// Returns the term (frames NOT yet activated), the target, and the ρ.
fn framed_circle_term() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 64usize;
    // p ≥ 12 (SAE_FRAME_MIN_AUTO_OUTPUT_DIM) so the cold low-rank decoder profiles
    // out a Grassmann frame → the FACTORED β-tier (where the #1783 skip lived).
    let p = 16usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let m = phi.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        (1.0 / (1.0 + b as f64)) * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });
    let atom = SaeManifoldAtom::new(
        "iso_dev_1783",
        SaeAtomBasisKind::Periodic,
        1,
        phi.clone(),
        jet.clone(),
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    // Target = the exact decoded curve, so the isometry gauge has a live pullback
    // metric (the `#795` fixtures use the same construction).
    let target = phi.dot(&decoder);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![1.0_f64.ln()]]);
    (term, target, rho)
}

/// An analytic-penalty registry carrying ONLY the isometry gauge for the single
/// circle atom (the primary `d_atom = 1` regime).
fn isometry_registry(n: usize) -> gam_terms::analytic_penalties::AnalyticPenaltyRegistry {
    use gam_terms::analytic_penalties::{
        AnalyticPenaltyKind, AnalyticPenaltyRegistry, IsometryPenalty, PsiSlice,
    };
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
        IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(1)), 1),
    )));
    registry
}

/// #1783 CORE FIX: a framed circle (`d_atom = 1`) fit with the isometry gauge ON,
/// in the reporter's DEFERRED-factored (large-`K`) β-tier, must install the
/// device-resident SAE PCG data. Before the fix the isometry spuriously flagged a
/// deferred β penalty, `set_device_sae_pcg_data` was skipped, and `device_sae_pcg`
/// stayed `None` — the device seam declined at its first guard and the fit ran on
/// CPU (GPU 0%).
///
/// The fix is regime-aware, not a blanket "never record": `add_sae_isometry_beta_
/// penalty` writes a REAL dense `H_ββ` block into `sys.hbb` on the DENSE β-tier
/// (`dense_beta_curvature == true`) and NOTHING on the DEFERRED-factored tier
/// (`dense_beta_curvature == false`, the reporter's `beta_dim ≫ 4096` regime). So
/// this test pins BOTH regimes off one fixture via the β-penalty probe threshold:
///   * deferred tier (probe = 1 ⇒ `beta_dim > 1` ⇒ `dense_beta_curvature=false`):
///     no dense `hbb`, device data INSTALLED — the #1783 fix / the B200 shape.
///   * dense tier (default probe = 4096 ⇒ tiny `beta_dim=48` ⇒
///     `dense_beta_curvature=true`): a genuine dense isometry `hbb` is written,
///     which the framed device kernel cannot model, so the device correctly
///     DECLINES (CPU fallback) — and, critically, `dense_written` is set so the
///     assembly still APPLIES that curvature (the #795 decoder-side coupling).
///     Dropping the record here (the over-broad first cut of the #1783 fix) would
///     silently forfeit that curvature.
#[test]
fn framed_circle_isometry_fit_builds_device_sae_pcg_data_1783() {
    let (mut term, target, rho) = framed_circle_term();
    let n = target.nrows();

    // The curved low-rank decoder must profile out a Grassmann frame, so the fit
    // takes the FACTORED β-tier — the exact path the #1783 skip lived on.
    let activated = term
        .auto_activate_decoder_frames()
        .expect("frame auto-activation must succeed");
    assert!(
        activated > 0 && term.frames_active(),
        "the framed β-tier must engage for this fixture (activated={activated}); \
         otherwise the test would exercise the full-B path where device data is \
         always built and the #1783 skip never applied"
    );

    let registry = isometry_registry(n);

    // ── DEFERRED-factored tier (the reporter's large-K regime) ──────────────
    // Probe threshold 1 forces `dense_beta_curvature == false` for any
    // `beta_dim > 1`, exactly as the reporter's `beta_dim ≫ 4096` fit does. The
    // isometry writes NO dense `hbb`, so the device data MUST be installed.
    let sys_deferred = term
        .assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
            target.view(),
            &rho,
            Some(&registry),
            1.0,
            1,
        )
        .expect("framed deferred-factored circle + isometry assemble must succeed");
    assert!(
        sys_deferred.device_sae_pcg.is_some(),
        "#1783: the deferred-factored (large-K) framed circle + isometry regime — \
         the reporter's actual B200 shape — must carry device_sae_pcg data so the \
         device-resident SAE PCG seam is reachable (was None: silent CPU, GPU 0%)"
    );
    assert!(
        sys_deferred
            .device_sae_pcg
            .as_ref()
            .unwrap()
            .frame
            .is_some(),
        "the framed fit must carry the factored (Grassmann-frame) device payload"
    );

    // Control: the SAME deferred fit WITHOUT the isometry also builds device data.
    // This shows the isometry was the sole reason the device was skipped in the
    // deferred regime — the fix brings the isometry case to parity with the
    // no-penalty case, it does not newly enable anything the path did not support.
    let sys_bare_deferred = term
        .assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
            target.view(),
            &rho,
            None,
            1.0,
            1,
        )
        .expect("framed deferred bare assemble must succeed");
    assert!(
        sys_bare_deferred.device_sae_pcg.is_some(),
        "the deferred framed circle fit without penalties must also carry device \
         data (baseline for the isometry parity)"
    );
    // The border layout is identical with and without the isometry (the gauge adds
    // no β-block width), so the device border matches — the isometry rides the
    // existing frame border, not a widened one.
    assert_eq!(
        sys_deferred.k, sys_bare_deferred.k,
        "the isometry gauge must not change the factored border width"
    );

    // ── DENSE β-tier (tiny beta_dim=48 ≤ 4096 ⇒ dense_beta_curvature=true) ───
    // Here `add_sae_isometry_beta_penalty` writes a genuine dense `H_ββ` into
    // `sys.hbb`. The framed device kernel cannot model an extra dense border term,
    // so the device correctly DECLINES; the curvature is applied on the CPU path
    // instead. (Contrast: the bare dense fit, with no dense term, DOES install.)
    let sys_dense_iso = term
        .assemble_arrow_schur(target.view(), &rho, Some(&registry))
        .expect("framed dense circle + isometry assemble must succeed");
    assert!(
        sys_dense_iso.device_sae_pcg.is_none(),
        "#1783 regression pin: on the DENSE β-tier the isometry writes a real dense \
         H_ββ the framed device kernel cannot model, so the device must decline \
         (CPU fallback) — and `dense_written` must be set so that curvature is \
         still applied, not silently dropped"
    );
    let sys_dense_bare = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("framed dense bare assemble must succeed");
    assert!(
        sys_dense_bare.device_sae_pcg.is_some(),
        "on the DENSE β-tier a fit with NO dense penalty carries no border dense \
         term, so the device data is installed — proving it is the isometry's dense \
         H_ββ (not the frame layout) that declines the dense-tier device path"
    );
}

/// #1783 SCALE HONESTY (pure policy, CPU-observable): the reporter's exact
/// realistic `d_atom = 1` shape must clear the reduced-Schur offload gate, so on a
/// CUDA host the device WOULD be selected. The gate was never the blocker for
/// `d_atom = 1` — this pins that it admits the thin-atom regime at token scale so
/// a future regression cannot re-introduce a `d == 1` threshold miss.
#[test]
fn offload_gate_admits_d_atom_1_at_token_scale_1783() {
    let policy = gam_gpu::policy::GpuDispatchPolicy::default();

    // Reporter's B200 run: X = (40456, 2048), K = 256, d_atom = 1, circle. The
    // factored border for 256 curve atoms clears DEVICE_LOOP_MIN_P (32), and even
    // a single CG apply clears MATVEC_OFFLOAD_FLOPS_MIN:
    //   n·(2·d·k + d²) = 40456·(2·1·256 + 1) ≈ 2.07e7 ≥ 1e7.
    assert!(
        policy.reduced_schur_matvec_should_offload(40_456, 256, 1, 1),
        "#1783: the reporter's realistic d_atom=1 shape (n=40456, k=256, d=1) must \
         clear the offload gate at a single CG apply — the gate was never the \
         blocker for thin curve atoms"
    );

    // The earlier K=64 case at n=24576 also clears (with the conservative default
    // CG budget the seam derives).
    assert!(
        policy.reduced_schur_matvec_should_offload(
            24_576,
            64,
            1,
            gam_gpu::policy::GpuDispatchPolicy::MATVEC_OFFLOAD_MIN_CG_ITERS,
        ),
        "#1783: the earlier K=64 d_atom=1 shape must also clear the offload gate"
    );

    // Honesty floor: a genuinely tiny thin-atom shape still stays on the CPU (the
    // launch/staging cost dominates), so admitting d=1 at scale did not defeat the
    // small-shape guard.
    assert!(
        !policy.reduced_schur_matvec_should_offload(64, 9, 1, 1),
        "a tiny framed circle fixture must NOT engage the device (honest fallback)"
    );
}

/// #1017 production-shape fixture whose *actual assembled border* remains large
/// after the automatic Grassmann factorization. The color example is important
/// for end-to-end convergence, but its rank-3 decoder collapses `15360` model
/// coefficients to a border no wider than `9`, so it cannot exercise the
/// production InexactPCG/resident-frame lane. This fixture goes through the real
/// SAE assembler and lands just beyond the direct-solve boundary.
#[test]
fn production_factored_large_border_routes_to_resident_inexact_pcg_1017() {
    const K_ATOMS: usize = 32;
    const M: usize = 8;
    const P: usize = 64;
    const FRAME_RANK: usize = 8;
    const LATENT_DIM: usize = 1;
    const N_OBS: usize = 32;

    let mut term = low_rank_factored_htbeta_term(K_ATOMS, M, P, FRAME_RANK, LATENT_DIM, N_OBS);
    let rho = factored_htbeta_rho(K_ATOMS, LATENT_DIM);
    let target = Array2::<f64>::from_shape_fn((N_OBS, P), |(row, col)| {
        1.0e-3 * ((row + 1) as f64 * (col + 3) as f64).sin()
    });
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("production large-border assembly");

    assert_eq!(term.beta_dim(), K_ATOMS * M * P);
    assert_eq!(sys.k, K_ATOMS * M * FRAME_RANK);
    assert_eq!(sys.k, 2_048, "fixture must clear DIRECT_SOLVE_MAX_K=2000");
    let device = sys
        .device_sae_pcg
        .as_ref()
        .expect("large factored assembly must install device operands");
    assert!(
        device.frame.is_some(),
        "production fixture must use the framed device payload"
    );
    let plan = term
        .streaming_plan()
        .admitted_or_error(N_OBS, P, K_ATOMS)
        .expect("fixture admitted by production memory plan");
    let mut options = plan.solve_options_for_border_dim(sys.k);
    assert_eq!(
        options.mode,
        gam_solve::arrow_schur::ArrowSolverMode::InexactPCG,
        "the actual assembled border, not full beta_dim, selects the solver"
    );
    let cg_iters = options
        .pcg
        .max_iterations
        .min(options.trust_region.max_iterations);
    let offload_admitted = gam_gpu::policy::GpuDispatchPolicy::default()
        .reduced_schur_matvec_should_offload(sys.rows.len(), sys.k, sys.d, cg_iters);
    let operand_report = device.operand_byte_report();
    eprintln!(
        "#1017 production large-border telemetry: beta_dim={} factored_border={} \
         rows={} d={} mode={:?} cg_iters={} framed={} offload_admitted={} {}",
        term.beta_dim(),
        sys.k,
        sys.rows.len(),
        sys.d,
        options.mode,
        cg_iters,
        device.frame.is_some(),
        offload_admitted,
        operand_report,
    );
    assert!(
        offload_admitted,
        "representative production system must clear resident-device admission"
    );

    // On the A100 validation host this same exact production fixture must cross
    // the final runtime gate, upload one resident frame, and consume it in the
    // actual InexactPCG inner solve. CPU CI keeps the routing assertions above.
    #[cfg(target_os = "linux")]
    if gam_gpu::device_runtime::GpuRuntime::global().is_some() {
        let started = std::time::Instant::now();
        let frame = gam_solve::arrow_schur::prepare_sae_resident_frame(&sys, &options, None)
            .expect("live CUDA runtime must admit the production resident frame");
        options.sae_resident_frame = Some(frame);
        let (_delta_t, _delta_beta, diagnostics) =
            gam_solve::arrow_schur::solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
                .expect("resident InexactPCG production solve");
        assert!(
            diagnostics.used_device_arrow,
            "production A100 solve must report genuine device execution"
        );
        eprintln!(
            "#1017 A100 resident solve telemetry: used_device_arrow={} elapsed_ms={:.3} \
             pcg_iterations={} matvec_calls={} ridge_escalations={} {}",
            diagnostics.used_device_arrow,
            started.elapsed().as_secs_f64() * 1.0e3,
            diagnostics.iterations,
            diagnostics.matvec_calls,
            diagnostics.ridge_escalations,
            operand_report,
        );
    }
}
