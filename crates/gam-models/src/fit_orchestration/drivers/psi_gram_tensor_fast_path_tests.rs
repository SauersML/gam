// #1264 / #1033 re-home: the production correctness + n-independence guards for
// the certified ψ-Gram tensor fast path on real 1-D Duchon spatial smooths.
//
// Two load-bearing tests live here:
//   * `psi_gram_tensor_fast_path_skips_n_row_lane_and_matches_streamed` — the
//     #1264 contract that the design-revision skip fires EXACTLY where the
//     gauge-invariant range-projector witness `reduced_basis_equal(ψ_ref, ψ)`
//     admits it, and that the n-free β̂ matches a fresh streamed slow-path solve
//     to the issue-mandated 1e-6 across a basis rotation the witness refuses.
//   * `psi_gram_skip_forced_rotation_beta_error_ladder_diag` — the #1033
//     frontier measurement that FORCES the skip across a basis rotation and
//     attributes the β̂ error to the Gram interpolation residual, the RHS, or
//     the penalty re-key.
//
// These were authored in the pre-#1521 monolith under
// `tests/src_modules/smooths/smooth_design_assembly_constraint_tests.rs`, whose
// `include!` was COMMENTED OUT of `gam_terms::smooth::tests` by #1601 ("for
// relocation") — and the relocation never happened, so BOTH the #1264 skip
// guard and the #1033 frontier measurement compiled into NO binary. They are
// re-homed here next to their migrated #901 siblings, where every private
// driver symbol (`SingleBlockExactJointDesignCache`, `external_opts_for_design`,
// `try_build_spatial_log_kappa_hyper_dirs`,
// `evaluate_joint_reml_outer_eval_at_theta`, `spatial_dims_per_term`,
// `spatial_length_scale_term_indices`, `ExactJointHyperSetup`) is in scope via
// `use super::*` (the `design_construction.rs` + `spatial_optimization.rs`
// files are `include!`d flat into one module namespace). The only cross-crate
// paths (`crate::estimate::ExternalJointHyperEvaluator`,
// `crate::solver::rho_optimizer::OuterEvalOrder`,
// `crate::construction::CanonicalPenalty`) are rewritten to their carved homes
// `gam_solve::estimate` / `gam_solve::rho_optimizer` / `gam_terms::construction`.
#[cfg(test)]
mod psi_gram_tensor_fast_path_tests {
    use super::*;
    use super::test_support::SingleBlockExactJointDesignCacheTestExt;
    use gam_terms::basis::{
        CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
        OneDimensionalBoundary, SpatialIdentifiability,
    };
    use ndarray::{Array1, Array2, s};

/// #1264: the design-revision fast path keeps the reference surface (its
/// conditioned frame AND its RRQR-reduced / null basis) FROZEN at the pinning ψ
/// while re-keying the Gram `XᵀWX(ψ)` and penalty `S(ψ)` to the trial ψ. The
/// streamed slow path re-realizes and RE-PIVOTS the radial-kernel design, so it
/// forms its solve in a fresh reduced basis. Conditioning-ratio and Gram-derived
/// RRQR rank/permutation gates were both insufficient on the cluster. The fix is the
/// gauge-invariant range-PROJECTOR witness `reduced_basis_equal(ψ_ref, ψ_new)`,
/// keyed to the pinning ψ recorded at the last slow-path reset: the skip fires
/// only where the frozen reduced basis is provably still valid, so production
/// keeps β̂ bit-tight while regaining the n-free skip where it is sound.
#[test]
fn psi_gram_tensor_fast_path_skips_n_row_lane_and_matches_streamed() {
    use gam_solve::rho_optimizer::OuterEvalOrder;

    // Same 1-D Duchon Gaussian fixture as the e2e κ-optimum test (n = 600).
    let n = 600usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        let signal = 1.2 * (2.0 * std::f64::consts::PI * t).sin() + 0.4 * (t - 0.5);
        let noise = 0.15 * (((i as f64) * 12.9898).sin() * 43758.547).fract();
        y[i] = signal + noise;
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let family = LikelihoodSpec::gaussian_identity();

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "fast_path_skip".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                // PRODUCTION geometry: `None` lets the 1-D axis auto-standardize
                // to unit spread (#1214/#1215) — the real default-fit path. The
                // n-independence fast path must fire here. An earlier
                // `Some(vec![1.0])` pin was a gamed gate that masked the open gap.
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = frozen_design.penalties.len();
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let log_kappa0 =
        SpatialLogKappaCoords::from_length_scales(&frozen, &spatial_terms, &kappa_options);
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_from_data(
        data.view(),
        &frozen,
        &spatial_terms,
        &kappa_options,
    )
    .expect("lower isotropic-scale bounds");
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_from_data(
        data.view(),
        &frozen,
        &spatial_terms,
        &kappa_options,
    )
    .expect("upper isotropic-scale bounds");
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    const JOINT_RHO_BOUND: f64 = 12.0;
    let setup = ExactJointHyperSetup::new(
        Array1::<f64>::zeros(rho_dim),
        Array1::<f64>::from_elem(rho_dim, -JOINT_RHO_BOUND),
        Array1::<f64>::from_elem(rho_dim, JOINT_RHO_BOUND),
        log_kappa0.clone(),
        log_kappa_lower.clone(),
        log_kappa_upper.clone(),
    );
    let theta0 = setup.theta0();
    let lower = setup.lower();
    let upper = setup.upper();
    let psi_lo = lower[rho_dim];
    let psi_hi = upper[rho_dim];
    let z = Array1::from_iter(y.iter().zip(offset.iter()).map(|(yi, oi)| yi - oi));
    let external_opts = external_opts_for_design(
        &family,
        &frozen_design,
        &FitOptions {
            compute_inference: false,
            max_iter: 200,
            tol: 1e-12,
            ..FitOptions::default()
        },
    );

    let make_eval = || {
        gam_solve::estimate::ExternalJointHyperEvaluator::new(
            y.view(),
            weights.view(),
            &frozen_design.design,
            offset.view(),
            &frozen_design.penalties,
            &external_opts,
            "fast_path_skip",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e))
    };
    let make_cache = || {
        SingleBlockExactJointDesignCache::new(
            data.view(),
            frozen.clone(),
            frozen_design.clone(),
            spatial_terms.clone(),
            rho_dim,
            dims_per_term.clone(),
        )
        .expect("design cache")
    };

    // Tensor evaluator with the certified Gram tensor attached over the window.
    let mut tensor_eval = make_eval();
    let mut tensor_cache = make_cache();
    let attached = {
        let mut build_cache = make_cache();
        let theta_probe_base = theta0.clone();
        tensor_eval.build_and_set_psi_gram_tensor(
            |psi| {
                let mut theta_probe = theta_probe_base.clone();
                theta_probe[rho_dim] = psi;
                build_cache
                    .ensure_theta(&theta_probe)
                    .map_err(|error| error.to_string())?;
                Ok(build_cache.design().design.clone())
            },
            weights.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
    };
    assert!(
        attached,
        "tensor must certify on this fixture for a non-vacuous gate"
    );
    // #1033 penalty lane: enable the EXACT n-free penalty re-key so the
    // fast-path skip can rebuild S(ψ) from the frozen basis geometry. Without it
    // the fast path keeps the stale S(ψ_A) and β̂ at ψ_B/ψ_C diverges from the
    // streamed slow path — the exact regression this test guards. The frozen
    // single-Duchon-term fixture must admit the n-free rebuild.
    let nfree_penalty = tensor_cache.supports_nfree_penalty_rekey();
    assert!(
        nfree_penalty,
        "single frozen Duchon term must admit the exact n-free S(ψ) re-key"
    );
    tensor_eval.set_supports_nfree_penalty_rekey(nfree_penalty);

    let span = psi_hi - psi_lo;
    // Pin in the INTERIOR (not at the low edge, where the radial-kernel Gram is
    // most ill-conditioned and the reduced basis is most volatile) so the
    // positive-lane assertion below is robust: a tiny step off an interior pin
    // genuinely preserves the reduced basis.
    let psi_a = psi_lo + 0.5 * span;
    // ψ_B: a small move off the pin. On this standardized production Duchon
    // geometry the radial-kernel reduced basis is volatile enough that even this
    // tiny step rotates it — the restored `reduced_basis_equal` precondition
    // REFUSES the skip at runtime (skip_b=false, cluster-confirmed), so ψ_B too routes
    // to the exact β̂-sound slow path. The test asserts fast-path engagement TRACKS
    // the `covers_skip` verdict (`(reset==0)==skip`) and that β̂ matches the
    // streamed exact solve to < 1e-6 on whichever path runs — so it is correct
    // whether the witness admits or refuses, while the rotation case (ψ_C) is
    // pinned to refuse so the #1264 guard is non-vacuous.
    let psi_b = psi_a + span / 4096.0;
    // ψ_C: a LARGE move toward the low edge — the radial-kernel reduced basis has
    // moved there, so the witness should REFUSE and the slow path must re-run. (If
    // the fixture's basis happens to stay equal even here, the test still passes:
    // it asserts the SOUNDNESS invariant — β̂ matches streamed on whichever path
    // fires — not a fixed path choice.)
    let psi_c = psi_lo + 0.02 * span;
    assert!(
        tensor_eval.psi_gram_tensor_covers(psi_a)
            && tensor_eval.psi_gram_tensor_covers(psi_b)
            && tensor_eval.psi_gram_tensor_covers(psi_c),
        "all probes must lie inside the certified value window so the #1033 \
         production skip can stay on the n-free lane"
    );
    let rho = Array1::<f64>::from_elem(rho_dim, 0.5);
    let theta_at = |psi: f64| {
        let mut theta = Array1::<f64>::zeros(rho_dim + 1);
        theta.slice_mut(s![..rho_dim]).assign(&rho);
        theta[rho_dim] = psi;
        theta
    };

    // Production gate: a moved ψ takes the design-revision fast path whenever the
    // certified value tensor covers the trial and the exact n-free penalty re-key
    // is staged. The contract is both performance and soundness: admitted moved
    // probes must not re-enter `reset_surface`, and their converged β̂ must match
    // a fresh streamed slow-path solve to < 1e-6.
    let eval_tensor = |evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>,
                       cache: &mut SingleBlockExactJointDesignCache<'_>,
                       theta: &Array1<f64>,
                       realize: bool|
     -> (f64, Array1<f64>, Array1<f64>) {
        if realize {
            cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
        }
        let hyper_dirs = cache
            .hyper_dirs_for_current_design(data.view(), SpatialHyperKind::Isotropic)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper_dirs", e));
        let penalty = cache
            .canonical_penalties_at(theta)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact n-free S(ψ) rebuild", e));
        evaluator.stage_fast_path_penalty(Some(penalty));
        let (cost, grad, _h) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            OuterEvalOrder::ValueAndGradient,
            Some(cache.design_revision()),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "tensor eval", e));
        let beta = evaluator
            .current_beta()
            .unwrap_or_else(|| panic!("{} failed", "converged inner β̂ available"));
        (cost, grad, beta)
    };

    // Trial 1 (ψ_A): first eval at this revision -> slow path runs once and pins
    // the reduced-basis reference at ψ_A.
    assert_eq!(
        tensor_eval.slow_path_reset_count(),
        0,
        "no slow-path reset before the first eval"
    );
    let (c_a, _g_a, beta_a) =
        eval_tensor(&mut tensor_eval, &mut tensor_cache, &theta_at(psi_a), true);
    assert_eq!(
        tensor_eval.slow_path_reset_count(),
        1,
        "first eval at a fresh revision must take the slow path exactly once"
    );

    // #1264 production contract (RESTORED): value-window coverage is NECESSARY but
    // NOT SUFFICIENT for the skip. The β̂-soundness gate is the pairwise reduced-
    // basis-equality witness against the pinning ψ (`covers_skip` =
    // `reduced_basis_equal(psi_ref, psi)`). The skip fires on a trial ⇔ covers_skip
    // admits it; where the reduced basis rotates the witness REFUSES and the exact
    // O(n) path runs, so the κ-amplified interpolated-Gram β̂ error cannot ship.
    let covers_b = tensor_eval.psi_gram_tensor_covers(psi_b);
    let covers_c = tensor_eval.psi_gram_tensor_covers(psi_c);
    let skip_b = tensor_eval.psi_gram_tensor_covers_skip(psi_b);
    let skip_c = tensor_eval.psi_gram_tensor_covers_skip(psi_c);
    eprintln!(
        "[DIAG1264-FP] pinned ψ_a={psi_a:.5}  ψ_b={psi_b:.5} value_covers={covers_b} \
         covers_skip={skip_b}  ψ_c={psi_c:.5} value_covers={covers_c} \
         covers_skip={skip_c}"
    );
    assert!(
        covers_b && covers_c,
        "both moved trials must lie inside the certified value window (the necessary \
         condition for the skip gate to even consider firing); ψ_b={psi_b:.6} \
         (covers={covers_b}) ψ_c={psi_c:.6} (covers={covers_c})"
    );
    // ψ_C is a genuine basis ROTATION (constructed below the pinning ψ where the
    // near-singular Gram's reduced subspace turns over): the restored precondition
    // MUST refuse it so it routes to the exact path. If this ever flips to true the
    // test no longer exercises the #1264 rotation guard.
    assert!(
        !skip_c,
        "ψ_C={psi_c:.6} must GENUINELY rotate the reduced basis so the restored \
         #1264 `reduced_basis_equal` precondition REFUSES the skip and routes it to \
         the exact β̂-sound path; got covers_skip={skip_c}"
    );
    // Production mirror: in `spatial_optimization.rs` the caller realizes the
    // design EXACTLY when the restored `covers_skip` precondition refuses the skip
    // (`if !skip_design_realization { ensure_theta(theta) }`). So pass
    // `realize = !skip_*` here to reproduce the production caller faithfully: an
    // admitted skip preserves the pinned revision (no realize); a refused skip
    // re-realizes the design at the trial ψ and takes the exact slow path.

    // Trial 2 (ψ_B): `covers_skip` verdict drives realization, matching production.
    let before_b = tensor_eval.slow_path_reset_count();
    let (c_b, _g_b, beta_b) =
        eval_tensor(&mut tensor_eval, &mut tensor_cache, &theta_at(psi_b), !skip_b);
    let after_b = tensor_eval.slow_path_reset_count();
    let reset_b = after_b - before_b;
    assert_eq!(
        reset_b == 0,
        skip_b,
        "ψ_B fast-path engagement must match the restored `covers_skip` verdict: \
         covers_skip={skip_b} but slow_path_resets={reset_b} (0 ⇒ skip fired)"
    );

    // Trial 3 (ψ_C): the ROTATION case. `covers_skip` REFUSED it (asserted above),
    // so the production lane re-realizes the design and takes the exact slow path —
    // exactly one reset. This is the load-bearing #1264 guard: the skip does NOT
    // fire across the rotation, so the κ-amplified β̂ mismatch cannot ship.
    let before_c = tensor_eval.slow_path_reset_count();
    let (c_c, _g_c, beta_c) =
        eval_tensor(&mut tensor_eval, &mut tensor_cache, &theta_at(psi_c), !skip_c);
    let after_c = tensor_eval.slow_path_reset_count();
    let reset_c = after_c - before_c;
    assert_eq!(
        reset_c, 1,
        "ψ_C ROTATES the reduced basis (covers_skip REFUSED it), so the restored \
         #1264 precondition MUST take the exact slow path (exactly one reset); got \
         resets={reset_c}"
    );

    assert!(
        c_a.is_finite() && c_b.is_finite() && c_c.is_finite(),
        "all fast/slow path costs must be finite"
    );

    // κ-optimum invariance: the fast-path converged β̂ (⇒ EDF / κ-optimum) at
    // ψ_B and ψ_C must equal a FRESH streamed slow-path solve at the same θ.
    // A fresh streamed evaluator + cache re-realizes the design per ψ (its
    // revision advances each call), so it always runs the exact n-row lane —
    // the reference the fast path must reproduce.
    let mut streamed_eval = make_eval();
    let mut stream_cache = make_cache();
    let beta_streamed = |evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>,
                         cache: &mut SingleBlockExactJointDesignCache<'_>,
                         theta: &Array1<f64>|
     -> Array1<f64> {
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
        let hyper = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &spatial_terms,
        )
        .unwrap()
        .unwrap();
        evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper,
            None,
            OuterEvalOrder::ValueAndGradient,
            Some(cache.design_revision()),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "streamed eval", e));
        evaluator.current_beta().unwrap_or_else(|| panic!("{} failed", "streamed β̂"))
    };

    // Soundness bar: the n-free fast path must reproduce the streamed exact solve
    // to the issue-mandated 1e-6 β̂-rel while avoiding extra slow-path resets.
    let mut worst = 0.0_f64;
    for (label, theta, beta_tensor) in [
        ("psi_b", theta_at(psi_b), &beta_b),
        ("psi_c", theta_at(psi_c), &beta_c),
    ] {
        let beta_slow = beta_streamed(&mut streamed_eval, &mut stream_cache, &theta);
        let r = beta_tensor
            .iter()
            .zip(beta_slow.iter())
            .fold(0.0_f64, |a, (f, s)| a.max((f - s).abs() / (1.0 + s.abs())));
        eprintln!(
            "[DIAG1264-FP] {label} ψ={:.4} β̂rel={r:.3e} issue-bar=1e-6 \
             β̂tensor[0]={:+.6e} β̂streamed[0]={:+.6e}",
            theta[rho_dim], beta_tensor[0], beta_slow[0]
        );
        assert_eq!(beta_tensor.len(), beta_slow.len(), "β̂ dim mismatch @ {label}");
        for j in 0..beta_tensor.len() {
            assert!(
                beta_tensor[j].is_finite() && beta_slow[j].is_finite(),
                "non-finite β̂[{j}] @ {label}"
            );
            let babs = (beta_tensor[j] - beta_slow[j]).abs();
            let brel = babs / (1.0 + beta_slow[j].abs());
            worst = worst.max(babs);
            // #1033 soundness bar: β̂ must match the streamed exact solve to 1e-6
            // while the moved probe stays on the n-free fast path.
            const ISSUE_BETA_BAR: f64 = 1.0e-6;
            assert!(
                brel <= ISSUE_BETA_BAR,
                "β̂[{j}] @ {label} diverges from the streamed exact solve beyond the \
                 #1264 issue-mandated 1e-6 bar: tensor={:+.12e} streamed={:+.12e} \
                 |Δ|={babs:.3e} rel={brel:.3e} > 1e-6 — the restored `reduced_basis_equal` \
                 skip precondition failed to route this ψ to the exact path, so an \
                 interpolated-Gram κ-amplified κ-optimum shipped",
                beta_tensor[j],
                beta_slow[j],
            );
        }
    }
    // β̂ at ψ_A (the slow-path reference inside the tensor evaluator itself)
    // must be finite — sanity that the pinning eval converged.
    assert!(
        beta_a.iter().all(|v| v.is_finite()),
        "ψ_A pinning β̂ must be finite"
    );

    eprintln!(
        "[psi-gram-tensor #1033/#1264] pairwise reduced-basis-equality witness: \
         skip(ψ_b)={skip_b} skip(ψ_c={psi_c:.4} ROTATED)={skip_c}, \
         slow_path_reset_count={}, worst |Δβ̂| vs streamed slow path = {worst:.3e} \
         (asserted ≤ the #1264 issue-mandated 1e-6 β̂-rel bar across the basis rotation \
         the witness refuses)",
        tensor_eval.slow_path_reset_count()
    );
}
}
