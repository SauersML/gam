// End-to-end finite-difference oracles for the isotropic-κ (log-κ) joint REML
// outer gradient on real Duchon / Matérn spatial smooths — the #901 gate.
//
// These tests are the headline reproduction #901 was filed against: they
// differentiate the *production* joint REML cost (`evaluate_cost_only`) by
// central finite differences in ρ=log λ and ψ=log κ, and compare it against
// the analytic outer gradient the κ-optimizer actually follows
// (`evaluate_joint_reml_outer_eval_at_theta`). The bug class #901 tracks — the
// range(Sλ)-projected logdet dropping both the penalty-null Schur curvature
// (ρ sign flips) and the moving-subspace dU_S/dψ term (~1e5 ψ blow-ups) — is a
// REML-criterion gradient error that ONLY surfaces on a non-Gaussian GLM
// spatial smooth with a genuinely rank-deficient penalty subspace, which the
// synthetic-matrix unit tests in `gam-custom-family` cannot exercise.
//
// The fixture set was authored in the pre-#1521 monolith under
// `tests/src_modules/smooths/smooth_design_assembly_constraint_tests.rs`. The
// #1521 crate carve moved its private dependencies
// (`SingleBlockExactJointDesignCache`, `try_build_spatial_log_kappa_hyper_dirs`,
// `evaluate_joint_reml_outer_eval_at_theta`, `external_opts_for_design`,
// `spatial_dims_per_term`, `spatial_length_scale_term_indices`,
// `try_build_spatial_term_log_kappa_derivative`) DOWN into the gam-models
// `fit_orchestration::drivers` module, but #1601 commented the test `include!`
// out of `gam_terms::smooth::tests` "for relocation" and the relocation never
// happened — so the #901 gate compiled into NO binary. It is re-homed here,
// where every private driver symbol is in scope via `use super::*` (the
// `design_construction.rs` + `spatial_optimization.rs` files are `include!`d
// flat into one module namespace), and the only cross-crate paths
// (`crate::estimate::ExternalJointHyperEvaluator`,
// `crate::solver::rho_optimizer::OuterEvalOrder`) are rewritten to their carved
// homes `gam_solve::estimate` / `gam_solve::rho_optimizer`.
#[cfg(test)]
mod iso_kappa_reml_gradient_fd_tests {
    use super::*;
    use super::test_support::SingleBlockExactJointDesignCacheTestExt;
    use gam_terms::basis::{
        DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec, MaternNu,
        OneDimensionalBoundary, SpatialIdentifiability,
    };
    use ndarray::{Array1, Array2, s};

#[test]
fn iso_kappa_duchon_binomial_probit_joint_gradient_matches_finite_difference() {
    let (pass, worst, violations) = iso_kappa_fd_variant_driver(
        "duchon_probit_n80",
        80,
        LikelihoodSpec::binomial_probit(),
        false,
        false,
    );
    assert!(
        pass,
        "Duchon BinomialProbit n=80 FD failed; worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// Shared driver for iso-κ joint REML gradient FD variants. Returns the
/// worst psi rel_err across the four theta probes (zero / psi_only / base /
/// alt) and panics with full violations only if `assert_pass` is true.
/// Knobs let one-at-a-time variants of the original BinomialProbit Duchon
/// failure isolate which dimension triggers the analytic-vs-FD blow-up.
///
/// `well_conditioned` selects a label set that keeps the inner probit fit well
/// inside the smooth IRLS regime (μ near ½, max|η| small). This matters at
/// small n: the analytic ψ=log κ outer gradient is mathematically exact (the
/// #901 intrinsic-pseudo-logdet kernel), but its GLM cubic-curvature trace term
/// `tr(H_pen⁺ · Xᵀdiag(c⊙X_ψβ̂)X)` is the near-cancellation of two O(10³) halves,
/// so it amplifies the inner PIRLS stationarity floor (‖g‖≈2e-6, the LM-ridge
/// noise floor on near-separable binary data) by ~1.5e3 ≈ 3e-3. Under genuine
/// near-separation (max|η|≈8.8 at n=20 with the original boundary-split labels)
/// BOTH the analytic gradient and the FD oracle inherit that floor on the
/// converged β̂, and their independent ~2e-6 errors blow up to ~1e-2 in the
/// amplified cubic — the FD comparison is then ill-posed, not the gradient.
/// A balanced label set keeps the cancellation halves O(1) so the oracle
/// verifies the *gradient formula* rather than the *conditioning floor*. Proof:
/// the same n=20 Duchon-probit config matches FD to 6e-7 under balanced labels
/// vs 8e-3 under the separated labels (and ρ matches to 1e-5 in both).
fn iso_kappa_fd_variant_driver(
    label: &str,
    n: usize,
    family: LikelihoodSpec,
    skip_psi: bool,
    well_conditioned: bool,
) -> (bool, f64, Vec<String>) {
    // A `"*_2d"` label builds an ordinary 2-D feature cloud (the production
    // `matern(x1, x2)` regime: operator triplet {mass, tension, stiffness}, with
    // the per-axis tension and mixed-curvature stiffness blocks that only carry
    // cross-axis structure when d ≥ 2). This is the fast unit-level reproduction
    // of the #1122 stall, whose end-to-end pin is
    // `matern_2d_iso_kappa_outer_gradient_matches_fd`.
    let two_d = label.ends_with("_2d");
    let d = if two_d { 2 } else { 1 };
    let mut data = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        let eta = if two_d {
            // A low-discrepancy second axis (golden-ratio fill) keeps the 2-D
            // cloud well-spread, and a genuinely 2-D truth exercises both the
            // signal and the cross-axis curvature blocks.
            let t2 = (i as f64 * 0.618_033_988_749_894_9).fract();
            data[[i, 1]] = t2;
            1.4 * (2.0 * std::f64::consts::PI * t).sin()
                + 0.9 * (2.0 * std::f64::consts::PI * t2).cos()
                + 0.5 * (t - 0.5)
        } else {
            1.4 * (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (t - 0.5)
        };
        let raw = eta + 0.7 * (3.7 * (i as f64) + 1.0).sin();
        y[i] = if family.is_gaussian_identity() {
            raw
        } else if well_conditioned {
            // Smooth, non-separating Bernoulli labels: a deterministic
            // logistic-probability threshold against a fixed phase grid keeps
            // the fitted μ away from {0,1} so the inner Newton system — and the
            // cubic-curvature ψ-trace built from it — stays well-conditioned.
            let p = 1.0 / (1.0 + (-0.6 * (2.0 * std::f64::consts::PI * t).sin()).exp());
            let u = 0.5 * ((5.0 * (i as f64) + 0.5).sin() + 1.0);
            if u < p { 1.0 } else { 0.0 }
        } else if raw > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    // Duchon is the historical iso-κ FD probe basis; a `"matern_*"` label
    // routes the Matérn ν=5/2 kernel instead so the same gold-standard
    // analytic-vs-FD outer-gradient check covers the Matérn iso-κ REML
    // gradient assembly (which has no other end-to-end FD pin). Thin-plate
    // is deliberately excluded from κ-axis enrollment (see
    // `spatial_term_supports_hyper_optimization`).
    let basis = if label.starts_with("matern") {
        SmoothBasisSpec::Matern {
            feature_cols: (0..d).collect(),
            spec: MaternBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                periodic: None,
                length_scale: 1.0,
                nu: MaternNu::FiveHalves,
                include_intercept: false,
                // The realized Matérn design ALWAYS carries the operator triplet
                // ({mass, tension, stiffness}, see
                // `matern_operator_penalty_triplet_at_length_scale`); the
                // `double_penalty` flag selects the COLD-build value-path penalty
                // but the κ-optimizer re-keys onto the operator triplet either
                // way. A `"*_dp"` label keeps the production default
                // `double_penalty: true` to mirror `matern(x1, x2)` exactly.
                double_penalty: label.contains("_dp"),
                identifiability: MaternIdentifiability::CenterSumToZero,
                aniso_log_scales: None,
                nullspace_shrinkage_survived: None,
            },
            input_scales: None,
        }
    } else {
        SmoothBasisSpec::Duchon {
            feature_cols: vec![0],
            spec: DuchonBasisSpec {
                radial_reparam: None,
                periodic: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                length_scale: Some(1.0),
                power: 1.0,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                boundary: OneDimensionalBoundary::Open,
            },
            input_scales: None,
        }
    };
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "variant_1d".to_string(),
            basis,
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    // Isotropic κ: one log-κ axis regardless of feature dimension `d` (the 2-D
    // cloud still enrolls a single isotropic κ, not a per-axis η).
    assert_eq!(dims_per_term, vec![1], "{label}: expect one log-κ axis");
    let rho_dim = frozen_design.penalties.len();
    let psi_dim: usize = dims_per_term.iter().sum();
    assert!(psi_dim >= 1);
    eprintln!(
        "[{label} TOPOLOGY] d={d} rho_dim={rho_dim} psi_dim={psi_dim} \
         penalty_sources={:?}",
        frozen_design
            .penalties
            .iter()
            .map(|p| p.col_range.clone())
            .collect::<Vec<_>>()
    );

    let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "single-block cache", e));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "iso-κ variant FD evaluator",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e));

    let cost_at = |theta: &Array1<f64>,
                   cache: &mut SingleBlockExactJointDesignCache<'_>,
                   evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> f64 {
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
        let design = cache.design();
        evaluator
            .evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                theta,
                rho_dim,
                None,
                "iso-κ variant FD cost-only",
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "cost-only eval", e))
    };

    let analytic_at = |theta: &Array1<f64>,
                       cache: &mut SingleBlockExactJointDesignCache<'_>,
                       evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> (f64, Array1<f64>) {
        cache.ensure_theta(theta).expect("ensure_theta");
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper dirs build", e))
        .expect("hyper dirs present");
        let (cost, grad, _hess) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "outer eval", e));
        (cost, grad)
    };

    let theta_dim = rho_dim + psi_dim;
    let theta_zero = Array1::<f64>::zeros(theta_dim);
    let mut theta_base = Array1::<f64>::zeros(theta_dim);
    for j in 0..rho_dim {
        theta_base[j] = 0.2 - 0.1 * j as f64;
    }
    let mut theta_psi_only = Array1::<f64>::zeros(theta_dim);
    for k in 0..psi_dim {
        theta_psi_only[rho_dim + k] = 0.4;
    }
    let mut theta_alt = theta_base.clone();
    for j in 0..rho_dim {
        theta_alt[j] = 1.0 + 0.05 * j as f64;
    }
    for k in 0..psi_dim {
        theta_alt[rho_dim + k] = 0.4;
    }

    let h = 1e-5_f64;
    let rel_tol = 5e-3_f64;
    let mut violations: Vec<String> = Vec::new();
    let mut worst_psi_rel = 0.0_f64;
    for (probe, theta) in [
        ("zero", &theta_zero),
        ("psi_only", &theta_psi_only),
        ("base", &theta_base),
        ("alt", &theta_alt),
    ] {
        let (cost_an, grad_an) = analytic_at(theta, &mut cache, &mut evaluator);
        assert!(cost_an.is_finite(), "{label} {probe}: cost not finite");
        // Objective↔gradient desync probe: the analytic gradient path
        // (evaluate_joint_reml_outer_eval_at_theta) and the cost-only FD
        // path (evaluate_cost_only) must agree on the COST itself at the
        // unperturbed θ. If they disagree, FD differences a different
        // function than the gradient differentiates and no gradient fix
        // can make them match. eprintln for the diagnostic build only.
        let cost_via_fd_path = cost_at(theta, &mut cache, &mut evaluator);
        eprintln!(
            "[{label} {probe}] COST an={:+.10e} fd_path={:+.10e} diff={:.3e}",
            cost_an,
            cost_via_fd_path,
            (cost_an - cost_via_fd_path).abs()
        );
        for j in 0..theta_dim {
            let is_psi = j >= rho_dim;
            if skip_psi && is_psi {
                continue;
            }
            let mut plus = theta.clone();
            plus[j] += h;
            let mut minus = theta.clone();
            minus[j] -= h;
            let cp = cost_at(&plus, &mut cache, &mut evaluator);
            let cm = cost_at(&minus, &mut cache, &mut evaluator);
            let fd = (cp - cm) / (2.0 * h);
            let denom = fd.abs().max(grad_an[j].abs()).max(1e-3);
            let rel = (grad_an[j] - fd).abs() / denom;
            let kind = if is_psi { "psi" } else { "rho" };
            eprintln!(
                "[{label} {probe}] {kind} j={j} an={:+.4e} fd={:+.4e} rel={:.3e}",
                grad_an[j], fd, rel
            );
            if is_psi && rel > worst_psi_rel {
                worst_psi_rel = rel;
            }
            if rel >= rel_tol {
                violations.push(format!(
                    "{probe} {kind} j={j}: analytic={:+.6e} fd={:+.6e} rel={:.3e}",
                    grad_an[j], fd, rel
                ));
            }
        }
    }
    let pass = violations.is_empty();
    eprintln!(
        "[{label} SUMMARY] pass={pass} worst_psi_rel={worst_psi_rel:.3e} \
             violations={}",
        violations.len()
    );
    (pass, worst_psi_rel, violations)
}

#[test]
fn iso_kappa_duchon_gaussian_identity_fd() {
    let (pass, worst, violations) = iso_kappa_fd_variant_driver(
        "duchon_gaussian",
        80,
        LikelihoodSpec::gaussian_identity(),
        false,
        false,
    );
    assert!(
        pass,
        "Gaussian Identity FD failed; worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// The Matérn ν=5/2 analogue of `iso_kappa_duchon_gaussian_identity_fd`.
///
/// The isotropic-analytic κ optimizer was observed to stall at n≳1000 on a
/// well-conditioned 1-D Matérn Gaussian fit (grad_norm ≈ 0.5·|f|, nowhere
/// near stationary) while the Duchon path converges — and the Matérn iso-κ
/// *outer* REML gradient had no end-to-end FD pin (only basis-level log-κ
/// derivative tests). This closes that gap: it differences the same exact
/// analytic ψ=log κ outer gradient that the optimizer follows against a
/// central finite difference of the REML cost. If the analytic gradient is
/// wrong, the optimizer's stall is explained and this fails loudly.
#[test]
fn iso_kappa_matern_gaussian_identity_fd() {
    let (pass, worst, violations) = iso_kappa_fd_variant_driver(
        "matern_gaussian",
        80,
        LikelihoodSpec::gaussian_identity(),
        false,
        false,
    );
    assert!(
        pass,
        "Matérn iso-κ Gaussian-identity outer-gradient FD failed; \
             worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}
/// Fast unit-level reproduction of the #1122 stall: an ordinary 2-D
/// `matern(x1, x2)` Gaussian fit whose isotropic-κ outer REML gradient must
/// match a central finite difference of the production REML cost. This is the
/// d=2 analogue of `iso_kappa_matern_gaussian_identity_fd`: the 1-D Matérn
/// already matched FD, so the desync that stalled the κ-optimizer at its
/// iteration cap (analytic ≠ FD on `psi_kappa`, #1122) lives in the cross-axis
/// tension / mixed-curvature stiffness operator blocks that only carry
/// off-diagonal structure when d ≥ 2.
#[test]
fn iso_kappa_matern_2d_gaussian_identity_fd() {
    let (pass, worst, violations) = iso_kappa_fd_variant_driver(
        "matern_gaussian_2d",
        120,
        LikelihoodSpec::gaussian_identity(),
        false,
        false,
    );
    assert!(
        pass,
        "Matérn 2-D iso-κ Gaussian-identity outer-gradient FD failed (the #1122 \
             cross-axis operator-penalty ψ-derivative desync); worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// DIAGNOSTIC (#1122): is the residual ~1.6e-3 relative gap in the production
/// end-to-end audit (`matern_2d_iso_kappa_outer_gradient_matches_fd`,
/// analytic=16.11 vs fd=16.08 at the small auto-init length scale) a genuine
/// missing derivative term, or a finite-difference truncation artifact of the
/// steep κ^{2m} operator penalty at the production-init `log κ ≈ 2.5`?
///
/// This sweeps the central-FD step `h` on the ψ=log κ axis at a high-κ θ that
/// mirrors the production init (the operator triplet scales like κ^{2m} with
/// m = ν + d/2 = 3.5, so V(ψ) has a large third derivative and the central-FD
/// truncation error ∝ h²·V''' dominates). A TRUNCATION artifact shrinks ≈ 100×
/// per 10× shrink in `h` (until the roundoff floor); a REAL derivative bug
/// leaves an `h`-independent floor. The analytic gradient is computed once;
/// only `h` changes. This is a diagnostic oracle (FD is sanctioned in tests).
#[test]
fn iso_kappa_matern_2d_psi_fd_step_sweep_diagnostic() {
    use ndarray::Array2 as NdArray2;
    let n = 150usize;
    let d = 2usize;
    // EXACT mirror of the end-to-end gate's dataset
    // (`matern_2d_iso_kappa_outer_gradient_matches_fd`): uniform-random 2-D
    // cloud via splitmix64 (seed 0x9A7E_7212_0001), truth sin(2πa)·sin(2πb) +
    // N(0, 0.05²). Reproducing the same X(ψ) is the only way the fast harness
    // sees the SAME analytic ψ-gradient (≈ +16.11) and the SAME h-flat gap.
    let mut st: u64 = 0x9A7E_7212_0001;
    fn splitmix(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_unit(s: &mut u64) -> f64 {
        (splitmix(s) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_gauss(s: &mut u64) -> f64 {
        let u1 = next_unit(s).max(1.0e-12);
        let u2 = next_unit(s);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    let mut data = NdArray2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    let sigma = 0.05;
    for i in 0..n {
        let a = next_unit(&mut st);
        let b = next_unit(&mut st);
        data[[i, 0]] = a;
        data[[i, 1]] = b;
        y[i] = (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
            + sigma * next_gauss(&mut st);
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    // CRITICAL (#1122): the FrozenTransform `Z` and the nullspace-shrinkage
    // decision are computed ONCE at the BASE length scale the design is frozen
    // at, then held fixed across the κ-sweep. Production does a pilot ρ-fit /
    // data re-seed BEFORE freezing, so its frozen base ls is 0.28665832
    // (ψ_base = −ln(ls) = 1.2494 = the audit θ₀ ψ), NOT the raw
    // `auto_initial_length_scale` (0.0812 → ψ=2.51). Freezing the harness at the
    // auto-init ls gave a DIFFERENT `Z` (and a different objective: V≈16.31 vs
    // production ≈16.11), which is why the fast harness was internally
    // consistent yet never reproduced the production audit gap. Freeze at the
    // production base ls so the harness `Z` matches production byte-for-byte and
    // the probe ψ = 1.2494 lands AT the freeze point.
    let length_scale = 0.286_658_32_f64;
    eprintln!("[PSI-SWEEP] length_scale={length_scale:.6} log_kappa={:.4}", -length_scale.ln());
    // The production default center count for n=150, d=2 is 37 (see
    // `default_matern_center_count`/`default_num_centers`).
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern_2d".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 37 },
                    periodic: None,
                    length_scale,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                    nullspace_shrinkage_survived: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap();
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap();
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap();
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = frozen_design.penalties.len();
    let psi_dim: usize = dims_per_term.iter().sum();
    let theta_dim = rho_dim + psi_dim;
    let family = LikelihoodSpec::gaussian_identity();
    let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap();
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "psi-sweep FD evaluator",
    )
    .unwrap();

    // TEMP #1122 diagnostic: HARNESS DESIGN FINGERPRINT, to diff against the
    // production `/tmp/gam_prod_fingerprint.txt`. The harness is internally
    // consistent (analytic≈FD to 4.5e-9) but evaluates V≈16.31 while production
    // evaluates V≈16.11 — so the mirror is structurally imperfect. This dumps
    // the SAME fields the production FINGERPRINT block dumps. Removed before
    // merge.
    {
        eprintln!("[FINGERPRINT] HARNESS design.design dims = ({}, {})", frozen_design.design.nrows(), frozen_design.design.ncols());
        eprintln!("[FINGERPRINT] HARNESS n_penalties = {}", frozen_design.penalties.len());
        for (pi, p) in frozen_design.penalties.iter().enumerate() {
            let fro: f64 = p.local.iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!(
                "[FINGERPRINT] HARNESS penalty[{pi}] col_range={:?} local_dims={:?} hint={:?} fro={fro:.10e}",
                p.col_range, p.local.dim(), p.structure_hint
            );
        }
        eprintln!("[FINGERPRINT] HARNESS nullspace_dims = {:?}", frozen_design.nullspace_dims);
        for &ti in spatial_terms.iter() {
            if let Some(t) = frozen.smooth_terms.get(ti) {
                let s = match &t.basis {
                    SmoothBasisSpec::Matern { spec, .. } => format!(
                        "Matern{{nu={:?}, ls={:.8}, dp={}, ident={}, aniso={:?}, centers_kind={}}}",
                        spec.nu,
                        spec.length_scale,
                        spec.double_penalty,
                        match &spec.identifiability {
                            MaternIdentifiability::FrozenTransform { transform, nullspace_shrinkage_survived } =>
                                format!("FrozenTransform{{z_dims={:?}, survived={:?}}}", transform.dim(), nullspace_shrinkage_survived),
                            other => format!("{other:?}"),
                        },
                        spec.aniso_log_scales,
                        match &spec.center_strategy {
                            CenterStrategy::UserProvided(c) => format!("UserProvided(n={})", c.nrows()),
                            other => format!("{other:?}"),
                        },
                    ),
                    other => format!("{other:?}"),
                };
                eprintln!("[FINGERPRINT] HARNESS term[{ti}] basis_kind={s}");
            }
            if let Some(t) = frozen_design.smooth.terms.get(ti) {
                if let gam_terms::basis::BasisMetadata::Matern {
                    centers, input_scales, length_scale, ..
                } = &t.metadata
                {
                    let csum: f64 = centers.iter().map(|v| v.abs()).sum();
                    let c00 = centers.get((0, 0)).copied().unwrap_or(f64::NAN);
                    let c01 = centers.get((0, 1)).copied().unwrap_or(f64::NAN);
                    eprintln!(
                        "[FINGERPRINT] HARNESS meta.Matern length_scale={length_scale:.10} input_scales={input_scales:?} centers_abs_sum={csum:.10e} c[0,0]={c00:.10} c[0,1]={c01:.10}"
                    );
                }
            }
        }
    }

    let cost_at = |theta: &Array1<f64>,
                   cache: &mut SingleBlockExactJointDesignCache<'_>,
                   evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> f64 {
        cache.ensure_theta(theta).unwrap();
        let design = cache.design();
        evaluator
            .evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                theta,
                rho_dim,
                None,
                "psi-sweep cost-only",
                None,
            )
            .unwrap()
    };
    let analytic_at = |theta: &Array1<f64>,
                       cache: &mut SingleBlockExactJointDesignCache<'_>,
                       evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> Array1<f64> {
        cache.ensure_theta(theta).unwrap();
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap()
        .expect("hyper dirs present");
        let (_c, grad, _h) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .unwrap();
        grad
    };

    // θ mirroring the production audit θ₀ captured from the end-to-end gate
    // (`matern_2d_iso_kappa_outer_gradient_matches_fd`): the warm-started ρ is
    // strongly negative (λ ≈ e^{-4..-6}, the penalty is nearly OFF, so the
    // criterion is data-fit + ½log|H+Sλ| dominated) and ψ = log κ ≈ 1.25. This
    // is the regime that exposed the residual ~1.6e-3 gap; the earlier (ρ=0,
    // ψ=2.5) probe did not. rho_dim is 3 in both (operator triplet), so the ρ
    // slots line up; if rho_dim differs we pad with the last value.
    let prod_rho = [-3.632_687_635_594_657, -5.970_607_752_248_795, -4.804_720_434_766_625];
    let prod_psi = 1.249_464_308_750_002_1;
    let mut theta = Array1::<f64>::zeros(theta_dim);
    for j in 0..rho_dim {
        theta[j] = prod_rho.get(j).copied().unwrap_or(*prod_rho.last().unwrap());
    }
    for k in 0..psi_dim {
        theta[rho_dim + k] = prod_psi;
    }
    eprintln!("[PSI-SWEEP] rho_dim={rho_dim} probing theta={:?}", theta.to_vec());
    let grad = analytic_at(&theta, &mut cache, &mut evaluator);
    let psi_idx = rho_dim;
    let analytic = grad[psi_idx];
    eprintln!("[PSI-SWEEP] analytic ∂V/∂ψ (ValueAndGradient) = {analytic:+.8e}");
    {
        // TEMP #1122: value + data checksums at θ₀, to diff vs production. The
        // designs now fingerprint-match, but the harness ∂V/∂ψ (≈41) ≠ production
        // (≈16). Same X/penalty/θ → the COST itself differs: isolate whether it
        // is the data (y/weights/offset) or the evaluator options.
        let v0 = cost_at(&theta, &mut cache, &mut evaluator);
        let y_abs_sum: f64 = y.iter().map(|v| v.abs()).sum();
        let y0 = y.get(0).copied().unwrap_or(f64::NAN);
        let xd = cache.design().design.to_dense();
        let x_abs_sum: f64 = xd.iter().map(|v| v.abs()).sum();
        let x00 = xd.get((0, 0)).copied().unwrap_or(f64::NAN);
        let x01 = xd.get((0, 1)).copied().unwrap_or(f64::NAN);
        let x_row0_sum: f64 = xd.row(0).iter().map(|v| v.abs()).sum();
        eprintln!(
            "[FINGERPRINT] HARNESS V(theta0)={v0:.10e} y_abs_sum={y_abs_sum:.10e} y[0]={y0:.10} X_abs_sum={x_abs_sum:.10e} X[0,0]={x00:.10} X[0,1]={x01:.10} X_row0_abs={x_row0_sum:.10e} dims=({},{}) n={n}",
            xd.nrows(), xd.ncols()
        );
    }
    // The PRODUCTION audit takes its analytic gradient from a
    // ValueGradientHessian eval, NOT ValueAndGradient. If the ψ-gradient
    // returned by the two orders differs, the audit differences the value path
    // against a gradient computed in a different lane → an objective↔gradient
    // desync that no FD step can close. Probe both at the SAME θ₀.
    {
        cache.ensure_theta(&theta).unwrap();
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap()
        .expect("hyper dirs present");
        let (_c, grad_vgh, _h) = evaluate_joint_reml_outer_eval_at_theta(
            &mut evaluator,
            cache.design(),
            &theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueGradientHessian,
            None,
        )
        .unwrap();
        eprintln!(
            "[PSI-SWEEP] analytic ∂V/∂ψ (ValueGradientHessian) = {:+.8e} (delta vs V&G = {:.3e})",
            grad_vgh[psi_idx],
            (grad_vgh[psi_idx] - analytic).abs()
        );
    }
    // VALUE oracle #2: the production audit differences `eval_full(Value)` →
    // `evaluate_joint_reml_outer_eval_at_theta(.., Value)`, NOT
    // `evaluate_cost_only`. If THIS value path disagrees with the gradient while
    // `evaluate_cost_only` agrees, the desync is between the two value lanes.
    let value_via_outer_eval = |theta: &Array1<f64>,
                                cache: &mut SingleBlockExactJointDesignCache<'_>,
                                evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> f64 {
        cache.ensure_theta(theta).unwrap();
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap()
        .expect("hyper dirs present");
        let (c, _g, _h) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::Value,
            None,
        )
        .unwrap();
        c
    };

    let mut prev_gap: Option<f64> = None;
    let mut min_gap = f64::INFINITY;
    for &h in &[1e-2_f64, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] {
        let mut plus = theta.clone();
        plus[psi_idx] += h;
        let mut minus = theta.clone();
        minus[psi_idx] -= h;
        let cp = cost_at(&plus, &mut cache, &mut evaluator);
        let cm = cost_at(&minus, &mut cache, &mut evaluator);
        let fd = (cp - cm) / (2.0 * h);
        let gap = (analytic - fd).abs();
        // Second FD using the outer-eval Value lane (the production audit's lane).
        let cp2 = value_via_outer_eval(&plus, &mut cache, &mut evaluator);
        let cm2 = value_via_outer_eval(&minus, &mut cache, &mut evaluator);
        let fd2 = (cp2 - cm2) / (2.0 * h);
        let gap2 = (analytic - fd2).abs();
        min_gap = min_gap.min(gap);
        let shrink = prev_gap.map(|p| p / gap).unwrap_or(f64::NAN);
        eprintln!(
            "[PSI-SWEEP] h={h:.0e} fd_costonly={fd:+.8e} gap={gap:.3e} | fd_outereval={fd2:+.8e} gap2={gap2:.3e} shrink={shrink:.2}"
        );
        prev_gap = Some(gap);
    }
    eprintln!("[PSI-SWEEP] min_gap_over_sweep={min_gap:.3e} analytic={analytic:.8e} (truncation→shrinks ~100×/decade; real bug→h-flat floor)");
    // The gradient is correct iff some step drives the gap well below the
    // audit's 1e-3·|fd| DESYNC band — i.e. the residual is FD truncation, not a
    // missing derivative term. A real derivative bug would floor the gap
    // regardless of `h`.
    assert!(
        min_gap < 5e-3 * analytic.abs().max(1.0),
        "ψ=log κ outer gradient never matches FD across the h-sweep \
         (min_gap={min_gap:.3e}, analytic={analytic:.6e}): this is a REAL \
         derivative bug, not FD truncation"
    );
}

/// The production-default (`double_penalty: true`) 2-D Matérn variant. This is
/// the closest unit-level mirror of `matern(x1, x2)` and isolates whether the
/// #1122 stall is driven by the double-penalty value-path / re-key topology.
#[test]
fn iso_kappa_matern_2d_dp_gaussian_identity_fd() {
    let (pass, worst, violations) = iso_kappa_fd_variant_driver(
        "matern_gaussian_2d_dp",
        120,
        LikelihoodSpec::gaussian_identity(),
        false,
        false,
    );
    assert!(
        pass,
        "Matérn 2-D double-penalty iso-κ Gaussian-identity outer-gradient FD \
             failed (the #1122 stall); worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn iso_kappa_duchon_binomial_logit_fd() {
    let (pass, worst, violations) =
        iso_kappa_fd_variant_driver("duchon_logit", 80, LikelihoodSpec::binomial_logit(), false, false);
    assert!(
        pass,
        "BinomialLogit FD failed; worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

// No `iso_kappa_thinplate_*_fd` companion to the Duchon FD tests above:
// thin-plate is deliberately excluded from the spatial κ-axis enrollment
// by `spatial_term_supports_hyper_optimization` (a scalar TPS κ creates
// the flat ρ/κ valleys tracked in #718 / #721 / #731 / #732), so there
// is no analytic κ-gradient on which an FD comparison could land.

#[test]
fn iso_kappa_duchon_n_smaller_fd() {
    let (pass, worst, violations) = iso_kappa_fd_variant_driver(
        "duchon_probit_n20",
        20,
        LikelihoodSpec::binomial_probit(),
        false,
        // Well-conditioned labels: at n=20 the separated label set drives the
        // inner probit fit to max|η|≈8.8, where the GLM cubic-curvature ψ-trace
        // amplifies the inner PIRLS KKT floor (~2e-6) by ~1.5e3 into a ~1e-2
        // analytic-vs-FD gap that is a conditioning artifact of BOTH sides, not
        // a gradient error (#901 kernel is exact: balanced labels match to 6e-7).
        true,
    );
    assert!(
        pass,
        "Duchon Probit n=20 FD failed; worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn iso_kappa_duchon_no_psi_fd() {
    let (pass, _worst, violations) = iso_kappa_fd_variant_driver(
        "duchon_probit_rho_only",
        80,
        LikelihoodSpec::binomial_probit(),
        true,
        false,
    );
    assert!(
        pass,
        "Duchon Probit ρ-only FD failed:\n  {}",
        violations.join("\n  ")
    );
}

/// Owned 1-D Duchon BinomialProbit setup shared verbatim across the
/// `duchon_probit_*` mechanism pins. Holds only non-self-referential
/// owners; each test constructs its own `external_opts` / cache /
/// evaluator inline (the borrow-entangled, per-test-labelled parts).
struct DuchonProbitSetup {
    data: Array2<f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    frozen: TermCollectionSpec,
    frozen_design: TermCollectionDesign,
    spatial_terms: Vec<usize>,
    dims_per_term: Vec<usize>,
    rho_dim: usize,
    psi_dim: usize,
}

/// Builds the verbatim 1-D Duchon BinomialProbit data + frozen design used
/// by the ψ-trace / per-row / PIRLS-determinism mechanism pins.
fn build_duchon_probit_setup() -> DuchonProbitSetup {
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        let eta = 1.4 * (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (t - 0.5);
        y[i] = if eta + 0.7 * (3.7 * (i as f64) + 1.0).sin() > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
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
    let psi_dim: usize = dims_per_term.iter().sum();
    DuchonProbitSetup {
        data,
        y,
        weights,
        offset,
        frozen,
        frozen_design,
        spatial_terms,
        dims_per_term,
        rho_dim,
        psi_dim,
    }
}

/// Behavioral pin for the iso-κ Duchon ψ-axis under BinomialProbit: the
/// analytic outer gradient must agree with a centered finite difference of the
/// production objective.
#[test]
fn iso_kappa_duchon_outer_gradient_matches_centered_fd() {
    let DuchonProbitSetup {
        data,
        y,
        weights,
        offset,
        frozen,
        frozen_design,
        spatial_terms,
        dims_per_term,
        rho_dim,
        psi_dim,
    } = build_duchon_probit_setup();
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    };

    let external_opts = external_opts_for_design(
        &LikelihoodSpec::binomial_probit(),
        &frozen_design,
        &fit_opts,
    );
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "cache", e));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "iso-kappa Duchon gradient FD pin",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e));

    let theta_dim = rho_dim + psi_dim;
    let theta_zero = Array1::<f64>::zeros(theta_dim);

    let eval_at =
        |theta: &Array1<f64>,
         order: gam_solve::rho_optimizer::OuterEvalOrder,
         cache: &mut SingleBlockExactJointDesignCache<'_>,
         evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>| {
            cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
            let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
                data.view(),
                cache.spec(),
                cache.design(),
                &cache.spatial_terms,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper dirs build", e))
            .expect("hyper dirs present");
            evaluate_joint_reml_outer_eval_at_theta(
                evaluator,
                cache.design(),
                theta,
                rho_dim,
                hyper_dirs,
                None,
                order,
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "outer eval", e))
        };

    let (cost_at_zero, grad_at_zero, _hess) = eval_at(
        &theta_zero,
        gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
        &mut cache,
        &mut evaluator,
    );

    let h = 1e-5_f64;
    let psi_idx = rho_dim;
    let mut theta_p = theta_zero.clone();
    theta_p[psi_idx] += h;
    let mut theta_m = theta_zero.clone();
    theta_m[psi_idx] -= h;
    let (cost_p, _, _) = eval_at(
        &theta_p,
        gam_solve::rho_optimizer::OuterEvalOrder::Value,
        &mut cache,
        &mut evaluator,
    );
    let (cost_m, _, _) = eval_at(
        &theta_m,
        gam_solve::rho_optimizer::OuterEvalOrder::Value,
        &mut cache,
        &mut evaluator,
    );
    let fd_psi_gradient = (cost_p - cost_m) / (2.0 * h);
    let analytic_psi_gradient = grad_at_zero[psi_idx];
    let scale = 1.0 + analytic_psi_gradient.abs().max(fd_psi_gradient.abs());
    let rel = (analytic_psi_gradient - fd_psi_gradient).abs() / scale;
    assert!(
        rel < 1e-3,
        "Duchon ψ outer gradient must match centered FD of the production objective: \
             analytic={:+.4e}, fd={:+.4e}, rel={:+.3e}",
        analytic_psi_gradient,
        fd_psi_gradient,
        rel
    );

    assert!(
        cost_at_zero.is_finite() && grad_at_zero.iter().all(|v| v.is_finite()),
        "ψ-gradient and cost must be finite at θ=0"
    );
}
#[test]
fn iso_kappa_duchon_dx_dpsi_matches_fd() {
    // Compare the production frozen-spec dX/dψ path against centered FD
    // of X(ψ+h) - X(ψ-h). This intentionally goes through
    // `try_build_spatial_term_log_kappa_derivative`: the formula layer owns
    // the frozen centers, length-scale compensation, and composed
    // identifiability transform.
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let spec_orig = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let design = build_term_collection_design(data.view(), &spec_orig).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec_orig, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));

    let build_design_at = |psi: f64| -> Array2<f64> {
        // Rebuild design at psi via direct kernel build using frozen spec.
        let mut s = frozen.clone();
        if let SmoothBasisSpec::Duchon {
            spec: ref mut duchon,
            ..
        } = s.smooth_terms[0].basis
        {
            duchon.length_scale = Some((-psi).exp());
        }
        let d = build_term_collection_design(data.view(), &s).unwrap_or_else(|e| panic!("{} failed: {:?}", "rebuild", e));
        d.design.to_dense()
    };

    // Build derivative at psi=0.
    let psi_eval = 0.0_f64;
    let derivative_bundle =
        try_build_spatial_term_log_kappa_derivative(data.view(), &frozen, &design, 0)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula Duchon derivative should build", e))
            .expect("Duchon derivative should be available");
    let global_range = derivative_bundle.0;
    let p_total = derivative_bundle.1;
    let implicit_operator = derivative_bundle.8;
    let op = implicit_operator.unwrap_or_else(|| panic!("{} failed", "Duchon derivative should expose implicit operator"));
    let p = op.p_out();
    assert_eq!(p_total, design.design.ncols());
    assert_eq!(global_range.end - global_range.start, p);

    // FD reference.
    let h = 1e-4_f64;
    let x_plus = build_design_at(psi_eval + h);
    let x_minus = build_design_at(psi_eval - h);
    eprintln!(
        "[DXDPSI_FD] X(+h)[0,0..3]={:?} X(-h)[0,0..3]={:?}",
        x_plus.row(0).iter().take(3).copied().collect::<Vec<_>>(),
        x_minus.row(0).iter().take(3).copied().collect::<Vec<_>>(),
    );
    eprintln!(
        "[DXDPSI_FD] X(+h) shape={:?} X(-h) shape={:?} p_out={}",
        x_plus.shape(),
        x_minus.shape(),
        p,
    );
    // Also build at psi_eval to compare cols.
    let x_at = build_design_at(psi_eval);
    let orig_design = build_term_collection_design(data.view(), &spec_orig).unwrap_or_else(|e| panic!("{} failed: {:?}", "rebuild orig", e));
    eprintln!(
        "[DXDPSI_FD] X(psi_eval) shape={:?} orig_design.ncols={}",
        x_at.shape(),
        orig_design.design.ncols(),
    );

    // Multiply analytic operator by unit basis vectors.
    let mut analytic = Array2::<f64>::zeros((n, p));
    let mut basisv = Array1::<f64>::zeros(p);
    for j in 0..p {
        basisv[j] = 1.0;
        let col = op.forward_mul(0, &basisv.view()).unwrap_or_else(|e| panic!("{} failed: {:?}", "forward_mul", e));
        analytic.column_mut(j).assign(&col);
        basisv[j] = 0.0;
    }

    // Also check transpose_mul: X_tau^T v for v of length n.
    // FD reference: X_tau^T v should be (X(+h)^T - X(-h)^T)/(2h) · v.
    let smooth_start = global_range.start;
    let v_test = Array1::<f64>::from_shape_fn(n, |i| (i as f64 * 0.07).sin());
    let analytic_tv = op.transpose_mul(0, &v_test.view()).unwrap_or_else(|e| panic!("{} failed: {:?}", "transpose_mul", e));
    let fd_tv_full = (&x_plus.t() - &x_minus.t()) / (2.0 * h);
    let fd_tv = fd_tv_full.dot(&v_test);
    // Extract smooth portion only
    let fd_tv_smooth = fd_tv.slice(s![smooth_start..(smooth_start + p)]).to_owned();
    let max_tv_diff = analytic_tv
        .iter()
        .zip(fd_tv_smooth.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let max_tv_abs = analytic_tv.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    eprintln!(
        "[DXDPSI_TV] max|analytic_tv - fd_tv|={:.3e}  max|analytic_tv|={:.3e}",
        max_tv_diff, max_tv_abs
    );
    eprintln!(
        "[DXDPSI_TV] analytic_tv={:?}",
        analytic_tv.iter().take(p).copied().collect::<Vec<_>>()
    );
    eprintln!(
        "[DXDPSI_TV] fd_tv_smooth={:?}",
        fd_tv_smooth.iter().take(p).copied().collect::<Vec<_>>()
    );
    let fd_full = (&x_plus - &x_minus) / (2.0 * h);
    let fd = fd_full
        .slice(s![.., smooth_start..(smooth_start + p)])
        .to_owned();
    let mut max_diff = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for i in 0..n {
        for j in 0..p {
            let d = (analytic[[i, j]] - fd[[i, j]]).abs();
            if d > max_diff {
                max_diff = d;
            }
            if analytic[[i, j]].abs() > max_abs {
                max_abs = analytic[[i, j]].abs();
            }
        }
    }
    eprintln!(
        "[DXDPSI_FD] max|analytic - fd|={:.3e}  max|analytic|={:.3e}",
        max_diff, max_abs
    );
    eprintln!(
        "[DXDPSI_FD] analytic[0,..]={:?}",
        analytic.row(0).iter().take(p).copied().collect::<Vec<_>>(),
    );
    eprintln!(
        "[DXDPSI_FD] fd[0,..]={:?}",
        fd.row(0).iter().take(p).copied().collect::<Vec<_>>(),
    );
    assert!(max_diff < 5e-3 * max_abs.max(1e-3), "dX/dψ mismatch");
}
}
