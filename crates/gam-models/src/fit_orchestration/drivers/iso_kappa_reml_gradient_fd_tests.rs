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

/// The realized fixture behind [`iso_kappa_fd_variant_driver`], split out so a
/// targeted diagnostic can probe ONE θ with its own finite-difference stencil
/// instead of re-running the driver's whole probe grid at the driver's single
/// hard-wired step. Everything here is a pure function of
/// `(label, n, family, well_conditioned)`.
struct IsoKappaFixture {
    data: Array2<f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    family: LikelihoodSpec,
    frozen: TermCollectionSpec,
    frozen_design: TermCollectionDesign,
    spatial_terms: Vec<usize>,
    dims_per_term: Vec<usize>,
    rho_dim: usize,
    psi_dim: usize,
}

fn build_iso_kappa_fixture(
    label: &str,
    n: usize,
    family: LikelihoodSpec,
    well_conditioned: bool,
) -> IsoKappaFixture {
    // A `"*_2d"` label builds an ordinary 2-D feature cloud (the production
    // `matern(x1, x2)` regime: operator triplet {mass, tension, stiffness}, with
    // the per-axis tension and mixed-curvature stiffness blocks that only carry
    // cross-axis structure when d ≥ 2). This is the fast unit-level reproduction
    // of the #1122 stall, whose end-to-end pin is
    // `matern_2d_iso_kappa_outer_gradient_matches_fd`.
    let d = duchon_dims_from_label(label);
    let two_d = d >= 2;
    // gam#2735: the anisotropic arm enrols `d` per-axis ψ coordinates instead
    // of one isotropic log-κ, so the same oracle differentiates every one of
    // them. Only meaningful for `d ≥ 2`.
    let aniso = label.contains("_aniso") && two_d;
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
            // gam#979 — axes beyond the second are the large-scale benchmark's
            // standardized-PC regime: a low-discrepancy fill per axis (distinct
            // irrational multipliers, so no two axes alias), spread to the
            // ±2 range of a standardized coordinate rather than [0, 1]. The
            // truth picks up a mild contribution from every extra axis so no
            // kernel direction is degenerate at the seed.
            let mut extra = 0.0;
            for a in 2..d {
                let multiplier = ((a + 2) as f64 * 2.0 + 1.0).sqrt().fract();
                let value = 4.0 * ((i as f64 * multiplier).fract() - 0.5);
                data[[i, a]] = value;
                extra += 0.1 * (3.0 * value + a as f64).sin();
            }
            1.4 * (2.0 * std::f64::consts::PI * t).sin()
                + 0.9 * (2.0 * std::f64::consts::PI * t2).cos()
                + 0.5 * (t - 0.5)
                + extra
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
                length_scale: gam_terms::basis::MaternLengthScale::fixed(1.0),
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
            },
            input_scale: None,
        }
    } else {
        SmoothBasisSpec::Duchon {
            feature_cols: (0..d).collect(),
            spec: DuchonBasisSpec {
                radial_reparam: None,
                periodic: None,
                center_strategy: CenterStrategy::FarthestPoint {
                    num_centers: duchon_centers_from_label(label),
                },
                length_scale: Some(1.0),
                // gam#979 — the label can move the hybrid's two integer orders
                // off the historical `(Linear, 1)` pair. `bench/large_scale`'s
                // shared CTN preprocessor ships `duchon(pc1..pc16, order=0,
                // power=9, length_scale=1)`, i.e. `(Zero, 9)`: a Matérn-blend
                // whose polynomial null space is the constant alone. Every
                // Duchon ψ gate in this file and in `gam-terms` was pinned at
                // `Linear`, so an order-Zero-only defect in the κ chain had no
                // gate to fail; these two knobs let the same oracle walk the
                // shipped configuration.
                power: duchon_power_from_label(label),
                nullspace_order: duchon_nullspace_order_from_label(label),
                identifiability: SpatialIdentifiability::default(),
                // gam#2735 — an `"*_aniso*"` label asks for the per-axis η the
                // hybrid Duchon now enrols as OUTER ψ coordinates. The
                // contrasts are deliberately non-zero and unequal: an all-zero
                // vector is the `auto_seed_aniso_contrasts` sentinel and would
                // be replaced by knot-cloud geometry, which is a different
                // (and here nearly symmetric) starting point.
                aniso_log_scales: if aniso {
                    Some((0..d).map(|a| 0.25 - 0.5 * a as f64).collect())
                } else {
                    None
                },
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                boundary: OneDimensionalBoundary::Open,
            },
            input_scale: None,
        }
    };
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "variant_1d".to_string(),
            basis,
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    // Isotropic κ: one log-κ axis regardless of feature dimension `d` (the 2-D
    // cloud still enrolls a single isotropic κ, not a per-axis η) — UNLESS the
    // term carries explicit per-axis contrasts, in which case gam#2735 enrols
    // one raw ψ_a per axis. This assertion is itself the enrollment gate: a
    // regression that re-froze Duchon anisotropy would report `[1]` here.
    let expected_dims = if aniso { vec![d] } else { vec![1] };
    assert_eq!(
        dims_per_term, expected_dims,
        "{label}: expect {expected_dims:?} spatial ψ axes"
    );
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

    IsoKappaFixture {
        data,
        y,
        weights,
        offset,
        family,
        frozen,
        frozen_design,
        spatial_terms,
        dims_per_term,
        rho_dim,
        psi_dim,
    }
}

/// `"*_order0*"` selects the constant-only Duchon null space (`p = 1`);
/// anything else keeps the historical `Linear` (`p = 2`) fixture (gam#979).
fn duchon_nullspace_order_from_label(label: &str) -> DuchonNullspaceOrder {
    if label.contains("_order0") {
        DuchonNullspaceOrder::Zero
    } else {
        DuchonNullspaceOrder::Linear
    }
}

/// `"*_power9*"` selects the spectral power the large-scale CTN preprocessor
/// ships; anything else keeps the historical `s = 1` fixture (gam#979).
fn duchon_power_from_label(label: &str) -> f64 {
    if label.contains("_power9") { 9.0 } else { 1.0 }
}

/// `"*_16d*"` builds the benchmark's sixteen-PC cloud, `"*_2d"` the historical
/// two-axis fixture, anything else the 1-D line (gam#979).
fn duchon_dims_from_label(label: &str) -> usize {
    if label.contains("_16d") {
        16
    } else if label.contains("_3d") {
        3
    } else if label.ends_with("_2d") || label.contains("_2d_") {
        2
    } else {
        1
    }
}

/// `"*_centers24*"` selects the benchmark's 24 farthest-point centers; anything
/// else keeps the historical 8 (gam#979).
fn duchon_centers_from_label(label: &str) -> usize {
    if label.contains("_centers24") { 24 } else { 8 }
}

/// The `FitOptions` every iso-κ FD probe evaluates the production criterion
/// under: inference off, inner tolerance far below any gradient gap the oracle
/// resolves, and no shrinkage floor (so the criterion is the criterion, not a
/// regularized surrogate).
fn iso_kappa_fd_fit_options() -> FitOptions {
    FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        ..FitOptions::default()
    }
}

impl IsoKappaFixture {
    fn cache(&self) -> SingleBlockExactJointDesignCache<'_> {
        SingleBlockExactJointDesignCache::new(
            self.data.view(),
            self.frozen.clone(),
            self.frozen_design.clone(),
            self.spatial_terms.clone(),
            self.rho_dim,
            self.dims_per_term.clone(),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "single-block cache", e))
    }

    fn evaluator<'a>(
        &'a self,
        external_opts: &'a ExternalOptimOptions,
    ) -> gam_solve::estimate::ExternalJointHyperEvaluator<'a> {
        gam_solve::estimate::ExternalJointHyperEvaluator::new(
            self.y.view(),
            self.weights.view(),
            &self.frozen_design.design,
            self.offset.view(),
            &self.frozen_design.penalties,
            external_opts,
            "iso-κ variant FD evaluator",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e))
    }

    fn external_opts(&self) -> ExternalOptimOptions {
        external_opts_for_design(&self.family, &self.frozen_design, &iso_kappa_fd_fit_options())
    }
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
///
/// ANSWERED (#2461): truncation. The question this sweep poses by hand is the
/// same one `ridders_derivative` now answers automatically for every component
/// — the production audit this diagnostic shadows extrapolates across the
/// ladder and reports its own uncertainty, so the ~1.6e-3 residual it was
/// written to explain is no longer in the comparison and the end-to-end gate's
/// tolerance came down from `5e-2` to `5e-3`. Kept because a printed sweep at
/// the production init is still the cheapest way to SEE the law, and because
/// its `V(ψ)` scaling argument (`κ^{2m}`, `m = ν + d/2 = 3.5`) is the reason
/// this fixture has a large third derivative in the first place.
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
            frozen_parametric_residualization: None,
            name: "matern_2d".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 37 },
                    periodic: None,
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(length_scale),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
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
                        spec.length_scale.resolved().unwrap(),
                        spec.double_penalty,
                        match &spec.identifiability {
                            MaternIdentifiability::FrozenTransform { transform } =>
                                format!("FrozenTransform{{z_dims={:?}}}", transform.dim()),
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
                    centers, input_scale, length_scale, ..
                } = &t.metadata
                {
                    let csum: f64 = centers.iter().map(|v| v.abs()).sum();
                    let c00 = centers.get((0, 0)).copied().unwrap_or(f64::NAN);
                    let c01 = centers.get((0, 1)).copied().unwrap_or(f64::NAN);
                    eprintln!(
                        "[FINGERPRINT] HARNESS meta.Matern length_scale={length_scale:.10} input_scale={input_scale:?} centers_abs_sum={csum:.10e} c[0,0]={c00:.10} c[0,1]={c01:.10}"
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

// No `iso_kappa_thinplate_*_fd` companion to the Duchon FD tests above:
// thin-plate is deliberately excluded from the spatial κ-axis enrollment
// by `spatial_term_supports_hyper_optimization` (a scalar TPS κ creates
// the flat ρ/κ valleys tracked in #718 / #721 / #731 / #732), so there
// is no analytic κ-gradient on which an FD comparison could land.

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
            frozen_parametric_residualization: None,
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

/// #2425 MEASUREMENT (reports, never fails): is the marginal `psi_only` ρ₁
/// analytic-vs-FD gap central-difference TRUNCATION or a noise floor?
#[test]
fn zz_measure_psi_only_rho1_fd_step_law_2425() {
    // The one component that fails `iso_kappa_duchon_n_smaller_fd`, and it is
    // near-zero and marginal in EVERY config, not just at n=20:
    //
    //   duchon_probit_n20      an=-1.9454e-5 fd=-1.4377e-5  rel=5.077e-3  FAIL
    //   duchon_probit_rho_only an=-1.9108e-5 fd=-1.5746e-5  rel=3.362e-3  pass
    //   duchon_logit           an=+9.1413e-6 fd=+6.5455e-6  rel=2.596e-3  pass
    //   duchon_gaussian        an=+1.7330e-5 fd=+1.6750e-5  rel=5.799e-4  pass
    //
    // The driver's `denom = max(|fd|, |an|, 1e-3)` floors the denominator at
    // 1e-3, so on a component of size ~1.5e-5 the 5e-3 relative gate is really
    // an ABSOLUTE 5e-6 gate, and every config sits within a factor of two of it.
    // Before anyone moves that number, the h-law decides which side is wrong:
    //
    //   gap ∝ h²   → central-difference TRUNCATION. The analytic gradient is
    //                right and the ORACLE's fixed h=1e-5 is too coarse for a
    //                component with large third derivative; Richardson (or a
    //                smaller h) fixes it and no tolerance needs to move.
    //   gap flat, or growing as h shrinks → a noise floor (the inner PIRLS
    //                stationarity residual propagating into both sides). Then
    //                the absolute floor must be DERIVED from that residual
    //                rather than left at a magic 1e-3 denominator.
    //
    // Reports only. `ratio` is gap/h²: constant ⇒ truncation.
    //
    // The general form of this question — "which of the three laws is the gap
    // following, here, at this θ?" — is no longer asked by hand: the driver
    // measures every component with `ridders_derivative`, which walks the ladder
    // and reports its own uncertainty, so a component whose law is `ν/h` comes
    // back Unresolved rather than as a gradient violation (#2461). This
    // measurement survives because it PINS the noise floor `ν ≈ 1.5e-11` at a
    // known θ, which is a property of the evaluator worth keeping visible; the
    // conclusion it drew about which side to fix is now drawn automatically.
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
        "psi_only rho1 FD step law",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e));

    // The driver's `psi_only` probe: every ρ at 0, ψ at 0.4.
    let theta_dim = rho_dim + psi_dim;
    let mut theta = Array1::<f64>::zeros(theta_dim);
    for k in 0..psi_dim {
        theta[rho_dim + k] = 0.4;
    }
    const COORD: usize = 1;

    let eval_at = |theta: &Array1<f64>,
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

    let (cost, grad, _h) = eval_at(
        &theta,
        gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
        &mut cache,
        &mut evaluator,
    );
    let analytic = grad[COORD];
    eprintln!(
        "[zz-steplaw-2425] cost={cost:.12e} analytic_rho{COORD}={analytic:+.10e}"
    );
    let mut previous: Option<(f64, f64)> = None;
    for h in [
        1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7,
    ] {
        let mut plus = theta.clone();
        plus[COORD] += h;
        let mut minus = theta.clone();
        minus[COORD] -= h;
        let (cp, _, _) = eval_at(
            &plus,
            gam_solve::rho_optimizer::OuterEvalOrder::Value,
            &mut cache,
            &mut evaluator,
        );
        let (cm, _, _) = eval_at(
            &minus,
            gam_solve::rho_optimizer::OuterEvalOrder::Value,
            &mut cache,
            &mut evaluator,
        );
        let fd = (cp - cm) / (2.0 * h);
        let gap = analytic - fd;
        // Richardson against the previous (3x coarser) rung: for a clean
        // O(h²) stencil this cancels the leading truncation term.
        let richardson = previous.map(|(hc, fdc)| {
            let ratio = (hc / h).powi(2);
            (ratio * fd - fdc) / (ratio - 1.0)
        });
        eprintln!(
            "[zz-steplaw-2425] h={h:.1e} fd={fd:+.10e} gap={gap:+.4e} \
             gap/h2={:.4e} richardson={}",
            gap / (h * h),
            richardson
                .map(|r| format!("{r:+.10e} (gap {:+.4e})", analytic - r))
                .unwrap_or_else(|| "-".to_string())
        );
        previous = Some((h, fd));
    }
}

/// #2461 — the step-law discriminator the issue asks for, on the row that
/// actually fails: `duchon_gaussian`, ψ (`j = rho_dim`), at the `rho1@R` rung
/// of the #2425 saturation ladder.
///
/// The reported defect is a *constant* 0.54% analytic-vs-FD relative gap on a
/// ψ-gradient of magnitude ~249, stable to four digits over twelve e-folds of
/// ρ₁. The ladder measures it at ONE step (`h = 3e-4`), which cannot separate
/// the three candidate laws. This sweeps `h` at a fixed rung and reports both
/// `gap/h²` (constant ⇒ central-difference truncation, analytic side right)
/// and `gap·h` (constant ⇒ evaluator noise), plus a Richardson extrapolation
/// against the 3× coarser rung, which kills the leading `h²` term and so
/// converges to the true derivative if truncation is the story.
///
/// Reporting only — it is the measurement that decides which side to fix.
#[test]
fn zz_measure_duchon_psi_fd_step_law_at_rho1_2461() {
    const RUNG: f64 = 15.0;
    let fixture = build_iso_kappa_fixture(
        "duchon_gaussian",
        80,
        LikelihoodSpec::gaussian_identity(),
        false,
    );
    let rho_dim = fixture.rho_dim;
    let psi_dim = fixture.psi_dim;
    let external_opts = fixture.external_opts();
    let mut cache = fixture.cache();
    let mut evaluator = fixture.evaluator(&external_opts);
    let data = fixture.data.view();

    // The ladder's `rho1@RUNG` probe: `theta_base` with ρ₁ raised to the rung.
    let mut theta = Array1::<f64>::zeros(rho_dim + psi_dim);
    for j in 0..rho_dim {
        theta[j] = 0.2 - 0.1 * j as f64;
    }
    theta[1] = RUNG;
    let coord = rho_dim;

    let eval_at = |theta: &Array1<f64>,
                   order: gam_solve::rho_optimizer::OuterEvalOrder,
                       cache: &mut SingleBlockExactJointDesignCache<'_>,
                       evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> (f64, Array1<f64>) {
        cache
            .ensure_theta(theta)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data,
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper dirs build", e))
        .expect("hyper dirs present");
        let (cost, grad, _) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            order,
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "outer eval", e));
        (cost, grad)
    };

    let (cost, grad) = eval_at(
        &theta,
        gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
        &mut cache,
        &mut evaluator,
    );
    let analytic = grad[coord];
    eprintln!(
        "[zz-2461-steplaw] rung=rho1@{RUNG} cost={cost:.12e} analytic_psi={analytic:+.10e}"
    );
    let mut previous: Option<(f64, f64)> = None;
    for h in [
        3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6,
    ] {
        let mut plus = theta.clone();
        plus[coord] += h;
        let mut minus = theta.clone();
        minus[coord] -= h;
        let (cp, _) = eval_at(
            &plus,
            gam_solve::rho_optimizer::OuterEvalOrder::Value,
            &mut cache,
            &mut evaluator,
        );
        let (cm, _) = eval_at(
            &minus,
            gam_solve::rho_optimizer::OuterEvalOrder::Value,
            &mut cache,
            &mut evaluator,
        );
        let fd = (cp - cm) / (2.0 * h);
        let gap = analytic - fd;
        let richardson = previous.map(|(hc, fdc): (f64, f64)| {
            let ratio = (hc / h).powi(2);
            (ratio * fd - fdc) / (ratio - 1.0)
        });
        eprintln!(
            "[zz-2461-steplaw] h={h:.1e} fd={fd:+.10e} gap={gap:+.6e} \
             gap/h2={:+.4e} gap*h={:+.4e} richardson={}",
            gap / (h * h),
            gap * h,
            richardson
                .map(|r| format!("{r:+.10e} (gap {:+.6e})", analytic - r))
                .unwrap_or_else(|| "-".to_string())
        );
        previous = Some((h, fd));
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
    // Compare the production frozen-spec design jets against centered FD in
    // the collection's invariant LOCAL chart.  The collection may choose a
    // different RRQR/whitened coefficient basis at every ψ, but differentiating
    // those arbitrary basis vectors is not statistical geometry.  Freeze the
    // current coefficient chart and apply the fixed row projector P_C to both
    // the analytic jet and the FD value path (gam#2760).
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let spec_orig = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
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
                input_scale: None,
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
    let smooth_start = global_range.start;
    let gauge = design.smooth.terms[0]
        .collection_gauge
        .as_ref()
        .expect("the intrinsic Duchon term must carry its collection gauge");
    let projector = FixedRowSpaceProjector::from_constraint_block(
        gauge.constraint_block.view(),
    )
    .expect("collection row-space projector");
    assert!(projector.rank() > 0, "the regression must arm a nonempty row gauge");
    let project_frozen_chart = |mut full: Array2<f64>| -> Array2<f64> {
        let mut smooth = full
            .slice(s![.., smooth_start..(smooth_start + p)])
            .to_owned();
        projector
            .project_matrix_in_place(&mut smooth)
            .expect("fixed-chart value projection");
        full.slice_mut(s![.., smooth_start..(smooth_start + p)])
            .assign(&smooth);
        full
    };

    // FD reference.
    let h = 1e-4_f64;
    let x_plus = project_frozen_chart(build_design_at(psi_eval + h));
    let x_minus = project_frozen_chart(build_design_at(psi_eval - h));
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
    let x_at = project_frozen_chart(build_design_at(psi_eval));
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
    let mut analytic_second = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        basisv[j] = 1.0;
        let col = op
            .forward_mul_second_diag(0, &basisv.view())
            .unwrap_or_else(|e| panic!("second forward_mul failed: {e:?}"));
        analytic_second.column_mut(j).assign(&col);
        basisv[j] = 0.0;
    }

    let relative_cross_residual = |jet: &Array2<f64>| -> f64 {
        let cross = gauge.constraint_block.t().dot(jet);
        let cross_norm = cross.iter().map(|value| value * value).sum::<f64>().sqrt();
        let constraint_norm = gauge
            .constraint_block
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let jet_norm = jet.iter().map(|value| value * value).sum::<f64>().sqrt();
        cross_norm / (constraint_norm * jet_norm).max(f64::MIN_POSITIVE)
    };
    let first_orthogonality = relative_cross_residual(&analytic);
    let second_orthogonality = relative_cross_residual(&analytic_second);
    assert!(
        first_orthogonality <= 1e-8 && second_orthogonality <= 1e-8,
        "collection-gauged design jets must stay orthogonal to C: first={first_orthogonality:.3e}, second={second_orthogonality:.3e}"
    );

    // Also check transpose_mul: X_tau^T v for v of length n.
    // FD reference: X_tau^T v should be (X(+h)^T - X(-h)^T)/(2h) · v.
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
    let fd_second_full = (&x_plus - &(2.0 * &x_at) + &x_minus) / (h * h);
    let fd_second = fd_second_full
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
    let max_second_diff = analytic_second
        .iter()
        .zip(fd_second.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let max_second_abs = analytic_second
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_second_diff < 2e-2 * max_second_abs.max(1e-3),
        "d²X/dψ² mismatch: max_diff={max_second_diff:.3e}, scale={max_second_abs:.3e}"
    );
}

/// #2454 MEASUREMENT (reports, never fails): put the MONOTONE fixture's own spec
/// through an evaluator whose gradient can be finite-difference-checked.
///
/// Everything claiming `∂V/∂ρ = −c·λ` on #2454 came from the standard-REML
/// certificate's printed probe gradients, which have never been FD-verified.
/// Every ladder that HAS been FD-verified runs the joint iso-κ evaluator and
/// saturates correctly. The two disagree, so the discriminator is to run the
/// SAME fixture — `length_scale = 12.0`, 12 centers, n=60, d=2,
/// `double_penalty: true`, `CenterSumToZero` — through the checkable evaluator.
///
///   `|g|` decaying, FD agreeing ⇒ the criterion is bounded for this fixture and
///        the standard-REML path's printed gradient is the defect;
///   `|g| ∝ e^ρ` reproduced AND FD-confirmed ⇒ the criterion really is unbounded
///        and both of my mechanism hypotheses were wrong.
///
/// `ĉ = −e^ρ·∂V/∂ρ` is reported alongside so the tail law is directly readable.
#[test]
fn zz_measure_monotone_fixture_through_checkable_evaluator_2454() {
    let n = 60usize;
    let d = 2usize;
    let mut data = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (i as f64 * 0.17).sin();
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        y[i] = (3.0 * x0).cos() + 0.35 * x1;
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        ..FitOptions::default()
    };
    let family = LikelihoodSpec::gaussian_identity();

    let design = build_term_collection_design(data.view(), &spec)
        .unwrap_or_else(|e| panic!("design failed: {e:?}"));
    let frozen = freeze_term_collection_from_design(&spec, &design)
        .unwrap_or_else(|e| panic!("freeze failed: {e:?}"));
    let frozen_design = build_term_collection_design(data.view(), &frozen)
        .unwrap_or_else(|e| panic!("frozen design failed: {e:?}"));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = frozen_design.penalties.len();
    let psi_dim: usize = dims_per_term.iter().sum();
    eprintln!("[zz-mono-2454] rho_dim={rho_dim} psi_dim={psi_dim} p={}", frozen_design.design.ncols());

    let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("cache failed: {e:?}"));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "#2454 monotone fixture evaluator",
    )
    .unwrap_or_else(|e| panic!("evaluator failed: {e:?}"));

    let cost_at = |theta: &Array1<f64>,
                   cache: &mut SingleBlockExactJointDesignCache<'_>,
                   evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> f64 {
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("ensure_theta: {e:?}"));
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
                "#2454 cost-only",
                None,
            )
            .unwrap_or_else(|e| panic!("cost-only: {e:?}"))
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
        .unwrap_or_else(|e| panic!("hyper dirs: {e:?}"))
        .expect("hyper dirs present");
        let (cost, grad, _h) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .unwrap_or_else(|e| panic!("outer eval: {e:?}"));
        (cost, grad)
    };

    // #2454 STEP-LAW at the sign-disagreement point. At rho=15 the analytic and
    // FD gradients disagree by 166% AND on the SIGN (an=+6.499e-2, fd=-9.835e-2,
    // gap 0.163). A cost-noise floor of nu~1e-11 gives nu/h ~ 3e-8 at h=3e-4,
    // seven orders too small, so either the inner solve's noise explodes here or
    // truncation needs a third derivative of order 1e7. Sweeping h separates
    // them WITHOUT needing a higher-precision reference:
    //     gap proportional to 1/h  => NOISE (inner PIRLS not resolving at rho=15)
    //     gap proportional to h^2  => TRUNCATION (FD wrong; analytic stands)
    //     gap flat in h            => the ANALYTIC gradient is wrong
    {
        let value = 15.0_f64;
        let mut theta = Array1::<f64>::zeros(rho_dim + psi_dim);
        for j in 0..rho_dim {
            theta[j] = value;
        }
        let (cost, grad) = analytic_at(&theta, &mut cache, &mut evaluator);
        let an = grad[0];
        eprintln!("[zz-steplaw15-2454] rho=15 COST={cost:+.12e} analytic_rho0={an:+.8e}");
        for hh in [1e-2_f64, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6] {
            let mut plus = theta.clone();
            plus[0] += hh;
            let mut minus = theta.clone();
            minus[0] -= hh;
            let cp = cost_at(&plus, &mut cache, &mut evaluator);
            let cm = cost_at(&minus, &mut cache, &mut evaluator);
            let fd = (cp - cm) / (2.0 * hh);
            let gap = an - fd;
            eprintln!(
                "[zz-steplaw15-2454] h={hh:.1e} fd={fd:+.8e} gap={gap:+.4e} \
                 gap_times_h={:.4e} gap_over_h2={:.4e}",
                gap * hh,
                gap / (hh * hh)
            );
        }
    }
    let h = 3e-4_f64;
    for value in [6.0_f64, 9.0, 12.0, 15.0, 18.0, 21.0] {
        let mut theta = Array1::<f64>::zeros(rho_dim + psi_dim);
        for j in 0..rho_dim {
            theta[j] = value;
        }
        let (cost, grad) = analytic_at(&theta, &mut cache, &mut evaluator);
        for j in 0..rho_dim {
            let mut plus = theta.clone();
            plus[j] += h;
            let mut minus = theta.clone();
            minus[j] -= h;
            let cp = cost_at(&plus, &mut cache, &mut evaluator);
            let cm = cost_at(&minus, &mut cache, &mut evaluator);
            let fd = (cp - cm) / (2.0 * h);
            let an = grad[j];
            let denom = fd.abs().max(an.abs()).max(1e-12);
            eprintln!(
                "[zz-mono-2454] rho={value:5.1} j={j} COST={cost:+.10e} an={an:+.6e} \
                 fd={fd:+.6e} rel={:.3e} chat_an={:+.6e}",
                (an - fd).abs() / denom,
                -value.exp() * an
            );
        }
    }
}

/// One rung of the #2454 ladder: the analytic outer gradient, the central
/// finite difference of the same criterion, and the two penalty-energy
/// spellings behind the `fixed_beta` channel, at one ρ and one ρ-coordinate.
#[derive(Clone, Copy, Debug)]
struct RhoGradientLadderRow2454 {
    rho: f64,
    coordinate: usize,
    lambda: f64,
    cost: f64,
    analytic_total: f64,
    finite_difference_total: f64,
    analytic_fixed_beta: f64,
    finite_difference_fixed_beta: f64,
    analytic_logdet_h: f64,
    finite_difference_logdet_h: f64,
    analytic_logdet_s: f64,
    finite_difference_logdet_s: f64,
    analytic_kkt: f64,
    finite_difference_kkt: f64,
    block_quadratic: f64,
    penalty_energy_criterion: f64,
    penalty_energy_blocks: f64,
    penalized_rank: usize,
    declared_null_dim: usize,
    beta_null_energy: f64,
    /// The rank the criterion's `−½log|S(λ)|₊` ranges over, and its value. The
    /// criterion's asymptotic slope in ρ is `½(penalized_rank − logdet_rank)`,
    /// so these two integers decide whether an interior optimum exists at all.
    logdet_rank: usize,
    logdet_value: f64,
}

/// Run the #2454 ladder on the fixture the issue was opened against: the
/// monotone spatial-length-scale spec (`matern` ν=5/2, `length_scale = 12.0`,
/// 12 `FarthestPoint` centers, n=60, d=2, `double_penalty: true`,
/// `CenterSumToZero`), put through the joint iso-κ evaluator so its gradient is
/// finite-difference-checkable.
///
/// Every ρ coordinate is held at the same `value` and each is differenced in
/// turn. The typed ρ-block audit (`enable_rho_outer_audit`) supplies both the
/// criterion VALUE split and the matching per-ρ analytic gradient split, so each
/// component can be graded against the gradient part that owns it instead of
/// only their sum.
fn rho_gradient_part_ladder_2454(
    rho_values: &[f64],
    step: f64,
) -> Vec<RhoGradientLadderRow2454> {
    rho_gradient_part_ladder_family_2454(rho_values, step, LikelihoodSpec::gaussian_identity())
}

/// The #2454 ladder over an arbitrary family.
///
/// #2623/#2644 DISCRIMINATOR. The Gaussian-identity ladder above can only grade
/// three of the four criterion channels: a Gaussian fit is dispersion-profiled
/// (`DispersionHandling::ProfiledGaussian`), and the inner-KKT envelope
/// correction `−½rᵀH⁻¹r` is gated on `DispersionHandling::Fixed` — so the `kkt`
/// column is identically zero on every Gaussian rung and its analytic-vs-FD
/// comparison is vacuous. Every non-Gaussian family takes the `Fixed` branch,
/// which is where the correction, and its ρ-derivative
/// `−a_kᵀq + ½qᵀA_kq`, actually run.
///
/// Threading the family through the SAME ladder (one fixture, one audit, one FD
/// stencil) is what makes the two arms comparable: a channel that is clean on
/// the Gaussian arm and dirty on the binomial arm is the KKT channel by
/// elimination, with no second code path to have drifted.
fn rho_gradient_part_ladder_family_2454(
    rho_values: &[f64],
    step: f64,
    family: LikelihoodSpec,
) -> Vec<RhoGradientLadderRow2454> {
    use gam_solve::estimate::outer_eval_capture::{
        enable_rho_outer_audit, take_rho_outer_audit, PenaltyEnergyAudit, RhoOuterAudit,
    };

    let n = 60usize;
    let d = 2usize;
    let mut data = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (i as f64 * 0.17).sin();
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        let eta = (3.0 * x0).cos() + 0.35 * x1;
        y[i] = if family.is_gaussian_identity() {
            eta
        } else {
            // Smooth, non-separating Bernoulli labels (the `well_conditioned`
            // recipe from `build_iso_kappa_fixture`): a deterministic logistic
            // threshold against a fixed phase grid keeps fitted μ away from
            // {0,1}, so `H` stays well conditioned and the FD oracle grades the
            // KKT channel rather than the conditioning of the inner solve.
            let p = 1.0 / (1.0 + (-0.6 * eta).exp());
            let u = 0.5 * ((5.0 * (i as f64) + 0.5).sin() + 1.0);
            if u < p {
                1.0
            } else {
                0.0
            }
        };
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        ..FitOptions::default()
    };
    let design = build_term_collection_design(data.view(), &spec)
        .unwrap_or_else(|e| panic!("design failed: {e:?}"));
    let frozen = freeze_term_collection_from_design(&spec, &design)
        .unwrap_or_else(|e| panic!("freeze failed: {e:?}"));
    let frozen_design = build_term_collection_design(data.view(), &frozen)
        .unwrap_or_else(|e| panic!("frozen design failed: {e:?}"));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = frozen_design.penalties.len();
    let psi_dim: usize = dims_per_term.iter().sum();

    let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("cache failed: {e:?}"));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "#2454 part decomposition",
    )
    .unwrap_or_else(|e| panic!("evaluator failed: {e:?}"));

    // Cost-only evaluation that ALSO returns the criterion component split and
    // the two penalty-energy spellings.
    let cost_parts_at = |theta: &Array1<f64>,
                         cache: &mut SingleBlockExactJointDesignCache<'_>,
                         evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> (f64, [f64; 4], PenaltyEnergyAudit) {
        cache
            .ensure_theta(theta)
            .unwrap_or_else(|e| panic!("ensure_theta: {e:?}"));
        enable_rho_outer_audit();
        let design = cache.design();
        let cost = evaluator
            .evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                theta,
                rho_dim,
                None,
                "#2454 cost-only parts",
                None,
            )
            .unwrap_or_else(|e| panic!("cost-only: {e:?}"));
        let audit: RhoOuterAudit = take_rho_outer_audit().expect("rho audit armed");
        let (audit_cost, components) = audit
            .criterion
            .expect("criterion components recorded for every eval");
        assert!(
            (audit_cost - cost).abs() <= 1e-9 * cost.abs().max(1.0),
            "audit cost {audit_cost:+.12e} disagrees with returned cost {cost:+.12e}"
        );
        (
            cost,
            components,
            audit.penalty_energy.expect("penalty energy recorded"),
        )
    };

    let analytic_at = |theta: &Array1<f64>,
                       cache: &mut SingleBlockExactJointDesignCache<'_>,
                       evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> (f64, RhoOuterAudit) {
        cache.ensure_theta(theta).expect("ensure_theta");
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap_or_else(|e| panic!("hyper dirs: {e:?}"))
        .expect("hyper dirs present");
        enable_rho_outer_audit();
        let (cost, _grad, _h) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .unwrap_or_else(|e| panic!("outer eval: {e:?}"));
        (cost, take_rho_outer_audit().expect("rho audit armed"))
    };

    let mut rows = Vec::new();
    for &value in rho_values {
        let mut theta = Array1::<f64>::zeros(rho_dim + psi_dim);
        for j in 0..rho_dim {
            theta[j] = value;
        }
        let (cost, audit) = analytic_at(&theta, &mut cache, &mut evaluator);
        let energy = audit.penalty_energy.expect("penalty energy recorded");
        let frame = audit.penalty_frame.as_ref();
        for j in 0..rho_dim {
            let mut plus = theta.clone();
            plus[j] += step;
            let mut minus = theta.clone();
            minus[j] -= step;
            let (cost_plus, comp_plus, _) = cost_parts_at(&plus, &mut cache, &mut evaluator);
            let (cost_minus, comp_minus, _) = cost_parts_at(&minus, &mut cache, &mut evaluator);
            let part = audit
                .parts
                .iter()
                .find(|part| part.index == j)
                .copied()
                .expect("rho gradient part recorded");
            let fd = |idx: usize| (comp_plus[idx] - comp_minus[idx]) / (2.0 * step);
            rows.push(RhoGradientLadderRow2454 {
                rho: value,
                coordinate: j,
                lambda: part.lambda,
                cost,
                analytic_total: part.total,
                finite_difference_total: (cost_plus - cost_minus) / (2.0 * step),
                analytic_fixed_beta: part.fixed_beta,
                finite_difference_fixed_beta: fd(0),
                analytic_logdet_h: part.logdet_h,
                finite_difference_logdet_h: fd(1),
                analytic_logdet_s: part.logdet_s,
                finite_difference_logdet_s: fd(2),
                analytic_kkt: part.total
                    - (part.fixed_beta + part.logdet_h + part.logdet_s),
                finite_difference_kkt: fd(3),
                block_quadratic: part.block_quadratic,
                penalty_energy_criterion: energy.stable,
                penalty_energy_blocks: energy.block_sum,
                penalized_rank: frame.map_or(0, |f| f.e_rows),
                declared_null_dim: frame.map_or(0, |f| f.null_dim),
                beta_null_energy: frame.map_or(f64::NAN, |f| f.beta_null_energy),
                logdet_rank: frame.map_or(0, |f| f.penalty_logdet_rank),
                logdet_value: frame.map_or(f64::NAN, |f| f.penalty_logdet_value),
            });
        }
    }
    rows
}

/// #2454 GATE: the outer ρ-gradient must differentiate the penalty the
/// criterion applies, so its analytic-vs-FD error must NOT scale with λ.
///
/// The defect this pins was an additive `+5.0e-8·λ` in the `fixed_beta`
/// channel: the criterion's penalty is the split-projected `S̃(λ) = E(λ)ᵀE(λ)`
/// (the reparameterization keeps only balanced-penalty eigendirections above a
/// relative rank tolerance, so `H`, `log|S|₊` and the inner solve share one rank
/// structure), while the gradient read the UNPROJECTED blocks `S_k`, whose own
/// root ranks exceed the split's penalized rank on this fixture. β̂ is
/// unpenalized in the discarded directions, so `Σ_k λ_k β̂ᵀS_kβ̂` charged energy
/// the criterion never paid — and `∂/∂ρ_k` multiplied that phantom by `λ_k`.
///
/// The gate is deliberately expressed as a SCALING statement rather than a
/// tolerance on one rung. A tolerance at `ρ = 15` alone could be met by a
/// coincidentally small coefficient; only "the error at λ = 3.3e6 is the same
/// size as the error at λ = 4.0e2" rules out a term proportional to λ. The
/// measured pre-fix ratio across this span was ~8000× (i.e. exactly `λ`); the
/// post-fix ratio is ~1.7×, the central-difference truncation floor.
#[test]
fn outer_rho_gradient_error_does_not_scale_with_lambda_2454() {
    let step = 3e-4_f64;
    let rows = rho_gradient_part_ladder_2454(&[6.0, 9.0, 12.0, 15.0], step);
    assert!(!rows.is_empty(), "ladder produced no rungs");
    assert!(
        rows.iter().all(|row| row.penalized_rank > 0),
        "the penalty-frame audit did not report a penalized rank; the gate would \
         then be grading an unarmed instrument"
    );

    let worst_at = |rho: f64| -> f64 {
        rows.iter()
            .filter(|row| row.rho == rho)
            .map(|row| (row.analytic_total - row.finite_difference_total).abs())
            .fold(0.0_f64, f64::max)
    };
    let low = worst_at(6.0);
    let high = worst_at(15.0);
    let lambda_ratio = 15.0_f64.exp() / 6.0_f64.exp();

    let report = || -> String {
        rows.iter()
            .map(|row| {
                format!(
                    "rho={:5.1} j={} lambda={:.4e} an={:+.10e} fd={:+.10e} gap={:+.4e} \
                     gap/lambda={:+.4e} | fixed_beta an={:+.6e} fd={:+.6e} | \
                     energy criterion={:+.10e} blocks={:+.10e} ratio={:.9} | \
                     penalized_rank={} null_dim={} beta_null_energy={:.4e}",
                    row.rho,
                    row.coordinate,
                    row.lambda,
                    row.analytic_total,
                    row.finite_difference_total,
                    row.analytic_total - row.finite_difference_total,
                    (row.analytic_total - row.finite_difference_total) / row.lambda,
                    row.analytic_fixed_beta,
                    row.finite_difference_fixed_beta,
                    row.penalty_energy_criterion,
                    row.penalty_energy_blocks,
                    row.penalty_energy_criterion / row.penalty_energy_blocks,
                    row.penalized_rank,
                    row.declared_null_dim,
                    row.beta_null_energy,
                )
            })
            .collect::<Vec<_>>()
            .join("\n  ")
    };

    // 1. Absolute: the gradient must be usable in the saturated region at all.
    //    Pre-fix this reached 1.63e-1 against a true gradient of -1.77, i.e. it
    //    reported the wrong sign relative to FD.
    const ABSOLUTE_GAP_BOUND: f64 = 1.0e-4;
    assert!(
        high <= ABSOLUTE_GAP_BOUND,
        "#2454: worst |analytic - fd| at rho=15 is {high:.4e}, above {ABSOLUTE_GAP_BOUND:.1e}\n  {}",
        report()
    );

    // 2. Scaling: the error must not grow like lambda. Allow a generous 50x
    //    against a lambda ratio of 8103x, so the gate distinguishes "flat" from
    //    "proportional to lambda" without being brittle about the FD floor.
    const SCALING_HEADROOM: f64 = 50.0;
    let floor = 1.0e-9;
    assert!(
        high <= SCALING_HEADROOM * low.max(floor),
        "#2454: worst |analytic - fd| grew from {low:.4e} at rho=6 to {high:.4e} at rho=15 \
         (ratio {:.1}x) while lambda grew {lambda_ratio:.0}x -- that is a gradient error \
         proportional to lambda\n  {}",
        high / low.max(floor),
        report()
    );

    // 3. Structural: the two penalty-energy spellings must be ONE quantity. This
    //    is the mechanism itself rather than its symptom, and it holds at every
    //    lambda rather than only where the FD oracle is sharp.
    for row in &rows {
        let ratio = row.penalty_energy_criterion / row.penalty_energy_blocks;
        assert!(
            (ratio - 1.0).abs() <= 1.0e-9,
            "#2454: the criterion's penalty energy {:.12e} and the gradient's block sum \
             {:.12e} differ by {:.3e} relative at rho={:.1} -- the gradient is \
             differentiating a penalty the criterion does not apply\n  {}",
            row.penalty_energy_criterion,
            row.penalty_energy_blocks,
            ratio - 1.0,
            row.rho,
            report()
        );
    }
}

/// #2454 GATE: the criterion's two determinant terms must be two views of ONE
/// penalty, so `log|S|₊` may not charge a rank `log|H|` cannot inflate.
///
/// `V = −ℓ + ½β̂ᵀS̃β̂ + ½log|H| − ½log|S|₊` with `H = −∇²ℓ + S̃`, where
/// `S̃(λ) = Π(Σ_k λ_k S_k)Π` is the penalty the reparameterization built and the
/// inner solve minimized. `½log|H|` therefore grows at most `rank(S̃)/2` per
/// unit ρ, no matter how large λ becomes. If `−½log|S|₊` is taken on a penalty
/// of a DIFFERENT rank `r`, the two halves saturate at different rates and
///
/// ```text
/// ∂V/∂ρ → ½(rank(S̃) − r)   per unit ρ, forever,
/// ```
///
/// which for `r > rank(S̃)` is a criterion unbounded below: no interior
/// optimum, no λ=∞ face, and an outer search that arrives at the box edge.
/// That is what this fixture measured — `rank(S̃) = 8` against `r = 10`, giving
/// a `−1` per unit ρ tail that COST followed from ρ≈27 onward.
///
/// The gate is expressed as a rank identity rather than as a tolerance on the
/// criterion, because it is exact at EVERY λ — it does not need the FD oracle
/// to be sharp, or `log|H|` to have saturated, or the search to have reached
/// the tail. Both halves are read off the criterion's own recorded state: the
/// declared rank, and the growth of `log|S|₊` itself, which is affine in ρ at
/// fixed rank with slope exactly `rank(S)`.
///
/// Deliberately NOT read off the gradient's `logdet_s` part. That attribution
/// is route-dependent: the cancellation-free fused route returns
/// `tr(G_ε·Ḣ_k) − det1_k` as one number and books it whole to `logdet_h`, which
/// is honest but makes `logdet_s` zero. The criterion value carries the same
/// fact with no such ambiguity.
#[test]
fn penalty_logdet_ranks_the_same_subspace_the_hessian_carries_2454() {
    let rungs = [6.0_f64, 12.0, 18.0, 24.0];
    let rows = rho_gradient_part_ladder_2454(&rungs, 3e-4);
    assert!(!rows.is_empty(), "ladder produced no rungs");

    let report = || -> String {
        rows.iter()
            .filter(|row| row.coordinate == 0)
            .map(|row| {
                format!(
                    "rho={:5.1} COST={:+.10e} penalized_rank={} logdet_rank={} \
                     log|S|_+={:+.8e}",
                    row.rho, row.cost, row.penalized_rank, row.logdet_rank, row.logdet_value,
                )
            })
            .collect::<Vec<_>>()
            .join("\n  ")
    };

    for row in &rows {
        assert!(
            row.penalized_rank > 0 && row.logdet_rank > 0,
            "the penalty-frame audit did not report both ranks; the gate would then be \
             grading an unarmed instrument\n  {}",
            report()
        );
        // 1. The two ranks are one rank. `H` carries `S̃`; `log|S|₊` must too.
        assert_eq!(
            row.logdet_rank,
            row.penalized_rank,
            "#2454: the criterion's log|S|_+ ranges over rank {} while the penalty H \
             carries has rank {} at rho={:.1}. The LAML pair's asymptotic slope is then \
             {:+.1} per unit rho and the criterion has no interior optimum.\n  {}",
            row.logdet_rank,
            row.penalized_rank,
            row.rho,
            0.5 * (row.penalized_rank as f64 - row.logdet_rank as f64),
            report()
        );
    }

    // 2. The same statement as a RATE, which is what the tail actually feels.
    //    Every ρ coordinate moves together on this ladder, so
    //    `Δlog|S|₊/Δρ = Σ_k λ_k tr(S⁺S_k) = tr(S⁺S) = rank(S)` exactly, and
    //    `½log|H|` can answer at most `rank(S̃)` of it.
    let at = |rho: f64| -> Option<&RhoGradientLadderRow2454> {
        rows.iter().find(|row| row.rho == rho && row.coordinate == 0)
    };
    for pair in rungs.windows(2) {
        let (Some(low), Some(high)) = (at(pair[0]), at(pair[1])) else {
            continue;
        };
        let charged = (high.logdet_value - low.logdet_value) / (pair[1] - pair[0]);
        let carried = high.penalized_rank as f64;
        assert!(
            (charged - carried).abs() <= 1.0e-6 * carried.max(1.0),
            "#2454: over rho {:.1} -> {:.1} the criterion's log|S|_+ grows at \
             {charged:.9} per unit rho while H carries only {carried:.1} penalized \
             directions. The LAML pair therefore keeps a residual slope of {:+.4} per \
             unit rho that no lambda can cancel, and the criterion has no interior \
             optimum.\n  {}",
            pair[0],
            pair[1],
            0.5 * (carried - charged),
            report()
        );
    }
}

/// #2454 MEASUREMENT (reports, never fails): the full part decomposition.
///
/// Prints, per ρ and per ρ-coordinate, each criterion component's analytic
/// gradient part beside the central difference of the component it owns. This is
/// how the defect was localised to `fixed_beta` (its gap equalled the total's to
/// every digit, while `kkt` was exactly zero on both sides), and it is kept so a
/// future desync can be attributed to a channel in one run rather than graded
/// only as a sum.
#[test]
fn zz_measure_rho_gradient_part_decomposition_2454() {
    let rows = rho_gradient_part_ladder_2454(
        &[6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0],
        3e-4,
    );
    for row in &rows {
        if row.coordinate == 0 {
            eprintln!(
                "[zz-parts-2454] rho={:5.1} COST={:+.12e} penalized_rank={} null_dim={} \
                 logdet_rank={} logdet_S={:+.6e} \
                 beta_null_energy={:.4e} energy criterion={:+.12e} blocks={:+.12e} \
                 ratio={:.10}",
                row.rho,
                row.cost,
                row.penalized_rank,
                row.declared_null_dim,
                row.logdet_rank,
                row.logdet_value,
                row.beta_null_energy,
                row.penalty_energy_criterion,
                row.penalty_energy_blocks,
                row.penalty_energy_criterion / row.penalty_energy_blocks,
            );
        }
        eprintln!(
            "[zz-parts-2454]  j={} lambda={:.6e} q_k={:+.10e} lambda_q={:+.10e}",
            row.coordinate,
            row.lambda,
            row.block_quadratic,
            row.lambda * row.block_quadratic,
        );
        for (name, analytic, fd) in [
            ("total     ", row.analytic_total, row.finite_difference_total),
            (
                "fixed_beta",
                row.analytic_fixed_beta,
                row.finite_difference_fixed_beta,
            ),
            (
                "logdet_h  ",
                row.analytic_logdet_h,
                row.finite_difference_logdet_h,
            ),
            (
                "logdet_s  ",
                row.analytic_logdet_s,
                row.finite_difference_logdet_s,
            ),
            ("kkt       ", row.analytic_kkt, row.finite_difference_kkt),
        ] {
            eprintln!(
                "[zz-parts-2454]    {name} an={analytic:+.10e} fd={fd:+.10e} \
                 gap={:+.6e} gap_over_lambda={:+.6e}",
                analytic - fd,
                (analytic - fd) / row.lambda,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// gam#979 — component attribution: the Duchon ψ-derivative OBJECTS, differenced
// one at a time against the rebuilt forward objects.
//
// The criterion-level gates above answer "is ∇V right?". When one fails, the
// answer says nothing about WHICH of the assembled pieces moved: the design
// derivative `∂X/∂ψ`, or the penalty derivatives `∂S_k/∂ψ`. This harness
// differences each analytic block against a central difference of the SAME
// realized object the κ-optimizer rebuilds at `ψ ± h` (`ensure_theta`), and
// reports the Frobenius norms beside the gap — a derivative that is merely
// off by a factor and one that is missing altogether (norm ratio ≈ 0) are
// different defects, and the norm ratio is what tells them apart.
// ---------------------------------------------------------------------------

struct DuchonPsiComponentReport {
    design_rel_gap: f64,
    design_norm_ratio: f64,
    penalty_rel_gaps: Vec<f64>,
    penalty_norm_ratios: Vec<f64>,
}

fn frobenius(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn duchon_psi_component_report(label: &str, n: usize) -> DuchonPsiComponentReport {
    let fixture = build_iso_kappa_fixture(label, n, LikelihoodSpec::gaussian_identity(), false);
    assert_eq!(fixture.psi_dim, 1, "{label}: one isotropic ψ axis");
    let mut cache = fixture.cache();
    let theta_dim = fixture.rho_dim + fixture.psi_dim;
    // ψ = 0 is the shipped `length_scale=1` seed, where the large-scale line
    // searches fail.
    let theta = Array1::<f64>::zeros(theta_dim);
    let psi_slot = fixture.rho_dim;
    cache.ensure_theta(&theta).expect("ensure_theta at the seed");
    let p_total = cache.design().design.ncols();
    let derivs = crate::spatial_psi_bridge::build_block_spatial_psi_derivatives(
        fixture.data.view(),
        cache.spec(),
        cache.design(),
    )
    .expect("spatial psi derivatives build")
    .expect("spatial psi derivatives present");
    assert_eq!(derivs.len(), 1, "{label}: one ψ block");
    let deriv = &derivs[0];
    let x_psi_an: Array2<f64> = if deriv.x_psi.nrows() == n && deriv.x_psi.ncols() == p_total {
        deriv.x_psi.clone()
    } else {
        let op = deriv
            .implicit_operator
            .as_ref()
            .expect("dense x_psi placeholder implies an implicit operator");
        crate::custom_family::CustomFamilyPsiDerivativeOperator::as_materializable(op.as_ref())
            .expect("materializable implicit operator")
            .materialize_first(deriv.implicit_axis)
            .expect("materialize ∂X/∂ψ")
    };
    assert_eq!(x_psi_an.dim(), (n, p_total), "{label}: ∂X/∂ψ shape");
    // The criterion never materializes ∂X/∂ψ: it drives the operator's
    // streaming products. Pin them to the materialized block here so a defect
    // in one streaming branch cannot hide behind a passing materialized gate.
    if let Some(op) = deriv.implicit_operator.as_ref() {
        let op = op.as_ref();
        let axis = deriv.implicit_axis;
        let u = Array1::from_shape_fn(p_total, |j| ((j as f64) * 0.731 + 0.2).sin());
        let v = Array1::from_shape_fn(n, |i| ((i as f64) * 0.377 - 0.1).cos());
        let fwd = op.forward_mul(axis, &u.view()).expect("operator forward_mul");
        let fwd_dense = x_psi_an.dot(&u);
        let fwd_gap = (&fwd - &fwd_dense).mapv(f64::abs).sum() / fwd_dense.mapv(f64::abs).sum().max(1e-300);
        let tr = op.transpose_mul(axis, &v.view()).expect("operator transpose_mul");
        let tr_dense = x_psi_an.t().dot(&v);
        let tr_gap = (&tr - &tr_dense).mapv(f64::abs).sum() / tr_dense.mapv(f64::abs).sum().max(1e-300);
        let chunk = op.row_chunk_first(axis, 0..n).expect("operator row_chunk_first");
        let chunk_gap = frobenius(&(&chunk - &x_psi_an)) / frobenius(&x_psi_an).max(1e-300);
        eprintln!(
            "[{label} COMPONENT] streaming vs materialized ∂X/∂ψ: forward_mul rel={fwd_gap:.3e} \
             transpose_mul rel={tr_gap:.3e} row_chunk rel={chunk_gap:.3e}"
        );
        assert!(fwd_gap < 1e-10, "{label}: forward_mul disagrees with materialized ∂X/∂ψ by {fwd_gap:.3e}");
        assert!(tr_gap < 1e-10, "{label}: transpose_mul disagrees with materialized ∂X/∂ψ by {tr_gap:.3e}");
        assert!(chunk_gap < 1e-10, "{label}: row_chunk_first disagrees with materialized ∂X/∂ψ by {chunk_gap:.3e}");
    }
    // The spatial bridge emits the penalty ψ-derivatives as dense per-penalty
    // components (`s_psi_components`); the `PenaltyMatrix` form is only
    // populated by the transformation-normal assembler.
    let penalty_components: Vec<(usize, Array2<f64>)> = deriv
        .s_psi_components
        .as_ref()
        .expect("dense penalty psi components from the spatial bridge")
        .iter()
        .map(|(idx, component)| (*idx, component.clone()))
        .collect();
    assert!(!penalty_components.is_empty(), "{label}: at least one penalty ψ block");

    // The forward objects at ψ ± h, through the κ-optimizer's own realizer.
    let realize = |cache: &mut SingleBlockExactJointDesignCache<'_>,
                   psi: f64|
     -> (Array2<f64>, Vec<Array2<f64>>) {
        let mut probe = theta.clone();
        probe[psi_slot] = psi;
        cache.ensure_theta(&probe).expect("ensure_theta at ψ ± h");
        let design = cache.design();
        let x = design.design.to_dense();
        let penalties = penalty_components
            .iter()
            .map(|(idx, _)| design.penalties[*idx].to_penalty_matrix(p_total).to_dense())
            .collect();
        (x, penalties)
    };
    // Two central-difference steps, ratio 2: a truncation-limited estimate
    // moves by 4× between them, a defect does not.
    let mut best_design_gap = f64::INFINITY;
    let mut design_norm_ratio = f64::NAN;
    let mut best_penalty_gaps = vec![f64::INFINITY; penalty_components.len()];
    let mut penalty_norm_ratios = vec![f64::NAN; penalty_components.len()];
    for &h in &[2.0e-3_f64, 1.0e-3] {
        let (x_plus, s_plus) = realize(&mut cache, h);
        let (x_minus, s_minus) = realize(&mut cache, -h);
        let x_fd = (&x_plus - &x_minus) / (2.0 * h);
        let design_gap = frobenius(&(&x_psi_an - &x_fd)) / frobenius(&x_fd).max(1e-300);
        eprintln!(
            "[{label} COMPONENT] h={h:.1e} design |an|={:.6e} |fd|={:.6e} rel_gap={design_gap:.3e}",
            frobenius(&x_psi_an),
            frobenius(&x_fd)
        );
        if design_gap < best_design_gap {
            best_design_gap = design_gap;
            design_norm_ratio = frobenius(&x_psi_an) / frobenius(&x_fd).max(1e-300);
        }
        for (k, (idx, analytic)) in penalty_components.iter().enumerate() {
            let s_fd = (&s_plus[k] - &s_minus[k]) / (2.0 * h);
            let gap = frobenius(&(analytic - &s_fd)) / frobenius(&s_fd).max(1e-300);
            eprintln!(
                "[{label} COMPONENT] h={h:.1e} penalty[{idx}] |an|={:.6e} |fd|={:.6e} rel_gap={gap:.3e}",
                frobenius(analytic),
                frobenius(&s_fd)
            );
            if gap < best_penalty_gaps[k] {
                best_penalty_gaps[k] = gap;
                penalty_norm_ratios[k] = frobenius(analytic) / frobenius(&s_fd).max(1e-300);
            }
        }
    }
    // Restore the seed so the cache is left where it was found.
    cache.ensure_theta(&theta).expect("ensure_theta back at the seed");
    DuchonPsiComponentReport {
        design_rel_gap: best_design_gap,
        design_norm_ratio,
        penalty_rel_gaps: best_penalty_gaps,
        penalty_norm_ratios,
    }
}

fn assert_duchon_psi_components(label: &str, n: usize) {
    let report = duchon_psi_component_report(label, n);
    eprintln!(
        "[{label} COMPONENT SUMMARY] design rel_gap={:.3e} norm_ratio={:.3e} penalties rel_gaps={:?} norm_ratios={:?}",
        report.design_rel_gap,
        report.design_norm_ratio,
        report.penalty_rel_gaps,
        report.penalty_norm_ratios
    );
    // A central difference at h = 1e-3 on an analytic interpoland is accurate
    // to ~1e-6 relative; 1e-4 leaves room for the evaluator's own rounding at
    // the tiny-magnitude high-dimensional kernels without admitting a defect.
    let tol = 1e-4;
    assert!(
        report.design_rel_gap < tol,
        "{label}: analytic ∂X/∂ψ differs from the rebuilt design by {:.3e} (norm ratio {:.3e})",
        report.design_rel_gap,
        report.design_norm_ratio
    );
    for (k, gap) in report.penalty_rel_gaps.iter().enumerate() {
        assert!(
            *gap < tol,
            "{label}: analytic ∂S_{k}/∂ψ differs from the rebuilt penalty by {gap:.3e} (norm ratio {:.3e})",
            report.penalty_norm_ratios[k]
        );
    }
}

/// gam#979 — the `Linear` control at 3-D, power 9.
#[test]
fn duchon_hybrid_psi_components_match_fd_linear_power9_3d() {
    assert_duchon_psi_components("duchon_gaussian_linear_power9_3d", 240);
}

/// gam#979 — the shipped null-space order at 3-D, power 9.
#[test]
fn duchon_hybrid_psi_components_match_fd_order0_power9_3d() {
    assert_duchon_psi_components("duchon_gaussian_order0_power9_3d", 240);
}

/// gam#979 — the `Linear` control at the benchmark shape.
#[test]
fn duchon_hybrid_psi_components_match_fd_linear_power9_16d() {
    assert_duchon_psi_components("duchon_gaussian_linear_power9_16d_centers24", 1700);
}

/// gam#979 — the benchmark shape itself: sixteen axes, 24 centers, power 9,
/// constant-only null space, radial-profile row counts.
#[test]
fn duchon_hybrid_psi_components_match_fd_order0_power9_16d() {
    assert_duchon_psi_components("duchon_gaussian_order0_power9_16d_centers24", 1700);
}

}
