//! gam#979 — the exact-joint outer gradient of the transformation-normal
//! criterion, differenced against the criterion itself.
//!
//! `fit_transformation_normal` hands the spatial κ-optimizer a closure that
//! evaluates the profiled LAML criterion `V(ρ, ψ)` and its analytic gradient.
//! Every ψ-derivative ingredient of that gradient carries its own
//! finite-difference gate — the CTN ψ terms at fixed β, the `G_x(κ)` penalty
//! channel, the Duchon radial jets — but the ASSEMBLED gradient the line search
//! actually follows had none. The large-scale CTN preprocessor
//! (`duchon(pc1..pc16, centers=24, order=0, power=9, length_scale=1)`) then
//! spent every strong-Wolfe search walking a direction whose analytic slope
//! was `−74` while the criterion rose along it, fell back to Armijo
//! backtracking at ~55 evaluations per iteration, and never finished inside
//! its budget. That is what this gate measures: at the production chart's
//! orders, the ψ component of `∇V` against a Ridders-extrapolated central
//! difference of `V`, through the SAME geometry constructor the optimizer's
//! cache uses (`build_transformation_exact_geometry`).
//!
//! Every ρ component is differenced in the same pass, so a ρ↔ψ cross-term
//! defect cannot masquerade as a ψ-only one.

#![cfg(test)]

use super::*;
use gam_linalg::test_support::fd_checker::{FdVerdict, RiddersConfig, ridders_derivative};
use gam_terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability,
};
use gam_terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, set_spatial_length_scale,
    spatial_term_psi_to_length_scale_and_aniso,
};

/// One transformation-normal spatial fixture: the frozen boot chart the
/// κ-optimizer realizes designs from, the response basis it builds once, and
/// everything `fit_transformation_normal`'s `make_family` closure captures.
struct CtnKappaFixture {
    data: Array2<f64>,
    response: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    boot_spec: TermCollectionSpec,
    spatial_terms: Vec<usize>,
    rho_dim: usize,
    config: TransformationNormalConfig,
    resp_val: Array2<f64>,
    resp_deriv: Array2<f64>,
    resp_penalties: Vec<Array2<f64>>,
    resp_knots: Array1<f64>,
    resp_transform: Array2<f64>,
}

/// Deterministic, non-symmetric covariate cloud in `d` dimensions and a
/// response whose conditional law genuinely moves with every axis, so no ψ
/// direction is degenerate at the seed.
fn synthetic_ctn_rows(n: usize, d: usize) -> (Array2<f64>, Array1<f64>) {
    let mut data = Array2::<f64>::zeros((n, d));
    let mut response = Array1::<f64>::zeros(n);
    for i in 0..n {
        let u = i as f64 / (n as f64 - 1.0);
        for a in 0..d {
            let phase = 0.618_033_988_749_894_9 * (a as f64 + 1.0);
            data[[i, a]] = 1.6 * ((i as f64 * phase).fract() - 0.5) + 0.2 * (7.0 * u + a as f64).sin();
        }
        let x0 = data[[i, 0]];
        let x1 = if d > 1 { data[[i, 1]] } else { 0.0 };
        let x2 = if d > 2 { data[[i, 2]] } else { 0.0 };
        let location = 0.8 * (2.0 * x0).sin() + 0.4 * x1 + 0.3 * x0 * x2;
        let spread = (0.35 * x1).exp();
        let noise = 0.7 * (3.7 * i as f64 + 1.0).sin() + 0.3 * (1.3 * i as f64).cos();
        response[i] = location + spread * noise;
    }
    (data, response)
}

fn build_fixture(
    order: DuchonNullspaceOrder,
    power: f64,
    d: usize,
    n: usize,
    centers: usize,
) -> CtnKappaFixture {
    let (data, response) = synthetic_ctn_rows(n, d);
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_pcs".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: centers,
                    },
                    length_scale: Some(1.0),
                    power,
                    nullspace_order: order,
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
    // Mirror `fit_transformation_normal`: a bootstrap build freezes the chart,
    // the response complexity is resolved against the bootstrap tensor width,
    // and the response basis is built ONCE — it never moves with κ.
    let boot_design =
        build_term_collection_design(data.view(), &spec).expect("bootstrap covariate design");
    let boot_spec = freeze_term_collection_from_design(&spec, &boot_design)
        .expect("freeze bootstrap covariate chart");
    let base_config = TransformationNormalConfig::default();
    let config = TransformationNormalConfig {
        response_num_internal_knots: effective_response_num_internal_knots(
            &base_config,
            response.len(),
            boot_design.design.ncols(),
            response.view(),
        ),
        ..base_config
    };
    let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
        build_response_basis(&response, &config).expect("response basis");
    let spatial_terms = spatial_length_scale_term_indices(&boot_spec);
    assert_eq!(spatial_terms, vec![0], "the Duchon term must enroll one κ axis");
    let mut fixture = CtnKappaFixture {
        data,
        response,
        weights,
        offset,
        boot_spec,
        spatial_terms,
        rho_dim: 0,
        config,
        resp_val,
        resp_deriv,
        resp_penalties,
        resp_knots,
        resp_transform,
    };
    // The ρ dimension is whatever the family's own penalty layout says it is
    // (covariate, response, double), read off a probe family over the
    // bootstrap design exactly as `fit_transformation_normal` does.
    let probe_family = fixture
        .make_family(&boot_design)
        .expect("probe transformation family");
    fixture.rho_dim = probe_family
        .penalty_scale_log_lambdas()
        .expect("probe penalty-scale seed")
        .len();
    assert!(fixture.rho_dim >= 2, "the tensor penalty layout must carry several ρ axes");
    fixture
}

impl CtnKappaFixture {
    fn make_family(&self, cov_design: &TermCollectionDesign) -> Result<TransformationNormalFamily, String> {
        let effective_offset = cov_design
            .compose_offset(self.offset.view(), "transformation-normal FD gate")
            .map_err(|error| error.to_string())?;
        TransformationNormalFamily::from_prebuilt_response_basis(
            &self.response,
            self.resp_val.clone(),
            self.resp_deriv.clone(),
            self.resp_penalties.clone(),
            self.resp_knots.clone(),
            self.config.response_degree,
            self.resp_transform.clone(),
            &self.weights,
            &effective_offset,
            cov_design.design.clone(),
            cov_design
                .penalties
                .iter()
                .map(|bp| bp.to_penalty_matrix(cov_design.design.ncols()))
                .collect(),
            &self.config,
            None,
        )
    }

    /// The geometry at `theta = (ρ, ψ)`, realized the way the κ-optimizer
    /// realizes it: the frozen boot chart with its length scale moved to
    /// `exp(−ψ)` (`spatial_term_psi_to_length_scale_and_aniso`), rebuilt, then
    /// frozen and rebuilt again through the production constructor.
    fn geometry_at(&self, theta: &Array1<f64>) -> TransformationExactGeometryCache {
        let rho = theta.slice(s![..self.rho_dim]).to_owned();
        let hyper_values = theta.slice(s![self.rho_dim..]).to_owned();
        let psi = hyper_values.as_slice().expect("contiguous ψ");
        let (length_scale, aniso) = spatial_term_psi_to_length_scale_and_aniso(psi);
        assert!(aniso.is_none(), "isotropic fixture must decode a scalar ψ");
        let mut spec_at_psi = self.boot_spec.clone();
        set_spatial_length_scale(
            &mut spec_at_psi,
            self.spatial_terms[0],
            length_scale.expect("scalar ψ decodes to a length scale"),
        )
        .expect("set Duchon length scale");
        let design_at_psi = build_term_collection_design(self.data.view(), &spec_at_psi)
            .expect("covariate design at ψ");
        let effective_spec = freeze_term_collection_from_design(&spec_at_psi, &design_at_psi)
            .expect("freeze transformation geometry");
        let key = transformation_spatial_geometry_key(&effective_spec, &self.spatial_terms)
            .expect("transformation geometry key");
        build_transformation_exact_geometry(
            self.data.view(),
            effective_spec,
            key,
            &rho,
            &hyper_values,
            &|design| self.make_family(design),
        )
        .expect("transformation exact geometry")
    }

    /// One criterion evaluation at `theta`, from `anchor` — the coefficient
    /// mode the probe restarts from. `fit_transformation_normal` freezes the
    /// mode selected at the first derivative-bearing evaluation as the branch
    /// anchor and restarts every later trial from it, because the finite-
    /// support criterion has several coefficient modes and a cold start at
    /// `ψ + h` can land on a different one than a cold start at `ψ`: the
    /// difference quotient then measures a mode switch, not a derivative. The
    /// gate therefore anchors every probe of one ladder to the mode converged
    /// at the ladder's base point, exactly as the optimizer's branch does.
    fn evaluate(
        &self,
        theta: &Array1<f64>,
        options: &BlockwiseFitOptions,
        anchor: Option<&CustomFamilyWarmStart>,
        eval_mode: gam_problem::EvalMode,
    ) -> CustomFamilyJointHyperModeSelection {
        let geometry = self.geometry_at(theta);
        let rho = theta.slice(s![..self.rho_dim]).to_owned();
        evaluate_custom_family_joint_hyper_best_mode_shared(
            &geometry.family,
            &geometry.blocks,
            options,
            &rho,
            Arc::clone(&geometry.hyper_layout),
            &[anchor.cloned()],
            eval_mode,
        )
        .expect("transformation exact joint evaluation")
    }

    /// The inner cycle budget `fit_transformation_normal` scopes to this
    /// tensor width, so a probe converges under the same cap production does.
    fn options(&self) -> BlockwiseFitOptions {
        let geometry = self.geometry_at(&Array1::<f64>::zeros(self.rho_dim + 1));
        let realized_p_total = geometry.family.p_total();
        let ctn_inner_cap = CTN_INNER_MAX_CYCLES_BASE
            .saturating_add(realized_p_total.saturating_mul(CTN_INNER_MAX_CYCLES_PER_DIM))
            .min(CTN_INNER_MAX_CYCLES_CEILING);
        let defaults = BlockwiseFitOptions::default();
        BlockwiseFitOptions {
            // Far tighter than any gradient gap the oracle resolves, so the
            // differenced criterion is the criterion.
            inner_tol: 1e-10,
            inner_max_cycles: defaults.inner_max_cycles.min(ctn_inner_cap),
            compute_covariance: false,
            ..defaults
        }
    }
}

struct CtnKappaFdReport {
    pass: bool,
    violations: Vec<String>,
    worst_psi_rel: f64,
}

/// The shipped large-scale CTN preprocessor chart: hybrid Duchon–Matérn with
/// the constant-only null space and spectral power 9, in three PC dimensions.
#[test]
fn ctn_exact_joint_gradient_matches_fd_duchon_order0_power9() {
    let fixture = build_fixture(DuchonNullspaceOrder::Zero, 9.0, 3, 240, 10);
    let CtnKappaFdReport { pass, violations, worst_psi_rel } =
        ctn_kappa_fd_driver("ctn_duchon_order0_power9_3d", &fixture);
    assert!(
        pass,
        "transformation-normal exact-joint gradient (order=0, power=9) disagrees with the \
         criterion; worst_psi_rel={worst_psi_rel:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// The `Linear` null-space control at the same power: the order the rest of
/// the Duchon ψ gates pin.
#[test]
fn ctn_exact_joint_gradient_matches_fd_duchon_linear_power9() {
    let fixture = build_fixture(DuchonNullspaceOrder::Linear, 9.0, 3, 240, 10);
    let CtnKappaFdReport { pass, violations, worst_psi_rel } =
        ctn_kappa_fd_driver("ctn_duchon_linear_power9_3d", &fixture);
    assert!(
        pass,
        "transformation-normal exact-joint gradient (order=1, power=9) disagrees with the \
         criterion; worst_psi_rel={worst_psi_rel:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// The benchmark's own shape: sixteen PC axes, 24 farthest-point centers, and
/// enough rows that the row-center sweeps take the certified radial-profile
/// path. This is the chart `gam fit --transformation-normal` runs in
/// `bench/large_scale`, where the shipped binary's line searches fail.
#[test]
fn ctn_exact_joint_gradient_matches_fd_duchon_order0_power9_16d() {
    let fixture = build_fixture(DuchonNullspaceOrder::Zero, 9.0, 16, 1200, 24);
    let CtnKappaFdReport { pass, violations, worst_psi_rel } =
        ctn_kappa_fd_driver("ctn_duchon_order0_power9_16d", &fixture);
    assert!(
        pass,
        "transformation-normal exact-joint gradient (order=0, power=9, 16-D) disagrees with \
         the criterion; worst_psi_rel={worst_psi_rel:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// The `Linear` null-space control at the benchmark shape.
#[test]
fn ctn_exact_joint_gradient_matches_fd_duchon_linear_power9_16d() {
    let fixture = build_fixture(DuchonNullspaceOrder::Linear, 9.0, 16, 1200, 24);
    let CtnKappaFdReport { pass, violations, worst_psi_rel } =
        ctn_kappa_fd_driver("ctn_duchon_linear_power9_16d", &fixture);
    assert!(
        pass,
        "transformation-normal exact-joint gradient (order=1, power=9, 16-D) disagrees with \
         the criterion; worst_psi_rel={worst_psi_rel:.3e}\n  {}",
        violations.join("\n  ")
    );
}
