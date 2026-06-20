use super::*;
use crate::custom_family::BlockWorkingSet;
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::solver::gauge::Gauge;
use crate::types::{LinkComponent, MixtureLinkSpec, SasLinkSpec};
use faer::sparse::{SparseColMat, Triplet};
use ndarray::{Array1, array};

#[derive(Clone, Copy)]
struct SurvivalLsLocationScaleRow {
    eta_location: f64,
    eta_logscale: f64,
    entry_index: f64,
    exit_index: f64,
    exit_index_derivative: f64,
    event: f64,
    weight: f64,
}

struct SurvivalLsLocationScaleNllProgram<'a> {
    inverse_link: &'a InverseLink,
    deriv_log_scale: f64,
    row: SurvivalLsLocationScaleRow,
}

fn survival_ls_log_survival_stack(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<[f64; 5], String> {
    let (log_s, r, dr, ddr, dddr) =
        SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
            inverse_link,
            eta,
        )?;
    Ok([log_s, -r, -dr, -ddr, -dddr])
}

fn survival_ls_log_pdf_stack(
    inverse_link: &InverseLink,
    eta: f64,
    deriv_log_scale: f64,
) -> Result<[f64; 5], String> {
    let (log_pdf, d1, d2, d3, d4) =
        SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
            inverse_link,
            eta,
            deriv_log_scale,
        )?;
    Ok([log_pdf, d1, d2, d3, d4])
}

fn survival_ls_positive_log_stack(value: f64) -> [f64; 5] {
    let (log_v, d1, d2, d3, d4) = SurvivalLocationScaleFamily::logwith_derivatives_positive(value);
    [log_v, d1, d2, d3, d4]
}

impl SurvivalExactRowKernel {
    #[inline]
    fn location_scale_nll_tower(
        self,
        row: SurvivalLsLocationScaleRow,
    ) -> crate::families::jet_tower::Tower4<2> {
        use crate::families::jet_tower::Tower4;

        let eta_location = Tower4::<2>::variable(row.eta_location, 0);
        let eta_logscale = Tower4::<2>::variable(row.eta_logscale, 1);
        let inv_sigma = (-eta_logscale).exp();
        let q_entry = (Tower4::<2>::constant(row.entry_index) - eta_location) * inv_sigma;
        let q_exit = (Tower4::<2>::constant(row.exit_index) - eta_location) * inv_sigma;
        let g = Tower4::<2>::constant(row.exit_index_derivative) * inv_sigma;

        let mut nll = q_entry
            .compose_unary([self.log_s0, -self.r0, -self.dr0, -self.ddr0, -self.dddr0])
            .scale(row.weight);

        let censored_weight = row.weight * (1.0 - row.event);
        if censored_weight != 0.0 {
            nll = nll
                + q_exit
                    .compose_unary([self.log_s1, -self.r1, -self.dr1, -self.ddr1, -self.dddr1])
                    .scale(-censored_weight);
        }

        let event_weight = row.weight * row.event;
        if event_weight != 0.0 {
            nll = nll
                + q_exit
                    .compose_unary([
                        self.logphi1,
                        self.dlogphi1,
                        self.d2logphi1,
                        self.d3logphi1,
                        self.d4logphi1,
                    ])
                    .scale(-event_weight)
                + g.compose_unary([
                    self.log_g,
                    self.d_log_g,
                    self.d2_log_g,
                    self.d3_log_g,
                    self.d4_log_g,
                ])
                .scale(-event_weight);
        }

        nll
    }
}

impl crate::families::jet_tower::RowNllProgram<2> for SurvivalLsLocationScaleNllProgram<'_> {
    fn n_rows(&self) -> usize {
        1
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        if row != 0 {
            return Err("survival LS location-scale jet row out of range".to_string());
        }
        Ok([self.row.eta_location, self.row.eta_logscale])
    }

    fn row_nll(
        &self,
        row: usize,
        p: &[crate::families::jet_tower::Tower4<2>; 2],
    ) -> Result<crate::families::jet_tower::Tower4<2>, String> {
        use crate::families::jet_tower::Tower4;

        if row != 0 {
            return Err("survival LS location-scale jet row out of range".to_string());
        }
        if self.row.weight <= 0.0 {
            return Ok(Tower4::<2>::zero());
        }

        let eta_location = p[0];
        let eta_logscale = p[1];
        let inv_sigma = (-eta_logscale).exp();
        let q_entry = (self.row.entry_index - eta_location.v) * inv_sigma.v;
        let q_exit = (self.row.exit_index - eta_location.v) * inv_sigma.v;
        let g = self.row.exit_index_derivative * inv_sigma.v;

        let stack_entry = survival_ls_log_survival_stack(self.inverse_link, q_entry)?;
        let mut kernel = SurvivalExactRowKernel {
            w: self.row.weight,
            d: self.row.event,
            log_s0: stack_entry[0],
            r0: -stack_entry[1],
            dr0: -stack_entry[2],
            ddr0: -stack_entry[3],
            dddr0: -stack_entry[4],
            log_s1: 0.0,
            r1: 0.0,
            dr1: 0.0,
            ddr1: 0.0,
            dddr1: 0.0,
            logphi1: 0.0,
            dlogphi1: 0.0,
            d2logphi1: 0.0,
            d3logphi1: 0.0,
            d4logphi1: 0.0,
            log_g: 0.0,
            d_log_g: 0.0,
            d2_log_g: 0.0,
            d3_log_g: 0.0,
            d4_log_g: 0.0,
        };

        let censored_weight = self.row.weight * (1.0 - self.row.event);
        if censored_weight != 0.0 {
            let stack_exit = survival_ls_log_survival_stack(self.inverse_link, q_exit)?;
            kernel.log_s1 = stack_exit[0];
            kernel.r1 = -stack_exit[1];
            kernel.dr1 = -stack_exit[2];
            kernel.ddr1 = -stack_exit[3];
            kernel.dddr1 = -stack_exit[4];
        }

        let event_weight = self.row.weight * self.row.event;
        if event_weight != 0.0 {
            let stack_pdf =
                survival_ls_log_pdf_stack(self.inverse_link, q_exit, self.deriv_log_scale)?;
            kernel.logphi1 = stack_pdf[0];
            kernel.dlogphi1 = stack_pdf[1];
            kernel.d2logphi1 = stack_pdf[2];
            kernel.d3logphi1 = stack_pdf[3];
            kernel.d4logphi1 = stack_pdf[4];
            let stack_g = survival_ls_positive_log_stack(g);
            kernel.log_g = stack_g[0];
            kernel.d_log_g = stack_g[1];
            kernel.d2_log_g = stack_g[2];
            kernel.d3_log_g = stack_g[3];
            kernel.d4_log_g = stack_g[4];
        }

        Ok(kernel.location_scale_nll_tower(self.row))
    }
}

fn sparse_design_from_dense(dense: &Array2<f64>) -> DesignMatrix {
    let mut triplets = Vec::new();
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            let value = dense[[i, j]];
            if value != 0.0 {
                triplets.push(Triplet::new(i, j, value));
            }
        }
    }
    DesignMatrix::from(
        SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
            .expect("build sparse design"),
    )
}

/// Parity test for issue #410: the survival covariate spatial-ψ derivative
/// blocks (`Static` template) are produced by the *shared* exact-derivative
/// engine, not a survival-local re-implementation. A custom/built-in family
/// and the survival family with identical anisotropic-Matérn specs must
/// therefore yield bit-identical ψ-derivative blocks — design embedding,
/// penalty components, anisotropic cross-rows, and implicit-operator action.
#[test]
fn survival_static_spatial_psi_blocks_match_shared_engine() {
    use crate::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use crate::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};

    let n = 12usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (0.41 * i as f64).sin() + 0.15 * x0;
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
    }

    let spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            name: "spatial".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::EqualMass { num_centers: 6 },
                    length_scale: 0.7,
                    nu: MaternNu::ThreeHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: Some(vec![0.0, 0.0]),
                    nullspace_shrinkage_survived: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let base_design =
        build_term_collection_design(data.view(), &spec).expect("build base spatial design");
    let resolvedspec =
        freeze_term_collection_from_design(&spec, &base_design).expect("freeze spatial term spec");
    let resolved_design = build_term_collection_design(data.view(), &resolvedspec)
        .expect("rebuild frozen spatial design");

    // Built-in / canonical path: the shared exact-derivative engine.
    let shared = crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives(
        data.view(),
        &resolvedspec,
        &resolved_design,
    )
    .expect("shared engine spatial psi derivatives")
    .expect("anisotropic spatial derivative rows from shared engine");

    // Survival consumer path: the Static adapter must delegate to the same engine.
    let survival = build_survival_covariate_block_psi_derivatives(
        data.view(),
        &resolvedspec,
        &resolved_design,
        &SurvivalCovariateTermBlockTemplate::Static,
    )
    .expect("survival static spatial psi derivatives")
    .expect("anisotropic spatial derivative rows from survival adapter");

    assert_eq!(
        shared.len(),
        survival.len(),
        "shared engine and survival adapter must emit the same number of ψ blocks"
    );

    let psi_dim = shared.len();
    for (axis, (a, b)) in shared.iter().zip(survival.iter()).enumerate() {
        assert_eq!(
            a.penalty_index, b.penalty_index,
            "penalty_index axis {axis}"
        );
        assert_eq!(
            a.implicit_axis, b.implicit_axis,
            "implicit_axis axis {axis}"
        );
        assert_eq!(
            a.implicit_group_id, b.implicit_group_id,
            "implicit_group_id axis {axis}"
        );
        assert_eq!(a.x_psi, b.x_psi, "x_psi axis {axis}");
        assert_eq!(
            a.s_psi_components, b.s_psi_components,
            "s_psi_components axis {axis}"
        );
        assert_eq!(a.x_psi_psi, b.x_psi_psi, "x_psi_psi axis {axis}");
        assert_eq!(
            a.s_psi_psi_components, b.s_psi_psi_components,
            "s_psi_psi_components axis {axis}"
        );

        // Implicit-operator action parity: identical embedding and identical
        // forward/transpose maps on deterministic probe vectors.
        match (a.implicit_operator.as_ref(), b.implicit_operator.as_ref()) {
            (Some(op_a), Some(op_b)) => {
                assert_eq!(op_a.n_data(), op_b.n_data(), "operator n_data axis {axis}");
                assert_eq!(op_a.p_out(), op_b.p_out(), "operator p_out axis {axis}");
                let p = op_a.p_out();
                let u: Array1<f64> = (0..p)
                    .map(|j| 0.3 + 0.11 * (j as f64) - 0.07 * ((axis + j) as f64).cos())
                    .collect();
                for probe_axis in 0..psi_dim {
                    let fwd_a = op_a
                        .forward_mul(probe_axis, &u.view())
                        .expect("shared forward_mul");
                    let fwd_b = op_b
                        .forward_mul(probe_axis, &u.view())
                        .expect("survival forward_mul");
                    assert_eq!(
                        fwd_a, fwd_b,
                        "forward_mul mismatch block {axis} probe-axis {probe_axis}"
                    );
                    let nd = op_a.n_data();
                    let v: Array1<f64> = (0..nd)
                        .map(|r| 0.2 - 0.05 * (r as f64) + 0.13 * ((r + probe_axis) as f64).sin())
                        .collect();
                    let tr_a = op_a
                        .transpose_mul(probe_axis, &v.view())
                        .expect("shared transpose_mul");
                    let tr_b = op_b
                        .transpose_mul(probe_axis, &v.view())
                        .expect("survival transpose_mul");
                    assert_eq!(
                        tr_a, tr_b,
                        "transpose_mul mismatch block {axis} probe-axis {probe_axis}"
                    );
                }
            }
            (None, None) => {}
            _ => panic!("implicit_operator presence diverged at axis {axis}"),
        }
    }
}

fn test_link_wiggle_metadata(beta_link_wiggle: &Array1<f64>) -> (Array1<f64>, usize) {
    let seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    for degree in [2usize, 3, 1] {
        for num_internal_knots in 0..=8 {
            let cfg = WiggleBlockConfig {
                degree,
                num_internal_knots,
                penalty_order: 2,
                double_penalty: false,
            };
            if let Ok((block, knots)) =
                crate::families::wiggle::buildwiggle_block_input_from_seed(seed.view(), &cfg)
                && block.design.ncols() == beta_link_wiggle.len()
            {
                return (knots, degree);
            }
        }
    }
    panic!(
        "could not synthesize valid link wiggle metadata for {} coefficients",
        beta_link_wiggle.len()
    );
}

fn test_survival_fit(
    beta_time: Array1<f64>,
    beta_threshold: Array1<f64>,
    beta_log_sigma: Array1<f64>,
    beta_link_wiggle: Option<Array1<f64>>,
) -> UnifiedFitResult {
    let lambdas_linkwiggle = beta_link_wiggle.as_ref().map(|_| Array1::zeros(0));
    let (link_wiggle_knots, link_wiggle_degree) = beta_link_wiggle
        .as_ref()
        .map(|beta| {
            let (knots, degree) = test_link_wiggle_metadata(beta);
            (Some(knots), Some(degree))
        })
        .unwrap_or((None, None));
    survival_fit_from_parts(SurvivalLocationScaleFitResultParts {
        beta_time,
        beta_threshold,
        beta_log_sigma,
        beta_link_wiggle,
        link_wiggle_knots,
        link_wiggle_degree,
        lambdas_time: Array1::zeros(0),
        lambdas_threshold: Array1::zeros(0),
        lambdas_log_sigma: Array1::zeros(0),
        lambdas_linkwiggle,
        log_likelihood: 0.0,
        reml_score: 0.0,
        stable_penalty_term: 0.0,
        penalized_objective: 0.0,
        used_device: false,
        outer_iterations: 0,
        outer_gradient_norm: None,
        outer_converged: true,
        covariance_conditional: None,
        geometry: None,
    })
    .expect("valid survival test fit")
}

fn survival_exact_newton_test_family() -> SurvivalLocationScaleFamily {
    SurvivalLocationScaleFamily {
        n: 3,
        y: array![1.0, 0.0, 1.0],
        w: array![1.0, 0.8, 1.2],
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: 1e-8,
        x_time_entry: Arc::new(array![[1.0], [1.0], [1.0]]),
        x_time_exit: Arc::new(array![[1.2], [0.9], [1.4]]),
        x_time_deriv: Arc::new(array![[1.0], [1.0], [1.0]]),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [0.4],
            [-0.6]
        ])),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [-0.3],
            [0.5]
        ])),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: crate::resource::ResourcePolicy::default_library(),
    }
}

fn survival_exact_newton_test_states(
    family: &SurvivalLocationScaleFamily,
    beta_t: f64,
    beta_thr: f64,
    beta_ls: f64,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    let mut eta_time = Array1::<f64>::zeros(3 * n);
    for i in 0..n {
        eta_time[i] = family.x_time_exit[[i, 0]] * beta_t;
        eta_time[n + i] = family.x_time_entry[[i, 0]] * beta_t;
        eta_time[2 * n + i] = family.x_time_deriv[[i, 0]] * beta_t;
    }
    let eta_thr =
        Array1::from_iter((0..n).map(|i| family.x_threshold.dot_row(i, &array![beta_thr])));
    let eta_ls = Array1::from_iter((0..n).map(|i| family.x_log_sigma.dot_row(i, &array![beta_ls])));
    vec![
        ParameterBlockState {
            beta: array![beta_t],
            eta: eta_time,
        },
        ParameterBlockState {
            beta: array![beta_thr],
            eta: eta_thr,
        },
        ParameterBlockState {
            beta: array![beta_ls],
            eta: eta_ls,
        },
    ]
}

/// Total data-fit log-likelihood `ℓ = Σ_i w_i·log L_i` of the survival
/// location-scale family at the given block states, evaluated with an
/// arbitrary inverse link (the rest of the family fixed). Mirrors the
/// `offset_channel_geometry` row loop: the dynamic geometry (u0 = h0+q0,
/// u1 = h1+q1) depends only on the block states, so swapping the link
/// re-evaluates only the kernel coefficients. Used to finite-difference the
/// inverse-link data-fit θ-gradient.
fn survival_ls_total_log_likelihood_with_link(
    family: &SurvivalLocationScaleFamily,
    block_states: &[ParameterBlockState],
    link: &InverseLink,
) -> f64 {
    let mut probe = family.clone();
    probe.inverse_link = link.clone();
    let dynamic = probe
        .build_dynamic_geometry(block_states)
        .expect("dynamic geometry");
    let mut ll = 0.0;
    for i in 0..probe.n {
        if probe.w[i] <= 0.0 {
            continue;
        }
        let state = probe.row_predictor_state(
            dynamic.h_entry[i],
            dynamic.h_exit[i],
            dynamic.hdot_exit[i],
            dynamic.q_entry[i],
            dynamic.q_exit[i],
            dynamic.qdot_exit[i],
        );
        if let Some(kernel) = probe.exact_row_kernel(i, state).expect("row kernel") {
            ll += kernel.log_likelihood();
        }
    }
    ll
}

/// FD check for `SurvivalLocationScaleFamily::link_param_data_fit_gradient`:
/// the analytic `∂(−ℓ)/∂θ_link` for the SAS link `(ε, log δ)` must match a
/// central difference of the data-fit `−ℓ` at fixed β. This is the exact
/// data-fit term of the inverse-link profile-NLL gradient.
#[test]
fn link_param_data_fit_gradient_matches_finite_difference_sas() {
    let mut family = survival_exact_newton_test_family();
    let epsilon0 = 0.15;
    let log_delta0 = -0.25;
    family.inverse_link = InverseLink::Sas(
        state_from_sasspec(SasLinkSpec {
            initial_epsilon: epsilon0,
            initial_log_delta: log_delta0,
        })
        .expect("sas state"),
    );
    let states = survival_exact_newton_test_states(&family, 0.35, 0.3, -0.1);

    let analytic = family
        .link_param_data_fit_gradient(&states)
        .expect("link param data-fit gradient")
        .expect("SAS link has free parameters");
    assert_eq!(analytic.len(), 2, "SAS link has two parameters (ε, log δ)");

    let neg_ll = |epsilon: f64, log_delta: f64| -> f64 {
        let link = InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: log_delta,
            })
            .expect("sas state"),
        );
        -survival_ls_total_log_likelihood_with_link(&family, &states, &link)
    };

    let h = 1e-6;
    let fd_epsilon =
        (neg_ll(epsilon0 + h, log_delta0) - neg_ll(epsilon0 - h, log_delta0)) / (2.0 * h);
    let fd_log_delta =
        (neg_ll(epsilon0, log_delta0 + h) - neg_ll(epsilon0, log_delta0 - h)) / (2.0 * h);

    assert!(
        (analytic[0] - fd_epsilon).abs() <= 1e-5 * fd_epsilon.abs().max(1.0),
        "∂(−ℓ)/∂ε mismatch: analytic={}, fd={}",
        analytic[0],
        fd_epsilon
    );
    assert!(
        (analytic[1] - fd_log_delta).abs() <= 1e-5 * fd_log_delta.abs().max(1.0),
        "∂(−ℓ)/∂log δ mismatch: analytic={}, fd={}",
        analytic[1],
        fd_log_delta
    );
}

impl SurvivalLsLocationScaleRow {
    fn from_standardized_q(
        eta_location: f64,
        eta_logscale: f64,
        q_entry: f64,
        q_exit: f64,
        exit_index_derivative: f64,
        event: f64,
        weight: f64,
    ) -> Self {
        let sigma = eta_logscale.exp();
        Self {
            eta_location,
            eta_logscale,
            entry_index: eta_location + q_entry * sigma,
            exit_index: eta_location + q_exit * sigma,
            exit_index_derivative,
            event,
            weight,
        }
    }
}

#[derive(Clone, Copy)]
struct SlsHandWitnessScalarMap {
    v: f64,
    g: [f64; 2],
    h: [[f64; 2]; 2],
    t3: [[[f64; 2]; 2]; 2],
    t4: [[[[f64; 2]; 2]; 2]; 2],
}

impl SlsHandWitnessScalarMap {
    fn standardized_residual(index: f64, eta_location: f64, eta_logscale: f64) -> Self {
        let inv_sigma = (-eta_logscale).exp();
        let q = (index - eta_location) * inv_sigma;
        let mut map = Self {
            v: q,
            g: [-inv_sigma, -q],
            h: [[0.0; 2]; 2],
            t3: [[[0.0; 2]; 2]; 2],
            t4: [[[[0.0; 2]; 2]; 2]; 2],
        };
        map.h[0][1] = inv_sigma;
        map.h[1][0] = inv_sigma;
        map.h[1][1] = q;
        for (a, b, c) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            map.t3[a][b][c] = -inv_sigma;
        }
        map.t3[1][1][1] = -q;
        for (a, b, c, d) in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
            map.t4[a][b][c][d] = inv_sigma;
        }
        map.t4[1][1][1][1] = q;
        map
    }

    fn exit_derivative(index_derivative: f64, eta_logscale: f64) -> Self {
        let value = index_derivative * (-eta_logscale).exp();
        let mut map = Self {
            v: value,
            g: [0.0, -value],
            h: [[0.0; 2]; 2],
            t3: [[[0.0; 2]; 2]; 2],
            t4: [[[[0.0; 2]; 2]; 2]; 2],
        };
        map.h[1][1] = value;
        map.t3[1][1][1] = -value;
        map.t4[1][1][1][1] = value;
        map
    }
}

struct SlsHandWitnessChannels {
    value: f64,
    gradient: [f64; 2],
    hessian: [[f64; 2]; 2],
    t3: [[[f64; 2]; 2]; 2],
    t4: [[[[f64; 2]; 2]; 2]; 2],
}

impl SlsHandWitnessChannels {
    fn zero() -> Self {
        Self {
            value: 0.0,
            gradient: [0.0; 2],
            hessian: [[0.0; 2]; 2],
            t3: [[[0.0; 2]; 2]; 2],
            t4: [[[[0.0; 2]; 2]; 2]; 2],
        }
    }

    fn add_unary(&mut self, map: &SlsHandWitnessScalarMap, stack: [f64; 5], scale: f64) {
        self.value += scale * stack[0];
        for i in 0..2 {
            self.gradient[i] += scale * stack[1] * map.g[i];
            for j in 0..2 {
                self.hessian[i][j] +=
                    scale * (stack[1] * map.h[i][j] + stack[2] * map.g[i] * map.g[j]);
                for k in 0..2 {
                    self.t3[i][j][k] += scale
                        * (stack[1] * map.t3[i][j][k]
                            + stack[2]
                                * (map.g[i] * map.h[j][k]
                                    + map.g[j] * map.h[i][k]
                                    + map.g[k] * map.h[i][j])
                            + stack[3] * map.g[i] * map.g[j] * map.g[k]);
                    for l in 0..2 {
                        self.t4[i][j][k][l] += scale
                            * (stack[1] * map.t4[i][j][k][l]
                                + stack[2]
                                    * (map.g[i] * map.t3[j][k][l]
                                        + map.g[j] * map.t3[i][k][l]
                                        + map.g[k] * map.t3[i][j][l]
                                        + map.g[l] * map.t3[i][j][k]
                                        + map.h[i][j] * map.h[k][l]
                                        + map.h[i][k] * map.h[j][l]
                                        + map.h[i][l] * map.h[j][k])
                                + stack[3]
                                    * (map.g[i] * map.g[j] * map.h[k][l]
                                        + map.g[i] * map.g[k] * map.h[j][l]
                                        + map.g[i] * map.g[l] * map.h[j][k]
                                        + map.g[j] * map.g[k] * map.h[i][l]
                                        + map.g[j] * map.g[l] * map.h[i][k]
                                        + map.g[k] * map.g[l] * map.h[i][j])
                                + stack[4] * map.g[i] * map.g[j] * map.g[k] * map.g[l]);
                    }
                }
            }
        }
    }

    fn third_contracted(&self, dir: &[f64; 2]) -> [[f64; 2]; 2] {
        let mut out = [[0.0; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    out[a][b] += self.t3[a][b][c] * dir[c];
                }
            }
        }
        out
    }

    fn fourth_contracted(&self, u: &[f64; 2], v: &[f64; 2]) -> [[f64; 2]; 2] {
        let mut out = [[0.0; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for d in 0..2 {
                        out[a][b] += self.t4[a][b][c][d] * u[c] * v[d];
                    }
                }
            }
        }
        out
    }
}

fn survival_ls_exact_row_kernel(
    inverse_link: &InverseLink,
    row: SurvivalLsLocationScaleRow,
) -> SurvivalExactRowKernel {
    let family = SurvivalLocationScaleFamily {
        n: 1,
        y: array![row.event],
        w: array![row.weight],
        inverse_link: inverse_link.clone(),
        derivative_guard: 1e-12,
        x_time_entry: Arc::new(array![[1.0]]),
        x_time_exit: Arc::new(array![[1.0]]),
        x_time_deriv: Arc::new(array![[1.0]]),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]])),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: crate::resource::ResourcePolicy::default_library(),
    };
    let inv_sigma = (-row.eta_logscale).exp();
    let state = family.row_predictor_state(
        row.entry_index * inv_sigma,
        row.exit_index * inv_sigma,
        row.exit_index_derivative * inv_sigma,
        -row.eta_location * inv_sigma,
        -row.eta_location * inv_sigma,
        0.0,
    );
    family
        .exact_row_kernel(0, state)
        .expect("survival LS exact row kernel")
        .expect("positive-weight oracle row")
}

fn hand_survival_ls_channels(
    inverse_link: &InverseLink,
    row: SurvivalLsLocationScaleRow,
) -> SlsHandWitnessChannels {
    let q_entry = SlsHandWitnessScalarMap::standardized_residual(
        row.entry_index,
        row.eta_location,
        row.eta_logscale,
    );
    let q_exit = SlsHandWitnessScalarMap::standardized_residual(
        row.exit_index,
        row.eta_location,
        row.eta_logscale,
    );
    let g = SlsHandWitnessScalarMap::exit_derivative(row.exit_index_derivative, row.eta_logscale);
    let mut channels = SlsHandWitnessChannels::zero();
    channels.add_unary(
        &q_entry,
        survival_ls_log_survival_stack(inverse_link, q_entry.v)
            .expect("survival witness log-survival stack"),
        row.weight,
    );
    let censored_weight = row.weight * (1.0 - row.event);
    if censored_weight != 0.0 {
        channels.add_unary(
            &q_exit,
            survival_ls_log_survival_stack(inverse_link, q_exit.v)
                .expect("survival witness log-survival stack"),
            -censored_weight,
        );
    }
    let event_weight = row.weight * row.event;
    if event_weight != 0.0 {
        channels.add_unary(
            &q_exit,
            survival_ls_log_pdf_stack(inverse_link, q_exit.v, 0.0)
                .expect("survival witness log-pdf stack"),
            -event_weight,
        );
        channels.add_unary(&g, survival_ls_positive_log_stack(g.v), -event_weight);
    }
    channels
}

fn hand_survival_ls_kernel_channels(
    channels: &SlsHandWitnessChannels,
    dirs: &[[f64; 2]],
) -> crate::families::jet_tower::KernelChannels<2> {
    let third = dirs
        .iter()
        .map(|dir| (*dir, channels.third_contracted(dir)))
        .collect::<Vec<_>>();
    let fourth = dirs
        .iter()
        .enumerate()
        .map(|(idx, u)| {
            let v = dirs[(idx + 1) % dirs.len()];
            (*u, v, channels.fourth_contracted(u, &v))
        })
        .collect::<Vec<_>>();
    crate::families::jet_tower::KernelChannels {
        value: channels.value,
        gradient: channels.gradient,
        hessian: channels.hessian,
        third,
        fourth,
    }
}

#[test]
fn survival_ls_location_scale_jet_program_matches_exact_row_kernel_all_channels() {
    use crate::families::jet_tower::{evaluate_program, verify_kernel_channels};

    let dirs = [[0.7, -1.1], [-0.4, 0.9], [1.3, 0.25]];
    let rows = vec![
        SurvivalLsLocationScaleRow::from_standardized_q(0.25, 0.2, -0.75, 0.45, 1.15, 1.0, 1.7),
        SurvivalLsLocationScaleRow::from_standardized_q(-0.4, -0.35, -1.4, 1.2, 0.85, 0.0, 0.65),
        SurvivalLsLocationScaleRow::from_standardized_q(1.1, 0.05, -6.0, 7.0, 1.4, 1.0, 1.25),
        SurvivalLsLocationScaleRow::from_standardized_q(-0.8, 0.4, -5.0, 5.0, 0.55, 0.0, 0.9),
    ];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        for (row_index, row_data) in rows.iter().copied().enumerate() {
            let program = SurvivalLsLocationScaleNllProgram {
                inverse_link: &inverse_link,
                deriv_log_scale: 0.0,
                row: row_data,
            };
            let tower = evaluate_program(&program, 0).expect("survival LS tower");
            let witness = hand_survival_ls_channels(&inverse_link, row_data);
            let exact_kernel = survival_ls_exact_row_kernel(&inverse_link, row_data);
            let exact_value = -exact_kernel.log_likelihood();
            assert!(
                (witness.value - exact_value).abs() <= 1e-11 * exact_value.abs().max(1.0),
                "exact row kernel value mismatch for {distribution:?} row {row_index}: witness={} exact={}",
                witness.value,
                exact_value
            );
            let claims = hand_survival_ls_kernel_channels(&witness, &dirs);
            verify_kernel_channels(&tower, &claims, 1e-12).unwrap_or_else(|err| {
                    panic!(
                        "survival LS K=2 RowNllProgram mismatch against hand witness for {distribution:?} row {row_index}: {err}"
                    )
                });
            let production_tower = exact_kernel.location_scale_nll_tower(row_data);
            // The production kernel pre-evaluates its primitive stacks in a
            // different association order than the program path; observed
            // margin is ~1.3e-12 on fourth-order channels, so 5e-12 bounds
            // association noise (a dropped term would miss by >=1e-6).
            verify_kernel_channels(&production_tower, &claims, 5e-12).unwrap_or_else(|err| {
                    panic!(
                        "survival LS K=2 production exact-kernel jet mismatch against hand witness for {distribution:?} row {row_index}: {err}"
                    )
                });
        }
    }
}

/// #932 (survival follow-up, the issue's named next step): the survival
/// location-scale JOINT row NLL written ONCE over `Tower4<9>` in the
/// production kernel's nine linear-predictor primaries
/// `(h0, h1, d_raw, eta_t_exit, eta_t_entry, eta_t_deriv, eta_ls_exit,
/// eta_ls_entry, eta_ls_deriv)` — the exact `SLS_ROW_K` channel layout of
/// [`SurvivalLsRowKernel`]. The whole nonlinear composition that the
/// production path hand-writes is expressed here as plain `Tower4`
/// arithmetic:
///
/// ```text
///   u0 = h0 − eta_t_entry · exp(−eta_ls_entry)            (entry index)
///   u1 = h1 − eta_t_exit  · exp(−eta_ls_exit)             (exit index)
///   g  = d_raw + exp(−eta_ls_exit)·(eta_t_exit·eta_ls_deriv − eta_t_deriv)
///   nll = w·[ log S(u0) − (1−d)·log S(u1) − d·(log f(u1) + log g) ]
/// ```
///
/// so the tower mechanizes EXACTLY the calculus the hand path splits
/// across `q_chain_derivs_scalar` + `compose_survival_dynamic_q` (the
/// per-row `D/D2/D3` map tensors of `SurvivalLsRowKernel::row_maps`) and
/// the `row_kernel` / `row_third_contracted` Faà di Bruno accumulation
/// loops — the entry/exit/qdot cross blocks where the #736 bug genus
/// lives. Tail-critical primitives enter through the family's OWN
/// hand-certified stacks (`survival_ls_log_survival_stack` /
/// `_log_pdf_stack` / `_positive_log_stack`), so no probit/CLogLog/logit
/// primitive is re-derived: only the composition is mechanized.
struct SurvivalLsJointNllProgram<'a> {
    inverse_link: &'a InverseLink,
    primaries: Vec<[f64; SLS_ROW_K]>,
    event: Vec<f64>,
    weight: Vec<f64>,
}

impl crate::families::jet_tower::RowNllProgram<SLS_ROW_K> for SurvivalLsJointNllProgram<'_> {
    fn n_rows(&self) -> usize {
        self.primaries.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; SLS_ROW_K], String> {
        self.primaries
            .get(row)
            .copied()
            .ok_or_else(|| format!("survival LS joint program: row {row} out of range"))
    }

    fn row_nll(
        &self,
        row: usize,
        p: &[crate::families::jet_tower::Tower4<SLS_ROW_K>; SLS_ROW_K],
    ) -> Result<crate::families::jet_tower::Tower4<SLS_ROW_K>, String> {
        use crate::families::jet_tower::Tower4;

        let w = *self
            .weight
            .get(row)
            .ok_or_else(|| format!("survival LS joint program: weight row {row} missing"))?;
        let d = self.event[row];
        if w <= 0.0 {
            return Ok(Tower4::<SLS_ROW_K>::zero());
        }

        // Entry index: u0 = h0 + q0, q0 = −eta_t_entry · exp(−eta_ls_entry).
        let inv_sigma_entry = (-p[7]).exp();
        let u0 = p[0] - p[4] * inv_sigma_entry;
        // Exit index: u1 = h1 + q1, q1 = −eta_t_exit · exp(−eta_ls_exit).
        let inv_sigma_exit = (-p[6]).exp();
        let u1 = p[1] - p[3] * inv_sigma_exit;
        // Event Jacobian: g = d_raw + qdot,
        // qdot = exp(−eta_ls_exit)·(eta_t_exit·eta_ls_deriv − eta_t_deriv).
        let g = p[2] + inv_sigma_exit * (p[3] * p[8] - p[5]);

        // NLL = w·log S(u0) − w(1−d)·log S(u1) − w·d·(log f(u1) + log g),
        // term-for-term the sign layout of `SurvivalExactRowKernel::
        // log_likelihood` / `nll_index_tower` (left truncation divides the
        // likelihood by S(u0), so its log ADDS to the NLL).
        let mut nll = u0
            .compose_unary(survival_ls_log_survival_stack(self.inverse_link, u0.v)?)
            .scale(w);

        let censored_weight = w * (1.0 - d);
        if censored_weight != 0.0 {
            nll = nll
                + u1.compose_unary(survival_ls_log_survival_stack(self.inverse_link, u1.v)?)
                    .scale(-censored_weight);
        }

        let event_weight = w * d;
        if event_weight != 0.0 {
            nll = nll
                + u1.compose_unary(survival_ls_log_pdf_stack(self.inverse_link, u1.v, 0.0)?)
                    .scale(-event_weight)
                + g.compose_unary(survival_ls_positive_log_stack(g.v))
                    .scale(-event_weight);
        }

        Ok(nll)
    }
}

/// Build a fully time-varying, non-wiggle survival LS family whose three
/// blocks are single-column designs carrying the fixture primaries
/// verbatim (every block coefficient is 1), so all nine kernel channels —
/// including the entry and derivative threshold/log-sigma channels — are
/// live and mutually distinct.
fn survival_ls_joint_oracle_family(
    inverse_link: &InverseLink,
    primaries: &[[f64; SLS_ROW_K]],
    event: &[f64],
    weight: &[f64],
) -> SurvivalLocationScaleFamily {
    let n = primaries.len();
    let col = |ch: usize| Array2::from_shape_fn((n, 1), |(r, _)| primaries[r][ch]);
    let dense = |ch: usize| DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(col(ch)));
    SurvivalLocationScaleFamily {
        n,
        y: Array1::from(event.to_vec()),
        w: Array1::from(weight.to_vec()),
        inverse_link: inverse_link.clone(),
        derivative_guard: 1e-8,
        x_time_entry: Arc::new(col(0)),
        x_time_exit: Arc::new(col(1)),
        x_time_deriv: Arc::new(col(2)),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: dense(3),
        x_threshold_entry: Some(dense(4)),
        x_threshold_deriv: Some(dense(5)),
        x_log_sigma: dense(6),
        x_log_sigma_entry: Some(dense(7)),
        x_log_sigma_deriv: Some(dense(8)),
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: crate::resource::ResourcePolicy::default_library(),
    }
}

/// Block states matching [`survival_ls_joint_oracle_family`]: every block
/// coefficient is 1, and the eta vectors carry the stacked
/// `[exit; entry; derivative]` layout `validate_joint_states` expects for
/// time-varying blocks (the time block is always stacked).
fn survival_ls_joint_oracle_states(primaries: &[[f64; SLS_ROW_K]]) -> Vec<ParameterBlockState> {
    let n = primaries.len();
    let stacked = |exit: usize, entry: usize, deriv: usize| {
        let mut eta = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            eta[i] = primaries[i][exit];
            eta[n + i] = primaries[i][entry];
            eta[2 * n + i] = primaries[i][deriv];
        }
        eta
    };
    vec![
        ParameterBlockState {
            beta: array![1.0],
            eta: stacked(1, 0, 2),
        },
        ParameterBlockState {
            beta: array![1.0],
            eta: stacked(3, 4, 5),
        },
        ParameterBlockState {
            beta: array![1.0],
            eta: stacked(6, 7, 8),
        },
    ]
}

/// #932 universal oracle on the production `RowKernel<9>` implementation.
///
/// Audits every channel the hand-written [`SurvivalLsRowKernel`] emits —
/// value / gradient / Hessian / `row_third_contracted(dir)` — against the
/// single-expression `RowNllProgram<9>` tower truth, for every residual
/// distribution the family enumerates (Gaussian/probit, Gumbel/CLogLog =
/// Weibull-AFT, Logistic/logit = log-logistic-AFT), over a fixture grid
/// covering exact deaths, right-censored rows, a fractional event weight,
/// deep left-truncated entries, an effectively untruncated entry
/// (u0 ≈ −6), and extreme exit-index tails on both sides (u1 ≈ ±6). The
/// entry/exit/qdot cross blocks — the channels #736's sign flip class
/// corrupts — are contracted explicitly through dense 9-dim directions.
///
/// `row_fourth_contracted` is tower-derived from the same row program so the
/// generic RowKernel second-directional Hessian path can consume survival LS
/// without a family-specific refusal.
#[test]
fn survival_ls_joint_row_kernel_agrees_with_jet_tower_program_all_channels() {
    // Tower4<9> carries 9⁴ fourth-order entries (≈59 KiB per scalar by
    // value); the program evaluation keeps a handful of live towers plus
    // the 9-variable seed array on the stack, so run the body on a
    // dedicated wide-stack thread instead of the 2 MiB test default.
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_joint_jet_tower_oracle_body)
        .expect("spawn wide-stack oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS joint jet-tower oracle thread must complete"
    );
}

fn survival_ls_joint_jet_tower_oracle_body() {
    use crate::families::jet_tower::{KernelChannels, evaluate_program, verify_kernel_channels};
    use crate::families::row_kernel::RowKernel;

    // Channel layout per row:
    // [h0, h1, d_raw, eta_t_exit, eta_t_entry, eta_t_deriv,
    //  eta_ls_exit, eta_ls_entry, eta_ls_deriv]
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        // Exact death, moderate indices.
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        // Right-censored, small event Jacobian g (≈0.08, far above guard).
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        // Exact death, extreme right exit tail (u1 ≈ +6.2), entry
        // effectively untruncated (u0 ≈ −6.3).
        [-6.5, 5.6, 1.1, -0.7, -0.3, -0.15, 0.2, 0.4, 0.1],
        // Right-censored, extreme left exit tail (u1 ≈ −5.8).
        [-1.0, -5.2, 0.7, 0.5, 0.6, 0.3, -0.1, -0.3, -0.25],
        // Exact death with DEEP left truncation (u0 ≈ +1.9: S(u0) ≪ 1).
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, 0.35],
        // Fractional event target exercises the d∉{0,1} event_mix branch.
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 0.0, 1.0, 0.35];
    let weight = [1.0, 0.8, 1.2, 0.9, 1.1, 1.3];
    let n = primaries.len();

    // Dense deterministic directions: every one of the nine channels
    // participates in every contraction, so dropped/flipped cross blocks
    // (entry×exit, threshold×log-sigma, value×derivative) cannot hide.
    let dirs: [[f64; SLS_ROW_K]; 3] = [
        [0.7, -1.3, 0.5, 0.9, -0.6, 0.3, -1.1, 0.4, 0.8],
        [-0.4, 0.6, -1.1, 0.3, 1.2, -0.7, 0.5, -0.9, 0.2],
        [1.2, 0.2, -0.7, -0.5, 0.4, 1.0, -0.3, 0.6, -1.2],
    ];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        let states = survival_ls_joint_oracle_states(&primaries);
        let q = family
            .collect_joint_quantities(&states)
            .expect("collect joint quantities");
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = SurvivalLsRowKernel {
            family: &family,
            q: &q,
            dynamic: &dynamic,
            deriv_log_scale: 0.0,
            offsets: family.joint_block_offsets(),
        };
        let program = SurvivalLsJointNllProgram {
            inverse_link: &inverse_link,
            primaries: primaries.clone(),
            event: event.to_vec(),
            weight: weight.to_vec(),
        };

        for row in 0..n {
            // The program's recomputed indices must agree with the
            // production dynamic geometry to floating-point noise —
            // otherwise the oracle would compare towers seeded at
            // different points and prove nothing.
            let p = primaries[row];
            let u0_prog = p[0] - p[4] * (-p[7]).exp();
            let u1_prog = p[1] - p[3] * (-p[6]).exp();
            let g_prog = p[2] + (-p[6]).exp() * (p[3] * p[8] - p[5]);
            let u0_dyn = dynamic.h_entry[row] + dynamic.q_entry[row];
            let u1_dyn = dynamic.h_exit[row] + dynamic.q_exit[row];
            let g_dyn = dynamic.hdot_exit[row] + dynamic.qdot_exit[row];
            assert!(
                (u0_prog - u0_dyn).abs() <= 1e-12 * u0_dyn.abs().max(1.0),
                "{distribution:?} row {row}: entry index mismatch: program {u0_prog} dynamic {u0_dyn}"
            );
            assert!(
                (u1_prog - u1_dyn).abs() <= 1e-12 * u1_dyn.abs().max(1.0),
                "{distribution:?} row {row}: exit index mismatch: program {u1_prog} dynamic {u1_dyn}"
            );
            assert!(
                (g_prog - g_dyn).abs() <= 1e-12 * g_dyn.abs().max(1.0),
                "{distribution:?} row {row}: event Jacobian mismatch: program {g_prog} dynamic {g_dyn}"
            );
            assert!(
                g_prog > family.derivative_guard,
                "{distribution:?} row {row}: fixture must stay clear of the monotonicity \
                     guard so no clamping perturbs the comparison (g={g_prog})"
            );

            let tower = evaluate_program(&program, row).expect("survival LS joint tower");

            let (value, gradient, hessian) =
                RowKernel::row_kernel(&kernel, row).expect("hand kernel value/grad/hess");

            let third: Vec<([f64; SLS_ROW_K], [[f64; SLS_ROW_K]; SLS_ROW_K])> = dirs
                .iter()
                .map(|dir| {
                    let claim = RowKernel::row_third_contracted(&kernel, row, dir)
                        .expect("hand kernel third");
                    (*dir, claim)
                })
                .collect();
            let fourth: Vec<(
                [f64; SLS_ROW_K],
                [f64; SLS_ROW_K],
                [[f64; SLS_ROW_K]; SLS_ROW_K],
            )> = dirs
                .iter()
                .enumerate()
                .map(|(idx, u)| {
                    let v = dirs[(idx + 1) % dirs.len()];
                    let claim = RowKernel::row_fourth_contracted(&kernel, row, u, &v)
                        .expect("hand kernel fourth");
                    (*u, v, claim)
                })
                .collect();

            let claims = KernelChannels {
                value,
                gradient,
                hessian,
                third,
                fourth,
            };

            verify_kernel_channels(&tower, &claims, 1e-9).unwrap_or_else(|e| {
                panic!(
                    "{distribution:?} row {row}: hand SurvivalLsRowKernel disagrees with \
                         #932 jet-tower truth: {e}"
                )
            });
        }
    }
}

/// The hand-derived analytic joint-Hessian directional derivative
/// (`exact_newton_joint_hessian_directional_derivative_from_parts`, the path
/// that is always taken because `row_kernel_directional_supported()` is
/// hard-disabled) must agree with the jet-tower-certified generic row-kernel
/// directional derivative on a FULLY TIME-VARYING family — i.e. with the
/// derivative threshold/log-sigma channels (the velocity / `qdot` coordinate)
/// live. The pre-existing FD coverage only exercises non-time-varying
/// fixtures, where the velocity coordinate is inert, so it cannot witness a
/// dropped `qdot` third-order contribution.
#[test]
fn survival_ls_joint_directional_derivative_matches_tower_time_varying() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_joint_directional_derivative_time_varying_body)
        .expect("spawn wide-stack directional-derivative thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS joint directional-derivative time-varying oracle thread must complete"
    );
}

fn survival_ls_joint_directional_derivative_time_varying_body() {
    use crate::families::row_kernel::{RowSet, row_kernel_directional_derivative_generic};

    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, 0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 1.0, 1.0, 0.35];
    let weight = [1.0, 1.2, 1.1, 1.3];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        let states = survival_ls_joint_oracle_states(&primaries);
        let q = family
            .collect_joint_quantities(&states)
            .expect("collect joint quantities");
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = SurvivalLsRowKernel {
            family: &family,
            q: &q,
            dynamic: &dynamic,
            deriv_log_scale: 0.0,
            offsets: family.joint_block_offsets(),
        };
        for direction in [
            array![0.7, -0.5, 0.9],
            array![-1.1, 0.8, 0.3],
            array![0.4, 1.2, -0.6],
            array![1.0, 0.0, 0.0],
            array![0.0, 1.0, 0.0],
            array![0.0, 0.0, 1.0],
        ] {
            let dir_slice = direction.as_slice().expect("contiguous direction");
            let reference =
                row_kernel_directional_derivative_generic(&kernel, &RowSet::All, dir_slice)
                    .expect("tower-certified directional derivative");
            let hand = family
                .exact_newton_joint_hessian_directional_derivative_rescaled_from_parts(
                    &direction, &q, &dynamic, 0.0,
                )
                .expect("hand directional derivative")
                .expect("hand directional derivative present");
            assert_eq!(reference.dim(), hand.dim(), "directional dH shape");
            for ((a, b), &want) in reference.indexed_iter() {
                let got = hand[[a, b]];
                assert!(
                    (got - want).abs() <= 1e-7 * (1.0 + want.abs()),
                    "{distribution:?} dir={direction} joint directional dH[{a}][{b}] mismatch: \
                     hand={got} tower-reference={want}"
                );
            }
        }
    }
}

/// #921: the `RowKernel<9>` repackaging must reproduce the bespoke joint
/// assembly bit-for-bit. We build a non-time-varying, non-wiggle fixture
/// (the config the kernel covers), then assert the generic row-kernel engine
/// (`build_row_kernel_cache` → `row_kernel_hessian_dense` /
/// `row_kernel_log_likelihood`) matches the existing assembly oracle and the
/// bespoke per-row log-likelihood. The public `exact_newton_joint_hessian`
/// method now delegates to this RowKernel path for the covered non-wiggle
/// shape, so the test calls `assemble_joint_hessian_from_quantities`
/// directly to keep an independent oracle.
#[test]
fn survival_ls_row_kernel_matches_bespoke_assembly() {
    use crate::families::row_kernel::{
        RowSet, build_row_kernel_cache, row_kernel_directional_derivative, row_kernel_gradient,
        row_kernel_hessian_dense, row_kernel_log_likelihood,
    };

    let family = survival_exact_newton_test_family();
    let n = family.n;
    let beta_t = 0.3_f64;
    let beta_thr = -0.4_f64;
    let beta_ls = 0.2_f64;
    let states = survival_exact_newton_test_states(&family, beta_t, beta_thr, beta_ls);

    let q = family
        .collect_joint_quantities(&states)
        .expect("collect joint quantities");
    let dynamic = family
        .build_dynamic_geometry(&states)
        .expect("dynamic geometry");
    let kernel = SurvivalLsRowKernel {
        family: &family,
        q: &q,
        dynamic: &dynamic,
        deriv_log_scale: 0.0,
        offsets: family.joint_block_offsets(),
    };

    let cache = build_row_kernel_cache(&kernel, &RowSet::All).expect("row kernel cache");
    let h_new = row_kernel_hessian_dense(&kernel, &cache, &RowSet::All);
    let h_old = family
        .assemble_joint_hessian_from_quantities(&q, &states)
        .expect("joint Hessian oracle")
        .expect("joint Hessian oracle present");
    assert_eq!(h_new.dim(), h_old.dim(), "joint hessian shape");
    for ((a, b), &old) in h_old.indexed_iter() {
        let new = h_new[[a, b]];
        assert!(
            (new - old).abs() <= 1e-9 * (1.0 + old.abs()),
            "joint Hessian [{a}][{b}] mismatch: new={new}, old={old}"
        );
    }

    // Log-likelihood: the generic engine returns ℓ = -Σ nll_i; the bespoke
    // per-row log-likelihood sums `exact_row_kernel(row).log_likelihood()`.
    let ll_new = row_kernel_log_likelihood(&cache, &RowSet::All);
    let mut ll_old = 0.0;
    for i in 0..n {
        let state = family.row_predictor_state(
            dynamic.h_entry[i],
            dynamic.h_exit[i],
            dynamic.hdot_exit[i],
            dynamic.q_entry[i],
            dynamic.q_exit[i],
            dynamic.qdot_exit[i],
        );
        if let Some(k) = family.exact_row_kernel(i, state).expect("row kernel") {
            ll_old += k.log_likelihood();
        }
    }
    assert!(
        (ll_new - ll_old).abs() <= 1e-9 * (1.0 + ll_old.abs()),
        "log-likelihood mismatch: new={ll_new}, old={ll_old}"
    );

    // Gradient assembles at the right coefficient dimension.
    let g_new = row_kernel_gradient(&kernel, &cache, &RowSet::All);
    assert_eq!(g_new.len(), *kernel.offsets.last().unwrap());

    let direction = array![0.17, -0.11, 0.07];
    let d_new = row_kernel_directional_derivative(
        &kernel,
        &RowSet::All,
        direction
            .as_slice()
            .expect("literal direction is contiguous"),
    )
    .expect("row-kernel directional derivative");
    let eps = 1e-5;
    let plus = survival_exact_newton_test_states(
        &family,
        beta_t + eps * direction[0],
        beta_thr + eps * direction[1],
        beta_ls + eps * direction[2],
    );
    let minus = survival_exact_newton_test_states(
        &family,
        beta_t - eps * direction[0],
        beta_thr - eps * direction[1],
        beta_ls - eps * direction[2],
    );
    let q_plus = family.collect_joint_quantities(&plus).expect("plus q");
    let q_minus = family.collect_joint_quantities(&minus).expect("minus q");
    let h_plus = family
        .assemble_joint_hessian_from_quantities(&q_plus, &plus)
        .expect("plus Hessian oracle")
        .expect("plus Hessian present");
    let h_minus = family
        .assemble_joint_hessian_from_quantities(&q_minus, &minus)
        .expect("minus Hessian oracle")
        .expect("minus Hessian present");
    let d_fd = (&h_plus - &h_minus) / (2.0 * eps);
    for ((a, b), &fd) in d_fd.indexed_iter() {
        let new = d_new[[a, b]];
        assert!(
            (new - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
            "directional Hessian [{a}][{b}] mismatch: new={new}, fd={fd}"
        );
    }
}

#[test]
fn survival_location_scale_coefficient_cost_delegates_to_joint_coupled_helper() {
    // SurvivalLocationScale couples time, threshold, log-σ, and optional
    // wiggle blocks per row. The override pulls n from `self.n` and
    // forwards specs to the shared joint-coupled helper.
    let family = survival_exact_newton_test_family();
    let n = family.n as u64;
    let p_time = 5usize;
    let p_threshold = 3usize;
    let p_log_sigma = 2usize;
    let mk_spec = |name: &str, p: usize| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((family.n, p)))),
        offset: Array1::zeros(family.n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![
        mk_spec("time", p_time),
        mk_spec("threshold", p_threshold),
        mk_spec("log_sigma", p_log_sigma),
    ];
    let p_total = (p_time + p_threshold + p_log_sigma) as u64;
    let expected = crate::custom_family::joint_coupled_coefficient_hessian_cost(n, &specs);
    assert_eq!(family.coefficient_hessian_cost(&specs), expected);
    assert_eq!(expected, n * p_total * p_total);
    assert!(
        expected > crate::custom_family::default_coefficient_hessian_cost(&specs),
        "joint-coupled cost must exceed block-diagonal default by the cross-block fill"
    );
}

/// Build a location-scale family whose three coefficient blocks are each
/// `p`-columns wide (and `n`-rows) so `joint_block_dims()` == `[p, p, p]`.
/// The advertisement guards (`validate_joint_specs`) compare the spec
/// widths against `joint_block_dims()`, so the family's design widths must
/// equal the spec widths for the HVP-availability path to be exercised
/// (gam#848); the previous fixture left the family at 1-column designs
/// while building width-200 specs, so the guard correctly rejected them.
fn survival_large_scale_block_test_family(p: usize) -> SurvivalLocationScaleFamily {
    let n = 3usize;
    let mut family = survival_exact_newton_test_family();
    family.x_threshold = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        Array2::<f64>::zeros((n, p)),
    ));
    family.x_log_sigma = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        Array2::<f64>::zeros((n, p)),
    ));
    family.x_time_entry = Arc::new(Array2::<f64>::zeros((n, p)));
    family.x_time_exit = Arc::new(Array2::<f64>::zeros((n, p)));
    family.x_time_deriv = Arc::new(Array2::<f64>::zeros((n, p)));
    family
}

#[test]
fn survival_location_scale_advertises_outer_hvp_at_large_scale_dimensions() {
    let family = survival_large_scale_block_test_family(200);
    let mk_spec = |name: &str, p: usize| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((family.n, p)))),
        offset: Array1::zeros(family.n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![
        mk_spec("time", 200),
        mk_spec("threshold", 200),
        mk_spec("log_sigma", 200),
    ];

    assert!(family.outer_hyper_hessian_hvp_available(&specs));
    assert!(crate::custom_family::use_joint_matrix_free_path(
        specs.iter().map(|spec| spec.design.ncols()).sum(),
        family.n,
    ));
    assert!(
        !family.outer_hyper_hessian_dense_available(&specs),
        "large-scale survival location-scale should expose the outer Hessian through HVPs, not dense pairwise assembly"
    );
}

#[test]
fn survival_location_scale_planner_keeps_analytic_hessian_at_large_scale_dimensions() {
    let family = survival_large_scale_block_test_family(200);
    let mk_spec = |name: &str, p: usize| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((family.n, p)))),
        offset: Array1::zeros(family.n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![
        mk_spec("time", 200),
        mk_spec("threshold", 200),
        mk_spec("log_sigma", 200),
    ];
    let options = crate::custom_family::BlockwiseFitOptions::default();

    let (gradient, hessian) =
        crate::custom_family::custom_family_outer_derivatives(&family, &specs, &options);
    assert_eq!(gradient, crate::solver::rho_optimizer::Derivative::Analytic);
    assert_eq!(
        hessian,
        crate::solver::rho_optimizer::DeclaredHessianForm::Either,
        "large survival location-scale fits must not be demoted to BFGS when the explicit HVP operator covers the dimensions"
    );
}

#[test]
fn time_block_post_update_leaves_beta_unchanged() {
    // The QP owns feasibility. The post-update hook may validate the
    // accepted beta, but it must not silently repair a missing constraint
    // row after the solver has produced a step.
    let family = survival_exact_newton_test_family();
    let spec = ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((1, 1)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    let feasible = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            }],
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &spec,
            array![0.5],
        )
        .expect("return time beta");
    assert_eq!(feasible, array![0.5]);

    let err = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            }],
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &spec,
            array![-2.0],
        )
        .expect_err("post-update must reject, not repair, infeasible time beta");
    assert!(
        err.contains("violates represented linear constraint"),
        "unexpected error: {err}"
    );
}

#[test]
fn time_block_feasible_step_stays_inside_derivative_guard() {
    let family = survival_exact_newton_test_family();
    let states = vec![
        ParameterBlockState {
            beta: array![0.1],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
    ];
    let alpha = family
        .max_feasible_step_size(
            &states,
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &array![-2.0],
        )
        .expect("time step ceiling")
        .expect("time step should be bounded");
    assert!((alpha - 0.04975).abs() <= 1e-12);
    let feasible = states[0].beta[0] + alpha * -2.0;
    assert!(feasible >= 0.0);
}

#[test]
fn latent_time_constraints_use_exact_derivative_guard_rows() {
    let constraints = structural_time_coefficient_constraints(
        &DesignMatrix::from(array![[1.0, 1.0], [2.0, -1.0]]),
        &array![0.25, 0.75],
        1.0,
    )
    .expect("exact derivative guard constraints")
    .expect("nonzero derivative rows");

    let scale0 = 2.0_f64.sqrt();
    let scale1 = 5.0_f64.sqrt();
    let expected_a = array![[1.0 / scale0, 1.0 / scale0], [2.0 / scale1, -1.0 / scale1]];
    let expected_b = array![0.75 / scale0, 0.25 / scale1];
    assert!(
        (&constraints.a - &expected_a)
            .iter()
            .all(|v| v.abs() <= 1e-12),
        "scaled A mismatch: got {:?}, expected {:?}",
        constraints.a,
        expected_a
    );
    assert!(
        (&constraints.b - &expected_b)
            .iter()
            .all(|v| v.abs() <= 1e-12),
        "scaled b mismatch: got {:?}, expected {:?}",
        constraints.b,
        expected_b
    );
}

#[test]
fn time_block_feasible_step_accepts_zero_beta_when_offset_encodes_guard() {
    let family = survival_exact_newton_test_family();
    let states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 1e-8],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
    ];
    let alpha = family
        .max_feasible_step_size(
            &states,
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &array![0.0],
        )
        .expect("zero-step structural state should be valid")
        .expect("time step should be bounded");
    assert_eq!(alpha, 1.0);
}

#[test]
fn linkwiggle_block_post_update_leaves_beta_unchanged() {
    let mut family = survival_exact_newton_test_family();
    family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])));
    family.wiggle_knots = Some(array![-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0]);
    family.wiggle_degree = Some(3);
    let spec = ParameterBlockSpec {
        name: "linkwiggle".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((1, 2)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let returned = family
        .post_update_block_beta(
            &[
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.1, 0.2],
                    eta: array![0.0, 0.0, 0.0],
                },
            ],
            SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
            &spec,
            array![0.3, 0.0],
        )
        .expect("return linkwiggle beta");
    assert_eq!(returned, array![0.3, 0.0]);

    let err = family
        .post_update_block_beta(
            &[
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.1, 0.2],
                    eta: array![0.0, 0.0, 0.0],
                },
            ],
            SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
            &spec,
            array![0.3, -0.1],
        )
        .expect_err("infeasible link-wiggle beta must be rejected");
    assert!(
        err.contains("violates represented nonnegativity"),
        "unexpected error: {err}"
    );
}

#[test]
fn linkwiggle_block_feasible_step_stays_nonnegative() {
    let mut family = survival_exact_newton_test_family();
    family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])));
    family.wiggle_knots = Some(array![-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0]);
    family.wiggle_degree = Some(3);
    let states = vec![
        ParameterBlockState {
            beta: array![0.1],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.2, 0.4],
            eta: array![0.0, 0.0, 0.0],
        },
    ];
    let alpha = family
        .max_feasible_step_size(
            &states,
            SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
            &array![-1.0, -0.1],
        )
        .expect("linkwiggle step ceiling")
        .expect("linkwiggle step should be bounded");
    assert!(alpha > 0.0 && alpha < 1.0);
    let feasible = &states[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE].beta
        + &(array![-1.0, -0.1] * alpha);
    assert!(feasible.iter().all(|&value| value >= 0.0));
}

fn survival_exact_newton_test_familywith_inverse_link(
    inverse_link: InverseLink,
) -> SurvivalLocationScaleFamily {
    SurvivalLocationScaleFamily {
        inverse_link,
        ..survival_exact_newton_test_family()
    }
}

fn sparse_survival_exact_newton_test_family() -> SurvivalLocationScaleFamily {
    let mut family = survival_exact_newton_test_family();
    family.x_threshold = sparse_design_from_dense(&array![[1.0], [0.4], [-0.6]]);
    family.x_log_sigma = sparse_design_from_dense(&array![[1.0], [-0.3], [0.5]]);
    family
}

#[test]
fn compose_survival_dynamic_q_uses_correct_qdot_ll_coefficient() {
    let base = survival_base_q_scalars(0.8, -0.35);
    let eta_t_deriv = 1.4;
    let eta_ls_deriv = -0.6;
    let wiggle_value = 0.2;
    let dq_dq0 = 1.1;
    let d2q_dq02 = -0.7;
    let d3q_dq03 = 0.45;

    let dyn_q = compose_survival_dynamic_q(
        base,
        eta_t_deriv,
        eta_ls_deriv,
        wiggle_value,
        dq_dq0,
        d2q_dq02,
        d3q_dq03,
    );

    let a = base.q_t;
    let b = base.q_ls;
    let d = base.q_ll;
    let e = base.q_tl_ls;
    let f = base.q_ll_ls;
    let r = safe_sum2(safe_product(a, eta_t_deriv), safe_product(b, eta_ls_deriv));
    let r_ls = safe_sum2(
        safe_product(base.q_tl, eta_t_deriv),
        safe_product(d, eta_ls_deriv),
    );
    let r_ll = safe_sum2(safe_product(e, eta_t_deriv), safe_product(f, eta_ls_deriv));
    let expected = safe_sum3(
        safe_product(d3q_dq03, safe_product(safe_product(b, b), r)),
        safe_product(
            d2q_dq02,
            safe_sum2(safe_product(d, r), 2.0 * safe_product(b, r_ls)),
        ),
        safe_product(dq_dq0, r_ll),
    );

    assert!(
        (dyn_q.qdot_ll - expected).abs() <= 1e-12,
        "qdot_ll mismatch: got {}, expected {}",
        dyn_q.qdot_ll,
        expected
    );
}

fn survival_exact_newton_threshold_states(beta_threshold: f64) -> Vec<ParameterBlockState> {
    vec![
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.1, 0.35, -0.2, 0.25, 0.6, 0.15, 0.5, 0.7, 0.6],
        },
        ParameterBlockState {
            beta: array![beta_threshold],
            eta: array![beta_threshold, 0.4 * beta_threshold, -0.6 * beta_threshold],
        },
        ParameterBlockState {
            beta: array![-0.15],
            eta: array![-0.15, 0.045, -0.075],
        },
    ]
}

fn survival_exact_newton_rebuild_states(
    beta_time: &Array1<f64>,
    beta_threshold: &Array1<f64>,
    beta_log_sigma: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    vec![
        ParameterBlockState {
            beta: beta_time.clone(),
            eta: array![
                beta_time[0],
                beta_time[0],
                beta_time[0],
                1.2 * beta_time[0],
                0.9 * beta_time[0],
                1.4 * beta_time[0],
                beta_time[0] + 0.5,
                beta_time[0] + 0.7,
                beta_time[0] + 0.6
            ],
        },
        ParameterBlockState {
            beta: beta_threshold.clone(),
            eta: array![
                beta_threshold[0],
                0.4 * beta_threshold[0],
                -0.6 * beta_threshold[0]
            ],
        },
        ParameterBlockState {
            beta: beta_log_sigma.clone(),
            eta: array![
                beta_log_sigma[0],
                -0.3 * beta_log_sigma[0],
                0.5 * beta_log_sigma[0]
            ],
        },
    ]
}

fn survival_outergradient_testspecs() -> Vec<ParameterBlockSpec> {
    vec![
        ParameterBlockSpec {
            name: "time_transform".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [1.0],
                [1.0],
                [1.2],
                [0.9],
                [1.4],
                [1.0],
                [1.0],
                [1.0]
            ])),
            offset: Array1::zeros(9),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [0.4],
                [-0.6]
            ])),
            offset: Array1::zeros(3),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.35]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [-0.3],
                [0.5]
            ])),
            offset: Array1::zeros(3),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![-0.15]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ]
}

fn survival_non_probit_test_links() -> Vec<(&'static str, InverseLink)> {
    vec![
        (
            "logistic",
            residual_distribution_inverse_link(ResidualDistribution::Logistic),
        ),
        (
            "cloglog",
            residual_distribution_inverse_link(ResidualDistribution::Gumbel),
        ),
        (
            "sas",
            InverseLink::Sas(
                state_from_sasspec(SasLinkSpec {
                    initial_epsilon: 0.1,
                    initial_log_delta: -0.2,
                })
                .expect("sas state"),
            ),
        ),
        (
            "beta-logistic",
            InverseLink::BetaLogistic(
                state_from_beta_logisticspec(SasLinkSpec {
                    initial_epsilon: 0.05,
                    initial_log_delta: 0.1,
                })
                .expect("beta-logistic state"),
            ),
        ),
    ]
}

#[test]
fn wip_outergradient_testspecs_shape() {
    let specs = survival_outergradient_testspecs();
    assert_eq!(specs.len(), 3);
    assert_eq!(specs[0].name, "time_transform");
    assert_eq!(specs[1].name, "threshold");
    assert_eq!(specs[2].name, "log_sigma");
}

#[test]
fn identified_time_block_preserves_input_designs() {
    let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
    let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    assert_eq!(prepared.design_entry, design_entry);
    assert_eq!(prepared.design_exit, design_exit);
    assert_eq!(prepared.design_derivative_exit, design_derivative_exit);
}

#[test]
fn identified_time_block_preserves_expected_nullspace_dimension() {
    let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
    let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry),
        design_exit: DesignMatrix::from(design_exit),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };

    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    let p = time_block.design_entry.ncols();

    assert_eq!(
        prepared.transform.gauge.raw_total(),
        p,
        "identifiability transform must stay in the original coefficient space"
    );
    assert_eq!(
        prepared.transform.gauge.reduced_total(),
        p,
        "anchored time basis should keep the full coefficient dimension"
    );
    assert_eq!(
        prepared.design_entry.ncols(),
        p,
        "prepared entry design should keep the full anchored basis width"
    );
    assert_eq!(
        prepared.design_exit.ncols(),
        p,
        "prepared exit design should keep the full anchored basis width"
    );
    assert_eq!(
        prepared.transform.gauge.block_transform(0),
        Array2::<f64>::eye(p)
    );
    assert_eq!(
        prepared.transform.gauge.affine_shift,
        Array1::<f64>::zeros(p)
    );
}

#[test]
fn identified_time_block_can_reduce_to_parametric_penalty_nullspace() {
    let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
    let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: Some(array![0.5, 0.2, 9.0]),
    };

    // log(t_exit) for the unit-log-t warp-slope pin (issue #892).
    let log_time_entry = array![-1.0_f64, -0.5, 0.0];
    let log_time_exit = array![0.0_f64, 0.405_465_108, 0.916_290_731];
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        true,
        log_time_entry.view(),
        log_time_exit.view(),
    )
    .expect("prepare time block");
    // Canonical gauge pin (#892): the warp slope is folded into the offset,
    // so the FREE time block collapses to the single row-constant direction.
    // The Gauge map is now p×1 (was p×2), with the pinned unit-log-t
    // warp carried by `Gauge::affine_shift` rather than a free column.
    assert_eq!(prepared.transform.gauge.raw_total(), 3);
    assert_eq!(prepared.transform.gauge.reduced_total(), 1);
    assert_eq!(prepared.transform.gauge.affine_shift.len(), 3);
    assert!(
        prepared
            .transform
            .gauge
            .affine_shift
            .iter()
            .any(|&v| v.abs() > 1e-9),
        "pinned warp must contribute a non-zero Gauge affine_shift"
    );
    assert_eq!(prepared.design_entry.ncols(), 1);
    assert_eq!(prepared.design_exit.ncols(), 1);
    assert_eq!(prepared.design_derivative_exit.ncols(), 1);
    assert!(prepared.coefficient_lower_bounds.is_none());
    // The reduced block lives on the penalty null space, so `zᵀ S z` is
    // exactly zero: there is no curvature left to penalize. An unpenalized
    // parametric block has no smoothing parameter, so the projected-to-zero
    // penalties are dropped entirely — the block carries ZERO penalties and
    // therefore contributes no ρ coordinate to the outer REML search
    // (issue #736/#735/#721).
    assert!(
        prepared.penalties.is_empty(),
        "reduced parametric time block must be unpenalized (no smoothing parameter), got {} penalties",
        prepared.penalties.len()
    );
    assert!(
        prepared.nullspace_dims.is_empty(),
        "reduced parametric time block carries no penalty null-space bookkeeping"
    );
}

#[test]
fn pinned_time_warp_affine_lift_round_trips() {
    // Golden round-trip (issue #892): on a rank-clean pinned reduced fit the
    // raw time coefficients must be reconstructed EXACTLY through the
    // Gauge-owned affine section `β_raw = T · θ + a`. A wrong lift silently
    // corrupts every reported survival time-coefficient, so this guards the
    // finalize math directly. Choose a known reduced free coefficient `θ` and
    // verify the lifted raw coefficient reproduces both the free constant
    // direction (`θ · z_c`) and the pinned unit-log-t warp (`a`),
    // and that the design image `X · β_raw` equals
    // `(X · z_c) θ + X · a` (the free design plus the folded
    // offset), which is what the geometry actually consumes.
    let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
    let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let log_time_entry = array![-1.0_f64, -0.5, 0.0];
    let log_time_exit = array![0.0_f64, 0.405_465_108, 0.916_290_731];
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        true,
        log_time_entry.view(),
        log_time_exit.view(),
    )
    .expect("prepare time block");
    // Pin fired: single free column + non-zero pinned warp.
    assert_eq!(prepared.transform.gauge.reduced_total(), 1);
    let theta = array![0.731_f64];
    let beta_raw = prepared
        .transform
        .gauge
        .lift_block_betas(&[theta.clone()])
        .remove(0);
    // β_raw equals the free contribution plus the pinned warp, exactly.
    let z_c = prepared.transform.gauge.block_transform(0);
    let expected_raw =
        &(&z_c.column(0).to_owned() * theta[0]) + &prepared.transform.gauge.affine_shift;
    for (got, want) in beta_raw.iter().zip(expected_raw.iter()) {
        assert!(
            (got - want).abs() <= 1e-12,
            "affine lift must reconstruct raw coefficients exactly: got {got}, want {want}"
        );
    }
    // The raw design image matches free-design·θ + augmented offset delta,
    // i.e. what the solver geometry sees: X·β_raw = (X·z_c)·θ + X·z_t.
    let raw_image = design_exit.dot(&beta_raw);
    let folded = &prepared.design_exit.column(0).to_owned() * theta[0]
        + &(&prepared.offset_exit - &time_block.offset_exit);
    for (got, want) in raw_image.iter().zip(folded.iter()) {
        assert!(
            (got - want).abs() <= 1e-9,
            "raw design image must equal free image plus folded offset: got {got}, want {want}"
        );
    }
    // The folded exit offset has unit slope vs log t (the canonical gauge).
    let delta = &prepared.offset_exit - &time_block.offset_exit;
    let log_mean = log_time_exit.sum() / 3.0;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for i in 0..3 {
        let xc = log_time_exit[i] - log_mean;
        sxx += xc * xc;
        sxy += xc * (delta[i] - delta.sum() / 3.0);
    }
    assert!(
        (sxy / sxx - 1.0).abs() <= 1e-9,
        "pinned warp must have unit data-scale slope vs log t, got {}",
        sxy / sxx
    );
}

#[test]
fn rank1_reduced_time_warp_removes_warp_and_flags_location_log_time() {
    // The real survival regime (issue #892): a 1st-difference time penalty
    // gives a DIMENSION-1 null space — a single monotone log-t column. The
    // reduce must REMOVE the time warp entirely (zero free columns, empty
    // designs + p×0 transform, zero value/derivative offsets so `h ≡ 0`, no
    // constraint, no penalties) and instead FLAG `location_log_time_offset`,
    // so the caller carries the σ-scaled `log t` baseline on the location `q`
    // channel (u = inv_sigma·(log t − η_t)). The threshold keeps its intercept
    // (`pinned_free_row_constant == false`). A penalty `diag(0,1,1)` has the
    // 1-D null space {e0}; design column 0 is monotone in log t.
    let design_entry = array![
        [0.0, 1.0, 0.2],
        [0.405_465_108, 1.0, 0.5],
        [0.916_290_731, 1.0, 1.0]
    ];
    let design_exit = array![
        [0.0, 0.5, 0.3],
        [0.405_465_108, 1.5, 0.8],
        [0.916_290_731, 2.5, 1.4]
    ];
    let design_derivative_exit = array![[1.0, 1.0, 0.2], [0.5, 1.0, 0.3], [0.3, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let log_time_entry = array![-1.0_f64, -0.5, 0.0];
    let log_time_exit = array![0.0_f64, 0.405_465_108, 0.916_290_731];
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        true,
        log_time_entry.view(),
        log_time_exit.view(),
    )
    .expect("prepare time block");

    // Warp removed: zero free columns, empty designs + p×0 transform.
    assert_eq!(prepared.transform.gauge.reduced_total(), 0);
    assert_eq!(prepared.transform.gauge.raw_total(), 3);
    assert_eq!(prepared.design_exit.ncols(), 0);
    assert_eq!(prepared.design_entry.ncols(), 0);
    assert_eq!(prepared.design_derivative_exit.ncols(), 0);
    assert_eq!(prepared.design_exit.nrows(), 3);
    assert_eq!(prepared.initial_beta, Some(Array1::<f64>::zeros(0)));
    // No free coefficients → no derivative-guard constraint, no penalties.
    assert!(prepared.linear_constraints.is_none());
    assert!(prepared.penalties.is_empty());
    // `h ≡ 0`: zero value offsets and zero derivative offset (the warp is gone;
    // the log-t baseline lives on the location channel, not here).
    assert_eq!(prepared.offset_exit, Array1::<f64>::zeros(3));
    assert_eq!(prepared.offset_entry, Array1::<f64>::zeros(3));
    assert_eq!(prepared.derivative_offset_exit, Array1::<f64>::zeros(3));
    // No affine shift; the location-log-time flag is set.
    assert!(
        prepared
            .transform
            .gauge
            .affine_shift
            .iter()
            .all(|&v| v.abs() <= 1e-12)
    );
    assert!(
        prepared.location_log_time_offset,
        "rank-1 reduce must flag the σ-scaled log-t location baseline"
    );
    // No free time column → threshold keeps its intercept.
    assert!(!prepared.pinned_free_row_constant);
}

#[test]
fn identified_time_block_uses_structural_coefficient_constraints() {
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]]),
        design_exit: DesignMatrix::from(array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]]),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: Some(array![-0.5, 0.2, -1.5]),
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    assert_eq!(
        prepared.coefficient_lower_bounds,
        Some(array![f64::NEG_INFINITY, 0.0, 0.0])
    );
    let constraints = lower_bound_constraints(
        prepared
            .coefficient_lower_bounds
            .as_ref()
            .expect("time coefficient lower bounds"),
    )
    .expect("time coefficient constraints");
    assert_eq!(constraints.a, array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert_eq!(constraints.b, Array1::<f64>::zeros(2));
    assert_eq!(prepared.initial_beta, Some(array![-0.5, 0.2, 0.0]));
}

#[test]
fn identified_time_block_constrains_monotone_timewiggle_tail_coefficients() {
    let design_derivative_exit = array![
        [0.0, 1.0, 0.2, 0.0],
        [0.0, 1.0, 0.3, 0.0],
        [0.0, 1.0, 0.4, 0.0]
    ];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(array![
            [1.0, 0.0, 0.2, 0.0],
            [1.0, 1.0, 0.5, 0.0],
            [1.0, 2.0, 1.0, 0.0]
        ]),
        design_exit: DesignMatrix::from(array![
            [1.0, 0.5, 0.3, 0.0],
            [1.0, 1.5, 0.8, 0.0],
            [1.0, 2.5, 1.4, 0.0]
        ]),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![Array2::eye(4)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: Some(array![-0.5, 0.2, -1.5, -2.0]),
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        1,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    assert_eq!(
        prepared.coefficient_lower_bounds,
        Some(array![f64::NEG_INFINITY, 0.0, 0.0, 0.0])
    );
    assert_eq!(prepared.initial_beta, Some(array![-0.5, 0.2, 0.0, 0.0]));
}

#[test]
fn identified_time_block_rejects_offsets_below_derivative_guard() {
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]]),
        design_exit: DesignMatrix::from(array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]]),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::zeros(3),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let err = match prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    ) {
        Ok(_) => panic!("offsets below the guard must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("require derivative offsets to encode the derivative guard"),
        "unexpected error: {err}"
    );
}

#[test]
fn prepare_model_accepts_time_initializer_when_offset_completes_guard() {
    let n = 3usize;
    let derivative_guard = 5e-10;
    let derivative_offset_exit = Array1::from_elem(n, 6e-10);
    let spec = SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1.0),
        age_exit: Array1::from_elem(n, 5e9),
        event_target: array![1.0, 0.0, 1.0],
        weights: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard,
        max_iter: 4,
        tol: 1e-8,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((n, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((n, 1))),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: derivative_offset_exit.clone(),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![Array2::zeros((1, 1))],
            nullspace_dims: vec![1],
            initial_log_lambdas: None,
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        cache_mirror_sessions: Vec::new(),
    };

    let prepared = prepare_survival_location_scale_model(&spec)
        .expect("offset-supported time initializer should be accepted");
    let beta_init = prepared.blockspecs[0]
        .initial_beta
        .as_ref()
        .expect("time initializer should be present");
    let d_raw_init = Array2::ones((n, 1)).dot(beta_init) + &derivative_offset_exit;
    assert!(
        d_raw_init.iter().all(|v| *v >= derivative_guard),
        "initializer must satisfy derivative guard once offsets are included: {d_raw_init:?}"
    );
}

#[test]
fn prepare_model_seeds_structural_time_initializer_when_offset_equals_guard() {
    let n = 20usize;
    let p_time = 8usize;
    let derivative_guard = DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD;
    let derivative_offset_exit = Array1::from_elem(n, derivative_guard);
    let age_exit = Array1::from_iter((0..n).map(|i| 4.0 + (i as f64) * 14.0));
    let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        for j in 0..p_time {
            let center = (j as f64 + 0.5) / (p_time as f64);
            let x = 8.0 * (t - center);
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            design_derivative_exit[[i, j]] = 8.0 * sigmoid * (1.0 - sigmoid) / age_exit[i];
        }
    }

    let spec = SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1e-9),
        age_exit: age_exit.clone(),
        event_target: Array1::zeros(n),
        weights: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard,
        max_iter: 4,
        tol: 1e-8,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((n, p_time))),
            design_exit: DesignMatrix::from(Array2::zeros((n, p_time))),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: derivative_offset_exit.clone(),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![Array2::eye(p_time)],
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        cache_mirror_sessions: Vec::new(),
    };

    let prepared = prepare_survival_location_scale_model(&spec)
        .expect("guard-sized derivative offset should still seed time initializer");
    let beta_init = prepared.blockspecs[0]
        .initial_beta
        .as_ref()
        .expect("time initializer should be present");
    let d_raw_init = design_derivative_exit.dot(beta_init) + &derivative_offset_exit;

    assert!(beta_init.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(beta_init.iter().any(|v| *v > 0.0));
    assert!(
        d_raw_init
            .iter()
            .all(|v| v.is_finite() && *v >= derivative_guard),
        "initializer must satisfy derivative guard once offsets are included: {d_raw_init:?}"
    );
}

#[test]
fn prepare_model_assigns_distinct_descending_gauge_priorities() {
    // Regression for #366: every location-scale block previously carried
    // the uniform `gauge_priority: 100`, which made the redundant
    // intercept direction in the flat joint design un-attributable and
    // forced the identifiability audit to refuse (`fatal = true`).  The
    // four blocks must now own strictly descending priorities so the
    // surplus constant is attributed to the lower-priority block.
    let n = 4usize;
    let derivative_guard = 1e-6;
    let spec = SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1.0),
        age_exit: Array1::from_iter((0..n).map(|i| 5.0 + i as f64)),
        event_target: array![1.0, 0.0, 1.0, 1.0],
        weights: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard,
        max_iter: 4,
        tol: 1e-8,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((n, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((n, 1))),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::from_elem(n, 2e-6),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![Array2::zeros((1, 1))],
            nullspace_dims: vec![1],
            initial_log_lambdas: None,
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        cache_mirror_sessions: Vec::new(),
    };

    let prepared =
        prepare_survival_location_scale_model(&spec).expect("location-scale model prepares");

    let priority = |name: &str| {
        prepared
            .blockspecs
            .iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("missing block '{name}'"))
            .gauge_priority
    };
    let time = priority("time_transform");
    let threshold = priority("threshold");
    let log_sigma = priority("log_sigma");
    assert_eq!(
        time, 200,
        "time_transform must own the highest gauge priority"
    );
    assert!(
        time > threshold && threshold > log_sigma,
        "gauge priorities must be strictly descending so the redundant \
             intercept is attributable: time={time}, threshold={threshold}, \
             log_sigma={log_sigma}"
    );
    // The whole point of the fix: no two structural blocks may share a
    // gauge priority (equal priority is what produced the fatal audit).
    let mut seen = std::collections::HashSet::new();
    for block in &prepared.blockspecs {
        assert!(
            seen.insert(block.gauge_priority),
            "blocks must carry distinct gauge priorities; '{}' duplicates {}",
            block.name,
            block.gauge_priority,
        );
    }
}

#[test]
fn prepare_model_keeps_intercept_only_log_sigma_width() {
    let n = 4usize;
    let derivative_guard = 1e-6;
    let spec = SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1.0),
        age_exit: Array1::from_iter((0..n).map(|i| 5.0 + i as f64)),
        event_target: array![1.0, 0.0, 1.0, 1.0],
        weights: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Logistic),
        derivative_guard,
        max_iter: 4,
        tol: 1e-8,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((n, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((n, 1))),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::from_elem(n, 2e-6),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![Array2::zeros((1, 1))],
            nullspace_dims: vec![1],
            initial_log_lambdas: None,
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        cache_mirror_sessions: Vec::new(),
    };

    let prepared =
        prepare_survival_location_scale_model(&spec).expect("location-scale model prepares");
    assert_eq!(
        prepared.log_sigma_fixed_cols, 0,
        "constant log-sigma is a multiplicative free scale parameter and must not be dropped as an additive gauge"
    );
    assert_eq!(prepared.log_sigma_full_ncols, 1);
    let log_sigma = prepared
        .blockspecs
        .iter()
        .find(|block| block.name == "log_sigma")
        .expect("prepared model should contain log_sigma block");
    assert_eq!(
        log_sigma.design.ncols(),
        1,
        "intercept-only log_sigma must stay width 1 rather than canonicalizing to a zero-width block"
    );
}

#[test]
fn prepare_model_joint_audit_resolves_via_gauge_ownership() {
    // End-to-end exercise of the #366 root cause: build the three coupled
    // location-scale blocks with mutually-aliased intercept directions
    // (the exact pathology the released-0.1.135 repro hit) and confirm the
    // cross-block identifiability audit now *resolves* the rank deficiency
    // via gauge ownership instead of refusing the fit. Under the old
    // uniform `gauge_priority: 100` this same joint produced
    // `fatal = true` (`IdentifiabilityFailure`).
    use crate::identifiability::canonical::canonicalize_for_identifiability;

    let n = 8usize;
    // A shared constant column plus a per-block linear covariate. The
    // constant directions across the three blocks are exactly aliased, so
    // the flat joint design is rank-deficient by two — only resolvable by
    // a strict priority ordering.
    let mk = |col1: &[f64]| {
        let mut d = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            d[[i, 0]] = 1.0;
            d[[i, 1]] = col1[i];
        }
        DesignMatrix::from(d)
    };
    let lin: Vec<f64> = (0..n).map(|i| i as f64 - 3.5).collect();
    let quad: Vec<f64> = (0..n).map(|i| ((i as f64) - 3.5).powi(2)).collect();
    let cube: Vec<f64> = (0..n).map(|i| ((i as f64) - 3.5).powi(3)).collect();

    // Each block carries a shared constant in column 0; those three
    // constant directions are mutually aliased, so the flat joint design
    // is genuinely rank-deficient by two and only resolvable by a strict
    // priority ordering.
    let t_spec = spec_from_dense_for_test("time_transform", mk(&lin), 200);
    let thr_spec = spec_from_dense_for_test("threshold", mk(&quad), 150);
    let ls_spec = spec_from_dense_for_test("log_sigma", mk(&cube), 120);

    let specs = [t_spec, thr_spec, ls_spec];
    let canon = canonicalize_for_identifiability(&specs)
        .expect("distinct gauge priorities must resolve the aliased joint (issue #366)");
    assert!(
        !canon.audit.fatal,
        "joint audit must be non-fatal once the three blocks carry distinct \
             descending gauge priorities; summary: {}",
        canon.audit.summary,
    );
    // Raw p_total = 2+2+2 = 6; two aliased constants are dropped, so the
    // resolved joint rank is 4.
    let total_kept: usize = canon.audit.blocks.iter().map(|b| b.effective_dim).sum();
    assert_eq!(
        total_kept, 4,
        "expected joint rank 6 − 2 = 4 after gauge-attributed drops; got {total_kept}",
    );
}

fn spec_from_dense_for_test(
    name: &str,
    design: DesignMatrix,
    gauge_priority: u8,
) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design,
        offset: Array1::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

#[test]
fn identified_time_block_degenerate_entry_preserves_full_dimension() {
    let design_entry = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let design_exit = array![[0.1, 0.5, 0.9], [0.2, 0.6, 1.0], [0.3, 0.7, 1.0]];
    let design_derivative_exit = array![[0.1, 0.1, 0.0], [0.1, 0.1, 0.0], [0.1, 0.1, 0.0]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    assert_eq!(prepared.design_entry, design_entry);
    assert_eq!(prepared.design_exit, design_exit);
    assert_eq!(prepared.design_derivative_exit, design_derivative_exit);
}

#[test]
fn resolve_survival_time_anchor_defaults_to_earliest_entry() {
    let age_entry = array![5.0, 1.0, 3.0];
    let anchor = crate::families::survival::construction::resolve_survival_time_anchor_value(
        &age_entry, None,
    )
    .expect("resolve default anchor");
    assert!((anchor - 1.0).abs() <= 1e-12);
}

#[test]
fn survival_ratio_derivatives_prefer_correct_signs() {
    let dists = [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ];
    let zs = [-1.2, -0.5, 0.4, 0.6, 1.1];
    let h = 1e-6_f64;
    let tie_tol = 1e-12_f64;
    let nondeg_tol = 1e-12_f64;
    let mut saw_strict_dr = false;
    let mut saw_strict_ddr = false;

    for &dist in &dists {
        for &z in &zs {
            let r = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                f / s
            };
            let dr_plus = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                let fp = dist.pdf_derivative(u);
                let ratio = f / s;
                (ratio * ratio) + fp / s
            };
            let dr_minus = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                let fp = dist.pdf_derivative(u);
                let ratio = f / s;
                (ratio * ratio) - fp / s
            };
            let ddr_plus = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                let fp = dist.pdf_derivative(u);
                let fpp = dist.pdfsecond_derivative(u);
                let ratio = f / s;
                let dr = (ratio * ratio) + fp / s;
                (2.0 * ratio * dr) + (fpp / s + fp * f / (s * s))
            };
            let ddr_minus = |u: f64| {
                let f = dist.pdf(u);
                let s = 1.0 - dist.cdf(u);
                let fp = dist.pdf_derivative(u);
                let fpp = dist.pdfsecond_derivative(u);
                let ratio = f / s;
                let dr = (ratio * ratio) - fp / s;
                (2.0 * ratio * dr) - (fpp / s + fp * f / (s * s))
            };

            let drfd = (r(z + h) - r(z - h)) / (2.0 * h);
            let ddrfd = (dr_plus(z + h) - dr_plus(z - h)) / (2.0 * h);
            let dr_plus_err = (dr_plus(z) - drfd).abs();
            let dr_minus_err = (dr_minus(z) - drfd).abs();
            let ddr_plus_err = (ddr_plus(z) - ddrfd).abs();
            let ddr_minus_err = (ddr_minus(z) - ddrfd).abs();
            let f = dist.pdf(z);
            let s = 1.0 - dist.cdf(z);
            let fp = dist.pdf_derivative(z);
            let fpp = dist.pdfsecond_derivative(z);
            let dr_signal = (fp / s).abs();
            let ddr_signal = (fpp / s + fp * f / (s * s)).abs();

            if dr_signal > nondeg_tol {
                saw_strict_dr = true;
                assert!(
                    dr_plus_err + tie_tol < dr_minus_err,
                    "dr sign check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                    dist,
                    z,
                    dr_plus_err,
                    dr_minus_err,
                    dr_signal
                );
            } else {
                // At stationary points (fp≈0), plus/minus formulas coincide to first order.
                assert!(
                    (dr_plus_err - dr_minus_err).abs() <= tie_tol,
                    "dr tie check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                    dist,
                    z,
                    dr_plus_err,
                    dr_minus_err,
                    dr_signal
                );
            }

            if ddr_signal > nondeg_tol {
                saw_strict_ddr = true;
                assert!(
                    ddr_plus_err + tie_tol < ddr_minus_err,
                    "ddr sign check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                    dist,
                    z,
                    ddr_plus_err,
                    ddr_minus_err,
                    ddr_signal
                );
            } else {
                assert!(
                    (ddr_plus_err - ddr_minus_err).abs() <= tie_tol,
                    "ddr tie check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                    dist,
                    z,
                    ddr_plus_err,
                    ddr_minus_err,
                    ddr_signal
                );
            }
        }
    }

    assert!(
        saw_strict_dr,
        "expected at least one non-degenerate dr check"
    );
    assert!(
        saw_strict_ddr,
        "expected at least one non-degenerate ddr check"
    );
}

#[test]
fn survival_ratio_helper_matches_closed_form_identities() {
    let dists = [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ];
    let zs = [-1.4, -0.7, -0.1, 0.3, 0.9, 1.4];

    for &dist in &dists {
        for &z in &zs {
            let f = dist.pdf(z);
            let s = 1.0 - dist.cdf(z);
            let fp = dist.pdf_derivative(z);
            let fpp = dist.pdfsecond_derivative(z);

            let (r, dr) = SurvivalLocationScaleFamily::survival_ratio_first_derivative(f, fp, s);
            let ddr =
                SurvivalLocationScaleFamily::survival_ratiosecond_derivative(r, dr, f, fp, fpp, s);

            let r_expected = f / s;
            let dr_expected = (r_expected * r_expected) + fp / s;
            let ddr_expected = (2.0 * r_expected * dr_expected) + (fpp / s + fp * f / (s * s));

            assert!(
                (r - r_expected).abs() <= 1e-14,
                "r mismatch for {:?} at z={}: got {}, expected {}",
                dist,
                z,
                r,
                r_expected
            );
            assert!(
                (dr - dr_expected).abs() <= 1e-12,
                "dr mismatch for {:?} at z={}: got {}, expected {}",
                dist,
                z,
                dr,
                dr_expected
            );
            assert!(
                (ddr - ddr_expected).abs() <= 1e-10,
                "ddr mismatch for {:?} at z={}: got {}, expected {}",
                dist,
                z,
                ddr,
                ddr_expected
            );
        }
    }
}

#[test]
fn residual_pdfthird_derivative_matchessecond_derivativefd() {
    let dists = [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ];
    let zs = [-1.1, -0.4, 0.2, 0.9];
    let h = 1e-6_f64;

    for &dist in &dists {
        for &z in &zs {
            let fd =
                (dist.pdfsecond_derivative(z + h) - dist.pdfsecond_derivative(z - h)) / (2.0 * h);
            let analytic = dist.pdfthird_derivative(z);
            assert_eq!(
                analytic.signum(),
                fd.signum(),
                "pdf''' sign mismatch for {:?} at z={}: analytic={} fd={}",
                dist,
                z,
                analytic,
                fd
            );
            assert!(
                (analytic - fd).abs() < 5e-5,
                "pdf''' mismatch for {:?} at z={}: analytic={} fd={}",
                dist,
                z,
                analytic,
                fd
            );
        }
    }
}

/// #932: independent finite-difference witness of the residual-distribution
/// **fourth** PDF derivative `f''''(z)` for every residual distribution.
///
/// `pdfthird_derivative` was directly FD-guarded
/// (`residual_pdfthird_derivative_matchessecond_derivativefd`) but
/// `pdffourth_derivative` — the highest-order, most error-dense scalar tower
/// feeding the survival-LS outer-Hessian `m4` term — was only covered
/// transitively through the row-kernel oracle, where a sign slip can cancel
/// against another term. This pins it directly: a Richardson O(h⁴) central
/// difference of `pdfthird_derivative` (independent of the closed-form fourth)
/// must match `pdffourth_derivative`, and a planted sign flip must be rejected.
#[test]
fn residual_pdffourth_derivative_matches_independent_fd_witness() {
    let dists = [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ];
    let zs = [-1.3_f64, -0.5, 0.3, 1.1];
    // Richardson-extrapolated central difference of f'''(z): cancels the O(h²)
    // error of the plain central stencil, giving an O(h⁴) witness independent of
    // the analytic fourth-derivative code path.
    let central = |dist: &ResidualDistribution, z: f64, h: f64| {
        (dist.pdfthird_derivative(z + h) - dist.pdfthird_derivative(z - h)) / (2.0 * h)
    };
    for &dist in &dists {
        for &z in &zs {
            let h = 1e-3_f64;
            let coarse = central(&dist, z, h);
            let fine = central(&dist, z, h * 0.5);
            let fd = (4.0 * fine - coarse) / 3.0;
            let analytic = dist.pdffourth_derivative(z);
            assert!(
                (analytic - fd).abs() <= 1e-4 * analytic.abs().max(1.0) + 1e-7,
                "pdf'''' mismatch for {dist:?} at z={z}: analytic={analytic} fd={fd}"
            );
            // Planted-corruption tripwire: a sign flip must leave the witness band.
            if analytic.abs() > 1e-6 {
                let corrupted = -analytic;
                assert!(
                    (corrupted - fd).abs() > 1e-4 * analytic.abs().max(1.0) + 1e-7,
                    "witness failed to reject a planted pdf'''' sign flip for {dist:?} at z={z}"
                );
            }
        }
    }
}

/// #932: independent finite-difference witness of the log-survival and
/// log-pdf scalar derivative stacks across all residual links.
///
/// The survival-LS row oracle (`SurvivalLsJointNllProgram`) seeds its tower
/// from `exact_survival_neglog_derivatives_fourth_rescaled` /
/// `exact_log_pdf_derivatives_rescaled`, so it tests the Faà-di-Bruno
/// composition but TRUSTS those scalar stacks as inputs. Outside the
/// identity/probit closed-form special cases they had no general independent
/// witness. This pins each stack's d1..d4 by differencing its OWN value
/// channel (the value is independently anchored by the closed-form tests):
/// a Richardson-extrapolated central stencil of `log S(eta)` / `log f(eta)`
/// must reproduce the analytic derivative channels for logit / probit / cloglog
/// over a range of eta, and a planted sign flip must be rejected.
#[test]
fn survival_log_survival_and_pdf_stacks_match_independent_fd_witness() {
    let links = [
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog),
    ];
    let etas = [-0.8_f64, -0.2, 0.4, 1.0];

    // Richardson O(h⁴) central stencil of an arbitrary scalar f(eta) to the
    // requested derivative order (1..=4).
    fn stencil(order: usize) -> &'static [(i64, f64)] {
        match order {
            1 => &[(-1, -0.5), (1, 0.5)],
            2 => &[(-1, 1.0), (0, -2.0), (1, 1.0)],
            3 => &[(-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)],
            4 => &[(-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)],
            _ => panic!("stencil supports derivative orders 1..=4, got {order}"),
        }
    }
    let central = |value: &dyn Fn(f64) -> f64, eta: f64, order: usize, h: f64| -> f64 {
        let one = |hh: f64| {
            stencil(order)
                .iter()
                .map(|&(off, c)| c * value(eta + (off as f64) * hh))
                .sum::<f64>()
                / hh.powi(order as i32)
        };
        (4.0 * one(h * 0.5) - one(h)) / 3.0
    };

    for link in &links {
        // log S(eta): value = slot 0; analytic derivatives are -r, -dr, -ddr, -dddr.
        let log_s_value = |eta: f64| {
            SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
                link, eta,
            )
            .expect("log-survival stack")
            .0
        };
        // log f(eta): value = slot 0; analytic derivatives are d1..d4.
        let log_pdf_value = |eta: f64| {
            SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(link, eta, 0.0)
                .expect("log-pdf stack")
                .0
        };
        for &eta in &etas {
            let (_, r, dr, ddr, dddr) =
                SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
                    link, eta,
                )
                .expect("log-survival stack");
            let log_s_analytic = [-r, -dr, -ddr, -dddr];
            let (_, p1, p2, p3, p4) =
                SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(link, eta, 0.0)
                    .expect("log-pdf stack");
            let log_pdf_analytic = [p1, p2, p3, p4];

            for (k, &analytic) in log_s_analytic.iter().enumerate() {
                let order = k + 1;
                let h = if order <= 2 { 1e-3 } else { 3e-3 };
                let fd = central(&log_s_value, eta, order, h);
                assert!(
                    (analytic - fd).abs() <= 5e-4 * analytic.abs().max(1.0) + 1e-6,
                    "logS d{order} mismatch for {link:?} at eta={eta}: analytic={analytic} fd={fd}"
                );
                if analytic.abs() > 1e-5 {
                    assert!(
                        (-analytic - fd).abs() > 5e-4 * analytic.abs().max(1.0) + 1e-6,
                        "witness failed to reject logS d{order} sign flip for {link:?} at eta={eta}"
                    );
                }
            }
            for (k, &analytic) in log_pdf_analytic.iter().enumerate() {
                let order = k + 1;
                let h = if order <= 2 { 1e-3 } else { 3e-3 };
                let fd = central(&log_pdf_value, eta, order, h);
                assert!(
                    (analytic - fd).abs() <= 5e-4 * analytic.abs().max(1.0) + 1e-6,
                    "logpdf d{order} mismatch for {link:?} at eta={eta}: analytic={analytic} fd={fd}"
                );
            }
        }
    }
}

#[test]
fn exact_log_pdf_derivatives_match_probit_closed_form() {
    let eta = 3.25;
    let (logf, d1, d2, d3, d4) = SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
        &InverseLink::Standard(StandardLink::Probit),
        eta,
        0.0,
    )
    .expect("exact probit log-pdf derivatives");
    let expected_logf = -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!((logf - expected_logf).abs() <= 1e-15);
    assert!((d1 + eta).abs() <= 1e-15);
    assert!((d2 + 1.0).abs() <= 1e-15);
    assert_eq!(d3, 0.0);
    assert_eq!(d4, 0.0);
}

#[test]
fn exact_log_pdf_derivatives_rescaled_scale_cloglog_uniformly() {
    let eta = 501.0;
    let log_scale = 1.0;
    let (logf, d1, d2, d3, d4) = SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
        &InverseLink::Standard(StandardLink::CLogLog),
        eta,
        log_scale,
    )
    .expect("rescaled cloglog log-pdf derivatives");
    let (unscaled_logf, u1, u2, u3, u4) =
        SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
            &InverseLink::Standard(StandardLink::CLogLog),
            eta,
            0.0,
        )
        .expect("unscaled cloglog log-pdf derivatives");
    let scale = (-log_scale).exp();
    let expected_d1 = scale * u1;
    let expected_d2 = scale * u2;
    let expected_d3 = scale * u3;
    let expected_d4 = scale * u4;

    assert_eq!(logf, unscaled_logf);
    assert!((d1 - expected_d1).abs() <= 1e-12 * expected_d1.abs());
    assert!((d2 - expected_d2).abs() <= 1e-12 * expected_d2.abs());
    assert!((d3 - expected_d3).abs() <= 1e-12 * expected_d3.abs());
    assert!((d4 - expected_d4).abs() <= 1e-12 * expected_d4.abs());
}

#[test]
fn exact_survival_neglog_derivatives_rescaled_do_not_scale_cloglog_ratio() {
    let eta = 2.25_f64;
    let log_scale = 1.5_f64;
    let expected = eta.exp();

    let (log_s, r, dr, ddr, dddr) =
        SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
            &InverseLink::Standard(StandardLink::CLogLog),
            eta,
        )
        .expect("rescaled cloglog survival derivatives");

    assert!((log_s + expected).abs() <= 1e-15 * expected);
    for (label, actual) in [("r", r), ("dr", dr), ("ddr", ddr), ("dddr", dddr)] {
        assert!(
            (actual - expected).abs() <= 1e-15 * expected,
            "CLogLog survival ratio derivative {label} must ignore deriv_log_scale: actual={actual} expected={expected}"
        );
    }

    let ((pair_log_s, pair_r, pair_dr, pair_ddr, pair_dddr), _) =
        SurvivalLocationScaleFamily::clglog_exit_pair(eta, log_scale);
    assert!((pair_log_s + expected).abs() <= 1e-15 * expected);
    for (label, actual) in [
        ("pair r", pair_r),
        ("pair dr", pair_dr),
        ("pair ddr", pair_ddr),
        ("pair dddr", pair_dddr),
    ] {
        assert!(
            (actual - expected).abs() <= 1e-15 * expected,
            "fused CLogLog survival ratio derivative {label} must ignore deriv_log_scale: actual={actual} expected={expected}"
        );
    }
}

#[test]
fn exact_survival_neglog_derivatives_match_identity_closed_form() {
    let eta = 0.25;
    let s = 1.0 - eta;
    let inv = 1.0 / s;
    let (log_s, r, dr, ddr, dddr) =
        SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
            &InverseLink::Standard(StandardLink::Identity),
            eta,
        )
        .expect("exact identity survival derivatives");
    assert!((log_s - s.ln()).abs() <= 1e-15);
    assert!((r - inv).abs() <= 1e-15);
    assert!((dr - inv * inv).abs() <= 1e-15);
    assert!((ddr - 2.0 * inv.powi(3)).abs() <= 1e-15);
    assert!((dddr - 6.0 * inv.powi(4)).abs() <= 1e-12);
}

#[test]
fn survival_log_likelihood_only_matches_sum_of_exact_row_kernels() {
    let family = survival_exact_newton_test_family();
    let states = survival_exact_newton_rebuild_states(&array![0.1], &array![0.2], &array![-0.15]);
    let (h0, h1, d_raw, ..) = family.validate_joint_states(&states).expect("joint states");
    let dynamic = family
        .build_dynamic_geometry(&states)
        .expect("dynamic geometry");

    let mut row_sum = 0.0;
    for i in 0..family.n {
        let state = family.row_predictor_state(
            h0[i],
            h1[i],
            d_raw[i],
            dynamic.q_entry[i],
            dynamic.q_exit[i],
            dynamic.qdot_exit[i],
        );
        if let Some(kernel) = family.exact_row_kernel(i, state).expect("exact row kernel") {
            row_sum += kernel.log_likelihood();
        }
    }

    let scalar = family
        .log_likelihood_only(&states)
        .expect("scalar log-likelihood");
    assert!(
        (scalar - row_sum).abs() < 1e-12,
        "scalar survival log-likelihood should equal the sum of exact row kernels; scalar={} row_sum={}",
        scalar,
        row_sum
    );
}

#[test]
fn survival_joint_gradient_evaluation_matches_evaluate_block_gradients() {
    let family = survival_exact_newton_test_family();
    let states = survival_exact_newton_rebuild_states(&array![0.2], &array![0.35], &array![-0.15]);
    let specs = survival_outergradient_testspecs();
    let joint = family
        .exact_newton_joint_gradient_evaluation(&states, &specs)
        .expect("joint gradient evaluation")
        .expect("survival location-scale should provide joint gradient");
    let eval = family.evaluate(&states).expect("full evaluate");

    assert!((joint.log_likelihood - eval.log_likelihood).abs() <= 1e-12);

    let mut expected = Array1::<f64>::zeros(joint.gradient.len());
    let mut offset = 0usize;
    for (spec, work) in specs.iter().zip(eval.blockworking_sets.iter()) {
        let width = spec.design.ncols();
        let BlockWorkingSet::ExactNewton { gradient, .. } = work else {
            panic!("survival location-scale blocks should use exact Newton");
        };
        expected
            .slice_mut(s![offset..offset + width])
            .assign(gradient);
        offset += width;
    }

    for (actual, expected) in joint.gradient.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

#[test]
fn survival_exact_row_kernel_rejects_invalid_event_target_instead_of_clamping() {
    let mut family = survival_exact_newton_test_family();
    family.y[0] = 1.5;
    let states = survival_exact_newton_rebuild_states(&array![0.1], &array![0.2], &array![-0.15]);
    let err = match family.log_likelihood_only(&states) {
        Ok(_) => panic!("invalid event target should error"),
        Err(err) => err,
    };
    assert!(
        err.contains("event target must lie in [0,1]"),
        "expected explicit event-target validation error, got: {err}"
    );
}

#[test]
fn logwith_derivatives_positive_matches_exact_log() {
    let x = 0.25;
    let (log_x, d1, d2, d3, d4) = SurvivalLocationScaleFamily::logwith_derivatives_positive(x);
    assert!((log_x - x.ln()).abs() <= 1e-15);
    assert!((d1 - 1.0 / x).abs() <= 1e-15);
    assert!((d2 + 1.0 / (x * x)).abs() <= 1e-15);
    assert!((d3 - 2.0 / (x * x * x)).abs() <= 1e-15);
    assert!((d4 + 6.0 / (x * x * x * x)).abs() <= 1e-12);
}

#[test]
fn inverse_link_survival_prob_complements_failure_prob() {
    let eta = 0.37;
    let failure = inverse_link_failure_prob_checked(
        &residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        eta,
    )
    .expect("failure probability");
    let survival = inverse_link_survival_prob_checked(
        &residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        eta,
    )
    .expect("survival probability");
    assert!((survival - (1.0 - failure)).abs() <= 1e-14);
}

#[test]
fn lift_conditional_covariance_rejects_time_map_wider_than_raw() {
    let z = array![[1.0, 0.0]];
    let time_gauge = Gauge::from_block_transforms(&[z]);
    let cov_reduced = Array2::<f64>::eye(2);
    let err = lift_conditional_covariance(&cov_reduced, &time_gauge, 0, 0, 0, 0, 0, 0, 0)
        .expect_err(
            "a reduced time block wider than the raw time map must fail before ndarray assignment",
        );
    assert!(
        err.contains("time map is wider than tall"),
        "unexpected covariance-lift error: {err}"
    );
}

#[test]
fn lift_conditional_covariance_preserveswiggle_block() {
    let z = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]];
    let time_gauge = Gauge::from_block_transforms(&[z]);
    let cov_reduced = array![
        [2.0, 0.1, 0.2, 0.3, 0.4],
        [0.1, 3.0, 0.5, 0.6, 0.7],
        [0.2, 0.5, 4.0, 0.8, 0.9],
        [0.3, 0.6, 0.8, 5.0, 1.1],
        [0.4, 0.7, 0.9, 1.1, 6.0],
    ];
    let lifted = lift_conditional_covariance(&cov_reduced, &time_gauge, 1, 1, 0, 1, 1, 0, 1)
        .expect("covariance lift");
    assert_eq!(lifted.dim(), (6, 6));
    assert!((lifted[[5, 5]] - 6.0).abs() <= 1e-12);
    assert!((lifted[[0, 5]] - 0.4).abs() <= 1e-12);
    assert!((lifted[[3, 5]] - 0.9).abs() <= 1e-12);
    assert!((lifted[[4, 5]] - 1.1).abs() <= 1e-12);
}

#[test]
fn weighted_crossprod_dense_falls_back_when_row_scaled_product_would_overflow() {
    let left = array![[1.0e-200]];
    let right = array![[1.0e200]];
    let weights = array![1.0e200];

    let cross = weighted_crossprod_dense(&left, &weights, &right)
        .expect("stable weighted cross-product should avoid overflow");
    let expected = 1.0e200;
    let rel_err = ((cross[[0, 0]] - expected) / expected).abs();

    assert!(cross[[0, 0]].is_finite());
    assert!(
        rel_err <= 1e-12,
        "unexpected weighted cross-product: {}",
        cross[[0, 0]]
    );
}

#[test]
fn scale_dense_rows_saturates_without_nan_when_coefficients_are_huge() {
    let mat = array![[1.0e200], [2.0e-200]];
    let coeffs = array![1.0e200, 1.0e200];

    let scaled = scale_dense_rows(&mat, &coeffs)
        .expect("row scaling should saturate overflow instead of producing NaN");

    assert!(scaled.iter().all(|value| value.is_finite()));
    assert!(scaled[[0, 0]] > 1.0e300);
    assert!((scaled[[1, 0]] - 2.0).abs() <= 1e-12);
}

#[test]
fn threshold_exact_newton_hessian_matches_negative_gradient_jacobian() {
    let family = survival_exact_newton_test_family();
    let beta_t = 0.35;
    let states = survival_exact_newton_threshold_states(beta_t);
    let eval = family.evaluate(&states).expect("evaluate at center");
    let BlockWorkingSet::ExactNewton { gradient, hessian } =
        &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
    else {
        panic!("threshold block should use exact newton");
    };
    let hessian = hessian.to_dense();

    let eps = 1e-6;
    let eval_plus = family
        .evaluate(&survival_exact_newton_threshold_states(beta_t + eps))
        .expect("evaluate at beta + eps");
    let eval_minus = family
        .evaluate(&survival_exact_newton_threshold_states(beta_t - eps))
        .expect("evaluate at beta - eps");
    let grad_plus = match &eval_plus.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
    {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("threshold block should use exact newton"),
    };
    let grad_minus =
        match &eval_minus.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("threshold block should use exact newton"),
        };
    let fd_neggrad_jac = -(grad_plus - grad_minus) / (2.0 * eps);

    assert!(
        (gradient[0]).is_finite() && hessian[[0, 0]].is_finite(),
        "non-finite threshold exact-newton quantities: grad={} hess={}",
        gradient[0],
        hessian[[0, 0]]
    );
    assert_eq!(
        hessian[[0, 0]].signum(),
        fd_neggrad_jac.signum(),
        "threshold Hessian sign mismatch: analytic={} fd={}",
        hessian[[0, 0]],
        fd_neggrad_jac
    );
    assert!(
        (hessian[[0, 0]] - fd_neggrad_jac).abs() <= 1e-5,
        "threshold Hessian mismatch: analytic={} fd={}",
        hessian[[0, 0]],
        fd_neggrad_jac
    );
}

#[test]
fn log_sigma_exact_newton_hessian_matches_negative_gradient_jacobian() {
    let family = survival_exact_newton_test_familywith_inverse_link(
        residual_distribution_inverse_link(ResidualDistribution::Logistic),
    );
    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
    let eval = family.evaluate(&states).expect("evaluate at center");
    let BlockWorkingSet::ExactNewton { hessian, .. } =
        &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA]
    else {
        panic!("log-sigma block should use exact newton");
    };
    let hessian = hessian.to_dense();

    let eps = 1e-6;
    let grad_at = |beta_ls: f64| -> f64 {
        let eval = family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                &array![beta_ls],
            ))
            .expect("evaluate shifted log-sigma");
        match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("log-sigma block should use exact newton"),
        }
    };
    let fd_neggrad_jac =
        -(grad_at(beta_log_sigma[0] + eps) - grad_at(beta_log_sigma[0] - eps)) / (2.0 * eps);

    assert_eq!(
        hessian[[0, 0]].signum(),
        fd_neggrad_jac.signum(),
        "log-sigma Hessian sign mismatch: analytic={} fd={}",
        hessian[[0, 0]],
        fd_neggrad_jac
    );
    assert!(
        (hessian[[0, 0]] - fd_neggrad_jac).abs() <= 1e-5,
        "log-sigma Hessian mismatch: analytic={} fd={}",
        hessian[[0, 0]],
        fd_neggrad_jac
    );
}

#[test]
fn exact_newton_block_directional_derivatives_matchfd_for_non_probit_links() {
    let extracthessian = |eval: FamilyEvaluation, block_idx: usize| -> Array2<f64> {
        match &eval.blockworking_sets[block_idx] {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
        }
    };

    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let eps = 1e-6;

    for (label, inverse_link) in survival_non_probit_test_links() {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let base_eval = family.evaluate(&states).expect("base eval");

        for (block_idx, direction) in [
            (SurvivalLocationScaleFamily::BLOCK_TIME, array![1.0]),
            (SurvivalLocationScaleFamily::BLOCK_THRESHOLD, array![1.0]),
            (SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA, array![1.0]),
        ] {
            let analytic = family
                .exact_newton_hessian_directional_derivative(&states, block_idx, &direction)
                .expect("analytic dH")
                .expect("expected exact dH");

            let mut beta_time_plus = beta_time.clone();
            let mut beta_threshold_plus = beta_threshold.clone();
            let mut beta_log_sigma_plus = beta_log_sigma.clone();
            match block_idx {
                SurvivalLocationScaleFamily::BLOCK_TIME => {
                    beta_time_plus += &(eps * &direction);
                }
                SurvivalLocationScaleFamily::BLOCK_THRESHOLD => {
                    beta_threshold_plus += &(eps * &direction);
                }
                SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA => {
                    beta_log_sigma_plus += &(eps * &direction);
                }
                _ => panic!("unexpected block"),
            }

            let plus_states = survival_exact_newton_rebuild_states(
                &beta_time_plus,
                &beta_threshold_plus,
                &beta_log_sigma_plus,
            );
            let h_plus =
                extracthessian(family.evaluate(&plus_states).expect("plus eval"), block_idx);
            let h_base = extracthessian(base_eval.clone(), block_idx);
            let fd = (h_plus - h_base) / eps;
            crate::test_support::assert_matrix_derivativefd(
                &fd,
                &analytic,
                5e-4,
                &format!("survival {label} block {} dH", block_idx),
            );
        }
    }
}

#[test]
fn joint_exact_newton_hessian_matches_negative_gradient_jacobian_for_non_probit_links() {
    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let eps = 1e-6;

    for (label, inverse_link) in survival_non_probit_test_links() {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let analytic = family
            .exact_newton_joint_hessian(&states)
            .expect("joint exact hessian")
            .expect("expected exact joint hessian");

        let flattengrad = |eval: FamilyEvaluation| -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(3);
            for (block_idx, slot) in out.iter_mut().enumerate() {
                *slot = match &eval.blockworking_sets[block_idx] {
                    BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                    BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
                };
            }
            out
        };

        let mut fd = Array2::<f64>::zeros((3, 3));
        for j in 0..3 {
            let mut beta_time_plus = beta_time.clone();
            let mut beta_threshold_plus = beta_threshold.clone();
            let mut beta_log_sigma_plus = beta_log_sigma.clone();
            let mut beta_time_minus = beta_time.clone();
            let mut beta_threshold_minus = beta_threshold.clone();
            let mut beta_log_sigma_minus = beta_log_sigma.clone();
            match j {
                0 => {
                    beta_time_plus[0] += eps;
                    beta_time_minus[0] -= eps;
                }
                1 => {
                    beta_threshold_plus[0] += eps;
                    beta_threshold_minus[0] -= eps;
                }
                2 => {
                    beta_log_sigma_plus[0] += eps;
                    beta_log_sigma_minus[0] -= eps;
                }
                other => panic!("FD probe block {other} out of range (expected 0..3)"),
            }
            let grad_plus = flattengrad(
                family
                    .evaluate(&survival_exact_newton_rebuild_states(
                        &beta_time_plus,
                        &beta_threshold_plus,
                        &beta_log_sigma_plus,
                    ))
                    .expect("eval plus"),
            );
            let grad_minus = flattengrad(
                family
                    .evaluate(&survival_exact_newton_rebuild_states(
                        &beta_time_minus,
                        &beta_threshold_minus,
                        &beta_log_sigma_minus,
                    ))
                    .expect("eval minus"),
            );
            let col = -(grad_plus - grad_minus) / (2.0 * eps);
            fd.column_mut(j).assign(&col);
        }

        crate::test_support::assert_matrix_derivativefd(
            &fd,
            &analytic,
            2e-4,
            &format!("survival {label} joint H"),
        );
    }
}

#[test]
fn joint_exact_newton_score_matches_loglikelihoodfd_for_non_probit_links() {
    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let eps = 1e-6;

    for (label, inverse_link) in survival_non_probit_test_links() {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let eval = family.evaluate(&states).expect("evaluate");
        let analytic = Array1::from_vec(vec![
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_TIME] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
        ]);

        let objective = |bt: &Array1<f64>, bth: &Array1<f64>, bls: &Array1<f64>| -> f64 {
            family
                .evaluate(&survival_exact_newton_rebuild_states(bt, bth, bls))
                .expect("eval objective")
                .log_likelihood
        };

        let mut fd = Array1::<f64>::zeros(3);
        fd[0] = (objective(
            &array![beta_time[0] + eps],
            &beta_threshold,
            &beta_log_sigma,
        ) - objective(
            &array![beta_time[0] - eps],
            &beta_threshold,
            &beta_log_sigma,
        )) / (2.0 * eps);
        fd[1] = (objective(
            &beta_time,
            &array![beta_threshold[0] + eps],
            &beta_log_sigma,
        ) - objective(
            &beta_time,
            &array![beta_threshold[0] - eps],
            &beta_log_sigma,
        )) / (2.0 * eps);
        fd[2] = (objective(
            &beta_time,
            &beta_threshold,
            &array![beta_log_sigma[0] + eps],
        ) - objective(
            &beta_time,
            &beta_threshold,
            &array![beta_log_sigma[0] - eps],
        )) / (2.0 * eps);

        for j in 0..3 {
            let abs = (analytic[j] - fd[j]).abs();
            if analytic[j].abs().max(fd[j].abs()) >= 1e-8 {
                assert_eq!(
                    analytic[j].signum(),
                    fd[j].signum(),
                    "survival {label} joint score sign mismatch at {j}: analytic={} fd={}",
                    analytic[j],
                    fd[j]
                );
            }
            assert!(
                abs <= 1e-5,
                "survival {label} joint score mismatch at {j}: analytic={} fd={} abs={}",
                analytic[j],
                fd[j],
                abs
            );
        }
    }
}

#[test]
fn joint_exact_newton_log_sigma_block_matches_fd_in_far_exp_tail() {
    let family = survival_exact_newton_test_family();
    let beta_time = array![0.2];
    let beta_threshold = array![0.1 * crate::families::sigma_link::safe_exp(700.0)];
    let beta_log_sigma0 = 701.0_f64;
    let beta_log_sigma = array![beta_log_sigma0];

    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
    let eval = family.evaluate(&states).expect("evaluate");
    let (analytic_score, analytic_info) =
        match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                (gradient[0], hessian.to_dense()[[0, 0]])
            }
            _ => panic!("expected exact newton log-sigma block"),
        };

    let objective = |beta_ls: &Array1<f64>| -> f64 {
        family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                beta_ls,
            ))
            .expect("eval objective")
            .log_likelihood
    };
    let h = 1e-4;
    let ll_plus = objective(&array![beta_log_sigma0 + h]);
    let ll0 = objective(&array![beta_log_sigma0]);
    let ll_minus = objective(&array![beta_log_sigma0 - h]);
    let score_fd = (ll_plus - ll_minus) / (2.0 * h);
    let info_fd = -(ll_plus - 2.0 * ll0 + ll_minus) / (h * h);
    assert!(
        (analytic_score - score_fd).abs() < 1e-8,
        "the exact-newton survival log-sigma score should match the far-tail finite difference at beta_log_sigma={beta_log_sigma0}; got {} vs {}",
        analytic_score,
        score_fd
    );
    assert!(
        (analytic_info - info_fd).abs() < 1e-5,
        "the exact-newton survival log-sigma information should match the far-tail finite difference at beta_log_sigma={beta_log_sigma0}; got {} vs {}",
        analytic_info,
        info_fd
    );
}

#[test]
fn survival_q_chain_derivatives_match_exact_exp_link_in_far_tails() {
    let eta_t = 2.0;
    for &eta_ls in &[701.0_f64, -30.0_f64] {
        let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
        let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) = q_chain_derivs_scalar(eta_t, eta_ls);
        assert!((q_t + inv_sigma).abs() <= 1e-15);
        assert!((q_ls - eta_t * inv_sigma).abs() <= 1e-15);
        assert!((q_tl - inv_sigma).abs() <= 1e-15);
        assert!((q_ll + eta_t * inv_sigma).abs() <= 1e-15);
        assert!((q_tl_ls + inv_sigma).abs() <= 1e-15);
        assert!((q_ll_ls - eta_t * inv_sigma).abs() <= 1e-15);
        let h = 1e-6;
        let q = |ls: f64| -eta_t * exp_sigma_inverse_from_eta_scalar(ls);
        let q_fd = (q(eta_ls + h) - q(eta_ls - h)) / (2.0 * h);
        assert!(
            (q_ls - q_fd).abs() <= (1e-8 * q_fd.abs()).max(1e-8),
            "q_s finite difference mismatch at eta_ls={eta_ls}: analytic={q_ls} fd={q_fd}"
        );
    }
}

#[test]
fn survival_exact_log_sigma_dh_matches_far_tail_third_derivative() {
    let family = survival_exact_newton_test_family();
    let beta_time = array![0.2];
    let beta_threshold = array![0.1 * crate::families::sigma_link::safe_exp(700.0)];
    let beta_log_sigma0 = 701.0_f64;
    let beta_log_sigma = array![beta_log_sigma0];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

    let analytic = family
        .exact_newton_hessian_directional_derivative(
            &states,
            SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA,
            &array![1.0],
        )
        .expect("analytic dH")
        .expect("expected exact dH");

    let objective = |beta_ls: f64| -> f64 {
        family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                &array![beta_ls],
            ))
            .expect("eval objective")
            .log_likelihood
    };
    let h = 1e-4_f64;
    let fd3 = (objective(beta_log_sigma0 + 2.0 * h) - 2.0 * objective(beta_log_sigma0 + h)
        + 2.0 * objective(beta_log_sigma0 - h)
        - objective(beta_log_sigma0 - 2.0 * h))
        / (2.0 * h.powi(3));
    assert!(
        (analytic[[0, 0]] + fd3).abs() < 1e-3,
        "the exact-newton survival log-sigma dH entry should equal the negative third derivative in the far tail at beta_log_sigma={beta_log_sigma0}; got analytic {} vs expected {}",
        analytic[[0, 0]],
        -fd3
    );
}

#[test]
fn survival_joint_exact_log_sigma_dh_matches_far_tail_third_derivative() {
    let family = survival_exact_newton_test_family();
    let beta_time = array![0.2];
    let beta_threshold = array![0.1 * crate::families::sigma_link::safe_exp(700.0)];
    let beta_log_sigma0 = 701.0_f64;
    let beta_log_sigma = array![beta_log_sigma0];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &array![0.0, 0.0, 1.0])
        .expect("analytic joint dH")
        .expect("expected exact joint dH");

    let objective = |beta_ls: f64| -> f64 {
        family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                &array![beta_ls],
            ))
            .expect("eval objective")
            .log_likelihood
    };
    let h = 1e-4_f64;
    let fd3 = (objective(beta_log_sigma0 + 2.0 * h) - 2.0 * objective(beta_log_sigma0 + h)
        + 2.0 * objective(beta_log_sigma0 - h)
        - objective(beta_log_sigma0 - 2.0 * h))
        / (2.0 * h.powi(3));
    assert!(
        (analytic[[2, 2]] + fd3).abs() < 1e-3,
        "the exact joint survival dH log-sigma/log-sigma entry should equal the negative third derivative in the far tail at beta_log_sigma={beta_log_sigma0}; got analytic {} vs expected {}",
        analytic[[2, 2]],
        -fd3
    );
}

#[test]
fn joint_exact_newton_score_matches_loglikelihoodfd_near_fitted_non_probit_points() {
    let eps = 1e-6;
    let cases = vec![
        (
            "logistic-near-fit",
            residual_distribution_inverse_link(ResidualDistribution::Logistic),
            array![0.7746886451475979],
            array![-0.6407086184606554],
            array![-0.15],
        ),
        (
            "cloglog-near-fit",
            residual_distribution_inverse_link(ResidualDistribution::Gumbel),
            array![0.8153913537182474],
            array![14.123707996892579],
            array![1.4355329717917449],
        ),
    ];

    for (label, inverse_link, beta_time, beta_threshold, beta_log_sigma) in cases {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let eval = family.evaluate(&states).expect("evaluate");
        let analytic = Array1::from_vec(vec![
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_TIME] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
        ]);

        let objective = |bt: &Array1<f64>, bth: &Array1<f64>, bls: &Array1<f64>| -> f64 {
            family
                .evaluate(&survival_exact_newton_rebuild_states(bt, bth, bls))
                .expect("eval objective")
                .log_likelihood
        };

        let mut fd = Array1::<f64>::zeros(3);
        fd[0] = (objective(
            &array![beta_time[0] + eps],
            &beta_threshold,
            &beta_log_sigma,
        ) - objective(
            &array![beta_time[0] - eps],
            &beta_threshold,
            &beta_log_sigma,
        )) / (2.0 * eps);
        fd[1] = (objective(
            &beta_time,
            &array![beta_threshold[0] + eps],
            &beta_log_sigma,
        ) - objective(
            &beta_time,
            &array![beta_threshold[0] - eps],
            &beta_log_sigma,
        )) / (2.0 * eps);
        fd[2] = (objective(
            &beta_time,
            &beta_threshold,
            &array![beta_log_sigma[0] + eps],
        ) - objective(
            &beta_time,
            &beta_threshold,
            &array![beta_log_sigma[0] - eps],
        )) / (2.0 * eps);

        for j in 0..3 {
            let abs = (analytic[j] - fd[j]).abs();
            if analytic[j].abs().max(fd[j].abs()) >= 1e-8 {
                assert_eq!(
                    analytic[j].signum(),
                    fd[j].signum(),
                    "survival {label} joint score sign mismatch at {j}: analytic={} fd={}",
                    analytic[j],
                    fd[j]
                );
            }
            assert!(
                abs <= 5e-4,
                "survival {label} joint score mismatch at {j}: analytic={} fd={} abs={}",
                analytic[j],
                fd[j],
                abs
            );
        }
    }
}

#[test]
fn row_derivative_identities_hold_for_non_probit_links() {
    let beta_time = array![0.8153913537182474];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![0.4];

    for (label, inverse_link) in survival_non_probit_test_links() {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let (h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry, .., etaw) =
            family.validate_joint_states(&states).expect("joint states");
        // For time-invariant blocks, eta_ls_entry == eta_ls_exit.
        let inv_sigma = eta_ls_exit.mapv(exp_sigma_inverse_from_eta_scalar);
        let inv_sigma_entry = eta_ls_entry.mapv(exp_sigma_inverse_from_eta_scalar);

        for i in 0..family.n {
            let state = family.row_predictor_state(
                h0[i],
                h1[i],
                d_raw[i],
                -eta_t_entry[i] * inv_sigma_entry[i] + etaw.map_or(0.0, |w| w[i]),
                -eta_t_exit[i] * inv_sigma[i] + etaw.map_or(0.0, |w| w[i]),
                0.0,
            );
            let row = family
                .row_derivatives(i, state)
                .expect("row derivatives")
                .expect("active row");

            let ell_h0 = row.grad_time_eta_h0;
            let ell_h1 = row.grad_time_eta_h1;
            let ell_q = row.d1_q;
            let ell_h0q = row.h_time_h0;
            let ell_h1q = row.h_time_h1;
            let ell_qq = row.d2_q;
            assert!(
                (ell_q - ell_h0 - ell_h1).abs() <= 1e-10,
                "survival {label} row {i} violated ell_q = ell_h0 + ell_h1: q={} h0={} h1={}",
                ell_q,
                ell_h0,
                ell_h1
            );
            assert!(
                (ell_qq - ell_h0q - ell_h1q).abs() <= 1e-10,
                "survival {label} row {i} violated ell_qq = ell_h0q + ell_h1q: qq={} h0q={} h1q={}",
                ell_qq,
                ell_h0q,
                ell_h1q
            );
        }
    }
}

#[test]
fn posterior_mean_prediction_matches_deterministicwhen_covariance_iszero() {
    let input = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5]],
        eta_time_offset_exit: array![0.2],
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, -0.2
        ]])),
        eta_threshold_offset: array![0.0],
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.3
        ]])),
        eta_log_sigma_offset: array![0.0],
        x_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
    };
    let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
    let deterministic = predict_survival_location_scale(&input, &fit).expect("predict");
    let expected = inverse_link_survival_prob_checked(&input.inverse_link, deterministic.eta[0])
        .expect("expected survival");
    assert!((deterministic.survival_prob[0] - expected).abs() <= 1e-12);
    let posterior = predict_survival_location_scalewith_uncertainty(
        &input,
        &fit,
        &Array2::zeros((6, 6)),
        true,
        false,
    )
    .expect("posterior mean");
    assert!((deterministic.survival_prob[0] - posterior.survival_prob[0]).abs() <= 1e-10);
    let uncertainty = predict_survival_location_scalewith_uncertainty(
        &input,
        &fit,
        &Array2::zeros((6, 6)),
        false,
        true,
    )
    .expect("uncertainty");
    assert!(
        uncertainty
            .response_standard_error
            .as_ref()
            .expect("response sd")[0]
            <= 1e-12
    );
}

#[test]
fn sparse_exact_newton_matches_denseworking_sets() {
    let dense_family = survival_exact_newton_test_family();
    let sparse_family = sparse_survival_exact_newton_test_family();
    let states = survival_exact_newton_threshold_states(0.35);

    let dense_eval = dense_family.evaluate(&states).expect("dense evaluate");
    let sparse_eval = sparse_family.evaluate(&states).expect("sparse evaluate");
    assert!((dense_eval.log_likelihood - sparse_eval.log_likelihood).abs() <= 1e-12);
    assert_eq!(
        dense_eval.blockworking_sets.len(),
        sparse_eval.blockworking_sets.len()
    );
    for (dense_block, sparse_block) in dense_eval
        .blockworking_sets
        .iter()
        .zip(sparse_eval.blockworking_sets.iter())
    {
        match (dense_block, sparse_block) {
            (
                BlockWorkingSet::ExactNewton {
                    gradient: dense_g,
                    hessian: dense_h,
                },
                BlockWorkingSet::ExactNewton {
                    gradient: sparse_g,
                    hessian: sparse_h,
                },
            ) => {
                let dense_h = dense_h.to_dense();
                let sparse_h = sparse_h.to_dense();
                assert_eq!(dense_g.len(), sparse_g.len());
                assert_eq!(dense_h.dim(), sparse_h.dim());
                for i in 0..dense_g.len() {
                    assert!((dense_g[i] - sparse_g[i]).abs() <= 1e-12);
                }
                for i in 0..dense_h.nrows() {
                    for j in 0..dense_h.ncols() {
                        assert!((dense_h[[i, j]] - sparse_h[[i, j]]).abs() <= 1e-12);
                    }
                }
            }
            _ => panic!("expected exact-newton blocks"),
        }
    }

    let direction = array![0.2];
    let dense_dh = dense_family
        .exact_newton_hessian_directional_derivative(&states, 1, &direction)
        .expect("dense directional derivative")
        .expect("dense threshold directional derivative");
    let sparse_dh = sparse_family
        .exact_newton_hessian_directional_derivative(&states, 1, &direction)
        .expect("sparse directional derivative")
        .expect("sparse threshold directional derivative");
    assert_eq!(dense_dh.dim(), sparse_dh.dim());
    for i in 0..dense_dh.nrows() {
        for j in 0..dense_dh.ncols() {
            assert!((dense_dh[[i, j]] - sparse_dh[[i, j]]).abs() <= 1e-12);
        }
    }
}

#[test]
fn prediction_applies_threshold_and_log_sigma_offsets() {
    let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
    let input = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5]],
        eta_time_offset_exit: array![0.2],
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, -0.2
        ]])),
        eta_threshold_offset: array![0.7],
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.3
        ]])),
        eta_log_sigma_offset: array![0.4],
        x_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
    };
    let pred = predict_survival_location_scale(&input, &fit).expect("predict");

    let eta_t = array![1.0, -0.2].dot(&fit.beta_threshold()) + input.eta_threshold_offset[0];
    let eta_ls = array![1.0, 0.3].dot(&fit.beta_log_sigma()) + input.eta_log_sigma_offset[0];
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    let h = array![1.0, 0.5].dot(&fit.beta_time()) + input.eta_time_offset_exit[0];
    let expected_eta = h - eta_t * inv_sigma;
    let expected_survival = inverse_link_survival_prob_checked(&input.inverse_link, expected_eta)
        .expect("expected survival");

    assert!((pred.eta[0] - expected_eta).abs() <= 1e-12);
    assert!((pred.survival_prob[0] - expected_survival).abs() <= 1e-12);
}

#[test]
fn component_prediction_matches_full_design_for_repeated_prediction_grid() {
    let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
    let inverse_link = residual_distribution_inverse_link(ResidualDistribution::Gaussian);
    let x_time_exit = array![[1.0, 0.2], [1.0, 0.8], [0.5, -0.3], [0.5, 0.4]];
    let x_threshold = array![[1.0, -0.2], [1.0, -0.2], [0.0, 0.6], [0.0, 0.6]];
    let x_log_sigma = array![[1.0, 0.3], [1.0, 0.3], [0.0, -0.4], [0.0, -0.4]];
    let eta_time_offset_exit = array![0.2, 0.25, -0.1, -0.05];
    let eta_threshold_offset = array![0.7, 0.7, -0.2, -0.2];
    let eta_log_sigma_offset = array![0.4, 0.4, -0.3, -0.3];
    let full_input = SurvivalLocationScalePredictInput {
        x_time_exit: x_time_exit.clone(),
        eta_time_offset_exit: eta_time_offset_exit.clone(),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::from(x_threshold.clone()),
        eta_threshold_offset: eta_threshold_offset.clone(),
        x_log_sigma: DesignMatrix::from(x_log_sigma.clone()),
        eta_log_sigma_offset: eta_log_sigma_offset.clone(),
        x_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        inverse_link: inverse_link.clone(),
    };
    let full = predict_survival_location_scale(&full_input, &fit).expect("full predict");
    let eta_t = x_threshold.dot(&fit.beta_threshold()) + eta_threshold_offset;
    let eta_ls = x_log_sigma.dot(&fit.beta_log_sigma()) + eta_log_sigma_offset;
    let component = predict_survival_location_scale_from_linear_components(
        &x_time_exit,
        &eta_time_offset_exit,
        None,
        None,
        0,
        &eta_t,
        &eta_ls,
        None,
        None,
        &inverse_link,
        &fit,
    )
    .expect("component predict");

    for i in 0..full.eta.len() {
        assert!((full.eta[i] - component.eta[i]).abs() <= 1e-12);
        assert!((full.survival_prob[i] - component.survival_prob[i]).abs() <= 1e-12);
    }
}

#[test]
fn sparse_prediction_and_uncertainty_match_dense() {
    let fit = test_survival_fit(
        array![0.4, -0.1],
        array![0.2, 0.3],
        array![-0.5, 0.1],
        Some(array![0.05, -0.02]),
    );
    let x_threshold_dense = array![[1.0, -0.2], [0.0, 0.6]];
    let x_log_sigma_dense = array![[1.0, 0.3], [0.0, -0.4]];
    let eta_t = x_threshold_dense.dot(&fit.beta_threshold()) + Array1::from_vec(vec![0.7, -0.2]);
    let eta_ls = x_log_sigma_dense.dot(&fit.beta_log_sigma()) + Array1::from_vec(vec![0.4, 0.1]);
    let q0 = Array1::from_iter(
        eta_t
            .iter()
            .zip(eta_ls.iter())
            .map(|(&t, &ls)| -t * exp_sigma_inverse_from_eta_scalar(ls)),
    );
    let link_wiggle_degree = fit
        .artifacts
        .survival_link_wiggle_degree
        .expect("fit wiggle degree");
    let link_wiggle_knots = fit
        .artifacts
        .survival_link_wiggle_knots
        .clone()
        .expect("fit wiggle knots");
    let xwiggle_dense = survival_wiggle_basis_with_options(
        q0.view(),
        &link_wiggle_knots,
        link_wiggle_degree,
        BasisOptions::value(),
    )
    .expect("link wiggle design");
    let dense_input = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5], [1.0, -0.3]],
        eta_time_offset_exit: array![0.2, -0.1],
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_threshold_dense.clone(),
        )),
        eta_threshold_offset: array![0.7, -0.2],
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_log_sigma_dense.clone(),
        )),
        eta_log_sigma_offset: array![0.4, 0.1],
        x_link_wiggle: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            xwiggle_dense.clone(),
        ))),
        link_wiggle_knots: Some(link_wiggle_knots.clone()),
        link_wiggle_degree: Some(link_wiggle_degree),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
    };
    let sparse_input = SurvivalLocationScalePredictInput {
        x_threshold: sparse_design_from_dense(&x_threshold_dense),
        x_log_sigma: sparse_design_from_dense(&x_log_sigma_dense),
        x_link_wiggle: Some(sparse_design_from_dense(&xwiggle_dense)),
        ..dense_input.clone()
    };
    let covariance = array![
        [0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
        [0.01, 0.02, 0.0, 0.0, 0.0, 0.0, -0.005, 0.0],
        [0.0, 0.0, 0.04, 0.01, 0.0, 0.0, 0.006, 0.001],
        [0.0, 0.0, 0.01, 0.03, 0.0, 0.0, -0.004, 0.002],
        [0.0, 0.0, 0.0, 0.0, 0.02, 0.005, 0.003, 0.001],
        [0.0, 0.0, 0.0, 0.0, 0.005, 0.02, -0.002, 0.004],
        [0.01, -0.005, 0.006, -0.004, 0.003, -0.002, 0.025, 0.006],
        [0.0, 0.0, 0.001, 0.002, 0.001, 0.004, 0.006, 0.018],
    ];

    let dense_pred = predict_survival_location_scale(&dense_input, &fit).expect("dense predict");
    let sparse_pred = predict_survival_location_scale(&sparse_input, &fit).expect("sparse predict");
    assert_eq!(dense_pred.eta.len(), sparse_pred.eta.len());
    for i in 0..dense_pred.eta.len() {
        assert!((dense_pred.eta[i] - sparse_pred.eta[i]).abs() <= 1e-12);
        assert!((dense_pred.survival_prob[i] - sparse_pred.survival_prob[i]).abs() <= 1e-12);
    }

    let dense_unc = predict_survival_location_scalewith_uncertainty(
        &dense_input,
        &fit,
        &covariance,
        false,
        true,
    )
    .expect("dense uncertainty");
    let sparse_unc = predict_survival_location_scalewith_uncertainty(
        &sparse_input,
        &fit,
        &covariance,
        false,
        true,
    )
    .expect("sparse uncertainty");
    for i in 0..dense_unc.eta.len() {
        assert!((dense_unc.eta[i] - sparse_unc.eta[i]).abs() <= 1e-12);
        assert!((dense_unc.survival_prob[i] - sparse_unc.survival_prob[i]).abs() <= 1e-12);
        assert!(
            (dense_unc.eta_standard_error[i] - sparse_unc.eta_standard_error[i]).abs() <= 1e-12
        );
        let dense_sd = dense_unc
            .response_standard_error
            .as_ref()
            .expect("dense response sd")[i];
        let sparse_sd = sparse_unc
            .response_standard_error
            .as_ref()
            .expect("sparse response sd")[i];
        assert!((dense_sd - sparse_sd).abs() <= 1e-12);
    }

    let dense_pm = predict_survival_location_scalewith_uncertainty(
        &dense_input,
        &fit,
        &covariance,
        true,
        false,
    )
    .expect("dense wiggle posterior mean");
    let sparse_pm = predict_survival_location_scalewith_uncertainty(
        &sparse_input,
        &fit,
        &covariance,
        true,
        false,
    )
    .expect("sparse wiggle posterior mean");
    for i in 0..dense_pm.eta.len() {
        assert!((dense_pm.eta[i] - sparse_pm.eta[i]).abs() <= 1e-12);
        assert!((dense_pm.survival_prob[i] - sparse_pm.survival_prob[i]).abs() <= 1e-10);
    }
}

#[test]
fn gaussian_posterior_mean_matches_3d_ghq_small_case() {
    let input = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5]],
        eta_time_offset_exit: array![0.1],
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.25
        ]])),
        eta_threshold_offset: array![0.0],
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, -0.15
        ]])),
        eta_log_sigma_offset: array![0.0],
        x_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
    };
    let fit = test_survival_fit(
        array![0.3, -0.2],
        array![0.1, 0.2],
        array![-0.4, 0.15],
        None,
    );
    let covariance = array![
        [0.03, 0.01, 0.0, 0.0, 0.0, 0.0],
        [0.01, 0.02, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.04, 0.01, 0.0, 0.0],
        [0.0, 0.0, 0.01, 0.03, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.02, 0.005],
        [0.0, 0.0, 0.0, 0.0, 0.005, 0.02],
    ];
    let predicted =
        predict_survival_location_scalewith_uncertainty(&input, &fit, &covariance, true, false)
            .expect("posterior mean");

    let mu_h = input.x_time_exit.row(0).dot(&fit.beta_time()) + input.eta_time_offset_exit[0];
    let x_t = input.x_threshold.to_dense_arc();
    let x_ls = input.x_log_sigma.to_dense_arc();
    let mu_t = x_t.row(0).dot(&fit.beta_threshold());
    let mu_ls = x_ls.row(0).dot(&fit.beta_log_sigma());
    let cov_hh = covariance.slice(s![0..2, 0..2]).to_owned();
    let cov_tt = covariance.slice(s![2..4, 2..4]).to_owned();
    let cov_ll = covariance.slice(s![4..6, 4..6]).to_owned();
    let cov_ht = covariance.slice(s![0..2, 2..4]).to_owned();
    let cov_hl = covariance.slice(s![0..2, 4..6]).to_owned();
    let cov_tl = covariance.slice(s![2..4, 4..6]).to_owned();
    let var_h = input
        .x_time_exit
        .row(0)
        .dot(&cov_hh.dot(&input.x_time_exit.row(0).to_owned()));
    let var_t = x_t.row(0).dot(&cov_tt.dot(&x_t.row(0).to_owned()));
    let var_ls = x_ls.row(0).dot(&cov_ll.dot(&x_ls.row(0).to_owned()));
    let cov_ht_i = input
        .x_time_exit
        .row(0)
        .dot(&cov_ht.dot(&x_t.row(0).to_owned()));
    let cov_hl_i = input
        .x_time_exit
        .row(0)
        .dot(&cov_hl.dot(&x_ls.row(0).to_owned()));
    let cov_tl_i = x_t.row(0).dot(&cov_tl.dot(&x_ls.row(0).to_owned()));
    let quadctx = crate::quadrature::QuadratureContext::new();
    let ghq = crate::quadrature::normal_expectation_3d_adaptive(
        &quadctx,
        [mu_h, mu_t, mu_ls],
        [
            [var_h, cov_ht_i, cov_hl_i],
            [cov_ht_i, var_t, cov_tl_i],
            [cov_hl_i, cov_tl_i, var_ls],
        ],
        |h, t, ls| {
            inverse_link_survival_probvalue(
                &input.inverse_link,
                h - t * exp_sigma_inverse_from_eta_scalar(ls),
            )
        },
    );
    assert!((predicted.survival_prob[0] - ghq).abs() <= 2e-4);
}

#[test]
fn sparse_posterior_mean_matches_dense() {
    let x_threshold_dense = array![[1.0, 0.25], [0.0, -0.1]];
    let x_log_sigma_dense = array![[1.0, -0.15], [0.0, 0.2]];
    let dense_input = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5], [1.0, -0.4]],
        eta_time_offset_exit: array![0.1, -0.2],
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_threshold_dense.clone(),
        )),
        eta_threshold_offset: array![0.0, 0.05],
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_log_sigma_dense.clone(),
        )),
        eta_log_sigma_offset: array![0.0, -0.03],
        x_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
    };
    let sparse_input = SurvivalLocationScalePredictInput {
        x_threshold: sparse_design_from_dense(&x_threshold_dense),
        x_log_sigma: sparse_design_from_dense(&x_log_sigma_dense),
        ..dense_input.clone()
    };
    let fit = test_survival_fit(
        array![0.3, -0.2],
        array![0.1, 0.2],
        array![-0.4, 0.15],
        None,
    );
    let covariance = array![
        [0.03, 0.01, 0.0, 0.0, 0.0, 0.0],
        [0.01, 0.02, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.04, 0.01, 0.0, 0.0],
        [0.0, 0.0, 0.01, 0.03, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.02, 0.005],
        [0.0, 0.0, 0.0, 0.0, 0.005, 0.02],
    ];

    let dense_pm = predict_survival_location_scalewith_uncertainty(
        &dense_input,
        &fit,
        &covariance,
        true,
        false,
    )
    .expect("dense posterior mean");
    let sparse_pm = predict_survival_location_scalewith_uncertainty(
        &sparse_input,
        &fit,
        &covariance,
        true,
        false,
    )
    .expect("sparse posterior mean");
    for i in 0..dense_pm.eta.len() {
        assert!((dense_pm.eta[i] - sparse_pm.eta[i]).abs() <= 1e-12);
        assert!((dense_pm.survival_prob[i] - sparse_pm.survival_prob[i]).abs() <= 1e-10);
    }
}

#[test]
fn wiggle_posterior_mean_matches_exact_nested_4d_quadrature_small_case() {
    let fit = test_survival_fit(
        array![0.4, -0.1],
        array![0.2, 0.3],
        array![-0.5, 0.1],
        Some(array![0.05, -0.02]),
    );
    let x_threshold_dense = array![[1.0, -0.2]];
    let x_log_sigma_dense = array![[1.0, 0.3]];
    let eta_t = x_threshold_dense.dot(&fit.beta_threshold());
    let eta_ls = x_log_sigma_dense.dot(&fit.beta_log_sigma());
    let q0 = Array1::from_iter(
        eta_t
            .iter()
            .zip(eta_ls.iter())
            .map(|(&t, &ls)| -t * exp_sigma_inverse_from_eta_scalar(ls)),
    );
    let link_wiggle_degree = fit
        .artifacts
        .survival_link_wiggle_degree
        .expect("fit wiggle degree");
    let link_wiggle_knots = fit
        .artifacts
        .survival_link_wiggle_knots
        .clone()
        .expect("fit wiggle knots");
    let x_link_wiggle = survival_wiggle_basis_with_options(
        q0.view(),
        &link_wiggle_knots,
        link_wiggle_degree,
        BasisOptions::value(),
    )
    .expect("link wiggle design");
    let input = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5]],
        eta_time_offset_exit: array![0.2],
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_threshold_dense)),
        eta_threshold_offset: array![0.0],
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_log_sigma_dense)),
        eta_log_sigma_offset: array![0.0],
        x_link_wiggle: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_link_wiggle,
        ))),
        link_wiggle_knots: Some(link_wiggle_knots),
        link_wiggle_degree: Some(link_wiggle_degree),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
    };
    let covariance = array![
        [0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
        [0.01, 0.02, 0.0, 0.0, 0.0, 0.0, -0.005, 0.0],
        [0.0, 0.0, 0.04, 0.01, 0.0, 0.0, 0.006, 0.001],
        [0.0, 0.0, 0.01, 0.03, 0.0, 0.0, -0.004, 0.002],
        [0.0, 0.0, 0.0, 0.0, 0.02, 0.005, 0.003, 0.001],
        [0.0, 0.0, 0.0, 0.0, 0.005, 0.02, -0.002, 0.004],
        [0.01, -0.005, 0.006, -0.004, 0.003, -0.002, 0.025, 0.006],
        [0.0, 0.0, 0.001, 0.002, 0.001, 0.004, 0.006, 0.018],
    ];
    let predicted =
        predict_survival_location_scalewith_uncertainty(&input, &fit, &covariance, true, false)
            .expect("wiggle posterior mean");

    let x_t = input.x_threshold.to_dense_arc();
    let x_ls = input.x_log_sigma.to_dense_arc();
    let mu_h = input.x_time_exit.row(0).dot(&fit.beta_time()) + input.eta_time_offset_exit[0];
    let mu_t = x_t.row(0).dot(&fit.beta_threshold()) + input.eta_threshold_offset[0];
    let mu_ls = x_ls.row(0).dot(&fit.beta_log_sigma()) + input.eta_log_sigma_offset[0];
    let cov_hh = covariance.slice(s![0..2, 0..2]).to_owned();
    let cov_tt = covariance.slice(s![2..4, 2..4]).to_owned();
    let cov_ll = covariance.slice(s![4..6, 4..6]).to_owned();
    let cov_ht = covariance.slice(s![0..2, 2..4]).to_owned();
    let cov_hl = covariance.slice(s![0..2, 4..6]).to_owned();
    let cov_hw = covariance.slice(s![0..2, 6..8]).to_owned();
    let cov_tl = covariance.slice(s![2..4, 4..6]).to_owned();
    let cov_tw = covariance.slice(s![2..4, 6..8]).to_owned();
    let cov_lw = covariance.slice(s![4..6, 6..8]).to_owned();
    let var_h = input
        .x_time_exit
        .row(0)
        .dot(&cov_hh.dot(&input.x_time_exit.row(0).to_owned()));
    let var_t = x_t.row(0).dot(&cov_tt.dot(&x_t.row(0).to_owned()));
    let var_ls = x_ls.row(0).dot(&cov_ll.dot(&x_ls.row(0).to_owned()));
    let cov_ht_i = input
        .x_time_exit
        .row(0)
        .dot(&cov_ht.dot(&x_t.row(0).to_owned()));
    let cov_hl_i = input
        .x_time_exit
        .row(0)
        .dot(&cov_hl.dot(&x_ls.row(0).to_owned()));
    let cov_tl_i = x_t.row(0).dot(&cov_tl.dot(&x_ls.row(0).to_owned()));
    let quadctx = crate::quadrature::QuadratureContext::new();
    let cov_htl = [
        [var_h, cov_ht_i, cov_hl_i],
        [cov_ht_i, var_t, cov_tl_i],
        [cov_hl_i, cov_tl_i, var_ls],
    ];
    let htl_factor = factorize_psd_covariance(
        &covariance3_to_array2(cov_htl),
        "wiggle posterior mean test projected covariance",
    )
    .expect("factor projected covariance");
    let cov_wy = {
        let mut out = Array2::<f64>::zeros((2, 3));
        out.column_mut(0)
            .assign(&cov_hw.t().dot(&input.x_time_exit.row(0).to_owned()));
        out.column_mut(1)
            .assign(&cov_tw.t().dot(&x_t.row(0).to_owned()));
        out.column_mut(2)
            .assign(&cov_lw.t().dot(&x_ls.row(0).to_owned()));
        out
    };
    let cov_ww = covariance.slice(s![6..8, 6..8]).to_owned();
    let mut regression = cov_wy.dot(&htl_factor.eigenvectors);
    for col in 0..regression.ncols() {
        let scale = htl_factor.inv_sqrt_eigenvalues[col];
        regression
            .column_mut(col)
            .mapv_inplace(|value| value * scale);
    }
    let cov_cond =
        symmetrize_and_clip_covariance(&(cov_ww - regression.dot(&regression.t().to_owned())));
    let ghq = low_rank_normal_expectation_pair_3d_result(
        &quadctx,
        [mu_h, mu_t, mu_ls],
        cov_htl,
        15,
        "wiggle posterior mean test projected covariance",
        |x, z| {
            let mut cond_beta_w = fit.beta_link_wiggle().expect("wiggle beta");
            for j in 0..cond_beta_w.len() {
                for (col, &latent) in z.iter().enumerate() {
                    cond_beta_w[j] += regression[[j, col]] * latent;
                }
            }
            let q0 = survival_q0_from_eta(x[1], x[2]);
            let q0_arr = Array1::from_vec(vec![q0]);
            let basis = survival_wiggle_basis_with_options(
                q0_arr.view(),
                input.link_wiggle_knots.as_ref().expect("wiggle knots"),
                input.link_wiggle_degree.expect("wiggle degree"),
                BasisOptions::value(),
            )?;
            let b = basis.row(0).to_owned();
            let w_mean = b.dot(&cond_beta_w);
            let w_var = b.dot(&cov_cond.dot(&b)).max(0.0);
            crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
                &quadctx,
                [x[0] + q0 + w_mean],
                [[w_var]],
                21,
                |eta| {
                    let p = inverse_link_survival_prob_checked(&input.inverse_link, eta[0])?;
                    Ok((p, p * p))
                },
            )
        },
    )
    .expect("exact conditional wiggle ghq");
    assert!((predicted.survival_prob[0] - ghq.0).abs() <= 2e-4);
}

#[test]
fn predict_supports_sas_beta_logistic_and_mixture_links() {
    let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
    let base = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5]],
        eta_time_offset_exit: array![0.2],
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, -0.2
        ]])),
        eta_threshold_offset: array![0.0],
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.3
        ]])),
        eta_log_sigma_offset: array![0.0],
        x_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        inverse_link: InverseLink::Standard(StandardLink::Probit),
    };

    let sas = InverseLink::Sas(
        state_from_sasspec(SasLinkSpec {
            initial_epsilon: 0.1,
            initial_log_delta: -0.2,
        })
        .expect("sas state"),
    );
    let beta_logistic = InverseLink::BetaLogistic(
        state_from_beta_logisticspec(SasLinkSpec {
            initial_epsilon: 0.05,
            initial_log_delta: 0.1,
        })
        .expect("beta-logistic state"),
    );
    let mixture = InverseLink::Mixture(
        state_fromspec(&MixtureLinkSpec {
            components: vec![LinkComponent::Probit, LinkComponent::Logit],
            initial_rho: array![0.2],
        })
        .expect("mixture state"),
    );

    for link in [sas, beta_logistic, mixture] {
        let mut input = base.clone();
        input.inverse_link = link;
        let pred = predict_survival_location_scale(&input, &fit).expect("predict");
        assert!(pred.survival_prob[0].is_finite());
        assert!(pred.survival_prob[0] > 0.0 && pred.survival_prob[0] < 1.0);
        let cov = Array2::eye(6) * 1e-3;
        let pm = predict_survival_location_scalewith_uncertainty(&input, &fit, &cov, true, false)
            .expect("posterior mean");
        assert!(pm.survival_prob[0].is_finite());
        assert!(pm.survival_prob[0] > 0.0 && pm.survival_prob[0] < 1.0);
    }
}

/// Full-path structural monotonicity regression for the
/// heart_failure_survival workflow setup.
#[test]
fn heart_failure_full_fit_structural_time_coefficients() {
    // 20 rows with realistic-ish I-spline-like structure.
    let n = 20;
    let p_time = 8; // 8 time basis columns

    // Entry times all near zero (left-truncation at 0) — like __entry=0.
    let age_entry = Array1::from_elem(n, 1e-9_f64);
    // Exit times spread out like real survival data.
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        age_exit[i] = 4.0 + (i as f64) * 14.0; // 4 to 270
    }

    // Events: ~1/3 event rate.
    let mut event_target = Array1::<f64>::zeros(n);
    for i in [0, 3, 5, 8, 12, 17] {
        event_target[i] = 1.0;
    }
    let weights = Array1::ones(n);

    // Build I-spline-like time designs.
    // Entry design is all zeros (I-spline = 0 below knot range).
    let design_entry = Array2::<f64>::zeros((n, p_time));

    // Exit design: monotonically increasing I-spline-like columns.
    let mut design_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64); // 0 to 1
        for j in 0..p_time {
            let center = (j as f64 + 0.5) / (p_time as f64);
            // Smooth sigmoid-like I-spline approximation.
            let x = 8.0 * (t - center);
            design_exit[[i, j]] = 1.0 / (1.0 + (-x).exp());
        }
    }

    // Derivative design: derivative of I-spline columns.
    let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        for j in 0..p_time {
            let center = (j as f64 + 0.5) / (p_time as f64);
            let x = 8.0 * (t - center);
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            // Derivative of sigmoid * chain_rule (1/t).
            let deriv = 8.0 * sigmoid * (1.0 - sigmoid);
            let chain = 1.0 / age_exit[i];
            design_derivative_exit[[i, j]] = deriv * chain;
        }
    }

    // The workflow carries the derivative floor in the offsets, so the
    // structural time coefficients only need to stay non-negative.
    let derivative_offset_exit =
        Array1::from_elem(n, DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);
    let offset_entry = Array1::<f64>::zeros(n);
    let offset_exit = Array1::<f64>::zeros(n);

    // Simple difference penalty.
    let mut penalty = Array2::<f64>::zeros((p_time, p_time));
    for i in 0..(p_time - 1) {
        penalty[[i, i]] += 1.0;
        penalty[[i, i + 1]] -= 1.0;
        penalty[[i + 1, i]] -= 1.0;
        penalty[[i + 1, i + 1]] += 1.0;
    }

    let spec = SurvivalLocationScaleSpec {
        age_entry,
        age_exit,
        event_target,
        weights,
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        max_iter: 400,
        tol: 1e-6,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry),
            design_exit: DesignMatrix::from(design_exit),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry,
            offset_exit,
            derivative_offset_exit: derivative_offset_exit.clone(),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![penalty.clone()],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(array![0.0]),
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones((
                n, 1,
            )))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones((
                n, 1,
            )))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        cache_mirror_sessions: Vec::new(),
    };

    match fit_survival_location_scale_with_geometry(spec).map(|(fit, _)| fit) {
        Ok(result) => {
            // Structural-monotonicity invariant implied by the test's
            // name: the I-spline-like time block carries structural
            // lower bounds of zero (see
            // `structural_time_coefficient_lower_bounds`), and
            // the constrained solve/max-step limiter represents that
            // cone directly. Every accepted coefficient must therefore
            // satisfy β ≥ 0 — the precondition for the monotone
            // I-spline reconstruction the workflow consumes downstream.
            assert!(
                result.beta_time().iter().all(|&b| b.is_finite()),
                "structural time coefficients must be finite: {:?}",
                result.beta_time(),
            );
            assert!(
                result.beta_time().iter().all(|&b| b >= 0.0),
                "structural time coefficients must be non-negative after constrained solve: {:?}",
                result.beta_time(),
            );
            // Parallel invariant for BLOCK_LINK_WIGGLE: monotone-link
            // wiggle coefficients are structurally non-negative. This
            // test configures `linkwiggle_block: None`, so the block is
            // absent — but if it is ever enabled here the represented
            // block constraint must enforce the same invariant.
            if let Some(beta_link_wiggle) = result.beta_link_wiggle() {
                assert!(
                    beta_link_wiggle.iter().all(|&b| b.is_finite()),
                    "link-wiggle coefficients must be finite: {beta_link_wiggle:?}",
                );
                assert!(
                    beta_link_wiggle.iter().all(|&b| b >= 0.0),
                    "link-wiggle coefficients must be non-negative after constrained solve: {beta_link_wiggle:?}",
                );
            }
        }
        Err(e) => {
            panic!("fit_survival_location_scale failed: {e}");
        }
    }
}

/// Small structural-monotonicity regression for the
/// heart_failure_survival workflow setup.
#[test]
fn heart_failure_structural_time_small() {
    // 6 rows: 3 events, 3 non-events.  Single time column for simplicity.
    let n = 6;
    // I-spline-like designs: entry is all zero (left truncation at t=0),
    // exit has non-trivial values, derivative is the B-spline derivative.
    let x_entry = Array2::<f64>::zeros((n, 2));
    let x_exit = array![
        [0.1, 0.05],
        [0.3, 0.15],
        [0.5, 0.35],
        [0.7, 0.55],
        [0.9, 0.80],
        [1.0, 0.95],
    ];
    let x_deriv = array![
        [0.2, 0.1],
        [0.3, 0.2],
        [0.3, 0.3],
        [0.3, 0.3],
        [0.2, 0.3],
        [0.1, 0.2],
    ];
    let offset_deriv = Array1::from_elem(n, DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);

    let family = SurvivalLocationScaleFamily {
        n,
        y: array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        w: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        x_time_entry: Arc::new(x_entry),
        x_time_exit: Arc::new(x_exit.clone()),
        x_time_deriv: Arc::new(x_deriv.clone()),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0, 0.0]),
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones((
            n, 1,
        )))),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones((
            n, 1,
        )))),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: crate::resource::ResourcePolicy::default_library(),
    };

    // Build initial states with beta=0 and a feasible positive derivative offset.
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(2),
            eta: {
                let mut eta = Array1::<f64>::zeros(3 * n);
                eta.slice_mut(ndarray::s![2 * n..3 * n])
                    .fill(DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);
                eta
            },
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(n),
        },
    ];

    // Step 1: Verify initial evaluate succeeds on the feasible domain.
    let eval = family
        .evaluate(&states)
        .expect("initial evaluate with positive d_eta/dt should succeed");

    // Step 2: Extract time block gradient and Hessian.
    let (grad, hess) = match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { gradient, hessian } => {
            (gradient.clone(), hessian.to_dense())
        }
        _ => panic!("expected exact-newton for time block"),
    };

    // Step 3: Simulate Newton step (H + ridge*I) * delta = grad - S*beta.
    // With beta=0 and no penalty: (H + ridge*I) * delta = grad.
    let ridge = 1e-6_f64;
    let p = 2;
    let mut lhs = hess.clone();
    for i in 0..p {
        lhs[[i, i]] += ridge;
    }
    // Solve via direct inversion (2x2).
    let det = lhs[[0, 0]] * lhs[[1, 1]] - lhs[[0, 1]] * lhs[[1, 0]];
    let delta = if det.abs() > 1e-30 {
        let inv00 = lhs[[1, 1]] / det;
        let inv01 = -lhs[[0, 1]] / det;
        let inv10 = -lhs[[1, 0]] / det;
        let inv11 = lhs[[0, 0]] / det;
        array![
            inv00 * grad[0] + inv01 * grad[1],
            inv10 * grad[0] + inv11 * grad[1]
        ]
    } else {
        Array1::zeros(p)
    };
    assert!(
        delta.iter().all(|v| v.is_finite()),
        "Newton delta has non-finite entries: {:?}",
        delta
    );

    // Step 4: Compute new d_raw after the step.
    let new_d_raw = x_deriv.dot(&delta) + &offset_deriv;
    for (i, &v) in new_d_raw.iter().enumerate() {
        assert!(
            v.is_finite(),
            "d_raw[{i}] is non-finite ({v}) after Newton step with delta={:?}",
            delta
        );
    }

    // Step 5: Verify evaluate succeeds with the new state.
    let new_eta_time = {
        let mut eta = Array1::<f64>::zeros(3 * n);
        // h0 = x_entry * delta (all zero since x_entry is zero)
        // h1 = x_exit * delta
        let h1 = x_exit.dot(&delta);
        eta.slice_mut(ndarray::s![n..2 * n]).assign(&h1);
        // d_raw = x_deriv * delta + offset_deriv
        eta.slice_mut(ndarray::s![2 * n..3 * n]).assign(&new_d_raw);
        eta
    };
    let new_states = vec![
        ParameterBlockState {
            beta: delta.clone(),
            eta: new_eta_time,
        },
        states[1].clone(),
        states[2].clone(),
    ];
    family
        .evaluate(&new_states)
        .unwrap_or_else(|e| panic!("evaluate failed after Newton step: {e}"));
}

#[test]
fn evaluate_survival_location_scale_rejects_non_finite_d_eta_dt() {
    let n = 2;
    let family = SurvivalLocationScaleFamily {
        n,
        y: array![1.0, 0.0],
        w: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        x_time_entry: Arc::new(Array2::zeros((n, 1))),
        x_time_exit: Arc::new(Array2::ones((n, 1))),
        x_time_deriv: Arc::new(Array2::ones((n, 1))),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones((
            n, 1,
        )))),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones((
            n, 1,
        )))),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: crate::resource::ResourcePolicy::default_library(),
    };

    let mut eta_time = Array1::<f64>::zeros(3 * n);
    eta_time[2 * n] = f64::NAN;
    eta_time[2 * n + 1] = 0.25;
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_time,
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(n),
        },
    ];

    let eval = match family.evaluate(&states) {
        Ok(_) => panic!("non-finite d_eta/dt must be rejected"),
        Err(err) => err,
    };
    assert!(eval.contains("non-finite"));
}

#[test]
fn q_chain_derivatives_match_exact_exp_link_in_lower_tail() {
    let eta_t = 2.0;
    let eta_ls = -30.0;
    let q = |ls: f64| -eta_t * exp_sigma_inverse_from_eta_scalar(ls);
    let h = 1e-6;
    let q_left = q(eta_ls - h);
    let q_mid = q(eta_ls);
    let q_right = q(eta_ls + h);
    assert!(
        q_left != q_mid && q_right != q_mid,
        "exact exp-link q should remain eta_ls-sensitive in the lower tail"
    );

    let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) = q_chain_derivs_scalar(eta_t, eta_ls);
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    assert!((q_t + inv_sigma).abs() <= 1e-15);
    assert!((q_ls - eta_t * inv_sigma).abs() <= 1e-15);
    assert!((q_tl - inv_sigma).abs() <= 1e-15);
    assert!((q_ll + eta_t * inv_sigma).abs() <= 1e-15);
    assert!((q_tl_ls + inv_sigma).abs() <= 1e-15);
    assert!((q_ll_ls - eta_t * inv_sigma).abs() <= 1e-15);
}

#[test]
fn survival_q0dot_from_base_preserves_far_tail_cancellation() {
    let eta_t = 1e-10;
    let eta_ls = -700.0;
    let eta_t_deriv = 1.0 - 1e-12;
    let eta_ls_deriv = 1e10;
    let base = survival_base_q_scalars(eta_t, eta_ls);

    let factorized = survival_q0dot_from_base(base, eta_t_deriv, eta_ls_deriv);
    let expected = safe_product(
        exp_sigma_inverse_from_eta_scalar(eta_ls),
        eta_t.mul_add(eta_ls_deriv, -eta_t_deriv),
    );
    let expanded = safe_sum2(
        safe_product(base.q_t, eta_t_deriv),
        safe_product(base.q_ls, eta_ls_deriv),
    );

    assert!(factorized.is_finite());
    assert!(expected.is_finite());
    assert!(
        (factorized - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "factorized qdot mismatch: got {factorized}, expected {expected}"
    );
    assert!(expanded.abs() >= 1e200);
    assert!(factorized.abs() <= 1e206);
}

#[test]
fn compensated_difference_carries_explicit_roundoff_bound() {
    let lhs = 1.0e217 + 1.0e201;
    let rhs = 1.0e217;
    let diff = compensated_difference(lhs, rhs);

    assert!(diff.value.is_finite());
    assert!(diff.roundoff_slack.is_finite());
    assert!(diff.roundoff_slack >= 0.0);
    assert!(diff.operand_scale >= rhs.abs());
}

#[test]
fn logistic_residual_tail_derivatives_should_match_stable_closed_forms() {
    let z = 50.0_f64;
    let e = (-z).exp();
    let denom = 1.0_f64 + e;
    let stable_pdf = e / denom.powi(2);
    let stable_d1 = e * (e - 1.0) / denom.powi(3);
    let stable_d2 = e * (e * e - 4.0 * e + 1.0) / denom.powi(4);
    let stable_d3 = e * (e * e * e - 11.0 * e * e + 11.0 * e - 1.0) / denom.powi(5);

    let dist = ResidualDistribution::Logistic;
    assert!(
        (dist.pdf(z) - stable_pdf).abs() < 1e-30,
        "logistic residual pdf should equal the stable tail formula at z={z}; got {} vs {}",
        dist.pdf(z),
        stable_pdf
    );
    assert!(
        (dist.pdf_derivative(z) - stable_d1).abs() < 1e-30,
        "logistic residual pdf' should equal the stable tail formula at z={z}; got {} vs {}",
        dist.pdf_derivative(z),
        stable_d1
    );
    assert!(
        (dist.pdfsecond_derivative(z) - stable_d2).abs() < 1e-30,
        "logistic residual pdf'' should equal the stable tail formula at z={z}; got {} vs {}",
        dist.pdfsecond_derivative(z),
        stable_d2
    );
    assert!(
        (dist.pdfthird_derivative(z) - stable_d3).abs() < 1e-30,
        "logistic residual pdf''' should equal the stable tail formula at z={z}; got {} vs {}",
        dist.pdfthird_derivative(z),
        stable_d3
    );
}

#[test]
fn gumbel_cdf_negative_tail_should_match_expm1_form() {
    let z = -50.0_f64;
    let ez = z.exp();
    let stable_cdf = -(-ez).exp_m1();
    let dist = ResidualDistribution::Gumbel;
    assert!(stable_cdf > 0.0);
    assert!(
        (dist.cdf(z) - stable_cdf).abs() < 1e-30,
        "gumbel cdf should equal -expm1(-exp(z)) in the negative tail at z={z}; got {} vs {}",
        dist.cdf(z),
        stable_cdf
    );
}

#[test]
fn probit_survival_helper_matches_upper_tail_probability() {
    let eta = 10.0_f64;
    let stable_survival = 0.5 * statrs::function::erf::erfc(eta / std::f64::consts::SQRT_2);
    assert!(stable_survival > 0.0);
    let helper = inverse_link_survival_probvalue(&InverseLink::Standard(StandardLink::Probit), eta);
    assert!(
        (helper - stable_survival).abs() < 1e-30,
        "probit survival helper should use the upper-tail erfc form at eta={eta}; got {} vs {}",
        helper,
        stable_survival
    );
}

#[test]
fn cloglog_survival_helper_matches_negative_tail_function() {
    let eta = -100.0_f64;
    let stable_survival = (-(eta.exp())).exp();
    let helper =
        inverse_link_survival_probvalue(&InverseLink::Standard(StandardLink::CLogLog), eta);
    assert_eq!(stable_survival, 1.0);
    assert!(
        (helper - stable_survival).abs() < 1e-30,
        "cloglog survival helper should evaluate exp(-exp(eta)) itself, not a clamped surrogate, at eta={eta}; got {} vs {}",
        helper,
        stable_survival
    );
}

#[test]
fn positive_log_cumulative_hazard_maps_to_baseline_cloglog_survival() {
    let cumulative_hazard = 4.0_f64;
    let eta = cumulative_hazard.ln();
    let survival =
        inverse_link_survival_probvalue(&InverseLink::Standard(StandardLink::CLogLog), eta);
    let expected = (-cumulative_hazard).exp();
    assert!(
        (survival - expected).abs() < 1e-15,
        "baseline cloglog survival should be exp(-H0) when eta = log(H0); got {} vs {}",
        survival,
        expected
    );
}
