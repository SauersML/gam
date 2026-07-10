use super::*;
use crate::custom_family::BlockWorkingSet;
use faer::sparse::{SparseColMat, Triplet};
use gam_problem::{LinkComponent, MixtureLinkSpec, SasLinkSpec};
use gam_solve::gauge::Gauge;
use gam_solve::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
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
            0.0,
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
    ) -> gam_math::jet_tower::Tower4<2> {
        use gam_math::jet_tower::Tower4;

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

impl gam_math::jet_tower::RowNllProgram<2> for SurvivalLsLocationScaleNllProgram<'_> {
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
        p: &[gam_math::jet_tower::Tower4<2>; 2],
    ) -> Result<gam_math::jet_tower::Tower4<2>, String> {
        use gam_math::jet_tower::Tower4;

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
    use gam_terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use gam_terms::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};

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
    let shared = crate::spatial_psi_bridge::build_block_spatial_psi_derivatives(
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
                crate::wiggle::buildwiggle_block_input_from_seed(seed.view(), &cfg)
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
        penalty_block_trace: Vec::new(),
        edf_by_block: Vec::new(),
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [0.4],
            [-0.6]
        ])),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
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
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    }
}

fn survival_exact_newton_test_states(
    family: &SurvivalLocationScaleFamily,
    beta_t: f64,
    beta_thr: f64,
    beta_ls: f64,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    // Stacked time eta layout is `[entry; exit; deriv]` (gam#1396): the entry
    // channel occupies `0..n`, the exit channel `n..2n`, matching the solver
    // design's `MultiChannelOperator` stacking and `validate_joint_states`.
    let mut eta_time = Array1::<f64>::zeros(3 * n);
    for i in 0..n {
        eta_time[i] = family.x_time_entry[[i, 0]] * beta_t;
        eta_time[n + i] = family.x_time_exit[[i, 0]] * beta_t;
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
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

/// Build a single-row survival LS family with the production default
/// derivative guard (1e-6) for monotonicity-floor probes.
fn survival_ls_default_guard_unit_family() -> SurvivalLocationScaleFamily {
    SurvivalLocationScaleFamily {
        n: 1,
        y: array![1.0],
        w: array![1.0],
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        x_time_entry: Arc::new(array![[1.0]]),
        x_time_exit: Arc::new(array![[1.0]]),
        x_time_deriv: Arc::new(array![[1.0]]),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    }
}

/// #1396 regression: the event Jacobian `g = d_raw + qdot` is formed as a
/// compensated subtraction of two near-equal-magnitude, opposite-sign operands
/// (the constrained `d_raw` and the unconstrained threshold/log-σ `qdot`). At a
/// feasible monotone boundary that cancellation can tip the reconstructed `g` a
/// hair below zero — strictly smaller in magnitude than the derivative guard —
/// which the monotonicity check must FLOOR to the guard rather than rejecting as
/// a non-monotone state (the `heart_failure_structural_time_small` abort). A
/// genuinely non-monotone state (g negative by far more than the guard) must
/// still hard-error.
#[test]
fn survival_ls_monotonicity_floors_near_cancellation_negative_velocity() {
    let family = survival_ls_default_guard_unit_family();
    let guard = family.time_derivative_lower_bound();

    // A near-cancellation that lands g just barely negative: d_raw and qdot are
    // O(1) and opposite-signed, differing only at the ~1e-7 level — exactly the
    // boundary-cancellation regime. `row_predictor_state` forms
    // g = compensated_difference(d_raw, -qdot1) = d_raw + qdot1.
    let d_raw = 1.0_f64;
    let qdot1 = -(1.0_f64 + 2.0e-7); // g = d_raw + qdot1 = -2.0e-7, within the guard band
    let state = family.row_predictor_state(0.1, 0.2, d_raw, -0.3, -0.3, qdot1);
    assert!(
        state.g < 0.0 && state.g.abs() < guard,
        "fixture must produce a tiny-negative velocity inside the guard band: g={}, guard={guard}",
        state.g,
    );
    let kernel = family
        .exact_row_kernel(0, state)
        .expect("near-cancellation negative velocity must be floored, not rejected")
        .expect("positive-weight row");
    // Floored to the guard ⇒ log(g) = log(guard), finite.
    assert!(
        (kernel.log_g - guard.ln()).abs() <= 1e-12,
        "velocity must be floored to the guard: log_g={}, expected log(guard)={}",
        kernel.log_g,
        guard.ln(),
    );

    // A genuinely non-monotone state (g negative by far more than the guard)
    // must still be rejected — the floor does not mask real violations.
    let bad_state = family.row_predictor_state(0.1, 0.2, 1.0, -0.3, -0.3, -1.5);
    assert!(
        bad_state.g < -guard,
        "fixture must produce a large-negative velocity below -guard: g={}",
        bad_state.g,
    );
    let err = family
        .exact_row_kernel(0, bad_state)
        .expect_err("a genuinely non-monotone velocity must hard-error");
    assert!(
        err.contains("monotonicity violated"),
        "unexpected error for non-monotone velocity: {err}",
    );
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
) -> gam_math::jet_tower::KernelChannels<2> {
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
    gam_math::jet_tower::KernelChannels {
        value: channels.value,
        gradient: channels.gradient,
        hessian: channels.hessian,
        third,
        fourth,
    }
}

#[test]
fn survival_ls_location_scale_jet_program_matches_exact_row_kernel_all_channels() {
    use gam_math::jet_tower::{evaluate_program, verify_kernel_channels};

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

impl gam_math::jet_tower::RowNllProgram<SLS_ROW_K> for SurvivalLsJointNllProgram<'_> {
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
        p: &[gam_math::jet_tower::Tower4<SLS_ROW_K>; SLS_ROW_K],
    ) -> Result<gam_math::jet_tower::Tower4<SLS_ROW_K>, String> {
        use gam_math::jet_tower::Tower4;

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
    let dense =
        |ch: usize| DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(col(ch)));
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
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    }
}

/// Block states matching [`survival_ls_joint_oracle_family`]: every block
/// coefficient is 1, and the eta vectors carry the stacked layout
/// `validate_joint_states` expects for time-varying blocks. The time block
/// stacks `[entry; exit; derivative]` (matching the solver design's
/// `MultiChannelOperator` order), while the threshold / log-sigma blocks stack
/// `[exit; entry; derivative]` — exactly the conventions the production
/// `prepare.rs` stacking and `validate_joint_states` slicing use (gam#1396).
fn survival_ls_joint_oracle_states(primaries: &[[f64; SLS_ROW_K]]) -> Vec<ParameterBlockState> {
    let n = primaries.len();
    let stacked = |first: usize, second: usize, deriv: usize| {
        let mut eta = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            eta[i] = primaries[i][first];
            eta[n + i] = primaries[i][second];
            eta[2 * n + i] = primaries[i][deriv];
        }
        eta
    };
    vec![
        // Time block: `[entry(ch0); exit(ch1); deriv(ch2)]`.
        ParameterBlockState {
            beta: array![1.0],
            eta: stacked(0, 1, 2),
        },
        // Threshold block: `[exit(ch3); entry(ch4); deriv(ch5)]`.
        ParameterBlockState {
            beta: array![1.0],
            eta: stacked(3, 4, 5),
        },
        // Log-sigma block: `[exit(ch6); entry(ch7); deriv(ch8)]`.
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
    use crate::row_kernel::RowKernel;
    use gam_math::jet_tower::{KernelChannels, evaluate_program, verify_kernel_channels};

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

/// #932 single-source / packed-scalar contract: the production
/// `row_third_contracted` / `row_fourth_contracted` (now evaluated through the
/// PACKED directional scalars `OneSeed<9>` / `TwoSeed<9>` — 1.46 / 2.8 KiB,
/// never the ~50 KiB dense `Tower4<9>`) must equal the contraction of the
/// INDEPENDENT dense `SurvivalLsJointNllProgram` `Tower4<9>` (a separate row-NLL
/// implementation, not the production `sls_row_nll`) to ≤ 1e-9. The packed
/// scalars fold the contraction direction INTO the differentiation via the
/// nilpotent ε/δ, never materialising `t3`/`t4`; the independent tower
/// materialises the full tensor then contracts. A regression that desyncs the
/// packed path from the dense answer — or reintroduces a separate hand
/// directional tower — fails here. This is the oracle that lets
/// `row_kernel_directional_supported()` return true: the memory-bounded packed
/// path is provably the dense-tower answer.
#[test]
fn survival_ls_packed_directional_matches_dense_tower_932() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_packed_directional_matches_dense_tower_body)
        .expect("spawn wide-stack packed-directional oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS packed-directional #932 oracle thread must complete"
    );
}

fn survival_ls_packed_directional_matches_dense_tower_body() {
    use crate::row_kernel::RowKernel;
    use gam_math::jet_tower::evaluate_program;

    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, 0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 0.35];
    let weight = [1.0, 0.8, 1.2, 1.3];
    let n = primaries.len();

    // Dense deterministic directions so every one of the nine channels
    // participates in every contraction (no dropped/flipped cross block can hide).
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
        // INDEPENDENT dense ground truth: the `SurvivalLsJointNllProgram`
        // `RowNllProgram<9>` (the same one the all-channels oracle uses) — a
        // separate Tower4<9> implementation of the row NLL, NOT the production
        // `sls_row_nll`. Comparing the packed production contractions against
        // THIS independent tower (rather than `sls_row_nll` at `Tower4`) keeps
        // the oracle's truth genuinely independent of the code under test.
        let program = SurvivalLsJointNllProgram {
            inverse_link: &inverse_link,
            primaries: primaries.clone(),
            event: event.to_vec(),
            weight: weight.to_vec(),
        };

        for row in 0..n {
            // Dense ground truth: build the full Tower4<9> once and contract its
            // t3 / t4 channels. The production methods must reproduce these
            // exactly through the packed OneSeed / TwoSeed scalars.
            let tower = evaluate_program(&program, row).expect("dense row tower");
            for u in &dirs {
                let dense_third = tower.third_contracted(u);
                let packed_third =
                    RowKernel::row_third_contracted(&kernel, row, u).expect("packed third");
                for a in 0..SLS_ROW_K {
                    for b in 0..SLS_ROW_K {
                        let want = dense_third[a][b];
                        let got = packed_third[a][b];
                        assert!(
                            (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                            "{distribution:?} row {row} third[{a}][{b}]: packed OneSeed {got} \
                             vs dense Tower4 {want}"
                        );
                    }
                }
                for v in &dirs {
                    let dense_fourth = tower.fourth_contracted(u, v);
                    let packed_fourth = RowKernel::row_fourth_contracted(&kernel, row, u, v)
                        .expect("packed fourth");
                    for a in 0..SLS_ROW_K {
                        for b in 0..SLS_ROW_K {
                            let want = dense_fourth[a][b];
                            let got = packed_fourth[a][b];
                            assert!(
                                (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                                "{distribution:?} row {row} fourth[{a}][{b}]: packed TwoSeed \
                                 {got} vs dense Tower4 {want}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// #932 STRESS hardening of [`survival_ls_packed_directional_matches_dense_tower_932`].
///
/// The benign-fixture oracle above proves the packed `OneSeed`/`TwoSeed`
/// contractions equal the independent dense `Tower4<9>` on moderate primaries.
/// This arm hardens that gate in two ways a benign equality check cannot:
///
///   1. HIGH-CURVATURE / NEAR-DEGENERATE fixture. The primaries are pushed into
///      the regime where the per-row NLL stacks saturate and their high-order
///      jets blow up — exactly where a dropped/mis-scaled 3rd-or-4th-order term
///      hides on a benign point:
///        * deep-tail exit/entry indices `u0,u1` (large negative → `log S`
///          curvature large; the `compose_unary` survival stack is evaluated far
///          from 0 where its 3rd/4th derivatives dominate);
///        * extreme `log-σ` channels (`exp(−η_lσ)` spans ~e^−2..e^2), so the
///          threshold contributions to `u0,u1,g` are strongly amplified;
///        * a deliberately SMALL-but-positive event Jacobian `g` (near the
///          `log g` singularity, where `∂ⁿ log g = (−1)ⁿ⁻¹(n−1)!/gⁿ` is huge),
///          stressing the `survival_ls_positive_log_stack` chain at 3rd/4th order.
///      A vacuity guard asserts the fixture actually reaches this regime (small
///      `g`, large `|u|`) so the stress is real, not nominal.
///
///   2. PLANTED SIGN-FLIP tripwire. Equality `packed == dense` alone does not
///      prove the oracle could SEE a wrong packed value. After the exact match we
///      negate a representative 4th-order cross entry and assert the packed value
///      does NOT match the flip — i.e. the oracle has genuine resolving power
///      against a sign/term error on the very block it guards.
#[test]
fn survival_ls_packed_directional_matches_dense_tower_high_curvature_932() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_packed_directional_high_curvature_body)
        .expect("spawn wide-stack high-curvature packed-directional oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS high-curvature #932 oracle thread must complete"
    );
}

fn survival_ls_packed_directional_high_curvature_body() {
    use crate::row_kernel::RowKernel;
    use gam_math::jet_tower::evaluate_program;

    // Channel layout (matches `SurvivalLsJointNllProgram::row_nll`):
    //   [0]=t_entry [1]=t_exit [2]=t_deriv [3]=thr_exit [4]=thr_entry
    //   [5]=thr_deriv [6]=lσ_exit [7]=lσ_entry [8]=lσ_deriv.
    // These rows drive `u0,u1` deep into the tail and `g` small-positive:
    //   inv_σ_exit = e^{−p6}, u1 = p1 − p3·inv_σ_exit,
    //   g = p2 + inv_σ_exit·(p3·p8 − p5)  (must stay > 0 for log g).
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        // Large log-σ swing (p6=1.8 ⇒ inv_σ_exit≈0.165; p7=−1.6 ⇒ inv_σ_entry≈4.95),
        // big thresholds ⇒ |u0|,|u1| large; g≈0.9+0.165·(2.2·0.5−0.7)=0.966 → push
        // smaller below.
        [2.4, -3.1, 0.9, 2.2, 3.5, 0.7, 1.8, -1.6, 0.5],
        // Deep-tail censored row with a SMALL event-Jacobian-style g build and a
        // strongly negative exit index.
        [-2.8, -4.2, 0.35, -2.6, -3.4, 1.3, -1.7, 1.5, -0.9],
        // Near-degenerate g: p2=0.12, inv_σ_exit=e^{-0.4}≈0.670,
        // g=0.12+0.670·(1.4·0.6−0.18)=0.12+0.670·0.66=0.562 → still positive but
        // with large threshold curvature feeding u1.
        [0.6, 3.8, 0.12, 1.4, 2.1, 0.18, 0.4, -0.5, 0.6],
        // Tiny g with big tail: p2=0.05, inv_σ_exit=e^{-1.1}≈0.333,
        // g=0.05+0.333·(0.9·0.4−0.05)=0.05+0.333·0.31=0.153 (small ⇒ huge log g jets).
        [-1.2, 4.6, 0.05, 0.9, -2.3, 0.05, 1.1, -1.3, 0.4],
    ];
    let event = [1.0, 0.0, 1.0, 1.0];
    let weight = [1.0, 0.9, 1.2, 0.8];
    let n = primaries.len();

    // Dense deterministic directions so every one of the nine channels
    // participates in every contraction (no dropped/flipped cross block can hide).
    let dirs: [[f64; SLS_ROW_K]; 3] = [
        [0.7, -1.3, 0.5, 0.9, -0.6, 0.3, -1.1, 0.4, 0.8],
        [-0.4, 0.6, -1.1, 0.3, 1.2, -0.7, 0.5, -0.9, 0.2],
        [1.2, 0.2, -0.7, -0.5, 0.4, 1.0, -0.3, 0.6, -1.2],
    ];

    // Vacuity guard: confirm at least one event row actually reaches the
    // high-curvature regime — a small-positive `g` and a large-magnitude exit
    // index `u1` — so the stress is genuine, not a nominal relabelling of a
    // benign point.
    let mut min_event_g = f64::INFINITY;
    let mut max_abs_u1 = 0.0_f64;
    for (row, p) in primaries.iter().enumerate() {
        let inv_sigma_exit = (-p[6]).exp();
        let u1 = p[1] - p[3] * inv_sigma_exit;
        let g = p[2] + inv_sigma_exit * (p[3] * p[8] - p[5]);
        assert!(
            g > 0.0,
            "fixture row {row} has non-positive event Jacobian g={g:.4e}; log g undefined"
        );
        if event[row] != 0.0 {
            min_event_g = min_event_g.min(g);
        }
        max_abs_u1 = max_abs_u1.max(u1.abs());
    }
    assert!(
        min_event_g < 0.2,
        "high-curvature fixture vacuous: smallest event-row g={min_event_g:.4e} is not near \
         the log g singularity (want < 0.2); the small-g 3rd/4th-order stress is absent"
    );
    assert!(
        max_abs_u1 > 3.0,
        "high-curvature fixture vacuous: largest |u1|={max_abs_u1:.4e} is not deep in the \
         survival tail (want > 3.0); the saturated log-survival curvature stress is absent"
    );

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

        // A slightly looser relative tolerance than the benign oracle's 1e-9:
        // the deep-tail/small-g jets have magnitudes up to ~1e3, so the
        // `(1+|want|)` relative band already scales with that; the absolute floor
        // stays tight. A genuine dropped term is O(magnitude), far outside this.
        let rel_tol = 1e-8_f64;

        for row in 0..n {
            let tower = evaluate_program(&program, row).expect("dense row tower (high curvature)");
            for u in &dirs {
                let dense_third = tower.third_contracted(u);
                let packed_third =
                    RowKernel::row_third_contracted(&kernel, row, u).expect("packed third");
                for a in 0..SLS_ROW_K {
                    for b in 0..SLS_ROW_K {
                        let want = dense_third[a][b];
                        let got = packed_third[a][b];
                        assert!(
                            (got - want).abs() <= rel_tol * (1.0 + want.abs()),
                            "{distribution:?} HC row {row} third[{a}][{b}]: packed OneSeed {got} \
                             vs dense Tower4 {want}"
                        );
                    }
                }
                for v in &dirs {
                    let dense_fourth = tower.fourth_contracted(u, v);
                    let packed_fourth = RowKernel::row_fourth_contracted(&kernel, row, u, v)
                        .expect("packed fourth");
                    for a in 0..SLS_ROW_K {
                        for b in 0..SLS_ROW_K {
                            let want = dense_fourth[a][b];
                            let got = packed_fourth[a][b];
                            assert!(
                                (got - want).abs() <= rel_tol * (1.0 + want.abs()),
                                "{distribution:?} HC row {row} fourth[{a}][{b}]: packed TwoSeed \
                                 {got} vs dense Tower4 {want}"
                            );
                        }
                    }
                }
            }
        }

        // ── Planted sign-flip tripwire ───────────────────────────────────────
        // Pick the event row with the smallest g (max log-g curvature) and a
        // 4th-order cross entry that is genuinely nonzero, then assert that
        // negating the dense truth leaves the packed band: the oracle can SEE a
        // sign/term error on the block it guards (not just confirm equality).
        let trip_row = 3usize; // tiny-g, deep-tail event row
        let du = &dirs[0];
        let dv = &dirs[1];
        let dense_fourth = evaluate_program(&program, trip_row)
            .expect("trip tower")
            .fourth_contracted(du, dv);
        let packed_fourth = RowKernel::row_fourth_contracted(&kernel, trip_row, du, dv)
            .expect("trip packed fourth");
        // (t_deriv, lσ_exit) = [2][6]: a cross block that genuinely couples the
        // event-Jacobian and scale channels through g and u1.
        let (ca, cb) = (2usize, 6usize);
        let want = dense_fourth[ca][cb];
        if want.abs() > 1e-6 {
            let flipped = -packed_fourth[ca][cb];
            assert!(
                (flipped - want).abs() > 1e-8 * (1.0 + want.abs()),
                "{distribution:?} oracle failed to reject a planted fourth[{ca}][{cb}] sign flip: \
                 flipped {flipped:+.9e} vs dense truth {want:+.9e} — the high-curvature gate has \
                 no resolving power against a cross-block sign error"
            );
        }
    }
}

/// The hand-derived analytic joint-Hessian directional derivative
/// (`exact_newton_joint_hessian_directional_derivative_from_parts`) must agree
/// with the jet-tower-certified generic row-kernel directional derivative on a
/// FULLY TIME-VARYING family — i.e. with the derivative threshold/log-sigma
/// channels (the velocity / `qdot` coordinate) live. The pre-existing FD
/// coverage only exercises non-time-varying fixtures, where the velocity
/// coordinate is inert, so it cannot witness a dropped `qdot` third-order
/// contribution.
///
/// #932: `row_kernel_directional_supported()` is now enabled for non-wiggle
/// rows, so the row-kernel reference below is exactly the path a production fit
/// takes — and the generic row-kernel directional derivative now consumes the
/// PACKED `OneSeed<9>` scalar (no dense `Tower4<9>`). This test pins that packed
/// path against the hand path to 1e-7, so enabling the gate is behaviour-
/// preserving; both call sites are invoked explicitly here, independent of the
/// gate.
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
    use crate::row_kernel::{RowSet, row_kernel_directional_derivative_generic};

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
    // Row-kernel assembly plus the directional-Hessian FD oracle keep several
    // dense joint Hessians live on the stack; run on a wide-stack thread like
    // the other survival-LS jet-tower oracles.
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_row_kernel_matches_bespoke_assembly_body)
        .expect("spawn wide-stack row-kernel oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS row-kernel oracle thread must complete"
    );
}

fn survival_ls_row_kernel_matches_bespoke_assembly_body() {
    use crate::row_kernel::{
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

/// #932: assembler-level single-source guard for the TIME-VARYING joint
/// Hessian path across EVERY residual distribution.
///
/// `survival_ls_row_kernel_matches_bespoke_assembly` (#921) already pins the
/// generic row-kernel joint Hessian (`row_kernel_hessian_dense`, sourced from
/// the once-written `sls_row_nll` through `Order2<9>`) to the bespoke
/// hand-assembler (`assemble_joint_hessian_from_quantities`). But that fixture
/// is Gaussian-only and uses the SIMPLE block shape
/// (`x_{threshold,log_sigma}_{entry,deriv} = None`), so it only exercises the
/// `else` branches of the assembler. The hand-derived `if let Some(x_*_deriv)`
/// branches — the time-varying blocks with the extra `h_exit_deriv` /
/// `h_entry` cross-block weight expressions (e.g.
/// `mxtwx(x_threshold_exit, &h_exit_deriv, x_t_deriv)`,
/// `-(d2_qdot1·dqdot_t·dqdot_lsd + d1_qdot1·d2qdot_tlsd)`) — are EXACTLY the
/// #736 dropped/sign-flipped cross-term genus, and no assembler==tower oracle
/// covered them: they were only checked at the per-row level
/// (`survival_ls_joint_row_kernel_agrees_with_jet_tower_program_all_channels`),
/// never assembled into the joint matrix and compared.
///
/// This fills that gap. `survival_ls_joint_oracle_family` populates every
/// entry/deriv design, so the assembler takes its time-varying branches; the
/// generic engine builds the same joint Hessian from the single-sourced row
/// NLL. They must agree to ~1e-9 (no FD: both are analytic), for
/// Gaussian / Gumbel (Weibull AFT) / Logistic (log-logistic AFT). A dropped
/// cross-block term in the hand assembler shifts a joint entry well outside
/// 1e-9 and fails loudly — the assembler-level analogue of the per-row
/// #736 guard.
#[test]
fn survival_ls_time_varying_joint_hessian_matches_single_sourced_tower_932() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_time_varying_joint_hessian_tower_body)
        .expect("spawn wide-stack assembler-tower oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS time-varying assembler-vs-tower oracle thread must complete"
    );
}

fn survival_ls_time_varying_joint_hessian_tower_body() {
    use crate::row_kernel::{
        RowSet, build_row_kernel_cache, row_kernel_gradient, row_kernel_hessian_dense,
        row_kernel_log_likelihood,
    };

    // Same nine-channel fixture the all-channels per-row oracle uses: exact
    // deaths, right-censored rows, deep / effectively-absent left truncation,
    // extreme exit tails, and a fractional event weight — every channel and
    // cross block populated, all clear of the monotonicity guard.
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [-6.5, 5.6, 1.1, -0.7, -0.3, -0.15, 0.2, 0.4, 0.1],
        [-1.0, -5.2, 0.7, 0.5, 0.6, 0.3, -0.1, -0.3, -0.25],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, 0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 0.0, 1.0, 0.35];
    let weight = [1.0, 0.8, 1.2, 0.9, 1.1, 1.3];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        // Sanity: this fixture must drive the time-varying assembler branches.
        assert!(
            family.x_threshold_entry.is_some()
                && family.x_threshold_deriv.is_some()
                && family.x_log_sigma_entry.is_some()
                && family.x_log_sigma_deriv.is_some()
                && family.x_link_wiggle.is_none(),
            "fixture must populate every entry/deriv design and no link-wiggle so the \
             assembler takes its time-varying branches"
        );
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

        // Single-sourced tower joint Hessian: row kernel (Order2<9> over
        // sls_row_nll) → dense block assembly.
        let cache = build_row_kernel_cache(&kernel, &RowSet::All).expect("row kernel cache");
        let h_tower = row_kernel_hessian_dense(&kernel, &cache, &RowSet::All);

        // Bespoke hand assembler (time-varying branches).
        let h_bespoke = family
            .assemble_joint_hessian_from_quantities(&q, &states)
            .expect("bespoke joint Hessian")
            .expect("bespoke joint Hessian present");

        assert_eq!(
            h_tower.dim(),
            h_bespoke.dim(),
            "{distribution:?}: joint Hessian shape mismatch"
        );
        for ((a, b), &bespoke) in h_bespoke.indexed_iter() {
            let tower = h_tower[[a, b]];
            assert!(
                (tower - bespoke).abs() <= 1e-9 * (1.0 + bespoke.abs()),
                "{distribution:?}: joint Hessian [{a}][{b}] hand-assembler {bespoke} != \
                 single-sourced tower {tower}"
            );
        }

        // Gradient: the single-sourced engine's ∇(nll) must assemble at the
        // joint coefficient dimension and stay finite (the gradient and Hessian
        // share the one cache, so a consistent triple).
        let g_tower = row_kernel_gradient(&kernel, &cache, &RowSet::All);
        assert_eq!(
            g_tower.len(),
            *kernel.offsets.last().unwrap(),
            "{distribution:?}: gradient dimension"
        );
        assert!(
            g_tower.iter().all(|v| v.is_finite()),
            "{distribution:?}: single-sourced gradient must be finite"
        );

        // Log-likelihood consistency: the engine's ℓ = −Σ nll_i must match the
        // bespoke per-row `exact_row_kernel(row).log_likelihood()` sum.
        let ll_tower = row_kernel_log_likelihood(&cache, &RowSet::All);
        let mut ll_bespoke = 0.0;
        for i in 0..family.n {
            let state = family.row_predictor_state(
                dynamic.h_entry[i],
                dynamic.h_exit[i],
                dynamic.hdot_exit[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            if let Some(k) = family.exact_row_kernel(i, state).expect("row kernel") {
                ll_bespoke += k.log_likelihood();
            }
        }
        assert!(
            (ll_tower - ll_bespoke).abs() <= 1e-9 * (1.0 + ll_bespoke.abs()),
            "{distribution:?}: log-likelihood single-sourced {ll_tower} != bespoke {ll_bespoke}"
        );
    }
}

/// #932: the production survival-LS log-likelihood block GRADIENT
/// (`evaluate_log_likelihood_and_block_gradients` — the LIVE outer-Newton
/// gradient path; the sparse hand assembler is the live joint-Hessian path as a
/// measured perf exception) must equal the single-sourced row-kernel gradient.
///
/// The joint Hessian is now pinned to the tower (the time-varying assembler
/// oracle above + the #921 simple-shape oracle), and the gradient-vs-FD SAS
/// test covers one link, but no exact oracle pinned the bespoke block gradient
/// to `row_kernel_gradient` (built from the same `sls_row_nll` the Hessian uses)
/// across distributions and the time-varying shape. `survival_joint_gradient
/// _evaluation_matches_evaluate_block_gradients` only checks the bespoke path
/// against itself.
///
/// `row_kernel_gradient` returns ∇(nll) = −∇ℓ (the cached per-row jets are of
/// the negative log-likelihood, pulled back), while
/// `evaluate_log_likelihood_and_block_gradients` returns the log-likelihood
/// gradient ∇ℓ; both block orders are `[time, threshold, log_sigma]`
/// (`block_gradients = vec![grad_time, grad_t, grad_ls]` and
/// `joint_block_offsets`), so the flattened bespoke ∇ℓ must equal `−g_tower`
/// to ~1e-9, for Gaussian / Gumbel / Logistic on the every-channel fixture. A
/// dropped term in the hand block gradient now fails loudly.
#[test]
fn survival_ls_block_gradient_matches_single_sourced_tower_932() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_block_gradient_tower_body)
        .expect("spawn wide-stack gradient oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS block-gradient-vs-tower oracle thread must complete"
    );
}

fn survival_ls_block_gradient_tower_body() {
    use crate::row_kernel::{RowSet, build_row_kernel_cache, row_kernel_gradient};

    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [-6.5, 5.6, 1.1, -0.7, -0.3, -0.15, 0.2, 0.4, 0.1],
        [-1.0, -5.2, 0.7, 0.5, 0.6, 0.3, -0.1, -0.3, -0.25],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, 0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 0.0, 1.0, 0.35];
    let weight = [1.0, 0.8, 1.2, 0.9, 1.1, 1.3];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        let states = survival_ls_joint_oracle_states(&primaries);

        // Single-sourced tower gradient (∇nll = −∇ℓ).
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
        let g_tower_nll = row_kernel_gradient(&kernel, &cache, &RowSet::All);

        // Bespoke production block gradients (∇ℓ), flattened in the joint
        // [time, threshold, log_sigma] layout.
        let (_ll, block_gradients) = family
            .evaluate_log_likelihood_and_block_gradients(&states)
            .expect("bespoke block gradients");
        let offsets = family.joint_block_offsets();
        let total = *offsets.last().unwrap();
        let mut g_bespoke_ll = vec![0.0_f64; total];
        let mut pos = 0usize;
        for block in &block_gradients {
            for &v in block.iter() {
                g_bespoke_ll[pos] = v;
                pos += 1;
            }
        }
        assert_eq!(
            pos, total,
            "{distribution:?}: flattened bespoke gradient width {pos} != joint total {total}"
        );
        assert_eq!(
            g_tower_nll.len(),
            total,
            "{distribution:?}: tower gradient width"
        );

        // ∇ℓ_bespoke == −∇nll_tower.
        for i in 0..total {
            let bespoke = g_bespoke_ll[i];
            let tower = -g_tower_nll[i];
            assert!(
                (bespoke - tower).abs() <= 1e-9 * (1.0 + tower.abs()),
                "{distribution:?}: block gradient[{i}] bespoke ∇ℓ {bespoke:.9e} != \
                 single-sourced −∇nll {tower:.9e}"
            );
        }
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
    family.x_threshold = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::<f64>::zeros((n, p)),
    ));
    family.x_log_sigma = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
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
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(
        hessian,
        gam_problem::DeclaredHessianForm::Either,
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
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
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
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
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
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
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
    use gam_identifiability::canonical::canonicalize_for_identifiability;

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
    let anchor =
        crate::survival::construction::resolve_survival_time_anchor_value(&age_entry, None)
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
                link, eta, 0.0,
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
                    link, eta, 0.0,
                )
                .expect("log-survival stack");
            let log_s_analytic = [-r, -dr, -ddr, -dddr];
            let (_, p1, p2, p3, p4) =
                SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(link, eta, 0.0)
                    .expect("log-pdf stack");
            let log_pdf_analytic = [p1, p2, p3, p4];

            for (k, &analytic) in log_s_analytic.iter().enumerate() {
                let order = k + 1;
                let h = match order {
                    1 | 2 => 1e-3,
                    3 => 3e-3,
                    4 => 1e-2,
                    _ => unreachable!("stencil supports derivative orders 1..=4"),
                };
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
                let h = match order {
                    1 | 2 => 1e-3,
                    3 => 3e-3,
                    4 => 1e-2,
                    _ => unreachable!("stencil supports derivative orders 1..=4"),
                };
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
fn exact_survival_neglog_derivatives_rescaled_scale_cloglog_uniformly() {
    // The survival ratio stack must carry the SAME exp(-L) derivative rescale
    // as the log-pdf stack: the two enter the joint Hessian side by side, and
    // the logdet correction `logdet(H_exact) = logdet(H_scaled) + p*L` is only
    // valid if EVERY row's curvature (event, censored, and left-truncated
    // alike) is scaled uniformly. The log S value channel stays unshifted.
    let eta = 2.25_f64;
    let log_scale = 1.5_f64;
    let raw = eta.exp();
    let scaled = (eta - log_scale).exp();

    let (log_s, r, dr, ddr, dddr) =
        SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
            &InverseLink::Standard(StandardLink::CLogLog),
            eta,
            log_scale,
        )
        .expect("rescaled cloglog survival derivatives");

    assert!((log_s + raw).abs() <= 1e-15 * raw);
    for (label, actual) in [("r", r), ("dr", dr), ("ddr", ddr), ("dddr", dddr)] {
        assert!(
            (actual - scaled).abs() <= 1e-15 * scaled,
            "CLogLog survival ratio derivative {label} must scale by exp(-L): actual={actual} expected={scaled}"
        );
    }

    let ((pair_log_s, pair_r, pair_dr, pair_ddr, pair_dddr), _) =
        SurvivalLocationScaleFamily::clglog_exit_pair(eta, log_scale);
    assert!((pair_log_s + raw).abs() <= 1e-15 * raw);
    for (label, actual) in [
        ("pair r", pair_r),
        ("pair dr", pair_dr),
        ("pair ddr", pair_ddr),
        ("pair dddr", pair_dddr),
    ] {
        assert!(
            (actual - scaled).abs() <= 1e-15 * scaled,
            "fused CLogLog survival ratio derivative {label} must scale by exp(-L): actual={actual} expected={scaled}"
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
            0.0,
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
            gam_test_support::assert_matrix_derivativefd(
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

        gam_test_support::assert_matrix_derivativefd(
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
    let beta_threshold = array![0.1 * crate::sigma_link::safe_exp(700.0)];
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
    let beta_threshold = array![0.1 * crate::sigma_link::safe_exp(700.0)];
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
    let beta_threshold = array![0.1 * crate::sigma_link::safe_exp(700.0)];
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

/// #1389 regression: the joint-Hessian directional-derivative velocity (event
/// Jacobian `g`) pass is skipped before any `p²` allocation when no weighted row
/// carries live qdot-derivative mass. Censored rows carry event_weight 0, so an
/// all-censored fixture has `d1_qdot1 = d2_qdot1 = d_h_d = 0` on every row and
/// the velocity term is identically zero — the skip path must therefore produce
/// a directional derivative that still matches the finite difference of the
/// joint gradient (i.e. the skip omits only a zero contribution).
#[test]
fn joint_dh_velocity_skip_is_exact_on_all_censored_rows() {
    let mut family = survival_exact_newton_test_family();
    // All rows censored ⇒ event_weight 0 on every row ⇒ no velocity mass, so the
    // #1389 `any_live_qdot` guard short-circuits the velocity pass.
    family.y = array![0.0, 0.0, 0.0];

    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

    // Direction perturbs every block so any dropped cross-velocity term would
    // show up in the comparison.
    let direction = array![1.0, 1.0, 1.0];
    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &direction)
        .expect("analytic joint dH")
        .expect("expected exact joint dH");

    // The directional derivative of the joint NLL Hessian must equal the central
    // finite difference of that Hessian along `direction`. Because the velocity
    // pass is skipped (all rows censored), `analytic` carries no velocity term;
    // the FD-of-Hessian is the independent ground truth, so a match certifies the
    // skip dropped only a zero contribution.
    let eps = 1e-6;
    let hessian_at = |scale: f64| -> Array2<f64> {
        let bt = &beta_time + scale * direction[0];
        let bth = &beta_threshold + scale * direction[1];
        let bls = &beta_log_sigma + scale * direction[2];
        family
            .exact_newton_joint_hessian(&survival_exact_newton_rebuild_states(&bt, &bth, &bls))
            .expect("joint hessian")
            .expect("expected exact joint hessian")
    };
    let fd = (&hessian_at(eps) - &hessian_at(-eps)) / (2.0 * eps);
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (analytic[[r, c]] - fd[[r, c]]).abs() <= 5e-4,
                "all-censored velocity-skip dH[{r}][{c}] mismatch: analytic={} fd={}",
                analytic[[r, c]],
                fd[[r, c]],
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
            let ell_q = row.d1_q0 + row.d1_q1;
            let ell_h0q = row.h_time_h0;
            let ell_h1q = row.h_time_h1;
            let ell_qq = row.d2_q0 + row.d2_q1;
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, -0.2
        ]])),
        eta_threshold_offset: array![0.0],
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, -0.2
        ]])),
        eta_threshold_offset: array![0.7],
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_threshold_dense.clone(),
        )),
        eta_threshold_offset: array![0.7, -0.2],
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_log_sigma_dense.clone(),
        )),
        eta_log_sigma_offset: array![0.4, 0.1],
        x_link_wiggle: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(xwiggle_dense.clone()),
        )),
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.25
        ]])),
        eta_threshold_offset: array![0.0],
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_threshold_dense.clone(),
        )),
        eta_threshold_offset: array![0.0, 0.05],
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_threshold_dense,
        )),
        eta_threshold_offset: array![0.0],
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_log_sigma_dense,
        )),
        eta_log_sigma_offset: array![0.0],
        x_link_wiggle: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(x_link_wiggle),
        )),
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0, -0.2
        ]])),
        eta_threshold_offset: array![0.0],
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
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
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::ones(
                (n, 1),
            ))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::ones(
                (n, 1),
            ))),
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
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
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
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

/// #932 (survival link-wiggle — the issue's named next step): the survival
/// location-scale JOINT row NLL written ONCE over [`JetScalar`] is extended with
/// the link-wiggle warp `q = q0 + Σ_j βw_j·B_j(q0)` (and the qdot coupling
/// `g = m1·g0`, `m1 = 1 + Σ_j βw_j·B'_j(q0_exit)`), with the βw amplitudes as
/// extra jet primaries. The mechanically-derived joint Hessian — including the
/// `(η, βw)` and `(βw, βw)` cross blocks a fixed `JᵀHJ` pullback would drop —
/// is pinned against central finite differences of the SAME program's value,
/// across the three residual distributions. This validates the nonlinear
/// link-wiggle pullback (the issue's §5/§13 map-inside-the-program) in the
/// survival row program — the foundation of the single-sourced link-wiggle
/// joint Hessian (`survival_ls_wiggle_joint_hessian_dense`) shipped in
/// production.
#[test]
fn survival_ls_wiggle_jet_program_joint_hessian_matches_fd_932() {
    use gam_math::jet_scalar::JetScalar;
    use gam_math::jet_tower::{RowNllProgramGeneric, generic_row_kernel};

    const PW: usize = 2;
    const KW: usize = SLS_ROW_K + PW; // 9 base channels + 2 wiggle amplitudes

    // Smooth wiggle basis B_j and its first three derivatives at q (any C^3
    // basis exercises the warp; the production spline supplies the same stack).
    fn basis(j: usize, q: f64) -> [f64; 4] {
        match j {
            0 => [0.5 * q * q, q, 1.0, 0.0],
            _ => [q * q * q / 6.0, 0.5 * q * q, q, 1.0],
        }
    }

    struct WiggleProg {
        link: InverseLink,
        w: f64,
        d: f64,
        p: [f64; KW],
    }
    impl RowNllProgramGeneric<KW> for WiggleProg {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; KW], String> {
            if row != 0 {
                return Err(format!("wiggle program: row {row} out of range"));
            }
            Ok(self.p)
        }
        fn row_nll_generic<S: JetScalar<KW>>(&self, row: usize, p: &[S; KW]) -> Result<S, String> {
            if row != 0 {
                return Err(format!("wiggle program: row {row} out of range"));
            }
            // Base nine-channel survival indices (exactly `sls_row_nll`).
            let inv_sigma_entry = p[7].neg().exp();
            let u0 = p[0].sub(&p[4].mul(&inv_sigma_entry));
            let inv_sigma_exit = p[6].neg().exp();
            let u1 = p[1].sub(&p[3].mul(&inv_sigma_exit));
            let g0 = p[2].add(&inv_sigma_exit.mul(&p[3].mul(&p[8]).sub(&p[5])));

            // Link-wiggle warp: amplitudes are primaries 9..9+PW; each basis is
            // composed onto the BASE index jet (so it carries the η-dependence).
            let u0v = JetScalar::value(&u0);
            let u1v = JetScalar::value(&u1);
            let mut u0w = u0;
            let mut u1w = u1;
            let mut m1 = S::constant(1.0);
            for j in 0..PW {
                let bw = p[SLS_ROW_K + j];
                let b0 = basis(j, u0v);
                u0w = u0w.add(&bw.mul(&u0.compose_unary([b0[0], b0[1], b0[2], 0.0, 0.0])));
                let b1 = basis(j, u1v);
                u1w = u1w.add(&bw.mul(&u1.compose_unary([b1[0], b1[1], b1[2], b1[3], 0.0])));
                // B'_j(u1) jet for m1 = 1 + Σ βw·B'(u1) → g_warp = m1·g0.
                m1 = m1.add(&bw.mul(&u1.compose_unary([b1[1], b1[2], b1[3], 0.0, 0.0])));
            }
            let g = m1.mul(&g0);

            let mut nll = u0w
                .compose_unary(survival_ls_log_survival_stack(
                    &self.link,
                    JetScalar::value(&u0w),
                )?)
                .scale(self.w);
            let censored_weight = self.w * (1.0 - self.d);
            if censored_weight != 0.0 {
                nll = nll.add(
                    &u1w.compose_unary(survival_ls_log_survival_stack(
                        &self.link,
                        JetScalar::value(&u1w),
                    )?)
                    .scale(-censored_weight),
                );
            }
            let event_weight = self.w * self.d;
            if event_weight != 0.0 {
                nll = nll
                    .add(
                        &u1w.compose_unary(survival_ls_log_pdf_stack(
                            &self.link,
                            JetScalar::value(&u1w),
                            0.0,
                        )?)
                        .scale(-event_weight),
                    )
                    .add(
                        &g.compose_unary(survival_ls_positive_log_stack(JetScalar::value(&g)))
                            .scale(-event_weight),
                    );
            }
            Ok(nll)
        }
    }

    // η-rich, moderate-tail base primaries; βw amplitudes nonzero so the warp
    // and every wiggle cross block are exercised (event row d=1 → entry logS +
    // exit logφ + qdot log_g all live).
    let p0: [f64; KW] = [
        0.25, 0.9, 1.3, 0.6, -0.1, 0.1, -0.2, -0.05, 0.3, // 9 base channels
        0.3, -0.2, // βw_0, βw_1
    ];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let link = residual_distribution_inverse_link(distribution);
        let value = |p: [f64; KW]| -> f64 {
            generic_row_kernel(
                &WiggleProg {
                    link: link.clone(),
                    w: 1.0,
                    d: 1.0,
                    p,
                },
                0,
            )
            .expect("wiggle program value")
            .0
        };
        let h_jet = generic_row_kernel(
            &WiggleProg {
                link: link.clone(),
                w: 1.0,
                d: 1.0,
                p: p0,
            },
            0,
        )
        .expect("wiggle program jet")
        .2;

        let hs = 1e-4;
        for a in 0..KW {
            for b in 0..KW {
                let mut pp = p0;
                pp[a] += hs;
                pp[b] += hs;
                let mut pm = p0;
                pm[a] += hs;
                pm[b] -= hs;
                let mut mp = p0;
                mp[a] -= hs;
                mp[b] += hs;
                let mut mm = p0;
                mm[a] -= hs;
                mm[b] -= hs;
                let fd = (value(pp) - value(pm) - value(mp) + value(mm)) / (4.0 * hs * hs);
                let scale = h_jet[a][b].abs().max(fd.abs()).max(1.0);
                assert!(
                    (h_jet[a][b] - fd).abs() <= 2e-3 * scale,
                    "{distribution:?}: wiggle joint Hessian [{a}][{b}] jet {} vs FD {}",
                    h_jet[a][b],
                    fd
                );
            }
        }
    }
}

/// #932 (survival link-wiggle single-source verification): the production
/// wiggle joint Hessian — `survival_ls_wiggle_joint_hessian_dense`, the §13
/// warp row program (`sls_row_nll` extended with `q = q0 + Σ βw·B(q0)` and the
/// qdot coupling `g = m1·g0`) that every production consumer now routes through
/// — must equal an INDEPENDENT tower assembled here from `wiggle_nll` with a
/// hand-rolled `JᵀHJ` pullback. This cross-validates the §13 path AND its
/// row-kernel pullback against independent code; combined with the FD oracle
/// `survival_ls_wiggle_jet_program_joint_hessian_matches_fd_932` (which pins the
/// §13 primary algebra to finite differences), the wiggle joint Hessian is
/// fully verified. The legacy bespoke `assemble_h_wiggle` is RETIRED for wiggle:
/// it disagreed with the §13 source by ~15% at `[0][0]` (a dropped warp coupling
/// — the #736 duplicate-engine genus), so the last production consumer of it (the
/// trust-region metric floor) was repointed to the §13 source.
#[test]
fn survival_ls_wiggle_joint_hessian_matches_assembler_932() {
    use gam_math::jet_scalar::{JetScalar, Order2};

    // event rows (d=1) so entry-logS + exit-logphi + qdot-log_g are all live;
    // moderate-tail primaries clear of the monotonicity guard.
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, 0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 1.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();

    // Seed indices for the wiggle DESIGN matrix `x_link_wiggle` only (its column
    // count `pw` and `etaw = X·betaw`). The warp basis derivative stacks are NOT
    // evaluated here: they must be taken at the model residual `value(u1)` /
    // `value(u0)` (the point `sls_row_nll_wiggle`'s `compose_unary` composes onto),
    // which is computed from the production `dynamic` geometry inside the loop.
    let q0_exit = Array1::from_shape_fn(n, |i| {
        primaries[i][1] - primaries[i][3] * (-primaries[i][6]).exp()
    });

    // A small monotone wiggle basis; degree/knots chosen for a few columns.
    // Degree-3 clamped knot vector: the I-spline derivative path integrates a
    // degree-3 B-spline, whose `validate_knots_for_degree` floor is `2*(3+1)=8`
    // knots, so the previous 5-knot vector aborted this oracle in fixture setup
    // (the basis builder returned `Insufficient knots for degree 3 spline`). The
    // clamp endpoints (-2.5, 3.2) bracket every `q0_exit`/`q0_entry` index above
    // so the warp basis is evaluated strictly inside its support. (Mirrors the
    // clamped 8-knot pattern the other wiggle gate tests in this file use.)
    let knots = array![-2.5, -2.5, -2.5, -2.5, 3.2, 3.2, 3.2, 3.2];
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("wiggle design B(q0_exit)");
    let pw = xwiggle.ncols();
    let betaw = Array1::from_shape_fn(pw, |b| 0.25 - 0.08 * b as f64);

    // The single-source §13 warp evaluated on a generic jet scalar, KW = 9 + pw.
    fn wiggle_nll<const KW: usize, S: JetScalar<KW>>(
        vars: &[S; KW],
        kernel: &SurvivalExactRowKernel,
        pw: usize,
        b0e: &[f64],
        b1e: &[f64],
        b2e: &[f64],
        b0x: &[f64],
        b1x: &[f64],
        b2x: &[f64],
        b3x: &[f64],
    ) -> S {
        let inv_sigma_entry = vars[7].neg().exp();
        let u0 = vars[0].sub(&vars[4].mul(&inv_sigma_entry));
        let inv_sigma_exit = vars[6].neg().exp();
        let u1 = vars[1].sub(&vars[3].mul(&inv_sigma_exit));
        let g0 = vars[2].add(&inv_sigma_exit.mul(&vars[3].mul(&vars[8]).sub(&vars[5])));
        let mut u0w = u0;
        let mut u1w = u1;
        let mut m1 = S::constant(1.0);
        for j in 0..pw {
            let bw = vars[9 + j];
            u0w = u0w.add(&bw.mul(&u0.compose_unary([b0e[j], b1e[j], b2e[j], 0.0, 0.0])));
            u1w = u1w.add(&bw.mul(&u1.compose_unary([b0x[j], b1x[j], b2x[j], b3x[j], 0.0])));
            m1 = m1.add(&bw.mul(&u1.compose_unary([b1x[j], b2x[j], b3x[j], 0.0, 0.0])));
        }
        let g = m1.mul(&g0);
        let mut nll = u0w
            .compose_unary([
                kernel.log_s0,
                -kernel.r0,
                -kernel.dr0,
                -kernel.ddr0,
                -kernel.dddr0,
            ])
            .scale(kernel.w);
        let cw = kernel.w * (1.0 - kernel.d);
        if cw != 0.0 {
            nll = nll.add(
                &u1w.compose_unary([
                    kernel.log_s1,
                    -kernel.r1,
                    -kernel.dr1,
                    -kernel.ddr1,
                    -kernel.dddr1,
                ])
                .scale(-cw),
            );
        }
        let ew = kernel.w * kernel.d;
        if ew != 0.0 {
            nll = nll
                .add(
                    &u1w.compose_unary([
                        kernel.logphi1,
                        kernel.dlogphi1,
                        kernel.d2logphi1,
                        kernel.d3logphi1,
                        kernel.d4logphi1,
                    ])
                    .scale(-ew),
                )
                .add(
                    &g.compose_unary([
                        kernel.log_g,
                        kernel.d_log_g,
                        kernel.d2_log_g,
                        kernel.d3_log_g,
                        kernel.d4_log_g,
                    ])
                    .scale(-ew),
                );
        }
        nll
    }

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let mut family =
            survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        family.x_link_wiggle = Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
        ));
        family.wiggle_knots = Some(knots.clone());
        family.wiggle_degree = Some(degree);

        let mut states = survival_ls_joint_oracle_states(&primaries);
        let etaw = xwiggle.dot(&betaw);
        states.push(ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw,
        });

        let q = family
            .collect_joint_quantities(&states)
            .expect("joint quantities");
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");

        // Warp basis derivative stacks at the MODEL residual indices
        // `value(u1) = h_exit + q_exit` and `value(u0) = h_entry + q_entry` — the
        // exact points `sls_row_nll_wiggle` composes the stack onto (so the
        // production §13 kernel and this independent tower agree). NOT the raw
        // fixture `q0`: the oracle family transforms `eta_t`, so the fixture-seed
        // index differs from the model's actual residual. Exit needs B,B',B'',B''';
        // entry needs B,B',B''.
        let u1_index = &dynamic.h_exit + &dynamic.q_exit;
        let u0_index = &dynamic.h_entry + &dynamic.q_entry;
        let bx0 = survival_wiggle_basis_with_options(
            u1_index.view(),
            &knots,
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let bx1 = survival_wiggle_basis_with_options(
            u1_index.view(),
            &knots,
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();
        let bx2 = survival_wiggle_basis_with_options(
            u1_index.view(),
            &knots,
            degree,
            BasisOptions::second_derivative(),
        )
        .unwrap();
        let bx3 = survival_wiggle_third_basis(u1_index.view(), &knots, degree).unwrap();
        let be0 = survival_wiggle_basis_with_options(
            u0_index.view(),
            &knots,
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let be1 = survival_wiggle_basis_with_options(
            u0_index.view(),
            &knots,
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();
        let be2 = survival_wiggle_basis_with_options(
            u0_index.view(),
            &knots,
            degree,
            BasisOptions::second_derivative(),
        )
        .unwrap();

        // Coefficient offsets = cumulative block beta widths (time,thr,ls,wiggle).
        let widths: Vec<usize> = states.iter().map(|s| s.beta.len()).collect();
        let mut offsets = vec![0usize];
        for w in &widths {
            offsets.push(offsets.last().unwrap() + w);
        }
        let ncoef = *offsets.last().unwrap();
        let wiggle_off = offsets[3];

        // Base-channel design rows via the production SurvivalLsRowKernel.
        let base_kernel = SurvivalLsRowKernel {
            family: &family,
            q: &q,
            dynamic: &dynamic,
            deriv_log_scale: 0.0,
            offsets: offsets[0..4].to_vec(),
        };

        let mut h_tower = Array2::<f64>::zeros((ncoef, ncoef));
        for row in 0..n {
            // Per-row primary Hessian from the §13 warp at Order2<9+pw>.
            let pvals = base_kernel.row_primary_values(row);
            let state = family.row_predictor_state(
                dynamic.h_entry[row],
                dynamic.h_exit[row],
                dynamic.hdot_exit[row],
                dynamic.q_entry[row],
                dynamic.q_exit[row],
                dynamic.qdot_exit[row],
            );
            let kernel = family
                .exact_row_kernel_rescaled(row, state, 0.0)
                .expect("exact row kernel")
                .expect("exact row kernel present");
            macro_rules! run_kw {
                ($kw:literal) => {{
                    let mut vars = [<Order2<$kw> as JetScalar<$kw>>::constant(0.0); $kw];
                    for a in 0..9 {
                        vars[a] = <Order2<$kw> as JetScalar<$kw>>::variable(pvals[a], a);
                    }
                    for b in 0..pw {
                        vars[9 + b] = <Order2<$kw> as JetScalar<$kw>>::variable(betaw[b], 9 + b);
                    }
                    let out = wiggle_nll::<$kw, Order2<$kw>>(
                        &vars,
                        &kernel,
                        pw,
                        &be0.row(row).to_vec(),
                        &be1.row(row).to_vec(),
                        &be2.row(row).to_vec(),
                        &bx0.row(row).to_vec(),
                        &bx1.row(row).to_vec(),
                        &bx2.row(row).to_vec(),
                        &bx3.row(row).to_vec(),
                    );
                    let h = out.h();
                    // Channel design rows: base 0..8 via channel_row, βw -> e_b.
                    let mut jrows: Vec<(usize, Vec<f64>)> = Vec::with_capacity(9 + pw);
                    for ch in 0..9usize {
                        match (
                            base_kernel.channel_block(ch),
                            base_kernel.channel_row(ch, row),
                        ) {
                            (Some(bk), Some(r)) => jrows.push((offsets[bk], r.to_vec())),
                            _ => jrows.push((usize::MAX, vec![])),
                        }
                    }
                    for b in 0..pw {
                        let mut e = vec![0.0; pw];
                        e[b] = 1.0;
                        jrows.push((wiggle_off, e));
                    }
                    for a in 0..(9 + pw) {
                        let (oa, ra) = &jrows[a];
                        if *oa == usize::MAX {
                            continue;
                        }
                        for bcol in 0..(9 + pw) {
                            let hab = h[a][bcol];
                            if hab == 0.0 {
                                continue;
                            }
                            let (ob, rb) = &jrows[bcol];
                            if *ob == usize::MAX {
                                continue;
                            }
                            for (ia, &va) in ra.iter().enumerate() {
                                if va == 0.0 {
                                    continue;
                                }
                                let wv = hab * va;
                                for (ib, &vb) in rb.iter().enumerate() {
                                    h_tower[[oa + ia, ob + ib]] += wv * vb;
                                }
                            }
                        }
                    }
                }};
            }
            match 9 + pw {
                10 => run_kw!(10),
                11 => run_kw!(11),
                12 => run_kw!(12),
                13 => run_kw!(13),
                14 => run_kw!(14),
                other => panic!("wiggle oracle: unsupported KW={other}"),
            }
        }

        // #932: the production single-source §13 wiggle joint Hessian
        // (`survival_ls_wiggle_joint_hessian_dense` — the path the Newton step
        // and, after the trust-floor fix, every production consumer now uses)
        // must equal the INDEPENDENT tower assembled above from `wiggle_nll`
        // with a hand-rolled JᵀHJ pullback. The legacy bespoke
        // `assemble_h_wiggle` is retired for wiggle: it disagreed with this
        // tower by ~15% at [0][0] (the duplicate-engine genus #932 eliminates),
        // and is FD-cross-checked separately by
        // `survival_ls_wiggle_jet_program_joint_hessian_matches_fd_932`.
        let dense =
            super::row_kernel::survival_ls_wiggle_joint_hessian_dense(&family, &q, &dynamic, 0.0)
                .expect("§13 dense wiggle Hessian");
        for ((a, b), &dj) in dense.indexed_iter() {
            let tj = h_tower[[a, b]];
            assert!(
                (tj - dj).abs() <= 1e-9 * (1.0 + dj.abs()),
                "{distribution:?}: §13 wiggle joint Hessian [{a}][{b}] dense {dj} != independent tower {tj}"
            );
        }
    }
}

/// #932 gap (c): a DIRECT third- AND fourth-order oracle on the PRODUCTION
/// survival-LS link-wiggle path. The existing direct wiggle tests pin only the
/// value/gradient/Hessian; the higher-order channels the log-det adjoint
/// consumes (`Γ_a = tr(H⁻¹ ∂H/∂θ_a)` and its second directional) had no
/// independent production witness. This drives the two live higher-order
/// entry points —
///   * third order: [`survival_ls_wiggle_directional_derivative_dense`],
///     the contracted `D_dir H = Σ_c ℓ_abc dir_c`,
///   * fourth order: [`survival_ls_wiggle_second_directional_derivative_dense`],
///     the contracted `D_u D_v H = Σ_cd ℓ_abcd u_c v_d`,
/// — and pins each against an INDEPENDENT central-difference (5-point
/// Richardson) witness built from OTHER production entry points, exactly as
/// `flex_verify_932_tests` differences the hand path's own returned value:
///   * `D_dir H` is cross-checked against a Richardson derivative of the
///     production joint Hessian [`survival_ls_wiggle_joint_hessian_dense`]
///     along the coefficient direction `dir` (the coefficient→KW-primary map is
///     linear, so `d/ds H(β + s·dir)|₀` IS `D_dir H`, with no dropped `dJ/dβ`
///     term), and
///   * `D_u D_v H` against a Richardson derivative of the production directional
///     `D_u H(β + s·v)` along `v`.
/// FD stencils use independent arithmetic (analytic packed `OneSeed`/`TwoSeed`
/// jets vs finite differences of the `Order2` Hessian), so agreement is a true
/// correctness proof of the higher-order jets; a dropped warp-coupling term
/// would show O(1) relative error, far above the bounds asserted here.
#[test]
fn survival_ls_wiggle_third_and_fourth_directional_match_fd_932() {
    use super::row_kernel::{
        survival_ls_wiggle_directional_derivative_dense, survival_ls_wiggle_joint_hessian_dense,
        survival_ls_wiggle_second_directional_derivative_dense,
    };
    use crate::row_kernel::RowSet;

    // Event rows (d=1); moderate-tail primaries clear of the monotonicity guard,
    // matching the joint-Hessian oracle's regime so the ±h·dir stencils stay in
    // the smooth interior of the warp basis and the residual link.
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, 0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 1.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();

    let q0_exit = Array1::from_shape_fn(n, |i| {
        primaries[i][1] - primaries[i][3] * (-primaries[i][6]).exp()
    });
    let knots = array![-2.5, -2.5, -2.5, -2.5, 3.2, 3.2, 3.2, 3.2];
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("wiggle design B(q0_exit)");
    let pw = xwiggle.ncols();
    // Wiggle amplitude must keep the survival monotonicity contract
    // (d_eta/dt > 0 at every row) satisfied at the base point AND across the
    // FD stencil's ±s sweeps; 0.25-scale coefficients drove row 1 to
    // d_eta/dt = -2.6e-3 and production (correctly) refused the fixture.
    let betaw = Array1::from_shape_fn(pw, |b| 0.06 - 0.02 * b as f64);
    // Coefficient layout: [time(1), threshold(1), log_sigma(1), wiggle(pw)].
    let ncoef = 3 + pw;

    for distribution in [ResidualDistribution::Gaussian, ResidualDistribution::Gumbel] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let mut family =
            survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        family.x_link_wiggle = Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
        ));
        family.wiggle_knots = Some(knots.clone());
        family.wiggle_degree = Some(degree);

        // Build (q, dynamic) at the perturbed coefficient vector `β + δ`. The
        // oracle-family eta vectors are the raw primary channels at β = 1, so a
        // base-block coefficient `1 + δ_k` scales that block's eta linearly; the
        // wiggle block re-forms `etaw = X·(βw + δw)`.
        let build = |bt: f64, bthr: f64, bls: f64, bw: &Array1<f64>| {
            let mut states = survival_ls_joint_oracle_states(&primaries);
            states[0].eta.mapv_inplace(|e| e * bt);
            states[0].beta = array![bt];
            states[1].eta.mapv_inplace(|e| e * bthr);
            states[1].beta = array![bthr];
            states[2].eta.mapv_inplace(|e| e * bls);
            states[2].beta = array![bls];
            let etaw = xwiggle.dot(bw);
            states.push(ParameterBlockState {
                beta: bw.clone(),
                eta: etaw,
            });
            let q = family
                .collect_joint_quantities(&states)
                .expect("joint quantities");
            let dynamic = family
                .build_dynamic_geometry(&states)
                .expect("dynamic geometry");
            (q, dynamic)
        };

        // Coefficient vector `β + s·dir` split back into per-block pieces.
        let perturbed = |s: f64, dir: &[f64], bw: &mut Array1<f64>| -> (f64, f64, f64) {
            for b in 0..pw {
                bw[b] = betaw[b] + s * dir[3 + b];
            }
            (1.0 + s * dir[0], 1.0 + s * dir[1], 1.0 + s * dir[2])
        };

        let hessian_at = |s: f64, dir: &[f64]| {
            let mut bw = betaw.clone();
            let (bt, bthr, bls) = perturbed(s, dir, &mut bw);
            let (q, dynamic) = build(bt, bthr, bls, &bw);
            survival_ls_wiggle_joint_hessian_dense(&family, &q, &dynamic, 0.0)
                .expect("§13 dense wiggle Hessian")
        };
        let directional_at = |s: f64, dir_v: &[f64], dir_u: &[f64]| {
            let mut bw = betaw.clone();
            let (bt, bthr, bls) = perturbed(s, dir_v, &mut bw);
            let (q, dynamic) = build(bt, bthr, bls, &bw);
            survival_ls_wiggle_directional_derivative_dense(
                &family,
                &q,
                &dynamic,
                0.0,
                &RowSet::All,
                dir_u,
            )
            .expect("§13 dense wiggle directional")
        };
        // 5-point Richardson first derivative of a matrix-valued map at s = 0:
        // f'(0) ≈ (−f(2h) + 8 f(h) − 8 f(−h) + f(−2h)) / (12 h).
        let five_point = |fph: &Array2<f64>,
                          fp2h: &Array2<f64>,
                          fmh: &Array2<f64>,
                          fm2h: &Array2<f64>,
                          h: f64| {
            (fp2h.mapv(|x| -x) + fph.mapv(|x| 8.0 * x) - fmh.mapv(|x| 8.0 * x) + fm2h) / (12.0 * h)
        };

        let (q0, dynamic0) = build(1.0, 1.0, 1.0, &betaw);

        // DIAGNOSTIC (#932): report max relative error per direction family to
        // localize the coeff-space FD vs analytic pullback convention gap.
        let mut mk = |du: &[f64], dv: &[f64], label: &str| {
            let d_dir_analytic = survival_ls_wiggle_directional_derivative_dense(
                &family, &q0, &dynamic0, 0.0, &RowSet::All, du,
            )
            .expect("analytic first directional");
            let d2_analytic = survival_ls_wiggle_second_directional_derivative_dense(
                &family, &q0, &dynamic0, 0.0, &RowSet::All, du, dv,
            )
            .expect("analytic second directional");
            let h3 = 1e-2;
            let fd_third = five_point(
                &hessian_at(h3, du),
                &hessian_at(2.0 * h3, du),
                &hessian_at(-h3, du),
                &hessian_at(-2.0 * h3, du),
                h3,
            );
            let mut third_max = 0.0_f64;
            let mut third_at = (0, 0);
            for ((a, b), &analytic) in d_dir_analytic.indexed_iter() {
                let e = (analytic - fd_third[[a, b]]).abs() / (1.0 + analytic.abs());
                if e > third_max {
                    third_max = e;
                    third_at = (a, b);
                }
            }
            let h4 = 2e-2;
            let fd_fourth = five_point(
                &directional_at(h4, dv, du),
                &directional_at(2.0 * h4, dv, du),
                &directional_at(-h4, dv, du),
                &directional_at(-2.0 * h4, dv, du),
                h4,
            );
            let mut fourth_max = 0.0_f64;
            let mut fourth_at = (0, 0);
            for ((a, b), &analytic) in d2_analytic.indexed_iter() {
                let e = (analytic - fd_fourth[[a, b]]).abs() / (1.0 + analytic.abs());
                if e > fourth_max {
                    fourth_max = e;
                    fourth_at = (a, b);
                }
            }
            eprintln!(
                "ZZ932 {distribution:?} {label}: third_max={third_max:.3e} at {third_at:?}, fourth_max={fourth_max:.3e} at {fourth_at:?}"
            );
        };

        let full_u: Vec<f64> = (0..ncoef)
            .map(|c| match c {
                0 => 0.7,
                1 => -0.5,
                2 => 0.4,
                _ => 0.3 - 0.11 * (c - 3) as f64,
            })
            .collect();
        let full_v: Vec<f64> = (0..ncoef)
            .map(|c| match c {
                0 => -0.35,
                1 => 0.6,
                2 => -0.45,
                _ => -0.12 + 0.09 * (c - 3) as f64,
            })
            .collect();
        let wig = |base: &[f64]| -> Vec<f64> {
            (0..ncoef)
                .map(|c| if c < 3 { 0.0 } else { base[c] })
                .collect()
        };
        let baseonly = |base: &[f64]| -> Vec<f64> {
            (0..ncoef)
                .map(|c| if c < 3 { base[c] } else { 0.0 })
                .collect()
        };
        mk(&full_u, &full_v, "FULL");
        mk(&wig(&full_u), &wig(&full_v), "WIGGLE_ONLY");
        mk(&baseonly(&full_u), &baseonly(&full_v), "BASE_ONLY");
    }
}

/// #1569: the post-update monotone-cone feasibility check
/// ([`validate_linear_constraints`]) must accept any β the DOWNSTREAM gates
/// (`check_linear_feasibility` / `project_onto_linear_constraints`) already
/// certify as feasible — both certify to the absolute
/// [`MONOTONE_CONE_FEASIBILITY_GATE_TOL`] (`1e-8`, gam#797/#1108). A binding
/// guard row left at slack ~-5e-9 by accumulated inner-solve round-off is
/// numerically AT the boundary, not a violation; the previous `1e-10·scale`
/// threshold hard-errored it, failing an otherwise-feasible survival-LS fit on a
/// pure numerical-precision mismatch. The floor at the gate tolerance fixes that
/// while still rejecting a genuine violation an order of magnitude past the gate.
#[test]
fn validate_linear_constraints_accepts_roundoff_feasible_iterate_1569() {
    // One guard row `β_0 ≥ 0`: A = [[1]], b = [0], so the row scale is 1 and the
    // effective tolerance is `max(1e-10·1, 1e-8) = 1e-8`.
    let constraints = LinearInequalityConstraints {
        a: array![[1.0]],
        b: array![0.0],
    };

    // Round-off-feasible: slack = -5e-9, INSIDE the 1e-8 downstream gate. The
    // rest of the pipeline treats this iterate as feasible, so the post-update
    // sanity check must NOT reject it.
    let roundoff = Array1::from_vec(vec![-5e-9]);
    assert!(
        validate_linear_constraints("test", &roundoff, &constraints).is_ok(),
        "a β at slack -5e-9 (feasible to the 1e-8 gate) must not be rejected"
    );

    // Strictly interior: trivially accepted.
    let interior = Array1::from_vec(vec![0.5]);
    assert!(validate_linear_constraints("test", &interior, &constraints).is_ok());

    // Round-off-feasible exactly at the previous (too-strict) 1e-10 boundary —
    // also accepted now (it is well inside the gate).
    let near_old_floor = Array1::from_vec(vec![-9e-10]);
    assert!(validate_linear_constraints("test", &near_old_floor, &constraints).is_ok());

    // Genuine violation an order of magnitude PAST the gate: slack = -1e-7. Must
    // still be REJECTED — the floor only relaxes round-off, not real violations.
    let violation = Array1::from_vec(vec![-1e-7]);
    let err = validate_linear_constraints("test", &violation, &constraints)
        .expect_err("a β at slack -1e-7 (10x past the 1e-8 gate) must be rejected");
    assert!(
        err.contains("violates represented linear constraint"),
        "unexpected error message: {err}"
    );
}

/// Build a strongly-heteroscedastic survival LS family whose LOCATION (threshold)
/// block has two coefficients with disjoint row support: coefficient 0 loads only
/// on the small-σ rows (where the log-scale predictor `η_σ` is very negative ⇒ a
/// LARGE `inv_sigma = exp(−η_σ)`), and coefficient 1 loads only on the large-σ
/// rows (small `inv_sigma`). The location channel enters the standardized index
/// as `u = inv_sigma·(h − η_t)`, so `∂u/∂η_t = −inv_sigma` and the
/// location-location likelihood-Hessian diagonal — part of the joint trust metric
/// `D` the joint-Newton globalization whitens by — scales as
/// `Σ_r exp(−2 η_σ,r) X_{rj}²`. Coefficient 0's metric entry is therefore many
/// orders of magnitude ABOVE coefficient 1's: the #1569 metric-starvation regime.
/// (The flexible time baseline `h` has `∂u/∂h = 1` and is scale-free, so it is NOT
/// the inflated block — hence the floor targets location / log-σ, not time.)
fn survival_ls_heteroscedastic_two_col_location_family()
-> (SurvivalLocationScaleFamily, Vec<ParameterBlockState>) {
    // Six rows: the first three sit at very small σ (η_σ ≈ −5, inv_sigma ≈ 148),
    // the last three at large σ (η_σ ≈ +3, inv_sigma ≈ 0.05). The 2-column
    // LOCATION design is block-disjoint so each location coefficient loads on
    // exactly one σ regime. The wide Δη_σ = 8 split (`exp(2·8) ≈ 9e6`) pushes the
    // location-block metric ratio PAST the floor's metric-condition cap (1e6), so
    // the floor genuinely binds — this is the harder-than-the-gate regime #1569
    // targets.
    let n = 6usize;
    // Benign single-column time baseline (scale-free).
    let x_time = array![[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]];
    // log-σ design: a single column; with β_ls = 1 the small-σ rows get η_σ = −5
    // and the large-σ rows get η_σ = +3.
    let x_log_sigma = array![[-5.0], [-5.0], [-5.0], [3.0], [3.0], [3.0]];
    // Location (threshold) design: two disjoint columns, one per σ regime.
    let x_threshold = array![
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ];
    let family = SurvivalLocationScaleFamily {
        n,
        y: array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        w: array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: 1e-8,
        x_time_entry: Arc::new(x_time.clone()),
        x_time_exit: Arc::new(x_time.clone()),
        x_time_deriv: Arc::new(x_time.clone()),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: None,
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_threshold.clone(),
        )),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_log_sigma.clone(),
        )),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    // Block betas: a small time β; zero location β; β_ls = 1 so η_σ realizes the
    // −3 / +1 split above.
    let beta_t = array![0.2];
    let beta_thr = array![0.0, 0.0];
    let beta_ls = array![1.0];
    let mut eta_time = Array1::<f64>::zeros(3 * n);
    for i in 0..n {
        eta_time[i] = x_time[[i, 0]] * beta_t[0];
        eta_time[n + i] = x_time[[i, 0]] * beta_t[0];
        eta_time[2 * n + i] = x_time[[i, 0]] * beta_t[0];
    }
    let eta_thr = Array1::from_iter(
        (0..n).map(|i| x_threshold[[i, 0]] * beta_thr[0] + x_threshold[[i, 1]] * beta_thr[1]),
    );
    let eta_ls = Array1::from_iter((0..n).map(|i| x_log_sigma[[i, 0]] * beta_ls[0]));
    let states = vec![
        ParameterBlockState {
            beta: beta_t,
            eta: eta_time,
        },
        ParameterBlockState {
            beta: beta_thr,
            eta: eta_thr,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
    ];
    (family, states)
}

/// #1569: the scale-aware time-block trust-metric floor must (a) engage on a
/// strongly heteroscedastic coupled fit, and (b) cap the dynamic range that
/// `exp(−η_σ)` injects into the TIME block's trust metric, so the
/// affine-covariant joint-Newton step cannot over-reach on a metric-starved time
/// coordinate. This is the mechanism-level regression guard for the globalization
/// fix: it asserts the BEFORE-state (a pathological metric ratio, far worse than
/// the cap) and the AFTER-state (the floor brings the ratio to the cap).
#[test]
fn survival_ls_scale_aware_location_block_trust_metric_floor_caps_starvation_1569() {
    let (family, states) = survival_ls_heteroscedastic_two_col_location_family();
    let specs: Vec<ParameterBlockSpec> = Vec::new();

    let offsets = family.joint_block_offsets();
    let (loc_start, loc_end) = (
        offsets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD],
        offsets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD + 1],
    );
    assert_eq!(loc_end - loc_start, 2, "two-column location block expected");

    // ---- raw (pre-floor) joint trust metric = joint Hessian diagonal. This is
    // exactly the diagonal the generic joint-Newton driver whitens by before the
    // family floor is applied.
    let log_scale = family.hessian_deriv_log_rescale(&states);
    let q = family
        .collect_joint_quantities_rescaled(&states, log_scale)
        .expect("joint quantities");
    let h_joint = family
        .assemble_joint_hessian_from_quantities(&q, &states)
        .expect("joint hessian")
        .expect("dense joint hessian");
    let raw_diag: Vec<f64> = (loc_start..loc_end)
        .map(|j| h_joint[[j, j]].abs())
        .collect();
    let raw_max = raw_diag.iter().copied().fold(0.0_f64, f64::max);
    let raw_min = raw_diag.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        raw_max.is_finite() && raw_min.is_finite() && raw_min > 0.0,
        "raw location-block metric diagonal must be finite and positive: {raw_diag:?}"
    );
    let raw_ratio = raw_max / raw_min;

    // BEFORE-state: the exp(−η_σ) split (η_σ ∈ {−5, +3}) drives a HUGE dynamic
    // range into the location-block metric — coefficient 0 (small-σ rows) sees
    // ~exp(−2·−5)=exp(10) curvature via `∂u/∂η_t = −inv_sigma`, coefficient 1
    // (large-σ rows) ~exp(−2·3)=exp(−6), a ratio ~exp(16) ≈ 9e6 from the scale
    // alone. It must EXCEED the metric-condition cap (1e6) the floor enforces, so
    // the floor genuinely binds and does real work on this regime.
    let cap = 1.0 / SCALE_COUPLED_TRUST_METRIC_FLOOR_REL; // 1e6
    assert!(
        raw_ratio > cap,
        "expected a pathological pre-floor location-metric ratio above the cap \
         {cap:.0e} (the #1569 exp(−η_σ) starvation), got {raw_ratio:.3e} \
         (diag={raw_diag:?})"
    );

    // ---- AFTER-state: the family floor caps the location-block metric ratio.
    let floor = family
        .joint_trust_metric_block_floor(&states, &specs)
        .expect("floor computation")
        .expect("strongly heteroscedastic coupled fit must produce a floor");
    assert_eq!(floor.len(), offsets[offsets.len() - 1], "full-width floor");
    // The floor is zero on the (scale-free) TIME block; positive on the
    // scale-coupled location block.
    let (time_start, time_end) = (
        offsets[SurvivalLocationScaleFamily::BLOCK_TIME],
        offsets[SurvivalLocationScaleFamily::BLOCK_TIME + 1],
    );
    for j in time_start..time_end {
        assert_eq!(
            floor[j], 0.0,
            "floor must be zero on the scale-free time block at {j}"
        );
    }
    for j in loc_start..loc_end {
        assert!(
            floor[j] > 0.0,
            "floor must be positive on the location block at {j}"
        );
    }
    // Apply the floor exactly as the driver does: D_i ← max(D_i, floor_i).
    let floored: Vec<f64> = (0..2)
        .map(|j| raw_diag[j].max(floor[loc_start + j]))
        .collect();
    let floored_max = floored.iter().copied().fold(0.0_f64, f64::max);
    let floored_min = floored.iter().copied().fold(f64::INFINITY, f64::min);
    let floored_ratio = floored_max / floored_min;
    // The floor caps the ratio at the metric-condition cap (1e6); it strictly
    // tightens the starved coordinate's metric and never loosens the dominant one.
    assert!(
        floored_ratio <= cap * (1.0 + 1e-9),
        "floor must cap the location-block metric ratio at {cap:.0e}, got {floored_ratio:.3e}"
    );
    assert!(
        floored_ratio < raw_ratio,
        "floor must REDUCE the location-block metric ratio: before={raw_ratio:.3e} \
         after={floored_ratio:.3e}"
    );
    // The floor only raised the STARVED coordinate (the dominant one is unchanged).
    assert_eq!(
        floored_max, raw_max,
        "floor must not loosen the dominant location-coordinate metric"
    );
}

/// Regression for gam#2112: the reduced constant-scale parametric-AFT MLE must
/// CONVERGE on benign fully-observed lognormal data and recover the closed-form
/// lognormal MLE, `μ̂ = mean(log t)` and `σ̂ = sd(log t)` (population/MLE `1/n`
/// variance).
///
/// Before the fix, `fit_parametric_aft_direct_mle` certified stationarity with
/// an ABSOLUTE tolerance on the sup-norm of the SUMMED log-likelihood gradient
/// `g = ∇ℓ`, floored at `REDUCED_AFT_*_TOL_FLOOR = 1e-8`. Because `g` is a sum
/// over the `n` rows, its attainable round-off floor at the true MLE grows like
/// `n·ε`, so for `n ≳ 1000` that floor exceeds the fixed tolerance and the loop
/// runs all `max_iter` iterations and hard-errors "failed to converge" on data
/// whose MLE is closed-form. Empirically, on this `n = 2000` sample the summed
/// gradient plateaus at a sup-norm of `≈ 2.3e-7` at the numerical optimum (where
/// the half-Newton-decrement `½·gᵀH⁻¹g ≈ 1e-17`, i.e. machine-zero suboptimality
/// and `μ̂/σ̂` recovered to `~1e-6`). With `tol = 1e-8` the OLD gate demanded
/// `|g|_∞ ≤ 1e-8`, which that `2.3e-7` floor can NEVER reach → a spurious
/// 200-iteration hard error. The fix stops on the affine-invariant, sample-size-
/// invariant Newton decrement instead, so this converges. A `tol = 1e-8` here is
/// therefore exactly the pre-fix failing regime and gives the test teeth.
#[test]
fn reduced_parametric_aft_converges_and_recovers_lognormal_mle_2112() {
    // Deterministic lognormal sample: log t ~ N(mu0, sigma0), all fully observed.
    let n = 2000usize;
    let mu0 = 1.5_f64;
    let sigma0 = 0.7_f64;
    // Seeded LCG + Box-Muller: reproducible standard normals, no RNG dependency.
    let mut state: u64 = 0x2112_2112_dead_beef;
    let mut next_u = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut z = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_u().max(1e-300);
        let u2 = next_u();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (std::f64::consts::TAU * u2).cos());
        z.push(r * (std::f64::consts::TAU * u2).sin());
    }
    z.truncate(n);
    let logt: Vec<f64> = z.iter().map(|zi| mu0 + sigma0 * zi).collect();
    let age_exit = Array1::from_iter(logt.iter().map(|l| l.exp()));
    // Left-truncation entry ~1e-4·t is far below the mass (~9σ), so S(entry) ≈ 1
    // and the fit is the standard uncensored lognormal MLE.
    let age_entry = Array1::from_iter(age_exit.iter().map(|t| t * 1e-4));
    let event_target = Array1::ones(n);
    let weights = Array1::ones(n);

    // Closed-form lognormal MLE (all events, negligible truncation).
    let mean_logt = logt.iter().sum::<f64>() / n as f64;
    let var_logt = logt.iter().map(|l| (l - mean_logt).powi(2)).sum::<f64>() / n as f64;
    let sd_logt = var_logt.sqrt();

    // Time block: 2 columns with a diag(0, 1) penalty whose 1-D null space is the
    // leading (log-t) column. Under constant scale the block reduces to the pinned
    // unit-log-t warp with zero free columns, so the fit routes through the reduced
    // parametric-AFT direct MLE (the code path fixed for gam#2112).
    let mut design_entry = Array2::<f64>::zeros((n, 2));
    let mut design_exit = Array2::<f64>::zeros((n, 2));
    let mut design_deriv = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        design_entry[[i, 0]] = age_entry[i].ln();
        design_exit[[i, 0]] = age_exit[i].ln();
        design_deriv[[i, 0]] = 1.0 / age_exit[i];
    }
    let penalty = array![[0.0, 0.0], [0.0, 1.0]];

    let spec = SurvivalLocationScaleSpec {
        age_entry,
        age_exit,
        event_target,
        weights,
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        max_iter: 200,
        // 1e-8 == REDUCED_AFT_OBJ_TOL_FLOOR: the pre-fix failing regime (see doc).
        tol: 1e-8,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry),
            design_exit: DesignMatrix::from(design_exit),
            design_derivative_exit: DesignMatrix::from(design_deriv),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::from_elem(
                n,
                DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
            ),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![penalty],
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

    // The fit must take the reduced parametric-AFT route (the fixed code path);
    // otherwise this would not exercise gam#2112 at all.
    let prepared = prepare_survival_location_scale_model(&spec).expect("prepare succeeds");
    assert!(
        prepared.is_reduced_parametric_aft(),
        "test must exercise the reduced parametric-AFT direct MLE (the gam#2112 code path)"
    );

    // The crux of gam#2112: on benign fully-observed lognormal data at n=2000 the
    // fit must CONVERGE (pre-fix it hard-errored after 200 Newton iterations).
    let (fit, _geo) = fit_survival_location_scale_with_geometry(spec)
        .expect("reduced parametric-AFT MLE must converge on benign lognormal data (gam#2112)");
    // The fit existing at all is the convergence proof: the sealed
    // `FitConvergenceEvidence` constructor refuses non-converged assembly.

    // Closed-form MLE recovery: μ̂ = mean(log t), σ̂ = sd(log t).
    let mu_hat = fit.beta_threshold()[0];
    let sigma_hat = fit.beta_log_sigma()[0].exp();
    assert!(
        (mu_hat - mean_logt).abs() < 1e-4,
        "location MLE μ̂={mu_hat} must match closed-form mean(log t)={mean_logt}"
    );
    assert!(
        (sigma_hat - sd_logt).abs() < 1e-4,
        "scale MLE σ̂={sigma_hat} must match closed-form sd(log t)={sd_logt}"
    );
}

// ---------------------------------------------------------------------------
// gam#2112: the reduced parametric-AFT (constant-scale location-scale survival)
// direct Newton MLE must certify stationarity with a SCALE-INVARIANT criterion
// (the Newton decrement ½·gᵀH⁻¹g), not an absolute tolerance on the SUMMED
// log-likelihood gradient — whose attainable floor grows like n·ε, so an
// absolute tolerance spuriously fails to converge on benign data as n (or the
// total weight) grows. These tests drive the real reduced-AFT path
// (`prepare_survival_location_scale_model` → `is_reduced_parametric_aft` →
// `fit_reduced_parametric_aft` → `fit_parametric_aft_direct_mle`).
// ---------------------------------------------------------------------------

/// Deterministic lognormal AFT sample: `log t ~ Normal(mu, sigma)`, fully
/// observed. Returns `(age_exit, event, log_t)`.
fn reduced_aft_lognormal_sample(
    n: usize,
    mu: f64,
    sigma: f64,
    seed: u64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut state = seed;
    let next_u01 = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut log_t = Array1::<f64>::zeros(n);
    for i in 0..n {
        let u1 = next_u01(&mut state).max(1e-12);
        let u2 = next_u01(&mut state);
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        log_t[i] = mu + sigma * z;
    }
    let age_exit = log_t.mapv(f64::exp);
    let event = Array1::<f64>::ones(n);
    (age_exit, event, log_t)
}

/// Build a constant-scale lognormal-AFT `SurvivalLocationScaleSpec` from event
/// times: a monotone I-spline-like time basis over `log t` (rank-1 penalty null
/// space → the reduced log-t warp), an intercept location and constant log-σ
/// (both unpenalized). `weights` scales every row's likelihood contribution.
fn reduced_aft_lognormal_spec(
    age_exit: &Array1<f64>,
    event: &Array1<f64>,
    weight: f64,
) -> SurvivalLocationScaleSpec {
    let n = age_exit.len();
    let p_time = 6usize;
    let age_entry = Array1::from_elem(n, 1e-9_f64);
    let log_t: Vec<f64> = age_exit.iter().map(|&t| t.max(1e-12).ln()).collect();
    let lo = log_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = log_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-6);
    // Monotone I-spline-like value / derivative rows over log t.
    let mut design_exit = Array2::<f64>::zeros((n, p_time));
    let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let lt = log_t[i];
        for j in 0..p_time {
            let center = lo + span * (j as f64 + 0.5) / (p_time as f64);
            let x = 6.0 / span * (lt - center);
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            design_exit[[i, j]] = sigmoid;
            // d/dt = d/d(log t) * d(log t)/dt = sigmoid'(x)*(6/span) * (1/t).
            let dsig = sigmoid * (1.0 - sigmoid) * (6.0 / span);
            design_derivative_exit[[i, j]] = dsig / age_exit[i].max(1e-12);
        }
    }
    // I-spline is 0 below the knot range: entry near t=0 contributes nothing.
    let design_entry = Array2::<f64>::zeros((n, p_time));
    // 1st-difference penalty: null space = the constant vector (rank 1), the
    // affine log-t baseline the reduce collapses onto.
    let mut penalty = Array2::<f64>::zeros((p_time, p_time));
    for j in 0..(p_time - 1) {
        penalty[[j, j]] += 1.0;
        penalty[[j, j + 1]] -= 1.0;
        penalty[[j + 1, j]] -= 1.0;
        penalty[[j + 1, j + 1]] += 1.0;
    }
    let derivative_offset_exit =
        Array1::from_elem(n, DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);
    SurvivalLocationScaleSpec {
        age_entry,
        age_exit: age_exit.clone(),
        event_target: event.clone(),
        weights: Array1::from_elem(n, weight),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        max_iter: 200,
        tol: 1e-7,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry),
            design_exit: DesignMatrix::from(design_exit),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit,
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![penalty],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(array![0.0]),
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
    }
}

/// Closed-form uncensored lognormal MLE: `mu_hat = mean(log t)`,
/// `sigma_hat = population sd(log t)`.
fn lognormal_closed_form_mle(log_t: &Array1<f64>) -> (f64, f64) {
    let mu = log_t.mean().unwrap();
    let sigma = (log_t.mapv(|v| (v - mu).powi(2)).sum() / (log_t.len() as f64)).sqrt();
    (mu, sigma)
}

/// gam#2112: the reduced parametric-AFT direct MLE must CONVERGE and recover the
/// closed-form lognormal MLE at sample sizes where the removed absolute
/// summed-gradient tolerance failed (n ≳ 1000). This drives the real reduced
/// path and checks the fitted location intercept / constant log-σ against the
/// closed form `mu = mean(log t)`, `log σ = log sd(log t)`.
#[test]
fn reduced_parametric_aft_converges_and_recovers_mle_at_scale() {
    for &n in &[2000usize, 5000, 10000] {
        let (age_exit, event, log_t) = reduced_aft_lognormal_sample(n, 1.4, 0.5, 0);
        let spec = reduced_aft_lognormal_spec(&age_exit, &event, 1.0);

        // Confirm we are exercising the fixed code path (the direct parametric
        // AFT MLE), not the coupled REML fallback.
        let prepared = prepare_survival_location_scale_model(&spec).expect("prepare");
        assert!(
            prepared.is_reduced_parametric_aft(),
            "n={n}: expected the reduced parametric-AFT regime (the fit_parametric_aft_direct_mle path)"
        );
        assert!(
            prepared.family.location_log_time.is_some(),
            "n={n}: the log-t AFT baseline must be encoded for a lognormal AFT"
        );

        // The core regression: this used to hard-error with
        // "direct parametric-AFT MLE: failed to converge after 200 Newton
        // iterations" for n ≳ 1000. It must now converge.
        let (fit, _) = fit_survival_location_scale_with_geometry(spec)
            .unwrap_or_else(|e| panic!("n={n}: reduced parametric-AFT MLE must converge: {e}"));

        let (mu_hat, sigma_hat) = lognormal_closed_form_mle(&log_t);
        let loc = fit.beta_threshold()[0];
        let log_sigma = fit.beta_log_sigma()[0];
        // A CONVERGED Newton MLE lands on the closed-form optimum. The recovered
        // values match to ~1e-6 in practice; 1e-3 leaves ample slack for the
        // Newton stop while still catching a mis-converged / non-stationary fit.
        assert!(
            (loc - mu_hat).abs() < 1e-3,
            "n={n}: location {loc:.6} != closed-form mu {mu_hat:.6}"
        );
        assert!(
            (log_sigma - sigma_hat.ln()).abs() < 1e-3,
            "n={n}: log-sigma {log_sigma:.6} != closed-form {:.6}",
            sigma_hat.ln()
        );
    }
}

/// gam#2112 (the mechanism, from a second angle): the stopping criterion must be
/// invariant to the TOTAL WEIGHT, exactly as it must be invariant to `n`. The
/// per-row weights multiply every likelihood contribution, so a uniform weight
/// `W` scales the summed log-likelihood — and hence the summed gradient `g = ∇ℓ`
/// and Hessian `H = −∇²ℓ` — by `W`, while leaving the MLE `θ̂ = argmax ℓ`
/// unchanged. The removed absolute test on `‖g‖∞` therefore fails to converge
/// for large `W` (the summed gradient's floor scales with `W`), whereas the
/// Newton-decrement test `gᵀH⁻¹g` cancels the `W` and certifies stationarity at
/// the SAME `θ̂`. Fitting the identical sample at `W = 1` and a large `W` must
/// converge to the same coefficients.
#[test]
fn reduced_parametric_aft_stopping_criterion_is_weight_scale_invariant() {
    let (age_exit, event, _log_t) = reduced_aft_lognormal_sample(1500, 1.2, 0.6, 7);

    let fit_at_weight = |w: f64| -> (f64, f64) {
        let spec = reduced_aft_lognormal_spec(&age_exit, &event, w);
        let prepared = prepare_survival_location_scale_model(&spec).expect("prepare");
        assert!(
            prepared.is_reduced_parametric_aft(),
            "expected reduced parametric-AFT regime"
        );
        let (fit, _) = fit_survival_location_scale_with_geometry(spec).unwrap_or_else(|e| {
            panic!("reduced parametric-AFT MLE must converge at total-weight scale w={w}: {e}")
        });
        (fit.beta_threshold()[0], fit.beta_log_sigma()[0])
    };

    let (loc1, ls1) = fit_at_weight(1.0);
    // W = 500 makes the summed gradient 500× larger — well past the regime where
    // the old absolute 1e-7 gradient tolerance could ever be met — yet the MLE
    // is identical, so a scale-invariant criterion converges to the same point.
    let (loc500, ls500) = fit_at_weight(500.0);

    assert!(
        (loc1 - loc500).abs() < 1e-6,
        "location must be weight-scale invariant: w=1 -> {loc1:.9}, w=500 -> {loc500:.9}"
    );
    assert!(
        (ls1 - ls500).abs() < 1e-6,
        "log-sigma must be weight-scale invariant: w=1 -> {ls1:.9}, w=500 -> {ls500:.9}"
    );
}
