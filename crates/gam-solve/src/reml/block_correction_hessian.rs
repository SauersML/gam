//! The exact outer ρ-Hessian of the latched #784 block-local correction.
//!
//! The correction enters the REML/LAML cost as `−Δ_b` with
//! `Δ_b = Σ_k V_k + Φ`: one Gauss–Hermite piece `V_k` per block axis (or one
//! for a single-axis block) plus, when the block is split by axis, the analytic
//! mixed-axis term `Φ` ([`mixed_axis_laplace_term`]). Every quantity either
//! reads depends on ρ through three smooth objects, and this module carries
//! each of them to second order:
//!
//! * the mode `β̂(ρ)`, by the implicit function theorem on `Sβ̂ = ∇ℓ(β̂)`;
//! * the block eigenpairs `(λ_r, u_r)(ρ)` of `H = XᵀWX + S_λ`, by simple-
//!   eigenvalue perturbation theory (the caller has already refused any
//!   near-degenerate pair, where the eigenframe is not differentiable);
//! * the row curvature `W(η̂)` and its η-derivatives `c, d, e, f`.
//!
//! A piece is `V = log Σ_q w_q e^{−F(z_q Y)} − log Σ_q w_q` over fixed
//! standard-normal nodes `z_q` with `Y = X u/√λ`, so
//! `−∂²V = E_p[∂²F] − Var_p(∂F)` under the node posterior `p ∝ w e^{−F}`, and
//! `F`'s row derivatives in `(η̂, s)` are closed forms of `ψ`, `ψ'` and `ψ''`.
//! No quantity here is differenced; the tests difference it.

use super::block_quadrature_correction::{block_axis_target, mixed_axis_laplace_term};
use super::*;

/// How a row's likelihood curvature `ψ''(η)` is evaluated off the mode, where
/// the second derivative of the excess `F` reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RowCurvature {
    /// Binomial-logit: `ψ'' = w μ(1−μ)`.
    CanonicalLogit,
    /// Poisson-log: `ψ'' = w e^η`.
    PoissonLog,
    /// The family's own observed information.
    Observed,
}

/// Whether the correction's ρ-Hessian exists in closed form for this fit, and
/// how its row curvature is evaluated. `Err` names the mathematical reason.
///
/// The excess `F = Σ ψ(η̂+s) − ψ(η̂) − ψ'(η̂)s − ½W s²` has the row derivatives
/// this module uses only when `W = ψ''(η̂)`: the exported curvature is the
/// likelihood's observed information, or a canonical link where the Fisher
/// weights coincide with it. The split block's `Φ` additionally needs the
/// fourth η-derivative of `W`, which is carried for the families whose
/// curvature has a closed-form derivative chain.
pub(super) fn block_correction_row_curvature(
    pirls_result: &PirlsResult,
    inverse_link: &InverseLink,
    axis_split: bool,
    block_dim: usize,
) -> Result<RowCurvature, String> {
    if !matches!(pirls_result.firth, crate::pirls::FirthDiagnostics::Inactive) {
        return Err(
            "the Firth-penalized inner objective has no closed-form ρ-derivative of its \
             Jeffreys term through the block eigenframe"
                .to_string(),
        );
    }
    if pirls_result.derivatives_unsupported {
        return Err("the inner solve exports no likelihood derivatives past the curvature".to_string());
    }
    let likelihood = &pirls_result.likelihood;
    let response = reml_spec(likelihood).response;
    let curvature = match (&response, inverse_link) {
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
            RowCurvature::CanonicalLogit
        }
        (ResponseFamily::Poisson, InverseLink::Standard(StandardLink::Log)) => {
            RowCurvature::PoissonLog
        }
        _ => {
            if !matches!(
                pirls_result.exported_laplace_curvature,
                crate::pirls::ExportedLaplaceCurvature::ObservedExact
            ) || !crate::pirls::supports_observed_hessian_curvature_for_likelihood(
                likelihood,
                inverse_link,
            ) {
                return Err(format!(
                    "the exported curvature of {response:?} is not the likelihood's observed \
                     information, so W ≠ ψ''(η̂) and the excess F has no closed-form second \
                     derivative in η̂"
                ));
            }
            RowCurvature::Observed
        }
    };
    if axis_split
        && !matches!(curvature, RowCurvature::CanonicalLogit | RowCurvature::PoissonLog)
        && !matches!(
            (&response, inverse_link),
            (ResponseFamily::Gamma, InverseLink::Standard(StandardLink::Log))
        )
    {
        return Err(format!(
            "the mixed-axis term Φ of a split {block_dim}-axis block needs ∂⁴W/∂η⁴, which \
             {response:?} carries in no closed form"
        ));
    }
    if !axis_split && block_dim >= 2 {
        return Err(format!(
            "a {block_dim}-axis block integrated by one tensor rule has no exact curvature, so \
             its excess has no closed-form second derivative"
        ));
    }
    Ok(curvature)
}

/// `∂⁴W/∂η⁴` per row for a split block's `Φ`, from `e = ∂³W/∂η³`.
///
/// Logit carries it from the inverse-link 5-jet; `W = w e^{±η}` families
/// (Poisson-log, and Gamma-log's `W = w y e^{−η}/φ`) repeat or alternate it.
pub(super) fn curvature_fourth_derivative(
    pirls_result: &PirlsResult,
    inverse_link: &InverseLink,
    prior_weights: &Array1<f64>,
    e: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    let response = reml_spec(&pirls_result.likelihood).response;
    match (&response, inverse_link) {
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
            let eta = &pirls_result.final_eta;
            Ok(Array1::from_shape_fn(eta.len(), |i| {
                prior_weights[i] * crate::mixture_link::logit_inverse_link_jet5(eta[i]).d5
            }))
        }
        (ResponseFamily::Poisson, InverseLink::Standard(StandardLink::Log)) => Ok(e.clone()),
        (ResponseFamily::Gamma, InverseLink::Standard(StandardLink::Log)) => Ok(-e),
        _ => crate::bail_invalid_estim!(
            "#784 mixed-axis ρ-Hessian: {response:?} carries no closed-form ∂⁴W/∂η⁴"
        ),
    }
}

/// `ψ''(η)` per row at a displaced linear predictor.
fn displaced_row_curvature(
    target: &Gam784BlockTarget<'_>,
    curvature: RowCurvature,
    eta: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    let n = eta.len();
    let pw = &target.prior_weights;
    let phi = target.phi;
    Ok(match curvature {
        RowCurvature::CanonicalLogit => Array1::from_shape_fn(n, |i| {
            pw[i] * crate::mixture_link::logit_inverse_link_jet5(eta[i]).d1 / phi
        }),
        RowCurvature::PoissonLog => Array1::from_shape_fn(n, |i| pw[i] * eta[i].exp() / phi),
        RowCurvature::Observed => {
            crate::pirls::compute_observed_hessian_curvature_arrays(
                &target.likelihood,
                &target.inverse_link,
                eta,
                target.y.view(),
                &Array1::zeros(n),
                target.prior_weights.view(),
            )?
            .0
        }
    })
}

/// `S v` for a block-local penalty, on a direction (no prior-mean centring).
pub(super) fn penalty_local_matvec(
    penalty: &gam_terms::construction::CanonicalPenalty,
    v: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(v.len());
    let local = penalty
        .local
        .dot(&v.slice(ndarray::s![penalty.col_range.clone()]));
    out.slice_mut(ndarray::s![penalty.col_range.clone()])
        .assign(&local);
    out
}

/// The ρ-pairs `j ≤ l` the second-order quantities are indexed by.
fn rho_pairs(n_rho: usize) -> Vec<(usize, usize)> {
    (0..n_rho)
        .flat_map(|j| (j..n_rho).map(move |l| (j, l)))
        .collect()
}

/// Everything the Hessian reads besides the block target.
pub(super) struct BlockCorrectionHessianInputs<'x> {
    pub(super) curvature: RowCurvature,
    pub(super) axis_split: bool,
    /// The Gauss–Hermite order of each block axis (one entry for `m = 1`).
    pub(super) axis_orders: &'x [usize],
    /// The eigensystem of `H` the block was drawn from.
    pub(super) evals: &'x Array1<f64>,
    pub(super) evecs: &'x Array2<f64>,
    pub(super) block_cols: &'x [usize],
    /// `c = ∂W/∂η` and `d = ∂²W/∂η²` per row.
    pub(super) c: &'x Array1<f64>,
    pub(super) d: &'x Array1<f64>,
    /// `e = ∂³W/∂η³` and `f = ∂⁴W/∂η⁴`, for a split block's `Φ`.
    pub(super) e_f: Option<(&'x Array1<f64>, &'x Array1<f64>)>,
}

/// The cost-side second-order content of `−Δ_b`.
pub(super) struct BlockCorrectionCostHessian {
    /// `∂²(−Δ_b)/∂ρ∂ρᵀ`.
    pub(super) hessian: Array2<f64>,
    /// `∂(−Δ_b)/∂ρ` from the same closed forms, against which the spliced
    /// channel gradient is checked.
    pub(super) implied_gradient: Array1<f64>,
    /// Each piece's `V_k` from the nodes this module integrated.
    pub(super) piece_values: Vec<f64>,
    /// `Φ`, for a split block.
    pub(super) mixed_value: Option<f64>,
}

/// The mode's ρ-motion: `β̇_j`, `E_j = X β̇_j`, and per pair `η̈_jl` and the
/// curvature's second motion `Ẅ_jl = c ⊙ η̈_jl + d ⊙ E_j ⊙ E_l`.
struct ModeMotion {
    eta_dot: Vec<Array1<f64>>,
    eta_ddot: Vec<Array1<f64>>,
    w_ddot: Vec<Array1<f64>>,
}

/// One block axis's whitened design column `Y = X u/√λ` and its ρ-motion.
struct AxisMotion {
    y: Array1<f64>,
    y_dot: Vec<Array1<f64>>,
    y_ddot: Vec<Array1<f64>>,
}

struct Geometry<'g, 't> {
    target: &'g Gam784BlockTarget<'t>,
    evals: &'g Array1<f64>,
    evecs: &'g Array2<f64>,
    c: &'g Array1<f64>,
    d: &'g Array1<f64>,
    pairs: Vec<(usize, usize)>,
}

impl Geometry<'_, '_> {
    fn x(&self) -> &Array2<f64> {
        self.target.x_transformed
    }

    /// `λ_j S_j v`.
    fn penalty_mv(&self, j: usize, v: ArrayView1<'_, f64>) -> Array1<f64> {
        penalty_local_matvec(&self.target.penalties[j], v) * self.target.lambdas[j]
    }

    /// `H⁻¹ v` through the block's eigensystem.
    fn solve(&self, v: &Array1<f64>) -> Array1<f64> {
        self.evecs.dot(&(self.evecs.t().dot(v) / self.evals))
    }

    fn mode_motion(&self) -> ModeMotion {
        let x = self.x();
        let n_rho = self.target.lambdas.len();
        let score: Vec<Array1<f64>> = (0..n_rho)
            .map(|j| &self.target.penalty_scores[j] * self.target.lambdas[j])
            .collect();
        // H β̇_j = −λ_j S_j β̂.
        let beta_dot: Vec<Array1<f64>> = score.iter().map(|a| -self.solve(a)).collect();
        let eta_dot: Vec<Array1<f64>> = beta_dot.iter().map(|b| x.dot(b)).collect();
        let mut eta_ddot = Vec::with_capacity(self.pairs.len());
        let mut w_ddot = Vec::with_capacity(self.pairs.len());
        for &(j, l) in &self.pairs {
            // Differentiating H β̇_j = −λ_j S_j β̂ along ρ_l, with
            // Ḣ_l = λ_l S_l + Xᵀ diag(c ⊙ E_l) X.
            let mut rhs = self.penalty_mv(l, beta_dot[j].view());
            rhs += &self.penalty_mv(j, beta_dot[l].view());
            rhs += &x.t().dot(&(self.c * &eta_dot[l] * &eta_dot[j]));
            if j == l {
                rhs += &score[j];
            }
            let eta_jl = -x.dot(&self.solve(&rhs));
            w_ddot.push(self.c * &eta_jl + &(self.d * &eta_dot[j] * &eta_dot[l]));
            eta_ddot.push(eta_jl);
        }
        ModeMotion {
            eta_dot,
            eta_ddot,
            w_ddot,
        }
    }

    /// The motion of the eigenpair in column `col` of the eigensystem.
    fn axis_motion(&self, col: usize, mode: &ModeMotion) -> AxisMotion {
        let x = self.x();
        let n_rho = self.target.lambdas.len();
        let lambda = self.evals[col];
        let u = self.evecs.column(col);
        // R = Σ_{q≠col} u_q u_qᵀ/(λ − σ_q), so u̇ = R Ḣ u.
        let gains = Array1::from_shape_fn(self.evals.len(), |q| {
            if q == col {
                0.0
            } else {
                (lambda - self.evals[q]).recip()
            }
        });
        let resolvent = |v: &Array1<f64>| self.evecs.dot(&(self.evecs.t().dot(v) * &gains));
        let xu = x.dot(&u);
        let s_u: Vec<Array1<f64>> = (0..n_rho).map(|j| self.penalty_mv(j, u)).collect();
        // b_j = Ḣ_j u.
        let b: Vec<Array1<f64>> = (0..n_rho)
            .map(|j| &s_u[j] + &x.t().dot(&(self.c * &mode.eta_dot[j] * &xu)))
            .collect();
        let lambda_dot: Vec<f64> = b.iter().map(|b_j| u.dot(b_j)).collect();
        let u_dot: Vec<Array1<f64>> = b.iter().map(|b_j| resolvent(b_j)).collect();
        let xu_dot: Vec<Array1<f64>> = u_dot.iter().map(|v| x.dot(v)).collect();

        let inv_sqrt = lambda.sqrt().recip();
        let inv_32 = inv_sqrt / lambda;
        let inv_52 = inv_32 / lambda;
        let y = &xu * inv_sqrt;
        let y_dot: Vec<Array1<f64>> = (0..n_rho)
            .map(|j| &xu_dot[j] * inv_sqrt - &(&xu * (0.5 * inv_32 * lambda_dot[j])))
            .collect();
        let mut y_ddot = Vec::with_capacity(self.pairs.len());
        for (k, &(j, l)) in self.pairs.iter().enumerate() {
            let w_jl = &mode.w_ddot[k];
            // λ̈ = uᵀḦu + 2 u̇_jᵀ Ḣ_l u.
            let mut lambda_ddot = (w_jl * &xu * &xu).sum() + 2.0 * u_dot[j].dot(&b[l]);
            // Ḧu + Ḣ_j u̇_l + Ḣ_l u̇_j − λ̇_l u̇_j − λ̇_j u̇_l.
            let mut v1 = self.penalty_mv(l, u_dot[j].view());
            v1 += &self.penalty_mv(j, u_dot[l].view());
            let row_part = w_jl * &xu
                + &(self.c * &mode.eta_dot[l] * &xu_dot[j])
                + &(self.c * &mode.eta_dot[j] * &xu_dot[l]);
            v1 += &x.t().dot(&row_part);
            if j == l {
                lambda_ddot += u.dot(&s_u[j]);
                v1 += &s_u[j];
            }
            v1.scaled_add(-lambda_dot[l], &u_dot[j]);
            v1.scaled_add(-lambda_dot[j], &u_dot[l]);
            // ü = R v1 − u (u̇_j·u̇_l); the second term keeps |u| = 1.
            let mut u_ddot = resolvent(&v1);
            u_ddot.scaled_add(-u_dot[j].dot(&u_dot[l]), &u);
            let mut y_jl = x.dot(&u_ddot) * inv_sqrt;
            y_jl.scaled_add(
                -0.5 * inv_32 * lambda_dot[l],
                &xu_dot[j],
            );
            y_jl.scaled_add(
                -0.5 * inv_32 * lambda_dot[j],
                &xu_dot[l],
            );
            y_jl.scaled_add(
                0.75 * inv_52 * lambda_dot[j] * lambda_dot[l] - 0.5 * inv_32 * lambda_ddot,
                &xu,
            );
            y_ddot.push(y_jl);
        }
        AxisMotion { y, y_dot, y_ddot }
    }
}

/// One Gauss–Hermite piece's value and its cost-side ρ-gradient and Hessian.
struct PieceSecondOrder {
    value: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

/// One end of a truncated axis in the whitened coordinate, `e = √λ·t_cut`, and
/// its ρ-motion.
///
/// The end is where row `r`'s predictor reaches its domain boundary `b`:
/// `η̂_r + Y_r e = b` with `Y = X u/√λ`, so differentiating at fixed `b`
///
/// ```text
///   ė_j  = −(E_{j,r} + e Ẏ_{j,r}) / Y_r,
///   ë_jl = −(η̈_{jl,r} + ė_j Ẏ_{l,r} + ė_l Ẏ_{j,r} + e Ÿ_{jl,r}) / Y_r.
/// ```
///
/// The active row is the tightest cut, which is locally constant in ρ.
struct AxisEnd {
    value: f64,
    /// `∂ ln Z/∂e`.
    log_mass_gradient: f64,
    /// Which of the transport's sensitivities `(∂τ/∂lower, ∂τ/∂upper)` is this end's.
    side: usize,
    dot: Vec<f64>,
    ddot: Vec<f64>,
}

impl AxisEnd {
    fn new(
        value: f64,
        log_mass_gradient: f64,
        side: usize,
        row: usize,
        axis: &AxisMotion,
        mode: &ModeMotion,
        pairs: &[(usize, usize)],
    ) -> Result<Self, EstimationError> {
        let y_r = axis.y[row];
        if !(y_r != 0.0) || !value.is_finite() {
            crate::bail_invalid_estim!(
                "#784 ρ-Hessian: a feasible-interval end at row {row} has whitened slope {y_r} \
                 and position {value}"
            );
        }
        let dot: Vec<f64> = (0..axis.y_dot.len())
            .map(|j| -(mode.eta_dot[j][row] + value * axis.y_dot[j][row]) / y_r)
            .collect();
        let ddot = pairs
            .iter()
            .enumerate()
            .map(|(k, &(j, l))| {
                -(mode.eta_ddot[k][row]
                    + dot[j] * axis.y_dot[l][row]
                    + dot[l] * axis.y_dot[j][row]
                    + value * axis.y_ddot[k][row])
                    / y_r
            })
            .collect();
        Ok(Self {
            value,
            log_mass_gradient,
            side,
            dot,
            ddot,
        })
    }
}

/// A piece is integrated on the corrector's own rule: the standard-normal
/// Gauss–Hermite nodes `u_q`, transported onto the block's feasible interval
/// when the likelihood ends inside the Laplace Gaussian
/// ([`gam_math::quadrature::TruncatedNormalTransport`]), with
///
/// ```text
///   V = ln Σ_q w_q e^{−F(Y τ_q)} − ln Σ_q w_q + ln Z,   τ_q = τ(u_q; ends).
/// ```
///
/// On a whole-line axis `τ_q = u_q` and `Z = 1`. On a truncated one the ends
/// move with ρ, and with them every node and the mass. From
/// `Φ(τ) = Φ(α)(1 − Φ(u)) + Φ(β)Φ(u)` and `φ' = −xφ`, the node's end
/// sensitivities `τ_a` satisfy `τ_ab = τ τ_a τ_b − δ_ab e_a τ_a`, and from
/// `Z = Φ(β) − Φ(α)` with `G_a = ∂ ln Z/∂e_a`,
/// `∂² ln Z/∂e_a∂e_b = −G_a G_b − δ_ab e_a G_a`.
fn piece_second_order(
    piece_target: &Gam784BlockTarget<'_>,
    curvature: RowCurvature,
    order: usize,
    axis: &AxisMotion,
    mode: &ModeMotion,
    geometry: &Geometry<'_, '_>,
) -> Result<PieceSecondOrder, EstimationError> {
    let n_rho = axis.y_dot.len();
    let pairs = &geometry.pairs;
    let lambda = piece_target.block_lambdas[0];
    let sqrt_lambda = lambda.sqrt();
    let rule = gam_math::quadrature::standard_normal_gauss_hermite_rule(order).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "#784 ρ-Hessian: Gauss–Hermite rule of order {order}: {error}"
        ))
    })?;

    let truncation = piece_target.axis_truncation();
    let cuts = truncation
        .as_deref()
        .map_or([None, None], |truncation| [truncation.lower(), truncation.upper()]);
    let transport = if cuts.iter().any(Option::is_some) {
        let end = |cut: Option<gam_problem::laplace_sampler_contract::BlockAxisCut>, open: f64| {
            cut.map_or(open, |cut| sqrt_lambda * cut.t)
        };
        Some(
            gam_math::quadrature::TruncatedNormalTransport::new(
                end(cuts[0], f64::NEG_INFINITY),
                end(cuts[1], f64::INFINITY),
            )
            .map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "#784 ρ-Hessian: the mode is not inside the block's feasible interval: {error}"
                ))
            })?,
        )
    } else {
        None
    };
    let mut ends: Vec<AxisEnd> = Vec::with_capacity(2);
    let mut log_mass_of_interval = 0.0;
    if let Some(transport) = &transport {
        let (lower_gradient, upper_gradient) = transport.log_mass_endpoint_gradient();
        let sides = [
            (transport.lower(), lower_gradient),
            (transport.upper(), upper_gradient),
        ];
        for (side, (cut, (value, gradient))) in cuts.iter().zip(sides).enumerate() {
            if let Some(cut) = cut {
                ends.push(AxisEnd::new(value, gradient, side, cut.row, axis, mode, pairs)?);
            }
        }
        log_mass_of_interval = transport.log_mass();
    }

    // Each node's image `τ` and its end sensitivities `(∂τ/∂lower, ∂τ/∂upper)`.
    let mut nodes: Vec<(f64, [f64; 2], f64)> = Vec::with_capacity(rule.len());
    for &(u, w) in &rule {
        let node = match &transport {
            Some(transport) => {
                let tau = transport.transport(u).map_err(|error| {
                    EstimationError::InvalidInput(format!(
                        "#784 ρ-Hessian: transporting a node onto the feasible interval: {error}"
                    ))
                })?;
                let (lower, upper) = transport.endpoint_sensitivities(u, tau);
                (tau, [lower, upper], w)
            }
            None => (u, [0.0, 0.0], w),
        };
        nodes.push(node);
    }
    let draws = Array2::from_shape_fn((1, nodes.len()), |(_, q)| nodes[q].0 / sqrt_lambda);
    let batched = piece_target.excess_with_displaced_neg_score_batch(&draws);
    if batched.len() != nodes.len() {
        crate::bail_invalid_estim!(
            "#784 ρ-Hessian: the excess batch returned {} nodes for {}",
            batched.len(),
            nodes.len()
        );
    }
    let ngs_base = piece_target
        .base_neg_score()
        .map_err(EstimationError::InvalidInput)?;
    let w_mode = &piece_target.weights_obs;
    let log_norm = log_sum_exp(nodes.iter().map(|&(_, _, w)| w.ln()));
    let feasible: Vec<(f64, [f64; 2], f64, Array1<f64>)> = batched
        .into_iter()
        .zip(nodes.iter())
        .filter_map(|((excess, ngs), &(tau, sensitivity, w))| match ngs {
            Some(ngs) if excess.is_finite() => Some((tau, sensitivity, w.ln() - excess, ngs)),
            _ => None,
        })
        .collect();
    if feasible.is_empty() {
        crate::bail_invalid_estim!("#784 ρ-Hessian: every Gauss–Hermite node was infeasible");
    }
    let log_mass = log_sum_exp(feasible.iter().map(|(_, _, lw, _)| *lw));
    let value = log_mass - log_norm + log_mass_of_interval;

    let mut expected_second = Array1::<f64>::zeros(pairs.len());
    let mut node_gradients: Vec<(f64, Array1<f64>)> = Vec::with_capacity(feasible.len());
    for (tau, sensitivity, lw, ngs) in feasible {
        let prob = (lw - log_mass).exp();
        let s = &axis.y * tau;
        let psi2 = displaced_row_curvature(piece_target, curvature, &(&piece_target.eta_hat + &s))?;
        // Row derivatives of F_i = ψ(η̂+s) − ψ(η̂) − ψ'(η̂)s − ½W s², W = ψ''(η̂).
        let f_s = &ngs - &ngs_base - &(w_mode * &s);
        let f_ss = &psi2 - w_mode;
        let s2 = &s * &s;
        let f_e = &f_s - &(geometry.c * &s2 * 0.5);
        let f_es = &f_ss - &(geometry.c * &s);
        let f_ee = &f_es - &(geometry.d * &s2 * 0.5);
        // The node `s = Y τ` moves with the axis and, on a truncated axis, with
        // the ends: ṡ_j = Ẏ_j τ + Y τ̇_j.
        let tau_dot: Vec<f64> = (0..n_rho)
            .map(|j| ends.iter().map(|end| sensitivity[end.side] * end.dot[j]).sum())
            .collect();
        let s_dot: Vec<Array1<f64>> = (0..n_rho)
            .map(|j| &axis.y_dot[j] * tau + &(&axis.y * tau_dot[j]))
            .collect();
        let f_s_y = f_s.dot(&axis.y);
        let f_s_y_dot: Vec<f64> = axis.y_dot.iter().map(|y_j| f_s.dot(y_j)).collect();
        let node_gradient =
            Array1::from_shape_fn(n_rho, |j| f_e.dot(&mode.eta_dot[j]) + f_s.dot(&s_dot[j]));
        for (k, &(j, l)) in pairs.iter().enumerate() {
            let (e_j, e_l) = (&mode.eta_dot[j], &mode.eta_dot[l]);
            // τ̈_jl = Σ_ab τ_ab ė_a,j ė_b,l + Σ_a τ_a ë_a,jl.
            let mut tau_ddot = 0.0;
            for a in &ends {
                let tau_a = sensitivity[a.side];
                tau_ddot += tau_a * a.ddot[k] - a.value * tau_a * a.dot[j] * a.dot[l];
                for b in &ends {
                    tau_ddot += tau * tau_a * sensitivity[b.side] * a.dot[j] * b.dot[l];
                }
            }
            // f_s·s̈_jl with s̈_jl = Ÿ_jl τ + Ẏ_j τ̇_l + Ẏ_l τ̇_j + Y τ̈_jl.
            let f_s_s_ddot = tau * f_s.dot(&axis.y_ddot[k])
                + tau_dot[l] * f_s_y_dot[j]
                + tau_dot[j] * f_s_y_dot[l]
                + tau_ddot * f_s_y;
            expected_second[k] += prob
                * ((&f_ee * e_j).dot(e_l)
                    + (&f_es * e_j).dot(&s_dot[l])
                    + (&f_es * e_l).dot(&s_dot[j])
                    + (&f_ss * &s_dot[j]).dot(&s_dot[l])
                    + f_e.dot(&mode.eta_ddot[k])
                    + f_s_s_ddot);
        }
        node_gradients.push((prob, node_gradient));
    }
    let mut node_gradient_mean = Array1::<f64>::zeros(n_rho);
    for (prob, g) in &node_gradients {
        node_gradient_mean.scaled_add(*prob, g);
    }
    // The cost is −V: E_p[∂²F] − Var_p(∂F) − ∂² ln Z, with gradient
    // E_p[∂F] − ∂ ln Z.
    let mut gradient = node_gradient_mean.clone();
    for end in &ends {
        for j in 0..n_rho {
            gradient[j] -= end.log_mass_gradient * end.dot[j];
        }
    }
    let mut hessian = Array2::<f64>::zeros((n_rho, n_rho));
    for (k, &(j, l)) in pairs.iter().enumerate() {
        let mut h = expected_second[k];
        for (prob, g) in &node_gradients {
            h -= prob * (g[j] - node_gradient_mean[j]) * (g[l] - node_gradient_mean[l]);
        }
        for a in &ends {
            h -= a.log_mass_gradient * a.ddot[k] - a.value * a.log_mass_gradient * a.dot[j] * a.dot[l];
            for b in &ends {
                h += a.log_mass_gradient * b.log_mass_gradient * a.dot[j] * b.dot[l];
            }
        }
        hessian[(j, l)] = h;
        hessian[(l, j)] = h;
    }
    Ok(PieceSecondOrder {
        value,
        gradient,
        hessian,
    })
}

fn log_sum_exp(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let max = values.clone().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values.map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// The cost-side ρ-gradient and ρ-Hessian of `−Δ_b` at this evaluation.
pub(super) fn block_correction_cost_hessian(
    target: &Gam784BlockTarget<'_>,
    inputs: &BlockCorrectionHessianInputs<'_>,
) -> Result<BlockCorrectionCostHessian, EstimationError> {
    let m = inputs.block_cols.len();
    let n_rho = target.lambdas.len();
    let geometry = Geometry {
        target,
        evals: inputs.evals,
        evecs: inputs.evecs,
        c: inputs.c,
        d: inputs.d,
        pairs: rho_pairs(n_rho),
    };
    let mode = geometry.mode_motion();
    let axes: Vec<AxisMotion> = inputs
        .block_cols
        .iter()
        .map(|&col| geometry.axis_motion(col, &mode))
        .collect();

    let piece_count = if inputs.axis_split { m } else { 1 };
    if !inputs.axis_split && m != 1 {
        crate::bail_invalid_estim!(
            "#784 ρ-Hessian: a {m}-axis block under one tensor rule has no closed-form Hessian"
        );
    }
    let mut hessian = Array2::<f64>::zeros((n_rho, n_rho));
    let mut implied_gradient = Array1::<f64>::zeros(n_rho);
    let mut piece_values = Vec::with_capacity(piece_count);
    for k in 0..piece_count {
        let axis_target = block_axis_target(target, k);
        let piece = piece_second_order(
            &axis_target,
            inputs.curvature,
            inputs.axis_orders[k],
            &axes[k],
            &mode,
            &geometry,
        )?;
        hessian += &piece.hessian;
        implied_gradient += &piece.gradient;
        piece_values.push(piece.value);
    }

    let mixed_value = if inputs.axis_split {
        let Some((e, f)) = inputs.e_f else {
            crate::bail_invalid_estim!("#784 ρ-Hessian: a split block needs the e and f carriers");
        };
        let n = target.x_transformed.nrows();
        let a = Array2::from_shape_fn((n, m), |(i, r)| axes[r].y[i]);
        let a_dot: Vec<Array2<f64>> = (0..n_rho)
            .map(|j| Array2::from_shape_fn((n, m), |(i, r)| axes[r].y_dot[j][i]))
            .collect();
        let a_ddot: Vec<Array2<f64>> = (0..geometry.pairs.len())
            .map(|k| Array2::from_shape_fn((n, m), |(i, r)| axes[r].y_ddot[k][i]))
            .collect();
        let derivatives = mixed_axis_laplace_rho_derivatives(
            a.view(),
            inputs.c,
            inputs.d,
            e,
            f,
            MixedAxisMotion {
                a_dot: &a_dot,
                a_ddot: &a_ddot,
                eta_dot: &mode.eta_dot,
                eta_ddot: &mode.eta_ddot,
                pairs: &geometry.pairs,
            },
        );
        hessian -= &derivatives.hessian;
        implied_gradient -= &derivatives.gradient;
        Some(mixed_axis_laplace_term(a.view(), inputs.c, inputs.d, e).value)
    } else {
        None
    };
    Ok(BlockCorrectionCostHessian {
        hessian,
        implied_gradient,
        piece_values,
        mixed_value,
    })
}

/// `∂Φ/∂ρ` and `∂²Φ/∂ρ∂ρᵀ` of the mixed-axis term.
pub(super) struct MixedAxisRhoDerivatives {
    pub(super) gradient: Array1<f64>,
    pub(super) hessian: Array2<f64>,
}

/// The ρ-motion of the whitened rows and the mode: `Ȧ_j`, `Ä_jl`, `η̇_j`,
/// `η̈_jl`, the second-order quantities indexed by `pairs`.
#[derive(Clone, Copy)]
pub(super) struct MixedAxisMotion<'a> {
    pub(super) a_dot: &'a [Array2<f64>],
    pub(super) a_ddot: &'a [Array2<f64>],
    pub(super) eta_dot: &'a [Array1<f64>],
    pub(super) eta_ddot: &'a [Array1<f64>],
    pub(super) pairs: &'a [(usize, usize)],
}

/// The ρ-derivatives of `Φ(A, c(η), d(η))` along the whitened rows' motion
/// `Ȧ_j`, `Ä_jl` and the mode's `E_j = η̇_j`, `η̈_jl`.
///
/// With `Q_i = |a_i|⁴ − Σ_r a_ir⁴`, `u = Σ c_i |a_i|² a_i` and
/// `T = Σ c_i a_i^{⊗3}`,
///
///   Φ = ⅛|u|² + (1/12)⟨T, T⟩ − (5/24) Σ_r T_rrr² − ⅛ Σ_i d_i Q_i,
///
/// so each derivative is the product rule on `u`, `T` and `Q` with
/// `ċ = d E`, `ḋ = e E`, `c̈ = e E_j E_l + d η̈`, `d̈ = f E_j E_l + e η̈`.
/// The cost is `O(n·m³·n_ρ + n·m²·n_ρ²)`.
pub(super) fn mixed_axis_laplace_rho_derivatives(
    a: ndarray::ArrayView2<'_, f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
    e: &Array1<f64>,
    f: &Array1<f64>,
    motion: MixedAxisMotion<'_>,
) -> MixedAxisRhoDerivatives {
    let MixedAxisMotion {
        a_dot,
        a_ddot,
        eta_dot,
        eta_ddot,
        pairs,
    } = motion;
    let (n, m) = a.dim();
    let n_rho = a_dot.len();
    let at = |p: usize, q: usize, r: usize| (p * m + q) * m + r;
    let m3 = m * m * m;

    let mut u = Array1::<f64>::zeros(m);
    let mut tensor = vec![0.0_f64; m3];
    let mut u_dot = vec![Array1::<f64>::zeros(m); n_rho];
    let mut t_dot = vec![vec![0.0_f64; m3]; n_rho];
    let mut q_dot = vec![0.0_f64; n_rho];
    let sq: Vec<f64> = (0..n).map(|i| a.row(i).dot(&a.row(i))).collect();
    let quartic_q: Vec<f64> = (0..n)
        .map(|i| sq[i] * sq[i] - a.row(i).iter().map(|v| v.powi(4)).sum::<f64>())
        .collect();
    // ∂_j Q_i = 4|a|²(a·Ȧ) − 4 Σ a³Ȧ.
    let q_first = |i: usize, ad: ArrayView1<'_, f64>| {
        let ai = a.row(i);
        4.0 * sq[i] * ai.dot(&ad)
            - 4.0 * ai.iter().zip(ad.iter()).map(|(x, y)| x * x * x * y).sum::<f64>()
    };
    for i in 0..n {
        let ai = a.row(i);
        u.scaled_add(c[i] * sq[i], &ai);
        for p in 0..m {
            for q in 0..m {
                let cpq = c[i] * ai[p] * ai[q];
                for r in 0..m {
                    tensor[at(p, q, r)] += cpq * ai[r];
                }
            }
        }
        for j in 0..n_rho {
            let ad = a_dot[j].row(i);
            let c_dot = d[i] * eta_dot[j][i];
            let d_dot = e[i] * eta_dot[j][i];
            let a_ad = ai.dot(&ad);
            // ∂(|a|² a) = 2(a·Ȧ)a + |a|²Ȧ.
            u_dot[j].scaled_add(c_dot * sq[i], &ai);
            u_dot[j].scaled_add(2.0 * c[i] * a_ad, &ai);
            u_dot[j].scaled_add(c[i] * sq[i], &ad);
            let tj = &mut t_dot[j];
            for p in 0..m {
                for q in 0..m {
                    for r in 0..m {
                        tj[at(p, q, r)] += c_dot * ai[p] * ai[q] * ai[r]
                            + c[i]
                                * (ad[p] * ai[q] * ai[r]
                                    + ai[p] * ad[q] * ai[r]
                                    + ai[p] * ai[q] * ad[r]);
                    }
                }
            }
            q_dot[j] += d_dot * quartic_q[i] + d[i] * q_first(i, ad);
        }
    }
    let t_diag: Vec<f64> = (0..m).map(|r| tensor[at(r, r, r)]).collect();
    let gradient = Array1::from_shape_fn(n_rho, |j| {
        0.25 * u.dot(&u_dot[j])
            + tensor
                .iter()
                .zip(t_dot[j].iter())
                .map(|(x, y)| x * y)
                .sum::<f64>()
                / 6.0
            - (5.0 / 12.0) * (0..m).map(|r| t_diag[r] * t_dot[j][at(r, r, r)]).sum::<f64>()
            - 0.125 * q_dot[j]
    });

    let n_pairs = pairs.len();
    let mut u_ddot = vec![Array1::<f64>::zeros(m); n_pairs];
    let mut tt = vec![0.0_f64; n_pairs];
    let mut t_ddot_diag = vec![vec![0.0_f64; m]; n_pairs];
    let mut q_ddot = vec![0.0_f64; n_pairs];
    let mut taa = vec![0.0_f64; m];
    let mut ta = vec![0.0_f64; m * m];
    for i in 0..n {
        let ai = a.row(i);
        // T(a, ·, ·), T(a, a, ·) and T(a, a, a) at this row.
        for p in 0..m {
            for q in 0..m {
                ta[p * m + q] = (0..m).map(|r| tensor[at(p, q, r)] * ai[r]).sum();
            }
        }
        for q in 0..m {
            taa[q] = (0..m).map(|p| ta[p * m + q] * ai[p]).sum();
        }
        let taaa: f64 = (0..m).map(|q| taa[q] * ai[q]).sum();
        for (k, &(j, l)) in pairs.iter().enumerate() {
            let (aj, al, add) = (a_dot[j].row(i), a_dot[l].row(i), a_ddot[k].row(i));
            let (ej, el, ejl) = (eta_dot[j][i], eta_dot[l][i], eta_ddot[k][i]);
            let (cj, cl) = (d[i] * ej, d[i] * el);
            let c2 = e[i] * ej * el + d[i] * ejl;
            let (dj, dl) = (e[i] * ej, e[i] * el);
            let d2 = f[i] * ej * el + e[i] * ejl;
            let a_aj = ai.dot(&aj);
            let a_al = ai.dot(&al);
            let aj_al = aj.dot(&al);
            let a_add = ai.dot(&add);
            let sqi = sq[i];
            let ci = c[i];
            let uk = &mut u_ddot[k];
            for r in 0..m {
                let d_l = 2.0 * a_al * ai[r] + sqi * al[r];
                let d_j = 2.0 * a_aj * ai[r] + sqi * aj[r];
                let d_jl = 2.0 * (aj_al + a_add) * ai[r]
                    + 2.0 * a_aj * al[r]
                    + 2.0 * a_al * aj[r]
                    + sqi * add[r];
                uk[r] += c2 * sqi * ai[r] + cj * d_l + cl * d_j + ci * d_jl;
            }
            let al_taa: f64 = (0..m).map(|r| al[r] * taa[r]).sum();
            let aj_taa: f64 = (0..m).map(|r| aj[r] * taa[r]).sum();
            let add_taa: f64 = (0..m).map(|r| add[r] * taa[r]).sum();
            let mut aj_ta_al = 0.0;
            for p in 0..m {
                for q in 0..m {
                    aj_ta_al += aj[p] * ta[p * m + q] * al[q];
                }
            }
            tt[k] += c2 * taaa
                + 3.0 * cj * al_taa
                + 3.0 * cl * aj_taa
                + ci * (3.0 * add_taa + 6.0 * aj_ta_al);
            for r in 0..m {
                let a2 = ai[r] * ai[r];
                t_ddot_diag[k][r] += c2 * a2 * ai[r]
                    + 3.0 * cj * a2 * al[r]
                    + 3.0 * cl * a2 * aj[r]
                    + ci * (3.0 * a2 * add[r] + 6.0 * ai[r] * aj[r] * al[r]);
            }
            let q2: f64 = 8.0 * a_aj * a_al + 4.0 * sqi * (aj_al + a_add)
                - (0..m)
                    .map(|r| {
                        12.0 * ai[r] * ai[r] * aj[r] * al[r] + 4.0 * ai[r].powi(3) * add[r]
                    })
                    .sum::<f64>();
            q_ddot[k] += d2 * quartic_q[i]
                + dj * q_first(i, al)
                + dl * q_first(i, aj)
                + d[i] * q2;
        }
    }
    let mut hessian = Array2::<f64>::zeros((n_rho, n_rho));
    for (k, &(j, l)) in pairs.iter().enumerate() {
        let tjtl: f64 = t_dot[j]
            .iter()
            .zip(t_dot[l].iter())
            .map(|(x, y)| x * y)
            .sum();
        let diag: f64 = (0..m)
            .map(|r| {
                t_dot[j][at(r, r, r)] * t_dot[l][at(r, r, r)] + t_diag[r] * t_ddot_diag[k][r]
            })
            .sum();
        let h = 0.25 * (u_dot[j].dot(&u_dot[l]) + u.dot(&u_ddot[k])) + (tjtl + tt[k]) / 6.0
            - (5.0 / 12.0) * diag
            - 0.125 * q_ddot[k];
        hessian[(j, l)] = h;
        hessian[(l, j)] = h;
    }
    MixedAxisRhoDerivatives { gradient, hessian }
}

#[cfg(test)]
mod block_correction_hessian_tests {
    use super::*;

    struct Fixture {
        a: Array2<f64>,
        c: Array1<f64>,
        d: Array1<f64>,
        e: Array1<f64>,
        f: Array1<f64>,
        a_dot: Vec<Array2<f64>>,
        a_ddot: Vec<Array2<f64>>,
        eta_dot: Vec<Array1<f64>>,
        eta_ddot: Vec<Array1<f64>>,
        pairs: Vec<(usize, usize)>,
    }

    fn fixture() -> Fixture {
        let (n, m, n_rho) = (7usize, 3usize, 2usize);
        let a = Array2::from_shape_fn((n, m), |(i, r)| {
            ((i * 7 + r * 3) as f64 * 0.61).sin() * 0.4 + 0.05 * r as f64
        });
        let c = Array1::from_shape_fn(n, |i| (i as f64 * 1.3).cos() * 0.7);
        let d = Array1::from_shape_fn(n, |i| 0.3 + (i as f64 * 0.9).sin() * 0.2);
        let e = Array1::from_shape_fn(n, |i| (i as f64 * 2.1).cos() * 0.5);
        let f = Array1::from_shape_fn(n, |i| (i as f64 * 0.7).sin() * 0.4 - 0.1);
        let a_dot = (0..n_rho)
            .map(|j| {
                Array2::from_shape_fn((n, m), |(i, r)| {
                    ((i * 5 + r * 11 + j * 3) as f64 * 0.37).cos() * 0.3
                })
            })
            .collect();
        let pairs = rho_pairs(n_rho);
        let a_ddot = pairs
            .iter()
            .map(|&(j, l)| {
                Array2::from_shape_fn((n, m), |(i, r)| {
                    ((i * 3 + r * 2 + j * 7 + l * 13) as f64 * 0.53).sin() * 0.2
                })
            })
            .collect();
        let eta_dot = (0..n_rho)
            .map(|j| Array1::from_shape_fn(n, |i| ((i + 2 * j) as f64 * 0.83).sin() * 0.6))
            .collect();
        let eta_ddot = pairs
            .iter()
            .map(|&(j, l)| {
                Array1::from_shape_fn(n, |i| ((i * 2 + j + 3 * l) as f64 * 0.47).cos() * 0.4)
            })
            .collect();
        Fixture {
            a,
            c,
            d,
            e,
            f,
            a_dot,
            a_ddot,
            eta_dot,
            eta_ddot,
            pairs,
        }
    }

    fn pair_index(pairs: &[(usize, usize)], j: usize, l: usize) -> usize {
        let key = (j.min(l), j.max(l));
        pairs.iter().position(|&p| p == key).expect("pair")
    }

    /// Φ along the second-order path `A(t) = a + Σ t_j Ȧ_j + ½ Σ t_j t_l Ä_jl`
    /// with `c` and `d` Taylor-carried along `δ(t) = Σ t_j E_j + ½ Σ t_j t_l η̈_jl`,
    /// whose first two t-derivatives at 0 are the motion the analytic form reads.
    fn phi_along(fx: &Fixture, t: &[f64]) -> f64 {
        let mut a = fx.a.clone();
        let mut delta = Array1::<f64>::zeros(fx.c.len());
        for j in 0..t.len() {
            a.scaled_add(t[j], &fx.a_dot[j]);
            delta.scaled_add(t[j], &fx.eta_dot[j]);
            for l in 0..t.len() {
                let k = pair_index(&fx.pairs, j, l);
                a.scaled_add(0.5 * t[j] * t[l], &fx.a_ddot[k]);
                delta.scaled_add(0.5 * t[j] * t[l], &fx.eta_ddot[k]);
            }
        }
        let delta2 = &delta * &delta;
        let c = &fx.c + &(&fx.d * &delta) + &(&fx.e * &delta2 * 0.5);
        let d = &fx.d + &(&fx.e * &delta) + &(&fx.f * &delta2 * 0.5);
        mixed_axis_laplace_term(a.view(), &c, &d, &fx.e).value
    }

    #[test]
    fn mixed_axis_rho_derivatives_match_central_differences() {
        let fx = fixture();
        let derivatives = mixed_axis_laplace_rho_derivatives(
            fx.a.view(),
            &fx.c,
            &fx.d,
            &fx.e,
            &fx.f,
            MixedAxisMotion {
                a_dot: &fx.a_dot,
                a_ddot: &fx.a_ddot,
                eta_dot: &fx.eta_dot,
                eta_ddot: &fx.eta_ddot,
                pairs: &fx.pairs,
            },
        );
        let n_rho = fx.a_dot.len();
        let h = 1e-6;
        let term = mixed_axis_laplace_term(fx.a.view(), &fx.c, &fx.d, &fx.e);
        for j in 0..n_rho {
            let mut plus = vec![0.0; n_rho];
            let mut minus = vec![0.0; n_rho];
            plus[j] = h;
            minus[j] = -h;
            let fd = (phi_along(&fx, &plus) - phi_along(&fx, &minus)) / (2.0 * h);
            assert!(
                (fd - derivatives.gradient[j]).abs() <= 1e-7 * fd.abs().max(1.0),
                "∂Φ/∂ρ_{j}: analytic {} against FD {fd}",
                derivatives.gradient[j]
            );
            // The same slope through the term's own a- and η-gradients.
            let chained = (&term.a_gradient * &fx.a_dot[j]).sum()
                + term.eta_gradient.dot(&fx.eta_dot[j]);
            assert!(
                (chained - derivatives.gradient[j]).abs() <= 1e-12 * chained.abs().max(1.0),
                "∂Φ/∂ρ_{j}: {} against the chained term gradient {chained}",
                derivatives.gradient[j]
            );
        }
        let h = 1e-4;
        let center = phi_along(&fx, &vec![0.0; n_rho]);
        for j in 0..n_rho {
            for l in j..n_rho {
                let at = |sj: f64, sl: f64| {
                    let mut t = vec![0.0; n_rho];
                    t[j] += sj;
                    t[l] += sl;
                    phi_along(&fx, &t)
                };
                let fd = if j == l {
                    (at(h, 0.0) - 2.0 * center + at(-h, 0.0)) / (h * h)
                } else {
                    (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4.0 * h * h)
                };
                let analytic = derivatives.hessian[(j, l)];
                assert!(
                    (fd - analytic).abs() <= 1e-6 * fd.abs().max(1.0),
                    "∂²Φ/∂ρ_{j}∂ρ_{l}: analytic {analytic} against FD {fd}"
                );
                assert_eq!(derivatives.hessian[(l, j)], analytic);
            }
        }
    }

    #[test]
    fn a_single_axis_has_no_mixed_motion() {
        let fx = fixture();
        let column = |x: &Array2<f64>| x.column(0).to_owned().insert_axis(ndarray::Axis(1));
        let a_dot: Vec<Array2<f64>> = fx.a_dot.iter().map(column).collect();
        let a_ddot: Vec<Array2<f64>> = fx.a_ddot.iter().map(column).collect();
        let derivatives = mixed_axis_laplace_rho_derivatives(
            column(&fx.a).view(),
            &fx.c,
            &fx.d,
            &fx.e,
            &fx.f,
            MixedAxisMotion {
                a_dot: &a_dot,
                a_ddot: &a_ddot,
                eta_dot: &fx.eta_dot,
                eta_ddot: &fx.eta_ddot,
                pairs: &fx.pairs,
            },
        );
        assert!(derivatives.gradient.iter().all(|g| g.abs() <= 1e-14));
        assert!(derivatives.hessian.iter().all(|h| h.abs() <= 1e-14));
    }
}
