//! The exact outer ρ-Hessian of the latched #784 block-local correction.
//!
//! The correction enters the REML/LAML cost as `−Δ_b` with
//! `Δ_b = Σ_k V_k + Ψ`: one Gauss–Hermite piece `V_k` per block axis (or one
//! for a single-axis block) plus, when the block is split by axis, the
//! mixed-axis term `Ψ = Σ_T κ_T f_T` ([`MixedAxisRule`]). Every quantity either
//! reads depends on ρ through three smooth objects, and this module carries
//! each of them to second order:
//!
//! * the mode `β̂(ρ)`, by the implicit function theorem on `Sβ̂ = ∇ℓ(β̂)`;
//! * the block eigenpairs `(λ_r, u_r)(ρ)` of `H = XᵀWX + S_λ`, by simple-
//!   eigenvalue perturbation theory (the caller has already refused any
//!   near-degenerate pair, where the eigenframe is not differentiable);
//! * the row curvature `W(η̂)` and its η-derivatives `c, d`.
//!
//! A piece is `V = log Σ_q w_q e^{−F(z_q Y)} − log Σ_q w_q` over fixed
//! standard-normal nodes `z_q` with `Y = X u/√λ`, so
//! `−∂²V = E_p[∂²F] − Var_p(∂F)` under the node posterior `p ∝ w e^{−F}`, and
//! `F`'s row derivatives in `(η̂, s)` are closed forms of `ψ`, `ψ'` and `ψ''`.
//! Each `f_T` of `Ψ` is the same log-ratio on the three-point rule over the
//! axes of `T`, at `s = Σ_r z_r Y_r`, so it reads the same row derivatives.
//! No quantity here is differenced; the tests difference it.

use super::block_quadrature_correction::{
    MixedAxisPosterior, MixedAxisRule, block_axis_target, visit_mixed_axis_nodes,
};
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
/// weights coincide with it. The split block's `Ψ` reads the same excess on
/// nodes that move several axes at once, so it needs nothing past these.
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
    if !axis_split && block_dim >= 2 {
        return Err(format!(
            "a {block_dim}-axis block integrated by one tensor rule has no exact curvature, so \
             its excess has no closed-form second derivative"
        ));
    }
    Ok(curvature)
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
    /// `Ψ` from the nodes this module integrated, for a split block.
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
        cut: gam_problem::laplace_sampler_contract::BlockAxisCut,
        axis: &AxisMotion,
        mode: &ModeMotion,
        pairs: &[(usize, usize)],
    ) -> Result<Self, EstimationError> {
        let row = cut.row;
        let y_r = axis.y[row];
        // The cut's slope `∂s_row/∂t` is `√λ·Y_r`; a sign disagreement means the
        // truncation and this Hessian describe different block directions.
        if !(y_r * cut.row_slope > 0.0) || !value.is_finite() {
            crate::bail_invalid_estim!(
                "#784 ρ-Hessian: a feasible-interval end at row {row} has whitened slope {y_r}, \
                 cut slope {} and position {value}",
                cut.row_slope
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

/// The row derivatives of `F_i = ψ(η̂+s) − ψ(η̂) − ψ'(η̂)s − ½W s²`,
/// `W = ψ''(η̂)`, at one node `s`, in `(η̂, s)`: `f_s = ψ'(η̂+s) − ψ'(η̂) − W s`,
/// `f_ss = ψ''(η̂+s) − W`, and through `∂W/∂η̂ = c`, `∂²W/∂η̂² = d`,
/// `f_e = f_s − ½c s²`, `f_es = f_ss − c s`, `f_ee = f_es − ½d s²`.
struct NodeRowDerivatives {
    f_s: Array1<f64>,
    f_ss: Array1<f64>,
    f_e: Array1<f64>,
    f_es: Array1<f64>,
    f_ee: Array1<f64>,
}

impl NodeRowDerivatives {
    fn new(
        target: &Gam784BlockTarget<'_>,
        curvature: RowCurvature,
        geometry: &Geometry<'_, '_>,
        s: &Array1<f64>,
        neg_score: &Array1<f64>,
        base_neg_score: &Array1<f64>,
    ) -> Result<Self, EstimationError> {
        let psi2 = displaced_row_curvature(target, curvature, &(&target.eta_hat + s))?;
        let w_mode = &target.weights_obs;
        let f_s = neg_score - base_neg_score - &(w_mode * s);
        let f_ss = &psi2 - w_mode;
        let s2 = s * s;
        let f_e = &f_s - &(geometry.c * &s2 * 0.5);
        let f_es = &f_ss - &(geometry.c * s);
        let f_ee = &f_es - &(geometry.d * &s2 * 0.5);
        Ok(Self {
            f_s,
            f_ss,
            f_e,
            f_es,
            f_ee,
        })
    }

    /// `∂F/∂ρ_j = f_e·E_j + f_s·ṡ_j` for a node moving at `ṡ_j`.
    fn gradient(&self, mode: &ModeMotion, s_dot: &[Array1<f64>]) -> Array1<f64> {
        Array1::from_shape_fn(s_dot.len(), |j| {
            self.f_e.dot(&mode.eta_dot[j]) + self.f_s.dot(&s_dot[j])
        })
    }

    /// `∂²F/∂ρ_j∂ρ_l` for pair `k = (j, l)`, less the node's own acceleration
    /// term `f_s·s̈_jl`.
    fn second_at_fixed_acceleration(
        &self,
        mode: &ModeMotion,
        s_dot: &[Array1<f64>],
        k: usize,
        j: usize,
        l: usize,
    ) -> f64 {
        let (e_j, e_l) = (&mode.eta_dot[j], &mode.eta_dot[l]);
        (&self.f_ee * e_j).dot(e_l)
            + (&self.f_es * e_j).dot(&s_dot[l])
            + (&self.f_es * e_l).dot(&s_dot[j])
            + (&self.f_ss * &s_dot[j]).dot(&s_dot[l])
            + self.f_e.dot(&mode.eta_ddot[k])
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
                ends.push(AxisEnd::new(value, gradient, side, *cut, axis, mode, pairs)?);
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
        let row = NodeRowDerivatives::new(piece_target, curvature, geometry, &s, &ngs, &ngs_base)?;
        // The node `s = Y τ` moves with the axis and, on a truncated axis, with
        // the ends: ṡ_j = Ẏ_j τ + Y τ̇_j.
        let tau_dot: Vec<f64> = (0..n_rho)
            .map(|j| ends.iter().map(|end| sensitivity[end.side] * end.dot[j]).sum())
            .collect();
        let s_dot: Vec<Array1<f64>> = (0..n_rho)
            .map(|j| &axis.y_dot[j] * tau + &(&axis.y * tau_dot[j]))
            .collect();
        let f_s_y = row.f_s.dot(&axis.y);
        let f_s_y_dot: Vec<f64> = axis.y_dot.iter().map(|y_j| row.f_s.dot(y_j)).collect();
        let node_gradient = row.gradient(mode, &s_dot);
        for (k, &(j, l)) in pairs.iter().enumerate() {
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
            let f_s_s_ddot = tau * row.f_s.dot(&axis.y_ddot[k])
                + tau_dot[l] * f_s_y_dot[j]
                + tau_dot[j] * f_s_y_dot[l]
                + tau_ddot * f_s_y;
            expected_second[k] +=
                prob * (row.second_at_fixed_acceleration(mode, &s_dot, k, j, l) + f_s_s_ddot);
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
        let mixed = mixed_axis_second_order(target, inputs.curvature, &axes, &mode, &geometry)?;
        hessian += &mixed.hessian;
        implied_gradient += &mixed.gradient;
        Some(mixed.value)
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

/// `Ψ` of a split block with its cost-side ρ-gradient and ρ-Hessian.
///
/// Every node `z` of the rule sits at `s = Σ_r z_r Y_r` with `z` fixed, so it
/// moves at `ṡ_j = Σ_r z_r Ẏ_{r,j}` and `s̈_jl = Σ_r z_r Ÿ_{r,jl}`. The block
/// target is evaluated once, with its displaced scores, and each feasible
/// node's `∂ΔF` and `∂²ΔF` are kept for [`mixed_axis_cost_second_order`].
fn mixed_axis_second_order(
    target: &Gam784BlockTarget<'_>,
    curvature: RowCurvature,
    axes: &[AxisMotion],
    mode: &ModeMotion,
    geometry: &Geometry<'_, '_>,
) -> Result<PieceSecondOrder, EstimationError> {
    let m = axes.len();
    let n = target.eta_hat.len();
    let n_rho = target.lambdas.len();
    let pairs = &geometry.pairs;
    let rule = MixedAxisRule::new(m);
    let q = rule.node_count();
    let ngs_base = target
        .base_neg_score()
        .map_err(EstimationError::InvalidInput)?;
    let mut excesses = vec![f64::NAN; q];
    let mut node_gradients = Array2::<f64>::zeros((q, n_rho));
    let mut node_seconds = Array2::<f64>::zeros((q, pairs.len()));
    // One node's displacement, its motions and its row derivatives are live at
    // a time.
    let fixed_bytes = n
        .saturating_mul(8 + n_rho)
        .saturating_mul(std::mem::size_of::<f64>());
    visit_mixed_axis_nodes(target, &rule, true, fixed_bytes, |start, z, results| {
        for (k, (excess, score)) in results.into_iter().enumerate() {
            let node = start + k;
            excesses[node] = excess;
            if !excess.is_finite() {
                continue;
            }
            let Some(score) = score else {
                crate::bail_invalid_estim!(
                    "#784 ρ-Hessian: mixed-axis node {node} is feasible and has no displaced score"
                );
            };
            // Most nodes move one to three axes; the others contribute nothing.
            let moved: Vec<(f64, &AxisMotion)> = axes
                .iter()
                .enumerate()
                .filter_map(|(r, axis)| (z[(r, k)] != 0.0).then_some((z[(r, k)], axis)))
                .collect();
            let mut s = Array1::<f64>::zeros(n);
            for &(z_r, axis) in &moved {
                s.scaled_add(z_r, &axis.y);
            }
            let s_dot: Vec<Array1<f64>> = (0..n_rho)
                .map(|j| {
                    let mut v = Array1::<f64>::zeros(n);
                    for &(z_r, axis) in &moved {
                        v.scaled_add(z_r, &axis.y_dot[j]);
                    }
                    v
                })
                .collect();
            let row = NodeRowDerivatives::new(target, curvature, geometry, &s, &score, &ngs_base)?;
            node_gradients
                .row_mut(node)
                .assign(&row.gradient(mode, &s_dot));
            for (p, &(j, l)) in pairs.iter().enumerate() {
                let f_s_s_ddot: f64 = moved
                    .iter()
                    .map(|&(z_r, axis)| z_r * row.f_s.dot(&axis.y_ddot[p]))
                    .sum();
                node_seconds[(node, p)] =
                    row.second_at_fixed_acceleration(mode, &s_dot, p, j, l) + f_s_s_ddot;
            }
        }
        Ok(())
    })?;
    let posterior = rule.posterior(&excesses)?;
    let (gradient, hessian) =
        mixed_axis_cost_second_order(&posterior, &node_gradients, &node_seconds, pairs);
    Ok(PieceSecondOrder {
        value: posterior.value,
        gradient,
        hessian,
    })
}

/// `∂(−Ψ)/∂ρ` and `∂²(−Ψ)/∂ρ∂ρᵀ` from each node's `g = ∂ΔF/∂ρ` (row `z` of
/// `node_gradients`) and `h_jl = ∂²ΔF/∂ρ_j∂ρ_l` (row `z` of `node_seconds`,
/// one column per pair).
///
/// With `f_T = log Σ w_T e^{−ΔF} − log Σ w_T` and `p_T` its node posterior,
/// `∂f_T = −E_{p_T}[g]` and `∂²f_T = Var_{p_T}(g) − E_{p_T}[h]`, so
///
///   ∂(−Ψ)  = Σ_T κ_T E_{p_T}[g],
///   ∂²(−Ψ) = Σ_T κ_T (E_{p_T}[h] − Var_{p_T}(g)).
///
/// Only a piece's feasible nodes are read.
fn mixed_axis_cost_second_order(
    posterior: &MixedAxisPosterior,
    node_gradients: &Array2<f64>,
    node_seconds: &Array2<f64>,
    pairs: &[(usize, usize)],
) -> (Array1<f64>, Array2<f64>) {
    let n_rho = node_gradients.ncols();
    let mut gradient = Array1::<f64>::zeros(n_rho);
    let mut hessian = Array2::<f64>::zeros((n_rho, n_rho));
    for piece in &posterior.pieces {
        let mut mean = Array1::<f64>::zeros(n_rho);
        for &(node, prob) in &piece.nodes {
            mean.scaled_add(prob, &node_gradients.row(node));
        }
        let mut second = vec![0.0_f64; pairs.len()];
        for &(node, prob) in &piece.nodes {
            let centred = &node_gradients.row(node) - &mean;
            for (k, &(j, l)) in pairs.iter().enumerate() {
                second[k] += prob * (node_seconds[(node, k)] - centred[j] * centred[l]);
            }
        }
        gradient.scaled_add(piece.coefficient, &mean);
        for (k, &(j, l)) in pairs.iter().enumerate() {
            let h = piece.coefficient * second[k];
            hessian[(j, l)] += h;
            if j != l {
                hessian[(l, j)] += h;
            }
        }
    }
    (gradient, hessian)
}

#[cfg(test)]
mod mixed_axis_second_order_tests {
    use super::*;

    /// `ΔF(z; θ) = θ₀ P(z) + θ₁² Q(z) + θ₀θ₁ R(z)`, infeasible where the first
    /// two axes both sit at `+√3`.
    fn excess(z: ndarray::ArrayView1<'_, f64>, theta: [f64; 2]) -> f64 {
        if z[0] > 1.0 && z[1] > 1.0 {
            return f64::INFINITY;
        }
        let (p, q, r) = polynomials(z);
        theta[0] * p + theta[1] * theta[1] * q + theta[0] * theta[1] * r
    }

    fn polynomials(z: ndarray::ArrayView1<'_, f64>) -> (f64, f64, f64) {
        let p = z[0].powi(3) + 0.5 * z[1] * z[2] * z[2] - 0.3 * z[3] * z[0];
        let q = 0.2 * z[0] * z[0] * z[1] * z[1] - 0.1 * z[2].powi(4) + 0.15 * z[1] * z[3].powi(2);
        let r = 0.4 * z[0] * z[1] * z[2] + 0.25 * z[3].powi(3);
        (p, q, r)
    }

    fn cost(rule: &MixedAxisRule, theta: [f64; 2]) -> f64 {
        let excesses: Vec<f64> = rule.nodes.columns().into_iter().map(|z| excess(z, theta)).collect();
        -rule.posterior(&excesses).expect("posterior").value
    }

    fn assembled(rule: &MixedAxisRule, theta: [f64; 2]) -> (Array1<f64>, Array2<f64>) {
        let pairs = rho_pairs(2);
        let q = rule.node_count();
        let mut excesses = vec![0.0; q];
        // An infeasible node's rows are NaN: the assembly must never read them.
        let mut gradients = Array2::<f64>::from_elem((q, 2), f64::NAN);
        let mut seconds = Array2::<f64>::from_elem((q, pairs.len()), f64::NAN);
        for (node, z) in rule.nodes.columns().into_iter().enumerate() {
            excesses[node] = excess(z, theta);
            if !excesses[node].is_finite() {
                continue;
            }
            let (p, q_z, r) = polynomials(z);
            gradients[(node, 0)] = p + theta[1] * r;
            gradients[(node, 1)] = 2.0 * theta[1] * q_z + theta[0] * r;
            for (k, &(j, l)) in pairs.iter().enumerate() {
                seconds[(node, k)] = match (j, l) {
                    (0, 0) => 0.0,
                    (0, 1) => r,
                    _ => 2.0 * q_z,
                };
            }
        }
        let posterior = rule.posterior(&excesses).expect("posterior");
        mixed_axis_cost_second_order(&posterior, &gradients, &seconds, &pairs)
    }

    #[test]
    fn assembly_matches_central_differences_of_the_cost() {
        let rule = MixedAxisRule::new(4);
        let theta = [0.3, -0.4];
        let (gradient, hessian) = assembled(&rule, theta);
        let h = 1e-5;
        for j in 0..2 {
            let mut plus = theta;
            plus[j] += h;
            let mut minus = theta;
            minus[j] -= h;
            let fd = (cost(&rule, plus) - cost(&rule, minus)) / (2.0 * h);
            assert!(
                (fd - gradient[j]).abs() <= 1e-8 * fd.abs().max(1.0),
                "∂(−Ψ)/∂θ_{j}: assembled {} against FD {fd}",
                gradient[j]
            );
            let fd_row = (&assembled(&rule, plus).0 - &assembled(&rule, minus).0) / (2.0 * h);
            for l in 0..2 {
                assert!(
                    (fd_row[l] - hessian[(j, l)]).abs() <= 1e-8 * fd_row[l].abs().max(1.0),
                    "∂²(−Ψ)/∂θ_{j}∂θ_{l}: assembled {} against FD {}",
                    hessian[(j, l)],
                    fd_row[l]
                );
            }
        }
        assert!(hessian.iter().all(|v| v.is_finite()) && hessian[(0, 1)] != 0.0);
    }
}
