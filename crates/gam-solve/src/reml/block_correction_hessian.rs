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
use gam_math::probability::positive_log_sum_exp;
use super::*;
use gam_linalg::faer_ndarray::{fast_ab, fast_atb, fast_atv, fast_av, fast_xt_diag_y};
use ndarray::ShapeBuilder;

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

/// The one-axis rule a piece was integrated by, whose nodes its second-order pass
/// differentiates.
#[derive(Clone, Copy, Debug)]
pub(super) enum PieceRule<'r> {
    /// The standard-normal Gauss–Hermite rule of this order, transported onto the
    /// feasible interval of a truncated axis.
    GaussHermite { order: usize },
    /// The composite Gauss–Kronrod rule on the latched partition: fixed nodes in
    /// the oriented standardized axis, never truncated.
    Composite {
        nodes: &'r [gam_problem::laplace_sampler_contract::CompositeNode],
    },
}

/// Everything the Hessian reads besides the block target.
pub(super) struct BlockCorrectionHessianInputs<'x> {
    pub(super) curvature: RowCurvature,
    pub(super) axis_split: bool,
    /// The rule each piece was integrated by (one piece for `m = 1`).
    pub(super) piece_rules: &'x [PieceRule<'x>],
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

impl AxisMotion {
    /// The same motion along `−u`.
    fn reflected(self) -> Self {
        Self {
            y: -self.y,
            y_dot: self.y_dot.into_iter().map(|y| -y).collect(),
            y_ddot: self.y_ddot.into_iter().map(|y| -y).collect(),
        }
    }
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

    /// `Q diag(gains) Qᵀ M` through the block's eigensystem `Q`, every column of
    /// `M` in one pair of products: `H⁻¹ M` with `gains = 1/σ`, the eigenpair
    /// resolvent with `gains = 1/(λ − σ_q)`.
    fn spectral_columns(&self, gains: &Array1<f64>, m: &Array2<f64>) -> Array2<f64> {
        let mut coordinates = fast_atb(self.evecs, m);
        for (mut row, &gain) in coordinates.rows_mut().into_iter().zip(gains) {
            row *= gain;
        }
        fast_ab(self.evecs, &coordinates)
    }

    /// The per-row carriers of every ρ-pair as the columns of one `n × pairs`
    /// matrix, so each design contraction `Xᵀ(·)` is one product rather than
    /// one strided matrix–vector pass over `X` per pair.
    fn pair_rows(&self, n: usize, row: impl Fn(usize, usize, usize) -> Array1<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((n, self.pairs.len()).f());
        for (k, &(j, l)) in self.pairs.iter().enumerate() {
            out.column_mut(k).assign(&row(k, j, l));
        }
        out
    }

    fn mode_motion(&self) -> ModeMotion {
        let x = self.x();
        let n = x.nrows();
        let n_rho = self.target.lambdas.len();
        let score: Vec<Array1<f64>> = (0..n_rho)
            .map(|j| &self.target.penalty_scores[j] * self.target.lambdas[j])
            .collect();
        let inverse = self.evals.mapv(f64::recip);
        // H β̇_j = −λ_j S_j β̂.
        let beta_dot = -self.spectral_columns(&inverse, &columns_matrix(self.evecs.nrows(), &score));
        let eta_dot = matrix_columns(&fast_ab(x, &beta_dot));
        let beta_dot = matrix_columns(&beta_dot);
        // Differentiating H β̇_j = −λ_j S_j β̂ along ρ_l, with
        // Ḣ_l = λ_l S_l + Xᵀ diag(c ⊙ E_l) X.
        let mut rhs = fast_atb(
            x,
            &self.pair_rows(n, |_, j, l| self.c * &eta_dot[l] * &eta_dot[j]),
        );
        for (k, &(j, l)) in self.pairs.iter().enumerate() {
            let mut column = rhs.column_mut(k);
            column += &self.penalty_mv(l, beta_dot[j].view());
            column += &self.penalty_mv(j, beta_dot[l].view());
            if j == l {
                column += &score[j];
            }
        }
        let eta_ddot = matrix_columns(&-fast_ab(x, &self.spectral_columns(&inverse, &rhs)));
        let w_ddot = self
            .pairs
            .iter()
            .zip(&eta_ddot)
            .map(|(&(j, l), eta_jl)| self.c * eta_jl + &(self.d * &eta_dot[j] * &eta_dot[l]))
            .collect();
        ModeMotion {
            eta_dot,
            eta_ddot,
            w_ddot,
        }
    }

    /// The motion of the eigenpair in column `col` of the eigensystem.
    fn axis_motion(&self, col: usize, mode: &ModeMotion) -> AxisMotion {
        let x = self.x();
        let n = x.nrows();
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
        let xu = fast_av(x, &u);
        let s_u: Vec<Array1<f64>> = (0..n_rho).map(|j| self.penalty_mv(j, u)).collect();
        // b_j = Ḣ_j u.
        let mut b = fast_atb(
            x,
            &columns_matrix(
                n,
                &(0..n_rho)
                    .map(|j| self.c * &mode.eta_dot[j] * &xu)
                    .collect::<Vec<_>>(),
            ),
        );
        for (mut b_j, s_u_j) in b.columns_mut().into_iter().zip(&s_u) {
            b_j += s_u_j;
        }
        let u_dot = self.spectral_columns(&gains, &b);
        let xu_dot = matrix_columns(&fast_ab(x, &u_dot));
        let (b, u_dot) = (matrix_columns(&b), matrix_columns(&u_dot));
        let lambda_dot: Vec<f64> = b.iter().map(|b_j| u.dot(b_j)).collect();

        let inv_sqrt = lambda.sqrt().recip();
        let inv_32 = inv_sqrt / lambda;
        let inv_52 = inv_32 / lambda;
        let y = &xu * inv_sqrt;
        let y_dot: Vec<Array1<f64>> = (0..n_rho)
            .map(|j| &xu_dot[j] * inv_sqrt - &(&xu * (0.5 * inv_32 * lambda_dot[j])))
            .collect();
        // Ḧu + Ḣ_j u̇_l + Ḣ_l u̇_j − λ̇_l u̇_j − λ̇_j u̇_l, every pair at once.
        let mut v1 = fast_atb(
            x,
            &self.pair_rows(n, |k, j, l| {
                &mode.w_ddot[k] * &xu
                    + &(self.c * &mode.eta_dot[l] * &xu_dot[j])
                    + &(self.c * &mode.eta_dot[j] * &xu_dot[l])
            }),
        );
        let mut lambda_ddot = Vec::with_capacity(self.pairs.len());
        for (k, &(j, l)) in self.pairs.iter().enumerate() {
            // λ̈ = uᵀḦu + 2 u̇_jᵀ Ḣ_l u.
            let mut lambda_jl = (&mode.w_ddot[k] * &xu * &xu).sum() + 2.0 * u_dot[j].dot(&b[l]);
            let mut v1_jl = v1.column_mut(k);
            v1_jl += &self.penalty_mv(l, u_dot[j].view());
            v1_jl += &self.penalty_mv(j, u_dot[l].view());
            if j == l {
                lambda_jl += u.dot(&s_u[j]);
                v1_jl += &s_u[j];
            }
            v1_jl.scaled_add(-lambda_dot[l], &u_dot[j]);
            v1_jl.scaled_add(-lambda_dot[j], &u_dot[l]);
            lambda_ddot.push(lambda_jl);
        }
        // ü = R v1 − u (u̇_j·u̇_l); the second term keeps |u| = 1.
        let mut u_ddot = self.spectral_columns(&gains, &v1);
        for (k, &(j, l)) in self.pairs.iter().enumerate() {
            u_ddot.column_mut(k).scaled_add(-u_dot[j].dot(&u_dot[l]), &u);
        }
        let xu_ddot = fast_ab(x, &u_ddot);
        let y_ddot = self
            .pairs
            .iter()
            .enumerate()
            .map(|(k, &(j, l))| {
                let mut y_jl = &xu_ddot.column(k) * inv_sqrt;
                y_jl.scaled_add(-0.5 * inv_32 * lambda_dot[l], &xu_dot[j]);
                y_jl.scaled_add(-0.5 * inv_32 * lambda_dot[j], &xu_dot[l]);
                y_jl.scaled_add(
                    0.75 * inv_52 * lambda_dot[j] * lambda_dot[l] - 0.5 * inv_32 * lambda_ddot[k],
                    &xu,
                );
                y_jl
            })
            .collect();
        AxisMotion { y, y_dot, y_ddot }
    }
}

/// Vectors of length `n` as the columns of a matrix.
fn columns_matrix(n: usize, columns: &[Array1<f64>]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, columns.len()).f());
    for (mut target, column) in out.columns_mut().into_iter().zip(columns) {
        target.assign(column);
    }
    out
}

fn matrix_columns(m: &Array2<f64>) -> Vec<Array1<f64>> {
    m.columns().into_iter().map(|column| column.to_owned()).collect()
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

/// One node's row derivatives of the excess `F` at `s = Y τ`: `∂F/∂s`,
/// `∂²F/∂s²`, and the mode-motion derivatives `∂F/∂η̂`, `∂²F/∂η̂∂s`,
/// `∂²F/∂η̂²`.
struct NodeRows {
    f_s: Array1<f64>,
    f_ss: Array1<f64>,
    f_e: Array1<f64>,
    f_es: Array1<f64>,
    f_ee: Array1<f64>,
}

impl NodeRows {
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

/// The node posterior's weighted row sums that `E_p[∂²F]` contracts against
/// the motion.
///
/// With `ṡ_j = Ẏ_j τ + Y τ̇_j`, `τ̇_j = Σ_a σ_a ė_a,j` and
/// `τ̈_jl = Σ_a σ_a (ë_a,jl − e_a ė_a,j ė_a,l) + τ Σ_ab σ_a σ_b ė_a,j ė_b,l`,
/// a node's second derivative
///
/// ```text
///   E_jᵀ F_ee E_l + E_jᵀ F_es ṡ_l + E_lᵀ F_es ṡ_j + ṡ_jᵀ F_ss ṡ_l
///     + F_e·η̈_jl + f_s·(Ÿ_jl τ + Ẏ_j τ̇_l + Ẏ_l τ̇_j + Y τ̈_jl)
/// ```
///
/// is bilinear in its rows and polynomial in `(τ, σ)`, so its posterior mean
/// is the motion's quadratic forms under the rows' `p`-, `pτ`-, `pτ²`-,
/// `pσ_a`-, `pτσ_a`- and `pσ_aσ_b`-weighted sums: one pass over the rows per
/// node plus one per pair, instead of one per node and pair.
struct NodeMoments {
    ends: usize,
    /// `Σ p F_ee`.
    ee: Array1<f64>,
    /// `Σ p τ F_es` and, per end, `Σ p σ_a F_es`.
    es_tau: Array1<f64>,
    es_end: Vec<Array1<f64>>,
    /// `Σ p τ² F_ss`, per end `Σ p τ σ_a F_ss`, per end pair `Σ p σ_a σ_b F_ss`.
    ss_tau: Array1<f64>,
    ss_tau_end: Vec<Array1<f64>>,
    ss_end: Vec<Array1<f64>>,
    /// `Σ p F_e`.
    e: Array1<f64>,
    /// `Σ p τ f_s` and, per end, `Σ p σ_a f_s`.
    s_tau: Array1<f64>,
    s_end: Vec<Array1<f64>>,
    /// Per end `Σ p σ_a (f_s·Y)`, per end pair `Σ p τ σ_a σ_b (f_s·Y)`.
    sy_end: Vec<f64>,
    sy_tau_end: Vec<f64>,
}

impl NodeMoments {
    fn new(n: usize, ends: usize) -> Self {
        let rows = |count: usize| vec![Array1::<f64>::zeros(n); count];
        Self {
            ends,
            ee: Array1::zeros(n),
            es_tau: Array1::zeros(n),
            es_end: rows(ends),
            ss_tau: Array1::zeros(n),
            ss_tau_end: rows(ends),
            ss_end: rows(ends * ends),
            e: Array1::zeros(n),
            s_tau: Array1::zeros(n),
            s_end: rows(ends),
            sy_end: vec![0.0; ends],
            sy_tau_end: vec![0.0; ends * ends],
        }
    }

    /// Adds a node of posterior mass `prob` at `τ` with end sensitivities
    /// `sigma` (one per end, in the ends' order).
    fn add(&mut self, prob: f64, tau: f64, sigma: &[f64], rows: &NodeRows, f_s_y: f64) {
        self.ee.scaled_add(prob, &rows.f_ee);
        self.es_tau.scaled_add(prob * tau, &rows.f_es);
        self.ss_tau.scaled_add(prob * tau * tau, &rows.f_ss);
        self.e.scaled_add(prob, &rows.f_e);
        self.s_tau.scaled_add(prob * tau, &rows.f_s);
        for (a, &s_a) in sigma.iter().enumerate() {
            self.es_end[a].scaled_add(prob * s_a, &rows.f_es);
            self.ss_tau_end[a].scaled_add(prob * tau * s_a, &rows.f_ss);
            self.s_end[a].scaled_add(prob * s_a, &rows.f_s);
            self.sy_end[a] += prob * s_a * f_s_y;
            for (b, &s_b) in sigma.iter().enumerate() {
                let ab = a * self.ends + b;
                self.ss_end[ab].scaled_add(prob * s_a * s_b, &rows.f_ss);
                self.sy_tau_end[ab] += prob * tau * s_a * s_b * f_s_y;
            }
        }
    }

    /// `E_p[∂²F/∂ρ_j∂ρ_l]` for every pair.
    fn expected_second(
        &self,
        axis: &AxisMotion,
        mode: &ModeMotion,
        ends: &[AxisEnd],
        pairs: &[(usize, usize)],
    ) -> Array1<f64> {
        let n = self.ee.len();
        let eta_dot = columns_matrix(n, &mode.eta_dot);
        let y_dot = columns_matrix(n, &axis.y_dot);
        let ee = fast_xt_diag_y(&eta_dot, &self.ee, &eta_dot);
        let es = fast_xt_diag_y(&eta_dot, &self.es_tau, &y_dot);
        let ss = fast_xt_diag_y(&y_dot, &self.ss_tau, &y_dot);
        let along_y = |m: &Array1<f64>| m * &axis.y;
        // Per end, over j: E_jᵀ M_a Y, Ẏ_jᵀ M_a Y and Ẏ_j·M_a.
        let es_end: Vec<Array1<f64>> =
            self.es_end.iter().map(|m| fast_atv(&eta_dot, &along_y(m))).collect();
        let ss_tau_end: Vec<Array1<f64>> =
            self.ss_tau_end.iter().map(|m| fast_atv(&y_dot, &along_y(m))).collect();
        let s_end: Vec<Array1<f64>> = self.s_end.iter().map(|m| fast_atv(&y_dot, m)).collect();
        let ss_end: Vec<f64> = self.ss_end.iter().map(|m| along_y(m).dot(&axis.y)).collect();
        Array1::from_shape_fn(pairs.len(), |k| {
            let (j, l) = pairs[k];
            let mut value = ee[(j, l)]
                + es[(j, l)]
                + es[(l, j)]
                + ss[(j, l)]
                + self.e.dot(&mode.eta_ddot[k])
                + self.s_tau.dot(&axis.y_ddot[k]);
            for (a, end_a) in ends.iter().enumerate() {
                let (a_j, a_l) = (end_a.dot[j], end_a.dot[l]);
                value += a_l * (es_end[a][j] + ss_tau_end[a][j] + s_end[a][j])
                    + a_j * (es_end[a][l] + ss_tau_end[a][l] + s_end[a][l])
                    + (end_a.ddot[k] - end_a.value * a_j * a_l) * self.sy_end[a];
                for (b, end_b) in ends.iter().enumerate() {
                    let ab = a * self.ends + b;
                    value += a_j * end_b.dot[l] * (ss_end[ab] + self.sy_tau_end[ab]);
                }
            }
            value
        })
    }
}

/// A piece is integrated on the corrector's own rule: the standard-normal
/// Gauss–Hermite nodes `u_q`, transported onto the block's feasible interval
/// when the likelihood ends inside the Laplace Gaussian
/// ([`gam_math::quadrature::TruncatedNormalTransport`]), or the composite rule's
/// latched nodes `z_q` with their masses `w_q = w_K ψ` (never truncated), with
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
    piece_rule: PieceRule<'_>,
    axis: &AxisMotion,
    mode: &ModeMotion,
    geometry: &Geometry<'_, '_>,
) -> Result<PieceSecondOrder, EstimationError> {
    let n_rho = axis.y_dot.len();
    let pairs = &geometry.pairs;
    let lambda = piece_target.block_lambdas[0];
    let sqrt_lambda = lambda.sqrt();
    // Each node's position on the standardized axis and its log mass.
    let rule: Vec<(f64, f64)> = match piece_rule {
        PieceRule::GaussHermite { order } => {
            gam_math::quadrature::standard_normal_gauss_hermite_rule(order)
                .map_err(|error| {
                    EstimationError::InvalidInput(format!(
                        "#784 ρ-Hessian: Gauss–Hermite rule of order {order}: {error}"
                    ))
                })?
                .into_iter()
                .map(|(u, w)| (u, w.ln()))
                .collect()
        }
        PieceRule::Composite { nodes } => nodes.iter().map(|node| (node.z, node.ln_weight)).collect(),
    };

    let truncation = piece_target.axis_truncation();
    let cuts = truncation
        .as_deref()
        .map_or([None, None], |truncation| [truncation.lower(), truncation.upper()]);
    if matches!(piece_rule, PieceRule::Composite { .. }) && cuts.iter().any(Option::is_some) {
        crate::bail_invalid_estim!(
            "#784 ρ-Hessian: a composite piece is integrated on the whole axis, but its target \
             is truncated"
        );
    }
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
    for &(u, ln_w) in &rule {
        let node = match &transport {
            Some(transport) => {
                let tau = transport.transport(u).map_err(|error| {
                    EstimationError::InvalidInput(format!(
                        "#784 ρ-Hessian: transporting a node onto the feasible interval: {error}"
                    ))
                })?;
                let (lower, upper) = transport.endpoint_sensitivities(u, tau);
                (tau, [lower, upper], ln_w)
            }
            None => (u, [0.0, 0.0], ln_w),
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
    let log_node_weights: Vec<f64> = nodes.iter().map(|&(_, _, ln_w)| ln_w).collect();
    let log_norm = positive_log_sum_exp(&log_node_weights);
    let feasible: Vec<(f64, [f64; 2], f64, Array1<f64>)> = batched
        .into_iter()
        .zip(nodes.iter())
        .filter_map(|((excess, ngs), &(tau, sensitivity, ln_w))| match ngs {
            Some(ngs) if excess.is_finite() => Some((tau, sensitivity, ln_w - excess, ngs)),
            _ => None,
        })
        .collect();
    if feasible.is_empty() {
        crate::bail_invalid_estim!("#784 ρ-Hessian: every node of the piece's rule was infeasible");
    }
    let log_feasible_weights: Vec<f64> = feasible.iter().map(|(_, _, lw, _)| *lw).collect();
    let log_mass = positive_log_sum_exp(&log_feasible_weights);
    let value = log_mass - log_norm + log_mass_of_interval;

    let mut moments = NodeMoments::new(piece_target.eta_hat.len(), ends.len());
    let mut node_gradients: Vec<(f64, Array1<f64>)> = Vec::with_capacity(feasible.len());
    for (tau, sensitivity, lw, ngs) in feasible {
        let prob = (lw - log_mass).exp();
        let s = &axis.y * tau;
        let row = NodeRows::new(piece_target, curvature, geometry, &s, &ngs, &ngs_base)?;
        // The node `s = Y τ` moves with the axis and, on a truncated axis, with
        // the ends: ṡ_j = Ẏ_j τ + Y τ̇_j with τ̇_j = Σ_a σ_a ė_a,j.
        let sigma: Vec<f64> = ends.iter().map(|end| sensitivity[end.side]).collect();
        let tau_dot: Vec<f64> = (0..n_rho)
            .map(|j| ends.iter().zip(&sigma).map(|(end, s_a)| s_a * end.dot[j]).sum())
            .collect();
        let f_s_y = row.f_s.dot(&axis.y);
        let node_gradient = Array1::from_shape_fn(n_rho, |j| {
            row.f_e.dot(&mode.eta_dot[j]) + tau * row.f_s.dot(&axis.y_dot[j]) + tau_dot[j] * f_s_y
        });
        moments.add(prob, tau, &sigma, &row, f_s_y);
        node_gradients.push((prob, node_gradient));
    }
    let expected_second = moments.expected_second(axis, mode, &ends, pairs);
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
    // Each axis in the target's orientation (`γ_r > 0`): the pieces' nodes and cuts
    // are on `target.block_vecs`, which is `±` the eigensystem's column. The sign is
    // locally constant in ρ, so the motion flips with the axis.
    let axes: Vec<AxisMotion> = inputs
        .block_cols
        .iter()
        .enumerate()
        .map(|(r, &col)| {
            let motion = geometry.axis_motion(col, &mode);
            if target.block_vecs.column(r).dot(&inputs.evecs.column(col)) < 0.0 {
                motion.reflected()
            } else {
                motion
            }
        })
        .collect();

    let piece_count = if inputs.axis_split { m } else { 1 };
    if inputs.piece_rules.len() != piece_count {
        crate::bail_invalid_estim!(
            "#784 ρ-Hessian: {} piece rules for {piece_count} pieces",
            inputs.piece_rules.len()
        );
    }
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
            inputs.piece_rules[k],
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
            let row = NodeRows::new(target, curvature, geometry, &s, &score, &ngs_base)?;
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

    /// The node-moment contraction of `E_p[∂²F]` equals the per-node, per-pair
    /// sum it replaces, on a whole line and on an axis truncated at both ends.
    #[test]
    fn node_moments_match_the_per_node_expected_second() {
        let (n, n_rho) = (9usize, 3usize);
        let pairs = rho_pairs(n_rho);
        let row = |seed: f64| Array1::from_shape_fn(n, |i| ((i as f64 + 1.0) * seed).sin());
        let mode = ModeMotion {
            eta_dot: (0..n_rho).map(|j| row(0.71 + 0.13 * j as f64)).collect(),
            eta_ddot: (0..pairs.len()).map(|k| row(0.37 + 0.29 * k as f64)).collect(),
            w_ddot: (0..pairs.len()).map(|k| row(0.53 + 0.17 * k as f64)).collect(),
        };
        let axis = AxisMotion {
            y: row(1.19),
            y_dot: (0..n_rho).map(|j| row(0.43 + 0.31 * j as f64)).collect(),
            y_ddot: (0..pairs.len()).map(|k| row(0.61 + 0.23 * k as f64)).collect(),
        };
        let end = |side: usize, value: f64| AxisEnd {
            value,
            log_mass_gradient: 0.0,
            side,
            dot: (0..n_rho).map(|j| ((j + 3 * side) as f64 * 0.9 + 0.2).cos()).collect(),
            ddot: (0..pairs.len()).map(|k| ((k + 5 * side) as f64 * 0.7 + 0.4).sin()).collect(),
        };
        let nodes: Vec<(f64, f64, [f64; 2], NodeRows)> = (0..5)
            .map(|q| {
                let seed = 0.3 + 0.11 * q as f64;
                let rows = NodeRows {
                    f_s: row(seed + 0.01),
                    f_ss: row(seed + 0.02),
                    f_e: row(seed + 0.03),
                    f_es: row(seed + 0.04),
                    f_ee: row(seed + 0.05),
                };
                let prob = 0.1 + 0.05 * q as f64;
                let tau = (q as f64 * 1.7).sin() * 1.5;
                (prob, tau, [0.4 + 0.1 * q as f64, -0.3 + 0.2 * q as f64], rows)
            })
            .collect();
        for ends in [vec![], vec![end(0, -1.3), end(1, 0.9)]] {
            let mut moments = NodeMoments::new(n, ends.len());
            let mut reference = Array1::<f64>::zeros(pairs.len());
            for (prob, tau, sensitivity, rows) in &nodes {
                let sigma: Vec<f64> = ends.iter().map(|end| sensitivity[end.side]).collect();
                let tau_dot: Vec<f64> = (0..n_rho)
                    .map(|j| ends.iter().zip(&sigma).map(|(end, s_a)| s_a * end.dot[j]).sum())
                    .collect();
                let s_dot: Vec<Array1<f64>> = (0..n_rho)
                    .map(|j| &axis.y_dot[j] * *tau + &(&axis.y * tau_dot[j]))
                    .collect();
                let f_s_y = rows.f_s.dot(&axis.y);
                for (k, &(j, l)) in pairs.iter().enumerate() {
                    let (e_j, e_l) = (&mode.eta_dot[j], &mode.eta_dot[l]);
                    let mut tau_ddot = 0.0;
                    for (a, s_a) in ends.iter().zip(&sigma) {
                        tau_ddot += s_a * a.ddot[k] - a.value * s_a * a.dot[j] * a.dot[l];
                        for (b, s_b) in ends.iter().zip(&sigma) {
                            tau_ddot += tau * s_a * s_b * a.dot[j] * b.dot[l];
                        }
                    }
                    let f_s_s_ddot = tau * rows.f_s.dot(&axis.y_ddot[k])
                        + tau_dot[l] * rows.f_s.dot(&axis.y_dot[j])
                        + tau_dot[j] * rows.f_s.dot(&axis.y_dot[l])
                        + tau_ddot * f_s_y;
                    reference[k] += prob
                        * ((&rows.f_ee * e_j).dot(e_l)
                            + (&rows.f_es * e_j).dot(&s_dot[l])
                            + (&rows.f_es * e_l).dot(&s_dot[j])
                            + (&rows.f_ss * &s_dot[j]).dot(&s_dot[l])
                            + rows.f_e.dot(&mode.eta_ddot[k])
                            + f_s_s_ddot);
                }
                moments.add(*prob, *tau, &sigma, rows, f_s_y);
            }
            let contracted = moments.expected_second(&axis, &mode, &ends, &pairs);
            let scale = reference.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            for k in 0..pairs.len() {
                assert!(
                    (contracted[k] - reference[k]).abs() <= 1e-12 * scale,
                    "{} ends, pair {:?}: contracted {} vs per-node {}",
                    ends.len(),
                    pairs[k],
                    contracted[k],
                    reference[k],
                );
            }
        }
    }
}
