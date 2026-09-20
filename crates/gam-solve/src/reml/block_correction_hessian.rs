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
    if axis_split && fourth_derivative_rule(&response, inverse_link).is_none() {
        return Err(format!(
            "the mixed-axis term Φ of a split {block_dim}-axis block needs ∂⁴W/∂η⁴, and the \
             observed curvature of {response:?} under {inverse_link:?} is not a combination of \
             at most two powers or exponentials of η, the forms whose ∂⁴W/∂η⁴ follows from \
             ∂W/∂η and ∂³W/∂η³"
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

/// The shape of a row's observed curvature `W(η)` that fixes `∂⁴W/∂η⁴`.
///
/// Up to a constant, `W = a·g(η) + b·h(η)` with row coefficients `a, b`
/// (prior weight, dispersion, `y`) that do not depend on `η`. The pair
/// `c = ∂W/∂η`, `e = ∂³W/∂η³` then fixes `(a, b)`, and so `f = ∂⁴W/∂η⁴`; a
/// constant term drops out of all three.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CurvatureShape {
    /// Binomial-logit: `W = w μ'(η)`, differentiated by the inverse-link jet.
    LogitJet,
    /// `W` constant in `η`.
    Constant,
    /// `g = η^p`, `h = η^q`: `f = α c/η³ + β e/η`.
    Powers(f64, Option<f64>),
    /// `g = e^{rη}`, `h = e^{sη}`: `f = α c + β e`.
    Rates(f64, Option<f64>),
}

/// `ψ(η)` below is each family's negative log-likelihood up to `w/φ`, and
/// `W = ψ''`: e.g. inverse-Gaussian `1/μ²` has `ψ = yη/2 − √η`, so
/// `W ∝ η^{−3/2}`, and Gaussian-inverse has `ψ = (y − 1/η)²/2`, so
/// `W = 3η^{−4} − 2yη^{−3}`.
fn fourth_derivative_rule(
    response: &ResponseFamily,
    inverse_link: &InverseLink,
) -> Option<CurvatureShape> {
    use CurvatureShape::{Constant, LogitJet, Powers, Rates};
    let InverseLink::Standard(link) = inverse_link else {
        return None;
    };
    Some(match (response, link) {
        (ResponseFamily::Binomial, StandardLink::Logit) => LogitJet,
        // ψ = μ − y ln μ.
        (ResponseFamily::Poisson, StandardLink::Log) => Rates(1.0, None),
        (ResponseFamily::Poisson, StandardLink::Identity | StandardLink::Sqrt) => {
            Powers(-2.0, None)
        }
        (ResponseFamily::Poisson, StandardLink::Inverse) => Powers(-3.0, Some(-2.0)),
        // ψ = y/μ + ln μ.
        (ResponseFamily::Gamma, StandardLink::Log) => Rates(-1.0, None),
        (ResponseFamily::Gamma, StandardLink::Inverse) => Powers(-2.0, None),
        (ResponseFamily::Gamma, StandardLink::Identity) => Powers(-3.0, Some(-2.0)),
        (ResponseFamily::Gamma, StandardLink::Sqrt) => Powers(-4.0, Some(-2.0)),
        (ResponseFamily::Gamma, StandardLink::InverseSquared) => Powers(-1.5, Some(-2.0)),
        // ψ = y/(2μ²) − 1/μ.
        (ResponseFamily::InverseGaussian, StandardLink::InverseSquared) => Powers(-1.5, None),
        (ResponseFamily::InverseGaussian, StandardLink::Inverse) => Constant,
        (ResponseFamily::InverseGaussian, StandardLink::Log) => Rates(-2.0, Some(-1.0)),
        (ResponseFamily::InverseGaussian, StandardLink::Identity) => Powers(-4.0, Some(-3.0)),
        (ResponseFamily::InverseGaussian, StandardLink::Sqrt) => Powers(-6.0, Some(-4.0)),
        // ψ = (y − μ)²/2.
        (ResponseFamily::Gaussian, StandardLink::Inverse) => Powers(-4.0, Some(-3.0)),
        (ResponseFamily::Gaussian, StandardLink::InverseSquared) => Powers(-2.5, Some(-3.0)),
        (ResponseFamily::Gaussian, StandardLink::Log) => Rates(2.0, Some(1.0)),
        (ResponseFamily::Gaussian, StandardLink::Sqrt) => Powers(2.0, None),
        _ => return None,
    })
}

/// `(α, β)` with `g⁗ = α g′ + β g‴` on each basis function, from its
/// `(g‴/g′, g⁗/g′)` in reduced units: `((p−1)(p−2), (p−1)(p−2)(p−3))` on
/// `η^p` (whose powers of `η` the `c/η³`, `e/η` scaling absorbs), and
/// `(r², r³)` on `e^{rη}`. One basis function takes `α = 0`; two solve the
/// pair `α + β t_k = q_k`.
fn fourth_derivative_coefficients(first: (f64, f64), second: Option<(f64, f64)>) -> (f64, f64) {
    let (t1, q1) = first;
    match second {
        None if t1 == 0.0 => (0.0, 0.0),
        None => (0.0, q1 / t1),
        Some((t2, q2)) => {
            let beta = (q1 - q2) / (t1 - t2);
            (q1 - beta * t1, beta)
        }
    }
}

/// `∂⁴W/∂η⁴` per row for a split block's `Φ`, from `c = ∂W/∂η` and
/// `e = ∂³W/∂η³` through the curvature's shape.
pub(super) fn curvature_fourth_derivative(
    pirls_result: &PirlsResult,
    inverse_link: &InverseLink,
    prior_weights: &Array1<f64>,
    c: &Array1<f64>,
    e: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    let response = reml_spec(&pirls_result.likelihood).response;
    let Some(shape) = fourth_derivative_rule(&response, inverse_link) else {
        crate::bail_invalid_estim!(
            "#784 mixed-axis ρ-Hessian: the curvature of {response:?} under {inverse_link:?} \
             has no ∂⁴W/∂η⁴ from its η-derivatives"
        );
    };
    Ok(fourth_from_shape(shape, pirls_result.final_eta.view(), prior_weights, c, e))
}

fn fourth_from_shape(
    shape: CurvatureShape,
    eta: ArrayView1<'_, f64>,
    prior_weights: &Array1<f64>,
    c: &Array1<f64>,
    e: &Array1<f64>,
) -> Array1<f64> {
    let power = |p: f64| ((p - 1.0) * (p - 2.0), (p - 1.0) * (p - 2.0) * (p - 3.0));
    let rate = |r: f64| (r * r, r * r * r);
    let n = eta.len();
    match shape {
        CurvatureShape::LogitJet => Array1::from_shape_fn(n, |i| {
            prior_weights[i] * crate::mixture_link::logit_inverse_link_jet5(eta[i]).d5
        }),
        CurvatureShape::Constant => Array1::zeros(n),
        CurvatureShape::Powers(p, q) => {
            let (alpha, beta) = fourth_derivative_coefficients(power(p), q.map(power));
            Array1::from_shape_fn(n, |i| alpha * c[i] / eta[i].powi(3) + beta * e[i] / eta[i])
        }
        CurvatureShape::Rates(r, s) => {
            let (alpha, beta) = fourth_derivative_coefficients(rate(r), s.map(rate));
            Array1::from_shape_fn(n, |i| alpha * c[i] + beta * e[i])
        }
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
    let w_mode = &piece_target.weights_obs;
    let log_norm = log_sum_exp(nodes.iter().map(|&(_, _, ln_w)| ln_w));
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
    let log_mass = log_sum_exp(feasible.iter().map(|(_, _, lw, _)| *lw));
    let value = log_mass - log_norm + log_mass_of_interval;

    let mut moments = NodeMoments::new(piece_target.eta_hat.len(), ends.len());
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
        // the ends: ṡ_j = Ẏ_j τ + Y τ̇_j with τ̇_j = Σ_a σ_a ė_a,j.
        let sigma: Vec<f64> = ends.iter().map(|end| sensitivity[end.side]).collect();
        let tau_dot: Vec<f64> = (0..n_rho)
            .map(|j| ends.iter().zip(&sigma).map(|(end, s_a)| s_a * end.dot[j]).sum())
            .collect();
        let f_s_y = f_s.dot(&axis.y);
        let node_gradient = Array1::from_shape_fn(n_rho, |j| {
            f_e.dot(&mode.eta_dot[j]) + tau * f_s.dot(&axis.y_dot[j]) + tau_dot[j] * f_s_y
        });
        let rows = NodeRows {
            f_s,
            f_ss,
            f_e,
            f_es,
            f_ee,
        };
        moments.add(prob, tau, &sigma, &rows, f_s_y);
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

    /// Every listed curvature shape reproduces the family's own observed
    /// curvature: `f` from `(c, e)` matches the second difference of
    /// `∂²W/∂η²` as `compute_observed_hessian_curvature_arrays` evaluates it.
    #[test]
    fn curvature_shapes_give_the_observed_fourth_derivative() {
        let families = [
            ResponseFamily::Poisson,
            ResponseFamily::Gamma,
            ResponseFamily::InverseGaussian,
            ResponseFamily::Gaussian,
        ];
        let links = [
            StandardLink::Identity,
            StandardLink::Log,
            StandardLink::Sqrt,
            StandardLink::Inverse,
            StandardLink::InverseSquared,
        ];
        let prior = Array1::from_elem(1, 1.7);
        let h = 2e-3;
        let mut checked = 0;
        for response in &families {
            for &link in &links {
                let inverse_link = InverseLink::Standard(link);
                let Some(shape) = fourth_derivative_rule(response, &inverse_link) else {
                    continue;
                };
                let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    response.clone(),
                    inverse_link.clone(),
                ));
                if !crate::pirls::supports_observed_hessian_curvature_for_likelihood(
                    &likelihood,
                    &inverse_link,
                ) {
                    // Poisson-log reads `W = w e^η` directly (`RowCurvature::PoissonLog`).
                    assert!(
                        matches!((response, link), (ResponseFamily::Poisson, StandardLink::Log)),
                        "{response:?}/{link:?} has a curvature shape but no observed curvature",
                    );
                    let e = Array1::from_elem(1, 1.7 * 0.8_f64.exp());
                    let f = fourth_from_shape(shape, Array1::from_elem(1, 0.8).view(), &prior, &e, &e);
                    assert!((f[0] - e[0]).abs() <= 1e-15 * e[0]);
                    checked += 1;
                    continue;
                }
                // A count for Poisson, a positive real otherwise.
                let y = if matches!(response, ResponseFamily::Poisson) { 2.0 } else { 1.3 };
                let curvature_at = |eta: f64| {
                    crate::pirls::compute_observed_hessian_curvature_arrays(
                        &likelihood,
                        &inverse_link,
                        &Array1::from_elem(1, eta),
                        Array1::from_elem(1, y).view(),
                        &Array1::zeros(1),
                        prior.view(),
                    )
                    .expect("observed curvature")
                };
                // Richardson-extrapolated central differences of `∂²W/∂η²`:
                // O(h⁴) truncation with a step whose rounding stays far below it.
                let differences = |eta0: f64, step: f64| {
                    let (_, _, d0) = curvature_at(eta0);
                    let (_, _, d_plus) = curvature_at(eta0 + step);
                    let (_, _, d_minus) = curvature_at(eta0 - step);
                    (
                        (d_plus[0] - d_minus[0]) / (2.0 * step),
                        (d_plus[0] - 2.0 * d0[0] + d_minus[0]) / (step * step),
                    )
                };
                for eta0 in [0.45, 0.8, 1.6] {
                    let (w0, c, _) = curvature_at(eta0);
                    let (e_h, f_h) = differences(eta0, h);
                    let (e_half, f_half) = differences(eta0, 0.5 * h);
                    let e = Array1::from_elem(1, (4.0 * e_half - e_h) / 3.0);
                    let f_fd = (4.0 * f_half - f_h) / 3.0;
                    let f = fourth_from_shape(shape, Array1::from_elem(1, eta0).view(), &prior, &c, &e)[0];
                    // The evaluated `∂²W/∂η²` carries rounding on the scale of
                    // `W/η²`, which the second difference divides by `(h/2)²`.
                    let rounding =
                        4096.0 * f64::EPSILON * (w0[0] / (eta0 * eta0)).abs() / (0.25 * h * h);
                    assert!(
                        (f - f_fd).abs() <= 1e-7 * f_fd.abs().max(1.0) + rounding,
                        "{response:?}/{link:?} at η={eta0}: shape f={f}, differenced f={f_fd}",
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 18, "every tabulated non-logit shape is checked");
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
