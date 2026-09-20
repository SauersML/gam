//! Layer 3: the honest full-conformal set, whose fitting map re-selects the
//! smoothing strength by REML on every augmented data set.
//!
//! # The fitting map
//!
//! The model has one penalty `S` (the stored `Sλ = λ̂S`; the stored scale is
//! irrelevant, see below). For a candidate response `z` at the test row `x_*`
//! the map fits the AUGMENTED rows `{(x_i, y_i)} ∪ {(x_*, z)}` with the strength
//! `ρ̂(z)` that globally minimizes the profiled Gaussian REML criterion
//!
//! ```text
//!   V(ρ; z) = c·ln D(ρ; z) + ln|M + e^ρ Sλ| − |P|·ρ ,   c = n + 1 − nullity(S)
//! ```
//!
//! over the resolvability domain of the augmented Gram against `Sλ`
//! (`rho_domain::resolvability_interval`), and scores every row with that fit.
//! `ρ` is measured relative to the stored `Sλ`, so `ρ = 0` is the fit the user
//! trained. Every ingredient — the augmented Gram `M = X_augᵀX_aug`, `D`, the
//! domain — is a symmetric function of the `n + 1` rows, and the global
//! minimizer set does not depend on how rows are ordered, so the map treats the
//! test row exactly like a training row: exchangeability alone gives
//! `P(y_* ∈ C_α) ≥ 1 − α` in finite samples. (Rescaling the stored `Sλ` shifts
//! `ρ` by a constant and moves the domain with it, so the set does not depend on
//! the stored `λ̂`. The basis, and a fixed ridge the normal matrix may carry,
//! come from the training covariates; the map re-selects the scale of that
//! fixed penalty, which is a symmetric function of the responses.)
//!
//! # One factorization, then closed forms
//!
//! One Cholesky of Layer 1's matrix `B = M + Sλ` and one symmetric
//! eigendecomposition of `L⁻¹SλL⁻ᵀ` give a basis `V'` with `V'ᵀMV' = I` and
//! `V'ᵀSλV' = diag(s_k)`. With `t(z) = V'ᵀX_augᵀy_aug = t₀ + t₁z`,
//! `γ_k = logistic(ρ + ln s_k)` and `δ_k = 1 − γ_k`:
//!
//! ```text
//!   D(ρ; z) = R(z) + Σ_k γ_k t_k(z)² ,   R = unpenalized residual sum of squares
//!   V(ρ; z) = c·ln D − Σ_{k∈P} ln γ_k   (+ constants)
//!   r_i(ρ; z) = y_i − Σ_k (X_aug V')_ik δ_k t_k(z)
//! ```
//!
//! so `D`, the stationarity numerator and every residual are polynomials in `z`
//! with coefficients that are monotone or unimodal in `ρ`.
//!
//! # Bound, then local refit
//!
//! `z` ranges over three charts that cover the extended line with a
//! homogeneous coordinate (`η`, `zη` affine in the chart variable `s ∈ [−1,1]`
//! or `[0,1]`), so the tails `|z| → ∞` are compact cells rather than cut off.
//! On a `z`-cell the engine keeps a TUBE of `ρ`-boxes that provably contains
//! every global minimizer `ρ̂(z)`:
//!
//! - a box on which `∂V/∂ρ > 0` (resp. `< 0`) for every `z` in the cell holds
//!   its only candidate at its left (right) endpoint, and that endpoint is a
//!   minimizer only if it is the end of the domain: inside the domain a
//!   minimizer is stationary, so the box holds none;
//! - a box whose REML value exceeds a surviving candidate's for every `z` in the
//!   cell holds no global minimizer.
//!
//! Both tests are exact sign statements about quadratics in the chart variable
//! (monotonicity of `γ`, `δ`, `L` in `ρ`), decided with their rounding band.
//! On the same cell every rank comparison `e_i ≥ e_*` is enclosed over the tube
//! (a centered form in `δ`), so the cell is a member, a non-member, or
//! undecided. A box whose own `ρ`-width keeps it undecided is bisected in `ρ`;
//! an undecided cell is bisected in `z`. Three undecided states are final and
//! the cell is kept, so the returned set is a superset of the honest set, equal
//! to it up to `f64` resolution:
//!
//! - a comparison blurred by rounding — even exact `δ` over the box leaves
//!   `e_i² − e_*²` within the rounding of its own evaluation;
//! - candidates that disagree while their REML values are tied within the
//!   rounding band, so which one is the global minimizer is not determined
//!   (the criterion separates strengths near a minimizer only to
//!   `√(band/curvature)`, so a breakpoint is located to that resolution);
//! - a cell at the floor of `f64` resolution (a breakpoint).
//!
//! There is no grid: the cells and boxes that get split are exactly the ones
//! the data make undecided.
//!
//! At every finite endpoint of the returned set one cold 1-D REML refit runs
//! through [`gam_solve::rho_optimizer::OuterProblem`] — the seed-path engine
//! every other smoothing selection uses — and is checked against the tube: a
//! refit whose criterion beats every tube candidate by more than the rounding
//! band contradicts the bound, and the row is refused rather than reported.
//!
//! # What is refused
//!
//! Anything the map above does not cover returns the frozen-ρ Layer-1 set with a
//! typed [`ConformalRefusal`], never a silent answer: several penalties
//! (their relative strengths were selected without the test row, so re-selecting
//! one common scale is not the fitting map), an unknown penalty structure (a
//! payload written before the penalty count was persisted), a singular augmented
//! Gram, an undefined REML criterion, and a refit the bound cannot account for.

use faer::Side;
use ndarray::{Array1, Array2};

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};

use super::{
    ConformalInterval, ExactGaussianFullConformal, FullConformalSet, required_dominating_count,
    response_solve_growth, solve_lower_triangular, solve_lower_triangular_transposed,
    validate_inputs,
};

/// Why a row's set is the frozen-ρ set rather than the honest one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConformalRefusal {
    /// More than one smoothing parameter: their ratios were selected on the
    /// training rows alone, so the map is not symmetric in the augmented row.
    MultiPenalty,
    /// The penalty count was not persisted (a payload older than this field).
    UnknownPenaltyStructure,
    /// A direction of the coefficient space carries no augmented data curvature
    /// (the augmented Gram is singular on the penalty's range).
    AugmentedGramSingular,
    /// The REML criterion is undefined: no residual degrees of freedom, or an
    /// exact fit with zero penalized residual sum of squares.
    RemlUndefined,
    /// A cold REML refit found a criterion value below every candidate the bound
    /// retained.
    RefitOutsideTube,
    /// The outer engine could not complete a local refit.
    RefitFailed,
}

impl ConformalRefusal {
    pub fn label(self) -> &'static str {
        match self {
            ConformalRefusal::MultiPenalty => "refused:multi_penalty",
            ConformalRefusal::UnknownPenaltyStructure => "refused:unknown_penalty_structure",
            ConformalRefusal::AugmentedGramSingular => "refused:augmented_gram_singular",
            ConformalRefusal::RemlUndefined => "refused:reml_undefined",
            ConformalRefusal::RefitOutsideTube => "refused:refit_outside_tube",
            ConformalRefusal::RefitFailed => "refused:refit_failed",
        }
    }

    fn code(self) -> i32 {
        match self {
            ConformalRefusal::MultiPenalty => -1,
            ConformalRefusal::UnknownPenaltyStructure => -2,
            ConformalRefusal::AugmentedGramSingular => -3,
            ConformalRefusal::RemlUndefined => -4,
            ConformalRefusal::RefitOutsideTube => -5,
            ConformalRefusal::RefitFailed => -6,
        }
    }
}

/// What a row's full-conformal set guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConformalCertificate {
    /// The fitting map has no smoothing parameter to re-select, so the exact
    /// Layer-1 set at the stored penalty IS the honest set.
    ExactFrozen,
    /// The set of the REML re-selecting map, from the bound plus local refits.
    HonestRefit,
    /// The frozen-ρ set, carrying no finite-sample guarantee, and why.
    Refused(ConformalRefusal),
}

impl ConformalCertificate {
    /// `exact_frozen`, `honest_refit`, or `refused:<reason>`.
    pub fn label(self) -> &'static str {
        match self {
            ConformalCertificate::ExactFrozen => "exact_frozen",
            ConformalCertificate::HonestRefit => "honest_refit",
            ConformalCertificate::Refused(reason) => reason.label(),
        }
    }

    /// Numeric code for column output: `0` exact_frozen, `1` honest_refit,
    /// negative for a refusal (`-1` multi_penalty, `-2` unknown_penalty_structure,
    /// `-3` augmented_gram_singular, `-4` reml_undefined, `-5` refit_outside_tube, `-6` refit_failed).
    pub fn code(self) -> i32 {
        match self {
            ConformalCertificate::ExactFrozen => 0,
            ConformalCertificate::HonestRefit => 1,
            ConformalCertificate::Refused(reason) => reason.code(),
        }
    }

    /// Whether the set carries the finite-sample coverage guarantee.
    pub fn is_guaranteed(self) -> bool {
        !matches!(self, ConformalCertificate::Refused(_))
    }
}

/// Work one row's set cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HonestConformalCost {
    /// Cholesky factorizations of a `p × p` normal matrix.
    pub factorizations: usize,
    /// Symmetric `p × p` eigendecompositions.
    pub eigendecompositions: usize,
    /// Cold REML refits run through the outer engine.
    pub extra_refits: usize,
    /// Candidate-response cells the bound examined.
    pub z_cells: usize,
}

/// One row's full-conformal set with its certificate and cost.
#[derive(Clone, Debug)]
pub struct HonestFullConformal {
    pub set: FullConformalSet,
    pub certificate: ConformalCertificate,
    pub cost: HonestConformalCost,
    /// The frozen plug-in mean `x_*ᵀ(XᵀX + Sλ)⁻¹Xᵀy` — the fixed point of the
    /// augmented fit, where the test residual vanishes.
    pub plug_in_mean: f64,
}

// ── Enclosure arithmetic ───────────────────────────────────────────────────
//
// Every quantity the bound decides on is a polynomial of degree ≤ 2 in the
// chart variable `s`, with `|s| ≤ 1` on every chart. Each coefficient carries
// the absolute sum of the summands it was formed from; the rounding band of a
// value is Wilkinson's growth for the whole computation times that magnitude
// sum (Higham, Lemma 3.1), which `|s| ≤ 1` bounds by the plain sum.

#[derive(Clone, Copy, Debug)]
struct Affine {
    c: [f64; 2],
    m: [f64; 2],
}

#[derive(Clone, Copy, Debug)]
struct Quad {
    c: [f64; 3],
    m: [f64; 3],
}

impl Affine {
    fn exact(c0: f64, c1: f64) -> Self {
        Affine {
            c: [c0, c1],
            m: [c0.abs(), c1.abs()],
        }
    }

    fn value(&self, s: f64) -> f64 {
        self.c[0] + self.c[1] * s
    }

    fn add_scaled(&mut self, k: f64, other: &Affine) {
        for j in 0..2 {
            self.c[j] += k * other.c[j];
            self.m[j] += k.abs() * other.m[j];
        }
    }

    fn mul(&self, other: &Affine) -> Quad {
        Quad {
            c: [
                self.c[0] * other.c[0],
                self.c[0] * other.c[1] + self.c[1] * other.c[0],
                self.c[1] * other.c[1],
            ],
            m: [
                self.m[0] * other.m[0],
                self.m[0] * other.m[1] + self.m[1] * other.m[0],
                self.m[1] * other.m[1],
            ],
        }
    }

    fn magnitude(&self) -> f64 {
        self.m[0] + self.m[1]
    }

    /// `max |value|` over `[s1, s2]` (attained at an endpoint).
    fn abs_max(&self, s1: f64, s2: f64) -> f64 {
        self.value(s1).abs().max(self.value(s2).abs())
    }
}

impl Quad {
    fn zero() -> Self {
        Quad {
            c: [0.0; 3],
            m: [0.0; 3],
        }
    }

    fn add_scaled(&mut self, k: f64, other: &Quad) {
        for j in 0..3 {
            self.c[j] += k * other.c[j];
            self.m[j] += k.abs() * other.m[j];
        }
    }

    fn scaled(&self, k: f64) -> Quad {
        let mut out = Quad::zero();
        out.add_scaled(k, self);
        out
    }

    fn value(&self, s: f64) -> f64 {
        self.c[0] + s * (self.c[1] + s * self.c[2])
    }

    fn magnitude(&self) -> f64 {
        self.m[0] + self.m[1] + self.m[2]
    }

    /// Exact `(min, max)` of the computed quadratic over `[s1, s2]`: the
    /// endpoints and, when interior, the vertex.
    fn range(&self, s1: f64, s2: f64) -> (f64, f64) {
        let mut lo = self.value(s1).min(self.value(s2));
        let mut hi = self.value(s1).max(self.value(s2));
        if self.c[2] != 0.0 {
            let vertex = -0.5 * self.c[1] / self.c[2];
            if vertex > s1 && vertex < s2 {
                let v = self.value(vertex);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        (lo, hi)
    }
}

/// `(γ, δ) = (logistic(x), logistic(−x))`, each formed without cancellation.
fn logistic_pair(x: f64) -> (f64, f64) {
    let e = (-x.abs()).exp();
    let small = e / (1.0 + e);
    let large = 1.0 / (1.0 + e);
    if x >= 0.0 { (large, small) } else { (small, large) }
}

/// `ln(1 + eˣ)` without overflow or cancellation.
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

impl Affine {
    /// `a·η + b·ζ` for the `z`-affine `a + b·z` on a chart where `η` and `ζ = zη`
    /// are affine in the chart variable.
    fn homogenize(&self, chart: &Chart) -> Affine {
        let [a, b] = self.c;
        let [ma, mb] = self.m;
        Affine {
            c: [a * chart.e[0] + b * chart.f[0], a * chart.e[1] + b * chart.f[1]],
            m: [
                ma * chart.e[0].abs() + mb * chart.f[0].abs(),
                ma * chart.e[1].abs() + mb * chart.f[1].abs(),
            ],
        }
    }
}

// ── The augmented problem in the penalty's eigenbasis ─────────────────────

/// Everything the bound decides on, from one Cholesky of `B = M + Sλ` and one
/// symmetric eigendecomposition. Affine quantities are in `z` (`c[0] + c[1]·z`).
struct Basis {
    n: usize,
    /// Coefficient count of the design.
    p: usize,
    /// `ln s_k` for the penalized modes `k ∈ P`.
    ln_s: Vec<f64>,
    /// `τ_k(z) = t0_k + t1_k·z` for `k ∈ P`.
    tau: Vec<Affine>,
    /// Residuals of the full unpenalized projection, training rows then the
    /// test row: `R(z) = Σ g_i(z)²`.
    projection_residuals: Vec<Affine>,
    /// Residuals with only the unpenalized modes fitted, training rows.
    rows: Vec<Affine>,
    /// The same for the test row.
    star: Affine,
    /// `(X_aug V')_ik` for `k ∈ P`: training rows, then the test row last.
    loadings: Array2<f64>,
    /// `n + 1 − nullity(S)`, the profiled criterion's `ln D` weight.
    c: f64,
    /// The resolvability domain of the augmented Gram against `Sλ`.
    domain: (f64, f64),
    /// Wilkinson growth charged against every magnitude sum.
    growth: f64,
}

/// `A + Aᵀ` halved: removes the asymmetry two triangular solves leave.
fn symmetrized(matrix: &Array2<f64>) -> Array2<f64> {
    let p = matrix.nrows();
    let mut out = matrix.clone();
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}

impl Basis {
    /// Build the basis, or the refusal the data force. `Err` is a hard failure
    /// (dimensions, a non-SPD `B`) that the frozen engine would report too.
    fn build(
        x: &Array2<f64>,
        y: &Array1<f64>,
        s_lambda: &Array2<f64>,
        x_star: &Array1<f64>,
    ) -> Result<Result<Basis, ConformalRefusal>, String> {
        let n = x.nrows();
        let p = x.ncols();
        let mut gram = x.t().dot(x);
        for i in 0..p {
            for j in 0..p {
                gram[[i, j]] += x_star[i] * x_star[j];
            }
        }
        let chol = (&gram + s_lambda)
            .cholesky(Side::Lower)
            .map_err(|e| format!("honest full conformal: augmented normal matrix not SPD: {e:?}"))?;
        let lower = chol.lower_triangular();
        let w = solve_lower_triangular(&lower, s_lambda);
        let congruence = symmetrized(&solve_lower_triangular(&lower, &w.t().to_owned()));
        let (_, u) = congruence.eigh(Side::Lower).map_err(|e| {
            format!("honest full conformal: penalty eigendecomposition failed: {e:?}")
        })?;
        let mut v = solve_lower_triangular_transposed(&lower, &u);

        // Rayleigh quotients against the matrices themselves, not the
        // eigenvalues: a mode the data barely see has `ν_k` formed from its own
        // rows rather than as `1 − λ_k`.
        let sv = s_lambda.dot(&v);
        let xv_raw = x.dot(&v);
        let xsv_raw = v.t().dot(x_star);
        let mu: Vec<f64> = (0..p).map(|k| v.column(k).dot(&sv.column(k))).collect();
        let nu: Vec<f64> = (0..p)
            .map(|k| xv_raw.column(k).dot(&xv_raw.column(k)) + xsv_raw[k] * xsv_raw[k])
            .collect();
        let nu_threshold =
            gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(&nu);
        if nu.iter().any(|&value| !(value > nu_threshold)) {
            return Ok(Err(ConformalRefusal::AugmentedGramSingular));
        }
        // |P| is the rank the fit's own penalty pseudo-logdet counts on `Sλ`;
        // the penalized modes are the |P| largest Rayleigh quotients.
        let (s_evals, _) = s_lambda.eigh(Side::Lower).map_err(|e| {
            format!("honest full conformal: penalty eigendecomposition failed: {e:?}")
        })?;
        let s_evals = s_evals.to_vec();
        let s_threshold =
            gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(&s_evals);
        let rank = s_evals.iter().filter(|&&e| e > s_threshold).count();
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| mu[b].total_cmp(&mu[a]));
        let mut penalized = vec![false; p];
        for &k in order.iter().take(rank) {
            if !(mu[k] > 0.0) {
                return Ok(Err(ConformalRefusal::AugmentedGramSingular));
            }
            penalized[k] = true;
        }
        let nullity = p - rank;
        if n + 1 <= nullity {
            return Ok(Err(ConformalRefusal::RemlUndefined));
        }

        // V' = V/√ν: V'ᵀMV' = I, V'ᵀSλV' = diag(s_k).
        for k in 0..p {
            let scale = nu[k].sqrt();
            v.column_mut(k).mapv_inplace(|entry| entry / scale);
        }
        let xv = x.dot(&v);
        let t1 = v.t().dot(x_star);
        let mut t0 = vec![0.0; p];
        let mut t0_mag = vec![0.0; p];
        let mut t1_mag = vec![0.0; p];
        for k in 0..p {
            for i in 0..n {
                t0[k] += xv[[i, k]] * y[i];
                t0_mag[k] += (xv[[i, k]] * y[i]).abs();
            }
            for j in 0..p {
                t1_mag[k] += (v[[j, k]] * x_star[j]).abs();
            }
        }

        // Residual affines: `y_aug − Σ_{k ∈ modes} (X_aug V')_k τ_k(z)`.
        let residual = |response: Affine, loading: &dyn Fn(usize) -> f64, all: bool| {
            let mut out = response;
            for k in 0..p {
                if all || !penalized[k] {
                    let mode = Affine {
                        c: [t0[k], t1[k]],
                        m: [t0_mag[k], t1_mag[k]],
                    };
                    out.add_scaled(-loading(k), &mode);
                }
            }
            out
        };
        let mut projection_residuals = Vec::with_capacity(n + 1);
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let response = Affine::exact(y[i], 0.0);
            let loading = |k: usize| xv[[i, k]];
            projection_residuals.push(residual(response, &loading, true));
            rows.push(residual(response, &loading, false));
        }
        let test_loading = |k: usize| t1[k];
        let test_response = Affine::exact(0.0, 1.0);
        projection_residuals.push(residual(test_response, &test_loading, true));
        let star = residual(test_response, &test_loading, false);

        let modes: Vec<usize> = (0..p).filter(|&k| penalized[k]).collect();
        let ln_s: Vec<f64> = modes.iter().map(|&k| (mu[k] / nu[k]).ln()).collect();
        let tau = modes
            .iter()
            .map(|&k| Affine {
                c: [t0[k], t1[k]],
                m: [t0_mag[k], t1_mag[k]],
            })
            .collect();
        let mut loadings = Array2::<f64>::zeros((n + 1, modes.len()));
        for (col, &k) in modes.iter().enumerate() {
            for i in 0..n {
                loadings[[i, col]] = xv[[i, k]];
            }
            loadings[[n, col]] = t1[k];
        }
        // The resolvability domain of the augmented Gram against `Sλ`: its
        // generalized eigenvalues on the penalty's range are `1/s_k`.
        let inverse_strengths: Vec<f64> = ln_s.iter().map(|&l| (-l).exp()).collect();
        let domain = gam_solve::estimate::rho_domain::coordinate_domain(
            gam_solve::estimate::rho_domain::resolvability_interval(&inverse_strengths),
            None,
        );
        Ok(Ok(Basis {
            n,
            p,
            ln_s,
            tau,
            projection_residuals,
            rows,
            star,
            loadings,
            c: (n + 1 - nullity) as f64,
            domain,
            growth: response_solve_growth(n + 1, p),
        }))
    }

    /// `(γ_k, δ_k)` at `ρ` for every penalized mode.
    fn shrinkage(&self, rho: f64) -> Vec<(f64, f64)> {
        self.ln_s.iter().map(|&l| logistic_pair(rho + l)).collect()
    }

    /// `L(ρ) = Σ_P softplus(−ρ − ln s_k)`, the criterion's determinant part.
    fn log_det_part(&self, rho: f64) -> f64 {
        self.ln_s.iter().map(|&l| softplus(-rho - l)).sum()
    }

    /// The frozen (`ρ = 0`) Layer-1 engine in this basis — no second
    /// factorization.
    fn frozen_engine(&self) -> ExactGaussianFullConformal {
        let n = self.n;
        let delta: Vec<f64> = self.shrinkage(0.0).iter().map(|&(_, d)| d).collect();
        let mut u = Array1::<f64>::zeros(n + 1);
        let mut w = Array1::<f64>::zeros(n + 1);
        for i in 0..=n {
            let base = if i < n { self.rows[i] } else { self.star };
            let mut value = base.c;
            for (col, tau) in self.tau.iter().enumerate() {
                let k = self.loadings[[i, col]] * delta[col];
                value[0] -= k * tau.c[0];
                value[1] -= k * tau.c[1];
            }
            u[i] = value[0];
            w[i] = value[1];
        }
        ExactGaussianFullConformal { u, w, n }
    }
}

// ── Charts of the extended candidate line ─────────────────────────────────

/// A chart `z(s) = (f0 + f1·s)/(e0 + e1·s)`: the inner chart is the window
/// `z_c + σ·s`, `s ∈ [−1, 1]`; the upper and lower charts are `z_c ± σ/s`,
/// `s ∈ [0, 1]`, with `s = 0` the point at infinity. Every quantity the bound
/// reads is homogeneous of even degree in `(η, zη)`, so the tails are cells
/// like any other.
#[derive(Clone, Copy, Debug)]
struct Chart {
    e: [f64; 2],
    f: [f64; 2],
}

impl Chart {
    fn charts(z_c: f64, sigma: f64) -> [Chart; 3] {
        [
            Chart {
                e: [1.0, 0.0],
                f: [z_c, sigma],
            },
            Chart {
                e: [0.0, 1.0],
                f: [sigma, z_c],
            },
            Chart {
                e: [0.0, 1.0],
                f: [-sigma, z_c],
            },
        ]
    }

    /// The cells each chart starts from.
    fn root_cells(index: usize) -> &'static [(f64, f64)] {
        if index == 0 {
            &[(-1.0, 0.0), (0.0, 1.0)]
        } else {
            &[(0.0, 1.0)]
        }
    }

    fn z_at(&self, s: f64) -> f64 {
        let eta = self.e[0] + self.e[1] * s;
        let zeta = self.f[0] + self.f[1] * s;
        if eta == 0.0 {
            if zeta > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY }
        } else {
            zeta / eta
        }
    }
}

/// One chart's homogenized quantities.
struct ChartData {
    chart: Chart,
    tau: Vec<Affine>,
    tau_sq: Vec<Quad>,
    /// `η²·R(z)`.
    projection_rss: Quad,
    rows: Vec<Affine>,
    star: Affine,
}

impl ChartData {
    fn new(basis: &Basis, chart: Chart) -> Self {
        let tau: Vec<Affine> = basis.tau.iter().map(|t| t.homogenize(&chart)).collect();
        let tau_sq = tau.iter().map(|t| t.mul(t)).collect();
        let mut projection_rss = Quad::zero();
        for g in &basis.projection_residuals {
            let g = g.homogenize(&chart);
            projection_rss.add_scaled(1.0, &g.mul(&g));
        }
        ChartData {
            chart,
            tau,
            tau_sq,
            projection_rss,
            rows: basis.rows.iter().map(|r| r.homogenize(&chart)).collect(),
            star: basis.star.homogenize(&chart),
        }
    }

    /// `η²·D(ρ; z)` as a quadratic in `s`, from the shrinkage at `ρ`.
    fn rss(&self, shrinkage: &[(f64, f64)]) -> Quad {
        let mut out = self.projection_rss;
        for (sq, &(gamma, _)) in self.tau_sq.iter().zip(shrinkage) {
            out.add_scaled(gamma, sq);
        }
        out
    }

    /// `Σ_P w_k·τ_k²` for per-mode weights.
    fn weighted_tau_sq(&self, weights: impl Iterator<Item = f64>) -> Quad {
        let mut out = Quad::zero();
        for (sq, weight) in self.tau_sq.iter().zip(weights) {
            out.add_scaled(weight, sq);
        }
        out
    }
}

/// The criterion at one point of one chart, with its two `ρ`-derivatives and
/// its rounding band. `None` where `D` is not positive.
fn criterion_at(basis: &Basis, data: &ChartData, rho: f64, s: f64) -> Option<(f64, f64, f64, f64)> {
    let shrinkage = basis.shrinkage(rho);
    let rss = data.rss(&shrinkage);
    let d = rss.value(s);
    if !(d > 0.0) {
        return None;
    }
    let mut first = 0.0;
    let mut second = 0.0;
    let mut delta_sum = 0.0;
    let mut curvature_sum = 0.0;
    for (tau, &(gamma, delta)) in data.tau.iter().zip(&shrinkage) {
        let t2 = tau.value(s).powi(2);
        first += gamma * delta * t2;
        second += gamma * delta * (delta - gamma) * t2;
        delta_sum += delta;
        curvature_sum += gamma * delta;
    }
    let log_det = basis.log_det_part(rho);
    let c = basis.c;
    let value = c * d.ln() + log_det;
    let grad = c * first / d - delta_sum;
    let hess = c * (second / d - (first / d).powi(2)) + curvature_sum;
    let band = basis.growth * (c * d.ln().abs() + log_det + c * rss.magnitude() / d);
    Some((value, grad, hess, band))
}

// ── The tube: ρ-boxes holding every global minimizer on a cell ────────────

type RhoBox = (f64, f64);

fn quad_lower(q: &Quad, s1: f64, s2: f64, growth: f64) -> f64 {
    q.range(s1, s2).0 - growth * q.magnitude()
}

fn quad_upper(q: &Quad, s1: f64, s2: f64, growth: f64) -> f64 {
    q.range(s1, s2).1 + growth * q.magnitude()
}

/// `(min, max)` of `γδ = logistic(x)·logistic(−x)` over `x ∈ [x1, x2]`: it is
/// unimodal with its peak `1/4` at `x = 0`.
fn curvature_range(x1: f64, x2: f64) -> (f64, f64) {
    let at = |x: f64| {
        let (g, d) = logistic_pair(x);
        g * d
    };
    let (a, b) = (at(x1), at(x2));
    let hi = if x1 <= 0.0 && x2 >= 0.0 { 0.25 } else { a.max(b) };
    (a.min(b), hi)
}

/// Sorted, with every degenerate box another box contains removed.
fn deduplicated(mut boxes: Vec<RhoBox>) -> Vec<RhoBox> {
    boxes.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1)));
    boxes.dedup();
    // A point box `(a, a)` is contained in another box exactly when an earlier
    // box reaches `a` or the next box starts at `a`.
    let mut out = Vec::with_capacity(boxes.len());
    let mut reach = f64::NEG_INFINITY;
    for (i, &(a, b)) in boxes.iter().enumerate() {
        let covered = reach >= a || boxes.get(i + 1).is_some_and(|&(c, _)| c == a);
        if a < b || !covered {
            out.push((a, b));
        }
        reach = reach.max(b);
    }
    out
}

/// The best of the boxes' endpoints and midpoints and the seeds, by the
/// criterion at the chart point `s`.
fn best_candidate(
    basis: &Basis,
    data: &ChartData,
    s: f64,
    boxes: &[RhoBox],
    seeds: &[f64],
) -> Option<f64> {
    let (lo_dom, hi_dom) = basis.domain;
    let mut best: Option<(f64, f64)> = None;
    let candidates = boxes
        .iter()
        .flat_map(|&(a, b)| [a, 0.5 * (a + b), b])
        .chain(seeds.iter().copied().filter(|&r| r >= lo_dom && r <= hi_dom));
    for rho in candidates {
        if let Some((value, ..)) = criterion_at(basis, data, rho, s) {
            if best.is_none_or(|(_, v)| value < v) {
                best = Some((rho, value));
            }
        }
    }
    best.map(|(rho, _)| rho)
}

/// One pruning pass over the cell `[s1, s2]`. Returns the surviving boxes and
/// the best candidate strength at the cell's midpoint.
fn prune(
    basis: &Basis,
    data: &ChartData,
    s1: f64,
    s2: f64,
    boxes: Vec<RhoBox>,
    seeds: &[f64],
) -> (Vec<RhoBox>, Option<f64>) {
    let growth = basis.growth;
    let c = basis.c;
    // Monotonicity: where ∂V/∂ρ has one sign for every ρ in the box and every
    // z in the cell, the box's only candidate is the endpoint V decreases to,
    // and that endpoint is a minimizer only as an end of the domain: inside
    // it, a minimizer is stationary.
    let (lo_dom, hi_dom) = basis.domain;
    let mut shrunk = Vec::with_capacity(boxes.len());
    for (r1, r2) in boxes {
        let sh1 = basis.shrinkage(r1);
        let sh2 = basis.shrinkage(r2);
        let rss1 = data.rss(&sh1);
        if quad_lower(&rss1, s1, s2, growth) > 0.0 {
            let rss2 = data.rss(&sh2);
            let ranges: Vec<(f64, f64)> = basis
                .ln_s
                .iter()
                .map(|&l| curvature_range(r1 + l, r2 + l))
                .collect();
            let delta_sum1: f64 = sh1.iter().map(|&(_, d)| d).sum();
            let delta_sum2: f64 = sh2.iter().map(|&(_, d)| d).sum();
            let mut rising = data
                .weighted_tau_sq(ranges.iter().map(|&(lo, _)| lo))
                .scaled(c);
            rising.add_scaled(-delta_sum1, &rss2);
            if quad_lower(&rising, s1, s2, growth) > 0.0 {
                if r1 <= lo_dom {
                    shrunk.push((r1, r1));
                }
                continue;
            }
            let mut falling = data
                .weighted_tau_sq(ranges.iter().map(|&(_, hi)| hi))
                .scaled(c);
            falling.add_scaled(-delta_sum2, &rss1);
            if quad_upper(&falling, s1, s2, growth) < 0.0 {
                if r2 >= hi_dom {
                    shrunk.push((r2, r2));
                }
                continue;
            }
        }
        shrunk.push((r1, r2));
    }

    // Dominance: a box whose criterion exceeds one candidate's at every z of
    // the cell holds no global minimizer. The candidate is the best of the
    // boxes' endpoints and midpoints and the seeds at the cell's midpoint.
    let Some(rho_best) = best_candidate(basis, data, 0.5 * (s1 + s2), &shrunk, seeds) else {
        return (deduplicated(shrunk), None);
    };
    let survivors = shrunk
        .into_iter()
        .filter(|&(r1, r2)| !dominated(basis, data, s1, s2, rho_best, r1, r2))
        .collect();
    (deduplicated(survivors), Some(rho_best))
}

/// Whether the candidate `rho_best` beats every `ρ ∈ [r1, r2]` at every `z` of
/// the cell by more than the rounding band: `V ≥ c·ln D(r1) + L(r2)` on the box.
fn dominated(basis: &Basis, data: &ChartData, s1: f64, s2: f64, rho_best: f64, r1: f64, r2: f64) -> bool {
    let (growth, c) = (basis.growth, basis.c);
    let log_det_best = basis.log_det_part(rho_best);
    let log_det2 = basis.log_det_part(r2);
    let kappa = ((log_det2 - log_det_best - growth * (log_det2 + log_det_best)) / c).exp();
    let mut test = data.rss(&basis.shrinkage(rho_best));
    test.add_scaled(-kappa, &data.rss(&basis.shrinkage(r1)));
    quad_upper(&test, s1, s2, growth) < 0.0
}

/// Bisect every flagged box and prune the result; the seeds become the best
/// candidate the pass found.
fn split_and_prune(
    basis: &Basis,
    data: &ChartData,
    s1: f64,
    s2: f64,
    tube: &[RhoBox],
    flagged: &[bool],
    seeds: &mut Vec<f64>,
) -> Vec<RhoBox> {
    let mut split = Vec::with_capacity(2 * tube.len());
    for (&(a, b), &flag) in tube.iter().zip(flagged) {
        let mid = 0.5 * (a + b);
        if flag && mid > a && mid < b {
            split.push((a, mid));
            split.push((mid, b));
        } else {
            split.push((a, b));
        }
    }
    let (after, best) = prune(basis, data, s1, s2, split, seeds);
    if let Some(rho) = best {
        *seeds = vec![rho];
    }
    after
}

// ── Rank decisions over a cell and its tube ───────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Member,
    NonMember,
    /// Neither, with whether the rounding band alone kept it undecided (a
    /// breakpoint at the arithmetic's resolution, or an exact tie).
    Undecided { band_limited: bool },
}

/// One box's membership verdict over the cell: with the rounding band,
/// without it, and at the box's centre alone (no `δ` radius, no band), whether
/// only comparisons blurred at the arithmetic's resolution keep it undecided,
/// and whether some comparison it leaves open is held open by the box's own
/// `δ`-width more than by the cell.
#[derive(Clone, Copy, Debug)]
struct BoxVerdict {
    banded: Option<bool>,
    exact: Option<bool>,
    centre: Option<bool>,
    resolution_limited: bool,
    /// A resolvable open comparison whose enclosure's `δ`-width part exceeds
    /// the range `m·p` covers over the cell. Halving the cell shrinks that
    /// range but leaves the width part, so no `z`-split alone can close the
    /// comparison; halving the box shrinks the width part.
    width_dominated: bool,
}

impl BoxVerdict {
    /// Undecided on its own enclosure, not at resolution, while its centre is
    /// decided: the box's `ρ`-width, not the cell, keeps it open, so bisecting
    /// the box can decide it.
    fn rho_limited(&self) -> bool {
        self.width_limited() && self.centre.is_some()
    }

    /// Undecided by a comparison that a narrower cell or box can resolve.
    fn width_limited(&self) -> bool {
        self.exact.is_none() && !self.resolution_limited
    }

    /// Undecided short of resolution with an open comparison that its
    /// `ρ`-width dominates: bisecting the box, not only the cell, is what
    /// narrows it. Near a breakpoint the centre is undecided too, so
    /// `rho_limited` alone never narrows the tube there, and every `z`-half
    /// inherits the same wide boxes (gam#3338).
    fn width_dominated(&self) -> bool {
        self.width_limited() && self.width_dominated
    }
}

/// The cell's verdict from its boxes' verdicts.
fn verdict(boxes: &[BoxVerdict]) -> Verdict {
    if boxes.is_empty() {
        // No box can hold a minimizer only if rounding broke the bound; the
        // cell is kept, never decided from nothing.
        return Verdict::Undecided { band_limited: true };
    }
    if boxes.iter().any(BoxVerdict::width_limited) {
        return Verdict::Undecided {
            band_limited: false,
        };
    }
    if boxes.iter().all(|b| b.banded == Some(true)) {
        Verdict::Member
    } else if boxes.iter().all(|b| b.banded == Some(false)) {
        Verdict::NonMember
    } else {
        // Undecided either because rounding blurs a comparison every box
        // otherwise agrees on, or because the boxes disagree.
        Verdict::Undecided {
            band_limited: !disagreement(boxes),
        }
    }
}

/// Whether two boxes reach opposite exact verdicts.
fn disagreement(boxes: &[BoxVerdict]) -> bool {
    let member = boxes.iter().any(|b| b.exact == Some(true));
    let non_member = boxes.iter().any(|b| b.exact == Some(false));
    member && non_member
}

/// How one comparison `|e_i| ≥ |e_*|` resolves over a cell and one ρ-box.
#[derive(Clone, Copy)]
enum Comparison {
    Dominating,
    NotDominating,
    Uncertain,
}

/// Each box's membership verdict for `z ∈ [s1, s2]`.
///
/// On a box `δ_k = δ̄_k + ε_k` with `|ε_k| ≤ h_k` (`δ` is monotone in `ρ`), so
/// `m = e_i − e_*` and `p = e_i + e_*` are affine in `s` up to radii
/// `Σ_k |ℓ_ik ∓ ℓ_*k|·h_k·max|τ_k|`, and `m·p` — whose sign is the comparison —
/// is a quadratic up to the centered-form remainder.
fn box_verdicts(
    basis: &Basis,
    data: &ChartData,
    s1: f64,
    s2: f64,
    tube: &[RhoBox],
    required: usize,
) -> Vec<BoxVerdict> {
    let growth = basis.growth;
    let n = basis.n;
    let modes = basis.ln_s.len();
    let tau_max: Vec<f64> = data
        .tau
        .iter()
        .map(|t| t.abs_max(s1, s2) + growth * t.magnitude())
        .collect();
    let mut out = Vec::with_capacity(tube.len());
    for &(r1, r2) in tube {
        let low = basis.shrinkage(r2);
        let high = basis.shrinkage(r1);
        // Per-mode radii of `δ_k·τ_k`: the box's `δ`-width, and the rounding
        // of the shrinkage and of `τ_k`.
        let mut mid = Vec::with_capacity(modes);
        let mut width = Vec::with_capacity(modes);
        let mut rounding = Vec::with_capacity(modes);
        for k in 0..modes {
            let (d_lo, d_hi) = (low[k].1, high[k].1);
            mid.push(0.5 * (d_lo + d_hi));
            width.push(0.5 * (d_hi - d_lo) * tau_max[k]);
            rounding.push(growth * d_hi * tau_max[k]);
        }
        let mut star = data.star;
        for k in 0..modes {
            star.add_scaled(-basis.loadings[[n, k]] * mid[k], &data.tau[k]);
        }
        // (banded, exact, centre): dominating, and undecided counts; and the
        // exact slot's undecided comparisons that rounding does not blur.
        let mut dominating = [0usize; 3];
        let mut uncertain = [0usize; 3];
        let mut resolvable = 0usize;
        let mut width_dominated = false;
        for i in 0..n {
            let mut row = data.rows[i];
            let (mut width_m, mut width_p) = (0.0, 0.0);
            let (mut rounding_m, mut rounding_p) = (0.0, 0.0);
            for k in 0..modes {
                let (li, ls) = (basis.loadings[[i, k]], basis.loadings[[n, k]]);
                row.add_scaled(-li * mid[k], &data.tau[k]);
                width_m += (li - ls).abs() * width[k];
                width_p += (li + ls).abs() * width[k];
                rounding_m += (li - ls).abs() * rounding[k];
                rounding_p += (li + ls).abs() * rounding[k];
            }
            let mut m = row;
            m.add_scaled(-1.0, &star);
            let mut p = row;
            p.add_scaled(1.0, &star);
            let (lo, hi) = m.mul(&p).range(s1, s2);
            let (m_max, p_max) = (m.abs_max(s1, s2), p.abs_max(s1, s2));
            let centred = |rad_m: f64, rad_p: f64| m_max * rad_p + p_max * rad_m + rad_m * rad_p;
            let (rad_m, rad_p) = (width_m + rounding_m, width_p + rounding_p);
            let remainder = centred(rad_m, rad_p);
            let band = growth * (m.magnitude() + rad_m) * (p.magnitude() + rad_p);
            for (slot, (remainder, band)) in [(remainder, band), (remainder, 0.0), (0.0, 0.0)]
                .into_iter()
                .enumerate()
            {
                match compare(lo - remainder, hi + remainder, band) {
                    Comparison::Dominating => dominating[slot] += 1,
                    Comparison::Uncertain => uncertain[slot] += 1,
                    Comparison::NotDominating => {}
                }
            }
            // Blurred: even exact `δ` over the box leaves `m·p` within the
            // rounding of its own evaluation, so no refinement decides it.
            let width_part = centred(width_m, width_p);
            let blurred = lo.abs().max(hi.abs()) + width_part <= band + (remainder - width_part);
            if !blurred && matches!(compare(lo - remainder, hi + remainder, 0.0), Comparison::Uncertain) {
                resolvable += 1;
                width_dominated |= width_part > hi - lo;
            }
        }
        let decide = |slot: usize| {
            if dominating[slot] >= required {
                Some(true)
            } else if dominating[slot] + uncertain[slot] < required {
                Some(false)
            } else {
                None
            }
        };
        let exact = decide(1);
        out.push(BoxVerdict {
            banded: decide(0),
            exact,
            centre: decide(2),
            resolution_limited: exact.is_none() && resolvable == 0,
            width_dominated,
        });
    }
    out
}

fn compare(lo: f64, hi: f64, band: f64) -> Comparison {
    if lo - band >= 0.0 {
        Comparison::Dominating
    } else if hi + band < 0.0 {
        Comparison::NotDominating
    } else {
        Comparison::Uncertain
    }
}

/// A bound on `|∂V/∂ρ|` over the box and the cell, or `None` where `D` is not
/// bounded away from zero. `∂V/∂ρ = c·Σ_k γ_kδ_k τ_k² / D − Σ_k δ_k`, with `D`
/// rising and `δ` falling in `ρ` and `γδ` unimodal.
fn gradient_bound(basis: &Basis, data: &ChartData, s1: f64, s2: f64, r1: f64, r2: f64) -> Option<f64> {
    gradient_range(basis, data, s1, s2, r1, r2).map(|(lo, hi)| lo.abs().max(hi.abs()))
}

/// An enclosure of `∂V/∂ρ` over the box and the cell, or `None` where `D` is
/// not bounded away from zero.
fn gradient_range(basis: &Basis, data: &ChartData, s1: f64, s2: f64, r1: f64, r2: f64) -> Option<(f64, f64)> {
    let growth = basis.growth;
    let (sh1, sh2) = (basis.shrinkage(r1), basis.shrinkage(r2));
    let d_lo = quad_lower(&data.rss(&sh1), s1, s2, growth);
    if !(d_lo > 0.0) {
        return None;
    }
    let d_hi = quad_upper(&data.rss(&sh2), s1, s2, growth);
    let ranges: Vec<(f64, f64)> = basis
        .ln_s
        .iter()
        .map(|&l| curvature_range(r1 + l, r2 + l))
        .collect();
    let f_lo = quad_lower(&data.weighted_tau_sq(ranges.iter().map(|&(lo, _)| lo)), s1, s2, growth).max(0.0);
    let f_hi = quad_upper(&data.weighted_tau_sq(ranges.iter().map(|&(_, hi)| hi)), s1, s2, growth);
    let delta_sum = |sh: &[(f64, f64)]| sh.iter().map(|&(_, d)| d).sum::<f64>();
    let (fit_lo, shrink_hi) = (basis.c * f_lo / d_hi, delta_sum(&sh1));
    let (fit_hi, shrink_lo) = (basis.c * f_hi / d_lo, delta_sum(&sh2));
    Some((
        fit_lo - shrink_hi - growth * (fit_lo + shrink_hi),
        fit_hi - shrink_lo + growth * (fit_hi + shrink_lo),
    ))
}

/// Which boxes are tied with the best candidate at the arithmetic's
/// resolution: `|V(ρ; z) − V(ρ_best; z)|` is within the rounding band for
/// every `ρ` in the box and every `z` in the cell. No refinement can tell a
/// tied box's candidates from the best — the dominance test needs a gap wider
/// than the same band — so which of them is the global minimizer is not
/// determined in `f64`.
fn ties(basis: &Basis, data: &ChartData, s1: f64, s2: f64, tube: &[RhoBox], rho_best: Option<f64>) -> Vec<bool> {
    let Some(rho_best) = rho_best else {
        return vec![false; tube.len()];
    };
    // A box survives dominance while its criterion is within one rounding
    // band of the best's somewhere on the cell; the tie test evaluates the
    // same gap again, so its band is the dominance band plus its own.
    let growth = 2.0 * basis.growth;
    let c = basis.c;
    let rss_best = data.rss(&basis.shrinkage(rho_best));
    let log_det_best = basis.log_det_part(rho_best);
    let within = |q: &Quad| q.range(s1, s2).0 + growth * q.magnitude() >= 0.0;
    tube.iter()
        .map(|&(r1, r2)| {
            // Mean value form about the box's midpoint `m`:
            // `|V(ρ) − V(m)| ≤ max|∂V/∂ρ|·(r2 − r1)/2`, second order in the
            // box width near a minimizer.
            let Some(gradient) = gradient_bound(basis, data, s1, s2, r1, r2) else {
                return false;
            };
            let m = 0.5 * (r1 + r2);
            let spread = gradient * 0.5 * (r2 - r1);
            let rss_m = data.rss(&basis.shrinkage(m));
            let log_det_m = basis.log_det_part(m);
            let band = growth * (log_det_m + log_det_best);
            let above = ((log_det_best - log_det_m + band - spread) / c).exp();
            let mut upper = rss_best.scaled(above);
            upper.add_scaled(-1.0, &rss_m);
            let below = ((log_det_best - log_det_m - band + spread) / c).exp();
            let mut lower = rss_m;
            lower.add_scaled(-below, &rss_best);
            within(&upper) && within(&lower)
        })
        .collect()
}

/// The tube, its boxes' verdicts, and which boxes are tied at resolution, once
/// refining `ρ` can no longer help: every untied box whose own `ρ`-width keeps
/// it undecided — its centre is decided, or an open comparison's enclosure is
/// wider through the box's `δ`-width than `m·p` varies over the cell — is
/// bisected and the tube pruned again. On a cell too narrow to bisect, every
/// untied box undecided short of resolution is bisected, and so are untied
/// boxes that disagree, so a candidate that is not a minimizer is
/// pruned away. A tied box is never split: no split can separate its
/// candidates from the best. Every intermediate tube is valid, so the rule
/// only decides the work.
fn settle(
    basis: &Basis,
    data: &ChartData,
    s1: f64,
    s2: f64,
    mut tube: Vec<RhoBox>,
    seeds: &mut Vec<f64>,
    required: usize,
) -> (Vec<RhoBox>, Vec<BoxVerdict>, Vec<bool>) {
    let mid = 0.5 * (s1 + s2);
    let floor = !(mid > s1 && mid < s2);
    loop {
        let mut verdicts = box_verdicts(basis, data, s1, s2, &tube, required);
        let rho_best = best_candidate(basis, data, mid, &tube, seeds);
        let tied = ties(basis, data, s1, s2, &tube, rho_best);
        for (v, &tie) in verdicts.iter_mut().zip(&tied) {
            // A tied box whose `ρ`-width alone keeps it undecided is at
            // resolution: its candidates cannot be told from the best, so
            // bisecting it only multiplies them.
            if tie && v.rho_limited() {
                v.resolution_limited = true;
            }
        }
        let disagree = floor && disagreement(&verdicts);
        // On an undecided cell, a box whose midpoint the best candidate
        // dominates survives only through its own width: bisecting it prunes it.
        let open = !matches!(verdict(&verdicts), Verdict::Member | Verdict::NonMember);
        let flagged: Vec<bool> = verdicts
            .iter()
            .zip(&tied)
            .zip(&tube)
            .map(|((v, &tie), &(a, b))| {
                let dominance_limited = open
                    && rho_best.is_some_and(|best| {
                        let m = 0.5 * (a + b);
                        dominated(basis, data, s1, s2, best, m, m)
                    });
                !tie && (v.rho_limited()
                    || v.width_dominated()
                    || (floor && v.width_limited())
                    || (disagree && v.exact.is_some())
                    || dominance_limited)
            })
            .collect();
        let splittable = tube.iter().zip(&flagged).any(|(&(a, b), &flag)| {
            let mid = 0.5 * (a + b);
            flag && mid > a && mid < b
        });
        if !splittable {
            return (tube, verdicts, tied);
        }
        tube = split_and_prune(basis, data, s1, s2, &tube, &flagged, seeds);
    }
}

/// The pruned tube of the whole domain at a cell, and its seeds.
fn root_tube(basis: &Basis, data: &ChartData, s1: f64, s2: f64) -> (Vec<RhoBox>, Vec<f64>) {
    let mut seeds = vec![0.0];
    let (tube, best) = prune(basis, data, s1, s2, vec![basis.domain], &seeds);
    if let Some(rho) = best {
        seeds = vec![rho];
    }
    (tube, seeds)
}

// ── Branch and bound over the candidate line ──────────────────────────────

struct Cell {
    chart: usize,
    s1: f64,
    s2: f64,
    tube: Vec<RhoBox>,
    seeds: Vec<f64>,
}

/// A retained interval of candidate values, with the chart point of each
/// endpoint.
struct Retained {
    lo: f64,
    hi: f64,
    lo_point: (usize, f64),
    hi_point: (usize, f64),
}

/// The honest set as maximal intervals, and the number of cells examined.
fn branch_and_bound(basis: &Basis, charts: &[ChartData], required: usize) -> (Vec<Retained>, usize) {
    let mut stack = Vec::new();
    for (index, data) in charts.iter().enumerate() {
        for &(s1, s2) in Chart::root_cells(index) {
            let (tube, seeds) = root_tube(basis, data, s1, s2);
            stack.push(Cell {
                chart: index,
                s1,
                s2,
                tube,
                seeds,
            });
        }
    }
    let mut cells = 0;
    let mut kept: Vec<Retained> = Vec::new();
    while let Some(mut cell) = stack.pop() {
        cells += 1;
        let data = &charts[cell.chart];
        let (tube, verdicts, box_ties) = settle(
            basis,
            data,
            cell.s1,
            cell.s2,
            std::mem::take(&mut cell.tube),
            &mut cell.seeds,
            required,
        );
        let tied = !box_ties.is_empty() && box_ties.iter().all(|&t| t);
        let mid = 0.5 * (cell.s1 + cell.s2);
        let decision = match verdict(&verdicts) {
            // Tied candidates that disagree: the honest map is not determined
            // at the arithmetic's resolution here, so the cell is kept. A box
            // the cell's width keeps open is still bisected: a tie says which
            // strength wins, not where the rank changes.
            Verdict::Undecided { .. } if tied && !verdicts.iter().any(BoxVerdict::width_limited) => {
                Verdict::Undecided { band_limited: true }
            }
            decision => decision,
        };
        let keep = match decision {
            Verdict::Member => true,
            Verdict::NonMember => false,
            // A breakpoint at the arithmetic's resolution: kept, so the
            // returned set contains the honest one.
            Verdict::Undecided { band_limited: true } => true,
            // A cell `f64` cannot bisect: kept, for the same reason.
            Verdict::Undecided { .. } if !(mid > cell.s1 && mid < cell.s2) => true,
            Verdict::Undecided { .. } => {
                // The cell straddles a breakpoint, or its candidates disagree:
                // bisect it, and bisect the disagreeing boxes so the tube
                // follows the halves.
                let disagree = disagreement(&verdicts);
                let flagged: Vec<bool> = verdicts
                    .iter()
                    .zip(&box_ties)
                    .map(|(v, &tie)| disagree && !tie && v.exact.is_some())
                    .collect();
                for (a, b) in [(cell.s1, mid), (mid, cell.s2)] {
                    let mut seeds = cell.seeds.clone();
                    let tube = split_and_prune(basis, data, a, b, &tube, &flagged, &mut seeds);
                    stack.push(Cell {
                        chart: cell.chart,
                        s1: a,
                        s2: b,
                        tube,
                        seeds,
                    });
                }
                false
            }
        };
        if keep {
            let (z1, z2) = (data.chart.z_at(cell.s1), data.chart.z_at(cell.s2));
            let (p1, p2) = ((cell.chart, cell.s1), (cell.chart, cell.s2));
            kept.push(if z1 <= z2 {
                Retained {
                    lo: z1,
                    hi: z2,
                    lo_point: p1,
                    hi_point: p2,
                }
            } else {
                Retained {
                    lo: z2,
                    hi: z1,
                    lo_point: p2,
                    hi_point: p1,
                }
            });
        }
    }
    // Adjacent cells share their boundary value exactly (the charts meet at
    // bit-identical `z_c ± σ`), so touching cells merge.
    kept.sort_by(|a, b| a.lo.total_cmp(&b.lo));
    let mut merged: Vec<Retained> = Vec::new();
    for piece in kept {
        match merged.last_mut() {
            Some(last) if piece.lo <= last.hi => {
                if piece.hi > last.hi {
                    last.hi = piece.hi;
                    last.hi_point = piece.hi_point;
                }
            }
            _ => merged.push(piece),
        }
    }
    (merged, cells)
}

// ── Local refits ──────────────────────────────────────────────────────────

/// Why a local refit could not confirm the bound at a set endpoint.
enum RefitCheck {
    Consistent,
    Refused(ConformalRefusal),
}

/// One cold REML refit at the chart point `s`, through the outer engine,
/// checked against the tube at that point: a refit whose criterion lies below
/// the lower bound of every retained box contradicts the bound.
fn refit_at(basis: &Basis, data: &ChartData, s: f64, required: usize) -> RefitCheck {
    use gam_problem::{Derivative, HessianValue, OuterEval};
    use gam_solve::estimate::EstimationError;
    use gam_solve::rho_optimizer::OuterProblem;

    let (start, mut seeds) = root_tube(basis, data, s, s);
    let (tube, ..) = settle(basis, data, s, s, start, &mut seeds, required);
    let c = basis.c;
    let mut lower_bound = f64::INFINITY;
    let mut bound_band = 0.0_f64;
    for &(r1, r2) in &tube {
        let rss = data.rss(&basis.shrinkage(r1));
        let d = rss.value(s) - basis.growth * rss.magnitude();
        if !(d > 0.0) {
            return RefitCheck::Refused(ConformalRefusal::RemlUndefined);
        }
        let log_det = basis.log_det_part(r2);
        lower_bound = lower_bound.min(c * d.ln() + log_det);
        bound_band = bound_band.max(basis.growth * (c * d.ln().abs() + log_det));
    }

    let context = format!("honest full conformal refit at z={}", data.chart.z_at(s));
    let refuse = |reason: String| EstimationError::TrialPointRefused { reason };
    let at = |rho: f64| {
        criterion_at(basis, data, rho, s)
            .ok_or_else(|| refuse(format!("{context}: penalized RSS not positive at ρ={rho}")))
    };
    let (lower, upper) = basis.domain;
    // The augmented criterion carries the test point as one more row.
    let problem = OuterProblem::new(1)
        .with_problem_size(basis.n + 1, basis.p)
        .with_gradient(Derivative::Analytic)
        .with_hessian(gam_problem::DeclaredHessianForm::Dense)
        .with_bounds(Array1::from_elem(1, lower), Array1::from_elem(1, upper));
    let mut objective = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| at(rho[0]).map(|(value, ..)| value),
        |_: &mut (), rho: &Array1<f64>| {
            let (value, grad, hess, _) = at(rho[0])?;
            Ok(OuterEval {
                cost: value,
                gradient: Array1::from_vec(vec![grad]),
                hessian: HessianValue::Dense(Array2::from_elem((1, 1), hess)),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<gam_problem::EfsEval, EstimationError>>,
    );
    let Ok(result) = problem.run(&mut objective, &context) else {
        return RefitCheck::Refused(ConformalRefusal::RefitFailed);
    };
    let Some((value, _, _, band)) = criterion_at(basis, data, result.rho[0], s) else {
        return RefitCheck::Refused(ConformalRefusal::RefitFailed);
    };
    if value + band < lower_bound - bound_band {
        RefitCheck::Refused(ConformalRefusal::RefitOutsideTube)
    } else {
        RefitCheck::Consistent
    }
}

// ── Entry point ───────────────────────────────────────────────────────────

/// The full-conformal set of the REML re-selecting map at miscoverage `alpha`,
/// with its certificate (see the module doc).
///
/// `penalty_count` is the number of smoothing parameters the fit selected
/// (`None` when a payload predates that field). Inputs are validated exactly
/// as [`ExactGaussianFullConformal::new`] validates them.
pub fn honest_full_conformal(
    x: &Array2<f64>,
    y: &Array1<f64>,
    prior_weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
    penalty_count: Option<usize>,
    x_star: &Array1<f64>,
    alpha: f64,
) -> Result<HonestFullConformal, String> {
    validate_inputs(x, y, prior_weights, s_lambda, x_star)?;
    let frozen_answer = |engine: ExactGaussianFullConformal,
                         certificate: ConformalCertificate,
                         cost: HonestConformalCost| HonestFullConformal {
        set: engine.prediction_set(alpha),
        certificate,
        cost,
        plug_in_mean: engine.plug_in_mean(),
    };
    let one_factorization = HonestConformalCost {
        factorizations: 1,
        ..HonestConformalCost::default()
    };
    let reason = match penalty_count {
        None => Some(ConformalRefusal::UnknownPenaltyStructure),
        Some(count) if count >= 2 => Some(ConformalRefusal::MultiPenalty),
        _ => None,
    };
    if let Some(reason) = reason {
        let engine = ExactGaussianFullConformal::new(x, y, prior_weights, s_lambda, x_star)?;
        return Ok(frozen_answer(
            engine,
            ConformalCertificate::Refused(reason),
            one_factorization,
        ));
    }
    if penalty_count == Some(0) {
        let engine = ExactGaussianFullConformal::new(x, y, prior_weights, s_lambda, x_star)?;
        return Ok(frozen_answer(
            engine,
            ConformalCertificate::ExactFrozen,
            one_factorization,
        ));
    }

    let mut cost = HonestConformalCost {
        factorizations: 1,
        eigendecompositions: 2,
        ..HonestConformalCost::default()
    };
    let basis = match Basis::build(x, y, s_lambda, x_star)? {
        Ok(basis) => basis,
        Err(reason) => {
            let engine = ExactGaussianFullConformal::new(x, y, prior_weights, s_lambda, x_star)?;
            cost.factorizations += 1;
            return Ok(frozen_answer(
                engine,
                ConformalCertificate::Refused(reason),
                cost,
            ));
        }
    };
    let frozen = basis.frozen_engine();
    if basis.ln_s.is_empty() {
        return Ok(frozen_answer(frozen, ConformalCertificate::ExactFrozen, cost));
    }
    let n = basis.n;
    let required = required_dominating_count(n, alpha);
    let plug_in_mean = frozen.plug_in_mean();
    let honest = |intervals: Vec<ConformalInterval>, cost: HonestConformalCost| HonestFullConformal {
        set: FullConformalSet {
            intervals,
            alpha,
            n_augmented: n + 1,
        },
        certificate: ConformalCertificate::HonestRefit,
        cost,
        plug_in_mean,
    };
    // Membership needs no comparison at all: the same for every fitting map.
    if required == 0 {
        return Ok(honest(
            vec![ConformalInterval {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
            }],
            cost,
        ));
    }
    if required > n {
        return Ok(honest(Vec::new(), cost));
    }

    // The window: the frozen set's reach around the plug-in mean, plus the
    // residual scale. It only places the chart seam; every z is covered.
    let rss_at_center: f64 = basis
        .projection_residuals
        .iter()
        .map(|g| g.value(plug_in_mean).powi(2))
        .chain(
            basis
                .tau
                .iter()
                .zip(basis.shrinkage(0.0))
                .map(|(t, (gamma, _))| gamma * t.value(plug_in_mean).powi(2)),
        )
        .sum();
    if !(rss_at_center > 0.0) {
        return Ok(frozen_answer(
            frozen,
            ConformalCertificate::Refused(ConformalRefusal::RemlUndefined),
            cost,
        ));
    }
    let frozen_set = frozen.prediction_set(alpha);
    let reach = frozen_set
        .intervals
        .iter()
        .flat_map(|itv| [itv.lo, itv.hi])
        .filter(|z| z.is_finite())
        .map(|z| (z - plug_in_mean).abs())
        .fold(0.0, f64::max);
    let sigma = reach + (rss_at_center / basis.c).sqrt();
    let charts: Vec<ChartData> = Chart::charts(plug_in_mean, sigma)
        .into_iter()
        .map(|chart| ChartData::new(&basis, chart))
        .collect();

    let (retained, cells) = branch_and_bound(&basis, &charts, required);
    cost.z_cells = cells;
    let mut intervals = Vec::with_capacity(retained.len());
    for piece in &retained {
        for (z, (chart, s)) in [(piece.lo, piece.lo_point), (piece.hi, piece.hi_point)] {
            if z.is_finite() {
                cost.extra_refits += 1;
                if let RefitCheck::Refused(reason) = refit_at(&basis, &charts[chart], s, required) {
                    return Ok(HonestFullConformal {
                        set: frozen_set,
                        certificate: ConformalCertificate::Refused(reason),
                        cost,
                        plug_in_mean,
                    });
                }
            }
        }
        // `z_at` rounds; widen outward by the arithmetic's own growth so the
        // reported interval contains the retained cells.
        let widen = |z: f64, sign: f64| {
            if z.is_finite() { z + sign * basis.growth * z.abs() } else { z }
        };
        let (lo, hi) = (widen(piece.lo, -1.0), widen(piece.hi, 1.0));
        match intervals.last_mut() {
            Some(ConformalInterval { hi: last_hi, .. }) if lo <= *last_hi => *last_hi = hi,
            _ => intervals.push(ConformalInterval { lo, hi }),
        }
    }
    Ok(honest(intervals, cost))
}

#[cfg(test)]
mod tests;
