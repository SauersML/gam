//! Analytic infinite-smoothing rail FACE certificate (#2348 Inc 5).
//!
//! The [`asymptote_certificate`](super::asymptote_certificate) module confirms a
//! rail by *measuring* the exponential tail: probe `∂V/∂ρ_j` a few e-folds back
//! from the bound and check that the pencil constant `ĉ = −e^{ρ}∂V/∂ρ` holds
//! still. That is evidence about ONE coordinate along ONE ray, gathered at
//! finite `λ` where the criterion's logdet pair is already cancelling. This
//! module proves the same statement *analytically, at the face itself*, for
//! EVERY direction out of it at once.
//!
//! # The limit is finite, and it is the null-space-restricted fit
//!
//! Write the (profiled-Gaussian) criterion as
//!
//! ```text
//!     V(λ) = (n − M_p)/2 · log D_p + ½ log|H| − ½ log|S_λ|₊ + const,
//!     H = XᵀWX + S_λ,   S_λ = S_F + S_R,   S_F = Σ_{j∈F} λ_j S_j.
//! ```
//!
//! Let `N = ⋂_{j∈F} null(S_j)` be the common null space of the railed
//! penalties, `Z` an orthonormal basis of `N`, and `Q` an orthonormal basis of
//! its orthogonal complement inside the model space — the subspace the face
//! *releases*. In the `[Z, Q]` basis `S_F` lives entirely in the `QQ` block, so
//!
//! ```text
//!     log|H|    = log|S_F^{QQ}| + tr((S_F^{QQ})⁻¹·Schur_Z(K)) + log|ZᵀKZ|   + O(λ⁻²),
//!     log|S_λ|₊ = log|S_F^{QQ}| + tr((S_F^{QQ})⁻¹·Schur_Z(S_R)) + log|ZᵀS_RZ|₊ + O(λ⁻²),
//! ```
//!
//! with `K = XᵀWX + S_R` and `Schur_Z(M) = QᵀMQ − QᵀMZ(ZᵀMZ)⁻ZᵀMQ`. The
//! divergent `log|S_F^{QQ}|` — the `r_F·log λ` that makes each logdet blow up
//! on its own — **cancels exactly between the two**. What is left is finite:
//!
//! ```text
//!     V_∞ = (n − M_p)/2 · log D_p^∞ + ½log|Zᵀ(XᵀWX + S_R)Z| − ½log|ZᵀS_RZ|₊ + const,
//! ```
//!
//! which is precisely the REML criterion of the model restricted to `N` — the
//! λ→∞ limit model, fitted with the surviving penalties. `M_p = dim null(S_λ)`
//! is the same at the face as at any finite `λ > 0`, so no rank bookkeeping
//! moves. **An infinite smoothing parameter is not a numerical accident; it is
//! an ordinary fit of a smaller model, and this is its criterion value.**
//!
//! # The first-order term, and why one matrix decides the whole face
//!
//! Expanding the penalized fit (whose `O(λ⁻¹)` coefficient offset is
//! `S_F⁺g_c`, `g_c` the limit score) and the two traces above, everything of
//! first order collects into ONE symmetric form on the released subspace:
//!
//! ```text
//!     V(λ) = V_∞ + ½ tr( (QᵀS_FQ)⁻¹ C ) + O(λ⁻²),
//!     C = Schur_Z(XᵀWX + S_R) − Schur_Z(S_R) − g_Q g_Qᵀ/φ̂,   g_Q = Qᵀg_c.
//! ```
//!
//! Read it as the empirical-Bayes trade: `Schur_Z(K) − Schur_Z(S_R)` is the
//! *conditional information* the released directions would carry (the Occam
//! cost of un-freezing them), `g_Qg_Qᵀ/φ̂` is the *fit gain* from letting them
//! move. With no surviving penalty this is `I_{Q|Z} − g_Qg_Qᵀ/φ̂`, so `C ≻ 0`
//! is exactly the statement that the score test for releasing the constrained
//! directions falls below one: **no evidence, keep the null.**
//!
//! Three consequences make this a face certificate rather than a coordinate
//! one:
//!
//! 1. `(QᵀS_FQ)⁻¹ ≻ 0` for every positive `λ_F`, so `C ≻ 0` ⟹ `V(λ) > V_∞`
//!    for *all* finite smoothing parameters on the face — no direction, ray or
//!    mixture, needs to be enumerated.
//! 2. The same `C` governs every SUB-face. Releasing only `G ⊆ F` leaves the
//!    model constrained to `N_{F\G}`, whose released part is a subspace of
//!    `span(Q)`; the Schur complement against the *fixed* `Z` block restricts
//!    to it by compression. So one positive-definiteness test covers all
//!    `2^{|F|} − 1` ways to come off the face.
//! 3. Setting `S_F = λ_j S_j` recovers the measured law exactly:
//!    `∂V/∂ρ_j = −c_j e^{−ρ_j}` with `c_j = ½ tr((Q_jᵀS_jQ_j)⁻¹ Q_jᵀCQ_j)`.
//!    The probed pencil constant is a *measurement of this number*, and the
//!    two can be compared.
//!
//! # `C ≻ 0` is sufficient, not necessary: the KKT test
//!
//! What the face needs is `f(t) = ½tr((Σ_j A_j/t_j)⁻¹C) > 0` on the closed
//! orthant `t_j = e^{−ρ_j} ≥ 0` (`A_j = QᵀS_jQ`), not positivity of `C` in
//! every direction of `span(Q)`: a smoothing parameter moves its whole range
//! at once, so an eigen-direction of `C` that no weighting of the face can
//! isolate is not a way off the face. When the released ranges are linearly
//! independent (`Σ_j rank A_j = q`), a congruence `R = [R_1 … R_m]` puts
//! every `A_j = R_j M_j R_jᵀ` in block form together, and
//!
//! ```text
//!     (Σ_j A_j/t_j)⁻¹ = R⁻ᵀ·diag(t_j M_j⁻¹)·R⁻¹   ⟹   f(t) = Σ_j c_j t_j
//! ```
//!
//! exactly — `f` is LINEAR, and the face is a strict minimizer iff every
//! identified `c_j > 0`. A single-penalty face is always of this kind, which
//! is why the positive-definiteness gate over-refused it: with one `λ` the
//! first-order change is `c·e^{−ρ}` whatever the sign pattern of `C`; a
//! measured negative `c_j` refutes a face of any kind.
//!
//! When the ranges OVERLAP `f` is genuinely nonlinear, and the axis laws `c_j`
//! are only its values at the simplex vertices: with `A_1 = diag(1,1,0)`,
//! `A_2 = diag(0,1,1)` and `C = diag(1,−5,1)`, `f = ½(t_1 + t_2 − 5t_1t_2/(t_1+t_2))`
//! has both vertex values `½` yet `f(½,½) = −⅛` — a per-axis test is unsound.
//! The weighted parallel sum `M(t) = (Σ_j A_j/t_j)⁻¹` is matrix-concave on the
//! orthant, so with the spectral split `C = C₊ − C₋` (both PSD) the expansion
//! `f = ½tr(MC₊) − ½tr(MC₋)` is a difference of concave functions and a
//! simplicial branch-and-bound decides it (`certify_overlapping_face`).
//!
//! A face coordinate whose own penalty releases nothing once the others are at
//! `λ = ∞` (its released subspace is empty) is **unidentified there**: `V` does
//! not depend on `λ_j` at all, exactly, not merely to first order. Such a
//! coordinate reports `c_j = 0` and is typed [`FaceCoordinateKind::Unidentified`]
//! — it must not be required to carry an outward derivative, and it must not
//! block the face.
//!
//! # The LAML extension: families whose working weights move with `β̂`
//!
//! For a fixed-unit-dispersion LAML criterion (binomial, poisson)
//!
//! ```text
//!     V(λ) = −ℓ(β̂) + ½ β̂ᵀS_λβ̂ + ½ log|H| − ½ log|S_λ|₊,   H = I(β̂) + S_λ,
//!     I(β) = Xᵀ W(η) X,   W_i = −∂²ℓ_i/∂η_i²   (the criterion's observed weights),
//! ```
//!
//! the same expansion goes through with `K = I(β̂_∞) + S_R`, plus exactly one
//! new first-order term: the Laplace logdet's curvature moves with `β̂`, and
//! the face's `O(λ⁻¹)` coefficient offset `δβ = S_F⁺g_c` carries that motion
//! into first order,
//!
//! ```text
//!     Δ(½log|H|) = Σ_r [∂(½log|H_∞|)/∂β_r]·δβ_r = dᵀ δβ,
//!     d = ½ Xᵀ( c ⊙ a ),   c_i = dW_i/dη_i,   a_i = x_iᵀ Z (ZᵀKZ)⁻¹ Zᵀ x_i,
//! ```
//!
//! `a` the LIMIT model's leverage — no large-λ object anywhere. One care is
//! needed with `δβ = (K + S_F)⁻¹ g_c`: its PINNED component is `O(λ⁻¹)` too,
//! `δ_Z = −(ZᵀKZ)⁻¹ZᵀKQ·(QᵀS_FQ)⁻¹g_Q` (the block inverse's off-diagonal),
//! so `dᵀδβ` reads `d` through the K-OBLIQUE reduction onto the released
//! subspace — the same reduction the Schur complement applies to `K` — and
//! not the orthogonal compression:
//!
//! ```text
//!     d̃ = d − K Z (ZᵀKZ)⁻¹ Zᵀ d,     dᵀδβ = d̃_Qᵀ (QᵀS_FQ)⁻¹ g_Q.
//! ```
//!
//! (The Gaussian form never meets this subtlety: its only score-linear term
//! carries `g_c`, and `Zᵀg_c = 0` kills the oblique part exactly. Dropping
//! the reduction here is a FIRST-order error — measured at 1.7e-2 relative
//! against the production LAML gradient before it was applied.) In trace
//! form `dᵀδβ = ½tr((QᵀS_FQ)⁻¹(g_Q d̃_Qᵀ + d̃_Q g_Qᵀ))`, so the whole
//! extension is a symmetric rank-2 correction to the form:
//!
//! ```text
//!     C_LAML = Schur_Z(I(β̂_∞)+S_R) − Schur_Z(S_R) − g_Q g_Qᵀ + (g_Q d̃_Qᵀ + d̃_Q g_Qᵀ).
//! ```
//!
//! (The fit term's `−g_Qg_Qᵀ` carries no dispersion divisor: for these
//! families `−ℓ` and `H` are already in the same units, which is also why the
//! profiled-Gaussian form divides by `φ̂` — there the fit term is
//! `(n−M_p)/2·log D_p` and its first variation is `ΔD_p/(2φ̂)`.) The two
//! penalty logdets have no `β̂`-dependence, so their Schur content is
//! unchanged, and everything downstream — the `C ≻ 0` proof, per-coordinate
//! `c_j`, the `Unidentified` typing — is
//! family-blind because none of it depends on how `C` was built. Gaussian
//! identity is the `c ≡ 0` member: the rank-2 term vanishes identically and
//! the form reduces to the REML one, which is the built-in exactness check.
//!
//! This module is pure linear algebra on the supplied [`RailFaceLimit`]; the
//! objective that owns the design and the penalties builds that input.

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::roundoff::accumulation_growth;
use crate::model_types::FacePositivityRoute;
use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Relative eigenvalue threshold separating a subspace's range from its null
/// space. A symmetric eigensolver returns eigenvalues with backward error
/// `O(ε‖A‖)`; anything below `√ε·‖A‖` cannot be distinguished from an exact
/// zero by that spectrum, and everything above it is a genuine direction. This
/// is the same `√ε` split the outer certificate's PSD verdict uses.
fn subspace_split_threshold(spectrum_norm: f64) -> f64 {
    f64::EPSILON.sqrt() * spectrum_norm
}

/// Whether a face coordinate carries a strict outward derivative of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FaceCoordinateKind {
    /// Releasing this coordinate alone (the rest of the face held at `λ = ∞`)
    /// frees a nonempty subspace, and the criterion strictly increases along
    /// it: `c_j > 0`.
    StrictOutward,
    /// Releasing this coordinate alone frees nothing — the other face
    /// penalties already pin every direction this one would penalize. `V` is
    /// exactly independent of `λ_j` there, so the coordinate is unidentified
    /// at the face and carries no derivative to certify.
    Unidentified,
}

/// The analytic λ→∞ face-limit data an objective supplies for one rail face.
///
/// Every matrix is expressed in an orthonormal basis `Q` of the subspace the
/// face releases (`N^⊥`, `N = ⋂_{j∈F} null(S_j)`); `q` is its dimension.
#[derive(Clone, Debug)]
pub struct RailFaceLimit {
    /// The ρ-coordinates on the face, ascending.
    pub face: Vec<usize>,
    /// `ρ_j` at the certified point, in `face` order. Only used to price the
    /// remaining value gap and estimand travel of the shipped fit; the proof
    /// itself does not depend on it.
    pub face_rho: Vec<f64>,
    /// The symmetric `q×q` first-order form `C`.
    pub first_order_form: Array2<f64>,
    /// `QᵀS_jQ` for each face coordinate, in `face` order.
    pub released_penalties: Vec<Array2<f64>>,
    /// `Qᵀg_c`: the limit score in the released directions.
    pub released_score: Array1<f64>,
    /// Rigorous bound on `‖ΔC‖₂`, the floating-point error of the assembled
    /// form. With `γ_p = p·u/(1 − p·u)` (`u = ε/2`, `p` the coefficient
    /// dimension), every product and Schur solve that built `C` contributes
    /// at most `γ_p` relative error on its operands' scale, the pinned solve
    /// amplified by its conditioning:
    /// `γ_p·[(‖K‖ + ‖S_R‖)(1 + cond) + ‖g_Q‖²/φ̂ + 2‖g_Q‖‖d̃_Q‖]`. It is
    /// measured on the operands `C` was built FROM, not on `C` itself — the
    /// Schur differences cancel, and an error bound read off the cancelled
    /// result would understate the rounding it carries.
    pub form_error_bound: f64,
    /// The λ=∞ fit itself: coefficients of the null-space-restricted model, in
    /// the model's own coefficient basis. This is the limit the certificate is
    /// about — the fit a face-certified optimum reports — and the same object a
    /// continuation anchored at maximal smoothing starts from.
    pub limit_beta: Array1<f64>,
    /// Profiled dispersion at the limit fit, the `φ̂` the first-order form's
    /// fit term is divided by.
    pub limit_dispersion: f64,
    /// `Qᵀd`: the LAML curvature-drift vector in the released directions,
    /// recorded so the proof object shows how much of `C` is the rank-2
    /// `g_Q d_Qᵀ + d_Q g_Qᵀ` correction. `None` on the profiled-Gaussian
    /// form, whose working weights do not move with the coefficients.
    pub released_curvature_drift: Option<Array1<f64>>,
}

/// A proven rail face: the analytic first-order data behind the mint.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RailFaceProof {
    /// Which exact positivity test decided the face.
    pub route: FacePositivityRoute,
    /// The route's decisive statistic — `λ_min(C)` on the positive-form
    /// route, the binding coordinate's `c_j` on the independent-ranges route.
    /// The proof is `statistic > band`.
    pub statistic: f64,
    /// The rounding band `statistic` had to clear.
    pub band: f64,
    /// `λ_min(C)`, reported on either route.
    pub min_curvature: f64,
    /// `‖C‖₂`.
    pub form_norm: f64,
    /// Analytic pencil constants `c_j` in `face` order: `∂V/∂ρ_j → −c_j e^{−ρ_j}`.
    pub tail_constants: Vec<f64>,
    /// The rounding band `τ_j` of each `c_j`, in `face` order (`0` for an
    /// unidentified coordinate, which carries no constant to resolve).
    pub tail_bands: Vec<f64>,
    /// Per-coordinate identifiability at the face, in `face` order.
    pub coordinate_kinds: Vec<FaceCoordinateKind>,
    /// The joint pencil constant along the ray that releases the whole face
    /// together with unit weights.
    pub joint_tail_constant: f64,
    /// `V(ρ̂) − V_∞`: the entire criterion improvement still available by
    /// running the face out to `λ = ∞`, priced analytically at the certified ρ.
    pub value_gap: f64,
    /// `‖β(ρ̂) − β(∞)‖`, the exact first-order estimand travel to the limit fit.
    pub estimand_travel: f64,
}

/// Verdict of the analytic face certificate.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RailFaceVerdict {
    /// The first-order expansion `½tr((Σ_j A_j/t_j)⁻¹C)` is strictly positive
    /// for every way of coming off the face: the criterion strictly increases
    /// for every finite smoothing parameter on the face and on every sub-face.
    Certified(RailFaceProof),
    /// The analytic data does not prove the face. Carries the measured
    /// evidence, never a bare flag.
    Refused { reason: String },
}

/// Symmetrize exactly (`(M + Mᵀ)/2` is bitwise symmetric because `f64`
/// addition is commutative), which the strict self-adjoint eigensolver
/// requires.
fn symmetrized(matrix: &Array2<f64>) -> Array2<f64> {
    let n = matrix.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
        }
    }
    out
}

/// Eigenvectors of a PSD matrix whose eigenvalues are indistinguishable from
/// zero — an orthonormal basis of its null space. Returns `q×m`.
fn null_space_basis(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let q = matrix.nrows();
    let sym = symmetrized(matrix);
    let (values, vectors) = sym
        .eigh(Side::Lower)
        .map_err(|err| format!("null-space eigendecomposition failed: {err}"))?;
    let norm = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let cut = subspace_split_threshold(norm);
    let keep: Vec<usize> = (0..values.len()).filter(|&i| values[i].abs() <= cut).collect();
    let mut basis = Array2::<f64>::zeros((q, keep.len()));
    for (col, &i) in keep.iter().enumerate() {
        for row in 0..q {
            basis[[row, col]] = vectors[[row, i]];
        }
    }
    Ok(basis)
}

/// Eigenpairs of a PSD matrix above the range/null split: an orthonormal
/// `m×r` basis of its range together with the matching eigenvalues.
fn range_eigenpairs(matrix: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>), String> {
    let m = matrix.nrows();
    let sym = symmetrized(matrix);
    let (values, vectors) = sym
        .eigh(Side::Lower)
        .map_err(|err| format!("range eigendecomposition failed: {err}"))?;
    let norm = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let cut = subspace_split_threshold(norm);
    let keep: Vec<usize> = (0..values.len()).filter(|&i| values[i] > cut).collect();
    let mut basis = Array2::<f64>::zeros((m, keep.len()));
    let mut kept = Array1::<f64>::zeros(keep.len());
    for (col, &i) in keep.iter().enumerate() {
        kept[col] = values[i];
        for row in 0..m {
            basis[[row, col]] = vectors[[row, i]];
        }
    }
    Ok((basis, kept))
}

/// Rank of a PSD block above the `√ε` range/null split — the same split every
/// other subspace decision in this module uses, exported so callers building
/// a reduced (limit-model) penalty layout count exactly as the form does.
pub(crate) fn released_rank(matrix: &Array2<f64>) -> Result<usize, String> {
    let (_, values) = range_eigenpairs(matrix)?;
    Ok(values.len())
}

/// `½·tr(A⁻¹C)` for a symmetric positive-definite `A`, computed on `A`'s own
/// spectrum so a near-singular `A` reports its failure instead of amplifying
/// round-off through an explicit inverse.
fn half_trace_inverse_product(a: &Array2<f64>, c: &Array2<f64>) -> Result<f64, String> {
    let (basis, values) = range_eigenpairs(a)?;
    if values.len() != a.nrows() {
        return Err(format!(
            "released penalty is singular on the released subspace: rank {} < {}",
            values.len(),
            a.nrows()
        ));
    }
    // tr(A⁻¹C) = Σ_a (uₐᵀ C uₐ)/σₐ over A's eigenpairs.
    let mut total = 0.0_f64;
    for (col, &sigma) in values.iter().enumerate() {
        let u = basis.column(col);
        let cu = c.dot(&u);
        total += u.dot(&cu) / sigma;
    }
    Ok(0.5 * total)
}

/// Prove (or refuse) a rail face from its analytic λ→∞ limit data.
///
/// The first-order expansion off the face is `f(t) = ½tr((Σ_j A_j/t_j)⁻¹C)`,
/// `t_j = e^{−ρ_j}`, and the face is proven when `f > 0` on the whole closed
/// simplex of release directions. Three exact tests decide that:
///
/// * `C ≻ 0` ([`FacePositivityRoute::PositiveForm`]) is sufficient for any
///   face geometry;
/// * when the released ranges are linearly independent, `f = Σ_j c_j t_j` is
///   exactly linear and every identified `c_j > 0` is necessary AND
///   sufficient ([`FacePositivityRoute::IndependentRanges`]) — the KKT test,
///   which certifies faces whose `C` is indefinite along directions no
///   weighting of the face can isolate;
/// * when they overlap, a simplicial branch-and-bound on the concave split of
///   `f` ([`FacePositivityRoute::SimplexBound`]) proves `f > 0` cell by cell or
///   exhibits a release direction that descends.
///
/// A coordinate with a measured negative slope refutes the face on every
/// route. Every sign decision clears its own rounding band, derived from the
/// limit's [`RailFaceLimit::form_error_bound`]; refusals carry the measured
/// statistic and band so a declined face explains itself.
pub(crate) fn certify_rail_face(limit: &RailFaceLimit) -> RailFaceVerdict {
    let refuse = |reason: String| RailFaceVerdict::Refused { reason };
    let q = limit.first_order_form.nrows();
    if limit.face.is_empty() {
        return refuse("empty rail face".to_string());
    }
    if q == 0 {
        return refuse("the face releases no direction".to_string());
    }
    if limit.first_order_form.ncols() != q
        || limit.released_score.len() != q
        || limit.released_penalties.len() != limit.face.len()
        || limit.face_rho.len() != limit.face.len()
        || limit
            .released_penalties
            .iter()
            .any(|a| a.nrows() != q || a.ncols() != q)
    {
        return refuse("face limit data has inconsistent dimensions".to_string());
    }
    if limit.first_order_form.iter().any(|v| !v.is_finite())
        || limit.released_score.iter().any(|v| !v.is_finite())
        || limit
            .released_penalties
            .iter()
            .any(|a| a.iter().any(|v| !v.is_finite()))
    {
        return refuse("face limit data is not finite".to_string());
    }

    if !limit.form_error_bound.is_finite() || limit.form_error_bound < 0.0 {
        return refuse(format!(
            "face limit form error bound {:.3e} is not a finite non-negative number",
            limit.form_error_bound
        ));
    }

    let form = symmetrized(&limit.first_order_form);
    let (values, form_vectors) = match form.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(err) => return refuse(format!("first-order form eigendecomposition failed: {err}")),
    };
    let min_curvature = values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
    let form_norm = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    // `λ_min(C)` is a measured fact only outside the error of forming `C`
    // (bounded on its operands, `form_error_bound`) plus the symmetric
    // eigensolver's own backward error `γ_q‖C‖` — Weyl moves every eigenvalue
    // by at most the norm of the perturbation.
    let gamma_q = accumulation_growth(q);
    let curvature_band = limit.form_error_bound + gamma_q * form_norm;
    let positive_form = min_curvature > curvature_band;

    // Per-coordinate law: hold the rest of the face at λ=∞ and release only j.
    // The directions it frees are the range of its own penalty inside the null
    // space of the others; on that subspace the ordinary one-coordinate tail
    // law holds with `c_j = ½tr((Q_jᵀS_jQ_j)⁻¹ Q_jᵀCQ_j)`.
    let mut tail_constants = Vec::with_capacity(limit.face.len());
    let mut tail_bands = Vec::with_capacity(limit.face.len());
    let mut coordinate_kinds = Vec::with_capacity(limit.face.len());
    let mut released_ranks = Vec::with_capacity(limit.face.len());
    for idx in 0..limit.face.len() {
        match released_rank(&limit.released_penalties[idx]) {
            Ok(rank) => released_ranks.push(rank),
            Err(err) => return refuse(err),
        }
        let mut rest = Array2::<f64>::zeros((q, q));
        for (other, penalty) in limit.released_penalties.iter().enumerate() {
            if other != idx {
                rest += penalty;
            }
        }
        let free = match null_space_basis(&rest) {
            Ok(basis) => basis,
            Err(err) => return refuse(err),
        };
        if free.ncols() == 0 {
            tail_constants.push(0.0);
            tail_bands.push(0.0);
            coordinate_kinds.push(FaceCoordinateKind::Unidentified);
            continue;
        }
        let own = free.t().dot(&limit.released_penalties[idx]).dot(&free);
        let (own_basis, own_values) = match range_eigenpairs(&own) {
            Ok(pair) => pair,
            Err(err) => return refuse(err),
        };
        if own_values.is_empty() {
            tail_constants.push(0.0);
            tail_bands.push(0.0);
            coordinate_kinds.push(FaceCoordinateKind::Unidentified);
            continue;
        }
        let released = free.dot(&own_basis);
        let compressed = released.t().dot(&form).dot(&released);
        let mut c_j = 0.0_f64;
        for (a, &sigma) in own_values.iter().enumerate() {
            c_j += compressed[[a, a]] / sigma;
        }
        c_j *= 0.5;
        // `|Δc_j| ≤ ½Σ_a |u_aᵀΔC u_a|/σ_a ≤ ½tr(own⁻¹)·‖ΔC‖₂`, where `ΔC`
        // carries both the forming error and the compression's own rounding —
        // together the `curvature_band` — plus the own block's eigen-rounding,
        // relative `γ_q·κ(own)` on `c_j` itself.
        let inverse_trace: f64 = own_values.iter().map(|sigma| 1.0 / sigma).sum();
        let own_conditioning = own_values.iter().fold(0.0_f64, |acc, v| acc.max(*v))
            / own_values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
        let tau_j =
            0.5 * inverse_trace * curvature_band + c_j.abs() * gamma_q * own_conditioning;
        if !c_j.is_finite() || !tau_j.is_finite() {
            return refuse(format!(
                "face coordinate {} has a non-finite analytic pencil constant {c_j:.3e} \
                 (band {tau_j:.3e})",
                limit.face[idx]
            ));
        }
        if c_j < -tau_j {
            // A measured NEGATIVE slope: `f(t) = c_j t_j` along this
            // coordinate's own release, so pulling it back from λ=∞ lowers
            // the criterion. This refutes the face on every route.
            return refuse(format!(
                "releasing face coordinate {} alone lowers the criterion at first order: \
                 c={c_j:.6e} < −τ={:.3e} — λ=∞ is not a minimizer on this face",
                limit.face[idx], tau_j
            ));
        }
        tail_constants.push(c_j);
        tail_bands.push(tau_j);
        coordinate_kinds.push(FaceCoordinateKind::StrictOutward);
    }

    // Decide positivity of `f(t) = ½tr((Σ_j A_j/t_j)⁻¹C)` on the simplex.
    let released_rank_total: usize = released_ranks.iter().sum();
    let (route, statistic, band) = if positive_form {
        // `C ≻ 0` makes every compression positive definite, so a
        // non-positive `c_j` here can only be a numerically collapsed
        // released block — evidence the geometry is not resolvable.
        if let Some(idx) = (0..limit.face.len()).find(|&idx| {
            coordinate_kinds[idx] == FaceCoordinateKind::StrictOutward
                && !(tail_constants[idx] > 0.0)
        }) {
            return refuse(format!(
                "face coordinate {} has a non-positive analytic pencil constant {:.6e} \
                 against a positive-definite face form — released geometry is degenerate",
                limit.face[idx], tail_constants[idx]
            ));
        }
        (FacePositivityRoute::PositiveForm, min_curvature, curvature_band)
    } else if released_rank_total == q {
        // Independent released ranges: `f` is exactly linear, `Σ_j c_j t_j`,
        // so it is positive on the simplex iff every identified slope is — the
        // KKT test at the face, necessary and sufficient. `C` itself may be
        // indefinite: a single penalty cannot move its range's directions
        // independently, so a negative eigen-direction of `C` that no
        // weighting of the face can isolate is not a descent direction.
        let mut binding: Option<(f64, f64)> = None;
        for idx in 0..limit.face.len() {
            if coordinate_kinds[idx] != FaceCoordinateKind::StrictOutward {
                continue;
            }
            let (c_j, tau_j) = (tail_constants[idx], tail_bands[idx]);
            if !(c_j > tau_j) {
                return refuse(format!(
                    "face coordinate {} has an unresolved first-order slope: |c|={:.3e} ≤ \
                     τ={tau_j:.3e}, so its sign is not a measured fact; the face's first-order \
                     form is indefinite (λ_min(C)={min_curvature:.6e} ≤ band \
                     {curvature_band:.3e}) and cannot decide it either",
                    limit.face[idx],
                    c_j.abs()
                ));
            }
            if binding.is_none_or(|(bc, bt)| c_j - tau_j < bc - bt) {
                binding = Some((c_j, tau_j));
            }
        }
        let Some((c_bind, tau_bind)) = binding else {
            return refuse("no face coordinate carries an identified first-order slope".to_string());
        };
        (FacePositivityRoute::IndependentRanges, c_bind, tau_bind)
    } else if released_rank_total < q {
        // The ranks sum below `q`, so their union cannot span the released
        // subspace: some released direction is penalized by no face member.
        return refuse(format!(
            "the face's released ranges do not span the released subspace (Σ rank A_j = \
             {released_rank_total} < q = {q})"
        ));
    } else {
        // Overlapping released ranges: `f` is nonlinear in `t`, and the axis
        // laws `c_j` are only its vertex values — a joint release can descend
        // with every one of them positive. Decide the whole simplex. A
        // coordinate whose penalty releases nothing leaves `M(t)` independent
        // of its `t_j`, so the simplex runs over the positive-rank members.
        let active: Vec<usize> =
            (0..limit.face.len()).filter(|&idx| released_ranks[idx] > 0).collect();
        let overlap = OverlapFace {
            penalties: active.iter().map(|&idx| &limit.released_penalties[idx]).collect(),
            labels: active.iter().map(|&idx| limit.face[idx]).collect(),
            form_pairs: (0..q)
                .map(|a| (values[a], form_vectors.column(a).to_owned()))
                .collect(),
            q,
        };
        match certify_overlapping_face(&overlap, curvature_band) {
            Ok((lower, cell_band)) => (FacePositivityRoute::SimplexBound, lower, cell_band),
            Err(reason) => {
                return refuse(format!(
                    "face first-order form is indefinite (λ_min(C)={min_curvature:.6e} ≤ band \
                     {curvature_band:.3e}, ‖C‖={form_norm:.3e}) and the face's released ranges \
                     overlap (Σ rank A_j = {released_rank_total} > q = {q}); on the release \
                     simplex: {reason}"
                ));
            }
        }
    };

    // The joint law: release the whole face together with unit weights.
    let mut unit_face = Array2::<f64>::zeros((q, q));
    for penalty in limit.released_penalties.iter() {
        unit_face += penalty;
    }
    let joint_tail_constant = match half_trace_inverse_product(&unit_face, &form) {
        Ok(value) => value,
        Err(err) => return refuse(format!("joint face law unavailable: {err}")),
    };

    // Price the shipped point: `V(ρ̂) − V_∞ = ½tr((Σ_j λ_j QᵀS_jQ)⁻¹C)` and the
    // matching coefficient offset `(Σ_j λ_j QᵀS_jQ)⁻¹Qᵀg_c`.
    let mut certified_face = Array2::<f64>::zeros((q, q));
    for (penalty, &rho) in limit.released_penalties.iter().zip(limit.face_rho.iter()) {
        let lambda = rho.exp();
        if !lambda.is_finite() || lambda <= 0.0 {
            return refuse(format!(
                "face coordinate ρ={rho:.3e} does not exponentiate to a usable λ"
            ));
        }
        certified_face
            .iter_mut()
            .zip(penalty.iter())
            .for_each(|(dst, src)| *dst += lambda * src);
    }
    let value_gap = match half_trace_inverse_product(&certified_face, &form) {
        Ok(value) => value,
        Err(err) => return refuse(format!("value gap at the certified point unavailable: {err}")),
    };
    let estimand_travel = match range_eigenpairs(&certified_face) {
        Ok((basis, values)) if values.len() == q => {
            let mut offset = Array1::<f64>::zeros(q);
            for (col, &sigma) in values.iter().enumerate() {
                let u = basis.column(col);
                let coeff = u.dot(&limit.released_score) / sigma;
                for row in 0..q {
                    offset[row] += coeff * u[row];
                }
            }
            offset.dot(&offset).sqrt()
        }
        Ok(_) => {
            return refuse(
                "the certified-point face penalty is singular on the released subspace"
                    .to_string(),
            );
        }
        Err(err) => return refuse(err),
    };
    if !value_gap.is_finite() || !estimand_travel.is_finite() || value_gap < 0.0 {
        return refuse(format!(
            "face pricing is not usable: value_gap={value_gap:.3e} travel={estimand_travel:.3e}"
        ));
    }

    RailFaceVerdict::Certified(RailFaceProof {
        route,
        statistic,
        band,
        min_curvature,
        form_norm,
        tail_constants,
        tail_bands,
        coordinate_kinds,
        joint_tail_constant,
        value_gap,
        estimand_travel,
    })
}

/// The overlapping-range face expansion, evaluated on the computed spectral
/// split of the first-order form.
///
/// With `C̃ = Σ_a λ_a u_a u_aᵀ` the computed eigen-reconstruction of `C`,
/// `C̃₊ = Σ_{λ_a>0} λ_a u_a u_aᵀ` and `C̃₋ = Σ_{λ_a<0} |λ_a| u_a u_aᵀ` are
/// exactly PSD, and `f̃ = P − N` with `P = ½tr(M C̃₊)`, `N = ½tr(M C̃₋)`. The
/// weighted parallel sum `M(t) = (Σ_j A_j/t_j)⁻¹` is matrix-concave on the
/// orthant (it is the parallel sum of the `t_j A_j⁻¹`, each linear in `t`),
/// so `P`, `N` and `tr M` are concave: `f̃` is a difference of concave
/// functions, and on a simplex cell `P` is bounded below by its vertex values
/// while `N` and `tr M` are bounded above by their tangent planes at any
/// interior point. `C̃ − C` is inside `curvature_band` (the forming error plus
/// the eigensolver's backward error), so `|f − f̃| ≤ ½tr M·curvature_band`.
struct OverlapFace<'a> {
    /// `A_j` for the positive-rank face coordinates, in simplex order.
    penalties: Vec<&'a Array2<f64>>,
    /// The face's ρ-coordinate for each simplex coordinate, for messages.
    labels: Vec<usize>,
    /// Eigenpairs `(λ_a, u_a)` of the symmetrized first-order form.
    form_pairs: Vec<(f64, Array1<f64>)>,
    q: usize,
}

/// `P`, `N`, `tr M` at one point of the release simplex, with rigorous
/// bounds on each one's floating-point error.
struct OverlapPoint {
    positive: f64,
    negative: f64,
    trace: f64,
    positive_error: f64,
    negative_error: f64,
    trace_error: f64,
    /// Interior points only: `∂_j N`, `∂_j tr M` and their error bounds.
    gradients: Option<OverlapGradients>,
}

struct OverlapGradients {
    negative: Vec<f64>,
    negative_error: Vec<f64>,
    trace: Vec<f64>,
    trace_error: Vec<f64>,
}

impl OverlapPoint {
    fn value(&self) -> f64 {
        self.positive - self.negative
    }

    /// `|f(t) − computed f̃(t)|`: the evaluation's own rounding plus the
    /// form's `½tr M·curvature_band`.
    fn band(&self, curvature_band: f64) -> f64 {
        self.positive_error
            + self.negative_error
            + 0.5 * (self.trace + self.trace_error) * curvature_band
    }
}

fn format_simplex_point(labels: &[usize], t: &[f64]) -> String {
    let parts: Vec<String> = labels
        .iter()
        .zip(t)
        .map(|(label, value)| format!("t{label}={value:.4}"))
        .collect();
    format!("[{}]", parts.join(", "))
}

impl OverlapFace<'_> {
    /// Evaluate the expansion at a point of the closed simplex. A zero
    /// coordinate is an exact λ=∞ pin: `M = Q(Qᵀ(Σ_{t_j>0}A_j/t_j)Q)⁻¹Qᵀ` on
    /// the null space `Q` of the pinned penalties.
    fn evaluate(&self, t: &[f64], with_gradients: bool) -> Result<OverlapPoint, String> {
        let q = self.q;
        let m = self.penalties.len();
        let gamma_q = accumulation_growth(q);
        let mut pinned = Array2::<f64>::zeros((q, q));
        let mut weighted = Array2::<f64>::zeros((q, q));
        let mut any_pinned = false;
        for (penalty, &tj) in self.penalties.iter().zip(t) {
            if tj == 0.0 {
                pinned += *penalty;
                any_pinned = true;
            } else {
                weighted.scaled_add(1.0 / tj, *penalty);
            }
        }
        // The pinned null space `Q` and its Davis–Kahan leakage: a computed
        // basis leans into the pinned range by at most `‖ΔS_Z‖/gap`, the
        // sum's formation and eigensolve error over its smallest range
        // eigenvalue.
        let (basis, leak) = if any_pinned {
            let (_, range_values) = range_eigenpairs(&pinned)?;
            let null = null_space_basis(&pinned)?;
            if null.ncols() == 0 {
                return Err(format!(
                    "the face penalties pinned at {} leave no direction free — the released \
                     coordinate is unidentified at the face — so ½tr(M(t)C) vanishes there and \
                     no cell touching it bounds the law away from zero",
                    format_simplex_point(&self.labels, t)
                ));
            }
            let top = range_values.iter().fold(0.0_f64, |acc, v| acc.max(*v));
            let gap = range_values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
            let perturbation = (gamma_q + (m as f64) * accumulation_growth(m + 1)) * top;
            (null, perturbation / gap)
        } else {
            (Array2::<f64>::eye(q), 0.0)
        };
        let reduced = symmetrized(&basis.t().dot(&weighted).dot(&basis));
        let (values, vectors) = reduced
            .eigh(Side::Lower)
            .map_err(|err| format!("face weighting eigendecomposition failed: {err}"))?;
        let sigma_min = values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
        let sigma_max = values.iter().fold(0.0_f64, |acc, v| acc.max(*v));
        if !(sigma_min > 0.0) || !sigma_max.is_finite() {
            return Err(format!(
                "the released face weighting is singular at {} (λ_min={sigma_min:.3e})",
                format_simplex_point(&self.labels, t)
            ));
        }
        // Every arithmetic stage — the eigensolve's backward error, the
        // weighted sum forming `A(t)`, the explicit product forming `M` and
        // each trace against it — contributes its `γ` relative to ‖A‖ or ‖M‖;
        // `κ(A)` converts that norm-wise error into a Loewner-relative one,
        // `M/(1+η) ≼ M̃ ≼ M/(1−η)`, so `|P̃ − P| ≤ ρP` with `ρ = η/(1−η)`.
        let conditioning = sigma_max / sigma_min;
        let arithmetic = gamma_q
            + (m as f64) * accumulation_growth(m + 1)
            + (q as f64 + 1.0) * accumulation_growth(q * q + q);
        let eta = arithmetic * conditioning;
        // The leaked basis perturbs `M` by at most `2θ(1 + ‖A_free‖/σ_min)‖M‖`:
        // once through the outer `Q`, once through the reduced solve.
        let weighted_norm = weighted.iter().map(|v| v * v).sum::<f64>().sqrt();
        let leak_relative = 2.0 * leak * (1.0 + weighted_norm / sigma_min);
        if !(eta < 0.5) || !(leak_relative < 0.5) {
            return Err(format!(
                "½tr(M(t)C) is not resolvable at {}: the face weighting's conditioning {:.3e} \
                 leaves relative error η={eta:.3e}, basis leakage {leak_relative:.3e}",
                format_simplex_point(&self.labels, t),
                conditioning
            ));
        }
        let rho = eta / (1.0 - eta);
        let leak_absolute = leak_relative / sigma_min;
        let lifted = basis.dot(&vectors);
        let mut scaled = lifted.clone();
        for (col, &sigma) in values.iter().enumerate() {
            scaled.column_mut(col).mapv_inplace(|v| v / sigma);
        }
        let weighting_inverse = scaled.dot(&lifted.t());

        let mut positive = 0.0_f64;
        let mut negative = 0.0_f64;
        let mut positive_mass = 0.0_f64;
        let mut negative_mass = 0.0_f64;
        let mut negative_images: Vec<(f64, Array1<f64>)> = Vec::new();
        for (lambda, u) in self.form_pairs.iter() {
            let image = weighting_inverse.dot(u);
            let quadratic = u.dot(&image);
            if *lambda > 0.0 {
                positive += lambda * quadratic;
                positive_mass += lambda;
            } else if *lambda < 0.0 {
                negative += -lambda * quadratic;
                negative_mass += -lambda;
                negative_images.push((-lambda, image));
            }
        }
        positive *= 0.5;
        negative *= 0.5;
        let trace: f64 = (0..q).map(|i| weighting_inverse[[i, i]]).sum();
        let positive_error = rho * positive + leak_absolute * 0.5 * positive_mass;
        let negative_error = rho * negative + leak_absolute * 0.5 * negative_mass;
        let trace_error = rho * trace + leak_absolute * (q as f64);

        let gradients = if with_gradients && !any_pinned {
            // `∂M/∂t_j = t_j⁻² M A_j M`. With `K = M^{½}A_jM^{½} ≼ t_j I` (as
            // `A_j/t_j ≼ A(t)`), `0 ≤ ∂_jN ≤ N/t_j`, and a Loewner-relative
            // `ρ` on `M` moves `∂_jN` by at most `(2ρ+ρ²)·N/t_j`; the same
            // holds for `tr M`.
            let square = weighting_inverse.dot(&weighting_inverse);
            let spread = (2.0 * rho + rho * rho) * (1.0 + rho);
            let mut grad_negative = Vec::with_capacity(m);
            let mut grad_negative_error = Vec::with_capacity(m);
            let mut grad_trace = Vec::with_capacity(m);
            let mut grad_trace_error = Vec::with_capacity(m);
            for (penalty, &tj) in self.penalties.iter().zip(t) {
                let inv_sq = 1.0 / (tj * tj);
                let mut value = 0.0_f64;
                for (weight, image) in negative_images.iter() {
                    value += weight * image.dot(&penalty.dot(image));
                }
                grad_negative.push(0.5 * inv_sq * value);
                grad_negative_error.push(spread * negative / tj);
                let trace_value: f64 = (0..q)
                    .map(|row| penalty.row(row).dot(&square.column(row)))
                    .sum();
                grad_trace.push(inv_sq * trace_value);
                grad_trace_error.push(spread * trace / tj);
            }
            Some(OverlapGradients {
                negative: grad_negative,
                negative_error: grad_negative_error,
                trace: grad_trace,
                trace_error: grad_trace_error,
            })
        } else {
            None
        };
        Ok(OverlapPoint {
            positive,
            negative,
            trace,
            positive_error,
            negative_error,
            trace_error,
            gradients,
        })
    }
}

/// `a + b` is exact in `f64` iff its TwoSum error term vanishes.
fn exact_sum(a: f64, b: f64) -> bool {
    let s = a + b;
    let bb = s - a;
    (a - (s - bb)) + (b - bb) == 0.0
}

/// Prove `f(t) = ½tr(M(t)C) > 0` on the release simplex of an overlapping
/// face by simplicial branch-and-bound on the concave split `f̃ = P − N`.
///
/// On a cell with vertices `v_i` and centroid `z`, `P − (N(z) + ∇N(z)·(x−z))`
/// is concave and bounds `f̃` below, so `min_i ℓ_i`,
/// `ℓ_i = P(v_i) − N(z) − ∇N(z)·(v_i − z)`, bounds it on the cell; the cell is
/// proven when that clears the evaluation errors plus `½·sup_cell tr M` times
/// the form band (`tr M` is bounded by its own tangent plane at `z`). A cell
/// whose centroid measures `f < −band` refutes the face; one whose centroid or
/// bound sits inside its band is unresolved, since no refinement can decide
/// a sign the arithmetic does not resolve. Otherwise the cell is split along
/// its longest edge, whose midpoint is exact in `f64` (dyadic vertices); a
/// midpoint that is not exact means the simplex cannot be refined further,
/// which refuses. `f` is analytic on the closed simplex and the lower bound's
/// gap is `O(diam²)`, so a face that is positive with margin is proven after
/// finitely many splits.
///
/// Returns the binding cell's `(min_i ℓ_i, band)` or the refusal reason.
fn certify_overlapping_face(
    overlap: &OverlapFace<'_>,
    curvature_band: f64,
) -> Result<(f64, f64), String> {
    let m = overlap.penalties.len();
    let gamma_m = accumulation_growth(m + 1);
    let mut vertices: Vec<Vec<f64>> = Vec::new();
    let mut vertex_values: Vec<OverlapPoint> = Vec::new();
    let mut vertex_index: std::collections::HashMap<Vec<u64>, usize> =
        std::collections::HashMap::new();
    let refute = |t: &[f64], point: &OverlapPoint| {
        format!(
            "releasing the face along {} lowers the criterion at first order: \
             ½tr(M(t)C)={:.6e} < −band {:.3e} — λ=∞ is not a minimizer on this face",
            format_simplex_point(&overlap.labels, t),
            point.value(),
            point.band(curvature_band)
        )
    };
    // The simplex centre first: a face refuted in its interior (the axis
    // laws flat or positive, the joint release negative) says so even when a
    // vertex is structurally degenerate.
    let centre = vec![1.0 / (m as f64); m];
    let at_centre = overlap.evaluate(&centre, false)?;
    if at_centre.value() < -at_centre.band(curvature_band) {
        return Err(refute(&centre, &at_centre));
    }
    for j in 0..m {
        let mut e = vec![0.0_f64; m];
        e[j] = 1.0;
        let point = overlap.evaluate(&e, false)?;
        // `f(e_j)` is the coordinate's own axis law `c_j`. Every cell touching
        // a vertex where `f` is not resolved positive keeps a lower bound at or
        // below that vertex's value, so no refinement could prove the face.
        let vertex_band = point.band(curvature_band);
        if point.value() < -vertex_band {
            return Err(refute(&e, &point));
        }
        if !(point.value() > vertex_band) {
            return Err(format!(
                "the face's first-order law is unresolved at the vertex {}: \
                 |½tr(M(t)C)|={:.3e} ≤ band {vertex_band:.3e}",
                format_simplex_point(&overlap.labels, &e),
                point.value().abs()
            ));
        }
        vertex_index.insert(e.iter().map(|v| v.to_bits()).collect(), vertices.len());
        vertices.push(e);
        vertex_values.push(point);
    }
    let mut stack: Vec<Vec<usize>> = vec![(0..m).collect()];
    let mut binding: Option<(f64, f64)> = None;
    while let Some(cell) = stack.pop() {
        let mut centroid = vec![0.0_f64; m];
        for &v in cell.iter() {
            for (acc, value) in centroid.iter_mut().zip(vertices[v].iter()) {
                *acc += value;
            }
        }
        let scale = 1.0 / (cell.len() as f64);
        centroid.iter_mut().for_each(|v| *v *= scale);
        let at_centroid = overlap.evaluate(&centroid, true)?;
        let value = at_centroid.value();
        let point_band = at_centroid.band(curvature_band);
        if value < -point_band {
            return Err(refute(&centroid, &at_centroid));
        }
        if !(value > point_band) {
            return Err(format!(
                "the face's first-order law is unresolved at {}: |½tr(M(t)C)|={:.3e} ≤ band \
                 {point_band:.3e}, so its sign on the release simplex is not a measured fact",
                format_simplex_point(&overlap.labels, &centroid),
                value.abs()
            ));
        }
        let Some(gradients) = at_centroid.gradients.as_ref() else {
            return Err("interior simplex point carried no gradient".to_string());
        };
        let mut lower = f64::INFINITY;
        let mut worst_error = 0.0_f64;
        let mut trace_rise = f64::NEG_INFINITY;
        for &v in cell.iter() {
            let vertex = &vertices[v];
            let at_vertex = &vertex_values[v];
            let mut linear = 0.0_f64;
            let mut linear_error = 0.0_f64;
            let mut trace_linear = 0.0_f64;
            for j in 0..m {
                let step = vertex[j] - centroid[j];
                linear += gradients.negative[j] * step;
                linear_error +=
                    (gradients.negative_error[j] + gamma_m * gradients.negative[j].abs()) * step.abs();
                trace_linear += gradients.trace[j] * step
                    + (gradients.trace_error[j] + gamma_m * gradients.trace[j].abs()) * step.abs();
            }
            let bound = at_vertex.positive - at_centroid.negative - linear;
            let error = at_vertex.positive_error
                + at_centroid.negative_error
                + linear_error
                + accumulation_growth(3)
                    * (at_vertex.positive + at_centroid.negative + linear.abs());
            lower = lower.min(bound);
            worst_error = worst_error.max(error);
            trace_rise = trace_rise.max(trace_linear);
        }
        let sup_trace = at_centroid.trace + at_centroid.trace_error + trace_rise;
        let band = worst_error + 0.5 * sup_trace * curvature_band;
        if lower > band {
            if binding.is_none_or(|(bl, bb)| lower - band < bl - bb) {
                binding = Some((lower, band));
            }
            continue;
        }
        if value - lower <= band + point_band {
            return Err(format!(
                "the face's first-order law is unresolved near {}: the cell's lower bound \
                 {lower:.6e} is already within its band of ½tr(M(t)C)={value:.6e} yet does not \
                 clear the band {band:.3e}",
                format_simplex_point(&overlap.labels, &centroid)
            ));
        }
        // Longest-edge bisection.
        let mut edge = (0usize, 1usize);
        let mut longest = f64::NEG_INFINITY;
        for a in 0..cell.len() {
            for b in (a + 1)..cell.len() {
                let length: f64 = vertices[cell[a]]
                    .iter()
                    .zip(vertices[cell[b]].iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum();
                if length > longest {
                    longest = length;
                    edge = (a, b);
                }
            }
        }
        let (va, vb) = (cell[edge.0], cell[edge.1]);
        let mut midpoint = Vec::with_capacity(m);
        for (x, y) in vertices[va].iter().zip(vertices[vb].iter()) {
            if !exact_sum(*x, *y) {
                return Err(format!(
                    "the release simplex cannot be refined exactly past {}: the face's \
                     first-order law ½tr(M(t)C)={value:.6e} is not resolved against band \
                     {band:.3e} at f64 resolution",
                    format_simplex_point(&overlap.labels, &centroid)
                ));
            }
            midpoint.push(0.5 * (x + y));
        }
        let key: Vec<u64> = midpoint.iter().map(|v| v.to_bits()).collect();
        let mid = match vertex_index.get(&key) {
            Some(&existing) => existing,
            None => {
                let point = overlap.evaluate(&midpoint, false)?;
                let id = vertices.len();
                vertex_index.insert(key, id);
                vertices.push(midpoint);
                vertex_values.push(point);
                id
            }
        };
        let mut first = cell.clone();
        first[edge.0] = mid;
        let mut second = cell;
        second[edge.1] = mid;
        stack.push(first);
        stack.push(second);
    }
    binding.ok_or_else(|| "the release simplex produced no certified cell".to_string())
}

/// Self-adjoint eigendecomposition of an exactly-symmetrized copy.
fn symmetric_eigh(matrix: &Array2<f64>) -> Option<(Array1<f64>, Array2<f64>)> {
    symmetrized(matrix).eigh(Side::Lower).ok()
}

/// [`subspace_split_threshold`] applied to an already-computed spectrum.
fn range_null_split(values: &Array1<f64>) -> f64 {
    subspace_split_threshold(values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())))
}

/// Gather selected eigenvectors into a `p×m` orthonormal basis.
fn basis_columns(vectors: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let rows = vectors.nrows();
    let mut basis = Array2::<f64>::zeros((rows, cols.len()));
    for (out, &src) in cols.iter().enumerate() {
        for row in 0..rows {
            basis[[row, out]] = vectors[[row, src]];
        }
    }
    basis
}

/// Moore–Penrose inverse rebuilt from an already-computed spectrum, dropping
/// everything at or below `cut`.
fn spectral_pseudo_inverse(values: &Array1<f64>, vectors: &Array2<f64>, cut: f64) -> Array2<f64> {
    let n = values.len();
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        if values[i] > cut {
            let inv = 1.0 / values[i];
            let v = vectors.column(i);
            for a in 0..n {
                let va = inv * v[a];
                for b in 0..n {
                    out[[a, b]] += va * v[b];
                }
            }
        }
    }
    out
}

/// Why a rail-face limit is or is not available.
///
/// This is deliberately not an `Option`. There is more than one closed form for
/// the criterion (the profiled-Gaussian one here; a LAML one for families whose
/// working weights move with `β̂`), and a bare `None` would collapse two
/// statements that call for opposite responses: *"this criterion is outside the
/// closed form you asked"* — where a DIFFERENT form may still apply — and
/// *"the closed form applies but this face is unusable"*, which is a statement
/// about the face itself and which no other form will rescue.
#[derive(Clone, Debug)]
pub enum RailFaceLimitOutcome {
    /// The limit was formed.
    Available(Box<RailFaceLimit>),
    /// The criterion is outside the closed form that was asked for. Says
    /// nothing about the face; another closed form may still apply.
    OutsideClosedForm {
        /// Which clause of the form's scope failed.
        reason: String,
    },
    /// The closed form applies, but the face cannot be used: it releases no
    /// direction, the limit model is not identified, or a rank bookkeeping
    /// check did not hold. No other closed form changes this.
    FaceUnavailable {
        /// The measured evidence behind the refusal.
        reason: String,
    },
}

impl RailFaceLimitOutcome {
    /// The limit, when one was formed.
    pub fn available(self) -> Option<RailFaceLimit> {
        match self {
            Self::Available(limit) => Some(*limit),
            _ => None,
        }
    }

}

/// The per-face penalty geometry shared by every closed form: the face's
/// penalties at unit strength, the survivors at their certified strengths,
/// and the all-unit sum whose spectrum backs the rank bookkeeping. Factored
/// out so the Gaussian and LAML forms can never disagree about what the face
/// IS.
pub(crate) struct FacePenaltySplit {
    /// The face, ascending and deduplicated.
    pub(crate) face_sorted: Vec<usize>,
    /// `Σ_{j∈F} S_j` at unit strength.
    pub(crate) s_face_unit: Array2<f64>,
    /// `Σ_{j∉F} λ_j S_j` at the certified survivor strengths.
    pub(crate) s_rest: Array2<f64>,
    /// `Σ_j S_j` at unit strength, for the pseudo-determinant rank split.
    pub(crate) s_unit_all: Array2<f64>,
    /// Full-width unit `S_j` for each face coordinate, in `face_sorted` order.
    pub(crate) face_penalties: Vec<Array2<f64>>,
}

/// Split the canonical penalties around a face. `Err` carries the typed
/// decline the caller returns verbatim.
pub(crate) fn split_face_penalties(
    penalties: &[CanonicalPenalty],
    rho: &Array1<f64>,
    face: &[usize],
    p: usize,
) -> Result<FacePenaltySplit, RailFaceLimitOutcome> {
    let n_penalties = penalties.len();
    if p == 0 || n_penalties == 0 || face.is_empty() || rho.len() < n_penalties {
        return Err(RailFaceLimitOutcome::OutsideClosedForm {
            reason: "face-limit inputs are shape-inconsistent or empty".to_string(),
        });
    }
    let mut face_sorted = face.to_vec();
    face_sorted.sort_unstable();
    face_sorted.dedup();
    if face_sorted.len() != face.len()
        || face_sorted.last().copied().unwrap_or(usize::MAX) >= n_penalties
    {
        return Err(RailFaceLimitOutcome::FaceUnavailable {
            reason: "the face repeats a coordinate or indexes outside the penalty layout"
                .to_string(),
        });
    }

    let mut s_face_unit = Array2::<f64>::zeros((p, p));
    let mut s_rest = Array2::<f64>::zeros((p, p));
    let mut s_unit_all = Array2::<f64>::zeros((p, p));
    let mut face_penalties: Vec<Array2<f64>> = face_sorted
        .iter()
        .map(|_| Array2::<f64>::zeros((p, p)))
        .collect();
    for (j, penalty) in penalties.iter().enumerate() {
        let lambda = rho[j].exp();
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(RailFaceLimitOutcome::OutsideClosedForm {
                reason: "a smoothing parameter does not exponentiate to a usable lambda"
                    .to_string(),
            });
        }
        let cols = penalty.col_range.clone();
        if cols.end > p {
            return Err(RailFaceLimitOutcome::OutsideClosedForm {
                reason: "a penalty's column range runs past the coefficient layout".to_string(),
            });
        }
        let slot = face_sorted.iter().position(|&f| f == j);
        for (li, gi) in cols.clone().enumerate() {
            for (lj, gj) in cols.clone().enumerate() {
                let value = penalty.local[[li, lj]];
                s_unit_all[[gi, gj]] += value;
                match slot {
                    Some(idx) => {
                        s_face_unit[[gi, gj]] += value;
                        face_penalties[idx][[gi, gj]] += value;
                    }
                    None => s_rest[[gi, gj]] += lambda * value,
                }
            }
        }
    }
    Ok(FacePenaltySplit {
        face_sorted,
        s_face_unit,
        s_rest,
        s_unit_all,
        face_penalties,
    })
}

/// Orthonormal bases of the subspace a face releases (`Q`) and the one it
/// pins (`Z`), from the face's unit-strength penalty sum.
pub(crate) struct FaceBases {
    pub(crate) q_basis: Array2<f64>,
    pub(crate) z_basis: Array2<f64>,
}

pub(crate) fn face_release_bases(
    s_face_unit: &Array2<f64>,
) -> Result<FaceBases, RailFaceLimitOutcome> {
    let (face_values, face_vectors) = match symmetric_eigh(s_face_unit) {
        Some(pair) => pair,
        None => {
            return Err(RailFaceLimitOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the face geometry failed".to_string(),
            });
        }
    };
    let face_cut = range_null_split(&face_values);
    let released_cols: Vec<usize> = (0..face_values.len())
        .filter(|&i| face_values[i] > face_cut)
        .collect();
    let pinned_cols: Vec<usize> = (0..face_values.len())
        .filter(|&i| face_values[i] <= face_cut)
        .collect();
    if released_cols.is_empty() {
        // The face penalizes nothing: λ=∞ there is not a statement about
        // the model at all.
        return Err(RailFaceLimitOutcome::FaceUnavailable {
            reason: "the face's penalties release no direction: lambda=infinity there says nothing about the model".to_string(),
        });
    }
    Ok(FaceBases {
        q_basis: basis_columns(&face_vectors, &released_cols),
        z_basis: basis_columns(&face_vectors, &pinned_cols),
    })
}

/// Eigenpairs of the pinned block `ZᵀKZ` — the λ=∞ model's curvature — with
/// the identification gate and the conditioning the certificate's margin is
/// scaled by. `Ok(None)` when the face pins nothing.
pub(crate) struct PinnedBlock {
    pub(crate) values: Array1<f64>,
    pub(crate) vectors: Array2<f64>,
    pub(crate) conditioning: f64,
}

pub(crate) fn pinned_block_eigenpairs(
    k_matrix: &Array2<f64>,
    z_basis: &Array2<f64>,
) -> Result<Option<PinnedBlock>, RailFaceLimitOutcome> {
    if z_basis.ncols() == 0 {
        return Ok(None);
    }
    let kzz = z_basis.t().dot(k_matrix).dot(z_basis);
    let (values, vectors) = match symmetric_eigh(&kzz) {
        Some(pair) => pair,
        None => {
            return Err(RailFaceLimitOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the face geometry failed".to_string(),
            });
        }
    };
    let largest = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let smallest = values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
    if !(smallest > f64::EPSILON.sqrt() * largest) || !(largest > 0.0) {
        // The λ=∞ model is not identified; there is no limit fit to
        // certify against.
        return Err(RailFaceLimitOutcome::FaceUnavailable {
            reason: format!(
                "the lambda=infinity model is not identified: reduced Hessian spectrum spans {smallest:.3e}..{largest:.3e}"
            ),
        });
    }
    Ok(Some(PinnedBlock {
        conditioning: largest / smallest,
        values,
        vectors,
    }))
}

/// Everything the final assembly needs beyond the face geometry: the
/// criterion's data curvature `K = I + S_R`, the limit fit and its score, and
/// the two knobs that differ between closed forms — the dispersion the fit
/// term is divided by (`φ̂` for profiled-Gaussian REML, exactly `1` for
/// fixed-dispersion LAML) and the optional curvature-drift vector `d`, zero
/// exactly when the working weights do not move with the coefficients.
struct FaceLimitAssembly<'a> {
    rho: &'a Array1<f64>,
    split: FacePenaltySplit,
    bases: FaceBases,
    pinned: Option<PinnedBlock>,
    k_matrix: Array2<f64>,
    limit_beta: Array1<f64>,
    limit_score: Array1<f64>,
    /// Natural magnitude the pinned-stationarity residual is judged against.
    score_scale: f64,
    /// Absolute stationarity tolerance for the pinned residual: `√ε·scale`
    /// when the limit fit is a direct linear solve (Gaussian), the inner
    /// solver's own certified standard when it is an iterative P-IRLS mode —
    /// judging an iterative fit by `√ε` would decline every converged limit.
    stationarity_tolerance: f64,
    dispersion: f64,
    curvature_drift: Option<Array1<f64>>,
}

/// The shared tail of both closed forms: pinned-stationarity check, the two
/// Schur complements, the pseudo-determinant rank bookkeeping, and the
/// first-order form itself.
fn assemble_face_limit(input: FaceLimitAssembly<'_>) -> RailFaceLimitOutcome {
    let FaceLimitAssembly {
        rho,
        split,
        bases,
        pinned,
        k_matrix,
        limit_beta,
        limit_score,
        score_scale,
        stationarity_tolerance,
        dispersion,
        curvature_drift,
    } = input;
    let FacePenaltySplit {
        face_sorted,
        s_face_unit: _,
        s_rest,
        s_unit_all,
        face_penalties,
    } = split;
    let FaceBases { q_basis, z_basis } = bases;
    let released = q_basis.ncols();
    let pinned_count = z_basis.ncols();

    if pinned_count > 0 {
        // The limit fit is stationary in the pinned directions by
        // construction; a residual there means the reduced solve did not
        // hold, so the rest of the algebra is not trustworthy either.
        let pinned_residual = z_basis.t().dot(&limit_score);
        if pinned_residual.dot(&pinned_residual).sqrt() > stationarity_tolerance {
            return RailFaceLimitOutcome::FaceUnavailable {
                reason: format!(
                    "the limit fit is not stationary in the pinned directions: residual {:.3e} against tolerance {stationarity_tolerance:.3e} (scale {score_scale:.3e})",
                    pinned_residual.dot(&pinned_residual).sqrt()
                ),
            };
        }
    }
    let released_score = q_basis.t().dot(&limit_score);

    // ── the two logdets' analytic first-order content ───────────────────
    // `½log|H|` leaves `Schur_Z(K)` behind once its `log|QᵀS_FQ|`
    // divergence is removed; `−½log|S_λ|₊` leaves `Schur_Z(S_R)`. Their
    // divergences are the same matrix and cancel exactly.
    let qkq = q_basis.t().dot(&k_matrix).dot(&q_basis);
    let (schur_k, conditioning) = match pinned.as_ref() {
        None => (qkq, 1.0_f64),
        Some(block) => {
            let qkz = q_basis.t().dot(&k_matrix).dot(&z_basis);
            let kzz_inverse = spectral_pseudo_inverse(
                &block.values,
                &block.vectors,
                f64::EPSILON.sqrt()
                    * block.values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())),
            );
            (qkq - qkz.dot(&kzz_inverse).dot(&qkz.t()), block.conditioning)
        }
    };
    let qsq = q_basis.t().dot(&s_rest).dot(&q_basis);
    let (schur_s_rest, surviving_rank, survivor_conditioning) = if pinned_count == 0 {
        (qsq, 0usize, 1.0_f64)
    } else {
        let zsz = z_basis.t().dot(&s_rest).dot(&z_basis);
        let (values, vectors) = match symmetric_eigh(&zsz) {
            Some(pair) => pair,
            None => {
                return RailFaceLimitOutcome::FaceUnavailable {
                    reason: "a symmetric eigendecomposition of the face geometry failed"
                        .to_string(),
                };
            }
        };
        let cut = range_null_split(&values);
        let kept: Vec<f64> = values.iter().copied().filter(|v| *v > cut).collect();
        let cond = match (
            kept.iter().fold(0.0_f64, |acc, v| acc.max(*v)),
            kept.iter().fold(f64::INFINITY, |acc, v| acc.min(*v)),
        ) {
            (hi, lo) if lo > 0.0 && hi.is_finite() => hi / lo,
            _ => 1.0,
        };
        let zsz_pseudo = spectral_pseudo_inverse(&values, &vectors, cut);
        let qsz = q_basis.t().dot(&s_rest).dot(&z_basis);
        (qsq - qsz.dot(&zsz_pseudo).dot(&qsz.t()), kept.len(), cond)
    };
    // The pseudo-determinant split `log|S_λ|₊ = log|QᵀS_λQ| +
    // log|Schur_Q(S_λ)|₊` needs the ranks to add up. That is an identity
    // (`dim(range S_R + range S_F) = q + dim P_N range S_R`), so a
    // mismatch means a rank determination is unreliable here.
    let (all_values, _) = match symmetric_eigh(&s_unit_all) {
        Some(pair) => pair,
        None => {
            return RailFaceLimitOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the face geometry failed".to_string(),
            };
        }
    };
    let all_cut = range_null_split(&all_values);
    let penalty_rank = all_values.iter().filter(|v| **v > all_cut).count();
    if penalty_rank != released + surviving_rank {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: format!(
                "rank bookkeeping does not close: rank(S_lambda)={penalty_rank} but released={released} + surviving={surviving_rank}"
            ),
        };
    }

    // ── the first-order form ────────────────────────────────────────────
    let mut first_order_form = &schur_k - &schur_s_rest;
    for a in 0..released {
        for b in 0..released {
            first_order_form[[a, b]] -= released_score[a] * released_score[b] / dispersion;
        }
    }
    let released_curvature_drift = curvature_drift.map(|drift| q_basis.t().dot(&drift));
    if let Some(drift_q) = released_curvature_drift.as_ref() {
        // The LAML rank-2 correction: the Laplace logdet's curvature moves
        // with β̂, and the face's O(λ⁻¹) coefficient offset carries that
        // motion into first order (see the module derivation).
        for a in 0..released {
            for b in 0..released {
                first_order_form[[a, b]] +=
                    released_score[a] * drift_q[b] + drift_q[a] * released_score[b];
            }
        }
    }
    if first_order_form.iter().any(|v| !v.is_finite()) {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: "the assembled first-order form is not finite".to_string(),
        };
    }

    let released_penalties: Vec<Array2<f64>> = face_penalties
        .iter()
        .map(|s| q_basis.t().dot(s).dot(&q_basis))
        .collect();
    let face_rho: Vec<f64> = face_sorted.iter().map(|&j| rho[j]).collect();

    // ── the rounding error of the assembled form ────────────────────────
    // Frobenius norms bound the spectral ones, so the bound stays rigorous.
    let form_conditioning = conditioning.max(survivor_conditioning).max(1.0);
    let frobenius = |m: &Array2<f64>| m.iter().map(|v| v * v).sum::<f64>().sqrt();
    let score_norm = released_score.dot(&released_score).sqrt();
    let drift_norm = released_curvature_drift
        .as_ref()
        .map_or(0.0, |drift_q| drift_q.dot(drift_q).sqrt());
    let form_error_bound = accumulation_growth(k_matrix.nrows())
        * ((frobenius(&k_matrix) + frobenius(&s_rest)) * (1.0 + form_conditioning)
            + score_norm * score_norm / dispersion
            + 2.0 * score_norm * drift_norm);
    if !form_error_bound.is_finite() {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: "the assembled first-order form's rounding bound is not finite".to_string(),
        };
    }

    RailFaceLimitOutcome::Available(Box::new(RailFaceLimit {
        face: face_sorted,
        face_rho,
        first_order_form,
        released_penalties,
        released_score,
        form_error_bound,
        limit_beta,
        limit_dispersion: dispersion,
        released_curvature_drift,
    }))
}

/// Build the analytic λ→∞ face-limit data from a model's parts.
///
/// This is the Gaussian-identity closed form derived at the top of this module,
/// factored so any caller that owns a design, a response, prior weights and
/// canonical penalties can obtain the λ=∞ limit — the null-space-restricted
/// fit, its profiled dispersion, the limit score, and the first-order form —
/// without going through a `RemlState`. A caller whose criterion is NOT the
/// profiled-Gaussian REML must not use it: with `β̂`-dependent working weights
/// the logdet terms gain a third-derivative contribution at the same order,
/// which is `laml_rail_face_limit`'s rank-2 term.
///
/// `response` must already be net of any offset. `penalties` are in ρ-block
/// order, so `rho[j]` is `penalties[j]`'s log smoothing parameter, and `face`
/// indexes the same order. A decline is typed, never an error: this
/// certificate simply has nothing to say there.
///
/// The returned `limit_beta` is the λ=∞ fit itself, which is also the canonical
/// maximal-smoothing anchor a continuation can start from (#2366) instead of
/// solving at a large-but-finite ρ.
pub(crate) fn gaussian_rail_face_limit(
    design: ArrayView2<'_, f64>,
    response: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    penalties: &[CanonicalPenalty],
    rho: &Array1<f64>,
    face: &[usize],
) -> RailFaceLimitOutcome {
    let p = design.ncols();
    let n = design.nrows();
    if p == 0 || n == 0 || response.len() != n || weights.len() != n {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "face-limit inputs are shape-inconsistent or empty".to_string(),
        };
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "prior weights are not finite and non-negative".to_string(),
        };
    }
    let split = match split_face_penalties(penalties, rho, face, p) {
        Ok(split) => split,
        Err(outcome) => return outcome,
    };
    let bases = match face_release_bases(&split.s_face_unit) {
        Ok(bases) => bases,
        Err(outcome) => return outcome,
    };
    let pinned = bases.z_basis.ncols();

    // ── weighted design cross-products ──────────────────────────────────
    let weight_sqrt: Array1<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let mut x_scaled = design.to_owned();
    for (i, mut row) in x_scaled.axis_iter_mut(Axis(0)).enumerate() {
        let scale = weight_sqrt[i];
        row.mapv_inplace(|v| v * scale);
    }
    let response_scaled: Array1<f64> = response
        .iter()
        .zip(weight_sqrt.iter())
        .map(|(y, w)| y * w)
        .collect();
    let xtwx = x_scaled.t().dot(&x_scaled);
    let xtwy = x_scaled.t().dot(&response_scaled);
    let k_matrix = &xtwx + &split.s_rest;

    // ── the limit fit: the model restricted to the pinned subspace ──────
    let pinned_block = match pinned_block_eigenpairs(&k_matrix, &bases.z_basis) {
        Ok(block) => block,
        Err(outcome) => return outcome,
    };
    let mut limit_beta = Array1::<f64>::zeros(p);
    if let Some(block) = pinned_block.as_ref() {
        let rhs = bases.z_basis.t().dot(&xtwy);
        let rotated = block.vectors.t().dot(&rhs);
        let mut alpha = Array1::<f64>::zeros(pinned);
        for i in 0..pinned {
            alpha[i] = rotated[i] / block.values[i];
        }
        limit_beta = bases.z_basis.dot(&block.vectors.dot(&alpha));
    }

    // ── the profiled dispersion at the limit fit ────────────────────────
    let fitted = design.dot(&limit_beta);
    let mut weighted_rss = 0.0_f64;
    for i in 0..n {
        let residual = response[i] - fitted[i];
        weighted_rss += weights[i] * residual * residual;
    }
    let penalty_energy = limit_beta.dot(&split.s_rest.dot(&limit_beta));
    let penalized_deviance = weighted_rss + penalty_energy;
    // The profiled scale's denominator must be the CRITERION's own
    // degrees-of-freedom bookkeeping: `M_p = p − rank(Σ_k S_k)`, the JOINT
    // structural rank the criterion's penalty pseudo-logdet reports
    // (`structural_rank_from_canonical_penalties`, every λ_k > 0 on the
    // face). The sum of the per-penalty ranks agrees only when the ranges
    // are disjoint; when they OVERLAP it over-counts the rank, under-counts
    // `M_p`, and mis-states the profiled scale the certificate reproduces.
    let criterion_penalty_rank = match gam_terms::construction::balanced_penalty_structural_rank(
        penalties
            .iter()
            .map(|penalty| (penalty.local_ref().view(), penalty.col_range.clone())),
        p,
    ) {
        Ok(rank) => rank,
        Err(error) => {
            return RailFaceLimitOutcome::FaceUnavailable {
                reason: format!("the criterion's joint penalty rank is unavailable: {error}"),
            };
        }
    };
    let null_dim = p.saturating_sub(criterion_penalty_rank);
    if n <= null_dim {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: format!("no residual degrees of freedom at the limit: n={n} <= M_p={null_dim}"),
        };
    }
    // The REML profiled scale, `D_p/(n − M_p)`, evaluated AT the limit fit.
    // `M_p` does not move along the face, so the same denominator holds at
    // ρ̂ and at λ=∞.
    let dispersion = penalized_deviance / ((n - null_dim) as f64);
    if !dispersion.is_finite() || dispersion <= 0.0 {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: format!("the limit fit's profiled dispersion is unusable: {dispersion:.3e}"),
        };
    }

    // ── the limit score ─────────────────────────────────────────────────
    let limit_score = &xtwy - &k_matrix.dot(&limit_beta);
    let score_scale = xtwy
        .dot(&xtwy)
        .sqrt()
        .max(limit_score.dot(&limit_score).sqrt())
        .max(1.0);

    assemble_face_limit(FaceLimitAssembly {
        rho,
        split,
        bases,
        pinned: pinned_block,
        k_matrix,
        limit_beta,
        limit_score,
        score_scale,
        // The Gaussian limit fit is a direct linear solve, so its pinned
        // residual is roundoff-scale and `√ε` is the honest standard.
        stationarity_tolerance: f64::EPSILON.sqrt() * score_scale,
        dispersion,
        curvature_drift: None,
    })
}

/// Row bundle the LAML closed form reads off the CONVERGED λ=∞ limit fit.
///
/// All row vectors are length-`n` in design row order, in the criterion's own
/// units — prior weights folded in, exactly as the inner solver reports them:
///
/// * `working_weights` — the observed-information diagonal
///   `w_i = −∂²ℓ_i/∂η_i²` at the limit fit, the same `W` whose `XᵀWX` the
///   criterion's `½log|H|` uses;
/// * `score_residuals` — `u_i = ∂ℓ_i/∂η_i`, so `∇ℓ = Xᵀu`;
/// * `weight_eta_derivatives` — `c_i = dW_i/dη_i`, the third-derivative array
///   the exact LAML ρ-gradient already consumes (`solve_c_array`).
pub struct LamlFaceParts<'a> {
    /// The λ=∞ fit in the model's own coefficient basis (`Zα̂`).
    pub limit_beta: Array1<f64>,
    pub working_weights: ArrayView1<'a, f64>,
    pub score_residuals: ArrayView1<'a, f64>,
    pub weight_eta_derivatives: ArrayView1<'a, f64>,
    /// The RELATIVE stationarity standard the limit fit's mode was certified
    /// to by its own solver (the inner KKT tolerance). The pinned-direction
    /// residual check applies this same scale-invariant standard in the model
    /// basis — an iterative mode judged by the Gaussian path's `√ε` would be
    /// declined every time, converged or not.
    pub convergence_tolerance: f64,
}

/// Build the analytic λ→∞ face-limit data for a fixed-unit-dispersion LAML
/// criterion (see the module derivation: `C_LAML` adds the symmetric rank-2
/// curvature-drift term to the Gaussian form and drops the `1/φ̂` fit-term
/// scale).
///
/// The caller supplies the converged limit fit — the model restricted to the
/// face's null space, solved by the SAME inner engine the criterion itself
/// uses — through [`LamlFaceParts`]. Everything here is then exact arithmetic
/// on those outputs: no probe, no finite difference, and no evaluation at a
/// large λ anywhere.
pub(crate) fn laml_rail_face_limit(
    design: ArrayView2<'_, f64>,
    penalties: &[CanonicalPenalty],
    rho: &Array1<f64>,
    face: &[usize],
    parts: LamlFaceParts<'_>,
) -> RailFaceLimitOutcome {
    let p = design.ncols();
    let n = design.nrows();
    if p == 0
        || n == 0
        || parts.limit_beta.len() != p
        || parts.working_weights.len() != n
        || parts.score_residuals.len() != n
        || parts.weight_eta_derivatives.len() != n
    {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "face-limit inputs are shape-inconsistent or empty".to_string(),
        };
    }
    if parts.limit_beta.iter().any(|v| !v.is_finite())
        || parts.working_weights.iter().any(|v| !v.is_finite())
        || parts.score_residuals.iter().any(|v| !v.is_finite())
        || parts.weight_eta_derivatives.iter().any(|v| !v.is_finite())
    {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "the limit fit's row bundle is not finite".to_string(),
        };
    }
    let split = match split_face_penalties(penalties, rho, face, p) {
        Ok(split) => split,
        Err(outcome) => return outcome,
    };
    let bases = match face_release_bases(&split.s_face_unit) {
        Ok(bases) => bases,
        Err(outcome) => return outcome,
    };
    if bases.z_basis.ncols() == 0 {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: "the face releases every direction: the lambda=infinity limit model is empty"
                .to_string(),
        };
    }

    // I(β̂_∞) = Xᵀ·diag(w)·X, formed WITHOUT a square-root split: observed
    // weights may be negative on non-canonical links, and `H ≻ 0` is a
    // statement about the sum, not about the rows.
    let mut x_by_w = design.to_owned();
    for (i, mut row) in x_by_w.axis_iter_mut(Axis(0)).enumerate() {
        let w = parts.working_weights[i];
        row.mapv_inplace(|v| v * w);
    }
    let information = symmetrized(&design.t().dot(&x_by_w));
    // `H = XᵀWX + S_rest`: the operator the limit fit's criterion factors. No
    // PIRLS path adds a stabilization ridge (#2901 V22), so there is no δI to
    // carry into the pinned block.
    let k_matrix = &information + &split.s_rest;

    let pinned_block = match pinned_block_eigenpairs(&k_matrix, &bases.z_basis) {
        Ok(Some(block)) => block,
        // `z_basis.ncols() > 0` was checked above, so this arm is dead — but
        // a decline is the right dead-arm behavior for a certificate: never
        // abort a fit over an internal expectation.
        Ok(None) => {
            return RailFaceLimitOutcome::FaceUnavailable {
                reason: "the face releases every direction: the lambda=infinity limit model is empty"
                    .to_string(),
            };
        }
        Err(outcome) => return outcome,
    };

    // g_c = ∇ℓ(β̂_∞) − S_R β̂_∞: the Lagrange force the face carries.
    let grad_ell = design.t().dot(&parts.score_residuals);
    // With `H = XᵀWX + S_rest` the limit fit's stationarity condition is
    // `∇ℓ − S_restβ = 0`: no ridge term pulls on the score (#2901 V22).
    let penalty_pull = split.s_rest.dot(&parts.limit_beta);
    let limit_score = &grad_ell - &penalty_pull;
    let score_scale = grad_ell
        .dot(&grad_ell)
        .sqrt()
        .max(limit_score.dot(&limit_score).sqrt())
        .max(1.0);
    if !(parts.convergence_tolerance.is_finite() && parts.convergence_tolerance > 0.0) {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "the limit fit's convergence tolerance is not usable".to_string(),
        };
    }
    // The inner solver certified `‖g‖ ≤ tol·(1 + natural scale)` in its own
    // frame; apply the identical scale-invariant standard to the rebuilt
    // gradient in this frame, floored by the Gaussian path's `√ε·scale` so a
    // tolerance tighter than roundoff cannot decline an exact solve.
    let natural_scale =
        1.0 + grad_ell.dot(&grad_ell).sqrt() + penalty_pull.dot(&penalty_pull).sqrt();
    let stationarity_tolerance = (parts.convergence_tolerance * natural_scale)
        .max(f64::EPSILON.sqrt() * score_scale);

    // The curvature-drift vector `d = ½Xᵀ(c ⊙ a)`, `a` the LIMIT model's
    // leverage `a_i = x_iᵀ Z (ZᵀKZ)⁻¹ Zᵀ x_i` — assembled from the pinned
    // block's own eigenpairs so it prices exactly the inverse the Schur
    // complement uses.
    let xz = design.dot(&bases.z_basis);
    let rotated = xz.dot(&pinned_block.vectors);
    let mut drift_rows = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut leverage = 0.0_f64;
        for k in 0..rotated.ncols() {
            leverage += rotated[[i, k]] * rotated[[i, k]] / pinned_block.values[k];
        }
        drift_rows[i] = 0.5 * parts.weight_eta_derivatives[i] * leverage;
    }
    let raw_drift = design.t().dot(&drift_rows);
    // The K-oblique reduction (module derivation): the face's coefficient
    // offset has an O(λ⁻¹) pinned component, so the drift acts through
    // `d̃ = d − KZ(ZᵀKZ)⁻¹Zᵀd` — the same reduction the Schur complement
    // applies to K itself. The orthogonal compression alone is a
    // first-order error.
    let pinned_drift = bases.z_basis.t().dot(&raw_drift);
    let rotated_pinned = pinned_block.vectors.t().dot(&pinned_drift);
    let mut solved = Array1::<f64>::zeros(rotated_pinned.len());
    for k in 0..rotated_pinned.len() {
        solved[k] = rotated_pinned[k] / pinned_block.values[k];
    }
    let oblique = k_matrix.dot(&bases.z_basis.dot(&pinned_block.vectors.dot(&solved)));
    let curvature_drift = &raw_drift - &oblique;

    assemble_face_limit(FaceLimitAssembly {
        rho,
        split,
        bases,
        pinned: Some(pinned_block),
        k_matrix,
        limit_beta: parts.limit_beta,
        limit_score,
        score_scale,
        stationarity_tolerance,
        // The family's dispersion is exactly 1 for the gated families, so the
        // fit term's `−g_Qg_Qᵀ` carries no divisor and the reported limit
        // dispersion is the family's own.
        dispersion: 1.0,
        curvature_drift: Some(curvature_drift),
    })
}

#[cfg(test)]
mod rail_face_tests {
    use super::*;

    fn limit(
        face: Vec<usize>,
        face_rho: Vec<f64>,
        form: Array2<f64>,
        penalties: Vec<Array2<f64>>,
        score: Array1<f64>,
    ) -> RailFaceLimit {
        RailFaceLimit {
            face,
            face_rho,
            first_order_form: form,
            released_penalties: penalties,
            released_score: score,
            form_error_bound: 0.0,
            limit_beta: Array1::zeros(0),
            limit_dispersion: 1.0,
            released_curvature_drift: None,
        }
    }

    fn diag(values: &[f64]) -> Array2<f64> {
        let n = values.len();
        let mut m = Array2::<f64>::zeros((n, n));
        for (i, &v) in values.iter().enumerate() {
            m[[i, i]] = v;
        }
        m
    }

    /// One railed coordinate, diagonal geometry: the analytic pencil constant
    /// is `½·Σ C_aa/σ_a` in closed form, the value gap is `c·e^{−ρ}` (the tail
    /// law's own integral), and the estimand travel is `‖(λA)⁻¹g‖`.
    #[test]
    fn single_coordinate_face_matches_the_closed_form_tail_law() {
        let rho = 30.0_f64;
        let lim = limit(
            vec![1],
            vec![rho],
            diag(&[4.0, 6.0]),
            vec![diag(&[2.0, 3.0])],
            Array1::from(vec![1.0, 2.0]),
        );
        match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => {
                let expected = 0.5 * (4.0 / 2.0 + 6.0 / 3.0);
                assert!(
                    (proof.tail_constants[0] - expected).abs() <= 1.0e-12 * expected,
                    "analytic c_j={} should equal the closed form {expected}",
                    proof.tail_constants[0]
                );
                assert_eq!(proof.coordinate_kinds[0], FaceCoordinateKind::StrictOutward);
                assert!(
                    (proof.joint_tail_constant - expected).abs() <= 1.0e-12 * expected,
                    "a one-coordinate face's joint law is its own law"
                );
                // V(ρ̂) − V_∞ = ½tr((λA)⁻¹C) = c·e^{−ρ}: the tail law's exact
                // remaining value gap.
                let expected_gap = expected * (-rho).exp();
                assert!(
                    (proof.value_gap - expected_gap).abs() <= 1.0e-12 * expected_gap,
                    "value gap {} should be c·e^(−ρ) = {expected_gap}",
                    proof.value_gap
                );
                // ‖(λA)⁻¹g‖ with λ = e^ρ.
                let lambda = rho.exp();
                let expected_travel =
                    ((1.0 / (lambda * 2.0)).powi(2) + (2.0 / (lambda * 3.0)).powi(2)).sqrt();
                assert!(
                    (proof.estimand_travel - expected_travel).abs()
                        <= 1.0e-9 * expected_travel.max(f64::MIN_POSITIVE),
                    "estimand travel {} should be ‖(λA)⁻¹g‖ = {expected_travel}",
                    proof.estimand_travel
                );
            }
            other => panic!("a positive-definite face form must certify, got {other:?}"),
        }
    }

    /// A single-penalty face moves its whole released range at once, so the
    /// first-order change off it is `f(t) = t·½tr(A⁻¹C)` EXACTLY, whatever the
    /// sign pattern of `C`: a negative eigen-direction of `C` that the one
    /// smoothing parameter cannot isolate is not a way off the face. The
    /// positive-definiteness gate refused this face; the KKT test certifies it
    /// on the independent-ranges route with the measured slope as its
    /// statistic, and the priced value gap is still the tail law `c·e^{−ρ}`.
    #[test]
    fn single_penalty_face_with_indefinite_form_certifies_by_its_slope() {
        let rho = 25.0_f64;
        let lim = limit(
            vec![0],
            vec![rho],
            diag(&[4.0, -1.0e-3]),
            vec![diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        let proof = match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => proof,
            other => panic!("a positive slope proves a one-penalty face, got {other:?}"),
        };
        assert_eq!(proof.route, FacePositivityRoute::IndependentRanges);
        assert!(proof.min_curvature < 0.0, "the fixture's form is indefinite");
        let expected = 0.5 * (4.0 - 1.0e-3);
        assert!(
            (proof.tail_constants[0] - expected).abs() <= 1.0e-12 * expected,
            "c={} should be ½tr(A⁻¹C) = {expected}",
            proof.tail_constants[0]
        );
        assert_eq!(proof.statistic, proof.tail_constants[0]);
        assert!(proof.statistic > proof.band && proof.band >= 0.0);
        let expected_gap = expected * (-rho).exp();
        assert!(
            (proof.value_gap - expected_gap).abs() <= 1.0e-12 * expected_gap,
            "value gap {} should be c·e^(−ρ) = {expected_gap}",
            proof.value_gap
        );
    }

    /// A measured NEGATIVE slope means releasing the coordinate lowers the
    /// criterion: λ=∞ is not the optimum and the face must refuse, naming it.
    #[test]
    fn negative_slope_refutes_the_face() {
        let lim = limit(
            vec![0],
            vec![25.0],
            diag(&[-1.0, 0.5]),
            vec![diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        match certify_rail_face(&lim) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("lowers the criterion") && reason.contains("c="),
                "refusal should name the descending coordinate and its slope: {reason}"
            ),
            other => panic!("a negative slope must not certify, got {other:?}"),
        }
    }

    /// A slope inside its rounding band is not a proof: `c = 0` leaves the
    /// criterion unchanged to first order, so the sign is not a measured fact
    /// and the certificate refuses rather than minting a maybe-optimum.
    #[test]
    fn flat_slope_is_not_a_proof() {
        let lim = limit(
            vec![0],
            vec![25.0],
            diag(&[1.0, -1.0]),
            vec![diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        match certify_rail_face(&lim) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("unresolved first-order slope"),
                "refusal should say the slope's sign is unresolved: {reason}"
            ),
            other => panic!("a flat slope must not certify, got {other:?}"),
        }
    }

    /// Two penalties with linearly independent but NON-orthogonal released
    /// ranges and an indefinite `C`. The congruence argument makes
    /// `½tr((A_0/t_0 + A_1/t_1)⁻¹C) = c_0 t_0 + c_1 t_1` exactly, so the
    /// per-coordinate slopes decide the face — checked here against the
    /// directly evaluated expansion at lopsided weightings.
    #[test]
    fn independent_ranges_face_is_linear_in_t_and_certifies_on_its_slopes() {
        let mut a0 = Array2::<f64>::zeros((3, 3));
        a0[[0, 0]] = 2.0;
        a0[[0, 1]] = 0.5;
        a0[[1, 0]] = 0.5;
        a0[[1, 1]] = 1.0;
        let v = [0.3, 0.0, 1.0];
        let mut a1 = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                a1[[i, j]] = 3.0 * v[i] * v[j];
            }
        }
        let mut form = diag(&[4.0, -1.0, 2.0]);
        form[[0, 1]] = 1.0;
        form[[1, 0]] = 1.0;
        form[[1, 2]] = 0.5;
        form[[2, 1]] = 0.5;
        let lim = limit(
            vec![0, 1],
            vec![20.0, 22.0],
            form.clone(),
            vec![a0.clone(), a1.clone()],
            Array1::from(vec![0.1, -0.2, 0.3]),
        );
        let proof = match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => proof,
            other => panic!("positive slopes on independent ranges must certify, got {other:?}"),
        };
        assert_eq!(proof.route, FacePositivityRoute::IndependentRanges);
        assert!(proof.min_curvature < 0.0, "the fixture's form is indefinite");
        // Releasing coordinate 1 alone frees `null(A_0) = e_3`: c_1 = ½·C₃₃/3.
        assert!((proof.tail_constants[1] - 1.0 / 3.0).abs() <= 1.0e-12);
        for (t0, t1) in [(1.0, 1.0e-3), (1.0e-3, 1.0), (0.3, 0.7), (2.0, 5.0)] {
            let mut mixed = Array2::<f64>::zeros((3, 3));
            mixed.scaled_add(1.0 / t0, &a0);
            mixed.scaled_add(1.0 / t1, &a1);
            let direct = half_trace_inverse_product(&mixed, &form)
                .expect("a positive weighting of independent ranges is invertible");
            let linear = proof.tail_constants[0] * t0 + proof.tail_constants[1] * t1;
            assert!(
                (direct - linear).abs() <= 1.0e-10 * linear.abs(),
                "the expansion at t=({t0},{t1}) is {direct}, the linear law {linear}"
            );
            assert!(direct > 0.0);
        }
    }

    /// With OVERLAPPING released ranges and an indefinite `C`, a coordinate
    /// that is unidentified at the face (`A_0 = I` already pins everything
    /// coordinate 1 penalizes) makes `f` vanish at its simplex vertex: no cell
    /// touching it bounds the law away from zero, and the simplex bound says
    /// so — after checking the centre, where `f = 0.65` does not refute.
    #[test]
    fn overlapping_face_with_an_unidentified_vertex_refuses_by_name() {
        let lim = limit(
            vec![0, 1],
            vec![25.0, 25.0],
            diag(&[4.0, -0.1]),
            vec![diag(&[1.0, 1.0]), Array2::from_elem((2, 2), 1.0)],
            Array1::from(vec![0.0, 0.0]),
        );
        match certify_rail_face(&lim) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("ranges overlap") && reason.contains("unidentified at the face"),
                "refusal should name the overlap and the vanishing vertex: {reason}"
            ),
            other => panic!("an unidentified overlapping vertex cannot certify, got {other:?}"),
        }
    }

    /// `A_1 = diag(1,1,0)`, `A_2 = diag(0,1,1)`, `C = diag(1, c, 1)`: the
    /// ranges share `e_2`, and `M(t) = diag(t_1, t_1t_2/(t_1+t_2), t_2)`, so
    /// `f = ½(t_1 + t_2 + c·t_1t_2/(t_1+t_2))` in closed form. Both axis laws
    /// are `c_j = ½` whatever `c`.
    fn shared_direction_face(c: f64) -> RailFaceLimit {
        limit(
            vec![3, 5],
            vec![22.0, 24.0],
            diag(&[1.0, c, 1.0]),
            vec![diag(&[1.0, 1.0, 0.0]), diag(&[0.0, 1.0, 1.0])],
            Array1::from(vec![0.0, 0.0, 0.0]),
        )
    }

    /// `c = −3`: `C` is indefinite and the ranges overlap — the gate this
    /// replaces refused it — yet `f = ½(1 − 3t_1t_2) ≥ ⅛` on the simplex. The
    /// root cell's bound is exactly that minimum (`P ≡ ½`, and by Euler the
    /// tangent of the homogeneous `N` at the centre is `∇N·v = ⅜`).
    #[test]
    fn overlapping_face_positive_on_the_simplex_certifies_by_its_bound() {
        let proof = match certify_rail_face(&shared_direction_face(-3.0)) {
            RailFaceVerdict::Certified(proof) => proof,
            other => panic!("f ≥ ⅛ on the simplex must certify, got {other:?}"),
        };
        assert_eq!(proof.route, FacePositivityRoute::SimplexBound);
        assert!(proof.min_curvature < 0.0, "the fixture's form is indefinite");
        assert!(
            (proof.statistic - 0.125).abs() <= 1.0e-12,
            "the root cell's lower bound is min f = ⅛, got {}",
            proof.statistic
        );
        assert!(proof.statistic > proof.band && proof.band >= 0.0);
        for &c_j in proof.tail_constants.iter() {
            assert!((c_j - 0.5).abs() <= 1.0e-12, "axis law c_j={c_j} should be ½");
        }
    }

    /// `c = −5`: both axis laws are still `+½`, so a per-axis KKT test would
    /// certify — but `f(½,½) = ½(1 − 5/4) = −⅛`. Releasing the two penalties
    /// together lowers the criterion; the face is refuted at that point.
    #[test]
    fn overlapping_face_descending_jointly_is_refuted_despite_positive_axis_laws() {
        match certify_rail_face(&shared_direction_face(-5.0)) {
            RailFaceVerdict::Refused { reason } => {
                assert!(
                    reason.contains("lowers the criterion")
                        && reason.contains("t3=0.5000")
                        && reason.contains("t5=0.5000"),
                    "refusal should name the descending joint release: {reason}"
                );
                assert!(
                    reason.contains("-1.250000e-1"),
                    "refusal should carry the measured f(½,½) = −⅛: {reason}"
                );
            }
            other => panic!("a jointly descending face must not certify, got {other:?}"),
        }
    }

    /// Two copies of the same penalty (`A_1 = A_2 = I`): each coordinate is
    /// unidentified alone, but together `M(t) = t_1t_2/(t_1+t_2)·I`, so the
    /// face's law is `f = ½·t_1t_2/(t_1+t_2)·tr C`. With `tr C < 0` the joint
    /// release descends — refuted at the centre before the degenerate vertices
    /// are ever consulted; with `tr C > 0` it vanishes at both vertices and
    /// refuses as unresolved there.
    #[test]
    fn duplicated_penalty_face_is_decided_by_the_trace_of_its_form() {
        let duplicated = |form: Array2<f64>| {
            limit(
                vec![0, 1],
                vec![25.0, 25.0],
                form,
                vec![diag(&[1.0, 1.0]), diag(&[1.0, 1.0])],
                Array1::from(vec![0.0, 0.0]),
            )
        };
        match certify_rail_face(&duplicated(diag(&[1.0, -2.0]))) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("lowers the criterion") && reason.contains("-1.250000e-1"),
                "tr C < 0 must refute at the centre with f = ½·¼·(−1): {reason}"
            ),
            other => panic!("tr C < 0 descends jointly, got {other:?}"),
        }
        match certify_rail_face(&duplicated(diag(&[2.0, -1.0]))) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("unidentified at the face"),
                "tr C > 0 vanishes at the vertices and must refuse there: {reason}"
            ),
            other => panic!("a law vanishing at both vertices cannot certify, got {other:?}"),
        }
    }

    /// An asymmetric overlap the root cell cannot decide, so the bound has to
    /// REFINE: `A_2 = diag(0,4,1)` gives `f = ½(t_1 + t_2 − 8t_1t_2/(t_2+4t_1))`,
    /// minimized on the simplex at `t_1 = ⅓` with `f = 1/18`, while the root
    /// bound at vertex `e_2` is `½ − 0.64 < 0`. The certified statistic must be
    /// a genuine lower bound: `statistic − band ≤ min f`.
    #[test]
    fn overlapping_face_certifies_after_refinement_with_a_sound_lower_bound() {
        let lim = limit(
            vec![0, 1],
            vec![22.0, 24.0],
            diag(&[1.0, -8.0, 1.0]),
            vec![diag(&[1.0, 1.0, 0.0]), diag(&[0.0, 4.0, 1.0])],
            Array1::from(vec![0.0, 0.0, 0.0]),
        );
        let proof = match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => proof,
            other => panic!("min f = 1/18 > 0 must certify after refinement, got {other:?}"),
        };
        assert_eq!(proof.route, FacePositivityRoute::SimplexBound);
        assert!(proof.statistic > proof.band);
        let true_min = 1.0 / 18.0;
        assert!(
            proof.statistic - proof.band <= true_min + 1.0e-12,
            "the certified bound {} − {} exceeds the true minimum {true_min}",
            proof.statistic,
            proof.band
        );
    }

    /// The decision is basis-free: rotating every `A_j` and `C` by the same
    /// orthogonal `R` leaves `f` unchanged, so the rotated shared-direction face
    /// certifies with the same bound — exercising the non-diagonal arithmetic
    /// (the spectral split of `C`, the parallel sum, the pinned null spaces).
    /// Three penalties sharing one direction, each with a private one, make the
    /// simplex two-dimensional and its edge midpoints pin a coordinate exactly.
    #[test]
    fn rotated_three_penalty_overlap_certifies_and_refutes_as_its_closed_form() {
        // `A_j = e_j e_jᵀ + e_4e_4ᵀ`, `C = diag(1,1,1,c)`:
        // `f = ½(t_1 + t_2 + t_3 + c/Σ_j t_j⁻¹)`, minimized at the centre with
        // `f = ½(1 + c/9)`; every axis law is `½`.
        let (s, k) = (0.6_f64, 0.8_f64);
        let mut r = Array2::<f64>::zeros((4, 4));
        // A product of two plane rotations, (0,3) and (1,2).
        r[[0, 0]] = k;
        r[[0, 3]] = -s;
        r[[3, 0]] = s;
        r[[3, 3]] = k;
        r[[1, 1]] = s;
        r[[1, 2]] = -k;
        r[[2, 1]] = k;
        r[[2, 2]] = s;
        let rotate = |m: &Array2<f64>| r.dot(m).dot(&r.t());
        let penalties = |r: &dyn Fn(&Array2<f64>) -> Array2<f64>| {
            (0..3)
                .map(|j| {
                    let mut a = Array2::<f64>::zeros((4, 4));
                    a[[j, j]] = 1.0;
                    a[[3, 3]] = 1.0;
                    r(&a)
                })
                .collect::<Vec<_>>()
        };
        let face = |c: f64| {
            limit(
                vec![1, 2, 4],
                vec![20.0, 21.0, 22.0],
                rotate(&diag(&[1.0, 1.0, 1.0, c])),
                penalties(&rotate),
                Array1::from(vec![0.0, 0.0, 0.0, 0.0]),
            )
        };
        let proof = match certify_rail_face(&face(-6.0)) {
            RailFaceVerdict::Certified(proof) => proof,
            other => panic!("f ≥ ½(1 − 6/9) = ⅙ must certify, got {other:?}"),
        };
        assert_eq!(proof.route, FacePositivityRoute::SimplexBound);
        assert!(
            (proof.statistic - 1.0 / 6.0).abs() <= 1.0e-10,
            "the bound should reproduce the closed-form minimum ⅙, got {}",
            proof.statistic
        );
        match certify_rail_face(&face(-12.0)) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("lowers the criterion"),
                "f(centre) = ½(1 − 12/9) < 0 must refute: {reason}"
            ),
            other => panic!("a jointly descending three-penalty face must not certify, got {other:?}"),
        }
    }

    /// Multi-coordinate face with OVERLAPPING penalties. Coordinate 0 penalizes
    /// only the first released direction; coordinate 1 penalizes both. With
    /// coordinate 1 at λ=∞ every direction is already pinned, so releasing
    /// coordinate 0 alone changes the model not at all: it is unidentified at
    /// the face and reports `c_0 = 0` — while the face itself still certifies
    /// and coordinate 1 keeps its own strict law on the direction that only it
    /// pins.
    #[test]
    fn coalesced_face_coordinate_is_unidentified_not_a_refusal() {
        let lim = limit(
            vec![2, 3],
            vec![30.0, 30.0],
            diag(&[4.0, 6.0]),
            vec![diag(&[1.0, 0.0]), diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => {
                assert_eq!(proof.coordinate_kinds[0], FaceCoordinateKind::Unidentified);
                assert_eq!(proof.tail_constants[0], 0.0);
                assert_eq!(proof.coordinate_kinds[1], FaceCoordinateKind::StrictOutward);
                // Releasing coordinate 1 alone frees the direction coordinate 0
                // does not penalize: c_1 = ½·C_11/σ_11 = ½·6/1.
                assert!(
                    (proof.tail_constants[1] - 3.0).abs() <= 1.0e-12,
                    "c_1={} should be ½·C₁₁/σ₁₁ = 3",
                    proof.tail_constants[1]
                );
                // The joint ray releases both with unit weights: A = diag(2,1).
                let expected_joint = 0.5 * (4.0 / 2.0 + 6.0 / 1.0);
                assert!(
                    (proof.joint_tail_constant - expected_joint).abs() <= 1.0e-12 * expected_joint,
                    "joint constant {} should be {expected_joint}",
                    proof.joint_tail_constant
                );
            }
            other => panic!("a coalesced but positive-definite face must certify, got {other:?}"),
        }
    }

    /// The face proof covers every SUB-face, not just the coordinate axes and
    /// the joint ray: for any positive weights the released-penalty sum stays
    /// positive definite on the released subspace, so the predicted gap
    /// `½tr((Σλ_jA_j)⁻¹C)` is positive. Check that against the certified form
    /// on a deliberately lopsided, non-diagonal geometry.
    #[test]
    fn certified_face_raises_the_criterion_for_every_positive_weighting() {
        let mut form = Array2::<f64>::zeros((2, 2));
        form[[0, 0]] = 3.0;
        form[[0, 1]] = 1.0;
        form[[1, 0]] = 1.0;
        form[[1, 1]] = 2.0;
        let mut a1 = Array2::<f64>::zeros((2, 2));
        a1[[0, 0]] = 2.0;
        a1[[0, 1]] = 0.5;
        a1[[1, 0]] = 0.5;
        a1[[1, 1]] = 1.0;
        let lim = limit(
            vec![0, 1],
            vec![20.0, 22.0],
            form.clone(),
            vec![diag(&[1.0, 0.25]), a1],
            Array1::from(vec![0.3, -0.7]),
        );
        let proof = match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => proof,
            other => panic!("expected a certified face, got {other:?}"),
        };
        assert!(proof.min_curvature > 0.0);
        for (w0, w1) in [(1.0, 1.0e-6), (1.0e-6, 1.0), (1.0, 1.0), (17.0, 0.03)] {
            let mut mixed = Array2::<f64>::zeros((2, 2));
            mixed
                .iter_mut()
                .zip(lim.released_penalties[0].iter())
                .for_each(|(dst, src)| *dst += w0 * src);
            mixed
                .iter_mut()
                .zip(lim.released_penalties[1].iter())
                .for_each(|(dst, src)| *dst += w1 * src);
            let gap = half_trace_inverse_product(&mixed, &form)
                .expect("a positive weighting stays invertible on the released subspace");
            assert!(
                gap > 0.0,
                "releasing the face with weights ({w0}, {w1}) must raise the criterion, got {gap}"
            );
        }
    }

    /// Structural refusals never panic and never certify: an empty face, a
    /// dimension mismatch, and a non-finite form all decline with a reason.
    #[test]
    fn malformed_face_data_declines() {
        let empty = limit(
            Vec::new(),
            Vec::new(),
            diag(&[1.0]),
            Vec::new(),
            Array1::from(vec![0.0]),
        );
        assert!(matches!(
            certify_rail_face(&empty),
            RailFaceVerdict::Refused { .. }
        ));

        let mismatched = limit(
            vec![0],
            vec![30.0],
            diag(&[1.0, 1.0]),
            vec![diag(&[1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        assert!(matches!(
            certify_rail_face(&mismatched),
            RailFaceVerdict::Refused { .. }
        ));

        let non_finite = limit(
            vec![0],
            vec![30.0],
            diag(&[f64::NAN, 1.0]),
            vec![diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        assert!(matches!(
            certify_rail_face(&non_finite),
            RailFaceVerdict::Refused { .. }
        ));
    }

    /// The LAML form against a fully hand-expanded 2-coefficient fixture:
    /// `X = [[1,0],[1,1],[1,2]]`, one face penalty `diag(0,1)` (releases the
    /// slope, pins the intercept), no survivors. Every quantity — the Schur
    /// complement, the score term, the leverage, the rank-2 drift — is a
    /// scalar identity here, so a sign error or a misplaced factor of two in
    /// the assembly misses by a lot, not by rounding.
    #[test]
    fn laml_face_form_matches_the_hand_expansion() {
        use crate::estimate::PenaltySpec;
        let mut design = Array2::<f64>::zeros((3, 2));
        for (i, t) in [0.0_f64, 1.0, 2.0].iter().enumerate() {
            design[[i, 0]] = 1.0;
            design[[i, 1]] = *t;
        }
        let w = Array1::from(vec![1.0_f64, 0.5, 2.0]);
        // Pinned stationarity demands `Σu_i = 0` exactly (Z = intercept).
        let u = Array1::from(vec![0.3_f64, -0.5, 0.2]);
        let c = Array1::from(vec![0.4_f64, -0.2, 0.3]);
        let beta_inf = Array1::from(vec![0.7_f64, 0.0]);
        let mut bend = Array2::<f64>::zeros((2, 2));
        bend[[1, 1]] = 1.0;
        let (penalties, _) = gam_terms::construction::canonicalize_penalty_specs(
            &[PenaltySpec::Dense(bend)],
            &[1],
            2,
            "laml_face_hand_fixture",
        )
        .expect("canonicalize the hand fixture penalty");
        let rho = Array1::from(vec![20.0_f64]);

        // Hand expansion. K = XᵀWX with K00 = Σw, K01 = Σw·t, K11 = Σw·t²;
        // g = Xᵀu = [0, −0.1]; a_i = 1/K00; d₀ = ½·Σc_i/K00,
        // d₁ = ½·Σ(c_i·t_i)/K00. The drift acts through the K-oblique
        // reduction `d̃₁ = d₁ − (K₀₁/K₀₀)·d₀` — the pinned component of the
        // face's coefficient offset is first-order too, and this scalar case
        // makes the correction visible by hand: d₀ ≠ 0 and K₀₁ ≠ 0, so the
        // orthogonal compression alone would be measurably wrong.
        let k00 = 3.5_f64;
        let k01 = 4.5_f64;
        let k11 = 8.5_f64;
        let g1 = -0.1_f64;
        let d0 = 0.5 * (0.4 + (-0.2) + 0.3) / k00;
        let d1 = 0.5 * (0.0 * 0.4 + 1.0 * (-0.2) + 2.0 * 0.3) / k00;
        let d1_oblique = d1 - (k01 / k00) * d0;
        let c_hand = (k11 - k01 * k01 / k00) - g1 * g1 + 2.0 * g1 * d1_oblique;
        let expected_constant = 0.5 * c_hand;

        let limit = laml_rail_face_limit(
            design.view(),
            &penalties,
            &rho,
            &[0],
            LamlFaceParts {
                limit_beta: beta_inf.clone(),
                working_weights: w.view(),
                score_residuals: u.view(),
                weight_eta_derivatives: c.view(),
                convergence_tolerance: f64::EPSILON,
            },
        )
        .available()
        .expect("the hand fixture is inside the LAML closed form");
        let drift = limit
            .released_curvature_drift
            .as_ref()
            .expect("the LAML form must record its curvature drift");
        assert!(
            (drift[0].abs() - d1_oblique.abs()).abs() <= 1.0e-14,
            "released drift |d̃_Q|={} should be the hand value {}",
            drift[0].abs(),
            d1_oblique.abs()
        );
        match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => {
                assert!(
                    (proof.tail_constants[0] - expected_constant).abs()
                        <= 1.0e-12 * expected_constant,
                    "analytic c_0={} should equal the hand expansion {expected_constant}",
                    proof.tail_constants[0]
                );
            }
            other => panic!("the hand fixture must certify, got {other:?}"),
        }

        // The `c ≡ 0` member is the Gaussian reduction: the drift is recorded
        // as exactly zero and the constant loses exactly the rank-2 term.
        let zeros = Array1::<f64>::zeros(3);
        let reduced = laml_rail_face_limit(
            design.view(),
            &penalties,
            &rho,
            &[0],
            LamlFaceParts {
                limit_beta: beta_inf,
                working_weights: w.view(),
                score_residuals: u.view(),
                weight_eta_derivatives: zeros.view(),
                convergence_tolerance: f64::EPSILON,
            },
        )
        .available()
        .expect("the zero-drift fixture is inside the LAML closed form");
        assert_eq!(
            reduced
                .released_curvature_drift
                .as_ref()
                .expect("drift recorded")[0],
            0.0
        );
        match certify_rail_face(&reduced) {
            RailFaceVerdict::Certified(proof) => {
                let expected_no_drift = 0.5 * ((k11 - k01 * k01 / k00) - g1 * g1);
                assert!(
                    (proof.tail_constants[0] - expected_no_drift).abs()
                        <= 1.0e-12 * expected_no_drift,
                    "zero-drift c_0={} should drop exactly the rank-2 term ({expected_no_drift})",
                    proof.tail_constants[0]
                );
            }
            other => panic!("the zero-drift fixture must certify, got {other:?}"),
        }
    }

}
