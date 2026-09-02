//! First two moments of the constrained (truncated) Laplace posterior.
//!
//! # What the posterior is when inequality constraints are active
//!
//! A shape- or box-constrained fit restricts the PRIOR SUPPORT to the feasible
//! set `C = {β : Aβ ≥ b}` (a user asserting monotonicity is asserting that the
//! non-monotone coefficient vectors have prior mass zero). The posterior is
//! therefore the penalized likelihood restricted to `C` and renormalized:
//!
//! ```text
//! π(β | y) ∝ exp(−ℓ_p(β)) · 1_C(β)
//! ```
//!
//! Expanding `ℓ_p` at the constrained mode `β̂` — where the KKT conditions give
//! `g := ∇ℓ_p(β̂) = A_actᵀλ` with `λ ≥ 0` and `H := ∇²ℓ_p(β̂)` — and completing
//! the square gives, to Laplace order,
//!
//! ```text
//! π ≈ N(β; β_unc, Σ) truncated to C,   β_unc = β̂ − Σ·∇ℓ_p(β̂),   Σ = φ·H⁻¹
//! ```
//!
//! The Gaussian is centred at the UNCONSTRAINED centre `β_unc`, not at the
//! boundary mode: truncating a Gaussian does not move its pre-truncation mean,
//! so a law centred at the KKT mode is a different (half-normal-shaped)
//! distribution. This is the same law `sample_truncated_gaussian_posterior`
//! draws from, and for the same reason.
//!
//! # The exact decomposition this module computes
//!
//! Let `A` be the retained constraint rows (`q × p`), `W = A Σ Aᵀ` and
//! `G = Σ Aᵀ W⁻¹`, and let `P = I − G A` be the `Σ⁻¹`-orthogonal projector onto
//! `null(A)`. Split the deviation `d = β − β_unc` as `d = P d + G(A d)` and set
//! `t = P d`, `u = A β − b = (A β_unc − b) + A d`. Under the UNTRUNCATED law:
//!
//! * `Cov(t, u) = P Σ Aᵀ = Σ Aᵀ − Σ Aᵀ W⁻¹ (A Σ Aᵀ) = 0`, so `t` and `u` are
//!   independent;
//! * `E[t] = 0` and `Cov(t) = P Σ Pᵀ = Z(Zᵀ Σ⁻¹ Z)⁻¹Zᵀ` for any basis `Z` of
//!   `null(A)`;
//! * `u ~ N(A β_unc − b, W)`.
//!
//! Feasibility is exactly `u ≥ 0`, which constrains ONLY `u`. Because `t` is
//! independent of `u`, conditioning leaves `t` untouched, so the truncated
//! moments are exactly
//!
//! ```text
//! E_π[β]   = β_unc + G·(E[u] − E_untrunc[u])
//! Cov_π[β] = Σ − G·(W − Cov[u])·Gᵀ
//! ```
//!
//! with `(E[u], Cov[u])` the first two moments of the `q`-dimensional Gaussian
//! `N(A β_unc − b, W)` restricted to the orthant `u ≥ 0`.
//!
//! # Why the two obvious answers are both wrong
//!
//! Reading the covariance formula at its two endpoints:
//!
//! * `Cov[u] := W` (no truncation) returns `Σ` — the full unconstrained
//!   covariance. This is the answer that ignores the constraint entirely, and
//!   it over-states the spread along every constrained direction.
//! * `Cov[u] := 0` returns `Σ − G W Gᵀ = P Σ Pᵀ`, the active-face reduction
//!   `Z(ZᵀHZ)⁻¹Zᵀ`, which reports EXACTLY ZERO variance for a fully pinned
//!   coordinate. That is the `λ → ∞` limit: it is correct only for a
//!   constraint whose multiplier is infinite.
//!
//! An EQUALITY / gauge / identifiability constraint genuinely deletes a degree
//! of freedom and zero variance is right for it — but those are absorbed into
//! the basis in this codebase, so the deleted direction is not a coordinate at
//! all. An INEQUALITY does not delete a direction, it halves one: the posterior
//! along the constraint normal is supported on a half-line, its mode sits at
//! the endpoint, and its variance does not. In the scalar case `X ~ N(μ,σ²)`
//! truncated to `[0,∞)` with `α = −μ/σ` and `λ(α) = φ(α)/Φ̄(α)`,
//! `Var(X) = σ²(1 + αλ − λ²)`, which is `0.3634σ²` when the mode sits exactly
//! on the bound and decays as `σ²/α²` only as the multiplier diverges.
//!
//! Since truncating a Gaussian to a convex set can only shrink its covariance,
//! `P Σ Pᵀ ≺ Cov_π[β] ≺ Σ` strictly at every finite multiplier.
//!
//! # No tightness predicate
//!
//! Nothing here classifies a row as "active". A row enters through its
//! standardized slack `s_j = (a_jᵀβ_unc − b_j)/sd_j` and its contribution
//! varies smoothly with it, so a row at slack `1e-9` and a row at slack `0`
//! give nearly the same answer instead of differing by a full `σ²`. The only
//! cut on TIGHTNESS is dropping rows whose truncated mass `Φ̄(s_j)` is below
//! double-precision resolution, a bound read off `f64::EPSILON` rather than
//! tuned.
//!
//! # Which rows the answer is built from
//!
//! There is a second cut, and it is about independence rather than tightness. A
//! shape constraint is normally imposed at every data row — a monotone survival
//! baseline arrives as one derivative-guard row per observed exit time — so the
//! system can carry far more rows than the coefficient block has columns, and
//! `W = A Σ Aᵀ` is then rank-deficient by construction. Its `q × q` inverse is
//! not a quantity double precision has.
//!
//! So the correction is built on a RETAINED FACE. Rows are walked in ascending
//! standardized slack and one joins only while its constraint normal is
//! independent enough of those already accepted for the lift `G = Σ Aᵀ W⁻¹` to
//! be solved to the accuracy the moments are reported to; the assembled face is
//! then checked as a whole, because per-row independence is necessary and not
//! sufficient, and its least independent row is dropped until the check passes.
//!
//! The two halves of that cut cost different amounts, and it is worth being
//! exact about which is which.
//!
//! * The PER-ROW floor is nearly free. A row it refuses lies within `O(θ)` of
//!   the span of the rows already accepted, with `θ` below `5e-7` radians at
//!   the floor, and the accepted row it is nearly parallel to has the SMALLER
//!   standardized slack because the walk is ordered by slack — so its wall is
//!   the binding one and the refused row imposes nothing new. The exception is
//!   a row ANTI-parallel to an accepted one, which carries no new direction
//!   while genuinely halving the support; that one is kept, as the far wall of
//!   a two-sided bound, rather than dropped.
//!
//! * The WHOLE-FACE check is not free, and no `O(θ)` argument covers it. It
//!   fires exactly when every row cleared the floor and the face is still worse
//!   conditioned than any of its rows, and the direction it then drops can sit
//!   at a genuinely resolvable angle — `pivot/diagonal = 1e-3` is `θ ≈ 0.03`
//!   radians, not `5e-7`. What is dropped there is a real constraint, and the
//!   reported truncation is the one carried by the retained face.
//!
//! That trade is deliberate and it is the honest one available. The moments are
//! computed for a BOX in the retained coordinates; a face that kept mutually
//! dependent rows would cut the same region along diagonals, which is a general
//! polyhedron and not something `box_truncated_moments` computes — and the lift
//! that carries the answer back out of those coordinates cannot be formed at
//! all when `W` is numerically singular. The alternatives are therefore a
//! subset-truncated posterior or none, not a subset-truncated posterior or an
//! exact one. Which rows survived is reported (`log::info!`) whenever the
//! whole-face check dropped anything, so the choice is visible on the fit that
//! made it rather than inferable from this file.

use gam_math::probability::{
    normal_cdf, normal_logsf, signed_probit_logcdf_and_mills_ratio, standard_normal_quantile,
    standard_normal_quantile_from_log_cdf,
};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Relative accuracy demanded of the orthant-moment cubature, measured against
/// the PRE-TRUNCATION scale `sd_i = sqrt(W_ii)` so the criterion is invariant
/// to how the constraint rows happen to be scaled.
///
/// The target is set by what the number is FOR. The cubature resolves `Δ`, the
/// variance the truncation removes, and the reported variance is `Σ − GΔGᵀ`
/// with the removed part never exceeding the total. A relative error `ε` in `Δ`
/// therefore moves a reported variance by at most `ε` relative, and a reported
/// standard error by at most `ε/2`. At `1e-3` no reported interval half-width
/// can move in its fourth significant digit — far below the Laplace
/// approximation's own error, and far below any resolution a credible interval
/// carries. Demanding more is not free: the orthant integrand is unbounded at
/// the cube boundary (the second moment grows like `log 1/(1−x)`), so the
/// quasi-Monte-Carlo rate is closer to `N⁻¹` than the `N⁻²` a bounded
/// integrand would give, and every extra digit costs two decades of nodes.
const ORTHANT_MOMENT_RELATIVE_TOLERANCE: f64 = 1e-3;

/// Cubature point count at the first pass. Each subsequent pass EXTENDS the
/// node set to twice its length — the Kronecker sequence is a prefix sequence,
/// so refinement reuses every node already evaluated — until the moments agree
/// to [`ORTHANT_MOMENT_RELATIVE_TOLERANCE`]. This is a starting point rather
/// than a budget.
const ORTHANT_MOMENT_INITIAL_POINTS: usize = 1 << 11;

/// Node count past which the cubature is declared non-convergent and the caller
/// gets an error rather than an unconverged covariance. Silently reporting the
/// last iterate would ship an uncertified number into every interval built from
/// this fit.
const ORTHANT_MOMENT_MAXIMUM_POINTS: usize = 1 << 20;

/// The low-rank correction that turns the unconstrained Laplace covariance into
/// the truncated-posterior covariance.
///
/// Carrying the factored form rather than a dense `p × p` matrix lets a
/// consumer that never materializes `Σ` (the factorized inference path, the
/// prediction backends) apply the same correction with `q` extra solves:
/// `xᵀΣ_π x = xᵀΣx − ‖Δ^{1/2} Gᵀ x‖²`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstrainedPosteriorCorrection {
    /// `G = Σ Aᵀ W⁻¹`, `p × q`.
    pub lift: Array2<f64>,
    /// `Δ = W − Cov[u] ⪰ 0`, `q × q`: the variance the truncation removes from
    /// the constraint-normal coordinates.
    pub removed_normal_variance: Array2<f64>,
    /// `E[u] − E_untrunc[u]`, `q`: how far truncation moves the posterior mean
    /// in constraint-normal coordinates. Positive componentwise for a
    /// half-line coordinate — the posterior mean is interior even when the mode
    /// is not — and of either sign for a coordinate that also carries an upper
    /// limit, where the far face pulls the mean back down.
    pub normal_mean_shift: Array1<f64>,
    /// Indices, into the caller's constraint system, of the rows retained.
    pub rows: Vec<usize>,
    /// Upper limit on each retained coordinate: `u_k ≤ normal_upper_limits[k]`,
    /// with `f64::INFINITY` where the coordinate is a half-line.
    ///
    /// A two-sided coefficient bound `l ≤ β_j ≤ u` arrives as two rows whose
    /// normals are exactly anti-parallel. The second carries no constraint-normal
    /// DIRECTION the first does not already carry, so the rank filter drops it —
    /// correctly, as a direction. It is still a constraint, and this is where it
    /// is kept (#2523).
    ///
    /// `#[serde(default)]` with an empty vector reading as "every retained
    /// coordinate is a half-line" is the encoding of a model saved before upper
    /// limits existed, which is exactly what those models meant. Live
    /// constructions always carry one entry per retained row.
    ///
    /// `+∞` is the COMMON case here — every one-sided shape or box constraint
    /// contributes one — and JSON has no literal for it. Without an explicit
    /// codec `serde_json` writes the entry as `null` and `Vec<f64>` then refuses
    /// its own output on the way back in, which made every shape-constrained fit
    /// that retains a face unloadable (#2601). `serde_extended_real::vec_f64`
    /// makes `null` mean `+∞` in both directions; that is byte-identical to what
    /// was already being written, so models saved before the codec existed read
    /// back with the meaning they always had.
    #[serde(default, with = "gam_problem::serde_extended_real::vec_f64")]
    pub normal_upper_limits: Vec<f64>,
}

impl ConstrainedPosteriorCorrection {
    /// `Σ ← Σ − G Δ Gᵀ`, in place. The correction is rank `q`, so this never
    /// allocates a second `p × p` matrix next to the one being corrected.
    pub fn apply_to_covariance_in_place(&self, covariance: &mut Array2<f64>) {
        let scaled = self.lift.dot(&self.removed_normal_variance);
        let p = covariance.nrows();
        for i in 0..p {
            for j in 0..=i {
                let removed = scaled.row(i).dot(&self.lift.row(j));
                covariance[[i, j]] -= removed;
                if i != j {
                    covariance[[j, i]] = covariance[[i, j]];
                }
            }
        }
    }

    /// `Σ_π = Σ − G Δ Gᵀ`.
    pub fn apply_to_covariance(&self, covariance: &Array2<f64>) -> Array2<f64> {
        let mut corrected = covariance.clone();
        self.apply_to_covariance_in_place(&mut corrected);
        corrected
    }

    /// The same `Σ_π`, assembled as a SUM OF TWO GRAMS instead of as a
    /// subtraction — so its diagonal cannot be a cancellation and cannot come
    /// out negative (#2705 group A).
    ///
    /// `Σ − GΔGᵀ` is the difference of two nearly equal numbers exactly where
    /// the answer matters most. A coordinate the constraint PINS has essentially
    /// all of its variance removed: on `y ~ s(x, shape=convex)` the measured
    /// entry went from `Σ_ii = 2.30e-2` to `6.23e-13` — eleven digits gone — and
    /// on the neighbouring sqrt fixture the same subtraction lands at
    /// `−3.09e-15`, which is not a small variance but a rounding residue with a
    /// sign. Everything downstream (`se_from_covariance`, the dense SE loop)
    /// then has to argue about whether that sign is real.
    ///
    /// Split the correction at `Δ = W − C`, `C = Cov[u] ⪰ 0` the truncated
    /// constraint-normal covariance, and the same quantity is two Grams:
    ///
    /// ```text
    ///     Σ − GΔGᵀ = (Σ − G W Gᵀ) + G C Gᵀ = P Σ Pᵀ + G C Gᵀ,   P = I − G A.
    /// ```
    ///
    /// With `Σ = L Lᵀ` and `C = L_C L_Cᵀ` that is `(P L)(P L)ᵀ + (G L_C)(G L_C)ᵀ`,
    /// and every diagonal entry is a sum of squares. The cancellation does not
    /// disappear — it moves INSIDE `P L`, where each entry carries an absolute
    /// error `O(ε‖L‖)` and is then SQUARED, so a pinned coordinate's variance
    /// picks up `O(p ε² Σ_ii)` instead of `O(ε Σ_ii)`: sixteen orders smaller,
    /// and non-negative by construction rather than by luck.
    ///
    /// `covariance` must be the SPD matrix this correction was built for, and
    /// `constraints` the system its `rows` index. Both are exactly what the
    /// caller already holds where a dense `Σ` exists at all.
    ///
    /// `C` is a cubature result, so it can carry a small negative eigenvalue —
    /// `certify_removed_variance` admits `Δ_ii` up to `slack·W_ii` past `W_ii`.
    /// Eigenvalues inside that certified band are read as the zero they are
    /// approximating; anything below it is refused, because a `C` that is
    /// materially indefinite is a broken moment computation and not a rounding
    /// question.
    pub fn truncated_covariance_psd(
        &self,
        covariance: &Array2<f64>,
        constraints: &LinearInequalityConstraints,
    ) -> Result<Array2<f64>, String> {
        use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};

        let p = covariance.nrows();
        if covariance.ncols() != p {
            return Err(format!(
                "truncated covariance needs a square Σ, got {}x{}",
                covariance.nrows(),
                covariance.ncols()
            ));
        }
        if self.lift.nrows() != p {
            return Err(format!(
                "truncated covariance: the lift has {} rows against a {p}x{p} Σ",
                self.lift.nrows()
            ));
        }
        if constraints.a.ncols() != p {
            return Err(format!(
                "truncated covariance: the constraint system has {} columns against a {p}x{p} Σ",
                constraints.a.ncols()
            ));
        }
        let q = self.rows.len();
        if self.lift.ncols() != q || self.removed_normal_variance.dim() != (q, q) {
            return Err(format!(
                "truncated covariance: {q} retained row(s) against a lift of {} column(s) and a \
                 removed-variance block of {:?}",
                self.lift.ncols(),
                self.removed_normal_variance.dim()
            ));
        }
        let mut retained = Array2::<f64>::zeros((q, p));
        for (position, &row) in self.rows.iter().enumerate() {
            if row >= constraints.a.nrows() {
                return Err(format!(
                    "truncated covariance: retained row {row} is outside the {}-row constraint \
                     system it indexes",
                    constraints.a.nrows()
                ));
            }
            retained.row_mut(position).assign(&constraints.a.row(row));
        }

        // `W = A Σ Aᵀ` on the retained face, recomputed from the very Σ being
        // corrected so `C = W − Δ` cannot mix two different covariances.
        let sigma_at = covariance.dot(&retained.t());
        let mut w = retained.dot(&sigma_at);
        gam_linalg::matrix::symmetrize_in_place(&mut w);
        let mut truncated_normal = &w - &self.removed_normal_variance;
        gam_linalg::matrix::symmetrize_in_place(&mut truncated_normal);

        let (eigenvalues, eigenvectors) = truncated_normal
            .eigh(faer::Side::Lower)
            .map_err(|error| format!("truncated constraint-normal covariance eigendecomposition: {error:?}"))?;
        // The cubature certifies `Δ` to `ORTHANT_MOMENT_RELATIVE_TOLERANCE`
        // relative to the PRE-TRUNCATION scale, so that same band — carried on
        // the largest pre-truncation variance, times the face dimension the
        // certificate is stated over — is the resolution at which an eigenvalue
        // of `C = W − Δ` can be called negative.
        let pre_truncation_scale = (0..q).fold(0.0_f64, |worst, index| worst.max(w[[index, index]]));
        let negative_floor = -ORTHANT_MOMENT_RELATIVE_TOLERANCE * (q as f64) * pre_truncation_scale;
        let mut normal_factor = Array2::<f64>::zeros((q, q));
        for index in 0..q {
            let eigenvalue = eigenvalues[index];
            if !eigenvalue.is_finite() {
                return Err(format!(
                    "truncated constraint-normal covariance has a non-finite eigenvalue at {index}"
                ));
            }
            if eigenvalue < negative_floor {
                return Err(format!(
                    "the truncated constraint-normal covariance is materially indefinite: \
                     eigenvalue {eigenvalue:.6e} at {index} is below the cubature's own \
                     resolution {negative_floor:.6e} (pre-truncation scale \
                     {pre_truncation_scale:.6e} over {q} retained row(s))"
                ));
            }
            let scale = eigenvalue.max(0.0).sqrt();
            for row in 0..q {
                normal_factor[[row, index]] = eigenvectors[[row, index]] * scale;
            }
        }

        let sigma_factor = covariance
            .cholesky(faer::Side::Lower)
            .map_err(|error| {
                format!("truncated covariance requires an SPD Σ to factor: {error:?}")
            })?
            .lower_triangular();
        // `P L = L − G(A L)`, `O(p²q)` rather than the `O(p³)` a materialized
        // `P` would cost.
        let projected_factor = &sigma_factor - &self.lift.dot(&retained.dot(&sigma_factor));
        let normal_lift = self.lift.dot(&normal_factor);

        let mut truncated = projected_factor.dot(&projected_factor.t());
        truncated += &normal_lift.dot(&normal_lift.t());
        gam_linalg::matrix::symmetrize_in_place(&mut truncated);
        // State the diagonal's non-negativity rather than inheriting it from
        // whatever order a GEMM happens to accumulate in — the same reason
        // `smoothing_correction_gram` writes its own diagonal.
        for index in 0..p {
            let projected_row = projected_factor.row(index);
            let normal_row = normal_lift.row(index);
            truncated[[index, index]] =
                projected_row.dot(&projected_row) + normal_row.dot(&normal_row);
        }
        Ok(truncated)
    }

    /// `diag(G Δ Gᵀ)` — the per-coefficient variance the truncation removes,
    /// for consumers that only ever build the covariance diagonal.
    pub fn removed_variance_diagonal(&self) -> Array1<f64> {
        let scaled = self.lift.dot(&self.removed_normal_variance);
        let p = self.lift.nrows();
        let mut diagonal = Array1::<f64>::zeros(p);
        for i in 0..p {
            diagonal[i] = scaled.row(i).dot(&self.lift.row(i));
        }
        diagonal
    }

    /// The absolute per-coefficient uncertainty this correction contributes to
    /// `diag(Σ − GΔGᵀ)`.
    ///
    /// `Δ` is a cubature result, not an exact quantity: it is certified to
    /// `ORTHANT_MOMENT_RELATIVE_TOLERANCE` relative (`certify_removed_variance`),
    /// and `(GΔGᵀ)_ii = g_iᵀ Δ g_i` is monotone in `Δ` in the PSD order, so a
    /// relative error `ε` in `Δ` moves the removed variance by at most
    /// `ε · (GΔGᵀ)_ii`. That product — and NOT the floating-point backward error
    /// of the subtraction — is the resolution at which `diag(Σ − GΔGᵀ)` can be
    /// read.
    ///
    /// This accessor exists so that `ORTHANT_MOMENT_RELATIVE_TOLERANCE` is
    /// declared once and converted into a consumer-facing allowance once. #2705
    /// group A is fits refused because the consumer (`se_from_covariance`,
    /// budget `16·n·eps` ≈ 1e-14 relative) and the producer (this cubature,
    /// budget 1e-3 relative) hold two independent budgets for one number, ~11
    /// orders apart, with nothing carrying the producer's across the boundary.
    pub fn diagonal_uncertainty(&self) -> Array1<f64> {
        self.removed_variance_diagonal() * ORTHANT_MOMENT_RELATIVE_TOLERANCE
    }

    /// `E_π[β] = β_unc + G·(E[u] − E_untrunc[u])`.
    pub fn posterior_mean(&self, unconstrained_center: &Array1<f64>) -> Array1<f64> {
        unconstrained_center + &self.lift.dot(&self.normal_mean_shift)
    }

    /// Upper limit per retained coordinate, materializing the legacy encoding of
    /// [`Self::normal_upper_limits`].
    pub fn upper_limits(&self) -> Vec<f64> {
        if self.normal_upper_limits.is_empty() {
            vec![f64::INFINITY; self.rows.len()]
        } else {
            self.normal_upper_limits.clone()
        }
    }
}

/// Live properness evidence carried by a declined cone-posterior moment route.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConePropernessEvidence {
    Certificate(crate::cone_reduction::ConeProperness),
    CertificationFailed { reason: String },
}

impl ConePropernessEvidence {
    pub fn is_proper(&self) -> Option<bool> {
        match self {
            Self::Certificate(certificate) => certificate.is_proper(),
            Self::CertificationFailed { .. } => None,
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Self::Certificate(certificate) => certificate.summary(),
            Self::CertificationFailed { reason } => format!(
                "cone-truncated posterior properness could not be certified: {reason}"
            ),
        }
    }

    fn validate(&self, ambient_dimension: usize, constraint_count: usize) -> Result<(), String> {
        match self {
            Self::CertificationFailed { reason } => {
                if reason.trim().is_empty() {
                    return Err(
                        "cone properness certification failure has an empty reason".to_string(),
                    );
                }
            }
            Self::Certificate(certificate) => {
                if certificate.reduced.dim() != (constraint_count, constraint_count) {
                    return Err(format!(
                        "cone properness reduced precision has shape {:?}, expected ({constraint_count}, {constraint_count})",
                        certificate.reduced.dim(),
                    ));
                }
                if certificate.reduced.iter().any(|value| !value.is_finite())
                    || certificate
                        .copositive_minimum
                        .is_some_and(|value| !value.is_finite())
                {
                    return Err(
                        "cone properness certificate contains a non-finite value".to_string(),
                    );
                }
                let total = |inertia: crate::cone_reduction::Inertia| {
                    inertia.positive + inertia.zero + inertia.negative
                };
                if total(certificate.ambient_inertia) != ambient_dimension
                    || total(certificate.reduced_inertia) != constraint_count
                    || total(certificate.lineality_inertia)
                        != ambient_dimension.saturating_sub(constraint_count)
                {
                    return Err(format!(
                        "cone properness inertia dimensions disagree with ambient p={ambient_dimension} and face q={constraint_count}"
                    ));
                }
                if certificate.ambient_inertia.positive
                    != certificate.reduced_inertia.positive
                        + certificate.lineality_inertia.positive
                    || certificate.ambient_inertia.zero
                        != certificate.reduced_inertia.zero
                            + certificate.lineality_inertia.zero
                    || certificate.ambient_inertia.negative
                        != certificate.reduced_inertia.negative
                            + certificate.lineality_inertia.negative
                {
                    return Err(
                        "cone properness certificate violates Haynsworth inertia additivity"
                            .to_string(),
                    );
                }
                if certificate.is_proper() == Some(false) {
                    return Err(
                        "a proved-improper cone posterior cannot be stored as a moment decline"
                            .to_string(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Why a converged constrained fit has no reportable posterior moments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConePosteriorMomentDecline {
    pub ambient_precision_failure: String,
    pub properness: ConePropernessEvidence,
}

impl ConePosteriorMomentDecline {
    pub fn summary(&self) -> String {
        format!(
            "ambient covariance route declined ({}); {}",
            self.ambient_precision_failure,
            self.properness.summary(),
        )
    }
}

/// Required wire discriminator for the formerly-overloaded `None` state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstrainedPosteriorMomentStatus {
    Available,
    Declined(ConePosteriorMomentDecline),
}

/// Persisted identity of an inequality-truncated Laplace posterior.
///
/// `mode` is always the feasible optimizer solution. When moments are
/// available, the ambient centre and correction define the posterior-mean
/// estimand. When they are declined, the cone, mode, and properness evidence
/// survive without fabricating either unavailable moment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstrainedPosteriorGeometry {
    /// Exact inequality system `Aβ ≥ b` in the same coefficient frame as the
    /// locations, correction lift, and ambient precision.
    pub constraints: LinearInequalityConstraints,
    /// Feasible optimizer solution; never a substitute for a declined posterior mean.
    pub mode: Array1<f64>,
    unconstrained_center: Option<Array1<f64>>,
    /// Present only when `moment_status` is `Available`; `None` then means the
    /// constraint is invisible at f64 resolution.
    correction: Option<ConstrainedPosteriorCorrection>,
    /// Whether reportable moments exist, with typed evidence when they do not.
    pub moment_status: ConstrainedPosteriorMomentStatus,
}

impl ConstrainedPosteriorGeometry {
    pub fn with_moments(
        constraints: LinearInequalityConstraints,
        mode: Array1<f64>,
        unconstrained_center: Array1<f64>,
        correction: Option<ConstrainedPosteriorCorrection>,
    ) -> Self {
        Self {
            constraints,
            mode,
            unconstrained_center: Some(unconstrained_center),
            correction,
            moment_status: ConstrainedPosteriorMomentStatus::Available,
        }
    }

    pub fn with_decline(
        constraints: LinearInequalityConstraints,
        mode: Array1<f64>,
        decline: ConePosteriorMomentDecline,
    ) -> Self {
        Self {
            constraints,
            mode,
            unconstrained_center: None,
            correction: None,
            moment_status: ConstrainedPosteriorMomentStatus::Declined(decline),
        }
    }

    pub fn decline(&self) -> Option<&ConePosteriorMomentDecline> {
        match &self.moment_status {
            ConstrainedPosteriorMomentStatus::Available => None,
            ConstrainedPosteriorMomentStatus::Declined(decline) => Some(decline),
        }
    }

    pub fn unconstrained_center(&self) -> Result<&Array1<f64>, String> {
        match &self.moment_status {
            ConstrainedPosteriorMomentStatus::Available => self
                .unconstrained_center
                .as_ref()
                .ok_or_else(|| {
                    "available constrained posterior is missing its ambient centre".to_string()
                }),
            ConstrainedPosteriorMomentStatus::Declined(decline) => Err(format!(
                "constrained posterior has no ambient centre because its moments were declined: {}",
                decline.summary(),
            )),
        }
    }

    pub fn correction(&self) -> Result<Option<&ConstrainedPosteriorCorrection>, String> {
        match &self.moment_status {
            ConstrainedPosteriorMomentStatus::Available => Ok(self.correction.as_ref()),
            ConstrainedPosteriorMomentStatus::Declined(decline) => Err(format!(
                "constrained posterior has no moment correction because its moments were declined: {}",
                decline.summary(),
            )),
        }
    }

    pub fn available_parts_mut(
        &mut self,
    ) -> Option<(&mut Array1<f64>, Option<&mut ConstrainedPosteriorCorrection>)> {
        match &self.moment_status {
            ConstrainedPosteriorMomentStatus::Available => Some((
                self.unconstrained_center.as_mut()?,
                self.correction.as_mut(),
            )),
            ConstrainedPosteriorMomentStatus::Declined(_) => None,
        }
    }

    pub fn posterior_mean(&self) -> Result<Array1<f64>, String> {
        let center = self.unconstrained_center()?;
        Ok(self
            .correction()?
            .map(|correction| correction.posterior_mean(center))
            .unwrap_or_else(|| center.clone()))
    }

    pub fn validate_for_dimension(&self, dimension: usize) -> Result<(), String> {
        if self.constraints.a.ncols() != dimension
            || self.constraints.a.nrows() != self.constraints.b.len()
        {
            return Err(format!(
                "constrained posterior inequalities have shape {}x{} with {} bounds, expected {dimension} columns",
                self.constraints.a.nrows(),
                self.constraints.a.ncols(),
                self.constraints.b.len()
            ));
        }
        if self.mode.len() != dimension {
            return Err(format!(
                "constrained posterior mode has length {}, expected {dimension}",
                self.mode.len(),
            ));
        }
        if self
            .mode
            .iter()
            .chain(self.unconstrained_center.iter().flat_map(|center| center.iter()))
            .chain(self.constraints.a.iter())
            .chain(self.constraints.b.iter())
            .any(|value| !value.is_finite())
        {
            return Err("constrained posterior geometry contains a non-finite value".to_string());
        }
        match &self.moment_status {
            ConstrainedPosteriorMomentStatus::Available => {
                if self
                    .unconstrained_center
                    .as_ref()
                    .is_none_or(|center| center.len() != dimension)
                {
                    return Err(format!(
                        "available constrained posterior centre has length {:?}, expected {dimension}",
                        self.unconstrained_center.as_ref().map(Array1::len),
                    ));
                }
            }
            ConstrainedPosteriorMomentStatus::Declined(decline) => {
                if self.unconstrained_center.is_some() || self.correction.is_some() {
                    return Err(
                        "declined constrained posterior must not carry fabricated ambient moments"
                            .to_string(),
                    );
                }
                if decline.ambient_precision_failure.trim().is_empty() {
                    return Err(
                        "constrained posterior moment decline has an empty ambient-precision reason"
                            .to_string(),
                    );
                }
                decline.properness.validate(dimension, self.constraints.a.nrows())?;
            }
        }
        if let Some(correction) = self.correction.as_ref() {
            let q = correction.lift.ncols();
            if correction.lift.nrows() != dimension {
                return Err(format!(
                    "constrained posterior lift has {} rows, expected {dimension}",
                    correction.lift.nrows()
                ));
            }
            if correction.removed_normal_variance.dim() != (q, q)
                || correction.normal_mean_shift.len() != q
                || correction.rows.len() != q
            {
                return Err(format!(
                    "constrained posterior normal geometry is inconsistent: lift={}x{q}, removed={:?}, mean={}, rows={}",
                    correction.lift.nrows(),
                    correction.removed_normal_variance.dim(),
                    correction.normal_mean_shift.len(),
                    correction.rows.len()
                ));
            }
            let mut unique_rows = correction.rows.clone();
            unique_rows.sort_unstable();
            unique_rows.dedup();
            if unique_rows.len() != q
                || unique_rows
                    .iter()
                    .any(|&row| row >= self.constraints.a.nrows())
            {
                return Err(format!(
                    "constrained posterior retained rows {:?} are not unique valid indices for {} inequalities",
                    correction.rows,
                    self.constraints.a.nrows()
                ));
            }
            if correction
                .lift
                .iter()
                .chain(correction.removed_normal_variance.iter())
                .chain(correction.normal_mean_shift.iter())
                .any(|value| !value.is_finite())
            {
                return Err(
                    "constrained posterior correction contains a non-finite value".to_string()
                );
            }
            if !correction.normal_upper_limits.is_empty()
                && correction.normal_upper_limits.len() != q
            {
                return Err(format!(
                    "constrained posterior carries {} upper limits for {q} retained rows",
                    correction.normal_upper_limits.len()
                ));
            }
            // `+∞` is the half-line coordinate and is admissible; anything at or
            // below the wall would make the retained region empty.
            if correction
                .normal_upper_limits
                .iter()
                .any(|limit| !(*limit > 0.0))
            {
                return Err(format!(
                    "constrained posterior upper limits must be positive, got {:?}",
                    correction.normal_upper_limits
                ));
            }
        }
        Ok(())
    }
}

/// Equal-tailed interval for one linear projection of an inequality-truncated
/// Gaussian posterior.
///
/// `ambient_covariance` is the pre-truncation covariance `Σ` in the active
/// coefficient frame and `contrast` defines the scalar `cᵀβ`.  The affine
/// shift of a saved coefficient gauge is deliberately not accepted here:
/// callers add that deterministic shift to both returned endpoints.
///
/// The decomposition in this module makes the projection
///
/// ```text
/// cᵀβ = cᵀβ_unc + cᵀt + (Gᵀc)ᵀ(u - E_untrunc[u]),
/// ```
///
/// where `cᵀt` is an independent scalar Gaussian and `u` is the retained
/// orthant-truncated Gaussian.  The interval therefore comes from the quantiles
/// of that convolution, not from `posterior_mean ± z·posterior_sd`.
/// The decomposition every scalar-projection consumer in this module needs, in
/// one place.
///
/// Two consumers read it — the equal-tailed interval and
/// [`constrained_projection_law`] — and it is exactly the kind of derivation
/// that is individually reasonable and quietly different when written twice.
/// That is the failure genus this sweep is about (#2385), so it is written once.
struct TruncatedProjection {
    /// `E_π[cᵀβ]`.
    posterior_mean: f64,
    /// Retained constraint-normal coordinates `u`: centre, covariance, walls.
    normal_center: Array1<f64>,
    normal_covariance: Array2<f64>,
    upper_limits: Vec<f64>,
    /// `Gᵀc`: how a displacement of `u` moves the projection.
    projection_lift: Array1<f64>,
    /// `Var(cᵀt)`, the tangent component that stays Gaussian and independent of
    /// `u`. Exactly zero when `c` is carried entirely by the retained
    /// constraint normals, which is the case a product cone on one block gives.
    residual_variance: f64,
}

struct ProjectionDecomposition {
    ambient_mean: f64,
    ambient_variance: f64,
    /// `None` exactly when the geometry carries no correction: the truncation
    /// is invisible at f64 resolution and the projection is the ambient normal.
    truncated: Option<TruncatedProjection>,
}

fn decompose_projection(
    ambient_covariance: &Array2<f64>,
    geometry: &ConstrainedPosteriorGeometry,
    contrast: &Array1<f64>,
) -> Result<ProjectionDecomposition, String> {
    let p = contrast.len();
    geometry.validate_for_dimension(p)?;
    if ambient_covariance.dim() != (p, p) {
        return Err(format!(
            "constrained projection needs a {p}x{p} ambient covariance, got {:?}",
            ambient_covariance.dim()
        ));
    }
    if ambient_covariance.iter().any(|value| !value.is_finite())
        || contrast.iter().any(|value| !value.is_finite())
    {
        return Err(
            "constrained projection received a non-finite covariance or contrast".to_string(),
        );
    }

    let ambient_mean = contrast.dot(geometry.unconstrained_center()?);
    let sigma_c = ambient_covariance.dot(contrast);
    let ambient_variance = contrast.dot(&sigma_c);
    let covariance_scale = ambient_covariance
        .diag()
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let contrast_scale = contrast.dot(contrast).max(f64::MIN_POSITIVE);
    let variance_floor = (p.max(1) as f64) * f64::EPSILON * covariance_scale * contrast_scale;
    if ambient_variance < -variance_floor || !ambient_variance.is_finite() {
        return Err(format!(
            "constrained projection has invalid ambient variance {ambient_variance:.6e}"
        ));
    }
    let ambient_variance = ambient_variance.max(0.0);

    let Some(correction) = geometry.correction()? else {
        return Ok(ProjectionDecomposition {
            ambient_mean,
            ambient_variance,
            truncated: None,
        });
    };

    let q = correction.rows.len();
    let mut normal_center = Array1::<f64>::zeros(q);
    let mut normal_covariance = Array2::<f64>::zeros((q, q));
    let mut sigma_a = Array2::<f64>::zeros((p, q));
    for (position, &row) in correction.rows.iter().enumerate() {
        let a = geometry.constraints.a.row(row);
        normal_center[position] =
            a.dot(geometry.unconstrained_center()?) - geometry.constraints.b[row];
        sigma_a
            .column_mut(position)
            .assign(&ambient_covariance.dot(&a));
    }
    for i in 0..q {
        let ai = geometry.constraints.a.row(correction.rows[i]);
        for j in 0..=i {
            let value = ai.dot(&sigma_a.column(j));
            normal_covariance[[i, j]] = value;
            normal_covariance[[j, i]] = value;
        }
    }

    let projection_lift = correction.lift.t().dot(contrast);
    let normal_component_variance = projection_lift.dot(&normal_covariance.dot(&projection_lift));
    let residual_variance = ambient_variance - normal_component_variance;
    let residual_floor = (p.max(q).max(1) as f64)
        * f64::EPSILON
        * ambient_variance
            .max(normal_component_variance)
            .max(f64::MIN_POSITIVE);
    if residual_variance < -residual_floor || !residual_variance.is_finite() {
        return Err(format!(
            "constrained projection decomposition produced residual variance \
             {residual_variance:.6e} from ambient {ambient_variance:.6e}"
        ));
    }
    let residual_variance = residual_variance.max(0.0);
    let posterior_mean = ambient_mean + projection_lift.dot(&correction.normal_mean_shift);
    let upper_limits = correction.upper_limits();
    if upper_limits.len() != q {
        return Err(format!(
            "constrained projection: {q} retained rows carry {} upper limits",
            upper_limits.len()
        ));
    }
    Ok(ProjectionDecomposition {
        ambient_mean,
        ambient_variance,
        truncated: Some(TruncatedProjection {
            posterior_mean,
            normal_center,
            normal_covariance,
            upper_limits,
            projection_lift,
            residual_variance,
        }),
    })
}

/// The LAW of one scalar projection `cᵀβ` of an inequality-truncated Gaussian
/// posterior — not two of its quantiles, and not a normal fitted to its first
/// two moments.
///
/// This is what the module's own decomposition produces:
///
/// ```text
/// cᵀβ = cᵀβ_unc + (Gᵀc)ᵀ(u - E_untrunc[u]) + cᵀt,
/// ```
///
/// a discrete mixture over the retained orthant's cubature nodes convolved with
/// one independent Gaussian. Two properties a normal cannot have, and both are
/// why this exists (#2446):
///
/// * **Every node is feasible.** The nodes are points of the retained orthant,
///   so the law puts no mass on coefficient vectors the fit excluded. The
///   normal with the same first two moments does — measurably, a few percent of
///   its mass — because the pushforward of a cone-truncated joint through `cᵀ`
///   is not a normal for `q > 1`.
/// * **Its error is a RATE, not a floor.** Matching two moments is exact for a
///   normal and wrong by a fixed amount for this law, so no extra work reduces
///   it. The node sum converges with the cubature.
///
/// A consumer that integrates a SMOOTH functional barely notices the first
/// property. One that integrates an indicator — any quantile, any exceedance
/// probability — reads the location of mass at first order, and for it the
/// moment-matched normal is not admissible at all.
pub struct ConstrainedProjectionLaw {
    /// `(location, weight)` per cubature node, weights summing to one. A
    /// geometry with no correction is one node of weight one at the ambient
    /// mean.
    pub nodes: Vec<(f64, f64)>,
    /// Variance of the independent Gaussian each node is convolved with. Zero
    /// when the contrast is carried entirely by the retained constraint
    /// normals, and then the mixture IS the whole law.
    pub residual_variance: f64,
}

impl ConstrainedProjectionLaw {
    /// `E[cᵀβ]` under this law.
    pub fn mean(&self) -> f64 {
        self.nodes
            .iter()
            .map(|(location, weight)| location * weight)
            .sum()
    }

    /// `Var(cᵀβ)` under this law: the mixture's own spread plus the tangent.
    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        let spread = self
            .nodes
            .iter()
            .map(|(location, weight)| weight * (location - mean) * (location - mean))
            .sum::<f64>();
        spread + self.residual_variance
    }
}

/// One point of the JOINT rule over an inequality-truncated Gaussian posterior:
/// a feasible constraint-normal coordinate and the tangent coordinates drawn
/// from the same low-discrepancy point.
#[derive(Clone, Debug)]
pub struct ConstrainedPosteriorJointPoint {
    /// `u = Aβ − b` on the retained rows. Inside the retained region by
    /// construction — the separation-of-variables map only ever produces points
    /// of `[0, upper]`, so a consumer integrating against these points puts
    /// exactly zero mass on coefficient vectors the fit excluded.
    pub normal_coordinates: Array1<f64>,
    /// Independent standard-normal coordinates for the tangent block, length
    /// `tangent_dimension`. The tangent of an inequality-truncated Gaussian is
    /// exactly Gaussian and exactly independent of `u`: conditioning on `Aβ`
    /// leaves `N(β_unc + G(u − E_untrunc[u]), Σ − GAΣ)` whatever the truncation
    /// does to `u`, which is what lets one rule carry both blocks.
    pub tangent: Array1<f64>,
    /// Normalized weight. The weights sum to one.
    pub weight: f64,
}

/// A single low-discrepancy rule over the constraint-normal AND tangent
/// coordinates of an inequality-truncated Gaussian.
///
/// Why this exists (#2679). A consumer that integrates a nonlinear functional
/// of `β` over this posterior has, until now, had two options, and both are
/// wrong for a different reason:
///
/// * Integrate the moment-matched NORMAL. Its error is a FLOOR — the
///   pushforward of a cone-truncated joint is not normal for `q > 1`, so
///   matching two moments cannot be improved by any amount of extra work — and
///   it puts a measurable fraction of its mass outside the cone.
/// * Nest a Gaussian tensor rule inside every node of the truncated cubature.
///   That is exact, and it costs `nodes × outer × inner` evaluations per
///   evaluation point, which is not a production integration rule.
///
/// This is the third option: `points` points of ONE rule in dimension
/// `q + tangent_dimension`. The first `q` lattice coordinates run the same
/// separation-of-variables map, tilt and weighting the module's own cubature
/// uses — so the `u`-marginal is bit-identical to it at equal point counts —
/// and the remaining coordinates carry standard normals for the tangent. The
/// per-evaluation-point cost is `points`, with no factor of the cubature's node
/// count in it.
///
/// The price is real and is the caller's to gate: a lattice rule on the smooth
/// tangent block does not have the spectral accuracy of the Gauss-Hermite
/// tensor rule it replaces. A consumer must measure itself against a reference
/// built from the DENSITY rather than from this rule before using it.
pub fn constrained_posterior_joint_cubature(
    normal_center: &Array1<f64>,
    normal_covariance: &Array2<f64>,
    upper_limits: &[f64],
    tangent_dimension: usize,
    points: usize,
) -> Result<Vec<ConstrainedPosteriorJointPoint>, String> {
    let q = normal_center.len();
    if q == 0 {
        return Err("joint constrained cubature needs at least one constraint normal".to_string());
    }
    if normal_covariance.dim() != (q, q) || upper_limits.len() != q {
        return Err(format!(
            "joint constrained cubature geometry mismatch: centre={q}, covariance={:?}, \
             upper limits={}",
            normal_covariance.dim(),
            upper_limits.len()
        ));
    }
    if points == 0 {
        return Err("joint constrained cubature needs a positive point count".to_string());
    }
    if upper_limits.iter().any(|limit| !(*limit > 0.0)) {
        return Err(format!(
            "joint constrained cubature: every upper limit must sit strictly above its wall, \
             got {upper_limits:?}"
        ));
    }
    let factor = gam_linalg::triangular::cholesky_factor_in_place(
        normal_covariance.view(),
        gam_linalg::triangular::CholeskyGuard::FiniteStrict,
    )
    .ok_or_else(|| {
        "joint constrained cubature: the constraint-normal covariance is not numerically \
         positive definite"
            .to_string()
    })?;
    let generator = kronecker_generator(q + tangent_dimension);
    let tilt = minimax_tilt(normal_center, upper_limits, factor.view());
    let mut accumulator = JointCubatureAccumulator {
        points: Vec::with_capacity(points),
    };
    accumulate_orthant_nodes(
        &mut accumulator,
        normal_center,
        upper_limits,
        None,
        factor.view(),
        &generator,
        tilt.as_ref(),
        tangent_dimension,
        0,
        points,
    )?;
    accumulator.normalized()
}

/// Sink that keeps whole joint points rather than accumulating their moments.
struct JointCubatureAccumulator {
    /// `weight` carries the UNNORMALIZED log weight until [`Self::normalized`]
    /// rescales it: the log scale spans hundreds of decades on a deeply pinned
    /// face, so no weight is exponentiated before the maximum is known.
    points: Vec<ConstrainedPosteriorJointPoint>,
}

impl JointCubatureAccumulator {
    fn normalized(self) -> Result<Vec<ConstrainedPosteriorJointPoint>, String> {
        let max_log_weight = self
            .points
            .iter()
            .map(|point| point.weight)
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_log_weight.is_finite() {
            return Err("joint constrained cubature accumulated no finite node weight".to_string());
        }
        let weight_sum = self
            .points
            .iter()
            .map(|point| (point.weight - max_log_weight).exp())
            .sum::<f64>();
        if !(weight_sum.is_finite() && weight_sum > 0.0) {
            return Err(format!(
                "joint constrained cubature has invalid normalized weight sum {weight_sum:?}"
            ));
        }
        let mut points = self.points;
        for point in points.iter_mut() {
            point.weight = (point.weight - max_log_weight).exp() / weight_sum;
        }
        Ok(points)
    }
}

impl OrthantNodeSink for JointCubatureAccumulator {
    fn push(&mut self, log_weight: f64, point: &Array1<f64>) {
        self.push_joint(log_weight, point, &[]);
    }

    fn push_joint(&mut self, log_weight: f64, point: &Array1<f64>, tangent: &[f64]) {
        self.points.push(ConstrainedPosteriorJointPoint {
            normal_coordinates: point.clone(),
            tangent: Array1::from_vec(tangent.to_vec()),
            weight: log_weight,
        });
    }
}

/// Equal-tailed interval for one linear projection of an inequality-truncated
/// Gaussian posterior.
///
/// `ambient_covariance` is the pre-truncation covariance `Σ` in the active
/// coefficient frame and `contrast` defines the scalar `cᵀβ`.  The affine
/// shift of a saved coefficient gauge is deliberately not accepted here:
/// callers add that deterministic shift to both returned endpoints.
///
/// The interval comes from the quantiles of the convolution
/// `decompose_projection` produces, not from `posterior_mean ± z·posterior_sd`.
pub fn constrained_projection_equal_tailed_interval(
    ambient_covariance: &Array2<f64>,
    geometry: &ConstrainedPosteriorGeometry,
    contrast: &Array1<f64>,
    level: f64,
) -> Result<(f64, f64), String> {
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(format!(
            "constrained projection interval level must lie in (0, 1), got {level}"
        ));
    }
    let decomposition = decompose_projection(ambient_covariance, geometry, contrast)?;
    let ambient_mean = decomposition.ambient_mean;
    let ambient_variance = decomposition.ambient_variance;
    let alpha = 0.5 * (1.0 - level);

    let Some(truncated) = decomposition.truncated else {
        let sd = ambient_variance.sqrt();
        if sd == 0.0 {
            return Ok((ambient_mean, ambient_mean));
        }
        let z = standard_normal_quantile(1.0 - alpha)
            .map_err(|error| format!("constrained projection normal quantile: {error}"))?;
        return Ok((ambient_mean - z * sd, ambient_mean + z * sd));
    };

    let TruncatedProjection {
        posterior_mean,
        normal_center,
        normal_covariance,
        upper_limits,
        projection_lift,
        residual_variance,
    } = truncated;
    let q = normal_center.len();
    if q == 1 && residual_variance == 0.0 && projection_lift[0] != 0.0 {
        let scalar_quantile = |probability: f64| -> Result<f64, String> {
            let normal_probability = if projection_lift[0] > 0.0 {
                probability
            } else {
                1.0 - probability
            };
            let value = scalar_truncated_quantile(
                normal_center[0],
                normal_covariance[[0, 0]],
                upper_limits[0],
                normal_probability,
            )?;
            Ok(ambient_mean + projection_lift[0] * (value - normal_center[0]))
        };
        return Ok((scalar_quantile(alpha)?, scalar_quantile(1.0 - alpha)?));
    }
    let nodes = converged_projection_nodes(
        &normal_center,
        &normal_covariance,
        &upper_limits,
        &projection_lift,
        ambient_mean,
    )?;
    let lower = projection_quantile(
        &nodes,
        residual_variance,
        alpha,
        posterior_mean,
        ambient_variance.sqrt(),
    )?;
    let upper = projection_quantile(
        &nodes,
        residual_variance,
        1.0 - alpha,
        posterior_mean,
        ambient_variance.sqrt(),
    )?;
    Ok((lower, upper))
}

/// Quantile of `N(mean, variance)` restricted to `[0, upper]`.
fn scalar_truncated_quantile(
    mean: f64,
    variance: f64,
    upper: f64,
    probability: f64,
) -> Result<f64, String> {
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!(
            "scalar truncated quantile needs positive finite variance, got {variance:?}"
        ));
    }
    if !(probability.is_finite() && probability > 0.0 && probability < 1.0) {
        return Err(format!(
            "scalar truncated quantile probability must lie in (0, 1), got {probability}"
        ));
    }
    if !(upper > 0.0) {
        return Err(format!(
            "scalar truncated quantile needs the upper limit above the wall, got {upper:?}"
        ));
    }
    let sd = variance.sqrt();
    let alpha = -mean / sd;
    if !upper.is_finite() {
        // P(Z > z | Z >= alpha) = (1-p) P(Z >= alpha). Work entirely in
        // log-survival space so a deeply pinned face never forms `1-Phi(alpha)`.
        let log_tail = (1.0 - probability).ln() + normal_logsf(alpha);
        let z = -standard_normal_quantile_from_log_cdf(log_tail)
            .map_err(|error| format!("scalar truncated quantile: {error}"))?;
        return Ok(mean + sd * z);
    }
    let beta = (upper - mean) / sd;
    // Same reflection as the moments: put the retained slab in the upper tail so
    // its mass is a difference of directly-evaluated tail probabilities. Under
    // the reflection the probability runs the other way.
    let reflect = alpha + beta < 0.0;
    let (low, high, probability) = if reflect {
        (-beta, -alpha, 1.0 - probability)
    } else {
        (alpha, beta, probability)
    };
    let log_tail_low = normal_logsf(low);
    let removed = normal_logsf(high) - log_tail_low;
    // `Φ̄(z) = Φ̄(low)·(1 − p(1 − e^removed))`, the inversion the cubature uses.
    let log_tail = log_tail_low + (-probability * -removed.exp_m1()).ln_1p();
    let z = -standard_normal_quantile_from_log_cdf(log_tail)
        .map_err(|error| format!("scalar truncated quantile: {error}"))?;
    let z = z.clamp(low, high);
    // Reflected, `z` standardizes `−X` about `−mean`, so `X = mean − sd·z`.
    Ok(if reflect {
        mean - sd * z
    } else {
        mean + sd * z
    })
}

/// Build the truncated-posterior correction for a fit carrying linear
/// inequality constraints, or `None` when no constraint row is close enough to
/// the posterior centre to move the answer at double precision.
///
/// * `covariance` — `Σ`, the PRE-TRUNCATION posterior covariance on the same
///   coefficient frame as `constraints` and `unconstrained_center`. This is the
///   dispersion-scaled `φ·H⁻¹`: truncation is a statement about the posterior's
///   own spread, so it must be applied in the scaled metric, not to `H⁻¹`.
/// * `unconstrained_center` — `β_unc = β̂ − Σ·∇ℓ_p(β̂)`.
/// * `constraints` — `A β ≥ b`.
///
/// `None` is returned when every row's standardized slack exceeds the
/// resolution horizon, which includes the case of a fit whose constraints are
/// all inactive. Callers must then report `Σ` unchanged, bit for bit.
pub fn constrained_posterior_correction_from_covariance(
    covariance: &Array2<f64>,
    unconstrained_center: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Result<Option<ConstrainedPosteriorCorrection>, String> {
    let p = covariance.nrows();
    if covariance.ncols() != p {
        return Err(format!(
            "constrained posterior correction needs a square covariance, got {}x{}",
            covariance.nrows(),
            covariance.ncols()
        ));
    }
    if constraints.a.ncols() != p {
        return Err(format!(
            "constrained posterior correction: covariance is {p}x{p} but the constraint \
             system has {} columns",
            constraints.a.ncols()
        ));
    }
    let sigma_times_at = covariance.dot(&constraints.a.t());
    constrained_posterior_correction(sigma_times_at.view(), unconstrained_center, constraints)
}

/// Same correction for a caller that never materializes `Σ`.
///
/// Everything the decomposition needs from the covariance is the `p × m` block
/// `Σ Aᵀ` — column `j` is `Σ a_j`, `W_ij = a_iᵀ(Σ a_j)`, and the lift is
/// `(Σ Aᵀ)W⁻¹` — so a factorized inference path supplies `m` solves instead of
/// a `p × p` inverse.
pub fn constrained_posterior_correction(
    sigma_times_constraint_transpose: ArrayView2<'_, f64>,
    unconstrained_center: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Result<Option<ConstrainedPosteriorCorrection>, String> {
    let p = sigma_times_constraint_transpose.nrows();
    if sigma_times_constraint_transpose.ncols() != constraints.a.nrows() {
        return Err(format!(
            "constrained posterior correction: the constraint system has {} rows but \
             Sigma·Aᵀ has {} columns",
            constraints.a.nrows(),
            sigma_times_constraint_transpose.ncols()
        ));
    }
    if unconstrained_center.len() != p {
        return Err(format!(
            "constrained posterior correction: Sigma·Aᵀ has {p} rows but the centre has \
             length {}",
            unconstrained_center.len()
        ));
    }
    if constraints.a.ncols() != p {
        return Err(format!(
            "constrained posterior correction: Sigma·Aᵀ has {p} rows but the constraint \
             system has {} columns",
            constraints.a.ncols()
        ));
    }

    let candidates = constraint_face_candidates(
        sigma_times_constraint_transpose,
        unconstrained_center,
        constraints,
    )?;
    if candidates.is_empty() {
        return Ok(None);
    }

    // The retention floor is the accuracy the retained face must deliver, and
    // it asks for exactly the accuracy this module reports its moments to. It
    // is necessary and not sufficient (see [`assemble_retained_face`]), so when
    // the assembled face's lift misses the identity that defines it, a row has
    // to come out and the face has to be rebuilt.
    //
    // THE LADDER IS INDEXED BY THE FACE, NOT BY A FLOOR (#2714). Two rules have
    // been tried here and both failed for the same underlying reason — the
    // quantity being searched over was a real number that only stands in for
    // the face:
    //
    // 1. `demanded_accuracy /= departure/tolerance` read its step size off the
    //    PER-ROW error model `departure ≈ ε·diagonal/pivot`, which holds only
    //    when the face is no worse conditioned than its worst row — i.e. never,
    //    in the only case the ladder runs in. One pass could carry the floor
    //    from `1e-3` past every intermediate face to below `f64::EPSILON`.
    //
    // 2. Stepping to `max_r d_r`, with `d_r = (k+1)·ε·diagonal_r/pivot_r` the
    //    floor at which accepted row `r` drops, is exact in real arithmetic —
    //    the test becomes `pivot > pivot` — and is NOT exact in floating point.
    //    `d_r` is a rounded quotient and the retention test recomputes
    //    `(k+1)·ε·diagonal_r/d_r`, a second rounded quotient; that round trip
    //    lands strictly below `pivot_r` for about 5% of `(k, diagonal, pivot)`
    //    triples, so the row the step was aimed at is retained, the rebuilt
    //    face is bit-identical, `max_r d_r` recomputes to the value it already
    //    has, and the walk stops descending. Inverting a floating-point
    //    comparison to name the next face cannot be made reliable by rounding
    //    the quotient the other way either: the retention test is the
    //    definition of the face, and only the test can decide it.
    //
    // So the walk carries the face itself. A rejected face names the accepted
    // row whose `pivot/diagonal` is smallest — the most nearly dependent one,
    // which is both the row rule 2 was trying to reach and the row that
    // contributes least constraint information — and that row is EXCLUDED by
    // index, together with any opposite face folded into it, before the face is
    // rebuilt at the unchanged floor. What leaves is a DIRECTION, not a row:
    // promoting an excluded row's anti-parallel partner in its place would
    // report the slacker of two walls as a one-sided bound.
    //
    // Termination is now structural rather than numerical. The excluded set
    // gains at least one row per pass, never re-adds one (an excluded row is
    // skipped before it can be accepted or folded), and is a subset of the
    // candidates — so there are at most `candidates.len()` passes; the first
    // unexcluded row always clears the floor (`accepted = 0` makes
    // `pivot = diagonal` and the floor `ε·diagonal/1e-3`), so a nonempty
    // unexcluded set always yields a nonempty face; and the last such face is a
    // single row, whose `1×1` lift is exact and whose departure gate cannot
    // fail. No float comparison is inverted anywhere on that argument.
    //
    // Excluding at the UNCHANGED floor is also strictly less lossy than rule 2,
    // which tightened the floor for every surviving row as a side effect of
    // dropping one. Every face rule 2 could reach is still reachable here — run
    // the walk excluding precisely the rows that floor rejected, and each
    // surviving row clears the looser floor a fortiori — while faces that only
    // a tighter floor would have destroyed are kept.
    let demanded_accuracy = ORTHANT_MOMENT_RELATIVE_TOLERANCE;
    let mut first_pass = true;
    let mut faces_tried = 0usize;
    let mut ladder: Vec<LadderRung> = Vec::new();
    let mut excluded: Vec<usize> = Vec::new();
    let mut last_refused: Option<RefusedFace> = None;
    while excluded.len() <= candidates.len() {
        let Some(face) = assemble_retained_face(
            &candidates,
            demanded_accuracy,
            constraints,
            unconstrained_center,
            &excluded,
        )?
        else {
            if first_pass {
                return Ok(None);
            }
            report_terminal_refusal(last_refused.as_ref(), &ladder, excluded.len());
            return Err(format!(
                "no constraint face survives the accuracy its own lift must deliver: excluding \
                 {} of {} candidate row(s) at the retention floor {demanded_accuracy:.3e} left \
                 no retained row after {faces_tried} face(s){}",
                excluded.len(),
                candidates.len(),
                render_ladder(&ladder, candidates.len())
            ));
        };
        first_pass = false;
        faces_tried += 1;
        // `G = Σ Aᵀ W⁻¹` solved through the factor built above, one column of
        // `Gᵀ` at a time: `W Gᵀ_col = (Σ Aᵀ)ᵀ_col`.
        let lift = cholesky_solve_right(&face.factor, &face.sigma_at)?;
        let departure = lift_identity_departure(&lift, constraints, &face.rows)?;
        ladder.push(LadderRung {
            excluded: excluded.len(),
            retained: face.rows.len(),
            departure,
        });
        if departure > ORTHANT_MOMENT_RELATIVE_TOLERANCE {
            last_refused = Some(RefusedFace {
                rows: face.rows.clone(),
                w: face.w.clone(),
            });
            if face.rows.len() == 1 {
                report_terminal_refusal(last_refused.as_ref(), &ladder, excluded.len());
                return Err(format!(
                    "a single retained constraint row still misses the identity that defines \
                     its lift: max|A G - I| = {departure:.6e} exceeds \
                     {ORTHANT_MOMENT_RELATIVE_TOLERANCE:.1e}, which one row cannot be \
                     ill-conditioned enough to cause{}",
                    render_ladder(&ladder, candidates.len())
                ));
            }
            // Drop the most nearly dependent accepted DIRECTION — that row and
            // any opposite face folded into it — and rebuild. A face with two
            // or more rows always names one, so the empty case is a statement
            // about `assemble_retained_face`, not a fallback.
            if face.least_independent_direction.is_empty() {
                report_terminal_refusal(last_refused.as_ref(), &ladder, excluded.len());
                return Err(format!(
                    "the constraint face misses the identity that defines its lift \
                     (max|A G - I| = {departure:.6e} against \
                     {ORTHANT_MOMENT_RELATIVE_TOLERANCE:.1e}) over {} retained row(s), and \
                     names no least-independent direction to drop, after {faces_tried} \
                     face(s){}",
                    face.rows.len(),
                    render_ladder(&ladder, candidates.len())
                ));
            }
            excluded.extend_from_slice(&face.least_independent_direction);
            continue;
        }

        let q = face.rows.len();
        let mut normal_center = Array1::<f64>::zeros(q);
        for (position, &row_index) in face.rows.iter().enumerate() {
            normal_center[position] =
                constraints.a.row(row_index).dot(unconstrained_center) - constraints.b[row_index];
        }

        let (normal_mean, normal_covariance) = box_truncated_moments(
            &normal_center,
            &face.upper,
            &face.w,
            face.factor.view(),
        )?;

        let mut removed = &face.w - &normal_covariance;
        gam_linalg::matrix::symmetrize_in_place(&mut removed);
        certify_removed_variance(&removed, &face.w)?;

        if !excluded.is_empty() {
            // The walk ran, so the reported truncation is carried by a SUBSET
            // of the rows the user asked for. That is a fact about the answer,
            // not about the solve — the dropped rows are within `O(θ)` of the
            // span of the retained ones, but "within `O(θ)`" is a claim the
            // reader is entitled to see stated on their own fit.
            log::info!(
                "[CONSTRAINED-FACE] {} of {} candidate constraint row(s) retained after \
                 dropping {} nearly dependent direction(s) over {faces_tried} face(s); the \
                 retained lift satisfies its identity to {departure:.3e}",
                face.rows.len(),
                candidates.len(),
                excluded.len()
            );
        }

        return Ok(Some(ConstrainedPosteriorCorrection {
            lift,
            removed_normal_variance: removed,
            normal_mean_shift: normal_mean - normal_center,
            rows: face.rows,
            normal_upper_limits: face.upper,
        }));
    }
    report_terminal_refusal(last_refused.as_ref(), &ladder, excluded.len());
    Err(format!(
        "the constraint-normal lift never reached the accuracy it is certified to: \
         {faces_tried} constraint face(s) were tried and every candidate row was excluded. \
         The walk drops one accepted direction per pass and a single-row face's lift is \
         exact, so this says the single-row face was never reached — which it must be.{}",
        render_ladder(&ladder, candidates.len())
    ))
}

/// Print the evidence a terminal refusal of the retention walk rests on: the
/// last face it refused, at full precision, and the trail that got there.
fn report_terminal_refusal(
    last_refused: Option<&RefusedFace>,
    ladder: &[LadderRung],
    excluded: usize,
) {
    let Some(face) = last_refused else {
        return;
    };
    let departure = ladder.last().map_or(f64::NAN, |rung| rung.departure);
    log_refused_face(face, departure, excluded);
}

/// One rung of the retention walk: how many rows had been excluded when it ran,
/// the face that produced, and how far that face's lift missed the identity
/// that defines it.
///
/// Recorded because every terminal refusal in this walk is a statement about a
/// TRAJECTORY — "the walk never reached a face whose lift is accurate" — and a
/// message that reports only its last rung cannot be told apart from one that
/// took a single wrong step. #2714's witness reached the terminal message after
/// exactly one rung under the pre-`92332c45c` rule, and the message said
/// nothing that would have distinguished that from forty.
struct LadderRung {
    excluded: usize,
    retained: usize,
    departure: f64,
}

/// Render the ladder trail for a terminal message, newest rung last.
///
/// The walk can take one pass per candidate row, and a message carrying two
/// hundred rungs is one nobody reads, so the ends are printed and the middle is
/// counted. The ends are what carry the diagnosis: the first rung is the face
/// the retention floor produced on its own and the last is the one the walk
/// died on. The middle is not summarizable beyond its count — the RETAINED
/// count is not monotone, because excluding a nearly dependent row raises the
/// pivots of the rows after it and can admit ones the floor had refused — but
/// the `excluded` column is, and it is the walk's actual index.
fn render_ladder(ladder: &[LadderRung], candidates: usize) -> String {
    const SHOWN_AT_EACH_END: usize = 4;
    let mut rendered = format!(" [walk over {candidates} candidate row(s):");
    let render_rung = |rung: &LadderRung, into: &mut String| {
        into.push_str(&format!(
            " (excluded={} retained={} departure={:.3e})",
            rung.excluded, rung.retained, rung.departure
        ));
    };
    if ladder.len() <= 2 * SHOWN_AT_EACH_END + 1 {
        for rung in ladder {
            render_rung(rung, &mut rendered);
        }
    } else {
        for rung in &ladder[..SHOWN_AT_EACH_END] {
            render_rung(rung, &mut rendered);
        }
        rendered.push_str(&format!(
            " ... {} further rung(s) ...",
            ladder.len() - 2 * SHOWN_AT_EACH_END
        ));
        for rung in &ladder[ladder.len() - SHOWN_AT_EACH_END..] {
            render_rung(rung, &mut rendered);
        }
    }
    rendered.push(']');
    rendered
}

/// Emit a refused face itself, at full precision, on the diagnostic channel.
///
/// Called ONLY from the walk's terminal error paths, not per rung. Dropping
/// rows is the walk working, not the walk failing — a `q × q` matrix printed on
/// every rung would be a megabyte of warnings for a correction that then
/// succeeds — while a terminal refusal is unreproducible without the face,
/// because the walk's decisions are a function of `W` alone and the fit that
/// produced `W` costs three quarters of an hour to reach this line on the
/// #2714 witness. Printing it there turns that refusal into a unit fixture.
fn log_refused_face(face: &RefusedFace, departure: f64, excluded: usize) {
    if !log::log_enabled!(log::Level::Warn) {
        return;
    }
    let q = face.rows.len();
    let mut rendered = String::new();
    for i in 0..q {
        for j in 0..q {
            rendered.push_str(&format!("{:.17e},", face.w[[i, j]]));
        }
    }
    log::warn!(
        "[CONSTRAINED-FACE] refused excluded={excluded} departure={departure:.6e} \
         q={q} rows={:?} w=[{rendered}]",
        face.rows
    );
}

/// The last face the walk refused, kept so a terminal error can print it.
///
/// Only `rows` and `W` are retained: they are what makes the refusal
/// reproducible, and keeping the whole [`RetainedFace`] alive across a pass
/// would hold its `p × q` block and its factor for a walk that may run once per
/// candidate row.
struct RefusedFace {
    rows: Vec<usize>,
    w: Array2<f64>,
}

/// The constraint rows that can still move a moment, in the order the rank
/// filter walks them: `(row index, standardized slack, Σ a_row)`.
///
/// A row whose remaining feasible mass `Φ̄(s)` is below `f64::EPSILON` cannot
/// change any moment at double precision, so the horizon is read off the machine
/// epsilon rather than chosen.
///
/// The order is by standardized slack, and that is a STATISTICAL choice which
/// stays: two near-parallel rows with different offsets are not the same
/// constraint, and the tighter one dominates, so retaining the slacker of a pair
/// would quietly relax the constraint by a multiple of its own standard
/// deviation. Ordering by pivot magnitude instead would reveal the face's
/// conditioning but pay exactly that cost — which is why
/// [`assemble_retained_face`]'s per-row floor is necessary and not sufficient,
/// and why the caller has to check the assembled face.
///
/// Named and extracted so a test can build the same candidate list the
/// production walk builds, rather than a second implementation of it.
fn constraint_face_candidates(
    sigma_times_constraint_transpose: ArrayView2<'_, f64>,
    unconstrained_center: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Result<Vec<(usize, f64, Array1<f64>)>, String> {
    let slack_horizon = -standard_normal_quantile(f64::EPSILON)
        .map_err(|error| format!("resolution horizon for the constraint slack: {error}"))?;
    let mut candidates: Vec<(usize, f64, Array1<f64>)> = Vec::new();
    for row_index in 0..constraints.a.nrows() {
        let row = constraints.a.row(row_index).to_owned();
        let sigma_row = sigma_times_constraint_transpose
            .column(row_index)
            .to_owned();
        let variance = row.dot(&sigma_row);
        if !(variance.is_finite() && variance > 0.0) {
            // The constraint normal has no posterior spread at all: the fit
            // cannot move along it, so the truncation removes nothing.
            continue;
        }
        let slack = (row.dot(unconstrained_center) - constraints.b[row_index]) / variance.sqrt();
        if !slack.is_finite() {
            return Err(format!(
                "constraint row {row_index} produced a non-finite standardized slack"
            ));
        }
        if slack < slack_horizon {
            candidates.push((row_index, slack, sigma_row));
        }
    }
    candidates.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok(candidates)
}

/// One assembled constraint face: the retained rows in acceptance order, the
/// lower Cholesky factor of `W = A Σ Aᵀ` built while retaining them, `W` itself,
/// and the `Σ Aᵀ` block restricted to those rows.
struct RetainedFace {
    rows: Vec<usize>,
    factor: Array2<f64>,
    w: Array2<f64>,
    sigma_at: Array2<f64>,
    /// Upper limit per retained coordinate, `f64::INFINITY` for a half-line.
    upper: Vec<f64>,
    /// The retained row whose `pivot/diagonal` is smallest, together with every
    /// refused row whose far wall was folded into it. Empty for an empty face.
    ///
    /// `pivot/diagonal` is the squared sine of the angle between a row's
    /// constraint normal and the span of the rows accepted before it, in the
    /// `Σ` metric, so the minimizer is the retained row that is most nearly
    /// dependent on the others — the one carrying the least constraint
    /// information and the most of the face's ill-conditioning. It is what the
    /// retention walk drops when the assembled face's lift misses its own
    /// identity, and it is reported BY INDEX so that decision is a set
    /// operation rather than the inversion of a floating-point comparison
    /// (#2714).
    ///
    /// The FOLDED rows travel with it because the walk drops a DIRECTION, not a
    /// row. A row anti-parallel to an accepted one is refused as a direction and
    /// keeps its wall as that row's upper limit (see
    /// [`record_opposed_face_limit`]); if the accepted row is then dropped for
    /// being nearly dependent, its opposite face is the same nearly dependent
    /// direction and is no more liftable. Leaving it behind would let the next
    /// pass accept it in the dropped row's place — with the SLACKER of the two
    /// walls, since the walk is ordered by slack, and with nothing to fold its
    /// own far wall into — turning a two-sided bound into a one-sided one on
    /// the wrong side.
    ///
    /// The first accepted row has an empty forward-substitution column, hence
    /// `pivot == diagonal` and a ratio of exactly `1`, which is the maximum the
    /// ratio can take; combined with the walk's tie-break toward the later (and
    /// therefore slacker) row, that makes the tightest wall the one row a face
    /// of two or more can never drop.
    least_independent_direction: Vec<usize>,
}

/// Greedy pivoted-Cholesky rank filter on `W = A Σ Aᵀ`, walking the candidates
/// in their slack order and skipping any whose constraint-row index appears in
/// `excluded`.
///
/// The factor is built incrementally here and handed back, so the face is
/// factorized exactly once: a second factorization of the same `W` under a
/// different guard would let one matrix be judged by two standards, and near the
/// retention floor those two standards disagree.
///
/// `excluded` is the caller's record of the rows its previous faces already
/// judged too nearly dependent to lift accurately. It is a plain index list
/// rather than a set because it is walked once per candidate and never holds
/// more entries than the face had rows.
fn assemble_retained_face(
    candidates: &[(usize, f64, Array1<f64>)],
    demanded_accuracy: f64,
    constraints: &LinearInequalityConstraints,
    unconstrained_center: &Array1<f64>,
    excluded: &[usize],
) -> Result<Option<RetainedFace>, String> {
    let columns = constraints.a.ncols();
    // Each of `cross[k]`, `W_kk` and `diagonal` is one length-`p` inner product
    // of a constraint row against a `Σ` column, so each carries the standard
    // `γ_p ≈ p·ε` dot-product rounding; the anti-parallel test below compares
    // two products of such quantities, which propagates to about `4(p+1)ε`.
    // Nothing here is fitted to a fixture: it is the resolution at which "the
    // same direction, reversed" stops being decidable in double precision.
    let antiparallel_tolerance = 4.0 * (columns as f64 + 1.0) * f64::EPSILON;
    let mut rows: Vec<usize> = Vec::new();
    let mut least_independent: Option<(usize, f64)> = None;
    let mut sigma_a_columns: Vec<Array1<f64>> = Vec::new();
    let mut upper: Vec<f64> = Vec::new();
    // Per accepted position, the refused rows whose far wall was folded into
    // it. Parallel to `rows` and `upper`.
    let mut folded: Vec<Vec<usize>> = Vec::new();
    let mut w_accepted = Array2::<f64>::zeros((0, 0));
    let mut factor = Array2::<f64>::zeros((0, 0));
    for (row_index, _, sigma_row) in candidates {
        if excluded.contains(row_index) {
            continue;
        }
        let row = constraints.a.row(*row_index);
        let accepted = rows.len();
        let diagonal = row.dot(sigma_row);
        let mut cross = Array1::<f64>::zeros(accepted);
        for (position, column) in sigma_a_columns.iter().enumerate() {
            cross[position] = row.dot(column);
        }
        // Forward-substitute the new column through the accepted factor.
        let mut new_column = Array1::<f64>::zeros(accepted);
        for i in 0..accepted {
            let mut sum = cross[i];
            for k in 0..i {
                sum -= factor[[i, k]] * new_column[k];
            }
            new_column[i] = sum / factor[[i, i]];
        }
        let pivot = diagonal - new_column.dot(&new_column);
        // `pivot / diagonal` is the squared sine of the angle between this row's
        // constraint normal and the span of the rows accepted before it, in the
        // `Σ` metric. The lift `G = Σ Aᵀ W⁻¹` is solved through this same factor,
        // so its relative error grows like `ε · diagonal / pivot`. A floor at the
        // bare DETECTABILITY limit — `pivot ≈ ε · diagonal`, i.e. "reject only a
        // row that is dependent to the last bit" — therefore retains rows whose
        // lift carries no correct digits: measured 1.3e-2 relative error against
        // an exact rational reference at `pivot = 2 ε · diagonal`.
        //
        // So the floor is the one that keeps the retained face's own numerical
        // error under the accuracy demanded of it, and dropping instead costs
        // `O(θ)` with `θ` the angle between the two normals — below `5e-7`
        // radians at the first pass's floor — so a dropped row imposes no
        // constraint the retained one does not already impose.
        //
        // This is necessary and NOT sufficient, which is why the caller checks
        // the assembled face and excludes its least independent row when it
        // falls short: `min pivot / diagonal` is the smallest pivot of the
        // correlation matrix and bounds its smallest eigenvalue only when the
        // elimination is ordered by pivot magnitude. This walk is ordered by
        // slack, so every row can clear the floor while the face as a whole
        // does not.
        let rank_floor = (accepted + 1) as f64 * f64::EPSILON * diagonal / demanded_accuracy;
        if !(pivot.is_finite() && pivot > rank_floor) {
            // Redundant AS A DIRECTION. That is not the same as redundant as a
            // CONSTRAINT, and the two come apart exactly at an anti-parallel
            // row: `l ≤ β_j ≤ u` arrives as `e_jᵀβ ≥ l` and `−e_jᵀβ ≥ −u`, and
            // the second adds no constraint-normal direction while halving the
            // support. Dropping it reports a one-sided posterior for a
            // two-sided bound (#2523).
            //
            // A row PARALLEL to an accepted one is genuinely implied, and the
            // slack ordering is what makes that true rather than hoped: rows are
            // walked by ascending standardized slack and `sd` scales with the
            // row, so the accepted row's `slack/sd` is the smaller, which is
            // exactly the statement that its wall is the binding one. Those
            // still drop, unchanged.
            if let Some(position) = record_opposed_face_limit(
                *row_index,
                &cross,
                diagonal,
                &AcceptedFace {
                    w_accepted: &w_accepted,
                    rows: &rows,
                    constraints,
                    unconstrained_center,
                    antiparallel_tolerance,
                },
                &mut upper,
            )? {
                folded[position].push(*row_index);
            }
            continue;
        }
        let mut grown = Array2::<f64>::zeros((accepted + 1, accepted + 1));
        grown
            .slice_mut(ndarray::s![..accepted, ..accepted])
            .assign(&factor);
        for i in 0..accepted {
            grown[[accepted, i]] = new_column[i];
        }
        grown[[accepted, accepted]] = pivot.sqrt();
        factor = grown;

        let mut grown_w = Array2::<f64>::zeros((accepted + 1, accepted + 1));
        grown_w
            .slice_mut(ndarray::s![..accepted, ..accepted])
            .assign(&w_accepted);
        for i in 0..accepted {
            grown_w[[accepted, i]] = cross[i];
            grown_w[[i, accepted]] = cross[i];
        }
        grown_w[[accepted, accepted]] = diagonal;
        w_accepted = grown_w;

        rows.push(*row_index);
        sigma_a_columns.push(sigma_row.clone());
        upper.push(f64::INFINITY);
        folded.push(Vec::new());
        // The independence ratio of the row just accepted. `diagonal > 0` is
        // guaranteed by `constraint_face_candidates`, and `pivot > rank_floor > 0`
        // by the branch above, so the ratio is a finite positive number and the
        // comparison never sees a NaN.
        //
        // `<=` keeps the LAST minimizer, and the rows arrive in ascending slack
        // order, so a tie is broken toward the slacker row. That matters at the
        // one tie that is structural rather than accidental: the first accepted
        // row has an empty `new_column`, hence `pivot == diagonal` and a ratio
        // of exactly `1`, so on a face whose rows are all mutually orthogonal
        // in the `Σ` metric every ratio is `1` and a `<` rule would drop the
        // TIGHTEST wall.
        let independence = pivot / diagonal;
        if least_independent.is_none_or(|(_, best)| independence <= best) {
            least_independent = Some((rows.len() - 1, independence));
        }
    }
    if rows.is_empty() {
        return Ok(None);
    }

    let q = rows.len();
    let p = sigma_a_columns[0].len();
    let mut sigma_at = Array2::<f64>::zeros((p, q));
    for (position, column) in sigma_a_columns.iter().enumerate() {
        sigma_at.column_mut(position).assign(column);
    }
    let least_independent_direction = match least_independent {
        Some((position, _)) => {
            let mut direction = vec![rows[position]];
            direction.extend_from_slice(&folded[position]);
            direction
        }
        None => Vec::new(),
    };
    Ok(Some(RetainedFace {
        rows,
        factor,
        w: w_accepted,
        sigma_at,
        upper,
        least_independent_direction,
    }))
}

/// The accepted face an anti-parallel test reads: the retained rows with their
/// `W` block, the constraint system those rows index into, the centre the walls
/// are measured from, and the correlation at which two normals count as opposed.
///
/// These five never vary independently at the call site — they all describe one
/// accepted face — so they travel as one borrow rather than as five parameters
/// whose covariance the signature does not state.
struct AcceptedFace<'a> {
    w_accepted: &'a Array2<f64>,
    rows: &'a [usize],
    constraints: &'a LinearInequalityConstraints,
    unconstrained_center: &'a Array1<f64>,
    antiparallel_tolerance: f64,
}

/// Keep the far wall of a two-sided bound that the rank filter has just refused
/// as a direction.
///
/// The refused row `a_r` carries no constraint-normal direction beyond the
/// accepted ones. When it is the OPPOSITE face of one of them — `a_r = −γ a_k`
/// for some `γ > 0`, up to a remainder with no posterior variance — it still
/// bounds that coordinate, from above:
///
/// ```text
/// a_rᵀβ ≥ b_r   ⟺   a_kᵀβ ≤ (ν − b_r)/γ   ⟺   u_k ≤ δ/γ,
/// ```
///
/// with `u_k = a_kᵀβ − b_k`, `ν = (a_r + γ a_k)ᵀβ` (almost surely constant,
/// precisely because its posterior variance is the pivot that just failed) and
/// `δ = (a_rᵀβ_unc − b_r) + γ(a_kᵀβ_unc − b_k)`.
///
/// The test is run in the `Σ` metric on quantities the filter has already
/// formed: `a_r = −γ a_k` makes the correlation `cross_k/√(W_kk·diagonal)`
/// exactly `−1`, and `γ = −cross_k/W_kk`. Running it in that metric rather than
/// on the raw rows is not a convenience — two rows that differ by a direction
/// with no posterior spread impose the same constraint on this posterior, and
/// the `Σ` metric is what sees that.
///
/// Rows that are refused for any other reason are left dropped, which is what
/// they were before. That is a narrower repair than "every dependent row", and
/// deliberately so: a row depending on two or more accepted normals at once cuts
/// the face along a diagonal, which no per-coordinate limit can represent.
///
/// Returns the accepted POSITION the wall was folded into, so the caller can
/// keep the pair together: the folded row is the opposite face of that accepted
/// row's direction, and if the retention walk later drops the direction the far
/// wall has to go with it rather than be promoted in its place (#2714).
fn record_opposed_face_limit(
    row_index: usize,
    cross: &Array1<f64>,
    diagonal: f64,
    face: &AcceptedFace<'_>,
    upper: &mut [f64],
) -> Result<Option<usize>, String> {
    let mut opposed: Option<(usize, f64, f64)> = None;
    for position in 0..face.rows.len() {
        let w_kk = face.w_accepted[[position, position]];
        let scale = (w_kk * diagonal).sqrt();
        if !(scale.is_finite() && scale > 0.0) {
            continue;
        }
        let correlation = cross[position] / scale;
        if correlation + 1.0 > face.antiparallel_tolerance {
            continue;
        }
        let gamma = -cross[position] / w_kk;
        if !(gamma.is_finite() && gamma > 0.0) {
            continue;
        }
        // Two accepted rows cannot both be anti-parallel to this one without
        // being parallel to each other, which the filter already refused; take
        // the most opposed and let the identity gate catch a face that is not.
        if opposed.is_none_or(|(_, best, _)| correlation < best) {
            opposed = Some((position, correlation, gamma));
        }
    }
    let Some((position, _, gamma)) = opposed else {
        return Ok(None);
    };
    let accepted_row = face.rows[position];
    let delta = (face.constraints.a.row(row_index).dot(face.unconstrained_center)
        - face.constraints.b[row_index])
        + gamma
            * (face.constraints.a.row(accepted_row).dot(face.unconstrained_center)
                - face.constraints.b[accepted_row]);
    let limit = delta / gamma;
    if !(limit.is_finite() && limit > 0.0) {
        return Err(format!(
            "constraint rows {accepted_row} and {row_index} bound the same coefficient \
             direction from opposite sides with no width between them (upper limit \
             {limit:.6e} above the lower wall): the retained region is empty or a single \
             point, which is an equality constraint and not a posterior this module can \
             report moments for"
        ));
    }
    if limit < upper[position] {
        upper[position] = limit;
    }
    Ok(Some(position))
}

/// `max |A G - I|` over the retained rows.
///
/// `G = Σ Aᵀ W⁻¹` satisfies `A G = I` on those rows EXACTLY, because
/// `A (Σ Aᵀ) = W` by construction of `W`. The departure from that identity is
/// therefore not a modelling approximation: it is precisely the accuracy the
/// retained face's conditioning destroyed, measured on the object that is
/// actually used rather than on a proxy for it. Across a sweep of near-parallel
/// constraint normals it tracked the true error in `G` — against an exact
/// rational reference — to three significant digits at every angle.
fn lift_identity_departure(
    lift: &Array2<f64>,
    constraints: &LinearInequalityConstraints,
    rows: &[usize],
) -> Result<f64, String> {
    let q = rows.len();
    let mut departure = 0.0_f64;
    for (i, &row_index) in rows.iter().enumerate() {
        let row = constraints.a.row(row_index);
        for j in 0..q {
            let entry = row.dot(&lift.column(j));
            let target = if i == j { 1.0 } else { 0.0 };
            let deviation = (entry - target).abs();
            if !deviation.is_finite() {
                return Err(format!(
                    "the constraint-normal lift is not finite at retained row {row_index}, \
                     constraint-normal coordinate {j}"
                ));
            }
            departure = departure.max(deviation);
        }
    }
    Ok(departure)
}

/// Solve `X W = B` for `X` given the lower Cholesky factor `L` of the symmetric
/// `W = L Lᵀ`, i.e. return `B W⁻¹`. `W` is symmetric so `X = (W⁻¹ Bᵀ)ᵀ`.
fn cholesky_solve_right(factor: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
    let q = factor.nrows();
    if b.ncols() != q {
        return Err(format!(
            "constraint-normal solve: factor is {q}x{q} but the right-hand side has {} columns",
            b.ncols()
        ));
    }
    let rows = b.nrows();
    let mut out = Array2::<f64>::zeros((rows, q));
    let mut work = Array1::<f64>::zeros(q);
    for r in 0..rows {
        for i in 0..q {
            let mut sum = b[[r, i]];
            for k in 0..i {
                sum -= factor[[i, k]] * work[k];
            }
            work[i] = sum / factor[[i, i]];
        }
        for i in (0..q).rev() {
            let mut sum = work[i];
            for k in (i + 1)..q {
                sum -= factor[[k, i]] * out[[r, k]];
            }
            out[[r, i]] = sum / factor[[i, i]];
        }
    }
    Ok(out)
}

/// Refuse a correction that is not a genuine variance REMOVAL. `Δ = W − Cov[u]`
/// must be positive semidefinite (truncation cannot inflate a Gaussian's
/// covariance) and must not exceed `W` (it cannot remove more variance than
/// there was). Either failure means the cubature returned something that is not
/// the moment of a distribution, which is a numerical failure and not a number
/// to report.
fn certify_removed_variance(removed: &Array2<f64>, w: &Array2<f64>) -> Result<(), String> {
    let q = removed.nrows();
    // Scale-free bound: both `Δ` and `W − Δ = Cov[u]` are checked against the
    // cubature's own accuracy in the pre-truncation metric.
    let slack = ORTHANT_MOMENT_RELATIVE_TOLERANCE * (q as f64);
    for i in 0..q {
        let scale = w[[i, i]];
        if removed[[i, i]] < -slack * scale {
            return Err(format!(
                "truncated orthant moments inflated the constraint-normal variance at index {i} \
                 (removed {:.6e} against scale {scale:.6e}); truncation cannot increase a \
                 Gaussian covariance",
                removed[[i, i]]
            ));
        }
        if removed[[i, i]] > (1.0 + slack) * scale {
            return Err(format!(
                "truncated orthant moments removed more variance than exists at index {i} \
                 (removed {:.6e} against scale {scale:.6e})",
                removed[[i, i]]
            ));
        }
        for j in 0..q {
            if !removed[[i, j]].is_finite() {
                return Err(format!(
                    "truncated orthant moments produced a non-finite entry at ({i},{j})"
                ));
            }
        }
    }
    Ok(())
}

/// First two moments of `u ~ N(mean, covariance)` restricted to the box
/// `0 ≤ u ≤ upper`, where `upper_i = f64::INFINITY` makes coordinate `i` the
/// half-line the orthant is built from.
///
/// One dimension has the closed form and is evaluated exactly. Higher
/// dimensions use the Genz separation-of-variables transformation, under which
/// EVERY moment is an integral of the same integrand over the unit cube — so a
/// single cubature delivers the normalizing probability, the mean and the second
/// moment together, instead of the `O(q²)` separate orthant probabilities the
/// Tallis face/edge recursion would need. The transformation is already a
/// product of intervals; a finite upper limit changes only where each interval
/// ends, which is why a box costs the same cubature as an orthant.
fn box_truncated_moments(
    mean: &Array1<f64>,
    upper: &[f64],
    covariance: &Array2<f64>,
    factor: ArrayView2<'_, f64>,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let q = mean.len();
    if covariance.nrows() != q || covariance.ncols() != q {
        return Err(format!(
            "truncated moments: mean has length {q} but the covariance is {}x{}",
            covariance.nrows(),
            covariance.ncols()
        ));
    }
    if upper.len() != q {
        return Err(format!(
            "truncated moments: mean has length {q} but {} upper limits were supplied",
            upper.len()
        ));
    }
    if upper.iter().any(|limit| !(*limit > 0.0)) {
        return Err(format!(
            "truncated moments: every upper limit must sit strictly above its wall, got {upper:?}"
        ));
    }
    if q == 1 {
        return scalar_truncated_moments(mean[0], covariance[[0, 0]], upper[0]);
    }
    if factor.nrows() != q || factor.ncols() != q {
        return Err(format!(
            "orthant moments: the constraint-normal covariance is {q}x{q} but the Cholesky \
             factor supplied with it is {}x{}",
            factor.nrows(),
            factor.ncols()
        ));
    }

    let generator = kronecker_generator(q);
    // One tilt for the whole face: it depends only on the geometry, not on the
    // node, so it is solved once and reused by every refinement pass.
    let tilt = minimax_tilt(mean, upper, factor);
    let mut accumulator = OrthantAccumulator::new(q);
    let mut evaluated = 0usize;
    let mut previous: Option<(Array1<f64>, Array2<f64>)> = None;
    loop {
        let target = if evaluated == 0 {
            ORTHANT_MOMENT_INITIAL_POINTS
        } else {
            evaluated * 2
        };
        accumulate_orthant_nodes(
            &mut accumulator,
            mean,
            upper,
            None,
            factor,
            &generator,
            tilt.as_ref(),
            0,
            evaluated,
            target,
        )?;
        evaluated = target;
        let current = accumulator.moments()?;
        if let Some(ref last) = previous
            && moment_relative_change(last, &current, covariance)
                <= ORTHANT_MOMENT_RELATIVE_TOLERANCE
        {
            return Ok(current);
        }
        if evaluated >= ORTHANT_MOMENT_MAXIMUM_POINTS {
            let change = previous
                .as_ref()
                .map(|last| moment_relative_change(last, &current, covariance))
                .unwrap_or(f64::INFINITY);
            // Report the FACE, not only the verdict on it. "Did not converge"
            // names a number without the geometry that decided it, and the two
            // properties that actually govern this integrand's difficulty are
            // measurable in three lines: how deep the walls sit in standardized
            // units, and how correlated the constraint normals are. Measured
            // on synthetic faces (`orthant_depth_measure_2601_tests`), depth
            // alone is harmless — an 11-dimensional face 4 sd deep with
            // UNCORRELATED normals converges in 16k nodes — while intermediate
            // correlation at the same depth does not converge at 2^20 (#2601).
            let depth: Vec<f64> = (0..q)
                .map(|i| -mean[i] / covariance[[i, i]].sqrt())
                .collect();
            let depth_min = depth.iter().copied().fold(f64::INFINITY, f64::min);
            let depth_max = depth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut corr_max: f64 = 0.0;
            for i in 0..q {
                for j in 0..i {
                    let denominator = (covariance[[i, i]] * covariance[[j, j]]).sqrt();
                    if denominator > 0.0 {
                        corr_max = corr_max.max((covariance[[i, j]] / denominator).abs());
                    }
                }
            }
            log::debug!(
                "[orthant-face] q={q} mean={:?} covariance={:?}",
                mean.as_slice().map(<[f64]>::to_vec),
                covariance.as_slice().map(<[f64]>::to_vec),
            );
            return Err(format!(
                "truncated moments for a {q}-dimensional constraint face did not converge: \
                 relative moment change {change:.3e} still exceeds \
                 {ORTHANT_MOMENT_RELATIVE_TOLERANCE:.1e} at {evaluated} cubature nodes \
                 (wall depth {depth_min:.2}..{depth_max:.2} sd, \
                 max |correlation| between constraint normals {corr_max:.3})"
            ));
        }
        previous = Some(current);
    }
}

/// Running log-scaled first and second moment accumulator for the cubature.
///
/// Node weights span hundreds of decades between a barely-truncated face and a
/// deeply pinned one, so the accumulators carry an explicit log scale and are
/// rescaled whenever a heavier node arrives. Accumulating the weights directly
/// would underflow the whole face to zero and leave the normalized moments as
/// `0/0`.
struct OrthantAccumulator {
    log_scale: f64,
    weight_sum: f64,
    weighted_mean: Array1<f64>,
    weighted_second: Array2<f64>,
}

trait OrthantNodeSink {
    fn push(&mut self, log_weight: f64, point: &Array1<f64>);

    /// Same node, with the TANGENT block of the joint rule attached.
    ///
    /// `tangent` holds the standard-normal coordinates drawn from the lattice
    /// dimensions past the constraint-normal block, and is empty whenever the
    /// caller asked for none. The default drops them, so every sink that only
    /// wants the truncated marginal is unaffected — and with
    /// `tangent_dimension = 0` the arithmetic reaching [`Self::push`] is
    /// bit-identical to the rule before the joint block existed, because the
    /// Kronecker generator of a larger dimension has the smaller one as its
    /// exact prefix.
    fn push_joint(&mut self, log_weight: f64, point: &Array1<f64>, tangent: &[f64]) {
        // A sink that never asked for a tangent block must never be handed
        // one. Dropping the coordinates instead would make a joint rule read as
        // a marginal rule with the same node count, which is a wrong answer
        // that looks exactly like a right one.
        assert!(
            tangent.is_empty(),
            "a sink with no joint tangent block was handed {} tangent coordinates",
            tangent.len()
        );
        self.push(log_weight, point);
    }
}

impl OrthantAccumulator {
    fn new(q: usize) -> Self {
        Self {
            log_scale: f64::NEG_INFINITY,
            weight_sum: 0.0,
            weighted_mean: Array1::zeros(q),
            weighted_second: Array2::zeros((q, q)),
        }
    }

    fn push(&mut self, log_weight: f64, point: &Array1<f64>) {
        let q = point.len();
        if log_weight > self.log_scale {
            let rescale = (self.log_scale - log_weight).exp();
            self.weight_sum *= rescale;
            self.weighted_mean *= rescale;
            self.weighted_second *= rescale;
            self.log_scale = log_weight;
        }
        let weight = (log_weight - self.log_scale).exp();
        self.weight_sum += weight;
        for i in 0..q {
            self.weighted_mean[i] += weight * point[i];
            for j in 0..=i {
                self.weighted_second[[i, j]] += weight * point[i] * point[j];
            }
        }
    }

    fn moments(&self) -> Result<(Array1<f64>, Array2<f64>), String> {
        if !(self.weight_sum.is_finite() && self.weight_sum > 0.0) {
            return Err(format!(
                "orthant cubature accumulated no feasible mass (weight sum {:?}); the \
                 constraint face has no representable interior",
                self.weight_sum
            ));
        }
        let q = self.weighted_mean.len();
        let mean = &self.weighted_mean / self.weight_sum;
        let mut covariance = Array2::<f64>::zeros((q, q));
        for i in 0..q {
            for j in 0..=i {
                let centered = self.weighted_second[[i, j]] / self.weight_sum - mean[i] * mean[j];
                covariance[[i, j]] = centered;
                covariance[[j, i]] = centered;
            }
        }
        Ok((mean, covariance))
    }
}

impl OrthantNodeSink for OrthantAccumulator {
    fn push(&mut self, log_weight: f64, point: &Array1<f64>) {
        OrthantAccumulator::push(self, log_weight, point);
    }
}

/// One affine wall, already expressed in the cubature's own standardized
/// coordinates.
///
/// The box ceiling the cubature already carries is a wall whose normal is a
/// single coordinate, so it constrains that coordinate directly. A general wall
/// `aᵀu ≤ c` does not: after `u = mean + L z` it reads `(Lᵀa)ᵀz ≤ c − aᵀmean`,
/// which constrains the LAST coordinate its transformed normal touches, given
/// the ones drawn before it. That is the whole difference between the two, and
/// it is why the limit has to be computed per node rather than once per
/// coordinate.
///
/// `pivot` is that last touched coordinate. The wall says nothing about any
/// coordinate before it and must not be consulted there.
#[derive(Clone, Debug)]
pub struct StandardizedCeiling {
    /// `Lᵀa`, in the cubature's standardized coordinates.
    coefficients: Array1<f64>,
    /// `c − aᵀmean`.
    bound: f64,
    pivot: usize,
}

impl StandardizedCeiling {
    /// Build the standardized form of `normal · u ≤ bound` for the law
    /// `u ~ N(mean, L Lᵀ)`.
    ///
    /// Refuses a wall whose transformed normal vanishes: such a wall constrains
    /// no coordinate of the cubature and is either redundant or infeasible, and
    /// the two cannot be told apart from the normal alone.
    pub fn new(
        normal: &Array1<f64>,
        bound: f64,
        mean: &Array1<f64>,
        factor: ArrayView2<'_, f64>,
    ) -> Result<Self, String> {
        let q = mean.len();
        if normal.len() != q {
            return Err(format!(
                "affine ceiling: the normal has length {} but the law has {q} coordinates",
                normal.len()
            ));
        }
        let mut coefficients = Array1::<f64>::zeros(q);
        for j in 0..q {
            let mut total = 0.0;
            for k in j..q {
                total += factor[[k, j]] * normal[k];
            }
            coefficients[j] = total;
        }
        let scale = coefficients
            .iter()
            .fold(0.0f64, |worst, value| worst.max(value.abs()));
        if !(scale.is_finite() && scale > 0.0) {
            return Err(format!(
                "affine ceiling: the standardized normal vanished (scale {scale:?}); the wall                  constrains no cubature coordinate"
            ));
        }
        let floor = 8.0 * f64::EPSILON * scale;
        let pivot = (0..q)
            .rev()
            .find(|j| coefficients[*j].abs() > floor)
            .ok_or_else(|| "affine ceiling: no coordinate clears the pivot floor".to_string())?;
        let offset = normal.dot(mean);
        if !(bound - offset).is_finite() {
            return Err(format!(
                "affine ceiling: the standardized bound is not finite (bound {bound:?},                  offset {offset:?})"
            ));
        }
        Ok(Self {
            coefficients,
            bound: bound - offset,
            pivot,
        })
    }

    /// The wall's limit on coordinate `pivot` given the coordinates before it,
    /// as `(lower, upper)` additions — a positive pivot coefficient caps the
    /// coordinate from above, a negative one raises its floor.
    fn limit(&self, z: &Array1<f64>) -> (f64, f64) {
        let mut remaining = self.bound;
        for j in 0..self.pivot {
            remaining -= self.coefficients[j] * z[j];
        }
        let coefficient = self.coefficients[self.pivot];
        let limit = remaining / coefficient;
        if coefficient > 0.0 {
            (f64::NEG_INFINITY, limit)
        } else {
            (limit, f64::INFINITY)
        }
    }
}

/// Evaluate Genz nodes `first..last` of the Kronecker sequence and fold them
/// into `accumulator`.
/// Exponential tilt for the separation-of-variables proposal, chosen at the
/// saddle point of the log-weight.
///
/// ## Why a tilt at all
///
/// The Genz estimator draws `z_i` from the standard normal truncated to its
/// conditional interval `[wall_i, oo)` and weights the node by that interval's
/// mass. When the constraint normals are CORRELATED, `wall_i` depends strongly
/// on the coordinates drawn before it, so the product of masses swings from
/// node to node. Measured on the face #2601's `monotone_decreasing` fit
/// produces: the log-weights span 26 decades and the effective sample size is
/// 589 of 65,536 nodes -- 0.9%. The estimator is then plain Monte Carlo with
/// `N_eff = ESS` no matter the node budget, which reproduces its observed error
/// to within a few percent (`1/sqrt(ESS)`), and no point set can repair that:
/// the PROPOSAL is wrong, not the quadrature.
///
/// Sampling `z_i ~ N(mu_i, 1)` truncated to the same interval and reweighting by
/// `exp(mu_i^2/2 - mu_i z_i)` leaves the estimator EXACTLY unbiased for any
/// `mu`, so the choice is a pure conditioning decision carrying no correctness
/// risk -- only variance.
///
/// ## Which tilt, and one that was measured and REJECTED
///
/// The obvious candidate is the dominating point: the mode of the truncated
/// density in `z` coordinates, i.e. the projection of the origin onto
/// `{z : mean + L z >= 0}`. That is the classical importance-sampling shift for
/// a convex region, and it is WRONG here -- measured, on this face, ESS
/// *fell* from 589 to 3.1 and the weight range grew from 26 decades to 65.
/// The reason is structural: the SOV proposal is already conditionally
/// truncated, so it only ever produces feasible points and its conditional mean
/// already sits above the wall. Shifting it by a further `mu > 0` pushes every
/// draw deep into the tail and the likelihood ratio `exp(mu^2/2 - mu z)` then
/// swings harder than the mass product it was meant to flatten.
///
/// What the weight actually is, is
///
/// ```text
/// psi(z, mu) = sum_i [ log Phibar(wall_i(z) - mu_i) + mu_i^2/2 - mu_i z_i ]
/// ```
///
/// so the tilt that flattens it is the one that makes `psi` STATIONARY -- the
/// saddle point in `(z, mu)` jointly (Botev, JRSS-B 2017, minimax tilting).
/// Differentiating gives a closed pair of conditions, with `rho(t) =
/// phi(t)/Phibar(t)` the normal hazard:
///
/// ```text
/// (A)   z_i  =  mu_i + rho(wall_i(z) - mu_i)
/// (B)   mu_k =  sum_{i>k} rho(wall_i(z) - mu_i) * L_ik / L_ii
/// ```
///
/// (A) places each coordinate at the conditional mean its own tilted, truncated
/// law would give; (B) sets each tilt to cancel, to first order, the movement
/// that coordinate induces in every LATER wall -- which is exactly the
/// correlation-driven fluctuation that destroyed the effective sample size.
/// `mu_q = 0` falls out of (B): the last coordinate moves no wall.
///
/// Solved by Gauss-Seidel with damping. Convergence is not required for
/// correctness -- any iterate is an unbiased tilt -- so the iteration is capped
/// and its last iterate used.
///
/// Returns `None` when the region already contains the origin (`mean >= 0`:
/// the walls are slack, the weights are already flat, and the untilted rule is
/// the right one) or when any coordinate carries a finite ceiling. A
/// two-sided coordinate is bounded on both sides, so its mass cannot swing the
/// way a half-line's can; leaving it untilted costs a little variance and keeps
/// the hazard evaluation on the one branch that is unconditionally stable.
fn minimax_tilt(
    mean: &Array1<f64>,
    upper: &[f64],
    factor: ArrayView2<'_, f64>,
) -> Option<Array1<f64>> {
    let q = mean.len();
    if q == 0 || factor.nrows() != q || factor.ncols() != q || upper.len() != q {
        return None;
    }
    if upper.iter().any(|limit| limit.is_finite()) {
        return None;
    }
    if mean.iter().all(|&m| m >= 0.0) {
        return None;
    }
    for i in 0..q {
        if !(factor[[i, i]] > 0.0) {
            return None;
        }
    }

    // `rho(t) = phi(t)/Phibar(t)`, evaluated through logs so a deep-tail wall
    // does not form `0/0`. `normal_logsf` keeps the upper tail that `1 - Phi`
    // destroys.
    let hazard = |t: f64| -> f64 {
        let log_density = -0.5 * t * t - 0.5 * (std::f64::consts::TAU).ln();
        let log_tail = normal_logsf(t);
        if !log_tail.is_finite() {
            // Beyond the representable tail the hazard is asymptotically `t`
            // (Mills), which is the correct limit and stays finite.
            return t.max(0.0);
        }
        (log_density - log_tail).exp()
    };

    const MAX_PASSES: usize = 200;
    // Under-relaxation of the Gauss-Seidel moment sweep, not a matrix ridge:
    // the hazard map `rho(t) = phi(t)/Phibar(t)` is steep in the deep tail, so
    // an undamped fixed-point sweep oscillates between the walls instead of
    // contracting. Taking half of each proposed update is the standard
    // successive-under-relaxation choice and leaves the FIXED POINT unchanged
    // (a converged sweep satisfies the same stationarity equations), so it can
    // only affect how many of the `MAX_PASSES` sweeps are needed.
    const DAMPING: f64 = 0.5;
    let mut mu = Array1::<f64>::zeros(q);
    let mut z = Array1::<f64>::zeros(q);
    let mut rho = Array1::<f64>::zeros(q);
    for _ in 0..MAX_PASSES {
        // (A), forward: each wall depends only on the coordinates before it, so
        // one Gauss-Seidel sweep resolves the whole chain.
        for i in 0..q {
            let mut bound = -mean[i];
            for j in 0..i {
                bound -= factor[[i, j]] * z[j];
            }
            let wall = bound / factor[[i, i]];
            let value = hazard(wall - mu[i]);
            if !value.is_finite() {
                return None;
            }
            rho[i] = value;
            z[i] = mu[i] + value;
        }
        // (B), backward.
        let mut moved = 0.0f64;
        for k in 0..q {
            let mut sum = 0.0;
            for i in (k + 1)..q {
                sum += rho[i] * factor[[i, k]] / factor[[i, i]];
            }
            let target = sum;
            let updated = mu[k] + DAMPING * (target - mu[k]);
            moved = moved.max((updated - mu[k]).abs());
            mu[k] = updated;
        }
        if moved <= 1e-12 {
            break;
        }
    }
    if mu.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(mu)
}

fn accumulate_orthant_nodes<S: OrthantNodeSink>(
    accumulator: &mut S,
    mean: &Array1<f64>,
    upper: &[f64],
    ceiling: Option<&StandardizedCeiling>,
    factor: ArrayView2<'_, f64>,
    generator: &[f64],
    // `tilt`: per-coordinate exponential tilt of the proposal
    // (`dominating_point_tilt`). `None` is the untilted estimator, exactly as
    // before. Any tilt leaves the estimator unbiased; it changes only how
    // evenly the node weights are spread.
    tilt: Option<&Array1<f64>>,
    // `tangent_dimension`: how many extra STANDARD-NORMAL coordinates each node
    // carries, drawn from lattice dimensions `q..q + tangent_dimension` of the
    // SAME point. This is what makes the joint rule one rule: the truncated
    // constraint-normal block and the Gaussian tangent block share a single
    // low-discrepancy point, so a consumer integrating over both pays one
    // evaluation per point instead of the product of two rules.
    tangent_dimension: usize,
    first: usize,
    last: usize,
) -> Result<(), String> {
    let q = mean.len();
    if generator.len() < q + tangent_dimension {
        return Err(format!(
            "orthant cubature needs a {}-dimensional generator for {q} constraint normals and \
             {tangent_dimension} tangent coordinates, got {}",
            q + tangent_dimension,
            generator.len()
        ));
    }
    let mut z = Array1::<f64>::zeros(q);
    let mut point = Array1::<f64>::zeros(q);
    let mut tangent = vec![0.0f64; tangent_dimension];
    for node in first..last {
        let offset = node as f64 + 0.5;
        let mut log_weight = 0.0f64;
        for i in 0..q {
            // Everything below runs in the TILTED coordinate `z_i - mu_i`: the
            // conditional interval shifts down by `mu_i` and the arithmetic is
            // untouched, so a zero tilt is bit-identical to the untilted rule.
            let mu = tilt.map(|t| t[i]).unwrap_or(0.0);
            let mut bound = -mean[i];
            for j in 0..i {
                bound -= factor[[i, j]] * z[j];
            }
            let mut wall = bound / factor[[i, i]] - mu;
            // The affine wall, if it pivots here, is a second candidate limit on
            // the SAME interval. Merging it before the interval is resolved is
            // what keeps one path through the reflection and the mass: from here
            // down, an affine-bounded coordinate and a box-bounded one are the
            // same arithmetic on the same `[wall, ceiling]`.
            let mut affine_ceiling = f64::INFINITY;
            if let Some(wall_rule) = ceiling
                && wall_rule.pivot == i
            {
                let (raised, capped) = wall_rule.limit(&z);
                if raised - mu > wall {
                    wall = raised - mu;
                }
                affine_ceiling = capped - mu;
            }
            // Tent-periodized Kronecker lattice. The raw sequence leaves the
            // integrand non-periodic across the cube face, which costs the
            // lattice rule most of its rate; folding `x ↦ 1 − |2x − 1|`
            // preserves the uniform measure and periodizes it.
            let lattice = {
                let raw = offset * generator[i];
                let fractional = raw - raw.floor();
                1.0 - (2.0 * fractional - 1.0).abs()
            };
            if !upper[i].is_finite() && !affine_ceiling.is_finite() {
                let log_tail = normal_logsf(wall);
                if !log_tail.is_finite() {
                    // The remaining feasible mass along this coordinate
                    // underflowed to zero: the node contributes nothing and
                    // cannot be renormalized, so drop it rather than propagate a
                    // NaN.
                    log_weight = f64::NEG_INFINITY;
                    break;
                }
                log_weight += log_tail;
                // `Φ̄(z_i) = (1 − x_i)·Φ̄(lower)` inverted on the upper tail, so
                // a deeply pinned coordinate never forms `1 − Φ(·)` in
                // probability space. Both factors can round to one (an inactive
                // coordinate at the very edge of the lattice cell), which would
                // ask for `Φ̄⁻¹(1)`; the smallest representable log-probability
                // answers that with the far-left endpoint, which is what the
                // region actually is there.
                let log_fraction = (1.0 - lattice).max(f64::MIN_POSITIVE).ln();
                let log_upper_tail = log_fraction + log_tail;
                let resolved = if log_upper_tail < 0.0 {
                    log_upper_tail
                } else {
                    -f64::MIN_POSITIVE
                };
                let shifted = -standard_normal_quantile_from_log_cdf(resolved)
                    .map_err(|error| format!("orthant cubature coordinate {i}: {error}"))?;
                z[i] = shifted + mu;
                // Likelihood ratio of the standard normal to the tilted one:
                // `phi(z)/phi(z - mu) = exp(mu^2/2 - mu z)`. Zero tilt adds zero.
                log_weight += 0.5 * mu * mu - mu * z[i];
                continue;
            }

            // Bounded coordinate. The conditional interval is `[wall, ceiling]`
            // and `ceiling − wall = upper_i / L_ii` exactly, so the width never
            // goes through a subtraction of two conditional means.
            // The box width stays subtraction-free exactly as before; the
            // affine limit is an independent candidate and the tighter one wins.
            let boxed = if upper[i].is_finite() {
                wall + upper[i] / factor[[i, i]]
            } else {
                f64::INFINITY
            };
            let ceiling = boxed.min(affine_ceiling);
            if !(ceiling > wall) {
                // The two walls have crossed: this node's feasible interval is
                // empty, which is a property of the region and not a failure.
                log_weight = f64::NEG_INFINITY;
                break;
            }
            // Reflect an interval that sits in the LOWER tail. Both `Φ̄` values
            // are then within rounding of one and their difference — the
            // interval's entire mass — would be computed as a cancellation
            // between them. Under `z ↦ −z` the same interval is `[−ceiling,
            // −wall]` with both endpoints in the upper tail, where `Φ̄` is
            // evaluated directly. This is the regime a two-sided bound reaches
            // whenever the unconstrained fit lands beyond the far wall, which is
            // exactly when such a bound is worth declaring.
            let reflect = wall + ceiling < 0.0;
            let (low, high) = if reflect {
                (-ceiling, -wall)
            } else {
                (wall, ceiling)
            };
            let log_tail_low = normal_logsf(low);
            let log_tail_high = normal_logsf(high);
            if !log_tail_low.is_finite() {
                log_weight = f64::NEG_INFINITY;
                break;
            }
            // `removed ≤ 0` is the log of the fraction of the half-line's mass
            // that the far wall takes away.
            let removed = log_tail_high - log_tail_low;
            let log_mass = log_tail_low + log1mexp(removed);
            if !log_mass.is_finite() {
                // The slab is narrower than double precision can resolve at this
                // conditional position; it carries no representable mass.
                log_weight = f64::NEG_INFINITY;
                break;
            }
            log_weight += log_mass;
            // `Φ̄(z) = Φ̄(low)·(1 − x(1 − e^removed))`: the same upper-tail
            // inversion as the half-line, with the retained fraction shortened
            // to the slab.
            let retained = (-lattice * (-removed.exp_m1())).ln_1p();
            let log_upper_tail = log_tail_low + retained;
            let resolved = if log_upper_tail < 0.0 {
                log_upper_tail
            } else {
                -f64::MIN_POSITIVE
            };
            let sampled = -standard_normal_quantile_from_log_cdf(resolved)
                .map_err(|error| format!("truncated cubature coordinate {i}: {error}"))?;
            // The inversion is exact in probability space, so an excursion past
            // either endpoint is rounding in `Φ̄⁻¹` alone; the node belongs to
            // the interval by construction and is placed there.
            let clamped = sampled.clamp(low, high);
            z[i] = if reflect { -clamped } else { clamped } + mu;
            log_weight += 0.5 * mu * mu - mu * z[i];
        }
        if !log_weight.is_finite() {
            continue;
        }
        for i in 0..q {
            let mut value = mean[i];
            for j in 0..=i {
                value += factor[[i, j]] * z[j];
            }
            point[i] = value;
        }
        // Tangent block. The tent fold `x ↦ 1 − |2·frac − 1|` is measure
        // preserving on the unit interval, so the folded coordinate is still
        // uniform and `Φ⁻¹` of it is still standard normal. Both tails are
        // inverted from the SMALL side's own logarithm — `1 − lattice` is
        // `|2·frac − 1|` exactly, formed without a subtraction — so a
        // coordinate at either end of the cell never goes through `1 − Φ`.
        for (slot, tangent_value) in tangent.iter_mut().enumerate() {
            let raw = offset * generator[q + slot];
            let fractional = raw - raw.floor();
            let upper_side = (2.0 * fractional - 1.0).abs();
            let lattice = 1.0 - upper_side;
            *tangent_value = if lattice <= 0.5 {
                standard_normal_quantile_from_log_cdf(lattice.max(f64::MIN_POSITIVE).ln())
                    .map_err(|error| format!("joint cubature tangent coordinate {slot}: {error}"))?
            } else {
                -standard_normal_quantile_from_log_cdf(upper_side.max(f64::MIN_POSITIVE).ln())
                    .map_err(|error| format!("joint cubature tangent coordinate {slot}: {error}"))?
            };
        }
        accumulator.push_joint(log_weight, &point, &tangent);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct WeightedProjectionNode {
    conditional_mean: f64,
    weight: f64,
}

struct ProjectionNodeAccumulator<'a> {
    moments: OrthantAccumulator,
    normal_center: &'a Array1<f64>,
    projection_lift: &'a Array1<f64>,
    ambient_mean: f64,
    nodes: Vec<(f64, f64)>,
}

impl<'a> ProjectionNodeAccumulator<'a> {
    fn new(
        normal_center: &'a Array1<f64>,
        projection_lift: &'a Array1<f64>,
        ambient_mean: f64,
    ) -> Self {
        Self {
            moments: OrthantAccumulator::new(normal_center.len()),
            normal_center,
            projection_lift,
            ambient_mean,
            nodes: Vec::new(),
        }
    }

    fn normalized_nodes(self) -> Result<Vec<WeightedProjectionNode>, String> {
        let max_log_weight = self
            .nodes
            .iter()
            .map(|(_, log_weight)| *log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_log_weight.is_finite() {
            return Err(
                "orthant projection cubature accumulated no finite node weight".to_string(),
            );
        }
        let weight_sum = self
            .nodes
            .iter()
            .map(|(_, log_weight)| (*log_weight - max_log_weight).exp())
            .sum::<f64>();
        if !(weight_sum.is_finite() && weight_sum > 0.0) {
            return Err(format!(
                "orthant projection cubature has invalid normalized weight sum {weight_sum:?}"
            ));
        }
        Ok(self
            .nodes
            .into_iter()
            .map(|(conditional_mean, log_weight)| WeightedProjectionNode {
                conditional_mean,
                weight: (log_weight - max_log_weight).exp() / weight_sum,
            })
            .collect())
    }
}

impl OrthantNodeSink for ProjectionNodeAccumulator<'_> {
    fn push(&mut self, log_weight: f64, point: &Array1<f64>) {
        self.moments.push(log_weight, point);
        let conditional_mean = self.ambient_mean
            + self
                .projection_lift
                .iter()
                .zip(point.iter().zip(self.normal_center.iter()))
                .map(|(&lift, (&value, &center))| lift * (value - center))
                .sum::<f64>();
        self.nodes.push((conditional_mean, log_weight));
    }
}

fn converged_projection_nodes(
    mean: &Array1<f64>,
    covariance: &Array2<f64>,
    upper: &[f64],
    projection_lift: &Array1<f64>,
    ambient_mean: f64,
) -> Result<Vec<WeightedProjectionNode>, String> {
    let q = mean.len();
    if covariance.dim() != (q, q) || projection_lift.len() != q || upper.len() != q {
        return Err(format!(
            "truncated projection geometry mismatch: mean={q}, covariance={:?}, lift={}, \
             upper limits={}",
            covariance.dim(),
            projection_lift.len(),
            upper.len()
        ));
    }
    let factor = gam_linalg::triangular::cholesky_factor_in_place(
        covariance.view(),
        gam_linalg::triangular::CholeskyGuard::FiniteStrict,
    )
    .ok_or_else(|| {
        "orthant projection: the constraint-normal covariance is not numerically positive definite"
            .to_string()
    })?;
    let generator = kronecker_generator(q);
    let tilt = minimax_tilt(mean, upper, factor.view());
    let mut accumulator = ProjectionNodeAccumulator::new(mean, projection_lift, ambient_mean);
    let mut evaluated = 0usize;
    let mut previous: Option<(Array1<f64>, Array2<f64>)> = None;
    loop {
        let target = if evaluated == 0 {
            ORTHANT_MOMENT_INITIAL_POINTS
        } else {
            evaluated * 2
        };
        accumulate_orthant_nodes(
            &mut accumulator,
            mean,
            upper,
            None,
            factor.view(),
            &generator,
            tilt.as_ref(),
            0,
            evaluated,
            target,
        )?;
        evaluated = target;
        let current = accumulator.moments.moments()?;
        if let Some(ref last) = previous
            && moment_relative_change(last, &current, covariance)
                <= ORTHANT_MOMENT_RELATIVE_TOLERANCE
        {
            return accumulator.normalized_nodes();
        }
        if evaluated >= ORTHANT_MOMENT_MAXIMUM_POINTS {
            let change = previous
                .as_ref()
                .map(|last| moment_relative_change(last, &current, covariance))
                .unwrap_or(f64::INFINITY);
            return Err(format!(
                "orthant projection for a {q}-dimensional constraint face did not converge: \
                 relative moment change {change:.3e} still exceeds \
                 {ORTHANT_MOMENT_RELATIVE_TOLERANCE:.1e} at {evaluated} cubature nodes"
            ));
        }
        previous = Some(current);
    }
}

fn projection_quantile(
    nodes: &[WeightedProjectionNode],
    residual_variance: f64,
    probability: f64,
    posterior_mean: f64,
    ambient_sd: f64,
) -> Result<f64, String> {
    if nodes.is_empty() {
        return Err("orthant projection quantile received no cubature nodes".to_string());
    }
    if residual_variance == 0.0 {
        let mut ordered = nodes.to_vec();
        ordered.sort_by(|left, right| left.conditional_mean.total_cmp(&right.conditional_mean));
        let mut cumulative = 0.0;
        for node in &ordered {
            cumulative += node.weight;
            if cumulative >= probability {
                return Ok(node.conditional_mean);
            }
        }
        return Ok(ordered
            .last()
            .expect("non-empty projection node set")
            .conditional_mean);
    }

    let residual_sd = residual_variance.sqrt();
    let cdf = |value: f64| {
        nodes
            .iter()
            .map(|node| {
                node.weight * normal_cdf((value - node.conditional_mean) / residual_sd)
            })
            .sum::<f64>()
    };
    let mut step = ambient_sd.max(residual_sd).max(f64::MIN_POSITIVE);
    let mut lower = posterior_mean - step;
    let mut upper = posterior_mean + step;
    while cdf(lower) > probability {
        step *= 2.0;
        lower = posterior_mean - step;
        if !lower.is_finite() {
            return Err(format!(
                "orthant projection quantile could not bracket lower probability {probability}"
            ));
        }
    }
    step = ambient_sd.max(residual_sd).max(f64::MIN_POSITIVE);
    while cdf(upper) < probability {
        step *= 2.0;
        upper = posterior_mean + step;
        if !upper.is_finite() {
            return Err(format!(
                "orthant projection quantile could not bracket upper probability {probability}"
            ));
        }
    }

    let resolution = f64::EPSILON.sqrt() * ambient_sd.max(residual_sd);
    loop {
        let midpoint = lower + 0.5 * (upper - lower);
        if midpoint == lower || midpoint == upper || upper - lower <= resolution {
            return Ok(midpoint);
        }
        if cdf(midpoint) < probability {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
}

/// `log(1 − exp(d))` for `d ≤ 0`, evaluated on whichever of the two branches
/// keeps the cancellation out of the result.
///
/// `d` is always the log of the fraction of a half-line's mass that an upper
/// limit removes, so `d = −∞` is the unbounded coordinate and returns exactly
/// `0`. That exactness is what makes an infinite upper limit reproduce the
/// half-line arithmetic bit for bit rather than merely closely.
fn log1mexp(d: f64) -> f64 {
    if d == f64::NEG_INFINITY {
        return 0.0;
    }
    if d >= 0.0 {
        return f64::NEG_INFINITY;
    }
    if d > -std::f64::consts::LN_2 {
        (-d.exp_m1()).ln()
    } else {
        (-d.exp()).ln_1p()
    }
}

/// Closed-form moments of `N(mean, variance)` restricted to `[0, upper]`.
///
/// `upper = f64::INFINITY` is the half-line, and takes the inverse-Mills branch
/// unchanged — a one-sided bound is not routed through the two-sided formula and
/// then hoped to agree with itself.
fn scalar_truncated_moments(
    mean: f64,
    variance: f64,
    upper: f64,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!(
            "scalar truncated moments need a positive finite variance, got {variance:?}"
        ));
    }
    let sd = variance.sqrt();
    // `alpha` is the truncation point in standardized units; the feasible half
    // is `z ≥ alpha`, and `mills` is the inverse Mills ratio `φ(α)/Φ̄(α)`
    // obtained on the numerically stable `Φ` branch by reflection.
    let alpha = -mean / sd;
    if !upper.is_finite() {
        let mills = signed_probit_logcdf_and_mills_ratio(-alpha).1;
        if !(mills.is_finite() && mills >= 0.0) {
            return Err(format!(
                "scalar truncated moments: inverse Mills ratio at {alpha} is {mills:?}"
            ));
        }
        let truncated_mean = mean + sd * mills;
        let truncated_variance = variance * (1.0 + alpha * mills - mills * mills);
        if !(truncated_variance.is_finite() && truncated_variance >= 0.0) {
            return Err(format!(
                "scalar truncated moments produced variance {truncated_variance:?} at \
                 standardized truncation point {alpha}"
            ));
        }
        return Ok((
            Array1::from_elem(1, truncated_mean),
            Array2::from_elem((1, 1), truncated_variance),
        ));
    }
    if !(upper > 0.0) {
        return Err(format!(
            "scalar truncated moments need the upper limit above the wall, got {upper:?}"
        ));
    }
    let beta = (upper - mean) / sd;
    // Reflect so the retained interval sits in the upper tail: every difference
    // below is then between two directly-evaluated tail quantities instead of
    // between two numbers within rounding of one.
    let reflect = alpha + beta < 0.0;
    let (low, high, centre) = if reflect {
        (-beta, -alpha, -mean)
    } else {
        (alpha, beta, mean)
    };
    let log_tail_low = normal_logsf(low);
    let log_tail_high = normal_logsf(high);
    let log_mass = log_tail_low + log1mexp(log_tail_high - log_tail_low);
    if !log_mass.is_finite() {
        return Err(format!(
            "scalar truncated moments: the interval [0, {upper:.6e}] around mean {mean:.6e} \
             with standard deviation {sd:.6e} carries no representable mass"
        ));
    }
    // `φ(high)/φ(low) = exp(½(low² − high²))`, factored as a difference of
    // squares so the exponent is not the cancellation of two large numbers. The
    // reflection above makes it non-positive.
    let log_density_ratio = 0.5 * (low - high) * (low + high);
    let density_ratio = log_density_ratio.exp();
    let scale = (-0.5 * low * low - 0.5 * (2.0 * std::f64::consts::PI).ln() - log_mass).exp();
    let first = scale * -log_density_ratio.exp_m1();
    let second = scale * (low - high * density_ratio);
    let truncated_mean = centre + sd * first;
    let truncated_variance = variance * (1.0 + second - first * first);
    if !(truncated_variance.is_finite() && truncated_variance >= 0.0) {
        return Err(format!(
            "scalar truncated moments produced variance {truncated_variance:?} on the \
             standardized interval [{low}, {high}]"
        ));
    }
    Ok((
        Array1::from_elem(1, if reflect { -truncated_mean } else { truncated_mean }),
        Array2::from_elem((1, 1), truncated_variance),
    ))
}

/// Largest relative moment change between two cubature passes, measured in the
/// pre-truncation scale `sd_i = sqrt(W_ii)` so the criterion does not depend on
/// how the constraint rows happen to be scaled.
fn moment_relative_change(
    previous: &(Array1<f64>, Array2<f64>),
    current: &(Array1<f64>, Array2<f64>),
    w: &Array2<f64>,
) -> f64 {
    let q = current.0.len();
    let mut worst = 0.0f64;
    for i in 0..q {
        let sd_i = w[[i, i]].sqrt();
        worst = worst.max((current.0[i] - previous.0[i]).abs() / sd_i);
        for j in 0..q {
            let sd_j = w[[j, j]].sqrt();
            worst =
                worst.max((current.1[[i, j]] - previous.1[[i, j]]).abs() / (sd_i * sd_j));
        }
    }
    worst
}

/// Kronecker (Richtmyer) lattice generator `α_i = frac(√p_i)` over the primes.
/// Deterministic and table-free: the sequence is reproduced from the primes
/// themselves, so the reported covariance does not depend on a stored vector of
/// magic direction numbers or on any random seed.
fn kronecker_generator(dimension: usize) -> Vec<f64> {
    let mut generator = Vec::with_capacity(dimension);
    let mut candidate = 2u64;
    while generator.len() < dimension {
        if is_prime(candidate) {
            let root = (candidate as f64).sqrt();
            generator.push(root - root.floor());
        }
        candidate += 1;
    }
    generator
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    let mut divisor = 2u64;
    while divisor * divisor <= value {
        if value % divisor == 0 {
            return false;
        }
        divisor += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Independent reference: Simpson quadrature of `N(mean, variance)`
    /// restricted to `[0, ∞)`, with the density rescaled by its value at the
    /// truncation point so a deeply pinned centre does not underflow.
    fn quadrature_truncated_moments(mean: f64, variance: f64) -> (f64, f64) {
        let sd = variance.sqrt();
        let alpha = -mean / sd;
        let panels = 400_000usize;
        let upper = alpha + 60.0;
        let step = (upper - alpha) / panels as f64;
        let mut mass = 0.0f64;
        let mut first = 0.0f64;
        let mut second = 0.0f64;
        for index in 0..=panels {
            let z = alpha + step * index as f64;
            let simpson = if index == 0 || index == panels {
                1.0
            } else if index % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let density = (-(z * z - alpha * alpha) / 2.0).exp();
            mass += simpson * density;
            first += simpson * density * z;
            second += simpson * density * z * z;
        }
        let m1 = first / mass;
        let m2 = second / mass;
        (mean + sd * m1, variance * (m2 - m1 * m1))
    }

    /// The scalar closed form against the textbook truncated-normal moments at
    /// the three regimes the estimand argument turns on.
    #[test]
    fn scalar_truncated_moments_match_the_closed_form_at_every_regime() {
        // Mode exactly on the bound: half-normal, variance (1 - 2/pi) sigma^2.
        let (mean, variance) = scalar_truncated_moments(0.0, 1.0, f64::INFINITY).expect("half normal");
        let expected_mean = (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (mean[0] - expected_mean).abs() < 1e-12,
            "half-normal mean {} vs {expected_mean}",
            mean[0]
        );
        let expected_variance = 1.0 - 2.0 / std::f64::consts::PI;
        assert!(
            (variance[[0, 0]] - expected_variance).abs() < 1e-12,
            "half-normal variance {} vs {expected_variance}",
            variance[[0, 0]]
        );
        assert!(
            variance[[0, 0]] > 0.36 && variance[[0, 0]] < 0.37,
            "a coefficient whose mode sits exactly on its bound keeps a THIRD of its \
             unconstrained variance, not zero: got {}",
            variance[[0, 0]]
        );

        // Strongly pinned: checked against an INDEPENDENT quadrature of the
        // truncated density rather than against an asymptotic, because the
        // leading `sigma^2/alpha^2` term carries an O(alpha^-4) deficit that a
        // tolerance would have to absorb.
        for center in [-2.0, -4.0, -8.0] {
            let (deep_mean, deep) = scalar_truncated_moments(center, 1.0, f64::INFINITY).expect("deep tail");
            let (reference_mean, reference_variance) = quadrature_truncated_moments(center, 1.0);
            assert!(
                (deep_mean[0] - reference_mean).abs() < 1e-9 * reference_mean.abs().max(1.0),
                "closed-form mean {} vs quadrature {reference_mean} at centre {center}",
                deep_mean[0]
            );
            assert!(
                (deep[[0, 0]] / reference_variance - 1.0).abs() < 1e-8,
                "closed-form variance {} vs quadrature {reference_variance} at centre {center}",
                deep[[0, 0]]
            );
            assert!(
                deep[[0, 0]] > 0.0,
                "a finite multiplier never gives zero variance, got {} at centre {center}",
                deep[[0, 0]]
            );
        }
        // ...and it does head to zero like sigma^2/alpha^2, which is the ONLY
        // limit in which the active-face answer becomes correct.
        let (_, at_eight) = scalar_truncated_moments(-8.0, 1.0, f64::INFINITY).expect("deep tail");
        assert!(
            at_eight[[0, 0]] * 64.0 > 0.9 && at_eight[[0, 0]] * 64.0 < 1.0,
            "variance times alpha^2 should approach one from below, got {}",
            at_eight[[0, 0]] * 64.0
        );

        // Constraint far away: the moments relax back to the untruncated ones,
        // but only to the order of the tail mass the constraint still removes —
        // a bound five standard deviations below the centre still moves the mean
        // by `sd·φ(5)/Φ(5) ≈ 3e-6`, which is exactly the smooth dependence on
        // slack that makes a tightness predicate unnecessary.
        let (far_mean, far_variance) = scalar_truncated_moments(10.0, 4.0, f64::INFINITY).expect("inactive");
        let (reference_mean, reference_variance) = quadrature_truncated_moments(10.0, 4.0);
        assert!(
            (far_mean[0] - reference_mean).abs() < 1e-9,
            "inactive-bound mean {} vs quadrature {reference_mean}",
            far_mean[0]
        );
        assert!(
            (far_variance[[0, 0]] - reference_variance).abs() < 1e-9,
            "inactive-bound variance {} vs quadrature {reference_variance}",
            far_variance[[0, 0]]
        );
        assert!(
            (far_mean[0] - 10.0).abs() < 1e-5 && far_mean[0] > 10.0,
            "a bound five sd away moves the mean by the tail mass and no more, got {}",
            far_mean[0]
        );
        assert!(
            (far_variance[[0, 0]] - 4.0).abs() < 1e-4 && far_variance[[0, 0]] < 4.0,
            "a bound five sd away shrinks the variance by the tail mass and no more, got {}",
            far_variance[[0, 0]]
        );
    }

    #[test]
    fn equal_tailed_projection_interval_is_asymmetric_for_a_half_normal() {
        let covariance = array![[1.0]];
        let center = array![0.0];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0]], array![0.0]).expect("constraint");
        let correction =
            constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("correction")
                .expect("active half-space");
        let geometry = ConstrainedPosteriorGeometry {
            constraints,
            mode: array![0.0],
            unconstrained_center: Some(center),
            correction: Some(correction),
            moment_status: ConstrainedPosteriorMomentStatus::Available,
        };
        let (lower, upper) = constrained_projection_equal_tailed_interval(
            &covariance,
            &geometry,
            &array![1.0],
            0.95,
        )
        .expect("equal-tailed interval");

        // For Z | Z>=0, F(z)=2 Phi(z)-1. The equal-tailed endpoints are
        // Phi^-1((1+p)/2), p in {0.025, 0.975}.
        let expected_lower = standard_normal_quantile(0.5125).expect("lower quantile");
        let expected_upper = standard_normal_quantile(0.9875).expect("upper quantile");
        assert!(
            (lower - expected_lower).abs() < 2e-3,
            "half-normal lower endpoint {lower} vs {expected_lower}"
        );
        assert!(
            (upper - expected_upper).abs() < 2e-3,
            "half-normal upper endpoint {upper} vs {expected_upper}"
        );
        let posterior_mean = (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (posterior_mean - lower) < (upper - posterior_mean),
            "the exact skew interval must not collapse back to mean +/- z*sd"
        );
    }

    #[test]
    fn equal_tailed_projection_sweep_has_exact_mass_and_repairs_the_short_symmetric_band() {
        let covariance = array![[1.0]];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0]], array![0.0]).expect("constraint");
        let alpha = 0.025;
        let ambient_width =
            2.0 * standard_normal_quantile(1.0 - alpha).expect("ambient quantile");
        let mut saw_repaired_short_symmetric_band = false;

        for center_value in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let center = array![center_value];
            let correction = constrained_posterior_correction_from_covariance(
                &covariance,
                &center,
                &constraints,
            )
            .expect("correction")
            .expect("finite lower truncation");
            let posterior_variance =
                1.0 - correction.removed_variance_diagonal()[0];
            let geometry = ConstrainedPosteriorGeometry {
                constraints: constraints.clone(),
                mode: array![center_value.max(0.0)],
                unconstrained_center: Some(center),
                correction: Some(correction),
                moment_status: ConstrainedPosteriorMomentStatus::Available,
            };
            let (lower, upper) = constrained_projection_equal_tailed_interval(
                &covariance,
                &geometry,
                &array![1.0],
                0.95,
            )
            .expect("equal-tailed interval");

            let mass_below_bound = normal_cdf(-center_value);
            let retained_mass = 1.0 - mass_below_bound;
            let truncated_cdf = |value: f64| {
                (normal_cdf(value - center_value) - mass_below_bound) / retained_mass
            };
            assert!(
                (truncated_cdf(lower) - alpha).abs() < 2e-8
                    && (truncated_cdf(upper) - (1.0 - alpha)).abs() < 2e-8,
                "centre {center_value}: endpoints [{lower}, {upper}] do not enclose exact \
                 posterior mass 0.95"
            );
            assert!(
                lower >= 0.0,
                "centre {center_value}: lower endpoint {lower} escaped the saved cone"
            );
            assert!(
                upper - lower <= ambient_width + 1e-10,
                "centre {center_value}: truncation widened [{lower}, {upper}] beyond the \
                 ambient Gaussian interval"
            );

            if center_value == 3.0 {
                let symmetric_width = 2.0
                    * standard_normal_quantile(1.0 - alpha).expect("symmetric quantile")
                    * posterior_variance.sqrt();
                assert!(
                    upper - lower > symmetric_width,
                    "the exact 3-SE interval must repair the moment-matched symmetric interval's \
                     short, under-covering band: exact width {}, symmetric width {symmetric_width}",
                    upper - lower
                );
                saw_repaired_short_symmetric_band = true;
            }
        }

        assert!(
            saw_repaired_short_symmetric_band,
            "the sweep must include its 3-SE regression cell"
        );
    }

    /// A constraint row whose normal is nearly a combination of the accepted
    /// ones must be dropped — and the reason matters. It is not dropped because
    /// it is undetectable: its pivot sits two hundred times above the bare
    /// `ε·diagonal` limit at which an exactly dependent row stops being
    /// distinguishable. It is dropped because retaining it reports a lift
    /// `G = Σ Aᵀ W⁻¹` whose error exceeds the accuracy this module certifies its
    /// own moments to, and a wrong lift is worse than a missing row that imposes
    /// nothing the retained one does not already impose.
    ///
    /// Both arms are asserted so the gate discriminates: a filter that never
    /// drops fails the near-degenerate arm, one that always drops fails the
    /// resolvable arm.
    #[test]
    fn a_constraint_row_below_the_lift_accuracy_floor_is_dropped_though_detectable() {
        let identity = Array2::<f64>::eye(4);
        let center = Array1::<f64>::zeros(4);

        let mut resolvable = Array2::<f64>::zeros((3, 4));
        resolvable[[0, 0]] = 1.0;
        resolvable[[1, 1]] = 1.0;
        resolvable[[2, 2]] = 1.0;
        let constraints = LinearInequalityConstraints::new(resolvable, Array1::<f64>::zeros(3))
            .expect("orthogonal constraint rows");
        let correction =
            constrained_posterior_correction_from_covariance(&identity, &center, &constraints)
                .expect("orthogonal face")
                .expect("an active face at zero slack");
        assert_eq!(
            correction.rows,
            vec![0, 1, 2],
            "three mutually independent constraint normals must all be retained"
        );

        // Row 1 is row 0 rotated by `sine` in the `Σ` metric, so its pivot is
        // exactly `sine²` against a diagonal of `1 + sine²`.
        let sine = 3.0e-7;
        let pivot = sine * sine;
        let diagonal = 1.0 + pivot;
        let detectability_limit = 2.0 * f64::EPSILON * diagonal;
        assert!(
            pivot > detectability_limit,
            "the fixture must be DETECTABLE, or the drop below proves nothing: pivot \
             {pivot:e} against the bare rank limit {detectability_limit:e}"
        );
        assert!(
            pivot < detectability_limit / ORTHANT_MOMENT_RELATIVE_TOLERANCE,
            "the fixture must sit below the accuracy the first pass demands"
        );

        let mut degenerate = Array2::<f64>::zeros((3, 4));
        degenerate[[0, 0]] = 1.0;
        degenerate[[1, 0]] = 1.0;
        degenerate[[1, 1]] = sine;
        degenerate[[2, 2]] = 1.0;
        let constraints = LinearInequalityConstraints::new(degenerate, Array1::<f64>::zeros(3))
            .expect("near-parallel constraint rows");
        let correction =
            constrained_posterior_correction_from_covariance(&identity, &center, &constraints)
                .expect("near-degenerate face")
                .expect("an active face at zero slack");
        assert_eq!(
            correction.rows,
            vec![0, 2],
            "the near-parallel row must be dropped: retaining it reports a lift whose own \
             defining identity A·G = I fails by more than the certified accuracy"
        );
    }

    /// The retention floor is necessary and NOT sufficient, so the assembled
    /// face has to be checked and the floor raised until it delivers. Because
    /// the filter walks candidates in slack order rather than in pivot order —
    /// a statistical choice, since two near-parallel rows with different offsets
    /// are not the same constraint and the tighter one dominates — its per-row
    /// pivots do not reveal the assembled face's conditioning. Every pivot can
    /// clear the floor while the face as a whole does not.
    ///
    /// A Vandermonde face in clustered nodes is the sharp case: seven rows whose
    /// effective rank is five, all well inside the slack horizon so nothing is
    /// dropped for statistical reasons. Measured `max|A G − I|` on the face this
    /// fixture produces:
    ///
    /// * bare detectability floor, no check — `3.26e-1`, 326× the accuracy this
    ///   module reports its moments to;
    /// * one pass at the derived floor — `1.21e-1`, still 121× over;
    /// * the floor raised by the amount it missed by — `3.53e-5`, inside, in two
    ///   passes.
    ///
    /// So this gate fails on the shipped filter, fails on a single-pass floor
    /// change, and passes only when the realized lift governs the retained face.
    #[test]
    fn the_retained_face_satisfies_the_identity_that_defines_its_lift() {
        const ROWS: usize = 7;
        const DIMENSION: usize = 8;
        const DEGREE: usize = 5;
        const SPACING: f64 = 1.0e-2;

        let mut a = Array2::<f64>::zeros((ROWS, DIMENSION));
        for row in 0..ROWS {
            let node = row as f64 * SPACING;
            for power in 0..DEGREE {
                a[[row, power]] = node.powi(power as i32);
            }
        }
        let constraints = LinearInequalityConstraints::new(a.clone(), Array1::<f64>::zeros(ROWS))
            .expect("clustered Vandermonde rows");

        // Place the centre so every row sits at ~7 standardized units of slack:
        // inside the resolution horizon, so each row is a genuine candidate and
        // nothing is dropped for being statistically irrelevant, while the
        // truncation itself is nearly invisible — which keeps this gate a
        // statement about the lift rather than about the cubature.
        let covariance = Array2::<f64>::eye(DIMENSION);
        let mut center = Array1::<f64>::zeros(DIMENSION);
        center[0] = 7.0;
        for row in 0..ROWS {
            let normal = a.row(row);
            let slack = normal.dot(&center) / normal.dot(&normal).sqrt();
            assert!(
                slack < 8.12 && slack > 6.0,
                "row {row} must be a candidate inside the resolution horizon, got slack {slack}"
            );
        }

        let correction =
            constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("clustered Vandermonde face")
                .expect("an active face inside the horizon");

        assert!(
            correction.rows.len() < ROWS,
            "the fixture must exercise the filter: all {ROWS} rows were retained"
        );
        assert!(
            correction.rows.len() >= 2,
            "the face must not collapse to a single row, or the identity below is vacuous: \
             retained {:?}",
            correction.rows
        );

        let mut departure = 0.0_f64;
        for (i, &row_index) in correction.rows.iter().enumerate() {
            for j in 0..correction.rows.len() {
                let entry = a.row(row_index).dot(&correction.lift.column(j));
                let target = if i == j { 1.0 } else { 0.0 };
                departure = departure.max((entry - target).abs());
            }
        }
        assert!(
            departure <= ORTHANT_MOMENT_RELATIVE_TOLERANCE,
            "the reported lift must satisfy A·G = I, the identity it is defined by, to the \
             accuracy this module certifies its moments to: max|A G - I| = {departure:e} on \
             the retained rows {:?}",
            correction.rows
        );
    }

    /// The retained-face walk and its oracle, sharing one face assembler.
    ///
    /// `excluded` is the set of constraint-row indices held out of the greedy
    /// walk. Returns the retained rows and whether their lift satisfies the
    /// identity that defines it, or `None` when the exclusion leaves no face.
    fn face_at_exclusion(
        candidates: &[(usize, f64, Array1<f64>)],
        constraints: &LinearInequalityConstraints,
        center: &Array1<f64>,
        excluded: &[usize],
    ) -> Option<(Vec<usize>, bool)> {
        let face = assemble_retained_face(
            candidates,
            ORTHANT_MOMENT_RELATIVE_TOLERANCE,
            constraints,
            center,
            excluded,
        )
        .expect("face assembly")?;
        let lift = cholesky_solve_right(&face.factor, &face.sigma_at).expect("lift solve");
        let departure =
            lift_identity_departure(&lift, constraints, &face.rows).expect("identity departure");
        Some((face.rows, departure <= ORTHANT_MOMENT_RELATIVE_TOLERANCE))
    }

    /// #2714: the walk must return the LARGEST admissible face, and it must
    /// reach it by a rule that terminates.
    ///
    /// Two earlier rules searched a REAL NUMBER — the retention floor — where
    /// the object being chosen is a SET of rows, and both failed on that:
    ///
    /// * `demanded_accuracy /= departure/tolerance` stepped by a factor read off
    ///   a per-row error model the filter itself documents as not tight, so it
    ///   skipped faces and could leave the loop in one pass;
    /// * stepping to `max_r (k+1)·ε·diagonal_r/pivot_r` is exact in real
    ///   arithmetic and not in floating point — see
    ///   `the_floor_round_trip_retains_the_row_it_was_aimed_at_2714`, which
    ///   measures the round trip directly.
    ///
    /// The oracle here is brute force over EVERY exclusion set, which is the
    /// full family the walk searches (128 of them at `ROWS = 7`), rather than
    /// the floor-indexed subfamily the previous version of this test swept.
    /// Both sides use the SAME `constraint_face_candidates` /
    /// `assemble_retained_face`, so this compares search strategies over one
    /// face family rather than two implementations of the face.
    #[test]
    fn the_walk_returns_the_largest_admissible_face_2714() {
        const ROWS: usize = 7;
        const DIMENSION: usize = 8;
        const DEGREE: usize = 5;
        const SPACING: f64 = 1.0e-2;

        let mut a = Array2::<f64>::zeros((ROWS, DIMENSION));
        for row in 0..ROWS {
            let node = row as f64 * SPACING;
            for power in 0..DEGREE {
                a[[row, power]] = node.powi(power as i32);
            }
        }
        let constraints = LinearInequalityConstraints::new(a.clone(), Array1::<f64>::zeros(ROWS))
            .expect("clustered Vandermonde rows");
        let covariance = Array2::<f64>::eye(DIMENSION);
        let mut center = Array1::<f64>::zeros(DIMENSION);
        center[0] = 7.0;

        let correction =
            constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("clustered Vandermonde face")
                .expect("an active face inside the horizon");

        let sigma_at = covariance.dot(&constraints.a.t());
        let candidates = constraint_face_candidates(sigma_at.view(), &center, &constraints)
            .expect("candidate rows");
        let candidate_rows: Vec<usize> = candidates.iter().map(|(row, _, _)| *row).collect();

        let mut admissible_faces: Vec<Vec<usize>> = Vec::new();
        let mut distinct_faces: Vec<Vec<usize>> = Vec::new();
        for mask in 0..(1u32 << candidate_rows.len()) {
            let excluded: Vec<usize> = candidate_rows
                .iter()
                .enumerate()
                .filter(|(position, _)| mask & (1 << position) != 0)
                .map(|(_, row)| *row)
                .collect();
            let Some((rows, admissible)) =
                face_at_exclusion(&candidates, &constraints, &center, &excluded)
            else {
                continue;
            };
            if !distinct_faces.contains(&rows) {
                distinct_faces.push(rows.clone());
            }
            if admissible && !admissible_faces.contains(&rows) {
                admissible_faces.push(rows);
            }
        }
        let largest = admissible_faces
            .iter()
            .map(Vec::len)
            .max()
            .expect("some exclusion set must yield a face satisfying its own identity");

        // Non-vacuity: the fixture has to make the walk WORK. If the unexcluded
        // face were already admissible, or if only one face existed, this test
        // would pass on a walk that never ran.
        assert!(
            distinct_faces.len() >= 3,
            "#2714: the exclusion sweep saw only {} distinct face(s), so the fixture does not \
             exercise a walk: {distinct_faces:?}",
            distinct_faces.len()
        );
        let (unexcluded, unexcluded_admissible) =
            face_at_exclusion(&candidates, &constraints, &center, &[])
                .expect("the unexcluded face");
        assert!(
            !unexcluded_admissible,
            "#2714: the unexcluded face {unexcluded:?} already satisfies its own lift identity, \
             so the walk is not exercised"
        );
        assert!(
            unexcluded.len() > largest,
            "#2714: the unexcluded face {unexcluded:?} is no larger than the {largest}-row \
             answer, so nothing had to be dropped"
        );

        // The walk's face must BE one of the admissible faces — this is what
        // says it returned a face it can actually lift — and it must be one of
        // the largest, which is what says it stopped dropping as soon as it
        // could. Asserting membership plus size rather than equality with one
        // enumerated face is deliberate: several distinct faces reach the
        // maximum, so an equality would be pinning the oracle's enumeration
        // order, not a property of the walk.
        assert!(
            admissible_faces.contains(&correction.rows),
            "#2714: the walk returned {:?}, which is not among the {} faces whose lift satisfies \
             its own identity: {admissible_faces:?}",
            correction.rows,
            admissible_faces.len()
        );
        assert_eq!(
            correction.rows.len(),
            largest,
            "#2714: the walk returned the {}-row face {:?} where an admissible face of {largest} \
             rows exists. Dropping the least independent accepted row is the step that reaches \
             the largest one; stepping a retention floor cannot, because the floor is a proxy \
             for the face and the proxy is not injective.",
            correction.rows.len(),
            correction.rows
        );
        // And among the largest, the one the SLACK ordering asks for: the walk
        // opens on the tightest candidate row and only ever drops rows that are
        // nearly dependent on rows already accepted, so the binding wall cannot
        // be the row that leaves. An enumeration over exclusion sets has no
        // such preference — on this fixture it also reaches a maximum-size face
        // that drops the tightest row — so this is the assertion that separates
        // the walk from "any largest face".
        assert_eq!(
            correction.rows.first(),
            candidate_rows.first(),
            "#2714: the walk returned {:?}, which does not retain the tightest candidate row \
             {:?}. The slack ordering is the reason a dropped row imposes no constraint the \
             retained ones do not; dropping the binding wall would relax the posterior by a \
             multiple of its own standard deviation.",
            correction.rows,
            candidate_rows.first()
        );
    }

    /// The production entry point on the geometry #2714's witness actually has:
    /// far MORE constraint rows than the coefficient block has columns, so
    /// `W = A Σ Aᵀ` is structurally rank-deficient and the retention walk is the
    /// only thing between the fit and a face it can lift.
    ///
    /// A shape-constrained survival fit imposes its monotonicity guard at every
    /// observed exit time — one constraint row per data row, on a time block of
    /// a few coefficients — and the rows are near-collinear because adjacent
    /// times give near-identical derivative-basis rows. This fixture is that
    /// shape with the data removed: 40 Vandermonde rows at closely spaced nodes
    /// on 5 columns, identity covariance, every row within the resolution
    /// horizon of its wall. Nothing here is a recorded spectrum; the geometry
    /// is written down and the numbers follow from it.
    ///
    /// The old retention ladder PANICS on this input. Its step
    /// `d ← max_r (k+1)·ε·diagonal_r/pivot_r` retains the row it was aimed at
    /// on pass 26, rebuilds a bit-identical face, recomputes the same step, and
    /// trips `assert!(next < demanded_accuracy)` — which is why this test's
    /// primary assertion is that the call RETURNS. `constrained_posterior_correction`
    /// is reached from the terminal geometry of every constrained fit, so a
    /// panic there is a panic in a library, on data.
    #[test]
    fn a_rank_deficient_constraint_system_still_yields_a_liftable_face_2714() {
        const ROWS: usize = 40;
        const DIMENSION: usize = 5;
        const SPACING: f64 = 2.0e-2;

        let mut a = Array2::<f64>::zeros((ROWS, DIMENSION));
        for row in 0..ROWS {
            let node = row as f64 * SPACING;
            for power in 0..DIMENSION {
                a[[row, power]] = node.powi(power as i32);
            }
        }
        let constraints = LinearInequalityConstraints::new(a, Array1::<f64>::zeros(ROWS))
            .expect("clustered Vandermonde rows");
        let covariance = Array2::<f64>::eye(DIMENSION);
        // Every row's value at the centre is its constant term, so every row
        // sits one unit above its wall in unscaled units and inside the
        // resolution horizon once divided by its own spread: all 40 are
        // candidates, against 5 columns.
        let mut center = Array1::<f64>::zeros(DIMENSION);
        center[0] = 1.0;

        let sigma_at = covariance.dot(&constraints.a.t());
        let candidates = constraint_face_candidates(sigma_at.view(), &center, &constraints)
            .expect("candidate rows");
        assert!(
            candidates.len() > DIMENSION,
            "#2714: the fixture must be rank-deficient to exercise the walk, and it offers only \
             {} candidate row(s) against {DIMENSION} columns",
            candidates.len()
        );
        let (unexcluded, unexcluded_admissible) =
            face_at_exclusion(&candidates, &constraints, &center, &[])
                .expect("the unexcluded face");
        assert!(
            !unexcluded_admissible,
            "#2714: the unexcluded face {unexcluded:?} already satisfies its own lift identity, \
             so the walk never runs and this fixture asserts nothing"
        );

        let correction =
            constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("a rank-deficient constraint system must still produce a face")
                .expect("an active face inside the horizon");

        // The face it settled on has to satisfy the identity that DEFINES the
        // lift, recomputed here from the returned `lift` rather than from the
        // walk's own bookkeeping.
        let departure = lift_identity_departure(&correction.lift, &constraints, &correction.rows)
            .expect("identity departure of the returned lift");
        assert!(
            departure <= ORTHANT_MOMENT_RELATIVE_TOLERANCE,
            "#2714: the returned {}-row face {:?} misses the identity that defines its lift by \
             {departure:.6e}, above {ORTHANT_MOMENT_RELATIVE_TOLERANCE:.1e}",
            correction.rows.len(),
            correction.rows
        );
        assert!(
            correction.rows.len() < unexcluded.len(),
            "#2714: the walk returned {:?}, which is not smaller than the inadmissible \
             unexcluded face {unexcluded:?} — so it accepted a face it had already rejected",
            correction.rows
        );
        assert_eq!(
            correction.rows.first(),
            candidates.first().map(|(row, _, _)| row),
            "#2714: the walk dropped the tightest candidate row"
        );
    }

    /// #2714: the walk drops a DIRECTION, and a two-sided bound is one
    /// direction.
    ///
    /// A row anti-parallel to an accepted one is refused as a direction and
    /// keeps its wall as that row's upper limit (#2523), so a two-sided bound
    /// reaches the moments as one retained row carrying a finite `upper`. If
    /// the retention walk then drops that row for being nearly dependent and
    /// leaves its partner in the candidate pool, the next pass accepts the
    /// PARTNER in its place — with a full half-line and nothing to fold its own
    /// far wall into, because the row that carried it is excluded. The bound
    /// silently becomes one-sided, and on the wrong side: the walk is ordered
    /// by ascending slack, so the row that leaves is the tighter wall.
    ///
    /// The fixture is the rank-deficient geometry with every row given an
    /// opposite face, and the assertion is the invariant rather than one
    /// hand-picked pair: NO retained row may report an infinite upper limit,
    /// because every candidate direction in this system is two-sided.
    #[test]
    fn dropping_a_direction_takes_its_opposite_face_with_it_2714() {
        const NODES: usize = 40;
        const DIMENSION: usize = 5;
        const SPACING: f64 = 2.0e-2;
        // The far wall, in the same units as the near one. Every row's value at
        // the centre is `1`, so `3` leaves a two-unit-wide feasible slab in
        // every constraint-normal direction — wide enough that the region is
        // not near-empty and narrow enough that the far wall is inside the
        // resolution horizon and therefore a live candidate.
        const FAR_WALL: f64 = 3.0;

        let mut a = Array2::<f64>::zeros((2 * NODES, DIMENSION));
        let mut b = Array1::<f64>::zeros(2 * NODES);
        for node in 0..NODES {
            let position = node as f64 * SPACING;
            for power in 0..DIMENSION {
                let entry = position.powi(power as i32);
                a[[node, power]] = entry;
                a[[NODES + node, power]] = -entry;
            }
            b[node] = 0.0;
            b[NODES + node] = -FAR_WALL;
        }
        let constraints =
            LinearInequalityConstraints::new(a, b).expect("two-sided Vandermonde slabs");
        let covariance = Array2::<f64>::eye(DIMENSION);
        let mut center = Array1::<f64>::zeros(DIMENSION);
        center[0] = 1.0;

        let sigma_at = covariance.dot(&constraints.a.t());
        let candidates = constraint_face_candidates(sigma_at.view(), &center, &constraints)
            .expect("candidate rows");
        assert_eq!(
            candidates.len(),
            2 * NODES,
            "#2714: both walls of every slab must be candidates, or the fixture is not \
             two-sided where the walk runs"
        );
        let (unexcluded, unexcluded_admissible) =
            face_at_exclusion(&candidates, &constraints, &center, &[])
                .expect("the unexcluded face");
        assert!(
            !unexcluded_admissible,
            "#2714: the unexcluded face {unexcluded:?} already satisfies its own lift identity, \
             so no direction is ever dropped and this fixture asserts nothing"
        );

        let correction =
            constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("a two-sided rank-deficient system must still produce a face")
                .expect("an active face inside the horizon");

        // Non-vacuity: the folding has to have happened at all.
        assert_eq!(
            correction.normal_upper_limits.len(),
            correction.rows.len(),
            "#2714: one upper limit per retained row"
        );
        for (position, &row) in correction.rows.iter().enumerate() {
            assert!(
                correction.normal_upper_limits[position].is_finite(),
                "#2714: retained row {row} reports an infinite upper limit on a system where \
                 EVERY direction is two-sided. Its opposite face was left in the candidate pool \
                 when the direction that carried the fold was dropped, so a two-sided bound is \
                 being reported as a half-line: rows {:?}, limits {:?}",
                correction.rows,
                correction.normal_upper_limits
            );
        }
        // And the near wall is the one that survived: every retained row is a
        // NEAR wall (index below `NODES`), never the far wall promoted in its
        // place.
        for &row in &correction.rows {
            assert!(
                row < NODES,
                "#2714: the walk retained far-wall row {row}, which sits at slack {FAR_WALL} \
                 against the near wall's 1 — the slacker of the pair replaced the binding one: \
                 {:?}",
                correction.rows
            );
        }
    }

    /// The measurement that kills the retention-floor step (#2714), stated as a
    /// property of `f64` rather than as a story about one fit.
    ///
    /// The previous rule named the next face by the floor `d_r =
    /// (k+1)·ε·diagonal/pivot` at which accepted row `r` drops, because the
    /// retention test `pivot > (k+1)·ε·diagonal/d` then reads `pivot > pivot`.
    /// It does not: both sides are ROUNDED quotients, and the round trip lands
    /// strictly below `pivot` often enough to be reached by any fit. When it
    /// does, the rebuilt face is bit-identical, the recomputed step is the value
    /// the floor already has, and the walk stops descending — which the old code
    /// caught with a debug assertion, i.e. by panicking inside a library.
    ///
    /// The sweep is over the same three quantities the filter forms, across the
    /// magnitudes a penalized posterior actually produces. A single surviving
    /// triple is enough to refute the step rule; the count is reported so a
    /// change in the arithmetic cannot silently make this vacuous.
    #[test]
    fn the_floor_round_trip_retains_the_row_it_was_aimed_at_2714() {
        let mut retained = 0usize;
        let mut exact_stalls = 0usize;
        let mut examined = 0usize;
        for accepted in 0..12usize {
            for diagonal_exponent in -8i32..=4 {
                for pivot_decades in 1..=15i32 {
                    for tweak in 0..64u32 {
                        let diagonal = 10.0_f64.powi(diagonal_exponent)
                            * (1.0 + f64::from(tweak) / 64.0);
                        let pivot = diagonal * 10.0_f64.powi(-pivot_decades);
                        let scale = (accepted + 1) as f64 * f64::EPSILON * diagonal;
                        let step = scale / pivot;
                        let rebuilt_floor = scale / step;
                        examined += 1;
                        if pivot > rebuilt_floor {
                            retained += 1;
                            // The face is then unchanged, so the next step is
                            // recomputed from the same three numbers.
                            if scale / pivot == step {
                                exact_stalls += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            retained > 0,
            "#2714: the floor round trip never retained the row it was aimed at across \
             {examined} triples, which would make this refutation vacuous"
        );
        assert_eq!(
            retained, exact_stalls,
            "#2714: {retained} of {examined} triples retained the row the step was aimed at, and \
             {exact_stalls} of those recompute the same step. Every retention IS a stall — the \
             face is bit-identical, so the step is a function of unchanged inputs — and the old \
             rule's descent assertion fires on each one."
        );
    }

    /// The cubature must reproduce the closed form when the orthant factorizes
    /// into independent coordinates, which is the only multivariate case with
    /// an exact answer to check against. The bound is the module's own
    /// certified accuracy, [`ORTHANT_MOMENT_RELATIVE_TOLERANCE`], measured on
    /// the pre-truncation scale — asserting tighter would assert something the
    /// algorithm does not promise.
    #[test]
    fn cubature_reproduces_independent_coordinates_within_its_certified_accuracy() {
        let mean = array![-0.5, 0.25, -1.5];
        let covariance = array![[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]];
        let factor = gam_linalg::triangular::cholesky_factor_in_place(
            covariance.view(),
            gam_linalg::triangular::CholeskyGuard::FiniteStrict,
        )
        .expect("independent orthant covariance factors");
        let (moment_mean, moment_covariance) =
            box_truncated_moments(&mean, &vec![f64::INFINITY; mean.len()], &covariance, factor.view())
                .expect("independent orthant");
        for i in 0..3 {
            let (exact_mean, exact_variance) =
                scalar_truncated_moments(mean[i], covariance[[i, i]], f64::INFINITY).expect("scalar");
            let scale = covariance[[i, i]].sqrt();
            assert!(
                (moment_mean[i] - exact_mean[0]).abs()
                    < ORTHANT_MOMENT_RELATIVE_TOLERANCE * scale,
                "coordinate {i} mean {} vs exact {}",
                moment_mean[i],
                exact_mean[0]
            );
            assert!(
                (moment_covariance[[i, i]] - exact_variance[[0, 0]]).abs()
                    < ORTHANT_MOMENT_RELATIVE_TOLERANCE * covariance[[i, i]],
                "coordinate {i} variance {} vs exact {}",
                moment_covariance[[i, i]],
                exact_variance[[0, 0]]
            );
            for j in 0..3 {
                if i != j {
                    assert!(
                        moment_covariance[[i, j]].abs()
                            < ORTHANT_MOMENT_RELATIVE_TOLERANCE
                                * scale
                                * covariance[[j, j]].sqrt(),
                        "independent coordinates must stay uncorrelated under an orthant \
                         truncation, got {} at ({i},{j})",
                        moment_covariance[[i, j]]
                    );
                }
            }
        }
    }

    /// The whole point of the module: the correction lands strictly between the
    /// two answers the two fit paths ship today.
    #[test]
    fn correction_lands_strictly_between_full_space_and_active_face() {
        let covariance = array![[1.0, 0.4], [0.4, 1.0]];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0, 0.0]], array![0.0]).expect("cone");
        // Unconstrained centre BELOW the bound: the constrained mode is pinned.
        let center = array![-0.6, 0.3];
        let correction = constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
            .expect("correction")
            .expect("an active row");
        let truncated = correction.apply_to_covariance(&covariance);

        // Active-face answer: the same formula with the normal variance removed
        // in full.
        let mut face = covariance.clone();
        let full_removal = correction.lift.dot(&array![[1.0]]).dot(&correction.lift.t());
        face -= &full_removal;

        assert!(
            truncated[[0, 0]] > face[[0, 0]] + 1e-6,
            "truncated variance {} must exceed the active-face answer {}",
            truncated[[0, 0]],
            face[[0, 0]]
        );
        assert!(
            truncated[[0, 0]] < covariance[[0, 0]] - 1e-6,
            "truncated variance {} must fall below the unconstrained answer {}",
            truncated[[0, 0]],
            covariance[[0, 0]]
        );
        assert!(
            face[[0, 0]].abs() < 1e-12,
            "the active-face answer for a single pinned coordinate is exactly zero, got {}",
            face[[0, 0]]
        );
        assert!(
            correction.normal_mean_shift[0] > 0.0,
            "truncation moves the posterior mean INTO the feasible region, shift was {}",
            correction.normal_mean_shift[0]
        );
    }

    /// A constraint far from the posterior centre must leave the covariance
    /// untouched, so unconstrained-in-practice fits keep their exact bytes.
    #[test]
    fn inactive_constraints_produce_no_correction() {
        let covariance = array![[1.0, 0.0], [0.0, 1.0]];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0, 0.0]], array![0.0]).expect("cone");
        let center = array![40.0, 0.0];
        let correction = constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
            .expect("correction");
        assert!(
            correction.is_none(),
            "a bound 40 posterior standard deviations away cannot move any moment at double \
             precision"
        );
    }

    /// A duplicated constraint row must not make `W` singular.
    #[test]
    fn redundant_rows_are_dropped_by_the_rank_filter() {
        let covariance = array![[1.0, 0.2], [0.2, 1.0]];
        let constraints = LinearInequalityConstraints::new(
            array![[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            array![0.0, 0.0, 0.0],
        )
        .expect("cone");
        let center = array![-0.2, -0.3];
        let correction = constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
            .expect("correction")
            .expect("active rows");
        assert_eq!(
            correction.rows.len(),
            2,
            "the duplicated half-space must be filtered out, kept rows {:?}",
            correction.rows
        );
    }

    /// The correction must never inflate a variance or drive one negative.
    #[test]
    fn corrected_covariance_stays_between_zero_and_the_unconstrained_answer() {
        let covariance = array![
            [1.0, 0.3, 0.1],
            [0.3, 1.2, -0.2],
            [0.1, -0.2, 0.8]
        ];
        let constraints = LinearInequalityConstraints::new(
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            array![0.0, 0.0],
        )
        .expect("cone");
        for center in [
            array![-2.0, -1.0, 0.5],
            array![0.0, 0.0, 0.0],
            array![-0.1, 0.4, -3.0],
        ] {
            let correction = constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("correction")
                .expect("active rows");
            let truncated = correction.apply_to_covariance(&covariance);
            for i in 0..3 {
                assert!(
                    truncated[[i, i]] > 0.0,
                    "coordinate {i} lost all variance at centre {center:?}: {}",
                    truncated[[i, i]]
                );
                assert!(
                    truncated[[i, i]] <= covariance[[i, i]] + 1e-9,
                    "coordinate {i} gained variance at centre {center:?}: {} vs {}",
                    truncated[[i, i]],
                    covariance[[i, i]]
                );
            }
            let diagonal = correction.removed_variance_diagonal();
            for i in 0..3 {
                assert!(
                    (diagonal[i] - (covariance[[i, i]] - truncated[[i, i]])).abs() < 1e-9,
                    "the diagonal-only accessor must agree with the dense correction at {i}"
                );
            }
        }
    }
    // ---------------------------------------------------------------- #2523

    /// The two rows a `linear(min=l, max=u)` term emits: `+e_0ᵀβ ≥ l` and
    /// `−e_0ᵀβ ≥ −u`, exactly as `gam-terms/src/smooth/term_design.rs` builds
    /// them.
    fn two_sided_bound_rows(lower: f64, upper: f64, columns: usize) -> LinearInequalityConstraints {
        let mut a = Array2::<f64>::zeros((2, columns));
        a[[0, 0]] = 1.0;
        a[[1, 0]] = -1.0;
        LinearInequalityConstraints::new(a, array![lower, -upper])
            .expect("two-sided bound rows")
    }

    /// Independent reference: Simpson quadrature of `N(mean, variance)`
    /// restricted to `[0, upper]`, built directly from the density rather than
    /// from any tail function the code under test also uses.
    fn quadrature_box_moments(mean: f64, variance: f64, upper: f64) -> (f64, f64) {
        let sd = variance.sqrt();
        let low = -mean / sd;
        let high = (upper - mean) / sd;
        let reference = if low <= 0.0 && 0.0 <= high {
            0.0
        } else if high < 0.0 {
            high
        } else {
            low
        };
        let panels = 400_000usize;
        let step = (high - low) / panels as f64;
        let (mut mass, mut first, mut second) = (0.0f64, 0.0f64, 0.0f64);
        for index in 0..=panels {
            let z = low + step * index as f64;
            let simpson = if index == 0 || index == panels {
                1.0
            } else if index % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let density = (-(z * z - reference * reference) / 2.0).exp();
            mass += simpson * density;
            first += simpson * density * z;
            second += simpson * density * z * z;
        }
        let m1 = first / mass;
        let m2 = second / mass;
        (mean + sd * m1, variance * (m2 - m1 * m1))
    }

    /// The defect: a row that is anti-parallel to an accepted one carries no new
    /// constraint-normal DIRECTION and is still a constraint. It must survive as
    /// the coordinate's upper limit rather than be dropped as redundant.
    ///
    /// The fixture is the one measured on #2523 — `Σ = I`, `0 ≤ β₀ ≤ 2`, ambient
    /// centre `0.6` — where both walls sit far inside the resolution horizon
    /// (`0.6` and `1.4` standardized against `8.126`), so neither is discarded
    /// as statistically irrelevant.
    #[test]
    fn two_sided_coefficient_bound_keeps_its_far_wall_2523() {
        let columns = 3;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.6, 0.0, 0.0];
        let constraints = two_sided_bound_rows(0.0, 2.0, columns);
        let correction = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &constraints,
        )
        .expect("two-sided correction")
        .expect("an active two-sided bound corrects the posterior");

        assert_eq!(
            correction.rows.len(),
            1,
            "the anti-parallel row adds no direction, so exactly one is retained"
        );
        let limits = correction.upper_limits();
        assert_eq!(limits.len(), 1);
        assert!(
            (limits[0] - 2.0).abs() < 1e-12,
            "the far wall of [0, 2] must arrive as the coordinate's upper limit, got {}",
            limits[0]
        );

        // The retained law is now `u ~ N(0.6, 1)` on `[0, 2]`, not on `[0, ∞)`.
        // Both moments must be the bounded ones.
        let (bounded_mean, bounded_variance) =
            quadrature_box_moments(0.6, 1.0, 2.0);
        let (half_line_mean, half_line_variance) = quadrature_truncated_moments(0.6, 1.0);
        let reported_mean = 0.6 + correction.normal_mean_shift[0];
        let reported_variance = 1.0 - correction.removed_normal_variance[[0, 0]];
        assert!(
            (reported_mean - bounded_mean).abs() < 1e-6,
            "reported mean {reported_mean} must be the [0,2] mean {bounded_mean}, \
             not the [0,inf) mean {half_line_mean}"
        );
        assert!(
            (reported_variance - bounded_variance).abs() < 1e-6,
            "reported variance {reported_variance} must be the [0,2] variance \
             {bounded_variance}, not the [0,inf) variance {half_line_variance}"
        );
        // Discrimination: the two laws are far apart, so agreeing with one is
        // evidence against the other rather than a bound both would clear.
        assert!(
            (bounded_mean - half_line_mean).abs() > 0.1
                && (bounded_variance - half_line_variance).abs() > 0.1,
            "the fixture must separate the two answers: means {bounded_mean} vs \
             {half_line_mean}, variances {bounded_variance} vs {half_line_variance}"
        );
    }

    /// Control for the test above: push the far wall past the resolution horizon
    /// and the answer must return to the half-line one BIT FOR BIT. A coordinate
    /// with no reachable upper limit is not merely close to the old arithmetic,
    /// it takes it.
    #[test]
    fn a_far_wall_beyond_the_horizon_restores_the_half_line_answer_exactly() {
        let columns = 3;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.6, 0.0, 0.0];
        let two_sided = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &two_sided_bound_rows(0.0, 40.0, columns),
        )
        .expect("wide two-sided correction")
        .expect("the lower wall is still active");

        let mut lower_only = Array2::<f64>::zeros((1, columns));
        lower_only[[0, 0]] = 1.0;
        let one_sided = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &LinearInequalityConstraints::new(lower_only, array![0.0]).expect("lower wall"),
        )
        .expect("one-sided correction")
        .expect("an active lower bound corrects the posterior");

        assert_eq!(two_sided.rows.len(), 1);
        assert_eq!(
            two_sided.upper_limits(),
            vec![f64::INFINITY],
            "a wall 39.4 standard deviations away is not a candidate at all"
        );
        assert_eq!(
            two_sided.normal_mean_shift[0], one_sided.normal_mean_shift[0],
            "no reachable upper limit must reproduce the half-line mean shift exactly"
        );
        assert_eq!(
            two_sided.removed_normal_variance[[0, 0]],
            one_sided.removed_normal_variance[[0, 0]],
            "no reachable upper limit must reproduce the half-line variance exactly"
        );
    }

    /// A retained face whose coordinates are half-lines must SURVIVE
    /// persistence (#2601).
    ///
    /// `+∞` is the value an unbounded upper limit takes — not a sentinel, not a
    /// defect — and JSON has no literal for it. Before the extended-real codec,
    /// `serde_json` wrote each one as `null` and `Vec<f64>` refused its own
    /// output on the way back in with `invalid type: null, expected f64`, so
    /// every shape-constrained fit that retained a face produced a model that
    /// could not be loaded. The failure surfaced at LOAD, arbitrarily far from
    /// the fit, and named neither the field nor the fit.
    ///
    /// This is the type-level guard: whatever the fit produces, a correction
    /// carrying infinite, finite, and mixed limits must come back bit-for-bit.
    #[test]
    fn a_half_line_upper_limit_survives_the_json_round_trip_2601() {
        for limits in [
            vec![f64::INFINITY; 3],
            vec![2.5, f64::INFINITY, 1e300],
            Vec::new(),
        ] {
            let q = limits.len().max(1);
            let correction = ConstrainedPosteriorCorrection {
                lift: Array2::<f64>::zeros((4, q)),
                removed_normal_variance: Array2::<f64>::eye(q),
                normal_mean_shift: Array1::<f64>::zeros(q),
                rows: (0..q).collect(),
                normal_upper_limits: limits.clone(),
            };
            let json = serde_json::to_string(&correction).expect("serialize correction");
            let back: ConstrainedPosteriorCorrection =
                serde_json::from_str(&json).unwrap_or_else(|e| {
                    panic!("a correction with limits {limits:?} must reload: {e}\n{json}")
                });
            assert_eq!(
                back.normal_upper_limits, limits,
                "upper limits must round-trip bit for bit"
            );
            assert_eq!(back.upper_limits(), correction.upper_limits());
        }
    }

    /// The end-to-end shape of the same defect: a correction produced by the
    /// solver (not hand-built) must reload. `Σ = I` with a lower bound only
    /// gives exactly the `+∞`-limit face that #2601's `[concave]` fit produced.
    #[test]
    fn a_solver_produced_half_line_correction_reloads_2601() {
        let columns = 3;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.6, 0.0, 0.0];
        let mut lower_only = Array2::<f64>::zeros((1, columns));
        lower_only[[0, 0]] = 1.0;
        let correction = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &LinearInequalityConstraints::new(lower_only, array![0.0]).expect("lower wall"),
        )
        .expect("one-sided correction")
        .expect("an active lower bound corrects the posterior");
        assert_eq!(
            correction.normal_upper_limits,
            vec![f64::INFINITY],
            "precondition: a half-line coordinate carries an infinite upper limit"
        );

        let json = serde_json::to_string(&correction).expect("serialize");
        let back: ConstrainedPosteriorCorrection =
            serde_json::from_str(&json).expect("a solver-produced correction must reload");
        assert_eq!(back.normal_upper_limits, vec![f64::INFINITY]);

        // And the structural write-side guard must NOT mistake the legitimate
        // half-line for the defect it exists to catch.
        assert!(
            gam_problem::ensure_serialized_floats_are_finite(&correction).is_ok(),
            "an unbounded upper limit is a value, not a non-finite defect"
        );
    }

    /// The regime a two-sided bound is declared FOR: the unconstrained fit lands
    /// beyond the far wall, so the retained slab sits deep in a tail. This is
    /// what the reflection inside the cubature and the closed form exist for; a
    /// sign error there puts the posterior mean outside its own box.
    #[test]
    fn a_box_the_unconstrained_centre_overshoots_stays_inside_itself() {
        let columns = 2;
        let covariance = Array2::<f64>::eye(columns);
        // Ambient centre at -3 with the box [-1, 1]: the coordinate
        // `u = beta_0 + 1` has untruncated mean -2 and lives on [0, 2].
        let centre = array![-3.0, 0.0];
        let correction = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &two_sided_bound_rows(-1.0, 1.0, columns),
        )
        .expect("overshooting correction")
        .expect("both walls bind");

        let limits = correction.upper_limits();
        assert!(
            (limits[0] - 2.0).abs() < 1e-12,
            "the slab is two units wide, got {}",
            limits[0]
        );
        let reported_mean = -2.0 + correction.normal_mean_shift[0];
        let reported_variance = 1.0 - correction.removed_normal_variance[[0, 0]];
        assert!(
            reported_mean > 0.0 && reported_mean < limits[0],
            "the posterior mean of a law supported on [0, {}] cannot sit outside it, \
             got {reported_mean}",
            limits[0]
        );
        // Popoviciu: any law on an interval of width `w` has variance at most
        // `w²/4`. A one-sided answer here would report ~0.06 on a coordinate the
        // box confines to at most 1.0, so this separates them.
        assert!(
            reported_variance > 0.0 && reported_variance <= limits[0] * limits[0] / 4.0,
            "variance {reported_variance} exceeds the width bound for [0, {}]",
            limits[0]
        );
        let (expected_mean, expected_variance) = quadrature_box_moments(-2.0, 1.0, 2.0);
        assert!(
            (reported_mean - expected_mean).abs() < 1e-6
                && (reported_variance - expected_variance).abs() < 1e-6,
            "deep-tail slab moments {reported_mean}/{reported_variance} against the \
             independent quadrature {expected_mean}/{expected_variance}"
        );
    }

    /// The two branches of the scalar closed form are separate code, so they are
    /// checked against each other where they must agree: an upper limit far
    /// enough out that it removes no representable mass.
    #[test]
    fn the_two_sided_scalar_form_meets_the_mills_branch_at_a_distant_wall() {
        for &(mean, variance) in &[(0.6f64, 1.0f64), (-2.5, 1.0), (0.0, 4.0), (3.0, 0.25)] {
            let sd: f64 = variance.sqrt();
            let distant = mean + 40.0 * sd;
            let (bounded_mean, bounded_variance) =
                scalar_truncated_moments(mean, variance, distant).expect("bounded");
            let (open_mean, open_variance) =
                scalar_truncated_moments(mean, variance, f64::INFINITY).expect("half line");
            assert!(
                (bounded_mean[0] - open_mean[0]).abs() <= 1e-12 * open_mean[0].abs().max(1.0),
                "mean {} vs {} at mean={mean} variance={variance}",
                bounded_mean[0],
                open_mean[0]
            );
            assert!(
                (bounded_variance[[0, 0]] - open_variance[[0, 0]]).abs()
                    <= 1e-12 * open_variance[[0, 0]].abs().max(1.0),
                "variance {} vs {} at mean={mean} variance={variance}",
                bounded_variance[[0, 0]],
                open_variance[[0, 0]]
            );
        }
    }

    /// The two-sided closed form across the regimes the reflection switches on,
    /// against Simpson quadrature of the density itself.
    #[test]
    fn the_two_sided_scalar_form_matches_an_independent_quadrature() {
        for &(mean, variance, upper) in &[
            (0.6f64, 1.0f64, 2.0f64),
            (-1.5, 1.0, 0.5),
            (3.0, 0.25, 0.4),
            (-4.0, 1.0, 0.2),
            (0.05, 1.0, 0.1),
            (-2.0, 1.0, 2.0),
            (0.5, 9.0, 12.0),
        ] {
            let (moment_mean, moment_variance) =
                scalar_truncated_moments(mean, variance, upper).expect("two-sided moments");
            let (reference_mean, reference_variance) =
                quadrature_box_moments(mean, variance, upper);
            let scale = variance.sqrt();
            assert!(
                (moment_mean[0] - reference_mean).abs() < 1e-9 * scale,
                "mean {} vs {reference_mean} at mean={mean} variance={variance} upper={upper}",
                moment_mean[0]
            );
            assert!(
                (moment_variance[[0, 0]] - reference_variance).abs() < 1e-9 * variance,
                "variance {} vs {reference_variance} at mean={mean} variance={variance} \
                 upper={upper}",
                moment_variance[[0, 0]]
            );
            assert!(
                moment_mean[0] > 0.0 && moment_mean[0] < upper,
                "the mean of a law on [0, {upper}] must lie inside it, got {}",
                moment_mean[0]
            );
        }
    }

    /// The multivariate cubature over a genuine box, against the same
    /// coordinatewise reference on a product law where the two agree exactly.
    /// A diagonal covariance makes the box-truncated joint the product of its
    /// box-truncated marginals, so the reference needs no second cubature.
    #[test]
    fn the_box_cubature_reproduces_a_product_law_it_cannot_shortcut() {
        let mean = array![0.4, -1.2, 0.9];
        let covariance = Array2::from_diag(&array![1.0, 0.5, 2.0]);
        let upper = vec![1.5, 0.8, f64::INFINITY];
        let factor = gam_linalg::triangular::cholesky_factor_in_place(
            covariance.view(),
            gam_linalg::triangular::CholeskyGuard::FiniteStrict,
        )
        .expect("diagonal factor");
        let (cubature_mean, cubature_covariance) =
            box_truncated_moments(&mean, &upper, &covariance, factor.view()).expect("box moments");
        for i in 0..mean.len() {
            let (reference_mean, reference_variance) =
                scalar_truncated_moments(mean[i], covariance[[i, i]], upper[i])
                    .expect("marginal closed form");
            let sd = covariance[[i, i]].sqrt();
            assert!(
                (cubature_mean[i] - reference_mean[0]).abs()
                    < ORTHANT_MOMENT_RELATIVE_TOLERANCE * sd,
                "coordinate {i} mean {} vs {}",
                cubature_mean[i],
                reference_mean[0]
            );
            assert!(
                (cubature_covariance[[i, i]] - reference_variance[[0, 0]]).abs()
                    < ORTHANT_MOMENT_RELATIVE_TOLERANCE * covariance[[i, i]],
                "coordinate {i} variance {} vs {}",
                cubature_covariance[[i, i]],
                reference_variance[[0, 0]]
            );
        }
        // Independence survives truncation to a product region, so every
        // off-diagonal must vanish. This is what a mis-indexed upper limit would
        // break first.
        for i in 0..mean.len() {
            for j in 0..mean.len() {
                if i == j {
                    continue;
                }
                let sd = (covariance[[i, i]] * covariance[[j, j]]).sqrt();
                assert!(
                    cubature_covariance[[i, j]].abs() < ORTHANT_MOMENT_RELATIVE_TOLERANCE * sd,
                    "a product law truncated to a box stays a product law: entry ({i},{j}) \
                     is {}",
                    cubature_covariance[[i, j]]
                );
            }
        }
    }

    /// The reflection inside the cubature, gated where it decides the answer
    /// rather than where it is merely present.
    ///
    /// A slab sitting `d` standard deviations BELOW the mean has both endpoints
    /// deep in the lower tail, where `Φ̄` is within rounding of one and the
    /// slab's whole mass is the difference between them. Measured against a
    /// Simpson reference on a diagonal `q = 2` law — where the box-truncated
    /// joint is exactly the product of its box-truncated marginals, so the
    /// reference needs no second cubature — the unreflected arithmetic holds to
    /// `d = 9` and then fails completely: at `d = 12` it returns zero for a mean
    /// of 1.9019, and by `d = 40` every node has underflowed and there is no
    /// mass left to normalize. Reflected, the error is 2.2e-5 at `d = 12` and
    /// keeps FALLING with depth, reaching 6.1e-6 at `d = 40`.
    ///
    /// `d = 12` is therefore the shallowest depth at which this test can tell
    /// the two apart, which is why it is the depth used.
    #[test]
    fn a_slab_twelve_deviations_below_the_mean_keeps_its_mass() {
        let depth = 12.0_f64;
        let mean = array![depth, depth];
        let covariance = Array2::<f64>::eye(2);
        let upper = vec![2.0, 2.0];
        let factor = gam_linalg::triangular::cholesky_factor_in_place(
            covariance.view(),
            gam_linalg::triangular::CholeskyGuard::FiniteStrict,
        )
        .expect("identity factor");
        let (cubature_mean, cubature_covariance) =
            box_truncated_moments(&mean, &upper, &covariance, factor.view())
                .expect("a slab deep in a tail still carries mass");
        let (reference_mean, reference_variance) = quadrature_box_moments(depth, 1.0, 2.0);
        // The density rises across the whole slab, so the mean sits near the far
        // wall; a lost slab would report 0 and a mis-signed reflection would
        // report the mirror image near the near wall.
        assert!(
            reference_mean > 1.85 && reference_mean < 2.0,
            "the fixture must place the mean near the far wall, got {reference_mean}"
        );
        for i in 0..2 {
            assert!(
                (cubature_mean[i] - reference_mean).abs() < 1e-3,
                "coordinate {i} mean {} against the Simpson reference {reference_mean}",
                cubature_mean[i]
            );
            assert!(
                (cubature_covariance[[i, i]] - reference_variance).abs() < 1e-3,
                "coordinate {i} variance {} against the Simpson reference {reference_variance}",
                cubature_covariance[[i, i]]
            );
        }
    }

    /// Independent reference: Simpson CDF of `N(mean, variance)` restricted to
    /// `[0, upper]`, evaluated at `x`. Built from the density, so it shares no
    /// tail function with the quantile under test.
    fn quadrature_box_cdf(mean: f64, variance: f64, upper: f64, x: f64) -> f64 {
        let sd = variance.sqrt();
        let low = -mean / sd;
        let high = (upper - mean) / sd;
        let point = (x - mean) / sd;
        let reference = if low <= 0.0 && 0.0 <= high {
            0.0
        } else if high < 0.0 {
            high
        } else {
            low
        };
        let mass = |from: f64, to: f64| -> f64 {
            let panels = 200_000usize;
            let step = (to - from) / panels as f64;
            let mut total = 0.0f64;
            for index in 0..=panels {
                let z = from + step * index as f64;
                let simpson = if index == 0 || index == panels {
                    1.0
                } else if index % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                total += simpson * (-(z * z - reference * reference) / 2.0).exp();
            }
            total * step
        };
        mass(low, point) / mass(low, high)
    }

    /// The quantile's reflection is a SEPARATE sign from the mass's, and a
    /// mass-only gate passes straight over a broken one.
    ///
    /// Credit to the #2529 lane, who hit exactly this: their first deep-tail
    /// interval returned the upper endpoint for every requested fraction — a
    /// perfectly in-bounds, monotone-looking answer — while the mass beside it
    /// stayed correct, and every case shallower than about 9σ passed. This
    /// module has a clamp into `[low, high]` on the inverted quantile, which is
    /// justified as rounding but is exactly the construct that would render
    /// such a collapse as a clean in-range number.
    ///
    /// So this asserts the round trip rather than the range: distinct, strictly
    /// increasing answers, strictly interior, and `F(Q(f)) = f` against a
    /// Simpson CDF built from the density. At `d = 12` the reported errors are
    /// at 1e-14; a collapse would show as equal answers and `F` pinned at one.
    #[test]
    fn the_deep_tail_quantile_round_trips_rather_than_collapsing_to_an_endpoint() {
        let mean = 12.0_f64;
        let variance = 1.0_f64;
        let upper = 2.0_f64;
        let fractions = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
        let mut previous = 0.0_f64;
        for &fraction in &fractions {
            let point = scalar_truncated_quantile(mean, variance, upper, fraction)
                .expect("a slab deep in a tail still has quantiles");
            assert!(
                point > 0.0 && point < upper,
                "the {fraction} quantile of a law on [0, {upper}] must be interior, got {point}"
            );
            assert!(
                point > previous,
                "quantiles must be strictly increasing in the probability; {fraction} gave \
                 {point} against {previous} for the fraction before it, which is what a \
                 collapse to one endpoint looks like"
            );
            previous = point;
            let recovered = quadrature_box_cdf(mean, variance, upper, point);
            assert!(
                (recovered - fraction).abs() < 1e-6,
                "round trip at {fraction}: the quantile returned {point}, whose independent \
                 Simpson CDF is {recovered}"
            );
        }
        // The span is bracketed rather than merely bounded below, because a
        // degenerate rule can be increasing and interior and still useless, and
        // a rule with the wrong scale can be wide.
        //
        // The bracket is derived, not chosen. Across a slab this far into a
        // tail the Gaussian log-density is very nearly linear, with slope
        // `(mean - x)/variance` — steepest at the near wall, shallowest at the
        // far one. An exponential law of rate `λ` puts its 1%-99% span at
        // `ln(99)/λ`, so the true span must sit between the values those two
        // slopes give. At `mean = 12`, `variance = 1`, `upper = 2` that is
        // `[ln(99)/12, ln(99)/10] = [0.3829, 0.4595]`, and the rule returns
        // 0.4453. The first version of this assertion used `0.25 * upper`, a
        // guess, and it was wrong in the direction that would have made the
        // test reject a correct answer — the span is set by the density's
        // steepness, not by the width of the box.
        let first = scalar_truncated_quantile(mean, variance, upper, 0.01).expect("low");
        let last = scalar_truncated_quantile(mean, variance, upper, 0.99).expect("high");
        let span = last - first;
        let steepest = mean / variance;
        let shallowest = (mean - upper) / variance;
        let narrowest = 99.0_f64.ln() / steepest;
        let widest = 99.0_f64.ln() / shallowest;
        assert!(
            span >= narrowest && span <= widest,
            "the 1%-99% span {span} is outside the [{narrowest}, {widest}] the log-density's \
             own slopes allow across this slab"
        );
    }

    /// A two-sided bound with no width between its walls is an equality
    /// constraint, and this module reports moments of a density. It refuses
    /// rather than reporting the moments of a point.
    #[test]
    fn coincident_two_sided_walls_are_refused_not_collapsed() {
        let columns = 2;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.5, 0.0];
        let error = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &two_sided_bound_rows(0.25, 0.25, columns),
        )
        .expect_err("an empty slab has no posterior to report");
        assert!(
            error.contains("no width between them"),
            "the refusal must name the geometry, got: {error}"
        );
    }

    /// The interval path reads the same box the moments do. An equal-tailed
    /// interval for a bounded coordinate cannot leave the coordinate's own
    /// bounds — which is exactly the confidently-wrong report #2523 describes.
    #[test]
    fn a_two_sided_projection_interval_stays_within_its_own_bounds() {
        let columns = 2;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.6, 0.0];
        let constraints = two_sided_bound_rows(0.0, 1.0, columns);
        let correction = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &constraints,
        )
        .expect("correction")
        .expect("active");
        let geometry = ConstrainedPosteriorGeometry {
            constraints,
            mode: array![0.6, 0.0],
            unconstrained_center: Some(centre),
            correction: Some(correction),
            moment_status: ConstrainedPosteriorMomentStatus::Available,
        };
        let (low, high) = constrained_projection_equal_tailed_interval(
            &covariance,
            &geometry,
            &array![1.0, 0.0],
            0.95,
        )
        .expect("two-sided projection interval");
        assert!(
            low >= -1e-9 && high <= 1.0 + 1e-9,
            "a coefficient declared to lie in [0, 1] cannot be reported in [{low}, {high}]"
        );
        assert!(low < high, "the interval must be non-degenerate");
    }
}

/// Coverage gate for #2417.
///
/// The estimand is settled by COVERAGE against a known truth, not by the two
/// fit paths agreeing: two paths agreeing on a wrong covariance is not
/// progress. Each cell simulates from a known constrained model, refits with
/// the production constrained-quadratic solver, and measures what fraction of
/// nominal-95% intervals actually contain the truth under four procedures:
///
/// * **full space** `Σ = φH⁻¹` centred at the constrained mode — what the PIRLS
///   path reported before this change, with no reference to the active geometry;
/// * **active face** `Z(ZᵀHZ)⁻¹Zᵀ` centred at the mode — what the blockwise path
///   reports, reproduced here with the SAME tightness predicate it uses
///   (`scaled slack ≤ ACTIVE_SET_WORKING_FACE_TOL`, `covariance.rs:1583-1592`);
/// * **truncated** `Σ − G(W − Cov[u])Gᵀ` centred at the mode — what this module
///   computes and what the fit now reports;
/// * **truncated, mean-centred** — the same covariance around the truncated
///   posterior MEAN rather than the mode. Not what the fit ships (moving the
///   reported coefficients is out of scope for #2417); measured so the size of
///   the mode-vs-mean effect is on the record rather than asserted.
///
/// Among procedures that reach nominal coverage, expected interval length is
/// the tie-break.
#[cfg(test)]
mod orthant_tilt_2601_tests {
    use super::*;

    /// The face that #2601's `monotone_decreasing` fit actually produces,
    /// captured from the failing fit rather than guessed at.
    ///
    /// `y ~ s(x, shape=monotone_decreasing)` on 300 rows of clean increasing
    /// linear data — the `[monotone_decreasing]` parametrisation of
    /// `test_gaussian_reml_fit_all_shape_constraints_do_not_panic`. All eleven
    /// monotonicity coordinates are active.
    ///
    ///   wall depth   -0.66 .. +2.65 sd   (mild; two walls are on the FEASIBLE side)
    ///   max |corr|    0.691
    ///   corr eigenvalues 0.144 .. 1.920
    fn refusing_face() -> (Array1<f64>, Array2<f64>) {
        let mean = Array1::from_vec(vec![
            -1.73263658148929162e-01, -1.61028415044015161e-01, -1.53745491056003519e-01,
            -1.14020228922959738e-01, -1.13372022294778579e-01, -5.68386260921473555e-02,
            -2.01787006225858517e-02, 7.79317061445884696e-04, 1.73520774327455551e-03,
            2.00473386863282976e-02, 4.22790949294645502e-02,
        ]);
        let w = Array2::from_shape_vec(
            (11, 11),
            vec![
                4.28613165678045412e-03, -6.40620724624387243e-04, -6.68752057617351208e-04,
                -5.83957865274831135e-04, -5.55233848052857893e-04, -7.90532734874588739e-04,
                -8.09846363272953844e-04, -2.35978352506513071e-04, -4.59953354398140966e-04,
                -2.40276816847910670e-04, -4.66860409610777736e-04, -6.40620724624387243e-04,
                4.30519348418363125e-03, -5.77958088890881253e-04, -7.08255560103122385e-04,
                -6.52329255706248488e-04, -4.23821703180157501e-04, -7.53554384768477959e-04,
                -1.29045376459765944e-04, -2.55856263721782311e-04, -3.25781213177120355e-04,
                -6.88688636712164021e-04, -6.68752057617351208e-04, -5.77958088890881253e-04,
                4.33564940679504254e-03, -6.74857893211149419e-04, -6.22390211789637811e-04,
                -7.39567533429216599e-04, -4.59996677084055332e-04, -3.33480282492368946e-04,
                -7.09611827680745955e-04, -1.43809374573677527e-04, -2.85910172307334801e-04,
                -5.83957865274831135e-04, -7.08255560103122385e-04, -6.74857893211149419e-04,
                4.23561983569142528e-03, -3.68170739599590739e-04, -2.42878795544258373e-04,
                -6.65817316820568449e-04, -7.99047220448408411e-05, -1.55911037671571136e-04,
                -1.76197943321878327e-04, -2.51754204620259080e-04, -5.55233848052857893e-04,
                -6.52329255706248488e-04, -6.22390211789637811e-04, -3.68170739599590739e-04,
                4.29588162752489629e-03, -7.57152955247313302e-04, -2.80188801979815898e-04,
                -2.28980327675770300e-04, -3.66373527157145380e-04, -9.83175770979363879e-05,
                -1.94759571987295659e-04, -7.90532734874588739e-04, -4.23821703180157501e-04,
                -7.39567533429216599e-04, -2.42878795544258373e-04, -7.57152955247313302e-04,
                4.82590971142383366e-03, -2.30789443520484135e-04, 5.11092628516201207e-04,
                4.74012818803116673e-04, -1.01494286876908989e-04, -2.04446376075374456e-04,
                -8.09846363272953844e-04, -7.53554384768477959e-04, -4.59996677084055332e-04,
                -6.65817316820568449e-04, -2.80188801979815898e-04, -2.30789443520484135e-04,
                4.97201788205004890e-03, -9.74102815703460092e-05, -1.93235956110176968e-04,
                5.67517929096417986e-04, 5.90297796785945318e-04, -2.35978352506513071e-04,
                -1.29045376459765944e-04, -3.33480282492368946e-04, -7.99047220448408411e-05,
                -2.28980327675770300e-04, 5.11092628516201207e-04, -9.74102815703460092e-05,
                1.15689169020833748e-03, 1.48276767553545967e-03, -4.51194193048010442e-05,
                -9.11302898497036765e-05, -4.59953354398140966e-04, -2.55856263721782311e-04,
                -7.09611827680745955e-04, -1.55911037671571136e-04, -3.66373527157145380e-04,
                4.74012818803116673e-04, -1.93235956110176968e-04, 1.48276767553545967e-03,
                4.05500634945223787e-03, -8.98171912603273025e-05, -1.81408583551147815e-04,
                -2.40276816847910670e-04, -3.25781213177120355e-04, -1.43809374573677527e-04,
                -1.76197943321878327e-04, -9.83175770979363879e-05, -1.01494286876908989e-04,
                5.67517929096417986e-04, -4.51194193048010442e-05, -8.98171912603273025e-05,
                1.17935427809862667e-03, 1.52706063180176217e-03, -4.66860409610777736e-04,
                -6.88688636712164021e-04, -2.85910172307334801e-04, -2.51754204620259080e-04,
                -1.94759571987295659e-04, -2.04446376075374456e-04, 5.90297796785945318e-04,
                -9.11302898497036765e-05, -1.81408583551147815e-04, 1.52706063180176217e-03,
                4.14230573743777988e-03,
            ],
        )
        .expect("11x11 captured constraint-normal covariance");
        (mean, w)
    }

    fn lower_cholesky(w: &Array2<f64>) -> Array2<f64> {
        let q = w.nrows();
        let mut factor = Array2::<f64>::zeros((q, q));
        for i in 0..q {
            for j in 0..=i {
                let mut sum = w[[i, j]];
                for k in 0..j {
                    sum -= factor[[i, k]] * factor[[j, k]];
                }
                if i == j {
                    factor[[i, i]] = f64::sqrt(sum);
                } else {
                    factor[[i, j]] = sum / factor[[j, j]];
                }
            }
        }
        factor
    }

    /// Effective sample size of the self-normalized node weights, as a fraction
    /// of the nodes evaluated. This is the quantity that decides whether the
    /// estimator is a cubature or a Monte Carlo draw wearing one's clothes.
    fn weight_efficiency(
        mean: &Array1<f64>,
        upper: &[f64],
        factor: ArrayView2<'_, f64>,
        tilt: Option<&Array1<f64>>,
        nodes: usize,
    ) -> (f64, f64) {
        struct WeightSpy {
            inner: OrthantAccumulator,
            log_weights: Vec<f64>,
        }
        impl OrthantNodeSink for WeightSpy {
            fn push(&mut self, log_weight: f64, point: &Array1<f64>) {
                self.log_weights.push(log_weight);
                self.inner.push(log_weight, point);
            }
        }
        let q = mean.len();
        let generator = kronecker_generator(q);
        let mut spy = WeightSpy {
            inner: OrthantAccumulator::new(q),
            log_weights: Vec::new(),
        };
        accumulate_orthant_nodes(
            &mut spy, mean, upper, None, factor, &generator, tilt, 0, 0, nodes,
        )
        .expect("orthant nodes");
        let finite: Vec<f64> = spy
            .log_weights
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let sum: f64 = finite.iter().map(|v| (v - hi).exp()).sum();
        let sum_sq: f64 = finite.iter().map(|v| (2.0 * (v - hi)).exp()).sum();
        (
            (sum * sum / sum_sq) / finite.len() as f64,
            (hi - lo) / std::f64::consts::LN_10,
        )
    }

    /// The mechanism, measured on the real face: without the tilt the node
    /// weights span 26 decades and 99% of the evaluated nodes contribute
    /// nothing; with it they span ~1 decade and essentially every node counts.
    ///
    /// This is the assertion that would catch a regression of #2601 mechanism 3
    /// from the direction the convergence test cannot see. A future change that
    /// broke the tilt but left the tolerance reachable by brute force would pass
    /// a convergence check and fail this one.
    #[test]
    fn the_tilt_turns_a_monte_carlo_draw_back_into_a_cubature_2601() {
        let (mean, w) = refusing_face();
        let q = mean.len();
        let upper = vec![f64::INFINITY; q];
        let factor = lower_cholesky(&w);

        let (untilted_ess, untilted_decades) =
            weight_efficiency(&mean, &upper, factor.view(), None, 1 << 16);
        let tilt = minimax_tilt(&mean, &upper, factor.view()).expect("this face is tilted");
        let (tilted_ess, tilted_decades) =
            weight_efficiency(&mean, &upper, factor.view(), Some(&tilt), 1 << 16);

        println!(
            "MEASURE2601 untilted ess={:.4}% over {:.1} decades; \
             tilted ess={:.4}% over {:.1} decades",
            100.0 * untilted_ess,
            untilted_decades,
            100.0 * tilted_ess,
            tilted_decades,
        );
        assert!(
            untilted_ess < 0.05,
            "precondition: the untilted proposal wastes the node budget on this \
             face (ess {:.4}% of nodes)",
            100.0 * untilted_ess
        );
        assert!(
            tilted_ess > 0.5,
            "the tilt must make most nodes count; got ess {:.4}% of nodes over \
             {tilted_decades:.1} decades of weight",
            100.0 * tilted_ess
        );
        assert!(
            tilted_decades < 5.0,
            "the tilted weights must be nearly flat; got {tilted_decades:.1} decades"
        );
    }

    /// The behaviour #2601 mechanism 3 reports: this face must produce moments,
    /// and they must be the RIGHT moments.
    ///
    /// Correctness is scored against the UNTILTED rule carried far past the
    /// production cap. That is an independent proposal — a different estimator
    /// of the same integral — so agreement is evidence about the answer rather
    /// than about one estimator's self-consistency. The band is set by the
    /// reference's own error (~1.3e-2 at 2^20, measured), not by what the
    /// production rule happens to produce.
    #[test]
    fn the_face_that_refuses_2601_produces_the_right_moments() {
        let (mean, w) = refusing_face();
        let q = mean.len();
        let upper = vec![f64::INFINITY; q];
        let factor = lower_cholesky(&w);

        // Pin the geometry, so a future capture that drifts cannot silently
        // inherit the conclusion drawn from this one.
        let sd: Vec<f64> = (0..q).map(|i| f64::sqrt(w[[i, i]])).collect();
        let depth: Vec<f64> = (0..q).map(|i| -mean[i] / sd[i]).collect();
        let depth_min = depth.iter().copied().fold(f64::INFINITY, f64::min);
        let depth_max = depth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut corr_max = 0.0f64;
        for i in 0..q {
            for j in 0..i {
                corr_max = corr_max.max(f64::abs(w[[i, j]] / (sd[i] * sd[j])));
            }
        }
        assert!(
            depth_max < 3.0 && depth_min < 0.0,
            "the refusing face is MILD in depth ({depth_min:.2}..{depth_max:.2} sd), \
             which is what rules depth out as the cause"
        );
        assert!(
            corr_max > 0.6,
            "the refusing face is strongly correlated (max |corr| = {corr_max:.3}), \
             which is the regime that fails"
        );

        let (produced_mean, produced_cov) = box_truncated_moments(&mean, &upper, &w, factor.view())
            .expect("the face #2601 reports must produce moments");

        let generator = kronecker_generator(q);
        let mut reference = OrthantAccumulator::new(q);
        accumulate_orthant_nodes(
            &mut reference,
            &mean,
            &upper,
            None,
            factor.view(),
            &generator,
            None,
            0,
            0,
            1 << 20,
        )
        .expect("untilted reference nodes");
        let truth = reference.moments().expect("reference moments");
        let gap = moment_relative_change(&(produced_mean, produced_cov), &truth, &w);
        println!("MEASURE2601 gap vs untilted 2^20 reference: {gap:.3e}");
        assert!(
            gap < 3.0e-2,
            "the tilted rule must agree with an INDEPENDENT untilted reference; \
             gap {gap:.3e} (the reference's own error at 2^20 is ~1.3e-2)"
        );
    }

    /// Before/after across the synthetic sweep that first identified
    /// correlation as the driver: every face the untilted rule could not
    /// resolve at the 2^20 cap, the tilted one resolves — and the ones it
    /// already resolved it resolves no slower.
    ///
    /// The sweep is what rules out tail depth: an 11-dimensional face FOUR
    /// standard deviations deep converges in 16k nodes when the constraint
    /// normals are uncorrelated, while correlation at ANY depth, including
    /// 0.5 sd, does not converge at 2^20 without the tilt.
    #[test]
    fn the_tilt_resolves_every_correlated_face_the_sweep_could_not() {
        for &q in &[4usize, 8, 11] {
            for &c in &[0.5_f64, 1.0, 2.0, 4.0] {
                for &corr in &[0.0_f64, 0.6, 0.9] {
                    let mut w = Array2::<f64>::zeros((q, q));
                    for i in 0..q {
                        for j in 0..q {
                            w[[i, j]] = corr.powi((i as i32 - j as i32).abs());
                        }
                    }
                    let mean = Array1::<f64>::from_elem(q, -c);
                    let upper = vec![f64::INFINITY; q];
                    let factor = lower_cholesky(&w);
                    let (ess, decades) =
                        weight_efficiency(&mean, &upper, factor.view(), None, 1 << 14);
                    let tilt = minimax_tilt(&mean, &upper, factor.view());
                    let (tilted_ess, tilted_decades) =
                        weight_efficiency(&mean, &upper, factor.view(), tilt.as_ref(), 1 << 14);
                    let outcome = box_truncated_moments(&mean, &upper, &w, factor.view());
                    println!(
                        "MEASURE2601 q={q} depth={c} corr={corr} \
                         ess {:.2}%->{:.2}% decades {decades:.1}->{tilted_decades:.1} {}",
                        100.0 * ess,
                        100.0 * tilted_ess,
                        if outcome.is_ok() { "converged" } else { "REFUSED" },
                    );
                    assert!(
                        outcome.is_ok(),
                        "q={q} depth={c} corr={corr} must converge: {:?}",
                        outcome.err()
                    );
                    assert!(
                        tilted_ess >= ess * 0.9,
                        "the tilt must never make a face WORSE: q={q} depth={c} \
                         corr={corr} ess {:.4}% -> {:.4}%",
                        100.0 * ess,
                        100.0 * tilted_ess
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod coverage_gate_tests {
    use super::*;
    use gam_linalg::triangular::{CholeskyGuard, cholesky_factor_in_place, cholesky_solve_vector};

    /// Deterministic SplitMix64 — the gate must produce the same numbers on
    /// every host, so no external RNG and no thread-local state.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.unit().max(1.0e-12), self.unit());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// Realized coverage and mean half-width of one interval procedure.
    struct CoverageTally {
        covered: usize,
        replicates: usize,
        total_half_width: f64,
    }

    impl CoverageTally {
        fn new() -> Self {
            Self {
                covered: 0,
                replicates: 0,
                total_half_width: 0.0,
            }
        }
        fn record(&mut self, center: f64, half_width: f64, truth: f64) {
            self.replicates += 1;
            self.total_half_width += half_width;
            if (truth - center).abs() <= half_width {
                self.covered += 1;
            }
        }
        fn coverage(&self) -> f64 {
            self.covered as f64 / self.replicates as f64
        }
        fn mean_half_width(&self) -> f64 {
            self.total_half_width / self.replicates as f64
        }
    }

    /// The four procedures the gate compares, in one bundle.
    struct CellResult {
        full_space: CoverageTally,
        active_face: CoverageTally,
        truncated: CoverageTally,
        truncated_mean_centred: CoverageTally,
        pinned_fraction: f64,
    }

    /// Two-sided nominal level the gate reports against.
    const NOMINAL_HALF_WIDTH_MULTIPLIER: f64 = 1.959_963_984_540_054;
    const NOMINAL_COVERAGE: f64 = 0.95;

    /// `Σ = σ²(XᵀX)⁻¹` for a fixed design.
    fn gaussian_posterior_covariance(gram: &Array2<f64>, noise_variance: f64) -> Array2<f64> {
        let p = gram.nrows();
        let factor = cholesky_factor_in_place(gram.view(), CholeskyGuard::FiniteStrict)
            .expect("simulation design is full rank");
        let mut covariance = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let mut unit = Array1::<f64>::zeros(p);
            unit[j] = 1.0;
            let column = cholesky_solve_vector(&factor, &unit);
            for i in 0..p {
                covariance[[i, j]] = noise_variance * column[i];
            }
        }
        covariance
    }

    /// Rows of `A β ≥ b` that are tight at `beta`, under the SAME scaled-slack
    /// predicate the blockwise covariance path applies at `β̂`.
    fn tight_rows_at(constraints: &LinearInequalityConstraints, beta: &Array1<f64>) -> Vec<usize> {
        let mut tight = Vec::new();
        for row_index in 0..constraints.a.nrows() {
            let row = constraints.a.row(row_index).to_owned();
            let norm = row.dot(&row).sqrt();
            if norm > 0.0
                && (row.dot(beta) - constraints.b[row_index]) / norm
                    <= crate::active_set::ACTIVE_SET_WORKING_FACE_TOL
            {
                tight.push(row_index);
            }
        }
        tight
    }

    /// `Σ_face = Σ − ΣA_tᵀ(A_tΣA_tᵀ)⁻¹A_tΣ` on the tight rows: the active-face
    /// reduction written as the same low-rank removal, so the comparator and
    /// the estimand under test differ ONLY in whether the constraint-normal
    /// variance is removed in full or only by the truncated part.
    fn active_face_variance(
        covariance: &Array2<f64>,
        constraints: &LinearInequalityConstraints,
        tight: &[usize],
        index: usize,
    ) -> f64 {
        if tight.is_empty() {
            return covariance[[index, index]];
        }
        let q = tight.len();
        let mut sigma_at = Array2::<f64>::zeros((covariance.nrows(), q));
        for (position, &row_index) in tight.iter().enumerate() {
            let column = covariance.dot(&constraints.a.row(row_index).to_owned());
            sigma_at.column_mut(position).assign(&column);
        }
        let mut normal = Array2::<f64>::zeros((q, q));
        for (i, &row_i) in tight.iter().enumerate() {
            for j in 0..q {
                normal[[i, j]] = constraints
                    .a
                    .row(row_i)
                    .to_owned()
                    .dot(&sigma_at.column(j).to_owned());
            }
        }
        let Some(factor) = cholesky_factor_in_place(normal.view(), CholeskyGuard::FiniteStrict)
        else {
            // A rank-deficient tight face pins every direction it spans; the
            // face answer for this coordinate is zero variance.
            return 0.0;
        };
        let row = sigma_at.row(index).to_owned();
        let solved = cholesky_solve_vector(&factor, &row);
        covariance[[index, index]] - row.dot(&solved)
    }

    /// One simulation cell: a fixed design, a known truth strictly inside
    /// `A β ≥ 0`, and `replicates` refits under Gaussian noise with a KNOWN
    /// noise scale, so the comparison isolates the covariance question from
    /// dispersion estimation.
    fn run_cell(
        design: &Array2<f64>,
        truth: &Array1<f64>,
        constraints: &LinearInequalityConstraints,
        reported_index: usize,
        noise_sd: f64,
        replicates: usize,
        seed: u64,
    ) -> CellResult {
        let n = design.nrows();
        let p = design.ncols();
        let gram = design.t().dot(design);
        let covariance = gaussian_posterior_covariance(&gram, noise_sd * noise_sd);
        let mut rng = SplitMix64::new(seed);
        let mut result = CellResult {
            full_space: CoverageTally::new(),
            active_face: CoverageTally::new(),
            truncated: CoverageTally::new(),
            truncated_mean_centred: CoverageTally::new(),
            pinned_fraction: 0.0,
        };
        let mean_response = design.dot(truth);
        let mut pinned = 0usize;

        for _ in 0..replicates {
            let mut response = Array1::<f64>::zeros(n);
            for i in 0..n {
                response[i] = mean_response[i] + noise_sd * rng.normal();
            }
            let rhs = design.t().dot(&response);
            let start = crate::active_set::feasible_point_for_linear_constraints(constraints, p)
                .expect("the simulation cone has an interior");
            let (beta_hat, _) = crate::active_set::solve_quadratic_with_linear_constraints(
                &gram,
                &rhs,
                &start,
                constraints,
                None,
            )
            .expect("constrained quadratic solve");

            let full_half_width =
                NOMINAL_HALF_WIDTH_MULTIPLIER * covariance[[reported_index, reported_index]].sqrt();
            result.full_space.record(
                beta_hat[reported_index],
                full_half_width,
                truth[reported_index],
            );

            let tight = tight_rows_at(constraints, &beta_hat);
            if !tight.is_empty() {
                pinned += 1;
            }
            let face_variance =
                active_face_variance(&covariance, constraints, &tight, reported_index);
            result.active_face.record(
                beta_hat[reported_index],
                NOMINAL_HALF_WIDTH_MULTIPLIER * face_variance.max(0.0).sqrt(),
                truth[reported_index],
            );

            // `β_unc = β̂ − Σ ∇ℓ_p(β̂)` with `∇ℓ_p(β̂) = XᵀXβ̂ − Xᵀy`. For this
            // Gaussian cell that is exactly the unconstrained least-squares
            // solution — the centre a truncated Gaussian keeps.
            let penalized_gradient = gram.dot(&beta_hat) - &rhs;
            let center = &beta_hat
                - &(covariance.dot(&penalized_gradient) / (noise_sd * noise_sd));
            let correction =
                constrained_posterior_correction_from_covariance(&covariance, &center, constraints)
                    .expect("truncated correction");
            let (truncated_half_width, truncated_center) = match correction {
                None => (full_half_width, beta_hat[reported_index]),
                Some(ref correction) => {
                    let variance = covariance[[reported_index, reported_index]]
                        - correction.removed_variance_diagonal()[reported_index];
                    (
                        NOMINAL_HALF_WIDTH_MULTIPLIER * variance.max(0.0).sqrt(),
                        correction.posterior_mean(&center)[reported_index],
                    )
                }
            };
            result.truncated.record(
                beta_hat[reported_index],
                truncated_half_width,
                truth[reported_index],
            );
            result.truncated_mean_centred.record(
                truncated_center,
                truncated_half_width,
                truth[reported_index],
            );
        }
        result.pinned_fraction = pinned as f64 / replicates as f64;
        result
    }

    fn report_cell(label: &str, cell: &CellResult) {
        eprintln!(
            "[#2417 coverage] {label}: nominal {NOMINAL_COVERAGE:.2}, {} replicates, mode pinned \
             in {:.1}% of them",
            cell.full_space.replicates,
            100.0 * cell.pinned_fraction
        );
        for (name, tally) in [
            ("full space          ", &cell.full_space),
            ("active face         ", &cell.active_face),
            ("truncated           ", &cell.truncated),
            ("truncated+mean shift", &cell.truncated_mean_centred),
        ] {
            eprintln!(
                "[#2417 coverage]   {name} coverage {:.4}  mean half-width {:.5}",
                tally.coverage(),
                tally.mean_half_width()
            );
        }
    }

    /// A single box bound with the truth half a standard error inside the
    /// feasible region: the regime where the constrained mode pins in about a
    /// third of replicates, so the active-face answer reports a ZERO-WIDTH
    /// interval a third of the time and cannot possibly cover.
    #[test]
    fn box_bound_at_half_a_standard_error_separates_the_three_covariances() {
        let n = 60;
        let mut rng = SplitMix64::new(20_417);
        let mut design = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            design[[i, 1]] = rng.normal();
        }
        let gram = design.t().dot(&design);
        let noise_sd = 1.0;
        let covariance = gaussian_posterior_covariance(&gram, noise_sd * noise_sd);
        let standard_error = covariance[[1, 1]].sqrt();
        let truth = Array1::from_vec(vec![0.3, 0.5 * standard_error]);
        let constraints =
            LinearInequalityConstraints::new(ndarray::array![[0.0, 1.0]], ndarray::array![0.0])
                .expect("nonnegativity bound");

        let cell = run_cell(&design, &truth, &constraints, 1, noise_sd, 4000, 91_137);
        report_cell("box bound, truth 0.5 se", &cell);

        // The mode pins often enough that a zero-width interval is not a corner
        // case; if it stopped pinning the cell would stop testing anything.
        assert!(
            cell.pinned_fraction > 0.2,
            "the cell must actually exercise the boundary, pinned fraction {:.3}",
            cell.pinned_fraction
        );
        assert!(
            cell.active_face.coverage() < 0.80,
            "the active-face covariance must under-cover catastrophically here — it reports a \
             zero-width interval whenever the mode pins — but coverage was {:.4}",
            cell.active_face.coverage()
        );
        assert!(
            cell.truncated.coverage() >= NOMINAL_COVERAGE - 0.01,
            "the truncated covariance must reach nominal coverage, got {:.4}",
            cell.truncated.coverage()
        );
        assert!(
            cell.truncated_mean_centred.coverage() >= NOMINAL_COVERAGE - 0.01,
            "and it must still reach it once the centre moves to the truncated posterior \
             mean, got {:.4}",
            cell.truncated_mean_centred.coverage()
        );
        assert!(
            cell.truncated.mean_half_width() < 0.85 * cell.full_space.mean_half_width(),
            "the truncated covariance must buy its coverage with materially SHORTER intervals \
             than the full-space answer: {:.5} vs {:.5}",
            cell.truncated.mean_half_width(),
            cell.full_space.mean_half_width()
        );
        assert!(
            cell.full_space.coverage() >= NOMINAL_COVERAGE,
            "the full-space covariance over-covers by construction, got {:.4}",
            cell.full_space.coverage()
        );
    }

    /// The truth pushed further from the bound, where the mode is pinned less
    /// often but is much further from the truth when it is. This cell is the
    /// counterexample to narrowing the covariance ALONE.
    ///
    /// Measured here: the truncated covariance around the constrained MODE
    /// covers 0.873 against a nominal 0.95 — worse than both the full-space
    /// answer (0.976) and the active face (0.910) — while the SAME covariance
    /// around the truncated posterior MEAN covers 0.966 at an interval 16%
    /// shorter than full space. Truncating the spread without moving the
    /// location keeps the interval centred on a point the posterior says is its
    /// least-likely feasible value, then makes it narrower. The two halves of
    /// the estimand are not separable, and this test exists so that fact cannot
    /// be lost: a covariance-only change is a REGRESSION here.
    #[test]
    fn narrowing_the_covariance_without_moving_the_mean_is_a_regression() {
        let n = 60;
        let mut rng = SplitMix64::new(31_417);
        let mut design = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            design[[i, 1]] = rng.normal();
        }
        let gram = design.t().dot(&design);
        let noise_sd = 1.0;
        let covariance = gaussian_posterior_covariance(&gram, noise_sd * noise_sd);
        let standard_error = covariance[[1, 1]].sqrt();
        let truth = Array1::from_vec(vec![-0.2, 1.5 * standard_error]);
        let constraints =
            LinearInequalityConstraints::new(ndarray::array![[0.0, 1.0]], ndarray::array![0.0])
                .expect("nonnegativity bound");

        let cell = run_cell(&design, &truth, &constraints, 1, noise_sd, 4000, 47_903);
        report_cell("box bound, truth 1.5 se", &cell);

        assert!(
            cell.truncated.coverage() < NOMINAL_COVERAGE - 0.02,
            "this cell exists BECAUSE the mode-centred truncated interval under-covers here; \
             if it stopped doing so the counterexample would no longer be testing anything, \
             got {:.4}",
            cell.truncated.coverage()
        );
        assert!(
            cell.truncated.coverage() < cell.active_face.coverage(),
            "the point of the cell: narrowing the covariance while leaving the interval \
             centred on the mode is worse than the active-face answer it replaces, {:.4} vs \
             {:.4}",
            cell.truncated.coverage(),
            cell.active_face.coverage()
        );
        assert!(
            cell.truncated_mean_centred.coverage() >= NOMINAL_COVERAGE - 0.01,
            "moving the centre to the truncated posterior mean recovers nominal coverage with \
             the same covariance, got {:.4}",
            cell.truncated_mean_centred.coverage()
        );
        assert!(
            cell.truncated_mean_centred.mean_half_width() < cell.full_space.mean_half_width(),
            "and it does so with shorter intervals than the full-space answer: {:.5} vs {:.5}",
            cell.truncated_mean_centred.mean_half_width(),
            cell.full_space.mean_half_width()
        );
    }

    /// Two coupled bounds, so the correction runs through the multivariate
    /// orthant cubature rather than the scalar closed form.
    #[test]
    fn two_coupled_bounds_exercise_the_orthant_cubature() {
        let n = 80;
        let mut rng = SplitMix64::new(74_211);
        let mut design = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            let shared = rng.normal();
            design[[i, 1]] = shared;
            // Correlated with column 1, so the two bounds are coupled and the
            // constraint-normal covariance `W` is not diagonal.
            design[[i, 2]] = 0.7 * shared + 0.7 * rng.normal();
        }
        let gram = design.t().dot(&design);
        let noise_sd = 1.0;
        let covariance = gaussian_posterior_covariance(&gram, noise_sd * noise_sd);
        let truth = Array1::from_vec(vec![
            0.25,
            0.5 * covariance[[1, 1]].sqrt(),
            0.5 * covariance[[2, 2]].sqrt(),
        ]);
        let constraints = LinearInequalityConstraints::new(
            ndarray::array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ndarray::array![0.0, 0.0],
        )
        .expect("two nonnegativity bounds");

        let cell = run_cell(&design, &truth, &constraints, 1, noise_sd, 600, 55_301);
        report_cell("two coupled bounds, truth 0.5 se", &cell);

        assert!(
            cell.active_face.coverage() < 0.85,
            "the active-face covariance must under-cover here too, got {:.4}",
            cell.active_face.coverage()
        );
        assert!(
            cell.truncated.coverage() >= NOMINAL_COVERAGE - 0.03,
            "the truncated covariance must reach nominal coverage through the orthant \
             cubature, got {:.4}",
            cell.truncated.coverage()
        );
        assert!(
            cell.truncated.mean_half_width() < cell.full_space.mean_half_width(),
            "shorter intervals at nominal coverage: {:.5} vs {:.5}",
            cell.truncated.mean_half_width(),
            cell.full_space.mean_half_width()
        );
    }

}

#[cfg(test)]
mod affine_ceiling_tests {
    use super::*;
    use ndarray::array;

    /// Run the cubature over `{u >= 0}` intersected with an optional affine
    /// wall, and return the log mass.
    fn log_mass(
        mean: &Array1<f64>,
        sd: &Array1<f64>,
        ceiling: Option<&StandardizedCeiling>,
        upper: &[f64],
        nodes: usize,
    ) -> Option<f64> {
        let q = mean.len();
        let mut factor = Array2::<f64>::zeros((q, q));
        for i in 0..q {
            factor[[i, i]] = sd[i];
        }
        let generator = kronecker_generator(q);
        let mut accumulator = OrthantAccumulator::new(q);
        accumulate_orthant_nodes(
            &mut accumulator,
            mean,
            upper,
            ceiling,
            factor.view(),
            &generator,
            None,
            0,
            0,
            nodes,
        )
        .expect("cubature");
        // The accumulator carries its scale separately so a face spanning
        // hundreds of decades never underflows; the mass is that scale times the
        // accumulated sum, averaged over the nodes evaluated.
        if !(accumulator.weight_sum.is_finite() && accumulator.weight_sum > 0.0) {
            return None;
        }
        Some(accumulator.log_scale + accumulator.weight_sum.ln() - (nodes as f64).ln())
    }

    fn diagonal_factor(sd: &Array1<f64>) -> Array2<f64> {
        let q = sd.len();
        let mut factor = Array2::<f64>::zeros((q, q));
        for i in 0..q {
            factor[[i, i]] = sd[i];
        }
        factor
    }

    #[test]
    fn a_coordinate_normal_reproduces_the_box_it_is_the_degenerate_case_of() {
        // The box ceiling and an affine wall whose normal is a single
        // coordinate describe the SAME region. They are computed differently on
        // purpose — the box keeps its subtraction-free width — so this asserts
        // they agree, which is the property that lets one arithmetic path serve
        // both.
        let mean = array![0.35, -0.20];
        let sd = array![1.0, 0.8];
        let factor = diagonal_factor(&sd);
        let width = 1.4;

        let boxed = log_mass(&mean, &sd, None, &[width, f64::INFINITY], 1 << 14)
            .expect("box mass");
        let wall = StandardizedCeiling::new(&array![1.0, 0.0], width, &mean, factor.view())
            .expect("coordinate wall");
        let affine = log_mass(
            &mean,
            &sd,
            Some(&wall),
            &[f64::INFINITY, f64::INFINITY],
            1 << 14,
        )
        .expect("affine mass");
        assert_eq!(wall.pivot, 0, "a normal touching only coordinate 0 pivots there");
        assert!(
            (boxed - affine).abs() < 1e-12,
            "box {boxed:.15} and affine {affine:.15} describe the same region"
        );
    }

    #[test]
    fn an_affine_ceiling_removes_the_mass_it_should() {
        // Region: {u >= 0} ∩ {u0 + u1 <= c}, a triangle. With independent
        // coordinates the answer is a one-dimensional integral, which Simpson
        // resolves to far better than the cubature's own tolerance — an
        // independent reference rather than a second run of the same rule.
        let mean = array![0.4, -0.3];
        let sd = array![0.9, 1.1];
        let factor = diagonal_factor(&sd);
        let bound = 1.6;
        let wall = StandardizedCeiling::new(&array![1.0, 1.0], bound, &mean, factor.view())
            .expect("sum wall");
        assert_eq!(wall.pivot, 1, "a wall touching both coordinates pivots on the last");

        let got = log_mass(
            &mean,
            &sd,
            Some(&wall),
            &[f64::INFINITY, f64::INFINITY],
            1 << 16,
        )
        .expect("triangle mass");

        // Simpson over u0 in [0, bound] of  φ((u0−m0)/s0)/s0 · P(0 ≤ u1 ≤ bound−u0)
        let panels = 4000usize;
        let step = bound / panels as f64;
        let density = |x: f64| {
            let z = (x - mean[0]) / sd[0];
            (-0.5 * z * z).exp() / (sd[0] * (2.0 * std::f64::consts::PI).sqrt())
        };
        let inner = |x: f64| {
            let hi = (bound - x - mean[1]) / sd[1];
            let lo = -mean[1] / sd[1];
            if hi <= lo {
                0.0
            } else {
                normal_cdf(hi) - normal_cdf(lo)
            }
        };
        let integrand = |x: f64| density(x) * inner(x);
        let mut total = integrand(0.0) + integrand(bound);
        for k in 1..panels {
            let x = k as f64 * step;
            total += integrand(x) * if k % 2 == 0 { 2.0 } else { 4.0 };
        }
        let reference = (total * step / 3.0).ln();
        assert!(
            (got - reference).abs() < 5e-4,
            "cubature {got:.12} against the Simpson reference {reference:.12}"
        );

        // ... and the wall must actually bite: the same region without it is
        // strictly larger. A ceiling that changed nothing would pass the check
        // above just as well.
        let unbounded = log_mass(
            &mean,
            &sd,
            None,
            &[f64::INFINITY, f64::INFINITY],
            1 << 16,
        )
        .expect("unbounded mass");
        assert!(
            unbounded > got + 0.05,
            "the wall removed {:.4} nats, which is not enough to call it active",
            unbounded - got
        );
    }

    #[test]
    fn a_wall_that_crosses_the_orthant_leaves_no_mass() {
        // `u0 + u1 <= -1` is disjoint from the closed orthant, so every node's
        // interval is empty and the cubature reports no feasible mass rather
        // than a small wrong one.
        let mean = array![0.2, 0.1];
        let sd = array![1.0, 1.0];
        let factor = diagonal_factor(&sd);
        let wall = StandardizedCeiling::new(&array![1.0, 1.0], -1.0, &mean, factor.view())
            .expect("infeasible wall");
        assert!(
            log_mass(&mean, &sd, Some(&wall), &[f64::INFINITY, f64::INFINITY], 1 << 10).is_none(),
            "an empty region reports no mass"
        );
    }

    #[test]
    fn a_vanishing_normal_is_refused_rather_than_pivoted_arbitrarily() {
        let mean = array![0.0, 0.0];
        let factor = diagonal_factor(&array![1.0, 1.0]);
        let message = StandardizedCeiling::new(&array![0.0, 0.0], 1.0, &mean, factor.view())
            .expect_err("a zero normal constrains nothing");
        assert!(
            message.contains("vanished"),
            "the refusal must say the normal vanished, got: {message}"
        );
        let mismatched = StandardizedCeiling::new(&array![1.0], 1.0, &mean, factor.view())
            .expect_err("a normal of the wrong length is refused");
        assert!(mismatched.contains("length"), "got: {mismatched}");
    }

    #[test]
    fn the_pivot_follows_the_factor_not_just_the_normal() {
        // `Lᵀa` is what decides the pivot, so a normal touching only coordinate
        // 0 can still reach earlier coordinates through a dense factor — but
        // never a LATER one, because L is lower triangular. This pins the
        // direction of the transform, which a transpose slip would invert.
        let mean = array![0.0, 0.0, 0.0];
        let mut factor = Array2::<f64>::zeros((3, 3));
        factor[[0, 0]] = 1.0;
        factor[[1, 0]] = 0.7;
        factor[[1, 1]] = 1.0;
        factor[[2, 0]] = 0.3;
        factor[[2, 1]] = 0.4;
        factor[[2, 2]] = 1.0;
        let wall = StandardizedCeiling::new(&array![0.0, 0.0, 1.0], 2.0, &mean, factor.view())
            .expect("last-coordinate normal");
        assert_eq!(wall.pivot, 2);
        let early = StandardizedCeiling::new(&array![1.0, 0.0, 0.0], 2.0, &mean, factor.view())
            .expect("first-coordinate normal");
        assert_eq!(
            early.pivot, 0,
            "a normal on coordinate 0 cannot reach a later coordinate through a lower-triangular factor"
        );
    }
}

#[cfg(test)]
mod projection_law_2446_tests {
    use super::*;
    use ndarray::array;

    /// The integrand a locscale response-moment consumer actually evaluates:
    /// a survival probability read off an inverse link at `eta0 + w`. Smooth,
    /// bounded, and monotone, so nothing about the comparison below depends on
    /// picking an integrand that flatters the mixture.
    fn integrand(w: f64) -> f64 {
        1.0 / (1.0 + (-0.5 + w).exp())
    }

    /// Composite Simpson on `[0, upper]`, odd `points`.
    fn simpson<F: Fn(f64) -> f64>(lower: f64, upper: f64, points: usize, f: F) -> f64 {
        assert!(points % 2 == 1, "Simpson needs an odd point count");
        let h = (upper - lower) / ((points - 1) as f64);
        let mut total = 0.0;
        for index in 0..points {
            let weight = if index == 0 || index == points - 1 {
                1.0
            } else if index % 2 == 1 {
                4.0
            } else {
                2.0
            };
            total += weight * f(lower + h * (index as f64));
        }
        total * h / 3.0
    }

    /// Same reference for an arbitrary scalar functional of `cᵀβ`.
    fn exact_orthant_expectation_of<F: Fn(f64) -> f64>(
        center: &Array1<f64>,
        covariance: &Array2<f64>,
        contrast: &Array1<f64>,
        functional: F,
    ) -> f64 {
        let det = covariance[[0, 0]] * covariance[[1, 1]] - covariance[[0, 1]] * covariance[[1, 0]];
        let inverse = array![
            [covariance[[1, 1]] / det, -covariance[[0, 1]] / det],
            [-covariance[[1, 0]] / det, covariance[[0, 0]] / det]
        ];
        let density = |b0: f64, b1: f64| -> f64 {
            let d0 = b0 - center[0];
            let d1 = b1 - center[1];
            let quadratic = inverse[[0, 0]] * d0 * d0
                + 2.0 * inverse[[0, 1]] * d0 * d1
                + inverse[[1, 1]] * d1 * d1;
            (-0.5 * quadratic).exp()
        };
        // 12 sd past the wall in each coordinate leaves `exp(-72)` of the mass
        // outside the box, which is below the reference's own quadrature error
        // by more than twenty orders.
        let upper0 = center[0].max(0.0) + 12.0 * covariance[[0, 0]].sqrt();
        let upper1 = center[1].max(0.0) + 12.0 * covariance[[1, 1]].sqrt();
        let points = 2001;
        let mass = simpson(0.0, upper0, points, |b0| {
            simpson(0.0, upper1, points, |b1| density(b0, b1))
        });
        let weighted = simpson(0.0, upper0, points, |b0| {
            simpson(0.0, upper1, points, |b1| {
                density(b0, b1) * functional(contrast[0] * b0 + contrast[1] * b1)
            })
        });
        weighted / mass
    }

    /// `E[f(w)]` under `N(mean, variance)` on the WHOLE line — the law the
    /// locscale warp integral ships today: the normal carrying the constrained
    /// posterior's first two moments.
    fn normal_expectation(mean: f64, variance: f64) -> f64 {
        let sd = variance.sqrt();
        let points = 4001;
        simpson(mean - 12.0 * sd, mean + 12.0 * sd, points, |w| {
            let z = (w - mean) / sd;
            (-0.5 * z * z).exp() * integrand(w)
        }) / (sd * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// #2679: the same claim for the JOINT rule, on a fixture whose contrast is
    /// NOT carried entirely by the constraint normals.
    ///
    /// The law is `x = cᵀu + s·t` with `u` the two-row truncated cone above and
    /// `t` an independent standard normal — the exact structure a response
    /// moment sees, where the cone moves the whole coefficient vector and the
    /// tangent adds the part of the predictor's variance the constraint normals
    /// do not carry.
    ///
    /// The reference folds the tangent into the functional analytically-in-form
    /// (`g(x) = E_t[f(x + s·t)]`, a 1-D Gaussian Simpson) and then integrates
    /// `g` against the exact truncated density by the SAME tensor Simpson rule
    /// the #2446 test uses. Neither half calls the cubature under test.
    ///
    /// Three separate properties are asserted, because the rule has to have all
    /// three before it can price a moment on a predict path:
    ///
    /// * **support** — every point is feasible, exactly, not to a tolerance;
    /// * **tangent measure** — the tangent block really is standard normal
    ///   under the SOV importance weights, which is free to fail (the weights
    ///   are a function of the constraint-normal coordinates of the same
    ///   lattice point, so an aliased generator would correlate them);
    /// * **accuracy** — against the density reference, and by a wide margin
    ///   over the moment-matched normal that is what ships today.
    #[test]
    fn joint_cubature_carries_the_tangent_block_at_a_bounded_point_count_2679() {
        let ambient = array![[0.40, 0.24], [0.24, 0.36]];
        let center = array![0.05, -0.10];
        let contrast = array![0.70, 0.30];
        // Tangent share of the predictor. Large enough that a rule which simply
        // dropped the tangent would miss by far more than the bound below.
        let tangent_sd = 0.6_f64;
        let constraints =
            LinearInequalityConstraints::new(array![[1.0, 0.0], [0.0, 1.0]], array![0.0, 0.0])
                .expect("build the two-row non-negativity cone");
        let correction =
            constrained_posterior_correction_from_covariance(&ambient, &center, &constraints)
                .expect("the correction is computable on this face")
                .expect("a centre straddling both walls must retain the face");
        let mut retained = correction.rows.clone();
        retained.sort_unstable();
        assert_eq!(
            retained,
            vec![0, 1],
            "the fixture must retain BOTH rows or the pushforward is the closed-form case"
        );

        // `A = I`, so the constraint-normal coordinate IS the coefficient
        // vector and `W = Σ`, `E_untrunc[u] = centre`.
        let upper_limits = correction.upper_limits();
        let normal_center = Array1::from_vec(
            correction
                .rows
                .iter()
                .map(|&row| center[row])
                .collect::<Vec<_>>(),
        );
        let normal_covariance = {
            let mut out = Array2::<f64>::zeros((2, 2));
            for (i, &row_i) in correction.rows.iter().enumerate() {
                for (j, &row_j) in correction.rows.iter().enumerate() {
                    out[[i, j]] = ambient[[row_i, row_j]];
                }
            }
            out
        };
        let lift_contrast = Array1::from_vec(
            correction
                .rows
                .iter()
                .map(|&row| contrast[row])
                .collect::<Vec<_>>(),
        );

        const POINTS: usize = 1 << 13;
        let joint = constrained_posterior_joint_cubature(
            &normal_center,
            &normal_covariance,
            &upper_limits,
            1,
            POINTS,
        )
        .expect("joint cubature on a retained two-row face");
        assert_eq!(
            joint.len(),
            POINTS,
            "the joint rule's cost is the point count it was asked for and nothing else"
        );

        // (a) support. Exact, not to a tolerance: the SOV map cannot produce an
        // infeasible point, so any tolerance here would be hiding a bug.
        let infeasible_points = joint
            .iter()
            .filter(|point| point.normal_coordinates.iter().any(|&value| value < 0.0))
            .count();
        assert_eq!(
            infeasible_points, 0,
            "every joint point must lie in the retained cone; {infeasible_points} of {POINTS} did \
             not"
        );

        let weight_sum = joint.iter().map(|point| point.weight).sum::<f64>();
        assert!(
            (weight_sum - 1.0).abs() < 1e-9,
            "joint weights must be normalized, got {weight_sum:.12e}"
        );

        // (b) the tangent block is standard normal under the SAME weights.
        let tangent_mean = joint
            .iter()
            .map(|point| point.weight * point.tangent[0])
            .sum::<f64>();
        let tangent_second = joint
            .iter()
            .map(|point| point.weight * point.tangent[0] * point.tangent[0])
            .sum::<f64>();
        eprintln!(
            "[2679] points={POINTS} tangent_mean={tangent_mean:.6e} \
             tangent_second={tangent_second:.6e}"
        );
        assert!(
            tangent_mean.abs() < 2.0e-2,
            "the tangent block must integrate to a zero mean under the SOV weights, got \
             {tangent_mean:.6e}"
        );
        assert!(
            (tangent_second - 1.0).abs() < 5.0e-2,
            "the tangent block must integrate to unit variance under the SOV weights, got \
             {tangent_second:.6e}"
        );

        let joint_value = joint
            .iter()
            .map(|point| {
                let normal_part = lift_contrast.dot(&point.normal_coordinates);
                point.weight * integrand(normal_part + tangent_sd * point.tangent[0])
            })
            .sum::<f64>();

        // (c) accuracy against a reference built from the density.
        let convolved = |x: f64| -> f64 {
            simpson(
                x - 12.0 * tangent_sd,
                x + 12.0 * tangent_sd,
                4001,
                |value| {
                    let z = (value - x) / tangent_sd;
                    (-0.5 * z * z).exp() * integrand(value)
                },
            ) / (tangent_sd * (2.0 * std::f64::consts::PI).sqrt())
        };
        let reference = exact_orthant_expectation_of(&center, &ambient, &contrast, convolved);

        // What ships today: the moment-matched normal, with the tangent's
        // variance folded into it — which is exactly how the locscale rule
        // treats the same decomposition.
        let posterior_mean = contrast.dot(&correction.posterior_mean(&center));
        let corrected = correction.apply_to_covariance(&ambient);
        let posterior_variance =
            contrast.dot(&corrected.dot(&contrast)) + tangent_sd * tangent_sd;
        let normal_value = normal_expectation(posterior_mean, posterior_variance);

        let joint_error = (joint_value - reference).abs();
        let normal_error = (normal_value - reference).abs();
        eprintln!(
            "[2679] reference={reference:.12e} joint={joint_value:.12e} (err {joint_error:.3e}) \
             normal={normal_value:.12e} (err {normal_error:.3e})"
        );
        assert!(
            normal_error > 1.0e-4,
            "the fixture must leave the moment-matched normal measurably wrong, or the \
             comparison below is vacuous; got {normal_error:.3e}"
        );
        assert!(
            joint_error < 0.2 * normal_error,
            "the joint rule must be decisively closer to the exact pushforward than the \
             moment-matched normal: joint error {joint_error:.3e} vs normal error \
             {normal_error:.3e} against reference {reference:.12e}"
        );
    }
}
