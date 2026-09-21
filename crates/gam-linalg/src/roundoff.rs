//! Derived floating-point roundoff bands.
//!
//! Numerical code is full of the question "is this quantity indistinguishable
//! from zero?" — a negative eigenvalue that should be a zero one, a residual
//! that should be a converged one, a directional derivative that should be a
//! flat one. The answer is not a taste parameter: it is the backward-error band
//! of the arithmetic that produced the quantity, and that band is a function of
//! how many operations were accumulated and how large the accumulated terms
//! were. Both are known at the call site.
//!
//! The bound is Wilkinson's, in the form given by Higham (*Accuracy and
//! Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Lemma 3.1 and §3.1):
//!
//! ```text
//! |fl(s) − s|  ≤  γ_k · Σ|terms|,      γ_k = k·u / (1 − k·u),
//! ```
//!
//! with `u` the unit roundoff and `k` the number of rounded operations on an
//! accumulation path. Writing a bare multiple of `EPSILON` instead substitutes
//! a constant for `γ_k`, which is wrong in both directions as the problem size
//! moves — too tight for long accumulations, so exact-arithmetic-zero
//! quantities get rejected as materially nonzero, and needlessly loose for
//! short ones.
//!
//! # `k` is not always `n`
//!
//! The two accumulations that look alike carry different depths, and conflating
//! them is the easiest way to get a bound that is wrong by one factor of `u`:
//!
//! * An **inner product** of length `n` forms `n` products and then sums them
//!   with `n − 1` additions, so `k = n` — this is [`accumulation_band`].
//! * A **sum of terms that are already formed** commits only the `n − 1`
//!   additions, so `k = n − 1`: a caller holding `Σ|xᵢ|` and the term count
//!   passes `n - 1` to [`accumulation_growth`] directly rather than using
//!   [`accumulation_band`].
//! * A **pairwise (tree) reduction** has depth `⌈log₂ n⌉` rather than `n − 1`;
//!   that is a genuinely tighter bound and the right `k` whenever the
//!   reduction is pairwise.
//! * A **compensated (Kahan/Neumaier) sum** has a bound with no `n` in it at
//!   all — [`compensated_band`].

// Owned by gam-math, the lowest crate, so the math crate's bounds read the same definitions.
pub use gam_math::roundoff::{UNIT_ROUNDOFF, accumulation_growth};

/// Backward-error band of an inner product of length `terms` whose summands
/// have absolute sum `absolute_sum`: `γ_terms · absolute_sum`.
///
/// This is the magnitude below which the computed result is indistinguishable
/// from zero: an inner product whose exact value is zero can be computed as
/// anything within `±accumulation_band(n, Σ|xᵢyᵢ|)`, and nothing outside it.
///
/// The depth is `terms`, not `terms − 1`, because forming each product rounds
/// once before any addition happens. For a sum of pre-formed terms see the
/// module documentation.
pub fn accumulation_band(terms: usize, absolute_sum: f64) -> f64 {
    accumulation_growth(terms) * absolute_sum
}

/// How the two triangles of a square matrix handed to a strict symmetric
/// routine were produced (#4350).
///
/// A strict routine (a certified solve, an unjittered Cholesky, a self-adjoint
/// eigendecomposition) reads one triangle, so it must first establish that the
/// other one says the same thing. How far the two may legitimately disagree is
/// not a property of the matrix: it is a property of the arithmetic that
/// assembled it, which only the caller can see. A fixed allowance in ULPs
/// cannot be right, because the legitimate disagreement is exactly zero for a
/// mirrored matrix and grows linearly with the accumulation length for a full
/// GEMM. [`symmetric_assembly_band`] turns this provenance into the band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetricAssembly {
    /// Every off-diagonal value was rounded once and written to both
    /// triangles: a triangular accumulation mirrored across the diagonal
    /// (`fast_ata`, `fast_xt_diag_x`), an explicit `(M + Mᵀ)/2`, an entrywise
    /// sum or scaling of such matrices, or a structurally symmetric
    /// construction (diagonal, tridiagonal, a Gram written once per pair).
    ///
    /// IEEE-754 addition and multiplication are commutative and correctly
    /// rounded, so the same operations on the same operands give the same bits:
    /// every such matrix is bitwise symmetric. The band is exactly zero, and
    /// any disagreement is a construction defect rather than rounding.
    Mirrored,
    /// Each triangle is its own floating-point accumulation of
    /// positive-semidefinite pieces: the rows `w_k·x_k·x_kᵀ` (`w_k ≥ 0`) of a
    /// weighted Gram computed by a full GEMM, plus any PSD blocks (penalties,
    /// ridges) added on top, with at most `depth` rounded operations on any
    /// entry's accumulation path.
    ///
    /// `depth` is counted as [`accumulation_band`] counts it: an `n`-row Gram
    /// forms each product in one rounding (two when a weight multiplies in) and
    /// sums the `n` products in `n − 1` additions, and each further matrix
    /// added onto the result is one more addition on every entry's path.
    PsdAccumulation { depth: usize },
}

impl SymmetricAssembly {
    /// The assembly of a penalized Gram `XᵀWX + EᵀE`.
    ///
    /// `gram_rows` counts the rows whose products a NON-mirroring route
    /// accumulates into each entry, and `penalty_root_rows` the rows of the
    /// penalty root `E`, whose Gram is a full GEMM. Each row's weighted product
    /// rounds twice (`w_k·x_ki`, then `·x_kj`) and the rows are summed in
    /// `terms − 1` additions, which is [`weighted_gram_assembly_band`]'s depth
    /// with `formation_roundings = 2`.
    ///
    /// Pass `0` for `gram_rows` when the design half was built by
    /// `fast_xt_diag_x`, which accumulates one triangle and mirrors it, so both
    /// triangles carry identical bits and that half can leave no disagreement
    /// at all; the penalty Gram is then the only term that can.
    pub const fn penalized_gram(gram_rows: usize, penalty_root_rows: usize) -> Self {
        let terms = gram_rows.saturating_add(penalty_root_rows);
        Self::PsdAccumulation {
            depth: terms.saturating_sub(1).saturating_add(2),
        }
    }

    /// The assembly after every entry is multiplied by one scalar: one more
    /// rounding on an accumulated entry's path, while a mirrored matrix stays
    /// bitwise symmetric (the same product on the same operand in both triangles).
    pub const fn scaled(self) -> Self {
        match self {
            Self::Mirrored => Self::Mirrored,
            Self::PsdAccumulation { depth } => Self::PsdAccumulation {
                depth: depth.saturating_add(1),
            },
        }
    }

    /// The assembly of the entrywise sum with another symmetric PSD matrix.
    /// Two mirrored operands sum to a mirrored matrix; an accumulation gains one
    /// addition on every entry's path, plus the other operand's own path when
    /// that operand is itself an accumulation.
    pub const fn psd_sum(self, other: Self) -> Self {
        match (self, other) {
            (Self::Mirrored, Self::Mirrored) => Self::Mirrored,
            (Self::PsdAccumulation { depth }, Self::Mirrored)
            | (Self::Mirrored, Self::PsdAccumulation { depth }) => Self::PsdAccumulation {
                depth: depth.saturating_add(1),
            },
            (Self::PsdAccumulation { depth: a }, Self::PsdAccumulation { depth: b }) => {
                Self::PsdAccumulation {
                    depth: a.saturating_add(b).saturating_add(1),
                }
            }
        }
    }
}

/// The largest disagreement `|A_ij − A_ji|` the assembly `assembly` can leave
/// between the two triangles of a matrix whose computed diagonal entries at
/// rows `i` and `j` are `diagonal_i` and `diagonal_j` (#4350).
///
/// This is the single definition every strict symmetric routine reads.
///
/// * [`SymmetricAssembly::Mirrored`]: `0` — both triangles hold one rounded
///   value.
/// * [`SymmetricAssembly::PsdAccumulation`]: both triangles accumulate the
///   same summands `t_ij = t_ji` (the pieces are symmetric), each with error
///   at most `γ_d·Σ_t |t_ij|` (Higham, *ASNA* 2nd ed., §3.1), so
///   `|fl(A_ij) − fl(A_ji)| ≤ 2γ_d·Σ_t |t_ij|`. Every piece `T` is PSD, so
///   `|T_ij| ≤ √(T_ii·T_jj)`, and Cauchy–Schwarz over the pieces gives
///   `Σ_t √(T_t,ii·T_t,jj) ≤ √(Σ_t T_t,ii · Σ_t T_t,jj) = √(A_ii·A_jj)` in exact
///   arithmetic. The diagonal summands are all non-negative, so the computed
///   diagonal underestimates the exact one by at most a factor `1 − γ_d`, and
///
///   ```text
///   |fl(A_ij) − fl(A_ji)|  ≤  2γ_d · √(fl(A_ii)·fl(A_jj)) / (1 − γ_d).
///   ```
///
///   The bound is scale-covariant (`A ↦ cA` scales it by `c`) and local to the
///   `(i, j)` block, so a well-scaled block inside a badly scaled matrix keeps
///   a tight band; and it is read at the scale the accumulation ran at, not at
///   `|A_ij|`, which is free to cancel to nothing.
pub fn symmetric_assembly_band(
    assembly: SymmetricAssembly,
    diagonal_i: f64,
    diagonal_j: f64,
) -> f64 {
    match assembly {
        SymmetricAssembly::Mirrored => 0.0,
        SymmetricAssembly::PsdAccumulation { depth } => {
            let growth = accumulation_growth(depth);
            if !(growth < 1.0) {
                return f64::INFINITY;
            }
            // Each factor is square-rooted before multiplying so a large finite
            // diagonal cannot overflow the product.
            let scale = diagonal_i.abs().sqrt() * diagonal_j.abs().sqrt();
            2.0 * growth * scale / (1.0 - growth)
        }
    }
}

/// Rounding band `p·ε·‖H‖₂` of a symmetric `p×p` matrix's computed spectrum,
/// read off its eigenvalues.
///
/// A backward-stable symmetric eigensolver returns the exact spectrum of `H + E`
/// with `‖E‖₂ = O(p·ε·‖H‖₂)`, so by Weyl an eigenvalue whose magnitude is at or
/// below this band is not resolved from zero by the decomposition that produced
/// it, and a sign inside it is not a measurement.
pub fn symmetric_spectrum_rounding_band(eigenvalues: &[f64]) -> f64 {
    let spectral_radius = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    eigenvalues.len() as f64 * f64::EPSILON * spectral_radius
}

/// The rank of a symmetric Gram, read off its eigenvalues: those above
/// [`symmetric_spectrum_rounding_band`] plus `assembly_band`.
///
/// `assembly_band` is the caller's bound, in eigenvalue units, on the error the
/// Gram's formation left in the matrix (zero for exact input). A frozen rank and
/// every later trial of the same block read this one predicate. An eigenvalue
/// inside the band is not resolved from zero by the decomposition that produced
/// it, and its sign carries no information.
pub fn resolved_eigenvalue_count(eigenvalues: &[f64], assembly_band: f64) -> usize {
    let band = resolved_eigenvalue_band(eigenvalues, assembly_band);
    eigenvalues.iter().filter(|&&value| value > band).count()
}

/// The threshold [`resolved_eigenvalue_count`] compares against:
/// [`symmetric_spectrum_rounding_band`] plus `assembly_band`. Exposed for
/// callers that act on each resolved eigenpair (a pseudo-inverse inverts
/// exactly the eigenvalues above it) so the rank and the inversion read one
/// predicate.
pub fn resolved_eigenvalue_band(eigenvalues: &[f64], assembly_band: f64) -> f64 {
    symmetric_spectrum_rounding_band(eigenvalues) + assembly_band
}

/// Spectral-norm bound on the rounding a weighted Gram `AᵀWA` picks up when it
/// is formed as the inner products of `terms` rows, each product rounding
/// `formation_roundings` times before the additions.
///
/// Entrywise the error is at most `γ_k·(|A|ᵀ|W||A|)`, `k = terms − 1 +
/// formation_roundings` (Higham, *ASNA* 2nd ed., §3.1). That majorant is PSD,
/// so its spectral norm — and by Perron–Frobenius monotonicity the error's —
/// is bounded by its trace `Σᵢ |wᵢ|·‖aᵢ‖²`, which the caller passes as
/// `weighted_row_norm_sum`.
pub fn weighted_gram_assembly_band(
    terms: usize,
    formation_roundings: usize,
    weighted_row_norm_sum: f64,
) -> f64 {
    accumulation_growth(terms.saturating_sub(1) + formation_roundings) * weighted_row_norm_sum
}

/// Forward-error band of a **compensated** summation (Kahan–Babuška–Neumaier)
/// whose summands have absolute sum `absolute_sum`, where building each summand
/// cost `formation_roundings` floating-point operations.
///
/// Compensated summation satisfies `|fl(S) − S| ≤ 2u·Σ|xᵢ|` **with no
/// dependence on the term count** (Higham, *ASNA* 2nd ed., §4.3) — shedding the
/// `n` is the entire reason to pay for the compensation. Scaling a compensated
/// sum's band by `n` anyway applies the naive-summation model to an algorithm
/// chosen to defeat it, and inflates the band by a factor of `n`.
///
/// `formation_roundings` covers the arithmetic that produces each summand
/// before any addition happens: an inner-product term formed as `(a/p)*(b/q)`
/// costs three. Count the operations at the call site, where they are visible.
pub fn compensated_band(formation_roundings: usize, absolute_sum: f64) -> f64 {
    (2.0 + formation_roundings as f64) * UNIT_ROUNDOFF * absolute_sum
}

/// The absolute summands behind a gradient summed from terms (#2976, #2822).
///
/// A gradient summed from rows carries the rounding of that sum, which scales with
/// the summands' magnitudes and not with the assembled result: near a mode each
/// row's term is `O(1)` while their sum is small. Coordinate `j` of the computed sum
/// is within `accumulation_growth(accumulation_depth) · absolute_sums[j]` of the exact
/// sum of the same computed terms.
pub struct GradientAccumulation {
    /// The sequential depth of the floating-point reduction that sums the terms:
    /// the `m` of the `γ_m` that bands the sum.
    pub accumulation_depth: usize,
    /// `Σ |terms|` per coordinate: every product the reduction adds into that
    /// coordinate, in absolute value.
    pub absolute_sums: ndarray::Array1<f64>,
}

/// Backward-error band on the singular values of an `m × n` factor with largest
/// singular value `sigma_max`.
///
/// A backward-stable SVD returns the exact singular values of `A + δA` with
/// `‖δA‖₂ ≤ max(m, n)·ε·‖A‖₂`, and by Weyl each computed `σᵢ` is within that of
/// the true one. A singular value above the band is resolved. At or below it the
/// decomposition has not separated it from zero.
pub fn factor_singular_band(rows: usize, cols: usize, sigma_max: f64) -> f64 {
    rows.max(cols) as f64 * f64::EPSILON * sigma_max
}

/// Band below which a computed quadratic `fl(vᵀSv)` of a symmetric `p × p`
/// matrix does not separate `v` from `S`'s null space (#4045).
///
/// It has two parts, both read off the operands:
///
/// - **Resolution of `S`.** A symmetric matrix is resolved only to
///   `‖ΔS‖₂ ≤ p·ε·‖S‖₂`, the same resolution
///   [`symmetric_spectrum_rounding_band`] assigns to its spectrum. If `v`
///   lies in the null space of some `S + ΔS` in that ball, then
///   `|vᵀSv| = |vᵀΔSv| ≤ p·ε·‖S‖₂·‖v‖²`. Here `‖S‖₂` is bounded by
///   `‖S‖_∞`, the maximum absolute row sum, which majorizes the spectral
///   radius of a symmetric matrix.
/// - **Evaluation.** Forming `Sv` and then `vᵀ(Sv)` is a length-`2p`
///   accumulation path over `|v|ᵀ|S||v|` ([`accumulation_band`]).
///
/// An error in `v` of size `δ` enters only at second order,
/// `(v+δ)ᵀS(v+δ) = δᵀSδ` for `Sv = 0`, so `v`'s own rounding does not appear.
pub fn null_quadratic_band(
    matrix: ndarray::ArrayView2<'_, f64>,
    vector: ndarray::ArrayView1<'_, f64>,
) -> f64 {
    let p = matrix.nrows();
    let mut operator_scale = 0.0_f64;
    let mut absolute_quadratic = 0.0_f64;
    for (row, &vi) in matrix.rows().into_iter().zip(vector.iter()) {
        let mut row_sum = 0.0_f64;
        let mut row_quadratic = 0.0_f64;
        for (&entry, &vj) in row.iter().zip(vector.iter()) {
            row_sum += entry.abs();
            row_quadratic += entry.abs() * vj.abs();
        }
        operator_scale = operator_scale.max(row_sum);
        absolute_quadratic += vi.abs() * row_quadratic;
    }
    let norm_squared = vector.dot(&vector);
    p as f64 * f64::EPSILON * operator_scale * norm_squared
        + accumulation_band(2 * p, absolute_quadratic)
}

/// Rounded operations one Householder reflector commits on a length-`rows`
/// column, as the `c` of `γ_c` in `fl(P̂b) = (P + ΔP)b`, `‖ΔP‖_F ≤ γ_c`, with `P`
/// the exact reflector of the exact column (Higham, *ASNA* 2nd ed., Lemmas
/// 19.2–19.3).
///
/// The count follows the reflector faer builds (`make_householder_in_place`)
/// and applies (the column-pivoted QR update), `P = I − vvᵀ/τ` with `v₀ = 1`.
/// It uses `m = rows` and Higham's Lemma 3.3: `γ_a + γ_b + γ_aγ_b ≤ γ_{a+b}`
/// and `cγ_a ≤ γ_{ca}`.
///
/// - **The norm.** `‖x_tail‖₂` is an `m`-term sum of squares, the combination
///   of its scaled accumulators and a square root: `γ_{m+2}`. `hypot` with the
///   head adds 2: `γ_{m+4}`.
/// - **The vector.** `h = x₀ + sign(x₀)‖x‖` (no cancellation) adds 1, `1/h`
///   adds 1, and `v_i = x_i·(1/h)` adds 1, so `|Δv_i| ≤ γ_{m+7}|v_i|`.
/// - **The scalar.** `τ = (1 + (‖x_tail‖·|1/h|)²)/2` is a product (`γ_{2m+9}`),
///   a square and a rounding (`γ_{4m+19}`), and an add (`γ_{4m+20}`). The halving
///   is exact, and `1/τ` adds 1, giving `γ_{4m+21}`.
/// - **The rank-one term.** `v_iv_j/τ` carries `v`'s error twice and `1/τ`'s
///   once, `γ_{6m+35}`. The exact term has `‖|v||v|ᵀ/τ‖_F = vᵀv/τ = 2`, so its
///   error is at most `2γ_{6m+35}`.
/// - **The application.** `b_i − v_i·((vᵀb)/τ)` is an `m`-term inner product,
///   a multiply and a fused multiply-add: `γ_{m+2}`. It acts on
///   `|b| + |v̂||v̂|ᵀ|b|/τ̂`, whose norm is at most `(3 + 2γ_{6m+35})‖b‖`.
///
/// In total, `2γ_{6m+35} + 3γ_{m+2} + 2γ_{m+2}γ_{6m+35} ≤ 2γ_{7m+37} + γ_{m+2}`,
/// which is at most `γ_{15m+76}`.
pub const fn householder_reflector_roundings(rows: usize) -> usize {
    rows.saturating_mul(15).saturating_add(76)
}

/// Roundings the column-pivoted QR adds outside its reflectors. It scales the
/// input by the rounded reciprocal of its largest column norm (two roundings
/// per entry, a relative backward perturbation of each column) and scales `R̂`
/// back (one rounding per entry, `‖ΔR‖_F ≤ u‖R‖_F`, which is `QΔR` on the
/// matrix). The power-of-two pre-scaling is exact.
const PIVOTED_QR_SCALING_ROUNDINGS: usize = 3;

/// Normwise backward-error band of this crate's column-pivoted Householder QR
/// of a `rows × cols` matrix whose computed `R̂` has Frobenius norm
/// `r_frobenius` (#4045).
///
/// Householder QR is backward stable. The computed `R̂` is the exact triangular
/// factor of `(A + ΔA)Π = QR̂`, where `Q` is exactly orthogonal and
/// `‖Δa_j‖₂ ≤ γ_K‖a_j‖₂` for every column (Higham, *ASNA* 2nd ed., Thm 19.4,
/// with Lemma 3.7 for the product of the reflectors). Here `K` is the reflector
/// count `min(rows, cols)` times [`householder_reflector_roundings`], plus the
/// three scaling roundings. Column pivoting only reorders the columns, so the
/// bound is unchanged. Summing the columns gives `‖ΔA‖_F ≤ γ_K‖A‖_F`.
///
/// Since `Q` is orthogonal, `‖R̂‖_F = ‖A + ΔA‖_F ≥ ‖A‖_F − ‖ΔA‖_F`, and so
/// `‖ΔA‖_F ≤ γ_K/(1 − γ_K)·‖R̂‖_F`. That is the band returned, read off the
/// factor the caller already holds.
///
/// Modified Gram–Schmidt on an `m × n` matrix is numerically equivalent to
/// Householder QR on `[0_n; A]` (Björck & Paige, 1992). Its band is therefore
/// this one with `rows = m + n`, which also covers MGS's own (fewer) roundings.
///
/// Returns `+∞` once `K·u ≥ 1`, where the bound says nothing.
pub fn householder_qr_backward_band(rows: usize, cols: usize, r_frobenius: f64) -> f64 {
    let operations = rows
        .min(cols)
        .saturating_mul(householder_reflector_roundings(rows))
        .saturating_add(PIVOTED_QR_SCALING_ROUNDINGS);
    let gamma = accumulation_growth(operations);
    if !(gamma < 1.0) {
        return f64::INFINITY;
    }
    gamma / (1.0 - gamma) * r_frobenius
}

/// Rank partition of a quadratic `S = AᵀA`, read off its energy factor `A`.
///
/// `λᵢ(S) = σᵢ(A)²`, so `A` carries `S`'s spectrum at `A`'s own conditioning, and
/// the rank counts the singular values above [`factor_singular_band`]. Forming
/// `AᵀA` first squares the conditioning: a mode with `σᵢ/σ₁ = 1e-10` is resolved
/// here, but its eigenvalue lies below the Gram's rounding band, where it comes
/// back as roundoff of either sign.
pub struct FactorRankPartition {
    /// Singular values of `A`, descending.
    pub singular_values: Vec<f64>,
    /// The right singular vectors as rows, in the same order (`n × n`).
    pub right_vectors: ndarray::Array2<f64>,
    /// Number of singular values above the backward-error band.
    pub rank: usize,
}

impl FactorRankPartition {
    /// The root rows `σᵢ·vᵢᵀ` for the leading `rank` modes, so `RᵀR` equals `S`
    /// on the resolved range.
    pub fn root_rows(&self, rank: usize) -> ndarray::Array2<f64> {
        let mut root = self.right_vectors.slice(ndarray::s![..rank, ..]).to_owned();
        for (mut row, &sigma) in root.rows_mut().into_iter().zip(self.singular_values.iter()) {
            row.mapv_inplace(|value| value * sigma);
        }
        root
    }
}

/// Partition `S = AᵀA` by the singular values of `A` (see [`FactorRankPartition`]).
pub fn factor_rank_partition(
    factor: &ndarray::Array2<f64>,
) -> Result<FactorRankPartition, crate::faer_ndarray::FaerLinalgError> {
    use crate::faer_ndarray::FaerSvd;
    let (rows, cols) = factor.dim();
    // A thin SVD of an `m × n` factor with `m < n` omits part of the null space
    // of `AᵀA`. Zero rows add no energy and complete the right singular basis.
    let completed;
    let square = if rows < cols {
        let mut padded = ndarray::Array2::<f64>::zeros((cols, cols));
        padded.slice_mut(ndarray::s![..rows, ..]).assign(factor);
        completed = padded;
        &completed
    } else {
        factor
    };
    let (_, sigma, vt) = square.svd(false, true)?;
    let vt = vt.ok_or(crate::faer_ndarray::FaerLinalgError::SvdNoConvergence {
        context: "factor_rank_partition: right singular vectors",
    })?;
    let mut order: Vec<usize> = (0..sigma.len()).collect();
    order.sort_by(|&i, &j| sigma[j].total_cmp(&sigma[i]));
    let singular_values: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
    let mut right_vectors = ndarray::Array2::<f64>::zeros((order.len(), cols));
    for (row, &i) in order.iter().enumerate() {
        right_vectors.row_mut(row).assign(&vt.row(i));
    }
    let sigma_max = singular_values.first().copied().unwrap_or(0.0);
    let band = factor_singular_band(rows, cols, sigma_max);
    let rank = singular_values.iter().filter(|&&value| value > band).count();
    Ok(FactorRankPartition {
        singular_values,
        right_vectors,
        rank,
    })
}

/// Forward-error band of a penalty trace `t = λ·Σ_c r_cᵀ H⁻¹ r_c` for a symmetric
/// `H`, computed as `t̂ = λ·Σ_c r_cᵀ x̂_c` from solved columns `x̂_c ≈ H⁻¹ r_c`
/// (#2901).
///
/// With the true residual `ρ_c = H·x̂_c − r_c`, exactly `x_c = x̂_c − H⁻¹ρ_c`, so
///
/// ```text
///   t − t̂ = −λ·Σ_c r_cᵀ H⁻¹ ρ_c = −λ·Σ_c x_cᵀ ρ_c,
///   |t − t̂| ≤ λ·Σ_c ‖ρ_c‖₂·(‖x̂_c‖₂ + ‖H⁻¹‖₂·‖ρ_c‖₂).
/// ```
///
/// - **Residual.** The computed residual `residual` differs from `ρ_c` by its own
///   formation. Each entry is an inner product of `rows + 1` terms bounded by
///   `rows·‖H‖_max·‖x̂_c‖_max + |r_ic|`, so `‖ρ_c‖₂` is charged the computed norm
///   plus that band.
/// - **Inverse norm.** For symmetric `H`, `‖H⁻¹‖₂ ≤ ‖H⁻¹‖₁`. That norm is
///   `inverse_one_norm_estimate`, the Hager–Higham lower-bound estimate
///   ([`crate::condition::estimate_inverse_one_norm`]), so the second-order term
///   is an estimate, not a certificate.
/// - **Formation of `t̂`.** Forming `t̂` is an inner product of `rows·columns`
///   terms, which adds `accumulation_band(rows·columns, λ·Σ|r_ic·x̂_ic|)`.
///
/// The band measures the solve that produced the trace. A trace published
/// outside `[0, rank]` by more than this band is not rounding.
pub fn solved_penalty_trace_band(
    lambda: f64,
    rhs: ndarray::ArrayView2<'_, f64>,
    solution: ndarray::ArrayView2<'_, f64>,
    residual: ndarray::ArrayView2<'_, f64>,
    matrix_max_abs: f64,
    inverse_one_norm_estimate: f64,
) -> Result<f64, String> {
    let (rows, columns) = rhs.dim();
    if solution.dim() != (rows, columns) || residual.dim() != (rows, columns) {
        return Err(format!(
            "solved_penalty_trace_band: a {rows}x{columns} right-hand side against a {}x{} solution \
             and a {}x{} residual",
            solution.nrows(),
            solution.ncols(),
            residual.nrows(),
            residual.ncols()
        ));
    }
    let residual_growth = accumulation_growth(rows + 1);
    let mut solve_band = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    for column in 0..columns {
        let solution_max = solution
            .column(column)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let product_bound = rows as f64 * matrix_max_abs * solution_max;
        let mut solution_norm_sq = 0.0_f64;
        let mut residual_norm_sq = 0.0_f64;
        let mut formation_norm_sq = 0.0_f64;
        for row in 0..rows {
            let x = solution[[row, column]];
            let r = rhs[[row, column]];
            solution_norm_sq += x * x;
            residual_norm_sq += residual[[row, column]] * residual[[row, column]];
            let formation = residual_growth * (product_bound + r.abs());
            formation_norm_sq += formation * formation;
            absolute_sum += (r * x).abs();
        }
        let charged_residual = residual_norm_sq.sqrt() + formation_norm_sq.sqrt();
        solve_band +=
            charged_residual * (solution_norm_sq.sqrt() + inverse_one_norm_estimate * charged_residual);
    }
    Ok(lambda.abs() * (solve_band + accumulation_growth(rows * columns) * absolute_sum))
}

/// The trace `t̂ = λ·Σ_c r_cᵀ x̂_c` of solved penalty-root columns, with its
/// [`solved_penalty_trace_band`] (#2901).
///
/// `rhs` holds the columns `r_c` a route solved against `operator`, and `solution`
/// the computed `x̂_c`. For a pseudoinverse solve, `rhs` is the root projected onto
/// the operator's range, the right-hand side that solve reproduces. The true
/// residual `operator·x̂_c − r_c` is formed here, so every route prices its trace
/// against the operator its solve represents.
pub fn solved_penalty_trace(
    lambda: f64,
    rhs: ndarray::ArrayView2<'_, f64>,
    solution: ndarray::ArrayView2<'_, f64>,
    operator: ndarray::ArrayView2<'_, f64>,
    inverse_one_norm_estimate: f64,
) -> Result<(f64, f64), String> {
    if operator.nrows() != rhs.nrows() || operator.ncols() != solution.nrows() {
        return Err(format!(
            "solved_penalty_trace: a {}x{} operator against a {}x{} right-hand side and a {}x{} \
             solution",
            operator.nrows(),
            operator.ncols(),
            rhs.nrows(),
            rhs.ncols(),
            solution.nrows(),
            solution.ncols()
        ));
    }
    let residual = operator.dot(&solution) - &rhs;
    let matrix_max_abs = operator
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let band = solved_penalty_trace_band(
        lambda,
        rhs,
        solution,
        residual.view(),
        matrix_max_abs,
        inverse_one_norm_estimate,
    )?;
    let trace: f64 = rhs.iter().zip(solution.iter()).map(|(r, x)| r * x).sum();
    Ok((lambda * trace, band))
}

// Backward-error bands of a solve `A·X̂ ≈ B`, read by the max-norm certificate
//
//     η = ‖A·X̂ − B‖_max / (n·‖A‖_max·‖X̂‖_max + ‖B‖_max)
//
// (`crate::utils::certify_linear_system_residual`). Each band is an upper
// bound on the η the named algorithm can produce in binary64 (#4038), so a
// residual above it is not rounding. Every band has two parts:
//
// * the SOLVE's backward error. A solve with `(A + ΔA)·x̂_j = b_j` for every
//   column leaves the exact residual `−ΔA·x̂_j`, whose entries are at most
//   `n·‖ΔA‖_max·‖X̂‖_max`, so it contributes `‖ΔA‖_max / ‖A‖_max` to η;
// * the residual's FORMATION. Each computed residual entry is an `n`-term
//   inner product and one subtraction, within `γ_{n+1}·(|A||x̂_j| + |b_j|)` of
//   the exact one in any summation order (Higham, *ASNA* 2nd ed., §3.1), which
//   contributes `γ_{n+1}` to η.

/// The residual-formation share of every solve band: `γ_{n+1}`.
fn residual_formation_band(dimension: usize) -> f64 {
    accumulation_growth(dimension + 1)
}

/// η band of a solve by an unpivoted Cholesky factor `A = R̂ᵀR̂` of an SPD `A`.
///
/// Higham (*ASNA* 2nd ed., Thm 10.4): the solve satisfies `(A + ΔA)x̂ = b` with
/// `|ΔA| ≤ γ_{3n+1}·|R̂ᵀ||R̂|`. Entry `(i, j)` of `|R̂ᵀ||R̂|` is at most
/// `‖r_i‖₂‖r_j‖₂` for the columns `r_i` of `R̂`, and the factor's own diagonal
/// (Thm 10.3, `R̂ᵀR̂ = A + ΔA₁`, `|ΔA₁| ≤ γ_{n+1}|R̂ᵀ||R̂|`) gives
/// `‖r_i‖₂² ≤ a_ii / (1 − γ_{n+1})`. So `‖ΔA‖_max ≤ γ_{3n+1}·‖A‖_max / (1 − γ_{n+1})`
/// with no growth factor at all, which is why the band depends on `n` alone.
/// Blocked and recursive variants reorder the same inner products, and a
/// symmetric permutation (a sparse fill-reducing ordering) is exact, so the band
/// covers them.
pub fn cholesky_solve_backward_band(dimension: usize) -> f64 {
    let solve = accumulation_growth(3 * dimension + 1);
    let factor = accumulation_growth(dimension + 1);
    if !(factor < 1.0) {
        return f64::INFINITY;
    }
    solve / (1.0 - factor) + residual_formation_band(dimension)
}

/// η band of a solve by a symmetric indefinite factor `P A Pᵀ = L̂ B̂ L̂ᵀ`
/// (unit lower `L̂`, block-diagonal `B̂` with 1×1 and 2×2 pivots).
///
/// Higham (*ASNA* 2nd ed., Thm 11.3–11.4): the factorization and the solve
/// together satisfy `(A + ΔA)x̂ = b` with `|ΔA| ≤ γ_k·(|A| + P ᵀ|L̂||B̂||L̂ᵀ|P)`,
/// where `k` counts the roundings on one entry's path: the `n`-term inner
/// products of the factorization and of the two triangular solves, plus the
/// pivot-block arithmetic twice (once to form the multipliers, once to solve the
/// block). `pivot_block_roundings` is that block constant `c_B` — `2` for a 1×1
/// pivot applied as a reciprocal and a product, and the measured constant of the
/// 2×2 block formula otherwise — so `k = 3n + 2·c_B + 3`.
///
/// `factor_product_bound` is an upper bound `G` on `‖ |L̂||B̂||L̂ᵀ| ‖_max`,
/// measured on the factor. Unlike Cholesky this term is NOT bounded by `‖A‖_max`:
/// it is the element growth of the pivoting, and it is the only honest source of
/// the band's size.
pub fn symmetric_factor_solve_backward_band(
    dimension: usize,
    pivot_block_roundings: f64,
    factor_product_bound: f64,
    matrix_max_abs: f64,
) -> f64 {
    if !(matrix_max_abs > 0.0) || !factor_product_bound.is_finite() || factor_product_bound < 0.0
    {
        return f64::INFINITY;
    }
    if !(pivot_block_roundings.is_finite() && pivot_block_roundings >= 0.0) {
        return f64::INFINITY;
    }
    let operations = 3.0 * dimension as f64 + 2.0 * pivot_block_roundings.ceil() + 3.0;
    if !(operations * UNIT_ROUNDOFF < 1.0) {
        return f64::INFINITY;
    }
    let growth = accumulation_growth(operations as usize);
    growth * (matrix_max_abs + factor_product_bound) / matrix_max_abs
        + residual_formation_band(dimension)
}

/// A certified upper bound `ω ≥ ‖UᵀU − I‖₂` for a computed `n × rank` basis
/// `U` meant to be orthonormal, from `computed_gram_defect_frobenius =
/// ‖fl(fl(UᵀU) − I)‖_F`.
///
/// Every entry of `fl(UᵀU)` is an `n`-term inner product within
/// `γ_n·‖u_a‖₂‖u_b‖₂ ≤ γ_n·(1 + ω)` of the exact one, so the `rank × rank`
/// formation error has Frobenius norm at most `γ_n·rank·(1 + ω)`. With the
/// computed norm inflated for its diagonal subtraction and its `rank²`-term
/// accumulation (`ω̂`), `ω ≤ ω̂ + γ_n·rank·(1 + ω)`, i.e.
/// `ω ≤ (ω̂ + γ_n·rank)/(1 − γ_n·rank)`. Infinite when that is no bound.
pub fn orthonormality_defect_bound(
    computed_gram_defect_frobenius: f64,
    dimension: usize,
    rank: usize,
) -> f64 {
    let formation = accumulation_growth(dimension) * rank as f64;
    if !(formation < 1.0) || !computed_gram_defect_frobenius.is_finite() {
        return f64::INFINITY;
    }
    let computed = gam_math::roundoff::inflated(
        computed_gram_defect_frobenius,
        rank.saturating_mul(rank).saturating_add(1),
    );
    (computed + formation) / (1.0 - formation)
}

/// A certified upper bound on `‖H·U − U·Σ̃‖₂` from its computed value
/// `computed_residual_frobenius = ‖fl(H·U − fl(U·Σ̃))‖_F`, for `U` with `rank`
/// columns and orthonormality defect `ω ≥ ‖UᵀU − I‖₂`
/// ([`orthonormality_defect_bound`]) and `Σ̃` diagonal with largest entry
/// `max_abs_eigenvalue`.
///
/// Each computed entry is an `n`-term inner product, one product (or quotient)
/// and one subtraction, within `γ_{n+1}·(|H||U| + |U||Σ̃|)` of the exact one. A
/// column has `‖u‖₁ ≤ √n·‖u‖₂ ≤ √n·√(1 + ω)`, so `(|H||U|)_{ia} ≤
/// √n·√(1 + ω)·‖H‖_max`, and the Frobenius norm of the majorant over the
/// `n × rank` entries is at most `√rank·√(1 + ω)·(n·‖H‖_max + max|σ̃|)`.
/// `‖·‖₂ ≤ ‖·‖_F` and the triangle inequality give the bound; the computed
/// Frobenius norm is inflated for its own `n·rank`-term accumulation.
pub fn eigen_residual_two_norm_bound(
    computed_residual_frobenius: f64,
    dimension: usize,
    rank: usize,
    orthonormality_defect: f64,
    matrix_max_abs: f64,
    max_abs_eigenvalue: f64,
) -> f64 {
    if !(orthonormality_defect.is_finite() && orthonormality_defect >= 0.0) {
        return f64::INFINITY;
    }
    let computed = gam_math::roundoff::inflated(
        computed_residual_frobenius,
        dimension.saturating_mul(rank),
    );
    computed
        + accumulation_growth(dimension + 1)
            * (rank as f64 * (1.0 + orthonormality_defect)).sqrt()
            * (dimension as f64 * matrix_max_abs + max_abs_eigenvalue)
}

/// η band of the spectral solve `X̂ = fl(U·Z)`, `Z = fl(S·C)`, `C = fl(Uᵀ·B)`,
/// certified against `B̃ = fl(U·C)`, for `U` with `rank` columns and
/// orthonormality defect `ω ≥ ‖UᵀU − I‖₂`, `S = diag(s_a)` the stored inverse
/// eigenvalues, `Σ̃ = S⁻¹` exactly, and `eigen_residual_two_norm ≥
/// ‖H·U − U·Σ̃‖₂` ([`eigen_residual_two_norm_bound`]).
///
/// With `R_e = HU − UΣ̃`, `Z = SC + E_Z`, `X̂ = UZ + E_X` and `B̃ = UC + E_B`,
/// exactly `Σ̃Z = C + Σ̃E_Z` and
///
/// ```text
///   H·X̂ − B̃ = U·Σ̃·E_Z + R_e·Z + H·E_X − E_B,
/// ```
///
/// with `|Σ̃E_Z| ≤ u|C|` (one product), `|E_X| ≤ γ_rank|U||Z|` and
/// `|E_B| ≤ γ_rank|U||C|`. The rows of `U` have norm at most `ν = √(1 + ω)`, so
/// column `j` of the four terms is at most `ν·u‖c_j‖₂`, `‖R_e‖₂‖z_j‖₂`,
/// `n‖H‖_max·γ_rank·ν‖z_j‖₂` and `γ_rank·ν‖c_j‖₂` in the max norm. Since
/// `‖Uz‖₂ ≥ √(1 − ω)‖z‖₂` and `‖|U||z|‖₂ ≤ ‖U‖_F‖z‖₂ ≤ √(rank)·ν‖z‖₂`,
/// `√n‖X̂‖_max ≥ ‖x̂_j‖₂ ≥ θ‖z_j‖₂` with `θ = √(1 − ω) − γ_rank·√rank·ν`, and
/// likewise `√n‖B̃‖_max ≥ θ‖c_j‖₂`. Against the certificate's denominator
/// `n‖H‖_max‖X̂‖_max + ‖B̃‖_max` the terms are at most `(ν/θ)·√n·u`,
/// `‖R_e‖₂/(θ√n‖H‖_max)`, `(ν/θ)·√n·γ_rank` and `(ν/θ)·√n·γ_rank`. The
/// residual's formation adds `γ_{n+1}`. Infinite when `θ ≤ 0`.
pub fn spectral_solve_backward_band(
    dimension: usize,
    rank: usize,
    matrix_max_abs: f64,
    eigen_residual_two_norm: f64,
    orthonormality_defect: f64,
) -> f64 {
    if !(matrix_max_abs > 0.0)
        || !eigen_residual_two_norm.is_finite()
        || !(orthonormality_defect >= 0.0 && orthonormality_defect < 1.0)
    {
        return f64::INFINITY;
    }
    let nu = (1.0 + orthonormality_defect).sqrt();
    let theta = (1.0 - orthonormality_defect).sqrt()
        - accumulation_growth(rank) * (rank as f64).sqrt() * nu;
    if !(theta > 0.0) {
        return f64::INFINITY;
    }
    let root_n = (dimension as f64).sqrt();
    residual_formation_band(dimension)
        + eigen_residual_two_norm / (theta * root_n * matrix_max_abs)
        + nu / theta * root_n * (UNIT_ROUNDOFF + 2.0 * accumulation_growth(rank))
}

/// η band of a positive semidefinite pseudo-inverse accumulated from `rank`
/// eigenpairs, `X̂ = fl(Σ_a fl(fl(s_a·v_a)·v_aᵀ))` with every `s_a = fl(1/σ_a) > 0`,
/// certified against the projector `P̂ = fl(Σ_a v_a·v_aᵀ)`, for eigenvectors
/// with orthonormality defect `ω ≥ ‖VᵀV − I‖₂` and `eigen_residual_two_norm ≥
/// ‖H·V − V·Σ̃‖₂`, `Σ̃ = diag(1/s_a)` exactly.
///
/// Exactly `H·V·S·Vᵀ − V·Vᵀ = (H·V − V·Σ̃)·S·Vᵀ`: column `c` of it is
/// `R_e·y_c`, `y_c = S·Vᵀe_c`, and the exact `X = V·S·Vᵀ` has column
/// `V·y_c`, so `‖y_c‖₂ ≤ ‖X e_c‖₂/√(1 − ω) ≤ √n‖X‖_max/√(1 − ω)`. The
/// accumulation errors, by weighted Cauchy–Schwarz on the positive `s_a`, are
/// `|F_X| ≤ γ_{rank+3}·√(X_rr X_cc) ≤ γ_{rank+3}·‖X‖_max` (so also
/// `‖X‖_max ≤ ‖X̂‖_max/(1 − γ_{rank+3})`) and `|F_P| ≤ γ_rank·√(P_rr P_cc) ≤
/// γ_rank·(1 + ω)`, and `‖P̂‖_max ≥ trace(P)/n − γ_rank(1 + ω) ≥
/// rank·(1 − ω)/n − γ_rank(1 + ω)`. Against the certificate's denominator the
/// three parts are at most `‖R_e‖₂/(√(1 − ω)·(1 − γ_{rank+3})·√n‖H‖_max)`,
/// `γ_{rank+3}/(1 − γ_{rank+3})` and `γ_rank(1 + ω)/‖P̂‖_max`'s lower bound.
/// The residual's formation adds `γ_{n+1}`.
pub fn psd_pseudo_inverse_backward_band(
    dimension: usize,
    rank: usize,
    matrix_max_abs: f64,
    eigen_residual_two_norm: f64,
    orthonormality_defect: f64,
) -> f64 {
    if rank == 0 {
        // Nothing accumulated: `X̂ = P̂ = 0` exactly.
        return residual_formation_band(dimension);
    }
    if !(matrix_max_abs > 0.0)
        || !eigen_residual_two_norm.is_finite()
        || !(orthonormality_defect >= 0.0 && orthonormality_defect < 1.0)
    {
        return f64::INFINITY;
    }
    let n = dimension as f64;
    let accumulation = accumulation_growth(rank + 3);
    let projector_error = accumulation_growth(rank) * (1.0 + orthonormality_defect);
    let projector_floor = rank as f64 * (1.0 - orthonormality_defect) / n - projector_error;
    if !(accumulation < 1.0) || !(projector_floor > 0.0) {
        return f64::INFINITY;
    }
    residual_formation_band(dimension)
        + eigen_residual_two_norm
            / ((1.0 - orthonormality_defect).sqrt() * (1.0 - accumulation) * n.sqrt() * matrix_max_abs)
        + accumulation / (1.0 - accumulation)
        + projector_error / projector_floor
}

#[cfg(test)]
mod tests {
    use super::*;

    /// To first order in `u` the Cholesky band is `(3n+1)u + (n+1)u =
    /// (4n+2)u = (2n+1)ε`: the derivation's own count, which is what replaced
    /// the unexplained `256·n·ε` (#4038), and it is below that at every `n`.
    #[test]
    fn cholesky_solve_band_is_its_rounding_count_4038() {
        for dimension in [1_usize, 2, 10, 100, 10_000] {
            let band = cholesky_solve_backward_band(dimension);
            let first_order = (4 * dimension + 2) as f64 * UNIT_ROUNDOFF;
            assert!(band.is_finite() && band > 0.0);
            assert!(
                (band / first_order - 1.0).abs() < 1.0e-9,
                "n={dimension}: band {band:e} against first-order {first_order:e}"
            );
            assert!(band < 256.0 * dimension as f64 * f64::EPSILON);
        }
    }

    /// The indefinite band carries the factor's growth `G` and nothing else: at
    /// `G = ‖A‖_max` it is twice the entry band, it grows linearly in `G`, and a
    /// growth that is not finite leaves no bound.
    #[test]
    fn symmetric_factor_band_is_priced_by_measured_growth_4038() {
        let dimension = 20;
        let entry = accumulation_growth(3 * dimension + 2 * 2 + 3);
        let formation = accumulation_growth(dimension + 1);
        let unit_growth = symmetric_factor_solve_backward_band(dimension, 2.0, 3.0, 3.0);
        assert!((unit_growth - (2.0 * entry + formation)).abs() <= 1.0e-12 * unit_growth);
        let large_growth = symmetric_factor_solve_backward_band(dimension, 2.0, 3.0e6, 3.0);
        assert!((large_growth - ((1.0 + 1.0e6) * entry + formation)).abs() <= 1.0e-9 * large_growth);
        assert!(symmetric_factor_solve_backward_band(dimension, 2.0, f64::INFINITY, 3.0).is_infinite());
        assert!(symmetric_factor_solve_backward_band(dimension, f64::NAN, 3.0, 3.0).is_infinite());
        assert!(symmetric_factor_solve_backward_band(dimension, 2.0, 3.0, 0.0).is_infinite());
    }

    /// An exactly orthonormal basis still gets the Gram's formation error, and a
    /// defect that is no bound (formation `≥ 1`, non-finite) is infinite.
    #[test]
    fn orthonormality_defect_bound_covers_gram_formation_4038() {
        let omega = orthonormality_defect_bound(0.0, 50, 10);
        let formation = accumulation_growth(50) * 10.0;
        assert!(omega >= formation && omega < 2.0 * formation);
        assert!(orthonormality_defect_bound(f64::NAN, 50, 10).is_infinite());
        let measured = orthonormality_defect_bound(1.0e-12, 50, 10);
        assert!(measured >= 1.0e-12 + formation);
    }

    /// The spectral and pseudo-inverse bands reduce to their rounding terms at a
    /// zero eigen-residual, grow with a measured one, and refuse a basis that is
    /// not near-orthonormal.
    #[test]
    fn spectral_and_pseudo_inverse_bands_charge_the_measured_residual_4038() {
        let (dimension, rank) = (30, 12);
        let omega = orthonormality_defect_bound(0.0, dimension, rank);
        let spectral = spectral_solve_backward_band(dimension, rank, 2.0, 0.0, omega);
        assert!(spectral.is_finite() && spectral < 1.0e-12);
        let spectral_residual = spectral_solve_backward_band(dimension, rank, 2.0, 1.0e-6, omega);
        assert!(spectral_residual > spectral + 1.0e-6 / (2.0 * (dimension as f64).sqrt() * 2.0));
        assert!(spectral_solve_backward_band(dimension, rank, 2.0, 0.0, 1.0).is_infinite());
        assert!(spectral_solve_backward_band(dimension, rank, 0.0, 0.0, omega).is_infinite());

        let pseudo = psd_pseudo_inverse_backward_band(dimension, rank, 2.0, 0.0, omega);
        assert!(pseudo.is_finite() && pseudo < 1.0e-12);
        assert!(psd_pseudo_inverse_backward_band(dimension, rank, 2.0, 1.0e-6, omega) > pseudo);
        assert!(psd_pseudo_inverse_backward_band(dimension, rank, 2.0, 0.0, 1.0).is_infinite());
        assert_eq!(
            psd_pseudo_inverse_backward_band(dimension, 0, 2.0, f64::NAN, f64::NAN),
            accumulation_growth(dimension + 1)
        );
    }

    /// The composed trace is `λ·Σ r·x̂`, and its band covers the error of a
    /// displaced solution through the residual it forms against the operator.
    #[test]
    fn a_solved_penalty_trace_prices_its_solution_against_the_operator_2901() {
        let operator = ndarray::array![[2.0, 0.0], [0.0, 4.0]];
        let rhs = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let exact = ndarray::array![[0.5, 0.0], [0.0, 0.25]];
        let (trace, band) =
            solved_penalty_trace(3.0, rhs.view(), exact.view(), operator.view(), 0.5)
                .expect("exact solve");
        assert_eq!(trace, 2.25);
        let displaced = ndarray::array![[0.501, 0.0], [0.0, 0.25]];
        let (displaced_trace, displaced_band) =
            solved_penalty_trace(3.0, rhs.view(), displaced.view(), operator.view(), 0.5)
                .expect("displaced solve");
        let error = (displaced_trace - trace).abs();
        assert!(
            error <= displaced_band,
            "displaced trace error {error:e} escapes its band {displaced_band:e}"
        );
        assert!(
            error > band,
            "the exact solve's band {band:e} must not cover a displacement of {error:e}"
        );
    }

    /// #2901: a solve left inexact by a known perturbation moves the trace by
    /// `−λ·Σ x_cᵀρ_c`, and the band carries it. `H = diag(4, 0.5)`, `r = (1, 1)`,
    /// `λ = 3`: the exact trace is `3·(1/4 + 2) = 6.75`. A solution off by
    /// `δ = (1e-3, −2e-3)` has residual `Hδ` and moves the trace by `3·(1e-3 − 2e-3)`.
    /// The control passes a zero residual and shows the residual term is what
    /// covers the error.
    #[test]
    fn a_solved_trace_band_covers_the_error_its_residual_carries_2901() {
        let lambda = 3.0;
        let rhs = ndarray::array![[1.0], [1.0]];
        let exact = ndarray::array![[0.25], [2.0]];
        let offset = ndarray::array![[1.0e-3], [-2.0e-3]];
        let solution = &exact + &offset;
        let residual = ndarray::array![[4.0 * 1.0e-3], [0.5 * -2.0e-3]];
        let exact_trace = lambda * (0.25 + 2.0);
        let computed_trace = lambda * (solution[[0, 0]] + solution[[1, 0]]);
        let inverse_one_norm = 2.0;
        let band = solved_penalty_trace_band(
            lambda,
            rhs.view(),
            solution.view(),
            residual.view(),
            4.0,
            inverse_one_norm,
        )
        .unwrap();
        assert!(
            (computed_trace - exact_trace).abs() <= band,
            "the trace error {:.4e} escaped its band {band:.4e}",
            (computed_trace - exact_trace).abs()
        );
        let unmeasured = solved_penalty_trace_band(
            lambda,
            rhs.view(),
            solution.view(),
            ndarray::Array2::<f64>::zeros((2, 1)).view(),
            4.0,
            inverse_one_norm,
        )
        .unwrap();
        assert!(
            (computed_trace - exact_trace).abs() > unmeasured,
            "without the residual the band {unmeasured:.4e} must not cover {:.4e}",
            (computed_trace - exact_trace).abs()
        );
        assert!(
            solved_penalty_trace_band(
                lambda,
                rhs.view(),
                solution.view(),
                ndarray::Array2::<f64>::zeros((3, 1)).view(),
                4.0,
                inverse_one_norm
            )
            .is_err()
        );
    }

    /// One Gram-side predicate: eigenvalues above the eigensolver band plus the
    /// caller's assembly band are resolved, and roundoff of either sign is not.
    #[test]
    fn resolved_eigenvalue_count_reads_the_band_not_the_sign() {
        let spectrum = [1.0, 1.0e-8, 1.0e-17, -1.0e-17];
        assert_eq!(resolved_eigenvalue_count(&spectrum, 0.0), 2);
        assert_eq!(resolved_eigenvalue_count(&spectrum, 1.0e-7), 1);
        let band = symmetric_spectrum_rounding_band(&spectrum);
        assert!(1.0e-17 <= band, "the roundoff pair must sit inside the eigensolver band");
    }

    /// A graded factor whose small singular value is resolved from `A` though its
    /// eigenvalue lies below the Gram's rounding band. A wide factor returns a
    /// complete right singular basis, including the directions of `AᵀA`'s null
    /// space that a thin SVD omits.
    #[test]
    fn factor_rank_partition_resolves_below_the_gram_band_and_completes_wide_factors() {
        let graded = ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0e-10, 0.0], [0.0, 0.0, 0.0]];
        let partition = factor_rank_partition(&graded).expect("graded factor");
        assert_eq!(partition.rank, 2);
        let gram_band = symmetric_spectrum_rounding_band(&[1.0, 1.0e-20, 0.0]);
        assert!(
            1.0e-20 <= gram_band,
            "the second mode's eigenvalue must lie inside the Gram's band"
        );
        let root = partition.root_rows(partition.rank);
        let rebuilt = root.t().dot(&root);
        assert!((rebuilt[[0, 0]] - 1.0).abs() <= 4.0 * f64::EPSILON);
        assert!((rebuilt[[1, 1]] - 1.0e-20).abs() <= 1.0e-30);

        let wide = ndarray::array![[1.0, 2.0, 0.0, 0.0]];
        let wide_partition = factor_rank_partition(&wide).expect("wide factor");
        assert_eq!(wide_partition.rank, 1);
        assert_eq!(wide_partition.right_vectors.dim(), (4, 4));
        let gram = wide_partition
            .right_vectors
            .dot(&wide_partition.right_vectors.t());
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((gram[[i, j]] - expected).abs() <= 8.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn unit_roundoff_is_half_an_epsilon_gap() {
        assert_eq!(UNIT_ROUNDOFF * 2.0, f64::EPSILON);
        // 1.0 + u rounds to 1.0 (ties-to-even), 1.0 + 2u does not.
        assert_eq!(1.0_f64 + UNIT_ROUNDOFF, 1.0);
        assert!(1.0_f64 + 2.0 * UNIT_ROUNDOFF > 1.0);
    }

    #[test]
    fn growth_is_zero_for_exact_arithmetic_and_grows_linearly() {
        assert_eq!(accumulation_growth(0), 0.0);
        let one = accumulation_growth(1);
        assert!((one - UNIT_ROUNDOFF).abs() <= UNIT_ROUNDOFF * UNIT_ROUNDOFF * 4.0);
        // gamma_n / (n u) -> 1 from above, and is monotone in n.
        let mut previous = 0.0_f64;
        for n in [1_usize, 8, 64, 1024, 1 << 20] {
            let gamma = accumulation_growth(n);
            assert!(gamma > previous, "gamma must be monotone in n");
            assert!(gamma >= n as f64 * UNIT_ROUNDOFF);
            assert!(gamma <= n as f64 * UNIT_ROUNDOFF * 1.000_001);
            previous = gamma;
        }
    }

    #[test]
    fn compensated_band_is_the_naive_one_divided_by_the_term_count() {
        // The compensated band takes no term count at all -- independence from
        // `n` is a property of the signature, so the thing worth asserting is
        // the GAP against the naive bound, which is what a caller pays for by
        // scaling a compensated sum by `n` anyway.
        assert_eq!(compensated_band(0, 1.0), 2.0 * UNIT_ROUNDOFF);
        assert_eq!(compensated_band(3, 1.0), 5.0 * UNIT_ROUNDOFF);
        // Linear in the magnitude sum, as a forward-error bound must be.
        assert_eq!(compensated_band(3, 4.0), 4.0 * compensated_band(3, 1.0));
        // The gap grows without bound in n: that IS the defect it exists to
        // fix, so pin its size rather than merely asserting an inequality.
        let mut previous = 0.0_f64;
        for terms in [100_usize, 2_500, 10_000] {
            let ratio = accumulation_band(terms, 1.0) / compensated_band(3, 1.0);
            assert!(ratio > previous, "the gap must widen with n");
            // gamma_n ~ n*u, so the ratio approaches n/5.
            let expected = terms as f64 / 5.0;
            assert!(
                (ratio / expected - 1.0).abs() < 1.0e-6,
                "n={terms}: ratio {ratio} is not ~n/5 ({expected})"
            );
            previous = ratio;
        }
    }

    #[test]
    fn growth_saturates_rather_than_going_negative() {
        // n u >= 1 makes the textbook quotient negative; the band must not be.
        let vacuous = (1.0 / UNIT_ROUNDOFF).ceil() as usize;
        assert_eq!(accumulation_growth(vacuous), f64::INFINITY);
        assert_eq!(accumulation_growth(usize::MAX), f64::INFINITY);
    }

    #[test]
    fn band_bounds_a_cancelling_sum_that_is_exactly_zero() {
        // Every magnitude appears once positive and once negative, so the sum
        // is exactly zero in real arithmetic. The two signs are kept apart so
        // the cancellation is not absorbed by adjacent-pair exactness, which
        // would make the computed sum trivially 0.0.
        let magnitudes: Vec<f64> = (1..=256)
            .map(|k| (k as f64) * 0.1_f64.powi(k % 7))
            .collect();
        let terms: Vec<f64> = magnitudes
            .iter()
            .copied()
            .chain(magnitudes.iter().map(|value| -value))
            .collect();
        let absolute_sum: f64 = terms.iter().map(|value| value.abs()).sum();
        let computed: f64 = terms.iter().sum();
        let band = accumulation_band(terms.len(), absolute_sum);
        assert!(band > 0.0);
        assert!(
            computed.abs() <= band,
            "computed {computed:e} escaped its band {band:e}"
        );
    }
}
