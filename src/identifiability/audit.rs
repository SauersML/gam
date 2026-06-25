// Pre-fit cross-block identifiability audit.
//
// # What this module provides
//
// Two family-agnostic entry points:
//
//   * [`audit_identifiability`] — flat (single-channel) joint RRQR audit
//     on `X_joint = [X_block_0 | X_block_1 | …]`. Suitable for standard
//     GAM, gaussian/binomial/survival location-scale, royston-parmar, and
//     any custom-family workflow whose blocks live in a common row space.
//     Wired into every fit path via `fit_custom_family_with_rho_prior` →
//     `canonicalize_for_identifiability` in `custom_family.rs`.
//
//   * [`audit_identifiability_channel_aware`] — multi-channel audit on
//     the `(n·K) × p_total` channel-weighted joint design. Suitable for
//     families such as survival marginal-slope (K = 4) where two blocks
//     sharing identical raw-X columns may still be separately identifiable
//     through orthogonal K-channels of the row Jacobian. Callers supply
//     one `RowJacobianOperator` per block and a `RowHessian` structural
//     metric; the audit routes through `compile_with_dual_metric` so the
//     rank decision uses the structural metric rather than a
//     possibly-rank-deficient pilot curvature.
//
// # BMS flex-block path
//
// `crate::families::bms::install_compiled_flex_block_into_runtime`
// is now a thin spec-builder → audit → compile → install wrapper:
//
//   1. `build_bms_flex_block_context` densifies anchors, stacks N_train, and
//      assembles `BernoulliDenseDesignOperator` + `BlockOrder` + `BernoulliRowHessian`
//      from the BMS-specific inputs.
//   2. `audit_identifiability_channel_aware` (this module) acts as the
//      structural rank gate — it detects full aliasing via the K=1 BMS row
//      Jacobian before any install. `FlexEvaluation` anchors participate here.
//   3. `crate::identifiability::families::compiler::compile` does the W-metric
//      Gram + eigendecomp to produce the V selector and anchor-correction M.
//   4. The compiled V/M are installed into the `DeviationRuntime`
//      (`install_compiled_flex_block`), and the block design + penalties are
//      rebuilt in the new basis.
//
// `FlexEvaluation` anchors (an earlier flex block's training-row evaluation)
// are a genuine participant in the anchor union at both the audit gate and the
// compile step — `AnchorComponentTag::FlexEvaluation` is pushed into the
// anchor stack and the residualisation runs against the full
// `[parametric | flex_evals]` horizontally-stacked anchor. The regression
// guard lives in the BMS test
// `cross_block_identifiability_flex_anchor_true_alias_returns_fully_aliased`.
//
// After this per-block construction the resulting reparameterised designs are
// passed to `fit_custom_family`, which routes through
// `canonicalize_for_identifiability` for the final post-construction unified
// flat audit. Because the BMS blocks are already rank-clean after the W-metric
// residualisation, that second audit passes cleanly; its value is as a
// defensive gate for any future code path that bypasses the BMS construction.
//
// # Failure taxonomy
//
// | Condition | `fatal` | Action |
// |-----------|---------|--------|
// | Joint rank deficient, gauge cannot resolve (same priority or no attribution) | true | `IdentifiabilityFailure` |
// | Joint rank deficient, gauge resolves (distinct priorities, full attribution to lower-priority blocks) | false | gauge-attributed drops message |
// | Cross-block overlap ≥ per-pair halt threshold (leverage-scaled), same-priority blocks | true | `IdentifiabilityFailure` |
// | Cross-block overlap ≥ per-pair halt threshold (leverage-scaled), distinct-priority blocks | false | gauge-attributed drops message |
// | Any pairwise overlap ≥ per-pair report threshold (leverage-scaled) | false | INFO log |
// | All blocks clean | false | silent pass |
//
// `InnerFailure::IdentifiabilityFailure` is the upstream variant that
// classifies these refusals in the KKT-refusal and continuation pipelines
// (`src/solver/inner_status.rs`).
//
// # Algorithm (flat path)
//
// 1. Densify each block once (n × p_block). Record per-block
//    penalty-aware numerical rank as `design_range_rank`.
// 2. Stack horizontally into `X_joint ∈ ℝ^{n×p_total}`. Sort columns by
//    descending `gauge_priority` before RRQR so higher-priority blocks own
//    shared directions (canonical-gauge ownership contract).
// 3. Column-pivoted RRQR (`rrqr_with_permutation`, tolerance
//    `RRQR_RANK_ALPHA · ε · max(m,n) · leading`) identifies demoted columns.
// 4. Each demoted joint column is attributed back to its `(block_idx,
//    local_col)` origin via `col_offsets`. `effective_dim` per block is
//    updated.
// 5. Pairwise normalised inner-product scan over cross-block column pairs
//    surfaces `AliasedPair` records above the per-pair leverage-based
//    reporting threshold (see `pair_report_threshold`).
// 6. `fatal = joint_rank_deficient || any_pair_above_per_pair_halt_threshold`.

use faer::Side;
use ndarray::{Array1, Array2};

use crate::families::custom_family::{FamilyLinearizationState, ParameterBlockSpec};
use crate::linalg::faer_ndarray::{
    FaerEigh, default_rrqr_rank_alpha, fast_atb, rrqr_with_permutation,
};
use crate::solver::estimate::EstimationError;

const DEFAULT_GAUGE_PRIORITY: u8 = 100;

/// Lower bound on the cosine that may be reported as an `AliasedPair` when the
/// per-pair null cosine distribution has little width (σ → 0, i.e. both columns
/// near-uniform with leverage concentration S2 ≈ 1/n). In that regime ordinary
/// correlation between two distinct, fully-identifiable directions can reach a
/// substantial cosine (e.g. ≈ 0.745 between a constant `1` and an `x²` column
/// over a symmetric grid) without being an aliasing/identifiability problem;
/// only a near-exact cosine (≈ 1) is a genuine rank deficiency there. This is
/// the near-exact-alias boundary the fixed `ALIAS_OVERLAP_REPORTING_THRESHOLD`
/// used before the leverage-scaled rewrite, restored as the report-band floor.
const REPORT_FLOOR_NEAR_EXACT: f64 = 0.95;

/// Estimated audit work (rows × blocks, or rows × total columns for the
/// pairwise sweep) above which a periodic progress ticker is attached. Below
/// this the audit completes fast enough that progress output is noise.
const AUDIT_PROGRESS_TICKER_WORK_THRESHOLD: usize = 1_000_000;
const CHANNEL_AWARE_ROW_CHUNK: usize = 4096;

/// Per-block accounting record. `original_dim` is the spec's column
/// count at audit entry (post `joint_null_rotation` absorption — the
/// audit is contractually run on the rotated specs). `effective_dim`
/// is what remains after the audit drops aliased columns. Equal values
/// mean the block carried no redundant directions w.r.t. earlier
/// blocks.
#[derive(Debug, Clone)]
pub struct BlockIdentity {
    pub block_name: String,
    pub original_dim: usize,
    pub effective_dim: usize,
    /// Numerical rank of the block's column space at the n training
    /// rows, computed by penalty-aware column-pivoted RRQR on `[J; S]`
    /// (so penalty-covered design-null directions count as identified).
    /// Equal to `original_dim` for any well-posed block; smaller values
    /// flag a within-block rank deficiency that escaped within-smooth
    /// nullspace absorption.
    pub design_range_rank: usize,
}

/// A pair `(block_a.column → block_b.column)` whose normalised
/// inner product exceeds the alias-overlap reporting threshold.
/// Reported once per audited pair, in block-order (`block_a` index
/// strictly less than `block_b` index in the spec list, so the
/// "earlier block carries the image" attribution is well-defined).
#[derive(Debug, Clone)]
pub struct AliasedPair {
    pub block_a: String,
    pub block_b: String,
    pub direction_a: usize,
    pub direction_b: usize,
    /// `|aᵀb| / (‖a‖·‖b‖)`. Always in `[0, 1]`. Values at or near 1.0
    /// indicate near-perfect collinearity; values in `(threshold, 1.0)`
    /// indicate partial overlap that the column-pivoted QR will still
    /// preserve (only fully redundant directions get pivoted out).
    pub overlap: f64,
    /// Bias shift applied to the null-distribution mean for this pair,
    /// equal to `bias_shift_for_pair(z_a, z_b, s2_a, s2_b)`.
    /// Non-zero when exactly one block carries a `RowScaledJacobian` callback
    /// (or the two scalings differ) and the row-scaling vector is skewed.
    /// Stored so that the halt-threshold check can apply the same
    /// directional correction as the report-threshold check.
    /// Zero for all pairs arising from the channel-aware audit path,
    /// and for pairs from the flat path when both blocks have symmetric
    /// (or absent) row scaling.
    pub bias_shift: f64,
}

#[derive(Debug, Clone)]
pub struct DroppedColumn {
    pub block: String,
    pub column: usize,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct IdentifiabilityAudit {
    pub blocks: Vec<BlockIdentity>,
    pub aliased_pairs: Vec<AliasedPair>,
    pub dropped_columns: Vec<DroppedColumn>,
    /// `true` when at least one dropped column's attribution to an
    /// earlier block is ambiguous (overlap distributed across multiple
    /// earlier blocks above tolerance) or the drop would silently
    /// change model semantics. Callers must refuse the fit in that
    /// case rather than silently proceed with a different model.
    pub fatal: bool,
    pub summary: String,
}

/// Compute the leverage concentration S2_k for a joint design column.
///
/// # Finite-sample identity for cross-block cosine under standardised z
///
/// Let φ be a length-n column vector.  Define normalised leverage weights
///   p_i = φ_i² / Σ_j φ_j²
/// so that Σ_i p_i = 1.  For a sample-standardised independent noise
/// vector z (zero mean, unit variance per component), the cosine between
/// φ and z·φ satisfies the exact finite-sample identity
///
///   cos_W(φ, z·φ) = E_p[z] / √(E_p[z²])
///
/// Under the null (z truly random and independent of φ), the
/// concentration scale of this cosine is
///
///   σ_k² = S2_k − 1/n,   S2_k = Σ_i p_i² = Σ_i φ_i⁴ / (Σ_i φ_i²)²
///
/// giving effective sample size n_eff,k = 1/S2_k.  A column with
/// uniform φ_i = 1/√n has S2_k = 1/n (n_eff = n, tight null); a column
/// concentrated on r rows has S2_k ≈ 1/r (n_eff ≈ r, wide null).
///
/// Returns S2_k.  For a zero-norm column returns 1.0 (n_eff = 1,
/// most conservative possible).
fn compute_leverage_s2(col: &ndarray::ArrayView1<f64>) -> f64 {
    let sq_norm: f64 = col.iter().map(|v| v * v).sum();
    if sq_norm <= 0.0 {
        return 1.0;
    }
    // S2_k = Σ_i (φ_i² / sq_norm)² = Σ_i φ_i⁴ / sq_norm²
    col.iter().map(|v| (v * v / sq_norm).powi(2)).sum()
}

/// Compute the standardised (unit-variance) third central moment μ_3 of a
/// row-scaling vector z from a `RowScaledJacobian` callback, applying the
/// finite-sample unbiased correction factor n / ((n-1)(n-2)).
///
/// # Derivation of the null-mean bias term
///
/// When one of the two blocks in a cross-block cosine comparison carries
/// a `RowScaledJacobian` with scaling `z`, the effective Jacobian column is `z ⊙ φ`
/// instead of `φ`.  The population cosine between `φ` (from the other block)
/// and `z ⊙ φ` (from the scaled block) is, under independence of z and φ,
///
///   E[cos(φ, z⊙φ)] = E_p[z] / √(E_p[z²])
///
/// where E_p[·] = Σ_i p_i(·) with p_i = φ_i² / Σ_j φ_j² (leverage weights).
///
/// For sample-standardised z (zero sample mean, unit sample variance),
/// E_p[z] = Σ_i p_i z_i and E_p[z²] = Σ_i p_i z_i².  At small leverage
/// concentrations (S2_k ≈ 1/n, i.e. uniform φ) the leading-order expansion
/// of the cosine about E_p[z] = 0 gives:
///
///   E[cos] ≈ −(μ_3 / 2) · S2_k
///
/// where μ_3 = E[(z − z̄)³] / σ_z³ is the standardised third moment
/// (skewness) of z, and the negative sign comes from the sign of the
/// second-order term in the Taylor expansion of 1/√(E_p[z²]) around the
/// point E_p[z] = 0, E_p[z²] = 1.
///
/// Derivation sketch:
///   Let δ_i = z_i − z̄ (centred residuals, σ_z = 1 after standardisation).
///   cos = Σ_i p_i z_i / √(Σ_i p_i z_i²)
///       = Σ_i p_i δ_i / √(Σ_i p_i(1 + δ_i² + 2δ_i z̄ − z̄²))
///   Under independence and after isolating the O(S2_k) term, the mean
///   of Σ_i p_i δ_i vanishes (zero mean of z) but the covariance of the
///   numerator with the denominator's expansion produces a shift
///   proportional to E[δ_i³] = μ_3 and Σ_i p_i² = S2_k.
///
/// When BOTH blocks carry the same z, the shift cancels.  When NEITHER
/// carries a row-scaling, μ_3 = 0 (the raw-cosine case), and the formula
/// reduces to T11's symmetric form with shift = 0.
///
/// # Finite-sample correction
///
/// The raw (biased) third central moment estimator m_3 = Σ(z_i−z̄)³/n has
/// expectation μ_3 · σ³ · n(n−1)/n² + O(1/n²) under iid sampling.
/// The standard unbiased estimator uses the correction factor
///   n / ((n−1)(n−2))
/// (the G1 formula, identical to `scipy.stats.skew(bias=False)`).
/// For n ≥ 3, this gives a less biased estimator of μ_3.  For n < 3 we
/// return 0 (conservative — no correction applied, shift defaults to 0).
///
/// Returns μ_3 (dimensionless).  Returns 0.0 when σ_z ≤ 0 (constant z).
pub fn compute_skewness_mu3(z: &[f64]) -> f64 {
    let n = z.len();
    if n < 3 {
        return 0.0;
    }
    let mean = z.iter().sum::<f64>() / n as f64;
    let mut m2 = 0.0_f64;
    for &zi in z {
        let d = zi - mean;
        m2 += d * d;
    }
    m2 /= n as f64;
    let max_abs = z.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if m2 <= f64::EPSILON * max_abs.max(1.0).powi(2) {
        return 0.0;
    }
    let sigma = m2.sqrt();
    // Raw skewness, computed on standardized residuals with a wide symmetric
    // winsorization guard.  The audit uses skewness only to center a null-cosine
    // band; for symmetric heavy-tailed scalings (e.g. Student-t with undefined
    // third moment) a single leverage outlier must not masquerade as structural
    // skewness and shift the alias threshold.  The ±3.5σ guard is inert for
    // ordinary Gaussian/lognormal/Bernoulli audit scalings but makes the
    // statistic interpretable as a stable row-scaling asymmetry diagnostic.
    let raw_skew = z
        .iter()
        .map(|&zi| ((zi - mean) / sigma).clamp(-3.5, 3.5).powi(3))
        .sum::<f64>()
        / n as f64;
    // Finite-sample unbiased correction (G1 / Fisher-adjusted):
    //   μ_3_unbiased = (n / ((n−1)(n−2))) * n * raw_skew
    //   simplified to: raw_skew * n² / ((n−1)(n−2))
    let nf = n as f64;
    let correction = nf / ((nf - 1.0) * (nf - 2.0));
    correction * nf * raw_skew
}

/// Bias shift of the null cosine distribution for a pair of columns where
/// at most one of the two blocks carries a row-scaling vector z.
///
/// # Formula
///
/// Under independence of z and φ, and for standardised z, the null mean
/// of cos(φ_a, φ_b) when φ_b = z⊙φ_a is (to leading order in S2_k):
///
///   shift_k = −(μ_3 / 2) · S2_k
///
/// where S2_k = max(S2_a, S2_b) is the leverage concentration of the
/// more-concentrated column (the wider-null column controls the threshold
/// width; it also controls the shift magnitude via S2_k).
///
/// # When the shift is applied
///
/// * Both blocks carry the SAME row-scaling vector → the scaled and unscaled
///   columns are from the same distribution; the shift cancels exactly (shift = 0).
/// * Block A carries a `RowScaledJacobian` with scaling `z`, block B has none (or vice versa):
///   shift = −(μ_3(z) / 2) · S2_k.
/// * Both have `None`: shift = 0 (T11's symmetric form, μ_3 = 0).
/// * Both have DIFFERENT row-scaling vectors z_a ≠ z_b: shift is derived from
///   whichever of z_a or z_b produced the column with larger S2 (the dominant
///   concentration), as a conservative approximation.
///
/// The shift is clamped to ±0.5 to prevent a degenerate skewed z from
/// placing the null band entirely outside [−1, 1].
pub fn bias_shift_for_pair(z_a: Option<&[f64]>, z_b: Option<&[f64]>, s2_a: f64, s2_b: f64) -> f64 {
    // Both blocks have the same row scaling → shift cancels.
    match (z_a, z_b) {
        (Some(za), Some(zb)) if za.len() == zb.len() => {
            // Pointwise equality check: if the vectors are identical, shift = 0.
            let same = za.iter().zip(zb.iter()).all(|(a, b)| a == b);
            if same {
                return 0.0;
            }
        }
        (None, None) => return 0.0,
        _ => {}
    }
    // Identify which block's scaling to use for μ_3.
    // Use the block whose column has the LARGER S2 (dominates the null width)
    // because the shift formula is shift = −(μ_3/2)·S2_k and S2_k = max(S2_a, S2_b).
    let s2_dominant = s2_a.max(s2_b);
    let mu3 = if s2_a >= s2_b {
        match z_a {
            Some(z) => compute_skewness_mu3(z),
            None => match z_b {
                Some(z) => compute_skewness_mu3(z),
                None => 0.0,
            },
        }
    } else {
        match z_b {
            Some(z) => compute_skewness_mu3(z),
            None => match z_a {
                Some(z) => compute_skewness_mu3(z),
                None => 0.0,
            },
        }
    };
    // shift_k = −(μ_3 / 2) · S2_k
    let shift = -(mu3 / 2.0) * s2_dominant;
    // Clamp to ±0.5 to keep the band inside [−1, 1].
    shift.clamp(-0.5, 0.5)
}

/// Result of a gauge-priority-respecting rank-revealing factorization of a
/// joint design's Gram. Mirrors the shape of [`RrqrWithPermutation`] so it
/// drops straight into the existing attribution path: `column_permutation`
/// lists every original joint-column index, ACCEPTED (kept) columns first in
/// the order they were pivoted, then DEMOTED columns; `rank` is the count of
/// accepted columns; `rank_tol` is the absolute pivot tolerance.
pub(crate) struct PriorityTieredRank {
    pub(crate) rank: usize,
    pub(crate) column_permutation: Vec<usize>,
    pub(crate) rank_tol: f64,
}

/// Rank-revealing factorization of a joint design's Gram `G = JᵀJ (+ SᵀS)` that
/// honors `gauge_priority`: a higher-priority block's column is NEVER demoted in
/// favor of a collinear lower-priority block's column.
///
/// # Why a bespoke factorization
///
/// faer's `col_piv_qr` (used by both `rrqr_with_permutation` and
/// `rrqr_from_gram_with_permutation`) pivots purely by RESIDUAL NORM: at each
/// step it selects the not-yet-pivoted column with the largest residual,
/// regardless of any caller-supplied column order. Pre-permuting the columns
/// into descending-priority order therefore does NOT make priority govern the
/// kept/dropped decision — col-piv QR immediately re-sorts by norm, so a
/// higher-priority block whose columns carry a SMALLER norm than a collinear
/// lower-priority block is the one demoted (gam: the dynamic-wiggle block,
/// priority 100, lost all 6 columns to a priority-80 spatial block). The
/// gauge-priority contract — "the lower-priority block absorbs the alias drop" —
/// is unrepresentable through a single norm-pivoted QR.
///
/// # The factorization
///
/// Pivoted Cholesky on the Gram, constrained to descending priority TIERS. The
/// rank/pivot sequence of col-piv QR depends only on the column inner products
/// (the Gram), so a pivoted Cholesky on `G` reproduces the same per-pivot
/// residual magnitudes — but here we restrict each pivot choice to the highest
/// remaining priority tier, so all acceptable higher-priority columns are
/// committed to the kept basis BEFORE any lower-priority column is considered.
/// A lower-priority column is then accepted only if it still has residual norm
/// above tolerance AFTER projecting out every kept higher-or-equal-priority
/// column — i.e. it carries a genuinely new direction — and demoted otherwise.
/// Within one tier the choice is the standard largest-residual greedy pivot, so
/// numerical robustness inside a block matches col-piv QR.
///
/// `d[j]` tracks the current squared residual norm of column `j` (Schur
/// complement diagonal); `g[i][j]` is the running Schur-complemented Gram. The
/// tolerance matches the tall/`gram` RRQR paths exactly:
/// `rank_alpha · ε · max(m_rows, p) · max(√d_max⁰, 1)` where `√d_max⁰` is the
/// largest initial residual norm (= leading pivot magnitude `|R[0,0]|`).
pub(crate) fn priority_tiered_rank_from_gram(
    gram: &Array2<f64>,
    col_priority: &[u8],
    m_rows: usize,
    rank_alpha: f64,
) -> PriorityTieredRank {
    let p = gram.ncols();
    if p == 0 {
        return PriorityTieredRank {
            rank: 0,
            column_permutation: Vec::new(),
            rank_tol: 0.0,
        };
    }
    // Working Schur-complement copy of the Gram and its diagonal residuals.
    let mut g = gram.clone();
    let mut d: Vec<f64> = (0..p).map(|j| g[[j, j]].max(0.0)).collect();
    // Leading pivot magnitude = largest initial column norm = max √diagonal.
    let leading_diag = d.iter().cloned().fold(0.0_f64, f64::max).sqrt();
    let tol = rank_alpha * f64::EPSILON * (m_rows.max(p).max(1) as f64) * leading_diag.max(1.0);
    let tol_sq = tol * tol;

    // Distinct priority tiers in DESCENDING order (highest first). A column is
    // only eligible to pivot once every higher tier is exhausted.
    let mut tiers: Vec<u8> = col_priority.to_vec();
    tiers.sort_unstable_by(|a, b| b.cmp(a));
    tiers.dedup();

    let mut accepted: Vec<usize> = Vec::with_capacity(p);
    let mut demoted: Vec<usize> = Vec::new();
    let mut decided = vec![false; p];

    for &tier in &tiers {
        loop {
            // Largest-residual undecided column WITHIN this tier.
            let mut pivot: Option<usize> = None;
            let mut best = tol_sq;
            for j in 0..p {
                if decided[j] || col_priority[j] != tier {
                    continue;
                }
                if d[j] > best {
                    best = d[j];
                    pivot = Some(j);
                }
            }
            let Some(k) = pivot else { break };
            // Accept column k and eliminate it (one pivoted-Cholesky step):
            // Schur-complement the remaining columns against k.
            decided[k] = true;
            accepted.push(k);
            let pivot_val = d[k];
            if pivot_val <= 0.0 {
                continue;
            }
            // Row k of the current Schur Gram gives the coupling g[k][j].
            // Update g[i][j] -= g[i][k]*g[k][j]/pivot_val for undecided i,j,
            // and refresh residual diagonals d[j].
            let col_k: Vec<f64> = (0..p).map(|i| g[[i, k]]).collect();
            for i in 0..p {
                if decided[i] {
                    continue;
                }
                let gik = col_k[i];
                if gik == 0.0 {
                    continue;
                }
                let factor = gik / pivot_val;
                for j in 0..p {
                    if decided[j] {
                        continue;
                    }
                    g[[i, j]] -= factor * col_k[j];
                }
            }
            for j in 0..p {
                if !decided[j] {
                    d[j] = g[[j, j]].max(0.0);
                }
            }
        }
        // Every remaining undecided column in this tier has residual ≤ tol:
        // it lies in the span of the already-accepted (higher-or-equal-priority)
        // columns, so it is demoted here — never displacing a higher-priority one.
        for j in 0..p {
            if !decided[j] && col_priority[j] == tier {
                decided[j] = true;
                demoted.push(j);
            }
        }
    }

    let rank = accepted.len();
    let mut column_permutation = accepted;
    column_permutation.extend(demoted);
    PriorityTieredRank {
        rank,
        column_permutation,
        rank_tol: tol,
    }
}

/// Per-pair null cosine concentration scale.
///
/// σ = √(max(0, S2_max − 1/n)) where S2_max = max(S2_a, S2_b).
/// Taking the maximum of the two columns' S2 values selects the more
/// concentrated (smaller n_eff) column, which has the wider null
/// cosine distribution.  Using that wider scale for both the report
/// and halt thresholds prevents false positives when one column is
/// highly non-uniform.
fn pair_null_sigma(s2_a: f64, s2_b: f64, n: usize) -> f64 {
    let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f64 };
    let s2 = s2_a.max(s2_b);
    let excess = s2 - inv_n;
    // The null cosine variance is σ² = S2 − 1/n, an EXACT identity: a perfectly
    // uniform column has S2 = 1/n, so its null distribution has zero width. The
    // identity is computed from S2 = Σ φ⁴/(Σ φ²)², whose sequential f64
    // accumulation carries a rounding error bounded by ~n·ε relative to S2
    // (the standard summation bound). For a uniform column that residual is on
    // the order of 1.5e-17 at n=1000, S2=1e-3 — and because √ has unbounded
    // relative sensitivity at the origin, √(residual) inflates to ~3.9e-9, a
    // spurious σ that pulls the report/halt thresholds off their exact-uniform
    // values (gam#1397). The σ² identity is only meaningful above its own
    // accumulation noise, so we treat any excess at or below the summation
    // rounding scale as the exact zero it represents before taking the root.
    let n_terms = (n as f64).max(1.0);
    let rounding_floor = 16.0 * f64::EPSILON * n_terms * s2.abs().max(inv_n);
    if excess <= rounding_floor {
        return 0.0;
    }
    excess.sqrt()
}

/// Overlap threshold above which an `AliasedPair` is reported.
///
/// # K-multiplier and false-positive rate
///
/// The threshold is K_report · σ where σ = pair_null_sigma(s2_a, s2_b, n).
///
/// K_report is Bonferroni-corrected across the m_pairs total column pairs:
///   K_report = max(3, √(2 · ln(2 · m_pairs / α)))
/// with α = 0.05.  For m_pairs = 1 this gives K = 3 (three-sigma).
/// For m_pairs = 1000 (large large-scale audit) this gives K ≈ 5.1.
///
/// # Floor — the near-exact-alias boundary, NOT 0.10
///
/// The floor is the regime where σ → 0: both columns are near-uniform
/// (S2 ≈ 1/n), so the null cosine distribution is tightly concentrated and
/// has effectively no statistical width.  In that regime a moderate cosine
/// (e.g. the 0.745 uncentered cosine between a constant `1` column and a `x²`
/// column over a symmetric grid) is *ordinary correlation between two
/// distinct, fully-identifiable directions* — not aliasing.  Only a
/// near-exact cosine (≈ 1) is a genuine rank deficiency there.  A 0.10 floor
/// would flag any moderately-correlated pair of basis functions as an alias
/// (the WIP regression these constants replaced the fixed-0.95 report
/// threshold with).  We therefore floor at [`REPORT_FLOOR_NEAR_EXACT`] (the
/// near-exact-alias boundary) so σ → 0 approaches the exact-alias regime
/// rather than collapsing to a low value.  The ceiling 0.999 is the absolute
/// alias boundary.  For a column with n_eff = 100 and m_pairs = 100: σ ≈ 0.1,
/// K ≈ 4.3, K·σ ≈ 0.43 — but the band must still report a near-exact alias,
/// so the effective report threshold never drops below the floor.
fn pair_report_threshold(s2_a: f64, s2_b: f64, n: usize, m_pairs: usize) -> f64 {
    const ALIAS_BOUNDARY_COSINE: f64 = 0.999;
    const REPORT_BAND_FALSE_POSITIVE_RATE: f64 = 0.05;

    let sigma = pair_null_sigma(s2_a, s2_b, n);
    if sigma <= 0.0 {
        return REPORT_FLOOR_NEAR_EXACT;
    }
    let k_report = if m_pairs <= 1 {
        3.0_f64
    } else {
        (2.0 * (2.0 * m_pairs as f64 / REPORT_BAND_FALSE_POSITIVE_RATE).ln())
            .sqrt()
            .max(3.0)
    };
    // The statistical K·σ band is the *upper* bound on how wide the report
    // region may be; the near-exact-alias floor is the *lower* bound on the
    // overlap that may be called an alias when the null has little width. Take
    // the larger of the two so a wide-σ pair uses K·σ while a narrow-σ pair
    // (near-uniform columns) requires a near-exact cosine.
    (k_report * sigma)
        .max(REPORT_FLOOR_NEAR_EXACT)
        .min(ALIAS_BOUNDARY_COSINE)
}

/// Overlap threshold above which the audit halts the fit for this pair.
///
/// # K-multiplier and false-positive rate
///
/// The threshold is K_halt · σ where σ = pair_null_sigma(s2_a, s2_b, n).
///
/// K_halt = 10.0 (≈ √(2 · ln(2M / α)) for M ~ 1000 pairs and α = 1e-6).
/// For a column with n_eff = 100: σ ≈ 0.1, halt threshold ≈ 1.0 → clamped
/// to 0.999 (only near-exact aliases halt for wide-null columns).
/// For n_eff = 10000: σ ≈ 0.01, threshold = 0.10 (moderate overlaps
/// that are wildly outside the null distribution at that resolution).
///
/// The ceiling 0.999 ensures that floating-point near-exact aliases
/// (cos = 0.9999…) always fire the halt.  The floor 0.05 prevents
/// pathological over-sensitivity on very long columns.
fn pair_halt_threshold(s2_a: f64, s2_b: f64, n: usize) -> f64 {
    const ALIAS_BOUNDARY_COSINE: f64 = 0.999;

    let sigma = pair_null_sigma(s2_a, s2_b, n);
    if sigma <= 0.0 {
        return ALIAS_BOUNDARY_COSINE;
    }
    // Floor 0.05 prevents pathological over-sensitivity on very long columns.
    (10.0_f64 * sigma).clamp(0.05, ALIAS_BOUNDARY_COSINE)
}

/// Decide whether a cosine (signed) falls outside the bias-corrected null band.
///
/// The null distribution for cos(φ_a, φ_b) is approximately
///   N(shift, σ²)
/// where σ = pair_null_sigma(s2_a, s2_b, n) and
/// shift = bias_shift_for_pair(...).
///
/// Returns `true` when |cosine − shift| ≥ half_width.
///
/// The `half_width` argument is the K·σ half-band (either the report or
/// halt multiplied sigma).
fn cosine_outside_null_band(cosine: f64, shift: f64, half_width: f64) -> bool {
    (cosine - shift).abs() >= half_width
}

/// Compute the signed cosine between two normalised column vectors.
///
/// Returns `dot / (norm_a * norm_b)`.  The caller guarantees both norms
/// are positive.  `dot` is the raw inner product (may be negative).
fn signed_cosine(dot: f64, norm_a: f64, norm_b: f64) -> f64 {
    // Clamp to [-1, 1] to guard against floating-point rounding.
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Run the pre-fit cross-block identifiability audit on a finalised
/// list of `ParameterBlockSpec`s. The caller is responsible for
/// having applied any within-smooth `joint_null_rotation`; the audit
/// inspects the post-rotation columns only.
///
/// The algorithm:
///   1. Densify each block once (chunked, n × p_block; large-scale n,
///      small p_block — total joint width is the GAM smooth budget,
///      not n). Each block's penalty-aware numerical rank is recorded
///      as `design_range_rank`.
///   2. Stack horizontally into `X_joint ∈ ℝ^{n×p_total}` in spec
///      order; column-pivoted QR identifies columns linearly
///      dependent on earlier (pivot-rank-truncated) columns.
///   3. Each pivoted-out column is attributed to the earliest block
///      whose range absorbs it (largest projection norm). Ambiguous
///      attribution → `fatal = true`.
///   4. Report all (a, b) column pairs whose normalised inner
///      product exceeds the per-pair leverage-based reporting threshold
///      (`pair_report_threshold`).
pub fn audit_identifiability(
    specs: &[ParameterBlockSpec],
) -> Result<IdentifiabilityAudit, EstimationError> {
    // Default at-init linearization state: β = 0, no family scalars.
    // Families whose callbacks need β-dependent scalars must call
    // `audit_identifiability_with_state` directly with the current β.
    let p_total_hint: usize = specs.iter().map(|s| s.design.ncols()).sum();
    let zeros_beta = vec![0.0f64; p_total_hint];
    let init_state = FamilyLinearizationState {
        beta: &zeros_beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    audit_identifiability_impl(specs, &init_state)
}

/// Implementation body shared by [`audit_identifiability`] and
/// [`audit_identifiability_with_state`]. The two public entry points differ
/// only in how they construct the [`FamilyLinearizationState`] — every other
/// step (gauge-priority sort, joint RRQR, pairwise overlap scan, drop
/// attribution, gauge-resolves-rank-deficiency logic, hard-alias / fatal
/// classification) is identical, so it lives here once.
fn audit_identifiability_impl(
    specs: &[ParameterBlockSpec],
    state: &FamilyLinearizationState<'_>,
) -> Result<IdentifiabilityAudit, EstimationError> {
    if specs.is_empty() {
        return Ok(IdentifiabilityAudit {
            blocks: Vec::new(),
            aliased_pairs: Vec::new(),
            dropped_columns: Vec::new(),
            fatal: false,
            summary: "identifiability audit: no blocks supplied".to_string(),
        });
    }

    // Materialise each block's effective Jacobian first; the row-equality
    // invariant must apply to what the audit actually inspects
    // (the effective design exposed through `effective_jacobian_at`),
    // not to the raw `spec.design.nrows()`.  With the canonical-row
    // architecture `spec.design.nrows() == n_obs` is guaranteed by
    // construction; the survival LS stacked operator lives in
    // `spec.stacked_design`, which the audit never reads.  The
    // `effective_jacobian_at` route is still used here because
    // multi-output blocks (e.g. marginal-slope) produce `n_obs * k` rows.
    let mut dense_blocks: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        let beta_start = specs
            .iter()
            .take(idx)
            .map(|s| s.design.ncols())
            .sum::<usize>();
        let beta_end = beta_start + spec.design.ncols();
        let beta_block = if state.beta.len() >= beta_end {
            &state.beta[beta_start..beta_end]
        } else if state.beta.len() == spec.design.ncols() {
            // Allow direct block-local states in tests and one-off callers.
            state.beta
        } else {
            &[]
        };
        let block_state = FamilyLinearizationState {
            beta: beta_block,
            family_scalars: state.family_scalars.clone(),
            channel_hessian: state.channel_hessian.clone(),
            probit_frailty_scale: state.probit_frailty_scale,
        };
        let dense = spec
            .effective_jacobian_at(
                "identifiability::audit::audit_identifiability",
                &block_state,
            )
            .map_err(|e| EstimationError::LayoutError(format!("identifiability audit: {e}")))?;
        dense_blocks.push(dense);
    }
    // The per-observation row count is the count shared by the linear-predictor
    // blocks (`n`, or `n·k` for multi-output families). A GLOBAL-SCALAR parameter
    // — e.g. the lognormal frailty log-SD (#723), which parameterises the
    // integrated-out frailty distribution rather than any per-observation linear
    // predictor — honestly contributes a 1-row effective Jacobian and cannot
    // row-align with the per-observation blocks. Take `n` as the max row count
    // (the per-obs count) and admit blocks with FEWER rows as global-scalar: they
    // are audited for within-block rank below (identified iff their small
    // Jacobian has full column rank) and zero-padded into the joint cross-block
    // design, where their disjoint row support keeps them from aliasing a
    // per-observation column. For the homogeneous case (every block has `n` rows,
    // the historical invariant) `n` equals the old `dense_blocks[0].nrows()` and
    // every block fills all `n` rows, so this is bit-identical to the prior
    // equality check + assembly.
    let n = dense_blocks
        .iter()
        .map(|d| d.nrows())
        .max()
        .expect("dense_blocks is non-empty: specs.is_empty() returned early above");
    for (idx, dense) in dense_blocks.iter().enumerate() {
        if dense.nrows() == 0 || dense.nrows() > n {
            return Err(EstimationError::LayoutError(format!(
                "identifiability audit: block {} ({}) has {} effective-Jacobian rows, expected 1..={}",
                idx,
                specs[idx].name,
                dense.nrows(),
                n,
            )));
        }
    }

    // Structural (λ-invariant) penalty per block, used to make the rank
    // verdicts penalty-aware: a penalized direction that is design-null is still
    // identified (`JᵀWJ + S` non-singular there), so the audit must rank `[J; S]`
    // rather than `J`. `None` for unpenalized blocks ⇒ the historical raw-design
    // verdict is preserved exactly.
    let block_penalties: Vec<Option<Array2<f64>>> =
        specs.iter().map(block_structural_penalty_dense).collect();

    let mut blocks: Vec<BlockIdentity> = Vec::with_capacity(specs.len());
    let mut col_offsets: Vec<usize> = Vec::with_capacity(specs.len() + 1);
    col_offsets.push(0);
    let block_phase_started = std::time::Instant::now();
    // Per-block penalty-aware RRQR ranks. Each block's rank is a fully
    // INDEPENDENT factorisation of `[J_block; S_block]` ((n+p_block) × p_block);
    // nothing in one block's RRQR reads another block's result, and the joint
    // assembly below consumes only the scalar rank plus the (already-known)
    // column counts. Computing them with a parallel map over blocks is therefore
    // BIT-IDENTICAL to the previous serial loop — `block_penalty_aware_rank` is
    // pure and called with the same `(dense, structural_penalty)` arguments — but
    // overlaps the n-scale per-block factorisations (the ~0.8s biobank
    // per-block-QR phase) across the rayon pool instead of running them serially.
    // A single faer col-piv QR on a tall-thin matrix is mostly BLAS-2 over the
    // tiny p_block trailing panel, so fanning the blocks out wins where one block
    // alone cannot saturate the pool.
    let block_ranks: Vec<usize> = {
        use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
        let results: Result<Vec<usize>, EstimationError> = dense_blocks
            .par_iter()
            .zip(block_penalties.par_iter())
            .map(|(dense, penalty)| block_penalty_aware_rank(dense, penalty.as_ref()))
            .collect();
        results?
    };
    for (idx, spec) in specs.iter().enumerate() {
        let p_block = dense_blocks[idx].ncols();
        // Penalty-aware, rank-revealing: rank of `[J; S]`, so penalty-covered
        // (design-null but regularized) directions count as identified. RRQR is
        // rank-revealing (the prior plain-QR diagonal was not), so the reported
        // range_rank is now an honest numerical rank.
        blocks.push(BlockIdentity {
            block_name: spec.name.clone(),
            original_dim: p_block,
            effective_dim: p_block,
            design_range_rank: block_ranks[idx],
        });
        let next_offset = col_offsets[col_offsets.len() - 1] + p_block;
        col_offsets.push(next_offset);
    }
    log::info!(
        "[STAGE] identifiability audit: per-block QR complete blocks={} elapsed={:.3}s",
        specs.len(),
        block_phase_started.elapsed().as_secs_f64(),
    );
    let p_total = *col_offsets.last().expect("col_offsets non-empty");

    // Permanent layout diagnostic: name every block with its gauge priority,
    // global column span, raw column count, and within-block range rank. A
    // block whose `range_rank < original_dim` carries a redundancy that
    // survived within-smooth nullspace absorption — i.e. an INTRA-block alias
    // (e.g. a smooth's polynomial-nullspace constant colliding with the
    // parametric intercept). Intra-block deficiencies cannot be resolved by
    // cross-block gauge_priority ordering, so surfacing them here turns an
    // opaque "first attributed drop: block X local column N" into an immediate
    // "block X is internally rank-deficient" signal.
    {
        let layout: Vec<String> = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let deficient = if b.design_range_rank < b.original_dim {
                    "  ⚠WITHIN-BLOCK-DEFICIENT"
                } else {
                    ""
                };
                format!(
                    "{}(prio={}, cols[{}..{}], dim={}, range_rank={}{})",
                    b.block_name,
                    specs[i].gauge_priority,
                    col_offsets[i],
                    col_offsets[i + 1],
                    b.original_dim,
                    b.design_range_rank,
                    deficient,
                )
            })
            .collect();
        log::info!(
            "[STAGE] identifiability audit: block layout p_total={} | {}",
            p_total,
            layout.join(" | "),
        );
    }

    if p_total == 0 {
        return Ok(IdentifiabilityAudit {
            blocks,
            aliased_pairs: Vec::new(),
            dropped_columns: Vec::new(),
            fatal: false,
            summary: "identifiability audit: every block is empty".to_string(),
        });
    }

    // Per-block effective rank-geometry design. A block that carries a
    // multi-channel `stacked_design` (survival location-scale / latent-survival
    // time-transform: its solver eta is the `k·n`-row `[entry; exit; deriv]·β`
    // operator) has its TRUE column geometry spread across all `k` channels, not
    // in the `n`-row canonical `design` (which is only the channel-the-audit-saw
    // slice). Feeding the audit the `n`-row `design` mis-represents that span: a
    // time column that looks collinear with another block's covariate in the
    // first `n` rows is genuinely distinguished by the deriv/entry channels
    // (gam#1197). Use the `k·n`-row stacked operator as the block's effective
    // rank/alias geometry whenever present; plain blocks are aligned to it below.
    // `stacked_design` has the SAME column count as `design`, so the joint column
    // layout (`col_offsets`) is unchanged.
    let block_effective_designs: Vec<std::borrow::Cow<'_, Array2<f64>>> = specs
        .iter()
        .enumerate()
        .map(|(idx, spec)| match spec.stacked_design.as_ref() {
            Some(stacked) => stacked
                .try_to_dense_arc("identifiability::audit stacked_design rank geometry")
                .map(|arc| std::borrow::Cow::Owned(arc.as_ref().clone()))
                .unwrap_or_else(|_| std::borrow::Cow::Borrowed(&dense_blocks[idx])),
            None => std::borrow::Cow::Borrowed(&dense_blocks[idx]),
        })
        .collect();
    // Joint row count spans the tallest effective block (a `k·n`-row stacked
    // operator).
    let r_joint = block_effective_designs
        .iter()
        .map(|d| d.nrows())
        .max()
        .unwrap_or(n)
        .max(n);

    // ── Observation-channel alignment for plain blocks (gam#1197) ───────────
    //
    // A stacked block's `k·n`-row operator partitions the joint row space into
    // `k` per-observation bands of `n` rows (e.g. `[entry; exit; deriv]`). A
    // plain per-observation block (no `stacked_design`) contributes to the SAME
    // additive predictor the stacked block evaluates, so its true effective
    // image is NOT just rows `0..n` (which coincide with only the FIRST band):
    // it is replicated across every OBSERVATION band — the bands in which the
    // predictor (not its derivative) is evaluated — and is zero in pure-
    // derivative bands. For a time-invariant covariate this matches the family's
    // own `x_threshold_entry = x_threshold`, `x_threshold_deriv = None` wiring.
    //
    // Without this alignment, packing a plain block into rows `0..n` only made
    // its intercept land in the stacked block's ENTRY band alone, so it was no
    // longer collinear with the stacked block's additive constant (which spans
    // entry AND exit) — the redundant shared intercept stopped being detected,
    // the joint ranked full, and the genuinely-aliased constant was never
    // demoted (then the downstream MAP-uniqueness check fired on the still-
    // collinear predictor space). Replicating the plain block across the
    // observation bands restores the intercept alias while the stacked block's
    // derivative band keeps a genuine covariate (e.g. `age`) distinguished.
    //
    // Observation bands are detected structurally from the tallest stacked
    // block: a band `b` is an observation band iff the stacked operator carries
    // an (approximately) constant non-zero column over that band's rows — i.e.
    // the additive intercept is evaluated there. The derivative band annihilates
    // constants, so its constant column is ~0 and it is correctly excluded. With
    // no stacked block present, `k_bands == 1` and this is the historical
    // single-band packing, bit-identical for plain GAM designs.
    let k_bands = (r_joint / n).max(1);
    let observation_bands: Vec<usize> = if k_bands <= 1 {
        vec![0]
    } else {
        // Use the tallest stacked block to define the band partition.
        let stacked_ref = block_effective_designs
            .iter()
            .max_by_key(|d| d.nrows())
            .expect("non-empty blocks");
        let mut bands: Vec<usize> = (0..k_bands)
            .filter(|&b| {
                let lo = b * n;
                let hi = ((b + 1) * n).min(stacked_ref.nrows());
                if hi <= lo {
                    return false;
                }
                // Band is an observation band iff the block's INTERCEPT column
                // (column 0, the additive constant by the universal
                // intercept-first convention) is ~constant and non-zero over it
                // — i.e. the additive constant is evaluated there. In a pure
                // derivative band the intercept coefficient maps to ZERO (the
                // derivative of a constant), so column 0 is ~0 there and the band
                // is correctly excluded — even though the band may still carry a
                // constant column for a NON-intercept basis coefficient (e.g. the
                // unit derivative of a linear time term), which must NOT count as
                // an observation band.
                if stacked_ref.ncols() == 0 {
                    return false;
                }
                let first = stacked_ref[[lo, 0]];
                first.abs() > 1e-12
                    && (lo..hi)
                        .all(|r| (stacked_ref[[r, 0]] - first).abs() <= 1e-9 * first.abs().max(1.0))
            })
            .collect();
        if bands.is_empty() {
            // Degenerate: no detectable constant-bearing band (no intercept in
            // the stacked operator). Fall back to band 0 so plain blocks keep
            // their historical n-row support rather than vanishing.
            bands.push(0);
        }
        bands
    };

    let mut x_joint = Array2::<f64>::zeros((r_joint, p_total));
    for (idx, block) in block_effective_designs.iter().enumerate() {
        let start = col_offsets[idx];
        let end = col_offsets[idx + 1];
        if end == start {
            continue;
        }
        let br = block.nrows();
        if specs[idx].stacked_design.is_some() || br != n {
            // Stacked blocks (own multi-band geometry) and global-scalar blocks
            // (rows < n, disjoint support) pack at their native rows `..br`.
            x_joint
                .slice_mut(ndarray::s![..br, start..end])
                .assign(block.as_ref());
        } else {
            // Plain per-observation block: replicate its `n`-row design into
            // every observation band so its intercept aligns with the stacked
            // block's additive constant across the same bands (gam#1197).
            for &b in &observation_bands {
                let lo = b * n;
                let hi = (lo + n).min(r_joint);
                if hi > lo {
                    x_joint
                        .slice_mut(ndarray::s![lo..hi, start..end])
                        .assign(&block.slice(ndarray::s![..(hi - lo), ..]));
                }
            }
        }
    }

    // Joint Gram G = Xᵀ·X of the UNAUGMENTED design, computed once with the
    // blocked, parallel faer crossproduct. This is the single n-scale pass over
    // the joint design — it serves BOTH the joint RRQR rank verdict (squared into
    // the Gram, see `joint_gram_aug` + `rrqr_from_gram_with_permutation` below)
    // AND the pairwise overlap scan further down (which reads `joint_gram[[ja,
    // jb]]` as an O(1) cross-block dot product). Computing it here, before the
    // RRQR, removes the redundant second full-design stream that the tall
    // `rrqr_with_permutation(&x_joint_rank_input, …)` performed: at biobank scale
    // (n≈2·10⁵) re-streaming the 194k×85 / 194k×45 design for the rank verdict was
    // the dominant ~0.94s joint-RRQR cost. Mathematically G[[ja, jb]] = Σ_i
    // X[i,ja]·X[i,jb] = caᵀcb, exactly the column geometry col-piv QR consumes.
    let joint_gram = {
        let unit_weights = Array1::<f64>::ones(r_joint);
        crate::linalg::faer_ndarray::fast_xt_diag_x_with_parallelism(
            &x_joint,
            &unit_weights,
            faer::get_global_parallelism(),
        )
    };

    // Penalty-augmented joint design for the RANK verdict only: stack the
    // block-diagonal structural penalties beneath `x_joint`, so the joint rank
    // is `rank([X_joint; S_blockdiag])` and the only fatal deficiencies are
    // directions that are BOTH data-null AND penalty-null (`ker(J) ∩ ker(S)`).
    // The pairwise overlap scan below keeps using the *unaugmented* `x_joint`
    // (penalty rows would distort the data-correlation cosines it measures);
    // only the RRQR rank/attribution consumes this augmented matrix. When no
    // block carries a penalty the augmented section is empty and this is bit-
    // identical to RRQR on `x_joint` — unpenalized families are unaffected.
    let n_penalty_rows: usize = block_penalties
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            s.as_ref()
                .map_or(0, |_| col_offsets[idx + 1] - col_offsets[idx])
        })
        .sum();
    // Gram of the penalty-augmented joint design, `Gₐ = [X_joint; S_blockdiag]ᵀ ·
    // [X_joint; S_blockdiag]`. Because the penalty rows are stacked block-
    // diagonally beneath `x_joint`, this equals `Xᵀ·X + Σ_block Sᵀ·S` placed into
    // the owning block's diagonal sub-square: no second n-row stream is needed —
    // we add the tiny per-block `SᵀS` (p_block × p_block) onto the already-
    // computed `joint_gram`. `priority_tiered_rank_from_gram` runs a priority-
    // tiered pivoted Cholesky on this Gram; its rank cut equals col-piv QR on the
    // tall `[X_joint; S_blockdiag]` (both depend only on the column inner
    // products). The `m_rows` it needs for the tolerance scaling is the tall row
    // count `r_joint + n_penalty_rows`.
    let joint_gram_aug: Array2<f64> = if n_penalty_rows == 0 {
        joint_gram.clone()
    } else {
        let mut g = joint_gram.clone();
        for (idx, s_opt) in block_penalties.iter().enumerate() {
            if let Some(s) = s_opt {
                let start = col_offsets[idx];
                let end = col_offsets[idx + 1];
                // SᵀS for this block's diagonal sub-square; S is (h × h) here
                // (`block_structural_penalty_dense` returns a p_block-square
                // structural penalty), so SᵀS is h × h.
                let sts = fast_atb(s, s);
                let mut sub = g.slice_mut(ndarray::s![start..end, start..end]);
                sub += &sts;
            }
        }
        g
    };
    let joint_rank_m_rows = r_joint + n_penalty_rows;

    // The gauge-ownership contract is realised by `priority_tiered_rank_from_gram`
    // below, which pivots strictly within descending gauge_priority tiers (NOT by
    // a column reorder fed to a norm-pivoted QR, which would ignore it). This is
    // invariant to spec-list order — Python custom families, `bms/install_flex.rs`,
    // and the `audit_priority_perm_invariance` tests all pass scrambled spec lists
    // and must see the SAME verdict and drop attribution. For survival
    // marginal-slope it drops the shared affine direction from marginal_surface
    // (not the higher-priority time_surface) and the shared deviation direction
    // from score_warp_dev (the lowest priority); with all priorities equal the
    // factorization degenerates to a single ordinary pivoted-Cholesky tier, so
    // legacy equal-priority callers are unaffected.
    let col_block_idx: Vec<usize> = (0..specs.len())
        .flat_map(|i| std::iter::repeat(i).take(col_offsets[i + 1] - col_offsets[i]))
        .collect();
    // Per-joint-column gauge priority, inherited from the owning block.
    let col_priority: Vec<u8> = col_block_idx
        .iter()
        .map(|&bi| specs[bi].gauge_priority)
        .collect();

    // GAUGE-PRIORITY-RESPECTING rank-revealing factorization of the
    // penalty-augmented joint Gram. The rank and the demoted-column attribution
    // reflect `ker(J) ∩ ker(S)`, not raw `ker(J)`, because the Gram is
    // `JᵀJ + SᵀS`.
    //
    // A plain column-pivoted QR (faer `col_piv_qr`, used by both
    // `rrqr_with_permutation` and `rrqr_from_gram_with_permutation`) pivots by
    // RESIDUAL NORM and ignores any caller-supplied column order, so merely
    // pre-sorting columns into descending-priority order does NOT make priority
    // govern which block keeps a shared direction — col-piv QR re-sorts by norm
    // and a higher-priority block whose columns carry a smaller norm than a
    // collinear lower-priority block is the one demoted (gam: the dynamic-wiggle
    // block at priority 100 lost all 6 of its columns to a priority-80 spatial
    // block, surfacing as "dynamic wiggle design col mismatch: got 6, expected
    // 0"). `priority_tiered_rank_from_gram` instead pivots strictly within
    // descending priority TIERS, committing every acceptable higher-priority
    // column to the kept basis before any lower-priority column is considered, so
    // the lower-priority block always absorbs the alias drop — the canonical-
    // gauge contract this audit is built on. It returns the demoted columns in
    // ORIGINAL joint-column indices, so no priority<->original remap is needed.
    let alpha = default_rrqr_rank_alpha();
    let rrqr_started = std::time::Instant::now();
    let block_priority_summary: Vec<String> = specs
        .iter()
        .map(|s| format!("{}={}", s.name, s.gauge_priority))
        .collect();
    log::info!(
        "[STAGE] identifiability audit: joint priority-tiered RRQR start n={} p_total={} \
         blocks=[{}]",
        n,
        p_total,
        block_priority_summary.join(", "),
    );
    let tiered =
        priority_tiered_rank_from_gram(&joint_gram_aug, &col_priority, joint_rank_m_rows, alpha);
    log::info!(
        "[STAGE] identifiability audit: joint priority-tiered RRQR end rank={}/{} elapsed={:.3}s",
        tiered.rank,
        p_total,
        rrqr_started.elapsed().as_secs_f64(),
    );
    let joint_rank = tiered.rank;
    let joint_rank_tol = tiered.rank_tol;
    let demoted_joint_cols: Vec<usize> = tiered.column_permutation[tiered.rank..].to_vec();

    // Pairwise overlap report on the joint design's normalised
    // columns. O(p_total² · n) — fine at GAM smooth widths. We only
    // record pairs whose blocks are distinct (within-block aliasing
    // is the within-smooth nullspace problem, owned by nullspace-lead).
    //
    // Thresholds are column-specific, derived from the leverage
    // concentration S2_k = Σ_i p_i² (p_i = φ_i²/‖φ‖²).  See the
    // doc-comments on `compute_leverage_s2`, `pair_report_threshold`,
    // and `pair_halt_threshold` for the underlying finite-sample identity.
    //
    // Bias correction: when one block carries a RowScaledJacobian with scaling z,
    // the null distribution of the cross-block cosine is no longer centred
    // at 0.  The null mean is approximately shift_k = −(μ_3/2)·S2_k where
    // μ_3 is the standardised third moment of z (skewness).  We test
    // |cosine − shift_k| >= half_width instead of |cosine| >= half_width.
    // See `bias_shift_for_pair` and `compute_skewness_mu3` for the derivation.
    let pairwise_started = std::time::Instant::now();
    log::info!(
        "[STAGE] identifiability audit: pairwise overlap scan start n={} p_total={} blocks={}",
        n,
        p_total,
        specs.len(),
    );
    // Per-column norm and leverage concentration S2_k = Σ_i φ_i⁴ / (Σ_i φ_i²)².
    // Each column is independent, so the p_total O(n) passes run as a parallel
    // map over columns. The per-column arithmetic (sequential `Σ φ²` for the
    // norm, `compute_leverage_s2` for S2) is UNCHANGED from the prior serial
    // loop, so the result is BIT-IDENTICAL — only the iteration over columns is
    // parallelised.
    let (col_norms, col_s2): (Array1<f64>, Array1<f64>) = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pairs: Vec<(f64, f64)> = (0..p_total)
            .into_par_iter()
            .map(|j| {
                let col = x_joint.column(j);
                let nrm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
                let s2 = compute_leverage_s2(&col);
                (nrm, s2)
            })
            .collect();
        let mut norms = Array1::<f64>::zeros(p_total);
        let mut s2s = Array1::<f64>::zeros(p_total);
        for (j, (nrm, s2)) in pairs.into_iter().enumerate() {
            norms[j] = nrm;
            s2s[j] = s2;
        }
        (norms, s2s)
    };
    // Total number of cross-block column pairs for Bonferroni correction.
    let total_cross_pairs: usize = {
        let mut cnt = 0usize;
        for a_idx in 0..specs.len() {
            let a_cols = col_offsets[a_idx + 1] - col_offsets[a_idx];
            for b_idx in (a_idx + 1)..specs.len() {
                let b_cols = col_offsets[b_idx + 1] - col_offsets[b_idx];
                cnt = cnt.saturating_add(a_cols.saturating_mul(b_cols));
            }
        }
        cnt.max(1)
    };
    let pairwise_block_progress_ticker = (n.saturating_mul(p_total)
        >= AUDIT_PROGRESS_TICKER_WORK_THRESHOLD)
        .then(crate::util::loop_progress::LoopProgress::default_interval);
    // The full joint Gram `G = Xᵀ·X` was already assembled once (before the joint
    // RRQR) and is reused here: every cross-block column dot product below is an
    // O(1) lookup `joint_gram[[ja, jb]]` instead of an O(n) scalar pass, so the
    // pairwise scan is O(p²) scalar bookkeeping over the shared GEMM. The single
    // n-row stream now feeds BOTH the rank verdict and the overlap scan.
    // The pairwise scan produces TWO independent classifications per cross-block
    // column pair, because reporting and halting answer different questions and
    // must not be coupled:
    //
    //   • `aliased_pairs` (REPORT band) — human-facing diagnostics. Floored at
    //     `REPORT_FLOOR_NEAR_EXACT` (0.95) so that ordinary moderate correlation
    //     between two distinct, fully-identifiable basis functions is not paraded
    //     as an alias.
    //   • `halt_pairs` (HALT band) — the structural fittability verdict. A pure
    //     leverage-scaled K·σ band with NO near-exact floor, because the question
    //     "is this design fittable?" has nothing to do with whether the overlap is
    //     visually near 1: for a tightly-concentrated null (high n_eff) a cosine of
    //     0.71 is already many σ outside the null and unfittable, while for a wide
    //     null (low n_eff) even 0.9 is ordinary sampling noise.
    //
    // The halt band can — and routinely does — sit BELOW the report floor (e.g.
    // n_eff≈200 → halt half-width ≈0.71 < 0.95). Deriving the halt verdict by
    // filtering the REPORTED set therefore silently discards exactly the pairs in
    // the (halt, report) gap that are unfittable yet below the diagnostic floor
    // (gam#1397). Each band is computed directly from the cosine here, so the two
    // verdicts are fully decoupled.
    let mut aliased_pairs: Vec<AliasedPair> = Vec::new();
    let mut halt_pairs: Vec<AliasedPair> = Vec::new();
    let n_block_pairs = specs.len().saturating_mul(specs.len().saturating_sub(1)) / 2;
    for a_block_idx in 0..specs.len() {
        let a_start = col_offsets[a_block_idx];
        let a_end = col_offsets[a_block_idx + 1];
        // Extract the row-scaling vector for block A (used for the bias shift).
        // Only RowScaledJacobian callbacks expose this; all other callbacks return None.
        let z_a_arc = specs[a_block_idx]
            .jacobian_callback
            .as_ref()
            .and_then(|cb| cb.eta_row_scaling_for_skewness());
        let z_a: Option<&[f64]> = z_a_arc.as_deref();
        for b_block_idx in (a_block_idx + 1)..specs.len() {
            let b_start = col_offsets[b_block_idx];
            let b_end = col_offsets[b_block_idx + 1];
            // Extract the row-scaling vector for block B.
            let z_b_arc = specs[b_block_idx]
                .jacobian_callback
                .as_ref()
                .and_then(|cb| cb.eta_row_scaling_for_skewness());
            let z_b: Option<&[f64]> = z_b_arc.as_deref();
            for ja in a_start..a_end {
                let na = col_norms[ja];
                if na == 0.0 {
                    continue;
                }
                for jb in b_start..b_end {
                    let nb = col_norms[jb];
                    if nb == 0.0 {
                        continue;
                    }
                    // Precomputed Gram entry caᵀcb (see joint_gram above).
                    let dot = joint_gram[[ja, jb]];
                    // Signed cosine: preserves direction for bias-shift test.
                    let cosine = signed_cosine(dot, na, nb);
                    let s2_ja = col_s2[ja];
                    let s2_jb = col_s2[jb];
                    // Bias shift: non-zero when exactly one block carries
                    // row-scaling (or the two scalings differ).
                    let shift = bias_shift_for_pair(z_a, z_b, s2_ja, s2_jb);
                    // Store the unsigned |cosine| in AliasedPair.overlap for
                    // backwards compatibility and human-readable diagnostics.
                    // Store `shift` so each band applies the same directional
                    // (skewness) correction to the null mean.
                    let overlap = cosine.abs();
                    let make_pair = || AliasedPair {
                        block_a: specs[a_block_idx].name.clone(),
                        block_b: specs[b_block_idx].name.clone(),
                        direction_a: ja - a_start,
                        direction_b: jb - b_start,
                        overlap,
                        bias_shift: shift,
                    };
                    // REPORT band (diagnostics, floored at the near-exact-alias
                    // boundary).
                    let report_half_width =
                        pair_report_threshold(s2_ja, s2_jb, n, total_cross_pairs);
                    if cosine_outside_null_band(cosine, shift, report_half_width) {
                        aliased_pairs.push(make_pair());
                    }
                    // HALT band (structural fittability, pure leverage K·σ with
                    // no near-exact floor). Computed directly from the same cosine
                    // so it is independent of the report verdict (gam#1397).
                    let halt_half_width = pair_halt_threshold(s2_ja, s2_jb, n);
                    if cosine_outside_null_band(cosine, shift, halt_half_width) {
                        halt_pairs.push(make_pair());
                    }
                }
            }
            if let Some(ticker) = pairwise_block_progress_ticker.as_ref() {
                ticker.tick(1, |done, secs| {
                    log::info!(
                        "[STAGE] identifiability audit: pairwise overlap progress {done}/{n_block_pairs} block pairs in {secs:.1}s",
                    );
                });
            }
        }
    }
    log::info!(
        "[STAGE] identifiability audit: pairwise overlap scan done in {:.3}s ({} aliased pairs)",
        pairwise_started.elapsed().as_secs_f64(),
        aliased_pairs.len(),
    );

    // Attribute each demoted joint column back to its canonical gauge owner.
    //
    // RRQR is priority-ordered, but column pivoting can still name the
    // higher-priority representative of a two-block alias. The model-space
    // redundancy is resolved by dropping the lower-priority participant, so
    // when RRQR has named a *higher*-priority column we re-attribute to the
    // lower-priority partner.  Same-priority aliases remain attributed to
    // the raw demoted column and are fatal below.
    //
    // # ≥3-way alias correctness
    //
    // For a 3-block shared direction (e.g. [high=200, mid=150, low=80] with
    // a common constant column), RRQR demotes the two lower-priority
    // representatives (mid and low). The raw demoted column for mid is on
    // the LOWER side of the (high, mid) alias pair and on the HIGHER side
    // of the (mid, low) alias pair. A `max_by(overlap)` selection over
    // matching distinct-priority pairs would tie-break to the *last* pair
    // in iteration order and re-attribute the mid drop to low — which
    // double-counts the low drop and leaves the mid loss unattributed.
    //
    // The correct rule: if the raw demoted column is already on the
    // lower-priority side of at least one matching distinct-priority
    // alias pair, RRQR's placement agrees with gauge ownership and the
    // attribution stays on the raw block. Only when the raw column is on
    // the higher-priority side of EVERY matching distinct-priority pair
    // (RRQR named the canonical owner) do we re-attribute to the
    // best-overlap partner's lower-priority side.
    let block_priority_for_attribution: std::collections::HashMap<&str, u8> = specs
        .iter()
        .map(|s| (s.name.as_str(), s.gauge_priority))
        .collect();
    let mut dropped_columns: Vec<DroppedColumn> = Vec::new();
    for &joint_col in &demoted_joint_cols {
        let (block_idx, local_col) = locate_block_column(&col_offsets, joint_col)?;
        let raw_block_name = specs[block_idx].name.clone();
        let raw_priority = specs[block_idx].gauge_priority;

        let pair_priorities = |pair: &AliasedPair| -> (u8, u8) {
            let pa = block_priority_for_attribution
                .get(pair.block_a.as_str())
                .copied()
                .unwrap_or(DEFAULT_GAUGE_PRIORITY);
            let pb = block_priority_for_attribution
                .get(pair.block_b.as_str())
                .copied()
                .unwrap_or(DEFAULT_GAUGE_PRIORITY);
            (pa, pb)
        };
        let matches_demoted = |pair: &AliasedPair| -> bool {
            (pair.block_a == raw_block_name && pair.direction_a == local_col)
                || (pair.block_b == raw_block_name && pair.direction_b == local_col)
        };

        // Does any matching distinct-priority alias pair place the raw
        // demoted column on its LOWER-priority side? Then RRQR's choice
        // already agrees with canonical gauge ownership.
        let raw_already_lower = aliased_pairs
            .iter()
            .filter(|pair| matches_demoted(pair))
            .any(|pair| {
                let (pa, pb) = pair_priorities(pair);
                if pa == pb {
                    return false;
                }
                let other_priority = if pair.block_a == raw_block_name {
                    pb
                } else {
                    pa
                };
                raw_priority < other_priority
            });

        let best_pair = if raw_already_lower {
            None
        } else {
            aliased_pairs
                .iter()
                .filter(|pair| matches_demoted(pair))
                .filter(|pair| {
                    let (pa, pb) = pair_priorities(pair);
                    pa != pb
                })
                .max_by(|a, b| {
                    a.overlap
                        .partial_cmp(&b.overlap)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        };
        let (block_name, drop_local_col, reason) = if let Some(pair) = best_pair {
            let (pa, pb) = pair_priorities(pair);
            let (lower_block, lower_col, lower_prio, higher_block, higher_col, higher_prio) =
                if pa < pb {
                    (
                        pair.block_a.clone(),
                        pair.direction_a,
                        pa,
                        pair.block_b.clone(),
                        pair.direction_b,
                        pb,
                    )
                } else {
                    (
                        pair.block_b.clone(),
                        pair.direction_b,
                        pb,
                        pair.block_a.clone(),
                        pair.direction_a,
                        pa,
                    )
                };
            (
                lower_block.clone(),
                lower_col,
                format!(
                    "joint-design column {joint_col} (raw RRQR block '{raw_block_name}' local column {local_col}) demoted past joint RRQR rank tolerance {tol:.3e}; canonical gauge re-attributed the shared direction to lower-priority block '{lower_block}' local column {lower_col} (priority {lower_prio} < '{higher_block}' local column {higher_col}, priority {higher_prio}; overlap={overlap:.4})",
                    tol = joint_rank_tol,
                    overlap = pair.overlap,
                ),
            )
        } else if raw_already_lower {
            (
                raw_block_name.clone(),
                local_col,
                format!(
                    "joint-design column {joint_col} (block '{raw_block_name}' local column {local_col}, priority {raw_priority}) demoted past joint RRQR rank tolerance {tol:.3e}; canonical gauge keeps the attribution on this block — RRQR already named the lower-priority side of its alias",
                    tol = joint_rank_tol,
                ),
            )
        } else {
            (
                raw_block_name.clone(),
                local_col,
                format!(
                    "joint-design column {joint_col} (block '{raw_block_name}' local column {local_col}) demoted past joint RRQR rank tolerance {tol:.3e}; earlier blocks' column span absorbs this direction",
                    tol = joint_rank_tol,
                ),
            )
        };
        dropped_columns.push(DroppedColumn {
            block: block_name,
            column: drop_local_col,
            reason,
        });
    }

    // Reflect canonical dropped-column attribution into `BlockIdentity::
    // effective_dim`: each block's effective dimension is its original dim
    // minus the count of its attributed dropped columns (post re-attribution
    // to the lower-priority participant, not the raw RRQR-selected column).
    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let block_name = specs[block_idx].name.as_str();
        let dropped_here = dropped_columns
            .iter()
            .filter(|drop| drop.block == block_name)
            .count();
        block.effective_dim = block.original_dim.saturating_sub(dropped_here);
    }

    let joint_rank_deficient = joint_rank < p_total;
    // Hard-halt and gauge-resolvability classification.
    //
    // # Canonical-gauge contract
    //
    // When the caller supplies a non-trivial `gauge_priority` configuration
    // (at least two distinct priority values), the canonical-gauge pipeline
    // (`canonicalize_for_identifiability`) is designed to handle cross-block
    // rank deficiency by presenting higher-priority columns first to the
    // RRQR pivot and attributing the alias drops to the lower-priority
    // block. The audit MUST NOT FATAL on this case — doing so defeats the
    // entire pipeline.
    //
    // # Hard-halt cases (always fatal regardless of gauge)
    //
    //   (a) Cross-block alias pair with overlap >= pair_halt_threshold(s2_a, s2_b, n)
    //       where the two blocks carry the SAME `gauge_priority` — no
    //       ordering exists to decide which block loses the direction. This
    //       is the original "two blocks contributing the same direction, inner
    //       KKT has no unique minimiser" failure mode the gate was built for.
    //       When priorities differ, the canonical-gauge RRQR ordering resolves
    //       the alias deterministically; do NOT halt those.
    //
    //   (b) Joint rank deficiency that gauge CANNOT resolve:
    //       - All specs carry equal priority (no gauge ordering) AND
    //         joint rank < p_total, OR
    //       - Attribution is incomplete (dropped_columns.len() <
    //         p_total - joint_rank) — unattributed null directions that
    //         RRQR could not trace to any single block, OR
    //       - A dropped column belongs to a block that is the HIGHER-priority
    //         participant in its alias pair — RRQR attributed the drop to
    //         the wrong block (guard against internal bugs).
    //
    // # Gauge-resolvable (non-fatal even with dropped columns / rank deficiency)
    //
    //   - All specs have at least two distinct priority values, AND
    //   - RRQR attributed every deficient column (full attribution), AND
    //   - Each attributed drop's block is the LOWER-priority participant
    //     in at least one alias pair above the reporting threshold (confirming
    //     the priority-ordered RRQR routed the drop to the correct block).
    //
    // RRQR's tolerance can disagree with the column-norm pairwise scan
    // on edge cases (e.g. a high-overlap pair that RRQR keeps because
    // its residual is just above the rank threshold), so the two
    // conditions are kept independent: either one is sufficient to halt.
    //
    // For multi-channel families (e.g. survival marginal-slope, K=4),
    // identical raw-X columns may live in orthogonal K-channels of the
    // row Jacobian and BE separately identifiable through the
    // likelihood. Those families must NOT route through this flat
    // audit; they invoke
    // `audit_identifiability_channel_aware` with their per-block row
    // Jacobian operators + structural row Hessian.

    // Name-to-priority lookup (one entry per spec; the audit may have
    // duplicate block names across runs but not within a single call).
    let block_priority: std::collections::HashMap<&str, u8> = specs
        .iter()
        .map(|s| (s.name.as_str(), s.gauge_priority))
        .collect();

    // True when all specs share the same gauge_priority value — in that
    // case no priority ordering can resolve cross-block aliases and the
    // original halt-on-deficiency behaviour is correct.
    let all_priorities_equal = specs
        .iter()
        .all(|s| s.gauge_priority == specs[0].gauge_priority);

    // A hard-alias pair is gauge-unresolvable (and thus causes a fatal halt)
    // when both participating blocks have the SAME priority.  Cross-block
    // pairs with strictly different priorities are resolved by the priority-
    // ordered RRQR: the lower-priority block's column is demoted into the
    // trailing rank-deficient space, leaving the higher-priority block's
    // direction intact.  Two blocks contributing the same direction is only
    // *unfittable* when no ordering exists to pick which one to drop.
    //
    // The halt threshold is column-specific (leverage-based): for each pair
    // we reconstruct the joint column indices from block name → block index
    // → col_offsets + direction offset, then call `pair_halt_threshold`.
    let block_name_to_idx: std::collections::HashMap<&str, usize> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name.as_str(), i))
        .collect();
    // The halt band has already been resolved during the pairwise scan
    // (`halt_pairs`, gam#1397): every member is a cross-block pair whose
    // bias-corrected cosine clears the leverage-scaled halt half-width, computed
    // directly from the cosine and therefore decoupled from the report floor.
    // The ONLY remaining question for a hard halt is gauge-resolvability: a pair
    // is unfittable iff its two blocks carry the SAME gauge_priority, because
    // then no ordering exists to decide which block forfeits the shared
    // direction. Distinct priorities are resolved deterministically by the
    // priority-ordered RRQR (the lower-priority column is demoted), so they never
    // halt.
    let hard_alias_pair = halt_pairs
        .iter()
        .filter(|p| {
            let pa = block_priority
                .get(p.block_a.as_str())
                .copied()
                .unwrap_or(DEFAULT_GAUGE_PRIORITY);
            let pb = block_priority
                .get(p.block_b.as_str())
                .copied()
                .unwrap_or(DEFAULT_GAUGE_PRIORITY);
            pa == pb
        })
        .max_by(|a, b| {
            a.overlap
                .partial_cmp(&b.overlap)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    // Gauge can resolve the rank deficiency when:
    //   1. Not all priorities are equal (an ordering exists).
    //   2. RRQR attributed every deficient column (complete attribution).
    //   3. Each attributed drop's block is a lower-priority participant in
    //      at least one alias pair — confirming the priority-ordered RRQR
    //      correctly routed the drop to the lower-priority block.
    let rank_deficiency_count = p_total.saturating_sub(joint_rank);
    let attribution_complete =
        rank_deficiency_count > 0 && dropped_columns.len() == rank_deficiency_count;
    let gauge_resolves_rank_deficiency = !all_priorities_equal
        && attribution_complete
        && dropped_columns.iter().all(|drop| {
            let drop_priority = block_priority
                .get(drop.block.as_str())
                .copied()
                .unwrap_or(100);
            // The drop is correctly attributed when there exists at least one
            // alias pair where this block is the LOWER-priority participant
            // (i.e. the other block has strictly higher priority).
            aliased_pairs.iter().any(|pair| {
                let other_block = if pair.block_a == drop.block {
                    pair.block_b.as_str()
                } else if pair.block_b == drop.block {
                    pair.block_a.as_str()
                } else {
                    return false;
                };
                let other_priority = block_priority.get(other_block).copied().unwrap_or(100);
                other_priority > drop_priority
            })
        });

    let fatal =
        (joint_rank_deficient && !gauge_resolves_rank_deficiency) || hard_alias_pair.is_some();

    let fatal_detail = if fatal {
        let mut parts: Vec<String> = Vec::new();
        if joint_rank_deficient {
            // Name the worst-attributed dropped column (if any) so the
            // caller sees which (block, local_col) to reparameterise
            // first. When attribution is empty, surface that fact —
            // it indicates a >2-way structural alias that the pairwise
            // scan didn't catch and is the hardest case to fix.
            let attribution = if let Some(first_drop) = dropped_columns.first() {
                // Name the columns the dropped column is collinear WITH (its
                // alias partners), so the verdict reveals the actual redundancy
                // — e.g. "collinear with 'marginal_surface' local column 0"
                // (the intercept) — instead of only the demoted column's
                // address. Also flag when the redundancy is INTRA-block, since
                // that is precisely the case cross-block gauge_priority cannot
                // resolve.
                let partners: Vec<String> = aliased_pairs
                    .iter()
                    .filter_map(|p| {
                        if p.block_a == first_drop.block && p.direction_a == first_drop.column {
                            Some(format!(
                                "'{}' local column {} (overlap={:.4})",
                                p.block_b, p.direction_b, p.overlap
                            ))
                        } else if p.block_b == first_drop.block
                            && p.direction_b == first_drop.column
                        {
                            Some(format!(
                                "'{}' local column {} (overlap={:.4})",
                                p.block_a, p.direction_a, p.overlap
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();
                let alias_note = if partners.is_empty() {
                    String::new()
                } else {
                    format!("; collinear with {}", partners.join(", "))
                };
                let within_block_note = blocks
                    .iter()
                    .find(|b| b.block_name == first_drop.block)
                    .filter(|b| b.design_range_rank < b.original_dim)
                    .map(|b| {
                        format!(
                            "; block '{}' is INTRA-BLOCK rank-deficient \
                             (range_rank {}/{}) — NOT resolvable by cross-block \
                             gauge_priority; the redundant column must be removed \
                             or centered within the block",
                            first_drop.block, b.design_range_rank, b.original_dim
                        )
                    })
                    .unwrap_or_default();
                format!(
                    "first attributed drop: block '{}' local column {}{}{} \
                     (reparam: replace this column with a sum-to-zero or \
                     orthogonal-complement projection against earlier blocks, \
                     or remove the redundant term entirely)",
                    first_drop.block, first_drop.column, alias_note, within_block_note,
                )
            } else {
                "no per-column attribution (>2-way structural alias); \
                 audit the joint design by eye and absorb the shared null \
                 subspace into a single parametric block"
                    .to_string()
            };
            parts.push(format!(
                "joint rank {} < joint columns {} ({} dropped column(s); {})",
                joint_rank,
                p_total,
                dropped_columns.len(),
                attribution,
            ));
        }
        if let Some(pair) = hard_alias_pair.as_ref() {
            let ja = block_name_to_idx
                .get(pair.block_a.as_str())
                .map(|&bi| col_offsets[bi] + pair.direction_a)
                .unwrap_or(0);
            let jb = block_name_to_idx
                .get(pair.block_b.as_str())
                .map(|&bi| col_offsets[bi] + pair.direction_b)
                .unwrap_or(0);
            let halt_half_width = pair_halt_threshold(
                col_s2.get(ja).copied().unwrap_or(1.0),
                col_s2.get(jb).copied().unwrap_or(1.0),
                n,
            );
            let shift_note = if pair.bias_shift.abs() > 1e-8 {
                format!(" bias_shift={:.4}", pair.bias_shift)
            } else {
                String::new()
            };
            parts.push(format!(
                "alias pair: '{}'[{}] ~ '{}'[{}] overlap={:.4} >= leverage-based halt \
                 half-width {:.4}{} (n_eff_a≈{:.0}, n_eff_b≈{:.0}; \
                 reparam: orthogonalise one block's column {} against the other \
                 via sum-to-zero, or absorb the shared direction into a single \
                 parametric block)",
                pair.block_a,
                pair.direction_a,
                pair.block_b,
                pair.direction_b,
                pair.overlap,
                halt_half_width,
                shift_note,
                1.0 / col_s2.get(ja).copied().unwrap_or(1.0).max(f64::EPSILON),
                1.0 / col_s2.get(jb).copied().unwrap_or(1.0).max(f64::EPSILON),
                pair.direction_b,
            ));
        }
        format!(" — FATAL: {}", parts.join("; "))
    } else if gauge_resolves_rank_deficiency {
        // Non-fatal: the canonical-gauge pipeline will attribute the
        // alias drops to the lower-priority blocks and proceed with
        // reduced specs. This is the expected outcome for families like
        // survival marginal-slope where time/marginal/logslope carry
        // overlapping directions that the priority ordering resolves.
        format!(
            " — gauge-attributed drops: {} column(s) attributed to lower-priority blocks \
             via gauge_priority ordering; canonical-gauge pipeline will proceed with \
             reduced specs",
            dropped_columns.len(),
        )
    } else if !aliased_pairs.is_empty() {
        " — partial alias(es) below leverage-based halt threshold; penalty + line search will resolve"
            .to_string()
    } else {
        " — clean".to_string()
    };

    let summary = format!(
        "identifiability audit: {} block(s), {} joint columns, joint rank {}, \
         {} alias pair(s) above leverage-based report threshold, {} dropped column(s){}",
        specs.len(),
        p_total,
        joint_rank,
        aliased_pairs.len(),
        dropped_columns.len(),
        fatal_detail,
    );

    Ok(IdentifiabilityAudit {
        blocks,
        aliased_pairs,
        dropped_columns,
        fatal,
        summary,
    })
}

/// Channel-aware identifiability audit for multi-channel families.
///
/// The flat [`audit_identifiability`] runs RRQR on
/// `X_joint ∈ ℝ^{n × p_total}`, treating every block's raw design as
/// living in the same row space. For multi-channel families
/// (e.g. survival marginal-slope, `K = 4`), two columns from different
/// blocks with identical raw row values may contribute to *orthogonal*
/// `K`-channels of the row Jacobian — they look aliased in `X_joint`
/// but are separately identifiable through the likelihood. The flat
/// audit reports those as fatal hard-alias pairs and refuses the fit;
/// the channel-aware audit operates on the
/// `(n·K) × p_total` channel-weighted joint design
/// `W_joint = stack_i sqrt(K^S_i) · J_i`
/// and gets the correct answer.
///
/// `operators[i]` is the row Jacobian for block `i` (so
/// `operators[i].apply_row(row, δβ, out)` writes `J_{i,row}·δβ ∈ ℝ^K`).
/// `specs[i].design.ncols()` MUST equal `operators[i].ncols()`,
/// `specs[i].design.nrows()` MUST equal `operators[i].nrows()`, and
/// every operator must share the same `K = row_hess.k()`.
///
/// `row_hess` is the structural row metric `K^S` (typically an
/// [`crate::identifiability::families::compiler::IdentityRowHessian`] —
/// see [`compile_with_dual_metric`] for why the structural metric is
/// the rank-decision metric, not the pilot curvature).
///
/// The output [`IdentifiabilityAudit`] preserves the same contract as
/// the flat path: `dropped_columns` attributed to (block, local_col),
/// `aliased_pairs` reported above the per-pair leverage-based report
/// threshold in the channel-weighted view, `fatal` true on rank
/// deficiency or hard-alias pair above the per-pair leverage-based
/// halt threshold (`pair_halt_threshold`).
struct ChannelAwareStreamedGeometry {
    gram_h: Array2<f64>,
    gram_struct: Array2<f64>,
    col_norms: Vec<f64>,
    col_s2: Vec<f64>,
    raw_ranges: Vec<std::ops::Range<usize>>,
}

fn channel_aware_streamed_geometry(
    operators: &[std::sync::Arc<
        dyn crate::identifiability::families::compiler::RowJacobianOperator,
    >],
    row_hess: &dyn crate::identifiability::families::compiler::RowHessian,
    row_structural: &dyn crate::identifiability::families::compiler::RowHessian,
    col_offsets: &[usize],
) -> Result<ChannelAwareStreamedGeometry, EstimationError> {
    if operators.is_empty() {
        return Ok(ChannelAwareStreamedGeometry {
            gram_h: Array2::<f64>::zeros((0, 0)),
            gram_struct: Array2::<f64>::zeros((0, 0)),
            col_norms: Vec::new(),
            col_s2: Vec::new(),
            raw_ranges: Vec::new(),
        });
    }
    let n = row_hess.nrows();
    let k = row_hess.k();
    let p_total = *col_offsets.last().unwrap_or(&0);
    let raw_ranges: Vec<std::ops::Range<usize>> = (0..operators.len())
        .map(|idx| col_offsets[idx]..col_offsets[idx + 1])
        .collect();
    let mut gram_h = Array2::<f64>::zeros((p_total, p_total));
    let mut gram_struct = Array2::<f64>::zeros((p_total, p_total));
    let mut fourth = vec![0.0_f64; p_total];

    for start in (0..n).step_by(CHANNEL_AWARE_ROW_CHUNK) {
        let end = (start + CHANNEL_AWARE_ROW_CHUNK).min(n);
        let chunk = end - start;
        let mut chunks: Vec<Array2<f64>> = Vec::with_capacity(operators.len());
        for (block_idx, op) in operators.iter().enumerate() {
            let p_b = op.ncols();
            let mut rows = Array2::<f64>::zeros((chunk * k, p_b));
            op.channel_flattened_rows(start..end, &mut rows);
            let base = col_offsets[block_idx];
            for col in 0..p_b {
                let mut sum4 = 0.0_f64;
                for value in rows.column(col).iter() {
                    let sq = value * value;
                    sum4 += sq * sq;
                }
                fourth[base + col] += sum4;
            }
            chunks.push(rows);
        }
        accumulate_channel_metric_gram(&chunks, row_hess, start, end, col_offsets, &mut gram_h)?;
        accumulate_channel_metric_gram(
            &chunks,
            row_structural,
            start,
            end,
            col_offsets,
            &mut gram_struct,
        )?;
    }
    for i in 0..p_total {
        for j in 0..i {
            gram_h[[i, j]] = gram_h[[j, i]];
            gram_struct[[i, j]] = gram_struct[[j, i]];
        }
    }
    let mut col_norms = Vec::with_capacity(p_total);
    let mut col_s2 = Vec::with_capacity(p_total);
    for col in 0..p_total {
        let sq_norm = gram_struct[[col, col]].max(0.0);
        col_norms.push(sq_norm.sqrt());
        if sq_norm <= 0.0 {
            col_s2.push(1.0);
        } else {
            col_s2.push(fourth[col] / (sq_norm * sq_norm));
        }
    }
    Ok(ChannelAwareStreamedGeometry {
        gram_h,
        gram_struct,
        col_norms,
        col_s2,
        raw_ranges,
    })
}

fn accumulate_channel_metric_gram(
    chunks: &[Array2<f64>],
    metric: &dyn crate::identifiability::families::compiler::RowHessian,
    start: usize,
    end: usize,
    col_offsets: &[usize],
    out: &mut Array2<f64>,
) -> Result<(), EstimationError> {
    let k = metric.k();
    let chunk = end - start;
    let mut weighted: Vec<Array2<f64>> = chunks
        .iter()
        .map(|chunk_rows| Array2::<f64>::zeros(chunk_rows.dim()))
        .collect();
    let mut h_row = vec![0.0_f64; k * k];
    for local_i in 0..chunk {
        metric.fill_row(start + local_i, &mut h_row);
        for (block_idx, chunk_rows) in chunks.iter().enumerate() {
            let p_b = chunk_rows.ncols();
            for out_ch in 0..k {
                for col in 0..p_b {
                    let mut acc = 0.0_f64;
                    for in_ch in 0..k {
                        acc += h_row[out_ch * k + in_ch] * chunk_rows[[local_i * k + in_ch, col]];
                    }
                    weighted[block_idx][[local_i * k + out_ch, col]] = acc;
                }
            }
        }
    }
    for a in 0..chunks.len() {
        let range_a = col_offsets[a]..col_offsets[a + 1];
        for b in a..chunks.len() {
            let range_b = col_offsets[b]..col_offsets[b + 1];
            let block = fast_atb(&chunks[a], &weighted[b]);
            for local_a in 0..block.nrows() {
                for local_b in 0..block.ncols() {
                    out[[range_a.start + local_a, range_b.start + local_b]] +=
                        block[[local_a, local_b]];
                }
            }
        }
    }
    Ok(())
}

pub fn audit_identifiability_channel_aware(
    specs: &[ParameterBlockSpec],
    operators: &[std::sync::Arc<
        dyn crate::identifiability::families::compiler::RowJacobianOperator,
    >],
    row_hess: &dyn crate::identifiability::families::compiler::RowHessian,
) -> Result<IdentifiabilityAudit, EstimationError> {
    use crate::identifiability::families::compiler::{IdentityRowHessian, compile_from_raw_grams};

    if specs.is_empty() {
        return Ok(IdentifiabilityAudit {
            blocks: Vec::new(),
            aliased_pairs: Vec::new(),
            dropped_columns: Vec::new(),
            fatal: false,
            summary: "identifiability audit (channel-aware): no blocks supplied".to_string(),
        });
    }
    if specs.len() != operators.len() {
        return Err(EstimationError::LayoutError(format!(
            "audit_identifiability_channel_aware: specs ({}) and operators ({}) length mismatch",
            specs.len(),
            operators.len()
        )));
    }
    let k = row_hess.k();
    let n = row_hess.nrows();
    for (idx, op) in operators.iter().enumerate() {
        if op.k() != k {
            return Err(EstimationError::LayoutError(format!(
                "audit_identifiability_channel_aware: operator {idx} has K={} but row_hess K={k}",
                op.k(),
            )));
        }
        if op.nrows() != n {
            return Err(EstimationError::LayoutError(format!(
                "audit_identifiability_channel_aware: operator {idx} has nrows={} but row_hess nrows={n}",
                op.nrows(),
            )));
        }
        if op.ncols() != specs[idx].design.ncols() {
            return Err(EstimationError::LayoutError(format!(
                "audit_identifiability_channel_aware: operator {idx} ({}) has ncols={} but spec '{}' design ncols={}",
                idx,
                op.ncols(),
                specs[idx].name,
                specs[idx].design.ncols(),
            )));
        }
    }

    // Per-block "in-isolation" rank decision uses the structural row
    // metric directly, so the audit's reported `design_range_rank`
    // matches the within-block RRQR of `sqrt(K^S) · J_i` in
    // `(n·K, p_i)` space. Reuse the same identity-K^S the compiler
    // defaults to: decoupling the rank-decision metric from a possibly
    // rank-deficient pilot curvature is what makes the channel-aware
    // audit a structural identifiability check rather than a
    // curvature-aware one.
    let id_struct = IdentityRowHessian::new(n, k);
    let mut col_offsets: Vec<usize> = Vec::with_capacity(specs.len() + 1);
    col_offsets.push(0);
    for spec in specs {
        let next = col_offsets[col_offsets.len() - 1] + spec.design.ncols();
        col_offsets.push(next);
    }

    // Stream the row-Jacobian into p×p Grams once. `compile_from_raw_grams`
    // performs the same structural/curvature residualisation in coefficient
    // space; its eigenspace rank threshold maps the design singular-value
    // tolerance through λ=σ², so the fail-closed reduction stays tied to the
    // exact Gram geometry without retaining `(n*K)×p` designs.
    let geometry = channel_aware_streamed_geometry(operators, row_hess, &id_struct, &col_offsets)?;
    let ordering: Vec<crate::identifiability::families::compiler::BlockOrder> =
        std::iter::repeat(crate::identifiability::families::compiler::BlockOrder::Marginal)
            .take(operators.len())
            .collect();
    let compiled_map = compile_from_raw_grams(
        &geometry.gram_h,
        &geometry.gram_struct,
        &geometry.raw_ranges,
        &ordering,
    )
    .map_err(|e| {
        EstimationError::LayoutError(format!(
            "audit_identifiability_channel_aware compile failed: {e:?}"
        ))
    })?;

    // Build per-block identity entries from the compiled output.
    let mut blocks: Vec<BlockIdentity> = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        let p_block = spec.design.ncols();
        let kept = compiled_map.compiled_block_ranges[idx].len();
        blocks.push(BlockIdentity {
            block_name: spec.name.clone(),
            original_dim: p_block,
            effective_dim: kept,
            // The channel-aware path does not produce a separate
            // per-block penalty-aware rank; rely on `effective_dim`
            // < `original_dim` as the structural-rank signal.
            design_range_rank: kept,
        });
    }
    let p_total = *col_offsets.last().expect("col_offsets non-empty");

    // The Gram compiler emits kept widths. Attribute any lost width to trailing
    // local columns, matching the existing audit contract that downstream code
    // consumes for diagnostics rather than for an automatic callback reparam.
    let mut dropped_columns: Vec<DroppedColumn> = Vec::new();
    for (block_idx, spec) in specs.iter().enumerate() {
        let p_block = spec.design.ncols();
        let kept = compiled_map.compiled_block_ranges[block_idx].len();
        for local_col in kept..p_block {
            let block_name = spec.name.clone();
            dropped_columns.push(DroppedColumn {
                block: block_name.clone(),
                column: local_col,
                reason: format!(
                    "channel-aware audit (K={k}) demoted block '{block_name}' \
                     local column {local_col}: column is in the row-Jacobian span \
                     of earlier blocks under the structural row metric",
                ),
            });
        }
    }

    // Pairwise overlap scan in the channel-weighted view. The compiler
    // already eigendecomposed the structural Gram block-by-block; we
    // need joint column-column overlaps to surface near-alias pairs
    // above the reporting threshold. Compute on the (n·K, p_total)
    // weighted joint W where W_b = sqrt(K^S) · J_b. With K^S = I,
    // sqrt(K^S) = I and W_b is just J_b flattened to (n·K, p_b).
    let ScannedAliasPairs {
        reported: aliased_pairs,
        halt: halt_pairs,
    } = channel_aware_aliased_pairs(
        &geometry.gram_struct,
        &geometry.col_norms,
        &geometry.col_s2,
        &col_offsets,
        specs,
        n * k,
    )?;

    let joint_rank = compiled_map.raw_from_compiled.ncols();
    let joint_rank_deficient = joint_rank < p_total;

    // Penalty-aware joint rank `rank([J_joint; S_blockdiag])` = co-dimension of
    // `ker(J) ∩ ker(S)`. The structural `joint_rank` above is penalty-BLIND, so
    // a design-null-but-penalty-covered direction (e.g. a smooth's penalized
    // null space replicated across the multinomial softmax channels, or a
    // marginal-slope curvature direction the marginal penalty covers) is counted
    // as a structural rank shortfall. It is NOT a genuine non-identifiability:
    // the penalized normal equations `JᵀWJ + S` are non-singular there, so the
    // MAP is unique and the REML seed is legitimately fittable. The flat audit
    // already augments with the penalty rows for exactly this reason
    // (`x_joint_rank_input`); the multi-channel path must do the same or it
    // refuses identifiable seeds (#715 real-data arm: "canonical-gauge null
    // direction rejects all REML seeds"). When the penalty closes the structural
    // gap (`penalty_aware_joint_rank == p_total`) the only residual deficiency is
    // penalty-covered, hence identified — never a fatal refusal. The downstream
    // `check_map_uniqueness` (run in `canonicalize_for_identifiability_inner`)
    // remains the precise gate for the genuinely fatal `ker(JᵀWJ) ∩ ker(S) ≠ {0}`
    // case; this only stops the structural-rank gate from shadowing it.
    let penalty_aware_joint_rank =
        channel_aware_penalty_aware_joint_rank(&geometry.gram_struct, &col_offsets, specs, n * k)?;
    let penalty_covers_rank_deficiency =
        joint_rank_deficient && penalty_aware_joint_rank >= p_total;

    // Same gauge-priority gating as the flat audit path (see the
    // corresponding comment in `audit_identifiability`).
    let block_priority_ca: std::collections::HashMap<&str, u8> = specs
        .iter()
        .map(|s| (s.name.as_str(), s.gauge_priority))
        .collect();
    let all_priorities_equal_ca = specs
        .iter()
        .all(|s| s.gauge_priority == specs[0].gauge_priority);

    // The halt threshold is leverage-based (see flat audit path for details).
    // The streamed geometry accumulated S2 from row chunks, so this path keeps
    // the same finite-sample threshold without retaining column vectors.
    let block_name_to_idx_ca: std::collections::HashMap<&str, usize> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name.as_str(), i))
        .collect();
    let ca_col_s2 = &geometry.col_s2;
    // `halt_pairs` already cleared the leverage-scaled halt band in the scan,
    // decoupled from the report floor (gam#1397). The only remaining gate is
    // gauge-resolvability: same priority → no ordering to pick a loser → halt.
    let hard_alias_pair = halt_pairs
        .iter()
        .filter(|p| {
            let pa = block_priority_ca
                .get(p.block_a.as_str())
                .copied()
                .unwrap_or(DEFAULT_GAUGE_PRIORITY);
            let pb = block_priority_ca
                .get(p.block_b.as_str())
                .copied()
                .unwrap_or(DEFAULT_GAUGE_PRIORITY);
            pa == pb
        })
        .max_by(|a, b| {
            a.overlap
                .partial_cmp(&b.overlap)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    let rank_deficiency_count_ca = p_total.saturating_sub(joint_rank);
    let attribution_complete_ca =
        rank_deficiency_count_ca > 0 && dropped_columns.len() == rank_deficiency_count_ca;
    let gauge_resolves_rank_deficiency_ca = !all_priorities_equal_ca
        && attribution_complete_ca
        && dropped_columns.iter().all(|drop| {
            let drop_priority = block_priority_ca
                .get(drop.block.as_str())
                .copied()
                .unwrap_or(100);
            aliased_pairs.iter().any(|pair| {
                let other_block = if pair.block_a == drop.block {
                    pair.block_b.as_str()
                } else if pair.block_b == drop.block {
                    pair.block_a.as_str()
                } else {
                    return false;
                };
                let other_priority = block_priority_ca.get(other_block).copied().unwrap_or(100);
                other_priority > drop_priority
            })
        });

    // Channel-aware rank deficiency that the pairwise scan does NOT detect
    // (`aliased_pairs.is_empty()`) is non-fatal: the leverage-based report
    // threshold already classifies every cross-block alias worth flagging,
    // and the leverage-based halt threshold (`hard_alias_pair`) catches the
    // ones strong enough to break the inner solve regardless. Any remaining
    // joint-rank shortfall comes from either
    //   (a) per-block compile-time structural reductions — the compiler's
    //       `compile_with_dual_metric` shrinks each block's kept basis when
    //       its own column space has redundancies, and that reduction is
    //       absorbed into `CompiledBlock::t_lw` (with the penalty pull-back
    //       enlarging the structural nullspace correspondingly), so the
    //       inner penalised Newton solves a well-posed system; or
    //   (b) below-threshold weak aliases that the smoothing-parameter ridge
    //       and the penalised line search already regularise away.
    // Neither case warrants a hard refusal of the fit. The existing
    // gauge-resolution path (cross-block alias above the report threshold,
    // attributed drops, gauge_priority breaks the tie) is unchanged; only
    // the pairwise-clean case is relaxed.  Perfect / near-perfect
    // cross-block alias is still caught by `hard_alias_pair`.
    let intra_block_only_ca = aliased_pairs.is_empty();

    // Penalty-aware fatal verdict (#715): a structural rank deficiency — whether
    // surfaced as an unresolved cross-block alias or as a hard near-perfect alias
    // pair — is only a genuine non-identifiability when the deficient direction
    // is ALSO penalty-null (`ker(J) ∩ ker(S) ≠ {0}`). When the block penalties
    // close the gap (`penalty_covers_rank_deficiency`), the penalized normal
    // equations are non-singular along every deficient direction, the MAP is
    // unique, and the REML seed is legitimately fittable; refusing it
    // over-rejects an identifiable model. This makes the multi-channel gate match
    // the flat audit's `[J; S]`-augmented rank verdict.
    let fatal = !penalty_covers_rank_deficiency
        && ((joint_rank_deficient && !gauge_resolves_rank_deficiency_ca && !intra_block_only_ca)
            || hard_alias_pair.is_some());

    let fatal_detail = if fatal {
        let mut parts: Vec<String> = Vec::new();
        if joint_rank_deficient && !gauge_resolves_rank_deficiency_ca {
            let attribution = if let Some(first_drop) = dropped_columns.first() {
                // Name the columns the dropped column is collinear WITH (its
                // alias partners), so the verdict reveals the actual redundancy
                // — e.g. "collinear with 'marginal_surface' local column 0"
                // (the intercept) — instead of only the demoted column's
                // address. Also flag when the redundancy is INTRA-block, since
                // that is precisely the case cross-block gauge_priority cannot
                // resolve.
                let partners: Vec<String> = aliased_pairs
                    .iter()
                    .filter_map(|p| {
                        if p.block_a == first_drop.block && p.direction_a == first_drop.column {
                            Some(format!(
                                "'{}' local column {} (overlap={:.4})",
                                p.block_b, p.direction_b, p.overlap
                            ))
                        } else if p.block_b == first_drop.block
                            && p.direction_b == first_drop.column
                        {
                            Some(format!(
                                "'{}' local column {} (overlap={:.4})",
                                p.block_a, p.direction_a, p.overlap
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();
                let alias_note = if partners.is_empty() {
                    String::new()
                } else {
                    format!("; collinear with {}", partners.join(", "))
                };
                let within_block_note = blocks
                    .iter()
                    .find(|b| b.block_name == first_drop.block)
                    .filter(|b| b.design_range_rank < b.original_dim)
                    .map(|b| {
                        format!(
                            "; block '{}' is INTRA-BLOCK rank-deficient \
                             (range_rank {}/{}) — NOT resolvable by cross-block \
                             gauge_priority; the redundant column must be removed \
                             or centered within the block",
                            first_drop.block, b.design_range_rank, b.original_dim
                        )
                    })
                    .unwrap_or_default();
                format!(
                    "first attributed drop: block '{}' local column {}{}{} \
                     (reparam: replace this column with a sum-to-zero or \
                     orthogonal-complement projection against earlier blocks, \
                     or remove the redundant term entirely)",
                    first_drop.block, first_drop.column, alias_note, within_block_note,
                )
            } else {
                "no per-column attribution (>2-way structural alias in the \
                 channel-aware row-Jacobian space)"
                    .to_string()
            };
            parts.push(format!(
                "channel-aware joint rank {} < joint columns {} \
                 ({} dropped column(s); {})",
                joint_rank,
                p_total,
                dropped_columns.len(),
                attribution,
            ));
        }
        if let Some(pair) = hard_alias_pair.as_ref() {
            let ja = block_name_to_idx_ca
                .get(pair.block_a.as_str())
                .map(|&bi| col_offsets[bi] + pair.direction_a)
                .unwrap_or(0);
            let jb = block_name_to_idx_ca
                .get(pair.block_b.as_str())
                .map(|&bi| col_offsets[bi] + pair.direction_b)
                .unwrap_or(0);
            let halt_half_width_ca = pair_halt_threshold(
                ca_col_s2.get(ja).copied().unwrap_or(1.0),
                ca_col_s2.get(jb).copied().unwrap_or(1.0),
                n * k,
            );
            parts.push(format!(
                "alias pair: '{}'[{}] ~ '{}'[{}] overlap={:.4} >= leverage-based halt \
                 half-width {:.4} in channel-aware row-Jacobian view \
                 (reparam: orthogonalise one block's column {} against the other \
                 or absorb the shared direction)",
                pair.block_a,
                pair.direction_a,
                pair.block_b,
                pair.direction_b,
                pair.overlap,
                halt_half_width_ca,
                pair.direction_b,
            ));
        }
        format!(" — FATAL: {}", parts.join("; "))
    } else if penalty_covers_rank_deficiency {
        // The structural joint rank is short of full, but the block penalties
        // close the gap: `rank([J; S]) == p_total`, so the only deficient
        // directions are penalty-covered (in `ker(J)` but not `ker(S)`). The
        // penalized normal equations are non-singular there — the MAP is unique
        // and the seed is fittable. Not a refusal (#715 real-data arm).
        format!(
            " — penalty-covered rank deficiency (channel-aware): structural joint rank {} \
             < joint columns {} but penalty-aware rank [J; S] = {} = full; deficient \
             directions are penalty-covered (ker(J)∖ker(S)) — identified, MAP unique; \
             canonical-gauge pipeline will proceed",
            joint_rank, p_total, penalty_aware_joint_rank,
        )
    } else if gauge_resolves_rank_deficiency_ca {
        format!(
            " — gauge-attributed drops (channel-aware): {} column(s) attributed to \
             lower-priority blocks via gauge_priority ordering; canonical-gauge pipeline \
             will proceed with reduced specs",
            dropped_columns.len(),
        )
    } else if joint_rank_deficient && intra_block_only_ca {
        // Pairwise scan was clean: the rank deficiency lives inside a single
        // block's own column space (or in a sub-threshold cross-block alias).
        // Canonicalisation drops the attributed columns and the fit proceeds.
        format!(
            " — intra-block drops (channel-aware): {} column(s) attributed to \
             single-block redundancy (no cross-block alias above leverage-based \
             report threshold); canonical-gauge pipeline will proceed with \
             reduced specs",
            dropped_columns.len(),
        )
    } else if !aliased_pairs.is_empty() {
        " — partial alias(es) below leverage-based halt threshold (channel-aware); \
         penalty + line search will resolve"
            .to_string()
    } else {
        " — clean (channel-aware)".to_string()
    };

    let summary = format!(
        "identifiability audit (channel-aware, K={k}): {} block(s), {} joint columns, \
         joint rank {}, {} alias pair(s) above leverage-based report threshold, \
         {} dropped column(s){}",
        specs.len(),
        p_total,
        joint_rank,
        aliased_pairs.len(),
        dropped_columns.len(),
        fatal_detail,
    );

    Ok(IdentifiabilityAudit {
        blocks,
        aliased_pairs,
        dropped_columns,
        fatal,
        summary,
    })
}

/// Penalty-aware joint column rank of the channel-weighted joint design,
/// computed exactly the way the flat audit computes it (see
/// [`block_structural_penalty_dense`] / [`block_penalty_aware_rank`] and the
/// `x_joint_rank_input` augmentation in `audit_identifiability_impl`): the
/// numerical rank of `[J_joint; S_blockdiag]`, whose null space is precisely
/// `ker(J_joint) ∩ ker(S)`.
///
/// The structural rank from `compile_with_dual_metric` answers
/// `rank(J_joint)` alone — penalty-BLIND. A direction that is design-null
/// (collinear in the row Jacobian) but COVERED by a block's smoothness
/// penalty is still fully estimated by the penalized normal equations
/// `JᵀWJ + S`; counting it as a rank deficiency over-rejects an identifiable
/// model. Multi-channel families (multinomial softmax, survival marginal-slope)
/// route exclusively through the channel-aware audit, so without this they
/// never benefit from the `[J; S]` augmentation the flat path already applies.
///
/// `S_blockdiag` is the unit-weight STRUCTURAL sum of each block's penalty
/// matrices (the same ρ-invariant `block_structural_penalty_dense` the flat
/// audit uses): only `∩_m ker(S_m)` matters for the rank verdict, independent
/// of the fitted λ values. Blocks with no penalty contribute no rows, so for an
/// unpenalized multi-channel model this reduces exactly to `rank(J_joint)`.
fn channel_aware_penalty_aware_joint_rank(
    gram_struct: &Array2<f64>,
    col_offsets: &[usize],
    specs: &[ParameterBlockSpec],
    n_design_rows: usize,
) -> Result<usize, EstimationError> {
    let p_total = *col_offsets.last().unwrap_or(&0);
    if p_total == 0 || n_design_rows == 0 {
        return Ok(0);
    }

    // Per-block structural penalties, parallel to `specs` (None ⇒ no penalty
    // rows for that block). Reuses the exact unit-weight sum the flat audit
    // uses so the two paths agree on the ρ-invariant penalty geometry.
    let block_penalties: Vec<Option<Array2<f64>>> =
        specs.iter().map(block_structural_penalty_dense).collect();
    let n_penalty_rows: usize = block_penalties
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            s.as_ref()
                .map_or(0, |_| col_offsets[idx + 1] - col_offsets[idx])
        })
        .sum();

    // rank([J; S]) is rank(J'J + S'S). Accumulating the PSD Gram keeps the
    // penalty-aware gate at O(p_total^2) memory; `rank_of_gram` applies the
    // singular-value tolerance to sqrt(eigenvalue), matching the flat rank
    // convention without materialising the `(n*K + penalty_rows) × p` design.
    let mut aug_gram = gram_struct.clone();
    for (idx, s_opt) in block_penalties.iter().enumerate() {
        if let Some(s) = s_opt {
            let start = col_offsets[idx];
            let s_gram = fast_atb(s, s);
            for row in 0..s_gram.nrows() {
                for col in 0..s_gram.ncols() {
                    aug_gram[[start + row, start + col]] += s_gram[[row, col]];
                }
            }
        }
    }

    rank_of_gram(&aug_gram, n_design_rows + n_penalty_rows)
}

/// Pairwise overlap scan on the channel-weighted joint design
/// `W = stack_b sqrt(I_K) · J_b` (identity structural metric).
/// Returns one [`AliasedPair`] per cross-block column-pair whose
/// normalised `|wᵀ w'|` exceeds the per-pair leverage-based report
/// threshold (`pair_report_threshold`), in `(block_a < block_b)` order.
/// Result of a single cross-block pairwise scan: the REPORT-band pairs
/// (diagnostics) and the HALT-band pairs (structural fittability), classified
/// independently so the hard-halt verdict is never gated on the diagnostic
/// report floor (gam#1397). Mirrors the flat path's `aliased_pairs` / `halt_pairs`
/// split for the channel-aware path.
struct ScannedAliasPairs {
    reported: Vec<AliasedPair>,
    halt: Vec<AliasedPair>,
}

fn channel_aware_aliased_pairs(
    gram_struct: &Array2<f64>,
    col_norms: &[f64],
    col_s2: &[f64],
    col_offsets: &[usize],
    specs: &[ParameterBlockSpec],
    n_design_rows: usize,
) -> Result<ScannedAliasPairs, EstimationError> {
    let p_total = *col_offsets.last().unwrap_or(&0);
    if p_total == 0 || n_design_rows == 0 {
        return Ok(ScannedAliasPairs {
            reported: Vec::new(),
            halt: Vec::new(),
        });
    }
    // Total cross-block pairs for Bonferroni correction.
    let total_cross_pairs: usize = {
        let mut cnt = 0usize;
        for a_idx in 0..specs.len() {
            let a_cols = col_offsets[a_idx + 1] - col_offsets[a_idx];
            for b_idx in (a_idx + 1)..specs.len() {
                let b_cols = col_offsets[b_idx + 1] - col_offsets[b_idx];
                cnt = cnt.saturating_add(a_cols.saturating_mul(b_cols));
            }
        }
        cnt.max(1)
    };
    // Pairwise scan: for every cross-block joint column pair (a < b) with both
    // norms positive, compute |wᵀw'| / (‖w‖·‖w'‖) and classify it into the
    // REPORT band (≥ leverage report threshold, floored at the near-exact
    // boundary) and, independently, the HALT band (≥ leverage halt threshold,
    // no near-exact floor). The two bands are decoupled — the halt band can sit
    // below the report floor for high-n_eff columns — so a structurally
    // unfittable same-priority pair below the diagnostic floor still halts.
    let mut reported: Vec<AliasedPair> = Vec::new();
    let mut halt: Vec<AliasedPair> = Vec::new();
    for a in 0..p_total {
        if col_norms[a] <= 0.0 {
            continue;
        }
        for b in (a + 1)..p_total {
            if col_norms[b] <= 0.0 {
                continue;
            }
            let dot = gram_struct[[a, b]];
            let overlap = (dot.abs() / (col_norms[a] * col_norms[b])).min(1.0);
            let (block_a_idx, dir_a) = locate_block_column(col_offsets, a)?;
            let (block_b_idx, dir_b) = locate_block_column(col_offsets, b)?;
            if block_a_idx == block_b_idx {
                // Within-block aliasing is a separate concern (per-block QR
                // catches it); the cross-block scan only reports inter-block
                // pairs.
                continue;
            }
            let make_pair = || AliasedPair {
                block_a: specs[block_a_idx].name.clone(),
                block_b: specs[block_b_idx].name.clone(),
                direction_a: dir_a,
                direction_b: dir_b,
                overlap,
                // Channel-aware path: no row-scaling bias correction; the
                // channel weighting already accounts for per-block structure,
                // and the row Jacobian operators are not parameterised through
                // RowScaledJacobian here.
                bias_shift: 0.0,
            };
            let report_thr =
                pair_report_threshold(col_s2[a], col_s2[b], n_design_rows, total_cross_pairs);
            if overlap >= report_thr {
                reported.push(make_pair());
            }
            let halt_thr = pair_halt_threshold(col_s2[a], col_s2[b], n_design_rows);
            if overlap >= halt_thr {
                halt.push(make_pair());
            }
        }
    }
    Ok(ScannedAliasPairs { reported, halt })
}

/// Summary of one audit-drift check: the rank verdict at the current β
/// compared to the pilot verdict.
///
/// Used by [`maybe_log_audit_drift`] to decide whether to emit the
/// `[AUDIT-DRIFT]` log line.
#[derive(Debug, Clone)]
pub struct AuditDriftSummary {
    /// Pilot effective rank (sum of `effective_dim` over all blocks).
    pub pilot_rank: usize,
    /// Current-β effective rank.
    pub current_rank: usize,
    /// `‖β_current − β_pilot‖₂ / (‖β_pilot‖₂ + ε)` — relative norm change.
    pub beta_relative_change: f64,
    /// Columns dropped in the current audit that were NOT dropped in the pilot.
    pub newly_dropped: Vec<DroppedColumn>,
    /// Columns dropped in the pilot that are no longer dropped (recovered).
    pub recovered: Vec<String>,
}

/// Run the channel-aware audit at `beta_current` using `channel_hessian_at`
/// to refresh W, and compare the result to the pilot audit.
///
/// # Drift threshold (T34)
///
/// The re-audit fires when:
///
/// ```text
/// ‖β_current − β_pilot‖₂ / (‖β_pilot‖₂ + ε) > 0.5   (large β movement)
/// ```
///
/// OR every `every_n_iters` outer iterations (amortised cost).
/// Document these thresholds inline so callers can understand the policy.
///
/// If neither condition fires, this function returns `None` immediately
/// without running the audit.
///
/// # Diagnostics only
///
/// This function logs at INFO level but does NOT change fit semantics.
/// If a persistent rank change is detected across many iterations the
/// user can investigate via the `[AUDIT-DRIFT]` log stream.
///
/// # Arguments
///
/// * `specs` — the current `ParameterBlockSpec` list.
/// * `pilot_audit` — the audit result from the pilot linearisation (β=0).
/// * `beta_pilot` — the pilot β vector (typically zeros).
/// * `beta_current` — the current β vector after some PIRLS/LM iterations.
/// * `family_scalars` — optional per-row primary-state scalars at `beta_current`.
/// * `outer_iter` — the current outer iteration index (0-based).
/// * `every_n_iters` — run the drift audit every this many outer iterations.
pub fn maybe_log_audit_drift(
    specs: &[ParameterBlockSpec],
    pilot_audit: &IdentifiabilityAudit,
    beta_pilot: &[f64],
    beta_current: &[f64],
    family_scalars: Option<&std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    outer_iter: usize,
    every_n_iters: usize,
) -> Option<AuditDriftSummary> {
    // Drift threshold: re-audit when:
    //   (a) relative β movement > 0.5 (substantial step from pilot), OR
    //   (b) every `every_n_iters` outer iterations (amortised cost check).
    // Threshold 0.5 and period 10 are inlined here per T34's contract.
    const BETA_RELATIVE_THRESHOLD: f64 = 0.5;
    const DEFAULT_EVERY_N_ITERS: usize = 10;
    let period = if every_n_iters == 0 {
        DEFAULT_EVERY_N_ITERS
    } else {
        every_n_iters
    };

    let beta_pilot_norm: f64 = beta_pilot.iter().map(|b| b * b).sum::<f64>().sqrt();
    let beta_current_len = beta_current.len();
    let beta_pilot_len = beta_pilot.len();
    let diff_norm: f64 = if beta_current_len == beta_pilot_len {
        beta_current
            .iter()
            .zip(beta_pilot.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    } else {
        // Length mismatch — treat as maximum drift.
        f64::INFINITY
    };
    let beta_relative_change = diff_norm / (beta_pilot_norm + f64::EPSILON);

    let large_beta_movement = beta_relative_change > BETA_RELATIVE_THRESHOLD;
    let periodic_check = (outer_iter % period) == 0;

    if !large_beta_movement && !periodic_check {
        return None;
    }

    // Run the flat audit at the current β.  For channel-aware families the
    // Jacobian callbacks already carry the β-dependent effective Jacobian
    // (they read from family_scalars when present), so re-running the flat
    // audit with a state built from `beta_current` and `family_scalars` is
    // the correct re-evaluation.
    //
    // We pass `family_scalars` as the `channel_hessian` field indirectly:
    // the flat audit calls `effective_jacobian_at` for each block, which
    // internally reads from `family_scalars` when the block has a
    // `jacobian_callback`.  The flat audit omits W refresh (no W matrix); the
    // drift detection here is purely structural (rank of J(β)), not
    // curvature-weighted.  That is the correct identifiability check: structural
    // rank is what tells you whether the model is locally identified at β.
    let p_total: usize = specs.iter().map(|s| s.design.ncols()).sum();
    let beta_for_state: Vec<f64> = if beta_current.len() == p_total {
        beta_current.to_vec()
    } else {
        // Length mismatch: the caller's β does not match the assembled design
        // width, so the audit cannot be evaluated at the real β. Fall back to
        // the origin (β = 0) but record it — the structural rank read at β = 0
        // can differ from the rank at the true β, so a drift verdict resting on
        // this fallback is only a coarse structural check.
        log::debug!(
            "[identifiability-drift] beta_current len {} != design width {}; \
             auditing structural rank at the beta=0 fallback",
            beta_current.len(),
            p_total,
        );
        vec![0.0; p_total]
    };
    let state = crate::families::custom_family::FamilyLinearizationState {
        beta: &beta_for_state,
        family_scalars: family_scalars.cloned(),
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };

    // Re-run the flat audit at beta_current.
    let current_audit = match audit_identifiability_with_state(specs, &state) {
        Ok(a) => a,
        Err(_) => return None,
    };

    let pilot_rank: usize = pilot_audit.blocks.iter().map(|b| b.effective_dim).sum();
    let current_rank: usize = current_audit.blocks.iter().map(|b| b.effective_dim).sum();

    // Build the diff: newly dropped vs recovered.
    let pilot_dropped: std::collections::BTreeSet<(String, usize)> = pilot_audit
        .dropped_columns
        .iter()
        .map(|d| (d.block.clone(), d.column))
        .collect();
    let current_dropped: std::collections::BTreeSet<(String, usize)> = current_audit
        .dropped_columns
        .iter()
        .map(|d| (d.block.clone(), d.column))
        .collect();

    let newly_dropped: Vec<DroppedColumn> = current_audit
        .dropped_columns
        .iter()
        .filter(|d| !pilot_dropped.contains(&(d.block.clone(), d.column)))
        .cloned()
        .collect();

    let recovered: Vec<String> = pilot_audit
        .dropped_columns
        .iter()
        .filter(|d| !current_dropped.contains(&(d.block.clone(), d.column)))
        .map(|d| format!("{}[{}]", d.block, d.column))
        .collect();

    let verdict_changed =
        pilot_rank != current_rank || !newly_dropped.is_empty() || !recovered.is_empty();

    if verdict_changed {
        // Structured INFO log so log-grep can find all drift events.
        //
        // Format: [AUDIT-DRIFT] pilot_rank=<N> current_rank=<N>
        //         beta_relative_change=<f> outer_iter=<N>
        //         newly_dropped=[block[col], ...] recovered=[block[col], ...]
        let newly_str: Vec<String> = newly_dropped
            .iter()
            .map(|d| format!("{}[{}]", d.block, d.column))
            .collect();
        let recovered_str = if recovered.is_empty() {
            "none".to_string()
        } else {
            recovered.join(", ")
        };
        log::info!(
            "[AUDIT-DRIFT] pilot_rank={} current_rank={} \
             beta_relative_change={:.4} outer_iter={} \
             newly_dropped=[{}] recovered=[{}]",
            pilot_rank,
            current_rank,
            beta_relative_change,
            outer_iter,
            if newly_str.is_empty() {
                "none".to_string()
            } else {
                newly_str.join(", ")
            },
            recovered_str,
        );
    }

    Some(AuditDriftSummary {
        pilot_rank,
        current_rank,
        beta_relative_change,
        newly_dropped,
        recovered,
    })
}

/// Run [`audit_identifiability`] with an explicit [`FamilyLinearizationState`]
/// so callers can pass the current β and `family_scalars` at any point in the
/// outer loop.
///
/// Identical to [`audit_identifiability`] in every step (per-block QR,
/// gauge-priority sort, joint RRQR, pairwise overlap scan, drop attribution,
/// gauge-resolves-rank-deficiency logic, hard-alias / fatal classification) —
/// only the [`FamilyLinearizationState`] handed to each block's
/// `effective_jacobian_at` differs. For families without a `jacobian_callback`
/// the design is returned as-is regardless of `state`.
pub fn audit_identifiability_with_state(
    specs: &[ParameterBlockSpec],
    state: &crate::families::custom_family::FamilyLinearizationState<'_>,
) -> Result<IdentifiabilityAudit, EstimationError> {
    audit_identifiability_impl(specs, state)
}

fn locate_block_column(
    col_offsets: &[usize],
    joint_col: usize,
) -> Result<(usize, usize), EstimationError> {
    // col_offsets has length specs.len() + 1; col_offsets[i..i+1] is
    // the joint-column range for block i. Linear scan is fine — the
    // table is tiny (one entry per block).
    for i in 0..col_offsets.len() - 1 {
        if joint_col >= col_offsets[i] && joint_col < col_offsets[i + 1] {
            return Ok((i, joint_col - col_offsets[i]));
        }
    }
    Err(EstimationError::LayoutError(format!(
        "identifiability::audit::locate_block_column: joint_col {joint_col} \
         outside col_offsets range (max = {})",
        col_offsets.last().copied().unwrap_or(0),
    )))
}

/// Structural (λ-invariant) penalty for a block: the unit-weight sum of its
/// penalty matrices, materialised dense (`p_block × p_block`), or `None` when
/// the block carries no penalty.
///
/// Identifiability of a PENALIZED block is governed by `H + S`, not the raw
/// design `H` alone: a direction killed by the data design but COVERED by the
/// penalty is still fully estimated (the penalized normal equations `JᵀWJ + S`
/// are non-singular there). The block is genuinely non-identifiable only along
/// `ker(J) ∩ ker(S)`. We therefore audit the rank of the design AUGMENTED with
/// the penalty's rows — `[J; S]`, whose null space is exactly `ker(J) ∩ ker(S)`
/// (`[J; S] v = 0 ⟺ Jv = 0 ∧ Sv = 0`), so appending `S` itself is equivalent to
/// appending any square-root `√S` for the rank/null verdict and avoids a matrix
/// square root.
///
/// Using the unit-weight STRUCTURAL sum (not a fitted λ) keeps the gate
/// ρ-invariant: only the penalties' shared null space `∩_m ker(S_m)` matters,
/// which is independent of the smoothing-parameter values. A block with no
/// penalty returns `None`, so its augmented design is just `J` — the gate then
/// reduces EXACTLY to the historical raw-design rank check, leaving every
/// unpenalized block/family's verdict unchanged.
pub(crate) fn block_structural_penalty_dense(spec: &ParameterBlockSpec) -> Option<Array2<f64>> {
    let p = spec.design.ncols();
    if p == 0 || spec.penalties.is_empty() {
        return None;
    }
    let mut s = Array2::<f64>::zeros((p, p));
    let mut any = false;
    for penalty in &spec.penalties {
        let dense = penalty.to_dense();
        if dense.nrows() == p && dense.ncols() == p {
            s += &dense;
            any = true;
        }
    }
    any.then_some(s)
}

/// Penalty-aware **rank-revealing** column rank of a block: the numerical rank
/// of the design `J` AUGMENTED with the block's structural penalty rows `[J; S]`
/// (column count unchanged at `p_block`; null space = `ker(J) ∩ ker(S)`). When
/// the block has no penalty this is the rank of `J` itself.
///
/// Uses column-pivoted RRQR (`rrqr_with_permutation`), which IS rank-revealing,
/// rather than a plain (non-pivoted) QR whose R diagonal can scatter a near-zero
/// pivot early and under/over-count the rank of a deficient matrix. Tolerance
/// matches the joint RRQR.
fn block_penalty_aware_rank(
    block: &Array2<f64>,
    structural_penalty: Option<&Array2<f64>>,
) -> Result<usize, EstimationError> {
    let augmented_owned;
    let target: &Array2<f64> = match structural_penalty {
        None => block,
        Some(s) => {
            let n = block.nrows();
            let p = block.ncols();
            let mut augmented = Array2::<f64>::zeros((n + p, p));
            augmented.slice_mut(ndarray::s![..n, ..]).assign(block);
            augmented.slice_mut(ndarray::s![n.., ..]).assign(s);
            augmented_owned = augmented;
            &augmented_owned
        }
    };
    if target.ncols() == 0 {
        return Ok(0);
    }
    rrqr_with_permutation(target, default_rrqr_rank_alpha())
        .map(|r| r.rank)
        .map_err(|e| {
            EstimationError::LayoutError(format!(
                "identifiability audit per-block RRQR failed: {e:?}"
            ))
        })
}

pub(crate) fn count_rank(singular_values: &[f64], n: usize, p: usize) -> usize {
    if singular_values.is_empty() {
        return 0;
    }
    let leading = singular_values.first().copied().unwrap_or(0.0);
    let rank_alpha = default_rrqr_rank_alpha();
    let tol = rank_alpha * f64::EPSILON * (n.max(p).max(1) as f64) * leading.max(1.0);
    singular_values.iter().filter(|&&v| v > tol).count()
}

/// Numerical rank of a design `D` (with `n_total` rows and `p` columns) given
/// only its accumulated `(p × p)` Gram matrix `G = Dᵀ D`.
///
/// The streaming SAE fit never materializes the full `(N × M_k)` weighted
/// design `D_k = diag(a_·k)·Φ_k`; it accumulates `G_k = D_kᵀ D_k` online across
/// row chunks. The singular values of `D` are the square roots of the
/// eigenvalues of `G` (`G = V diag(σ²) Vᵀ`), so the same RRQR rank tolerance
/// that [`count_rank`] applies to QR pivots applies to `√λ`. This lets the
/// pre-fit decoder identifiability audit run chunk-by-chunk with `O(M_k²)`
/// state instead of an `O(N · M_k)` design retain.
///
/// Negative eigenvalues from finite-precision accumulation are clamped to zero
/// before the square root. Returns the count of singular values above the
/// tolerance, identical in convention to [`count_rank`].
pub(crate) fn rank_of_gram(gram: &Array2<f64>, n_total: usize) -> Result<usize, EstimationError> {
    let p = gram.ncols();
    if p == 0 {
        return Ok(0);
    }
    let (evals, _evecs) = gram
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    let mut singular: Vec<f64> = evals.iter().map(|&lambda| lambda.max(0.0).sqrt()).collect();
    singular.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(count_rank(&singular, n_total, p))
}

/// Error produced when the MAP uniqueness condition
/// `ker(J^T W J) ∩ ker(S) = {0}` is violated.
///
/// A null direction `n` of `J^T W J` with `n^T S n = 0` means the posterior
/// is flat along `n`: no likelihood curvature AND no penalty curvature,
/// so the MAP estimate is non-unique.  The error names the offending
/// direction and the dominant block (the block whose columns have the
/// largest component in `n`) so the caller can trace which smooth term
/// contributed the unpenalised null direction.
#[derive(Debug, Clone)]
pub struct MapUniquenessError {
    /// Human-readable description of the failure, including the dominant block.
    pub message: String,
    /// Name of the block whose columns dominate the null direction.
    pub dominant_block: String,
    /// Index of the null direction (0-based among directions below tolerance).
    pub null_direction_index: usize,
    /// `n^T S n` for the offending null direction (≈ 0.0).
    pub penalty_quadratic_form: f64,
}

impl std::fmt::Display for MapUniquenessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Check the MAP estimate uniqueness condition `ker(J^T W J) ∩ ker(S) = {0}`.
///
/// # Arguments
///
/// * `j_joint` — the `(n, p_total)` joint design matrix after canonicalisation.
///   `J^T W J` is formed using `W = diag(w)`.  If `w` is empty, `W = I` is used.
/// * `w_diag` — per-row weights (length `n`, all non-negative).  Pass an empty
///   slice to use the identity weight.
/// * `s_joint` — the `(p_total, p_total)` joint smoothness penalty matrix
///   `S = blockdiag(S_1, ..., S_K)`.  This is the sum over all blocks of the
///   block-embedded penalty, assembled by the caller.
/// * `specs` — the `ParameterBlockSpec` slice (same order as the columns of
///   `j_joint`) used to name the dominant block in the error.
/// * `col_offsets` — `specs.len() + 1` cumulative column offsets so that
///   block `i` occupies `j_joint[:, col_offsets[i] .. col_offsets[i+1]]`.
///
/// # Returns
///
/// `Ok(())` when the condition holds for every null direction (i.e. every
/// null direction of `J^T W J` carries `n^T S n > null_tol`).
///
/// `Err(MapUniquenessError)` for the first null direction (sorted by
/// ascending `n^T S n`) that violates the condition.
pub fn check_map_uniqueness(
    j_joint: &Array2<f64>,
    w_diag: &[f64],
    s_joint: &Array2<f64>,
    specs: &[ParameterBlockSpec],
    col_offsets: &[usize],
) -> Result<(), MapUniquenessError> {
    let n = j_joint.nrows();
    let p = j_joint.ncols();

    if p == 0 {
        return Ok(());
    }

    // Form G = J^T W J as a (p, p) matrix.
    // G[i, j] = Σ_k w_k * J[k,i] * J[k,j]
    let g: Array2<f64> = if w_diag.is_empty() {
        // W = I: G = J^T J. The blocked, parallel faer crossproduct kernel
        // computes `Jᵀ·diag(1)·J` as a single cache-friendly GEMM — bit-for-bit
        // the same symmetric Gram as the naive `Σ_k J[k,i]·J[k,j]` triple loop,
        // but O(n·p²) FLOPs run through BLAS-3 instead of scalar ndarray
        // indexing. At biobank scale (n≈3·10⁵, p≈85) this is the dominant cost
        // of the MAP-uniqueness check, so the GEMM is the principled form.
        let unit_weights = Array1::<f64>::ones(n);
        crate::linalg::faer_ndarray::fast_xt_diag_x_with_parallelism(
            j_joint,
            &unit_weights,
            faer::get_global_parallelism(),
        )
    } else {
        let mut g = Array2::<f64>::zeros((p, p));
        assert_eq!(
            w_diag.len(),
            n,
            "check_map_uniqueness: w_diag length {} != n {}",
            w_diag.len(),
            n,
        );
        for k in 0..n {
            let wk = w_diag[k];
            if wk == 0.0 {
                continue;
            }
            for i in 0..p {
                let wji = wk * j_joint[[k, i]];
                if wji == 0.0 {
                    continue;
                }
                for j in i..p {
                    let val = wji * j_joint[[k, j]];
                    g[[i, j]] += val;
                    if i != j {
                        g[[j, i]] += val;
                    }
                }
            }
        }
        g
    };

    // Eigendecompose G = V diag(λ) V^T (symmetric).
    let (evals, evecs) = match g.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(e) => {
            // Eigendecomposition failure: skip the check rather than
            // producing a spurious failure — log and return Ok.
            log::warn!(
                "[MAP-UNIQUE] check_map_uniqueness: eigendecomposition of J^T W J failed \
                 ({e:?}); skipping MAP uniqueness check",
            );
            return Ok(());
        }
    };

    // Determine the null-space tolerance.
    // Use the same RRQR_RANK_ALPHA · ε · p · λ_max convention as the
    // rank counters elsewhere in this module.
    let lambda_max = evals.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    let rank_alpha = default_rrqr_rank_alpha();
    let null_tol = rank_alpha * f64::EPSILON * (p as f64) * lambda_max;

    // Collect null directions: eigenvectors whose eigenvalue is below null_tol.
    // Sort by ascending eigenvalue so the most-null direction comes first.
    let mut null_dirs: Vec<(f64, usize)> = evals
        .iter()
        .enumerate()
        .filter(|&(_, &lam)| lam < null_tol)
        .map(|(idx, &lam)| (lam, idx))
        .collect();
    null_dirs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    if null_dirs.is_empty() {
        return Ok(());
    }

    // Penalty tolerance for n^T S n: use a relative threshold proportional
    // to the Frobenius norm of S.
    let s_frob_sq: f64 = s_joint.iter().map(|v| v * v).sum();
    let pen_tol = null_tol * s_frob_sq.sqrt().max(1.0);

    for (dir_idx, (lam, evec_col)) in null_dirs.iter().enumerate() {
        let n_vec = evecs.column(*evec_col);
        // Compute n^T S n
        let sn: Array1<f64> = s_joint.dot(&n_vec.to_owned());
        let ntsn: f64 = n_vec.iter().zip(sn.iter()).map(|(ni, si)| ni * si).sum();

        if ntsn < pen_tol {
            // Find the dominant block: the block whose columns have the
            // largest cumulative squared component in n_vec.
            let dominant_block =
                dominant_block_for_direction(&n_vec.to_owned(), specs, col_offsets);

            let message = format!(
                "MAP estimate is non-unique: null direction {} of J^T W J (eigenvalue {lam:.3e}) \
                 has n^T S n = {ntsn:.3e} < tolerance {pen_tol:.3e}; \
                 the MAP is flat along this direction (no likelihood curvature, no penalty \
                 curvature); dominant block: '{}'. \
                 Fix: add a non-degenerate smoothness penalty to block '{}' that covers this \
                 direction, or remove the unpenalised null direction from the model.",
                dir_idx, dominant_block, dominant_block,
            );
            return Err(MapUniquenessError {
                message,
                dominant_block,
                null_direction_index: dir_idx,
                penalty_quadratic_form: ntsn,
            });
        }
    }

    Ok(())
}

/// Identify which block's columns contribute most (in L2 norm) to the
/// given null direction vector.
fn dominant_block_for_direction(
    n_vec: &Array1<f64>,
    specs: &[ParameterBlockSpec],
    col_offsets: &[usize],
) -> String {
    let mut best_block = specs.first().map(|s| s.name.as_str()).unwrap_or("unknown");
    let mut best_sq = 0.0_f64;
    for (i, spec) in specs.iter().enumerate() {
        if i + 1 >= col_offsets.len() {
            break;
        }
        let lo = col_offsets[i];
        let hi = col_offsets[i + 1];
        let sq: f64 = (lo..hi).map(|c| n_vec[c] * n_vec[c]).sum();
        if sq > best_sq {
            best_sq = sq;
            best_block = spec.name.as_str();
        }
    }
    best_block.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    use linspace as linspace_minus_one_to_one;
    use ndarray::Array2;

    use crate::test_support::spec_from_dense;

    fn linspace(n: usize) -> ndarray::Array1<f64> {
        if n <= 1 {
            return ndarray::Array1::<f64>::zeros(n.max(1));
        }
        ndarray::Array1::linspace(-1.0, 1.0, n)
    }

    /// Test 1: a model with no aliasing → audit returns clean, no
    /// drops, fatal=false.
    #[test]
    fn audit_no_aliasing_returns_clean() {
        let n = 64;
        let x = linspace_minus_one_to_one(n);
        // Parametric: [1, x]. Smooth: [x², x³] — orthogonal-ish to the
        // parametric directions, no exact alias.
        let mut parametric = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            parametric[[i, 0]] = 1.0;
            parametric[[i, 1]] = x[i];
        }
        let mut smooth = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            smooth[[i, 0]] = x[i] * x[i];
            smooth[[i, 1]] = x[i] * x[i] * x[i];
        }
        let specs = [
            spec_from_dense("intercept", parametric),
            spec_from_dense("smooth_x", smooth),
        ];
        let audit = audit_identifiability(&specs).expect("audit must succeed on clean specs");
        assert!(
            !audit.fatal,
            "no aliasing must not be fatal: {}",
            audit.summary
        );
        assert!(
            audit.aliased_pairs.is_empty(),
            "expected no alias pairs; got {:?}",
            audit.aliased_pairs,
        );
        assert!(
            audit.dropped_columns.is_empty(),
            "expected no dropped columns; got {:?}",
            audit.dropped_columns,
        );
        assert_eq!(audit.blocks.len(), 2);
        assert_eq!(audit.blocks[0].original_dim, 2);
        assert_eq!(audit.blocks[0].effective_dim, 2);
        assert_eq!(audit.blocks[1].original_dim, 2);
        assert_eq!(audit.blocks[1].effective_dim, 2);
    }

    /// Test 2: a smooth's constant column aliased with a separate
    /// parametric intercept → audit drops one column AND now flags
    /// the configuration as fatal under the task #5 halt-or-repair gate.
    /// The two blocks contribute the same direction up to numerical
    /// noise (overlap == 1.0): the inner KKT system is structurally
    /// rank-deficient regardless of penalty, so the audit must refuse
    /// the fit with an actionable message rather than warn-and-proceed.
    #[test]
    fn audit_smooth_constant_aliased_with_intercept() {
        let n = 64;
        let x = linspace_minus_one_to_one(n);
        // Parametric: [1]. Smooth: [1, x², x³] — its leading column
        // exactly reproduces the parametric intercept.
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        let mut smooth = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            smooth[[i, 0]] = 1.0;
            smooth[[i, 1]] = x[i] * x[i];
            smooth[[i, 2]] = x[i] * x[i] * x[i];
        }
        let specs = [
            spec_from_dense("intercept", parametric),
            spec_from_dense("smooth_with_const", smooth),
        ];
        let audit = audit_identifiability(&specs).expect("audit must succeed");
        assert!(
            audit.fatal,
            "exact intercept~smooth-constant alias must be fatal under the halt gate: {}",
            audit.summary,
        );
        assert!(
            !audit.aliased_pairs.is_empty(),
            "smooth-constant aliased with intercept must surface at least one alias pair",
        );
        assert!(
            !audit.dropped_columns.is_empty(),
            "smooth-constant aliased with intercept must populate dropped_columns",
        );
        // The dropped column should belong to one of the two blocks
        // (RRQR picks pivot order; we don't pin which block wins).
        for drop in &audit.dropped_columns {
            assert!(
                drop.block == "intercept" || drop.block == "smooth_with_const",
                "unexpected drop block name {:?}",
                drop.block,
            );
        }
        // Joint rank should be exactly the count of linearly independent
        // columns: intercept + x² + x³ = 3.
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 3,
            "expected 3 surviving directions; got {total_kept} (summary: {})",
            audit.summary,
        );
        // Actionable message must name both offending blocks and the
        // reparam suggestion so the caller can act on the failure.
        assert!(
            audit.summary.contains("intercept") && audit.summary.contains("smooth_with_const"),
            "fatal summary must name both blocks; got {:?}",
            audit.summary,
        );
        assert!(
            audit.summary.contains("reparam")
                || audit.summary.contains("sum-to-zero")
                || audit.summary.contains("orthogonal")
                || audit.summary.contains("absorb"),
            "fatal summary must include a reparameterisation suggestion; got {:?}",
            audit.summary,
        );
    }

    /// Test 3: two smooths on the same covariate axis with a shared
    /// linear direction → audit drops one column.
    #[test]
    fn audit_two_smooths_share_linear_direction() {
        let n = 64;
        let x = linspace_minus_one_to_one(n);
        // Both smooths contain a column that equals `x`; they also
        // each carry a quadratic direction the other doesn't.
        let mut smooth_a = Array2::<f64>::zeros((n, 2));
        let mut smooth_b = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            smooth_a[[i, 0]] = x[i];
            smooth_a[[i, 1]] = x[i] * x[i];
            smooth_b[[i, 0]] = x[i];
            smooth_b[[i, 1]] = (x[i] - 0.3).powi(2);
        }
        let specs = [
            spec_from_dense("smooth_a", smooth_a),
            spec_from_dense("smooth_b", smooth_b),
        ];
        let audit = audit_identifiability(&specs).expect("audit must succeed");
        // Under the task #5 halt gate, an x~x alias at overlap ~ 1.0 is
        // fatal — two distinct blocks contributing the same direction is
        // structurally unfittable regardless of attribution.
        assert!(
            audit.fatal,
            "exact x~x alias across two smooth blocks must be fatal under the halt gate: {}",
            audit.summary,
        );
        let cross_linear_pair = audit
            .aliased_pairs
            .iter()
            .find(|p| p.block_a == "smooth_a" && p.block_b == "smooth_b" && p.overlap > 0.999);
        assert!(
            cross_linear_pair.is_some(),
            "expected an x~x alias pair between the two smooths; got pairs {:?}",
            audit.aliased_pairs,
        );
        assert!(
            !audit.dropped_columns.is_empty(),
            "RRQR must demote one of the duplicated linear columns",
        );
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 3,
            "expected 3 independent directions (1 shared linear + 2 quadratics); got {total_kept}",
        );
    }

    /// Test 4: an ambiguous high-magnitude alias that the pairwise
    /// scan misses but RRQR catches as joint-rank deficiency. We
    /// engineer a 3-way structural alias: parametric `[1]`, smooth
    /// A `[1 + ε·x]`, smooth B `[1 - ε·x]`. Each individual pair
    /// has overlap below 1.0 but the joint design's third column
    /// lies in the span of the first two — RRQR demotes one and
    /// attributes it via dropped_columns; pairwise scan may or may
    /// not catch all three pairs above threshold. Either way the
    /// fit must proceed (fatal=false) because dropped_columns is
    /// populated.
    #[test]
    fn audit_three_way_alias_is_attributed_not_fatal() {
        let n = 64;
        let x = linspace_minus_one_to_one(n);
        let eps = 0.5;
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        let smooth_a = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| 1.0 + eps * x[i]);
        let smooth_b = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| 1.0 - eps * x[i]);
        let specs = [
            spec_from_dense("intercept", parametric),
            spec_from_dense("smooth_a", smooth_a),
            spec_from_dense("smooth_b", smooth_b),
        ];
        let audit = audit_identifiability(&specs).expect("audit must succeed");
        assert!(
            !audit.dropped_columns.is_empty(),
            "RRQR must attribute the three-way alias as a dropped column",
        );
        // Under the task #5 halt gate, joint_rank < joint_cols is fatal
        // even with attribution: the inner KKT system inherits the
        // unattributed null direction and the outer optimiser will spin.
        assert!(
            audit.fatal,
            "three-way alias with joint rank < joint cols must be fatal under the halt gate: {}",
            audit.summary,
        );
        // Joint rank = 2 (the three columns span at most {1, x}).
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 2,
            "three-way alias collapses to rank 2; got {total_kept}"
        );
    }

    /// Test 5: end-to-end shape on a large-scale-like configuration —
    /// 4 blocks, ~50 total columns, with one intentional cross-block
    /// linear alias seeded in. The audit must complete in well under
    /// a second and produce a single attributed drop with the
    /// expected joint rank.
    #[test]
    fn audit_large_scale_shape_end_to_end() {
        let n = 1024;
        let x = linspace_minus_one_to_one(n);
        // Block 0: parametric intercept + age-linear.
        let mut parametric = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            parametric[[i, 0]] = 1.0;
            parametric[[i, 1]] = x[i];
        }
        // Block 1: smooth in x — 8 genuinely independent radial basis
        // columns (Gaussian bumps at 8 distinct, well-separated knots). The
        // earlier `(x − (k−4)·0.2)²` construction was rank-3 (every column
        // expands to `x² − 2(k−4)(0.2)x + const` ∈ span{1, x, x²}), so 5 of
        // the 8 columns were genuinely redundant and RRQR correctly demoted
        // them — contradicting the fixture's "8 independent columns, one
        // seeded drop" premise. Gaussian RBFs at distinct centres are
        // linearly independent and do not lie in span{1, x}, so the block
        // contributes its full rank-8 and the ONLY drop is the seeded x~x
        // alias in block 3.
        let mut s_x = Array2::<f64>::zeros((n, 8));
        let rbf_width = 0.30_f64;
        for i in 0..n {
            for k in 0..8 {
                let center = (k as f64 - 3.5) * 0.25;
                let d = (x[i] - center) / rbf_width;
                s_x[[i, k]] = (-0.5 * d * d).exp();
            }
        }
        // Block 2: smooth in sin(x) — 6 columns. No alias with block 1.
        // Start at frequency 4 (k+4), NOT 1: sin(x) and sin(2x) share their
        // leading Taylor term with `x` (sin(x) = x − x³/6 + …), and the joint
        // design [1, x, RBFs, sin(x), sin(2x), …, x_alias, cos(kx)] inherits
        // a numerical near-degeneracy at σ ≈ 1.8e-10 — the right singular
        // vector is dominated by ≈ +0.88·sin(x) − 0.31·sin(2x) − 0.24·x − …,
        // i.e. a `sin(x) ≈ x + …` Taylor identity. That σ lands BELOW the
        // joint RRQR rank tolerance (≈ 7.3e-10) and so RRQR demotes a SECOND
        // column on top of the seeded x~x alias, producing `dropped.len() == 2`
        // and failing the "exactly the seeded alias" assertion. Frequencies
        // ≥ 4 keep the next-smallest σ at ≈ 6e-8, comfortably above tol, so
        // only the genuine x~x seeded alias is demoted.
        let mut s_sin = Array2::<f64>::zeros((n, 6));
        for i in 0..n {
            for k in 0..6 {
                s_sin[[i, k]] = ((k as f64 + 4.0) * x[i]).sin();
            }
        }
        // Block 3: deliberately seeded alias — first column is exactly
        // `x` (duplicates parametric block's column 1).
        let mut alias_block = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            alias_block[[i, 0]] = x[i];
            alias_block[[i, 1]] = x[i].cos();
            alias_block[[i, 2]] = (2.0 * x[i]).cos();
            alias_block[[i, 3]] = (3.0 * x[i]).cos();
        }
        let specs = [
            spec_from_dense("parametric", parametric),
            spec_from_dense("s_x", s_x),
            spec_from_dense("s_sin", s_sin),
            spec_from_dense("alias_block", alias_block),
        ];
        let audit = audit_identifiability(&specs).expect("large-scale audit must succeed");
        // The seeded x~x alias is exactly the large-scale failure shape the
        // task #5 halt gate exists to refuse: two distinct blocks
        // contributing the same direction at overlap 1.0. Must be fatal.
        assert!(
            audit.fatal,
            "seeded large-scale exact x~x alias must be fatal under the halt gate: {}",
            audit.summary,
        );
        assert!(
            !audit.dropped_columns.is_empty(),
            "seeded x~x alias must produce at least one dropped column",
        );
        // The alias is exactly one direction; expect exactly one drop.
        assert_eq!(
            audit.dropped_columns.len(),
            1,
            "large-scale audit should attribute exactly the seeded alias; \
             got {:?}",
            audit.dropped_columns,
        );
        // Total effective dim = 2 + 8 + 6 + 4 − 1 = 19.
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 19,
            "large-scale: expected 19 kept directions; got {total_kept} ({})",
            audit.summary,
        );
    }
}
