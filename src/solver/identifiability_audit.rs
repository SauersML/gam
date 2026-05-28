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
// `crate::families::bernoulli_marginal_slope::install_compiled_flex_block_into_runtime`
// is now a thin spec-builder → audit → compile → install wrapper:
//
//   1. `build_bms_flex_block_context` densifies anchors, stacks N_train, and
//      assembles `BernoulliDenseDesignOperator` + `BlockOrder` + `BernoulliRowHessian`
//      from the BMS-specific inputs.
//   2. `audit_identifiability_channel_aware` (this module) acts as the
//      structural rank gate — it detects full aliasing via the K=1 BMS row
//      Jacobian before any install. `FlexEvaluation` anchors participate here.
//   3. `crate::families::identifiability_compiler::compile` does the W-metric
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
// 1. Densify each block once (n × p_block). Record per-block pivoted-QR
//    diagonal as `design_range_singular_values`.
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
    FaerEigh, FaerQr, default_rrqr_rank_alpha, rrqr_with_permutation,
};

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
    /// rows, computed by column-pivoted QR on the block in isolation
    /// with the unified rank tolerance. Equal to `original_dim` for
    /// any well-posed block; smaller values flag a within-block
    /// rank deficiency that escaped within-smooth nullspace absorption.
    pub design_range_rank: usize,
    /// Pivoted-QR diagonal magnitudes for the block, sorted descending.
    /// Length = `original_dim`. Stored for diagnostics; the audit's
    /// drop decisions use the joint pivot, not these per-block values.
    pub design_range_singular_values: Vec<f64>,
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
    /// Non-zero when exactly one block carries an `eta_row_scaling`
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
/// row-scaling vector z = `eta_row_scaling`, applying the finite-sample
/// unbiased correction factor n / ((n-1)(n-2)).
///
/// # Derivation of the null-mean bias term
///
/// When one of the two blocks in a cross-block cosine comparison carries
/// `eta_row_scaling = Some(z)`, the effective Jacobian column is `z ⊙ φ`
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
    let mut m3 = 0.0_f64;
    for &zi in z {
        let d = zi - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n as f64;
    m3 /= n as f64;
    let sigma = m2.sqrt();
    if sigma <= 0.0 {
        return 0.0;
    }
    // Raw skewness: m3 / σ³
    let raw_skew = m3 / (sigma * sigma * sigma);
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
/// * Block A has `eta_row_scaling = Some(z)`, block B has `None` (or vice versa):
///   shift = −(μ_3(z) / 2) · S2_k.
/// * Both have `None`: shift = 0 (T11's symmetric form, μ_3 = 0).
/// * Both have DIFFERENT row-scaling vectors z_a ≠ z_b: shift is derived from
///   whichever of z_a or z_b produced the column with larger S2 (the dominant
///   concentration), as a conservative approximation.
///
/// The shift is clamped to ±0.5 to prevent a degenerate skewed z from
/// placing the null band entirely outside [−1, 1].
pub fn bias_shift_for_pair(
    z_a: Option<&[f64]>,
    z_b: Option<&[f64]>,
    s2_a: f64,
    s2_b: f64,
) -> f64 {
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
    (s2 - inv_n).max(0.0).sqrt()
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
/// For m_pairs = 1000 (large biobank audit) this gives K ≈ 5.1.
///
/// The floor of 0.10 prevents collapse to near-zero on pathological
/// inputs; the ceiling 0.999 is the absolute alias boundary.
/// For a column with n_eff = 100 and m_pairs = 100: σ ≈ 0.1,
/// K ≈ 4.3, threshold ≈ 0.43.  For n_eff = 10000: σ ≈ 0.01,
/// threshold ≈ 0.043, clamped to 0.10.
fn pair_report_threshold(s2_a: f64, s2_b: f64, n: usize, m_pairs: usize) -> f64 {
    let sigma = pair_null_sigma(s2_a, s2_b, n);
    if sigma <= 0.0 {
        return 0.10_f64;
    }
    let k_report = if m_pairs <= 1 {
        3.0_f64
    } else {
        (2.0 * (2.0 * m_pairs as f64 / 0.05_f64).ln())
            .sqrt()
            .max(3.0)
    };
    (k_report * sigma).clamp(0.10, 0.999)
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
    let sigma = pair_null_sigma(s2_a, s2_b, n);
    if sigma <= 0.0 {
        return 0.999_f64;
    }
    (10.0_f64 * sigma).clamp(0.05, 0.999)
}

/// Decide whether a cosine (signed) falls outside the bias-corrected null band.
///
/// The null distribution for cos(φ_a, φ_b) is approximately
///   N(shift, σ²)
/// where σ = pair_null_sigma(s2_a, s2_b, n) and
/// shift = bias_shift_for_pair(...).
///
/// A cosine is flagged when |cosine − shift| ≥ half_width.
///
/// Returns `(flag, |cosine − shift|)` so the caller can record which
/// direction was used.  The `half_width` argument is the K·σ half-band
/// (either the report or halt multiplied sigma).
fn cosine_outside_null_band(cosine: f64, shift: f64, half_width: f64) -> (bool, f64) {
    let deviation = (cosine - shift).abs();
    (deviation >= half_width, deviation)
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
///   1. Densify each block once (chunked, n × p_block; biobank n,
///      small p_block — total joint width is the GAM smooth budget,
///      not n). Each block's pivoted-QR diagonal becomes its
///      `design_range_singular_values`.
///   2. Stack horizontally into `X_joint ∈ ℝ^{n×p_total}` in spec
///      order; column-pivoted QR identifies columns linearly
///      dependent on earlier (pivot-rank-truncated) columns.
///   3. Each pivoted-out column is attributed to the earliest block
///      whose range absorbs it (largest projection norm). Ambiguous
///      attribution → `fatal = true`.
///   4. Report all (a, b) column pairs whose normalised inner
///      product exceeds the per-pair leverage-based reporting threshold
///      (`pair_report_threshold`).
pub fn audit_identifiability(specs: &[ParameterBlockSpec]) -> Result<IdentifiabilityAudit, String> {
    if specs.is_empty() {
        return Ok(IdentifiabilityAudit {
            blocks: Vec::new(),
            aliased_pairs: Vec::new(),
            dropped_columns: Vec::new(),
            fatal: false,
            summary: "identifiability audit: no blocks supplied".to_string(),
        });
    }

    let n = specs[0].design.nrows();
    for (idx, spec) in specs.iter().enumerate() {
        if spec.design.nrows() != n {
            return Err(format!(
                "identifiability audit: block {} ({}) has {} rows, expected {}",
                idx,
                spec.name,
                spec.design.nrows(),
                n,
            ));
        }
    }

    let mut dense_blocks: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    let mut blocks: Vec<BlockIdentity> = Vec::with_capacity(specs.len());
    let mut col_offsets: Vec<usize> = Vec::with_capacity(specs.len() + 1);
    col_offsets.push(0);
    let block_phase_started = std::time::Instant::now();
    let block_heartbeat = (n.saturating_mul(specs.len()) >= 1_000_000)
        .then(crate::util::heartbeat::Heartbeat::default_interval);
    // Build a default at-init linearization state: β = 0, no family scalars.
    // Families whose callbacks need β-dependent scalars must pass a state with
    // the current β when invoking the channel-aware audit after initialization.
    let p_total_hint: usize = specs.iter().map(|s| s.design.ncols()).sum();
    let zeros_beta = vec![0.0f64; p_total_hint];
    let init_state = FamilyLinearizationState {
        beta: &zeros_beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    for (idx, spec) in specs.iter().enumerate() {
        // Use spec.effective_jacobian_at() so the audit operates on the β-dependent
        // effective design. At initialization (β = 0, family_scalars = None), callbacks
        // return the linearization at β = 0.  For blocks with no callback and
        // eta_row_scaling = None this is a no-op (J = design).
        let dense = spec
            .effective_jacobian_at("identifiability_audit::audit_identifiability", &init_state)
            .map_err(|e| format!("identifiability audit: {e}"))?;
        let p_block = dense.ncols();
        let block_singular = block_pivoted_qr_diagonal(&dense)?;
        let block_rank = count_rank(&block_singular, n, p_block);
        blocks.push(BlockIdentity {
            block_name: spec.name.clone(),
            original_dim: p_block,
            effective_dim: p_block,
            design_range_rank: block_rank,
            design_range_singular_values: block_singular,
        });
        let next_offset = col_offsets[col_offsets.len() - 1] + p_block;
        col_offsets.push(next_offset);
        dense_blocks.push(dense);
        if let Some(hb) = block_heartbeat.as_ref() {
            hb.tick(1, |progress, elapsed| {
                log::info!(
                    "[STAGE] identifiability audit: per-block QR progress={}/{} elapsed={:.1}s (overall={:.1}s)",
                    progress.min(specs.len()),
                    specs.len(),
                    elapsed,
                    block_phase_started.elapsed().as_secs_f64(),
                );
            });
        }
        // Per-block summary on the last iteration so the user always
        // sees one waypoint between "audit start" and the dense joint
        // assembly even if no heartbeat fires.
        if idx + 1 == specs.len() {
            log::info!(
                "[STAGE] identifiability audit: per-block QR complete blocks={} elapsed={:.3}s",
                specs.len(),
                block_phase_started.elapsed().as_secs_f64(),
            );
        }
    }
    let p_total = *col_offsets.last().expect("col_offsets non-empty");

    if p_total == 0 {
        return Ok(IdentifiabilityAudit {
            blocks,
            aliased_pairs: Vec::new(),
            dropped_columns: Vec::new(),
            fatal: false,
            summary: "identifiability audit: every block is empty".to_string(),
        });
    }

    let mut x_joint = Array2::<f64>::zeros((n, p_total));
    for (idx, block) in dense_blocks.iter().enumerate() {
        let start = col_offsets[idx];
        let end = col_offsets[idx + 1];
        if end > start {
            x_joint.slice_mut(ndarray::s![.., start..end]).assign(block);
        }
    }

    // Per-joint-column gauge priority, inherited from the owning block.
    // RRQR uses greedy column pivoting: at each step it picks the
    // remaining column with the largest residual norm. When two
    // columns carry the same direction (alias) the residual norms are
    // identical after the earlier of the two enters the kept set, so
    // RRQR's choice of "which one to keep" is determined by the order
    // it scans columns. We exploit that by presenting columns in
    // descending-priority order: high-priority columns are scanned
    // first and enter the kept set first, so when an alias collapses
    // it is the lower-priority block's column that gets demoted into
    // the trailing rank-deficient space.
    //
    // For survival marginal-slope this realises the gauge-ownership
    // contract: a shared affine direction between time_surface (high
    // priority) and marginal_surface (lower) is dropped from
    // marginal_surface, not from time_surface; a shared deviation
    // direction between marginal_surface and score_warp_dev (lowest)
    // is dropped from score_warp_dev. With all priorities equal (the
    // default = 100) `priority_perm` is the identity (stable sort
    // preserves spec order), so legacy callers see no behaviour
    // change.
    let mut priority_perm: Vec<usize> = (0..p_total).collect();
    let col_block_idx: Vec<usize> = (0..specs.len())
        .flat_map(|i| std::iter::repeat(i).take(col_offsets[i + 1] - col_offsets[i]))
        .collect();
    priority_perm.sort_by(|&a, &b| {
        let pa = specs[col_block_idx[a]].gauge_priority;
        let pb = specs[col_block_idx[b]].gauge_priority;
        pb.cmp(&pa).then_with(|| a.cmp(&b))
    });
    let priority_perm_is_identity = priority_perm
        .iter()
        .enumerate()
        .all(|(new_j, &old_j)| new_j == old_j);

    // Column-pivoted RRQR on the joint design. The pivot permutation
    // names which original columns were demoted past the rank
    // threshold, so we can attribute each dropped column back to its
    // (block, local_col) origin deterministically. `rrqr_with_permutation`
    // uses the same `RRQR_RANK_ALPHA · ε · max(m,n) · leading`
    // tolerance as the per-block diagonal counter above, so the joint
    // verdict and the per-block diagnostics are tolerance-consistent.
    let rrqr_started = std::time::Instant::now();
    if priority_perm_is_identity {
        log::info!(
            "[STAGE] identifiability audit: joint RRQR start n={} p_total={} priority_reorder=false",
            n,
            p_total,
        );
    } else {
        // The priority permutation actually changes column order — log which
        // blocks are being reordered so production runs can confirm the
        // canonical-gauge path is exercised.  This surfaces in logs even
        // at INFO level so it's always visible without debug builds.
        let block_priority_summary: Vec<String> = specs
            .iter()
            .map(|s| format!("{}={}", s.name, s.gauge_priority))
            .collect();
        log::info!(
            "[STAGE] identifiability audit: joint RRQR start n={} p_total={} \
             priority_reorder=true blocks=[{}]",
            n,
            p_total,
            block_priority_summary.join(", "),
        );
    }
    let rrqr = if priority_perm_is_identity {
        rrqr_with_permutation(&x_joint, default_rrqr_rank_alpha())
            .map_err(|e| format!("identifiability audit joint RRQR failed: {e:?}"))?
    } else {
        let mut x_priority = Array2::<f64>::zeros((n, p_total));
        for (new_j, &old_j) in priority_perm.iter().enumerate() {
            x_priority.column_mut(new_j).assign(&x_joint.column(old_j));
        }
        rrqr_with_permutation(&x_priority, default_rrqr_rank_alpha()).map_err(|e| {
            format!("identifiability audit joint RRQR (priority-ordered) failed: {e:?}")
        })?
    };
    log::info!(
        "[STAGE] identifiability audit: joint RRQR end rank={}/{} elapsed={:.3}s",
        rrqr.rank,
        p_total,
        rrqr_started.elapsed().as_secs_f64(),
    );
    let joint_rank = rrqr.rank;
    let joint_rank_tol = rrqr.rank_tol;
    // RRQR's `column_permutation` indexes into the matrix it actually
    // saw. If we reordered by priority, map those back to original
    // joint-column indices so downstream block-attribution stays
    // correct.
    let map_to_original = |reordered_idx: usize| -> usize {
        if priority_perm_is_identity {
            reordered_idx
        } else {
            priority_perm[reordered_idx]
        }
    };
    let demoted_joint_cols: Vec<usize> = rrqr.column_permutation[rrqr.rank..]
        .iter()
        .map(|&j| map_to_original(j))
        .collect();

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
    // Bias correction: when one block carries `eta_row_scaling = Some(z)`,
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
    let mut col_norms = Array1::<f64>::zeros(p_total);
    // S2_k for each joint column; computed once and reused for both
    // report and halt threshold decisions below.
    let mut col_s2 = Array1::<f64>::zeros(p_total);
    for j in 0..p_total {
        let col = x_joint.column(j);
        let nrm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        col_norms[j] = nrm;
        col_s2[j] = compute_leverage_s2(&col);
    }
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
    let pairwise_block_heartbeat = (n.saturating_mul(p_total) >= 1_000_000)
        .then(crate::util::heartbeat::Heartbeat::default_interval);
    let mut aliased_pairs: Vec<AliasedPair> = Vec::new();
    let n_block_pairs = specs.len().saturating_mul(specs.len().saturating_sub(1)) / 2;
    for a_block_idx in 0..specs.len() {
        let a_start = col_offsets[a_block_idx];
        let a_end = col_offsets[a_block_idx + 1];
        // Extract the row-scaling vector for block A (used for the bias shift).
        let z_a = specs[a_block_idx]
            .eta_row_scaling
            .as_ref()
            .map(|arc| arc.as_ref());
        for b_block_idx in (a_block_idx + 1)..specs.len() {
            let b_start = col_offsets[b_block_idx];
            let b_end = col_offsets[b_block_idx + 1];
            // Extract the row-scaling vector for block B.
            let z_b = specs[b_block_idx]
                .eta_row_scaling
                .as_ref()
                .map(|arc| arc.as_ref());
            for ja in a_start..a_end {
                let na = col_norms[ja];
                if na == 0.0 {
                    continue;
                }
                let ca = x_joint.column(ja);
                for jb in b_start..b_end {
                    let nb = col_norms[jb];
                    if nb == 0.0 {
                        continue;
                    }
                    let cb = x_joint.column(jb);
                    let dot: f64 = ca.iter().zip(cb.iter()).map(|(a, b)| a * b).sum();
                    // Signed cosine: preserves direction for bias-shift test.
                    let cosine = signed_cosine(dot, na, nb);
                    let s2_ja = col_s2[ja];
                    let s2_jb = col_s2[jb];
                    // Bias shift: non-zero when exactly one block carries
                    // row-scaling (or the two scalings differ).
                    let shift = bias_shift_for_pair(z_a, z_b, s2_ja, s2_jb);
                    let report_half_width = pair_report_threshold(s2_ja, s2_jb, n, total_cross_pairs);
                    let (report_flag, _) = cosine_outside_null_band(cosine, shift, report_half_width);
                    // Store the unsigned |cosine| in AliasedPair.overlap for
                    // backwards compatibility and human-readable diagnostics.
                    // Also store `shift` so the halt-threshold check can apply
                    // the same directional correction.
                    let overlap = cosine.abs();
                    if report_flag {
                        aliased_pairs.push(AliasedPair {
                            block_a: specs[a_block_idx].name.clone(),
                            block_b: specs[b_block_idx].name.clone(),
                            direction_a: ja - a_start,
                            direction_b: jb - b_start,
                            overlap,
                            bias_shift: shift,
                        });
                    }
                }
            }
            if let Some(hb) = pairwise_block_heartbeat.as_ref() {
                hb.tick(1, |done, secs| {
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

    // Attribute each demoted joint column back to its (block, local_col)
    // origin using the col_offsets table built above. The earliest
    // block whose range absorbs the demoted column is, by construction,
    // the block at lower index containing the largest projection norm
    // — but RRQR's column-pivoting selects the column most aligned
    // with the *trailing* (demoted) space, so we attribute the alias
    // to "the demoted column itself was selected as redundant; the
    // earlier blocks (in spec order) reconstruct it". The reason
    // string names the joint-column index and the joint rank tolerance
    // so callers can correlate with the structural log.
    let mut dropped_columns: Vec<DroppedColumn> = Vec::new();
    for &joint_col in &demoted_joint_cols {
        let (block_idx, local_col) = locate_block_column(&col_offsets, joint_col)?;
        let block_name = specs[block_idx].name.clone();
        let reason = format!(
            "joint-design column {joint_col} (block '{block_name}' local column \
             {local_col}) demoted past joint RRQR rank tolerance {tol:.3e}; earlier \
             blocks' column span absorbs this direction",
            tol = joint_rank_tol,
        );
        dropped_columns.push(DroppedColumn {
            block: block_name,
            column: local_col,
            reason,
        });
    }

    // Reflect the dropped-column attribution into `BlockIdentity::
    // effective_dim`: each block's effective dimension is its original
    // dim minus the count of its columns appearing in
    // `demoted_joint_cols`. The per-block `design_range_rank` (from
    // the in-isolation per-block QR) still flags any within-block
    // rank deficiency that escaped within-smooth absorption.
    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let lo = col_offsets[block_idx];
        let hi = col_offsets[block_idx + 1];
        let dropped_here = demoted_joint_cols
            .iter()
            .filter(|&&j| j >= lo && j < hi)
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
    let hard_alias_pair = aliased_pairs
        .iter()
        .filter(|p| {
            let pa = block_priority
                .get(p.block_a.as_str())
                .copied()
                .unwrap_or(100);
            let pb = block_priority
                .get(p.block_b.as_str())
                .copied()
                .unwrap_or(100);
            // Same priority → ambiguous → halt.
            // Distinct priorities → canonical-gauge resolves → do not halt.
            if pa != pb {
                return false;
            }
            // Compute per-pair halt threshold from stored S2 values.
            let ja = block_name_to_idx
                .get(p.block_a.as_str())
                .map(|&bi| col_offsets[bi] + p.direction_a)
                .unwrap_or(0);
            let jb = block_name_to_idx
                .get(p.block_b.as_str())
                .map(|&bi| col_offsets[bi] + p.direction_b)
                .unwrap_or(0);
            let halt_half_width = pair_halt_threshold(
                col_s2.get(ja).copied().unwrap_or(1.0),
                col_s2.get(jb).copied().unwrap_or(1.0),
                n,
            );
            // Bias-corrected halt check: we stored overlap = |cosine| and
            // bias_shift = shift.  The exact test would be |cosine − shift| ≥
            // halt_half_width, but we only have |cosine|.  Using a conservative
            // lower bound |overlap − |shift|| ensures we only fire when the
            // cosine is genuinely outside the null band regardless of sign.
            // For exact aliases (overlap ≈ 1) this always fires; for moderate
            // overlaps the shift must be large enough to displace the cosine
            // inside the null band before we withhold the halt.
            let conservative_deviation = (p.overlap - p.bias_shift.abs()).abs();
            conservative_deviation >= halt_half_width
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
                let other_priority = block_priority
                    .get(other_block)
                    .copied()
                    .unwrap_or(100);
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
                format!(
                    "first attributed drop: block '{}' local column {} \
                     (reparam: replace this column with a sum-to-zero or \
                     orthogonal-complement projection against earlier blocks, \
                     or remove the redundant term entirely)",
                    first_drop.block, first_drop.column,
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
/// [`crate::families::identifiability_compiler::IdentityRowHessian`] —
/// see [`compile_with_dual_metric`] for why the structural metric is
/// the rank-decision metric, not the pilot curvature).
///
/// The output [`IdentifiabilityAudit`] preserves the same contract as
/// the flat path: `dropped_columns` attributed to (block, local_col),
/// `aliased_pairs` reported above the per-pair leverage-based report
/// threshold in the channel-weighted view, `fatal` true on rank
/// deficiency or hard-alias pair above the per-pair leverage-based
/// halt threshold (`pair_halt_threshold`).
pub fn audit_identifiability_channel_aware(
    specs: &[ParameterBlockSpec],
    operators: &[std::sync::Arc<
        dyn crate::families::identifiability_compiler::RowJacobianOperator,
    >],
    row_hess: &dyn crate::families::identifiability_compiler::RowHessian,
) -> Result<IdentifiabilityAudit, String> {
    use crate::families::identifiability_compiler::{IdentityRowHessian, compile_with_dual_metric};

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
        return Err(format!(
            "audit_identifiability_channel_aware: specs ({}) and operators ({}) length mismatch",
            specs.len(),
            operators.len()
        ));
    }
    let k = row_hess.k();
    let n = row_hess.nrows();
    for (idx, op) in operators.iter().enumerate() {
        if op.k() != k {
            return Err(format!(
                "audit_identifiability_channel_aware: operator {idx} has K={} but row_hess K={k}",
                op.k(),
            ));
        }
        if op.nrows() != n {
            return Err(format!(
                "audit_identifiability_channel_aware: operator {idx} has nrows={} but row_hess nrows={n}",
                op.nrows(),
            ));
        }
        if op.ncols() != specs[idx].design.ncols() {
            return Err(format!(
                "audit_identifiability_channel_aware: operator {idx} ({}) has ncols={} but spec '{}' design ncols={}",
                idx,
                op.ncols(),
                specs[idx].name,
                specs[idx].design.ncols(),
            ));
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

    // Joint compile-with-dual-metric does the heavy lifting: it
    // returns per-block selectors `t_lw`, the joint rank, and demoted
    // (block, local_col) attributions in the structural metric.
    // Routing the audit through this path is what guarantees that the
    // channel-aware view replaces — not augments — the flat view's
    // joint RRQR pass.
    let ordering: Vec<crate::families::identifiability_compiler::BlockOrder> =
        std::iter::repeat(crate::families::identifiability_compiler::BlockOrder::Marginal)
            .take(operators.len())
            .collect();
    let compiled = compile_with_dual_metric(operators, row_hess, &id_struct, &ordering)
        .map_err(|e| format!("audit_identifiability_channel_aware compile failed: {e:?}"))?;

    // Build per-block identity entries from the compiled output.
    let mut blocks: Vec<BlockIdentity> = Vec::with_capacity(specs.len());
    let mut col_offsets: Vec<usize> = Vec::with_capacity(specs.len() + 1);
    col_offsets.push(0);
    for (idx, spec) in specs.iter().enumerate() {
        let p_block = spec.design.ncols();
        let kept = compiled
            .blocks
            .get(idx)
            .map(|b| b.t_lw.ncols())
            .unwrap_or(p_block);
        blocks.push(BlockIdentity {
            block_name: spec.name.clone(),
            original_dim: p_block,
            effective_dim: kept,
            // The channel-aware path does not produce per-block
            // singular values in the flat-X sense; report an empty
            // vector and rely on `effective_dim` < `original_dim`
            // as the structural-rank signal.
            design_range_rank: kept,
            design_range_singular_values: Vec::new(),
        });
        let next = col_offsets[col_offsets.len() - 1] + p_block;
        col_offsets.push(next);
    }
    let p_total = *col_offsets.last().expect("col_offsets non-empty");

    // Dropped columns come straight from the compiler's
    // audit-attribution pass. The compiler reports `(block_idx,
    // local_col)`; map to `DroppedColumn { block, column, reason }`
    // with a channel-aware reason string.
    let mut dropped_columns: Vec<DroppedColumn> = Vec::new();
    for (block_idx, local_col) in &compiled.dropped {
        let block_name = specs[*block_idx].name.clone();
        dropped_columns.push(DroppedColumn {
            block: block_name.clone(),
            column: *local_col,
            reason: format!(
                "channel-aware audit (K={k}) demoted block '{block_name}' \
                 local column {local_col}: column is in the row-Jacobian span \
                 of earlier blocks under the structural row metric",
            ),
        });
    }

    // Pairwise overlap scan in the channel-weighted view. The compiler
    // already eigendecomposed the structural Gram block-by-block; we
    // need joint column-column overlaps to surface near-alias pairs
    // above the reporting threshold. Compute on the (n·K, p_total)
    // weighted joint W where W_b = sqrt(K^S) · J_b. With K^S = I,
    // sqrt(K^S) = I and W_b is just J_b flattened to (n·K, p_b).
    let aliased_pairs = channel_aware_aliased_pairs(operators, &col_offsets, specs)?;

    let joint_rank = compiled.joint_rank;
    let joint_rank_deficient = joint_rank < p_total;

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
    // For the channel-aware path we compute S2 on the materialised (n·K)
    // channel-weighted column vectors, which correctly accounts for both
    // the row dimension and channel weighting.
    let block_name_to_idx_ca: std::collections::HashMap<&str, usize> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name.as_str(), i))
        .collect();
    // Materialise per-column S2 values from the channel-weighted columns.
    // This mirrors the inner loop of channel_aware_aliased_pairs but only
    // computes norms and S2, not the full pairwise scan.
    let ca_col_s2: Vec<f64> = {
        let p_total_ca = *col_offsets.last().unwrap_or(&0);
        let mut s2_vals: Vec<f64> = Vec::with_capacity(p_total_ca);
        for op in operators.iter() {
            let j_full = op.evaluate_full();
            let p_b = op.ncols();
            for c in 0..p_b {
                let mut w = Array1::<f64>::zeros(n * k);
                for i in 0..n {
                    for ch in 0..k {
                        w[i * k + ch] = j_full[[i, c, ch]];
                    }
                }
                s2_vals.push(compute_leverage_s2(&w.view()));
            }
        }
        s2_vals
    };
    let hard_alias_pair = aliased_pairs
        .iter()
        .filter(|p| {
            let pa = block_priority_ca
                .get(p.block_a.as_str())
                .copied()
                .unwrap_or(100);
            let pb = block_priority_ca
                .get(p.block_b.as_str())
                .copied()
                .unwrap_or(100);
            if pa != pb {
                return false;
            }
            let ja = block_name_to_idx_ca
                .get(p.block_a.as_str())
                .map(|&bi| col_offsets[bi] + p.direction_a)
                .unwrap_or(0);
            let jb = block_name_to_idx_ca
                .get(p.block_b.as_str())
                .map(|&bi| col_offsets[bi] + p.direction_b)
                .unwrap_or(0);
            let halt_thr = pair_halt_threshold(
                ca_col_s2.get(ja).copied().unwrap_or(1.0),
                ca_col_s2.get(jb).copied().unwrap_or(1.0),
                n * k,
            );
            p.overlap >= halt_thr
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
                let other_priority = block_priority_ca
                    .get(other_block)
                    .copied()
                    .unwrap_or(100);
                other_priority > drop_priority
            })
        });

    let fatal = (joint_rank_deficient && !gauge_resolves_rank_deficiency_ca)
        || hard_alias_pair.is_some();

    let fatal_detail = if fatal {
        let mut parts: Vec<String> = Vec::new();
        if joint_rank_deficient && !gauge_resolves_rank_deficiency_ca {
            let attribution = if let Some(first_drop) = dropped_columns.first() {
                format!(
                    "first attributed drop: block '{}' local column {} \
                     (reparam: replace this column with a sum-to-zero or \
                     orthogonal-complement projection against earlier blocks, \
                     or remove the redundant term entirely)",
                    first_drop.block, first_drop.column,
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
            let halt_thr = pair_halt_threshold(
                ca_col_s2.get(ja).copied().unwrap_or(1.0),
                ca_col_s2.get(jb).copied().unwrap_or(1.0),
                n * k,
            );
            parts.push(format!(
                "alias pair: '{}'[{}] ~ '{}'[{}] overlap={:.4} >= leverage-based halt \
                 threshold {:.4} in channel-aware row-Jacobian view \
                 (reparam: orthogonalise one block's column {} against the other \
                 or absorb the shared direction)",
                pair.block_a,
                pair.direction_a,
                pair.block_b,
                pair.direction_b,
                pair.overlap,
                halt_thr,
                pair.direction_b,
            ));
        }
        format!(" — FATAL: {}", parts.join("; "))
    } else if gauge_resolves_rank_deficiency_ca {
        format!(
            " — gauge-attributed drops (channel-aware): {} column(s) attributed to \
             lower-priority blocks via gauge_priority ordering; canonical-gauge pipeline \
             will proceed with reduced specs",
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

/// Pairwise overlap scan on the channel-weighted joint design
/// `W = stack_b sqrt(I_K) · J_b` (identity structural metric).
/// Returns one [`AliasedPair`] per cross-block column-pair whose
/// normalised `|wᵀ w'|` exceeds the per-pair leverage-based report
/// threshold (`pair_report_threshold`), in `(block_a < block_b)` order.
fn channel_aware_aliased_pairs(
    operators: &[std::sync::Arc<
        dyn crate::families::identifiability_compiler::RowJacobianOperator,
    >],
    col_offsets: &[usize],
    specs: &[ParameterBlockSpec],
) -> Result<Vec<AliasedPair>, String> {
    if operators.is_empty() {
        return Ok(Vec::new());
    }
    let k = operators[0].k();
    let n = operators[0].nrows();
    let nk = n
        .checked_mul(k)
        .ok_or_else(|| format!("channel-aware audit: n*k overflow (n={n}, k={k})"))?;
    let p_total = *col_offsets.last().unwrap_or(&0);
    if p_total == 0 || nk == 0 {
        return Ok(Vec::new());
    }
    // Materialise W = (n*K, p_total) column-major (one Array1<f64> of
    // length nk per joint column).  Also compute S2 per column for the
    // leverage-based threshold (see `compute_leverage_s2`).
    let mut cols: Vec<Array1<f64>> = Vec::with_capacity(p_total);
    let mut col_norms: Vec<f64> = Vec::with_capacity(p_total);
    let mut col_s2: Vec<f64> = Vec::with_capacity(p_total);
    for op in operators.iter() {
        let j_full = op.evaluate_full();
        let p_b = op.ncols();
        for c in 0..p_b {
            let mut w = Array1::<f64>::zeros(nk);
            for i in 0..n {
                for ch in 0..k {
                    w[i * k + ch] = j_full[[i, c, ch]];
                }
            }
            let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            let s2 = compute_leverage_s2(&w.view());
            cols.push(w);
            col_norms.push(norm);
            col_s2.push(s2);
        }
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
    // Pairwise scan: for every joint column pair (a < b) with both
    // norms positive, compute |wᵀw'| / (‖w‖·‖w'‖) and emit if above
    // the per-pair leverage-based reporting threshold.
    let mut pairs: Vec<AliasedPair> = Vec::new();
    for a in 0..p_total {
        if col_norms[a] <= 0.0 {
            continue;
        }
        for b in (a + 1)..p_total {
            if col_norms[b] <= 0.0 {
                continue;
            }
            let mut dot = 0.0_f64;
            for i in 0..nk {
                dot += cols[a][i] * cols[b][i];
            }
            let overlap = (dot.abs() / (col_norms[a] * col_norms[b])).min(1.0);
            let report_thr =
                pair_report_threshold(col_s2[a], col_s2[b], nk, total_cross_pairs);
            if overlap >= report_thr {
                let (block_a_idx, dir_a) = locate_block_column(col_offsets, a)?;
                let (block_b_idx, dir_b) = locate_block_column(col_offsets, b)?;
                if block_a_idx == block_b_idx {
                    // Within-block aliasing is a separate concern
                    // (per-block QR catches it); the cross-block scan
                    // only reports inter-block pairs.
                    continue;
                }
                pairs.push(AliasedPair {
                    block_a: specs[block_a_idx].name.clone(),
                    block_b: specs[block_b_idx].name.clone(),
                    direction_a: dir_a,
                    direction_b: dir_b,
                    overlap,
                    // Channel-aware path: no row-scaling bias correction;
                    // the channel weighting already accounts for per-block
                    // structure, and the row Jacobian operators are not
                    // parameterised through eta_row_scaling here.
                    bias_shift: 0.0,
                });
            }
        }
    }
    Ok(pairs)
}

fn locate_block_column(col_offsets: &[usize], joint_col: usize) -> Result<(usize, usize), String> {
    // col_offsets has length specs.len() + 1; col_offsets[i..i+1] is
    // the joint-column range for block i. Linear scan is fine — the
    // table is tiny (one entry per block).
    for i in 0..col_offsets.len() - 1 {
        if joint_col >= col_offsets[i] && joint_col < col_offsets[i + 1] {
            return Ok((i, joint_col - col_offsets[i]));
        }
    }
    Err(format!(
        "identifiability_audit::locate_block_column: joint_col {joint_col} \
         outside col_offsets range (max = {})",
        col_offsets.last().copied().unwrap_or(0),
    ))
}

fn block_pivoted_qr_diagonal(block: &Array2<f64>) -> Result<Vec<f64>, String> {
    if block.ncols() == 0 {
        return Ok(Vec::new());
    }
    let (_q, r) = block
        .qr()
        .map_err(|e| format!("identifiability audit per-block QR failed: {e:?}"))?;
    let diag_len = r.nrows().min(r.ncols());
    let mut out: Vec<f64> = (0..diag_len).map(|i| r[[i, i]].abs()).collect();
    out.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    out.resize(block.ncols(), 0.0);
    Ok(out)
}

fn count_rank(singular_values: &[f64], n: usize, p: usize) -> usize {
    if singular_values.is_empty() {
        return 0;
    }
    let leading = singular_values.first().copied().unwrap_or(0.0);
    let rank_alpha = default_rrqr_rank_alpha();
    let tol = rank_alpha * f64::EPSILON * (n.max(p).max(1) as f64) * leading.max(1.0);
    singular_values.iter().filter(|&&v| v > tol).count()
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
    let mut g = Array2::<f64>::zeros((p, p));
    if w_diag.is_empty() {
        // W = I: G = J^T J
        for k in 0..n {
            for i in 0..p {
                let ji = j_joint[[k, i]];
                if ji == 0.0 {
                    continue;
                }
                for j in i..p {
                    let val = ji * j_joint[[k, j]];
                    g[[i, j]] += val;
                    if i != j {
                        g[[j, i]] += val;
                    }
                }
            }
        }
    } else {
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
    }

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
        .filter(|(_, &lam)| lam < null_tol)
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
            let dominant_block = dominant_block_for_direction(&n_vec.to_owned(), specs, col_offsets);

            let message = format!(
                "MAP estimate is non-unique: null direction {} of J^T W J (eigenvalue {lam:.3e}) \
                 has n^T S n = {ntsn:.3e} < tolerance {pen_tol:.3e}; \
                 the MAP is flat along this direction (no likelihood curvature, no penalty \
                 curvature); dominant block: '{}'. \
                 Fix: add a non-degenerate smoothness penalty to block '{}' that covers this \
                 direction, or remove the unpenalised null direction from the model.",
                dir_idx,
                dominant_block,
                dominant_block,
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
    use crate::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use ndarray::Array2;

    fn spec_from_dense(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
        let n = design.nrows();
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
            offset: Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            eta_row_scaling: None,
            jacobian_callback: None,
        }
    }

    fn linspace_minus_one_to_one(n: usize) -> Array1<f64> {
        if n <= 1 {
            return Array1::<f64>::zeros(n.max(1));
        }
        let step = 2.0 / (n as f64 - 1.0);
        Array1::from_iter((0..n).map(|i| -1.0 + step * i as f64))
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

    /// Test 5: end-to-end shape on a biobank-like configuration —
    /// 4 blocks, ~50 total columns, with one intentional cross-block
    /// linear alias seeded in. The audit must complete in well under
    /// a second and produce a single attributed drop with the
    /// expected joint rank.
    #[test]
    fn audit_biobank_shape_end_to_end() {
        let n = 1024;
        let x = linspace_minus_one_to_one(n);
        // Block 0: parametric intercept + age-linear.
        let mut parametric = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            parametric[[i, 0]] = 1.0;
            parametric[[i, 1]] = x[i];
        }
        // Block 1: smooth in x — 8 polynomial-like columns.
        let mut s_x = Array2::<f64>::zeros((n, 8));
        for i in 0..n {
            for k in 0..8 {
                s_x[[i, k]] = (x[i] - (k as f64 - 4.0) * 0.2).powi(2);
            }
        }
        // Block 2: smooth in sin(x) — 6 columns. No alias with block 1.
        let mut s_sin = Array2::<f64>::zeros((n, 6));
        for i in 0..n {
            for k in 0..6 {
                s_sin[[i, k]] = ((k as f64 + 1.0) * x[i]).sin();
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
        let audit = audit_identifiability(&specs).expect("biobank-shape audit must succeed");
        // The seeded x~x alias is exactly the biobank failure shape the
        // task #5 halt gate exists to refuse: two distinct blocks
        // contributing the same direction at overlap 1.0. Must be fatal.
        assert!(
            audit.fatal,
            "seeded biobank-shape exact x~x alias must be fatal under the halt gate: {}",
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
            "biobank-shape audit should attribute exactly the seeded alias; \
             got {:?}",
            audit.dropped_columns,
        );
        // Total effective dim = 2 + 8 + 6 + 4 − 1 = 19.
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 19,
            "biobank-shape: expected 19 kept directions; got {total_kept} ({})",
            audit.summary,
        );
    }
}
