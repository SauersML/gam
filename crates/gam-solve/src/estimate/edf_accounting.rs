//! One accounting for the penalized effective-degrees-of-freedom bundle.
//!
//! Every fitting route computes the per-block penalty traces
//! `tr_k = λ_k·tr(H⁻¹ S_k)` with whatever linear algebra its parameterisation
//! affords — a factorized solve against the canonical transformed blocks, a
//! dense product against a latent covariance, a Cholesky solve against an
//! assembled Hessian. That part is genuinely route-specific and stays where it
//! is. What is *not* route-specific is the **accounting** those traces feed:
//! which ceiling a trace is admitted against, and within what band of its solve
//! (#2901), what `edf_by_block` is measured against, and what floor `edf_total`
//! may not fall below.
//!
//! Those four rules were written out independently at six sites and did not
//! agree (issue #2470). Two disagreements were load-bearing:
//!
//! * **The per-block ceiling.** `rank(S_k)` and `block_cols` are not the same
//!   number; they differ by `nullity(S_k)`, which is a whole integer of reported
//!   complexity for every penalized block. `rank(S_k)` is the correct one, and
//!   not merely by convention: it is the quantity the REML criterion already
//!   prices as `rank(S_k)·ρ_k`, so it is the ceiling that agrees with the
//!   objective being optimized. Passing the ranks in explicitly is deliberate —
//!   a caller must *state* its rank oracle rather than reach for a column count
//!   because that is what happened to be in scope.
//! * **The floor.** `edf_total` cannot fall below the joint penalty null-space
//!   dimension `mp = p − rank(Σ_k S_k)`: those directions are unpenalized, so no
//!   amount of smoothing removes them. Clamping to `[0, p]` instead lets a noisy
//!   trace report an effective dimension below the mathematically attainable
//!   minimum, and nothing downstream notices.
//!
//! `edf_total` feeds `σ̂² = RSS/(n − edf_total)`, conditional AIC, the
//! likelihood-ratio reference df and every interval width, so a disagreement
//! here is not a reporting curiosity.
//!
//! Both ends of `[0, rank_k]` are theorems only under a premise (#2901).
//! `tr_k ≤ rank_k` holds when `H ⪰ λ_k S̃_k`, which a nonnegative data curvature
//! guarantees and an observed Hessian of an arbitrary likelihood does not: the
//! tilted double well of #2366 traces 2.945 against a rank of 1 at a certified
//! mode. `tr_k ≥ 0` holds when `H ≻ 0`, which a converged mode need not have
//! either: a mode on a feasible cone is only copositive there, and the Gaussian
//! location-scale wiggle fit of #2635 keeps its mode with an indefinite ambient
//! precision. A certified `H ⪰ λ_k S̃_k` gives `H ⪰ 0`, and `H` is nonsingular
//! because its solve produced the trace, so it carries both premises. So each
//! block carries an [`EdfRankBound`], and only a certified block is admitted
//! against, and clamped to, `[0, rank_k]`.
//!
//! Because the floor `mp` is stated here, this is also the one place that can
//! see `edf_total` LAND on it — a fit that kept none of the penalized
//! directions its design offered and returned the model no amount of smoothing
//! can remove. #2607 and #2579 both produced exactly that, by unrelated routes,
//! and both reported `Converged` with nothing said. `collapsed_to_penalty_null_space`
//! names the state by its outcome, so one check covers every route into it.

use gam_linalg::faer_ndarray::{FaerLblt, SymmetricInertia};
use gam_runtime::resource::MemoryGovernor;
use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

/// Why a penalty block's trace is confined to `[0, rank_k]` (#2901).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EdfRankCertificate {
    /// Every working weight defining `H` is nonnegative, so `XᵀWX ⪰ 0` and
    /// `H ⪰ λ_k S̃_k` for every block without a factorization.
    Structural,
    /// The `LDLᵀ` factorization of `H − λ_k S̃_k`, shifted by its rounding band, has no
    /// negative pivot, so `λ_min(H − λ_k S̃_k) ≥ −band` ([`numerical_rank_bound`],
    /// [`sparse_numerical_rank_bound`]). `smallest_pivot` is the smallest eigenvalue of
    /// the factor's pivot blocks.
    Numerical { smallest_pivot: f64, band: f64 },
}

/// The rank-bound status published beside a block's trace (#2901), exactly one per
/// block.
///
/// `H ⪰ λ_k S̃_k` is sufficient for `tr_k ≤ rank_k`, not necessary. A published
/// label for `Uncertified` says "rank bound not certified", never "exceeds rank", and
/// one for `NotAssessed` says "rank bound not assessed".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EdfRankBound {
    Certified(EdfRankCertificate),
    /// No certificate that `tr_k ≤ rank_k`: the `LDLᵀ` factorization of
    /// `H − λ_k S̃_k`, shifted by its rounding band, has a negative pivot, so by
    /// Sylvester's law of inertia `H − λ_k S̃_k` has an eigenvalue below that band and
    /// the sufficient condition fails. The trace may still lie in `[0, rank_k]`; it is
    /// published unclamped and is not a violation. Nor is a trace below zero: without
    /// `H ⪰ λ_k S̃_k` nothing certifies `H ≻ 0` either. Its `edf_by_block` entry is
    /// `rank_k − tr_k` unclamped, and no clamp applies to a total containing it.
    Uncertified { smallest_pivot: f64, band: f64 },
    /// The memory governor refused the certificate's factorization workspace, so the
    /// rank bound was not assessed. `reason` is the refusal, naming the requested bytes
    /// and the budget. The trace publishes unclamped, as an `Uncertified` one does, and
    /// the fit does not fail. A block whose structural certificate applies never
    /// requests the workspace.
    NotAssessed { reason: String },
}

impl EdfRankBound {
    /// Whether the block's trace is certified to lie in `[0, rank_k]`.
    pub fn is_certified(&self) -> bool {
        matches!(self, Self::Certified(_))
    }
}

/// The `p × p` copies the dense certificate holds live: the shifted difference, the
/// Bunch–Kaufman factor's copy of its lower triangle, and the factorization's workspace.
const EDF_RANK_CERTIFICATE_COPIES: usize = 3;

/// Certify `H ⪰ λ_k S̃_k` for one block from the inertia of `H − λ_k S̃_k` shifted by
/// its rounding band, factored by [`FaerLblt`] (#2901).
///
/// `scaled_penalty_block` is `λ_k S̃_k` on the block it penalizes, whose first
/// coordinate is `block_start`. The quadratic form of the difference sees only its
/// symmetric part, which is what is formed. Forming it charges each entry
/// `2u·(|H_ij| + |λ_k S̃_k,ij|)`, and the spectral norm of that charge is at most its
/// Frobenius norm `f`. The shift `s = f + p·ε·‖Â‖_F` lifts a zero eigenvalue of a
/// positive semidefinite difference clear of the factorization's rounding, so such a
/// block certifies where a Cholesky factorization of the difference refuses. With `e`
/// the [`gam_math::roundoff::symmetric_inertia_band`] of the factors, no negative
/// pivot certifies `λ_min(H − λ_k S̃_k) ≥ −band` with `band = s + f + e`.
///
/// The factorization's copies are reserved on `governor` first, and a refused
/// reservation publishes [`EdfRankBound::NotAssessed`] with the refusal.
pub fn numerical_rank_bound(
    hessian: ArrayView2<'_, f64>,
    scaled_penalty_block: ArrayView2<'_, f64>,
    block_start: usize,
    governor: &MemoryGovernor,
) -> Result<EdfRankBound, super::EstimationError> {
    let p = hessian.nrows();
    let block = scaled_penalty_block.nrows();
    if hessian.ncols() != p || scaled_penalty_block.ncols() != block || block_start + block > p {
        return Err(super::EstimationError::InvalidInput(format!(
            "EDF rank certificate: a {}x{} Hessian against a {}x{} penalty block at {block_start}",
            hessian.nrows(),
            hessian.ncols(),
            scaled_penalty_block.nrows(),
            scaled_penalty_block.ncols()
        )));
    }
    if p == 0 {
        return Ok(EdfRankBound::Certified(EdfRankCertificate::Numerical {
            smallest_pivot: 0.0,
            band: 0.0,
        }));
    }
    let reservation = match governor.try_reserve_dense_f64_copies(
        p,
        p,
        EDF_RANK_CERTIFICATE_COPIES,
        "EDF rank certificate",
    ) {
        Ok(reservation) => reservation,
        Err(refusal) => {
            return Ok(EdfRankBound::NotAssessed {
                reason: refusal.to_string(),
            });
        }
    };
    let span = block_start..block_start + block;
    let penalty = |i: usize, j: usize| -> f64 {
        if span.contains(&i) && span.contains(&j) {
            scaled_penalty_block[[i - block_start, j - block_start]]
        } else {
            0.0
        }
    };
    let mut shifted = faer::Mat::<f64>::zeros(p, p);
    let mut formation_sq = 0.0_f64;
    let mut difference_sq = 0.0_f64;
    for j in 0..p {
        for i in j..p {
            let value =
                0.5 * ((hessian[[i, j]] - penalty(i, j)) + (hessian[[j, i]] - penalty(j, i)));
            shifted[(i, j)] = value;
            let charge = 2.0
                * gam_linalg::roundoff::UNIT_ROUNDOFF
                * (hessian[[i, j]].abs() + penalty(i, j).abs());
            let multiplicity = if i == j { 1.0 } else { 2.0 };
            formation_sq += multiplicity * charge * charge;
            difference_sq += multiplicity * value * value;
        }
    }
    let formation = formation_sq.sqrt();
    let shift = formation + p as f64 * f64::EPSILON * difference_sq.sqrt();
    let mut shifted_sq = 0.0_f64;
    for j in 0..p {
        shifted[(j, j)] += shift;
        for i in j..p {
            let multiplicity = if i == j { 1.0 } else { 2.0 };
            shifted_sq += multiplicity * shifted[(i, j)] * shifted[(i, j)];
        }
    }
    let inertia = FaerLblt::new(shifted.as_ref(), faer::Side::Lower).inertia();
    drop(reservation);
    if !inertia.smallest_pivot.is_finite() {
        return Err(super::EstimationError::InvalidInput(
            "EDF rank certificate: the LDLᵀ pivots of H − λS are not finite".to_string(),
        ));
    }
    let band = shift
        + formation
        + gam_math::roundoff::symmetric_inertia_band(
            p,
            shifted_sq.sqrt(),
            inertia.factor_magnitude,
        );
    Ok(rank_bound_from_inertia(&inertia, band))
}

/// No negative pivot certifies the block; a negative one does not.
fn rank_bound_from_inertia(inertia: &SymmetricInertia, band: f64) -> EdfRankBound {
    if inertia.negative == 0 {
        EdfRankBound::Certified(EdfRankCertificate::Numerical {
            smallest_pivot: inertia.smallest_pivot,
            band,
        })
    } else {
        EdfRankBound::Uncertified {
            smallest_pivot: inertia.smallest_pivot,
            band,
        }
    }
}

/// [`numerical_rank_bound`] for a sparse `H` stored as its upper triangle, factored by
/// [`gam_linalg::sparse_exact::sparse_symmetric_inertia`] on the sparse Cholesky's own
/// ordering and symbolic analysis (#2901).
///
/// The shifted difference keeps `H`'s sparsity plus the penalty block. Its shift and
/// band are the dense certificate's, with the factor term `‖L̂‖_F²·max|D̂|`. The
/// workspace reserved on `governor` is the factor's values and the matrix's canonical
/// and permuted copies, sized by the symbolic analysis. The simplicial `LDLᵀ` does not
/// pivot, so a zero pivot stops it before its inertia is known, and the block
/// publishes `Uncertified` with a smallest pivot of zero.
pub fn sparse_numerical_rank_bound(
    hessian_upper: &faer::sparse::SparseColMat<usize, f64>,
    scaled_penalty_block: ArrayView2<'_, f64>,
    block_start: usize,
    governor: &MemoryGovernor,
) -> Result<EdfRankBound, super::EstimationError> {
    let p = hessian_upper.nrows();
    let block = scaled_penalty_block.nrows();
    if hessian_upper.ncols() != p
        || scaled_penalty_block.ncols() != block
        || block_start + block > p
    {
        return Err(super::EstimationError::InvalidInput(format!(
            "EDF rank certificate: a sparse {}x{} Hessian against a {}x{} penalty block at \
             {block_start}",
            hessian_upper.nrows(),
            hessian_upper.ncols(),
            scaled_penalty_block.nrows(),
            scaled_penalty_block.ncols()
        )));
    }
    if p == 0 {
        return Ok(EdfRankBound::Certified(EdfRankCertificate::Numerical {
            smallest_pivot: 0.0,
            band: 0.0,
        }));
    }
    // Upper-triangle entries keyed `(col, row)`, so iteration is column-major and
    // sorted by row: each holds the difference and the absolute sum it was formed from.
    let mut entries = std::collections::BTreeMap::<(usize, usize), (f64, f64)>::new();
    let (symbolic, values) = hessian_upper.parts();
    for col in 0..p {
        for idx in symbolic.col_ptr()[col]..symbolic.col_ptr()[col + 1] {
            let row = symbolic.row_idx()[idx];
            let entry = entries.entry((row.max(col), row.min(col))).or_insert((0.0, 0.0));
            entry.0 += values[idx];
            entry.1 += values[idx].abs();
        }
    }
    for b in 0..block {
        for a in 0..=b {
            let value = 0.5 * (scaled_penalty_block[[a, b]] + scaled_penalty_block[[b, a]]);
            let entry = entries
                .entry((block_start + b, block_start + a))
                .or_insert((0.0, 0.0));
            entry.0 -= value;
            entry.1 += scaled_penalty_block[[a, b]].abs();
        }
    }
    for j in 0..p {
        entries.entry((j, j)).or_insert((0.0, 0.0));
    }
    let mut formation_sq = 0.0_f64;
    let mut difference_sq = 0.0_f64;
    for (&(col, row), &(value, absolute)) in &entries {
        let multiplicity = if row == col { 1.0 } else { 2.0 };
        let charge = 2.0 * gam_linalg::roundoff::UNIT_ROUNDOFF * absolute;
        formation_sq += multiplicity * charge * charge;
        difference_sq += multiplicity * value * value;
    }
    let formation = formation_sq.sqrt();
    let shift = formation + p as f64 * f64::EPSILON * difference_sq.sqrt();
    let mut shifted_sq = 0.0_f64;
    let mut triplets = Vec::with_capacity(entries.len());
    for (&(col, row), &(value, _)) in &entries {
        let value = if row == col { value + shift } else { value };
        let multiplicity = if row == col { 1.0 } else { 2.0 };
        shifted_sq += multiplicity * value * value;
        triplets.push(faer::sparse::Triplet::new(row, col, value));
    }
    let shifted = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(p, p, &triplets)
        .map_err(|error| {
            super::EstimationError::InvalidInput(format!(
                "EDF rank certificate: assembling the sparse shifted difference failed: {error:?}"
            ))
        })?;
    let factor_nnz = gam_linalg::sparse_exact::sparse_spd_factor_nnz(&shifted).map_err(|error| {
        super::EstimationError::InvalidInput(format!(
            "EDF rank certificate: sparse symbolic analysis failed: {error}"
        ))
    })?;
    let entry_bytes = std::mem::size_of::<f64>() + std::mem::size_of::<usize>();
    let workspace_bytes = factor_nnz
        .checked_add(entries.len().saturating_mul(2))
        .and_then(|count| count.checked_mul(entry_bytes))
        .unwrap_or(usize::MAX);
    let reservation = match governor.try_reserve(workspace_bytes, "EDF rank certificate") {
        Ok(reservation) => reservation,
        Err(refusal) => {
            return Ok(EdfRankBound::NotAssessed {
                reason: refusal.to_string(),
            });
        }
    };
    let inertia = gam_linalg::sparse_exact::sparse_symmetric_inertia(&shifted).map_err(|error| {
        super::EstimationError::InvalidInput(format!(
            "EDF rank certificate: sparse LDLᵀ of H − λS failed: {error}"
        ))
    })?;
    drop(reservation);
    Ok(match inertia {
        Some(inertia) => {
            let band = shift
                + formation
                + gam_math::roundoff::symmetric_inertia_band(
                    p,
                    shifted_sq.sqrt(),
                    inertia.factor_magnitude,
                );
            rank_bound_from_inertia(&inertia, band)
        }
        None => EdfRankBound::Uncertified {
            smallest_pivot: 0.0,
            band: shift + formation,
        },
    })
}

/// The three EDF quantities a fit publishes, produced together so they cannot
/// disagree with one another, and the rank-bound status each block was admitted
/// under.
#[derive(Clone, Debug, PartialEq)]
pub struct EdfBundle {
    /// `p − Σ_k tr_k`, clamped to `[mp, p]` when every block is certified, and
    /// unclamped otherwise.
    pub edf_total: f64,
    /// `rank_k − tr_k` per penalty block, clamped to `[0, rank_k]` for a certified
    /// block and unclamped for any other.
    pub edf_by_block: Vec<f64>,
    /// The admitted per-block traces `tr_k`: a certified block's clamped to
    /// `[0, rank_k]`, any other block's raw. Retained because the per-term
    /// EDF decomposition is assembled from them (issue #1219), so downstream must
    /// read the same numbers this accounting used rather than re-clamping the raw
    /// values itself.
    pub penalty_block_trace: Vec<f64>,
    /// Each block's rank-bound status, aligned 1:1 with `penalty_block_trace`.
    pub rank_bound: Vec<EdfRankBound>,
}

/// Assemble the EDF bundle from per-block traces, each admitted within the
/// rounding band of the solve that produced it (#2901).
///
/// - **Non-finite.** A non-finite trace, including the `+∞` of an overflowing
///   product, carries no value and is refused on every block.
/// - **Certified.** A block certified by its [`EdfRankBound`] has `H ⪰ λ_k S̃_k ⪰ 0`
///   with `H` nonsingular, so both `tr_k ≥ 0` and `tr_k ≤ rank_k` are theorems. It
///   is refused below `−band` or above `rank_k + band`: the Hessian and this penalty
///   were not one operator. On `y ~ s(x) + s(x, g, bs='fs')` a raw fs trace of
///   6.09e4 against a rank of 22 became `edf = 7.322` under the old clamp, where the
///   operator's own value is 9.309. An admitted certified trace is clamped to
///   `[0, rank_k]`.
/// - **Not certified.** An `Uncertified` or `NotAssessed` block publishes its raw
///   trace, with no clamp and no refusal, below zero too. The Gaussian
///   location-scale wiggle fit of #2635 converges on its feasible cone with an
///   indefinite ambient precision and traces −2.66e-4 on its location block; a
///   refusal there deleted the converged fit.
///
/// `trace_bands`, `rank_bounds` and `block_ranks` are aligned 1:1 with the traces.
/// Each band comes from [`gam_linalg::roundoff::solved_penalty_trace_band`] on the
/// solve that formed its trace. `coefficient_count` is `p`. `joint_penalty_nullity`
/// is `mp = p − rank(Σ_k S_k)`, taken as a parameter rather than derived from
/// `block_ranks`: the joint rank is the rank of the *stacked* penalty root, which is
/// not in general the sum of the per-block ranks. Traces are summed with compensated
/// (Kahan) addition because `edf_total` is a difference of two like-sized
/// quantities, where naive summation error lands directly in the reported effective
/// dimension.
pub fn penalized_edf_bundle_within_bands(
    raw_block_traces: &[f64],
    trace_bands: &[f64],
    rank_bounds: &[EdfRankBound],
    block_ranks: &[usize],
    coefficient_count: usize,
    joint_penalty_nullity: f64,
) -> Result<EdfBundle, super::EstimationError> {
    assert_blocks_aligned(raw_block_traces.len(), block_ranks.len());
    assert_blocks_aligned(trace_bands.len(), block_ranks.len());
    assert_blocks_aligned(rank_bounds.len(), block_ranks.len());
    let mut penalty_block_trace = Vec::with_capacity(raw_block_traces.len());
    for (block, (((&raw, &band), bound), &rank)) in raw_block_traces
        .iter()
        .zip(trace_bands.iter())
        .zip(rank_bounds.iter())
        .zip(block_ranks.iter())
        .enumerate()
    {
        let ceiling = rank as f64;
        let refusal = super::EstimationError::EdfTraceOutsideRank {
            block,
            trace: raw,
            rank,
            band,
        };
        if !(raw.is_finite() && band.is_finite()) {
            return Err(refusal);
        }
        if bound.is_certified() {
            if raw < -band || raw > ceiling + band {
                return Err(refusal);
            }
            penalty_block_trace.push(raw.clamp(0.0, ceiling));
        } else {
            penalty_block_trace.push(raw);
        }
    }
    Ok(assemble_bundle(
        penalty_block_trace,
        rank_bounds,
        block_ranks,
        coefficient_count,
        joint_penalty_nullity,
    ))
}

/// The accounting once every trace is admitted. A block's clamp applies only when
/// it is certified, and the `[mp, p]` clamp of the total only when every block is.
fn assemble_bundle(
    penalty_block_trace: Vec<f64>,
    rank_bounds: &[EdfRankBound],
    block_ranks: &[usize],
    coefficient_count: usize,
    joint_penalty_nullity: f64,
) -> EdfBundle {
    let edf_by_block: Vec<f64> = penalty_block_trace
        .iter()
        .zip(rank_bounds.iter())
        .zip(block_ranks.iter())
        .map(|((&trace, bound), &rank)| {
            let ceiling = rank as f64;
            if bound.is_certified() {
                (ceiling - trace).clamp(0.0, ceiling)
            } else {
                ceiling - trace
            }
        })
        .collect();
    let p = coefficient_count as f64;
    let raw_total = p - super::penalty::kahan_sum(penalty_block_trace.iter().copied());
    let every_block_certified = rank_bounds.iter().all(EdfRankBound::is_certified);
    let edf_total = if every_block_certified {
        raw_total.clamp(joint_penalty_nullity.min(p), p)
    } else {
        raw_total
    };
    if every_block_certified
        && collapsed_to_penalty_null_space(edf_total, coefficient_count, joint_penalty_nullity)
    {
        let mp = joint_penalty_nullity.clamp(0.0, p);
        log::warn!(
            "fit collapsed to its penalty null space: effective df {edf_total:.3} of {p} \
             coefficients, against a joint penalty nullity of {mp}. The {} penalized \
             directions this design offered were smoothed away entirely -- the model \
             returned is the one no amount of smoothing can remove. On a saturated or \
             near-saturated design that is the criterion's own optimum rather than a \
             solver failure; otherwise it is the signature of a lambda railed at its \
             ceiling (#2607).",
            p - mp,
        );
    }
    EdfBundle {
        edf_total,
        edf_by_block,
        penalty_block_trace,
        rank_bound: rank_bounds.to_vec(),
    }
}

/// How many of the penalized directions a design offered survived into the fit,
/// and whether that count is small enough to call the result the penalty's own
/// null model.
///
/// `mp = p − rank(Σ_k S_k)` is the dimension smoothing cannot touch, so
/// `attainable = p − mp` is what λ-selection actually chooses over and
/// `spent = edf_total − mp` is what it kept. #2607 and #2579 reached the SAME
/// visible end state — `Converged`, `edf` within rounding of the intercept —
/// from two unrelated mechanisms (a saturated design whose REML optimum is
/// maximum smoothing; a df floor that iterated an emptied penalty list). Naming
/// the state by its *outcome* rather than by either route is what makes one
/// check cover both, and any third route that lands here.
///
/// The two bounds are what separate a collapse from an ordinary answer:
///
/// * `spent <= 0.5` — less than half of ONE direction retained out of every
///   direction on offer. `hifreq_tensor_k10` measured `edf = 1.294` against
///   `mp = 1`, i.e. `spent = 0.294` of an attainable 575.
/// * `attainable >= 20` — a single smooth term legitimately shrinks to its own
///   null space whenever the truth really is linear, and a `k = 10` marginal
///   offers only ~8 penalized directions, so firing there would report a
///   correct model selection as a defect. Twenty is the point past which a
///   design has been given substantial flexibility and kept none of it; it is a
///   threshold on how much was discarded, not on how well the fit did.
///
/// Returns `false` for any non-finite input rather than warning about arithmetic
/// that has already failed somewhere upstream.
pub fn collapsed_to_penalty_null_space(
    edf_total: f64,
    coefficient_count: usize,
    joint_penalty_nullity: f64,
) -> bool {
    /// Penalized directions a design must offer before keeping none of them is
    /// reported as a collapse rather than as a linear truth being found.
    const MIN_ATTAINABLE_DIRECTIONS: f64 = 20.0;
    /// Retained penalized df, below which the fit IS its penalty null model.
    const MAX_RETAINED_DF: f64 = 0.5;
    if !edf_total.is_finite() || !joint_penalty_nullity.is_finite() {
        return false;
    }
    let p = coefficient_count as f64;
    let mp = joint_penalty_nullity.clamp(0.0, p);
    let attainable = p - mp;
    let spent = edf_total - mp;
    attainable >= MIN_ATTAINABLE_DIRECTIONS && spent <= MAX_RETAINED_DF
}

/// Length agreement between the traces and their ranks is a caller contract, not
/// a runtime condition to recover from: a mismatch means the caller paired the
/// wrong penalty blocks, and silently zipping to the shorter of the two would
/// drop a block's complexity from `edf_total` without a word.
fn assert_blocks_aligned(traces: usize, ranks: usize) {
    assert_eq!(
        traces, ranks,
        "penalized_edf_bundle_within_bands: {traces} traces against {ranks} block ranks; \
         they are aligned 1:1 with the penalty blocks"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structural(blocks: usize) -> Vec<EdfRankBound> {
        vec![EdfRankBound::Certified(EdfRankCertificate::Structural); blocks]
    }

    fn uncertified() -> EdfRankBound {
        EdfRankBound::Uncertified {
            smallest_pivot: -1.0,
            band: 1.0e-15,
        }
    }

    #[test]
    fn a_trace_is_admitted_against_its_block_rank_not_its_column_count() {
        // A rank-2 penalty on a 5-column block: a trace at 2 saturates the
        // block, and the reported block EDF is measured against 2. Using the
        // column count as the ceiling would report `5 - 2 = 3` here instead of
        // `0`, which is exactly the nullity(S_k) = 3 overstatement this
        // accounting exists to remove.
        let bundle =
            penalized_edf_bundle_within_bands(&[2.0], &[0.0], &structural(1), &[2], 5, 3.0)
                .unwrap();
        assert_eq!(bundle.penalty_block_trace, vec![2.0]);
        assert_eq!(bundle.edf_by_block, vec![0.0]);
    }

    /// #2901: a ceiling-λ redundant block that overflows its product to `+∞` has no
    /// value to publish. The old accounting saturated it at the rank; it is refused
    /// by name, certified or not.
    #[test]
    fn a_positive_overflow_trace_is_refused_instead_of_saturated_2901() {
        for bound in [structural(1).remove(0), uncertified()] {
            let refusal =
                penalized_edf_bundle_within_bands(&[f64::INFINITY], &[0.0], &[bound], &[3], 6, 3.0)
                    .unwrap_err();
            assert!(
                matches!(
                    refusal,
                    super::super::EstimationError::EdfTraceOutsideRank { block: 0, rank: 3, .. }
                ),
                "{refusal}"
            );
        }
    }

    /// #2901: NaN carries no value and `−∞` points the way a trace of a positive
    /// definite `H` cannot go. Both are refused, where the old accounting propagated
    /// them as NaN for a later validator.
    #[test]
    fn a_nan_or_negative_overflow_trace_is_refused_2901() {
        for raw in [f64::NAN, f64::NEG_INFINITY] {
            for bound in [structural(1).remove(0), uncertified()] {
                let refusal =
                    penalized_edf_bundle_within_bands(&[raw], &[0.0], &[bound], &[3], 6, 3.0)
                        .unwrap_err();
                assert!(
                    matches!(
                        refusal,
                        super::super::EstimationError::EdfTraceOutsideRank { block: 0, rank: 3, .. }
                    ),
                    "raw trace {raw}: {refusal}"
                );
            }
        }
    }

    /// #2901: the fs seed-0 fit's per-block traces (probe 1147204 state C). The raw
    /// rotated roots give the fs block 6.09e4 against a rank of 22. A certified block
    /// that far above its rank is refused by name, where the clamp published
    /// `edf = 7.322`. The projected blocks give 20.013 inside the interval, and
    /// publish `39 − Σ = 9.309`.
    #[test]
    fn a_trace_outside_its_rank_beyond_its_band_refuses_and_the_projected_one_publishes_2901() {
        let ranks = [10, 1, 22, 3, 3];
        let bands = [1.0e-9; 5];
        let raw = [4.472975313998, 0.2178209581733, 6.088685603962e4, 2.147964574147, 2.839105270806];
        let refusal =
            penalized_edf_bundle_within_bands(&raw, &bands, &structural(5), &ranks, 39, 3.0)
                .unwrap_err();
        assert!(
            matches!(
                refusal,
                super::super::EstimationError::EdfTraceOutsideRank { block: 2, rank: 22, .. }
            ),
            "{refusal}"
        );
        let projected = [4.472975313998, 0.2178209581733, 2.001294300186e1, 2.147963642435, 2.839078039171];
        let bundle =
            penalized_edf_bundle_within_bands(&projected, &bands, &structural(5), &ranks, 39, 3.0)
                .unwrap();
        assert!(
            (bundle.edf_total - 9.309).abs() <= 5.0e-4,
            "the projected blocks publish edf {}",
            bundle.edf_total
        );
        assert_eq!(bundle.penalty_block_trace, projected.to_vec());
    }

    /// #2901: a railed block whose solve is conditioned near 1e15 can compute a
    /// trace a little above its rank. Its band is then wide, and the trace
    /// publishes at the rank: vacuous, but honest. An overflowed `+∞` trace and a
    /// NaN trace carry no value and are refused.
    #[test]
    fn a_railed_trace_within_its_band_publishes_and_a_non_finite_trace_refuses_2901() {
        let railed =
            penalized_edf_bundle_within_bands(&[22.3], &[189.0], &structural(1), &[22], 39, 3.0)
                .unwrap();
        assert_eq!(railed.penalty_block_trace, vec![22.0]);
        for raw in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let refusal =
                penalized_edf_bundle_within_bands(&[raw], &[189.0], &structural(1), &[22], 39, 3.0)
                    .unwrap_err();
            assert!(
                matches!(
                    refusal,
                    super::super::EstimationError::EdfTraceOutsideRank { block: 0, rank: 22, .. }
                ),
                "{refusal}"
            );
        }
    }

    /// A trace a little below zero, within its solve's band, is rounding and is
    /// admitted at zero. Below `−band` on a certified block, whose `H ⪰ λS ⪰ 0` makes
    /// the trace nonnegative, the Hessian and the penalty are not one operator, and it
    /// is refused.
    #[test]
    fn a_negative_trace_is_admitted_at_zero() {
        let bundle =
            penalized_edf_bundle_within_bands(&[-0.25], &[0.25], &structural(1), &[4], 4, 0.0)
                .unwrap();
        assert_eq!(bundle.penalty_block_trace, vec![0.0]);
        assert_eq!(bundle.edf_by_block, vec![4.0]);
        let refusal =
            penalized_edf_bundle_within_bands(&[-0.25], &[0.125], &structural(1), &[4], 4, 0.0)
                .unwrap_err();
        assert!(
            matches!(
                refusal,
                super::super::EstimationError::EdfTraceOutsideRank { block: 0, rank: 4, .. }
            ),
            "{refusal}"
        );
    }

    /// #2901: the Gaussian location-scale wiggle fit of #2635 converges on its feasible
    /// cone with an indefinite ambient precision, and its location block traced
    /// −2.656741e-4 against a band of 1.5616e-13 (job 1223783). Nothing certifies
    /// `H ≻ 0` on an uncertified block, so the trace publishes raw with
    /// `edf_by_block = rank − tr` unclamped, where refusing it deleted the converged fit.
    #[test]
    fn an_uncertified_block_publishes_a_trace_below_its_band_2901() {
        let bounds = [structural(1).remove(0), uncertified()];
        let bundle = penalized_edf_bundle_within_bands(
            &[1.5, -2.656741e-4],
            &[0.0, 1.5616e-13],
            &bounds,
            &[2, 3],
            6,
            1.0,
        )
        .unwrap();
        assert_eq!(bundle.penalty_block_trace, vec![1.5, -2.656741e-4]);
        assert_eq!(bundle.edf_by_block, vec![0.5, 3.0 + 2.656741e-4]);
        let total = 6.0 - (1.5 - 2.656741e-4);
        assert!(
            (bundle.edf_total - total).abs() <= 4.0 * f64::EPSILON * 6.0,
            "edf_total {} against the unclamped {total}",
            bundle.edf_total
        );
        assert_eq!(bundle.rank_bound, bounds.to_vec());
    }

    #[test]
    fn edf_total_cannot_fall_below_the_joint_penalty_null_space() {
        // p = 10 with mp = 3 unpenalized directions. Even a fully saturated
        // certified penalty cannot remove them, so the floor is 3, not 0. A `[0, p]`
        // clamp would report 0 here — an effective dimension below the
        // mathematically attainable minimum, with nothing downstream to notice.
        let bundle =
            penalized_edf_bundle_within_bands(&[7.0], &[0.0], &structural(1), &[7], 10, 3.0)
                .unwrap();
        assert_eq!(bundle.edf_total, 3.0);
        // #2901: the floor is a consequence of every block being rank-bounded. A
        // block whose data curvature is indefinite can absorb more than its rank,
        // and the total it publishes is `p − Σ tr` unclamped.
        let unbounded =
            penalized_edf_bundle_within_bands(&[9.0], &[0.0], &[uncertified()], &[7], 10, 3.0)
                .unwrap();
        assert_eq!(unbounded.edf_total, 1.0);
    }

    /// #2901: an `Uncertified` block publishes its raw trace and an unclamped
    /// `rank − trace`, and the total containing it is not clamped either. A certified
    /// block beside it keeps its clamps.
    #[test]
    fn an_uncertified_block_publishes_its_raw_trace_unclamped_2901() {
        let bounds = [structural(1).remove(0), uncertified()];
        let bundle =
            penalized_edf_bundle_within_bands(&[1.5, 2.945], &[0.0, 1.0e-15], &bounds, &[2, 1], 4, 1.0)
                .unwrap();
        assert_eq!(bundle.penalty_block_trace, vec![1.5, 2.945]);
        assert_eq!(bundle.edf_by_block, vec![0.5, 1.0 - 2.945]);
        assert_eq!(bundle.edf_total, 4.0 - (1.5 + 2.945));
        assert_eq!(bundle.rank_bound, bounds.to_vec());
    }

    #[test]
    fn edf_total_is_p_minus_the_admitted_traces_when_interior() {
        let bundle = penalized_edf_bundle_within_bands(
            &[1.5, 2.25],
            &[0.0, 0.0],
            &structural(2),
            &[4, 5],
            12,
            3.0,
        )
        .unwrap();
        assert_eq!(bundle.penalty_block_trace, vec![1.5, 2.25]);
        assert_eq!(bundle.edf_by_block, vec![2.5, 2.75]);
        assert_eq!(bundle.edf_total, 12.0 - 3.75);
    }

    #[test]
    fn an_unpenalized_fit_reports_every_coefficient() {
        let bundle = penalized_edf_bundle_within_bands(&[], &[], &[], &[], 6, 6.0).unwrap();
        assert_eq!(bundle.edf_total, 6.0);
        assert!(bundle.edf_by_block.is_empty());
        assert!(bundle.penalty_block_trace.is_empty());
    }

    #[test]
    #[should_panic(expected = "aligned 1:1 with the penalty blocks")]
    fn mismatched_traces_and_ranks_are_refused_not_zipped_short() {
        penalized_edf_bundle_within_bands(&[1.0, 2.0], &[0.0, 0.0], &structural(2), &[3], 5, 0.0)
            .expect("the alignment assertion panics before any trace is admitted");
    }

    /// #2901: a block whose certificate workspace the governor refused publishes its raw
    /// trace and an unclamped `rank − trace`, as an `Uncertified` block does.
    #[test]
    fn a_block_not_assessed_publishes_its_raw_trace_unclamped_2901() {
        let bounds = [EdfRankBound::NotAssessed {
            reason: "EDF rank certificate: cannot reserve 48 bytes".to_string(),
        }];
        let bundle =
            penalized_edf_bundle_within_bands(&[2.945], &[1.0e-15], &bounds, &[1], 3, 2.0)
                .unwrap();
        assert_eq!(bundle.penalty_block_trace, vec![2.945]);
        assert_eq!(bundle.edf_by_block, vec![1.0 - 2.945]);
        assert_eq!(bundle.edf_total, 3.0 - 2.945);
        assert_eq!(bundle.rank_bound, bounds.to_vec());
    }

    /// #2901: no negative pivot of `H − λS` shifted by its band certifies the block, and
    /// a negative pivot does not. `[[5, 1], [1, 3]] − diag(4, 1)` is positive definite.
    /// The #2366 double well at its mode, `H = 1` against `λS = 2.945`, is not, and
    /// neither is `[[1, 1], [1, 1]] − I`, whose factorization takes a 2×2 pivot.
    #[test]
    fn the_numerical_certificate_reads_the_inertia_of_the_shifted_difference_2901() {
        let certified = numerical_rank_bound(
            ndarray::array![[5.0, 1.0], [1.0, 3.0]].view(),
            ndarray::array![[4.0, 0.0], [0.0, 1.0]].view(),
            0,
            MemoryGovernor::global(),
        )
        .unwrap();
        match certified {
            EdfRankBound::Certified(EdfRankCertificate::Numerical {
                smallest_pivot,
                band,
            }) => {
                assert!(smallest_pivot > 0.0, "{smallest_pivot}");
                assert!(band > 0.0 && band <= 1.0e-12, "{band:e}");
            }
            other => panic!("a positive definite difference must certify: {other:?}"),
        }
        for (hessian, penalty) in [
            (ndarray::array![[1.0]], ndarray::array![[2.945]]),
            (
                ndarray::array![[1.0, 1.0], [1.0, 1.0]],
                ndarray::array![[1.0, 0.0], [0.0, 1.0]],
            ),
        ] {
            match numerical_rank_bound(hessian.view(), penalty.view(), 0, MemoryGovernor::global())
                .unwrap()
            {
                EdfRankBound::Uncertified {
                    smallest_pivot,
                    band,
                } => {
                    assert!(smallest_pivot < -band, "{smallest_pivot} against {band:e}");
                }
                other => panic!("an indefinite difference must not certify: {other:?}"),
            }
        }
    }

    /// #2901: a positive semidefinite difference with a zero eigenvalue satisfies
    /// `H ⪰ λS` and certifies. `[[2, 1], [1, 2]] − I = [[1, 1], [1, 1]]` has eigenvalues
    /// 0 and 2, and Cholesky refuses its zero pivot. This is the positive control that
    /// the certificate is not a Cholesky attempt.
    #[test]
    fn a_semidefinite_difference_with_a_zero_eigenvalue_certifies_where_cholesky_refuses_2901() {
        let hessian = ndarray::array![[2.0, 1.0], [1.0, 2.0]];
        let penalty = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let difference =
            faer::Mat::<f64>::from_fn(2, 2, |i, j| hessian[[i, j]] - penalty[[i, j]]);
        assert!(
            gam_linalg::faer_ndarray::FaerLlt::new(difference.as_ref(), faer::Side::Lower)
                .is_err(),
            "Cholesky refuses the zero eigenvalue of [[1, 1], [1, 1]]"
        );
        match numerical_rank_bound(hessian.view(), penalty.view(), 0, MemoryGovernor::global())
            .unwrap()
        {
            EdfRankBound::Certified(EdfRankCertificate::Numerical { smallest_pivot, .. }) => {
                assert!(smallest_pivot >= 0.0, "{smallest_pivot}");
            }
            other => panic!("a semidefinite difference must certify: {other:?}"),
        }
    }

    /// #2901: the penalty block is embedded at its first coordinate, and the sparse
    /// certificate decides what the dense one decides. `H = diag(5, 5, 1)` against
    /// `λS = [[4]]` certifies at coordinate 0 and not at coordinate 2.
    #[test]
    fn the_sparse_certificate_agrees_with_the_dense_one_on_the_embedded_block_2901() {
        let hessian = ndarray::Array2::from_diag(&ndarray::array![5.0, 5.0, 1.0]);
        let sparse_hessian = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
            3,
            3,
            &[
                faer::sparse::Triplet::new(0, 0, 5.0),
                faer::sparse::Triplet::new(1, 1, 5.0),
                faer::sparse::Triplet::new(2, 2, 1.0),
            ],
        )
        .expect("a diagonal sparse Hessian");
        let penalty = ndarray::array![[4.0]];
        for (start, certified) in [(0, true), (2, false)] {
            let dense =
                numerical_rank_bound(hessian.view(), penalty.view(), start, MemoryGovernor::global())
                    .unwrap();
            let sparse = sparse_numerical_rank_bound(
                &sparse_hessian,
                penalty.view(),
                start,
                MemoryGovernor::global(),
            )
            .unwrap();
            assert_eq!(dense.is_certified(), certified, "{dense:?}");
            assert_eq!(sparse.is_certified(), certified, "{sparse:?}");
        }
    }

    #[test]
    fn the_state_2607_recorded_is_detected_by_its_outcome() {
        // The numbers `hifreq_tensor_k10` reported while it was saturated:
        // `edf = 1.294` of `p = 576`, joint penalty nullity 1 (the intercept —
        // a te() double penalty leaves nothing else unpenalized). Every one of
        // the 575 penalized directions was smoothed away, the fit reported
        // `Converged`, and nothing said so.
        assert!(collapsed_to_penalty_null_space(1.294, 576, 1.0));
        // The same fixture BEFORE the collapse, from the history #2585 records:
        // `edf = 227.938` of the same 576. Same design, same nullity — only the
        // outcome differs, which is the whole point of naming the state by its
        // outcome.
        assert!(!collapsed_to_penalty_null_space(227.938, 576, 1.0));
    }

    #[test]
    fn a_linear_truth_under_one_smooth_is_a_selection_not_a_collapse() {
        // A `k = 10` marginal offers ~8 penalized directions on top of a
        // 2-dimensional null space. When the truth really is linear, REML
        // shrinking that smooth to exactly its null space is the CORRECT
        // answer, and reporting it as a collapse would turn a good model
        // selection into a warning on a large fraction of honest fits.
        assert!(!collapsed_to_penalty_null_space(2.0, 10, 2.0));
        // Widen the same shape past the point where keeping nothing stops being
        // an ordinary selection, holding `spent` fixed at zero: the bound is on
        // how much was discarded, so this and the case above must disagree.
        assert!(collapsed_to_penalty_null_space(2.0, 30, 2.0));
    }

    #[test]
    fn an_unpenalized_design_can_never_collapse() {
        // `mp = p` means λ selects over nothing at all: `edf_total` is pinned to
        // `p` by construction, so there is no collapse available to report and a
        // predicate keyed on `spent` alone would fire on every such fit.
        assert!(!collapsed_to_penalty_null_space(40.0, 40, 40.0));
    }

    #[test]
    fn a_non_finite_edf_is_not_reported_as_a_collapse() {
        // NaN compares false against every bound, so `spent <= 0.5` would be
        // false for NaN but true for −inf. Neither is a statement about
        // smoothing: the arithmetic already failed upstream, and the finiteness
        // validators own that.
        for edf in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert!(
                !collapsed_to_penalty_null_space(edf, 576, 1.0),
                "non-finite edf {edf} is an upstream arithmetic failure, not a collapse"
            );
        }
        assert!(!collapsed_to_penalty_null_space(1.294, 576, f64::NAN));
    }
}
