//! Analytic ZERO-smoothing rail face certificate (#2348 Inc 5, lower face).
//!
//! [`rail_face`](super::rail_face) proves an infinite-smoothing face from its
//! exact `λ = ∞` limit. This module is the opposite end: a set `L` of
//! coordinates railed at `λ_j → 0`, where the penalty LEAVES the model instead
//! of pinning it.
//!
//! # What happens to the criterion as `λ_L → 0`
//!
//! Write the profiled-Gaussian REML criterion as
//!
//! ```text
//!     V(λ) = (n − M_p)/2 · log D_p + ½ log|H| − ½ log|S_λ|₊ + prior,
//!     H = XᵀWX + S_R + S_L,   S_R = Σ_{k∉L} λ_k S_k,   S_L = Σ_{j∈L} λ_j S_j.
//! ```
//!
//! Four cases, and only the first is a face:
//!
//! 1. **Covered** — `range(S_j) ⊆ range(S_R)` for every `j ∈ L`, and
//!    `H₀ = XᵀWX + S_R ≻ 0`, and the limit fit leaves a residual. Then no rank
//!    changes as `λ_L → 0`: `log|S_λ|₊ = log|Uᵀ S_λ U|` on the fixed range `U` of
//!    `S_R`, `log|H|` is analytic at `H₀`, `D_p` is analytic at `D_p⁰ > 0`, and
//!    `M_p` is constant. So `V` is jointly ANALYTIC in `λ_L` on a neighbourhood
//!    of the closed orthant's corner, and
//!
//!    ```text
//!        V(λ_L) = V(0) + Σ_{j∈L} c′_j λ_j + O(|λ_L|²),
//!        c′_j = (n − M_p)/(2D_p⁰)·β̂₀ᵀS_jβ̂₀ + ½tr(H₀⁻¹S_j) − ½tr(S_R⁺S_j) + prior_j,
//!    ```
//!
//!    `β̂₀ = H₀⁻¹XᵀWy` the zero-smoothing fit (the envelope theorem removes
//!    `∂β̂/∂λ` from `D_p`). The first order is EXACTLY LINEAR, so — unlike the
//!    `λ = ∞` face, whose released ranges can overlap — the orthant test is the
//!    per-axis KKT test: the face is a strict local minimizer iff every
//!    `c′_j > 0`. Read the terms as the empirical-Bayes trade: the fit term is
//!    the price of shrinking `β̂₀` along `S_j`, `½tr(H₀⁻¹S_j) − ½tr(S_R⁺S_j) ≤ 0`
//!    (because `H₀ ⪰ S_R` on the covered range) is the Occam gain of adding
//!    the penalty. `c′_j > 0` says the data would rather keep `S_j` out.
//! 2. **Uncovered** — some `S_j` reaches outside `range(S_R)`. Then
//!    `log|S_λ|₊` gains `r′·log λ` and `V → +∞`: the corner is a BARRIER, and
//!    a coordinate railed there is on the wrong rail. Refused.
//! 3. **Exact fit** — `D_p⁰ = 0`: `V → −∞`, the criterion has no minimum on
//!    this face at all. Refused.
//! 4. **Not identified** — `H₀` is singular: the zero-smoothing model has no
//!    fit to expand around. Refused.
//!
//! The ρ-prior enters only when its cost is exactly `rate·λ_j`
//! ([`gam_spec::RhoPrior::lower_tail_linear_rate`]); any other prior's
//! ρ-gradient survives into the tail and there is no face to certify.

use super::rail_face::{basis_columns, range_null_split, symmetric_eigh, split_face_penalties};
use super::rail_face::RailFaceLimitOutcome;
use gam_linalg::roundoff::accumulation_growth;
use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// The zero-smoothing face data: the exact first-order law off `λ_L = 0` and
/// the rounding band of each slope.
#[derive(Clone, Debug)]
pub struct ZeroSmoothingFace {
    /// The ρ-coordinates on the face, ascending.
    pub face: Vec<usize>,
    /// `ρ_j` at the certified point, in `face` order. Only prices the value
    /// gap; the slopes are properties of the `λ_L = 0` limit.
    pub face_rho: Vec<f64>,
    /// `c′_j = ∂V/∂λ_j` at `λ_L = 0`, in `face` order.
    pub slopes: Vec<f64>,
    /// Rigorous rounding band `τ_j` of each `c′_j`, in `face` order.
    pub slope_bands: Vec<f64>,
    /// The zero-smoothing fit `β̂₀ = H₀⁻¹XᵀWy`.
    pub limit_beta: Array1<f64>,
    /// Profiled dispersion `D_p⁰/(n − M_p)` at the limit fit.
    pub limit_dispersion: f64,
    /// `‖β̂(ρ̂) − β̂₀‖ = ‖(H₀ + S_L)⁻¹S_Lβ̂₀‖`, exact at the certified `λ_L`.
    pub estimand_travel: f64,
}

/// Why a zero-smoothing face law is or is not available. The same two
/// declines as [`RailFaceLimitOutcome`], for the same reason: "outside this
/// closed form" and "this face is not a face" call for different responses.
#[derive(Clone, Debug)]
pub enum ZeroSmoothingFaceOutcome {
    /// The law was formed.
    Available(Box<ZeroSmoothingFace>),
    /// The criterion is outside the closed form. Says nothing about the face.
    OutsideClosedForm {
        /// Which clause of the form's scope failed.
        reason: String,
    },
    /// The closed form applies, but `λ_L = 0` is not a face of the criterion
    /// (uncovered, exact fit, unidentified) or a check did not hold.
    FaceUnavailable {
        /// The measured evidence behind the refusal.
        reason: String,
    },
}

impl ZeroSmoothingFaceOutcome {
    /// The face law, when one was formed.
    pub fn available(self) -> Option<ZeroSmoothingFace> {
        match self {
            Self::Available(face) => Some(*face),
            _ => None,
        }
    }
}

impl From<RailFaceLimitOutcome> for ZeroSmoothingFaceOutcome {
    fn from(outcome: RailFaceLimitOutcome) -> Self {
        match outcome {
            RailFaceLimitOutcome::OutsideClosedForm { reason } => {
                Self::OutsideClosedForm { reason }
            }
            RailFaceLimitOutcome::FaceUnavailable { reason } => Self::FaceUnavailable { reason },
            RailFaceLimitOutcome::Available(_) => Self::FaceUnavailable {
                reason: "the face split returned a lambda = infinity limit".to_string(),
            },
        }
    }
}

/// A proven zero-smoothing face.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ZeroSmoothingProof {
    /// The binding (smallest-margin) coordinate's slope `c′_j`.
    pub statistic: f64,
    /// That slope's rounding band `τ_j`.
    pub band: f64,
    /// `V(ρ̂) − V(0) = Σ_j c′_j λ_j` to first order.
    pub value_gap: f64,
    /// `‖β̂(ρ̂) − β̂₀‖`.
    pub estimand_travel: f64,
}

/// Prove (or refuse) a zero-smoothing face from its first-order law: every
/// `c′_j` must clear its own band. The binding coordinate is the one with the
/// smallest margin `c′_j − τ_j`.
pub(crate) fn certify_zero_smoothing_face(
    face: &ZeroSmoothingFace,
) -> Result<ZeroSmoothingProof, String> {
    let m = face.face.len();
    if m == 0 || face.slopes.len() != m || face.slope_bands.len() != m || face.face_rho.len() != m
    {
        return Err("the zero-smoothing face law is shape-inconsistent".to_string());
    }
    let mut binding = 0usize;
    for j in 0..m {
        let (slope, band) = (face.slopes[j], face.slope_bands[j]);
        if !slope.is_finite() || !band.is_finite() || band < 0.0 {
            return Err(format!(
                "coordinate {} carries a non-finite slope {slope:.3e} or band {band:.3e}",
                face.face[j]
            ));
        }
        if slope - band < face.slopes[binding] - face.slope_bands[binding] {
            binding = j;
        }
    }
    let (statistic, band) = (face.slopes[binding], face.slope_bands[binding]);
    if !(statistic > band) {
        return Err(format!(
            "the zero-smoothing face is not proven a minimizer: coordinate {} has slope \
             c'={statistic:.6e} against its rounding band {band:.3e}",
            face.face[binding]
        ));
    }
    let value_gap = face
        .slopes
        .iter()
        .zip(face.face_rho.iter())
        .map(|(&slope, &rho)| slope * rho.exp())
        .sum();
    Ok(ZeroSmoothingProof {
        statistic,
        band,
        value_gap,
        estimand_travel: face.estimand_travel,
    })
}

fn frobenius(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `½tr(M⁻¹A)` on `M`'s own eigenpairs, `M ≻ 0`.
fn half_trace_on_spectrum(values: &Array1<f64>, vectors: &Array2<f64>, a: &Array2<f64>) -> f64 {
    let mut total = 0.0_f64;
    for (col, &sigma) in values.iter().enumerate() {
        let v = vectors.column(col);
        total += v.dot(&a.dot(&v)) / sigma;
    }
    0.5 * total
}

/// `M⁻¹b` on `M`'s own eigenpairs, `M ≻ 0`.
fn solve_on_spectrum(values: &Array1<f64>, vectors: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let rotated = vectors.t().dot(b);
    let scaled: Array1<f64> = rotated
        .iter()
        .zip(values.iter())
        .map(|(r, s)| r / s)
        .collect();
    vectors.dot(&scaled)
}

/// Build the covered zero-smoothing face law of the profiled-Gaussian REML
/// criterion from a model's parts.
///
/// `response` must already be net of any offset; `penalties` are in ρ-block
/// order and `face` indexes that order. `prior_rates[i]` is the exact
/// `λ`-slope of the ρ-prior on `face[i]`
/// ([`gam_spec::RhoPrior::lower_tail_linear_rate`]). A decline is typed,
/// never an error.
pub(crate) fn gaussian_zero_smoothing_face(
    design: ArrayView2<'_, f64>,
    response: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    penalties: &[CanonicalPenalty],
    rho: &Array1<f64>,
    face: &[usize],
    prior_rates: &[f64],
) -> ZeroSmoothingFaceOutcome {
    let p = design.ncols();
    let n = design.nrows();
    if p == 0 || n == 0 || response.len() != n || weights.len() != n {
        return ZeroSmoothingFaceOutcome::OutsideClosedForm {
            reason: "zero-smoothing face inputs are shape-inconsistent or empty".to_string(),
        };
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return ZeroSmoothingFaceOutcome::OutsideClosedForm {
            reason: "prior weights are not finite and non-negative".to_string(),
        };
    }
    if prior_rates.len() != face.len() || prior_rates.iter().any(|r| !r.is_finite() || *r < 0.0) {
        return ZeroSmoothingFaceOutcome::OutsideClosedForm {
            reason: "the rho-prior slopes do not match the face".to_string(),
        };
    }
    let split = match split_face_penalties(penalties, rho, face, p) {
        Ok(split) => split,
        Err(outcome) => return outcome.into(),
    };
    let face_prior: Vec<f64> = split
        .face_sorted
        .iter()
        .map(|j| {
            face.iter()
                .position(|f| f == j)
                .map_or(f64::NAN, |slot| prior_rates[slot])
        })
        .collect();
    let face_norm = frobenius(&split.s_face_unit);
    if !(face_norm > 0.0) {
        return ZeroSmoothingFaceOutcome::FaceUnavailable {
            reason: "the face's penalties are zero: lambda = 0 there says nothing about the model"
                .to_string(),
        };
    }

    // ── coverage: every face penalty inside the survivors' range ────────
    // Built directly from the survivors (not `S_all − S_F`) so the rank split
    // sees no cancellation.
    let mut s_survivor_unit = Array2::<f64>::zeros((p, p));
    for (j, penalty) in penalties.iter().enumerate() {
        if split.face_sorted.contains(&j) {
            continue;
        }
        let cols = penalty.col_range.clone();
        for (li, gi) in cols.clone().enumerate() {
            for (lj, gj) in cols.clone().enumerate() {
                s_survivor_unit[[gi, gj]] += penalty.local[[li, lj]];
            }
        }
    }
    let (survivor_values, survivor_vectors) = match symmetric_eigh(&s_survivor_unit) {
        Some(pair) => pair,
        None => {
            return ZeroSmoothingFaceOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the survivor geometry failed"
                    .to_string(),
            };
        }
    };
    let survivor_cut = range_null_split(&survivor_values);
    let range_cols: Vec<usize> = (0..p)
        .filter(|&i| survivor_values[i] > survivor_cut)
        .collect();
    let null_cols: Vec<usize> = (0..p)
        .filter(|&i| survivor_values[i] <= survivor_cut)
        .collect();
    let null_basis = basis_columns(&survivor_vectors, &null_cols);
    let leak = frobenius(&null_basis.t().dot(&split.s_face_unit).dot(&null_basis));
    if leak > f64::EPSILON.sqrt() * face_norm {
        return ZeroSmoothingFaceOutcome::FaceUnavailable {
            reason: format!(
                "the face penalties reach outside the surviving penalties' range (leak \
                 {:.3e} of their norm): log|S|+ loses rank as lambda -> 0, so V -> +infinity \
                 there — a barrier, not a minimizer",
                leak / face_norm
            ),
        };
    }
    let range_basis = basis_columns(&survivor_vectors, &range_cols);

    // ── the zero-smoothing model: H₀ = XᵀWX + S_R and its fit ───────────
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
    let h0 = &xtwx + &split.s_rest;
    let (h_values, h_vectors) = match symmetric_eigh(&h0) {
        Some(pair) => pair,
        None => {
            return ZeroSmoothingFaceOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the zero-smoothing Hessian failed"
                    .to_string(),
            };
        }
    };
    let h_largest = h_values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let h_smallest = h_values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
    if !(h_smallest > f64::EPSILON.sqrt() * h_largest) || !(h_largest > 0.0) {
        return ZeroSmoothingFaceOutcome::FaceUnavailable {
            reason: format!(
                "the lambda = 0 model is not identified: XᵀWX + S_R spans \
                 {h_smallest:.3e}..{h_largest:.3e}"
            ),
        };
    }
    let limit_beta = solve_on_spectrum(&h_values, &h_vectors, &xtwy);

    // ── the profiled deviance and its degrees of freedom ────────────────
    let fitted = design.dot(&limit_beta);
    let residual: Array1<f64> = &response - &fitted;
    let weighted_rss: f64 = (0..n).map(|i| weights[i] * residual[i] * residual[i]).sum();
    let penalty_energy = limit_beta.dot(&split.s_rest.dot(&limit_beta));
    let deviance = weighted_rss + penalty_energy;
    // Same joint structural rank as the λ=∞ form; it does not move on a
    // covered face, which is the whole point of coverage.
    let criterion_penalty_rank = match gam_terms::construction::balanced_penalty_structural_rank(
        penalties
            .iter()
            .map(|penalty| (penalty.local_ref().view(), penalty.col_range.clone())),
        p,
    ) {
        Ok(rank) => rank,
        Err(error) => {
            return ZeroSmoothingFaceOutcome::FaceUnavailable {
                reason: format!("the criterion's joint penalty rank is unavailable: {error}"),
            };
        }
    };
    let null_dim = p.saturating_sub(criterion_penalty_rank);
    if n <= null_dim {
        return ZeroSmoothingFaceOutcome::FaceUnavailable {
            reason: format!("no residual degrees of freedom at the limit: n={n} <= M_p={null_dim}"),
        };
    }
    let half_dof = 0.5 * (n - null_dim) as f64;

    // ── rounding of the fit and the deviance ────────────────────────────
    // `γ` is Wilkinson's accumulation growth for the longest inner product;
    // the coefficient error is the backward-stable solve's forward error,
    // amplified by `H₀`'s conditioning (Frobenius bounds the spectral norm).
    let gamma = accumulation_growth(n.max(p));
    let h_frobenius = frobenius(&h0);
    let h_conditioning = h_frobenius / h_smallest;
    let beta_norm = limit_beta.dot(&limit_beta).sqrt();
    let beta_error = gamma * (1.0 + h_conditioning) * beta_norm;
    let beta_abs = limit_beta.mapv(f64::abs);
    let s_rest_abs = split.s_rest.mapv(f64::abs);
    let mut residual_error = 0.0_f64;
    let mut rss_abs = 0.0_f64;
    for i in 0..n {
        let row_abs: f64 = design
            .row(i)
            .iter()
            .zip(beta_abs.iter())
            .map(|(x, b)| x.abs() * b)
            .sum();
        let rounding = gamma * (response[i].abs() + row_abs);
        residual_error += weights[i] * (2.0 * residual[i].abs() * rounding + rounding * rounding);
        rss_abs += weights[i] * residual[i] * residual[i];
    }
    // `D_p` is stationary at `β̂₀`, so a coefficient error enters it at
    // second order, through `H₀`.
    let deviance_error = h_frobenius * beta_error * beta_error
        + residual_error
        + gamma * rss_abs
        + gamma * beta_abs.dot(&s_rest_abs.dot(&beta_abs));
    if !(deviance > 0.0) || !(deviance_error < deviance) || !deviance_error.is_finite() {
        return ZeroSmoothingFaceOutcome::FaceUnavailable {
            reason: format!(
                "the lambda = 0 fit is exact: profiled deviance {deviance:.3e} does not clear \
                 its rounding {deviance_error:.3e}, so V -> -infinity on this face"
            ),
        };
    }
    let dispersion = deviance / (2.0 * half_dof);

    // ── the survivors' pseudo-inverse on their fixed range ──────────────
    let survivor_block = range_basis.t().dot(&split.s_rest).dot(&range_basis);
    let (block_values, block_vectors) = match symmetric_eigh(&survivor_block) {
        Some(pair) => pair,
        None => {
            return ZeroSmoothingFaceOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the survivor block failed".to_string(),
            };
        }
    };
    let block_smallest = block_values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
    if !(block_smallest > 0.0) {
        return ZeroSmoothingFaceOutcome::FaceUnavailable {
            reason: format!(
                "the surviving penalties are not positive on their own range: smallest \
                 eigenvalue {block_smallest:.3e}"
            ),
        };
    }
    let block_conditioning = frobenius(&survivor_block) / block_smallest;

    // ── the slopes and their bands ──────────────────────────────────────
    let mut slopes = Vec::with_capacity(split.face_sorted.len());
    let mut slope_bands = Vec::with_capacity(split.face_sorted.len());
    for (slot, s_j) in split.face_penalties.iter().enumerate() {
        let energy = limit_beta.dot(&s_j.dot(&limit_beta));
        let fit_term = half_dof * energy / deviance;
        let hessian_trace = half_trace_on_spectrum(&h_values, &h_vectors, s_j);
        let reduced = range_basis.t().dot(s_j).dot(&range_basis);
        let survivor_trace = half_trace_on_spectrum(&block_values, &block_vectors, &reduced);
        let prior = face_prior[slot];
        let slope = fit_term + hessian_trace - survivor_trace + prior;

        let s_norm = frobenius(s_j);
        let energy_error = 2.0 * (energy.max(0.0) * s_norm).sqrt() * beta_error
            + s_norm * beta_error * beta_error
            + gamma * beta_abs.dot(&s_j.mapv(f64::abs).dot(&beta_abs));
        let ratio = deviance_error / deviance;
        let fit_band =
            half_dof * (energy_error / deviance + energy.abs() * deviance_error / (deviance * deviance))
                / (1.0 - ratio);
        let hessian_band = gamma * (1.0 + h_conditioning) * hessian_trace.abs();
        let survivor_band = gamma * (1.0 + block_conditioning) * survivor_trace.abs();
        let sum_band =
            gamma * (fit_term.abs() + hessian_trace.abs() + survivor_trace.abs() + prior.abs());
        slopes.push(slope);
        slope_bands.push(fit_band + hessian_band + survivor_band + sum_band);
    }
    if slopes.iter().chain(slope_bands.iter()).any(|v| !v.is_finite()) {
        return ZeroSmoothingFaceOutcome::FaceUnavailable {
            reason: "the zero-smoothing slopes or their bands are not finite".to_string(),
        };
    }

    // ── the shipped fit's exact distance from the limit ─────────────────
    // `β̂(λ_L) − β̂₀ = −(H₀ + S_L)⁻¹ S_L β̂₀` exactly, `S_L` at the certified λ.
    let mut s_face_weighted = Array2::<f64>::zeros((p, p));
    for (&j, s_j) in split.face_sorted.iter().zip(split.face_penalties.iter()) {
        s_face_weighted.scaled_add(rho[j].exp(), s_j);
    }
    let shipped_hessian = &h0 + &s_face_weighted;
    let estimand_travel = match symmetric_eigh(&shipped_hessian) {
        Some((values, vectors)) if values.iter().all(|v| *v > 0.0) => {
            let offset =
                solve_on_spectrum(&values, &vectors, &s_face_weighted.dot(&limit_beta));
            offset.dot(&offset).sqrt()
        }
        _ => {
            return ZeroSmoothingFaceOutcome::FaceUnavailable {
                reason: "the shipped Hessian is not positive definite".to_string(),
            };
        }
    };

    let face_rho: Vec<f64> = split.face_sorted.iter().map(|&j| rho[j]).collect();
    ZeroSmoothingFaceOutcome::Available(Box::new(ZeroSmoothingFace {
        face: split.face_sorted,
        face_rho,
        slopes,
        slope_bands,
        limit_beta,
        limit_dispersion: dispersion,
        estimand_travel,
    }))
}

#[cfg(test)]
mod zero_smoothing_face_tests {
    use super::*;

    fn law(slopes: Vec<f64>, bands: Vec<f64>) -> ZeroSmoothingFace {
        let m = slopes.len();
        ZeroSmoothingFace {
            face: (0..m).collect(),
            face_rho: vec![-10.0; m],
            slopes,
            slope_bands: bands,
            limit_beta: Array1::zeros(0),
            limit_dispersion: 1.0,
            estimand_travel: 0.0,
        }
    }

    /// The first order is linear, so the orthant test is per axis and the
    /// binding coordinate is the one with the smallest margin, not the
    /// smallest slope.
    #[test]
    fn the_binding_coordinate_is_the_smallest_margin() {
        let proof = certify_zero_smoothing_face(&law(vec![3.0, 2.0], vec![2.5, 0.1]))
            .expect("both slopes clear their bands");
        assert_eq!((proof.statistic, proof.band), (3.0, 2.5));
        let expected_gap = 5.0 * (-10.0_f64).exp();
        assert!((proof.value_gap - expected_gap).abs() <= 1.0e-15 * expected_gap);
        let refused = certify_zero_smoothing_face(&law(vec![3.0, 0.05], vec![0.1, 0.1]));
        assert!(refused.is_err(), "a slope inside its band proves nothing: {refused:?}");
    }
}
