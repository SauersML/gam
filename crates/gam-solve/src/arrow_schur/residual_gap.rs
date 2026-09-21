//! The attainable accuracy of a recursively updated conjugate-gradient residual (#2627).
//!
//! CG never forms `b − A x_k`. It updates `r_{k+1} = r_k − α_k A p_k`, and in floating point the
//! recursive residual and the true one drift apart. Once `‖r̂_k‖` falls to the size of that drift
//! the recursive residual keeps shrinking while the true residual stagnates, so a threshold below
//! the drift certifies nothing. This module owns the bound on the drift and the one stop predicate
//! every PCG loop of the arrow-Schur solver reads, host and device alike.
//!
//! # The gap
//!
//! Write the updates as executed, first order in the unit roundoff `u`, under the IEEE model
//! `|fl(a ∘ b) − a ∘ b| ≤ u·|a ∘ b|` applied to each product and each sum:
//!
//! ```text
//! x̂_{j+1} = x̂_j + α_j p_j + ξ_j,              |ξ_j| ≤ u·(|α_j p_j| + |x̂_{j+1}|)
//! r̂_{j+1} = r̂_j − α_j (A p_j + μ_j) + η_j,     |η_j| ≤ u·(|α_j Âp_j| + |r̂_{j+1}|),   ‖μ_j‖ ≤ ν_A·‖p_j‖
//! ```
//!
//! where `Âp_j = A p_j + μ_j` is the matvec as computed. The gap `δ_j = b − A x̂_j − r̂_j` obeys
//! `δ_{j+1} = δ_j − A ξ_j + α_j μ_j − η_j`. The same computed `α_j` and `p_j` enter both updates,
//! so the rounding in forming them cancels out of the gap. A loop started at `x̂_0 = 0` with `r̂_0`
//! a copy of `b` has `δ_0 = 0` exactly, hence
//!
//! ```text
//! ‖δ_k‖ ≤ D_k = Σ_{j<k} [ u·N_A·(‖x̂_{j+1}‖ + |α_j|·‖p_j‖) + ν_A·|α_j|·‖p_j‖ + u·(|α_j|·‖Âp_j‖ + ‖r̂_{j+1}‖) ]
//! ```
//!
//! with `N_A ≥ ‖A‖₂`. This is the classical-CG residual-gap recursion of Greenbaum, "Estimating the
//! attainable accuracy of recursively computed residual methods", SIMAX 18(3):535–551, 1997, in
//! the form restated with explicit local rounding bounds by Cools, Yetkin, Agullo, Giraud and
//! Vanroose, SIMAX 39(1):426–450, 2018, §2.1, eqs. (2.11)–(2.15): the gap of classical CG is the
//! plain sum of the local rounding errors of the `x` and `r` updates, with no amplification. Their
//! bounds charge rounding relative to the operands of each operation; the IEEE model charges it
//! relative to the result, which is the form above.
//!
//! # The stop
//!
//! `| ‖b − A x̂_k‖ − ‖r̂_k‖ | ≤ D_k`, so the recursive residual certifies two things:
//!
//! * `‖r̂_k‖ + D_k ≤ rel·‖b‖`: the true residual meets the relative tolerance.
//! * `‖r̂_k‖ ≤ D_k`: the recursive residual no longer resolves the true one. The true residual is
//!   at most `2·D_k`, and no later recursive residual can certify an improvement.
//!
//! No constant is chosen here. The coefficients are `u`, the operator's declared `N_A` and `ν_A`,
//! and norms the loop holds. `D_k` is homogeneous of degree one in `(A, b)` jointly, exactly like
//! `rel·‖b‖`, so a rescaled system stops for the same reason at the same iteration. It is a
//! rigorous bound, pessimistic by up to the iteration count because it sums where the gap often
//! behaves like a maximum.

use super::*;
use super::certified_shift::{
    border_accumulation_depth, round_up, shared_block_norm_bound, substitution_band,
};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};

/// What a CG matvec declares about its own arithmetic: `norm_upper ≥ ‖A‖₂`, and `apply_band` with
/// `‖fl(A p) − A p‖₂ ≤ apply_band·‖p‖₂` for the apply as executed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatvecRoundingBound {
    pub(crate) norm_upper: f64,
    pub(crate) apply_band: f64,
}

impl MatvecRoundingBound {
    /// A dense square matrix applied by one accumulation of `n` products per row.
    ///
    /// Row `i` of the computed product satisfies `|fl(A p)_i − (A p)_i| ≤ γ_n·Σ_j |a_ij|·|p_j|`
    /// whatever the summation order (Higham, *Accuracy and Stability of Numerical Algorithms*,
    /// 2nd ed., §3.1: each term passes through at most `n` rounded operations), so
    /// `‖fl(A p) − A p‖₂ ≤ γ_n·‖|A|‖₂·‖p‖₂`. Both `‖A‖₂` and `‖|A|‖₂` are bounded by
    /// `√(‖A‖₁·‖A‖_∞)`, which takes the same value for `A` and `|A|`.
    pub(crate) fn dense(a: ArrayView2<'_, f64>) -> Self {
        let n = a.nrows();
        let mut column_sums = vec![0.0_f64; a.ncols()];
        let mut max_row_sum = 0.0_f64;
        for row in a.rows() {
            let mut row_sum = 0.0_f64;
            for (column_sum, value) in column_sums.iter_mut().zip(row.iter()) {
                row_sum += value.abs();
                *column_sum += value.abs();
            }
            max_row_sum = max_row_sum.max(row_sum);
        }
        let max_column_sum = column_sums.iter().fold(0.0_f64, |acc, &sum| acc.max(sum));
        let norm_upper = (max_row_sum * max_column_sum).sqrt();
        Self {
            norm_upper,
            apply_band: accumulation_growth(n) * norm_upper,
        }
    }

    /// The reduced border `S = H_ββ + ρ_β·I − Σ_i H_βt^(i) A_i⁻¹ H_tβ^(i)` as the matrix-free
    /// `ReducedSchurOperator` applies it, with `A_i = L_i L_iᵀ` the row factors it solves
    /// against (#2627).
    ///
    /// * `‖A_i⁻¹‖₂ = ‖L_i⁻¹‖₂² ≤ ‖L_i⁻¹‖_F²`, so each row term is at most
    ///   `q_i = s_i²·‖L_i⁻¹‖_F²` and `‖S‖₂ ≤ N_β + ρ_β + Σ_i q_i`. `N_β` is the shared block's
    ///   majorant bound and `s_i` the declared cross-block bounds. Under the Faddeev–Popov
    ///   pin `P S P + Q Qᵀ`, whose two terms act on orthogonal ranges with `Q` orthonormal,
    ///   the norm is at most `max(‖S‖₂, 1)`.
    /// * The apply rounds through the certificate's formation band (`certified_shift`): the
    ///   accumulation of `|H_ββ| + ρ_β·I + Σ_i |H_βt^(i)||Ẑ_i|` at
    ///   [`border_accumulation_depth`], the shared apply's `γ_{k_β}·N_β`, and the
    ///   substitutions' `ω_i·d_i·q_i`, with `κ(L_i) ≤ ‖L_i‖_F·‖L_i⁻¹‖_F`. The pin's two
    ///   projections and its added term charge `γ_{2r+2}` of the norm bound for `r` gauge
    ///   directions.
    ///
    /// `‖L_i⁻¹‖_F` is formed by forward substitution on the identity, so it carries the
    /// same substitution band, which is charged back onto it.
    pub(crate) fn reduced_border(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
    ) -> Result<Self, String> {
        let (cross_norms, cross_apply_depth) = sys.cross_block_row_norm_bounds()?;
        if htt_factors.len() != cross_norms.len() {
            return Err(format!(
                "the reduced border carries {} row factors for {} rows",
                htt_factors.len(),
                cross_norms.len()
            ));
        }
        let (border_norm, border_apply_depth) = shared_block_norm_bound(sys)?;
        let mut coupled = 0.0_f64;
        let mut substituted = 0.0_f64;
        let mut formed = 0.0_f64;
        let mut widest_row = 0usize;
        for (row, &cross_norm) in cross_norms.iter().enumerate() {
            let factor = htt_factors.factor(row);
            let dim = factor.nrows();
            if dim == 0 {
                continue;
            }
            widest_row = widest_row.max(dim);
            let (factor_frobenius, computed_inverse_frobenius) = triangular_frobenius_norms(factor);
            let kappa = round_up(round_up(factor_frobenius * computed_inverse_frobenius).powi(2));
            let band = substitution_band(dim, kappa).ok_or_else(|| {
                format!("row {row}'s factor carries no substitution band at κ ≤ {kappa:e}")
            })?;
            let inverse_frobenius = round_up(computed_inverse_frobenius * round_up(1.0 + band));
            let coupling = round_up(
                round_up(cross_norm * cross_norm) * round_up(inverse_frobenius * inverse_frobenius),
            );
            let spread = round_up(dim as f64 * coupling);
            coupled += coupling;
            substituted += round_up(band * spread);
            formed += round_up(round_up(1.0 + band) * spread);
        }
        let rows = cross_norms.len();
        let terms = rows.saturating_add(1);
        let coupled = guaranteed_norm_upper_bound(coupled, terms);
        let substituted = guaranteed_norm_upper_bound(substituted, terms);
        let formed = guaranteed_norm_upper_bound(formed, terms);
        let shifted = round_up(border_norm + ridge_beta.abs());
        let mut norm_upper = round_up(shifted + coupled);
        let accumulated = round_up(
            accumulation_growth(border_accumulation_depth(cross_apply_depth, widest_row, rows))
                * round_up(shifted + formed),
        );
        let applied = round_up(accumulation_growth(border_apply_depth) * border_norm);
        let mut apply_band = round_up(round_up(accumulated + applied) + substituted);
        if let Some(quotient) = sys.beta_gauge_quotient.as_ref() {
            norm_upper = norm_upper.max(1.0);
            let pinned = accumulation_growth(2 * quotient.dimension() + 2) * norm_upper;
            apply_band = round_up(apply_band + round_up(pinned));
        }
        if !(norm_upper.is_finite() && apply_band.is_finite()) {
            return Err(format!(
                "the reduced border carries no finite bound (norm {norm_upper:e}, apply band \
                 {apply_band:e})"
            ));
        }
        Ok(Self {
            norm_upper,
            apply_band,
        })
    }
}

/// `(‖L‖_F, ‖L⁻¹‖_F)` for a lower-triangular factor, the second by forward substitution on
/// the identity. Each is a guaranteed upper bound on the computed quantity. The error of the
/// substitutions themselves is charged by the caller.
fn triangular_frobenius_norms(factor: ArrayView2<'_, f64>) -> (f64, f64) {
    let dim = factor.nrows();
    let factor_frobenius = frobenius_norm_upper_bound(factor.iter().copied());
    let mut inverse_squares = 0.0_f64;
    let mut count = 0usize;
    let mut column = vec![0.0_f64; dim];
    for j in 0..dim {
        column.fill(0.0);
        for i in j..dim {
            let mut acc = if i == j { 1.0 } else { 0.0 };
            for k in j..i {
                acc -= factor[[i, k]] * column[k];
            }
            column[i] = acc / factor[[i, i]];
        }
        for &value in &column[j..] {
            inverse_squares += value * value;
            count += 1;
        }
    }
    (
        factor_frobenius,
        guaranteed_norm_upper_bound(inverse_squares.sqrt(), count + 1),
    )
}

/// The running bound `D_k` on the gap `‖b − A x̂_k − r̂_k‖₂` of a CG loop started from `x̂_0 = 0`
/// with `r̂_0 = b` copied, so `D_0 = 0`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResidualGap {
    matvec: MatvecRoundingBound,
    bound: f64,
}

impl ResidualGap {
    pub(crate) fn new(matvec: MatvecRoundingBound) -> Self {
        Self { matvec, bound: 0.0 }
    }

    /// Charge one executed update `x̂ ← x̂ + α p`, `r̂ ← r̂ − α Âp`, from the norms of the direction,
    /// the computed matvec, and the updated iterate and residual.
    pub(crate) fn charge_step(
        &mut self,
        alpha: f64,
        direction_norm: f64,
        matvec_norm: f64,
        iterate_norm: f64,
        residual_norm: f64,
    ) {
        let step_norm = alpha.abs() * direction_norm;
        self.bound += UNIT_ROUNDOFF * self.matvec.norm_upper * (iterate_norm + step_norm)
            + self.matvec.apply_band * step_norm
            + UNIT_ROUNDOFF * (alpha.abs() * matvec_norm + residual_norm);
    }

    /// `D_k`, in the units of the residual.
    pub(crate) fn bound(&self) -> f64 {
        self.bound
    }

    /// The stop the recursive residual `‖r̂_k‖` certifies against `relative_tolerance·‖b‖`, or
    /// `None` while neither arm holds.
    pub(crate) fn certified_stop(
        &self,
        residual_norm: f64,
        rhs_norm: f64,
        relative_tolerance: f64,
    ) -> Option<PcgStopReason> {
        if residual_norm + self.bound <= relative_tolerance * rhs_norm {
            Some(PcgStopReason::RelativeToleranceMet)
        } else if residual_norm <= self.bound {
            Some(PcgStopReason::AttainableFloorReached)
        } else {
            None
        }
    }
}
