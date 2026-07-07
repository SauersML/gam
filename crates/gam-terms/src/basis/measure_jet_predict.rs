//! Predict-side measure-jet honesty: the closed-form extrapolation variance
//! from the frame notes (`docs/measure_jet_frame.md` §5).
//!
//! The current Gaussian representers decay off-support toward the parametric
//! backbone with small posterior variance — confident reversion, which the
//! honesty contract forbids. The structural fix prices ignorance off the web from the SAME
//! fitted spectrum that smooths on it: every band level ℓ carries a fitted
//! amplitude λ̂_ℓ (prior precision of the level's innovations), and a query
//! that the level-ℓ kernel mass does not cover simply has an UNKNOWN level-ℓ
//! innovation — prior variance λ̂_ℓ⁻¹, collected in full.
//!
//! # The formula (and its algebraic relation to §5)
//!
//! With `q̄_ℓ = (Σ_i m_i q_ℓ(c_i)) / (Σ_i m_i)` the web-averaged scale-ℓ
//! support and `a_ℓ(x★) = min(q_ℓ(x★)/q̄_ℓ, 1)` the scale-correct on-web-ness
//! weight in `[0, 1]`, let
//! `ℓ★ = min{ℓ : q_ℓ(x★) ≥ coverage_floor · q̄_ℓ}` be the first covering
//! level (ε★ = ε_{ℓ★}). Then, for per-level spectra,
//!
//! ```text
//!   Var_extrap(x★) = Σ_{ℓ < ℓ★} λ̂_ℓ⁻¹  +  Σ_{ℓ ≥ ℓ★} (1 − a_ℓ(x★)) · λ̂_ℓ⁻¹
//!                  = Σ_ℓ λ̂_ℓ⁻¹  −  Σ_{ℓ: ε_ℓ ≥ ε★} a_ℓ(x★) · λ̂_ℓ⁻¹ .
//! ```
//!
//! The second line is the §5 statement: the total prior ignorance of the
//! spectrum minus the part the query's coverage recovers — the recovered sum
//! runs over the covered levels `ε_ℓ ≥ ε★` exactly as written in the charter.
//! In fused mode the band has one precision, so the same coverage idea reduces
//! to one charge: `λ_fused⁻¹` if no level clears its floor, otherwise
//! `(1 − max_ℓ a_ℓ(x★)) · λ_fused⁻¹`.
//! On-web queries (ε★ = ε_0, a_ℓ ≈ 1 everywhere) recover the full spectrum
//! and pay ≈ 0 extra; far-off queries recover (almost) nothing and pay the
//! full Σ_ℓ λ̂_ℓ⁻¹. Levels FINER than the first covering scale get no credit
//! for stray sub-floor kernel mass: below ε★ the prediction is a jet
//! extension, not an interpolation, so those innovations are charged as pure
//! ignorance.
//!
//! # Never-covered convention
//!
//! If no band level clears the coverage floor (ε★ lies past the band), the
//! covered set is EMPTY: in per-level mode every level contributes its full
//! λ̂_ℓ⁻¹ and `Var_extrap = Σ_ℓ λ̂_ℓ⁻¹`; in fused mode the single band
//! amplitude contributes once. The variance saturates at the spectrum's total
//! prior ignorance instead of growing without bound, which is the honest
//! statement: the model's coefficient prior is the only information it ever
//! claimed about such a point.
//!
//! # Monotonicity (the distance-honesty theorem)
//!
//! Claim: if `q ≤ q′` pointwise (the support row of the farther query is
//! nowhere larger), then `Var_extrap(q) ≥ Var_extrap(q′)`.
//!
//! Proof. `{ℓ : q_ℓ ≥ coverage_floor · q̄_ℓ} ⊆
//! {ℓ : q′_ℓ ≥ coverage_floor · q̄_ℓ}` for the scale-specific floors, so
//! `ℓ★(q) ≥ ℓ★(q′)`. Compare the per-level weights `w_ℓ`:
//! - `ℓ < ℓ★(q′)`: both weights are 1;
//! - `ℓ ≥ ℓ★(q)`: `w_ℓ(q) = 1 − a_ℓ(q) ≥ 1 − a_ℓ(q′) = w_ℓ(q′)`;
//! - `ℓ★(q′) ≤ ℓ < ℓ★(q)`: `w_ℓ(q) = 1 ≥ 1 − a_ℓ(q′) = w_ℓ(q′)`.
//! Every weight is no smaller and every `λ̂_ℓ⁻¹ > 0`, so the sum is no
//! smaller. ∎
//!
//! Since the Gaussian kernel mass `q_ℓ(x★)` is pointwise nonincreasing as
//! `x★` recedes from every center simultaneously, intervals widen
//! monotonically with distance from the web. The ε★ gate introduces the only
//! discontinuity, and it is bounded: a level crossing the floor changes its
//! weight by at most `a_ℓ ≤ coverage_floor`, so the total jump is at most
//! `coverage_floor · Σ_ℓ λ̂_ℓ⁻¹` and vanishes as the floor tightens.
//!
//! # Units
//!
//! The result is on the scale of physical `λ̂⁻¹`: callers must unnormalize the
//! fitted Frobenius-normalized precision first (`λ_phys = λ_tilde / c`). Family
//! dispersion scaling remains outside this pure spectrum-side kernel.

use ndarray::{Array1, ArrayView1, ArrayView2};

use super::BasisError;

/// Analytic ambient gradient `∇f̂(x★)` of a frozen measure-jet term's fitted
/// contribution to the linear predictor, in the (standardized) ambient
/// coordinates the frozen geometry lives in.
///
/// The term's fitted contribution is the augmented representer expansion
/// `f̂(x) = Σ_i z_i · K(x, c_i)  +  Σ_h h_h · (xᵀ T)_h`, where
/// `K(x, c_i) = exp(−‖x − c_i‖² / (2 ℓ²))` is the Gaussian representer, `z`
/// the raw-space representer coefficients (`z_full[..m]`), `T` the
/// ambient-linear head lift (`d × head_rank`) and `h` the raw-space head
/// coefficients (`z_full[m..]`). Both coefficient blocks are the raw (pre-gauge)
/// coefficients — lift the fitted reduced coefficients through the frozen
/// identifiability transform first (`β_raw = Z · β̂`).
///
/// Differentiating in `x` (exact, hand-derived — no FD, no autodiff):
///
/// ```text
///   ∂K(x, c_i)/∂x_a = K(x, c_i) · (c_{i,a} − x_a) / ℓ² ,
///   ∂(xᵀ T)_h/∂x_a  = T_{a,h} ,
/// ```
///
/// so the ambient gradient is
///
/// ```text
///   ∇f̂(x)_a = Σ_i z_i · K(x, c_i) · (c_{i,a} − x_a) / ℓ²  +  Σ_h h_h · T_{a,h} .
/// ```
///
/// This is the first-order (jet) term of the measure-jet expansion the frame
/// notes already carry; here it is read out in closed form for the
/// errors-in-variables predictive-variance term
/// `Var_input(x★) = ∇f̂(x★)ᵀ Σ_x ∇f̂(x★)` (issue #2225). `head_transform`/
/// `head_coeffs` are `None`/empty when the term carries no ambient-linear head.
pub fn measure_jet_ambient_gradient(
    query: ArrayView1<'_, f64>,
    centers: ArrayView2<'_, f64>,
    representer_coeffs: ArrayView1<'_, f64>,
    length_scale: f64,
    head_transform: Option<ArrayView2<'_, f64>>,
    head_coeffs: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, BasisError> {
    let d = query.len();
    let m = centers.nrows();
    if centers.ncols() != d {
        crate::bail_dim_basis!(
            "measure-jet ambient gradient: query dimension {d} disagrees with centers ({} × {})",
            m,
            centers.ncols()
        );
    }
    if representer_coeffs.len() != m {
        crate::bail_dim_basis!(
            "measure-jet ambient gradient: {} representer coefficients for {m} centers",
            representer_coeffs.len()
        );
    }
    if !(length_scale.is_finite() && length_scale > 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet ambient gradient needs a positive finite length_scale; got {length_scale}"
        );
    }
    let inv_l2 = 1.0 / (length_scale * length_scale);
    let mut grad = Array1::<f64>::zeros(d);
    for i in 0..m {
        let center = centers.row(i);
        let mut sq = 0.0_f64;
        for a in 0..d {
            let delta = query[a] - center[a];
            sq += delta * delta;
        }
        let k = (-0.5 * sq * inv_l2).exp();
        let coeff = representer_coeffs[i] * k * inv_l2;
        for a in 0..d {
            // ∂K/∂x_a = K · (c_{i,a} − x_a) / ℓ².
            grad[a] += coeff * (center[a] - query[a]);
        }
    }
    if let Some(t) = head_transform {
        let head_rank = t.ncols();
        if head_coeffs.len() != head_rank {
            crate::bail_dim_basis!(
                "measure-jet ambient gradient: {} head coefficients for a head lift with \
                 {head_rank} columns",
                head_coeffs.len()
            );
        }
        if t.nrows() != d {
            crate::bail_dim_basis!(
                "measure-jet ambient gradient: head lift has {} rows but ambient dimension is {d}",
                t.nrows()
            );
        }
        for a in 0..d {
            let mut acc = 0.0_f64;
            for h in 0..head_rank {
                acc += head_coeffs[h] * t[(a, h)];
            }
            grad[a] += acc;
        }
    } else if !head_coeffs.is_empty() {
        crate::bail_dim_basis!(
            "measure-jet ambient gradient: {} head coefficients supplied without a head lift",
            head_coeffs.len()
        );
    }
    Ok(grad)
}

#[derive(Clone, Copy)]
pub enum MeasureJetExtrapolationSpectrum<'a> {
    /// One physical precision per band level.
    PerLevel(&'a [f64]),
    /// One physical precision for the fused band. It is charged once, with the
    /// band's best coverage fraction.
    Fused(f64),
}

/// Frame-note §5: closed-form extrapolation variance at a query — the price of
/// ignorance off the web, read from the fitted spectrum. `ε★` = the first
/// covering scale (smallest band scale at which the query's kernel mass
/// clears `coverage_floor` × `q̄_ℓ`); levels finer than `ε★` contribute
/// their full prior variance `λ̂_ℓ⁻¹`, levels from `ε★` up contribute the
/// uncovered fraction `(1 − a_ℓ(x★)) · λ̂_ℓ⁻¹` with
/// `a_ℓ(x★) = min(q_ℓ(x★)/q̄_ℓ, 1)` the smooth on-web-ness weight.
/// Equivalently (see the module docs) the total prior ignorance
/// `Σ_ℓ λ̂_ℓ⁻¹` minus the §5 coverage-recovered sum
/// `Σ_{ℓ: ε_ℓ ≥ ε★} a_ℓ(x★)/λ̂_ℓ`. On-web queries (ε★ = ε_0, a ≈ 1) pay
/// ≈ 0 extra; queries never covered by the band pay `Σ_ℓ λ̂_ℓ⁻¹` exactly —
/// intervals widen monotonically with distance (theorem in the module docs).
///
/// Inputs: `support_row` = `q_ℓ(x★)` per band scale (one row of
/// [`super::measure_jet_support_curve`]), `eps_band` the realized ascending
/// band, `support_means` = `q̄_ℓ` per band scale, `spectrum` the physical
/// precision spectrum, `coverage_floor` ∈ (0, 1) (e.g. 0.05).
pub fn measure_jet_extrapolation_variance(
    support_row: ArrayView1<'_, f64>,
    eps_band: &[f64],
    support_means: &[f64],
    spectrum: MeasureJetExtrapolationSpectrum<'_>,
    coverage_floor: f64,
) -> Result<f64, BasisError> {
    let n_levels = eps_band.len();
    if n_levels == 0 {
        crate::bail_invalid_basis!("measure-jet extrapolation variance needs a nonempty band");
    }
    if support_row.len() != n_levels || support_means.len() != n_levels {
        crate::bail_dim_basis!(
            "measure-jet extrapolation variance needs one support value and one support mean per \
             band scale: {} support values, {} support means, {} scales",
            support_row.len(),
            support_means.len(),
            n_levels
        );
    }
    for (l, pair) in eps_band.windows(2).enumerate() {
        if pair[1] <= pair[0] {
            crate::bail_invalid_basis!(
                "measure-jet band must be strictly ascending: eps[{l}] = {} vs eps[{}] = {}",
                pair[0],
                l + 1,
                pair[1]
            );
        }
    }
    if eps_band.iter().any(|e| !(e.is_finite() && *e > 0.0)) {
        crate::bail_invalid_basis!("measure-jet band scales must be finite and positive");
    }
    if support_row.iter().any(|q| !(q.is_finite() && *q >= 0.0)) {
        crate::bail_invalid_basis!(
            "measure-jet support row must be finite and nonnegative (kernel masses)"
        );
    }
    if support_means.iter().any(|q| !(q.is_finite() && *q > 0.0)) {
        crate::bail_invalid_basis!("measure-jet support means must be finite and positive");
    }
    if !(coverage_floor.is_finite() && coverage_floor > 0.0 && coverage_floor < 1.0) {
        crate::bail_invalid_basis!(
            "measure-jet coverage floor must lie strictly in (0, 1); got {coverage_floor}"
        );
    }
    match spectrum {
        MeasureJetExtrapolationSpectrum::PerLevel(lambda_hat) => {
            if lambda_hat.len() != n_levels {
                crate::bail_dim_basis!(
                    "measure-jet per-level extrapolation variance needs one physical precision per \
                     band scale: {} precisions, {} scales",
                    lambda_hat.len(),
                    n_levels
                );
            }
            if lambda_hat.iter().any(|l| !(l.is_finite() && *l > 0.0)) {
                crate::bail_invalid_basis!(
                    "measure-jet per-scale amplitudes must be finite and positive (physical precisions)"
                );
            }
            let first_covering = support_row
                .iter()
                .zip(support_means.iter())
                .position(|(q, q_bar)| *q >= coverage_floor * *q_bar)
                .unwrap_or(n_levels);
            let mut variance = 0.0_f64;
            for (l, ((&q, &q_bar), &lam)) in support_row
                .iter()
                .zip(support_means.iter())
                .zip(lambda_hat.iter())
                .enumerate()
            {
                let weight = if l < first_covering {
                    1.0
                } else {
                    1.0 - (q / q_bar).min(1.0)
                };
                variance += weight / lam;
            }
            Ok(variance)
        }
        MeasureJetExtrapolationSpectrum::Fused(lambda_hat) => {
            if !(lambda_hat.is_finite() && lambda_hat > 0.0) {
                crate::bail_invalid_basis!(
                    "measure-jet fused amplitude must be finite and positive (physical precision)"
                );
            }
            let mut best_coverage = 0.0_f64;
            let mut covered = false;
            for (&q, &q_bar) in support_row.iter().zip(support_means.iter()) {
                let coverage = (q / q_bar).min(1.0);
                best_coverage = best_coverage.max(coverage);
                if q >= coverage_floor * q_bar {
                    covered = true;
                }
            }
            let weight = if covered { 1.0 - best_coverage } else { 1.0 };
            Ok(weight / lambda_hat)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, arr1};

    /// Shared deterministic fixture: a 5-level dyadic band with a
    /// non-constant fitted spectrum.
    pub(crate) fn band() -> Vec<f64> {
        vec![0.05, 0.1, 0.2, 0.4, 0.8]
    }

    pub(crate) fn lambdas() -> Vec<f64> {
        vec![40.0, 11.0, 3.5, 1.25, 0.6]
    }

    pub(crate) fn support_means(eps: &[f64]) -> Vec<f64> {
        vec![TOTAL; eps.len()]
    }

    pub(crate) const FLOOR: f64 = 0.05;
    pub(crate) const TOTAL: f64 = 1.0;

    pub(crate) fn total_ignorance(lams: &[f64]) -> f64 {
        lams.iter().map(|l| 1.0 / l).sum()
    }

    /// The exact single-unit-mass support curve at distance `d`:
    /// q_ℓ(d) = total · exp(−d²/(2ε_ℓ²)) — the physical family the support
    /// diagnostic produces for a one-center web.
    pub(crate) fn support_at_distance(d: f64, eps: &[f64]) -> Array1<f64> {
        Array1::from_iter(eps.iter().map(|e| TOTAL * (-d * d / (2.0 * e * e)).exp()))
    }

    /// (a) Monotone in distance: along the exact kernel-mass family the
    /// support row is pointwise nonincreasing in d, so the variance must be
    /// nondecreasing — including across every coverage-floor crossing in the
    /// sweep.
    #[test]
    pub(crate) fn extrapolation_variance_is_monotone_in_distance() {
        let eps = band();
        let lams = lambdas();
        let q_bar = support_means(&eps);
        let mut prev = -1.0_f64;
        // 0 → 6 in steps of 0.015: spans on-web through far-off, crossing
        // the floor at every band level along the way.
        for step in 0..400 {
            let d = 0.015 * step as f64;
            let row = support_at_distance(d, &eps);
            let v = measure_jet_extrapolation_variance(
                row.view(),
                &eps,
                &q_bar,
                MeasureJetExtrapolationSpectrum::PerLevel(&lams),
                FLOOR,
            )
            .expect("valid inputs");
            assert!(
                v >= prev,
                "variance decreased with distance: variance({d:.3}) = {v:.12} < {prev:.12}"
            );
            prev = v;
        }
        // And the saturation: the far end of the sweep reaches the full
        // prior ignorance (never-covered convention).
        assert!(
            (prev - total_ignorance(&lams)).abs() <= 1e-12,
            "far-field variance must saturate at Σ 1/λ̂: got {prev}"
        );
    }

    /// (a′) Pointwise domination, no geometric family assumed: a support row
    /// that is pointwise smaller never yields smaller variance — exercised
    /// on a NON-monotone-in-ℓ row pair as well.
    #[test]
    pub(crate) fn extrapolation_variance_is_monotone_under_pointwise_domination() {
        let eps = band();
        let lams = lambdas();
        let q_bar = support_means(&eps);
        let rows = [
            arr1(&[0.9, 0.95, 0.99, 1.0, 1.0]),
            arr1(&[0.02, 0.3, 0.06, 0.8, 0.97]),
            arr1(&[0.0, 0.0, 0.04, 0.2, 0.6]),
            arr1(&[0.04, 0.04, 0.04, 0.04, 0.049]),
        ];
        for row in &rows {
            for shrink in [1.0, 0.9, 0.7, 0.3, 0.0] {
                let smaller = row.mapv(|q| shrink * q);
                let v_big = measure_jet_extrapolation_variance(
                    row.view(),
                    &eps,
                    &q_bar,
                    MeasureJetExtrapolationSpectrum::PerLevel(&lams),
                    FLOOR,
                )
                .expect("valid inputs");
                let v_small = measure_jet_extrapolation_variance(
                    smaller.view(),
                    &eps,
                    &q_bar,
                    MeasureJetExtrapolationSpectrum::PerLevel(&lams),
                    FLOOR,
                )
                .expect("valid inputs");
                assert!(
                    v_small >= v_big,
                    "pointwise-smaller support gave smaller variance: {v_small} < {v_big} \
                     (row {row:?}, shrink {shrink})"
                );
            }
        }
    }

    /// (b) On-web limit: full kernel mass at every scale prices ZERO extra
    /// variance; near-full mass prices at most the uncovered fraction of the
    /// total prior ignorance.
    #[test]
    pub(crate) fn extrapolation_variance_vanishes_on_web() {
        let eps = band();
        let lams = lambdas();
        let q_bar = support_means(&eps);
        let full = Array1::from_elem(eps.len(), TOTAL);
        let v_full = measure_jet_extrapolation_variance(
            full.view(),
            &eps,
            &q_bar,
            MeasureJetExtrapolationSpectrum::PerLevel(&lams),
            FLOOR,
        )
        .expect("valid inputs");
        assert_eq!(v_full, 0.0, "full coverage must price zero extra variance");

        let near = Array1::from_elem(eps.len(), 0.97 * TOTAL);
        let v_near = measure_jet_extrapolation_variance(
            near.view(),
            &eps,
            &q_bar,
            MeasureJetExtrapolationSpectrum::PerLevel(&lams),
            FLOOR,
        )
        .expect("valid inputs");
        let budget = total_ignorance(&lams);
        assert!(
            v_near <= 0.05 * budget,
            "near-full coverage must price a small fraction of Σ 1/λ̂: {v_near} vs budget {budget}"
        );
    }

    /// (c) Off-web limit: zero support everywhere (never covered) collects
    /// the spectrum's total prior ignorance Σ 1/λ̂ EXACTLY.
    #[test]
    pub(crate) fn extrapolation_variance_saturates_off_web() {
        let eps = band();
        let lams = lambdas();
        let q_bar = support_means(&eps);
        let zero = Array1::<f64>::zeros(eps.len());
        let v = measure_jet_extrapolation_variance(
            zero.view(),
            &eps,
            &q_bar,
            MeasureJetExtrapolationSpectrum::PerLevel(&lams),
            FLOOR,
        )
        .expect("valid inputs");
        assert_eq!(
            v,
            total_ignorance(&lams),
            "never-covered query must pay Σ 1/λ̂ exactly"
        );
    }

    /// (d) Spectrum scaling: doubling every fitted amplitude halves the
    /// variance — the λ̂⁻¹ pricing is exact, in every coverage regime.
    #[test]
    pub(crate) fn extrapolation_variance_halves_when_amplitudes_double() {
        let eps = band();
        let lams = lambdas();
        let q_bar = support_means(&eps);
        let doubled: Vec<f64> = lams.iter().map(|l| 2.0 * l).collect();
        // Mixed regime: some levels below the floor, some covered partially,
        // some fully — both weight branches exercised.
        let rows = [
            support_at_distance(0.35, &eps),
            Array1::<f64>::zeros(eps.len()),
            Array1::from_elem(eps.len(), 0.5),
        ];
        for row in &rows {
            let v1 = measure_jet_extrapolation_variance(
                row.view(),
                &eps,
                &q_bar,
                MeasureJetExtrapolationSpectrum::PerLevel(&lams),
                FLOOR,
            )
            .expect("valid inputs");
            let v2 = measure_jet_extrapolation_variance(
                row.view(),
                &eps,
                &q_bar,
                MeasureJetExtrapolationSpectrum::PerLevel(&doubled),
                FLOOR,
            )
            .expect("valid inputs");
            assert!(
                (2.0 * v2 - v1).abs() <= 1e-15 * v1.max(1.0),
                "doubling λ̂ must halve the variance: {v1} vs 2×{v2}"
            );
        }
    }

    /// Convention pin: the ε★ gate. Sub-floor mass at every level is
    /// never-covered (full Σ 1/λ̂, no credit for stray mass); the moment ONE
    /// level clears the floor, that level and every coarser one switch to
    /// the smooth uncovered-fraction weight while finer levels stay fully
    /// charged.
    #[test]
    pub(crate) fn extrapolation_variance_gate_convention() {
        let eps = band();
        let lams = lambdas();
        let q_bar = support_means(&eps);
        let sub_floor = Array1::from_elem(eps.len(), 0.049 * TOTAL);
        let v_sub = measure_jet_extrapolation_variance(
            sub_floor.view(),
            &eps,
            &q_bar,
            MeasureJetExtrapolationSpectrum::PerLevel(&lams),
            FLOOR,
        )
        .expect("valid inputs");
        assert_eq!(
            v_sub,
            total_ignorance(&lams),
            "sub-floor mass earns no credit: full Σ 1/λ̂"
        );

        // Coverage exactly at the floor on the coarsest level only.
        let mut at_floor = sub_floor.clone();
        at_floor[eps.len() - 1] = FLOOR * TOTAL;
        let v_floor = measure_jet_extrapolation_variance(
            at_floor.view(),
            &eps,
            &q_bar,
            MeasureJetExtrapolationSpectrum::PerLevel(&lams),
            FLOOR,
        )
        .expect("valid inputs");
        let expected: f64 = lams[..eps.len() - 1].iter().map(|l| 1.0 / l).sum::<f64>()
            + (1.0 - FLOOR) / lams[eps.len() - 1];
        assert!(
            (v_floor - expected).abs() <= 1e-15,
            "floor-clearing coarsest level must take weight 1 − a: {v_floor} vs {expected}"
        );
        // The gate's discontinuity is bounded by the documented
        // coverage_floor · Σ 1/λ̂ budget.
        assert!(
            v_sub - v_floor <= FLOOR * total_ignorance(&lams) + 1e-15,
            "gate jump exceeds the documented coverage_floor bound"
        );
    }

    #[test]
    pub(crate) fn fused_extrapolation_charges_single_band_amplitude_once() {
        let eps = band();
        let q_bar = support_means(&eps);
        let lam = 2.5;
        let zero = Array1::<f64>::zeros(eps.len());
        let v_zero = measure_jet_extrapolation_variance(
            zero.view(),
            &eps,
            &q_bar,
            MeasureJetExtrapolationSpectrum::Fused(lam),
            FLOOR,
        )
        .expect("valid inputs");
        assert_eq!(
            v_zero,
            1.0 / lam,
            "never-covered fused band must pay one amplitude, not one per level"
        );

        let covered = arr1(&[0.01, 0.2, 0.4, 0.75, 0.5]);
        let v_covered = measure_jet_extrapolation_variance(
            covered.view(),
            &eps,
            &q_bar,
            MeasureJetExtrapolationSpectrum::Fused(lam),
            FLOOR,
        )
        .expect("valid inputs");
        let expected = (1.0 - 0.75) / lam;
        assert!(
            (v_covered - expected).abs() <= 1e-15,
            "fused band must use the best covered level once: {v_covered} vs {expected}"
        );
    }

    /// Closed-form `f̂(x)`: the augmented representer expansion the analytic
    /// gradient differentiates — used only as the finite-difference oracle.
    fn eval_fitted(
        query: ArrayView1<'_, f64>,
        centers: ArrayView2<'_, f64>,
        z: ArrayView1<'_, f64>,
        length_scale: f64,
        head: Option<ArrayView2<'_, f64>>,
        head_coeffs: ArrayView1<'_, f64>,
    ) -> f64 {
        let inv_two_l2 = 1.0 / (2.0 * length_scale * length_scale);
        let mut val = 0.0_f64;
        for i in 0..centers.nrows() {
            let mut sq = 0.0_f64;
            for a in 0..query.len() {
                let dlt = query[a] - centers[(i, a)];
                sq += dlt * dlt;
            }
            val += z[i] * (-sq * inv_two_l2).exp();
        }
        if let Some(t) = head {
            for h in 0..t.ncols() {
                let mut proj = 0.0_f64;
                for a in 0..query.len() {
                    proj += query[a] * t[(a, h)];
                }
                val += head_coeffs[h] * proj;
            }
        }
        val
    }

    /// The analytic ambient gradient matches a central finite difference of the
    /// fitted surface to FD accuracy — the delta-method propagator #2225 wires
    /// into the predictive variance is the true ∇f̂ (representers + head).
    #[test]
    pub(crate) fn ambient_gradient_matches_central_difference() {
        use ndarray::{arr1, arr2};
        let centers = arr2(&[[0.0, 0.0], [1.0, 0.5], [-0.7, 0.9], [0.4, -1.1]]);
        let z = arr1(&[0.8, -1.3, 0.5, 2.0]);
        let length_scale = 0.6;
        // A rank-2 ambient-linear head lift T (d=2, head_rank=2).
        let t = arr2(&[[1.0, 0.2], [-0.3, 0.9]]);
        let head_coeffs = arr1(&[0.7, -0.4]);
        let query = arr1(&[0.15, -0.2]);

        let grad = measure_jet_ambient_gradient(
            query.view(),
            centers.view(),
            z.view(),
            length_scale,
            Some(t.view()),
            head_coeffs.view(),
        )
        .expect("valid gradient");

        let h = 1e-6;
        for a in 0..query.len() {
            let mut qp = query.clone();
            let mut qm = query.clone();
            qp[a] += h;
            qm[a] -= h;
            let fp = eval_fitted(
                qp.view(),
                centers.view(),
                z.view(),
                length_scale,
                Some(t.view()),
                head_coeffs.view(),
            );
            let fm = eval_fitted(
                qm.view(),
                centers.view(),
                z.view(),
                length_scale,
                Some(t.view()),
                head_coeffs.view(),
            );
            let fd = (fp - fm) / (2.0 * h);
            assert!(
                (grad[a] - fd).abs() <= 1e-6 * (1.0 + fd.abs()),
                "axis {a}: analytic {} vs central FD {fd}",
                grad[a]
            );
        }
    }

    /// No head: the representer-only gradient still matches FD, and a stray
    /// head coefficient without a lift is rejected.
    #[test]
    pub(crate) fn ambient_gradient_representer_only_and_head_guard() {
        use ndarray::{arr1, arr2};
        let centers = arr2(&[[0.0], [0.5], [-0.4]]);
        let z = arr1(&[1.0, -2.0, 0.5]);
        let length_scale = 0.3;
        let query = arr1(&[0.1]);
        let empty = Array1::<f64>::zeros(0);
        let grad = measure_jet_ambient_gradient(
            query.view(),
            centers.view(),
            z.view(),
            length_scale,
            None,
            empty.view(),
        )
        .expect("valid gradient");
        let h = 1e-6;
        let mut qp = query.clone();
        let mut qm = query.clone();
        qp[0] += h;
        qm[0] -= h;
        let fd = (eval_fitted(qp.view(), centers.view(), z.view(), length_scale, None, empty.view())
            - eval_fitted(qm.view(), centers.view(), z.view(), length_scale, None, empty.view()))
            / (2.0 * h);
        assert!((grad[0] - fd).abs() <= 1e-6 * (1.0 + fd.abs()));

        let stray = arr1(&[1.0]);
        assert!(
            measure_jet_ambient_gradient(
                query.view(),
                centers.view(),
                z.view(),
                length_scale,
                None,
                stray.view(),
            )
            .is_err(),
            "head coefficients without a head lift must be rejected"
        );
    }
}
