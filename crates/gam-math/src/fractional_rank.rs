//! Reference law of a quadratic form truncated at a FRACTIONAL rank.
//!
//! A penalized smooth whose effective degrees of freedom `r` is not an integer
//! is tested (Wood 2013, *Biometrika* 100, 221–228) on its first `k = ⌊r⌋`
//! whitened covariance eigen-directions plus a `ν = r − k` share of the
//! `(k+1)`-st. With `u_i` the standardized coordinates on those directions,
//! independent standard normals under the null, the statistic
//!
//! ```text
//! T_r = Σ_{i≤k} u_i² + ν·u_{k+1}²
//! ```
//!
//! is distributed EXACTLY as
//!
//! ```text
//! L_r = χ²_k + ν·χ²₁,
//! ```
//!
//! which has mean `r`, is continuous in `r`, and is `χ²_r` at every integer.
//! With the scale estimated on `ρ` residual degrees of freedom the statistic is
//! divided by `χ²_ρ/ρ`, independent of `L_r`.

use crate::probability::{TailProbability, WeightedChiSquareTerm, signed_weighted_chi_square_sf};

/// Survival probability of the fractional-rank reference law at `statistic`,
/// with its relative error bound:
///
/// * `residual_df = None` (scale known): `P(χ²_k + ν·χ²₁ > statistic)`;
/// * `residual_df = Some(ρ)` (scale estimated): `P((χ²_k + ν·χ²₁)/(χ²_ρ/ρ) > statistic)`,
///   the event `χ²_k + ν·χ²₁ − (statistic/ρ)·χ²_ρ > 0`,
///
/// with `k = ⌊rank⌋`, `ν = rank − k`. Both are one signed weighted chi-square
/// tail, so they carry its relative accuracy into the far tail; at an integer
/// rank the `ν` term vanishes and the law is `χ²_k` or `k·F_{k,ρ}` exactly.
///
/// Returns `NaN` in both fields for a `rank` that is not finite and positive, a
/// `residual_df` that is not finite and positive, or a `NaN` or negative
/// `statistic`.
pub fn fractional_rank_sf(statistic: f64, rank: f64, residual_df: Option<f64>) -> TailProbability {
    let invalid = TailProbability { probability: f64::NAN, relative_error: f64::NAN };
    if !(rank.is_finite() && rank > 0.0) || statistic.is_nan() || statistic < 0.0 {
        return invalid;
    }
    let whole = rank.floor();
    let share = rank - whole;
    let mut terms = Vec::with_capacity(3);
    if whole > 0.0 {
        terms.push(WeightedChiSquareTerm { weight: 1.0, degrees_of_freedom: whole });
    }
    if share > 0.0 {
        terms.push(WeightedChiSquareTerm { weight: share, degrees_of_freedom: 1.0 });
    }
    match residual_df {
        None => signed_weighted_chi_square_sf(&terms, statistic),
        Some(rho) if rho.is_finite() && rho > 0.0 => {
            if statistic == f64::INFINITY {
                return TailProbability { probability: 0.0, relative_error: 0.0 };
            }
            terms.push(WeightedChiSquareTerm { weight: -statistic / rho, degrees_of_freedom: rho });
            signed_weighted_chi_square_sf(&terms, 0.0)
        }
        Some(_) => invalid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::{chi_square_sf, fisher_snedecor_sf};
    use crate::special::gauss_legendre;

    fn known(statistic: f64, rank: f64) -> f64 {
        fractional_rank_sf(statistic, rank, None).probability
    }

    fn estimated(statistic: f64, rank: f64, residual_df: f64) -> f64 {
        fractional_rank_sf(statistic, rank, Some(residual_df)).probability
    }

    #[test]
    fn integer_ranks_are_the_chi_square_and_f_laws() {
        for &k in &[1.0, 2.0, 5.0] {
            for &x in &[0.5, 3.0, 9.0, 40.0] {
                let chi = chi_square_sf(x, k);
                assert!((known(x, k) - chi).abs() <= 1e-12 * chi, "k {k} x {x}");
                let f = fisher_snedecor_sf(x / k, k, 30.0);
                assert!((estimated(x, k, 30.0) - f).abs() <= 1e-11 * f, "k {k} x {x}");
            }
        }
    }

    #[test]
    fn known_scale_holds_relative_accuracy_deep_in_the_tail() {
        // References exact in relative terms, independent of the inversion.
        //
        // k = 1: `Z₁² + ν Z₂²` in polar coordinates is `P(> x) = mean_θ exp(−x/(2c(θ)))`,
        // `c = cos²θ + ν sin²θ`, a periodic analytic integrand the trapezoid rule
        // resolves to rounding.
        let nu = 0.4;
        for &x in &[0.2, 3.0, 60.0, 300.0] {
            let panels = 4096;
            let polar = (0..panels)
                .map(|i| {
                    let theta = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / panels as f64;
                    let c = theta.cos().powi(2) + nu * theta.sin().powi(2);
                    (-x / (2.0 * c)).exp()
                })
                .sum::<f64>()
                / panels as f64;
            let value = known(x, 1.0 + nu);
            assert!((value - polar).abs() <= 1e-11 * polar, "x {x}: {value} vs {polar}");
        }
        // k = 2: `Z₀² + Z₁² + ν Z₂²` is `χ²₃·c(ω)` along a uniform direction ω on
        // the sphere, `c = 1 − (1 − ν)w²` with `w = ω₂` uniform on `[−1, 1]`.
        let (nodes, weights) = gauss_legendre(256);
        for &x in &[1.0, 40.0, 200.0] {
            let sphere = nodes
                .iter()
                .zip(weights.iter())
                .map(|(&w, &g)| 0.5 * g * chi_square_sf(x / (1.0 - (1.0 - nu) * w * w), 3.0))
                .sum::<f64>();
            let value = known(x, 2.0 + nu);
            assert!(value > 0.0);
            assert!((value - sphere).abs() <= 1e-10 * sphere, "x {x}: {value} vs {sphere}");
        }
    }

    #[test]
    fn the_law_is_continuous_at_both_ends_of_the_share() {
        for &k in &[1.0, 2.0, 5.0] {
            for &x in &[0.5, 3.0, 9.0] {
                let above = known(x, k + 1e-9);
                assert!((above - chi_square_sf(x, k)).abs() <= 1e-8, "k {k} x {x}");
                let below = known(x, k + 1.0 - 1e-9);
                assert!((below - chi_square_sf(x, k + 1.0)).abs() <= 1e-8, "k {k} x {x}");
                let above = estimated(x, k + 1e-9, 40.0);
                assert!((above - fisher_snedecor_sf(x / k, k, 40.0)).abs() <= 1e-8, "k {k} x {x}");
                let below = estimated(x, k + 1.0 - 1e-9, 40.0);
                let next = fisher_snedecor_sf(x / (k + 1.0), k + 1.0, 40.0);
                assert!((below - next).abs() <= 1e-8, "k {k} x {x}");
            }
        }
    }

    #[test]
    fn the_law_is_monotone_in_the_statistic_and_in_the_rank() {
        let mut previous = 1.0;
        for i in 1..200 {
            let value = known(0.1 * i as f64, 3.3);
            assert!(value <= previous && value >= 0.0, "i {i}");
            previous = value;
        }
        let mut previous = 0.0;
        for i in 1..40 {
            let value = known(4.0, 0.1 * i as f64);
            assert!(value >= previous, "i {i}");
            previous = value;
        }
    }

    #[test]
    fn below_rank_one_is_the_scaled_one_degree_law() {
        let value = known(0.3, 0.6);
        let reference = chi_square_sf(0.5, 1.0);
        assert!((value - reference).abs() <= 1e-12 * reference);
        let value = estimated(0.3, 0.6, 25.0);
        let reference = fisher_snedecor_sf(0.5, 1.0, 25.0);
        assert!((value - reference).abs() <= 1e-11 * reference);
    }

    #[test]
    fn estimated_scale_approaches_the_known_scale_as_residual_df_grows() {
        for &rank in &[1.3, 2.6, 5.5] {
            for &x in &[0.5, 2.0, 6.0] {
                let gap = (known(x * rank, rank) - estimated(x * rank, rank, 1e6)).abs();
                assert!(gap <= 1e-3, "rank {rank} x {x}");
            }
        }
    }

    #[test]
    fn invalid_inputs_are_nan() {
        assert!(known(1.0, 0.0).is_nan());
        assert!(known(-1.0, 2.5).is_nan());
        assert!(known(f64::NAN, 2.5).is_nan());
        assert!(estimated(1.0, 2.5, 0.0).is_nan());
        assert_eq!(known(0.0, 2.5), 1.0);
        assert_eq!(known(f64::INFINITY, 2.5), 0.0);
        assert_eq!(estimated(f64::INFINITY, 2.5, 10.0), 0.0);
    }
}
