//! The unit roundoff and Wilkinson's accumulation factor, owned in the lowest crate so every rounding bound in the
//! workspace reads one definition. `gam_linalg::roundoff` re-exports both at its own paths, beside the bands built on
//! them.
//!
//! The factor is Wilkinson's, in the form given by Higham (*Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//! SIAM 2002, Lemma 3.1): a product of `k` factors `(1 + δᵢ)^{±1}` with `|δᵢ| ≤ u` is `1 + θ` with
//! `|θ| ≤ γ_k = k·u/(1 − k·u)`. A first-order count `k·u` sits below `γ_k` and agrees with it to `O(u²)`.

/// Unit roundoff `u = EPSILON/2`.
///
/// `EPSILON` is the gap between `1.0` and the next representable `f64`; the
/// error of a single correctly-rounded operation is at most half that gap
/// relative to the result, which is the quantity every backward-error bound is
/// stated in. The factor of two between the two is the single most common
/// source of "the same tolerance, twice, 2× apart".
pub const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// Wilkinson's growth factor `γ_n = n·u / (1 − n·u)` for an `n`-operation
/// accumulation.
///
/// Returns infinity once `n·u ≥ 1`, where the bound carries no information —
/// an accumulation that long has no useful error bound, and reporting an
/// infinite band is the honest answer rather than a negative or wrapped one.
pub const fn accumulation_growth(operations: usize) -> f64 {
    let scaled = operations as f64 * UNIT_ROUNDOFF;
    if !(scaled < 1.0) {
        return f64::INFINITY;
    }
    scaled / (1.0 - scaled)
}

/// An upper bound on a positive expression whose computed value is `value` after `operations` rounded operations:
/// `exact ≤ value·(1 − u)^−k ≤ value·(1 + γ_k)`. The three extra operations cover forming the factor and the product.
pub fn inflated(value: f64, operations: usize) -> f64 {
    value * (1.0 + accumulation_growth(operations + 3))
}

/// Rounding band on `|‖x̂‖² − 1|` for a length-`dim` vector normalized in f64, `x̂ = x/‖x‖`, and then measured in
/// f64. It is the widest a correctly normalized vector can read off the unit sphere, so a larger defect was not
/// normalized in f64.
///
/// First order in `u`, with `γ_n` from [`accumulation_growth`]:
/// - `‖x‖²` sums `dim` rounded squares, so it errs by `γ_dim` relative, and its square root by `γ_dim/2 + u`.
/// - Each coordinate then takes at most two more roundings (a quotient `xᵢ/‖x‖`, or a product with a rounded
///   reciprocal), so the exact `‖x̂‖²` is `1` within `γ_dim + 6u = γ_{dim+6}`.
/// - Measuring `‖x̂‖²` sums `dim` rounded squares again, adding `γ_dim`, and `γ_{dim+6} + γ_dim ≤ γ_{2·dim+6}`.
pub fn unit_normalization_band(dim: usize) -> f64 {
    accumulation_growth(dim.saturating_mul(2).saturating_add(6))
}

/// Passes a reorthogonalized Gram–Schmidt append makes over its basis. Twice is enough: after a second pass of
/// classical or modified Gram–Schmidt the residual is orthogonal to the basis to working precision, and a third pass
/// changes nothing resolvable (Giraud, Langou and Rozložník, *Numer. Math.* 101, 2005). An append that runs these
/// passes reads its residual against [`gram_schmidt_residual_band`] with this count.
pub const GRAM_SCHMIDT_PASSES: usize = 2;

/// Rounding band on the residual of a length-`dim` vector of norm `norm` after `passes` Gram–Schmidt passes against
/// `directions` unit vectors.
///
/// First order in `u`, with `γ_n` from [`accumulation_growth`]:
/// - One projection step against a unit `q` forms `c = qᵀx`. Each of its `dim` summands passes its product and at most
///   `dim − 1` additions, so `|δc| ≤ γ_dim·Σ|qᵢxᵢ| ≤ γ_dim‖x‖` (Cauchy–Schwarz). That moves the residual by at most
///   `γ_dim‖x‖` along `q`.
/// - The update `xᵢ − c·qᵢ` passes two rounded operations per entry, the product and the subtraction, so
///   `|δxᵢ| ≤ γ₂(|xᵢ| + |c·qᵢ|)`. In norm that is at most `γ₂(‖x‖ + |c|) ≤ 2γ₂‖x‖ ≤ γ₄‖x‖`, because `|c| ≤ ‖x‖`.
/// - One step therefore errs by at most `(γ_dim + γ₄)‖x‖ ≤ γ_{dim+4}‖x‖` (`γ_a + γ_b ≤ γ_{a+b}`), and the append takes
///   `passes·directions` steps.
/// - The unit directions (normalized by computed norms) and the two norms a caller compares perturb this by relative
///   factors `1 + O(dim·u)`. Those are second order against a band already `O(passes·directions·dim·u)`.
///
/// A residual at or below this band is not resolved from zero by the arithmetic that produced it. For
/// `directions = 0` the band is zero, and any nonzero vector is resolved.
pub fn gram_schmidt_residual_band(passes: usize, directions: usize, dim: usize, norm: f64) -> f64 {
    passes.saturating_mul(directions) as f64 * accumulation_growth(dim.saturating_add(4)) * norm
}

/// Backward-error band on the eigenvalues of a symmetric `n × n` matrix whose inertia was read off an `LDLᵀ`
/// factorization `P A Pᵀ = L̂ D̂ L̂ᵀ` (#2901).
///
/// The computed factors are the exact factorization of `A + E`, with `‖E‖₂` of order
/// `n·ε·(‖A‖₂ + ‖ |L̂||D̂||L̂ᵀ| ‖₂)`: the growth of the factors enters beside the matrix (Higham, *Accuracy and
/// Stability of Numerical Algorithms*, ch. 11). `matrix_norm` bounds `‖A‖₂` and `factor_magnitude` bounds the factor
/// term. By Weyl each eigenvalue of `A` lies within this band of one of `A + E`, whose signs the factorization counts.
pub fn symmetric_inertia_band(dim: usize, matrix_norm: f64, factor_magnitude: f64) -> f64 {
    dim as f64 * f64::EPSILON * (matrix_norm + factor_magnitude)
}
