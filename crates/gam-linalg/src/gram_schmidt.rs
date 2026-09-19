//! Reorthogonalized Gram–Schmidt over rows, with the rank decision the
//! arithmetic can defend (#2469).
//!
//! [`ReorthogonalizedRowBasis`] takes rows one at a time. Each row is reduced
//! against the directions already held, in
//! [`GRAM_SCHMIDT_PASSES`](gam_math::roundoff::GRAM_SCHMIDT_PASSES) passes of
//! modified Gram–Schmidt. When admitted, it joins as the unit direction `q̂_j` of
//! its residual. The basis keeps:
//! - the triangular factor `R̂`, where `R̂_ij` is the total projection of admitted
//!   row `a_j` on `q̂_i` and `R̂_jj = ρ̂_j` is its residual norm;
//! - `Û`, the inverse of `R̂`, built one column per admission;
//! - each admitted row's admission band `β_j`.
//!
//! # When a row is independent
//!
//! Terms are kept to first order in `u`.
//! - Reducing a row `x` against `k` held directions rounds its residual by at most
//!   `β_x = gram_schmidt_residual_band(passes, k, dim, ‖x‖)`. That band's
//!   derivation counts every operation.
//! - So an admitted row satisfies `a_j + δa_j = Σ_{i≤j} R̂_ij q̂_i` with
//!   `‖δa_j‖ ≤ β_j`. The held directions span the perturbed rows `a_j + δa_j`
//!   exactly.
//! - Write a new row as `r = Σ_j w_j a_j + r⊥` over the admitted rows. Then
//!   `r = Σ_j w_j (a_j + δa_j) − Σ_j w_j δa_j + r⊥`, and the first sum lies in the
//!   directions' span. So the distance of `r` from that span is `‖r⊥‖` to within
//!   `Σ_j |w_j|·β_j`.
//! - The weights solve `R̂ w = ĉ`, where `ĉ` are the row's total projections on the
//!   directions. Back substitution returns `ŵ` with `(R̂ + ΔR) ŵ = ĉ` and
//!   `|ΔR| ≤ γ_k·|R̂|` (Higham, *ASNA*, Thm 8.5). So `|w − ŵ| ≤ γ_k·|R̂⁻¹||R̂||ŵ|`, and
//!   `Σ_j |w_j|·β_j ≤ βᵀ|ŵ| + γ_k·βᵀ|Û||R̂||ŵ|`. The second term is the charge for the
//!   solve itself. It is `κ`-dependent through `|Û||R̂|`, Skeel's componentwise
//!   condition, and it is charged, not assumed small.
//! - The computed residual norm `ρ̂_r` is therefore `‖r⊥‖` to within
//!   `β_r + βᵀ|ŵ| + γ_k·βᵀ|Û||R̂||ŵ|`. A row is independent only when `ρ̂_r` exceeds
//!   that sum. At or below it the arithmetic cannot tell the row from one in the
//!   span.
//!
//! Dropped as second order: the rounding in `Û` (it enters only the solve's charge,
//! which is already `O(u)`), the rounding of each normalization `q̂_j = residual/ρ̂_j`,
//! and the rounding of the band's own sums. Each is `O(u)` relative to a term that
//! is already `O(u)`.
//!
//! The weights carry the conditioning that `β_r` alone cannot see. Take a unit row
//! formed as a fourth difference of six near-collinear unit rows in eight
//! dimensions (#2600). Its weights are `(1, −4, 6, −4, 1, 0)` over the difference's
//! length `≈ 2e-3`, so `Σ_j |w_j| ≈ 8000`. Its own band `≈ 1.6e-14` would resolve any
//! residual above it, but the directions it is measured against are good only to
//! `βᵀ|ŵ| ≈ 4e-11`. The second pass keeps the directions orthonormal to working
//! precision. It does not move their span, so it does not shrink `βᵀ|ŵ|`.
//!
//! Rows are the caller's exact data. A zero or non-finite row is never admitted.

use gam_math::roundoff::{GRAM_SCHMIDT_PASSES, accumulation_growth, gram_schmidt_residual_band};
use ndarray::{Array1, Array2, ArrayView1};

/// Orthonormal directions of the rows admitted so far, with the factor and bands
/// that decide the next admission (see the module documentation).
#[derive(Clone, Debug, Default)]
pub struct ReorthogonalizedRowBasis {
    directions: Vec<Array1<f64>>,
    /// Column `j` of `R̂`: admitted row `j`'s projections on directions `0..j`,
    /// then its residual norm `ρ̂_j`.
    factor_columns: Vec<Vec<f64>>,
    /// Column `j` of `Û = R̂⁻¹`, entries `0..=j`.
    inverse_columns: Vec<Vec<f64>>,
    /// `β_j`: the rounding band of admitted row `j`'s own reduction.
    admission_bands: Vec<f64>,
}

impl ReorthogonalizedRowBasis {
    /// An empty basis.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of admitted rows, which is the number of directions.
    pub fn len(&self) -> usize {
        self.directions.len()
    }

    /// Whether no row has been admitted.
    pub fn is_empty(&self) -> bool {
        self.directions.is_empty()
    }

    /// The orthonormal directions, one per admitted row, in admission order.
    pub fn directions(&self) -> &[Array1<f64>] {
        &self.directions
    }

    /// The directions, consuming the basis.
    pub fn into_directions(self) -> Vec<Array1<f64>> {
        self.directions
    }

    /// The `k × k` upper-triangular `R̂` with `admitted rows = R̂ᵀ·directions` to
    /// within each row's admission band.
    pub fn triangular_factor(&self) -> Array2<f64> {
        let k = self.factor_columns.len();
        let mut factor = Array2::<f64>::zeros((k, k));
        for (column, entries) in self.factor_columns.iter().enumerate() {
            for (row, &value) in entries.iter().enumerate() {
                factor[[row, column]] = value;
            }
        }
        factor
    }

    /// Reduce `row` against the held directions and admit it when its residual
    /// is resolved: `ρ̂ > β_row + βᵀ|ŵ| + γ_k·βᵀ|Û||R̂||ŵ|` (see the module
    /// documentation). Returns whether it joined the basis.
    pub fn admit(&mut self, row: ArrayView1<'_, f64>) -> bool {
        let norm = row.dot(&row).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return false;
        }
        let held = self.directions.len();
        let own_band = gram_schmidt_residual_band(GRAM_SCHMIDT_PASSES, held, row.len(), norm);
        let mut residual = row.to_owned();
        let mut projections = vec![0.0_f64; held];
        for _ in 0..GRAM_SCHMIDT_PASSES {
            for (direction, projection) in self.directions.iter().zip(projections.iter_mut()) {
                let step = residual.dot(direction);
                residual.scaled_add(-step, direction);
                *projection += step;
            }
        }
        let residual_norm = residual.dot(&residual).sqrt();
        if !residual_norm.is_finite() {
            return false;
        }
        // Back substitution `R̂ ŵ = ĉ`, one column of `R̂` at a time.
        let mut weights = projections.clone();
        for column in (0..held).rev() {
            let entries = &self.factor_columns[column];
            weights[column] /= entries[column];
            let weight = weights[column];
            for (above, &entry) in weights.iter_mut().zip(entries.iter()).take(column) {
                *above -= entry * weight;
            }
        }
        // The solve's charge `γ_k·βᵀ|Û||R̂||ŵ|`: first `|R̂||ŵ|`, then `|Û|` of it.
        let mut factor_times_weights = vec![0.0_f64; held];
        for (entries, weight) in self.factor_columns.iter().zip(weights.iter()) {
            for (sum, &entry) in factor_times_weights.iter_mut().zip(entries.iter()) {
                *sum += entry.abs() * weight.abs();
            }
        }
        let mut inverse_times = vec![0.0_f64; held];
        for (entries, &value) in self.inverse_columns.iter().zip(factor_times_weights.iter()) {
            for (sum, &entry) in inverse_times.iter_mut().zip(entries.iter()) {
                *sum += entry.abs() * value;
            }
        }
        let solve_growth = accumulation_growth(held);
        let inherited: f64 = self
            .admission_bands
            .iter()
            .zip(weights.iter().zip(inverse_times.iter()))
            .map(|(band, (weight, charge))| band * (weight.abs() + solve_growth * charge))
            .sum();
        if !(residual_norm > own_band + inherited) {
            return false;
        }
        // Column `k` of `Û` for `R̂_new = [[R̂, ĉ], [0, ρ̂]]` is `[−ŵ/ρ̂; 1/ρ̂]`.
        let mut inverse_column: Vec<f64> = weights.iter().map(|weight| -weight / residual_norm).collect();
        inverse_column.push(1.0 / residual_norm);
        projections.push(residual_norm);
        self.factor_columns.push(projections);
        self.inverse_columns.push(inverse_column);
        self.admission_bands.push(own_band);
        self.directions.push(residual / residual_norm);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::ReorthogonalizedRowBasis;
    use gam_math::roundoff::{GRAM_SCHMIDT_PASSES, gram_schmidt_residual_band};
    use ndarray::{Array1, Array2, array};

    /// A residual resolved above the row's own band joins, since exact rows carry
    /// no inherited error. After `e₁`, `(1, 1e-14, 0)` leaves `1e-14`, above
    /// `2·γ₇ ≈ 1.6e-15`. A row in the span, a zero row and a non-finite row do not
    /// join. `R̂` reproduces the rows, and `Û` inverts it.
    #[test]
    fn a_residual_above_its_band_joins_and_the_factor_reproduces_the_rows_2469() {
        let gap = 1.0e-14_f64;
        let band = gram_schmidt_residual_band(GRAM_SCHMIDT_PASSES, 1, 3, 1.0);
        assert!(gap > band, "fixture premise: {gap:.1e} above the band {band:.3e}");
        let mut basis = ReorthogonalizedRowBasis::new();
        assert!(basis.admit(array![2.0_f64, 0.0, 0.0].view()));
        assert!(basis.admit(array![2.0_f64, 2.0 * gap, 0.0].view()));
        assert!(!basis.admit(array![0.6_f64, 0.0, 0.0].view()), "a row in the span is dependent");
        assert!(!basis.admit(array![0.0_f64, 0.0, 0.0].view()));
        assert!(!basis.admit(array![f64::NAN, 0.0, 0.0].view()));
        assert_eq!(basis.len(), 2);
        let factor = basis.triangular_factor();
        assert_eq!(factor[[1, 0]], 0.0);
        assert!((factor[[0, 0]] - 2.0).abs() <= 2.0 * f64::EPSILON);
        assert!((factor[[1, 1]] - 2.0 * gap).abs() <= band);
        let rows = factor.t().dot(&Array2::from_shape_fn((2, 3), |(i, j)| basis.directions()[i][j]));
        assert!((rows[[1, 1]] - 2.0 * gap).abs() <= band);
        let inverse = Array2::from_shape_fn((2, 2), |(i, j)| {
            basis.inverse_columns[j].get(i).copied().unwrap_or(0.0)
        });
        let identity = factor.dot(&inverse);
        assert!((identity[[0, 0]] - 1.0).abs() <= 2.0 * f64::EPSILON);
        assert!((identity[[1, 1]] - 1.0).abs() <= 2.0 * f64::EPSILON);
        assert!(identity[[0, 1]].abs() <= 2.0 * f64::EPSILON, "{identity:?}");
    }

    /// The six #2600 rows: unit rows of `(1, x, …, x⁷)` at nodes `1, 1.05, …, 1.25`,
    /// then their unit-normalized fourth difference, which is in their span by
    /// construction.
    fn near_collinear_face_2600() -> Array2<f64> {
        let p = 8usize;
        let mut rows = Array2::<f64>::zeros((7, p));
        for index in 0..6 {
            let node = 1.0 + 0.05 * index as f64;
            let mut power = 1.0_f64;
            for column in 0..p {
                rows[[index, column]] = power;
                power *= node;
            }
            let norm = rows.row(index).dot(&rows.row(index)).sqrt();
            rows.row_mut(index).mapv_inplace(|value| value / norm);
        }
        let mut difference = Array1::<f64>::zeros(p);
        for (index, weight) in [1.0_f64, -4.0, 6.0, -4.0, 1.0].iter().enumerate() {
            difference.scaled_add(*weight, &rows.row(index));
        }
        let length = difference.dot(&difference).sqrt();
        rows.row_mut(6).assign(&(&difference / length));
        rows
    }

    /// On the #2600 face the six near-collinear rows join, and their fourth
    /// difference does not. Its weights on the six are near `(1, −4, 6, −4, 1, 0)/2e-3`,
    /// so the directions it is measured against are good only to `βᵀ|ŵ|`, orders
    /// above its own band.
    #[test]
    fn a_difference_of_near_collinear_rows_is_dependent_2600_2469() {
        let rows = near_collinear_face_2600();
        let mut basis = ReorthogonalizedRowBasis::new();
        for index in 0..6 {
            assert!(basis.admit(rows.row(index)), "row {index} is independent");
        }
        assert!(!basis.admit(rows.row(6)), "the fourth difference is in the span");
        assert_eq!(basis.len(), 6);
    }
}
