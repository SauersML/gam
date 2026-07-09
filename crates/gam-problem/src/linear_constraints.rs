use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct LinearInequalityConstraints {
    pub a: Array2<f64>,
    pub b: Array1<f64>,
}

impl LinearInequalityConstraints {
    /// Construct with the equal-row-count invariant enforced. The dimensions
    /// `a.nrows() == b.len()` are required by every downstream KKT / active-set
    /// routine; routing every construction site through this constructor
    /// eliminates a class of "rows out of sync" bugs at the type boundary.
    #[inline]
    pub fn new(a: Array2<f64>, b: Array1<f64>) -> Result<Self, String> {
        if a.nrows() != b.len() {
            return Err(format!(
                "LinearInequalityConstraints: row count mismatch (A has {} rows, b has length {})",
                a.nrows(),
                b.len(),
            ));
        }
        if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
            return Err(
                "LinearInequalityConstraints: A and b must be finite (a NaN row silently \
                 evades every feasibility comparison downstream)"
                    .to_string(),
            );
        }
        Ok(Self { a, b })
    }

    /// Canonicalize the system `Aβ ≥ b` into the scale-free form every
    /// downstream tolerance assumes, PRESERVING row count and order (so cached
    /// active-set row indices and warm-start hints remain valid):
    ///
    /// * non-finite entries are rejected (a NaN row compares as neither
    ///   feasible nor infeasible and silently evades active-set logic);
    /// * an exactly-zero row `0ᵀβ ≥ b_i` is INFEASIBLE for `b_i > 0`
    ///   (rejected loudly); for `b_i ≤ 0` it is vacuous and kept verbatim —
    ///   its geometric slack is `+∞`, so it can never activate downstream;
    /// * every nonzero row is normalized to unit norm, `(a_i, b_i)/‖a_i‖`, so
    ///   `b_i` becomes the signed distance of the constraint hyperplane from
    ///   the origin and every absolute slack / violation / rank tolerance
    ///   applied later is automatically scale-relative: `1e-20·β ≥ 1e-20` and
    ///   `β ≥ 1` canonicalize to the same row, as they are the same
    ///   half-space.
    pub fn canonicalized(&self) -> Result<Self, String> {
        let m = self.a.nrows();
        if self.b.len() != m {
            return Err(format!(
                "LinearInequalityConstraints: row count mismatch (A has {m} rows, b has length {})",
                self.b.len(),
            ));
        }
        if self.a.iter().any(|v| !v.is_finite()) || self.b.iter().any(|v| !v.is_finite()) {
            return Err("LinearInequalityConstraints: A and b must be finite".to_string());
        }
        let mut a = self.a.clone();
        let mut b = self.b.clone();
        for i in 0..m {
            let norm = a.row(i).dot(&a.row(i)).sqrt();
            if norm > 0.0 {
                a.row_mut(i).mapv_inplace(|v| v / norm);
                b[i] /= norm;
            } else if b[i] > 0.0 {
                return Err(format!(
                    "LinearInequalityConstraints: row {i} is zero with positive bound \
                     b = {:.3e}; the constraint 0ᵀβ ≥ b is infeasible",
                    b[i],
                ));
            }
        }
        Ok(Self { a, b })
    }

    /// Build the per-coordinate `β_i >= lower_bounds[i]` inequality system.
    /// Non-finite entries are treated as "no bound" and skipped; returns
    /// `None` when every entry is non-finite so callers can short-circuit
    /// the no-constraint case without allocating the empty A/b pair.
    pub fn from_per_coordinate_lower_bounds(lower_bounds: &Array1<f64>) -> Option<Self> {
        let active_rows: Vec<usize> = (0..lower_bounds.len())
            .filter(|&i| lower_bounds[i].is_finite())
            .collect();
        if active_rows.is_empty() {
            return None;
        }
        let p = lower_bounds.len();
        let mut a = Array2::<f64>::zeros((active_rows.len(), p));
        let mut b = Array1::<f64>::zeros(active_rows.len());
        for (r, &idx) in active_rows.iter().enumerate() {
            a[[r, idx]] = 1.0;
            b[r] = lower_bounds[idx];
        }
        Some(Self { a, b })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn new_ok_when_rows_match_b_len() {
        let a = Array2::<f64>::eye(3);
        let b = Array1::<f64>::zeros(3);
        assert!(LinearInequalityConstraints::new(a, b).is_ok());
    }

    #[test]
    fn new_err_on_row_count_mismatch() {
        let a = Array2::<f64>::eye(3);
        let b = Array1::<f64>::zeros(2);
        assert!(LinearInequalityConstraints::new(a, b).is_err());
    }

    #[test]
    fn from_lower_bounds_none_when_all_non_finite() {
        let bounds = array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        assert!(LinearInequalityConstraints::from_per_coordinate_lower_bounds(&bounds).is_none());
    }

    #[test]
    fn from_lower_bounds_selects_finite_entries() {
        // bounds = [NaN, 1.0, NaN] → one active constraint: β₁ ≥ 1.0
        let bounds = array![f64::NAN, 1.0_f64, f64::NAN];
        let c = LinearInequalityConstraints::from_per_coordinate_lower_bounds(&bounds).unwrap();
        assert_eq!(c.a.nrows(), 1);
        assert_eq!(c.a.ncols(), 3);
        assert_eq!(c.a[[0, 1]], 1.0);
        assert_eq!(c.b[0], 1.0);
    }

    #[test]
    fn canonicalized_is_invariant_to_row_rescaling() {
        // 1e-20·β ≥ 1e-20 is the same half-space as β ≥ 1 and must canonicalize
        // to the identical unit row.
        let tiny = LinearInequalityConstraints {
            a: array![[1e-20_f64]],
            b: array![1e-20_f64],
        }
        .canonicalized()
        .unwrap();
        assert!((tiny.a[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((tiny.b[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn canonicalized_rejects_infeasible_zero_row() {
        // 0·β ≥ 1 is impossible and must fail loudly, not vanish.
        let c = LinearInequalityConstraints {
            a: array![[0.0_f64, 0.0]],
            b: array![1.0_f64],
        };
        assert!(c.canonicalized().is_err());
    }

    #[test]
    fn canonicalized_keeps_row_indices_and_normalizes_nonzero_rows() {
        // Row order/count are preserved (warm active-set ids stay valid); the
        // vacuous zero row is kept verbatim, the real row is unit-normalized.
        let c = LinearInequalityConstraints {
            a: array![[0.0_f64, 0.0], [3.0, 4.0]],
            b: array![-2.0_f64, 10.0],
        };
        let canon = c.canonicalized().unwrap();
        assert_eq!(canon.a.nrows(), 2);
        assert_eq!(canon.a[[0, 0]], 0.0);
        assert_eq!(canon.b[0], -2.0);
        assert!((canon.a[[1, 0]] - 0.6).abs() < 1e-15);
        assert!((canon.a[[1, 1]] - 0.8).abs() < 1e-15);
        assert!((canon.b[1] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn canonicalized_rejects_nan_rows() {
        let c = LinearInequalityConstraints {
            a: array![[f64::NAN, 0.0]],
            b: array![0.0_f64],
        };
        assert!(c.canonicalized().is_err());
    }

    #[test]
    fn new_rejects_non_finite_entries() {
        let a = array![[f64::NAN]];
        let b = array![0.0_f64];
        assert!(LinearInequalityConstraints::new(a, b).is_err());
    }

    #[test]
    fn from_lower_bounds_multiple_active_rows() {
        let bounds = array![0.5_f64, f64::NAN, -1.0];
        let c = LinearInequalityConstraints::from_per_coordinate_lower_bounds(&bounds).unwrap();
        assert_eq!(c.a.nrows(), 2);
        // First row: col 0 active with bound 0.5
        assert_eq!(c.a[[0, 0]], 1.0);
        assert_eq!(c.b[0], 0.5);
        // Second row: col 2 active with bound -1.0
        assert_eq!(c.a[[1, 2]], 1.0);
        assert_eq!(c.b[1], -1.0);
    }
}
