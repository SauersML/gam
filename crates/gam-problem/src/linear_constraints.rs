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
