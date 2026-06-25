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
