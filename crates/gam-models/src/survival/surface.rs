//! Typed survival-surface interpolation and resource policy.
//!
//! Saved survival, cumulative-hazard, hazard, and standard-error matrices all
//! share one boundary law and one chunk geometry.  Frontends name the surface
//! kind; they do not supply numeric extrapolation constants or independently
//! decide when/how to tile the matrix.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

fn validate_dense_surface_shape(n_rows: usize, n_columns: usize) -> Result<(), String> {
    let cells = n_rows.checked_mul(n_columns).ok_or_else(|| {
        format!("survival surface shape {n_rows}x{n_columns} overflows the addressable cell count")
    })?;
    let bytes = cells
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| {
            format!("survival surface shape {n_rows}x{n_columns} overflows its byte count")
        })?;
    let cap = gam_runtime::resource::MemoryGovernor::global().single_materialization_cap_bytes();
    if bytes > cap {
        return Err(format!(
            "dense survival surface {n_rows}x{n_columns} requires {bytes} bytes, exceeding the current single-materialization cap of {cap} bytes; consume survival chunks or stream CSV instead"
        ));
    }
    Ok(())
}

/// Semantic kind of a saved survival surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvivalSurfaceKind {
    Survival,
    CumulativeHazard,
    Hazard,
    StandardError,
}

impl SurvivalSurfaceKind {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "survival" => Ok(Self::Survival),
            "cumulative_hazard" => Ok(Self::CumulativeHazard),
            "hazard" => Ok(Self::Hazard),
            "survival_se" => Ok(Self::StandardError),
            other => Err(format!("unknown survival surface kind '{other}'")),
        }
    }

    pub const fn policy(self) -> SurvivalSurfacePolicy {
        match self {
            Self::Survival => SurvivalSurfacePolicy {
                left_value: Some(1.0),
                right_value: None,
                positive_infinity_value: Some(0.0),
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
            Self::CumulativeHazard => SurvivalSurfacePolicy {
                left_value: Some(0.0),
                right_value: None,
                positive_infinity_value: Some(f64::INFINITY),
                lower_bound: Some(0.0),
                upper_bound: None,
            },
            Self::Hazard | Self::StandardError => SurvivalSurfacePolicy {
                left_value: None,
                right_value: None,
                positive_infinity_value: None,
                lower_bound: Some(0.0),
                upper_bound: None,
            },
        }
    }
}

/// Boundary and codomain law attached to a [`SurvivalSurfaceKind`].
///
/// A missing left/right value means flat endpoint continuation.  Positive
/// infinity is separate from finite right extrapolation: saved surfaces carry
/// no identified finite-time tail beyond their final knot, while survival and
/// cumulative hazard still have the mathematical limits 0 and +∞ at `t=+∞`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvivalSurfacePolicy {
    pub left_value: Option<f64>,
    pub right_value: Option<f64>,
    pub positive_infinity_value: Option<f64>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
}

impl SurvivalSurfacePolicy {
    #[inline]
    fn clamp(self, mut value: f64) -> f64 {
        if let Some(lower) = self.lower_bound
            && value < lower
        {
            value = lower;
        }
        if let Some(upper) = self.upper_bound
            && value > upper
        {
            value = upper;
        }
        value
    }
}

/// Borrowed, validated survival surface.
pub struct SurvivalSurface<'a> {
    kind: SurvivalSurfaceKind,
    grid: ArrayView1<'a, f64>,
    values: ArrayView2<'a, f64>,
    order: Vec<usize>,
}

impl<'a> SurvivalSurface<'a> {
    pub fn new(
        kind: SurvivalSurfaceKind,
        grid: ArrayView1<'a, f64>,
        values: ArrayView2<'a, f64>,
    ) -> Result<Self, String> {
        let (n_rows, n_knots) = values.dim();
        if n_knots == 0 || grid.len() != n_knots {
            return Err(format!(
                "survival surface requires a non-empty grid matching its columns; grid={}, surface={n_rows}x{n_knots}",
                grid.len()
            ));
        }
        for (index, &time) in grid.iter().enumerate() {
            if !time.is_finite() {
                return Err(format!(
                    "survival surface grid[{index}] must be finite, got {time}"
                ));
            }
        }
        let mut order: Vec<usize> = (0..grid.len()).collect();
        order.sort_by(|&left, &right| grid[left].total_cmp(&grid[right]));
        for pair in order.windows(2) {
            if grid[pair[1]] <= grid[pair[0]] {
                return Err(format!(
                    "survival surface grid values must be unique; grid[{}]={} duplicates grid[{}]={}",
                    pair[1], grid[pair[1]], pair[0], grid[pair[0]]
                ));
            }
        }
        Ok(Self {
            kind,
            grid,
            values,
            order,
        })
    }

    pub fn nrows(&self) -> usize {
        self.values.nrows()
    }

    /// Evaluate one row at one time under the kind's authoritative policy.
    pub fn value_at(&self, row: usize, query: f64) -> Result<f64, String> {
        if row >= self.values.nrows() {
            return Err(format!(
                "survival surface row {row} is out of bounds for {} rows",
                self.values.nrows()
            ));
        }
        if query.is_nan() {
            return Ok(f64::NAN);
        }
        let policy = self.kind.policy();
        let last = self.grid.len() - 1;
        let first_column = self.order[0];
        let last_column = self.order[last];
        let value = if query == f64::INFINITY {
            policy
                .positive_infinity_value
                .unwrap_or(self.values[[row, last_column]])
        } else if query < self.grid[first_column] {
            policy
                .left_value
                .unwrap_or(self.values[[row, first_column]])
        } else if query == self.grid[first_column] {
            self.values[[row, first_column]]
        } else if query > self.grid[last_column] {
            policy
                .right_value
                .unwrap_or(self.values[[row, last_column]])
        } else if query == self.grid[last_column] {
            self.values[[row, last_column]]
        } else {
            let upper = self
                .order
                .partition_point(|&column| self.grid[column] <= query);
            let lower = upper - 1;
            let lower_column = self.order[lower];
            let upper_column = self.order[upper];
            let x0 = self.grid[lower_column];
            let x1 = self.grid[upper_column];
            let y0 = self.values[[row, lower_column]];
            let y1 = self.values[[row, upper_column]];
            y0 + (query - x0) * (y1 - y0) / (x1 - x0)
        };
        Ok(policy.clamp(value))
    }

    /// Evaluate all rows on a shared query grid.
    pub fn interpolate(&self, query: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        let n_rows = self.values.nrows();
        let n_query = query.len();
        validate_dense_surface_shape(n_rows, n_query)?;
        let query_values = query.to_vec();
        let rows: Result<Vec<Vec<f64>>, String> = (0..n_rows)
            .into_par_iter()
            .map(|row| {
                query_values
                    .iter()
                    .map(|&time| self.value_at(row, time))
                    .collect()
            })
            .collect();
        Array2::from_shape_vec((n_rows, n_query), rows?.into_iter().flatten().collect())
            .map_err(|error| format!("failed to assemble survival surface: {error}"))
    }

    /// Evaluate through explicit tiles chosen by the shared resource policy.
    pub fn interpolate_chunked(
        &self,
        query: ArrayView1<'_, f64>,
        chunk_policy: SurvivalSurfaceChunkPolicy,
        people_chunk: Option<usize>,
        time_chunk: Option<usize>,
    ) -> Result<Array2<f64>, String> {
        validate_dense_surface_shape(self.values.nrows(), query.len())?;
        let chunks =
            chunk_policy.chunks(self.values.nrows(), query.len(), people_chunk, time_chunk)?;
        let mut output = Array2::<f64>::zeros((self.values.nrows(), query.len()));
        for chunk in chunks {
            for row in chunk.row_start..chunk.row_end {
                for time_index in chunk.time_start..chunk.time_end {
                    output[[row, time_index]] = self.value_at(row, query[time_index])?;
                }
            }
        }
        Ok(output)
    }
}

/// One half-open tile in a row × time surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurvivalSurfaceChunk {
    pub row_start: usize,
    pub row_end: usize,
    pub time_start: usize,
    pub time_end: usize,
}

/// Resource-derived chunk geometry for survival surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurvivalSurfaceChunkPolicy {
    target_cells: usize,
}

/// Convert a survival-probability matrix to failure probabilities.
pub fn failure_probability_from_survival(
    survival: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    validate_dense_surface_shape(survival.nrows(), survival.ncols())?;
    let values: Result<Vec<f64>, String> = survival
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            if value.is_nan() {
                return Ok(f64::NAN);
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "survival probability at flat index {index} must lie in [0, 1], got {value}"
                ));
            }
            Ok(1.0 - value)
        })
        .collect();
    Array2::from_shape_vec(survival.dim(), values?)
        .map_err(|error| format!("failed to assemble failure surface: {error}"))
}

impl Default for SurvivalSurfaceChunkPolicy {
    fn default() -> Self {
        let target_bytes =
            gam_runtime::resource::ResourcePolicy::default_library().row_chunk_target_bytes;
        Self {
            target_cells: (target_bytes / std::mem::size_of::<f64>()).max(1),
        }
    }
}

impl SurvivalSurfaceChunkPolicy {
    pub fn target_cells(self) -> usize {
        self.target_cells
    }

    /// Shape used when neither dimension is explicitly pinned.
    pub fn default_shape(self) -> (usize, usize) {
        let time = (self.target_cells as f64).sqrt().floor() as usize;
        let time = time.max(1);
        let people = (self.target_cells / time).max(1);
        (people, time)
    }

    pub fn should_chunk(self, n_rows: usize, n_times: usize) -> bool {
        n_rows.saturating_mul(n_times) > self.target_cells
    }

    pub fn resolve_shape(
        self,
        people: Option<usize>,
        times: Option<usize>,
    ) -> Result<(usize, usize), String> {
        if people == Some(0) {
            return Err("people_chunk must be positive".to_string());
        }
        if times == Some(0) {
            return Err("time_grid_chunk must be positive".to_string());
        }
        Ok(match (people, times) {
            (Some(people), Some(times)) => (people, times),
            (Some(people), None) => (people, (self.target_cells / people).max(1)),
            (None, Some(times)) => ((self.target_cells / times).max(1), times),
            (None, None) => self.default_shape(),
        })
    }

    pub fn chunks(
        self,
        n_rows: usize,
        n_times: usize,
        people: Option<usize>,
        times: Option<usize>,
    ) -> Result<Vec<SurvivalSurfaceChunk>, String> {
        let (people, times) = self.resolve_shape(people, times)?;
        let row_tiles = n_rows.div_ceil(people);
        let time_tiles = n_times.div_ceil(times);
        let mut chunks = Vec::with_capacity(row_tiles.saturating_mul(time_tiles));
        for row_start in (0..n_rows).step_by(people) {
            let row_end = (row_start + people).min(n_rows);
            for time_start in (0..n_times).step_by(times) {
                chunks.push(SurvivalSurfaceChunk {
                    row_start,
                    row_end,
                    time_start,
                    time_end: (time_start + times).min(n_times),
                });
            }
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn survival_and_cumulative_hazard_share_consistent_boundaries() {
        let grid = array![1.0, 2.0];
        let survival_values = array![[0.8, 0.5]];
        let cumulative_values = survival_values.mapv(|value| -value.ln());
        let survival = SurvivalSurface::new(
            SurvivalSurfaceKind::Survival,
            grid.view(),
            survival_values.view(),
        )
        .unwrap();
        let cumulative = SurvivalSurface::new(
            SurvivalSurfaceKind::CumulativeHazard,
            grid.view(),
            cumulative_values.view(),
        )
        .unwrap();
        for time in [-1.0, 1.0, 3.0] {
            let s = survival.value_at(0, time).unwrap();
            let h = cumulative.value_at(0, time).unwrap();
            assert!((s - (-h).exp()).abs() < 1e-15);
        }
        assert_eq!(survival.value_at(0, f64::INFINITY).unwrap(), 0.0);
        assert_eq!(
            cumulative.value_at(0, f64::INFINITY).unwrap(),
            f64::INFINITY
        );
    }

    #[test]
    fn chunk_defaults_come_from_the_runtime_byte_policy() {
        let policy = SurvivalSurfaceChunkPolicy::default();
        let (people, times) = policy.default_shape();
        assert!(people > 0 && times > 0);
        assert!(people.saturating_mul(times) <= policy.target_cells());
        assert!(policy.target_cells() - people * times < times);
    }

    #[test]
    fn explicit_chunk_dimension_derives_its_companion_from_the_same_budget() {
        let policy = SurvivalSurfaceChunkPolicy::default();
        let (people, times) = policy.resolve_shape(None, Some(8)).unwrap();
        assert_eq!(times, 8);
        assert_eq!(people, policy.target_cells() / 8);
    }

    #[test]
    fn unsorted_saved_grid_uses_the_same_typed_interpolant() {
        let grid = array![2.0, 0.0, 1.0];
        let values = array![[2.0, 0.0, 1.0]];
        let surface =
            SurvivalSurface::new(SurvivalSurfaceKind::Hazard, grid.view(), values.view()).unwrap();
        assert_eq!(surface.value_at(0, 0.5).unwrap(), 0.5);
        assert_eq!(surface.value_at(0, 1.5).unwrap(), 1.5);
    }

    #[test]
    fn dense_shape_overflow_is_rejected_before_allocation() {
        let error = validate_dense_surface_shape(usize::MAX, 2).unwrap_err();
        assert!(error.contains("overflows"), "unexpected error: {error}");
    }
}
