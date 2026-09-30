//! Factored edits of a native linear use site, and the site's native read (#2951).
//!
//! A linear use site with native weight `W*` (`d_out × d_in`, the torch `Linear`
//! layout) is edited by a [`FactoredEdit`] `Σ_k u_k v_kᵀ`, which owns its factors and
//! is never formed as a `d_out × d_in` matrix. [`native_linear`] executes the site on
//! its original tensor.
//!
//! Rows are observations. The read streams rows in tiles sized by the library
//! row-chunk rule ([`byte_balanced_row_chunk`]) and reserves its result and tile
//! scratch on the process [`MemoryGovernor`] before allocating, so a request that
//! cannot fit is a typed refusal rather than an out-of-memory abort.

use gam_linalg::faer_ndarray::fast_ab_into;
use gam_runtime::resource::{
    Governed, MemoryGovernor, MemoryReservation, MemoryReservationError, byte_balanced_row_chunk,
    dense_f64_bytes,
};
use ndarray::{Array2, ArrayView2, s};

/// Refusal from a factored edit or a native read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// An operand's shape disagrees with the site. A vector's shape is `(len, 1)`.
    Shape {
        operand: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// An edit factor holds a non-finite entry.
    NonFinite { operand: &'static str },
    /// The byte footprint of the request does not fit in `usize`.
    SizeOverflow { context: &'static str },
    /// The process memory governor refused the request's footprint.
    Memory(MemoryReservationError),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape {
                operand,
                expected,
                found,
            } => write!(
                f,
                "structured edit refused: {operand} has shape {found:?}, the site needs {expected:?}"
            ),
            Self::NonFinite { operand } => {
                write!(f, "structured edit refused: {operand} holds a non-finite entry")
            }
            Self::SizeOverflow { context } => {
                write!(f, "{context}: the byte footprint overflows usize")
            }
            Self::Memory(err) => write!(f, "structured edit refused by the memory governor: {err}"),
        }
    }
}

impl std::error::Error for ApplyError {}

impl From<MemoryReservationError> for ApplyError {
    fn from(err: MemoryReservationError) -> Self {
        Self::Memory(err)
    }
}

/// A structured edit `Σ_k u_k v_kᵀ`, owning its factors.
#[derive(Clone, Debug)]
pub struct FactoredEdit {
    left: Array2<f64>,
    right: Array2<f64>,
}

impl FactoredEdit {
    /// `left` is `d_out × R` with columns `u_k`; `right` is `d_in × R` with
    /// columns `v_k`. Refuses mismatched ranks and non-finite entries.
    pub fn new(left: Array2<f64>, right: Array2<f64>) -> Result<Self, ApplyError> {
        expect_dim(
            "right factor",
            (right.nrows(), left.ncols()),
            right.dim(),
        )?;
        expect_finite("left factor", left.view())?;
        expect_finite("right factor", right.view())?;
        Ok(Self { left, right })
    }

    pub fn left(&self) -> ArrayView2<'_, f64> {
        self.left.view()
    }

    pub fn right(&self) -> ArrayView2<'_, f64> {
        self.right.view()
    }

    pub fn output_dim(&self) -> usize {
        self.left.nrows()
    }

    pub fn input_dim(&self) -> usize {
        self.right.nrows()
    }

    pub fn term_count(&self) -> usize {
        self.left.ncols()
    }
}

/// The bytes one native read holds: its result, and one row tile of the product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadFootprint {
    pub tile_rows: usize,
    pub result_bytes: usize,
    pub scratch_bytes: usize,
}

impl ReadFootprint {
    /// [`native_linear`] over `n_rows` rows into `output_dim` columns.
    pub fn native_linear(n_rows: usize, output_dim: usize) -> Result<Self, ApplyError> {
        const CONTEXT: &str = "native linear read";
        let tile_rows = byte_balanced_row_chunk(output_dim, n_rows);
        Ok(Self {
            tile_rows,
            result_bytes: f64_bytes(n_rows, output_dim, CONTEXT)?,
            scratch_bytes: f64_bytes(tile_rows, output_dim, CONTEXT)?,
        })
    }

    /// Charge the footprint to `governor` before allocating. The governor is an argument
    /// (#4565), so a read can be exercised against a budget of its own.
    fn reserve(
        &self,
        governor: &MemoryGovernor,
        context: &str,
    ) -> Result<(MemoryReservation, MemoryReservation), ApplyError> {
        let result = governor.try_reserve(self.result_bytes, context)?;
        let scratch = governor.try_reserve(self.scratch_bytes, context)?;
        Ok((result, scratch))
    }
}

/// A linear use site on its original tensor: `h W*ᵀ` over the input rows. The result's
/// charge moves into the returned value; the tile scratch is released on return.
pub fn native_linear(
    governor: &MemoryGovernor,
    native: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
) -> Result<Governed<Array2<f64>>, ApplyError> {
    expect_dim("input rows", (input.nrows(), native.ncols()), input.dim())?;
    let footprint = ReadFootprint::native_linear(input.nrows(), native.nrows())?;
    let (result, _scratch) = footprint.reserve(governor, "native linear read")?;
    Ok(result.bind(execute_native(native, input, footprint.tile_rows)))
}

fn execute_native(native: ArrayView2<'_, f64>, input: ArrayView2<'_, f64>, tile_rows: usize) -> Array2<f64> {
    let (n_rows, output_dim) = (input.nrows(), native.nrows());
    let native_transpose = native.t();
    let mut output = Array2::<f64>::zeros((n_rows, output_dim));
    let mut start = 0;
    while start < n_rows {
        let end = n_rows.min(start + tile_rows);
        let mut tile = Array2::<f64>::zeros((end - start, output_dim));
        fast_ab_into(&input.slice(s![start..end, ..]), &native_transpose, &mut tile);
        output.slice_mut(s![start..end, ..]).assign(&tile);
        start = end;
    }
    output
}

fn expect_dim(
    operand: &'static str,
    expected: (usize, usize),
    found: (usize, usize),
) -> Result<(), ApplyError> {
    if expected == found {
        Ok(())
    } else {
        Err(ApplyError::Shape {
            operand,
            expected,
            found,
        })
    }
}

fn expect_finite(operand: &'static str, values: ArrayView2<'_, f64>) -> Result<(), ApplyError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ApplyError::NonFinite { operand })
    }
}

fn f64_bytes(rows: usize, cols: usize, context: &'static str) -> Result<usize, ApplyError> {
    dense_f64_bytes(rows, cols).ok_or(ApplyError::SizeOverflow { context })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_governor;
    use gam_linalg::roundoff::{accumulation_band, accumulation_growth};
    use ndarray::Zip;

    fn fixture(rows: usize, cols: usize, phase: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i as f64 + 1.0) * (0.37 + phase) + (j as f64 + 1.0) * (0.61 - 0.5 * phase)).sin()
                + 0.25 * ((i * (j + 3)) as f64 * 0.113 + phase).cos()
        })
    }

    /// Output entry `(r, o)` of `h W*ᵀ` is `d_in` products and `d_in − 1` additions, so two
    /// routes to it each lie within `γ_(d_in) |h| |W*|ᵀ` of the exact value; `|h| |W*|ᵀ` is an
    /// accumulation of nonnegative terms through as many operations and is at most its computed
    /// value over `1 − γ_(d_in)`.
    #[test]
    fn a_native_read_is_the_direct_product_whatever_its_row_tiles() {
        let (n_rows, input_dim, output_dim) = (37, 48, 40);
        let native = fixture(output_dim, input_dim, 0.11);
        let input = fixture(n_rows, input_dim, 0.83);
        let direct = input.dot(&native.t());
        let absolute = input.mapv(f64::abs).dot(&native.mapv(f64::abs).t());
        let band = absolute
            .mapv(|sum| accumulation_band(input_dim, sum) / (1.0 - accumulation_growth(input_dim)));
        let read = native_linear(test_governor(), native.view(), input.view()).expect("a small read is admitted");
        let tiled = execute_native(native.view(), input.view(), 5);
        for (label, route) in [("governed read", &*read), ("five-row tiles", &tiled)] {
            assert!(
                Zip::from(route).and(&direct).and(&band).all(|&x, &y, &e| (x - y).abs() <= 2.0 * e),
                "{label} leaves the roundoff band of the direct product"
            );
        }
        // Positive control: a 1e-6 relative change in one weight is a different read.
        let mut perturbed = native.clone();
        perturbed[[3, 7]] *= 1.0 + 1e-6;
        let wrong = native_linear(test_governor(), perturbed.view(), input.view()).expect("admitted");
        assert!(
            !Zip::from(&*wrong).and(&direct).and(&band).all(|&x, &y, &e| (x - y).abs() <= 2.0 * e),
            "the band check cannot see a 1e-6 relative change in one weight"
        );
    }

    #[test]
    fn an_unfittable_request_is_refused_before_allocation() {
        // #4565: charged to this test's own governor, not the process-wide one.
        let governor = MemoryGovernor::with_budget_bytes(1 << 40);
        let small = ReadFootprint::native_linear(64, 40).expect("small footprint");
        assert!(
            small.reserve(&governor, "apply admission test").is_ok(),
            "a {small:?} footprint must be admitted (positive control)"
        );
        // 2^40 rows at LLM width is 2^55 bytes of output: no host holds it.
        let huge = ReadFootprint::native_linear(1 << 40, 4096).expect("the footprint itself fits usize");
        assert!(
            matches!(
                huge.reserve(&governor, "apply admission test"),
                Err(ApplyError::Memory(MemoryReservationError::BudgetExceeded { .. }))
            ),
            "a {huge:?} footprint was not refused by the memory governor"
        );
        assert!(
            matches!(ReadFootprint::native_linear(usize::MAX, 4096), Err(ApplyError::SizeOverflow { .. })),
            "an overflowing footprint was not refused"
        );
    }

    #[test]
    fn a_misshapen_or_non_finite_operand_is_refused() {
        let native = fixture(36, 40, 0.3);
        assert!(
            native_linear(test_governor(), native.view(), fixture(9, 40, 0.4).view()).is_ok(),
            "a well-shaped read must be admitted (positive control)"
        );
        assert_eq!(
            native_linear(test_governor(), native.view(), fixture(9, 39, 0.4).view()).err(),
            Some(ApplyError::Shape {
                operand: "input rows",
                expected: (9, 40),
                found: (9, 39),
            })
        );
        assert!(FactoredEdit::new(fixture(36, 3, 0.5), fixture(40, 3, 0.6)).is_ok());
        assert!(
            matches!(
                FactoredEdit::new(fixture(36, 3, 0.5), fixture(40, 2, 0.6)),
                Err(ApplyError::Shape { operand: "right factor", .. })
            ),
            "mismatched factor ranks were not refused"
        );
        let mut poisoned = fixture(40, 3, 0.6);
        poisoned[[7, 2]] = f64::NAN;
        assert_eq!(
            FactoredEdit::new(fixture(36, 3, 0.5), poisoned).err(),
            Some(ApplyError::NonFinite { operand: "right factor" })
        );
        let infinite = fixture(36, 3, 0.5).mapv(|entry| entry / 0.0);
        assert_eq!(
            FactoredEdit::new(infinite, fixture(40, 3, 0.6)).err(),
            Some(ApplyError::NonFinite { operand: "left factor" })
        );
    }
}
