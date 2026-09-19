//! The metric Gram of a known block's writers, cached once per (block, M) (#2946).
//!
//! # Why a cache
//!
//! Every pair pass reads `D_jk = u_jᵀ M u_k`. It depends on neither the retained frame nor the declared context, yet a
//! streaming pass rebuilds it per call: `h² p` flops, the dominant phase of a gradient call at LLM width (8–11 s of
//! 11–15 s at `h = 12288`, `p = 4096`; #2946 comment 5719097233). [`ReaderGram`] builds it once and hands out rows.
//!
//! # What is stored
//!
//! Only `j ≤ k`, packed row-major, so row `j` is the contiguous slice `D[j, j..h]`: `h (h + 1)/2` entries, 0.6 GB at
//! `h = 12288`. The pair pass sweeps the same half.
//!
//! # One tile routine for both routes
//!
//! [`fill_upper_rows`] writes one tile `j ∈ [s, e)` of packed upper rows from one product `U_Jᵀ (M U)[:, s..]`, row `j`
//! contributing its columns `k ≥ j`, over the tiles [`upper_tile_rows`] draws. [`ReaderGram::new`] is that routine
//! over every tile, and a pass that streams instead of caching calls the same routine over the same tiles, so the cached
//! and streamed rows are the same words. The products run at the pool's degree through `gam_linalg`, whose products
//! give the same words at every degree and pool width, so neither route depends on the thread count.
//!
//! # Memory
//!
//! The footprint is charged to the process-wide [`MemoryGovernor`] before anything is allocated, and the reservation
//! is bound to the storage, so the two cannot come apart. A footprint the ledger cannot admit is refused typed with
//! the ledger's evidence, and the caller streams instead; the cache never makes a block unevaluable and never allocates
//! what the ledger declined.

use gam_linalg::faer_ndarray::fast_atb;
use gam_runtime::resource::{Governed, MemoryGovernor, MemoryReservationError, byte_balanced_row_chunk};
use ndarray::{ArrayView2, s};
use std::fmt;

/// A refusal of the reader Gram.
#[derive(Debug, Clone, PartialEq)]
pub enum ReaderGramError {
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinite { context: &'static str },
    /// A tile `[start, end)` outside `[0, width]`, or with `start > end`.
    InvalidTile { start: usize, end: usize, width: usize },
    /// `h (h + 1)/2` entries of `f64` does not fit a `usize` byte count.
    SizeOverflow { width: usize },
    /// The process-wide ledger cannot admit the footprint; the caller streams instead.
    Admission { error: MemoryReservationError },
}

impl fmt::Display for ReaderGramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected {expected}, got {got}"),
            Self::NonFinite { context } => write!(f, "{context} holds a non-finite entry"),
            Self::InvalidTile { start, end, width } => {
                write!(f, "reader Gram tile [{start}, {end}) is not a tile of width {width}")
            }
            Self::SizeOverflow { width } => {
                write!(f, "a packed reader Gram of width {width} has no representable byte count")
            }
            Self::Admission { error } => write!(f, "the reader Gram was not admitted: {error}"),
        }
    }
}

impl std::error::Error for ReaderGramError {}

/// `D = Uᵀ M U` over `j ≤ k`, packed row-major, with its live memory charged to the process-wide ledger.
#[derive(Debug)]
pub struct ReaderGram {
    width: usize,
    packed: Governed<Vec<f64>>,
}

impl ReaderGram {
    /// Build the Gram of writers `U` (`p × h`) under the metric, given `metric_writers = M U` (`p × h`).
    pub fn new(writers: ArrayView2<'_, f64>, metric_writers: ArrayView2<'_, f64>) -> Result<Self, ReaderGramError> {
        let width = writers.ncols();
        require_shapes(writers, metric_writers)?;
        let bytes = packed_bytes(width).ok_or(ReaderGramError::SizeOverflow { width })?;
        let reservation = MemoryGovernor::global()
            .try_reserve(bytes, "reader Gram D = Uᵀ M U over j ≤ k")
            .map_err(|error| ReaderGramError::Admission { error })?;
        require_finite("reader Gram writers", writers.iter())?;
        require_finite("reader Gram metric writers", metric_writers.iter())?;
        let mut packed = reservation.bind(vec![0.0; bytes / std::mem::size_of::<f64>()]);
        let tile = upper_tile_rows(width);
        for start in (0..width).step_by(tile) {
            let end = (start + tile).min(width);
            fill_upper_rows(
                writers,
                metric_writers,
                start,
                end,
                &mut packed[row_offset(width, start)..row_offset(width, end)],
            )?;
        }
        Ok(Self { width, packed })
    }

    /// The width `h`.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The bytes this cache holds against the process-wide ledger.
    pub fn resident_bytes(&self) -> usize {
        self.packed.reserved_bytes()
    }

    /// Row `unit` of `D` over `k ≥ unit`: entry `0` is `D_jj`, entry `k − unit` is `D_jk`.
    pub fn upper_row(&self, unit: usize) -> &[f64] {
        let offset = row_offset(self.width, unit);
        &self.packed[offset..offset + self.width - unit]
    }
}

/// The rows per tile of the reader Gram's products, for a block of width `width`: the library row-chunk rule for a
/// `t × width` product, so a tile's product stays near the library row target. A pass that streams `D` uses the same
/// tiles, so its rows are the cached rows' words.
pub fn upper_tile_rows(width: usize) -> usize {
    byte_balanced_row_chunk(width, width)
}

/// Write rows `j ∈ [start, end)` of `D = Uᵀ M U` over `k ≥ j`, packed in row order, into `out`, from one product
/// `U_Jᵀ (M U)[:, start..]`. `out` holds `Σ_{j ∈ [start, end)} (h − j)` entries. Shapes and lengths are checked here;
/// finiteness is the caller's, checked once per block.
pub fn fill_upper_rows(
    writers: ArrayView2<'_, f64>,
    metric_writers: ArrayView2<'_, f64>,
    start: usize,
    end: usize,
    out: &mut [f64],
) -> Result<(), ReaderGramError> {
    require_shapes(writers, metric_writers)?;
    let width = writers.ncols();
    if start > end || end > width {
        return Err(ReaderGramError::InvalidTile { start, end, width });
    }
    require_length(
        "reader Gram tile output",
        row_offset(width, end) - row_offset(width, start),
        out.len(),
    )?;
    if start == end {
        return Ok(());
    }
    let product = fast_atb(
        &writers.slice(s![.., start..end]),
        &metric_writers.slice(s![.., start..]),
    );
    let mut cursor = 0;
    for unit in start..end {
        let row = product.row(unit - start);
        let length = width - unit;
        for (slot, &entry) in out[cursor..cursor + length].iter_mut().zip(row.iter().skip(unit - start)) {
            *slot = entry;
        }
        cursor += length;
    }
    Ok(())
}

/// `Σ_{i<unit} (width − i)`, the packed start of row `unit`; `row_offset(width, width)` is the packed length.
fn row_offset(width: usize, unit: usize) -> usize {
    unit * width - unit * unit.saturating_sub(1) / 2
}

/// `h (h + 1)/2 · 8`, or `None` when it does not fit a `usize`.
fn packed_bytes(width: usize) -> Option<usize> {
    let cells = if width % 2 == 0 {
        (width / 2).checked_mul(width.checked_add(1)?)?
    } else {
        width.checked_mul(width.checked_add(1)? / 2)?
    };
    cells.checked_mul(std::mem::size_of::<f64>())
}

fn require_shapes(writers: ArrayView2<'_, f64>, metric_writers: ArrayView2<'_, f64>) -> Result<(), ReaderGramError> {
    require_length("reader Gram metric writer rows", writers.nrows(), metric_writers.nrows())?;
    require_length("reader Gram metric writer columns", writers.ncols(), metric_writers.ncols())
}

fn require_length(context: &'static str, expected: usize, got: usize) -> Result<(), ReaderGramError> {
    if expected == got {
        Ok(())
    } else {
        Err(ReaderGramError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

fn require_finite<'a>(
    context: &'static str,
    mut values: impl Iterator<Item = &'a f64>,
) -> Result<(), ReaderGramError> {
    if values.all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ReaderGramError::NonFinite { context })
    }
}

#[cfg(test)]
#[path = "reader_gram_tests.rs"]
mod reader_gram_tests;
