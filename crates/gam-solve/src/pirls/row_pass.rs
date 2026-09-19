//! Parallel certified row passes without per-row error storage.
//!
//! Every PIRLS row oracle returns `Result<Row, EstimationError>`. Collecting
//! those into a `Vec<Result<Row, _>>` before scanning for the first failure
//! costs `size_of::<Result<Row, EstimationError>>()` bytes per row — dominated
//! by the error payload, not by the row — on every working-state update,
//! deviance evaluation and log-kernel evaluation. At large `n` that transient
//! is many times the size of an `n`-vector and is the main source of allocator
//! grow/trim churn. These passes write each certified row straight into its
//! slot and keep only the smallest failing row's error.

use crate::estimate::EstimationError;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

/// Evaluate `row(i)` for every slot of `out` in parallel.
///
/// On success every slot holds its certified row. On failure the error of the
/// smallest failing row index is returned, independent of scheduling; the
/// slot contents are then unspecified, so callers that promise atomic output
/// must pass scratch storage rather than their caller-visible buffers.
pub(crate) fn par_rows_into<T, F>(out: &mut [T], row: F) -> Result<(), EstimationError>
where
    T: Send,
    F: Fn(usize) -> Result<T, EstimationError> + Sync + Send,
{
    let first_failure = out
        .par_iter_mut()
        .enumerate()
        .find_map_first(|(i, slot)| match row(i) {
            Ok(value) => {
                *slot = value;
                None
            }
            Err(error) => Some(error),
        });
    match first_failure {
        None => Ok(()),
        Some(error) => Err(error),
    }
}

/// Certify `n` rows in parallel into a freshly owned vector of rows.
///
/// The only allocation is the `n * size_of::<T>()` result; no caller-visible
/// output exists unless every row certifies, and a failure reports the
/// smallest failing row.
pub(crate) fn par_certified_rows<T, F>(n: usize, row: F) -> Result<Vec<T>, EstimationError>
where
    T: Send + Clone + Default,
    F: Fn(usize) -> Result<T, EstimationError> + Sync + Send,
{
    let mut rows = vec![T::default(); n];
    par_rows_into(&mut rows, row)?;
    Ok(rows)
}
