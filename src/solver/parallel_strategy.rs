//! Workload-aware parallel scheduling for row reductions.
//!
//! This module deliberately does not size or replace Rayon's global pool.
//! The pool is process infrastructure; each numerical kernel still owns the
//! decision of how much parallelism is useful for its arithmetic shape.

/// Minimum useful arithmetic per row-reduction task. Below this, Rayon tasks
/// are mostly scheduler/reduction traffic rather than useful work.
const TARGET_WORK_PER_TASK: usize = 16_000_000;

/// Keep at least this many rows in one task even for very heavy row kernels.
/// Smaller chunks make progress logs prettier but generally hurt cache locality
/// and reduction overhead.
const MIN_ROWS_PER_TASK: usize = 512;

/// Avoid creating enormous chunks for thin row kernels; if a reduction is
/// worth parallelizing at all, these chunk sizes keep enough tasks available.
const MAX_ROWS_PER_TASK: usize = 16_384;

/// Maximum number of row-reduction tasks per worker. More tasks improve load
/// balance when row costs are uneven, but beyond this the reducer is dominated
/// by tiny partial matrices/vectors.
const MAX_TASKS_PER_WORKER: usize = 4;

/// Return a row chunk size for a parallel row reduction, or `None` when the
/// caller should stay serial.
///
/// `row_work_units` is an operation-local relative cost estimate for one row.
/// It need not be exact; it only separates cheap row scaling / Gram updates
/// from expensive row kernels that solve roots, evaluate special functions, or
/// assemble high-order jets. `reduction_cells` is the number of f64 cells in one
/// per-task accumulator, used to avoid creating many large partials.
pub(crate) fn row_reduction_chunk_rows(
    n_rows: usize,
    row_work_units: usize,
    reduction_cells: usize,
    min_parallel_work: usize,
) -> Option<usize> {
    if n_rows == 0 || row_work_units == 0 {
        return None;
    }
    let workers = rayon::current_num_threads();
    let total_work = n_rows.saturating_mul(row_work_units);
    if workers <= 1 || total_work < min_parallel_work {
        return None;
    }

    let min_rows_by_work = TARGET_WORK_PER_TASK
        .div_ceil(row_work_units.max(1))
        .clamp(MIN_ROWS_PER_TASK, MAX_ROWS_PER_TASK);
    let tasks_by_rows = n_rows.div_ceil(min_rows_by_work).max(1);
    if tasks_by_rows <= 1 {
        return None;
    }

    let task_cap_by_workers = workers.saturating_mul(MAX_TASKS_PER_WORKER).max(1);
    let task_cap_by_reduction = reduction_task_cap(reduction_cells);
    let tasks = tasks_by_rows
        .min(task_cap_by_workers)
        .min(task_cap_by_reduction)
        .max(1);
    if tasks <= 1 {
        return None;
    }
    Some(n_rows.div_ceil(tasks).max(1))
}

/// Number of chunks that [`row_reduction_chunk_rows`] will create for `n_rows`.
pub(crate) fn row_reduction_chunk_count(n_rows: usize, chunk_rows: usize) -> usize {
    if n_rows == 0 {
        0
    } else {
        n_rows.div_ceil(chunk_rows.max(1))
    }
}

fn reduction_task_cap(reduction_cells: usize) -> usize {
    let bytes = reduction_cells.saturating_mul(std::mem::size_of::<f64>());
    if bytes <= 64 * 1024 {
        usize::MAX
    } else if bytes <= 1024 * 1024 {
        128
    } else if bytes <= 8 * 1024 * 1024 {
        32
    } else {
        8
    }
}
