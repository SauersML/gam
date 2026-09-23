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

/// Return a row chunk size for a parallel row reduction, or `None` when the
/// caller should stay serial.
///
/// `row_work_units` is an operation-local relative cost estimate for one row.
/// It need not be exact; it only separates cheap row scaling / Gram updates
/// from expensive row kernels that solve roots, evaluate special functions, or
/// assemble high-order jets. `reduction_cells` is the number of f64 cells in one
/// per-task accumulator, used to avoid creating many large partials.
///
/// **The chunking is a function of the shape alone.** A reduction sums its
/// per-chunk partials, so the chunk boundaries fix the summation order; a
/// chunking read from the pool width made every such reduction's bits a
/// function of `RAYON_NUM_THREADS`, and made a one-thread pool take a different
/// (unchunked) path from every wider one. Sizing a chunk by the work it carries
/// already leaves every worker tasks to steal once the reduction is large enough
/// to be worth splitting; a one-thread pool runs the same chunks in order.
pub fn row_reduction_chunk_rows(
    n_rows: usize,
    row_work_units: usize,
    reduction_cells: usize,
    min_parallel_work: usize,
) -> Option<usize> {
    if n_rows == 0 || row_work_units == 0 {
        return None;
    }
    let total_work = n_rows.saturating_mul(row_work_units);
    if total_work < min_parallel_work {
        return None;
    }

    let min_rows_by_work = TARGET_WORK_PER_TASK
        .div_ceil(row_work_units.max(1))
        .clamp(MIN_ROWS_PER_TASK, MAX_ROWS_PER_TASK);
    let tasks_by_rows = n_rows.div_ceil(min_rows_by_work).max(1);
    let tasks = tasks_by_rows.min(reduction_task_cap(reduction_cells));
    if tasks <= 1 {
        return None;
    }
    Some(n_rows.div_ceil(tasks).max(1))
}

/// Rows per block for a product that contracts a long row axis into a small
/// output (`AᵀB`, `Aᵀ·diag(w)·B`), or `None` when the output is too large for
/// the row split and the product belongs to faer's output-tiled GEMM.
///
/// Each block is one task: a sequential GEMM of its rows into a private
/// `output_cells` partial, the partials combined over a fixed pairwise tree.
/// The block is sized so one task carries `TARGET_WORK_PER_TASK` of
/// multiply-adds (`rows · output_cells`), within the same row band every other
/// row reduction uses. The split applies only while a `MIN_ROWS_PER_TASK`
/// block still fits inside that budget: beyond it the output is wide enough
/// that faer's own tiling has output tiles for every worker, and partials of
/// that size would be the reduction traffic this module exists to avoid.
///
/// The answer is a function of the shape alone — never of the pool width, the
/// nesting depth or the caller's degree — so the summation order, and with it
/// every bit of the product, is the same at every thread count.
pub fn row_contraction_block_rows(output_cells: usize) -> Option<usize> {
    if output_cells == 0 || output_cells > TARGET_WORK_PER_TASK / MIN_ROWS_PER_TASK {
        return None;
    }
    Some(
        TARGET_WORK_PER_TASK
            .div_ceil(output_cells)
            .clamp(MIN_ROWS_PER_TASK, MAX_ROWS_PER_TASK),
    )
}

/// Number of chunks that [`row_reduction_chunk_rows`] will create for `n_rows`.
pub fn row_reduction_chunk_count(n_rows: usize, chunk_rows: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_rows_zero_n_rows_returns_none() {
        assert_eq!(row_reduction_chunk_rows(0, 100, 1, 1), None);
    }

    #[test]
    fn chunk_rows_zero_work_units_returns_none() {
        assert_eq!(row_reduction_chunk_rows(1000, 0, 1, 1), None);
    }

    #[test]
    fn chunk_rows_below_min_parallel_work_returns_none() {
        // total_work = 10 * 5 = 50 < min_parallel_work = 10_000
        assert_eq!(row_reduction_chunk_rows(10, 5, 1, 10_000), None);
    }

    #[test]
    fn row_contraction_blocks_carry_one_task_of_work() {
        assert_eq!(row_contraction_block_rows(0), None);
        // A 10×10 Gram: the budget would ask for 160k rows, capped at the band.
        assert_eq!(row_contraction_block_rows(100), Some(MAX_ROWS_PER_TASK));
        // A 50×50 Gram: 16e6 / 2500 rows.
        assert_eq!(row_contraction_block_rows(2500), Some(6400));
        // The widest output whose smallest block still fits one task.
        let widest = TARGET_WORK_PER_TASK / MIN_ROWS_PER_TASK;
        assert_eq!(row_contraction_block_rows(widest), Some(MIN_ROWS_PER_TASK));
        assert_eq!(row_contraction_block_rows(widest + 1), None);
    }

    #[test]
    fn chunk_count_zero_rows_is_zero() {
        assert_eq!(row_reduction_chunk_count(0, 100), 0);
    }

    #[test]
    fn chunk_count_exact_division() {
        assert_eq!(row_reduction_chunk_count(9, 3), 3);
    }

    #[test]
    fn chunk_count_ceiling_division() {
        assert_eq!(row_reduction_chunk_count(10, 3), 4);
    }

    #[test]
    fn chunk_count_zero_chunk_size_treated_as_one() {
        assert_eq!(row_reduction_chunk_count(7, 0), 7);
    }

    #[test]
    fn chunk_count_chunk_equals_n_rows() {
        assert_eq!(row_reduction_chunk_count(5, 5), 1);
    }

    #[test]
    fn chunk_count_chunk_larger_than_n_rows() {
        assert_eq!(row_reduction_chunk_count(3, 100), 1);
    }
}
