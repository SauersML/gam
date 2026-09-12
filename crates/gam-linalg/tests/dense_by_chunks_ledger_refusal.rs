//! `DesignMatrix::try_to_dense_by_chunks` charges its construction window on the
//! process-wide memory ledger, so a design the process cannot hold is refused
//! before allocation instead of exhausting memory (SPEC rule 10, gam#2900).
//!
//! The ledger is process-wide, so this test lives in its own binary: no other
//! test can hold or release reservations while it runs.

use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_runtime::resource::MemoryGovernor;
use ndarray::Array2;

#[test]
fn dense_by_chunks_refuses_a_design_the_ledger_cannot_admit() {
    let (rows, cols) = (64usize, 32usize);
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::from_elem(
        (rows, cols),
        1.0,
    )));
    let design_bytes = rows * cols * std::mem::size_of::<f64>();
    let governor = MemoryGovernor::global();

    // Occupy the ledger down to half of one design's footprint.
    let pressure = governor
        .try_reserve(
            governor.remaining_bytes().saturating_sub(design_bytes / 2),
            "ledger pressure fixture",
        )
        .expect("the ledger admits a charge of its own remaining headroom");
    let error = design
        .try_to_dense_by_chunks("pressured densification")
        .expect_err("a design the ledger cannot admit must be refused before allocation");
    assert!(
        error.contains("refusing to densify 64x32 design")
            && error.contains("pressured densification"),
        "refusal must name the design and the caller's context: {error}"
    );

    // Releasing the pressure admits the same design.
    drop(pressure);
    let dense = design
        .try_to_dense_by_chunks("released densification")
        .expect("the released ledger admits the design");
    assert_eq!(dense.dim(), (rows, cols));
    assert!(dense.iter().all(|&value| value == 1.0));
}
