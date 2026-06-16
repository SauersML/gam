use faer::sparse::{SparseColMat, Triplet};
use gam::linalg::matrix::SparseDesignMatrix;

#[test]
fn sparse_design_to_csr_arc_returns_some_for_well_formed_matrix() {
    let sparse = SparseColMat::try_new_from_triplets(
        4,
        3,
        &[
            Triplet::new(0, 0, 1.5),
            Triplet::new(1, 1, -2.0),
            Triplet::new(2, 2, 0.25),
            Triplet::new(3, 0, 4.0),
            Triplet::new(1, 2, 3.5),
        ],
    )
    .expect("Expected sparse constructor to accept a well-formed triplet list");

    let design = SparseDesignMatrix::new(sparse);
    let csr = design.to_csr_arc();

    assert!(
        csr.is_some(),
        "Expected to_csr_arc to return Some for a well-formed sparse matrix"
    );
}
