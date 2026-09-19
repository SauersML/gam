//! A rank certificate whose factorization workspace the memory governor refuses
//! publishes `EdfRankBound::NotAssessed`, and the EDF accounting publishes the block's
//! raw trace without failing the fit (#2901).
//!
//! The ledger is process-wide, so this test lives in its own binary: no other test can
//! hold or release reservations while it runs.

use gam_runtime::resource::MemoryGovernor;
use gam_solve::estimate::{EdfRankBound, numerical_rank_bound, penalized_edf_bundle_within_bands};
use ndarray::array;

#[test]
fn a_refused_certificate_workspace_publishes_the_raw_trace_not_assessed_2901() {
    // `H = I` against `λS = 4I` on a rank-2 block: the difference is indefinite, and the
    // trace `λ·tr(H⁻¹S) = 8` lies above the rank.
    let hessian = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let penalty = array![[4.0, 0.0], [0.0, 4.0]];
    let governor = MemoryGovernor::global();

    // Occupy the whole ledger, so the certificate's three 3x3 copies are refused.
    let pressure = governor
        .try_reserve(governor.remaining_bytes(), "ledger pressure fixture")
        .expect("the ledger admits a charge of its own remaining headroom");
    let bound = numerical_rank_bound(hessian.view(), penalty.view(), 0, governor)
        .expect("a refused workspace is a status, not an error");
    match &bound {
        EdfRankBound::NotAssessed { reason } => assert!(
            reason.contains("EDF rank certificate: cannot reserve 216 bytes")
                && reason.contains("bytes already reserved"),
            "the refusal names the certificate, its requested bytes and the budget: {reason}"
        ),
        other => panic!("a refused workspace must publish NotAssessed: {other:?}"),
    }
    let bundle = penalized_edf_bundle_within_bands(
        &[8.0],
        &[1.0e-15],
        std::slice::from_ref(&bound),
        &[2],
        3,
        1.0,
    )
    .expect("a block that was not assessed publishes without failing the fit");
    assert_eq!(bundle.penalty_block_trace, vec![8.0]);
    assert_eq!(bundle.edf_by_block, vec![2.0 - 8.0]);
    assert_eq!(bundle.edf_total, 3.0 - 8.0);
    assert_eq!(bundle.rank_bound, vec![bound]);

    // Releasing the pressure admits the workspace, and the same block is assessed.
    drop(pressure);
    let assessed = numerical_rank_bound(hessian.view(), penalty.view(), 0, governor)
        .expect("the released ledger admits the certificate");
    assert!(
        matches!(assessed, EdfRankBound::Uncertified { .. }),
        "{assessed:?}"
    );
}
