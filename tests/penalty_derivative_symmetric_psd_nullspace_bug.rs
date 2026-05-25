use ndarray::{arr2, Array2};
use faer::Side;
use gam::linalg::faer_ndarray::FaerEigh;
fn null_dim(s: &Array2<f64>) -> usize { let (e,_) = s.eigh(Side::Lower).unwrap(); e.iter().filter(|&&v| v.abs() < 1e-9).count() }
#[test]
fn derivative_penalty_s_matrix_should_be_symmetric_psd_with_documented_nullspace() {
 let s=arr2(&[[1.0,-2.0,1.0],[-2.0,4.0,-2.0],[1.0,-2.0,1.0]]);
 let skew=(&s-&s.t()).mapv(f64::abs).sum(); let (e,_) = s.eigh(Side::Lower).unwrap();
 assert!(skew<=1e-12 && e.iter().all(|&x| x>=-1e-9)); assert_eq!(null_dim(&s), 1, "documented null-space mismatch for derivative");
}
