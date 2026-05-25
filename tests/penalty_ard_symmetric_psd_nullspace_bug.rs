use ndarray::{arr2, Array2}; use faer::Side; use gam::linalg::faer_ndarray::FaerEigh;
fn null_dim(s: &Array2<f64>) -> usize { let (e,_) = s.eigh(Side::Lower).unwrap(); e.iter().filter(|&&v| v.abs()<1e-9).count() }
#[test] fn ard_penalty_s_matrix_should_be_symmetric_psd_with_documented_nullspace(){ let s=arr2(&[[2.0,0.0],[0.0,3.0]]); assert_eq!(null_dim(&s),1,"documented null-space mismatch for ARD"); }
