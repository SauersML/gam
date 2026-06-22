use gam::terms::basis::{DuchonNullspaceOrder, build_duchon_operator_penalty_matrices};
use ndarray::Array2;

#[test]
fn bug_duchon_collocation_gram_symmetry_or_psd_violation() {
    let centers = Array2::from_shape_vec(
        (7, 2),
        vec![
            -1.0, -0.8, -0.5, 0.2, 0.0, 0.0, 0.4, -0.2, 0.8, 0.7, 1.1, -0.4, -1.2, 0.9,
        ],
    )
    .expect("shape");

    let ops = build_duchon_operator_penalty_matrices(
        centers.view(),
        None,
        Some(1.0),
        1.0,
        DuchonNullspaceOrder::Linear,
        None,
        None,
    )
    .expect("duchon collocation penalty matrices");

    let s = &ops.mass;
    let (n, m) = s.dim();
    assert_eq!(n, m, "bug duchon gram non-square: {n}x{m}");

    let mut max_asym = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    for i in 0..n {
        min_diag = min_diag.min(s[[i, i]]);
        for j in 0..n {
            max_asym = max_asym.max((s[[i, j]] - s[[j, i]]).abs());
        }
    }
    assert!(
        max_asym <= 1e-10,
        "bug duchon gram symmetry violated: max |K-K^T|={max_asym:e}"
    );
    assert!(
        min_diag >= -1e-10,
        "bug duchon gram PSD diagonal violated: min diag={min_diag:e}"
    );
}
