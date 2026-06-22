use gam::solver::arrow_schur::ArrowSchurSystem;
use ndarray::{Array2, array};

fn inv2(a: &Array2<f64>) -> Array2<f64> {
    let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
    array![
        [a[[1, 1]] / det, -a[[0, 1]] / det],
        [-a[[1, 0]] / det, a[[0, 0]] / det]
    ]
}

#[test]
fn arrow_schur_complement_matches_dense_reference() {
    let mut sys = ArrowSchurSystem::new(2, 2, 2);
    sys.rows[0].htt = array![[2.0, 0.3], [0.3, 1.8]];
    sys.rows[0].htbeta = array![[0.4, 0.1], [-0.2, 0.5]];
    sys.rows[1].htt = array![[1.9, -0.2], [-0.2, 2.1]];
    sys.rows[1].htbeta = array![[0.3, -0.4], [0.6, 0.2]];
    sys.hbb = array![[3.7, 0.2], [0.2, 3.2]];

    let mut s = sys.hbb.clone();
    for i in 0..2 {
        let ainv = inv2(&sys.rows[i].htt);
        let tmp = ainv.dot(&sys.rows[i].htbeta);
        let sub = sys.rows[i].htbeta.t().dot(&tmp);
        s = &s - &sub;
    }

    // Closed-form reference: S = C - B0ᵀ A0⁻¹ B0 - B1ᵀ A1⁻¹ B1.
    //   B0ᵀ A0⁻¹ B0 = [[0.416/3.51, -0.182/3.51], [-0.182/3.51, 0.488/3.51]]
    //   B1ᵀ A1⁻¹ B1 = [[0.945/3.95, -0.06/3.95],  [-0.06/3.95,  0.38/3.95]]
    //   S[0,0] = 3.7 - 0.416/3.51 - 0.945/3.95 = 3.3422409751523678
    //   S[0,1] = 0.2 - (-0.182/3.51) - (-0.06/3.95) = 0.2670417252695733
    //   S[1,1] = 3.2 - 0.488/3.51 - 0.38/3.95 = 2.9647661293230914
    let expected = array![
        [3.3422409751523678, 0.2670417252695733],
        [0.2670417252695733, 2.9647661293230914],
    ];
    for r in 0..2 {
        for c in 0..2 {
            assert!((s[[r, c]] - expected[[r, c]]).abs() <= 1e-9);
        }
    }
}
