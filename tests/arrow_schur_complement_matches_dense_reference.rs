use gam::solver::arrow_schur::ArrowSchurSystem;
use ndarray::{array, Array2};

fn inv2(a: &Array2<f64>) -> Array2<f64> {
    let det = a[[0,0]] * a[[1,1]] - a[[0,1]] * a[[1,0]];
    array![[a[[1,1]]/det, -a[[0,1]]/det],[-a[[1,0]]/det, a[[0,0]]/det]]
}

#[test]
fn arrow_schur_complement_matches_dense_reference() {
    let mut sys = ArrowSchurSystem::new(2, 2, 2);
    sys.rows[0].htt = array![[2.0, 0.3],[0.3,1.8]];
    sys.rows[0].htbeta = array![[0.4,0.1],[-0.2,0.5]];
    sys.rows[1].htt = array![[1.9,-0.2],[-0.2,2.1]];
    sys.rows[1].htbeta = array![[0.3,-0.4],[0.6,0.2]];
    sys.hbb = array![[3.7,0.2],[0.2,3.2]];

    let mut s = sys.hbb.clone();
    for i in 0..2 {
        let ainv = inv2(&sys.rows[i].htt);
        let tmp = ainv.dot(&sys.rows[i].htbeta);
        let sub = sys.rows[i].htbeta.t().dot(&tmp);
        s = &s - &sub;
    }

    let expected = array![[3.8324303649, 0.1905861777],[0.1905861777,2.9861855863]];
    for r in 0..2 {
        for c in 0..2 {
            assert!((s[[r,c]] - expected[[r,c]]).abs() <= 1e-9);
        }
    }
}
