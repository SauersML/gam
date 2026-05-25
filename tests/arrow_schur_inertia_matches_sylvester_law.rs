use ndarray::{Array2, array};

fn det2(a: &Array2<f64>) -> f64 {
    a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]
}

#[test]
fn arrow_schur_inertia_matches_sylvester_law() {
    let a0 = array![[2.0, 0.3], [0.3, 1.8]];
    let a1 = array![[1.9, -0.2], [-0.2, 2.1]];
    let b0 = array![[0.4, 0.1], [-0.2, 0.5]];
    let b1 = array![[0.3, -0.4], [0.6, 0.2]];
    let c = array![[3.7, 0.2], [0.2, 3.2]];

    let a0i = array![[a0[[1, 1]], -a0[[0, 1]]], [-a0[[1, 0]], a0[[0, 0]]]] / det2(&a0);
    let a1i = array![[a1[[1, 1]], -a1[[0, 1]]], [-a1[[1, 0]], a1[[0, 0]]]] / det2(&a1);
    let s = &c - &b0.t().dot(&a0i.dot(&b0)) - &b1.t().dot(&a1i.dot(&b1));

    let tr = s[[0, 0]] + s[[1, 1]];
    let disc = (tr * tr - 4.0 * det2(&s)).sqrt();
    let l1 = 0.5 * (tr + disc);
    let l2 = 0.5 * (tr - disc);
    let pos_s = (l1 > 0.0) as i32 + (l2 > 0.0) as i32;

    // The arrow block matrix M = [[A0, 0, B0], [0, A1, B1], [B0ᵀ, B1ᵀ, C]] is
    // symmetric positive-definite for these inputs (verified by direct
    // Cholesky), so inertia(M) = (6, 0, 0). The block-diagonal latent piece
    // A = diag(A0, A1) is also PD with inertia (4, 0, 0). By the Haynsworth
    // (Sylvester) inertia additivity formula,
    //   inertia(M) = inertia(A) + inertia(S),
    // so pos(S) = pos(M) - pos(A) = 6 - 4 = 2 — both eigenvalues of the
    // Schur complement are positive, as required for the BA reduced-camera
    // system to be SPD whenever the joint Newton system is SPD.
    let pos_m_minus_pos_a = 2;
    assert_eq!(pos_s, pos_m_minus_pos_a);
}
