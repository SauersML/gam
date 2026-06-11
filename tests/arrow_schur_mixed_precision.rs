use gam::solver::arrow_schur::{
    ArrowSchurSystem, ArrowSolveOptions, MixedPrecisionPolicy, MixedPrecisionStatus,
};
use ndarray::{Array1, array};

fn certified_options() -> ArrowSolveOptions {
    ArrowSolveOptions::direct().with_mixed_precision_policy(MixedPrecisionPolicy::certified())
}

fn assert_close(lhs: &Array1<f64>, rhs: &Array1<f64>, tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for i in 0..lhs.len() {
        assert!(
            (lhs[i] - rhs[i]).abs() <= tol,
            "entry {i}: {} vs {} exceeds {tol}",
            lhs[i],
            rhs[i]
        );
    }
}

#[test]
fn certified_mixed_precision_matches_f64_on_well_conditioned_arrow_solve() {
    let mut sys = ArrowSchurSystem::new(3, 2, 2);
    sys.rows[0].htt = array![[4.0, 0.3], [0.3, 3.0]];
    sys.rows[0].htbeta = array![[0.4, -0.2], [0.1, 0.5]];
    sys.rows[0].gt = array![0.2, -0.3];
    sys.rows[1].htt = array![[5.0, -0.4], [-0.4, 2.8]];
    sys.rows[1].htbeta = array![[0.3, 0.2], [-0.1, 0.6]];
    sys.rows[1].gt = array![-0.7, 0.4];
    sys.rows[2].htt = array![[3.5, 0.2], [0.2, 4.2]];
    sys.rows[2].htbeta = array![[-0.2, 0.3], [0.7, -0.4]];
    sys.rows[2].gt = array![0.5, 0.1];
    sys.hbb = array![[7.0, 0.4], [0.4, 6.0]];
    sys.gb = array![0.6, -0.8];

    let f64_solve = sys
        .solve_with_options(0.0, 0.0, &ArrowSolveOptions::direct())
        .expect("f64 direct solve");
    let mixed = sys
        .solve_with_options(0.0, 0.0, &certified_options())
        .expect("certified mixed solve");

    assert!(matches!(
        mixed.2.mixed_precision_status,
        MixedPrecisionStatus::Certified { .. }
    ));
    assert_close(&mixed.0, &f64_solve.0, 1e-9);
    assert_close(&mixed.1, &f64_solve.1, 1e-9);
}

#[test]
fn certified_mixed_precision_refines_ill_conditioned_but_admissible_fixture() {
    let mut sys = ArrowSchurSystem::new(2, 2, 2);
    sys.rows[0].htt = array![[1.0, 0.0], [0.0, 1.0e-4]];
    sys.rows[0].htbeta = array![[0.05, -0.02], [1.0e-4, -2.0e-4]];
    sys.rows[0].gt = array![0.3, -0.1];
    sys.rows[1].htt = array![[1.5, 0.0], [0.0, 2.0e-4]];
    sys.rows[1].htbeta = array![[-0.04, 0.03], [2.0e-4, 1.0e-4]];
    sys.rows[1].gt = array![-0.2, 0.4];
    sys.hbb = array![[4.0, 0.1], [0.1, 3.5]];
    sys.gb = array![0.2, -0.6];

    let f64_solve = sys
        .solve_with_options(0.0, 0.0, &ArrowSolveOptions::direct())
        .expect("f64 direct solve");
    let mixed = sys
        .solve_with_options(0.0, 0.0, &certified_options())
        .expect("mixed solve should certify below the f32 kappa gate");

    assert!(matches!(
        mixed.2.mixed_precision_status,
        MixedPrecisionStatus::Certified { .. }
    ));
    assert_close(&mixed.0, &f64_solve.0, 5e-8);
    assert_close(&mixed.1, &f64_solve.1, 5e-8);
}

#[test]
fn mixed_precision_kappa_gate_falls_back_to_f64_on_near_singular_schur() {
    let mut sys = ArrowSchurSystem::new(0, 0, 2);
    sys.hbb = array![[1.0, 0.0], [0.0, 5.0e-8]];
    sys.gb = array![1.0, 5.0e-8];

    let mixed = sys
        .solve_with_options(0.0, 0.0, &certified_options())
        .expect("f64 fallback should solve this Schur system");

    assert_eq!(
        mixed.2.mixed_precision_status,
        MixedPrecisionStatus::F64Fallback
    );
    assert_close(&mixed.1, &array![-1.0, -1.0], 1e-9);
}
