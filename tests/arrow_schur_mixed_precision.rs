use gam::solver::arrow_schur::{
    ArrowSchurSystem, ArrowSolveOptions, ArrowSolvePrecisionPolicy, MixedPrecisionStatus,
};
use ndarray::{Array1, Array2, array};

fn certified_options() -> ArrowSolveOptions {
    ArrowSolveOptions::direct()
        .with_solve_precision_policy(ArrowSolvePrecisionPolicy::certified_mixed())
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

/// Assemble the dense symmetric arrow operator `H` and the right-hand side
/// `b = -g` for a two-row, `d=2`, `k=2` fixture, in the solver's variable
/// order `x = (Δt_row0, Δt_row1, Δβ)`. Mirrors `arrow_operator_apply` /
/// `arrow_rhs`: block-diagonal `H_tt` rows, symmetric `H_tβ` coupling, and the
/// `H_ββ` penalty block.
fn dense_arrow_system(sys: &ArrowSchurSystem) -> (Array2<f64>, Array1<f64>) {
    assert_eq!(sys.rows.len(), 2);
    assert_eq!(sys.k, 2);
    let mut h = Array2::<f64>::zeros((6, 6));
    for (r, &base) in [0usize, 2usize].iter().enumerate() {
        let row = &sys.rows[r];
        for a in 0..2 {
            for c in 0..2 {
                h[[base + a, base + c]] = row.htt[[a, c]];
                // H_tβ and its transpose H_βt.
                h[[base + a, 4 + c]] = row.htbeta[[a, c]];
                h[[4 + c, base + a]] = row.htbeta[[a, c]];
            }
        }
    }
    for a in 0..2 {
        for c in 0..2 {
            h[[4 + a, 4 + c]] = sys.hbb[[a, c]];
        }
    }
    let mut b = Array1::<f64>::zeros(6);
    for (r, &base) in [0usize, 2usize].iter().enumerate() {
        for c in 0..2 {
            b[base + c] = -sys.rows[r].gt[c];
        }
    }
    for c in 0..2 {
        b[4 + c] = -sys.gb[c];
    }
    (h, b)
}

/// Relative backward error `‖b − Hx‖∞ / (‖H‖∞‖x‖∞ + ‖b‖∞)` — the quantity the
/// certified mixed-precision path drives below its tolerance. Stacks the row
/// deltas and the β delta into the solver's variable order before measuring.
fn arrow_backward_error(
    sys: &ArrowSchurSystem,
    delta_t: &Array1<f64>,
    delta_beta: &Array1<f64>,
) -> f64 {
    let (h, b) = dense_arrow_system(sys);
    let mut x = Array1::<f64>::zeros(6);
    for i in 0..4 {
        x[i] = delta_t[i];
    }
    for i in 0..2 {
        x[4 + i] = delta_beta[i];
    }
    let r = &b - &h.dot(&x);
    let inf = |v: &Array1<f64>| v.iter().fold(0.0_f64, |m, &e| m.max(e.abs()));
    let h_inf = (0..6)
        .map(|i| (0..6).map(|j| h[[i, j]].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let denom = h_inf * inf(&x) + inf(&b);
    if denom > 0.0 {
        inf(&r) / denom
    } else {
        inf(&r)
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

    // This fixture is deliberately ill-conditioned: the 1e-4 / 2e-4 H_tt pivots
    // drive Δt components to magnitude 1e3-2e3. The certified mixed-precision
    // path guarantees BACKWARD stability (a small residual), not bit-for-bit
    // forward agreement with the f64 solve — on an ill-conditioned system the
    // forward error is κ·(backward error), so the large Δt components legitimately
    // differ from the f64 solve at the ~1e-4 absolute level while the relative
    // error stays at the f32-refinement floor. Asserting an absolute forward
    // tolerance here would be asserting something neither solver can provide.
    //
    // So verify the property the feature actually certifies: the returned
    // solution has a tiny relative backward error. The f64 solve is checked the
    // same way as a self-consistency guard on the assembled dense operator.
    let f64_backward = arrow_backward_error(&sys, &f64_solve.0, &f64_solve.1);
    let mixed_backward = arrow_backward_error(&sys, &mixed.0, &mixed.1);
    assert!(
        f64_backward <= 1e-12,
        "assembled dense operator disagrees with the solver: f64 backward error {f64_backward:e}"
    );
    assert!(
        mixed_backward <= 1e-9,
        "certified mixed solve is not backward stable: backward error {mixed_backward:e}"
    );

    // The relative forward error is bounded by the conditioning; pin it well
    // below O(1) so a genuinely broken solve (wrong scale / garbage) is caught,
    // without asserting unattainable absolute agreement on the 1e3-scale entries.
    for (m, f) in mixed.0.iter().zip(f64_solve.0.iter()) {
        assert!(
            (m - f).abs() <= 1e-5 * f.abs() + 1e-9,
            "Δt relative forward error too large: mixed={m} f64={f}"
        );
    }
    for (m, f) in mixed.1.iter().zip(f64_solve.1.iter()) {
        assert!(
            (m - f).abs() <= 1e-5 * f.abs() + 1e-9,
            "Δβ relative forward error too large: mixed={m} f64={f}"
        );
    }
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
        MixedPrecisionStatus::F64Fallback,
        "near-singular Schur must fall back to f64",
    );
    assert_close(&mixed.1, &array![-1.0, -1.0], 1e-9);
}
