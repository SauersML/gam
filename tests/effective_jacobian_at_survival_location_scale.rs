/// Finite-difference verification for `SurvivalLocationScaleFamily::block_effective_jacobian`.
///
/// The survival location-scale model has three linear outputs per observation:
///   - output 0: η_time       ← time_transform block (block 0)
///   - output 1: η_threshold  ← threshold block (block 1)
///   - output 2: η_log_sigma  ← log_sigma block (block 2)
///
/// The optional linkwiggle block (block 3) has an all-zero effective linear Jacobian.
///
/// For each block we verify:
///   J[r*n + i, j] ≈ (eta_r(β + ε·e_j) - eta_r(β - ε·e_j)) / (2ε)
///
/// where eta_r[i] = X_r[i,:] · β_r + offset_r.  Tolerance: 1e-7 relative.
use gam::custom_family::{FamilyLinearizationState, ParameterBlockSpec};
use gam::families::survival::location_scale::survival_location_scale_block_effective_jacobian;
use gam::matrix::DesignMatrix;
use ndarray::{Array1, Array2};

const N: usize = 7;
const EPS: f64 = 1e-6;
const TOL_REL: f64 = 1e-7;

fn make_spec(name: &str, x: Array2<f64>) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::from(x),
        offset: Array1::zeros(N),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn design_p2() -> Array2<f64> {
    // n=7, p=2
    let mut x = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        x[[i, 0]] = (i as f64 + 1.0) / N as f64;
        x[[i, 1]] = ((i as f64) * 1.3).sin();
    }
    x
}

fn design_p3() -> Array2<f64> {
    // n=7, p=3
    let mut x = Array2::<f64>::zeros((N, 3));
    for i in 0..N {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (i as f64) * 0.4 - 1.2;
        x[[i, 2]] = ((i as f64) * 0.9).cos();
    }
    x
}

fn design_p4() -> Array2<f64> {
    // n=7, p=4
    let mut x = Array2::<f64>::zeros((N, 4));
    for i in 0..N {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (i as f64) * 0.3;
        x[[i, 2]] = ((i as f64) * 0.5).sin();
        x[[i, 3]] = ((i as f64) * 0.6).cos();
    }
    x
}

fn design_p2_wiggle() -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        x[[i, 0]] = ((i as f64) * 0.4).sin();
        x[[i, 1]] = ((i as f64) * 0.8).cos();
    }
    x
}

fn compute_eta(spec: &ParameterBlockSpec, beta: &Array1<f64>) -> Array1<f64> {
    let x = spec
        .design
        .try_to_dense_arc("test")
        .expect("design to dense")
        .as_ref()
        .clone();
    let mut eta = x.dot(beta);
    eta += &spec.offset;
    eta
}

fn fd_jacobian_for_spec(spec: &ParameterBlockSpec, beta0: &Array1<f64>) -> Array2<f64> {
    let p = beta0.len();
    let mut jac = Array2::<f64>::zeros((N, p));
    for j in 0..p {
        let mut bp = beta0.clone();
        let mut bm = beta0.clone();
        bp[j] += EPS;
        bm[j] -= EPS;
        let ep = compute_eta(spec, &bp);
        let em = compute_eta(spec, &bm);
        for i in 0..N {
            jac[[i, j]] = (ep[i] - em[i]) / (2.0 * EPS);
        }
    }
    jac
}

fn make_state(beta: &[f64]) -> FamilyLinearizationState<'_> {
    FamilyLinearizationState {
        beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    }
}

const N_OUTPUTS: usize = 3;

fn assert_block_jacobian(
    label: &str,
    block_idx: usize,
    specs: &[ParameterBlockSpec],
    own_output: Option<usize>,
) {
    let p = specs[block_idx].design.ncols();
    let beta0 = Array1::zeros(p);
    let state = make_state(beta0.as_slice().unwrap());

    let jac_box = survival_location_scale_block_effective_jacobian(specs, block_idx)
        .expect("block_effective_jacobian");
    let jac = jac_box
        .effective_jacobian_at(&state)
        .expect("effective_jacobian_at");

    assert_eq!(
        jac.nrows(),
        N_OUTPUTS * N,
        "{label}: expected {} rows, got {}",
        N_OUTPUTS * N,
        jac.nrows()
    );
    assert_eq!(
        jac.ncols(),
        p,
        "{label}: expected {p} cols, got {}",
        jac.ncols()
    );

    for r in 0..N_OUTPUTS {
        let row_start = r * N;
        let jac_r = jac.slice(ndarray::s![row_start..row_start + N, ..]);

        if own_output == Some(r) {
            // This block owns output r: J_r == FD of X_r·β
            let fd = fd_jacobian_for_spec(&specs[block_idx], &beta0);
            for i in 0..N {
                for j in 0..p {
                    let analytic = jac_r[[i, j]];
                    let fd_val = fd[[i, j]];
                    let scale = fd_val.abs().max(1e-10);
                    let rel_err = (analytic - fd_val).abs() / scale;
                    assert!(
                        rel_err < TOL_REL,
                        "{label} block{block_idx} output{r} [{i},{j}]: analytic={analytic:.3e} fd={fd_val:.3e} rel={rel_err:.2e}"
                    );
                }
            }
        } else {
            // Zero block
            for i in 0..N {
                for j in 0..p {
                    assert!(
                        jac_r[[i, j]].abs() < 1e-14,
                        "{label} block{block_idx} output{r} [{i},{j}]: expected 0, got {}",
                        jac_r[[i, j]]
                    );
                }
            }
        }
    }
}

#[test]
fn survival_ls_time_block_jacobian() {
    let specs = vec![
        make_spec("time_transform", design_p4()),
        make_spec("threshold", design_p2()),
        make_spec("log_sigma", design_p3()),
    ];
    // time block (0) → output 0
    assert_block_jacobian("SurvivalLS/time", 0, &specs, Some(0));
}

#[test]
fn survival_ls_threshold_block_jacobian() {
    let specs = vec![
        make_spec("time_transform", design_p4()),
        make_spec("threshold", design_p2()),
        make_spec("log_sigma", design_p3()),
    ];
    // threshold block (1) → output 1
    assert_block_jacobian("SurvivalLS/threshold", 1, &specs, Some(1));
}

#[test]
fn survival_ls_log_sigma_block_jacobian() {
    let specs = vec![
        make_spec("time_transform", design_p4()),
        make_spec("threshold", design_p2()),
        make_spec("log_sigma", design_p3()),
    ];
    // log_sigma block (2) → output 2
    assert_block_jacobian("SurvivalLS/log_sigma", 2, &specs, Some(2));
}

#[test]
fn survival_ls_linkwiggle_block_jacobian_is_zero() {
    let specs = vec![
        make_spec("time_transform", design_p4()),
        make_spec("threshold", design_p2()),
        make_spec("log_sigma", design_p3()),
        make_spec("linkwiggle", design_p2_wiggle()),
    ];
    // linkwiggle block (3) → zero Jacobian for all outputs
    assert_block_jacobian("SurvivalLS/linkwiggle", 3, &specs, None);
}

#[test]
fn survival_ls_jacobian_beta_independent() {
    let specs = vec![
        make_spec("time_transform", design_p4()),
        make_spec("threshold", design_p2()),
        make_spec("log_sigma", design_p3()),
    ];
    for block_idx in 0..3 {
        let p = specs[block_idx].design.ncols();
        let beta_zero = vec![0.0f64; p];
        let beta_nonzero: Vec<f64> = (0..p).map(|j| (j as f64 + 1.0) * 0.25).collect();

        let jac_box = survival_location_scale_block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");

        let jac0 = jac_box
            .effective_jacobian_at(&make_state(&beta_zero))
            .expect("at zero");
        let jac1 = jac_box
            .effective_jacobian_at(&make_state(&beta_nonzero))
            .expect("at nonzero");

        for r in 0..jac0.nrows() {
            for c in 0..jac0.ncols() {
                assert!(
                    (jac0[[r, c]] - jac1[[r, c]]).abs() < 1e-14,
                    "block{block_idx}: Jacobian should be β-independent at [{r},{c}]"
                );
            }
        }
    }
}

#[test]
fn survival_ls_n_outputs_is_three() {
    let specs = vec![
        make_spec("time_transform", design_p4()),
        make_spec("threshold", design_p2()),
        make_spec("log_sigma", design_p3()),
    ];
    for block_idx in 0..3 {
        let jac_box = survival_location_scale_block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");
        assert_eq!(
            jac_box.n_outputs(),
            3,
            "block {block_idx} n_outputs should be 3"
        );
    }
}

#[test]
fn survival_ls_out_of_range_returns_err() {
    let specs = vec![
        make_spec("time_transform", design_p4()),
        make_spec("threshold", design_p2()),
        make_spec("log_sigma", design_p3()),
    ];
    let result = survival_location_scale_block_effective_jacobian(&specs, 10);
    assert!(result.is_err(), "expected Err for block_idx=10");
}

/// Jacobian values match design matrix entries exactly (not just up to FD error).
#[test]
fn survival_ls_jacobian_equals_design_exactly() {
    let xa = design_p4();
    let xb = design_p2();
    let xc = design_p3();
    let specs = vec![
        make_spec("time_transform", xa.clone()),
        make_spec("threshold", xb.clone()),
        make_spec("log_sigma", xc.clone()),
    ];

    let designs = [xa, xb, xc];
    for block_idx in 0..3 {
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box = survival_location_scale_block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");

        // Owned output rows must equal design exactly
        let row_start = block_idx * N;
        let jac_own = jac.slice(ndarray::s![row_start..row_start + N, ..]);
        let x_ref = &designs[block_idx];
        for i in 0..N {
            for j in 0..p {
                assert!(
                    (jac_own[[i, j]] - x_ref[[i, j]]).abs() < 1e-15,
                    "block{block_idx} owned output: jac[{i},{j}]={} != design[{i},{j}]={}",
                    jac_own[[i, j]],
                    x_ref[[i, j]]
                );
            }
        }
    }
}

/// Box<dyn BlockEffectiveJacobian> satisfies Send bound.
#[test]
fn survival_ls_jacobian_box_is_send() {
    fn assert_send<T: Send>(t: T) {
        std::mem::drop(t);
    }
    let specs = vec![
        make_spec("time_transform", design_p4()),
        make_spec("threshold", design_p2()),
        make_spec("log_sigma", design_p3()),
    ];
    let jac = survival_location_scale_block_effective_jacobian(&specs, 0)
        .expect("block_effective_jacobian");
    assert_send(jac);
}
