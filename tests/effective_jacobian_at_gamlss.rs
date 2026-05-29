/// Finite-difference verification for `block_effective_jacobian` on all five
/// GAMLSS workspace variants.
///
/// For each family and each block, we verify that the stacked Jacobian
/// J returned by `block_effective_jacobian` satisfies:
///
///   J[r*n + i, j] ≈ (eta_r(β + ε·e_j) - eta_r(β - ε·e_j)) / (2ε)
///
/// where `eta_r[i] = X_r[i,:] · β_r + offset_r` is the r-th output's linear
/// predictor at observation i.  Tolerance: 1e-7 relative.
use gam::custom_family::{FamilyLinearizationState, ParameterBlockSpec};
use gam::families::gamlss::{
    BinomialLocationScaleFamily, BinomialLocationScaleWiggleFamily, BinomialMeanWiggleFamily,
    GaussianLocationScaleFamily, GaussianLocationScaleWiggleFamily,
};
use gam::matrix::DesignMatrix;
use ndarray::{Array1, Array2};

const N: usize = 6;
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

fn design_a() -> Array2<f64> {
    // n=6, p=2 design: col0 = 1..6, col1 = sinusoidal
    let mut x = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        x[[i, 0]] = (i as f64 + 1.0) / N as f64;
        x[[i, 1]] = ((i as f64) * 1.2).sin();
    }
    x
}

fn design_b() -> Array2<f64> {
    // n=6, p=3 design
    let mut x = Array2::<f64>::zeros((N, 3));
    for i in 0..N {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (i as f64) * 0.5 - 1.0;
        x[[i, 2]] = ((i as f64) * 0.8).cos();
    }
    x
}

fn design_c() -> Array2<f64> {
    // n=6, p=2 design for wiggle block (small values)
    let mut x = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        x[[i, 0]] = ((i as f64) * 0.3).sin();
        x[[i, 1]] = ((i as f64) * 0.7).cos();
    }
    x
}

/// Compute eta_r for a given spec and beta via X·β + offset.
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

/// Finite-difference Jacobian for output r of a family with n_outputs.
/// `compute_block_eta` maps beta_block -> eta_r[0..N].
fn fd_jacobian<F>(compute_eta_r: F, beta0: &Array1<f64>, p: usize) -> Array2<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut jac = Array2::<f64>::zeros((N, p));
    for j in 0..p {
        let mut bp = beta0.clone();
        let mut bm = beta0.clone();
        bp[j] += EPS;
        bm[j] -= EPS;
        let ep = compute_eta_r(&bp);
        let em = compute_eta_r(&bm);
        for i in 0..N {
            jac[[i, j]] = (ep[i] - em[i]) / (2.0 * EPS);
        }
    }
    jac
}

/// Verify stacked Jacobian for one block.
///
/// `own_output` is `Some(r)` when this block linearly drives output `r`,
/// or `None` when the block has a zero Jacobian for all outputs (e.g. wiggle).
///
/// For each output channel `r`:
///   - `r == own_output`: rows must match the FD of `X_block · β`.
///   - `r != own_output` (or `own_output` is `None`): rows must be exactly 0.
fn check_jac(
    name: &str,
    block_idx: usize,
    specs: &[ParameterBlockSpec],
    jac_full: &Array2<f64>,
    n_outputs: usize,
    own_output: Option<usize>,
) {
    let p = specs[block_idx].design.ncols();
    let beta0 = Array1::zeros(p);
    let fd = fd_jacobian(|b| compute_eta(&specs[block_idx], b), &beta0, p);

    for r in 0..n_outputs {
        let row_start = r * N;
        let jac_r = jac_full.slice(ndarray::s![row_start..row_start + N, ..]);
        if own_output == Some(r) {
            // Owned output: Jacobian rows must equal finite-diff of X_block·β
            for i in 0..N {
                for j in 0..p {
                    let analytic = jac_r[[i, j]];
                    let finite_diff = fd[[i, j]];
                    let scale = finite_diff.abs().max(1e-10);
                    let rel_err = (analytic - finite_diff).abs() / scale;
                    assert!(
                        rel_err < TOL_REL,
                        "{name} block{block_idx} output{r} [{i},{j}]: analytic={analytic:.3e} fd={finite_diff:.3e} rel={rel_err:.2e}"
                    );
                }
            }
        } else {
            // Not-owned output: Jacobian rows must be exactly zero
            for i in 0..N {
                for j in 0..p {
                    assert!(
                        jac_r[[i, j]].abs() < 1e-14,
                        "{name} block{block_idx} output{r} [{i},{j}]: expected 0, got {}",
                        jac_r[[i, j]]
                    );
                }
            }
        }
    }
}

fn make_state(beta: &[f64]) -> FamilyLinearizationState<'_> {
    FamilyLinearizationState {
        beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    }
}

#[test]
fn gaussian_location_scale_block_jacobian() {
    let xa = design_a();
    let xb = design_b();
    let specs = vec![make_spec("mu", xa), make_spec("log_sigma", xb)];
    let n_outputs = 2;

    for block_idx in 0..2 {
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box = GaussianLocationScaleFamily::block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");
        assert_eq!(jac.nrows(), n_outputs * N, "jacobian rows");
        assert_eq!(jac.ncols(), p, "jacobian cols");
        check_jac(
            "GaussianLocationScaleFamily",
            block_idx,
            &specs,
            &jac,
            n_outputs,
            Some(block_idx),
        );
    }
}

#[test]
fn gaussian_location_scale_wiggle_block_jacobian() {
    let xa = design_a();
    let xb = design_b();
    let xc = design_c();
    let specs = vec![
        make_spec("mu", xa),
        make_spec("log_sigma", xb),
        make_spec("wiggle", xc),
    ];
    let n_outputs = 2;

    // mu and log_sigma blocks have standard additive Jacobians
    for block_idx in 0..2 {
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box =
            GaussianLocationScaleWiggleFamily::block_effective_jacobian(&specs, block_idx)
                .expect("block_effective_jacobian");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");
        assert_eq!(jac.nrows(), n_outputs * N);
        assert_eq!(jac.ncols(), p);
        check_jac(
            "GaussianLocationScaleWiggleFamily",
            block_idx,
            &specs,
            &jac,
            n_outputs,
            Some(block_idx),
        );
    }

    // wiggle block (block 2) has an all-zero stacked Jacobian (nonlinear link)
    {
        let block_idx = 2;
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box =
            GaussianLocationScaleWiggleFamily::block_effective_jacobian(&specs, block_idx)
                .expect("block_effective_jacobian for wiggle");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");
        assert_eq!(jac.nrows(), n_outputs * N);
        assert_eq!(jac.ncols(), p);
        check_jac(
            "GaussianLocationScaleWiggleFamily wiggle",
            block_idx,
            &specs,
            &jac,
            n_outputs,
            None,
        );
    }
}

#[test]
fn binomial_mean_wiggle_block_jacobian() {
    let xa = design_a();
    let xc = design_c();
    let specs = vec![make_spec("eta", xa), make_spec("wiggle", xc)];
    let n_outputs = 1;

    // eta block (block 0)
    {
        let block_idx = 0;
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box = BinomialMeanWiggleFamily::block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");
        assert_eq!(jac.nrows(), n_outputs * N);
        assert_eq!(jac.ncols(), p);
        check_jac(
            "BinomialMeanWiggleFamily",
            block_idx,
            &specs,
            &jac,
            n_outputs,
            Some(0),
        );
    }

    // wiggle block (block 1) - zero Jacobian
    {
        let block_idx = 1;
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box = BinomialMeanWiggleFamily::block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");
        assert_eq!(jac.nrows(), n_outputs * N);
        assert_eq!(jac.ncols(), p);
        check_jac(
            "BinomialMeanWiggleFamily wiggle",
            block_idx,
            &specs,
            &jac,
            n_outputs,
            None,
        );
    }
}

#[test]
fn binomial_location_scale_block_jacobian() {
    let xa = design_a();
    let xb = design_b();
    let specs = vec![make_spec("threshold", xa), make_spec("log_sigma", xb)];
    let n_outputs = 2;

    for block_idx in 0..2 {
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box = BinomialLocationScaleFamily::block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");
        assert_eq!(jac.nrows(), n_outputs * N);
        assert_eq!(jac.ncols(), p);
        check_jac(
            "BinomialLocationScaleFamily",
            block_idx,
            &specs,
            &jac,
            n_outputs,
            Some(block_idx),
        );
    }
}

#[test]
fn binomial_location_scale_wiggle_block_jacobian() {
    let xa = design_a();
    let xb = design_b();
    let xc = design_c();
    let specs = vec![
        make_spec("threshold", xa),
        make_spec("log_sigma", xb),
        make_spec("wiggle", xc),
    ];
    let n_outputs = 2;

    // threshold and log_sigma blocks
    for block_idx in 0..2 {
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box =
            BinomialLocationScaleWiggleFamily::block_effective_jacobian(&specs, block_idx)
                .expect("block_effective_jacobian");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");
        assert_eq!(jac.nrows(), n_outputs * N);
        assert_eq!(jac.ncols(), p);
        check_jac(
            "BinomialLocationScaleWiggleFamily",
            block_idx,
            &specs,
            &jac,
            n_outputs,
            Some(block_idx),
        );
    }

    // wiggle block (block 2) - zero Jacobian
    {
        let block_idx = 2;
        let p = specs[block_idx].design.ncols();
        let beta0 = vec![0.0f64; p];
        let state = make_state(&beta0);
        let jac_box =
            BinomialLocationScaleWiggleFamily::block_effective_jacobian(&specs, block_idx)
                .expect("block_effective_jacobian");
        let jac = jac_box
            .effective_jacobian_at(&state)
            .expect("effective_jacobian_at");
        assert_eq!(jac.nrows(), n_outputs * N);
        assert_eq!(jac.ncols(), p);
        check_jac(
            "BinomialLocationScaleWiggleFamily wiggle",
            block_idx,
            &specs,
            &jac,
            n_outputs,
            None,
        );
    }
}

/// Non-zero beta: verify Jacobian is β-independent for linear (additive) blocks.
#[test]
fn gaussian_location_scale_jacobian_beta_independent() {
    let xa = design_a();
    let xb = design_b();
    let specs = vec![make_spec("mu", xa), make_spec("log_sigma", xb)];

    for block_idx in 0..2 {
        let p = specs[block_idx].design.ncols();
        let beta_zero = vec![0.0f64; p];
        let beta_nonzero: Vec<f64> = (0..p).map(|j| (j as f64 + 1.0) * 0.3).collect();

        let jac_box = GaussianLocationScaleFamily::block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");

        let jac0 = jac_box
            .effective_jacobian_at(&make_state(&beta_zero))
            .expect("at zero");
        let jac1 = jac_box
            .effective_jacobian_at(&make_state(&beta_nonzero))
            .expect("at nonzero");

        // For additive linear blocks J is β-independent
        for r in 0..jac0.nrows() {
            for c in 0..jac0.ncols() {
                assert!(
                    (jac0[[r, c]] - jac1[[r, c]]).abs() < 1e-14,
                    "block{block_idx}: Jacobian should be β-independent, got diff at [{r},{c}]"
                );
            }
        }
    }
}

/// n_outputs() returns 2 for location-scale families.
#[test]
fn n_outputs_is_two_for_location_scale() {
    let xa = design_a();
    let xb = design_b();
    let specs = vec![make_spec("mu", xa), make_spec("log_sigma", xb)];

    for block_idx in 0..2 {
        let jac_box = GaussianLocationScaleFamily::block_effective_jacobian(&specs, block_idx)
            .expect("block_effective_jacobian");
        assert_eq!(jac_box.n_outputs(), 2, "block {block_idx} n_outputs");
    }
}

/// Out-of-range block_idx returns Err.
#[test]
fn out_of_range_block_idx_returns_err() {
    let xa = design_a();
    let xb = design_b();
    let specs = vec![make_spec("mu", xa), make_spec("log_sigma", xb)];
    let result = GaussianLocationScaleFamily::block_effective_jacobian(&specs, 5);
    assert!(result.is_err(), "expected Err for out-of-range block_idx");
}

/// Box<dyn BlockEffectiveJacobian> can be sent across threads (Send + Sync bound).
#[test]
fn jacobian_box_is_send() {
    fn assert_send<T: Send>(t: T) {
        std::mem::drop(t);
    }
    let xa = design_a();
    let xb = design_b();
    let specs = vec![make_spec("mu", xa), make_spec("log_sigma", xb)];
    let jac = GaussianLocationScaleFamily::block_effective_jacobian(&specs, 0)
        .expect("block_effective_jacobian");
    // Verify the jacobian produces a well-formed matrix.
    let state = FamilyLinearizationState {
        beta: &[],
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let j = jac
        .effective_jacobian_at(&state)
        .expect("effective_jacobian_at");
    assert!(j.nrows() > 0, "jacobian must have rows");
    assert!(j.ncols() > 0, "jacobian must have columns");
    // Compile-time Send invariant: constructing the box must be Send.
    assert_send(jac);
}
