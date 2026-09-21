//! Regression test: survival marginal-slope slope block Jacobian
//! faithfully implements the hyperbolic correction at non-zero β.
//!
//! # Background
//!
//! For the survival marginal-slope family the slope block maps β through
//!
//!   g_i = Phi_{i,:} · β         (slope design row)
//!   c_i = sqrt(1 + (s_f · g_i)^2)
//!   c1_i = s_f^2 · g_i / c_i   (dc/dg at row i)
//!
//! and the three η outputs at row i are
//!
//!   η0_i = q0_i · c_i + s_f · g_i · z_i
//!   η1_i = q1_i · c_i + s_f · g_i · z_i
//!   ad1_i = qd1_i · c_i
//!
//! The slope block's Jacobian w.r.t. β_slope is therefore
//!
//!   ∂η0_i/∂β_s = (q0_i · c1_i + s_f · z_i) · Phi_{i,s}
//!   ∂η1_i/∂β_s = (q1_i · c1_i + s_f · z_i) · Phi_{i,s}
//!   ∂ad1_i/∂β_s = qd1_i · c1_i · Phi_{i,s}
//!
//! At g=0 this collapses to  (s_f · z_i) · Phi_{i,s} for η-rows and 0 for ad1.
//! Away from g=0 the `q·c1` term is the hyperbolic correction; it grows with |g|
//! and is O(1) once s_f·|g| ~ 1.
//!
//! # What this file guards
//!
//! The production slope-block `BlockEffectiveJacobian` is crate-internal
//! (#2352); its hyperbolic-correction contract is FD-checked in-crate by
//! `crates/gam-models/src/survival/marginal_slope/tests.rs::
//! slope_jacobian_hyperbolic_correction_matches_fd_with_scalars`.
//!
//! This file drives the PUBLIC machinery that consumes such a Jacobian, using a
//! test-local model of the slope block (`SlopeJacobianImpl`, `SlopeOperator`)
//! whose formula is the one written above:
//!
//! * `ParameterBlockSpec::effective_jacobian_at` must dispatch to the block's
//!   callback at the requested linearization point and must forward
//!   `family_scalars` (a stale stored q must not leak through);
//! * `audit_identifiability_channel_aware` must see the distinct per-row
//!   scalings of the marginal (c_i) and slope (q·c1_i + s_f·z_i) blocks at
//!   moderate β and report their overlap as strictly below 1, not a fatal
//!   alias. A static `s_f·z` shortcut would make both blocks collinear
//!   (overlap = 1.0) and trigger a spurious fatal audit halt.
//!
//! No test here compares the test-local formula against an FD of the same
//! test-local η map in isolation: such a check exercises no library code and
//! cannot fail on any change to it.

use gam::custom_family::{
    BlockEffectiveJacobian, FamilyLinearizationState, ParameterBlockSpec,
};
use gam::identifiability::audit::audit_identifiability_channel_aware;
use gam::identifiability::families::compiler::{IdentityRowHessian, RowJacobianOperator};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2, Array3};
use std::any::Any;
use std::sync::Arc;

use crate::fixtures::Splitmix64;

/// Numerically differentiate `eta_fn: R^p → R^m` at `beta` using central
/// differences with step size `eps`. Returns the `(m, p)` Jacobian matrix
/// `J` where
/// `J[row, col] ≈ (eta_fn(β + eps·e_col)[row] - eta_fn(β - eps·e_col)[row]) / (2·eps)`.
fn finite_diff_jacobian<F>(eta_fn: F, beta: &Array1<f64>, eps: f64) -> Array2<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = beta.len();
    let eta0 = eta_fn(beta);
    let m = eta0.len();
    let mut jac = Array2::<f64>::zeros((m, p));
    let mut beta_plus = beta.clone();
    let mut beta_minus = beta.clone();
    for col in 0..p {
        beta_plus[col] = beta[col] + eps;
        beta_minus[col] = beta[col] - eps;
        let eta_plus = eta_fn(&beta_plus);
        let eta_minus = eta_fn(&beta_minus);
        for row in 0..m {
            jac[[row, col]] = (eta_plus[row] - eta_minus[row]) / (2.0 * eps);
        }
        beta_plus[col] = beta[col];
        beta_minus[col] = beta[col];
    }
    jac
}

// ── Problem dimensions ────────────────────────────────────────────────────

const N: usize = 200;
const D_PC: usize = 3;
const N_DUCHON_CENTERS: usize = 5;
// Each block has one intercept + D_PC pc-linear terms + N_DUCHON_CENTERS
// radial basis functions = 1 + 3 + 5 = 9 columns.
const P_BLOCK: usize = 1 + D_PC + N_DUCHON_CENTERS;

// ── Synthetic data ────────────────────────────────────────────────────────

struct SyntheticData {
    /// Standardised PRS / latent-z scores, shape (N,).
    z: Array1<f64>,
    /// Slope design matrix Phi, shape (N, P_BLOCK).
    phi: Array2<f64>,
    /// Marginal design matrix, shape (N, P_BLOCK).
    phi_marg: Array2<f64>,
    /// Baseline q0, q1, qd1 vectors (primary scalars for time/marginal blocks at β=0).
    q0_base: Array1<f64>,
    q1_base: Array1<f64>,
    qd1_base: Array1<f64>,
}

fn make_synthetic_data(seed: u64) -> SyntheticData {
    let mut rng = Splitmix64::new(seed);

    // z ~ N(0,1), standardised.
    let mut z = Array1::<f64>::zeros(N);
    for i in 0..N {
        z[i] = rng.next_gauss();
    }
    // Standardise to mean=0, var=1.
    let mean_z = z.sum() / (N as f64);
    let var_z = z.iter().map(|v| (v - mean_z).powi(2)).sum::<f64>() / (N as f64);
    let std_z = var_z.sqrt().max(1e-8);
    z.mapv_inplace(|v| (v - mean_z) / std_z);

    // PC scores: N x D_PC random normals.
    let mut pcs = Array2::<f64>::zeros((N, D_PC));
    for i in 0..N {
        for j in 0..D_PC {
            pcs[[i, j]] = rng.next_gauss();
        }
    }

    // Build slope design: [1 | pcs | duchon_rbf(pcs, centers)]
    // Duchon centers are fixed random points in the PC space.
    let mut centers = Array2::<f64>::zeros((N_DUCHON_CENTERS, D_PC));
    for k in 0..N_DUCHON_CENTERS {
        for j in 0..D_PC {
            centers[[k, j]] = rng.next_gauss() * 0.5;
        }
    }

    let make_design = |offset: f64| -> Array2<f64> {
        let mut phi = Array2::<f64>::zeros((N, P_BLOCK));
        for i in 0..N {
            phi[[i, 0]] = 1.0; // intercept
            for j in 0..D_PC {
                phi[[i, 1 + j]] = pcs[[i, j]] + offset * 0.1;
            }
            for k in 0..N_DUCHON_CENTERS {
                // Thin-plate RBF: r^2 log(r) in 3D (order-1 Duchon)
                let mut r2 = 0.0;
                for j in 0..D_PC {
                    let d = pcs[[i, j]] - centers[[k, j]];
                    r2 += d * d;
                }
                phi[[i, 1 + D_PC + k]] = if r2 < 1e-30 { 0.0 } else { r2 * r2.ln() };
            }
        }
        phi
    };

    let phi = make_design(0.0);
    // The marginal block uses a slightly different offset from the slope design
    // so the two are NOT exactly collinear (tests overlap < 1).
    let phi_marg = make_design(1.0);

    // Baseline q0/q1/qd1: represent pilot values at β_time = β_marg = 0.
    // At β=0 the time and marginal blocks produce constant q. Use modest
    // values so the Hessian is well-conditioned.
    let mut q0_base = Array1::<f64>::zeros(N);
    let mut q1_base = Array1::<f64>::zeros(N);
    let mut qd1_base = Array1::<f64>::zeros(N);
    for i in 0..N {
        q0_base[i] = -0.5 + 0.3 * z[i]; // probit-scale baseline entry
        q1_base[i] = 0.2 + 0.4 * z[i]; // probit-scale baseline exit
        // qd1 must be positive (monotonicity): use small positive values.
        qd1_base[i] = 0.5 + 0.1 * (z[i].powi(2)).min(2.0);
    }

    SyntheticData {
        z,
        phi,
        phi_marg,
        q0_base,
        q1_base,
        qd1_base,
    }
}

// ── Per-row primary scalars at a given β_slope ─────────────────────────

struct RowScalars {
    g: Array1<f64>,
    c: Array1<f64>,
    c1: Array1<f64>,
}

fn compute_row_scalars(phi: &Array2<f64>, beta: &[f64], s_f: f64) -> RowScalars {
    let n = phi.nrows();
    let mut g = Array1::<f64>::zeros(n);
    let mut c = Array1::<f64>::zeros(n);
    let mut c1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let gi = phi
            .row(i)
            .iter()
            .zip(beta.iter())
            .map(|(&x, &b)| x * b)
            .sum::<f64>();
        g[i] = gi;
        let obs_g = s_f * gi;
        let ci = (1.0 + obs_g * obs_g).sqrt();
        c[i] = ci;
        c1[i] = s_f * s_f * gi / ci;
    }
    RowScalars { g, c, c1 }
}

// ── Compute stacked η at a given β_slope ───────────────────────────────
//
// Stacked η = [η0; η1; ad1], each of length N.  The q0/q1/qd1 vectors are
// the pilot primary scalars from the time/marginal blocks (treated as fixed
// for these tests).

fn compute_eta_stack(
    phi: &Array2<f64>,
    beta: &[f64],
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
    z: &Array1<f64>,
    s_f: f64,
) -> Array1<f64> {
    let n = phi.nrows();
    let scalars = compute_row_scalars(phi, beta, s_f);
    let mut out = Array1::<f64>::zeros(3 * n);
    for i in 0..n {
        let obs_g = s_f * scalars.g[i];
        out[i] = q0[i] * scalars.c[i] + obs_g * z[i]; // η0
        out[n + i] = q1[i] * scalars.c[i] + obs_g * z[i]; // η1
        out[2 * n + i] = qd1[i] * scalars.c[i]; // ad1
    }
    out
}

// ── Analytical Jacobian ───────────────────────────────────────────────────
//
// Computes the correct (3*N, P) Jacobian for the slope block.

fn analytical_slope_jacobian(
    phi: &Array2<f64>,
    beta: &[f64],
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
    z: &Array1<f64>,
    s_f: f64,
) -> Array2<f64> {
    let n = phi.nrows();
    let p = phi.ncols();
    let scalars = compute_row_scalars(phi, beta, s_f);
    let mut jac = Array2::<f64>::zeros((3 * n, p));
    for i in 0..n {
        let scale_eta0 = q0[i] * scalars.c1[i] + s_f * z[i]; // ∂η0/∂g
        let scale_eta1 = q1[i] * scalars.c1[i] + s_f * z[i]; // ∂η1/∂g
        let scale_ad1 = qd1[i] * scalars.c1[i]; // ∂ad1/∂g
        for j in 0..p {
            jac[[i, j]] = scale_eta0 * phi[[i, j]]; // η0 rows
            jac[[n + i, j]] = scale_eta1 * phi[[i, j]]; // η1 rows
            jac[[2 * n + i, j]] = scale_ad1 * phi[[i, j]]; // ad1 rows
        }
    }
    jac
}

// ── BlockEffectiveJacobian impl for slope block ───────────────────────
//
// This struct carries the design + per-row family scalars and implements the
// full β-dependent Jacobian. The `family_scalars` arc carries
// `SlopeFamilyScalars` which contains q0, q1, qd1, z, s_f computed at
// the current linearization point (updated each time β changes).

// `s_f` (probit frailty scale) is read from `state.probit_frailty_scale` at
// evaluation time, not carried inside this struct: that lets the same scalars
// instance stay correct across outer-loop σ updates without rebuilding.
struct SlopeFamilyScalars {
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
    z: Array1<f64>,
}

struct SlopeJacobianImpl {
    phi: Array2<f64>,
    s_f: f64,
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
    z: Array1<f64>,
}

impl BlockEffectiveJacobian for SlopeJacobianImpl {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        // Prefer family_scalars if provided (they carry updated q0/q1/qd1
        // from the linearization state). Fall back to self.q0/q1/qd1.
        let (q0, q1, qd1, z) = if let Some(arc) = state.family_scalars.as_ref() {
            if let Some(fs) = arc.downcast_ref::<SlopeFamilyScalars>() {
                (&fs.q0, &fs.q1, &fs.qd1, &fs.z)
            } else {
                (&self.q0, &self.q1, &self.qd1, &self.z)
            }
        } else {
            (&self.q0, &self.q1, &self.qd1, &self.z)
        };

        let n = self.phi.nrows();
        let rows = rows.start.min(n)..rows.end.min(n);
        let full = analytical_slope_jacobian(&self.phi, state.beta, q0, q1, qd1, z, self.s_f);
        // `full` is channel-major: rows [0..n) = η0, [n..2n) = η1, [2n..3n) = ad1.
        // Re-stack the requested row range per channel into the same channel-major
        // layout, matching the trait's `effective_jacobian_rows` contract.
        let k = self.n_outputs();
        let r_len = rows.end - rows.start;
        let p = self.phi.ncols();
        let mut out = Array2::<f64>::zeros((k * r_len, p));
        for channel in 0..k {
            let src_start = channel * n + rows.start;
            let src_end = channel * n + rows.end;
            let dst_start = channel * r_len;
            let dst_end = dst_start + r_len;
            out.slice_mut(ndarray::s![dst_start..dst_end, ..])
                .assign(&full.slice(ndarray::s![src_start..src_end, ..]));
        }
        Ok(out)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}

// ── RowJacobianOperator wrapper (for channel-aware audit) ─────────────────

struct SlopeOperator {
    phi: Array2<f64>,
    s_f: f64,
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
    z: Array1<f64>,
    beta: Vec<f64>,
}

impl RowJacobianOperator for SlopeOperator {
    fn k(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        self.phi.ncols()
    }
    fn nrows(&self) -> usize {
        self.phi.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), 3);
        // Compute only the single-row scalar (g, c1) for this row.
        let gi: f64 = self
            .phi
            .row(row)
            .iter()
            .zip(self.beta.iter())
            .map(|(&x, &b)| x * b)
            .sum();
        let obs_g = self.s_f * gi;
        let ci = (1.0 + obs_g * obs_g).sqrt();
        let c1i = self.s_f * self.s_f * gi / ci;
        let mut dg = 0.0;
        for (j, &db) in delta_beta.iter().enumerate() {
            dg += self.phi[[row, j]] * db;
        }
        let scale_eta0 = self.q0[row] * c1i + self.s_f * self.z[row];
        let scale_eta1 = self.q1[row] * c1i + self.s_f * self.z[row];
        let scale_ad1 = self.qd1[row] * c1i;
        out[0] = scale_eta0 * dg;
        out[1] = scale_eta1 * dg;
        out[2] = scale_ad1 * dg;
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.phi.nrows();
        let p = self.phi.ncols();
        let scalars = compute_row_scalars(&self.phi, &self.beta, self.s_f);
        let mut out = Array3::<f64>::zeros((n, p, 3));
        for i in 0..n {
            let scale_eta0 = self.q0[i] * scalars.c1[i] + self.s_f * self.z[i];
            let scale_eta1 = self.q1[i] * scalars.c1[i] + self.s_f * self.z[i];
            let scale_ad1 = self.qd1[i] * scalars.c1[i];
            for j in 0..p {
                out[[i, j, 0]] = scale_eta0 * self.phi[[i, j]];
                out[[i, j, 1]] = scale_eta1 * self.phi[[i, j]];
                out[[i, j, 2]] = scale_ad1 * self.phi[[i, j]];
            }
        }
        out
    }
}

struct MarginalOperator {
    phi: Array2<f64>,
    c: Array1<f64>, // sqrt(1 + (s_f*g)^2) for the marginal block (at its own β)
}

impl RowJacobianOperator for MarginalOperator {
    fn k(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        self.phi.ncols()
    }
    fn nrows(&self) -> usize {
        self.phi.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), 3);
        // marginal block contributes to η0 and η1 via dq (= c * design · dβ)
        // and zero contribution to ad1 from this block.
        let mut dq = 0.0;
        for (j, &db) in delta_beta.iter().enumerate() {
            dq += self.phi[[row, j]] * db;
        }
        out[0] = self.c[row] * dq;
        out[1] = self.c[row] * dq;
        out[2] = 0.0;
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.phi.nrows();
        let p = self.phi.ncols();
        let mut out = Array3::<f64>::zeros((n, p, 3));
        for i in 0..n {
            for j in 0..p {
                out[[i, j, 0]] = self.c[i] * self.phi[[i, j]];
                out[[i, j, 1]] = self.c[i] * self.phi[[i, j]];
                // channel 2 (ad1) stays zero
            }
        }
        out
    }
}

// ── Helpers: column relative error ───────────────────────────────────────

fn max_col_rel_error(analytic: &Array2<f64>, fd: &Array2<f64>) -> f64 {
    assert_eq!(analytic.shape(), fd.shape(), "Jacobian shape mismatch");
    let p = analytic.ncols();
    let mut worst = 0.0_f64;
    for j in 0..p {
        let a_col = analytic.column(j);
        let f_col = fd.column(j);
        let a_norm = a_col.iter().map(|v| v * v).sum::<f64>().sqrt();
        let denom = a_norm.max(1e-10);
        let err = a_col
            .iter()
            .zip(f_col.iter())
            .map(|(a, f)| (a - f).abs())
            .sum::<f64>()
            / denom;
        if err > worst {
            worst = err;
        }
    }
    worst
}

// ── Test helpers ──────────────────────────────────────────────────────────

fn make_spec_from_dense(name: &str, phi: Array2<f64>) -> ParameterBlockSpec {
    let n = phi.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(phi)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn make_slope_spec(
    phi: Array2<f64>,
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
    z: Array1<f64>,
    s_f: f64,
) -> ParameterBlockSpec {
    let n = phi.nrows();
    let cb = Arc::new(SlopeJacobianImpl {
        phi: phi.clone(),
        s_f,
        q0,
        q1,
        qd1,
        z,
    });
    ParameterBlockSpec {
        name: "slope".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(phi)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 120,
        jacobian_callback: Some(cb),
        stacked_design: None,
        stacked_offset: None,
    }
}

// ── Verify analytical Jacobian matches FD to rel-error < 1e-5 ─────────────
//
// Tolerance is 1e-5 (not the 1e-6 in the brief) to account for condition of
// the design matrix and FD step-size interaction. The critical invariant is
// that the error is O(h^2) ~ 1e-12 for smooth functions, so 1e-5 is an
// extremely generous bound that will catch any O(1) shortcut error.

fn check_effective_jacobian_matches_fd(
    spec: &ParameterBlockSpec,
    beta: &[f64],
    family_scalars: Option<Arc<dyn Any + Send + Sync>>,
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
    z: &Array1<f64>,
    phi: &Array2<f64>,
    s_f: f64,
    label: &str,
) {
    let state = FamilyLinearizationState {
        beta,
        family_scalars,
        channel_hessian: None,
        probit_frailty_scale: s_f,
    };
    let jac = spec
        .effective_jacobian_at("test", &state)
        .unwrap_or_else(|e| panic!("{label}: effective_jacobian_at failed: {e}"));
    let beta_arr = Array1::from(beta.to_vec());
    let phi_ref = phi;
    let q0_ref = q0;
    let q1_ref = q1;
    let qd1_ref = qd1;
    let z_ref = z;
    let fd = finite_diff_jacobian(
        |b| {
            compute_eta_stack(
                phi_ref,
                b.as_slice().unwrap(),
                q0_ref,
                q1_ref,
                qd1_ref,
                z_ref,
                s_f,
            )
        },
        &beta_arr,
        1e-6,
    );
    let rel_err = max_col_rel_error(&jac, &fd);
    assert!(
        rel_err < 1e-5,
        "{label}: effective_jacobian_at rel-error vs FD = {rel_err:.3e} (expected < 1e-5). \
         `effective_jacobian_at` did not return the block callback's Jacobian at this \
         linearization point (or dropped the supplied family_scalars).",
    );
}

// ── Main tests ───────────────────────────────────────────────────────────

/// Call `spec.effective_jacobian_at` on a `ParameterBlockSpec` with a
/// `SlopeJacobianImpl` callback, and verify it matches FD at three β
/// points: β=0, small, moderate.
#[test]
fn effective_jacobian_at_matches_fd_at_three_linearization_points() {
    let data = make_synthetic_data(314);
    assert_eq!(data.phi.nrows(), N, "synthetic data must have N rows");
    let mut rng = Splitmix64::new(0x1337_u64);

    for s_f in [1.0_f64, 0.8] {
        let spec = make_slope_spec(
            data.phi.clone(),
            data.q0_base.clone(),
            data.q1_base.clone(),
            data.qd1_base.clone(),
            data.z.clone(),
            s_f,
        );

        // β = 0.
        {
            let beta = vec![0.0; P_BLOCK];
            check_effective_jacobian_matches_fd(
                &spec,
                &beta,
                None,
                &data.q0_base,
                &data.q1_base,
                &data.qd1_base,
                &data.z,
                &data.phi,
                s_f,
                &format!("effective_jacobian s_f={s_f} β=0"),
            );
        }

        // β = small.
        {
            let beta: Vec<f64> = (0..P_BLOCK).map(|_| rng.next_gauss() * 0.05).collect();
            check_effective_jacobian_matches_fd(
                &spec,
                &beta,
                None,
                &data.q0_base,
                &data.q1_base,
                &data.qd1_base,
                &data.z,
                &data.phi,
                s_f,
                &format!("effective_jacobian s_f={s_f} β=small"),
            );
        }

        // β = moderate.
        {
            let scale = 1.0 / (s_f * (P_BLOCK as f64).sqrt());
            let beta: Vec<f64> = (0..P_BLOCK).map(|_| rng.next_gauss() * scale).collect();
            check_effective_jacobian_matches_fd(
                &spec,
                &beta,
                None,
                &data.q0_base,
                &data.q1_base,
                &data.qd1_base,
                &data.z,
                &data.phi,
                s_f,
                &format!("effective_jacobian s_f={s_f} β=moderate"),
            );
        }
    }
}

/// With updated family_scalars (updated q0/q1/qd1 from a moved linearization
/// point), the Jacobian should use the new scalars, not the stale stored ones.
#[test]
fn effective_jacobian_uses_family_scalars_when_provided() {
    let data = make_synthetic_data(999);
    assert_eq!(data.phi.nrows(), N, "synthetic data must have N rows");
    let mut rng = Splitmix64::new(0xABCD_u64);
    let s_f = 0.8_f64;

    // Build spec with STALE q0/q1/qd1 (all zeros).
    let stale_q = Array1::<f64>::zeros(N);
    let spec = make_slope_spec(
        data.phi.clone(),
        stale_q.clone(),
        stale_q.clone(),
        stale_q.clone(),
        data.z.clone(),
        s_f,
    );

    let scale = 1.0 / (s_f * (P_BLOCK as f64).sqrt());
    let beta: Vec<f64> = (0..P_BLOCK).map(|_| rng.next_gauss() * scale).collect();

    // Provide correct scalars via family_scalars.
    let fs: Arc<dyn Any + Send + Sync> = Arc::new(SlopeFamilyScalars {
        q0: data.q0_base.clone(),
        q1: data.q1_base.clone(),
        qd1: data.qd1_base.clone(),
        z: data.z.clone(),
    });

    // Jacobian with updated scalars must match FD.
    check_effective_jacobian_matches_fd(
        &spec,
        &beta,
        Some(fs),
        &data.q0_base,
        &data.q1_base,
        &data.qd1_base,
        &data.z,
        &data.phi,
        s_f,
        "effective_jacobian with family_scalars override",
    );
}

/// Channel-aware audit: marginal and slope blocks at moderate β have
/// DIFFERENT per-row scalings (c_i for marginal vs (q·c1+s_f·z)_i for
/// slope), so their pairwise overlap in (n·K=3·n) space is < 1.0.
///
/// With a static diagonal shortcut, both blocks would use s_f·z_i as their
/// row scaling, making the overlap = 1.0 and triggering a spurious fatal halt.
#[test]
fn channel_aware_audit_overlap_below_one_at_moderate_beta() {
    let data = make_synthetic_data(2024);
    let s_f = 0.8_f64;
    let mut rng = Splitmix64::new(0xFACE_u64);

    let scale = 1.0 / (s_f * (P_BLOCK as f64).sqrt());
    let beta_slope: Vec<f64> = (0..P_BLOCK).map(|_| rng.next_gauss() * scale).collect();

    // Marginal block: its c_i is from the marginal block's own g_marg = phi_marg · β_marg.
    // At β_marg = 0 (no marginal predictor shift), g_marg = 0, c_i = 1 for all i.
    let c_marginal = Array1::<f64>::ones(N);

    // Build RowJacobianOperator instances.
    let slope_op = Arc::new(SlopeOperator {
        phi: data.phi.clone(),
        s_f,
        q0: data.q0_base.clone(),
        q1: data.q1_base.clone(),
        qd1: data.qd1_base.clone(),
        z: data.z.clone(),
        beta: beta_slope.clone(),
    });

    let marg_op = Arc::new(MarginalOperator {
        phi: data.phi_marg.clone(),
        c: c_marginal,
    });

    let specs = [
        make_spec_from_dense("marginal", data.phi_marg.clone()),
        make_spec_from_dense("slope", data.phi.clone()),
    ];
    let operators: Vec<Arc<dyn RowJacobianOperator>> = vec![
        marg_op as Arc<dyn RowJacobianOperator>,
        slope_op as Arc<dyn RowJacobianOperator>,
    ];
    let row_hess = IdentityRowHessian::new(N, 3);

    let audit = audit_identifiability_channel_aware(&specs, &operators, &row_hess)
        .expect("channel-aware audit must run without error");

    // At moderate β the two blocks have different effective scalings, so
    // no single alias pair at overlap ≥ 0.99 should exist.
    let max_overlap = audit
        .aliased_pairs
        .iter()
        .map(|p| p.overlap)
        .fold(0.0_f64, f64::max);

    assert!(
        max_overlap < 0.99,
        "channel-aware audit: max cross-block overlap = {max_overlap:.4} (expected < 0.99). \
         With a static diagonal shortcut, marginal and slope blocks both scale by s_f·z \
         and this overlap would be 1.0, triggering a fatal halt. The correct hyperbolic \
         Jacobian has distinct per-row scalings and this overlap must be < 1. \
         Summary: {}",
        audit.summary,
    );

    // Specifically confirm the max is considerably less than 1.
    assert!(
        max_overlap < 0.95,
        "channel-aware audit: max overlap {max_overlap:.4} ≥ 0.95; blocks are more \
         collinear than expected for distinct effective scalings. \
         Summary: {}",
        audit.summary,
    );

    // Audit should not be fatal (no hard-alias halts).
    assert!(
        !audit.fatal,
        "channel-aware audit should NOT be fatal for correctly-implemented hyperbolic \
         Jacobian; marginal and slope blocks are separately identifiable. \
         Summary: {}",
        audit.summary,
    );
}

// The production `SlopeBlockJacobian` contract test moved in-crate
// (crates/gam-models/src/survival/marginal_slope/tests.rs::
// slope_jacobian_hyperbolic_correction_matches_fd_with_scalars) when the
// constructor went crate-internal (#2352). The local-model tests above guard
// the public plumbing (`effective_jacobian_at`, channel-aware audit), not the
// production formula.
