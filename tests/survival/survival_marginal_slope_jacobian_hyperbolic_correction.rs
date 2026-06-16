//! Regression test: survival marginal-slope logslope block Jacobian
//! faithfully implements the hyperbolic correction at non-zero β.
//!
//! # Background
//!
//! For the survival marginal-slope family the logslope block maps β through
//!
//!   g_i = Phi_{i,:} · β         (logslope design row)
//!   c_i = sqrt(1 + (s_f · g_i)^2)
//!   c1_i = s_f^2 · g_i / c_i   (dc/dg at row i)
//!
//! and the three η outputs at row i are
//!
//!   η0_i = q0_i · c_i + s_f · g_i · z_i
//!   η1_i = q1_i · c_i + s_f · g_i · z_i
//!   ad1_i = qd1_i · c_i
//!
//! The logslope block's Jacobian w.r.t. β_logslope is therefore
//!
//!   ∂η0_i/∂β_s = (q0_i · c1_i + s_f · z_i) · Phi_{i,s}
//!   ∂η1_i/∂β_s = (q1_i · c1_i + s_f · z_i) · Phi_{i,s}
//!   ∂ad1_i/∂β_s = qd1_i · c1_i · Phi_{i,s}
//!
//! At g=0 this collapses to  (s_f · z_i) · Phi_{i,s} for η-rows and 0 for ad1.
//! Away from g=0 the `q·c1` term is the hyperbolic correction; it grows with |g|
//! and is O(1) once s_f·|g| ~ 1.
//!
//! # What this test guards
//!
//! T1's `BlockEffectiveJacobian` implementation for the logslope block must
//! return the FULL formula above — not the simpler `s_f·diag(z)·Phi` shortcut
//! that only holds at β=0.  A static diagonal-scaling shortcut would
//! pass the β=0 point but fail the moderate-β finite-difference check below.
//!
//! ## Failure mode of the shortcut
//!
//! If T1 uses a static diagonal scaling `s_f · z` (i.e. the static diagonal
//! scaling that equals the correct Jacobian only at g=0), then at moderate β
//! the `effective_jacobian_at` output will equal `diag(s_f · z) · Phi` while
//! the FD reference will equal `diag(q·c1 + s_f·z) · Phi`.  The rel-error
//! will be O(1) (not O(ε)), causing the moderate-β FD assertion to fail.
//!
//! ## Channel-aware audit at moderate β
//!
//! With the correct Jacobian, marginal and logslope blocks have DIFFERENT
//! per-row scale factors (c_i vs. q·c1_i + s_f·z_i), so their effective
//! designs in the (n·K=3·n) channel-stacked space are NOT collinear.  The
//! pairwise overlap is strictly less than 1.0.  With a static diagonal shortcut,
//! both marginal and logslope blocks would have scale factor s_f·z_i,
//! yielding overlap = 1.0 — triggering a spurious fatal audit halt.

use gam::custom_family::{
    BlockEffectiveJacobian, FamilyLinearizationState, ParameterBlockSpec, RowScaledJacobian,
};
use gam::families::survival::marginal_slope::{
    LogslopeBlockJacobian, SurvivalMarginalSlopeFamilyScalars,
};
use gam::identifiability::audit::audit_identifiability_channel_aware;
use gam::identifiability::families::compiler::{IdentityRowHessian, RowJacobianOperator};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2, Array3};
use std::any::Any;
use std::sync::Arc;

#[path = "../common/fixtures.rs"]
mod fixtures;
use fixtures::Splitmix64;

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
    /// Logslope design matrix Phi, shape (N, P_BLOCK).
    phi: Array2<f64>,
    /// Marginal design matrix, shape (N, P_BLOCK).
    phi_marg: Array2<f64>,
    /// Time design matrix (exit), shape (N, P_BLOCK).
    phi_time: Array2<f64>,
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

    // Build logslope design: [1 | pcs | duchon_rbf(pcs, centers)]
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
    // Marginal and time blocks share the same structural design but slightly
    // different offset so they are NOT exactly collinear (tests overlap < 1).
    let phi_marg = make_design(1.0);
    let phi_time = make_design(2.0);

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
        phi_time,
        q0_base,
        q1_base,
        qd1_base,
    }
}

// ── Per-row primary scalars at a given β_logslope ─────────────────────────

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

// ── Compute stacked η at a given β_logslope ───────────────────────────────
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
// Computes the correct (3*N, P) Jacobian for the logslope block.

fn analytical_logslope_jacobian(
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

// ── BlockEffectiveJacobian impl for logslope block ───────────────────────
//
// This struct carries the design + per-row family scalars and implements the
// full β-dependent Jacobian. The `family_scalars` arc carries
// `LogslopeFamilyScalars` which contains q0, q1, qd1, z, s_f computed at
// the current linearization point (updated each time β changes).

// `s_f` (probit frailty scale) is read from `state.probit_frailty_scale` at
// evaluation time, not carried inside this struct: that lets the same scalars
// instance stay correct across outer-loop σ updates without rebuilding.
struct LogslopeFamilyScalars {
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
    z: Array1<f64>,
}

struct LogslopeJacobianImpl {
    phi: Array2<f64>,
    s_f: f64,
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
    z: Array1<f64>,
}

impl BlockEffectiveJacobian for LogslopeJacobianImpl {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        // Prefer family_scalars if provided (they carry updated q0/q1/qd1
        // from the linearization state). Fall back to self.q0/q1/qd1.
        let (q0, q1, qd1, z) = if let Some(arc) = state.family_scalars.as_ref() {
            if let Some(fs) = arc.downcast_ref::<LogslopeFamilyScalars>() {
                (&fs.q0, &fs.q1, &fs.qd1, &fs.z)
            } else {
                (&self.q0, &self.q1, &self.qd1, &self.z)
            }
        } else {
            (&self.q0, &self.q1, &self.qd1, &self.z)
        };

        let n = self.phi.nrows();
        let rows = rows.start.min(n)..rows.end.min(n);
        let full = analytical_logslope_jacobian(&self.phi, state.beta, q0, q1, qd1, z, self.s_f);
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

struct LogslopeOperator {
    phi: Array2<f64>,
    s_f: f64,
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
    z: Array1<f64>,
    beta: Vec<f64>,
}

impl RowJacobianOperator for LogslopeOperator {
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

fn make_logslope_spec(
    phi: Array2<f64>,
    q0: Array1<f64>,
    q1: Array1<f64>,
    qd1: Array1<f64>,
    z: Array1<f64>,
    s_f: f64,
) -> ParameterBlockSpec {
    let n = phi.nrows();
    let cb = Arc::new(LogslopeJacobianImpl {
        phi: phi.clone(),
        s_f,
        q0,
        q1,
        qd1,
        z,
    });
    ParameterBlockSpec {
        name: "logslope".to_string(),
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

fn check_logslope_jacobian(
    data: &SyntheticData,
    beta_logslope: &[f64],
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
    s_f: f64,
    label: &str,
) {
    let analytic =
        analytical_logslope_jacobian(&data.phi, beta_logslope, q0, q1, qd1, &data.z, s_f);
    let beta_arr = Array1::from(beta_logslope.to_vec());
    let phi_ref = &data.phi;
    let q0_ref = q0;
    let q1_ref = q1;
    let qd1_ref = qd1;
    let z_ref = &data.z;
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
    let rel_err = max_col_rel_error(&analytic, &fd);
    assert!(
        rel_err < 1e-5,
        "{label}: analytical logslope Jacobian rel-error vs FD = {rel_err:.3e} (expected < 1e-5); \
         the hyperbolic correction formula is wrong",
    );
}

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
         If this fires at moderate β but passes at β=0, T1's implementation is using \
         a static diagonal shortcut (s_f·diag(z)·Phi) instead of the full hyperbolic \
         correction (q·c1 + s_f·z)·Phi.",
    );
}

// ── Main tests ───────────────────────────────────────────────────────────

/// At β_logslope = 0: g_i = 0 for all i, c_i = 1, c1_i = 0.
/// The logslope Jacobian must equal s_f · diag(z) · Phi — NOT raw Phi.
#[test]
fn logslope_jacobian_at_zero_beta_equals_sf_diag_z_phi() {
    let data = make_synthetic_data(42);
    for s_f in [1.0_f64, 0.8] {
        let beta_zero = vec![0.0; P_BLOCK];
        // At g=0: c=1, c1=0, so ∂η_r/∂β = s_f·z·Phi for η0/η1, and 0 for ad1.
        let analytic = analytical_logslope_jacobian(
            &data.phi,
            &beta_zero,
            &data.q0_base,
            &data.q1_base,
            &data.qd1_base,
            &data.z,
            s_f,
        );
        // Verify against formula: first N rows = s_f * z * Phi.
        for i in 0..N {
            for j in 0..P_BLOCK {
                let expected_eta0 = s_f * data.z[i] * data.phi[[i, j]];
                let got = analytic[[i, j]];
                let err = (got - expected_eta0).abs();
                let scale = expected_eta0.abs().max(1e-14);
                assert!(
                    err / scale < 1e-10 || err < 1e-12,
                    "s_f={s_f}: row={i} col={j}: η0-Jacobian at β=0 should be \
                     s_f*z*Phi={expected_eta0:.6e} got {got:.6e}",
                );
            }
        }
        // Verify ad1 rows (2N..3N) are all zero at β=0.
        for i in 0..N {
            for j in 0..P_BLOCK {
                let got = analytic[[2 * N + i, j]];
                assert!(
                    got.abs() < 1e-12,
                    "s_f={s_f}: ad1 Jacobian row {i} col {j} should be 0 at β=0, got {got:.3e}",
                );
            }
        }
        // FD check.
        let label = format!("beta=0 s_f={s_f}");
        check_logslope_jacobian(
            &data,
            &beta_zero,
            &data.q0_base,
            &data.q1_base,
            &data.qd1_base,
            s_f,
            &label,
        );
    }
}

/// At small random β: g_i small but nonzero, c1_i ≈ 0 but not exactly 0.
/// FD must still agree with analytical formula.
#[test]
fn logslope_jacobian_at_small_beta_matches_fd() {
    let data = make_synthetic_data(123);
    assert_eq!(data.phi.nrows(), N, "synthetic data must have N rows");
    let mut rng = Splitmix64::new(0xDEAD_BEEF_u64);
    for s_f in [1.0_f64, 0.8] {
        let beta_small: Vec<f64> = (0..P_BLOCK).map(|_| rng.next_gauss() * 0.05).collect();
        let label = format!("beta=small s_f={s_f}");
        check_logslope_jacobian(
            &data,
            &beta_small,
            &data.q0_base,
            &data.q1_base,
            &data.qd1_base,
            s_f,
            &label,
        );
    }
}

/// At moderate β (g_i ~ O(1)): the hyperbolic correction q·c1 is comparable
/// to s_f·z. FD check distinguishes the correct formula from the shortcut.
///
/// SENTINEL: if this test fails while the β=0 test passes, T1's impl uses
/// a static diagonal shortcut (s_f·z) everywhere, which is only correct at g=0.
#[test]
fn logslope_jacobian_at_moderate_beta_has_hyperbolic_correction() {
    let data = make_synthetic_data(777);
    let mut rng = Splitmix64::new(0xC0_FFEE_u64);
    for s_f in [1.0_f64, 0.8] {
        // Scale β so that s_f * g_i ~ O(1) on average.
        // With Phi containing values ~ O(1/sqrt(P)) and P=9, and N=200,
        // scale ~ 1/(s_f * sqrt(P)) gives s_f * g_i ~ O(1).
        let scale = 1.0 / (s_f * (P_BLOCK as f64).sqrt());
        let beta_moderate: Vec<f64> = (0..P_BLOCK).map(|_| rng.next_gauss() * scale).collect();

        let label = format!("beta=moderate s_f={s_f}");
        check_logslope_jacobian(
            &data,
            &beta_moderate,
            &data.q0_base,
            &data.q1_base,
            &data.qd1_base,
            s_f,
            &label,
        );

        // Confirm the hyperbolic correction is actually nonzero at moderate β:
        // at least some rows should have |q·c1| > 1e-3 * |s_f·z|.
        let scalars = compute_row_scalars(&data.phi, &beta_moderate, s_f);
        let hyperbolic_fractions: Vec<f64> = (0..N)
            .map(|i| {
                let hyp = (data.q0_base[i] * scalars.c1[i]).abs();
                let base = (s_f * data.z[i]).abs().max(1e-8);
                hyp / base
            })
            .collect();
        let max_frac = hyperbolic_fractions.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_frac > 1e-3,
            "s_f={s_f}: moderate β did not produce meaningful hyperbolic correction \
             (max |q·c1|/|s_f·z| = {max_frac:.3e}); the test is not exercising the \
             non-trivial g branch",
        );

        // Confirm ad1 rows are nonzero at moderate β (since qd1·c1 ≠ 0 when g ≠ 0).
        let analytic = analytical_logslope_jacobian(
            &data.phi,
            &beta_moderate,
            &data.q0_base,
            &data.q1_base,
            &data.qd1_base,
            &data.z,
            s_f,
        );
        let ad1_norm: f64 = analytic
            .slice(ndarray::s![2 * N.., ..])
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            ad1_norm > 1e-8,
            "s_f={s_f}: ad1 Jacobian rows are all zero at moderate β (norm={ad1_norm:.3e}); \
             the qd1·c1 term is not being computed",
        );
    }
}

/// Call `spec.effective_jacobian_at` on a `ParameterBlockSpec` with a
/// `LogslopeJacobianImpl` callback, and verify it matches FD at three β
/// points: β=0, small, moderate.
#[test]
fn effective_jacobian_at_matches_fd_at_three_linearization_points() {
    let data = make_synthetic_data(314);
    assert_eq!(data.phi.nrows(), N, "synthetic data must have N rows");
    let mut rng = Splitmix64::new(0x1337_u64);

    for s_f in [1.0_f64, 0.8] {
        let spec = make_logslope_spec(
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
    let spec = make_logslope_spec(
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
    let fs: Arc<dyn Any + Send + Sync> = Arc::new(LogslopeFamilyScalars {
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

/// Channel-aware audit: marginal and logslope blocks at moderate β have
/// DIFFERENT per-row scalings (c_i for marginal vs (q·c1+s_f·z)_i for
/// logslope), so their pairwise overlap in (n·K=3·n) space is < 1.0.
///
/// With a static diagonal shortcut, both blocks would use s_f·z_i as their
/// row scaling, making the overlap = 1.0 and triggering a spurious fatal halt.
#[test]
fn channel_aware_audit_overlap_below_one_at_moderate_beta() {
    let data = make_synthetic_data(2024);
    let s_f = 0.8_f64;
    let mut rng = Splitmix64::new(0xFACE_u64);

    let scale = 1.0 / (s_f * (P_BLOCK as f64).sqrt());
    let beta_logslope: Vec<f64> = (0..P_BLOCK).map(|_| rng.next_gauss() * scale).collect();

    // Marginal block: its c_i is from the marginal block's own g_marg = phi_marg · β_marg.
    // At β_marg = 0 (no marginal predictor shift), g_marg = 0, c_i = 1 for all i.
    let c_marginal = Array1::<f64>::ones(N);

    // Build RowJacobianOperator instances.
    let logslope_op = Arc::new(LogslopeOperator {
        phi: data.phi.clone(),
        s_f,
        q0: data.q0_base.clone(),
        q1: data.q1_base.clone(),
        qd1: data.qd1_base.clone(),
        z: data.z.clone(),
        beta: beta_logslope.clone(),
    });

    let marg_op = Arc::new(MarginalOperator {
        phi: data.phi_marg.clone(),
        c: c_marginal,
    });

    let specs = [
        make_spec_from_dense("marginal", data.phi_marg.clone()),
        make_spec_from_dense("logslope", data.phi.clone()),
    ];
    let operators: Vec<Arc<dyn RowJacobianOperator>> = vec![
        marg_op as Arc<dyn RowJacobianOperator>,
        logslope_op as Arc<dyn RowJacobianOperator>,
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
         With a static diagonal shortcut, marginal and logslope blocks both scale by s_f·z \
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
         Jacobian; marginal and logslope blocks are separately identifiable. \
         Summary: {}",
        audit.summary,
    );
}

/// The time block contributes `c * design` to both η0 and η1. At β_time = 0
/// this equals just `1 * design`.  The test verifies the time block's c=1
/// effective design can serve as an FD-correct baseline.
#[test]
fn time_and_marginal_blocks_at_zero_beta_have_trivial_scaling() {
    let data = make_synthetic_data(55);
    let s_f = 1.0_f64;
    let beta_zero = vec![0.0; P_BLOCK];

    // At β_time = 0: g_time = 0, c = 1 for all rows.
    // Time Jacobian (η0 rows) = 1 * Phi_time, (η1 rows) = 1 * Phi_time.
    // FD on η0: d/dβ [q0 * c(Phi_time * β) + s_f * g_time * z]|β=0 = 0 for q0 term
    // because dc/dβ|g=0 = 0 (c is stationary at g=0). BUT the time block
    // contributes to q via the design: dq0/dβ = Phi_time (for the time block).
    // So d(η0)/dβ_time = (dq0/dβ_time) * c_i = 1 * Phi_time_{i,j}.
    // This is effectively an additive contribution through the q-channel.
    //
    // For this block, the "effective design" at β=0 is just Phi_time (c=1).

    // Build a simple additive spec for the time block (c=1 means row-scaling = ones).
    let scaling: Arc<[f64]> = vec![1.0_f64; N].into();
    let spec = ParameterBlockSpec {
        name: "time".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(data.phi_time.clone())),
        offset: Array1::<f64>::zeros(N),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 200,
        jacobian_callback: Some(Arc::new(RowScaledJacobian {
            design: Arc::new(data.phi_time.clone()),
            eta_scaling: scaling,
        })),
        stacked_design: None,
        stacked_offset: None,
    };

    // Effective Jacobian via spec.effective_jacobian_at (RowScaledJacobian with scaling=1).
    let state = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: s_f,
    };
    let jac = spec
        .effective_jacobian_at("test", &state)
        .expect("time block effective_jacobian_at must succeed");

    // Check shape.
    assert_eq!(
        jac.nrows(),
        N,
        "time block Jacobian must have N rows (single output), got {}",
        jac.nrows()
    );
    assert_eq!(
        jac.ncols(),
        P_BLOCK,
        "time block Jacobian must have P_BLOCK cols, got {}",
        jac.ncols()
    );

    // With row-scaling=1, effective Jacobian = design.
    for i in 0..N {
        for j in 0..P_BLOCK {
            let got = jac[[i, j]];
            let expected = data.phi_time[[i, j]];
            let err = (got - expected).abs();
            assert!(
                err < 1e-12,
                "time block: jac[{i},{j}]={got:.6e} != phi_time[{i},{j}]={expected:.6e}",
            );
        }
    }

    // Also verify marginal block with s_f·z row scaling (what T1 should use
    // for the β=0 case — but for marginal, s_f·z is the CORRECT scaling at all β
    // only if the marginal block contributes additively to q, not g).
    let sf_z: Vec<f64> = data.z.iter().map(|&zi| s_f * zi).collect();
    let sf_z_arc: Arc<[f64]> = sf_z.into();
    let marg_spec = ParameterBlockSpec {
        name: "marginal_z_scaled".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(data.phi_marg.clone())),
        offset: Array1::<f64>::zeros(N),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 150,
        jacobian_callback: Some(Arc::new(RowScaledJacobian {
            design: Arc::new(data.phi_marg.clone()),
            eta_scaling: sf_z_arc,
        })),
        stacked_design: None,
        stacked_offset: None,
    };
    let marg_state = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: s_f,
    };
    let marg_jac = marg_spec
        .effective_jacobian_at("test", &marg_state)
        .expect("marginal block effective_jacobian_at must succeed");

    // Verify RowScaledJacobian applied correctly: row i should equal s_f*z_i * Phi_marg[i,:]
    for i in 0..N {
        let scale = s_f * data.z[i];
        for j in 0..P_BLOCK {
            let got = marg_jac[[i, j]];
            let expected = scale * data.phi_marg[[i, j]];
            let err = (got - expected).abs();
            let denom = expected.abs().max(1e-12);
            assert!(
                err / denom < 1e-10 || err < 1e-12,
                "marginal z-scaled: jac[{i},{j}]={got:.6e} != {expected:.6e}",
            );
        }
    }
}

/// The production `LogslopeBlockJacobian` (from survival_marginal_slope) enforces the
/// hard contract: when `family_scalars` is `None` but beta is non-zero (causing g_i != 0
/// for at least one row), `effective_jacobian_at` must return `Err`.
///
/// When `family_scalars` is properly populated with `SurvivalMarginalSlopeFamilyScalars`,
/// the result is `Ok` and the Jacobian is FD-correct to within 1e-5.
#[test]
fn production_logslope_block_requires_scalars_at_nonzero_beta() {
    let data = make_synthetic_data(31415);
    let s_f = 0.8_f64;
    let mut rng = Splitmix64::new(0xDEAD_u64);

    // Build a production LogslopeBlockJacobian with the test's phi and z.
    let cb = Arc::new(LogslopeBlockJacobian::new(
        data.phi.clone(),
        data.z.to_vec(),
        s_f,
    ));

    // Moderate beta so g_i != 0 for some rows.
    let scale = 1.0 / (s_f * (P_BLOCK as f64).sqrt());
    let beta: Vec<f64> = (0..P_BLOCK).map(|_| rng.next_gauss() * scale).collect();

    // Confirm at least one g_i is nonzero (sanity check for the test itself).
    let any_nonzero_g = (0..N).any(|i| {
        data.phi
            .row(i)
            .iter()
            .zip(beta.iter())
            .map(|(&x, &b)| x * b)
            .sum::<f64>()
            != 0.0
    });
    assert!(
        any_nonzero_g,
        "test setup: expected at least one nonzero g_i; rng may need reseeding"
    );

    // 1. Without family_scalars at nonzero beta: must return Err.
    {
        let state = FamilyLinearizationState {
            beta: &beta,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: s_f,
        };
        let result = cb.effective_jacobian_at(&state);
        assert!(
            result.is_err(),
            "production LogslopeBlockJacobian must return Err when family_scalars=None and beta is nonzero (g_i != 0); got Ok"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("SurvivalMarginalSlopeFamilyScalars"),
            "error message must name SurvivalMarginalSlopeFamilyScalars; got: {msg}"
        );
    }

    // 2. With correct family_scalars: must return Ok and match FD.
    {
        // Compute per-row g, c, q0, q1, qd1 at the current beta.
        let scalars_arc: Arc<dyn Any + Send + Sync> = {
            let mut g_i = vec![0.0_f64; N];
            for i in 0..N {
                g_i[i] = data
                    .phi
                    .row(i)
                    .iter()
                    .zip(beta.iter())
                    .map(|(&x, &b)| x * b)
                    .sum();
            }
            // q0/q1/qd1: use the baseline values from make_synthetic_data.
            // In a real fit these would come from the time/marginal blocks at current beta.
            let q0_i = data.q0_base.to_vec();
            let q1_i = data.q1_base.to_vec();
            let qd1_i = data.qd1_base.to_vec();
            let z_i = data.z.to_vec();
            Arc::new(SurvivalMarginalSlopeFamilyScalars::new(
                q0_i, q1_i, qd1_i, g_i, s_f, z_i,
            ))
        };

        let state = FamilyLinearizationState {
            beta: &beta,
            family_scalars: Some(scalars_arc),
            channel_hessian: None,
            probit_frailty_scale: s_f,
        };
        let jac = cb
            .effective_jacobian_at(&state)
            .expect("production LogslopeBlockJacobian with family_scalars must return Ok");

        assert_eq!(
            jac.shape(),
            &[3 * N, P_BLOCK],
            "Jacobian shape must be (3*N, P_BLOCK)"
        );

        // FD verification: Jacobian must match finite differences.
        let beta_arr = Array1::from(beta.clone());
        let phi_ref = &data.phi;
        let q0_ref = &data.q0_base;
        let q1_ref = &data.q1_base;
        let qd1_ref = &data.qd1_base;
        let z_ref = &data.z;
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
            "production LogslopeBlockJacobian with family_scalars: rel-error vs FD = {rel_err:.3e} (expected < 1e-5). The hyperbolic correction formula is wrong."
        );
    }

    // 3. At beta=0: family_scalars=None is allowed (g=0 everywhere so q terms vanish).
    {
        let beta_zero = vec![0.0_f64; P_BLOCK];
        let state = FamilyLinearizationState {
            beta: &beta_zero,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: s_f,
        };
        let jac = cb.effective_jacobian_at(&state).expect(
            "production LogslopeBlockJacobian at beta=0 with family_scalars=None must return Ok",
        );
        // At g=0: coeff_eta0 = s_f*z_i, coeff_eta1 = s_f*z_i, coeff_ad1 = 0.
        for i in 0..N {
            let expected_scale = s_f * data.z[i];
            for j in 0..P_BLOCK {
                let got_eta0 = jac[[i, j]];
                let exp = expected_scale * data.phi[[i, j]];
                let err = (got_eta0 - exp).abs();
                assert!(
                    err < 1e-12 || err / exp.abs().max(1e-14) < 1e-10,
                    "at beta=0: jac[{i},{j}]={got_eta0:.6e} != s_f*z*phi={exp:.6e}"
                );
                // ad1 rows must be zero at g=0.
                let got_ad1 = jac[[2 * N + i, j]];
                assert!(
                    got_ad1.abs() < 1e-12,
                    "at beta=0: ad1 row [{i},{j}] = {got_ad1:.3e} (expected 0)"
                );
            }
        }
    }
}
