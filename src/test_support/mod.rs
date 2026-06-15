//! Generic testing utilities.

pub mod cli_harness;
pub mod reference;

use crate::families::custom_family::{ParameterBlockSpec, PenaltyMatrix};
use crate::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator};
use crate::resource::MatrixMaterializationError;
use ndarray::{Array1, Array2, Axis, array, s};
use std::ops::Range;
use std::sync::Arc;

pub struct BinomialLocationScaleBaseFixture {
    pub n: usize,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub threshold_design: DesignMatrix,
    pub log_sigma_design: DesignMatrix,
    pub threshold_spec: ParameterBlockSpec,
    pub log_sigma_spec: ParameterBlockSpec,
}

pub fn binomial_location_scale_base_fixture() -> BinomialLocationScaleBaseFixture {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design =
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem((n, 1), 1.0)));
    let log_sigma_design =
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem((n, 1), 1.0)));
    let threshold_spec = ParameterBlockSpec {
        name: "threshold".to_string(),
        design: threshold_design.clone(),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.2]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let log_sigma_spec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: log_sigma_design.clone(),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![-0.2],
        initial_beta: Some(array![-0.1]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    BinomialLocationScaleBaseFixture {
        n,
        y,
        weights,
        threshold_design,
        log_sigma_design,
        threshold_spec,
        log_sigma_spec,
    }
}

#[derive(Clone)]
struct NoDensifyOperator {
    dense: Array2<f64>,
}

impl LinearOperator for NoDensifyOperator {
    fn nrows(&self) -> usize {
        self.dense.nrows()
    }

    fn ncols(&self) -> usize {
        self.dense.ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.dot(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.t().dot(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "NoDensifyOperator weight length mismatch: weights={}, nrows={}",
                weights.len(),
                self.nrows()
            ));
        }
        let weighted = &self.dense * &weights.view().insert_axis(Axis(1));
        Ok(self.dense.t().dot(&weighted))
    }
}

impl DenseDesignOperator for NoDensifyOperator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        out.assign(&self.dense.slice(s![rows, ..]));
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        // `NoDensifyOperator` is a test fixture asserting that
        // operator-aware code paths never densify.
        // SAFETY: a call here means a code path under test bypassed
        // `row_chunk_into` and tried to materialize — the regression
        // this fixture is designed to catch.
        panic!("NoDensifyOperator must stay lazy")
    }
}

pub fn no_densify_design(dense: Array2<f64>) -> DesignMatrix {
    DesignMatrix::from(DenseDesignMatrix::from(Arc::new(NoDensifyOperator {
        dense,
    })))
}

/// Assert that a central difference of an array-producing function matches the analytical derivative.
#[macro_export]
macro_rules! assert_central_difference_array {
    ($x:expr, $h:expr, |$var:ident| $eval:expr, $analytical:expr, $tol:expr) => {
        let f_plus = {
            let $var = $x + $h;
            $eval
        };
        let f_minus = {
            let $var = $x - $h;
            $eval
        };
        assert_eq!(f_plus.len(), $analytical.len());
        for j in 0..$analytical.len() {
            let fd = (f_plus[j] - f_minus[j]) / (2.0 * $h);
            approx::assert_abs_diff_eq!(fd, $analytical[j], epsilon = $tol);
        }
    };
}

/// Asserts that a finite difference dense matrix matches an analytically
/// computed directional derivative matrix to a *relative* tolerance
/// `rel_tol·(1 + |analytic|)`, plus component-wise sign agreement.
///
/// Use this (rather than the absolute-tolerance [`assert_matrix_derivativefd`])
/// when the comparison's dominant components are O(0.1–1) and the finite
/// difference is contaminated by a small, non-smooth solver channel — e.g. an
/// adaptive PIRLS stabilization ridge whose magnitude shifts discontinuously
/// across the ± FD re-solves. There the exact analytic IFT derivative (which
/// correctly excludes that solver-only ridge) and the FD disagree by a fixed
/// *fraction* of the component magnitude, not a fixed absolute amount, so an
/// absolute bound tuned for the small components is spuriously tight on the
/// large ones. The two underlying derivative channels are validated separately
/// against their own FDs, so this asserts the composite to the achievable
/// relative precision rather than weakening the per-channel checks (gam#855).
pub fn assert_matrix_derivativefd_rel(
    fd: &Array2<f64>,
    analytic: &Array2<f64>,
    rel_tol: f64,
    label: &str,
) {
    assert_eq!(analytic.dim(), fd.dim(), "{} dimensions must match", label);
    for i in 0..analytic.nrows() {
        for j in 0..analytic.ncols() {
            let analytic_ij = analytic[[i, j]];
            let fd_ij = fd[[i, j]];
            let tol = rel_tol * (1.0 + analytic_ij.abs());
            if analytic_ij.abs() > tol && fd_ij.abs() > tol {
                assert_eq!(
                    analytic_ij.signum(),
                    fd_ij.signum(),
                    "{} sign mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic_ij,
                    fd_ij
                );
            }
            let diff = (analytic_ij - fd_ij).abs();
            assert!(
                diff <= tol,
                "{} value mismatch at ({}, {}): analytic={}, fd={}, abs_diff={}, rel_tol={}, tol={}",
                label,
                i,
                j,
                analytic_ij,
                fd_ij,
                diff,
                rel_tol,
                tol
            );
        }
    }
}

/// Asserts that a finite difference dense matrix closely matches an analytically computed
/// directional derivative matrix, both in tolerance and in component-wise sign.
pub fn assert_matrix_derivativefd(fd: &Array2<f64>, analytic: &Array2<f64>, tol: f64, label: &str) {
    assert_eq!(analytic.dim(), fd.dim(), "{} dimensions must match", label);
    for i in 0..analytic.nrows() {
        for j in 0..analytic.ncols() {
            let analytic_ij = analytic[[i, j]];
            let fd_ij = fd[[i, j]];
            let diff = (analytic_ij - fd_ij).abs();

            if analytic_ij.abs() > tol && fd_ij.abs() > tol {
                assert_eq!(
                    analytic_ij.signum(),
                    fd_ij.signum(),
                    "{} sign mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic_ij,
                    fd_ij
                );
            }
            assert!(
                diff <= tol,
                "{} value mismatch at ({}, {}): analytic={}, fd={}, abs_diff={}, tol={}",
                label,
                i,
                j,
                analytic_ij,
                fd_ij,
                diff,
                tol
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Debug stash: thread-local capture of (op_total, U) from the ext-grad path,
//  used by the iso-κ Duchon FD investigation test. Empty in production runs.
//  Moved here from src/solver/reml/unified.rs so the `test_support` crate-wide
//  test module hosts it without an item-level `#[cfg(test)]` annotation.
// ═══════════════════════════════════════════════════════════════════════════

pub mod debug_stash {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Number of live [`CaptureGuard`]s. The ext-gradient path in
    /// `solver::reml::unified` only assembles the EIG-DECOMP diagnostic stash
    /// while this is non-zero. Filling the stash is NOT free — it recomputes
    /// the ψ-drift, runs three additional spectral traces (including the
    /// unprojected full-space one) and a second cubic IFT-correction pass per
    /// outer gradient eval — so production fits must never pay for it. Tests
    /// that consume [`take_terms`] opt in by holding a guard across the eval.
    static CAPTURE_REQUESTS: AtomicUsize = AtomicUsize::new(0);

    /// True while at least one [`CaptureGuard`] is alive.
    pub fn capture_requested() -> bool {
        CAPTURE_REQUESTS.load(Ordering::Relaxed) > 0
    }

    /// RAII opt-in to EIG-DECOMP stash capture; see [`capture_requested`].
    /// Counted (not boolean) so concurrently-running tests cannot disable
    /// each other's capture: the flag stays up until every guard drops.
    /// Stash delivery itself stays per-thread (TLS via the per-call sink),
    /// so concurrent captures do not interleave.
    #[must_use = "capture stops when the guard is dropped"]
    pub struct CaptureGuard(());

    impl CaptureGuard {
        pub fn request() -> Self {
            CAPTURE_REQUESTS.fetch_add(1, Ordering::Relaxed);
            Self(())
        }
    }

    impl Drop for CaptureGuard {
        fn drop(&mut self) {
            CAPTURE_REQUESTS.fetch_sub(1, Ordering::Relaxed);
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct TermStash {
        /// Per-row diagonal of term4: `c · X_τβ̂` (n-vector).
        pub c_x_tau_beta_diag: Option<ndarray::Array1<f64>>,
        /// `X · v_ψ` per row, where v_ψ = hop⁻¹·stored_g. The correction
        /// matrix is `−X' diag(c · X · v_ψ) X`, so multiplying this by c
        /// (from PIRLS) gives the diagonal entering the correction sandwich.
        pub c_x_v_psi_diag: Option<ndarray::Array1<f64>>,
        /// Unprojected eigenmode trace Σ φ'(σ_j)·(Uᵀ op U)_jj.
        pub unprojected_tr: Option<f64>,
        /// The production `trace_logdet_i` value that actually enters the
        /// outer gradient.
        pub production_tr: Option<f64>,
        /// Whether `penalty_subspace_trace` was Some for this coordinate.
        pub projection_active: Option<bool>,
        /// #901-layer-2 split: `tr(K · B_i)` — the FROZEN-β̂ basis/penalty
        /// drift component of the ψ logdet trace (no cubic IFT correction).
        pub frozen_tr: Option<f64>,
        /// #901-layer-2 split: `tr(K · D_βH[−v_i])` — the cubic IFT-correction
        /// component of the ψ logdet trace.
        pub correction_tr: Option<f64>,
        /// #901-layer-2 candidate fix: the cubic correction recomputed with
        /// the IFT direction taken from the kernel's own pseudo-inverse
        /// `v = H_pen⁺·coord.g` instead of the full `hop.solve`.
        pub correction_tr_proj: Option<f64>,
        /// HVP ψ-gradient attribution (#740): the cost-derivative `a` term that
        /// enters the outer gradient as `a + ½·logdet_h − ½·logdet_s`. Exposed
        /// so a per-component FD of the outer VALUE can be matched against each
        /// analytic piece (`a` ↔ FD of `−ℓ(β̂)+½β̂ᵀSλβ̂`, `½·production_tr` ↔ FD
        /// of `½log|H+Sλ|`, `½·ld_s` ↔ FD of `½log|Sλ|₊`) to localize the
        /// disagreement to a single term rather than the assembled gradient.
        pub coord_a: Option<f64>,
        /// HVP ψ-gradient attribution (#740): the `∂log|Sλ|₊/∂ψ` penalty-logdet
        /// derivative `ld_s` for this coordinate (see [`Self::coord_a`]).
        pub coord_ld_s: Option<f64>,
        /// HVP ψ-gradient attribution (#740): the VALUE component `log|H+Sλ|₊`
        /// (the `½` factor lives in the outer cost). FD of this across ψ±eps (β̂
        /// re-solved each side) is the TOTAL logdet-H ψ-derivative (frozen B_i +
        /// IFT) and is matched against `production_tr` to test probe (b) — is the
        /// frozen `∂W/∂ψ` logdet-H drift the true frozen-β logdet derivative.
        pub coord_log_det_h: Option<f64>,
        /// HVP ψ-gradient attribution (#740): the VALUE component `log|Sλ|₊`.
        pub coord_log_det_s: Option<f64>,
        /// HVP ψ-gradient attribution (#740): the TOTAL outer cost (objective)
        /// `[−ℓ(β̂)+½β̂ᵀSλβ̂] + ½log|H+Sλ|₊ − ½log|Sλ|₊`. FD of
        /// `cost − ½·log_det_h + ½·log_det_s` (the logdet-EXCLUDED cost piece)
        /// across ψ±eps is matched against `coord_a` to test probe (a) — does the
        /// `a` term carry the full likelihood/penalty-vs-ψ derivative at β̂.
        pub coord_cost: Option<f64>,
    }

    thread_local! {
        static TERMS: RefCell<TermStash> = const { RefCell::new(TermStash {
            c_x_tau_beta_diag: None,
            c_x_v_psi_diag: None,
            unprojected_tr: None,
            production_tr: None,
            projection_active: None,
            frozen_tr: None,
            correction_tr: None,
            correction_tr_proj: None,
            coord_a: None,
            coord_ld_s: None,
            coord_log_det_h: None,
            coord_log_det_s: None,
            coord_cost: None,
        }) };
    }

    pub fn take_terms() -> TermStash {
        TERMS.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
    }

    /// HVP ψ-gradient attribution (#740): global sink for the `a`-channel split
    /// `(a_likelihood, a_penalty_quadratic)` of the FIRST ψ (ext) coordinate.
    /// Written from `build_psi_hyper_coords` (which may run on a different thread
    /// than the gradient par_iter that fills [`TermStash`], so a process-global
    /// `Mutex` is used rather than the per-thread `TERMS` slot) and read by the
    /// diagnostic test. Only set while a [`CaptureGuard`] is live.
    static A_SPLIT_SINK: std::sync::Mutex<Option<(f64, f64)>> = std::sync::Mutex::new(None);

    /// Record the first ψ-coordinate's `a = a_likelihood + a_penalty_quadratic`
    /// split. Overwrites; the assembly visits coordinates in θ order, so the
    /// caller gates this to the first ext coordinate.
    pub fn store_a_split(a_likelihood: f64, a_penalty_quadratic: f64) {
        if let Ok(mut slot) = A_SPLIT_SINK.lock() {
            *slot = Some((a_likelihood, a_penalty_quadratic));
        }
    }

    /// Drain the recorded `a`-channel split.
    pub fn take_a_split() -> Option<(f64, f64)> {
        A_SPLIT_SINK.lock().ok().and_then(|mut slot| slot.take())
    }

    /// Replace the calling thread's `TermStash` with `stash`. Called by
    /// `reml_laml_evaluate` from the calling thread AFTER its ext-coord
    /// par_iter has produced the stash on a rayon worker and returned
    /// it via a per-call mutex sink. Tests read the stored stash via
    /// `take_terms()` from the same thread.
    pub fn store_terms(stash: TermStash) {
        TERMS.with(|cell| *cell.borrow_mut() = stash);
    }
}
