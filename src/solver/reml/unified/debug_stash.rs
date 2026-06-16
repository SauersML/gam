//! Opt-in REML diagnostic capture for finite-difference investigation tests.
//!
//! Production evaluations only check the capture guard and skip every expensive
//! stash path unless a test explicitly requests diagnostics.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Number of live [`CaptureGuard`]s. The ext-gradient path only assembles the
/// EIG-DECOMP diagnostic stash while this is non-zero. Filling the stash is not
/// free: it recomputes the psi drift, runs additional spectral traces, and
/// repeats the cubic IFT-correction pass for the captured coordinate.
static CAPTURE_REQUESTS: AtomicUsize = AtomicUsize::new(0);

/// True while at least one [`CaptureGuard`] is alive.
pub(crate) fn capture_requested() -> bool {
    CAPTURE_REQUESTS.load(Ordering::Relaxed) > 0
}

/// RAII opt-in to EIG-DECOMP stash capture; see [`capture_requested`].
///
/// Counted instead of boolean so concurrently-running tests cannot disable
/// each other's capture. Stash delivery itself stays per-thread.
#[must_use = "capture stops when the guard is dropped"]
pub(crate) struct CaptureGuard(());

impl CaptureGuard {
    pub(crate) fn request() -> Self {
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
pub(crate) struct TermStash {
    /// Per-row diagonal of term4: `c * X_tau_beta`.
    pub(crate) c_x_tau_beta_diag: Option<ndarray::Array1<f64>>,
    /// `X * v_psi` per row, where `v_psi = hop^-1 * stored_g`.
    pub(crate) c_x_v_psi_diag: Option<ndarray::Array1<f64>>,
    /// Unprojected eigenmode trace.
    pub(crate) unprojected_tr: Option<f64>,
    /// The production `trace_logdet_i` value that enters the outer gradient.
    pub(crate) production_tr: Option<f64>,
    /// Whether `penalty_subspace_trace` was active for this coordinate.
    pub(crate) projection_active: Option<bool>,
    /// Frozen-beta basis/penalty drift component of the psi logdet trace.
    pub(crate) frozen_tr: Option<f64>,
    /// Cubic IFT-correction component of the psi logdet trace.
    pub(crate) correction_tr: Option<f64>,
    /// Cubic correction recomputed with the projected pseudo-inverse direction.
    pub(crate) correction_tr_proj: Option<f64>,
    /// Cost-derivative `a` term entering the outer gradient.
    pub(crate) coord_a: Option<f64>,
    /// Penalty-logdet derivative for this coordinate.
    pub(crate) coord_ld_s: Option<f64>,
    /// Value component `log|H+S_lambda|_+`.
    pub(crate) coord_log_det_h: Option<f64>,
    /// Value component `log|S_lambda|_+`.
    pub(crate) coord_log_det_s: Option<f64>,
    /// Total outer objective at the captured coordinate.
    pub(crate) coord_cost: Option<f64>,
    /// Inner KKT residual infinity norm.
    pub(crate) inner_kkt_residual_inf: Option<f64>,
    /// Whether the batched envelope-only outer-gradient fast path fired.
    pub(crate) batched_envelope_override_fired: Option<bool>,
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
        inner_kkt_residual_inf: None,
        batched_envelope_override_fired: None,
    }) };
}

static A_SPLIT_SINK: std::sync::Mutex<Option<(f64, f64)>> = std::sync::Mutex::new(None);

/// Record the first psi-coordinate's `a = a_likelihood + a_penalty_quadratic`
/// split. Overwrites; callers gate this to the first extended coordinate.
pub(crate) fn store_a_split(a_likelihood: f64, a_penalty_quadratic: f64) {
    if let Ok(mut slot) = A_SPLIT_SINK.lock() {
        *slot = Some((a_likelihood, a_penalty_quadratic));
    }
}

static KKT_PROBE_SINK: std::sync::Mutex<Option<(f64, bool)>> = std::sync::Mutex::new(None);

/// Record the inner KKT residual inf-norm and whether the batched
/// envelope-only outer-gradient override fired.
pub(crate) fn store_kkt_probe(residual_inf: f64, batched_override_fired: bool) {
    if let Ok(mut slot) = KKT_PROBE_SINK.lock() {
        *slot = Some((residual_inf, batched_override_fired));
    }
}

/// Replace the calling thread's [`TermStash`].
pub(crate) fn store_terms(stash: TermStash) {
    TERMS.with(|cell| *cell.borrow_mut() = stash);
}

#[cfg(test)]
mod debug_stash_tests {
    use super::*;

    impl CaptureGuard {
        pub(crate) fn take_terms(&self) -> TermStash {
            TERMS.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
        }

        pub(crate) fn take_a_split(&self) -> Option<(f64, f64)> {
            A_SPLIT_SINK.lock().ok().and_then(|mut slot| slot.take())
        }

        pub(crate) fn take_kkt_probe(&self) -> Option<(f64, bool)> {
            KKT_PROBE_SINK.lock().ok().and_then(|mut slot| slot.take())
        }
    }
}
