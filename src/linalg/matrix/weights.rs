//! Typed weight views.
//!
//! Sign character of a working-weight vector is a *static* property of the
//! caller's math (Fisher-scoring vs observed-Hessian, PSD-Gram vs asymmetric
//! `X_iᵀ W X_j`, IRLS-diagonal vs derivative-correction). Encoding it in the
//! type system pushes the runtime sign-scan back to the call site where the
//! vector was constructed — one scan, at the boundary — instead of asserting
//! inside every kernel that consumes the weights.
//!
//! Conventions:
//! * `PsdWeightsView<'_>` is owned/lifetime-bound to a 1D float view whose
//!   constructor has already discharged the `w_i ≥ 0` obligation. PSD-Gram
//!   kernels (`weighted_crossprod_dense_view`, `dense_diag_gram_view`,
//!   `sparse_csr_weighted_xtwx_*`) accept only this view, so the `assert!`
//!   that previously fired inside the kernels migrates entirely to
//!   `PsdWeights::try_new`. PSD callers either go through this constructor,
//!   `from_view_unchecked` (audited site, recorded reason), or
//!   `SignedWeightsView::as_psd` (consolidating the few scan sites that
//!   still need to ask the question at runtime — e.g. PIRLS step
//!   acceptance).
//! * `SignedWeightsView<'_>` is the universal sign-honest view, freely
//!   constructable from any `&Array1<f64>` / `ArrayView1<'_, f64>` / `&[f64]`.
//!   The diagonal-Gram kernels and the shared per-row accumulator
//!   `weighted_crossprod_dense_rows` consume it — they are linear in `w` and
//!   sign-correct without a PSD precondition (and are reused by the
//!   asymmetric `X_iᵀ W X_j` path inside `BlockDesignOperator::cross_block`,
//!   where `c · X v` is genuinely signed).
//!
//! The two newtypes are zero-cost: `repr(transparent)` over `ArrayView1<'_,
//! f64>`, with `into_view()` / `as_slice()` / `len()` projections so kernel
//! bodies still see the underlying array view.

use ndarray::{Array1, ArrayView1};
use std::ops::Deref;
use std::sync::Arc;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SignedWeightsView<'a>(ArrayView1<'a, f64>);

impl<'a> SignedWeightsView<'a> {
    /// Borrow any `ArrayView1<'_, f64>` as a sign-honest weight view. This is
    /// free of obligation: signed weights are the most general case, and the
    /// consumers (`weighted_crossprod_dense_rows`, observed-Hessian Gram
    /// kernels, `BlockDesignOperator::cross_block`) all do sign-correct math.
    #[inline]
    pub fn new(view: ArrayView1<'a, f64>) -> Self {
        Self(view)
    }

    /// Borrow an `&Array1<f64>` as a sign-honest weight view.
    #[inline]
    pub fn from_array(array: &'a Array1<f64>) -> Self {
        Self(array.view())
    }

    /// Borrow a contiguous slice as a sign-honest weight view.
    #[inline]
    pub fn from_slice(slice: &'a [f64]) -> Self {
        Self(ArrayView1::from(slice))
    }

    /// Underlying `ArrayView1<'_, f64>` for kernel bodies.
    #[inline]
    pub fn view(&self) -> ArrayView1<'a, f64> {
        self.0
    }

    /// Length of the weight vector (= row count of the design it weights).
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True iff the underlying view is empty (parity with `Array1::is_empty`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Contiguous slice if the underlying view is in standard layout.
    #[inline]
    pub fn as_slice(&self) -> Option<&[f64]> {
        self.0.as_slice()
    }

    /// Attempt to promote a signed view to a PSD view. Performs one linear
    /// sign-scan; consolidates the runtime check at the few sites that still
    /// need to ask the question (e.g. PIRLS step acceptance, where the same
    /// scan was previously inlined as `weights.iter().any(|&w| w < 0.0)`).
    #[inline]
    pub fn as_psd(self) -> Option<PsdWeightsView<'a>> {
        if self.0.iter().all(|&w| w >= 0.0) {
            Some(PsdWeightsView(self.0))
        } else {
            None
        }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct PsdWeightsView<'a>(ArrayView1<'a, f64>);

impl<'a> PsdWeightsView<'a> {
    /// Construct a PSD weight view, discharging the `w_i ≥ 0` precondition
    /// once at the call site. The previous runtime `assert!` inside
    /// `weighted_crossprod_dense_view` / `dense_diag_gram_view` migrates entirely to this
    /// constructor — kernels that accept `PsdWeightsView` no longer need to
    /// recheck.
    #[inline]
    pub fn try_new(view: ArrayView1<'a, f64>) -> Result<Self, String> {
        if view.iter().all(|&w| w >= 0.0) {
            Ok(Self(view))
        } else {
            Err("PsdWeights::try_new: weights must be nonneg (use SignedWeightsView for observed-Hessian assembly)".to_string())
        }
    }

    /// As `try_new`, taking an owned `&Array1<f64>`.
    #[inline]
    pub fn try_from_array(array: &'a Array1<f64>) -> Result<Self, String> {
        Self::try_new(array.view())
    }

    /// Construct a PSD view *without* re-scanning. The caller asserts (in
    /// human review) that the weights are nonneg by construction — e.g. the
    /// canonical-link Fisher weights `μ(1-μ)` for Binomial-logit, the squared
    /// magnitude of a vector, or the result of a prior scan that the type
    /// system cannot reproject through the call graph (e.g. across an FFI
    /// boundary). Pair with a comment explaining *why* the scan is redundant.
    #[inline]
    pub fn from_view_unchecked(view: ArrayView1<'a, f64>) -> Self {
        Self(view)
    }

    /// Forget the PSD guarantee and degrade to the sign-honest view. The
    /// signed kernels accept this view directly; useful when the same buffer
    /// is consumed by both a PSD-Gram path and a sign-honest accumulator.
    #[inline]
    pub fn as_signed(self) -> SignedWeightsView<'a> {
        SignedWeightsView(self.0)
    }

    /// Underlying `ArrayView1<'_, f64>` for kernel bodies.
    #[inline]
    pub fn view(&self) -> ArrayView1<'a, f64> {
        self.0
    }

    /// Length of the weight vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True iff the underlying view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Contiguous slice if the underlying view is in standard layout.
    #[inline]
    pub fn as_slice(&self) -> Option<&[f64]> {
        self.0.as_slice()
    }
}

/// Owned, shareable counterpart to [`SignedWeightsView`].
///
/// A handful of long-lived hyper-derivative operator structs in
/// `solver/reml/{hyper,unified}.rs` (`TauTauPairHyperOperator`,
/// `ImplicitHyperOperator`, `SparseDirectionalHyperOperator`) cache the
/// observed-Hessian working weight diagonal as `Arc<Array1<f64>>` and consume
/// it via several distinct signed kernels inside their `mul_vec` bodies
/// (`Wᵀ X v`, `Wᵀ X_τ v`, `Xᵀ diag(c ⊙ X_τ β̂) X v`, ...). Encoding the
/// sign character at the struct boundary closes the residual implicit-sign
/// gap that the function-boundary [`SignedWeightsView`] / [`PsdWeightsView`]
/// could not reach: those views are constructed at the kernel call site, so
/// the cached struct field is the only place the sign character could
/// otherwise leak as untyped `Arc<Array1<f64>>`.
///
/// The newtype derefs to `Array1<f64>` so existing arithmetic like
/// `&*self.w_diag * &x_v` is unchanged. `view_signed()` produces the
/// borrowed function-boundary view when a kernel is called.
#[derive(Clone)]
#[repr(transparent)]
pub struct SignedWeightsArc(Arc<Array1<f64>>);

impl SignedWeightsArc {
    /// Wrap an existing `Arc<Array1<f64>>` as a sign-honest owned weight
    /// buffer. Cheap (Arc clone is a refcount bump); no allocation, no scan.
    #[inline]
    pub fn from_arc(arc: Arc<Array1<f64>>) -> Self {
        Self(arc)
    }

    /// Take ownership of an `Array1<f64>` and wrap it in an Arc.
    #[inline]
    pub fn from_array(array: Array1<f64>) -> Self {
        Self(Arc::new(array))
    }

    /// Borrow as a function-boundary [`SignedWeightsView`] for crossing into
    /// a signed kernel (`weighted_crossprod_dense_rows`, `xt_diag_x_signed_op`,
    /// `BlockDesignOperator::cross_block`).
    #[inline]
    pub fn view_signed(&self) -> SignedWeightsView<'_> {
        SignedWeightsView::from_array(self.0.as_ref())
    }

    /// Inner `Arc<Array1<f64>>` for sites that genuinely need the shared
    /// pointer (e.g. cloning into a sibling operator that holds its own
    /// `SignedWeightsArc`). Prefer `Clone` on the newtype itself when the
    /// destination accepts a `SignedWeightsArc`.
    #[inline]
    pub fn as_arc(&self) -> &Arc<Array1<f64>> {
        &self.0
    }
}

impl Deref for SignedWeightsArc {
    type Target = Array1<f64>;

    #[inline]
    fn deref(&self) -> &Array1<f64> {
        self.0.as_ref()
    }
}

impl AsRef<Array1<f64>> for SignedWeightsArc {
    #[inline]
    fn as_ref(&self) -> &Array1<f64> {
        self.0.as_ref()
    }
}
