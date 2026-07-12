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
//! * `FiniteSignedWeightsView<'_>` is the universal weighted-operator view:
//!   negative entries are retained, while one deterministic scan rejects the
//!   first nonfinite row before a Gram/Hessian kernel can mutate output.
//! * `SignedWeightsView<'_>` is an unvalidated row-geometry borrow for APIs
//!   that perform their own joint certificate over weights and companion
//!   arrays. It is deliberately not accepted by weighted matrix operators.
//!
//! The view newtypes are zero-cost: `repr(transparent)` over `ArrayView1<'_,
//! f64>`, with narrow projections so kernel bodies still see the underlying
//! array view.

use ndarray::{Array1, ArrayView1};
use std::ops::Deref;
use std::sync::Arc;

/// A sign-honest weight diagonal whose entries have all been certified finite.
///
/// The distinction from [`SignedWeightsView`] is operational rather than
/// algebraic: matrix-free normal products are evaluated many times inside PCG,
/// so they must certify the row diagonal once at the solve boundary instead of
/// rescanning it on every matvec.  Negative and signed-zero values are retained
/// exactly; only `NaN` and infinities are rejected.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct FiniteSignedWeightsView<'a>(ArrayView1<'a, f64>);

impl<'a> FiniteSignedWeightsView<'a> {
    /// Certify a signed weight vector.  Failure names the smallest offending
    /// row, so the result is deterministic and independent of parallelism.
    #[inline]
    pub fn try_new(view: ArrayView1<'a, f64>) -> Result<Self, String> {
        if let Some((row, value)) = view
            .iter()
            .copied()
            .enumerate()
            .find(|(_, w)| !w.is_finite())
        {
            return Err(format!(
                "non-finite weight at row {row}: {value:?}; every weight must be finite"
            ));
        }
        Ok(Self(view))
    }

    #[inline]
    pub fn try_from_array(array: &'a Array1<f64>) -> Result<Self, String> {
        Self::try_new(array.view())
    }

    #[inline]
    pub fn view(&self) -> ArrayView1<'a, f64> {
        self.0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SignedWeightsView<'a>(ArrayView1<'a, f64>);

impl<'a> SignedWeightsView<'a> {
    /// Borrow any `ArrayView1<'_, f64>` for row-geometry APIs that perform
    /// their own full certificate. Weighted matrix operators require
    /// [`FiniteSignedWeightsView`] instead.
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
        PsdWeightsView::try_new(self.0).ok()
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
        for (row, &weight) in view.iter().enumerate() {
            if !weight.is_finite() {
                return Err(format!(
                    "PsdWeightsView::try_new: non-finite weight at row {row}: {weight:?}"
                ));
            }
            if weight < 0.0 {
                return Err(format!(
                    "PsdWeightsView::try_new: negative weight at row {row}: {weight:?}; use SignedWeightsView for observed-Hessian assembly"
                ));
            }
        }
        Ok(Self(view))
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
/// (`Wᵀ X v`, `Wᵀ X_τ v`, ...). Encoding the sign character at the struct
/// boundary closes the residual implicit-sign gap that a function-boundary
/// borrowed view could not reach: those views are constructed at the call site, so
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

    /// Borrow as an unvalidated function-boundary [`SignedWeightsView`] for
    /// row-geometry consumers that perform their own joint certificate.
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── SignedWeightsView ─────────────────────────────────────────────────────

    #[test]
    fn signed_view_from_slice_len_and_values() {
        let s = [1.0_f64, -2.0, 3.0];
        let w = SignedWeightsView::from_slice(&s);
        assert_eq!(w.len(), 3);
        assert!(!w.is_empty());
        assert_eq!(w.as_slice().unwrap(), &s);
    }

    #[test]
    fn signed_view_from_array_round_trips() {
        let a = array![5.0_f64, -1.0];
        let w = SignedWeightsView::from_array(&a);
        assert_eq!(w.len(), 2);
        assert_eq!(w.view()[0], 5.0);
        assert_eq!(w.view()[1], -1.0);
    }

    #[test]
    fn signed_view_empty_is_empty() {
        let s: [f64; 0] = [];
        let w = SignedWeightsView::from_slice(&s);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn signed_view_as_psd_succeeds_when_all_nonneg() {
        let a = array![0.0_f64, 1.0, 2.0];
        let w = SignedWeightsView::from_array(&a);
        assert!(w.as_psd().is_some());
    }

    #[test]
    fn signed_view_as_psd_fails_on_negative_entry() {
        let a = array![1.0_f64, -0.001, 2.0];
        let w = SignedWeightsView::from_array(&a);
        assert!(w.as_psd().is_none());
    }

    // ── PsdWeightsView ────────────────────────────────────────────────────────

    #[test]
    fn psd_try_new_ok_for_all_nonneg() {
        let a = array![0.0_f64, 1.0, 2.0];
        assert!(PsdWeightsView::try_new(a.view()).is_ok());
    }

    #[test]
    fn psd_try_new_ok_for_all_zeros() {
        let a = array![0.0_f64, 0.0];
        assert!(PsdWeightsView::try_new(a.view()).is_ok());
    }

    #[test]
    fn psd_try_new_err_for_negative_entry() {
        let a = array![1.0_f64, -1e-10, 2.0];
        assert!(PsdWeightsView::try_new(a.view()).is_err());
    }

    #[test]
    fn psd_try_new_rejects_positive_infinity() {
        let a = array![1.0_f64, f64::INFINITY];
        let err = PsdWeightsView::try_new(a.view())
            .err()
            .expect("infinite PSD weight must be rejected");
        assert!(err.contains("row 1"), "unexpected diagnostic: {err}");
    }

    #[test]
    fn psd_try_from_array_round_trips() {
        let a = array![3.0_f64, 4.0];
        let psd = PsdWeightsView::try_from_array(&a).unwrap();
        assert_eq!(psd.len(), 2);
        assert_eq!(psd.view()[0], 3.0);
    }

    #[test]
    fn psd_as_signed_preserves_values() {
        let a = array![7.0_f64, 8.0];
        let psd = PsdWeightsView::try_from_array(&a).unwrap();
        let signed = psd.as_signed();
        assert_eq!(signed.len(), 2);
        assert_eq!(signed.view()[1], 8.0);
    }

    // ── SignedWeightsArc ──────────────────────────────────────────────────────

    #[test]
    fn signed_weights_arc_from_array_view_signed_len() {
        let w = SignedWeightsArc::from_array(array![1.0, 2.0, 3.0]);
        assert_eq!(w.view_signed().len(), 3);
    }

    #[test]
    fn signed_weights_arc_deref_gives_array() {
        let w = SignedWeightsArc::from_array(array![10.0_f64, 20.0]);
        assert_eq!((*w)[0], 10.0);
        assert_eq!((*w)[1], 20.0);
    }

    #[test]
    fn finite_signed_view_preserves_negative_and_signed_zero() {
        let a = array![-3.5_f64, -0.0, 2.0];
        let weights = FiniteSignedWeightsView::try_from_array(&a).unwrap();
        assert_eq!(weights.view()[0].to_bits(), (-3.5_f64).to_bits());
        assert_eq!(weights.view()[1].to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn finite_signed_view_reports_smallest_nonfinite_row() {
        let a = array![1.0_f64, f64::NAN, f64::INFINITY];
        let err = FiniteSignedWeightsView::try_from_array(&a)
            .err()
            .expect("non-finite weights must fail certification");
        assert!(err.contains("row 1"), "unexpected diagnostic: {err}");
    }
}
