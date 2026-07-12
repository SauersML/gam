use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub use gam_linalg::{RidgeDeterminantMode, RidgePolicy};

/// Lower floor on positive working weights shared by likelihood families and
/// PIRLS row assembly so weighted normal equations stay numerically well posed.
pub const MIN_WEIGHT: f64 = 1e-12;

pub use gam_spec::*;

/// Storage form of the ridge penalty matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RidgeMatrixForm {
    /// Ridge matrix is `delta * I`.
    ScaledIdentity,
}

/// Concrete ridge metadata stamped into a fitted PIRLS result.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RidgePassport {
    /// Stabilization magnitude for matrix form `delta * I`.
    pub delta: f64,
    pub matrix_form: RidgeMatrixForm,
    pub policy: RidgePolicy,
}

impl RidgePassport {
    pub const fn scaled_identity(delta: f64, policy: RidgePolicy) -> Self {
        Self {
            delta,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            policy,
        }
    }

    #[inline]
    pub const fn penalty_logdet_ridge(self) -> f64 {
        if self.policy.include_penalty_logdet {
            self.delta
        } else {
            0.0
        }
    }

    #[inline]
    pub const fn laplacehessianridge(self) -> f64 {
        if self.policy.include_laplacehessian {
            self.delta
        } else {
            0.0
        }
    }
}

// ============================================================================
// StabilizationLedger: canonical accounting for every fixed/heuristic ridge
// added anywhere in the solver, linear-algebra, or family code paths.
//
// Three semantically distinct ridge uses must NEVER be conflated:
//   1. SolverDampingOnly      — Levenberg/trust-region damping; never enters
//                               objective, gradient, logdet, Hessian, or any
//                               saved/serialized model artifact.
//   2. NumericalPerturbation  — added strictly so a linear solve is well-
//                               posed (e.g. Cholesky of a near-singular
//                               matrix). Carries an optional backward-error
//                               bound. Does NOT change the objective.
//   3. ExplicitPrior          — model-level `delta * I` (or block-diagonal)
//                               prior. Appears in quadratic, log normalizer,
//                               Laplace Hessian, serialization, diagnostics.
//
// `RidgePassport` above already encodes the inclusion-flag matrix for the
// PIRLS Laplace ridge specifically; this ledger is the broader sibling that
// every other call site (RidgePlanner, matrix_inverse_with_regularization,
// LAML rho-Hessian inversion, survival stabilization, custom-family
// `ridge_floor`) routes through, so a downstream consumer can ask
// `ledger.quadratic_delta()` rather than rediscovering the policy. The three
// inclusion bits were lifted into the `StabilizationKind` discriminant so the
// (kind, inclusion-flags) invariant is enforced statically — heterogeneous
// combinations like "ExplicitPrior with quadratic excluded" no longer typecheck.
// ============================================================================

/// Inertia of a symmetric matrix (count of positive / zero / negative
/// eigenvalues). Used by `bump_with_matrix` and other indefinite-aware
/// stabilization rules to drive δ from spectral evidence rather than a
/// condition-number heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inertia {
    pub positive: usize,
    pub zero: usize,
    pub negative: usize,
}

/// Why a stabilization δ was chosen at this site.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StabilizationRule {
    /// δ is a hard-coded constant in the source.
    FixedConstant,
    /// δ chosen so the SPD floor τ is met: δ = max(0, τ - λ_min(H)).
    InertiaTarget { spd_floor: f64 },
    /// δ chosen via a condition-number / sqrt-ratio heuristic.
    Heuristic,
    /// User- or family-specified prior precision.
    UserSpecified,
    /// δ derived from a back-off escalation after a factorization failure.
    BackoffEscalation { attempts: usize },
}

/// Three semantically distinct flavours a ridge δ can have.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StabilizationKind {
    None,
    /// LM/TR damping. NEVER enters the objective, gradient, logdet, Hessian,
    /// or any saved model artifact. Lives only inside the trust-region step.
    SolverDampingOnly,
    /// Added strictly so a linear solve succeeds. The objective/Hessian the
    /// caller sees is unchanged; the perturbation is a property of the
    /// solver, not the model. `backward_error_bound` is the max change to
    /// the solution norm imputable to the perturbation, when known.
    NumericalPerturbation {
        backward_error_bound: Option<f64>,
    },
    /// Part of the model. Enters quadratic, log normalizer, Hessian,
    /// serialization, and user-visible summaries.
    ExplicitPrior,
}

/// Canonical record of a single stabilization δ applied at a single site.
///
/// Construct via the helper constructors (`solver_damping`,
/// `numerical_perturbation`, `explicit_prior`) so the `included_in_*`
/// invariants are guaranteed to match `kind`. Direct field construction is
/// public for serialization round-trips only.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StabilizationLedger {
    pub kind: StabilizationKind,
    pub delta: f64,
    pub matrix_form: RidgeMatrixForm,
    pub chosen_by: StabilizationRule,
    pub inertia_before: Option<Inertia>,
    pub inertia_after: Option<Inertia>,
}

impl StabilizationLedger {
    /// "No stabilization applied at this site" sentinel.
    pub const fn none() -> Self {
        Self {
            kind: StabilizationKind::None,
            delta: 0.0,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by: StabilizationRule::FixedConstant,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// LM/TR damping. δ is invisible to the objective, gradient, and any
    /// saved artifact. Asserting this invariant at every read site is the
    /// whole reason the ledger exists.
    pub const fn solver_damping(delta: f64, chosen_by: StabilizationRule) -> Self {
        Self {
            kind: StabilizationKind::SolverDampingOnly,
            delta,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// Solver-only perturbation that leaves the objective unchanged. The
    /// caller may attach a backward-error bound when one is available
    /// (e.g. from iterative refinement / Wilkinson-style analysis).
    pub const fn numerical_perturbation(
        delta: f64,
        chosen_by: StabilizationRule,
        backward_error_bound: Option<f64>,
    ) -> Self {
        Self {
            kind: StabilizationKind::NumericalPerturbation {
                backward_error_bound,
            },
            delta,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// Model-level explicit prior. δ enters every accounting pass: the
    /// quadratic penalty, the Laplace Hessian, the penalty log-determinant,
    /// and serialization.
    pub const fn explicit_prior(delta: f64, matrix_form: RidgeMatrixForm) -> Self {
        Self {
            kind: StabilizationKind::ExplicitPrior,
            delta,
            matrix_form,
            chosen_by: StabilizationRule::UserSpecified,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// Bridge from the existing `RidgePassport` so PIRLS-side code (which
    /// already passes a `RidgePassport` through every call) can hand a
    /// ledger to anything that wants the new uniform view.
    ///
    /// `RidgePolicy` is homogeneous-by-construction: every constructor sets
    /// the three inclusion flags identically. A passport whose policy
    /// excludes every accounting term is morally a numerical perturbation
    /// (the ridge is there to make the solve work but the objective ignores
    /// it); a passport whose policy includes every accounting term is an
    /// explicit prior. Heterogeneous flag combinations cannot be produced
    /// by the public `RidgePolicy` API and have no inhabitants downstream.
    pub const fn from_passport(passport: RidgePassport) -> Self {
        let any_included = passport.policy.include_quadratic_penalty
            || passport.policy.include_laplacehessian
            || passport.policy.include_penalty_logdet;
        let kind = if any_included {
            StabilizationKind::ExplicitPrior
        } else {
            StabilizationKind::NumericalPerturbation {
                backward_error_bound: None,
            }
        };
        Self {
            kind,
            delta: passport.delta,
            matrix_form: passport.matrix_form,
            chosen_by: StabilizationRule::FixedConstant,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// δ value to fold into the quadratic penalty term, or 0.0 if this
    /// ledger entry is not part of the model. Derived from `kind`: only
    /// [`StabilizationKind::ExplicitPrior`] contributes.
    #[inline]
    pub const fn quadratic_delta(&self) -> f64 {
        match self.kind {
            StabilizationKind::ExplicitPrior => self.delta,
            StabilizationKind::None
            | StabilizationKind::SolverDampingOnly
            | StabilizationKind::NumericalPerturbation { .. } => 0.0,
        }
    }

    /// δ value to add to the Laplace Hessian, or 0.0 if not included.
    /// Derived from `kind`: only [`StabilizationKind::ExplicitPrior`]
    /// contributes.
    #[inline]
    pub const fn laplace_hessian_delta(&self) -> f64 {
        match self.kind {
            StabilizationKind::ExplicitPrior => self.delta,
            StabilizationKind::None
            | StabilizationKind::SolverDampingOnly
            | StabilizationKind::NumericalPerturbation { .. } => 0.0,
        }
    }

    /// δ value to add inside log|S + δ I|, or 0.0 if not included.
    /// Derived from `kind`: only [`StabilizationKind::ExplicitPrior`]
    /// contributes.
    #[inline]
    pub const fn penalty_logdet_delta(&self) -> f64 {
        match self.kind {
            StabilizationKind::ExplicitPrior => self.delta,
            StabilizationKind::None
            | StabilizationKind::SolverDampingOnly
            | StabilizationKind::NumericalPerturbation { .. } => 0.0,
        }
    }
}
/// Generate a `#[repr(transparent)]` `Array1<f64>` newtype with the
/// `new`/`Deref`/`DerefMut`/`AsRef`/`From` boilerplate used by unconstrained
/// numeric vectors in this module.
macro_rules! array1_f64_newtype {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name(pub Array1<f64>);

        impl $name {
            #[inline]
            pub fn new(values: Array1<f64>) -> Self {
                Self(values)
            }

            #[inline]
            pub fn zeros(len: usize) -> Self {
                Self(Array1::zeros(len))
            }
        }

        impl Deref for $name {
            type Target = Array1<f64>;
            #[inline]
            fn deref(&self) -> &Self::Target { &self.0 }
        }

        impl DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
        }

        impl AsRef<Array1<f64>> for $name {
            #[inline]
            fn as_ref(&self) -> &Array1<f64> { &self.0 }
        }

        impl From<Array1<f64>> for $name {
            #[inline]
            fn from(values: Array1<f64>) -> Self { Self(values) }
        }

        impl From<$name> for Array1<f64> {
            #[inline]
            fn from(values: $name) -> Self { values.0 }
        }
    };
}

array1_f64_newtype!(Coefficients);
array1_f64_newtype!(LinearPredictor);

/// Index into `TermCollectionSpec::smooth_terms` (and the parallel
/// `TermCollectionDesign::smooth.terms` slice produced from it).
///
/// This is **not** a penalty/ρ index, **not** a column index, and **not** a
/// coefficient-offset index. Keeping it behind a `#[repr(transparent)]`
/// newtype makes those confusables a compile error: a `SmoothTermIdx` cannot
/// be silently used to index `rho`, `beta`, or a design column.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SmoothTermIdx(usize);

impl SmoothTermIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// Sentinel used by transient builders that must allocate a coord config
    /// before the smooth term it references has been positioned in the spec.
    /// Every code path that constructs a sentinel must overwrite it before
    /// the value escapes the builder.
    #[inline]
    pub const fn placeholder() -> Self {
        Self(usize::MAX)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }

    #[inline]
    pub const fn is_placeholder(self) -> bool {
        self.0 == usize::MAX
    }
}

impl std::fmt::Display for SmoothTermIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into the canonical penalty list `&[CanonicalPenalty]` — equivalently,
/// the position of a smoothing parameter in the ρ / λ vector.
///
/// Penalty/ρ indices are not interchangeable with `SmoothTermIdx` (a smooth
/// term can carry multiple canonical penalties — e.g. tensor-product double
/// penalties — and structural penalties don't correspond to any smooth term).
/// Keeping them as separate newtypes makes the historical bug pattern
/// "indexed `rho` with a smooth-term ordinal" impossible to express.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PenaltyIdx(usize);

impl PenaltyIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for PenaltyIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into a single smooth term's set of basis functions — i.e. the `k`
/// in "the k-th basis function `B_k(x)` of this term".
///
/// Distinct from:
///   * [`SmoothTermIdx`] — selects *which* smooth term in the spec.
///   * [`PenaltyIdx`]    — selects *which* ρ/λ entry / canonical penalty.
///   * A design-matrix column index — which lives in the *combined* layout
///     after intercept/parametric blocks and per-term offsets are applied;
///     a `BasisIdx` is term-local, a column index is model-global.
///
/// Keeping this as its own `#[repr(transparent)]` newtype makes the
/// historically-easy confusion "indexed a global column slice with a
/// term-local basis ordinal" (or vice versa) a compile error.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BasisIdx(usize);

impl BasisIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for BasisIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into the user-facing design matrix `data: Array2<f64>` — i.e. the
/// position of a covariate column in the raw input frame, *before* any
/// per-family basis expansion or intercept/parametric layout is applied.
///
/// Distinct from:
///   * [`BasisIdx`] — term-local basis-function ordinal `k` of `B_k(x)`.
///   * [`SmoothTermIdx`] — position in `TermCollectionSpec::smooth_terms`.
///   * A coefficient-vector offset `β[i]` — spans the combined design after
///     expansion, which is much wider than the user-facing data matrix.
///
/// Keeping this as its own `#[repr(transparent)]` newtype rules out the easy
/// confusion of indexing the raw data frame with an expanded-column offset.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ColIdx(usize);

impl ColIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for ColIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index of an observation (row) in the user-facing data frame / design
/// matrix — i.e. the `i` in "the i-th observation".
///
/// Distinct from every column-type index in this module ([`ColIdx`],
/// [`BasisIdx`], [`SmoothTermIdx`], [`PenaltyIdx`]) and from coefficient
/// offsets. Keeping rows behind their own `#[repr(transparent)]` newtype
/// makes the classic `data[[col, row]]` transposition a compile error.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RowIdx(usize);

impl RowIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for RowIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct LogSmoothingParamsView<'a>(ArrayView1<'a, f64>);

impl<'a> LogSmoothingParamsView<'a> {
    /// Borrow a smoothing vector only after every coordinate satisfies the
    /// exact shared logarithmic-strength contract.
    pub fn new(
        values: ArrayView1<'a, f64>,
    ) -> Result<Self, crate::IndexedLogStrengthDomainError> {
        for (coordinate, &value) in values.iter().enumerate() {
            crate::validate_log_strength(value).map_err(|_| {
                crate::IndexedLogStrengthDomainError { coordinate, value }
            })?;
        }
        Ok(Self(values))
    }

    /// Exact physical strengths for this already-validated vector.
    pub fn exact_exp(&self) -> Array1<f64> {
        // `new` established the private invariant; the borrow prevents the
        // source array from being mutated for this view's lifetime.
        self.0.mapv(f64::exp)
    }
}

impl<'a> Deref for LogSmoothingParamsView<'a> {
    type Target = ArrayView1<'a, f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod newtype_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn smooth_term_idx_new_get_roundtrip() {
        let idx = SmoothTermIdx::new(7);
        assert_eq!(idx.get(), 7);
        assert!(!idx.is_placeholder());
        assert_eq!(format!("{idx}"), "7");
    }

    #[test]
    fn smooth_term_idx_placeholder_is_detected() {
        let p = SmoothTermIdx::placeholder();
        assert!(p.is_placeholder());
        assert_eq!(p.get(), usize::MAX);
    }

    #[test]
    fn smooth_term_idx_ordering() {
        let a = SmoothTermIdx::new(1);
        let b = SmoothTermIdx::new(2);
        assert!(a < b);
        assert_eq!(a, SmoothTermIdx::new(1));
    }

    #[test]
    fn coefficients_zeros_and_deref() {
        let c = Coefficients::zeros(3);
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn coefficients_from_array1() {
        let arr = array![1.0, 2.0, 3.0];
        let c = Coefficients::from(arr.clone());
        assert_eq!(*c, arr);
    }

    #[test]
    fn log_smoothing_params_view_is_validated_and_exponentiates_exactly() {
        let arr = array![crate::LOG_STRENGTH_MIN, 0.0, crate::LOG_STRENGTH_MAX];
        let rho = LogSmoothingParamsView::new(arr.view()).expect("closed domain");
        for (actual, expected) in rho.exact_exp().iter().zip(arr.iter()) {
            assert_eq!(actual.to_bits(), expected.exp().to_bits());
        }

        let invalid = array![0.0, crate::LOG_STRENGTH_MAX + 1.0];
        let error = LogSmoothingParamsView::new(invalid.view()).unwrap_err();
        assert_eq!(error.coordinate, 1);
        assert_eq!(error.value, crate::LOG_STRENGTH_MAX + 1.0);
    }

    #[test]
    fn linear_predictor_zeros_and_deref() {
        let lp = LinearPredictor::zeros(4);
        assert_eq!(lp.len(), 4);
        assert!(lp.iter().all(|&v| v == 0.0));
    }
}

#[cfg(test)]
mod ridge_policy_tests {
    use super::{RidgePassport, RidgePolicy, StabilizationKind, StabilizationLedger};

    #[test]
    fn solver_only_ridge_policy_stays_off_objective_accounting() {
        let passport = RidgePassport::scaled_identity(1.0e-4, RidgePolicy::solver_only());

        assert!(
            !passport.policy.include_quadratic_penalty,
            "solver-only ridge must not add a quadratic prior"
        );
        assert_eq!(
            passport.penalty_logdet_ridge(),
            0.0,
            "solver-only ridge must not shift the penalty logdet"
        );
        assert_eq!(
            passport.laplacehessianridge(),
            0.0,
            "solver-only ridge must not shift the Laplace Hessian"
        );

        let ledger = StabilizationLedger::from_passport(passport);
        assert!(
            matches!(
                ledger.kind,
                StabilizationKind::NumericalPerturbation {
                    backward_error_bound: None
                }
            ),
            "solver-only ridge is a numerical perturbation, not an explicit prior"
        );
        assert_eq!(
            ledger.quadratic_delta(),
            0.0,
            "solver-only ridge must not contribute to the optimized objective"
        );
        assert_eq!(
            ledger.laplace_hessian_delta(),
            0.0,
            "solver-only ridge must not contribute to REML curvature accounting"
        );
        assert_eq!(
            ledger.penalty_logdet_delta(),
            0.0,
            "solver-only ridge must not contribute to determinant accounting"
        );
    }
}
