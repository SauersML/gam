use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub use gam_linalg::{RidgeDeterminantMode, RidgePolicy};

pub use gam_spec::*;

/// Storage form of the ridge penalty matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RidgeMatrixForm {
    /// Ridge matrix is `delta * I`.
    ScaledIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidStabilization {
    reason: String,
}

impl InvalidStabilization {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl std::fmt::Display for InvalidStabilization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid stabilization metadata: {}", self.reason)
    }
}

impl std::error::Error for InvalidStabilization {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct RidgePassportWire {
    delta: f64,
    matrix_form: RidgeMatrixForm,
    policy: RidgePolicy,
}

/// Validated ridge metadata stamped into a fitted PIRLS result.
///
/// Construction and deserialization both reject non-finite or negative
/// magnitudes; fields are private so invalid state cannot be assembled with a
/// literal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RidgePassportWire", into = "RidgePassportWire")]
pub struct RidgePassport {
    delta: f64,
    matrix_form: RidgeMatrixForm,
    policy: RidgePolicy,
}

impl RidgePassport {
    pub fn scaled_identity(delta: f64, policy: RidgePolicy) -> Result<Self, InvalidStabilization> {
        if !(delta.is_finite() && delta >= 0.0) {
            return Err(InvalidStabilization::new(format!(
                "ridge delta must be finite and non-negative, got {delta:?}"
            )));
        }
        Ok(Self {
            delta: if delta == 0.0 { 0.0 } else { delta },
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            policy,
        })
    }

    /// Exact zero-ridge passport; this fixed sentinel has no unchecked input.
    pub const fn zero(policy: RidgePolicy) -> Self {
        Self {
            delta: 0.0,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            policy,
        }
    }

    #[inline]
    pub const fn delta(self) -> f64 {
        self.delta
    }

    #[inline]
    pub const fn matrix_form(self) -> RidgeMatrixForm {
        self.matrix_form
    }

    #[inline]
    pub const fn policy(self) -> RidgePolicy {
        self.policy
    }

    #[inline]
    pub const fn penalty_logdet_ridge(self) -> f64 {
        if self.policy.accounts_for_objective() {
            self.delta
        } else {
            0.0
        }
    }

    #[inline]
    pub const fn laplace_hessian_ridge(self) -> f64 {
        if self.policy.accounts_for_objective() {
            self.delta
        } else {
            0.0
        }
    }
}

impl TryFrom<RidgePassportWire> for RidgePassport {
    type Error = InvalidStabilization;

    fn try_from(wire: RidgePassportWire) -> Result<Self, Self::Error> {
        let mut passport = Self::scaled_identity(wire.delta, wire.policy)?;
        passport.matrix_form = wire.matrix_form;
        Ok(passport)
    }
}

impl From<RidgePassport> for RidgePassportWire {
    fn from(passport: RidgePassport) -> Self {
        Self {
            delta: passport.delta,
            matrix_form: passport.matrix_form,
            policy: passport.policy,
        }
    }
}

// ============================================================================
// StabilizationLedger: canonical accounting for every fixed/heuristic ridge
// added anywhere in the solver, linear-algebra, or family code paths.
//
// Four semantically distinct ridge uses must NEVER be conflated:
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
//   4. ApproximationOnly      — changes a named downstream approximation
//                               (for example sigma-point cubature covariance)
//                               but not the fitted model or its objective.
//
// `RidgePassport` above already encodes the inclusion-flag matrix for the
// PIRLS Laplace ridge specifically; this ledger is the broader sibling for
// every declared solver, approximation, and model ridge, so a downstream consumer can ask
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
#[serde(try_from = "InertiaWire", into = "InertiaWire")]
pub struct Inertia {
    positive: usize,
    zero: usize,
    negative: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct InertiaWire {
    positive: usize,
    zero: usize,
    negative: usize,
}

impl Inertia {
    pub fn new(
        positive: usize,
        zero: usize,
        negative: usize,
    ) -> Result<Self, InvalidStabilization> {
        let total = positive
            .checked_add(zero)
            .and_then(|value| value.checked_add(negative))
            .ok_or_else(|| InvalidStabilization::new("inertia count sum overflows usize"))?;
        if total == 0 {
            return Err(InvalidStabilization::new(
                "inertia must describe a non-empty matrix",
            ));
        }
        Ok(Self {
            positive,
            zero,
            negative,
        })
    }

    pub const fn positive(self) -> usize {
        self.positive
    }

    pub const fn zero(self) -> usize {
        self.zero
    }

    pub const fn negative(self) -> usize {
        self.negative
    }

    pub fn total(self) -> usize {
        self.positive + self.zero + self.negative
    }
}

impl TryFrom<InertiaWire> for Inertia {
    type Error = InvalidStabilization;

    fn try_from(wire: InertiaWire) -> Result<Self, Self::Error> {
        Self::new(wire.positive, wire.zero, wire.negative)
    }
}

impl From<Inertia> for InertiaWire {
    fn from(inertia: Inertia) -> Self {
        Self {
            positive: inertia.positive,
            zero: inertia.zero,
            negative: inertia.negative,
        }
    }
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

/// Four semantically distinct flavours a ridge δ can have.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StabilizationKind {
    None,
    /// LM/TR damping. NEVER enters the objective, gradient, logdet, Hessian,
    /// or any saved model artifact. Lives only inside the trust-region step.
    SolverDampingOnly,
    /// Added strictly so a linear solve succeeds. The objective/Hessian the
    /// caller sees is unchanged; the perturbation is a property of the
    /// solver, not the model. Its optional backward-error bound lives on the
    /// enclosing ledger.
    NumericalPerturbation,
    /// An explicit part of a downstream approximation, not of the fitted
    /// model. Unlike `NumericalPerturbation`, consumers must not report the
    /// result as if the unperturbed estimand had been evaluated.
    ApproximationOnly,
    /// Part of the model. Enters quadratic, log normalizer, Hessian,
    /// serialization, and user-visible summaries.
    ExplicitPrior,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct StabilizationLedgerWire {
    kind: StabilizationKind,
    delta: f64,
    matrix_form: RidgeMatrixForm,
    chosen_by: StabilizationRule,
    backward_error_bound: Option<f64>,
    inertia_before: Option<Inertia>,
    inertia_after: Option<Inertia>,
}

/// Canonical validated record of one stabilization applied at one site.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StabilizationLedgerWire", into = "StabilizationLedgerWire")]
pub struct StabilizationLedger {
    kind: StabilizationKind,
    delta: f64,
    matrix_form: RidgeMatrixForm,
    chosen_by: StabilizationRule,
    backward_error_bound: Option<f64>,
    inertia_before: Option<Inertia>,
    inertia_after: Option<Inertia>,
}

impl StabilizationLedger {
    /// "No stabilization applied at this site" sentinel.
    pub const fn none() -> Self {
        Self {
            kind: StabilizationKind::None,
            delta: 0.0,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by: StabilizationRule::FixedConstant,
            backward_error_bound: None,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// LM/TR damping. δ is invisible to the objective, gradient, and any
    /// saved artifact. Asserting this invariant at every read site is the
    /// whole reason the ledger exists.
    pub fn solver_damping(
        delta: f64,
        chosen_by: StabilizationRule,
    ) -> Result<Self, InvalidStabilization> {
        Self::try_new(StabilizationKind::SolverDampingOnly, delta, chosen_by, None)
    }

    /// Solver-only perturbation that leaves the objective unchanged. The
    /// caller may attach a backward-error bound when one is available
    /// (e.g. from iterative refinement / Wilkinson-style analysis).
    pub fn numerical_perturbation(
        delta: f64,
        chosen_by: StabilizationRule,
        backward_error_bound: Option<f64>,
    ) -> Result<Self, InvalidStabilization> {
        Self::try_new(
            StabilizationKind::NumericalPerturbation,
            delta,
            chosen_by,
            backward_error_bound,
        )
    }

    /// Ridge declared as part of a downstream approximation (for example the
    /// regularized rho covariance used to place sigma points).
    pub fn approximation_only(
        delta: f64,
        chosen_by: StabilizationRule,
    ) -> Result<Self, InvalidStabilization> {
        Self::try_new(StabilizationKind::ApproximationOnly, delta, chosen_by, None)
    }

    /// Model-level explicit prior. δ enters every accounting pass: the
    /// quadratic penalty, the Laplace Hessian, the penalty log-determinant,
    /// and serialization.
    pub fn explicit_prior(
        delta: f64,
        matrix_form: RidgeMatrixForm,
    ) -> Result<Self, InvalidStabilization> {
        let mut ledger = Self::try_new(
            StabilizationKind::ExplicitPrior,
            delta,
            StabilizationRule::UserSpecified,
            None,
        )?;
        ledger.matrix_form = matrix_form;
        Ok(ledger)
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
        let kind = if passport.policy.accounts_for_objective() {
            StabilizationKind::ExplicitPrior
        } else {
            StabilizationKind::NumericalPerturbation
        };
        Self {
            kind,
            delta: passport.delta,
            matrix_form: passport.matrix_form,
            chosen_by: StabilizationRule::FixedConstant,
            backward_error_bound: None,
            inertia_before: None,
            inertia_after: None,
        }
    }

    fn try_new(
        kind: StabilizationKind,
        delta: f64,
        chosen_by: StabilizationRule,
        backward_error_bound: Option<f64>,
    ) -> Result<Self, InvalidStabilization> {
        if matches!(kind, StabilizationKind::None) {
            return Err(InvalidStabilization::new(
                "None stabilization must be constructed with StabilizationLedger::none",
            ));
        }
        if !(delta.is_finite() && delta >= 0.0) {
            return Err(InvalidStabilization::new(format!(
                "stabilization delta must be finite and non-negative, got {delta:?}"
            )));
        }
        Self::validate_rule(chosen_by)?;
        if let Some(bound) = backward_error_bound
            && !(bound.is_finite() && bound >= 0.0)
        {
            return Err(InvalidStabilization::new(format!(
                "backward-error bound must be finite and non-negative, got {bound:?}"
            )));
        }
        if !matches!(kind, StabilizationKind::NumericalPerturbation)
            && backward_error_bound.is_some()
        {
            return Err(InvalidStabilization::new(
                "only a numerical perturbation may carry a backward-error bound",
            ));
        }
        Ok(Self {
            kind,
            delta: if delta == 0.0 { 0.0 } else { delta },
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by,
            backward_error_bound,
            inertia_before: None,
            inertia_after: None,
        })
    }

    fn validate_rule(rule: StabilizationRule) -> Result<(), InvalidStabilization> {
        match rule {
            StabilizationRule::InertiaTarget { spd_floor }
                if !(spd_floor.is_finite() && spd_floor > 0.0) =>
            {
                Err(InvalidStabilization::new(format!(
                    "inertia-target SPD floor must be finite and strictly positive, got {spd_floor:?}"
                )))
            }
            StabilizationRule::BackoffEscalation { attempts } if attempts == 0 => Err(
                InvalidStabilization::new("backoff escalation must record at least one attempt"),
            ),
            _ => Ok(()),
        }
    }

    pub fn with_inertia(
        mut self,
        before: Option<Inertia>,
        after: Option<Inertia>,
    ) -> Result<Self, InvalidStabilization> {
        if before.is_some() != after.is_some() {
            return Err(InvalidStabilization::new(
                "inertia diagnostics must record both the pre- and post-stabilization matrix",
            ));
        }
        if let (Some(before), Some(after)) = (before, after)
            && before.total() != after.total()
        {
            return Err(InvalidStabilization::new(format!(
                "inertia dimensions disagree: before={}, after={}",
                before.total(),
                after.total()
            )));
        }
        self.inertia_before = before;
        self.inertia_after = after;
        Ok(self)
    }

    pub const fn kind(self) -> StabilizationKind {
        self.kind
    }

    pub const fn delta(self) -> f64 {
        self.delta
    }

    pub const fn matrix_form(self) -> RidgeMatrixForm {
        self.matrix_form
    }

    pub const fn chosen_by(self) -> StabilizationRule {
        self.chosen_by
    }

    pub const fn backward_error_bound(self) -> Option<f64> {
        self.backward_error_bound
    }

    pub const fn inertia_before(self) -> Option<Inertia> {
        self.inertia_before
    }

    pub const fn inertia_after(self) -> Option<Inertia> {
        self.inertia_after
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
            | StabilizationKind::NumericalPerturbation
            | StabilizationKind::ApproximationOnly => 0.0,
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
            | StabilizationKind::NumericalPerturbation
            | StabilizationKind::ApproximationOnly => 0.0,
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
            | StabilizationKind::NumericalPerturbation
            | StabilizationKind::ApproximationOnly => 0.0,
        }
    }
}

impl TryFrom<StabilizationLedgerWire> for StabilizationLedger {
    type Error = InvalidStabilization;

    fn try_from(wire: StabilizationLedgerWire) -> Result<Self, Self::Error> {
        if matches!(wire.kind, StabilizationKind::None) {
            if wire.delta != 0.0
                || wire.chosen_by != StabilizationRule::FixedConstant
                || wire.backward_error_bound.is_some()
                || wire.inertia_before.is_some()
                || wire.inertia_after.is_some()
            {
                return Err(InvalidStabilization::new(
                    "None stabilization must have zero delta and no diagnostic payload",
                ));
            }
            return Ok(Self::none());
        }
        let mut ledger = Self::try_new(
            wire.kind,
            wire.delta,
            wire.chosen_by,
            wire.backward_error_bound,
        )?;
        if matches!(wire.kind, StabilizationKind::ExplicitPrior)
            && wire.chosen_by != StabilizationRule::UserSpecified
        {
            return Err(InvalidStabilization::new(
                "an explicit prior must be recorded as user specified",
            ));
        }
        ledger.matrix_form = wire.matrix_form;
        ledger.with_inertia(wire.inertia_before, wire.inertia_after)
    }
}

impl From<StabilizationLedger> for StabilizationLedgerWire {
    fn from(ledger: StabilizationLedger) -> Self {
        Self {
            kind: ledger.kind,
            delta: ledger.delta,
            matrix_form: ledger.matrix_form,
            chosen_by: ledger.chosen_by,
            backward_error_bound: ledger.backward_error_bound,
            inertia_before: ledger.inertia_before,
            inertia_after: ledger.inertia_after,
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
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl AsRef<Array1<f64>> for $name {
            #[inline]
            fn as_ref(&self) -> &Array1<f64> {
                &self.0
            }
        }

        impl From<Array1<f64>> for $name {
            #[inline]
            fn from(values: Array1<f64>) -> Self {
                Self(values)
            }
        }

        impl From<$name> for Array1<f64> {
            #[inline]
            fn from(values: $name) -> Self {
                values.0
            }
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
    pub fn new(values: ArrayView1<'a, f64>) -> Result<Self, crate::IndexedLogStrengthDomainError> {
        crate::validate_log_strengths(values.iter().copied())?;
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
    use super::{
        Inertia, RidgeMatrixForm, RidgePassport, RidgePolicy, StabilizationKind,
        StabilizationLedger, StabilizationRule,
    };
    use serde_json::json;

    #[test]
    fn solver_only_ridge_policy_stays_off_objective_accounting() {
        let passport = RidgePassport::scaled_identity(1.0e-4, RidgePolicy::solver_only())
            .expect("finite non-negative ridge");

        assert!(
            !passport.policy().accounts_for_objective(),
            "solver-only ridge must not add a quadratic prior"
        );
        assert_eq!(
            passport.penalty_logdet_ridge(),
            0.0,
            "solver-only ridge must not shift the penalty logdet"
        );
        assert_eq!(
            passport.laplace_hessian_ridge(),
            0.0,
            "solver-only ridge must not shift the Laplace Hessian"
        );

        let ledger = StabilizationLedger::from_passport(passport);
        assert!(
            matches!(ledger.kind(), StabilizationKind::NumericalPerturbation),
            "solver-only ridge is a numerical perturbation, not an explicit prior"
        );
        assert_eq!(ledger.backward_error_bound(), None);
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

    #[test]
    fn approximation_ridge_is_passported_without_becoming_a_model_prior() {
        let ledger =
            StabilizationLedger::approximation_only(1.0e-8, StabilizationRule::FixedConstant)
                .expect("valid approximation ridge");
        assert!(matches!(
            ledger.kind(),
            StabilizationKind::ApproximationOnly
        ));
        assert_eq!(ledger.delta(), 1.0e-8);
        assert_eq!(ledger.quadratic_delta(), 0.0);
        assert_eq!(ledger.laplace_hessian_delta(), 0.0);
        assert_eq!(ledger.penalty_logdet_delta(), 0.0);
    }

    #[test]
    fn stabilization_magnitudes_reject_negative_and_non_finite_values() {
        for invalid in [-1.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            assert!(RidgePassport::scaled_identity(invalid, RidgePolicy::solver_only()).is_err());
            assert!(
                StabilizationLedger::solver_damping(invalid, StabilizationRule::FixedConstant)
                    .is_err()
            );
        }
    }

    #[test]
    fn serde_cannot_bypass_passport_validation() {
        let negative = json!({
            "delta": -1.0,
            "matrix_form": "ScaledIdentity",
            "policy": "SolverOnly"
        });
        assert!(serde_json::from_value::<RidgePassport>(negative).is_err());

        let passport = RidgePassport::scaled_identity(
            2.5e-7,
            RidgePolicy::positive_part_approximate_objective(),
        )
        .expect("valid ridge");
        let roundtrip: RidgePassport =
            serde_json::from_value(serde_json::to_value(passport).expect("serialize passport"))
                .expect("deserialize validated passport");
        assert_eq!(roundtrip, passport);
    }

    #[test]
    fn serde_cannot_bypass_ledger_semantics() {
        let invalid_none = json!({
            "kind": "None",
            "delta": 1.0,
            "matrix_form": "ScaledIdentity",
            "chosen_by": "FixedConstant",
            "backward_error_bound": null,
            "inertia_before": null,
            "inertia_after": null
        });
        assert!(serde_json::from_value::<StabilizationLedger>(invalid_none).is_err());

        let invalid_prior_rule = json!({
            "kind": "ExplicitPrior",
            "delta": 1.0,
            "matrix_form": "ScaledIdentity",
            "chosen_by": "Heuristic",
            "backward_error_bound": null,
            "inertia_before": null,
            "inertia_after": null
        });
        assert!(serde_json::from_value::<StabilizationLedger>(invalid_prior_rule).is_err());

        let bound_on_wrong_kind = json!({
            "kind": "ApproximationOnly",
            "delta": 1.0,
            "matrix_form": "ScaledIdentity",
            "chosen_by": "FixedConstant",
            "backward_error_bound": 1.0e-10,
            "inertia_before": null,
            "inertia_after": null
        });
        assert!(serde_json::from_value::<StabilizationLedger>(bound_on_wrong_kind).is_err());
    }

    #[test]
    fn rule_and_inertia_metadata_are_validated() {
        assert!(
            StabilizationLedger::solver_damping(
                1.0,
                StabilizationRule::InertiaTarget { spd_floor: 0.0 }
            )
            .is_err()
        );
        assert!(
            StabilizationLedger::solver_damping(
                1.0,
                StabilizationRule::BackoffEscalation { attempts: 0 }
            )
            .is_err()
        );
        assert!(Inertia::new(0, 0, 0).is_err());
        assert!(Inertia::new(usize::MAX, 1, 0).is_err());

        let before = Inertia::new(2, 1, 0).expect("valid inertia");
        let wrong_dimension = Inertia::new(3, 1, 0).expect("valid inertia");
        let ledger = StabilizationLedger::numerical_perturbation(
            1.0e-8,
            StabilizationRule::BackoffEscalation { attempts: 2 },
            Some(1.0e-12),
        )
        .expect("valid perturbation");
        assert!(ledger.with_inertia(Some(before), None).is_err());
        assert!(
            ledger
                .with_inertia(Some(before), Some(wrong_dimension))
                .is_err()
        );
    }

    #[test]
    fn objective_accounting_is_structural() {
        let explicit = StabilizationLedger::explicit_prior(0.25, RidgeMatrixForm::ScaledIdentity)
            .expect("valid explicit prior");
        assert_eq!(explicit.quadratic_delta(), 0.25);
        assert_eq!(explicit.laplace_hessian_delta(), 0.25);
        assert_eq!(explicit.penalty_logdet_delta(), 0.25);

        for policy in [
            RidgePolicy::exact_full_objective(),
            RidgePolicy::positive_part_approximate_objective(),
        ] {
            let passport = RidgePassport::scaled_identity(0.25, policy).expect("valid ridge");
            assert_eq!(passport.penalty_logdet_ridge(), 0.25);
            assert_eq!(passport.laplace_hessian_ridge(), 0.25);
        }
    }
}
