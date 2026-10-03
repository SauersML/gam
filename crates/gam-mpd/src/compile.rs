//! The native edit compiler (#2951): from an explanation's control settings `α` to the
//! parameter edits `ρ(α)` of the original network, with `ρ(0) = θ`, or a proof that none
//! exist.
//!
//! An explanation's internal controls (a gate law, a routing score, a shared operator) are
//! not automatically independently editable in the native network. The contract a
//! decomposition meets is `P_α(x) ≈ F_{ρ(α)}(x)` under declared interventions, so every
//! claimed control change is compiled here into native tensors, or reported as coupled or
//! infeasible with a witness.
//!
//! # Two kinds of setting
//!
//! * An **internal control state** is computed by the model per input: a gate value, an
//!   attention weight, a routing score. The compiler never sets one directly.
//! * An **external edit setting** is a fixed implementation change of the stored
//!   parameters, the same for every input. [`NativeEditPlan`] is one: a list of global
//!   factored edits of registered storage tensors.
//!
//! A control is compiled when a fixed external edit moves the internal state the way the
//! explanation's control says, over the declared control family.
//!
//! # Settings are set-type, so histories have a normal form
//!
//! Every edit atom the compiler accepts sets a control or operator to a value (a use's
//! response on declared inputs, a stored row, a per-control gain relative to the native
//! control, new chart coordinates, new query/key weights); none adds a change to whatever
//! came before. Set atoms form an intervention algebra (Geiger et al., *Causal
//! Abstraction*, JMLR 2025, Defs 15–18, Thm 21): a later set of a control annihilates an
//! earlier one (left-annihilativity), sets of distinct controls commute, and every history
//! of atoms reduces to one normal-form [`Setting`], last write per control. An additive
//! "add δ" vocabulary has no such normal form (their Remark 22). Each compiler consumes the
//! normal form and always compiles from the stored `θ`, never from an edited state, so
//! `ρ` depends on a history only through its normal form.
//!
//! `ρ` is an interventional on the native parameters (their Def 11) and is built as a
//! section of the abstraction map `ω` from native edits to explanation settings (Def 25,
//! Remark 29): `ω(ρ(α)) = α`. Each compiler certifies exactly that equation, as the
//! residual of the explanation's control read back from the native edit. Off-target damage
//! is the approximate-transformation distance (Def 41) restricted to declared off-target
//! inputs ([`linear::off_target_damage`]); declaring those inputs as set-to-native
//! requirements makes the minimum-norm edit leave them unchanged.
//!
//! # The plan is `ρ(α) − θ`
//!
//! [`NativeEditPlan`] holds `ρ(α) − θ` as global edits `ΔW = left · rightᵀ` of storage
//! tensors (`apply::FactoredEdit`). The empty plan is `ρ(0) = θ`: it executes the original tensors on their original path, never the compiler's arithmetic.
//! A compiled edit is always global on storage, so every tied use of a stored tensor moves
//! with it; the compiler never emits a use-specific edit, which would untie a tie
//! ([`ties`]).
//!
//! # Status of a compiled control
//!
//! [`ControlRealization`] is built on [`EvidenceStatus`] and keeps three cases apart:
//!
//! * **Exactly realized.** An algebraic construction over the whole control family: the
//!   residual is [`EvidenceStatus::Exact`] with [`ExactBasis::Algebraic`], carrying the
//!   achieved residual of the stored edit and its evaluation band.
//! * **Empirically validated.** A fixed native edit tested on a declared finite domain:
//!   the residual is exhaustive over that domain ([`ExactBasis::Exhaustive`]) or a
//!   [`EvidenceStatus::UniformBound`] over it, and says nothing beyond it.
//! * **Descriptive.** No independent native control was established. It names why and,
//!   when one exists, carries a [`EvidenceStatus::Counterexample`] witness.
//!
//! The constructors refuse a status stronger than its evidence.
//!
//! # Owners
//!
//! * [`linear`]: linear-site feasibility, `ΔW X = Y` iff `ker X ⊆ ker Y` (two-sided for
//!   tied and transposed uses), minimum-norm edits under a declared metric.
//! * [`controls`]: coupled controls of a factorization `γ̂ = A φ(B h)`, their coupling
//!   classes, and coordinated writer/reader edits.
//! * [`chart`]: fixed-rank charts of a low-rank joint operator, the dependent block
//!   computed, and the lift to the native factors through the `GL(r)` gauge; factors wider
//!   than the chart rank are reduced through `GL(h)` first, with a band.
//! * [`bilinear`]: query/key edits with every cross term, certified through the exact
//!   finite softmax, and set-type score requirements solved for `(Q′, K′)` together by
//!   alternating exact linear solves.
//! * [`path`]: requirements downstream of norms and MLPs, solved at one or more linear
//!   sites through the exact forward (Gauss–Newton, exact Jacobian products, LSQR) and
//!   certified by executing the edited path.
//! * [`null`]: physically null edits, `sup_{uᵀGu ≤ 1} uᵀKu` on `range(G)`, refused when
//!   `ker G ⊄ ker K`.
//! * [`ties`]: declared ties between separately stored blocks, checked on a plan.

use std::fmt;

use gam_runtime::resource::MemoryReservationError;

use super::apply::{ApplyError, FactoredEdit};
use super::bounds::BoundError;
use super::dense::DenseError;
use super::gauge::GaugeRefusal;
use super::joint_operators::JointRefusal;
use super::lift::{LiftError, TensorId, TensorRegistry};
use super::secant::SecantError;
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis};

pub mod bilinear;
pub mod chart;
pub mod controls;
pub mod linear;
pub mod null;
pub mod path;
pub mod ties;

#[cfg(test)]
mod tests;

/// Proof that a [`ControlRealization`] passed through its constructor. Only this module
/// and its children can build one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checked(());

/// Why no independent native control was established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptiveReason {
    /// A requested change is nonzero on a direction the inputs annihilate: `ker X ⊄ ker Y`.
    KernelViolation,
    /// Requirements on the two sides of one stored matrix (a tied read and write, a
    /// transposed use, a stored-row read) demand different values of one bilinear
    /// coordinate `x_lᵀ ΔW x_r`.
    IncompatibleSides,
    /// The only edit would change one side of a declared tie and not the other.
    TieBroken,
    /// The requested per-control change moves controls that the permitted sites couple.
    CoupledControls,
    /// The stored edit's residual is resolved from zero beyond its own band: the
    /// algebraic construction did not survive its arithmetic.
    ResidualResolved,
}

/// What a compiled control's native edit establishes. Built on [`EvidenceStatus`]; see the
/// module documentation.
#[derive(Clone, Debug, PartialEq)]
pub enum ControlRealization<W, D> {
    /// An algebraic construction over the whole control family.
    ExactlyRealized {
        residual: EvidenceStatus<W, D>,
        checked: Checked,
    },
    /// A fixed native edit tested on a declared finite domain.
    EmpiricallyValidated {
        residual: EvidenceStatus<W, D>,
        checked: Checked,
    },
    /// No independent native control.
    Descriptive {
        reason: DescriptiveReason,
        witness: Option<EvidenceStatus<W, D>>,
        checked: Checked,
    },
}

impl<W, D> ControlRealization<W, D> {
    /// Refused unless `residual` is exact by an algebraic identity.
    pub fn exactly_realized(residual: EvidenceStatus<W, D>) -> Result<Self, CompileError> {
        match residual {
            EvidenceStatus::Exact {
                basis: ExactBasis::Algebraic,
                ..
            } => Ok(Self::ExactlyRealized {
                residual,
                checked: Checked(()),
            }),
            _ => Err(CompileError::StatusTooStrong {
                claimed: "exactly realized",
                evidence: status_name(&residual),
            }),
        }
    }

    /// Exactly realized on a declared finite family: refused unless `residual` is exhaustive
    /// over that family and lies within its own numerical error, so the requirement holds at
    /// every member up to the evaluation band. It states nothing beyond the family, which
    /// the residual's domain names.
    pub fn exactly_realized_on_family(residual: EvidenceStatus<W, D>) -> Result<Self, CompileError> {
        match residual {
            EvidenceStatus::Exact {
                basis: ExactBasis::Exhaustive { .. },
                value,
                numerical_error,
                ..
            } if value.abs() <= numerical_error => Ok(Self::ExactlyRealized {
                residual,
                checked: Checked(()),
            }),
            _ => Err(CompileError::StatusTooStrong {
                claimed: "exactly realized on the declared family",
                evidence: status_name(&residual),
            }),
        }
    }

    /// Refused unless `residual` is exhaustive over a finite family or a uniform bound over
    /// a declared region.
    pub fn empirically_validated(residual: EvidenceStatus<W, D>) -> Result<Self, CompileError> {
        match residual {
            EvidenceStatus::Exact {
                basis: ExactBasis::Exhaustive { .. },
                ..
            }
            | EvidenceStatus::UniformBound { .. } => Ok(Self::EmpiricallyValidated {
                residual,
                checked: Checked(()),
            }),
            _ => Err(CompileError::StatusTooStrong {
                claimed: "empirically validated",
                evidence: status_name(&residual),
            }),
        }
    }

    /// Refused when a witness is given that is not a counterexample.
    pub fn descriptive(
        reason: DescriptiveReason,
        witness: Option<EvidenceStatus<W, D>>,
    ) -> Result<Self, CompileError> {
        match &witness {
            None | Some(EvidenceStatus::Counterexample { .. }) => Ok(Self::Descriptive {
                reason,
                witness,
                checked: Checked(()),
            }),
            Some(other) => Err(CompileError::StatusTooStrong {
                claimed: "descriptive witness",
                evidence: status_name(other),
            }),
        }
    }

    /// Whether a native control was established (exactly or on a declared domain).
    pub fn is_native_control(&self) -> bool {
        !matches!(self, Self::Descriptive { .. })
    }

    /// The residual status, when a native control was established.
    pub fn residual(&self) -> Option<&EvidenceStatus<W, D>> {
        match self {
            Self::ExactlyRealized { residual, .. } | Self::EmpiricallyValidated { residual, .. } => Some(residual),
            Self::Descriptive { .. } => None,
        }
    }
}

fn status_name<W, D>(status: &EvidenceStatus<W, D>) -> &'static str {
    match status {
        EvidenceStatus::Exact {
            basis: ExactBasis::Algebraic,
            ..
        } => "exact (algebraic)",
        EvidenceStatus::Exact { .. } => "exact (exhaustive)",
        EvidenceStatus::UniformBound { .. } => "uniform bound",
        EvidenceStatus::StatisticalEstimate { .. } => "statistical estimate",
        EvidenceStatus::Counterexample { .. } => "counterexample",
        EvidenceStatus::Unresolved { .. } => "unresolved",
    }
}

/// One set-type atom: control `control` is set to `value`.
#[derive(Clone, Debug, PartialEq)]
pub struct SetAtom<V> {
    pub control: String,
    pub value: V,
}

/// The normal form of a history of set atoms: the last value set per control. The empty
/// setting is the native one.
#[derive(Clone, Debug, PartialEq)]
pub struct Setting<V> {
    values: std::collections::BTreeMap<String, V>,
}

impl<V> Default for Setting<V> {
    fn default() -> Self {
        Self {
            values: std::collections::BTreeMap::new(),
        }
    }
}

impl<V> Setting<V> {
    /// The native setting: no control set.
    pub fn native() -> Self {
        Self::default()
    }

    /// The normal form of a history applied in order.
    pub fn from_history(history: impl IntoIterator<Item = SetAtom<V>>) -> Self {
        history.into_iter().fold(Self::native(), |setting, atom| setting.set(atom))
    }

    /// This setting followed by one more atom; a later set of a control annihilates the
    /// earlier one.
    pub fn set(mut self, atom: SetAtom<V>) -> Self {
        self.values.insert(atom.control, atom.value);
        self
    }

    /// This setting followed by `later`: `later`'s values win on every control it sets.
    pub fn then(mut self, later: Self) -> Self {
        self.values.extend(later.values);
        self
    }

    pub fn get(&self, control: &str) -> Option<&V> {
        self.values.get(control)
    }

    pub fn is_native(&self) -> bool {
        self.values.is_empty()
    }

    /// The set controls and their values, in control order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.values.iter()
    }
}

/// One global edit `ΔW = left · rightᵀ` of a stored tensor, in its stored orientation.
#[derive(Clone, Debug)]
pub struct CompiledParameterEdit {
    pub storage: TensorId,
    pub delta: FactoredEdit,
}

/// `ρ(α) − θ` as global edits of storage tensors. The empty plan is `ρ(0) = θ`.
#[derive(Clone, Debug, Default)]
pub struct NativeEditPlan {
    edits: Vec<CompiledParameterEdit>,
}

impl NativeEditPlan {
    /// `ρ(0) = θ`: no edit, so the original tensors run on their original path.
    pub fn native() -> Self {
        Self::default()
    }

    /// A plan of global edits, each of a registered matrix storage tensor with the edit's
    /// shape, at most one per storage. An edit named by an alias is refused: an edit acts
    /// on storage, and the alias would hide which tie it moves.
    pub fn new(registry: &TensorRegistry, edits: Vec<CompiledParameterEdit>) -> Result<Self, CompileError> {
        for (index, edit) in edits.iter().enumerate() {
            let tensor = registry
                .storage(&edit.storage)
                .ok_or_else(|| CompileError::NotStorage(edit.storage.0.clone()))?;
            let shape = (edit.delta.output_dim(), edit.delta.input_dim());
            if tensor.shape != [shape.0, shape.1] {
                return Err(CompileError::EditShape {
                    storage: edit.storage.0.clone(),
                    stored: tensor.shape.clone(),
                    edit: shape,
                });
            }
            if edits[..index].iter().any(|earlier| earlier.storage == edit.storage) {
                return Err(CompileError::DuplicateEdit(edit.storage.0.clone()));
            }
        }
        Ok(Self { edits })
    }

    pub fn edits(&self) -> &[CompiledParameterEdit] {
        &self.edits
    }

    /// Whether the plan is `ρ(0) = θ`.
    pub fn is_native(&self) -> bool {
        self.edits.is_empty()
    }
}

/// A compiled control: its native plan and what the plan establishes. `plan` is `None`
/// exactly when the realization is descriptive.
#[derive(Clone, Debug)]
pub struct CompiledControl<W, D> {
    pub control: String,
    pub plan: Option<NativeEditPlan>,
    pub realization: ControlRealization<W, D>,
}

impl<W, D> CompiledControl<W, D> {
    /// Refuses a plan beside a descriptive realization, and a missing plan beside a native
    /// one.
    pub fn new(
        control: String,
        plan: Option<NativeEditPlan>,
        realization: ControlRealization<W, D>,
    ) -> Result<Self, CompileError> {
        if plan.is_some() != realization.is_native_control() {
            return Err(CompileError::PlanStatusMismatch { control });
        }
        Ok(Self {
            control,
            plan,
            realization,
        })
    }
}

/// A refused compilation.
#[derive(Debug)]
pub enum CompileError {
    /// An operand's shape disagrees with the problem.
    Shape {
        what: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// An operand holds a NaN or an infinity.
    NonFinite { what: &'static str },
    /// A declared scale, radius or setting is outside its domain.
    InvalidDeclaration { what: &'static str, reason: String },
    /// The name is not a registered storage tensor.
    NotStorage(String),
    /// An edit's shape disagrees with the stored tensor.
    EditShape {
        storage: String,
        stored: Vec<usize>,
        edit: (usize, usize),
    },
    /// More than one edit of one storage tensor.
    DuplicateEdit(String),
    /// A use site reads a different storage tensor than the problem edits, or reads it
    /// in a way the problem cannot constrain.
    UseSite { site: String, reason: String },
    /// A status stronger than its evidence.
    StatusTooStrong {
        claimed: &'static str,
        evidence: &'static str,
    },
    /// A plan beside a descriptive status, or none beside a native one.
    PlanStatusMismatch { control: String },
    Lift(LiftError),
    Apply(ApplyError),
    Dense(DenseError),
    Evidence(EvidenceStatusError),
    Secant(SecantError),
    Bound(BoundError),
    Joint(JointRefusal),
    Gauge(GaugeRefusal),
    Memory(MemoryReservationError),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { what, expected, found } => {
                write!(f, "native edit compiler: {what} has shape {found:?}, expected {expected:?}")
            }
            Self::NonFinite { what } => write!(f, "native edit compiler: {what} holds a non-finite entry"),
            Self::InvalidDeclaration { what, reason } => {
                write!(f, "native edit compiler: declared {what} refused: {reason}")
            }
            Self::NotStorage(name) => write!(f, "native edit compiler: {name:?} is not a registered storage tensor"),
            Self::EditShape { storage, stored, edit } => write!(
                f,
                "native edit compiler: an edit of shape {edit:?} does not fit {storage:?} of shape {stored:?}"
            ),
            Self::DuplicateEdit(name) => write!(f, "native edit compiler: more than one edit of {name:?}"),
            Self::UseSite { site, reason } => write!(f, "native edit compiler: use site {site:?}: {reason}"),
            Self::StatusTooStrong { claimed, evidence } => write!(
                f,
                "native edit compiler: a {claimed} status cannot rest on {evidence} evidence"
            ),
            Self::PlanStatusMismatch { control } => write!(
                f,
                "native edit compiler: control {control:?} has a plan exactly when its status is native"
            ),
            Self::Lift(error) => write!(f, "{error}"),
            Self::Apply(error) => write!(f, "{error}"),
            Self::Dense(error) => write!(f, "{error}"),
            Self::Evidence(error) => write!(f, "{error}"),
            Self::Secant(error) => write!(f, "{error}"),
            Self::Bound(error) => write!(f, "{error}"),
            Self::Joint(refusal) => write!(f, "joint operator refused: {refusal:?}"),
            Self::Gauge(refusal) => write!(f, "gauge refused: {refusal:?}"),
            Self::Memory(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<LiftError> for CompileError {
    fn from(error: LiftError) -> Self {
        Self::Lift(error)
    }
}

impl From<ApplyError> for CompileError {
    fn from(error: ApplyError) -> Self {
        Self::Apply(error)
    }
}

impl From<DenseError> for CompileError {
    fn from(error: DenseError) -> Self {
        Self::Dense(error)
    }
}

impl From<EvidenceStatusError> for CompileError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

impl From<SecantError> for CompileError {
    fn from(error: SecantError) -> Self {
        Self::Secant(error)
    }
}

impl From<BoundError> for CompileError {
    fn from(error: BoundError) -> Self {
        Self::Bound(error)
    }
}

impl From<JointRefusal> for CompileError {
    fn from(error: JointRefusal) -> Self {
        Self::Joint(error)
    }
}

impl From<GaugeRefusal> for CompileError {
    fn from(error: GaugeRefusal) -> Self {
        Self::Gauge(error)
    }
}

impl From<MemoryReservationError> for CompileError {
    fn from(error: MemoryReservationError) -> Self {
        Self::Memory(error)
    }
}

pub(crate) fn require_finite(what: &'static str, values: impl IntoIterator<Item = f64>) -> Result<(), CompileError> {
    if values.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(CompileError::NonFinite { what })
    }
}

pub(crate) fn require_shape(
    what: &'static str,
    expected: (usize, usize),
    found: (usize, usize),
) -> Result<(), CompileError> {
    if expected == found {
        Ok(())
    } else {
        Err(CompileError::Shape { what, expected, found })
    }
}
