//! Occurrence scopes of parameter edits (#2951).
//!
//! One stored tensor can be read at several use sites: a tied embedding and
//! unembedding, a tied encoder whose decoder reads the transpose, a shared body
//! called at several depths. Use `u` reads `τ_u(Θ)`, where the tie `τ_u` is the
//! identity or the transpose of a linear read ([`TieOrientation`]). A read that applies
//! no linear map, such as an embedding lookup or a norm gain, reads the stored values
//! ([`UseMap::Stored`]), so its `τ_u` is the identity. An edit `Δ` of the stored tensor
//! then names two different experiments:
//!
//! * a **global** edit changes the stored tensor, so every use `u` reads
//!   `τ_u(Θ + Δ)`;
//! * a **use-specific** edit changes one occurrence, so use `u` reads
//!   `τ_u(Θ + Δ)` and every other use `v` reads `τ_v(Θ)`.
//!
//! When two uses are multiplied, or a nonlinearity separates them, the global
//! response is not the sum of the use-specific responses. Only first derivatives
//! add: along a global edit the loss moves by `Σ_u ⟨G_u, τ_u(Δ)⟩_F`, where `G_u` is
//! the gradient with respect to what use `u` reads. The cotangent of a global edit
//! is therefore the tied adjoint `Σ_u τ_u*(G_u)`, and the transpose is its own
//! Frobenius adjoint ([`edit_cotangent`]). In the moment geometry, a globally masked
//! component has one generator part `τ_u(v)` in the block of every use it reaches
//! ([`EditScope::affected_uses`]), and an occurrence-level mask has a part in one
//! use's block only.
//!
//! # Use sites and forward paths
//!
//! A use site is the `k`-th read of a storage tensor in one forward, as the executing
//! framework's discovery pass numbers it. An ordinal addresses a read only on the
//! forward path it was discovered on: a full forward and a key-value-cache decode
//! step read the same tensors a different number of times. One registry holds one
//! discovery pass, so a record keeps the registry's [`TeacherFingerprint`], which
//! folds every registered use site, and refuses to resolve against a registry whose
//! fingerprint differs.
//!
//! # Positions and the key-value cache
//!
//! An edit reaches every position of a pass or only declared query positions
//! ([`PositionScope`]). A use evaluated at a position outside the scope reads the
//! original tensor. A key-value cache filled before the edit is the scope
//! `p..length` on the key and value projections, where `p` is the first uncached
//! position: the cached prefix keeps its clean keys and values, and every later
//! query attends over both.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::apply::{ApplyError, FactoredEdit};
use super::field::{CotangentTerm, ParameterCotangent};
use crate::inference::intervention_shard::{InterventionChange, ParameterEditScope};
use super::lift::{
    LiftError, TeacherFingerprint, TensorId, TensorRegistry, TieOrientation, UseMap, UseSiteId,
};

/// Which uses of a stored tensor read an edit.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EditScope {
    /// Every use that reads the storage tensor, under any name, reads the edited
    /// value.
    Global(TensorId),
    /// Only this use reads the edited value. Every other use of the same storage
    /// reads the original.
    UseSite(UseSiteId),
}

impl EditScope {
    /// The storage tensor the edit changes and its matrix shape `(rows, cols)`.
    ///
    /// A global scope names the storage itself, not an alias of it, so one global
    /// experiment has one representation. An edit is held as matrix factors, so the
    /// storage must be matrix-shaped.
    pub fn storage(
        &self,
        registry: &TensorRegistry,
    ) -> Result<(TensorId, (usize, usize)), OccurrenceError> {
        let storage = match self {
            Self::Global(name) => {
                let storage = registry.storage_of(name)?;
                if storage != name {
                    return Err(OccurrenceError::NotStorage {
                        name: name.clone(),
                        storage: storage.clone(),
                    });
                }
                storage.clone()
            }
            Self::UseSite(use_site) => registry.resolve_use_site(use_site)?.storage,
        };
        let shape = registry
            .storage(&storage)
            .map(|tensor| tensor.shape.clone())
            .ok_or_else(|| LiftError::UnknownTensor(storage.0.clone()))?;
        if let [rows, cols] = shape[..] {
            return Ok((storage, (rows, cols)));
        }
        Err(OccurrenceError::NotAMatrix { storage, shape })
    }

    /// The uses that read the edited value, in the registry's id order. No
    /// experiment edits a tensor that nothing reads, so a global scope on an unused
    /// tensor is refused.
    pub fn affected_uses(
        &self,
        registry: &TensorRegistry,
    ) -> Result<Vec<UseSiteId>, OccurrenceError> {
        let storage = self.storage(registry)?.0;
        match self {
            Self::Global(..) => {
                let uses: Vec<UseSiteId> =
                    registry.use_sites_of(&storage).into_iter().cloned().collect();
                if uses.is_empty() {
                    return Err(OccurrenceError::UnusedTensor(storage));
                }
                Ok(uses)
            }
            Self::UseSite(use_site) => Ok(vec![use_site.clone()]),
        }
    }
}

/// The query positions at which a use reads the edited value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionScope {
    declared: Option<Vec<usize>>,
}

impl PositionScope {
    /// The edit reaches every position of the pass.
    pub fn every() -> Self {
        Self { declared: None }
    }

    /// The edit reaches only `positions`, which must be non-empty and strictly
    /// increasing, so one scope has one representation.
    pub fn declared(positions: Vec<usize>) -> Result<Self, OccurrenceError> {
        if positions.is_empty() {
            return Err(OccurrenceError::InvalidPositions(
                "no positions are declared".to_string(),
            ));
        }
        if let Some(pair) = positions.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(OccurrenceError::InvalidPositions(format!(
                "positions must be strictly increasing; {} is followed by {}",
                pair[0], pair[1]
            )));
        }
        Ok(Self {
            declared: Some(positions),
        })
    }

    /// The declared positions, or `None` when the edit reaches every position.
    pub fn positions(&self) -> Option<&[usize]> {
        self.declared.as_deref()
    }

    /// Whether a use evaluated at `position` reads the edited value.
    pub fn reaches(&self, position: usize) -> bool {
        match &self.declared {
            None => true,
            Some(positions) => positions.binary_search(&position).is_ok(),
        }
    }

    /// Refuse a declared position outside a sequence of `length` positions.
    pub fn check_within(&self, length: usize) -> Result<(), OccurrenceError> {
        match self.declared.as_deref().and_then(|positions| positions.last()) {
            Some(&last) if last >= length => Err(OccurrenceError::InvalidPositions(format!(
                "position {last} is outside a sequence of length {length}"
            ))),
            _ => Ok(()),
        }
    }
}

/// The executing framework's name for one discovered read of a parameter: the module
/// whose call made the read and the op that made it, such as `("embed_out", "F.linear")`
/// for a tied unembedding read outside any module call. Discovery reports it beside the
/// read's use site. A use-specific edit carries it so that a runner can refuse an ordinal
/// that addresses another read. The module is the innermost executing module's qualified
/// name, which is empty for a read in the root module's own forward. So only the op must
/// be non-empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadLabel {
    pub module: String,
    pub op: String,
}

/// One parameter edit: which uses read it, at which positions, and the change of
/// the stored tensor, held as its factors in the stored orientation. The fields are
/// private, so every record has been checked against the registry by
/// [`ParameterEditRecord::new`].
#[derive(Clone, Debug)]
pub struct ParameterEditRecord {
    scope: EditScope,
    positions: PositionScope,
    delta: FactoredEdit,
    registry: TeacherFingerprint,
}

impl ParameterEditRecord {
    /// `delta` is in the stored orientation, so it must have the stored tensor's
    /// shape whatever the ties of the uses it reaches. A delta with no term changes
    /// nothing, so it is refused; [`FactoredEdit::new`] already refused non-finite
    /// factors.
    pub fn new(
        registry: &TensorRegistry,
        scope: EditScope,
        positions: PositionScope,
        delta: FactoredEdit,
    ) -> Result<Self, OccurrenceError> {
        let (storage, stored) = scope.storage(registry)?;
        let shape = (delta.output_dim(), delta.input_dim());
        if shape != stored {
            return Err(OccurrenceError::DeltaShape {
                storage,
                stored,
                delta: shape,
            });
        }
        if delta.term_count() == 0 {
            return Err(OccurrenceError::InvalidDelta(
                "the edit has no term".to_string(),
            ));
        }
        scope.affected_uses(registry)?;
        Ok(Self {
            scope,
            positions,
            delta,
            registry: registry.teacher_fingerprint(),
        })
    }

    pub fn scope(&self) -> &EditScope {
        &self.scope
    }

    pub fn positions(&self) -> &PositionScope {
        &self.positions
    }

    pub fn delta(&self) -> &FactoredEdit {
        &self.delta
    }

    /// The fingerprint of the registry the record was checked against.
    pub fn registry(&self) -> TeacherFingerprint {
        self.registry
    }

    /// The edit `τ_u(Δ)` that use `use_site` reads, or `None` when the record does
    /// not reach that use. `(Σ_k u_k v_kᵀ)ᵀ = Σ_k v_k u_kᵀ`, so a transposed use
    /// reads the swapped factors and the matrix is never formed. A registry with a
    /// different fingerprint numbers its uses on another path, so it is refused.
    pub fn delta_read_at(
        &self,
        registry: &TensorRegistry,
        use_site: &UseSiteId,
    ) -> Result<Option<FactoredEdit>, OccurrenceError> {
        let found = registry.teacher_fingerprint();
        if found != self.registry {
            return Err(OccurrenceError::RegistryMismatch {
                record: self.registry,
                registry: found,
            });
        }
        let read = registry.resolve_use_site(use_site)?;
        let reached = match &self.scope {
            EditScope::Global(storage) => &read.storage == storage,
            EditScope::UseSite(edited) => edited == use_site,
        };
        if !reached {
            return Ok(None);
        }
        match read.map {
            UseMap::Linear(TieOrientation::Identity) | UseMap::Stored => {
                Ok(Some(self.delta.clone()))
            }
            UseMap::Linear(TieOrientation::Transpose) => Ok(Some(FactoredEdit::new(
                self.delta.right().to_owned(),
                self.delta.left().to_owned(),
            )?)),
        }
    }

    /// The record as the declared changes a framework runner applies. Each
    /// [`InterventionChange::ParameterEdit`] names the storage tensor and carries
    /// `left · rightᵀ` in the stored orientation, row-major, whatever the tie of the read
    /// it reaches: the runner edits the stored value, and the read applies its own tie.
    ///
    /// - A global record that reaches every position is one [`ParameterEditScope::Global`]
    ///   change.
    /// - A global record with declared positions is one [`ParameterEditScope::UseSite`]
    ///   change per use it reaches, each seen at those positions. A global change reaches
    ///   every position of the pass, and every use reading the edit at those positions is
    ///   the same experiment. A key-value cache filled before a global edit is this case.
    /// - A use-specific record is one use-site change at its read's ordinal.
    ///
    /// A use-site change names its read by `labels`, which discovery reported for the
    /// read, so a read with no label, or with an empty op, is refused. A change carries at
    /// most `min(rows, cols)` terms, so a record with more is refused, not refactored.
    /// The plan carrying the changes must declare, as its forward path, the discovery
    /// pass this registry holds: an ordinal addresses a read only on that path, which is
    /// why a registry with another fingerprint is refused.
    pub fn intervention_changes(
        &self,
        registry: &TensorRegistry,
        labels: &BTreeMap<UseSiteId, ReadLabel>,
    ) -> Result<Vec<InterventionChange>, OccurrenceError> {
        let found = registry.teacher_fingerprint();
        if found != self.registry {
            return Err(OccurrenceError::RegistryMismatch {
                record: self.registry,
                registry: found,
            });
        }
        let (storage, (rows, cols)) = self.scope.storage(registry)?;
        let rank = self.delta.term_count();
        if rank > rows.min(cols) {
            return Err(OccurrenceError::RankExceedsShape {
                storage,
                rank,
                rows,
                cols,
            });
        }
        let positions = self.positions.positions().map(<[usize]>::to_vec);
        let scopes = match (&self.scope, positions) {
            (EditScope::Global(..), None) => vec![ParameterEditScope::Global],
            (_, positions) => {
                let affected = self.scope.affected_uses(registry)?;
                let reads = registry.use_sites_of(&storage).len();
                let numbered: Vec<UseSiteId> = (0..reads)
                    .map(|ordinal| UseSiteId::read(&storage, ordinal))
                    .collect();
                if let Some(use_site) = affected.iter().find(|use_site| !numbered.contains(use_site)) {
                    return Err(OccurrenceError::UnnumberedUseSite {
                        use_site: use_site.clone(),
                        storage,
                        reads,
                    });
                }
                // In ordinal order, so one record has one list of changes.
                let mut scopes = Vec::with_capacity(affected.len());
                for (ordinal, use_site) in numbered.iter().enumerate() {
                    if !affected.contains(use_site) {
                        continue;
                    }
                    let label = labels
                        .get(use_site)
                        .filter(|label| !label.op.is_empty())
                        .ok_or_else(|| OccurrenceError::UnlabelledRead(use_site.clone()))?;
                    scopes.push(ParameterEditScope::UseSite {
                        ordinal,
                        read_module: label.module.clone(),
                        read_op: label.op.clone(),
                        positions: positions.clone(),
                    });
                }
                scopes
            }
        };
        let left: Vec<f64> = self.delta.left().iter().copied().collect();
        let right: Vec<f64> = self.delta.right().iter().copied().collect();
        Ok(scopes
            .into_iter()
            .map(|scope| InterventionChange::ParameterEdit {
                parameter: storage.0.clone(),
                rows,
                cols,
                rank,
                left: left.clone(),
                right: right.clone(),
                scope,
            })
            .collect())
    }
}

/// The cotangent of an edit of `scope`: `Σ_u τ_u*(G_u)` over exactly the uses the
/// scope reaches, in the stored orientation.
///
/// `per_use` holds `G_u`, the gradient with respect to what use `u` reads, in that
/// use's orientation. A transposed use's outer-product term is pulled back by
/// swapping its factors, `(Σ_i g_i x_iᵀ)ᵀ = Σ_i x_i g_iᵀ`, so it is never formed. A
/// partial sum is a wrong gradient that looks like a right one, so a reached use
/// left out is refused, and so is a use the scope does not reach.
pub fn edit_cotangent(
    registry: &TensorRegistry,
    scope: &EditScope,
    per_use: Vec<(UseSiteId, CotangentTerm)>,
) -> Result<ParameterCotangent, OccurrenceError> {
    let stored = scope.storage(registry)?.1;
    let reached = scope.affected_uses(registry)?;
    let mut seen = BTreeSet::new();
    let mut terms = Vec::with_capacity(per_use.len());
    for (use_site, term) in per_use {
        let read = registry.resolve_use_site(&use_site)?;
        if !reached.contains(&use_site) {
            return Err(OccurrenceError::UnreachedUse {
                scope: scope.clone(),
                use_site,
            });
        }
        if seen.contains(&use_site) {
            return Err(OccurrenceError::RepeatedUse(use_site));
        }
        let pulled_back = match (read.map, term) {
            (UseMap::Linear(TieOrientation::Identity) | UseMap::Stored, term) => term,
            (UseMap::Linear(TieOrientation::Transpose), CotangentTerm::Dense(g)) => {
                CotangentTerm::Dense(g.t().to_owned())
            }
            (UseMap::Linear(TieOrientation::Transpose), CotangentTerm::Outer { output, input }) => {
                CotangentTerm::Outer {
                    output: input,
                    input: output,
                }
            }
        };
        let shape = match &pulled_back {
            CotangentTerm::Dense(g) => g.dim(),
            CotangentTerm::Outer { output, input } => (output.ncols(), input.ncols()),
        };
        if shape != stored {
            return Err(OccurrenceError::GradientShape {
                use_site,
                stored,
                pulled_back: shape,
            });
        }
        seen.insert(use_site);
        terms.push(pulled_back);
    }
    if let Some(use_site) = reached.into_iter().find(|use_site| !seen.contains(use_site)) {
        return Err(OccurrenceError::MissingUse {
            scope: scope.clone(),
            use_site,
        });
    }
    ParameterCotangent::from_terms(stored.0, stored.1, terms).map_err(OccurrenceError::Cotangent)
}

/// Refuse an external execution whose use sites disagree with the registry or the records.
///
/// `discovered` lists every use site the executed forward read, in execution order.
/// `substituted` lists every use site that read an edited value. Both use the executor's
/// numbering.
///
/// A forward that read a different set of use sites, or numbered them differently, ran
/// another path, where the ordinals address other reads. So the discovered reads of each
/// storage tensor must be exactly its registered sites, in ordinal order.
///
/// An edit that reached the wrong use is a different experiment, even when its numbers
/// agree. So every use a record reaches must be substituted exactly once, and no other use
/// may be.
///
/// Comparing ordinals needs the registry to number the `n` sites of each storage tensor
/// as its reads `storage#0` to `storage#(n-1)`. `TensorRegistry::register_use_site` accepts
/// any id, so a registered site outside that numbering, a gap or another name, is refused
/// first, as a registry defect, before any discovered read is blamed for it.
pub fn check_substitutions(
    registry: &TensorRegistry,
    records: &[ParameterEditRecord],
    discovered: &[UseSiteId],
    substituted: &[UseSiteId],
) -> Result<(), OccurrenceError> {
    let found = registry.teacher_fingerprint();
    if let Some(record) = records.iter().find(|record| record.registry != found) {
        return Err(OccurrenceError::RegistryMismatch {
            record: record.registry,
            registry: found,
        });
    }
    for storage in registry.storage_ids() {
        let sites = registry.use_sites_of(storage);
        let reads: BTreeSet<UseSiteId> = (0..sites.len())
            .map(|ordinal| UseSiteId::read(storage, ordinal))
            .collect();
        // The sites are distinct and as many as the reads, so they are the reads exactly
        // when each is one of them.
        if let Some(use_site) = sites.into_iter().find(|use_site| !reads.contains(*use_site)) {
            return Err(OccurrenceError::UnnumberedUseSite {
                use_site: use_site.clone(),
                storage: storage.clone(),
                reads: reads.len(),
            });
        }
    }
    let mut next_ordinal: BTreeMap<TensorId, usize> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for use_site in discovered {
        let storage = match registry.resolve_use_site(use_site) {
            Ok(read) => read.storage,
            Err(LiftError::UnknownUseSite(..)) => {
                return Err(OccurrenceError::UnregisteredDiscovery(use_site.clone()));
            }
            Err(err) => return Err(err.into()),
        };
        if !seen.insert(use_site.clone()) {
            return Err(OccurrenceError::RepeatedDiscovery(use_site.clone()));
        }
        let ordinal = next_ordinal.entry(storage.clone()).or_insert(0);
        let expected = UseSiteId::read(&storage, *ordinal);
        if use_site != &expected {
            return Err(OccurrenceError::OrdinalOutOfOrder {
                use_site: use_site.clone(),
                expected,
            });
        }
        *ordinal += 1;
    }
    for storage in registry.storage_ids() {
        if let Some(missing) = registry
            .use_sites_of(storage)
            .into_iter()
            .find(|use_site| !seen.contains(*use_site))
        {
            return Err(OccurrenceError::UndiscoveredUse(missing.clone()));
        }
    }
    let mut planned = BTreeSet::new();
    for record in records {
        planned.extend(record.scope.affected_uses(registry)?);
    }
    let mut applied = BTreeSet::new();
    for use_site in substituted {
        if !planned.contains(use_site) {
            return Err(OccurrenceError::UnplannedSubstitution(use_site.clone()));
        }
        if !applied.insert(use_site.clone()) {
            return Err(OccurrenceError::RepeatedSubstitution(use_site.clone()));
        }
    }
    match planned.into_iter().find(|use_site| !applied.contains(use_site)) {
        Some(use_site) => Err(OccurrenceError::MissingSubstitution(use_site)),
        None => Ok(()),
    }
}

/// Typed refusals of occurrence scopes, edit records, edit cotangents and executed
/// substitutions.
#[derive(Clone, Debug, PartialEq)]
pub enum OccurrenceError {
    /// The registry refused a name or a use site.
    Lift(LiftError),
    /// A global scope names an alias. It must name the storage the alias reads.
    NotStorage { name: TensorId, storage: TensorId },
    /// An edit is held as matrix factors, so its storage must be a matrix.
    NotAMatrix {
        storage: TensorId,
        shape: Vec<usize>,
    },
    /// Nothing reads the tensor, so no experiment can edit it.
    UnusedTensor(TensorId),
    InvalidDelta(String),
    InvalidPositions(String),
    /// The delta is not in the stored orientation of its tensor.
    DeltaShape {
        storage: TensorId,
        stored: (usize, usize),
        delta: (usize, usize),
    },
    /// The record was checked against a registry with another fingerprint, whose
    /// use-site ordinals may number the reads of another forward path.
    RegistryMismatch {
        record: TeacherFingerprint,
        registry: TeacherFingerprint,
    },
    /// The factored edit refused the factors a use reads.
    Apply(ApplyError),
    /// A cotangent term was supplied for a use the scope does not reach.
    UnreachedUse {
        scope: EditScope,
        use_site: UseSiteId,
    },
    RepeatedUse(UseSiteId),
    /// A use the scope reaches has no cotangent term, so the sum would be partial.
    MissingUse {
        scope: EditScope,
        use_site: UseSiteId,
    },
    /// A use's term, pulled back through its tie, is not the stored shape.
    GradientShape {
        use_site: UseSiteId,
        stored: (usize, usize),
        pulled_back: (usize, usize),
    },
    /// The cotangent refused the pulled-back terms.
    Cotangent(String),
    /// The registry holds `reads` use sites of `storage`, and `use_site` is not one of
    /// its reads `storage#0` to `storage#(reads-1)`, so no ordinal of an executed forward
    /// addresses it.
    UnnumberedUseSite {
        use_site: UseSiteId,
        storage: TensorId,
        reads: usize,
    },
    /// The executor discovered a use site the registry does not hold.
    UnregisteredDiscovery(UseSiteId),
    /// The executor reported one use site as discovered twice.
    RepeatedDiscovery(UseSiteId),
    /// A storage tensor's reads were not discovered in ordinal order, so the executor
    /// numbered another path.
    OrdinalOutOfOrder {
        use_site: UseSiteId,
        expected: UseSiteId,
    },
    /// A registered use site was not discovered, so the forward read the tensors on
    /// another path.
    UndiscoveredUse(UseSiteId),
    /// A use site read an edited value that no record reaches.
    UnplannedSubstitution(UseSiteId),
    RepeatedSubstitution(UseSiteId),
    /// A use that a record reaches did not read the edited value.
    MissingSubstitution(UseSiteId),
    /// A declared parameter edit carries at most `min(rows, cols)` terms, and the
    /// record's delta has `rank`.
    RankExceedsShape {
        storage: TensorId,
        rank: usize,
        rows: usize,
        cols: usize,
    },
    /// Discovery gave no label, or an empty op, for a read a use-site change names, so
    /// a runner could not check that the change's ordinal addresses that read.
    UnlabelledRead(UseSiteId),
}

impl From<LiftError> for OccurrenceError {
    fn from(err: LiftError) -> Self {
        Self::Lift(err)
    }
}

impl From<ApplyError> for OccurrenceError {
    fn from(err: ApplyError) -> Self {
        Self::Apply(err)
    }
}

impl fmt::Display for OccurrenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lift(err) => write!(f, "occurrence: {err}"),
            Self::NotStorage { name, storage } => write!(
                f,
                "occurrence: a global edit names the alias {} of storage {}; it must name the storage",
                name.0, storage.0
            ),
            Self::NotAMatrix { storage, shape } => write!(
                f,
                "occurrence: an edit is held as matrix factors, but storage {} has shape {shape:?}",
                storage.0
            ),
            Self::UnusedTensor(storage) => write!(
                f,
                "occurrence: storage {} has no use site, so no edit of it is an experiment",
                storage.0
            ),
            Self::InvalidDelta(reason) => write!(f, "occurrence: invalid edit: {reason}"),
            Self::InvalidPositions(reason) => {
                write!(f, "occurrence: invalid position scope: {reason}")
            }
            Self::DeltaShape {
                storage,
                stored,
                delta,
            } => write!(
                f,
                "occurrence: the edit of storage {} has shape {delta:?}; the storage has shape {stored:?}",
                storage.0
            ),
            Self::RegistryMismatch { record, registry } => write!(
                f,
                "occurrence: the record was checked against registry fingerprint {:#018x}, but this registry has {:#018x}; a use-site ordinal addresses a read only on the forward path it was discovered on",
                record.0, registry.0
            ),
            Self::Apply(err) => write!(f, "occurrence: {err}"),
            Self::UnreachedUse { scope, use_site } => write!(
                f,
                "occurrence: a cotangent term was supplied for use {}, which an edit of scope {scope:?} does not reach",
                use_site.0
            ),
            Self::RepeatedUse(use_site) => {
                write!(f, "occurrence: use {} has more than one cotangent term", use_site.0)
            }
            Self::MissingUse { scope, use_site } => write!(
                f,
                "occurrence: use {} is reached by an edit of scope {scope:?} but has no cotangent term",
                use_site.0
            ),
            Self::GradientShape {
                use_site,
                stored,
                pulled_back,
            } => write!(
                f,
                "occurrence: the cotangent term of use {} has shape {pulled_back:?} after its tie; the storage has shape {stored:?}",
                use_site.0
            ),
            Self::Cotangent(reason) => write!(f, "occurrence: invalid cotangent: {reason}"),
            Self::UnnumberedUseSite {
                use_site,
                storage,
                reads,
            } => write!(
                f,
                "occurrence: the registry holds {reads} use sites of storage {}, which must be its reads {}#0 onward with no gap; registered use site {} is not one of them",
                storage.0, storage.0, use_site.0
            ),
            Self::UnregisteredDiscovery(use_site) => write!(
                f,
                "occurrence: the executor discovered use site {}, which the registry does not hold",
                use_site.0
            ),
            Self::RepeatedDiscovery(use_site) => write!(
                f,
                "occurrence: the executor discovered use site {} twice",
                use_site.0
            ),
            Self::OrdinalOutOfOrder { use_site, expected } => write!(
                f,
                "occurrence: the executor discovered {} where the next read of its storage is {}, so it numbered another path",
                use_site.0, expected.0
            ),
            Self::UndiscoveredUse(use_site) => write!(
                f,
                "occurrence: registered use site {} was not discovered, so the forward ran another path",
                use_site.0
            ),
            Self::UnplannedSubstitution(use_site) => write!(
                f,
                "occurrence: use site {} read an edited value that no record reaches",
                use_site.0
            ),
            Self::RepeatedSubstitution(use_site) => write!(
                f,
                "occurrence: use site {} was substituted twice",
                use_site.0
            ),
            Self::MissingSubstitution(use_site) => write!(
                f,
                "occurrence: use site {} is reached by a record but did not read the edited value",
                use_site.0
            ),
            Self::RankExceedsShape {
                storage,
                rank,
                rows,
                cols,
            } => write!(
                f,
                "occurrence: the edit of storage {} has {rank} terms; a declared parameter edit of a ({rows}, {cols}) tensor carries at most {}",
                storage.0,
                (*rows).min(*cols)
            ),
            Self::UnlabelledRead(use_site) => write!(
                f,
                "occurrence: discovery gave no label, or an empty op, for read {}, which a use-site parameter edit must name",
                use_site.0
            ),
        }
    }
}

impl std::error::Error for OccurrenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::intervention_shard::{
        CleanPass, ExperimentUnit, InterventionExperiment, InterventionExperimentPlan, KlPositions,
        Readout,
    };
    use crate::parameter_decomposition::field::FieldCoefficient;
    use gam_linalg::roundoff::accumulation_growth;
    use ndarray::{Array1, Array2, Axis, array};

    /// A registry holding `weight` as storage and one use site per tie, `weight#0`,
    /// `weight#1`, … in forward order, which is also their id order.
    fn registry_with_uses(
        weight: &Array2<f64>,
        ties: &[TieOrientation],
    ) -> (TensorRegistry, TensorId, Vec<UseSiteId>) {
        let mut registry = TensorRegistry::default();
        let storage = TensorId("weight".to_string());
        registry
            .register_storage(storage.clone(), weight.view().into_dyn())
            .expect("the storage registers");
        let uses = ties
            .iter()
            .enumerate()
            .map(|(ordinal, &tie)| {
                let site = UseSiteId::read(&storage, ordinal);
                registry
                    .register_use_site(site.clone(), storage.clone(), UseMap::Linear(tie))
                    .expect("the use site registers");
                site
            })
            .collect();
        (registry, storage, uses)
    }

    /// The linear chain `ℓ = cᵀ R_{u_K} ⋯ R_{u_1} x` over the uses of one tensor, where
    /// `R_u` is what use `u` reads. It is multilinear in what the uses read, so a
    /// global edit scaled by `ε` moves `ℓ` by a polynomial of degree at most the use
    /// count, and a use-specific edit moves it affinely.
    struct Chain {
        weight: Array2<f64>,
        input: Array1<f64>,
        readout: Array1<f64>,
    }

    impl Chain {
        fn rectangular() -> Self {
            Self {
                weight: array![[0.7, -1.3], [0.4, 0.9], [-1.1, 0.25]],
                input: array![0.6, -1.4],
                readout: array![1.2, -0.5, 0.8],
            }
        }

        fn square() -> Self {
            Self {
                weight: array![[0.7, -1.3, 0.2], [0.4, 0.9, -0.6], [-1.1, 0.25, 0.5]],
                input: array![0.6, -1.4, 0.35],
                readout: array![1.2, -0.5, 0.8],
            }
        }

        /// The same chain on absolute values: evaluated on it, the chain returns an
        /// upper bound on the absolute sum of the monomials of every quantity.
        fn absolute(&self) -> Self {
            Self {
                weight: self.weight.mapv(f64::abs),
                input: self.input.mapv(f64::abs),
                readout: self.readout.mapv(f64::abs),
            }
        }
    }

    fn rank_one_delta(chain: &Chain) -> FactoredEdit {
        let (rows, cols) = chain.weight.dim();
        let left = Array2::from_shape_vec((rows, 1), [0.3, -0.8, 1.5][..rows].to_vec())
            .expect("the left factor has its shape");
        let right = Array2::from_shape_vec((cols, 1), [-0.9, 0.45, 1.1][..cols].to_vec())
            .expect("the right factor has its shape");
        FactoredEdit::new(left, right).expect("the factors agree on the term count")
    }

    fn absolute_record(
        registry: &TensorRegistry,
        record: &ParameterEditRecord,
    ) -> ParameterEditRecord {
        let absolute = FactoredEdit::new(
            record.delta().left().mapv(f64::abs),
            record.delta().right().mapv(f64::abs),
        )
        .expect("the absolute factors agree on the term count");
        ParameterEditRecord::new(
            registry,
            record.scope().clone(),
            record.positions().clone(),
            absolute,
        )
        .expect("the absolute record is valid")
    }

    fn dense(delta: &FactoredEdit) -> Array2<f64> {
        delta.left().dot(&delta.right().t())
    }

    fn factors(delta: &FactoredEdit) -> (Array2<f64>, Array2<f64>) {
        (delta.left().to_owned(), delta.right().to_owned())
    }

    /// What use `use_site` reads: `τ_u(Θ)`, plus `scale · τ_u(Δ)` when the record
    /// reaches that use.
    fn read_at(
        registry: &TensorRegistry,
        chain: &Chain,
        record: &ParameterEditRecord,
        scale: f64,
        use_site: &UseSiteId,
    ) -> Array2<f64> {
        let map = registry
            .resolve_use_site(use_site)
            .expect("the use is registered")
            .map;
        let original = match map {
            UseMap::Linear(TieOrientation::Identity) | UseMap::Stored => chain.weight.clone(),
            UseMap::Linear(TieOrientation::Transpose) => chain.weight.t().to_owned(),
        };
        match record
            .delta_read_at(registry, use_site)
            .expect("the use resolves")
        {
            Some(delta) => original + dense(&delta) * scale,
            None => original,
        }
    }

    fn response(
        registry: &TensorRegistry,
        chain: &Chain,
        uses: &[UseSiteId],
        record: &ParameterEditRecord,
        scale: f64,
    ) -> Array1<f64> {
        uses.iter().fold(chain.input.clone(), |state, use_site| {
            read_at(registry, chain, record, scale, use_site).dot(&state)
        })
    }

    fn loss(
        registry: &TensorRegistry,
        chain: &Chain,
        uses: &[UseSiteId],
        record: &ParameterEditRecord,
        scale: f64,
    ) -> f64 {
        chain
            .readout
            .dot(&response(registry, chain, uses, record, scale))
    }

    /// Rounded operations in [`response`], counted as totals so that any evaluation
    /// order the kernels choose stays inside the count: per use, the `r`-term product
    /// `rows·cols·(2r − 1)`, the scale and the addition `2·rows·cols`, and the
    /// matrix-vector product at most `2·rows·cols`.
    fn response_operations(chain: &Chain, uses: usize, terms: usize) -> usize {
        let (rows, cols) = chain.weight.dim();
        uses * rows * cols * (2 * terms + 3)
    }

    /// [`response_operations`] plus the readout inner product.
    fn loss_operations(chain: &Chain, uses: usize, terms: usize) -> usize {
        response_operations(chain, uses, terms) + 2 * chain.readout.len()
    }

    /// `p'(0)` of a polynomial of degree at most four, from `p(±1)` and `p(±2)`:
    /// `[8(p(1) − p(−1)) − (p(2) − p(−2))]/12` has no truncation term at that degree.
    fn five_point_derivative(p: impl Fn(f64) -> f64) -> f64 {
        (8.0 * (p(1.0) - p(-1.0)) - (p(2.0) - p(-2.0))) / 12.0
    }

    /// The rounding band of [`five_point_derivative`]. Each `p(h)` is within
    /// `γ_N·S(h)` of its exact value, where `S` evaluates `p` on absolute values, and
    /// the three subtractions and the division add four operations.
    fn five_point_band(operations: usize, absolute: impl Fn(f64) -> f64) -> f64 {
        accumulation_growth(operations + 4)
            * (8.0 * (absolute(1.0) + absolute(-1.0)) + absolute(2.0) + absolute(-2.0))
            / 12.0
    }

    /// The cotangent term of the `k`-th use, `G_u = ∂ℓ/∂R_u = b_{k+1} s_kᵀ` in that use's
    /// orientation, with `s_0 = x`, `s_{k+1} = R_k s_k`, `b_K = c` and
    /// `b_k = R_kᵀ b_{k+1}`. The use at `dense_use` gets the materialized term and every
    /// other use the one-row outer product, so both term kinds are pulled back.
    fn per_use_terms(
        registry: &TensorRegistry,
        chain: &Chain,
        uses: &[UseSiteId],
        record: &ParameterEditRecord,
        dense_use: usize,
    ) -> Vec<(UseSiteId, CotangentTerm)> {
        let reads: Vec<Array2<f64>> = uses
            .iter()
            .map(|use_site| read_at(registry, chain, record, 0.0, use_site))
            .collect();
        let mut states = vec![chain.input.clone()];
        for read in &reads {
            let next = read.dot(states.last().expect("a state exists"));
            states.push(next);
        }
        let mut covectors = vec![chain.readout.clone()];
        for read in reads.iter().rev() {
            let next = read.t().dot(covectors.last().expect("a covector exists"));
            covectors.push(next);
        }
        covectors.reverse();
        uses.iter()
            .enumerate()
            .map(|(k, use_site)| {
                let output = covectors[k + 1].view().insert_axis(Axis(0)).to_owned();
                let input = states[k].view().insert_axis(Axis(0)).to_owned();
                let term = if k == dense_use {
                    CotangentTerm::Dense(output.t().dot(&input))
                } else {
                    CotangentTerm::Outer { output, input }
                };
                (use_site.clone(), term)
            })
            .collect()
    }

    /// Rounded operations in [`per_use_terms`], [`edit_cotangent`] and the Frobenius
    /// pairing with a rank-`r` delta, as totals: per use, a forward and a backward
    /// matrix-vector product and at most the dense outer product `4·rows·cols +
    /// rows·cols`, and the pairing of one term, at most `2·rows·cols·r + 4·(rows +
    /// cols)·r`, then one addition per term.
    fn pairing_operations(chain: &Chain, uses: usize, terms: usize) -> usize {
        let (rows, cols) = chain.weight.dim();
        uses * (5 * rows * cols + 2 * rows * cols * terms + 4 * (rows + cols) * terms + 1)
    }

    fn pair(
        registry: &TensorRegistry,
        scope: &EditScope,
        per_use: Vec<(UseSiteId, CotangentTerm)>,
        delta: &FactoredEdit,
    ) -> Result<f64, OccurrenceError> {
        edit_cotangent(registry, scope, per_use)?
            .frobenius_inner(&FieldCoefficient::Factored {
                left: delta.left().to_owned(),
                right: delta.right().to_owned(),
            })
            .map_err(OccurrenceError::Cotangent)
    }

    const TIES: [TieOrientation; 3] = [
        TieOrientation::Identity,
        TieOrientation::Transpose,
        TieOrientation::Identity,
    ];

    #[test]
    fn global_and_use_specific_edits_are_different_experiments_on_a_tied_tensor() {
        let chain = Chain::rectangular();
        let absolute_chain = chain.absolute();
        let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        let global = ParameterEditRecord::new(
            &registry,
            EditScope::Global(storage.clone()),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the global record is valid");
        let middle = ParameterEditRecord::new(
            &registry,
            EditScope::UseSite(uses[1].clone()),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the use-specific record is valid");
        let global_response = response(&registry, &chain, &uses, &global, 1.0);
        let middle_response = response(&registry, &chain, &uses, &middle, 1.0);
        let global_absolute = response(
            &registry,
            &absolute_chain,
            &uses,
            &absolute_record(&registry, &global),
            1.0,
        );
        let middle_absolute = response(
            &registry,
            &absolute_chain,
            &uses,
            &absolute_record(&registry, &middle),
            1.0,
        );
        let growth = accumulation_growth(response_operations(&chain, uses.len(), 1) + 1);
        let excess = (0..global_response.len())
            .map(|i| {
                (global_response[i] - middle_response[i]).abs()
                    - growth * (global_absolute[i] + middle_absolute[i])
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            excess > 0.0,
            "the global and the use-specific edit agree within roundoff: excess {excess:e}"
        );

        // Positive control: on an untied tensor the two scopes reach the same single
        // use, so the same arithmetic runs and no difference exists to detect.
        let (untied, untied_storage, untied_uses) =
            registry_with_uses(&chain.weight, &[TieOrientation::Identity]);
        let untied_global = ParameterEditRecord::new(
            &untied,
            EditScope::Global(untied_storage),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the untied global record is valid");
        let untied_use = ParameterEditRecord::new(
            &untied,
            EditScope::UseSite(untied_uses[0].clone()),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the untied use-specific record is valid");
        assert_eq!(
            response(&untied, &chain, &untied_uses, &untied_global, 1.0),
            response(&untied, &chain, &untied_uses, &untied_use, 1.0)
        );
    }

    #[test]
    fn first_derivatives_of_use_specific_edits_add_to_the_global_derivative() {
        let chain = Chain::rectangular();
        let absolute_chain = chain.absolute();
        let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        let operations = loss_operations(&chain, uses.len(), 1);
        let record = |scope| {
            ParameterEditRecord::new(
                &registry,
                scope,
                PositionScope::every(),
                rank_one_delta(&chain),
            )
            .expect("the record is valid")
        };
        let global = record(EditScope::Global(storage));
        let global_absolute = absolute_record(&registry, &global);
        let global_derivative =
            five_point_derivative(|scale| loss(&registry, &chain, &uses, &global, scale));
        let mut band = five_point_band(operations, |scale| {
            loss(&registry, &absolute_chain, &uses, &global_absolute, scale.abs())
        });
        let finite_growth = accumulation_growth(operations + 1);
        let mut summed = 0.0;
        let mut finite_sum = 0.0;
        let mut finite_band = finite_growth
            * (loss(&registry, &absolute_chain, &uses, &global_absolute, 1.0)
                + loss(&registry, &absolute_chain, &uses, &global_absolute, 0.0));
        for use_site in &uses {
            let specific = record(EditScope::UseSite(use_site.clone()));
            let specific_absolute = absolute_record(&registry, &specific);
            summed +=
                five_point_derivative(|scale| loss(&registry, &chain, &uses, &specific, scale));
            band += five_point_band(operations, |scale| {
                loss(&registry, &absolute_chain, &uses, &specific_absolute, scale.abs())
            });
            finite_sum += loss(&registry, &chain, &uses, &specific, 1.0)
                - loss(&registry, &chain, &uses, &specific, 0.0);
            finite_band += finite_growth
                * (loss(&registry, &absolute_chain, &uses, &specific_absolute, 1.0)
                    + loss(&registry, &absolute_chain, &uses, &specific_absolute, 0.0));
        }
        let combining = accumulation_growth(uses.len() + 1);
        let gap = (global_derivative - summed).abs();
        assert!(
            gap <= band * (1.0 + combining) + combining * (global_derivative.abs() + summed.abs()),
            "the global derivative {global_derivative} is not the sum {summed} of the use-specific derivatives: gap {gap:e}, band {band:e}"
        );

        // Positive control: the finite responses do not add, because the uses are
        // multiplied, so the same comparison on them fails.
        let global_finite = loss(&registry, &chain, &uses, &global, 1.0)
            - loss(&registry, &chain, &uses, &global, 0.0);
        let finite_gap = (global_finite - finite_sum).abs();
        assert!(
            finite_gap
                > finite_band * (1.0 + combining)
                    + combining * (global_finite.abs() + finite_sum.abs()),
            "the finite global response {global_finite} matches the sum {finite_sum} of use-specific responses within roundoff"
        );
    }

    #[test]
    fn edit_cotangent_pulls_each_use_back_through_its_tie() {
        for chain in [Chain::rectangular(), Chain::square()] {
            let absolute_chain = chain.absolute();
            let (rows, cols) = chain.weight.dim();
            let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
            let pairing_band_growth =
                accumulation_growth(pairing_operations(&chain, uses.len(), 1));
            let derivative_operations = loss_operations(&chain, uses.len(), 1);
            for dense_use in 0..uses.len() {
                let global_scope = EditScope::Global(storage.clone());
                let global = ParameterEditRecord::new(
                    &registry,
                    global_scope.clone(),
                    PositionScope::every(),
                    rank_one_delta(&chain),
                )
                .expect("the global record is valid");
                let global_absolute = absolute_record(&registry, &global);
                let paired = pair(
                    &registry,
                    &global_scope,
                    per_use_terms(&registry, &chain, &uses, &global, dense_use),
                    global.delta(),
                )
                .expect("every use has one term");
                let paired_absolute = pair(
                    &registry,
                    &global_scope,
                    per_use_terms(&registry, &absolute_chain, &uses, &global_absolute, dense_use),
                    global_absolute.delta(),
                )
                .expect("every use has one term");
                let derivative =
                    five_point_derivative(|scale| loss(&registry, &chain, &uses, &global, scale));
                let band = five_point_band(derivative_operations, |scale| {
                    loss(&registry, &absolute_chain, &uses, &global_absolute, scale.abs())
                }) + pairing_band_growth * paired_absolute;
                let gap = (paired - derivative).abs();
                assert!(
                    gap <= band * (1.0 + accumulation_growth(1)),
                    "shape ({rows}, {cols}), dense use {dense_use}: the tied cotangent pairs to {paired}, the exact derivative is {derivative}: gap {gap:e}, band {band:e}"
                );
            }

            // A use-specific edit's cotangent is its own use pulled back alone.
            let middle_scope = EditScope::UseSite(uses[1].clone());
            let middle = ParameterEditRecord::new(
                &registry,
                middle_scope.clone(),
                PositionScope::every(),
                rank_one_delta(&chain),
            )
            .expect("the use-specific record is valid");
            let middle_absolute = absolute_record(&registry, &middle);
            let only_middle = |terms: Vec<(UseSiteId, CotangentTerm)>| {
                terms
                    .into_iter()
                    .filter(|entry| entry.0 == uses[1])
                    .collect::<Vec<_>>()
            };
            let middle_paired = pair(
                &registry,
                &middle_scope,
                only_middle(per_use_terms(&registry, &chain, &uses, &middle, 0)),
                middle.delta(),
            )
            .expect("the reached use has its term");
            let middle_paired_absolute = pair(
                &registry,
                &middle_scope,
                only_middle(per_use_terms(&registry, &absolute_chain, &uses, &middle_absolute, 0)),
                middle_absolute.delta(),
            )
            .expect("the reached use has its term");
            let middle_derivative =
                five_point_derivative(|scale| loss(&registry, &chain, &uses, &middle, scale));
            let middle_band = five_point_band(derivative_operations, |scale| {
                loss(&registry, &absolute_chain, &uses, &middle_absolute, scale.abs())
            }) + pairing_band_growth * middle_paired_absolute;
            assert!(
                (middle_paired - middle_derivative).abs()
                    <= middle_band * (1.0 + accumulation_growth(1)),
                "shape ({rows}, {cols}): the use-specific cotangent pairs to {middle_paired}, the exact derivative is {middle_derivative}"
            );
        }

        // Positive control: declaring the transposed use as an identity read leaves a
        // square term's shape intact but pairs it with the wrong entries, and the same
        // comparison refutes it.
        let chain = Chain::square();
        let absolute_chain = chain.absolute();
        let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        let (misdeclared, misdeclared_storage, misdeclared_uses) =
            registry_with_uses(&chain.weight, &[TieOrientation::Identity; 3]);
        assert_eq!(misdeclared_uses, uses);
        let global = ParameterEditRecord::new(
            &registry,
            EditScope::Global(storage),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the global record is valid");
        let global_absolute = absolute_record(&registry, &global);
        let misdeclared_scope = EditScope::Global(misdeclared_storage);
        let wrong = pair(
            &misdeclared,
            &misdeclared_scope,
            per_use_terms(&registry, &chain, &uses, &global, 0),
            global.delta(),
        )
        .expect("the misdeclared reads still give square terms");
        let wrong_absolute = pair(
            &misdeclared,
            &misdeclared_scope,
            per_use_terms(&registry, &absolute_chain, &uses, &global_absolute, 0),
            global_absolute.delta(),
        )
        .expect("the misdeclared reads still give square terms");
        let derivative =
            five_point_derivative(|scale| loss(&registry, &chain, &uses, &global, scale));
        let band = five_point_band(loss_operations(&chain, uses.len(), 1), |scale| {
            loss(&registry, &absolute_chain, &uses, &global_absolute, scale.abs())
        }) + accumulation_growth(pairing_operations(&chain, uses.len(), 1)) * wrong_absolute;
        assert!(
            (wrong - derivative).abs() > band * (1.0 + accumulation_growth(1)),
            "a term pulled back through the wrong tie pairs to {wrong}, within roundoff of the exact derivative {derivative}"
        );
    }

    #[test]
    fn edit_cotangent_refuses_partial_repeated_unreached_misoriented_and_malformed_terms() {
        let chain = Chain::rectangular();
        let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        let scope = EditScope::Global(storage);
        let global = ParameterEditRecord::new(
            &registry,
            scope.clone(),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the global record is valid");
        let terms = per_use_terms(&registry, &chain, &uses, &global, uses.len());
        let picks = |indices: &[usize]| {
            indices
                .iter()
                .map(|&k| terms[k].clone())
                .collect::<Vec<_>>()
        };
        assert!(edit_cotangent(&registry, &scope, picks(&[0, 1, 2])).is_ok());
        assert_eq!(
            edit_cotangent(&registry, &scope, picks(&[0, 1])).err(),
            Some(OccurrenceError::MissingUse {
                scope: scope.clone(),
                use_site: uses[2].clone()
            })
        );
        assert_eq!(
            edit_cotangent(&registry, &scope, picks(&[0, 1, 2, 0])).err(),
            Some(OccurrenceError::RepeatedUse(uses[0].clone()))
        );
        let middle = EditScope::UseSite(uses[1].clone());
        assert!(edit_cotangent(&registry, &middle, picks(&[1])).is_ok());
        assert_eq!(
            edit_cotangent(&registry, &middle, picks(&[0])).err(),
            Some(OccurrenceError::UnreachedUse {
                scope: middle.clone(),
                use_site: uses[0].clone()
            })
        );

        // The transposed use given a term already in the stored orientation is swapped
        // into the use orientation instead, which is not the stored shape.
        let (output, input) = match &terms[1].1 {
            CotangentTerm::Outer { output, input } => Some((output.clone(), input.clone())),
            CotangentTerm::Dense(..) => None,
        }
        .expect("every term of this fixture is an outer product");
        assert_eq!((output.ncols(), input.ncols()), (2, 3));
        let misoriented = CotangentTerm::Outer {
            output: input.clone(),
            input: output.clone(),
        };
        assert_eq!(
            edit_cotangent(&registry, &middle, vec![(uses[1].clone(), misoriented)]).err(),
            Some(OccurrenceError::GradientShape {
                use_site: uses[1].clone(),
                stored: (3, 2),
                pulled_back: (2, 3)
            })
        );

        // A term whose two factors disagree on the row count has the right widths, so
        // only the cotangent's own check refuses it.
        let two_rows = ndarray::concatenate(Axis(0), &[output.view(), output.view()])
            .expect("the rows stack");
        let malformed = CotangentTerm::Outer {
            output: two_rows,
            input,
        };
        assert!(matches!(
            edit_cotangent(&registry, &middle, vec![(uses[1].clone(), malformed)]).err(),
            Some(OccurrenceError::Cotangent(..))
        ));
    }

    #[test]
    fn records_hold_the_stored_orientation_and_reach_only_their_scope() {
        let chain = Chain::rectangular();
        let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        let delta = rank_one_delta(&chain);
        let at_use_orientation =
            FactoredEdit::new(delta.right().to_owned(), delta.left().to_owned())
                .expect("the swapped factors agree on the term count");
        assert_eq!(
            ParameterEditRecord::new(
                &registry,
                EditScope::Global(storage.clone()),
                PositionScope::every(),
                at_use_orientation,
            )
            .err(),
            Some(OccurrenceError::DeltaShape {
                storage: storage.clone(),
                stored: (3, 2),
                delta: (2, 3)
            })
        );

        let global = ParameterEditRecord::new(
            &registry,
            EditScope::Global(storage.clone()),
            PositionScope::every(),
            delta.clone(),
        )
        .expect("the stored orientation is accepted");
        let middle = ParameterEditRecord::new(
            &registry,
            EditScope::UseSite(uses[1].clone()),
            PositionScope::every(),
            delta.clone(),
        )
        .expect("the stored orientation is accepted at a transposed use");
        let read = |record: &ParameterEditRecord, use_site: &UseSiteId| {
            record
                .delta_read_at(&registry, use_site)
                .expect("the use resolves")
                .map(|read| factors(&read))
        };
        let (left, right) = factors(&delta);
        let swapped = (right.clone(), left.clone());
        assert_eq!(read(&global, &uses[0]), Some((left.clone(), right.clone())));
        assert_eq!(read(&global, &uses[1]), Some(swapped.clone()));
        assert_eq!(read(&global, &uses[2]), Some((left, right)));
        assert_eq!(read(&middle, &uses[0]), None);
        assert_eq!(read(&middle, &uses[1]), Some(swapped));
        assert_eq!(read(&middle, &uses[2]), None);
        let transposed_read = global
            .delta_read_at(&registry, &uses[1])
            .expect("the use resolves")
            .expect("the global edit reaches the transposed use");
        assert_eq!(dense(&transposed_read), dense(&delta).t().to_owned());
        assert_eq!(
            EditScope::Global(storage.clone()).affected_uses(&registry),
            Ok(uses.clone())
        );
        assert_eq!(
            EditScope::UseSite(uses[1].clone()).affected_uses(&registry),
            Ok(vec![uses[1].clone()])
        );

        let (unused, unused_storage, no_uses) = registry_with_uses(&chain.weight, &[]);
        assert!(no_uses.is_empty());
        assert_eq!(
            ParameterEditRecord::new(
                &unused,
                EditScope::Global(unused_storage.clone()),
                PositionScope::every(),
                rank_one_delta(&chain),
            )
            .err(),
            Some(OccurrenceError::UnusedTensor(unused_storage))
        );
    }

    #[test]
    fn scopes_name_matrix_storage_through_the_registry() {
        let chain = Chain::rectangular();
        let (mut registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        assert_eq!(
            EditScope::Global(storage.clone()).storage(&registry),
            Ok((storage.clone(), (3, 2)))
        );
        assert_eq!(
            EditScope::UseSite(uses[1].clone()).storage(&registry),
            Ok((storage.clone(), (3, 2)))
        );
        let alias = TensorId("head.weight".to_string());
        registry
            .register_alias(alias.clone(), storage.clone())
            .expect("the alias registers");
        assert_eq!(
            EditScope::Global(alias.clone()).storage(&registry),
            Err(OccurrenceError::NotStorage {
                name: alias,
                storage
            })
        );
        let unknown = UseSiteId("nowhere".to_string());
        assert_eq!(
            EditScope::UseSite(unknown.clone()).storage(&registry),
            Err(OccurrenceError::Lift(LiftError::UnknownUseSite(unknown.0)))
        );
        let bias = TensorId("bias".to_string());
        registry
            .register_storage(bias.clone(), array![0.1, -0.2, 0.3].view().into_dyn())
            .expect("the bias registers");
        registry
            .register_use_site(UseSiteId::read(&bias, 0), bias.clone(), UseMap::Stored)
            .expect("the bias use registers");
        assert_eq!(
            EditScope::Global(bias.clone()).storage(&registry),
            Err(OccurrenceError::NotAMatrix {
                storage: bias,
                shape: vec![3]
            })
        );
    }

    #[test]
    fn a_stored_read_reaches_global_edits_in_the_stored_orientation() {
        // An embedding table read by lookup applies no linear map, so it registers as a
        // stored read. A global edit reaches it unchanged, and its cotangent is already a
        // gradient with respect to the stored values. The tied head reads the transpose.
        let weight = array![[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]];
        let mut registry = TensorRegistry::default();
        let storage = TensorId("embedding".to_string());
        registry
            .register_storage(storage.clone(), weight.view().into_dyn())
            .expect("the storage registers");
        let lookup = UseSiteId::read(&storage, 0);
        let head = UseSiteId::read(&storage, 1);
        registry
            .register_use_site(lookup.clone(), storage.clone(), UseMap::Stored)
            .expect("the lookup registers");
        registry
            .register_use_site(
                head.clone(),
                storage.clone(),
                UseMap::Linear(TieOrientation::Transpose),
            )
            .expect("the head registers");
        let delta = FactoredEdit::new(array![[1.0], [-1.0], [2.0]], array![[0.5], [-2.0]])
            .expect("the factors agree on the term count");
        let scope = EditScope::Global(storage.clone());
        let global = ParameterEditRecord::new(
            &registry,
            scope.clone(),
            PositionScope::every(),
            delta.clone(),
        )
        .expect("the global record is valid");
        assert_eq!(
            scope.affected_uses(&registry),
            Ok(vec![lookup.clone(), head.clone()])
        );
        let read = |use_site: &UseSiteId| {
            global
                .delta_read_at(&registry, use_site)
                .expect("the use resolves")
                .map(|read| factors(&read))
        };
        let (left, right) = factors(&delta);
        assert_eq!(read(&lookup), Some((left.clone(), right.clone())));
        assert_eq!(read(&head), Some((right, left)));

        // Every entry is a small dyadic number, so every product and sum below is exact
        // and the pairing must equal its hand value bit for bit.
        let lookup_gradient = array![[2.0, 0.0], [0.0, -1.0], [0.5, 1.0]];
        let head_gradient = array![[1.0, -0.5, 0.0], [0.25, 2.0, -1.0]];
        let cotangent = edit_cotangent(
            &registry,
            &scope,
            vec![
                (lookup.clone(), CotangentTerm::Dense(lookup_gradient.clone())),
                (head.clone(), CotangentTerm::Dense(head_gradient.clone())),
            ],
        )
        .expect("both uses have one term");
        let paired = cotangent
            .frobenius_inner(&FieldCoefficient::Factored {
                left: delta.left().to_owned(),
                right: delta.right().to_owned(),
            })
            .expect("the pairing is finite");
        let expected = (&(&lookup_gradient + &head_gradient.t()) * &dense(&delta)).sum();
        assert_eq!(paired, expected);

        // Positive control: the stored read's term given in the head's transposed shape
        // is refused, because a stored read pulls back as the identity.
        assert_eq!(
            edit_cotangent(
                &registry,
                &EditScope::UseSite(lookup.clone()),
                vec![(lookup.clone(), CotangentTerm::Dense(head_gradient))],
            )
            .err(),
            Some(OccurrenceError::GradientShape {
                use_site: lookup,
                stored: (3, 2),
                pulled_back: (2, 3)
            })
        );
    }

    #[test]
    fn substitutions_must_match_the_discovered_path_and_the_planned_uses() {
        let chain = Chain::rectangular();
        let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        let global = ParameterEditRecord::new(
            &registry,
            EditScope::Global(storage.clone()),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the global record is valid");
        let middle = ParameterEditRecord::new(
            &registry,
            EditScope::UseSite(uses[1].clone()),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the use-specific record is valid");
        let records = [global, middle.clone()];
        let pick = |indices: &[usize]| indices.iter().map(|&k| uses[k].clone()).collect::<Vec<_>>();
        assert_eq!(check_substitutions(&registry, &records, &uses, &uses), Ok(()));
        assert_eq!(
            check_substitutions(&registry, &[middle.clone()], &uses, &pick(&[1])),
            Ok(())
        );

        // An edit that reached another use is refused even though its numbers could agree.
        assert_eq!(
            check_substitutions(&registry, &[middle.clone()], &uses, &pick(&[0])),
            Err(OccurrenceError::UnplannedSubstitution(uses[0].clone()))
        );
        assert_eq!(
            check_substitutions(&registry, &records, &uses, &pick(&[0, 2])),
            Err(OccurrenceError::MissingSubstitution(uses[1].clone()))
        );
        assert_eq!(
            check_substitutions(&registry, &records, &uses, &pick(&[0, 1, 1, 2])),
            Err(OccurrenceError::RepeatedSubstitution(uses[1].clone()))
        );

        // Another forward path: a read left out, an extra read, a repeated read, or reads
        // numbered out of order.
        assert_eq!(
            check_substitutions(&registry, &records, &pick(&[0, 1]), &uses),
            Err(OccurrenceError::UndiscoveredUse(uses[2].clone()))
        );
        let extra = UseSiteId::read(&storage, 3);
        let mut longer = uses.clone();
        longer.push(extra.clone());
        assert_eq!(
            check_substitutions(&registry, &records, &longer, &uses),
            Err(OccurrenceError::UnregisteredDiscovery(extra))
        );
        assert_eq!(
            check_substitutions(&registry, &records, &pick(&[0, 0, 1, 2]), &uses),
            Err(OccurrenceError::RepeatedDiscovery(uses[0].clone()))
        );
        assert_eq!(
            check_substitutions(&registry, &records, &pick(&[1, 0, 2]), &uses),
            Err(OccurrenceError::OrdinalOutOfOrder {
                use_site: uses[1].clone(),
                expected: uses[0].clone()
            })
        );

        // Records checked against another path's registry are refused before any site is
        // compared.
        let (decode, decode_storage, decode_uses) = registry_with_uses(
            &chain.weight,
            &[
                TieOrientation::Identity,
                TieOrientation::Transpose,
                TieOrientation::Identity,
                TieOrientation::Identity,
            ],
        );
        assert_eq!(decode_storage, storage);
        assert_eq!(
            check_substitutions(&decode, &records, &decode_uses, &uses),
            Err(OccurrenceError::RegistryMismatch {
                record: registry.teacher_fingerprint(),
                registry: decode.teacher_fingerprint(),
            })
        );
    }

    #[test]
    fn substitutions_refuse_a_registry_whose_sites_are_not_numbered_as_reads() {
        let chain = Chain::rectangular();
        let storage = TensorId("weight".to_string());
        let read = |ordinal: usize| UseSiteId::read(&storage, ordinal);
        let registry_holding = |sites: &[UseSiteId]| {
            let mut registry = TensorRegistry::default();
            registry
                .register_storage(storage.clone(), chain.weight.view().into_dyn())
                .expect("the storage registers");
            for site in sites {
                registry
                    .register_use_site(
                        site.clone(),
                        storage.clone(),
                        UseMap::Linear(TieOrientation::Identity),
                    )
                    .expect("the use site registers");
            }
            registry
        };

        // Eleven reads, the fewest with a two-digit ordinal, registered last to first. Id
        // order puts weight#10 before weight#2 and registration order is reversed; only the
        // ordinals give the forward order, and the forward that reads them in it passes.
        let eleven: Vec<UseSiteId> = (0..11).map(read).collect();
        let backwards: Vec<UseSiteId> = eleven.iter().rev().cloned().collect();
        let numbered = registry_holding(&backwards);
        assert_eq!(check_substitutions(&numbered, &[], &eleven, &[]), Ok(()));

        // A gap: two sites registered as reads 0 and 2. A forward that reads the tensor twice
        // reports reads 0 and 1, and the registry is refused, not the forward.
        let gapped = registry_holding(&[read(0), read(2)]);
        assert_eq!(
            check_substitutions(&gapped, &[], &[read(0), read(1)], &[]),
            Err(OccurrenceError::UnnumberedUseSite {
                use_site: read(2),
                storage: storage.clone(),
                reads: 2,
            })
        );

        // A site registered under another name than its read.
        let named = UseSiteId("decoder".to_string());
        let renamed = registry_holding(&[read(0), named.clone()]);
        assert_eq!(
            check_substitutions(&renamed, &[], &[read(0), read(1)], &[]),
            Err(OccurrenceError::UnnumberedUseSite {
                use_site: named,
                storage: storage.clone(),
                reads: 2,
            })
        );
    }

    #[test]
    fn records_become_the_declared_changes_a_runner_applies() {
        let chain = Chain::rectangular();
        let (rows, cols) = chain.weight.dim();
        let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        // Rank two, so carrying the factors row-major differs from carrying them
        // column-major.
        let delta = FactoredEdit::new(
            array![[0.3, 1.2], [-0.8, 0.1], [1.5, -0.6]],
            array![[-0.9, 0.7], [0.45, -0.2]],
        )
        .expect("the factors agree on the term count");
        let record = |scope: EditScope, positions: PositionScope| {
            ParameterEditRecord::new(&registry, scope, positions, delta.clone())
                .expect("the record is valid")
        };
        let label = |ordinal: usize| ReadLabel {
            module: format!("layers.{ordinal}"),
            op: "F.linear".to_string(),
        };
        let labels: BTreeMap<UseSiteId, ReadLabel> = uses
            .iter()
            .enumerate()
            .map(|(ordinal, use_site)| (use_site.clone(), label(ordinal)))
            .collect();
        let at_read = |ordinal: usize, positions: Option<Vec<usize>>| ParameterEditScope::UseSite {
            ordinal,
            read_module: label(ordinal).module,
            read_op: label(ordinal).op,
            positions,
        };
        // `left` is (rows, rank) and `right` is (cols, rank), both row-major, in the
        // stored orientation.
        let carried = |scope: ParameterEditScope| InterventionChange::ParameterEdit {
            parameter: storage.0.clone(),
            rows,
            cols,
            rank: 2,
            left: vec![0.3, 1.2, -0.8, 0.1, 1.5, -0.6],
            right: vec![-0.9, 0.7, 0.45, -0.2],
            scope,
        };

        // A global record at every position is one global change and names no read.
        let global = record(EditScope::Global(storage.clone()), PositionScope::every())
            .intervention_changes(&registry, &BTreeMap::new())
            .expect("a global record converts");
        assert_eq!(global, vec![carried(ParameterEditScope::Global)]);

        // A use-specific record on the transposed read still carries the stored
        // orientation: the runner edits the stored value, and the read transposes it.
        let middle = record(
            EditScope::UseSite(uses[1].clone()),
            PositionScope::declared(vec![2, 4]).expect("the positions are increasing"),
        )
        .intervention_changes(&registry, &labels)
        .expect("a use-specific record converts");
        assert_eq!(middle, vec![carried(at_read(1, Some(vec![2, 4])))]);

        // A global edit behind a key-value cache filled up to position 3 reaches every
        // read at positions 3.. only: one use-site change per read, in ordinal order.
        let cached = record(
            EditScope::Global(storage.clone()),
            PositionScope::declared(vec![3, 4, 5]).expect("the positions are increasing"),
        )
        .intervention_changes(&registry, &labels)
        .expect("a positioned global record converts");
        assert_eq!(
            cached,
            (0..TIES.len())
                .map(|ordinal| carried(at_read(ordinal, Some(vec![3, 4, 5]))))
                .collect::<Vec<_>>()
        );

        // The plan accepts every conversion, and only the global change needs its own
        // clean forward.
        let plan = |changes: Vec<InterventionChange>| {
            InterventionExperimentPlan::new(
                Vec::new(),
                Vec::new(),
                vec![InterventionExperiment {
                    unit: ExperimentUnit {
                        group: 0,
                        sequence: 0,
                        length: 6,
                    },
                    changes,
                    readouts: vec![Readout::Kl(KlPositions::Declared(vec![5]))],
                }],
                0,
                Some("full-forward:len6".to_string()),
            )
            .expect("the plan accepts the converted changes")
        };
        assert_eq!(plan(global).experiments()[0].clean_pass(), CleanPass::SeparateForward);
        assert_eq!(plan(middle).experiments()[0].clean_pass(), CleanPass::SameBatch);
        let cached_plan = plan(cached);
        assert_eq!(cached_plan.experiments()[0].clean_pass(), CleanPass::SameBatch);
        assert_eq!(cached_plan.experiments()[0].edited_positions(), vec![3, 4, 5]);

        // The root module's name is empty, so a read in its own forward, such as a tied
        // head applied through F.linear, is named by its op alone.
        let use_specific = record(EditScope::UseSite(uses[1].clone()), PositionScope::every());
        let mut in_root = labels.clone();
        in_root.insert(
            uses[1].clone(),
            ReadLabel {
                module: String::new(),
                op: "F.linear".to_string(),
            },
        );
        assert_eq!(
            use_specific.intervention_changes(&registry, &in_root),
            Ok(vec![carried(ParameterEditScope::UseSite {
                ordinal: 1,
                read_module: String::new(),
                read_op: "F.linear".to_string(),
                positions: None,
            })])
        );

        // A read discovery did not label, or labelled with an empty op, is refused.
        assert_eq!(
            use_specific.intervention_changes(&registry, &BTreeMap::new()),
            Err(OccurrenceError::UnlabelledRead(uses[1].clone()))
        );
        let mut unnamed_op = labels.clone();
        unnamed_op.insert(
            uses[1].clone(),
            ReadLabel {
                module: "layers.1".to_string(),
                op: String::new(),
            },
        );
        assert_eq!(
            use_specific.intervention_changes(&registry, &unnamed_op),
            Err(OccurrenceError::UnlabelledRead(uses[1].clone()))
        );
        let mut two_labels = labels.clone();
        two_labels.remove(&uses[2]);
        assert_eq!(
            record(
                EditScope::Global(storage.clone()),
                PositionScope::declared(vec![3]).expect("one position is increasing"),
            )
            .intervention_changes(&registry, &two_labels),
            Err(OccurrenceError::UnlabelledRead(uses[2].clone()))
        );

        // Three terms on a 3x2 tensor exceed what a declared change carries.
        let three_terms = ParameterEditRecord::new(
            &registry,
            EditScope::Global(storage.clone()),
            PositionScope::every(),
            FactoredEdit::new(
                array![[0.3, 1.2, 0.5], [-0.8, 0.1, -0.4], [1.5, -0.6, 0.2]],
                array![[-0.9, 0.7, 0.3], [0.45, -0.2, 0.6]],
            )
            .expect("the factors agree on the term count"),
        )
        .expect("the record is valid");
        assert_eq!(
            three_terms.intervention_changes(&registry, &labels),
            Err(OccurrenceError::RankExceedsShape {
                storage: storage.clone(),
                rank: 3,
                rows,
                cols,
            })
        );

        // An ordinal addresses a read only on the discovery pass of the record's registry.
        let (decode, ..) = registry_with_uses(
            &chain.weight,
            &[
                TieOrientation::Identity,
                TieOrientation::Transpose,
                TieOrientation::Identity,
                TieOrientation::Identity,
            ],
        );
        assert_eq!(
            use_specific.intervention_changes(&decode, &labels),
            Err(OccurrenceError::RegistryMismatch {
                record: registry.teacher_fingerprint(),
                registry: decode.teacher_fingerprint(),
            })
        );
    }

    #[test]
    fn records_refuse_empty_deltas_and_registries_of_another_forward_path() {
        let chain = Chain::rectangular();
        let (registry, storage, uses) = registry_with_uses(&chain.weight, &TIES);
        let scope = EditScope::UseSite(uses[0].clone());
        let accepted = ParameterEditRecord::new(
            &registry,
            scope.clone(),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("a one-term delta is accepted");
        assert_eq!(accepted.registry(), registry.teacher_fingerprint());
        let empty = FactoredEdit::new(Array2::zeros((3, 0)), Array2::zeros((2, 0)))
            .expect("zero terms have consistent factors");
        assert!(matches!(
            ParameterEditRecord::new(&registry, scope, PositionScope::every(), empty).err(),
            Some(OccurrenceError::InvalidDelta(_))
        ));

        // A decode path that reads the tensor once more registers a fourth use. The
        // shared ids still resolve there, so only the fingerprint tells the paths apart.
        let (decode, decode_storage, decode_uses) = registry_with_uses(
            &chain.weight,
            &[
                TieOrientation::Identity,
                TieOrientation::Transpose,
                TieOrientation::Identity,
                TieOrientation::Identity,
            ],
        );
        assert_eq!(decode_storage, storage);
        assert_eq!(decode_uses[..3], uses[..]);
        assert!(decode.resolve_use_site(&uses[0]).is_ok());
        let global = ParameterEditRecord::new(
            &registry,
            EditScope::Global(storage),
            PositionScope::every(),
            rank_one_delta(&chain),
        )
        .expect("the global record is valid");
        assert!(matches!(
            global.delta_read_at(&registry, &uses[0]),
            Ok(Some(..))
        ));
        assert_eq!(
            global.delta_read_at(&decode, &uses[0]).err(),
            Some(OccurrenceError::RegistryMismatch {
                record: registry.teacher_fingerprint(),
                registry: decode.teacher_fingerprint(),
            })
        );
    }

    #[test]
    fn position_scopes_are_canonical_bounded_and_model_a_filled_cache() {
        let every = PositionScope::every();
        assert_eq!(every.positions(), None);
        assert!(every.reaches(0) && every.reaches(1_000));
        assert!(every.check_within(1).is_ok());

        assert!(matches!(
            PositionScope::declared(Vec::new()),
            Err(OccurrenceError::InvalidPositions(_))
        ));
        assert!(matches!(
            PositionScope::declared(vec![2, 2]),
            Err(OccurrenceError::InvalidPositions(_))
        ));
        assert!(matches!(
            PositionScope::declared(vec![3, 1]),
            Err(OccurrenceError::InvalidPositions(_))
        ));

        // A cache filled through position 3 keeps the clean keys and values there, so
        // the edit of the key and value projections reaches positions 4.. only.
        let length = 7;
        let uncached = PositionScope::declared((4..length).collect()).expect("increasing");
        assert_eq!(uncached.positions(), Some(&[4, 5, 6][..]));
        assert!((0..4).all(|position| !uncached.reaches(position)));
        assert!((4..length).all(|position| uncached.reaches(position)));
        assert!(uncached.check_within(length).is_ok());
        assert!(matches!(
            uncached.check_within(6),
            Err(OccurrenceError::InvalidPositions(_))
        ));
    }
}
