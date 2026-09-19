//! Native lift: the tensor registry and the exact residual anchor.
//!
//! The registry names the teacher's parameters the way the executing framework
//! names them, the same `parameter` string an
//! `inference::intervention_shard::InterventionChange::ParameterEdit` carries. It
//! records each storage tensor's shape and a fingerprint of its values, the other
//! names of that storage, and every use site in the executed graph with the name
//! it reads and what it does with it: a linear map in a named orientation, or a
//! read of the stored values. A global edit of a storage tensor reaches every use
//! site that reads it under any name; a use-specific edit reaches one site.
//!
//! The anchor lifts components of one matrix-shaped storage tensor `Theta_*` of
//! shape `(rows, cols)`:
//!
//! ```text
//! Theta(m) = m_Delta Theta_* + sum_j beta_j B_j,   beta = sum_c (m_c - m_Delta) v_c,
//! ```
//!
//! where basis matrix `B_j` is block `j` of one `apply::FactoredEdit`. This equals
//! `sum_c m_c P_c + m_Delta (Theta_* - sum_c P_c)` with `P_c = sum_j v_cj B_j`, so
//! the residual `Theta_* - sum_c P_c` is carried exactly without being formed, and
//! the tensor depends on the mask only through the moment `beta`. The anchor
//! executes through the matrix-free kernels of `apply`, which never form an edited
//! tensor. Algebraic equality is not bitwise equality, so at the all-on mask it
//! executes `Theta_*` through the native product and never through the anchor
//! arithmetic. At a unit residual mask the edit `Theta(m) - Theta_*` is a factored
//! product, which a framework runner applies as a `ParameterEdit`.

use std::collections::BTreeMap;
use std::fmt;

use gam_linalg::utils::splitmix64_hash;
use gam_runtime::resource::Governed;
use ndarray::{Array1, Array2, ArrayView2, ArrayViewD};
use serde::{Deserialize, Serialize};

use super::apply::{ApplyError, FactorView, FactoredEdit, apply_anchored_linear, native_linear};
use super::field::CotangentTerm;
use super::occurrence::{OccurrenceError, ParameterEditRecord};
use crate::inference::intervention_shard::{ExperimentUnit, InterventionChange, ParameterEditScope};

/// The stable id of one named parameter: its name in the executing framework.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub String);

/// The stable id of one place in the executed graph that reads a parameter.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UseSiteId(pub String);

impl UseSiteId {
    /// The id of the `ordinal`-th read of `storage` in one forward pass, counted in
    /// execution order from zero: `{storage}#{ordinal}`. Repeated calls of a shared
    /// body read the same storage at different ordinals, so they stay distinct.
    pub fn read(storage: &TensorId, ordinal: usize) -> Self {
        Self(format!("{}#{ordinal}", storage.0))
    }
}

/// The orientation of the matrix a linear use multiplies by.
///
/// A linear use applies the map `y = x A^T` to its input rows `x`. The orientation
/// names `A`: the stored `Theta` or its transpose. It belongs to the map the consuming
/// operation applies, not to how the framework reached the tensor: `x @ W.t()`
/// multiplies by `A = W` and is an identity use, while `x @ W` is a transposed one.
/// The read width is `A`'s column count and the written width its row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TieOrientation {
    /// `A = Theta`: input rows of the stored column count, output rows of the stored
    /// row count.
    Identity,
    /// `A = Theta^T`: input rows of the stored row count, output rows of the stored
    /// column count.
    Transpose,
}

/// What a use site does with the parameter it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UseMap {
    /// The use multiplies its input rows by a matrix-shaped parameter, in the named
    /// orientation.
    Linear(TieOrientation),
    /// The use reads the stored values without applying them as a linear map: an index
    /// lookup, a bias or norm weight, an element-wise use. It has no map widths and no
    /// factored cotangent, and its gradient is taken with respect to the stored values.
    Stored,
}

/// A fingerprint of the registered storage values, names and use sites.
///
/// It folds 64-bit words through the SplitMix64 finalizer. Each step is a bijection
/// of the running state, so two equally long word streams that differ in exactly
/// one word never share a fingerprint; any other difference shares one only by a
/// collision of the mixer. It identifies a teacher; it certifies nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TeacherFingerprint(pub u64);

/// One registered storage tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageTensor {
    /// The row-major shape the values were registered in.
    pub shape: Vec<usize>,
    /// The fingerprint of the shape and every value's bit pattern.
    pub fingerprint: u64,
}

/// One registered use site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UseSite {
    /// The name the site reads: storage, or an alias of it.
    pub reads: TensorId,
    /// What the site does with the parameter.
    pub map: UseMap,
}

/// The storage a use site reads, and what it does with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedRead {
    /// The storage tensor.
    pub storage: TensorId,
    /// What the site does with the parameter.
    pub map: UseMap,
}

/// The teacher's parameter tensors, their names and their use sites.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TensorRegistry {
    storage: BTreeMap<TensorId, StorageTensor>,
    aliases: BTreeMap<TensorId, TensorId>,
    use_sites: BTreeMap<UseSiteId, UseSite>,
}

impl TensorRegistry {
    /// Registers the storage tensor `id` holding `values`.
    pub fn register_storage(
        &mut self,
        id: TensorId,
        values: ArrayViewD<'_, f64>,
    ) -> Result<(), LiftError> {
        self.refuse_taken(&id)?;
        let shape = values.shape().to_vec();
        let fingerprint = values_fingerprint(&shape, values.iter());
        self.storage.insert(id, StorageTensor { shape, fingerprint });
        Ok(())
    }

    /// Registers `alias` as a second name for the storage tensor `storage`.
    ///
    /// An alias names the same parameters, so it holds them as stored; a transposed
    /// multiplication belongs to the use site that performs it. An alias names storage
    /// directly, never another alias.
    pub fn register_alias(&mut self, alias: TensorId, storage: TensorId) -> Result<(), LiftError> {
        self.refuse_taken(&alias)?;
        if self.storage.contains_key(&storage) {
            self.aliases.insert(alias, storage);
            return Ok(());
        }
        Err(if self.aliases.contains_key(&storage) {
            LiftError::AliasOfAlias {
                alias: alias.0,
                target: storage.0,
            }
        } else {
            LiftError::UnknownTensor(storage.0)
        })
    }

    /// Registers the use site `site`, which reads the registered name `reads` and does
    /// `map` with it.
    pub fn register_use_site(
        &mut self,
        site: UseSiteId,
        reads: TensorId,
        map: UseMap,
    ) -> Result<(), LiftError> {
        if self.use_sites.contains_key(&site) {
            return Err(LiftError::DuplicateUseSite(site.0));
        }
        let storage = self.storage_of(&reads)?;
        if let UseMap::Linear(..) = map {
            let shape = &self.storage[storage].shape;
            if shape.len() != 2 {
                return Err(LiftError::NotAMatrix {
                    tensor: storage.0.clone(),
                    shape: shape.clone(),
                });
            }
        }
        self.use_sites.insert(site, UseSite { reads, map });
        Ok(())
    }

    /// The storage tensor, if `id` names one.
    pub fn storage(&self, id: &TensorId) -> Option<&StorageTensor> {
        self.storage.get(id)
    }

    /// Every storage tensor id, in id order.
    pub fn storage_ids(&self) -> impl Iterator<Item = &TensorId> + '_ {
        self.storage.keys()
    }

    /// The storage tensor a registered name reads.
    pub fn storage_of(&self, name: &TensorId) -> Result<&TensorId, LiftError> {
        if let Some(entry) = self.storage.get_key_value(name) {
            return Ok(entry.0);
        }
        self.aliases
            .get(name)
            .ok_or_else(|| LiftError::UnknownTensor(name.0.clone()))
    }

    /// The storage a registered use site reads, and what it does with it.
    pub fn resolve_use_site(&self, site: &UseSiteId) -> Result<ResolvedRead, LiftError> {
        let record = self
            .use_sites
            .get(site)
            .ok_or_else(|| LiftError::UnknownUseSite(site.0.clone()))?;
        Ok(ResolvedRead {
            storage: self.storage_of(&record.reads)?.clone(),
            map: record.map,
        })
    }

    /// Every use site that reads `storage` under any of its names, in id order: the
    /// sites a global edit of `storage` reaches.
    pub fn use_sites_of(&self, storage: &TensorId) -> Vec<&UseSiteId> {
        self.use_sites
            .iter()
            .filter(|entry| {
                self.storage_of(&entry.1.reads)
                    .map(|read| read == storage)
                    .unwrap_or(false)
            })
            .map(|entry| entry.0)
            .collect()
    }

    /// The fingerprint of everything registered. It does not depend on the order
    /// of registration.
    pub fn teacher_fingerprint(&self) -> TeacherFingerprint {
        let mut fold = WordFold::default();
        fold.absorb(self.storage.len() as u64);
        for (id, tensor) in &self.storage {
            fold.absorb_str(&id.0);
            fold.absorb(tensor.fingerprint);
        }
        fold.absorb(self.aliases.len() as u64);
        for (alias, storage) in &self.aliases {
            fold.absorb_str(&alias.0);
            fold.absorb_str(&storage.0);
        }
        fold.absorb(self.use_sites.len() as u64);
        for (site, record) in &self.use_sites {
            fold.absorb_str(&site.0);
            fold.absorb_str(&record.reads.0);
            fold.absorb(match record.map {
                UseMap::Linear(TieOrientation::Identity) => 0,
                UseMap::Linear(TieOrientation::Transpose) => 1,
                UseMap::Stored => 2,
            });
        }
        TeacherFingerprint(fold.0)
    }

    fn refuse_taken(&self, id: &TensorId) -> Result<(), LiftError> {
        if self.storage.contains_key(id) || self.aliases.contains_key(id) {
            Err(LiftError::DuplicateName(id.0.clone()))
        } else {
            Ok(())
        }
    }
}

/// The running state of a fingerprint: `state <- splitmix64_hash(state ^ word)`.
#[derive(Default)]
struct WordFold(u64);

impl WordFold {
    fn absorb(&mut self, word: u64) {
        self.0 = splitmix64_hash(self.0 ^ word);
    }

    fn absorb_str(&mut self, text: &str) {
        self.absorb(text.len() as u64);
        for chunk in text.as_bytes().chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            self.absorb(u64::from_le_bytes(bytes));
        }
    }
}

fn values_fingerprint<'a>(shape: &[usize], values: impl Iterator<Item = &'a f64>) -> u64 {
    let mut fold = WordFold::default();
    fold.absorb(shape.len() as u64);
    for &extent in shape {
        fold.absorb(extent as u64);
    }
    for value in values {
        fold.absorb(value.to_bits());
    }
    fold.0
}

/// The coefficient vectors `v_c` of the components in the basis.
#[derive(Clone, Debug, PartialEq)]
pub enum ComponentCoefficients {
    /// Component `c` is basis matrix `c`, `v_c = e_c`, without storing the identity.
    Basis,
    /// Row `c` of a `(components, basis matrices)` matrix is `v_c`.
    Dense(Array2<f64>),
}

/// A mask on the anchor: `residual` is `m_Delta` and `components[c]` is `m_c`.
///
/// Any finite values are admitted. The mask domain is an experiment declaration,
/// and binary, continuous and signed masks are all lifted exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct AnchorMask {
    /// The residual mask `m_Delta`.
    pub residual: f64,
    /// The component masks `m_c`.
    pub components: Vec<f64>,
}

impl AnchorMask {
    /// The mask that executes the teacher: every entry exactly one.
    pub fn all_on(components: usize) -> Self {
        Self {
            residual: 1.0,
            components: vec![1.0; components],
        }
    }

    /// Whether every entry is exactly one.
    pub fn is_all_on(&self) -> bool {
        self.residual == 1.0 && self.components.iter().all(|&mask| mask == 1.0)
    }
}

/// The exact residual anchor of one matrix-shaped storage tensor.
#[derive(Clone, Debug)]
pub struct ResidualAnchor<'a> {
    storage: TensorId,
    native: ArrayView2<'a, f64>,
    basis: FactoredEdit,
    block_ranks: Vec<usize>,
    coefficients: ComponentCoefficients,
}

impl<'a> ResidualAnchor<'a> {
    /// Anchors the basis matrices and component coefficients on the teacher values
    /// `native` of the registered storage tensor `storage`.
    ///
    /// `basis` holds every basis matrix as consecutive terms: basis matrix `j` is the
    /// next `block_ranks[j]` terms. Refuses values whose fingerprint is not the
    /// registered one, so an anchor never executes against a different teacher than
    /// the registry names.
    pub fn new(
        registry: &TensorRegistry,
        storage: TensorId,
        native: ArrayView2<'a, f64>,
        basis: FactoredEdit,
        block_ranks: Vec<usize>,
        coefficients: ComponentCoefficients,
    ) -> Result<Self, LiftError> {
        let Some(registered) = registry.storage(&storage) else {
            return Err(LiftError::UnknownTensor(storage.0));
        };
        let (rows, cols) = native.dim();
        if registered.shape != [rows, cols] {
            return Err(LiftError::NativeShape {
                tensor: storage.0,
                registered: registered.shape.clone(),
                native: (rows, cols),
            });
        }
        let found = values_fingerprint(&registered.shape, native.iter());
        if found != registered.fingerprint {
            return Err(LiftError::TeacherMismatch {
                tensor: storage.0,
                registered: registered.fingerprint,
                found,
            });
        }
        if basis.output_dim() != rows || basis.input_dim() != cols {
            return Err(LiftError::FactorShape {
                output_dim: basis.output_dim(),
                input_dim: basis.input_dim(),
                rows,
                cols,
            });
        }
        let ranked_terms: usize = block_ranks.iter().sum();
        if ranked_terms != basis.term_count() {
            return Err(LiftError::BlockRanks {
                ranked_terms,
                terms: basis.term_count(),
            });
        }
        if let ComponentCoefficients::Dense(vectors) = &coefficients {
            if vectors.ncols() != block_ranks.len() {
                return Err(LiftError::CoefficientWidth {
                    blocks: block_ranks.len(),
                    coefficients: vectors.ncols(),
                });
            }
            if !vectors.iter().all(|value| value.is_finite()) {
                return Err(LiftError::NonFiniteCoefficients);
            }
        }
        Ok(Self {
            storage,
            native,
            basis,
            block_ranks,
            coefficients,
        })
    }

    /// The storage tensor this anchor lifts.
    pub fn storage(&self) -> &TensorId {
        &self.storage
    }

    /// The number of components `C`.
    pub fn component_count(&self) -> usize {
        match &self.coefficients {
            ComponentCoefficients::Basis => self.block_ranks.len(),
            ComponentCoefficients::Dense(vectors) => vectors.nrows(),
        }
    }

    /// The moment `beta = sum_c (m_c - m_Delta) v_c`, one entry per basis matrix.
    pub fn moment(&self, mask: &AnchorMask) -> Result<Vec<f64>, LiftError> {
        self.check_mask(mask)?;
        Ok(self.moment_of_valid_mask(mask))
    }

    /// The teacher's own product of the storage tensor on input rows: `x Theta_*^T`
    /// for an identity use and `x Theta_*` for a transposed one.
    pub fn native_apply(
        &self,
        inputs: ArrayView2<'_, f64>,
        orientation: TieOrientation,
    ) -> Result<Governed<Array2<f64>>, LiftError> {
        Ok(match orientation {
            TieOrientation::Identity => native_linear(self.native, inputs)?,
            TieOrientation::Transpose => native_linear(self.native.t(), inputs)?,
        })
    }

    /// `Theta(m)` on the input rows of a linear use in `orientation`, without forming
    /// `Theta(m)`.
    pub fn apply(
        &self,
        mask: &AnchorMask,
        inputs: ArrayView2<'_, f64>,
        orientation: TieOrientation,
    ) -> Result<Governed<Array2<f64>>, LiftError> {
        self.check_mask(mask)?;
        if mask.is_all_on() {
            return self.native_apply(inputs, orientation);
        }
        let term_scales = self.term_scales(&self.moment_of_valid_mask(mask));
        Ok(match orientation {
            TieOrientation::Identity => apply_anchored_linear(
                self.native,
                mask.residual,
                self.basis.view(),
                term_scales.view(),
                inputs,
            )?,
            TieOrientation::Transpose => apply_anchored_linear(
                self.native.t(),
                mask.residual,
                FactorView::new(self.basis.right(), self.basis.left())?,
                term_scales.view(),
                inputs,
            )?,
        })
    }

    /// The global edit `Theta(m) - Theta_*` as the declared change a framework runner
    /// applies to the stored parameter for the whole pass, or `None` when the mask
    /// leaves the teacher unedited.
    ///
    /// The edit is `(m_Delta - 1) Theta_* + sum_j beta_j B_j`. Only at `m_Delta = 1`
    /// is it the product `left right^T` a parameter edit carries: the basis terms
    /// with a nonzero scale, each left column scaled by its basis matrix's moment.
    pub fn parameter_edit(
        &self,
        mask: &AnchorMask,
    ) -> Result<Option<InterventionChange>, LiftError> {
        self.check_mask(mask)?;
        if mask.residual != 1.0 {
            return Err(LiftError::ResidualScaleIsNotAProduct {
                residual: mask.residual,
            });
        }
        let term_scales = self.term_scales(&self.moment_of_valid_mask(mask));
        let active: Vec<usize> = term_scales
            .iter()
            .enumerate()
            .filter(|term| *term.1 != 0.0)
            .map(|term| term.0)
            .collect();
        let rank = active.len();
        if rank == 0 {
            return Ok(None);
        }
        let (rows, cols) = self.native.dim();
        if rank > rows.min(cols) {
            return Err(LiftError::EditRankExceedsShape { rank, rows, cols });
        }
        let mut left = Vec::with_capacity(rows * rank);
        for row in self.basis.left().outer_iter() {
            left.extend(active.iter().map(|&term| term_scales[term] * row[term]));
        }
        let mut right = Vec::with_capacity(cols * rank);
        for row in self.basis.right().outer_iter() {
            right.extend(active.iter().map(|&term| row[term]));
        }
        Ok(Some(InterventionChange::ParameterEdit {
            parameter: self.storage.0.clone(),
            rows,
            cols,
            rank,
            left,
            right,
            scope: ParameterEditScope::Global,
        }))
    }

    /// Each basis term's scale: its basis matrix's moment, repeated over the block.
    fn term_scales(&self, beta: &[f64]) -> Array1<f64> {
        self.block_ranks
            .iter()
            .zip(beta)
            .flat_map(|block| std::iter::repeat(*block.1).take(*block.0))
            .collect()
    }

    fn moment_of_valid_mask(&self, mask: &AnchorMask) -> Vec<f64> {
        match &self.coefficients {
            ComponentCoefficients::Basis => mask
                .components
                .iter()
                .map(|&component| component - mask.residual)
                .collect(),
            ComponentCoefficients::Dense(vectors) => {
                let mut beta = vec![0.0; self.block_ranks.len()];
                for (vector, &component) in vectors.outer_iter().zip(&mask.components) {
                    let anchored = component - mask.residual;
                    if anchored == 0.0 {
                        continue;
                    }
                    for (entry, &coefficient) in beta.iter_mut().zip(vector.iter()) {
                        *entry += anchored * coefficient;
                    }
                }
                beta
            }
        }
    }

    fn check_mask(&self, mask: &AnchorMask) -> Result<(), LiftError> {
        let components = self.component_count();
        if mask.components.len() != components {
            return Err(LiftError::MaskLength {
                components,
                mask: mask.components.len(),
            });
        }
        if !mask.residual.is_finite() || !mask.components.iter().all(|value| value.is_finite()) {
            return Err(LiftError::NonFiniteMask);
        }
        Ok(())
    }
}

/// A refusal of the registry or the anchor.
#[derive(Clone, Debug, PartialEq)]
pub enum LiftError {
    /// The name is already registered, as storage or as an alias.
    DuplicateName(String),
    /// The use site is already registered.
    DuplicateUseSite(String),
    /// No storage tensor or alias has this name.
    UnknownTensor(String),
    /// No use site has this id.
    UnknownUseSite(String),
    /// An alias must name storage, not another alias.
    AliasOfAlias { alias: String, target: String },
    /// A linear use needs a matrix-shaped storage tensor.
    NotAMatrix { tensor: String, shape: Vec<usize> },
    /// The anchor's teacher values do not have the registered shape.
    NativeShape {
        tensor: String,
        registered: Vec<usize>,
        native: (usize, usize),
    },
    /// The anchor's teacher values are not the registered values.
    TeacherMismatch {
        tensor: String,
        registered: u64,
        found: u64,
    },
    /// The basis factors do not form matrices of the storage tensor's shape.
    FactorShape {
        output_dim: usize,
        input_dim: usize,
        rows: usize,
        cols: usize,
    },
    /// The block ranks do not partition the basis terms.
    BlockRanks { ranked_terms: usize, terms: usize },
    /// The coefficient vectors do not have one entry per basis matrix.
    CoefficientWidth { blocks: usize, coefficients: usize },
    /// A coefficient vector holds a non-finite value.
    NonFiniteCoefficients,
    /// The mask does not have one entry per component.
    MaskLength { components: usize, mask: usize },
    /// The mask holds a non-finite value.
    NonFiniteMask,
    /// A residual mask other than one scales the teacher tensor, which no factored
    /// product carries.
    ResidualScaleIsNotAProduct { residual: f64 },
    /// The edit has more terms than a parameter edit of this shape admits.
    EditRankExceedsShape { rank: usize, rows: usize, cols: usize },
    /// The matrix-free kernel refused the call.
    Apply(ApplyError),
}

impl From<ApplyError> for LiftError {
    fn from(err: ApplyError) -> Self {
        Self::Apply(err)
    }
}

impl fmt::Display for LiftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(formatter, "tensor name {name} is already registered"),
            Self::DuplicateUseSite(site) => write!(formatter, "use site {site} is already registered"),
            Self::UnknownTensor(name) => write!(formatter, "no registered tensor is named {name}"),
            Self::UnknownUseSite(site) => write!(formatter, "no registered use site is named {site}"),
            Self::AliasOfAlias { alias, target } => write!(
                formatter,
                "alias {alias} names the alias {target}; an alias must name storage"
            ),
            Self::NotAMatrix { tensor, shape } => write!(
                formatter,
                "a linear use needs a matrix, but {tensor} has shape {shape:?}"
            ),
            Self::NativeShape {
                tensor,
                registered,
                native,
            } => write!(
                formatter,
                "{tensor} is registered with shape {registered:?} but the anchor holds {native:?}"
            ),
            Self::TeacherMismatch {
                tensor,
                registered,
                found,
            } => write!(
                formatter,
                "{tensor} values fingerprint to {found:#018x}, not the registered {registered:#018x}"
            ),
            Self::FactorShape {
                output_dim,
                input_dim,
                rows,
                cols,
            } => write!(
                formatter,
                "basis factors form ({output_dim}, {input_dim}) matrices, not ({rows}, {cols})"
            ),
            Self::BlockRanks {
                ranked_terms,
                terms,
            } => write!(
                formatter,
                "block ranks cover {ranked_terms} terms of a basis with {terms}"
            ),
            Self::CoefficientWidth {
                blocks,
                coefficients,
            } => write!(
                formatter,
                "coefficient vectors have {coefficients} entries for {blocks} basis matrices"
            ),
            Self::NonFiniteCoefficients => write!(formatter, "a coefficient vector holds a non-finite value"),
            Self::MaskLength { components, mask } => write!(
                formatter,
                "the mask has {mask} component entries for {components} components"
            ),
            Self::NonFiniteMask => write!(formatter, "the mask holds a non-finite value"),
            Self::ResidualScaleIsNotAProduct { residual } => write!(
                formatter,
                "a residual mask of {residual} scales the teacher tensor, which no parameter edit carries"
            ),
            Self::EditRankExceedsShape { rank, rows, cols } => write!(
                formatter,
                "the edit has {rank} terms, more than a ({rows}, {cols}) parameter edit admits"
            ),
            Self::Apply(err) => write!(formatter, "{err}"),
        }
    }
}

impl std::error::Error for LiftError {}

/// One forward pass of the teacher under typed parameter edits.
#[derive(Clone, Debug)]
pub struct ParameterExperiment {
    /// The token unit the pass executes.
    pub unit: ExperimentUnit,
    /// The edits applied together in this pass.
    pub edits: Vec<ParameterEditRecord>,
    /// The declared readouts. An executor returns one block per readout, in order.
    pub readouts: Vec<ParameterReadout>,
}

/// One declared readout of a pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParameterReadout {
    /// The model output's rows at declared positions: `(positions, output width)`.
    Output { positions: Vec<usize> },
    /// The rows a linear use multiplied, one per position of the unit:
    /// `(length, width read)`.
    UseSiteInput(UseSiteId),
    /// The rows a linear use wrote, one per position of the unit:
    /// `(length, width written)`.
    UseSiteOutput(UseSiteId),
}

/// The forward-error status of executed readouts.
#[derive(Clone, Debug)]
pub enum ForwardRoundoff {
    /// A derived entrywise band for every readout block, in the blocks' shapes.
    Derived(Vec<Array2<f64>>),
    /// No derived band, as from an external executor that supplies none. A
    /// certificate built on these readouts is unresolved.
    Unresolved,
}

/// The executed readouts of one experiment.
#[derive(Clone, Debug)]
pub struct ExecutedExperiment {
    /// One block per declared readout, in order.
    pub readouts: Vec<Array2<f64>>,
    /// Their forward-error status.
    pub roundoff: ForwardRoundoff,
}

/// A forward executor of the teacher under typed parameter edits.
///
/// The Rust lift implements it natively, and the Python surface wraps an external
/// framework object as the same trait. An executor returns executed rows only:
/// divergences, cotangent contractions and bounds are computed in Rust from them.
pub trait ParameterExecutor {
    /// Executes every experiment. A batch of one is the scalar case.
    fn forward(
        &self,
        experiments: &[ParameterExperiment],
    ) -> Result<Vec<ExecutedExperiment>, ExecutorError>;

    /// Pulls each experiment's readout cotangents (one block per readout, in the
    /// readout's shape) back to the cotangent term of every use site its edits reach,
    /// in that use's orientation. `occurrence::edit_cotangent` sums the terms through
    /// each use's tie.
    fn vjp(
        &self,
        experiments: &[ParameterExperiment],
        cotangents: &[Vec<Array2<f64>>],
    ) -> Result<Vec<Vec<(UseSiteId, CotangentTerm)>>, ExecutorError>;
}

impl ParameterExperiment {
    /// The shape each readout block must have: `(positions, output_width)` for the
    /// model output, and `(length, width)` for a linear use's rows, with the width read
    /// or written in the use's orientation.
    pub fn readout_shapes(
        &self,
        registry: &TensorRegistry,
        output_width: usize,
    ) -> Result<Vec<(usize, usize)>, ExecutorError> {
        let mut shapes = Vec::with_capacity(self.readouts.len());
        for (index, readout) in self.readouts.iter().enumerate() {
            let shape = match readout {
                ParameterReadout::Output { positions } => {
                    if positions.windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err(ExecutorError::UnorderedPositions { readout: index });
                    }
                    if let Some(&position) = positions.iter().find(|&&position| position >= self.unit.length) {
                        return Err(ExecutorError::Position {
                            readout: index,
                            position,
                            length: self.unit.length,
                        });
                    }
                    (positions.len(), output_width)
                }
                ParameterReadout::UseSiteInput(site) | ParameterReadout::UseSiteOutput(site) => {
                    let read = registry.resolve_use_site(site)?;
                    let UseMap::Linear(orientation) = read.map else {
                        return Err(ExecutorError::NotALinearUse {
                            readout: index,
                            site: site.clone(),
                        });
                    };
                    let stored = registry
                        .storage(&read.storage)
                        .map(|tensor| tensor.shape.clone())
                        .unwrap_or_default();
                    let [rows, cols] = stored[..] else {
                        return Err(ExecutorError::Lift(LiftError::NotAMatrix {
                            tensor: read.storage.0,
                            shape: stored,
                        }));
                    };
                    let (written, width_read) = match orientation {
                        TieOrientation::Identity => (rows, cols),
                        TieOrientation::Transpose => (cols, rows),
                    };
                    let width = if matches!(readout, ParameterReadout::UseSiteInput(..)) {
                        width_read
                    } else {
                        written
                    };
                    (self.unit.length, width)
                }
            };
            shapes.push(shape);
        }
        Ok(shapes)
    }
}

impl ExecutedExperiment {
    /// Refuses executed readouts unless there is one finite block of the declared
    /// shape per readout, and, when a band is derived, one finite non-negative band of
    /// the same shape per block. `shapes` comes from
    /// [`ParameterExperiment::readout_shapes`].
    pub fn check(&self, shapes: &[(usize, usize)]) -> Result<(), ExecutorError> {
        check_blocks("readout block", &self.readouts, shapes)?;
        if let ForwardRoundoff::Derived(bands) = &self.roundoff {
            check_blocks("roundoff band", bands, shapes)?;
            if let Some(index) = bands.iter().position(|band| band.iter().any(|&value| value < 0.0)) {
                return Err(ExecutorError::NegativeBand { readout: index });
            }
        }
        Ok(())
    }
}

fn check_blocks(
    operand: &'static str,
    blocks: &[Array2<f64>],
    shapes: &[(usize, usize)],
) -> Result<(), ExecutorError> {
    if blocks.len() != shapes.len() {
        return Err(ExecutorError::ReadoutCount {
            operand,
            declared: shapes.len(),
            returned: blocks.len(),
        });
    }
    for (index, (block, &expected)) in blocks.iter().zip(shapes).enumerate() {
        if block.dim() != expected {
            return Err(ExecutorError::Shape {
                readout: index,
                operand,
                expected,
                found: block.dim(),
            });
        }
        if !block.iter().all(|value| value.is_finite()) {
            return Err(ExecutorError::NonFinite {
                readout: index,
                operand,
            });
        }
    }
    Ok(())
}

/// A refusal at the executor boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecutorError {
    /// The registry refused a name or a use site.
    Lift(LiftError),
    /// An edit record refused.
    Occurrence(OccurrenceError),
    /// An output readout declares a position outside the unit.
    Position {
        readout: usize,
        position: usize,
        length: usize,
    },
    /// An output readout's positions are not strictly increasing.
    UnorderedPositions { readout: usize },
    /// A use-site readout names a stored read, which applies no linear map and so has
    /// no input or output rows.
    NotALinearUse { readout: usize, site: UseSiteId },
    /// The executor returned a different number of blocks than readouts were declared.
    ReadoutCount {
        operand: &'static str,
        declared: usize,
        returned: usize,
    },
    /// A block does not have its declared shape.
    Shape {
        readout: usize,
        operand: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A block holds a non-finite value.
    NonFinite {
        readout: usize,
        operand: &'static str,
    },
    /// A derived roundoff band holds a negative value.
    NegativeBand { readout: usize },
    /// An external executor's refusal, carried verbatim.
    External(String),
}

impl From<LiftError> for ExecutorError {
    fn from(err: LiftError) -> Self {
        Self::Lift(err)
    }
}

impl From<OccurrenceError> for ExecutorError {
    fn from(err: OccurrenceError) -> Self {
        Self::Occurrence(err)
    }
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lift(err) => write!(formatter, "{err}"),
            Self::Occurrence(err) => write!(formatter, "{err}"),
            Self::Position {
                readout,
                position,
                length,
            } => write!(
                formatter,
                "readout {readout} declares position {position} in a unit of length {length}"
            ),
            Self::UnorderedPositions { readout } => write!(
                formatter,
                "readout {readout} declares positions that are not strictly increasing"
            ),
            Self::NotALinearUse { readout, site } => write!(
                formatter,
                "readout {readout} names use site {}, which applies no linear map and has no rows",
                site.0
            ),
            Self::ReadoutCount {
                operand,
                declared,
                returned,
            } => write!(
                formatter,
                "the executor returned {returned} {operand}s for {declared} declared readouts"
            ),
            Self::Shape {
                readout,
                operand,
                expected,
                found,
            } => write!(
                formatter,
                "{operand} {readout} has shape {found:?}; the readout declares {expected:?}"
            ),
            Self::NonFinite { readout, operand } => {
                write!(formatter, "{operand} {readout} holds a non-finite value")
            }
            Self::NegativeBand { readout } => {
                write!(formatter, "roundoff band {readout} holds a negative value")
            }
            Self::External(message) => write!(formatter, "external executor: {message}"),
        }
    }
}

impl std::error::Error for ExecutorError {}

#[cfg(test)]
mod executor_tests {
    use super::*;

    /// A registry holding a `(5, 3)` weight read by an identity and a transposed linear
    /// use, and a length-5 bias read as stored.
    fn registry_with_uses() -> (TensorRegistry, UseSiteId, UseSiteId, UseSiteId) {
        let mut registry = TensorRegistry::default();
        let weight = TensorId("mlp.weight".to_string());
        let bias = TensorId("mlp.bias".to_string());
        registry
            .register_storage(weight.clone(), Array2::<f64>::zeros((5, 3)).view().into_dyn())
            .expect("fresh name");
        registry
            .register_storage(bias.clone(), Array1::<f64>::zeros(5).view().into_dyn())
            .expect("fresh name");
        let identity = UseSiteId::read(&weight, 0);
        let transposed = UseSiteId::read(&weight, 1);
        let stored = UseSiteId::read(&bias, 0);
        registry
            .register_use_site(identity.clone(), weight.clone(), UseMap::Linear(TieOrientation::Identity))
            .expect("fresh site");
        registry
            .register_use_site(transposed.clone(), weight, UseMap::Linear(TieOrientation::Transpose))
            .expect("fresh site");
        registry
            .register_use_site(stored.clone(), bias, UseMap::Stored)
            .expect("fresh site");
        (registry, identity, transposed, stored)
    }

    fn experiment(readouts: Vec<ParameterReadout>) -> ParameterExperiment {
        ParameterExperiment {
            unit: ExperimentUnit {
                group: 0,
                sequence: 0,
                length: 4,
            },
            edits: Vec::new(),
            readouts,
        }
    }

    #[test]
    fn readout_shapes_follow_the_map_orientation_and_refuse_readouts_without_rows() {
        let (registry, identity, transposed, stored) = registry_with_uses();
        let declared = experiment(vec![
            ParameterReadout::Output { positions: vec![0, 3] },
            ParameterReadout::UseSiteInput(identity.clone()),
            ParameterReadout::UseSiteOutput(identity),
            ParameterReadout::UseSiteInput(transposed.clone()),
            ParameterReadout::UseSiteOutput(transposed),
        ]);
        assert_eq!(
            declared.readout_shapes(&registry, 11),
            Ok(vec![(2, 11), (4, 3), (4, 5), (4, 5), (4, 3)])
        );

        // Each refusal fires on its bad input; the declared experiment above is the good one.
        assert_eq!(
            experiment(vec![ParameterReadout::Output { positions: vec![4] }]).readout_shapes(&registry, 11),
            Err(ExecutorError::Position { readout: 0, position: 4, length: 4 })
        );
        assert_eq!(
            experiment(vec![ParameterReadout::Output { positions: vec![2, 2] }]).readout_shapes(&registry, 11),
            Err(ExecutorError::UnorderedPositions { readout: 0 })
        );
        assert_eq!(
            experiment(vec![ParameterReadout::UseSiteInput(stored.clone())]).readout_shapes(&registry, 11),
            Err(ExecutorError::NotALinearUse { readout: 0, site: stored })
        );
        assert!(matches!(
            experiment(vec![ParameterReadout::UseSiteOutput(UseSiteId("missing#0".to_string()))])
                .readout_shapes(&registry, 11),
            Err(ExecutorError::Lift(LiftError::UnknownUseSite(..)))
        ));
    }

    #[test]
    fn executed_readouts_are_refused_unless_every_block_and_band_has_its_declared_shape() {
        let shapes = vec![(2, 11), (4, 3)];
        let blocks = || vec![Array2::<f64>::zeros((2, 11)), Array2::<f64>::zeros((4, 3))];
        assert_eq!(
            ExecutedExperiment { readouts: blocks(), roundoff: ForwardRoundoff::Derived(blocks()) }.check(&shapes),
            Ok(())
        );
        assert_eq!(
            ExecutedExperiment { readouts: blocks(), roundoff: ForwardRoundoff::Unresolved }.check(&shapes),
            Ok(())
        );

        // Positive controls: each malformed return is refused, naming what is wrong.
        let mut wrong_block = blocks();
        wrong_block[1] = Array2::zeros((4, 2));
        assert_eq!(
            ExecutedExperiment { readouts: wrong_block, roundoff: ForwardRoundoff::Unresolved }.check(&shapes),
            Err(ExecutorError::Shape { readout: 1, operand: "readout block", expected: (4, 3), found: (4, 2) })
        );
        assert_eq!(
            ExecutedExperiment { readouts: blocks()[..1].to_vec(), roundoff: ForwardRoundoff::Unresolved }
                .check(&shapes),
            Err(ExecutorError::ReadoutCount { operand: "readout block", declared: 2, returned: 1 })
        );
        let mut wrong_band = blocks();
        wrong_band[0] = Array2::zeros((3, 11));
        assert_eq!(
            ExecutedExperiment { readouts: blocks(), roundoff: ForwardRoundoff::Derived(wrong_band) }.check(&shapes),
            Err(ExecutorError::Shape { readout: 0, operand: "roundoff band", expected: (2, 11), found: (3, 11) })
        );
        let mut nan_block = blocks();
        nan_block[0][[1, 4]] = f64::NAN;
        assert_eq!(
            ExecutedExperiment { readouts: nan_block, roundoff: ForwardRoundoff::Unresolved }.check(&shapes),
            Err(ExecutorError::NonFinite { readout: 0, operand: "readout block" })
        );
        let mut negative_band = blocks();
        negative_band[1][[0, 0]] = -1.0;
        assert_eq!(
            ExecutedExperiment { readouts: blocks(), roundoff: ForwardRoundoff::Derived(negative_band) }.check(&shapes),
            Err(ExecutorError::NegativeBand { readout: 1 })
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::accumulation_growth;
    use gam_linalg::utils::splitmix64;
    use ndarray::{Axis, concatenate};

    fn uniform(state: &mut u64) -> f64 {
        (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn random_matrix(rows: usize, cols: usize, state: &mut u64) -> Array2<f64> {
        let values: Vec<f64> = std::iter::repeat_with(|| uniform(state))
            .take(rows * cols)
            .collect();
        Array2::from_shape_vec((rows, cols), values).expect("row-major values fill the shape")
    }

    /// Basis matrices `B_j = L_j R_j^T` as `(L_j, R_j)` blocks.
    fn random_blocks(
        rows: usize,
        cols: usize,
        ranks: &[usize],
        state: &mut u64,
    ) -> Vec<(Array2<f64>, Array2<f64>)> {
        ranks
            .iter()
            .map(|&rank| (random_matrix(rows, rank, state), random_matrix(cols, rank, state)))
            .collect()
    }

    /// The blocks as consecutive terms of one edit, and their ranks.
    fn concatenated(blocks: &[(Array2<f64>, Array2<f64>)]) -> (FactoredEdit, Vec<usize>) {
        let lefts: Vec<ArrayView2<'_, f64>> = blocks.iter().map(|block| block.0.view()).collect();
        let rights: Vec<ArrayView2<'_, f64>> = blocks.iter().map(|block| block.1.view()).collect();
        let edit = FactoredEdit::new(
            concatenate(Axis(1), &lefts).expect("blocks share output rows"),
            concatenate(Axis(1), &rights).expect("blocks share input rows"),
        )
        .expect("finite factors of matching rank");
        (edit, blocks.iter().map(|block| block.0.ncols()).collect())
    }

    fn registry_with(id: &str, native: &Array2<f64>) -> TensorRegistry {
        let mut registry = TensorRegistry::default();
        registry
            .register_storage(TensorId(id.to_string()), native.view().into_dyn())
            .expect("a fresh name registers");
        registry
    }

    /// `x M^T` for an identity use and `x M` for a transposed one.
    fn read(inputs: &Array2<f64>, matrix: &Array2<f64>, orientation: TieOrientation) -> Array2<f64> {
        match orientation {
            TieOrientation::Identity => inputs.dot(&matrix.t()),
            TieOrientation::Transpose => inputs.dot(matrix),
        }
    }

    /// `m_Delta Theta_* + sum_c (m_c - m_Delta) sum_j v_cj L_j R_j^T`, formed entry by
    /// entry: the edited tensor a framework holds after writing the edit.
    fn edited_tensor(
        native: &Array2<f64>,
        blocks: &[(Array2<f64>, Array2<f64>)],
        vectors: &Array2<f64>,
        mask: &AnchorMask,
    ) -> Array2<f64> {
        let mut edited = native * mask.residual;
        for (vector, &component) in vectors.outer_iter().zip(&mask.components) {
            for (block, &coefficient) in blocks.iter().zip(vector.iter()) {
                edited += &(block.0.dot(&block.1.t()) * ((component - mask.residual) * coefficient));
            }
        }
        edited
    }

    /// `sum_c m_c P_c`: the components alone, with the residual dropped.
    fn components_only(
        blocks: &[(Array2<f64>, Array2<f64>)],
        vectors: &Array2<f64>,
        mask: &AnchorMask,
    ) -> Array2<f64> {
        let mut total = Array2::zeros((blocks[0].0.nrows(), blocks[0].1.nrows()));
        for (vector, &component) in vectors.outer_iter().zip(&mask.components) {
            for (block, &coefficient) in blocks.iter().zip(vector.iter()) {
                total += &(block.0.dot(&block.1.t()) * (component * coefficient));
            }
        }
        total
    }

    /// The edited tensor's polynomial with every leaf replaced by its magnitude and
    /// every difference `m_c - m_Delta` by `|m_c| + |m_Delta|`: entrywise, the absolute
    /// sum of its monomials.
    fn absolute_edit_magnitude(
        native: &Array2<f64>,
        blocks: &[(Array2<f64>, Array2<f64>)],
        vectors: &Array2<f64>,
        mask: &AnchorMask,
    ) -> Array2<f64> {
        let mut magnitude = native.mapv(f64::abs) * mask.residual.abs();
        for (vector, &component) in vectors.outer_iter().zip(&mask.components) {
            for (block, &coefficient) in blocks.iter().zip(vector.iter()) {
                let product = block.0.mapv(f64::abs).dot(&block.1.mapv(f64::abs).t());
                magnitude += &(product * ((component.abs() + mask.residual.abs()) * coefficient.abs()));
            }
        }
        magnitude
    }

    /// One mask and orientation of the anchor-versus-edited-tensor comparison.
    struct Agreement {
        case: String,
        /// The largest `|anchor - edited| / band` over the outputs.
        worst_ratio: f64,
        /// Whether dropping the residual leaves the band somewhere; `None` when the
        /// residual mask is zero and there is no residual to drop.
        residual_visible: Option<bool>,
    }

    fn anchor_agreement(
        native: &Array2<f64>,
        blocks: &[(Array2<f64>, Array2<f64>)],
        vectors: &Array2<f64>,
        coefficients: ComponentCoefficients,
        masks: &[AnchorMask],
        state: &mut u64,
    ) -> Vec<Agreement> {
        let registry = registry_with("layer.weight", native);
        let (basis, block_ranks) = concatenated(blocks);
        let terms = basis.term_count();
        let anchor = ResidualAnchor::new(
            &registry,
            TensorId("layer.weight".to_string()),
            native.view(),
            basis,
            block_ranks,
            coefficients,
        )
        .expect("the registered teacher anchors");
        let components = vectors.nrows();
        let widest_rank = blocks.iter().map(|block| block.0.ncols()).max().unwrap_or(0);
        let mut agreements = Vec::new();
        for orientation in [TieOrientation::Identity, TieOrientation::Transpose] {
            let width = match orientation {
                TieOrientation::Identity => native.ncols(),
                TieOrientation::Transpose => native.nrows(),
            };
            let inputs = random_matrix(3, width, state);
            // Rounded operations on the deepest root-to-leaf path (Higham, ASNA §3.1),
            // whatever order a kernel sums in. The anchor forms a term scale (a
            // difference, a product and `C` additions) or a projection `x r_k` (`width`
            // operations), scales once, contracts over the `T` terms and adds the
            // native part once. The edited tensor forms each `L_j R_j^T` entry (`r_j`
            // operations) or the weight (a difference and a product), scales once, adds
            // `C J` terms and then reads `x` (`width` operations).
            let anchor_depth = (components + 2).max(width) + 1 + terms + 1;
            let edited_depth = (widest_rank + 1).max(3) + components * blocks.len() + width;
            let growth = accumulation_growth(anchor_depth) + accumulation_growth(edited_depth);
            for mask in masks {
                let lifted = anchor
                    .apply(mask, inputs.view(), orientation)
                    .expect("a valid mask applies");
                let reference = read(&inputs, &edited_tensor(native, blocks, vectors, mask), orientation);
                let magnitude = absolute_edit_magnitude(native, blocks, vectors, mask);
                let band = read(&inputs.mapv(f64::abs), &magnitude, orientation) * growth;
                let worst_ratio = lifted
                    .iter()
                    .zip(reference.iter())
                    .zip(band.iter())
                    .map(|((a, b), bound)| (a - b).abs() / bound)
                    .fold(0.0_f64, f64::max);
                let residual_visible = (mask.residual != 0.0).then(|| {
                    let dropped = read(&inputs, &components_only(blocks, vectors, mask), orientation);
                    dropped
                        .iter()
                        .zip(reference.iter())
                        .zip(band.iter())
                        .any(|((a, b), bound)| (a - b).abs() > *bound)
                });
                agreements.push(Agreement {
                    case: format!("{orientation:?} {mask:?}"),
                    worst_ratio,
                    residual_visible,
                });
            }
        }
        agreements
    }

    #[test]
    fn the_anchor_equals_the_edited_tensor_for_continuous_binary_and_signed_masks() {
        let mut state = 0x2951_0001_u64;
        let native = random_matrix(5, 4, &mut state);
        let blocks = random_blocks(5, 4, &[1, 2, 3], &mut state);
        let vectors = random_matrix(4, 3, &mut state);
        let dense_masks = [
            AnchorMask { residual: 0.37, components: vec![0.12, 0.83, 0.5, 0.29] },
            AnchorMask { residual: 1.0, components: vec![1.0, 0.0, 0.0, 1.0] },
            AnchorMask { residual: 0.0, components: vec![0.0, 1.0, 1.0, 0.0] },
            AnchorMask { residual: -0.4, components: vec![-0.7, 1.9, 0.0, -2.3] },
        ];
        let mut agreements = anchor_agreement(
            &native,
            &blocks,
            &vectors,
            ComponentCoefficients::Dense(vectors.clone()),
            &dense_masks,
            &mut state,
        );
        let identity = Array2::eye(3);
        let basis_masks = [
            AnchorMask { residual: 0.61, components: vec![0.05, 0.9, 0.33] },
            AnchorMask { residual: 1.0, components: vec![0.0, 1.0, 0.0] },
            AnchorMask { residual: 1.3, components: vec![-1.1, 0.0, 2.2] },
        ];
        agreements.extend(anchor_agreement(
            &native,
            &blocks,
            &identity,
            ComponentCoefficients::Basis,
            &basis_masks,
            &mut state,
        ));

        assert_eq!(agreements.len(), 2 * (dense_masks.len() + basis_masks.len()));
        for agreement in &agreements {
            assert!(
                agreement.worst_ratio <= 1.0,
                "{}: the anchor departs from the edited tensor by {} times the roundoff band",
                agreement.case,
                agreement.worst_ratio
            );
            // Positive control: dropping the residual is the error the anchor exists
            // to avoid, and this comparison sees it wherever there is a residual.
            assert_ne!(
                agreement.residual_visible,
                Some(false),
                "{}: a residual-free lift stays inside the band, so the fixture cannot see the residual",
                agreement.case
            );
        }
        assert!(agreements.iter().any(|agreement| agreement.residual_visible == Some(true)));
    }

    #[test]
    fn the_all_on_mask_executes_the_teacher_bit_for_bit() {
        let mut state = 0x2951_0002_u64;
        let native = random_matrix(6, 4, &mut state);
        let (basis, block_ranks) = concatenated(&random_blocks(6, 4, &[2, 1], &mut state));
        let vectors = random_matrix(3, 2, &mut state);
        let registry = registry_with("mlp.weight", &native);
        let anchor = ResidualAnchor::new(
            &registry,
            TensorId("mlp.weight".to_string()),
            native.view(),
            basis,
            block_ranks,
            ComponentCoefficients::Dense(vectors),
        )
        .expect("the registered teacher anchors");
        for orientation in [TieOrientation::Identity, TieOrientation::Transpose] {
            let (width, teacher_view) = match orientation {
                TieOrientation::Identity => (4, native.view()),
                TieOrientation::Transpose => (6, native.t()),
            };
            let inputs = random_matrix(5, width, &mut state);
            let teacher = native_linear(teacher_view, inputs.view()).expect("a small product is admitted");
            let lifted = anchor
                .apply(&AnchorMask::all_on(3), inputs.view(), orientation)
                .expect("the all-on mask applies");
            assert!(
                lifted.iter().zip(teacher.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{orientation:?}: the all-on lift is not the teacher's native product bit for bit"
            );
            // Positive control: these components act on these inputs, so switching one
            // off moves the bits. An all-on equality on components that vanish would
            // hold vacuously.
            let mut one_off = AnchorMask::all_on(3);
            one_off.components[0] = 0.0;
            let moved = anchor
                .apply(&one_off, inputs.view(), orientation)
                .expect("a valid mask applies");
            assert!(
                moved.iter().zip(teacher.iter()).any(|(a, b)| a.to_bits() != b.to_bits()),
                "{orientation:?}: switching a component off left every output bit unchanged"
            );
        }
    }

    #[test]
    fn a_unit_residual_mask_is_the_parameter_edit_a_runner_applies() {
        let mut state = 0x2951_0006_u64;
        let native = random_matrix(6, 5, &mut state);
        let blocks = random_blocks(6, 5, &[1, 2], &mut state);
        let vectors = random_matrix(3, 2, &mut state);
        let registry = registry_with("mlp.dense.weight", &native);
        let (basis, block_ranks) = concatenated(&blocks);
        let anchor = ResidualAnchor::new(
            &registry,
            TensorId("mlp.dense.weight".to_string()),
            native.view(),
            basis,
            block_ranks,
            ComponentCoefficients::Dense(vectors.clone()),
        )
        .expect("the registered teacher anchors");
        let mask = AnchorMask { residual: 1.0, components: vec![0.2, 1.0, -0.5] };
        let edit = anchor
            .parameter_edit(&mask)
            .expect("a unit residual mask has a product form");
        let (parameter, rows, cols, rank, left, right) = match edit {
            Some(InterventionChange::ParameterEdit {
                parameter,
                rows,
                cols,
                rank,
                left,
                right,
                scope: ParameterEditScope::Global,
            }) => (parameter, rows, cols, rank, left, right),
            other => panic!("expected a global parameter edit, got {other:?}"),
        };
        assert_eq!((parameter.as_str(), rows, cols, rank), ("mlp.dense.weight", 6, 5, 3));
        let product = Array2::from_shape_vec((rows, rank), left)
            .expect("left factor is row-major (rows, rank)")
            .dot(&Array2::from_shape_vec((cols, rank), right).expect("right factor is row-major (cols, rank)").t());
        let reference = edited_tensor(&native, &blocks, &vectors, &mask) - &native;
        // Rounded operations on the deepest path. The edit forms a term scale (a
        // difference, a product and `C` additions), scales a left entry once and sums
        // `rank` products. The reference forms the edited tensor (see
        // `anchor_agreement`) and subtracts the teacher once.
        let components = vectors.nrows();
        let edit_depth = components + 2 + 1 + rank;
        let reference_depth = (2 + 1).max(3) + components * blocks.len() + 1;
        let band = absolute_edit_magnitude(&native, &blocks, &vectors, &mask)
            * (accumulation_growth(edit_depth) + accumulation_growth(reference_depth));
        assert!(
            product
                .iter()
                .zip(reference.iter())
                .zip(band.iter())
                .all(|((a, b), bound)| (a - b).abs() <= *bound),
            "the parameter edit departs from Theta(m) - Theta_* beyond the roundoff band"
        );
        // Positive control: the edit is not vacuously small.
        assert!(reference.iter().zip(band.iter()).any(|(value, bound)| value.abs() > *bound));

        assert!(matches!(anchor.parameter_edit(&AnchorMask::all_on(3)), Ok(None)));
        assert!(matches!(
            anchor.parameter_edit(&AnchorMask { residual: 0.5, components: vec![0.2, 1.0, -0.5] }),
            Err(LiftError::ResidualScaleIsNotAProduct { .. })
        ));
        let small = random_matrix(2, 2, &mut state);
        let small_registry = registry_with("small.weight", &small);
        let (small_basis, small_ranks) = concatenated(&random_blocks(2, 2, &[2, 1], &mut state));
        let small_anchor = ResidualAnchor::new(
            &small_registry,
            TensorId("small.weight".to_string()),
            small.view(),
            small_basis,
            small_ranks,
            ComponentCoefficients::Basis,
        )
        .expect("the registered teacher anchors");
        assert!(matches!(
            small_anchor.parameter_edit(&AnchorMask { residual: 1.0, components: vec![0.0, 0.0] }),
            Err(LiftError::EditRankExceedsShape { rank: 3, rows: 2, cols: 2 })
        ));
        assert!(small_anchor
            .parameter_edit(&AnchorMask { residual: 1.0, components: vec![0.0, 1.0] })
            .expect("two terms fit a (2, 2) edit")
            .is_some());
    }

    #[test]
    fn the_teacher_fingerprint_reads_values_and_wiring_not_registration_order() {
        let mut state = 0x2951_0003_u64;
        let embedding = random_matrix(6, 3, &mut state);
        let mut bias = ndarray::Array1::from_vec(vec![0.0, 0.25, -1.5]);
        let wte = TensorId("wte.weight".to_string());
        let build = |bias: &ndarray::Array1<f64>, unembed_map: UseMap, forward: bool| {
            let mut registry = TensorRegistry::default();
            let bias_id = TensorId("ln_f.bias".to_string());
            if forward {
                registry.register_storage(wte.clone(), embedding.view().into_dyn()).expect("fresh name");
                registry.register_storage(bias_id, bias.view().into_dyn()).expect("fresh name");
            } else {
                registry.register_storage(bias_id, bias.view().into_dyn()).expect("fresh name");
                registry.register_storage(wte.clone(), embedding.view().into_dyn()).expect("fresh name");
            }
            let lm_head = TensorId("lm_head.weight".to_string());
            registry.register_alias(lm_head.clone(), wte.clone()).expect("storage alias");
            registry
                .register_use_site(UseSiteId::read(&wte, 0), wte.clone(), UseMap::Stored)
                .expect("fresh site");
            registry
                .register_use_site(UseSiteId::read(&wte, 1), lm_head, unembed_map)
                .expect("fresh site");
            registry
        };
        let identity_head = UseMap::Linear(TieOrientation::Identity);
        let forward = build(&bias, identity_head, true).teacher_fingerprint();
        let reversed = build(&bias, identity_head, false).teacher_fingerprint();
        assert_eq!(forward, reversed, "registration order moved the fingerprint");

        // Positive controls: one sign bit of one value, each other map of one use site
        // and one extra use site each move it.
        bias[0] = -0.0;
        let signed_zero = build(&bias, identity_head, true).teacher_fingerprint();
        assert_ne!(forward, signed_zero, "+0.0 and -0.0 fingerprint alike");
        bias[0] = 0.0;
        let transposed = build(&bias, UseMap::Linear(TieOrientation::Transpose), true).teacher_fingerprint();
        assert_ne!(forward, transposed, "a use site's orientation does not reach the fingerprint");
        let stored = build(&bias, UseMap::Stored, true).teacher_fingerprint();
        assert_ne!(forward, stored, "a stored read fingerprints like a linear use");
        assert_ne!(transposed, stored, "a stored read fingerprints like a transposed use");
        let mut extra = build(&bias, identity_head, true);
        let bias_id = TensorId("ln_f.bias".to_string());
        extra
            .register_use_site(UseSiteId::read(&bias_id, 0), bias_id, UseMap::Stored)
            .expect("fresh site");
        assert_ne!(forward, extra.teacher_fingerprint(), "a use site does not reach the fingerprint");
    }

    #[test]
    fn tied_names_resolve_to_one_storage_and_a_global_edit_reaches_every_use() {
        let mut state = 0x2951_0004_u64;
        let embedding = random_matrix(6, 3, &mut state);
        let norm = ndarray::Array1::from_vec(vec![1.0, 0.5, 2.0]);
        let wte = TensorId("wte.weight".to_string());
        let lm_head = TensorId("lm_head.weight".to_string());
        let ln_f = TensorId("ln_f.weight".to_string());
        let mut registry = TensorRegistry::default();
        registry.register_storage(wte.clone(), embedding.view().into_dyn()).expect("fresh name");
        registry.register_storage(ln_f.clone(), norm.view().into_dyn()).expect("fresh name");
        registry.register_alias(lm_head.clone(), wte.clone()).expect("an alias of storage");
        // The embedding looks rows up, the tied head multiplies by the transpose, and the
        // final norm scales element-wise.
        let embed = UseSiteId::read(&wte, 0);
        let unembed = UseSiteId::read(&wte, 1);
        let final_norm = UseSiteId::read(&ln_f, 0);
        registry
            .register_use_site(embed.clone(), wte.clone(), UseMap::Stored)
            .expect("fresh site");
        registry
            .register_use_site(unembed.clone(), lm_head.clone(), UseMap::Linear(TieOrientation::Transpose))
            .expect("a matrix may be multiplied transposed");
        registry
            .register_use_site(final_norm.clone(), ln_f.clone(), UseMap::Stored)
            .expect("a vector may be read as stored");

        assert_eq!(unembed, UseSiteId("wte.weight#1".to_string()));
        assert_eq!(registry.storage_of(&lm_head), Ok(&wte));
        assert_eq!(
            registry.resolve_use_site(&unembed),
            Ok(ResolvedRead { storage: wte.clone(), map: UseMap::Linear(TieOrientation::Transpose) })
        );
        assert_eq!(
            registry.resolve_use_site(&embed),
            Ok(ResolvedRead { storage: wte.clone(), map: UseMap::Stored })
        );
        assert_eq!(registry.use_sites_of(&wte), vec![&embed, &unembed]);
        assert_eq!(registry.use_sites_of(&ln_f), vec![&final_norm]);
        assert_eq!(registry.storage_ids().collect::<Vec<_>>(), vec![&ln_f, &wte]);

        // Each refusal fires on its bad input, and the matching good input passes.
        assert_eq!(
            registry.register_alias(wte.clone(), ln_f.clone()),
            Err(LiftError::DuplicateName("wte.weight".to_string()))
        );
        assert_eq!(
            registry.register_alias(TensorId("chained".to_string()), lm_head.clone()),
            Err(LiftError::AliasOfAlias { alias: "chained".to_string(), target: "lm_head.weight".to_string() })
        );
        for orientation in [TieOrientation::Identity, TieOrientation::Transpose] {
            assert_eq!(
                registry.register_use_site(UseSiteId::read(&ln_f, 1), ln_f.clone(), UseMap::Linear(orientation)),
                Err(LiftError::NotAMatrix { tensor: "ln_f.weight".to_string(), shape: vec![3] })
            );
        }
        assert_eq!(
            registry.register_use_site(UseSiteId::read(&ln_f, 1), ln_f.clone(), UseMap::Stored),
            Ok(())
        );
        assert_eq!(
            registry.register_use_site(
                UseSiteId("stray".to_string()),
                TensorId("missing".to_string()),
                UseMap::Stored
            ),
            Err(LiftError::UnknownTensor("missing".to_string()))
        );
        assert_eq!(
            registry.register_use_site(embed, wte, UseMap::Linear(TieOrientation::Identity)),
            Err(LiftError::DuplicateUseSite("wte.weight#0".to_string()))
        );
    }

    #[test]
    fn the_anchor_refuses_values_that_are_not_the_registered_teacher() {
        let mut state = 0x2951_0005_u64;
        let native = random_matrix(4, 3, &mut state);
        let (basis, block_ranks) = concatenated(&random_blocks(4, 3, &[1], &mut state));
        let registry = registry_with("attn.weight", &native);
        let id = TensorId("attn.weight".to_string());
        let anchor = ResidualAnchor::new(
            &registry,
            id.clone(),
            native.view(),
            basis.clone(),
            block_ranks.clone(),
            ComponentCoefficients::Basis,
        )
        .expect("the registered teacher anchors");

        let mut flipped = native.clone();
        flipped[[2, 1]] = f64::from_bits(flipped[[2, 1]].to_bits() ^ 1);
        assert!(matches!(
            ResidualAnchor::new(
                &registry,
                id.clone(),
                flipped.view(),
                basis.clone(),
                block_ranks.clone(),
                ComponentCoefficients::Basis
            ),
            Err(LiftError::TeacherMismatch { .. })
        ));
        assert!(matches!(
            ResidualAnchor::new(
                &registry,
                id.clone(),
                native.t(),
                basis.clone(),
                block_ranks.clone(),
                ComponentCoefficients::Basis
            ),
            Err(LiftError::NativeShape { .. })
        ));
        let (wide, wide_ranks) = concatenated(&random_blocks(5, 3, &[1], &mut state));
        assert!(matches!(
            ResidualAnchor::new(&registry, id.clone(), native.view(), wide, wide_ranks, ComponentCoefficients::Basis),
            Err(LiftError::FactorShape { output_dim: 5, input_dim: 3, rows: 4, cols: 3 })
        ));
        assert!(matches!(
            ResidualAnchor::new(&registry, id, native.view(), basis, vec![2], ComponentCoefficients::Basis),
            Err(LiftError::BlockRanks { ranked_terms: 2, terms: 1 })
        ));

        let inputs = random_matrix(2, 3, &mut state);
        assert!(anchor.apply(&AnchorMask::all_on(1), inputs.view(), TieOrientation::Identity).is_ok());
        assert!(matches!(
            anchor.apply(&AnchorMask::all_on(2), inputs.view(), TieOrientation::Identity),
            Err(LiftError::MaskLength { components: 1, mask: 2 })
        ));
        assert!(matches!(
            anchor.apply(
                &AnchorMask { residual: f64::NAN, components: vec![1.0] },
                inputs.view(),
                TieOrientation::Identity
            ),
            Err(LiftError::NonFiniteMask)
        ));
        assert!(matches!(
            anchor.apply(&AnchorMask::all_on(1), inputs.view(), TieOrientation::Transpose),
            Err(LiftError::Apply(ApplyError::Shape { .. }))
        ));
    }
}
