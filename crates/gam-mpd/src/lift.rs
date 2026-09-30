//! Native lift: the tensor registry.
//!
//! The registry names the teacher's parameters the way the executing framework
//! names them, the same `parameter` string an
//! `inference::intervention_shard::InterventionChange::ParameterEdit` carries. It
//! records each storage tensor's shape and a fingerprint of its values, the other
//! names of that storage, and every use site in the executed graph with the name
//! it reads and what it does with it: a linear map in a named orientation, or a
//! read of the stored values. A global edit of a storage tensor reaches every use
//! site that reads it under any name; a use-specific edit reaches one site.

use std::collections::BTreeMap;
use std::fmt;

use gam_linalg::utils::splitmix64_hash;
use ndarray::ArrayViewD;
use serde::{Deserialize, Serialize};


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

/// A refusal of the registry.
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
        }
    }
}

impl std::error::Error for LiftError {}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::utils::splitmix64;
    use ndarray::Array2;

    fn uniform(state: &mut u64) -> f64 {
        (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn random_matrix(rows: usize, cols: usize, state: &mut u64) -> Array2<f64> {
        let values: Vec<f64> = std::iter::repeat_with(|| uniform(state))
            .take(rows * cols)
            .collect();
        Array2::from_shape_vec((rows, cols), values).expect("row-major values fill the shape")
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
}
