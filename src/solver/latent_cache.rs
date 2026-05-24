//! Persistent cache for latent-coordinate REML design evaluations.
//!
//! This follows the same invalidation shape as the design-revision pattern in
//! `src/terms/smooth.rs`'s `SpatialLogKappa` path (around the
//! `SpatialLogKappa` cache near line 12805) and the `EvalShared` rho-keyed
//! cache in `src/solver/reml/mod.rs` (around line 3525).  REML's outer
//! evaluator is reentrant for each theta: the rho component is already covered
//! by `EvalShared`, while the design-moving component is fully determined by
//! the latent fingerprint.  Together, `(rho, latent_fingerprint)` is sufficient
//! to reuse the realized surface until the caller bumps the design revision or
//! explicitly invalidates this cache.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use sha2::{Digest, Sha256};

use crate::basis::{DuchonNullspaceOrder, MaternNu, RadialScalarKind};
use crate::estimate::EstimationError;
use crate::estimate::reml::DirectionalHyperParam;
use crate::solver::persistent_warm_start::StableHasher;
use crate::solver::riemannian_retraction::{Retraction, RetractionKind};
use crate::terms::latent_coord::{
    AuxPriorFamily, AuxPriorStrength, LatentCoordValues, LatentIdMode, LatentManifold,
};
use crate::terms::smooth::TermCollectionDesign;

const DEFAULT_LATENT_CACHE_CAPACITY: usize = 4;
const DEFAULT_PERSISTENT_LATENT_CACHE_CAPACITY: usize = 16;
const DEFAULT_PERSISTENT_LATENT_CACHE_BYTE_BUDGET: usize = 1024 * 1024 * 1024;

static PERSISTENT_LATENT_DESIGN_CACHE: OnceLock<Mutex<PersistentLatentDesignCache>> =
    OnceLock::new();

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LatentRetractionRegistry {
    block: Option<RetractionKind>,
}

impl LatentRetractionRegistry {
    pub(crate) fn all_euclidean() -> Self {
        Self { block: None }
    }

    pub(crate) fn new(block: RetractionKind) -> Self {
        if block.is_euclidean() {
            Self::all_euclidean()
        } else {
            Self { block: Some(block) }
        }
    }

    pub(crate) fn is_all_euclidean(&self) -> bool {
        self.block.is_none()
    }

    pub(crate) fn ambient_dim(&self, fallback_dim: usize) -> usize {
        self.block
            .as_ref()
            .map_or(fallback_dim, RetractionKind::ambient_dim)
    }

    pub(crate) fn metric_weights(&self, fallback_dim: usize) -> Vec<f64> {
        self.block
            .as_ref()
            .map_or_else(|| vec![1.0; fallback_dim], RetractionKind::metric_weights)
    }

    pub(crate) fn validate_dim(&self, latent_dim: usize, context: &str) -> Result<(), String> {
        let dim = self.ambient_dim(latent_dim);
        if dim != latent_dim {
            return Err(format!(
                "{context} retraction ambient dimension {dim} does not match latent d={latent_dim}"
            ));
        }
        Ok(())
    }

    pub(crate) fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        debug_assert_eq!(base.len(), tangent.len());
        if let Some(block) = self.block.as_ref() {
            block.retract(base, tangent);
        } else {
            for (value, delta) in base.iter_mut().zip(tangent.iter()) {
                *value += *delta;
            }
        }
    }
}

/// O(N) identity summary for a flat latent-coordinate vector.
#[derive(Clone, Debug)]
pub(crate) struct LatentFingerprint {
    pub(crate) hash: u64,
    pub(crate) len: usize,
}

impl LatentFingerprint {
    pub(crate) fn from_flat(flat: &[f64]) -> Self {
        let mut hasher = DefaultHasher::new();
        flat.len().hash(&mut hasher);
        for &value in flat {
            value.to_bits().hash(&mut hasher);
        }
        Self {
            hash: hasher.finish(),
            len: flat.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CacheDigest([u8; 32]);

struct CacheDigestBuilder {
    hasher: Sha256,
}

impl CacheDigestBuilder {
    fn new(namespace: &str) -> Self {
        let mut out = Self {
            hasher: Sha256::new(),
        };
        out.write_str(namespace);
        out
    }

    fn write_bool(&mut self, value: bool) {
        self.hasher.update([u8::from(value)]);
    }

    fn write_u8(&mut self, value: u8) {
        self.hasher.update([value]);
    }

    fn write_u64(&mut self, value: u64) {
        self.hasher.update(value.to_le_bytes());
    }

    fn write_f64(&mut self, value: f64) {
        self.hasher.update(value.to_bits().to_le_bytes());
    }

    fn write_str(&mut self, value: &str) {
        self.write_usize(value.len());
        self.hasher.update(value.as_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.hasher.update((value as u64).to_le_bytes());
    }

    fn finish(self) -> CacheDigest {
        let digest = self.hasher.finalize();
        let mut out = [0_u8; 32];
        out.copy_from_slice(&digest);
        CacheDigest(out)
    }
}

#[derive(Clone)]
pub(crate) enum LatentBasisKind {
    // Basis/evaluator family for Phi(t); the per-row latent values live in LatentCoordValues.
    Matern {
        centers: Array2<f64>,
        length_scale: f64,
        nu: MaternNu,
        aniso_log_scales: Vec<f64>,
        chunk_size: Option<usize>,
    },
    Duchon {
        centers: Array2<f64>,
        length_scale: Option<f64>,
        power: usize,
        nullspace_order: DuchonNullspaceOrder,
        aniso_log_scales: Vec<f64>,
    },
    Sphere {
        centers: Array2<f64>,
        penalty_order: usize,
    },
    PeriodicBspline {
        domain_start: f64,
        period: f64,
        degree: usize,
        num_basis: usize,
    },
    TensorBspline {
        knots: Vec<Array1<f64>>,
        degrees: Vec<usize>,
    },
    Pca {
        basis_matrix: Array2<f64>,
        centered: bool,
        center_mean_fingerprint: Option<u64>,
        smooth_penalty: f64,
        pca_basis_path: Option<PathBuf>,
        chunk_size: usize,
    },
}

impl LatentBasisKind {
    fn centers(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Matern { centers, .. }
            | Self::Duchon { centers, .. }
            | Self::Sphere { centers, .. } => Some(centers),
            Self::PeriodicBspline { .. } | Self::TensorBspline { .. } => None,
            Self::Pca { .. } => None,
        }
    }

    fn streams_radial_cache(&self) -> bool {
        matches!(
            self,
            Self::Matern {
                chunk_size: Some(_),
                ..
            }
        )
    }

    fn cache_digest(&self) -> CacheDigest {
        let mut hasher = CacheDigestBuilder::new("latent-basis-v1");
        match self {
            Self::Matern {
                centers,
                length_scale,
                nu,
                aniso_log_scales,
                chunk_size,
            } => {
                hasher.write_usize(0);
                hasher.write_usize(centers.nrows());
                hasher.write_usize(centers.ncols());
                hasher.write_f64(*length_scale);
                hasher.write_usize(matern_nu_signature(*nu));
                hash_f64_slice(aniso_log_scales, &mut hasher);
                hash_optional_usize(*chunk_size, &mut hasher);
                hash_matrix(centers, &mut hasher);
            }
            Self::Duchon {
                centers,
                length_scale,
                power,
                nullspace_order,
                aniso_log_scales,
            } => {
                hasher.write_usize(1);
                hasher.write_usize(centers.nrows());
                hasher.write_usize(centers.ncols());
                hash_optional_f64(*length_scale, &mut hasher);
                hasher.write_usize(*power);
                hash_duchon_nullspace_order(*nullspace_order, &mut hasher);
                hash_f64_slice(aniso_log_scales, &mut hasher);
                hash_matrix(centers, &mut hasher);
            }
            Self::Sphere {
                centers,
                penalty_order,
            } => {
                hasher.write_usize(2);
                hasher.write_usize(centers.nrows());
                hasher.write_usize(centers.ncols());
                hasher.write_usize(*penalty_order);
                hash_matrix(centers, &mut hasher);
            }
            Self::PeriodicBspline {
                domain_start,
                period,
                degree,
                num_basis,
            } => {
                hasher.write_usize(3);
                hasher.write_f64(*domain_start);
                hasher.write_f64(*period);
                hasher.write_usize(*degree);
                hasher.write_usize(*num_basis);
            }
            Self::TensorBspline { knots, degrees } => {
                hasher.write_usize(4);
                hasher.write_usize(degrees.len());
                for &degree in degrees {
                    hasher.write_usize(degree);
                }
                hasher.write_usize(knots.len());
                for axis_knots in knots {
                    hash_vector(axis_knots, &mut hasher);
                }
            }
            Self::Pca {
                basis_matrix,
                centered,
                center_mean_fingerprint,
                smooth_penalty,
                pca_basis_path,
                chunk_size,
            } => {
                hasher.write_usize(5);
                hasher.write_u8(*centered as u8);
                if let Some(fp) = center_mean_fingerprint {
                    hasher.write_u64(*fp);
                }
                hasher.write_u64(smooth_penalty.to_bits());
                if let Some(path) = pca_basis_path {
                    hasher.write_u8(1);
                    hasher.hasher.update(path.to_string_lossy().as_bytes());
                    if let Ok(meta) = std::fs::metadata(path) {
                        hasher.write_u64(meta.len());
                        if let Ok(modified) = meta.modified()
                            && let Ok(elapsed) =
                                modified.duration_since(std::time::SystemTime::UNIX_EPOCH)
                        {
                            hasher.write_u64(elapsed.as_secs());
                            hasher.write_u64(elapsed.subsec_nanos() as u64);
                        }
                    }
                } else {
                    hasher.write_u8(0);
                }
                hasher.write_usize(*chunk_size);
                hasher.write_usize(basis_matrix.nrows());
                hasher.write_usize(basis_matrix.ncols());
                hash_matrix(basis_matrix, &mut hasher);
            }
        }
        hasher.finish()
    }
}

pub(crate) fn pca_center_mean_fingerprint(mean: &Array1<f64>) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.write_usize(mean.len());
    for &value in mean.iter() {
        hasher.write_f64(value);
    }
    hasher.finish_u64()
}

fn matern_nu_signature(nu: MaternNu) -> usize {
    match nu {
        MaternNu::Half => 0,
        MaternNu::ThreeHalves => 1,
        MaternNu::FiveHalves => 2,
        MaternNu::SevenHalves => 3,
        MaternNu::NineHalves => 4,
    }
}

fn hash_duchon_nullspace_order(order: DuchonNullspaceOrder, hasher: &mut CacheDigestBuilder) {
    match order {
        DuchonNullspaceOrder::Zero => {
            hasher.write_usize(0);
        }
        DuchonNullspaceOrder::Linear => {
            hasher.write_usize(1);
        }
        DuchonNullspaceOrder::Degree(degree) => {
            hasher.write_usize(2);
            hasher.write_usize(degree);
        }
    }
}

fn hash_optional_f64(value: Option<f64>, hasher: &mut CacheDigestBuilder) {
    match value {
        Some(value) => {
            hasher.write_bool(true);
            hasher.write_f64(value);
        }
        None => {
            hasher.write_bool(false);
        }
    }
}

fn hash_optional_usize(value: Option<usize>, hasher: &mut CacheDigestBuilder) {
    match value {
        Some(value) => {
            hasher.write_bool(true);
            hasher.write_usize(value);
        }
        None => {
            hasher.write_bool(false);
        }
    }
}

fn hash_f64_slice(values: &[f64], hasher: &mut CacheDigestBuilder) {
    hasher.write_usize(values.len());
    for &value in values {
        hasher.write_f64(value);
    }
}

fn hash_matrix(matrix: &Array2<f64>, hasher: &mut CacheDigestBuilder) {
    hasher.write_usize(matrix.nrows());
    hasher.write_usize(matrix.ncols());
    for &value in matrix.iter() {
        hasher.write_f64(value);
    }
}

fn hash_vector(vector: &Array1<f64>, hasher: &mut CacheDigestBuilder) {
    hasher.write_usize(vector.len());
    for &value in vector.iter() {
        hasher.write_f64(value);
    }
}

fn latent_metadata_cache_digest(latent: &LatentCoordValues) -> CacheDigest {
    let mut hasher = CacheDigestBuilder::new("latent-cache-metadata-v1");
    hasher.write_usize(latent.n_obs());
    hasher.write_usize(latent.latent_dim());
    hash_latent_manifold(latent.manifold(), &mut hasher);
    hash_latent_id_mode(latent.id_mode(), &mut hasher);
    hasher.finish()
}

fn hash_latent_id_mode(id_mode: &LatentIdMode, hasher: &mut CacheDigestBuilder) {
    match id_mode {
        LatentIdMode::AuxPrior {
            u,
            family,
            strength,
        } => {
            hasher.write_usize(0);
            hash_matrix(u, hasher);
            hash_aux_prior_family(*family, hasher);
            hash_aux_prior_strength(*strength, hasher);
        }
        LatentIdMode::AuxPriorDimSelection {
            u,
            family,
            strength,
            init_log_precision,
        } => {
            hasher.write_usize(1);
            hash_matrix(u, hasher);
            hash_aux_prior_family(*family, hasher);
            hash_aux_prior_strength(*strength, hasher);
            hash_optional_vector(init_log_precision.as_ref(), hasher);
        }
        LatentIdMode::DimSelection { init_log_precision } => {
            hasher.write_usize(2);
            hash_optional_vector(init_log_precision.as_ref(), hasher);
        }
        LatentIdMode::None => {
            hasher.write_usize(3);
        }
    }
}

fn hash_aux_prior_family(family: AuxPriorFamily, hasher: &mut CacheDigestBuilder) {
    hasher.write_usize(match family {
        AuxPriorFamily::Ridge => 0,
        AuxPriorFamily::Linear => 1,
    });
}

fn hash_aux_prior_strength(strength: AuxPriorStrength, hasher: &mut CacheDigestBuilder) {
    match strength {
        AuxPriorStrength::Auto => {
            hasher.write_usize(0);
        }
        AuxPriorStrength::Fixed(value) => {
            hasher.write_usize(1);
            hasher.write_f64(value);
        }
    }
}

fn hash_optional_vector(vector: Option<&Array1<f64>>, hasher: &mut CacheDigestBuilder) {
    match vector {
        Some(vector) => {
            hasher.write_bool(true);
            hash_vector(vector, hasher);
        }
        None => {
            hasher.write_bool(false);
        }
    }
}

fn hash_latent_manifold(manifold: &LatentManifold, hasher: &mut CacheDigestBuilder) {
    match manifold {
        LatentManifold::Euclidean => {
            hasher.write_usize(0);
        }
        LatentManifold::Circle => {
            hasher.write_usize(1);
        }
        LatentManifold::Sphere { dim } => {
            hasher.write_usize(2);
            hasher.write_usize(*dim);
        }
        LatentManifold::Interval { lo, hi } => {
            hasher.write_usize(3);
            hasher.write_f64(*lo);
            hasher.write_f64(*hi);
        }
        LatentManifold::Product(parts) => {
            hasher.write_usize(4);
            hasher.write_usize(parts.len());
            for part in parts {
                hash_latent_manifold(part, hasher);
            }
        }
        LatentManifold::ProductWithMetric { manifolds, weights } => {
            hasher.write_usize(5);
            hasher.write_usize(manifolds.len());
            for part in manifolds {
                hash_latent_manifold(part, hasher);
            }
            hash_f64_slice(weights, hasher);
        }
    }
}

#[derive(Clone)]
pub(crate) struct RadialDistanceMatrices {
    pub(crate) squared: Array2<f64>,
    pub(crate) distance: Array2<f64>,
}

#[derive(Clone)]
pub(crate) struct BasisDerivativeJets {
    pub(crate) phi: Option<Array2<f64>>,
    pub(crate) q: Option<Array2<f64>>,
    pub(crate) t: Option<Array2<f64>>,
    pub(crate) phi_r: Option<Array2<f64>>,
    pub(crate) phi_rr: Option<Array2<f64>>,
    pub(crate) operator_resident: bool,
}

impl BasisDerivativeJets {
    fn empty() -> Self {
        Self {
            phi: None,
            q: None,
            t: None,
            phi_r: None,
            phi_rr: None,
            operator_resident: false,
        }
    }
}

#[derive(Clone)]
pub(crate) struct CachedDesign {
    pub(crate) latent_id: u64,
    pub(crate) fingerprint: LatentFingerprint,
    basis_digest: CacheDigest,
    latent_metadata_digest: CacheDigest,
    latent_bits: Arc<[u64]>,
    cacheable: bool,
    pub(crate) design: TermCollectionDesign,
    pub(crate) hyper_dirs: Vec<DirectionalHyperParam>,
    pub(crate) radial_distances: RadialDistanceMatrices,
    pub(crate) basis_derivative_jets: BasisDerivativeJets,
}

pub(crate) struct ComputedLatentDesign {
    pub(crate) design: TermCollectionDesign,
    pub(crate) hyper_dirs: Vec<DirectionalHyperParam>,
}

pub(crate) struct LatentDesignLookup<'a> {
    pub(crate) cached: &'a CachedDesign,
    pub(crate) entry_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PersistentLatentDesignKey {
    latent_id: u64,
    flat_hash: u64,
    basis_digest: CacheDigest,
    latent_metadata_digest: CacheDigest,
}

struct PersistentLatentDesignEntry {
    fingerprint: LatentFingerprint,
    cached: Arc<CachedDesign>,
    bytes: usize,
}

pub(crate) struct PersistentLatentDesignCache {
    entries: HashMap<PersistentLatentDesignKey, PersistentLatentDesignEntry>,
    lru: VecDeque<PersistentLatentDesignKey>,
    capacity: usize,
    byte_budget: usize,
    cache_bytes: usize,
}

impl Default for PersistentLatentDesignCache {
    fn default() -> Self {
        Self::new(DEFAULT_PERSISTENT_LATENT_CACHE_CAPACITY)
    }
}

impl PersistentLatentDesignCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            capacity: capacity.max(1),
            byte_budget: DEFAULT_PERSISTENT_LATENT_CACHE_BYTE_BUDGET,
            cache_bytes: 0,
        }
    }

    pub(crate) fn lookup(
        &mut self,
        latent: &LatentCoordValues,
        basis_digest: CacheDigest,
        latent_metadata_digest: CacheDigest,
        fingerprint: &LatentFingerprint,
    ) -> Result<Option<Arc<CachedDesign>>, EstimationError> {
        let key = PersistentLatentDesignKey {
            latent_id: latent.latent_id(),
            flat_hash: fingerprint.hash,
            basis_digest,
            latent_metadata_digest,
        };
        let Some(entry) = self.entries.get(&key) else {
            return Ok(None);
        };
        let cached = entry.cached.clone();
        let entry_fingerprint = entry.fingerprint.clone();
        self.touch(key);
        if entry_fingerprint.len != fingerprint.len {
            return Ok(None);
        }
        if entry_fingerprint.hash == fingerprint.hash
            && cached.cacheable
            && cached.basis_digest == basis_digest
            && cached.latent_metadata_digest == latent_metadata_digest
            && latent_bits_match(latent, &cached.latent_bits)
        {
            return Ok(Some(cached));
        }
        Ok(None)
    }

    pub(crate) fn insert(&mut self, cached: Arc<CachedDesign>) {
        if !cached.cacheable {
            return;
        }
        let bytes = cached.resident_byte_count();
        if bytes > self.byte_budget {
            return;
        }
        let key = PersistentLatentDesignKey {
            latent_id: cached.latent_id,
            flat_hash: cached.fingerprint.hash,
            basis_digest: cached.basis_digest,
            latent_metadata_digest: cached.latent_metadata_digest,
        };
        let entry = PersistentLatentDesignEntry {
            fingerprint: cached.fingerprint.clone(),
            cached,
            bytes,
        };
        if let Some(old) = self.entries.insert(key, entry) {
            self.cache_bytes = self.cache_bytes.saturating_sub(old.bytes);
        }
        self.cache_bytes = self.cache_bytes.saturating_add(bytes);
        self.touch(key);
        self.evict_to_limits();
    }

    fn evict_to_limits(&mut self) {
        while self.entries.len() > self.capacity || self.cache_bytes > self.byte_budget {
            let Some(evicted) = self.lru.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&evicted) {
                self.cache_bytes = self.cache_bytes.saturating_sub(entry.bytes);
            }
        }
    }

    fn touch(&mut self, key: PersistentLatentDesignKey) {
        if let Some(index) = self.lru.iter().position(|queued| *queued == key) {
            self.lru.remove(index);
        }
        self.lru.push_back(key);
    }
}

pub(crate) struct LatentDesignCache {
    entries: Vec<LatentDesignCacheEntry>,
    capacity: usize,
    clock: u64,
    iteration: u64,
    next_entry_id: u64,
}

struct LatentDesignCacheEntry {
    id: u64,
    cached: Arc<CachedDesign>,
    last_used: u64,
    iteration: u64,
}

impl Default for LatentDesignCache {
    fn default() -> Self {
        Self::new(DEFAULT_LATENT_CACHE_CAPACITY)
    }
}

impl LatentDesignCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity: capacity.max(1),
            clock: 0,
            iteration: 0,
            next_entry_id: 0,
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.entries.clear();
    }

    pub(crate) fn invalidate_all(&mut self) {
        self.entries.clear();
        self.clock = self.clock.wrapping_add(1);
        self.iteration = self.iteration.wrapping_add(1);
    }

    pub(crate) fn lookup_or_compute<F>(
        &mut self,
        latent: Arc<LatentCoordValues>,
        basis_kind: LatentBasisKind,
        compute: F,
    ) -> Result<LatentDesignLookup<'_>, EstimationError>
    where
        F: FnOnce() -> Result<ComputedLatentDesign, EstimationError>,
    {
        self.iteration = self.iteration.wrapping_add(1);
        self.clock = self.clock.wrapping_add(1);
        let flat = latent.as_flat();
        let flat_slice = flat
            .as_slice()
            .expect("LatentCoordValues flat storage must be contiguous");
        let fingerprint = LatentFingerprint::from_flat(flat_slice);
        let basis_digest = basis_kind.cache_digest();
        let latent_metadata_digest = latent_metadata_cache_digest(&latent);
        let cacheable = flat_slice.iter().all(|value| value.is_finite());
        if cacheable {
            if let Some(index) = self.find_entry(&latent, basis_digest, latent_metadata_digest) {
                self.entries[index].last_used = self.clock;
                return Ok(LatentDesignLookup {
                    cached: self.entries[index].cached.as_ref(),
                    entry_id: self.entries[index].id,
                });
            }
        }
        if cacheable {
            if let Some(cached) = lookup_persistent_latent_design(
                &latent,
                basis_digest,
                latent_metadata_digest,
                &fingerprint,
            )? {
                let id = self.next_entry_id;
                self.next_entry_id = self.next_entry_id.wrapping_add(1);
                self.insert(cached, id);
                return self.lookup_inserted(id);
            }
        }

        let computed = compute()?;
        let radial_distances = if basis_kind.streams_radial_cache() {
            RadialDistanceMatrices {
                squared: Array2::<f64>::zeros((0, 0)),
                distance: Array2::<f64>::zeros((0, 0)),
            }
        } else {
            match basis_kind.centers() {
                Some(centers) => build_radial_distances(&latent, centers)?,
                None => RadialDistanceMatrices {
                    squared: Array2::<f64>::zeros((0, 0)),
                    distance: Array2::<f64>::zeros((0, 0)),
                },
            }
        };
        let basis_derivative_jets =
            build_basis_derivative_jets(&latent, &basis_kind, &radial_distances)?;
        let id = self.next_entry_id;
        self.next_entry_id = self.next_entry_id.wrapping_add(1);
        let entry = Arc::new(CachedDesign {
            latent_id: latent.latent_id(),
            fingerprint,
            basis_digest,
            latent_metadata_digest,
            latent_bits: latent_bits(&latent),
            cacheable,
            design: computed.design,
            hyper_dirs: computed.hyper_dirs,
            radial_distances,
            basis_derivative_jets,
        });
        if cacheable {
            insert_persistent_latent_design(Arc::clone(&entry))?;
        }
        self.insert(entry, id);
        self.lookup_inserted(id)
    }

    fn find_entry(
        &mut self,
        latent: &LatentCoordValues,
        basis_digest: CacheDigest,
        latent_metadata_digest: CacheDigest,
    ) -> Option<usize> {
        self.entries.iter().position(|entry| {
            entry.cached.cacheable
                && entry.cached.basis_digest == basis_digest
                && entry.cached.latent_metadata_digest == latent_metadata_digest
                && entry.cached.latent_id == latent.latent_id()
                && latent_bits_match(latent, &entry.cached.latent_bits)
        })
    }

    fn lookup_inserted(&self, id: u64) -> Result<LatentDesignLookup<'_>, EstimationError> {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return Err(EstimationError::InvalidInput(
                "inserted latent design cache entry missing".to_string(),
            ));
        };
        Ok(LatentDesignLookup {
            cached: self.entries[index].cached.as_ref(),
            entry_id: self.entries[index].id,
        })
    }

    fn insert(&mut self, cached: Arc<CachedDesign>, id: u64) {
        self.entries.push(LatentDesignCacheEntry {
            id,
            cached,
            last_used: self.clock,
            iteration: self.iteration,
        });
        while self.entries.len() > self.capacity {
            if let Some(evict_index) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| (entry.last_used, entry.iteration))
                .map(|(index, _)| index)
            {
                self.entries.remove(evict_index);
            } else {
                break;
            }
        }
    }
}

impl CachedDesign {
    fn resident_byte_count(&self) -> usize {
        self.resident_scalar_count()
            .saturating_mul(std::mem::size_of::<f64>())
            .saturating_add(
                self.hyper_dirs
                    .iter()
                    .map(DirectionalHyperParam::resident_byte_count)
                    .sum::<usize>(),
            )
    }

    fn resident_scalar_count(&self) -> usize {
        let mut count = self
            .design
            .design
            .nrows()
            .saturating_mul(self.design.design.ncols());
        count = count.saturating_add(
            self.design
                .coefficient_lower_bounds
                .as_ref()
                .map_or(0, |values| values.len()),
        );
        count = count.saturating_add(self.radial_distances.squared.len());
        count = count.saturating_add(self.radial_distances.distance.len());
        count = count.saturating_add(
            self.basis_derivative_jets
                .phi
                .as_ref()
                .map_or(0, |values| values.len()),
        );
        count = count.saturating_add(
            self.basis_derivative_jets
                .q
                .as_ref()
                .map_or(0, |values| values.len()),
        );
        count = count.saturating_add(
            self.basis_derivative_jets
                .t
                .as_ref()
                .map_or(0, |values| values.len()),
        );
        count = count.saturating_add(
            self.basis_derivative_jets
                .phi_r
                .as_ref()
                .map_or(0, |values| values.len()),
        );
        count = count.saturating_add(
            self.basis_derivative_jets
                .phi_rr
                .as_ref()
                .map_or(0, |values| values.len()),
        );
        count.saturating_add(usize::from(self.basis_derivative_jets.operator_resident))
    }
}

fn lookup_persistent_latent_design(
    latent: &LatentCoordValues,
    basis_digest: CacheDigest,
    latent_metadata_digest: CacheDigest,
    fingerprint: &LatentFingerprint,
) -> Result<Option<Arc<CachedDesign>>, EstimationError> {
    let cache = PERSISTENT_LATENT_DESIGN_CACHE
        .get_or_init(|| Mutex::new(PersistentLatentDesignCache::default()));
    let mut guard = cache.lock().map_err(|_| {
        EstimationError::InvalidInput("persistent latent design cache mutex poisoned".to_string())
    })?;
    guard.lookup(latent, basis_digest, latent_metadata_digest, fingerprint)
}

fn insert_persistent_latent_design(cached: Arc<CachedDesign>) -> Result<(), EstimationError> {
    let cache = PERSISTENT_LATENT_DESIGN_CACHE
        .get_or_init(|| Mutex::new(PersistentLatentDesignCache::default()));
    let mut guard = cache.lock().map_err(|_| {
        EstimationError::InvalidInput("persistent latent design cache mutex poisoned".to_string())
    })?;
    guard.insert(cached);
    Ok(())
}

fn latent_bits(latent: &LatentCoordValues) -> Arc<[u64]> {
    latent
        .as_flat()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>()
        .into()
}

fn latent_bits_match(latent: &LatentCoordValues, cached_bits: &[u64]) -> bool {
    latent.as_flat().len() == cached_bits.len()
        && latent
            .as_flat()
            .iter()
            .zip(cached_bits.iter())
            .all(|(value, bits)| value.to_bits() == *bits)
}

fn build_radial_distances(
    latent: &LatentCoordValues,
    centers: &Array2<f64>,
) -> Result<RadialDistanceMatrices, EstimationError> {
    let t = latent.as_matrix();
    if t.ncols() != centers.ncols() {
        return Err(EstimationError::InvalidInput(format!(
            "latent design cache center dimension mismatch: latent d={}, centers d={}",
            t.ncols(),
            centers.ncols()
        )));
    }
    let mut squared = Array2::<f64>::zeros((t.nrows(), centers.nrows()));
    let mut distance = Array2::<f64>::zeros((t.nrows(), centers.nrows()));
    for row in 0..t.nrows() {
        for center in 0..centers.nrows() {
            let mut r2 = 0.0_f64;
            for axis in 0..t.ncols() {
                let delta = t[[row, axis]] - centers[[center, axis]];
                r2 += delta * delta;
            }
            squared[[row, center]] = r2;
            distance[[row, center]] = r2.sqrt();
        }
    }
    Ok(RadialDistanceMatrices { squared, distance })
}

fn build_basis_derivative_jets(
    latent: &LatentCoordValues,
    basis_kind: &LatentBasisKind,
    distances: &RadialDistanceMatrices,
) -> Result<BasisDerivativeJets, EstimationError> {
    match basis_kind {
        LatentBasisKind::Matern {
            length_scale,
            nu,
            chunk_size,
            ..
        } => {
            if chunk_size.is_some() {
                drop(latent);
                drop(distances);
                return Ok(BasisDerivativeJets {
                    operator_resident: true,
                    ..BasisDerivativeJets::empty()
                });
            }
            let radial = RadialScalarKind::Matern {
                length_scale: *length_scale,
                nu: *nu,
            };
            let mut phi = Array2::<f64>::zeros(distances.distance.raw_dim());
            let mut q = Array2::<f64>::zeros(distances.distance.raw_dim());
            let mut t = Array2::<f64>::zeros(distances.distance.raw_dim());
            for row in 0..distances.distance.nrows() {
                for center in 0..distances.distance.ncols() {
                    let (phi_value, q_value, t_value) = radial
                        .eval_design_triplet(distances.distance[[row, center]])
                        .map_err(EstimationError::from)?;
                    phi[[row, center]] = phi_value;
                    q[[row, center]] = q_value;
                    t[[row, center]] = t_value;
                }
            }
            Ok(BasisDerivativeJets {
                phi: Some(phi),
                q: Some(q),
                t: Some(t),
                phi_r: None,
                phi_rr: None,
                operator_resident: false,
            })
        }
        LatentBasisKind::Duchon { .. } => {
            drop(latent);
            Ok(BasisDerivativeJets {
                operator_resident: true,
                ..BasisDerivativeJets::empty()
            })
        }
        LatentBasisKind::Sphere { .. }
        | LatentBasisKind::PeriodicBspline { .. }
        | LatentBasisKind::Pca { .. }
        | LatentBasisKind::TensorBspline { .. } => {
            drop(latent);
            drop(distances);
            Ok(BasisDerivativeJets {
                operator_resident: true,
                ..BasisDerivativeJets::empty()
            })
        }
    }
}
