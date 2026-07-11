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

use ndarray::{Array1, Array2, ArrayView2};

use crate::estimate::EstimationError;
use crate::estimate::reml::DirectionalHyperParam;
pub use gam_problem::LatentRetractionRegistry;
use gam_runtime::warm_start::{Fingerprint, Fingerprinter};
use gam_terms::basis::{DuchonNullspaceOrder, MaternNu, RadialScalarKind};
use gam_terms::latent::{
    AuxPriorFamily, AuxPriorStrength, LatentCoordValues, LatentIdMode, LatentManifold,
};
use gam_terms::smooth::{TermCollectionDesign, TermCollectionSpec};

const DEFAULT_LATENT_CACHE_CAPACITY: usize = 4;
const DEFAULT_PERSISTENT_LATENT_CACHE_CAPACITY: usize = 16;
const DEFAULT_PERSISTENT_LATENT_CACHE_BYTE_BUDGET: usize = 1024 * 1024 * 1024;

static PERSISTENT_LATENT_DESIGN_CACHE: OnceLock<Mutex<PersistentLatentDesignCache>> =
    OnceLock::new();

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

pub type CacheDigest = Fingerprint;

/// Open a [`Fingerprinter`] pre-seeded with a length-prefixed namespace
/// string so different cache-digest call sites cannot alias.
///
/// This is a thin convenience wrapper over [`Fingerprinter::write_str`]
/// — it exists so the call sites read as `cache_digest_builder("…-v1")`
/// instead of repeating the namespace-framing pattern at every callsite.
fn cache_digest_builder(namespace: &str) -> Fingerprinter {
    let mut out = Fingerprinter::new();
    out.write_str(namespace);
    out
}

#[derive(Clone)]
pub enum LatentBasisKind {
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
        power: f64,
        nullspace_order: DuchonNullspaceOrder,
        aniso_log_scales: Vec<f64>,
    },
    Sphere {
        centers: Array2<f64>,
        penalty_order: usize,
        chunk_size: Option<usize>,
    },
    PeriodicBspline {
        domain_start: f64,
        period: f64,
        degree: usize,
        num_basis: usize,
        chunk_size: Option<usize>,
    },
    TensorBspline {
        knots: Vec<Array1<f64>>,
        degrees: Vec<usize>,
        chunk_size: Option<usize>,
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
            } | Self::Sphere {
                chunk_size: Some(_),
                ..
            }
        )
    }

    fn cache_digest(&self) -> CacheDigest {
        let mut hasher = cache_digest_builder("latent-basis-v1");
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
                hasher.write_f64_slice(aniso_log_scales);
                hash_optional_usize(*chunk_size, &mut hasher);
                hasher.write_f64_array2(centers);
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
                hasher.write_u64(power.to_bits());
                hash_duchon_nullspace_order(*nullspace_order, &mut hasher);
                hasher.write_f64_slice(aniso_log_scales);
                hasher.write_f64_array2(centers);
            }
            Self::Sphere {
                centers,
                penalty_order,
                chunk_size,
            } => {
                hasher.write_usize(2);
                hasher.write_usize(centers.nrows());
                hasher.write_usize(centers.ncols());
                hasher.write_usize(*penalty_order);
                hash_optional_usize(*chunk_size, &mut hasher);
                hasher.write_f64_array2(centers);
            }
            Self::PeriodicBspline {
                domain_start,
                period,
                degree,
                num_basis,
                chunk_size,
            } => {
                hasher.write_usize(3);
                hasher.write_f64(*domain_start);
                hasher.write_f64(*period);
                hasher.write_usize(*degree);
                hasher.write_usize(*num_basis);
                hash_optional_usize(*chunk_size, &mut hasher);
            }
            Self::TensorBspline {
                knots,
                degrees,
                chunk_size,
            } => {
                hasher.write_usize(4);
                hasher.write_usize(degrees.len());
                for &degree in degrees {
                    hasher.write_usize(degree);
                }
                hash_optional_usize(*chunk_size, &mut hasher);
                hasher.write_usize(knots.len());
                for axis_knots in knots {
                    hasher.write_f64_array1(axis_knots);
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
                    hasher.write_bytes(path.to_string_lossy().as_bytes());
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
                hasher.write_f64_array2(basis_matrix);
            }
        }
        hasher.finalize()
    }
}

pub fn pca_center_mean_fingerprint(mean: &Array1<f64>) -> u64 {
    let mut hasher = Fingerprinter::new();
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

fn hash_duchon_nullspace_order(order: DuchonNullspaceOrder, hasher: &mut Fingerprinter) {
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

fn hash_optional_f64(value: Option<f64>, hasher: &mut Fingerprinter) {
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

fn hash_optional_usize(value: Option<usize>, hasher: &mut Fingerprinter) {
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

fn latent_metadata_cache_digest(latent: &LatentCoordValues) -> CacheDigest {
    let mut hasher = cache_digest_builder("latent-cache-metadata-v1");
    hasher.write_usize(latent.n_obs());
    hasher.write_usize(latent.latent_dim());
    hash_latent_manifold(latent.manifold(), &mut hasher);
    hash_latent_id_mode(latent.id_mode(), &mut hasher);
    hasher.finalize()
}

pub fn latent_design_context_cache_digest(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    term_index: gam_problem::SmoothTermIdx,
    analytic_rho_count: usize,
    feature_cols: &[usize],
) -> Result<CacheDigest, EstimationError> {
    let mut hasher = cache_digest_builder("latent-design-context-v1");
    hasher.write_usize(data.nrows());
    hasher.write_usize(data.ncols());
    for row in 0..data.nrows() {
        for col in 0..data.ncols() {
            hasher.write_f64(data[[row, col]]);
        }
    }
    let spec_bytes = serde_json::to_vec(spec).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "failed to serialize latent design cache context: {err}"
        ))
    })?;
    hasher.write_usize(spec_bytes.len());
    hasher.write_bytes(&spec_bytes);
    hasher.write_usize(term_index.get());
    hasher.write_usize(analytic_rho_count);
    hasher.write_usize(feature_cols.len());
    for &col in feature_cols {
        hasher.write_usize(col);
    }
    Ok(hasher.finalize())
}

fn hash_latent_id_mode(id_mode: &LatentIdMode, hasher: &mut Fingerprinter) {
    match id_mode {
        LatentIdMode::AuxPrior {
            u,
            family,
            strength,
        } => {
            hasher.write_usize(0);
            hasher.write_f64_array2(u);
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
            hasher.write_f64_array2(u);
            hash_aux_prior_family(*family, hasher);
            hash_aux_prior_strength(*strength, hasher);
            hash_optional_vector(init_log_precision.as_ref(), hasher);
        }
        LatentIdMode::DimSelection { init_log_precision } => {
            hasher.write_usize(2);
            hash_optional_vector(init_log_precision.as_ref(), hasher);
        }
        LatentIdMode::IsometryToReference {
            reference,
            strength,
        } => {
            hasher.write_usize(5);
            hasher.write_f64_array2(reference);
            hash_aux_prior_strength(*strength, hasher);
        }
        LatentIdMode::AuxOutcome {
            head,
            init_log_precision,
        } => {
            hasher.write_usize(4);
            hash_behavioral_head(head, hasher);
            hash_optional_vector(init_log_precision.as_ref(), hasher);
        }
        LatentIdMode::None => {
            hasher.write_usize(3);
        }
    }
}

fn hash_behavioral_head(
    head: &gam_terms::decoders::behavioral_head::BehavioralHead,
    hasher: &mut Fingerprinter,
) {
    use gam_terms::decoders::behavioral_head::AuxOutcomeFamily;
    match head.family() {
        AuxOutcomeFamily::Binomial => hasher.write_usize(0),
        AuxOutcomeFamily::Multinomial { n_classes } => {
            hasher.write_usize(1);
            hasher.write_usize(n_classes);
        }
    }
    hasher.write_usize(head.n_obs());
    hasher.write_f64(head.effective_labeled_count());
}

fn hash_aux_prior_family(family: AuxPriorFamily, hasher: &mut Fingerprinter) {
    hasher.write_usize(match family {
        AuxPriorFamily::Ridge => 0,
        AuxPriorFamily::Linear => 1,
    });
}

fn hash_aux_prior_strength(strength: AuxPriorStrength, hasher: &mut Fingerprinter) {
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

fn hash_optional_vector(vector: Option<&Array1<f64>>, hasher: &mut Fingerprinter) {
    match vector {
        Some(vector) => {
            hasher.write_bool(true);
            hasher.write_f64_array1(vector);
        }
        None => {
            hasher.write_bool(false);
        }
    }
}

fn hash_latent_manifold(manifold: &LatentManifold, hasher: &mut Fingerprinter) {
    match manifold {
        LatentManifold::Euclidean => {
            hasher.write_usize(0);
        }
        LatentManifold::Circle { period } => {
            hasher.write_usize(1);
            hasher.write_f64(*period);
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
            hasher.write_f64_slice(weights);
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
pub struct CachedDesign {
    pub(crate) latent_id: u64,
    pub(crate) fingerprint: LatentFingerprint,
    basis_digest: CacheDigest,
    latent_metadata_digest: CacheDigest,
    design_context_digest: CacheDigest,
    latent_bits: Arc<[u64]>,
    cacheable: bool,
    pub design: TermCollectionDesign,
    pub hyper_dirs: Vec<DirectionalHyperParam>,
    pub(crate) radial_distances: RadialDistanceMatrices,
    pub(crate) basis_derivative_jets: BasisDerivativeJets,
}

pub struct ComputedLatentDesign {
    pub design: TermCollectionDesign,
    pub hyper_dirs: Vec<DirectionalHyperParam>,
}

pub struct LatentDesignLookup<'a> {
    pub cached: &'a CachedDesign,
    pub entry_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PersistentLatentDesignKey {
    latent_id: u64,
    flat_hash: u64,
    basis_digest: CacheDigest,
    latent_metadata_digest: CacheDigest,
    design_context_digest: CacheDigest,
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
        design_context_digest: CacheDigest,
        fingerprint: &LatentFingerprint,
    ) -> Result<Option<Arc<CachedDesign>>, EstimationError> {
        let key = PersistentLatentDesignKey {
            latent_id: latent.latent_id(),
            flat_hash: fingerprint.hash,
            basis_digest,
            latent_metadata_digest,
            design_context_digest,
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
            && cached.design_context_digest == design_context_digest
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
            design_context_digest: cached.design_context_digest,
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

pub struct LatentDesignCache {
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

    pub fn invalidate(&mut self) {
        self.entries.clear();
    }

    pub fn invalidate_all(&mut self) {
        self.entries.clear();
        self.clock = self.clock.wrapping_add(1);
        self.iteration = self.iteration.wrapping_add(1);
    }

    pub fn lookup_or_compute<F>(
        &mut self,
        latent: Arc<LatentCoordValues>,
        basis_kind: LatentBasisKind,
        design_context_digest: CacheDigest,
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
        if cacheable
            && let Some(index) = self.find_entry(
                &latent,
                basis_digest,
                latent_metadata_digest,
                design_context_digest,
            )
        {
            self.entries[index].last_used = self.clock;
            return Ok(LatentDesignLookup {
                cached: self.entries[index].cached.as_ref(),
                entry_id: self.entries[index].id,
            });
        }
        if cacheable
            && let Some(cached) = lookup_persistent_latent_design(
                &latent,
                basis_digest,
                latent_metadata_digest,
                design_context_digest,
                &fingerprint,
            )?
        {
            let id = self.next_entry_id;
            self.next_entry_id = self.next_entry_id.wrapping_add(1);
            self.insert(cached, id);
            return self.lookup_inserted(id);
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
        let basis_derivative_jets = build_basis_derivative_jets(&basis_kind, &radial_distances)?;
        let id = self.next_entry_id;
        self.next_entry_id = self.next_entry_id.wrapping_add(1);
        let entry = Arc::new(CachedDesign {
            latent_id: latent.latent_id(),
            fingerprint,
            basis_digest,
            latent_metadata_digest,
            design_context_digest,
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
        design_context_digest: CacheDigest,
    ) -> Option<usize> {
        self.entries.iter().position(|entry| {
            entry.cached.cacheable
                && entry.cached.basis_digest == basis_digest
                && entry.cached.latent_metadata_digest == latent_metadata_digest
                && entry.cached.design_context_digest == design_context_digest
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
    design_context_digest: CacheDigest,
    fingerprint: &LatentFingerprint,
) -> Result<Option<Arc<CachedDesign>>, EstimationError> {
    let cache = PERSISTENT_LATENT_DESIGN_CACHE
        .get_or_init(|| Mutex::new(PersistentLatentDesignCache::default()));
    let mut guard = cache.lock().map_err(|_| {
        EstimationError::InvalidInput("persistent latent design cache mutex poisoned".to_string())
    })?;
    guard.lookup(
        latent,
        basis_digest,
        latent_metadata_digest,
        design_context_digest,
        fingerprint,
    )
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
        LatentBasisKind::Duchon { .. } => Ok(BasisDerivativeJets {
            operator_resident: true,
            ..BasisDerivativeJets::empty()
        }),
        LatentBasisKind::Sphere { .. }
        | LatentBasisKind::PeriodicBspline { .. }
        | LatentBasisKind::Pca { .. }
        | LatentBasisKind::TensorBspline { .. } => Ok(BasisDerivativeJets {
            operator_resident: true,
            ..BasisDerivativeJets::empty()
        }),
    }
}
