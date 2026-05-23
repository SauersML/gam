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
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array1, Array2};

use crate::basis::{DuchonNullspaceOrder, MaternNu, RadialScalarKind};
use crate::estimate::EstimationError;
use crate::estimate::reml::DirectionalHyperParam;
use crate::terms::latent_coord::LatentCoordValues;
use crate::terms::smooth::TermCollectionDesign;

const DEFAULT_LATENT_CACHE_CAPACITY: usize = 4;
const DEFAULT_PERSISTENT_LATENT_CACHE_CAPACITY: usize = 16;
const DISABLE_PERSISTENT_LATENT_CACHE_ENV: &str = "GAMFIT_DISABLE_PERSISTENT_LATENT_CACHE";

static PERSISTENT_LATENT_DESIGN_CACHE: OnceLock<Mutex<PersistentLatentDesignCache>> =
    OnceLock::new();

/// O(N) identity summary for a flat latent-coordinate vector.
#[derive(Clone, Debug)]
pub(crate) struct LatentFingerprint {
    pub(crate) hash: u64,
    pub(crate) len: usize,
    pub(crate) iteration: u64,
}

impl LatentFingerprint {
    pub(crate) fn from_flat(flat: &[f64], iteration: u64) -> Self {
        let mut hasher = DefaultHasher::new();
        flat.len().hash(&mut hasher);
        for &value in flat {
            value.to_bits().hash(&mut hasher);
        }
        Self {
            hash: hasher.finish(),
            len: flat.len(),
            iteration,
        }
    }
}

#[derive(Clone)]
pub(crate) enum LatentBasisKind {
    Matern {
        centers: Array2<f64>,
        length_scale: f64,
        nu: MaternNu,
    },
    Duchon {
        centers: Array2<f64>,
        length_scale: Option<f64>,
        nullspace_order: DuchonNullspaceOrder,
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
}

impl LatentBasisKind {
    fn centers(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Matern { centers, .. }
            | Self::Duchon { centers, .. }
            | Self::Sphere { centers, .. } => Some(centers),
            Self::PeriodicBspline { .. } | Self::TensorBspline { .. } => None,
        }
    }

    fn signature(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        match self {
            Self::Matern {
                centers,
                length_scale,
                nu,
            } => {
                0_u8.hash(&mut hasher);
                centers.nrows().hash(&mut hasher);
                centers.ncols().hash(&mut hasher);
                length_scale.to_bits().hash(&mut hasher);
                std::mem::discriminant(nu).hash(&mut hasher);
                hash_matrix(centers, &mut hasher);
            }
            Self::Duchon {
                centers,
                length_scale,
                nullspace_order,
            } => {
                1_u8.hash(&mut hasher);
                centers.nrows().hash(&mut hasher);
                centers.ncols().hash(&mut hasher);
                length_scale.map(f64::to_bits).hash(&mut hasher);
                nullspace_order.hash(&mut hasher);
                hash_matrix(centers, &mut hasher);
            }
            Self::Sphere {
                centers,
                penalty_order,
            } => {
                2_u8.hash(&mut hasher);
                centers.nrows().hash(&mut hasher);
                centers.ncols().hash(&mut hasher);
                penalty_order.hash(&mut hasher);
                hash_matrix(centers, &mut hasher);
            }
            Self::PeriodicBspline {
                domain_start,
                period,
                degree,
                num_basis,
            } => {
                3_u8.hash(&mut hasher);
                domain_start.to_bits().hash(&mut hasher);
                period.to_bits().hash(&mut hasher);
                degree.hash(&mut hasher);
                num_basis.hash(&mut hasher);
            }
            Self::TensorBspline { knots, degrees } => {
                4_u8.hash(&mut hasher);
                degrees.hash(&mut hasher);
                knots.len().hash(&mut hasher);
                for axis_knots in knots {
                    hash_vector(axis_knots, &mut hasher);
                }
            }
        }
        hasher.finish()
    }
}

fn hash_matrix(matrix: &Array2<f64>, hasher: &mut DefaultHasher) {
    for &value in matrix.iter() {
        let normalized = if value == 0.0 { 0.0 } else { value };
        normalized.to_bits().hash(hasher);
    }
}

fn hash_vector(vector: &Array1<f64>, hasher: &mut DefaultHasher) {
    vector.len().hash(hasher);
    for &value in vector.iter() {
        let normalized = if value == 0.0 { 0.0 } else { value };
        normalized.to_bits().hash(hasher);
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
    pub(crate) id: u64,
    pub(crate) latent_id: u64,
    pub(crate) fingerprint: LatentFingerprint,
    pub(crate) basis_signature: u64,
    latent_bits: Arc<[u64]>,
    pub(crate) design: TermCollectionDesign,
    pub(crate) hyper_dirs: Vec<DirectionalHyperParam>,
    pub(crate) radial_distances: RadialDistanceMatrices,
    pub(crate) basis_derivative_jets: BasisDerivativeJets,
    last_used: u64,
}

pub(crate) struct ComputedLatentDesign {
    pub(crate) design: TermCollectionDesign,
    pub(crate) hyper_dirs: Vec<DirectionalHyperParam>,
}

pub(crate) struct LatentDesignLookup<'a> {
    pub(crate) cached: &'a CachedDesign,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PersistentLatentDesignKey {
    latent_id: u64,
    flat_hash: u64,
    basis_signature: u64,
}

struct PersistentLatentDesignEntry {
    fingerprint: LatentFingerprint,
    cached: Arc<CachedDesign>,
}

pub(crate) struct PersistentLatentDesignCache {
    entries: HashMap<PersistentLatentDesignKey, PersistentLatentDesignEntry>,
    lru: VecDeque<PersistentLatentDesignKey>,
    capacity: usize,
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
        }
    }

    pub(crate) fn lookup(
        &mut self,
        latent: &LatentCoordValues,
        basis_signature: u64,
        fingerprint: &LatentFingerprint,
    ) -> Result<Option<Arc<CachedDesign>>, EstimationError> {
        let key = PersistentLatentDesignKey {
            latent_id: latent.latent_id(),
            flat_hash: fingerprint.hash,
            basis_signature,
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
            && latent_bits_match(latent, &cached.latent_bits)
        {
            return Ok(Some(cached));
        }
        Ok(None)
    }

    pub(crate) fn insert(&mut self, cached: Arc<CachedDesign>) {
        let key = PersistentLatentDesignKey {
            latent_id: cached.latent_id,
            flat_hash: cached.fingerprint.hash,
            basis_signature: cached.basis_signature,
        };
        let entry = PersistentLatentDesignEntry {
            fingerprint: cached.fingerprint.clone(),
            cached,
        };
        self.entries.insert(key, entry);
        self.touch(key);
        while self.entries.len() > self.capacity {
            let Some(evicted) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&evicted);
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
    entries: Vec<CachedDesign>,
    capacity: usize,
    clock: u64,
    iteration: u64,
    next_entry_id: u64,
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
        let fingerprint = LatentFingerprint::from_flat(flat_slice, self.iteration);
        let basis_signature = basis_kind.signature();
        if let Some(index) = self.find_entry(&latent, basis_signature) {
            self.entries[index].last_used = self.clock;
            return Ok(LatentDesignLookup {
                cached: &self.entries[index],
            });
        }
        if let Some(cached) = lookup_persistent_latent_design(&latent, basis_signature, &fingerprint)?
        {
            let id = self.next_entry_id;
            self.next_entry_id = self.next_entry_id.wrapping_add(1);
            let mut entry = (*cached).clone();
            entry.id = id;
            entry.fingerprint.iteration = self.iteration;
            entry.last_used = self.clock;
            self.insert(entry);
            return self.lookup_inserted(id);
        }

        let computed = compute()?;
        let radial_distances = match basis_kind.centers() {
            Some(centers) => build_radial_distances(&latent, centers)?,
            None => RadialDistanceMatrices {
                squared: Array2::<f64>::zeros((0, 0)),
                distance: Array2::<f64>::zeros((0, 0)),
            },
        };
        let basis_derivative_jets =
            build_basis_derivative_jets(&latent, &basis_kind, &radial_distances)?;
        let id = self.next_entry_id;
        self.next_entry_id = self.next_entry_id.wrapping_add(1);
        let entry = CachedDesign {
            id,
            latent_id: latent.latent_id(),
            fingerprint,
            basis_signature,
            latent_bits: latent_bits(&latent),
            design: computed.design,
            hyper_dirs: computed.hyper_dirs,
            radial_distances,
            basis_derivative_jets,
            last_used: self.clock,
        };
        let _resident_scalars = entry.resident_scalar_count();
        insert_persistent_latent_design(Arc::new(entry.clone()))?;
        self.insert(entry);
        self.lookup_inserted(id)
    }

    fn find_entry(&mut self, latent: &LatentCoordValues, basis_signature: u64) -> Option<usize> {
        self.entries.iter().position(|entry| {
            entry.basis_signature == basis_signature
                && entry.latent_id == latent.latent_id()
                && latent_bits_match(latent, &entry.latent_bits)
        })
    }

    fn lookup_inserted(&self, id: u64) -> Result<LatentDesignLookup<'_>, EstimationError> {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return Err(EstimationError::InvalidInput(
                "inserted latent design cache entry missing".to_string(),
            ));
        };
        Ok(LatentDesignLookup {
            cached: &self.entries[index],
        })
    }

    fn insert(&mut self, entry: CachedDesign) {
        self.entries.push(entry);
        while self.entries.len() > self.capacity {
            if let Some(evict_index) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| (entry.last_used, entry.fingerprint.iteration))
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
    fn resident_scalar_count(&self) -> usize {
        self.radial_distances.squared.len()
            + self.radial_distances.distance.len()
            + self
                .basis_derivative_jets
                .phi
                .as_ref()
                .map_or(0, |values| values.len())
            + self
                .basis_derivative_jets
                .q
                .as_ref()
                .map_or(0, |values| values.len())
            + self
                .basis_derivative_jets
                .t
                .as_ref()
                .map_or(0, |values| values.len())
            + self
                .basis_derivative_jets
                .phi_r
                .as_ref()
                .map_or(0, |values| values.len())
            + self
                .basis_derivative_jets
                .phi_rr
                .as_ref()
                .map_or(0, |values| values.len())
            + usize::from(self.basis_derivative_jets.operator_resident)
    }
}

fn lookup_persistent_latent_design(
    latent: &LatentCoordValues,
    basis_signature: u64,
    fingerprint: &LatentFingerprint,
) -> Result<Option<Arc<CachedDesign>>, EstimationError> {
    if persistent_latent_cache_disabled() {
        return Ok(None);
    }
    let cache = PERSISTENT_LATENT_DESIGN_CACHE
        .get_or_init(|| Mutex::new(PersistentLatentDesignCache::default()));
    let mut guard = cache.lock().map_err(|_| {
        EstimationError::InvalidInput("persistent latent design cache mutex poisoned".to_string())
    })?;
    guard.lookup(latent, basis_signature, fingerprint)
}

fn insert_persistent_latent_design(cached: Arc<CachedDesign>) -> Result<(), EstimationError> {
    if persistent_latent_cache_disabled() {
        return Ok(());
    }
    let cache = PERSISTENT_LATENT_DESIGN_CACHE
        .get_or_init(|| Mutex::new(PersistentLatentDesignCache::default()));
    let mut guard = cache.lock().map_err(|_| {
        EstimationError::InvalidInput("persistent latent design cache mutex poisoned".to_string())
    })?;
    guard.insert(cached);
    Ok(())
}

fn persistent_latent_cache_disabled() -> bool {
    std::env::var(DISABLE_PERSISTENT_LATENT_CACHE_ENV)
        .map(|value| value == "1")
        .unwrap_or(false)
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
            length_scale, nu, ..
        } => {
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
            let _ = latent;
            Ok(BasisDerivativeJets {
                operator_resident: true,
                ..BasisDerivativeJets::empty()
            })
        }
        LatentBasisKind::Sphere { .. }
        | LatentBasisKind::PeriodicBspline { .. }
        | LatentBasisKind::TensorBspline { .. } => {
            let _ = latent;
            let _ = distances;
            Ok(BasisDerivativeJets {
                operator_resident: true,
                ..BasisDerivativeJets::empty()
            })
        }
    }
}
