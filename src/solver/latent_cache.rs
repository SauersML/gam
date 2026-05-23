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
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ndarray::Array2;

use crate::basis::{DuchonNullspaceOrder, MaternNu, RadialScalarKind};
use crate::estimate::EstimationError;
use crate::estimate::reml::DirectionalHyperParam;
use crate::terms::latent_coord::LatentCoordValues;
use crate::terms::smooth::TermCollectionDesign;

const DEFAULT_LATENT_CACHE_CAPACITY: usize = 4;
const DEFAULT_RELATIVE_L2_TOLERANCE: f64 = 1e-12;

/// O(N) identity summary for a flat latent-coordinate vector.
#[derive(Clone, Debug)]
pub(crate) struct LatentFingerprint {
    pub(crate) hash: u64,
    pub(crate) max_coordinate_norm: f64,
    pub(crate) l2_norm: f64,
    pub(crate) len: usize,
    pub(crate) iteration: u64,
}

impl LatentFingerprint {
    pub(crate) fn from_flat(flat: &[f64], iteration: u64) -> Self {
        let mut hasher = DefaultHasher::new();
        let mut max_coordinate_norm = 0.0_f64;
        let mut l2 = 0.0_f64;
        flat.len().hash(&mut hasher);
        for &value in flat {
            let normalized = if value == 0.0 { 0.0 } else { value };
            normalized.to_bits().hash(&mut hasher);
            max_coordinate_norm = max_coordinate_norm.max(value.abs());
            l2 += value * value;
        }
        Self {
            hash: hasher.finish(),
            max_coordinate_norm,
            l2_norm: l2.sqrt(),
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
}

impl LatentBasisKind {
    fn centers(&self) -> &Array2<f64> {
        match self {
            Self::Matern { centers, .. } | Self::Duchon { centers, .. } => centers,
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
    pub(crate) fingerprint: LatentFingerprint,
    pub(crate) latent_flat: Arc<[f64]>,
    pub(crate) basis_signature: u64,
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

pub(crate) struct LatentDesignCache {
    entries: Vec<CachedDesign>,
    capacity: usize,
    relative_l2_tolerance: f64,
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
            relative_l2_tolerance: DEFAULT_RELATIVE_L2_TOLERANCE,
            clock: 0,
            iteration: 0,
            next_entry_id: 0,
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.entries.clear();
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
        if let Some(index) = self.find_entry(flat_slice, &fingerprint, basis_signature) {
            self.entries[index].last_used = self.clock;
            return Ok(LatentDesignLookup {
                cached: &self.entries[index],
            });
        }

        let computed = compute()?;
        let radial_distances = build_radial_distances(&latent, basis_kind.centers())?;
        let basis_derivative_jets =
            build_basis_derivative_jets(&latent, &basis_kind, &radial_distances)?;
        let id = self.next_entry_id;
        self.next_entry_id = self.next_entry_id.wrapping_add(1);
        let entry = CachedDesign {
            id,
            fingerprint,
            latent_flat: Arc::from(flat.to_vec()),
            basis_signature,
            design: computed.design,
            hyper_dirs: computed.hyper_dirs,
            radial_distances,
            basis_derivative_jets,
            last_used: self.clock,
        };
        let _resident_scalars = entry.resident_scalar_count();
        self.insert(entry);
        let index = self
            .entries
            .iter()
            .position(|entry| entry.id == id)
            .expect("inserted latent design cache entry missing");
        Ok(LatentDesignLookup {
            cached: &self.entries[index],
        })
    }

    fn find_entry(
        &mut self,
        flat: &[f64],
        fingerprint: &LatentFingerprint,
        basis_signature: u64,
    ) -> Option<usize> {
        self.entries.iter().position(|entry| {
            entry.basis_signature == basis_signature
                && entry.fingerprint.len == fingerprint.len
                && (entry.fingerprint.hash == fingerprint.hash
                    || relative_l2(
                        flat,
                        &entry.latent_flat,
                        entry
                            .fingerprint
                            .l2_norm
                            .max(entry.fingerprint.max_coordinate_norm),
                    )
                        <= self.relative_l2_tolerance)
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

fn relative_l2(current: &[f64], cached: &[f64], cached_norm: f64) -> f64 {
    if current.len() != cached.len() {
        return f64::INFINITY;
    }
    let diff = current
        .iter()
        .zip(cached.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    diff / cached_norm.max(1.0)
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
    }
}
