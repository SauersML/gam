//! SAE-manifold term configuration.
//!
//! This is the formal Methodspace row from `proposals/sae_manifold.md`:
//!
//! ```text
//! Z_i ~= sum_k a_ik g_k(t_ik),     g_k(t) = Phi_k(t) B_k
//! ```
//!
//! Tier assignment:
//!
//! * beta: [`SaeManifoldAtom::decoder_coefficients`] (`B_k`, one block per atom).
//! * ext-coords: [`SaeAssignment`] (`logits -> a_ik` and per-atom
//!   `LatentCoordValues`). Per-row latent coordinates are written `t`; existing
//!   kernel-shape state remains with carriers such as `SpatialLogKappaCoords`.
//! * rho: [`SaeManifoldRho`] (`lambda_sparse`, `lambda_smooth`, `alpha_kj`) plus
//!   the discrete `K` selected by the Python `compare_models` wrapper.
//!
//! The per-row local block is exactly the audit-revised shape:
//!
//! ```text
//! ext_i = (logits_i[0..K], t_i0[0..d_0], ..., t_iK[0..d_K])
//! dim(ext_i) = K + sum_k d_k
//! ```
//!
//! [`SaeManifoldTerm::assemble_arrow_schur`] materializes the Gauss-Newton
//! bordered Hessian in that layout and hands it to
//! [`crate::solver::arrow_schur::ArrowSchurSystem`].

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayView4, s};
use std::sync::Arc;

use crate::solver::arrow_schur::{ArrowRowBlock, ArrowSchurError, ArrowSchurSystem};
use crate::terms::analytic_penalties::{
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry,
    IBPAssignmentPenalty, PenaltyTier, PsiSlice, SoftmaxAssignmentSparsityPenalty,
};
use crate::terms::latent_coord::{LatentCoordValues, LatentIdMode, LatentManifold};

const SAE_MANIFOLD_ARMIJO_C1: f64 = 1.0e-4;
const SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS: usize = 12;

/// Decay law for deterministic Gumbel/concrete assignment temperature.
#[derive(Debug, Clone)]
pub enum ScheduleKind {
    Geometric { rate: f64 },
    Linear { steps: usize },
    ReciprocalIter,
}

/// Outer-state temperature annealing for SAE assignment relaxations.
///
/// Annealing drives the continuous concrete/softmax assignment toward the
/// discrete argmax or IBP active-set solution while PIRLS solves smooth
/// positive-temperature subproblems. In the zero-floor limit, softmax becomes
/// argmax and the IBP-MAP sigmoid active set becomes exact; a positive
/// `tau_min` optimizes the corresponding near-discrete MAP problem.
#[derive(Debug, Clone)]
pub struct GumbelTemperatureSchedule {
    pub tau_start: f64,
    pub tau_min: f64,
    pub decay: ScheduleKind,
    pub iter_count: usize,
}

impl GumbelTemperatureSchedule {
    #[must_use = "build error must be handled"]
    pub fn new(tau_start: f64, tau_min: f64, decay: ScheduleKind) -> Result<Self, String> {
        let sched = Self {
            tau_start,
            tau_min,
            decay,
            iter_count: 0,
        };
        sched.validate()?;
        Ok(sched)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.tau_start.is_finite() && self.tau_start > 0.0) {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_start must be finite and positive; got {}",
                self.tau_start
            ));
        }
        if !(self.tau_min.is_finite() && self.tau_min > 0.0) {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_min must be finite and positive; got {}",
                self.tau_min
            ));
        }
        if self.tau_min > self.tau_start {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_min ({}) cannot exceed tau_start ({})",
                self.tau_min, self.tau_start
            ));
        }
        match self.decay {
            ScheduleKind::Geometric { rate } => {
                if !(rate.is_finite() && rate > 0.0 && rate < 1.0) {
                    return Err(format!(
                        "GumbelTemperatureSchedule::Geometric: rate must be in (0, 1); got {rate}"
                    ));
                }
            }
            ScheduleKind::Linear { steps } => {
                if steps == 0 {
                    return Err("GumbelTemperatureSchedule::Linear: steps must be positive".into());
                }
            }
            ScheduleKind::ReciprocalIter => {}
        }
        Ok(())
    }

    pub fn current_tau(&self, iter: usize) -> f64 {
        let raw = match self.decay {
            ScheduleKind::Geometric { rate } => self.tau_start * rate.powf(iter as f64),
            ScheduleKind::Linear { steps } => {
                if iter >= steps {
                    self.tau_min
                } else {
                    let frac = iter as f64 / steps as f64;
                    self.tau_start + frac * (self.tau_min - self.tau_start)
                }
            }
            ScheduleKind::ReciprocalIter => self.tau_start / (1.0 + iter as f64),
        };
        raw.max(self.tau_min)
    }

    pub fn step(&mut self) -> f64 {
        let tau = self.current_tau(self.iter_count);
        self.iter_count += 1;
        tau
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchStrategy {
    Fixed,
    ExponentialSweep { values: Vec<f64> },
}

impl SearchStrategy {
    #[must_use]
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed)
    }

    #[must_use]
    pub fn sweep_values(&self) -> Option<&[f64]> {
        match self {
            Self::Fixed => None,
            Self::ExponentialSweep { values } => Some(values),
        }
    }
}

/// Basis/topology tag for one SAE manifold atom.
///
/// The evaluated basis and input-location jet live on [`SaeManifoldAtom`].
/// This enum records the user-facing topology choice so downstream diagnostics
/// and Python wrappers can round-trip whether the atom was a Duchon patch,
/// periodic curve, sphere, or a caller-supplied precomputed basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaeAtomBasisKind {
    Duchon,
    Periodic,
    Sphere,
    Torus,
    EuclideanPatch,
    Precomputed(String),
}

impl SaeAtomBasisKind {
    fn latent_manifold(&self, latent_dim: usize) -> LatentManifold {
        match self {
            // `Periodic` uses [`PeriodicHarmonicEvaluator`], whose basis
            // functions are `cos(2π·h·t), sin(2π·h·t)` — i.e. `t` is a
            // fraction of one period, not radians. The latent manifold must
            // wrap modulo `1.0` to match this convention; wrapping modulo
            // `2π` (as `LatentManifold::Circle` does) would scramble the
            // fraction-of-period interpretation and cause #174-style
            // failures where Newton updates push `t` outside `[0, 1)` and
            // the optimiser sees a discontinuous landscape.
            Self::Periodic => {
                if latent_dim == 1 {
                    LatentManifold::Circle { period: 1.0 }
                } else {
                    LatentManifold::Product(
                        (0..latent_dim)
                            .map(|_| LatentManifold::Circle { period: 1.0 })
                            .collect(),
                    )
                }
            }
            // `Sphere` is parameterised via a (lat, lon) intrinsic chart; the
            // chart evaluator already enforces sphere geometry through its
            // cos/sin terms (in radians, multiplying lat/lon directly into
            // `sin`/`cos`), so the latent optimiser sees a 2-D product
            // manifold: lat is a bounded interval `[-π/2, π/2]` (clamped by
            // the chart) and lon is an `S^1` angle wrapped modulo `2π`.
            // Treating it as `LatentManifold::Sphere { dim: 2 }` would
            // require ambient unit-vectors of length 2 (impossible for S^2).
            Self::Sphere => LatentManifold::Product(vec![
                LatentManifold::Interval {
                    lo: -std::f64::consts::FRAC_PI_2,
                    hi: std::f64::consts::FRAC_PI_2,
                },
                LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                },
            ]),
            // `Torus` uses [`TorusHarmonicEvaluator`], which shares the
            // fraction-of-period convention with `PeriodicHarmonicEvaluator`
            // (basis is `cos(2π·h·t)`, `sin(2π·h·t)` on each axis). Each
            // per-axis latent wraps modulo `1.0`.
            Self::Torus => {
                if latent_dim == 1 {
                    LatentManifold::Circle { period: 1.0 }
                } else {
                    LatentManifold::Product(
                        (0..latent_dim)
                            .map(|_| LatentManifold::Circle { period: 1.0 })
                            .collect(),
                    )
                }
            }
            Self::Duchon | Self::EuclideanPatch | Self::Precomputed(_) => LatentManifold::Euclidean,
        }
    }
}

pub trait SaeBasisEvaluator: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String>;
}

/// Periodic harmonic basis evaluator for a single-dimensional circle latent.
///
/// Produces `M = 2*num_harmonics + 1` basis functions
/// `[1, sin(2π·1·t), cos(2π·1·t), …, sin(2π·H·t), cos(2π·H·t)]` where
/// `H = (M − 1) / 2`. The latent must have `latent_dim == 1`.
#[derive(Debug, Clone)]
pub struct PeriodicHarmonicEvaluator {
    pub num_basis: usize,
}

impl PeriodicHarmonicEvaluator {
    pub fn new(num_basis: usize) -> Result<Self, String> {
        if num_basis == 0 || num_basis % 2 == 0 {
            return Err(format!(
                "PeriodicHarmonicEvaluator requires odd num_basis >= 1; got {num_basis}"
            ));
        }
        Ok(Self { num_basis })
    }
}

impl SaeBasisEvaluator for PeriodicHarmonicEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            phi[[row, 0]] = 1.0;
            for h in 1..=num_harmonics {
                let angle = two_pi * (h as f64) * t;
                let s = angle.sin();
                let c = angle.cos();
                let s_idx = 2 * h - 1;
                let c_idx = 2 * h;
                phi[[row, s_idx]] = s;
                phi[[row, c_idx]] = c;
                jet[[row, s_idx, 0]] = two_pi * (h as f64) * c;
                jet[[row, c_idx, 0]] = -two_pi * (h as f64) * s;
            }
        }
        Ok((phi, jet))
    }
}

/// Raw-angle periodic evaluator for the minimal SAE-manifold front-end.
///
/// The basis is exactly `[cos(t), sin(t)]` with `t` measured in radians. If
/// the latent coordinate has more than one axis, the first axis carries the
/// circle phase and the remaining axes are left available to the optimizer but
/// do not enter this basis.
#[derive(Debug, Clone)]
pub struct RawPeriodicCircleEvaluator {
    pub latent_dim: usize,
}

impl RawPeriodicCircleEvaluator {
    pub fn new(latent_dim: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("RawPeriodicCircleEvaluator requires latent_dim >= 1".to_string());
        }
        Ok(Self { latent_dim })
    }
}

impl SaeBasisEvaluator for RawPeriodicCircleEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "RawPeriodicCircleEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = Array3::<f64>::zeros((n, 2, self.latent_dim));
        for row in 0..n {
            let t = coords[[row, 0]];
            phi[[row, 0]] = t.cos();
            phi[[row, 1]] = t.sin();
            jet[[row, 0, 0]] = -t.sin();
            jet[[row, 1, 0]] = t.cos();
        }
        Ok((phi, jet))
    }
}

/// Lat/lon sphere chart evaluator used by the Rust-owned minimal SAE path.
#[derive(Debug, Clone)]
pub struct SphereChartEvaluator;

impl SaeBasisEvaluator for SphereChartEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "SphereChartEvaluator expects latent_dim == 2, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, 7));
        let mut jet = Array3::<f64>::zeros((n, 7, 2));
        for row in 0..n {
            let raw_lat = coords[[row, 0]];
            let lat = raw_lat.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
            // The clamp truncates derivatives w.r.t. the raw input coordinate
            // outside the interior `(-π/2, π/2)`: in the saturated region the
            // phi entries are constant in `coords[[row, 0]]`, so the chain rule
            // contributes a zero factor on the `∂/∂coords[row,0]` axis. Failing
            // to apply this leaks a non-zero analytic gradient where finite
            // differences correctly report zero, sending Newton steps in lat
            // in a direction the loss does not actually decrease along.
            let lat_active =
                raw_lat > -std::f64::consts::FRAC_PI_2 && raw_lat < std::f64::consts::FRAC_PI_2;
            let chain_lat = if lat_active { 1.0 } else { 0.0 };
            let lon = coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            let x = clat * clon;
            let y = clat * slon;
            let z = slat;
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = x;
            phi[[row, 2]] = y;
            phi[[row, 3]] = z;
            phi[[row, 4]] = x * y;
            phi[[row, 5]] = y * z;
            phi[[row, 6]] = x * z;

            let dx_dlat = -slat * clon * chain_lat;
            let dx_dlon = -clat * slon;
            let dy_dlat = -slat * slon * chain_lat;
            let dy_dlon = clat * clon;
            let dz_dlat = clat * chain_lat;
            jet[[row, 1, 0]] = dx_dlat;
            jet[[row, 1, 1]] = dx_dlon;
            jet[[row, 2, 0]] = dy_dlat;
            jet[[row, 2, 1]] = dy_dlon;
            jet[[row, 3, 0]] = dz_dlat;
            jet[[row, 4, 0]] = dx_dlat * y + x * dy_dlat;
            jet[[row, 4, 1]] = dx_dlon * y + x * dy_dlon;
            jet[[row, 5, 0]] = dy_dlat * z + y * dz_dlat;
            jet[[row, 5, 1]] = dy_dlon * z;
            jet[[row, 6, 0]] = dx_dlat * z + x * dz_dlat;
            jet[[row, 6, 1]] = dx_dlon * z;
        }
        Ok((phi, jet))
    }
}

/// Tensor-product periodic harmonic evaluator for a `d`-dimensional torus
/// `T^d = (S^1)^d`. The basis is the tensor product over each axis of the
/// 1-D circle basis
/// `[1, cos(2π·1·t), sin(2π·1·t), …, cos(2π·H·t), sin(2π·H·t)]`
/// (each axis contributes `2H+1` factors, so the total basis size is
/// `(2H+1)^d`). The latent coords are angular phases in `[0, 1)` (consistent
/// with the periodic 1-D atoms).
#[derive(Debug, Clone)]
pub struct TorusHarmonicEvaluator {
    pub latent_dim: usize,
    pub num_harmonics: usize,
}

impl TorusHarmonicEvaluator {
    pub fn new(latent_dim: usize, num_harmonics: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("TorusHarmonicEvaluator requires latent_dim >= 1".to_string());
        }
        if num_harmonics == 0 {
            return Err("TorusHarmonicEvaluator requires num_harmonics >= 1".to_string());
        }
        Ok(Self {
            latent_dim,
            num_harmonics,
        })
    }

    pub fn axis_basis_size(&self) -> usize {
        2 * self.num_harmonics + 1
    }

    pub fn basis_size(&self) -> usize {
        // (2H+1)^d — computed iteratively to surface overflow.
        let axis_m = self.axis_basis_size();
        let mut total: usize = 1;
        for _ in 0..self.latent_dim {
            total = total
                .checked_mul(axis_m)
                .expect("TorusHarmonicEvaluator: basis size overflowed usize");
        }
        total
    }
}

impl SaeBasisEvaluator for TorusHarmonicEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "TorusHarmonicEvaluator: expected latent_dim {d}, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let axis_m = self.axis_basis_size();
        let m = self.basis_size();
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, d));
        // Per-axis evaluation buffer: phi_axis[axis][col] and dphi_axis[axis][col].
        let mut phi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut dphi_axis = vec![vec![0.0_f64; axis_m]; d];
        for row in 0..n {
            for axis in 0..d {
                let t = coords[[row, axis]];
                phi_axis[axis][0] = 1.0;
                dphi_axis[axis][0] = 0.0;
                for h in 1..=h_max {
                    let freq = two_pi * (h as f64);
                    let angle = freq * t;
                    let s = angle.sin();
                    let c = angle.cos();
                    let s_idx = 2 * h - 1;
                    let c_idx = 2 * h;
                    phi_axis[axis][s_idx] = s;
                    phi_axis[axis][c_idx] = c;
                    dphi_axis[axis][s_idx] = freq * c;
                    dphi_axis[axis][c_idx] = -freq * s;
                }
            }
            // Enumerate the Cartesian product of per-axis indices in
            // lexicographic order (axis 0 is the slowest).
            let mut idx = vec![0usize; d];
            for flat in 0..m {
                let mut val = 1.0_f64;
                for axis in 0..d {
                    val *= phi_axis[axis][idx[axis]];
                }
                phi[[row, flat]] = val;
                // ∂/∂coords[row, axis_target] = product over axes, replacing
                // phi_axis[axis_target] with its derivative.
                for axis_target in 0..d {
                    let mut deriv = 1.0_f64;
                    for axis in 0..d {
                        deriv *= if axis == axis_target {
                            dphi_axis[axis][idx[axis]]
                        } else {
                            phi_axis[axis][idx[axis]]
                        };
                    }
                    jet[[row, flat, axis_target]] = deriv;
                }
                // Increment lexicographic index (last axis fastest).
                for axis in (0..d).rev() {
                    idx[axis] += 1;
                    if idx[axis] < axis_m {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
        Ok((phi, jet))
    }
}

/// Affine Euclidean/Duchon fallback for the minimal fit entrypoint.
#[derive(Debug, Clone)]
pub struct AffineCoordinateEvaluator {
    pub latent_dim: usize,
}

impl AffineCoordinateEvaluator {
    pub fn new(latent_dim: usize) -> Self {
        Self { latent_dim }
    }
}

impl SaeBasisEvaluator for AffineCoordinateEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "AffineCoordinateEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let m = self.latent_dim + 1;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, self.latent_dim));
        phi.column_mut(0).fill(1.0);
        for row in 0..n {
            for axis in 0..self.latent_dim {
                phi[[row, axis + 1]] = coords[[row, axis]];
                jet[[row, axis + 1, axis]] = 1.0;
            }
        }
        Ok((phi, jet))
    }
}

/// Static basis evaluator: returns a frozen `(Phi, dPhi/dt)` snapshot regardless
/// of the supplied coordinates. Lets the multi-step Newton loop compose for
/// basis kinds whose true refresh routine is not yet wired in Rust — the
/// Newton step still updates logits, coordinates, and decoder β while the
/// basis design remains the caller-provided snapshot.
#[derive(Debug, Clone)]
pub struct StaticBasisEvaluator {
    pub phi: Array2<f64>,
    pub jet: Array3<f64>,
}

impl StaticBasisEvaluator {
    pub fn new(phi: Array2<f64>, jet: Array3<f64>) -> Result<Self, String> {
        let (n, m) = phi.dim();
        let jet_dim = jet.dim();
        if jet_dim.0 != n || jet_dim.1 != m {
            return Err(format!(
                "StaticBasisEvaluator: jet shape {:?} incompatible with phi shape {:?}",
                jet_dim,
                phi.dim()
            ));
        }
        Ok(Self { phi, jet })
    }
}

impl SaeBasisEvaluator for StaticBasisEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.nrows() != self.phi.nrows() {
            return Err(format!(
                "StaticBasisEvaluator expected {} rows, got {}",
                self.phi.nrows(),
                coords.nrows()
            ));
        }
        Ok((self.phi.clone(), self.jet.clone()))
    }
}

/// One manifold atom.
///
/// `basis_values` is `Phi_k(t_{ik})`, shape `(N, M_k)`.
/// `basis_jacobian` is `d Phi_k / d t_{ik}`, shape `(N, M_k, d_k)`.
/// `decoder_coefficients` is `B_k`, shape `(M_k, p)`.
/// `smooth_penalty` is `P_k`, shape `(M_k, M_k)`.
#[derive(Debug, Clone)]
pub struct SaeManifoldAtom {
    pub name: String,
    pub basis_kind: SaeAtomBasisKind,
    pub latent_dim: usize,
    pub basis_values: Array2<f64>,
    pub basis_jacobian: Array3<f64>,
    pub decoder_coefficients: Array2<f64>,
    pub smooth_penalty: Array2<f64>,
    pub basis_evaluator: Option<Arc<dyn SaeBasisEvaluator>>,
}

impl SaeManifoldAtom {
    #[must_use = "build error must be handled"]
    pub fn new(
        name: impl Into<String>,
        basis_kind: SaeAtomBasisKind,
        latent_dim: usize,
        basis_values: Array2<f64>,
        basis_jacobian: Array3<f64>,
        decoder_coefficients: Array2<f64>,
        smooth_penalty: Array2<f64>,
    ) -> Result<Self, String> {
        let n = basis_values.nrows();
        let m = basis_values.ncols();
        let p = decoder_coefficients.ncols();
        if basis_jacobian.dim() != (n, m, latent_dim) {
            return Err(format!(
                "SaeManifoldAtom::new: basis_jacobian must be ({n}, {m}, {latent_dim}); got {:?}",
                basis_jacobian.dim()
            ));
        }
        if decoder_coefficients.nrows() != m {
            return Err(format!(
                "SaeManifoldAtom::new: decoder rows {} must equal basis size {m}",
                decoder_coefficients.nrows()
            ));
        }
        if smooth_penalty.dim() != (m, m) {
            return Err(format!(
                "SaeManifoldAtom::new: smooth penalty must be ({m}, {m}); got {:?}",
                smooth_penalty.dim()
            ));
        }
        if p == 0 {
            return Err("SaeManifoldAtom::new: decoder output dimension must be positive".into());
        }
        Ok(Self {
            name: name.into(),
            basis_kind,
            latent_dim,
            basis_values,
            basis_jacobian,
            decoder_coefficients,
            smooth_penalty,
            basis_evaluator: None,
        })
    }

    pub fn with_basis_evaluator(mut self, evaluator: Arc<dyn SaeBasisEvaluator>) -> Self {
        self.basis_evaluator = Some(evaluator);
        self
    }

    pub fn refresh_basis(&mut self, coords: ArrayView2<'_, f64>) -> Result<(), String> {
        let evaluator = self.basis_evaluator.as_ref().ok_or_else(|| {
            format!(
                "SaeManifoldAtom {} has no basis evaluator; caller must rebuild the term after each coordinate step",
                self.name
            )
        })?;
        let (phi, jet) = evaluator.evaluate(coords)?;
        if phi.dim() != self.basis_values.dim() {
            return Err(format!(
                "SaeManifoldAtom::refresh_basis: evaluator returned Phi {:?}, expected {:?}",
                phi.dim(),
                self.basis_values.dim()
            ));
        }
        if jet.dim() != self.basis_jacobian.dim() {
            return Err(format!(
                "SaeManifoldAtom::refresh_basis: evaluator returned jet {:?}, expected {:?}",
                jet.dim(),
                self.basis_jacobian.dim()
            ));
        }
        self.basis_values = phi;
        self.basis_jacobian = jet;
        Ok(())
    }

    pub fn n_obs(&self) -> usize {
        self.basis_values.nrows()
    }

    pub fn basis_size(&self) -> usize {
        self.basis_values.ncols()
    }

    pub fn output_dim(&self) -> usize {
        self.decoder_coefficients.ncols()
    }

    /// `g_k(t_{ik}) = Phi_k(t_{ik}) B_k`.
    pub fn decoded_row(&self, row: usize) -> Array1<f64> {
        let p = self.output_dim();
        let mut out = Array1::<f64>::zeros(p);
        self.fill_decoded_row(row, out.as_slice_mut().expect("contiguous"));
        out
    }

    /// In-place fill of `g_k(t_{ik})` into a caller-supplied buffer of length `p`.
    /// Hot-loop variant used by the arrow-Schur assembly to avoid per-row
    /// allocations.
    pub fn fill_decoded_row(&self, row: usize, out: &mut [f64]) {
        let p = self.output_dim();
        let m = self.basis_size();
        assert_eq!(out.len(), p);
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        for basis_col in 0..m {
            let phi = self.basis_values[[row, basis_col]];
            if phi == 0.0 {
                continue;
            }
            for out_col in 0..p {
                out[out_col] += phi * self.decoder_coefficients[[basis_col, out_col]];
            }
        }
    }

    /// `d g_k(t_{ik}) / d t_{ik,j}` for one row and latent axis.
    pub fn decoded_derivative_row(&self, row: usize, latent_axis: usize) -> Array1<f64> {
        let p = self.output_dim();
        let mut out = Array1::<f64>::zeros(p);
        self.fill_decoded_derivative_row(row, latent_axis, out.as_slice_mut().expect("contiguous"));
        out
    }

    /// In-place fill of `d g_k / d t_{ik,axis}` into a caller-supplied buffer of
    /// length `p`. Hot-loop variant used by the arrow-Schur assembly.
    pub fn fill_decoded_derivative_row(&self, row: usize, latent_axis: usize, out: &mut [f64]) {
        let p = self.output_dim();
        let m = self.basis_size();
        assert_eq!(out.len(), p);
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        for basis_col in 0..m {
            let dphi = self.basis_jacobian[[row, basis_col, latent_axis]];
            if dphi == 0.0 {
                continue;
            }
            for out_col in 0..p {
                out[out_col] += dphi * self.decoder_coefficients[[basis_col, out_col]];
            }
        }
    }
}

/// Assignment prior/relaxation used by [`SaeAssignment`].
#[derive(Debug, Clone, Copy)]
pub enum AssignmentMode {
    /// Row-wise simplex assignment with entropy sparsity.
    Softmax { temperature: f64, sparsity: f64 },
    /// Deterministic concrete relaxation of a truncated IBP active set.
    IBPMap {
        temperature: f64,
        alpha: f64,
        learnable_alpha: bool,
    },
    /// Independent sigmoid activations with a hard JumpReLU active gate.
    JumpReLU { temperature: f64, threshold: f64 },
}

impl AssignmentMode {
    #[must_use]
    pub fn softmax(temperature: f64) -> Self {
        Self::Softmax {
            temperature,
            sparsity: 1.0,
        }
    }

    #[must_use]
    pub fn ibp_map(temperature: f64, alpha: f64, learnable_alpha: bool) -> Self {
        Self::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        }
    }

    #[must_use]
    pub fn jumprelu(temperature: f64, threshold: f64) -> Self {
        Self::JumpReLU {
            temperature,
            threshold,
        }
    }

    pub fn temperature(&self) -> f64 {
        match *self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::IBPMap { temperature, .. }
            | AssignmentMode::JumpReLU { temperature, .. } => temperature,
        }
    }

    fn set_temperature(&mut self, new_temperature: f64) -> Result<(), String> {
        if !(new_temperature.is_finite() && new_temperature > 0.0) {
            return Err(format!(
                "AssignmentMode: temperature must be finite and positive; got {new_temperature}"
            ));
        }
        match self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::IBPMap { temperature, .. }
            | AssignmentMode::JumpReLU { temperature, .. } => {
                *temperature = new_temperature;
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        let temperature = self.temperature();
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(format!(
                "AssignmentMode: temperature must be finite and positive; got {temperature}"
            ));
        }
        match *self {
            AssignmentMode::Softmax { sparsity, .. } => {
                if !(sparsity.is_finite() && sparsity > 0.0) {
                    return Err(format!(
                        "AssignmentMode::Softmax: sparsity must be finite and positive; got {sparsity}"
                    ));
                }
            }
            AssignmentMode::IBPMap { alpha, .. } => {
                if !(alpha.is_finite() && alpha > 0.0) {
                    return Err(format!(
                        "AssignmentMode::IBPMap: alpha must be finite and positive; got {alpha}"
                    ));
                }
            }
            AssignmentMode::JumpReLU { threshold, .. } => {
                if !threshold.is_finite() {
                    return Err(format!(
                        "AssignmentMode::JumpReLU: threshold must be finite; got {threshold}"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Per-row latent assignment state.
///
/// The free assignment parameter is `logits`; non-negative assignments are
/// derived by row-wise softmax, independent IBP-MAP sigmoid active indicators,
/// or JumpReLU gates. `coords[k]` holds `t_{.,k}` for atom `k`.
#[derive(Debug, Clone)]
pub struct SaeAssignment {
    pub logits: Array2<f64>,
    pub coords: Vec<LatentCoordValues>,
    pub mode: AssignmentMode,
}

impl SaeAssignment {
    #[must_use = "build error must be handled"]
    pub fn new(
        logits: Array2<f64>,
        coords: Vec<LatentCoordValues>,
        temperature: f64,
    ) -> Result<Self, String> {
        Self::with_mode(logits, coords, AssignmentMode::softmax(temperature))
    }

    #[must_use = "build error must be handled"]
    pub fn with_mode(
        logits: Array2<f64>,
        coords: Vec<LatentCoordValues>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        mode.validate()?;
        let n = logits.nrows();
        let k = logits.ncols();
        if coords.len() != k {
            return Err(format!(
                "SaeAssignment::new: coords length {} must equal K={k}",
                coords.len()
            ));
        }
        for (atom, coord) in coords.iter().enumerate() {
            if coord.n_obs() != n {
                return Err(format!(
                    "SaeAssignment::new: coord atom {atom} has n_obs={} but logits has {n}",
                    coord.n_obs()
                ));
            }
        }
        Ok(Self {
            logits,
            coords,
            mode,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.logits.nrows()
    }

    pub fn k_atoms(&self) -> usize {
        self.logits.ncols()
    }

    pub fn total_coord_dim(&self) -> usize {
        self.coords.iter().map(|c| c.latent_dim()).sum()
    }

    pub fn row_block_dim(&self) -> usize {
        self.k_atoms() + self.total_coord_dim()
    }

    pub fn coord_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = self.k_atoms();
        for coord in &self.coords {
            out.push(cursor);
            cursor += coord.latent_dim();
        }
        out
    }

    pub fn assignments(&self) -> Array2<f64> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.assignments_row(row);
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        out
    }

    pub fn try_assignments(&self) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.try_assignments_row(row)?;
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        Ok(out)
    }

    pub fn assignments_row(&self, row: usize) -> Array1<f64> {
        self.try_assignments_row(row)
            .expect("assignment logits must be finite")
    }

    pub fn try_assignments_row(&self, row: usize) -> Result<Array1<f64>, String> {
        validate_finite_logits(self.logits.row(row), row)?;
        if self.k_atoms() == 1 {
            return Ok(Array1::from_vec(vec![1.0]));
        }
        match self.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                Ok(softmax_row(self.logits.row(row), temperature))
            }
            AssignmentMode::IBPMap {
                temperature, alpha, ..
            } => Ok(ibp_map_row(self.logits.row(row), temperature, alpha)),
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => Ok(jumprelu_row(self.logits.row(row), temperature, threshold)),
        }
    }

    /// Flatten extension coordinates in row-major SAE layout:
    /// `(logits_i[0..K], t_i0[0..d_0], ..., t_iK[0..d_K])` for every row.
    pub fn flatten_ext_coords(&self) -> Array1<f64> {
        let n = self.n_obs();
        let q = self.row_block_dim();
        let k = self.k_atoms();
        let offsets = self.coord_offsets();
        let mut out = Array1::<f64>::zeros(n * q);
        for row in 0..n {
            let base = row * q;
            for atom in 0..k {
                out[base + atom] = self.logits[[row, atom]];
            }
            for atom in 0..k {
                let d = self.coords[atom].latent_dim();
                let t_row = self.coords[atom].row(row);
                for axis in 0..d {
                    out[base + offsets[atom] + axis] = t_row[axis];
                }
            }
        }
        out
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_no_gauge(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        temperature: f64,
    ) -> Result<Self, String> {
        let coords = coord_blocks
            .iter()
            .map(|c| LatentCoordValues::from_matrix(c.view(), LatentIdMode::None))
            .collect();
        Self::new(logits, coords, temperature)
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_mode(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        let coords = coord_blocks
            .iter()
            .map(|c| LatentCoordValues::from_matrix(c.view(), LatentIdMode::None))
            .collect();
        Self::with_mode(logits, coords, mode)
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_mode_and_manifolds(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        manifolds: Vec<LatentManifold>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        if coord_blocks.len() != manifolds.len() {
            return Err(format!(
                "SaeAssignment::from_blocks_with_mode_and_manifolds: coord block length {} != manifold length {}",
                coord_blocks.len(),
                manifolds.len()
            ));
        }
        let coords = coord_blocks
            .iter()
            .zip(manifolds)
            .map(|(c, manifold)| {
                LatentCoordValues::from_matrix_with_manifold(c.view(), LatentIdMode::None, manifold)
            })
            .collect();
        Self::with_mode(logits, coords, mode)
    }
}

/// REML-selected continuous hyperparameters for SAE-manifold.
#[derive(Debug, Clone)]
pub struct SaeManifoldRho {
    /// `log(lambda_sparse)` for softmax entropy or JumpReLU gated L1, or the
    /// learnable `log(alpha)` offset for IBP-MAP assignment.
    pub log_lambda_sparse: f64,
    /// `log(lambda_smooth)` shared by the per-atom decoder penalties.
    pub log_lambda_smooth: f64,
    /// Per-atom, per-axis `log(alpha_kj)` ARD strengths.
    pub log_ard: Vec<Array1<f64>>,
}

impl SaeManifoldRho {
    #[must_use]
    pub fn new(log_lambda_sparse: f64, log_lambda_smooth: f64, log_ard: Vec<Array1<f64>>) -> Self {
        Self {
            log_lambda_sparse,
            log_lambda_smooth,
            log_ard,
        }
    }

    pub fn lambda_sparse(&self) -> f64 {
        self.log_lambda_sparse.exp()
    }

    pub fn lambda_smooth(&self) -> f64 {
        self.log_lambda_smooth.exp()
    }
}

/// Loss breakdown for diagnostics and evidence ranking.
#[derive(Debug, Clone, Copy)]
pub struct SaeManifoldLoss {
    pub data_fit: f64,
    pub assignment_sparsity: f64,
    pub smoothness: f64,
    pub ard: f64,
}

impl SaeManifoldLoss {
    pub const fn total(&self) -> f64 {
        self.data_fit + self.assignment_sparsity + self.smoothness + self.ard
    }

    /// Laplace/REML wrappers rank larger evidence higher. This local score is
    /// the negative penalized objective, used when a full `RemlState` is not
    /// driving the term yet.
    pub const fn evidence_proxy(&self) -> f64 {
        -self.total()
    }
}

/// Full SAE-manifold term.
#[derive(Debug, Clone)]
pub struct SaeManifoldTerm {
    pub atoms: Vec<SaeManifoldAtom>,
    pub assignment: SaeAssignment,
    temperature_schedule: Option<GumbelTemperatureSchedule>,
}

impl SaeManifoldTerm {
    #[must_use = "build error must be handled"]
    pub fn new(atoms: Vec<SaeManifoldAtom>, assignment: SaeAssignment) -> Result<Self, String> {
        if atoms.is_empty() {
            return Err("SaeManifoldTerm::new: at least one atom required".into());
        }
        let n = atoms[0].n_obs();
        let p = atoms[0].output_dim();
        if assignment.n_obs() != n || assignment.k_atoms() != atoms.len() {
            return Err(format!(
                "SaeManifoldTerm::new: assignment shape ({}, {}) does not match atoms ({n}, {})",
                assignment.n_obs(),
                assignment.k_atoms(),
                atoms.len()
            ));
        }
        for (k, atom) in atoms.iter().enumerate() {
            if atom.n_obs() != n {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} has n_obs={} but atom 0 has {n}",
                    atom.n_obs()
                ));
            }
            if atom.output_dim() != p {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} output_dim={} but atom 0 has {p}",
                    atom.output_dim()
                ));
            }
            if atom.latent_dim != assignment.coords[k].latent_dim() {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} latent_dim={} but assignment coord has {}",
                    atom.latent_dim,
                    assignment.coords[k].latent_dim()
                ));
            }
        }
        Ok(Self {
            atoms,
            assignment,
            temperature_schedule: None,
        })
    }

    pub fn set_temperature_schedule(
        &mut self,
        sched: GumbelTemperatureSchedule,
    ) -> Result<(), String> {
        sched.validate()?;
        self.assignment
            .mode
            .set_temperature(sched.current_tau(sched.iter_count))?;
        self.temperature_schedule = Some(sched);
        Ok(())
    }

    fn advance_temperature_schedule(&mut self) -> Result<Option<f64>, String> {
        let Some(schedule) = self.temperature_schedule.as_mut() else {
            return Ok(None);
        };
        schedule.validate()?;
        let tau = schedule.step();
        self.assignment.mode.set_temperature(tau)?;
        Ok(Some(tau))
    }

    pub fn n_obs(&self) -> usize {
        self.assignment.n_obs()
    }

    pub fn k_atoms(&self) -> usize {
        self.atoms.len()
    }

    pub fn output_dim(&self) -> usize {
        self.atoms[0].output_dim()
    }

    pub fn beta_dim(&self) -> usize {
        let p = self.output_dim();
        self.atoms.iter().map(|a| a.basis_size() * p).sum()
    }

    pub fn beta_offsets(&self) -> Vec<usize> {
        let p = self.output_dim();
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            out.push(cursor);
            cursor += atom.basis_size() * p;
        }
        out
    }

    pub fn flatten_beta(&self) -> Array1<f64> {
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        let mut out = Array1::<f64>::zeros(self.beta_dim());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            for basis_col in 0..m {
                for out_col in 0..p {
                    out[off + basis_col * p + out_col] =
                        atom.decoder_coefficients[[basis_col, out_col]];
                }
            }
        }
        out
    }

    pub fn set_flat_beta(&mut self, beta: ArrayView1<'_, f64>) -> Result<(), String> {
        if beta.len() != self.beta_dim() {
            return Err(format!(
                "set_flat_beta: beta length {} != expected {}",
                beta.len(),
                self.beta_dim()
            ));
        }
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        for (atom_idx, atom) in self.atoms.iter_mut().enumerate() {
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            for basis_col in 0..m {
                for out_col in 0..p {
                    atom.decoder_coefficients[[basis_col, out_col]] =
                        beta[off + basis_col * p + out_col];
                }
            }
        }
        Ok(())
    }

    pub fn fitted(&self) -> Array2<f64> {
        self.try_fitted().expect("assignment logits must be finite")
    }

    pub fn try_fitted(&self) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, p));
        // Reuse a single scratch buffer across all (row, atom) pairs instead of
        // allocating a fresh `Array1<f64>` of length p per call.
        let mut g_buf = vec![0.0_f64; p];
        for row in 0..n {
            let a = self.assignment.try_assignments_row(row)?;
            for atom_idx in 0..k_atoms {
                self.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
                let a_k = a[atom_idx];
                let mut out_row = out.row_mut(row);
                for out_col in 0..p {
                    out_row[out_col] += a_k * g_buf[out_col];
                }
            }
        }
        Ok(out)
    }

    pub fn loss(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<SaeManifoldLoss, String> {
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::loss: Z must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        let fitted = self.try_fitted()?;
        let mut data_fit = 0.0_f64;
        for row in 0..target.nrows() {
            for out_col in 0..target.ncols() {
                let r = target[[row, out_col]] - fitted[[row, out_col]];
                data_fit += 0.5 * r * r;
            }
        }
        let assignment_sparsity = assignment_prior_value(&self.assignment, rho);
        let smoothness = self.decoder_smoothness_value(rho.lambda_smooth());
        let ard = self.ard_value(rho)?;
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
        })
    }

    fn decoder_smoothness_value(&self, lambda_smooth: f64) -> f64 {
        let p = self.output_dim();
        let mut acc = 0.0;
        for atom in &self.atoms {
            let m = atom.basis_size();
            for out_col in 0..p {
                for i in 0..m {
                    for j in 0..m {
                        acc += 0.5
                            * lambda_smooth
                            * atom.decoder_coefficients[[i, out_col]]
                            * atom.smooth_penalty[[i, j]]
                            * atom.decoder_coefficients[[j, out_col]];
                    }
                }
            }
        }
        acc
    }

    fn ard_value(&self, rho: &SaeManifoldRho) -> Result<f64, String> {
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atoms but term has {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let n = self.n_obs();
        let mut acc = 0.0;
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            if rho.log_ard[atom_idx].len() != d {
                return Err(format!(
                    "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                    rho.log_ard[atom_idx].len()
                ));
            }
            for axis in 0..d {
                let log_alpha = rho.log_ard[atom_idx][axis];
                let alpha = log_alpha.exp();
                let mut sq = 0.0;
                for row in 0..n {
                    let v = coord.row(row)[axis];
                    sq += v * v;
                }
                // Negative log Gaussian prior for precision alpha:
                // 0.5 * alpha * ||t||^2 - 0.5 * n * log(alpha).
                acc += 0.5 * alpha * sq - 0.5 * (n as f64) * log_alpha;
            }
        }
        Ok(acc)
    }

    /// Assemble the enlarged `(logits, t)` row-local Arrow-Schur system.
    pub fn assemble_arrow_schur(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<ArrowSchurSystem, String> {
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: Z must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: log_ard length {} != K {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let q = self.assignment.row_block_dim();
        let beta_dim = self.beta_dim();
        let beta_offsets = self.beta_offsets();
        let coord_offsets = self.assignment.coord_offsets();
        let lambda_smooth = rho.lambda_smooth();
        let (assignment_grad, assignment_hdiag) =
            assignment_prior_grad_hdiag(&self.assignment, rho)?;
        let mut sys = ArrowSchurSystem::new(n, q, beta_dim);

        // Decoder smoothness penalty in the beta block.
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let off = beta_offsets[atom_idx];
            for out_col in 0..p {
                for i in 0..m {
                    let beta_i = off + i * p + out_col;
                    let mut grad = 0.0;
                    for j in 0..m {
                        let beta_j = off + j * p + out_col;
                        let s_ij =
                            0.5 * (atom.smooth_penalty[[i, j]] + atom.smooth_penalty[[j, i]]);
                        sys.hbb[[beta_i, beta_j]] += lambda_smooth * s_ij;
                        grad += lambda_smooth * s_ij * atom.decoder_coefficients[[j, out_col]];
                    }
                    sys.gb[beta_i] += grad;
                }
            }
        }

        // Hoist per-row temporaries outside the row loop: these allocations
        // previously fired N times per assembly, and each `decoded_row` /
        // `decoded_derivative_row` call inside the loop allocated its own
        // `Array1<f64>` of length p.
        let mut decoded = Array2::<f64>::zeros((k_atoms, p));
        let mut dg_buf = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        let mut local_jac = Array2::<f64>::zeros((q, p));
        // Stick-breaking prior for IBP-MAP depends only on (k_atoms, alpha)
        // which are constant across rows; precompute once.
        let ibp_prior_vec = match self.assignment.mode {
            AssignmentMode::IBPMap { alpha, .. } => {
                Some(ibp_stick_breaking_prior(k_atoms, alpha).to_vec())
            }
            _ => None,
        };
        let ibp_prior_slice = ibp_prior_vec.as_deref();
        // Scratch buffer for per-(row, atom) decoded outputs. The full `decoded`
        // matrix retains all atoms for this row so the assignment-Jacobian
        // helper can read it.
        let mut decoded_scratch = vec![0.0_f64; p];
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row(row)?;
            fitted.fill(0.0);
            for atom_idx in 0..k_atoms {
                let a_k = assignments[atom_idx];
                self.atoms[atom_idx].fill_decoded_row(row, &mut decoded_scratch);
                for out_col in 0..p {
                    decoded[[atom_idx, out_col]] = decoded_scratch[out_col];
                    fitted[out_col] += a_k * decoded_scratch[out_col];
                }
            }
            for out_col in 0..p {
                error[out_col] = fitted[out_col] - target[[row, out_col]];
            }

            local_jac.fill(0.0);
            fill_assignment_logit_jvp_rows(
                self.assignment.mode,
                self.assignment.logits.row(row),
                assignments.view(),
                decoded.view(),
                fitted.view(),
                ibp_prior_slice,
                &mut local_jac,
            );
            // Coordinate columns.
            for atom_idx in 0..k_atoms {
                let d = self.atoms[atom_idx].latent_dim;
                let off = coord_offsets[atom_idx];
                let a_k = assignments[atom_idx];
                for axis in 0..d {
                    self.atoms[atom_idx].fill_decoded_derivative_row(row, axis, &mut dg_buf);
                    for out_col in 0..p {
                        local_jac[[off + axis, out_col]] = a_k * dg_buf[out_col];
                    }
                }
            }

            let mut block = ArrowRowBlock::new(q, beta_dim);
            for a in 0..q {
                let mut g = 0.0;
                for out_col in 0..p {
                    g += local_jac[[a, out_col]] * error[out_col];
                }
                block.gt[a] += g;
                for b in 0..q {
                    let mut h = 0.0;
                    for out_col in 0..p {
                        h += local_jac[[a, out_col]] * local_jac[[b, out_col]];
                    }
                    block.htt[[a, b]] += h;
                }
            }

            // Assignment prior in logit space.
            let assignment_base = row * k_atoms;
            for atom_idx in 0..k_atoms {
                block.gt[atom_idx] += assignment_grad[assignment_base + atom_idx];
                block.htt[[atom_idx, atom_idx]] += assignment_hdiag[assignment_base + atom_idx];
            }

            // ARD on each on-atom coordinate.
            for atom_idx in 0..k_atoms {
                let coord = &self.assignment.coords[atom_idx];
                let d = coord.latent_dim();
                if rho.log_ard[atom_idx].len() != d {
                    return Err(format!(
                        "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                        rho.log_ard[atom_idx].len()
                    ));
                }
                let off = coord_offsets[atom_idx];
                let row_t = coord.row(row);
                for axis in 0..d {
                    let alpha = rho.log_ard[atom_idx][axis].exp();
                    block.gt[off + axis] += alpha * row_t[axis];
                    block.htt[[off + axis, off + axis]] += alpha;
                }
            }

            // Beta gradient/Hessian and local-beta cross block.
            //
            // The per-row beta Jacobian is
            //   J_β[out_col, beta_idx] = a_k · phi_k[basis_col]   if out_col == out_col(beta_idx)
            //                            0                         otherwise
            // so the data-fit Gauss-Newton beta-Hessian factors as a rank-`p`
            // sum of outer products. We pre-compute the per-(atom, basis_col)
            // scalar `a_k · phi_k` once and reuse it across the `out_col`,
            // `local_col`, and inner `(atom_j, basis_col2)` loops. This keeps
            // the asymptotic O((K·M)² · P) work but removes a P-factor of
            // redundant scalar arithmetic and ndarray indexing in the hot
            // inner Hessian assembly.
            let mut a_phi: Vec<(usize, f64)> = Vec::with_capacity(k_atoms * 4);
            for atom_idx in 0..k_atoms {
                let atom = &self.atoms[atom_idx];
                let atom_beta_off = beta_offsets[atom_idx];
                let m = atom.basis_size();
                let a_k = assignments[atom_idx];
                for basis_col in 0..m {
                    let phi = atom.basis_values[[row, basis_col]];
                    a_phi.push((atom_beta_off + basis_col * p, a_k * phi));
                }
            }
            for &(beta_base_i, j_beta_i) in a_phi.iter() {
                if j_beta_i == 0.0 {
                    // Skip rank-1 outer product whose left factor is zero;
                    // saves a full (q + Σ M) · p pass for masked / inactive
                    // atoms (e.g. assignment exactly zeroed by JumpReLU).
                    continue;
                }
                for out_col in 0..p {
                    let beta_idx = beta_base_i + out_col;
                    sys.gb[beta_idx] += j_beta_i * error[out_col];
                    for local_col in 0..q {
                        block.htbeta[[local_col, beta_idx]] +=
                            local_jac[[local_col, out_col]] * j_beta_i;
                    }
                    for &(beta_base_j, j_beta_j) in a_phi.iter() {
                        let beta_j = beta_base_j + out_col;
                        sys.hbb[[beta_idx, beta_j]] += j_beta_i * j_beta_j;
                    }
                }
            }
            sys.rows[row] = block;
        }
        if let Some(registry) = analytic_penalties {
            self.add_sae_analytic_penalty_contributions(&mut sys, registry)
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        }
        self.apply_sae_riemannian_geometry(&mut sys);
        Ok(sys)
    }

    fn ext_coord_matrix(&self) -> Array2<f64> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let flat = self.assignment.flatten_ext_coords();
        let mut out = Array2::<f64>::zeros((n, q));
        for row in 0..n {
            for col in 0..q {
                out[[row, col]] = flat[row * q + col];
            }
        }
        out
    }

    fn ext_coord_manifold(&self) -> LatentManifold {
        let mut parts = Vec::with_capacity(self.assignment.row_block_dim());
        for _ in 0..self.k_atoms() {
            parts.push(LatentManifold::Euclidean);
        }
        let mut any_constrained = false;
        for coord in &self.assignment.coords {
            if coord.manifold().is_euclidean() {
                for _ in 0..coord.latent_dim() {
                    parts.push(LatentManifold::Euclidean);
                }
            } else {
                any_constrained = true;
                parts.push(coord.manifold().clone());
            }
        }
        if any_constrained {
            LatentManifold::Product(parts)
        } else {
            LatentManifold::Euclidean
        }
    }

    fn apply_sae_riemannian_geometry(&self, sys: &mut ArrowSchurSystem) {
        let manifold = self.ext_coord_manifold();
        if manifold.is_euclidean() {
            return;
        }
        let ext = self.ext_coord_matrix();
        let latent =
            LatentCoordValues::from_matrix_with_manifold(ext.view(), LatentIdMode::None, manifold);
        sys.apply_riemannian_latent_geometry(&latent);
    }

    pub fn update_ard_reml(&self, rho: &mut SaeManifoldRho) -> Result<(), String> {
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "SaeManifoldTerm::update_ard_reml: log_ard length {} != K {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let n = self.n_obs() as f64;
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            if rho.log_ard[atom_idx].len() != d {
                return Err(format!(
                    "SaeManifoldTerm::update_ard_reml: atom {atom_idx} log_ard length {} != dim {d}",
                    rho.log_ard[atom_idx].len()
                ));
            }
            for axis in 0..d {
                let mut sq = 0.0;
                for row in 0..coord.n_obs() {
                    let v = coord.row(row)[axis];
                    sq += v * v;
                }
                let alpha = n / sq.max(1.0e-12);
                rho.log_ard[atom_idx][axis] = alpha.ln().clamp(-8.0, 16.0);
            }
        }
        Ok(())
    }

    fn add_sae_analytic_penalty_contributions(
        &self,
        sys: &mut ArrowSchurSystem,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<(), ArrowSchurError> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let logits_flat = flat_logits(self.assignment.logits.view());
        let beta = self.flatten_beta();
        for (penalty, (rho_slice, tier, name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi => {
                    if matches!(
                        penalty,
                        AnalyticPenaltyKind::IBPAssignment(_)
                            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
                    ) {
                        self.add_sae_logit_penalty(sys, penalty, logits_flat.view(), rho_local);
                    } else if self.k_atoms() == 1 && sae_penalty_is_row_block_supported(penalty) {
                        let off = self.assignment.coord_offsets()[0];
                        let coord = &self.assignment.coords[0];
                        self.add_sae_coord_penalty(sys, off, coord, penalty, rho_local);
                    } else {
                        return Err(ArrowSchurError::SchurFactorFailed {
                            reason: format!(
                                "analytic penalty {name:?} cannot be injected into the SAE-manifold row layout; multi-atom coordinate or cross-row penalties require an explicit atom target"
                            ),
                        });
                    }
                }
                PenaltyTier::Beta => {
                    self.add_sae_beta_penalty(sys, penalty, beta.view(), rho_local);
                }
                PenaltyTier::Rho => {}
            }
        }
        Ok(())
    }

    fn add_sae_logit_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        penalty: &AnalyticPenaltyKind,
        target: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let n = self.n_obs();
        let k = self.k_atoms();
        let grad = penalty.grad_target(target, rho_local);
        for row in 0..n {
            for atom in 0..k {
                sys.rows[row].gt[atom] += grad[row * k + atom];
            }
        }
        if let Some(diag) = penalty.hessian_diag(target, rho_local) {
            for row in 0..n {
                for atom in 0..k {
                    sys.rows[row].htt[[atom, atom]] += diag[row * k + atom];
                }
            }
        }
    }

    fn add_sae_coord_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        off: usize,
        coord: &LatentCoordValues,
        penalty: &AnalyticPenaltyKind,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let n = coord.n_obs();
        let d = coord.latent_dim();
        let target = coord.as_flat().view();
        let grad = penalty.grad_target(target, rho_local);
        for row in 0..n {
            for axis in 0..d {
                sys.rows[row].gt[off + axis] += grad[row * d + axis];
            }
        }
        if let Some(diag) = penalty.hessian_diag(target, rho_local) {
            for row in 0..n {
                for axis in 0..d {
                    sys.rows[row].htt[[off + axis, off + axis]] += diag[row * d + axis];
                }
            }
            return;
        }
        let mut probe = Array1::<f64>::zeros(n * d);
        for axis in 0..d {
            probe.fill(0.0);
            for row in 0..n {
                probe[row * d + axis] = 1.0;
            }
            let hv = penalty.hvp(target, rho_local, probe.view());
            for row in 0..n {
                for b in 0..d {
                    sys.rows[row].htt[[off + b, off + axis]] += hv[row * d + b];
                }
            }
        }
    }

    fn add_sae_beta_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let k = self.beta_dim();
        let grad = penalty.grad_target(target_beta, rho_local);
        for j in 0..k {
            sys.gb[j] += grad[j];
        }
        if let Some(diag) = penalty.hessian_diag(target_beta, rho_local) {
            for j in 0..k {
                sys.hbb[[j, j]] += diag[j];
            }
            return;
        }
        let mut probe = Array1::<f64>::zeros(k);
        for j in 0..k {
            probe.fill(0.0);
            probe[j] = 1.0;
            let hv = penalty.hvp(target_beta, rho_local, probe.view());
            for i in 0..k {
                sys.hbb[[i, j]] += hv[i];
            }
        }
    }

    pub fn solve_newton_step(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let sys = self
            .assemble_arrow_schur(target, rho, analytic_penalties)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        // Self-heal against non-PD per-row blocks produced by PCA-seeded
        // latent coordinates on subset / out-of-sample data (#163, #175):
        // route every Newton-step solve through the Ceres-style LM ridge
        // escalation, reusing the caller-supplied Tikhonov ridges
        // (`ridge_ext_coord`, `ridge_beta`) as the base damping. No new
        // tuning knobs — just the existing proximal-correction schedule.
        sys.solve_with_lm_escalation(ridge_ext_coord, ridge_beta)
    }

    pub fn apply_newton_step(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
    ) -> Result<(), String> {
        self.apply_newton_step_impl(delta_ext_coord, delta_beta, step_size, true)
    }

    pub fn apply_newton_step_external_basis_refresh(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
    ) -> Result<(), String> {
        self.apply_newton_step_impl(delta_ext_coord, delta_beta, step_size, false)
    }

    fn apply_newton_step_impl(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
        refresh_basis: bool,
    ) -> Result<(), String> {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: step_size must be finite and positive; got {step_size}"
            ));
        }
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        if delta_ext_coord.len() != n * q {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: delta_ext_coord length {} != expected {}",
                delta_ext_coord.len(),
                n * q
            ));
        }
        if delta_beta.len() != self.beta_dim() {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: delta_beta length {} != expected {}",
                delta_beta.len(),
                self.beta_dim()
            ));
        }

        let k_atoms = self.k_atoms();
        let coord_offsets = self.assignment.coord_offsets();
        for row in 0..n {
            let row_base = row * q;
            for atom_idx in 0..k_atoms {
                self.assignment.logits[[row, atom_idx]] +=
                    step_size * delta_ext_coord[row_base + atom_idx];
            }
        }

        for atom_idx in 0..k_atoms {
            let d = self.assignment.coords[atom_idx].latent_dim();
            let mut delta_coord = Array1::<f64>::zeros(n * d);
            for row in 0..n {
                let row_base = row * q + coord_offsets[atom_idx];
                for axis in 0..d {
                    delta_coord[row * d + axis] = step_size * delta_ext_coord[row_base + axis];
                }
            }
            self.assignment.coords[atom_idx].retract_flat_delta(delta_coord.view());
            if refresh_basis {
                let coords = self.assignment.coords[atom_idx].as_matrix();
                self.atoms[atom_idx].refresh_basis(coords.view())?;
            }
        }

        let mut beta = self.flatten_beta();
        for idx in 0..beta.len() {
            beta[idx] += step_size * delta_beta[idx];
        }
        self.set_flat_beta(beta.view())
    }

    pub fn run_joint_fit_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeManifoldLoss, String> {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_joint_fit_arrow_schur: step_size must be finite and positive; got {step_size}"
            ));
        }
        for _ in 0..max_iter {
            self.advance_temperature_schedule()?;
            self.update_ard_reml(rho)?;
            let pre_step_loss = self.loss(target, rho)?;
            let pre_step_total = pre_step_loss.total();
            let sys = self
                .assemble_arrow_schur(target, rho, analytic_penalties)
                .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            // Inner Newton step with principled LM-style ridge escalation. The
            // PCA-seed starting state on a small batch (e.g. `predict` on a
            // strict subset of the training set) can produce a per-row
            // `H_tt + ridge_t·I` whose Cholesky has a negative pivot, or a
            // near-singular Schur complement, at the caller's nominal ridges.
            // Rather than abort, mirror the proximal-correction outer wrapper
            // and grow both ridges geometrically until the linear system
            // factors. This is the same LM-trust-region damping the convergent
            // proximal_correction path applies; we route it through the same
            // factor-failure error variants so legitimate, non-recoverable
            // errors (PCG divergence with no factor failure, adaptive-step
            // exhaustion, …) still surface immediately.
            let (delta_ext_coord, delta_beta) = sys
                .solve_with_lm_escalation(ridge_ext_coord, ridge_beta)
                .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            let directional_decrease = sae_manifold_newton_directional_decrease(
                &sys,
                delta_ext_coord.view(),
                delta_beta.view(),
            );
            let snapshot = self.clone();
            if !(pre_step_total.is_finite()
                && directional_decrease.is_finite()
                && directional_decrease > 0.0)
            {
                *self = snapshot;
                break;
            }

            let mut trial_step_size = step_size;
            let mut accepted = false;
            for _ in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                *self = snapshot.clone();
                let trial_result = self
                    .apply_newton_step(delta_ext_coord.view(), delta_beta.view(), trial_step_size)
                    .and_then(|()| self.loss(target, rho));
                if let Ok(post_step_loss) = trial_result {
                    let post_step_total = post_step_loss.total();
                    let armijo_bound = pre_step_total
                        - SAE_MANIFOLD_ARMIJO_C1 * trial_step_size * directional_decrease;
                    if post_step_total.is_finite() && post_step_total <= armijo_bound {
                        accepted = true;
                        break;
                    }
                }
                trial_step_size *= 0.5;
            }
            if !accepted {
                *self = snapshot;
                break;
            }
        }
        self.update_ard_reml(rho)?;
        self.loss(target, rho)
    }

    pub fn run_single_external_basis_refresh_step_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeManifoldLoss, String> {
        self.advance_temperature_schedule()?;
        self.update_ard_reml(rho)?;
        let pre_step_loss = self.loss(target, rho)?;
        let (delta_ext_coord, delta_beta) = self
            .solve_newton_step(target, rho, analytic_penalties, ridge_ext_coord, ridge_beta)
            .map_err(|err| {
                format!(
                    "SaeManifoldTerm::run_single_external_basis_refresh_step_arrow_schur: {err}"
                )
            })?;
        self.apply_newton_step_external_basis_refresh(
            delta_ext_coord.view(),
            delta_beta.view(),
            step_size,
        )?;
        self.update_ard_reml(rho)?;
        Ok(pre_step_loss)
    }

    /// Build the analytic-penalty descriptors that correspond to the current
    /// SAE term. This is the bridge into `analytic_penalties.rs` for callers
    /// that want to register the same ρ axes with a REML driver.
    pub fn analytic_penalty_descriptors(&self) -> (AnalyticPenaltyKind, Vec<ARDPenalty>) {
        let assignment = match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                AnalyticPenaltyKind::SoftmaxAssignmentSparsity(Arc::new(
                    SoftmaxAssignmentSparsityPenalty::new(self.k_atoms(), temperature),
                ))
            }
            AssignmentMode::IBPMap {
                temperature,
                alpha,
                learnable_alpha,
            } => {
                let penalty =
                    IBPAssignmentPenalty::new(self.k_atoms(), alpha, temperature, learnable_alpha);
                let penalty = match self.temperature_schedule.clone() {
                    Some(schedule) => penalty.with_temperature_schedule(schedule),
                    None => penalty,
                };
                AnalyticPenaltyKind::IBPAssignment(Arc::new(penalty))
            }
            AssignmentMode::JumpReLU { .. } => {
                // SAFETY: `analytic_penalty_descriptors` is only called for
                // assignment modes that have a corresponding REML descriptor
                // (Softmax, IBPMap). JumpReLU is handled by the built-in
                // gated-L1 assignment prior and never reaches this bridge —
                // callers must dispatch on `self.assignment.mode` first. The
                // panic guards against a future caller forgetting to do so.
                panic!(
                    "JumpReLU assignment mode uses the built-in gated L1 assignment prior and has no AnalyticPenaltyKind descriptor"
                )
            }
        };
        let mut ard = Vec::with_capacity(self.k_atoms());
        for coord in &self.assignment.coords {
            ard.push(ARDPenalty::new(
                PsiSlice::full(coord.len(), Some(coord.latent_dim())),
                coord.latent_dim(),
            ));
        }
        (assignment, ard)
    }
}

fn sae_manifold_newton_directional_decrease(
    sys: &ArrowSchurSystem,
    delta_ext_coord: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
) -> f64 {
    assert_eq!(delta_ext_coord.len(), sys.rows.len() * sys.d);
    assert_eq!(delta_beta.len(), sys.k);
    let mut gradient_dot_step = 0.0;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let row_base = row_idx * sys.d;
        for axis in 0..sys.d {
            gradient_dot_step += row.gt[axis] * delta_ext_coord[row_base + axis];
        }
    }
    for idx in 0..sys.k {
        gradient_dot_step += sys.gb[idx] * delta_beta[idx];
    }
    -gradient_dot_step
}

fn softmax_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let k = logits.len();
    let inv_tau = 1.0 / temperature;
    let mut max_logit = f64::NEG_INFINITY;
    for &v in logits.iter() {
        max_logit = max_logit.max(v);
    }
    let mut out = Array1::<f64>::zeros(k);
    let mut sum = 0.0;
    for i in 0..k {
        let v = ((logits[i] - max_logit) * inv_tau).exp();
        out[i] = v;
        sum += v;
    }
    assert!(sum.is_finite() && sum > 0.0);
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

fn validate_finite_logits(logits: ArrayView1<'_, f64>, row: usize) -> Result<(), String> {
    for (col, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "SaeAssignment: non-finite assignment logit at row {row}, atom {col}: {v}"
            ));
        }
    }
    Ok(())
}

fn sigmoid_scalar(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Truncated-IBP stick-breaking prior weights `π_k = (α/(α+1))^k` for
/// k = 0, .., K-1. Under a Beta(α, 1) stick-breaking construction these are
/// the prior means of the active-set probabilities, so IBP-MAP assignment
/// mass should decay geometrically in `k` even when logits are tied.
fn ibp_stick_breaking_prior(k_atoms: usize, alpha: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(k_atoms);
    let ratio = alpha / (alpha + 1.0);
    let mut acc = 1.0;
    for k in 0..k_atoms {
        out[k] = acc;
        acc *= ratio;
    }
    out
}

/// IBP-MAP row activations: per-atom sigmoid likelihood times the truncated
/// stick-breaking prior mass. With tied logits the prior dominates and yields
/// strictly decreasing activations in atom index.
fn ibp_map_row(logits: ArrayView1<'_, f64>, temperature: f64, alpha: f64) -> Array1<f64> {
    let prior = ibp_stick_breaking_prior(logits.len(), alpha);
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = sigmoid_scalar(logits[i] / temperature) * prior[i];
    }
    out
}

fn jumprelu_row(logits: ArrayView1<'_, f64>, temperature: f64, threshold: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        if logits[i] > threshold {
            out[i] = sigmoid_scalar(logits[i] / temperature);
        }
    }
    out
}

fn fill_assignment_logit_jvp_rows(
    mode: AssignmentMode,
    logits: ArrayView1<'_, f64>,
    assignments: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    fitted: ArrayView1<'_, f64>,
    ibp_prior: Option<&[f64]>,
    local_jac: &mut Array2<f64>,
) {
    if assignments.len() == 1 {
        for logit_col in 0..assignments.len() {
            for out_col in 0..fitted.len() {
                local_jac[[logit_col, out_col]] = 0.0;
            }
        }
        return;
    }

    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            // da_k/dl_j = a_k (1[k=j] - a_j) / tau, contracted against
            // the assignment-weighted fitted row.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = assignments[logit_col]
                        * (decoded[[logit_col, out_col]] - fitted[out_col])
                        * inv_tau;
                }
            }
        }
        AssignmentMode::IBPMap { temperature, .. } => {
            // Truncated-IBP concrete relaxation: z_k = σ(l_k/τ) · π_k where
            // π_k is the stick-breaking prior. Thus
            // dz_k/dl_k = σ(l/τ)(1-σ(l/τ))/τ · π_k = a_k(π_k - a_k)/(π_k τ).
            let inv_tau = 1.0 / temperature;
            let prior = ibp_prior
                .expect("fill_assignment_logit_jvp_rows: IBPMap requires precomputed prior");
            for logit_col in 0..assignments.len() {
                let pi_k = prior[logit_col];
                let a_k = assignments[logit_col];
                let sig = if pi_k > 0.0 { a_k / pi_k } else { 0.0 };
                let dz = sig * (1.0 - sig) * inv_tau * pi_k;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = dz * decoded[[logit_col, out_col]];
                }
            }
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // Standard STE for the hard gate: the sigmoid derivative
            // contributes only on logits above the JumpReLU threshold.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                if logits[logit_col] <= threshold {
                    continue;
                }
                let activation = sigmoid_scalar(logits[logit_col] * inv_tau);
                let da = activation * (1.0 - activation) * inv_tau;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = da * decoded[[logit_col, out_col]];
                }
            }
        }
    }
}

fn flat_logits(logits: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for row in 0..logits.nrows() {
        let start = row * logits.ncols();
        for col in 0..logits.ncols() {
            out[start + col] = logits[[row, col]];
        }
    }
    out
}

fn assignment_prior_value(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
    if assignment.k_atoms() == 1 {
        return 0.0;
    }

    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)
            .expect("assignment logits must be finite");
    }
    let target = flat_logits(assignment.logits.view());
    match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                Array1::zeros(0)
            };
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            let sparsity_strength = rho.log_lambda_sparse.exp();
            let mut acc = 0.0;
            for &logit in target.iter() {
                if logit > threshold {
                    acc += sigmoid_scalar(logit / temperature);
                }
            }
            sparsity_strength * acc
        }
    }
}

fn assignment_prior_grad_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    if assignment.k_atoms() == 1 {
        let n_obs = assignment.n_obs();
        return Ok((Array1::zeros(n_obs), Array1::zeros(n_obs)));
    }

    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            let grad = penalty.grad_target(target.view(), rho_view.view());
            let diag = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "softmax assignment hessian diag unavailable".to_string())?;
            Ok((grad, diag))
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                Array1::zeros(0)
            };
            let grad = penalty.grad_target(target.view(), rho_view.view());
            let diag = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "IBP assignment hessian diag unavailable".to_string())?;
            Ok((grad, diag))
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            let sparsity_strength = rho.log_lambda_sparse.exp();
            let inv_tau = 1.0 / temperature;
            let inv_tau2 = inv_tau * inv_tau;
            let mut grad = Array1::<f64>::zeros(target.len());
            let mut diag = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                let logit = target[idx];
                if logit <= threshold {
                    continue;
                }
                let activation = sigmoid_scalar(logit * inv_tau);
                let slope = activation * (1.0 - activation);
                grad[idx] = sparsity_strength * slope * inv_tau;
                diag[idx] = sparsity_strength * slope * slope * inv_tau2;
            }
            Ok((grad, diag))
        }
    }
}

fn sae_penalty_is_row_block_supported(penalty: &AnalyticPenaltyKind) -> bool {
    matches!(
        penalty,
        AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::JumpReLU(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            | AnalyticPenaltyKind::IBPAssignment(_)
            | AnalyticPenaltyKind::RowPrecisionPrior(_)
            | AnalyticPenaltyKind::ParametricRowPrecisionPrior(_)
            | AnalyticPenaltyKind::ScadMcp(_)
    )
}

/// Helper for padded FFI callers. Arrays use `(K, N, M_max)` and
/// `(K, N, M_max, D_max)` storage, with `basis_sizes` and `latent_dims`
/// selecting each atom's active prefix.
///
/// `evaluators`, when non-empty, must have length `K`. Each entry attaches an
/// optional [`SaeBasisEvaluator`] to the matching atom so the Rust Newton
/// loop can refresh `Phi`/`dPhi/dt` between iterations without rebuilding the
/// term from Python. An empty slice leaves every atom in snapshot-only mode.
#[must_use = "build error must be handled"]
pub fn term_from_padded_blocks_with_mode(
    n_obs: usize,
    p_out: usize,
    basis_kinds: &[SaeAtomBasisKind],
    basis_values: ArrayView3<'_, f64>,
    basis_jacobian: ArrayView4<'_, f64>,
    basis_sizes: &[usize],
    latent_dims: &[usize],
    decoder_coefficients: ArrayView3<'_, f64>,
    smooth_penalties: ArrayView3<'_, f64>,
    logits: ArrayView2<'_, f64>,
    coords: &[Array2<f64>],
    mode: AssignmentMode,
    evaluators: &[Option<Arc<dyn SaeBasisEvaluator>>],
) -> Result<SaeManifoldTerm, String> {
    let k_atoms = basis_sizes.len();
    if latent_dims.len() != k_atoms || basis_kinds.len() != k_atoms || coords.len() != k_atoms {
        return Err("term_from_padded_blocks: K-length metadata mismatch".into());
    }
    if !evaluators.is_empty() && evaluators.len() != k_atoms {
        return Err(format!(
            "term_from_padded_blocks: evaluators length {} must equal K={k_atoms} or be empty",
            evaluators.len()
        ));
    }
    if logits.dim() != (n_obs, k_atoms) {
        return Err(format!(
            "term_from_padded_blocks: logits must be ({n_obs}, {k_atoms}); got {:?}",
            logits.dim()
        ));
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let m = basis_sizes[k];
        let d = latent_dims[k];
        let phi = basis_values.slice(s![k, 0..n_obs, 0..m]).to_owned();
        let jet = basis_jacobian.slice(s![k, 0..n_obs, 0..m, 0..d]).to_owned();
        let b = decoder_coefficients.slice(s![k, 0..m, 0..p_out]).to_owned();
        let s = smooth_penalties.slice(s![k, 0..m, 0..m]).to_owned();
        let atom = SaeManifoldAtom::new(
            format!("atom_{k}"),
            basis_kinds[k].clone(),
            d,
            phi,
            jet,
            b,
            s,
        )?;
        let atom = match evaluators.get(k).and_then(|slot| slot.clone()) {
            Some(evaluator) => atom.with_basis_evaluator(evaluator),
            None => atom,
        };
        atoms.push(atom);
    }
    let manifolds = basis_kinds
        .iter()
        .zip(latent_dims.iter().copied())
        .map(|(kind, d)| kind.latent_manifold(d))
        .collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits.to_owned(),
        coords.to_vec(),
        manifolds,
        mode,
    )?;
    SaeManifoldTerm::new(atoms, assignment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn search_strategy_exposes_fixed_and_sweep_values() {
        assert!(SearchStrategy::Fixed.is_fixed());

        let strategy = SearchStrategy::ExponentialSweep {
            values: vec![0.1, 1.0, 10.0],
        };
        assert!(!strategy.is_fixed());
        assert_eq!(strategy.sweep_values(), Some([0.1, 1.0, 10.0].as_slice()));
    }

    fn periodic_basis(coords: &Array2<f64>) -> (Array2<f64>, Array3<f64>) {
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, 3));
        let mut jet = Array3::<f64>::zeros((n, 3, 1));
        for row in 0..n {
            let x = coords[[row, 0]].rem_euclid(1.0);
            let angle = 2.0 * std::f64::consts::PI * x;
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = angle.sin();
            phi[[row, 2]] = angle.cos();
            jet[[row, 1, 0]] = 2.0 * std::f64::consts::PI * angle.cos();
            jet[[row, 2, 0]] = -2.0 * std::f64::consts::PI * angle.sin();
        }
        (phi, jet)
    }

    #[test]
    fn ibp_path_refreshes_periodic_basis_for_two_newton_iterations() {
        let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.2], [-0.3], [0.4]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((4, 1)),
            vec![coords0],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.10], [0.05], [-0.15], [0.20]];
        let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
        let loss0 = term.loss(target.view(), &rho).unwrap().total();
        let basis0 = term.atoms[0].basis_values.clone();

        let loss = term
            .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 2, 0.05, 1.0e-3, 1.0e-3)
            .unwrap();

        assert!(loss.total().is_finite());
        assert!(loss.total() <= loss0 + 1.0e-8);
        assert!(
            term.assignment.coords[0]
                .as_flat()
                .iter()
                .all(|v| v.is_finite())
        );
        assert!(term.assignment.assignments().iter().all(|v| v.is_finite()));
        let basis_delta = (&term.atoms[0].basis_values - &basis0).mapv(f64::abs).sum();
        assert!(basis_delta > 1.0e-10);
    }

    /// Regression test for https://github.com/SauersML/gam/issues/163.
    ///
    /// `ManifoldSAE.predict(X_subset)` reseeds the latent coordinates via PCA
    /// on a possibly small batch (here: a strict subset of the training data),
    /// which can produce a per-row `H_tt + ridge_t·I` that is not
    /// positive-definite at the caller's nominal `ridge_t = 1e-6`. The fit
    /// path tolerates this via the proximal LM correction outer wrapper;
    /// previously, `run_joint_fit_arrow_schur` invoked `sys.solve(...)`
    /// directly and surfaced the per-row Cholesky failure to the caller. The
    /// fix routes recoverable factor failures through a Levenberg-Marquardt
    /// damping schedule (mirrors the `proximal_correction` outer loop),
    /// so an inner step with a degenerate Hessian no longer aborts the
    /// Newton driver.
    #[test]
    fn run_joint_fit_arrow_schur_escalates_ridge_on_non_pd_row_block() {
        // Construct a periodic atom whose row block is rank-deficient when
        // the assignment column is zero — `H_tt` is then driven entirely by
        // the smoothness penalty / external coord ridge and floats just
        // above zero. At ridge_t = 1e-6 the per-row Cholesky finds a tiny
        // negative pivot from rounding error; the escalation loop should
        // recover.
        let coords = array![[0.1], [0.4], [0.7]];
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            // Decoder that maps to a single output dim with small magnitude
            array![[0.05], [-0.05], [0.05]],
            // No external smoothness penalty on the decoder, so the only
            // regularization on `t` comes from `ridge_ext_coord`.
            Array2::<f64>::zeros((3, 3)),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            // Zero assignment mass → H_tt has zero data contribution.
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.20], [-0.10], [0.45]];
        // log_lambda_smooth driven low so the analytic penalty contributes
        // essentially nothing to H_tt either.
        let mut rho = SaeManifoldRho::new(0.0, -20.0, vec![Array1::<f64>::zeros(1)]);

        // The Python-side `predict` default. Before the fix this returned
        // `Err(... per-row H_tt^(?) Cholesky failed ... non-PD pivot ...)`;
        // afterward the escalation loop bumps ridge_t until the per-row
        // factor succeeds, and run_joint_fit_arrow_schur returns Ok.
        let result =
            term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6);
        assert!(
            result.is_ok(),
            "run_joint_fit_arrow_schur should recover from degenerate H_tt via LM ridge escalation; got: {result:?}",
        );
    }

    /// Regression test for https://github.com/SauersML/gam/issues/163 and #175.
    ///
    /// `ManifoldSAE.reconstruct(X_oos)` (and `.predict(X_subset)`) reach the
    /// Rust core via `sae_manifold_predict_oos` → `sae_manifold_fit_inner` →
    /// the same `run_joint_fit_arrow_schur` Newton driver. The driver in turn
    /// calls `solve_newton_step` for single-shot refinement; before this fix
    /// that path invoked `sys.solve(...)` directly, bypassing the LM ridge
    /// escalation and surfacing the per-row Cholesky failure to the Python
    /// caller as `"row N H_tt was non-PD at ridge_t=0.000001"`. The fix routes
    /// `solve_newton_step` through `solve_with_lm_escalation` so every entry
    /// point — including OOS predict — geometrically grows the proximal ridge
    /// from the caller's nominal `ridge_ext_coord` / `ridge_beta` until the
    /// factor succeeds.
    #[test]
    fn solve_newton_step_escalates_ridge_on_non_pd_row_block() {
        // Same degenerate-H_tt construction as the predict/reconstruct
        // reproducer: zero assignment mass + zero smoothness penalty means
        // the only mass on H_tt comes from `ridge_t·I`, and at the nominal
        // 1e-6 the Cholesky still finds a tiny negative pivot from rounding.
        let coords = array![[0.1], [0.4], [0.7]];
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            array![[0.05], [-0.05], [0.05]],
            Array2::<f64>::zeros((3, 3)),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.20], [-0.10], [0.45]];
        let rho = SaeManifoldRho::new(0.0, -20.0, vec![Array1::<f64>::zeros(1)]);

        // Direct `solve_newton_step` call (the predict path's single-shot
        // refinement entry). Must Ok via LM escalation, not bubble up the
        // raw per-row factor failure.
        let result = term.solve_newton_step(target.view(), &rho, None, 1.0e-6, 1.0e-6);
        assert!(
            result.is_ok(),
            "solve_newton_step should recover from degenerate H_tt via LM ridge escalation; got: {result:?}",
        );
    }

    #[test]
    fn sae_arrow_schur_beta_quadratic_model_matches_penalized_loss_change() {
        let coords = array![[0.10], [0.35], [0.80]];
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            array![[0.65], [-0.45], [0.25]],
            array![[3.0, 0.4, -0.2], [0.1, 2.5, 0.3], [-0.5, 0.2, 1.8]],
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode(
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.20], [-0.10], [0.45]];
        let rho = SaeManifoldRho::new(0.0, 1.3_f64.ln(), vec![array![0.9_f64.ln()]]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();

        let beta0 = term.flatten_beta();
        let loss0 = term.loss(target.view(), &rho).unwrap().total();
        let mut direction = sys.gb.mapv(|v| -v);
        let direction_norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(direction_norm > 1.0e-12);
        for value in direction.iter_mut() {
            *value /= direction_norm;
        }

        let epsilon = 1.0e-3;
        let delta = direction.mapv(|v| epsilon * v);
        let beta_trial = beta0 + &delta;
        term.set_flat_beta(beta_trial.view()).unwrap();
        let actual = term.loss(target.view(), &rho).unwrap().total() - loss0;

        let linear = sys.gb.dot(&delta);
        let mut quadratic = 0.0;
        for row in 0..delta.len() {
            for col in 0..delta.len() {
                quadratic += 0.5 * delta[row] * sys.hbb[[row, col]] * delta[col];
            }
        }
        let predicted = linear + quadratic;
        let error = (actual - predicted).abs();
        assert!(
            error <= 1.0e-4,
            "actual={actual:.12e}, predicted={predicted:.12e}, error={error:.12e}"
        );
    }

    #[derive(Debug)]
    struct TestPeriodicEvaluator;

    impl SaeBasisEvaluator for TestPeriodicEvaluator {
        fn evaluate(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            Ok(periodic_basis(&coords.to_owned()))
        }
    }

    fn assert_jacobian_matches_central_difference<E: SaeBasisEvaluator>(
        evaluator: &E,
        coords: Array2<f64>,
        tolerance: f64,
    ) {
        let epsilon = 1.0e-6;
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let (n_rows, n_basis) = phi.dim();
        let latent_dim = coords.ncols();
        assert_eq!(jet.dim(), (n_rows, n_basis, latent_dim));

        for row in 0..n_rows {
            for axis in 0..latent_dim {
                let mut plus = coords.clone();
                let mut minus = coords.clone();
                plus[[row, axis]] += epsilon;
                minus[[row, axis]] -= epsilon;
                let (phi_plus, plus_jet) = evaluator.evaluate(plus.view()).unwrap();
                let (phi_minus, minus_jet) = evaluator.evaluate(minus.view()).unwrap();
                assert_eq!(plus_jet.dim(), jet.dim());
                assert_eq!(minus_jet.dim(), jet.dim());

                for basis in 0..n_basis {
                    let finite_difference =
                        (phi_plus[[row, basis]] - phi_minus[[row, basis]]) / (2.0 * epsilon);
                    let analytic = jet[[row, basis, axis]];
                    let error = (analytic - finite_difference).abs();
                    assert!(
                        error <= tolerance,
                        "row={row} basis={basis} axis={axis}: analytic={analytic:.12e}, \
                         finite_difference={finite_difference:.12e}, error={error:.12e}, \
                         tolerance={tolerance:.12e}"
                    );
                }
            }
        }
    }

    #[test]
    fn sae_basis_evaluator_jacobians_match_central_differences() {
        assert_jacobian_matches_central_difference(
            &PeriodicHarmonicEvaluator::new(7).unwrap(),
            array![[-0.37], [0.0], [0.125], [0.41]],
            1.0e-6,
        );

        assert_jacobian_matches_central_difference(
            &RawPeriodicCircleEvaluator::new(3).unwrap(),
            array![[-1.2, 0.3, 2.0], [0.0, -0.4, 0.8], [2.4, 1.1, -0.7]],
            1.0e-6,
        );

        let sphere_coords = array![[-0.7, -1.2], [-0.25, 0.0], [0.35, 0.9], [0.8, 2.1]];
        assert_jacobian_matches_central_difference(
            &SphereChartEvaluator,
            sphere_coords.clone(),
            1.0e-6,
        );
        let (sphere_phi, sphere_jet) = SphereChartEvaluator.evaluate(sphere_coords.view()).unwrap();
        assert_eq!(sphere_phi.dim(), (sphere_coords.nrows(), 7));
        assert_eq!(sphere_jet.dim(), (sphere_coords.nrows(), 7, 2));
        for row in 0..sphere_coords.nrows() {
            let lat = sphere_coords[[row, 0]];
            let lon = sphere_coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            let z = slat;
            let dx_dlon = -clat * slon;
            let dy_dlon = clat * clon;
            assert_eq!(sphere_jet[[row, 3, 1]], 0.0);
            assert!((sphere_jet[[row, 5, 1]] - dy_dlon * z).abs() <= 1.0e-12);
            assert!((sphere_jet[[row, 6, 1]] - dx_dlon * z).abs() <= 1.0e-12);
        }

        assert_jacobian_matches_central_difference(
            &AffineCoordinateEvaluator::new(3),
            array![[0.0, -1.0, 2.0], [3.5, 0.25, -0.75]],
            1.0e-6,
        );

        // Torus T^2 with H=3 → 49-column tensor product.
        let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
        assert_jacobian_matches_central_difference(
            &TorusHarmonicEvaluator::new(2, 3).unwrap(),
            torus_coords.clone(),
            1.0e-6,
        );
        let (torus_phi, torus_jet) = TorusHarmonicEvaluator::new(2, 3)
            .unwrap()
            .evaluate(torus_coords.view())
            .unwrap();
        assert_eq!(torus_phi.dim(), (torus_coords.nrows(), 49));
        assert_eq!(torus_jet.dim(), (torus_coords.nrows(), 49, 2));
        for row in 0..torus_coords.nrows() {
            // Column 0 = product of the two constant axis terms = 1.
            assert!((torus_phi[[row, 0]] - 1.0).abs() <= 1.0e-12);
            assert!(torus_jet[[row, 0, 0]].abs() <= 1.0e-12);
            assert!(torus_jet[[row, 0, 1]].abs() <= 1.0e-12);
        }
    }

    /// Torus T^2 fit on synthetic data with a known two-frequency signal.
    /// Drives a single torus atom through the [`SaeManifoldTerm`] Newton loop
    /// and checks that the in-sample reconstruction R² clears 0.5.
    #[test]
    fn sae_torus_atom_recovers_two_frequency_synthetic() {
        let n = 96usize;
        let p = 4usize;
        let h = 3usize;
        let d = 2usize;
        let evaluator = TorusHarmonicEvaluator::new(d, h).unwrap();
        let m = evaluator.basis_size();
        // True coords on T^2 (phase in [0, 1)).
        let mut true_coords = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            true_coords[[i, 0]] = ((i as f64) * 0.137).rem_euclid(1.0);
            true_coords[[i, 1]] = ((i as f64) * 0.241 + 0.13).rem_euclid(1.0);
        }
        // Synthetic target: a low-frequency periodic signal on T^2 mixed
        // linearly into a p-dim ambient.
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t1 = 2.0 * std::f64::consts::PI * true_coords[[i, 0]];
            let t2 = 2.0 * std::f64::consts::PI * true_coords[[i, 1]];
            z[[i, 0]] = t1.sin() + 0.3 * t2.cos();
            z[[i, 1]] = t1.cos() + 0.2 * (t1 + t2).sin();
            z[[i, 2]] = t2.sin();
            z[[i, 3]] = 0.5 * (t1 - t2).cos();
        }
        let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
        // Initialise from the true coords (this test exercises basis correctness
        // and decoder fit, not coordinate identification on T^2).
        let (phi0, jet0) = evaluator.evaluate(true_coords.view()).unwrap();
        // Penalty: identity-on-non-constant + tiny floor on constant.
        let mut penalty = Array2::<f64>::eye(m);
        penalty *= 1.0e-4;
        let atom = SaeManifoldAtom::new(
            "torus_atom",
            SaeAtomBasisKind::Torus,
            d,
            phi0,
            jet0,
            Array2::<f64>::zeros((m, p)),
            penalty,
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TorusHarmonicEvaluator::new(d, h).unwrap()));

        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![true_coords],
            vec![LatentManifold::Product(vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ])],
            AssignmentMode::softmax(0.5),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(1)]);
        let ridge = 1.0e-6;
        for _ in 0..10 {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
                .unwrap();
            if !loss.total().is_finite() {
                break;
            }
        }
        let fitted = term.fitted();
        assert_eq!(fitted.dim(), (n, p));
        let mut sse = 0.0_f64;
        for ((row, col), v) in fitted.indexed_iter() {
            let r = v - z[[row, col]];
            sse += r * r;
        }
        let r2 = 1.0 - sse / sst.max(1.0e-12);
        assert!(
            r2 >= 0.5,
            "torus atom R² too low: {r2:.4} (sst={sst:.4}, sse={sse:.4})"
        );
    }

    /// Sphere S² fit on a synthetic spherical signal. Drives a single sphere
    /// atom through the [`SaeManifoldTerm`] Newton loop and checks in-sample
    /// R² ≥ 0.5.
    #[test]
    fn sae_sphere_atom_recovers_synthetic_signal() {
        let n = 96usize;
        let p = 3usize;
        let d = 2usize;
        // True (lat, lon) coords.
        let mut true_coords = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let t = (i as f64) / (n as f64);
            true_coords[[i, 0]] = -0.5 + 1.0 * t; // lat in [-0.5, 0.5]
            true_coords[[i, 1]] = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * t;
        }
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let lat = true_coords[[i, 0]];
            let lon = true_coords[[i, 1]];
            let x = lat.cos() * lon.cos();
            let y = lat.cos() * lon.sin();
            let zc = lat.sin();
            z[[i, 0]] = x;
            z[[i, 1]] = y;
            z[[i, 2]] = zc;
        }
        let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
        let (phi0, jet0) = SphereChartEvaluator.evaluate(true_coords.view()).unwrap();
        let m = phi0.ncols();
        let mut penalty = Array2::<f64>::eye(m);
        penalty *= 1.0e-4;
        let atom = SaeManifoldAtom::new(
            "sphere_atom",
            SaeAtomBasisKind::Sphere,
            d,
            phi0,
            jet0,
            Array2::<f64>::zeros((m, p)),
            penalty,
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(SphereChartEvaluator));

        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![true_coords],
            vec![LatentManifold::Product(vec![
                LatentManifold::Interval {
                    lo: -std::f64::consts::FRAC_PI_2,
                    hi: std::f64::consts::FRAC_PI_2,
                },
                LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                },
            ])],
            AssignmentMode::softmax(0.5),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(1)]);
        let ridge = 1.0e-6;
        for _ in 0..10 {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
                .unwrap();
            if !loss.total().is_finite() {
                break;
            }
        }
        let fitted = term.fitted();
        assert_eq!(fitted.dim(), (n, p));
        let mut sse = 0.0_f64;
        for ((row, col), v) in fitted.indexed_iter() {
            let r = v - z[[row, col]];
            sse += r * r;
        }
        let r2 = 1.0 - sse / sst.max(1.0e-12);
        assert!(
            r2 >= 0.5,
            "sphere atom R² too low: {r2:.4} (sst={sst:.4}, sse={sse:.4})"
        );
    }

    /// Mirror of the Python `test_sae_manifold_softmax_dispatch` shape: drive a
    /// single periodic atom on a 1-harmonic synthetic target with 10 Newton
    /// steps end-to-end in Rust and check that the multi-step loop achieves
    /// in-sample R² ≥ 0.95.
    #[test]
    fn sae_manifold_fit_10_steps_one_harmonic_reaches_high_r2() {
        let n = 64usize;
        let m = 3usize;
        let p = 1usize;

        let true_t: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * true_t[i];
            z[[i, 0]] = 0.7 * angle.sin() + 0.3 * angle.cos();
        }
        let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();

        let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
        let mut coords0_data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            // Phase-shifted initialization so the optimizer must do real work.
            coords0_data[[i, 0]] = (true_t[i] + 0.25).rem_euclid(1.0);
        }
        let (phi0, jet0) = evaluator.evaluate(coords0_data.view()).unwrap();

        let atom = SaeManifoldAtom::new(
            "periodic_atom",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            Array2::<f64>::zeros((m, p)),
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));

        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords0_data],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.5),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);

        let max_iter = 10usize;
        let learning_rate = 1.0;
        let ridge = 1.0e-6;
        let mut prev_total = f64::INFINITY;
        for _ in 0..max_iter {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, learning_rate, ridge, ridge)
                .unwrap();
            let total = loss.total();
            if !total.is_finite() {
                break;
            }
            let denom = prev_total.abs().max(1.0e-12);
            let rel = (prev_total - total).abs() / denom;
            prev_total = total;
            if rel < 1.0e-6 {
                break;
            }
        }

        let fitted = term.fitted();
        assert_eq!(fitted.dim(), (n, p));
        let mut ssr = 0.0;
        for i in 0..n {
            let r = z[[i, 0]] - fitted[[i, 0]];
            ssr += r * r;
        }
        let r2 = 1.0 - ssr / sst.max(1.0e-12);
        assert!(
            r2 >= 0.95,
            "10-step in-sample R² = {r2:.4} (ssr={ssr:.6}, sst={sst:.6}) should be >= 0.95"
        );
    }

    /// Regression test for issue #177: softmax assignment used to bail out of
    /// the row-block Hessian assembly with "softmax assignment hessian diag
    /// unavailable". The penalty now exposes the analytic diagonal extracted
    /// from its row-dense HVP, so the joint-fit driver completes one step.
    #[test]
    fn softmax_assignment_hessian_diag_is_available_for_k2() {
        let n = 4usize;
        let k = 2usize;
        let logits =
            Array2::<f64>::from_shape_fn((n, k), |(i, j)| 0.1 * (i as f64) - 0.2 * (j as f64));
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            manifolds,
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);
        let (grad, diag) = assignment_prior_grad_hdiag(&assignment, &rho)
            .expect("softmax assignment Hessian diagonal must be available");
        assert_eq!(grad.len(), n * k);
        assert_eq!(diag.len(), n * k);
        assert!(grad.iter().all(|v| v.is_finite()));
        assert!(diag.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn jumprelu_assignment_prior_hessian_diag_is_psd_over_logit_sweep() {
        let n = 6usize;
        let k = 2usize;
        let temperature = 0.35_f64;
        let threshold = 0.1_f64;
        let logits = Array2::<f64>::from_shape_vec(
            (n, k),
            vec![
                -2.0, -0.2, 0.0, 0.05, 0.1, 0.15, 0.4, 0.9, 1.5, 2.5, 4.0, 6.0,
            ],
        )
        .expect("valid logit grid");
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            coords,
            manifolds,
            AssignmentMode::jumprelu(temperature, threshold),
        )
        .expect("valid JumpReLU assignment");
        let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
        let (grad, diag) = assignment_prior_grad_hdiag(&assignment, &rho)
            .expect("JumpReLU assignment prior hessian diag");
        let inv_tau = 1.0 / temperature;
        let inv_tau2 = inv_tau * inv_tau;
        let sparsity_strength = rho.log_lambda_sparse.exp();

        assert_eq!(grad.len(), n * k);
        assert_eq!(diag.len(), n * k);
        for (idx, &entry) in diag.iter().enumerate() {
            let logit = logits[[idx / k, idx % k]];
            let expected = if logit > threshold {
                let activation = sigmoid_scalar(logit * inv_tau);
                let slope = activation * (1.0 - activation);
                sparsity_strength * slope * slope * inv_tau2
            } else {
                0.0
            };
            assert!(
                entry.is_finite() && entry >= 0.0,
                "JumpReLU gated hessian_diag majorizer must be finite and PSD at index {idx}; entry={entry}"
            );
            assert_abs_diff_eq!(entry, expected, epsilon = 1e-12);
        }
    }

    /// Regression test for issue #174: K>=2 periodic atoms with zero-init
    /// decoder used to collapse to A≈0 because the assignment prior was the
    /// only term with non-zero gradient at iter 0. The pyffi entry point now
    /// seeds decoder coefficients via a joint LSQ projection of Z onto
    /// [a_init · Phi_k]. This test exercises that exact seeding strategy
    /// in pure Rust and verifies the joint Newton fit reaches positive R²
    /// on a clean K=2 periodic torus signal, mirroring the failing
    /// reproducer in #174.
    #[test]
    fn ibp_map_k2_periodic_torus_recovers_signal_with_lsq_init() {
        use crate::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
        use faer::Side as FaerSide;

        let n = 200usize;
        let p = 8usize;
        let k = 2usize;
        let m = 5usize; // 1 (constant) + 2 harmonics * 2 (sin/cos) = 5

        // Build a synthetic K=2 torus signal Z = [cos th1, sin th1, cos th2, sin th2] @ mix
        // with two latent angles. Deterministic seed via index arithmetic.
        let mut theta = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            theta[[i, 0]] = ((i as f64) * 0.07) % 1.0;
            theta[[i, 1]] = ((i as f64) * 0.13 + 0.31) % 1.0;
        }
        let mut raw = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let a1 = 2.0 * std::f64::consts::PI * theta[[i, 0]];
            let a2 = 2.0 * std::f64::consts::PI * theta[[i, 1]];
            raw[[i, 0]] = a1.cos();
            raw[[i, 1]] = a1.sin();
            raw[[i, 2]] = a2.cos();
            raw[[i, 3]] = a2.sin();
        }
        // Deterministic 4x8 mixing matrix.
        let mix = Array2::<f64>::from_shape_fn((4, p), |(i, j)| {
            ((i as f64 + 1.0) * 0.37 + (j as f64) * 0.21).sin()
        });
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                let mut acc = 0.0;
                for r in 0..4 {
                    acc += raw[[i, r]] * mix[[r, j]];
                }
                z[[i, j]] = acc;
            }
        }
        // Centre Z so R² is well-defined relative to mean.
        let mut col_mean = Array1::<f64>::zeros(p);
        for j in 0..p {
            let mut acc = 0.0;
            for i in 0..n {
                acc += z[[i, j]];
            }
            col_mean[j] = acc / n as f64;
        }
        for i in 0..n {
            for j in 0..p {
                z[[i, j]] -= col_mean[j];
            }
        }

        // Atom coordinates: use the (shifted) true angles so the periodic
        // basis aligns with the signal — the test isolates the decoder-init
        // collapse, not coordinate recovery.
        let mut coords_k = vec![Array2::<f64>::zeros((n, 1)); k];
        for i in 0..n {
            coords_k[0][[i, 0]] = (theta[[i, 0]] + 0.05).rem_euclid(1.0);
            coords_k[1][[i, 0]] = (theta[[i, 1]] + 0.07).rem_euclid(1.0);
        }
        // Periodic basis (constant + 2 harmonics → M=5) for each atom.
        let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
        let mut phi_k = Vec::with_capacity(k);
        let mut jet_k = Vec::with_capacity(k);
        for atom_idx in 0..k {
            let (phi, jet) = evaluator.evaluate(coords_k[atom_idx].view()).unwrap();
            phi_k.push(phi);
            jet_k.push(jet);
        }

        // LSQ seed: joint design X = [0.5 * Phi_1 | 0.5 * Phi_2] (IBP-MAP
        // logit 0 gives sigmoid(0/tau) = 0.5 for both atoms), solve normal
        // equations with a small ridge.
        let m_total = k * m;
        let mut x = Array2::<f64>::zeros((n, m_total));
        for atom_idx in 0..k {
            for i in 0..n {
                for col in 0..m {
                    x[[i, atom_idx * m + col]] = 0.5 * phi_k[atom_idx][[i, col]];
                }
            }
        }
        let mut xtx = fast_ata(&x);
        let mut trace = 0.0_f64;
        for i in 0..m_total {
            trace += xtx[[i, i]];
        }
        let jitter = (trace / m_total as f64).max(1.0) * 1.0e-8;
        for i in 0..m_total {
            xtx[[i, i]] += jitter;
        }
        let xtz = fast_atb(&x, &z);
        let b_joint = xtx
            .cholesky(FaerSide::Lower)
            .expect("LSQ Cholesky")
            .solve_mat(&xtz);

        let mut atoms = Vec::with_capacity(k);
        for atom_idx in 0..k {
            let mut b = Array2::<f64>::zeros((m, p));
            for col in 0..m {
                for j in 0..p {
                    b[[col, j]] = b_joint[[atom_idx * m + col, j]];
                }
            }
            let atom = SaeManifoldAtom::new(
                format!("torus_atom_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi_k[atom_idx].clone(),
                jet_k[atom_idx].clone(),
                b,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));
            atoms.push(atom);
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, k)),
            coords_k,
            vec![LatentManifold::Circle { period: 1.0 }; k],
            AssignmentMode::ibp_map(0.7, 1.0, false),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);

        let mut prev_total = f64::INFINITY;
        for _ in 0..30 {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
                .unwrap();
            let total = loss.total();
            if !total.is_finite() {
                break;
            }
            let denom = prev_total.abs().max(1.0e-12);
            let rel = (prev_total - total).abs() / denom;
            prev_total = total;
            if rel < 1.0e-6 {
                break;
            }
        }

        let fitted = term.fitted();
        let mut ssr = 0.0;
        let mut sst = 0.0;
        for i in 0..n {
            for j in 0..p {
                let r = z[[i, j]] - fitted[[i, j]];
                ssr += r * r;
                sst += z[[i, j]] * z[[i, j]];
            }
        }
        let r2 = 1.0 - ssr / sst.max(1.0e-12);
        assert!(
            r2 > 0.5,
            "K=2 periodic torus IBP-MAP R² = {r2:.4} (ssr={ssr:.4}, sst={sst:.4}) should be > 0.5 with LSQ-seeded decoder"
        );
        // Also confirm at least one atom remains active (assignment did not
        // collapse to ~0) — the active mass averaged over rows must exceed
        // a non-trivial threshold.
        let assignments = term.assignment.assignments();
        let mean_active: f64 = assignments.iter().copied().sum::<f64>() / (n as f64);
        assert!(
            mean_active > 0.2,
            "mean active mass across rows = {mean_active:.4} should exceed 0.2; assignment did not collapse"
        );
    }

    /// Regression test for issue #174 + #177 combined: softmax assignment
    /// with K=2 periodic atoms should not crash and should reduce loss.
    #[test]
    fn softmax_k2_periodic_completes_joint_fit_step() {
        let n = 64usize;
        let p = 4usize;
        let k = 2usize;
        let m = 3usize;

        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            z[[i, 0]] = a.sin();
            z[[i, 1]] = a.cos();
            z[[i, 2]] = (2.0 * a).sin();
            z[[i, 3]] = (2.0 * a).cos();
        }

        let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
        let mut coords_k = vec![Array2::<f64>::zeros((n, 1)); k];
        for i in 0..n {
            coords_k[0][[i, 0]] = (i as f64) / (n as f64);
            coords_k[1][[i, 0]] = ((i as f64) * 2.0 / (n as f64)).rem_euclid(1.0);
        }
        let mut atoms = Vec::new();
        for atom_idx in 0..k {
            let (phi, jet) = evaluator.evaluate(coords_k[atom_idx].view()).unwrap();
            // Non-trivial decoder init (simulate LSQ seeding) so the data-fit
            // signal is non-zero at iter 0.
            let b = Array2::<f64>::from_shape_fn((m, p), |(i, j)| {
                0.1 * ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
            });
            let atom = SaeManifoldAtom::new(
                format!("a_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                b,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));
            atoms.push(atom);
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, k)),
            coords_k,
            vec![LatentManifold::Circle { period: 1.0 }; k],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);

        // First step must succeed (previously bailed with hessian-diag error).
        let loss0 = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
            .expect("softmax K=2 must complete first joint-fit step");
        assert!(loss0.total().is_finite());
        let loss1 = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
            .expect("softmax K=2 must complete second joint-fit step");
        assert!(loss1.total().is_finite());
    }
}
