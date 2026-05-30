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

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, s};
use std::sync::Arc;

use crate::solver::arrow_schur::{
    ArrowRowBlock, ArrowSchurError, ArrowSchurSystem, ArrowSolveOptions, BetaPenaltyOp,
    CompositePenaltyOp, DensePenaltyOp, KroneckerPenaltyOp, SparseBlockKroneckerPenaltyOp,
    SparseGBlock, StreamingArrowSchur, solve_streaming_reduced_beta,
};
use crate::terms::analytic_penalties::{
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry,
    IBPAssignmentPenalty, IsometryPenalty, MechanismSparsityPenalty, PenaltyTier, PsiSlice,
    SoftmaxAssignmentSparsityPenalty,
};
use crate::terms::latent_coord::{LatentCoordValues, LatentIdMode, LatentManifold};

use crate::linalg::faer_ndarray::FaerEigh;
use crate::solver::arrow_schur::{ArrowFactorCache, solve_arrow_newton_step_with_options};
use crate::solver::estimate::EstimationError;
use crate::solver::evidence::arrow_log_det_from_cache;
use crate::solver::outer_strategy::{
    DeclaredHessianForm, Derivative, EfsEval, HessianResult, OuterCapability, OuterEval,
    OuterObjective, SeedOutcome,
};
use faer::Side;

const SAE_MANIFOLD_ARMIJO_C1: f64 = 1.0e-4;
const SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS: usize = 12;

/// Fixed precision of the proper Gaussian identifiability prior on the
/// assignment logits, `½·κ·‖logit_row‖²` per observation row.
///
/// The softmax assignment is shift-invariant per row (adding a constant to
/// every logit leaves `a = softmax(logit)` unchanged), so the logit
/// Gauss-Newton curvature along the all-ones direction is *exactly* zero for
/// every `K`: with the JVP `∂a_k/∂l_j = a_k(δ_kj − a_j)/τ`, the shift
/// direction gives `Σ_j ∂a_k/∂l_j = a_k(1 − Σ_j a_j)/τ = 0`. For the
/// single-atom `K=1` case the whole logit block is structurally zero. The
/// assignment-sparsity prior (`assignment_hdiag`) does not in general cover
/// that null direction. Left unregularised, the per-row latent Hessian
/// `H_tt` is rank-deficient, so the REML/Laplace evidence `½log|H|` and the
/// undamped selected-inverse traces are ill-defined at `ridge_t = 0`.
///
/// This κ is a *proper* prior (it enters the loss, gradient, and Hessian
/// diagonal consistently), not a solver jitter: it pins the otherwise-flat
/// logit gauge to 0, exactly as `mgcv` ridges an unpenalised smooth null
/// space. It is a fixed model constant — independent of ρ — so it shifts the
/// evidence by a ρ-independent amount and does not bias ρ-optimisation, while
/// making `H_tt` positive-definite by construction so the exact undamped
/// `arrow_log_det_from_cache` evidence and the IFT/EFS traces are well-posed.
const SAE_LOGIT_IDENTIFIABILITY_PRECISION: f64 = 1.0e-4;

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
            // fraction of one period, not radians. The latent manifold
            // wraps modulo `period = 1.0` to match this convention.
            // Wrapping modulo `2π` instead would scramble the
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

    /// Object-safe forwarder to [`SaeBasisSecondJet::second_jet`] for callers
    /// holding `&dyn SaeBasisEvaluator` / `Arc<dyn SaeBasisEvaluator>`.
    ///
    /// Default returns `None`, meaning "this evaluator has no analytic second
    /// jet". Evaluators that also implement [`SaeBasisSecondJet`] override
    /// this to wrap the typed call in `Some(...)`. This sidesteps the lack of
    /// dyn-downcasting for non-`Any` traits while keeping `SaeBasisSecondJet`
    /// as the strongly-typed compile-time bound for tests / generics.
    fn second_jet_dyn(&self, _coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        None
    }
}

/// Bases that expose an analytic second jet
/// `H[n, m, a, c] = ∂²Phi_k[n, m] / (∂t_{n,a} ∂t_{n,c})`,
/// shape `(n_rows, n_basis, latent_dim, latent_dim)`.
///
/// Implemented only by evaluators with a closed-form Hessian (periodic
/// harmonic, sphere chart, torus). Callers that need an analytic
/// `∂J/∂t` require this bound; evaluators without it must use a
/// derivative-free fallback. Replaces the previous `Option<Array4<f64>>`
/// return on the base trait so the "no second jet" case is encoded by
/// trait absence rather than a sentinel `None`, and shape mismatches
/// surface as descriptive errors instead of silently collapsing to
/// `None`.
pub trait SaeBasisSecondJet: SaeBasisEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String>;
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
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

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

impl SaeBasisSecondJet for PeriodicHarmonicEvaluator {
    /// Second derivative of the 1D Fourier basis on the unit circle.
    ///
    /// For `Phi = [1, sin(2π h t), cos(2π h t), ...]` we have
    /// `Phi'' = [0, -(2π h)² sin(...), -(2π h)² cos(...), ...]`, i.e.
    /// the second derivative is `-(2π h)² · phi(t)` on each harmonic pair.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator::second_jet: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut h = Array4::<f64>::zeros((n, m, 1, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            for k in 1..=num_harmonics {
                let freq = two_pi * (k as f64);
                let freq2 = freq * freq;
                let angle = freq * t;
                let s = angle.sin();
                let c = angle.cos();
                let s_idx = 2 * k - 1;
                let c_idx = 2 * k;
                h[[row, s_idx, 0, 0]] = -freq2 * s;
                h[[row, c_idx, 0, 0]] = -freq2 * c;
            }
        }
        Ok(h)
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

/// Diagonal of the chart-local seven-column sphere basis penalty.
///
/// The columns are `[1, x, y, z, xy, yz, xz]`; the constant column carries a
/// numerically-negligible ridge (`1e-8`) so the penalty stays positive
/// definite, the three linear columns are penalized at unit weight, and the
/// three bilinear columns at weight `4` (their second-order angular content).
/// This is the single source of truth for the chart penalty shared between the
/// core SAE path and the PyFFI `sphere_chart_basis_with_jet` helper.
pub const SPHERE_CHART_PENALTY_DIAGONAL: [f64; 7] = [1e-8, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0];

/// Shared single source of truth for the chart-local sphere basis and its
/// analytic first-derivative (lat/lon) jet.
///
/// `coords` is an `(N, 2)` array of latitude/longitude pairs in radians. The
/// returned `phi` has shape `(N, 7)` with columns `[1, x, y, z, xy, yz, xz]`
/// for the unit-sphere embedding `x = cos(lat)cos(lon)`, `y = cos(lat)sin(lon)`,
/// `z = sin(lat)`; the returned `jet` has shape `(N, 7, 2)` with the last axis
/// indexing `[∂/∂lat, ∂/∂lon]`.
///
/// Latitude is clamped to `[-π/2, π/2]`. The clamp truncates derivatives w.r.t.
/// the raw input coordinate outside the interior `(-π/2, π/2)`: in the
/// saturated region the `phi` entries are constant in `coords[[row, 0]]`, so the
/// chain rule contributes a zero factor on the `∂/∂lat` axis. Failing to apply
/// this `chain_lat` gating leaks a non-zero analytic gradient where finite
/// differences correctly report zero, sending Newton steps in lat in a
/// direction the loss does not actually decrease along. Both the core path
/// ([`SphereChartEvaluator`]) and the PyFFI helper route through this function,
/// so the saturated-latitude gating is identical everywhere.
pub fn sphere_chart_basis_jet(
    coords: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    if coords.ncols() != 2 {
        return Err(format!(
            "sphere_chart_basis_jet expects latent_dim == 2, got {}",
            coords.ncols()
        ));
    }
    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 7));
    let mut jet = Array3::<f64>::zeros((n, 7, 2));
    for row in 0..n {
        let raw_lat = coords[[row, 0]];
        let lat = raw_lat.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
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

/// Lat/lon sphere chart evaluator used by the Rust-owned minimal SAE path.
#[derive(Debug, Clone)]
pub struct SphereChartEvaluator;

impl SaeBasisEvaluator for SphereChartEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        sphere_chart_basis_jet(coords)
    }
}

impl SaeBasisSecondJet for SphereChartEvaluator {
    /// Analytic Hessian of the 7-column lat/lon sphere chart basis.
    ///
    /// With `x = cos(lat) cos(lon)`, `y = cos(lat) sin(lon)`, `z = sin(lat)`
    /// and `a = 1{lat ∈ (−π/2, π/2)}` (the clamp's chain factor; outside the
    /// interior every lat-partial is zero, including in the Hessian), the
    /// non-trivial second derivatives are
    ///
    /// ```text
    /// x_{lat,lat} = -x · a,     x_{lon,lon} = -x,     x_{lat,lon} = sin(lat)·sin(lon)·a
    /// y_{lat,lat} = -y · a,     y_{lon,lon} = -y,     y_{lat,lon} = -sin(lat)·cos(lon)·a
    /// z_{lat,lat} = -z · a,     z_{lon,lon} =  0,     z_{lat,lon} =  0
    /// ```
    ///
    /// Bilinear basis entries `xy, yz, xz` follow the product rule
    /// `(fg)_{αβ} = f_{αβ} g + f_α g_β + f_β g_α + f g_{αβ}`. The boundary
    /// chain factor `a` is idempotent (`a² = a`), so reapplying it on a
    /// double-lat derivative is a no-op.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "SphereChartEvaluator::second_jet expects latent_dim == 2, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut h = Array4::<f64>::zeros((n, 7, 2, 2));
        for row in 0..n {
            let raw_lat = coords[[row, 0]];
            let lat = raw_lat.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
            let lat_active =
                raw_lat > -std::f64::consts::FRAC_PI_2 && raw_lat < std::f64::consts::FRAC_PI_2;
            let a = if lat_active { 1.0 } else { 0.0 };
            let lon = coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            let x = clat * clon;
            let y = clat * slon;
            let z = slat;
            let dx = [-slat * clon * a, -clat * slon];
            let dy = [-slat * slon * a, clat * clon];
            let dz = [clat * a, 0.0];
            let hx = [[-x * a, slat * slon * a], [slat * slon * a, -x]];
            let hy = [[-y * a, -slat * clon * a], [-slat * clon * a, -y]];
            let hz = [[-z * a, 0.0], [0.0, 0.0]];
            for axis_a in 0..2 {
                for axis_b in 0..2 {
                    h[[row, 1, axis_a, axis_b]] = hx[axis_a][axis_b];
                    h[[row, 2, axis_a, axis_b]] = hy[axis_a][axis_b];
                    h[[row, 3, axis_a, axis_b]] = hz[axis_a][axis_b];
                }
            }
            let pair = |hf: [[f64; 2]; 2],
                        df: [f64; 2],
                        f: f64,
                        hg: [[f64; 2]; 2],
                        dg: [f64; 2],
                        g: f64|
             -> [[f64; 2]; 2] {
                let mut out = [[0.0; 2]; 2];
                for axis_a in 0..2 {
                    for axis_b in 0..2 {
                        out[axis_a][axis_b] = hf[axis_a][axis_b] * g
                            + df[axis_a] * dg[axis_b]
                            + df[axis_b] * dg[axis_a]
                            + f * hg[axis_a][axis_b];
                    }
                }
                out
            };
            let hxy = pair(hx, dx, x, hy, dy, y);
            let hyz = pair(hy, dy, y, hz, dz, z);
            let hxz = pair(hx, dx, x, hz, dz, z);
            for axis_a in 0..2 {
                for axis_b in 0..2 {
                    h[[row, 4, axis_a, axis_b]] = hxy[axis_a][axis_b];
                    h[[row, 5, axis_a, axis_b]] = hyz[axis_a][axis_b];
                    h[[row, 6, axis_a, axis_b]] = hxz[axis_a][axis_b];
                }
            }
        }
        Ok(h)
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
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

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

impl SaeBasisSecondJet for TorusHarmonicEvaluator {
    /// Hessian of the tensor-product torus basis.
    ///
    /// Each basis function factors as `Φ_flat = Π_axis f_axis(t_axis)`, so
    ///
    /// * `∂² Φ / ∂t_a ∂t_b = (Π_{k ∉ {a, b}} f_k) · f_a'(t_a) · f_b'(t_b)`
    ///   when `a ≠ b`,
    /// * `∂² Φ / ∂t_a²    = (Π_{k ≠ a} f_k) · f_a''(t_a)` on the diagonal.
    ///
    /// Per-axis the basis is `[1, sin(2π h t), cos(2π h t), …]`, so
    /// `f_axis''(t) = -(2π h)² · f_axis(t)` on the harmonic columns and 0 on
    /// the constant column.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "TorusHarmonicEvaluator::second_jet expects latent_dim == {d}, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let axis_m = self.axis_basis_size();
        let m = self.basis_size();
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut hess = Array4::<f64>::zeros((n, m, d, d));
        let mut phi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut dphi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut d2phi_axis = vec![vec![0.0_f64; axis_m]; d];
        for row in 0..n {
            for axis in 0..d {
                let t = coords[[row, axis]];
                phi_axis[axis][0] = 1.0;
                dphi_axis[axis][0] = 0.0;
                d2phi_axis[axis][0] = 0.0;
                for k in 1..=h_max {
                    let freq = two_pi * (k as f64);
                    let freq2 = freq * freq;
                    let angle = freq * t;
                    let s = angle.sin();
                    let c = angle.cos();
                    let s_idx = 2 * k - 1;
                    let c_idx = 2 * k;
                    phi_axis[axis][s_idx] = s;
                    phi_axis[axis][c_idx] = c;
                    dphi_axis[axis][s_idx] = freq * c;
                    dphi_axis[axis][c_idx] = -freq * s;
                    d2phi_axis[axis][s_idx] = -freq2 * s;
                    d2phi_axis[axis][c_idx] = -freq2 * c;
                }
            }
            let mut idx = vec![0usize; d];
            for flat in 0..m {
                for axis_a in 0..d {
                    for axis_b in 0..d {
                        let mut prod = 1.0_f64;
                        for axis in 0..d {
                            let factor = if axis == axis_a && axis == axis_b {
                                d2phi_axis[axis][idx[axis]]
                            } else if axis == axis_a || axis == axis_b {
                                dphi_axis[axis][idx[axis]]
                            } else {
                                phi_axis[axis][idx[axis]]
                            };
                            prod *= factor;
                        }
                        hess[[row, flat, axis_a, axis_b]] = prod;
                    }
                }
                for axis in (0..d).rev() {
                    idx[axis] += 1;
                    if idx[axis] < axis_m {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
        Ok(hess)
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
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

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

impl SaeBasisSecondJet for AffineCoordinateEvaluator {
    /// Second derivative of the affine basis `[1, t_1, ..., t_d]`.
    ///
    /// Every basis function is at most linear in `t`, so all second derivatives
    /// are identically zero. Returns the all-zeros tensor of shape
    /// `(n_obs, d+1, d, d)`.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "AffineCoordinateEvaluator::second_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let m = self.latent_dim + 1;
        let d = self.latent_dim;
        Ok(Array4::<f64>::zeros((n, m, d, d)))
    }
}

/// Scale-free Duchon atom evaluator for the SAE-manifold Newton loop.
///
/// Recomputes the radial+polynomial design `Φ(t)` and its first/second
/// input-location jets at arbitrary latent coordinates against a fixed set of
/// `centers` and Duchon null-space `order`. The column layout — the
/// kernel block `Φ_radial(t)·Z` followed by the polynomial block `P(t)`,
/// both carrying the same scalar kernel amplification `α` — matches
/// [`crate::basis::build_duchon_basis`] under the SAE atom's spec
/// (`length_scale = None`, `power = 0`, no identifiability transform). The
/// forward design and the jet are produced from a single core entry point
/// ([`crate::basis::duchon_sae_atom_basis_with_jet`]) so they always agree on
/// column count and scaling — the exact contract issue #247 pinned.
#[derive(Debug, Clone)]
pub struct DuchonCoordinateEvaluator {
    pub centers: Array2<f64>,
    pub order: crate::basis::DuchonNullspaceOrder,
}

impl DuchonCoordinateEvaluator {
    /// Build from the atom's centers and Duchon `m` (`m = 1` → constant
    /// null space, `m = 2` → constant+linear, `m = k+1` → degree-`k`).
    pub fn new(centers: Array2<f64>, m: usize) -> Result<Self, String> {
        if centers.ncols() == 0 {
            return Err("DuchonCoordinateEvaluator: centers must have at least one column".into());
        }
        if m == 0 {
            return Err("DuchonCoordinateEvaluator: Duchon m must be at least 1".into());
        }
        let order = match m {
            1 => crate::basis::DuchonNullspaceOrder::Zero,
            2 => crate::basis::DuchonNullspaceOrder::Linear,
            other => crate::basis::DuchonNullspaceOrder::Degree(other - 1),
        };
        Ok(Self { centers, order })
    }
}

impl SaeBasisEvaluator for DuchonCoordinateEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.centers.ncols() {
            return Err(format!(
                "DuchonCoordinateEvaluator: expected latent_dim {}, got {}",
                self.centers.ncols(),
                coords.ncols()
            ));
        }
        crate::basis::duchon_sae_atom_basis_with_jet(coords, self.centers.view(), self.order)
            .map_err(|err| err.to_string())
    }
}

impl SaeBasisSecondJet for DuchonCoordinateEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.centers.ncols() {
            return Err(format!(
                "DuchonCoordinateEvaluator::second_jet: expected latent_dim {}, got {}",
                self.centers.ncols(),
                coords.ncols()
            ));
        }
        crate::basis::duchon_sae_atom_second_jet(coords, self.centers.view(), self.order)
            .map_err(|err| err.to_string())
    }
}

/// Flat Euclidean tangent-patch evaluator for the SAE-manifold Newton loop.
///
/// The basis is the set of monomials of total degree ≤ `max_degree` in the
/// atom's latent coordinates (a zero-curvature polynomial expansion, distinct
/// from the thin-plate Duchon kernel). It recomputes the monomial design and
/// its first/second derivatives at arbitrary coordinates, so the inner Newton
/// latent update stays consistent with the deployed design.
#[derive(Debug, Clone)]
pub struct EuclideanPatchEvaluator {
    pub latent_dim: usize,
    pub max_degree: usize,
}

impl EuclideanPatchEvaluator {
    pub fn new(latent_dim: usize, max_degree: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("EuclideanPatchEvaluator: latent_dim must be positive".into());
        }
        Ok(Self {
            latent_dim,
            max_degree,
        })
    }

    fn order(&self) -> crate::basis::DuchonNullspaceOrder {
        match self.max_degree {
            0 => crate::basis::DuchonNullspaceOrder::Zero,
            1 => crate::basis::DuchonNullspaceOrder::Linear,
            k => crate::basis::DuchonNullspaceOrder::Degree(k),
        }
    }
}

impl SaeBasisEvaluator for EuclideanPatchEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = crate::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        let mut phi = Array2::<f64>::zeros((n, m));
        for (col, alpha) in exponents.iter().enumerate() {
            for row in 0..n {
                let mut value = 1.0_f64;
                for (axis, &exp) in alpha.iter().enumerate() {
                    if exp != 0 {
                        value *= coords[[row, axis]].powi(exp as i32);
                    }
                }
                phi[[row, col]] = value;
            }
        }
        let jet = crate::basis::duchon_polynomial_first_derivative_nd(coords, self.order());
        if jet.shape() != [n, m, self.latent_dim] {
            return Err(format!(
                "EuclideanPatchEvaluator: monomial jet shape {:?} disagrees with ({n}, {m}, {})",
                jet.shape(),
                self.latent_dim
            ));
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for EuclideanPatchEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator::second_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = crate::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        let d = self.latent_dim;
        let mut hess = Array4::<f64>::zeros((n, m, d, d));
        for (col, alpha) in exponents.iter().enumerate() {
            for a in 0..d {
                if alpha[a] == 0 {
                    continue;
                }
                for c in 0..d {
                    if a != c && alpha[c] == 0 {
                        continue;
                    }
                    let lead = if a == c {
                        (alpha[a] as f64) * (alpha[a].saturating_sub(1) as f64)
                    } else {
                        (alpha[a] as f64) * (alpha[c] as f64)
                    };
                    if lead == 0.0 {
                        continue;
                    }
                    for row in 0..n {
                        let mut value = lead;
                        for axis in 0..d {
                            let mut exp = alpha[axis];
                            if axis == a {
                                exp = exp.saturating_sub(1);
                            }
                            if axis == c {
                                exp = exp.saturating_sub(1);
                            }
                            if exp != 0 {
                                value *= coords[[row, axis]].powi(exp as i32);
                            }
                        }
                        hess[[row, col, a, c]] = value;
                    }
                }
            }
        }
        Ok(hess)
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
    /// Same evaluator upcast to `dyn SaeBasisSecondJet` when the
    /// implementation provides a closed-form Hessian. `None` for
    /// evaluators that only implement the base [`SaeBasisEvaluator`]
    /// trait. Installed via [`Self::with_basis_second_jet`]; the base
    /// [`Self::with_basis_evaluator`] populates only the supertrait
    /// slot. Used by [`refresh_isometry_caches_from_atom`] to install
    /// the `H` cache on isometry penalties when the second jet is
    /// analytically available.
    pub basis_second_jet: Option<Arc<dyn SaeBasisSecondJet>>,
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
            basis_second_jet: None,
        })
    }

    pub fn with_basis_evaluator(mut self, evaluator: Arc<dyn SaeBasisEvaluator>) -> Self {
        self.basis_evaluator = Some(evaluator);
        self.basis_second_jet = None;
        self
    }

    /// Install an evaluator that additionally exposes a closed-form
    /// second jet. Populates both the base [`SaeBasisEvaluator`] slot
    /// (used by [`Self::refresh_basis`] and the standard evaluate path)
    /// and the [`SaeBasisSecondJet`] slot (consumed by
    /// [`refresh_isometry_caches_from_atom`] for the `H` cache).
    pub fn with_basis_second_jet(mut self, evaluator: Arc<dyn SaeBasisSecondJet>) -> Self {
        let base: Arc<dyn SaeBasisEvaluator> = evaluator.clone();
        self.basis_evaluator = Some(base);
        self.basis_second_jet = Some(evaluator);
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

    /// Flatten ρ into the contiguous outer-coordinate vector the generic
    /// `OuterObjective` engine optimises over.
    ///
    /// Layout: `[log_lambda_sparse, log_lambda_smooth, <ARD>]`, where the
    /// ARD tail concatenates each atom `k`'s per-axis `log_ard[k][j]` in
    /// atom order, axis `j` in `0..d_k`. [`Self::from_flat`] is the exact
    /// inverse and reads the per-atom dims from `self`.
    pub fn to_flat(&self) -> Array1<f64> {
        let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
        let mut out = Array1::<f64>::zeros(2 + ard_len);
        out[0] = self.log_lambda_sparse;
        out[1] = self.log_lambda_smooth;
        let mut cursor = 2usize;
        for axis in &self.log_ard {
            for &v in axis.iter() {
                out[cursor] = v;
                cursor += 1;
            }
        }
        out
    }

    /// Rebuild a ρ with this ρ's per-atom ARD dimensions from a flat
    /// outer-coordinate vector produced by [`Self::to_flat`].
    ///
    /// The per-atom dims are taken from `&self` (the ARD layout is a fixed
    /// property of the term shape; the engine only moves the values). The
    /// flat vector must have length `2 + Σ_k d_k`.
    pub fn from_flat(&self, flat: ArrayView1<'_, f64>) -> SaeManifoldRho {
        let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
        assert_eq!(
            flat.len(),
            2 + ard_len,
            "SaeManifoldRho::from_flat: flat length {} != 2 + Σ d_k = {}",
            flat.len(),
            2 + ard_len
        );
        let mut log_ard = Vec::with_capacity(self.log_ard.len());
        let mut cursor = 2usize;
        for axis in &self.log_ard {
            let d = axis.len();
            let mut block = Array1::<f64>::zeros(d);
            for (j, slot) in block.iter_mut().enumerate() {
                *slot = flat[cursor + j];
            }
            cursor += d;
            log_ard.push(block);
        }
        SaeManifoldRho {
            log_lambda_sparse: flat[0],
            log_lambda_smooth: flat[1],
            log_ard,
        }
    }
}

/// Kronecker-factored per-row beta Jacobian primitive for SAE-manifold.
///
/// The per-row beta Jacobian has exact Kronecker form
///
/// ```text
/// J_{β,i} = φ_i^T ⊗ I_p
/// ```
///
/// where `φ_i ∈ ℝ^{m_i}` (active per-row atom·basis scalar weights, the
/// `a_k * phi` products in the assembly loop) and `p` is the decoder output
/// dimension.  The four trait methods implement the four operations that the
/// Arrow-Schur solver needs without ever forming the dense `(q × K·p)` block:
///
/// * `apply_jbeta`:   `u = J_β x`   (gather along active support)
/// * `scatter_jbeta_t`: `y += J_βᵀ u`  (scatter)
/// * `apply_l`:       `w = L u`     (q × p Jacobian apply)
/// * `apply_l_t`:     `u += Lᵀ v`   (q × p Jacobian transpose-accumulate)
///
/// The inner Schur row contribution
///
/// ```text
/// S_i = J_{β,i}^T (I - L_i^T A_i^{-1} L_i) J_{β,i}
/// ```
///
/// is applied in `O(m_i p + q p + q²)` per row per PCG iteration using
/// the five-step sequence:
///
/// ```text
/// u_p        = Σ_s φ_i[s] * x_β[s, :]    // gather (apply_jbeta)
/// w_q        = L_i u_p                    // q × p apply (apply_l)
/// v_q        = A_i^{-1} w_q               // existing per-row factor
/// u_p       -= L_i^T v_q                  // q × p apply-t (apply_l_t)
/// y_β[s, :] += φ_i[s] * u_p              // scatter (scatter_jbeta_t)
/// ```
pub trait SaeKroneckerRow {
    /// `u_out[j] = Σ_s φ_i[s] * x_beta[s * p + j]` for `j in 0..p`.
    ///
    /// Gather step: projects the full `K·p` beta vector down to the `p`-dimensional
    /// decoded output space using the active per-row support weights.
    fn apply_jbeta(&self, row: usize, x_beta: &[f64], u_out: &mut [f64]);

    /// `y_beta[s * p + j] += φ_i[s] * u[j]` for each active `(s, j)`.
    ///
    /// Scatter step: distributes the `p`-dimensional residual back into the
    /// full `K·p` beta gradient using the active per-row support weights.
    fn scatter_jbeta_t(&self, row: usize, u: &[f64], y_beta: &mut [f64]);

    /// `w_out[c] = Σ_j L[c, j] * u[j]` — apply the `q × p` local Jacobian.
    fn apply_l(&self, row: usize, u: &[f64], w_out: &mut [f64]);

    /// `u_out[j] += Σ_c L[c, j] * v[c]` — accumulate `Lᵀ v` into `u_out`.
    fn apply_l_t(&self, row: usize, v: &[f64], u_out: &mut [f64]);
}

/// Per-row Kronecker data for the SAE-manifold beta Jacobian.
///
/// Each row `i` stores:
/// * `a_phi_row`: sparse support — `(beta_base_idx, scalar_weight)` pairs,
///   one entry per active `(atom, basis_col)` combination.
/// * `local_jac_row`: the `(q × p)` assignment + coordinate Jacobian `L_i`
///   (same matrix written into `block.htt` via `local_jac` in the assembler).
///
/// Together these implement `J_β = φᵀ ⊗ I_p` without materializing the dense
/// `(q × K·p)` block.  Storage is `O(m_i · q · p)` per row rather than
/// `O(q · K · p)`.
#[derive(Debug, Clone)]
pub struct SaeKroneckerRows {
    /// Decoder output dimension `p`.
    p: usize,
    /// Per-row sparse support: `a_phi[i]` is a `Vec<(beta_base_idx, weight)>`.
    a_phi: Vec<Vec<(usize, f64)>>,
    /// Per-row local Jacobian `L_i`, shape `(q_i × p)` flattened row-major.
    ///
    /// Element `(c, j)` is at `local_jac[i][c * p + j]`.
    /// For heterogeneous (active-set) systems, each row may have a different
    /// `q_i = local_jac[i].len() / p`.
    local_jac: Vec<Vec<f64>>,
}

impl SaeKroneckerRows {
    /// Build from per-row data collected during `assemble_arrow_schur`. The
    /// row count is implicit in `a_phi.len()` and `local_jac.len()`; the
    /// constructor asserts they agree so callers cannot pass mismatched rows.
    pub fn new(p: usize, a_phi: Vec<Vec<(usize, f64)>>, local_jac: Vec<Vec<f64>>) -> Self {
        assert_eq!(
            a_phi.len(),
            local_jac.len(),
            "SaeKroneckerRows: a_phi rows ({}) != local_jac rows ({})",
            a_phi.len(),
            local_jac.len(),
        );
        Self {
            p,
            a_phi,
            local_jac,
        }
    }
}

impl SaeKroneckerRow for SaeKroneckerRows {
    fn apply_jbeta(&self, row: usize, x_beta: &[f64], u_out: &mut [f64]) {
        for val in u_out.iter_mut() {
            *val = 0.0;
        }
        for &(beta_base, phi) in &self.a_phi[row] {
            if phi == 0.0 {
                continue;
            }
            for j in 0..self.p {
                u_out[j] += phi * x_beta[beta_base + j];
            }
        }
    }

    fn scatter_jbeta_t(&self, row: usize, u: &[f64], y_beta: &mut [f64]) {
        for &(beta_base, phi) in &self.a_phi[row] {
            if phi == 0.0 {
                continue;
            }
            for j in 0..self.p {
                y_beta[beta_base + j] += phi * u[j];
            }
        }
    }

    fn apply_l(&self, row: usize, u: &[f64], w_out: &mut [f64]) {
        let jac = &self.local_jac[row];
        // Per-row q_i = jac.len() / p (supports heterogeneous active-set layouts).
        let q_i = jac.len() / self.p;
        for c in 0..q_i {
            let mut acc = 0.0_f64;
            for j in 0..self.p {
                acc += jac[c * self.p + j] * u[j];
            }
            w_out[c] = acc;
        }
    }

    fn apply_l_t(&self, row: usize, v: &[f64], u_out: &mut [f64]) {
        let jac = &self.local_jac[row];
        let q_i = jac.len() / self.p;
        for c in 0..q_i {
            let vc = v[c];
            if vc == 0.0 {
                continue;
            }
            for j in 0..self.p {
                u_out[j] += jac[c * self.p + j] * vc;
            }
        }
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

/// Per-row active-set layout for sparse SAE assignment (any mode).
///
/// When the assignment is sparse — structurally (JumpReLU gate) or
/// effectively (softmax / IBP-MAP at large `K`, where the assignment mass
/// concentrates on a small support) — only a subset of `K` atoms are active
/// per observation.  The Arrow-Schur row block for observation `i` has dim
/// `q_active_i = |active_atoms_i| + Σ_{k ∈ active_i} d_k` rather than
/// `q = K + Σ_k d_k`.  This struct records which atoms are active per row
/// and maps compressed block positions back to full-q positions so that
/// `apply_newton_step` can unpack the compact `delta_t` from the solve.
///
/// For JumpReLU the active set is exactly the gated support
/// (`a_{n,k} ≠ 0`), so the compact solve is identity to the dense solve.
/// For softmax / IBP-MAP the active set is the union of a top-`k_active_cap`
/// truncation and a magnitude cutoff on `a_{n,k}`; this is only enabled when
/// `K` is large enough that the dense `(m_total · p)²` data Gram would not
/// fit the host / device working-set budget, and the dropped atoms carry
/// `O(a_{n,k}²)` curvature that is negligible by construction of the cutoff.
#[derive(Debug, Clone)]
pub struct SaeRowLayout {
    /// `active_atoms[row]` — sorted indices of active atoms for that row.
    pub active_atoms: Vec<Vec<usize>>,
    /// For row `i`, active atom `active_atoms[i][j]` has its coord block
    /// starting at compressed position `coord_starts[i][j]`.
    pub coord_starts: Vec<Vec<usize>>,
    /// Full-q coordinate offset for atom `k` (length `k_atoms`).
    pub coord_offsets_full: Vec<usize>,
    /// Per-atom coordinate dimensions, indexed by atom index.
    pub coord_dims: Vec<usize>,
}

impl SaeRowLayout {
    /// JumpReLU structural active set: atoms with `logit > threshold`.
    fn from_jumprelu(
        n: usize,
        k_atoms: usize,
        threshold: f64,
        logits: &Array2<f64>,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let mut per_row = Vec::with_capacity(n);
        for row in 0..n {
            let row_logits = logits.row(row);
            let active: Vec<usize> = (0..k_atoms)
                .filter(|&k| row_logits[k] > threshold)
                .collect();
            per_row.push(active);
        }
        Self::from_active_atoms(per_row, coord_dims, coord_offsets_full)
    }

    /// Mode-agnostic effective active set for dense-weight modes (softmax /
    /// IBP-MAP) at large `K`: keep, per row, the top-`k_active_cap` atoms by
    /// `|a_{n,k}|` whose magnitude also exceeds `cutoff`.
    ///
    /// `assignments[row]` is the dense length-`K` assignment vector `a_{n,·}`.
    /// The active set is always non-empty (the single largest-magnitude atom is
    /// retained even if below `cutoff`) so every row keeps a valid block.
    fn from_dense_weights(
        assignments: &[Array1<f64>],
        k_active_cap: usize,
        cutoff: f64,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let cap = k_active_cap.max(1);
        let mut per_row = Vec::with_capacity(assignments.len());
        for a in assignments {
            let k = a.len();
            // Rank atoms by descending |a_k|; keep those above cutoff, capped.
            let mut idx: Vec<usize> = (0..k).collect();
            idx.sort_by(|&i, &j| {
                a[j].abs()
                    .partial_cmp(&a[i].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut active: Vec<usize> = idx
                .iter()
                .copied()
                .take(cap)
                .filter(|&k_idx| a[k_idx].abs() > cutoff)
                .collect();
            if active.is_empty() {
                // Retain the single largest-magnitude atom so the row block is
                // never empty (a degenerate empty block would zero the row).
                if let Some(&top) = idx.first() {
                    active.push(top);
                }
            }
            active.sort_unstable();
            per_row.push(active);
        }
        Self::from_active_atoms(per_row, coord_dims, coord_offsets_full)
    }

    /// Build from explicit per-row active-atom index lists.
    fn from_active_atoms(
        active_atoms: Vec<Vec<usize>>,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let mut coord_starts_all = Vec::with_capacity(active_atoms.len());
        for active in &active_atoms {
            let mut starts = Vec::with_capacity(active.len());
            let mut cursor = active.len();
            for &k in active {
                starts.push(cursor);
                cursor += coord_dims[k];
            }
            coord_starts_all.push(starts);
        }
        Self {
            active_atoms,
            coord_starts: coord_starts_all,
            coord_offsets_full,
            coord_dims,
        }
    }

    /// Per-row compressed dim.
    pub fn row_q_active(&self, row: usize) -> usize {
        let active = &self.active_atoms[row];
        let coord_sum: usize = active.iter().map(|&k| self.coord_dims[k]).sum();
        active.len() + coord_sum
    }

    /// Expand a compact `delta_t` row slice back into full-q, zeros for inactive.
    pub fn expand_row(&self, row: usize, delta_t_row: &[f64], out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let active = &self.active_atoms[row];
        let starts = &self.coord_starts[row];
        for (j, &k) in active.iter().enumerate() {
            out[k] = delta_t_row[j];
            let d = self.coord_dims[k];
            let full_off = self.coord_offsets_full[k];
            for axis in 0..d {
                out[full_off + axis] = delta_t_row[starts[j] + axis];
            }
        }
    }
}

/// Full SAE-manifold term.
#[derive(Debug, Clone)]
pub struct SaeManifoldTerm {
    pub atoms: Vec<SaeManifoldAtom>,
    pub assignment: SaeAssignment,
    temperature_schedule: Option<GumbelTemperatureSchedule>,
    /// Active-set row layout from the most recent `assemble_arrow_schur` call.
    /// `None` for dense modes (Softmax / IBPMap) or when not yet assembled.
    last_row_layout: Option<SaeRowLayout>,
}

/// Snapshot of exactly the mutable term state that an `apply_newton_step` +
/// `loss` line-search trial perturbs: per-atom decoder coefficients and the
/// `refresh_basis`-rebuilt basis evaluations (`basis_values`, `basis_jacobian`),
/// plus the assignment logits and latent coordinates.
///
/// Static fields (atom names, basis kinds, smoothness penalties, basis-evaluator
/// `Arc`s, assignment mode, temperature schedule, `last_row_layout`) are *not*
/// snapshotted: they are invariant across an inner Newton line search, so the
/// previous `self.clone()` per halving re-copied them needlessly. Cloning only
/// the mutated arrays keeps the `O(N·M·d)` `basis_jacobian` copy off the
/// per-halving hot path (one snapshot before the search, one restore per
/// rejected trial) instead of firing it on every Armijo backtrack.
#[derive(Debug)]
struct SaeManifoldMutableState {
    /// Per-atom `(basis_values, basis_jacobian, decoder_coefficients)`.
    atoms: Vec<(Array2<f64>, Array3<f64>, Array2<f64>)>,
    logits: Array2<f64>,
    coords: Vec<LatentCoordValues>,
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
            last_row_layout: None,
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

    /// Construction-time validation: every Psi-tier analytic penalty in the
    /// registry must be dispatchable into the SAE arrow-Schur row layout.
    ///
    /// Two invariants are enforced upfront so the dispatch loop in
    /// `add_sae_analytic_penalty_contributions` is total (no runtime
    /// "unsupported penalty" fallthrough, no per-call K-gating):
    ///
    /// 1. Every Psi-tier penalty is either a logit-target penalty
    ///    (`IBPAssignment`, `SoftmaxAssignmentSparsity`) or in
    ///    [`sae_penalty_is_row_block_supported`]. Penalty kinds with
    ///    cross-row structure (`TotalVariation`, `Monotonicity`,
    ///    `NuclearNorm`, `BlockSparsity`, `IvaeRidgeMeanGauge`,
    ///    `Orthogonality`, `NestedPrefix`, `SheafConsistency`) cannot be
    ///    expressed in the SAE row-block layout and are refused here.
    ///
    /// 2. If any Psi-tier row-block penalty is present, every atom shares
    ///    the same coord latent dim. The current registry model carries one
    ///    `latent_dim` per descriptor (the "t" latent block declares one
    ///    `d` value); per-atom dispatch with heterogeneous `d_k` would
    ///    require per-atom registry entries or per-kind in-place
    ///    reshaping. Mixed-d row-block fits are rejected with an actionable
    ///    error pointing at the configuration mismatch.
    ///
    /// The K=1 case trivially satisfies (2). Beta-tier and rho-tier
    /// penalties are not constrained here.
    fn validate_analytic_penalty_registry(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<(), String> {
        let mut row_block_penalty_present = false;
        for penalty in &registry.penalties {
            if penalty.tier() != PenaltyTier::Psi {
                continue;
            }
            let is_logit = matches!(
                penalty,
                AnalyticPenaltyKind::IBPAssignment(_)
                    | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            );
            if is_logit {
                continue;
            }
            if !sae_penalty_is_row_block_supported(penalty) {
                return Err(format!(
                    "SAE-manifold term refuses analytic penalty {:?}: this kind \
                     has cross-row structure and cannot be expressed in the \
                     arrow-Schur row layout. Use only row-block-supported \
                     coord penalties (ARD, BlockOrthogonality, \
                     Sparsity/TopK/JumpReLU, RowPrecisionPrior, \
                     ParametricRowPrecisionPrior, ScadMcp, Isometry) on the \
                     coord latent block, or move the penalty to a non-SAE \
                     term",
                    penalty.name()
                ));
            }
            row_block_penalty_present = true;
        }
        if row_block_penalty_present {
            let mut dims = self.assignment.coords.iter().map(|c| c.latent_dim());
            if let Some(first) = dims.next() {
                if let Some(mismatch) = dims.find(|d| *d != first) {
                    return Err(format!(
                        "SAE-manifold term refuses row-block analytic penalty: \
                         atoms have heterogeneous coord latent dims (saw {first} \
                         and {mismatch}). Row-block penalties (ARD, \
                         BlockOrthogonality, ...) target the unified \"t\" \
                         latent block whose declared `d` matches one shape; \
                         per-atom dispatch with mixed `d_k` would silently \
                         truncate or expand axes. Configure all atoms with the \
                         same `atom_dim`, or split the row-block penalty into \
                         per-atom descriptors keyed to per-atom latent blocks"
                    ));
                }
            }
        }
        Ok(())
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

    /// Per-atom β column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// Returns one `Range<usize>` per atom, covering that atom's decoder
    /// coefficients in the flat β vector:
    ///   `[beta_offsets[k] .. beta_offsets[k] + basis_size[k] * p_out]`.
    ///
    /// Pass to [`ArrowSchurSystem::set_block_offsets`] so that
    /// [`crate::solver::arrow_schur::JacobiPreconditioner`] builds one dense
    /// Schur sub-block per atom instead of scalar-diagonal inversion.
    pub fn beta_block_offsets(&self) -> Arc<[std::ops::Range<usize>]> {
        let p = self.output_dim();
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            let width = atom.basis_size() * p;
            ranges.push(cursor..cursor + width);
            cursor += width;
        }
        Arc::from(ranges.into_boxed_slice())
    }

    /// Decide whether the sparse per-row active-set layout is engaged for the
    /// dense-weight assignment modes (softmax / IBP-MAP), and if so derive the
    /// per-row active-atom cap and magnitude cutoff.
    ///
    /// The decision is auto-derived from the problem size and the
    /// device/host working-set budget — never a CLI flag or kwarg. JumpReLU is
    /// not handled here (it always uses its structural gate via
    /// [`SaeRowLayout::from_jumprelu`]). The dense Gauss-Newton data Gram `G`
    /// is `(m_total × m_total)` f64; if its dense form fits the budget we keep
    /// the exact full-support solve (every atom active per row), so small-`K`
    /// problems are bit-for-bit unchanged. Above that, we cap each row to the
    /// `k_active` atoms that make the *sparse* Gram fit the same budget, with a
    /// relative magnitude cutoff that drops assignment mass contributing
    /// negligible `O(a²)` curvature.
    ///
    /// Returns `Some((k_active_cap, cutoff))` to engage sparsity, or `None` to
    /// keep the dense full-support layout.
    fn sparse_active_plan(&self) -> Option<(usize, f64)> {
        const BYTES_PER_F64: usize = 8;
        // Host in-core ceiling for the dense Gram on a CPU-only build, mirroring
        // the streaming dispatcher's host budget (`sae_streaming_plan`).
        const HOST_GRAM_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
        // Relative magnitude cutoff: assignment mass below this fraction of the
        // row's peak `|a_k|` enters the Gram only as `O(a²)` curvature and is
        // dropped. Chosen so dropped terms are ~1e-6 of the peak self-coupling.
        const RELATIVE_CUTOFF: f64 = 1.0e-3;

        let k_atoms = self.k_atoms();
        if k_atoms <= 1 {
            return None;
        }
        // The per-row tangent projection used for non-Euclidean atom latents
        // requires the uniform-`q` dense layout; the compact active-set path
        // skips it. So the effective (truncating) sparse plan only engages
        // when the ext-coord manifold is Euclidean. JumpReLU's structural gate
        // is exact and handled separately, so this restriction does not affect
        // it. At huge `K` on a curved manifold the streaming driver still
        // bounds memory; only the per-atom truncation is withheld.
        if !self.ext_coord_manifold().is_euclidean() {
            return None;
        }
        let p = self.output_dim();
        let m_total: usize = self.atoms.iter().map(|a| a.basis_size()).sum();
        // Dense data Gram footprint: (m_total · m_total) f64.
        let dense_gram_bytes = m_total
            .saturating_mul(m_total)
            .saturating_mul(BYTES_PER_F64);

        let budget = match crate::gpu::runtime::GpuRuntime::global() {
            // Allow up to one quarter of the device budget for the dense Gram,
            // matching the streaming dispatcher's in-core fraction.
            Some(rt) => rt.memory_budget_bytes / 4,
            None => HOST_GRAM_BYTES,
        };
        if dense_gram_bytes <= budget {
            return None;
        }

        // Sparse Gram footprint scales with the per-row active basis count
        // `k_active · m_atom`. Solve for the largest `k_active` whose sparse
        // Gram `(k_active · m_atom)²` still fits the budget.
        let m_atom = (m_total as f64 / k_atoms as f64).max(1.0);
        let max_active_basis = ((budget as f64 / BYTES_PER_F64 as f64).sqrt() / m_atom).floor();
        let k_active_cap = (max_active_basis as usize).clamp(1, k_atoms);
        // p does not enter the Gram dimension (it is carried by the `⊗ I_p`
        // structure), but a degenerate `p == 0` term has no decoder columns.
        if p == 0 {
            return None;
        }
        Some((k_active_cap, RELATIVE_CUTOFF))
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
        self.loss_scaled(target, rho, 1.0)
    }

    /// Penalized objective with a `penalty_scale` applied to the β-tier
    /// (decoder smoothness) penalty, mirroring
    /// [`Self::assemble_arrow_schur_scaled`]. The streaming line search sums
    /// per-chunk `loss_scaled(..., n_chunk / N)` so that the global smoothness
    /// penalty is counted exactly once across a pass while the per-row data,
    /// assignment-prior, and ARD terms sum naturally. `penalty_scale == 1.0`
    /// recovers the full-batch objective.
    pub fn loss_scaled(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        penalty_scale: f64,
    ) -> Result<SaeManifoldLoss, String> {
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::loss_scaled: penalty_scale must be finite and positive; got {penalty_scale}"
            ));
        }
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
        let smoothness = penalty_scale * self.decoder_smoothness_value(rho.lambda_smooth());
        let ard = self.ard_value(rho)?;
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
        })
    }

    fn decoder_smoothness_value(&self, lambda_smooth: f64) -> f64 {
        // Smoothness penalty value is `0.5·λ·Σ_oc B[:,oc]ᵀ S B[:,oc]`. Form the
        // `S·B` matrix product once per atom (O(M²·p)) and reduce against `B`
        // with a single O(M·p) Hadamard sum, instead of the previous
        // four-factor multiply-accumulate inside an `O(M²·p)` triple loop.
        // The quadratic form only sees the symmetric part of `S`, so reusing
        // the raw (un-symmetrised) `smooth_penalty` here is numerically
        // identical to the symmetrised assembly form.
        let mut acc = 0.0;
        for atom in &self.atoms {
            let sb = atom.smooth_penalty.dot(&atom.decoder_coefficients);
            acc += 0.5 * lambda_smooth * (&atom.decoder_coefficients * &sb).sum();
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
    ///
    /// Full-batch entry point: a single chunk covering all rows, with the
    /// β-tier penalties (decoder smoothness, ARD, analytic β penalties) carrying
    /// their full strength. The streaming driver calls
    /// [`Self::assemble_arrow_schur_scaled`] directly with a `penalty_scale`
    /// equal to the minibatch fraction `n_chunk / N`, so that the sum of the
    /// per-chunk β-tier contributions over a full pass reconstructs exactly the
    /// single global β penalty (the smoothness/ARD/β terms are functions of `B`
    /// and the global coordinates, not of the chunk's rows).
    pub fn assemble_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_scaled(target, rho, analytic_penalties, 1.0)
    }

    /// Assemble the row-local Arrow-Schur system with a `penalty_scale` applied
    /// to the β-tier (decoder smoothness, ARD prior, analytic β penalties).
    ///
    /// `penalty_scale == 1.0` recovers the full-batch assembly. The streaming
    /// driver passes the minibatch fraction `n_chunk / N` so that the β-tier
    /// reduced-Schur and gradient contributions of the chunks sum to exactly one
    /// global copy across a full pass (data-fit, assignment-prior, and per-row
    /// coord/logit analytic terms are *not* scaled — they are genuine per-row
    /// sums).
    pub fn assemble_arrow_schur_scaled(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
    ) -> Result<ArrowSchurSystem, String> {
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur_scaled: penalty_scale must be finite and positive; got {penalty_scale}"
            ));
        }
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
        // β-tier decoder smoothness is a global (B-only) penalty; under a
        // minibatch pass it is scaled by the chunk fraction so the per-chunk
        // contributions sum to one global copy.
        let lambda_smooth = rho.lambda_smooth() * penalty_scale;
        let (assignment_grad, assignment_hdiag) =
            assignment_prior_grad_hdiag(&self.assignment, rho)?;

        // Decoder smoothness penalty: build one KroneckerPenaltyOp per atom
        // (structure = λ·S_k ⊗ I_p, offset = beta_offsets[k]) instead of
        // materialising the dense K×K block.  The gradient is a dense K-vector
        // accumulated into `smooth_grad_gb` and written into sys.gb after sys
        // is constructed (#296).
        let mut smooth_ops: Vec<Arc<dyn BetaPenaltyOp>> = Vec::with_capacity(self.atoms.len());
        let mut smooth_grad_gb = vec![0.0_f64; beta_dim];
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let off = beta_offsets[atom_idx];
            // Symmetrise and scale the smoothness penalty matrix.
            let mut scaled_s = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    let s_ij = 0.5 * (atom.smooth_penalty[[i, j]] + atom.smooth_penalty[[j, i]]);
                    scaled_s[[i, j]] = lambda_smooth * s_ij;
                }
            }
            // Gradient: g[beta_i] += (λ S_k B_k)[i, out_col]. Concentrate the
            // scattered triple loop into a single (m×m)·(m×p) GEMM (mirrors the
            // pattern in `decoder_smoothness_value`) for cache locality.
            let sb = scaled_s.dot(&atom.decoder_coefficients);
            for out_col in 0..p {
                for i in 0..m {
                    let beta_i = off + i * p + out_col;
                    smooth_grad_gb[beta_i] += sb[[i, out_col]];
                }
            }
            // KroneckerPenaltyOp: factor_a = λ·S_k (m×m), factor_b = I_p (p×p).
            let identity_p = Array2::<f64>::eye(p);
            smooth_ops.push(Arc::new(KroneckerPenaltyOp {
                factor_a: scaled_s,
                factor_b: identity_p,
                global_offset: off,
                k: beta_dim,
            }));
        }

        // Per-row active-set layout. Engaged for two regimes:
        //   * JumpReLU — structural gate (`logit > threshold`); the compact
        //     solve is bit-identical to the dense one (gated atoms carry zero
        //     assignment mass).
        //   * Softmax / IBP-MAP at large `K` — the dense `(m_total · p)²` data
        //     Gram is infeasible, so each row is truncated to its
        //     top-`k_active` atoms above a relative magnitude cutoff
        //     ([`Self::sparse_active_plan`]). Small-`K` problems return `None`
        //     and keep the exact full-support layout.
        // The compact row block is sized `q_active = |active| + Σ_{k∈active}
        // d_k` instead of the full `q`.
        let coord_dims: Vec<usize> = self
            .assignment
            .coords
            .iter()
            .map(|c| c.latent_dim())
            .collect();
        let row_layout: Option<SaeRowLayout> = match self.assignment.mode {
            AssignmentMode::JumpReLU { threshold, .. } => Some(SaeRowLayout::from_jumprelu(
                n,
                k_atoms,
                threshold,
                &self.assignment.logits,
                coord_dims.clone(),
                self.assignment.coord_offsets(),
            )),
            AssignmentMode::Softmax { .. } | AssignmentMode::IBPMap { .. } => {
                match self.sparse_active_plan() {
                    Some((k_active_cap, relative_cutoff)) => {
                        // Build per-row dense assignments once to derive the
                        // active set; the row loop re-derives `assignments`
                        // (cheap softmax) and reuses these active sets.
                        let mut assignments_all = Vec::with_capacity(n);
                        for row in 0..n {
                            assignments_all.push(self.assignment.try_assignments_row(row)?);
                        }
                        // Absolute cutoff = relative_cutoff · max row peak, so a
                        // single threshold drops sub-1e-3 mass across all rows.
                        let peak = assignments_all
                            .iter()
                            .flat_map(|a| a.iter())
                            .fold(0.0_f64, |m, &v| m.max(v.abs()));
                        let cutoff = relative_cutoff * peak;
                        Some(SaeRowLayout::from_dense_weights(
                            &assignments_all,
                            k_active_cap,
                            cutoff,
                            coord_dims.clone(),
                            self.assignment.coord_offsets(),
                        ))
                    }
                    None => None,
                }
            }
        };
        // Build the Arrow-Schur system: heterogeneous row dims when a compact
        // layout is active, uniform `q` otherwise.
        let mut sys = if let Some(ref layout) = row_layout {
            let per_row_dims: Vec<usize> = (0..n).map(|row| layout.row_q_active(row)).collect();
            ArrowSchurSystem::new_with_per_row_dims(per_row_dims, beta_dim)
        } else {
            ArrowSchurSystem::new(n, q, beta_dim)
        };
        // Apply accumulated smoothness-penalty gradients into sys.gb.
        for (i, g) in smooth_grad_gb.iter().enumerate() {
            sys.gb[i] += g;
        }
        // Hoist per-row temporaries outside the row loop: these allocations
        // previously fired N times per assembly, and each `decoded_row` /
        // `decoded_derivative_row` call inside the loop allocated its own
        // `Array1<f64>` of length p.
        let mut decoded = Array2::<f64>::zeros((k_atoms, p));
        let mut dg_buf = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        // Data-fit Gauss-Newton β-Hessian is block-diagonal across the `p`
        // output channels and identical in each: with the flat β layout
        // `β[μ·p + oc] = B[μ, oc]` (μ enumerating (atom, basis_col)) the GN
        // outer product `Jβᵀ Jβ` couples only equal `oc`, with the same
        // `(M_total × M_total)` block `G[μ, μ'] = Σ_rows (a_k φ_k[m])(a_{k'} φ_{k'}[m'])`
        // for every channel. So `H_data = G ⊗ I_p`. The `μ` index of an `a_phi`
        // entry whose global β base is `beta_base` is `beta_base / p` (every
        // `beta_offset` and the `basis_col·p` stride are multiples of `p`).
        //
        // `G` is only non-zero on `(atom_i, atom_j)` pairs that co-occur in
        // some row's active set, so we accumulate it as a sparse map of dense
        // per-atom-pair `(m_i × m_j)` blocks keyed by `(atom_i, atom_j)` rather
        // than as a dense `(m_total × m_total)` matrix. At `K = 100K` with
        // per-row active sets of size `k_active ≪ K`, only `O(N · k_active²)`
        // pairs are ever touched, so the data Gram (and every matvec /
        // diagonal pass over it via `SparseBlockKroneckerPenaltyOp`) tracks the
        // active atoms instead of `K²`. In the dense full-support layout the
        // map degenerates to every co-occurring pair, reproducing the dense
        // Gram exactly. A `BTreeMap` key order keeps the installed op's
        // fingerprint deterministic. The `μ`-space offset of atom `k` is
        // `beta_offsets[k] / p`.
        let m_total: usize = self.atoms.iter().map(|a| a.basis_size()).sum();
        let mu_offsets: Vec<usize> = beta_offsets.iter().map(|&off| off / p).collect();
        let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
            std::collections::BTreeMap::new();
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
        // Kronecker htbeta storage: per-row sparse support and local Jacobian.
        // These replace the O(q · K · p) dense htbeta write with O(m_i · q · p)
        // storage; the Arrow-Schur solver accesses them via htbeta_matvec.
        let mut kron_a_phi: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut kron_jac: Vec<Vec<f64>> = Vec::with_capacity(n);
        // Dense full-support index `[0, k_atoms)`, used by the row loop when no
        // compact layout is engaged so the active-atom iteration is uniform.
        let all_atoms_index: Vec<usize> = (0..k_atoms).collect();
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row(row)?;
            // Reconstruction uses the row's active support: for the dense
            // full-support layout this is all atoms (exact); for a compact
            // layout the dropped atoms carry negligible `O(a)` reconstruction
            // mass and zero curvature, so excluding them keeps `fitted`,
            // `error`, and the logit-JVP cross term `(decoded[k] − fitted)`
            // mutually consistent with the curvature actually assembled.
            fitted.fill(0.0);
            let row_active_owned: Option<&[usize]> =
                row_layout.as_ref().map(|l| l.active_atoms[row].as_slice());
            match row_active_owned {
                Some(active) => {
                    for &atom_idx in active {
                        let a_k = assignments[atom_idx];
                        self.atoms[atom_idx].fill_decoded_row(row, &mut decoded_scratch);
                        for out_col in 0..p {
                            decoded[[atom_idx, out_col]] = decoded_scratch[out_col];
                            fitted[out_col] += a_k * decoded_scratch[out_col];
                        }
                    }
                }
                None => {
                    for atom_idx in 0..k_atoms {
                        let a_k = assignments[atom_idx];
                        self.atoms[atom_idx].fill_decoded_row(row, &mut decoded_scratch);
                        for out_col in 0..p {
                            decoded[[atom_idx, out_col]] = decoded_scratch[out_col];
                            fitted[out_col] += a_k * decoded_scratch[out_col];
                        }
                    }
                }
            }
            for out_col in 0..p {
                error[out_col] = fitted[out_col] - target[[row, out_col]];
            }

            // Determine whether this row uses the compact active-set layout.
            //   * JumpReLU: only gated atoms (logit > threshold) enter.
            //   * Softmax / IBP-MAP at large K: only the top-`k_active` atoms.
            //   * Otherwise (small K): the dense uniform-q layout.
            let (q_row, local_jac_row) = if let Some(ref layout) = row_layout {
                let active = &layout.active_atoms[row];
                let starts = &layout.coord_starts[row];
                let q_active = layout.row_q_active(row);
                let mut jac_compact = Array2::<f64>::zeros((q_active, p));
                // Logit JVP rows for active atoms only, using the per-mode
                // assignment sensitivity `da_k/dl_k` contracted into the
                // decoded / fitted-corrected output direction.
                let logits_row = self.assignment.logits.row(row);
                for (j, &k) in active.iter().enumerate() {
                    fill_active_atom_logit_jvp(
                        self.assignment.mode,
                        k,
                        logits_row[k],
                        assignments[k],
                        decoded.row(k),
                        fitted.view(),
                        ibp_prior_slice,
                        &mut jac_compact,
                        j,
                    );
                }
                // Coordinate JVP rows for active atoms only.
                for (j, &k) in active.iter().enumerate() {
                    let d = self.atoms[k].latent_dim;
                    let a_k = assignments[k];
                    let coord_start = starts[j];
                    for axis in 0..d {
                        self.atoms[k].fill_decoded_derivative_row(row, axis, &mut dg_buf);
                        for out_col in 0..p {
                            jac_compact[[coord_start + axis, out_col]] = a_k * dg_buf[out_col];
                        }
                    }
                }
                (q_active, jac_compact)
            } else {
                // Fresh per-row Jacobian, structurally identical to the
                // JumpReLU branch: every (q × p) element is unconditionally
                // overwritten below (logit JVP rows + coordinate rows), so the
                // `Array2::zeros` allocation needs no separate `fill(0.0)` and
                // the populated buffer is returned by move without a clone.
                let mut jac_row = Array2::<f64>::zeros((q, p));
                fill_assignment_logit_jvp_rows(
                    self.assignment.mode,
                    self.assignment.logits.row(row),
                    assignments.view(),
                    decoded.view(),
                    fitted.view(),
                    ibp_prior_slice,
                    &mut jac_row,
                );
                // Coordinate columns for all atoms.
                for atom_idx in 0..k_atoms {
                    let d = self.atoms[atom_idx].latent_dim;
                    let off = coord_offsets[atom_idx];
                    let a_k = assignments[atom_idx];
                    for axis in 0..d {
                        self.atoms[atom_idx].fill_decoded_derivative_row(row, axis, &mut dg_buf);
                        for out_col in 0..p {
                            jac_row[[off + axis, out_col]] = a_k * dg_buf[out_col];
                        }
                    }
                }
                (q, jac_row)
            };

            // Build the per-row Arrow-Schur block at the row's active dim.
            let mut block = ArrowRowBlock::new(q_row, beta_dim);
            for a in 0..q_row {
                let mut g = 0.0;
                for out_col in 0..p {
                    g += local_jac_row[[a, out_col]] * error[out_col];
                }
                block.gt[a] += g;
                for b in 0..q_row {
                    let mut h = 0.0;
                    for out_col in 0..p {
                        h += local_jac_row[[a, out_col]] * local_jac_row[[b, out_col]];
                    }
                    block.htt[[a, b]] += h;
                }
            }

            // Assignment prior in logit space.
            // For compact layout: position `j` = active_atoms index.
            // For dense layout: position `atom_idx` directly.
            let assignment_base = row * k_atoms;
            if let Some(ref layout) = row_layout {
                let active = &layout.active_atoms[row];
                for (j, &k) in active.iter().enumerate() {
                    block.gt[j] += assignment_grad[assignment_base + k];
                    block.htt[[j, j]] += assignment_hdiag[assignment_base + k];
                }
            } else {
                for atom_idx in 0..k_atoms {
                    block.gt[atom_idx] += assignment_grad[assignment_base + atom_idx];
                    block.htt[[atom_idx, atom_idx]] += assignment_hdiag[assignment_base + atom_idx];
                }
            }

            // ARD on each on-atom coordinate.
            // For compact layout: only active atoms; coord positions use compact starts.
            // For dense layout: all atoms; coord positions use coord_offsets.
            if let Some(ref layout) = row_layout {
                let active = &layout.active_atoms[row];
                let starts = &layout.coord_starts[row];
                for (j, &k) in active.iter().enumerate() {
                    let coord = &self.assignment.coords[k];
                    let d = coord.latent_dim();
                    if rho.log_ard[k].len() != d {
                        return Err(format!(
                            "ARD rho atom {k} has len {} but atom dim is {d}",
                            rho.log_ard[k].len()
                        ));
                    }
                    let row_t = coord.row(row);
                    for axis in 0..d {
                        // ARD on coords is a genuine per-row prior (each row
                        // contributes 0.5·α·t_row²), so it is NOT minibatch-
                        // scaled — the per-chunk row sums already reconstruct
                        // the full ‖t‖² across a pass.
                        let alpha = rho.log_ard[k][axis].exp();
                        block.gt[starts[j] + axis] += alpha * row_t[axis];
                        block.htt[[starts[j] + axis, starts[j] + axis]] += alpha;
                    }
                }
            } else {
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
            }

            // Beta gradient/Hessian — Kronecker form J_β = φᵀ ⊗ I_p.
            //
            // The per-row beta Jacobian is
            //   J_β[out_col, beta_idx] = a_k · phi_k[basis_col]   if out_col == out_col(beta_idx)
            //                            0                         otherwise
            // so the data-fit Gauss-Newton beta-Hessian factors as a rank-`p`
            // sum of outer products. We pre-compute the per-(atom, basis_col)
            // scalar `a_k · phi_k` once and reuse it across the `out_col`
            // and inner `(atom_j, basis_col2)` loops.
            //
            // The dense (q × K·p) htbeta block is NOT written here.  Instead,
            // `a_phi` and `local_jac_row` are captured per-row into `SaeKroneckerRows`
            // and installed as `sys.htbeta_matvec` after the row loop.  All
            // Arrow-Schur inner paths (schur_matvec, reduced_rhs_beta,
            // build_dense_schur_*, JacobiPreconditioner) route through
            // `sys_htbeta_apply_row` / `sys_htbeta_accumulate_transpose` which
            // already prefer `htbeta_matvec` over the dense slab.
            //
            // Only the row's active atoms contribute `a_phi` support and data
            // curvature: in a compact layout (JumpReLU gate or large-K
            // top-`k_active` truncation) the inactive atoms carry zero (gated)
            // or sub-cutoff assignment mass and are excluded — this is what
            // keeps both the htbeta support and the `G` accumulation
            // `O(k_active)` rather than `O(K)`. In the dense full-support
            // layout `row_active` spans all atoms.
            let row_active: &[usize] = match row_layout {
                Some(ref layout) => layout.active_atoms[row].as_slice(),
                None => &all_atoms_index,
            };
            let mut a_phi: Vec<(usize, f64)> = Vec::with_capacity(row_active.len() * 4);
            // Per-active-atom weighted basis row `a_k · φ_k[·]`, retained so the
            // data Gram blocks can be accumulated as clean per-atom-pair outer
            // products `(a_k φ_k) (a_{k'} φ_{k'})ᵀ`.
            let mut weighted_phi: Vec<(usize, Vec<f64>)> = Vec::with_capacity(row_active.len());
            for &atom_idx in row_active {
                let atom = &self.atoms[atom_idx];
                let atom_beta_off = beta_offsets[atom_idx];
                let m = atom.basis_size();
                let a_k = assignments[atom_idx];
                let mut wphi = Vec::with_capacity(m);
                for basis_col in 0..m {
                    let phi = atom.basis_values[[row, basis_col]];
                    let w = a_k * phi;
                    a_phi.push((atom_beta_off + basis_col * p, w));
                    wphi.push(w);
                }
                weighted_phi.push((atom_idx, wphi));
            }
            // β data-fit gradient `gᵦ += J_βᵀ error`.
            for &(beta_base_i, j_beta_i) in a_phi.iter() {
                if j_beta_i == 0.0 {
                    continue;
                }
                for out_col in 0..p {
                    sys.gb[beta_base_i + out_col] += j_beta_i * error[out_col];
                    // No htbeta write — the Kronecker matvec handles this.
                    // No dense hbb write — the sparse `G ⊗ I_p` op installed
                    // after the loop carries the data-fit GN β-Hessian.
                }
            }
            // Data-fit GN β-Hessian: accumulate the channel-independent block
            // `G[μ_i, μ_j] += (a_k φ_k)[μ_i] (a_{k'} φ_{k'})[μ_j]` into the
            // sparse per-atom-pair map (the `out_col` dimension is carried by
            // `I_p`). Only co-occurring `(atom_i, atom_j)` pairs are touched.
            for ai in 0..weighted_phi.len() {
                let (atom_i, ref wphi_i) = weighted_phi[ai];
                let m_i = wphi_i.len();
                for aj in 0..weighted_phi.len() {
                    let (atom_j, ref wphi_j) = weighted_phi[aj];
                    let m_j = wphi_j.len();
                    let blk = g_blocks
                        .entry((atom_i, atom_j))
                        .or_insert_with(|| Array2::<f64>::zeros((m_i, m_j)));
                    for li in 0..m_i {
                        let wi = wphi_i[li];
                        if wi == 0.0 {
                            continue;
                        }
                        for lj in 0..m_j {
                            blk[[li, lj]] += wi * wphi_j[lj];
                        }
                    }
                }
            }
            // Save per-row Kronecker data for htbeta_matvec construction.
            kron_a_phi.push(a_phi);
            // Flatten local_jac_row row-major into a plain Vec<f64> (q_row * p entries).
            let mut jac_flat = vec![0.0_f64; q_row * p];
            for c in 0..q_row {
                for j in 0..p {
                    jac_flat[c * p + j] = local_jac_row[[c, j]];
                }
            }
            kron_jac.push(jac_flat);
            sys.rows[row] = block;
        }
        // Apply Riemannian geometry to the per-row row blocks (htt, gt) and
        // also to the per-row Kronecker local Jacobians stored in kron_jac.
        // When the SAE ext-coord manifold is non-Euclidean (any atom latent
        // on sphere / circle / interval), the local Jacobian rows that map
        // into the t-block tangent space must be projected via the per-row
        // tangent projector P_i.  This mirrors what
        // `apply_riemannian_latent_geometry` does to `row.htbeta`, applied
        // here to the (q × p) kron_jac so the Kronecker htbeta_matvec uses
        // the Riemannian-projected form.
        // Apply Riemannian geometry only for the dense uniform-q layout. Any
        // compact active-set layout (JumpReLU gate or large-K softmax/IBP
        // truncation) has heterogeneous q_i; the Riemannian projector path
        // requires a uniform latent dimension. The sparse plan only engages on
        // Euclidean ext-coord manifolds (see `sparse_active_plan`), so skipping
        // the projector here is correct — there is nothing to project.
        if row_layout.is_none() {
            self.apply_sae_riemannian_geometry(&mut sys);
            let manifold = self.ext_coord_manifold();
            if !manifold.is_euclidean() {
                let ext = self.ext_coord_matrix();
                // Project the local Jacobian columns onto the tangent space at
                // each row's ext-coord point. Each column `j` of the row's
                // (q_row × p) Jacobian is an ambient-space vector of length
                // `q_row`; the manifold projector acts on one such column at a
                // time. Working directly on the row-major `jac_flat` storage via
                // a single reusable `col_buf` avoids the two dense (q × p) copies
                // (flatten→Array2, project, unflatten→Vec) that previously fired
                // per row. `t_buf` still holds the row's ext-coord vector.
                let mut t_buf = vec![0.0_f64; q];
                let mut col_buf = Array1::<f64>::zeros(q);
                for row_idx in 0..n {
                    let ext_row = ext.row(row_idx);
                    for (slot, &v) in t_buf.iter_mut().zip(ext_row.iter()) {
                        *slot = v;
                    }
                    let t_i = ArrayView1::from(t_buf.as_slice());
                    let jac_flat = &mut kron_jac[row_idx];
                    let q_row = jac_flat.len() / p;
                    for j in 0..p {
                        for c in 0..q_row {
                            col_buf[c] = jac_flat[c * p + j];
                        }
                        let projected_col =
                            manifold.project_to_tangent(t_i, col_buf.slice(ndarray::s![..q_row]));
                        for c in 0..q_row {
                            jac_flat[c * p + j] = projected_col[c];
                        }
                    }
                }
            }
        }
        // Build and install the Kronecker htbeta_matvec.
        //
        // `SaeKroneckerRows` holds per-row `(a_phi, local_jac)` and implements
        // the cross-block operator without ever materialising the dense
        // `(q × K·p)` slab.  The cross-block factorises as `H_tβ = L · J_β`,
        // where `J_β = φᵀ ⊗ I_p` projects a length-`K` β vector onto the
        // `p`-dimensional decoded output space (`apply_jbeta`) and `L_i` is
        // the per-row `(q_i × p)` assignment+coordinate Jacobian that lifts
        // that p-vector into the row's `q_i`-dim tangent block (`apply_l`).
        // Both factors are required: the contract of `set_row_htbeta_operator`
        // is `out.len() == d` (= `q_i`), so writing `apply_jbeta`'s p-vector
        // output directly into a length-`q_i` buffer overflows whenever
        // `p > q_i` (the common case once `p` reflects real feature width).
        // Symmetric for the transpose: `H_βt = J_βᵀ · Lᵀ`, so apply `Lᵀ`
        // first to map the q_i-vector back to p-space, then scatter through
        // the support.
        {
            let kron = Arc::new(SaeKroneckerRows::new(p, kron_a_phi, kron_jac));
            let kron_t = Arc::clone(&kron);
            let p_dim = p;
            sys.set_row_htbeta_operator(
                move |row_idx, x, out| {
                    // out = L_i · (J_β · x). Allocate a length-p scratch buffer
                    // for the intermediate decoded-output vector; both factors
                    // overwrite their output buffers (`apply_jbeta` zeroes
                    // before accumulating, `apply_l` writes per-row), so no
                    // pre-zeroing of `u_p`/`out` is needed.
                    let out_slice = out.as_slice_mut().expect("out is always standard-layout");
                    let mut u_p = vec![0.0_f64; p_dim];
                    if let Some(xs) = x.as_slice() {
                        kron.apply_jbeta(row_idx, xs, &mut u_p);
                    } else {
                        let x_vec: Vec<f64> = x.iter().copied().collect();
                        kron.apply_jbeta(row_idx, &x_vec, &mut u_p);
                    }
                    kron.apply_l(row_idx, &u_p, out_slice);
                },
                move |row_idx, v, out| {
                    // out += J_βᵀ · (Lᵀ · v). `apply_l_t` accumulates into a
                    // zero-initialised length-p buffer to produce the p-vector
                    // `Lᵀ v`; `scatter_jbeta_t` then adds φ_i[s] · u_p[j] into
                    // the length-K β accumulator at each active `(s, j)`.
                    let out_slice = out.as_slice_mut().expect("out is always standard-layout");
                    let mut u_p = vec![0.0_f64; p_dim];
                    if let Some(vs) = v.as_slice() {
                        kron_t.apply_l_t(row_idx, vs, &mut u_p);
                    } else {
                        let v_vec: Vec<f64> = v.iter().copied().collect();
                        kron_t.apply_l_t(row_idx, &v_vec, &mut u_p);
                    }
                    kron_t.scatter_jbeta_t(row_idx, &u_p, out_slice);
                },
            );
        }
        let mut beta_penalty_written = false;
        if let Some(registry) = analytic_penalties {
            // Upfront validation: refuse penalty kinds the SAE row layout
            // cannot host, and refuse mixed-d row-block configurations.
            // This makes the dispatch loop below total — no runtime
            // "unsupported penalty" fallthrough, no K-gating.
            self.validate_analytic_penalty_registry(registry)
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
            beta_penalty_written = self
                .add_sae_analytic_penalty_contributions(&mut sys, registry, penalty_scale)
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        }
        // Wire per-atom β block ranges so the Jacobi preconditioner builds one
        // dense Schur sub-block per atom (block-Jacobi) instead of scalar-diagonal
        // inversion.  Each atom's decoder coefficients form a natural block:
        // `[beta_offsets[k] .. beta_offsets[k] + basis_size[k] * p_out]`.
        sys.set_block_offsets(self.beta_block_offsets());
        // Install the composite BetaPenaltyOp (#296): smoothness contributions
        // via per-atom KroneckerPenaltyOp (avoid dense K×K materialisation), the
        // data-fit Gauss-Newton β-Hessian as the structured `G ⊗ I_p`
        // SparseBlockKroneckerPenaltyOp (block-sparse over co-occurring
        // `(atom, atom')` pairs, block-diagonal across the `p` output channels,
        // identical per channel), plus — only when a Beta-tier analytic penalty
        // was written — the dense `sys.hbb` residual contribution. When no beta
        // penalty fired, `sys.hbb` is all-zero and the dense `(K·p)²` operator
        // is skipped entirely. The sparse data op tracks only the active-atom
        // couplings, so its storage and matvec cost scale with `k_active`, not
        // `K`, at `K = 100K`.
        {
            // Convert the per-atom-pair coupling map into `SparseGBlock`s keyed
            // by μ-space offsets. Empty blocks (no co-occurrence) are simply
            // absent from the map.
            let g_sparse_blocks: Vec<SparseGBlock> = g_blocks
                .into_iter()
                .filter_map(|((atom_i, atom_j), data)| {
                    if data.iter().all(|&v| v == 0.0) {
                        None
                    } else {
                        Some(SparseGBlock {
                            row_off: mu_offsets[atom_i],
                            col_off: mu_offsets[atom_j],
                            data,
                        })
                    }
                })
                .collect();
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = smooth_ops;
            ops.push(Arc::new(SparseBlockKroneckerPenaltyOp {
                p,
                dim_a: m_total,
                k: beta_dim,
                blocks: g_sparse_blocks,
            }));
            if beta_penalty_written {
                ops.push(Arc::new(DensePenaltyOp(sys.hbb.clone())));
            }
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: beta_dim, ops }));
        }
        // Store the active-set layout for `apply_newton_step`.
        self.last_row_layout = row_layout;
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

    /// Numerical rank of a symmetric matrix: the count of eigenvalues
    /// exceeding `tol · max_eig`, with `tol = 1e-9` (the conventional
    /// relative spectral cutoff used elsewhere in the codebase).
    ///
    /// Used to count the penalised dimension of each atom's `smooth_penalty`
    /// `S_k` so the REML criterion's `−½·p·rank(S)·log λ_smooth` Occam term
    /// uses the *effective* penalty rank rather than the ambient basis size
    /// (a thin-plate / B-spline penalty has a non-trivial null space).
    fn symmetric_rank(s: &Array2<f64>) -> Result<usize, String> {
        let m = s.ncols();
        if m == 0 {
            return Ok(0);
        }
        // Symmetrise defensively — `smooth_penalty` is conceptually symmetric
        // but may be stored with tiny asymmetry from assembly arithmetic.
        let mut sym = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                sym[[i, j]] = 0.5 * (s[[i, j]] + s[[j, i]]);
            }
        }
        let (evals, _evecs) = sym
            .eigh(Side::Lower)
            .map_err(|e| format!("SaeManifoldTerm::symmetric_rank: eigh failed: {e}"))?;
        let max_eig = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v));
        if !(max_eig > 0.0) {
            return Ok(0);
        }
        let tol = 1.0e-9 * max_eig;
        Ok(evals.iter().filter(|&&v| v > tol).count())
    }

    /// True REML criterion for the SAE term at a FIXED ρ.
    ///
    /// Runs the inner `(t, β)` arrow-Schur Newton solve to convergence at the
    /// supplied ρ (with NO in-loop ARD update — ρ is owned by the engine),
    /// then forms the Laplace/REML cost
    ///
    /// ```text
    /// V(ρ) = ℓ_pen(t̂, β̂; ρ) + ½ log|H(t̂, β̂; ρ)|
    ///        − ½ · p · (Σ_k rank S_k) · log λ_smooth
    /// ```
    ///
    /// where `ℓ_pen = loss.total()` is the penalised objective at the inner
    /// optimum and `½ log|H|` is the Laplace normaliser. `H` is the joint
    /// `(t, β)` Hessian assembled by the arrow-Schur system; its `H_tt` block
    /// carries `α = exp(log_ard)` on its diagonal, so as α grows `½ log|H|`
    /// rises while the `−½·n·log α` already inside `loss.ard` falls — their
    /// balance IS the effective-dof term that the deleted `α = n/‖t‖²` rule
    /// dropped, which is why the criterion needs no clamp to stay finite on a
    /// collapsing axis.
    ///
    /// The final `−½·p·rank(S)·log λ_smooth` term is the smoothing-penalty
    /// normaliser `−½ log|λ S|_+` restricted to its ρ-dependent part: `S_k` is
    /// shared across all `p` decoder output channels (the `⊗ I_p` Kronecker
    /// structure), so `log|λ S|_+ = p·rank(S)·log λ + p·log|S|_+`, and the
    /// `½ p·log|S|_+` piece is ρ-independent. ALL ρ-independent additive
    /// constants (the `2π` Laplace constant, the base `½ p·log|S|_+` penalty
    /// logdet, the assignment-prior normaliser) are DROPPED here: they shift
    /// `V` by a constant and do not affect the ρ-argmin the engine seeks.
    ///
    /// Returns `(V, loss)` so the engine can both rank ρ and surface the inner
    /// loss breakdown.
    pub fn reml_criterion(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        let (v, loss, _cache) = self.reml_criterion_with_cache(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        Ok((v, loss))
    }

    /// As [`Self::reml_criterion`], but also returns the converged undamped
    /// `ArrowFactorCache` so callers (the EFS fixed-point step) can read the
    /// selected-inverse traces `(H⁻¹)_tt` / `(H⁻¹)_ββ` without re-factoring.
    /// The cache is the single shared O(K³) Direct factor; both the
    /// log-determinant criterion and the Fellner-Schall ρ-step consume it.
    #[allow(clippy::too_many_arguments)]
    pub fn reml_criterion_with_cache(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        // 1. Run the inner (t, β) Newton solve to convergence at FIXED ρ.
        //    `run_joint_fit_arrow_schur` no longer touches ρ.
        let mut rho_fixed = rho.clone();
        let loss = self.run_joint_fit_arrow_schur(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;

        // 2. One final UNDAMPED assemble + factor to obtain the converged
        //    joint Hessian log-determinant. We force ridge = 0 and the dense
        //    `Direct` Schur mode so `arrow_log_det_from_cache` returns the
        //    exact `log|H| = Σ_i log|H_tt^(i)| + log|Schur_β|` (it rejects
        //    damped factors and InexactPCG caches, which have no dense Schur
        //    factor). This is the same evidence convention the main GAM REML
        //    path uses.
        let sys = self
            .assemble_arrow_schur(target, rho, registry)
            .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
        // Evidence-only factorization: the Newton step (Δt, Δβ) is discarded
        // and only the factor cache is consumed — the exact undamped log-det
        // and the selected-inverse traces. As ρ sweeps to extremes (e.g. a
        // wide ARD-α sweep), H_tt is genuinely PD but can be ill-conditioned
        // (κ large); the standard Direct guard rejects that to protect
        // Newton-step accuracy, but the log-det is exact from diag(L)
        // regardless of κ and the traces only need the (PD) factor. So
        // tolerate the ill-conditioning rejection here (a genuine non-PD pivot
        // still errors). The cache stays undamped at ridge=0, so
        // `arrow_log_det_from_cache` remains exact.
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let (delta_t, delta_beta, cache): (Array1<f64>, Array1<f64>, ArrowFactorCache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
        // The Laplace normaliser ½log|H| is only the correct REML criterion at
        // the inner optimum (t̂, β̂). The undamped Newton step computed here is
        // the natural KKT residual of that solve: a step whose magnitude
        // dwarfs the iterate means the inner loop did not converge at this ρ
        // and the log-det would be evaluated off the manifold. Surface that as
        // a recoverable error rather than returning a meaningless criterion.
        let step_norm_sq: f64 = delta_t.iter().map(|&v| v * v).sum::<f64>()
            + delta_beta.iter().map(|&v| v * v).sum::<f64>();
        if !step_norm_sq.is_finite() {
            return Err(format!(
                "SaeManifoldTerm::reml_criterion: undamped Newton residual is non-finite at \
                 the inner optimum (‖Δ‖²={step_norm_sq}); the joint Hessian factorisation is \
                 degenerate at this ρ"
            ));
        }
        let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
            "SaeManifoldTerm::reml_criterion: arrow_log_det_from_cache returned None at \
             ridge=0 Direct mode (no dense Schur factor); the joint Hessian log-det is \
             required for the Laplace normaliser"
                .to_string()
        })?;

        // 3. Smoothing-penalty Occam term: −½·p·(Σ_k rank S_k)·log λ_smooth.
        let p_out = self.output_dim() as f64;
        let mut smooth_rank_total = 0usize;
        for atom in &self.atoms {
            smooth_rank_total += Self::symmetric_rank(&atom.smooth_penalty)?;
        }
        let occam = 0.5 * p_out * (smooth_rank_total as f64) * rho.log_lambda_smooth;

        let v = loss.total() + 0.5 * log_det - occam;
        Ok((v, loss, cache))
    }

    /// Per-atom, per-axis coordinate sum-of-squares `‖t_kj‖² = Σ_i t_{i,k,j}²`.
    ///
    /// This is the data-fit sufficient statistic for the ARD precision update
    /// (the numerator-side `‖t‖²` of the deleted `α = n/‖t‖²` rule). Returned
    /// per atom as an `Array1` of length `d_k`.
    fn ard_coord_sumsq(&self) -> Vec<Array1<f64>> {
        let mut out = Vec::with_capacity(self.k_atoms());
        for coord in &self.assignment.coords {
            let d = coord.latent_dim();
            let mut sq = Array1::<f64>::zeros(d);
            for row in 0..coord.n_obs() {
                let t = coord.row(row);
                for axis in 0..d {
                    sq[axis] += t[axis] * t[axis];
                }
            }
            out.push(sq);
        }
        out
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹) =
    /// Σ_i [(H⁻¹)_tt]_{(i,k,j),(i,k,j)}` from the converged factor cache.
    ///
    /// `cache.latent_block_inverse_diagonal()` returns the diagonal of the
    /// latent block `(H⁻¹)_tt` in the cache's compact per-row `delta_t`
    /// layout (length `row_offsets[N]`); each per-row block is laid out as
    /// `[logit scalars…, then per-active-atom coord axes…]`. This routine
    /// sums those diagonal entries over the coord positions belonging to each
    /// `(atom k, axis j)` across all observation rows where atom `k` is active.
    ///
    /// `self.last_row_layout` must be the layout from the *same* assemble that
    /// produced `cache`:
    /// - `Some(layout)`: compact active-set mode (JumpReLU / large-K
    ///   softmax-IBP truncation). For row `i`, atom `k`'s position in the
    ///   active list gives its compact coord-block start `coord_starts[i][pos]`;
    ///   inactive atoms contribute 0 (the prior dominates there anyway).
    /// - `None`: dense full-support layout, uniform row dim
    ///   `q = K + Σ d_k`; atom `k`'s coord block sits at the fixed full-row
    ///   offset `coord_offsets[k]` after the `K` logit scalars.
    ///
    /// This `tr_kj(H⁻¹)` is exactly the posterior-variance term the deleted
    /// `α = n/‖t‖²` rule dropped; the corrected Mackay/Fellner-Schall fixed
    /// point is `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))`.
    fn ard_inverse_traces(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        let inv_diag = cache.latent_block_inverse_diagonal()?;
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            traces[k][axis] += inv_diag[row_base + block_start + axis];
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            traces[k][axis] += inv_diag[row_base + block_start + axis];
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    /// Decoder smoothness penalty quadratic form `Σ_k Σ_oc B_k[:,oc]ᵀ S_k B_k[:,oc]`.
    ///
    /// This is `βᵀ (⊕_k S_k ⊗ I_p) β` — the un-scaled (λ-free) penalty energy
    /// in the flat β layout, the denominator of the λ_smooth Fellner-Schall
    /// update. `S_k` is symmetrised defensively (as the assembler does).
    fn decoder_smoothness_quadratic_form(&self) -> f64 {
        let mut acc = 0.0_f64;
        for atom in &self.atoms {
            let s = &atom.smooth_penalty;
            let b = &atom.decoder_coefficients;
            let m = atom.basis_size();
            let p = atom.output_dim();
            for oc in 0..p {
                for i in 0..m {
                    for j in 0..m {
                        let s_ij = 0.5 * (s[[i, j]] + s[[j, i]]);
                        acc += b[[i, oc]] * s_ij * b[[j, oc]];
                    }
                }
            }
        }
        acc
    }

    /// Effective penalized dof of the decoder smoothness penalty:
    /// `tr(S_β⁻¹ · M)` with `M = ⊕_k (λ_smooth · S_k) ⊗ I_p` embedded in the
    /// flat β layout, where `S_β⁻¹ = (H⁻¹)_ββ` is the Schur-complement inverse.
    ///
    /// Built per keystone's documented pattern on
    /// [`ArrowFactorCache::schur_inverse_apply`]:
    /// `tr(S_β⁻¹ M) = Σ_col e_colᵀ S_β⁻¹ M e_col`. Column `(k, μ, oc)` of `M`
    /// (global index `off_k + μ·p + oc`) is `λ·S_k[:,μ] ⊗ e_oc` — nonzero only
    /// at `off_k + ν·p + oc` for `ν in 0..M_k` — so we materialise just that
    /// sparse K-vector, apply `S_β⁻¹`, and read back `result[col]`. The
    /// `⊗ I_p` only couples equal `oc`, but `S_β` itself couples channels
    /// through the data-fit block, so all `p` channels are summed (no
    /// channel-block-identity shortcut). Total cost `beta_dim` Schur solves.
    fn decoder_smoothness_effective_dof(
        &self,
        cache: &ArrowFactorCache,
        lambda_smooth: f64,
    ) -> Result<f64, ArrowSchurError> {
        let p = self.output_dim();
        let beta_offsets = self.beta_offsets();
        let k = cache.k;
        let mut trace = 0.0_f64;
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = beta_offsets[atom_idx];
            for mu in 0..m {
                for oc in 0..p {
                    let col = off + mu * p + oc;
                    // M[:,col] = λ · S_k[:,mu] ⊗ e_oc (nonzero at off+ν·p+oc).
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * p + oc] = lambda_smooth * s_nu_mu;
                    }
                    let z = cache.schur_inverse_apply(m_col.view())?;
                    trace += z[col];
                }
            }
        }
        Ok(trace)
    }

    /// Returns `true` when a Beta-tier analytic penalty was accumulated into
    /// the dense `sys.hbb` block (so the caller knows to wrap it in a
    /// `DensePenaltyOp`); `false` leaves `sys.hbb` all-zero and lets the
    /// caller skip the dense `(K·p)²` operator entirely.
    fn add_sae_analytic_penalty_contributions(
        &self,
        sys: &mut ArrowSchurSystem,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
    ) -> Result<bool, ArrowSchurError> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let logits_flat = flat_logits(self.assignment.logits.view());
        let beta = self.flatten_beta();
        let mut beta_penalty_written = false;
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
                    } else {
                        // Every other Psi-tier penalty here is row-block
                        // supported with a coord-shape that matches each
                        // atom — `validate_analytic_penalty_registry`
                        // refused everything else upfront, so this branch
                        // is total and the K=1 vs K>=2 path is the same
                        // loop. Row-block coord penalties (ARD,
                        // BlockOrthogonality, Sparsity/TopK/JumpReLU,
                        // RowPrecisionPrior, ScadMcp, Isometry) target the
                        // "t" latent block (n_obs × d) and apply per atom
                        // — accumulate into the corresponding row offsets.
                        assert!(
                            sae_penalty_is_row_block_supported(penalty),
                            "validate_analytic_penalty_registry should have \
                             refused non-row-block Psi-tier penalty {:?} \
                             (registry layout name {name:?})",
                            penalty.name()
                        );
                        let offsets = self.assignment.coord_offsets();
                        for atom_idx in 0..self.k_atoms() {
                            let off = offsets[atom_idx];
                            let coord = &self.assignment.coords[atom_idx];
                            // Isometry requires per-step cache refresh from the
                            // atom's second jet before grad_target / hvp are live.
                            // The registry-held IsometryPenalty was constructed
                            // with p_out equal to the latent dim (the "t" block's
                            // `d` in the JSON latent spec) rather than the true
                            // decoder output dimension; we build a corrected copy
                            // here with the right p_out and populated caches.
                            if let AnalyticPenaltyKind::Isometry(iso) = penalty {
                                let atom = &self.atoms[atom_idx];
                                // The registry penalty was built with p_out equal
                                // to the latent dim (the "t" JSON block's `d`),
                                // but IsometryPenalty.p_out must match the atom's
                                // decoder output dim. Clone and fix p_out before
                                // calling refresh_isometry_caches_from_atom.
                                let p = atom.decoder_coefficients.ncols();
                                let mut corrected: IsometryPenalty = (**iso).clone();
                                corrected.p_out = p;
                                let coords_mat = coord.as_matrix();
                                let second_jet_installed = refresh_isometry_caches_from_atom(
                                    &corrected,
                                    atom,
                                    coords_mat.view(),
                                )
                                .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
                                if !second_jet_installed {
                                    // refresh_isometry_caches_from_atom returns
                                    // false when the atom's basis_second_jet slot
                                    // is None (atom was registered via
                                    // with_basis_evaluator, not with_basis_second_jet).
                                    // Fall back to second_jet_dyn, which evaluators
                                    // like AffineCoordinateEvaluator override even
                                    // when the SaeBasisSecondJet supertrait Arc is
                                    // not in the slot.
                                    match atom
                                        .basis_evaluator
                                        .as_ref()
                                        .and_then(|e| e.second_jet_dyn(coords_mat.view()))
                                    {
                                        Some(Ok(hess)) => {
                                            let n_obs = coords_mat.nrows();
                                            let d = atom.latent_dim;
                                            let m = atom.basis_size();
                                            if hess.dim() != (n_obs, m, d, d) {
                                                return Err(ArrowSchurError::SchurFactorFailed {
                                                    reason: format!(
                                                        "SAE Isometry atom '{}': second_jet_dyn \
                                                     returned shape {:?}, expected \
                                                     ({n_obs}, {m}, {d}, {d})",
                                                        atom.name,
                                                        hess.dim()
                                                    ),
                                                });
                                            }
                                            let b = &atom.decoder_coefficients;
                                            let mut jac2 = Array2::<f64>::zeros((n_obs, p * d * d));
                                            for n in 0..n_obs {
                                                for i in 0..p {
                                                    for a in 0..d {
                                                        for c in 0..d {
                                                            let mut acc = 0.0;
                                                            for mm in 0..m {
                                                                acc += hess[[n, mm, a, c]]
                                                                    * b[[mm, i]];
                                                            }
                                                            jac2[[n, (i * d + a) * d + c]] = acc;
                                                        }
                                                    }
                                                }
                                            }
                                            corrected
                                                .set_jacobian_second_cache(Some(Arc::new(jac2)));
                                        }
                                        Some(Err(reason)) => {
                                            return Err(ArrowSchurError::SchurFactorFailed {
                                                reason,
                                            });
                                        }
                                        None => {
                                            return Err(ArrowSchurError::SchurFactorFailed {
                                                reason: format!(
                                                    "IsometryPenalty requested for SAE atom \
                                                 '{}' (basis kind {:?}) but this evaluator \
                                                 does not expose an analytic second jet; \
                                                 use AffineCoordinateEvaluator, \
                                                 SphereChartEvaluator, \
                                                 PeriodicHarmonicEvaluator, or \
                                                 TorusHarmonicEvaluator for SAE-Isometry",
                                                    atom.name, atom.basis_kind
                                                ),
                                            });
                                        }
                                    }
                                }
                                let corrected_kind =
                                    AnalyticPenaltyKind::Isometry(Arc::new(corrected));
                                self.add_sae_coord_penalty(
                                    sys,
                                    off,
                                    coord,
                                    &corrected_kind,
                                    rho_local,
                                );
                            } else {
                                self.add_sae_coord_penalty(sys, off, coord, penalty, rho_local);
                            }
                        }
                    }
                }
                PenaltyTier::Beta => {
                    // β-tier analytic penalties are global (B-only); minibatch-
                    // scaled so per-chunk sums reconstruct one global copy.
                    self.add_sae_beta_penalty(sys, penalty, beta.view(), rho_local, penalty_scale);
                    beta_penalty_written = true;
                }
                PenaltyTier::Rho => {}
            }
        }
        Ok(beta_penalty_written)
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
        // The ArrowSchur `htt` block is the Newton / PIRLS curvature operator and
        // must stay PSD. Nonconvex sparsifiers (log, JumpReLU) have an indefinite
        // true Hessian, so we accumulate the PSD majorizer here — never the exact
        // `hessian_diag`, which goes negative and would destroy the solve's
        // positive-definiteness. Convex penalties' majorizer equals their exact
        // Hessian, so this is exact for them.
        if let Some(diag) = penalty.psd_majorizer_diag(target, rho_local) {
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
        // `htt` is the PSD Newton / PIRLS curvature block: accumulate the PSD
        // majorizer (exact for convex penalties), not the indefinite exact
        // Hessian, for the same reason as `add_sae_logit_penalty` above.
        if let Some(diag) = penalty.psd_majorizer_diag(target, rho_local) {
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
            let hv = penalty.psd_majorizer_hvp(target, rho_local, probe.view());
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
        penalty_scale: f64,
    ) {
        // MechanismSparsityPenalty is a group-lasso over a single
        // (latent_dim, p) decoder matrix and indexes its target via
        // `target.range.start + latent * p + feature`, treating its range as
        // one contiguous (M, p) block. The flat SAE β layout concatenates the
        // per-atom decoder blocks `[B_1 (M_1×p), B_2 (M_2×p), …]`, so for K≥2
        // (and in general for K=1, where it collapses to the same single
        // block) the penalty must operate per atom on its own
        // `[beta_offsets[k] .. beta_offsets[k+1])` slice with `latent_dim = M_k`.
        // Build a per-atom view of the penalty (cloning only the cheap
        // descriptor: range + latent_dim) and accumulate each atom's
        // contribution into the corresponding β segment. This removes the
        // K≥2 limitation (#240) at root rather than guarding it away.
        if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
            let beta_offsets = self.beta_offsets();
            let p = self.output_dim();
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let m = atom.basis_size();
                let start = beta_offsets[atom_idx];
                let end = start + m * p;
                let mut per_atom: MechanismSparsityPenalty = (**base).clone();
                per_atom.target = PsiSlice {
                    range: start..end,
                    latent_dim: Some(m),
                };
                self.add_sae_mech_sparsity_atom(
                    sys,
                    &per_atom,
                    target_beta,
                    rho_local,
                    start,
                    end,
                    penalty_scale,
                );
            }
            return;
        }
        let k = self.beta_dim();
        let grad = penalty.grad_target(target_beta, rho_local);
        for j in 0..k {
            sys.gb[j] += penalty_scale * grad[j];
        }
        // `hbb` is the PSD Newton / PIRLS curvature block for the β tier:
        // accumulate the PSD majorizer (exact for convex penalties), not the
        // indefinite exact Hessian, so the solve stays positive-definite.
        if let Some(diag) = penalty.psd_majorizer_diag(target_beta, rho_local) {
            for j in 0..k {
                sys.hbb[[j, j]] += penalty_scale * diag[j];
            }
            return;
        }
        let mut probe = Array1::<f64>::zeros(k);
        for j in 0..k {
            probe.fill(0.0);
            probe[j] = 1.0;
            let hv = penalty.psd_majorizer_hvp(target_beta, rho_local, probe.view());
            for i in 0..k {
                sys.hbb[[i, j]] += penalty_scale * hv[i];
            }
        }
    }

    /// Accumulate one atom's MechanismSparsity contribution into `sys`. The
    /// `per_atom` penalty has its `target.range` set to that atom's β segment
    /// `[start, end)` and `latent_dim = M_k`, so `grad_target` / `hvp` return
    /// full-length β vectors whose nonzero support lies inside `[start, end)`.
    /// The Hessian probe only needs to sweep that segment, and its support is
    /// likewise confined to `[start, end)`, so the inner accumulation is
    /// quadratic in the atom's block size rather than the full β dimension.
    fn add_sae_mech_sparsity_atom(
        &self,
        sys: &mut ArrowSchurSystem,
        per_atom: &MechanismSparsityPenalty,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        start: usize,
        end: usize,
        penalty_scale: f64,
    ) {
        let grad = per_atom.grad_target(target_beta, rho_local);
        for j in start..end {
            sys.gb[j] += penalty_scale * grad[j];
        }
        let k = self.beta_dim();
        let mut probe = Array1::<f64>::zeros(k);
        for j in start..end {
            probe.fill(0.0);
            probe[j] = 1.0;
            // `hbb` is the PSD Newton / PIRLS curvature block, so probe the PSD
            // majorizer. The group-lasso Hessian `factor·(I − ŵŵᵀ)/‖w‖` is
            // already PSD, so its majorizer equals the exact Hessian (the trait
            // default delegates), but we use the majorizer name to honor the
            // curvature-block contract uniformly with the other SAE penalties.
            let hv = per_atom.psd_majorizer_hvp(target_beta, rho_local, probe.view());
            for i in start..end {
                sys.hbb[[i, j]] += penalty_scale * hv[i];
            }
        }
    }

    pub fn solve_newton_step(
        &mut self,
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
            .map(|(delta_t, delta_beta, _diag)| (delta_t, delta_beta))
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

    /// Capture the mutable state perturbed by an `apply_newton_step` +
    /// `loss` line-search trial (decoder coefficients, basis evaluations,
    /// assignment logits, latent coordinates). See
    /// [`SaeManifoldMutableState`].
    fn snapshot_mutable_state(&self) -> SaeManifoldMutableState {
        let atoms = self
            .atoms
            .iter()
            .map(|atom| {
                (
                    atom.basis_values.clone(),
                    atom.basis_jacobian.clone(),
                    atom.decoder_coefficients.clone(),
                )
            })
            .collect();
        SaeManifoldMutableState {
            atoms,
            logits: self.assignment.logits.clone(),
            coords: self.assignment.coords.clone(),
        }
    }

    /// Restore the mutable state captured by [`Self::snapshot_mutable_state`].
    /// Assigns into the existing arrays in place so the restore reuses the
    /// already-allocated buffers rather than reallocating per trial.
    fn restore_mutable_state(&mut self, snapshot: &SaeManifoldMutableState) {
        for (atom, (basis_values, basis_jacobian, decoder)) in
            self.atoms.iter_mut().zip(snapshot.atoms.iter())
        {
            atom.basis_values.assign(basis_values);
            atom.basis_jacobian.assign(basis_jacobian);
            atom.decoder_coefficients.assign(decoder);
        }
        self.assignment.logits.assign(&snapshot.logits);
        self.assignment.coords.clone_from(&snapshot.coords);
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
        let k_atoms = self.k_atoms();
        if delta_beta.len() != self.beta_dim() {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: delta_beta length {} != expected {}",
                delta_beta.len(),
                self.beta_dim()
            ));
        }

        // When last_row_layout is set (any compact active-set mode — JumpReLU
        // gate or large-K softmax/IBP truncation), delta_ext_coord uses a
        // variable-stride layout where row i occupies
        // [compact_offset_i .. compact_offset_i + q_active_i].
        // We expand each row back to full-q before applying.
        if let Some(ref layout) = self.last_row_layout.clone() {
            let total_len: usize = (0..n).map(|row| layout.row_q_active(row)).sum();
            if delta_ext_coord.len() != total_len {
                return Err(format!(
                    "SaeManifoldTerm::apply_newton_step: compact delta_ext_coord length {} != expected {}",
                    delta_ext_coord.len(),
                    total_len
                ));
            }
            // Expand compact layout to full-q flat buffer.
            let mut full_delta = vec![0.0_f64; n * q];
            let mut compact_off = 0usize;
            for row in 0..n {
                let q_active = layout.row_q_active(row);
                // Collect compact row (handles both contiguous and strided views).
                let compact_row: Vec<f64> = delta_ext_coord
                    .slice(ndarray::s![compact_off..compact_off + q_active])
                    .iter()
                    .copied()
                    .collect();
                layout.expand_row(row, &compact_row, &mut full_delta[row * q..(row + 1) * q]);
                compact_off += q_active;
            }
            // Apply logits from expanded buffer.
            for row in 0..n {
                let row_base = row * q;
                for atom_idx in 0..k_atoms {
                    self.assignment.logits[[row, atom_idx]] +=
                        step_size * full_delta[row_base + atom_idx];
                }
            }
            // Apply coords from expanded buffer.
            let coord_offsets = self.assignment.coord_offsets();
            for atom_idx in 0..k_atoms {
                let d = self.assignment.coords[atom_idx].latent_dim();
                let mut delta_coord = Array1::<f64>::zeros(n * d);
                for row in 0..n {
                    let row_base = row * q + coord_offsets[atom_idx];
                    for axis in 0..d {
                        delta_coord[row * d + axis] = full_delta[row_base + axis];
                    }
                }
                self.assignment.coords[atom_idx].retract_flat_delta(delta_coord.view());
                if refresh_basis {
                    let coords = self.assignment.coords[atom_idx].as_matrix();
                    self.atoms[atom_idx].refresh_basis(coords.view())?;
                }
            }
        } else {
            // Dense layout: uniform q per row.
            if delta_ext_coord.len() != n * q {
                return Err(format!(
                    "SaeManifoldTerm::apply_newton_step: delta_ext_coord length {} != expected {}",
                    delta_ext_coord.len(),
                    n * q
                ));
            }
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
        // ── Pre-fit decoder identifiability audit ──────────────────────────
        //
        // Each decoder atom `k` contributes `η_i += a_ik · Φ_k(t_ik) · B_k`,
        // with `B_k ∈ ℝ^{M_k × p}`. The decoder Hessian for atom `k` is
        // `H_data = G_k ⊗ I_p` where `G_k = (diag(a_·k)·Φ_k)ᵀ (diag(a_·k)·Φ_k)`
        // (the diagonal `(atom_k, atom_k)` block of the sparse data Gram `G`
        // assembled in `assemble_arrow_schur`); the
        // `p` output channels share the identical `M_k × M_k` Gram, so decoder
        // identifiability is fully determined by the per-atom `(n, M_k)` design
        // `D_k = diag(a_·k)·Φ_k`. The `p`-fold output replication carries no
        // extra structural information and must NOT be materialised — doing so
        // (the former `(n·p, M_k·p)` channel-block route through the
        // cross-block flat audit) broadcast an `(n·p)`-row Jacobian into the
        // `n`-row placeholder design and panicked inside ndarray.
        //
        // We therefore run the rank check directly on each `D_k`. A
        // rank-deficient `D_k` means atom `k`'s decoder block Hessian is
        // singular (its Cholesky will fail or produce garbage steps), which is
        // surfaced as an immediate fatal error. Near-rank-deficient columns are
        // logged as INFO so callers can see which atoms are weakly identified.
        //
        // The check is performed chunk-aware through the per-atom Gram
        // accumulator: the full-batch path is the single-chunk special case.
        // `D_k`'s singular spectrum equals `√spec(G_k)` with
        // `G_k = D_kᵀ D_k`, so accumulating `G_k` over the whole design and
        // taking its eigenvalues reproduces the former pivoted-QR rank exactly
        // while never retaining an `(N × M_k)` design.
        {
            let mut grams = self.empty_decoder_gram_accumulator();
            self.accumulate_decoder_gram(&mut grams);
            self.finalize_decoder_identifiability_audit(&grams, self.n_obs())?;
        }
        for _ in 0..max_iter {
            self.advance_temperature_schedule()?;
            // ρ (including the ARD precisions) is owned by the outer engine
            // (`SaeManifoldOuterObjective`) and held FIXED across this inner
            // (t, β) Newton solve. The inner loop solves the joint manifold +
            // decoder system at the engine's current ρ; the engine alone
            // moves ρ by minimising the true REML criterion (see
            // `SaeManifoldTerm::reml_criterion`). The former in-loop
            // `update_ard_reml` rule (α = n / ‖t‖²) dropped the logdet /
            // effective-dof term and collapsed α on near-degenerate axes; it
            // has been removed in favour of the criterion-driven update.
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
            let (delta_ext_coord, delta_beta, _diag) = sys
                .solve_with_lm_escalation(ridge_ext_coord, ridge_beta)
                .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            let directional_decrease = sae_manifold_newton_directional_decrease(
                &sys,
                delta_ext_coord.view(),
                delta_beta.view(),
            );
            // Relative-scale floor on the directional decrease. When the
            // gradient is nearly orthogonal to the Newton step (ill-conditioned
            // near-convergence), `directional_decrease` collapses to O(machine
            // epsilon · ‖g‖ · ‖Δ‖). At that scale the Armijo bound
            // `pre_step_total − c1·step·directional_decrease` is numerically
            // indistinguishable from `pre_step_total`, so the line search would
            // "accept" on rounding noise. Treat that as converged and stop. The
            // norms are the natural scale of the inner product; the relative
            // constant keeps the reduction term distinguishable from rounding at
            // full step size given SAE_MANIFOLD_ARMIJO_C1 = 1e-4.
            let mut grad_norm_sq = 0.0;
            for (row_idx, row) in sys.rows.iter().enumerate() {
                let di = sys.row_dims[row_idx];
                for axis in 0..di {
                    grad_norm_sq += row.gt[axis] * row.gt[axis];
                }
            }
            for idx in 0..sys.k {
                grad_norm_sq += sys.gb[idx] * sys.gb[idx];
            }
            let mut step_norm_sq = 0.0;
            for &v in delta_ext_coord.iter() {
                step_norm_sq += v * v;
            }
            for &v in delta_beta.iter() {
                step_norm_sq += v * v;
            }
            let directional_decrease_floor = 1.0e-14 * grad_norm_sq.sqrt() * step_norm_sq.sqrt();
            // Snapshot only the state that `apply_newton_step` + `loss`
            // perturb (decoder coefficients, basis evaluations, assignment
            // logits/coords) once before the line search. Each rejected trial
            // restores from this snapshot in place; the static atom metadata,
            // smoothness penalties and basis-evaluator `Arc`s are never
            // re-cloned. This replaces the per-halving full `self.clone()`,
            // whose dominant cost was copying the `O(N·M·d)` `basis_jacobian`
            // and `O(N·M)` `basis_values` on every backtrack.
            let snapshot = self.snapshot_mutable_state();
            if !(pre_step_total.is_finite()
                && directional_decrease.is_finite()
                && directional_decrease > 0.0
                && directional_decrease > directional_decrease_floor)
            {
                // Pre-step state is unperturbed here; restore is a no-op but
                // keeps the invariant explicit.
                self.restore_mutable_state(&snapshot);
                break;
            }

            let mut trial_step_size = step_size;
            let mut accepted = false;
            for halving in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                if halving > 0 {
                    // Reset to the pre-step state before re-applying at the
                    // halved step. The first trial starts from the pre-step
                    // state already, so the restore is only needed after a
                    // rejected trial mutated `self`.
                    self.restore_mutable_state(&snapshot);
                }
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
                self.restore_mutable_state(&snapshot);
                break;
            }
        }
        // ρ is owned by the outer engine and unchanged here; just return the
        // converged inner loss at the fixed ρ.
        self.loss(target, rho)
    }

    /// Allocate one zero `(M_k × M_k)` Gram accumulator per atom for the
    /// chunk-aware decoder identifiability audit.
    fn empty_decoder_gram_accumulator(&self) -> Vec<Array2<f64>> {
        self.atoms
            .iter()
            .map(|atom| {
                let m = atom.basis_size();
                Array2::<f64>::zeros((m, m))
            })
            .collect()
    }

    /// Accumulate this term's (chunk's) contribution to the per-atom weighted
    /// design Gram `G_k += D_kᵀ D_k`, with `D_k = diag(a_·k)·Φ_k`.
    ///
    /// `grams[k]` must be `(M_k × M_k)`. Streaming callers invoke this once per
    /// chunk against the freshly materialized chunk term; the full-batch path
    /// invokes it once against `self`. The Gram is symmetric and channel-free
    /// (the `p`-fold output replication is carried by the `⊗ I_p` Kronecker
    /// structure, so it adds no rank information), so accumulating `Φ` weighted
    /// by the per-row assignment exactly reproduces the data-fit decoder block
    /// curvature `G_k` that `assemble_arrow_schur` installs.
    fn accumulate_decoder_gram(&self, grams: &mut [Array2<f64>]) {
        let n = self.n_obs();
        let assignments = self.assignment.assignments();
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            if m == 0 {
                continue;
            }
            let assign_col = assignments.column(atom_idx);
            let gram = &mut grams[atom_idx];
            // G_k += Σ_row a_row² · φ_row φ_rowᵀ. Hoist the weighted row into a
            // scratch vector so the rank-1 update is one O(M²) pass per row.
            let mut weighted = vec![0.0_f64; m];
            for row in 0..n {
                let a_k = assign_col[row];
                if a_k == 0.0 {
                    continue;
                }
                for col in 0..m {
                    weighted[col] = a_k * atom.basis_values[[row, col]];
                }
                for i in 0..m {
                    let wi = weighted[i];
                    if wi == 0.0 {
                        continue;
                    }
                    for j in 0..m {
                        gram[[i, j]] += wi * weighted[j];
                    }
                }
            }
        }
    }

    /// Decide rank-deficiency of each accumulated decoder Gram and surface the
    /// same fatal / INFO contract as the former pivoted-QR audit.
    fn finalize_decoder_identifiability_audit(
        &self,
        grams: &[Array2<f64>],
        n_total: usize,
    ) -> Result<(), String> {
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            if m == 0 {
                continue;
            }
            let rank =
                crate::solver::identifiability_audit::rank_of_gram(&grams[atom_idx], n_total)
                    .map_err(|e| {
                        format!(
                            "SaeManifoldTerm: pre-fit decoder audit (atom '{}'): \
                         Gram eigendecomposition failed: {e}",
                            atom.name,
                        )
                    })?;
            if rank < m {
                let dropped = m - rank;
                if rank == 0 {
                    return Err(format!(
                        "SaeManifoldTerm: pre-fit identifiability audit: decoder atom '{}' has \
                         rank-0 weighted design (n={n_total}, M_k={m}); all assignment weights \
                         vanish or the basis is degenerate, so the Arrow-Schur Newton system for \
                         this block is singular",
                        atom.name,
                    ));
                }
                log::info!(
                    "[SAE-AUDIT] decoder atom '{}' weighted design is rank-deficient \
                     (rank={rank}/{m}, {dropped} weakly-identified column(s), n={n_total}); the \
                     Arrow-Schur ridge will regularise the deficient directions",
                    atom.name,
                );
            }
        }
        Ok(())
    }

    /// Materialize a row-chunk `[start, end)` of this term as a standalone
    /// `SaeManifoldTerm`, recomputing `(basis_values, basis_jacobian)` on demand
    /// from the persisted decoder + atom geometry and the caller-supplied
    /// per-chunk `(logits, coords)`.
    ///
    /// The streaming joint fit NEVER persists the `(N × M)` basis or `(N × K)`
    /// logit buffers. Instead, for each chunk the caller re-seeds the chunk's
    /// latent state (the SAE PCA seed restricted to the chunk's `Z` rows for the
    /// coordinates, and the chunk's gating logits) and this constructor rebuilds
    /// a small `n_chunk`-row term whose atoms share the global decoder
    /// coefficients (`B_k`), smoothness penalties, and basis evaluators with
    /// `self`. Each atom's basis is re-evaluated at the chunk coordinates via
    /// its `basis_evaluator`, so the chunk term is exactly the restriction of
    /// the global model to those rows.
    ///
    /// Errors if any atom lacks a basis evaluator (a streaming fit must be able
    /// to re-evaluate `Φ(t)` at the per-chunk coordinates) or if the supplied
    /// shapes disagree with the term's atom layout.
    pub fn materialize_chunk(
        &self,
        chunk_logits: Array2<f64>,
        chunk_coords: Vec<Array2<f64>>,
    ) -> Result<SaeManifoldTerm, String> {
        let k_atoms = self.k_atoms();
        if chunk_logits.ncols() != k_atoms {
            return Err(format!(
                "SaeManifoldTerm::materialize_chunk: chunk_logits has {} cols but K={k_atoms}",
                chunk_logits.ncols()
            ));
        }
        if chunk_coords.len() != k_atoms {
            return Err(format!(
                "SaeManifoldTerm::materialize_chunk: chunk_coords has {} atoms but K={k_atoms}",
                chunk_coords.len()
            ));
        }
        let n_chunk = chunk_logits.nrows();
        let mut atoms = Vec::with_capacity(k_atoms);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let coords = &chunk_coords[atom_idx];
            if coords.nrows() != n_chunk || coords.ncols() != atom.latent_dim {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom {atom_idx} coords shape {:?} != ({n_chunk}, {})",
                    coords.dim(),
                    atom.latent_dim
                ));
            }
            let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' has no basis evaluator; a \
                     streaming fit must re-evaluate Φ(t) at each chunk's coordinates",
                    atom.name
                )
            })?;
            let (phi, jet) = evaluator.evaluate(coords.view())?;
            let m = atom.basis_size();
            if phi.dim() != (n_chunk, m) {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' evaluator returned Φ {:?}, expected ({n_chunk}, {m})",
                    atom.name,
                    phi.dim()
                ));
            }
            if jet.dim() != (n_chunk, m, atom.latent_dim) {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' evaluator returned jet {:?}, expected ({n_chunk}, {m}, {})",
                    atom.name,
                    jet.dim(),
                    atom.latent_dim
                ));
            }
            let mut chunk_atom = SaeManifoldAtom::new(
                atom.name.clone(),
                atom.basis_kind.clone(),
                atom.latent_dim,
                phi,
                jet,
                atom.decoder_coefficients.clone(),
                atom.smooth_penalty.clone(),
            )?;
            chunk_atom.basis_evaluator = atom.basis_evaluator.clone();
            chunk_atom.basis_second_jet = atom.basis_second_jet.clone();
            atoms.push(chunk_atom);
        }
        // Rebuild the assignment from the chunk's logits + coords, preserving
        // each atom's latent manifold and the global assignment mode.
        let coord_values: Vec<LatentCoordValues> = chunk_coords
            .iter()
            .zip(self.assignment.coords.iter())
            .map(|(c, src)| {
                LatentCoordValues::from_matrix_with_manifold(
                    c.view(),
                    LatentIdMode::None,
                    src.manifold().clone(),
                )
            })
            .collect();
        let assignment =
            SaeAssignment::with_mode(chunk_logits, coord_values, self.assignment.mode)?;
        let mut term = SaeManifoldTerm::new(atoms, assignment)?;
        // The temperature schedule is global outer state; the chunk term is
        // assembled at the schedule's current temperature, which the caller
        // already baked into `self.assignment.mode` before materializing.
        term.temperature_schedule = self.temperature_schedule.clone();
        Ok(term)
    }

    /// Streaming / minibatch joint fit: refine the shared decoder coefficients
    /// `B_k` (and the ARD ρ axes) by sweeping the rows in chunks of
    /// `chunk_size`, accumulating the reduced Schur system over the shared β
    /// online, and NEVER materializing the `(N × M)` / `(N × K)` per-row
    /// buffers.
    ///
    /// For each outer iteration:
    ///
    /// 1. Each chunk `[start, end)` re-seeds its per-row latent state from the
    ///    chunk's `Z` slice (`chunk_init` supplies `(logits, coords)` — the SAE
    ///    PCA seed restricted to the chunk), materializes a small chunk term via
    ///    [`Self::materialize_chunk`], and assembles its Arrow-Schur system with
    ///    the β-tier penalties scaled by the chunk fraction `n_chunk / N` (so
    ///    they sum to exactly one global copy across the pass).
    /// 2. The chunk's reduced contribution `H_βt(H_tt)⁻¹H_tβ` and `H_βt(H_tt)⁻¹g_t`
    ///    are accumulated into a single global [`StreamingArrowSchur`] over β,
    ///    consuming each chunk's Kronecker `htbeta_matvec` procedurally.
    /// 3. After one full pass, the global reduced system is solved for `Δβ` with
    ///    the same LM ridge escalation as the full-batch driver, and a streaming
    ///    Armijo line search on `Δβ` accepts the step against the summed
    ///    per-chunk loss.
    /// 4. ARD ρ is refreshed online from the accumulated `Σ‖t‖²` and row count.
    ///
    /// Only the global decoder coefficients persist across chunks and outer
    /// iterations; the per-row `(logits, coords)` are re-seeded each pass and
    /// discarded. `self`'s own per-row buffers are left untouched — the fitted
    /// decoder is written back into `self`'s atoms.
    pub fn run_joint_fit_arrow_schur_streaming<F>(
        &mut self,
        n_total: usize,
        chunk_size: usize,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        mut chunk_init: F,
    ) -> Result<SaeManifoldLoss, String>
    where
        F: FnMut(usize, usize) -> Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>), String>,
    {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: step_size must be finite and positive; got {step_size}"
            ));
        }
        if chunk_size == 0 {
            return Err(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: chunk_size must be positive"
                    .to_string(),
            );
        }
        if n_total == 0 {
            return Err(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: n_total must be positive"
                    .to_string(),
            );
        }
        let beta_dim = self.beta_dim();

        // ── Chunk-aware pre-fit decoder identifiability audit ───────────────
        {
            let mut grams = self.empty_decoder_gram_accumulator();
            let mut start = 0usize;
            while start < n_total {
                let end = (start + chunk_size).min(n_total);
                let (logits, coords, _z_chunk) = chunk_init(start, end)?;
                let chunk = self.materialize_chunk(logits, coords)?;
                chunk.accumulate_decoder_gram(&mut grams);
                start = end;
            }
            self.finalize_decoder_identifiability_audit(&grams, n_total)?;
        }

        let mut last_loss = SaeManifoldLoss {
            data_fit: 0.0,
            assignment_sparsity: 0.0,
            smoothness: 0.0,
            ard: 0.0,
        };
        for _ in 0..max_iter {
            self.advance_temperature_schedule()?;
            // ── Pass 1: accumulate the global reduced Schur over β online. ──
            let options = ArrowSolveOptions::automatic(beta_dim);
            let mut s_acc = Array2::<f64>::zeros((beta_dim, beta_dim));
            let mut rhs_acc = Array1::<f64>::zeros(beta_dim);
            let mut gb_acc = Array1::<f64>::zeros(beta_dim);
            // ρ (including the ARD precisions) is owned by the outer engine and
            // held FIXED across this streaming inner solve; the former online
            // `Σ t²` ARD accumulator + `update_ard_reml_from_sumsq` rule has
            // been removed in favour of the criterion-driven update.
            let mut pre_step_total = 0.0_f64;
            // Retain only the per-chunk row ranges so the line search can
            // re-materialize each chunk by re-invoking `chunk_init` at trial β
            // values. The chunk's `(logits, coords, Z)` are re-provided by the
            // seeder each time — never retained — so the pass stays O(Σ M_k²)
            // in memory rather than O(N · M) / O(N · K).
            let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
            let mut start = 0usize;
            while start < n_total {
                let end = (start + chunk_size).min(n_total);
                let n_chunk = end - start;
                let penalty_scale = n_chunk as f64 / n_total as f64;
                let (logits, coords, z_chunk) = chunk_init(start, end)?;
                if z_chunk.dim() != (n_chunk, self.output_dim()) {
                    return Err(format!(
                        "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: chunk [{start}, {end}) \
                         Z slice shape {:?} != ({n_chunk}, {})",
                        z_chunk.dim(),
                        self.output_dim()
                    ));
                }
                let mut chunk = self.materialize_chunk(logits, coords)?;
                chunk_ranges.push((start, end));
                pre_step_total += chunk
                    .loss_scaled(z_chunk.view(), rho, penalty_scale)?
                    .total();
                let sys = chunk
                    .assemble_arrow_schur_scaled(
                        z_chunk.view(),
                        rho,
                        analytic_penalties,
                        penalty_scale,
                    )
                    .map_err(|err| {
                        format!("SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: {err}")
                    })?;
                // Accumulate the chunk's data-fit β gradient (its g_β already
                // carries the minibatch-scaled β-penalty gradient).
                for j in 0..beta_dim {
                    gb_acc[j] += sys.gb[j];
                }
                Self::accumulate_chunk_reduced_schur(
                    &sys,
                    ridge_ext_coord,
                    &options,
                    &mut s_acc,
                    &mut rhs_acc,
                )?;
                start = end;
            }
            // The summed chunk β-blocks already reconstruct the full
            // `H_ββ` (data-fit GN `G ⊗ I_p` + smoothness + analytic β); add the
            // global β ridge exactly once, and form the reduced RHS. After this
            // step `rhs_acc = Σ_i H_βt^(i)(H_tt^(i))⁻¹g_t^(i) − g_β` is the
            // negated Schur-reduced β gradient `−g_reduced`, so the reduced
            // system `S Δβ = rhs_acc` yields the marginal Newton step in β with
            // the per-row latent eliminated.
            for j in 0..beta_dim {
                s_acc[[j, j]] += ridge_beta;
                rhs_acc[j] -= gb_acc[j];
            }
            // ── Solve the global reduced β system with LM ridge escalation. ──
            let delta_beta =
                solve_streaming_reduced_beta(&s_acc, &rhs_acc, &options).map_err(|err| {
                    format!("SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: {err}")
                })?;
            // ── Streaming Armijo line search on Δβ. ──
            // The directional decrease uses the *reduced* β gradient
            // `g_reduced = −rhs_acc`, the true gradient of the β-marginal
            // objective along which the line search backtracks (the per-row
            // latent block is profiled out, not stepped, in streaming).
            let beta0 = self.flatten_beta();
            let mut directional_decrease = 0.0_f64;
            for j in 0..beta_dim {
                // dd = −(g_reduced · Δβ) = −((−rhs_acc) · Δβ) = rhs_acc · Δβ.
                directional_decrease += rhs_acc[j] * delta_beta[j];
            }
            if !(pre_step_total.is_finite()
                && directional_decrease.is_finite()
                && directional_decrease > 0.0)
            {
                // No descent direction available; ρ is engine-owned and fixed,
                // so just record the loss and stop.
                last_loss = self.streaming_loss(&chunk_ranges, rho, n_total, &mut chunk_init)?;
                break;
            }
            let mut trial_step = step_size;
            let mut accepted_loss: Option<SaeManifoldLoss> = None;
            for _ in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                let mut trial_beta = beta0.clone();
                for j in 0..beta_dim {
                    trial_beta[j] += trial_step * delta_beta[j];
                }
                self.set_flat_beta(trial_beta.view())?;
                let trial_loss =
                    self.streaming_loss(&chunk_ranges, rho, n_total, &mut chunk_init)?;
                let trial_total = trial_loss.total();
                let armijo_bound =
                    pre_step_total - SAE_MANIFOLD_ARMIJO_C1 * trial_step * directional_decrease;
                if trial_total.is_finite() && trial_total <= armijo_bound {
                    accepted_loss = Some(trial_loss);
                    break;
                }
                trial_step *= 0.5;
            }
            match accepted_loss {
                Some(loss) => {
                    last_loss = loss;
                }
                None => {
                    // Restore the pre-step β before stopping. ρ is engine-owned
                    // and held fixed across the streaming inner solve.
                    self.set_flat_beta(beta0.view())?;
                    last_loss =
                        self.streaming_loss(&chunk_ranges, rho, n_total, &mut chunk_init)?;
                    break;
                }
            }
        }
        Ok(last_loss)
    }

    /// Accumulate one chunk system's reduced-Schur contribution into the shared
    /// `(β × β)` accumulator and reduced RHS, consuming the chunk's Kronecker
    /// `htbeta_matvec` procedurally via [`StreamingArrowSchur`].
    ///
    /// The chunk system's β-block already carries the chunk's data-fit
    /// Gauss-Newton curvature `G_chunk ⊗ I_p` (a genuine per-row sum) plus its
    /// minibatch-scaled smoothness / analytic-β penalty. So the contribution
    /// `s_acc_chunk = hbb_chunk − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)` and
    /// `rhs_acc_chunk = +Σ_i H_βt^(i)(H_tt^(i))⁻¹g_t^(i)` sum across a full pass
    /// to `H_ββ − Σ_all_i (…)` and `Σ_all_i (…)` respectively, with the global
    /// β ridge added exactly once by the caller. No per-chunk ridge is applied.
    fn accumulate_chunk_reduced_schur(
        sys: &ArrowSchurSystem,
        ridge_ext_coord: f64,
        options: &ArrowSolveOptions,
        s_acc: &mut Array2<f64>,
        rhs_acc: &mut Array1<f64>,
    ) -> Result<(), String> {
        let k = sys.k;
        let chunk_n = sys.rows.len();
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_n.max(1));
        // `reset_accumulator(0.0)` seeds `s_acc` with the chunk's dense β-block
        // (`hbb_chunk`, including the data-fit GN block and the minibatch-scaled
        // penalty) and no ridge; `accumulate_chunk` then subtracts the per-row
        // reduction. The global β ridge is applied once by the streaming driver.
        streaming
            .reset_accumulator(0.0)
            .map_err(|e| e.to_string())?;
        streaming
            .accumulate_chunk(0, chunk_n, ridge_ext_coord, options.mode)
            .map_err(|e| e.to_string())?;
        let (contrib_s, contrib_rhs) = streaming.take_accumulators();
        for i in 0..k {
            rhs_acc[i] += contrib_rhs[i];
            for j in 0..k {
                s_acc[[i, j]] += contrib_s[[i, j]];
            }
        }
        Ok(())
    }

    /// Streaming total loss: sum of the minibatch-scaled per-chunk losses at the
    /// current β, re-materializing each chunk from a fresh re-seed via
    /// `chunk_init`. The β-penalty terms are scaled by the chunk fraction so the
    /// global smoothness penalty is counted once across the pass.
    fn streaming_loss<F>(
        &self,
        chunk_ranges: &[(usize, usize)],
        rho: &SaeManifoldRho,
        n_total: usize,
        chunk_init: &mut F,
    ) -> Result<SaeManifoldLoss, String>
    where
        F: FnMut(usize, usize) -> Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>), String>,
    {
        let mut data_fit = 0.0_f64;
        let mut assignment_sparsity = 0.0_f64;
        let mut smoothness = 0.0_f64;
        let mut ard = 0.0_f64;
        for &(start, end) in chunk_ranges {
            let n_chunk = end - start;
            let penalty_scale = n_chunk as f64 / n_total as f64;
            let (logits, coords, z_chunk) = chunk_init(start, end)?;
            let chunk = self.materialize_chunk(logits, coords)?;
            let loss = chunk.loss_scaled(z_chunk.view(), rho, penalty_scale)?;
            data_fit += loss.data_fit;
            assignment_sparsity += loss.assignment_sparsity;
            smoothness += loss.smoothness;
            ard += loss.ard;
        }
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
        })
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
        // ρ is owned by the outer engine and held fixed across this single
        // external-basis-refresh Newton step; no in-loop ARD update.
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

/// Outer REML objective for the SAE-manifold term.
///
/// Routes the SAE's smoothing hyperparameters ρ
/// (`log_lambda_sparse`, `log_lambda_smooth`, per-atom/axis `log_ard`)
/// through the *one* generic [`OuterObjective`] engine + cascade that the
/// main GAM REML path uses, instead of the SAE's deleted forked
/// `update_ard_reml` fixed-point rule. Each outer eval runs the inner
/// `(t, β)` arrow-Schur Newton solve at the engine's current ρ and returns
/// the true REML criterion (see [`SaeManifoldTerm::reml_criterion`]).
///
/// The SAE's outer coordinates ρ are all penalty-like / τ (precisions and
/// log-smoothing-strengths), so `psi_dim = 0`: there are no design-moving
/// (ψ) coordinates. No analytic outer gradient/Hessian is exposed yet
/// (task v2 wires the selected-inverse block-trace ρ-gradient), so this
/// is a cost-only objective and the engine routes it to a derivative-free /
/// finite-difference outer strategy per the planner.
pub struct SaeManifoldOuterObjective {
    term: SaeManifoldTerm,
    /// Pristine term to restore from on `reset` (multi-start baseline).
    baseline_term: SaeManifoldTerm,
    target: Array2<f64>,
    registry: Option<AnalyticPenaltyRegistry>,
    /// ρ template carrying the per-atom ARD dims; `from_flat` reads its
    /// layout. Updated to each evaluated ρ so `into_fitted` can report the
    /// last ρ the engine settled on.
    current_rho: SaeManifoldRho,
    /// Pristine ρ to restore from on `reset`.
    baseline_rho: SaeManifoldRho,
    inner_max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    /// Last inner loss breakdown observed (for `into_fitted`).
    last_loss: Option<SaeManifoldLoss>,
    /// Optional warm-start β slot. When the cache / continuation walk seeds a
    /// β, the next inner solve opens from it instead of cold.
    seeded_beta: Option<Array1<f64>>,
}

impl SaeManifoldOuterObjective {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        term: SaeManifoldTerm,
        target: Array2<f64>,
        registry: Option<AnalyticPenaltyRegistry>,
        init_rho: SaeManifoldRho,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Self {
        let baseline_term = term.clone();
        let baseline_rho = init_rho.clone();
        Self {
            term,
            baseline_term,
            target,
            registry,
            current_rho: init_rho,
            baseline_rho,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            last_loss: None,
            seeded_beta: None,
        }
    }

    /// Consume the objective, returning the inner-fitted term, the last ρ the
    /// engine evaluated, and the inner loss breakdown at that ρ.
    pub fn into_fitted(self) -> (SaeManifoldTerm, SaeManifoldRho, SaeManifoldLoss) {
        let loss = self.last_loss.unwrap_or_else(|| SaeManifoldLoss {
            data_fit: 0.0,
            assignment_sparsity: 0.0,
            smoothness: 0.0,
            ard: 0.0,
        });
        (self.term, self.current_rho, loss)
    }

    /// Shared cost path: evaluate the REML criterion at `rho_flat`, updating
    /// the cached ρ / loss and (optionally) priming the inner solve from a
    /// seeded β. Returns `(cost, β̂)`.
    fn evaluate(&mut self, rho_flat: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), String> {
        let rho = self.baseline_rho.from_flat(rho_flat);
        if let Some(beta) = self.seeded_beta.take() {
            // Warm-start the inner decoder coefficients before the solve.
            if beta.len() == self.term.beta_dim() {
                self.term.set_flat_beta(beta.view())?;
            }
        }
        let (cost, loss) = self.term.reml_criterion(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        self.current_rho = rho;
        self.last_loss = Some(loss);
        let beta_hat = self.term.flatten_beta();
        Ok((cost, beta_hat))
    }

    /// Fellner-Schall / Mackay multiplicative fixed-point step on ρ at
    /// `rho_flat`. Runs the inner `(t, β)` solve to convergence at fixed ρ
    /// (sharing the single Direct factor with the REML criterion), then
    /// returns `(cost, additive-log-steps, β̂)`.
    ///
    /// All ρ coords are log-quantities, so the engine's additive step
    /// `rho_new = rho + step` IS the multiplicative FS update. Per coord:
    /// - ARD axis (k,j): `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))`,
    ///   `step = ln α_new − log_ard[k][j]`. The `tr_kj(H⁻¹)` posterior
    ///   variance (from the selected-inverse latent diagonal) is exactly the
    ///   term the deleted `α=n/‖t‖²` rule dropped, so α cannot collapse on a
    ///   degenerate axis: as `‖t‖²→0`, `tr_kj(H⁻¹)→1/α` bounds the
    ///   denominator and the fixed point has a finite root.
    /// - λ_smooth: `λ_new = [p·Σ_k rank S_k − tr(S_β⁻¹ M)] / βᵀ(⊕S_k⊗I_p)β`
    ///   (Wood-Fasiolo EFS), `step = ln λ_new − log_lambda_smooth`.
    /// - λ_sparse: 0.0 — the assignment-sparsity priors (softmax entropy,
    ///   gated L1, IBP) are non-quadratic, so no Gaussian-logdet FS fixed
    ///   point exists; it stays cost-driven (the cascade still moves it via
    ///   the cost path when EFS is not the active lane for that coord).
    fn efs_step(&mut self, rho_flat: ArrayView1<'_, f64>) -> Result<EfsEval, String> {
        let rho = self.baseline_rho.from_flat(rho_flat);
        if let Some(beta) = self.seeded_beta.take()
            && beta.len() == self.term.beta_dim()
        {
            self.term.set_flat_beta(beta.view())?;
        }
        let (cost, loss, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        self.current_rho = rho.clone();
        self.last_loss = Some(loss);

        let n_obs = self.term.n_obs() as f64;
        let sumsq = self.term.ard_coord_sumsq();
        let traces = self
            .term
            .ard_inverse_traces(&cache)
            .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: ARD traces: {e}"))?;

        // Build the flat step vector in `to_flat` layout:
        // [0]=log_lambda_sparse, [1]=log_lambda_smooth, then per-atom axes.
        let n_params = rho.to_flat().len();
        let mut steps = vec![0.0_f64; n_params];

        // λ_sparse (index 0): non-quadratic prior → no FS fixed point. Step 0.
        steps[0] = 0.0;

        // λ_smooth (index 1): Wood-Fasiolo EFS multiplicative update.
        let lambda_smooth = rho.lambda_smooth();
        let p_out = self.term.output_dim() as f64;
        let mut smooth_rank_total = 0usize;
        for atom in &self.term.atoms {
            smooth_rank_total += SaeManifoldTerm::symmetric_rank(&atom.smooth_penalty)?;
        }
        let rank_total = p_out * (smooth_rank_total as f64);
        let quad = self.term.decoder_smoothness_quadratic_form();
        let eff_dof = self
            .term
            .decoder_smoothness_effective_dof(&cache, lambda_smooth)
            .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: smooth dof: {e}"))?;
        // λ_new = (penalty_rank − effective_dof) / penalty_energy. The
        // numerator is the unpenalised-direction count; guard the FS ratio
        // against a vanishing penalty energy or a non-positive numerator
        // (which can occur transiently far from the optimum) by holding
        // λ_smooth fixed (step 0) — the cost path still moves it then.
        if quad > 0.0 && rank_total - eff_dof > 0.0 && lambda_smooth > 0.0 {
            let lambda_new = (rank_total - eff_dof) / quad;
            if lambda_new.is_finite() && lambda_new > 0.0 {
                steps[1] = lambda_new.ln() - rho.log_lambda_smooth;
            }
        }

        // ARD axes (indices 2..): Mackay fixed point with posterior variance.
        let mut cursor = 2usize;
        for (k, axis_logard) in rho.log_ard.iter().enumerate() {
            let d = axis_logard.len();
            for j in 0..d {
                let denom = sumsq[k][j] + traces[k][j];
                if denom > 0.0 {
                    let alpha_new = n_obs / denom;
                    if alpha_new.is_finite() && alpha_new > 0.0 {
                        steps[cursor + j] = alpha_new.ln() - axis_logard[j];
                    }
                }
            }
            cursor += d;
        }

        let beta_hat = self.term.flatten_beta();
        Ok(EfsEval {
            cost,
            steps,
            beta: Some(beta_hat),
            psi_gradient: None,
            psi_indices: None,
            inner_hessian_scale: None,
        })
    }
}

impl OuterObjective for SaeManifoldOuterObjective {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.baseline_rho.to_flat().len(),
            // ρ are all penalty-like / τ coordinates: precisions and
            // log-smoothing strengths. No design-moving ψ coordinates.
            psi_dim: 0,
            // EFS fixed-point lane is the right driver for these penalty-like
            // coords: the multiplicative Fellner-Schall/Mackay step is O(1)
            // selected-inverse trace per outer iter, vs the cost-only path's
            // O(K³) dense Schur per cost eval × many derivative-free evals —
            // intractable at biobank K. `eval_efs` implements it.
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.evaluate(rho.view())
            .map(|(cost, _beta)| cost)
            .map_err(EstimationError::RemlOptimizationFailed)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let n_params = self.baseline_rho.to_flat().len();
        let (cost, beta_hat) = self
            .evaluate(rho.view())
            .map_err(EstimationError::RemlOptimizationFailed)?;
        Ok(OuterEval {
            cost,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
            inner_beta_hint: Some(beta_hat),
        })
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        self.efs_step(rho.view())
            .map_err(EstimationError::RemlOptimizationFailed)
    }

    fn reset(&mut self) {
        self.term = self.baseline_term.clone();
        self.current_rho = self.baseline_rho.clone();
        self.last_loss = None;
        self.seeded_beta = None;
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.len() != self.term.beta_dim() {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "SaeManifoldOuterObjective::seed_inner_state: β length {} != decoder dim {}",
                beta.len(),
                self.term.beta_dim()
            )));
        }
        self.seeded_beta = Some(beta.clone());
        Ok(SeedOutcome::Installed)
    }
}

fn sae_manifold_newton_directional_decrease(
    sys: &ArrowSchurSystem,
    delta_ext_coord: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
) -> f64 {
    // delta_ext_coord has variable-stride layout for heterogeneous systems.
    assert_eq!(delta_ext_coord.len(), sys.row_offsets[sys.rows.len()]);
    assert_eq!(delta_beta.len(), sys.k);
    let mut gradient_dot_step = 0.0;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let row_base = sys.row_offsets[row_idx];
        let di = sys.row_dims[row_idx];
        for axis in 0..di {
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
        out[i] = crate::linalg::utils::stable_logistic(logits[i] / temperature) * prior[i];
    }
    out
}

/// IBP-MAP concrete relaxation activations together with the diagonal Jacobian
/// `∂z_k/∂l_k`, for the torch autograd `Function` to consume so that torch's
/// IBP-Gumbel forward applies the same stick-breaking prior `π_k` and
/// temperature scaling as the Rust closed-form path
/// (`SaeAssignment::try_assignments_row` → [`ibp_map_row`]).
///
/// With `z_k = σ(l_k/τ) · π_k` the per-atom derivative is
/// `∂z_k/∂l_k = σ(l_k/τ) (1 − σ(l_k/τ)) · π_k / τ`. The map is diagonal in `k`
/// (each activation depends only on its own logit), so the Jacobian is returned
/// as the per-atom diagonal vector.
#[must_use]
pub fn ibp_map_row_value_grad(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    alpha: f64,
) -> (Array1<f64>, Array1<f64>) {
    let prior = ibp_stick_breaking_prior(logits.len(), alpha);
    let inv_tau = 1.0 / temperature;
    let mut value = Array1::<f64>::zeros(logits.len());
    let mut grad = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        let sig = crate::linalg::utils::stable_logistic(logits[i] * inv_tau);
        value[i] = sig * prior[i];
        grad[i] = sig * (1.0 - sig) * inv_tau * prior[i];
    }
    (value, grad)
}

fn jumprelu_row(logits: ArrayView1<'_, f64>, temperature: f64, threshold: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        if logits[i] > threshold {
            out[i] = crate::linalg::utils::stable_logistic(logits[i] / temperature);
        }
    }
    out
}

/// Fill the single compact logit-JVP row `jac_compact[[j, ·]]` for active atom
/// `k`, using the per-mode assignment sensitivity `da_k/dl_k` contracted into
/// the decoded / fitted-corrected output direction. This is the active-set
/// analogue of [`fill_assignment_logit_jvp_rows`]: it reproduces that
/// function's diagonal logit row exactly for the atom `k`, but writes into the
/// compact position `j` of a heterogeneous-`q` row block instead of the dense
/// full-`K` Jacobian. `fitted` is the row's *active-set* reconstruction so the
/// softmax cross term `(decoded_k − fitted)` is consistent with the curvature
/// the compact block carries.
#[allow(clippy::too_many_arguments)]
fn fill_active_atom_logit_jvp(
    mode: AssignmentMode,
    k: usize,
    logit_k: f64,
    a_k: f64,
    decoded_k: ArrayView1<'_, f64>,
    fitted: ArrayView1<'_, f64>,
    ibp_prior: Option<&[f64]>,
    jac_compact: &mut Array2<f64>,
    j: usize,
) {
    let p = fitted.len();
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            // da_k/dl_k contracted: a_k (decoded_k − fitted) / τ.
            let inv_tau = 1.0 / temperature;
            for out_col in 0..p {
                jac_compact[[j, out_col]] = a_k * (decoded_k[out_col] - fitted[out_col]) * inv_tau;
            }
        }
        AssignmentMode::IBPMap { temperature, .. } => {
            // z_k = σ(l_k/τ)·π_k ⇒ dz_k/dl_k = a_k(π_k − a_k)/(π_k τ) · π_k form
            // (matches `fill_assignment_logit_jvp_rows`).
            let inv_tau = 1.0 / temperature;
            let prior =
                ibp_prior.expect("fill_active_atom_logit_jvp: IBPMap requires precomputed prior");
            let pi_k = prior[k];
            let sig = if pi_k > 0.0 { a_k / pi_k } else { 0.0 };
            let dz = sig * (1.0 - sig) * inv_tau * pi_k;
            for out_col in 0..p {
                jac_compact[[j, out_col]] = dz * decoded_k[out_col];
            }
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // Hard-gate STE: sigmoid derivative, only above threshold.
            if logit_k <= threshold {
                return;
            }
            let inv_tau = 1.0 / temperature;
            let activation = crate::linalg::utils::stable_logistic(logit_k * inv_tau);
            let da = activation * (1.0 - activation) * inv_tau;
            for out_col in 0..p {
                jac_compact[[j, out_col]] = da * decoded_k[out_col];
            }
        }
    }
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
                let activation = crate::linalg::utils::stable_logistic(logits[logit_col] * inv_tau);
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
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)
            .expect("assignment logits must be finite");
    }
    let target = flat_logits(assignment.logits.view());
    // Proper Gaussian identifiability prior ½·κ·‖logit‖² over every logit,
    // pinning the shift-invariant softmax gauge (and the wholly-flat K=1
    // logit) so the assembled H_tt is PD. See
    // `SAE_LOGIT_IDENTIFIABILITY_PRECISION`. Applied for all K and all modes;
    // it is the only assignment-prior term when `k_atoms() == 1`.
    let identifiability: f64 =
        0.5 * SAE_LOGIT_IDENTIFIABILITY_PRECISION * target.iter().map(|&l| l * l).sum::<f64>();
    if assignment.k_atoms() == 1 {
        return identifiability;
    }
    let sparsity_value = match assignment.mode {
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
                    acc += crate::linalg::utils::stable_logistic(logit / temperature);
                }
            }
            sparsity_strength * acc
        }
    };
    sparsity_value + identifiability
}

fn assignment_prior_grad_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    // Proper Gaussian identifiability prior ½·κ·‖logit‖²: gradient `κ·logit`,
    // Hessian diagonal `κ` on every logit. This pins the shift-invariant
    // softmax gauge (and the flat K=1 logit) so the per-row H_tt logit block
    // is PD. See `SAE_LOGIT_IDENTIFIABILITY_PRECISION`. Computed for all K;
    // the sparsity prior (absent at K=1) is added on top below.
    let kappa = SAE_LOGIT_IDENTIFIABILITY_PRECISION;
    let mut grad = Array1::<f64>::zeros(target.len());
    let mut diag = Array1::<f64>::zeros(target.len());
    for idx in 0..target.len() {
        grad[idx] = kappa * target[idx];
        diag[idx] = kappa;
    }
    if assignment.k_atoms() == 1 {
        return Ok((grad, diag));
    }
    let (sparsity_grad, sparsity_diag) = match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "softmax assignment hessian diag unavailable".to_string())?;
            (g, d)
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
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "IBP assignment hessian diag unavailable".to_string())?;
            (g, d)
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            let sparsity_strength = rho.log_lambda_sparse.exp();
            let inv_tau = 1.0 / temperature;
            let inv_tau2 = inv_tau * inv_tau;
            let mut g = Array1::<f64>::zeros(target.len());
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                let logit = target[idx];
                if logit <= threshold {
                    continue;
                }
                let activation = crate::linalg::utils::stable_logistic(logit * inv_tau);
                let slope = activation * (1.0 - activation);
                g[idx] = sparsity_strength * slope * inv_tau;
                d[idx] = sparsity_strength * slope * slope * inv_tau2;
            }
            (g, d)
        }
    };
    grad += &sparsity_grad;
    diag += &sparsity_diag;
    Ok((grad, diag))
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
            | AnalyticPenaltyKind::BlockOrthogonality(_)
            | AnalyticPenaltyKind::Isometry(_)
    )
}

/// The JSON descriptor `kind` strings for the SAE row-block analytic penalties
/// this build supports (i.e. those `sae_penalty_is_row_block_supported`
/// accepts). Co-located with that matcher so the two cannot drift. The FFI
/// `build_info` advertises this list so the Python wrapper can detect a stale
/// extension that predates a given penalty and raise a clear `NotImplementedError`
/// instead of forwarding a descriptor the binary will reject with a cryptic
/// Schur error (issue #338).
pub fn sae_row_block_penalty_kinds() -> &'static [&'static str] {
    &[
        "ard",
        "top_k_activation",
        "jumprelu",
        "sparsity",
        "softmax_assignment_sparsity",
        "ibp_assignment",
        "row_precision_prior",
        "parametric_row_precision_prior",
        "scad_mcp",
        "block_orthogonality",
        "isometry",
    ]
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

/// Build the per-row Jacobian `J` and Hessian `H` of the decoded output
/// `Z_n = Phi_n B` with respect to the latent coordinates `t_n` of a single
/// SAE atom and install them on the supplied [`IsometryPenalty`].
///
/// Layout follows the convention used by [`IsometryPenalty::grad_target`] and
/// friends:
///
/// * `J ∈ ℝ^{n_obs × (p · d)}`, flattened as `J[n, i*d + a]` —
///   `J[n, i, a] = ∂Z_{n,i} / ∂t_{n,a} = Σ_m dPhi[n, m, a] · B[m, i]`.
/// * `H ∈ ℝ^{n_obs × (p · d · d)}`, flattened as `H[n, (i*d + a)*d + c]` —
///   `H[n, i, a, c] = ∂J[n, i, a] / ∂t_{n, c} = Σ_m d²Phi[n, m, a, c] · B[m, i]`.
///
/// Returns `Ok(true)` when both caches were installed (i.e. the atom was
/// built via [`SaeManifoldAtom::with_basis_second_jet`], so its
/// `basis_second_jet` slot holds a [`SaeBasisSecondJet`] implementation
/// that supplies the analytic Hessian). Returns `Ok(false)` when only the
/// base [`SaeBasisEvaluator`] is installed (no second jet available) — in
/// that case only the first-jet `jacobian_cache` is installed and the
/// penalty's `has_jacobian_second_source` check still has a chance to
/// succeed via a pre-supplied `duchon_radial_source`. Returns `Err` on
/// shape mismatches (which would indicate a buggy evaluator) or when the
/// second-jet implementation itself fails (e.g. wrong latent dimension).
///
/// This entry point takes `&IsometryPenalty` rather than `&mut` because the
/// caches are interior-mutable (see [`IsometryPenalty::refresh_caches`]).
pub fn refresh_isometry_caches_from_atom(
    penalty: &IsometryPenalty,
    atom: &SaeManifoldAtom,
    coords: ArrayView2<'_, f64>,
) -> Result<bool, String> {
    let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
        format!(
            "refresh_isometry_caches_from_atom: atom {} has no basis evaluator",
            atom.name
        )
    })?;
    let (_phi, jet) = evaluator.evaluate(coords)?;

    let n_obs = coords.nrows();
    let d = atom.latent_dim;
    let m = atom.basis_size();
    let p = atom.decoder_coefficients.ncols();
    if penalty.p_out != p {
        return Err(format!(
            "refresh_isometry_caches_from_atom: penalty.p_out={} but atom.decoder.cols={p}",
            penalty.p_out
        ));
    }
    if jet.dim() != (n_obs, m, d) {
        return Err(format!(
            "refresh_isometry_caches_from_atom: evaluator first jet has shape {:?}, expected ({n_obs}, {m}, {d})",
            jet.dim()
        ));
    }

    // J[n, i*d + a] = Σ_m dPhi[n, m, a] · B[m, i].
    let b = &atom.decoder_coefficients;
    let mut jac = Array2::<f64>::zeros((n_obs, p * d));
    for n in 0..n_obs {
        for i in 0..p {
            for a in 0..d {
                let mut acc = 0.0;
                for mm in 0..m {
                    acc += jet[[n, mm, a]] * b[[mm, i]];
                }
                jac[[n, i * d + a]] = acc;
            }
        }
    }

    // The second jet is sourced from the optional `basis_second_jet`
    // slot. The trait split (`SaeBasisEvaluator` vs `SaeBasisSecondJet`)
    // encodes "no closed-form Hessian" as trait absence: when the atom
    // was built with `with_basis_evaluator` (base trait only) the slot
    // is `None` and the `H` cache is not installed. When the atom was
    // built with `with_basis_second_jet` the slot holds the same Arc
    // upcast to the supertrait, and `second_jet` returns the analytic
    // Hessian here.
    let jac2_opt = if let Some(second_eval) = atom.basis_second_jet.as_ref() {
        let hess = second_eval.second_jet(coords)?;
        if hess.dim() != (n_obs, m, d, d) {
            return Err(format!(
                "refresh_isometry_caches_from_atom: evaluator second jet has shape {:?}, expected ({n_obs}, {m}, {d}, {d})",
                hess.dim()
            ));
        }
        let mut jac2 = Array2::<f64>::zeros((n_obs, p * d * d));
        for n in 0..n_obs {
            for i in 0..p {
                for a in 0..d {
                    for c in 0..d {
                        let mut acc = 0.0;
                        for mm in 0..m {
                            acc += hess[[n, mm, a, c]] * b[[mm, i]];
                        }
                        jac2[[n, (i * d + a) * d + c]] = acc;
                    }
                }
            }
        }
        Some(Arc::new(jac2))
    } else {
        None
    };

    let installed = jac2_opt.is_some();
    penalty.refresh_caches(Some(Arc::new(jac)), jac2_opt);
    Ok(installed)
}

/// Walk an [`AnalyticPenaltyRegistry`] and refresh every Isometry penalty
/// against the SAE atom it owns. The alignment rule is positional within each
/// `(latent_dim, p_out)` signature: the penalty's `target.latent_dim` must
/// equal the atom's `latent_dim` AND the penalty's `p_out` must equal the
/// atom's decoder column count `p`. Multi-atom configurations install one
/// isometry penalty per atom, so the *k*-th isometry penalty matching a given
/// signature is paired with the *k*-th atom matching that same signature. This
/// reduces to the unambiguous single-atom/single-penalty case wired by
/// `solver/workflow.rs`, and never collapses multiple penalties onto the first
/// matching atom (which would leave every later atom's coords un-refreshed).
///
/// Returns the number of penalties that got both caches populated (i.e. the
/// number of atoms whose `basis_second_jet` slot holds a
/// [`SaeBasisSecondJet`] implementation supplying the analytic Hessian).
pub fn refresh_isometry_caches_from_term(
    registry: &AnalyticPenaltyRegistry,
    term: &SaeManifoldTerm,
    coords_per_atom: &[Array2<f64>],
) -> Result<usize, String> {
    if coords_per_atom.len() != term.atoms.len() {
        return Err(format!(
            "refresh_isometry_caches_from_term: coords_per_atom length {} != number of atoms {}",
            coords_per_atom.len(),
            term.atoms.len()
        ));
    }
    let mut refreshed_with_second = 0usize;
    // Per-signature cursor: how many atoms matching a given (latent_dim, p_out)
    // have already been consumed by earlier isometry penalties. Pairing the
    // k-th penalty of a signature with the k-th atom of that signature gives a
    // stable one-to-one mapping for multi-atom configs.
    let mut consumed_per_signature: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for entry in registry.penalties.iter() {
        let AnalyticPenaltyKind::Isometry(p) = entry else {
            continue;
        };
        let Some(p_latent_dim) = p.target.latent_dim else {
            continue;
        };
        let signature = (p_latent_dim, p.p_out);
        let already_consumed = consumed_per_signature.entry(signature).or_insert(0);
        // Advance to the (already_consumed)-th atom matching this signature.
        let mut seen = 0usize;
        let mut paired: Option<usize> = None;
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let matches = atom.latent_dim == p_latent_dim
                && atom.decoder_coefficients.ncols() == p.p_out
                && atom.basis_evaluator.is_some();
            if !matches {
                continue;
            }
            if seen == *already_consumed {
                paired = Some(atom_idx);
                break;
            }
            seen += 1;
        }
        let Some(atom_idx) = paired else {
            continue;
        };
        *already_consumed += 1;
        let atom = &term.atoms[atom_idx];
        let coords = coords_per_atom[atom_idx].view();
        if refresh_isometry_caches_from_atom(p, atom, coords)? {
            refreshed_with_second += 1;
        }
    }
    Ok(refreshed_with_second)
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

    /// `snapshot_mutable_state` / `restore_mutable_state` (the in-place
    /// line-search save/restore that replaced the per-halving full
    /// `self.clone()`) must restore exactly the state an `apply_newton_step`
    /// trial perturbs: decoder coefficients, the `refresh_basis`-rebuilt
    /// basis evaluations, assignment logits, and latent coordinates. Pins
    /// item-1 of the SAE hot-path CPU-perf refactor.
    #[test]
    fn snapshot_restore_round_trips_mutated_state() {
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

        // Capture pre-step state, then apply a non-trivial Newton step that
        // refreshes the basis (changing basis_values/jacobian, decoder
        // coefficients, logits, and coords).
        let snapshot = term.snapshot_mutable_state();
        let pre_basis = term.atoms[0].basis_values.clone();
        let pre_jet = term.atoms[0].basis_jacobian.clone();
        let pre_decoder = term.atoms[0].decoder_coefficients.clone();
        let pre_logits = term.assignment.logits.clone();
        let pre_coords = term.assignment.coords[0].as_matrix();

        let q = term.assignment.row_block_dim();
        let beta_dim = term.beta_dim();
        let delta_ext = Array1::<f64>::from_elem(4 * q, 0.3);
        let delta_beta = Array1::<f64>::from_elem(beta_dim, -0.4);
        term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
            .unwrap();

        // Something must actually have changed, else the test is vacuous.
        assert!(
            (&term.atoms[0].basis_values - &pre_basis)
                .mapv(f64::abs)
                .sum()
                > 1e-9
                || (&term.atoms[0].decoder_coefficients - &pre_decoder)
                    .mapv(f64::abs)
                    .sum()
                    > 1e-9,
            "apply_newton_step did not perturb the snapshotted state"
        );

        // Restore and confirm every snapshotted field matches the pre-step
        // values bit-for-bit.
        term.restore_mutable_state(&snapshot);
        assert_eq!(term.atoms[0].basis_values, pre_basis);
        assert_eq!(term.atoms[0].basis_jacobian, pre_jet);
        assert_eq!(term.atoms[0].decoder_coefficients, pre_decoder);
        assert_eq!(term.assignment.logits, pre_logits);
        assert_eq!(term.assignment.coords[0].as_matrix(), pre_coords);
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
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
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
        // Use penalty_op to include all H_ββ contributions (GN + smoothness)
        // rather than reading sys.hbb directly, which no longer holds the
        // smoothness term after the #296 BetaPenaltyOp migration.
        let mut hbb_delta = Array1::<f64>::zeros(delta.len());
        {
            let op = sys.effective_penalty_op();
            let d_slice = delta.as_slice().expect("delta is contiguous");
            let hd_slice = hbb_delta.as_slice_mut().expect("hbb_delta is contiguous");
            op.matvec(d_slice, hd_slice);
        }
        let quadratic = 0.5 * delta.dot(&hbb_delta);
        let predicted = linear + quadratic;
        let error = (actual - predicted).abs();
        assert!(
            error <= 1.0e-4,
            "actual={actual:.12e}, predicted={predicted:.12e}, error={error:.12e}"
        );
    }

    /// `SaeRowLayout::from_dense_weights` must keep, per row, the
    /// top-`k_active_cap` atoms above the magnitude cutoff (always at least
    /// one), with compact coord starts that reproduce the `expand_row`
    /// round-trip back to full-q positions.
    #[test]
    fn sae_row_layout_from_dense_weights_top_k_and_cutoff() {
        // 3 atoms, coord dims [2, 1, 2] ⇒ full q = 3 + 5 = 8.
        let coord_dims = vec![2usize, 1, 2];
        let coord_offsets_full = vec![3usize, 5, 6];
        let assignments = vec![
            // Row 0: weights [0.7, 0.01, 0.29]; cutoff 0.05, cap 2 ⇒ {0, 2}.
            Array1::from_vec(vec![0.7, 0.01, 0.29]),
            // Row 1: weights [0.001, 0.002, 0.0005]; all below cutoff ⇒ keep
            // single largest-magnitude atom {1}.
            Array1::from_vec(vec![0.001, 0.002, 0.0005]),
        ];
        let layout =
            SaeRowLayout::from_dense_weights(&assignments, 2, 0.05, coord_dims, coord_offsets_full);
        assert_eq!(layout.active_atoms[0], vec![0, 2]);
        assert_eq!(layout.active_atoms[1], vec![1]);
        // Row 0 compact dim = |{0,2}| + d_0 + d_2 = 2 + 2 + 2 = 6.
        assert_eq!(layout.row_q_active(0), 6);
        // Row 1 compact dim = 1 + d_1 = 1 + 1 = 2.
        assert_eq!(layout.row_q_active(1), 2);
        // expand_row round-trip for row 0: compact [logit0, logit2, t0_0,
        // t0_1, t2_0, t2_1] → full-q with zeros for inactive atom 1.
        let compact = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut full = vec![0.0_f64; 8];
        layout.expand_row(0, &compact, &mut full);
        // logits: full[0] = atom0 logit, full[2] = atom2 logit, full[1] = 0.
        assert_eq!(full[0], 1.0);
        assert_eq!(full[1], 0.0);
        assert_eq!(full[2], 2.0);
        // coords: atom0 at offset 3 (d=2), atom2 at offset 6 (d=2); atom1
        // (offset 5, d=1) is inactive ⇒ zero.
        assert_eq!(full[3], 3.0);
        assert_eq!(full[4], 4.0);
        assert_eq!(full[5], 0.0);
        assert_eq!(full[6], 5.0);
        assert_eq!(full[7], 6.0);
    }

    /// MechanismSparsityPenalty must reach the SAE arrow-Schur system's
    /// `gb` (beta-tier gradient) when its target slice is shaped to match a
    /// single-atom decoder block (M, p_out). The group lasso over rows of
    /// that (M, p_out) matrix translates to a non-zero gradient on every
    /// (basis_row, feature) entry whose corresponding decoder coefficient
    /// is non-zero, and the FFI-side `"beta"` latent block is what makes
    /// the descriptor builder see exactly that target shape.
    #[test]
    fn sae_mechsparsity_beta_block_routes_through_arrow_schur_gb() {
        let coords = array![[0.10], [0.35], [0.80]];
        let (phi, jet) = periodic_basis(&coords);
        // Decoder shape: (M=3 basis × p=4 features); flatten_beta lays out
        // [basis_col * p + feature] which is exactly the (M, p) row-major
        // shape MechSparsity targets.
        let decoder = array![
            [0.7, -0.2, 0.05, 0.4],
            [-0.5, 0.6, -0.1, 0.3],
            [0.2, 0.0, -0.4, -0.6],
        ];
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder.clone(),
            Array2::<f64>::eye(3),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

        // Two groups partition the 4 features: {0,1} and {2,3}. Each row
        // of the decoder block has non-zero entries in both groups, so the
        // group-norm denominator is finite and every column-feature in a
        // non-trivially-loaded group sees a positive penalty gradient.
        let m = 3usize;
        let p = 4usize;
        let slice = PsiSlice::full(m * p, Some(m));
        let penalty = MechanismSparsityPenalty::new(
            slice,
            vec![vec![0, 1], vec![2, 3]],
            1.0,
            1.0e-6,
            (term.n_obs()) as f64,
            false,
        )
        .unwrap();
        let mut registry = AnalyticPenaltyRegistry::new();
        registry.push(AnalyticPenaltyKind::MechanismSparsity(Arc::new(penalty)));

        let target = array![
            [0.20, 0.10, -0.05, 0.25],
            [-0.10, 0.30, 0.15, -0.20],
            [0.45, -0.05, 0.10, 0.30],
        ];
        let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, Some(&registry))
            .unwrap();

        assert_eq!(sys.gb.len(), m * p, "gb should match flatten_beta length");
        let mut absmax = 0.0_f64;
        for v in sys.gb.iter().copied() {
            assert!(v.is_finite());
            if v.abs() > absmax {
                absmax = v.abs();
            }
        }
        assert!(
            absmax > 1.0e-6,
            "MechSparsity must inject a non-trivial gradient into the SAE arrow-Schur gb; absmax={absmax:.3e}"
        );
        // Closed-form check: the row=1 column=0 entry of grad is
        //   w / sqrt(|G|) * b[1,0] / ||b[1, group={0,1}]||
        // where group {0,1} has size 2 → factor sqrt(2). With unit weight
        // and tiny eps, the expected magnitude matches Penalty::grad_target.
        let beta = term.flatten_beta();
        let expected = {
            // ||b[1, {0,1}]|| ≈ sqrt(0.5² + 0.6²) = sqrt(0.61)
            let s = (0.5_f64.powi(2) + 0.6_f64.powi(2) + 1.0e-12).sqrt();
            (2.0_f64).sqrt() * (-0.5_f64) / s
        };
        let observed = sys.gb[1 * p + 0];
        assert!(
            (observed - expected).abs() <= 1.0e-6,
            "expected MechSparsity gb entry at (basis=1, feat=0) ≈ {expected:.6e}, got {observed:.6e} (beta entry = {})",
            beta[1 * p + 0]
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

    /// Parity guard for the sphere chart: the shared engine
    /// [`sphere_chart_basis_jet`] is the single source of derivative truth used
    /// by both the core SAE path ([`SphereChartEvaluator::evaluate`]) and the
    /// PyFFI `sphere_chart_basis_with_jet` helper, which now routes through the
    /// exact same function. This pins the behavior that previously drifted —
    /// the saturated-latitude `chain_lat` gating — at both saturated and
    /// non-saturated latitudes, so a future divergence would fail here.
    #[test]
    fn sphere_chart_basis_jet_is_single_source_of_truth() {
        // A mix of interior and clamp-saturated (|lat| >= π/2) latitudes,
        // including the exact boundary, which the gate treats as saturated.
        let coords = array![
            [-1.2, -2.4],                         // interior
            [0.35, 0.9],                          // interior
            [std::f64::consts::FRAC_PI_2, 0.4],   // upper boundary (saturated)
            [-std::f64::consts::FRAC_PI_2, -1.1], // lower boundary (saturated)
            [2.3, 0.7],                           // beyond upper clamp (saturated)
            [-3.0, 1.9],                          // beyond lower clamp (saturated)
        ];

        // The core evaluator adapter must be bit-identical to the shared engine
        // — they are the same code path, so any difference is a regression in
        // the thin adapter rather than a tolerance question.
        let (engine_phi, engine_jet) = sphere_chart_basis_jet(coords.view()).unwrap();
        let (adapter_phi, adapter_jet) = SphereChartEvaluator.evaluate(coords.view()).unwrap();
        assert_eq!(engine_phi, adapter_phi);
        assert_eq!(engine_jet, adapter_jet);

        for row in 0..coords.nrows() {
            let raw_lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            let saturated =
                !(raw_lat > -std::f64::consts::FRAC_PI_2 && raw_lat < std::f64::consts::FRAC_PI_2);

            let lat = raw_lat.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            let x = clat * clon;
            let y = clat * slon;
            let z = slat;

            // Basis is always the clamped embedding, gating or not.
            assert!((engine_phi[[row, 0]] - 1.0).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 1]] - x).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 2]] - y).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 3]] - z).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 4]] - x * y).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 5]] - y * z).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 6]] - x * z).abs() <= 1.0e-12);

            // Longitude derivatives never depend on the latitude clamp.
            let dx_dlon = -clat * slon;
            let dy_dlon = clat * clon;
            assert!((engine_jet[[row, 1, 1]] - dx_dlon).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 2, 1]] - dy_dlon).abs() <= 1.0e-12);
            assert_eq!(engine_jet[[row, 3, 1]], 0.0);
            assert!((engine_jet[[row, 4, 1]] - (dx_dlon * y + x * dy_dlon)).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 5, 1]] - dy_dlon * z).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 6, 1]] - dx_dlon * z).abs() <= 1.0e-12);

            // Latitude derivatives are gated to exactly zero in the saturated
            // region (where finite differences also report zero) and equal the
            // ungated analytic values in the interior.
            let chain_lat = if saturated { 0.0 } else { 1.0 };
            let dx_dlat = -slat * clon * chain_lat;
            let dy_dlat = -slat * slon * chain_lat;
            let dz_dlat = clat * chain_lat;
            if saturated {
                assert_eq!(engine_jet[[row, 1, 0]], 0.0);
                assert_eq!(engine_jet[[row, 2, 0]], 0.0);
                assert_eq!(engine_jet[[row, 3, 0]], 0.0);
                assert_eq!(engine_jet[[row, 4, 0]], 0.0);
                assert_eq!(engine_jet[[row, 5, 0]], 0.0);
                assert_eq!(engine_jet[[row, 6, 0]], 0.0);
            }
            assert!((engine_jet[[row, 1, 0]] - dx_dlat).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 2, 0]] - dy_dlat).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 3, 0]] - dz_dlat).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 4, 0]] - (dx_dlat * y + x * dy_dlat)).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 5, 0]] - (dy_dlat * z + y * dz_dlat)).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 6, 0]] - (dx_dlat * z + x * dz_dlat)).abs() <= 1.0e-12);
        }

        // The chart penalty diagonal is also shared with the PyFFI helper.
        assert_eq!(
            SPHERE_CHART_PENALTY_DIAGONAL,
            [1e-8, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
        );
    }

    /// Central-difference oracle for `second_jet`: differentiate the analytic
    /// first jet (which is FD-validated by the test above) coordinate-wise.
    /// Tolerance is `1e-5` per the charter; the FD step is 1e-4 (the sweet
    /// spot before f64 cancellation dominates a centered difference of an
    /// `O(1)` Jacobian).
    fn assert_second_jet_matches_central_difference<E: SaeBasisSecondJet>(
        evaluator: &E,
        coords: Array2<f64>,
        tolerance: f64,
    ) -> Result<(), String> {
        let epsilon = 1.0e-4;
        let second = evaluator.second_jet(coords.view())?;
        let (_phi, jet) = evaluator.evaluate(coords.view())?;
        let (n_rows, n_basis, latent_dim, latent_dim_b) = second.dim();
        assert_eq!(latent_dim, latent_dim_b);
        assert_eq!((n_rows, n_basis, latent_dim), jet.dim());
        for row in 0..n_rows {
            for axis_c in 0..latent_dim {
                let mut plus = coords.clone();
                let mut minus = coords.clone();
                plus[[row, axis_c]] += epsilon;
                minus[[row, axis_c]] -= epsilon;
                let (_, jet_plus) = evaluator.evaluate(plus.view()).unwrap();
                let (_, jet_minus) = evaluator.evaluate(minus.view()).unwrap();
                for basis in 0..n_basis {
                    for axis_a in 0..latent_dim {
                        let fd = (jet_plus[[row, basis, axis_a]] - jet_minus[[row, basis, axis_a]])
                            / (2.0 * epsilon);
                        let analytic = second[[row, basis, axis_a, axis_c]];
                        let error = (analytic - fd).abs();
                        assert!(
                            error <= tolerance,
                            "row={row} basis={basis} axis_a={axis_a} axis_c={axis_c}: \
                             analytic={analytic:.12e}, fd={fd:.12e}, error={error:.12e}, \
                             tol={tolerance:.12e}"
                        );
                    }
                }
            }
        }
        // Hessian symmetry in (axis_a, axis_c).
        for row in 0..n_rows {
            for basis in 0..n_basis {
                for axis_a in 0..latent_dim {
                    for axis_b in 0..latent_dim {
                        let h_ab = second[[row, basis, axis_a, axis_b]];
                        let h_ba = second[[row, basis, axis_b, axis_a]];
                        assert!(
                            (h_ab - h_ba).abs() <= 1.0e-12,
                            "second_jet not symmetric: row={row} basis={basis} \
                             ({axis_a},{axis_b})={h_ab:.6e} vs ({axis_b},{axis_a})={h_ba:.6e}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn isometry_periodic_second_jet_matches_fd() -> Result<(), String> {
        assert_second_jet_matches_central_difference(
            &PeriodicHarmonicEvaluator::new(7).unwrap(),
            array![[-0.37], [0.0], [0.125], [0.41]],
            1.0e-5,
        )?;
        Ok(())
    }

    #[test]
    fn isometry_sphere_second_jet_matches_fd() -> Result<(), String> {
        // Stay inside the interior `(-π/2, π/2)` for lat so the chain factor
        // is active — that is where the Hessian carries information.
        let sphere_coords = array![[-0.7, -1.2], [-0.25, 0.0], [0.35, 0.9], [0.8, 2.1]];
        assert_second_jet_matches_central_difference(&SphereChartEvaluator, sphere_coords, 1.0e-5)?;
        Ok(())
    }

    #[test]
    fn isometry_torus_second_jet_matches_fd() -> Result<(), String> {
        let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
        let evaluator = TorusHarmonicEvaluator::new(2, 3).unwrap();
        assert!(evaluator.basis_size() > 0);
        assert_second_jet_matches_central_difference(&evaluator, torus_coords, 1.0e-5)?;
        Ok(())
    }

    /// Issue #247: the Duchon coordinate evaluator must return a forward design
    /// and a derivative jet with *matching column counts* — the original bug
    /// was a radial-only design paired with a radial+polynomial jet (or vice
    /// versa), which the consumer rejected as a "design/jet column mismatch".
    #[test]
    fn duchon_coordinate_evaluator_phi_and_jet_share_column_count() {
        for (d, centers) in [
            (1usize, array![[-1.0], [-0.4], [0.1], [0.6], [1.2], [1.9]]),
            (
                2usize,
                array![
                    [-1.0, -0.8],
                    [-0.3, 0.4],
                    [0.2, -0.5],
                    [0.7, 0.9],
                    [1.1, -0.2],
                    [1.6, 0.6],
                ],
            ),
        ] {
            let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
            let coords = match d {
                1 => array![[-0.5], [0.0], [0.3], [0.8]],
                _ => array![[-0.5, 0.2], [0.0, -0.3], [0.3, 0.7], [0.8, -0.1]],
            };
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            assert_eq!(
                phi.ncols(),
                jet.shape()[1],
                "Duchon d={d}: Phi has {} columns but jet has {}",
                phi.ncols(),
                jet.shape()[1]
            );
            assert_eq!(jet.shape()[0], coords.nrows());
            assert_eq!(jet.shape()[2], d);
        }
    }

    /// The Duchon evaluator's analytic first jet must equal the finite
    /// difference of its own forward design — i.e. `dPhi/dt` is the true
    /// derivative of `Phi(t)`, with no stray amplification/column mismatch.
    #[test]
    fn duchon_coordinate_evaluator_jacobian_matches_fd() {
        let centers = array![
            [-1.0, -0.8],
            [-0.3, 0.4],
            [0.2, -0.5],
            [0.7, 0.9],
            [1.1, -0.2],
            [1.6, 0.6],
        ];
        let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
        // Keep probe points away from any center so the radial kernel is smooth.
        let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
        assert_jacobian_matches_central_difference(&evaluator, coords, 1.0e-4);
    }

    /// The Duchon evaluator's analytic second jet must match the FD of its
    /// (FD-validated) first jet.
    #[test]
    fn duchon_coordinate_evaluator_second_jet_matches_fd() -> Result<(), String> {
        let centers = array![
            [-1.0, -0.8],
            [-0.3, 0.4],
            [0.2, -0.5],
            [0.7, 0.9],
            [1.1, -0.2],
            [1.6, 0.6],
        ];
        let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
        let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
        assert_second_jet_matches_central_difference(&evaluator, coords, 1.0e-4)?;
        Ok(())
    }

    /// The Euclidean tangent-patch evaluator's monomial design and its
    /// first/second jets must be mutually consistent under finite differences.
    #[test]
    fn euclidean_patch_evaluator_jets_match_fd() -> Result<(), String> {
        let evaluator = EuclideanPatchEvaluator::new(2, 2).unwrap();
        let coords = array![[0.0, -1.0], [3.5, 0.25], [-0.75, 1.2], [0.4, 0.9]];
        assert_jacobian_matches_central_difference(&evaluator, coords.clone(), 1.0e-6);
        assert_second_jet_matches_central_difference(&evaluator, coords, 1.0e-5)?;
        // The degree-2 patch in d=2 has columns {1, x, y, x², xy, y²}.
        let (phi, _jet) = evaluator.evaluate(array![[0.0, 0.0]].view())?;
        assert_eq!(phi.ncols(), 6);
        Ok(())
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
        // ARD log-precision is per-axis (length == atom latent dim), not a
        // single scalar — see `SaeManifoldRho::to_flat` / `from_flat` and
        // the validation in `negative_log_ard_prior` (`ARD rho atom k has
        // len ... but atom dim is d`).
        let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(d)]);
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
            // Expected = JumpReLU gated majorizer PLUS the fixed logit
            // identifiability prior κ (added on every logit for all modes so
            // the per-row H_tt logit block is PD).
            let sparsity = if logit > threshold {
                let activation = crate::linalg::utils::stable_logistic(logit * inv_tau);
                let slope = activation * (1.0 - activation);
                sparsity_strength * slope * slope * inv_tau2
            } else {
                0.0
            };
            let expected = sparsity + SAE_LOGIT_IDENTIFIABILITY_PRECISION;
            assert!(
                entry.is_finite() && entry >= SAE_LOGIT_IDENTIFIABILITY_PRECISION,
                "JumpReLU gated hessian_diag majorizer + identifiability prior must be finite and \
                 ≥ κ at index {idx}; entry={entry}"
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

    /// End-to-end Isometry wiring oracle.
    ///
    /// Build a SAE atom around an evaluator whose `second_jet` is now
    /// implemented (periodic / sphere / torus), construct an
    /// [`IsometryPenalty`] with matching `latent_dim` and `p_out`, refresh
    /// the caches via [`refresh_isometry_caches_from_atom`], and check that
    ///
    ///   * `IsometryPenalty.value(target, rho)` is strictly positive (the
    ///     decoder we feed in is not orthonormal so the pullback metric is
    ///     not the identity, and the Euclidean reference picks up the gap).
    ///   * `IsometryPenalty.grad_target(target, rho)` is non-zero on at
    ///     least one latent-coordinate component.
    ///   * The analytic gradient matches a finite-difference oracle of
    ///     `value()` w.r.t. `target` (a single coord), where each FD probe
    ///     drives a fresh cache refresh — this is exactly the chain of
    ///     calls the SAE outer loop will make.
    ///
    /// The FD oracle re-uses the existing [`refresh_isometry_caches_from_atom`]
    /// helper for both the analytic side and the FD side, so any layout
    /// mismatch between `J`/`H` would show up as a tolerance failure rather
    /// than a silently zero gradient.
    fn assert_isometry_wiring_matches_fd(
        evaluator: Arc<dyn SaeBasisSecondJet>,
        coords: Array2<f64>,
    ) {
        let n_obs = coords.nrows();
        let latent_dim = coords.ncols();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        let p: usize = 3;
        // A deterministic non-orthonormal decoder: deterministic LCG-ish
        // floats keep the test reproducible without needing rand.
        let mut decoder = Array2::<f64>::zeros((m, p));
        for i in 0..m {
            for j in 0..p {
                let x = (i as f64) * 0.371 + (j as f64) * 0.193 + 0.5;
                decoder[[i, j]] = (x.sin() * 0.9) + 0.1 * ((i + j) as f64).cos();
            }
        }
        let smooth = Array2::<f64>::eye(m);
        let atom = SaeManifoldAtom::new(
            "iso_wire_test",
            SaeAtomBasisKind::Periodic,
            latent_dim,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            smooth,
        )
        .unwrap()
        .with_basis_second_jet(evaluator);

        let target_slice = PsiSlice::full(n_obs * latent_dim, Some(latent_dim));
        let penalty = IsometryPenalty::new_euclidean(target_slice, p);
        let rho = Array1::<f64>::zeros(1);

        // Without a refresh, the safe default is zero and the gradient is
        // all zeros. Confirm the precondition so the post-refresh contrast
        // is meaningful.
        let target_flat: Array1<f64> = coords.iter().copied().collect();
        let v0 = penalty.value(target_flat.view(), rho.view());
        assert_eq!(v0, IsometryPenalty::DEFAULT_VALUE_ON_MISSING_CACHE);
        let g0 = penalty.grad_target(target_flat.view(), rho.view());
        assert!(
            g0.iter().all(|x| *x == 0.0),
            "grad_target without cache must be all zeros, got {g0:?}"
        );

        // Refresh and re-evaluate.
        let installed_second =
            refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
        assert!(
            installed_second,
            "evaluator must implement second_jet for this oracle to run"
        );

        let value = penalty.value(target_flat.view(), rho.view());
        assert!(
            value > 1.0e-6,
            "expected non-trivial isometry loss after cache refresh, got {value}"
        );
        let grad = penalty.grad_target(target_flat.view(), rho.view());
        assert_eq!(grad.len(), target_flat.len());
        let max_abs = grad.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        assert!(
            max_abs > 1.0e-6,
            "expected non-zero isometry gradient on at least one component, max |grad|={max_abs}"
        );

        // FD check: bump one coord, refresh, compare value(t±h e_j) against
        // analytic grad[j]. Pick coord (row 0, axis 0).
        let h_fd = 1.0e-5;
        let probe_idx = 0usize; // (row=0, axis=0) flattens to 0.
        let mut coords_plus = coords.clone();
        coords_plus[[0, 0]] += h_fd;
        let mut coords_minus = coords.clone();
        coords_minus[[0, 0]] -= h_fd;

        refresh_isometry_caches_from_atom(&penalty, &atom, coords_plus.view()).unwrap();
        let target_plus: Array1<f64> = coords_plus.iter().copied().collect();
        let v_plus = penalty.value(target_plus.view(), rho.view());

        refresh_isometry_caches_from_atom(&penalty, &atom, coords_minus.view()).unwrap();
        let target_minus: Array1<f64> = coords_minus.iter().copied().collect();
        let v_minus = penalty.value(target_minus.view(), rho.view());

        // Reinstall the base caches before reading grad at the base point.
        refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
        let grad_base = penalty.grad_target(target_flat.view(), rho.view());

        let fd = (v_plus - v_minus) / (2.0 * h_fd);
        let analytic = grad_base[probe_idx];
        // Both `value` and `grad_target` use the cached `J` (and `grad_target`
        // also the cached `H`). With finite differencing the cache itself,
        // the analytic-vs-FD agreement bounds the entire pipeline (J build,
        // H build, accessor read, pullback metric, gradient assembly) to
        // O(h²) error. Tolerance 1e-3 leaves headroom for the per-evaluator
        // characteristic magnitude.
        assert!(
            (analytic - fd).abs() <= 1.0e-3 + 1.0e-4 * analytic.abs().max(fd.abs()),
            "isometry grad/FD mismatch at coord 0: analytic={analytic:.6e}, fd={fd:.6e}"
        );
    }

    #[test]
    fn isometry_wiring_periodic_matches_fd() {
        assert_isometry_wiring_matches_fd(
            Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap()),
            array![[0.12], [0.37], [0.58], [0.81]],
        );
    }

    #[test]
    fn isometry_wiring_sphere_matches_fd() {
        assert_isometry_wiring_matches_fd(
            Arc::new(SphereChartEvaluator),
            array![[-0.5, 0.3], [0.2, -1.1], [0.7, 0.9]],
        );
    }

    #[test]
    fn isometry_wiring_torus_matches_fd() {
        assert_isometry_wiring_matches_fd(
            Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
            array![[0.13, 0.42], [0.66, 0.19], [0.88, 0.55]],
        );
    }

    /// Multi-atom isometry pairing regression.
    ///
    /// Two SAE atoms share the same `(latent_dim, p_out)` signature but live
    /// on different coordinate blocks. The registry holds one isometry penalty
    /// per atom. The previous `find()` first-match logic paired *both*
    /// penalties to atom 0, so atom 1's coords were never installed into the
    /// second penalty's Jacobian cache — silently mislabeling the second
    /// atom's geometry as the first's. The positional pairing must instead
    /// refresh penalty `i` against atom `i`.
    ///
    /// We pin this by computing, independently, the Jacobian cache each atom
    /// would produce in isolation, then asserting that after
    /// `refresh_isometry_caches_from_term` the two registry penalties carry
    /// *distinct* caches matching their *own* atoms.
    #[test]
    fn refresh_isometry_caches_pairs_each_penalty_to_its_own_atom() {
        let latent_dim = 1usize;
        let p_out = 3usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());

        // Distinct coords per atom so the cached Jacobians must differ.
        let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
        let coords1 = array![[0.13], [0.41], [0.62], [0.91]];

        let build_atom = |name: &str, coords: &Array2<f64>, seed: f64| {
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let m = phi.ncols();
            let mut decoder = Array2::<f64>::zeros((m, p_out));
            for i in 0..m {
                for j in 0..p_out {
                    let x = (i as f64) * 0.371 + (j as f64) * 0.193 + seed;
                    decoder[[i, j]] = (x.sin() * 0.9) + 0.1 * ((i + j) as f64).cos();
                }
            }
            let smooth = Array2::<f64>::eye(m);
            SaeManifoldAtom::new(
                name,
                SaeAtomBasisKind::Periodic,
                latent_dim,
                phi,
                jet,
                decoder,
                smooth,
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone() as Arc<dyn SaeBasisSecondJet>)
        };

        let atom0 = build_atom("atom0", &coords0, 0.5);
        let atom1 = build_atom("atom1", &coords1, 1.7);

        // Independent ground-truth caches: refresh a standalone penalty
        // against each atom in isolation.
        let slice0 = PsiSlice::full(coords0.nrows() * latent_dim, Some(latent_dim));
        let control0 = IsometryPenalty::new_euclidean(slice0, p_out);
        refresh_isometry_caches_from_atom(&control0, &atom0, coords0.view()).unwrap();
        let expected0 = control0
            .jacobian_cache()
            .expect("control penalty 0 must have a Jacobian cache");

        let slice1 = PsiSlice::full(coords1.nrows() * latent_dim, Some(latent_dim));
        let control1 = IsometryPenalty::new_euclidean(slice1, p_out);
        refresh_isometry_caches_from_atom(&control1, &atom1, coords1.view()).unwrap();
        let expected1 = control1
            .jacobian_cache()
            .expect("control penalty 1 must have a Jacobian cache");

        // The two atoms genuinely differ, else the test is vacuous.
        assert_ne!(
            *expected0, *expected1,
            "atom 0 and atom 1 must produce distinct Jacobian caches"
        );

        // Build the term and a registry with one isometry penalty per atom.
        let logits = Array2::<f64>::zeros((coords0.nrows(), 2));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();

        let mut registry = AnalyticPenaltyRegistry::new();
        let pslice0 = PsiSlice::full(coords0.nrows() * latent_dim, Some(latent_dim));
        let pslice1 = PsiSlice::full(coords1.nrows() * latent_dim, Some(latent_dim));
        registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
            IsometryPenalty::new_euclidean(pslice0, p_out),
        )));
        registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
            IsometryPenalty::new_euclidean(pslice1, p_out),
        )));

        let coords_per_atom = vec![coords0.clone(), coords1.clone()];
        let refreshed =
            refresh_isometry_caches_from_term(&registry, &term, &coords_per_atom).unwrap();
        assert_eq!(refreshed, 2, "both penalties should install second caches");

        let cache0 = match &registry.penalties[0] {
            AnalyticPenaltyKind::Isometry(p) => p
                .jacobian_cache()
                .expect("penalty 0 cache must be populated"),
            _ => panic!("expected isometry penalty at index 0"),
        };
        let cache1 = match &registry.penalties[1] {
            AnalyticPenaltyKind::Isometry(p) => p
                .jacobian_cache()
                .expect("penalty 1 cache must be populated"),
            _ => panic!("expected isometry penalty at index 1"),
        };

        // Penalty i must carry atom i's cache — not both atom 0's.
        assert_eq!(
            *cache0, *expected0,
            "penalty 0 must be refreshed against atom 0"
        );
        assert_eq!(
            *cache1, *expected1,
            "penalty 1 must be refreshed against atom 1 (regression: old find() paired it to atom 0)"
        );
        assert_ne!(
            *cache0, *cache1,
            "the two penalties must not collapse onto the same atom"
        );
    }
}
