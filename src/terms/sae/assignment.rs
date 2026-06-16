//! Assignment gates and sparsity-prior helpers for the SAE manifold term.
//! Mechanically split from `sae_manifold.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::solver::evidence::{HybridAtomCandidate, HybridAtomChoice, select_hybrid_atom};
use crate::terms::analytic_penalties::{
    AnalyticPenalty, IBPAssignmentPenalty, IbpHessianDiagThirdChannels,
    SoftmaxAssignmentSparsityPenalty, resolve_learnable_weight,
};
use crate::terms::latent::{LatentCoordValues, LatentIdMode, LatentManifold};
use crate::terms::sae::manifold::SaeManifoldRho;

/// #976 Layer-1 guard: cap on one accepted iteration's assignment-logit
/// update, in units of the gate temperature τ (the gate's natural length
/// scale — every assignment mode reads logits through `σ(·/τ)` /
/// `softmax(·/τ)`). A 4τ move spans the gate's whole soft range, so healthy
/// convergence is never throttled, but no single inner iteration can carry a
/// gate from contention to numerically-zero support: a collapse takes
/// multiple accepted iterations, which guarantees the per-iteration
/// active-mass guard observes the decay before it completes. The clamp is
/// applied where the step is realised; when it binds, the realised objective
/// is evaluated on the clamped state, so the Armijo comparison stays
/// value-consistent (the unclamped quadratic model is merely conservative,
/// and step halvings shrink the trial below the cap).
pub(crate) const SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS: f64 = 4.0;

/// #976 Layer-1 guard: per-atom active-mass floor. The collapse statistic is
/// the atom's MAXIMUM assignment mass over rows, not its mean: a legitimately
/// sparse atom has a small mean but high mass on its own rows, while only an
/// atom with no material support anywhere — the #853 failure — has a small
/// max. An atom whose max mass falls below this floor is re-seeded (once) or
/// recorded as terminally collapsed; never a silent death, never a fit error.
pub(crate) const SAE_ATOM_ACTIVE_MASS_FLOOR: f64 = 1.0e-3;

/// #976 Layer-1 guard: re-seed budget per atom per joint fit. One second
/// chance from a fresh basin; a second breach means the collapse is (locally)
/// the objective's verdict at the current hyperparameters, which is recorded
/// as a terminal collapse event and left for the structure-search death move
/// to adjudicate — re-seeding in a loop would fight the optimizer.
pub(crate) const SAE_ATOM_COLLAPSE_RESEED_BUDGET: usize = 1;

/// #976 Layer-1 guard (decoder arm): an atom whose decoder block Frobenius norm
/// has fallen to this fraction of the dictionary's MEDIAN decoder norm carries
/// no material reconstruction signal — it has degenerated to (near-)zero output
/// and decodes the same nothing as every other collapsed atom. This is the
/// real-data K>1 failure that the gate-mass floor cannot see: the assignment
/// gates can stay spread across rows (mass guard satisfied) while the decoders
/// all collapse to ~0, giving EV≈0 and a rank-deficient per-row coordinate
/// Hessian on every row (the 0→K·n evidence-deflation jump). The statistic is a
/// RATIO to the dictionary median so it is scale-free and never fires for a
/// uniformly-small but well-conditioned decoder; only an atom that has fallen
/// far behind its peers is caught. By construction this is a no-op for K=1
/// (a single atom has no peer to fall behind, and the median equals its own
/// norm), so the K=1 path is byte-for-byte unchanged.
pub(crate) const SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO: f64 = 1.0e-3;

/// #976 Layer-1 guard (simultaneous-collapse arm): the reconstruction
/// explained-variance below which a K>=2 dictionary is judged to have
/// CO-collapsed — every atom degenerate together, so the median-relative
/// [`SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO`] test sees no atom "behind" its peers
/// and stays silent. This is the real-data K=2 failure (both atoms fall into one
/// seed/basin and the fit explains ~0 variance). The floor sits far below any
/// healthy curved-dictionary fit (real OLMo K=1 reaches EV ~0.22, K=3 ~0.40), so
/// only a genuinely degenerate dictionary trips it; a merely-difficult target
/// never does. When tripped, the guard reseeds all-but-the-strongest atom onto
/// distinct residual PCs to break the shared basin.
pub(crate) const SAE_DICTIONARY_COLLAPSE_EV_FLOOR: f64 = 0.02;

/// #976 / #1117 K>1 robustness: bounded DICTIONARY-level multi-start budget for
/// the simultaneous co-collapse arm (the EV-floor branch of
/// [`crate::terms::sae::manifold::SaeManifoldTerm::enforce_decoder_norm_guard`]).
/// Distinct from the per-atom [`SAE_ATOM_COLLAPSE_RESEED_BUDGET`] (= 1): that
/// budget governs reseeding ONE atom's gate logits against an optimizer that
/// keeps killing it, where a loop would fight the optimizer. A co-collapse
/// reseed is categorically different — it is a full-dictionary multi-start that
/// re-diversifies ALL atoms onto distinct principal directions of a FRESHLY
/// recomputed residual, so successive attempts explore genuinely different
/// basins. A single such reseed empirically cannot always break a K≥3 three-way
/// basin (identical (K, seed) flips EV≈0.40 ↔ 0.00), so this arm gets a small
/// bounded budget of independent multi-starts. It is consumed ONLY when the
/// whole dictionary explains < [`SAE_DICTIONARY_COLLAPSE_EV_FLOOR`] of the
/// variance — a no-op for any healthy fit (real OLMo K=1 ~0.22, K=2 ~0.40).
pub(crate) const SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET: usize = 3;

/// Machine-precision support cutoff for the smooth JumpReLU assignment prior,
/// in units of the gate temperature below the hard threshold. The forward gate
/// remains hard-zero at and below `threshold`, but the prior value/gradient and
/// compact Newton layout keep every logit with `(logit - threshold)/tau > -36`.
/// At the excluded edge `sigma(-36) ~= 2e-16`, so dropped value/gradient/Hessian
/// terms are below f64 noise instead of creating an algorithmic discontinuity.
pub(crate) const JUMPRELU_OPTIMIZATION_LOGIT_CUTOFF: f64 = -36.0;

/// Shared support predicate for JumpReLU optimization inclusion. This is
/// strictly weaker than the hard forward gate `logit > threshold`, which still
/// governs data-fit reconstruction and its logit JVP.
#[inline]
pub(crate) fn jumprelu_in_optimization_band(logit: f64, threshold: f64, temperature: f64) -> bool {
    (logit - threshold) / temperature > JUMPRELU_OPTIMIZATION_LOGIT_CUTOFF
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
    /// Hard-thresholded bounded gate: each atom is off (gate = 0) when its logit
    /// is at or below `threshold`, and on with a threshold-centered shifted
    /// sigmoid `σ((logit − threshold) / temperature) ∈ [0.5, 1)` above it. This
    /// is NOT literal JumpReLU `z·1[z>θ]` — the gate carries no magnitude; it is
    /// a member of the gate family (softmax simplex / IBP sigmoid / this hard
    /// gate) and stays bounded in [0, 1]. Reconstruction magnitude lives entirely
    /// in the decoder curve `g_k(t) = φ(t)ᵀ B_k`. The discontinuity at `threshold`
    /// (0 → 0.5) is the intended "jump".
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

    pub(crate) fn set_temperature(&mut self, new_temperature: f64) -> Result<(), String> {
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

    pub(crate) fn validate(&self) -> Result<(), String> {
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

    pub(crate) fn resolved_ibp_alpha(&self, rho: &SaeManifoldRho) -> Option<f64> {
        match *self {
            AssignmentMode::IBPMap {
                alpha,
                learnable_alpha,
                ..
            } => Some(if learnable_alpha {
                resolve_learnable_weight(alpha, rho.log_lambda_sparse)
            } else {
                alpha
            }),
            _ => None,
        }
    }
}

/// Per-row latent assignment state.
///
/// The stored assignment parameter is `logits`; non-negative assignments are
/// derived by row-wise softmax, independent IBP-MAP sigmoid active indicators,
/// or JumpReLU gates. Softmax logits are canonicalized to the reference chart
/// `logits[K - 1] = 0`, so the row-local Newton coordinates contain only the
/// first `K - 1` logits (`0` coordinates for `K = 1`). Gate-style modes keep
/// all `K` logits as identifiable scalar parameters. `coords[k]` holds
/// `t_{.,k}` for atom `k`.
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
        mut logits: Array2<f64>,
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
        for row in 0..n {
            validate_finite_logits(logits.row(row), row)?;
        }
        if matches!(mode, AssignmentMode::Softmax { .. }) {
            canonicalize_softmax_logits(&mut logits);
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

    pub fn assignment_coord_dim(&self) -> usize {
        match self.mode {
            AssignmentMode::Softmax { .. } => self.k_atoms().saturating_sub(1),
            AssignmentMode::IBPMap { .. } | AssignmentMode::JumpReLU { .. } => self.k_atoms(),
        }
    }

    pub fn row_block_dim(&self) -> usize {
        self.assignment_coord_dim() + self.total_coord_dim()
    }

    pub fn coord_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = self.assignment_coord_dim();
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

    pub fn assignments_row(&self, row: usize) -> Array1<f64> {
        self.try_assignments_row(row)
            .expect("assignment logits must be finite")
    }

    pub fn try_assignments_row(&self, row: usize) -> Result<Array1<f64>, String> {
        self.try_assignments_row_with_alpha(row, None)
    }

    pub(crate) fn try_assignments_row_for_rho(
        &self,
        row: usize,
        rho: &SaeManifoldRho,
    ) -> Result<Array1<f64>, String> {
        self.try_assignments_row_with_alpha(row, self.mode.resolved_ibp_alpha(rho))
    }

    fn try_assignments_row_with_alpha(
        &self,
        row: usize,
        resolved_ibp_alpha: Option<f64>,
    ) -> Result<Array1<f64>, String> {
        validate_finite_logits(self.logits.row(row), row)?;
        // Only Softmax collapses to a fixed assignment at K==1: its
        // assignment_coord_dim is K-1 = 0, so there is no free logit. IBPMap and
        // JumpReLU keep a free per-atom gate logit even at K==1
        // (assignment_coord_dim = K = 1), so they must fall through to their real
        // row functions or the logit would move the prior but not the gate.
        if self.k_atoms() == 1 && matches!(self.mode, AssignmentMode::Softmax { .. }) {
            return Ok(Array1::from_vec(vec![1.0]));
        }
        match self.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                Ok(softmax_row(self.logits.row(row), temperature))
            }
            AssignmentMode::IBPMap {
                temperature, alpha, ..
            } => Ok(ibp_map_row(
                self.logits.row(row),
                temperature,
                resolved_ibp_alpha.unwrap_or(alpha),
            )),
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => Ok(jumprelu_row(self.logits.row(row), temperature, threshold)),
        }
    }

    pub(crate) fn persist_resolved_ibp_alpha(&mut self, rho: &SaeManifoldRho) -> bool {
        let AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha: true,
        } = self.mode
        else {
            return false;
        };
        let resolved_alpha = resolve_learnable_weight(alpha, rho.log_lambda_sparse);
        self.mode = AssignmentMode::IBPMap {
            temperature,
            alpha: resolved_alpha,
            learnable_alpha: false,
        };
        true
    }

    pub(crate) fn assignments_for_rho(&self, rho: &SaeManifoldRho) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.try_assignments_row_for_rho(row, rho)?;
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        Ok(out)
    }

    /// Flatten extension coordinates in row-major SAE layout:
    /// `(assignment chart_i, t_i0[0..d_0], ..., t_iK[0..d_K])` for every row.
    /// Softmax contributes the first `K - 1` reference logits and omits the
    /// fixed reference logit; gate-style assignment modes contribute all `K`
    /// logits.
    pub fn flatten_ext_coords(&self) -> Array1<f64> {
        let n = self.n_obs();
        let q = self.row_block_dim();
        let k = self.k_atoms();
        let assignment_dim = self.assignment_coord_dim();
        let offsets = self.coord_offsets();
        let mut out = Array1::<f64>::zeros(n * q);
        for row in 0..n {
            let base = row * q;
            for atom in 0..assignment_dim {
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

pub(crate) fn sae_sigmoid_derivatives_from_value(
    value: f64,
    inv_tau: f64,
    scale: f64,
) -> (f64, f64, f64) {
    let sig = if scale > 0.0 { value / scale } else { 0.0 };
    let dz = scale * sig * (1.0 - sig) * inv_tau;
    let d2z = scale * sig * (1.0 - sig) * (1.0 - 2.0 * sig) * inv_tau * inv_tau;
    (value, dz, d2z)
}

pub(crate) fn neutral_gate_weights(mode: AssignmentMode, k_atoms: usize) -> Array1<f64> {
    match mode {
        AssignmentMode::Softmax { .. } => Array1::from_elem(k_atoms, 1.0 / (k_atoms.max(1) as f64)),
        AssignmentMode::IBPMap {
            temperature, alpha, ..
        } => ibp_map_row(Array1::<f64>::zeros(k_atoms).view(), temperature, alpha),
        AssignmentMode::JumpReLU { .. } => Array1::from_elem(k_atoms, 0.5),
    }
}

pub(crate) fn softmax_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
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

pub(crate) fn validate_finite_logits(
    logits: ArrayView1<'_, f64>,
    row: usize,
) -> Result<(), String> {
    for (col, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "SaeAssignment: non-finite assignment logit at row {row}, atom {col}: {v}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn canonicalize_softmax_logits(logits: &mut Array2<f64>) {
    let k = logits.ncols();
    if k == 0 {
        return;
    }
    if k == 1 {
        logits.fill(0.0);
        return;
    }
    for row in 0..logits.nrows() {
        let reference = logits[[row, k - 1]];
        for col in 0..k - 1 {
            logits[[row, col]] -= reference;
        }
        logits[[row, k - 1]] = 0.0;
    }
}

/// Deterministic ordered geometric-shrinkage MAP weights
/// `π_k = (α/(α+1))^k` for k = 0, .., K-1, with the first atom intentionally
/// left unshrunk (`π_0 = 1`, the always-available base atom). This is NOT a
/// sampled or variational Indian-Buffet-Process posterior: it is a fixed,
/// deterministic per-atom shrinkage schedule that biases assignment mass to
/// decay geometrically with atom index even when logits are tied. `α` is a
/// shrinkage rate (larger `α` ⇒ slower decay), not an IBP concentration in the
/// sampling sense. The geometric form coincides with the prior means of a
/// Beta(α, 1) stick-breaking construction, which is the motivation for the
/// schedule, but no sticks are drawn here.
pub(crate) fn ibp_stick_breaking_prior(k_atoms: usize, alpha: f64) -> Array1<f64> {
    // Accumulate the geometric schedule `π_k = ratio^k` in LOG space so the
    // prior stays a finite *soft* weight even for large `K`. The naive product
    // `acc *= ratio` underflows to exact `0.0` once `ratio^k < f64::MIN_POSITIVE`
    // (e.g. `(0.1/1.1)^320`), which would turn the soft shrinkage prior into a
    // HARD mask: such atoms would receive zero assignment AND zero logit
    // gradient (the gradient is multiplied by `π_k`), so they could never
    // reactivate. Working in log-space and flooring the exponentiated weight at
    // the smallest positive normal keeps every atom's gradient path alive while
    // preserving the geometric ordering.
    let mut out = Array1::<f64>::zeros(k_atoms);
    let log_ratio = (alpha / (alpha + 1.0)).ln();
    for k in 0..k_atoms {
        let log_pi = (k as f64) * log_ratio;
        out[k] = log_pi.exp().max(f64::MIN_POSITIVE);
    }
    out
}

/// IBP-MAP row activations: per-atom sigmoid likelihood times the truncated
/// stick-breaking prior mass. With tied logits the prior dominates and yields
/// strictly decreasing activations in atom index.
pub fn ibp_map_row(logits: ArrayView1<'_, f64>, temperature: f64, alpha: f64) -> Array1<f64> {
    let prior = ibp_stick_breaking_prior(logits.len(), alpha);
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = crate::linalg::utils::stable_logistic(logits[i] / temperature) * prior[i];
    }
    out
}

pub fn jumprelu_row(logits: ArrayView1<'_, f64>, temperature: f64, threshold: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        // Hard gate: strictly zero below threshold (the intended "jump"). Above
        // threshold the surrogate is centered at the threshold so the gate is
        // most informative exactly at the boundary it switches on:
        // σ((l−θ)/τ) ∈ [0.5, 1). Magnitude lives in the decoder, so the gate
        // stays bounded in [0, 1] by design.
        if logits[i] > threshold {
            out[i] = crate::linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
        }
    }
    out
}

pub(crate) struct ActiveAtomLogitJvp<'a> {
    pub(crate) mode: AssignmentMode,
    pub(crate) k: usize,
    pub(crate) logit_k: f64,
    pub(crate) a_k: f64,
    pub(crate) decoded_k: ArrayView1<'a, f64>,
    pub(crate) fitted: ArrayView1<'a, f64>,
    pub(crate) ibp_prior: Option<&'a [f64]>,
    pub(crate) compact_index: usize,
}

/// Fill the single compact logit-JVP row for active atom `k`, using the
/// per-mode assignment sensitivity `da_k/dl_k` contracted into the decoded /
/// fitted-corrected output direction. This is the active-set analogue of
/// [`fill_assignment_logit_jvp_rows`]: it reproduces that function's diagonal
/// logit row exactly for the atom `k`, but writes into a compact position of a
/// heterogeneous-`q` row block instead of the dense full-`K` Jacobian. `fitted`
/// is the row's *active-set* reconstruction so the softmax cross term
/// `(decoded_k − fitted)` is consistent with the curvature the compact block
/// carries.
pub(crate) fn fill_active_atom_logit_jvp(
    input: ActiveAtomLogitJvp<'_>,
    jac_compact: &mut Array2<f64>,
) {
    let ActiveAtomLogitJvp {
        mode,
        k,
        logit_k,
        a_k,
        decoded_k,
        fitted,
        ibp_prior,
        compact_index,
    } = input;
    let p = fitted.len();
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            // da_k/dl_k contracted: a_k (decoded_k − fitted) / τ.
            let inv_tau = 1.0 / temperature;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] =
                    a_k * (decoded_k[out_col] - fitted[out_col]) * inv_tau;
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
                jac_compact[[compact_index, out_col]] = dz * decoded_k[out_col];
            }
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // The data-fit Jacobian follows the hard forward gate. Below the
            // threshold the reconstruction contribution is exactly zero, so the
            // data-fit logit derivative must also be zero. Band-only atoms stay
            // in the compact row for prior terms, not phantom reconstruction
            // slope.
            if logit_k <= threshold {
                return;
            }
            let inv_tau = 1.0 / temperature;
            let activation = crate::linalg::utils::stable_logistic((logit_k - threshold) * inv_tau);
            let da = activation * (1.0 - activation) * inv_tau;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] = da * decoded_k[out_col];
            }
        }
    }
}

pub(crate) fn fill_assignment_logit_jvp_rows(
    mode: AssignmentMode,
    logits: ArrayView1<'_, f64>,
    assignments: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    fitted: ArrayView1<'_, f64>,
    ibp_prior: Option<&[f64]>,
    local_jac: &mut Array2<f64>,
) {
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            if assignments.len() == 1 {
                return;
            }
            // da_k/dl_j = a_k (1[k=j] - a_j) / tau, contracted against
            // the assignment-weighted fitted row. The dense row layout uses
            // the reference-logit chart, so only columns `0..K-1` are free;
            // the final reference logit is fixed at zero and has no row.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() - 1 {
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
            // Data-fit sensitivity follows the hard forward gate: rows at or
            // below the threshold have zero reconstruction value and therefore
            // zero data-fit logit derivative. The wider machine-precision prior
            // support is a compact-layout/prior rule, not a data-fit STE.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                if logits[logit_col] <= threshold {
                    continue;
                }
                let activation = crate::linalg::utils::stable_logistic(
                    (logits[logit_col] - threshold) * inv_tau,
                );
                let da = activation * (1.0 - activation) * inv_tau;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = da * decoded[[logit_col, out_col]];
                }
            }
        }
    }
}

pub(crate) fn flat_logits(logits: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for row in 0..logits.nrows() {
        let start = row * logits.ncols();
        for col in 0..logits.ncols() {
            out[start + col] = logits[[row, col]];
        }
    }
    out
}

pub(crate) fn assignment_prior_value(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)
            .expect("assignment logits must be finite");
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return 0.0;
    }
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
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                // Keep the fixed-alpha value path on the same weighting branch as
                // assignment_prior_grad_hdiag; that gradient path owns the
                // lambda_sparse convention for IBP assignment sparsity.
                penalty.weight = rho.lambda_sparse();
                Array1::zeros(0)
            };
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // Sparsity penalty uses the same threshold-centered surrogate and
            // machine-precision support as its gradient/Hessian. Data-fit
            // reconstruction remains hard-gated by `jumprelu_row`.
            let sparsity_strength = rho.lambda_sparse();
            let mut acc = 0.0;
            for &logit in target.iter() {
                if jumprelu_in_optimization_band(logit, threshold, temperature) {
                    acc += crate::linalg::utils::stable_logistic((logit - threshold) / temperature);
                }
            }
            sparsity_strength * acc
        }
    }
}

pub(crate) fn assignment_prior_log_strength_derivative(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> f64 {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)
            .expect("assignment logits must be finite");
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return 0.0;
    }
    match assignment.mode {
        AssignmentMode::Softmax { .. } | AssignmentMode::JumpReLU { .. } => {
            assignment_prior_value(assignment, rho)
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            if learnable_alpha {
                let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse]);
                penalty.grad_rho(target.view(), rho_view.view())[0]
            } else {
                penalty.weight = rho.lambda_sparse();
                penalty.value(target.view(), Array1::<f64>::zeros(0).view())
            }
        }
    }
}

pub(crate) fn assignment_prior_log_strength_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<Array1<f64>, String> {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| {
                    "softmax assignment log-strength hessian diag unavailable".to_string()
                })
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            let sparsity_strength = rho.lambda_sparse();
            let inv_tau = 1.0 / temperature;
            let inv_tau2 = inv_tau * inv_tau;
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                let logit = target[idx];
                if !jumprelu_in_optimization_band(logit, threshold, temperature) {
                    continue;
                }
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                d[idx] = sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2;
            }
            Ok(d)
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            if learnable_alpha {
                let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse]);
                Ok(penalty.hessian_diag_log_alpha_derivative(target.view(), rho_view.view()))
            } else {
                penalty.weight = rho.lambda_sparse();
                penalty
                    .hessian_diag(target.view(), Array1::<f64>::zeros(0).view())
                    .ok_or_else(|| {
                        "IBP assignment log-strength hessian diag unavailable".to_string()
                    })
            }
        }
    }
}

pub(crate) fn assignment_prior_log_strength_target_mixed(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<Array1<f64>, String> {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    match assignment.mode {
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha: true,
        } => {
            let penalty = IBPAssignmentPenalty::new(assignment.k_atoms(), alpha, temperature, true);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse]);
            Ok(penalty.log_alpha_target_mixed_derivative(target.view(), rho_view.view()))
        }
        _ => Ok(assignment_prior_grad_hdiag(assignment, rho)?.0),
    }
}

pub(crate) fn assignment_prior_grad_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    let mut grad = Array1::<f64>::zeros(target.len());
    let mut diag = Array1::<f64>::zeros(target.len());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
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
            // Scale the IBP assignment-sparsity prior by `lambda_sparse`, exactly
            // like the Softmax and JumpReLU branches do (Softmax folds it into the
            // penalty's rho coordinate, JumpReLU multiplies `sparsity_strength`).
            // Previously the IBP penalty used its hardcoded `weight = 1.0` and the
            // `rho.log_lambda_sparse` coordinate never reached it (the rho_view was
            // empty for the common `learnable_alpha = false` config), so the prior
            // ran at full strength with no way to dial it down — and its
            // Beta-Bernoulli BCE energy `−mass·ln π_k − (n−mass)·ln(1−π_k)` toward
            // the self-referential empirical active fraction `π_k` has its global
            // minimum at the all-off gate, so at full weight it over-shrank the
            // assignment off both atoms even with a truth-seeded decoder (#853).
            // Routing `lambda_sparse` into the penalty weight makes the prior a
            // genuine, user-controllable lever balanced against the data fit.
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            // When `alpha` is learnable, `log_lambda_sparse` already modulates
            // it through `resolved_alpha(rho)`, so the weight stays 1.0 to avoid
            // double-counting that coordinate. Only when `alpha` is fixed (so the
            // sparse coordinate would otherwise be ignored entirely) does
            // `lambda_sparse` become the prior's weight lever.
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                penalty.weight = rho.lambda_sparse();
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
            // Gradient and exact diagonal Hessian of the sparsity value's
            // threshold-centered surrogate σ((l−θ)/τ), using the same
            // machine-precision support as the value path. Data-fit JVP support
            // is narrower and follows the hard forward gate.
            let sparsity_strength = rho.lambda_sparse();
            let inv_tau = 1.0 / temperature;
            let inv_tau2 = inv_tau * inv_tau;
            let mut g = Array1::<f64>::zeros(target.len());
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                let logit = target[idx];
                if !jumprelu_in_optimization_band(logit, threshold, temperature) {
                    continue;
                }
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                g[idx] = sparsity_strength * slope * inv_tau;
                d[idx] = sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2;
            }
            (g, d)
        }
    };
    grad += &sparsity_grad;
    diag += &sparsity_diag;
    Ok((grad, diag))
}

/// Build the exact IBP `hessian_diag` logit third-derivative channels (#1006)
/// for the SAE log-det adjoint Γ, using the SAME penalty configuration —
/// `alpha`/`tau`/`learnable_alpha` and the `lambda_sparse` weight convention —
/// that [`assignment_prior_grad_hdiag`] assembles into `htt`. Returns `None`
/// for non-IBP assignment modes (no cross-row empirical-π coupling to correct).
pub(crate) fn ibp_assignment_third_channels(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<Option<IbpHessianDiagThirdChannels>, String> {
    let AssignmentMode::IBPMap {
        temperature,
        alpha,
        learnable_alpha,
    } = assignment.mode
    else {
        return Ok(None);
    };
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    let mut penalty =
        IBPAssignmentPenalty::new(assignment.k_atoms(), alpha, temperature, learnable_alpha);
    // Mirror assignment_prior_grad_hdiag exactly: when alpha is learnable the
    // sparse coordinate already modulates it through resolved_alpha(rho), so the
    // weight stays 1.0; otherwise lambda_sparse becomes the prior's weight lever.
    let rho_view = if learnable_alpha {
        Array1::from_vec(vec![rho.log_lambda_sparse])
    } else {
        penalty.weight = rho.lambda_sparse();
        Array1::zeros(0)
    };
    Ok(Some(penalty.hessian_diag_logit_third_channels(
        target.view(),
        rho_view.view(),
    )))
}

/// #1026 hybrid curved + linear-tail adjudication for one SAE atom slot.
///
/// A hybrid dictionary lets each atom slot be either a CURVED atom (its fitted
/// `latent_dim ≥ 1` manifold chart, whose decoded image may turn) or its LINEAR
/// special case (the euclidean-d=1-linear atom — one straight decoder direction,
/// `γ(t) = t·b`, zero turning). The two are nested: the linear atom is exactly
/// the curved family restricted to its straight sub-model, so a hybrid slot
/// cannot lose to pure-linear at matched actives — it strictly generalizes it.
///
/// This is the single call the SAE fitter makes per atom to choose the split by
/// EVIDENCE rather than fiat. It packages the atom's two already-fitted
/// candidates — each scored on the COMMON rank-aware Laplace scale (`−V = NLE`,
/// lower wins, identical to the union/mixture rungs) on the same rows — and
/// routes them through [`select_hybrid_atom`]. The curved candidate's fitted
/// turning `Θ` (from
/// [`crate::terms::sae::chart_canonicalization::d1_atom_fitted_turning`]) enters
/// as the decision feature: a `Θ → 0` atom yields to the cheaper linear tail by
/// construction (the dominance floor — a curved atom buys nothing on a straight
/// feature), a high-`Θ` atom takes the curved parameterization when its
/// curvature lowers the NLE by more than its extra-parameter price (the `Θ/√ε`
/// crossover).
///
/// `manifold` is the atom's fitted chart manifold; a non-curveable (already
/// Euclidean-flat) chart can only present the linear candidate, which this
/// helper enforces by ignoring any curved candidate offered for a flat chart —
/// a flat chart has no curvature to price, so the linear special case is its
/// only honest parameterization. Curveable charts present both candidates.
///
/// # Wiring into the fitter (the one call into `sae_manifold.rs`)
///
/// The post-fit pass in `sae_manifold.rs` already computes each d=1 atom's
/// fitted turning `Θ` (the read-only EV-vs-Θ diagnostic). To make the split
/// load-bearing, that pass supplies, per atom, the curved-candidate NLE +
/// parameter count + `Θ` and the linear-candidate NLE + parameter count (both
/// fitted on the atom's rows), and calls this helper; the returned
/// [`HybridAtomChoice`] tells the fitter which parameterization to keep for that
/// slot. The fitting of the two candidates lives in `sae_manifold.rs` (the
/// manifold-chart fitter); the SELECTION/scoring lives here.
pub fn select_hybrid_atom_parameterization(
    manifold: &LatentManifold,
    curved: Option<HybridAtomCandidate>,
    linear: HybridAtomCandidate,
) -> HybridAtomChoice {
    // A flat (Euclidean) chart has no curvature to price: its only honest
    // parameterization is the linear special case, so any curved candidate
    // offered for it is dropped before the evidence comparison. Curveable charts
    // (Circle / Sphere / Torus / curved products) present both candidates.
    let curved = if manifold.is_euclidean() {
        None
    } else {
        curved
    };
    let candidates: Vec<HybridAtomCandidate> = match curved {
        Some(c) => vec![linear, c],
        None => vec![linear],
    };
    // `candidates` is never empty (it always contains the linear candidate), so
    // the selector always returns a choice.
    select_hybrid_atom(&candidates).expect("hybrid atom slot always has the linear candidate")
}

#[cfg(test)]
mod hybrid_split_tests {
    use super::*;
    use crate::solver::evidence::HybridAtomParam;

    #[test]
    fn flat_chart_drops_curved_candidate_and_keeps_linear() {
        // A Euclidean chart has no curvature: even if a curved candidate with a
        // lower NLE is offered, the helper drops it (a flat chart cannot honestly
        // present a curved parameterization).
        let linear = HybridAtomCandidate::linear(100.0, 2);
        let curved = HybridAtomCandidate::curved(1, 1.0, 5, Some(2.0));
        let choice =
            select_hybrid_atom_parameterization(&LatentManifold::Euclidean, Some(curved), linear);
        assert!(choice.param.is_linear());
    }

    #[test]
    fn curveable_chart_selects_curved_when_turning_pays() {
        // A Circle chart presents both candidates; a turning feature whose curved
        // fit beats the linear secant on evidence selects curved.
        let linear = HybridAtomCandidate::linear(100.0, 2);
        let curved = HybridAtomCandidate::curved(1, 70.0, 5, Some(2.0 * std::f64::consts::PI));
        let choice = select_hybrid_atom_parameterization(
            &LatentManifold::Circle {
                period: 2.0 * std::f64::consts::PI,
            },
            Some(curved),
            linear,
        );
        assert_eq!(choice.param, HybridAtomParam::Curved { latent_dim: 1 });
    }

    #[test]
    fn curveable_chart_falls_back_to_linear_when_no_curved_candidate() {
        let linear = HybridAtomCandidate::linear(33.0, 2);
        let choice = select_hybrid_atom_parameterization(
            &LatentManifold::Circle {
                period: 2.0 * std::f64::consts::PI,
            },
            None,
            linear,
        );
        assert!(choice.param.is_linear());
        assert_eq!(choice.num_parameters, 2);
    }
}
