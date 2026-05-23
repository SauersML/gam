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
//!   `LatentCoordValues`). Per-row latent coordinates are written `t`; reserve
//!   `ψ` for existing kernel-shape state such as `SpatialLogKappaCoords`.
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
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, IBPAssignmentPenalty, PsiSlice,
    SoftmaxAssignmentSparsityPenalty,
};
use crate::terms::latent_coord::{LatentCoordValues, LatentIdMode};

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
    EuclideanPatch,
    Precomputed(String),
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
}

impl SaeManifoldAtom {
    #[allow(clippy::too_many_arguments)]
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
        })
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
        let m = self.basis_size();
        let mut out = Array1::<f64>::zeros(p);
        for basis_col in 0..m {
            let phi = self.basis_values[[row, basis_col]];
            if phi == 0.0 {
                continue;
            }
            for out_col in 0..p {
                out[out_col] += phi * self.decoder_coefficients[[basis_col, out_col]];
            }
        }
        out
    }

    /// `d g_k(t_{ik}) / d t_{ik,j}` for one row and latent axis.
    pub fn decoded_derivative_row(&self, row: usize, latent_axis: usize) -> Array1<f64> {
        let p = self.output_dim();
        let m = self.basis_size();
        let mut out = Array1::<f64>::zeros(p);
        for basis_col in 0..m {
            let dphi = self.basis_jacobian[[row, basis_col, latent_axis]];
            if dphi == 0.0 {
                continue;
            }
            for out_col in 0..p {
                out[out_col] += dphi * self.decoder_coefficients[[basis_col, out_col]];
            }
        }
        out
    }
}

/// Assignment prior/relaxation used by [`SaeAssignment`].
#[derive(Debug, Clone)]
pub enum AssignmentMode {
    /// Row-wise simplex assignment with entropy sparsity.
    Softmax { temperature: f64, sparsity: f64 },
    /// Deterministic concrete relaxation of a truncated IBP active set.
    IBPMap {
        temperature: f64,
        alpha: f64,
        learnable_alpha: bool,
    },
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

    pub fn temperature(&self) -> f64 {
        match *self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::IBPMap { temperature, .. } => temperature,
        }
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
        }
        Ok(())
    }
}

/// Per-row latent assignment state.
///
/// The free assignment parameter is `logits`; non-negative assignments are
/// derived by row-wise softmax or by independent IBP-MAP sigmoid active
/// indicators. `coords[k]` holds `t_{.,k}` for atom `k`.
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

    pub fn assignments_row(&self, row: usize) -> Array1<f64> {
        match self.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                softmax_row(self.logits.row(row), temperature)
            }
            AssignmentMode::IBPMap { temperature, .. } => {
                sigmoid_row(self.logits.row(row), temperature)
            }
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
}

/// REML-selected continuous hyperparameters for SAE-manifold.
#[derive(Debug, Clone)]
pub struct SaeManifoldRho {
    /// `log(lambda_sparse)` for softmax entropy, or the learnable `log(alpha)`
    /// offset for IBP-MAP assignment.
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
    pub fn total(&self) -> f64 {
        self.data_fit + self.assignment_sparsity + self.smoothness + self.ard
    }

    /// Laplace/REML wrappers rank larger evidence higher. This local score is
    /// the negative penalized objective, used when a full `RemlState` is not
    /// driving the term yet.
    pub fn evidence_proxy(&self) -> f64 {
        -self.total()
    }
}

/// Full SAE-manifold term.
#[derive(Debug, Clone)]
pub struct SaeManifoldTerm {
    pub atoms: Vec<SaeManifoldAtom>,
    pub assignment: SaeAssignment,
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
        Ok(Self { atoms, assignment })
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
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let a = self.assignment.assignments_row(row);
            for atom_idx in 0..k_atoms {
                let g = self.atoms[atom_idx].decoded_row(row);
                for out_col in 0..p {
                    out[[row, out_col]] += a[atom_idx] * g[out_col];
                }
            }
        }
        out
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
        let fitted = self.fitted();
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
        let (assignment_grad, assignment_hdiag) = assignment_prior_grad_hdiag(&self.assignment, rho);
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

        for row in 0..n {
            let assignments = self.assignment.assignments_row(row);
            let mut decoded: Vec<Array1<f64>> = Vec::with_capacity(k_atoms);
            let mut fitted = Array1::<f64>::zeros(p);
            for atom_idx in 0..k_atoms {
                let g = self.atoms[atom_idx].decoded_row(row);
                for out_col in 0..p {
                    fitted[out_col] += assignments[atom_idx] * g[out_col];
                }
                decoded.push(g);
            }
            let mut error = Array1::<f64>::zeros(p);
            for out_col in 0..p {
                error[out_col] = fitted[out_col] - target[[row, out_col]];
            }

            let mut local_jac = Array2::<f64>::zeros((q, p));
            match self.assignment.mode {
                AssignmentMode::Softmax { temperature, .. } => {
                    // da_k/dl_j = a_k (1[k=j] - a_j) / tau, contracted
                    // against the assignment-weighted fitted row.
                    let inv_tau = 1.0 / temperature;
                    for logit_col in 0..k_atoms {
                        for out_col in 0..p {
                            local_jac[[logit_col, out_col]] = assignments[logit_col]
                                * (decoded[logit_col][out_col] - fitted[out_col])
                                * inv_tau;
                        }
                    }
                }
                AssignmentMode::IBPMap { temperature, .. } => {
                    // Independent concrete-Bernoulli active indicators:
                    // dz_k/dl_k = z_k(1-z_k)/tau.
                    let inv_tau = 1.0 / temperature;
                    for logit_col in 0..k_atoms {
                        let dz = assignments[logit_col] * (1.0 - assignments[logit_col]) * inv_tau;
                        for out_col in 0..p {
                            local_jac[[logit_col, out_col]] =
                                dz * decoded[logit_col][out_col];
                        }
                    }
                }
            }
            // Coordinate columns.
            for atom_idx in 0..k_atoms {
                let d = self.atoms[atom_idx].latent_dim;
                let off = coord_offsets[atom_idx];
                for axis in 0..d {
                    let dg = self.atoms[atom_idx].decoded_derivative_row(row, axis);
                    for out_col in 0..p {
                        local_jac[[off + axis, out_col]] = assignments[atom_idx] * dg[out_col];
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
            for atom_idx in 0..k_atoms {
                let atom = &self.atoms[atom_idx];
                let atom_beta_off = beta_offsets[atom_idx];
                let m = atom.basis_size();
                let a_k = assignments[atom_idx];
                for basis_col in 0..m {
                    let phi = atom.basis_values[[row, basis_col]];
                    for out_col in 0..p {
                        let beta_idx = atom_beta_off + basis_col * p + out_col;
                        let j_beta = a_k * phi;
                        sys.gb[beta_idx] += j_beta * error[out_col];
                        for local_col in 0..q {
                            block.htbeta[[local_col, beta_idx]] +=
                                local_jac[[local_col, out_col]] * j_beta;
                        }
                        for atom_j in 0..k_atoms {
                            let atom2 = &self.atoms[atom_j];
                            let m2 = atom2.basis_size();
                            let off2 = beta_offsets[atom_j];
                            let a_j = assignments[atom_j];
                            for basis_col2 in 0..m2 {
                                let beta_j = off2 + basis_col2 * p + out_col;
                                sys.hbb[[beta_idx, beta_j]] +=
                                    j_beta * a_j * atom2.basis_values[[row, basis_col2]];
                            }
                        }
                    }
                }
            }
            sys.rows[row] = block;
        }
        Ok(sys)
    }

    pub fn solve_newton_step(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let sys = self
            .assemble_arrow_schur(target, rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        sys.solve(ridge_ext_coord, ridge_beta)
    }

    pub fn apply_newton_step(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
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

            // The FFI caller supplies basis values and jets at the current grid.
            // Advance the stored basis by the same first-order model used by
            // the Gauss-Newton arrow-Schur step.
            let m = self.atoms[atom_idx].basis_size();
            for row in 0..n {
                for basis_col in 0..m {
                    let mut dphi = 0.0;
                    for axis in 0..d {
                        dphi += self.atoms[atom_idx].basis_jacobian[[row, basis_col, axis]]
                            * delta_coord[row * d + axis];
                    }
                    self.atoms[atom_idx].basis_values[[row, basis_col]] += dphi;
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
        rho: &SaeManifoldRho,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeManifoldLoss, String> {
        for _ in 0..max_iter {
            let (delta_ext_coord, delta_beta) = self
                .solve_newton_step(target, rho, ridge_ext_coord, ridge_beta)
                .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            self.apply_newton_step(delta_ext_coord.view(), delta_beta.view(), step_size)?;
        }
        self.loss(target, rho)
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
            } => AnalyticPenaltyKind::IBPAssignment(Arc::new(IBPAssignmentPenalty::new(
                self.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            ))),
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

fn softmax_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let k = logits.len();
    let inv_tau = 1.0 / temperature;
    let mut max_scaled = f64::NEG_INFINITY;
    for &v in logits.iter() {
        max_scaled = max_scaled.max(v * inv_tau);
    }
    let mut out = Array1::<f64>::zeros(k);
    let mut sum = 0.0;
    for i in 0..k {
        let v = (logits[i] * inv_tau - max_scaled).exp();
        out[i] = v;
        sum += v;
    }
    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / k as f64;
        out.fill(uniform);
        return out;
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

fn sigmoid_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        let x = logits[i] / temperature;
        out[i] = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        };
    }
    out
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
            let penalty =
                IBPAssignmentPenalty::new(assignment.k_atoms(), alpha, temperature, learnable_alpha);
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                Array1::zeros(0)
            };
            penalty.value(target.view(), rho_view.view())
        }
    }
}

fn assignment_prior_grad_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> (Array1<f64>, Array1<f64>) {
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
                .expect("softmax assignment hessian diag");
            (grad, diag)
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let penalty =
                IBPAssignmentPenalty::new(assignment.k_atoms(), alpha, temperature, learnable_alpha);
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                Array1::zeros(0)
            };
            let grad = penalty.grad_target(target.view(), rho_view.view());
            let diag = penalty
                .hessian_diag(target.view(), rho_view.view())
                .expect("IBP assignment hessian diag");
            (grad, diag)
        }
    }
}

/// Helper for padded FFI callers. Arrays use `(K, N, M_max)` and
/// `(K, N, M_max, D_max)` storage, with `basis_sizes` and `latent_dims`
/// selecting each atom's active prefix.
#[allow(clippy::too_many_arguments)]
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
) -> Result<SaeManifoldTerm, String> {
    let k_atoms = basis_sizes.len();
    if latent_dims.len() != k_atoms || basis_kinds.len() != k_atoms || coords.len() != k_atoms {
        return Err("term_from_padded_blocks: K-length metadata mismatch".into());
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
        atoms.push(SaeManifoldAtom::new(
            format!("atom_{k}"),
            basis_kinds[k].clone(),
            d,
            phi,
            jet,
            b,
            s,
        )?);
    }
    let assignment = SaeAssignment::from_blocks_with_mode(logits.to_owned(), coords.to_vec(), mode)?;
    SaeManifoldTerm::new(atoms, assignment)
}
