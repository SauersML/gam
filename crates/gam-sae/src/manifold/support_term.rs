//! Canonical support-sparse curved term and fixed-point inner solve.
//!
//! Hard-TopK gates are read-only binary support. Consequently a row's only
//! live local parameters are the heterogeneous coordinates
//! `concat_{k in S_i} t_ik`; no gate/logit coordinate exists. This term owns
//! that representation directly and evaluates basis values and analytic jets
//! only for active `(row, atom)` pairs.

use crate::assignment::AssignmentMode;
use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
use ndarray::{Array1, Array2, ArrayView2};
use std::ops::Range;
use std::sync::Arc;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaeSupportStationarity {
    pub decoder_l2: f64,
    pub decoder_max_abs: f64,
    pub coordinate_l2: f64,
    pub coordinate_max_abs: f64,
}

impl SaeSupportStationarity {
    pub fn max_abs(self) -> f64 {
        self.decoder_max_abs.max(self.coordinate_max_abs)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SaeSupportFixedPointReport {
    pub iterations: usize,
    pub objective: f64,
    pub stationarity: SaeSupportStationarity,
    pub max_recurrence_change: f64,
    /// True only after a second complete decoder/coordinate cycle recurs within
    /// the same tolerance at the raw (undamped) stationarity point.
    pub recurred: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SaeSupportCoordinateFixedPointReport {
    pub iterations: usize,
    pub objective: f64,
    pub coordinate_l2: f64,
    pub coordinate_max_abs: f64,
    pub max_recurrence_change: f64,
    /// True only after two complete frozen-decoder coordinate cycles recur at
    /// the raw coordinate stationarity point.
    pub recurred: bool,
}

struct ActiveAtomEval {
    phi: Array1<f64>,
    decoded: Array1<f64>,
    /// Coordinate-major decoded jet, `(d_k, P)`.
    jacobian: Array2<f64>,
}

#[derive(Clone)]
struct SupportBasisBlock {
    beta_offset: usize,
    phi: Array1<f64>,
}

#[derive(Clone)]
struct SupportLinearizedRow {
    blocks: Vec<SupportBasisBlock>,
    jacobian: Array2<f64>,
}

#[derive(Clone)]
struct SupportBetaOperator {
    rows: Vec<SupportLinearizedRow>,
    beta_offsets: Vec<usize>,
    basis_sizes: Vec<usize>,
    penalties: Vec<Array2<f64>>,
    lambda_smooth: Vec<f64>,
    output_dim: usize,
    beta_dim: usize,
}

impl SupportBetaOperator {
    fn apply(&self, vector: ndarray::ArrayView1<'_, f64>, out: &mut Array1<f64>) {
        debug_assert_eq!(vector.len(), self.beta_dim);
        debug_assert_eq!(out.len(), self.beta_dim);
        out.fill(0.0);
        let mut output = vec![0.0; self.output_dim];
        for row in &self.rows {
            output.fill(0.0);
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        output[channel] += block.phi[basis] * vector[base + channel];
                    }
                }
            }
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        out[base + channel] += block.phi[basis] * output[channel];
                    }
                }
            }
        }
        for atom in 0..self.penalties.len() {
            let lambda = self.lambda_smooth[atom];
            let m = self.basis_sizes[atom];
            let offset = self.beta_offsets[atom];
            for left in 0..m {
                for right in 0..m {
                    let weight = lambda * self.penalties[atom][[left, right]];
                    for channel in 0..self.output_dim {
                        out[offset + left * self.output_dim + channel] +=
                            weight * vector[offset + right * self.output_dim + channel];
                    }
                }
            }
        }
    }

    fn htbeta_forward(
        &self,
        row: usize,
        vector: ndarray::ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) {
        let linearized = &self.rows[row];
        let mut output = vec![0.0; self.output_dim];
        for block in &linearized.blocks {
            for basis in 0..block.phi.len() {
                let base = block.beta_offset + basis * self.output_dim;
                for channel in 0..self.output_dim {
                    output[channel] += block.phi[basis] * vector[base + channel];
                }
            }
        }
        out.fill(0.0);
        for axis in 0..linearized.jacobian.nrows() {
            for channel in 0..self.output_dim {
                out[axis] += linearized.jacobian[[axis, channel]] * output[channel];
            }
        }
    }

    fn htbeta_transpose(
        &self,
        row: usize,
        vector: ndarray::ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) {
        let linearized = &self.rows[row];
        let mut output = vec![0.0; self.output_dim];
        for axis in 0..linearized.jacobian.nrows() {
            for channel in 0..self.output_dim {
                output[channel] += linearized.jacobian[[axis, channel]] * vector[axis];
            }
        }
        for block in &linearized.blocks {
            for basis in 0..block.phi.len() {
                let base = block.beta_offset + basis * self.output_dim;
                for channel in 0..self.output_dim {
                    out[base + channel] += block.phi[basis] * output[channel];
                }
            }
        }
    }
}

/// One hard-TopK curved model with no dense assignment specialization.
#[derive(Debug, Clone)]
pub struct SaeSupportSparseTerm {
    pub atoms: Vec<SaeManifoldAtom>,
    pub assignment: SaeAssignmentState,
    output_dim: usize,
    /// Inverted support index. Total entries are exactly `N·support_k`.
    atom_rows: Vec<Vec<(usize, usize)>>,
}

impl SaeSupportSparseTerm {
    #[must_use = "term construction error must be handled"]
    pub fn new(
        atoms: Vec<SaeManifoldAtom>,
        assignment: SaeAssignmentState,
    ) -> Result<Self, String> {
        let k_atoms = atoms.len();
        if k_atoms == 0 || assignment.k_atoms() != k_atoms {
            return Err(format!(
                "SaeSupportSparseTerm::new: atom count {k_atoms} != assignment K={}",
                assignment.k_atoms()
            ));
        }
        let support_k = match assignment.mode() {
            AssignmentMode::TopK { k } => k,
            other => {
                return Err(format!(
                    "SaeSupportSparseTerm::new requires hard TopK assignment state; got {other:?}"
                ));
            }
        };
        let output_dim = atoms[0].output_dim();
        if output_dim == 0 {
            return Err(
                "SaeSupportSparseTerm::new: decoder output dimension must be positive".into(),
            );
        }
        for (atom, template) in atoms.iter().enumerate() {
            if template.output_dim() != output_dim {
                return Err(format!(
                    "SaeSupportSparseTerm::new: atom {atom} output dimension {} != {output_dim}",
                    template.output_dim()
                ));
            }
            if template.latent_dim != assignment.atom_coord_dim(atom) {
                return Err(format!(
                    "SaeSupportSparseTerm::new: atom {atom} latent dim {} != assignment dim {}",
                    template.latent_dim,
                    assignment.atom_coord_dim(atom)
                ));
            }
            if template.basis_evaluator.is_none() {
                return Err(format!(
                    "SaeSupportSparseTerm::new: atom {atom} has no analytic basis evaluator"
                ));
            }
        }
        let mut atom_rows = vec![Vec::new(); k_atoms];
        for row in 0..assignment.n_obs() {
            let support = assignment.support_indices(row);
            if support.len() != support_k {
                return Err(format!(
                    "SaeSupportSparseTerm::new: row {row} support width {} != top_k={support_k}",
                    support.len()
                ));
            }
            for (slot, &atom) in support.iter().enumerate() {
                atom_rows[atom as usize].push((row, slot));
            }
        }
        Ok(Self {
            atoms,
            assignment,
            output_dim,
            atom_rows,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.assignment.n_obs()
    }

    pub fn k_atoms(&self) -> usize {
        self.atoms.len()
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn active_pair_count(&self) -> usize {
        self.atom_rows.iter().map(Vec::len).sum()
    }

    /// Route new rows against this fitted decoder without constructing a
    /// `rows × K` score matrix. Candidate reconstruction improvements are
    /// streamed one atom at a time and only the best `support_k` candidates,
    /// including their heterogeneous coordinates, survive for each row.
    pub fn reroute_fixed_decoder(
        &self,
        target: ArrayView2<'_, f64>,
        support_k: usize,
        random_state: u64,
    ) -> Result<Self, String> {
        if target.ncols() != self.output_dim || target.nrows() == 0 {
            return Err(format!(
                "SaeSupportSparseTerm::reroute_fixed_decoder: target {:?} must have positive rows and P={}",
                target.dim(),
                self.output_dim
            ));
        }
        if support_k == 0 || support_k > self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm::reroute_fixed_decoder requires 1 <= support_k <= K={}; got {support_k}",
                self.k_atoms()
            ));
        }
        if target.iter().any(|value| !value.is_finite()) {
            return Err(
                "SaeSupportSparseTerm::reroute_fixed_decoder: target contains a non-finite value"
                    .into(),
            );
        }

        struct Candidate {
            atom: usize,
            score: f64,
            coords: Vec<f64>,
        }
        let better = |left: &Candidate, right: &Candidate| {
            left.score > right.score || (left.score == right.score && left.atom < right.atom)
        };
        let mut indices = Vec::with_capacity(target.nrows());
        let mut gate_params = Vec::with_capacity(target.nrows());
        let mut coords = Vec::with_capacity(target.nrows());
        for row in target.rows() {
            let row_values = row.as_slice().ok_or_else(|| {
                "SaeSupportSparseTerm::reroute_fixed_decoder: target row is not contiguous"
                    .to_string()
            })?;
            let mut selected = Vec::<Candidate>::with_capacity(support_k);
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                let candidate_coords = (0..atom.latent_dim)
                    .map(|axis| {
                        let raw = super::support_seed::projection(
                            row_values,
                            atom_index,
                            axis + 1,
                            random_state,
                        );
                        super::support_seed::chart_coordinate(&atom.basis_kind, axis, raw)
                    })
                    .collect::<Vec<_>>();
                let coordinate =
                    Array2::from_shape_vec((1, atom.latent_dim), candidate_coords.clone())
                        .map_err(|error| {
                            format!("SaeSupportSparseTerm::reroute_fixed_decoder: {error}")
                        })?;
                let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                    format!(
                        "SaeSupportSparseTerm::reroute_fixed_decoder: atom {atom_index} has no evaluator"
                    )
                })?;
                let (phi, _) = evaluator.evaluate(coordinate.view())?;
                let decoded = phi.row(0).dot(&atom.decoder_coefficients);
                let score = row
                    .iter()
                    .zip(decoded.iter())
                    .map(|(truth, fit)| 2.0 * truth * fit - fit * fit)
                    .sum::<f64>();
                let candidate = Candidate {
                    atom: atom_index,
                    score,
                    coords: candidate_coords,
                };
                if selected.len() < support_k {
                    selected.push(candidate);
                } else {
                    let mut worst = 0usize;
                    for slot in 1..selected.len() {
                        if better(&selected[worst], &selected[slot]) {
                            worst = slot;
                        }
                    }
                    if better(&candidate, &selected[worst]) {
                        selected[worst] = candidate;
                    }
                }
            }
            selected.sort_by_key(|candidate| candidate.atom);
            indices.push(
                selected
                    .iter()
                    .map(|candidate| candidate.atom as u32)
                    .collect(),
            );
            gate_params.push(selected.iter().map(|candidate| candidate.score).collect());
            coords.push(
                selected
                    .into_iter()
                    .flat_map(|candidate| candidate.coords)
                    .collect(),
            );
        }
        let atom_specs = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, template)| SaeAssignmentAtomSpec {
                latent_dim: template.latent_dim,
                id_mode: gam_terms::latent::LatentIdMode::None,
                manifold: template.basis_kind.latent_manifold(template.latent_dim),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
                latent_id: super::support_seed::splitmix64(atom as u64),
            })
            .collect();
        let assignment = SaeAssignmentState::from_topk_support_heterogeneous(
            target.nrows(),
            self.k_atoms(),
            support_k,
            atom_specs,
            indices,
            gate_params,
            coords,
        )?;
        Self::new(self.atoms.clone(), assignment)
    }

    pub(crate) fn beta_layout(&self) -> Result<(Vec<usize>, usize), String> {
        let mut offsets = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            offsets.push(cursor);
            cursor =
                cursor
                    .checked_add(atom.basis_size().checked_mul(self.output_dim).ok_or_else(
                        || "SaeSupportSparseTerm: beta block width overflow".to_string(),
                    )?)
                    .ok_or_else(|| "SaeSupportSparseTerm: beta dimension overflow".to_string())?;
        }
        Ok((offsets, cursor))
    }

    /// Assemble the exact support-row Gauss-Newton Arrow system. `H_bb` and
    /// every `H_tb` row are installed as sparse matvec/adjoint operators; the
    /// only resident row matrices are `q_i×q_i`, with
    /// `q_i = sum_{k in S_i} d_k`.
    pub fn assemble_arrow_schur(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<ArrowSchurSystem, String> {
        if target.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::assemble_arrow_schur: target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        self.validate_smoothing(lambda_smooth)?;
        if ard_precisions.len() != self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm::assemble_arrow_schur: ARD blocks {} != K={}",
                ard_precisions.len(),
                self.k_atoms()
            ));
        }
        for (atom, values) in ard_precisions.iter().enumerate() {
            if values.len() != self.assignment.atom_coord_dim(atom)
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(format!(
                    "SaeSupportSparseTerm::assemble_arrow_schur: atom {atom} ARD must contain {} finite positive precisions",
                    self.assignment.atom_coord_dim(atom)
                ));
            }
        }
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        let per_row_dims = (0..self.n_obs())
            .map(|row| self.assignment.coords_row(row).len())
            .collect::<Vec<_>>();
        let mut system = ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(
            per_row_dims,
            beta_dim,
            0,
        );
        let mut linearized_rows = Vec::with_capacity(self.n_obs());
        let mut hbb_diag = Array1::<f64>::zeros(beta_dim);
        for row in 0..self.n_obs() {
            let q = self.assignment.coords_row(row).len();
            let mut fitted = Array1::<f64>::zeros(self.output_dim);
            let mut jacobian = Array2::<f64>::zeros((q, self.output_dim));
            let mut blocks = Vec::with_capacity(self.assignment.support_indices(row).len());
            let mut cursor = 0usize;
            for slot in 0..self.assignment.support_indices(row).len() {
                let atom_idx = self.assignment.support_indices(row)[slot] as usize;
                let active = self.evaluate_active(row, slot)?;
                fitted += &active.decoded;
                for axis in 0..active.jacobian.nrows() {
                    jacobian
                        .row_mut(cursor + axis)
                        .assign(&active.jacobian.row(axis));
                }
                for basis in 0..active.phi.len() {
                    let base = beta_offsets[atom_idx] + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        hbb_diag[base + channel] += active.phi[basis] * active.phi[basis];
                    }
                }
                blocks.push(SupportBasisBlock {
                    beta_offset: beta_offsets[atom_idx],
                    phi: active.phi,
                });
                cursor += active.jacobian.nrows();
            }
            let residual = &target.row(row) - &fitted;
            system.rows[row].htt.assign(&jacobian.dot(&jacobian.t()));
            system.rows[row].gt.assign(&(-jacobian.dot(&residual)));
            let periods = self
                .assignment
                .support_indices(row)
                .iter()
                .flat_map(|&atom| self.assignment.atom_manifold(atom as usize).axis_periods())
                .collect::<Vec<_>>();
            let mut coord_cursor = 0usize;
            for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                let atom = atom as usize;
                for axis in 0..self.assignment.atom_coord_dim(atom) {
                    let coordinate = self.assignment.coords_for_slot(row, slot)[axis];
                    let prior = ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        coordinate,
                        periods[coord_cursor],
                    );
                    system.rows[row].gt[coord_cursor] += prior.grad;
                    system.rows[row].htt[[coord_cursor, coord_cursor]] +=
                        prior.psd_majorizer_hess();
                    coord_cursor += 1;
                }
            }
            for block in &blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        system.gb[base + channel] -= block.phi[basis] * residual[channel];
                    }
                }
            }
            linearized_rows.push(SupportLinearizedRow { blocks, jacobian });
        }
        for atom in 0..self.k_atoms() {
            let m = self.atoms[atom].basis_size();
            let lambda = lambda_smooth[atom];
            let sb = self.atoms[atom]
                .smooth_penalty
                .dot(&self.atoms[atom].decoder_coefficients);
            for basis in 0..m {
                let base = beta_offsets[atom] + basis * self.output_dim;
                for channel in 0..self.output_dim {
                    system.gb[base + channel] += lambda * sb[[basis, channel]];
                    hbb_diag[base + channel] +=
                        lambda * self.atoms[atom].smooth_penalty[[basis, basis]];
                }
            }
        }
        let operator = Arc::new(SupportBetaOperator {
            rows: linearized_rows,
            beta_offsets: beta_offsets.clone(),
            basis_sizes: self.atoms.iter().map(SaeManifoldAtom::basis_size).collect(),
            penalties: self
                .atoms
                .iter()
                .map(|atom| atom.smooth_penalty.clone())
                .collect(),
            lambda_smooth: lambda_smooth.to_vec(),
            output_dim: self.output_dim,
            beta_dim,
        });
        let shared = Arc::clone(&operator);
        system.set_shared_beta_operator(move |vector, out| shared.apply(vector, out), hbb_diag);
        let forward = Arc::clone(&operator);
        let transpose = Arc::clone(&operator);
        system.set_row_htbeta_operator(
            move |row, vector, out| forward.htbeta_forward(row, vector, out),
            move |row, vector, out| transpose.htbeta_transpose(row, vector, out),
        );
        let block_offsets: Arc<[Range<usize>]> = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, template)| {
                beta_offsets[atom]..beta_offsets[atom] + template.basis_size() * self.output_dim
            })
            .collect::<Vec<_>>()
            .into();
        system.set_block_offsets(block_offsets);
        system.refresh_row_hessian_fingerprint();
        Ok(system)
    }

    fn evaluate_active(&self, row: usize, slot: usize) -> Result<ActiveAtomEval, String> {
        let atom_idx = self.assignment.support_indices(row)[slot] as usize;
        let atom = &self.atoms[atom_idx];
        let d = atom.latent_dim;
        let coords =
            Array2::from_shape_vec((1, d), self.assignment.coords_for_slot(row, slot).to_vec())
                .map_err(|error| format!("SaeSupportSparseTerm::evaluate_active: {error}"))?;
        let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
            format!("SaeSupportSparseTerm::evaluate_active: atom {atom_idx} has no evaluator")
        })?;
        let (phi, jet) = evaluator.evaluate(coords.view())?;
        let m = atom.basis_size();
        if phi.dim() != (1, m) || jet.dim() != (1, m, d) {
            return Err(format!(
                "SaeSupportSparseTerm::evaluate_active: atom {atom_idx} evaluator shapes Phi={:?}, jet={:?}, expected (1,{m}) and (1,{m},{d})",
                phi.dim(),
                jet.dim()
            ));
        }
        let phi = phi.row(0).to_owned();
        let decoded = phi.dot(&atom.decoder_coefficients);
        let mut jacobian = Array2::<f64>::zeros((d, self.output_dim));
        for axis in 0..d {
            for basis in 0..m {
                let weight = jet[[0, basis, axis]];
                for output in 0..self.output_dim {
                    jacobian[[axis, output]] += weight * atom.decoder_coefficients[[basis, output]];
                }
            }
        }
        Ok(ActiveAtomEval {
            phi,
            decoded,
            jacobian,
        })
    }

    fn reconstruct_row(&self, row: usize) -> Result<Array1<f64>, String> {
        let mut fitted = Array1::<f64>::zeros(self.output_dim);
        for slot in 0..self.assignment.support_indices(row).len() {
            let active = self.evaluate_active(row, slot)?;
            fitted += &active.decoded;
        }
        Ok(fitted)
    }

    /// Direct active-row reconstruction. No K-wide gate or basis row exists.
    pub fn reconstruct(&self) -> Result<Array2<f64>, String> {
        let mut fitted = Array2::<f64>::zeros((self.n_obs(), self.output_dim));
        for row in 0..self.n_obs() {
            fitted.row_mut(row).assign(&self.reconstruct_row(row)?);
        }
        Ok(fitted)
    }

    /// Raw response residual `target - fitted`, deliberately before any
    /// smoothing or coordinate-prior transformation.
    pub fn raw_residual(&self, target: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if target.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::raw_residual: target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        Ok(&target - &self.reconstruct()?)
    }

    fn validate_smoothing(&self, lambda_smooth: &[f64]) -> Result<(), String> {
        if lambda_smooth.len() != self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm: smoothing length {} != K={}",
                lambda_smooth.len(),
                self.k_atoms()
            ));
        }
        if lambda_smooth
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "SaeSupportSparseTerm: smoothing strengths must be finite and non-negative".into(),
            );
        }
        Ok(())
    }

    fn validate_ard(&self, ard_precisions: &[Vec<f64>]) -> Result<(), String> {
        if ard_precisions.len() != self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm: ARD blocks {} != K={}",
                ard_precisions.len(),
                self.k_atoms()
            ));
        }
        for (atom, values) in ard_precisions.iter().enumerate() {
            if values.len() != self.assignment.atom_coord_dim(atom)
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(format!(
                    "SaeSupportSparseTerm: atom {atom} ARD must contain {} finite positive precisions",
                    self.assignment.atom_coord_dim(atom)
                ));
            }
        }
        Ok(())
    }

    /// Gaussian loss plus the declared final-function seminorm
    /// `0.5 λ_k tr(B_k' S_ref,k B_k)`.
    pub fn penalized_objective(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        self.validate_smoothing(lambda_smooth)?;
        self.validate_ard(ard_precisions)?;
        let residual = self.raw_residual(target)?;
        let mut value = 0.5 * residual.iter().map(|entry| entry * entry).sum::<f64>();
        for (atom, &lambda) in self.atoms.iter().zip(lambda_smooth) {
            let sb = atom.smooth_penalty.dot(&atom.decoder_coefficients);
            value += 0.5
                * lambda
                * atom
                    .decoder_coefficients
                    .iter()
                    .zip(sb.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
        }
        for row in 0..self.n_obs() {
            for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                let atom = atom as usize;
                let periods = self.assignment.atom_manifold(atom).axis_periods();
                for axis in 0..self.assignment.atom_coord_dim(atom) {
                    value += ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        self.assignment.coords_for_slot(row, slot)[axis],
                        periods[axis],
                    )
                    .value;
                }
            }
        }
        if value.is_finite() {
            Ok(value)
        } else {
            Err("SaeSupportSparseTerm::penalized_objective is non-finite".into())
        }
    }

    /// Canonical Moore-Penrose solution of a symmetric PSD normal equation.
    /// Null directions are set to zero; an RHS component in the numerical null
    /// space is a malformed normal equation and is refused.
    fn solve_psd_minimum_norm(
        gram: &Array2<f64>,
        rhs: &Array2<f64>,
        context: &str,
    ) -> Result<Array2<f64>, String> {
        let m = gram.nrows();
        if gram.dim() != (m, m) || rhs.nrows() != m {
            return Err(format!(
                "{context}: normal-equation shape mismatch gram={:?}, rhs={:?}",
                gram.dim(),
                rhs.dim()
            ));
        }
        let symmetric = (gram + &gram.t()) * 0.5;
        let (eigenvalues, eigenvectors) = symmetric
            .eigh(Side::Lower)
            .map_err(|error| format!("{context}: eigendecomposition failed: {error}"))?;
        let scale = eigenvalues.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let tolerance = f64::EPSILON.sqrt() * scale * m.max(1) as f64;
        if eigenvalues.iter().any(|value| *value < -tolerance) {
            return Err(format!(
                "{context}: normal equation is not positive semidefinite"
            ));
        }
        let projected = eigenvectors.t().dot(rhs);
        let rhs_scale = rhs.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let mut scaled = Array2::<f64>::zeros(projected.dim());
        for mode in 0..m {
            if eigenvalues[mode] > tolerance {
                for column in 0..rhs.ncols() {
                    scaled[[mode, column]] = projected[[mode, column]] / eigenvalues[mode];
                }
            } else if projected
                .row(mode)
                .iter()
                .any(|value| value.abs() > tolerance * rhs_scale)
            {
                return Err(format!(
                    "{context}: RHS has a component in the normal-equation null space"
                ));
            }
        }
        Ok(eigenvectors.dot(&scaled))
    }

    /// One deterministic Gauss-Seidel decoder sweep. Each block update is the
    /// exact minimum-norm minimizer of the current final-function-penalized
    /// quadratic, not a coefficient-ridge surrogate.
    fn decoder_sweep(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
    ) -> Result<f64, String> {
        self.validate_smoothing(lambda_smooth)?;
        let mut fitted = self.reconstruct()?;
        let mut max_change = 0.0_f64;
        for atom_idx in 0..self.k_atoms() {
            let m = self.atoms[atom_idx].basis_size();
            let old_decoder = self.atoms[atom_idx].decoder_coefficients.clone();
            let mut gram = self.atoms[atom_idx].smooth_penalty.clone() * lambda_smooth[atom_idx];
            let mut rhs = Array2::<f64>::zeros((m, self.output_dim));
            let mut rows = Vec::with_capacity(self.atom_rows[atom_idx].len());
            for &(row, slot) in &self.atom_rows[atom_idx] {
                let active = self.evaluate_active(row, slot)?;
                for left in 0..m {
                    for right in 0..m {
                        gram[[left, right]] += active.phi[left] * active.phi[right];
                    }
                    for output in 0..self.output_dim {
                        let residual_without =
                            target[[row, output]] - fitted[[row, output]] + active.decoded[output];
                        rhs[[left, output]] += active.phi[left] * residual_without;
                    }
                }
                rows.push((row, active.phi, active.decoded));
            }
            let decoder =
                Self::solve_psd_minimum_norm(&gram, &rhs, "SaeSupportSparseTerm::decoder_sweep")?;
            for (new, old) in decoder.iter().zip(old_decoder.iter()) {
                max_change = max_change.max((new - old).abs());
            }
            self.atoms[atom_idx].decoder_coefficients = decoder;
            for (row, phi, old_decoded) in rows {
                let new_decoded = phi.dot(&self.atoms[atom_idx].decoder_coefficients);
                for output in 0..self.output_dim {
                    fitted[[row, output]] += new_decoded[output] - old_decoded[output];
                }
            }
        }
        Ok(max_change)
    }

    /// One direct active-row Gauss-Newton coordinate sweep with manifold-aware
    /// backtracking. Exact row snapshots provide rollback; inverse retractions
    /// are never assumed.
    fn coordinate_sweep(
        &mut self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        trust_radius: f64,
    ) -> Result<f64, String> {
        self.validate_ard(ard_precisions)?;
        if !(trust_radius.is_finite() && trust_radius > 0.0) {
            return Err(format!(
                "SaeSupportSparseTerm::coordinate_sweep: trust_radius must be finite and positive; got {trust_radius}"
            ));
        }
        let mut max_change = 0.0_f64;
        for row in 0..self.n_obs() {
            let q = self.assignment.coords_row(row).len();
            let mut fitted = Array1::<f64>::zeros(self.output_dim);
            let mut jacobian = Array2::<f64>::zeros((q, self.output_dim));
            let mut cursor = 0;
            for slot in 0..self.assignment.support_indices(row).len() {
                let active = self.evaluate_active(row, slot)?;
                fitted += &active.decoded;
                let d = active.jacobian.nrows();
                for axis in 0..d {
                    jacobian
                        .row_mut(cursor + axis)
                        .assign(&active.jacobian.row(axis));
                }
                cursor += d;
            }
            let residual = &target.row(row) - &fitted;
            let mut rhs_vector = jacobian.dot(&residual);
            let mut gram = jacobian.dot(&jacobian.t());
            let mut old_prior = 0.0;
            let mut prior_cursor = 0usize;
            for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                let atom = atom as usize;
                let periods = self.assignment.atom_manifold(atom).axis_periods();
                for axis in 0..self.assignment.atom_coord_dim(atom) {
                    let prior = ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        self.assignment.coords_for_slot(row, slot)[axis],
                        periods[axis],
                    );
                    old_prior += prior.value;
                    rhs_vector[prior_cursor] -= prior.grad;
                    gram[[prior_cursor, prior_cursor]] += prior.psd_majorizer_hess();
                    prior_cursor += 1;
                }
            }
            let rhs = rhs_vector
                .clone()
                .into_shape_with_order((q, 1))
                .map_err(|error| format!("SaeSupportSparseTerm::coordinate_sweep: {error}"))?;
            let mut delta = Self::solve_psd_minimum_norm(
                &gram,
                &rhs,
                "SaeSupportSparseTerm::coordinate_sweep",
            )?
            .column(0)
            .to_owned();
            let norm = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
            if norm > trust_radius {
                delta.mapv_inplace(|value| value * trust_radius / norm);
            }
            let directional = rhs_vector.dot(&delta);
            if directional <= f64::EPSILON {
                continue;
            }
            let old_coords = self.assignment.coords_row(row).to_vec();
            let old_loss =
                0.5 * residual.iter().map(|value| value * value).sum::<f64>() + old_prior;
            let mut accepted = None;
            for halving in 0..=24 {
                self.assignment.set_row_coords(row, &old_coords)?;
                let step = 2.0_f64.powi(-(halving as i32));
                let trial_delta = delta.iter().map(|value| step * value).collect::<Vec<_>>();
                self.assignment.apply_row_coord_step(row, &trial_delta)?;
                let trial_fitted = self.reconstruct_row(row)?;
                let mut trial_loss = 0.5
                    * target
                        .row(row)
                        .iter()
                        .zip(trial_fitted.iter())
                        .map(|(truth, fit)| (truth - fit).powi(2))
                        .sum::<f64>();
                for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                    let atom = atom as usize;
                    let periods = self.assignment.atom_manifold(atom).axis_periods();
                    for axis in 0..self.assignment.atom_coord_dim(atom) {
                        trial_loss += ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            self.assignment.coords_for_slot(row, slot)[axis],
                            periods[axis],
                        )
                        .value;
                    }
                }
                if trial_loss.is_finite() && trial_loss <= old_loss - 1.0e-4 * step * directional {
                    accepted = Some(step);
                    break;
                }
            }
            match accepted {
                Some(step) => {
                    for value in delta.iter() {
                        max_change = max_change.max((step * value).abs());
                    }
                }
                None => {
                    self.assignment.set_row_coords(row, &old_coords)?;
                    return Err(format!(
                        "SaeSupportSparseTerm::coordinate_sweep: row {row} has a raw descent direction but manifold line search found no decreasing step"
                    ));
                }
            }
        }
        Ok(max_change)
    }

    /// Raw (undamped) KKT residual of the exact objective.
    pub fn raw_stationarity(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<SaeSupportStationarity, String> {
        self.validate_smoothing(lambda_smooth)?;
        self.validate_ard(ard_precisions)?;
        let residual = self.raw_residual(target)?;
        let mut decoder_sq = 0.0;
        let mut decoder_max = 0.0_f64;
        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            let mut gradient =
                atom.smooth_penalty.dot(&atom.decoder_coefficients) * lambda_smooth[atom_idx];
            for &(row, slot) in &self.atom_rows[atom_idx] {
                let active = self.evaluate_active(row, slot)?;
                for basis in 0..atom.basis_size() {
                    for output in 0..self.output_dim {
                        gradient[[basis, output]] -= active.phi[basis] * residual[[row, output]];
                    }
                }
            }
            for value in gradient {
                decoder_sq += value * value;
                decoder_max = decoder_max.max(value.abs());
            }
        }
        let mut coordinate_sq = 0.0;
        let mut coordinate_max = 0.0_f64;
        for row in 0..self.n_obs() {
            for slot in 0..self.assignment.support_indices(row).len() {
                let atom = self.assignment.support_indices(row)[slot] as usize;
                let active = self.evaluate_active(row, slot)?;
                let periods = self.assignment.atom_manifold(atom).axis_periods();
                for axis in 0..active.jacobian.nrows() {
                    let mut gradient = 0.0;
                    for output in 0..self.output_dim {
                        gradient -= active.jacobian[[axis, output]] * residual[[row, output]];
                    }
                    gradient += ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        self.assignment.coords_for_slot(row, slot)[axis],
                        periods[axis],
                    )
                    .grad;
                    coordinate_sq += gradient * gradient;
                    coordinate_max = coordinate_max.max(gradient.abs());
                }
            }
        }
        Ok(SaeSupportStationarity {
            decoder_l2: decoder_sq.sqrt(),
            decoder_max_abs: decoder_max,
            coordinate_l2: coordinate_sq.sqrt(),
            coordinate_max_abs: coordinate_max,
        })
    }

    /// Raw coordinate KKT residual with decoder coefficients held fixed.
    pub fn raw_coordinate_stationarity(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<(f64, f64), String> {
        self.validate_ard(ard_precisions)?;
        let residual = self.raw_residual(target)?;
        let mut coordinate_sq = 0.0;
        let mut coordinate_max = 0.0_f64;
        for row in 0..self.n_obs() {
            for slot in 0..self.assignment.support_indices(row).len() {
                let atom = self.assignment.support_indices(row)[slot] as usize;
                let active = self.evaluate_active(row, slot)?;
                let periods = self.assignment.atom_manifold(atom).axis_periods();
                for axis in 0..active.jacobian.nrows() {
                    let likelihood_gradient = active
                        .jacobian
                        .row(axis)
                        .iter()
                        .zip(residual.row(row).iter())
                        .map(|(jet, error)| -jet * error)
                        .sum::<f64>();
                    let gradient = likelihood_gradient
                        + ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            self.assignment.coords_for_slot(row, slot)[axis],
                            periods[axis],
                        )
                        .grad;
                    coordinate_sq += gradient * gradient;
                    coordinate_max = coordinate_max.max(gradient.abs());
                }
            }
        }
        Ok((coordinate_sq.sqrt(), coordinate_max))
    }

    fn frozen_decoder_coordinate_objective(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        let residual = self.raw_residual(target)?;
        let mut objective = 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
        for row in 0..self.n_obs() {
            for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                let atom = atom as usize;
                let periods = self.assignment.atom_manifold(atom).axis_periods();
                for axis in 0..self.assignment.atom_coord_dim(atom) {
                    objective += ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        self.assignment.coords_for_slot(row, slot)[axis],
                        periods[axis],
                    )
                    .value;
                }
            }
        }
        if objective.is_finite() {
            Ok(objective)
        } else {
            Err("SaeSupportSparseTerm::frozen_decoder_coordinate_objective is non-finite".into())
        }
    }

    /// Frozen-decoder OOS coordinate solve over active supports only. A
    /// budget-exhausted or merely damped point is rejected; the returned state
    /// has recurred for two full raw-stationary coordinate cycles.
    pub fn solve_coordinates_fixed_decoder(
        &mut self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        max_iter: usize,
        tolerance: f64,
        trust_radius: f64,
    ) -> Result<SaeSupportCoordinateFixedPointReport, String> {
        if target.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::solve_coordinates_fixed_decoder: target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        if max_iter == 0 || !(tolerance.is_finite() && tolerance > 0.0) {
            return Err("SaeSupportSparseTerm::solve_coordinates_fixed_decoder requires positive max_iter and finite positive tolerance".into());
        }
        let mut previous_candidate = false;
        for iteration in 1..=max_iter {
            let max_change = self.coordinate_sweep(target, ard_precisions, trust_radius)?;
            let (coordinate_l2, coordinate_max_abs) =
                self.raw_coordinate_stationarity(target, ard_precisions)?;
            let candidate = max_change <= tolerance && coordinate_max_abs <= tolerance;
            if candidate && previous_candidate {
                return Ok(SaeSupportCoordinateFixedPointReport {
                    iterations: iteration,
                    objective: self.frozen_decoder_coordinate_objective(target, ard_precisions)?,
                    coordinate_l2,
                    coordinate_max_abs,
                    max_recurrence_change: max_change,
                    recurred: true,
                });
            }
            previous_candidate = candidate;
        }
        let (_, coordinate_max_abs) = self.raw_coordinate_stationarity(target, ard_precisions)?;
        Err(format!(
            "SaeSupportSparseTerm::solve_coordinates_fixed_decoder did not recur within {max_iter} cycles (raw coordinate KKT max={coordinate_max_abs:.6e})"
        ))
    }

    /// Alternate exact decoder blocks and direct active-row coordinate Newton
    /// steps until the raw KKT residual AND a full-cycle recurrence agree. A
    /// budget-exhausted iterate is an error; only converged fits are returned.
    pub fn solve_fixed_point(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        max_iter: usize,
        tolerance: f64,
        trust_radius: f64,
    ) -> Result<SaeSupportFixedPointReport, String> {
        if target.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::solve_fixed_point: target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        if max_iter == 0 || !(tolerance.is_finite() && tolerance > 0.0) {
            return Err("SaeSupportSparseTerm::solve_fixed_point requires positive max_iter and finite positive tolerance".into());
        }
        let mut previous_candidate = false;
        for iteration in 1..=max_iter {
            let decoder_change = self.decoder_sweep(target, lambda_smooth)?;
            let coordinate_change = self.coordinate_sweep(target, ard_precisions, trust_radius)?;
            let max_change = decoder_change.max(coordinate_change);
            let stationarity = self.raw_stationarity(target, lambda_smooth, ard_precisions)?;
            let candidate = max_change <= tolerance && stationarity.max_abs() <= tolerance;
            if candidate && previous_candidate {
                return Ok(SaeSupportFixedPointReport {
                    iterations: iteration,
                    objective: self.penalized_objective(target, lambda_smooth, ard_precisions)?,
                    stationarity,
                    max_recurrence_change: max_change,
                    recurred: true,
                });
            }
            previous_candidate = candidate;
        }
        let stationarity = self.raw_stationarity(target, lambda_smooth, ard_precisions)?;
        Err(format!(
            "SaeSupportSparseTerm::solve_fixed_point did not recur within {max_iter} cycles (raw KKT max={:.6e})",
            stationarity.max_abs()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment_state::SaeAssignmentAtomSpec;
    use ndarray::array;
    use std::sync::Arc;

    fn atom(
        name: &str,
        kind: SaeAtomBasisKind,
        d: usize,
        evaluator: Arc<dyn SaeBasisSecondJet>,
        coords: &[f64],
        decoder: Array2<f64>,
    ) -> SaeManifoldAtom {
        let coord = Array2::from_shape_vec((1, d), coords.to_vec()).expect("coords");
        let (phi, jet) = evaluator.evaluate(coord.view()).expect("evaluate");
        let m = phi.ncols();
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            kind,
            d,
            phi,
            jet,
            decoder,
            Array2::eye(m),
        )
        .expect("atom")
        .with_basis_second_jet(evaluator)
    }

    #[test]
    fn direct_reconstruction_uses_only_heterogeneous_support() {
        let periodic_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
        let patch_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(2, 1).expect("patch"));
        let atoms = vec![
            atom(
                "circle",
                SaeAtomBasisKind::Periodic,
                1,
                periodic_eval,
                &[0.0],
                array![[0.0], [1.0], [0.0]],
            ),
            atom(
                "plane",
                SaeAtomBasisKind::Linear,
                2,
                patch_eval,
                &[0.0, 0.0],
                array![[0.0], [2.0], [-1.0]],
            ),
        ];
        let specs = vec![
            SaeAssignmentAtomSpec {
                latent_dim: 1,
                id_mode: LatentIdMode::None,
                manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
                latent_id: 1,
            },
            SaeAssignmentAtomSpec::euclidean(2),
        ];
        let state = SaeAssignmentState::from_topk_support_heterogeneous(
            2,
            2,
            1,
            specs,
            vec![vec![0], vec![1]],
            vec![vec![9.0], vec![-4.0]],
            vec![vec![0.25], vec![3.0, 1.0]],
        )
        .expect("state");
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let fitted = term.reconstruct().expect("reconstruct");
        assert!((fitted[[0, 0]] - 1.0).abs() < 1.0e-12);
        assert!((fitted[[1, 0]] - 5.0).abs() < 1.0e-12);
        assert_eq!(term.active_pair_count(), 2);
    }

    #[test]
    fn decoder_sweep_decreases_final_function_objective() {
        let evaluator: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
        let atoms = vec![atom(
            "line",
            SaeAtomBasisKind::Linear,
            1,
            evaluator,
            &[0.0],
            Array2::zeros((2, 1)),
        )];
        let state = SaeAssignmentState::from_topk_support(
            3,
            1,
            1,
            1,
            vec![vec![0]; 3],
            vec![vec![1.0]; 3],
            vec![vec![-1.0], vec![0.0], vec![1.0]],
        )
        .expect("state");
        let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let target = array![[-1.0], [0.0], [1.0]];
        let ard = vec![vec![1.0]];
        let before = term
            .penalized_objective(target.view(), &[0.1], &ard)
            .expect("before");
        term.decoder_sweep(target.view(), &[0.1]).expect("sweep");
        let after = term
            .penalized_objective(target.view(), &[0.1], &ard)
            .expect("after");
        assert!(after < before);
        assert!(
            term.raw_stationarity(target.view(), &[0.1], &ard)
                .expect("kkt")
                .decoder_max_abs
                < 1.0e-10
        );
    }
}
