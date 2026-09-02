//! Canonical support-sparse curved term and fixed-point inner solve.
//!
//! Hard-TopK gates are read-only binary support. Consequently a row's only
//! live local parameters are the heterogeneous coordinates
//! `concat_{k in S_i} t_ik`; no gate/logit coordinate exists. This term owns
//! that representation directly and evaluates basis values and analytic jets
//! only for active `(row, atom)` pairs.

use crate::assignment::AssignmentMode;
use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
use gam_linalg::anderson::AndersonAccelerator;
use gam_linalg::utils::KahanSum;
use gam_solve::arrow_schur::reduced_schur_inverse_apply;
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

use super::*;

/// Rows per rayon task in the read-only active-set passes (#2575).
///
/// The unit of work is a row, but the unit of ALLOCATION should not be: each
/// task builds its evaluation scratch once and reuses it across the rows it
/// takes, so the chunk width sets how many rows amortise one scratch. Wide
/// enough that the per-task setup is negligible against the `support_k · M · P`
/// work per row, narrow enough that a 4-core host still gets even load at the
/// smallest shapes the lane admits.
const RECONSTRUCT_ROW_CHUNK: usize = 64;

/// Order of the Anderson multisecant model on the support fixed point (#2575).
///
/// This is a COST bound, not a tuning knob. The accelerator drops every
/// difference column whose contribution is below its own roundoff floor, so a
/// history longer than the map's informative secant subspace costs memory and
/// buys nothing rather than mispricing anything — which is why the depth can be
/// declared here instead of derived from the problem. What it bounds:
/// `2·depth·(N·support_k)` doubles of history, and an `order × order`
/// eigendecomposition per cycle, both negligible against one sweep's
/// `N·support_k·M·P` work.
///
/// Eight is the upper end of the range the literature reports gains over
/// (Walker & Ni, *SINUM* 2011, §4; Fang & Saad, *NLAA* 2009): past it, the
/// stored differences on a slowly-contracting map are numerically dependent and
/// the extra columns are exactly the ones the roundoff floor discards.
const SUPPORT_ANDERSON_DEPTH: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaeSupportStationarity {
    pub decoder_l2: f64,
    pub decoder_max_abs: f64,
    pub coordinate_l2: f64,
    pub coordinate_max_abs: f64,
    /// The decoder block's gradient divided by that block's OWN curvature
    /// diagonal — the Jacobi-scaled Newton step, i.e. how far the coefficients
    /// still have to move, in the units the coefficients live in.
    ///
    /// #2517. The raw gradient is not a free-floating number: the decoder sweep
    /// solves `(G_k + λS_k) B_k = rhs_k` exactly per atom, so near the fixed
    /// point `g ≈ (G + λS)·Δ`, and `G_k = Σ_rows φφᵀ` is a sum over the atom's
    /// OWN rows. The gradient is therefore the parameter error multiplied by
    /// **rows-per-atom** — measured at 12x to 75x across two decades of shape —
    /// and an absolute (or objective-relative) threshold on it is a threshold
    /// on `m·Δ`, which no amount of data quality can reach. Shrinking the
    /// residual does not help, because the extensivity lives in the Gram and
    /// not in `Σ_rows φ⊗r`: the in-class, 1e-4-residual arm stalls at the same
    /// order as the noisy one.
    ///
    /// Dividing by the curvature diagonal removes exactly that factor and
    /// leaves a quantity invariant to `n`, to rows-per-atom, and to basis
    /// scaling — the same domain-space discipline as #2548's per-block split.
    pub decoder_scaled_max_abs: f64,
    /// The coordinate block's counterpart: its gradient divided by its own
    /// curvature diagonal (`Σ_out J² + ARD curvature`), so both blocks are
    /// certified in the space their parameters live in rather than in two
    /// different gradient scales.
    pub coordinate_scaled_max_abs: f64,
}

impl SaeSupportStationarity {
    /// The raw (gradient-space) certificate, kept for reporting and for every
    /// consumer that compares against a historical number.
    pub fn max_abs(self) -> f64 {
        self.decoder_max_abs.max(self.coordinate_max_abs)
    }

    /// The parameter-space certificate: the larger of the two blocks' scaled
    /// Newton steps. This is what a fixed point should be certified on — see
    /// [`Self::decoder_scaled_max_abs`] for why the raw gradient cannot be.
    pub fn scaled_max_abs(self) -> f64 {
        self.decoder_scaled_max_abs
            .max(self.coordinate_scaled_max_abs)
    }

    /// Accept either of the two mathematically valid first-order currencies:
    /// the extensive raw gradient relative to the objective, or the
    /// componentwise curvature-scaled displacement relative to the iterate.
    pub fn kkt_certifies(
        self,
        objective_scale: f64,
        parameter_scale: f64,
        tolerance: f64,
    ) -> bool {
        objective_scale.is_finite()
            && objective_scale >= 1.0
            && parameter_scale.is_finite()
            && parameter_scale >= 1.0
            && tolerance.is_finite()
            && tolerance > 0.0
            && (self.max_abs() <= tolerance * objective_scale
                || self.scaled_max_abs() <= tolerance * parameter_scale)
    }
}

/// Typed refusal to mint a support-term stationarity certificate. Evaluation
/// failures and undefined parameter-space scales remain distinguishable to
/// callers; neither is converted into a permissive infinite bound.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeSupportStationarityError {
    Evaluation(String),
    ParameterScale(SaeInnerKktScaleError),
}

impl std::fmt::Display for SaeSupportStationarityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Evaluation(reason) => formatter.write_str(reason),
            Self::ParameterScale(reason) => write!(formatter, "parameter-space KKT unresolved: {reason}"),
        }
    }
}

impl std::error::Error for SaeSupportStationarityError {}

impl From<String> for SaeSupportStationarityError {
    fn from(reason: String) -> Self {
        Self::Evaluation(reason)
    }
}

impl From<SaeInnerKktScaleError> for SaeSupportStationarityError {
    fn from(reason: SaeInnerKktScaleError) -> Self {
        Self::ParameterScale(reason)
    }
}

impl From<SaeSupportStationarityError> for String {
    fn from(reason: SaeSupportStationarityError) -> Self {
        reason.to_string()
    }
}

fn accumulate_parameter_scaled_gradient(
    scaled_max: &mut f64,
    gradient: f64,
    curvature: f64,
    block: SaeInnerKktScaleBlock,
    component: usize,
) -> Result<(), SaeInnerKktScaleError> {
    if !gradient.is_finite() {
        return Err(SaeInnerKktScaleError::NonFiniteGradient {
            block,
            component,
            value: gradient,
        });
    }
    if !curvature.is_finite()
        || curvature < 0.0
        || (curvature == 0.0 && gradient != 0.0)
    {
        return Err(SaeInnerKktScaleError::InvalidCurvature {
            block,
            component,
            gradient,
            curvature,
        });
    }
    if curvature > 0.0 {
        let scaled = gradient.abs() / curvature;
        if !scaled.is_finite() {
            return Err(SaeInnerKktScaleError::NonFiniteScaledGradient {
                block,
                component,
                gradient,
                curvature,
            });
        }
        *scaled_max = scaled_max.max(scaled);
    }
    Ok(())
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

/// Reusable storage for ONE active `(row, slot)` evaluation (#2575).
///
/// Every read-only pass over the active set — reconstruct, the raw KKT
/// reductions, the penalized objective, the decoder sweep's normal equations,
/// the arrow assembly — needs the same four arrays for each `(row, slot)` pair:
/// the basis row `Φ(t)`, its jet, the decoded image `Φ·B`, and the coordinate
/// Jacobian `∂(Φ·B)/∂t`. The producer used to ALLOCATE all four (plus a coords
/// array and a `dot` result) on every call, and it is called `n·support_k` times
/// per sweep — 358,544 times per sweep at the #2502 flagship shape, which is
/// where the profiled 12.4% of self time in `malloc`/`free`/`memmove` went.
///
/// Held by the caller and reused across rows (per rayon worker, via
/// `map_init`), it is sized once for the first `(m, d, p)` it sees and only
/// resized when a later atom needs a different shape — so a homogeneous atom
/// portfolio allocates once per worker per sweep instead of once per pair.
/// Rows per task in the decoder sweep's row reduction. Sized so a chunk carries
/// enough work to cover task overhead while leaving many chunks per atom; the
/// value affects scheduling only, never the result, since the reduction it
/// splits is a plain sum.
const DECODER_ROW_CHUNK: usize = 512;

#[derive(Debug, Default, Clone)]
struct ActiveAtomScratch {
    /// `(1, m)` — the evaluator's own buffer shape.
    phi: Array2<f64>,
    /// `(1, m, d)`.
    jet: ndarray::Array3<f64>,
    /// `(P,)`.
    decoded: Array1<f64>,
    /// Coordinate-major decoded jet, `(d, P)`.
    jacobian: Array2<f64>,
}

impl ActiveAtomScratch {
    /// Resize to hold one `(m, d)` atom's evaluation against a `p`-wide
    /// response. A no-op when the shapes already match, which is the common
    /// case: the shapes are a property of the atom, not of the row.
    fn fit(&mut self, m: usize, d: usize, p: usize) {
        if self.phi.dim() != (1, m) {
            self.phi = Array2::zeros((1, m));
        }
        if self.jet.dim() != (1, m, d) {
            self.jet = ndarray::Array3::zeros((1, m, d));
        }
        if self.decoded.len() != p {
            self.decoded = Array1::zeros(p);
        }
        if self.jacobian.dim() != (d, p) {
            self.jacobian = Array2::zeros((d, p));
        }
    }

    /// The basis row as a flat `m`-vector view — every consumer reads it as
    /// `phi[basis]`, and the evaluator writes it as `(1, m)`.
    fn phi_row(&self) -> ndarray::ArrayView1<'_, f64> {
        self.phi.row(0)
    }
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
    /// For each atom, the `(row, block)` pairs touching it, in INCREASING row
    /// order. `apply`'s scatter walks this, and the ordering is load-bearing:
    /// it reproduces the serial sweep's accumulation order for every output
    /// element, which is what makes the fan-out bit-identical rather than
    /// merely deterministic.
    atom_blocks: Vec<Vec<(u32, u32)>>,
    beta_offsets: Vec<usize>,
    basis_sizes: Vec<usize>,
    penalties: Vec<Array2<f64>>,
    lambda_smooth: Vec<f64>,
    output_dim: usize,
    beta_dim: usize,
}

impl SupportBetaOperator {
    fn apply(&self, vector: ndarray::ArrayView1<'_, f64>, out: &mut Array1<f64>) {
        assert_eq!(
            vector.len(),
            self.beta_dim,
            "SupportBetaOperator input width must equal its declared beta dimension"
        );
        assert_eq!(
            out.len(),
            self.beta_dim,
            "SupportBetaOperator output width must equal its declared beta dimension"
        );
        use rayon::prelude::*;
        let width = self.output_dim;

        // PASS A -- gather, one independent P-wide slot per row. Each row writes
        // only its own slot, so there is no sharing to synchronise and each
        // element accumulates in the serial order.
        let mut gathered = vec![0.0_f64; self.rows.len() * width];
        gathered
            .par_chunks_mut(width)
            .zip(self.rows.par_iter())
            .for_each(|(slot, row)| {
                for block in &row.blocks {
                    for basis in 0..block.phi.len() {
                        let base = block.beta_offset + basis * width;
                        for channel in 0..width {
                            slot[channel] += block.phi[basis] * vector[base + channel];
                        }
                    }
                }
            });

        // PASS B -- scatter, one independent output block per atom. Atoms own
        // disjoint ranges of `out`, and each atom's rows are visited in
        // increasing row order, so every output element sums its contributions
        // in exactly the order the serial sweep did.
        let atom_outputs: Vec<Vec<f64>> = self
            .atom_blocks
            .par_iter()
            .enumerate()
            .map(|(atom, entries)| {
                let mut buffer = vec![0.0_f64; self.basis_sizes[atom] * width];
                for &(row_index, block_index) in entries {
                    let block = &self.rows[row_index as usize].blocks[block_index as usize];
                    let start = row_index as usize * width;
                    let slot = &gathered[start..start + width];
                    for basis in 0..block.phi.len() {
                        let target = basis * width;
                        for channel in 0..width {
                            buffer[target + channel] += block.phi[basis] * slot[channel];
                        }
                    }
                }
                buffer
            })
            .collect();

        out.fill(0.0);
        for (atom, buffer) in atom_outputs.iter().enumerate() {
            let offset = self.beta_offsets[atom];
            for (index, value) in buffer.iter().enumerate() {
                out[offset + index] += value;
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

/// Reusable storage for ONE row's coordinate solve (#2575).
///
/// Held per rayon worker and reused across every row that worker takes. The
/// row solve's working set is a function of the row's SUPPORT SHAPE — the
/// number of active slots, each slot's `(m, d)`, the compact coordinate width
/// `q` — and on this lane those are the same for almost every row (one support
/// width, one atom portfolio), so [`Self::fit`] resizes on the first row and is
/// a no-op thereafter.
#[derive(Debug, Default, Clone)]
struct RowSolveScratch {
    /// Per-slot offsets into the row's compact coordinate block.
    offsets: Vec<Range<usize>>,
    /// The row's support, in slot order.
    support: Vec<u32>,
    /// Per-slot `(basis width, latent dim)`.
    dims: Vec<(usize, usize)>,
    /// Per-slot evaluation at the CURRENT coordinates.
    current: Vec<ActiveAtomScratch>,
    /// Per-slot evaluation at the line search's trial coordinates.
    trial: Vec<ActiveAtomScratch>,
    fitted: Array1<f64>,
    /// `(q, P)` coordinate-major row Jacobian.
    jacobian: Array2<f64>,
    trial_fitted: Array1<f64>,
    trial_residual: Array1<f64>,
    trial_delta: Vec<f64>,
    fitted_delta: Vec<KahanSum>,
    old_coords: Vec<f64>,
}

impl RowSolveScratch {
    fn fit(&mut self, term: &SaeSupportSparseTerm, row: usize, q: usize, p: usize) {
        term.slot_offsets_into(row, &mut self.offsets);
        self.support.clear();
        self.support
            .extend_from_slice(term.assignment.support_indices(row));
        self.dims.clear();
        self.dims.extend(self.support.iter().map(|&atom| {
            let atom = atom as usize;
            (
                term.atoms[atom].basis_size(),
                term.atoms[atom].latent_dim(),
            )
        }));
        let slots = self.dims.len();
        self.current.resize_with(slots, ActiveAtomScratch::default);
        self.trial.resize_with(slots, ActiveAtomScratch::default);
        for (slot, &(m, d)) in self.dims.iter().enumerate() {
            self.current[slot].fit(m, d, p);
            self.trial[slot].fit(m, d, p);
        }
        if self.fitted.len() != p {
            self.fitted = Array1::zeros(p);
            self.trial_fitted = Array1::zeros(p);
            self.trial_residual = Array1::zeros(p);
            self.fitted_delta = vec![KahanSum::default(); p];
        }
        if self.jacobian.dim() != (q, p) {
            self.jacobian = Array2::zeros((q, p));
        }
        self.trial_delta.clear();
        self.trial_delta.resize(q, 0.0);
        self.old_coords.clear();
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
    /// Per-atom axis periodicity, resolved ONCE at construction (#2575).
    ///
    /// `SaeAssignmentState::atom_axis_periods` builds a fresh `Vec` on every
    /// call, and the ARD prior needs it at every `(row, slot, axis)` — including
    /// inside the coordinate line search, so up to 25 times per row per sweep.
    /// It is a property of the atom's declared manifold and retraction, both of
    /// which are fixed when the assignment state is built and are never
    /// mutated after, so resolving it per call was re-deriving a constant.
    atom_axis_periods: Vec<Vec<Option<f64>>>,
    /// `Some(passes)` selects the accelerated parallel decoder update for
    /// this term's fixed-point solves; `None` keeps the exact colour-class
    /// Gauss-Seidel sweep. See [`Self::set_decoder_fista_passes`].
    decoder_fista_passes: Option<usize>,    /// #2502 variable priced L0: when true AND pricing is armed, the router
    /// stops admitting a row's atoms once the priced gain turns non-positive
    /// (keeping at least one), instead of filling every TopK slot. The
    /// stopping rule is derived from the same description-length bill the
    /// ranking already pays -- no new constants.
    variable_priced_support: bool,
    /// When true, the admission charge is amortized over each atom's OWN
    /// firing count rather than the portfolio mean -- the usage prior, kept
    /// separable from the parameter differential because the two have
    /// opposite signs on different portfolios (+0.079 mixed, -0.167
    /// homogeneous, both measured).
    /// Rank affine atoms at their exactly-optimal coordinate during greedy
    /// selection instead of at the best grid point (#2502). Measured worth
    /// 0.0103 held-out on a linear dictionary; opt-in until an A/B confirms.
    exact_affine_ranking: bool,
    /// Multiplier on the routing grid's per-atom width (#2502). The grid gives
    /// each atom `basis_size().max(2)` candidate coordinates; this scales that
    /// count without changing the per-topology relationship it encodes. `1`
    /// reproduces the historical behaviour exactly.
    grid_refinement: usize,
    admission_usage_amortized: bool,
    /// `Some(sigma2)` arms DoF-priced admission (#2502): every support
    /// ranking expression subtracts the amortized description length of the
    /// atom's own parameters, `2*sigma2*ln2 * (m*P*(1/2)log2 N) / firings`,
    /// and the certified objective carries the matching per-used-atom charge.
    /// `None` (the default) is bit-identical to the unpriced router.
    admission_dof_sigma2: Option<f64>,
}

/// Analytic geometry needed to differentiate the support lane's profiled
/// reduced-Schur Laplace term.  It is deliberately evaluation-local: every
/// entry is tied to one converged `(term, rho)` state and cannot survive a
/// support move or an outer probe.
struct SupportOuterDifferentialSlot {
    atom: usize,
    coordinate_offset: usize,
    beta_offset: usize,
    phi: Array1<f64>,
    jet: Array2<f64>,
    second_jet: Array3<f64>,
}

struct SupportOuterDifferentialRow {
    slots: Vec<SupportOuterDifferentialSlot>,
    jacobian: Array2<f64>,
    residual: Array1<f64>,
    prior_hessian_remainder: Array1<f64>,
    prior_majorizer_derivative: Array1<f64>,
}

/// Dense representation of the exact/majorized stationarity pencil on the
/// small-problem lane.  `generalized_vectors` are B-orthonormal columns:
/// `A v_j = mu_j B v_j` and `v_j^T B v_j = 1`.
struct SupportOuterDensePencil {
    exact: Array2<f64>,
    majorizer: Array2<f64>,
    curvatures: Array1<f64>,
    generalized_vectors: Array2<f64>,
    minimum_backward_error: f64,
}

struct SupportNegativeCurvatureMode {
    curvature: f64,
    backward_error: f64,
    direction: SaeArrowVector,
}

fn support_arrow_cross_forward(
    system: &ArrowSchurSystem,
    row: usize,
    beta: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let q = system.row_dims[row];
    let block = &system.rows[row];
    let use_dense = system.htbeta_dense_supplement || system.htbeta_matvec.is_none();
    let mut out = Array1::<f64>::zeros(q);
    if use_dense {
        if block.htbeta.dim() != (q, system.k) {
            return Err(format!(
                "support outer differential: row {row} H_tbeta shape {:?} != ({q}, {})",
                block.htbeta.dim(),
                system.k,
            ));
        }
        out += &block.htbeta.dot(&beta);
    }
    if let Some(operator) = system.htbeta_matvec.as_ref() {
        operator(row, beta, &mut out);
    }
    Ok(out)
}

fn support_arrow_cross_transpose_add(
    system: &ArrowSchurSystem,
    row: usize,
    local: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) -> Result<(), String> {
    let q = system.row_dims[row];
    let block = &system.rows[row];
    let use_dense = system.htbeta_dense_supplement || system.htbeta_matvec.is_none();
    if use_dense {
        if block.htbeta.dim() != (q, system.k) {
            return Err(format!(
                "support outer differential: row {row} H_tbeta shape {:?} != ({q}, {})",
                block.htbeta.dim(),
                system.k,
            ));
        }
        *out += &block.htbeta.t().dot(&local);
    }
    if let Some(operator) = system.htbeta_transpose_matvec.as_ref() {
        operator(row, local, out);
    } else if system.htbeta_matvec.is_some() {
        return Err(format!(
            "support outer differential: row {row} has an H_tbeta operator without its transpose"
        ));
    }
    Ok(())
}

/// Apply the undamped Gauss--Newton/majorizer arrow represented by `system`.
/// The support assembler stores both shared blocks behind sparse closures, so
/// this small common apply is preferable to materialising either one.
fn support_arrow_majorizer_apply(
    system: &ArrowSchurSystem,
    vector: &SaeArrowVector,
) -> Result<SaeArrowVector, String> {
    let coordinate_dim = *system.row_offsets.last().unwrap_or(&0);
    if vector.t.len() != coordinate_dim || vector.beta.len() != system.k {
        return Err(format!(
            "support outer differential: arrow vector ({}, {}) != system ({coordinate_dim}, {})",
            vector.t.len(),
            vector.beta.len(),
            system.k,
        ));
    }
    let mut out_t = Array1::<f64>::zeros(coordinate_dim);
    let mut out_beta = Array1::<f64>::zeros(system.k);
    if let Some(operator) = system.hbb_matvec.as_ref() {
        operator(vector.beta.view(), &mut out_beta);
    } else if system.hbb.dim() == (system.k, system.k) {
        out_beta.assign(&system.hbb.dot(&vector.beta));
    } else {
        return Err(format!(
            "support outer differential: H_betabeta shape {:?} != ({}, {}) and no operator is installed",
            system.hbb.dim(),
            system.k,
            system.k,
        ));
    }
    for row in 0..system.rows.len() {
        let start = system.row_offsets[row];
        let end = system.row_offsets[row + 1];
        let local = vector.t.slice(ndarray::s![start..end]);
        let mut applied = system.rows[row].htt.dot(&local);
        applied += &support_arrow_cross_forward(system, row, vector.beta.view())?;
        out_t.slice_mut(ndarray::s![start..end]).assign(&applied);
        support_arrow_cross_transpose_add(system, row, local, &mut out_beta)?;
    }
    Ok(SaeArrowVector {
        t: out_t,
        beta: out_beta,
    })
}

/// Exact application of the majorizer inverse used as the flexible-GMRES
/// preconditioner.  The reduced CG cap is its algebraic dimension: in exact
/// arithmetic an SPD `k x k` system terminates in at most `k` directions.  A
/// floating-point solve that cannot meet the scalar-derived `sqrt(eps)` floor
/// in that span is refused rather than returned as an unbounded approximation.
fn support_arrow_majorizer_inverse(
    system: &ArrowSchurSystem,
    factors: &ArrowFactorSlab,
    rhs: &SaeArrowVector,
) -> Result<SaeArrowVector, String> {
    let backend = CpuBatchedBlockSolver;
    let coordinate_dim = *system.row_offsets.last().unwrap_or(&0);
    let mut latent_forward = Array1::<f64>::zeros(coordinate_dim);
    let mut eliminated = Array1::<f64>::zeros(system.k);
    for row in 0..system.rows.len() {
        let start = system.row_offsets[row];
        let end = system.row_offsets[row + 1];
        let solved = backend.solve_block_vector(
            factors.factor(row),
            rhs.t.slice(ndarray::s![start..end]),
        );
        latent_forward
            .slice_mut(ndarray::s![start..end])
            .assign(&solved);
        support_arrow_cross_transpose_add(system, row, solved.view(), &mut eliminated)?;
    }
    let reduced_rhs = &rhs.beta - &eliminated;
    let solved_beta = if system.k == 0 {
        Array1::<f64>::zeros(0)
    } else {
        let tolerance = f64::EPSILON.sqrt();
        let (solved, report) = reduced_schur_inverse_apply(
            system,
            factors,
            0.0,
            &backend,
            None,
            None,
            &reduced_rhs,
            None,
            tolerance,
            system.k,
        )
        .ok_or_else(|| {
            format!(
                "support outer differential: reduced-Schur preconditioner broke down at dimension {}",
                system.k
            )
        })?;
        if !report.converged() {
            return Err(format!(
                "support outer differential: reduced-Schur preconditioner did not converge in its {}-direction algebraic span (relative residual {:.3e}, tolerance {:.3e})",
                system.k,
                report.relative_residual,
                report.tolerance,
            ));
        }
        solved
    };
    let mut solved_t = latent_forward;
    for row in 0..system.rows.len() {
        let start = system.row_offsets[row];
        let end = system.row_offsets[row + 1];
        let cross = support_arrow_cross_forward(system, row, solved_beta.view())?;
        let correction = backend.solve_block_vector(factors.factor(row), cross.view());
        solved_t
            .slice_mut(ndarray::s![start..end])
            .scaled_add(-1.0, &correction);
    }
    Ok(SaeArrowVector {
        t: solved_t,
        beta: solved_beta,
    })
}

/// `(tr((G + lambda*S)^-1 G), dim null(S))` for one atom's blocks.
///
/// Both the curvature census and the Fellner-Schall update need exactly this
/// pair, and both carried their own copy until #2502 -- which is how a single
/// tolerance defect came to be present, and to need fixing, in two places.
///
/// Two invariants hold by construction and are what the consumers rely on.
/// Because `lambda*S` is positive semidefinite, every eigenvector `v` of
/// `G + lambda*S` satisfies `v'Gv <= v'(G + lambda*S)v`, so each mode
/// contributes at most one and the trace is at most `m`. The modes spanning
/// `S`'s null space see `G` alone and contribute exactly one each, so the
/// trace is at least `dim null(S)` -- which is what makes
/// `trace - null_dim >= 0` an effective degrees of freedom rather than an
/// arbitrary difference.
///
/// The mode tolerance is scaled by `trace(G)`, which bounds the largest
/// eigenvalue of a positive semidefinite `G` and therefore bounds the
/// numerator being divided. Scaling it by `max|eigenvalue(G + lambda*S)|`
/// instead -- the pre-#2502 form -- lets a large `lambda` swallow the
/// well-conditioned modes spanning `S`'s null space, collapsing the trace to
/// zero and returning `-null_dim`. Measured at `lambda = 6.339e15`.
pub(crate) fn penalized_trace_and_null_dim(
    gram: &Array2<f64>,
    penalty: &Array2<f64>,
    lambda: f64,
    context: &str,
) -> Result<(f64, f64), String> {
    let m = gram.nrows();
    let symmetric_penalty = (penalty + &penalty.t()) * 0.5;
    let (penalty_eigenvalues, penalty_vectors) = symmetric_penalty
        .eigh(Side::Lower)
        .map_err(|error| format!("{context}: penalty eigh: {error}"))?;
    let penalty_scale = penalty_eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    // The same machine-precision relative floor `solve_psd_minimum_norm` uses
    // to decide rank, so the two agree about what "zero" means.
    let penalty_tolerance = f64::EPSILON * penalty_scale * m.max(1) as f64;
    let null_dim = penalty_eigenvalues
        .iter()
        .filter(|value| **value <= penalty_tolerance)
        .count() as f64;

    // Jacobi scaling in `S`'s eigenbasis, the identity
    // `solve_penalized_normal_equations` uses for the same reason: it removes
    // lambda from the conditioning rather than tolerating it. With
    // `d_i = 1/sqrt(1 + lambda*s_i)` and `G~ = D U' G U D`,
    //     tr((G + lambda*S)^-1 G) = tr((G~ + P~)^-1 G~),
    // where `P~ = diag(lambda*s/(1 + lambda*s))` has every entry in [0, 1) for
    // every lambda. No matrix whose condition number is lambda is ever formed,
    // so there is no precision cliff: assembling `G + lambda*S` loses `G`
    // entirely once `lambda*max|S|` passes `max|G|/eps`, which is near 1e15
    // here and is where the production collapse to `edf = -null_dim` occurred.
    let rotated = penalty_vectors.t().dot(gram).dot(&penalty_vectors);
    let mut scaled = Array2::<f64>::zeros((m, m));
    let mut penalty_fraction = vec![0.0_f64; m];
    for row in 0..m {
        // A symmetric PSD penalty has no negative eigenvalues; rounding can
        // still deliver one a hair below zero, and it carries no penalty.
        let s_row = penalty_eigenvalues[row].max(0.0);
        let d_row = 1.0 / (1.0 + lambda * s_row).sqrt();
        penalty_fraction[row] = if s_row > 0.0 && lambda > 0.0 {
            let product = lambda * s_row;
            // The limit of `x / (1 + x)` is 1, but the expression itself is
            // `inf / inf` = NaN once the product overflows. Take the limit.
            if product.is_finite() {
                product / (1.0 + product)
            } else {
                1.0
            }
        } else {
            0.0
        };
        for column in 0..m {
            let s_column = penalty_eigenvalues[column].max(0.0);
            let d_column = 1.0 / (1.0 + lambda * s_column).sqrt();
            scaled[[row, column]] = d_row * rotated[[row, column]] * d_column;
        }
    }
    let mut shifted = scaled.clone();
    for row in 0..m {
        shifted[[row, row]] += penalty_fraction[row];
    }
    let symmetric = (&shifted + &shifted.t()) * 0.5;
    let (eigenvalues, eigenvectors) = symmetric
        .eigh(Side::Lower)
        .map_err(|error| format!("{context}: eigh: {error}"))?;
    // Both operands are now scaled to G's own magnitude, so the floor for a
    // meaningless ratio is set by that magnitude.
    let scaled_trace = (0..m).map(|mode| scaled[[mode, mode]]).sum::<f64>();
    let tolerance = f64::EPSILON * scaled_trace.max(1.0) * m.max(1) as f64;
    let projected = eigenvectors.t().dot(&scaled).dot(&eigenvectors);
    let mut trace = 0.0_f64;
    for mode in 0..m {
        if eigenvalues[mode] > tolerance {
            trace += projected[[mode, mode]] / eigenvalues[mode];
        }
    }
    Ok((trace, null_dim))
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
            // The kernels below subscript FOUR quantities per atom: the
            // decoder's basis rows, the decoder's output columns, the reference
            // Gram's width, and the coordinate block's latent width. This door
            // used to validate the last two only, so an atom whose decoder did
            // not span its own basis was ADMITTED and aborted later, inside a
            // rayon worker, as a bare `ndarray: index out of bounds` naming no
            // row and no atom (#2572). The atom states its own contract; check
            // it here, where the two shapes can still be named.
            template.validate_shape_contract().map_err(|error| {
                format!("SaeSupportSparseTerm::new: atom {atom}: {error}")
            })?;
            if template.output_dim() != output_dim {
                return Err(format!(
                    "SaeSupportSparseTerm::new: atom {atom} output dimension {} != {output_dim}",
                    template.output_dim()
                ));
            }
            if template.latent_dim() != assignment.atom_coord_dim(atom) {
                return Err(format!(
                    "SaeSupportSparseTerm::new: atom {atom} latent dim {} != assignment dim {}",
                    template.latent_dim(),
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
            if support.len() > support_k || support.is_empty() {
                return Err(format!(
                    "SaeSupportSparseTerm::new: row {row} support width {} must be in 1..=top_k={support_k}",
                    support.len()
                ));
            }
            for (slot, &atom) in support.iter().enumerate() {
                atom_rows[atom as usize].push((row, slot));
            }
        }
        let atom_axis_periods = (0..k_atoms)
            .map(|atom| assignment.atom_axis_periods(atom))
            .collect();
        Ok(Self {
            atoms,
            assignment,
            output_dim,
            atom_rows,
            decoder_fista_passes: None,
            admission_dof_sigma2: None,
            exact_affine_ranking: false,
            grid_refinement: 1,
            admission_usage_amortized: false,
            variable_priced_support: false,
            atom_axis_periods,
        })
    }

    /// Axis periodicity of one atom's coordinate block: `None` on a Euclidean
    /// axis, `Some(period)` on a circular one.
    fn atom_axis_periods(&self, atom: usize) -> &[Option<f64>] {
        &self.atom_axis_periods[atom]
    }

    /// Total width of the compact coordinate state `T` — the concatenation of
    /// every row's active coordinate block.
    pub fn coordinate_state_len(&self) -> usize {
        (0..self.n_obs())
            .map(|row| self.assignment.coords_row(row).len())
            .sum()
    }

    /// Copy `T` into caller storage, row-major over rows and slot-major within
    /// a row — the same order `install_coordinates` and
    /// `wrapped_coordinate_residual` read.
    fn snapshot_coordinates(&self, out: &mut Vec<f64>) {
        out.clear();
        for row in 0..self.n_obs() {
            out.extend_from_slice(self.assignment.coords_row(row));
        }
    }

    /// Apply one compact step to `T`, retracting each row onto its atoms'
    /// manifolds — the same retraction the coordinate sweep's line search uses,
    /// so an extrapolated step lands on the manifold by construction rather
    /// than by being projected back afterwards.
    fn retract_coordinates(&mut self, step: &[f64]) -> Result<(), String> {
        let mut coords_rows = self.assignment.take_coords();
        let mut cursor = 0usize;
        let mut outcome = Ok(());
        for (row, coords_row) in coords_rows.iter_mut().enumerate() {
            let end = cursor + coords_row.len();
            if end > step.len() {
                outcome = Err(format!(
                    "SaeSupportSparseTerm::retract_coordinates: step width {} is short of \
                     row {row}'s block end {end}",
                    step.len()
                ));
                break;
            }
            if let Err(error) = self
                .assignment
                .retract_row_coords(row, coords_row, &step[cursor..end])
            {
                outcome = Err(error);
                break;
            }
            cursor = end;
        }
        self.assignment.restore_coords(coords_rows)?;
        outcome?;
        if cursor != step.len() {
            return Err(format!(
                "SaeSupportSparseTerm::retract_coordinates: step width {} != compact \
                 coordinate width {cursor}",
                step.len()
            ));
        }
        Ok(())
    }

    /// Install a whole `T`, projecting each row onto its atoms' manifolds.
    /// Used to restore a rejected extrapolation, where the target state is an
    /// absolute snapshot rather than a step.
    fn install_coordinates(&mut self, values: &[f64]) -> Result<(), String> {
        let mut cursor = 0usize;
        for row in 0..self.n_obs() {
            let width = self.assignment.coords_row(row).len();
            let end = cursor + width;
            if end > values.len() {
                return Err(format!(
                    "SaeSupportSparseTerm::install_coordinates: state width {} is short of                      row {row}'s block end {end}",
                    values.len()
                ));
            }
            self.assignment.set_row_coords(row, &values[cursor..end])?;
            cursor = end;
        }
        if cursor != values.len() {
            return Err(format!(
                "SaeSupportSparseTerm::install_coordinates: state width {} != compact                  coordinate width {cursor}",
                values.len()
            ));
        }
        Ok(())
    }

    /// `after - before` on each coordinate axis, taken on the axis's own
    /// manifold.
    ///
    /// On a periodic axis the sweep's projection returns the image to a
    /// principal branch, so a literal difference across the branch cut reads as
    /// a whole period where the step was infinitesimal. Wrapping to the
    /// principal branch is what makes the residual the honest step — and what
    /// lets the accelerator treat `before + residual` as a lifted image whose
    /// differences are consistent across cycles.
    fn wrapped_coordinate_residual(&self, before: &[f64], after: &[f64], out: &mut Vec<f64>) {
        out.clear();
        let mut cursor = 0usize;
        for row in 0..self.n_obs() {
            for &atom in self.assignment.support_indices(row) {
                for &period in self.atom_axis_periods(atom as usize) {
                    let delta = after[cursor] - before[cursor];
                    out.push(match period {
                        Some(period) if period.is_finite() && period > 0.0 => {
                            delta - period * (delta / period).round()
                        }
                        _ => delta,
                    });
                    cursor += 1;
                }
            }
        }
    }

    /// Gauge-canonical recurrence distance for the complete continuous inner
    /// state. The caller profiles periodic phase origins before this seam, so
    /// the wrapped coordinate difference and decoder difference both live on
    /// the identifiable representative that defines the Arrow operator.
    fn canonical_state_recurrence_change(
        &self,
        start_coordinates: &[f64],
        start_decoders: &[Array2<f64>],
        end_coordinates: &mut Vec<f64>,
        coordinate_residual: &mut Vec<f64>,
    ) -> Result<f64, String> {
        if start_decoders.len() != self.k_atoms() {
            return Err(format!(
                "support recurrence decoder blocks {} != K={}",
                start_decoders.len(),
                self.k_atoms(),
            ));
        }
        self.snapshot_coordinates(end_coordinates);
        self.wrapped_coordinate_residual(
            start_coordinates,
            end_coordinates,
            coordinate_residual,
        );
        let mut max_change = coordinate_residual
            .iter()
            .fold(0.0_f64, |current, &value| current.max(value.abs()));
        for (atom_index, (saved, atom)) in
            start_decoders.iter().zip(&self.atoms).enumerate()
        {
            if saved.dim() != atom.decoder_coefficients().dim() {
                return Err(format!(
                    "support recurrence atom {atom_index} decoder shape {:?} != {:?}",
                    saved.dim(),
                    atom.decoder_coefficients().dim(),
                ));
            }
            for (&before, &after) in saved.iter().zip(atom.decoder_coefficients().iter()) {
                max_change = max_change.max((after - before).abs());
            }
        }
        if max_change.is_finite() {
            Ok(max_change)
        } else {
            Err("support recurrence produced a non-finite state change".to_string())
        }
    }

    pub fn n_obs(&self) -> usize {
        self.assignment.n_obs()
    }

    /// Intensive iterate scale paired with the componentwise curvature-scaled
    /// stationarity residual. Every installed active coordinate and decoder
    /// coefficient participates; non-finite state is a typed refusal.
    fn parameter_iterate_scale(&self) -> Result<f64, SaeInnerKktScaleError> {
        let mut max_abs = 0.0_f64;
        for row in 0..self.n_obs() {
            for slot in 0..self.assignment.support_indices(row).len() {
                for (component, &value) in self
                    .assignment
                    .coords_for_slot(row, slot)
                    .iter()
                    .enumerate()
                {
                    if !value.is_finite() {
                        return Err(SaeInnerKktScaleError::NonFiniteIterate {
                            family: "support-coordinate",
                            group: row,
                            component,
                            value,
                        });
                    }
                    max_abs = max_abs.max(value.abs());
                }
            }
        }
        for (atom, manifold_atom) in self.atoms.iter().enumerate() {
            for (component, &value) in manifold_atom.decoder_coefficients().iter().enumerate() {
                if !value.is_finite() {
                    return Err(SaeInnerKktScaleError::NonFiniteIterate {
                        family: "support-decoder",
                        group: atom,
                        component,
                        value,
                    });
                }
                max_abs = max_abs.max(value.abs());
            }
        }
        let scale = 1.0 + max_abs;
        if !scale.is_finite() {
            return Err(SaeInnerKktScaleError::IterateScaleOverflow { max_abs });
        }
        Ok(scale)
    }

    /// Whether two terms carry the same discrete routing decision. Slot order
    /// is immaterial: coordinates are attached to atoms, while the TopK support
    /// is a set. A router that returns the same sets has proposed no support
    /// move, even if rebuilding and re-polishing its coordinates happens to
    /// shave a few roundoff bits from the continuous objective.
    fn has_same_support_as(&self, other: &Self) -> bool {
        self.n_obs() == other.n_obs()
            && (0..self.n_obs()).all(|row| {
                let left = self.assignment.support_indices(row);
                let right = other.assignment.support_indices(row);
                left.len() == right.len() && left.iter().all(|atom| right.contains(atom))
            })
    }

    /// Profile the continuous phase-origin gauge of every one-dimensional
    /// periodic atom. A common phase shift of all routed coordinates can be
    /// counter-rotated through the harmonic decoder without changing the
    /// represented function or its reference roughness; only the periodic ARD
    /// prior selects an origin. Leaving that global coordinate split across N
    /// row blocks and one decoder block makes Gauss--Seidel crawl along an
    /// almost-flat orbit. The circular mean is the exact minimizer of that
    /// gauge block.
    ///
    /// The declared periodic basis owns the canonical Fourier ordering
    /// `[1, sin(2πt), cos(2πt), ...]`, so the decoder transport is the
    /// corresponding block-diagonal rotation, not a fitted approximation. It
    /// transports the roughness Gram by the inverse congruence, so data fit and
    /// smoothness are invariant by construction. A phase is installed only
    /// when its analytically evaluated ARD decrease clears a roundoff bound.
    fn profile_periodic_phase_origins(
        &mut self,
        ard_precisions: &[Vec<f64>],
    ) -> Result<usize, String> {
        let mut profiled = 0usize;
        for atom_index in 0..self.k_atoms() {
            if self.atoms[atom_index].basis_kind() != &SaeAtomBasisKind::Periodic
                || self.assignment.atom_coord_dim(atom_index) != 1
                || self.atom_rows[atom_index].is_empty()
            {
                continue;
            }
            let Some(period) = self.atom_axis_periods(atom_index)[0] else {
                return Err(format!(
                    "profile_periodic_phase_origins: periodic atom {atom_index} has no period"
                ));
            };
            if !(period.is_finite() && period > 0.0) {
                return Err(format!(
                    "profile_periodic_phase_origins: atom {atom_index} has invalid period {period}"
                ));
            }
            if ard_precisions[atom_index][0] == 0.0 {
                continue;
            }
            let kappa = std::f64::consts::TAU / period;
            let mut sine = 0.0_f64;
            let mut cosine = 0.0_f64;
            for &(row, slot) in &self.atom_rows[atom_index] {
                let phase = kappa * self.assignment.coords_for_slot(row, slot)[0];
                let (sin, cos) = phase.sin_cos();
                sine += sin;
                cosine += cos;
            }
            let resultant = sine.hypot(cosine);
            let count = self.atom_rows[atom_index].len() as f64;
            if resultant <= f64::EPSILON * count {
                // The profiled prior is phase-invariant when the circular
                // resultant vanishes, so no origin is statistically selected.
                continue;
            }
            let shift = sine.atan2(cosine) / kappa;
            if shift.abs() <= f64::EPSILON * period {
                continue;
            }

            let basis_size = self.atoms[atom_index].basis_size();
            if basis_size == 0 || basis_size % 2 == 0 {
                return Err(format!(
                    "profile_periodic_phase_origins: periodic atom {atom_index} basis width \
                     {basis_size} does not follow [1, sin, cos, ...]"
                ));
            }
            let mut transport = Array2::<f64>::eye(basis_size);
            for harmonic in 1..=(basis_size - 1) / 2 {
                let angle = std::f64::consts::TAU * harmonic as f64 * shift;
                let (sin, cos) = angle.sin_cos();
                let sine_index = 2 * harmonic - 1;
                let cosine_index = 2 * harmonic;
                transport[[sine_index, sine_index]] = cos;
                transport[[cosine_index, sine_index]] = sin;
                transport[[sine_index, cosine_index]] = -sin;
                transport[[cosine_index, cosine_index]] = cos;
            }

            let alpha = ard_precisions[atom_index][0];
            let mut old_energy = KahanSum::default();
            let mut new_energy = KahanSum::default();
            let mut shifted_coordinates = Vec::with_capacity(self.atom_rows[atom_index].len());
            for &(row, slot) in &self.atom_rows[atom_index] {
                let coordinate = self.assignment.coords_for_slot(row, slot)[0];
                let shifted = coordinate - shift;
                old_energy.add(ArdAxisPrior::eval(alpha, coordinate, Some(period)).value);
                new_energy.add(ArdAxisPrior::eval(alpha, shifted, Some(period)).value);
                shifted_coordinates.push((row, slot, shifted));
            }
            let old_energy = old_energy.sum();
            let new_energy = new_energy.sum();
            let energy_scale = 1.0 + old_energy.abs().max(new_energy.abs());
            let resolution = 512.0 * f64::EPSILON * count * energy_scale;
            if !(old_energy - new_energy > resolution) {
                continue;
            }

            let decoder = fast_ab(
                &transport,
                self.atoms[atom_index].decoder_coefficients(),
            );
            // `B_new = T B_old`, hence `S_new = T S_old Tᵀ` for orthogonal T.
            let smooth_left = fast_ab(&transport, self.atoms[atom_index].smooth_penalty());
            let smooth_penalty = smooth_left.dot(&transport.t());
            let basis_values = self.atoms[atom_index].basis_values.clone();
            let basis_jacobian = self.atoms[atom_index].basis_jacobian.clone();
            self.atoms[atom_index].install_reparameterized_basis(
                basis_values,
                basis_jacobian,
                decoder,
                smooth_penalty,
            )?;
            for (row, slot, coordinate) in shifted_coordinates {
                self.assignment
                    .set_slot_coords(row, slot, &[coordinate])?;
            }
            profiled += 1;
        }
        Ok(profiled)
    }

    pub fn k_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Decoder width of one atom: its block is `basis_size x output_dim`.
    pub fn atom_basis_size(&self, atom: usize) -> usize {
        self.atoms[atom].basis_size()
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// #2502 occupancy-earned topology. A periodic atom whose routed tokens
    /// occupy a small contiguous arc is a bounded curve wearing a circle: the
    /// empty arc's shape is pure penalty extrapolation, and the closed basis
    /// spends coefficients enforcing a closure the data never asked for
    /// (measured: the four strongest loops in a 250k-row fit carry their
    /// tokens on 10-30% of the circle, always ONE arc through the phase seam).
    ///
    /// Census each 1-D periodic atom's phases into `bins`; when the occupied
    /// fraction is at most `max_occupancy`, rebuild the atom as a Euclidean
    /// chart through the SAME planner pipeline the seed uses, unwrap every
    /// routed coordinate through the largest empty gap onto `[-1, 1]`, and
    /// zero the decoder block so the next decoder sweep refits it against the
    /// identical routed rows. Support, routing, and every other atom are
    /// untouched. Returns the converted atom indices.
    pub fn convert_underoccupied_loops(
        &mut self,
        random_state: u64,
    ) -> Result<Vec<usize>, String> {
        let mut converted = Vec::new();
        for atom_index in 0..self.k_atoms() {
            if self.atom_axis_periods[atom_index].len() != 1 {
                continue;
            }
            let Some(period) = self.atom_axis_periods[atom_index][0] else {
                continue;
            };
            if !(period.is_finite() && period > 0.0) {
                continue;
            }
            let pairs = self.atom_rows[atom_index].clone();
            if pairs.is_empty() {
                continue;
            }
            let mut fracs = Vec::with_capacity(pairs.len());
            for &(row, slot) in &pairs {
                let t = self.assignment.coords_for_slot(row, slot)[0];
                fracs.push((t / period).rem_euclid(1.0));
            }
            // Degeneracy test, independent of occupancy: sample the decoded
            // image around the whole period and compare its two principal
            // second moments. A circle spends equal power on both; an
            // ellipse collapsed to a diameter spends it all on one, and is
            // a line traversed out and back no matter how well occupied.
            // The threshold is the sampling resolution itself: an image
            // whose minor axis is below the chord length between adjacent
            // samples is not resolvable as anything but a segment.
            let probes = self.atoms[atom_index].basis_size().max(8) * 4;
            let mut image = Array2::<f64>::zeros((probes, self.output_dim));
            if let Some(evaluator) = self.atoms[atom_index].basis_evaluator.clone() {
                for probe in 0..probes {
                    let t = period * probe as f64 / probes as f64;
                    let coordinate = Array2::from_shape_vec((1, 1), vec![t])
                        .map_err(|error| format!("degeneracy probe: {error}"))?;
                    let (phi, _) = evaluator.evaluate(coordinate.view())?;
                    let decoded = phi
                        .row(0)
                        .dot(self.atoms[atom_index].decoder_coefficients());
                    for channel in 0..self.output_dim {
                        image[[probe, channel]] = decoded[channel];
                    }
                }
                let mut centre = vec![0.0_f64; self.output_dim];
                for probe in 0..probes {
                    for channel in 0..self.output_dim {
                        centre[channel] += image[[probe, channel]] / probes as f64;
                    }
                }
                let mut total = 0.0_f64;
                let mut along = 0.0_f64;
                // Power along the dominant direction vs total: one power
                // iteration on the centred image's Gram is enough to
                // separate a segment from a genuine ellipse.
                let mut direction = vec![0.0_f64; self.output_dim];
                for channel in 0..self.output_dim {
                    direction[channel] = image[[0, channel]] - centre[channel];
                }
                let mut norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
                for _ in 0..8 {
                    if !(norm > 0.0) {
                        break;
                    }
                    for value in direction.iter_mut() {
                        *value /= norm;
                    }
                    let mut next = vec![0.0_f64; self.output_dim];
                    for probe in 0..probes {
                        let mut dot = 0.0_f64;
                        for channel in 0..self.output_dim {
                            dot += (image[[probe, channel]] - centre[channel])
                                * direction[channel];
                        }
                        for channel in 0..self.output_dim {
                            next[channel] +=
                                dot * (image[[probe, channel]] - centre[channel]);
                        }
                    }
                    direction = next;
                    norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
                }
                if norm > 0.0 {
                    for value in direction.iter_mut() {
                        *value /= norm;
                    }
                    for probe in 0..probes {
                        let mut dot = 0.0_f64;
                        let mut sq = 0.0_f64;
                        for channel in 0..self.output_dim {
                            let centred = image[[probe, channel]] - centre[channel];
                            dot += centred * direction[channel];
                            sq += centred * centred;
                        }
                        total += sq;
                        along += dot * dot;
                    }
                }
                let across = (total - along).max(0.0);
                let resolution = total / probes as f64 / (probes as f64).powi(2);
                if total > 0.0 && across <= resolution * probes as f64 {
                    log::debug!(
                        "atom {atom_index}: periodic image is degenerate to a segment \
                         (across/total = {:.3e}); unrolling",
                        across / total
                    );
                    fracs.clear();
                    fracs.extend((0..pairs.len()).map(|slot| slot as f64 / pairs.len() as f64));
                }
            }
            // Exact largest-gap test, no binning. Under the null that a
            // closed loop's usage is uniform on the circle, the largest of
            // the n circular spacings G satisfies the exact bound
            //     P(G >= g) <= n * (1 - g)^(n-1),
            // so the observed gap g* refutes the closed topology at the
            // sample-size-derived level 1/n exactly when
            //     (n - 1) * ln(1 - g*) <= -2 ln n.
            // The level is 1/n rather than a tuned constant: one expected
            // false unroll per n routed tokens, vanishing for real atoms.
            let mut sorted = fracs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite phase"));
            let n_tokens = sorted.len();
            let mut gap_len = 0.0_f64;
            let mut gap_start = 0.0_f64;
            for i in 0..n_tokens {
                let here = sorted[i];
                let next = if i + 1 == n_tokens {
                    sorted[0] + 1.0
                } else {
                    sorted[i + 1]
                };
                if next - here > gap_len {
                    gap_len = next - here;
                    gap_start = here;
                }
            }
            let n_f = n_tokens as f64;
            let refuted = gap_len < 1.0
                && (n_f - 1.0) * (1.0 - gap_len).ln() <= -2.0 * n_f.ln()
                || gap_len >= 1.0;
            if !refuted {
                continue;
            }
            let arc_start = (gap_start + gap_len).rem_euclid(1.0);
            let arc_len = 1.0 - gap_len;
            // Fresh Euclidean atom through the seed's own planner pipeline, so
            // every downstream shape contract holds by construction.
            let kind = sae_atom_basis_kind_from_str("euclidean")?;
            let design_rows = super::support_seed::planner_design_rows(&kind);
            let mut plan_seed = ndarray::Array3::<f64>::zeros((1, design_rows, 1));
            for grid in 0..design_rows {
                plan_seed[[0, grid, 0]] =
                    -1.0 + 2.0 * (grid as f64 / (design_rows - 1) as f64);
            }
            let dummy_target = Array2::<f64>::zeros((design_rows, 1));
            let euclidean_basis = ["euclidean".to_string()];
            let mut plans = sae_build_atom_plans(
                dummy_target.view(),
                &euclidean_basis,
                &[1usize],
                plan_seed.view(),
                random_state.wrapping_add(atom_index as u64),
                &[None],
            )?;
            let plan = plans.pop().ok_or_else(|| {
                "convert_underoccupied_loops: planner returned no plan".to_string()
            })?;
            let probe_seed = ndarray::Array3::<f64>::zeros((1, 1, 1));
            let (phi_stack, jet_stack, penalty_stack, basis_sizes, _) =
                sae_build_padded_basis_stacks(
                    std::slice::from_ref(&plan),
                    probe_seed.view(),
                    1,
                )?;
            let m = basis_sizes[0];
            let phi = phi_stack.slice(ndarray::s![0, 0..1, 0..m]).to_owned();
            let jet = jet_stack
                .slice(ndarray::s![0, 0..1, 0..m, 0..1])
                .to_owned();
            let reference = SaeReferenceRoughness::ProvidedFunctionGram(
                penalty_stack.slice(ndarray::s![0, 0..m, 0..m]).to_owned(),
            );
            let evaluator = plan.geometry.build_evaluator()?;
            let replacement = SaeManifoldAtom::new(
                format!("{}_unrolled", self.atoms[atom_index].name),
                kind,
                1,
                phi,
                jet,
                Array2::<f64>::zeros((m, self.output_dim)),
                reference,
            )?
            .with_basis_second_jet(evaluator)
            .with_geometry_plan(plan.geometry.clone())?;
            self.atoms[atom_index] = replacement;
            self.assignment.convert_atom_to_euclidean(atom_index)?;
            for (&(row, slot), &frac) in pairs.iter().zip(&fracs) {
                let t_new = if arc_len > 0.0 {
                    let unwrapped = (frac - arc_start).rem_euclid(1.0).min(arc_len);
                    -1.0 + 2.0 * (unwrapped / arc_len).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                self.assignment.set_slot_coords(row, slot, &[t_new])?;
            }
            self.atom_axis_periods[atom_index] = vec![None];
            converted.push(atom_index);
        }
        Ok(converted)
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
        let zero_prior: Vec<Vec<f64>> = self
            .atoms
            .iter()
            .map(|atom| vec![0.0_f64; atom.latent_dim()])
            .collect();
        self.reroute_fixed_decoder_ard(target, support_k, random_state, &zero_prior)
    }

    /// [`Self::reroute_fixed_decoder`] scoring the coordinate prior too, so the
    /// greedy step and the caller's acceptance test agree on the objective.
    pub fn reroute_fixed_decoder_ard(
        &self,
        target: ArrayView2<'_, f64>,
        support_k: usize,
        random_state: u64,
        ard_precisions: &[Vec<f64>],
    ) -> Result<Self, String> {
        if ard_precisions.len() != self.k_atoms() {
            return Err(format!(
                "reroute_fixed_decoder_ard: ard_precisions length {} must equal K={}",
                ard_precisions.len(),
                self.k_atoms()
            ));
        }
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
        // Each row's routing reads only that row and the frozen decoders, so the
        // sweep is parallel by construction; an indexed `collect` restores row
        // order, making the result identical to the serial sweep it replaces.
        // This is the dominant cost of an out-of-sample reconstruct -- it scores
        // every atom against every row -- and it was leaving a 30-core box at
        // load 10.
        // ---- residual-greedy (OMP) routing --------------------------------
        // Marginal top-s is not the right selection rule for a K > P dictionary:
        // the atoms are necessarily coherent (Welch), so the s best-individually
        // atoms are near-duplicates and span far less than the s best jointly.
        // Greedy against the running residual fixes that; the chart argmax is
        // taken on a grid so the score is a property of the atom's image rather
        // than of its index.
        // One trial coordinate per basis coefficient. A basis carrying `m`
        // coefficients cannot resolve more than about `m` independent features
        // along its chart, so `m` is the basis's own resolution rather than a
        // tuning constant. Atoms may carry different widths, so slots are
        // addressed through a prefix offset instead of a uniform stride.
        //
        // Multi-axis atoms fall through to the marginal path below: a product
        // grid is exponential in the latent dimension, and the overcomplete
        // lane this serves admits 1-D charts.
        // #2502 DoF-priced admission. Raw SSE improvement rewards flexibility
        // twice -- a wider basis both fits better AND is searched on a finer
        // grid -- so when armed, every ranking expression below subtracts the
        // atom's amortized parameter cost: matched_dl's `m*P*(1/2)log2 N`
        // bits, divided by the atom's current firing count, converted at
        // `2*sigma2*ln 2` per bit to the gain scale. All-zero when disarmed,
        // so the priced router IS the unpriced router.
        let dof_charge: Vec<f64> = match self.admission_dof_sigma2 {
            None => vec![0.0_f64; self.k_atoms()],
            Some(sigma2) => {
                let l_param = 0.5 * (target.nrows().max(2) as f64).log2();
                // Amortize over the PORTFOLIO's mean firing count, not each
                // atom's own: dividing by the atom's own firings prices
                // rarity, not parameters, and a homogeneous portfolio then
                // pays a charge that varies only through usage -- measured,
                // that cost 0.167 EV. With the shared denominator the charge
                // varies only through basis size, and a homogeneous
                // portfolio receives a constant that cannot reorder anything.
                let mean_firings = (self
                    .atom_rows
                    .iter()
                    .map(|rows| rows.len())
                    .sum::<usize>()
                    .max(1) as f64)
                    / self.k_atoms().max(1) as f64;
                (0..self.k_atoms())
                    .map(|atom| {
                        let bits = self.atoms[atom].basis_size() as f64
                            * self.output_dim as f64
                            * l_param;
                        let denominator = if self.admission_usage_amortized {
                            self.atom_rows[atom].len().max(1) as f64
                        } else {
                            mean_firings.max(1.0)
                        };
                        2.0 * sigma2 * std::f64::consts::LN_2 * bits / denominator
                    })
                    .collect()
            }
        };
        if self.atoms.iter().all(|atom| atom.latent_dim() == 1) {
            let k_atoms = self.k_atoms();
            let mut grid_offset = Vec::with_capacity(k_atoms + 1);
            let mut slot_atom = Vec::new();
            let mut slots = 0usize;
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                grid_offset.push(slots);
                let width = atom.basis_size().max(2) * self.grid_refinement.max(1);
                slot_atom.extend(std::iter::repeat(atom_index).take(width));
                slots += width;
            }
            grid_offset.push(slots);
            let mut gamma = Array2::<f64>::zeros((slots, self.output_dim));
            let mut theta = vec![0.0_f64; slots];
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                    format!("reroute omp: atom {atom_index} has no evaluator")
                })?;
                let width = grid_offset[atom_index + 1] - grid_offset[atom_index];
                for g in 0..width {
                    // Sample the CHART coordinate, not the pre-squash
                    // variable. `chart_coordinate` squashes periodic kinds
                    // through `0.5 + atan(raw)/PI`, so a `raw` grid on
                    // [-1, 1] reaches only `t` in [0.25, 0.75] -- half the
                    // period -- and the greedy would rank such an atom
                    // without ever evaluating the other half. Cell centres
                    // are used there because a periodic cell's endpoints are
                    // the same point and `tan` diverges at them.
                    //
                    // Every other kind passes `raw` through unchanged and
                    // keeps the half-open sample; see the branch below for why
                    // the closed interval is not an improvement there.
                    let periodic_chart = matches!(
                        atom.basis_kind(),
                        SaeAtomBasisKind::Periodic
                            | SaeAtomBasisKind::Torus
                            | SaeAtomBasisKind::KleinBottle
                    );
                    let raw = if periodic_chart {
                        let u = (g as f64 + 0.5) / width as f64;
                        (std::f64::consts::PI * (u - 0.5)).tan()
                    } else {
                        // Half-open, as it has always been. The closed form
                        // drops `t = 0` at width 2, and `gamma(0) = b0` is the
                        // grid point nearest every row's optimum for a `linear`
                        // atom -- routed coordinates sit in about [-0.02, 0.02]
                        // while this grid spans [-1, 1]. Its gain is a lower
                        // bound on the exact gain, so removing it can only cost.
                        // The real fix for these atoms is the closed-form
                        // optimal-`t` gain, not a redistribution of two points.
                        -1.0 + 2.0 * (g as f64 / width as f64)
                    };
                    let t = super::support_seed::chart_coordinate(atom.basis_kind(), 0, raw);
                    let coordinate = Array2::from_shape_vec((1, 1), vec![t])
                        .map_err(|error| format!("reroute grid: {error}"))?;
                    let (phi, _) = evaluator.evaluate(coordinate.view())?;
                    let decoded = phi.row(0).dot(atom.decoder_coefficients());
                    let slot = grid_offset[atom_index] + g;
                    for channel in 0..self.output_dim {
                        gamma[[slot, channel]] = decoded[channel];
                    }
                    theta[slot] = t;
                }
            }
            // Affine atoms (`gamma(t) = A + t*B`) admit a closed-form optimal
            // coordinate, so they need no grid at all. A and B come from
            // EVALUATING the atom at two coordinates rather than from its
            // coefficients, which absorbs any affine reparameterisation the
            // evaluator applies. `None` means "rank this atom from the grid".
            let mut affine: Vec<Option<(Array1<f64>, Array1<f64>, f64, f64, f64)>> =
                vec![None; k_atoms];
            if self.exact_affine_ranking {
                for (atom_index, atom) in self.atoms.iter().enumerate() {
                    if atom.basis_size() != 2 {
                        continue;
                    }
                    let Some(evaluator) = atom.basis_evaluator.as_ref() else {
                        continue;
                    };
                    let decode = |t: f64| -> Result<Array1<f64>, String> {
                        let coordinate = Array2::from_shape_vec((1, 1), vec![t])
                            .map_err(|error| format!("exact affine probe: {error}"))?;
                        let (phi, _) = evaluator.evaluate(coordinate.view())?;
                        Ok(phi.row(0).dot(atom.decoder_coefficients()))
                    };
                    let base = decode(0.0)?;
                    let slope = &decode(1.0)? - &base;
                    let slope_norm = slope.dot(&slope).sqrt();
                    let base_norm = base.dot(&base).sqrt();
                    // Resolvable against the atom's own offset scale, not
                    // merely non-zero: `t* = along / ||B||` is unbounded, so a
                    // slope near the rounding of `A` produces an enormous
                    // coordinate from a bounded contribution. Such atoms take
                    // the grid path, which bounds the coordinate to the chart.
                    if !(slope_norm > f64::EPSILON * base_norm * self.output_dim as f64) {
                        continue;
                    }
                    let unit = &slope / slope_norm;
                    let base_sq = base.dot(&base);
                    let base_dot_unit = base.dot(&unit);
                    affine[atom_index] =
                        Some((base, unit, base_sq, base_dot_unit, slope_norm));
                }
            }
            let self_term: Vec<f64> = (0..slots)
                .map(|slot| {
                    (0..self.output_dim).map(|c| gamma[[slot, c]] * gamma[[slot, c]]).sum::<f64>()
                })
                .collect();
            // `2 * V(alpha, t)` -- the prior the objective charges for placing a
            // row at this chart coordinate, on the same scale as `gain`.
            let prior_term: Vec<f64> = (0..slots)
                .map(|slot| {
                    let atom_index = slot_atom[slot];
                    let period = self.atom_axis_periods(atom_index)[0];
                    2.0 * ArdAxisPrior::eval(
                        ard_precisions[atom_index][0],
                        theta[slot],
                        period,
                    )
                    .value
                })
                .collect();

            let routed: Vec<(Vec<u32>, Vec<f64>, Vec<f64>)> = (0..target.nrows())
                .into_par_iter()
                .map(|row| {
                    let mut residual: Vec<f64> =
                        (0..self.output_dim).map(|c| target[[row, c]]).collect();
                    let mut taken = vec![false; k_atoms];
                    let mut picked: Vec<(usize, f64, f64)> = Vec::with_capacity(support_k);
                    for _ in 0..support_k {
                        let mut best_gain = f64::NEG_INFINITY;
                        let mut best_atom = usize::MAX;
                        let mut best_theta = 0.0;
                        let mut best_slot = 0usize;
                        // The vector the winner actually contributes. Under
                        // exact ranking it is not a grid point, so the residual
                        // cannot be updated from `gamma` alone.
                        let mut best_decoded: Option<Array1<f64>> = None;
                        for atom_index in 0..k_atoms {
                            if taken[atom_index] {
                                continue;
                            }
                            if let Some((base, unit, base_sq, base_dot_unit, slope_norm)) =
                                affine[atom_index].as_ref()
                            {
                                // `gain(t*) = 2<r,A> - ||A||^2 + (<r,u> - <A,u>)^2`,
                                // the maximum over t, so it dominates any grid
                                // point of this atom.
                                let mut r_dot_base = 0.0;
                                let mut r_dot_unit = 0.0;
                                for c in 0..self.output_dim {
                                    r_dot_base += residual[c] * base[c];
                                    r_dot_unit += residual[c] * unit[c];
                                }
                                let along = r_dot_unit - base_dot_unit;
                                // The prior is charged at the grid's own
                                // resolution; taking this atom's first slot
                                // keeps the charge identical to the grid path
                                // rather than silently dropping it.
                                let gain = 2.0 * r_dot_base - base_sq + along * along
                                    - prior_term[grid_offset[atom_index]]
                                    - dof_charge[atom_index];
                                if gain > best_gain {
                                    best_gain = gain;
                                    best_atom = atom_index;
                                    best_theta = along / slope_norm;
                                    best_slot = grid_offset[atom_index];
                                    best_decoded = Some(base + &(unit * along));
                                }
                            } else {
                                for slot in grid_offset[atom_index]..grid_offset[atom_index + 1] {
                                    let mut cross = 0.0;
                                    for c in 0..self.output_dim {
                                        cross += residual[c] * gamma[[slot, c]];
                                    }
                                    let gain = 2.0 * cross
                                        - self_term[slot]
                                        - prior_term[slot]
                                        - dof_charge[atom_index];
                                    if gain > best_gain {
                                        best_gain = gain;
                                        best_atom = atom_index;
                                        best_theta = theta[slot];
                                        best_slot = slot;
                                        best_decoded = None;
                                    }
                                }
                            }
                        }
                        if best_atom == usize::MAX {
                            break;
                        }
                        if self.variable_priced_support
                            && self.admission_dof_sigma2.is_some()
                            && !picked.is_empty()
                            && best_gain <= 0.0
                        {
                            break;
                        }
                        taken[best_atom] = true;
                        // Orthogonal pursuit, for supports made entirely of
                        // affine atoms: re-fit every selected coordinate
                        // against the original target rather than keeping each
                        // at the value it had when it was picked. Worth +0.0050
                        // measured, and it lets the exact ranking contribute a
                        // further +0.0018. Restricted to all-affine supports
                        // because a grid-ranked atom's contribution is not
                        // linear in a coordinate we could re-fit here.
                        let all_affine = self.exact_affine_ranking
                            && picked.iter().all(|entry| affine[entry.0].is_some())
                            && affine[best_atom].is_some();
                        if all_affine {
                            let mut chosen: Vec<usize> =
                                picked.iter().map(|entry| entry.0).collect();
                            chosen.push(best_atom);
                            let width = chosen.len();
                            let mut offset = vec![0.0_f64; self.output_dim];
                            for &atom in &chosen {
                                let (base, _, _, _, _) = affine[atom]
                                    .as_ref()
                                    .expect("all_affine checked above");
                                for c in 0..self.output_dim {
                                    offset[c] += base[c];
                                }
                            }
                            let mut normal = Array2::<f64>::zeros((width, width));
                            let mut rhs = Array1::<f64>::zeros(width);
                            for (i, &atom_i) in chosen.iter().enumerate() {
                                let (_, unit_i, _, _, norm_i) = affine[atom_i]
                                    .as_ref()
                                    .expect("all_affine checked above");
                                for c in 0..self.output_dim {
                                    rhs[i] += (target[[row, c]] - offset[c])
                                        * unit_i[c]
                                        * norm_i;
                                }
                                for (j, &atom_j) in chosen.iter().enumerate() {
                                    let (_, unit_j, _, _, norm_j) = affine[atom_j]
                                        .as_ref()
                                        .expect("all_affine checked above");
                                    let mut dot = 0.0;
                                    for c in 0..self.output_dim {
                                        dot += unit_i[c] * unit_j[c];
                                    }
                                    normal[[i, j]] = dot * norm_i * norm_j;
                                }
                            }
                            if let Ok(solved) = Self::solve_psd_minimum_norm(
                                &normal,
                                &rhs.clone().insert_axis(ndarray::Axis(1)),
                                "reroute orthogonal pursuit",
                            ) {
                                for c in 0..self.output_dim {
                                    residual[c] = target[[row, c]] - offset[c];
                                }
                                for (i, &atom) in chosen.iter().enumerate() {
                                    let (_, unit, _, _, norm) = affine[atom]
                                        .as_ref()
                                        .expect("all_affine checked above");
                                    let coefficient = solved[[i, 0]];
                                    for c in 0..self.output_dim {
                                        residual[c] -= coefficient * norm * unit[c];
                                    }
                                    if atom == best_atom {
                                        best_theta = coefficient;
                                    } else if let Some(entry) =
                                        picked.iter_mut().find(|e| e.0 == atom)
                                    {
                                        entry.2 = coefficient;
                                    }
                                }
                                picked.push((best_atom, best_gain, best_theta));
                                continue;
                            }
                        }
                        match best_decoded.as_ref() {
                            Some(decoded) => {
                                for c in 0..self.output_dim {
                                    residual[c] -= decoded[c];
                                }
                            }
                            None => {
                                for c in 0..self.output_dim {
                                    residual[c] -= gamma[[best_slot, c]];
                                }
                            }
                        }
                        picked.push((best_atom, best_gain, best_theta));
                    }
                    picked.sort_by_key(|entry| entry.0);
                    (
                        picked.iter().map(|e| e.0 as u32).collect::<Vec<u32>>(),
                        picked.iter().map(|e| e.1).collect::<Vec<f64>>(),
                        picked.iter().map(|e| e.2).collect::<Vec<f64>>(),
                    )
                })
                .collect();

            let mut indices = Vec::with_capacity(target.nrows());
            let mut gate_params = Vec::with_capacity(target.nrows());
            let mut coords = Vec::with_capacity(target.nrows());
            for (row_indices, row_gates, row_coords) in routed {
                indices.push(row_indices);
                gate_params.push(row_gates);
                coords.push(row_coords);
            }
            let atom_specs = self
                .atoms
                .iter()
                .enumerate()
                .map(|(atom, template)| SaeAssignmentAtomSpec {
                    latent_dim: template.latent_dim(),
                    id_mode: gam_terms::latent::LatentIdMode::None,
                    manifold: template.basis_kind().latent_manifold(template.latent_dim()),
                    retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
                    latent_id: super::support_seed::splitmix64(atom as u64),
                })
                .collect();
            let assignment = SaeAssignmentState::from_topk_support_heterogeneous(
                target.nrows(),
                k_atoms,
                support_k,
                atom_specs,
                indices,
                gate_params,
                coords,
            )?;
            let mut routed = Self::new(self.atoms.clone(), assignment)?;
            routed.decoder_fista_passes = self.decoder_fista_passes;
            routed.admission_dof_sigma2 = self.admission_dof_sigma2;
            routed.variable_priced_support = self.variable_priced_support;
            routed.admission_usage_amortized = self.admission_usage_amortized;
            routed.exact_affine_ranking = self.exact_affine_ranking;
            routed.grid_refinement = self.grid_refinement;
            return Ok(routed);
        }
        type RowRoute = (Vec<u32>, Vec<f64>, Vec<f64>);
        let per_row: Vec<RowRoute> = target
            .axis_iter(ndarray::Axis(0))
            .into_par_iter()
            .map(|row| -> Result<RowRoute, String> {
            let row_values = row.as_slice().ok_or_else(|| {
                "SaeSupportSparseTerm::reroute_fixed_decoder: target row is not contiguous"
                    .to_string()
            })?;
            let mut selected = Vec::<Candidate>::with_capacity(support_k);
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                // The hashed coordinate is only a stand-in for "where on this atom's
                // curve does this row sit". Scoring an atom at an arbitrary point
                // makes selection near-uncorrelated with which atoms can actually
                // represent the row, so a 1-D atom searches its own curve at the
                // basis's resolution before being scored. Only a multi-axis atom,
                // whose product grid is exponential, still falls back to the hash.
                let route_grid = atom.basis_size().max(2);
                let candidate_coords = if atom.latent_dim() == 1 {
                    let periodic = matches!(
                        atom.basis_kind(),
                        super::SaeAtomBasisKind::Periodic
                    );
                    let mut best_t = 0.0_f64;
                    let mut best_s = f64::NEG_INFINITY;
                    for g in 0..route_grid {
                        let frac = g as f64 / route_grid as f64;
                        let t_try = if periodic { frac } else { -1.0 + 2.0 * frac };
                        let c_try = Array2::from_shape_vec((1, 1), vec![t_try])
                            .map_err(|error| format!("reroute grid: {error}"))?;
                        if let Some(ev) = atom.basis_evaluator.as_ref() {
                            let (phi_try, _) = ev.evaluate(c_try.view())?;
                            let dec = phi_try.row(0).dot(atom.decoder_coefficients());
                            let s_try: f64 = row
                                .iter()
                                .zip(dec.iter())
                                .map(|(truth, fit)| 2.0 * truth * fit - fit * fit)
                                .sum();
                            if s_try > best_s {
                                best_s = s_try;
                                best_t = t_try;
                            }
                        }
                    }
                    vec![best_t]
                } else {
                    // One trial per basis coefficient -- the SAME resolution
                    // rule the 1-D grid uses -- instead of one hashed point.
                    // Each trial is hash-drawn at a distinct salt and
                    // projected onto the manifold (the on-manifold invariant
                    // the seed path enforces; an off-manifold candidate's
                    // tangent projector stops being a projection, measured
                    // rhs_dot_delta = -0.67 on the first embedded sphere).
                    // A d>=2 atom now competes on the same footing as a 1-D
                    // atom rather than at wherever one hash landed.
                    let manifold = atom.basis_kind().latent_manifold(atom.latent_dim());
                    let trials = atom.basis_size().max(2);
                    let mut best_s = f64::NEG_INFINITY;
                    let mut best_cand: Vec<f64> = Vec::new();
                    for trial in 0..trials {
                        let raw: Vec<f64> = (0..atom.latent_dim())
                            .map(|axis| {
                                let raw = super::support_seed::projection(
                                    row_values,
                                    atom_index,
                                    axis + 1 + trial * atom.latent_dim(),
                                    random_state,
                                );
                                super::support_seed::chart_coordinate(
                                    atom.basis_kind(),
                                    axis,
                                    raw,
                                )
                            })
                            .collect();
                        let cand = manifold
                            .project_point(Array1::from_vec(raw).view())
                            .to_vec();
                        let c_try =
                            Array2::from_shape_vec((1, atom.latent_dim()), cand.clone())
                                .map_err(|error| {
                                    format!("reroute d>=2 trial: {error}")
                                })?;
                        if let Some(ev) = atom.basis_evaluator.as_ref() {
                            let (phi_try, _) = ev.evaluate(c_try.view())?;
                            let dec = phi_try.row(0).dot(atom.decoder_coefficients());
                            let s_try: f64 = row
                                .iter()
                                .zip(dec.iter())
                                .map(|(truth, fit)| 2.0 * truth * fit - fit * fit)
                                .sum();
                            if s_try > best_s {
                                best_s = s_try;
                                best_cand = cand;
                            }
                        }
                    }
                    if best_cand.is_empty() {
                        manifold
                            .project_point(
                                Array1::from_vec(vec![0.0; atom.latent_dim()]).view(),
                            )
                            .to_vec()
                    } else {
                        best_cand
                    }
                };
                let coordinate =
                    Array2::from_shape_vec((1, atom.latent_dim()), candidate_coords.clone())
                        .map_err(|error| {
                            format!("SaeSupportSparseTerm::reroute_fixed_decoder: {error}")
                        })?;
                let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                    format!(
                        "SaeSupportSparseTerm::reroute_fixed_decoder: atom {atom_index} has no evaluator"
                    )
                })?;
                let (phi, _) = evaluator.evaluate(coordinate.view())?;
                let decoded = phi.row(0).dot(atom.decoder_coefficients());
                let score = row
                    .iter()
                    .zip(decoded.iter())
                    .map(|(truth, fit)| 2.0 * truth * fit - fit * fit)
                    .sum::<f64>()
                    - dof_charge[atom_index];
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
            if self.variable_priced_support
                && self.admission_dof_sigma2.is_some()
                && selected.len() > 1
            {
                let best = selected
                    .iter()
                    .map(|candidate| candidate.score)
                    .fold(f64::NEG_INFINITY, f64::max);
                selected.retain(|candidate| candidate.score > 0.0 || candidate.score == best);
            }
            selected.sort_by_key(|candidate| candidate.atom);
            let row_indices: Vec<u32> =
                selected.iter().map(|candidate| candidate.atom as u32).collect();
            let row_gates: Vec<f64> =
                selected.iter().map(|candidate| candidate.score).collect();
            let row_coords: Vec<f64> = selected
                .into_iter()
                .flat_map(|candidate| candidate.coords)
                .collect();
            Ok((row_indices, row_gates, row_coords))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut indices = Vec::with_capacity(target.nrows());
        let mut gate_params = Vec::with_capacity(target.nrows());
        let mut coords = Vec::with_capacity(target.nrows());
        for (row_indices, row_gates, row_coords) in per_row {
            indices.push(row_indices);
            gate_params.push(row_gates);
            coords.push(row_coords);
        }
        let atom_specs = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, template)| SaeAssignmentAtomSpec {
                latent_dim: template.latent_dim(),
                id_mode: gam_terms::latent::LatentIdMode::None,
                manifold: template.basis_kind().latent_manifold(template.latent_dim()),
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
        let mut routed = Self::new(self.atoms.clone(), assignment)?;
        routed.decoder_fista_passes = self.decoder_fista_passes;
        routed.admission_dof_sigma2 = self.admission_dof_sigma2;
        routed.variable_priced_support = self.variable_priced_support;
        routed.admission_usage_amortized = self.admission_usage_amortized;
        // Both routing settings must survive the rebuild too. The early-return
        // path above carries them; omitting them here disarmed the ranking
        // after the first reroute, so a fit asked for exact ranking got it for
        // one cycle and grid ranking thereafter -- and a refined grid never
        // took effect at all, which made an A/B over it train bit-identically.
        routed.exact_affine_ranking = self.exact_affine_ranking;
        routed.grid_refinement = self.grid_refinement;
        Ok(routed)
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
        let row_layout = SaeRowLayout::from_assignment_state(&self.assignment)?;
        let per_row_dims = (0..self.n_obs())
            .map(|row| row_layout.row_q_active(row))
            .collect::<Vec<_>>();
        let mut system = ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(
            per_row_dims,
            beta_dim,
            0,
        );
        let mut linearized_rows = Vec::with_capacity(self.n_obs());
        let mut hbb_diag = Array1::<f64>::zeros(beta_dim);
        // One evaluation scratch for the whole assembly (#2575); `blocks` still
        // owns a copy of each active basis row because the linearized operator
        // outlives this loop.
        let mut scratch = ActiveAtomScratch::default();
        for row in 0..self.n_obs() {
            let q = row_layout.row_q_active(row);
            let mut fitted = Array1::<f64>::zeros(self.output_dim);
            let mut jacobian = Array2::<f64>::zeros((q, self.output_dim));
            let mut blocks = Vec::with_capacity(self.assignment.support_indices(row).len());
            for slot in 0..self.assignment.support_indices(row).len() {
                let atom_idx = self.assignment.support_indices(row)[slot] as usize;
                self.fill_active(row, slot, &mut scratch)?;
                fitted += &scratch.decoded;
                let cursor = row_layout.coord_starts[row][slot];
                for axis in 0..scratch.jacobian.nrows() {
                    jacobian
                        .row_mut(cursor + axis)
                        .assign(&scratch.jacobian.row(axis));
                }
                let phi = scratch.phi_row();
                for basis in 0..phi.len() {
                    let base = beta_offsets[atom_idx] + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        hbb_diag[base + channel] += phi[basis] * phi[basis];
                    }
                }
                blocks.push(SupportBasisBlock {
                    beta_offset: beta_offsets[atom_idx],
                    phi: phi.to_owned(),
                });
            }
            let residual = &target.row(row) - &fitted;
            system.rows[row].htt.assign(&jacobian.dot(&jacobian.t()));
            system.rows[row].gt.assign(&(-jacobian.dot(&residual)));
            let periods = self
                .assignment
                .support_indices(row)
                .iter()
                .flat_map(|&atom| self.atom_axis_periods(atom as usize).iter().copied())
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
                .smooth_penalty()
                .dot(self.atoms[atom].decoder_coefficients());
            for basis in 0..m {
                let base = beta_offsets[atom] + basis * self.output_dim;
                for channel in 0..self.output_dim {
                    system.gb[base + channel] += lambda * sb[[basis, channel]];
                    hbb_diag[base + channel] +=
                        lambda * self.atoms[atom].smooth_penalty()[[basis, basis]];
                }
            }
        }
        // Inverted index for `apply`'s scatter. Rows are walked in order here, so
        // each atom's list comes out sorted by row without a separate sort.
        let atom_of_offset: std::collections::HashMap<usize, usize> = beta_offsets
            .iter()
            .enumerate()
            .map(|(atom, &offset)| (offset, atom))
            .collect();
        let mut atom_blocks: Vec<Vec<(u32, u32)>> = vec![Vec::new(); beta_offsets.len()];
        for (row_index, row) in linearized_rows.iter().enumerate() {
            for (block_index, block) in row.blocks.iter().enumerate() {
                let atom = *atom_of_offset.get(&block.beta_offset).ok_or_else(|| {
                    format!(
                        "SupportBetaOperator: block beta_offset {} matches no atom",
                        block.beta_offset
                    )
                })?;
                atom_blocks[atom].push((row_index as u32, block_index as u32));
            }
        }
        // #2502: stage the SAE residency payload (#1017) so the reduced-Schur
        // PCG reads its Jacobi diagonal and its matvec off resident (L_i, Y_i)
        // factors instead of probing the matrix-free operator once per
        // (row, beta-column) with a triangular solve each. Profiled on the
        // manifold coupled joint-Newton phase: the probe build was 93% of all
        // cycles and priced ONE Newton cycle at hours (K=1024, P=128, n=50k).
        // Only `p`, `beta_dim`, `a_phi`, and `local_jac` are read on this
        // (CPU, InexactPcg) lane; every device consumer gates on
        // `frame.is_some()` or Direct mode, so the empty penalty blocks can
        // never engage a device kernel that would drop the smooth term.
        let width = self.output_dim;
        let a_phi: Vec<Vec<(usize, f64)>> = linearized_rows
            .iter()
            .map(|row| {
                row.blocks
                    .iter()
                    .flat_map(|block| {
                        block.phi.iter().enumerate().map(move |(basis, &value)| {
                            (block.beta_offset + basis * width, value)
                        })
                    })
                    .collect()
            })
            .collect();
        let local_jac: Vec<Vec<f64>> = linearized_rows
            .iter()
            .map(|row| row.jacobian.iter().copied().collect())
            .collect();
        // #2502 stage 2: populate the H_bb block families so the DEVICE PCG
        // solves the TRUE system. The legacy kernel composes H_bb entirely from
        // `sparse_g_blocks` (data-fit Gram `A (x) I_p`, mu-space offsets, both
        // orientations incl. the diagonal, accumulated from the SAME phi rows
        // the CPU operator gathers) + `smooth_blocks` (`lambda S_k (x) I_p` at
        // the atom's beta offset) + the caller's ridge. With the coupled phase's
        // preconditioner fixed (#1017 residency above), the profile moved to the
        // CG's H_bb matvec itself (85% across the two apply passes at K=1024,
        // n=50k) - which is exactly the part the device kernel executes.
        let mut mu_offsets = Vec::with_capacity(self.k_atoms());
        {
            let mut cursor = 0usize;
            for atom in &self.atoms {
                mu_offsets.push(cursor);
                cursor += atom.basis_size();
            }
        }
        let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
            std::collections::BTreeMap::new();
        for row in &linearized_rows {
            for block_i in &row.blocks {
                let atom_i = atom_of_offset[&block_i.beta_offset];
                for block_j in &row.blocks {
                    let atom_j = atom_of_offset[&block_j.beta_offset];
                    let blk = g_blocks.entry((atom_i, atom_j)).or_insert_with(|| {
                        Array2::<f64>::zeros((block_i.phi.len(), block_j.phi.len()))
                    });
                    for li in 0..block_i.phi.len() {
                        let wi = block_i.phi[li];
                        if wi == 0.0 {
                            continue;
                        }
                        for lj in 0..block_j.phi.len() {
                            blk[[li, lj]] += wi * block_j.phi[lj];
                        }
                    }
                }
            }
        }
        let sparse_g_blocks: Vec<gam_solve::arrow_schur::SparseGBlock> = g_blocks
            .into_iter()
            .filter_map(|((atom_i, atom_j), data)| {
                if data.iter().all(|&v| v == 0.0) {
                    None
                } else {
                    Some(gam_solve::arrow_schur::SparseGBlock {
                        row_off: mu_offsets[atom_i],
                        col_off: mu_offsets[atom_j],
                        data,
                    })
                }
            })
            .collect();
        let smooth_blocks: Vec<gam_solve::arrow_schur::DeviceSaeSmoothBlock> = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom_idx, atom)| gam_solve::arrow_schur::DeviceSaeSmoothBlock {
                global_offset: beta_offsets[atom_idx],
                factor_a: atom.smooth_penalty() * lambda_smooth[atom_idx],
            })
            .collect();
        system.set_device_sae_pcg_data(gam_solve::arrow_schur::DeviceSaePcgData {
            p: width,
            beta_dim,
            a_phi: a_phi.into(),
            local_jac: local_jac.into(),
            smooth_blocks,
            sparse_g_blocks,
            frame: None,
        });
        let operator = Arc::new(SupportBetaOperator {
            rows: linearized_rows,
            atom_blocks,
            beta_offsets: beta_offsets.clone(),
            basis_sizes: self.atoms.iter().map(SaeManifoldAtom::basis_size).collect(),
            penalties: self
                .atoms
                .iter()
                .map(|atom| atom.smooth_penalty().clone())
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

    fn support_outer_differential_rows(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        beta_offsets: &[usize],
    ) -> Result<Vec<SupportOuterDifferentialRow>, String> {
        let mut rows = Vec::with_capacity(self.n_obs());
        let mut scratch = ActiveAtomScratch::default();
        for row in 0..self.n_obs() {
            let support = self.assignment.support_indices(row);
            let q = support
                .iter()
                .map(|&atom| self.assignment.atom_coord_dim(atom as usize))
                .sum::<usize>();
            let mut fitted = Array1::<f64>::zeros(self.output_dim);
            let mut jacobian = Array2::<f64>::zeros((q, self.output_dim));
            let mut prior_hessian_remainder = Array1::<f64>::zeros(q);
            let mut prior_majorizer_derivative = Array1::<f64>::zeros(q);
            let mut slots = Vec::with_capacity(support.len());
            let mut coordinate_offset = 0usize;
            for slot in 0..support.len() {
                let atom_index = support[slot] as usize;
                let atom = &self.atoms[atom_index];
                let d = atom.latent_dim();
                let m = atom.basis_size();
                self.fill_active(row, slot, &mut scratch)?;
                fitted += &scratch.decoded;
                jacobian
                    .slice_mut(ndarray::s![coordinate_offset..coordinate_offset + d, ..])
                    .assign(&scratch.jacobian);

                let coordinates = self.assignment.coords_for_slot(row, slot);
                let coordinate_view = ArrayView2::from_shape((1, d), coordinates).map_err(
                    |error| {
                        format!(
                            "support outer differential: row {row}, atom {atom_index} coordinate view: {error}"
                        )
                    },
                )?;
                let second = if let Some(evaluator) = atom.basis_second_jet.as_ref() {
                    evaluator.second_jet(coordinate_view)?
                } else {
                    let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                        format!(
                            "support outer differential: atom {atom_index} ('{}') has no analytic basis evaluator",
                            atom.name
                        )
                    })?;
                    evaluator
                        .second_jet_dyn(coordinate_view)
                        .ok_or_else(|| {
                            format!(
                                "support outer differential: atom {atom_index} ('{}') does not expose an analytic second jet",
                                atom.name
                            )
                        })??
                };
                if second.dim() != (1, m, d, d) {
                    return Err(format!(
                        "support outer differential: row {row}, atom {atom_index} second jet shape {:?} != (1, {m}, {d}, {d})",
                        second.dim(),
                    ));
                }

                let periods = self.atom_axis_periods(atom_index);
                for axis in 0..d {
                    let alpha = ard_precisions[atom_index][axis];
                    let coordinate = coordinates[axis];
                    let prior = ArdAxisPrior::eval(alpha, coordinate, periods[axis]);
                    prior_hessian_remainder[coordinate_offset + axis] =
                        prior.negative_hessian_remainder();
                    prior_majorizer_derivative[coordinate_offset + axis] = match periods[axis] {
                        None => 0.0,
                        Some(period) => {
                            let kappa = std::f64::consts::TAU / period;
                            let phase = kappa * coordinate;
                            -alpha
                                * kappa
                                * phase.sin()
                                * ArdAxisPrior::clamp_slope(phase.cos())
                        }
                    };
                }

                slots.push(SupportOuterDifferentialSlot {
                    atom: atom_index,
                    coordinate_offset,
                    beta_offset: beta_offsets[atom_index],
                    phi: scratch.phi_row().to_owned(),
                    jet: scratch.jet.slice(ndarray::s![0, .., ..]).to_owned(),
                    second_jet: second.slice(ndarray::s![0, .., .., ..]).to_owned(),
                });
                coordinate_offset += d;
            }
            rows.push(SupportOuterDifferentialRow {
                slots,
                jacobian,
                residual: &target.row(row) - &fitted,
                prior_hessian_remainder,
                prior_majorizer_derivative,
            });
        }
        Ok(rows)
    }

    fn support_outer_exact_hessian_apply(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
        vector: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let mut out = support_arrow_majorizer_apply(system, vector)?;
        for (row_index, row) in rows.iter().enumerate() {
            let row_start = system.row_offsets[row_index];
            for slot in &row.slots {
                let atom = &self.atoms[slot.atom];
                let d = atom.latent_dim();
                let m = atom.basis_size();
                let local_t = vector.t.slice(ndarray::s![
                    row_start + slot.coordinate_offset
                        ..row_start + slot.coordinate_offset + d
                ]);

                // Exact residual curvature in the coordinate-coordinate block:
                // `-sum_p r_p d2f_p/dt_a dt_b`.
                for axis_a in 0..d {
                    let mut correction = 0.0_f64;
                    for axis_b in 0..d {
                        let mut residual_second = 0.0_f64;
                        for basis in 0..m {
                            let coefficient = slot.second_jet[[basis, axis_a, axis_b]];
                            for output in 0..self.output_dim {
                                residual_second += coefficient
                                    * atom.decoder_coefficients()[[basis, output]]
                                    * row.residual[output];
                            }
                        }
                        correction -= residual_second * local_t[axis_b];
                    }
                    // Exact coordinate-decoder residual cross block.
                    for basis in 0..m {
                        let derivative = slot.jet[[basis, axis_a]];
                        for output in 0..self.output_dim {
                            correction -= derivative
                                * row.residual[output]
                                * vector.beta
                                    [slot.beta_offset + basis * self.output_dim + output];
                        }
                    }
                    correction +=
                        row.prior_hessian_remainder[slot.coordinate_offset + axis_a]
                            * local_t[axis_a];
                    out.t[row_start + slot.coordinate_offset + axis_a] += correction;
                }

                // Symmetric decoder-coordinate residual cross block.
                for basis in 0..m {
                    let mut basis_direction = 0.0_f64;
                    for axis in 0..d {
                        basis_direction += slot.jet[[basis, axis]] * local_t[axis];
                    }
                    for output in 0..self.output_dim {
                        out.beta[slot.beta_offset + basis * self.output_dim + output] -=
                            row.residual[output] * basis_direction;
                    }
                }
            }
        }
        Ok(out)
    }

    fn support_outer_dense_hessian_pencil(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
        t_len: usize,
        beta_len: usize,
    ) -> Result<SupportOuterDensePencil, String> {
        let dim = t_len
            .checked_add(beta_len)
            .ok_or_else(|| "support outer Hessian-pencil dimension overflow".to_string())?;
        if dim == 0 {
            return Err("support outer Hessian pencil has zero dimension".to_string());
        }
        let mut exact = Array2::<f64>::zeros((dim, dim));
        let mut majorizer = Array2::<f64>::zeros((dim, dim));
        for column in 0..dim {
            let mut unit = SaeArrowVector {
                t: Array1::zeros(t_len),
                beta: Array1::zeros(beta_len),
            };
            if column < t_len {
                unit.t[column] = 1.0;
            } else {
                unit.beta[column - t_len] = 1.0;
            }
            let applied = self.support_outer_exact_hessian_apply(system, rows, &unit)?;
            exact
                .slice_mut(ndarray::s![..t_len, column])
                .assign(&applied.t);
            exact
                .slice_mut(ndarray::s![t_len.., column])
                .assign(&applied.beta);
            let applied_b = support_arrow_majorizer_apply(system, &unit)?;
            majorizer
                .slice_mut(ndarray::s![..t_len, column])
                .assign(&applied_b.t);
            majorizer
                .slice_mut(ndarray::s![t_len.., column])
                .assign(&applied_b.beta);
        }
        for row in 0..dim {
            for column in 0..row {
                let symmetric = 0.5 * (exact[[row, column]] + exact[[column, row]]);
                exact[[row, column]] = symmetric;
                exact[[column, row]] = symmetric;
                let symmetric_b =
                    0.5 * (majorizer[[row, column]] + majorizer[[column, row]]);
                majorizer[[row, column]] = symmetric_b;
                majorizer[[column, row]] = symmetric_b;
            }
        }
        if exact
            .iter()
            .chain(majorizer.iter())
            .any(|value| !value.is_finite())
        {
            return Err("support outer adjoint Hessian pencil is non-finite".to_string());
        }
        let (b_eigenvalues, b_eigenvectors) = majorizer
            .eigh(Side::Lower)
            .map_err(|error| format!("support outer adjoint majorizer eigensystem: {error}"))?;
        let b_scale = b_eigenvalues
            .iter()
            .fold(0.0_f64, |current, &value| current.max(value.abs()));
        if !(b_scale.is_finite() && b_scale > 0.0) {
            return Err("support outer adjoint majorizer has zero numerical rank".to_string());
        }
        let b_rank_floor = b_scale * f64::EPSILON * dim.max(1) as f64;
        // Deflate the majorizer's unresolved directions to unit stiffness rather
        // than refusing the pencil. The support charts carry genuine
        // decoder/coordinate gauges, so a numerically singular `B` is the
        // expected case; `joint_newton_step` states the same treatment for the
        // per-row coordinate block ("refusing to factor it would discard the
        // whole joint step over one row. Deflating that direction to unit
        // stiffness is what the dense manifold lane already does").
        //
        // DROPPING the directions instead -- restricting to `range(B)` -- is not
        // available: `A` does not preserve that subspace, and the reduced solve
        // measured a pseudoinverse residual of 2.710186e-2 against a 1.628494e-4
        // certificate. Clamping to `b_scale` (the STIFFEST resolved curvature,
        // never the floor) keeps `B` symmetric positive definite, so the
        // whitener stays square and invertible, `A v = mu B v` holds exactly,
        // and by Sylvester's law of inertia every curvature SIGN still equals
        // its Euclidean sign -- the only thing the negative-mode test reads.
        // Clamping to the floor would divide by a near-zero stiffness and
        // manufacture curvature in precisely the directions that have none.
        let stiffened: Vec<f64> = b_eigenvalues
            .iter()
            .map(|&value| if value > b_rank_floor { value } else { b_scale })
            .collect();
        let mut whitener = Array2::<f64>::zeros((dim, dim));
        for mode in 0..dim {
            let inverse_sqrt = stiffened[mode].sqrt().recip();
            for row in 0..dim {
                whitener[[row, mode]] = b_eigenvectors[[row, mode]] * inverse_sqrt;
            }
        }
        // The stored majorizer must be the one the eigenrelation was solved
        // against: the pseudoinverse builds its range right-hand side as `B v`
        // and certifies `A x` against it, so a `B` disagreeing with the
        // whitening would fail that certificate by construction.
        let mut stiffened_majorizer = Array2::<f64>::zeros((dim, dim));
        for mode in 0..dim {
            let stiffness = stiffened[mode];
            for row in 0..dim {
                let scaled = stiffness * b_eigenvectors[[row, mode]];
                for column in 0..dim {
                    stiffened_majorizer[[row, column]] +=
                        scaled * b_eigenvectors[[column, mode]];
                }
            }
        }
        for row in 0..dim {
            for column in 0..row {
                let symmetric = 0.5
                    * (stiffened_majorizer[[row, column]]
                        + stiffened_majorizer[[column, row]]);
                stiffened_majorizer[[row, column]] = symmetric;
                stiffened_majorizer[[column, row]] = symmetric;
            }
        }
        let mut whitened_exact = whitener.t().dot(&exact).dot(&whitener);
        for row in 0..dim {
            for column in 0..row {
                let symmetric =
                    0.5 * (whitened_exact[[row, column]] + whitened_exact[[column, row]]);
                whitened_exact[[row, column]] = symmetric;
                whitened_exact[[column, row]] = symmetric;
            }
        }
        let (curvatures, curvature_vectors) = whitened_exact
            .eigh(Side::Lower)
            .map_err(|error| format!("support outer adjoint (A, B) eigensystem: {error}"))?;
        let minimum_vector = curvature_vectors.column(0);
        let minimum_curvature = curvatures[0];
        let minimum_residual = &whitened_exact.dot(&minimum_vector)
            - &(minimum_vector.to_owned() * minimum_curvature);
        let eigensystem_scale = curvatures
            .iter()
            .fold(0.0_f64, |current, &value| current.max(value.abs()))
            .max(1.0);
        let minimum_backward_error = minimum_residual.dot(&minimum_residual).sqrt()
            + f64::EPSILON * dim.max(1) as f64 * eigensystem_scale;
        if !(minimum_backward_error.is_finite() && minimum_backward_error >= 0.0) {
            return Err(
                "support outer adjoint generalized eigensystem has non-finite backward error"
                    .to_string(),
            );
        }
        Ok(SupportOuterDensePencil {
            exact,
            majorizer: stiffened_majorizer,
            curvatures,
            generalized_vectors: whitener.dot(&curvature_vectors),
            minimum_backward_error,
        })
    }

    fn support_outer_negative_curvature_mode(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
    ) -> Result<Option<SupportNegativeCurvatureMode>, String> {
        let t_len = *system.row_offsets.last().unwrap_or(&0);
        let beta_len = system.k;
        let pencil = self.support_outer_dense_hessian_pencil(system, rows, t_len, beta_len)?;
        let curvature = pencil.curvatures[0];
        let resolution = pencil
            .minimum_backward_error
            .max(f64::EPSILON.sqrt());
        if curvature >= -resolution {
            return Ok(None);
        }
        let direction = pencil.generalized_vectors.column(0);
        Ok(Some(SupportNegativeCurvatureMode {
            curvature,
            backward_error: pencil.minimum_backward_error,
            direction: SaeArrowVector {
                t: direction.slice(ndarray::s![..t_len]).to_owned(),
                beta: direction.slice(ndarray::s![t_len..]).to_owned(),
            },
        }))
    }

    /// Moore--Penrose solve of the exact stationarity Hessian on an admitted
    /// small problem.  The support charts carry genuine decoder/coordinate
    /// gauges, so asking an ordinary inverse to resolve their zero modes is a
    /// category error: the profiled derivative is defined on the identifiable
    /// quotient and therefore uses `A^+`.
    ///
    /// The generalized eigensystem `(A, B)` is assembled from the same exact-A
    /// and majorizer-B matrix-free applies as the large-system Krylov route.
    /// Its dimensionless curvature ratio is resolved at `sqrt(eps)`, the same
    /// scalar-derived IFT quotient boundary used by the dense SAE lane;
    /// materially negative curvature is refused because a stationary saddle is
    /// not a fitted inner optimum.
    fn support_outer_dense_pseudoinverse_apply(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let t_len = rhs.t.len();
        let beta_len = rhs.beta.len();
        let dim = t_len
            .checked_add(beta_len)
            .ok_or_else(|| "support outer adjoint dimension overflow".to_string())?;
        if dim == 0 {
            return Ok(SaeArrowVector {
                t: Array1::zeros(0),
                beta: Array1::zeros(0),
            });
        }
        let pencil = self.support_outer_dense_hessian_pencil(system, rows, t_len, beta_len)?;
        let quotient_floor = f64::EPSILON
            .sqrt()
            .max(pencil.minimum_backward_error);
        if let Some(&negative) = pencil
            .curvatures
            .iter()
            .find(|&&value| value < -quotient_floor)
        {
            let negative_floor = -quotient_floor;
            return Err(format!(
                "support outer adjoint has resolved generalized negative curvature {negative:.6e} below the IFT quotient floor {negative_floor:.6e}"
            ));
        }
        let mut flat_rhs = Array1::<f64>::zeros(dim);
        flat_rhs.slice_mut(ndarray::s![..t_len]).assign(&rhs.t);
        flat_rhs
            .slice_mut(ndarray::s![t_len..])
            .assign(&rhs.beta);
        let mut solution = Array1::<f64>::zeros(dim);
        let mut range_rhs = Array1::<f64>::zeros(dim);
        for mode in 0..dim {
            let curvature = pencil.curvatures[mode];
            if curvature <= quotient_floor {
                continue;
            }
            let vector = pencil.generalized_vectors.column(mode);
            let projection = vector.dot(&flat_rhs);
            solution.scaled_add(projection / curvature, &vector);
            let b_vector = pencil.majorizer.dot(&vector);
            range_rhs.scaled_add(projection, &b_vector);
        }
        let residual = &pencil.exact.dot(&solution) - &range_rhs;
        let residual_norm = residual.dot(&residual).sqrt();
        let range_norm = range_rhs.dot(&range_rhs).sqrt();
        // Backward-error certificate for a linear solve, denominated in the
        // quantity it bounds: `||A x - b|| <= dim * EPSILON * (||A|| ||x|| + ||b||)`.
        //
        // The previous form was `sqrt(EPSILON) * ||b||`, which drops the
        // `||A|| ||x||` term entirely and carries no dependence on the
        // operator's scale. Measured at this site on two peel fixtures, the
        // dropped term is worth 5.5736e8x and 4.0229e7x of `||b||` -- a
        // different factor each time, so no constant can stand in for it. Over
        // the SAME two fixtures the achieved residual is
        // `5.0748` and `5.0492` times `EPSILON * (||A|| ||x|| + ||b||)`:
        // agreement to three significant figures across two problems whose
        // overshoot against the old bound differed by 14x. The solve is
        // backward stable; only the yardstick was wrong.
        //
        // This is not a widened tolerance. Where `||A|| ||x|| ~ ||b||` -- a well
        // conditioned solve -- it is TIGHTER than the old bound by the
        // `sqrt(EPSILON)`-versus-`EPSILON` factor. It admits more only where the
        // operator's own scale says it must.
        let exact_scale = pencil
            .exact
            .iter()
            .fold(0.0_f64, |current, &value| current.max(value.abs()));
        let solution_norm = solution.dot(&solution).sqrt();
        let backward_error_bound = dim.max(1) as f64
            * f64::EPSILON
            * (exact_scale * solution_norm + range_norm).max(1.0);
        if !(residual_norm.is_finite() && residual_norm <= backward_error_bound) {
            return Err(format!(
                "support outer adjoint pseudoinverse residual {residual_norm:.6e} exceeds its \
                 backward-error certificate {backward_error_bound:.6e} \
                 (dim={dim}, ||A||={exact_scale:.6e}, ||x||={solution_norm:.6e}, \
                 ||b||={range_norm:.6e})"
            ));
        }
        if solution.iter().any(|value| !value.is_finite()) {
            return Err("support outer adjoint pseudoinverse is non-finite".to_string());
        }
        Ok(SaeArrowVector {
            t: solution.slice(ndarray::s![..t_len]).to_owned(),
            beta: solution.slice(ndarray::s![t_len..]).to_owned(),
        })
    }

    /// Return `A^+ Gamma`, the one adjoint needed for the implicit derivative of
    /// the profiled reduced-Schur log determinant. `Gamma` is the exact
    /// derivative of the frozen rational surrogate with respect to the fitted
    /// inner state, assembled from the surrogate's own low-rank derivative
    /// vectors. `A` is the exact stationarity Jacobian of the penalized inner
    /// objective, not its Gauss--Newton majorizer.
    pub(crate) fn support_reduced_logdet_profile_adjoint(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        system: &ArrowSchurSystem,
        derivative_vectors: &[Array1<f64>],
    ) -> Result<SaeArrowVector, String> {
        if derivative_vectors.is_empty() {
            return Err(
                "support reduced-logdet profile adjoint requires a non-empty derivative bundle"
                    .to_string(),
            );
        }
        if derivative_vectors
            .iter()
            .any(|vector| vector.len() != system.k || vector.iter().any(|value| !value.is_finite()))
        {
            return Err(format!(
                "support reduced-logdet profile adjoint requires finite vectors of border width {}",
                system.k
            ));
        }
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        if beta_dim != system.k {
            return Err(format!(
                "support reduced-logdet profile adjoint beta layout {beta_dim} != system border {}",
                system.k
            ));
        }
        let rows = self.support_outer_differential_rows(target, ard_precisions, &beta_offsets)?;
        let factors = CpuBatchedBlockSolver
            .factor_blocks(&system.rows, 0.0, system.d, true)
            .map_err(|error| {
                format!(
                    "support reduced-logdet profile adjoint row factorization: {error}"
                )
            })?;
        let coordinate_dim = *system.row_offsets.last().unwrap_or(&0);
        let mut gamma = SaeArrowVector {
            t: Array1::<f64>::zeros(coordinate_dim),
            beta: Array1::<f64>::zeros(beta_dim),
        };
        let inverse_rank = 1.0 / derivative_vectors.len() as f64;
        for border_vector in derivative_vectors {
            for (row_index, row) in rows.iter().enumerate() {
                let row_start = system.row_offsets[row_index];
                let q = system.row_dims[row_index];
                let cross = support_arrow_cross_forward(system, row_index, border_vector.view())?;
                let mut local_t = CpuBatchedBlockSolver.solve_block_vector(
                    factors.factor(row_index),
                    cross.view(),
                );
                local_t.mapv_inplace(|value| -value);

                // Directional model response `df[z] = J z_t + D z_beta` for
                // the Schur envelope vector `z_t = -H_tt^-1 H_tbeta z_beta`.
                let mut directional_fit = row.jacobian.t().dot(&local_t);
                for slot in &row.slots {
                    let atom = &self.atoms[slot.atom];
                    for basis in 0..atom.basis_size() {
                        let weight = slot.phi[basis];
                        for output in 0..self.output_dim {
                            directional_fit[output] += weight
                                * border_vector
                                    [slot.beta_offset + basis * self.output_dim + output];
                        }
                    }
                }

                for slot in &row.slots {
                    let atom = &self.atoms[slot.atom];
                    let d = atom.latent_dim();
                    let m = atom.basis_size();
                    for axis_w in 0..d {
                        let mut derivative_fit = Array1::<f64>::zeros(self.output_dim);
                        // Coordinate derivative of `J^T z_t`.
                        for axis_a in 0..d {
                            let coefficient_t = local_t[slot.coordinate_offset + axis_a];
                            for basis in 0..m {
                                let coefficient =
                                    coefficient_t * slot.second_jet[[basis, axis_a, axis_w]];
                                for output in 0..self.output_dim {
                                    derivative_fit[output] += coefficient
                                        * atom.decoder_coefficients()[[basis, output]];
                                }
                            }
                        }
                        // Coordinate derivative of `D z_beta`.
                        for basis in 0..m {
                            let derivative = slot.jet[[basis, axis_w]];
                            for output in 0..self.output_dim {
                                derivative_fit[output] += derivative
                                    * border_vector
                                        [slot.beta_offset + basis * self.output_dim + output];
                            }
                        }
                        let local_index = slot.coordinate_offset + axis_w;
                        gamma.t[row_start + local_index] += inverse_rank
                            * (2.0 * directional_fit.dot(&derivative_fit)
                                + row.prior_majorizer_derivative[local_index]
                                    * local_t[local_index]
                                    * local_t[local_index]);
                    }

                    // Decoder derivative of `J^T z_t`; `D z_beta` is
                    // decoder-independent.
                    for basis in 0..m {
                        let mut jet_direction = 0.0_f64;
                        for axis in 0..d {
                            jet_direction += slot.jet[[basis, axis]]
                                * local_t[slot.coordinate_offset + axis];
                        }
                        for output in 0..self.output_dim {
                            gamma.beta
                                [slot.beta_offset + basis * self.output_dim + output] +=
                                inverse_rank * 2.0 * directional_fit[output] * jet_direction;
                        }
                    }
                }
                // `assert_eq!` rather than `debug_assert_eq!`: the scanner bans the
                // debug form, and a length invariant that only holds in debug builds
                // is not an invariant. Integer compare, negligible in release.
                assert_eq!(local_t.len(), q);
            }
        }
        if gamma
            .t
            .iter()
            .chain(gamma.beta.iter())
            .any(|value| !value.is_finite())
        {
            return Err(
                "support reduced-logdet profile adjoint assembled a non-finite theta derivative"
                    .to_string(),
            );
        }
        let full_dim = coordinate_dim
            .checked_add(beta_dim)
            .ok_or_else(|| "support reduced-logdet profile adjoint dimension overflow".to_string())?;
        // A dense pseudoinverse is admitted only when assembling all `dim`
        // analytic columns costs no more operator directions than the rational
        // derivative bundle already paid for, and its complete eigensystem
        // workspace fits the cgroup-aware in-core ledger.  This is a scale
        // transition derived from work and storage, not a dimension knob.
        let dense_workspace = (full_dim as u128)
            .saturating_mul(full_dim as u128)
            .saturating_mul(std::mem::size_of::<f64>() as u128)
            .saturating_mul(6);
        let in_core_budget = crate::manifold::sae_host_in_core_budget_bytes().0 as u128;
        if full_dim <= derivative_vectors.len() && dense_workspace <= in_core_budget {
            self.support_outer_dense_pseudoinverse_apply(system, &rows, &gamma)
        } else {
            solve_b_preconditioned_gmres_with(
                &gamma,
                |vector| self.support_outer_exact_hessian_apply(system, &rows, vector),
                |rhs| support_arrow_majorizer_inverse(system, &factors, rhs),
            )
            .map_err(|error| format!("support reduced-logdet profile adjoint solve: {error}"))
        }
    }

    /// Evaluate one active `(row, slot)` pair into caller-owned storage.
    ///
    /// The allocating counterpart this replaces (`evaluate_active`) built six
    /// fresh arrays per call and was called once per active pair per pass —
    /// `n·support_k` times per sweep (#2575). The evaluation itself is
    /// unchanged: it delegates to [`Self::fill_active_eval`], which is the one
    /// place that reads the evaluator and folds the decoder, so the row solve
    /// and every read-only pass now share a single producer.
    fn fill_active(
        &self,
        row: usize,
        slot: usize,
        scratch: &mut ActiveAtomScratch,
    ) -> Result<(), String> {
        let atom_idx = self.assignment.support_indices(row)[slot] as usize;
        let atom = &self.atoms[atom_idx];
        scratch.fit(atom.basis_size(), atom.latent_dim(), self.output_dim);
        let ActiveAtomScratch {
            phi,
            jet,
            decoded,
            jacobian,
        } = scratch;
        self.fill_active_eval(
            row,
            slot,
            self.assignment.coords_for_slot(row, slot),
            phi,
            jet,
            decoded,
            jacobian,
        )
    }

    /// Decode one atom's image at caller coordinates: `Φ(t)·B_k`, shape
    /// `(n, P)`. The atom's own evaluator is the single source of truth for
    /// the chart convention — callers never re-derive the basis.
    pub fn decode_atom_at(
        &self,
        atom_idx: usize,
        coords: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        if atom_idx >= self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm::decode_atom_at: atom {atom_idx} out of range K={}",
                self.k_atoms()
            ));
        }
        let atom = &self.atoms[atom_idx];
        if coords.ncols() != atom.latent_dim() {
            return Err(format!(
                "SaeSupportSparseTerm::decode_atom_at: coords width {} != atom latent dim {}",
                coords.ncols(),
                atom.latent_dim()
            ));
        }
        let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
            format!("SaeSupportSparseTerm::decode_atom_at: atom {atom_idx} has no evaluator")
        })?;
        let (phi, _jet) = evaluator.evaluate(coords)?;
        Ok(phi.dot(atom.decoder_coefficients()))
    }

    fn reconstruct_row_into(
        &self,
        row: usize,
        scratch: &mut ActiveAtomScratch,
        fitted: &mut Array1<f64>,
    ) -> Result<(), String> {
        fitted.fill(0.0);
        for slot in 0..self.assignment.support_indices(row).len() {
            self.fill_active(row, slot, scratch)?;
            *fitted += &scratch.decoded;
        }
        Ok(())
    }

    /// Direct active-row reconstruction. No K-wide gate or basis row exists.
    /// Rows are independent reads of shared state, so they decode in parallel.
    ///
    /// #2575: the per-row decode used to allocate a fresh `(P,)` row and six
    /// arrays per active pair, and the whole `(N, P)` result was collected as a
    /// `Vec` of owned rows before being copied into the output. Each rayon
    /// worker now carries ONE scratch and ONE row accumulator across all the
    /// rows it takes, and writes into its own disjoint slice of the output.
    pub fn reconstruct(&self) -> Result<Array2<f64>, String> {
        let mut fitted = Array2::<f64>::zeros((self.n_obs(), self.output_dim));
        self.reconstruct_into(&mut fitted)?;
        Ok(fitted)
    }

    /// [`Self::reconstruct`] into a caller-owned buffer. This exists because
    /// `solve_fixed_point` maintains ONE fitted matrix across its cycles
    /// instead of decoding all `n x top_k` active pairs from scratch several
    /// times per cycle — profiled at 97% of all frames on the #2502 lane, the
    /// full-matrix decode WAS the fit's runtime, and both sweeps already know
    /// exactly which rows they changed.
    fn reconstruct_into(&self, fitted: &mut Array2<f64>) -> Result<(), String> {
        if fitted.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::reconstruct_into: buffer {:?} != ({}, {})",
                fitted.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        let output_dim = self.output_dim;
        fitted
            .axis_chunks_iter_mut(ndarray::Axis(0), RECONSTRUCT_ROW_CHUNK)
            .into_par_iter()
            .enumerate()
            .try_for_each(|(chunk, mut block)| -> Result<(), String> {
                let mut scratch = ActiveAtomScratch::default();
                let mut row_fitted = Array1::<f64>::zeros(output_dim);
                let base = chunk * RECONSTRUCT_ROW_CHUNK;
                for local in 0..block.nrows() {
                    self.reconstruct_row_into(base + local, &mut scratch, &mut row_fitted)?;
                    block.row_mut(local).assign(&row_fitted);
                }
                Ok(())
            })?;
        Ok(())
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
            // alpha == 0.0 is the typed prior EXEMPTION for an axis whose
            // prior family is constant on its manifold (any axis of an
            // ambient unit vector): the prior evaluates to exact zeros there,
            // and the MacKay update never re-selects an exempt axis. Negative,
            // non-finite, and (to keep the exemption deliberate) subnormal
            // values remain refused.
            if values.len() != self.assignment.atom_coord_dim(atom)
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(format!(
                    "SaeSupportSparseTerm: atom {atom} ARD must contain {} finite non-negative precisions",
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
        self.penalized_objective_with_residual(&residual, lambda_smooth, ard_precisions)
    }

    /// [`Self::penalized_objective`] against a caller-supplied residual.
    pub fn penalized_objective_with_residual(
        &self,
        residual: &Array2<f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        self.validate_smoothing(lambda_smooth)?;
        self.validate_ard(ard_precisions)?;
        let mut value = 0.5 * residual.iter().map(|entry| entry * entry).sum::<f64>();
        for (atom, &lambda) in self.atoms.iter().zip(lambda_smooth) {
            let sb = atom.smooth_penalty().dot(atom.decoder_coefficients());
            value += 0.5
                * lambda
                * atom
                    .decoder_coefficients()
                    .iter()
                    .zip(sb.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
        }
        value += (0..self.n_obs())
            .into_par_iter()
            .map(|row| {
                let mut row_value = 0.0_f64;
                for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                    let atom = atom as usize;
                    let periods = self.atom_axis_periods(atom);
                    for axis in 0..self.assignment.atom_coord_dim(atom) {
                        row_value += ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            self.assignment.coords_for_slot(row, slot)[axis],
                            periods[axis],
                        )
                        .value;
                    }
                }
                row_value
            })
            .sum::<f64>();
        // #2502: the acceptance gate certifies the same priced objective the
        // router ranks by -- each atom in use charges its parameter bits at
        // the armed noise floor (objective scale: sigma2*ln2 per bit).
        if let Some(sigma2) = self.admission_dof_sigma2 {
            let l_param = 0.5 * (self.n_obs().max(2) as f64).log2();
            value += sigma2
                * std::f64::consts::LN_2
                * self
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(atom_index, _)| !self.atom_rows[*atom_index].is_empty())
                    .map(|(_, atom)| {
                        atom.basis_size() as f64 * self.output_dim as f64 * l_param
                    })
                    .sum::<f64>();
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
    /// Solve `(G + lambda*S) beta = rhs` WITHOUT ever forming `G + lambda*S`.
    ///
    /// Fellner-Schall legitimately sends `lambda` to ~1e16 for an atom the data
    /// gives no bend to: that is the ladder selecting the linear rung, not a
    /// divergence. Assembling `G + lambda*S` at that point produces a matrix
    /// whose condition number IS `lambda`, so the rank floor
    /// `solve_psd_minimum_norm` derives from the largest eigenvalue
    /// (`eps * max_eig * m`) grows past the atom's real least-squares
    /// information in `null(S)`, which is then misread as null space and the
    /// solve refuses. The data is not missing; the floor is set by the penalty.
    ///
    /// Diagonalising `S` and applying the Jacobi scaling
    /// `d_i = 1/sqrt(1 + lambda*s_i)` removes `lambda` from the conditioning
    /// entirely: the penalty's own contribution becomes
    /// `lambda*s_i/(1 + lambda*s_i)`, which lies in `[0, 1)` for every
    /// `lambda`, up to and including the limit. What remains is the intrinsic
    /// conditioning of `G`. This is an algebraic identity -- there is no
    /// threshold, tolerance, or clamp, and both limits are exact:
    /// `s_i = 0` leaves the unpenalised restricted least squares, and
    /// `lambda*s_i -> infinity` sends that coefficient to zero.
    fn solve_penalized_normal_equations(
        gram: &Array2<f64>,
        penalty: &Array2<f64>,
        lambda: f64,
        rhs: &Array2<f64>,
        context: &str,
    ) -> Result<Array2<f64>, String> {
        let m = gram.nrows();
        if penalty.dim() != (m, m) {
            return Err(format!(
                "{context}: penalty shape {:?} does not match gram {:?}",
                penalty.dim(),
                gram.dim()
            ));
        }
        if !(lambda >= 0.0) || !lambda.is_finite() {
            return Err(format!("{context}: smoothing {lambda} is not a finite non-negative scale"));
        }

        let symmetric_penalty = (penalty + &penalty.t()) * 0.5;
        let (penalty_eigenvalues, penalty_basis) = symmetric_penalty
            .eigh(Side::Lower)
            .map_err(|error| format!("{context}: penalty eigendecomposition failed: {error}"))?;

        // A penalty with a genuinely negative direction is not a roughness
        // measure, and the scaling below would take the square root of a
        // negative number; reject it rather than silently repairing it.
        let penalty_scale = penalty_eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let penalty_tolerance = f64::EPSILON * penalty_scale * m.max(1) as f64;
        if penalty_eigenvalues
            .iter()
            .any(|value| *value < -penalty_tolerance)
        {
            return Err(format!("{context}: smoothing penalty is not positive semidefinite"));
        }

        // Rotate into the basis where the penalty is diagonal.
        let rotated_gram = penalty_basis.t().dot(gram).dot(&penalty_basis);
        let rotated_rhs = penalty_basis.t().dot(rhs);

        let mut scaling = vec![0.0_f64; m];
        for mode in 0..m {
            let eigenvalue = penalty_eigenvalues[mode].max(0.0);
            scaling[mode] = 1.0 / (1.0 + lambda * eigenvalue).sqrt();
        }

        let mut scaled = Array2::<f64>::zeros((m, m));
        for left in 0..m {
            for right in 0..m {
                scaled[[left, right]] =
                    rotated_gram[[left, right]] * scaling[left] * scaling[right];
            }
        }
        for mode in 0..m {
            let eigenvalue = penalty_eigenvalues[mode].max(0.0);
            // `lambda*s/(1 + lambda*s)`, written so that `lambda = inf` would
            // give exactly 1 rather than a NaN from `inf * 0`.
            scaled[[mode, mode]] += lambda * eigenvalue * scaling[mode] * scaling[mode];
        }

        let mut scaled_rhs = rotated_rhs;
        for mode in 0..m {
            for column in 0..scaled_rhs.ncols() {
                scaled_rhs[[mode, column]] *= scaling[mode];
            }
        }

        let solution = Self::solve_psd_minimum_norm(&scaled, &scaled_rhs, context)?;

        // Undo the Jacobi scaling, then rotate back out of the penalty basis.
        let mut unscaled = solution;
        for mode in 0..m {
            for column in 0..unscaled.ncols() {
                unscaled[[mode, column]] *= scaling[mode];
            }
        }
        Ok(penalty_basis.dot(&unscaled))
    }

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
        let scale = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let tolerance = f64::EPSILON * scale * m.max(1) as f64;
        if eigenvalues.iter().any(|value| *value < -tolerance) {
            return Err(format!(
                "{context}: normal equation is not positive semidefinite"
            ));
        }
        let projected = eigenvectors.t().dot(rhs);
        let rhs_scale = rhs.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        let rhs_tolerance = f64::EPSILON * rhs_scale * m.max(1) as f64;
        let mut scaled = Array2::<f64>::zeros(projected.dim());
        for mode in 0..m {
            if eigenvalues[mode] > tolerance {
                for column in 0..rhs.ncols() {
                    scaled[[mode, column]] = projected[[mode, column]] / eigenvalues[mode];
                }
            } else if projected
                .row(mode)
                .iter()
                .any(|value| value.abs() > rhs_tolerance)
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
    /// Greedy conflict coloring of atoms by shared rows: atoms in one color
    /// class touch pairwise-disjoint row sets, so their Gauss-Seidel updates
    /// commute exactly.
    fn decoder_conflict_colors(&self) -> Vec<Vec<usize>> {
        let mut row_atoms: Vec<Vec<u32>> = vec![Vec::new(); self.n_obs()];
        for (atom_idx, rows) in self.atom_rows.iter().enumerate() {
            for &(row, _slot) in rows {
                row_atoms[row].push(atom_idx as u32);
            }
        }
        let mut color_of: Vec<u32> = vec![u32::MAX; self.k_atoms()];
        let mut classes: Vec<Vec<usize>> = Vec::new();
        let mut used: Vec<u32> = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            used.clear();
            for &(row, _slot) in &self.atom_rows[atom_idx] {
                for &other in &row_atoms[row] {
                    let color = color_of[other as usize];
                    if color != u32::MAX {
                        used.push(color);
                    }
                }
            }
            used.sort_unstable();
            used.dedup();
            let mut color = 0u32;
            for &taken in &used {
                if taken == color {
                    color += 1;
                } else if taken > color {
                    break;
                }
            }
            color_of[atom_idx] = color;
            if classes.len() <= color as usize {
                classes.resize(color as usize + 1, Vec::new());
            }
            classes[color as usize].push(atom_idx);
        }
        classes
    }

    /// Per-atom REML smoothing by the Fellner-Schall / MacKay fixed point.
    ///
    /// Returns the updated K-length `lambda_smooth`. See the module discussion:
    /// conditional on routing and coordinates the decoder information is
    /// `(G_k + lambda_k S_k) (x) I_P`, so the update is closed-form in the same
    /// `m x m` object `decoder_sweep` factors, with `tau_k` the per-channel
    /// effective degrees of freedom and `M0_k` the penalty null space:
    ///
    /// ```text
    ///   lambda_k <- sigma^2 * P * (tau_k - M0_k) / sum_c beta_kc' S_k beta_kc
    /// ```
    ///
    /// An atom carrying no rows, or whose fitted roughness is numerically zero,
    /// has no evidence to select from and keeps its incoming lambda. That is a
    /// refusal to update, not a clamp: there is no likelihood ridge to climb.
    ///
    /// This is an OUTER-loop quantity. `solve_fixed_point` certifies at fixed
    /// smoothing, so lambda must not move inside it.
    /// Iterate the smoothing and coordinate-prior updates against each other
    /// at FIXED coordinates and decoders until they stop moving each other
    /// (#2502).
    ///
    /// The alternation this replaces refits between every update, so each
    /// refit re-estimates the coordinates under a slightly stronger prior and
    /// the prior then reads its own effect back as evidence. Measured, that
    /// feedback carries alpha's median from 0.62 to 134 over five rounds at
    /// 1M rows, with train EV climbing and held-out EV falling. Holding the
    /// fit still while the two hyperparameters converge removes the feedback:
    /// both are closed forms of the same sufficient statistics, so this is a
    /// plain fixed-point iteration, and it stops when neither moves by more
    /// than the relative resolution its own inputs were measured at.
    /// Pool the per-atom smoothing scales toward one shared scale, weighting
    /// each atom by the effective df it actually carries (#2502).
    ///
    /// Past `K > P` the dictionary is coherent, so an atom's curvature block
    /// contains its neighbours' effect and its independently-estimated
    /// lambda is fitting that contamination. Measured: REML beats fixed
    /// lambda at 6x overcompleteness and loses by 0.092 at 63x, with rows
    /// per atom held constant. The shared scale is the effective-df-weighted
    /// geometric mean; an atom's weight toward its own estimate is its share
    /// structure inherits the pooled value and a well-determined one keeps
    /// its own. The shared scale is estimated WITHIN topology groups and the
    /// shrinkage is unit-information, matching `mackay_ard_precisions` on
    /// both counts. Nothing here is tuned.
    pub fn pooled_smoothing(
        &self,
        lambda_smooth: &[f64],
        effective_df: &[f64],
    ) -> Result<Vec<f64>, String> {
        if lambda_smooth.len() != self.k_atoms() || effective_df.len() != self.k_atoms() {
            return Err(format!(
                "pooled_smoothing: lambda ({}) and edf ({}) must both be K={}",
                lambda_smooth.len(),
                effective_df.len(),
                self.k_atoms()
            ));
        }
        // Grouped exactly as `mackay_ard_precisions` groups the coordinate
        // prior, and for the reason stated there: a periodic atom's penalty
        // scale is set by a bounded period and a Euclidean atom's is not, so
        // one shared log-scale across both families is a mean of two
        // incomparable quantities.
        let mut pooled = lambda_smooth.to_vec();
        let mut usable: Vec<(usize, f64, f64, bool)> = Vec::new();
        for atom in 0..self.k_atoms() {
            // Eligibility is membership in the evidence, not the size of it.
            // An atom whose lambda has railed has its fit driven into the
            // penalty null space, so its edf goes to zero -- and dropping it
            // for having zero edf exempted the runaway atoms from the repair
            // aimed at them. Atoms nothing routes to keep a lambda nothing
            // reads.
            if self.atom_rows[atom].is_empty() {
                continue;
            }
            let df = effective_df[atom];
            if !df.is_finite() {
                continue;
            }
            // `trace - null_dim` is non-negative in exact arithmetic and can
            // land a hair below zero by rounding.
            let df = df.max(0.0);
            let periodic = self
                .atom_axis_periods(atom)
                .iter()
                .any(|period| period.is_some());
            usable.push((atom, lambda_smooth[atom].ln(), df, periodic));
        }
        for group_periodic in [false, true] {
            let group: Vec<&(usize, f64, f64, bool)> = usable
                .iter()
                .filter(|entry| entry.3 == group_periodic)
                .collect();
            if group.is_empty() {
                continue;
            }
            // Only atoms with a finite log-lambda and positive df can speak
            // to where the shared scale sits; a railed lambda has no finite
            // log to average and a zero-df atom carries no weight. Every
            // atom in the group still RECEIVES the pooled value.
            let contributing: Vec<&(usize, f64, f64, bool)> = group
                .iter()
                .copied()
                .filter(|entry| entry.1.is_finite() && entry.2 > 0.0)
                .collect();
            let weight: f64 = contributing.iter().map(|entry| entry.2).sum();
            if !(weight > 0.0) {
                continue;
            }
            let shared =
                contributing.iter().map(|entry| entry.1 * entry.2).sum::<f64>() / weight;
            let mean_df = weight / contributing.len() as f64;
            for &(atom, log_lambda, df, _) in group {
                // Unit-information shrinkage, the same rule the coordinate
                // prior obeys: one average atom's worth of prior evidence.
                // It lies in [0, 1) with no cap, and is exactly 0 at df = 0,
                // where the atom takes the shared scale outright.
                let own = df / (df + mean_df);
                pooled[atom] = if own > 0.0 && log_lambda.is_finite() {
                    (own * log_lambda + (1.0 - own) * shared).exp()
                } else {
                    // Avoids 0.0 * inf = NaN for a railed lambda, which is
                    // the case this fix exists to bring into the pool.
                    shared.exp()
                };
            }
        }
        Ok(pooled)
    }

    pub fn joint_hyperparameter_fixed_point(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        relative_tolerance: f64,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>, usize), String> {
        if !(relative_tolerance > 0.0) {
            return Err(format!(
                "joint_hyperparameter_fixed_point: relative tolerance must be positive; got {relative_tolerance}"
            ));
        }
        let mut lambda = lambda_smooth.to_vec();
        let mut ard = ard_precisions.to_vec();
        // The iteration count is bounded by the relative tolerance itself: a
        // contraction that has not moved by more than `tol` has converged, and
        // one that keeps moving is reported through the returned count rather
        // than hidden behind a cap that would look like convergence.
        // The iteration is linearly convergent (measured contraction ~0.92 per
        // sweep in the lambda direction, alpha reaching machine zero in eight),
        // so the budget follows the tolerance directly rather than its square
        // root, and Aitken extrapolation below jumps to the limit of the
        // linearly-converging part instead of walking there.
        let max_sweeps = (1.0 / relative_tolerance).ceil() as usize;
        let mut sweeps = 0usize;
        let mut history: (Option<Vec<f64>>, Option<Vec<f64>>) = (None, None);
        // The fit does not move inside this loop, so its residual does not
        // either: compute it once.
        let frozen_residual = self.raw_residual(target)?;
        for _ in 0..max_sweeps.max(2) {
            let mut next_lambda =
                self.fellner_schall_smoothing_with_residual(&lambda, &frozen_residual)?;
            let next_ard = self.mackay_ard_precisions(&ard)?;
            // Convergence is measured in effective degrees of freedom, which
            // is BOUNDED by the basis size -- not in log lambda, which is not.
            // An atom whose curvature is unsupported sends its lambda to
            // infinity lawfully, so its |d log lambda| never vanishes and a
            // max over log-moves can never be satisfied. Measured: alpha
            // reached 3e-5 by sweep 8 while max |d log lambda| sat at 0.175
            // and decayed by 8% a sweep, purely from railing atoms.
            let edf_before = self.effective_curvature_df(&lambda)?;
            let edf_after = self.effective_curvature_df(&next_lambda)?;
            let lambda_move = edf_before
                .iter()
                .zip(edf_after.iter())
                .map(|(before, after)| (after - before).abs())
                .fold(0.0_f64, f64::max);
            let ard_move = next_ard
                .iter()
                .zip(ard.iter())
                .flat_map(|(new_atom, old_atom)| new_atom.iter().zip(old_atom.iter()))
                .filter(|(new, old)| **new > 0.0 && **old > 0.0)
                .map(|(new, old)| (new.ln() - old.ln()).abs())
                .fold(0.0_f64, f64::max);
            // Aitken: with x_{n+1} - x* ~ r (x_n - x*), three iterates give the
            // limit directly. Applied per atom in log lambda, and only where
            // the three iterates are consistent with a contraction (r in
            // (0, 1)); a railing atom fails that test and is left alone.
            if let (Some(prev), Some(prev2)) = (history.0.as_ref(), history.1.as_ref()) {
                for atom in 0..next_lambda.len() {
                    let (x0, x1, x2) = (prev2[atom], prev[atom], next_lambda[atom]);
                    if !(x0 > 0.0 && x1 > 0.0 && x2 > 0.0) {
                        continue;
                    }
                    let (l0, l1, l2) = (x0.ln(), x1.ln(), x2.ln());
                    let d1 = l1 - l0;
                    let d2 = l2 - l1;
                    if d1.abs() <= f64::EPSILON {
                        continue;
                    }
                    let rate = d2 / d1;
                    if rate > 0.0 && rate < 1.0 {
                        let limit = l2 + d2 * rate / (1.0 - rate);
                        if limit.is_finite() {
                            next_lambda[atom] = limit.exp();
                        }
                    }
                }
            }
            history = (Some(next_lambda.clone()), history.0.take());
            lambda = next_lambda;
            ard = next_ard;
            sweeps += 1;
            log::info!(
                "joint sweep {sweeps}: lambda_move={lambda_move:.4e} ard_move={ard_move:.4e}                  lambda_med={:.4e} alpha_med={:.4e}",
                {
                    let mut v = lambda.clone();
                    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                    v[v.len() / 2]
                },
                {
                    let mut v: Vec<f64> =
                        ard.iter().flatten().copied().filter(|x| *x > 0.0).collect();
                    if v.is_empty() {
                        0.0
                    } else {
                        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                        v[v.len() / 2]
                    }
                }
            );
            if lambda_move.max(ard_move) <= relative_tolerance {
                break;
            }
        }
        Ok((lambda, ard, sweeps))
    }

    pub fn fellner_schall_smoothing(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
    ) -> Result<Vec<f64>, String> {
        let residual = self.raw_residual(target)?;
        self.fellner_schall_smoothing_with_residual(lambda_smooth, &residual)
    }

    /// [`Self::fellner_schall_smoothing`] against a residual the caller already
    /// holds. The residual is a function of the FIT, not of `lambda`, so a
    /// loop that holds the fit still (the joint hyperparameter solve) can
    /// compute it once instead of once per sweep -- fifty full
    /// reconstructions per round at 250k x 8096, all of them identical.
    pub fn fellner_schall_smoothing_with_residual(
        &self,
        lambda_smooth: &[f64],
        residual: &Array2<f64>,
    ) -> Result<Vec<f64>, String> {
        self.validate_smoothing(lambda_smooth)?;
        let sse: f64 = residual.iter().map(|value| value * value).sum();

        // Per-atom effective df, and the roughness the fit actually spent.
        // Same shape as the census: per atom, reading `&self`, writing only
        // its own four numbers. This is the expensive half of a REML round --
        // it accumulates every routed row's phi outer product before the
        // eigen-decomposition -- so it is the half worth spreading.
        let per_atom = (0..self.k_atoms())
            .into_par_iter()
            .map(|atom_idx| -> Result<(f64, f64, f64, f64), String> {
            let m = self.atoms[atom_idx].basis_size();
            let penalty = self.atoms[atom_idx].smooth_penalty().clone();

            // `G_k` is the same accumulation `decoder_sweep` performs.
            let mut gram = Array2::<f64>::zeros((m, m));
            let mut scratch = ActiveAtomScratch::default();
            for &(row, slot) in &self.atom_rows[atom_idx] {
                self.fill_active(row, slot, &mut scratch)?;
                let phi = scratch.phi_row();
                for left in 0..m {
                    for right in 0..m {
                        gram[[left, right]] += phi[left] * phi[right];
                    }
                }
            }

            let (trace, atom_null_dim) = penalized_trace_and_null_dim(
                &gram,
                &penalty,
                lambda_smooth[atom_idx],
                "fellner_schall_smoothing",
            )?;
            let decoder = self.atoms[atom_idx].decoder_coefficients();
            let penalized = penalty.dot(decoder);
            let atom_roughness = decoder
                .iter()
                .zip(penalized.iter())
                .map(|(left, right)| left * right)
                .sum::<f64>();
            // Rounding error in that quadratic form is of order
            // `eps * max|S| * |b|^2 * m`. A roughness beneath it is noise, and
            // dividing by it is how lambda reached 1.571e227.
            let penalty_scale = penalty
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            let decoder_sq = decoder.iter().map(|value| value * value).sum::<f64>();
            let atom_floor =
                f64::EPSILON * penalty_scale * decoder_sq * m.max(1) as f64;
            Ok((trace, atom_null_dim, atom_roughness, atom_floor))
            })
            .collect::<Result<Vec<(f64, f64, f64, f64)>, String>>()?;
        let mut tau = vec![0.0_f64; self.k_atoms()];
        let mut null_dim = vec![0.0_f64; self.k_atoms()];
        let mut roughness = vec![0.0_f64; self.k_atoms()];
        // The level at which `b'Sb` stops being distinguishable from zero,
        // per atom, from the magnitudes that produced it.
        let mut roughness_floor = vec![0.0_f64; self.k_atoms()];
        for (atom_idx, (t, nd, r, fl)) in per_atom.into_iter().enumerate() {
            tau[atom_idx] = t;
            null_dim[atom_idx] = nd;
            roughness[atom_idx] = r;
            roughness_floor[atom_idx] = fl;
        }

        // Profiled scale: residual sum of squares over the residual degrees of
        // freedom, which is `n*P` less the df the decoders spent (`P` channels
        // share each atom's `tau`).
        let spent: f64 = tau.iter().sum::<f64>() * self.output_dim as f64;
        let total = (self.n_obs() * self.output_dim) as f64;
        let residual_df = total - spent;
        if !(residual_df > 0.0) {
            return Err(format!(
                "fellner_schall_smoothing: decoders spend {spent} of {total} degrees of freedom, leaving none for scale"
            ));
        }
        let sigma_sq = sse / residual_df;

        let mut updated = lambda_smooth.to_vec();
        for atom_idx in 0..self.k_atoms() {
            let signal = tau[atom_idx] - null_dim[atom_idx];
            // No rows, no roughness, or no df beyond the null space: nothing in
            // the likelihood distinguishes one lambda from another here.
            if self.atom_rows[atom_idx].is_empty()
                || !(roughness[atom_idx] > roughness_floor[atom_idx])
                || !(signal > 0.0)
            {
                continue;
            }
            let candidate =
                sigma_sq * self.output_dim as f64 * signal / roughness[atom_idx];
            if candidate.is_finite() && candidate > 0.0 {
                updated[atom_idx] = candidate;
            }
        }
        Ok(updated)
    }

    /// Per-atom effective degrees of freedom `tau_k` beyond the penalty null
    /// space, the statistically meaningful "is this atom's bend supported?"
    /// census. Reported alongside usage so a dictionary can be judged by the
    /// curvature the evidence pays for rather than by atom count.
    pub fn effective_curvature_df(
        &self,
        lambda_smooth: &[f64],
    ) -> Result<Vec<f64>, String> {
        self.validate_smoothing(lambda_smooth)?;
        // Per atom, and independent per atom: each entry reads shared state
        // through `&self`, writes only its own slot, and builds its own
        // scratch. The census runs once per REML round over every atom, and
        // at K=11010 the smoothing alternation cost 1.35x the wall clock of a
        // fixed-lambda fit for a held-out difference of 0.0001 -- so the
        // arithmetic here is worth spreading even though the verdict it
        // produces is currently cheap to predict.
        let out = (0..self.k_atoms())
            .into_par_iter()
            .map(|atom_idx| -> Result<f64, String> {
            // An atom no row routes to has NO evidence: its supported
            // curvature df is zero, full stop. Falling through computed
            // `0 - null_dim` = -1 for every such atom -- an impossible edf
            // that then poisoned any consumer differencing the census: one
            // support-move flip produced |d edf| = 1.0 EXACTLY, which is the
            // value the REML alternation kept stopping on.
            if self.atom_rows[atom_idx].is_empty() {
                return Ok(0.0);
            }
            let m = self.atoms[atom_idx].basis_size();
            let penalty = self.atoms[atom_idx].smooth_penalty().clone();
            let mut gram = Array2::<f64>::zeros((m, m));
            let mut scratch = ActiveAtomScratch::default();
            for &(row, slot) in &self.atom_rows[atom_idx] {
                self.fill_active(row, slot, &mut scratch)?;
                let phi = scratch.phi_row();
                for left in 0..m {
                    for right in 0..m {
                        gram[[left, right]] += phi[left] * phi[right];
                    }
                }
            }
            let (trace, null_dim) = penalized_trace_and_null_dim(
                &gram,
                &penalty,
                lambda_smooth[atom_idx],
                "effective_curvature_df",
            )?;
            Ok(trace - null_dim)
            })
            .collect::<Result<Vec<f64>, String>>()?;
        Ok(out)
    }

    /// MacKay selection of the coordinate-prior precisions, per atom and axis.
    ///
    /// ```text
    ///   alpha_ka <- gamma_ka / sum_i sq_equiv(t_i),
    ///   gamma_ka  = sum_i clamp(1 - alpha_ka / H_ii, 0, 1)
    /// ```
    ///
    /// `sq_equiv` is the Euclidean-equivalent `t^2` the prior exposes precisely
    /// so this fixed point stays consistent with the von-Mises energy on a
    /// periodic axis, and `H_ii` is the coordinate curvature the inner solver
    /// assembles: the Gauss-Newton `||gamma'(t_i)||^2` plus the prior's PSD
    /// majorizer. Both come from `fill_active`, which already decodes the atom's
    /// tangent into the scratch jacobian.
    ///
    /// `gamma` is the WELL-DETERMINED count -- each slot contributes the
    /// fraction of its coordinate the likelihood (rather than the prior) has
    /// pinned down. This is what makes the fixed point self-limiting: the
    /// crude `n / (sum t^2 + sum 1/H)` form kept a constant numerator while
    /// growing alpha drove BOTH denominator terms to zero together, so
    /// alpha -> infinity was an attractor whenever the decoded tangent was
    /// weak (measured: median 1 -> 38 -> 78 over three rounds, and the update
    /// stayed disabled for it). With gamma in the numerator a growing alpha
    /// erases its own evidence: alpha/H_ii -> 1, gamma -> 0, and the iteration
    /// settles instead of railing.
    ///
    /// An axis with no occupied slots keeps its incoming precision: there is no
    /// evidence to select from. Like the smoothing update this is an OUTER-loop
    /// quantity -- moving alpha moves the objective, and `solve_fixed_point`
    /// certifies at fixed priors.
    pub fn mackay_ard_precisions(
        &self,
        ard_precisions: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        self.validate_ard(ard_precisions)?;
        let mut updated = ard_precisions.to_vec();
        let mut scratch = ActiveAtomScratch::default();
        // (atom, axis, periodic?, gamma, energy) for every axis with any
        // evidence; the pooled hyperprior below is estimated from this same
        // pass, WITHIN topology groups -- a periodic axis's coordinate energy
        // is bounded by its period while a Euclidean axis's is not, so one
        // pooled mean across both would shrink each toward the other's scale.
        let mut pooled: Vec<(usize, usize, bool, f64, f64)> = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            let dim = self.assignment.atom_coord_dim(atom_idx);
            if dim == 0 || self.atom_rows[atom_idx].is_empty() {
                continue;
            }
            let periods = self.atom_axis_periods(atom_idx).to_vec();
            let mut energy = vec![0.0_f64; dim];
            let mut gamma = vec![0.0_f64; dim];
            let mut count = vec![0.0_f64; dim];
            for &(row, slot) in &self.atom_rows[atom_idx] {
                self.fill_active(row, slot, &mut scratch)?;
                let coords = self.assignment.coords_for_slot(row, slot);
                for axis in 0..dim {
                    let alpha = ard_precisions[atom_idx][axis];
                    // alpha == 0.0 is the typed prior exemption (an axis whose
                    // prior family is constant on its manifold, e.g. any axis
                    // of an ambient unit vector); evidence selection on such
                    // an axis would be fitting noise, so it stays exempt.
                    if alpha == 0.0 {
                        continue;
                    }
                    let prior = ArdAxisPrior::eval(alpha, coords[axis], periods[axis]);
                    // Gauss-Newton coordinate curvature: the decoded tangent's
                    // squared norm plus the prior curvature the assembly installs.
                    let mut tangent_sq = 0.0_f64;
                    for channel in 0..self.output_dim {
                        let value = scratch.jacobian[[axis, channel]];
                        tangent_sq += value * value;
                    }
                    let curvature = tangent_sq + prior.psd_majorizer_hess();
                    if !(curvature > 0.0) {
                        continue;
                    }
                    // Posterior SECOND MOMENT, not the point estimate: the
                    // Gauss-Newton curvature is the axis's posterior
                    // precision, so E[t^2 | data] = t_hat^2 + 1/curvature.
                    // With the point estimate alone the alternation ratchets:
                    // each round's shrink lowers sum t_hat^2, which raises
                    // the next alpha, which shrinks harder (measured on the
                    // micro-bed: alpha median 6.6 -> 25.8 across two rounds
                    // while lambda collapsed 1.5 -> 0.36). The variance term
                    // floors the energy at count/curvature, so
                    // alpha <= curvature always -- the update cannot outrun
                    // the evidence that feeds it.
                    energy[axis] += prior.sq_equiv + 1.0 / curvature;
                    // The slot's well-determined fraction, clamped to [0, 1]:
                    // curvature carries the prior majorizer, so alpha/curvature
                    // can exceed 1 only through majorizer slack, never evidence.
                    gamma[axis] += (1.0 - (alpha / curvature).min(1.0)).max(0.0);
                    count[axis] += 1.0;
                }
            }
            for axis in 0..dim {
                if count[axis] > 0.0 {
                    pooled.push((
                        atom_idx,
                        axis,
                        periods[axis].is_some(),
                        gamma[axis],
                        energy[axis],
                    ));
                }
            }
        }
        // Unit-information empirical-Bayes pooling (Kass-Wasserman): every
        // axis is shrunk toward the dictionary's pooled precision with
        // exactly ONE average axis of prior evidence,
        //     alpha = (gamma + mean_gamma) / (energy + mean_energy).
        // A well-determined axis dominates its own estimate; a thin-evidence
        // axis inherits the pooled value instead of dividing two near-zeros.
        // This replaces the former determination floors, damping factors and
        // ceiling outright: at K=8096 (~250 rows/atom) those rails did not
        // prevent the escape, they became its resting place -- the population
        // median sat ON the 1e3 ceiling while lambda collapsed, train EV
        // 0.8522 against held-out 0.5907. Pooling removes the mechanism
        // (near-zero/near-zero division) rather than capping its output.
        for group_periodic in [false, true] {
            let group: Vec<&(usize, usize, bool, f64, f64)> = pooled
                .iter()
                .filter(|entry| entry.2 == group_periodic)
                .collect();
            if group.is_empty() {
                continue;
            }
            let axes = group.len() as f64;
            let mean_gamma = group.iter().map(|entry| entry.3).sum::<f64>() / axes;
            let mean_energy = group.iter().map(|entry| entry.4).sum::<f64>() / axes;
            for &(atom_idx, axis, _, gamma_axis, energy_axis) in group {
                let denominator = energy_axis + mean_energy;
                if denominator > 0.0 {
                    let candidate = (gamma_axis + mean_gamma) / denominator;
                    if candidate.is_finite() && candidate > 0.0 {
                        updated[atom_idx][axis] = candidate;
                    }
                }
            }
        }
        Ok(updated)
    }

    /// Accelerated parallel decoder update on the joint decoder quadratic.
    ///
    /// Given coordinates the decoder problem is a convex quadratic whose full
    /// Hessian is majorized by the block-diagonal `s*G_k + lambda_k*S_k`
    /// (each row couples at most `s = top_k` blocks, so the row-wise
    /// Cauchy-Schwarz bound `(sum of s terms)^2 <= s * sum of squares` gives
    /// the `s` factor). One proximal step against that majorizer descends
    /// monotonically with EVERY atom updated at once -- width `K`, no colour
    /// classes -- and FISTA momentum recovers the rate the damping costs.
    /// Plain Jacobi is this update with the majorizer replaced by `G_k`
    /// alone, which is exactly why it diverges on shared rows.
    ///
    /// The majorizer factorizations are per-call constants (`phi` depends
    /// only on the frozen coordinates), so each pass costs one residual
    /// gather and one triangular solve per atom, all row- and atom-parallel.
    /// `fitted` obeys the same contract as [`Self::decoder_sweep`]: exact at
    /// entry, exact at exit.
    fn decoder_sweep_fista(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        fitted: &mut Array2<f64>,
        passes: usize,
    ) -> Result<f64, String> {
        self.validate_smoothing(lambda_smooth)?;
        if fitted.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::decoder_sweep_fista: fitted {:?} != ({}, {})",
                fitted.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        let support = self
            .assignment
            .support_indices(0)
            .len()
            .max(1) as f64;
        // Per-atom basis rows over the atom's support, gathered once: phi is a
        // function of the frozen coordinates only.
        let k_atoms = self.k_atoms();
        let phi_rows: Vec<Array2<f64>> = (0..k_atoms)
            .into_par_iter()
            .map_init(ActiveAtomScratch::default, |scratch, atom_idx| {
                let m = self.atoms[atom_idx].basis_size();
                let rows = self.atom_rows[atom_idx].len();
                let mut phi = Array2::<f64>::zeros((rows, m));
                for (local, &(row, slot)) in self.atom_rows[atom_idx].iter().enumerate() {
                    self.fill_active(row, slot, scratch)?;
                    phi.row_mut(local).assign(&scratch.phi_row());
                }
                Ok::<_, String>(phi)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Inverse routing map: for each row, its slots as (atom, local index
        // into that atom's row list). Built once; this is what lets the apply
        // parallelize over ROWS while the deltas are computed over ATOMS.
        let mut row_slot_map: Vec<Vec<(usize, usize)>> =
            vec![Vec::with_capacity(support as usize); self.n_obs()];
        for atom_idx in 0..k_atoms {
            for (local, &(row, _slot)) in self.atom_rows[atom_idx].iter().enumerate() {
                row_slot_map[row].push((atom_idx, local));
            }
        }
        // Majorizer factorizations: s*G_k with the penalty applied through the
        // same solver the exact sweep trusts (it never forms G + lambda*S).
        let grams: Vec<Array2<f64>> = (0..k_atoms)
            .into_par_iter()
            .map(|atom_idx| {
                let phi = &phi_rows[atom_idx];
                phi.t().dot(phi) * support
            })
            .collect();
        let mut previous: Vec<Array2<f64>> = (0..k_atoms)
            .map(|atom_idx| self.atoms[atom_idx].decoder_coefficients().clone())
            .collect();
        let mut momentum_t = 1.0_f64;
        let mut max_change = 0.0_f64;
        // `passes` is the FLOOR, not the count (#2502): six majorized passes
        // were measured sufficient at K=808 and insufficient at K=8096
        // (EM+FISTA 0.6490 vs EM+colour 0.7345) -- a fixed count
        // under-converges large decoders and the evidence updates then read
        // corrupted curvature. Passing continues while each pass still
        // improves, by the same no-longer-decreasing stall rule the outer
        // criteria use.
        let mut pass_index = 0usize;
        let mut previous_pass_change = f64::INFINITY;
        loop {
            // grad_k = -Phi_k^T R|rows(k) + lambda_k S_k B_k ; step against the
            // majorizer via the penalized solver:
            //   (s G_k + lambda_k S_k) D_k = Phi_k^T R|rows(k) - lambda_k S_k B_k
            //   B_k <- B_k + D_k
            let residual = &target - &*fitted;
            let updates: Vec<(Array2<f64>, Array2<f64>)> = (0..k_atoms)
                .into_par_iter()
                .map(|atom_idx| -> Result<(Array2<f64>, Array2<f64>), String> {
                    let phi = &phi_rows[atom_idx];
                    let m = phi.ncols();
                    let mut rhs = Array2::<f64>::zeros((m, self.output_dim));
                    for (local, &(row, _slot)) in self.atom_rows[atom_idx].iter().enumerate() {
                        let phi_row = phi.row(local);
                        for basis in 0..m {
                            rhs.row_mut(basis)
                                .scaled_add(phi_row[basis], &residual.row(row));
                        }
                    }
                    let decoder = self.atoms[atom_idx].decoder_coefficients();
                    let penalized =
                        self.atoms[atom_idx].smooth_penalty().dot(decoder) * lambda_smooth[atom_idx];
                    rhs -= &penalized;
                    let delta = Self::solve_penalized_normal_equations(
                        &grams[atom_idx],
                        self.atoms[atom_idx].smooth_penalty(),
                        lambda_smooth[atom_idx],
                        &rhs,
                        "SaeSupportSparseTerm::decoder_sweep_fista",
                    )?;
                    let new = decoder + &delta;
                    Ok((new, delta))
                })
                .collect::<Result<Vec<_>, _>>()?;
            // FISTA extrapolation over the block iterates, then install and
            // refresh `fitted` with each row's own delta -- rows are disjoint
            // writes, so the refresh parallelizes over row chunks.
            let next_t = 0.5 * (1.0 + (1.0 + 4.0 * momentum_t * momentum_t).sqrt());
            let beta = (momentum_t - 1.0) / next_t;
            momentum_t = next_t;
            let mut installed: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
            let mut pass_change = 0.0_f64;
            for (atom_idx, (new, delta)) in updates.into_iter().enumerate() {
                for value in delta.iter() {
                    max_change = max_change.max(value.abs());
                    pass_change = pass_change.max(value.abs());
                }
                let extrapolated = &new + &((&new - &previous[atom_idx]) * beta);
                previous[atom_idx] = new;
                installed.push(extrapolated);
            }
            // fitted deltas per atom (atom-parallel), THEN a row-parallel
            // apply through the inverted (row, slot) -> (atom, local) map:
            // rows are disjoint writes, so this is the full-width apply the
            // colour classes could never give the exact sweep.
            let fitted_deltas: Vec<Array2<f64>> = (0..k_atoms)
                .into_par_iter()
                .map(|atom_idx| {
                    let step =
                        &installed[atom_idx] - self.atoms[atom_idx].decoder_coefficients();
                    phi_rows[atom_idx].dot(&step)
                })
                .collect();
            for (atom_idx, decoder) in installed.into_iter().enumerate() {
                self.atoms[atom_idx].set_decoder_coefficients(decoder)?;
            }
            fitted
                .axis_chunks_iter_mut(ndarray::Axis(0), RECONSTRUCT_ROW_CHUNK)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk, mut block)| {
                    let base = chunk * RECONSTRUCT_ROW_CHUNK;
                    for local_row in 0..block.nrows() {
                        let row = base + local_row;
                        let mut out = block.row_mut(local_row);
                        for &(atom_idx, atom_local) in &row_slot_map[row] {
                            out += &fitted_deltas[atom_idx].row(atom_local);
                        }
                    }
                });
            pass_index += 1;
            if pass_index >= passes
                && (!(pass_change > 0.0) || pass_change >= previous_pass_change)
            {
                break;
            }
            previous_pass_change = pass_change;
        }
        Ok(max_change)
    }

    /// `fitted` is the CALLER's decoded matrix and must be exact for the
    /// current state at entry; the sweep keeps it exact through every decoder
    /// update (it already maintained an internal copy incrementally — the
    /// per-cycle `reconstruct()` here existed only to seed it, and was the
    /// profiled majority of the whole fit).
    fn decoder_sweep(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        fitted: &mut Array2<f64>,
    ) -> Result<f64, String> {
        self.validate_smoothing(lambda_smooth)?;
        if fitted.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::decoder_sweep: fitted {:?} != ({}, {})",
                fitted.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        let mut max_change = 0.0_f64;
        let classes = self.decoder_conflict_colors();
        // Parallel width of this sweep is the SIZE of a colour class, not the atom
        // count: atoms in one class are row-disjoint and solved together, but the
        // classes run in sequence. With top_k = s every row forces its s atoms into
        // s distinct classes, so a dense conflict graph collapses the width.
        if !classes.is_empty() {
            let widest = classes.iter().map(|c| c.len()).max().unwrap_or(0);
            let narrowest = classes.iter().map(|c| c.len()).min().unwrap_or(0);
            let mean = self.k_atoms() as f64 / classes.len() as f64;
            log::info!(
                "decoder sweep colouring: {} classes over {} atoms (widest {}, narrowest {}, mean {:.1} atoms/class)",
                classes.len(), self.k_atoms(), widest, narrowest, mean
            );
        }
        for class in &classes {
            // Atoms in one class are row-disjoint: solve in parallel against
            // the shared `fitted` snapshot (each atom reads only its own rows),
            // then apply the disjoint updates.
            let fitted_snapshot: &Array2<f64> = fitted;
            let solved: Vec<(usize, Array2<f64>, Array2<f64>, f64)> = class
                .par_iter()
                .map(|&atom_idx| -> Result<_, String> {
                    let m = self.atoms[atom_idx].basis_size();
                    let old_decoder = self.atoms[atom_idx].decoder_coefficients();
                    // G ALONE. The penalty is applied inside
                    // `solve_penalized_normal_equations`, which never forms
                    // `G + lambda*S` -- assembling that sum is what made a
                    // legitimately large `lambda` unsolvable.
                    let mut gram = Array2::<f64>::zeros((m, m));
                    let mut rhs = Array2::<f64>::zeros((m, self.output_dim));
                    // #2575: the atom's basis rows and decoded images used to be
                    // two fresh `Array1`s PER ROW on the atom's support, plus
                    // six more inside the allocating evaluator, and a third per
                    // row for the delta. They are one `(rows, m)` and one
                    // `(rows, P)` block now — which also turns the decoded
                    // refresh below into a single GEMM instead of a GEMV per row.
                    let atom_rows = &self.atom_rows[atom_idx];
                    let row_count = atom_rows.len();
                    let mut phi_rows = Array2::<f64>::zeros((row_count, m));
                    let mut decoded_rows = Array2::<f64>::zeros((row_count, self.output_dim));
                    // ROW-PARALLEL reduction. `gram` and `rhs` are sums over this
                    // atom's OWN rows, so they parallelise as a reduction without
                    // changing the update: the shared `fitted` snapshot is read,
                    // never written, and the Gauss-Seidel order across atoms and
                    // colour classes is untouched. This is the parallelism the
                    // colouring cannot provide -- with top_k = 8 the conflict
                    // graph is dense and the sweep degenerates to a mean width of
                    // three atoms per class, so the width has to come from the
                    // rows instead.
                    let chunk = phi_rows
                        .axis_chunks_iter_mut(ndarray::Axis(0), DECODER_ROW_CHUNK)
                        .into_par_iter()
                        .zip(
                            decoded_rows
                                .axis_chunks_iter_mut(ndarray::Axis(0), DECODER_ROW_CHUNK)
                                .into_par_iter(),
                        )
                        .enumerate()
                        .map(
                            |(block, (mut phi_block, mut decoded_block))|
                             -> Result<(Array2<f64>, Array2<f64>), String> {
                                let base = block * DECODER_ROW_CHUNK;
                                let mut local_gram = Array2::<f64>::zeros((m, m));
                                let mut local_rhs =
                                    Array2::<f64>::zeros((m, self.output_dim));
                                let mut scratch = ActiveAtomScratch::default();
                                // `residual_without` does not depend on the basis
                                // index, but it was rebuilt inside the `left` loop:
                                // the same `output_dim`-vector recomputed m times per
                                // row (2x for a linear atom, 7x for a sphere). Form it
                                // once per row into a reused buffer, then accumulate
                                // each basis`s contribution as a scaled row add.
                                let mut residual_without =
                                    Array1::<f64>::zeros(self.output_dim);
                                for local in 0..phi_block.nrows() {
                                    let (row, slot) = atom_rows[base + local];
                                    self.fill_active(row, slot, &mut scratch)?;
                                    let phi = scratch.phi_row();
                                    for output in 0..self.output_dim {
                                        residual_without[output] = target[[row, output]]
                                            - fitted_snapshot[[row, output]]
                                            + scratch.decoded[output];
                                    }
                                    for left in 0..m {
                                        for right in 0..m {
                                            local_gram[[left, right]] += phi[left] * phi[right];
                                        }
                                        local_rhs
                                            .row_mut(left)
                                            .scaled_add(phi[left], &residual_without);
                                    }
                                    phi_block.row_mut(local).assign(&phi);
                                    decoded_block.row_mut(local).assign(&scratch.decoded);
                                }
                                Ok((local_gram, local_rhs))
                            },
                        )
                        .collect::<Result<Vec<_>, String>>()?;
                    for (local_gram, local_rhs) in chunk {
                        gram += &local_gram;
                        rhs += &local_rhs;
                    }
                    let decoder = Self::solve_penalized_normal_equations(
                        &gram,
                        self.atoms[atom_idx].smooth_penalty(),
                        lambda_smooth[atom_idx],
                        &rhs,
                        "SaeSupportSparseTerm::decoder_sweep",
                    )?;
                    let mut atom_change = 0.0_f64;
                    for (new, old) in decoder.iter().zip(old_decoder.iter()) {
                        atom_change = atom_change.max((new - old).abs());
                    }
                    let mut deltas = phi_rows.dot(&decoder);
                    deltas -= &decoded_rows;
                    Ok((atom_idx, decoder, deltas, atom_change))
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (atom_idx, decoder, deltas, atom_change) in solved {
                max_change = max_change.max(atom_change);
                self.atoms[atom_idx].set_decoder_coefficients(decoder)?;
                for (index, &(row, _slot)) in self.atom_rows[atom_idx].iter().enumerate() {
                    // Whole-row add rather than a scalar loop over outputs. The
                    // sweep applies rows*top_k row-updates of width `output_dim`
                    // -- 250k x 8 x 128 = 2.56e8 scalar adds per sweep at the
                    // sizes this issue runs -- and the scalar form gives the
                    // compiler nothing to vectorise across. Same arithmetic, same
                    // order within a row, so the result is unchanged.
                    let mut target = fitted.row_mut(row);
                    target += &deltas.row(index);
                }
            }
        }
        Ok(max_change)
    }

    /// One direct active-row Gauss-Newton coordinate sweep with manifold-aware
    /// backtracking. Exact row snapshots provide rollback; inverse retractions
    /// are never assumed.
    /// When `fitted` is given, rows whose coordinates moved are re-decoded
    /// into it after the sweep, so it leaves exact for the new state. The
    /// refresh is an exact recompute of exactly the changed rows — no
    /// incremental drift enters from the coordinate side — and in the
    /// converged tail, where the per-row KKT skip leaves most rows untouched,
    /// it costs a small fraction of the full-matrix decode it replaces.
    fn coordinate_sweep(
        &mut self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        trust_radius: f64,
        stationarity_tolerance: f64,
        fitted: Option<&mut Array2<f64>>,
    ) -> Result<f64, String> {
        self.validate_ard(ard_precisions)?;
        if !(trust_radius.is_finite() && trust_radius > 0.0) {
            return Err(format!(
                "SaeSupportSparseTerm::coordinate_sweep: trust_radius must be finite and positive; got {trust_radius}"
            ));
        }
        if !(stationarity_tolerance.is_finite() && stationarity_tolerance > 0.0) {
            return Err(format!(
                "SaeSupportSparseTerm::coordinate_sweep: stationarity tolerance must be finite and positive; got {stationarity_tolerance}"
            ));
        }
        // Rows are independent given the frozen decoder: each owns a disjoint
        // coordinate block. Take the storage so rows solve in parallel with
        // `self` shared-read, then put it back (also on a row error).
        let mut coords_rows = self.assignment.take_coords();
        // #2575: one scratch per rayon worker, not one per row. The row solve's
        // working set is ~18 allocations sized by the row's support shape, which
        // is identical for almost every row on this lane, so a worker allocates
        // once and reuses across every row it takes.
        let row_results: Vec<Result<f64, String>> = coords_rows
            .par_iter_mut()
            .enumerate()
            .map_init(RowSolveScratch::default, |scratch, (row, coords_row)| {
                self.row_coordinate_solve(
                    row,
                    coords_row,
                    scratch,
                    target,
                    ard_precisions,
                    trust_radius,
                    stationarity_tolerance,
                )
            })
            .collect();
        self.assignment.restore_coords(coords_rows)?;
        let mut max_change = 0.0_f64;
        let mut row_changes = Vec::with_capacity(row_results.len());
        for row_result in row_results {
            let change = row_result?;
            max_change = max_change.max(change);
            row_changes.push(change);
        }
        if let Some(fitted) = fitted {
            if fitted.dim() != (self.n_obs(), self.output_dim) {
                return Err(format!(
                    "SaeSupportSparseTerm::coordinate_sweep: fitted {:?} != ({}, {})",
                    fitted.dim(),
                    self.n_obs(),
                    self.output_dim
                ));
            }
            let output_dim = self.output_dim;
            fitted
                .axis_chunks_iter_mut(ndarray::Axis(0), RECONSTRUCT_ROW_CHUNK)
                .into_par_iter()
                .enumerate()
                .try_for_each(|(chunk, mut block)| -> Result<(), String> {
                    let mut scratch = ActiveAtomScratch::default();
                    let mut row_fitted = Array1::<f64>::zeros(output_dim);
                    let base = chunk * RECONSTRUCT_ROW_CHUNK;
                    for local in 0..block.nrows() {
                        // A row that took no step (skipped at its KKT
                        // threshold, or every trial was rejected) decodes to
                        // exactly what the buffer already holds.
                        if row_changes[base + local] == 0.0 {
                            continue;
                        }
                        self.reconstruct_row_into(base + local, &mut scratch, &mut row_fitted)?;
                        block.row_mut(local).assign(&row_fitted);
                    }
                    Ok(())
                })?;
        }
        Ok(max_change)
    }

    /// Select the accelerated parallel decoder update for this term's
    /// fixed-point solves: `Some(passes)` runs `Self::decoder_sweep_fista`
    /// with that many majorized passes per cycle, `None` (the default) keeps
    /// the exact colour-class sweep. A typed knob on the term rather than an
    /// environment variable, so an A/B is two constructed terms, not two
    /// process environments.
    /// Arm or disarm DoF-priced admission with the noise floor `sigma2` the
    /// charge is denominated in (bits convert at `sigma2 * ln 2` on the
    /// objective scale, twice that on the router's gain scale).
    pub fn set_admission_dof_pricing(&mut self, sigma2: Option<f64>) {
        self.admission_dof_sigma2 = sigma2;
    }

    /// See the `variable_priced_support` field: derived per-token L0 under
    /// priced admission. No effect unless pricing is armed.
    pub fn set_variable_priced_support(&mut self, enabled: bool) {
        self.variable_priced_support = enabled;
    }

    /// See `Self::exact_affine_ranking`.
    pub fn set_exact_affine_ranking(&mut self, enabled: bool) {
        self.exact_affine_ranking = enabled;
    }

    /// See `Self::grid_refinement`. A value of zero is treated as one.
    pub fn set_grid_refinement(&mut self, refinement: usize) {
        self.grid_refinement = refinement.max(1);
    }

    /// See `Self::admission_usage_amortized`. No effect unless pricing is
    /// armed.
    pub fn set_admission_usage_amortization(&mut self, enabled: bool) {
        self.admission_usage_amortized = enabled;
    }

    pub fn set_decoder_fista_passes(&mut self, passes: Option<usize>) {
        self.decoder_fista_passes = passes;
    }

    /// Per-slot offset ranges into a row's compact coordinate block.
    fn slot_offsets_into(&self, row: usize, out: &mut Vec<Range<usize>>) {
        out.clear();
        let mut cursor = 0usize;
        for &atom in self.assignment.support_indices(row) {
            let d = self.assignment.atom_coord_dim(atom as usize);
            out.push(cursor..cursor + d);
            cursor += d;
        }
    }

    /// Fill one active slot's basis row, jet, decoded image, and coordinate
    /// Jacobian into caller-owned buffers — the allocation-free counterpart of
    /// [`Self::evaluate_active`] for the parallel row solve. The profiled
    /// inner-cycle cost (98.6% of every core in `__memset`) was these buffers
    /// being freshly zero-allocated for every slot of every line-search trial
    /// of every row; the basis itself goes through the trait's
    /// [`SaeBasisEvaluator::evaluate_into`].
    fn fill_active_eval(
        &self,
        row: usize,
        slot: usize,
        slot_coords: &[f64],
        phi: &mut Array2<f64>,
        jet: &mut ndarray::Array3<f64>,
        decoded: &mut Array1<f64>,
        jacobian: &mut Array2<f64>,
    ) -> Result<(), String> {
        let atom_idx = self.assignment.support_indices(row)[slot] as usize;
        let atom = &self.atoms[atom_idx];
        let d = atom.latent_dim();
        let m = atom.basis_size();
        if slot_coords.len() != d
            || phi.dim() != (1, m)
            || jet.dim() != (1, m, d)
            || decoded.len() != self.output_dim
            || jacobian.dim() != (d, self.output_dim)
        {
            return Err(format!(
                "SaeSupportSparseTerm::fill_active_eval: atom {atom_idx} buffer shapes \
                 coords={}, phi={:?}, jet={:?}, decoded={}, jacobian={:?} do not match \
                 (m={m}, d={d}, p={})",
                slot_coords.len(),
                phi.dim(),
                jet.dim(),
                decoded.len(),
                jacobian.dim(),
                self.output_dim
            ));
        }
        let coords = ndarray::ArrayView2::from_shape((1, d), slot_coords)
            .map_err(|error| format!("SaeSupportSparseTerm::fill_active_eval: {error}"))?;
        let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
            format!("SaeSupportSparseTerm::fill_active_eval: atom {atom_idx} has no evaluator")
        })?;
        evaluator.evaluate_into(phi, jet, coords)?;
        // Hoist the decoder and accumulate BY ROW. This is the hottest
        // function in the fit -- 40% of profiled samples -- and it runs once per
        // (row, slot). Re-resolving `decoder_coefficients()` inside the inner
        // loop cost `m * P` accessor calls and a bounds-checked 2-D index per
        // element; `scaled_add` over a contiguous decoder row is the same
        // arithmetic as an axpy, with one bounds check per row.
        let decoder = atom.decoder_coefficients();
        decoded.fill(0.0);
        for basis in 0..m {
            decoded.scaled_add(phi[[0, basis]], &decoder.row(basis));
        }
        jacobian.fill(0.0);
        for axis in 0..d {
            let mut jacobian_axis = jacobian.row_mut(axis);
            for basis in 0..m {
                jacobian_axis.scaled_add(jet[[0, basis, axis]], &decoder.row(basis));
            }
        }
        Ok(())
    }

    /// One row's exact Gauss-Newton coordinate step with manifold-aware
    /// backtracking, on the row's caller-held coordinate block. Semantically
    /// the serial sweep's row iteration.
    ///
    /// Storage-wise: nothing here allocates. The scratch is the CALLER's, held
    /// per rayon worker and reused across every row that worker takes (#2575).
    /// It used to be per-row — eighteen allocations per row, `N` rows per
    /// sweep, hundreds of sweeps per fit — and the doc comment's claim that
    /// "the line-search halvings allocate nothing" was true within a row and
    /// misleading across them; the profiled 12.4% of self time in
    /// `malloc`/`free`/`memmove` is what that cost.
    fn row_coordinate_solve(
        &self,
        row: usize,
        coords_row: &mut Vec<f64>,
        scratch: &mut RowSolveScratch,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        trust_radius: f64,
        stationarity_tolerance: f64,
    ) -> Result<f64, String> {
        let mut max_change = 0.0_f64;
        let q = coords_row.len();
        let p = self.output_dim;
        scratch.fit(self, row, q, p);
        let RowSolveScratch {
            offsets,
            support,
            dims,
            current,
            trial,
            fitted,
            jacobian,
            trial_fitted,
            trial_residual,
            trial_delta,
            fitted_delta,
            old_coords,
        } = scratch;
        let n_slots = offsets.len();
        fitted.fill(0.0);
        jacobian.fill(0.0);

        for slot in 0..n_slots {
            let slot_scratch = &mut current[slot];
            self.fill_active_eval(
                row,
                slot,
                &coords_row[offsets[slot].clone()],
                &mut slot_scratch.phi,
                &mut slot_scratch.jet,
                &mut slot_scratch.decoded,
                &mut slot_scratch.jacobian,
            )?;
            *fitted += &slot_scratch.decoded;
            for axis in 0..dims[slot].1 {
                jacobian
                    .row_mut(offsets[slot].start + axis)
                    .assign(&slot_scratch.jacobian.row(axis));
            }
        }
        let residual = &target.row(row) - &*fitted;
        let mut row_objective_scale =
            1.0 + 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
        let mut rhs_vector = jacobian.dot(&residual);
        let mut gram = jacobian.dot(&jacobian.t());
        let mut prior_cursor = 0usize;
        for (slot, &atom) in support.iter().enumerate() {
            let atom = atom as usize;
            let periods = self.atom_axis_periods(atom);
            for axis in 0..self.assignment.atom_coord_dim(atom) {
                let prior = ArdAxisPrior::eval(
                    ard_precisions[atom][axis],
                    coords_row[offsets[slot].start + axis],
                    periods[axis],
                );
                row_objective_scale += prior.value.abs();
                rhs_vector[prior_cursor] -= prior.grad;
                gram[[prior_cursor, prior_cursor]] += prior.psd_majorizer_hess();
                prior_cursor += 1;
            }
        }
        let raw_gradient_max = rhs_vector
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        // A row already satisfying the caller's KKT request is a certified
        // fixed point of this coordinate block. The row gradient scales
        // with the row's own residual energy, so the skip threshold is
        // relative to the row objective (mirroring the solve-level
        // certificate); an absolute threshold left every near-converged
        // row re-solving its trust region on every cycle.
        if raw_gradient_max <= stationarity_tolerance * row_objective_scale {
            return Ok(0.0);
        }
        // SPEC-22: the exact PSD trust-region subproblem is general outer
        // optimizer machinery and lives in `opt`. gam kept a private copy
        // until #2574.
        let delta = opt::solve_psd_trust_region(gram.view(), rhs_vector.view(), trust_radius)
        .map_err(|error| format!("SaeSupportSparseTerm::coordinate_sweep: {error}"))?;
        // `retract_row_coords` moves the point with the manifold exponential map,
        // which travels only the TANGENT component of the step -- anything radial is
        // discarded. So a step certified in the full ambient chart space is not the
        // step taken. MEASURED on a failing row: one axis asked to move 7.12e-1 --
        // `delta_max`, the largest component of the whole step -- realized exactly
        // 0.0, while every other axis realized its request to rel ~1e-9. Backtracking
        // then rescales only the components that do move and never revives the one
        // that does not, so no step size can satisfy Armijo and the row aborts at the
        // resolution floor. That is why intrinsic dimension >= 2 has never fitted,
        // while every 1-D chart was fine: on a flat chart the tangent space is
        // everything, `project_to_tangent` is the identity, and this is inert.
        //
        // Project the STEP, and only the step. The gradient must NOT be projected:
        // measured, doing so zeros entries that are genuinely large (-6.08 at a
        // coordinate pinned at pi/2), which corrupts both the trust-region right-hand
        // side and the descent certificate computed from it.
        let mut delta = delta;
        self.assignment.project_row_tangent(
            row,
            coords_row,
            delta.as_slice_mut().expect("trust-region step is contiguous"),
        )?;
        let mut directional = rhs_vector.dot(&delta);
        if !(directional > 0.0) {
            // Projection and the Gram solve do not commute, so the projected step is
            // not guaranteed to remain an ascent direction for the right-hand side.
            // Steepest descent within the tangent space is one by construction, and
            // is a real step rather than a failed row.
            let mut fallback = rhs_vector.to_owned();
            self.assignment.project_row_tangent(
                row,
                coords_row,
                fallback.as_slice_mut().expect("fallback step is contiguous"),
            )?;
            let norm = fallback.dot(&fallback).sqrt();
            if !(norm > 0.0) {
                return Ok(0.0);
            }
            delta = fallback * (trust_radius / norm);
            directional = rhs_vector.dot(&delta);
        }

        let delta_max = delta
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if !directional.is_finite() || directional < 0.0 {
            return Err(format!(
                "SaeSupportSparseTerm::coordinate_sweep: trust-region step is not a finite descent direction (rhs_dot_delta={directional})"
            ));
        }
        // `rhsᵀ delta` is quadratic in the gradient near a stationary point.
        // Comparing it with an absolute machine epsilon therefore invents a
        // sqrt(EPSILON) gradient floor (~1.5e-8 for f64), preventing tighter
        // KKT tolerances from ever being reached. Exact zero is the only
        // no-direction case; any positive value remains a valid descent
        // certificate regardless of magnitude.
        if directional == 0.0 {
            return Ok(0.0);
        }
        old_coords.clear();
        old_coords.extend_from_slice(coords_row);
        let mut accepted = None;
        let mut best_gap = f64::INFINITY;
        let mut best_step = 0.0_f64;
        let mut best_objective_delta = f64::NAN;
        let mut best_armijo_bound = f64::NAN;
        let evaluation_ops = 1usize
            + p
            + q
            + dims.iter().map(|&(m, _)| m * p).sum::<usize>();
        let gamma =
            evaluation_ops as f64 * f64::EPSILON / (1.0 - evaluation_ops as f64 * f64::EPSILON);
        let objective_resolution = gamma * row_objective_scale;
        for halving in 0..=24 {
            self.assignment.project_row_coords(row, old_coords, coords_row)?;
            let step = 2.0_f64.powi(-(halving as i32));
            for (target_slot, value) in trial_delta.iter_mut().zip(delta.iter()) {
                *target_slot = step * value;
            }
            self.assignment.retract_row_coords(row, coords_row, trial_delta)?;
            // Evaluate f(trial) - f(old) directly. Near stationarity the
            // decrease is O(||g||^2), so subtracting two O(1) objective
            // values loses the Armijo signal at exactly sqrt(EPSILON).
            // For r = y-f and prediction change d, the data-loss increment
            // is -r'd + 1/2 d'd; the prior authority supplies equally stable
            // per-axis energy increments. Kahan accumulation preserves their
            // first-order cancellation in a wide output/coordinate block.
            let mut objective_delta = KahanSum::default();
            for accumulator in fitted_delta.iter_mut() {
                *accumulator = KahanSum::default();
            }
            for slot in 0..n_slots {
                let atom = support[slot] as usize;
                let slot_trial = &mut trial[slot];
                self.fill_active_eval(
                    row,
                    slot,
                    &coords_row[offsets[slot].clone()],
                    &mut slot_trial.phi,
                    &mut slot_trial.jet,
                    &mut slot_trial.decoded,
                    &mut slot_trial.jacobian,
                )?;
                for basis in 0..dims[slot].0 {
                    // Subtract basis values before multiplying by decoder
                    // coefficients. This cancels shared constant/intercept
                    // components before rounding, instead of subtracting two
                    // already-decoded O(1) predictions to recover an O(step)
                    // difference.
                    let phi_delta = trial[slot].phi[[0, basis]] - current[slot].phi[[0, basis]];
                    for output in 0..p {
                        fitted_delta[output].add(
                            phi_delta * self.atoms[atom].decoder_coefficients()[[basis, output]],
                        );
                    }
                }
            }
            for (output, delta_sum) in fitted_delta.iter().enumerate() {
                let fitted_delta = delta_sum.sum();
                objective_delta
                    .add(fitted_delta.mul_add(0.5 * fitted_delta - residual[output], 0.0));
            }
            let mut coord_cursor = 0usize;
            for (slot, &atom) in support.iter().enumerate() {
                let atom = atom as usize;
                let periods = self.atom_axis_periods(atom);
                for axis in 0..self.assignment.atom_coord_dim(atom) {
                    objective_delta.add(ArdAxisPrior::value_delta(
                        ard_precisions[atom][axis],
                        old_coords[coord_cursor],
                        coords_row[offsets[slot].start + axis],
                        periods[axis],
                    ));
                    coord_cursor += 1;
                }
            }
            let objective_delta = objective_delta.sum();
            let armijo_bound = -1.0e-4 * step * directional;
            let gap = objective_delta - armijo_bound;
            if gap.is_finite() && gap < best_gap {
                best_gap = gap;
                best_step = step;
                best_objective_delta = objective_delta;
                best_armijo_bound = armijo_bound;
            }
            trial_fitted.fill(0.0);
            for slot_trial in trial.iter() {
                *trial_fitted += &slot_trial.decoded;
            }
            trial_residual.assign(&target.row(row));
            *trial_residual -= &*trial_fitted;
            let mut trial_gradient_max = 0.0_f64;
            for (slot, &atom) in support.iter().enumerate() {
                let atom = atom as usize;
                let periods = self.atom_axis_periods(atom);
                for axis in 0..dims[slot].1 {
                    let likelihood_gradient =
                        -trial[slot].jacobian.row(axis).dot(&*trial_residual);
                    let gradient = likelihood_gradient
                        + ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            coords_row[offsets[slot].start + axis],
                            periods[axis],
                        )
                        .grad;
                    trial_gradient_max = trial_gradient_max.max(gradient.abs());
                }
            }
            let armijo_accept = objective_delta.is_finite() && objective_delta <= armijo_bound;
            let roundoff_tie_accept = objective_delta.is_finite()
                && objective_delta.abs() <= objective_resolution
                && trial_gradient_max < raw_gradient_max;
            if armijo_accept || roundoff_tie_accept {
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
                self.assignment.project_row_coords(row, old_coords, coords_row)?;
                // The floor rung's REQUIRED Armijo decrease. When even that is
                // below the objective's own round-off resolution, the search
                // has bottomed out demanding verification of a decrease it
                // cannot measure -- no trial at any smaller step could ever
                // certify. Measured on the #2502 REML lane (row 228530):
                // required 9.5e-13 against resolution 1.7e-11, raw KKT 31 --
                // a real gradient whose descent is unresolvable at f64. Taking
                // no step leaves the row's KKT high, so the outer certificate
                // honestly refuses to certify; erroring instead discarded a
                // whole fitted model over one unmeasurable row.
                let floor_required = 1.0e-4 * 2.0_f64.powi(-24) * directional;
                if floor_required <= objective_resolution {
                    log::debug!(
                        "coordinate row {row}: line search unmeasurable at its floor \
                         (required decrease {floor_required:.3e} <= objective resolution \
                         {objective_resolution:.3e}, raw KKT max={raw_gradient_max:.3e}); \
                         taking no step"
                    );
                    return Ok(max_change);
                }
                return Err(format!(
                    "SaeSupportSparseTerm::coordinate_sweep: row {row} has a raw descent direction but manifold line search found no decreasing step \
                     (raw KKT max={raw_gradient_max:.17e}, rhs_dot_delta={directional:.17e}, \
                     delta_max={delta_max:.17e}, best_step={best_step:.17e}, \
                     best_objective_delta={best_objective_delta:.17e}, \
                     best_armijo_bound={best_armijo_bound:.17e}, gap={best_gap:.17e}, \
                     objective_resolution={objective_resolution:.17e})"
                ));
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
    ) -> Result<SaeSupportStationarity, SaeSupportStationarityError> {
        let residual = self
            .raw_residual(target)
            .map_err(SaeSupportStationarityError::Evaluation)?;
        self.raw_stationarity_with_residual(&residual, lambda_smooth, ard_precisions)
    }

    /// [`Self::raw_stationarity`] against a caller-supplied residual, so one
    /// residual pass per fixed-point cycle serves both the certificate and the
    /// objective. Atoms and rows are independent reads of shared state; both
    /// reductions run in parallel.
    pub fn raw_stationarity_with_residual(
        &self,
        residual: &Array2<f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<SaeSupportStationarity, SaeSupportStationarityError> {
        self.validate_smoothing(lambda_smooth)
            .map_err(SaeSupportStationarityError::Evaluation)?;
        self.validate_ard(ard_precisions)
            .map_err(SaeSupportStationarityError::Evaluation)?;
        if residual.dim() != (self.n_obs(), self.output_dim) {
            return Err(SaeSupportStationarityError::Evaluation(format!(
                "SaeSupportSparseTerm::raw_stationarity_with_residual: residual {:?} != ({}, {})",
                residual.dim(),
                self.n_obs(),
                self.output_dim
            )));
        }
        let (decoder_sq, decoder_max, decoder_scaled_max) = (0..self.k_atoms())
            .into_par_iter()
            .map_init(ActiveAtomScratch::default, |scratch, atom_idx| -> Result<(f64, f64, f64), SaeSupportStationarityError> {
                let atom = &self.atoms[atom_idx];
                let mut gradient = atom.smooth_penalty().dot(atom.decoder_coefficients())
                    * lambda_smooth[atom_idx];
                // #2517 — the block's OWN curvature diagonal, accumulated in the
                // same pass at no extra cost: `G_bb = Σ_rows φ_b²` plus the
                // penalty's `λ·S_bb`. Dividing the gradient by it converts the
                // certificate from gradient space (extensive in rows-per-atom)
                // to parameter space, which is where the fixed point actually
                // has to recur.
                let mut curvature = vec![0.0_f64; atom.basis_size()];
                for basis in 0..atom.basis_size() {
                    curvature[basis] = lambda_smooth[atom_idx] * atom.smooth_penalty()[[basis, basis]];
                }
                for &(row, slot) in &self.atom_rows[atom_idx] {
                    self.fill_active(row, slot, scratch)
                        .map_err(SaeSupportStationarityError::Evaluation)?;
                    let phi = scratch.phi_row();
                    for basis in 0..atom.basis_size() {
                        curvature[basis] += phi[basis] * phi[basis];
                        for output in 0..self.output_dim {
                            gradient[[basis, output]] -= phi[basis] * residual[[row, output]];
                        }
                    }
                }
                let mut sq = 0.0_f64;
                let mut max = 0.0_f64;
                let mut scaled_max = 0.0_f64;
                for basis in 0..atom.basis_size() {
                    // A basis function that is identically zero on every row of
                    // this atom's support carries no curvature AND no gradient;
                    // its scaled step is zero, not a division by zero.
                    let scale = curvature[basis];
                    for output in 0..self.output_dim {
                        let value = gradient[[basis, output]];
                        sq += value * value;
                        max = max.max(value.abs());
                        accumulate_parameter_scaled_gradient(
                            &mut scaled_max,
                            value,
                            scale,
                            SaeInnerKktScaleBlock::SharedDecoder,
                            atom_idx * self.output_dim * atom.basis_size()
                                + basis * self.output_dim
                                + output,
                        )?;
                    }
                }
                Ok((sq, max, scaled_max))
            })
            .try_reduce(
                || (0.0, 0.0, 0.0),
                |a, b| Ok((a.0 + b.0, a.1.max(b.1), a.2.max(b.2))),
            )?;
        let (coordinate_sq, coordinate_max, coordinate_scaled_max) = (0..self.n_obs())
            .into_par_iter()
            .map_init(ActiveAtomScratch::default, |scratch, row| -> Result<(f64, f64, f64), SaeSupportStationarityError> {
                let mut sq = 0.0_f64;
                let mut max = 0.0_f64;
                let mut scaled_max = 0.0_f64;
                for slot in 0..self.assignment.support_indices(row).len() {
                    let atom = self.assignment.support_indices(row)[slot] as usize;
                    self.fill_active(row, slot, scratch)
                        .map_err(SaeSupportStationarityError::Evaluation)?;
                    let periods = self.atom_axis_periods(atom);
                    for axis in 0..scratch.jacobian.nrows() {
                        let mut gradient = 0.0;
                        // #2517 — the Gauss-Newton curvature of this coordinate,
                        // in the same pass: `Σ_out J²` plus the ARD prior's own
                        // curvature. Same discipline as the decoder block, so
                        // both are certified in parameter space.
                        let mut curvature = 0.0;
                        for output in 0..self.output_dim {
                            let jacobian = scratch.jacobian[[axis, output]];
                            gradient -= jacobian * residual[[row, output]];
                            curvature += jacobian * jacobian;
                        }
                        let prior = ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            self.assignment.coords_for_slot(row, slot)[axis],
                            periods[axis],
                        );
                        gradient += prior.grad;
                        curvature += prior.psd_majorizer_hess();
                        sq += gradient * gradient;
                        max = max.max(gradient.abs());
                        accumulate_parameter_scaled_gradient(
                            &mut scaled_max,
                            gradient,
                            curvature,
                            SaeInnerKktScaleBlock::CoordinateRow { row },
                            slot * scratch.jacobian.nrows() + axis,
                        )?;
                    }
                }
                Ok((sq, max, scaled_max))
            })
            .try_reduce(
                || (0.0, 0.0, 0.0),
                |a, b| Ok((a.0 + b.0, a.1.max(b.1), a.2.max(b.2))),
            )?;
        Ok(SaeSupportStationarity {
            decoder_l2: decoder_sq.sqrt(),
            decoder_max_abs: decoder_max,
            coordinate_l2: coordinate_sq.sqrt(),
            coordinate_max_abs: coordinate_max,
            decoder_scaled_max_abs: decoder_scaled_max,
            coordinate_scaled_max_abs: coordinate_scaled_max,
        })
    }

    /// Raw coordinate KKT residual with decoder coefficients held fixed.
    pub fn raw_coordinate_stationarity(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<(f64, f64), String> {
        let residual = self.raw_residual(target)?;
        self.raw_coordinate_stationarity_with_residual(&residual, ard_precisions)
    }

    /// [`Self::raw_coordinate_stationarity`] off a caller-supplied residual —
    /// the frozen-decoder certifier evaluates this every cycle, and the serial
    /// row loop plus its own full-matrix decode was the profiled bulk of the
    /// fallback certification stage. Row-parallel, same reduction as the
    /// coordinate half of [`Self::raw_stationarity_with_residual`].
    fn raw_coordinate_stationarity_with_residual(
        &self,
        residual: &Array2<f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<(f64, f64), String> {
        self.validate_ard(ard_precisions)?;
        if residual.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::raw_coordinate_stationarity_with_residual: residual {:?} != ({}, {})",
                residual.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        let (coordinate_sq, coordinate_max) = (0..self.n_obs())
            .into_par_iter()
            .map_init(ActiveAtomScratch::default, |scratch, row| -> Result<(f64, f64), String> {
                let mut sq = 0.0_f64;
                let mut max = 0.0_f64;
                for slot in 0..self.assignment.support_indices(row).len() {
                    let atom = self.assignment.support_indices(row)[slot] as usize;
                    self.fill_active(row, slot, scratch)?;
                    let periods = self.atom_axis_periods(atom);
                    for axis in 0..scratch.jacobian.nrows() {
                        let likelihood_gradient = scratch
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
                        sq += gradient * gradient;
                        max = max.max(gradient.abs());
                    }
                }
                Ok((sq, max))
            })
            .try_reduce(|| (0.0, 0.0), |a, b| Ok((a.0 + b.0, a.1.max(b.1))))?;
        Ok((coordinate_sq.sqrt(), coordinate_max))
    }

    fn frozen_decoder_coordinate_objective(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        let residual = self.raw_residual(target)?;
        self.frozen_decoder_coordinate_objective_with_residual(&residual, ard_precisions)
    }

    fn frozen_decoder_coordinate_objective_with_residual(
        &self,
        residual: &Array2<f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        let mut objective = 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
        for row in 0..self.n_obs() {
            for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                let atom = atom as usize;
                let periods = self.atom_axis_periods(atom);
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
        let mut last_objective: Option<f64> = None;
        // Decoders are frozen here, so the coordinate sweep's per-changed-row
        // refresh is the ONLY thing that moves the decode: the maintained
        // matrix stays exact (each changed row is recomputed from state, not
        // incremented), and no drift re-verification is needed to certify.
        let mut fitted_state = self.reconstruct()?;
        for iteration in 1..=max_iter {
            let max_change = self.coordinate_sweep(
                target,
                ard_precisions,
                trust_radius,
                tolerance,
                Some(&mut fitted_state),
            )?;
            let residual = &target - &fitted_state;
            let (coordinate_l2, coordinate_max_abs) =
                self.raw_coordinate_stationarity_with_residual(&residual, ard_precisions)?;
            // Same scale-invariant certificate as solve_fixed_point: the raw
            // coordinate KKT sums data gradients over the full output width,
            // so it is certified relative to max(1, |objective|).
            let objective = self
                .frozen_decoder_coordinate_objective_with_residual(&residual, ard_precisions)?;
            let kkt_scale = objective.abs().max(1.0);
            let objective_recurred = last_objective
                .map(|previous: f64| (objective - previous).abs() <= tolerance * kkt_scale)
                .unwrap_or(false);
            last_objective = Some(objective);
            let candidate =
                objective_recurred && coordinate_max_abs <= tolerance * kkt_scale;
            if candidate && previous_candidate {
                return Ok(SaeSupportCoordinateFixedPointReport {
                    iterations: iteration,
                    objective,
                    coordinate_l2,
                    coordinate_max_abs,
                    max_recurrence_change: max_change,
                    recurred: true,
                });
            }
            previous_candidate = candidate;
        }
        let (_, coordinate_max_abs) = self.raw_coordinate_stationarity(target, ard_precisions)?;
        let objective = self.frozen_decoder_coordinate_objective(target, ard_precisions)?;
        Err(format!(
            "SaeSupportSparseTerm::solve_coordinates_fixed_decoder did not recur within {max_iter} cycles (raw coordinate KKT max={coordinate_max_abs:.6e}, relative to objective {objective:.6e}: {:.6e})",
            coordinate_max_abs / objective.abs().max(1.0)
        ))
    }

    /// Alternate exact decoder blocks and direct active-row coordinate Newton
    /// steps until the raw KKT residual AND a full-cycle recurrence agree. A
    /// budget-exhausted iterate is an error; only converged fits are returned.
    /// One JOINT Gauss-Newton step over `(T, B)`, Schur-eliminating the per-row
    /// coordinate blocks -- the cure for the alternating map's linear rate
    /// (#2575), safeguarded on the same penalized objective the certificate
    /// uses (#2517).
    ///
    /// WHY THE ALTERNATION CANNOT GET THERE ON ITS OWN. Write the joint
    /// Gauss-Newton Hessian in the two blocks the two sweeps own,
    ///
    /// ```text
    ///     H = [ A   C  ]        A = blockdiag_i H_tt^(i)   (rows, given B)
    ///         [ C'  B  ]        B = blockdiag_k H_bb^(k)   (atoms, given T)
    /// ```
    ///
    /// Each sweep is the EXACT minimiser of its own block: `decoder_sweep`
    /// solves `(G_k + lambda S_k) B_k = rhs_k` outright, and near the fixed
    /// point `row_coordinate_solve` takes the unconstrained block-Newton step.
    /// A cycle is therefore exact block Gauss-Seidel, whose error propagates by
    /// `M = A^-1 C B^-1 C'`, i.e.
    ///
    /// ```text
    ///     rho(M) = 1 - lambda_min(A^-1 S),   S = A - C B^-1 C'
    /// ```
    ///
    /// the Schur complement of the block the alternation eliminates. So the
    /// measured rate is not a tuning artefact and no extrapolator can remove
    /// it: it IS the cross-block coupling, and it approaches 1 exactly as the
    /// joint problem approaches a direction that only a SIMULTANEOUS,
    /// compensating change in both blocks can travel. This term has such
    /// directions by construction -- reparameterising an atom's coordinates and
    /// counter-transforming its decoder leaves `f` unchanged, so the objective's
    /// curvature along that orbit comes only from the (weak) ARD and smoothing
    /// priors, which is `lambda_min(A^-1 S) << 1`. Anderson cannot rescue it
    /// either: there is one such near-flat orbit PER ATOM, so the slow subspace
    /// has dimension `O(K)` while a depth-`d` multisecant model spans `d`.
    ///
    /// Solving the joint system removes `M` from the iteration entirely -- the
    /// step is the Newton step of the coupled quadratic model, so the coupling
    /// is not damped or extrapolated but eliminated.
    ///
    /// The arithmetic is the standard bundle-adjustment reduction, and this
    /// term's [`Self::assemble_arrow_schur`] already builds exactly that system
    /// with `H_bb` and every `H_tb` installed as operators; nothing in the solve
    /// path had ever consumed it. `InexactPCG` is the mode this system admits:
    /// it is matrix-free by construction (the only resident row matrices are the
    /// `q_i x q_i` blocks), and the dense modes would have to materialise a
    /// `beta_dim x beta_dim` shared block that the overcomplete lane cannot hold.
    ///
    /// Returns `Some(objective)` when a step was accepted -- state and `fitted`
    /// are updated to that point -- and `None` when the model refused or no
    /// backtracked step decreased the objective, leaving the state EXACTLY as it
    /// was found. A refused joint step is never an error: the alternating cycle
    /// is monotone on its own, so the caller simply continues.
    fn install_scaled_arrow_displacement(
        &mut self,
        coordinate_snapshot: &[f64],
        decoder_snapshot: &[Array2<f64>],
        beta_offsets: &[usize],
        direction: &SaeArrowVector,
        scale: f64,
        scaled_step: &mut Vec<f64>,
        trial_coordinates: &mut Vec<f64>,
    ) -> Result<bool, String> {
        if direction.t.len() != coordinate_snapshot.len() {
            return Err(format!(
                "support saddle direction coordinate width {} != state width {}",
                direction.t.len(),
                coordinate_snapshot.len(),
            ));
        }
        self.install_coordinates(coordinate_snapshot)?;
        scaled_step.clear();
        scaled_step.extend(direction.t.iter().map(|value| scale * value));
        self.retract_coordinates(scaled_step)?;
        for (atom, base) in decoder_snapshot.iter().enumerate() {
            let mut decoder = base.clone();
            let offset = beta_offsets[atom];
            for basis in 0..decoder.nrows() {
                for output in 0..self.output_dim {
                    decoder[[basis, output]] += scale
                        * direction.beta[offset + basis * self.output_dim + output];
                }
            }
            self.atoms[atom].set_decoder_coefficients(decoder)?;
        }
        self.snapshot_coordinates(trial_coordinates);
        let coordinate_changed = coordinate_snapshot
            .iter()
            .zip(trial_coordinates.iter())
            .any(|(before, after)| before.to_bits() != after.to_bits());
        let decoder_changed = decoder_snapshot
            .iter()
            .zip(&self.atoms)
            .any(|(before, atom)| {
                before
                    .iter()
                    .zip(atom.decoder_coefficients().iter())
                    .any(|(left, right)| left.to_bits() != right.to_bits())
            });
        Ok(coordinate_changed || decoder_changed)
    }

    /// Leave a resolved stationary saddle along its exact most-negative
    /// generalized `(A, B)` mode.  Generalized eigenvectors have arbitrary
    /// orientation, so both signs are evaluated at every radius; choosing a
    /// sign from the eigensolver would make the basin depend on incidental
    /// LAPACK phase conventions.
    ///
    /// The starting radius is finite in both relevant geometries: at most one
    /// caller trust radius in B norm, at most that radius on any coordinate,
    /// and at most a trust-radius fraction of the current decoder scale.  A
    /// radius that does not lower the actual penalized objective is halved until
    /// either a sign clears two representable objective spacings or the step no
    /// longer changes a state bit.  The latter is an arithmetic proof that no
    /// smaller floating-point trial exists, so this search needs no fitted
    /// iteration cap.
    fn escape_support_negative_curvature(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        mode: &SupportNegativeCurvatureMode,
        objective: f64,
        trust_radius: f64,
        coordinate_snapshot: &mut Vec<f64>,
        scaled_step: &mut Vec<f64>,
    ) -> Result<Option<f64>, String> {
        if !(objective.is_finite() && trust_radius.is_finite() && trust_radius > 0.0) {
            return Err("support saddle escape requires finite objective and positive trust radius"
                .to_string());
        }
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        if mode.direction.beta.len() != beta_dim {
            return Err(format!(
                "support saddle direction decoder width {} != state width {beta_dim}",
                mode.direction.beta.len(),
            ));
        }
        self.snapshot_coordinates(coordinate_snapshot);
        let decoder_snapshot = self
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients().clone())
            .collect::<Vec<_>>();
        let coordinate_direction_scale = mode
            .direction
            .t
            .iter()
            .fold(0.0_f64, |current, &value| current.max(value.abs()));
        let decoder_direction_scale = mode
            .direction
            .beta
            .iter()
            .fold(0.0_f64, |current, &value| current.max(value.abs()));
        if !(coordinate_direction_scale.is_finite()
            && decoder_direction_scale.is_finite()
            && coordinate_direction_scale.max(decoder_direction_scale) > 0.0)
        {
            return Err("support saddle escape received a zero or non-finite mode".to_string());
        }
        let parameter_scale = self
            .parameter_iterate_scale()
            .map_err(SaeSupportStationarityError::ParameterScale)?;
        let mut radius = trust_radius;
        if coordinate_direction_scale > 0.0 {
            radius = radius.min(trust_radius / coordinate_direction_scale);
        }
        if decoder_direction_scale > 0.0 {
            radius = radius.min(trust_radius * parameter_scale / decoder_direction_scale);
        }
        if !(radius.is_finite() && radius > 0.0) {
            return Err("support saddle escape could not form a finite initial radius".to_string());
        }

        let mut trial_coordinates = Vec::with_capacity(coordinate_snapshot.len());
        loop {
            let mut any_changed = false;
            let mut best: Option<(f64, f64)> = None;
            for sign in [-1.0_f64, 1.0_f64] {
                let signed_radius = sign * radius;
                let changed = self.install_scaled_arrow_displacement(
                    coordinate_snapshot,
                    &decoder_snapshot,
                    &beta_offsets,
                    &mode.direction,
                    signed_radius,
                    scaled_step,
                    &mut trial_coordinates,
                )?;
                any_changed |= changed;
                if !changed {
                    continue;
                }
                let trial = self.penalized_objective(target, lambda_smooth, ard_precisions)?;
                if !trial.is_finite() {
                    continue;
                }
                let scale = objective.abs().max(trial.abs()).max(1.0);
                let spacing = f64::from_bits(scale.to_bits() + 1) - scale;
                if objective - trial > 2.0 * spacing
                    && best.as_ref().is_none_or(|(cost, _)| trial < *cost)
                {
                    best = Some((trial, signed_radius));
                }
            }
            if let Some((trial, signed_radius)) = best {
                self.install_scaled_arrow_displacement(
                    coordinate_snapshot,
                    &decoder_snapshot,
                    &beta_offsets,
                    &mode.direction,
                    signed_radius,
                    scaled_step,
                    &mut trial_coordinates,
                )?;
                log::info!(
                    "support exact saddle escape: generalized curvature {:.6e} (backward error \
                     {:.3e}), B-radius {:.3e}, objective {:.6e} -> {:.6e}",
                    mode.curvature,
                    mode.backward_error,
                    signed_radius.abs(),
                    objective,
                    trial,
                );
                return Ok(Some(trial));
            }
            if !any_changed {
                break;
            }
            let next = 0.5 * radius;
            if !(next > 0.0 && next < radius) {
                break;
            }
            radius = next;
        }
        self.install_coordinates(coordinate_snapshot)?;
        for (atom, decoder) in decoder_snapshot.into_iter().enumerate() {
            self.atoms[atom].set_decoder_coefficients(decoder)?;
        }
        Ok(None)
    }

    fn admitted_support_negative_curvature_mode(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        operator_directions_already_budgeted: usize,
    ) -> Result<Option<SupportNegativeCurvatureMode>, String> {
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        let coordinate_dim = self.coordinate_state_len();
        let full_dim = coordinate_dim
            .checked_add(beta_dim)
            .ok_or_else(|| "support saddle classifier dimension overflow".to_string())?;
        let dense_workspace = (full_dim as u128)
            .saturating_mul(full_dim as u128)
            .saturating_mul(std::mem::size_of::<f64>() as u128)
            .saturating_mul(6);
        let in_core_budget = crate::manifold::sae_host_in_core_budget_bytes().0 as u128;
        // A terminal second-order audit may not silently turn a matrix-free
        // fit into more operator work than its complete inner solve.  When the
        // analytic pencil has at most the number of directions already bought
        // by that solve and its eigensystem fits the cgroup ledger, dense is the
        // exact and cheaper certificate.  Larger lanes retain their matrix-free
        // outer-adjoint refusal rather than materialising an unbounded matrix.
        if full_dim == 0
            || full_dim > operator_directions_already_budgeted
            || dense_workspace > in_core_budget
        {
            return Ok(None);
        }
        let system = self.assemble_arrow_schur(target, lambda_smooth, ard_precisions)?;
        let rows = self.support_outer_differential_rows(target, ard_precisions, &beta_offsets)?;
        self.support_outer_negative_curvature_mode(&system, &rows)
    }

    fn joint_newton_step(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        fitted: &mut Array2<f64>,
        objective: f64,
        stationarity_tolerance: f64,
        coordinate_snapshot: &mut Vec<f64>,
        scaled_step: &mut Vec<f64>,
    ) -> Result<Option<f64>, String> {
        let mut system = self.assemble_arrow_schur(target, lambda_smooth, ard_precisions)?;
        if system.k == 0 || self.n_obs() == 0 {
            return Ok(None);
        }
        // Opt the per-row factorisation into spectral discovery. A row whose
        // coordinate block is flat along one axis (a periodic atom sitting at a
        // stationary phase, an atom the router left with a single row) has a
        // singular `H_tt^(i)`, and refusing to factor it would discard the whole
        // joint step over one row. Deflating that direction to unit stiffness is
        // what the dense manifold lane already does for the same reason.
        SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut system);
        // The coupled step is used to satisfy this caller's KKT certificate,
        // so its linear solve cannot stop at InexactPCG's generic 1e-4 LM
        // default when the requested certificate is tighter. Both PCG knobs
        // feed a `max`, hence both must carry the same requested accuracy.
        let mut options = ArrowSolveOptions::inexact_pcg();
        options.pcg.relative_tolerance = stationarity_tolerance;
        options.trust_region.steihaug_relative_tolerance = stationarity_tolerance;
        // Levenberg ladder seeded from the system's OWN curvature scale, so the
        // first trial is a true Newton step and any damping that follows is
        // measured in the units the block diagonal is already in -- never an
        // absolute number. `sqrt(EPSILON)` is the smallest relative shift that
        // survives the f64 assembly of that diagonal.
        let curvature_scale = system
            .hbb_diag
            .as_ref()
            .map(|diag| diag.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs())))
            .unwrap_or(0.0)
            .max(
                system
                    .rows
                    .iter()
                    .map(|row| {
                        (0..row.htt.nrows())
                            .map(|i| row.htt[[i, i]].abs())
                            .fold(0.0_f64, f64::max)
                    })
                    .fold(0.0_f64, f64::max),
            );
        let seed_ridge = f64::EPSILON.sqrt() * curvature_scale;
        let mut step_pair = None;
        let mut ridge = 0.0_f64;
        for attempt in 0..4 {
            match system.solve_with_options(ridge, ridge, &options) {
                Ok((delta_t, delta_beta, diagnostics)) => {
                    let admissible = diagnostics.stopping_reason
                        == gam_solve::arrow_schur::PcgStopReason::Converged
                        && diagnostics.final_relative_residual.is_finite()
                        && diagnostics.final_relative_residual <= stationarity_tolerance;
                    if admissible {
                        step_pair = Some((delta_t, delta_beta));
                        break;
                    }
                    log::debug!(
                        "support joint Newton linear solve refused at ridge {ridge:.3e} \
                         (attempt {attempt}): stop={:?}, relative residual {:.3e}, requested {:.3e}",
                        diagnostics.stopping_reason,
                        diagnostics.final_relative_residual,
                        stationarity_tolerance,
                    );
                }
                Err(error) => {
                    log::debug!(
                        "support joint Newton refused at ridge {ridge:.3e} (attempt {attempt}): {error}"
                    );
                }
            }
            if !(seed_ridge > 0.0) {
                break;
            }
            ridge = if ridge > 0.0 { ridge * 16.0 } else { seed_ridge };
        }
        let (delta_t, delta_beta) = match step_pair {
            Some(pair) => pair,
            None => return Ok(None),
        };
        if delta_t.len() != self.coordinate_state_len() {
            return Err(format!(
                "SaeSupportSparseTerm::joint_newton_step: arrow step width {} != compact \
                 coordinate width {}",
                delta_t.len(),
                self.coordinate_state_len()
            ));
        }
        if !delta_t.iter().chain(delta_beta.iter()).all(|v| v.is_finite()) {
            return Ok(None);
        }
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        if delta_beta.len() != beta_dim {
            return Err(format!(
                "SaeSupportSparseTerm::joint_newton_step: arrow border width {} != decoder \
                 width {beta_dim}",
                delta_beta.len()
            ));
        }
        self.snapshot_coordinates(coordinate_snapshot);
        let restore: Vec<Array2<f64>> = self
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients().clone())
            .collect();
        let output_dim = self.output_dim;
        // Backtracking on the SAME objective the certificate reads, from the
        // full Newton length. The floor is the coordinate sweep's own halving
        // budget, so neither block can be walked further than the other.
        for halving in 0..=24 {
            let scale = 2.0_f64.powi(-(halving as i32));
            self.install_coordinates(coordinate_snapshot)?;
            scaled_step.clear();
            scaled_step.extend(delta_t.iter().map(|value| scale * value));
            self.retract_coordinates(scaled_step)?;
            for (atom, base) in restore.iter().enumerate() {
                let mut decoder = base.clone();
                let offset = beta_offsets[atom];
                for basis in 0..decoder.nrows() {
                    for channel in 0..output_dim {
                        decoder[[basis, channel]] +=
                            scale * delta_beta[offset + basis * output_dim + channel];
                    }
                }
                self.atoms[atom].set_decoder_coefficients(decoder)?;
            }
            let trial = self.penalized_objective(target, lambda_smooth, ard_precisions)?;
            // #2634 -- a strict decrease SMALLER THAN THE OBJECTIVE'S OWN
            // ARITHMETIC RESOLUTION is not a measured descent, and installing
            // the displacement that bought it is what makes this loop unable to
            // certify. `penalized_objective` sums `n_obs * output_dim` residual
            // cells plus the per-atom penalty blocks, so its resolution is
            // `~sqrt(cells) * EPSILON * |f|`; below that, `trial < objective`
            // is comparing two roundings of the same number.
            //
            // Measured on `tiered_returns_best_effort_open_certificate_at_k_gg_rank_2275`
            // (base ea5c5e7d1 + instrumentation, release, acn112). The terminal
            // cycles accepted a coupled step on all 256 coupled cycles for a
            // decrease of `-5.204170e-17` on an objective of `2.202036583315e-2`
            // -- fifteen ulps -- and that step displaced the state by
            // `4.350617e-6`, 9.47x what BOTH sweeps moved and 185x the
            // parameter-space Newton step the KKT limb certifies
            // (`2.348286e-8 <= 2.975866e-8`). The recurrence conjunct then
            // refused the fit for motion the loop had manufactured itself: the
            // reconstruction moved `6.869948e-10` for that `4.350617e-6`, and
            // the objective had been bit-identical for 28 cycles.
            //
            // Both ratios are CONSTANT across the tail -- `max_change` is
            // `9.47 x` the sweeps and `185.2 x` the certified Newton step to
            // four significant figures over two and a half decades of decay --
            // which is what says the two quantities are one quantity in two
            // metrics rather than two independent claims.
            //
            // This TIGHTENS acceptance; it relaxes no bound and exempts nothing.
            // A refused step leaves the state exactly as found and doubles the
            // caller's skip counter, which is the path this function already
            // documents. With it, the same witness stops dead: `max_change`
            // goes `2.085759e-6 -> 6.296042e-8 -> 0.000000e0` and certifies at
            // cycle 333 of 512 with 73 accepted coupled steps instead of 256.
            let objective_cells = (self.n_obs() * output_dim).max(1) as f64;
            let objective_resolution =
                objective_cells.sqrt() * f64::EPSILON * objective.abs();
            if trial.is_finite() && objective - trial > objective_resolution {
                self.reconstruct_into(fitted)?;
                return Ok(Some(trial));
            }
        }
        self.install_coordinates(coordinate_snapshot)?;
        for (atom, decoder) in restore.into_iter().enumerate() {
            self.atoms[atom].set_decoder_coefficients(decoder)?;
        }
        Ok(None)
    }

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
        let mut last_max_change = f64::NAN;
        let mut last_objective: Option<f64> = None;
        // #2575: the alternating map's contraction is linear at ρ ≈ 0.975 on
        // real activations, so most cycles are spent crawling the tail rather
        // than resolving a nonlinearity. Anderson extrapolates over the
        // COORDINATE block alone, which is the whole state of the map: the
        // decoder sweep is the EXACT block minimiser of `B` given `T`, so the
        // fixed point is `T ↦ C(D(T))` and carrying `B` in the history would
        // store `Σ_k M_k·P` redundant numbers per column.
        let mut accelerator = AndersonAccelerator::new(SUPPORT_ANDERSON_DEPTH)
            .map_err(|error| format!("SaeSupportSparseTerm::solve_fixed_point: {error}"))?;
        let mut cycle_start = Vec::with_capacity(self.coordinate_state_len());
        let mut cycle_end = Vec::with_capacity(self.coordinate_state_len());
        let mut cycle_residual = Vec::with_capacity(self.coordinate_state_len());
        let mut cycle_start_decoders: Vec<Array2<f64>> = Vec::with_capacity(self.k_atoms());
        // `x_k − x_{k-1}` in the accelerator's difference-only contract; zero
        // before the first cycle, where it is ignored.
        let mut taken_step = vec![0.0_f64; self.coordinate_state_len()];
        let mut accepted_extrapolations = 0usize;
        // ONE decoded matrix for the whole solve. Seeded exactly once; the
        // decoder sweep keeps it exact through its updates and the coordinate
        // sweep re-decodes exactly the rows it moved, so the certificate
        // residual below is a subtraction, not a fresh `n x top_k` decode.
        // The decoder side accumulates increments, so any cycle that would
        // CERTIFY re-verifies on a from-scratch recompute before returning —
        // the certificate never rests on incrementally-maintained state.
        let mut fitted_state = self.reconstruct()?;
        let mut trial_fitted = Array2::<f64>::zeros(fitted_state.dim());
        // Support-move cadence. Measured on the #2502 lane: reroute proposals
        // ACCEPTED in early cycles carry the largest single-step objective
        // drops in the whole fit (1.85e6 -> 1.18e6 at cycle 3; 1.68e6 ->
        // 1.13e6 at cycle 56), yet the proposal only fired when the
        // certificate happened to -- an arm that never certifies at its
        // requested tolerance never re-routes at all. A plateau trigger
        // (>= 25 cycles since the last proposal AND < 0.5% relative
        // improvement since it) proposes the same guarded move on a schedule
        // the objective itself sets. The guard is unchanged -- accept only a
        // strict decrease -- so monotonicity survives by construction.
        let mut last_reroute_cycle = 0usize;
        let mut objective_at_last_reroute = f64::INFINITY;
        // #2575 joint-step bookkeeping. The joint step is attempted on every
        // cycle it is not skipping; a refusal doubles the skip so a lane where
        // the coupled model does not help pays a logarithmic number of extra
        // assemblies rather than one per cycle, and an acceptance clears it so
        // the terminal phase runs the coupled step every cycle. The doubling is
        // the schedule itself, not a tuned rate.
        let mut joint_snapshot = Vec::with_capacity(self.coordinate_state_len());
        let mut joint_scaled_step = Vec::with_capacity(self.coordinate_state_len());
        let mut joint_skip_remaining = 0usize;
        // PHASE. The caller's budget buys the ALTERNATION, and nothing about
        // that phase changes: no joint system is assembled, no coupled step is
        // proposed, and a fit that certifies inside `max_iter` takes exactly the
        // trajectory it has always taken. Only a fit that would otherwise return
        // the non-recurrence refusal enters phase two, where the coupled step is
        // armed and the same budget is spent again.
        //
        // That ordering is the whole safety argument. A coupled step is a strict
        // descent on the same objective, but it is still a DIFFERENT trajectory,
        // and a different trajectory can arrive at a different (better) optimum
        // that the caller's budget is no longer enough to certify at. Charging
        // that risk only to fits that were about to fail costs nothing they had,
        // and refusing is the outcome SPEC's "a fit object must only ever come
        // from a converged optimization" makes most expensive.
        let mut joint_armed = false;
        let mut joint_skip_width = 1usize;
        let mut joint_accepted = 0usize;
        let total_cycles = max_iter.saturating_mul(2);
        for iteration in 1..=total_cycles {
            if iteration > max_iter && !joint_armed {
                // Phase boundary. The alternation has spent the caller's budget
                // without certifying, so its rate has already answered the only
                // question the pace test asks. Arm the coupled step and start
                // the two-cycle recurrence over: the certificate must be earned
                // on the new trajectory, never inherited across the boundary.
                joint_armed = true;
                accelerator.reset();
                taken_step.clear();
                taken_step.resize(self.coordinate_state_len(), 0.0);
                last_objective = None;
                previous_candidate = false;
                log::info!(
                    "support fixed point: alternation did not recur in {max_iter} cycles; \
                     arming the coupled (Schur-eliminated joint Newton) phase"
                );
            }
            self.snapshot_coordinates(&mut cycle_start);
            if cycle_start_decoders.len() != self.k_atoms()
                || cycle_start_decoders
                    .iter()
                    .zip(&self.atoms)
                    .any(|(saved, atom)| saved.dim() != atom.decoder_coefficients().dim())
            {
                cycle_start_decoders = self
                    .atoms
                    .iter()
                    .map(|atom| atom.decoder_coefficients().clone())
                    .collect();
            } else {
                for (saved, atom) in cycle_start_decoders.iter_mut().zip(&self.atoms) {
                    saved.assign(atom.decoder_coefficients());
                }
            }
            // Expression statement rather than `let _ = …`: the sweep is run for
            // its effect on `fitted_state` and the returned change is unused, and
            // the ban scanner rejects both the bare and named underscore forms.
            match self.decoder_fista_passes {
                Some(passes) => {
                    self.decoder_sweep_fista(target, lambda_smooth, &mut fitted_state, passes)?
                }
                None => self.decoder_sweep(target, lambda_smooth, &mut fitted_state)?,
            };
            self.coordinate_sweep(
                target,
                ard_precisions,
                trust_radius,
                tolerance,
                Some(&mut fitted_state),
            )?;
            let phase_profiles = self.profile_periodic_phase_origins(ard_precisions)?;
            if phase_profiles > 0 {
                self.reconstruct_into(&mut fitted_state)?;
            }
            let mut residual = &target - &fitted_state;
            let mut stationarity =
                self.raw_stationarity_with_residual(&residual, lambda_smooth, ard_precisions)?;
            let previous_objective = last_objective;
            let mut objective =
                self.penalized_objective_with_residual(&residual, lambda_smooth, ard_precisions)?;
            let mut kkt_scale = objective.abs().max(1.0);
            let mut parameter_scale = self
                .parameter_iterate_scale()
                .map_err(SaeSupportStationarityError::ParameterScale)?;
            // The COUPLED step, in phase two only.
            //
            // Both sweeps minimise their own block exactly, so a cycle is exact
            // block Gauss-Seidel and the iterate's error contracts by a fixed
            // factor — the cross-block coupling (see `joint_newton_step`). Phase
            // one having spent the caller's whole budget without certifying IS
            // the measurement that this factor is too close to one for the
            // budget, so inside phase two there is nothing left to test: the
            // coupled step fires every cycle.
            //
            // (An earlier revision re-tested the alternation's pace per cycle
            // and skipped the coupled step while it looked on track. Measured on
            // the eight-arm sweep, that halved the cost and cost an order of
            // magnitude of accuracy on exactly the arms that need it — the
            // stalled `s = 3` arms ended at 1.0e-2 and 4.3e-2 certified against
            // 1.1e-3 and 6.8e-7 when the step fired every cycle. A test whose
            // answer is already known is not worth what skipping it costs.)
            //
            // Two conditions still hold it back. A point inside BOTH the KKT
            // and canonical-state recurrence limbs is never disturbed: one more
            // strict descent there would only keep the two-cycle certificate
            // from firing. KKT alone is insufficient because a weakly curved
            // block can keep moving the Laplace operator at negligible objective
            // change. And a REFUSED step doubles a skip counter, so a problem
            // whose coupled model does not help pays a logarithmic number of
            // assemblies rather than one per cycle.
            let pre_joint_change = self.canonical_state_recurrence_change(
                &cycle_start,
                &cycle_start_decoders,
                &mut cycle_end,
                &mut cycle_residual,
            )?;
            let certified = stationarity.kkt_certifies(
                kkt_scale,
                parameter_scale,
                tolerance,
            ) && pre_joint_change <= tolerance * parameter_scale;
            if joint_armed && joint_skip_remaining > 0 {
                joint_skip_remaining -= 1;
            } else if joint_armed && !certified {
                match self.joint_newton_step(
                    target,
                    lambda_smooth,
                    ard_precisions,
                    &mut fitted_state,
                    objective,
                    tolerance,
                    &mut joint_snapshot,
                    &mut joint_scaled_step,
                )? {
                    Some(_accepted_objective) => {
                        joint_accepted += 1;
                        joint_skip_width = 1;
                        // The coupled step can move along a periodic phase
                        // orbit after the cycle's first canonicalization. Put
                        // it back in the unique ARD-selected phase chart before
                        // either the recurrence certificate or the Laplace
                        // caller observes the state.
                        if self.profile_periodic_phase_origins(ard_precisions)? > 0 {
                            self.reconstruct_into(&mut fitted_state)?;
                        }
                        residual = &target - &fitted_state;
                        stationarity = self.raw_stationarity_with_residual(
                            &residual,
                            lambda_smooth,
                            ard_precisions,
                        )?;
                        objective = self.penalized_objective_with_residual(
                            &residual,
                            lambda_smooth,
                            ard_precisions,
                        )?;
                        kkt_scale = objective.abs().max(1.0);
                        parameter_scale = self
                            .parameter_iterate_scale()
                            .map_err(SaeSupportStationarityError::ParameterScale)?;
                    }
                    None => {
                        joint_skip_remaining = joint_skip_width;
                        joint_skip_width = joint_skip_width.saturating_mul(2);
                    }
                }
            }
            // Recurrence is measured on the fully canonicalized state, not on
            // the transient block-sweep path that reached it. The phase
            // profiler removes the exact periodic gauge first; any remaining
            // coordinate or decoder motion changes the Arrow/Laplace operator
            // and therefore must be below the same parameter-space tolerance
            // before this inner state can define an outer objective value.
            let max_change = self.canonical_state_recurrence_change(
                &cycle_start,
                &cycle_start_decoders,
                &mut cycle_end,
                &mut cycle_residual,
            )?;
            last_max_change = max_change;
            // The raw KKT is EXTENSIVE: each decoder entry sums per-row data
            // gradients over every row on the atom's support, so its natural
            // scale grows with rows-per-atom x residual scale. Certify the
            // scale-invariant first-order condition |g|_inf <= tol * max(1, |f|)
            // instead of an absolute bound an irreducible-residual problem can
            // never meet at any cycle budget.
            // All certificate limbs are relative and gauge-invariant. The
            // periodic phase orbit has already been profiled to its exact
            // ARD-selected representative, so recurrence of the canonical
            // coordinates/decoder is now meaningful and load-bearing: the
            // penalized objective can be unchanged while the Arrow Hessian and
            // its reduced-Schur log determinant still move.
            let mut objective_recurred = last_objective
                .map(|previous: f64| (objective - previous).abs() <= tolerance * kkt_scale)
                .unwrap_or(false);
            // #2517 — the KKT limb is certified in PARAMETER space, not in
            // gradient space. The decoder sweep solves `(G + λS)B = rhs`
            // exactly, so near the fixed point the block gradient is
            // `(G + λS)·Δ` and `G = Σ_rows φφᵀ` is extensive in rows-per-atom:
            // measured, the raw gradient is 12x-75x the remaining parameter
            // error across two decades of shape, so an absolute (or
            // objective-relative) bound on it is a bound on `m·Δ` that tightens
            // as data is ADDED. Dividing each block's gradient by its own
            // curvature diagonal removes exactly that factor and leaves the
            // Newton step, which is what a fixed point has to make small and is
            // invariant to n, to rows-per-atom, and to basis scaling.
            let mut candidate = objective_recurred
                && stationarity.kkt_certifies(kkt_scale, parameter_scale, tolerance)
                && max_change <= tolerance * parameter_scale;
            if candidate && previous_candidate {
                // About to certify: recompute the decode from scratch and
                // re-evaluate both limbs on it. If the maintained state had
                // drifted past the tolerance, this demotes the cycle to a
                // non-candidate instead of certifying a stale number.
                self.reconstruct_into(&mut fitted_state)?;
                residual = &target - &fitted_state;
                stationarity =
                    self.raw_stationarity_with_residual(&residual, lambda_smooth, ard_precisions)?;
                objective = self
                    .penalized_objective_with_residual(&residual, lambda_smooth, ard_precisions)?;
                kkt_scale = objective.abs().max(1.0);
                parameter_scale = self
                    .parameter_iterate_scale()
                    .map_err(SaeSupportStationarityError::ParameterScale)?;
                objective_recurred = previous_objective
                    .map(|previous: f64| (objective - previous).abs() <= tolerance * kkt_scale)
                    .unwrap_or(false);
                candidate = objective_recurred
                    && stationarity.kkt_certifies(kkt_scale, parameter_scale, tolerance)
                    && max_change <= tolerance * parameter_scale;
            }
            last_objective = Some(objective);
            if candidate && previous_candidate {
                // The alternating sweeps hold the SUPPORT fixed, so a point that
                // is stationary in the coordinates and the decoders can still be
                // improved by re-routing rows onto atoms that now explain them
                // better -- a TopK SAE re-selects its latents on every forward
                // pass, and this loop never did. Proposing the move HERE, at the
                // inner fixed point, is the dictionary-learning alternation the
                // scheme was missing, and it needs no cadence constant because
                // convergence is itself the trigger.
                //
                // The move is guarded on the certificate's own objective. A
                // re-route changes the objective discontinuously, so accepting
                // it unconditionally would destroy the monotonicity the
                // certificate rests on; accepting only a strict decrease keeps
                // the scheme monotone and makes the returned point locally
                // optimal against a support move as well as stationary within
                // one, which is strictly stronger than certifying a frozen
                // support.
                //
                // SCOPE: "locally optimal against a support move" means against
                // the proposal THIS router generates at its own fixed point --
                // residual-greedy selection at basis resolution, then polished.
                // It is not optimality over the space of supports, which is
                // combinatorial and is not claimed here.
                let support_k = match self.assignment.mode() {
                    AssignmentMode::TopK { k } => k,
                    _ => 0,
                };
                if support_k > 0 {
                    let mut moved =
                        self.reroute_fixed_decoder_ard(target, support_k, 0, ard_precisions)?;
                    // The re-routed term is freshly constructed, which resets
                    // typed solver knobs to their defaults -- carrying the
                    // decoder strategy across is what keeps an accepted move
                    // from silently reverting the solve to the colour sweep.
                    moved.set_decoder_fista_passes(self.decoder_fista_passes);
                    // The proposal arrives on the routing grid -- one of
                    // `basis_size` samples per atom -- while the incumbent sits
                    // at a converged continuous fixed point. Comparing them
                    // directly charges the proposal a quantization tax on every
                    // one of `n * support_k` slots and rejects good support
                    // moves for a reason that has nothing to do with the
                    // support. Solving the proposal's coordinates at frozen
                    // decoders is a strict decrease of its own objective, so
                    // this test accepts everything the unpolished one accepted
                    // and additionally the moves quantization was vetoing.
                    // A proposal that cannot be polished is a REJECTED proposal,
                    // never a dead fit.
                    //
                    // This branch runs INSIDE `candidate && previous_candidate`:
                    // the certificate has already been met, and the code
                    // immediately below it returns that certificate. So a bare
                    // `?` on the polish threw away a stationary,
                    // certificate-meeting incumbent because a speculative move
                    // nobody asked for would not converge — and it reported the
                    // PROPOSAL's non-convergence as the whole fit's error, from a
                    // function whose caller has no way to know a proposal was
                    // ever made.
                    //
                    // The plateau branch below already refuses to make that
                    // trade, for this reason, in these words. This one did not.
                    // Measured on `examples/issue_2575_joint_rate`: the
                    // `n=120 P=8 K=12 s=2` arm reaches certified 5.793e-7 against
                    // a 1e-6 tolerance and is nonetheless reported as stalled,
                    // carrying `solve_coordinates_fixed_decoder did not recur
                    // within 256 cycles` as its refusal (#2575).
                    if self.has_same_support_as(&moved) {
                        log::info!(
                            "support move at cycle {iteration} retained every discrete support; \
                             treating it as the no-op it is"
                        );
                    } else {
                        match moved.solve_coordinates_fixed_decoder(
                            target,
                            ard_precisions,
                            max_iter,
                            tolerance,
                            trust_radius,
                        ) {
                            Err(error) => {
                                log::info!(
                                    "support move unpolishable at cycle {iteration}, \
                                     rejected: {error}"
                                );
                            }
                            Ok(_) => {
                                let after = moved
                                    .penalized_objective(target, lambda_smooth, ard_precisions)?;
                                if after < objective {
                                    log::info!(
                                        "support move accepted at cycle {iteration}: objective \
                                         {objective:.6e} -> {after:.6e}"
                                    );
                                    *self = moved;
                                    self.reconstruct_into(&mut fitted_state)?;
                                    last_reroute_cycle = iteration;
                                    objective_at_last_reroute = after;
                                    // The map itself changed, so every difference the
                                    // accelerator holds describes a map that no longer
                                    // exists, and the two-cycle recurrence has to be
                                    // re-established against the new support.
                                    accelerator.reset();
                                    taken_step.clear();
                                    taken_step.resize(self.coordinate_state_len(), 0.0);
                                    last_objective = None;
                                    previous_candidate = false;
                                    continue;
                                }
                                log::info!(
                                    "support move rejected at cycle {iteration}: objective \
                                     {objective:.6e} -> {after:.6e}"
                                );
                            }
                        }
                    }
                }
                if let Some(mode) = self.admitted_support_negative_curvature_mode(
                    target,
                    lambda_smooth,
                    ard_precisions,
                    total_cycles,
                )? {
                    match self.escape_support_negative_curvature(
                        target,
                        lambda_smooth,
                        ard_precisions,
                        &mode,
                        objective,
                        trust_radius,
                        &mut joint_snapshot,
                        &mut joint_scaled_step,
                    )? {
                        Some(escaped_objective) => {
                            self.reconstruct_into(&mut fitted_state)?;
                            accelerator.reset();
                            taken_step.clear();
                            taken_step.resize(self.coordinate_state_len(), 0.0);
                            last_objective = None;
                            previous_candidate = false;
                            joint_skip_remaining = 0;
                            joint_skip_width = 1;
                            objective_at_last_reroute = escaped_objective;
                            continue;
                        }
                        None => {
                            return Err(format!(
                                "support fixed point reached a resolved stationary saddle \
                                 (generalized curvature {:.6e}, backward error {:.3e}) but \
                                 neither sign of its exact mode produced a representable \
                                 objective decrease",
                                mode.curvature, mode.backward_error,
                            ));
                        }
                    }
                }
                log::info!(
                    "support fixed-point cycle {iteration}: raw KKT max={:.3e} rel={:.3e} \
                     parameter KKT max={:.3e} rel={:.3e} max_change={:.3e} objective={:.6e} \
                     anderson_accepted={accepted_extrapolations} joint_accepted={joint_accepted}",
                    stationarity.max_abs(),
                    stationarity.max_abs() / kkt_scale,
                    stationarity.scaled_max_abs(),
                    stationarity.scaled_max_abs() / parameter_scale,
                    max_change,
                    objective
                );
                return Ok(SaeSupportFixedPointReport {
                    iterations: iteration,
                    objective,
                    stationarity,
                    max_recurrence_change: max_change,
                    recurred: true,
                });
            }
            previous_candidate = candidate;

            if iteration == 1 {
                // The baseline the first plateau test compares against; an
                // infinite sentinel here would make the trigger unsatisfiable.
                objective_at_last_reroute = objective;
            }
            let plateau = iteration >= last_reroute_cycle + 25
                && objective > objective_at_last_reroute * (1.0 - 5.0e-3);
            if plateau {
                let support_k = match self.assignment.mode() {
                    AssignmentMode::TopK { k } => k,
                    _ => 0,
                };
                if support_k > 0 {
                    last_reroute_cycle = iteration;
                    objective_at_last_reroute = objective;
                    // A proposal that cannot be polished is a REJECTED proposal,
                    // never a dead fit: the incumbent is untouched, so erroring
                    // out here would discard a healthy model over a speculative
                    // move (the exact discard shape this lane keeps re-finding).
                    let mut moved =
                        self.reroute_fixed_decoder_ard(target, support_k, 0, ard_precisions)?;
                    moved.set_decoder_fista_passes(self.decoder_fista_passes);
                    if self.has_same_support_as(&moved) {
                        log::info!(
                            "plateau support move at cycle {iteration} retained every discrete \
                             support; treating it as the no-op it is"
                        );
                    } else {
                        let polished = moved.solve_coordinates_fixed_decoder(
                            target,
                            ard_precisions,
                            max_iter,
                            tolerance,
                            trust_radius,
                        );
                        match polished {
                            Err(error) => {
                                // fall through to the normal cycle tail: the
                                // accelerator bookkeeping must see every cycle.
                                log::info!(
                                    "plateau support move unpolishable at cycle {iteration}: {error}"
                                );
                            }
                            Ok(_) => {
                                let after = moved.penalized_objective(
                                    target,
                                    lambda_smooth,
                                    ard_precisions,
                                )?;
                                if after < objective {
                                    log::info!(
                                        "plateau support move accepted at cycle {iteration}: \
                                         objective {objective:.6e} -> {after:.6e}"
                                    );
                                    *self = moved;
                                    self.reconstruct_into(&mut fitted_state)?;
                                    accelerator.reset();
                                    taken_step.clear();
                                    taken_step.resize(self.coordinate_state_len(), 0.0);
                                    last_objective = None;
                                    previous_candidate = false;
                                    continue;
                                }
                                log::info!(
                                    "plateau support move rejected at cycle {iteration}: \
                                     objective {objective:.6e} -> {after:.6e}"
                                );
                            }
                        }
                    }
                }
            }

            // The certified point is ALWAYS a plain post-sweep iterate: the
            // certificate above has already been evaluated and either returned
            // or not, and what follows only chooses where the NEXT cycle starts.
            //
            // Anderson has no descent guarantee, so the proposal is safeguarded
            // on the objective the certificate itself uses, at the SAME decoder
            // this cycle solved — a like-for-like comparison, and a conservative
            // one, because the next cycle's exact decoder solve can only lower
            // it further. On rejection the plain iterate is restored and the
            // history is dropped: differences taken across a rejected candidate
            // would fit a secant model to a trajectory that never happened.
            let proposal = accelerator
                .propose(&cycle_residual, &taken_step)
                .map_err(|error| format!("SaeSupportSparseTerm::solve_fixed_point: {error}"))?;
            // The step that reached the NEXT cycle's iterate, whichever arm is
            // taken. The accelerator only ever sees differences, so this is the
            // one piece of state the caller owes it.
            taken_step.clear();
            match proposal {
                None => taken_step.extend_from_slice(&cycle_residual),
                Some(proposal) => {
                    // The extrapolated step is applied through the SAME
                    // retraction the line search uses, from the cycle's own
                    // starting iterate — so `x_start + step` is on the manifold
                    // by construction, and the step the accelerator is told
                    // about is exactly the one that was taken.
                    self.install_coordinates(&cycle_start)?;
                    self.retract_coordinates(&proposal)?;
                    self.reconstruct_into(&mut trial_fitted)?;
                    let trial_residual = &target - &trial_fitted;
                    let extrapolated = self.penalized_objective_with_residual(
                        &trial_residual,
                        lambda_smooth,
                        ard_precisions,
                    )?;
                    if extrapolated < objective {
                        accepted_extrapolations += 1;
                        taken_step.extend_from_slice(&proposal);
                        std::mem::swap(&mut fitted_state, &mut trial_fitted);
                    } else {
                        // Restored to the iterate `fitted_state` already
                        // describes; the maintained state stays valid.
                        self.install_coordinates(&cycle_end)?;
                        accelerator.reset();
                        taken_step.extend_from_slice(&cycle_residual);
                    }
                }
            }
            log::info!(
                "support fixed-point cycle {iteration}: raw KKT max={:.3e} rel={:.3e} \
                 parameter KKT max={:.3e} rel={:.3e} max_change={:.3e} objective={:.6e} \
                 anderson={}/{} order={} joint={}",
                stationarity.max_abs(),
                stationarity.max_abs() / kkt_scale,
                stationarity.scaled_max_abs(),
                stationarity.scaled_max_abs() / parameter_scale,
                max_change,
                objective,
                accepted_extrapolations,
                iteration,
                accelerator.history_len(),
                joint_accepted
            );
        }
        let stationarity = self.raw_stationarity(target, lambda_smooth, ard_precisions)?;
        let objective = self.penalized_objective(target, lambda_smooth, ard_precisions)?;
        let parameter_scale = self
            .parameter_iterate_scale()
            .map_err(SaeSupportStationarityError::ParameterScale)?;
        // #2517 — report the certificate PER BLOCK, not as one scalar. The two
        // blocks are different quantities reached by different sweeps: the
        // decoder block is the exact-PSD-solve's own residual, the coordinate
        // block is the damped coordinate sweep's. A single `max` over both
        // cannot say which sweep failed to reach its own stationarity, so every
        // reader of this refusal has had to guess. Measured across eight shapes
        // (n 120..480, P 4..8, K 9..24, top_k 1..3, residual 1e-4 and 1e-2)
        // this refusal fires in EVERY arm at relative KKT 1.5e-3..9.7e-3 while
        // `max_change` is already ~1e-3, i.e. the iterate has stopped moving
        // and the certificate has not been reached — the split is what
        // distinguishes "a sweep is not solving its block" from "the blocks
        // disagree at the joint point".
        Err(format!(
            "SaeSupportSparseTerm::solve_fixed_point did not recur within {max_iter} \
             alternating cycles nor in the {max_iter} coupled cycles that follow \
             (raw KKT max={:.6e}, relative to objective {:.6e}: {:.6e}; \
             per block: decoder max={:.6e} l2={:.6e}, coordinate max={:.6e} l2={:.6e}; \
             parameter-space Newton step max={:.6e}, scale={parameter_scale:.6e}, relative={:.6e} \
             vs tolerance {tolerance:.6e} (decoder {:.6e}, coordinate {:.6e}); \
             certificate accepts raw-relative OR parameter-relative KKT; \
             last parameter max_change={last_max_change:.6e}, \
             joint Newton steps accepted={joint_accepted}, gauge-invariant limbs required)",
            stationarity.max_abs(),
            objective,
            stationarity.max_abs() / objective.abs().max(1.0),
            stationarity.decoder_max_abs,
            stationarity.decoder_l2,
            stationarity.coordinate_max_abs,
            stationarity.coordinate_l2,
            stationarity.scaled_max_abs(),
            stationarity.scaled_max_abs() / parameter_scale,
            stationarity.decoder_scaled_max_abs,
            stationarity.coordinate_scaled_max_abs,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment_state::SaeAssignmentAtomSpec;
    use ndarray::array;

    /// Three atoms of two bases over two channels, with every atom shared by two
    /// rows so the scatter has a real accumulation order to preserve.
    fn beta_operator_fixture() -> SupportBetaOperator {
        let width = 2usize;
        let basis_sizes = vec![2usize, 2, 2];
        let beta_offsets = vec![0usize, 4, 8];
        // Deliberately not round: partial sums of these are not exactly
        // representable, so a reassociated sum would differ in the low bits.
        let phi = |a: f64, b: f64| ndarray::Array1::from(vec![a, b]);
        let mk = |offset: usize, a: f64, b: f64| SupportBasisBlock {
            beta_offset: offset,
            phi: phi(a, b),
        };
        let third = 1.0_f64 / 3.0;
        let root = 2.0_f64.sqrt() / 7.0;
        let rows = vec![
            SupportLinearizedRow {
                blocks: vec![mk(0, third, root), mk(4, -root, third * 0.5)],
                jacobian: ndarray::Array2::zeros((1, 1)),
            },
            SupportLinearizedRow {
                blocks: vec![mk(4, third * 1.7, -root), mk(8, root * 3.1, third)],
                jacobian: ndarray::Array2::zeros((1, 1)),
            },
            SupportLinearizedRow {
                blocks: vec![mk(0, -third * 0.9, root * 2.3), mk(8, third, -root)],
                jacobian: ndarray::Array2::zeros((1, 1)),
            },
        ];
        let mut atom_blocks: Vec<Vec<(u32, u32)>> = vec![Vec::new(); 3];
        for (row_index, row) in rows.iter().enumerate() {
            for (block_index, block) in row.blocks.iter().enumerate() {
                let atom = beta_offsets
                    .iter()
                    .position(|&o| o == block.beta_offset)
                    .expect("offset belongs to an atom");
                atom_blocks[atom].push((row_index as u32, block_index as u32));
            }
        }
        let penalties = vec![
            array![[2.0, -0.5], [-0.5, 1.25]],
            array![[1.0, third], [third, 3.0]],
            array![[0.75, 0.0], [0.0, 0.5]],
        ];
        SupportBetaOperator {
            rows,
            atom_blocks,
            beta_offsets,
            basis_sizes,
            penalties,
            lambda_smooth: vec![0.7, 1.9, third],
            output_dim: width,
            beta_dim: 12,
        }
    }

    /// The serial sweep `apply` replaced, kept here as the reference the fan-out
    /// has to reproduce exactly.
    fn beta_operator_apply_serially(
        op: &SupportBetaOperator,
        vector: ndarray::ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) {
        out.fill(0.0);
        let mut output = vec![0.0; op.output_dim];
        for row in &op.rows {
            output.fill(0.0);
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * op.output_dim;
                    for channel in 0..op.output_dim {
                        output[channel] += block.phi[basis] * vector[base + channel];
                    }
                }
            }
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * op.output_dim;
                    for channel in 0..op.output_dim {
                        out[base + channel] += block.phi[basis] * output[channel];
                    }
                }
            }
        }
        for atom in 0..op.penalties.len() {
            let lambda = op.lambda_smooth[atom];
            let m = op.basis_sizes[atom];
            let offset = op.beta_offsets[atom];
            for left in 0..m {
                for right in 0..m {
                    let weight = lambda * op.penalties[atom][[left, right]];
                    for channel in 0..op.output_dim {
                        out[offset + left * op.output_dim + channel] +=
                            weight * vector[offset + right * op.output_dim + channel];
                    }
                }
            }
        }
    }

    #[test]
    fn beta_operator_fan_out_is_bit_identical_to_the_serial_sweep() {
        let op = beta_operator_fixture();
        let vector = Array1::from(
            (0..12)
                .map(|i| ((i as f64) * 0.37).sin() + (i as f64) / 7.0)
                .collect::<Vec<f64>>(),
        );

        let mut expected = Array1::<f64>::zeros(12);
        beta_operator_apply_serially(&op, vector.view(), &mut expected);
        let mut actual = Array1::<f64>::zeros(12);
        op.apply(vector.view(), &mut actual);

        for index in 0..12 {
            // EXACT: reassociating the sum would move the low bits, and moving
            // them is precisely what this test exists to forbid.
            assert_eq!(
                actual[index].to_bits(),
                expected[index].to_bits(),
                "entry {index}: fan-out {} is not bit-identical to serial {}",
                actual[index],
                expected[index]
            );
        }
        // Guard against a fixture that made the assertion trivial.
        assert!(
            expected.iter().any(|v| v.abs() > 1e-6),
            "fixture produced an all-zero reference, so the comparison proves nothing"
        );
    }

    #[test]
    fn beta_operator_fan_out_is_stable_across_repeated_application() {
        // rayon may split the work differently between calls; the result must
        // not depend on how it happened to schedule.
        let op = beta_operator_fixture();
        let vector = Array1::from((0..12).map(|i| 1.0 / (i as f64 + 1.3)).collect::<Vec<f64>>());
        let mut first = Array1::<f64>::zeros(12);
        op.apply(vector.view(), &mut first);
        for _ in 0..8 {
            let mut again = Array1::<f64>::zeros(12);
            op.apply(vector.view(), &mut again);
            for index in 0..12 {
                assert_eq!(again[index].to_bits(), first[index].to_bits());
            }
        }
    }

    /// `S` is rank 2 with null direction `e3`; `G` is full rank.
    fn penalized_solve_fixture() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let penalty = array![[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
        let gram = array![[3.0, 1.0, 0.5], [1.0, 4.0, 0.25], [0.5, 0.25, 2.0]];
        let rhs = array![[1.0], [2.0], [3.0]];
        (gram, penalty, rhs)
    }

    #[test]
    fn penalized_solve_survives_the_smoothing_fellner_schall_actually_produces() {
        let (gram, penalty, rhs) = penalized_solve_fixture();
        // The magnitude Fellner-Schall reaches when an atom's roughness goes to
        // zero -- the ladder picking the linear rung, not a divergence.
        let lambda = 2.2e16;

        // The old route: assemble `G + lambda*S` and solve it. This must FAIL,
        // or the fix below is answering a question nobody asked.
        let mut assembled = &penalty * lambda;
        assembled += &gram;
        let assembled_result = SaeSupportSparseTerm::solve_psd_minimum_norm(
            &assembled,
            &rhs,
            "assembled",
        );
        assert!(
            assembled_result.is_err(),
            "assembling G + lambda*S was expected to lose null(S) to its own rank floor, \
             but it returned {assembled_result:?}"
        );

        let solved = SaeSupportSparseTerm::solve_penalized_normal_equations(
            &gram, &penalty, lambda, &rhs, "penalized",
        )
        .expect("the lambda-free scaling must solve what the assembled matrix could not");

        // Exact limit, by hand: the penalty annihilates everything outside
        // null(S) = span(e3), so beta = e3 * (rhs_3 / G_33) = e3 * 3/2.
        assert!(solved[[0, 0]].abs() < 1e-9, "range(S) must be driven to zero, got {}", solved[[0, 0]]);
        assert!(solved[[1, 0]].abs() < 1e-9, "range(S) must be driven to zero, got {}", solved[[1, 0]]);
        assert!(
            (solved[[2, 0]] - 1.5).abs() < 1e-9,
            "null(S) must keep its unpenalised least squares value 1.5, got {}",
            solved[[2, 0]]
        );
    }

    #[test]
    fn penalized_solve_agrees_with_the_assembled_matrix_where_that_is_conditioned() {
        let (gram, penalty, rhs) = penalized_solve_fixture();
        // Stability at 1e16 is worthless if it moved the answer at lambdas that
        // were never in trouble.
        for lambda in [0.0, 1e-3, 1.0, 25.0, 1e4] {
            let mut assembled = &penalty * lambda;
            assembled += &gram;
            let reference =
                SaeSupportSparseTerm::solve_psd_minimum_norm(&assembled, &rhs, "reference")
                    .expect("well-conditioned assembled solve");
            let solved = SaeSupportSparseTerm::solve_penalized_normal_equations(
                &gram, &penalty, lambda, &rhs, "penalized",
            )
            .expect("well-conditioned scaled solve");
            for index in 0..3 {
                let gap = (solved[[index, 0]] - reference[[index, 0]]).abs();
                assert!(
                    gap < 1e-9 * reference[[index, 0]].abs().max(1.0),
                    "lambda={lambda} entry {index}: scaled {} vs assembled {}",
                    solved[[index, 0]],
                    reference[[index, 0]]
                );
            }
        }
    }
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

    /// #2634 second-order gate: a periodic chart can be first-order attracted
    /// to a blockwise fixed point whose exact joint curvature is negative.  The
    /// generalized mode has arbitrary sign, so the escape must test both signs,
    /// choose an actual objective decrease, and leave the ordinary fixed-point
    /// solver in a certifiable minimum basin.
    #[test]
    fn exact_generalized_mode_escapes_a_periodic_saddle_2634() {
        let evaluator: Arc<dyn SaeBasisSecondJet> =
            Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
        // At t=1/2 the von-Mises prior has curvature -1.  A small sine decoder
        // contributes only (0.1*2pi)^2 through Gauss--Newton, so the exact
        // coordinate curvature remains resolved negative while the majorizer
        // pencil is strictly positive and therefore classifiable.
        let atoms = vec![atom(
            "periodic-saddle",
            SaeAtomBasisKind::Periodic,
            1,
            evaluator,
            &[0.5],
            array![[0.0], [0.1], [0.0]],
        )];
        let specs = vec![SaeAssignmentAtomSpec {
            latent_dim: 1,
            id_mode: LatentIdMode::None,
            manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
            retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
            latent_id: 1,
        }];
        let state = SaeAssignmentState::from_topk_support_heterogeneous(
            1,
            1,
            1,
            specs,
            vec![vec![0]],
            vec![vec![1.0]],
            vec![vec![0.5]],
        )
        .expect("state");
        let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let target = array![[0.0_f64]];
        let lambda = vec![1.0_f64];
        let ard = vec![vec![1.0_f64]];
        let system = term
            .assemble_arrow_schur(target.view(), &lambda, &ard)
            .expect("arrow system");
        let (beta_offsets, _) = term.beta_layout().expect("beta layout");
        let rows = term
            .support_outer_differential_rows(target.view(), &ard, &beta_offsets)
            .expect("exact differential rows");
        let mode = term
            .support_outer_negative_curvature_mode(&system, &rows)
            .expect("curvature classification")
            .expect("the planted saddle has a resolved negative mode");
        assert!(
            mode.curvature < -f64::EPSILON.sqrt(),
            "planted generalized curvature must be resolved negative, got {}",
            mode.curvature,
        );

        let before = term
            .penalized_objective(target.view(), &lambda, &ard)
            .expect("saddle objective");
        let escaped = term
            .escape_support_negative_curvature(
                target.view(),
                &lambda,
                &ard,
                &mode,
                before,
                1.0,
                &mut Vec::new(),
                &mut Vec::new(),
            )
            .expect("exact saddle escape")
            .expect("one orientation must descend exact negative curvature");
        assert!(escaped < before, "escape must strictly lower the objective");

        let report = term
            .solve_fixed_point(target.view(), &lambda, &ard, 64, 1.0e-8, 1.0)
            .expect("escaped state converges to a minimum basin");
        assert!(report.recurred && report.objective < before);
    }

}
