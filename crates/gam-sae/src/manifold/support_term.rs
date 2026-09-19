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
    /// diagonal: a diagonal-preconditioned first-order residual.
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
    ///
    /// It is NOT the remaining parameter displacement (#2933 F08). That is
    /// `A⁻¹g` with the full coupled Hessian, and a diagonal cannot see coupled
    /// weakly curved directions: for `H = [[1, 1−ε], [1−ε, 1]]` at `θ − θ* =
    /// (1, −1)` the diagonal-scaled gradient is `ε` while the displacement is 1.
    /// A likelihood-flat reparameterisation curved only by the priors is the
    /// same shape. This quantity schedules the certificate;
    /// [`SaeSupportNewtonDisplacement`] is the certificate.
    pub decoder_scaled_max_abs: f64,
    /// The coordinate block's counterpart: its gradient divided by its own
    /// curvature diagonal (`Σ_out J² + ARD curvature`).
    pub coordinate_scaled_max_abs: f64,
}

impl SaeSupportStationarity {
    /// The raw (gradient-space) residual, kept for reporting and for every
    /// consumer that compares against a historical number.
    pub fn max_abs(self) -> f64 {
        self.decoder_max_abs.max(self.coordinate_max_abs)
    }

    /// The larger of the two blocks' diagonal-preconditioned residuals. See
    /// [`Self::decoder_scaled_max_abs`] for why it is intensive and why it is
    /// not a displacement.
    pub fn scaled_max_abs(self) -> f64 {
        self.decoder_scaled_max_abs
            .max(self.coordinate_scaled_max_abs)
    }

    /// Whether the first-order residual is small enough to price the exact
    /// Newton displacement: the extensive raw gradient relative to the
    /// objective, or the componentwise diagonal-scaled residual relative to the
    /// iterate. A SCHEDULE, never a certificate (#2933 F08): the raw limb is
    /// relative to an objective that can carry any additive constant, and the
    /// diagonal limb is blind to coupled weakly curved directions.
    pub(crate) fn first_order_screen(
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

/// The exact Newton displacement `Δ = A⁻¹g` at an installed support state, per
/// block, in parameter units (#2933 F08).
///
/// `A` is the exact stationarity Jacobian of the penalized inner objective, the
/// Gauss–Newton arrow plus the residual second-jet and exact prior curvature: the
/// operator the outer profile adjoint inverts, not its majorizer. Near a
/// nondegenerate stationary point `θ − θ* = Δ + O(‖Δ‖²)`, so each component is how
/// far that parameter still has to move, including along directions only a
/// simultaneous change of coordinates and decoder can travel. It is covariant
/// under reparameterisation and intensive under row replication, because `A` and
/// `g` scale together.
#[derive(Debug, Clone, PartialEq)]
pub struct SaeSupportNewtonDisplacement {
    pub decoder_max_abs: f64,
    pub coordinate_max_abs: f64,
    /// `gᵀA⁻¹g`, the squared Newton decrement; the quadratic model predicts a
    /// decrease of half of it.
    pub decrement_sq: f64,
    /// The coordinate block of `Δ`, in the compact row layout
    /// (`assemble_arrow_schur`'s `row_offsets`), so `θ* ≈ θ − Δ`. A consumer that
    /// bounds a quantity's error at an inexact inner state reads `−Δᵀ ∂_θ(…)` off it.
    pub coordinates: Array1<f64>,
    /// The decoder block of `Δ`, in `beta_layout` order:
    /// `offset(atom) + basis · P + channel`.
    pub decoder: Array1<f64>,
}

impl SaeSupportNewtonDisplacement {
    pub fn max_abs(&self) -> f64 {
        self.decoder_max_abs.max(self.coordinate_max_abs)
    }

    /// The inner certificate: every parameter's remaining Newton displacement is
    /// within `tolerance` of the iterate scale.
    pub(crate) fn certifies(&self, parameter_scale: f64, tolerance: f64) -> bool {
        parameter_scale.is_finite()
            && parameter_scale >= 1.0
            && tolerance.is_finite()
            && tolerance > 0.0
            && self.max_abs().is_finite()
            && self.max_abs() <= tolerance * parameter_scale
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
    /// The exact Newton displacement the state certified on (#2933 F08).
    pub newton_displacement: SaeSupportNewtonDisplacement,
    pub max_recurrence_change: f64,
    /// True only after a second complete decoder/coordinate cycle recurs within
    /// the same tolerance at the raw (undamped) stationarity point and the exact
    /// Newton displacement there is within tolerance.
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

    /// Guaranteed upper bounds on `‖H_tβ^(r)‖₂` of [`Self::htbeta_forward`], which is
    /// `jacobian_r · J_r` for the gather `J_r` over the row's basis blocks (#2627):
    /// `‖jacobian_r‖_F · √(‖J_r‖_∞·‖J_r‖₁)`, with the gather factor from
    /// [`super::kronecker::gather_norm_squared_upper_bound`].
    fn row_norm_bounds(&self) -> Arc<[f64]> {
        let width = self.output_dim;
        let bounds: Vec<f64> = self
            .rows
            .iter()
            .map(|row| {
                let mut support: Vec<(usize, f64)> = row
                    .blocks
                    .iter()
                    .flat_map(|block| {
                        block
                            .phi
                            .iter()
                            .enumerate()
                            .map(move |(basis, &phi)| (block.beta_offset + basis * width, phi))
                    })
                    .collect();
                let gather_squared =
                    super::kronecker::gather_norm_squared_upper_bound(&mut support, width);
                let mut jacobian_squares = 0.0_f64;
                for &value in row.jacobian.iter() {
                    jacobian_squares += value * value;
                }
                let depth = (row.jacobian.len() + 1) + (2 * support.len() + 2) + 1;
                gam_solve::arrow_schur::guaranteed_norm_upper_bound(
                    jacobian_squares.sqrt() * gather_squared.sqrt(),
                    depth,
                )
            })
            .collect();
        Arc::from(bounds.into_boxed_slice())
    }

    /// The cross block's declaration for `set_row_htbeta_operator` (#2627):
    /// [`Self::row_norm_bounds`] and the apply depth. [`Self::htbeta_forward`] gathers over
    /// the row's basis blocks (`b_r` terms per channel) and applies the Jacobian
    /// (`output_dim` terms), so its depth is `b_r + output_dim`.
    /// [`Self::htbeta_transpose`] accumulates `jacobianᵀ v` (`d_r` terms) and scatters
    /// `φ·output` into each β entry once per covering basis (at most `b_r` additions after
    /// the product), so its depth is `d_r + b_r + 1`.
    fn htbeta_declaration(&self) -> gam_solve::arrow_schur::RowHtbetaDeclaration {
        let widest_basis = self
            .rows
            .iter()
            .map(|row| row.blocks.iter().map(|block| block.phi.len()).sum::<usize>())
            .max()
            .unwrap_or(0);
        let widest_latent = self
            .rows
            .iter()
            .map(|row| row.jacobian.nrows())
            .max()
            .unwrap_or(0);
        gam_solve::arrow_schur::RowHtbetaDeclaration {
            row_norm_bounds: self.row_norm_bounds(),
            apply_depth: widest_basis + self.output_dim.max(widest_latent) + 1,
        }
    }
}

/// #2576: the support `H_ββ` states its own blocks and diagonal. Behind the
/// closure adapter the block-Jacobi build probed it one column at a time, a full
/// `apply` over every row per column: k = 7680 applies per coupled step on the
/// 3000x48 chart (profile job 609002). Every method adds the same terms in the
/// order `apply` accumulates them, so the preconditioner it builds is unchanged.
impl gam_solve::arrow_schur::BetaPenaltyOp for SupportBetaOperator {
    fn dim(&self) -> usize {
        self.beta_dim
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let mut applied = Array1::<f64>::zeros(self.beta_dim);
        self.apply(ndarray::ArrayView1::from(x), &mut applied);
        for (target, value) in y.iter_mut().zip(applied.iter()) {
            *target += value;
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        self.matvec(beta, out);
    }

    fn diagonal(&self, diag: &mut [f64]) {
        let width = self.output_dim;
        for row in &self.rows {
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * width;
                    for channel in 0..width {
                        diag[base + channel] += block.phi[basis] * block.phi[basis];
                    }
                }
            }
        }
        for atom in 0..self.penalties.len() {
            let lambda = self.lambda_smooth[atom];
            let offset = self.beta_offsets[atom];
            for basis in 0..self.basis_sizes[atom] {
                for channel in 0..width {
                    diag[offset + basis * width + channel] +=
                        lambda * self.penalties[atom][[basis, basis]];
                }
            }
        }
    }

    fn block(
        &self,
        id: gam_solve::arrow_schur::BetaBlockId,
        offsets: &[Range<usize>],
        out: &mut Array2<f64>,
    ) {
        let range = &offsets[id.0];
        let width = self.output_dim;
        let overlaps = |start: usize, len: usize| start < range.end && start + len > range.start;
        let local =
            |index: usize| (range.start <= index && index < range.end).then(|| index - range.start);
        for row in &self.rows {
            for left in &row.blocks {
                if !overlaps(left.beta_offset, left.phi.len() * width) {
                    continue;
                }
                for right in &row.blocks {
                    if !overlaps(right.beta_offset, right.phi.len() * width) {
                        continue;
                    }
                    for li in 0..left.phi.len() {
                        for lj in 0..right.phi.len() {
                            let weight = left.phi[li] * right.phi[lj];
                            for channel in 0..width {
                                if let (Some(bi), Some(bj)) = (
                                    local(left.beta_offset + li * width + channel),
                                    local(right.beta_offset + lj * width + channel),
                                ) {
                                    out[[bi, bj]] += weight;
                                }
                            }
                        }
                    }
                }
            }
        }
        for atom in 0..self.penalties.len() {
            let offset = self.beta_offsets[atom];
            let m = self.basis_sizes[atom];
            if !overlaps(offset, m * width) {
                continue;
            }
            let lambda = self.lambda_smooth[atom];
            for left in 0..m {
                for right in 0..m {
                    let weight = lambda * self.penalties[atom][[left, right]];
                    for channel in 0..width {
                        if let (Some(bi), Some(bj)) = (
                            local(offset + left * width + channel),
                            local(offset + right * width + channel),
                        ) {
                            out[[bi, bj]] += weight;
                        }
                    }
                }
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut dense = Array2::<f64>::zeros((self.beta_dim, self.beta_dim));
        self.block(
            gam_solve::arrow_schur::BetaBlockId(0),
            &[0..self.beta_dim],
            &mut dense,
        );
        dense
    }

    fn fingerprint(&self, hasher: &mut gam_runtime::warm_start::Fingerprinter) {
        hasher.write_str("sae-support-beta-operator-v1");
        hasher.write_usize(self.beta_dim);
        hasher.write_usize(self.output_dim);
        for row in &self.rows {
            hasher.write_usize(row.blocks.len());
            for block in &row.blocks {
                hasher.write_usize(block.beta_offset);
                for &value in block.phi.iter() {
                    hasher.write_f64(value);
                }
            }
        }
        for atom in 0..self.penalties.len() {
            hasher.write_usize(self.beta_offsets[atom]);
            hasher.write_usize(self.basis_sizes[atom]);
            hasher.write_f64(self.lambda_smooth[atom]);
            for &value in self.penalties[atom].iter() {
                hasher.write_f64(value);
            }
        }
    }

    /// `M = Σ_r |J_r|ᵀ|J_r| + ⊕_k |λ_k|·max(|S_k|, |S_k|ᵀ) ⊗ I_P` (#2627): the data
    /// Gram's gather and scatter read `|φ|`, and each smoothing block reads its own
    /// majorant, so `M ≥ |P|` and `M` is symmetric.
    fn accumulate_abs_majorant_matvec(&self, x: &[f64], out: &mut [f64]) -> usize {
        let width = self.output_dim;
        let mut slot = vec![0.0_f64; width];
        let mut widest_row = 0usize;
        let mut total_basis = 0usize;
        for row in &self.rows {
            slot.fill(0.0);
            let row_basis: usize = row.blocks.iter().map(|block| block.phi.len()).sum();
            widest_row = widest_row.max(row_basis);
            total_basis += row_basis;
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let weight = block.phi[basis].abs();
                    let base = block.beta_offset + basis * width;
                    for channel in 0..width {
                        slot[channel] += weight * x[base + channel];
                    }
                }
            }
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let weight = block.phi[basis].abs();
                    let base = block.beta_offset + basis * width;
                    for channel in 0..width {
                        out[base + channel] += weight * slot[channel];
                    }
                }
            }
        }
        let mut widest_penalty = 0usize;
        for atom in 0..self.penalties.len() {
            let lambda = self.lambda_smooth[atom].abs();
            let m = self.basis_sizes[atom];
            widest_penalty = widest_penalty.max(m);
            let offset = self.beta_offsets[atom];
            let penalty = &self.penalties[atom];
            for left in 0..m {
                for channel in 0..width {
                    let mut acc = 0.0_f64;
                    for right in 0..m {
                        acc += penalty[[left, right]].abs().max(penalty[[right, left]].abs())
                            * x[offset + right * width + channel];
                    }
                    out[offset + left * width + channel] += lambda * acc;
                }
            }
        }
        widest_row + total_basis + widest_penalty + 4
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
    /// Per-atom period of the ARD coordinate prior, resolved alongside
    /// `atom_axis_periods` from the same fixed geometry: the kind's
    /// [`SaeAtomBasisKind::ard_axis_periods`] of the wrap periods, so a quotient
    /// atom's half-turned axis carries its deck-invariant half period
    /// (#2933 F25). Every prior consumer reads this cache; `atom_axis_periods`
    /// stays the chart's wrap period.
    atom_ard_axis_periods: Vec<Vec<Option<f64>>>,
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
    majorizer_eigenvectors: Array2<f64>,
    majorizer_stiffness: Array1<f64>,
    curvatures: Array1<f64>,
    generalized_vectors: Array2<f64>,
    minimum_backward_error: f64,
}

struct SupportNegativeCurvatureMode {
    curvature: f64,
    backward_error: f64,
    direction: SaeArrowVector,
}

/// The dimensionless resolution of a generalized curvature of the dense support
/// pencil `(A, B)`: `√ε`, the smallest relative change an f64 assembly of either
/// matrix resolves. The saddle classifier refuses no mode at or above
/// `−max(√ε, backward error)`, the pseudoinverse drops every mode at or below
/// `max(√ε, backward error)`, and the positive-definite certificate that spares
/// the classifier its eigensystems certifies at half of it.
fn support_outer_curvature_floor() -> f64 {
    f64::EPSILON.sqrt()
}

/// A certificate that `A − τ·B` is positive definite ([`certify_shifted_pd`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CertifiedShiftedPd {
    pub(crate) dim: usize,
    pub(crate) tau: f64,
    /// Rounding band, in the units of the scaled matrix, that the factorization
    /// was taken against.
    pub(crate) delta: f64,
    /// How `delta` was derived.
    pub(crate) band: BandProvenance,
}

/// The derivation behind a [`CertifiedShiftedPd`]'s rounding band.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum BandProvenance {
    /// One dense diagonally scaled Cholesky: `δ = (3n² + n + 2)·ε·‖S₀‖_F`, with
    /// `scaled_frobenius = ‖S₀‖_F`.
    Dense { scaled_frobenius: f64 },
}

/// Why [`certify_shifted_pd`] did not certify `A − τ·B ≻ 0`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ShiftedPdRefusal {
    /// `A` and `B` are not square matrices of one size.
    ShapeMismatch { a: (usize, usize), b: (usize, usize) },
    /// `τ` or an entry of `A − τ·B` is not finite.
    NonFinite,
    /// A diagonal entry of `A − τ·B` is not positive, so that matrix is not
    /// positive definite.
    NonPositiveDiagonal { index: usize, value: f64 },
    /// The Cholesky factorization of `S₀ − δ·I` did not complete: `A − τ·B` is
    /// not certified at this band, which does not say it is indefinite. `pivot`
    /// is the index of the non-positive pivot the factorization stopped at, when
    /// the factorization reports one.
    CholeskyFailed { delta: f64, pivot: Option<usize> },
}

/// Certify that the dense symmetric `A − τ·B` is positive definite, for any sign
/// of `τ`.
///
/// Let `M = A − τ·B` and `D = diag(M)`. A non-positive diagonal entry refutes
/// positive definiteness outright. Otherwise `S₀ = D^{-1/2}·M·D^{-1/2}` has a unit
/// diagonal, and a diagonal congruence preserves inertia, so `M ≻ 0` iff `S₀ ≻ 0`.
/// Forming `S₀` in f64 moves each entry by at most `2ε` of itself.
///
/// A Cholesky factorization of `S` that runs to completion computes `RᵀR = S + ΔS`
/// with `|ΔS| ≤ γ_{3n+1}·|Rᵀ|·|R|` (Higham, *Accuracy and Stability of Numerical
/// Algorithms*, 2nd ed., Theorem 10.5), so to first order `‖ΔS‖₂ ≤ (3n + 1)·n·ε·‖S‖₂`.
/// The factorization is applied to `S₀ − δ·I` with `δ = (3n² + n + 2)·ε·‖S₀‖_F`,
/// which bounds both that backward error and the scaling error because
/// `‖S₀‖₂ ≤ ‖S₀‖_F`. If it completes, `S₀ − δ·I + E ≻ 0` for some `‖E‖₂ ≤ δ`, so
/// `λ_min(S₀) > 0` and `M ≻ 0`. Any failure is a refusal and never a claim of
/// indefiniteness, so a caller falls back to an exact decision. The band is the
/// bound, not a tuned margin.
pub(crate) fn certify_shifted_pd(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    tau: f64,
) -> Result<CertifiedShiftedPd, ShiftedPdRefusal> {
    let dim = a.nrows();
    if a.dim() != (dim, dim) || b.dim() != (dim, dim) {
        return Err(ShiftedPdRefusal::ShapeMismatch {
            a: a.dim(),
            b: b.dim(),
        });
    }
    if !tau.is_finite() {
        return Err(ShiftedPdRefusal::NonFinite);
    }
    let mut scaled = a.to_owned();
    scaled.scaled_add(-tau, &b);
    if scaled.iter().any(|value| !value.is_finite()) {
        return Err(ShiftedPdRefusal::NonFinite);
    }
    let mut inverse_root = Vec::with_capacity(dim);
    for index in 0..dim {
        let value = scaled[[index, index]];
        if !(value > 0.0) {
            return Err(ShiftedPdRefusal::NonPositiveDiagonal { index, value });
        }
        inverse_root.push(value.sqrt().recip());
    }
    for row in 0..dim {
        for column in 0..dim {
            scaled[[row, column]] *= inverse_root[row] * inverse_root[column];
        }
    }
    let scaled_frobenius = scaled.iter().map(|value| value * value).sum::<f64>().sqrt();
    let n = dim as f64;
    let delta = (3.0 * n * n + n + 2.0) * f64::EPSILON * scaled_frobenius;
    for index in 0..dim {
        scaled[[index, index]] -= delta;
    }
    match scaled.cholesky(Side::Lower) {
        Ok(_) => Ok(CertifiedShiftedPd {
            dim,
            tau,
            delta,
            band: BandProvenance::Dense { scaled_frobenius },
        }),
        Err(error) => Err(ShiftedPdRefusal::CholeskyFailed {
            delta,
            pivot: match error {
                gam_linalg::faer_ndarray::FaerLinalgError::Cholesky(
                    faer::linalg::solvers::LltError::NonPositivePivot { index },
                ) => Some(index),
                _ => None,
            },
        }),
    }
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

/// The displacement the last accepted support coupled step applied, carried to
/// the next coupled step together with the support layout it was taken under
/// (#2576). See `joint_newton_step`.
struct SupportJointStepMemory {
    t: Array1<f64>,
    beta: Array1<f64>,
    support: Vec<u32>,
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

/// Application of the majorizer inverse used as the flexible-GMRES
/// preconditioner.
///
/// The reduced solve is conjugate gradients on the SPD reduced Schur, capped at its
/// algebraic dimension `k`. Flexible GMRES admits a different preconditioner at
/// every Arnoldi direction and certifies the physical residual `‖rhs − A x‖`
/// itself, so a CG iterate that has not reached the `sqrt(eps)` floor within `k`
/// directions is still a legitimate `P_j(v_j)`; only the iteration count of the
/// outer solve depends on how good it is. Refusing that iterate ended the whole
/// outer evaluation on exactly the narrow, ill-conditioned borders where
/// finite-precision CG needs more than `k` directions: the exact-arithmetic
/// termination premise #2576 corrected for the rational log-det surrogate. A
/// breakdown is still refused.
fn support_arrow_majorizer_inverse(
    system: &ArrowSchurSystem,
    factors: &ArrowFactorSlab,
    rhs: &SaeArrowVector,
    exact_reduced_inverse: Option<&[Array1<f64>]>,
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
    } else if let Some(vectors) = exact_reduced_inverse {
        // #2576: a dense-spectrum evidence bundle spans the inverse of the reduced Schur
        // the evidence priced, so the reduced solve is one fold over its vectors, not a
        // √ε CG. Where that evidence pinned a numerically null direction at unit
        // stiffness, the fold applies the same pin; flexible GMRES admits either.
        exact_reduced_inverse_apply(vectors, &reduced_rhs)
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
            log::debug!(
                "support outer differential: reduced-Schur preconditioner iterate at relative \
                 residual {:.3e} (floor {:.3e}) after its {}-direction span; flexible GMRES \
                 certifies the physical residual",
                report.relative_residual,
                report.tolerance,
                system.k,
            );
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

/// #2576: `S⁻¹ r` off a derivative bundle that spans the reduced Schur's inverse
/// exactly, `(1/k) Σ_i x_i (x_iᵀ r)` with `x_i = √(k/λ_i)·v_i` over the complete dense
/// spectrum. The vectors fold over the length-only tree, so the result does not
/// depend on thread count.
fn exact_reduced_inverse_apply(vectors: &[Array1<f64>], rhs: &Array1<f64>) -> Array1<f64> {
    let scale = 1.0 / vectors.len() as f64;
    gam_linalg::pairwise_reduce::par_deterministic_block_fold(
        vectors.len(),
        |range: core::ops::Range<usize>| {
            let mut partial = Array1::<f64>::zeros(rhs.len());
            for vector in &vectors[range] {
                partial.scaled_add(scale * vector.dot(rhs), vector);
            }
            partial
        },
        |mut left: Array1<f64>, right: Array1<f64>| {
            left += &right;
            left
        },
    )
    .unwrap_or_else(|| Array1::<f64>::zeros(rhs.len()))
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
    // `G~` is PSD and `P~` diagonal, so the shifted matrix's largest eigenvalue is at
    // most `tr(G~) + max P~`. The trace also covers the rotation's formation
    // rounding. An eigenvalue within `m·ε` of that magnitude is a null mode, not rank.
    let scaled_trace = (0..m).map(|mode| scaled[[mode, mode]]).sum::<f64>();
    let largest_penalty_fraction = penalty_fraction.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = f64::EPSILON * (scaled_trace + largest_penalty_fraction) * m.max(1) as f64;
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
        let atom_axis_periods: Vec<Vec<Option<f64>>> = (0..k_atoms)
            .map(|atom| assignment.atom_axis_periods(atom))
            .collect();
        let atom_ard_axis_periods = atom_axis_periods
            .iter()
            .enumerate()
            .map(|(atom, periods)| atoms[atom].basis_kind().ard_axis_periods(periods))
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
            atom_ard_axis_periods,
        })
    }

    /// Axis periodicity of one atom's coordinate block: `None` on a Euclidean
    /// axis, `Some(period)` on a circular one.
    fn atom_axis_periods(&self, atom: usize) -> &[Option<f64>] {
        &self.atom_axis_periods[atom]
    }

    /// Period of one atom's ARD coordinate prior per axis; see the
    /// `atom_ard_axis_periods` field.
    fn atom_ard_axis_periods(&self, atom: usize) -> &[Option<f64>] {
        &self.atom_ard_axis_periods[atom]
    }

    /// Negative log normalizer of the ARD coordinate prior, summed over every
    /// active slot (#2933 F27 S1).
    ///
    /// Per slot this is `Σ_factors log Z_ard(α) − log(sheets)`: the per-row partition
    /// over the coordinate's actual support, its Laplace constant, and the quotient
    /// sheet count, as in the dense criterion's `loss.ard` (#2933 F24, F25, F26). A
    /// TopK coordinate exists only on a row's active support, so an atom pays its
    /// normalizer once per row that selects it. A zero precision is the typed
    /// exemption for an embedded-sphere axis, whose Bingham partition stays finite.
    /// On any other support it leaves no normalizer and is refused.
    pub(crate) fn ard_log_partition_total(
        &self,
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        self.validate_ard(ard_precisions)?;
        let mut total = 0.0_f64;
        for atom in 0..self.k_atoms() {
            let slots = self.atom_rows[atom].len();
            if slots == 0 {
                continue;
            }
            let alpha = Array1::from(ard_precisions[atom].clone());
            let log_alpha = alpha.mapv(f64::ln);
            let supports = self.assignment.atom_prior_supports(atom);
            let partition = SaeManifoldTerm::ard_log_partition(
                &supports,
                self.atom_ard_axis_periods(atom),
                log_alpha.view(),
                alpha.view(),
            )?;
            let per_slot = partition.per_factor.iter().sum::<f64>()
                - self.atoms[atom]
                    .basis_kind()
                    .ard_quotient_log_sheets(self.assignment.atom_coord_dim(atom));
            if !per_slot.is_finite() {
                return Err(format!(
                    "SaeSupportSparseTerm::ard_log_partition_total: atom {atom} has no finite ARD \
                     partition at precisions {:?} on supports {supports:?}",
                    ard_precisions[atom]
                ));
            }
            total += slots as f64 * per_slot;
        }
        Ok(total)
    }

    /// Total width of the compact coordinate state `T` — the concatenation of
    /// every row's active coordinate block.
    pub(crate) fn coordinate_state_len(&self) -> usize {
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

    /// #2576 — the share of a cycle's coordinate motion on one-dimensional
    /// Euclidean atoms that is an affine reparameterization `Δt = a + b·t` of each
    /// atom's own coordinates. For a polynomial decoder that is the translation
    /// and scale orbit: the counter-transformed decoder leaves the data fit
    /// unchanged, so only the ARD prior and the coefficient ridge carry curvature
    /// along it. `start` and `residual` are the cycle's compact coordinate snapshot
    /// and its wrapped displacement. Returns the displacement-weighted share over
    /// those atoms and, for the atom with the largest displacement energy, its
    /// index, its own share and that energy.
    fn euclidean_affine_motion_share(
        &self,
        start: &[f64],
        residual: &[f64],
    ) -> (f64, Option<(usize, f64, f64)>) {
        // Per atom: rows, Σt, Σt², Σd, Σt·d, Σd².
        let mut moments = vec![[0.0_f64; 6]; self.k_atoms()];
        let mut cursor = 0usize;
        for row in 0..self.n_obs() {
            for &atom in self.assignment.support_indices(row) {
                let periods = self.atom_axis_periods(atom as usize);
                let line = matches!(periods, [None]);
                for _ in periods {
                    if line {
                        let (t, d) = (start[cursor], residual[cursor]);
                        let entry = &mut moments[atom as usize];
                        entry[0] += 1.0;
                        entry[1] += t;
                        entry[2] += t * t;
                        entry[3] += d;
                        entry[4] += t * d;
                        entry[5] += d * d;
                    }
                    cursor += 1;
                }
            }
        }
        let mut explained_total = 0.0_f64;
        let mut energy_total = 0.0_f64;
        let mut binding: Option<(usize, f64, f64)> = None;
        for (atom, [count, sum_t, sum_tt, sum_d, sum_td, sum_dd]) in
            moments.into_iter().enumerate()
        {
            // A line through two rows fits them exactly, so only atoms with at
            // least three rows say anything about the motion's shape.
            if count < 3.0 || !(sum_dd > 0.0) {
                continue;
            }
            let spread = sum_tt - sum_t * sum_t / count;
            let covariance = sum_td - sum_t * sum_d / count;
            // Least squares of d on [1, t]: the mean term plus the centred slope term.
            let mut explained = sum_d * sum_d / count;
            if spread > 0.0 {
                explained += covariance * covariance / spread;
            }
            let explained = explained.min(sum_dd);
            explained_total += explained;
            energy_total += sum_dd;
            if binding.is_none_or(|(_, _, energy)| sum_dd > energy) {
                binding = Some((atom, explained / sum_dd, sum_dd));
            }
        }
        let share = if energy_total > 0.0 {
            explained_total / energy_total
        } else {
            0.0
        };
        (share, binding)
    }

    pub fn n_obs(&self) -> usize {
        self.assignment.n_obs()
    }

    /// The smallest decrease of the penalized objective that is a measured descent
    /// rather than two roundings of one number (#2634). `penalized_objective` sums
    /// `n_obs · output_dim` residual cells plus the penalty blocks, so its
    /// resolution is `√cells · EPSILON · |f|`. Installing a state that a smaller
    /// "decrease" bought is motion the loop manufactures and then refuses to
    /// certify.
    fn objective_descent_resolution(&self, objective: f64) -> f64 {
        let cells = (self.n_obs() * self.output_dim()).max(1) as f64;
        cells.sqrt() * f64::EPSILON * objective.abs()
    }

    /// The relative tolerance the support fixed point certifies to, derived from the
    /// objective's arithmetic resolution rather than chosen (#2023, #2469).
    ///
    /// `penalized_objective` sums `n_obs · output_dim` residual cells plus the penalty
    /// blocks, so it is known only to `r·|f|` with `r = √(n_obs · output_dim)·ε` (the
    /// #2634 descent resolution). A stationary point is resolved no finer than `√r`:
    /// near a minimum `f − f* ≈ ½·h·δ²`, so a displacement below `√(2·r·|f|/h)` moves
    /// the objective by less than its resolution, and the curvature-scaled first-order
    /// residual `g/h = δ` carries the same bound. Every limb of the certificate
    /// (objective recurrence, first-order residual, state recurrence), the per-row
    /// coordinate skip and the coupled step's linear solve are asked for `√r` and no
    /// finer.
    pub fn fixed_point_tolerance(&self) -> f64 {
        let cells = (self.n_obs() * self.output_dim()).max(1) as f64;
        (cells.sqrt() * f64::EPSILON).sqrt()
    }

    /// Atoms no row selects. A support move can leave some, and each one's decoder
    /// block then carries only its penalty, so that penalty's null space is a
    /// direction the data do not identify (#2576).
    pub(crate) fn atoms_without_rows(&self) -> Vec<usize> {
        self.atom_rows
            .iter()
            .enumerate()
            .filter_map(|(atom, rows)| rows.is_empty().then_some(atom))
            .collect()
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
            let mut sine_absolute = 0.0_f64;
            let mut cosine_absolute = 0.0_f64;
            for &(row, slot) in &self.atom_rows[atom_index] {
                let phase = kappa * self.assignment.coords_for_slot(row, slot)[0];
                let (sin, cos) = phase.sin_cos();
                sine += sin;
                cosine += cos;
                sine_absolute += sin.abs();
                cosine_absolute += cos.abs();
            }
            let resultant = sine.hypot(cosine);
            // Each phasor term is formed by one product and one `sin_cos`, the
            // terms are summed naively over the atom's rows, and `hypot` rounds
            // once more, so the resultant is uncertain by at most `γ_{rows+2}` of
            // the components' absolute sums.
            let rows = self.atom_rows[atom_index].len();
            let resultant_band = gam_linalg::roundoff::accumulation_growth(rows + 2)
                * sine_absolute.hypot(cosine_absolute);
            if resultant <= resultant_band {
                // The profiled prior is phase-invariant when the circular
                // resultant vanishes, so no origin is statistically selected.
                continue;
            }
            let shift = sine.atan2(cosine) / kappa;
            // A perturbation of the resultant by its band turns its direction by
            // at most band/length, so a shift inside that angle is numerical zero.
            if shift.abs() <= (resultant_band / resultant) / kappa {
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
            // A prior term costs at most eight roundings to form (the shift, `κ`,
            // `κ·t`, the half-angle sine and its square, `κ²`, `α/κ²` and the
            // product). A compensated sum's band carries no row count, and the
            // subtraction rounds once more.
            let resolution =
                gam_linalg::roundoff::compensated_band(9, old_energy.abs() + new_energy.abs());
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
            let kappa_derivative = self.atoms[atom_index]
                .smooth_penalty_kappa_derivative()?
                .map(|derivative| fast_ab(&transport, derivative).dot(&transport.t()));
            let basis_values = self.atoms[atom_index].basis_values.clone();
            let basis_jacobian = self.atoms[atom_index].basis_jacobian.clone();
            self.atoms[atom_index].install_reparameterized_basis(
                basis_values,
                basis_jacobian,
                decoder,
                smooth_penalty,
                kappa_derivative,
            )?;
            for (row, slot, coordinate) in shifted_coordinates {
                self.assignment
                    .set_slot_coords(row, slot, &[coordinate])?;
            }
            profiled += 1;
        }
        Ok(profiled)
    }

    /// #2576 — profile the affine gauge of every one-dimensional `Linear` atom.
    ///
    /// A linear atom `f(t) = β₀ + c·t` represents the same function under
    /// `t = a + b·t'` with `β₀' = β₀ + a·c` and `c' = b·c`, so the data fit is exactly
    /// invariant along that two-parameter orbit and only the priors curve it: the
    /// coordinate ARD energy `½α Σ t²` and the atom's slope ridge `½λ s‖c‖²` (its
    /// function penalty, `polynomial_reference_penalty`). Coordinate sweeps crawl
    /// along such a weakly curved orbit, and the recurrence distance — which removes
    /// only periodic phase gauges — counts the crawl as state change, so the fixed
    /// point cannot recur while it lasts.
    ///
    /// Along the orbit that prior energy is `½α Σ (t − a)²/b² + ½λ s b² ‖c‖²`,
    /// minimised by the mean `a*` of the routed coordinates and `b*⁴ = α S / (λ s ‖c‖²)`
    /// with `S = Σ (t − a*)²`. The candidate is installed only when the exact prior
    /// energy of the atom — its ARD terms and its smoothing quadratic form, both
    /// re-evaluated — decreases by more than a roundoff bound, so a profiled atom is
    /// left where it is.
    fn profile_linear_affine_gauges(
        &mut self,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<usize, String> {
        let mut profiled = 0usize;
        for atom_index in 0..self.k_atoms() {
            if self.atoms[atom_index].basis_kind() != &SaeAtomBasisKind::Linear
                || self.assignment.atom_coord_dim(atom_index) != 1
                || self.atom_rows[atom_index].is_empty()
                || self.atoms[atom_index].basis_size() != 2
                || self.atoms[atom_index]
                    .basis_values
                    .column(0)
                    .iter()
                    .any(|&value| value != 1.0)
            {
                continue;
            }
            let alpha = ard_precisions[atom_index][0];
            let lambda = lambda_smooth[atom_index];
            if !(alpha.is_finite() && alpha > 0.0 && lambda.is_finite() && lambda >= 0.0) {
                continue;
            }
            let rows = self.atom_rows[atom_index].len();
            let mut coordinate_sum = KahanSum::default();
            for &(row, slot) in &self.atom_rows[atom_index] {
                coordinate_sum.add(self.assignment.coords_for_slot(row, slot)[0]);
            }
            let shift = coordinate_sum.sum() / rows as f64;
            let mut spread = KahanSum::default();
            for &(row, slot) in &self.atom_rows[atom_index] {
                let centered = self.assignment.coords_for_slot(row, slot)[0] - shift;
                spread.add(centered * centered);
            }
            let spread = spread.sum();
            let slope_ridge = self.atoms[atom_index].smooth_penalty()[[1, 1]];
            let slope_energy = self.atoms[atom_index]
                .decoder_coefficients()
                .row(1)
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            let curvature = lambda * slope_ridge * slope_energy;
            let scale = if spread > 0.0 && curvature.is_finite() && curvature > 0.0 {
                (alpha * spread / curvature).sqrt().sqrt()
            } else {
                1.0
            };
            if !(scale.is_finite() && scale > 0.0) {
                continue;
            }

            let old_decoder = self.atoms[atom_index].decoder_coefficients().clone();
            let mut new_decoder = old_decoder.clone();
            for output in 0..new_decoder.ncols() {
                let slope = old_decoder[[1, output]];
                new_decoder[[0, output]] += shift * slope;
                new_decoder[[1, output]] = scale * slope;
            }
            let penalty = self.atoms[atom_index].smooth_penalty().clone();
            let smoothing_energy = |decoder: &Array2<f64>| -> f64 {
                let penalized = penalty.dot(decoder);
                0.5 * lambda
                    * decoder
                        .iter()
                        .zip(penalized.iter())
                        .map(|(left, right)| left * right)
                        .sum::<f64>()
            };
            let mut old_energy = KahanSum::default();
            let mut new_energy = KahanSum::default();
            let mut profiled_coordinates = Vec::with_capacity(rows);
            for &(row, slot) in &self.atom_rows[atom_index] {
                let coordinate = self.assignment.coords_for_slot(row, slot)[0];
                let reparameterized = (coordinate - shift) / scale;
                old_energy.add(ArdAxisPrior::eval(alpha, coordinate, None).value);
                new_energy.add(ArdAxisPrior::eval(alpha, reparameterized, None).value);
                profiled_coordinates.push((row, slot, reparameterized));
            }
            old_energy.add(smoothing_energy(&old_decoder));
            new_energy.add(smoothing_energy(&new_decoder));
            let old_energy = old_energy.sum();
            let new_energy = new_energy.sum();
            let resolution =
                gam_linalg::roundoff::compensated_band(9, old_energy.abs() + new_energy.abs());
            if !(old_energy - new_energy > resolution) {
                continue;
            }

            let basis_values = self.atoms[atom_index].basis_values.clone();
            let basis_jacobian = self.atoms[atom_index].basis_jacobian.clone();
            // The Gram is kept, so its ∂S/∂κ is kept with it.
            let kappa_derivative = self.atoms[atom_index]
                .smooth_penalty_kappa_derivative()?
                .cloned();
            self.atoms[atom_index].install_reparameterized_basis(
                basis_values,
                basis_jacobian,
                new_decoder,
                penalty,
                kappa_derivative,
            )?;
            for (row, slot, coordinate) in profiled_coordinates {
                self.assignment
                    .set_slot_coords(row, slot, &[coordinate])?;
            }
            profiled += 1;
        }
        Ok(profiled)
    }

    /// #2576 — profile the affine gauge of every one-dimensional degree-2
    /// `EuclideanPatch` atom, the patch counterpart of
    /// [`Self::profile_linear_affine_gauges`].
    ///
    /// A patch `f(t) = β₀ + β₁t + β₂t²` represents the same function under
    /// `t = a + b·t'` with `β₀' = β₀ + aβ₁ + a²β₂`, `β₁' = b(β₁ + 2aβ₂)` and
    /// `β₂' = b²β₂`, so the data fit is invariant along that orbit and only the
    /// coordinate ARD energy and the patch's Dirichlet energy over its fixed
    /// reference rows (the Gram `G`, zero on the constant) curve it.
    ///
    /// The translation is exact: at `b = 1` the orbit energy is quadratic in `a`, so
    /// `a* = (α Σt − 2λ Σ_o (g₁₁β₁β₂ + g₁₂β₂²)) / (αM + 4λ g₁₁ Σ_o β₂²)`. The scale
    /// then minimises `½αS u⁻² + ½λ(c₂u² + 2c₃u³ + c₄u⁴)` by Newton in `ln u` from
    /// `u = 1`, taking only strictly decreasing steps while the log-curvature is
    /// positive; the energy is coercive, so the minimiser is finite, exactly when the
    /// routed coordinates spread (`S > 0`) and the penalty bends the orbit
    /// (`c₂ > 0` or `c₄ > 0`). As for linear atoms, the candidate is installed only
    /// when the atom's exact prior energy decreases by more than a roundoff bound.
    fn profile_euclidean_patch_affine_gauges(
        &mut self,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<usize, String> {
        let mut profiled = 0usize;
        for atom_index in 0..self.k_atoms() {
            if self.atoms[atom_index].basis_kind() != &SaeAtomBasisKind::EuclideanPatch
                || self.assignment.atom_coord_dim(atom_index) != 1
                || self.atom_rows[atom_index].is_empty()
                || self.atoms[atom_index].basis_size() != 3
                || self.atoms[atom_index]
                    .basis_values
                    .column(0)
                    .iter()
                    .any(|&value| value != 1.0)
            {
                continue;
            }
            let alpha = ard_precisions[atom_index][0];
            let lambda = lambda_smooth[atom_index];
            if !(alpha.is_finite() && alpha > 0.0 && lambda.is_finite() && lambda >= 0.0) {
                continue;
            }
            let penalty = self.atoms[atom_index].smooth_penalty().clone();
            if (0..3).any(|index| penalty[[0, index]] != 0.0 || penalty[[index, 0]] != 0.0) {
                continue;
            }
            let (g11, g12, g22) = (penalty[[1, 1]], penalty[[1, 2]], penalty[[2, 2]]);
            let old_decoder = self.atoms[atom_index].decoder_coefficients().clone();
            let rows = self.atom_rows[atom_index].len();
            let mut coordinate_sum = KahanSum::default();
            for &(row, slot) in &self.atom_rows[atom_index] {
                coordinate_sum.add(self.assignment.coords_for_slot(row, slot)[0]);
            }
            let (mut q12, mut q22) = (0.0_f64, 0.0_f64);
            for output in 0..old_decoder.ncols() {
                q12 += old_decoder[[1, output]] * old_decoder[[2, output]];
                q22 += old_decoder[[2, output]] * old_decoder[[2, output]];
            }
            let shift = (alpha * coordinate_sum.sum() - 2.0 * lambda * (g11 * q12 + g12 * q22))
                / (alpha * rows as f64 + 4.0 * lambda * g11 * q22);
            if !shift.is_finite() {
                continue;
            }
            let mut spread = KahanSum::default();
            for &(row, slot) in &self.atom_rows[atom_index] {
                let centered = self.assignment.coords_for_slot(row, slot)[0] - shift;
                spread.add(centered * centered);
            }
            let spread = spread.sum();
            let (mut slope_energy, mut cross_energy, mut bend_energy) = (0.0_f64, 0.0_f64, 0.0_f64);
            for output in 0..old_decoder.ncols() {
                let slope = old_decoder[[1, output]] + 2.0 * shift * old_decoder[[2, output]];
                let bend = old_decoder[[2, output]];
                slope_energy += slope * slope;
                cross_energy += slope * bend;
                bend_energy += bend * bend;
            }
            let (c2, c3, c4) = (g11 * slope_energy, g12 * cross_energy, g22 * bend_energy);
            let mut log_scale = 0.0_f64;
            if spread > 0.0 && (c2 > 0.0 || c4 > 0.0) {
                let orbit_energy = |u: f64| {
                    0.5 * alpha * spread / (u * u)
                        + 0.5 * lambda * (c2 * u * u + 2.0 * c3 * u * u * u + c4 * u * u * u * u)
                };
                let mut energy = orbit_energy(1.0);
                loop {
                    let u = log_scale.exp();
                    let gradient = u
                        * (-alpha * spread / (u * u * u)
                            + lambda * (c2 * u + 3.0 * c3 * u * u + 2.0 * c4 * u * u * u));
                    let curvature = gradient
                        + u * u
                            * (3.0 * alpha * spread / (u * u * u * u)
                                + lambda * (c2 + 6.0 * c3 * u + 6.0 * c4 * u * u));
                    if !(gradient.is_finite() && curvature.is_finite() && curvature > 0.0) {
                        break;
                    }
                    let mut step = -gradient / curvature;
                    let mut accepted = false;
                    while step.abs() > f64::EPSILON {
                        let candidate = orbit_energy((log_scale + step).exp());
                        if candidate < energy {
                            log_scale += step;
                            energy = candidate;
                            accepted = true;
                            break;
                        }
                        step *= 0.5;
                    }
                    if !accepted {
                        break;
                    }
                }
            }
            let scale = log_scale.exp();
            if !(scale.is_finite() && scale > 0.0) {
                continue;
            }

            let mut new_decoder = old_decoder.clone();
            for output in 0..new_decoder.ncols() {
                let constant = old_decoder[[0, output]];
                let linear = old_decoder[[1, output]];
                let quadratic = old_decoder[[2, output]];
                new_decoder[[0, output]] = constant + shift * linear + shift * shift * quadratic;
                new_decoder[[1, output]] = scale * (linear + 2.0 * shift * quadratic);
                new_decoder[[2, output]] = scale * scale * quadratic;
            }
            let smoothing_energy = |decoder: &Array2<f64>| -> f64 {
                let penalized = penalty.dot(decoder);
                0.5 * lambda
                    * decoder
                        .iter()
                        .zip(penalized.iter())
                        .map(|(left, right)| left * right)
                        .sum::<f64>()
            };
            let mut old_energy = KahanSum::default();
            let mut new_energy = KahanSum::default();
            let mut profiled_coordinates = Vec::with_capacity(rows);
            for &(row, slot) in &self.atom_rows[atom_index] {
                let coordinate = self.assignment.coords_for_slot(row, slot)[0];
                let reparameterized = (coordinate - shift) / scale;
                old_energy.add(ArdAxisPrior::eval(alpha, coordinate, None).value);
                new_energy.add(ArdAxisPrior::eval(alpha, reparameterized, None).value);
                profiled_coordinates.push((row, slot, reparameterized));
            }
            old_energy.add(smoothing_energy(&old_decoder));
            new_energy.add(smoothing_energy(&new_decoder));
            let old_energy = old_energy.sum();
            let new_energy = new_energy.sum();
            let resolution =
                gam_linalg::roundoff::compensated_band(9, old_energy.abs() + new_energy.abs());
            if !(old_energy - new_energy > resolution) {
                continue;
            }

            let basis_values = self.atoms[atom_index].basis_values.clone();
            let basis_jacobian = self.atoms[atom_index].basis_jacobian.clone();
            // The Gram is kept, so its ∂S/∂κ is kept with it.
            let kappa_derivative = self.atoms[atom_index]
                .smooth_penalty_kappa_derivative()?
                .cloned();
            self.atoms[atom_index].install_reparameterized_basis(
                basis_values,
                basis_jacobian,
                new_decoder,
                penalty,
                kappa_derivative,
            )?;
            for (row, slot, coordinate) in profiled_coordinates {
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
            self.atom_ard_axis_periods[atom_index] = self.atoms[atom_index]
                .basis_kind()
                .ard_axis_periods(&self.atom_axis_periods[atom_index]);
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
                    let period = self.atom_ard_axis_periods(atom_index)[0];
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
                .map(|template| SaeAssignmentAtomSpec {
                    latent_dim: template.latent_dim(),
                    manifold: template.basis_kind().latent_manifold(template.latent_dim()),
                    retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
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
            .map(|template| SaeAssignmentAtomSpec {
                latent_dim: template.latent_dim(),
                manifold: template.basis_kind().latent_manifold(template.latent_dim()),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
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
                .flat_map(|&atom| self.atom_ard_axis_periods(atom as usize).iter().copied())
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
        // #2576: every hot path (the reduced-Schur matvec, the Jacobi diagonal and
        // blocks) routes through `penalty_op`. Installing the operator itself gives
        // them its exact blocks; the closure adapter above could only probe
        // columns. `hbb_matvec`/`hbb_diag` stay for the helpers that read them.
        system.set_penalty_op(Arc::clone(&operator) as Arc<dyn gam_solve::arrow_schur::BetaPenaltyOp>);
        let forward = Arc::clone(&operator);
        let transpose = Arc::clone(&operator);
        system.set_row_htbeta_operator(
            move |row, vector, out| forward.htbeta_forward(row, vector, out),
            move |row, vector, out| transpose.htbeta_transpose(row, vector, out),
            operator.htbeta_declaration(),
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
        // Rows are independent; the per-row work (an active-slot fill, a
        // second-jet evaluation and a handful of small allocations per slot)
        // fans across the pool exactly as `raw_stationarity_with_residual`
        // does. Serial, the 3000-row chart of #2576 spent seconds here on
        // every classifier call.
        (0..self.n_obs())
            .into_par_iter()
            .map_init(
                ActiveAtomScratch::default,
                |scratch, row| -> Result<SupportOuterDifferentialRow, String> {
                    self.support_outer_differential_row(
                        target,
                        ard_precisions,
                        beta_offsets,
                        row,
                        scratch,
                    )
                },
            )
            .collect::<Result<Vec<_>, String>>()
    }

    /// One active slot's analytic second jet `∂²Φ/∂t∂t` at `coordinates`, shape
    /// `(1, m, d, d)`: from the atom's dedicated second-jet evaluator, else its basis
    /// evaluator's dynamic hook. `None` when the basis exposes neither. Every consumer of
    /// a slot's exact curvature reads this one seam, so no two of them can disagree about
    /// it.
    fn slot_second_jet(
        &self,
        atom_index: usize,
        coordinates: ArrayView2<'_, f64>,
    ) -> Result<Option<ndarray::Array4<f64>>, String> {
        let atom = &self.atoms[atom_index];
        if let Some(evaluator) = atom.basis_second_jet.as_ref() {
            return evaluator.second_jet(coordinates).map(Some);
        }
        match atom
            .basis_evaluator
            .as_ref()
            .and_then(|evaluator| evaluator.second_jet_dyn(coordinates))
        {
            Some(second) => second.map(Some),
            None => Ok(None),
        }
    }

    fn support_outer_differential_row(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        beta_offsets: &[usize],
        row: usize,
        scratch: &mut ActiveAtomScratch,
    ) -> Result<SupportOuterDifferentialRow, String> {
        {
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
                self.fill_active(row, slot, scratch)?;
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
                let second = self
                    .slot_second_jet(atom_index, coordinate_view)?
                    .ok_or_else(|| {
                        format!(
                            "support outer differential: atom {atom_index} ('{}') does not expose an analytic second jet",
                            atom.name
                        )
                    })?;
                if second.dim() != (1, m, d, d) {
                    return Err(format!(
                        "support outer differential: row {row}, atom {atom_index} second jet shape {:?} != (1, {m}, {d}, {d})",
                        second.dim(),
                    ));
                }

                let periods = self.atom_ard_axis_periods(atom_index);
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
            Ok(SupportOuterDifferentialRow {
                slots,
                jacobian,
                residual: &target.row(row) - &fitted,
                prior_hessian_remainder,
                prior_majorizer_derivative,
            })
        }
    }

    /// `Σ_m Σ_p ∂²φ_m/∂t_a∂t_b · B_mp · r_p` for one slot of one row: the second
    /// derivative of the slot's decoder output contracted with the row residual.
    /// The exact coordinate block carries it with a minus sign.
    fn support_outer_residual_second_derivative(
        &self,
        row: &SupportOuterDifferentialRow,
        slot: &SupportOuterDifferentialSlot,
        axis_a: usize,
        axis_b: usize,
    ) -> f64 {
        let atom = &self.atoms[slot.atom];
        let decoder = atom.decoder_coefficients();
        let mut residual_second = 0.0_f64;
        for basis in 0..atom.basis_size() {
            let coefficient = slot.second_jet[[basis, axis_a, axis_b]];
            for output in 0..self.output_dim {
                residual_second += coefficient * decoder[[basis, output]] * row.residual[output];
            }
        }
        residual_second
    }

    /// Row factors of the preconditioner for the solves against the exact `A`: the
    /// Newton displacement (#2933 F08) and the reduced-logdet adjoint. Each row block
    /// is the majorizer block `B_i` plus the positive part of the exact block's excess
    /// over it, `B_i + (A_i − B_i)_+` ([`Self::support_exact_a_preconditioner_rows`]).
    fn support_exact_a_preconditioner(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
    ) -> Result<ArrowFactorSlab, String> {
        let preconditioner_rows = self.support_exact_a_preconditioner_rows(system, rows)?;
        CpuBatchedBlockSolver
            .factor_blocks(&preconditioner_rows, 0.0, system.d, true)
            .map_err(|error| format!("support exact-A preconditioner row factorization: {error}"))
    }

    /// `B_i + (A_i − B_i)_+` for every row block.
    ///
    /// `B` majorizes the prior's signed curvature, not the residual curvature
    /// `−Σ_p r_p ∂²f_p` that `A` adds, which is positive wherever the decoder curves
    /// toward the row's residual. Where a coordinate's decoder tangent vanishes and
    /// its periodic prior sits at the antipode, `B_i` keeps no curvature while `A_i`
    /// does. On the two-circle Tier-2 witness one row had `B_i = 5.8e-14` against
    /// `A_i = 12.4` (lane probe 1249076), so `B⁻¹` amplified that coordinate by
    /// 7.65e13 and flexible GMRES could not represent the gradient in the span it
    /// built. `B_i + (A_i − B_i)_+` majorizes `A_i`, equals `B_i` wherever `B_i`
    /// already does, and is positive definite wherever `A_i` has positive curvature.
    /// It changes how fast the solve converges, never what it certifies: flexible
    /// GMRES certifies the physical residual `‖rhs − AΔ‖`.
    fn support_exact_a_preconditioner_rows(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
    ) -> Result<Vec<ArrowRowBlock>, String> {
        if rows.len() != system.rows.len() {
            return Err(format!(
                "support Newton preconditioner: {} differential rows for {} system rows",
                rows.len(),
                system.rows.len()
            ));
        }
        system
            .rows
            .iter()
            .zip(rows)
            .map(|(block, row)| {
                let q = block.htt.nrows();
                let mut excess = Array2::<f64>::zeros((q, q));
                for slot in &row.slots {
                    let offset = slot.coordinate_offset;
                    let d = self.atoms[slot.atom].latent_dim();
                    for axis_a in 0..d {
                        for axis_b in 0..d {
                            excess[[offset + axis_a, offset + axis_b]] -= self
                                .support_outer_residual_second_derivative(row, slot, axis_a, axis_b);
                        }
                        excess[[offset + axis_a, offset + axis_a]] +=
                            row.prior_hessian_remainder[offset + axis_a];
                    }
                }
                let symmetric = (&excess + &excess.t()) * 0.5;
                let (values, vectors) = symmetric.eigh(Side::Lower).map_err(|error| {
                    format!("support Newton preconditioner: row excess eigendecomposition: {error}")
                })?;
                let mut htt = block.htt.clone();
                for (index, &value) in values.iter().enumerate() {
                    if value > 0.0 {
                        let direction = vectors.column(index);
                        for left in 0..q {
                            for right in 0..q {
                                htt[[left, right]] += value * direction[left] * direction[right];
                            }
                        }
                    }
                }
                Ok(ArrowRowBlock {
                    htt,
                    htbeta: Array2::<f64>::zeros((q, 0)),
                    gt: Array1::<f64>::zeros(q),
                })
            })
            .collect()
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
                        let residual_second = self.support_outer_residual_second_derivative(
                            row, slot, axis_a, axis_b,
                        );
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

    /// The symmetrized dense exact Hessian `A` and majorizer `B` of the support
    /// stationarity pencil. Every dense consumer reads the pencil through this seam.
    ///
    /// #2576: assembled from the arrow's blocks, not probed column by column. A column
    /// probe applies both operators to a unit vector, so each of the `dim` columns paid a
    /// full pass over every row and over the whole `H_ββ` operator, twice. At 3000x48
    /// (`dim` 19680) job 1190710's stack samples of its 12-14 min terminal certificates sat
    /// in that probe loop. Here each entry is the one term its probe reads,
    /// accumulated in the order the probe's sums take it:
    /// - `H_ββ` from the installed penalty operator's `to_dense`, the operator
    ///   `assemble_arrow_schur` installs in lock-step with `hbb_matvec`;
    /// - each row's `H_tt`, and its `H_tβ` row from the same transpose apply the probe of
    ///   that coordinate reads;
    /// - the exact residual corrections of `support_outer_exact_hessian_apply`, row-local.
    /// The matrices are the probe's bit for bit
    /// (`dense_pencil_assembly_is_its_column_probes_2576`).
    fn support_outer_dense_hessian_matrices(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
        t_len: usize,
        beta_len: usize,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let dim = t_len
            .checked_add(beta_len)
            .ok_or_else(|| "support outer Hessian-pencil dimension overflow".to_string())?;
        if dim == 0 {
            return Err("support outer Hessian pencil has zero dimension".to_string());
        }
        if rows.len() != system.rows.len()
            || *system.row_offsets.last().unwrap_or(&0) != t_len
            || system.k != beta_len
        {
            return Err(format!(
                "support outer Hessian pencil: {} differential rows, coordinate length {t_len} and \
                 beta length {beta_len} do not describe the arrow system ({} rows, {}, {})",
                rows.len(),
                system.rows.len(),
                system.row_offsets.last().unwrap_or(&0),
                system.k,
            ));
        }
        let hbb = match (system.hbb_matvec.as_ref(), system.penalty_op.as_ref()) {
            (Some(_), Some(operator)) => operator.to_dense(),
            (Some(_), None) => {
                return Err(
                    "support outer Hessian pencil: H_betabeta operator installed without its \
                     penalty operator"
                        .to_string(),
                );
            }
            (None, _) if system.hbb.dim() == (beta_len, beta_len) => system.hbb.clone(),
            (None, _) => {
                return Err(format!(
                    "support outer Hessian pencil: H_betabeta shape {:?} != ({beta_len}, \
                     {beta_len}) and no operator is installed",
                    system.hbb.dim(),
                ));
            }
        };
        if hbb.dim() != (beta_len, beta_len) {
            return Err(format!(
                "support outer Hessian pencil: dense H_betabeta shape {:?} != ({beta_len}, \
                 {beta_len})",
                hbb.dim(),
            ));
        }
        let mut majorizer = Array2::<f64>::zeros((dim, dim));
        majorizer.slice_mut(ndarray::s![t_len.., t_len..]).assign(&hbb);
        drop(hbb);
        let mut cross = Array1::<f64>::zeros(beta_len);
        for row in 0..system.rows.len() {
            let start = system.row_offsets[row];
            let q = system.row_offsets[row + 1] - start;
            let htt = &system.rows[row].htt;
            if htt.dim() != (q, q) {
                return Err(format!(
                    "support outer Hessian pencil: row {row} H_tt shape {:?} != ({q}, {q})",
                    htt.dim(),
                ));
            }
            majorizer
                .slice_mut(ndarray::s![start..start + q, start..start + q])
                .assign(htt);
            let mut unit = Array1::<f64>::zeros(q);
            for axis in 0..q {
                unit[axis] = 1.0;
                cross.fill(0.0);
                support_arrow_cross_transpose_add(system, row, unit.view(), &mut cross)?;
                unit[axis] = 0.0;
                majorizer
                    .slice_mut(ndarray::s![start + axis, t_len..])
                    .assign(&cross);
                majorizer
                    .slice_mut(ndarray::s![t_len.., start + axis])
                    .assign(&cross);
            }
        }
        let mut exact = majorizer.clone();
        for (row_index, row) in rows.iter().enumerate() {
            let row_start = system.row_offsets[row_index];
            for slot in &row.slots {
                let atom = &self.atoms[slot.atom];
                let d = atom.latent_dim();
                let m = atom.basis_size();
                for axis_a in 0..d {
                    let t_a = row_start + slot.coordinate_offset + axis_a;
                    // Exact residual curvature `-sum_p r_p d2f_p/dt_a dt_b`, and the
                    // prior's Hessian remainder on the diagonal.
                    for axis_b in 0..d {
                        let residual_second = self.support_outer_residual_second_derivative(
                            row, slot, axis_a, axis_b,
                        );
                        let mut correction = -residual_second;
                        if axis_a == axis_b {
                            correction +=
                                row.prior_hessian_remainder[slot.coordinate_offset + axis_a];
                        }
                        exact[[t_a, row_start + slot.coordinate_offset + axis_b]] += correction;
                    }
                    // Exact coordinate-decoder residual cross block, both triangles.
                    for basis in 0..m {
                        let derivative = slot.jet[[basis, axis_a]];
                        for output in 0..self.output_dim {
                            let beta_index =
                                t_len + slot.beta_offset + basis * self.output_dim + output;
                            exact[[t_a, beta_index]] -= derivative * row.residual[output];
                            exact[[beta_index, t_a]] -= row.residual[output] * derivative;
                        }
                    }
                }
            }
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
        Ok((exact, majorizer))
    }

    /// The generalized eigensystem of the dense pencil `(A, B)` from its
    /// symmetrized matrices ([`Self::support_outer_dense_hessian_matrices`]).
    fn support_outer_dense_hessian_pencil(
        exact: Array2<f64>,
        majorizer: Array2<f64>,
    ) -> Result<SupportOuterDensePencil, String> {
        let dim = exact.nrows();
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
        // whitening would fail that certificate by construction. The pencil
        // keeps that `B = V·diag(s)·Vᵀ` as its eigenvectors `V` and the stiffness
        // `s` the whitener used, and never forms it: the pseudoinverse only
        // applies `B` to a vector, and the negative-mode classifier does not read
        // it (#2634). Forming it was a scalar `dim³` loop, 1716 s of a single core
        // at dim 7392.
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
            .fold(0.0_f64, |current, &value| current.max(value.abs()));
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
            majorizer_eigenvectors: b_eigenvectors,
            majorizer_stiffness: Array1::from(stiffened),
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
        let (exact, majorizer) =
            self.support_outer_dense_hessian_matrices(system, rows, t_len, beta_len)?;
        // `A + (floor/2)·sym(B) ≻ 0` implies `A + (floor/2)·B_stiff ≻ 0`, because the
        // deflation that forms `B_stiff` only raises eigenvalues of `sym(B)`. Then
        // every generalized curvature exceeds `−floor/2 > −resolution`, there is no
        // mode, and neither eigensystem is needed (#2634: they were 192 s of each
        // 245 s admission at dim 7392). A refusal decides nothing.
        //
        // An index whose row is exactly zero in both symmetrized matrices is a common
        // null direction: the pencil splits as `(A_r ⊕ 0, B_r ⊕ 0)`, the deflation
        // gives that direction unit stiffness, and its curvature is exactly zero,
        // never below `−resolution`. It would make `A + (floor/2)·sym(B)` singular, so
        // the certificate is taken on the complement, where `B_r,stiff ⪰ sym(B_r)`
        // still holds. Only bitwise-zero rows qualify, never merely small ones.
        let certificate_shift = -0.5 * support_outer_curvature_floor();
        let common_null: Vec<bool> = (0..exact.nrows())
            .map(|index| {
                exact.row(index).iter().all(|value| *value == 0.0)
                    && majorizer.row(index).iter().all(|value| *value == 0.0)
            })
            .collect();
        let certificate = if common_null.iter().any(|&null| null) {
            let kept: Vec<usize> = (0..exact.nrows()).filter(|&index| !common_null[index]).collect();
            let reduced_exact = exact
                .select(ndarray::Axis(0), &kept)
                .select(ndarray::Axis(1), &kept);
            let reduced_majorizer = majorizer
                .select(ndarray::Axis(0), &kept)
                .select(ndarray::Axis(1), &kept);
            certify_shifted_pd(
                reduced_exact.view(),
                reduced_majorizer.view(),
                certificate_shift,
            )
        } else {
            certify_shifted_pd(exact.view(), majorizer.view(), certificate_shift)
        };
        match certificate {
            Ok(certificate) => {
                let BandProvenance::Dense { scaled_frobenius } = certificate.band;
                log::debug!(
                    "support saddle classifier certified A − τ·B ≻ 0 without an eigensystem: \
                     dim {}, τ {:.3e}, band {:.3e}, ‖S₀‖_F {:.3e}",
                    certificate.dim,
                    certificate.tau,
                    certificate.delta,
                    scaled_frobenius
                );
                return Ok(None);
            }
            Err(refusal) => {
                let reason = match refusal {
                    ShiftedPdRefusal::ShapeMismatch { a, b } => {
                        format!("shapes {a:?} and {b:?} disagree")
                    }
                    ShiftedPdRefusal::NonFinite => "the shifted matrix is not finite".to_string(),
                    ShiftedPdRefusal::NonPositiveDiagonal { index, value } => {
                        format!("diagonal entry {index} is {value:.3e}")
                    }
                    ShiftedPdRefusal::CholeskyFailed { delta, pivot } => {
                        format!("the factorization stopped at pivot {pivot:?} under band {delta:.3e}")
                    }
                };
                log::debug!(
                    "support saddle classifier could not certify positive curvature ({reason}); \
                     solving the generalized eigensystem"
                );
            }
        }
        let pencil = Self::support_outer_dense_hessian_pencil(exact, majorizer)?;
        let curvature = pencil.curvatures[0];
        let resolution = pencil
            .minimum_backward_error
            .max(support_outer_curvature_floor());
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
    ///
    /// One pencil serves every right-hand side in `rhs`: the eigensystem is the
    /// expensive half, and the per-probe responses of #2933 F29 need one solve per
    /// smoothing group against the same operator.
    fn support_outer_dense_pseudoinverse_apply(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
        rhs: &[SaeArrowVector],
    ) -> Result<Vec<SaeArrowVector>, String> {
        let Some(first) = rhs.first() else {
            return Ok(Vec::new());
        };
        let t_len = first.t.len();
        let beta_len = first.beta.len();
        if rhs
            .iter()
            .any(|vector| vector.t.len() != t_len || vector.beta.len() != beta_len)
        {
            return Err(
                "support outer adjoint right-hand sides disagree in block shape".to_string(),
            );
        }
        let dim = t_len
            .checked_add(beta_len)
            .ok_or_else(|| "support outer adjoint dimension overflow".to_string())?;
        if dim == 0 {
            return Ok(rhs
                .iter()
                .map(|_| SaeArrowVector {
                    t: Array1::zeros(0),
                    beta: Array1::zeros(0),
                })
                .collect());
        }
        let (exact, majorizer) =
            self.support_outer_dense_hessian_matrices(system, rows, t_len, beta_len)?;
        let pencil = Self::support_outer_dense_hessian_pencil(exact, majorizer)?;
        let quotient_floor = support_outer_curvature_floor().max(pencil.minimum_backward_error);
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
        let exact_scale = pencil
            .exact
            .iter()
            .fold(0.0_f64, |current, &value| current.max(value.abs()));
        let mut solutions = Vec::with_capacity(rhs.len());
        for rhs in rhs {
            let mut flat_rhs = Array1::<f64>::zeros(dim);
            flat_rhs.slice_mut(ndarray::s![..t_len]).assign(&rhs.t);
            flat_rhs
                .slice_mut(ndarray::s![t_len..])
                .assign(&rhs.beta);
            let mut solution = Array1::<f64>::zeros(dim);
            let mut range_direction = Array1::<f64>::zeros(dim);
            for mode in 0..dim {
                let curvature = pencil.curvatures[mode];
                if curvature <= quotient_floor {
                    continue;
                }
                let vector = pencil.generalized_vectors.column(mode);
                let projection = vector.dot(&flat_rhs);
                solution.scaled_add(projection / curvature, &vector);
                range_direction.scaled_add(projection, &vector);
            }
            // `Σ_j (v_jᵀb)·B v_j = B·Σ_j (v_jᵀb)·v_j`, so the range right-hand side is
            // one application of `B = V·diag(s)·Vᵀ`, `V·(s ⊙ (Vᵀ·w))`, per right-hand
            // side: `O(dim²)`, with `B` never formed.
            let mut majorizer_coordinates = gam_linalg::faer_ndarray::fast_atv(
                &pencil.majorizer_eigenvectors,
                &range_direction,
            );
            majorizer_coordinates
                .zip_mut_with(&pencil.majorizer_stiffness, |value, &stiffness| *value *= stiffness);
            let range_rhs =
                gam_linalg::faer_ndarray::fast_av(&pencil.majorizer_eigenvectors, &majorizer_coordinates);
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
            let solution_norm = solution.dot(&solution).sqrt();
            let backward_error_bound = dim.max(1) as f64
                * f64::EPSILON
                * (exact_scale * solution_norm + range_norm);
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
            solutions.push(SaeArrowVector {
                t: solution.slice(ndarray::s![..t_len]).to_owned(),
                beta: solution.slice(ndarray::s![t_len..]).to_owned(),
            });
        }
        Ok(solutions)
    }

    /// Return `A^+ Gamma`, the one adjoint needed for the implicit derivative of
    /// the Gauss--Newton arrow's log determinant
    /// `log|H| = Σ_i log|H_tt^(i)| + log|S|` (#2933 F27 S2). `Gamma` is the exact
    /// derivative, with respect to the fitted inner state, of the row blocks' log
    /// determinants and of the frozen rational surrogate of `log|S|`. The latter is
    /// assembled from the surrogate's own low-rank derivative vectors. `A` is the
    /// exact stationarity Jacobian of the penalized inner objective, not its
    /// Gauss--Newton majorizer.
    pub(crate) fn support_reduced_logdet_profile_adjoint(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        system: &ArrowSchurSystem,
        derivative: &RationalLogdetDerivativeBundle,
    ) -> Result<SaeArrowVector, String> {
        let derivative_vectors = derivative.vectors.as_slice();
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
        let inverse_rank = 1.0 / derivative_vectors.len() as f64;
        // #2576: each derivative vector contributes independently, so the vectors
        // fold in parallel over the length-only tree `schur_matvec` uses, and the
        // sum does not depend on thread count. On the 3000x48 chart the dense lane
        // hands this loop k = 7680 vectors. Serially it took about 125 s of one
        // outer evaluation (job 635403: the gap between the evidence line and the
        // adjoint FGMRES start).
        let accumulate = |range: core::ops::Range<usize>| -> Result<SaeArrowVector, String> {
            let mut gamma = SaeArrowVector {
                t: Array1::<f64>::zeros(coordinate_dim),
                beta: Array1::<f64>::zeros(beta_dim),
            };
            for border_vector in &derivative_vectors[range] {
                self.support_reduced_logdet_theta_derivative_add(
                    system,
                    &rows,
                    &factors,
                    border_vector.view(),
                    inverse_rank,
                    &mut gamma,
                )?;
            }
            Ok(gamma)
        };
        let mut gamma = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            derivative_vectors.len(),
            accumulate,
            |mut left: SaeArrowVector, right: SaeArrowVector| -> Result<SaeArrowVector, String> {
                left.t += &right.t;
                left.beta += &right.beta;
                Ok(left)
            },
        )?
        .ok_or_else(|| {
            "support reduced-logdet profile adjoint requires a non-empty derivative bundle"
                .to_string()
        })?;
        // The value integrates the row coordinate block as well (#2933 F27 S2), so
        // `Gamma` carries its derivative too, and the one adjoint prices the implicit
        // response of the whole `log|H|`.
        self.support_row_logdet_theta_derivative_add(system, &rows, &factors, &mut gamma)?;
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
        self.support_reduced_logdet_adjoint_solves(
            system,
            &rows,
            std::slice::from_ref(&gamma),
            derivative,
        )?
        .pop()
        .ok_or_else(|| {
            "support reduced-logdet profile adjoint solve returned no solution".to_string()
        })
    }

    /// Per-probe implicit responses of a surrogate `log|S|` derivative bundle
    /// (#2933 F29). Entry `[j, d]` is `w·⟨Σ_{z ∈ probe j} Γ(z), A⁺ directions[d]⟩`,
    /// with `w` the bundle's [`RationalLogdetDerivativeBundle::probe_sample_weight`]
    /// and `Γ(z)` one vector's inner-state derivative
    /// ([`Self::support_reduced_logdet_theta_derivative_add`]). `A⁺` is symmetric, so
    /// the mean over probes is the probe share of `⟨A⁺Γ, directions[d]⟩`, the implicit
    /// response [`Self::support_reduced_logdet_profile_adjoint`] feeds the gradient,
    /// and the spread over probes is that response's Hutchinson standard error. The
    /// cost is one adjoint per direction, not one per probe.
    pub(crate) fn support_reduced_logdet_probe_responses(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        system: &ArrowSchurSystem,
        bundle: &RationalLogdetDerivativeBundle,
        directions: &[SaeArrowVector],
    ) -> Result<Array2<f64>, String> {
        let probes = bundle.hutchinson_probe_count();
        if probes == 0 || directions.is_empty() {
            return Err(format!(
                "support reduced-logdet probe responses need Hutchinson probes and a \
                 direction; got {probes} probes and {} directions",
                directions.len()
            ));
        }
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        if beta_dim != system.k {
            return Err(format!(
                "support reduced-logdet probe responses beta layout {beta_dim} != system border {}",
                system.k
            ));
        }
        let rows = self.support_outer_differential_rows(target, ard_precisions, &beta_offsets)?;
        let factors = CpuBatchedBlockSolver
            .factor_blocks(&system.rows, 0.0, system.d, true)
            .map_err(|error| {
                format!("support reduced-logdet probe responses row factorization: {error}")
            })?;
        let coordinate_dim = *system.row_offsets.last().unwrap_or(&0);
        let adjoints = self.support_reduced_logdet_adjoint_solves(
            system,
            &rows,
            directions,
            bundle,
        )?;
        let weight = bundle.probe_sample_weight();
        let accumulate = |range: Range<usize>| -> Result<Vec<f64>, String> {
            let mut responses = Vec::with_capacity(range.len() * adjoints.len());
            let mut gamma = SaeArrowVector {
                t: Array1::<f64>::zeros(coordinate_dim),
                beta: Array1::<f64>::zeros(beta_dim),
            };
            for probe in range {
                gamma.t.fill(0.0);
                gamma.beta.fill(0.0);
                let vectors = bundle.probe_vectors(probe).ok_or_else(|| {
                    format!("support reduced-logdet probe responses: probe {probe} has no vectors")
                })?;
                for vector in vectors {
                    if vector.len() != system.k || vector.iter().any(|value| !value.is_finite()) {
                        return Err(format!(
                            "support reduced-logdet probe responses require finite vectors of \
                             border width {}",
                            system.k
                        ));
                    }
                    self.support_reduced_logdet_theta_derivative_add(
                        system,
                        &rows,
                        &factors,
                        vector.view(),
                        weight,
                        &mut gamma,
                    )?;
                }
                for adjoint in &adjoints {
                    responses.push(gamma.t.dot(&adjoint.t) + gamma.beta.dot(&adjoint.beta));
                }
            }
            Ok(responses)
        };
        let responses = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            probes,
            accumulate,
            |mut left: Vec<f64>, right: Vec<f64>| -> Result<Vec<f64>, String> {
                left.extend(right);
                Ok(left)
            },
        )?
        .ok_or_else(|| "support reduced-logdet probe responses folded no probe".to_string())?;
        if responses.iter().any(|value| !value.is_finite()) {
            return Err("support reduced-logdet probe responses are non-finite".to_string());
        }
        Array2::from_shape_vec((probes, adjoints.len()), responses)
            .map_err(|error| format!("support reduced-logdet probe responses shape: {error}"))
    }

    /// `A⁺ b` for every right-hand side `b` in `rhs`, with `A` the exact stationarity
    /// Jacobian of the penalized inner objective.
    ///
    /// A dense pseudoinverse is admitted only when assembling all `dim` analytic
    /// columns costs no more operator directions than the derivative bundle's
    /// vectors already paid for, and its complete eigensystem workspace fits the
    /// cgroup-aware in-core ledger. This is a scale transition derived from work and
    /// storage, not a dimension knob. Otherwise flexible GMRES runs, and where the
    /// bundle spans the reduced Schur's inverse exactly
    /// ([`RationalLogdetDerivativeBundle::exact_inverse_vectors`]) its preconditioner
    /// folds that inverse instead of running a CG per direction (#2576). Its row
    /// factors are the exact-A preconditioner's ([`Self::support_exact_a_preconditioner`]);
    /// with the folded border inverse they invert the arrow whose row blocks are
    /// `B_i + (A_i − B_i)_+` and whose reduced Schur is the one the evidence priced.
    fn support_reduced_logdet_adjoint_solves(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
        rhs: &[SaeArrowVector],
        derivative: &RationalLogdetDerivativeBundle,
    ) -> Result<Vec<SaeArrowVector>, String> {
        let derivative_vector_count = derivative.vectors.len();
        let coordinate_dim = *system.row_offsets.last().unwrap_or(&0);
        let full_dim = coordinate_dim
            .checked_add(system.k)
            .ok_or_else(|| "support reduced-logdet profile adjoint dimension overflow".to_string())?;
        let dense_workspace = (full_dim as u128)
            .saturating_mul(full_dim as u128)
            .saturating_mul(std::mem::size_of::<f64>() as u128)
            .saturating_mul(6);
        let in_core_budget = crate::manifold::sae_host_in_core_budget_bytes().0 as u128;
        if full_dim <= derivative_vector_count && dense_workspace <= in_core_budget {
            self.support_outer_dense_pseudoinverse_apply(system, rows, rhs)
        } else {
            let factors = self
                .support_exact_a_preconditioner(system, rows)
                .map_err(|error| format!("support reduced-logdet profile adjoint: {error}"))?;
            rhs.iter()
                .map(|rhs| {
                    solve_b_preconditioned_gmres_with(
                        rhs,
                        |vector| self.support_outer_exact_hessian_apply(system, rows, vector),
                        |residual| {
                            support_arrow_majorizer_inverse(
                                system,
                                &factors,
                                residual,
                                derivative.exact_inverse_vectors(),
                            )
                        },
                    )
                    .map_err(|error| {
                        format!("support reduced-logdet profile adjoint solve: {error}")
                    })
                })
                .collect()
        }
    }

    /// Add `Σ_i ∂log|H_tt^(i)|/∂θ` to `gamma` (#2933 F27 S2).
    ///
    /// Row `i`'s Gauss--Newton block is `H_tt = J Jᵀ + D`. Here `J_a = Σ_m ∂_aφ_m B_m`
    /// is the model's coordinate Jacobian and `D` the prior majorizer diagonal. With
    /// `M = H_tt⁻¹` and `G = M J`, `∂log|H_tt| = tr(M ∂H_tt) = 2⟨G, ∂J⟩ + Σ_a M_aa ∂D_aa`:
    /// - along a slot's coordinate `t_w`, `∂J_a = Σ_m ∂²_{aw}φ_m B_m` on the slot's
    ///   axes `a`, and `∂D_ww` is the majorizer derivative;
    /// - along a decoder entry `B_{m,o}` of the slot's atom, `∂J_{a,o} = ∂_aφ_m`.
    ///
    /// `H_tt` does not depend on the smoothing strengths, so this is the row
    /// normalizer's whole contribution to the gradient, through the fitted state.
    fn support_row_logdet_theta_derivative_add(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
        factors: &ArrowFactorSlab,
        gamma: &mut SaeArrowVector,
    ) -> Result<(), String> {
        for (row_index, row) in rows.iter().enumerate() {
            let row_start = system.row_offsets[row_index];
            let q = system.row_dims[row_index];
            if row.jacobian.nrows() != q {
                return Err(format!(
                    "support row log-det derivative: row {row_index} Jacobian spans {} \
                     coordinates but its block has {q}",
                    row.jacobian.nrows()
                ));
            }
            let inverse = CpuBatchedBlockSolver
                .solve_block_matrix(factors.factor(row_index), Array2::<f64>::eye(q).view());
            let weighted = inverse.dot(&row.jacobian);
            for slot in &row.slots {
                let atom = &self.atoms[slot.atom];
                let d = atom.latent_dim();
                let m = atom.basis_size();
                let decoder = atom.decoder_coefficients();
                for axis_w in 0..d {
                    let local_w = slot.coordinate_offset + axis_w;
                    let mut derivative =
                        inverse[[local_w, local_w]] * row.prior_majorizer_derivative[local_w];
                    for axis_a in 0..d {
                        let local_a = slot.coordinate_offset + axis_a;
                        for basis in 0..m {
                            let curvature = slot.second_jet[[basis, axis_a, axis_w]];
                            for output in 0..self.output_dim {
                                derivative += 2.0
                                    * weighted[[local_a, output]]
                                    * curvature
                                    * decoder[[basis, output]];
                            }
                        }
                    }
                    gamma.t[row_start + local_w] += derivative;
                }
                for basis in 0..m {
                    for output in 0..self.output_dim {
                        let mut derivative = 0.0_f64;
                        for axis_a in 0..d {
                            derivative += weighted[[slot.coordinate_offset + axis_a, output]]
                                * slot.jet[[basis, axis_a]];
                        }
                        gamma.beta[slot.beta_offset + basis * self.output_dim + output] +=
                            2.0 * derivative;
                    }
                }
            }
        }
        Ok(())
    }

    /// Add `weight·Γ(z)` to `gamma` for one border vector `z` of a `log|S|`
    /// derivative bundle. `Γ(z)` is the derivative, with respect to the fitted
    /// coordinates and decoder, of the Schur quadratic form along `z` with the row
    /// block eliminated, `z_t = −H_tt⁻¹ H_tβ z_β`. Summed over a bundle with weight
    /// `1/r`, it is the inner-state derivative of the surrogate the bundle represents.
    fn support_reduced_logdet_theta_derivative_add(
        &self,
        system: &ArrowSchurSystem,
        rows: &[SupportOuterDifferentialRow],
        factors: &ArrowFactorSlab,
        border_vector: ndarray::ArrayView1<'_, f64>,
        weight: f64,
        gamma: &mut SaeArrowVector,
    ) -> Result<(), String> {
        for (row_index, row) in rows.iter().enumerate() {
            let row_start = system.row_offsets[row_index];
            let q = system.row_dims[row_index];
            let cross = support_arrow_cross_forward(system, row_index, border_vector)?;
            let mut local_t =
                CpuBatchedBlockSolver.solve_block_vector(factors.factor(row_index), cross.view());
            local_t.mapv_inplace(|value| -value);

            // Directional model response `df[z] = J z_t + D z_beta` for
            // the Schur envelope vector `z_t = -H_tt^-1 H_tbeta z_beta`.
            let mut directional_fit = row.jacobian.t().dot(&local_t);
            for slot in &row.slots {
                let atom = &self.atoms[slot.atom];
                for basis in 0..atom.basis_size() {
                    let phi = slot.phi[basis];
                    for output in 0..self.output_dim {
                        directional_fit[output] += phi
                            * border_vector[slot.beta_offset + basis * self.output_dim + output];
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
                                derivative_fit[output] +=
                                    coefficient * atom.decoder_coefficients()[[basis, output]];
                            }
                        }
                    }
                    // Coordinate derivative of `D z_beta`.
                    for basis in 0..m {
                        let derivative = slot.jet[[basis, axis_w]];
                        for output in 0..self.output_dim {
                            derivative_fit[output] += derivative
                                * border_vector[slot.beta_offset + basis * self.output_dim + output];
                        }
                    }
                    let local_index = slot.coordinate_offset + axis_w;
                    gamma.t[row_start + local_index] += weight
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
                        jet_direction +=
                            slot.jet[[basis, axis]] * local_t[slot.coordinate_offset + axis];
                    }
                    for output in 0..self.output_dim {
                        gamma.beta[slot.beta_offset + basis * self.output_dim + output] +=
                            weight * 2.0 * directional_fit[output] * jet_direction;
                    }
                }
            }
            // `assert_eq!` rather than `debug_assert_eq!`: the scanner bans the
            // debug form, and a length invariant that only holds in debug builds
            // is not an invariant. Integer compare, negligible in release.
            assert_eq!(local_t.len(), q);
        }
        Ok(())
    }

    /// The rounding band `β_g` of the penalized objective's gradient as
    /// [`Self::assemble_arrow_schur`] accumulates it (#2933 F08): each component's
    /// `accumulation_band` over the terms that component sums, reduced to the
    /// Euclidean norm. A coordinate component sums `P` residual-weighted Jacobian cells
    /// and its prior gradient, the terms the row solve's skip band reads (#2469). A
    /// decoder component sums the `φ·r` cells of the atom's rows and the smoothing
    /// penalty's `λ·(S·B)` row. The residual and the Jacobian enter as computed.
    fn gradient_rounding_band(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        let residual = self.raw_residual(target)?;
        let decoder_sq = (0..self.k_atoms())
            .into_par_iter()
            .map_init(ActiveAtomScratch::default, |scratch, atom_idx| -> Result<f64, String> {
                let atom = &self.atoms[atom_idx];
                let m = atom.basis_size();
                let penalty = atom.smooth_penalty();
                let decoder = atom.decoder_coefficients();
                let mut data = Array2::<f64>::zeros((m, self.output_dim));
                for &(row, slot) in &self.atom_rows[atom_idx] {
                    self.fill_active(row, slot, scratch)?;
                    let phi = scratch.phi_row();
                    for basis in 0..m {
                        for output in 0..self.output_dim {
                            data[[basis, output]] += (phi[basis] * residual[[row, output]]).abs();
                        }
                    }
                }
                let terms = self.atom_rows[atom_idx].len() + m + 1;
                let mut sq = 0.0_f64;
                for basis in 0..m {
                    for output in 0..self.output_dim {
                        let penalty_sum = (0..m)
                            .map(|other| (penalty[[basis, other]] * decoder[[other, output]]).abs())
                            .sum::<f64>();
                        let band = gam_linalg::roundoff::accumulation_band(
                            terms,
                            data[[basis, output]] + lambda_smooth[atom_idx] * penalty_sum,
                        );
                        sq += band * band;
                    }
                }
                Ok(sq)
            })
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?;
        let coordinate_sq = (0..self.n_obs())
            .into_par_iter()
            .map_init(ActiveAtomScratch::default, |scratch, row| -> Result<f64, String> {
                let mut sq = 0.0_f64;
                for slot in 0..self.assignment.support_indices(row).len() {
                    let atom = self.assignment.support_indices(row)[slot] as usize;
                    self.fill_active(row, slot, scratch)?;
                    let periods = self.atom_ard_axis_periods(atom);
                    for axis in 0..scratch.jacobian.nrows() {
                        let data = scratch
                            .jacobian
                            .row(axis)
                            .iter()
                            .zip(residual.row(row).iter())
                            .map(|(jet, error)| (jet * error).abs())
                            .sum::<f64>();
                        let prior = ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            self.assignment.coords_for_slot(row, slot)[axis],
                            periods[axis],
                        );
                        let band = gam_linalg::roundoff::accumulation_band(
                            self.output_dim + 1,
                            data + prior.grad.abs(),
                        );
                        sq += band * band;
                    }
                }
                Ok(sq)
            })
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?;
        let band = (decoder_sq + coordinate_sq).sqrt();
        if band.is_finite() {
            Ok(band)
        } else {
            Err(format!("support gradient rounding band is not finite: {band:e}"))
        }
    }

    /// Solve `A Δ = g` at the installed state: the exact Newton displacement the
    /// fixed point certifies on, and the direction a refused certificate steps
    /// along (#2933 F08).
    ///
    /// The solve is the profile adjoint's large-system route: flexible GMRES on the
    /// exact stationarity Jacobian, right-preconditioned by the majorizer arrow whose
    /// row blocks also carry the positive part of the exact row excess
    /// ([`Self::support_exact_a_preconditioner_rows`]), and its reduced-Schur solve. It
    /// adds no dense factorization, and it certifies the physical residual rather than
    /// a preconditioned proxy. `A` may be indefinite; GMRES does not assume otherwise.
    ///
    /// The residual bar is `‖g − AΔ‖ ≤ max(√ε‖g‖, β_g + γ_dim·‖A‖·‖Δ‖)`
    /// ([`solve_b_preconditioned_gmres_to_rounding_floor`]), with `β_g` the gradient's
    /// rounding band ([`Self::gradient_rounding_band`]) and `γ_dim·‖A‖·‖Δ‖` the backward
    /// error of forming `AΔ`. The system is defined only to the digits `g` carries, and
    /// `AΔ` only to its own, so no solve can be asked for a residual below their sum. The
    /// `√ε` bar alone asks exactly that at a state converged to round-off: there `g` is
    /// noise of about `β_g`, and part of it need not lie in what `A` represents. The
    /// full Krylov space then leaves a residual of a few thousandths of `‖g‖`, and the
    /// solve refused the states the fixed point exists to return (every
    /// `support_outer` test at 69befd10c5, job 1109683: ‖g‖ ≈ 2e-15, residual 1e-17).
    /// A gradient above its band keeps the `√ε` bar, so a displaced state still prices
    /// its displacement and refuses.
    fn exact_newton_solve(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<(SaeSupportNewtonDisplacement, SaeArrowVector), String> {
        // #2576 — the solve's cost is where a certifying cycle's time goes, so each
        // certificate reports it, the same telemetry `joint_newton_step` carries.
        let started = std::time::Instant::now();
        let system = self.assemble_arrow_schur(target, lambda_smooth, ard_precisions)?;
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        if beta_dim != system.k {
            return Err(format!(
                "support Newton displacement beta layout {beta_dim} != system border {}",
                system.k
            ));
        }
        let coordinate_dim = *system.row_offsets.last().unwrap_or(&0);
        let mut gradient = SaeArrowVector {
            t: Array1::<f64>::zeros(coordinate_dim),
            beta: system.gb.clone(),
        };
        for (row, block) in system.rows.iter().enumerate() {
            gradient
                .t
                .slice_mut(ndarray::s![system.row_offsets[row]..system.row_offsets[row + 1]])
                .assign(&block.gt);
        }
        let rows = self.support_outer_differential_rows(target, ard_precisions, &beta_offsets)?;
        let factors = self
            .support_exact_a_preconditioner(&system, &rows)
            .map_err(|error| format!("support Newton displacement: {error}"))?;
        let gradient_band = self.gradient_rounding_band(target, lambda_smooth, ard_precisions)?;
        let prepared = started.elapsed();
        let (displacement, iterations) = solve_b_preconditioned_gmres_to_rounding_floor(
            &gradient,
            &SaeArrowVector {
                t: Array1::<f64>::zeros(coordinate_dim),
                beta: Array1::<f64>::zeros(beta_dim),
            },
            |vector| self.support_outer_exact_hessian_apply(&system, &rows, vector),
            |rhs| support_arrow_majorizer_inverse(&system, &factors, rhs, None),
            gradient_band,
        )
        .map_err(|error| {
            format!(
                "support Newton displacement solve (‖g‖ {:.3e}, gradient rounding band \
                 {gradient_band:.3e}): {error}",
                (gradient.t.dot(&gradient.t) + gradient.beta.dot(&gradient.beta)).sqrt()
            )
        })?;
        // `f64::max` returns its non-NaN operand, so finiteness is checked before
        // the reductions rather than read off them.
        if !displacement
            .t
            .iter()
            .chain(displacement.beta.iter())
            .all(|value| value.is_finite())
        {
            return Err("support Newton displacement is non-finite".to_string());
        }
        let max_abs =
            |values: &Array1<f64>| values.iter().fold(0.0_f64, |current, value| current.max(value.abs()));
        let decrement_sq =
            gradient.t.dot(&displacement.t) + gradient.beta.dot(&displacement.beta);
        if !decrement_sq.is_finite() {
            return Err("support Newton decrement is non-finite".to_string());
        }
        let certificate = SaeSupportNewtonDisplacement {
            decoder_max_abs: max_abs(&displacement.beta),
            coordinate_max_abs: max_abs(&displacement.t),
            decrement_sq,
            coordinates: displacement.t.clone(),
            decoder: displacement.beta.clone(),
        };
        log::info!(
            "support Newton displacement: max {:.3e} (decoder {:.3e}, coordinate {:.3e}) over \
             {coordinate_dim} coordinates and border {beta_dim}; assemble, differential rows, \
             row factors and gradient band {:.2}s, exact-A FGMRES {iterations} iterations {:.2}s",
            certificate.max_abs(),
            certificate.decoder_max_abs,
            certificate.coordinate_max_abs,
            prepared.as_secs_f64(),
            (started.elapsed() - prepared).as_secs_f64(),
        );
        Ok((certificate, displacement))
    }

    /// Take the exact Newton step `−Δ` a refused certificate already paid for
    /// (#2933 F08), backtracking only against a RESOLVED increase of the penalized
    /// objective.
    ///
    /// Each trial re-solves the decoder block given the stepped coordinates, with the
    /// same sweep the fixed point uses. The decoder is a linear least-squares block
    /// given the coordinates, so this is the variable-projection form of the step:
    /// its coordinate part is the reduced Newton step wherever the decoder gradient
    /// vanishes. It follows the valleys the certificate exists for. A
    /// reparameterisation orbit that is straight in the coordinates is curved in the
    /// decoder (`β₁ = β₁*/s` for a scaled chart), so the straight Newton line leaves
    /// it quadratically and a likelihood increase would force the step to a crawl.
    ///
    /// Near the stationary point the step's predicted decrease `½gᵀΔ` can lie below
    /// the objective's arithmetic resolution while the displacement is still above
    /// tolerance, so demanding a measured decrease would refuse exactly the steps the
    /// certificate needs. What guards progress instead is the caller's next
    /// certificate, which must find the displacement strictly contracted. Returns
    /// the objective at the installed step, or `None` with the state restored bit
    /// for bit when `Δ` is not a descent direction (`gᵀΔ ≤ 0`, the exact curvature
    /// is not positive along it) or no trial avoids a resolved increase.
    fn exact_newton_step(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        objective: f64,
        displacement: &SaeArrowVector,
        decrement_sq: f64,
        coordinate_snapshot: &mut Vec<f64>,
        scaled_step: &mut Vec<f64>,
        trial_fitted: &mut Array2<f64>,
    ) -> Result<Option<f64>, String> {
        if !(decrement_sq.is_finite() && decrement_sq > 0.0) {
            return Ok(None);
        }
        let (beta_offsets, _) = self.beta_layout()?;
        self.snapshot_coordinates(coordinate_snapshot);
        let decoder_snapshot = self
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients().clone())
            .collect::<Vec<_>>();
        let step = SaeArrowVector {
            t: displacement.t.mapv(|value| -value),
            beta: displacement.beta.mapv(|value| -value),
        };
        let resolution = self.objective_descent_resolution(objective);
        let mut trial_coordinates = Vec::with_capacity(coordinate_snapshot.len());
        let mut scale = 1.0_f64;
        loop {
            let changed = self.install_scaled_arrow_displacement(
                coordinate_snapshot,
                &decoder_snapshot,
                &beta_offsets,
                &step,
                scale,
                scaled_step,
                &mut trial_coordinates,
            )?;
            if !changed {
                break;
            }
            self.reconstruct_into(trial_fitted)?;
            match self.decoder_fista_passes {
                Some(passes) => {
                    self.decoder_sweep_fista(target, lambda_smooth, trial_fitted, passes)?
                }
                None => self.decoder_sweep(target, lambda_smooth, trial_fitted)?,
            };
            let trial_residual = &target - &*trial_fitted;
            let trial = self.penalized_objective_with_residual(
                &trial_residual,
                lambda_smooth,
                ard_precisions,
            )?;
            if trial.is_finite() && trial - objective <= resolution {
                return Ok(Some(trial));
            }
            // Past this rung the step's whole first-order change is within the
            // objective's resolution, so no smaller rung can tell an increase from
            // a decrease. The negated comparison also stops on NaN.
            if !(scale * decrement_sq > resolution) {
                break;
            }
            scale *= 0.5;
        }
        self.install_coordinates(coordinate_snapshot)?;
        for (atom, decoder) in decoder_snapshot.into_iter().enumerate() {
            self.atoms[atom].set_decoder_coefficients(decoder)?;
        }
        Ok(None)
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
    pub(crate) fn raw_residual(&self, target: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
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
    pub(crate) fn penalized_objective_with_residual(
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
                    let periods = self.atom_ard_axis_periods(atom);
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
    pub(crate) fn fellner_schall_smoothing_with_residual(
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

    /// One atom's decoder data Gram `Σ_i φ_k(t_i) φ_k(t_i)ᵀ` over the rows routed
    /// to it, at the current coordinates. The curvature census and the grouped
    /// LAML ρ domain read this one matrix.
    pub(crate) fn atom_decoder_gram(&self, atom_idx: usize) -> Result<Array2<f64>, String> {
        let m = self.atoms[atom_idx].basis_size();
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
        Ok(gram)
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
            let penalty = self.atoms[atom_idx].smooth_penalty().clone();
            let gram = self.atom_decoder_gram(atom_idx)?;
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
            let periods = self.atom_ard_axis_periods(atom_idx).to_vec();
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

    /// The exact Hessian of one row's frozen-decoder coordinate objective
    /// `½‖y − Σ_s f_s(t_s)‖² + Σ V_ard(t)`: the Gauss-Newton gram `J Jᵀ`, less the
    /// residual's second-jet term `Σ_o r_o ∂²f_o/∂t_a∂t_b` within each slot (the slots'
    /// decodes add, so cross-slot second derivatives vanish), plus each axis's exact
    /// prior curvature. `None` when a slot's retraction does not add the step, so a
    /// coordinate Newton step is not the motion the retraction takes, or when a slot's
    /// basis exposes no analytic second jet.
    fn exact_row_coordinate_hessian(
        &self,
        row: usize,
        coords_row: &[f64],
        offsets: &[Range<usize>],
        support: &[u32],
        jacobian: &Array2<f64>,
        residual: &Array1<f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<Option<Array2<f64>>, String> {
        let mut hessian = jacobian.dot(&jacobian.t());
        for (slot, &atom_index) in support.iter().enumerate() {
            let atom_index = atom_index as usize;
            if !self.assignment.atom_retraction_adds_the_step(atom_index) {
                return Ok(None);
            }
            let atom = &self.atoms[atom_index];
            let d = atom.latent_dim();
            let m = atom.basis_size();
            let start = offsets[slot].start;
            let coordinates = ArrayView2::from_shape((1, d), &coords_row[offsets[slot].clone()])
                .map_err(|error| {
                    format!(
                        "support row {row} exact coordinate Hessian: atom {atom_index} \
                         coordinate view: {error}"
                    )
                })?;
            let Some(second) = self.slot_second_jet(atom_index, coordinates)? else {
                return Ok(None);
            };
            if second.dim() != (1, m, d, d) {
                return Err(format!(
                    "support row {row} exact coordinate Hessian: atom {atom_index} second jet \
                     shape {:?} != (1, {m}, {d}, {d})",
                    second.dim()
                ));
            }
            // `D_basis · r` once per basis function, so the residual curvature of an axis
            // pair is one pass over the basis.
            let decoder = atom.decoder_coefficients();
            let weighted: Vec<f64> = (0..m)
                .map(|basis| decoder.row(basis).dot(residual))
                .collect();
            let periods = self.atom_ard_axis_periods(atom_index);
            for axis_a in 0..d {
                for axis_b in 0..d {
                    let residual_curvature: f64 = (0..m)
                        .map(|basis| second[[0, basis, axis_a, axis_b]] * weighted[basis])
                        .sum();
                    hessian[[start + axis_a, start + axis_b]] -= residual_curvature;
                }
                hessian[[start + axis_a, start + axis_a]] += ArdAxisPrior::eval(
                    ard_precisions[atom_index][axis_a],
                    coords_row[start + axis_a],
                    periods[axis_a],
                )
                .hess;
            }
        }
        Ok(Some(hessian))
    }

    /// Whether a row Hessian makes the trust-region model strictly convex: every
    /// eigenvalue above the REML positive-eigenspace band, the rank rule the support
    /// lane's penalty spectra use.
    fn row_hessian_is_positive_definite(hessian: &Array2<f64>) -> Result<bool, String> {
        let (eigenvalues, _) = hessian
            .eigh(Side::Lower)
            .map_err(|error| format!("support row coordinate Hessian eigh: {error}"))?;
        let values = eigenvalues.to_vec();
        let threshold =
            gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(&values);
        Ok(values.iter().all(|&value| value > threshold))
    }

    /// One row's coordinate step with manifold-aware backtracking, on the row's
    /// caller-held coordinate block. Semantically the serial sweep's row iteration.
    /// The trust-region model is the row's exact Hessian where every slot's retraction
    /// adds the step and that Hessian is positive definite, and the Gauss-Newton
    /// majorizer otherwise.
    ///
    /// Storage-wise: the Gauss-Newton path allocates nothing. The scratch is the CALLER's, held
    /// per rayon worker and reused across every row that worker takes (#2575).
    /// It used to be per-row — eighteen allocations per row, `N` rows per
    /// sweep, hundreds of sweeps per fit — and the doc comment's claim that
    /// "the line-search halvings allocate nothing" was true within a row and
    /// misleading across them; the profiled 12.4% of self time in
    /// `malloc`/`free`/`memmove` is what that cost.
    ///
    /// A row that takes a step also evaluates each slot's analytic second jet for the exact
    /// Hessian (#2576): one allocation per active slot, which buys a quadratic local rate
    /// where the majorizer contracts linearly.
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
        // Each raw gradient component's rounding band is γ over the terms it sums: the
        // residual-weighted jet cells and the prior gradient (#2469).
        let gradient_ops = p + 1;
        let gradient_gamma = gam_linalg::roundoff::accumulation_growth(gradient_ops);
        let mut raw_gradient_band = 0.0_f64;
        let mut prior_cursor = 0usize;
        for (slot, &atom) in support.iter().enumerate() {
            let atom = atom as usize;
            let periods = self.atom_ard_axis_periods(atom);
            for axis in 0..self.assignment.atom_coord_dim(atom) {
                let prior = ArdAxisPrior::eval(
                    ard_precisions[atom][axis],
                    coords_row[offsets[slot].start + axis],
                    periods[axis],
                );
                row_objective_scale += prior.value.abs();
                rhs_vector[prior_cursor] -= prior.grad;
                let terms = jacobian
                    .row(prior_cursor)
                    .iter()
                    .zip(residual.iter())
                    .map(|(jet, residual_cell)| (jet * residual_cell).abs())
                    .sum::<f64>()
                    + prior.grad.abs();
                raw_gradient_band = raw_gradient_band.max(gradient_gamma * terms);
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
        // #2576: the Gauss-Newton gram drops the residual's second-jet term and clamps the
        // signed periodic ARD curvature, so a row whose residual does not vanish contracts
        // only linearly near its fixed point. Job 609612's frozen-decoder polish lowered its
        // raw coordinate KKT only from 1.02e-2 to 4.83e-3 between sweeps 64 and 200. Where
        // the row's exact Hessian is available and positive definite, the trust-region model
        // is that Hessian and the step is the exact Newton step.
        let curvature = match self.exact_row_coordinate_hessian(
            row,
            coords_row,
            offsets,
            support,
            jacobian,
            &residual,
            ard_precisions,
        )? {
            Some(hessian) if Self::row_hessian_is_positive_definite(&hessian)? => hessian,
            _ => gram,
        };
        // SPEC-22: the exact PSD trust-region subproblem is general outer
        // optimizer machinery and lives in `opt`. gam kept a private copy
        // until #2574.
        let delta = opt::solve_psd_trust_region(curvature.view(), rhs_vector.view(), trust_radius)
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
        let mut best_step = 0.0_f64;
        let mut best_objective_delta = f64::INFINITY;
        let evaluation_ops = 1usize
            + p
            + q
            + dims.iter().map(|&(m, _)| m * p).sum::<usize>();
        let gamma = gam_linalg::roundoff::accumulation_growth(evaluation_ops);
        let objective_resolution = gamma * row_objective_scale;
        let mut step = 1.0_f64;
        loop {
            self.assignment.project_row_coords(row, old_coords, coords_row)?;
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
                let periods = self.atom_ard_axis_periods(atom);
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
            if objective_delta.is_finite() && objective_delta < best_objective_delta {
                best_objective_delta = objective_delta;
                best_step = step;
            }
            trial_fitted.fill(0.0);
            for slot_trial in trial.iter() {
                *trial_fitted += &slot_trial.decoded;
            }
            trial_residual.assign(&target.row(row));
            *trial_residual -= &*trial_fitted;
            let mut trial_gradient_max = 0.0_f64;
            let mut trial_gradient_band = 0.0_f64;
            for (slot, &atom) in support.iter().enumerate() {
                let atom = atom as usize;
                let periods = self.atom_ard_axis_periods(atom);
                for axis in 0..dims[slot].1 {
                    let jacobian_row = trial[slot].jacobian.row(axis);
                    let prior_gradient = ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        coords_row[offsets[slot].start + axis],
                        periods[axis],
                    )
                    .grad;
                    let gradient = -jacobian_row.dot(&*trial_residual) + prior_gradient;
                    trial_gradient_max = trial_gradient_max.max(gradient.abs());
                    let terms = jacobian_row
                        .iter()
                        .zip(trial_residual.iter())
                        .map(|(jet, residual_cell)| (jet * residual_cell).abs())
                        .sum::<f64>()
                        + prior_gradient.abs();
                    trial_gradient_band = trial_gradient_band.max(gradient_gamma * terms);
                }
            }
            // A step is taken only on a resolved change: the row objective falls by more than
            // its rounding band, or it ties inside that band while the gradient falls by more
            // than the two gradients' rounding bands. No fraction of the predicted decrease is
            // asked for (#2469).
            let resolved_decrease =
                objective_delta.is_finite() && objective_delta < -objective_resolution;
            let resolved_gradient_tie = objective_delta.is_finite()
                && objective_delta.abs() <= objective_resolution
                && trial_gradient_max < raw_gradient_max - (raw_gradient_band + trial_gradient_band);
            if resolved_decrease || resolved_gradient_tie {
                accepted = Some(step);
                break;
            }
            // Past this rung the step's whole first-order change `step·rhsᵀδ` is within
            // the objective's round-off resolution, so neither test can tell a smaller
            // step from the current point. The negated comparison also stops on NaN.
            if !(step * directional > objective_resolution) {
                break;
            }
            step *= 0.5;
        }
        match accepted {
            Some(step) => {
                for value in delta.iter() {
                    max_change = max_change.max((step * value).abs());
                }
            }
            None => {
                self.assignment.project_row_coords(row, old_coords, coords_row)?;
                // The search stops only at the rung whose whole first-order change
                // `step·rhsᵀδ` is within the objective's own round-off resolution, so no
                // trial at any smaller step could certify a decrease. Measured on the #2502
                // REML lane (row 228530): a real gradient (raw KKT 31) whose descent is
                // unresolvable at f64. Taking no step leaves the row's KKT high, so the
                // outer certificate honestly refuses to certify; erroring instead would
                // discard a whole fitted model over one unmeasurable row.
                log::debug!(
                    "coordinate row {row}: line search unmeasurable \
                     (rhs_dot_delta={directional:.3e}, delta_max={delta_max:.3e}, \
                     objective resolution {objective_resolution:.3e}, best_step={best_step:.3e}, \
                     best_objective_delta={best_objective_delta:.3e}, \
                     raw KKT max={raw_gradient_max:.3e}); taking no step"
                );
                return Ok(max_change);
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
                    let periods = self.atom_ard_axis_periods(atom);
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
                    let periods = self.atom_ard_axis_periods(atom);
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

    fn frozen_decoder_coordinate_objective_with_residual(
        &self,
        residual: &Array2<f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        let mut objective = 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
        for row in 0..self.n_obs() {
            for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                let atom = atom as usize;
                let periods = self.atom_ard_axis_periods(atom);
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

    /// Frozen-decoder OOS coordinate solve over active supports only.
    ///
    /// It sweeps until the certificate holds on two consecutive sweeps, with no
    /// iteration budget (#2469, SPEC rule 23). It refuses only when no later sweep
    /// can meet the certificate: once the objective has recurred, a sweep that moves
    /// no coordinate while the raw coordinate KKT still exceeds its bar. With the
    /// decoder frozen the sweep is deterministic, so a motionless sweep reproduces
    /// itself forever.
    ///
    /// Termination needs no budget either. A row step is accepted only on a resolved
    /// change. Either its row objective falls by more than its rounding band, which is at
    /// least `γ` because a row objective's scale is at least one. Or the objective ties
    /// inside that band while the row's gradient falls by more than its gradients' rounding
    /// bands. The objective is bounded below, and a floating-point gradient cannot keep
    /// falling by resolved amounts, so only finitely many steps are ever accepted.
    pub fn solve_coordinates_fixed_decoder(
        &mut self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
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
        if !(tolerance.is_finite() && tolerance > 0.0) {
            return Err("SaeSupportSparseTerm::solve_coordinates_fixed_decoder requires a finite positive tolerance".into());
        }
        let mut previous_candidate = false;
        let mut last_objective: Option<f64> = None;
        // Sweeps 1, 2, 4, ... and the motionless sweep, reported with a refusal: a slow
        // linear rate and a row that stopped moving leave different trajectories in the
        // raw coordinate KKT, the objective and the largest coordinate move (#2576, #2023).
        let mut trajectory: Vec<(usize, f64, f64, f64)> = Vec::new();
        // Decoders are frozen here, so the coordinate sweep's per-changed-row
        // refresh is the ONLY thing that moves the decode: the maintained
        // matrix stays exact (each changed row is recomputed from state, not
        // incremented), and no drift re-verification is needed to certify.
        let mut fitted_state = self.reconstruct()?;
        let mut iteration = 0usize;
        loop {
            iteration += 1;
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
            // No coordinate moved, so the next sweep starts from this sweep's state, takes
            // the same decisions and again moves nothing: the KKT limb that failed here
            // fails at every later sweep.
            let stalled = max_change == 0.0
                && objective_recurred
                && coordinate_max_abs > tolerance * kkt_scale;
            if iteration.is_power_of_two() || stalled {
                trajectory.push((iteration, coordinate_max_abs, objective, max_change));
            }
            if stalled {
                let trajectory = trajectory
                    .iter()
                    .map(|(sweep, kkt, sweep_objective, change)| {
                        format!(
                            "{sweep}: KKT {kkt:.3e}, objective {sweep_objective:.9e}, max change {change:.3e}"
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(format!(
                    "SaeSupportSparseTerm::solve_coordinates_fixed_decoder stalled at sweep {iteration}: no coordinate moved while the raw coordinate KKT max={coordinate_max_abs:.6e} exceeded tolerance {tolerance:.6e} relative to objective {objective:.6e} ({:.6e}); sweeps [{trajectory}]",
                    coordinate_max_abs / kkt_scale
                ));
            }
            previous_candidate = candidate;
        }
    }

    /// Alternate exact decoder blocks and direct active-row coordinate Newton
    /// steps until the raw KKT residual AND a full-cycle recurrence agree. A
    /// state no cycle can move measurably is refused; only converged fits are returned.
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
        // #2576: a certified fixed point is stationary AND curvature-audited, so the audit
        // is never skipped for the work a solve has or has not done; admitting it by a
        // cycle count let a solve that certified in fewer cycles than the pencil's dimension
        // certify a saddle. Its one admission is the resource it needs: the dense pencil
        // runs wherever its workspace fits the cgroup-aware in-core ledger. Where it does
        // not fit, this lane has no matrix-free curvature certificate, so the stationary
        // state is refused as unaudited instead of certified.
        if full_dim == 0 {
            return Ok(None);
        }
        if dense_workspace > in_core_budget {
            return Err(format!(
                "SaeSupportSparseTerm::solve_fixed_point: curvature not audited: the exact \
                 stationarity pencil of dimension {full_dim} needs a dense workspace of \
                 {dense_workspace} bytes against an in-core budget of {in_core_budget} bytes, \
                 and this lane has no matrix-free curvature certificate, so the stationary \
                 state is refused instead of certified"
            ));
        }
        let system = self.assemble_arrow_schur(target, lambda_smooth, ard_precisions)?;
        let rows = self.support_outer_differential_rows(target, ard_precisions, &beta_offsets)?;
        self.support_outer_negative_curvature_mode(&system, &rows)
    }

    /// Every row's discrete support, flattened with its length, so a carried
    /// coupled-step displacement is only reused under the layout it was taken in.
    fn support_fingerprint(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for row in 0..self.n_obs() {
            let support = self.assignment.support_indices(row);
            out.push(support.len() as u32);
            out.extend_from_slice(support);
        }
        out
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
        previous_step: &mut Option<SupportJointStepMemory>,
    ) -> Result<Option<f64>, String> {
        // #2576: a coupled cycle costs about 350 alternation cycles on the 3000x48
        // chart (job 557966), so each step reports where its time went.
        let step_start = std::time::Instant::now();
        let mut system = self.assemble_arrow_schur(target, lambda_smooth, ard_precisions)?;
        if system.k == 0 || self.n_obs() == 0 {
            *previous_step = None;
            return Ok(None);
        }
        // Opt the per-row factorisation into spectral discovery. A row whose
        // coordinate block is flat along one axis (a periodic atom sitting at a
        // stationary phase, an atom the router left with a single row) has a
        // singular `H_tt^(i)`, and refusing to factor it would discard the whole
        // joint step over one row. Deflating that direction to unit stiffness is
        // what the dense manifold lane already does for the same reason.
        SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut system);
        let assembled = step_start.elapsed();
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
        let mut solve_attempts = 0usize;
        let mut solve_iterations = 0usize;
        let mut solve_refusal = String::new();
        for attempt in 0..4 {
            solve_attempts += 1;
            match system.solve_with_options(ridge, ridge, &options) {
                Ok((delta_t, delta_beta, diagnostics)) => {
                    solve_iterations += diagnostics.iterations;
                    // #2576: every CG iterate from zero lowers the majorizer's reduced
                    // quadratic model, and eliminating Δt exactly only adds
                    // −½·g_tᵀH_tt⁻¹g_t, so the full model is negative and gᵀd < 0.
                    // An iterate that spent its product budget is therefore a
                    // descent direction the line search below already guards.
                    // Refusing it discarded the direction and re-ran the whole
                    // preconditioner ladder at three more ridges. The derived
                    // tolerance stays the CG's target and the certificate's bar. A
                    // gauge-pinned solve that spends its budget returns `Err`,
                    // and stays refused.
                    let admissible = matches!(
                        diagnostics.stopping_reason,
                        gam_solve::arrow_schur::PcgStopReason::Converged
                            | gam_solve::arrow_schur::PcgStopReason::BudgetExhausted
                    ) && diagnostics.final_relative_residual.is_finite();
                    if admissible {
                        step_pair = Some((delta_t, delta_beta));
                        break;
                    }
                    solve_refusal = format!(
                        "stop={:?}, relative residual {:.3e}, requested {:.3e}",
                        diagnostics.stopping_reason,
                        diagnostics.final_relative_residual,
                        stationarity_tolerance,
                    );
                    log::debug!(
                        "support joint Newton linear solve refused at ridge {ridge:.3e} \
                         (attempt {attempt}): {solve_refusal}"
                    );
                }
                Err(error) => {
                    solve_refusal = error.to_string();
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
        let solved = step_start.elapsed();
        let (delta_t, delta_beta) = match step_pair {
            Some(pair) => pair,
            None => {
                log::info!(
                    "support joint Newton: linear solve refused after {solve_attempts} attempt(s) \
                     and {solve_iterations} PCG iterations ({solve_refusal}); assemble {:.2}s, \
                     solve {:.2}s",
                    assembled.as_secs_f64(),
                    (solved - assembled).as_secs_f64(),
                );
                *previous_step = None;
                return Ok(None);
            }
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
            *previous_step = None;
            return Ok(None);
        }
        // #2576 — the step is the FIRST iteration of Steihaug–Toint CG on the
        // exact observed information `A`, preconditioned by the majorizer `B`.
        // `d` solves `B d = −g`, which is that iteration's search direction, and
        // one exact Hessian–vector product prices the objective's own curvature
        // `dᵀA d` along it, so the exact model's minimiser along `d` is
        // `s* = −g·d / dᵀA d`. Measured on the 3000×48 chart of #2576 (09-04),
        // the objective is quadratic along the majorizer step to 2–5 %, with
        // curvature 2.3–4.1× the majorizer's (the residual's second-jet term the
        // majorizer drops). The ladder from `s = 1` therefore accepted `½` or `¼`
        // on 67 of 93 steps where `s* ≈ ⅓` is the model's own answer.
        // Backtracking from `s*` keeps acceptance on the actual objective. A
        // non-positive exact curvature leaves `s*` undefined, while `d` is still a
        // descent direction, so the ladder then starts from `s = 1` exactly as
        // before. Further Steihaug iterations would each need another majorizer
        // inverse (the reduced-Schur CG, ~1.4 s per apply in that measurement), so
        // they are not taken here.
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        let gradient_dot_step = {
            let mut acc = 0.0_f64;
            for (row, block) in system.rows.iter().enumerate() {
                let offset = system.row_offsets[row];
                for j in 0..system.row_dims[row] {
                    acc += block.gt[j] * delta_t[offset + j];
                }
            }
            for (g, d) in system.gb.iter().zip(delta_beta.iter()) {
                acc += g * d;
            }
            acc
        };
        if delta_beta.len() != beta_dim {
            return Err(format!(
                "SaeSupportSparseTerm::joint_newton_step: arrow border width {} != decoder \
                 width {beta_dim}",
                delta_beta.len()
            ));
        }
        let rows = self.support_outer_differential_rows(target, ard_precisions, &beta_offsets)?;
        let direction = SaeArrowVector {
            t: delta_t.clone(),
            beta: delta_beta.clone(),
        };
        let applied = self.support_outer_exact_hessian_apply(&system, &rows, &direction)?;
        let exact_curvature = direction.t.dot(&applied.t) + direction.beta.dot(&applied.beta);
        // #2576 — the exact first scale along `d` made the model predict the
        // realised decrease (ratio 1.000 on the 3000x48 chart, job 442005), but
        // the terminal coupled steps then took s ≈ 0.43 cycle after cycle with a
        // decrease that GREW about 1.4% per cycle: the iterate walks a valley
        // along a direction `d` alone does not span. The last accepted
        // displacement `p` is that valley's secant, so the exact model is
        // minimised over span{d, p}: one more exact Hessian apply and a 2x2
        // solve, no second majorizer inverse. It is taken only when the 2x2 exact
        // curvature is positive definite beyond its round-off floor and the model
        // descends; otherwise the step is the 1D exact scale along `d`, and the
        // majorizer ladder from `s = 1` when `dᵀAd` is not positive.
        let previous = previous_step.take().filter(|memory| {
            memory.t.len() == delta_t.len()
                && memory.beta.len() == delta_beta.len()
                && memory.support == self.support_fingerprint()
        });
        let mut subspace = None;
        if let Some(memory) = previous.as_ref() {
            let other = SaeArrowVector {
                t: memory.t.clone(),
                beta: memory.beta.clone(),
            };
            let applied_other = self.support_outer_exact_hessian_apply(&system, &rows, &other)?;
            let cross = direction.t.dot(&applied_other.t) + direction.beta.dot(&applied_other.beta);
            let other_curvature = other.t.dot(&applied_other.t) + other.beta.dot(&applied_other.beta);
            let mut gradient_dot_other = 0.0_f64;
            for (row, block) in system.rows.iter().enumerate() {
                let offset = system.row_offsets[row];
                for j in 0..system.row_dims[row] {
                    gradient_dot_other += block.gt[j] * other.t[offset + j];
                }
            }
            for (g, value) in system.gb.iter().zip(other.beta.iter()) {
                gradient_dot_other += g * value;
            }
            let determinant = exact_curvature * other_curvature - cross * cross;
            let magnitude = exact_curvature
                .abs()
                .max(other_curvature.abs())
                .max(cross.abs());
            if exact_curvature > 0.0
                && other_curvature > 0.0
                && determinant.is_finite()
                && determinant > f64::EPSILON * magnitude * magnitude
            {
                let coefficient_d =
                    (cross * gradient_dot_other - other_curvature * gradient_dot_step) / determinant;
                let coefficient_p =
                    (cross * gradient_dot_step - exact_curvature * gradient_dot_other) / determinant;
                let linear = coefficient_d * gradient_dot_step + coefficient_p * gradient_dot_other;
                if coefficient_d.is_finite() && coefficient_p.is_finite() && linear < 0.0 {
                    subspace = Some((coefficient_d, coefficient_p, linear));
                }
            }
        }
        // `predicted(s) = -(s·linear + ½·s²·quadratic)` for whichever model chose
        // the first trial, so the acceptance log reads that model's accuracy.
        let (model, step_t, step_beta, model_linear, model_quadratic, first_scale) =
            match (subspace, previous.as_ref()) {
                (Some((coefficient_d, coefficient_p, linear)), Some(memory)) => (
                    "subspace",
                    &delta_t * coefficient_d + &memory.t * coefficient_p,
                    &delta_beta * coefficient_d + &memory.beta * coefficient_p,
                    linear,
                    -linear,
                    1.0,
                ),
                _ => {
                    let exact_scale = (gradient_dot_step < 0.0
                        && exact_curvature.is_finite()
                        && exact_curvature > 0.0)
                        .then(|| -gradient_dot_step / exact_curvature)
                        .filter(|scale| scale.is_finite() && *scale > 0.0);
                    match exact_scale {
                        Some(scale) => (
                            "exact",
                            delta_t.clone(),
                            delta_beta.clone(),
                            gradient_dot_step,
                            exact_curvature,
                            scale,
                        ),
                        None => (
                            "majorizer",
                            delta_t.clone(),
                            delta_beta.clone(),
                            gradient_dot_step,
                            -gradient_dot_step,
                            1.0,
                        ),
                    }
                }
            };
        let curved = step_start.elapsed();
        self.snapshot_coordinates(coordinate_snapshot);
        let restore: Vec<Array2<f64>> = self
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients().clone())
            .collect();
        let output_dim = self.output_dim;
        // Backtracking on the SAME objective the certificate reads, from the
        // model's first scale. It walks down until a rung's whole first-order change
        // `scale·|model_linear|` is within the objective's arithmetic resolution, the
        // coordinate sweep's own stopping rule, so neither block is walked past the
        // point where a decrease could still be measured.
        let mut halving = 0usize;
        loop {
            let scale = first_scale * 2.0_f64.powi(-(halving as i32));
            self.install_coordinates(coordinate_snapshot)?;
            scaled_step.clear();
            scaled_step.extend(step_t.iter().map(|value| scale * value));
            self.retract_coordinates(scaled_step)?;
            for (atom, base) in restore.iter().enumerate() {
                let mut decoder = base.clone();
                let offset = beta_offsets[atom];
                for basis in 0..decoder.nrows() {
                    for channel in 0..output_dim {
                        decoder[[basis, channel]] +=
                            scale * step_beta[offset + basis * output_dim + channel];
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
            // Measured on `tiered_certifies_at_k_gg_rank_once_the_support_step_descends_2275_2825`
            // (then named `tiered_returns_best_effort_open_certificate_at_k_gg_rank_2275`)
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
                // The prediction comes from the model the first trial was chosen
                // from, so the ratio reads that model's accuracy along this step.
                let predicted = -(scale * model_linear + 0.5 * scale * scale * model_quadratic);
                let searched = step_start.elapsed();
                log::info!(
                    "support joint Newton: accepted scale={scale:.6e} (2^-{halving} of \
                     {first_scale:.6e}, {model} model) predicted={predicted:+.3e} \
                     actual={:+.3e} ratio={:.3} objective={objective:.9e} -> {trial:.9e}; \
                     assemble {:.2}s, solve {:.2}s ({solve_iterations} PCG iterations over \
                     {solve_attempts} attempt(s)), exact curvature {:.2}s, line search {:.2}s \
                     over {} objective evaluation(s)",
                    objective - trial,
                    if predicted != 0.0 { (objective - trial) / predicted } else { f64::NAN },
                    assembled.as_secs_f64(),
                    (solved - assembled).as_secs_f64(),
                    (curved - solved).as_secs_f64(),
                    (searched - curved).as_secs_f64(),
                    halving + 1,
                );
                self.reconstruct_into(fitted)?;
                *previous_step = Some(SupportJointStepMemory {
                    t: step_t.mapv(|value| scale * value),
                    beta: step_beta.mapv(|value| scale * value),
                    support: self.support_fingerprint(),
                });
                return Ok(Some(trial));
            }
            // Past this rung the step's first-order change is within the objective's
            // resolution, so no smaller rung could measure a decrease. The negated
            // comparison also stops on NaN.
            if !(scale * -model_linear > objective_resolution) {
                break;
            }
            halving += 1;
        }
        log::info!(
            "support joint Newton: no measurable decrease along the {model} step after {} \
             objective evaluation(s); assemble {:.2}s, solve {:.2}s ({solve_iterations} PCG \
             iterations over {solve_attempts} attempt(s)), exact curvature {:.2}s, line search \
             {:.2}s",
            halving + 1,
            assembled.as_secs_f64(),
            (solved - assembled).as_secs_f64(),
            (curved - solved).as_secs_f64(),
            (step_start.elapsed() - curved).as_secs_f64(),
        );
        self.install_coordinates(coordinate_snapshot)?;
        for (atom, decoder) in restore.into_iter().enumerate() {
            self.atoms[atom].set_decoder_coefficients(decoder)?;
        }
        *previous_step = None;
        Ok(None)
    }

    pub fn solve_fixed_point(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
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
        if !(tolerance.is_finite() && tolerance > 0.0) {
            return Err("SaeSupportSparseTerm::solve_fixed_point requires a finite positive tolerance".into());
        }
        let mut previous_candidate = false;
        let mut last_max_change = f64::NAN;
        let mut last_objective: Option<f64> = None;
        // The last exact Newton displacement a refused certificate priced, with the
        // support it was priced under (#2933 F08): the next one must contract.
        let mut last_newton_displacement: Option<(f64, Vec<u32>)> = None;
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
        // (< 0.5% relative improvement over a window of >= 25 cycles) proposes
        // the same guarded move on a schedule the objective itself sets. The
        // window restarts at every proposal and at every test that finds
        // progress, so it measures the recent rate, not the whole descent since
        // the last proposal. The guard is unchanged -- accept only a strict
        // decrease -- so monotonicity survives by construction.
        let mut plateau_window_start = 0usize;
        let mut objective_at_window_start = f64::INFINITY;
        // #2575 joint-step bookkeeping. The joint step is attempted on every
        // cycle it is not skipping; a refusal doubles the skip so a lane where
        // the coupled model does not help pays a logarithmic number of extra
        // assemblies rather than one per cycle, and an acceptance clears it so
        // the terminal phase runs the coupled step every cycle. The doubling is
        // the schedule itself, not a tuned rate.
        let mut joint_snapshot = Vec::with_capacity(self.coordinate_state_len());
        let mut joint_scaled_step = Vec::with_capacity(self.coordinate_state_len());
        let mut joint_previous_step: Option<SupportJointStepMemory> = None;
        let mut joint_skip_remaining = 0usize;
        // PHASE (#2576). No cycle count phases or stops this loop. The alternation runs
        // while its cycles measurably move the fit, by the rule a row step is accepted on
        // (#2469): the objective falls by more than its rounding band, or it ties inside
        // that band while the raw KKT norm falls by more than both states' gradient
        // rounding bands. A cycle that does neither, and that the first-order screen does
        // not pass, is the measurement that the alternation cannot bring this state to its
        // certificate, and it arms the coupled step. No joint system is assembled before
        // that, so a fit whose alternation certifies takes exactly the trajectory it always
        // took.
        //
        // Inside the coupled phase such a cycle forces the coupled step, whatever its skip
        // schedule says. If that step is refused too, the state is a proven stall: neither
        // block sweep nor the coupled step moves it measurably, so the next cycle would
        // start from the same state and take the same decisions. The solve refuses there
        // instead of spending cycles on it.
        let mut joint_armed = false;
        let mut joint_skip_width = 1usize;
        let mut joint_accepted = 0usize;
        // What the progress rule knows about the state the last cycle ended in: its raw
        // KKT norm, and its gradient rounding band where that cycle priced one. Both are
        // `None` where something moved the state after they were read.
        let mut last_kkt_norm: Option<f64> = None;
        let mut last_gradient_band: Option<f64> = None;
        let mut iteration = 0usize;
        loop {
            iteration += 1;
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
            let phase_profiles = self.profile_periodic_phase_origins(ard_precisions)?
                + self.profile_linear_affine_gauges(lambda_smooth, ard_precisions)?
                + self.profile_euclidean_patch_affine_gauges(lambda_smooth, ard_precisions)?;
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
            // one ending on a cycle without measured progress IS the measurement
            // that this factor is too close to one for the alternation to reach
            // its certificate, so inside phase two there is nothing left to test:
            // the coupled step fires every cycle.
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
            let screened = stationarity.first_order_screen(
                kkt_scale,
                parameter_scale,
                tolerance,
            ) && pre_joint_change <= tolerance * parameter_scale;
            // #2576: whether this cycle's sweeps measurably moved the fit (see PHASE),
            // against the state the last cycle ended in. The gradient rounding band is
            // priced only where the objective ties, the one case that reads it.
            let kkt_norm = stationarity.decoder_l2.hypot(stationarity.coordinate_l2);
            let mut gradient_band = None;
            let swept_progress = match previous_objective {
                Some(previous) => {
                    let resolution = self.objective_descent_resolution(previous);
                    let change = objective - previous;
                    if change < -resolution {
                        Some(true)
                    } else if change.abs() <= resolution {
                        let band =
                            self.gradient_rounding_band(target, lambda_smooth, ard_precisions)?;
                        gradient_band = Some(band);
                        match (last_kkt_norm, last_gradient_band) {
                            (Some(previous_norm), Some(previous_band)) => {
                                Some(kkt_norm < previous_norm - (previous_band + band))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                None => None,
            };
            let stalled = swept_progress == Some(false) && !screened;
            if stalled && !joint_armed {
                // Phase boundary: the certificate must be earned on the coupled
                // trajectory, never inherited across it.
                joint_armed = true;
                accelerator.reset();
                taken_step.clear();
                taken_step.resize(self.coordinate_state_len(), 0.0);
                last_objective = None;
                previous_candidate = false;
                log::info!(
                    "support fixed point: cycle {iteration} made no measured progress (objective \
                     {objective:.9e}, raw KKT norm {kkt_norm:.3e}); arming the coupled \
                     (Schur-eliminated joint Newton) phase"
                );
            }
            // Whether this cycle ends in the state `kkt_norm` and `gradient_band` describe.
            let mut end_state_held = true;
            if joint_armed && joint_skip_remaining > 0 && !stalled {
                joint_skip_remaining -= 1;
            } else if joint_armed && !screened {
                match self.joint_newton_step(
                    target,
                    lambda_smooth,
                    ard_precisions,
                    &mut fitted_state,
                    objective,
                    tolerance,
                    &mut joint_snapshot,
                    &mut joint_scaled_step,
                    &mut joint_previous_step,
                )? {
                    Some(_accepted_objective) => {
                        joint_accepted += 1;
                        joint_skip_width = 1;
                        end_state_held = false;
                        // The coupled step can move along a periodic phase
                        // orbit after the cycle's first canonicalization. Put
                        // it back in the unique ARD-selected phase chart before
                        // either the recurrence certificate or the Laplace
                        // caller observes the state.
                        if self.profile_periodic_phase_origins(ard_precisions)?
                            + self.profile_linear_affine_gauges(lambda_smooth, ard_precisions)?
                            + self.profile_euclidean_patch_affine_gauges(
                                lambda_smooth,
                                ard_precisions,
                            )?
                            > 0
                        {
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
                    None if stalled => {
                        // A proven stall (see PHASE): refuse at the state reached.
                        let last_newton =
                            last_newton_displacement.as_ref().map(|(value, _)| *value);
                        return Err(format!(
                            "SaeSupportSparseTerm::solve_fixed_point stalled at cycle {iteration}: \
                             neither the block sweeps nor the coupled step moved the state \
                             measurably (objective {objective:.9e} tied within its rounding band \
                             {:.3e}; raw KKT norm {kkt_norm:.6e} against {:?} at the last cycle, \
                             gradient rounding bands {:?} and {gradient_band:?}; raw KKT \
                             max={:.6e}, relative to objective {:.6e}; per block: decoder \
                             max={:.6e} l2={:.6e}, coordinate max={:.6e} l2={:.6e}; \
                             diagonal-scaled residual max={:.6e}, scale={parameter_scale:.6e}, \
                             relative={:.6e} vs tolerance {tolerance:.6e} (decoder {:.6e}, \
                             coordinate {:.6e}); that screen schedules the exact Newton \
                             displacement certificate, last priced at {last_newton:?}; last \
                             parameter max_change={last_max_change:.6e}, joint Newton steps \
                             accepted={joint_accepted}, parameter change this \
                             cycle={pre_joint_change:.6e})",
                            self.objective_descent_resolution(objective),
                            last_kkt_norm,
                            last_gradient_band,
                            stationarity.max_abs(),
                            stationarity.max_abs() / kkt_scale,
                            stationarity.decoder_max_abs,
                            stationarity.decoder_l2,
                            stationarity.coordinate_max_abs,
                            stationarity.coordinate_l2,
                            stationarity.scaled_max_abs(),
                            stationarity.scaled_max_abs() / parameter_scale,
                            stationarity.decoder_scaled_max_abs,
                            stationarity.coordinate_scaled_max_abs,
                        ));
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
            // #2517 — the first-order screen is read in PARAMETER space, not in
            // gradient space. The decoder sweep solves `(G + λS)B = rhs`
            // exactly, so near the fixed point the block gradient is
            // `(G + λS)·Δ` and `G = Σ_rows φφᵀ` is extensive in rows-per-atom:
            // measured, the raw gradient is 12x-75x the remaining parameter
            // error across two decades of shape, so an absolute (or
            // objective-relative) bound on it is a bound on `m·Δ` that tightens
            // as data is ADDED. Dividing each block's gradient by its own
            // curvature diagonal removes that factor. It does not recover the
            // Newton step (#2933 F08): coupled weakly curved directions are
            // invisible to a diagonal, and a slowly contracting alternation
            // moves by `(1 − ρ)` of its remaining error per cycle. These limbs
            // only decide when the exact Newton displacement is worth pricing.
            let mut candidate = objective_recurred
                && stationarity.first_order_screen(kkt_scale, parameter_scale, tolerance)
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
                    && stationarity.first_order_screen(kkt_scale, parameter_scale, tolerance)
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
                        // #2576: the polish only has to descend. Adoption needs a measured
                        // decrease on the actual objective and resets the recurrence, so a
                        // polish that descended without recurring is still a valid
                        // comparison; only an objective that cannot be evaluated rejects
                        // the move. At the derived tolerance, job 604148 (3cf917ddf, 3000x48
                        // chart) logged 20 proposals as unpolishable; its plateau proposals'
                        // polishes did not recur within 200 cycles, at relative coordinate
                        // KKT ~1e-6.
                        let polish = moved.solve_coordinates_fixed_decoder(
                            target,
                            ard_precisions,
                            tolerance,
                            trust_radius,
                        );
                        if let Err(error) = &polish {
                            log::info!(
                                "support move polish did not recur at cycle {iteration}; \
                                 comparing the objective it reached: {error}"
                            );
                        }
                        match moved.penalized_objective(target, lambda_smooth, ard_precisions) {
                            Err(error) => {
                                log::info!(
                                    "support move unevaluable at cycle {iteration}, \
                                     rejected: {error}"
                                );
                            }
                            Ok(after) => {
                                if objective - after > self.objective_descent_resolution(objective) {
                                    log::info!(
                                        "support move accepted at cycle {iteration}: objective \
                                         {objective:.6e} -> {after:.6e}"
                                    );
                                    *self = moved;
                                    self.reconstruct_into(&mut fitted_state)?;
                                    plateau_window_start = iteration;
                                    objective_at_window_start = after;
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
                            objective_at_window_start = escaped_objective;
                            // A saddle escape starts a new basin; displacements priced
                            // before it do not bound the ones after it.
                            last_newton_displacement = None;
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
                // #2933 F08 — the certificate. Every limb above is a schedule: the
                // objective recurrence is relative to an objective that can carry
                // any additive constant, the state recurrence sees `(1 − ρ)` of the
                // remaining error of a slowly contracting alternation, and the
                // diagonal-scaled residual cannot see coupled weakly curved
                // directions. The exact Newton displacement is the first-order
                // distance to the stationary point, so the state is returned only
                // when every parameter's is within tolerance of the iterate scale.
                let (newton_displacement, newton_direction) =
                    self.exact_newton_solve(target, lambda_smooth, ard_precisions)?;
                if !newton_displacement.certifies(parameter_scale, tolerance) {
                    // A Newton step from inside the basin contracts the displacement.
                    // One that does not, under the same discrete support, is a state
                    // this iteration cannot bring to a certifiable stationary point,
                    // and refusing now spends no further exact solves on it.
                    let support = self.support_fingerprint();
                    if let Some((previous, _)) = last_newton_displacement
                        .as_ref()
                        .filter(|(_, previous_support)| *previous_support == support)
                    {
                        if !(newton_displacement.max_abs() < *previous) {
                            return Err(format!(
                                "SaeSupportSparseTerm::solve_fixed_point: exact Newton displacement \
                                 {:.6e} did not contract from {previous:.6e} after a Newton step \
                                 (decoder {:.6e}, coordinate {:.6e}; bound {:.6e} = tolerance \
                                 {tolerance:.6e} x scale {parameter_scale:.6e})",
                                newton_displacement.max_abs(),
                                newton_displacement.decoder_max_abs,
                                newton_displacement.coordinate_max_abs,
                                tolerance * parameter_scale,
                            ));
                        }
                    }
                    last_newton_displacement = Some((newton_displacement.max_abs(), support));
                    match self.exact_newton_step(
                        target,
                        lambda_smooth,
                        ard_precisions,
                        objective,
                        &newton_direction,
                        newton_displacement.decrement_sq,
                        &mut joint_snapshot,
                        &mut joint_scaled_step,
                        &mut trial_fitted,
                    )? {
                        Some(stepped_objective) => {
                            log::info!(
                                "support fixed-point cycle {iteration}: exact Newton displacement \
                                 {:.3e} (decoder {:.3e}, coordinate {:.3e}) exceeds {:.3e} while \
                                 the screen passed; Newton step installed, objective \
                                 {objective:.6e} -> {stepped_objective:.6e}",
                                newton_displacement.max_abs(),
                                newton_displacement.decoder_max_abs,
                                newton_displacement.coordinate_max_abs,
                                tolerance * parameter_scale,
                            );
                            self.reconstruct_into(&mut fitted_state)?;
                            accelerator.reset();
                            taken_step.clear();
                            taken_step.resize(self.coordinate_state_len(), 0.0);
                            last_objective = None;
                            previous_candidate = false;
                            objective_at_window_start = stepped_objective;
                            continue;
                        }
                        None => {
                            return Err(format!(
                                "SaeSupportSparseTerm::solve_fixed_point: exact Newton displacement \
                                 {:.6e} (decoder {:.6e}, coordinate {:.6e}) exceeds the bound \
                                 {:.6e}, and the Newton step neither descends (decrement² \
                                 {:.6e}) nor avoids a resolved objective increase at any scale",
                                newton_displacement.max_abs(),
                                newton_displacement.decoder_max_abs,
                                newton_displacement.coordinate_max_abs,
                                tolerance * parameter_scale,
                                newton_displacement.decrement_sq,
                            ));
                        }
                    }
                }
                log::info!(
                    "support fixed-point cycle {iteration}: raw KKT max={:.3e} rel={:.3e} \
                     diagonal-scaled max={:.3e} rel={:.3e} exact Newton displacement={:.3e} \
                     rel={:.3e} max_change={:.3e} objective={:.6e} \
                     anderson_accepted={accepted_extrapolations} joint_accepted={joint_accepted}",
                    stationarity.max_abs(),
                    stationarity.max_abs() / kkt_scale,
                    stationarity.scaled_max_abs(),
                    stationarity.scaled_max_abs() / parameter_scale,
                    newton_displacement.max_abs(),
                    newton_displacement.max_abs() / parameter_scale,
                    max_change,
                    objective
                );
                return Ok(SaeSupportFixedPointReport {
                    iterations: iteration,
                    objective,
                    stationarity,
                    newton_displacement,
                    max_recurrence_change: max_change,
                    recurred: true,
                });
            }
            previous_candidate = candidate;

            if iteration == 1 {
                // The baseline the first plateau test compares against; an
                // infinite sentinel here would make the trigger unsatisfiable.
                objective_at_window_start = objective;
            }
            let window_elapsed = iteration >= plateau_window_start + 25;
            let plateau = window_elapsed && objective > objective_at_window_start * (1.0 - 5.0e-3);
            if window_elapsed && !plateau {
                // #2576: the window made progress, so the next test starts here.
                // Held at the last proposal instead, the baseline stays at the
                // cycle-1 objective until a proposal is made, and the scheme is
                // monotone: once the opening cycles lower the objective by more
                // than 0.5% this test can never pass again, and no plateau move
                // is proposed for the rest of the solve. Job 557966 (zoo_micro
                // 3000x48, K=60, s=4) logged none in 266 cycles; its first
                // support move, at the coupled phase's certificate (cycle 239),
                // took the objective from 1.632347e3 to 1.405881e3.
                plateau_window_start = iteration;
                objective_at_window_start = objective;
            }
            if plateau {
                let support_k = match self.assignment.mode() {
                    AssignmentMode::TopK { k } => k,
                    _ => 0,
                };
                if support_k > 0 {
                    plateau_window_start = iteration;
                    objective_at_window_start = objective;
                    // A proposal whose objective cannot be evaluated is a REJECTED
                    // proposal, never a dead fit: the incumbent is untouched, so
                    // erroring out here would discard a healthy model over a
                    // speculative move (the exact discard shape this lane keeps
                    // re-finding). As at the certificate, the polish only has to
                    // descend (#2576): a polish that did not recur is still compared.
                    let mut moved =
                        self.reroute_fixed_decoder_ard(target, support_k, 0, ard_precisions)?;
                    moved.set_decoder_fista_passes(self.decoder_fista_passes);
                    if self.has_same_support_as(&moved) {
                        log::info!(
                            "plateau support move at cycle {iteration} retained every discrete \
                             support; treating it as the no-op it is"
                        );
                    } else {
                        let polish = moved.solve_coordinates_fixed_decoder(
                            target,
                            ard_precisions,
                            tolerance,
                            trust_radius,
                        );
                        if let Err(error) = &polish {
                            log::info!(
                                "plateau support move polish did not recur at cycle {iteration}; \
                                 comparing the objective it reached: {error}"
                            );
                        }
                        match moved.penalized_objective(target, lambda_smooth, ard_precisions) {
                            Err(error) => {
                                // fall through to the normal cycle tail: the
                                // accelerator bookkeeping must see every cycle.
                                log::info!(
                                    "plateau support move unevaluable at cycle {iteration}: {error}"
                                );
                            }
                            Ok(after) => {
                                if objective - after > self.objective_descent_resolution(objective) {
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
                    if objective - extrapolated > self.objective_descent_resolution(objective) {
                        accepted_extrapolations += 1;
                        end_state_held = false;
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
            // The state the next cycle's progress rule compares against (see PHASE).
            if end_state_held {
                last_kkt_norm = Some(stationarity.decoder_l2.hypot(stationarity.coordinate_l2));
                last_gradient_band = gradient_band;
            } else {
                last_kkt_norm = None;
                last_gradient_band = None;
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
            // #2576: whether the coupled phase's terminal motion lies along the
            // Euclidean atoms' affine gauge orbits.
            if joint_armed && log::log_enabled!(log::Level::Info) {
                match self.euclidean_affine_motion_share(&cycle_start, &cycle_residual) {
                    (share, Some((atom, atom_share, energy))) => log::info!(
                        "support fixed-point cycle {iteration}: Euclidean line coordinate motion \
                         affine share={share:.3}; largest-motion atom {atom} share={atom_share:.3} \
                         energy={energy:.3e}"
                    ),
                    (_, None) => log::info!(
                        "support fixed-point cycle {iteration}: no Euclidean line atom moved"
                    ),
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "support_term_tests.rs"]
mod tests;
