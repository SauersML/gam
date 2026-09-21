//! Joint fit of a manifold parameter decomposition (#2951).
//!
//! The loop is designed on the issue (comments 5716713394 and 5717011462):
//! * an anchored finite-intervention objective at fixed structure;
//! * conditionally Gaussian coefficient blocks marginalized exactly;
//! * a smooth outer criterion over labels, weights and the non-Gaussian blocks;
//! * witnesses from the separation oracle;
//! * structural proposals accepted on decoded code under a fidelity check.
//!
//! This file assembles those pieces and owns no numerical primitive. The anchor
//! belongs to `lift`, the field and its pullbacks to `field`, and REML to gam-solve.
//!
//! # Conditionally Gaussian coefficient block
//!
//! Take one parameter family whose basis matrices are in product form
//! `B_j = L_j R_jᵀ`, with `L_j` of shape `d_out × r_j`, `R_j` of shape
//! `d_in × r_j`, and `j = 1..K`. Hold the labels, weights and right factors fixed.
//! Under the residual anchor, readout row `r` at mask `m_r` on the current
//! intervened input `h_r` is
//!
//! ```text
//! y_r = m_Δ Θ_* h_r + Σ_j β_j(m_r) L_j R_jᵀ h_r,     β(m) = Σ_c (m_c − m_Δ) v_c.
//! ```
//!
//! So at fixed `(z, w, R, m, h)` the row is linear in the left factors. Output
//! coordinate `o` reads the design row whose block `j` is `β_j(m_r) R_jᵀ h_r`, at
//! columns `offset_j + a` with `offset_j = Σ_{k<j} r_k`. The function-space penalty
//!
//! ```text
//! Σ_jk S_jk ⟨L_j R_jᵀ, L_k R_kᵀ⟩_F = Σ_o Σ_jk S_jk L_{j,o}ᵀ (R_jᵀ R_k) L_{k,o}
//! ```
//!
//! is the same quadratic form for every output coordinate, with blocks
//! `S_jk R_jᵀ R_k`. The output coordinates are coordinates of one vector-valued
//! observation, so the block is exactly the shared-dispersion multi-response,
//! multi-penalty Gaussian REML problem of [`GaussianRemlMultiPenaltyProblem`].
//! Each field penalty (the energy and the null-space form) is assembled the same
//! way and keeps its own smoothing strength. The block declares a zero null space,
//! so a penalty set that leaves some direction free is refused (SPEC 12/14). The
//! posterior mean and smoothing strengths come from that owner's certified fit and
//! are never refitted here. A dense basis matrix is the case `R_j = I_{d_in}`: the design row
//! is `β(m_r) ⊗ h_r` and the penalty is `S ⊗ I_{d_in}`.
//!
//! The block's criterion `V` is differentiated by that owner too: its envelope
//! cotangents in the design, the responses and each assembled penalty
//! (`data_gradient`, `penalty_gradient`) are pulled back here, through the design's
//! exact adjoint and the penalties' chain to the right factors, by
//! [`gaussian_block_cotangents`].
//!
//! # Rows that are refused
//!
//! A row whose declared mask has `m_c = m_Δ` for every component executes `m_Δ Θ_*`
//! at every parameter value:
//! * at all-on it is the bit-identity guard (P16);
//! * with the residual removed and every component off it is the zero tensor.
//!
//! Neither observes anything this block or the outer criterion moves, so such a
//! row is refused. The refusal keys on the declared mask values, never on a
//! computed moment. `β(m_r)` can vanish by cancellation at the current labels and
//! weights while its derivatives in them do not, and such a row is admitted.
//!
//! # Structural proposals
//!
//! A proposal (share, split, refine, reduce, expose) is decided on decoded
//! artifacts only:
//! * the decoded reference must meet the declared tolerance;
//! * the candidate's fidelity status over the mask domain must be a bound on the
//!   supremum, not an estimate, and must not refute `sup d ≤ ε`;
//! * the decoded candidate must meet the tolerance with a strictly shorter code.
//!
//! Fidelity alone never accepts. An operator that interpolates the teacher still
//! loses when its code is longer.

use std::fmt;
use std::num::NonZeroU64;

use super::codec::code_saving_at_proven_fidelity;
use super::precision::{DecodedFidelity, DeclaredPrecision};
use super::supports::{EvidenceStatus, Extremum};
use gam_linalg::matrix::{array2_bits_fingerprint, dense_rowwise_kronecker};
use gam_linalg::roundoff::accumulation_growth;
use gam_solve::estimate::EstimationError;
use gam_solve::gaussian_reml_multi_penalty::{
    GaussianRemlMultiPenaltyDataGradientOutcome, GaussianRemlMultiPenaltyFit,
    GaussianRemlMultiPenaltyPenaltyGradientOutcome, GaussianRemlMultiPenaltyProblem,
    GaussianRemlMultiPenaltyRhoPlacement,
};
use ndarray::{Array2, ArrayView1, ArrayView2, s};

/// The rows of one conditionally Gaussian coefficient block.
#[derive(Clone, Copy, Debug)]
pub struct GaussianBlockRows<'a> {
    /// The declared component masks `m_c` of each row, `n × C`.
    pub component_masks: ArrayView2<'a, f64>,
    /// The declared residual mask `m_Δ` of each row, length `n`.
    pub residual_masks: ArrayView1<'a, f64>,
    /// The anchor moment `β(m_r) = Σ_c (m_c − m_Δ) v_c` of each row, `n × K`, read
    /// from the anchor.
    pub moments: ArrayView2<'a, f64>,
    /// The current intervened input `h_r` of the edited tensor, `n × d_in`: the
    /// input under every upstream edit of the row, never a cached clean input.
    pub inputs: ArrayView2<'a, f64>,
    /// The target readout minus the anchored native term `m_Δ Θ_* h_r`, `n × d_out`.
    pub responses: ArrayView2<'a, f64>,
}

/// A fitted conditionally Gaussian coefficient block.
#[derive(Clone, Debug)]
pub struct GaussianBlockFit {
    /// Posterior-mean left factors `L_j`, each `d_out × r_j`.
    pub left_factors: Vec<Array2<f64>>,
    /// The reduced REML problem, kept for the envelope derivatives at this fit.
    pub problem: GaussianRemlMultiPenaltyProblem,
    /// The certified multi-penalty REML fit of the block. Its coefficients are in
    /// design order (row `offset_j + a`, one column per output coordinate).
    pub reml: GaussianRemlMultiPenaltyFit,
    /// The `K × K` field penalties the block's penalties were assembled from, in the
    /// REML problem's penalty order: the coefficients of their chain to the right factors.
    /// Private, since nothing re-checks them against the assembled penalties.
    field_penalties: Vec<Array2<f64>>,
    /// Value fingerprints of the right factors the block was built from
    /// ([`array2_bits_fingerprint`]): the penalty chain is exact only at those.
    right_factor_fingerprints: Vec<u64>,
}

/// Why a conditionally Gaussian coefficient block was not fitted.
#[derive(Debug)]
pub enum GaussianBlockError {
    EmptyBlock {
        rows: usize,
        components: usize,
        basis: usize,
        input_dim: usize,
        output_dim: usize,
    },
    RowCountMismatch {
        what: &'static str,
        rows: usize,
        expected: usize,
    },
    RightFactorCount {
        basis: usize,
        right_factors: usize,
    },
    RightFactorShape {
        basis: usize,
        rows: usize,
        rank: usize,
        input_dim: usize,
    },
    /// No field penalty was given, so no prior is declared.
    NoFieldPenalties,
    PenaltyShape {
        penalty: usize,
        basis: usize,
        rows: usize,
        cols: usize,
    },
    /// A field penalty is not exactly symmetric.
    AsymmetricFieldPenalty {
        penalty: usize,
        row: usize,
        col: usize,
    },
    /// An assembled penalty pair differs by more than the derived rounding band of the
    /// products that formed it, so the difference is an assembly error, not rounding.
    AssembledPenaltyAsymmetric {
        penalty: usize,
        row: usize,
        col: usize,
        gap: f64,
        band: f64,
    },
    NonFinite {
        what: &'static str,
    },
    /// The row's declared mask has `m_c = m_Δ` for every component.
    NativeMultipleRow {
        row: usize,
    },
    /// The block's dense state exceeds the host in-core budget. `required_bytes`
    /// is `None` when the count overflows `usize`.
    AdmissionRefused {
        required_bytes: Option<usize>,
        budget_bytes: usize,
    },
    /// A design cotangent's width is not the design's.
    DesignCotangentWidth {
        columns: usize,
        expected: usize,
    },
    /// Right factor `basis` differs in value from the one the fit was built from (or
    /// the count differs, with `basis` the fit's count), so the fit's envelope
    /// derivatives are not taken at it.
    RightFactorsChanged {
        basis: usize,
    },
    Reml(EstimationError),
}

impl fmt::Display for GaussianBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBlock {
                rows,
                components,
                basis,
                input_dim,
                output_dim,
            } => write!(
                f,
                "conditionally Gaussian block refused: it needs positive rows, components, basis \
                 size, input and output dimensions; got n={rows}, C={components}, K={basis}, \
                 d_in={input_dim}, d_out={output_dim}"
            ),
            Self::RowCountMismatch {
                what,
                rows,
                expected,
            } => write!(
                f,
                "conditionally Gaussian block refused: the {what} have {rows} rows but the moments \
                 have {expected}"
            ),
            Self::RightFactorCount {
                basis,
                right_factors,
            } => write!(
                f,
                "conditionally Gaussian block refused: {right_factors} right factors for {basis} \
                 basis matrices"
            ),
            Self::RightFactorShape {
                basis,
                rows,
                rank,
                input_dim,
            } => write!(
                f,
                "conditionally Gaussian block refused: right factor {basis} is {rows}×{rank}; it \
                 needs {input_dim} rows and a positive rank"
            ),
            Self::NoFieldPenalties => write!(
                f,
                "conditionally Gaussian block refused: at least one field penalty is required"
            ),
            Self::PenaltyShape {
                penalty,
                basis,
                rows,
                cols,
            } => write!(
                f,
                "conditionally Gaussian block refused: field penalty {penalty} must be \
                 {basis}×{basis} to match the basis; got {rows}×{cols}"
            ),
            Self::AsymmetricFieldPenalty { penalty, row, col } => write!(
                f,
                "conditionally Gaussian block refused: field penalty {penalty} is not exactly \
                 symmetric at ({row}, {col})"
            ),
            Self::AssembledPenaltyAsymmetric {
                penalty,
                row,
                col,
                gap,
                band,
            } => write!(
                f,
                "conditionally Gaussian block refused: assembled penalty {penalty} differs from its \
                 transpose at ({row}, {col}) by {gap:e}, beyond its derived rounding band {band:e}"
            ),
            Self::NonFinite { what } => write!(
                f,
                "conditionally Gaussian block refused: the {what} contain a non-finite value"
            ),
            Self::NativeMultipleRow { row } => write!(
                f,
                "conditionally Gaussian block refused: row {row} declares m_c = m_Delta for every \
                 component, so it executes m_Delta Theta_* whatever the field, labels and weights \
                 are, and observes nothing the fit moves"
            ),
            Self::AdmissionRefused {
                required_bytes: Some(required),
                budget_bytes,
            } => write!(
                f,
                "conditionally Gaussian block refused: its dense state needs at least {required} \
                 bytes, above the host in-core budget of {budget_bytes} bytes"
            ),
            Self::AdmissionRefused {
                required_bytes: None,
                budget_bytes,
            } => write!(
                f,
                "conditionally Gaussian block refused: its dense byte count overflows usize \
                 (host in-core budget {budget_bytes} bytes)"
            ),
            Self::DesignCotangentWidth { columns, expected } => write!(
                f,
                "conditionally Gaussian block refused: the design cotangent has {columns} columns \
                 but the design has {expected}"
            ),
            Self::RightFactorsChanged { basis } => write!(
                f,
                "conditionally Gaussian block refused: right factor {basis} is not the one the fit \
                 was built from, so the fit is not converged at it"
            ),
            Self::Reml(error) => write!(f, "conditionally Gaussian block REML failed: {error}"),
        }
    }
}

impl std::error::Error for GaussianBlockError {}

/// Fits one conditionally Gaussian coefficient block exactly.
///
/// * `right_factors` holds the fixed `R_j`, `d_in × r_j`, one per basis matrix.
/// * `field_penalties` holds the `K × K` function-space penalties of the family's
///   field, e.g. the energy `S` and the null-space form over the functions `S` leaves
///   free. Each gets its own smoothing strength.
///
/// Refuses:
/// * an empty or mis-shaped block, an empty penalty list, or non-finite input;
/// * a row whose declared mask is a native multiple (see the module docs);
/// * a block whose dense state exceeds the host in-core budget.
///
/// The block declares a zero null space. So the REML owner refuses a penalty set
/// whose summed null space it resolves as non-empty, since a direction of the left
/// factors with a flat prior can never shrink to no effect (SPEC 12/14). That refusal
/// and every other one from the owner are returned unchanged.
pub fn fit_gaussian_coefficient_block(
    rows: GaussianBlockRows<'_>,
    right_factors: &[ArrayView2<'_, f64>],
    field_penalties: &[ArrayView2<'_, f64>],
) -> Result<GaussianBlockFit, GaussianBlockError> {
    let n = rows.moments.nrows();
    let components = rows.component_masks.ncols();
    let basis = rows.moments.ncols();
    let input_dim = rows.inputs.ncols();
    let output_dim = rows.responses.ncols();
    if n == 0 || components == 0 || basis == 0 || input_dim == 0 || output_dim == 0 {
        return Err(GaussianBlockError::EmptyBlock {
            rows: n,
            components,
            basis,
            input_dim,
            output_dim,
        });
    }
    for (what, count) in [
        ("component masks", rows.component_masks.nrows()),
        ("residual masks", rows.residual_masks.len()),
        ("inputs", rows.inputs.nrows()),
        ("responses", rows.responses.nrows()),
    ] {
        if count != n {
            return Err(GaussianBlockError::RowCountMismatch {
                what,
                rows: count,
                expected: n,
            });
        }
    }
    if right_factors.len() != basis {
        return Err(GaussianBlockError::RightFactorCount {
            basis,
            right_factors: right_factors.len(),
        });
    }
    for (index, factor) in right_factors.iter().enumerate() {
        if factor.nrows() != input_dim || factor.ncols() == 0 {
            return Err(GaussianBlockError::RightFactorShape {
                basis: index,
                rows: factor.nrows(),
                rank: factor.ncols(),
                input_dim,
            });
        }
    }
    if field_penalties.is_empty() {
        return Err(GaussianBlockError::NoFieldPenalties);
    }
    for (penalty, field_penalty) in field_penalties.iter().enumerate() {
        if field_penalty.nrows() != basis || field_penalty.ncols() != basis {
            return Err(GaussianBlockError::PenaltyShape {
                penalty,
                basis,
                rows: field_penalty.nrows(),
                cols: field_penalty.ncols(),
            });
        }
        if field_penalty.iter().any(|value| !value.is_finite()) {
            return Err(GaussianBlockError::NonFinite {
                what: "field penalty entries",
            });
        }
        for row in 0..basis {
            for col in (row + 1)..basis {
                if field_penalty[[row, col]] != field_penalty[[col, row]] {
                    return Err(GaussianBlockError::AsymmetricFieldPenalty { penalty, row, col });
                }
            }
        }
    }
    for (what, values) in [
        ("component masks", rows.component_masks),
        ("moments", rows.moments),
        ("inputs", rows.inputs),
        ("responses", rows.responses),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(GaussianBlockError::NonFinite { what });
        }
    }
    if rows.residual_masks.iter().any(|value| !value.is_finite()) {
        return Err(GaussianBlockError::NonFinite {
            what: "residual masks",
        });
    }
    if right_factors
        .iter()
        .any(|factor| factor.iter().any(|value| !value.is_finite()))
    {
        return Err(GaussianBlockError::NonFinite {
            what: "right factors",
        });
    }
    for row in 0..n {
        let residual = rows.residual_masks[row];
        if rows
            .component_masks
            .row(row)
            .iter()
            .all(|&mask| mask == residual)
        {
            return Err(GaussianBlockError::NativeMultipleRow { row });
        }
    }
    let ranks: Vec<usize> = right_factors.iter().map(|factor| factor.ncols()).collect();
    admit_dense_block(
        n,
        &ranks,
        field_penalties.len(),
        output_dim,
        crate::manifold::sae_host_in_core_budget_bytes().0,
    )?;
    let design = block_design(rows.moments, rows.inputs, right_factors);
    let penalties = field_penalties
        .iter()
        .enumerate()
        .map(|(penalty, field_penalty)| block_penalty(penalty, *field_penalty, right_factors))
        .collect::<Result<Vec<Array2<f64>>, GaussianBlockError>>()?;
    let problem = GaussianRemlMultiPenaltyProblem::new(design.view(), rows.responses, &penalties, 0)
        .map_err(GaussianBlockError::Reml)?;
    let reml = problem.fit(None).map_err(GaussianBlockError::Reml)?;
    let left_factors = left_factors_from_design_order(reml.coefficients.view(), right_factors);
    Ok(GaussianBlockFit {
        left_factors,
        problem,
        reml,
        field_penalties: field_penalties.iter().map(|penalty| penalty.to_owned()).collect(),
        right_factor_fingerprints: right_factors.iter().map(array2_bits_fingerprint).collect(),
    })
}

/// `[0, r_0, r_0 + r_1, …, Σ_j r_j]`: where each basis matrix's block starts.
fn design_offsets(right_factors: &[ArrayView2<'_, f64>]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(right_factors.len() + 1);
    let mut total = 0;
    offsets.push(total);
    for factor in right_factors {
        total += factor.ncols();
        offsets.push(total);
    }
    offsets
}

/// The design in design order: block `j` of row `r` is `β_j(m_r) R_jᵀ h_r`.
fn block_design(
    moments: ArrayView2<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    right_factors: &[ArrayView2<'_, f64>],
) -> Array2<f64> {
    let offsets = design_offsets(right_factors);
    let mut design = Array2::<f64>::zeros((moments.nrows(), offsets[right_factors.len()]));
    for (j, factor) in right_factors.iter().enumerate() {
        let projected = inputs.dot(factor);
        let block = dense_rowwise_kronecker(moments.slice(s![.., j..j + 1]), projected.view());
        design
            .slice_mut(s![.., offsets[j]..offsets[j + 1]])
            .assign(&block);
    }
    design
}

/// The penalty in design order: block `(j, k)` is `S_jk R_jᵀ R_k`, the quadratic
/// form `Σ_jk S_jk ⟨L_j R_jᵀ, L_k R_kᵀ⟩_F` of one output coordinate.
///
/// `S` is exactly symmetric (checked by the caller), so blocks `(j, k)` and `(k, j)`
/// are the same numbers reached through two separate Gram products. Each entry sums
/// `d_in` products and is then scaled once, so each computed side is within
/// `γ_{d_in+1}` times its absolute monomial sum. [`symmetrized_within_band`] refuses a
/// wider gap and averages the rest. `penalty` names the penalty in a refusal.
fn block_penalty(
    penalty: usize,
    field_penalty: ArrayView2<'_, f64>,
    right_factors: &[ArrayView2<'_, f64>],
) -> Result<Array2<f64>, GaussianBlockError> {
    let offsets = design_offsets(right_factors);
    let columns = offsets[right_factors.len()];
    let mut assembled = Array2::<f64>::zeros((columns, columns));
    let mut absolute = Array2::<f64>::zeros((columns, columns));
    for (j, left) in right_factors.iter().enumerate() {
        let left_absolute = left.mapv(f64::abs);
        for (k, right) in right_factors.iter().enumerate() {
            let strength = field_penalty[[j, k]];
            assembled
                .slice_mut(s![offsets[j]..offsets[j + 1], offsets[k]..offsets[k + 1]])
                .assign(&left.t().dot(right).mapv(|value| strength * value));
            absolute
                .slice_mut(s![offsets[j]..offsets[j + 1], offsets[k]..offsets[k + 1]])
                .assign(&left_absolute.t().dot(&right.mapv(f64::abs)).mapv(|value| strength.abs() * value));
        }
    }
    let input_dim = right_factors.first().map_or(0, |factor| factor.nrows());
    symmetrized_within_band(penalty, assembled, &absolute, accumulation_growth(input_dim + 1))
}

/// Averages `assembled` with its transpose, after refusing any pair whose gap exceeds
/// `growth` times the two sides' absolute monomial sums. Averaging then removes only
/// rounding and cannot hide an assembly error. An exactly symmetric pair is unchanged,
/// since (a + a)·½ = a.
fn symmetrized_within_band(
    penalty: usize,
    assembled: Array2<f64>,
    absolute: &Array2<f64>,
    growth: f64,
) -> Result<Array2<f64>, GaussianBlockError> {
    let columns = assembled.nrows();
    for row in 0..columns {
        for col in (row + 1)..columns {
            let gap = (assembled[[row, col]] - assembled[[col, row]]).abs();
            let band = growth * (absolute[[row, col]] + absolute[[col, row]]);
            if !(gap <= band) {
                return Err(GaussianBlockError::AssembledPenaltyAsymmetric {
                    penalty,
                    row,
                    col,
                    gap,
                    band,
                });
            }
        }
    }
    Ok((&assembled + &assembled.t()) * 0.5)
}

/// Reads design-order coefficients (row `offset_j + a`, column `o`) as `L_j[[o, a]]`.
fn left_factors_from_design_order(
    coefficients: ArrayView2<'_, f64>,
    right_factors: &[ArrayView2<'_, f64>],
) -> Vec<Array2<f64>> {
    let output_dim = coefficients.ncols();
    let offsets = design_offsets(right_factors);
    right_factors
        .iter()
        .enumerate()
        .map(|(j, factor)| {
            Array2::from_shape_fn((output_dim, factor.ncols()), |(o, a)| {
                coefficients[[offsets[j] + a, o]]
            })
        })
        .collect()
}

/// Cotangents of the block's design with respect to what built it.
///
/// For a scalar criterion `V(X)` with `G = ∂V/∂X` (`n × p`, design order), these
/// are `∂V/∂β` (`n × K`), `∂V/∂h` (`n × d_in`) and `∂V/∂R_j` (`d_in × r_j`) through
/// the design alone. A criterion whose penalties also depend on the right factors
/// (`S_jk R_jᵀ R_k`) adds that penalty chain; [`gaussian_block_cotangents`] adds it
/// for the block's REML criterion.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDesignCotangents {
    /// `∂V/∂β[r, j] = Σ_a G[r, offset_j + a] (R_jᵀ h_r)_a`.
    pub moments: Array2<f64>,
    /// `∂V/∂h[r, i] = Σ_j β[r, j] Σ_a G[r, offset_j + a] R_j[i, a]`.
    pub inputs: Array2<f64>,
    /// `∂V/∂R_j[i, a] = Σ_r β[r, j] h[r, i] G[r, offset_j + a]`.
    pub right_factors: Vec<Array2<f64>>,
}

/// The exact adjoint of the block's design. It pulls a design cotangent (e.g. the
/// REML owner's `data_gradient`) back to the moments, the inputs and the right
/// factors.
///
/// Block `j` of design row `r` is `β_j(m_r) R_jᵀ h_r`. That is linear in `β`, in
/// `h` and in `R_j` separately, so each pullback is the transpose of one linear map.
/// Labels and weights reach `β` through the anchor moment, and upstream parameters
/// reach `h` through the executed network. Those chains belong to their owners.
pub fn block_design_adjoint(
    design_cotangent: ArrayView2<'_, f64>,
    moments: ArrayView2<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    right_factors: &[ArrayView2<'_, f64>],
) -> Result<BlockDesignCotangents, GaussianBlockError> {
    let n = moments.nrows();
    let basis = moments.ncols();
    let input_dim = inputs.ncols();
    for (what, count) in [
        ("inputs", inputs.nrows()),
        ("design cotangent", design_cotangent.nrows()),
    ] {
        if count != n {
            return Err(GaussianBlockError::RowCountMismatch {
                what,
                rows: count,
                expected: n,
            });
        }
    }
    if right_factors.len() != basis {
        return Err(GaussianBlockError::RightFactorCount {
            basis,
            right_factors: right_factors.len(),
        });
    }
    for (index, factor) in right_factors.iter().enumerate() {
        if factor.nrows() != input_dim || factor.ncols() == 0 {
            return Err(GaussianBlockError::RightFactorShape {
                basis: index,
                rows: factor.nrows(),
                rank: factor.ncols(),
                input_dim,
            });
        }
    }
    let offsets = design_offsets(right_factors);
    if design_cotangent.ncols() != offsets[basis] {
        return Err(GaussianBlockError::DesignCotangentWidth {
            columns: design_cotangent.ncols(),
            expected: offsets[basis],
        });
    }
    for (what, values) in [
        ("design cotangent", design_cotangent),
        ("moments", moments),
        ("inputs", inputs),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(GaussianBlockError::NonFinite { what });
        }
    }
    if right_factors
        .iter()
        .any(|factor| factor.iter().any(|value| !value.is_finite()))
    {
        return Err(GaussianBlockError::NonFinite {
            what: "right factors",
        });
    }
    let mut moment_cotangents = Array2::<f64>::zeros((n, basis));
    let mut input_cotangents = Array2::<f64>::zeros((n, input_dim));
    let mut right_cotangents = Vec::with_capacity(basis);
    for (j, factor) in right_factors.iter().enumerate() {
        let block = design_cotangent.slice(s![.., offsets[j]..offsets[j + 1]]);
        let projected = inputs.dot(factor);
        for row in 0..n {
            moment_cotangents[[row, j]] = block.row(row).dot(&projected.row(row));
        }
        let scaled =
            Array2::from_shape_fn(block.dim(), |(row, a)| moments[[row, j]] * block[[row, a]]);
        input_cotangents += &scaled.dot(&factor.t());
        right_cotangents.push(inputs.t().dot(&scaled));
    }
    Ok(BlockDesignCotangents {
        moments: moment_cotangents,
        inputs: input_cotangents,
        right_factors: right_cotangents,
    })
}

/// Cotangents of the block's REML criterion `V` at its certified fit, with respect to
/// everything that built the block.
///
/// `V` reads the moments `β` and the inputs `h` through the design only, and the
/// responses `Y` directly. It reads each right factor `R_j` twice: through the design
/// (`β_j R_jᵀ h_r`) and through every assembled penalty `P_k`, whose block `(j, l)` is
/// `S^k_jl R_jᵀ R_l`. With `G_k = ∂V/∂P_k` symmetric, `dV = Σ_k tr(G_k dP_k)` gives the
/// penalty leg `∂V/∂R_j = Σ_k Σ_l (S^k_jl + S^k_lj) R_l G_k[l, j]`, where `G_k[l, j]` is
/// the `(l, j)` block of `G_k` in design order.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianBlockCotangents {
    /// `∂V/∂β`, `n × K`.
    pub moments: Array2<f64>,
    /// `∂V/∂h`, `n × d_in`.
    pub inputs: Array2<f64>,
    /// `∂V/∂R_j`, `d_in × r_j`: the design leg plus the penalty leg.
    pub right_factors: Vec<Array2<f64>>,
    /// `∂V/∂Y`, `n × d_out`.
    pub responses: Array2<f64>,
}

/// The block's cotangents where the REML owner's envelope forms are the total
/// derivative, or the typed reason they are not.
#[derive(Clone, Debug, PartialEq)]
pub enum GaussianBlockCotangentOutcome {
    /// Every smoothing strength is interior.
    Interior(GaussianBlockCotangents),
    /// Some smoothing strength is railed or unaudited. The owner's domain edge moves
    /// with the design and the penalties, so its envelope forms miss that edge's motion
    /// and none is returned.
    RhoAtDomainBound {
        placement: Vec<GaussianRemlMultiPenaltyRhoPlacement>,
    },
}

/// `∂V/∂(β, h, R, Y)` at the block's certified fit, for the rows and right factors it
/// was built from.
///
/// The REML owner differentiates `V`: `data_gradient` gives `∂V/∂X` and `∂V/∂Y`, and
/// `penalty_gradient` gives each `∂V/∂P_k`. This function only pulls them back:
/// `∂V/∂X` through [`block_design_adjoint`], and each `∂V/∂P_k` through the penalty
/// chain of [`GaussianBlockCotangents`].
///
/// Refuses, typed:
/// * right factors whose values differ from the fit's (`RightFactorsChanged`);
/// * moments, inputs or responses whose design or response differs from the fit's
///   (the owner's refusal, as `Reml`).
///
/// A railed or unaudited smoothing strength returns
/// [`GaussianBlockCotangentOutcome::RhoAtDomainBound`], never a partial gradient.
pub fn gaussian_block_cotangents(
    fit: &GaussianBlockFit,
    rows: GaussianBlockRows<'_>,
    right_factors: &[ArrayView2<'_, f64>],
) -> Result<GaussianBlockCotangentOutcome, GaussianBlockError> {
    if right_factors.len() != fit.right_factor_fingerprints.len() {
        return Err(GaussianBlockError::RightFactorsChanged {
            basis: fit.right_factor_fingerprints.len(),
        });
    }
    if let Some(basis) = right_factors
        .iter()
        .zip(fit.right_factor_fingerprints.iter())
        .position(|(factor, &fingerprint)| array2_bits_fingerprint(factor) != fingerprint)
    {
        return Err(GaussianBlockError::RightFactorsChanged { basis });
    }
    // The shapes the design is rebuilt at; the owner then refuses any value that differs from the fit's.
    let (n, basis) = rows.moments.dim();
    if basis != right_factors.len() {
        return Err(GaussianBlockError::RightFactorCount {
            basis,
            right_factors: right_factors.len(),
        });
    }
    if rows.inputs.nrows() != n {
        return Err(GaussianBlockError::RowCountMismatch {
            what: "inputs",
            rows: rows.inputs.nrows(),
            expected: n,
        });
    }
    if let Some(index) = right_factors
        .iter()
        .position(|factor| factor.nrows() != rows.inputs.ncols())
    {
        return Err(GaussianBlockError::RightFactorShape {
            basis: index,
            rows: right_factors[index].nrows(),
            rank: right_factors[index].ncols(),
            input_dim: rows.inputs.ncols(),
        });
    }
    let design = block_design(rows.moments, rows.inputs, right_factors);
    let data = match fit
        .problem
        .data_gradient(design.view(), rows.responses, &fit.reml)
        .map_err(GaussianBlockError::Reml)?
    {
        GaussianRemlMultiPenaltyDataGradientOutcome::Interior(gradient) => gradient,
        GaussianRemlMultiPenaltyDataGradientOutcome::RhoAtDomainBound { placement } => {
            return Ok(GaussianBlockCotangentOutcome::RhoAtDomainBound { placement });
        }
    };
    let penalty_gradients = match fit
        .problem
        .penalty_gradient(&fit.reml)
        .map_err(GaussianBlockError::Reml)?
    {
        GaussianRemlMultiPenaltyPenaltyGradientOutcome::Interior { gradients, .. } => gradients,
        GaussianRemlMultiPenaltyPenaltyGradientOutcome::RhoAtDomainBound { placement } => {
            return Ok(GaussianBlockCotangentOutcome::RhoAtDomainBound { placement });
        }
    };
    let design_leg = block_design_adjoint(data.grad_x.view(), rows.moments, rows.inputs, right_factors)?;
    let offsets = design_offsets(right_factors);
    let mut right = design_leg.right_factors;
    for (field_penalty, gradient) in fit.field_penalties.iter().zip(penalty_gradients.iter()) {
        for (j, total) in right.iter_mut().enumerate() {
            for (l, factor) in right_factors.iter().enumerate() {
                let block = gradient.slice(s![offsets[l]..offsets[l + 1], offsets[j]..offsets[j + 1]]);
                total.scaled_add(field_penalty[[j, l]] + field_penalty[[l, j]], &factor.dot(&block));
            }
        }
    }
    Ok(GaussianBlockCotangentOutcome::Interior(GaussianBlockCotangents {
        moments: design_leg.moments,
        inputs: design_leg.inputs,
        right_factors: right,
        responses: data.grad_y,
    }))
}

/// A lower bound on the block's peak dense state in bytes, or `None` on overflow.
///
/// With `p = Σ_j r_j`, `K` penalties and `m = d_out`, it counts what this file
/// materializes together with what the multi-penalty REML owner holds, as read at
/// eb05457fdd:
/// * here: the design `n·p`, the widest projected input `n·max_j r_j` while it is
///   folded in, and the assembled penalties `K·p²`;
/// * the owner: its penalty copies `K·p²`, the design factor `R`
///   (`min(n, p)·p`), the rotated head `Z` (`min(n, p)·m`), the stacked root, at
///   least `p²`, and the coefficients `p·m`.
///
/// The owner's transient workspace (the penalty roots, the compressed head, QR
/// reflectors) is on top of this. So the bound is necessary, not sufficient: it
/// refuses a block that cannot fit, and it does not certify one that can.
fn dense_block_bytes(rows: usize, ranks: &[usize], penalties: usize, outputs: usize) -> Option<usize> {
    let columns = ranks
        .iter()
        .try_fold(0usize, |total, &rank| total.checked_add(rank))?;
    let widest = ranks.iter().copied().max().unwrap_or(0);
    let squares = columns.checked_mul(columns)?;
    let penalty_squares = penalties.checked_mul(squares)?.checked_mul(2)?;
    let design = rows.checked_mul(columns.checked_add(widest)?)?;
    let reduction = rows.min(columns).checked_mul(columns.checked_add(outputs)?)?;
    let coefficients = columns.checked_mul(outputs)?;
    design
        .checked_add(penalty_squares)?
        .checked_add(reduction)?
        .checked_add(squares)?
        .checked_add(coefficients)?
        .checked_mul(std::mem::size_of::<f64>())
}

fn admit_dense_block(
    rows: usize,
    ranks: &[usize],
    penalties: usize,
    outputs: usize,
    budget_bytes: usize,
) -> Result<(), GaussianBlockError> {
    match dense_block_bytes(rows, ranks, penalties, outputs) {
        Some(required) if required <= budget_bytes => Ok(()),
        required_bytes => Err(GaussianBlockError::AdmissionRefused {
            required_bytes,
            budget_bytes,
        }),
    }
}

/// A structural proposal on the current artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    /// Two families of one shape class and label-manifold type become one field, or
    /// tied uses call one shared body.
    Share,
    /// A component becomes two whose sum is that component at the start.
    Split,
    /// One more basis function or label dimension, starting at its prior mean.
    Refine,
    /// A component, a rank or a basis function is removed, or a tensor returns to its
    /// native primitive.
    Reduce,
    /// A component is born from the residual at a violating witness, or a recovered
    /// structured coordinate becomes a program node.
    Expose,
}

/// An accepted structural proposal.
#[derive(Clone, Debug, PartialEq)]
pub struct ProposalAcceptance<W, D> {
    pub kind: ProposalKind,
    /// `L(reference) − L(candidate)` in bits. Always strictly positive.
    pub saving_bits: i128,
    /// Whether the fidelity status proves `sup d ≤ ε` over its domain. When false, no
    /// violation was proved either, and the status is carried as it was proved.
    pub fidelity_certified: bool,
    /// The decoded candidate's fidelity status over the declared mask domain.
    pub fidelity: EvidenceStatus<W, D>,
}

/// Why a structural proposal was not accepted.
#[derive(Clone, Debug, PartialEq)]
pub enum ProposalRejection<W, D> {
    /// The decoded reference misses the declared tolerance. The declared precision
    /// and tolerance are then inconsistent, so the loop is refused, not only this
    /// proposal.
    ReferenceMissesTolerance(String),
    /// The decoded candidate does not prove it meets the reference's declared tolerance:
    /// its verdict is not `Meets`, or its tolerance differs bitwise.
    CandidateMissesTolerance(String),
    /// The candidate's code is not strictly shorter.
    NoShorterCode { saving_bits: i128 },
    /// The fidelity status proves the supremum exceeds the tolerance.
    FidelityRefuted(EvidenceStatus<W, D>),
    /// A statistical estimate is about a mean, not the supremum the fidelity check
    /// needs (A6).
    EstimateIsNotAFidelityBound(EvidenceStatus<W, D>),
    /// The status brackets an infimum. Bounding an infimum from above says nothing
    /// about the supremum.
    NotASupremum(EvidenceStatus<W, D>),
}

impl<W: fmt::Debug, D: fmt::Debug> fmt::Display for ProposalRejection<W, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReferenceMissesTolerance(message) => write!(
                f,
                "structural loop refused: the decoded reference misses the declared tolerance, so \
                 the declared precision and tolerance are inconsistent ({message})"
            ),
            Self::CandidateMissesTolerance(message) => {
                write!(f, "proposal rejected: the decoded candidate misses the tolerance ({message})")
            }
            Self::NoShorterCode { saving_bits } => write!(
                f,
                "proposal rejected: the candidate's code is not strictly shorter (saving \
                 {saving_bits} bits)"
            ),
            Self::FidelityRefuted(status) => {
                write!(f, "proposal rejected: the fidelity status refutes the tolerance: {status:?}")
            }
            Self::EstimateIsNotAFidelityBound(status) => write!(
                f,
                "proposal refused: a statistical estimate bounds no supremum: {status:?}"
            ),
            Self::NotASupremum(status) => write!(
                f,
                "proposal refused: the fidelity status brackets an infimum, not the supremum: \
                 {status:?}"
            ),
        }
    }
}

impl<W: fmt::Debug, D: fmt::Debug> std::error::Error for ProposalRejection<W, D> {}

/// Decides one structural proposal.
///
/// `reference` and `candidate` pair each decoded artifact's exact code length in bits
/// with its decoded distortion evidence under the declared tolerance, as the precision
/// owner states it (`precision::decode_then_evaluate`). `fidelity` is the separation
/// oracle's status for `sup d` over the declared mask domain, on the decoded candidate.
/// It is a different quantity from the decoded distortion, read at the reference's
/// declared tolerance.
///
/// The rule reuses the owners' predicates (`code_saving_at_proven_fidelity`,
/// `EvidenceStatus::refutes_at_most` and `certifies_at_most`) and writes no second
/// comparison:
/// 1. The decoded reference must prove it meets its tolerance; otherwise the loop is
///    refused.
/// 2. An estimate or an infimum bracket is refused. A status that refutes
///    `sup d ≤ tolerance` rejects the candidate.
/// 3. The decoded candidate must prove it meets the same tolerance, bitwise, and its
///    code must be strictly shorter.
pub fn decide_proposal<W, D, V, E>(
    kind: ProposalKind,
    reference: (u64, &DecodedFidelity<V, E>),
    candidate: (u64, &DecodedFidelity<V, E>),
    fidelity: EvidenceStatus<W, D>,
) -> Result<ProposalAcceptance<W, D>, ProposalRejection<W, D>> {
    code_saving_at_proven_fidelity(reference, reference)
        .map_err(ProposalRejection::ReferenceMissesTolerance)?;
    let tolerance = reference.1.tolerance();
    match fidelity {
        EvidenceStatus::StatisticalEstimate { .. } => {
            return Err(ProposalRejection::EstimateIsNotAFidelityBound(fidelity));
        }
        EvidenceStatus::Unresolved {
            extremum: Extremum::Infimum,
            ..
        } => return Err(ProposalRejection::NotASupremum(fidelity)),
        EvidenceStatus::Exact { .. }
        | EvidenceStatus::UniformBound { .. }
        | EvidenceStatus::Counterexample { .. }
        | EvidenceStatus::Unresolved { .. } => {}
    }
    if fidelity.refutes_at_most(tolerance) {
        return Err(ProposalRejection::FidelityRefuted(fidelity));
    }
    let saving_bits = code_saving_at_proven_fidelity(reference, candidate)
        .map_err(ProposalRejection::CandidateMissesTolerance)?;
    if saving_bits <= 0 {
        return Err(ProposalRejection::NoShorterCode { saving_bits });
    }
    Ok(ProposalAcceptance {
        kind,
        saving_bits,
        fidelity_certified: fidelity.certifies_at_most(tolerance),
        fidelity,
    })
}

/// Whether rows execute the residual `Θ_* − Σ_c P_c`.
///
/// This is one declared value, shared by every row, by the mask generators and by
/// the separation oracle, so all three read the same zonotope:
/// * under `Kept`, `Θ(t) = Θ_* − B q`;
/// * under `Removed`, `Θ(t) = B Σ_c v_c − B q`.
///
/// There is no default: the two are different experiments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualState {
    /// Rows execute the residual: `m_Δ = 1`.
    Kept,
    /// Rows drop the residual: `m_Δ = 0`.
    Removed,
}

impl ResidualState {
    /// The residual mask `m_Δ` every row of the experiment declares.
    pub fn residual_mask(self) -> f64 {
        match self {
            Self::Kept => 1.0,
            Self::Removed => 0.0,
        }
    }
}

/// How a readout's distortion is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Divergence {
    /// Squared error on a layer-local readout. Its coefficient block is conditionally
    /// Gaussian and fitted exactly.
    SquaredError,
    /// `KL(p_native ‖ p_edited)` on a distribution readout.
    ///
    /// A deterministic teacher gives a soft-label row no dispersion, so each row stands
    /// for a declared number of teacher draws. That count sets the likelihood's scale for
    /// the Laplace evidence, which is an approximation and never exact.
    Kl { samples: NonZeroU64 },
}

/// The declared inputs of a manifold parameter decomposition fit. Every field is
/// required, and none has a default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpdExperiment {
    residual_state: ResidualState,
    tolerance: f64,
    precision: DeclaredPrecision,
    divergence: Divergence,
}

/// Why an experiment declaration was refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExperimentError {
    /// The fidelity tolerance must be finite and nonnegative.
    Tolerance { value: f64 },
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tolerance { value } => write!(
                f,
                "experiment refused: the declared fidelity tolerance {value} must be finite and \
                 nonnegative"
            ),
        }
    }
}

impl std::error::Error for ExperimentError {}

impl MpdExperiment {
    /// Declares an experiment.
    ///
    /// A tolerance is refused unless it is finite and nonnegative. That is the same
    /// predicate `precision::decode_then_evaluate` and the codec's saving comparison
    /// apply, so a declaration they would refuse never reaches them.
    pub fn new(
        residual_state: ResidualState,
        tolerance: f64,
        precision: DeclaredPrecision,
        divergence: Divergence,
    ) -> Result<Self, ExperimentError> {
        if !(tolerance.is_finite() && tolerance >= 0.0) {
            return Err(ExperimentError::Tolerance { value: tolerance });
        }
        Ok(Self {
            residual_state,
            tolerance,
            precision,
            divergence,
        })
    }

    /// The declared residual state.
    pub fn residual_state(&self) -> ResidualState {
        self.residual_state
    }

    /// The declared fidelity tolerance.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// The declared precision of real codes.
    pub fn precision(&self) -> DeclaredPrecision {
        self.precision
    }

    /// The declared divergence of the readout.
    pub fn divergence(&self) -> Divergence {
        self.divergence
    }
}

#[cfg(test)]
mod experiment_tests {
    use super::*;

    #[test]
    fn an_experiment_declares_every_input_and_refuses_an_invalid_tolerance() {
        let precision = DeclaredPrecision::new(12).expect("a normal dyadic step");
        let samples = NonZeroU64::new(64).expect("a nonzero sample count");
        for invalid in [f64::NAN, f64::INFINITY, -1e-300] {
            let refused =
                MpdExperiment::new(ResidualState::Kept, invalid, precision, Divergence::SquaredError);
            assert!(
                matches!(refused, Err(ExperimentError::Tolerance { .. })),
                "tolerance {invalid} must be refused, got {refused:?}"
            );
        }
        // Positive control: zero is a valid declared tolerance, the same edge the
        // decode-then-evaluate owner admits.
        let exact = MpdExperiment::new(ResidualState::Removed, 0.0, precision, Divergence::Kl { samples })
            .expect("a zero tolerance is admitted");
        assert_eq!(exact.tolerance(), 0.0, "the declared tolerance is kept");
        assert_eq!(exact.residual_state(), ResidualState::Removed, "the declared residual state is kept");
        assert_eq!(exact.divergence(), Divergence::Kl { samples }, "the declared divergence is kept");
        assert_eq!(exact.precision(), precision, "the declared precision is kept");
    }

    #[test]
    fn the_residual_state_fixes_one_residual_mask_for_every_row() {
        assert_eq!(ResidualState::Kept.residual_mask(), 1.0, "Kept executes the residual");
        assert_eq!(ResidualState::Removed.residual_mask(), 0.0, "Removed drops the residual");
        assert_ne!(
            ResidualState::Kept.residual_mask(),
            ResidualState::Removed.residual_mask(),
            "the two declared states are different experiments"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::faer_ndarray::FaerEigh;
    use ndarray::{Array1, array};

    const COMPONENTS: usize = 3;
    const BASIS: usize = 3;
    const INPUT_DIM: usize = 4;
    const OUTPUT_DIM: usize = 3;
    const RANK: usize = 2;
    const INPUT_ROWS: usize = 6;

    /// Basis evaluations `φ(z_c)` of three components (a quadratic basis at
    /// z = 0.5, −0.3, 0.8) and their weights on the scale gauge `Σ_c w_c = 1`.
    const COMPONENT_BASIS: [[f64; BASIS]; COMPONENTS] =
        [[1.0, 0.5, 0.25], [1.0, -0.3, 0.09], [1.0, 0.8, 0.64]];
    const COMPONENT_WEIGHTS: [f64; COMPONENTS] = [0.5, 0.3, 0.2];

    /// Declared component masks and the residual mask.
    type Mask = ([f64; COMPONENTS], f64);

    /// Residual-removed witness masks whose moments are linearly independent.
    const SUFFICIENCY_MASKS: [Mask; 3] = [
        ([1.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 0.0], 0.0),
        ([0.0, 1.0, 1.0], 0.0),
    ];

    /// The all-on setting: every component and the residual kept.
    const ALL_ON: Mask = ([1.0; COMPONENTS], 1.0);

    /// `β(m) = Σ_c (m_c − m_Δ) w_c φ(z_c)`, the fixture's anchor moment.
    fn moment(mask: Mask) -> [f64; BASIS] {
        let mut beta = [0.0; BASIS];
        for component in 0..COMPONENTS {
            let scale = (mask.0[component] - mask.1) * COMPONENT_WEIGHTS[component];
            for j in 0..BASIS {
                beta[j] += scale * COMPONENT_BASIS[component][j];
            }
        }
        beta
    }

    /// Right factors `R_j` (`4 × 2`): full column rank, neither orthonormal nor
    /// mutually orthogonal, so every Gram `R_jᵀ R_k` matters.
    fn generic_right_factors() -> Vec<Array2<f64>> {
        (0..BASIS)
            .map(|j| {
                Array2::from_shape_fn((INPUT_DIM, RANK), |(i, a)| {
                    (0.9 * ((i + 1) * (a + 1)) as f64 + 0.4 * j as f64).cos()
                })
            })
            .collect()
    }

    fn planted_left(j: usize, o: usize, a: usize) -> f64 {
        0.4 + 0.3 * j as f64 - 0.2 * o as f64 + 0.15 * a as f64 * (j as f64 + 1.0)
    }

    struct Fixture {
        component_masks: Array2<f64>,
        residual_masks: Array1<f64>,
        moments: Array2<f64>,
        inputs: Array2<f64>,
        responses: Array2<f64>,
        right: Vec<Array2<f64>>,
        penalty: Array2<f64>,
        null_penalty: Array2<f64>,
    }

    impl Fixture {
        /// The energy penalty and its null-space form: a proper summed prior.
        fn penalty_views(&self) -> Vec<ArrayView2<'_, f64>> {
            vec![self.penalty.view(), self.null_penalty.view()]
        }

        fn rows(&self) -> GaussianBlockRows<'_> {
            GaussianBlockRows {
                component_masks: self.component_masks.view(),
                residual_masks: self.residual_masks.view(),
                moments: self.moments.view(),
                inputs: self.inputs.view(),
                responses: self.responses.view(),
            }
        }

        fn right_views(&self) -> Vec<ArrayView2<'_, f64>> {
            self.right.iter().map(|factor| factor.view()).collect()
        }
    }

    /// Each mask is applied to the same six inputs. Responses are the planted
    /// factored block plus a deterministic perturbation, so the profiled REML
    /// deviance is positive.
    fn fixture(masks: &[Mask], right: Vec<Array2<f64>>) -> Fixture {
        let n = masks.len() * INPUT_ROWS;
        let mut component_masks = Array2::<f64>::zeros((n, COMPONENTS));
        let mut residual_masks = Array1::<f64>::zeros(n);
        let mut moments = Array2::<f64>::zeros((n, BASIS));
        let mut inputs = Array2::<f64>::zeros((n, INPUT_DIM));
        let mut responses = Array2::<f64>::zeros((n, OUTPUT_DIM));
        for (mask_index, &mask) in masks.iter().enumerate() {
            let beta = moment(mask);
            for x in 0..INPUT_ROWS {
                let row = mask_index * INPUT_ROWS + x;
                for component in 0..COMPONENTS {
                    component_masks[[row, component]] = mask.0[component];
                }
                residual_masks[row] = mask.1;
                for j in 0..BASIS {
                    moments[[row, j]] = beta[j];
                }
                for i in 0..INPUT_DIM {
                    inputs[[row, i]] = (0.7 * x as f64 + 1.1 * i as f64 + 0.3).sin();
                }
                for o in 0..OUTPUT_DIM {
                    let mut value = 0.01 * (2.1 * row as f64 + 0.9 * o as f64 + 0.4).sin();
                    for j in 0..BASIS {
                        for a in 0..right[j].ncols() {
                            let mut projected = 0.0;
                            for i in 0..INPUT_DIM {
                                projected += right[j][[i, a]] * inputs[[row, i]];
                            }
                            value += beta[j] * planted_left(j, o, a) * projected;
                        }
                    }
                    responses[[row, o]] = value;
                }
            }
        }
        // The second-difference penalty D₂ᵀD₂ on three coefficients has rank 1, so
        // constant and linear fields are unpenalized.
        let penalty = array![[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]];
        // The orthogonal projector onto null(D₂ᵀD₂) = span{(1, 1, 1), (−1, 0, 1)}. It
        // penalizes exactly the fields the energy leaves free, so the summed prior is
        // proper. This is a test stand-in for field.rs's null-space form.
        let null_penalty = array![
            [5.0 / 6.0, 1.0 / 3.0, -1.0 / 6.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [-1.0 / 6.0, 1.0 / 3.0, 5.0 / 6.0]
        ];
        Fixture {
            component_masks,
            residual_masks,
            moments,
            inputs,
            responses,
            right,
            penalty,
            null_penalty,
        }
    }

    #[test]
    fn block_fit_reads_left_factors_in_design_order() {
        let data = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = data.right_views();
        let fit = fit_gaussian_coefficient_block(data.rows(), &right, &data.penalty_views())
            .expect("the full-rank sufficiency fixture is admitted and fitted");
        assert_eq!(fit.left_factors.len(), BASIS);
        for factor in &fit.left_factors {
            assert_eq!(factor.dim(), (OUTPUT_DIM, RANK));
        }
        let coefficients = &fit.reml.coefficients;
        // Negative control: the blocks read in reverse basis order.
        let reversed: Vec<Array2<f64>> = (0..BASIS)
            .map(|j| {
                Array2::from_shape_fn((OUTPUT_DIM, RANK), |(o, a)| {
                    coefficients[[(BASIS - 1 - j) * RANK + a, o]]
                })
            })
            .collect();
        // Both sides sum the K·d_in·r monomials β_j h_i R_j[i,a] L_j[o,a], each with
        // three multiplications, so every rounding path has at most K·d_in·r + 3
        // operations (Higham γ).
        let growth = accumulation_growth(BASIS * INPUT_DIM * RANK + 3);
        let mut reversed_refuted = false;
        for row in 0..data.moments.nrows() {
            for o in 0..OUTPUT_DIM {
                let mut design_side = 0.0;
                for j in 0..BASIS {
                    for a in 0..RANK {
                        let mut projected = 0.0;
                        for i in 0..INPUT_DIM {
                            projected += data.inputs[[row, i]] * data.right[j][[i, a]];
                        }
                        design_side +=
                            data.moments[[row, j]] * projected * coefficients[[j * RANK + a, o]];
                    }
                }
                let contract = |left: &[Array2<f64>]| {
                    let mut value = 0.0;
                    let mut absolute = 0.0;
                    for j in 0..BASIS {
                        let mut inner = 0.0;
                        for i in 0..INPUT_DIM {
                            let mut entry = 0.0;
                            for a in 0..RANK {
                                entry += left[j][[o, a]] * data.right[j][[i, a]];
                                absolute += (data.moments[[row, j]]
                                    * left[j][[o, a]]
                                    * data.right[j][[i, a]]
                                    * data.inputs[[row, i]])
                                    .abs();
                            }
                            inner += entry * data.inputs[[row, i]];
                        }
                        value += data.moments[[row, j]] * inner;
                    }
                    (value, absolute)
                };
                let (tensor_side, absolute) = contract(fit.left_factors.as_slice());
                let band = 2.0 * growth * absolute;
                assert!(
                    (design_side - tensor_side).abs() <= band,
                    "row {row} output {o}: design order {design_side} vs Σ_j β_j L_j R_jᵀ h = \
                     {tensor_side} differ beyond the roundoff band {band}"
                );
                let (reversed_side, reversed_absolute) = contract(reversed.as_slice());
                if (design_side - reversed_side).abs() > 2.0 * growth * absolute.max(reversed_absolute) {
                    reversed_refuted = true;
                }
            }
        }
        assert!(
            reversed_refuted,
            "negative control: reading the blocks in reverse basis order must disagree with the \
             design beyond the roundoff band"
        );
    }

    #[test]
    fn block_penalty_is_the_function_space_penalty_of_each_output() {
        let data = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = data.right_views();
        let fit = fit_gaussian_coefficient_block(data.rows(), &right, &data.penalty_views())
            .expect("the full-rank sufficiency fixture is admitted and fitted");
        let columns = BASIS * RANK;
        let penalty = block_penalty(0, data.penalty.view(), &right)
            .expect("the symmetric fixture penalty assembles within its band");
        // Negative control: S ⊗ I_r, which wrongly treats every Gram R_jᵀ R_k as the identity.
        let gram_blind = Array2::from_shape_fn((columns, columns), |(a, b)| {
            if a % RANK == b % RANK {
                data.penalty[[a / RANK, b / RANK]]
            } else {
                0.0
            }
        });
        let coefficients = &fit.reml.coefficients;
        // Both sides sum the K²·d_in·r² monomials S_jk L_j[o,a] R_j[i,a] L_k[o,b] R_k[i,b],
        // each with four multiplications, so every rounding path has at most
        // K²·d_in·r² + 4 operations.
        let growth = accumulation_growth(BASIS * BASIS * INPUT_DIM * RANK * RANK + 4);
        let mut gram_blind_refuted = false;
        for o in 0..OUTPUT_DIM {
            let quadratic = |matrix: &Array2<f64>| {
                let mut value = 0.0;
                let mut absolute = 0.0;
                for a in 0..columns {
                    for b in 0..columns {
                        let term = coefficients[[a, o]] * matrix[[a, b]] * coefficients[[b, o]];
                        value += term;
                        absolute += term.abs();
                    }
                }
                (value, absolute)
            };
            let mut functional = 0.0;
            let mut absolute = 0.0;
            for j in 0..BASIS {
                for k in 0..BASIS {
                    for i in 0..INPUT_DIM {
                        let mut left_entry = 0.0;
                        let mut right_entry = 0.0;
                        for a in 0..RANK {
                            left_entry += fit.left_factors[j][[o, a]] * data.right[j][[i, a]];
                            right_entry += fit.left_factors[k][[o, a]] * data.right[k][[i, a]];
                        }
                        functional += data.penalty[[j, k]] * left_entry * right_entry;
                        for a in 0..RANK {
                            for b in 0..RANK {
                                absolute += (data.penalty[[j, k]]
                                    * fit.left_factors[j][[o, a]]
                                    * data.right[j][[i, a]]
                                    * fit.left_factors[k][[o, b]]
                                    * data.right[k][[i, b]])
                                    .abs();
                            }
                        }
                    }
                }
            }
            let band = 2.0 * growth * absolute;
            let (design, design_absolute) = quadratic(&penalty);
            assert!(
                (design - functional).abs() <= band,
                "output {o}: β'Pβ = {design} vs Σ_jk S_jk⟨L_j R_jᵀ, L_k R_kᵀ⟩ = {functional} \
                 differ beyond the roundoff band {band} (design-side absolute sum {design_absolute})"
            );
            let (blind, blind_absolute) = quadratic(&gram_blind);
            if (blind - functional).abs() > 2.0 * growth * absolute.max(blind_absolute) {
                gram_blind_refuted = true;
            }
        }
        assert!(
            gram_blind_refuted,
            "negative control: a penalty that ignores the right-factor Grams must disagree with \
             the function-space penalty beyond the roundoff band"
        );
    }

    #[test]
    fn identity_right_factors_reduce_to_the_dense_kronecker_block() {
        let identity: Vec<Array2<f64>> = (0..BASIS).map(|_| Array2::eye(INPUT_DIM)).collect();
        let data = fixture(&SUFFICIENCY_MASKS, identity);
        let right = data.right_views();
        let design = block_design(data.moments.view(), data.inputs.view(), &right);
        assert_eq!(
            design,
            dense_rowwise_kronecker(data.moments.view(), data.inputs.view()),
            "with R_j = I the design is β ⊗ h exactly: every product with an identity entry is exact"
        );
        let columns = BASIS * INPUT_DIM;
        let kronecker = Array2::from_shape_fn((columns, columns), |(a, b)| {
            if a % INPUT_DIM == b % INPUT_DIM {
                data.penalty[[a / INPUT_DIM, b / INPUT_DIM]]
            } else {
                0.0
            }
        });
        assert_eq!(
            block_penalty(0, data.penalty.view(), &right)
                .expect("the symmetric fixture penalty assembles within its band"),
            kronecker,
            "with R_j = I the penalty is S ⊗ I exactly"
        );
        // Negative control: doubling one right factor moves the design.
        let doubled: Vec<Array2<f64>> = (0..BASIS)
            .map(|j| {
                if j == 0 {
                    Array2::eye(INPUT_DIM) * 2.0
                } else {
                    Array2::eye(INPUT_DIM)
                }
            })
            .collect();
        let doubled_views: Vec<ArrayView2<'_, f64>> =
            doubled.iter().map(|factor| factor.view()).collect();
        assert_ne!(
            block_design(data.moments.view(), data.inputs.view(), &doubled_views),
            design,
            "negative control: a non-identity right factor must change the design"
        );
    }

    #[test]
    fn rows_are_refused_on_the_declared_mask_never_on_a_vanishing_moment() {
        // Guard: all-on, and every component off with the residual removed, both
        // execute m_Δ Θ_* at every parameter value.
        for native_multiple in [ALL_ON, ([0.0; COMPONENTS], 0.0)] {
            let mut masks = SUFFICIENCY_MASKS.to_vec();
            masks.push(native_multiple);
            let data = fixture(&masks, generic_right_factors());
            let right = data.right_views();
            let refused = fit_gaussian_coefficient_block(data.rows(), &right, &data.penalty_views());
            assert!(
                matches!(
                    refused,
                    Err(GaussianBlockError::NativeMultipleRow { row }) if row == 3 * INPUT_ROWS
                ),
                "mask {native_multiple:?}: the first native-multiple row must be refused, got {refused:?}"
            );
        }

        // Positive control: a residual-on ablation (component 3 removed from Θ_*).
        let mut ablation = SUFFICIENCY_MASKS.to_vec();
        ablation.push(([1.0, 1.0, 0.0], 1.0));
        let admitted = fixture(&ablation, generic_right_factors());
        let admitted_right = admitted.right_views();
        let fit =
            fit_gaussian_coefficient_block(admitted.rows(), &admitted_right, &admitted.penalty_views());
        assert!(
            matches!(&fit, Ok(block) if block.reml.evaluation.reml_score.is_finite()),
            "a residual-on ablation row must be admitted and fitted, got {fit:?}"
        );

        // A declared mask that differs from the residual mask, whose moment vanishes
        // by cancellation (as when v_1 + v_2 = 0 at the current labels and weights).
        // Its derivatives in the labels and weights do not vanish, so it is admitted.
        let mut cancelling = SUFFICIENCY_MASKS.to_vec();
        cancelling.push(([1.0, 1.0, 0.0], 0.0));
        let mut cancelled = fixture(&cancelling, generic_right_factors());
        cancelled.moments.row_mut(3 * INPUT_ROWS).fill(0.0);
        let cancelled_right = cancelled.right_views();
        let cancelled_fit = fit_gaussian_coefficient_block(
            cancelled.rows(),
            &cancelled_right,
            &cancelled.penalty_views(),
        );
        assert!(
            matches!(&cancelled_fit, Ok(block) if block.reml.evaluation.reml_score.is_finite()),
            "a row with a vanishing moment but a non-native mask must be admitted, got {cancelled_fit:?}"
        );
    }

    #[test]
    fn penalties_are_symmetrized_only_within_their_derived_assembly_band() {
        let data = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = data.right_views();
        let mut lopsided = data.penalty.clone();
        lopsided[[0, 1]] = -2.0 + 1e-3;
        let refused = fit_gaussian_coefficient_block(
            data.rows(),
            &right,
            &[lopsided.view(), data.null_penalty.view()],
        );
        assert!(
            matches!(
                refused,
                Err(GaussianBlockError::AsymmetricFieldPenalty { penalty: 0, row: 0, col: 1 })
            ),
            "a field penalty that is not exactly symmetric must be refused, got {refused:?}"
        );

        // The assembly guard, on a symmetric matrix, a rounding-sized gap and a wider one.
        let columns = 4;
        let symmetric = Array2::from_shape_fn((columns, columns), |(a, b)| 1.0 + (a + b) as f64);
        let absolute = symmetric.mapv(f64::abs);
        let growth = accumulation_growth(INPUT_DIM + 1);
        let unchanged = symmetrized_within_band(0, symmetric.clone(), &absolute, growth);
        assert!(
            matches!(&unchanged, Ok(averaged) if *averaged == symmetric),
            "an exactly symmetric penalty is unchanged by averaging, got {unchanged:?}"
        );
        let band = growth * (absolute[[0, 1]] + absolute[[1, 0]]);
        let mut rounded = symmetric.clone();
        rounded[[0, 1]] += 0.5 * band;
        assert!(
            symmetrized_within_band(0, rounded, &absolute, growth).is_ok(),
            "a gap inside the derived band is rounding and is averaged away"
        );
        // Positive control: a deliberately asymmetric block beyond the band is refused.
        let mut broken = symmetric.clone();
        broken[[0, 1]] += 8.0 * band;
        let refused_assembly = symmetrized_within_band(1, broken, &absolute, growth);
        assert!(
            matches!(
                refused_assembly,
                Err(GaussianBlockError::AssembledPenaltyAsymmetric { penalty: 1, row: 0, col: 1, .. })
            ),
            "an assembled gap beyond the band must be refused, got {refused_assembly:?}"
        );
    }

    #[test]
    fn an_energy_penalty_alone_is_refused_because_its_null_space_escapes_shrinkage() {
        let data = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = data.right_views();
        let energy_only = fit_gaussian_coefficient_block(data.rows(), &right, &[data.penalty.view()]);
        assert!(
            matches!(energy_only, Err(GaussianBlockError::Reml(..))),
            "the energy penalty leaves constant and linear fields free, so the owner must refuse \
             the declared zero null space, got {energy_only:?}"
        );
        // Positive control: the same rows with the null-space form added have a proper prior.
        let proper = fit_gaussian_coefficient_block(data.rows(), &right, &data.penalty_views());
        assert!(
            matches!(&proper, Ok(block) if block.reml.evaluation.reml_score.is_finite()),
            "energy plus null-space penalty must be admitted and fitted, got {proper:?}"
        );
        let empty = fit_gaussian_coefficient_block(data.rows(), &right, &[]);
        assert!(
            matches!(empty, Err(GaussianBlockError::NoFieldPenalties)),
            "an empty penalty list declares no prior and must be refused, got {empty:?}"
        );
    }

    #[test]
    fn dense_block_admission_refuses_one_byte_below_its_ledger() {
        let rows = SUFFICIENCY_MASKS.len() * INPUT_ROWS;
        let ranks = [RANK; BASIS];
        let columns = BASIS * RANK;
        let penalties = 2;
        let expected = (rows * (columns + RANK)
            + 2 * penalties * columns * columns
            + rows.min(columns) * (columns + OUTPUT_DIM)
            + columns * columns
            + columns * OUTPUT_DIM)
            * std::mem::size_of::<f64>();
        assert_eq!(
            dense_block_bytes(rows, &ranks, penalties, OUTPUT_DIM),
            Some(expected),
            "the ledger counts the design and widest projection, both penalty copies, the owner's \
             reduction and stacked root, and the coefficients"
        );
        assert!(
            admit_dense_block(rows, &ranks, penalties, OUTPUT_DIM, expected).is_ok(),
            "a budget equal to the ledger admits the block"
        );
        let refused = admit_dense_block(rows, &ranks, penalties, OUTPUT_DIM, expected - 1);
        assert!(
            matches!(
                refused,
                Err(GaussianBlockError::AdmissionRefused { required_bytes: Some(required), budget_bytes })
                    if required == expected && budget_bytes == expected - 1
            ),
            "one byte below the ledger must refuse, got {refused:?}"
        );
        let overflow = admit_dense_block(usize::MAX, &[2], 1, 1, usize::MAX);
        assert!(
            matches!(
                overflow,
                Err(GaussianBlockError::AdmissionRefused { required_bytes: None, .. })
            ),
            "an overflowing ledger must refuse, got {overflow:?}"
        );
    }

    /// A deterministic design cotangent in design order, standing in for a criterion's ∂V/∂X.
    fn design_cotangent(rows: usize, columns: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, columns), |(r, a)| (0.37 * r as f64 + 1.3 * a as f64 + 0.2).sin())
    }

    /// `Σ_ij left_ij right_ij`.
    fn frobenius(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
    }

    /// `Σ |G[r, j·r + a] β[r, j] h[r, i] R_j[i, a]|` over every monomial of the pairing.
    fn monomial_absolute_sum(
        cotangent: &Array2<f64>,
        moments: &Array2<f64>,
        inputs: &Array2<f64>,
        right: &[Array2<f64>],
    ) -> f64 {
        let mut total = 0.0;
        for row in 0..moments.nrows() {
            for (j, factor) in right.iter().enumerate() {
                for a in 0..factor.ncols() {
                    for i in 0..inputs.ncols() {
                        total += (cotangent[[row, j * RANK + a]]
                            * moments[[row, j]]
                            * inputs[[row, i]]
                            * factor[[i, a]])
                            .abs();
                    }
                }
            }
        }
        total
    }

    #[test]
    fn block_design_adjoint_pulls_back_moments_inputs_and_right_factors_exactly() {
        let data = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = data.right_views();
        let rows = data.moments.nrows();
        let cotangent = design_cotangent(rows, BASIS * RANK);
        let adjoint =
            block_design_adjoint(cotangent.view(), data.moments.view(), data.inputs.view(), &right)
                .expect("a well-shaped cotangent is pulled back");
        // Every pairing sums n·K·d_in·r monomials G·β·h·R, each with three
        // multiplications, so every rounding path has depth at most n·K·d_in·r + 3.
        let growth = accumulation_growth(rows * BASIS * INPUT_DIM * RANK + 3);

        // Moments: X is linear in β, so <G, X(dβ)> = <∂V/∂β, dβ>.
        let direction_moments =
            Array2::from_shape_fn((rows, BASIS), |(r, j)| (0.9 * r as f64 + 0.5 * j as f64).cos());
        let design_side = frobenius(
            &cotangent,
            &block_design(direction_moments.view(), data.inputs.view(), &right),
        );
        let adjoint_side = frobenius(&adjoint.moments, &direction_moments);
        let band =
            2.0 * growth * monomial_absolute_sum(&cotangent, &direction_moments, &data.inputs, &data.right);
        assert!(
            (design_side - adjoint_side).abs() <= band,
            "moments: <G, X(dβ)> = {design_side} vs <∂V/∂β, dβ> = {adjoint_side} beyond the band {band}"
        );
        let reversed =
            Array2::from_shape_fn((rows, BASIS), |(r, j)| adjoint.moments[[r, BASIS - 1 - j]]);
        assert!(
            (design_side - frobenius(&reversed, &direction_moments)).abs() > band,
            "negative control: the moment cotangents in reverse basis order must be refuted"
        );

        // Inputs: X is linear in h, so <G, X(dh)> = <∂V/∂h, dh>.
        let direction_inputs = Array2::from_shape_fn((rows, INPUT_DIM), |(r, i)| {
            (0.4 * r as f64 - 0.7 * i as f64 + 0.1).sin()
        });
        let design_side = frobenius(
            &cotangent,
            &block_design(data.moments.view(), direction_inputs.view(), &right),
        );
        let adjoint_side = frobenius(&adjoint.inputs, &direction_inputs);
        let band =
            2.0 * growth * monomial_absolute_sum(&cotangent, &data.moments, &direction_inputs, &data.right);
        assert!(
            (design_side - adjoint_side).abs() <= band,
            "inputs: <G, X(dh)> = {design_side} vs <∂V/∂h, dh> = {adjoint_side} beyond the band {band}"
        );
        let unscaled = block_design_adjoint(
            cotangent.view(),
            Array2::<f64>::ones((rows, BASIS)).view(),
            data.inputs.view(),
            &right,
        )
        .expect("a well-shaped cotangent is pulled back");
        assert!(
            (design_side - frobenius(&unscaled.inputs, &direction_inputs)).abs() > band,
            "negative control: an input cotangent that drops the moment scale must be refuted"
        );

        // Right factors: X is linear in each R_j, so <G, X(dR)> = Σ_j <∂V/∂R_j, dR_j>.
        let direction_right: Vec<Array2<f64>> = (0..BASIS)
            .map(|j| {
                Array2::from_shape_fn((INPUT_DIM, RANK), |(i, a)| {
                    (0.3 * i as f64 + 1.1 * a as f64 + 0.6 * j as f64).cos()
                })
            })
            .collect();
        let direction_views: Vec<ArrayView2<'_, f64>> =
            direction_right.iter().map(|factor| factor.view()).collect();
        let design_side = frobenius(
            &cotangent,
            &block_design(data.moments.view(), data.inputs.view(), &direction_views),
        );
        let adjoint_side: f64 = adjoint
            .right_factors
            .iter()
            .zip(&direction_right)
            .map(|(pullback, direction)| frobenius(pullback, direction))
            .sum();
        let band =
            2.0 * growth * monomial_absolute_sum(&cotangent, &data.moments, &data.inputs, &direction_right);
        assert!(
            (design_side - adjoint_side).abs() <= band,
            "right factors: <G, X(dR)> = {design_side} vs Σ<∂V/∂R_j, dR_j> = {adjoint_side} beyond the \
             band {band}"
        );
        let shifted: f64 = (0..BASIS)
            .map(|j| frobenius(&adjoint.right_factors[(j + 1) % BASIS], &direction_right[j]))
            .sum();
        assert!(
            (design_side - shifted).abs() > band,
            "negative control: pairing R_j with the next basis matrix's cotangent must be refuted"
        );
    }

    #[test]
    fn block_design_adjoint_refuses_a_cotangent_of_the_wrong_width() {
        let data = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = data.right_views();
        let rows = data.moments.nrows();
        let narrow = design_cotangent(rows, BASIS * RANK - 1);
        let refused =
            block_design_adjoint(narrow.view(), data.moments.view(), data.inputs.view(), &right);
        assert!(
            matches!(
                refused,
                Err(GaussianBlockError::DesignCotangentWidth { columns, expected })
                    if columns == BASIS * RANK - 1 && expected == BASIS * RANK
            ),
            "a cotangent one column short must be refused, got {refused:?}"
        );
        // Positive control: the design's own width is admitted.
        let wide = design_cotangent(rows, BASIS * RANK);
        let admitted =
            block_design_adjoint(wide.view(), data.moments.view(), data.inputs.view(), &right);
        assert!(admitted.is_ok(), "the design's own width must be admitted, got {admitted:?}");
    }

    /// A block whose design-order coefficients are drawn from its own prior at unit
    /// strengths, `B ~ N(0, (P_0 + P_1)⁻¹)` per output, with unit noise: both field
    /// penalties are material, so the REML optimum is interior. 3 masks × 20 inputs, 8
    /// outputs.
    fn prior_fixture() -> Fixture {
        const INPUTS: usize = 20;
        const OUTPUTS: usize = 8;
        let base = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = base.right_views();
        let n = SUFFICIENCY_MASKS.len() * INPUTS;
        let mut state = 0x2946_2951_u64;
        let mut draw = || {
            gam_math::probability::standard_normal_from_uniform_bits(gam_linalg::utils::splitmix64(&mut state))
                .expect("a standard normal draw")
        };
        let component_masks =
            Array2::from_shape_fn((n, COMPONENTS), |(row, c)| SUFFICIENCY_MASKS[row / INPUTS].0[c]);
        let residual_masks = Array1::from_shape_fn(n, |row| SUFFICIENCY_MASKS[row / INPUTS].1);
        let moments = Array2::from_shape_fn((n, BASIS), |(row, j)| moment(SUFFICIENCY_MASKS[row / INPUTS])[j]);
        let inputs = Array2::from_shape_fn((n, INPUT_DIM), |(row, i)| {
            (0.7 * (row % INPUTS) as f64 + 1.1 * i as f64 + 0.3).sin()
        });
        let precision = &block_penalty(0, base.penalty.view(), &right).expect("the energy block penalty")
            + &block_penalty(1, base.null_penalty.view(), &right).expect("the null-space block penalty");
        let (values, vectors) = precision
            .eigh(faer::Side::Lower)
            .expect("the prior precision's spectrum");
        assert!(values.iter().all(|&value| value > 0.0), "the summed prior must be proper: {values:?}");
        let standard = Array2::from_shape_simple_fn((precision.nrows(), OUTPUTS), &mut draw);
        let coefficients = vectors.dot(&Array2::from_shape_fn(standard.dim(), |(row, col)| {
            standard[[row, col]] / values[row].sqrt()
        }));
        let design = block_design(moments.view(), inputs.view(), &right);
        let noise = Array2::from_shape_simple_fn((n, OUTPUTS), &mut draw);
        let responses = design.dot(&coefficients) + noise;
        Fixture {
            component_masks,
            residual_masks,
            moments,
            inputs,
            responses,
            right: base.right.clone(),
            penalty: base.penalty.clone(),
            null_penalty: base.null_penalty.clone(),
        }
    }

    /// `V(ρ)` and its rounding bound for the block rebuilt from these arrays, at a fixed `ρ`.
    fn block_criterion_at(
        rho: ArrayView1<'_, f64>,
        moments: &Array2<f64>,
        inputs: &Array2<f64>,
        right: &[Array2<f64>],
        responses: &Array2<f64>,
        penalties: &[ArrayView2<'_, f64>],
    ) -> (f64, f64) {
        let right_views: Vec<ArrayView2<'_, f64>> = right.iter().map(|factor| factor.view()).collect();
        let design = block_design(moments.view(), inputs.view(), &right_views);
        let assembled = penalties
            .iter()
            .enumerate()
            .map(|(k, penalty)| block_penalty(k, *penalty, &right_views).expect("a perturbed block penalty"))
            .collect::<Vec<_>>();
        let evaluation = GaussianRemlMultiPenaltyProblem::new(design.view(), responses.view(), &assembled, 0)
            .expect("a perturbed block problem")
            .evaluate(rho)
            .expect("a perturbed block evaluation");
        (evaluation.reml_score, evaluation.reml_score_roundoff)
    }

    /// Central differences of `score` at `h` and `2h`: `D_h` and its band, the Richardson
    /// remainder `|D_2h − D_h|/3` plus both differences' rounding over their steps.
    fn richardson(score: impl Fn(f64) -> (f64, f64), step: f64) -> (f64, f64) {
        let central = |h: f64| {
            let (above, above_rounding) = score(h);
            let (below, below_rounding) = score(-h);
            ((above - below) / (2.0 * h), (above_rounding + below_rounding) / (2.0 * h))
        };
        let (at_h, rounding_h) = central(step);
        let (at_2h, rounding_2h) = central(2.0 * step);
        (at_h, (at_2h - at_h).abs() / 3.0 + (4.0 * rounding_h + rounding_2h) / 3.0)
    }

    fn inner(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
    }

    fn root_mean_square(matrix: &Array2<f64>) -> f64 {
        (matrix.iter().map(|value| value * value).sum::<f64>() / matrix.len() as f64).sqrt()
    }

    #[test]
    fn block_cotangents_match_central_differences_of_the_criterion_along_every_leg() {
        let data = prior_fixture();
        let right = data.right_views();
        let penalties = data.penalty_views();
        let fit = fit_gaussian_coefficient_block(data.rows(), &right, &penalties)
            .expect("the prior fixture is admitted and fitted");
        assert!(
            fit.reml
                .rho_placement
                .iter()
                .all(|placement| *placement == GaussianRemlMultiPenaltyRhoPlacement::Interior),
            "precondition: the prior fixture's strengths must be interior; placement {:?}",
            fit.reml.rho_placement
        );
        let cotangents = match gaussian_block_cotangents(&fit, data.rows(), &right)
            .expect("the block cotangents at an interior fit")
        {
            GaussianBlockCotangentOutcome::Interior(cotangents) => cotangents,
            other => panic!("an interior fit must return the block cotangents, got {other:?}"),
        };
        let rho = fit.reml.evaluation.rho.clone();

        // One direction per leg, scaled to the leg's own magnitude so the difference resolves it.
        let mut state = 0x2951_0001_u64;
        let mut draw = || {
            gam_math::probability::standard_normal_from_uniform_bits(gam_linalg::utils::splitmix64(&mut state))
                .expect("a standard normal draw")
        };
        let mut direction = |like: &Array2<f64>| {
            Array2::from_shape_simple_fn(like.dim(), &mut draw) * root_mean_square(like)
        };
        let d_moments = direction(&data.moments);
        let d_inputs = direction(&data.inputs);
        let d_responses = direction(&data.responses);
        let d_right: Vec<Array2<f64>> = data.right.iter().map(|factor| direction(factor)).collect();
        let step = f64::EPSILON.cbrt();
        let criterion = |moments: &Array2<f64>, inputs: &Array2<f64>, right: &[Array2<f64>], responses: &Array2<f64>| {
            block_criterion_at(rho.view(), moments, inputs, right, responses, &penalties)
        };
        let check = |leg: &str, analytic: f64, (difference, band): (f64, f64)| {
            assert!(
                analytic.abs() > band,
                "{leg}: the directional derivative {analytic:.3e} must exceed the band {band:.3e}, or agreement \
                 is vacuous"
            );
            assert!(
                (difference - analytic).abs() <= band,
                "{leg}: central difference {difference:.9e} vs cotangent {analytic:.9e}, band {band:.3e}"
            );
        };
        check(
            "∂V/∂β",
            inner(&cotangents.moments, &d_moments),
            richardson(
                |h| criterion(&(&data.moments + &(&d_moments * h)), &data.inputs, &data.right, &data.responses),
                step,
            ),
        );
        check(
            "∂V/∂h",
            inner(&cotangents.inputs, &d_inputs),
            richardson(
                |h| criterion(&data.moments, &(&data.inputs + &(&d_inputs * h)), &data.right, &data.responses),
                step,
            ),
        );
        check(
            "∂V/∂Y",
            inner(&cotangents.responses, &d_responses),
            richardson(
                |h| criterion(&data.moments, &data.inputs, &data.right, &(&data.responses + &(&d_responses * h))),
                step,
            ),
        );
        let perturbed_right = |h: f64| -> Vec<Array2<f64>> {
            data.right
                .iter()
                .zip(d_right.iter())
                .map(|(factor, change)| factor + &(change * h))
                .collect()
        };
        let right_difference = richardson(
            |h| criterion(&data.moments, &data.inputs, &perturbed_right(h), &data.responses),
            step,
        );
        let right_analytic: f64 = cotangents
            .right_factors
            .iter()
            .zip(d_right.iter())
            .map(|(cotangent, change)| inner(cotangent, change))
            .sum();
        check("∂V/∂R", right_analytic, right_difference);

        // Mutant: the right factors' cotangent without the penalty chain, i.e. what the design
        // adjoint alone gives from the owner's data gradient, is resolved apart.
        let design = block_design(data.moments.view(), data.inputs.view(), &right);
        let data_gradient = match fit
            .problem
            .data_gradient(design.view(), data.responses.view(), &fit.reml)
            .expect("the owner's data gradient")
        {
            GaussianRemlMultiPenaltyDataGradientOutcome::Interior(gradient) => gradient,
            other => panic!("an interior fit must return the data gradient, got {other:?}"),
        };
        let design_only = block_design_adjoint(
            data_gradient.grad_x.view(),
            data.moments.view(),
            data.inputs.view(),
            &right,
        )
        .expect("the design adjoint of the owner's data gradient");
        let mutant: f64 = design_only
            .right_factors
            .iter()
            .zip(d_right.iter())
            .map(|(cotangent, change)| inner(cotangent, change))
            .sum();
        let (difference, band) = right_difference;
        assert!(
            (difference - mutant).abs() > band,
            "mutant: the design leg alone gives {mutant:.9e}, inside the band {band:.3e} of {difference:.9e}, so \
             the penalty chain is not resolved"
        );
    }

    #[test]
    fn block_cotangents_refuse_arrays_the_fit_was_not_built_from() {
        let data = prior_fixture();
        let right = data.right_views();
        let fit = fit_gaussian_coefficient_block(data.rows(), &right, &data.penalty_views())
            .expect("the prior fixture is admitted and fitted");
        assert!(
            matches!(
                gaussian_block_cotangents(&fit, data.rows(), &right),
                Ok(GaussianBlockCotangentOutcome::Interior(..))
            ),
            "control: the fit's own rows and right factors are served"
        );
        let one_ulp = |matrix: &Array2<f64>| {
            let mut changed = matrix.clone();
            changed[[0, 0]] = f64::from_bits(changed[[0, 0]].to_bits() + 1);
            changed
        };

        let responses = one_ulp(&data.responses);
        let mut rows = data.rows();
        rows.responses = responses.view();
        match gaussian_block_cotangents(&fit, rows, &right) {
            Err(GaussianBlockError::Reml(error)) => assert!(
                error.to_string().contains("refuses `y`"),
                "a response one ulp away must be refused by the owner, naming y: {error}"
            ),
            other => panic!("a response one ulp away must be refused typed, got {other:?}"),
        }

        let moments = one_ulp(&data.moments);
        let mut rows = data.rows();
        rows.moments = moments.view();
        match gaussian_block_cotangents(&fit, rows, &right) {
            Err(GaussianBlockError::Reml(error)) => assert!(
                error.to_string().contains("refuses `x`"),
                "a moment one ulp away changes the design, so the owner must refuse it naming x: {error}"
            ),
            other => panic!("a moment one ulp away must be refused typed, got {other:?}"),
        }

        let changed_factor = one_ulp(&data.right[1]);
        let mut changed_right = right.clone();
        changed_right[1] = changed_factor.view();
        assert!(
            matches!(
                gaussian_block_cotangents(&fit, data.rows(), &changed_right),
                Err(GaussianBlockError::RightFactorsChanged { basis: 1 })
            ),
            "a right factor one ulp away must be refused, naming basis 1"
        );
        assert!(
            matches!(
                gaussian_block_cotangents(&fit, data.rows(), &right[..BASIS - 1]),
                Err(GaussianBlockError::RightFactorsChanged { basis }) if basis == BASIS
            ),
            "a missing right factor must be refused"
        );
    }

    #[test]
    fn block_cotangents_return_a_railed_strength_typed() {
        let data = prior_fixture();
        let right = data.right_views();
        let mut fit = fit_gaussian_coefficient_block(data.rows(), &right, &data.penalty_views())
            .expect("the prior fixture is admitted and fitted");
        let mut railed = fit.reml.rho_placement.clone();
        railed[1] = GaussianRemlMultiPenaltyRhoPlacement::UpperBound;
        fit.reml.rho_placement = railed.clone();
        assert_eq!(
            gaussian_block_cotangents(&fit, data.rows(), &right).expect("a railed fit is not an error"),
            GaussianBlockCotangentOutcome::RhoAtDomainBound { placement: railed },
            "a railed strength must return the typed outcome, never a partial gradient"
        );
    }
}

#[cfg(test)]
mod proposal_tests {
    use super::*;
    use crate::parameter_decomposition::precision::{DecodableArtifact, decode_then_evaluate};
    use crate::parameter_decomposition::supports::ExactBasis;

    /// The declared fidelity tolerance of these fixtures.
    const TOLERANCE: f64 = 0.1;

    type Status = EvidenceStatus<Vec<f64>, &'static str>;

    type Fidelity = DecodedFidelity<Vec<f64>, &'static str>;

    /// A decoded figure whose artifact is its own output, so its evidence is exact.
    struct Figure(f64);

    impl DecodableArtifact for Figure {
        type Decoded = f64;

        fn decode(&self) -> Result<f64, String> {
            Ok(self.0)
        }
    }

    /// Decoded distortion evidence under `tolerance`, built through the precision owner:
    /// the output's exact distance from a native reference of zero, with a stated rounding
    /// bound, over a one-member input family.
    fn decoded(distortion: f64, numerical_error: f64, tolerance: f64) -> Fidelity {
        decode_then_evaluate(
            &Figure(distortion),
            |value: &f64| Ok(*value),
            &0.0,
            |outputs: &f64, native: &f64| {
                EvidenceStatus::exact(
                    *outputs - *native,
                    numerical_error,
                    ExactBasis::Exhaustive { cardinality: 1 },
                    None,
                    "declared input family",
                )
                .map_err(|error| error.to_string())
            },
            tolerance,
        )
        .expect("a valid declared distortion")
    }

    fn certified() -> Status {
        EvidenceStatus::uniform_bound(0.09, 0.001, "declared mask box").expect("a valid uniform bound")
    }

    #[test]
    fn a_shorter_certified_candidate_is_accepted_and_code_that_is_not_shorter_is_rejected() {
        let reference = decoded(0.05, 0.001, TOLERANCE);
        let candidate = decoded(0.08, 0.001, TOLERANCE);
        let accepted =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &candidate), certified());
        assert!(
            matches!(
                &accepted,
                Ok(ProposalAcceptance { kind: ProposalKind::Split, saving_bits: 100, fidelity_certified: true, .. })
            ),
            "a 100-bit shorter certified candidate must be accepted, got {accepted:?}"
        );
        let equal =
            decide_proposal(ProposalKind::Share, (1000, &reference), (1000, &candidate), certified());
        assert!(
            matches!(equal, Err(ProposalRejection::NoShorterCode { saving_bits: 0 })),
            "equal code is not a strict decrease, got {equal:?}"
        );
        // An operator that interpolates the teacher at equal fidelity with a longer code
        // (mpd-modadd's rank caveat) loses on code alone.
        let interpolating_fidelity = decoded(0.05, 0.001, TOLERANCE);
        let interpolating = decide_proposal(
            ProposalKind::Expose,
            (1000, &reference),
            (1200, &interpolating_fidelity),
            certified(),
        );
        assert!(
            matches!(interpolating, Err(ProposalRejection::NoShorterCode { saving_bits: -200 })),
            "a longer interpolating operator must lose on code, got {interpolating:?}"
        );
    }

    #[test]
    fn a_refuting_estimated_or_infimum_fidelity_status_rejects_even_a_shorter_candidate() {
        let reference = decoded(0.05, 0.001, TOLERANCE);
        let shorter = decoded(0.08, 0.001, TOLERANCE);
        let mask = vec![1.0, 0.0, 1.0];
        let refuting: [Status; 3] = [
            EvidenceStatus::counterexample(0.3, 0.001, TOLERANCE, mask.clone()).expect("a violation"),
            EvidenceStatus::exact(0.2, 0.0, ExactBasis::Exhaustive { cardinality: 8 }, None, "binary masks")
                .expect("a valid exact value"),
            EvidenceStatus::unresolved(0.12, f64::INFINITY, Extremum::Supremum, Some(mask.clone()), "mask box")
                .expect("a valid bracket"),
        ];
        for status in refuting {
            let decision =
                decide_proposal(ProposalKind::Reduce, (1000, &reference), (900, &shorter), status);
            assert!(
                matches!(decision, Err(ProposalRejection::FidelityRefuted(..))),
                "a status whose lower bound exceeds the tolerance must reject, got {decision:?}"
            );
        }

        let estimate: Status =
            EvidenceStatus::statistical_estimate(0.02, 0.001, 64, "iid uniform masks").expect("a valid estimate");
        let estimated =
            decide_proposal(ProposalKind::Refine, (1000, &reference), (900, &shorter), estimate);
        assert!(
            matches!(estimated, Err(ProposalRejection::EstimateIsNotAFidelityBound(..))),
            "a stochastic-mask mean must never stand in for the supremum, got {estimated:?}"
        );
        // Positive control: the same figure as a uniform bound is accepted.
        let bound: Status =
            EvidenceStatus::uniform_bound(0.02, 0.001, "declared mask box").expect("a valid uniform bound");
        let bounded = decide_proposal(ProposalKind::Refine, (1000, &reference), (900, &shorter), bound);
        assert!(bounded.is_ok(), "the same figure as a uniform bound must be accepted, got {bounded:?}");

        let infimum =
            EvidenceStatus::unresolved(0.04, 0.09, Extremum::Infimum, Some(mask.clone()), "mask box").expect("a valid bracket");
        let infimum_decision =
            decide_proposal(ProposalKind::Expose, (1000, &reference), (900, &shorter), infimum);
        assert!(
            matches!(infimum_decision, Err(ProposalRejection::NotASupremum(..))),
            "an infimum bracket certifies nothing about the supremum, got {infimum_decision:?}"
        );
        // Positive control: the same bracket about the supremum certifies.
        let supremum = decide_proposal(
            ProposalKind::Expose,
            (1000, &reference),
            (900, &shorter),
            EvidenceStatus::unresolved(0.04, 0.09, Extremum::Supremum, Some(mask), "mask box").expect("a valid bracket"),
        );
        assert!(
            matches!(&supremum, Ok(acceptance) if acceptance.fidelity_certified),
            "the same bracket about the supremum must certify, got {supremum:?}"
        );
    }

    #[test]
    fn an_unrefuted_uncertified_fidelity_is_accepted_without_a_certificate() {
        let reference = decoded(0.05, 0.001, TOLERANCE);
        let candidate = decoded(0.08, 0.001, TOLERANCE);
        let decision = decide_proposal(
            ProposalKind::Split,
            (1000, &reference),
            (900, &candidate),
            EvidenceStatus::unresolved(0.04, f64::INFINITY, Extremum::Supremum, Some(vec![1.0, 1.0]), "mask box")
                .expect("a valid bracket"),
        );
        assert!(
            matches!(
                &decision,
                Ok(acceptance) if !acceptance.fidelity_certified
                    && matches!(acceptance.fidelity, EvidenceStatus::Unresolved { .. })
            ),
            "a lower witness below the tolerance with no derived upper bound is accepted as \
             uncertified, got {decision:?}"
        );
    }

    #[test]
    fn the_decoded_reference_and_candidate_must_meet_the_tolerance() {
        let shorter = decoded(0.08, 0.001, TOLERANCE);
        let violating_reference = decoded(0.2, 0.001, TOLERANCE);
        let inconsistent = decide_proposal(
            ProposalKind::Split,
            (1000, &violating_reference),
            (900, &shorter),
            certified(),
        );
        assert!(
            matches!(inconsistent, Err(ProposalRejection::ReferenceMissesTolerance(..))),
            "a decoded reference that misses the tolerance must refuse the loop, got {inconsistent:?}"
        );
        let reference = decoded(0.05, 0.001, TOLERANCE);
        // 0.099 ± 0.002 brackets the tolerance, so its verdict is Unresolved, never a pass.
        let unresolved = decoded(0.099, 0.002, TOLERANCE);
        let missing =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &unresolved), certified());
        assert!(
            matches!(missing, Err(ProposalRejection::CandidateMissesTolerance(..))),
            "a decoded candidate that does not prove it meets the tolerance must be rejected, got {missing:?}"
        );
        // Positive control: 0.097 + 0.002 = 0.099 is inside the tolerance by far more than one
        // ulp, so the outcome does not rest on rounding.
        let meeting_fidelity = decoded(0.097, 0.002, TOLERANCE);
        let meeting =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &meeting_fidelity), certified());
        assert!(meeting.is_ok(), "a candidate at the tolerance must be accepted, got {meeting:?}");

        // A candidate declared at a different tolerance is refused, even though it meets its own.
        let other_tolerance = decoded(0.05, 0.001, 0.2);
        let mismatched =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &other_tolerance), certified());
        assert!(
            matches!(mismatched, Err(ProposalRejection::CandidateMissesTolerance(..))),
            "a candidate scored at another tolerance must be refused, got {mismatched:?}"
        );
        // Positive control: the same figure at the reference's tolerance is accepted.
        let same_tolerance = decoded(0.05, 0.001, TOLERANCE);
        let matched =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &same_tolerance), certified());
        assert!(matched.is_ok(), "the same figure at the reference's tolerance must be accepted, got {matched:?}");
    }
}
