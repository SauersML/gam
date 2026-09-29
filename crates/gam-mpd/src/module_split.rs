//! Finest additive decompositions of a plain MLP block over every invertible
//! change of input coordinates (#2951, operation-first result 5).
//!
//! # Normal form
//!
//! A block `F(h) = L h + c + Σ_i v_i σ(w_iᵀ h + b_i)`, for `σ` the exact GELU
//! `t Φ(t)` or ReLU and an optional linear skip `L`, is first put in merged form
//! ([`MlpNormalForm::new`]):
//! - a unit with `w_i = 0` is the constant `v_i σ(b_i)` and moves into `c`;
//! - units with bitwise-equal affine forms `(w, b)` add their writes;
//! - a unit with the opposite form `(−w, −b)` uses `σ(−t) = σ(t) − t` (true for
//!   both activations): `v_k σ(−t) = v_k σ(t) − v_k t`, so its write adds and
//!   `−v_k wᵀ` moves into `L`, `−v_k b` into `c`;
//! - a merged write that cancels exactly drops the unit.
//!
//! Unmerged, the paired copy `σ(h) − σ(−h) = h` reads as `d` modules; merged it
//! is the linear block `L = I` with no units.
//!
//! # Exact finest decomposition
//!
//! The contract is additive: `F(h) = L h + c + Σ_a G_a(z_a)` with `z = S h` for
//! some invertible `S` and `z_a` a partition of its coordinates. Outputs add, so
//! no output projector enters, and `L h` is additive under every partition, so
//! the linear part never obstructs a decomposition. The Hessian is
//! `Σ_i v_i σ″(w_iᵀh + b_i) w_i w_iᵀ`; `σ″ = (2 − t²) φ(t)` for the GELU (a
//! Dirac mass at the kink for ReLU) is even, so the functions
//! `σ″(w_iᵀh + b_i)` of distinct merged forms are linearly independent
//! (distinct up to sign is exactly what merging guarantees). Mixed partials
//! across blocks must vanish, so every rank-one `v_i w_i w_iᵀ` is block
//! diagonal in `z`: each read lies in one block. Conversely a grouping of the
//! reads whose spans form a direct sum gives an `S` that makes the blocks
//! coordinate subspaces. So the finest decompositions are the finest direct-sum
//! groupings of the reads.
//!
//! With `W = U T` (`U` `m × r` orthonormal columns spanning `W`'s column space,
//! `T = UᵀW`, here from a resolved SVD so the null directions of `W` are
//! factored out first), the projector `Π = U Uᵀ` onto that column space is
//! invariant under `W → W S`, and a grouping is a direct sum iff `Π` is block
//! diagonal over it: an adapted basis makes `W S` block diagonal, whose column
//! space splits over the groups, and conversely a block-diagonal `Π` splits the
//! column space and so the rank. The finest direct-sum grouping is therefore
//! the connected components of the nonzero pattern of `Π`
//! ([`MlpNormalForm::additive_blocks`]). An entry is certified nonzero when it
//! clears its derived band (the Wedin bound on the projector's perturbation,
//! `U`'s orthonormality defect and the product's rounding); every other pair is
//! unresolved and reported, never decided: the finest partition joins certified
//! pairs only, the coarsest joins the unresolved ones too. The `d − r`
//! directions no unit reads are free and go to any block.
//!
//! # Optimal approximate split
//!
//! For a proposed unit subset `S` (indicator `D`), `M = Uᵀ D U` has eigenvalues
//! `λ ∈ [0, 1]`. Replacing `U` by `Û = D U P + (I − D) U (I − P)` for a
//! projector `P` makes `Ŵ = Û T` split exactly over `(S, Sᶜ)`, at loss
//! `‖U − Û‖²_F = tr M + tr(P (I − 2M))`, minimized by `P` onto the eigenvectors
//! with `λ > ½` (ties either way) at `E* = Σ min(λ, 1 − λ)`. The cut
//! `χ(S) = Σ_{i∈S, j∉S} Π_ij² = tr(M (I − M)) = Σ λ(1 − λ) ≥ E*/2`. All of it is
//! equivariant under `W → W S`. Since `|σ′| ≤ L_σ`,
//! `sup_{‖h‖≤R} ‖F(h) − F̂(h)‖ ≤ L_σ ‖V‖₂ ‖W − Ŵ‖₂ R`, and
//! `‖W − Ŵ‖₂ ≤ ‖T‖₂ √E* ≤ ‖T‖₂ √(2χ)`. `L_σ = Φ(√2) + √2 φ(√2)` for the exact
//! GELU and `1` for ReLU, from [`GaussianActivation::slope_bound_squared`].
//! [`AdditiveBlocks::laplacian_apply`] applies `𝓛 = diag(Π) − Π ⊙ Π` matrix-free
//! for callers that search for proposals; the search is theirs, and `S` is an
//! input here.
//!
//! # Relation to `response::interaction`
//!
//! `response::interaction` finds the finest additive blocks over declared input
//! ports, in the L2 sense under a declared product law. Here the blocks are
//! found over every invertible input frame at once, from the weights, and the
//! approximation contract is a worst-case sup-norm bound on a ball with no law.
//! Both read their blocks off one component owner,
//! `response::interaction::connected_components`. On trained weights the exact
//! pattern of `Π` is generically connected (operation-first result 5), so this
//! is a diagnostic, not a target.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use faer::Side;
use gam_linalg::faer_ndarray::{FaerLinalgError, FaerSvd, strict_symmetric_eigh};
use gam_linalg::roundoff::{
    SymmetricAssembly, accumulation_band, accumulation_growth, factor_singular_band, symmetric_spectrum_rounding_band,
};
use gam_math::gaussian_activation::{GaussianActivation, GaussianActivationError};
use gam_math::probability::normal_cdf_and_pdf;
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::state::{
    ObservabilityLetter, SpectralNormBounds, StateError, entrywise_band_norm, orthonormality_defect, reserve, spectral_norm_bounds,
};
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};
use gam_sae::response::interaction::connected_components;

/// What a module-split status ranges over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitDomain {
    /// The read matrix `W` of `units` merged units: an algebraic quantity of it.
    Reads { units: usize },
    /// Every input with `‖h‖ ≤ R`, the bound stated per unit radius.
    UnitBall { input_dimension: usize },
}

/// A merged unit's provenance: an original unit and whether its affine form is
/// the negation of the merged one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitSource {
    pub unit: usize,
    pub negated: bool,
}

/// The merged normal form `F(h) = L h + c + Σ_i v_i σ(w_iᵀ h + b_i)`.
#[derive(Clone, Debug)]
pub struct MlpNormalForm {
    activation: GaussianActivation,
    /// `L`, `d_out × d_in`: the skip plus `−v_k wᵀ` of every negated source.
    pub linear: Array2<f64>,
    /// Frobenius bound on `L`'s formation.
    pub linear_band: f64,
    /// `c`: `b_out`, the constants of zero-read units, and `−v_k b` of every
    /// negated source.
    pub offset: Array1<f64>,
    /// Merged reads `w_i`, `m × d_in`.
    pub reads: Array2<f64>,
    /// Merged biases `b_i`.
    pub biases: Array1<f64>,
    /// Merged writes `v_i`, `m × d_out`.
    pub writes: Array2<f64>,
    /// Frobenius bound on the rounding of each merged write.
    pub write_bands: Vec<f64>,
    /// The original units behind each merged unit.
    pub sources: Vec<Vec<UnitSource>>,
    /// Original units whose merged write cancels exactly.
    pub cancelled: Vec<usize>,
    /// Original units with a zero read, folded into `offset`.
    pub constant: Vec<usize>,
    /// An upper bound on `sup|σ′|`.
    pub lipschitz: f64,
}

/// The read frame `W = U T` and the projector `Π = U Uᵀ` with its entry bands.
#[derive(Clone, Debug)]
pub struct AdditiveBlocks {
    /// `U`, `m × r`, orthonormal columns.
    pub frame: Array2<f64>,
    /// `T = Σ Vᵀ`, `r × d_in`, so `W ≈ U T`.
    pub coordinates: Array2<f64>,
    /// `W`'s singular values in decreasing order; `r` of them resolved.
    pub singular_values: Vec<f64>,
    /// The resolved rank `r`: exact when it reaches `min(m, d_in)`.
    pub rank: EvidenceStatus<(), SplitDomain>,
    /// A bound on `‖Π̂ − Π‖₂` for the exact projector of the resolved rank
    /// (Wedin), plus `U`'s orthonormality defect.
    pub projector_band: f64,
    /// `d_in − r`: input directions no unit reads.
    pub free_input_dimension: usize,
    /// Pairs with `|Π_ij|` above its band.
    pub certified_pairs: usize,
    /// Pairs inside their band: not resolved from zero.
    pub unresolved_pairs: usize,
    /// Components of the certified pairs: the finest partition the arithmetic
    /// allows. Every pair across two of them is unresolved, not certified zero:
    /// a computed projector entry is never an exact zero.
    pub finest: Vec<Vec<usize>>,
    /// The unresolved pairs between each two finest components.
    pub unresolved_joins: Vec<UnresolvedJoin>,
    /// The optimal split of every finest component against the rest.
    pub components: Vec<OptimalSplit>,
}

/// Unresolved pairs between finest components `left < right`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnresolvedJoin {
    pub left: usize,
    pub right: usize,
    pub pairs: usize,
    /// The largest `|Π_ij|` among them, and its band.
    pub largest: f64,
    pub largest_band: f64,
}

/// The optimal exact split nearest to `W` for a proposed subset `S`.
#[derive(Clone, Debug)]
pub struct OptimalSplit {
    /// `S`, sorted.
    pub subset: Vec<usize>,
    /// Eigenvalues of `M = Uᵀ D U`, increasing.
    pub eigenvalues: Vec<f64>,
    /// `E* = Σ min(λ, 1 − λ) = min_P ‖U − Û‖²_F`.
    pub loss: EvidenceStatus<(), SplitDomain>,
    /// `χ(S) = Σ_{i∈S, j∉S} Π_ij² = Σ λ(1 − λ)`.
    pub cut: EvidenceStatus<(), SplitDomain>,
    /// Orthonormal rows (`k × r`) spanning `P`, the eigenvectors with `λ > ½`.
    pub projector_basis: Array2<f64>,
    /// `Ŵ = Û T`, `m × d_in`: its reads split exactly over `(S, Sᶜ)`.
    pub approximate_reads: Array2<f64>,
    /// `‖W − Ŵ‖₂` for the stored `Ŵ`.
    pub read_distance: SpectralNormBounds,
    /// `sup_{‖h‖≤1} ‖F(h) − F̂(h)‖ ≤ L_σ ‖V‖₂ ‖W − Ŵ‖₂` for `F̂` the block with
    /// reads `Ŵ`; scale by the declared radius `R`.
    pub native_bound: EvidenceStatus<(), SplitDomain>,
    /// The a-priori form `L_σ ‖V‖₂ ‖T‖₂ √(2χ)`, from `Π` alone.
    pub cut_bound: f64,
}

/// Why a module-split computation refused.
#[derive(Debug)]
pub enum ModuleSplitError {
    /// The activation has no Lipschitz owner here (SiLU).
    Activation { source: GaussianActivationError },
    /// Two dimensions that must agree do not.
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        found: usize,
    },
    /// A weight entry is not finite.
    NonFinite { context: &'static str },
    /// A subset names a unit outside `0..m`, or twice.
    InvalidSubset { unit: usize, units: usize },
    /// The block has no units after merging, so there is nothing to split.
    NoUnits,
    /// A shared linear-algebra owner refused.
    Linear { source: StateError },
    /// A status constructor refused.
    Evidence { source: EvidenceStatusError },
}

impl fmt::Display for ModuleSplitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Activation { source } => write!(formatter, "module split: activation: {source}"),
            Self::DimensionMismatch {
                context,
                expected,
                found,
            } => write!(formatter, "{context}: expected dimension {expected}, found {found}"),
            Self::NonFinite { context } => write!(formatter, "{context}: non-finite value"),
            Self::InvalidSubset { unit, units } => write!(
                formatter,
                "module split: subset unit {unit} is repeated or outside 0..{units}"
            ),
            Self::NoUnits => write!(formatter, "module split: no unit survives merging"),
            Self::Linear { source } => write!(formatter, "module split: {source}"),
            Self::Evidence { source } => write!(formatter, "module split: {source}"),
        }
    }
}

impl std::error::Error for ModuleSplitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linear { source } => Some(source),
            Self::Evidence { source } => Some(source),
            _ => None,
        }
    }
}

impl From<StateError> for ModuleSplitError {
    fn from(source: StateError) -> Self {
        Self::Linear { source }
    }
}

impl From<EvidenceStatusError> for ModuleSplitError {
    fn from(source: EvidenceStatusError) -> Self {
        Self::Evidence { source }
    }
}

fn linear_error(context: &'static str, source: FaerLinalgError) -> ModuleSplitError {
    ModuleSplitError::Linear {
        source: StateError::Svd { context, source },
    }
}

/// `σ(t)`: `t Φ(t)` for the exact GELU, `max(t, 0)` for ReLU.
pub fn activation_value(activation: GaussianActivation, t: f64) -> Result<f64, ModuleSplitError> {
    match activation {
        GaussianActivation::ExactGelu => Ok(t * normal_cdf_and_pdf(t).0),
        GaussianActivation::Relu => Ok(t.max(0.0)),
        GaussianActivation::Silu => Err(ModuleSplitError::Activation {
            source: GaussianActivationError::NoClosedForm { activation },
        }),
    }
}

/// An upper bound on `sup|σ′|`, from the activation owner's bound on its
/// square: the square root and the product each round by at most `u`.
fn slope_bound(activation: GaussianActivation) -> Result<f64, ModuleSplitError> {
    let squared = activation
        .slope_bound_squared()
        .map_err(|source| ModuleSplitError::Activation { source })?;
    Ok(squared.sqrt() * (1.0 + accumulation_growth(2)))
}

fn require_dimension(found: usize, expected: usize, context: &'static str) -> Result<(), ModuleSplitError> {
    if found == expected {
        Ok(())
    } else {
        Err(ModuleSplitError::DimensionMismatch {
            context,
            expected,
            found,
        })
    }
}

fn require_finite<'a>(
    values: impl IntoIterator<Item = &'a f64>,
    context: &'static str,
) -> Result<(), ModuleSplitError> {
    if values.into_iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ModuleSplitError::NonFinite { context })
    }
}

fn norm(values: ArrayView1<'_, f64>) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// The bit pattern of `(s w, s b)` with `s` making the first nonzero entry
/// positive and `-0.0` read as `0.0`, and whether `s = −1`. `None` for `w = 0`.
fn canonical_form(read: ArrayView1<'_, f64>, bias: f64) -> Option<(Vec<u64>, bool)> {
    let first = read.iter().find(|value| **value != 0.0)?;
    let negated = *first < 0.0;
    let sign = if negated { -1.0 } else { 1.0 };
    let key = read
        .iter()
        .chain(std::iter::once(&bias))
        .map(|value| (sign * value + 0.0).to_bits())
        .collect();
    Some((key, negated))
}

impl MlpNormalForm {
    /// The merged normal form of `F(h) = W_out σ(W_in h + b_in) + b_out + L h`,
    /// with `W_in` `n × d_in`, `W_out` `d_out × n` and the optional skip `L`.
    pub fn new(
        activation: GaussianActivation,
        w_in: ArrayView2<'_, f64>,
        b_in: ArrayView1<'_, f64>,
        w_out: ArrayView2<'_, f64>,
        b_out: ArrayView1<'_, f64>,
        skip: Option<ArrayView2<'_, f64>>,
    ) -> Result<Self, ModuleSplitError> {
        let lipschitz = slope_bound(activation)?;
        let (hidden, input) = w_in.dim();
        let output = w_out.nrows();
        require_dimension(b_in.len(), hidden, "normal form: b_in")?;
        require_dimension(w_out.ncols(), hidden, "normal form: W_out columns")?;
        require_dimension(b_out.len(), output, "normal form: b_out")?;
        require_finite(w_in.iter().chain(b_in.iter()), "normal form: W_in, b_in")?;
        require_finite(w_out.iter().chain(b_out.iter()), "normal form: W_out, b_out")?;
        let mut linear = Array2::<f64>::zeros((output, input));
        let mut absolute_linear = Array2::<f64>::zeros((output, input));
        if let Some(skip) = skip {
            require_dimension(skip.nrows(), output, "normal form: skip rows")?;
            require_dimension(skip.ncols(), input, "normal form: skip columns")?;
            require_finite(skip.iter(), "normal form: skip")?;
            linear.assign(&skip);
            absolute_linear.assign(&skip.mapv(f64::abs));
        }
        let mut offset = b_out.to_owned();

        let mut groups: HashMap<Vec<u64>, usize> = HashMap::new();
        let mut sources: Vec<Vec<UnitSource>> = Vec::new();
        let mut constant = Vec::new();
        for unit in 0..hidden {
            let read = w_in.row(unit);
            let Some((key, negated)) = canonical_form(read, b_in[unit]) else {
                // A zero read is the constant `v σ(b)`.
                offset.scaled_add(activation_value(activation, b_in[unit])?, &w_out.column(unit));
                constant.push(unit);
                continue;
            };
            let group = *groups.entry(key).or_insert_with(|| {
                sources.push(Vec::new());
                sources.len() - 1
            });
            sources[group].push(UnitSource { unit, negated });
        }
        let mut reads = Vec::new();
        let mut biases = Vec::new();
        let mut writes = Vec::new();
        let mut write_bands = Vec::new();
        let mut kept = Vec::new();
        let mut cancelled = Vec::new();
        let mut negated_terms = 0_usize;
        for group in sources {
            // The merged form is the canonical one, which every source's flag is relative to.
            let lead = group[0];
            let lead_sign = if lead.negated { -1.0 } else { 1.0 };
            let read = w_in.row(lead.unit).mapv(|value| lead_sign * value);
            let bias = lead_sign * b_in[lead.unit];
            let mut write = Array1::<f64>::zeros(output);
            let mut absolute_write = Array1::<f64>::zeros(output);
            for source in &group {
                let column = w_out.column(source.unit);
                write += &column;
                absolute_write += &column.mapv(f64::abs);
                if source.negated {
                    // `v σ(−t) = v σ(t) − v t` with `t = wᵀh + b`.
                    for (row, &weight) in column.iter().enumerate() {
                        for (col, &entry) in read.iter().enumerate() {
                            linear[[row, col]] -= weight * entry;
                            absolute_linear[[row, col]] += (weight * entry).abs();
                        }
                    }
                    offset.scaled_add(-bias, &column);
                    negated_terms += 1;
                }
            }
            // One rounded addition errs relative to its own result (`fl(a + b) = (a + b)(1 + δ)`),
            // so a merged pair that computes to zero cancels exactly; longer sums carry `γ_{n−1} Σ|v|`.
            let band = if group.len() <= 2 {
                accumulation_growth(1) * norm(write.view())
            } else {
                accumulation_growth(group.len() - 1) * norm(absolute_write.view())
            };
            if band == 0.0 && write.iter().all(|value| *value == 0.0) {
                cancelled.extend(group.iter().map(|source| source.unit));
                continue;
            }
            reads.push(read);
            biases.push(bias);
            writes.push(write);
            write_bands.push(band);
            kept.push(group);
        }
        let linear_band = entrywise_band_norm(
            absolute_linear.mapv(|magnitude| accumulation_band(negated_terms + 1, magnitude)),
        );
        let stack = |rows: &[Array1<f64>], width: usize| {
            let mut matrix = Array2::<f64>::zeros((rows.len(), width));
            for (index, row) in rows.iter().enumerate() {
                matrix.row_mut(index).assign(row);
            }
            matrix
        };
        Ok(Self {
            activation,
            linear,
            linear_band,
            offset,
            reads: stack(&reads, input),
            biases: Array1::from(biases),
            writes: stack(&writes, output),
            write_bands,
            sources: kept,
            cancelled,
            constant,
            lipschitz,
        })
    }

    /// The activation.
    pub fn activation(&self) -> GaussianActivation {
        self.activation
    }

    /// `L h + c + Σ_i v_i σ(w_iᵀ h + b_i)`.
    pub fn evaluate(&self, input: ArrayView1<'_, f64>) -> Result<Array1<f64>, ModuleSplitError> {
        self.evaluate_with_reads(self.reads.view(), input)
    }

    /// The block with its reads replaced by `reads` (`m × d_in`), such as an
    /// [`OptimalSplit::approximate_reads`].
    pub fn evaluate_with_reads(
        &self,
        reads: ArrayView2<'_, f64>,
        input: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, ModuleSplitError> {
        require_dimension(input.len(), self.linear.ncols(), "normal form: input")?;
        require_dimension(reads.nrows(), self.reads.nrows(), "normal form: read rows")?;
        require_dimension(reads.ncols(), self.reads.ncols(), "normal form: read columns")?;
        let mut value = self.linear.dot(&input) + &self.offset;
        let pre = reads.dot(&input) + &self.biases;
        for (unit, &argument) in pre.iter().enumerate() {
            value.scaled_add(activation_value(self.activation, argument)?, &self.writes.row(unit));
        }
        Ok(value)
    }

    /// The block as the letters of one weighted observability step: `L` and the
    /// merged rank-one units `v_i w_iᵀ`.
    pub fn observability_letters(&self) -> Vec<ObservabilityLetter<'_>> {
        vec![
            ObservabilityLetter::Linear(self.linear.view()),
            ObservabilityLetter::Units {
                reads: self.reads.view(),
                writes: self.writes.view(),
            },
        ]
    }

    /// The finest additive blocks: the resolved read frame, the pattern of `Π`
    /// with derived bands, its components, and the optimal split of every
    /// finest component against the rest.
    pub fn additive_blocks(&self, governor: &MemoryGovernor) -> Result<AdditiveBlocks, ModuleSplitError> {
        let (units, input) = self.reads.dim();
        if units == 0 {
            return Err(ModuleSplitError::NoUnits);
        }
        // The decomposition's copy, `U`, `Vᵀ`, `Π` and its absolute bound.
        let working = reserve(governor, units, units.max(input), 5, "additive blocks")?;
        let (left, sigma, right) = self
            .reads
            .svd(true, true)
            .map_err(|source| linear_error("additive blocks: reads", source))?;
        let (left, right) = match (left, right) {
            (Some(left), Some(right)) => (left, right),
            _ => {
                return Err(linear_error(
                    "additive blocks: reads",
                    FaerLinalgError::SvdNoConvergence {
                        context: "additive blocks: reads",
                    },
                ));
            }
        };
        let mut order: Vec<usize> = (0..sigma.len()).collect();
        order.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
        let singular_values: Vec<f64> = order.iter().map(|&index| sigma[index]).collect();
        let sigma_max = singular_values.first().copied().unwrap_or(0.0);
        let backward = factor_singular_band(units, input, sigma_max);
        let rank = singular_values.iter().filter(|&&value| value > backward).count();
        let ceiling = units.min(input);
        let domain = SplitDomain::Reads { units };
        let rank_status = if rank == ceiling {
            EvidenceStatus::exact(rank as f64, 0.0, ExactBasis::Algebraic, None, domain)?
        } else {
            EvidenceStatus::unresolved(rank as f64, ceiling as f64, Extremum::Supremum, None, domain)?
        };
        let mut frame = Array2::<f64>::zeros((units, rank));
        let mut coordinates = Array2::<f64>::zeros((rank, input));
        for (column, &index) in order.iter().take(rank).enumerate() {
            frame.column_mut(column).assign(&left.column(index));
            coordinates
                .row_mut(column)
                .assign(&right.row(index).mapv(|value| value * sigma[index]));
        }
        drop(left);
        drop(right);
        // Wedin: the exact rank-`r` projector moves by at most `e / (σ_r − e)`.
        let sigma_r = singular_values.get(rank.wrapping_sub(1)).copied().unwrap_or(0.0);
        let wedin = if rank == 0 { 0.0 } else { (backward / (sigma_r - backward)).min(1.0) };
        let projector_band = wedin + orthonormality_defect(&frame.t().to_owned());
        let projector = frame.dot(&frame.t());
        let absolute_frame = frame.mapv(f64::abs);
        let magnitude = absolute_frame.dot(&absolute_frame.t());
        let band = |i: usize, j: usize| projector_band + accumulation_band(rank, magnitude[[i, j]]);
        let pairs = || (0..units).flat_map(move |i| ((i + 1)..units).map(move |j| (i, j)));
        let certified = |&(i, j): &(usize, usize)| projector[[i, j]].abs() > band(i, j);
        let certified_pairs = pairs().filter(certified).count();
        let unresolved_pairs = units * (units - 1) / 2 - certified_pairs;
        let finest = connected_components(units, pairs().filter(certified));
        let mut label = vec![0; units];
        for (component, members) in finest.iter().enumerate() {
            for &unit in members {
                label[unit] = component;
            }
        }
        let mut joins: BTreeMap<(usize, usize), UnresolvedJoin> = BTreeMap::new();
        for (i, j) in pairs() {
            if label[i] != label[j] {
                let (left, right) = (label[i].min(label[j]), label[i].max(label[j]));
                let join = joins.entry((left, right)).or_insert(UnresolvedJoin {
                    left,
                    right,
                    pairs: 0,
                    largest: 0.0,
                    largest_band: 0.0,
                });
                join.pairs += 1;
                if projector[[i, j]].abs() >= join.largest {
                    join.largest = projector[[i, j]].abs();
                    join.largest_band = band(i, j);
                }
            }
        }
        drop(projector);
        drop(magnitude);
        drop(working);
        let mut blocks = AdditiveBlocks {
            frame,
            coordinates,
            singular_values,
            rank: rank_status,
            projector_band,
            free_input_dimension: input - rank,
            certified_pairs,
            unresolved_pairs,
            finest,
            unresolved_joins: joins.into_values().collect(),
            components: Vec::new(),
        };
        let mut components = Vec::with_capacity(blocks.finest.len());
        for members in &blocks.finest {
            components.push(self.optimal_split(governor, &blocks, members)?);
        }
        blocks.components = components;
        Ok(blocks)
    }

    /// The optimal exact split nearest to the reads for the proposed subset
    /// `subset` (merged unit indices) against its complement.
    pub fn optimal_split(
        &self,
        governor: &MemoryGovernor,
        blocks: &AdditiveBlocks,
        subset: &[usize],
    ) -> Result<OptimalSplit, ModuleSplitError> {
        let (units, input) = self.reads.dim();
        require_dimension(blocks.frame.nrows(), units, "optimal split: frame rows")?;
        let rank = blocks.frame.ncols();
        let mut inside = vec![false; units];
        for &unit in subset {
            if unit >= units || inside[unit] {
                return Err(ModuleSplitError::InvalidSubset { unit, units });
            }
            inside[unit] = true;
        }
        let mut sorted = subset.to_vec();
        sorted.sort_unstable();
        // `Û`, `Ŵ`, `W − Ŵ` and the `r × r` pieces.
        let working = reserve(governor, units.max(rank).max(1), input.max(rank).max(1), 5, "optimal split")?;
        let frame = &blocks.frame;
        let mut selected = frame.clone();
        for (unit, mut row) in selected.rows_mut().into_iter().enumerate() {
            if !inside[unit] {
                row.fill(0.0);
            }
        }
        let gram = frame.t().dot(&selected);
        // `(G + Gᵀ)/2` puts one rounded value in both triangles.
        let symmetric = (&gram + &gram.t()) * 0.5;
        let (values, vectors) = strict_symmetric_eigh(&symmetric, SymmetricAssembly::Mirrored, Side::Lower)
            .map_err(|source| linear_error("optimal split: M", source))?;
        let eigenvalues: Vec<f64> = values.to_vec();
        let kept: Vec<usize> = (0..rank).filter(|&index| values[index] > 0.5).collect();
        let mut projector_basis = Array2::<f64>::zeros((kept.len(), rank));
        for (row, &index) in kept.iter().enumerate() {
            projector_basis.row_mut(row).assign(&vectors.column(index));
        }
        let loss_value: f64 = eigenvalues.iter().map(|&lambda| lambda.min(1.0 - lambda).max(0.0)).sum();
        let cut_value: f64 = eigenvalues.iter().map(|&lambda| (lambda * (1.0 - lambda)).max(0.0)).sum();

        // Every eigenvalue is within `ρ` of one of the exact `M` of the computed `U`;
        // the computed `U` is within `√(2r)·wedin + defect` of an exact frame in Frobenius
        // norm, and `U ↦ U − Û` has norm at most one, so `√E*` moves by at most that.
        let absolute_frame = frame.mapv(f64::abs);
        let mut absolute_selected = absolute_frame.clone();
        for (unit, mut row) in absolute_selected.rows_mut().into_iter().enumerate() {
            if !inside[unit] {
                row.fill(0.0);
            }
        }
        let formation = entrywise_band_norm(
            absolute_frame
                .t()
                .dot(&absolute_selected)
                .mapv(|magnitude| accumulation_band(units, magnitude)),
        );
        let spectrum = symmetric_spectrum_rounding_band(&eigenvalues) + formation;
        let frame_error = (2.0 * rank as f64).sqrt() * blocks.projector_band;
        let square_root_band = |value: f64| {
            let root = value.max(0.0).sqrt();
            let high = (root + frame_error).powi(2) + rank as f64 * spectrum;
            let low = (root - frame_error).max(0.0).powi(2) - rank as f64 * spectrum;
            (value - low).max(high - value).max(0.0)
        };
        let domain = SplitDomain::Reads { units };
        let loss = EvidenceStatus::exact(loss_value, square_root_band(loss_value), ExactBasis::Algebraic, None, domain)?;
        let cut = EvidenceStatus::exact(cut_value, square_root_band(cut_value), ExactBasis::Algebraic, None, domain)?;

        let coordinates_in = frame.dot(&projector_basis.t());
        let projected = coordinates_in.dot(&projector_basis);
        let mut approximate_frame = frame.clone();
        for unit in 0..units {
            if inside[unit] {
                approximate_frame.row_mut(unit).assign(&projected.row(unit));
            } else {
                let complement = &frame.row(unit) - &projected.row(unit);
                approximate_frame.row_mut(unit).assign(&complement);
            }
        }
        let approximate_reads = approximate_frame.dot(&blocks.coordinates);
        let difference = &self.reads - &approximate_reads;
        let difference_band = entrywise_band_norm(
            (self.reads.mapv(f64::abs) + approximate_reads.mapv(f64::abs))
                .mapv(|magnitude| accumulation_growth(1) * magnitude),
        );
        let read_distance = spectral_norm_bounds(governor, &difference, difference_band, "optimal split: W − Ŵ")?;
        let write_band = self.write_bands.iter().map(|band| band * band).sum::<f64>().sqrt();
        let write_norm = spectral_norm_bounds(governor, &self.writes, write_band, "optimal split: V")?;
        let coordinate_norm = blocks.singular_values.first().copied().unwrap_or(0.0)
            + factor_singular_band(units, input, blocks.singular_values.first().copied().unwrap_or(0.0));
        drop(working);
        let native_upper = self.lipschitz * write_norm.upper * read_distance.upper * (1.0 + accumulation_growth(2));
        let native_bound = EvidenceStatus::uniform_bound(
            native_upper,
            native_upper - self.lipschitz * write_norm.lower * read_distance.lower,
            SplitDomain::UnitBall {
                input_dimension: input,
            },
        )?;
        let cut_upper = cut.upper_bound().unwrap_or(f64::INFINITY);
        let cut_bound = self.lipschitz * write_norm.upper * coordinate_norm * (2.0 * cut_upper).sqrt()
            * (1.0 + accumulation_growth(4));
        Ok(OptimalSplit {
            subset: sorted,
            eigenvalues,
            loss,
            cut,
            projector_basis,
            approximate_reads,
            read_distance,
            native_bound,
            cut_bound,
        })
    }
}

impl AdditiveBlocks {
    /// `𝓛 q` for the Laplacian `𝓛 = diag(Π) − Π ⊙ Π` of the pair weights
    /// `Π_ij²`, matrix-free: `(𝓛q)_i = ‖u_i‖² q_i − u_iᵀ (Uᵀ diag(q) U) u_i`, in
    /// `O(m r²)` time and `O(m r + r²)` memory.
    pub fn laplacian_apply(&self, q: ArrayView1<'_, f64>) -> Result<Array1<f64>, ModuleSplitError> {
        let units = self.frame.nrows();
        require_dimension(q.len(), units, "laplacian: vector")?;
        let rank = self.frame.ncols();
        let mut weighted = Array2::<f64>::zeros((rank, rank));
        for (unit, row) in self.frame.rows().into_iter().enumerate() {
            for a in 0..rank {
                for b in 0..rank {
                    weighted[[a, b]] += q[unit] * row[a] * row[b];
                }
            }
        }
        Ok(Array1::from_shape_fn(units, |unit| {
            let row = self.frame.row(unit);
            let degree = row.dot(&row);
            degree * q[unit] - row.dot(&weighted.dot(&row))
        }))
    }
}

#[cfg(test)]
#[path = "module_split_tests.rs"]
mod tests;
