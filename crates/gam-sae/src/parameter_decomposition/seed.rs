//! Exact initial decompositions from native tensors: a residual MLP block in
//! component coordinates, and the start states of a registered tensor on its
//! residual anchor (#2951).
//!
//! # Rank-revealing reads
//!
//! A weight `W ∈ ℝ^{p×c}` is seeded with `c` components through the read
//! `R = V̂ᵀ`, whose rows are the right singular vectors of `W` in descending order
//! of their singular values `σ̂₁ ≥ … ≥ σ̂_c` (gam-linalg `factor_rank_partition`,
//! which completes a wide weight with zero rows that add no energy). `R` is
//! square with orthonormal rows, so it covers the input, and the exact write of
//! `rewrite::ExactFactor` is unique: `I − R R⁺ = 0`, so every candidate gives
//! `U = W R⁻¹ = W V̂`, and the seed passes the zero candidate. Column `k` of `U`
//! is `W v̂_k`, so component `k` is the `k`-th principal term `σ_k u_k v_kᵀ` of `W`
//! up to the backward error below.
//!
//! # Rank and clusters
//!
//! A backward-stable SVD returns the exact singular triples of `W + E` with
//! `‖E‖₂ ≤ b = max(p, c)·ε·σ̂₁` ([`factor_singular_band`]), and by Weyl each `σ̂_k`
//! is within `b` of `σ_k`.
//! * The rank counts the `σ̂_k > b`: the components the decomposition resolved from
//!   zero. They lead the component order.
//! * Two adjacent singular values with `σ̂_k − σ̂_{k+1} ≤ 2b` may be equal in `W`.
//!   Wedin's theorem bounds the rotation of a run's singular subspaces by `b / δ`,
//!   where `δ` is the separation of the run's computed singular values from the
//!   rest of the exact spectrum, at least the computed gap less `b`. The bound is
//!   below one only for a computed gap above `2b`.
//!
//! Inside such a run, `(U Q, Qᵀ R)` for any orthogonal `Q` on the run gives the
//! same `W` to working precision, so the run's individual components are not
//! identified, and a mask `M` on the run acts as `U Q M Qᵀ R`. The only masks
//! invariant under every `Q` are `cI` on the run (P1, with the run's orthogonal
//! group in place of `GL(r)`). [`SingularSeed::clusters`] reports the maximal runs:
//! each is one mask group, and independent masks are intrinsic only across runs.
//!
//! # All on
//!
//! The seeded block is a `rewrite::ComponentMlp`. With both factors all on it
//! executes the original tensors on their original path, bit for bit.
//!
//! # Memory
//!
//! The read of a `p × c` weight is `c × c`, the least an exact component program can
//! hold, since coverage needs at least `c` components (P5). [`seed_mlp`] charges its
//! byte ledger to the process [`MemoryGovernor`] before any decomposition, so a
//! block that cannot fit is a typed refusal. The ledger counts the arrays this file
//! and the owners it calls visibly materialize. The owners' factorization
//! workspaces come on top of it, so the ledger refuses a block that cannot fit and
//! does not certify one that can. [`seed_kept_factor`] and [`seed_removed`] charge
//! their scratch the same way before any decomposition, and charge the basis they
//! keep once its rank is known.
//!
//! # Start states
//!
//! The fit declares one residual state for every row, generator and oracle, and calls
//! the builder of that state; none of them takes the state as an argument.
//! * Kept (`m_Δ = 1`):
//!   * [`seed_kept_literal`]: no components, so `Θ(m) = m_Δ Θ_*`. It holds no basis and
//!     applies matrix-free at any width.
//!   * [`seed_kept_factor`]: the `r` resolved rank-one terms `û_k r̂_kᵀ` of the
//!     rank-revealing exact factor, one component each. The unresolved tail stays in
//!     the residual, carried exactly.
//! * Removed (`m_Δ = 0`): [`seed_removed`], one component `P = Û R̂` of rank `c`. It
//!   executes components only, so it reproduces `Θ_*` algebraically, not bitwise.
//!
//! # Reproduction bound
//!
//! [`RemovedSeed::reproduction_bound`] bounds the parameter error `E = Θ_* − Û R̂` of
//! the computed factors. With `D = fl(Θ_* − fl(Û R̂))`, entry `(i, j)` unfolds into
//! `c + 1` summands, `θ_ij` and `c` products. Each product rounds once and at most `c`
//! additions follow, so `|d_ij − e_ij| ≤ γ_{c+1} A_ij` with
//! `A_ij = |θ_ij| + Σ_k |û_ik| |r̂_kj|`, whichever order a kernel sums in (Higham, ASNA
//! Lemma 3.1). The computed `Â` accumulates the same nonnegative terms along the same
//! depth, so `A ≤ Â / (1 − γ_{c+1})`, and `‖E‖_F ≤ ‖t‖_F` with
//! `t_ij = |d_ij| + γ_{c+1} Â_ij / (1 − γ_{c+1})`.
//!
//! Every later step is a nonnegative operation:
//! * forming the band (the divisor, the product, the quotient and `γ` itself: four);
//! * the sum with `|d_ij|` (one);
//! * the square, which doubles the depth before it and adds one;
//! * the `p·c − 1` additions and the square root.
//!
//! On that path of depth `K = 2(c + 6) + p·c + 1` the exact norm is at most the computed
//! one over `1 − γ_K`, and `γ_{K+2}` also covers forming the divisor and dividing.
//!
//! The figure audits the start artifact; it is not an output roundoff. Executing the
//! decoded start already puts any output effect of `Θ_* ≠ Û R̂` inside the measured
//! distortion. The figure enters a certificate only as `L·bound`, through a derived
//! Lipschitz constant `L` of the readout in `Θ` stated by whoever holds it.

use std::fmt;
use std::ops::Range;

use gam_linalg::faer_ndarray::FaerLinalgError;
use gam_linalg::roundoff::{accumulation_growth, factor_rank_partition, factor_singular_band};
use gam_math::gaussian_activation::GaussianActivation;
use gam_runtime::resource::{Governed, MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{Array1, Array2, ArrayView2, s};

use super::apply::FactoredEdit;
use super::lift::{ComponentCoefficients, LiftError, ResidualAnchor, TensorId, TensorRegistry};
use super::rewrite::{
    ComponentMlp, ComponentRead, ExactFactor, FactorRefusal, MlpFactor, NativeMlp, RewriteError,
};

/// The singular spectrum of a seeded weight, one entry per component.
#[derive(Clone, Debug, PartialEq)]
pub struct SingularSeed {
    singular_values: Vec<f64>,
    band: f64,
    rank: usize,
    clusters: Vec<Range<usize>>,
}

impl SingularSeed {
    /// `σ̂_k` of component `k`, descending.
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    /// The backward-error band `b = max(p, c)·ε·σ̂₁` of the decomposition.
    pub fn band(&self) -> f64 {
        self.band
    }

    /// The number of components resolved from zero, `#{k : σ̂_k > b}`.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The maximal runs of adjacent components whose singular values are at most
    /// `2b` apart, in component order. They partition the components, and each is
    /// one mask group (see the module documentation).
    pub fn clusters(&self) -> &[Range<usize>] {
        &self.clusters
    }
}

/// The maximal runs of `singular_values` (descending) whose adjacent gaps are at
/// most `2·band`.
fn unresolved_gap_clusters(singular_values: &[f64], band: f64) -> Vec<Range<usize>> {
    let mut clusters = Vec::new();
    let mut start = 0;
    for component in 1..singular_values.len() {
        if singular_values[component - 1] - singular_values[component] > 2.0 * band {
            clusters.push(start..component);
            start = component;
        }
    }
    if start < singular_values.len() {
        clusters.push(start..singular_values.len());
    }
    clusters
}

/// Why a weight has no rank-revealing read.
#[derive(Debug)]
pub enum SpectrumRefusal {
    /// The weight has no rows or no columns.
    Empty { rows: usize, cols: usize },
    /// The weight has a non-finite entry.
    NonFinite,
    /// The singular value decomposition failed.
    Decomposition(FaerLinalgError),
}

impl fmt::Display for SpectrumRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { rows, cols } => {
                write!(formatter, "a {rows}×{cols} weight has no component to seed")
            }
            Self::NonFinite => formatter.write_str("the weight has a non-finite entry"),
            Self::Decomposition(error) => {
                write!(formatter, "the weight's singular value decomposition failed: {error}")
            }
        }
    }
}

impl std::error::Error for SpectrumRefusal {}

/// A weight's rank-revealing read `R = V̂ᵀ` and its singular spectrum (see the
/// module documentation).
#[derive(Clone, Debug)]
pub struct RankRevealingRead {
    read: Array2<f64>,
    spectrum: SingularSeed,
}

impl RankRevealingRead {
    pub fn new(weight: &Array2<f64>) -> Result<Self, SpectrumRefusal> {
        let (rows, cols) = weight.dim();
        if rows == 0 || cols == 0 {
            return Err(SpectrumRefusal::Empty { rows, cols });
        }
        if !weight.iter().all(|value| value.is_finite()) {
            return Err(SpectrumRefusal::NonFinite);
        }
        let partition = factor_rank_partition(weight).map_err(SpectrumRefusal::Decomposition)?;
        let largest = partition.singular_values.first().copied().unwrap_or(0.0);
        let band = factor_singular_band(rows, cols, largest);
        let clusters = unresolved_gap_clusters(&partition.singular_values, band);
        Ok(Self {
            read: partition.right_vectors,
            spectrum: SingularSeed {
                singular_values: partition.singular_values,
                band,
                rank: partition.rank,
                clusters,
            },
        })
    }

    /// The read `R`, `c × c`: row `k` is the right singular vector of `σ̂_k`.
    pub fn read(&self) -> ArrayView2<'_, f64> {
        self.read.view()
    }

    pub fn spectrum(&self) -> &SingularSeed {
        &self.spectrum
    }

    pub fn into_parts(self) -> (Array2<f64>, SingularSeed) {
        (self.read, self.spectrum)
    }
}

/// A residual MLP block seeded in rank-revealing component coordinates.
#[derive(Clone, Debug)]
pub struct MlpSeed {
    block: ComponentMlp,
    read_in: SingularSeed,
    write_out: SingularSeed,
}

impl MlpSeed {
    /// The component program `h + Ū M̄ R̄ σ(U M R h + b₁) + b₂` through the
    /// rank-revealing reads of both weights.
    pub fn block(&self) -> &ComponentMlp {
        &self.block
    }

    /// The spectrum of `W₁`, one entry per read-in component.
    pub fn read_in(&self) -> &SingularSeed {
        &self.read_in
    }

    /// The spectrum of `W₂`, one entry per write-out component.
    pub fn write_out(&self) -> &SingularSeed {
        &self.write_out
    }
}

/// A refused MLP seed.
#[derive(Debug)]
pub enum SeedError {
    /// A weight of the block has no rank-revealing read.
    Spectrum {
        factor: MlpFactor,
        refusal: SpectrumRefusal,
    },
    /// The block's byte ledger overflows `usize`.
    SizeOverflow,
    /// The process memory governor refused the block's byte ledger.
    Memory(MemoryReservationError),
    /// The component program refused the seeded block.
    Rewrite(RewriteError),
}

impl fmt::Display for SeedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spectrum { factor, refusal } => {
                write!(formatter, "the {factor} weight has no rank-revealing read: {refusal}")
            }
            Self::SizeOverflow => formatter.write_str("the MLP seed's byte ledger overflows usize"),
            Self::Memory(error) => write!(formatter, "the memory governor refused the MLP seed: {error}"),
            Self::Rewrite(error) => write!(formatter, "the seeded component program refused: {error}"),
        }
    }
}

impl std::error::Error for SeedError {}

/// Seeds the block `h' = h + W₂ σ(W₁ h + b₁) + b₂` in rank-revealing component
/// coordinates (see the module documentation).
///
/// The result holds the governor charge of the seeded factors. The shapes are
/// checked by [`NativeMlp::new`], after both decompositions.
pub fn seed_mlp(
    read_in: Array2<f64>,
    bias_in: Array1<f64>,
    write_out: Array2<f64>,
    bias_out: Array1<f64>,
    activation: GaussianActivation,
) -> Result<Governed<MlpSeed>, SeedError> {
    let ledger = seed_ledger(read_in.dim(), write_out.dim()).ok_or(SeedError::SizeOverflow)?;
    let (factors, scratch) = reserve_seed(ledger)?;
    let spectrum = |factor: MlpFactor, weight: &Array2<f64>| {
        RankRevealingRead::new(weight)
            .map(RankRevealingRead::into_parts)
            .map_err(|refusal| SeedError::Spectrum { factor, refusal })
    };
    let (read_in_read, read_in_spectrum) = spectrum(MlpFactor::ReadIn, &read_in)?;
    let (write_out_read, write_out_spectrum) = spectrum(MlpFactor::WriteOut, &write_out)?;
    // A square read has one exact write, so the zero candidate selects nothing.
    let read_in_candidate = Array2::<f64>::zeros((read_in.nrows(), read_in_read.nrows()));
    let write_out_candidate = Array2::<f64>::zeros((write_out.nrows(), write_out_read.nrows()));
    let native = NativeMlp::new(read_in, bias_in, write_out, bias_out, activation)
        .map_err(|mismatch| SeedError::Rewrite(RewriteError::Shape(mismatch)))?;
    let block = ComponentMlp::new(
        native,
        ComponentRead {
            read: read_in_read.view(),
            candidate_write: read_in_candidate.view(),
        },
        ComponentRead {
            read: write_out_read.view(),
            candidate_write: write_out_candidate.view(),
        },
    )
    .map_err(SeedError::Rewrite)?;
    // The block holds its own copies of the reads; the scratch charge ends with them.
    drop((read_in_read, write_out_read, read_in_candidate, write_out_candidate, scratch));
    Ok(factors.bind(MlpSeed {
        block,
        read_in: read_in_spectrum,
        write_out: write_out_spectrum,
    }))
}

/// What seeding a block charges: the seeded factors it leaves, and the rest of its
/// peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SeedLedger {
    persistent_bytes: usize,
    scratch_bytes: usize,
}

/// The ledger of seeding a block whose read-in weight is `p₁ × c₁` and whose
/// write-out weight is `p₂ × c₂`, or `None` on overflow.
///
/// A `p × c` weight seeds `c` components. In `f64` entries:
/// * its decomposition holds the zero completion of a wide weight (`c²` when
///   `p < c`), the right singular vectors and their descending copy (`2c²`);
/// * its exact write holds the QR factors and the read's copy (`3c²`), plus the
///   residual, its transposed solve, the solve's product and the write (`4pc`);
/// * the factor it leaves is the write and the read (`pc + c²`).
///
/// The first read is held through the second decomposition. Both reads and both
/// zero candidates (`c₁² + c₂² + p₁c₁ + p₂c₂`) are held through both solves, and
/// the first factor through the second solve. The peak is the largest phase.
fn seed_ledger(read_in: (usize, usize), write_out: (usize, usize)) -> Option<SeedLedger> {
    let square = |weight: (usize, usize)| weight.1.checked_mul(weight.1);
    let rectangle = |weight: (usize, usize)| weight.0.checked_mul(weight.1);
    let decomposition = |weight: (usize, usize)| -> Option<usize> {
        let completion = if weight.0 < weight.1 { square(weight)? } else { 0 };
        completion.checked_add(square(weight)?.checked_mul(2)?)
    };
    let solve = |weight: (usize, usize)| -> Option<usize> {
        square(weight)?
            .checked_mul(3)?
            .checked_add(rectangle(weight)?.checked_mul(4)?)
    };
    let factor = |weight: (usize, usize)| -> Option<usize> {
        rectangle(weight)?.checked_add(square(weight)?)
    };
    let held = square(read_in)?
        .checked_add(square(write_out)?)?
        .checked_add(rectangle(read_in)?)?
        .checked_add(rectangle(write_out)?)?;
    let phases = [
        decomposition(read_in)?,
        square(read_in)?.checked_add(decomposition(write_out)?)?,
        held.checked_add(solve(read_in)?)?,
        held.checked_add(factor(read_in)?)?
            .checked_add(solve(write_out)?)?,
    ];
    let peak = phases.into_iter().max()?;
    let kept = factor(read_in)?.checked_add(factor(write_out)?)?;
    let bytes = std::mem::size_of::<f64>();
    Some(SeedLedger {
        persistent_bytes: kept.checked_mul(bytes)?,
        scratch_bytes: peak.checked_sub(kept)?.checked_mul(bytes)?,
    })
}

/// Charges the ledger to the process memory governor: the factors' charge, then the
/// scratch charge.
fn reserve_seed(ledger: SeedLedger) -> Result<(MemoryReservation, MemoryReservation), SeedError> {
    let governor = MemoryGovernor::global();
    let factors = governor
        .try_reserve(ledger.persistent_bytes, "MLP seed factors")
        .map_err(SeedError::Memory)?;
    let scratch = governor
        .try_reserve(ledger.scratch_bytes, "MLP seed scratch")
        .map_err(SeedError::Memory)?;
    Ok((factors, scratch))
}

/// Why a registered tensor has no seeded start.
#[derive(Debug)]
pub enum StartSeedError {
    /// The tensor has no rank-revealing read.
    Spectrum(SpectrumRefusal),
    /// The tensor has no exact factor through its read.
    Factor(FactorRefusal),
    /// The registry or the anchor refused the start.
    Lift(LiftError),
    /// The start's byte ledger overflows `usize`.
    SizeOverflow,
    /// The process memory governor refused the start's byte ledger.
    Memory(MemoryReservationError),
}

impl fmt::Display for StartSeedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spectrum(refusal) => write!(formatter, "the tensor has no rank-revealing read: {refusal}"),
            Self::Factor(refusal) => write!(formatter, "the tensor has no exact factor: {refusal}"),
            Self::Lift(error) => write!(formatter, "the anchor refused the start: {error}"),
            Self::SizeOverflow => formatter.write_str("the start's byte ledger overflows usize"),
            Self::Memory(error) => write!(formatter, "the memory governor refused the start: {error}"),
        }
    }
}

impl std::error::Error for StartSeedError {}

/// The Kept literal start of the registered matrix tensor `storage` (see the module
/// documentation).
pub fn seed_kept_literal<'a>(
    registry: &TensorRegistry,
    storage: TensorId,
    native: ArrayView2<'a, f64>,
) -> Result<ResidualAnchor<'a>, LiftError> {
    let (rows, cols) = native.dim();
    let basis = FactoredEdit::new(Array2::zeros((rows, 0)), Array2::zeros((cols, 0)))?;
    ResidualAnchor::new(registry, storage, native, basis, Vec::new(), ComponentCoefficients::Basis)
}

/// A Kept start whose components are the resolved terms of the rank-revealing exact
/// factor (see the module documentation).
#[derive(Clone, Debug)]
pub struct KeptFactorSeed<'a> {
    anchor: ResidualAnchor<'a>,
    spectrum: SingularSeed,
    right: Array2<f64>,
}

impl<'a> KeptFactorSeed<'a> {
    /// The anchor. Basis matrix `k` is `û_k r̂_kᵀ` for `k < rank`, one component each.
    pub fn anchor(&self) -> &ResidualAnchor<'a> {
        &self.anchor
    }

    /// The spectrum of the seeded tensor.
    pub fn spectrum(&self) -> &SingularSeed {
        &self.spectrum
    }

    /// The right factor `r̂_k` of basis matrix `k`, `c × 1`, in basis order: the fixed
    /// right factors of a conditionally Gaussian coefficient block.
    pub fn right_factors(&self) -> Vec<ArrayView2<'_, f64>> {
        (0..self.right.ncols())
            .map(|term| self.right.slice(s![.., term..term + 1]))
            .collect()
    }
}

/// Seeds the Kept factor start of the registered matrix tensor `storage`. The result
/// holds the governor charge of the basis it keeps.
pub fn seed_kept_factor<'a>(
    registry: &TensorRegistry,
    storage: TensorId,
    native: ArrayView2<'a, f64>,
) -> Result<Governed<KeptFactorSeed<'a>>, StartSeedError> {
    let (rows, cols) = native.dim();
    let scratch = reserve_start(start_scratch_bytes(rows, cols, false), "kept factor start scratch")?;
    let (factor, spectrum) = exact_factor(native)?;
    let rank = spectrum.rank();
    let kept = reserve_start(start_basis_bytes(rows, cols, rank, true), "kept factor start basis")?;
    let left = factor.write().slice(s![.., ..rank]).to_owned();
    let right = factor.read().slice(s![..rank, ..]).t().to_owned();
    // The basis holds its own copies; the factor and its scratch end here.
    drop((factor, scratch));
    let basis = FactoredEdit::new(left, right.clone())
        .map_err(|error| StartSeedError::Lift(LiftError::from(error)))?;
    let anchor = ResidualAnchor::new(
        registry,
        storage,
        native,
        basis,
        vec![1; rank],
        ComponentCoefficients::Basis,
    )
    .map_err(StartSeedError::Lift)?;
    Ok(kept.bind(KeptFactorSeed {
        anchor,
        spectrum,
        right,
    }))
}

/// The Removed start of a registered matrix tensor: one component through the exact
/// factor (see the module documentation).
#[derive(Clone, Debug)]
pub struct RemovedSeed<'a> {
    anchor: ResidualAnchor<'a>,
    spectrum: SingularSeed,
    reproduction_bound: f64,
}

impl<'a> RemovedSeed<'a> {
    /// The anchor: one basis matrix `Û R̂` of rank `c`, one component.
    pub fn anchor(&self) -> &ResidualAnchor<'a> {
        &self.anchor
    }

    /// The spectrum of the seeded tensor.
    pub fn spectrum(&self) -> &SingularSeed {
        &self.spectrum
    }

    /// An upper bound on the parameter error `‖Θ_* − Û R̂‖_F` of the executed component:
    /// an audit figure, not an output roundoff (see the module documentation).
    pub fn reproduction_bound(&self) -> f64 {
        self.reproduction_bound
    }
}

/// Seeds the Removed start of the registered matrix tensor `storage`. The result holds
/// the governor charge of the basis it keeps.
pub fn seed_removed<'a>(
    registry: &TensorRegistry,
    storage: TensorId,
    native: ArrayView2<'a, f64>,
) -> Result<Governed<RemovedSeed<'a>>, StartSeedError> {
    let (rows, cols) = native.dim();
    let scratch = reserve_start(start_scratch_bytes(rows, cols, true), "removed start scratch")?;
    let (factor, spectrum) = exact_factor(native)?;
    let reproduction_bound = reproduction_bound(native, factor.write(), factor.read());
    let kept = reserve_start(start_basis_bytes(rows, cols, cols, false), "removed start basis")?;
    let basis = FactoredEdit::new(factor.write().to_owned(), factor.read().t().to_owned())
        .map_err(|error| StartSeedError::Lift(LiftError::from(error)))?;
    // The basis holds its own copies; the factor and its scratch end here.
    drop((factor, scratch));
    let anchor = ResidualAnchor::new(
        registry,
        storage,
        native,
        basis,
        vec![cols],
        ComponentCoefficients::Basis,
    )
    .map_err(StartSeedError::Lift)?;
    Ok(kept.bind(RemovedSeed {
        anchor,
        spectrum,
        reproduction_bound,
    }))
}

/// The rank-revealing exact factor `U R = Θ_*` of one tensor, and its spectrum.
fn exact_factor(native: ArrayView2<'_, f64>) -> Result<(ExactFactor, SingularSeed), StartSeedError> {
    let (read, spectrum) = RankRevealingRead::new(&native.to_owned())
        .map_err(StartSeedError::Spectrum)?
        .into_parts();
    let candidate = Array2::<f64>::zeros((native.nrows(), read.nrows()));
    let factor = ExactFactor::solve_write(native, read.view(), candidate.view())
        .map_err(StartSeedError::Factor)?;
    Ok((factor, spectrum))
}

/// An upper bound on `‖Θ_* − U R‖_F` for the computed factors (see the module
/// documentation), or `+∞` where `γ` carries no information.
fn reproduction_bound(
    native: ArrayView2<'_, f64>,
    write: ArrayView2<'_, f64>,
    read: ArrayView2<'_, f64>,
) -> f64 {
    let (rows, components) = write.dim();
    let growth = accumulation_growth(components + 1);
    let entries = rows.saturating_mul(read.ncols());
    let depth = 2 * (components + 6) + entries + 1;
    let resolution = 1.0 - accumulation_growth(depth + 2);
    if !growth.is_finite() || !(resolution > 0.0) {
        return f64::INFINITY;
    }
    let difference = &native - &write.dot(&read);
    let absolute = native.mapv(f64::abs) + write.mapv(f64::abs).dot(&read.mapv(f64::abs));
    let sum_of_squares: f64 = difference
        .iter()
        .zip(absolute.iter())
        .map(|(entry, magnitude)| {
            let bound = entry.abs() + growth * magnitude / (1.0 - growth);
            bound * bound
        })
        .sum();
    sum_of_squares.sqrt() / resolution
}

/// The scratch ledger in bytes of seeding one `p × c` tensor on its anchor, or `None` on
/// overflow.
///
/// In `f64` entries, the peak of three phases:
/// * the owned teacher copy the decomposition reads (`pc`), the zero completion of a
///   wide tensor (`c²` when `p < c`) and the right singular vectors with their
///   descending copy (`2c²`);
/// * the read and the zero candidate (`c² + pc`), with the exact write's QR factors and
///   read copy (`3c²`) and its residual, transposed solve, product and write (`4pc`);
/// * with a reproduction bound, the factor (`pc + c²`), with the product, residual,
///   absolute teacher, absolute write and absolute product (`5pc`) and absolute read
///   (`c²`).
///
/// The basis the anchor keeps is charged apart once its rank is known.
fn start_scratch_bytes(rows: usize, cols: usize, reproduction: bool) -> Option<usize> {
    let pc = rows.checked_mul(cols)?;
    let cc = cols.checked_mul(cols)?;
    let completion = if rows < cols { cc } else { 0 };
    let decomposition = pc.checked_add(completion)?.checked_add(cc.checked_mul(2)?)?;
    let solve = cc.checked_mul(4)?.checked_add(pc.checked_mul(5)?)?;
    let bound = if reproduction {
        pc.checked_mul(6)?.checked_add(cc.checked_mul(2)?)?
    } else {
        0
    };
    decomposition
        .max(solve)
        .max(bound)
        .checked_mul(std::mem::size_of::<f64>())
}

/// The bytes of a kept basis of `terms` rank-one terms of a `p × c` tensor: the left and
/// right factors, and the second copy of the right factors a Kept factor seed exposes.
fn start_basis_bytes(rows: usize, cols: usize, terms: usize, keeps_right_factors: bool) -> Option<usize> {
    let right_copies = if keeps_right_factors { 2 } else { 1 };
    rows.checked_add(cols.checked_mul(right_copies)?)?
        .checked_mul(terms)?
        .checked_mul(std::mem::size_of::<f64>())
}

fn reserve_start(bytes: Option<usize>, context: &str) -> Result<MemoryReservation, StartSeedError> {
    let bytes = bytes.ok_or(StartSeedError::SizeOverflow)?;
    MemoryGovernor::global()
        .try_reserve(bytes, context)
        .map_err(StartSeedError::Memory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::apply::native_linear;
    use crate::parameter_decomposition::lift::{AnchorMask, TieOrientation};
    use crate::parameter_decomposition::rewrite::{ComponentMask, MlpMask};
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const WIDTH: usize = 4;
    const HIDDEN: usize = 6;
    const ROWS: usize = 5;

    fn uniform(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-1.0..1.0))
    }

    fn uniform_vector(rng: &mut StdRng, len: usize) -> Array1<f64> {
        Array1::from_shape_simple_fn(len, || rng.random_range(-1.0..1.0))
    }

    fn bitwise_equal(left: &Array2<f64>, right: &Array2<f64>) -> bool {
        left.dim() == right.dim()
            && left
                .iter()
                .zip(right.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// `A Bᵀ` for integer `A` (`6 × 2`) and `B` (`4 × 2`), each of full column
    /// rank. Every entry is a small integer, so the weight has rank exactly 2.
    fn planted_rank_two() -> Array2<f64> {
        let left = array![
            [1.0, 0.0],
            [2.0, 1.0],
            [0.0, 3.0],
            [-1.0, 2.0],
            [4.0, -1.0],
            [1.0, 1.0]
        ];
        let right = array![[2.0, 1.0], [0.0, -1.0], [1.0, 3.0], [-2.0, 1.0]];
        left.dot(&right.t())
    }

    #[test]
    fn a_planted_rank_leads_the_component_order_and_a_small_exact_direction_raises_it() {
        // `a` lies outside the column span of `A` and `b` outside that of `B`, so
        // `A Bᵀ + 2^-20 a bᵀ` has rank exactly 3. Its entries are integers plus
        // multiples of 2^-20, all exact in floating point.
        let outside_left = array![0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let outside_right = array![1.0, 1.0, 0.0, 0.0];
        let planted = planted_rank_two();
        let raised = &planted
            + &(outside_left
                .insert_axis(ndarray::Axis(1))
                .dot(&outside_right.insert_axis(ndarray::Axis(0)))
                * 2.0_f64.powi(-20));
        for (shape, weight, raised_weight) in [
            ("tall", planted.clone(), raised.clone()),
            ("wide", planted.t().to_owned(), raised.t().to_owned()),
        ] {
            let seed = RankRevealingRead::new(&weight).expect("a finite weight has a spectrum");
            let spectrum = seed.spectrum();
            assert_eq!(seed.read().dim(), (weight.ncols(), weight.ncols()), "{shape}: the read is square");
            assert_eq!(
                spectrum.singular_values().len(),
                weight.ncols(),
                "{shape}: one singular value per component"
            );
            assert!(
                spectrum.singular_values().windows(2).all(|pair| pair[0] >= pair[1]),
                "{shape}: the components must be in descending singular order, got {:?}",
                spectrum.singular_values()
            );
            assert_eq!(
                spectrum.rank(),
                2,
                "{shape}: an exact rank-2 weight must seed two resolved components, spectrum {:?} band {:e}",
                spectrum.singular_values(),
                spectrum.band()
            );

            // Positive control: the band resolves a 2^-20 exact direction.
            let raised_seed = RankRevealingRead::new(&raised_weight).expect("a finite weight has a spectrum");
            assert_eq!(
                raised_seed.spectrum().rank(),
                3,
                "{shape}: a 2^-20 exact direction must be resolved, spectrum {:?} band {:e}",
                raised_seed.spectrum().singular_values(),
                raised_seed.spectrum().band()
            );
        }
    }

    #[test]
    fn a_repeated_singular_value_is_one_cluster_and_a_resolved_gap_separates_it() {
        // Orthogonal columns of norms √8, √8, 1 and 0: a repeated singular value, a
        // simple one and a null direction, every entry exact.
        let repeated = array![[2.0, 2.0, 0.0, 0.0], [2.0, -2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let seed = RankRevealingRead::new(&repeated).expect("a finite weight has a spectrum");
        assert_eq!(seed.spectrum().rank(), 3, "spectrum {:?}", seed.spectrum().singular_values());
        assert_eq!(
            seed.spectrum().clusters().to_vec(),
            vec![0..2, 2..3, 3..4],
            "the repeated singular value must be one mask group: spectrum {:?} band {:e}",
            seed.spectrum().singular_values(),
            seed.spectrum().band()
        );

        // Positive control: scaling the second column by 1 + 2^-20 keeps every entry
        // exact and the columns orthogonal, and opens a gap of √8·2^-20.
        let mut separated = repeated.clone();
        separated
            .column_mut(1)
            .mapv_inplace(|entry| entry * (1.0 + 2.0_f64.powi(-20)));
        let separated_seed = RankRevealingRead::new(&separated).expect("a finite weight has a spectrum");
        assert_eq!(
            separated_seed.spectrum().clusters().to_vec(),
            vec![0..1, 1..2, 2..3, 3..4],
            "a gap of √8·2^-20 must separate the components: spectrum {:?} band {:e}",
            separated_seed.spectrum().singular_values(),
            separated_seed.spectrum().band()
        );
    }

    #[test]
    fn clusters_are_maximal_runs_of_unresolved_gaps() {
        // With b = 1e-3: gaps of 1.5e-3 chain three values into one run even though
        // its ends are 3e-3 apart, and a gap of 0.497 starts a new run.
        let band = 1.0e-3;
        assert_eq!(
            unresolved_gap_clusters(&[1.0, 0.9985, 0.997, 0.5], band),
            vec![0..3, 3..4],
            "unresolved gaps must chain into one maximal run"
        );
        // Positive control: a gap of 2.5e-3 exceeds 2b and separates.
        assert_eq!(
            unresolved_gap_clusters(&[1.0, 0.9975], band),
            vec![0..1, 1..2],
            "a gap above 2b must separate"
        );
        assert_eq!(unresolved_gap_clusters(&[], band), Vec::<Range<usize>>::new());
    }

    #[test]
    fn the_seeded_block_with_every_mask_on_executes_the_teacher_bit_for_bit() {
        let mut rng = StdRng::seed_from_u64(2951);
        let read_in = uniform(&mut rng, HIDDEN, WIDTH);
        let bias_in = uniform_vector(&mut rng, HIDDEN);
        let write_out = uniform(&mut rng, WIDTH, HIDDEN);
        let bias_out = uniform_vector(&mut rng, WIDTH);
        let inputs = uniform(&mut rng, ROWS, WIDTH) * 2.0;
        let teacher = NativeMlp::new(
            read_in.clone(),
            bias_in.clone(),
            write_out.clone(),
            bias_out.clone(),
            GaussianActivation::ExactGelu,
        )
        .expect("the teacher's shapes compose")
        .execute(inputs.view())
        .expect("the teacher executes");
        let seed = seed_mlp(read_in, bias_in, write_out, bias_out, GaussianActivation::ExactGelu)
            .expect("a random block is seeded");
        assert_eq!(seed.read_in().singular_values().len(), WIDTH);
        assert_eq!(seed.write_out().singular_values().len(), HIDDEN);
        assert_eq!(seed.read_in().rank(), WIDTH, "a random {HIDDEN}×{WIDTH} read-in has full rank");
        assert_eq!(seed.write_out().rank(), WIDTH, "a random {WIDTH}×{HIDDEN} write-out has rank {WIDTH}");
        assert_eq!(
            seed.write_out().clusters().last(),
            Some(&(WIDTH..HIDDEN)),
            "the write-out's null directions are one mask group, spectrum {:?}",
            seed.write_out().singular_values()
        );

        let all_on = seed
            .block()
            .execute(
                inputs.view(),
                MlpMask {
                    read_in: ComponentMask::AllOn,
                    write_out: ComponentMask::AllOn,
                },
            )
            .expect("the all-on seed executes");
        assert!(
            bitwise_equal(&all_on, &teacher),
            "the all-on seed must execute the teacher's tensors on their original path"
        );

        // Positive control: the factored path at the all-ones masks is the same block
        // algebraically but not the same bits.
        let read_in_ones = Array1::ones(WIDTH);
        let write_out_ones = Array1::ones(HIDDEN);
        let factored = seed
            .block()
            .execute(
                inputs.view(),
                MlpMask {
                    read_in: ComponentMask::Components(read_in_ones.view()),
                    write_out: ComponentMask::Components(write_out_ones.view()),
                },
            )
            .expect("the all-ones seed executes");
        assert!(
            !bitwise_equal(&factored, &teacher),
            "the bit-identity check must distinguish the factored all-ones path from the teacher"
        );
    }

    #[test]
    fn the_seed_ledger_counts_its_phases_and_the_governor_refuses_a_block_that_cannot_fit() {
        // Read-in 6×4 and write-out 4×6, in f64 entries:
        //   first decomposition: 2·16 = 32;
        //   second decomposition, first read held: 16 + (36 + 2·36) = 124;
        //   first solve, reads and candidates held: (16 + 36 + 24 + 24) + (3·16 + 4·24) = 244;
        //   second solve, first factor held: 100 + (24 + 16) + (3·36 + 4·24) = 344, the peak;
        //   factors kept: (24 + 16) + (24 + 36) = 100, so the scratch is 344 − 100 = 244.
        let ledger = seed_ledger((HIDDEN, WIDTH), (WIDTH, HIDDEN)).expect("small dimensions do not overflow");
        assert_eq!(
            ledger,
            SeedLedger {
                persistent_bytes: 100 * 8,
                scratch_bytes: 244 * 8,
            }
        );
        assert_eq!(
            seed_ledger((usize::MAX, 2), (2, usize::MAX)),
            None,
            "an overflowing ledger must be refused"
        );

        assert!(
            reserve_seed(ledger).is_ok(),
            "a {ledger:?} ledger must be admitted (positive control)"
        );
        // 2^55 bytes of factors: no host holds them.
        let huge = SeedLedger {
            persistent_bytes: 1 << 55,
            scratch_bytes: 0,
        };
        assert!(
            matches!(
                reserve_seed(huge),
                Err(SeedError::Memory(MemoryReservationError::BudgetExceeded { .. }))
            ),
            "a {huge:?} ledger was not refused by the memory governor"
        );
    }

    #[test]
    fn a_non_finite_or_empty_weight_is_refused_before_any_decomposition() {
        let mut rng = StdRng::seed_from_u64(2952);
        let read_in = uniform(&mut rng, HIDDEN, WIDTH);
        let bias_in = uniform_vector(&mut rng, HIDDEN);
        let mut write_out = uniform(&mut rng, WIDTH, HIDDEN);
        let bias_out = uniform_vector(&mut rng, WIDTH);
        write_out[[1, 2]] = f64::NAN;
        let refused = seed_mlp(
            read_in.clone(),
            bias_in.clone(),
            write_out.clone(),
            bias_out.clone(),
            GaussianActivation::Relu,
        );
        assert!(
            matches!(
                refused,
                Err(SeedError::Spectrum {
                    factor: MlpFactor::WriteOut,
                    refusal: SpectrumRefusal::NonFinite,
                })
            ),
            "a non-finite write-out must be refused, got {refused:?}"
        );
        let empty = RankRevealingRead::new(&Array2::<f64>::zeros((0, 3)));
        assert!(
            matches!(empty, Err(SpectrumRefusal::Empty { rows: 0, cols: 3 })),
            "an empty weight must be refused, got {empty:?}"
        );

        // Positive control: the same block with a finite entry is seeded.
        write_out[[1, 2]] = 0.25;
        let seeded = seed_mlp(read_in, bias_in, write_out, bias_out, GaussianActivation::Relu);
        assert!(seeded.is_ok(), "a finite block must be seeded, got {seeded:?}");
    }

    fn registry_with(id: &str, native: &Array2<f64>) -> TensorRegistry {
        let mut registry = TensorRegistry::default();
        registry
            .register_storage(TensorId(id.to_string()), native.view().into_dyn())
            .expect("a fresh name registers");
        registry
    }

    fn bits_of(values: &Array2<f64>) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn frobenius(values: &Array2<f64>) -> f64 {
        values.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    #[test]
    fn the_kept_literal_start_executes_the_residual_mask_on_the_native_product() {
        let mut rng = StdRng::seed_from_u64(2961);
        let native = uniform(&mut rng, HIDDEN, WIDTH);
        let inputs = uniform(&mut rng, ROWS, WIDTH);
        let registry = registry_with("mlp.fc_in.weight", &native);
        let anchor = seed_kept_literal(&registry, TensorId("mlp.fc_in.weight".to_string()), native.view())
            .expect("a registered tensor seeds its literal start");
        assert_eq!(anchor.component_count(), 0, "the literal start has no components");
        let teacher = native_linear(native.view(), inputs.view()).expect("a small product is admitted");
        let all_on = anchor
            .apply(&AnchorMask::all_on(0), inputs.view(), TieOrientation::Identity)
            .expect("the all-on literal start applies");
        assert_eq!(
            bits_of(&all_on),
            bits_of(&teacher),
            "the all-on literal start must execute the teacher bit for bit"
        );
        let half = AnchorMask {
            residual: 0.5,
            components: Vec::new(),
        };
        let scaled = anchor
            .apply(&half, inputs.view(), TieOrientation::Identity)
            .expect("a scaled literal start applies");
        assert_eq!(
            bits_of(&scaled),
            bits_of(&(&*teacher * 0.5)),
            "m_Delta = 0.5 must scale the native product once"
        );

        // Positive control: the scaled start is not the teacher.
        assert_ne!(bits_of(&scaled), bits_of(&teacher), "the bit check must see the residual mask");
    }

    #[test]
    fn the_kept_factor_start_keeps_the_resolved_terms_and_executes_the_teacher_all_on() {
        let native = planted_rank_two();
        let registry = registry_with("layer.weight", &native);
        let seed = seed_kept_factor(&registry, TensorId("layer.weight".to_string()), native.view())
            .expect("a planted rank-2 weight seeds its factor start");
        assert_eq!(seed.anchor().component_count(), 2, "an exact rank-2 weight keeps two terms");
        let right_factors = seed.right_factors();
        assert_eq!(right_factors.len(), 2, "one right factor per kept term");
        assert!(
            right_factors.iter().all(|factor| factor.dim() == (native.ncols(), 1)),
            "each right factor is c × 1"
        );
        let mut rng = StdRng::seed_from_u64(2962);
        for (orientation, width) in [
            (TieOrientation::Identity, native.ncols()),
            (TieOrientation::Transpose, native.nrows()),
        ] {
            let inputs = uniform(&mut rng, ROWS, width);
            let teacher = match orientation {
                TieOrientation::Identity => native_linear(native.view(), inputs.view()),
                TieOrientation::Transpose => native_linear(native.t(), inputs.view()),
            }
            .expect("a small product is admitted");
            let all_on = seed
                .anchor()
                .apply(&AnchorMask::all_on(2), inputs.view(), orientation)
                .expect("the all-on factor start applies");
            assert_eq!(
                bits_of(&all_on),
                bits_of(&teacher),
                "{orientation:?}: the all-on factor start must execute the teacher bit for bit"
            );

            // Positive control: switching the leading component off moves the bits.
            let mut one_off = AnchorMask::all_on(2);
            one_off.components[0] = 0.0;
            let moved = seed
                .anchor()
                .apply(&one_off, inputs.view(), orientation)
                .expect("a masked factor start applies");
            assert_ne!(
                bits_of(&moved),
                bits_of(&teacher),
                "{orientation:?}: the leading component must act"
            );
        }
    }

    #[test]
    fn the_removed_start_reports_a_reproduction_bound_that_resolves_a_small_write_move() {
        let mut rng = StdRng::seed_from_u64(2963);
        let native = uniform(&mut rng, HIDDEN, WIDTH);
        let registry = registry_with("layer.weight", &native);
        let seed = seed_removed(&registry, TensorId("layer.weight".to_string()), native.view())
            .expect("a random weight seeds its removed start");
        assert_eq!(seed.anchor().component_count(), 1, "the removed start is one component");
        assert_eq!(seed.spectrum().rank(), WIDTH, "a random {HIDDEN}×{WIDTH} weight has full rank");
        let bound = seed.reproduction_bound();
        let (factor, spectrum) = exact_factor(native.view()).expect("the same weight factors again");
        assert_eq!(&spectrum, seed.spectrum(), "the owners are deterministic on the same inputs");
        let residual = frobenius(&(&native - &factor.write().dot(&factor.read())));
        assert!(
            residual <= bound,
            "the computed residual {residual:e} must sit under the bound {bound:e}"
        );

        // Positive control: a write moved by 1e-9 per entry leaves the bound.
        let moved = &factor.write() + &(uniform(&mut rng, HIDDEN, WIDTH) * 1.0e-9);
        let moved_residual = frobenius(&(&native - &moved.dot(&factor.read())));
        assert!(
            moved_residual > bound,
            "the bound must resolve a 1e-9 move of the write: {moved_residual:e} within {bound:e}"
        );
    }

    #[test]
    fn a_start_is_refused_on_a_non_finite_tensor_or_a_different_teacher() {
        let planted = planted_rank_two();
        let registry = registry_with("layer.weight", &planted);
        let mut infinite = planted.clone();
        infinite[[0, 0]] = f64::INFINITY;
        let refused = seed_kept_factor(&registry, TensorId("layer.weight".to_string()), infinite.view());
        assert!(
            matches!(refused, Err(StartSeedError::Spectrum(SpectrumRefusal::NonFinite))),
            "a non-finite tensor has no factor start, got {refused:?}"
        );
        let mut different = planted.clone();
        different[[0, 0]] += 1.0;
        let mismatched = seed_removed(&registry, TensorId("layer.weight".to_string()), different.view());
        assert!(
            matches!(mismatched, Err(StartSeedError::Lift(LiftError::TeacherMismatch { .. }))),
            "values that are not the registered teacher must be refused, got {mismatched:?}"
        );

        // Positive control: the registered teacher seeds both starts.
        assert!(seed_kept_factor(&registry, TensorId("layer.weight".to_string()), planted.view()).is_ok());
        assert!(seed_removed(&registry, TensorId("layer.weight".to_string()), planted.view()).is_ok());
    }

    #[test]
    fn the_start_ledger_counts_its_phases_and_the_governor_refuses_a_start_that_cannot_fit() {
        // A 6×4 tensor with a reproduction bound, in f64 entries (pc = 24, c² = 16):
        // decomposition 24 + 2·16 = 56, solve 4·16 + 5·24 = 184, bound 6·24 + 2·16 = 176.
        assert_eq!(start_scratch_bytes(HIDDEN, WIDTH, true), Some(184 * 8));
        // A wide 4×6 tensor without one (pc = 24, c² = 36):
        // decomposition 24 + 36 + 2·36 = 132, solve 4·36 + 5·24 = 264.
        assert_eq!(start_scratch_bytes(WIDTH, HIDDEN, false), Some(264 * 8));
        // Two kept terms of a 6×4 factor seed: a left column of 6 and two right columns of 4 each.
        assert_eq!(start_basis_bytes(HIDDEN, WIDTH, 2, true), Some((6 + 2 * 4) * 2 * 8));
        assert_eq!(
            start_scratch_bytes(usize::MAX, 2, true),
            None,
            "an overflowing ledger must be refused"
        );
        assert!(
            matches!(reserve_start(None, "start ledger test"), Err(StartSeedError::SizeOverflow)),
            "an overflowing ledger must be refused before the governor"
        );
        assert!(
            reserve_start(start_scratch_bytes(HIDDEN, WIDTH, true), "start ledger test").is_ok(),
            "a small start must be admitted (positive control)"
        );
        // A 2^26×2^14 tensor needs about 2^45 bytes of solve state: no host holds them.
        let huge = reserve_start(start_scratch_bytes(1 << 26, 1 << 14, false), "start ledger test");
        assert!(
            matches!(huge, Err(StartSeedError::Memory(MemoryReservationError::BudgetExceeded { .. }))),
            "a start that cannot fit was not refused by the memory governor, got {huge:?}"
        );
    }
}
