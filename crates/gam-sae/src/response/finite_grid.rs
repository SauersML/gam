//! Exhaustive FANOVA of a response tabulated on a declared finite product grid (#2946 R7).
//!
//! # The law
//!
//! Each port `k` takes values in a declared finite level set of size `n_k` (the operands `a, b ∈ Z_p`
//! of modular addition, a token slot's vocabulary, a binary switch). The declared law is uniform on
//! the product `L_1 × … × L_p`. A uniform law on a product is the product of the uniform marginals,
//! so the ports are independent and [`super::interaction`] applies without change: the Hoeffding
//! terms are orthogonal, `V(S) = Σ_{∅≠T⊆S} ‖f_T‖²_M`, and [`super::interaction::total_interactions`],
//! `additive_blocks` and `cross_block_energy` read a [`FiniteGridResponse`] like any other
//! [`PortVariance`] producer. Nothing is sampled: `E[F | Z_S]` is the average over the other ports'
//! levels, taken over every cell, so each `V(S)` carries only the rounding band of its own
//! arithmetic.
//!
//! The output metric is declared through a factor `R` (`k × out`), with `M = RᵀR`: the analysis runs on
//! `G = R F`. For logits the natural factor is the centring projector [`centring_factor`], since a
//! softmax reads logits only up to a shift; the identity (`None`) keeps raw outputs.
//!
//! # The worst-case complement
//!
//! Energies are averages. The sup-norm question — how far is `F` from *every* response that is
//! additive across ports `i, j` — has an exact finite answer from below. For levels `x, a` of port `i`,
//! `y, b` of port `j` and the other ports held at a context `c`, the rectangle difference
//!
//! ```text
//!   Δ = F(x,y) − F(x,b) − F(a,y) + F(a,b)
//! ```
//!
//! vanishes on every `g(z_{−j}) + h(z_{−i})`. So `Δ_F = Δ_{F−g}` and `|Δ_F| ≤ 4 sup|F − g|`, which gives
//! `ε* = inf_g sup|F − g| ≥ max|Δ| / 4` over any additive-across-the-pair `g`, in the sup norm over cells
//! and output coordinates of `G`. The rectangle attaining `max|Δ|` is the witness. From above, the
//! pair-ANOVA residual `I = F − E[F|Z_{−j}] − E[F|Z_{−i}] + E[F|Z_{−ij}]` is what the additive projection
//! leaves, so `ε* ≤ max|I|`. `I` at a cell is the average of `Δ` over all anchors `(a, b)` in the same
//! context, so `max|I| ≤ max|Δ|` and `max|I|/4` is a valid, weaker lower bound as well. By linear
//! programming duality `ε*` is the supremum over signed dual certificates that annihilate additive
//! functions, and a rectangle is one of them. The report is therefore an
//! [`EvidenceStatus`] over [`Extremum::Supremum`]: the witness attains the lower side, and the bounds
//! meet (`Exact`) when they agree within their bands, as for `xy` on `{−1, 1}²` (`Δ = 4`, `ε* = 1`).
//!
//! When every rectangle vanishes within the uniform rounding band, the response is additive across
//! the pair on the whole grid, and `F(x,b) + F(a,y) − F(a,b)` (anchored at level 0 of both ports)
//! reconstructs it cell by cell: `Exact{Exhaustive}` with value 0 and the reconstruction attached.
//!
//! # Coordinates
//!
//! Additivity belongs to the port frame (see [`super::interaction`]). On a finite domain a change of
//! frame is a bijection of cells. [`FiniteGridResponse::reindex`] carries the response to a declared
//! new product grid under a caller's map, for example `(a, b) → (s, d) = (a + b, a − b) mod p`, which
//! is a bijection of `Z_p²` for odd `p` and not for even `p`. A bijection pushes the uniform law
//! forward to the uniform law, so the new ports are again independent, and the whole analysis runs in
//! the new coordinates. The map is a declaration: it is checked to be a bijection onto the new grid,
//! never searched for.
//!
//! # Concurvity: non-product domains are refused
//!
//! On a domain that is not a full product — a correlated input set such as `a ≤ b`, or a sample in
//! which some level pairs never occur — the additive terms are not identified. With `a` and `b`
//! dependent, a function of `a` can be partly written as a function of `b` on the support, so the
//! split of `F` into `f_a + f_b + f_ab` is not unique, the terms are not orthogonal, and the
//! retained variances no longer add up to term energies (the concurvity of additive models; Buja,
//! Hastie & Tibshirani 1989). A rectangle difference also needs all four corners on the domain. So
//! [`FiniteGridResponse::from_points`] refuses any point set that does not cover every cell of the
//! declared product exactly once, rather than reading an interaction share off an unidentified
//! decomposition.

use std::fmt;

use gam_linalg::roundoff::accumulation_growth;
use gam_runtime::resource::{MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{Array2, ArrayView2};

use super::interaction::PortVariance;
use super::subspace::BandedEnergy;
use crate::parameter_decomposition::supports::{
    EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum,
};

/// Why a finite-grid calculation refused.
#[derive(Clone, Debug, PartialEq)]
pub enum FiniteGridError {
    /// The grid declares no ports.
    NoPorts,
    /// A port declares an empty level set.
    EmptyLevelSet { port: usize },
    /// The number of cells or entries overflows `usize`.
    CellCountOverflow,
    /// The response has no output coordinates.
    NoOutputs,
    /// The response rows do not match the grid's cell count.
    RowCountMismatch { rows: usize, cells: usize },
    /// The output factor's columns do not match the response's outputs.
    OutputFactorMismatch {
        factor_columns: usize,
        outputs: usize,
    },
    /// A response or factor entry is NaN or infinite.
    NonFinite {
        what: &'static str,
        index: usize,
        value: f64,
    },
    /// The declared points are not every cell of the product exactly once (see the concurvity note).
    NonProductDomain { reason: String },
    /// A declared re-indexing is not a bijection onto the new grid.
    NotABijection { reason: String },
    /// A retained port set is not sorted, distinct and in range.
    InvalidRetained { retained: Vec<usize> },
    /// A port pair is not two distinct ports in range.
    InvalidPair { i: usize, j: usize },
    /// The rectangle lower bound exceeds the ANOVA-residual upper bound beyond both bands, so a band
    /// understated its error.
    InconsistentBounds { lower: f64, upper: f64 },
    /// An evidence constructor refused.
    Evidence(EvidenceStatusError),
    /// The governor refused a reservation.
    Memory(MemoryReservationError),
}

impl fmt::Display for FiniteGridError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPorts => write!(formatter, "finite grid: no ports declared"),
            Self::EmptyLevelSet { port } => {
                write!(formatter, "finite grid: port {port} declares no levels")
            }
            Self::CellCountOverflow => write!(formatter, "finite grid: the cell count overflows"),
            Self::NoOutputs => write!(formatter, "finite grid: the response has no outputs"),
            Self::RowCountMismatch { rows, cells } => write!(
                formatter,
                "finite grid: {rows} response rows for a grid of {cells} cells"
            ),
            Self::OutputFactorMismatch {
                factor_columns,
                outputs,
            } => write!(
                formatter,
                "finite grid: the output factor has {factor_columns} columns for {outputs} outputs"
            ),
            Self::NonFinite { what, index, value } => {
                write!(formatter, "finite grid: {what} entry {index} is {value}")
            }
            Self::NonProductDomain { reason } => write!(
                formatter,
                "finite grid: the domain is not the declared product ({reason}); additive terms are \
                 not identified on a non-product domain"
            ),
            Self::NotABijection { reason } => {
                write!(
                    formatter,
                    "finite grid: the declared re-indexing is not a bijection: {reason}"
                )
            }
            Self::InvalidRetained { retained } => write!(
                formatter,
                "finite grid: retained ports {retained:?} are not sorted, distinct and in range"
            ),
            Self::InvalidPair { i, j } => {
                write!(
                    formatter,
                    "finite grid: ({i}, {j}) is not a pair of distinct ports"
                )
            }
            Self::InconsistentBounds { lower, upper } => write!(
                formatter,
                "finite grid: rectangle lower bound {lower} exceeds ANOVA upper bound {upper}"
            ),
            Self::Evidence(error) => write!(formatter, "finite grid: {error}"),
            Self::Memory(error) => write!(formatter, "finite grid: {error}"),
        }
    }
}

impl std::error::Error for FiniteGridError {}

impl From<MemoryReservationError> for FiniteGridError {
    fn from(error: MemoryReservationError) -> Self {
        Self::Memory(error)
    }
}

impl From<EvidenceStatusError> for FiniteGridError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

/// The centring projector `I − 11ᵀ/n`: the output factor under which logits are compared up to a
/// shift, as a softmax reads them.
pub fn centring_factor(outputs: usize) -> Array2<f64> {
    let share = 1.0 / outputs as f64;
    Array2::from_shape_fn((outputs, outputs), |(row, column)| {
        if row == column { 1.0 - share } else { -share }
    })
}

fn checked_cells(levels: &[usize]) -> Result<usize, FiniteGridError> {
    if levels.is_empty() {
        return Err(FiniteGridError::NoPorts);
    }
    let mut cells = 1usize;
    for (port, &count) in levels.iter().enumerate() {
        if count == 0 {
            return Err(FiniteGridError::EmptyLevelSet { port });
        }
        cells = cells
            .checked_mul(count)
            .ok_or(FiniteGridError::CellCountOverflow)?;
    }
    Ok(cells)
}

/// Row-major strides, last port fastest.
fn strides_of(levels: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; levels.len()];
    for port in (0..levels.len().saturating_sub(1)).rev() {
        strides[port] = strides[port + 1] * levels[port + 1];
    }
    strides
}

fn reserve_f64(
    governor: &MemoryGovernor,
    count: usize,
    context: &str,
) -> Result<MemoryReservation, FiniteGridError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(FiniteGridError::CellCountOverflow)?;
    Ok(governor.try_reserve(bytes, context)?)
}

/// A response tabulated on every cell of a declared finite product grid, read under the uniform law.
#[derive(Debug)]
pub struct FiniteGridResponse {
    levels: Vec<usize>,
    strides: Vec<usize>,
    cells: usize,
    outputs: usize,
    /// `G = R F`, cell-major, `cells × outputs`.
    values: Vec<f64>,
    /// A bound on `|Ĝ − G|` per entry: the factor product's rounding.
    bands: Vec<f64>,
    governor: MemoryGovernor,
    reservation: MemoryReservation,
}

/// `E[G | Z_S]` on the retained ports' grid, with a band per entry.
#[derive(Clone, Debug)]
pub struct ConditionalMean {
    /// The level counts of the retained ports, in order; rows are row-major over them.
    pub levels: Vec<usize>,
    /// `Π levels × outputs`.
    pub values: Array2<f64>,
    pub bands: Array2<f64>,
}

/// One rectangle of a port pair: `Δ = G(x,y) − G(x,b) − G(a,y) + G(a,b)` at one output coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct Rectangle {
    /// The full cell `(x, y, context)`.
    pub corner: Vec<usize>,
    /// The opposite levels `(a, b)` of ports `i` and `j`.
    pub anchor: (usize, usize),
    /// The output coordinate of `G`.
    pub output: usize,
    /// `|Δ|` as computed.
    pub difference: f64,
    /// A bound on the error of `difference`.
    pub band: f64,
}

/// The claim's domain: every response on the grid that is additive across ports `i` and `j`.
#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveAcrossPair {
    pub levels: Vec<usize>,
    pub pair: (usize, usize),
}

/// The worst-case distance from a grid response to the responses additive across one port pair.
#[derive(Clone, Debug)]
pub struct RectangleComplement {
    pub pair: (usize, usize),
    /// The rectangle attaining `max|Δ|`, or `None` when a port of the pair has one level.
    pub witness: Option<Rectangle>,
    /// `max|Δ|` over every rectangle, context and output.
    pub max_difference: f64,
    /// A band that bounds the error of every rectangle's `Δ`.
    pub uniform_band: f64,
    /// `max|I|` of the pair-ANOVA residual, and `max(|I| + band)`, the certified upper bound on `ε*`.
    pub anova_residual_sup: f64,
    pub anova_upper: f64,
    /// `ε* = inf_{g additive across the pair} sup|G − g|`.
    pub evidence: EvidenceStatus<Rectangle, AdditiveAcrossPair>,
    /// `G(x,b) + G(a,y) − G(a,b)` anchored at level 0 of both ports, `cells × outputs`, present when every
    /// rectangle vanishes within the uniform band.
    pub reconstruction: Option<Array2<f64>>,
}

impl FiniteGridResponse {
    /// Tabulate `responses` (`cells × out`, rows row-major over `levels`, last port fastest) under the
    /// output factor `R` (`k × out`; `None` is the identity).
    pub fn new(
        levels: Vec<usize>,
        responses: ArrayView2<'_, f64>,
        output_factor: Option<ArrayView2<'_, f64>>,
        governor: &MemoryGovernor,
    ) -> Result<Self, FiniteGridError> {
        Self::build(levels, responses, None, output_factor, governor)
    }

    /// Tabulate responses given at declared cells `points[row]` (one level per port). Refused unless
    /// the points cover every cell of the product exactly once.
    pub fn from_points(
        levels: Vec<usize>,
        points: &[Vec<usize>],
        responses: ArrayView2<'_, f64>,
        output_factor: Option<ArrayView2<'_, f64>>,
        governor: &MemoryGovernor,
    ) -> Result<Self, FiniteGridError> {
        let cells = checked_cells(&levels)?;
        if points.len() != responses.nrows() {
            return Err(FiniteGridError::RowCountMismatch {
                rows: responses.nrows(),
                cells: points.len(),
            });
        }
        let strides = strides_of(&levels);
        let mut row_of_cell = vec![usize::MAX; cells];
        for (row, point) in points.iter().enumerate() {
            if point.len() != levels.len() || point.iter().zip(&levels).any(|(&l, &n)| l >= n) {
                return Err(FiniteGridError::NonProductDomain {
                    reason: format!("point {row} = {point:?} is not a cell of {levels:?}"),
                });
            }
            let cell: usize = point.iter().zip(&strides).map(|(l, s)| l * s).sum();
            if row_of_cell[cell] != usize::MAX {
                return Err(FiniteGridError::NonProductDomain {
                    reason: format!(
                        "cell {point:?} appears at rows {} and {row}, so the law is not uniform",
                        row_of_cell[cell]
                    ),
                });
            }
            row_of_cell[cell] = row;
        }
        let missing = row_of_cell.iter().filter(|&&row| row == usize::MAX).count();
        if missing > 0 {
            return Err(FiniteGridError::NonProductDomain {
                reason: format!("{missing} of {cells} cells of the product are absent"),
            });
        }
        Self::build(
            levels,
            responses,
            Some(&row_of_cell),
            output_factor,
            governor,
        )
    }

    fn build(
        levels: Vec<usize>,
        responses: ArrayView2<'_, f64>,
        row_of_cell: Option<&[usize]>,
        output_factor: Option<ArrayView2<'_, f64>>,
        governor: &MemoryGovernor,
    ) -> Result<Self, FiniteGridError> {
        let cells = checked_cells(&levels)?;
        if responses.nrows() != cells {
            return Err(FiniteGridError::RowCountMismatch {
                rows: responses.nrows(),
                cells,
            });
        }
        let raw_outputs = responses.ncols();
        if raw_outputs == 0 {
            return Err(FiniteGridError::NoOutputs);
        }
        for (index, &value) in responses.iter().enumerate() {
            if !value.is_finite() {
                return Err(FiniteGridError::NonFinite {
                    what: "response",
                    index,
                    value,
                });
            }
        }
        if let Some(factor) = output_factor {
            if factor.ncols() != raw_outputs {
                return Err(FiniteGridError::OutputFactorMismatch {
                    factor_columns: factor.ncols(),
                    outputs: raw_outputs,
                });
            }
            if factor.nrows() == 0 {
                return Err(FiniteGridError::NoOutputs);
            }
            for (index, &value) in factor.iter().enumerate() {
                if !value.is_finite() {
                    return Err(FiniteGridError::NonFinite {
                        what: "output factor",
                        index,
                        value,
                    });
                }
            }
        }
        let outputs = output_factor.map_or(raw_outputs, |factor| factor.nrows());
        let entries = cells
            .checked_mul(outputs)
            .ok_or(FiniteGridError::CellCountOverflow)?;
        let reservation = reserve_f64(
            governor,
            entries
                .checked_mul(2)
                .ok_or(FiniteGridError::CellCountOverflow)?,
            "finite grid response",
        )?;
        let mut values = vec![0.0; entries];
        let mut bands = vec![0.0; entries];
        let growth = accumulation_growth(raw_outputs);
        for cell in 0..cells {
            let row = responses.row(row_of_cell.map_or(cell, |rows| rows[cell]));
            let base = cell * outputs;
            match output_factor {
                None => {
                    for (slot, &value) in values[base..base + outputs].iter_mut().zip(row.iter()) {
                        *slot = value;
                    }
                }
                Some(factor) => {
                    for coordinate in 0..outputs {
                        let mut sum = 0.0;
                        let mut absolute = 0.0;
                        for (weight, &value) in factor.row(coordinate).iter().zip(row.iter()) {
                            sum += weight * value;
                            absolute += (weight * value).abs();
                        }
                        values[base + coordinate] = sum;
                        bands[base + coordinate] = growth * absolute;
                    }
                }
            }
        }
        Ok(Self {
            strides: strides_of(&levels),
            levels,
            cells,
            outputs,
            values,
            bands,
            governor: governor.clone(),
            reservation,
        })
    }

    /// The level counts of the ports.
    pub fn levels(&self) -> &[usize] {
        &self.levels
    }

    /// The output coordinates of `G = R F`.
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// The number of cells, the cardinality of the exhaustive family.
    pub fn cells(&self) -> usize {
        self.cells
    }

    /// The bytes this response holds against its governor.
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }

    /// `G` at `cell` and output `coordinate`, with its band.
    pub fn value(&self, cell: &[usize], coordinate: usize) -> Option<(f64, f64)> {
        if cell.len() != self.levels.len()
            || cell.iter().zip(&self.levels).any(|(&l, &n)| l >= n)
            || coordinate >= self.outputs
        {
            return None;
        }
        let index = self.cell_index(cell) * self.outputs + coordinate;
        Some((self.values[index], self.bands[index]))
    }

    fn cell_index(&self, cell: &[usize]) -> usize {
        cell.iter().zip(&self.strides).map(|(l, s)| l * s).sum()
    }

    fn decode(&self, mut cell: usize) -> Vec<usize> {
        self.strides
            .iter()
            .map(|&stride| {
                let level = cell / stride;
                cell %= stride;
                level
            })
            .collect()
    }

    fn check_retained(&self, retained: &[usize]) -> Result<(), FiniteGridError> {
        let sorted = retained.windows(2).all(|pair| pair[0] < pair[1]);
        if !sorted || retained.iter().any(|&port| port >= self.levels.len()) {
            return Err(FiniteGridError::InvalidRetained {
                retained: retained.to_vec(),
            });
        }
        Ok(())
    }

    /// `E[G | Z_S]` by exhaustive averaging over the other ports' levels.
    pub fn conditional_mean(&self, retained: &[usize]) -> Result<ConditionalMean, FiniteGridError> {
        self.check_retained(retained)?;
        let kept_levels: Vec<usize> = retained.iter().map(|&port| self.levels[port]).collect();
        let kept_strides = strides_of(&kept_levels);
        let groups: usize = kept_levels.iter().product();
        let per_group = self.cells / groups;
        let slots = groups * self.outputs;
        let scratch = reserve_f64(
            &self.governor,
            slots
                .checked_mul(3)
                .ok_or(FiniteGridError::CellCountOverflow)?,
            "finite grid conditional mean",
        )?;
        let mut sums = Array2::<f64>::zeros((groups, self.outputs));
        let mut absolutes = Array2::<f64>::zeros((groups, self.outputs));
        let mut band_sums = Array2::<f64>::zeros((groups, self.outputs));
        for cell in 0..self.cells {
            let group: usize = retained
                .iter()
                .zip(&kept_strides)
                .map(|(&port, &stride)| (cell / self.strides[port]) % self.levels[port] * stride)
                .sum();
            let base = cell * self.outputs;
            for coordinate in 0..self.outputs {
                let value = self.values[base + coordinate];
                sums[[group, coordinate]] += value;
                absolutes[[group, coordinate]] += value.abs();
                band_sums[[group, coordinate]] += self.bands[base + coordinate];
            }
        }
        let count = per_group as f64;
        let growth = accumulation_growth(per_group);
        let values = sums.mapv(|sum| sum / count);
        let bands = ndarray::Zip::from(&absolutes)
            .and(&band_sums)
            .map_collect(|&absolute, &band| growth * absolute / count + band / count);
        drop(scratch);
        Ok(ConditionalMean {
            levels: kept_levels,
            values,
            bands,
        })
    }

    /// Carry the response to the product grid `levels` under the declared cell map `bijection`, which
    /// must be a bijection from this grid's cells onto the new grid's cells.
    pub fn reindex<M>(
        &self,
        levels: Vec<usize>,
        bijection: M,
        governor: &MemoryGovernor,
    ) -> Result<Self, FiniteGridError>
    where
        M: Fn(&[usize]) -> Vec<usize>,
    {
        let cells = checked_cells(&levels)?;
        if cells != self.cells {
            return Err(FiniteGridError::NotABijection {
                reason: format!(
                    "the new grid {levels:?} has {cells} cells, the old one {}",
                    self.cells
                ),
            });
        }
        let strides = strides_of(&levels);
        let entries = cells * self.outputs;
        let reservation = reserve_f64(
            governor,
            entries
                .checked_mul(2)
                .ok_or(FiniteGridError::CellCountOverflow)?,
            "finite grid re-indexing",
        )?;
        let mut values = vec![0.0; entries];
        let mut bands = vec![0.0; entries];
        let mut source_of: Vec<Option<usize>> = vec![None; cells];
        for cell in 0..self.cells {
            let old = self.decode(cell);
            let image = bijection(&old);
            if image.len() != levels.len() || image.iter().zip(&levels).any(|(&l, &n)| l >= n) {
                return Err(FiniteGridError::NotABijection {
                    reason: format!("{old:?} maps to {image:?}, not a cell of {levels:?}"),
                });
            }
            let target: usize = image.iter().zip(&strides).map(|(l, s)| l * s).sum();
            if let Some(previous) = source_of[target] {
                return Err(FiniteGridError::NotABijection {
                    reason: format!(
                        "{:?} and {old:?} both map to {image:?}",
                        self.decode(previous)
                    ),
                });
            }
            source_of[target] = Some(cell);
            let (from, to) = (cell * self.outputs, target * self.outputs);
            values[to..to + self.outputs].copy_from_slice(&self.values[from..from + self.outputs]);
            bands[to..to + self.outputs].copy_from_slice(&self.bands[from..from + self.outputs]);
        }
        Ok(Self {
            levels,
            strides,
            cells,
            outputs: self.outputs,
            values,
            bands,
            governor: governor.clone(),
            reservation,
        })
    }

    /// The worst-case complement of additivity across ports `i` and `j`: every rectangle, context and
    /// output coordinate is evaluated.
    pub fn rectangle_complement(
        &self,
        i: usize,
        j: usize,
    ) -> Result<RectangleComplement, FiniteGridError> {
        let ports = self.levels.len();
        if i == j || i >= ports || j >= ports {
            return Err(FiniteGridError::InvalidPair { i, j });
        }
        let (rows, columns) = (self.levels[i], self.levels[j]);
        let (row_stride, column_stride) = (self.strides[i], self.strides[j]);
        let outputs = self.outputs;
        let entry = |base: usize, x: usize, y: usize, coordinate: usize| {
            (base + x * row_stride + y * column_stride) * outputs + coordinate
        };
        let triangle = accumulation_growth(3);
        let row_growth = accumulation_growth(columns);
        let column_growth = accumulation_growth(rows);
        let grand_growth = accumulation_growth(rows * columns);

        let mut max_entry_band = 0.0f64;
        let mut max_magnitude = 0.0f64;
        let mut best: Option<Rectangle> = None;
        let mut max_difference = 0.0f64;
        let mut anova_residual_sup = 0.0f64;
        let mut anova_upper = 0.0f64;
        let mut anova_band_at_sup = 0.0f64;
        let mut row_mean = vec![(0.0, 0.0); rows];
        let mut column_mean = vec![(0.0, 0.0); columns];
        let mut spread = vec![0.0; columns];

        for base in 0..self.cells {
            let levels_here = (base / row_stride) % rows + (base / column_stride) % columns;
            if levels_here != 0 {
                continue;
            }
            for coordinate in 0..outputs {
                let at = |x: usize, y: usize| {
                    let index = entry(base, x, y, coordinate);
                    (self.values[index], self.bands[index])
                };
                // Pair-ANOVA residual within this context.
                let (mut grand_sum, mut grand_absolute, mut grand_band) = (0.0, 0.0, 0.0);
                for (x, slot) in row_mean.iter_mut().enumerate() {
                    let (mut sum, mut absolute, mut band) = (0.0, 0.0, 0.0);
                    for y in 0..columns {
                        let (value, entry_band) = at(x, y);
                        sum += value;
                        absolute += value.abs();
                        band += entry_band;
                        max_entry_band = max_entry_band.max(entry_band);
                        max_magnitude = max_magnitude.max(value.abs());
                    }
                    grand_sum += sum;
                    grand_absolute += absolute;
                    grand_band += band;
                    let count = columns as f64;
                    *slot = (sum / count, row_growth * absolute / count + band / count);
                }
                for (y, slot) in column_mean.iter_mut().enumerate() {
                    let (mut sum, mut absolute, mut band) = (0.0, 0.0, 0.0);
                    for x in 0..rows {
                        let (value, entry_band) = at(x, y);
                        sum += value;
                        absolute += value.abs();
                        band += entry_band;
                    }
                    let count = rows as f64;
                    *slot = (sum / count, column_growth * absolute / count + band / count);
                }
                let count = (rows * columns) as f64;
                let grand = grand_sum / count;
                let grand_error = grand_growth * grand_absolute / count + grand_band / count;
                for x in 0..rows {
                    for y in 0..columns {
                        let (value, entry_band) = at(x, y);
                        let (row, row_band) = row_mean[x];
                        let (column, column_band) = column_mean[y];
                        let residual = value - row - column + grand;
                        let band = entry_band
                            + row_band
                            + column_band
                            + grand_error
                            + triangle * (value.abs() + row.abs() + column.abs() + grand.abs());
                        if residual.abs() > anova_residual_sup {
                            anova_residual_sup = residual.abs();
                            anova_band_at_sup = band;
                        }
                        anova_upper = anova_upper.max(residual.abs() + band);
                    }
                }
                // Rectangles: for each pair of rows, the spread of D(y) = G(x,y) − G(a,y).
                for x in 0..rows {
                    for a in (x + 1)..rows {
                        for (y, slot) in spread.iter_mut().enumerate() {
                            *slot = at(x, y).0 - at(a, y).0;
                        }
                        let (mut high, mut low) = (0usize, 0usize);
                        for y in 1..columns {
                            if spread[y] > spread[high] {
                                high = y;
                            }
                            if spread[y] < spread[low] {
                                low = y;
                            }
                        }
                        let difference = spread[high] - spread[low];
                        if difference > max_difference {
                            max_difference = difference;
                            let corners = [at(x, high), at(a, high), at(x, low), at(a, low)];
                            let band = corners.iter().map(|corner| corner.1).sum::<f64>()
                                + triangle
                                    * corners.iter().map(|corner| corner.0.abs()).sum::<f64>();
                            let mut corner = self.decode(base);
                            corner[i] = x;
                            corner[j] = high;
                            best = Some(Rectangle {
                                corner,
                                anchor: (a, low),
                                output: coordinate,
                                difference,
                                band,
                            });
                        }
                    }
                }
            }
        }

        let uniform_band = 4.0 * max_entry_band + triangle * 4.0 * max_magnitude;
        let domain = AdditiveAcrossPair {
            levels: self.levels.clone(),
            pair: (i, j),
        };
        let basis = ExactBasis::Exhaustive {
            cardinality: (self.cells as u64).saturating_mul(outputs as u64),
        };
        if max_difference <= uniform_band {
            let reservation = reserve_f64(
                &self.governor,
                self.cells * outputs,
                "finite grid additive reconstruction",
            )?;
            let mut reconstruction = Array2::<f64>::zeros((self.cells, outputs));
            for cell in 0..self.cells {
                let x = (cell / row_stride) % rows;
                let y = (cell / column_stride) % columns;
                let base = cell - x * row_stride - y * column_stride;
                for coordinate in 0..outputs {
                    reconstruction[[cell, coordinate]] = self.values[entry(base, x, 0, coordinate)]
                        + self.values[entry(base, 0, y, coordinate)]
                        - self.values[entry(base, 0, 0, coordinate)];
                }
            }
            drop(reservation);
            let evidence =
                EvidenceStatus::exact(0.0, max_difference + uniform_band, basis, None, domain)?;
            return Ok(RectangleComplement {
                pair: (i, j),
                witness: best,
                max_difference,
                uniform_band,
                anova_residual_sup,
                anova_upper,
                evidence,
                reconstruction: Some(reconstruction),
            });
        }
        let witness = best.expect("a positive rectangle difference has a witness");
        let quarter = witness.difference / 4.0;
        let quarter_band = witness.band / 4.0;
        let lower = (quarter - quarter_band).next_down();
        if lower > anova_upper {
            return Err(FiniteGridError::InconsistentBounds {
                lower,
                upper: anova_upper,
            });
        }
        let evidence = if anova_residual_sup - quarter <= quarter_band + anova_band_at_sup {
            let numerical_error = (anova_upper - quarter).max(quarter_band);
            EvidenceStatus::exact(
                quarter,
                numerical_error,
                basis,
                Some(witness.clone()),
                domain,
            )?
        } else {
            EvidenceStatus::unresolved(
                lower,
                anova_upper,
                Extremum::Supremum,
                Some(witness.clone()),
                domain,
            )?
        };
        Ok(RectangleComplement {
            pair: (i, j),
            witness: Some(witness),
            max_difference,
            uniform_band,
            anova_residual_sup,
            anova_upper,
            evidence,
            reconstruction: None,
        })
    }
}

impl PortVariance for FiniteGridResponse {
    type Error = FiniteGridError;

    fn port_count(&self) -> usize {
        self.levels.len()
    }

    /// `V(S) = (1/|L_S|) Σ_s ‖E[G | Z_S = s] − E G‖²`, exact up to the band of its own averaging.
    fn retained_variance(&self, retained: &[usize]) -> Result<BandedEnergy, FiniteGridError> {
        self.check_retained(retained)?;
        if retained.is_empty() {
            return Ok(BandedEnergy::ZERO);
        }
        let conditional = self.conditional_mean(retained)?;
        let mean = self.conditional_mean(&[])?;
        let difference_growth = accumulation_growth(1);
        let groups = conditional.values.nrows();
        let mut squares = 0.0;
        let mut error = 0.0;
        for group in 0..groups {
            for coordinate in 0..self.outputs {
                let value = conditional.values[[group, coordinate]];
                let centre = mean.values[[0, coordinate]];
                let deviation = value - centre;
                let band = conditional.bands[[group, coordinate]]
                    + mean.bands[[0, coordinate]]
                    + difference_growth * (value.abs() + centre.abs());
                squares += deviation * deviation;
                error += 2.0 * deviation.abs() * band + band * band;
            }
        }
        let count = groups as f64;
        let value = squares / count;
        Ok(BandedEnergy {
            value,
            band: error / count + accumulation_growth(groups * self.outputs + 1) * value,
        })
    }
}

#[cfg(test)]
#[path = "finite_grid_tests.rs"]
mod finite_grid_tests;
