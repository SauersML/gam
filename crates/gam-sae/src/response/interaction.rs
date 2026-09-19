//! Total interactions and finest additive blocks of a known block's response under a declared
//! product intervention law (#2946 R7).
//!
//! # Law and decomposition
//!
//! The experiment is declared, not observed. A known block `F` receives its input through `p`
//! intervention ports `Z_1, …, Z_p`, and the experiment makes them mutually independent. Under
//! #2946's Gaussian law `Z ~ N(0, I_d)`, any grouping of coordinates into ports qualifies, and
//! orthonormal port frames reduce to coordinate groups after the rotation `W → WR` (#2946 A2).
//! Under a product law, `F` has the Hoeffding decomposition `F = Σ_{T ⊆ [p]} f_T(Z_T)` (Hoeffding
//! 1948; Efron & Stein 1981), whose terms are orthogonal in the output metric:
//! `E[f_Tᵀ M f_T′] = 0` for `T ≠ T′`. So the retained variance of a port set `S`,
//!
//! ```text
//!   V(S) = E‖E[F | Z_S] − E F‖²_M = Σ_{∅ ≠ T ⊆ S} ‖f_T‖²_M,
//! ```
//!
//! is the energy of the terms that live inside `S`. For a known MLP block it is #2946 R4's `V(P)`
//! at the coordinate projector `P_S`. [`PortVariance`] is the contract any producer of `V(S)`
//! meets.
//!
//! # Total interactions (R7)
//!
//! Inclusion–exclusion over four retained variances gives
//!
//! ```text
//!   I_ij = V([p]) − V([p]∖i) − V([p]∖j) + V([p]∖{i,j}) = Σ_{T ⊇ {i,j}} ‖f_T‖²_M,
//! ```
//!
//! because `V([p]) − V([p]∖i) = Σ_{T ∋ i} ‖f_T‖²_M` (Sobol's total effect, unnormalized), and
//! subtracting the same quantity with `j` removed first leaves the terms that contain both.
//! This is the known superset importance of the pair (Liu & Owen 2006, "Estimating mean
//! dimensionality of analysis of variance decompositions"), also called the total interaction
//! index (Fruth, Roustant & Kuhnt 2014). Hooker (2004, "Discovering additive structure in black
//! box functions") read additive structure off the graph of these indices, and Muehlenstaedt,
//! Roustant, Carraro & Kuhnt (2012) used the same FANOVA graph for kriging. None of it is new
//! here. What this module adds is only that `V` is the block's exact analytic response under the
//! declared law, so every index carries a derived rounding band and no estimator noise.
//!
//! - **Finest additive blocks.** A partition splits `F = Σ_B g_B(Z_B)` iff every cross pair has
//!   `I_ij = 0`. If the cross pairs vanish, every term with `f_T ≠ 0` has all its pairs inside
//!   one block, so `T` lies inside one block. Conversely an additive split kills every crossing
//!   term, hence every cross `I_ij`. So the connected components of `{I_ij > 0}` are the finest
//!   additive blocks under that product law ([`TotalInteractions::additive_blocks`]).
//!   Components, not cliques: `z₂z₃ + z₃z₄` has `I₂₄ = 0` and still one block.
//! - **Cross-block energy.** For a partition, the energy an additive-across-blocks response
//!   discards is `E_cross = Σ_{T crossing} ‖f_T‖²_M = V([p]) − Σ_B V(B)`
//!   ([`TotalInteractions::cross_block_energy`]). Each crossing term contains at least one cross
//!   pair, so `Σ_{cross pairs} I_ij = Σ_T c(T)·‖f_T‖²_M ≥ E_cross`, where `c(T)` counts the
//!   cross pairs inside `T` ([`TotalInteractions::cross_block_bound`]). Equality holds when no
//!   term holds two cross pairs, in particular for every purely pairwise response. The bound
//!   prices every partition from one pairwise screen. It replaces a fixed fission threshold.
//!
//! # Resolution
//!
//! `I_ij > 0` is resolved when the computed value clears its band. The band is the four retained
//! variances' bands plus `γ₃·Σ|V|`, the rounding of three additions of pre-formed terms (Higham,
//! *ASNA* §3.1), to first order in `u`. A pair inside the band is not resolved from zero, so the
//! components are the finest blocks this arithmetic separates. There is no hand threshold. A
//! computed index below minus its band is refused. Under a product law it is a sum of energies,
//! so a producer that yields it has broken the contract (dependent ports, or a band that
//! understates its error).
//!
//! # What a total interaction does not see
//!
//! `I_ij` measures the non-additivity of the response in the declared port frame and nothing
//! more.
//! - A linear map that mixes every port into every output, `F(z) = Rz`, is additive: all
//!   `I_ij = 0` and each port is its own block, although no output reads a single port. A block
//!   certifies that the response splits as a sum over ports. It says nothing about which outputs a
//!   port reaches.
//! - Blocks belong to the port frame. `y₀² − y₁²` is additive in `y`, but in coordinates `z`
//!   with `y = Rz` it carries a `z₀z₁` term and one block.
//!
//! # Contrast with the empirical carve
//!
//! `gam_terms::structure::anova_atom` centers each factor basis against the empirical code sample
//! and sums each block's squared values over paired code rows. When the codes are dependent,
//! marginal centering does not make the terms orthogonal. `E[f₁(θ₁)f₂(θ₂)]` is the covariance of
//! two centered functions of dependent codes, and `E[f₁₂ f₂] ≠ 0` because `E[φ̃¹(θ₁) | θ₂] ≠ 0`.
//! So the blocks' row energies do not add up to the surface energy, and the interaction fraction
//! is not a share of variance. Nothing like the cross-block bound follows. Here the contract is a
//! declared product law, which makes the terms orthogonal by independence and every index a sum
//! of term energies. The carve reads observed codes. This module reads the block's response to a
//! declared randomized intervention.

use std::collections::BTreeMap;
use std::fmt;

use ndarray::Array2;

use super::subspace::{BandedEnergy, signed_sum};

/// Why a total-interaction screen refused.
#[derive(Clone, Debug, PartialEq)]
pub enum InteractionError {
    /// The law declares no ports.
    NoPorts,
    /// The producer of `V(S)` failed.
    Producer { retained: Vec<usize>, message: String },
    /// A producer returned a non-finite value, a non-finite or negative band, or a variance below
    /// minus its band.
    InvalidRetainedVariance {
        retained: Vec<usize>,
        value: f64,
        band: f64,
    },
    /// `V([p]) − V([p]∖i)` sits below minus its band. It is a sum of term energies under a product
    /// law, so the producer broke its contract.
    NegativeTotalEffect { port: usize, value: f64, band: f64 },
    /// `I_ij` sits below minus its band (same contract).
    NegativeInteraction {
        i: usize,
        j: usize,
        value: f64,
        band: f64,
    },
    /// `V([p]) − Σ_B V(B)` sits below minus its band (same contract).
    NegativeCrossBlockEnergy { value: f64, band: f64 },
    /// The screen and the law disagree on the port count.
    PortCountMismatch { screened: usize, law: usize },
    /// A partition does not place every port in exactly one non-empty block.
    InvalidPartition { reason: String },
}

impl fmt::Display for InteractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPorts => write!(formatter, "total interactions: the law declares no ports"),
            Self::Producer { retained, message } => write!(
                formatter,
                "total interactions: the retained variance of ports {retained:?} failed: {message}"
            ),
            Self::InvalidRetainedVariance {
                retained,
                value,
                band,
            } => write!(
                formatter,
                "total interactions: the retained variance of ports {retained:?} is {value} with \
                 band {band}; a variance is finite, non-negative within a finite non-negative band"
            ),
            Self::NegativeTotalEffect { port, value, band } => write!(
                formatter,
                "total interactions: the total effect of port {port} is {value}, below minus its \
                 band {band}; under a product law it is a sum of term energies"
            ),
            Self::NegativeInteraction { i, j, value, band } => write!(
                formatter,
                "total interactions: I_{i},{j} is {value}, below minus its band {band}; under a \
                 product law it is a sum of term energies"
            ),
            Self::NegativeCrossBlockEnergy { value, band } => write!(
                formatter,
                "total interactions: the cross-block energy is {value}, below minus its band \
                 {band}; under a product law it is a sum of term energies"
            ),
            Self::PortCountMismatch { screened, law } => write!(
                formatter,
                "total interactions: the screen covers {screened} ports but the law declares {law}"
            ),
            Self::InvalidPartition { reason } => {
                write!(formatter, "total interactions: invalid partition: {reason}")
            }
        }
    }
}

impl std::error::Error for InteractionError {}

/// A producer of retained variances over independent intervention ports.
///
/// `retained_variance(S)` returns `V(S) = E‖E[F | Z_S] − E F‖²_M` with a bound on its absolute
/// error, for a port set `S` given sorted, distinct and in range. The ports must be independent
/// under the declared law. That is the whole contract the screen relies on, and a producer that
/// breaks it surfaces as a negative index beyond its band.
pub trait PortVariance {
    type Error: fmt::Display;

    /// The number of independent ports `p`.
    fn port_count(&self) -> usize;

    /// `V(S)` for the retained ports `S`.
    fn retained_variance(&self, retained: &[usize]) -> Result<BandedEnergy, Self::Error>;
}

fn checked_retained_variance<L: PortVariance>(
    law: &L,
    retained: &[usize],
) -> Result<BandedEnergy, InteractionError> {
    if retained.is_empty() {
        return Ok(BandedEnergy::ZERO);
    }
    let energy =
        law.retained_variance(retained)
            .map_err(|error| InteractionError::Producer {
                retained: retained.to_vec(),
                message: error.to_string(),
            })?;
    let valid = energy.value.is_finite()
        && energy.band.is_finite()
        && energy.band >= 0.0
        && energy.value >= -energy.band;
    if !valid {
        return Err(InteractionError::InvalidRetainedVariance {
            retained: retained.to_vec(),
            value: energy.value,
            band: energy.band,
        });
    }
    Ok(energy)
}

fn ports_without(port_count: usize, removed: &[usize]) -> Vec<usize> {
    (0..port_count)
        .filter(|port| !removed.contains(port))
        .collect()
}

/// Index of the pair `i < j` in the row-major upper triangle over `port_count` ports.
fn pair_index(port_count: usize, i: usize, j: usize) -> usize {
    i * (2 * port_count - i - 1) / 2 + (j - i - 1)
}

fn union_root(parent: &mut [usize], mut port: usize) -> usize {
    while parent[port] != port {
        parent[port] = parent[parent[port]];
        port = parent[port];
    }
    port
}

/// The block label of every port, or why `partition` is not a partition of `0..port_count`.
fn partition_labels(
    port_count: usize,
    partition: &[Vec<usize>],
) -> Result<Vec<usize>, InteractionError> {
    let mut labels: Vec<Option<usize>> = vec![None; port_count];
    for (block_index, block) in partition.iter().enumerate() {
        if block.is_empty() {
            return Err(InteractionError::InvalidPartition {
                reason: format!("block {block_index} is empty"),
            });
        }
        for &port in block {
            match labels.get(port).copied() {
                None => {
                    return Err(InteractionError::InvalidPartition {
                        reason: format!(
                            "port {port} in block {block_index} is outside 0..{port_count}"
                        ),
                    });
                }
                Some(Some(previous)) => {
                    return Err(InteractionError::InvalidPartition {
                        reason: format!("port {port} is in blocks {previous} and {block_index}"),
                    });
                }
                Some(None) => labels[port] = Some(block_index),
            }
        }
    }
    labels
        .into_iter()
        .enumerate()
        .map(|(port, label)| {
            label.ok_or_else(|| InteractionError::InvalidPartition {
                reason: format!("port {port} is in no block"),
            })
        })
        .collect()
}

/// The pairwise total-interaction screen of one response over its ports.
#[derive(Clone, Debug)]
pub struct TotalInteractions {
    port_count: usize,
    total_variance: BandedEnergy,
    total_effects: Vec<BandedEnergy>,
    pairs: Vec<BandedEnergy>,
}

/// Screen every pair of ports: `V` at `[p]`, at each `[p]∖i` and at each `[p]∖{i,j}`, which is
/// `1 + p + p(p−1)/2` retained variances.
pub fn total_interactions<L: PortVariance>(law: &L) -> Result<TotalInteractions, InteractionError> {
    let port_count = law.port_count();
    if port_count == 0 {
        return Err(InteractionError::NoPorts);
    }
    let all: Vec<usize> = (0..port_count).collect();
    let total_variance = checked_retained_variance(law, &all)?;
    let mut without_one = Vec::with_capacity(port_count);
    for port in 0..port_count {
        without_one.push(checked_retained_variance(
            law,
            &ports_without(port_count, &[port]),
        )?);
    }
    let mut total_effects = Vec::with_capacity(port_count);
    for (port, without) in without_one.iter().enumerate() {
        let effect = signed_sum(&[total_variance], &[*without]);
        if effect.value < -effect.band {
            return Err(InteractionError::NegativeTotalEffect {
                port,
                value: effect.value,
                band: effect.band,
            });
        }
        total_effects.push(effect);
    }
    let mut pairs = Vec::with_capacity(port_count * (port_count - 1) / 2);
    for i in 0..port_count {
        for j in (i + 1)..port_count {
            let without_pair = checked_retained_variance(law, &ports_without(port_count, &[i, j]))?;
            let interaction = signed_sum(
                &[total_variance, without_pair],
                &[without_one[i], without_one[j]],
            );
            if interaction.value < -interaction.band {
                return Err(InteractionError::NegativeInteraction {
                    i,
                    j,
                    value: interaction.value,
                    band: interaction.band,
                });
            }
            pairs.push(interaction);
        }
    }
    Ok(TotalInteractions {
        port_count,
        total_variance,
        total_effects,
        pairs,
    })
}

impl TotalInteractions {
    /// The number of ports screened.
    pub fn port_count(&self) -> usize {
        self.port_count
    }

    /// `V([p])`, the response's total variance.
    pub fn total_variance(&self) -> BandedEnergy {
        self.total_variance
    }

    /// `V([p]) − V([p]∖i) = Σ_{T ∋ i} ‖f_T‖²_M`, or `None` when `port` is out of range.
    pub fn total_effect(&self, port: usize) -> Option<BandedEnergy> {
        self.total_effects.get(port).copied()
    }

    /// `I_ij = Σ_{T ⊇ {i,j}} ‖f_T‖²_M`, symmetric, or `None` when a port is out of range. The
    /// superset importance of `{i}` is its total effect, so `I_ii` is [`Self::total_effect`].
    pub fn interaction(&self, i: usize, j: usize) -> Option<BandedEnergy> {
        if i >= self.port_count || j >= self.port_count {
            return None;
        }
        if i == j {
            return self.total_effect(i);
        }
        let (low, high) = (i.min(j), i.max(j));
        self.pairs
            .get(pair_index(self.port_count, low, high))
            .copied()
    }

    /// Every `I_ij` as `(values, bands)`, each `p × p` and symmetric, with the total effects on the
    /// diagonal.
    pub fn matrix(&self) -> (Array2<f64>, Array2<f64>) {
        let port_count = self.port_count;
        let mut values = Array2::<f64>::zeros((port_count, port_count));
        let mut bands = Array2::<f64>::zeros((port_count, port_count));
        for i in 0..port_count {
            values[[i, i]] = self.total_effects[i].value;
            bands[[i, i]] = self.total_effects[i].band;
            for j in (i + 1)..port_count {
                let interaction = self.pairs[pair_index(port_count, i, j)];
                values[[i, j]] = interaction.value;
                values[[j, i]] = interaction.value;
                bands[[i, j]] = interaction.band;
                bands[[j, i]] = interaction.band;
            }
        }
        (values, bands)
    }

    /// The finest additive blocks: the connected components of the graph whose edges are the
    /// resolved-positive `I_ij`. Each block is sorted, and blocks are ordered by their smallest
    /// port.
    pub fn additive_blocks(&self) -> Vec<Vec<usize>> {
        let port_count = self.port_count;
        let mut parent: Vec<usize> = (0..port_count).collect();
        for i in 0..port_count {
            for j in (i + 1)..port_count {
                if self.pairs[pair_index(port_count, i, j)].resolved_positive() {
                    let root_i = union_root(&mut parent, i);
                    let root_j = union_root(&mut parent, j);
                    if root_i != root_j {
                        parent[root_i.max(root_j)] = root_i.min(root_j);
                    }
                }
            }
        }
        let mut blocks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for port in 0..port_count {
            let root = union_root(&mut parent, port);
            blocks.entry(root).or_default().push(port);
        }
        blocks.into_values().collect()
    }

    /// `Σ_{cross pairs} I_ij`, which bounds the cross-block energy of `partition` from above, with
    /// equality when no Hoeffding term holds two cross pairs.
    pub fn cross_block_bound(
        &self,
        partition: &[Vec<usize>],
    ) -> Result<BandedEnergy, InteractionError> {
        let labels = partition_labels(self.port_count, partition)?;
        let mut crossing = Vec::new();
        for i in 0..self.port_count {
            for j in (i + 1)..self.port_count {
                if labels[i] != labels[j] {
                    crossing.push(self.pairs[pair_index(self.port_count, i, j)]);
                }
            }
        }
        Ok(signed_sum(&crossing, &[]))
    }

    /// The exact cross-block energy `V([p]) − Σ_B V(B)`: what the best response that is additive
    /// across the blocks of `partition` leaves out.
    pub fn cross_block_energy<L: PortVariance>(
        &self,
        law: &L,
        partition: &[Vec<usize>],
    ) -> Result<BandedEnergy, InteractionError> {
        if law.port_count() != self.port_count {
            return Err(InteractionError::PortCountMismatch {
                screened: self.port_count,
                law: law.port_count(),
            });
        }
        partition_labels(self.port_count, partition)?;
        let mut within = Vec::with_capacity(partition.len());
        for block in partition {
            let mut retained = block.clone();
            retained.sort_unstable();
            within.push(checked_retained_variance(law, &retained)?);
        }
        let energy = signed_sum(&[self.total_variance], &within);
        if energy.value < -energy.band {
            return Err(InteractionError::NegativeCrossBlockEnergy {
                value: energy.value,
                band: energy.band,
            });
        }
        Ok(energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::accumulation_growth;
    use ndarray::array;

    type Polynomial = BTreeMap<Vec<u32>, Vec<f64>>;

    fn gaussian_moment(order: u32) -> f64 {
        if order % 2 == 1 {
            return 0.0;
        }
        let mut moment = 1.0;
        let mut factor = 1u32;
        while factor < order {
            moment *= f64::from(factor);
            factor += 2;
        }
        moment
    }

    fn rademacher_moment(order: u32) -> f64 {
        if order % 2 == 1 { 0.0 } else { 1.0 }
    }

    /// `U = −2` with probability 1/5 and `1/2` with probability 4/5: centered, unit variance,
    /// skewed.
    fn skewed_moment(order: u32) -> f64 {
        let exponent = order as i32;
        ((-2.0f64).powi(exponent) + 4.0 * 0.5f64.powi(exponent)) / 5.0
    }

    fn factorial(n: u32) -> f64 {
        (1..=n).map(f64::from).product()
    }

    /// A polynomial response under a product law given by each port's moment sequence. `V(S)`
    /// follows by moment algebra, independently of the production screen, with a band from its
    /// absolute shadow.
    struct PolynomialProductLaw {
        moments: Vec<fn(u32) -> f64>,
        terms: Polynomial,
        metric: Array2<f64>,
    }

    impl PolynomialProductLaw {
        fn expectation(&self, exponents: &[u32], retained: &[usize]) -> (f64, f64) {
            let mut moment = 1.0;
            let mut absolute = 1.0;
            for &port in retained {
                let value = (self.moments[port])(exponents[port]);
                moment *= value;
                absolute *= value.abs();
            }
            (moment, absolute)
        }
    }

    impl PortVariance for PolynomialProductLaw {
        type Error = String;

        fn port_count(&self) -> usize {
            self.moments.len()
        }

        fn retained_variance(&self, retained: &[usize]) -> Result<BandedEnergy, String> {
            let ports = self.moments.len();
            let outputs = self.metric.nrows();
            let mut conditional: BTreeMap<Vec<u32>, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
            let mut highest = 0u32;
            for (exponents, coefficients) in &self.terms {
                if exponents.len() != ports || coefficients.len() != outputs {
                    return Err(format!(
                        "term {exponents:?} does not match {ports} ports and {outputs} outputs"
                    ));
                }
                let mut key = exponents.clone();
                let mut factor = 1.0;
                let mut absolute_factor = 1.0;
                for port in 0..ports {
                    highest = highest.max(exponents[port]);
                    if !retained.contains(&port) {
                        let moment = (self.moments[port])(exponents[port]);
                        factor *= moment;
                        absolute_factor *= moment.abs();
                        key[port] = 0;
                    }
                }
                let slot = conditional
                    .entry(key)
                    .or_insert_with(|| (vec![0.0; outputs], vec![0.0; outputs]));
                for output in 0..outputs {
                    slot.0[output] += coefficients[output] * factor;
                    slot.1[output] += coefficients[output].abs() * absolute_factor;
                }
            }
            let mut mean = vec![0.0; outputs];
            let mut absolute_mean = vec![0.0; outputs];
            for (exponents, (coefficients, absolute_coefficients)) in &conditional {
                let (moment, absolute_moment) = self.expectation(exponents, retained);
                for output in 0..outputs {
                    mean[output] += coefficients[output] * moment;
                    absolute_mean[output] += absolute_coefficients[output] * absolute_moment;
                }
            }
            let mut value = 0.0;
            let mut absolute = 0.0;
            for (left, (left_coefficients, left_absolute)) in &conditional {
                for (right, (right_coefficients, right_absolute)) in &conditional {
                    let summed: Vec<u32> = left.iter().zip(right).map(|(a, b)| a + b).collect();
                    let (moment, absolute_moment) = self.expectation(&summed, retained);
                    for a in 0..outputs {
                        for b in 0..outputs {
                            value += left_coefficients[a]
                                * self.metric[[a, b]]
                                * right_coefficients[b]
                                * moment;
                            absolute += left_absolute[a]
                                * self.metric[[a, b]].abs()
                                * right_absolute[b]
                                * absolute_moment;
                        }
                    }
                }
            }
            for a in 0..outputs {
                for b in 0..outputs {
                    value -= mean[a] * self.metric[[a, b]] * mean[b];
                    absolute += absolute_mean[a] * self.metric[[a, b]].abs() * absolute_mean[b];
                }
            }
            let operations = 2 * ports
                + 4 * highest as usize
                + 8
                + self.terms.len()
                + (conditional.len() * conditional.len() + 1) * outputs * outputs;
            Ok(BandedEnergy {
                value,
                band: accumulation_growth(operations) * absolute,
            })
        }
    }

    /// `(b + wᵀz)³ = Σ_{a+|e|=3} 3!/(a!·e!) · bᵃ · wᵉ · zᵉ`, times `output`, added to `terms`.
    fn add_cubic_unit(terms: &mut Polynomial, bias: f64, reader: &[f64], output: &[f64]) {
        let dims = reader.len();
        for code in 0..4usize.pow(dims as u32) {
            let mut exponents = vec![0u32; dims];
            let mut rest = code;
            for exponent in exponents.iter_mut() {
                *exponent = (rest % 4) as u32;
                rest /= 4;
            }
            let degree: u32 = exponents.iter().sum();
            if degree > 3 {
                continue;
            }
            let mut scale = 6.0 / factorial(3 - degree) * bias.powi((3 - degree) as i32);
            for (coordinate, &exponent) in exponents.iter().enumerate() {
                scale *= reader[coordinate].powi(exponent as i32) / factorial(exponent);
            }
            if scale == 0.0 {
                continue;
            }
            let slot = terms
                .entry(exponents)
                .or_insert_with(|| vec![0.0; output.len()]);
            for (entry, &weight) in slot.iter_mut().zip(output) {
                *entry += weight * scale;
            }
        }
    }

    /// `F(z) = Σ_j u_j (b_j + w_jᵀz)³` with integer data, so the oracle's arithmetic is exact.
    struct CubicBlock {
        biases: Vec<f64>,
        readers: Array2<f64>,
        outputs: Array2<f64>,
    }

    impl CubicBlock {
        fn polynomial(&self) -> Polynomial {
            let mut terms = Polynomial::new();
            for (unit, &bias) in self.biases.iter().enumerate() {
                let reader = self.readers.row(unit).to_vec();
                let output = self.outputs.column(unit).to_vec();
                add_cubic_unit(&mut terms, bias, &reader, &output);
            }
            terms
        }

        fn gaussian_law(&self, metric: Array2<f64>) -> PolynomialProductLaw {
            PolynomialProductLaw {
                moments: vec![gaussian_moment as fn(u32) -> f64; self.readers.ncols()],
                terms: self.polynomial(),
                metric,
            }
        }
    }

    /// Blocks `{0,1}`, `{2,3,4}` (a chain: no unit reads both 2 and 4) and `{5}`.
    fn planted_block() -> CubicBlock {
        CubicBlock {
            biases: vec![1.0, 0.0, -1.0, 1.0],
            readers: array![
                [1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            ],
            outputs: array![[1.0, -1.0, 1.0, 1.0], [2.0, 1.0, 0.0, 1.0]],
        }
    }

    fn planted_metric() -> Array2<f64> {
        array![[2.0, 1.0], [1.0, 1.0]]
    }

    fn pair_only_energy(law: &PolynomialProductLaw, i: usize, j: usize) -> BandedEnergy {
        let variance = |retained: &[usize]| {
            law.retained_variance(retained)
                .expect("the oracle evaluates every port set")
        };
        signed_sum(
            &[variance(&[i, j][..])],
            &[variance(&[i][..]), variance(&[j][..])],
        )
    }

    /// Every set partition of `0..ports`, by restricted growth strings.
    fn set_partitions(ports: usize) -> Vec<Vec<Vec<usize>>> {
        let mut partitions = Vec::new();
        let mut labels = vec![0usize; ports];
        loop {
            let block_count = labels.iter().copied().max().map_or(0, |max| max + 1);
            let mut partition = vec![Vec::new(); block_count];
            for (port, &label) in labels.iter().enumerate() {
                partition[label].push(port);
            }
            partitions.push(partition);
            let mut advanced = false;
            let mut position = ports;
            while position > 1 {
                position -= 1;
                let prefix_max = labels[..position].iter().copied().max().unwrap_or(0);
                if labels[position] <= prefix_max {
                    labels[position] += 1;
                    for later in labels.iter_mut().skip(position + 1) {
                        *later = 0;
                    }
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                return partitions;
            }
        }
    }

    /// #2946 A5: for `F = U₁U₂U₃` with independent, centered, unit-variance ports, every total
    /// interaction is 1 while every pair-only ANOVA term is 0. A screen of pairwise terms alone
    /// would call this response additive. The ports are Gaussian, Rademacher and skewed, so the
    /// law is a product law and nothing else.
    #[test]
    fn a_three_way_product_has_unit_total_interactions_and_zero_pair_terms() {
        let ports = vec![
            gaussian_moment as fn(u32) -> f64,
            rademacher_moment,
            skewed_moment,
        ];
        let product = PolynomialProductLaw {
            moments: ports.clone(),
            terms: Polynomial::from([(vec![1, 1, 1], vec![1.0])]),
            metric: array![[1.0]],
        };
        let screen = total_interactions(&product).expect("the product law screens");
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let interaction = screen.interaction(i, j).expect("ports in range");
            assert!(
                (interaction.value - 1.0).abs() <= interaction.band,
                "I_{i}{j} = {interaction:?}; expected 1"
            );
            assert!(interaction.resolved_positive());
            let pair_term = pair_only_energy(&product, i, j);
            assert!(
                pair_term.value.abs() <= pair_term.band,
                "pair-only term {{{i},{j}}} = {pair_term:?}; expected 0"
            );
        }
        assert_eq!(screen.additive_blocks(), vec![vec![0, 1, 2]]);

        // Control: `U₁U₂ + U₃` has a resolved pair-only term, one resolved total interaction and
        // two blocks, so neither assertion above holds by construction.
        let pair_plus_main = PolynomialProductLaw {
            moments: ports,
            terms: Polynomial::from([(vec![1, 1, 0], vec![1.0]), (vec![0, 0, 1], vec![1.0])]),
            metric: array![[1.0]],
        };
        let control = total_interactions(&pair_plus_main).expect("the control screens");
        let coupled = control.interaction(0, 1).expect("ports in range");
        assert!((coupled.value - 1.0).abs() <= coupled.band);
        assert!(pair_only_energy(&pair_plus_main, 0, 1).resolved_positive());
        for (i, j) in [(0, 2), (1, 2)] {
            let interaction = control.interaction(i, j).expect("ports in range");
            assert!(
                interaction.value.abs() <= interaction.band,
                "I_{i}{j} = {interaction:?}; expected 0"
            );
        }
        assert_eq!(control.additive_blocks(), vec![vec![0, 1], vec![2]]);
    }

    /// #2946 A5: planted additive blocks of a cubic MLP block under `N(0, I₆)` come back as the
    /// connected components, a chain included. Coupling two blocks through one unit merges them.
    #[test]
    fn planted_additive_blocks_are_recovered_as_connected_components() {
        let block = planted_block();
        let screen = total_interactions(&block.gaussian_law(planted_metric()))
            .expect("the planted block screens");
        assert_eq!(
            screen.additive_blocks(),
            vec![vec![0, 1], vec![2, 3, 4], vec![5]]
        );
        for (i, j) in [(0, 1), (2, 3), (3, 4)] {
            let interaction = screen.interaction(i, j).expect("ports in range");
            assert!(
                interaction.resolved_positive(),
                "I_{i}{j} = {interaction:?}"
            );
        }
        let chain_ends = screen.interaction(2, 4).expect("ports in range");
        assert!(
            chain_ends.value.abs() <= chain_ends.band,
            "no unit reads ports 2 and 4 together: I_24 = {chain_ends:?}"
        );

        let coupled = CubicBlock {
            biases: vec![1.0, 0.0, -1.0, 1.0, 0.0],
            readers: array![
                [1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ],
            outputs: array![[1.0, -1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 0.0, 1.0, 1.0]],
        };
        let merged = total_interactions(&coupled.gaussian_law(planted_metric()))
            .expect("the coupled block screens");
        assert_eq!(merged.additive_blocks(), vec![vec![0, 1, 2, 3, 4], vec![5]]);
    }

    /// #2946 A5: the cross-block energy never exceeds the sum of cross-pair total interactions,
    /// on every partition of four ports. On a purely pairwise response the two are equal. A
    /// three-way term spanning several cross pairs leaves slack, so the bound is not an identity.
    #[test]
    fn cross_block_bound_holds_on_every_partition_and_is_attained_on_a_pairwise_block() {
        let gaussian = vec![gaussian_moment as fn(u32) -> f64; 4];
        let pairwise = PolynomialProductLaw {
            moments: gaussian.clone(),
            terms: Polynomial::from([
                (vec![1, 1, 0, 0], vec![1.0]),
                (vec![0, 1, 1, 0], vec![2.0]),
                (vec![0, 0, 1, 1], vec![1.0]),
                (vec![1, 0, 0, 0], vec![-1.0]),
            ]),
            metric: array![[1.0]],
        };
        let three_way = PolynomialProductLaw {
            moments: gaussian,
            terms: Polynomial::from([(vec![1, 1, 1, 0], vec![1.0]), (vec![0, 0, 1, 1], vec![1.0])]),
            metric: array![[1.0]],
        };
        let partitions = set_partitions(4);
        assert_eq!(partitions.len(), 15, "the Bell number B₄");
        let pairwise_screen = total_interactions(&pairwise).expect("the pairwise law screens");
        let three_way_screen = total_interactions(&three_way).expect("the three-way law screens");
        let mut slack_seen = false;
        for partition in &partitions {
            let energy = pairwise_screen
                .cross_block_energy(&pairwise, partition)
                .expect("a valid partition");
            let bound = pairwise_screen
                .cross_block_bound(partition)
                .expect("a valid partition");
            assert!(
                (energy.value - bound.value).abs() <= energy.band + bound.band,
                "pairwise {partition:?}: E_cross {energy:?} vs bound {bound:?}"
            );
            let energy = three_way_screen
                .cross_block_energy(&three_way, partition)
                .expect("a valid partition");
            let bound = three_way_screen
                .cross_block_bound(partition)
                .expect("a valid partition");
            assert!(
                energy.value <= bound.value + energy.band + bound.band,
                "three-way {partition:?}: E_cross {energy:?} exceeds bound {bound:?}"
            );
            slack_seen |= bound.value - energy.value > energy.band + bound.band;
        }
        assert!(slack_seen, "the three-way term leaves slack on some partition");
        let singletons: Vec<Vec<usize>> = (0..4).map(|port| vec![port]).collect();
        let energy = three_way_screen
            .cross_block_energy(&three_way, &singletons)
            .expect("a valid partition");
        let bound = three_way_screen
            .cross_block_bound(&singletons)
            .expect("a valid partition");
        assert!(
            (energy.value - 2.0).abs() <= energy.band && (bound.value - 4.0).abs() <= bound.band,
            "singletons: E_cross {energy:?} (expected 2), bound {bound:?} (expected 4)"
        );
        assert!(matches!(
            three_way_screen.cross_block_bound(&[vec![0, 1], vec![1, 2, 3]]),
            Err(InteractionError::InvalidPartition { .. })
        ));
    }

    /// #2946 A5, the limitation. A rotation that mixes both ports into both outputs is additive,
    /// so it carries no total interaction and splits into singleton blocks, although each output
    /// reads each port. Blocks also belong to the port frame: `y₀² − y₁²` is additive in `y` and
    /// one block in `z`, with `y = Rz`.
    #[test]
    fn a_linear_rotation_shows_no_cross_input_interaction() {
        let gaussian = vec![gaussian_moment as fn(u32) -> f64; 2];
        let rotation_terms =
            Polynomial::from([(vec![1, 0], vec![0.6, 0.8]), (vec![0, 1], vec![-0.8, 0.6])]);
        let rotation = PolynomialProductLaw {
            moments: gaussian.clone(),
            terms: rotation_terms.clone(),
            metric: Array2::eye(2),
        };
        let screen = total_interactions(&rotation).expect("the rotation screens");
        let mixing = screen.interaction(0, 1).expect("ports in range");
        assert!(
            mixing.value.abs() <= mixing.band,
            "a linear map is additive: I_01 = {mixing:?}"
        );
        assert_eq!(screen.additive_blocks(), vec![vec![0], vec![1]]);
        for output in 0..2 {
            let mut metric = Array2::<f64>::zeros((2, 2));
            metric[[output, output]] = 1.0;
            let single_output = PolynomialProductLaw {
                moments: gaussian.clone(),
                terms: rotation_terms.clone(),
                metric,
            };
            let per_output = total_interactions(&single_output).expect("one output screens");
            for port in 0..2 {
                let effect = per_output.total_effect(port).expect("port in range");
                assert!(
                    effect.resolved_positive(),
                    "output {output} reads port {port}: {effect:?}"
                );
            }
        }

        let additive_in_y = PolynomialProductLaw {
            moments: gaussian.clone(),
            terms: Polynomial::from([(vec![2, 0], vec![1.0]), (vec![0, 2], vec![-1.0])]),
            metric: array![[1.0]],
        };
        let same_function_in_z = PolynomialProductLaw {
            moments: gaussian,
            terms: Polynomial::from([
                (vec![2, 0], vec![-0.28]),
                (vec![1, 1], vec![-1.92]),
                (vec![0, 2], vec![0.28]),
            ]),
            metric: array![[1.0]],
        };
        let in_y = total_interactions(&additive_in_y).expect("the y frame screens");
        let in_z = total_interactions(&same_function_in_z).expect("the z frame screens");
        assert!(!in_y.interaction(0, 1).expect("ports in range").resolved_positive());
        assert!(in_z.interaction(0, 1).expect("ports in range").resolved_positive());
        assert_eq!(in_z.additive_blocks(), vec![vec![0, 1]]);
    }

    /// A law whose retained variances make `I_01 < 0` breaks the product contract. The screen
    /// refuses it instead of reading the pair as additive. The same table with consistent values
    /// screens to two blocks, then to one.
    #[test]
    fn a_negative_total_interaction_is_refused_not_read_as_additive() {
        struct TableLaw {
            table: BTreeMap<Vec<usize>, f64>,
        }
        impl PortVariance for TableLaw {
            type Error = String;
            fn port_count(&self) -> usize {
                2
            }
            fn retained_variance(&self, retained: &[usize]) -> Result<BandedEnergy, String> {
                self.table
                    .get(retained)
                    .map(|&value| BandedEnergy { value, band: 0.0 })
                    .ok_or_else(|| format!("no entry for {retained:?}"))
            }
        }
        let law = |joint: f64| TableLaw {
            table: BTreeMap::from([(vec![0, 1], joint), (vec![0], 1.0), (vec![1], 1.0)]),
        };
        assert!(matches!(
            total_interactions(&law(1.0)),
            Err(InteractionError::NegativeInteraction { i: 0, j: 1, .. })
        ));
        let additive = total_interactions(&law(2.0)).expect("a consistent table screens");
        assert_eq!(additive.additive_blocks(), vec![vec![0], vec![1]]);
        let bound = total_interactions(&law(3.0)).expect("a consistent table screens");
        assert_eq!(bound.additive_blocks(), vec![vec![0, 1]]);
    }
}
