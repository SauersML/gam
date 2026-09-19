//! Minimum-code supports over the singular components of the induction benchmark's heads (#2951 S2b).
//!
//! `mpd_induction_components --export REGISTRY_EXPORT --settings SETTINGS_JSON --out REPORT_JSON`
//!
//! `SETTINGS_JSON` declares `{"circuits": [[layer, head, "qk" | "ov"], …], "baseline_fractions": [f, …],
//! "absolute_tolerances": [ε, …], "fraction_bits": p, "haar_draws": M, "haar_seed": s}`. Every one is an
//! experiment input with no default (#2951 SPEC tension).
//!
//! # Components
//! A head's query-key form `B_QK = W_Q^hᵀ W_K^h` and value-output map `B_OV = W_O^h W_V^h` are invariant under
//! its `GL(d_head)` gauge. `seed::RankRevealingRead` reads each through its right singular vectors `V̂ᵀ`. A
//! layer's key read stacks its heads' `V̂_QK,hᵀ` (`n_heads · d_model` rows, full column rank) with the block
//! candidate write `W_K^h V̂_QK,h`, and its value read stacks `V̂_OV,hᵀ` with `W_V^h V̂_OV,h`, so
//! `rewrite::ExactFactor` leaves only roundoff off the blocks. Under a key mask the score of head `h` is
//! `σ x_tᵀ W_Q^hᵀ W_K^h V̂ diag(m) V̂ᵀ x_s = σ Σ_k m_k σ_k (û_kᵀ x_t)(v̂_kᵀ x_s)`: masking key component `k`
//! removes exactly the `k`-th singular term of the head's query-key form. A value mask does the same for the
//! value-output map. The query and output reads stay native.
//! * **Groups (P1).** `SingularSeed::clusters` are the maximal runs of singular values within `2b` of each other.
//!   Each is one mask group, since only `cI` on a run is invariant under the run's rotations.
//! * **Tail.** A declared circuit's components that `SingularSeed::rank` does not resolve from zero are always off in
//!   the artifact and are not coded; a cluster that straddles the rank joins the tail. The report gives each tail's
//!   size and largest singular value, and the measured logit gap between the teacher and the resolved-only network,
//!   so the tail's effect is inside every measured distortion.
//! * Components of circuits that are not declared stay on.
//!
//! # The artifact and its code
//! Each kept group sends its components' read rows and write columns in `precision`'s lattice code, and fidelity
//! is measured on the decoded artifact. A support's message is the enumerative subset codeword over the groups,
//! then `H` once in the prefix integer code, then every kept group's codeword padded to `H`, the longest:
//! `L(C, k) + L_int(H) + k·H`, a function of the support's size alone.
//!
//! # Evidence
//! `supports::BoxSeparationOracle` separates every support by refining boxes of group masks. The program runs a box
//! at its center, each free group's key or value columns at `1/2 ± 1/2` (`block::ComponentMasks`), so the logit radii
//! cover every mask of the box, and `bounds::kl_over_logit_boxes` bounds every induction row over them. Free groups
//! split in descending order of their singular mass. The tolerances are declared fractions of the network's own
//! do-nothing divergence (every declared group off), `ε_f = f · b̂`, and declared absolute tolerances; a tolerance
//! whose certified support is empty is labelled vacuous.
//!
//! # Nulls at equal code
//! At the certified support's per-circuit counts:
//! * **bottom:** each circuit's same number of groups of least singular mass, in the same decomposition, so the
//!   same code;
//! * **Haar:** a Haar-random orthonormal basis `O V̂_r` of each circuit's resolved row space (`O` the right singular
//!   vectors of a declared-seed Gaussian matrix) keeping the same number of components: the same count of reals at
//!   the same precision, so the same code. It depends only on `B`, so it is gauge-invariant. Each of the `M` draws is
//!   one exact mask; their mean is a statistical estimate, never a bound.
//!
//! Positive control: every null at a circuit's whole resolved rank reads the whole resolved space, as the
//! resolved-only network does.
//!
//! # The planted-duplicate OR control
//! A positive control of the whole route (enclosure, refinement and shrink), on the exact component network. A
//! declared circuit's component 0 is planted twice, `a` in its own row and `b` in the first tail row, each with half
//! the write, so the network depends on the pair only through `m_a + m_b`. Masks are linear in the read, so the pair
//! is exchangeable, not an OR by construction; an OR appears at a tolerance between the proven bounds of the
//! single-copy vertices (every group on, `a` off, `b` off) and of the both-off vertex. There the oracle must certify
//! `U∖{a}` and `U∖{b}` and refute `U∖{a, b}` with a witness perturbing exactly `{a, b}`. The declared circuits are
//! tried in order up to the first whose vertices separate; if none does, the control is reported missing, not passed.
//! A failed control fails the run.

use gam_sae::parameter_decomposition::attention::AttentionGeometry;
use gam_sae::parameter_decomposition::block::{AttentionLayerReads, ComponentAttentionLayer, ComponentMasks, ProjectionRead};
use gam_sae::parameter_decomposition::codec::{CodecError, prefix_integer_len_bits, subset_code_len_bits};
use gam_sae::parameter_decomposition::precision::{DecodableArtifact, DeclaredPrecision, LatticeCode};
use gam_sae::parameter_decomposition::rewrite::ComponentRead;
use gam_sae::parameter_decomposition::seed::RankRevealingRead;
use gam_sae::parameter_decomposition::supports::{
    BoxDivergence, BoxEnclosure, BoxSeparationOracle, CardinalityCode, ComponentSet, EvidenceStatus, ExactBasis,
    Extremum, FailureHypergraph, MaskBox, MaskSide, SeparationOracle, minimum_code_support,
};
use ndarray::{Array2, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;

#[path = "support/npy_header.rs"]
mod npy_header;
#[path = "support/induction_export.rs"]
mod induction_export;
#[path = "support/induction_network.rs"]
mod induction_network;
use induction_export::{LoadedExport, float64_array, load_export};
use induction_network::{
    InductionRows, Largest, Logged, Network, PROJECTIONS, RowDivergence, Rows, factored_layer, flag, layout,
    stored_layer,
};

const USAGE: &str = "usage: mpd_induction_components --export REGISTRY_EXPORT --settings SETTINGS_JSON --out REPORT_JSON";

#[derive(Deserialize)]
struct Settings {
    circuits: Vec<(usize, usize, String)>,
    baseline_fractions: Vec<f64>,
    absolute_tolerances: Vec<f64>,
    fraction_bits: i32,
    haar_draws: usize,
    haar_seed: u64,
}

/// A head's query-key form or value-output map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Circuit {
    Qk,
    Ov,
}

impl Circuit {
    fn name(self) -> &'static str {
        match self {
            Self::Qk => "qk",
            Self::Ov => "ov",
        }
    }
}

/// One head circuit's rank-revealing read: `V̂ᵀ` (`d_model × d_model`, rows descending by singular value), the
/// singular values, the backward-error band, the resolved rank and the resolved clusters.
struct Decomposition {
    layer: usize,
    head: usize,
    circuit: Circuit,
    read: Array2<f64>,
    singular_values: Vec<f64>,
    band: f64,
    rank: usize,
    /// The clusters that lie wholly within the resolved rank: the circuit's mask groups.
    groups: Vec<Range<usize>>,
    /// The first component of the tail, always off.
    tail_start: usize,
}

/// `B_QK = W_Q^hᵀ W_K^h` or `B_OV = W_O^h W_V^h` of one head, from torch `Linear` layout weights.
fn circuit_form(weights: &[Array2<f64>; 4], head: usize, head_dim: usize, circuit: Circuit) -> Array2<f64> {
    let rows = head * head_dim..(head + 1) * head_dim;
    match circuit {
        Circuit::Qk => weights[0].slice(s![rows.clone(), ..]).t().dot(&weights[1].slice(s![rows, ..])),
        Circuit::Ov => weights[3].slice(s![.., rows.clone()]).dot(&weights[2].slice(s![rows, ..])),
    }
}

fn decompose(weights: &[Array2<f64>; 4], layer: usize, head: usize, head_dim: usize, circuit: Circuit) -> Result<Decomposition, String> {
    let revealing =
        RankRevealingRead::new(&circuit_form(weights, head, head_dim, circuit)).map_err(|error| error.to_string())?;
    let spectrum = revealing.spectrum();
    let (singular_values, band, rank) = (spectrum.singular_values().to_vec(), spectrum.band(), spectrum.rank());
    let groups: Vec<Range<usize>> = spectrum.clusters().iter().filter(|cluster| cluster.end <= rank).cloned().collect();
    let tail_start = groups.last().map_or(0, |group| group.end);
    Ok(Decomposition {
        layer,
        head,
        circuit,
        read: revealing.read().to_owned(),
        singular_values,
        band,
        rank,
        groups,
        tail_start,
    })
}

/// A mask group: components `range` of declared circuit `circuit`.
#[derive(Clone, Debug)]
struct Group {
    circuit: usize,
    range: Range<usize>,
    /// Its components' summed singular values: the order free groups split in.
    mass: f64,
}

/// The component network: every layer's key and value reads factor through their heads' rank-revealing reads.
struct ComponentNetwork {
    network: Network,
    /// Per layer, the key and value reads' rows `(n_heads · d_model) × d_model` and candidate writes.
    factors: Vec<[(Array2<f64>, Array2<f64>); 2]>,
    model_dim: usize,
}

/// The per-component controls of one declared circuit, `d_model` of them.
type Sides = Vec<MaskSide>;

impl ComponentNetwork {
    /// The layers on `stored` weights with `reads[layer][circuit][head]`, each `d_model × d_model` with rows the
    /// components. The candidate write of head `h`'s block is `W^h R_hᵀ`, the exact write of an orthonormal read.
    fn build(
        loaded: &LoadedExport,
        stored: &[[Array2<f64>; 4]],
        reads: &[[Vec<Array2<f64>>; 2]],
    ) -> Result<Self, String> {
        let (geometry, score_scale) = layout(loaded)?;
        let mut layers = Vec::with_capacity(stored.len());
        let mut factors = Vec::with_capacity(stored.len());
        for (weights, layer_reads) in stored.iter().zip(reads) {
            let layer_factors = [0, 1].map(|circuit| stacked_factor(&geometry, &weights[1 + circuit], &layer_reads[circuit]));
            layers.push(component_layer(geometry, score_scale, weights, &layer_factors)?);
            factors.push(layer_factors);
        }
        Ok(Self {
            network: Network::new(loaded, layers)?,
            factors,
            model_dim: geometry.model_dim,
        })
    }

    /// The same network with the declared circuits' resolved read rows and write columns decoded from their lattice
    /// codewords at `precision`, and each group's codeword length in bits.
    fn decoded(&self, decompositions: &[Decomposition], groups: &[Group], precision: DeclaredPrecision) -> Result<(Self, Vec<u64>), String> {
        let head_dim = self.network.head_dim;
        let mut factors = self.factors.clone();
        let mut bits = Vec::with_capacity(groups.len());
        for group in groups {
            let decomposition = &decompositions[group.circuit];
            let slot = circuit_slot(decomposition.circuit);
            let (read, write) = &mut factors[decomposition.layer][slot];
            let head_rows = decomposition.head * head_dim..(decomposition.head + 1) * head_dim;
            let mut reals = Vec::new();
            for component in group.range.clone() {
                let row = decomposition.head * self.model_dim + component;
                reals.extend(read.row(row).iter());
                reals.extend(write.slice(s![head_rows.clone(), row]).iter());
            }
            let code = LatticeCode::encode(&reals, precision)?;
            bits.push(code.index_bits()?);
            let mut values = code.decode()?.into_iter();
            for component in group.range.clone() {
                let row = decomposition.head * self.model_dim + component;
                read.row_mut(row).iter_mut().zip(values.by_ref()).for_each(|(entry, value)| *entry = value);
                write
                    .slice_mut(s![head_rows.clone(), row])
                    .iter_mut()
                    .zip(values.by_ref())
                    .for_each(|(entry, value)| *entry = value);
            }
        }
        Ok((self.refactored(factors)?, bits))
    }

    /// The same network with other key and value factors, its key and value weights their products.
    fn refactored(&self, factors: Vec<[(Array2<f64>, Array2<f64>); 2]>) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(factors.len());
        for (layer, layer_factors) in self.network.layers.iter().zip(&factors) {
            let native = layer.native();
            let mut weights = PROJECTIONS.map(|projection| native.weight(projection).to_owned());
            for (slot, (read, write)) in layer_factors.iter().enumerate() {
                weights[1 + slot] = write.dot(read);
            }
            layers.push(component_layer(native.geometry(), native.score_scale(), &weights, layer_factors)?);
        }
        Ok(Self {
            network: self.network.with_layers(layers),
            factors,
            model_dim: self.model_dim,
        })
    }

    /// The logits of one sequence, `T × vocab`, with their radius: on the stored reads with no controls, or with
    /// the key and value reads through their components, each declared circuit's components under `sides`
    /// (`Free` reads `1/2 ± 1/2`) and every other circuit's on.
    fn logits(&self, tokens: &[i64], decompositions: &[Decomposition], sides: Option<&[Sides]>) -> Result<Rows, String> {
        let network = &self.network;
        let width = network.heads * self.model_dim;
        let controls: Vec<[(Array2<f64>, Option<Array2<f64>>); 2]> = (0..network.layers.len())
            .map(|layer| {
                [0, 1].map(|slot| {
                    let mut center = Array2::<f64>::ones((1, width));
                    let mut half_width = Array2::<f64>::zeros((1, width));
                    let mut free = false;
                    if let Some(sides) = sides {
                        for (decomposition, circuit_sides) in decompositions.iter().zip(sides) {
                            if decomposition.layer != layer || circuit_slot(decomposition.circuit) != slot {
                                continue;
                            }
                            for (component, side) in circuit_sides.iter().enumerate() {
                                let column = decomposition.head * self.model_dim + component;
                                center[[0, column]] = match side {
                                    MaskSide::On => 1.0,
                                    MaskSide::Off => 0.0,
                                    MaskSide::Free => 0.5,
                                };
                                if *side == MaskSide::Free {
                                    half_width[[0, column]] = 0.5;
                                    free = true;
                                }
                            }
                        }
                    }
                    (center, free.then_some(half_width))
                })
            })
            .collect();
        let reads: Vec<AttentionLayerReads<'_>> = controls
            .iter()
            .map(|layer_controls| {
                let read = move |slot: usize| match sides {
                    Some(_) => ProjectionRead::Components(ComponentMasks {
                        center: layer_controls[slot].0.view(),
                        half_width: layer_controls[slot].1.as_ref().map(|half_width| half_width.view()),
                    }),
                    None => ProjectionRead::Native,
                };
                AttentionLayerReads {
                    key: read(0),
                    value: read(1),
                    ..AttentionLayerReads::native()
                }
            })
            .collect();
        Ok(network.run(tokens, &reads)?.0)
    }
}

fn circuit_slot(circuit: Circuit) -> usize {
    match circuit {
        Circuit::Qk => 0,
        Circuit::Ov => 1,
    }
}

/// A layer's stacked read for one projection, `(n_heads · d_model) × d_model`, and its block candidate write
/// `(n_heads · d_head) × (n_heads · d_model)`.
fn stacked_factor(geometry: &AttentionGeometry, weight: &Array2<f64>, reads: &[Array2<f64>]) -> (Array2<f64>, Array2<f64>) {
    let (model, head_dim) = (geometry.model_dim, geometry.head_dim);
    let mut read = Array2::<f64>::zeros((reads.len() * model, model));
    let mut write = Array2::<f64>::zeros((reads.len() * head_dim, reads.len() * model));
    for (head, head_read) in reads.iter().enumerate() {
        read.slice_mut(s![head * model..(head + 1) * model, ..]).assign(head_read);
        let rows = head * head_dim..(head + 1) * head_dim;
        let block = weight.slice(s![rows.clone(), ..]).dot(&head_read.t());
        write.slice_mut(s![rows, head * model..(head + 1) * model]).assign(&block);
    }
    (read, write)
}

/// A layer whose query and output projections read through the identity and whose key and value projections read
/// through `factors`.
fn component_layer(
    geometry: AttentionGeometry,
    score_scale: f64,
    weights: &[Array2<f64>; 4],
    factors: &[(Array2<f64>, Array2<f64>); 2],
) -> Result<ComponentAttentionLayer, String> {
    let identities = [0, 3].map(|index| Array2::<f64>::eye(weights[index].ncols()));
    let reads = [
        ComponentRead {
            read: identities[0].view(),
            candidate_write: weights[0].view(),
        },
        ComponentRead {
            read: factors[0].0.view(),
            candidate_write: factors[0].1.view(),
        },
        ComponentRead {
            read: factors[1].0.view(),
            candidate_write: factors[1].1.view(),
        },
        ComponentRead {
            read: identities[1].view(),
            candidate_write: weights[3].view(),
        },
    ];
    factored_layer(geometry, score_scale, weights.each_ref().map(|weight| weight.clone()), reads)
}

/// A support's code: its subset codeword over the groups, the padded group length `H` once, and `H` bits per kept
/// group.
struct GroupCode {
    group_bits: u64,
}

impl CardinalityCode for GroupCode {
    type Error = CodecError;

    fn support_bits(&self, components: usize, size: usize) -> Result<u64, CodecError> {
        Ok(subset_code_len_bits(components, size)? + prefix_integer_len_bits(self.group_bits)? + size as u64 * self.group_bits)
    }
}

/// The finite family the oracle's evidence is stated over.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct GroupFamily {
    groups: usize,
    rows: usize,
}

/// The component network as a box program for `supports::BoxSeparationOracle`.
struct ComponentBoxes<'a, 'b> {
    artifact: &'a ComponentNetwork,
    decompositions: &'a [Decomposition],
    groups: &'a [Group],
    tokens: &'a [Vec<i64>],
    induction: &'b mut InductionRows,
    enclosed: &'b mut BTreeMap<MaskBox, BoxEnclosure<GroupFamily>>,
}

/// Each declared circuit's per-component controls under a box of group masks; tails are off.
fn group_sides(decompositions: &[Decomposition], groups: &[Group], mask: &MaskBox) -> Vec<Sides> {
    let mut sides: Vec<Sides> = decompositions.iter().map(|decomposition| vec![MaskSide::Off; decomposition.read.nrows()]).collect();
    for (group, side) in groups.iter().zip(mask.sides()) {
        for component in group.range.clone() {
            sides[group.circuit][component] = *side;
        }
    }
    sides
}

impl BoxDivergence for ComponentBoxes<'_, '_> {
    type Domain = GroupFamily;
    type Error = String;

    fn components(&self) -> usize {
        self.groups.len()
    }

    fn domain(&self) -> GroupFamily {
        GroupFamily {
            groups: self.groups.len(),
            rows: self.induction.rows.len(),
        }
    }

    fn enclose(&mut self, mask: &MaskBox) -> Result<BoxEnclosure<GroupFamily>, String> {
        if let Some(found) = self.enclosed.get(mask) {
            return Ok(found.clone());
        }
        let domain = self.domain();
        let sides = group_sides(self.decompositions, self.groups, mask);
        let (artifact, decompositions, tokens) = (self.artifact, self.decompositions, self.tokens);
        let rows = self
            .induction
            .at(mask, |sequence| artifact.logits(&tokens[sequence], decompositions, Some(&sides)))?;
        let status = if mask.is_vertex() {
            match InductionRows::largest(&rows) {
                Largest::Resolved { value, error } => {
                    EvidenceStatus::exact(value, error, ExactBasis::Exhaustive { cardinality: 1 }, Some(mask.clone()), domain)
                }
                Largest::Unresolved { lower } => {
                    EvidenceStatus::unresolved(lower, f64::INFINITY, Extremum::Supremum, Some(mask.clone()), domain)
                }
            }
        } else {
            let upper = rows.iter().map(|row| row.upper).try_fold(0.0_f64, |largest, upper| upper.map(|upper| largest.max(upper)));
            let error = rows.iter().filter_map(|row| row.resolved).fold(0.0_f64, |largest, (_, own)| largest.max(own));
            let lower = rows.iter().filter_map(|row| row.lower).fold(0.0_f64, f64::max);
            match upper {
                Some(upper) => EvidenceStatus::uniform_bound(upper, error, domain),
                None => EvidenceStatus::unresolved(lower, f64::INFINITY, Extremum::Supremum, None, domain),
            }
        };
        let evidence = status.map_err(|error| format!("{error:?}"))?;
        let mut split_order = mask.free();
        split_order.sort_by(|left, right| self.groups[*right].mass.total_cmp(&self.groups[*left].mass));
        let enclosure = BoxEnclosure { evidence, split_order };
        self.enclosed.insert(mask.clone(), enclosure.clone());
        Ok(enclosure)
    }
}

/// A divergence over the induction rows: its value and numerical error, or only its lower side.
#[derive(Serialize)]
struct Risk {
    value: Option<f64>,
    numerical_error: Option<f64>,
    lower: Option<f64>,
}

impl Risk {
    fn of(rows: &[RowDivergence]) -> Self {
        match InductionRows::largest(rows) {
            Largest::Resolved { value, error } => Self {
                value: Some(value),
                numerical_error: Some(error),
                lower: Some((value - error).next_down().max(0.0)),
            },
            Largest::Unresolved { lower } => Self {
                value: None,
                numerical_error: None,
                lower: Some(lower),
            },
        }
    }

    /// Whether the risk is proven at most `tolerance` (`meets`), proven above it (`violates`), or neither.
    fn verdict(&self, tolerance: f64) -> &'static str {
        match (self.value, self.numerical_error, self.lower) {
            (Some(value), Some(error), _) if (value + error).next_up() <= tolerance => "meets",
            (_, _, Some(lower)) if lower > tolerance => "violates",
            _ => "unresolved",
        }
    }
}

#[derive(Serialize)]
struct CircuitReport {
    name: String,
    singular_values: Vec<f64>,
    band: f64,
    rank: usize,
    groups: Vec<(usize, usize)>,
    tail: usize,
    tail_largest_singular_value: f64,
}

#[derive(Serialize)]
struct NullReport {
    /// Per declared circuit, the number of components the support keeps.
    kept_components: Vec<usize>,
    bottom: Risk,
    bottom_verdict: String,
    haar: Vec<Risk>,
    haar_verdicts: Vec<String>,
    /// The mean of the Haar draws' exact values, with its standard error: an estimate, never a bound.
    haar_mean: Option<(f64, f64)>,
}

#[derive(Serialize)]
struct ToleranceReport {
    scale: String,
    declared: f64,
    tolerance: f64,
    vacuous: bool,
    code_status: String,
    code_bits_lower: f64,
    code_bits_upper: f64,
    support: Vec<String>,
    separations: usize,
    edges: Vec<Vec<String>>,
    nulls: Option<NullReport>,
}

/// A status's proven sides.
#[derive(Serialize)]
struct Proven {
    lower: Option<f64>,
    upper: Option<f64>,
}

impl Proven {
    fn of<W, D>(status: &EvidenceStatus<W, D>) -> Self {
        Self {
            lower: status.lower_bound(),
            upper: status.upper_bound(),
        }
    }
}

/// The planted-duplicate OR control at one declared circuit.
#[derive(Serialize)]
struct OrControl {
    circuit: String,
    /// `a`, the circuit's first component, and `b`, its copy planted in the first tail row.
    pair: (String, String),
    /// Every group on, then `a`, `b` and both off.
    vertices: [Proven; 4],
    /// Strictly between the single-copy vertices' largest proven upper bound and the both-off proven lower bound.
    tolerance: Option<f64>,
    /// `U∖{a}`, `U∖{b}` and `U∖{a, b}` at the tolerance.
    decisions: Vec<String>,
    /// The groups the refutation of `U∖{a, b}` perturbs.
    witness: Option<Vec<String>>,
    verdict: String,
}

#[derive(Serialize)]
struct Report {
    checkpoint: u64,
    teacher_fingerprint: String,
    circuits: Vec<CircuitReport>,
    groups: Vec<String>,
    fraction_bits: i32,
    group_bits: Vec<u64>,
    padded_group_bits: u64,
    induction_rows: usize,
    teacher_vs_torch: f64,
    /// The largest logit gap between the teacher and the resolved-only network (every tail off, no lattice), with
    /// both radii: the tails' measured effect.
    tail_logit_gap: f64,
    tail_logit_radius: f64,
    family: GroupFamily,
    baseline: Risk,
    artifact_all_on: Risk,
    enclosures: usize,
    masks_evaluated: usize,
    /// Each declared circuit tried in order, up to the first that discriminates, and the overall verdict: `pass`,
    /// `fail`, or `missing` when no declared circuit discriminates.
    or_controls: Vec<OrControl>,
    or_control_verdict: String,
    tolerances: Vec<ToleranceReport>,
}

fn group_name(decompositions: &[Decomposition], group: &Group) -> String {
    let decomposition = &decompositions[group.circuit];
    let range = if group.range.len() == 1 {
        format!("{}", group.range.start)
    } else {
        format!("{}-{}", group.range.start, group.range.end - 1)
    };
    format!("L{}H{}.{}.{range}", decomposition.layer, decomposition.head, decomposition.circuit.name())
}

/// A Haar-random orthonormal `n × n` matrix: the right singular vectors of a Gaussian matrix drawn by Box-Muller.
fn haar(rng: &mut StdRng, n: usize) -> Result<Array2<f64>, String> {
    let gaussian = Array2::from_shape_simple_fn((n, n), || {
        let radius = (-2.0 * (1.0 - rng.random::<f64>()).ln()).sqrt();
        radius * (std::f64::consts::TAU * rng.random::<f64>()).cos()
    });
    Ok(RankRevealingRead::new(&gaussian).map_err(|error| error.to_string())?.read().to_owned())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(USAGE.to_string());
    }
    let export_dir = PathBuf::from(flag(&args, "--export", USAGE)?);
    let settings_path = PathBuf::from(flag(&args, "--settings", USAGE)?);
    let report_path = PathBuf::from(flag(&args, "--out", USAGE)?);
    let text = std::fs::read_to_string(&settings_path).map_err(|error| format!("read {}: {error}", settings_path.display()))?;
    let settings: Settings = serde_json::from_str(&text).map_err(|error| format!("{}: {error}", settings_path.display()))?;
    if settings.circuits.is_empty()
        || settings.baseline_fractions.is_empty()
        || settings
            .baseline_fractions
            .iter()
            .chain(&settings.absolute_tolerances)
            .any(|value| !(value.is_finite() && *value >= 0.0))
    {
        return Err("the settings must declare circuits, baseline fractions, and finite non-negative tolerances".to_string());
    }
    let precision = DeclaredPrecision::new(settings.fraction_bits)?;

    let loaded = load_export(&export_dir)?;
    let export = &loaded.export;
    let config = &export.config;
    let stored: Vec<[Array2<f64>; 4]> =
        (0..config.n_layers).map(|layer| stored_layer(&loaded, layer)).collect::<Result<_, _>>()?;
    // Every head's two rank-revealing reads, and the declared circuits' decompositions.
    let mut reads: Vec<[Vec<Array2<f64>>; 2]> = Vec::with_capacity(config.n_layers);
    let mut decompositions = Vec::with_capacity(settings.circuits.len());
    for (layer, weights) in stored.iter().enumerate() {
        let mut layer_reads = [Vec::new(), Vec::new()];
        for head in 0..config.n_heads {
            for circuit in [Circuit::Qk, Circuit::Ov] {
                let decomposition = decompose(weights, layer, head, config.d_head, circuit)?;
                layer_reads[circuit_slot(circuit)].push(decomposition.read.clone());
            }
        }
        reads.push(layer_reads);
    }
    for (layer, head, circuit) in &settings.circuits {
        let circuit = match circuit.as_str() {
            "qk" => Circuit::Qk,
            "ov" => Circuit::Ov,
            other => return Err(format!("circuit {other:?}: declare \"qk\" or \"ov\"")),
        };
        if *layer >= config.n_layers || *head >= config.n_heads {
            return Err(format!("circuit L{layer}H{head} is outside the network"));
        }
        decompositions.push(decompose(&stored[*layer], *layer, *head, config.d_head, circuit)?);
    }
    let groups: Vec<Group> = decompositions
        .iter()
        .enumerate()
        .flat_map(|(circuit, decomposition)| {
            decomposition.groups.iter().map(move |range| Group {
                circuit,
                range: range.clone(),
                mass: decomposition.singular_values[range.clone()].iter().sum(),
            })
        })
        .collect();
    if groups.is_empty() {
        return Err("the declared circuits resolve no component to search".to_string());
    }
    let teacher = ComponentNetwork::build(&loaded, &stored, &reads)?;
    let (artifact, group_bits) = teacher.decoded(&decompositions, &groups, precision)?;
    let padded_group_bits = group_bits.iter().copied().max().ok_or("no group")?;
    let code = GroupCode {
        group_bits: padded_group_bits,
    };
    let names: Vec<String> = groups.iter().map(|group| group_name(&decompositions, group)).collect();

    let mut induction = InductionRows::new(&export.tokens, |tokens| teacher.logits(tokens, &decompositions, None))?;
    let torch_logits = float64_array(&export.files, &export_dir, "native_logits", (export.sequences * config.seq_len, config.vocab))?;
    let (teacher_vs_torch, _, _) = induction_network::teacher_vs_torch(&induction.reference, &torch_logits, config.seq_len);

    // The tails' measured effect: the teacher against itself with every declared resolved component on and every
    // tail off, with no lattice.
    let resolved_only: Vec<Sides> = decompositions
        .iter()
        .map(|decomposition| {
            (0..decomposition.read.nrows())
                .map(|component| if component < decomposition.tail_start { MaskSide::On } else { MaskSide::Off })
                .collect()
        })
        .collect();
    let (mut tail_logit_gap, mut tail_logit_radius) = (0.0_f64, 0.0_f64);
    for (sequence, tokens) in export.tokens.iter().enumerate() {
        let resolved = teacher.logits(tokens, &decompositions, Some(&resolved_only))?;
        let reference = &induction.reference[sequence];
        for ((entry, &value), &radius) in resolved.values.indexed_iter().zip(resolved.radius.iter()) {
            let gap = (value - reference.values[entry]).abs();
            if gap > tail_logit_gap {
                tail_logit_gap = gap;
                tail_logit_radius = radius + reference.radius[entry];
            }
        }
    }

    let all_on = MaskBox::vertex(&ComponentSet::all(groups.len()));
    let all_off = MaskBox::vertex(
        &ComponentSet::new(groups.len(), Vec::new()).map_err(|error| error.to_string())?,
    );
    let mut enclosed = BTreeMap::new();
    let evaluate = |induction: &mut InductionRows, mask: &MaskBox| {
        let sides = group_sides(&decompositions, &groups, mask);
        induction.at(mask, |sequence| artifact.logits(&export.tokens[sequence], &decompositions, Some(&sides)))
    };
    let baseline = Risk::of(&evaluate(&mut induction, &all_off)?);
    let artifact_all_on = Risk::of(&evaluate(&mut induction, &all_on)?);
    let do_nothing = baseline
        .value
        .ok_or("the all-off divergence is unresolved, so no tolerance can be a fraction of it")?;
    println!(
        "[load] checkpoint={} groups={} rows={} group_bits={padded_group_bits} teacher_vs_torch={teacher_vs_torch:.3e} tail_gap={tail_logit_gap:.3e}±{tail_logit_radius:.3e} baseline={do_nothing:.6e} all_on={:?}",
        export.checkpoint,
        groups.len(),
        induction.rows.len(),
        artifact_all_on.value
    );

    let mut or_controls = Vec::new();
    for circuit in 0..decompositions.len() {
        let Some(control) = or_control(&teacher, &decompositions, &groups, &names, circuit, &export.tokens)? else {
            println!("[or-control] circuit {circuit}: not applicable, its first group is not component 0 alone or it has no tail row");
            continue;
        };
        println!(
            "[or-control] {} pair={:?} tolerance={:?} decisions={:?} witness={:?} verdict={}",
            control.circuit, control.pair, control.tolerance, control.decisions, control.witness, control.verdict
        );
        let decided = control.tolerance.is_some();
        or_controls.push(control);
        if decided {
            break;
        }
    }
    let or_control_verdict = match or_controls.last() {
        Some(control) if control.verdict == "pass" => "pass",
        Some(control) if control.tolerance.is_some() => "fail",
        _ => "missing",
    }
    .to_string();

    let entries: Vec<(&str, f64, f64)> = settings
        .baseline_fractions
        .iter()
        .map(|&fraction| ("relative", fraction, fraction * do_nothing))
        .chain(settings.absolute_tolerances.iter().map(|&tolerance| ("absolute", tolerance, tolerance)))
        .collect();
    let mut rng = StdRng::seed_from_u64(settings.haar_seed);
    let mut reports = Vec::with_capacity(entries.len());
    for &(scale, declared, tolerance) in &entries {
        let search = {
            let program = ComponentBoxes {
                artifact: &artifact,
                decompositions: &decompositions,
                groups: &groups,
                tokens: &export.tokens,
                induction: &mut induction,
                enclosed: &mut enclosed,
            };
            let oracle = BoxSeparationOracle::new(program, tolerance).map_err(|error| format!("{error:?}"))?;
            let mut oracle = Logged::new(oracle, format!("{scale} declared={declared:.3e}"));
            minimum_code_support(&mut oracle, &code, tolerance, FailureHypergraph::new(groups.len()))
        };
        let search = match search {
            Ok(search) => search,
            Err(error) => {
                println!("[components] {scale} declared={declared:.3e} tolerance={tolerance:.3e} refused: {error:?}");
                reports.push(ToleranceReport {
                    scale: scale.to_string(),
                    declared,
                    tolerance,
                    vacuous: false,
                    code_status: format!("{error:?}"),
                    code_bits_lower: f64::INFINITY,
                    code_bits_upper: f64::INFINITY,
                    support: Vec::new(),
                    separations: 0,
                    edges: Vec::new(),
                    nulls: None,
                });
                continue;
            }
        };
        let (code_status, code_bits_lower, code_bits_upper) = match &search.code {
            EvidenceStatus::Exact { value, .. } => ("exact", *value, *value),
            EvidenceStatus::Unresolved { lower, upper, .. } => ("unresolved", *lower, *upper),
            other => return Err(format!("tolerance {tolerance}: an unexpected code status {other:?}")),
        };
        let certified = search.certified.as_ref().map(|found| found.support.clone());
        let support: Vec<String> = certified
            .as_ref()
            .map_or_else(Vec::new, |support| support.members().iter().map(|&group| names[group].clone()).collect());
        let vacuous = code_status == "exact" && certified.as_ref().is_some_and(|support| support.is_empty());
        let edges = search
            .hypergraph
            .edges()
            .iter()
            .map(|edge| edge.perturbed.members().iter().map(|&group| names[group].clone()).collect())
            .collect();

        // Nulls at the certified support's per-circuit counts.
        let nulls = match &certified {
            Some(support) => {
                let mut kept_groups = vec![0usize; decompositions.len()];
                let mut kept_components = vec![0usize; decompositions.len()];
                for &group in support.members() {
                    kept_groups[groups[group].circuit] += 1;
                    kept_components[groups[group].circuit] += groups[group].range.len();
                }
                // Bottom: each circuit's same number of groups of least singular mass.
                let mut bottom_on = Vec::new();
                for (circuit, &count) in kept_groups.iter().enumerate() {
                    let mut members: Vec<usize> = (0..groups.len()).filter(|&group| groups[group].circuit == circuit).collect();
                    members.sort_by(|left, right| groups[*left].mass.total_cmp(&groups[*right].mass));
                    bottom_on.extend(members.into_iter().take(count));
                }
                let bottom_mask = MaskBox::vertex(
                    &ComponentSet::new(groups.len(), bottom_on)
                        .map_err(|error| error.to_string())?,
                );
                let bottom = Risk::of(&evaluate(&mut induction, &bottom_mask)?);
                // Haar: each circuit's resolved rows rotated by a Haar draw, keeping as many components.
                let mut haar_risks = Vec::with_capacity(settings.haar_draws);
                for _ in 0..settings.haar_draws {
                    let mut factors = teacher.factors.clone();
                    let mut sides: Vec<Sides> = Vec::with_capacity(decompositions.len());
                    for (circuit, decomposition) in decompositions.iter().enumerate() {
                        let resolved = decomposition.tail_start;
                        if resolved == 0 {
                            sides.push(vec![MaskSide::Off; decomposition.read.nrows()]);
                            continue;
                        }
                        let rotation = haar(&mut rng, resolved)?;
                        let slot = circuit_slot(decomposition.circuit);
                        let (read, write) = &mut factors[decomposition.layer][slot];
                        let rows = decomposition.head * teacher.model_dim..decomposition.head * teacher.model_dim + resolved;
                        let rotated = rotation.dot(&decomposition.read.slice(s![..resolved, ..]));
                        read.slice_mut(s![rows.clone(), ..]).assign(&rotated);
                        let head_rows = decomposition.head * config.d_head..(decomposition.head + 1) * config.d_head;
                        let block = stored[decomposition.layer][1 + slot].slice(s![head_rows.clone(), ..]).dot(&rotated.t());
                        write.slice_mut(s![head_rows, rows]).assign(&block);
                        sides.push(
                            (0..decomposition.read.nrows())
                                .map(|component| if component < kept_components[circuit] { MaskSide::On } else { MaskSide::Off })
                                .collect(),
                        );
                    }
                    let rotated = teacher.refactored(factors)?;
                    let (null_artifact, _) = rotated.decoded(&decompositions, &groups_of_rotated(&decompositions), precision)?;
                    let rows = induction
                        .evaluate(|sequence| null_artifact.logits(&export.tokens[sequence], &decompositions, Some(&sides)))?;
                    haar_risks.push(Risk::of(&rows));
                }
                let values: Vec<f64> = haar_risks.iter().filter_map(|risk| risk.value).collect();
                let haar_mean = (values.len() == haar_risks.len() && values.len() >= 2).then(|| {
                    let count = values.len() as f64;
                    let mean = values.iter().sum::<f64>() / count;
                    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (count - 1.0);
                    (mean, (variance / count).sqrt())
                });
                Some(NullReport {
                    kept_components,
                    bottom_verdict: bottom.verdict(tolerance).to_string(),
                    bottom,
                    haar_verdicts: haar_risks.iter().map(|risk| risk.verdict(tolerance).to_string()).collect(),
                    haar: haar_risks,
                    haar_mean,
                })
            }
            None => None,
        };
        println!(
            "[components] {scale} declared={declared:.3e} tolerance={tolerance:.3e} code={code_status} bits=[{code_bits_lower}, {code_bits_upper}] support={support:?} vacuous={vacuous} separations={} edges={} bottom={:?} haar={:?}",
            search.separations,
            search.hypergraph.edges().len(),
            nulls.as_ref().map(|nulls| nulls.bottom_verdict.clone()),
            nulls.as_ref().map(|nulls| nulls.haar_verdicts.clone()),
        );
        reports.push(ToleranceReport {
            scale: scale.to_string(),
            declared,
            tolerance,
            vacuous,
            code_status: code_status.to_string(),
            code_bits_lower,
            code_bits_upper,
            support,
            separations: search.separations,
            edges,
            nulls,
        });
    }

    let report = Report {
        checkpoint: export.checkpoint,
        teacher_fingerprint: format!("{:#018x}", loaded.registry.teacher_fingerprint().0),
        circuits: decompositions
            .iter()
            .map(|decomposition| CircuitReport {
                name: format!("L{}H{}.{}", decomposition.layer, decomposition.head, decomposition.circuit.name()),
                singular_values: decomposition.singular_values.clone(),
                band: decomposition.band,
                rank: decomposition.rank,
                groups: decomposition.groups.iter().map(|group| (group.start, group.end)).collect(),
                tail: decomposition.read.nrows() - decomposition.tail_start,
                tail_largest_singular_value: decomposition.singular_values.get(decomposition.tail_start).copied().unwrap_or(0.0),
            })
            .collect(),
        groups: names,
        fraction_bits: settings.fraction_bits,
        group_bits,
        padded_group_bits,
        induction_rows: induction.rows.len(),
        teacher_vs_torch,
        tail_logit_gap,
        tail_logit_radius,
        family: GroupFamily {
            groups: groups.len(),
            rows: induction.rows.len(),
        },
        baseline,
        artifact_all_on,
        enclosures: enclosed.len(),
        masks_evaluated: induction.evaluated(),
        or_controls,
        or_control_verdict,
        tolerances: reports,
    };
    let text = serde_json::to_string_pretty(&report).map_err(|error| format!("report: {error}"))?;
    std::fs::write(&report_path, text).map_err(|error| format!("write {}: {error}", report_path.display()))?;
    println!(
        "[done] enclosures={} masks_evaluated={} or_control={} report={}",
        report.enclosures,
        report.masks_evaluated,
        report.or_control_verdict,
        report_path.display()
    );
    if report.or_control_verdict == "fail" {
        return Err("the planted-duplicate OR control failed, so the box-oracle route is wrong".to_string());
    }
    Ok(())
}

/// The groups of a Haar-rotated decomposition: every resolved row its own component, since a random basis of the
/// resolved space has no repeated singular values to tie.
fn groups_of_rotated(decompositions: &[Decomposition]) -> Vec<Group> {
    decompositions
        .iter()
        .enumerate()
        .flat_map(|(circuit, decomposition)| {
            (0..decomposition.tail_start).map(move |component| Group {
                circuit,
                range: component..component + 1,
                mass: 0.0,
            })
        })
        .collect()
}

/// The planted-duplicate OR control at declared circuit `circuit`, on the exact component network (no lattice). The
/// circuit's first tail row becomes a copy of its component-0 read row and both write columns are halved, so the
/// network is unchanged up to that tail term, which is always off, and depends on the pair `a`, `b` only through
/// `m_a + m_b`. When the single-copy vertices' largest proven upper bound (every group on, `a` off, `b` off) lies
/// below the both-off vertex's proven lower bound, the tolerance between them must certify `U∖{a}` and `U∖{b}` and
/// refute `U∖{a, b}` with a shrunk witness perturbing exactly `{a, b}`: the OR edge. It can fail only if the
/// enclosure, the refinement or the shrink is wrong. `None` when the circuit's first group is not component 0 alone,
/// or it has no tail row to plant the copy in.
fn or_control(
    teacher: &ComponentNetwork,
    decompositions: &[Decomposition],
    groups: &[Group],
    names: &[String],
    circuit: usize,
    tokens: &[Vec<i64>],
) -> Result<Option<OrControl>, String> {
    let decomposition = &decompositions[circuit];
    let Some(a) = groups.iter().position(|group| group.circuit == circuit && group.range == (0..1)) else {
        return Ok(None);
    };
    if decomposition.tail_start >= decomposition.read.nrows() {
        return Ok(None);
    }
    let head_dim = teacher.network.head_dim;
    let (first, copy) = (decomposition.head * teacher.model_dim, decomposition.head * teacher.model_dim + decomposition.tail_start);
    let mut factors = teacher.factors.clone();
    let (read, write) = &mut factors[decomposition.layer][circuit_slot(decomposition.circuit)];
    let row = read.row(first).to_owned();
    read.row_mut(copy).assign(&row);
    let head_rows = decomposition.head * head_dim..(decomposition.head + 1) * head_dim;
    let half = write.slice(s![head_rows.clone(), first]).mapv(|entry| 0.5 * entry);
    write.slice_mut(s![head_rows.clone(), first]).assign(&half);
    write.slice_mut(s![head_rows, copy]).assign(&half);
    let planted = teacher.refactored(factors)?;
    let mut planted_groups = groups.to_vec();
    planted_groups.push(Group {
        circuit,
        range: decomposition.tail_start..decomposition.tail_start + 1,
        mass: groups[a].mass,
    });
    let b = groups.len();
    let mut planted_names = names.to_vec();
    planted_names.push(format!("{} (copy of {})", group_name(decompositions, &planted_groups[b]), names[a]));
    let count = planted_groups.len();
    let without = |off: &[usize]| {
        ComponentSet::new(count, off.to_vec()).map(|off| off.complement()).map_err(|error| error.to_string())
    };

    let mut rows = InductionRows::new(tokens, |sequence_tokens| teacher.logits(sequence_tokens, decompositions, None))?;
    let mut enclosed = BTreeMap::new();
    let mut program = ComponentBoxes {
        artifact: &planted,
        decompositions,
        groups: &planted_groups,
        tokens,
        induction: &mut rows,
        enclosed: &mut enclosed,
    };
    let mut vertices = Vec::with_capacity(4);
    for off in [vec![], vec![a], vec![b], vec![a, b]] {
        vertices.push(program.enclose(&MaskBox::vertex(&without(&off)?))?.evidence);
    }
    let single = vertices[..3]
        .iter()
        .map(|status| status.upper_bound())
        .try_fold(0.0_f64, |largest, upper| upper.map(|upper| largest.max(upper)));
    let tolerance = match (single, vertices[3].lower_bound()) {
        (Some(single), Some(both)) if single < both => {
            let midpoint = single + 0.5 * (both - single);
            (single <= midpoint && midpoint < both).then_some(midpoint)
        }
        _ => None,
    };
    let mut control = OrControl {
        circuit: format!("L{}H{}.{}", decomposition.layer, decomposition.head, decomposition.circuit.name()),
        pair: (planted_names[a].clone(), planted_names[b].clone()),
        vertices: std::array::from_fn(|vertex| Proven::of(&vertices[vertex])),
        tolerance,
        decisions: Vec::new(),
        witness: None,
        verdict: "undiscriminating".to_string(),
    };
    let Some(tolerance) = tolerance else {
        return Ok(Some(control));
    };
    let mut oracle = BoxSeparationOracle::new(program, tolerance).map_err(|error| format!("{error:?}"))?;
    let mut statuses = Vec::with_capacity(3);
    for off in [vec![a], vec![b], vec![a, b]] {
        let status = oracle.separate(&without(&off)?).map_err(|error| format!("{error:?}"))?;
        control.decisions.push(decision(&status, tolerance));
        statuses.push(status);
    }
    let edge = statuses[2].witness().filter(|_| statuses[2].refutes_at_most(tolerance)).map(MaskBox::perturbed);
    control.witness = edge.as_ref().map(|perturbed| perturbed.iter().map(|&group| planted_names[group].clone()).collect());
    control.verdict = if !(statuses[0].certifies_at_most(tolerance) && statuses[1].certifies_at_most(tolerance)) {
        "fail: a single-copy support is not certified".to_string()
    } else if edge.as_deref() != Some(&[a, b][..]) {
        "fail: U∖{a, b} is not refuted by a witness perturbing exactly {a, b}".to_string()
    } else {
        "pass".to_string()
    };
    Ok(Some(control))
}

/// A separation's decision at `tolerance`, with its proven sides.
fn decision<W, D>(status: &EvidenceStatus<W, D>, tolerance: f64) -> String {
    let (lower, upper) = (status.lower_bound(), status.upper_bound());
    if status.certifies_at_most(tolerance) {
        format!("certified: upper {upper:?}")
    } else if status.refutes_at_most(tolerance) {
        format!("refuted: lower {lower:?}")
    } else {
        format!("unresolved: [{lower:?}, {upper:?}]")
    }
}
