//! Minimum-code head supports of mpd-induction's trained network at declared tolerances (#2951 S2a, S3).
//!
//! `mpd_induction_supports --export REGISTRY_EXPORT --settings SETTINGS_JSON --out REPORT_JSON`
//!
//! `REGISTRY_EXPORT` is what `bench/mpd_induction_2951.py registry` writes (read through
//! `support/induction_export.rs`). `SETTINGS_JSON` declares
//! `{"baseline_fractions": [f, …], "absolute_tolerances": [ε, …], "fraction_bits": p, "transplant": null}`:
//! the fidelity tolerances and the precision of the heads' reals are experiment inputs with no default (#2951
//! SPEC tension).
//!
//! # Tolerances and controls
//! A network is scored against its own all-on logits, so an absolute tolerance compares networks of different
//! output scales: a network with flat outputs (random init) meets a moderate ε with every head off, which is a
//! fact about its scale and none about a mechanism. The primary tolerances are therefore fractions of the
//! network's own do-nothing divergence, `ε_f = f · b̂` with `b̂` the exact all-off divergence. The absolute
//! tolerances stay in the report, and any tolerance whose certified support is empty is labelled vacuous: the
//! do-nothing baseline meets it by construction.
//!
//! A control run declares `"transplant": {"report": TRAINED_REPORT, "top_share": q, "asserted": bool}`. The
//! trained network's certified support at each fraction is carried into this network by head names, and its risk
//! is ranked among every support of its size. It is shown outside the top share `q` when at least `⌈q·n⌉` of the
//! `n` same-size supports have a risk whose upper bound lies below its risk's lower bound. A support of every head,
//! or of none, has no rival of its size, so the test cannot discriminate there and says so.
//!
//! # The artifact and its code
//! The components are the heads, `C = n_layers · n_heads`. Head `(l, h)` owns the rows `h·d_head … (h+1)·d_head`
//! of `W_Q.l`, `W_K.l` and `W_V.l` and the same columns of `W_O.l`. The artifact sends each kept head's reals
//! in `precision`'s lattice code at step `2^-p`, and the decoder rebuilds them exactly from the indices. The
//! embedding, positional and unembedding tensors are the base every support shares, and they are not coded here.
//! Fidelity is measured on the decoded artifact (`precision.rs`), so every decision includes the lattice
//! distortion.
//!
//! A support's message is the `EnumerativeSubsetCode` codeword of the kept set, then `H` once in the prefix
//! integer code, then every kept head's lattice codeword padded to `H`. `H` is the longest head's codeword, and
//! the codewords are self-delimiting, so the padded message decodes. Its length
//! `L(C, k) + L_int(H) + k·H` depends on the support's size alone, as `minimum_code_support` requires. The subset
//! code by itself would make every head cheapest: keeping all `C` components costs `L_int(C + 1)` bits, below
//! most `L(C, k)`.
//!
//! # Execution with rigorous radii
//! Each network, the teacher on its stored tensors and the decoded artifact under a head mask, runs natively
//! with no torch, through one route for every stage (`block`), and every stage carries a per-entry bound on its
//! distance from the exact program's value at the exact input:
//! * `x₀ = W_E[token] + W_pos`, one IEEE addition: `γ₁|x̂₀|`;
//! * each layer is a `block::ComponentAttentionLayer` fed the previous residual rows with their radius. The
//!   teacher reads every projection natively. The artifact reads `W_O` through the identity read under the head
//!   mask, `U diag(m) R` with `R = I` and `U = W_O`, so a masked head's columns contribute `0·z = 0` exactly;
//! * the unembedding is `block::linear_read` of the last residual rows with their radius.
//!
//! # Rows, divergence and supports
//! A test sequence is a prefix, a segment of length `n` and the segment's repeat, over distinct tokens. So the
//! last token occurs exactly once earlier, at `T − 1 − n`, which gives `n` from the tokens alone; the repeat is
//! checked. The induction rows are the repeat positions `t = T − n, …, T − 2`.
//! * **Row divergence at mask `m`:** `KL(teacher all-on ‖ artifact at m)`, bounded over both logit boxes by
//!   `bounds::kl_over_logit_boxes`.
//! * **Mask domain:** the binary endpoint family `{0,1}^C`. P7's interior masks are not covered.
//! * **Oracle:** `R(S) = max { d(m) : m_S = 1 }` over the `2^{C−|S|}` members that keep `S` on, with
//!   `d(m) = max over every induction row`. It is evaluated lazily and memoized per mask.
//! * **Evidence:** `Exact` over that finite family, with the largest per-row box error as its numerical error.
//!   It is `Unresolved` when a row's box bound is unresolved.
//! * **Search:** `minimum_code_support` finds, per tolerance, the minimum-code support and the failure
//!   hypergraph, whose edges are the OR constraints.
//! * **Per input (P18):** each induction row gets its own minimum-code support. `sufficient_union` takes their
//!   union, and the library (the union's heads) plus one subset packet per row prices the per-row artifact.
//!
//! Beside every code the report gives:
//! * the do-nothing baseline, every head off;
//! * the artifact's distortion with every head on;
//! * the teacher's native logits measured against torch's exported native logits.
//!
//! # Stage S3: position-scoped supports at one occurrence
//! Declared by `"positions": {"rows": [[sequence, position], …], "baseline_fractions": [f, …], "null_seed": s}`,
//! each row one occurrence of the mechanism. A component is one head read at a set of positions: its `d_head`
//! columns of `W_O` at those rows of `block::ComponentMasks` (one center row per position). The row's token occurs
//! earlier at `t − n`, so an induction head at `t` reads its key at the source `t − n + 1`, read off the tokens
//! alone. A row `t`'s components are each head of a layer below the last at the source alone and at every other
//! position `s ≤ t` together, and each head of the last layer at `t` alone: the last layer's write at another
//! position reaches only that position's logits, and no position after `t` reaches it. Every other head and
//! position stays on. (One component per head and position, `n_heads (t + 1) + n_heads` of them, left the box
//! oracle's enclosures too wide to certify at `t = 30`: the first probe's third separation ran 15 minutes.)
//! * **Divergence:** `KL(teacher ‖ artifact under the mask)` at row `t` alone, over both logit boxes. The
//!   tolerances are fractions of the row's own all-off divergence.
//! * **Search:** `BoxSeparationOracle` and `minimum_code_support` with the padded head code over the row's
//!   components, a function of the size alone, so the minimum-code support keeps the fewest instances. The report
//!   also gives the code that sends each distinct kept head once, and how many instances the same heads keep
//!   when each is kept at every position.
//! * **The mechanism's prediction:** the support's lower-layer heads are needed at the source and not at the
//!   rest. The report counts the support's source and rest components.
//! * **Null at equal count:** each kept source component read at a uniformly drawn other position instead
//!   (declared seed), with every other head and position off, must miss the tolerance; the support alone must
//!   meet it.

use gam_sae::parameter_decomposition::attention::AttentionGeometry;
use gam_sae::parameter_decomposition::block::{
    AttentionLayerReads, AttentionProjection, ComponentAttentionLayer, ComponentMasks, ProjectionRead,
};
use gam_sae::parameter_decomposition::codec::{
    CodecError, encode_support_packets, prefix_integer_len_bits, subset_code_len_bits, union_support_library,
};
use gam_sae::parameter_decomposition::precision::{DecodableArtifact, DeclaredPrecision, LatticeCode};
use gam_sae::parameter_decomposition::rewrite::ComponentRead;
use gam_sae::parameter_decomposition::supports::{
    BoxDivergence, BoxEnclosure, BoxSeparationOracle, CardinalityCode, ComponentSet, EvidenceStatus, ExactBasis,
    Extremum, FailureHypergraph, InputSupport, MaskBox, MaskSide, SeparationOracle, SupportSearch,
    SupportSearchError, minimum_code_support, sufficient_union,
};
use ndarray::{Array2, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
    row_divergence, stored_layer,
};

const USAGE: &str = "usage: mpd_induction_supports --export REGISTRY_EXPORT --settings SETTINGS_JSON --out REPORT_JSON";

#[derive(Deserialize)]
struct Settings {
    /// Tolerances as fractions `f` of the network's own all-off divergence `b̂`: `ε_f = f · b̂`.
    baseline_fractions: Vec<f64>,
    /// Absolute tolerances, reported so a result that is vacuous at an absolute ε is visible as such.
    absolute_tolerances: Vec<f64>,
    fraction_bits: i32,
    transplant: Option<TransplantSettings>,
    /// Stage S3, when declared: position-scoped supports at declared induction rows.
    positions: Option<PositionSettings>,
}

/// The induction rows `(sequence, position)` S3 scores, its tolerances as fractions of each row's own all-off
/// divergence, and the seed of its moved-position null.
#[derive(Deserialize)]
struct PositionSettings {
    rows: Vec<(usize, usize)>,
    baseline_fractions: Vec<f64>,
    null_seed: u64,
}

/// The trained network's report whose certified supports this control receives, the declared top share `q`,
/// and whether the control asserts that each support lies outside it.
#[derive(Deserialize)]
struct TransplantSettings {
    report: PathBuf,
    top_share: f64,
    asserted: bool,
}

/// The fields of a trained report a transplant reads.
#[derive(Deserialize)]
struct TrainedReport {
    tolerances: Vec<TrainedTolerance>,
}

#[derive(Deserialize)]
struct TrainedTolerance {
    scale: String,
    declared: f64,
    code_status: String,
    support: Vec<String>,
}

/// A head mask: bit `c` set keeps component `c` on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeadMask(u32);

/// The finite family an oracle's evidence is stated over.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct HeadMaskFamily {
    components: usize,
    /// The induction rows the divergence is taken over.
    rows: usize,
}

/// A support's code: its subset codeword, the padded head length `H` once, and `H` bits per kept head.
struct HeadCode {
    head_bits: u64,
}

impl CardinalityCode for HeadCode {
    type Error = CodecError;

    fn support_bits(&self, components: usize, size: usize) -> Result<u64, CodecError> {
        Ok(subset_code_len_bits(components, size)? + prefix_integer_len_bits(self.head_bits)? + size as u64 * self.head_bits)
    }
}

/// A layer on `weights` (query, key, value and output in torch `Linear` layout) whose four projections read
/// through the identity, `W = W·I`: the exact write through `R = I` is `W` itself, so the component coordinates
/// of the output projection are its input columns, and head `h`'s components are its `d_head` columns of `W_O`.
fn identity_layer(
    geometry: AttentionGeometry,
    score_scale: f64,
    weights: [Array2<f64>; 4],
) -> Result<ComponentAttentionLayer, String> {
    let identities = weights.each_ref().map(|weight| Array2::<f64>::eye(weight.ncols()));
    let reads = [0, 1, 2, 3].map(|index| ComponentRead {
        read: identities[index].view(),
        candidate_write: weights[index].view(),
    });
    factored_layer(geometry, score_scale, weights.each_ref().map(|weight| weight.clone()), reads)
}

/// The benchmark's network with the heads as components: every layer reads its output projection through the
/// identity, so a head mask is a mask on its `d_head` columns of `W_O`.
struct HeadNetwork {
    network: Network,
}

impl HeadNetwork {
    fn teacher(loaded: &LoadedExport) -> Result<Self, String> {
        let config = &loaded.export.config;
        if config.n_layers * config.n_heads >= u32::BITS as usize {
            return Err(format!(
                "{} heads do not fit one {}-bit head mask",
                config.n_layers * config.n_heads,
                u32::BITS
            ));
        }
        let (geometry, score_scale) = layout(loaded)?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            layers.push(identity_layer(geometry, score_scale, stored_layer(loaded, layer)?)?);
        }
        Ok(Self {
            network: Network::new(loaded, layers)?,
        })
    }

    fn heads(&self) -> usize {
        self.network.heads
    }

    fn components(&self) -> usize {
        self.network.layers.len() * self.network.heads
    }

    /// The artifact that decodes every head's reals from their lattice codewords at `precision`, and each
    /// head's codeword length in bits.
    fn decoded(&self, precision: DeclaredPrecision) -> Result<(Self, Vec<u64>), String> {
        let (heads, head_dim) = (self.network.heads, self.network.head_dim);
        let mut layers = Vec::with_capacity(self.network.layers.len());
        let mut head_bits = Vec::with_capacity(self.components());
        for layer in &self.network.layers {
            let native = layer.native();
            let stored = PROJECTIONS.map(|projection| native.weight(projection));
            let [mut query, mut key, mut value, mut output] = stored.each_ref().map(|weight| weight.to_owned());
            for head in 0..heads {
                let block = head * head_dim..(head + 1) * head_dim;
                let mut reals: Vec<f64> = Vec::new();
                for matrix in &stored[..3] {
                    reals.extend(matrix.slice(s![block.clone(), ..]).iter());
                }
                reals.extend(stored[3].slice(s![.., block.clone()]).iter());
                let code = LatticeCode::encode(&reals, precision)?;
                head_bits.push(code.index_bits()?);
                let mut values = code.decode()?.into_iter();
                for matrix in [&mut query, &mut key, &mut value] {
                    matrix
                        .slice_mut(s![block.clone(), ..])
                        .iter_mut()
                        .zip(values.by_ref())
                        .for_each(|(entry, value)| *entry = value);
                }
                output
                    .slice_mut(s![.., block.clone()])
                    .iter_mut()
                    .zip(values.by_ref())
                    .for_each(|(entry, value)| *entry = value);
                if values.next().is_some() {
                    return Err(format!("head {head}: the decoded codeword holds more reals than the head"));
                }
            }
            layers.push(identity_layer(native.geometry(), native.score_scale(), [query, key, value, output])?);
        }
        Ok((
            Self {
                network: self.network.with_layers(layers),
            },
            head_bits,
        ))
    }

    /// The logits of one sequence at every position, `T × vocab`, with their radius: on the stored reads with
    /// no box, or with `W_O` read through its components under a box of head masks, whose vertex is one head
    /// mask. A free head's columns read `1/2 ± 1/2`, so the radii cover every mask of the box. With free heads,
    /// also each head's spread `max_t Σ |W_O[:, head]| (|z_t| + r_t)` over its layer's mixed rows `z`, the reach
    /// of its half-width, and zero for every head without them.
    fn logits(&self, tokens: &[i64], heads: Option<&MaskBox>) -> Result<(Rows, Vec<f64>), String> {
        let network = &self.network;
        let (count, head_dim) = (network.heads, network.head_dim);
        let width = count * head_dim;
        let controls: Vec<(Option<Array2<f64>>, Option<Array2<f64>>, bool)> = (0..network.layers.len())
            .map(|layer| {
                let side = |column: usize| heads.map(|heads| heads.sides()[layer * count + column / head_dim]);
                let center = heads.map(|_| {
                    Array2::from_shape_fn((1, width), |(_, column)| match side(column) {
                        Some(MaskSide::On) => 1.0,
                        Some(MaskSide::Free) => 0.5,
                        Some(MaskSide::Off) | None => 0.0,
                    })
                });
                let free_here = (0..width).any(|column| side(column) == Some(MaskSide::Free));
                let half_width = free_here.then(|| {
                    Array2::from_shape_fn((1, width), |(_, column)| {
                        if side(column) == Some(MaskSide::Free) { 0.5 } else { 0.0 }
                    })
                });
                (center, half_width, free_here)
            })
            .collect();
        let reads: Vec<AttentionLayerReads<'_>> = controls
            .iter()
            .map(|(center, half_width, _)| AttentionLayerReads {
                output: match center {
                    Some(center) => ProjectionRead::Components(ComponentMasks {
                        center: center.view(),
                        half_width: half_width.as_ref().map(|half_width| half_width.view()),
                    }),
                    None => ProjectionRead::Native,
                },
                ..AttentionLayerReads::native()
            })
            .collect();
        let (logits, executions) = network.run(tokens, &reads)?;
        let mut spreads = vec![0.0_f64; self.components()];
        for (layer, (execution, (_, _, free_here))) in executions.iter().zip(&controls).enumerate() {
            if !free_here {
                continue;
            }
            let output = network.layers[layer].native().weight(AttentionProjection::Output);
            let (mixed, mixed_radius) = (&execution.attention.mixed, &execution.attention.mixed_radius);
            for head in 0..count {
                let block = head * head_dim..(head + 1) * head_dim;
                spreads[layer * count + head] = (0..mixed.nrows())
                    .map(|row| {
                        block
                            .clone()
                            .map(|column| {
                                let column_mass: f64 = output.column(column).iter().map(|entry| entry.abs()).sum();
                                column_mass * (mixed[[row, column]].abs() + mixed_radius[[row, column]])
                            })
                            .sum::<f64>()
                    })
                    .fold(0.0_f64, f64::max);
            }
        }
        Ok((logits, spreads))
    }

    /// The logits of one sequence, `T × vocab`, with their radius, under a box of position-scoped head masks:
    /// `instances[i]` is head `head` of layer `layer` at its `positions`, under the box's control `i`, and every
    /// head at every other position stays on. A free instance's columns read `1/2 ± 1/2` at its positions. Also
    /// each free instance's spread `Σ_s Σ |W_O[:, head]| (|z_s| + r_s)` over its positions' mixed rows, and zero
    /// for the others.
    fn scoped_logits(&self, tokens: &[i64], instances: &[Instance], mask: &MaskBox) -> Result<(Rows, Vec<f64>), String> {
        let network = &self.network;
        let (count, head_dim) = (network.heads, network.head_dim);
        let layers = network.layers.len();
        let mut centers = vec![Array2::<f64>::ones((tokens.len(), count * head_dim)); layers];
        let mut half_widths = vec![Array2::<f64>::zeros((tokens.len(), count * head_dim)); layers];
        let mut free = vec![false; layers];
        for (instance, side) in instances.iter().zip(mask.sides()) {
            let (center, half_width) = match side {
                MaskSide::On => (1.0, 0.0),
                MaskSide::Off => (0.0, 0.0),
                MaskSide::Free => (0.5, 0.5),
            };
            let columns = instance.head * head_dim..(instance.head + 1) * head_dim;
            for &position in &instance.positions {
                centers[instance.layer].slice_mut(s![position, columns.clone()]).fill(center);
                half_widths[instance.layer].slice_mut(s![position, columns.clone()]).fill(half_width);
            }
            free[instance.layer] |= *side == MaskSide::Free;
        }
        let reads: Vec<AttentionLayerReads<'_>> = centers
            .iter()
            .zip(&half_widths)
            .zip(&free)
            .map(|((center, half_width), &free)| AttentionLayerReads {
                output: ProjectionRead::Components(ComponentMasks {
                    center: center.view(),
                    half_width: free.then(|| half_width.view()),
                }),
                ..AttentionLayerReads::native()
            })
            .collect();
        let (logits, executions) = network.run(tokens, &reads)?;
        let mut spreads = vec![0.0_f64; instances.len()];
        for ((spread, instance), side) in spreads.iter_mut().zip(instances).zip(mask.sides()) {
            if *side != MaskSide::Free {
                continue;
            }
            let output = network.layers[instance.layer].native().weight(AttentionProjection::Output);
            let attention = &executions[instance.layer].attention;
            *spread = (instance.head * head_dim..(instance.head + 1) * head_dim)
                .map(|column| {
                    let column_mass: f64 = output.column(column).iter().map(|entry| entry.abs()).sum();
                    let reach: f64 = instance
                        .positions
                        .iter()
                        .map(|&position| {
                            attention.mixed[[position, column]].abs() + attention.mixed_radius[[position, column]]
                        })
                        .sum();
                    column_mass * reach
                })
                .sum();
        }
        Ok((logits, spreads))
    }
}

/// Where an S3 component reads its head: the source position the mechanism predicts, every other position up to
/// the row, or the row itself (a last-layer head).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Source,
    Rest,
    Row,
}

/// One S3 component: head `head` of layer `layer` read at `positions` together.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Instance {
    layer: usize,
    head: usize,
    scope: Scope,
    positions: Vec<usize>,
}

impl Instance {
    fn name(&self) -> String {
        let at = match self.scope {
            Scope::Source => format!("src{}", self.positions[0]),
            Scope::Rest => "rest".to_string(),
            Scope::Row => format!("{}", self.positions[0]),
        };
        format!("L{}H{}@{at}", self.layer, self.head)
    }
}

/// The components of the induction row at `position`, whose mechanism reads its key at `source`: each head of a
/// layer below the last at the source alone and at every other position up to the row, and each head of the last
/// layer at the row alone. The last layer's write at another position reaches only that position's logits, and a
/// position after the row reaches no earlier one, so neither can move the row.
fn row_instances(layers: usize, heads: usize, position: usize, source: usize) -> Vec<Instance> {
    let mut instances = Vec::new();
    for layer in 0..layers {
        for head in 0..heads {
            if layer + 1 < layers {
                instances.push(Instance { layer, head, scope: Scope::Source, positions: vec![source] });
                let rest = (0..=position).filter(|&at| at != source).collect();
                instances.push(Instance { layer, head, scope: Scope::Rest, positions: rest });
            } else {
                instances.push(Instance { layer, head, scope: Scope::Row, positions: vec![position] });
            }
        }
    }
    instances
}

/// The finite family an S3 status is exhaustive over: one row's binary masks over its instances.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct ScopedFamily {
    sequence: usize,
    position: usize,
    instances: usize,
}

/// One induction row's decoded artifact under position-scoped head masks, as a box program: its divergence from
/// the teacher at that row, bounded over both logit boxes. A box runs once at its center with every free
/// instance at `1/2 ± 1/2`; free instances split widest first, by spread.
struct ScopedBoxes<'a, 'b> {
    artifact: &'a HeadNetwork,
    tokens: &'a [i64],
    reference: &'a Rows,
    family: ScopedFamily,
    instances: &'a [Instance],
    enclosed: &'b mut BTreeMap<MaskBox, BoxEnclosure<ScopedFamily>>,
}

impl BoxDivergence for ScopedBoxes<'_, '_> {
    type Domain = ScopedFamily;
    type Error = String;

    fn components(&self) -> usize {
        self.instances.len()
    }

    fn domain(&self) -> ScopedFamily {
        self.family
    }

    fn enclose(&mut self, mask: &MaskBox) -> Result<BoxEnclosure<ScopedFamily>, String> {
        if let Some(found) = self.enclosed.get(mask) {
            return Ok(found.clone());
        }
        let (moved, spreads) = self.artifact.scoped_logits(self.tokens, self.instances, mask)?;
        let row = row_divergence(self.reference, &moved, self.family.position)?;
        let domain = self.family;
        let evidence = match (mask.is_vertex(), row.resolved, row.upper) {
            (true, Some((value, error)), _) => {
                EvidenceStatus::exact(value, error, ExactBasis::Exhaustive { cardinality: 1 }, Some(mask.clone()), domain)
            }
            (false, resolved, Some(upper)) => {
                EvidenceStatus::uniform_bound(upper, resolved.map_or(0.0, |(_, error)| error), domain)
            }
            (vertex, _, _) => EvidenceStatus::unresolved(
                row.lower.unwrap_or(0.0),
                f64::INFINITY,
                Extremum::Supremum,
                vertex.then(|| mask.clone()),
                domain,
            ),
        }
        .map_err(|error| format!("{error:?}"))?;
        let mut split_order = mask.free();
        split_order.sort_by(|left, right| spreads[*right].total_cmp(&spreads[*left]));
        let enclosure = BoxEnclosure { evidence, split_order };
        self.enclosed.insert(mask.clone(), enclosure.clone());
        Ok(enclosure)
    }
}

/// Per-row divergences of the decoded artifact from the teacher, memoized per head mask.
struct Divergences<'a> {
    artifact: &'a HeadNetwork,
    tokens: &'a [Vec<i64>],
    induction: InductionRows,
}

impl<'a> Divergences<'a> {
    fn new(teacher: &HeadNetwork, artifact: &'a HeadNetwork, tokens: &'a [Vec<i64>]) -> Result<Self, String> {
        let induction = InductionRows::new(tokens, |sequence_tokens| Ok(teacher.logits(sequence_tokens, None)?.0))?;
        Ok(Self {
            artifact,
            tokens,
            induction,
        })
    }

    fn rows(&self) -> usize {
        self.induction.rows.len()
    }

    /// Every induction row's divergence at one head mask.
    fn rows_at(&mut self, mask: HeadMask) -> Result<Vec<RowDivergence>, String> {
        let vertex = head_vertex(mask, self.artifact.components())?;
        let (artifact, tokens) = (self.artifact, self.tokens);
        self.induction
            .at(&vertex, |sequence| Ok(artifact.logits(&tokens[sequence], Some(&vertex))?.0))
    }

    /// Every induction row's divergence bound over a box of head masks, from one execution of each sequence at
    /// the box's center, and each head's largest spread over the sequences.
    fn box_rows(&mut self, heads: &MaskBox) -> Result<(Vec<RowDivergence>, Vec<f64>), String> {
        let (artifact, tokens) = (self.artifact, self.tokens);
        let (rows, by_sequence) = self.induction.evaluate_with(|sequence| artifact.logits(&tokens[sequence], Some(heads)))?;
        let mut spreads = vec![0.0_f64; artifact.components()];
        for spread in by_sequence.values() {
            for (widest, &own) in spreads.iter_mut().zip(spread) {
                *widest = widest.max(own);
            }
        }
        Ok((rows, spreads))
    }
}

/// The vertex of one head mask: bit `c` set keeps head `c` on.
fn head_vertex(mask: HeadMask, components: usize) -> Result<MaskBox, String> {
    let on = (0..components).filter(|&component| mask.0 >> component & 1 == 1).collect();
    Ok(MaskBox::vertex(&ComponentSet::new(components, on).map_err(|error| error.to_string())?))
}

/// The head mask of a vertex: its heads kept on.
fn vertex_mask(vertex: &MaskBox) -> HeadMask {
    HeadMask(
        vertex
            .sides()
            .iter()
            .enumerate()
            .filter(|(_, side)| **side == MaskSide::On)
            .fold(0u32, |bits, (component, _)| bits | 1 << component),
    )
}

/// The separation oracle over head masks, for every induction row (`row: None`) or one row.
struct HeadOracle<'a, 'b> {
    divergences: &'b mut Divergences<'a>,
    row: Option<usize>,
}

impl HeadOracle<'_, '_> {
    fn domain(&self) -> HeadMaskFamily {
        HeadMaskFamily {
            components: self.divergences.artifact.components(),
            rows: if self.row.is_some() { 1 } else { self.divergences.rows() },
        }
    }

    fn divergence(&mut self, mask: HeadMask) -> Result<Largest, String> {
        let values = self.divergences.rows_at(mask)?;
        Ok(InductionRows::largest(match self.row {
            Some(row) => &values[row..=row],
            None => &values[..],
        }))
    }

    fn evidence(
        &self,
        divergence: Largest,
        cardinality: u64,
        witness: HeadMask,
    ) -> Result<EvidenceStatus<HeadMask, HeadMaskFamily>, String> {
        match divergence {
            Largest::Resolved { value, error } => EvidenceStatus::exact(
                value,
                error,
                ExactBasis::Exhaustive { cardinality },
                Some(witness),
                self.domain(),
            ),
            Largest::Unresolved { lower } => {
                EvidenceStatus::unresolved(lower, f64::INFINITY, Extremum::Supremum, Some(witness), self.domain())
            }
        }
        .map_err(|error| format!("{error:?}"))
    }
}

impl SeparationOracle for HeadOracle<'_, '_> {
    type Mask = HeadMask;
    type Domain = HeadMaskFamily;
    type Error = String;

    fn components(&self) -> usize {
        self.divergences.artifact.components()
    }

    fn perturbed_components(&self, mask: &HeadMask) -> Vec<usize> {
        (0..self.components()).filter(|&component| mask.0 >> component & 1 == 0).collect()
    }

    fn separate(&mut self, support: &ComponentSet) -> Result<EvidenceStatus<HeadMask, HeadMaskFamily>, String> {
        let kept = support.members().iter().fold(0u32, |bits, &component| bits | 1 << component);
        let free: Vec<usize> = support.complement().members().to_vec();
        let (mut value, mut error, mut lower, mut resolved) = (f64::NEG_INFINITY, 0.0_f64, 0.0_f64, true);
        let mut worst = HeadMask(kept);
        for choice in 0..1u64 << free.len() {
            let mask = free
                .iter()
                .enumerate()
                .fold(kept, |bits, (index, &component)| bits | (((choice >> index) & 1) as u32) << component);
            match self.divergence(HeadMask(mask))? {
                Largest::Resolved { value: center, error: own } => {
                    error = error.max(own);
                    lower = lower.max((center - own).next_down().max(0.0));
                    if center > value {
                        value = center;
                        worst = HeadMask(mask);
                    }
                }
                Largest::Unresolved { lower: own } => {
                    resolved = false;
                    if own >= lower {
                        lower = own;
                        worst = HeadMask(mask);
                    }
                }
            }
        }
        let divergence = if resolved { Largest::Resolved { value, error } } else { Largest::Unresolved { lower } };
        self.evidence(divergence, 1u64 << free.len(), worst)
    }

    fn evaluate(&mut self, mask: &HeadMask) -> Result<EvidenceStatus<HeadMask, HeadMaskFamily>, String> {
        let divergence = self.divergence(*mask)?;
        self.evidence(divergence, 1, *mask)
    }
}

/// The head network as a box program for `supports::BoxSeparationOracle`. A vertex is one head mask, read
/// from the memoized per-row divergences the exhaustive oracle reads. A box with free heads runs once at its
/// center, each free head's `W_O` columns at `1/2 ± 1/2`, so its logit radii cover every mask of the box, and
/// `kl_over_logit_boxes` bounds each induction row over them: the box's bound is the largest row's. Free heads
/// split widest first, by spread (`Network::logits`).
struct HeadBoxes<'a, 'b> {
    divergences: &'b mut Divergences<'a>,
    enclosed: &'b mut BTreeMap<MaskBox, BoxEnclosure<HeadMaskFamily>>,
}

impl BoxDivergence for HeadBoxes<'_, '_> {
    type Domain = HeadMaskFamily;
    type Error = String;

    fn components(&self) -> usize {
        self.divergences.artifact.components()
    }

    fn domain(&self) -> HeadMaskFamily {
        HeadMaskFamily {
            components: self.components(),
            rows: self.divergences.rows(),
        }
    }

    fn enclose(&mut self, heads: &MaskBox) -> Result<BoxEnclosure<HeadMaskFamily>, String> {
        if let Some(found) = self.enclosed.get(heads) {
            return Ok(found.clone());
        }
        let domain = self.domain();
        let enclosure = if heads.is_vertex() {
            let mut oracle = HeadOracle {
                divergences: &mut *self.divergences,
                row: None,
            };
            let evidence = match oracle.divergence(vertex_mask(heads))? {
                Largest::Resolved { value, error } => EvidenceStatus::exact(
                    value,
                    error,
                    ExactBasis::Exhaustive { cardinality: 1 },
                    Some(heads.clone()),
                    domain,
                ),
                Largest::Unresolved { lower } => {
                    EvidenceStatus::unresolved(lower, f64::INFINITY, Extremum::Supremum, Some(heads.clone()), domain)
                }
            }
            .map_err(|error| format!("{error:?}"))?;
            BoxEnclosure {
                evidence,
                split_order: Vec::new(),
            }
        } else {
            let (rows, spreads) = self.divergences.box_rows(heads)?;
            let upper = rows
                .iter()
                .map(|row| row.upper)
                .try_fold(0.0_f64, |largest, upper| upper.map(|upper| largest.max(upper)));
            let error = rows
                .iter()
                .filter_map(|row| row.resolved)
                .fold(0.0_f64, |largest, (_, own)| largest.max(own));
            let lower = rows.iter().filter_map(|row| row.lower).fold(0.0_f64, f64::max);
            let evidence = match upper {
                Some(upper) => EvidenceStatus::uniform_bound(upper, error, domain),
                None => EvidenceStatus::unresolved(lower, f64::INFINITY, Extremum::Supremum, None, domain),
            }
            .map_err(|error| format!("{error:?}"))?;
            let mut split_order = heads.free();
            split_order.sort_by(|left, right| spreads[*right].total_cmp(&spreads[*left]));
            BoxEnclosure { evidence, split_order }
        };
        self.enclosed.insert(heads.clone(), enclosure.clone());
        Ok(enclosure)
    }
}

/// A separation oracle that records every support it is asked about.
struct Recording<O> {
    oracle: O,
    queried: Vec<ComponentSet>,
}

impl<O: SeparationOracle> SeparationOracle for Recording<O> {
    type Mask = O::Mask;
    type Domain = O::Domain;
    type Error = O::Error;

    fn components(&self) -> usize {
        self.oracle.components()
    }

    fn perturbed_components(&self, mask: &O::Mask) -> Vec<usize> {
        self.oracle.perturbed_components(mask)
    }

    fn separate(&mut self, support: &ComponentSet) -> Result<EvidenceStatus<O::Mask, O::Domain>, O::Error> {
        self.queried.push(support.clone());
        self.oracle.separate(support)
    }

    fn evaluate(&mut self, mask: &O::Mask) -> Result<EvidenceStatus<O::Mask, O::Domain>, O::Error> {
        self.oracle.evaluate(mask)
    }
}

/// A search's minimum code: `exact`, `unresolved` or `all_on_violation`, with its bounds in bits.
fn code_bits<M, D, E: std::fmt::Debug>(
    found: &Result<SupportSearch<M, D>, SupportSearchError<E, CodecError>>,
) -> Result<(String, f64, f64), String> {
    match found {
        Ok(search) => match &search.code {
            EvidenceStatus::Exact { value, .. } => Ok(("exact".to_string(), *value, *value)),
            EvidenceStatus::Unresolved { lower, upper, .. } => Ok(("unresolved".to_string(), *lower, *upper)),
            other => Err(format!("an unexpected code status {other:?}")),
        },
        Err(SupportSearchError::AllOnViolation) => Ok(("all_on_violation".to_string(), f64::INFINITY, f64::INFINITY)),
        Err(error) => Err(format!("{error:?}")),
    }
}

/// The box oracle's positive control at one tolerance: its search's minimum code, and the supports either
/// search queried that the two oracles decide differently.
#[derive(Serialize)]
struct BoxControl {
    code_status: String,
    code_bits_lower: f64,
    code_bits_upper: f64,
    support: Vec<String>,
    separations: usize,
    edges: Vec<Vec<String>>,
    /// Supports either search queried, each separated by both oracles.
    queried: usize,
    disagreements: Vec<Vec<String>>,
    /// The same minimum code as the exhaustive search, and no disagreement.
    agrees: bool,
}

/// S2a's exhaustive oracle is the reference for `supports::BoxSeparationOracle` on the same program (#2951): the
/// box search must find the exhaustive search's minimum code, and every support either search queried must get
/// the same certify/refute decision from both oracles. A certified support may differ between the two searches
/// when two supports share the minimum code, so ties are reported, not compared.
fn box_control(
    divergences: &mut Divergences<'_>,
    enclosed: &mut BTreeMap<MaskBox, BoxEnclosure<HeadMaskFamily>>,
    code: &HeadCode,
    tolerance: f64,
    exhaustive: &Result<SupportSearch<HeadMask, HeadMaskFamily>, SupportSearchError<String, CodecError>>,
    exhaustive_queried: &[ComponentSet],
    heads: usize,
) -> Result<BoxControl, String> {
    let components = divergences.artifact.components();
    let (found, box_queried) = {
        let program = HeadBoxes {
            divergences: &mut *divergences,
            enclosed: &mut *enclosed,
        };
        let oracle = BoxSeparationOracle::new(program, tolerance).map_err(|error| format!("{error:?}"))?;
        let mut recording = Recording {
            oracle,
            queried: Vec::new(),
        };
        let found = minimum_code_support(&mut recording, code, tolerance, FailureHypergraph::new(components));
        (found, recording.queried)
    };
    let (code_status, code_bits_lower, code_bits_upper) = code_bits(&found)?;
    let reference = code_bits(exhaustive)?;
    let (support, separations, edges) = match &found {
        Ok(search) => (
            search.certified.as_ref().map_or_else(Vec::new, |certified| names(heads, certified.support.members())),
            search.separations,
            search.hypergraph.edges().iter().map(|edge| names(heads, edge.perturbed.members())).collect(),
        ),
        Err(_) => (Vec::new(), 0, Vec::new()),
    };
    let mut queried: Vec<ComponentSet> = Vec::new();
    for support in exhaustive_queried.iter().chain(box_queried.iter()) {
        if !queried.contains(support) {
            queried.push(support.clone());
        }
    }
    let mut disagreements = Vec::new();
    for support in &queried {
        let by_masks = HeadOracle {
            divergences: &mut *divergences,
            row: None,
        }
        .separate(support)?;
        let by_boxes = BoxSeparationOracle::new(
            HeadBoxes {
                divergences: &mut *divergences,
                enclosed: &mut *enclosed,
            },
            tolerance,
        )
        .map_err(|error| format!("{error:?}"))?
        .separate(support)
        .map_err(|error| format!("{error:?}"))?;
        let decision = |certifies: bool, refutes: bool| (certifies, refutes);
        if decision(by_masks.certifies_at_most(tolerance), by_masks.refutes_at_most(tolerance))
            != decision(by_boxes.certifies_at_most(tolerance), by_boxes.refutes_at_most(tolerance))
        {
            disagreements.push(names(heads, support.members()));
        }
    }
    let same_code = code_status == reference.0
        && code_bits_lower.to_bits() == reference.1.to_bits()
        && code_bits_upper.to_bits() == reference.2.to_bits();
    let agrees = same_code && disagreements.is_empty();
    Ok(BoxControl {
        code_status,
        code_bits_lower,
        code_bits_upper,
        support,
        separations,
        edges,
        queried: queried.len(),
        disagreements,
        agrees,
    })
}

/// A trained support ranked by risk among the supports of its size in this network.
#[derive(Serialize)]
struct TransplantReport {
    support: Vec<String>,
    size: usize,
    same_size: usize,
    risk_lower: Option<f64>,
    risk_upper: Option<f64>,
    /// Same-size supports whose risk's upper bound lies below this support's lower bound.
    certainly_lower_risk: usize,
    /// Same-size supports whose risk's lower bound lies below this support's upper bound.
    possibly_lower_risk: usize,
    top_share: f64,
    /// `⌈q·n⌉`: how many same-size supports must certainly beat it to place it outside the top share.
    needed: usize,
    /// `outside_top_share`, `not_shown_outside_top_share`, or `undiscriminating` when no rival of its size exists.
    status: String,
    asserted: bool,
}

/// Ranks `support` by its risk `R(S)` among every support of its size, from the exhaustive oracle.
fn transplant_rank(
    divergences: &mut Divergences<'_>,
    heads: usize,
    support: &ComponentSet,
    top_share: f64,
    asserted: bool,
) -> Result<TransplantReport, String> {
    let components = support.components();
    let size = support.len();
    let mut risk = |candidate: &ComponentSet| -> Result<(Option<f64>, Option<f64>), String> {
        let evidence = HeadOracle {
            divergences: &mut *divergences,
            row: None,
        }
        .separate(candidate)?;
        Ok((evidence.lower_bound(), evidence.upper_bound()))
    };
    let (lower, upper) = risk(support)?;
    let (mut same_size, mut certainly, mut possibly) = (0usize, 0usize, 0usize);
    for bits in 0..1u32 << components {
        if bits.count_ones() as usize != size {
            continue;
        }
        same_size += 1;
        let members = (0..components).filter(|&component| bits >> component & 1 == 1).collect();
        let candidate = ComponentSet::new(components, members).map_err(|error| error.to_string())?;
        if candidate == *support {
            continue;
        }
        let (their_lower, their_upper) = risk(&candidate)?;
        if let (Some(theirs), Some(ours)) = (their_upper, lower)
            && theirs < ours
        {
            certainly += 1;
        }
        if match (their_lower, upper) {
            (Some(theirs), Some(ours)) => theirs < ours,
            (_, None) => true,
            (None, Some(_)) => false,
        } {
            possibly += 1;
        }
    }
    let needed = (top_share * same_size as f64).ceil() as usize;
    let status = if needed >= same_size {
        "undiscriminating"
    } else if certainly >= needed {
        "outside_top_share"
    } else {
        "not_shown_outside_top_share"
    };
    Ok(TransplantReport {
        support: names(heads, support.members()),
        size,
        same_size,
        risk_lower: lower,
        risk_upper: upper,
        certainly_lower_risk: certainly,
        possibly_lower_risk: possibly,
        top_share,
        needed,
        status: status.to_string(),
        asserted,
    })
}

/// The run's transplant verdict over its asserted transplants (see [`Report`]).
fn transplant_verdict(declared: bool, reports: &[ToleranceReport]) -> String {
    if !declared {
        return "none".to_string();
    }
    let asserted: Vec<&TransplantReport> = reports
        .iter()
        .filter_map(|report| report.transplant.as_ref())
        .filter(|ranked| ranked.asserted && ranked.status != "undiscriminating")
        .collect();
    if asserted.is_empty() {
        "undiscriminating".to_string()
    } else if asserted.iter().all(|ranked| ranked.status == "outside_top_share") {
        "pass".to_string()
    } else {
        "fail".to_string()
    }
}

/// A head name `L{layer}H{head}` as its component index.
fn component_of(heads: usize, name: &str) -> Result<usize, String> {
    let parsed = name.strip_prefix('L').and_then(|rest| rest.split_once('H')).and_then(|(layer, head)| {
        Some((layer.parse::<usize>().ok()?, head.parse::<usize>().ok()?))
    });
    match parsed {
        Some((layer, head)) if head < heads => Ok(layer * heads + head),
        _ => Err(format!("{name:?} is not a head name L<layer>H<head> with head below {heads}")),
    }
}

#[derive(Serialize)]
struct Measured {
    largest: f64,
    sequence: usize,
    witness: (usize, usize),
}

/// A divergence over the family: the extremum and its numerical error, or only a lower side, with the heads
/// the worst mask keeps on.
#[derive(Serialize)]
struct Risk {
    value: Option<f64>,
    numerical_error: Option<f64>,
    lower: Option<f64>,
    worst_mask_on: Vec<String>,
    family: Option<HeadMaskFamily>,
}

#[derive(Serialize)]
struct PerRowReport {
    rows: usize,
    /// Rows whose own search did not close; they are left out of the union.
    unresolved_rows: usize,
    mean_support: f64,
    size_counts: Vec<usize>,
    union: Vec<String>,
    library_bits: u64,
    packet_bits: u64,
    total_bits: u64,
}

#[derive(Serialize)]
struct ToleranceReport {
    /// `relative` (a declared fraction of the all-off divergence) or `absolute`.
    scale: String,
    /// The declared fraction or absolute tolerance.
    declared: f64,
    tolerance: f64,
    /// The certified support is empty: the do-nothing baseline meets the tolerance, by construction.
    vacuous: bool,
    transplant: Option<TransplantReport>,
    /// `exact` when the search closed, `unresolved` when it did not, and `all_on_violation` when the artifact
    /// misses the tolerance with every head on.
    code_status: String,
    code_bits_lower: f64,
    code_bits_upper: f64,
    support: Vec<String>,
    support_risk: Option<Risk>,
    separations: usize,
    edges: Vec<Vec<String>>,
    per_row: Option<PerRowReport>,
    box_control: BoxControl,
}

#[derive(Serialize)]
struct Report {
    checkpoint: u64,
    teacher_fingerprint: String,
    components: Vec<String>,
    sequences: usize,
    induction_rows: usize,
    fraction_bits: i32,
    head_bits: Vec<u64>,
    padded_head_bits: u64,
    teacher_vs_torch: Measured,
    baseline: Risk,
    artifact_all_on: Risk,
    masks_evaluated: usize,
    /// The boxes and vertices the box oracle's program enclosed, each once.
    box_enclosures: usize,
    /// `pass` when every asserted, discriminating transplant is shown outside the top share, `fail` when one is
    /// not, `undiscriminating` when none can discriminate, and `none` without a transplant.
    transplant_verdict: String,
    tolerances: Vec<ToleranceReport>,
    /// Stage S3, when declared.
    positions: Option<Vec<PositionRowReport>>,
}

fn names(heads: usize, members: &[usize]) -> Vec<String> {
    members.iter().map(|component| format!("L{}H{}", component / heads, component % heads)).collect()
}

fn risk(heads: usize, evidence: &EvidenceStatus<HeadMask, HeadMaskFamily>) -> Risk {
    let on = |mask: &Option<HeadMask>| -> Vec<String> {
        mask.map(|mask| {
            let members: Vec<usize> = (0..u32::BITS as usize).filter(|&c| mask.0 >> c & 1 == 1).collect();
            names(heads, &members)
        })
        .unwrap_or_default()
    };
    match evidence {
        EvidenceStatus::Exact {
            value,
            numerical_error,
            witness,
            domain,
            ..
        } => Risk {
            value: Some(*value),
            numerical_error: Some(*numerical_error),
            lower: evidence.lower_bound(),
            worst_mask_on: on(witness),
            family: Some(*domain),
        },
        EvidenceStatus::Unresolved { lower, witness, domain, .. } => Risk {
            value: None,
            numerical_error: None,
            lower: Some(*lower),
            worst_mask_on: on(witness),
            family: Some(*domain),
        },
        _ => Risk {
            value: None,
            numerical_error: None,
            lower: evidence.lower_bound(),
            worst_mask_on: Vec::new(),
            family: None,
        },
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(USAGE.to_string());
    }
    let export_dir = PathBuf::from(flag(&args, "--export", USAGE)?);
    let settings_path = PathBuf::from(flag(&args, "--settings", USAGE)?);
    let report_path = PathBuf::from(flag(&args, "--out", USAGE)?);
    let text = std::fs::read_to_string(&settings_path)
        .map_err(|error| format!("read {}: {error}", settings_path.display()))?;
    let settings: Settings =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", settings_path.display()))?;
    if settings.baseline_fractions.is_empty()
        || settings
            .baseline_fractions
            .iter()
            .chain(&settings.absolute_tolerances)
            .any(|value| !(value.is_finite() && *value >= 0.0))
    {
        return Err(
            "the settings must declare one or more baseline fractions, and every fraction and absolute tolerance \
             must be finite and non-negative"
                .to_string(),
        );
    }
    if let Some(positions) = &settings.positions
        && (positions.rows.is_empty()
            || positions.baseline_fractions.is_empty()
            || positions.baseline_fractions.iter().any(|value| !(value.is_finite() && *value >= 0.0)))
    {
        return Err("S3's positions must declare rows and finite non-negative baseline fractions".to_string());
    }
    if let Some(transplant) = &settings.transplant
        && !(transplant.top_share > 0.0 && transplant.top_share <= 1.0)
    {
        return Err(format!("the transplant's top share {} must lie in (0, 1]", transplant.top_share));
    }
    let precision = DeclaredPrecision::new(settings.fraction_bits)?;

    let loaded = load_export(&export_dir)?;
    let export = &loaded.export;
    let config = &export.config;
    let teacher = HeadNetwork::teacher(&loaded)?;
    let (artifact, head_bits) = teacher.decoded(precision)?;
    let padded_head_bits = head_bits.iter().copied().max().ok_or("the network has no heads")?;
    let code = HeadCode {
        head_bits: padded_head_bits,
    };
    let components = teacher.components();
    let heads = teacher.heads();

    let torch_logits = float64_array(
        &export.files,
        &export_dir,
        "native_logits",
        (export.sequences * config.seq_len, config.vocab),
    )?;
    let mut divergences = Divergences::new(&teacher, &artifact, &export.tokens)?;
    let (largest, sequence, witness) =
        induction_network::teacher_vs_torch(&divergences.induction.reference, &torch_logits, config.seq_len);
    let teacher_vs_torch = Measured {
        largest,
        sequence,
        witness,
    };
    let rows = divergences.rows();
    println!(
        "[load] checkpoint={} teacher_fingerprint={:#018x} components={components} sequences={} induction_rows={rows} fraction_bits={} head_bits={head_bits:?} padded={padded_head_bits} teacher_vs_torch={:.3e}",
        export.checkpoint,
        loaded.registry.teacher_fingerprint().0,
        export.sequences,
        settings.fraction_bits,
        teacher_vs_torch.largest
    );

    let (baseline, artifact_all_on) = {
        let mut oracle = HeadOracle {
            divergences: &mut divergences,
            row: None,
        };
        let off = oracle.evaluate(&HeadMask(0))?;
        let on = oracle.evaluate(&HeadMask((1u32 << components) - 1))?;
        (risk(heads, &off), risk(heads, &on))
    };
    println!(
        "[baseline] every head off: divergence={:?} ± {:?}; artifact with every head on: divergence={:?} ± {:?}",
        baseline.value, baseline.numerical_error, artifact_all_on.value, artifact_all_on.numerical_error
    );

    let all_off = baseline
        .value
        .ok_or("the all-off divergence is unresolved, so no tolerance can be a fraction of it")?;
    let trained = match &settings.transplant {
        Some(transplant) => {
            let text = std::fs::read_to_string(&transplant.report)
                .map_err(|error| format!("read {}: {error}", transplant.report.display()))?;
            Some(
                serde_json::from_str::<TrainedReport>(&text)
                    .map_err(|error| format!("{}: {error}", transplant.report.display()))?,
            )
        }
        None => None,
    };
    let entries: Vec<(&str, f64, f64)> = settings
        .baseline_fractions
        .iter()
        .map(|&fraction| ("relative", fraction, fraction * all_off))
        .chain(settings.absolute_tolerances.iter().map(|&tolerance| ("absolute", tolerance, tolerance)))
        .collect();
    let mut reports = Vec::with_capacity(entries.len());
    let mut box_enclosures = BTreeMap::new();
    let mut box_failures = Vec::new();
    for &(scale, declared, tolerance) in &entries {
        // A relative tolerance carries the trained network's certified support at the same fraction.
        let transplant = match (&settings.transplant, &trained, scale) {
            (Some(transplant), Some(trained), "relative") => {
                let found = trained.tolerances.iter().find(|entry| {
                    entry.scale == "relative" && entry.declared.to_bits() == declared.to_bits() && entry.code_status == "exact"
                });
                match found {
                    Some(entry) => {
                        let members =
                            entry.support.iter().map(|name| component_of(heads, name)).collect::<Result<Vec<_>, _>>()?;
                        let support = ComponentSet::new(components, members).map_err(|error| error.to_string())?;
                        Some(transplant_rank(&mut divergences, heads, &support, transplant.top_share, transplant.asserted)?)
                    }
                    None => None,
                }
            }
            _ => None,
        };
        if let Some(ranked) = &transplant {
            println!(
                "[transplant] fraction={declared:.3e} support={:?} risk=[{:?}, {:?}] same_size={} certainly_lower={} possibly_lower={} needed={} status={}",
                ranked.support,
                ranked.risk_lower,
                ranked.risk_upper,
                ranked.same_size,
                ranked.certainly_lower_risk,
                ranked.possibly_lower_risk,
                ranked.needed,
                ranked.status
            );
        }
        let (search, exhaustive_queried) = {
            let mut recording = Recording {
                oracle: HeadOracle {
                    divergences: &mut divergences,
                    row: None,
                },
                queried: Vec::new(),
            };
            let found = minimum_code_support(&mut recording, &code, tolerance, FailureHypergraph::new(components));
            (found, recording.queried)
        };
        let control = box_control(
            &mut divergences,
            &mut box_enclosures,
            &code,
            tolerance,
            &search,
            &exhaustive_queried,
            heads,
        )?;
        println!(
            "[box-control] tolerance={tolerance:.3e} code={} bits=[{}, {}] support={:?} separations={} queried={} disagreements={} agrees={}",
            control.code_status,
            control.code_bits_lower,
            control.code_bits_upper,
            control.support,
            control.separations,
            control.queried,
            control.disagreements.len(),
            control.agrees
        );
        if !control.agrees {
            box_failures.push(tolerance);
        }
        let search = match search {
            Ok(search) => search,
            Err(SupportSearchError::AllOnViolation) => {
                println!("[supports] tolerance={tolerance:.3e} code=all_on_violation: the artifact misses it with every head on");
                reports.push(ToleranceReport {
                    scale: scale.to_string(),
                    declared,
                    tolerance,
                    vacuous: false,
                    transplant,
                    code_status: "all_on_violation".to_string(),
                    code_bits_lower: f64::INFINITY,
                    code_bits_upper: f64::INFINITY,
                    support: Vec::new(),
                    support_risk: None,
                    separations: 0,
                    edges: Vec::new(),
                    per_row: None,
                    box_control: control,
                });
                continue;
            }
            Err(error) => return Err(format!("tolerance {tolerance}: {error:?}")),
        };
        let (code_status, code_bits_lower, code_bits_upper) = match &search.code {
            EvidenceStatus::Exact { value, .. } => ("exact", *value, *value),
            EvidenceStatus::Unresolved { lower, upper, .. } => ("unresolved", *lower, *upper),
            other => return Err(format!("tolerance {tolerance}: an unexpected code status {other:?}")),
        };
        let (support, support_risk) = match &search.certified {
            Some(found) => (names(heads, found.support.members()), Some(risk(heads, &found.evidence))),
            None => (Vec::new(), None),
        };
        let edges = search
            .hypergraph
            .edges()
            .iter()
            .map(|edge| names(heads, edge.perturbed.members()))
            .collect();

        // P18: each induction row's own minimum-code support at the same tolerance.
        let mut per_input = Vec::new();
        let mut per_row_supports = Vec::new();
        let mut unresolved_rows = 0;
        let mut size_counts = vec![0usize; components + 1];
        for row in 0..rows {
            let mut oracle = HeadOracle {
                divergences: &mut divergences,
                row: Some(row),
            };
            let found = minimum_code_support(&mut oracle, &code, tolerance, FailureHypergraph::new(components));
            match found {
                Ok(found) => match (found.code, found.certified) {
                    (EvidenceStatus::Exact { .. }, Some(certified)) => {
                        size_counts[certified.support.len()] += 1;
                        per_row_supports.push(certified.support.members().to_vec());
                        per_input.push(InputSupport {
                            support: certified.support,
                            tolerance,
                            evidence: certified.evidence,
                        });
                    }
                    _ => unresolved_rows += 1,
                },
                Err(SupportSearchError::AllOnViolation) => unresolved_rows += 1,
                Err(error) => return Err(format!("tolerance {tolerance} row {row}: {error:?}")),
            }
        }
        let union = sufficient_union(components, &per_input).map_err(|error| format!("tolerance {tolerance}: {error:?}"))?;
        let library = union_support_library(&per_row_supports).map_err(|error| format!("{error:?}"))?;
        let library_bits = code
            .support_bits(components, library.components.len())
            .map_err(|error| format!("{error:?}"))?;
        let packet_bits: u64 = encode_support_packets(library.components.len(), &library.supports)
            .map_err(|error| format!("{error:?}"))?
            .iter()
            .map(|packet| packet.len_bits())
            .sum();
        let resolved = per_row_supports.len();
        let mean_support = if resolved == 0 { 0.0 } else { library.summed_support as f64 / resolved as f64 };
        println!(
            "[supports] tolerance={tolerance:.3e} code={code_status} bits=[{code_bits_lower}, {code_bits_upper}] support={support:?} risk={:?} separations={} edges={}",
            support_risk.as_ref().and_then(|risk| risk.value),
            search.separations,
            search.hypergraph.edges().len()
        );
        println!(
            "[supports] tolerance={tolerance:.3e} per_row rows={rows} unresolved={unresolved_rows} mean_support={mean_support:.3} sizes={size_counts:?} union={:?} library_bits={library_bits} packet_bits={packet_bits}",
            names(heads, union.support.members())
        );
        let vacuous = code_status == "exact" && search.certified.as_ref().is_some_and(|found| found.support.is_empty());
        println!(
            "[scale] {scale} declared={declared:.3e} tolerance={tolerance:.3e} vacuous={vacuous}{}",
            if vacuous { ": the do-nothing baseline meets the tolerance by construction" } else { "" }
        );
        reports.push(ToleranceReport {
            scale: scale.to_string(),
            declared,
            tolerance,
            vacuous,
            transplant,
            code_status: code_status.to_string(),
            code_bits_lower,
            code_bits_upper,
            support,
            support_risk,
            separations: search.separations,
            edges,
            per_row: Some(PerRowReport {
                rows,
                unresolved_rows,
                mean_support,
                size_counts,
                union: names(heads, union.support.members()),
                library_bits,
                packet_bits,
                total_bits: library_bits + packet_bits,
            }),
            box_control: control,
        });
    }

    // Stage S3: position-scoped supports at the declared induction rows.
    let positions = match &settings.positions {
        Some(declared) => {
            let mut rng = StdRng::seed_from_u64(declared.null_seed);
            let mut scored = Vec::with_capacity(declared.rows.len());
            for &(sequence, position) in &declared.rows {
                if !divergences.induction.rows.contains(&(sequence, position)) {
                    return Err(format!("({sequence}, {position}) is not an induction row"));
                }
                // The rows of a sequence are `T − n … T − 2`, so there are `n − 1` of them.
                let segment = divergences.induction.rows.iter().filter(|(row_sequence, _)| *row_sequence == sequence).count() + 1;
                let row = position_row(
                    &artifact,
                    &export.tokens[sequence],
                    &divergences.induction.reference[sequence],
                    (sequence, position),
                    segment,
                    &declared.baseline_fractions,
                    padded_head_bits,
                    &mut rng,
                )?;
                for tolerance in &row.tolerances {
                    println!(
                        "[positions] row=({sequence}, {position}) n={segment} source={} instances={} fraction={:.3e} tolerance={:.3e} code={} support={:?} at_source={} elsewhere={} heads={} scoped_bits={} global_instances={} alone={} moved={:?} moved_verdict={}",
                        row.source,
                        row.instances,
                        tolerance.fraction,
                        tolerance.tolerance,
                        tolerance.code_status,
                        tolerance.support,
                        tolerance.at_source,
                        tolerance.elsewhere,
                        tolerance.distinct_heads,
                        tolerance.scoped_code_bits,
                        tolerance.global_instances,
                        tolerance.alone_verdict,
                        tolerance.moved,
                        tolerance.moved_verdict
                    );
                }
                scored.push(row);
            }
            Some(scored)
        }
        None => None,
    };

    let report = Report {
        checkpoint: export.checkpoint,
        teacher_fingerprint: format!("{:#018x}", loaded.registry.teacher_fingerprint().0),
        components: names(heads, &(0..components).collect::<Vec<_>>()),
        sequences: export.sequences,
        induction_rows: rows,
        fraction_bits: settings.fraction_bits,
        head_bits,
        padded_head_bits,
        teacher_vs_torch,
        baseline,
        artifact_all_on,
        masks_evaluated: divergences.induction.evaluated(),
        box_enclosures: box_enclosures.len(),
        transplant_verdict: transplant_verdict(settings.transplant.is_some(), &reports),
        tolerances: reports,
        positions,
    };
    let text = serde_json::to_string_pretty(&report).map_err(|error| format!("report: {error}"))?;
    std::fs::write(&report_path, text).map_err(|error| format!("write {}: {error}", report_path.display()))?;
    println!(
        "[done] masks_evaluated={} box_enclosures={} report={}",
        report.masks_evaluated,
        report.box_enclosures,
        report_path.display()
    );
    if !box_failures.is_empty() {
        return Err(format!(
            "the box oracle's positive control failed at tolerances {box_failures:?}: see box_control in the report"
        ));
    }
    Ok(())
}

/// A status's proven sides.
#[derive(Serialize)]
struct Bounds {
    lower: Option<f64>,
    upper: Option<f64>,
}

impl Bounds {
    fn of<W, D>(status: &EvidenceStatus<W, D>) -> Self {
        Self {
            lower: status.lower_bound(),
            upper: status.upper_bound(),
        }
    }

    /// `meets` when the upper side is at most `tolerance`, `violates` when the lower side exceeds it.
    fn verdict(&self, tolerance: f64) -> &'static str {
        match (self.lower, self.upper) {
            (_, Some(upper)) if upper <= tolerance => "meets",
            (Some(lower), _) if lower > tolerance => "violates",
            _ => "unresolved",
        }
    }
}

/// S3 at one row and one tolerance.
#[derive(Serialize)]
struct PositionTolerance {
    fraction: f64,
    tolerance: f64,
    vacuous: bool,
    /// `exact`, `unresolved`, or `all_on_violation` when the artifact misses the tolerance with every instance on.
    code_status: String,
    code_bits_lower: f64,
    code_bits_upper: f64,
    support: Vec<String>,
    separations: usize,
    edges: usize,
    /// The support's lower-layer components at the source `t − n + 1` the mechanism reads, and at the rest.
    at_source: usize,
    elsewhere: usize,
    /// The support's distinct heads, its code with each distinct head's codeword sent once, and the components
    /// the same heads keep when each is kept at every position (a lower-layer head's source and rest).
    distinct_heads: usize,
    scoped_code_bits: u64,
    global_instances: usize,
    /// The support alone (every other head and position off), and the same support with each kept source
    /// component read at a drawn other position instead (declared seed), at equal count.
    alone: Option<Bounds>,
    alone_verdict: String,
    moved: Vec<String>,
    moved_risk: Option<Bounds>,
    moved_verdict: String,
}

#[derive(Serialize)]
struct PositionRowReport {
    sequence: usize,
    position: usize,
    segment: usize,
    source: usize,
    instances: usize,
    baseline: Bounds,
    all_on: Bounds,
    enclosures: usize,
    tolerances: Vec<PositionTolerance>,
}

/// Stage S3 at one declared induction row `(sequence, position)` of a sequence whose segment has length
/// `segment`: at each declared fraction `f` of the row's own all-off divergence, the minimum-code support over
/// the row's instances (`row_instances`) by `BoxSeparationOracle`. The code is the padded head code over the
/// instances, `L(C, k) + L_int(H) + k·H`, a function of the size alone, so the minimum-code support is a
/// minimum-count one; the report also gives the code that sends each distinct kept head once.
fn position_row(
    artifact: &HeadNetwork,
    tokens: &[i64],
    reference: &Rows,
    (sequence, position): (usize, usize),
    segment: usize,
    fractions: &[f64],
    head_bits: u64,
    rng: &mut StdRng,
) -> Result<PositionRowReport, String> {
    let layers = artifact.network.layers.len();
    let source = position + 1 - segment;
    let instances = row_instances(layers, artifact.heads(), position, source);
    let count = instances.len();
    let family = ScopedFamily { sequence, position, instances: count };
    let vertex = |on: Vec<usize>| {
        ComponentSet::new(count, on).map(|on| MaskBox::vertex(&on)).map_err(|error| error.to_string())
    };
    let mut enclosed = BTreeMap::new();
    let (baseline, all_on) = {
        let mut program = ScopedBoxes { artifact, tokens, reference, family, instances: &instances, enclosed: &mut enclosed };
        (program.enclose(&vertex(Vec::new())?)?.evidence, program.enclose(&vertex((0..count).collect())?)?.evidence)
    };
    let &EvidenceStatus::Exact { value: do_nothing, .. } = &baseline else {
        return Err(format!("row ({sequence}, {position}): the all-off divergence is unresolved"));
    };
    let code = HeadCode { head_bits };
    let mut tolerances = Vec::with_capacity(fractions.len());
    for &fraction in fractions {
        let tolerance = fraction * do_nothing;
        let program = ScopedBoxes { artifact, tokens, reference, family, instances: &instances, enclosed: &mut enclosed };
        let oracle = BoxSeparationOracle::new(program, tolerance).map_err(|error| format!("{error:?}"))?;
        let mut oracle = Logged::new(oracle, format!("row=({sequence}, {position}) fraction={fraction:.3e}"));
        let found = minimum_code_support(&mut oracle, &code, tolerance, FailureHypergraph::new(count));
        let (code_status, code_bits_lower, code_bits_upper) = code_bits(&found)?;
        let (members, separations, edges) = match &found {
            Ok(search) => (
                search.certified.as_ref().map(|found| found.support.members().to_vec()),
                search.separations,
                search.hypergraph.edges().len(),
            ),
            Err(_) => (None, 0, 0),
        };
        let support = members.clone().unwrap_or_default();
        let at_source = support.iter().filter(|&&member| instances[member].scope == Scope::Source).count();
        let elsewhere = support.iter().filter(|&&member| instances[member].scope == Scope::Rest).count();
        let mut heads: Vec<(usize, usize)> = support.iter().map(|&member| (instances[member].layer, instances[member].head)).collect();
        heads.sort_unstable();
        heads.dedup();
        let scoped_code_bits = subset_code_len_bits(count, support.len()).map_err(|error| format!("{error:?}"))?
            + prefix_integer_len_bits(head_bits).map_err(|error| format!("{error:?}"))?
            + heads.len() as u64 * head_bits;
        let global_instances = heads.iter().map(|&(layer, _)| if layer + 1 < layers { 2 } else { 1 }).sum();
        let alone = match &members {
            Some(members) => Some(oracle.evaluate(&vertex(members.clone())?).map_err(|error| format!("{error:?}"))?),
            None => None,
        };
        // The moved null: every kept source component read at a drawn other position instead, the kept rest and
        // row components as they are, and every other head and position off.
        let (moved, moved_risk) = match &members {
            Some(members) if at_source > 0 && position > 0 => {
                let mut moved = instances.clone();
                for &member in members {
                    if instances[member].scope != Scope::Source {
                        continue;
                    }
                    let drawn = rng.random_range(0..position);
                    let at = if drawn >= source { drawn + 1 } else { drawn };
                    let rest = moved
                        .iter()
                        .position(|other| other.layer == instances[member].layer && other.head == instances[member].head && other.scope == Scope::Rest)
                        .ok_or_else(|| format!("no rest component for {}", instances[member].name()))?;
                    moved[member].positions = vec![at];
                    moved[rest].positions = (0..=position).filter(|&other| other != at).collect();
                }
                let mut null_enclosed = BTreeMap::new();
                let mut program = ScopedBoxes { artifact, tokens, reference, family, instances: &moved, enclosed: &mut null_enclosed };
                let status = program.enclose(&vertex(members.clone())?)?.evidence;
                let names = members.iter().map(|&member| moved[member].name()).collect();
                (names, Some(Bounds::of(&status)))
            }
            _ => (Vec::new(), None),
        };
        let alone = alone.as_ref().map(Bounds::of);
        let verdict = |bounds: &Option<Bounds>| bounds.as_ref().map_or("none", |bounds| bounds.verdict(tolerance)).to_string();
        tolerances.push(PositionTolerance {
            fraction,
            tolerance,
            vacuous: code_status == "exact" && members.as_ref().is_some_and(Vec::is_empty),
            code_status,
            code_bits_lower,
            code_bits_upper,
            support: support.iter().map(|&member| instances[member].name()).collect(),
            separations,
            edges,
            at_source,
            elsewhere,
            distinct_heads: heads.len(),
            scoped_code_bits,
            global_instances,
            alone_verdict: verdict(&alone),
            alone,
            moved,
            moved_verdict: verdict(&moved_risk),
            moved_risk,
        });
    }
    Ok(PositionRowReport {
        sequence,
        position,
        segment,
        source,
        instances: count,
        baseline: Bounds::of(&baseline),
        all_on: Bounds::of(&all_on),
        enclosures: enclosed.len(),
        tolerances,
    })
}
