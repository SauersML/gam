//! mpd-induction's attention-only network on `block`'s layer route, as the induction support examples run it
//! (#2951). Like `npy_header.rs` and `induction_export.rs`, this is not an example target: it is compiled only where
//! it is `#[path]`-included, beside `induction_export.rs`.
//!
//! The network is `x₀ = W_E[token] + W_pos` (one IEEE addition, `γ₁|x̂₀|`), then each layer as a
//! `block::ComponentAttentionLayer` fed the previous residual rows with their radius, then `block::linear_read` for
//! `W_U`. Every stage's radius bounds its distance from the exact program's value at the exact input, so the logits'
//! radius bounds the whole network's, and `bounds::kl_over_logit_boxes` bounds each row's divergence over both
//! logit boxes.

use crate::induction_export::LoadedExport;
use gam_sae::parameter_decomposition::attention::{AttentionGeometry, ProjectedRows, RotaryEmbedding, RotaryPairing};
use gam_sae::parameter_decomposition::block::{
    AttentionLayerExecution, AttentionLayerReads, AttentionProjection, ComponentAttentionLayer, NativeAttentionLayer,
    linear_read,
};
use gam_sae::parameter_decomposition::bounds::kl_over_logit_boxes;
use gam_sae::parameter_decomposition::receipts::evaluation_band;
use gam_sae::parameter_decomposition::rewrite::ComponentRead;
use gam_sae::parameter_decomposition::supports::{EvidenceStatus, MaskBox};
use ndarray::Array2;
use std::collections::BTreeMap;

/// The four linear reads of a layer, in the order every per-projection array here uses.
pub const PROJECTIONS: [AttentionProjection; 4] = [
    AttentionProjection::Query,
    AttentionProjection::Key,
    AttentionProjection::Value,
    AttentionProjection::Output,
];

/// Computed rows and a per-entry bound on their distance from the exact program's rows.
pub struct Rows {
    pub values: Array2<f64>,
    pub radius: Array2<f64>,
}

impl Rows {
    pub fn projected(&self) -> ProjectedRows<'_> {
        ProjectedRows {
            values: self.values.view(),
            radius: self.radius.view(),
        }
    }
}

/// The benchmark's layer geometry and score scale. Torch divides the scores by `math.sqrt(d_head)` and the native
/// core multiplies them by its reciprocal. Both round nothing, so they are one program, only when `√d_head` is a
/// power of two; any other `d_head` is refused.
pub fn layout(loaded: &LoadedExport) -> Result<(AttentionGeometry, f64), String> {
    let config = &loaded.export.config;
    let root = (config.d_head as f64).sqrt();
    let whole = root as u64;
    if whole as f64 != root || !whole.is_power_of_two() {
        return Err(format!(
            "d_head {}: √d_head is not a power of two, so torch's division and the native multiplication round differently",
            config.d_head
        ));
    }
    let geometry = AttentionGeometry {
        model_dim: config.d_model,
        n_heads: config.n_heads,
        n_kv_heads: config.n_heads,
        head_dim: config.d_head,
    };
    Ok((geometry, 1.0 / root))
}

/// Layer `layer`'s stored query, key, value and output weights in torch `Linear` layout.
pub fn stored_layer(loaded: &LoadedExport, layer: usize) -> Result<[Array2<f64>; 4], String> {
    let [query, key, value, output] = ["W_Q", "W_K", "W_V", "W_O"].map(|name| loaded.weight(&format!("{name}.{layer}")));
    Ok([query?, key?, value?, output?])
}

/// A layer on `weights` whose four projections execute through the declared exact factors `reads`, in
/// [`PROJECTIONS`] order. Learned absolute positions: the source rotates no plane.
pub fn factored_layer(
    geometry: AttentionGeometry,
    score_scale: f64,
    weights: [Array2<f64>; 4],
    reads: [ComponentRead<'_>; 4],
) -> Result<ComponentAttentionLayer, String> {
    let [query, key, value, output] = weights;
    let native = NativeAttentionLayer::new(
        geometry,
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: Vec::new(),
            attention_scaling: 1.0,
        },
        score_scale,
        query,
        key,
        value,
        output,
    )
    .map_err(|error| error.to_string())?;
    let [query, key, value, output] = reads;
    ComponentAttentionLayer::new(native, query, key, value, output).map_err(|error| error.to_string())
}

/// The benchmark's network on its layers.
pub struct Network {
    pub embedding: Array2<f64>,
    pub positional: Array2<f64>,
    pub layers: Vec<ComponentAttentionLayer>,
    pub unembedding: Array2<f64>,
    pub heads: usize,
    pub head_dim: usize,
    pub positions: Vec<i64>,
}

impl Network {
    /// The export's embedding, positional and unembedding tensors around `layers`.
    pub fn new(loaded: &LoadedExport, layers: Vec<ComponentAttentionLayer>) -> Result<Self, String> {
        let config = &loaded.export.config;
        Ok(Self {
            embedding: loaded.weight("W_E")?,
            positional: loaded.weight("W_pos")?,
            layers,
            unembedding: loaded.weight("W_U")?,
            heads: config.n_heads,
            head_dim: config.d_head,
            positions: (0..config.seq_len as i64).collect(),
        })
    }

    /// The same embedding, positional and unembedding tensors around other layers.
    pub fn with_layers(&self, layers: Vec<ComponentAttentionLayer>) -> Self {
        Self {
            embedding: self.embedding.clone(),
            positional: self.positional.clone(),
            layers,
            unembedding: self.unembedding.clone(),
            heads: self.heads,
            head_dim: self.head_dim,
            positions: self.positions.clone(),
        }
    }

    /// The logits of one sequence at every position, `T × vocab`, with their radius, each layer under its own
    /// reads, and every layer's execution.
    pub fn run(
        &self,
        tokens: &[i64],
        reads: &[AttentionLayerReads<'_>],
    ) -> Result<(Rows, Vec<AttentionLayerExecution>), String> {
        if reads.len() != self.layers.len() {
            return Err(format!("{} layer reads for {} layers", reads.len(), self.layers.len()));
        }
        if let Some(&token) = tokens
            .iter()
            .find(|&&token| usize::try_from(token).map_or(true, |index| index >= self.embedding.nrows()))
        {
            return Err(format!("token {token} is outside the vocabulary of {}", self.embedding.nrows()));
        }
        let values = Array2::from_shape_fn((tokens.len(), self.embedding.ncols()), |(position, column)| {
            self.embedding[[tokens[position] as usize, column]] + self.positional[[position, column]]
        });
        let radius = values.mapv(|value| evaluation_band(1, value.abs()));
        let mut residual = Rows { values, radius };
        let mut executions = Vec::with_capacity(self.layers.len());
        for (index, (layer, &reads)) in self.layers.iter().zip(reads).enumerate() {
            let execution = layer
                .execute(reads, residual.projected(), &self.positions)
                .map_err(|error| format!("layer {index}: {error}"))?;
            residual = Rows {
                values: execution.output.clone(),
                radius: execution.output_radius.clone(),
            };
            executions.push(execution);
        }
        let (values, radius) =
            linear_read(self.unembedding.view(), residual.projected()).map_err(|error| format!("unembedding: {error}"))?;
        Ok((
            Rows {
                values: (*values).to_owned(),
                radius,
            },
            executions,
        ))
    }
}

/// The induction rows `T − n, …, T − 2` of one sequence, read off its tokens. The prefix and the segment are
/// distinct tokens, so the last token occurs exactly once earlier, at `T − 1 − n`; the repeat is checked.
pub fn induction_rows(sequence: usize, tokens: &[i64]) -> Result<Vec<usize>, String> {
    let last = tokens.len() - 1;
    let earlier: Vec<usize> = (0..last).filter(|&position| tokens[position] == tokens[last]).collect();
    let [source] = earlier[..] else {
        return Err(format!(
            "sequence {sequence}: its last token occurs {} times earlier, not once",
            earlier.len()
        ));
    };
    let segment = last - source;
    if 2 * segment > tokens.len()
        || tokens[tokens.len() - segment..] != tokens[tokens.len() - 2 * segment..tokens.len() - segment]
    {
        return Err(format!("sequence {sequence}: the last {segment} tokens do not repeat the {segment} before them"));
    }
    Ok((tokens.len() - segment..last).collect())
}

/// One row's divergence `KL(reference ‖ moved)` over both logit boxes: its exact center and numerical error when
/// the bound resolves, and its proven bounds.
#[derive(Clone, Copy, Debug)]
pub struct RowDivergence {
    pub resolved: Option<(f64, f64)>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

/// Row `position`'s divergence of `moved` from `reference`, over both logit boxes.
pub fn row_divergence(reference: &Rows, moved: &Rows, position: usize) -> Result<RowDivergence, String> {
    let status = kl_over_logit_boxes(
        reference.values.row(position),
        reference.radius.row(position),
        moved.values.row(position),
        moved.radius.row(position),
    )
    .map_err(|error| error.to_string())?;
    let resolved = match &status {
        EvidenceStatus::Exact {
            value, numerical_error, ..
        } => Some((*value, *numerical_error)),
        _ => None,
    };
    Ok(RowDivergence {
        resolved,
        lower: status.lower_bound(),
        upper: status.upper_bound(),
    })
}

/// A divergence over some rows: the largest center with the largest numerical error when every row resolved,
/// otherwise the largest proven lower side.
#[derive(Clone, Copy, Debug)]
pub enum Largest {
    Resolved { value: f64, error: f64 },
    Unresolved { lower: f64 },
}

/// The induction rows of the declared sequences, the teacher's all-on logits of each, and every evaluated mask's
/// per-row divergences, memoized by mask.
pub struct InductionRows {
    /// `(sequence, position)` of each induction row.
    pub rows: Vec<(usize, usize)>,
    pub reference: Vec<Rows>,
    cache: BTreeMap<MaskBox, Vec<RowDivergence>>,
}

impl InductionRows {
    /// The induction rows of `tokens`, against `reference(sequence)`, the teacher's logits of each sequence.
    pub fn new(
        tokens: &[Vec<i64>],
        mut reference: impl FnMut(&[i64]) -> Result<Rows, String>,
    ) -> Result<Self, String> {
        let mut rows = Vec::new();
        let mut logits = Vec::with_capacity(tokens.len());
        for (sequence, sequence_tokens) in tokens.iter().enumerate() {
            rows.extend(induction_rows(sequence, sequence_tokens)?.into_iter().map(|position| (sequence, position)));
            logits.push(reference(sequence_tokens)?);
        }
        Ok(Self {
            rows,
            reference: logits,
            cache: BTreeMap::new(),
        })
    }

    /// The masks evaluated so far.
    pub fn evaluated(&self) -> usize {
        self.cache.len()
    }

    /// Every induction row's divergence under `mask`, executing each sequence once through `execute(sequence)`,
    /// memoized for a vertex. A box is evaluated afresh each time: its enclosure is its caller's to keep.
    pub fn at(
        &mut self,
        mask: &MaskBox,
        execute: impl FnMut(usize) -> Result<Rows, String>,
    ) -> Result<Vec<RowDivergence>, String> {
        if let Some(found) = self.cache.get(mask) {
            return Ok(found.clone());
        }
        let divergences = self.evaluate(execute)?;
        if mask.is_vertex() {
            self.cache.insert(mask.clone(), divergences.clone());
        }
        Ok(divergences)
    }

    /// Every induction row's divergence of the network `execute(sequence)` runs, with no memo: for a network
    /// outside the masks `at` keys, such as another decomposition.
    pub fn evaluate(&self, mut execute: impl FnMut(usize) -> Result<Rows, String>) -> Result<Vec<RowDivergence>, String> {
        let mut executed: BTreeMap<usize, Rows> = BTreeMap::new();
        let mut divergences = Vec::with_capacity(self.rows.len());
        for &(sequence, position) in &self.rows {
            if !executed.contains_key(&sequence) {
                executed.insert(sequence, execute(sequence)?);
            }
            divergences.push(
                row_divergence(&self.reference[sequence], &executed[&sequence], position)
                    .map_err(|error| format!("row ({sequence}, {position}): {error}"))?,
            );
        }
        Ok(divergences)
    }

    /// The largest divergence over `rows`: exact, as the largest center with the largest numerical error, when every
    /// row resolved, and otherwise unresolved at the largest lower side.
    pub fn largest(rows: &[RowDivergence]) -> Largest {
        let (mut value, mut error, mut lower, mut resolved) = (0.0_f64, 0.0_f64, 0.0_f64, true);
        for row in rows {
            match row.resolved {
                Some((center, own)) => {
                    value = value.max(center);
                    error = error.max(own);
                    lower = lower.max((center - own).next_down().max(0.0));
                }
                None => {
                    resolved = false;
                    lower = lower.max(row.lower.unwrap_or(0.0));
                }
            }
        }
        if resolved { Largest::Resolved { value, error } } else { Largest::Unresolved { lower } }
    }
}

/// The largest absolute difference between the teacher's native logits and torch's exported ones, with its
/// sequence and `(position, column)`: a measured control, never part of any status.
pub fn teacher_vs_torch(reference: &[Rows], torch: &Array2<f64>, seq_len: usize) -> (f64, usize, (usize, usize)) {
    let mut largest = (0.0_f64, 0, (0, 0));
    for (sequence, logits) in reference.iter().enumerate() {
        for ((position, column), &native) in logits.values.indexed_iter() {
            let difference = (torch[[sequence * seq_len + position, column]] - native).abs();
            if difference > largest.0 {
                largest = (difference, sequence, (position, column));
            }
        }
    }
    largest
}

/// The value of `--name` in `args`.
pub fn flag<'a>(args: &'a [String], name: &str, usage: &str) -> Result<&'a str, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
        .ok_or_else(|| format!("missing {name}; {usage}"))
}
