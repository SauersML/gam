//! Executed transport of a circle atom between two layers, measured against the
//! geometric nearest-point correspondence (#2946).
//!
//! # Two correspondences
//!
//! [`measure_atom_transport_between`] carries a source chart coordinate `t` to
//! the target layer by nearest point, `t'(t) = E_{ℓ+1}(D_ℓ(t))`: decode the
//! source image, then project it onto the target image. No network enters, so
//! two layers that decode the same circle report the identity whatever the block
//! between them does. The executed transport runs the block:
//!
//! `τ_ℓ(t; c) = E_{ℓ+1}(T_ℓ(c + D_ℓ(t)) − T_ℓ(c))`.
//!
//! - `D_ℓ(t) = Φ(t)ᵀB^(ℓ)` is the atom's honest source decoder.
//! - `T_ℓ` is the executed transition from the layer-`ℓ` state to the
//!   layer-`ℓ+1` state.
//! - `c` is the other state: the patched row without this atom's decoded
//!   contribution.
//! - `E_{ℓ+1}` is the same continuous projection onto the target image that the
//!   nearest-point law uses.
//!
//! Subtracting the executed baseline `T_ℓ(c)` removes the transported other
//! state. Under the identity transition the readback is `D_ℓ(t)` and `τ` is
//! exactly the nearest-point correspondence: the geometric law is the executed
//! law of a block that does nothing.
//!
//! # The declared law of the other state
//!
//! `c` is not estimated here. It is whatever the patched row carries besides the
//! atom. The caller declares how rows are drawn, and the design is checked
//! against the declaration:
//! - [`OtherStateLaw::Held`]: every sample patches the one row, so `τ(·; c)` is
//!   one chart map and its classification is conditional on that `c`.
//! - [`OtherStateLaw::Resampled`]: every sample patches its own row, so the pairs
//!   `(t_k, τ(t_k; c_k))` are independent across samples and the classification
//!   averages over the rows' law.
//!
//! The class posteriors of [`classify_circle_transport`] treat the pairs as
//! independent draws, so the caller draws the source coordinates independently
//! from the coordinate law it declares (for a circle, uniform on `[0, 1)`).
//!
//! # The errors, reported apart
//!
//! `E_{ℓ+1}` returns a coordinate for any readback, however far the readback lies
//! from the target image. A rigid-looking chart map can therefore hide a readback
//! that left the atom, and a resultant cannot tell a smooth warp from a map with
//! no structure. The report keeps three parts in separate fields:
//! - **Encoder projection error.** Per sample, `‖y − D_{ℓ+1}(τ)‖²`: the part of the
//!   executed readback `y` that the projection discards. The report publishes its
//!   mean next to the target image's own scale
//!   `r² = ∫₀¹ ‖D_{ℓ+1}(t) − B^(ℓ+1)_0‖² dt = ½ Σ_{k≥1} ‖B^(ℓ+1)_k‖²`. The harmonic
//!   basis rows are orthogonal on `[0, 1)` with mean square ½, and row 0 decodes
//!   the image's mean point.
//! - **Rigid transition error.** The departure of the executed chart map from a
//!   rigid O(2) map: [`classify_circle_transport`]'s defect `1 − max(R₊, R₋)`, its
//!   standard error, and the Shift/Reflect/Mixing posteriors.
//! - **Non-rigid part.** The REML smooth map `h` of the executed pairs
//!   ([`fit_transport_map`]): its winding degree, its fold certificate, its
//!   isometry defect `mean((|h′| − 1)²)`, and the scatter of the pairs about `h`.
//!   Both a warp and mixing lower the resultant. A warp is a fold-free cover with
//!   a non-zero isometry defect and no scatter about `h`. Mixing leaves the pairs
//!   scattered about any smooth map.
//!
//! # The patch contract
//!
//! The transition is a caller closure `(n, splices) → readbacks`. The block can
//! be a Rust function or the real model, run by thin Python through the
//! `x_row + Δx` splice of `gamfit/torch/interventions.py`. For patched row `n`:
//! - **Patched state.** The layer-`ℓ` residual row `h_n` at row `n`'s token
//!   position. Its own source coordinate is `t̂_n`, so `h_n = c_n + D_ℓ(t̂_n)`.
//!   Each splice replaces that one row by `h_n + Δx`. No other position or layer
//!   is patched.
//! - **Splices.** Rust computes every `Δx`; Python only splices. Row 0 is the
//!   baseline `Δx = −D_ℓ(t̂_n)`, which leaves the other state `c_n`. Row `k ≥ 1`
//!   is `Δx = D_ℓ(t_k) − D_ℓ(t̂_n)`, which places the atom at `t_k`.
//! - **Read back.** For every splice, the layer-`ℓ+1` residual row at the same
//!   token position, in the units of the target layer's honest decoder, as the
//!   same row of the returned matrix. Rust forms `y_k = readback_k − readback_0`
//!   and encodes it. What the patch moves at later positions is not read.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::chart_coordinate_solve::{ChartBasisKind, PeriodicCurveExtrema};
use crate::inference::layer_transport::{ChartTopology, FittedTransport, fit_transport_map};
use crate::inference::transport_class::{CircleTransportReport, classify_circle_transport};
use crate::manifold::{
    AtomTransportReport, AtomTransportStatus, CrosscoderLayer, CrosscoderLayout, SaeManifoldTerm,
    honest_layer_decoder, measure_atom_transport_between,
};

/// The declared law of the other state `c`: how the rows a transition patches
/// are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OtherStateLaw {
    /// Every sample patches the same row, so `c` is held fixed.
    Held,
    /// Every sample patches its own row, so `c` is resampled per sample.
    Resampled,
}

/// The rows a transition patches and the source coordinates spliced into them.
#[derive(Clone, Debug)]
pub struct ExecutedTransportDesign {
    /// The declared law of the other state, checked against `samples`.
    pub other_state: OtherStateLaw,
    /// `t̂_n`: the source chart coordinate of patched row `n`.
    pub row_coordinates: Vec<f64>,
    /// `(n, t)`: place the atom at source coordinate `t` in row `n`.
    pub samples: Vec<(usize, f64)>,
}

impl ExecutedTransportDesign {
    /// The sample indices of each patched row, once the design matches its
    /// declared law: `Held` patches exactly one row, and `Resampled` patches every
    /// row exactly once.
    fn samples_by_row(&self) -> Result<Vec<Vec<usize>>, ExecutedTransportError> {
        let invalid = |message: String| Err(ExecutedTransportError::InvalidDesign(message));
        let rows = self.row_coordinates.len();
        if self.samples.is_empty() {
            return invalid("the design has no sample".to_string());
        }
        if let Some((row, coordinate)) = self
            .row_coordinates
            .iter()
            .enumerate()
            .find(|(_, coordinate)| !coordinate.is_finite())
        {
            return invalid(format!("row {row} has a non-finite own coordinate {coordinate}"));
        }
        let mut by_row = vec![Vec::new(); rows];
        for (index, &(row, coordinate)) in self.samples.iter().enumerate() {
            if row >= rows {
                return invalid(format!(
                    "sample {index} patches row {row}, but the design has {rows} rows"
                ));
            }
            if !coordinate.is_finite() {
                return invalid(format!("sample {index} has a non-finite coordinate {coordinate}"));
            }
            by_row[row].push(index);
        }
        match self.other_state {
            OtherStateLaw::Held if rows != 1 => invalid(format!(
                "a held other state patches one row, but the design has {rows} rows"
            )),
            OtherStateLaw::Held => Ok(by_row),
            OtherStateLaw::Resampled => match by_row.iter().position(|members| members.len() != 1) {
                Some(row) => invalid(format!(
                    "a resampled other state patches every row exactly once, but row {row} is \
                     patched {} times",
                    by_row[row].len()
                )),
                None => Ok(by_row),
            },
        }
    }
}

/// One executed transport sample.
#[derive(Clone, Copy, Debug)]
pub struct ExecutedTransportSample {
    /// The patched row.
    pub row: usize,
    /// The source chart coordinate `t` the atom was placed at.
    pub coordinate: f64,
    /// The executed transport `τ(t; c)`.
    pub executed: f64,
    /// The nearest-point correspondence `E_{ℓ+1}(D_ℓ(t))` at the same `t`.
    pub nearest_point: f64,
    /// The encoder projection error `‖y − D_{ℓ+1}(τ)‖²` of the executed readback.
    pub projection_residual: f64,
}

/// The executed transport of one circle atom, next to the nearest-point law
/// between the same two layers.
#[derive(Clone, Debug)]
pub struct ExecutedTransportReport {
    /// The declared law of the other state the samples were drawn under.
    pub other_state: OtherStateLaw,
    /// The geometric correspondence between the same layers.
    pub nearest_point: AtomTransportReport,
    /// The executed samples, in the design's sample order.
    pub samples: Vec<ExecutedTransportSample>,
    /// The rigid transition error: the O(2) classification of the pairs
    /// `(2πt, 2πτ(t; c))`.
    pub transition_law: CircleTransportReport,
    /// The non-rigid part: the REML smooth map of the same pairs, with its winding
    /// degree, fold certificate, isometry defect and residual scatter.
    pub smooth_map: FittedTransport,
    /// The encoder projection error: the mean of the samples'
    /// `projection_residual`.
    pub mean_projection_residual: f64,
    /// The target image's scale `r² = ½ Σ_{k≥1} ‖B^(ℓ+1)_k‖²`, the mean squared
    /// distance of its points from its mean point.
    pub target_radius_sq: f64,
}

/// Whether the executed transport measured one atom between two layers.
#[derive(Clone, Debug)]
pub enum ExecutedTransportStatus {
    Measured(ExecutedTransportReport),
    /// The nearest-point law does not describe this atom or layer pair, for the
    /// reason it gives.
    Undefined { reason: String },
}

/// Why an executed transport measurement refused.
#[derive(Clone, Debug)]
pub enum ExecutedTransportError {
    /// The design contradicts its declared law, or carries a non-finite or
    /// out-of-range entry.
    InvalidDesign(String),
    /// The caller's transition failed for patched row `row`, or returned a
    /// readback of the wrong shape or with a non-finite entry.
    Transition { row: usize, message: String },
    /// A measurement stage refused: the nearest-point law, the target projection,
    /// the classifier or the smooth map.
    Measurement(String),
}

impl std::fmt::Display for ExecutedTransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDesign(message) => {
                write!(formatter, "executed transport: invalid design: {message}")
            }
            Self::Transition { row, message } => {
                write!(formatter, "executed transport: transition of row {row}: {message}")
            }
            Self::Measurement(message) => write!(formatter, "executed transport: {message}"),
        }
    }
}

impl std::error::Error for ExecutedTransportError {}

/// Measure the executed transport of one circle atom between two crosscoder
/// layers, next to the nearest-point law between the same layers.
///
/// `transition(n, splices)` executes the splices of patched row `n` and returns
/// one layer-`ℓ+1` readback row per splice (module documentation, "The patch
/// contract"). An error it returns comes back as
/// [`ExecutedTransportError::Transition`] for that row.
pub fn measure_executed_atom_transport(
    term: &SaeManifoldTerm,
    layout: &CrosscoderLayout,
    atom: usize,
    source: CrosscoderLayer,
    target: CrosscoderLayer,
    design: &ExecutedTransportDesign,
    mut transition: impl FnMut(usize, ArrayView2<'_, f64>) -> Result<Array2<f64>, String>,
) -> Result<ExecutedTransportStatus, ExecutedTransportError> {
    let samples_by_row = design.samples_by_row()?;
    let nearest_point = match measure_atom_transport_between(term, layout, atom, source, target)
        .map_err(ExecutedTransportError::Measurement)?
    {
        AtomTransportStatus::Measured(report) => report,
        AtomTransportStatus::Undefined { reason } => {
            return Ok(ExecutedTransportStatus::Undefined { reason });
        }
    };

    // The honest decoders and the target projection the nearest-point law used.
    let physical_decoder = term.tier0_unscaled_full_width_decoder(atom);
    let b_src = honest_layer_decoder(&physical_decoder, layout, source)
        .map_err(ExecutedTransportError::Measurement)?;
    let b_tgt = honest_layer_decoder(&physical_decoder, layout, target)
        .map_err(ExecutedTransportError::Measurement)?;
    let basis = ChartBasisKind::Periodic {
        n_harmonics: nearest_point.n_harmonics,
    };
    let target_extrema = PeriodicCurveExtrema::from_gram(b_tgt.dot(&b_tgt.t()).view())
        .map_err(ExecutedTransportError::Measurement)?;

    let mut samples = vec![None; design.samples.len()];
    for (row, members) in samples_by_row.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let own = decode(basis, &b_src, design.row_coordinates[row]);
        let mut splices = Array2::<f64>::zeros((members.len() + 1, b_src.ncols()));
        splices.row_mut(0).assign(&own.mapv(|value| -value));
        for (splice, &index) in members.iter().enumerate() {
            let placed = decode(basis, &b_src, design.samples[index].1) - &own;
            splices.row_mut(splice + 1).assign(&placed);
        }
        let readback = transition(row, splices.view())
            .map_err(|message| ExecutedTransportError::Transition { row, message })?;
        if readback.dim() != splices.dim() {
            return Err(ExecutedTransportError::Transition {
                row,
                message: format!(
                    "read back a {:?} matrix for {:?} splices",
                    readback.dim(),
                    splices.dim()
                ),
            });
        }
        if let Some(value) = readback.iter().find(|value| !value.is_finite()) {
            return Err(ExecutedTransportError::Transition {
                row,
                message: format!("read back a non-finite entry {value}"),
            });
        }
        for (splice, &index) in members.iter().enumerate() {
            let coordinate = design.samples[index].1;
            let executed_state = &readback.row(splice + 1) - &readback.row(0);
            let (executed, projection_residual) =
                encode(basis, &b_tgt, &target_extrema, executed_state.view()).map_err(|error| {
                    ExecutedTransportError::Measurement(format!(
                        "projection of row {row}, sample {index}: {error}"
                    ))
                })?;
            let (nearest, _) = encode(
                basis,
                &b_tgt,
                &target_extrema,
                decode(basis, &b_src, coordinate).view(),
            )
            .map_err(|error| {
                ExecutedTransportError::Measurement(format!(
                    "nearest point of sample {index}: {error}"
                ))
            })?;
            samples[index] = Some(ExecutedTransportSample {
                row,
                coordinate,
                executed,
                nearest_point: nearest,
                projection_residual,
            });
        }
    }
    let samples = samples
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| ExecutedTransportError::Measurement("a sample was not executed".to_string()))?;

    let tau = std::f64::consts::TAU;
    let theta_in: Vec<f64> = samples.iter().map(|sample| tau * sample.coordinate).collect();
    let theta_out: Vec<f64> = samples.iter().map(|sample| tau * sample.executed).collect();
    let transition_law = classify_circle_transport(
        &theta_in,
        &theta_out,
        chain_position(source),
        chain_position(target),
    )
    .map_err(ExecutedTransportError::Measurement)?;
    let smooth_map = fit_transport_map(
        ArrayView1::from(theta_in.as_slice()),
        ArrayView1::from(theta_out.as_slice()),
        ChartTopology::Circle,
        ChartTopology::Circle,
    )
    .map_err(|error| {
        ExecutedTransportError::Measurement(format!("smooth map of the executed pairs: {error}"))
    })?;
    let mean_projection_residual = samples
        .iter()
        .map(|sample| sample.projection_residual)
        .sum::<f64>()
        / samples.len() as f64;
    let target_radius_sq = 0.5 * b_tgt.slice(s![1.., ..]).iter().map(|&v| v * v).sum::<f64>();

    Ok(ExecutedTransportStatus::Measured(ExecutedTransportReport {
        other_state: design.other_state,
        nearest_point,
        samples,
        transition_law,
        smooth_map,
        mean_projection_residual,
        target_radius_sq,
    }))
}

/// One decoded point `D(t) = Φ(t)ᵀB` of an `M × p` decoder.
fn decode(basis: ChartBasisKind, decoder: &Array2<f64>, t: f64) -> Array1<f64> {
    let mut phi = vec![0.0; basis.width()];
    basis.eval_into(t, &mut phi);
    decoder.t().dot(&ArrayView1::from(phi.as_slice()))
}

/// `E(y)`: the target coordinate nearest `y`, and the squared distance
/// `‖y − D(E(y))‖²` the projection discards.
fn encode(
    basis: ChartBasisKind,
    target: &Array2<f64>,
    extrema: &PeriodicCurveExtrema,
    state: ArrayView1<'_, f64>,
) -> Result<(f64, f64), String> {
    let linear = target.dot(&state);
    let linear = linear
        .as_slice()
        .ok_or_else(|| "target linear coefficients are not contiguous".to_string())?;
    let coordinate = extrema.minimize_squared_distance(linear)?.coordinate;
    let discarded = &state - &decode(basis, target, coordinate);
    Ok((coordinate, discarded.dot(&discarded)))
}

/// The layer's position in the crosscoder chain: the anchor is `0` and block `ℓ`
/// is `ℓ + 1`.
fn chain_position(layer: CrosscoderLayer) -> usize {
    match layer {
        CrosscoderLayer::Anchor => 0,
        CrosscoderLayer::Block(block) => block + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::transport_class::CircleTransportClass;
    use crate::manifold::{
        AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
        SaeBasisEvaluator, SaeManifoldAtom,
    };
    use std::f64::consts::TAU;
    use std::sync::Arc;

    const N_ROWS: usize = 32;
    const BLOCK_LOG_LAMBDA: f64 = 1.4;
    /// Each layer's width: the circle plane in columns 0 and 1, and one context
    /// direction in column 2.
    const LAYER_WIDTH: usize = 3;
    /// The planted rotation, in turns.
    const DELTA: f64 = 0.137_035_999_084;
    /// How far a transported coordinate may sit from its closed form: the accuracy
    /// the nearest-point law's own controls pin (`tests_transport_law_2234.rs`).
    const COORDINATE_TOL: f64 = 1.0e-9;
    const SAMPLES: usize = 64;
    /// The scrambled-pairs control uses the sample size of steer2234's scrambled
    /// classifier control (`transport_class.rs`).
    const SCRAMBLED_SAMPLES: usize = 512;

    /// A coordinate error `Δ` moves a squared unit-circle quantity by at most
    /// `(2πΔ)²`. That covers a squared distance from a point at radius `ρ ≤ 1`,
    /// which rises by `2ρ(1 − cos 2πΔ)`, and a resultant's `1 − cos 2πΔ`. The O(1)
    /// sums add a few `ε`.
    fn squared_tol() -> f64 {
        (TAU * COORDINATE_TOL).powi(2) + 8.0 * f64::EPSILON
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Distance on the period-1 circle.
    fn circular_gap(a: f64, b: f64) -> f64 {
        let d = (a - b).rem_euclid(1.0);
        d.min(1.0 - d)
    }

    /// A two-layer crosscoder whose anchor and downstream block decode the SAME
    /// unit circle `q(t) = (sin 2πt, cos 2πt, 0)`. The block is stored in the
    /// weighted fit space `√λ·B`.
    fn same_circle_crosscoder() -> (SaeManifoldTerm, CrosscoderLayout) {
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).expect("order-1 periodic basis"));
        let coords =
            Array2::<f64>::from_shape_fn((N_ROWS, 1), |(row, _)| row as f64 / N_ROWS as f64);
        let (phi, jet) = evaluator
            .evaluate(coords.view())
            .expect("basis at the fixture coordinates");
        let sqrt_lambda = (0.5 * BLOCK_LOG_LAMBDA).exp();
        let mut decoder = Array2::<f64>::zeros((3, 2 * LAYER_WIDTH));
        decoder[[1, 0]] = 1.0;
        decoder[[2, 1]] = 1.0;
        decoder[[1, LAYER_WIDTH]] = sqrt_lambda;
        decoder[[2, LAYER_WIDTH + 1]] = sqrt_lambda;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "executed-transport-circle",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .expect("circle atom")
        .with_basis_evaluator(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((N_ROWS, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .expect("single-atom assignment");
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("crosscoder term");
        let layout = CrosscoderLayout::new(
            LAYER_WIDTH,
            vec![LAYER_WIDTH],
            vec!["downstream".into()],
            vec![BLOCK_LOG_LAMBDA],
        )
        .expect("two-layer layout");
        term.set_crosscoder_layout(layout.clone())
            .expect("install the layout");
        (term, layout)
    }

    /// The harvested layer-`ℓ` row `h = c + q(t̂)`, with other state `c` and own
    /// coordinate `t̂`.
    fn harvested_row(context: [f64; LAYER_WIDTH], own: f64) -> [f64; LAYER_WIDTH] {
        let (sin, cos) = (TAU * own).sin_cos();
        [context[0] + sin, context[1] + cos, context[2]]
    }

    /// `q(t) ↦ q(t + turns)` on the circle plane.
    fn rotate_plane(state: &mut [f64], turns: f64) {
        let (sin, cos) = (TAU * turns).sin_cos();
        let (x0, x1) = (state[0], state[1]);
        state[0] = x0 * cos + x1 * sin;
        state[1] = -x0 * sin + x1 * cos;
    }

    /// The toy block: rotate the circle plane by `δ + ε·tanh(h₂)` turns, read from
    /// the context column, and pass the context through.
    fn context_rotation(delta: f64, epsilon: f64) -> impl Fn(&[f64]) -> Vec<f64> {
        move |state: &[f64]| {
            let mut out = state.to_vec();
            rotate_plane(&mut out, delta + epsilon * state[2].tanh());
            out
        }
    }

    /// The transition for harvested `rows` under a toy `block`: every splice
    /// patches its row and runs the block on the patched state.
    fn executed(
        rows: Vec<[f64; LAYER_WIDTH]>,
        block: impl Fn(&[f64]) -> Vec<f64>,
    ) -> impl FnMut(usize, ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        move |row: usize, splices: ArrayView2<'_, f64>| {
            let harvested = rows
                .get(row)
                .ok_or_else(|| format!("no harvested row {row}"))?;
            let mut readback = Array2::<f64>::zeros(splices.dim());
            for (index, splice) in splices.outer_iter().enumerate() {
                let patched: Vec<f64> = harvested
                    .iter()
                    .zip(splice.iter())
                    .map(|(value, delta)| value + delta)
                    .collect();
                readback
                    .row_mut(index)
                    .assign(&ArrayView1::from(block(&patched).as_slice()));
            }
            Ok(readback)
        }
    }

    /// Measure anchor → block on the same-circle crosscoder.
    fn measure_anchor_to_block(
        design: &ExecutedTransportDesign,
        transition: impl FnMut(usize, ArrayView2<'_, f64>) -> Result<Array2<f64>, String>,
    ) -> Result<ExecutedTransportStatus, ExecutedTransportError> {
        let (term, layout) = same_circle_crosscoder();
        measure_executed_atom_transport(
            &term,
            &layout,
            0,
            CrosscoderLayer::Anchor,
            CrosscoderLayer::Block(0),
            design,
            transition,
        )
    }

    /// The measured report, or why the fixture was not measured.
    fn measured(
        status: Result<ExecutedTransportStatus, ExecutedTransportError>,
    ) -> Result<ExecutedTransportReport, String> {
        match status.map_err(|error| error.to_string())? {
            ExecutedTransportStatus::Measured(report) => Ok(report),
            ExecutedTransportStatus::Undefined { reason } => Err(format!(
                "a periodic circle atom must be measured, got undefined: {reason}"
            )),
        }
    }

    /// Both layers decode the same circle and the block rotates it by `δ`: the
    /// nearest-point law reports the identity and the executed transport reports
    /// `δ`.
    #[test]
    fn a_planted_rotation_is_executed_transport_where_nearest_point_reports_identity() {
        let mut seed = 11u64;
        // The other state has components in the circle plane, so τ = t + δ only
        // if the executed baseline removes the transported other state.
        let context = [0.4, -0.7, 0.3];
        let own = 0.61;
        let design = ExecutedTransportDesign {
            other_state: OtherStateLaw::Held,
            row_coordinates: vec![own],
            samples: (0..SAMPLES).map(|_| (0, lcg(&mut seed))).collect(),
        };
        let rotation = measured(measure_anchor_to_block(
            &design,
            executed(vec![harvested_row(context, own)], context_rotation(DELTA, 0.0)),
        ))
        .expect("planted rotation");

        let (sign, phase) = rotation.nearest_point.phase_shift;
        assert_eq!(sign, 1.0, "two copies of one circle: the nearest-point sign must be +1");
        assert!(
            circular_gap(phase, 0.0) <= COORDINATE_TOL,
            "two copies of one circle: the nearest-point law must be the identity, got phase {phase}"
        );
        assert!(
            1.0 - rotation.nearest_point.phase_r2 <= COORDINATE_TOL,
            "the nearest-point identity must be an exact law, got phase_r2 = {}",
            rotation.nearest_point.phase_r2
        );
        assert_eq!(rotation.other_state, OtherStateLaw::Held);
        assert_eq!(rotation.samples.len(), SAMPLES);
        for sample in &rotation.samples {
            assert!(
                circular_gap(sample.nearest_point, sample.coordinate) <= COORDINATE_TOL,
                "nearest point of t = {} must be t, got {}",
                sample.coordinate,
                sample.nearest_point
            );
            assert!(
                circular_gap(sample.executed, sample.coordinate + DELTA) <= COORDINATE_TOL,
                "executed transport of t = {} must be t + δ, got {}",
                sample.coordinate,
                sample.executed
            );
            // The identity that the nearest point meets fails on the executed arm
            // by the planted δ.
            assert!(
                circular_gap(sample.executed, sample.nearest_point) >= DELTA - 2.0 * COORDINATE_TOL,
                "the executed arm must differ from the nearest point by δ at t = {}",
                sample.coordinate
            );
            assert!(
                sample.projection_residual <= squared_tol(),
                "a rotated circle stays on the image, got residual {} at t = {}",
                sample.projection_residual,
                sample.coordinate
            );
        }
        let law = &rotation.transition_law;
        assert_eq!(law.class, CircleTransportClass::Shift, "{law:?}");
        assert_eq!(law.winding, 1, "{law:?}");
        assert!(
            circular_gap(law.phase / TAU, DELTA) <= COORDINATE_TOL,
            "the executed law's phase must be δ, got {} turns",
            law.phase / TAU
        );
        assert!(law.defect <= squared_tol(), "a rotation is rigid, got defect {}", law.defect);
        assert_eq!(rotation.smooth_map.degree, Some(1), "{:?}", rotation.smooth_map);
        assert!(
            rotation.smooth_map.topology_preserved,
            "a rotation is a fold-free degree-1 cover: {:?}",
            rotation.smooth_map
        );
        assert!(rotation.mean_projection_residual <= squared_tol());
        assert!(
            (rotation.target_radius_sq - 1.0).abs() <= 8.0 * f64::EPSILON,
            "a unit circle has r² = 1, got {}",
            rotation.target_radius_sq
        );

        // Control: the executed transport of the identity block IS the
        // nearest-point correspondence.
        let identity = measured(measure_anchor_to_block(
            &design,
            executed(vec![harvested_row(context, own)], |state: &[f64]| state.to_vec()),
        ))
        .expect("identity block");
        for sample in &identity.samples {
            assert!(
                circular_gap(sample.executed, sample.nearest_point) <= COORDINATE_TOL,
                "the identity block must reproduce the nearest point at t = {}: {} vs {}",
                sample.coordinate,
                sample.executed,
                sample.nearest_point
            );
        }
        assert!(
            circular_gap(identity.transition_law.phase / TAU, 0.0) <= COORDINATE_TOL,
            "the identity block's law must be the identity, got {} turns",
            identity.transition_law.phase / TAU
        );
    }

    /// The block's rotation depends on the other state. A held `c` gives one rigid
    /// map at that `c`'s rotation; a resampled `c` averages the rotations, and
    /// its resultant has a closed form. A design that contradicts its declared
    /// law, and a transition that fails or reads back a bad matrix, are refused
    /// with their typed errors.
    #[test]
    fn a_held_and_a_resampled_other_state_are_checked_and_measure_different_laws() {
        let epsilon = 0.21;
        let mut seed = 5u64;
        let contexts: Vec<[f64; LAYER_WIDTH]> = (0..SAMPLES)
            .map(|_| {
                [
                    lcg(&mut seed) - 0.5,
                    lcg(&mut seed) - 0.5,
                    4.0 * (lcg(&mut seed) - 0.5),
                ]
            })
            .collect();
        let own: Vec<f64> = (0..SAMPLES).map(|_| lcg(&mut seed)).collect();
        let rows: Vec<[f64; LAYER_WIDTH]> = contexts
            .iter()
            .zip(&own)
            .map(|(&context, &coordinate)| harvested_row(context, coordinate))
            .collect();
        let turns = |context: &[f64; LAYER_WIDTH]| DELTA + epsilon * context[2].tanh();

        let held_design = ExecutedTransportDesign {
            other_state: OtherStateLaw::Held,
            row_coordinates: vec![own[0]],
            samples: (0..SAMPLES).map(|_| (0, lcg(&mut seed))).collect(),
        };
        let held = measured(measure_anchor_to_block(
            &held_design,
            executed(rows[..1].to_vec(), context_rotation(DELTA, epsilon)),
        ))
        .expect("held other state");
        let held_turns = turns(&contexts[0]);
        for sample in &held.samples {
            assert!(
                circular_gap(sample.executed, sample.coordinate + held_turns) <= COORDINATE_TOL,
                "held: t = {} must land at t + {held_turns}, got {}",
                sample.coordinate,
                sample.executed
            );
        }
        assert!(
            circular_gap(held.transition_law.phase / TAU, held_turns) <= COORDINATE_TOL,
            "held: the law's phase must be the held row's rotation {held_turns}, got {}",
            held.transition_law.phase / TAU
        );
        assert!(held.transition_law.defect <= squared_tol(), "{:?}", held.transition_law);

        let resampled_design = ExecutedTransportDesign {
            other_state: OtherStateLaw::Resampled,
            row_coordinates: own.clone(),
            samples: (0..SAMPLES).map(|row| (row, lcg(&mut seed))).collect(),
        };
        let resampled = measured(measure_anchor_to_block(
            &resampled_design,
            executed(rows.clone(), context_rotation(DELTA, epsilon)),
        ))
        .expect("resampled other state");
        assert_eq!(resampled.other_state, OtherStateLaw::Resampled);
        for sample in &resampled.samples {
            let expected = sample.coordinate + turns(&contexts[sample.row]);
            assert!(
                circular_gap(sample.executed, expected) <= COORDINATE_TOL,
                "resampled: row {} at t = {} must land at {expected}, got {}",
                sample.row,
                sample.coordinate,
                sample.executed
            );
        }
        // Pair k's angle difference is 2π·turns(c_k), so R₊ = |Σ_k e^{2πi·turns(c_k)}|/n.
        let (re, im) = contexts.iter().fold((0.0_f64, 0.0_f64), |(re, im), context| {
            let angle = TAU * turns(context);
            (re + angle.cos(), im + angle.sin())
        });
        let resultant = re.hypot(im) / SAMPLES as f64;
        assert!(
            1.0 - resultant > 2.0 * TAU * COORDINATE_TOL,
            "the control needs a resolvably non-rigid resampled law, got R₊ = {resultant}"
        );
        assert!(
            (resampled.transition_law.resultant_shift - resultant).abs() <= TAU * COORDINATE_TOL,
            "resampled: R₊ = {} must equal the closed form {resultant}",
            resampled.transition_law.resultant_shift
        );
        assert!(
            resampled.transition_law.defect > held.transition_law.defect + TAU * COORDINATE_TOL,
            "resampling the other state must raise the defect: held {}, resampled {}",
            held.transition_law.defect,
            resampled.transition_law.defect
        );

        let held_two_rows = ExecutedTransportDesign {
            other_state: OtherStateLaw::Held,
            row_coordinates: own[..2].to_vec(),
            samples: vec![(0, 0.1), (1, 0.2)],
        };
        let refused = measure_anchor_to_block(
            &held_two_rows,
            executed(rows.clone(), context_rotation(DELTA, epsilon)),
        );
        assert!(
            matches!(
                &refused,
                Err(ExecutedTransportError::InvalidDesign(message)) if message.contains("held other state")
            ),
            "a held design over two rows must be refused, got {refused:?}"
        );
        let resampled_repeat = ExecutedTransportDesign {
            other_state: OtherStateLaw::Resampled,
            row_coordinates: own[..2].to_vec(),
            samples: vec![(0, 0.1), (0, 0.2), (1, 0.3)],
        };
        let refused = measure_anchor_to_block(
            &resampled_repeat,
            executed(rows.clone(), context_rotation(DELTA, epsilon)),
        );
        assert!(
            matches!(
                &refused,
                Err(ExecutedTransportError::InvalidDesign(message)) if message.contains("resampled other state")
            ),
            "a resampled design that patches a row twice must be refused, got {refused:?}"
        );
        // The held design patches row 0, so this readback is one row short.
        let short_readback = measure_anchor_to_block(
            &held_design,
            |row: usize, splices: ArrayView2<'_, f64>| {
                Ok(Array2::<f64>::zeros((row + splices.nrows() - 1, splices.ncols())))
            },
        );
        assert!(
            matches!(
                &short_readback,
                Err(ExecutedTransportError::Transition { row: 0, message }) if message.contains("read back")
            ),
            "a readback with the wrong shape must be refused, got {short_readback:?}"
        );
        let non_finite_readback = measure_anchor_to_block(
            &held_design,
            |row: usize, splices: ArrayView2<'_, f64>| {
                Ok(Array2::<f64>::from_elem(splices.dim(), f64::NAN + row as f64))
            },
        );
        assert!(
            matches!(
                &non_finite_readback,
                Err(ExecutedTransportError::Transition { row: 0, message }) if message.contains("non-finite")
            ),
            "a non-finite readback must be refused, got {non_finite_readback:?}"
        );
        let raised = measure_anchor_to_block(
            &held_design,
            |row: usize, splices: ArrayView2<'_, f64>| {
                Err(format!("runner raised on row {row} with {} splices", splices.nrows()))
            },
        );
        assert!(
            matches!(
                &raised,
                Err(ExecutedTransportError::Transition { row: 0, message })
                    if message == &format!("runner raised on row 0 with {} splices", SAMPLES + 1)
            ),
            "a failing transition must come back unchanged for its row, got {raised:?}"
        );
    }

    /// Three blocks, each showing one part of the error and not the others:
    /// - a contraction leaves a rigid chart map but a readback off the image;
    /// - a warp along the circle stays on the image and is a fold-free cover, but
    ///   is not rigid;
    /// - a scrambled block stays on the image but has no structure.
    #[test]
    fn the_projection_error_the_rigid_defect_and_the_non_rigid_part_are_reported_apart() {
        let mut seed = 17u64;
        let coordinates: Vec<(usize, f64)> = (0..SAMPLES).map(|_| (0, lcg(&mut seed))).collect();
        let own = 0.25;
        let design = ExecutedTransportDesign {
            other_state: OtherStateLaw::Held,
            row_coordinates: vec![own],
            samples: coordinates.clone(),
        };

        // Contraction: halve the circle plane. The readback ½q(t) lies on q(t)'s
        // ray, so the encoder returns t and discards (1 − ½)² = ¼.
        let contraction = measured(measure_anchor_to_block(
            &design,
            executed(vec![harvested_row([0.3, 0.2, 0.5], own)], |state: &[f64]| {
                vec![0.5 * state[0], 0.5 * state[1], state[2]]
            }),
        ))
        .expect("contraction");
        for sample in &contraction.samples {
            assert!(
                circular_gap(sample.executed, sample.coordinate) <= COORDINATE_TOL,
                "contraction: t = {} must stay at t, got {}",
                sample.coordinate,
                sample.executed
            );
            assert!(
                (sample.projection_residual - 0.25).abs() <= squared_tol(),
                "contraction: the encoder must discard 1/4 at t = {}, got {}",
                sample.coordinate,
                sample.projection_residual
            );
        }
        assert!((contraction.mean_projection_residual - 0.25).abs() <= squared_tol());
        assert_eq!(contraction.transition_law.class, CircleTransportClass::Shift);
        assert!(
            contraction.transition_law.defect <= squared_tol(),
            "contraction: the chart map is rigid, got defect {}",
            contraction.transition_law.defect
        );

        // Warp: move along the circle to w(t) = t + β·sin(2πt)/2π, with
        // w′ = 1 + β·cos 2πt > 0. The readback stays on the image, but the map is
        // not rigid. The other state has no plane component, so the executed
        // baseline of this nonlinear block is the context alone.
        let beta = 0.6;
        let warp = move |state: &[f64]| {
            let radius = state[0].hypot(state[1]);
            let angle = state[0].atan2(state[1]);
            let warped = angle + beta * angle.sin();
            vec![radius * warped.sin(), radius * warped.cos(), state[2]]
        };
        let warped = measured(measure_anchor_to_block(
            &design,
            executed(vec![harvested_row([0.0, 0.0, 0.5], own)], warp),
        ))
        .expect("warp");
        for sample in &warped.samples {
            let expected = sample.coordinate + beta * (TAU * sample.coordinate).sin() / TAU;
            assert!(
                circular_gap(sample.executed, expected) <= COORDINATE_TOL,
                "warp: t = {} must land at {expected}, got {}",
                sample.coordinate,
                sample.executed
            );
            assert!(
                sample.projection_residual <= squared_tol(),
                "warp: the readback stays on the image, got residual {} at t = {}",
                sample.projection_residual,
                sample.coordinate
            );
        }
        assert!(warped.mean_projection_residual <= squared_tol());
        // Pair k's angle difference is β·sin 2πt_k, so R₊ = |Σ_k e^{iβ·sin 2πt_k}|/n.
        let (re, im) = coordinates.iter().fold((0.0_f64, 0.0_f64), |(re, im), sample| {
            let angle = beta * (TAU * sample.1).sin();
            (re + angle.cos(), im + angle.sin())
        });
        let resultant = re.hypot(im) / SAMPLES as f64;
        assert!(
            1.0 - resultant > 2.0 * TAU * COORDINATE_TOL,
            "the control needs a resolvably non-rigid warp, got R₊ = {resultant}"
        );
        assert!(
            (warped.transition_law.resultant_shift - resultant).abs() <= TAU * COORDINATE_TOL,
            "warp: R₊ = {} must equal the closed form {resultant}",
            warped.transition_law.resultant_shift
        );
        // The warp's non-rigid part: a fold-free degree-1 cover whose isometry
        // defect is resolved from rounding and exceeds the rigid contraction's.
        assert_eq!(warped.smooth_map.degree, Some(1), "{:?}", warped.smooth_map);
        assert!(
            warped.smooth_map.topology_preserved,
            "a warp with w′ > 0 is a fold-free degree-1 cover: {:?}",
            warped.smooth_map
        );
        assert!(
            warped.smooth_map.isometry_defect > warped.smooth_map.isometry_defect_band(),
            "warp: the isometry defect {} must be resolved from its rounding band {}",
            warped.smooth_map.isometry_defect,
            warped.smooth_map.isometry_defect_band()
        );
        assert!(
            warped.smooth_map.isometry_defect > contraction.smooth_map.isometry_defect,
            "the warp's isometry defect {} must exceed the rigid contraction's {}",
            warped.smooth_map.isometry_defect,
            contraction.smooth_map.isometry_defect
        );

        // Scrambled: the readback stays on the circle at a pseudo-random angle
        // hashed from the incoming one, so the target angle carries no function of
        // the source angle. The other state has no plane component, as for the
        // warp.
        let scramble = |state: &[f64]| {
            let radius = state[0].hypot(state[1]);
            let mut hash = state[0].atan2(state[1]).to_bits();
            let turns = lcg(&mut hash);
            let (sin, cos) = (TAU * turns).sin_cos();
            vec![radius * sin, radius * cos, state[2]]
        };
        let scrambled_design = ExecutedTransportDesign {
            other_state: OtherStateLaw::Held,
            row_coordinates: vec![own],
            samples: (0..SCRAMBLED_SAMPLES).map(|_| (0, lcg(&mut seed))).collect(),
        };
        let scrambled = measured(measure_anchor_to_block(
            &scrambled_design,
            executed(vec![harvested_row([0.0, 0.0, 0.5], own)], scramble),
        ))
        .expect("scrambled");
        assert!(
            scrambled.mean_projection_residual <= squared_tol(),
            "scrambled: the readback stays on the image, got {}",
            scrambled.mean_projection_residual
        );
        assert_eq!(
            scrambled.transition_law.class,
            CircleTransportClass::Mixing,
            "{:?}",
            scrambled.transition_law
        );
        assert_ne!(
            warped.transition_law.class,
            CircleTransportClass::Mixing,
            "a warp must not be classified as mixing: {:?}",
            warped.transition_law
        );
        assert!(
            scrambled.smooth_map.residual_rms > warped.smooth_map.residual_rms,
            "mixing must leave more scatter about the smooth map than a warp: scrambled {}, warp {}",
            scrambled.smooth_map.residual_rms,
            warped.smooth_map.residual_rms
        );
    }
}
