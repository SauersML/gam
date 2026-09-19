//! Real-model composition receipt for #2946: the MLP of GPT-NeoX block `l` composed into the MLP of block `l + 1`, on
//! the arrays `experiments/issue-2946/finite_response_teacher/export_block.py pair` writes. The torch side only runs one
//! forward pass with hooks; every number here is computed in Rust, through the landed owners: `raw_block` for the torch
//! layout and the declared-law absorption, `compose` for the composition, `state_blocks` for the common block
//! decomposition, and gam-math for the activation and the standard-normal draws.
//!
//! Stages:
//!
//! * `check`    the exported rows are the real model's function. `h_first = LN_l(x_in)`, `y_first = MLP_l(h_first)`,
//!   `h_second = LN_{l+1}(x_first)` and `y_second = MLP_{l+1}(h_second)` are recomputed in float64 from the float32
//!   rows and the float64 parameters, each against the first-order rounding bound of torch's float32 evaluation (every
//!   bound holds for any summation order; the float32 activation's own accuracy is not bounded). With a parallel
//!   residual, `x_first = x_in + attn_l + y_first`, so the attention write `attn_l` is read off and its size reported:
//!   it is what a surgery that removes attention throws away.
//! * `compose`  the declared-law composition. The law is `h = h₀ + L Z`, `Z ~ N(0, I_k)`, at the first MLP's post-norm
//!   input (`h₀`, `L` from `finite_response_teacher_2946 declare` on `h_first`). The stream entering block `l + 1` is
//!   `x = r̄ + y`, `y = MLP_l(h)`, with the rest of the stream `r = x_first − y_first` held at its row mean `r̄`. The
//!   second MLP reads `LN_{l+1}(x)`; the declared second stage is a SURGERY that freezes that norm's scale at the mean
//!   stream
//!   `x̄ = mean(x_first)`, so `LN(x) = β + γ ⊙ C x / s₀` with `C` the centring and `s₀ = √(‖C x̄‖²/D + ε)`, an affine
//!   map, and `F₂'(y) = MLP_{l+1}(β + γ ⊙ C (r̄ + y) / s₀)` is one known block reading `y`. At retained points `P z` of
//!   each frame (`top:q`, the leading `q` coordinates of `Z`, which with `L = V Λ^{1/2}` are the top-`q` principal
//!   directions of the declared law; `bottom:q`, the `q` least-variance coordinates the law resolves from zero; or
//!   `readers:q`, the top `q` right singular vectors of the first block's readers `W L`) it reports, against the
//!   executed layer-by-layer Monte Carlo
//!   `z → h → y → x → LN → MLP_{l+1}` under the same law:
//!   - the second stage's pre-activation conditional means and variances, which `gaussian_closure_response` gives
//!     EXACTLY (the mean composition `F₂'(E[y | P z])` gets the means right and every variance wrong, as zero);
//!   - the closure response `F̂₂(P z)`, which is APPROXIMATE, with its measured error next to the mean composition's;
//!   - the frozen-norm surgery's own error: the executed second stage with the true norm against the frozen one;
//!   - `E(P)` and `V(I)` of the second stage's pre-activations, exact through `compose_affine_stage`, against the
//!     coupled Monte Carlo `E|c(Z) − c(Z')|²/2` with `Z' = P Z + (I − P) Z̃`;
//!   - the wall time of every exact and executed route.
//!
//!   Every exact agreement is gated at a Bonferroni multiple of its standard error, and carries a magnitude floor: the
//!   compared quantity is shown to be resolved from zero at the same multiple, so an agreement cannot pass on two
//!   zeros.
//! * `blocks`   state blocks of the second MLP's residual-stream transition `x ↦ x + MLP_{l+1}(LN_{l+1}(x))` at the
//!   exported rows: the Jacobians `J_i = I + U diag(σ'(c_i)) W J_LN(x_i)` (true norm) are decomposed together by
//!   `StateBlocks::decompose`. `LN(x + t 1) = LN(x)`, so `J_i 1 = 1` for every `i`: the ones line is invariant under
//!   every map, but not under their transposes, so it is not an orthogonal state block. The stage reports both residuals
//!   and, as the positive control, recovers a planted pair of 256-dimensional blocks built from the same Jacobians in a
//!   random orthogonal frame.
//! * `a3-draw`, `a3`   acceptance A3 on one real block of the pair: the analytic `V(I)`, `E(P)`, `V(P)` of the known
//!   block (`raw_block` absorption of the declared law) against the block EXECUTED by torch
//!   (`export_block.py execute`, float64) on rows this file draws, with the teacher's paired estimators
//!   `V(I) = E|F(Z) − F(Z~)|²/2` and `E(P) = E|F(Z) − F(Z')|²/2`, `Z' = Q Qᵀ Z + (I − Q Qᵀ) Z~`, and the analytic best
//!   retained response `F̄_P(P z)` against each retained point's executed conditional mean. `top:k` retains
//!   everything, so `Z' = Z` exactly: its executed `E` is the row-alignment control. A float64 forward pass of this file
//!   on the leading drawn rows checks that the executed rows are this block's function.
//! * `a7`       acceptance A7 on the same real block: each frame's best retained response compiled into a compact GAM
//!   `g` (`response::compile`: Duchon smooth, multi-penalty REML), which reports R2's two terms separately, the exact
//!   `E(P)` and the held-out function error `Â`. Both are then read off the A3 draw's rows EXECUTED by torch: the
//!   executed total `‖F(Z) − g(Qᵀ Z)‖²_M` against `E(P) + Â`, and two estimates of the function error against `Â`,
//!   one free of the operator and one with the discarded part cancelled row by row. The negative control reads the
//!   same `g` through a perturbed frame and must miss; `bottom:q` is the blind frame.
//!
//! ```text
//! cargo run --profile test -p gam-cli --example finite_response_compose_2946 -- check --pair PAIR --rows R --out OUT
//! cargo run --profile test -p gam-cli --example finite_response_compose_2946 -- compose --pair PAIR --law LAW \
//!     --frames top:8,top:64,readers:8,readers:64 --points 4 --draws 65536 --chunk 4096 --pairs 4096 --seed S --out OUT
//! cargo run --profile test -p gam-cli --example finite_response_compose_2946 -- blocks --pair PAIR --maps 16 --seed S --out OUT
//! cargo run --profile test -p gam-cli --example finite_response_compose_2946 -- a3-draw --pair PAIR --law LAW \
//!     --block first --frames top:8,top:64,readers:8,top:512 --rows 8192 --points 4 --point-draws 2048 --seed S --out DRAW
//! python export_block.py execute --model M --revision R --layer L --inputs DRAW/h.npy --batch B --out DRAW/executed.npy
//! cargo run --profile test -p gam-cli --example finite_response_compose_2946 -- a3 --pair PAIR --law LAW --draw DRAW \
//!     --executed DRAW/executed.npy --out OUT
//! cargo run --profile test -p gam-cli --example finite_response_compose_2946 -- a7 --pair PAIR --law LAW --draw DRAW \
//!     --executed DRAW/executed.npy --frames top:2,top:4,bottom:2 --training-draws 4000 --holdout-draws 4000 --seed S \
//!     --out OUT
//! ```

use clap::{Parser, Subcommand};
use gam::faer_ndarray::{FaerSvd, fast_ab, fast_abt};
use gam::linalg::roundoff::factor_singular_band;
use gam::utils::splitmix64;
use gam_math::gaussian_activation::{GaussianActivation, gaussian_smoothing_derivatives};
use gam_math::probability::{standard_normal_from_uniform_bits, standard_normal_quantile};
use gam_sae::response::compile::{
    CompileDesign, CompiledResponse, FunctionRepresentation, compile_retained_response,
};
use gam_sae::response::compose::{compose_affine_stage, gaussian_closure_response};
use gam_sae::response::raw_block::UnabsorbedBlock;
use gam_sae::response::state_blocks::StateBlocks;
use gam_sae::response::subspace::KnownBlock;
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, Axis, Zip, s};
use npyz::{NpyFile, Order, WriterBuilder};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

/// float32 unit roundoff: torch evaluated the exported rows in float32.
const FLOAT32_UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;

/// The family-wise probability that any Monte Carlo comparison of one stage rejects a true agreement.
const FAMILY_WISE_FALSE_ALARM: f64 = 1e-6;

#[derive(Parser)]
#[command(about = "Real-model composition receipt for #2946 on an exported GPT-NeoX block pair")]
struct Cli {
    #[command(subcommand)]
    stage: Stage,
}

#[derive(Subcommand)]
enum Stage {
    /// Recompute the exported rows from the exported parameters.
    Check {
        #[arg(long)]
        pair: PathBuf,
        /// Leading rows compared.
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// The declared-law composition against the executed layer-by-layer Monte Carlo.
    Compose {
        #[arg(long)]
        pair: PathBuf,
        /// `finite_response_teacher_2946 declare` output on the pair's `h_first` rows.
        #[arg(long)]
        law: PathBuf,
        /// Retained frames, comma separated: `top:q` (the leading `q` coordinates of `Z`), `bottom:q` (the `q`
        /// least-variance coordinates the law resolves from zero) or `readers:q` (the top `q` right singular vectors
        /// of the first block's readers `W L`).
        #[arg(long, value_delimiter = ',')]
        frames: Vec<String>,
        /// Retained points per frame.
        #[arg(long)]
        points: usize,
        /// Monte Carlo draws of the discarded coordinates per retained point.
        #[arg(long)]
        draws: usize,
        /// Draws evaluated at once.
        #[arg(long)]
        chunk: usize,
        /// Coupled pairs `(Z, Z')` per frame for `E(P)`.
        #[arg(long)]
        pairs: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// A3: draw the declared law's rows for torch to execute on one real block of the pair.
    A3Draw {
        #[arg(long)]
        pair: PathBuf,
        /// `finite_response_teacher_2946 declare` output on that block's post-norm rows.
        #[arg(long)]
        law: PathBuf,
        /// `first` or `second`.
        #[arg(long)]
        block: String,
        /// Frames as in `compose`; `top:k` retains everything, the row-alignment control.
        #[arg(long, value_delimiter = ',')]
        frames: Vec<String>,
        /// Rows of `Z`, `Z~` and each frame's coupled `Z'`.
        #[arg(long)]
        rows: usize,
        /// Retained points of the first frame, for the best retained response.
        #[arg(long)]
        points: usize,
        /// Draws of the discarded coordinates per retained point.
        #[arg(long)]
        point_draws: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// A3: the analytic `V(I)`, `E(P)`, `V(P)` and best retained response against the executed block.
    A3 {
        #[arg(long)]
        pair: PathBuf,
        #[arg(long)]
        law: PathBuf,
        #[arg(long = "draw")]
        draw_dir: PathBuf,
        #[arg(long)]
        executed: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// A7: compile each frame's best retained response into a compact GAM and set its reported split against the
    /// executed block on an A3 draw.
    A7 {
        #[arg(long)]
        pair: PathBuf,
        #[arg(long)]
        law: PathBuf,
        #[arg(long = "draw")]
        draw_dir: PathBuf,
        #[arg(long)]
        executed: PathBuf,
        /// Frames to compile; each must be one of the A3 draw's frames, whose coupled rows it reads.
        #[arg(long, value_delimiter = ',')]
        frames: Vec<String>,
        /// Draws the compact GAM is fitted on.
        #[arg(long)]
        training_draws: usize,
        /// Independent draws its function error is estimated on.
        #[arg(long)]
        holdout_draws: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// State blocks of the second MLP's transition Jacobians on the residual stream.
    Blocks {
        #[arg(long)]
        pair: PathBuf,
        /// Jacobians, at evenly spaced exported rows.
        #[arg(long)]
        maps: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> ExitCode {
    let outcome = match Cli::parse().stage {
        Stage::Check { pair, rows, out } => check(&pair, rows, &out),
        Stage::Compose {
            pair,
            law,
            frames,
            points,
            draws,
            chunk,
            pairs,
            seed,
            out,
        } => compose(
            &pair,
            &law,
            &frames,
            &ComposeSizes {
                points,
                draws,
                chunk,
                pairs,
            },
            seed,
            &out,
        ),
        Stage::A3Draw {
            pair,
            law,
            block,
            frames,
            rows,
            points,
            point_draws,
            seed,
            out,
        } => a3_draw(
            &pair,
            &law,
            &block,
            &frames,
            &A3Sizes {
                rows,
                points,
                point_draws,
            },
            seed,
            &out,
        ),
        Stage::A3 {
            pair,
            law,
            draw_dir,
            executed,
            out,
        } => a3(&pair, &law, &draw_dir, &executed, &out),
        Stage::A7 {
            pair,
            law,
            draw_dir,
            executed,
            frames,
            training_draws,
            holdout_draws,
            seed,
            out,
        } => a7(
            &pair,
            &law,
            &draw_dir,
            &executed,
            &frames,
            CompileRun {
                training_draws,
                holdout_draws,
                seed,
            },
            &out,
        ),
        Stage::Blocks {
            pair,
            maps,
            seed,
            out,
        } => blocks(&pair, maps, seed, &out),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("[finite_response_compose_2946] error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// The exported pair: both blocks' MLPs in the torch layout, both norms, and the harvest's provenance.
struct Pair {
    dir: PathBuf,
    meta: Value,
    first: UnabsorbedBlock,
    second: UnabsorbedBlock,
    first_norm: LayerNorm,
    second_norm: LayerNorm,
}

impl Pair {
    fn load(dir: &Path) -> Result<Self, String> {
        let meta = read_json(&dir.join("meta.json"))?;
        if meta.get("stage").and_then(Value::as_str) != Some("pair") {
            return Err(format!("{} is not an export_block.py pair export", dir.display()));
        }
        if meta.get("use_parallel_residual").and_then(Value::as_bool) != Some(true) {
            return Err("the composition reads x_first = x_in + attn + y_first, a parallel residual".to_string());
        }
        if meta.get("norm_class").and_then(Value::as_str) != Some("LayerNorm") {
            return Err("the declared second stage freezes a LayerNorm".to_string());
        }
        let hidden_act = meta
            .get("hidden_act")
            .and_then(Value::as_str)
            .ok_or("meta.json names no hidden_act")?
            .to_string();
        let eps = meta
            .get("norm_eps")
            .and_then(Value::as_f64)
            .ok_or("meta.json names no norm_eps")?;
        let layer = |key: &str| {
            meta.get("layers")
                .and_then(|layers| layers.get(key))
                .and_then(Value::as_u64)
                .ok_or(format!("meta.json names no {key} layer"))
        };
        let (first_layer, second_layer) = (layer("first")?, layer("second")?);
        let hidden = meta
            .get("hidden_size")
            .and_then(Value::as_u64)
            .ok_or("meta.json names no hidden_size")? as usize;
        let block = |index: u64| -> Result<UnabsorbedBlock, String> {
            let mut parameters = BTreeMap::new();
            for name in [
                "dense_h_to_4h.weight",
                "dense_h_to_4h.bias",
                "dense_4h_to_h.weight",
                "dense_4h_to_h.bias",
            ] {
                parameters.insert(
                    name.to_string(),
                    read_array(&dir.join(format!("layer{index}.mlp.{name}.npy")))?,
                );
            }
            UnabsorbedBlock::from_torch_parameters(parameters, &hidden_act, Array2::eye(hidden))
                .map_err(|error| format!("layer {index} MLP: {error}"))
        };
        let norm = |index: u64| -> Result<LayerNorm, String> {
            Ok(LayerNorm {
                weight: read_vector(&dir.join(format!("layer{index}.post_attention_layernorm.weight.npy")))?,
                bias: read_vector(&dir.join(format!("layer{index}.post_attention_layernorm.bias.npy")))?,
                eps,
            })
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            first: block(first_layer)?,
            second: block(second_layer)?,
            first_norm: norm(first_layer)?,
            second_norm: norm(second_layer)?,
            meta,
        })
    }

    fn rows(&self, name: &str, max_rows: Option<usize>) -> Result<Array2<f64>, String> {
        read_rows::<f32>(&self.dir.join(format!("{name}.npy")), max_rows)
    }

    fn provenance(&self) -> Value {
        json!({
            "pair": self.dir.display().to_string(),
            "model": self.meta.get("model").cloned().unwrap_or(Value::Null),
            "revision": self.meta.get("revision").cloned().unwrap_or(Value::Null),
            "layers": self.meta.get("layers").cloned().unwrap_or(Value::Null),
            "hidden_act": self.meta.get("hidden_act").cloned().unwrap_or(Value::Null),
            "context_set": self.meta.get("context_set").cloned().unwrap_or(Value::Null),
            "arrays": self.meta.get("arrays").cloned().unwrap_or(Value::Null),
        })
    }
}

/// torch's `LayerNorm`: `γ ⊙ (x − mean x) / √(var x + ε) + β`, with the biased variance.
struct LayerNorm {
    weight: Array1<f64>,
    bias: Array1<f64>,
    eps: f64,
}

impl LayerNorm {
    /// The norm of every row, in float64.
    fn apply(&self, rows: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = rows.to_owned();
        for mut row in out.rows_mut() {
            let (centred, scale) = self.centred_and_scale(row.view());
            row.assign(&(&(&centred / scale) * &self.weight + &self.bias));
        }
        out
    }

    /// `C x` and `s = √(‖C x‖²/D + ε)`.
    fn centred_and_scale(&self, row: ArrayView1<'_, f64>) -> (Array1<f64>, f64) {
        let width = row.len() as f64;
        let mean = row.sum() / width;
        let centred = row.mapv(|value| value - mean);
        let variance = centred.dot(&centred) / width;
        (centred, (variance + self.eps).sqrt())
    }

    /// The first-order bound on torch's float32 evaluation of the norm of `row`, entry by entry, for any summation
    /// order: the mean's `γ_{D+1} mean|x|`, the centred entries, the variance's `γ_{D+1}` plus its centred-entry terms,
    /// the scale's add and square root, and the affine output's product and sum.
    fn float32_bound(&self, row: ArrayView1<'_, f64>) -> Array1<f64> {
        let width = row.len();
        let u = FLOAT32_UNIT_ROUNDOFF;
        let gamma = gamma_n(width + 1, u);
        let (centred, scale) = self.centred_and_scale(row);
        let mean_error = gamma * row.mapv(f64::abs).sum() / width as f64;
        let centred_error = centred.mapv(|value| mean_error + u * value.abs());
        let variance = centred.dot(&centred) / width as f64;
        let variance_error =
            gamma * variance + 2.0 * centred.mapv(f64::abs).dot(&centred_error) / width as f64;
        let scale_relative = 0.5 * variance_error / (variance + self.eps) + 2.0 * u;
        let normalised = &centred / scale;
        let normalised_error = &centred_error / scale + &(normalised.mapv(f64::abs) * (scale_relative + u));
        let out = &normalised * &self.weight + &self.bias;
        &(self.weight.mapv(f64::abs) * &normalised_error)
            + &((&(&normalised * &self.weight).mapv(f64::abs) + &out.mapv(f64::abs)) * u)
    }
}

/// `γ_n = n u / (1 − n u)`.
fn gamma_n(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}

/// `∂ᵏσ` for every `k < order` (`order` 1 or 2) at every entry of `pre`, through gam-math's owner at zero variance,
/// where `T_0 σ = σ`.
fn activation_jet(activation: GaussianActivation, pre: &Array2<f64>, order: usize) -> Result<Vec<Array2<f64>>, String> {
    if !(1..=2).contains(&order) {
        return Err(format!("activation jet of order {order}: only the value and the slope are read"));
    }
    let entries = pre
        .as_standard_layout()
        .as_slice()
        .ok_or("a standard-layout array has a contiguous slice")?
        .par_iter()
        .map(|&t| {
            let mut jet = [0.0; 2];
            gaussian_smoothing_derivatives(activation, t, 0.0, &mut jet[..order])
                .map_err(|error| format!("activation at {t}: {error}"))?;
            Ok(jet)
        })
        .collect::<Result<Vec<[f64; 2]>, String>>()?;
    (0..order)
        .map(|derivative| {
            Array2::from_shape_vec(pre.dim(), entries.iter().map(|jet| jet[derivative]).collect())
                .map_err(|error| format!("activation jet shape: {error}"))
        })
        .collect()
}

/// `σ` at every entry of `pre`.
fn activation_values(activation: GaussianActivation, pre: &Array2<f64>) -> Result<Array2<f64>, String> {
    activation_jet(activation, pre, 1)?
        .pop()
        .ok_or_else(|| "an order-1 jet holds the value".to_string())
}

/// `σ` and `σ'` at every entry of `pre`.
fn activation_values_and_slopes(
    activation: GaussianActivation,
    pre: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let mut jet = activation_jet(activation, pre, 2)?;
    let slopes = jet.pop().ok_or("an order-2 jet holds the slope")?;
    let values = jet.pop().ok_or("an order-2 jet holds the value")?;
    Ok((values, slopes))
}

/// `U σ(W h + b) + c` for every row `h` of `inputs`.
fn mlp_forward(block: &UnabsorbedBlock, inputs: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut pre = fast_abt(inputs, &block.readers);
    pre += &block.biases;
    let units = activation_values(block.activation, &pre)?;
    let mut out = fast_abt(&units, &block.writers);
    out += &block.output_bias;
    Ok(out)
}

/// `U σ(W h + b) + c` for every row `h`, with the first-order bound on an evaluation at unit roundoff `u` from the same
/// inputs: `γ_{n+1}` of each reader and output sum, carried through `|σ'|` (the evaluating activation's own accuracy is
/// not bounded).
fn mlp_forward_with_bound(
    block: &UnabsorbedBlock,
    inputs: &Array2<f64>,
    u: f64,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let reader_gamma = gamma_n(inputs.ncols() + 1, u);
    let mut pre = fast_abt(inputs, &block.readers);
    pre += &block.biases;
    let mut pre_bound = fast_abt(&inputs.mapv(f64::abs), &block.readers.mapv(f64::abs));
    pre_bound += &block.biases.mapv(f64::abs);
    pre_bound *= reader_gamma;
    let (units, slopes) = activation_values_and_slopes(block.activation, &pre)?;
    let unit_bound = &slopes.mapv(f64::abs) * &pre_bound + &(units.mapv(f64::abs) * u);
    let writer_gamma = gamma_n(units.ncols() + 1, u);
    let writers_abs = block.writers.mapv(f64::abs);
    let mut out = fast_abt(&units, &block.writers);
    out += &block.output_bias;
    let mut bound = fast_abt(&units.mapv(f64::abs), &writers_abs);
    bound += &block.output_bias.mapv(f64::abs);
    bound *= writer_gamma;
    bound += &fast_abt(&unit_bound, &writers_abs);
    Ok((out, bound))
}

/// `max |gap| / bound`, the entries beyond twice the bound, and the relative Frobenius gap of `computed` against the
/// exported `exported`.
fn gap_report(computed: &Array2<f64>, exported: &Array2<f64>, bound: &Array2<f64>) -> Value {
    let mut max_ratio = 0.0_f64;
    let mut max_gap = 0.0_f64;
    let mut beyond = 0_usize;
    let mut gap_sq = 0.0;
    let mut norm_sq = 0.0;
    Zip::from(computed).and(exported).and(bound).for_each(|&value, &reference, &limit| {
        let gap = (value - reference).abs();
        max_gap = max_gap.max(gap);
        max_ratio = max_ratio.max(gap / limit);
        if gap > 2.0 * limit {
            beyond += 1;
        }
        gap_sq += gap * gap;
        norm_sq += reference * reference;
    });
    json!({
        "entries": computed.len(),
        "max_abs_gap": max_gap,
        "max_gap_over_bound": max_ratio,
        "entries_beyond_twice_bound": beyond,
        "relative_frobenius_gap": (gap_sq / norm_sq).sqrt(),
    })
}

fn check(pair_dir: &Path, rows: usize, out: &Path) -> Result<(), String> {
    let pair = Pair::load(pair_dir)?;
    let x_in = pair.rows("x_in", Some(rows))?;
    let h_first = pair.rows("h_first", Some(rows))?;
    let y_first = pair.rows("y_first", Some(rows))?;
    let x_first = pair.rows("x_first", Some(rows))?;
    let h_second = pair.rows("h_second", Some(rows))?;
    let y_second = pair.rows("y_second", Some(rows))?;
    let norm_check = |norm: &LayerNorm, input: &Array2<f64>, exported: &Array2<f64>| {
        let computed = norm.apply(input.view());
        let mut bound = Array2::<f64>::zeros(input.dim());
        for (mut bound_row, row) in bound.rows_mut().into_iter().zip(input.rows()) {
            bound_row.assign(&norm.float32_bound(row));
        }
        gap_report(&computed, exported, &bound)
    };
    let started = Instant::now();
    let first_norm = norm_check(&pair.first_norm, &x_in, &h_first);
    let second_norm = norm_check(&pair.second_norm, &x_first, &h_second);
    let (first_mlp_value, first_mlp_bound) = mlp_forward_with_bound(&pair.first, &h_first, FLOAT32_UNIT_ROUNDOFF)?;
    let first_mlp = gap_report(&first_mlp_value, &y_first, &first_mlp_bound);
    let (second_mlp_value, second_mlp_bound) = mlp_forward_with_bound(&pair.second, &h_second, FLOAT32_UNIT_ROUNDOFF)?;
    let second_mlp = gap_report(&second_mlp_value, &y_second, &second_mlp_bound);
    let attention = &x_first - &x_in - &y_first;
    let frobenius = |rows: &Array2<f64>| rows.iter().map(|value| value * value).sum::<f64>().sqrt();
    let report = json!({
        "stage": "check",
        "provenance": pair.provenance(),
        "rows": rows,
        "first_norm_h_first_from_x_in": first_norm,
        "first_mlp_y_first_from_h_first": first_mlp,
        "second_norm_h_second_from_x_first": second_norm,
        "second_mlp_y_second_from_h_second": second_mlp,
        "parallel_residual": {
            "attention_write_frobenius": frobenius(&attention),
            "first_mlp_write_frobenius": frobenius(&y_first),
            "stream_in_frobenius": frobenius(&x_in),
            "attention_over_first_mlp_write": frobenius(&attention) / frobenius(&y_first),
        },
        "bound": "first-order float32 rounding of torch's evaluation from the same float32 inputs, any summation order; the float32 activation's own accuracy is not bounded",
        "seconds": started.elapsed().as_secs_f64(),
    });
    write_json(out, &report)?;
    println!("[check] {}", serde_json::to_string(&report).map_err(|error| error.to_string())?);
    Ok(())
}

/// One draw from the owner's standard-normal stream.
fn standard_normal(state: &mut u64) -> Result<f64, String> {
    standard_normal_from_uniform_bits(splitmix64(state))
}

/// `rows × cols` draws from the owner's stream.
fn standard_normal_rows(rows: usize, cols: usize, state: &mut u64) -> Result<Array2<f64>, String> {
    let mut values = Array2::<f64>::zeros((rows, cols));
    for value in values.iter_mut() {
        *value = standard_normal(state)?;
    }
    Ok(values)
}

/// The standard-error multiple for `comparisons` two-sided comparisons at `FAMILY_WISE_FALSE_ALARM` (Bonferroni).
fn comparison_multiple(comparisons: usize) -> Result<f64, String> {
    standard_normal_quantile(1.0 - FAMILY_WISE_FALSE_ALARM / (2.0 * comparisons as f64))
}

/// Column means, unbiased variances and fourth central moments of a stream of sample rows, accumulated chunk by chunk
/// as power sums about the first chunk's column means, so no chunk is held after it is added and the sums do not
/// cancel against a large mean.
struct ColumnMoments {
    shift: Array1<f64>,
    sums: [Array1<f64>; 4],
    count: usize,
}

impl ColumnMoments {
    fn new() -> Self {
        Self {
            shift: Array1::zeros(0),
            sums: [Array1::zeros(0), Array1::zeros(0), Array1::zeros(0), Array1::zeros(0)],
            count: 0,
        }
    }

    fn add(&mut self, samples: &Array2<f64>) -> Result<(), String> {
        if self.count == 0 {
            self.shift = samples.mean_axis(Axis(0)).ok_or("an empty chunk has no mean")?;
            let width = samples.ncols();
            self.sums = [Array1::zeros(width), Array1::zeros(width), Array1::zeros(width), Array1::zeros(width)];
        }
        let deviations = samples - &self.shift;
        let mut power = deviations.clone();
        for sum in self.sums.iter_mut() {
            *sum += &power.sum_axis(Axis(0));
            power *= &deviations;
        }
        self.count += samples.nrows();
        Ok(())
    }

    fn count(&self) -> usize {
        self.count
    }

    /// The column means.
    fn mean(&self) -> Array1<f64> {
        &self.shift + &(&self.sums[0] / self.count as f64)
    }

    /// The unbiased column variances.
    fn variance(&self) -> Array1<f64> {
        let n = self.count as f64;
        Zip::from(&self.sums[0])
            .and(&self.sums[1])
            .map_collect(|&first, &second| (second - first * first / n) / (n - 1.0))
    }

    /// The standard error of each column mean.
    fn mean_se(&self) -> Array1<f64> {
        let n = self.count as f64;
        self.variance().mapv(|variance| (variance / n).sqrt())
    }

    /// The large-sample standard error of each unbiased variance, `√((μ₄ − σ⁴)/n)`, with `μ₄` the fourth central
    /// moment.
    fn variance_se(&self) -> Array1<f64> {
        let n = self.count as f64;
        let variance = self.variance();
        Zip::from(&self.sums[0])
            .and(&self.sums[1])
            .and(&self.sums[2])
            .and(&self.sums[3])
            .and(&variance)
            .map_collect(|&s1, &s2, &s3, &s4, &sigma2| {
                let a = s1 / n;
                let fourth = s4 / n - 4.0 * a * s3 / n + 6.0 * a * a * s2 / n - 3.0 * a.powi(4);
                ((fourth - sigma2 * sigma2).max(0.0) / n).sqrt()
            })
    }
}

/// `max |a − b| / se` over entries.
fn max_z(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>, se: ArrayView1<'_, f64>) -> f64 {
    Zip::from(&a)
        .and(&b)
        .and(&se)
        .fold(0.0_f64, |largest, &x, &y, &e| largest.max((x - y).abs() / e))
}

fn euclidean(values: ArrayView1<'_, f64>) -> f64 {
    values.dot(&values).sqrt()
}

/// The sample mean and its standard error.
fn mean_with_se(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// The declared law's principal variances `λ_j = ‖L e_j‖²`, in the declared (descending) order of the columns of `L`,
/// and the band on `√λ_j` below which a coordinate is not resolved from zero.
///
/// `finite_response_teacher_2946 declare` takes `L` from the thin SVD of the centred rows scaled by `1/√(n − 1)`, so
/// `√λ_j` is a singular value of that factor. Two errors sit in it. The harvested rows are torch's float32 outputs,
/// each entry within `u₃₂` of its value, which moves every singular value by at most `u₃₂‖X‖_F/√(n − 1)` (Weyl, the
/// centring being a projector), with `‖X‖_F² = n‖h₀‖² + (n − 1)·tr Σ̂` over the stored rows. That is the final rounding
/// alone, so the float32 forward's accumulated error can only widen it. The SVD adds its backward error,
/// `factor_singular_band`. A post-norm law has one exactly null direction, since the norm centres every row, and its
/// float32 rows leave it at roundoff variance, inside this band.
struct DeclaredSpectrum {
    variances: Vec<f64>,
    root_band: f64,
}

impl DeclaredSpectrum {
    fn read(law: &Path) -> Result<Self, String> {
        let meta = read_json(&law.join("law.json"))?;
        let rows = meta
            .get("rows")
            .and_then(Value::as_u64)
            .ok_or("law.json names no row count")? as usize;
        let total_variance = meta
            .get("total_variance")
            .and_then(Value::as_f64)
            .ok_or("law.json names no total variance")?;
        if rows < 2 {
            return Err(format!("law.json declares {rows} rows, which carry no covariance"));
        }
        let baseline = read_vector(&law.join("h0.npy"))?;
        let loading = read_rows::<f64>(&law.join("L.npy"), None)?;
        let variances: Vec<f64> = loading.columns().into_iter().map(|column| column.dot(&column)).collect();
        let largest = variances.iter().copied().fold(0.0_f64, f64::max);
        let scaled_row_norm = (rows as f64 * baseline.dot(&baseline) / (rows - 1) as f64 + total_variance).sqrt();
        let root_band =
            FLOAT32_UNIT_ROUNDOFF * scaled_row_norm + factor_singular_band(rows, loading.nrows(), largest.sqrt());
        Ok(Self { variances, root_band })
    }

    /// The latent coordinates resolved from zero, in declared order.
    fn resolved(&self) -> Vec<usize> {
        (0..self.variances.len())
            .filter(|&coordinate| self.variances[coordinate].sqrt() > self.root_band)
            .collect()
    }
}

/// A retained frame `Q` (`k × q`, orthonormal) of the latent `Z ~ N(0, I_k)`.
struct LatentFrame {
    name: String,
    basis: Array2<f64>,
}

impl LatentFrame {
    /// `top:q`, the leading `q` coordinates of `Z` (with `L = V Λ^{1/2}` from the post-norm covariance, the top-`q`
    /// principal directions of the declared law); `bottom:q`, the `q` least-variance coordinates `spectrum` resolves
    /// from zero (the blind frame; the post-norm null direction is not among them); or `readers:q`, the top `q` right
    /// singular vectors of the block's readers `W L`.
    fn parse(spec: &str, block: &KnownBlock, spectrum: &DeclaredSpectrum) -> Result<Self, String> {
        let (kind, count) = spec
            .split_once(':')
            .ok_or(format!("frame {spec:?} is not KIND:COUNT"))?;
        let retained: usize = count.parse().map_err(|error| format!("frame {spec:?}: {error}"))?;
        let latent_dim = block.input_dim();
        if retained == 0 || retained > latent_dim {
            return Err(format!("frame {spec:?} retains {retained} of {latent_dim} coordinates"));
        }
        let basis = match kind {
            "top" => {
                let mut basis = Array2::<f64>::zeros((latent_dim, retained));
                for coordinate in 0..retained {
                    basis[[coordinate, coordinate]] = 1.0;
                }
                basis
            }
            "bottom" => {
                if spectrum.variances.len() != latent_dim {
                    return Err(format!(
                        "the law declares {} coordinates but the block reads {latent_dim}",
                        spectrum.variances.len()
                    ));
                }
                let resolved = spectrum.resolved();
                if resolved.len() < retained {
                    return Err(format!(
                        "frame {spec:?} asks for {retained} coordinates but the law resolves {} from zero",
                        resolved.len()
                    ));
                }
                let mut basis = Array2::<f64>::zeros((latent_dim, retained));
                for (column, &coordinate) in resolved[resolved.len() - retained..].iter().enumerate() {
                    basis[[coordinate, column]] = 1.0;
                }
                basis
            }
            "readers" => {
                let (singular, right) = block
                    .readers()
                    .svd(false, true)
                    .map(|parts| (parts.1, parts.2))
                    .map_err(|error| format!("SVD of the readers: {error}"))?;
                let right = right.ok_or("the SVD returned no right singular vectors")?;
                let mut order: Vec<usize> = (0..singular.len()).collect();
                order.sort_by(|a, b| singular[*b].total_cmp(&singular[*a]));
                let mut basis = Array2::<f64>::zeros((latent_dim, retained));
                for (column, &index) in order[..retained].iter().enumerate() {
                    basis.column_mut(column).assign(&right.row(index));
                }
                basis
            }
            other => return Err(format!("frame kind {other:?} is none of top, bottom and readers")),
        };
        Ok(Self {
            name: spec.to_string(),
            basis,
        })
    }

    fn retained(&self) -> usize {
        self.basis.ncols()
    }

    /// `n` retained points `Q c`, `c ~ N(0, I_q)`, as rows.
    fn points(&self, count: usize, state: &mut u64) -> Result<Array2<f64>, String> {
        Ok(fast_abt(&standard_normal_rows(count, self.retained(), state)?, &self.basis))
    }

    /// `Q Qᵀ z + (I − Q Qᵀ) z̃` for each row: `retained` supplies `Q Qᵀ z` (rows in `ℝ^k`, or one row broadcast to
    /// every draw), `free` the draws `z̃`.
    fn couple(&self, retained: &Array2<f64>, free: &Array2<f64>) -> Array2<f64> {
        let mut rows = free - &fast_abt(&fast_ab(free, &self.basis), &self.basis);
        rows += retained;
        rows
    }

    /// `Q Qᵀ z` for each row `z`.
    fn project(&self, rows: &Array2<f64>) -> Array2<f64> {
        fast_abt(&fast_ab(rows, &self.basis), &self.basis)
    }
}

/// The declared composition: the law, both known blocks, the held stream and the frozen norm.
struct Declared {
    first: KnownBlock,
    second: KnownBlock,
    held_stream: Array1<f64>,
    frozen_scale: f64,
}

impl Declared {
    fn new(pair: &Pair, law: &Path) -> Result<Self, String> {
        let first = declared_block(&pair.first, law)?;
        let x_first = pair.rows("x_first", None)?;
        let y_first = pair.rows("y_first", None)?;
        let held_stream = (&x_first - &y_first).mean_axis(Axis(0)).ok_or("no exported rows")?;
        let mean_stream = x_first.mean_axis(Axis(0)).ok_or("no exported rows")?;
        let (centred_mean, frozen_scale) = pair.second_norm.centred_and_scale(mean_stream.view());
        let hidden = held_stream.len();
        // `LN(r̄ + y) = β + γ ⊙ C r̄ / s₀ + (diag(γ) C / s₀) y` with the scale frozen at `s₀`.
        let held_mean = held_stream.sum() / hidden as f64;
        let held_centred = held_stream.mapv(|value| value - held_mean);
        let norm_baseline = &pair.second_norm.bias + &(&pair.second_norm.weight * &held_centred / frozen_scale);
        let mut norm_loading = Array2::<f64>::from_elem((hidden, hidden), -1.0 / hidden as f64);
        norm_loading.diag_mut().mapv_inplace(|value| value + 1.0);
        for (mut row, &weight) in norm_loading.rows_mut().into_iter().zip(pair.second_norm.weight.iter()) {
            row *= weight / frozen_scale;
        }
        let second = pair
            .second
            .absorb(norm_baseline.view(), norm_loading.view())
            .map_err(|error| format!("second block behind the frozen norm: {error}"))?;
        println!(
            "[compose] law rank {} ; frozen scale s0 {frozen_scale:.6e} at the mean stream (|C x_bar| {:.6e})",
            first.input_dim(),
            euclidean(centred_mean.view())
        );
        Ok(Self {
            first,
            second,
            held_stream,
            frozen_scale,
        })
    }

    /// The first block's output `y` and the second stage's pre-activations `c` behind the frozen norm, at latent rows.
    fn preactivations(&self, latent: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
        let first_out = known_forward(&self.first, latent)?;
        let mut second_pre = fast_abt(&first_out, &self.second.readers());
        second_pre += &self.second.biases();
        Ok((first_out, second_pre))
    }
}

/// `F(z) = U σ(W z + b) + c` of a known block at latent rows.
fn known_forward(block: &KnownBlock, latent: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut pre = fast_abt(latent, &block.readers());
    pre += &block.biases();
    let units = activation_values(block.activation(), &pre)?;
    let mut out = fast_abt(&units, &block.writers());
    out += &block.output_bias();
    Ok(out)
}

/// A torch-layout block under the law `h = h₀ + L Z` of `law` (`h0.npy`, `L.npy`).
fn declared_block(block: &UnabsorbedBlock, law: &Path) -> Result<KnownBlock, String> {
    let baseline = read_vector(&law.join("h0.npy"))?;
    let loading = read_rows::<f64>(&law.join("L.npy"), None)?;
    block
        .absorb(baseline.view(), loading.view())
        .map_err(|error| format!("block under the declared law: {error}"))
}

/// The run's sizes: retained points per frame, Monte Carlo draws per point and per chunk, and coupled pairs.
struct ComposeSizes {
    points: usize,
    draws: usize,
    chunk: usize,
    pairs: usize,
}

fn compose(
    pair_dir: &Path,
    law: &Path,
    frames: &[String],
    sizes: &ComposeSizes,
    seed: u64,
    out: &Path,
) -> Result<(), String> {
    let pair = Pair::load(pair_dir)?;
    let started = Instant::now();
    let declared = Declared::new(&pair, law)?;
    let setup_seconds = started.elapsed().as_secs_f64();
    let latent_dim = declared.first.input_dim();
    let units = declared.second.width();
    let outputs = declared.second.output_dim();
    let spectrum = DeclaredSpectrum::read(law)?;
    let mut state = seed;
    let mut frame_reports = Vec::new();
    for spec in frames {
        let frame = LatentFrame::parse(spec, &declared.first, &spectrum)?;
        if frame.retained() == latent_dim {
            return Err(format!("frame {spec:?} retains everything: nothing is composed through a discarded law"));
        }
        let retained_points = frame.points(sizes.points, &mut state)?;

        let closure_started = Instant::now();
        let closure = gaussian_closure_response(
            &declared.first,
            &declared.second,
            frame.basis.view(),
            retained_points.view(),
        )
        .map_err(|error| format!("Gaussian closure: {error}"))?;
        let closure_seconds = closure_started.elapsed().as_secs_f64();

        let mean_started = Instant::now();
        let first_mean = declared
            .first
            .retained_response(frame.basis.view(), retained_points.view())
            .map_err(|error| format!("first block retained response: {error}"))?;
        let mut mean_pre = fast_abt(&first_mean, &declared.second.readers());
        mean_pre += &declared.second.biases();
        let mean_units = activation_values(declared.second.activation(), &mean_pre)?;
        let mut mean_composition = fast_abt(&mean_units, &declared.second.writers());
        mean_composition += &declared.second.output_bias();
        let mean_seconds = mean_started.elapsed().as_secs_f64();

        let pre_comparisons = 2 * sizes.points * units;
        let pre_multiple = comparison_multiple(pre_comparisons)?;
        let response_multiple = comparison_multiple(3 * sizes.points * outputs)?;
        let mut point_reports = Vec::new();
        let mut executed_seconds = 0.0;
        let mut worst_mean_z = 0.0_f64;
        let mut worst_variance_z = 0.0_f64;
        let mut worst_first_z = 0.0_f64;
        let mut resolved_variances = 0_usize;
        let mut rejected_zero_variances = 0_usize;
        for point in 0..sizes.points {
            let executed_started = Instant::now();
            let anchor = retained_points.row(point).insert_axis(Axis(0)).to_owned();
            let mut first = ColumnMoments::new();
            let mut pre = ColumnMoments::new();
            let mut frozen = ColumnMoments::new();
            let mut surgery = ColumnMoments::new();
            let mut remaining = sizes.draws;
            while remaining > 0 {
                let rows = remaining.min(sizes.chunk);
                remaining -= rows;
                let free = standard_normal_rows(rows, latent_dim, &mut state)?;
                let latent = frame.couple(&anchor, &free);
                let (first_out, second_pre) = declared.preactivations(&latent)?;
                let second_units = activation_values(declared.second.activation(), &second_pre)?;
                let mut frozen_out = fast_abt(&second_units, &declared.second.writers());
                frozen_out += &declared.second.output_bias();
                let stream = &first_out + &declared.held_stream;
                let true_out = mlp_forward(&pair.second, &pair.second_norm.apply(stream.view()))?;
                first.add(&first_out)?;
                pre.add(&second_pre)?;
                surgery.add(&(&true_out - &frozen_out))?;
                frozen.add(&frozen_out)?;
            }
            executed_seconds += executed_started.elapsed().as_secs_f64();
            let mean_se = pre.mean_se();
            let variance_se = pre.variance_se();
            let executed_variance = pre.variance();
            let exact_means = closure.preactivation_means.row(point);
            let exact_variances = closure.preactivation_variances.row(point);
            worst_first_z = worst_first_z.max(max_z(first_mean.row(point), first.mean().view(), first.mean_se().view()));
            let mean_z = max_z(exact_means, pre.mean().view(), mean_se.view());
            let variance_z = max_z(exact_variances, executed_variance.view(), variance_se.view());
            worst_mean_z = worst_mean_z.max(mean_z);
            worst_variance_z = worst_variance_z.max(variance_z);
            let resolved = Zip::from(&exact_variances)
                .and(&variance_se)
                .fold(0, |count, &variance, &se| count + usize::from(variance > pre_multiple * se));
            resolved_variances += resolved;
            rejected_zero_variances += Zip::from(&executed_variance)
                .and(&variance_se)
                .fold(0, |count, &variance, &se| count + usize::from(variance > pre_multiple * se));
            let frozen_mean = frozen.mean();
            let frozen_se = frozen.mean_se();
            let closure_row = closure.approximate_response.row(point);
            let closure_error = &closure_row - &frozen_mean;
            let mean_composition_error = &mean_composition.row(point) - &frozen_mean;
            let surgery_mean = surgery.mean();
            let surgery_se = surgery.mean_se();
            point_reports.push(json!({
                "point": point,
                "draws": frozen.count(),
                "preactivation_mean_max_z": mean_z,
                "preactivation_variance_max_z": variance_z,
                "preactivation_variance_mean_exact": exact_variances.mean(),
                "preactivation_variance_resolved_units": resolved,
                "closure_error_norm": euclidean(closure_error.view()),
                "closure_error_max_z": max_z(closure_row, frozen_mean.view(), frozen_se.view()),
                "mean_composition_error_norm": euclidean(mean_composition_error.view()),
                "mean_composition_error_max_z": max_z(mean_composition.row(point), frozen_mean.view(), frozen_se.view()),
                "executed_mean_se_norm": euclidean(frozen_se.view()),
                "conditional_output_sd_norm": frozen.variance().sum().sqrt(),
                "retained_signal_norm": euclidean((&frozen_mean - &declared.second.output_bias()).view()),
                "frozen_norm_surgery_error_norm": euclidean(surgery_mean.view()),
                "frozen_norm_surgery_se_norm": euclidean(surgery_se.view()),
                "frozen_norm_surgery_max_z": max_z(surgery_mean.view(), Array1::<f64>::zeros(outputs).view(), surgery_se.view()),
            }));
        }

        // `E(P)` and `V(I)` of the second stage's pre-activations `c = A y + a`: exact through the affine stage, and by
        // the coupled Monte Carlo `E|c(Z) − c(Z')|²/2`, `Z' = Q Qᵀ Z + (I − Q Qᵀ) Z̃`, and `E|c(Z) − c(Z̃)|²/2`.
        let affine_started = Instant::now();
        let composed = compose_affine_stage(
            &declared.first,
            declared.second.readers(),
            declared.second.biases(),
            Array2::<f64>::eye(units).view(),
        )
        .map_err(|error| format!("affine stage: {error}"))?;
        let exact_total = composed.total_variance().value;
        let exact_discarded = composed
            .discarded_error(frame.basis.view())
            .map_err(|error| format!("affine stage discarded error: {error}"))?
            .value;
        let affine_seconds = affine_started.elapsed().as_secs_f64();
        let coupled_started = Instant::now();
        let base = standard_normal_rows(sizes.pairs, latent_dim, &mut state)?;
        let independent = standard_normal_rows(sizes.pairs, latent_dim, &mut state)?;
        let coupled = frame.couple(&frame.project(&base), &standard_normal_rows(sizes.pairs, latent_dim, &mut state)?);
        let base_pre = declared.preactivations(&base)?.1;
        let half_squared = |other: &Array2<f64>| -> Vec<f64> {
            (&base_pre - other)
                .rows()
                .into_iter()
                .map(|row| 0.5 * row.dot(&row))
                .collect()
        };
        let discarded_samples = half_squared(&declared.preactivations(&coupled)?.1);
        let total_samples = half_squared(&declared.preactivations(&independent)?.1);
        let coupled_seconds = coupled_started.elapsed().as_secs_f64();
        let (discarded_mc, discarded_se) = mean_with_se(&discarded_samples);
        let (total_mc, total_se) = mean_with_se(&total_samples);
        let energy_multiple = comparison_multiple(2)?;

        frame_reports.push(json!({
            "frame": frame.name,
            "retained": frame.retained(),
            "points": sizes.points,
            "draws_per_point": sizes.draws,
            "preactivations": {
                "units": units,
                "comparisons": pre_comparisons,
                "gate_multiple": pre_multiple,
                "first_block_retained_response_max_z": worst_first_z,
                "exact_mean_max_z": worst_mean_z,
                "exact_variance_max_z": worst_variance_z,
                "exact_means_agree": worst_mean_z <= pre_multiple,
                "exact_variances_agree": worst_variance_z <= pre_multiple,
                "magnitude_floor": {
                    "exact_variances_resolved_from_zero": resolved_variances,
                    "executed_variances_rejecting_the_mean_composition_zero": rejected_zero_variances,
                    "of": sizes.points * units,
                },
            },
            "response": {
                "outputs": outputs,
                "gate_multiple": response_multiple,
                "label": "APPROXIMATE: Gaussian closure of a GELU second stage; its error is measured, never bounded",
                "per_point": point_reports,
            },
            "energies": {
                "exact_total_variance": exact_total,
                "exact_discarded_error": exact_discarded,
                "mc_total_variance": total_mc,
                "mc_total_variance_se": total_se,
                "mc_discarded_error": discarded_mc,
                "mc_discarded_error_se": discarded_se,
                "pairs": sizes.pairs,
                "gate_multiple": energy_multiple,
                "total_z": (exact_total - total_mc).abs() / total_se,
                "discarded_z": (exact_discarded - discarded_mc).abs() / discarded_se,
                "discarded_resolved_from_zero": discarded_mc > energy_multiple * discarded_se,
            },
            "seconds": {
                "gaussian_closure": closure_seconds,
                "mean_composition": mean_seconds,
                "executed_layer_by_layer_mc": executed_seconds,
                "affine_stage_exact_energies": affine_seconds,
                "coupled_mc_energies": coupled_seconds,
            },
        }));
        println!(
            "[compose] {spec}: pre-activation mean max z {worst_mean_z:.3}, variance max z {worst_variance_z:.3} (gate {pre_multiple:.3}); E(P) exact {exact_discarded:.6e} vs mc {discarded_mc:.6e} +- {discarded_se:.2e}; closure {closure_seconds:.2}s, mc {executed_seconds:.2}s"
        );
    }
    let report = json!({
        "stage": "compose",
        "provenance": pair.provenance(),
        "law": read_json(&law.join("law.json"))?,
        "declared": {
            "law": "h = h0 + L Z, Z ~ N(0, I_k), at the first MLP's post-norm input; L = V Lambda^{1/2} of the post-norm covariance, so top:q is the first q axes of Z",
            "held_stream": "r = x_first - y_first held at its row mean",
            "second_stage": "SURGERY: MLP_{l+1}(beta + gamma * C (r_bar + y) / s0), the norm's scale frozen at the mean stream; its error against the true norm is reported per point",
            "frozen_scale": declared.frozen_scale,
        },
        "seed": seed,
        "setup_seconds": setup_seconds,
        "frames": frame_reports,
        "seconds": started.elapsed().as_secs_f64(),
    });
    write_json(out, &report)?;
    Ok(())
}

/// A3's sizes: rows of `Z`, `Z~` and each coupled `Z'`, retained points, and draws per point.
struct A3Sizes {
    rows: usize,
    points: usize,
    point_draws: usize,
}

/// The pair's `first` or `second` MLP.
fn pair_block<'a>(pair: &'a Pair, name: &str) -> Result<&'a UnabsorbedBlock, String> {
    match name {
        "first" => Ok(&pair.first),
        "second" => Ok(&pair.second),
        other => Err(format!("--block {other:?} is neither first nor second")),
    }
}

/// A3, stage 1: the rows `h = h₀ + L z` torch executes, stacked in named blocks. `z` and `z_tilde` are independent
/// draws of the law; each frame's `coupled` block is `Q Qᵀ Z + (I − Q Qᵀ) Z~` with the same `Z~` (so `top:k` gives
/// `Z' = Z` exactly, the row-alignment control); each retained point of the first frame has its own block of draws
/// of the discarded coordinates. Writes `h.npy` (float64), `points.npy` and `draw.json`.
fn a3_draw(
    pair_dir: &Path,
    law: &Path,
    block_name: &str,
    frames: &[String],
    sizes: &A3Sizes,
    seed: u64,
    out: &Path,
) -> Result<(), String> {
    let pair = Pair::load(pair_dir)?;
    let known = declared_block(pair_block(&pair, block_name)?, law)?;
    let baseline = read_vector(&law.join("h0.npy"))?;
    let loading = read_rows::<f64>(&law.join("L.npy"), None)?;
    let latent_dim = known.input_dim();
    let spectrum = DeclaredSpectrum::read(law)?;
    let frames = frames
        .iter()
        .map(|spec| LatentFrame::parse(spec, &known, &spectrum))
        .collect::<Result<Vec<_>, String>>()?;
    let points_frame = frames.first().ok_or("--frames names no frame")?;
    let mut state = seed;
    let z = standard_normal_rows(sizes.rows, latent_dim, &mut state)?;
    let z_tilde = standard_normal_rows(sizes.rows, latent_dim, &mut state)?;
    let mut blocks: Vec<(String, Array2<f64>)> = vec![("z".to_string(), z.clone()), ("z_tilde".to_string(), z_tilde.clone())];
    for frame in &frames {
        blocks.push((format!("coupled:{}", frame.name), frame.couple(&frame.project(&z), &z_tilde)));
    }
    let points = points_frame.points(sizes.points, &mut state)?;
    for point in 0..sizes.points {
        let anchor = points.row(point).insert_axis(Axis(0)).to_owned();
        let free = standard_normal_rows(sizes.point_draws, latent_dim, &mut state)?;
        blocks.push((format!("point:{point}"), points_frame.couple(&anchor, &free)));
    }
    std::fs::create_dir_all(out).map_err(|err| format!("create {}: {err}", out.display()))?;
    let total: usize = blocks.iter().map(|block| block.1.nrows()).sum();
    let hidden = baseline.len();
    let mut layout = Vec::new();
    let mut h_rows = Array2::<f64>::zeros((total, hidden));
    let mut start = 0;
    for (name, latent) in &blocks {
        let mut h = fast_abt(latent, &loading);
        h += &baseline;
        h_rows.slice_mut(s![start..start + latent.nrows(), ..]).assign(&h);
        layout.push(json!({"name": name, "start": start, "rows": latent.nrows()}));
        start += latent.nrows();
    }
    write_npy(&out.join("h.npy"), &[total as u64, hidden as u64], h_rows.iter().copied())?;
    write_npy(&out.join("points.npy"), &[sizes.points as u64, latent_dim as u64], points.iter().copied())?;
    write_npy(&out.join("z.npy"), &[sizes.rows as u64, latent_dim as u64], z.iter().copied())?;
    write_json(
        &out.join("draw.json"),
        &json!({
            "stage": "a3-draw",
            "provenance": pair.provenance(),
            "block": block_name,
            "law": law.display().to_string(),
            "frames": frames.iter().map(|frame| frame.name.clone()).collect::<Vec<_>>(),
            "points_frame": points_frame.name,
            "points": sizes.points,
            "point_draws": sizes.point_draws,
            "rows": sizes.rows,
            "seed": seed,
            "blocks": layout,
            "total_rows": total,
        }),
    )?;
    println!("[a3-draw] {total} rows of h = h0 + L z for block {block_name}");
    Ok(())
}

/// A3, stage 2: the analytic operator against the executed block.
fn a3(pair_dir: &Path, law: &Path, draw_dir: &Path, executed_path: &Path, out: &Path) -> Result<(), String> {
    let layout = read_json(&draw_dir.join("draw.json"))?;
    let block_name = layout
        .get("block")
        .and_then(Value::as_str)
        .ok_or("draw.json names no block")?
        .to_string();
    let pair = Pair::load(pair_dir)?;
    let raw = pair_block(&pair, &block_name)?;
    let construction_started = Instant::now();
    let known = declared_block(raw, law)?;
    let construction_seconds = construction_started.elapsed().as_secs_f64();
    let executed = read_rows::<f64>(executed_path, None)?;
    let block_rows = |name: &str| layout_rows(&layout, &executed, name);

    // The executed rows are this block's function: this file's float64 forward pass on the leading drawn rows.
    let forward_rows = 256.min(executed.nrows());
    let h_rows = read_rows::<f64>(&draw_dir.join("h.npy"), Some(forward_rows))?;
    let (forward_value, forward_bound) = mlp_forward_with_bound(raw, &h_rows, f64::EPSILON / 2.0)?;
    let forward = gap_report(&forward_value, &executed.slice(s![..forward_rows, ..]).to_owned(), &forward_bound);

    let z = block_rows("z")?;
    let z_tilde = block_rows("z_tilde")?;
    let total_samples = half_squared_distances(z, z_tilde);
    let (total_mc, total_se) = mean_with_se(&total_samples);
    let frame_specs = layout
        .get("frames")
        .and_then(Value::as_array)
        .ok_or("draw.json names no frames")?
        .iter()
        .map(|spec| spec.as_str().map(str::to_string).ok_or("a frame spec is not a string".to_string()))
        .collect::<Result<Vec<String>, String>>()?;
    let comparisons = 1 + 4 * frame_specs.len();
    let multiple = comparison_multiple(comparisons)?;
    let latent = read_rows::<f64>(&draw_dir.join("z.npy"), None)?;
    let spectrum = DeclaredSpectrum::read(law)?;
    let mut frame_reports = Vec::new();
    for spec in &frame_specs {
        let frame = LatentFrame::parse(spec, &known, &spectrum)?;
        let coupled = block_rows(&format!("coupled:{spec}"))?;
        let discarded_samples = half_squared_distances(z, coupled);
        let explained_samples: Vec<f64> = total_samples
            .iter()
            .zip(discarded_samples.iter())
            .map(|(total, discarded)| total - discarded)
            .collect();
        let (discarded_mc, discarded_se) = mean_with_se(&discarded_samples);
        let (explained_mc, explained_se) = mean_with_se(&explained_samples);
        let analytic_started = Instant::now();
        let discarded = known
            .discarded_error(frame.basis.view())
            .map_err(|error| format!("E(P) at {spec}: {error}"))?
            .value;
        let explained = known
            .explained_variance(frame.basis.view())
            .map_err(|error| format!("V(P) at {spec}: {error}"))?
            .value;
        let analytic_seconds = analytic_started.elapsed().as_secs_f64();
        // R2 over the whole law: `D = |F(Z) − g(P Z)|² − |F(Z) − F(Z')|²/2` has `E D = E|F̄_P − g|²`, which is 0 for
        // `g = F̄_P` and positive for the plug-in `g₀ = F(P Z)` (the positive control).
        let projected = frame.project(&latent);
        let r2_started = Instant::now();
        let best = known
            .retained_response(frame.basis.view(), projected.view())
            .map_err(|error| format!("retained response at {spec}: {error}"))?;
        let r2_seconds = r2_started.elapsed().as_secs_f64();
        let plug_in = known_forward(&known, &projected)?;
        let split_samples = |response: &Array2<f64>| -> Vec<f64> {
            (&z - response)
                .rows()
                .into_iter()
                .zip(discarded_samples.iter())
                .map(|(row, discarded)| row.dot(&row) - discarded)
                .collect()
        };
        let (best_split, best_split_se) = mean_with_se(&split_samples(&best));
        let (plug_in_split, plug_in_split_se) = mean_with_se(&split_samples(&plug_in));
        frame_reports.push(json!({
            "frame": spec,
            "r2_best_response_excess": best_split,
            "r2_best_response_excess_se": best_split_se,
            "r2_best_response_z": best_split.abs() / best_split_se,
            "r2_plug_in_excess": plug_in_split,
            "r2_plug_in_excess_se": plug_in_split_se,
            "r2_plug_in_resolved_from_zero": plug_in_split > multiple * plug_in_split_se,
            "r2_retained_response_seconds": r2_seconds,
            "retained": frame.retained(),
            "analytic_discarded_error": discarded,
            "analytic_explained_variance": explained,
            "mc_discarded_error": discarded_mc,
            "mc_discarded_error_se": discarded_se,
            "mc_explained_variance": explained_mc,
            "mc_explained_variance_se": explained_se,
            "discarded_z": (discarded - discarded_mc).abs() / discarded_se,
            "explained_z": (explained - explained_mc).abs() / explained_se,
            "discarded_resolved_from_zero": discarded_mc > multiple * discarded_se,
            "explained_resolved_from_zero": explained_mc > multiple * explained_se,
            "max_executed_discarded_sample": discarded_samples.iter().copied().fold(0.0_f64, f64::max),
            "analytic_seconds": analytic_seconds,
        }));
        println!(
            "[a3] {spec}: E(P) analytic {discarded:.6e} mc {discarded_mc:.6e} +- {discarded_se:.2e}; V(P) analytic {explained:.6e} mc {explained_mc:.6e} +- {explained_se:.2e}"
        );
    }

    // R1 at retained points of the first frame: the analytic best retained response against each point's executed mean.
    let points_frame = LatentFrame::parse(
        layout
            .get("points_frame")
            .and_then(Value::as_str)
            .ok_or("draw.json names no points frame")?,
        &known,
        &spectrum,
    )?;
    let points = read_rows::<f64>(&draw_dir.join("points.npy"), None)?;
    let response_started = Instant::now();
    let retained_response = known
        .retained_response(points_frame.basis.view(), points.view())
        .map_err(|error| format!("retained response: {error}"))?;
    let response_seconds = response_started.elapsed().as_secs_f64();
    let unconditional = z.mean_axis(Axis(0)).ok_or("no z rows")?;
    let response_multiple = comparison_multiple(2 * points.nrows() * known.output_dim())?;
    let mut point_reports = Vec::new();
    for point in 0..points.nrows() {
        let mut moments = ColumnMoments::new();
        moments.add(&block_rows(&format!("point:{point}"))?.to_owned())?;
        let mean = moments.mean();
        let se = moments.mean_se();
        let analytic = retained_response.row(point);
        point_reports.push(json!({
            "point": point,
            "draws": moments.count(),
            "max_z": max_z(analytic, mean.view(), se.view()),
            "error_norm": euclidean((&analytic - &mean).view()),
            "executed_mean_se_norm": euclidean(se.view()),
            "shift_from_unconditional_mean_norm": euclidean((&mean - &unconditional).view()),
            "shift_from_unconditional_mean_max_z": max_z(mean.view(), unconditional.view(), se.view()),
        }));
    }

    let report = json!({
        "stage": "a3",
        "provenance": pair.provenance(),
        "law": read_json(&law.join("law.json"))?,
        "draw": layout,
        "block": block_name,
        "forward_check_float64": forward,
        "analytic_total_variance": known.total_variance().value,
        "mc_total_variance": total_mc,
        "mc_total_variance_se": total_se,
        "total_z": (known.total_variance().value - total_mc).abs() / total_se,
        "gate_multiple": multiple,
        "comparisons": comparisons,
        "frames": frame_reports,
        "retained_response": {
            "frame": points_frame.name,
            "gate_multiple": response_multiple,
            "per_point": point_reports,
            "analytic_seconds": response_seconds,
        },
        "seconds": {
            "block_construction_with_total_variance": construction_seconds,
        },
    });
    write_json(out, &report)?;
    println!(
        "[a3] V(I) analytic {:.6e} mc {total_mc:.6e} +- {total_se:.2e}",
        known.total_variance().value
    );
    Ok(())
}

/// The compile's declared experiment: draws the compact GAM is fitted on, independent draws for its function error,
/// and the seed of the stream both come from.
struct CompileRun {
    training_draws: usize,
    holdout_draws: usize,
    seed: u64,
}

/// `‖r_i‖²_M` for each row `r_i` of `rows`.
fn metric_squared_norms(rows: &Array2<f64>, metric: ArrayView2<'_, f64>) -> Vec<f64> {
    let weighted = fast_ab(rows, &metric);
    rows.rows()
        .into_iter()
        .zip(weighted.rows())
        .map(|(row, weighted_row)| row.dot(&weighted_row))
        .collect()
}

/// `frame` with its last direction replaced by the unit residual of the latent axis farthest from its span (the first
/// such axis on a tie): orthonormal, and a different retained frame. Some axis is at squared distance at least
/// `1 − q/d` from a rank-`q` span, since the axes' squared projections onto it sum to `q`.
fn perturbed_frame(frame: &LatentFrame) -> Result<Array2<f64>, String> {
    let (latent_dim, retained) = frame.basis.dim();
    if retained >= latent_dim {
        return Err(format!("frame {} spans every latent axis; no axis lies outside it", frame.name));
    }
    let (axis, _) = frame
        .basis
        .rows()
        .into_iter()
        .map(|row| 1.0 - row.dot(&row))
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (axis, distance)| {
            if distance > best.1 { (axis, distance) } else { best }
        });
    // `e_a − Q Qᵀ e_a = e_a − Q (row a of Q)ᵀ`.
    let mut residual = frame.basis.dot(&frame.basis.row(axis)).mapv(|value| -value);
    residual[axis] += 1.0;
    let norm = euclidean(residual.view());
    let mut basis = frame.basis.clone();
    basis.column_mut(retained - 1).assign(&(residual / norm));
    Ok(basis)
}

/// One estimate of the function error `A` against the compile's held-out `Â`: its Bonferroni `z` and whether it is
/// resolved from zero at the same multiple (an agreement of two unresolved zeros carries no information).
fn function_error_report(samples: &[f64], compiled: &CompiledResponse, multiple: f64) -> Value {
    let split = compiled.split();
    let (mean, se) = mean_with_se(samples);
    let z = (mean - split.function_error).abs() / se.hypot(split.function_error_standard_error);
    json!({
        "mean": mean,
        "se": se,
        "z_against_holdout": z,
        "agrees": z <= multiple,
        "resolved_from_zero": mean > multiple * se,
    })
}

/// A7: the teacher receipt of the compact GAM on the executed block.
///
/// For each frame, `compile_retained_response` fits `g` to the analytic best retained response `F̄_P` and reports R2's
/// split `E‖F − g(PZ)‖²_M = E(P) + A`: `E(P)` exact from the operator, `Â` the held-out mean of `‖F̄_P − g‖²_M` on the
/// compile's own draws. The A3 draw's rows were executed by torch, so on them the split is measured from executed
/// outputs: `T = ‖F(Z) − g(Qᵀ Z)‖²_M` has `E T = E(P) + A`; `F(Z)` and `F(Z')` are two draws of one conditional law, so
/// `E‖F(Z) − F(Z')‖²_M = 2 E(P)` and the paired `D = T − ‖F(Z) − F(Z')‖²_M / 2` has `E D = A` with no operator in it.
/// `D' = T − ‖F(Z) − F̄_P(P Z)‖²_M = ‖F̄_P − g‖²_M + 2⟨F − F̄_P, F̄_P − g⟩_M` cancels the discarded part row by row,
/// so its standard error does not carry `Var ‖F − F̄_P‖²`; `E D' = A` holds when `F̄_P` is the executed conditional
/// mean, which A3 tests on the same rows. The negative control reads the same `g` through a deliberately wrong frame,
/// the frame's last direction swapped for the first latent axis outside its span ([`perturbed_frame`]): a wrong
/// retained response, whose executed total must miss `E(P) + Â` at the same multiple. The blind control is the frame
/// kind `bottom:q`, run beside the others: the least-variance coordinates the law resolves carry almost none of the
/// response, so its split is nearly all `E(P)`.
fn a7(
    pair_dir: &Path,
    law: &Path,
    draw_dir: &Path,
    executed_path: &Path,
    frames: &[String],
    run: CompileRun,
    out: &Path,
) -> Result<(), String> {
    let layout = read_json(&draw_dir.join("draw.json"))?;
    let block_name = layout
        .get("block")
        .and_then(Value::as_str)
        .ok_or("draw.json names no block")?
        .to_string();
    let drawn_frames: Vec<&str> = layout
        .get("frames")
        .and_then(Value::as_array)
        .ok_or("draw.json names no frames")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let pair = Pair::load(pair_dir)?;
    let known = declared_block(pair_block(&pair, &block_name)?, law)?;
    let spectrum = DeclaredSpectrum::read(law)?;
    let executed = read_rows::<f64>(executed_path, None)?;
    let block_rows = |name: &str| layout_rows(&layout, &executed, name);
    let latent = read_rows::<f64>(&draw_dir.join("z.npy"), None)?;
    let at_draw = block_rows("z")?.to_owned();
    if latent.nrows() != at_draw.nrows() {
        return Err(format!(
            "z.npy has {} rows but the executed z block has {}",
            latent.nrows(),
            at_draw.nrows()
        ));
    }
    let metric = known.metric();
    // Per frame: T against E(P) + Â, D and D' against Â, and the control's T and D'.
    let comparisons = 5 * frames.len();
    let multiple = comparison_multiple(comparisons)?;
    let mut frame_reports = Vec::new();
    for spec in frames {
        if !drawn_frames.contains(&spec.as_str()) {
            return Err(format!("frame {spec} is not one of the A3 draw's frames {drawn_frames:?}"));
        }
        let frame = LatentFrame::parse(spec, &known, &spectrum)?;
        let coupled = block_rows(&format!("coupled:{spec}"))?;
        let design = CompileDesign::pilot(run.training_draws, run.holdout_draws, frame.retained());
        let compile_started = Instant::now();
        let compiled = compile_retained_response(&known, frame.basis.view(), design, run.seed)
            .map_err(|error| format!("compile at {spec}: {error}"))?;
        let compile_seconds = compile_started.elapsed().as_secs_f64();
        let split = compiled.split();

        let compiled_at = compiled
            .evaluate_coordinates(fast_ab(&latent, &frame.basis).view())
            .map_err(|error| format!("compiled response at {spec}: {error}"))?;
        let perturbed = perturbed_frame(&frame)?;
        let perturbed_at = compiled
            .evaluate_coordinates(fast_ab(&latent, &perturbed).view())
            .map_err(|error| format!("compiled response through the perturbed frame at {spec}: {error}"))?;
        let best = known
            .retained_response(frame.basis.view(), frame.project(&latent).view())
            .map_err(|error| format!("retained response at {spec}: {error}"))?;

        let discarded: Vec<f64> = metric_squared_norms(&(&at_draw - &coupled), metric)
            .into_iter()
            .map(|value| 0.5 * value)
            .collect();
        let residual_to_best = metric_squared_norms(&(&at_draw - &best), metric);
        let executed_split = |fitted: &Array2<f64>| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let total = metric_squared_norms(&(&at_draw - fitted), metric);
            let paired = total.iter().zip(&discarded).map(|(t, d)| t - d).collect();
            let cancelled = total.iter().zip(&residual_to_best).map(|(t, r)| t - r).collect();
            (total, paired, cancelled)
        };
        let (total, paired, cancelled) = executed_split(&compiled_at);
        let (control_total, _, control_cancelled) = executed_split(&perturbed_at);
        let (total_mc, total_se) = mean_with_se(&total);
        let (control_total_mc, control_total_se) = mean_with_se(&control_total);
        let (discarded_mc, discarded_se) = mean_with_se(&discarded);
        let total_z = (total_mc - split.total()).abs() / total_se.hypot(split.function_error_standard_error);
        let control_total_z =
            (control_total_mc - split.total()).abs() / control_total_se.hypot(split.function_error_standard_error);

        let dominance = compiled.dominance_call();
        let price = compiled.price();
        let representation = match compiled.representation() {
            FunctionRepresentation::Constant => json!({"kind": "Constant"}),
            FunctionRepresentation::Reml {
                smoothing_parameters,
                placement,
                smoother_edf,
            } => json!({
                "kind": "Reml",
                "smoothing_parameters": smoothing_parameters,
                "placement": placement.iter().map(|place| format!("{place:?}")).collect::<Vec<_>>(),
                "smoother_edf": smoother_edf,
            }),
        };
        frame_reports.push(json!({
            "frame": spec,
            "retained": frame.retained(),
            "design": {
                "training_draws": design.training_draws,
                "holdout_draws": design.holdout_draws,
                "centers": design.centers,
            },
            "compile_seconds": compile_seconds,
            "representation": representation,
            "fitted_directions": compiled.fitted_directions(),
            "split": {
                "discarded_error_exact": split.discarded_error,
                "function_error_holdout": split.function_error,
                "function_error_holdout_se": split.function_error_standard_error,
                "function_error_resolved_from_zero": split.function_error > multiple * split.function_error_standard_error,
                "total": split.total(),
                "discarded_fraction": split.discarded_error / split.total(),
            },
            "dominance_call": {
                "action": format!("{:?}", dominance.action),
                "reversal_probability": dominance.reversal_probability,
            },
            "frame_step": compiled.frame_step().map(|step| json!({
                "discarded_error_after": step.discarded_error_after,
                "gain": step.gain,
            })),
            "price": {
                "reader_frame": price.reader_frame,
                "writer_frame": price.writer_frame,
                "output_mean": price.output_mean,
                "function_coefficients": price.function_coefficients,
                "function_edf": price.function_edf,
                "connections": price.connections,
                "unexplained_error": price.residual.unexplained_error,
                "executed_parameters": price.residual.executed_parameters,
            },
            "executed": {
                "rows": total.len(),
                "discarded_error": discarded_mc,
                "discarded_error_se": discarded_se,
                "total": total_mc,
                "total_se": total_se,
                "total_z_against_split": total_z,
                "total_agrees": total_z <= multiple,
                "function_error_paired": function_error_report(&paired, &compiled, multiple),
                "function_error_cancelled": function_error_report(&cancelled, &compiled, multiple),
            },
            "control_perturbed_frame": {
                "total": control_total_mc,
                "total_se": control_total_se,
                "total_z_against_split": control_total_z,
                "total_rejected": control_total_z > multiple,
                "function_error_cancelled": function_error_report(&control_cancelled, &compiled, multiple),
            },
        }));
        println!(
            "[a7] {spec}: E(P) {:.6e}; A-hat {:.6e} +- {:.2e}; executed total {total_mc:.6e} +- {total_se:.2e} (z {total_z:.2}); perturbed-frame control z {control_total_z:.2}; compile {compile_seconds:.1}s",
            split.discarded_error, split.function_error, split.function_error_standard_error
        );
    }
    write_json(
        out,
        &json!({
            "stage": "a7",
            "provenance": pair.provenance(),
            "law": read_json(&law.join("law.json"))?,
            "law_root_band": spectrum.root_band,
            "draw": layout,
            "block": block_name,
            "seed": run.seed,
            "gate_multiple": multiple,
            "comparisons": comparisons,
            "frames": frame_reports,
        }),
    )?;
    Ok(())
}

/// The rows of the named block of `layout` inside `executed`.
fn layout_rows<'a>(layout: &Value, executed: &'a Array2<f64>, name: &str) -> Result<ArrayView2<'a, f64>, String> {
    let entry = layout
        .get("blocks")
        .and_then(Value::as_array)
        .and_then(|blocks| blocks.iter().find(|entry| entry.get("name").and_then(Value::as_str) == Some(name)))
        .ok_or(format!("draw.json has no block {name}"))?;
    let start = entry.get("start").and_then(Value::as_u64).ok_or("block without start")? as usize;
    let rows = entry.get("rows").and_then(Value::as_u64).ok_or("block without rows")? as usize;
    if start + rows > executed.nrows() {
        return Err(format!("block {name} ends at row {} of {}", start + rows, executed.nrows()));
    }
    Ok(executed.slice(s![start..start + rows, ..]))
}

/// `|a_i − b_i|² / 2` for each row pair.
fn half_squared_distances(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Vec<f64> {
    (&a - &b).rows().into_iter().map(|row| 0.5 * row.dot(&row)).collect()
}

fn write_npy(path: &Path, shape: &[u64], values: impl IntoIterator<Item = f64>) -> Result<(), String> {
    let file = File::create(path).map_err(|err| format!("create {}: {err}", path.display()))?;
    let mut writer = npyz::WriteOptions::<f64>::new()
        .default_dtype()
        .shape(shape)
        .writer(BufWriter::new(file))
        .begin_nd()
        .map_err(|err| format!("begin {}: {err}", path.display()))?;
    writer
        .extend(values)
        .map_err(|err| format!("write {}: {err}", path.display()))?;
    writer
        .finish()
        .map_err(|err| format!("finish {}: {err}", path.display()))
}

/// `J = I + U diag(σ'(c)) W J_LN(x)` of `x ↦ x + MLP(LN(x))` at the stream `x`, with
/// `J_LN(x) = diag(γ) (C − x̂ x̂ᵀ / D) / s`, `x̂ = C x / s`.
fn transition_jacobian(block: &UnabsorbedBlock, norm: &LayerNorm, stream: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
    let hidden = stream.len();
    let (centred, scale) = norm.centred_and_scale(stream);
    let normalised = &centred / scale;
    let normed = &normalised * &norm.weight + &norm.bias;
    let mut pre = block.readers.dot(&normed);
    pre += &block.biases;
    let slopes = activation_jet(block.activation, &pre.insert_axis(Axis(0)), 2)?
        .pop()
        .ok_or("an order-2 jet holds the slope")?;
    let mut scaled_readers = &block.readers * &slopes.row(0).insert_axis(Axis(1));
    scaled_readers *= &norm.weight;
    // `B (C − x̂ x̂ᵀ/D) / s` with `B C = B − (B 1) 1ᵀ / D`.
    let row_sums = scaled_readers.sum_axis(Axis(1));
    let along = scaled_readers.dot(&normalised);
    let mut through_norm = scaled_readers;
    for ((mut row, &row_sum), &projection) in through_norm.rows_mut().into_iter().zip(row_sums.iter()).zip(along.iter()) {
        row -= &(&normalised * (projection / hidden as f64));
        row -= row_sum / hidden as f64;
        row /= scale;
    }
    let mut jacobian = fast_ab(&block.writers, &through_norm);
    jacobian.diag_mut().mapv_inplace(|value| value + 1.0);
    Ok(jacobian)
}

fn blocks(pair_dir: &Path, maps: usize, seed: u64, out: &Path) -> Result<(), String> {
    let pair = Pair::load(pair_dir)?;
    if pair.second.activation != GaussianActivation::ExactGelu {
        return Err("the transition Jacobian reads the exact GELU slope".to_string());
    }
    let streams = pair.rows("x_first", None)?;
    let (count, hidden) = streams.dim();
    if maps == 0 || maps > count {
        return Err(format!("--maps {maps} outside 1..={count}"));
    }
    let started = Instant::now();
    let jacobians = (0..maps)
        .map(|index| transition_jacobian(&pair.second, &pair.second_norm, streams.row(index * (count / maps))))
        .collect::<Result<Vec<_>, String>>()?;
    let jacobian_seconds = started.elapsed().as_secs_f64();
    let ones = Array1::<f64>::ones(hidden);
    let line_residual = jacobians
        .iter()
        .map(|jacobian| euclidean((&jacobian.dot(&ones) - &ones).view()) / euclidean(ones.view()))
        .fold(0.0_f64, f64::max);
    let transposed_line_residual = jacobians
        .iter()
        .map(|jacobian| euclidean((&jacobian.t().dot(&ones) - &ones).view()) / euclidean(ones.view()))
        .fold(f64::INFINITY, f64::min);

    let decompose_started = Instant::now();
    let decomposition = StateBlocks::decompose(jacobians.clone()).map_err(|error| format!("state blocks: {error}"))?;
    let decompose_seconds = decompose_started.elapsed().as_secs_f64();
    let describe = |decomposition: &StateBlocks| -> Value {
        Value::Array(
            decomposition
                .components
                .iter()
                .map(|component| {
                    json!({
                        "dimension": component.dimension(),
                        "block_dimension": component.block_dimension,
                        "multiplicity": component.multiplicity,
                        "division_algebra": format!("{:?}", component.division_algebra),
                        "angle_bound": component.angle_bound,
                        "block_basis": format!("{:?}", component.block_basis()),
                    })
                })
                .collect(),
        )
    };

    // Positive control: the same Jacobians' two diagonal 256-blocks, planted in a random orthogonal frame.
    let half = hidden / 2;
    let mut state = seed;
    let mut gaussian = Array2::<f64>::zeros((hidden, hidden));
    for value in gaussian.iter_mut() {
        *value = standard_normal(&mut state)?;
    }
    let frame = gaussian
        .svd(true, false)
        .map_err(|error| format!("SVD of the planted frame: {error}"))?
        .0
        .ok_or("the SVD returned no left singular vectors")?;
    let planted = jacobians
        .iter()
        .map(|jacobian| {
            let mut block_diagonal = Array2::<f64>::zeros((hidden, hidden));
            block_diagonal
                .slice_mut(s![..half, ..half])
                .assign(&jacobian.slice(s![..half, ..half]));
            block_diagonal
                .slice_mut(s![half.., half..])
                .assign(&jacobian.slice(s![half.., half..]));
            fast_abt(&fast_ab(&frame, &block_diagonal), &frame)
        })
        .collect::<Vec<_>>();
    let planted_started = Instant::now();
    let recovered = StateBlocks::decompose(planted).map_err(|error| format!("planted state blocks: {error}"))?;
    let planted_seconds = planted_started.elapsed().as_secs_f64();
    let planted_distances = recovered
        .components
        .iter()
        .map(|component| {
            [frame.slice(s![.., ..half]), frame.slice(s![.., half..])]
                .iter()
                .map(|span| projector_distance(component.basis.view(), *span))
                .collect::<Result<Vec<f64>, String>>()
                .map(|distances| distances.into_iter().fold(f64::INFINITY, f64::min))
        })
        .collect::<Result<Vec<f64>, String>>()?;

    let report = json!({
        "stage": "blocks",
        "provenance": pair.provenance(),
        "maps": maps,
        "rows": (0..maps).map(|index| index * (count / maps)).collect::<Vec<_>>(),
        "transition": "x -> x + MLP_{l+1}(LN_{l+1}(x)), the true norm, at the exported x_first rows",
        "components": describe(&decomposition),
        "commutant_dimension": decomposition.commutant_dimension(),
        "ones_line": {
            "max_relative_residual_J_1_minus_1": line_residual,
            "min_relative_residual_Jt_1_minus_1": transposed_line_residual,
            "reading": "J 1 = 1 for every map, so the ones line is invariant under every map; J^T 1 != 1, so it is not invariant under the *-algebra and is no orthogonal state block",
        },
        "planted_control": {
            "components": describe(&recovered),
            "projector_distance_to_nearest_planted_span": planted_distances,
            "seconds": planted_seconds,
        },
        "seconds": {
            "jacobians": jacobian_seconds,
            "decompose": decompose_seconds,
        },
    });
    write_json(out, &report)?;
    println!("[blocks] {}", serde_json::to_string(&report).map_err(|error| error.to_string())?);
    Ok(())
}

/// `‖B Bᵀ − F Fᵀ‖₂` for orthonormal `basis` and `span`: the sine of their largest principal angle,
/// `√(1 − σ_min(Bᵀ F)²)`, when their dimensions agree, and `1` when they differ.
fn projector_distance(basis: ArrayView2<'_, f64>, span: ArrayView2<'_, f64>) -> Result<f64, String> {
    if basis.ncols() != span.ncols() {
        return Ok(1.0);
    }
    let singular = basis
        .t()
        .dot(&span)
        .svd(false, false)
        .map_err(|error| format!("SVD of the principal-angle cosines: {error}"))?
        .1;
    let smallest = singular.iter().copied().fold(f64::INFINITY, f64::min).min(1.0);
    Ok((1.0 - smallest * smallest).max(0.0).sqrt())
}

fn open_npy(path: &Path) -> Result<NpyFile<BufReader<File>>, String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let npy = NpyFile::new(BufReader::new(file)).map_err(|err| format!("read .npy header {}: {err}", path.display()))?;
    if let Order::Fortran = npy.order() {
        return Err(format!("{} is Fortran-ordered", path.display()));
    }
    Ok(npy)
}

/// The leading `max_rows` rows (all rows when `None`) of a 2-D .npy, as float64.
fn read_rows<T: npyz::Deserialize + Into<f64>>(path: &Path, max_rows: Option<usize>) -> Result<Array2<f64>, String> {
    let npy = open_npy(path)?;
    let shape = npy.shape().to_vec();
    let [rows, cols] = shape.as_slice() else {
        return Err(format!("{} must be 2-D; it has {} axes", path.display(), shape.len()));
    };
    let rows = usize::try_from(*rows).map_err(|err| format!("{}: {err}", path.display()))?;
    let cols = usize::try_from(*cols).map_err(|err| format!("{}: {err}", path.display()))?;
    let take = max_rows.map_or(rows, |limit| limit.min(rows));
    let reader = npy
        .try_data::<T>()
        .map_err(|npy| format!("{} has dtype {}", path.display(), npy.dtype().descr()))?;
    let values = reader
        .take(take * cols)
        .map(|value| value.map(Into::into))
        .collect::<std::io::Result<Vec<f64>>>()
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    Array2::from_shape_vec((take, cols), values).map_err(|err| format!("{} has an invalid shape: {err}", path.display()))
}

/// A float64 .npy of any rank.
fn read_array(path: &Path) -> Result<ArrayD<f64>, String> {
    let npy = open_npy(path)?;
    let shape = npy
        .shape()
        .iter()
        .map(|&extent| usize::try_from(extent).map_err(|err| format!("{}: {err}", path.display())))
        .collect::<Result<Vec<usize>, String>>()?;
    let reader = npy
        .try_data::<f64>()
        .map_err(|npy| format!("{} has dtype {}", path.display(), npy.dtype().descr()))?;
    let values = reader
        .collect::<std::io::Result<Vec<f64>>>()
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    ArrayD::from_shape_vec(shape, values).map_err(|err| format!("{} has an invalid shape: {err}", path.display()))
}

fn read_vector(path: &Path) -> Result<Array1<f64>, String> {
    let npy = open_npy(path)?;
    if npy.shape().len() != 1 {
        return Err(format!("{} must be 1-D", path.display()));
    }
    let reader = npy
        .try_data::<f64>()
        .map_err(|npy| format!("{} has dtype {}", path.display(), npy.dtype().descr()))?;
    reader
        .collect::<std::io::Result<Vec<f64>>>()
        .map(Array1::from)
        .map_err(|err| format!("read {}: {err}", path.display()))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|err| format!("encode {}: {err}", path.display()))?;
    std::fs::write(path, text).map_err(|err| format!("write {}: {err}", path.display()))
}
