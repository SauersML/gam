//! mpd-lead-3's native S2 receipt on the modular-addition transformer (#2951 S2, S3): the held-out
//! rotation fidelity of decoded plane programs, executed by a native Rust forward with derived
//! forward-error radii.
//!
//! `mpd_modadd_s2_2951 --run RUN --settings SETTINGS --out REPORT`
//!
//! `RUN` is what `bench/mpd_modadd_2951.py execute` writes for stage `s2_native`: `export.json`,
//! and for each exported checkpoint `name` its tensors `<tensor>.<name>.npy` in their trained
//! `<f4` (`W_Q`, `W_K` and `W_V` as `n_heads·d_head × d_model`, head-major, the layout of torch's
//! head concatenation), and torch's float64 logits on the test pairs, `native_logits.<name>.npy`.
//!
//! # The native executor
//!
//! The network is the benchmark's one layer: `x = W_E[token] + W_pos` (one IEEE addition, `γ₁|x̂|`),
//! then [`NativeAttentionLayer`] fed those rows with their radius (learned absolute positions, so no
//! plane is rotated; score scale `fl(1/√d_head)`), then the ReLU MLP of [`NativeMlp`]'s tensors at the
//! `=` row, then `W_U`. Every read of the MLP and the unembedding is [`linear_read`]. Every stage carries a
//! forward-error radius against the exact network at the exact input: the attention layer's own radii,
//! then each read's rounding band plus `|W| r` of its input radius, and each bias addition's one rounding,
//! each sum stepped one float up. ReLU is 1-Lipschitz and exact. (The MLP's per-term band loops were 60%
//! of a forward's time in a `perf` profile on acn112, 09-19.) The exact
//! network here is the one with the score scale as rounded, `fl(1/√d_head)`: torch divides by
//! `√d_head` instead, so torch's logits are compared as a MEASURED agreement, not a certified one.
//!
//! # Scored quantities
//!
//! Two statistics are declared in the plan (#2951 comment 5716688586), each over a whole family
//! (rows × shifts), and the per-row KL quantiles at the declared levels are reported beside both:
//!
//! * the largest per-row KL. Per row, [`kl_over_logit_boxes`] bounds `KL(native shifted ‖ edited)`
//!   over both logit boxes, an exact value of the centres with its evaluation error and the box shift.
//!   The family's status is `Exact` with the largest value and the largest error, since
//!   `|max_i v_i − max_i t_i| ≤ max_i |v_i − t_i|`, or `Unresolved` when some row's box is too wide for
//!   the bound;
//! * the fraction of rows whose computed argmax differs from the reference's: an exact count on the
//!   computed logits, whose one rounding is the division. The rows whose two argmaxes are both
//!   separated from every other logit by more than the radii are counted too; on them the count is
//!   also the exact network's.
//!
//! The KL tolerances are the ladder induction's S2a declares, `{1e-3, 1e-2, 1e-1, 1}` nats, and the
//! plane programs' precision is its declared `fraction_bits = 16`, so the two benchmarks compare. The
//! disagreement fractions are a second declared ladder. Every ladder is read at every rung; nothing is
//! searched. The native reference of a shift is the native network on the shifted tokens; the executor
//! control checks, bit for bit, that it equals the row permutation of `W_E` read at the site.
//!
//! # Plane programs
//!
//! Frequencies are ordered by the closed-form power of `W_E`'s cycled rows ([`cyclic_planes`]), from
//! `W_E` alone. The program of a prefix `S` is [`plane_program_code`] at the declared precision; the
//! decoder rebuilds `U_S` from its lattice indices and takes the table from the teacher, and
//! [`frequency_edit`] forms the edit of every shift from `p`, `k` and `s`: each cycled row's own
//! components in the program's planes turn by `ω_k s`, with the characters of its cycle position as
//! coordinates, and every other plane and the `=` row stay fixed. (The joint operator
//! `cyclic_action::rotation_edit` takes each row's least-squares coordinates on `[c_0, U_S]`, which on
//! skewed planes also turn the other planes' content; see `cyclic_action`.) The executed artifact is
//! the binary64 edited table computed here, so every status is about the exact network at that
//! table and nothing an executor recomputes at another precision.
//!
//! * Fit: at the declared fit shift, on the training pairs, `K*(ε)` is the least prefix whose
//!   decoded program meets `ε`, for each declared tolerance of the ladder. That is a report at
//!   declared tolerances, not a search: the ladder is the one induction's S2a declares.
//! * Held out: each `K*(ε)` program, the full plane set (which reproduces every shift of a table of
//!   full row rank) and the unedited model, on the test pairs at every declared held-out shift.
//! * Code at fidelity: `decide_proposal(Expose)` with the full set as the reference and `K*(ε)` as
//!   the candidate, at each rung of each ladder. With `p ≤ d` the full set exists for any table, a
//!   random one included, so fidelity alone separates nothing; the saving in bits at proven fidelity
//!   does. A rejection is reported as the owner states it.
//! * Controls: every checkpoint other than the declared trained one is also read, on its own
//!   power-ordered planes, at each rung's least fit-chosen `K` whose trained program met the rung on
//!   the held-out family. The control's program of that size must miss it, reported with the row that
//!   witnesses the miss; a control that meets it is reported as a finding, never as a pass. A rung no
//!   trained program met held out has no separation to test, and says so. The
//!   checkpoints are C1 (step 0), C2 (random labels), C4 (the memorization checkpoint) and C3 (the
//!   trained model with `W_in`'s rows and `W_out`'s columns permuted independently, built by the
//!   driver from a declared seed).
//! * C5, null edits on the trained model at each such `K` up to half the planes: the `K` planes of
//!   least power with their own angles must miss the shifted reference where the chosen program met,
//!   and must meet the rung against the unedited model (no measured shift); a Haar-random `2K`-plane
//!   basis (declared `null_seed`) carrying the chosen planes' angles must miss the shifted reference.
//!
//! # Stage S3: the plane support
//!
//! Declared by `"support": {"sites": [site, …], "tolerances": [ε, …]}`, KL tolerances on the fit family.
//! The components are every plane of the decoded full program: its one codeword decodes each plane's reals
//! exactly as a subset's would. The frequency edit is linear in the plane masks, `E + Σ_k m_k C_k`, so a box of
//! masks is the center table with a derived radius (`PlaneBoxes`), and `supports::BoxSeparationOracle` with
//! `minimum_code_support` finds the minimum-code support under the padded plane code
//! `L(m, k) + L_int(H) + k·H`, a function of the size alone. The failure hypergraph's edges are the OR
//! constraints. When the code is exact, no proper subset is certified, so every member is necessary. The
//! support's own program is then scored on the held-out family as S2's programs are. Every checkpoint runs
//! it, so each control's support is read beside the trained one's.

use gam_math::gaussian_activation::GaussianActivation;
use gam_sae::parameter_decomposition::attention::{AttentionGeometry, ProjectedRows, RotaryEmbedding, RotaryPairing};
use gam_sae::parameter_decomposition::block::{AttentionLayerReads, NativeAttentionLayer, linear_read};
use gam_sae::parameter_decomposition::bounds::kl_over_logit_boxes;
use gam_sae::parameter_decomposition::cyclic_action::{
    CyclicPlanes, PlaneProgramCode, RowCycle, cyclic_planes, frequency_edit, plane_program_code,
};
use gam_sae::parameter_decomposition::fit::{ProposalKind, decide_proposal};
use gam_sae::parameter_decomposition::precision::{
    DecodableArtifact, DecodedFidelity, DeclaredPrecision, FidelityVerdict, decode_then_evaluate,
};
use gam_sae::parameter_decomposition::receipts::evaluation_band;
use gam_sae::parameter_decomposition::rewrite::NativeMlp;
use gam_sae::parameter_decomposition::seed::RankRevealingRead;
use gam_sae::parameter_decomposition::codec::{CodecError, prefix_integer_len_bits, subset_code_len_bits};
use gam_sae::parameter_decomposition::supports::{
    BoxDivergence, BoxEnclosure, BoxFamily, BoxOracleError, BoxSeparationOracle, CardinalityCode, ComponentSet,
    EvidenceStatus, ExactBasis, Extremum, FailureHypergraph, MaskBox, MaskSide, SeparationOracle, SupportSearchError,
    minimum_code_support,
};
use gam_linalg::roundoff::UNIT_ROUNDOFF;
use ndarray::{Array1, Array2, ArrayView1, Axis, Zip, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[path = "support/npy_header.rs"]
mod npy_header;
use npy_header::{NpyFloat, parse_npy_float_header, parse_npy_header};

const USAGE: &str = "usage: mpd_modadd_s2_2951 --run RUN --settings SETTINGS --out REPORT";

/// The benchmark's sequence: operand `a`, operand `b`, then `=`.
const POSITIONS: [i64; 3] = [0, 1, 2];

/// The fields of the settings this example reads; the driver also reads `runs`. Every ladder and level
/// is a declared experiment input: the example reports at each rung and searches none.
#[derive(Deserialize)]
struct Settings {
    stage: String,
    /// The checkpoint whose chosen `K` every other checkpoint is also read at.
    trained: String,
    sites: Vec<String>,
    fit_shift: usize,
    held_out_shifts: Vec<usize>,
    /// The ladder of KL tolerances, in nats.
    tolerances: Vec<f64>,
    /// The ladder of argmax-disagreement fractions.
    disagreements: Vec<f64>,
    /// The quantile levels of the per-row KL reported beside both statistics.
    kl_quantiles: Vec<f64>,
    fraction_bits: i32,
    /// The seed of control C5's Haar-random plane bases.
    null_seed: u64,
    /// Stage S3, when declared.
    support: Option<SupportSettings>,
}

#[derive(Deserialize)]
struct Export {
    stage: String,
    exports: Vec<Checkpoint>,
}

/// One exported checkpoint: its configuration, split and torch's executor control.
#[derive(Deserialize)]
struct Checkpoint {
    name: String,
    labels: String,
    step: u64,
    p: usize,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    d_mlp: usize,
    train_pairs: Vec<[usize; 2]>,
    test_pairs: Vec<[usize; 2]>,
    torch_control: Value,
}

fn flag(args: &[String], name: &str) -> Result<PathBuf, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| format!("missing {name}; {USAGE}"))
}

/// A one- or two-axis float `.npy` of the declared element type, widened to binary64 (exact for
/// every `f32`), with its shape.
fn read_npy(path: &Path, float: NpyFloat) -> Result<(Vec<usize>, Vec<f64>), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let header = parse_npy_float_header(&bytes, path)?;
    if header.float != float {
        return Err(format!("{}: expected {float:?}, got {:?}", path.display(), header.float));
    }
    let count: usize = header.shape.iter().product();
    let end = count
        .checked_mul(float.bytes())
        .and_then(|size| size.checked_add(header.data_off))
        .ok_or_else(|| format!("{}: size overflow", path.display()))?;
    if end != bytes.len() {
        return Err(format!("{}: {} bytes, expected {end}", path.display(), bytes.len()));
    }
    let data = &bytes[header.data_off..end];
    let values: Vec<f64> = match float {
        NpyFloat::F4 => data
            .chunks_exact(4)
            .map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
            .collect(),
        NpyFloat::F8 => data
            .chunks_exact(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect(),
        NpyFloat::F2 => return Err(format!("{}: <f2 is not a declared export", path.display())),
    };
    Ok((header.shape, values))
}

/// A two-axis array of the declared element type. A trained `<f4` tensor goes through the shared
/// 2-D `<f4` header contract, [`parse_npy_header`].
fn matrix(path: &Path, float: NpyFloat, rows: usize, cols: usize) -> Result<Array2<f64>, String> {
    let (shape, values) = match float {
        NpyFloat::F4 => {
            let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
            let (found_rows, found_cols, width, is_f4, data_off) = parse_npy_header(&bytes, path)?;
            if !is_f4 || width != float.bytes() {
                return Err(format!("{}: expected the trained <f4 values", path.display()));
            }
            let end = found_rows
                .checked_mul(found_cols)
                .and_then(|count| count.checked_mul(width))
                .and_then(|size| size.checked_add(data_off))
                .ok_or_else(|| format!("{}: size overflow", path.display()))?;
            if end != bytes.len() {
                return Err(format!("{}: {} bytes, expected {end}", path.display(), bytes.len()));
            }
            let values: Vec<f64> = bytes[data_off..end]
                .chunks_exact(4)
                .map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
                .collect();
            (vec![found_rows, found_cols], values)
        }
        _ => read_npy(path, float)?,
    };
    if shape != [rows, cols] {
        return Err(format!("{}: shape {shape:?}, expected [{rows}, {cols}]", path.display()));
    }
    Array2::from_shape_vec((rows, cols), values).map_err(|error| format!("{}: {error}", path.display()))
}

fn vector(path: &Path, len: usize) -> Result<Array1<f64>, String> {
    let (shape, values) = read_npy(path, NpyFloat::F4)?;
    if shape != [len] {
        return Err(format!("{}: shape {shape:?}, expected [{len}]", path.display()));
    }
    Ok(Array1::from(values))
}

fn write_report(out: &Path, report: &Value) -> Result<(), String> {
    let partial = out.with_extension("json.partial");
    let text = serde_json::to_string_pretty(report).map_err(|error| format!("report: {error}"))?;
    std::fs::write(&partial, text).map_err(|error| format!("write {}: {error}", partial.display()))?;
    std::fs::rename(&partial, out).map_err(|error| format!("rename to {}: {error}", out.display()))
}

fn up(value: f64) -> f64 {
    value.next_up()
}

/// The process's resident and peak resident memory, `VmRSS` and `VmHWM`, where the platform reports
/// them: a measurement printed at each phase boundary.
fn memory() -> String {
    std::fs::read_to_string("/proc/self/status")
        .map(|status| {
            status
                .lines()
                .filter(|line| line.starts_with("VmRSS") || line.starts_with("VmHWM"))
                .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_else(|error| format!("unavailable: {error}"))
}

/// Logits at `=` with their forward-error radius against the exact network at the executed rows.
#[derive(Clone, Debug, PartialEq)]
struct Logits {
    values: Vec<f64>,
    radius: Vec<f64>,
}

/// The first index of the largest value.
fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (index, &value)| if value > best.1 { (index, value) } else { best })
        .0
}

/// Every entry has the same bits.
fn bitwise_equal(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The logits at `=` of many rows with their radii, row-major in two allocations. Each forward's own
/// [`Logits`] is copied in and dropped on the worker thread that made it, so no per-row allocation outlives
/// its forward. With one `Arc<Logits>` kept per row, the live rows sat in glibc's per-thread heaps between
/// the forwards' short-lived allocations above its 128 KiB mmap threshold, and the heaps grew by about
/// 0.5 MB per live row at 8 threads: 15.2 GB resident with 30,645 rows alive (dx1 run
/// mpd-lead-modadd-m1). At 2 threads the 12,769 native rows held 2.1 GB, and 32 MB with glibc's mmap
/// threshold fixed at 128 KiB, which rules out the rows themselves.
struct LogitTable {
    width: usize,
    values: Vec<f64>,
    radius: Vec<f64>,
}

impl LogitTable {
    /// `rows` rows of `width` zeros.
    fn new(rows: usize, width: usize) -> Self {
        Self { width, values: vec![0.0; rows * width], radius: vec![0.0; rows * width] }
    }

    fn rows(&self) -> usize {
        self.values.len() / self.width
    }

    /// Row `row`'s logits and radius.
    fn row(&self, row: usize) -> (&[f64], &[f64]) {
        let range = row * self.width..(row + 1) * self.width;
        (&self.values[range.clone()], &self.radius[range])
    }

    /// Rows `start..start + items.len()`, each `forward(item)`, in parallel.
    fn fill<I: Sync>(
        &mut self,
        start: usize,
        items: &[I],
        forward: impl Fn(&I) -> Result<Logits, String> + Sync + Send,
    ) -> Result<(), String> {
        let width = self.width;
        let range = start * width..(start + items.len()) * width;
        if range.end > self.values.len() {
            return Err(format!("rows {start}..{} of a {}-row table", start + items.len(), self.rows()));
        }
        self.values[range.clone()]
            .par_chunks_mut(width)
            .zip(self.radius[range].par_chunks_mut(width))
            .zip(items.par_iter())
            .try_for_each(|((values, radius), item)| {
                let logits = forward(item)?;
                if logits.values.len() != width || logits.radius.len() != width {
                    return Err(format!("a forward returned {} logits and {} radii for a {width}-wide table", logits.values.len(), logits.radius.len()));
                }
                values.copy_from_slice(&logits.values);
                radius.copy_from_slice(&logits.radius);
                Ok(())
            })
    }
}

/// The benchmark's one-layer network on its stored tensors.
struct Network {
    table: Array2<f64>,
    positions: Array2<f64>,
    attention: NativeAttentionLayer,
    mlp: NativeMlp,
    unembed: Array2<f64>,
}

impl Network {
    fn load(run: &Path, checkpoint: &Checkpoint) -> Result<Self, String> {
        let file = |tensor: &str| run.join(format!("{tensor}.{}.npy", checkpoint.name));
        let (p, d, heads, head, hidden) =
            (checkpoint.p, checkpoint.d_model, checkpoint.n_heads, checkpoint.d_head, checkpoint.d_mlp);
        let geometry = AttentionGeometry { model_dim: d, n_heads: heads, n_kv_heads: heads, head_dim: head };
        let no_rotary = RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: Vec::new(),
            attention_scaling: 1.0,
        };
        let attention = NativeAttentionLayer::new(
            geometry,
            no_rotary,
            1.0 / (head as f64).sqrt(),
            matrix(&file("W_Q"), NpyFloat::F4, heads * head, d)?,
            matrix(&file("W_K"), NpyFloat::F4, heads * head, d)?,
            matrix(&file("W_V"), NpyFloat::F4, heads * head, d)?,
            matrix(&file("W_O"), NpyFloat::F4, d, heads * head)?,
        )
        .map_err(|error| error.to_string())?;
        let mlp = NativeMlp::new(
            matrix(&file("W_in"), NpyFloat::F4, hidden, d)?,
            vector(&file("b_in"), hidden)?,
            matrix(&file("W_out"), NpyFloat::F4, d, hidden)?,
            vector(&file("b_out"), d)?,
            GaussianActivation::Relu,
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            table: matrix(&file("W_E"), NpyFloat::F4, p + 1, d)?,
            positions: matrix(&file("W_pos"), NpyFloat::F4, POSITIONS.len(), d)?,
            attention,
            mlp,
            unembed: matrix(&file("W_U"), NpyFloat::F4, p, d)?,
        })
    }

    /// The logits at `=` for the three embedding rows the positions read.
    fn forward(&self, rows: [ArrayView1<'_, f64>; 3]) -> Result<Logits, String> {
        self.forward_within(rows, None)
    }

    /// The logits at `=` for embedding rows that lie within `radius` of the rows the exact network reads
    /// (none: the rows are exact).
    fn forward_within(
        &self,
        rows: [ArrayView1<'_, f64>; 3],
        radius: Option<[ArrayView1<'_, f64>; 3]>,
    ) -> Result<Logits, String> {
        let width = self.table.ncols();
        let mut residual = Array2::<f64>::zeros((POSITIONS.len(), width));
        for (position, row) in rows.iter().enumerate() {
            let mut target = residual.row_mut(position);
            target.assign(row);
            target += &self.positions.row(position);
        }
        // `x = W_E[token] + W_pos` is one IEEE addition: `γ₁|x̂|`, plus the rows' own radius.
        let mut residual_radius = residual.mapv(|value| evaluation_band(1, value.abs()));
        if let Some(radius) = radius {
            for (mut total, own) in residual_radius.outer_iter_mut().zip(radius) {
                total.zip_mut_with(&own, |total, &own| *total = up(*total + own));
            }
        }
        let embedded = ProjectedRows {
            values: residual.view(),
            radius: residual_radius.view(),
        };
        let layer = self
            .attention
            .execute(AttentionLayerReads::native(), embedded, &POSITIONS)
            .map_err(|error| error.to_string())?;
        let last = POSITIONS.len() - 1;
        let stream = layer.output.slice(s![last..last + 1, ..]);
        let stream_radius = layer.output_radius.slice(s![last..last + 1, ..]);
        let text = |error: &dyn std::fmt::Display| error.to_string();
        // Each MLP read is `block::linear_read`: its rounding band and `|W| r` of the input radius. Its bias adds
        // one rounding, `γ₁ |fl(y + b)|`. ReLU is 1-Lipschitz, so the activations keep the summed radius.
        let biased = |mut values: Array2<f64>, mut radius: Array2<f64>, bias: ArrayView1<'_, f64>| {
            values += &bias;
            radius.zip_mut_with(&values, |total, &value| *total = up(*total + evaluation_band(1, value.abs())));
            (values, radius)
        };
        let read_in = ProjectedRows { values: stream, radius: stream_radius };
        let (read, read_radius) = linear_read(self.mlp.read_in(), read_in).map_err(|error| text(&error))?;
        let (summed, summed_radius) = biased((*read).to_owned(), read_radius, self.mlp.bias_in());
        let activations = self.mlp.activate(summed.view()).map_err(|error| text(&error))?;
        let write_out = ProjectedRows { values: activations.view(), radius: summed_radius.view() };
        let (write, write_radius) = linear_read(self.mlp.write_out(), write_out).map_err(|error| text(&error))?;
        let (written, written_radius) = biased((*write).to_owned(), write_radius, self.mlp.bias_out());
        let mut hidden = stream.to_owned();
        hidden += &written;
        // `|fl(h + w) − (h + w)| ≤ u |fl(h + w)| / (1 − u)` for the residual addition.
        let hidden_radius = Zip::from(hidden.row(0))
            .and(stream_radius.row(0))
            .and(written_radius.row(0))
            .map_collect(|&value, &carried_in, &write| {
                up(up(carried_in + write) + up(UNIT_ROUNDOFF * value.abs() / (1.0 - UNIT_ROUNDOFF)))
            });
        let hidden_rows = ProjectedRows {
            values: hidden.view(),
            radius: hidden_radius.view().insert_axis(Axis(0)),
        };
        let (logits, logit_radius) = linear_read(self.unembed.view(), hidden_rows).map_err(|error| text(&error))?;
        Ok(Logits { values: logits.row(0).to_vec(), radius: logit_radius.row(0).to_vec() })
    }
}

/// Where an edited table is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Site {
    Pos0,
    Pos1,
    Global,
}

impl Site {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "pos0" => Ok(Self::Pos0),
            "pos1" => Ok(Self::Pos1),
            "global" => Ok(Self::Global),
            other => Err(format!("site {other:?} is not declared; the sites are pos0, pos1 and global")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Pos0 => "pos0",
            Self::Pos1 => "pos1",
            Self::Global => "global",
        }
    }

    /// The pair whose native execution is the reference of `(a, b)` at `shift`: the row permutation
    /// moves the cycled rows the site reads and fixes `=`.
    fn shifted(self, [a, b]: [usize; 2], shift: usize, p: usize) -> [usize; 2] {
        match self {
            Self::Pos0 => [(a + shift) % p, b],
            Self::Pos1 => [a, (b + shift) % p],
            Self::Global => [(a + shift) % p, (b + shift) % p],
        }
    }
}

/// The native network on every pair `(a, b)`, row `a·p + b`.
fn native_table(network: &Network, p: usize) -> Result<LogitTable, String> {
    let table = &network.table;
    let indices: Vec<usize> = (0..p * p).collect();
    let mut logits = LogitTable::new(p * p, p);
    logits.fill(0, &indices, |&index| {
        let (a, b) = (index / p, index % p);
        network.forward([table.row(a), table.row(b), table.row(p)])
    })?;
    Ok(logits)
}

/// The network with `edited` read at the site, on every pair, into rows `start..start + pairs.len()`.
fn edited_rows(
    network: &Network,
    edited: &Array2<f64>,
    site: Site,
    pairs: &[[usize; 2]],
    p: usize,
    logits: &mut LogitTable,
    start: usize,
) -> Result<(), String> {
    let native = &network.table;
    logits.fill(start, pairs, |&[a, b]| {
        let rows = match site {
            Site::Pos0 => [edited.row(a), native.row(b), native.row(p)],
            Site::Pos1 => [native.row(a), edited.row(b), native.row(p)],
            Site::Global => [edited.row(a), edited.row(b), edited.row(p)],
        };
        network.forward(rows)
    })
}

/// One row of a scored family: `(shift, a, b)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Witness {
    shift: usize,
    a: usize,
    b: usize,
}

/// The finite family a status is exhaustive over.
#[derive(Clone, Debug, PartialEq)]
struct Family {
    checkpoint: String,
    site: &'static str,
    shifts: Vec<usize>,
    pairs: usize,
}

/// The rows of one family, shift-major, with the witness of each row: a table's rows in order, or the
/// native table's rows at `index` (a family's references).
#[derive(Clone)]
struct Rows {
    table: Arc<LogitTable>,
    index: Option<Arc<[usize]>>,
    witnesses: Vec<Witness>,
}

impl Rows {
    /// Row `row`'s logits and radius.
    fn row(&self, row: usize) -> (&[f64], &[f64]) {
        self.table.row(self.index.as_ref().map_or(row, |index| index[row]))
    }
}

/// The native rows of a family's pairs, each pair read at `native_pair(shift, pair)`.
fn native_rows(
    native: &Arc<LogitTable>,
    shifts: &[usize],
    pairs: &[[usize; 2]],
    p: usize,
    native_pair: impl Fn(usize, [usize; 2]) -> [usize; 2],
) -> Rows {
    let mut index = Vec::with_capacity(shifts.len() * pairs.len());
    let mut witnesses = Vec::with_capacity(shifts.len() * pairs.len());
    for &shift in shifts {
        for &pair in pairs {
            let [a, b] = native_pair(shift, pair);
            index.push(a * p + b);
            witnesses.push(Witness { shift, a: pair[0], b: pair[1] });
        }
    }
    Rows { table: Arc::clone(native), index: Some(index.into()), witnesses }
}

/// The native references of a family: `f` at the shifted pair.
fn reference_rows(native: &Arc<LogitTable>, site: Site, shifts: &[usize], pairs: &[[usize; 2]], p: usize) -> Rows {
    native_rows(native, shifts, pairs, p, |shift, pair| site.shifted(pair, shift, p))
}

/// One row's comparison: the logit-box KL bound when the boxes allow it, the two computed argmaxes, and
/// whether each argmax is separated from every other logit by more than both radii.
#[derive(Clone, Copy, Debug)]
struct RowComparison {
    kl: Option<(f64, f64)>,
    disagrees: bool,
    certified: bool,
}

/// The computed argmax is the exact one when its lower end clears every other logit's upper end.
fn certified_argmax((values, radii): (&[f64], &[f64])) -> bool {
    let top = argmax(values);
    let floor = values[top] - radii[top];
    values
        .iter()
        .zip(radii)
        .enumerate()
        .all(|(index, (value, radius))| index == top || floor.next_down() > up(value + radius))
}

/// Compare every executed row with its reference; the KL bound only when `with_kl`.
fn compare_rows(executed: &Rows, reference: &Rows, family: &Family, with_kl: bool) -> Result<Vec<RowComparison>, String> {
    if executed.witnesses != reference.witnesses {
        return Err(format!("{family:?}: the executed and reference rows are not the same family"));
    }
    (0..executed.witnesses.len())
        .into_par_iter()
        .map(|row| {
            let (edited, native) = (executed.row(row), reference.row(row));
            let kl = if with_kl {
                let status = kl_over_logit_boxes(
                    ArrayView1::from(native.0),
                    ArrayView1::from(native.1),
                    ArrayView1::from(edited.0),
                    ArrayView1::from(edited.1),
                )
                .map_err(|error| error.to_string())?;
                match status {
                    EvidenceStatus::Exact { value, numerical_error, .. } => Some((value, numerical_error)),
                    EvidenceStatus::Unresolved { .. } => None,
                    other => return Err(format!("the logit-box KL returned an undeclared status {other:?}")),
                }
            } else {
                None
            };
            Ok(RowComparison {
                kl,
                disagrees: argmax(edited.0) != argmax(native.0),
                certified: certified_argmax(edited) && certified_argmax(native),
            })
        })
        .collect()
}

/// The exhaustive maximum of the per-row KL bound: `Exact` with the largest value and the largest
/// error, or `Unresolved` when some row's boxes are too wide for the bound.
fn kl_status(rows: &[RowComparison], executed: &Rows, family: &Family) -> Result<EvidenceStatus<Witness, Family>, String> {
    let cardinality = rows.len() as u64;
    let mut largest: Option<(usize, f64)> = None;
    let mut error = 0.0_f64;
    let mut lower: Option<(usize, f64)> = None;
    let mut unresolved = 0usize;
    for (index, row) in rows.iter().enumerate() {
        match row.kl {
            Some((value, numerical_error)) => {
                if largest.is_none_or(|(_, best)| value > best) {
                    largest = Some((index, value));
                }
                error = error.max(numerical_error);
                let attained = (value - numerical_error).next_down().max(0.0);
                if lower.is_none_or(|(_, best)| attained > best) {
                    lower = Some((index, attained));
                }
            }
            None => unresolved += 1,
        }
    }
    let witness = |index: usize| executed.witnesses[index];
    match (unresolved, largest) {
        (0, Some((index, value))) => EvidenceStatus::exact(
            value,
            error,
            ExactBasis::Exhaustive { cardinality },
            Some(witness(index)),
            family.clone(),
        ),
        _ => EvidenceStatus::unresolved(
            lower.map_or(0.0, |(_, value)| value),
            f64::INFINITY,
            Extremum::Supremum,
            lower.map(|(index, _)| witness(index)),
            family.clone(),
        ),
    }
    .map_err(|error| error.to_string())
}

/// The exact fraction of rows whose computed argmax differs from the reference's: a count on the
/// computed logits over the whole family, whose one rounding is the division.
fn disagreement_status(
    rows: &[RowComparison],
    executed: &Rows,
    family: &Family,
) -> Result<EvidenceStatus<Witness, Family>, String> {
    let disagreeing = rows.iter().filter(|row| row.disagrees).count();
    let fraction = disagreeing as f64 / rows.len() as f64;
    let witness = rows.iter().position(|row| row.disagrees).map(|index| executed.witnesses[index]);
    EvidenceStatus::exact(
        fraction,
        UNIT_ROUNDOFF * fraction,
        ExactBasis::Exhaustive { cardinality: rows.len() as u64 },
        witness,
        family.clone(),
    )
    .map_err(|error| error.to_string())
}

/// Everything one executed family shows against its references.
struct FamilyScore {
    kl: EvidenceStatus<Witness, Family>,
    disagreement: EvidenceStatus<Witness, Family>,
    disagreeing: usize,
    certified_rows: usize,
    rows: usize,
    /// `(level, value)` of the per-row KL centres at each declared quantile level, nearest rank.
    quantiles: Vec<(f64, f64)>,
}

fn family_score(executed: &Rows, reference: &Rows, family: &Family, levels: &[f64]) -> Result<FamilyScore, String> {
    let rows = compare_rows(executed, reference, family, true)?;
    let mut values: Vec<f64> = rows.iter().filter_map(|row| row.kl.map(|(value, _)| value)).collect();
    values.sort_by(f64::total_cmp);
    let quantiles = levels
        .iter()
        .map(|&level| {
            let rank = ((level * values.len() as f64).ceil() as usize).clamp(1, values.len().max(1));
            (level, values.get(rank - 1).copied().unwrap_or(f64::NAN))
        })
        .collect();
    Ok(FamilyScore {
        kl: kl_status(&rows, executed, family)?,
        disagreement: disagreement_status(&rows, executed, family)?,
        disagreeing: rows.iter().filter(|row| row.disagrees).count(),
        certified_rows: rows.iter().filter(|row| row.certified).count(),
        rows: rows.len(),
        quantiles,
    })
}

fn status_json<W: std::fmt::Debug, D>(status: &EvidenceStatus<W, D>) -> Value {
    match status {
        EvidenceStatus::Exact { value, numerical_error, witness, .. } => json!({
            "kind": "exact", "value": value, "numerical_error": numerical_error, "witness": format!("{witness:?}"),
        }),
        EvidenceStatus::UniformBound { upper, numerical_error, .. } => {
            json!({ "kind": "uniform_bound", "upper": upper, "numerical_error": numerical_error })
        }
        EvidenceStatus::StatisticalEstimate { estimate, standard_error, samples, .. } => json!({
            "kind": "statistical_estimate", "estimate": estimate, "standard_error": standard_error, "samples": samples,
        }),
        EvidenceStatus::Counterexample { value, numerical_error, threshold, witness, .. } => json!({
            "kind": "counterexample", "value": value, "numerical_error": numerical_error, "threshold": threshold,
            "witness": format!("{witness:?}"),
        }),
        EvidenceStatus::Unresolved { lower, upper, witness, .. } => json!({
            "kind": "unresolved", "lower": lower, "upper": upper, "witness": format!("{witness:?}"),
        }),
    }
}

fn family_json(family: &Family) -> Value {
    json!({ "checkpoint": family.checkpoint, "site": family.site, "shifts": family.shifts, "pairs": family.pairs })
}

/// The row a status names: the attaining row of an exact maximum, a disagreeing row of a fraction, or the
/// row attaining an unresolved status's lower side.
fn witness_json(status: &EvidenceStatus<Witness, Family>) -> Value {
    match status.witness() {
        Some(witness) => json!({ "shift": witness.shift, "a": witness.a, "b": witness.b }),
        None => Value::Null,
    }
}

fn score_json(score: &FamilyScore) -> Value {
    json!({
        "max_kl": status_json(&score.kl),
        "max_kl_row": witness_json(&score.kl),
        "disagreement": status_json(&score.disagreement),
        "disagreement_row": witness_json(&score.disagreement),
        "disagreeing_rows": score.disagreeing,
        "rows": score.rows,
        "rows_with_both_argmaxes_certified": score.certified_rows,
        "kl_quantiles": score.quantiles.iter().map(|(level, value)| json!({ "level": level, "value": value })).collect::<Vec<_>>(),
    })
}

fn score_line(score: &FamilyScore) -> String {
    let quantiles: Vec<String> = score.quantiles.iter().map(|(level, value)| format!("q{level}={value:.3e}")).collect();
    format!(
        "max_kl={} disagreement={}/{} ({}) certified_rows={} kl[{}]",
        status_json(&score.kl),
        score.disagreeing,
        score.rows,
        status_json(&score.disagreement),
        score.certified_rows,
        quantiles.join(" ")
    )
}

/// The decoded basis `U_S` (`d × 2|S|`) of a program: the encoder sent `U_Sᵀ` row-major.
fn decoded_basis(values: &[f64], width: usize) -> Result<Array2<f64>, String> {
    if values.len() % width != 0 {
        return Err(format!("{} decoded reals are not a whole number of {width}-rows", values.len()));
    }
    let transposed = Array2::from_shape_vec((values.len() / width, width), values.to_vec())
        .map_err(|error| error.to_string())?;
    Ok(transposed.t().to_owned())
}

/// The decoded program's edited table at `shift`: `W_E + L Rᵀ` from [`frequency_edit`].
fn edited_table(
    table: &Array2<f64>,
    planes: &CyclicPlanes,
    basis: &Array2<f64>,
    frequencies: &[usize],
    shift: usize,
) -> Result<Array2<f64>, String> {
    let edit = frequency_edit(planes.cycle(), basis.view(), frequencies, shift).map_err(|error| error.to_string())?;
    Ok(table + &edit.left.dot(&edit.right.t()))
}

/// Control C5's random planes: a Haar-random orthonormal basis `d × 2|S|` (the first `2|S|` right singular
/// vectors of a Gaussian matrix drawn by Box-Muller) carrying the planes' angles of `frequencies`, executed at
/// every shift of the target. It has no codeword: it is a null edit, not a program.
fn random_plane_rows(
    network: &Network,
    planes: &CyclicPlanes,
    frequencies: &[usize],
    rng: &mut StdRng,
    target: &Target<'_>,
) -> Result<Rows, String> {
    let width = network.table.ncols();
    let gaussian = Array2::from_shape_simple_fn((2 * frequencies.len(), width), || {
        let radius = (-2.0 * (1.0 - rng.random::<f64>()).ln()).sqrt();
        radius * (std::f64::consts::TAU * rng.random::<f64>()).cos()
    });
    let read = RankRevealingRead::new(&gaussian).map_err(|error| error.to_string())?;
    let basis = read.read().slice(s![..2 * frequencies.len(), ..]).t().to_owned();
    let p = planes.cycle().length();
    let mut logits = LogitTable::new(target.shifts.len() * target.pairs.len(), p);
    for (index, &shift) in target.shifts.iter().enumerate() {
        let edited = edited_table(&network.table, planes, &basis, frequencies, shift)?;
        edited_rows(network, &edited, target.site, target.pairs, p, &mut logits, index * target.pairs.len())?;
    }
    Ok(Rows { table: Arc::new(logits), index: None, witnesses: target.reference.witnesses.clone() })
}

/// The two declared ladders a program is read at: KL tolerances and disagreement fractions.
struct Ladders<'a> {
    tolerances: &'a [f64],
    disagreements: &'a [f64],
    quantiles: &'a [f64],
}

/// One plane program, executed once over a family, with its evidence at every rung of both ladders.
struct Scored {
    frequencies: Vec<usize>,
    code: PlaneProgramCode,
    score: FamilyScore,
    kl_fidelities: Vec<DecodedFidelity<Witness, Family>>,
    disagreement_fidelities: Vec<DecodedFidelity<Witness, Family>>,
    seconds: f64,
}

/// Where a program is executed: the site, the family's shifts and pairs, and their native references.
struct Target<'a> {
    site: Site,
    shifts: &'a [usize],
    pairs: &'a [[usize; 2]],
    reference: &'a Rows,
    family: &'a Family,
}

/// Decode `frequencies`' program, execute it at every shift of the target, and state its fidelity at
/// each rung of both ladders. The execution runs once: each rung's `decode_then_evaluate` decodes the
/// program again and hands back the executed rows only when that decode equals the one executed.
fn score_program(
    network: &Network,
    planes: &CyclicPlanes,
    frequencies: &[usize],
    precision: DeclaredPrecision,
    target: &Target<'_>,
    ladders: &Ladders<'_>,
) -> Result<Scored, String> {
    let started = Instant::now();
    let code = plane_program_code(planes, frequencies, precision).map_err(|error| error.to_string())?;
    let decoded = code.basis.decode()?;
    let basis = decoded_basis(&decoded, network.table.ncols())?;
    let p = planes.cycle().length();
    let mut logits = LogitTable::new(target.shifts.len() * target.pairs.len(), p);
    for (index, &shift) in target.shifts.iter().enumerate() {
        let edited = edited_table(&network.table, planes, &basis, frequencies, shift)?;
        edited_rows(network, &edited, target.site, target.pairs, p, &mut logits, index * target.pairs.len())?;
    }
    let executed = Rows { table: Arc::new(logits), index: None, witnesses: target.reference.witnesses.clone() };
    let score = family_score(&executed, target.reference, target.family, ladders.quantiles)?;
    let again = |decoded_again: &Vec<f64>| {
        if decoded_again == &decoded {
            Ok(executed.clone())
        } else {
            Err("the program decoded to other reals than the ones executed".to_string())
        }
    };
    let mut kl_fidelities = Vec::with_capacity(ladders.tolerances.len());
    for &tolerance in ladders.tolerances {
        kl_fidelities.push(decode_then_evaluate(
            &code.basis,
            again,
            target.reference,
            |outputs: &Rows, native: &Rows| {
                kl_status(&compare_rows(outputs, native, target.family, true)?, outputs, target.family)
            },
            tolerance,
        )?);
    }
    let mut disagreement_fidelities = Vec::with_capacity(ladders.disagreements.len());
    for &tolerance in ladders.disagreements {
        disagreement_fidelities.push(decode_then_evaluate(
            &code.basis,
            again,
            target.reference,
            |outputs: &Rows, native: &Rows| {
                disagreement_status(&compare_rows(outputs, native, target.family, false)?, outputs, target.family)
            },
            tolerance,
        )?);
    }
    Ok(Scored {
        frequencies: frequencies.to_vec(),
        code,
        score,
        kl_fidelities,
        disagreement_fidelities,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn kl_fidelities(scored: &Scored) -> &Vec<DecodedFidelity<Witness, Family>> {
    &scored.kl_fidelities
}

fn disagreement_fidelities(scored: &Scored) -> &Vec<DecodedFidelity<Witness, Family>> {
    &scored.disagreement_fidelities
}

fn verdicts(fidelities: &[DecodedFidelity<Witness, Family>]) -> Vec<String> {
    fidelities.iter().map(|fidelity| format!("{:?}", fidelity.verdict())).collect()
}

fn scored_json(scored: &Scored) -> Value {
    json!({
        "K": scored.frequencies.len(),
        "frequencies": scored.frequencies,
        "bits": { "subset": scored.code.subset_bits, "basis": scored.code.basis_bits, "total": scored.code.total_bits() },
        "score": score_json(&scored.score),
        "kl_verdicts": verdicts(&scored.kl_fidelities),
        "disagreement_verdicts": verdicts(&scored.disagreement_fidelities),
        "seconds": scored.seconds,
    })
}

/// The row permutation of `W_E` by `shift` over the cycled rows, `=` fixed.
fn permuted_table(table: &Array2<f64>, shift: usize, p: usize) -> Array2<f64> {
    let mut permuted = table.clone();
    for a in 0..p {
        permuted.row_mut(a).assign(&table.row((a + shift) % p));
    }
    permuted
}

/// The least power-ordered prefix whose fit verdict `Meets` each rung, per ladder.
#[derive(Clone, Debug, Default)]
struct Chosen {
    kl: Vec<Option<usize>>,
    disagreement: Vec<Option<usize>>,
}

/// What the trained checkpoint showed at one site, for the controls read at its `K`.
struct TrainedSite {
    chosen: Chosen,
    /// Per ladder and rung, the least `K` among the fit-chosen programs whose held-out fidelity meets the rung:
    /// the trained program a control must miss at, or `None` when no chosen program met it held out.
    kl_met: Vec<Option<usize>>,
    disagreement_met: Vec<Option<usize>>,
}

/// The decision at one rung: the full plane set as the decoded reference, the chosen prefix as the
/// candidate (`decide_proposal`, `Expose`).
fn proposal_json(
    tolerance: f64,
    chosen: Option<usize>,
    plane_count: usize,
    full: &Scored,
    full_fidelity: &DecodedFidelity<Witness, Family>,
    candidate: Option<(&Scored, &DecodedFidelity<Witness, Family>, &EvidenceStatus<Witness, Family>)>,
) -> Value {
    match (chosen, candidate) {
        (None, _) => json!({ "tolerance": tolerance, "outcome": "no prefix meets the rung at the fit shift" }),
        (Some(kept), None) if kept == plane_count => {
            json!({ "tolerance": tolerance, "outcome": "only the full plane set meets the rung at the fit shift" })
        }
        (Some(kept), None) => json!({ "tolerance": tolerance, "K": kept, "outcome": "the chosen prefix was refused at execution" }),
        (Some(_), Some((scored, fidelity, status))) => match decide_proposal(
            ProposalKind::Expose,
            (full.code.total_bits(), full_fidelity),
            (scored.code.total_bits(), fidelity),
            status.clone(),
        ) {
            Ok(accepted) => json!({
                "tolerance": tolerance, "K": scored.frequencies.len(), "outcome": "accepted",
                "saving_bits": accepted.saving_bits.to_string(), "fidelity_certified": accepted.fidelity_certified,
            }),
            Err(rejection) => json!({
                "tolerance": tolerance, "K": scored.frequencies.len(), "outcome": "rejected", "reason": rejection.to_string(),
            }),
        },
    }
}

/// A control read at the trained model's `K` on one rung: it must miss where the trained program met.
fn control_verdict(trained_met: bool, fidelity: &DecodedFidelity<Witness, Family>) -> &'static str {
    match (trained_met, fidelity.verdict()) {
        (false, _) => "trained program did not meet this rung: no separation to test",
        (true, FidelityVerdict::Violates) => "fails as required",
        (true, FidelityVerdict::Meets) => "FINDING: the control meets the rung at the trained model's K",
        (true, FidelityVerdict::Unresolved) => "unresolved",
    }
}

/// Stage S3's declarations: the sites the plane support is found at, and its KL tolerances on the fit family.
#[derive(Deserialize)]
struct SupportSettings {
    sites: Vec<String>,
    tolerances: Vec<f64>,
}

/// The finite family a plane-support status is exhaustive over: the binary masks of every plane, on the fit
/// family's rows at one site.
#[derive(Clone, Debug, PartialEq)]
struct PlaneFamily {
    checkpoint: String,
    site: &'static str,
    shift: usize,
    pairs: usize,
    planes: usize,
}

/// One row's KL over both logit boxes: its exact center and error when the bound resolves, and its proven sides.
struct RowBound {
    kl: Option<(f64, f64)>,
    lower: Option<f64>,
    upper: Option<f64>,
}

/// A plane support's code: its subset codeword over the planes, the padded plane length `H` once, and `H` bits
/// per kept plane, a function of the size alone.
struct PlaneCode {
    plane_bits: u64,
}

impl CardinalityCode for PlaneCode {
    type Error = CodecError;

    fn support_bits(&self, components: usize, size: usize) -> Result<u64, CodecError> {
        Ok(subset_code_len_bits(components, size)? + prefix_integer_len_bits(self.plane_bits)? + size as u64 * self.plane_bits)
    }
}

/// A box oracle that prints each separation it answers, so a long search's progress is in the run's log.
struct Logged<P: BoxDivergence> {
    oracle: BoxSeparationOracle<P>,
    label: String,
    separations: usize,
    started: Instant,
}

impl<P: BoxDivergence> SeparationOracle for Logged<P> {
    type Mask = MaskBox;
    type Domain = BoxFamily<P::Domain>;
    type Error = BoxOracleError<P::Error>;

    fn components(&self) -> usize {
        self.oracle.components()
    }

    fn perturbed_components(&self, mask: &MaskBox) -> Vec<usize> {
        self.oracle.perturbed_components(mask)
    }

    fn separate(&mut self, support: &ComponentSet) -> Result<EvidenceStatus<MaskBox, Self::Domain>, Self::Error> {
        let status = self.oracle.separate(support)?;
        self.separations += 1;
        println!(
            "[search] {} separation={} kept={} enclosures={} seconds={:.0} lower={:?} upper={:?}",
            self.label,
            self.separations,
            support.len(),
            self.oracle.enclosures(),
            self.started.elapsed().as_secs_f64(),
            status.lower_bound(),
            status.upper_bound()
        );
        Ok(status)
    }

    fn evaluate(&mut self, mask: &MaskBox) -> Result<EvidenceStatus<MaskBox, Self::Domain>, Self::Error> {
        self.oracle.evaluate(mask)
    }
}

/// Stage S3's program as a box program for `supports::BoxSeparationOracle`: the decoded full plane set's
/// frequency edit at the fit shift, plane `k` under its mask `m_k`, on the fit family. The edit is linear in
/// the masks, `E + Σ_k m_k C_k` with `C_k = fl(L_k U_kᵀ)`, and `m_k = 1/2` is plane `k`'s chord midpoint (P2).
/// A vertex's table is `E + Σ_{m_k = 1} C_k`, formed in plane order. A box runs its center, every free plane at
/// `1/2`, with the rows' radius `Σ_free |C_k|/2 + 2 γ_{m+1} (|E| + Σ_k |C_k|)`: the distance to every vertex's
/// table, both formations included. Its bound is the largest per-row KL from the shifted reference over both
/// logit boxes. Free planes split in descending order of their power.
struct PlaneBoxes<'a, 'b> {
    network: &'a Network,
    contributions: &'a [Array2<f64>],
    formation: &'a Array2<f64>,
    site: Site,
    pairs: &'a [[usize; 2]],
    reference: &'a Rows,
    family: &'a PlaneFamily,
    power: &'a [f64],
    enclosed: &'b mut BTreeMap<MaskBox, BoxEnclosure<PlaneFamily>>,
}

/// Each plane's computed edit `C_k` at `shift` from the decoded basis of every plane, and the formation band
/// `2 γ_{m+1} (|E| + Σ_k |C_k|)` a box's radius carries.
fn plane_contributions(
    table: &Array2<f64>,
    planes: &CyclicPlanes,
    basis: &Array2<f64>,
    shift: usize,
) -> Result<(Vec<Array2<f64>>, Array2<f64>), String> {
    let count = planes.cycle().plane_count();
    let frequencies: Vec<usize> = (1..=count).collect();
    let edit = frequency_edit(planes.cycle(), basis.view(), &frequencies, shift).map_err(|error| error.to_string())?;
    let mut magnitude = table.mapv(f64::abs);
    let mut contributions = Vec::with_capacity(count);
    for plane in 0..count {
        let columns = 2 * plane..2 * plane + 2;
        let contribution = edit.left.slice(s![.., columns.clone()]).dot(&edit.right.slice(s![.., columns]).t());
        magnitude.zip_mut_with(&contribution, |total, &value| *total = up(*total + value.abs()));
        contributions.push(contribution);
    }
    let formation = magnitude.mapv(|value| up(2.0 * evaluation_band(count + 1, value)));
    Ok((contributions, formation))
}

impl PlaneBoxes<'_, '_> {
    /// The table at a box's center and, when a plane is free, the rows' radius about it.
    fn table_at(&self, mask: &MaskBox) -> (Array2<f64>, Option<Array2<f64>>) {
        let mut table = self.network.table.clone();
        let mut radius = (!mask.is_vertex()).then(|| self.formation.clone());
        for (contribution, side) in self.contributions.iter().zip(mask.sides()) {
            match side {
                MaskSide::On => table += contribution,
                MaskSide::Off => {}
                MaskSide::Free => {
                    table.scaled_add(0.5, contribution);
                    if let Some(radius) = radius.as_mut() {
                        radius.zip_mut_with(contribution, |total, &value| *total = up(*total + 0.5 * value.abs()));
                    }
                }
            }
        }
        (table, radius)
    }
}

impl BoxDivergence for PlaneBoxes<'_, '_> {
    type Domain = PlaneFamily;
    type Error = String;

    fn components(&self) -> usize {
        self.contributions.len()
    }

    fn domain(&self) -> PlaneFamily {
        self.family.clone()
    }

    fn enclose(&mut self, mask: &MaskBox) -> Result<BoxEnclosure<PlaneFamily>, String> {
        if let Some(found) = self.enclosed.get(mask) {
            return Ok(found.clone());
        }
        let (table, radius) = self.table_at(mask);
        let (network, reference, site) = (self.network, self.reference, self.site);
        let native = &network.table;
        let p = native.nrows() - 1;
        let zero = Array1::<f64>::zeros(native.ncols());
        let rows: Vec<RowBound> = self
            .pairs
            .par_iter()
            .enumerate()
            .map(|(index, &[a, b])| {
                // The rows the positions read, and the edited table row behind each (none: a native read).
                let (rows, edited) = match site {
                    Site::Pos0 => ([table.row(a), native.row(b), native.row(p)], [Some(a), None, None]),
                    Site::Pos1 => ([native.row(a), table.row(b), native.row(p)], [None, Some(b), None]),
                    Site::Global => ([table.row(a), table.row(b), table.row(p)], [Some(a), Some(b), Some(p)]),
                };
                let within = radius
                    .as_ref()
                    .map(|radius| edited.map(|row| row.map_or(zero.view(), |row| radius.row(row))));
                let logits = network.forward_within(rows, within)?;
                let (native_values, native_radius) = reference.row(index);
                let status = kl_over_logit_boxes(
                    ArrayView1::from(native_values),
                    ArrayView1::from(native_radius),
                    ArrayView1::from(&logits.values),
                    ArrayView1::from(&logits.radius),
                )
                .map_err(|error| error.to_string())?;
                let kl = match &status {
                    EvidenceStatus::Exact { value, numerical_error, .. } => Some((*value, *numerical_error)),
                    _ => None,
                };
                Ok(RowBound { kl, lower: status.lower_bound(), upper: status.upper_bound() })
            })
            .collect::<Result<_, String>>()?;
        let domain = self.domain();
        let error = rows.iter().filter_map(|row| row.kl).fold(0.0_f64, |largest, (_, own)| largest.max(own));
        let lower = rows.iter().filter_map(|row| row.lower).fold(0.0_f64, f64::max);
        let evidence = if mask.is_vertex() {
            match rows.iter().map(|row| row.kl).collect::<Option<Vec<_>>>() {
                Some(values) => {
                    let value = values.iter().fold(0.0_f64, |largest, (value, _)| largest.max(*value));
                    EvidenceStatus::exact(value, error, ExactBasis::Exhaustive { cardinality: 1 }, Some(mask.clone()), domain)
                }
                None => EvidenceStatus::unresolved(lower, f64::INFINITY, Extremum::Supremum, Some(mask.clone()), domain),
            }
        } else {
            match rows.iter().map(|row| row.upper).try_fold(0.0_f64, |largest, upper| upper.map(|upper| largest.max(upper))) {
                Some(upper) => EvidenceStatus::uniform_bound(upper, error, domain),
                None => EvidenceStatus::unresolved(lower, f64::INFINITY, Extremum::Supremum, None, domain),
            }
        }
        .map_err(|error| format!("{error:?}"))?;
        let mut split_order = mask.free();
        split_order.sort_by(|left, right| self.power[*right].total_cmp(&self.power[*left]));
        let enclosure = BoxEnclosure { evidence, split_order };
        self.enclosed.insert(mask.clone(), enclosure.clone());
        Ok(enclosure)
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(USAGE.to_string());
    }
    let run = flag(&args, "--run")?;
    let settings_path = flag(&args, "--settings")?;
    let out = flag(&args, "--out")?;
    let text = std::fs::read_to_string(&settings_path)
        .map_err(|error| format!("read {}: {error}", settings_path.display()))?;
    let settings: Settings = serde_json::from_str(&text).map_err(|error| format!("settings: {error}"))?;
    if settings.stage != "s2_native" {
        return Err(format!("settings are stage {:?}, not s2_native", settings.stage));
    }
    if settings.held_out_shifts.contains(&settings.fit_shift) {
        return Err(format!("the fit shift {} is among the held-out shifts", settings.fit_shift));
    }
    for (name, ladder) in [("tolerances", &settings.tolerances), ("disagreements", &settings.disagreements)] {
        if let Some(&rung) = ladder.iter().find(|rung| !(rung.is_finite() && **rung >= 0.0)) {
            return Err(format!("declared {name} rung {rung} is not finite and non-negative"));
        }
    }
    if let Some(&level) = settings.kl_quantiles.iter().find(|level| !(**level > 0.0 && **level <= 1.0)) {
        return Err(format!("declared KL quantile level {level} is not in (0, 1]"));
    }
    let sites = settings.sites.iter().map(|name| Site::parse(name)).collect::<Result<Vec<_>, _>>()?;
    if let Some(support) = &settings.support {
        for name in &support.sites {
            Site::parse(name)?;
        }
        if support.tolerances.is_empty() || support.tolerances.iter().any(|tolerance| !(tolerance.is_finite() && *tolerance >= 0.0)) {
            return Err("the support stage must declare finite, non-negative tolerances".to_string());
        }
    }
    let precision = DeclaredPrecision::new(settings.fraction_bits)?;
    let ladders = Ladders {
        tolerances: &settings.tolerances,
        disagreements: &settings.disagreements,
        quantiles: &settings.kl_quantiles,
    };
    let mut null_rng = StdRng::seed_from_u64(settings.null_seed);
    let text = std::fs::read_to_string(run.join("export.json"))
        .map_err(|error| format!("read export.json in {}: {error}", run.display()))?;
    let export: Export = serde_json::from_str(&text).map_err(|error| format!("export.json: {error}"))?;
    if export.stage != "s2_native" {
        return Err(format!("export.json is stage {:?}, not s2_native", export.stage));
    }
    // The trained checkpoint runs first: every other checkpoint is also read at its `K`.
    let mut ordered: Vec<&Checkpoint> = export.exports.iter().collect();
    let trained_index = ordered
        .iter()
        .position(|checkpoint| checkpoint.name == settings.trained)
        .ok_or_else(|| format!("the declared trained checkpoint {:?} is not exported", settings.trained))?;
    let trained_checkpoint = ordered.remove(trained_index);
    ordered.insert(0, trained_checkpoint);
    println!(
        "[load] run={} checkpoints={} trained={} sites={:?} fit_shift={} held_out_shifts={} tolerances={:?} \
         disagreements={:?} kl_quantiles={:?} fraction_bits={} threads={} memory={}",
        run.display(),
        ordered.len(),
        settings.trained,
        settings.sites,
        settings.fit_shift,
        settings.held_out_shifts.len(),
        settings.tolerances,
        settings.disagreements,
        settings.kl_quantiles,
        settings.fraction_bits,
        rayon::current_num_threads(),
        memory()
    );

    let mut trained_sites: Vec<Option<TrainedSite>> = sites.iter().map(|_| None).collect();
    let mut checkpoints: Vec<Value> = Vec::new();
    let report = |checkpoints: &[Value]| {
        json!({
            "trained": settings.trained,
            "sites": settings.sites,
            "fit_shift": settings.fit_shift,
            "held_out_shifts": settings.held_out_shifts,
            "tolerances": settings.tolerances,
            "disagreements": settings.disagreements,
            "kl_quantiles": settings.kl_quantiles,
            "fraction_bits": settings.fraction_bits,
            "checkpoints": checkpoints,
        })
    };
    for checkpoint in ordered {
        let is_trained = checkpoint.name == settings.trained;
        let p = checkpoint.p;
        let started = Instant::now();
        let network = Network::load(&run, checkpoint)?;
        let native = Arc::new(native_table(&network, p)?);
        let native_seconds = started.elapsed().as_secs_f64();
        let largest_radius = native.radius.iter().copied().fold(0.0_f64, f64::max);
        println!(
            "[native] {} labels={} step={} pairs={} seconds={native_seconds:.1} per_forward_ms={:.3} largest_logit_radius={largest_radius:.3e} memory={}",
            checkpoint.name,
            checkpoint.labels,
            checkpoint.step,
            p * p,
            1e3 * native_seconds * rayon::current_num_threads() as f64 / (p * p) as f64,
            memory()
        );

        // Torch's float64 logits on the test pairs: a measured agreement (torch divides the scores by
        // √d_head, the native core multiplies by its rounded reciprocal).
        let torch = matrix(
            &run.join(format!("native_logits.{}.npy", checkpoint.name)),
            NpyFloat::F8,
            checkpoint.test_pairs.len(),
            p,
        )?;
        let mut torch_gap = 0.0_f64;
        let mut beyond_radius = 0usize;
        for (row, &[a, b]) in checkpoint.test_pairs.iter().enumerate() {
            let (values, radii) = native.row(a * p + b);
            let mut beyond = false;
            for (column, (&value, &radius)) in values.iter().zip(radii).enumerate() {
                let gap = (value - torch[[row, column]]).abs();
                torch_gap = torch_gap.max(gap);
                beyond |= gap > radius;
            }
            beyond_radius += usize::from(beyond);
        }
        println!(
            "[control] {} torch float64 logits: largest gap {torch_gap:.3e} (measured), rows beyond the native radius {beyond_radius} of {}; torch control {}",
            checkpoint.name,
            checkpoint.test_pairs.len(),
            checkpoint.torch_control
        );

        // Executor control: the permuted table read at the site reproduces the shifted native rows bit
        // for bit, and the global edit differs from the use-specific one.
        let started = Instant::now();
        let permuted = permuted_table(&network.table, settings.fit_shift, p);
        let mut control = serde_json::Map::new();
        let mut site_outputs = Vec::new();
        for site in [Site::Pos0, Site::Global] {
            let mut executed = LogitTable::new(checkpoint.test_pairs.len(), p);
            edited_rows(&network, &permuted, site, &checkpoint.test_pairs, p, &mut executed, 0)?;
            let mismatched = checkpoint
                .test_pairs
                .iter()
                .enumerate()
                .filter(|&(row, &pair)| {
                    let [a, b] = site.shifted(pair, settings.fit_shift, p);
                    let ((values, radii), (native_values, native_radii)) = (executed.row(row), native.row(a * p + b));
                    !(bitwise_equal(values, native_values) && bitwise_equal(radii, native_radii))
                })
                .count();
            control.insert(format!("{}_bitwise_mismatched_rows", site.name()), json!(mismatched));
            site_outputs.push(executed);
        }
        let differing = (0..checkpoint.test_pairs.len())
            .filter(|&row| argmax(site_outputs[0].row(row).0) != argmax(site_outputs[1].row(row).0))
            .count();
        control.insert("pos0_vs_global_argmax_differences".to_string(), json!(differing));
        control.insert("seconds".to_string(), json!(started.elapsed().as_secs_f64()));
        println!("[control] {} native executor {} memory={}", checkpoint.name, Value::Object(control.clone()), memory());

        let successor: Vec<usize> = (0..=p).map(|row| if row < p { (row + 1) % p } else { row }).collect();
        let cycle = RowCycle::from_successor(&successor).map_err(|error| error.to_string())?;
        let planes = cyclic_planes(network.table.view(), &cycle).map_err(|error| error.to_string())?;
        let plane_count = cycle.plane_count();
        let mut order: Vec<usize> = (1..=plane_count).collect();
        order.sort_by(|left, right| planes.power[right - 1].total_cmp(&planes.power[left - 1]));
        let prefix = |kept: usize| {
            let mut frequencies = order[..kept].to_vec();
            frequencies.sort_unstable();
            frequencies
        };
        println!("[planes] {} power order head {:?}", checkpoint.name, &order[..order.len().min(12)]);

        let mut site_reports = serde_json::Map::new();
        for (site_index, &site) in sites.iter().enumerate() {
            // Fit: the least power-ordered prefix whose decoded program meets each rung of each ladder at
            // the fit shift on the training pairs.
            let fit_family = Family {
                checkpoint: checkpoint.name.clone(),
                site: site.name(),
                shifts: vec![settings.fit_shift],
                pairs: checkpoint.train_pairs.len(),
            };
            let fit_reference = reference_rows(&native, site, &[settings.fit_shift], &checkpoint.train_pairs, p);
            let fit_target = Target {
                site,
                shifts: &[settings.fit_shift],
                pairs: &checkpoint.train_pairs,
                reference: &fit_reference,
                family: &fit_family,
            };
            let mut chosen = Chosen {
                kl: vec![None; settings.tolerances.len()],
                disagreement: vec![None; settings.disagreements.len()],
            };
            let mut fit_curve = Vec::new();
            for kept in 1..=plane_count {
                if chosen.kl.iter().chain(&chosen.disagreement).all(Option::is_some) {
                    break;
                }
                let scored = match score_program(&network, &planes, &prefix(kept), precision, &fit_target, &ladders) {
                    Ok(scored) => scored,
                    Err(error) => {
                        println!("[fit] {} {} K={kept} refused: {error}", checkpoint.name, site.name());
                        fit_curve.push(json!({ "K": kept, "refused": error }));
                        continue;
                    }
                };
                for (rung, fidelity) in scored.kl_fidelities.iter().enumerate() {
                    if chosen.kl[rung].is_none() && fidelity.verdict() == FidelityVerdict::Meets {
                        chosen.kl[rung] = Some(kept);
                    }
                }
                for (rung, fidelity) in scored.disagreement_fidelities.iter().enumerate() {
                    if chosen.disagreement[rung].is_none() && fidelity.verdict() == FidelityVerdict::Meets {
                        chosen.disagreement[rung] = Some(kept);
                    }
                }
                println!(
                    "[fit] {} {} K={kept} bits={} {} seconds={:.1} memory={}",
                    checkpoint.name,
                    site.name(),
                    scored.code.total_bits(),
                    score_line(&scored.score),
                    scored.seconds,
                    memory()
                );
                fit_curve.push(scored_json(&scored));
            }
            println!(
                "[fit] {} {} smallest K per rung: kl {:?} over {:?}; disagreement {:?} over {:?}",
                checkpoint.name,
                site.name(),
                chosen.kl,
                settings.tolerances,
                chosen.disagreement,
                settings.disagreements
            );

            // Held out: the chosen programs, the trained model's chosen K (for every other checkpoint),
            // the full plane set and the unedited model, on the test pairs at every held-out shift.
            let family = Family {
                checkpoint: checkpoint.name.clone(),
                site: site.name(),
                shifts: settings.held_out_shifts.clone(),
                pairs: checkpoint.test_pairs.len(),
            };
            let reference = reference_rows(&native, site, &settings.held_out_shifts, &checkpoint.test_pairs, p);
            let target = Target {
                site,
                shifts: &settings.held_out_shifts,
                pairs: &checkpoint.test_pairs,
                reference: &reference,
                family: &family,
            };
            let unedited = native_rows(&native, &settings.held_out_shifts, &checkpoint.test_pairs, p, |_, pair| pair);
            let baseline = family_score(&unedited, &reference, &family, ladders.quantiles)?;
            println!("[held_out] {} {} baseline {} memory={}", checkpoint.name, site.name(), score_line(&baseline), memory());
            let full_frequencies: Vec<usize> = (1..=plane_count).collect();
            let full = score_program(&network, &planes, &full_frequencies, precision, &target, &ladders)?;
            println!(
                "[held_out] {} {} full set K={plane_count} bits={} {} seconds={:.1} memory={}",
                checkpoint.name,
                site.name(),
                full.code.total_bits(),
                score_line(&full.score),
                full.seconds,
                memory()
            );
            let trained_chosen = trained_sites[site_index].as_ref().map(|trained| trained.chosen.clone());
            let mut distinct: Vec<usize> = chosen
                .kl
                .iter()
                .chain(&chosen.disagreement)
                .chain(trained_chosen.iter().flat_map(|trained| trained.kl.iter().chain(&trained.disagreement)))
                .flatten()
                .copied()
                .filter(|&kept| kept < plane_count)
                .collect();
            distinct.sort_unstable();
            distinct.dedup();
            let mut candidates = Vec::new();
            for &kept in &distinct {
                let scored = score_program(&network, &planes, &prefix(kept), precision, &target, &ladders)?;
                println!(
                    "[held_out] {} {} K={kept} bits={} {} seconds={:.1} memory={}",
                    checkpoint.name,
                    site.name(),
                    scored.code.total_bits(),
                    score_line(&scored.score),
                    scored.seconds,
                    memory()
                );
                candidates.push(scored);
            }
            let find = |kept: Option<usize>| kept.and_then(|kept| candidates.iter().find(|scored| scored.frequencies.len() == kept));
            let kl_proposals: Vec<Value> = settings
                .tolerances
                .iter()
                .enumerate()
                .map(|(rung, &tolerance)| {
                    let candidate = find(chosen.kl[rung]).map(|scored| (scored, &scored.kl_fidelities[rung], &scored.score.kl));
                    proposal_json(tolerance, chosen.kl[rung], plane_count, &full, &full.kl_fidelities[rung], candidate)
                })
                .collect();
            let disagreement_proposals: Vec<Value> = settings
                .disagreements
                .iter()
                .enumerate()
                .map(|(rung, &tolerance)| {
                    let candidate = find(chosen.disagreement[rung])
                        .map(|scored| (scored, &scored.disagreement_fidelities[rung], &scored.score.disagreement));
                    proposal_json(tolerance, chosen.disagreement[rung], plane_count, &full, &full.disagreement_fidelities[rung], candidate)
                })
                .collect();
            for proposal in kl_proposals.iter().chain(&disagreement_proposals) {
                println!("[proposal] {} {} {proposal}", checkpoint.name, site.name());
            }

            // A control read at the trained model's K: at every rung where the trained program met, the
            // control's program of that size must miss, with its witness row.
            let mut at_trained = Vec::new();
            if let (false, Some(trained)) = (is_trained, trained_sites[site_index].as_ref()) {
                for (ladder, trained_met, rungs) in [
                    ("kl", &trained.kl_met, &settings.tolerances),
                    ("disagreement", &trained.disagreement_met, &settings.disagreements),
                ] {
                    for (rung, &tolerance) in rungs.iter().enumerate() {
                        let Some(scored) = find(trained_met[rung]) else {
                            at_trained.push(json!({
                                "ladder": ladder, "tolerance": tolerance,
                                "outcome": "no fit-chosen trained program met this rung held out: no separation to test",
                            }));
                            continue;
                        };
                        let (fidelity, status) = if ladder == "kl" {
                            (&scored.kl_fidelities[rung], &scored.score.kl)
                        } else {
                            (&scored.disagreement_fidelities[rung], &scored.score.disagreement)
                        };
                        let outcome = control_verdict(true, fidelity);
                        println!(
                            "[at_trained_K] {} {} {ladder} tolerance={tolerance} K={} {outcome}: {} row {}",
                            checkpoint.name,
                            site.name(),
                            scored.frequencies.len(),
                            status_json(status),
                            witness_json(status)
                        );
                        at_trained.push(json!({
                            "ladder": ladder, "tolerance": tolerance, "K": scored.frequencies.len(), "outcome": outcome,
                            "status": status_json(status), "row": witness_json(status),
                        }));
                    }
                }
            }
            if is_trained {
                // `candidates` ascend in `K`, so the first that meets a rung is the least.
                let met = |rungs: usize, fidelities: fn(&Scored) -> &Vec<DecodedFidelity<Witness, Family>>| -> Vec<Option<usize>> {
                    (0..rungs)
                        .map(|rung| {
                            candidates
                                .iter()
                                .find(|scored| fidelities(scored)[rung].verdict() == FidelityVerdict::Meets)
                                .map(|scored| scored.frequencies.len())
                        })
                        .collect()
                };
                trained_sites[site_index] = Some(TrainedSite {
                    kl_met: met(settings.tolerances.len(), kl_fidelities),
                    disagreement_met: met(settings.disagreements.len(), disagreement_fidelities),
                    chosen: chosen.clone(),
                });
            }

            // Control C5, null edits on the trained model at each K whose chosen program met a rung held out, up
            // to half the planes: the K planes of least power with their own angles, against the shifted
            // reference (it must miss where the chosen program met) and against the unedited model (the
            // measured shift, which must meet the rung); and a Haar-random basis of 2K dimensions carrying the
            // chosen planes' angles, against the shifted reference (it must miss).
            let mut null_edits = Vec::new();
            if let (true, Some(trained)) = (is_trained, trained_sites[site_index].as_ref()) {
                let unedited_target = Target { reference: &unedited, ..target };
                let mut met: Vec<usize> = trained.kl_met.iter().chain(&trained.disagreement_met).flatten().copied().collect();
                met.sort_unstable();
                met.dedup();
                for &kept in met.iter().filter(|&&kept| 2 * kept <= plane_count) {
                    let mut bottom = order[plane_count - kept..].to_vec();
                    bottom.sort_unstable();
                    let shifted = score_program(&network, &planes, &bottom, precision, &target, &ladders)?;
                    let unmoved = score_program(&network, &planes, &bottom, precision, &unedited_target, &ladders)?;
                    let random = random_plane_rows(&network, &planes, &prefix(kept), &mut null_rng, &target)?;
                    let random_score = family_score(&random, &reference, &family, ladders.quantiles)?;
                    let mut rungs = Vec::new();
                    for (ladder, trained_met, tolerances) in [
                        ("kl", &trained.kl_met, &settings.tolerances),
                        ("disagreement", &trained.disagreement_met, &settings.disagreements),
                    ] {
                        for (rung, &tolerance) in tolerances.iter().enumerate() {
                            if trained_met[rung] != Some(kept) {
                                continue;
                            }
                            let (bottom_fidelity, unmoved_fidelity, random_status) = if ladder == "kl" {
                                (&shifted.kl_fidelities[rung], &unmoved.kl_fidelities[rung], &random_score.kl)
                            } else {
                                (&shifted.disagreement_fidelities[rung], &unmoved.disagreement_fidelities[rung], &random_score.disagreement)
                            };
                            let unmoved_outcome = match unmoved_fidelity.verdict() {
                                FidelityVerdict::Meets => "no measured shift, as required",
                                FidelityVerdict::Violates => "FINDING: the null edit moves the model beyond the rung",
                                FidelityVerdict::Unresolved => "unresolved",
                            };
                            let random_outcome = if random_status.refutes_at_most(tolerance) {
                                "fails as required"
                            } else if random_status.certifies_at_most(tolerance) {
                                "FINDING: the random planes meet the rung at the trained model's K"
                            } else {
                                "unresolved"
                            };
                            let entry = json!({
                                "ladder": ladder, "tolerance": tolerance, "K": kept,
                                "bottom_vs_shifted": control_verdict(true, bottom_fidelity),
                                "bottom_vs_unedited": unmoved_outcome,
                                "random_vs_shifted": random_outcome,
                            });
                            println!("[null_edit] {} {} {entry}", checkpoint.name, site.name());
                            rungs.push(entry);
                        }
                    }
                    null_edits.push(json!({
                        "K": kept,
                        "bottom_frequencies": bottom,
                        "bottom_vs_shifted": scored_json(&shifted),
                        "bottom_vs_unedited": scored_json(&unmoved),
                        "random_vs_shifted": score_json(&random_score),
                        "rungs": rungs,
                    }));
                }
            }
            // Stage S3: the minimum-code plane support on the fit family, with its OR edges, and the support's
            // program held out.
            let mut support_reports = Vec::new();
            if let Some(support) = settings.support.as_ref().filter(|support| support.sites.iter().any(|name| name == site.name())) {
                let every: Vec<usize> = (1..=plane_count).collect();
                let full_code = plane_program_code(&planes, &every, precision).map_err(|error| error.to_string())?;
                let basis = decoded_basis(&full_code.basis.decode()?, network.table.ncols())?;
                let (contributions, formation) = plane_contributions(&network.table, &planes, &basis, settings.fit_shift)?;
                let mut plane_bits = 0u64;
                for frequency in 1..=plane_count {
                    let own = plane_program_code(&planes, &[frequency], precision).map_err(|error| error.to_string())?;
                    plane_bits = plane_bits.max(own.basis_bits);
                }
                let family = PlaneFamily {
                    checkpoint: checkpoint.name.clone(),
                    site: site.name(),
                    shift: settings.fit_shift,
                    pairs: checkpoint.train_pairs.len(),
                    planes: plane_count,
                };
                let mut enclosed = BTreeMap::new();
                for &tolerance in &support.tolerances {
                    let found = {
                        let program = PlaneBoxes {
                            network: &network,
                            contributions: &contributions,
                            formation: &formation,
                            site,
                            pairs: &checkpoint.train_pairs,
                            reference: &fit_reference,
                            family: &family,
                            power: &planes.power,
                            enclosed: &mut enclosed,
                        };
                        let oracle = BoxSeparationOracle::new(program, tolerance).map_err(|error| format!("{error:?}"))?;
                        let mut oracle = Logged {
                            oracle,
                            label: format!("{} {} tolerance={tolerance:.3e}", checkpoint.name, site.name()),
                            separations: 0,
                            started: Instant::now(),
                        };
                        minimum_code_support(&mut oracle, &PlaneCode { plane_bits }, tolerance, FailureHypergraph::new(plane_count))
                    };
                    let (code_status, bits, frequencies, separations, edges) = match &found {
                        Ok(search) => {
                            let (status, bits) = match &search.code {
                                EvidenceStatus::Exact { value, .. } => ("exact", json!([value, value])),
                                EvidenceStatus::Unresolved { lower, upper, .. } => ("unresolved", json!([lower, upper])),
                                other => return Err(format!("an unexpected code status {other:?}")),
                            };
                            let edges: Vec<Vec<usize>> = search
                                .hypergraph
                                .edges()
                                .iter()
                                .map(|edge| edge.perturbed.members().iter().map(|&plane| plane + 1).collect())
                                .collect();
                            let frequencies = search
                                .certified
                                .as_ref()
                                .map(|found| found.support.members().iter().map(|&plane| plane + 1).collect::<Vec<usize>>());
                            (status, bits, frequencies, search.separations, edges)
                        }
                        Err(SupportSearchError::AllOnViolation) => ("all_on_violation", Value::Null, None, 0, Vec::new()),
                        Err(error) => return Err(format!("{} {} tolerance {tolerance}: {error:?}", checkpoint.name, site.name())),
                    };
                    let held_out = match &frequencies {
                        Some(frequencies) if !frequencies.is_empty() => {
                            Some(score_program(&network, &planes, frequencies, precision, &target, &ladders)?)
                        }
                        _ => None,
                    };
                    println!(
                        "[support] {} {} tolerance={tolerance:.3e} code={code_status} bits={bits} support={frequencies:?} separations={separations} edges={} held_out={}",
                        checkpoint.name,
                        site.name(),
                        edges.len(),
                        held_out.as_ref().map_or_else(|| "none".to_string(), |scored| score_line(&scored.score))
                    );
                    support_reports.push(json!({
                        "tolerance": tolerance,
                        "code_status": code_status,
                        "code_bits": bits,
                        "support": frequencies,
                        "separations": separations,
                        "edges": edges,
                        "enclosures": enclosed.len(),
                        "held_out": held_out.as_ref().map(scored_json),
                    }));
                }
            }
            site_reports.insert(
                site.name().to_string(),
                json!({
                    "fit_family": family_json(&fit_family),
                    "fit": fit_curve,
                    "smallest_K": { "kl": chosen.kl, "disagreement": chosen.disagreement },
                    "held_out": {
                        "family": family_json(&family),
                        "baseline": score_json(&baseline),
                        "full_set": scored_json(&full),
                        "candidates": candidates.iter().map(scored_json).collect::<Vec<_>>(),
                    },
                    "proposals": { "kl": kl_proposals, "disagreement": disagreement_proposals },
                    "at_trained_K": at_trained,
                    "null_edits": null_edits,
                    "support": support_reports,
                }),
            );
        }
        checkpoints.push(json!({
            "name": checkpoint.name,
            "labels": checkpoint.labels,
            "step": checkpoint.step,
            "trained": is_trained,
            "native_seconds": native_seconds,
            "largest_logit_radius": largest_radius,
            "torch_agreement": { "largest_gap": torch_gap, "rows_beyond_native_radius": beyond_radius, "measured": true },
            "torch_control": checkpoint.torch_control,
            "native_control": Value::Object(control),
            "power_order": order,
            "sites": Value::Object(site_reports),
        }));
        write_report(&out, &report(&checkpoints))?;
    }
    println!("[receipt] wrote {} memory={}", out.display(), memory());
    Ok(())
}
