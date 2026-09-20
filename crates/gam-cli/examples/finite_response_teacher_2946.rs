//! Teacher receipts for #2946 on an EXECUTED real MLP block (Pythia exact GELU,
//! Qwen3 SwiGLU). The torch side,
//! `experiments/issue-2946/finite_response_teacher/export_block.py`, only runs
//! forward passes and writes arrays; every number here is computed in Rust.
//!
//! Stages, in pipeline order:
//!
//! * `declare`  post-norm rows -> the declared law `h = h0 + L Z`, `Z ~ N(0, I_k)`:
//!   `h0` is the rows' mean and `L = V_k S_k / sqrt(n - 1)` comes from the thin
//!   SVD of the centred rows, so the law carries the context set's mean and its
//!   top-`k` covariance. Writes `h0.npy`, `L.npy`, `law.json`.
//! * `draw`     seeded `Z`, an independent `Z~`, and for every frame the coupled
//!   `Z' = PZ + (I - P)Z~` of R4. Writes `z.npy`, `z_tilde.npy`, every block's
//!   `h` rows stacked into one `h.npy` (blocks `Z`, `Z~`, then the frames), and
//!   `draw.json`. A principal frame copies its retained coordinates from `Z`
//!   exactly, so the frame `all:k` gives `Z' = Z` bit for bit. Its executed
//!   error then reads only the row-position rounding of forming `h` and of the
//!   batched torch execution (about 1e-17 per row on Pythia-70m, #2946): a
//!   row-alignment control, since misaligned rows give `E(P) ≈ V(I)`.
//! * `forward`  this file's `F(h)` from the exported parameters against the
//!   executed outputs, with the first-order rounding bound of one evaluation
//!   (a sum of `n` rounded products is within `n eps sum|terms|`). Two
//!   evaluations differ by at most twice that bound, plus the relative accuracy
//!   of the two activation implementations, which this stage does not bound.
//!   A wrong orientation or activation shows up as an O(1) relative gap.
//! * `mc`       Monte Carlo of the executed block under the declared law, output
//!   metric `M = I`. Given `PZ`, `F(Z)` and `F(Z')` are independent draws of the
//!   same conditional law, so `E<F(Z), F(Z')> = V(P) + |m|^2` and
//!   `E|F(Z) - F(Z')|^2 = 2 (V(I) - V(P)) = 2 E(P)`. Row by row:
//!   `V(I) = E|F(Z) - F(Z~)|^2 / 2`, `E(P) = E|F(Z) - F(Z')|^2 / 2`, and
//!   `V(P)` from the paired difference, each with standard error `sd / (2 sqrt n)`.
//! * `analytic` (A6, the gated Qwen3 MLP) the declared law absorbed into the
//!   SwiGLU block, gate readers `W_gate L` and biases `W_gate h0`, up readers
//!   `W_up L` and biases `W_up h0`, writers `W_down`, by
//!   `gam_sae::response::raw_block::UnabsorbedGatedBlock::absorb` into a
//!   `KnownGatedBlock`, and its Stein gated mean `g = F̄_P` on the draws of each
//!   principal frame. By R2, `E|F(Z) - g(PZ)|^2 = E(P) + E|F̄_P - g|^2`, so the
//!   row-paired `D = |F(Z) - g(PZ)|^2 - |F(Z) - F(Z')|^2 / 2` has
//!   `E D = E|F̄_P - g|^2`, which is 0 exactly when `g` is the conditional mean and
//!   at most the mean squared quadrature band for the computed one. The
//!   unsmoothed `g0 = F(Pz)`, which ignores the discarded input, is its positive
//!   control. The dense receipt (A3) is fr-commutant's runner.
//!
//! ```text
//! cargo run --release -p gam-cli --example finite_response_teacher_2946 -- declare --harvest H --rank K --out LAW
//! cargo run --release -p gam-cli --example finite_response_teacher_2946 -- draw --law LAW --rows N --seed S \
//!     --principal top8:8 --principal all:K [--frame NAME:Q.npy] --out DRAW
//! python export_block.py execute --model M --layer L --inputs DRAW/h.npy --batch B --out DRAW/executed.npy
//! cargo run --release -p gam-cli --example finite_response_teacher_2946 -- forward --harvest H --draw DRAW \
//!     --executed DRAW/executed.npy --rows R --out DRAW/forward.json
//! cargo run --release -p gam-cli --example finite_response_teacher_2946 -- mc --draw DRAW \
//!     --executed DRAW/executed.npy --out DRAW/receipt.json
//! cargo run --release -p gam-cli --example finite_response_teacher_2946 -- analytic --harvest H --law LAW \
//!     --draw DRAW --executed DRAW/executed.npy --rows N --out DRAW/analytic.json
//! ```

use clap::{Parser, Subcommand};
use gam::faer_ndarray::{FaerSvd, fast_atb};
use gam_sae::response::raw_block::UnabsorbedGatedBlock;
use ndarray::{Array1, Array2, ArrayD, ArrayView2, Axis, IxDyn, s};
use npyz::{NpyFile, Order, WriterBuilder};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use statrs::function::erf::erfc;
use std::collections::BTreeMap;
use std::f64::consts::{PI, SQRT_2};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Teacher receipts for #2946 on an executed real MLP block")]
struct Cli {
    #[command(subcommand)]
    stage: Stage,
}

#[derive(Subcommand)]
enum Stage {
    /// Declare `h0` and `L` from the harvested post-norm rows.
    Declare {
        #[arg(long)]
        harvest: PathBuf,
        /// Retained principal rank `k` of the declared factor `L`.
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Draw `Z`, `Z~` and each frame's coupled `Z'`, and write their `h` rows.
    Draw {
        #[arg(long)]
        law: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        seed: u64,
        /// A frame retaining the leading principal coordinates, as `NAME:COUNT`.
        #[arg(long)]
        principal: Vec<String>,
        /// A frame spanned by an orthonormal `(k x q)` float64 .npy, as `NAME:PATH`.
        #[arg(long)]
        frame: Vec<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Compare this file's `F(h)` with the executed outputs on the leading rows.
    Forward {
        #[arg(long)]
        harvest: PathBuf,
        #[arg(long = "draw")]
        draw_dir: PathBuf,
        #[arg(long)]
        executed: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Monte Carlo `V(I)`, `E(P)` and `V(P)` of the executed block.
    Mc {
        #[arg(long = "draw")]
        draw_dir: PathBuf,
        #[arg(long)]
        executed: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// A6: the Stein gated mean of the absorbed Qwen3 block against the executed block, by the R2 paired statistic.
    Analytic {
        #[arg(long)]
        harvest: PathBuf,
        #[arg(long)]
        law: PathBuf,
        #[arg(long = "draw")]
        draw_dir: PathBuf,
        #[arg(long)]
        executed: PathBuf,
        /// Leading draw rows the retained-response statistic reads.
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> ExitCode {
    let outcome = match Cli::parse().stage {
        Stage::Declare { harvest, rank, out } => declare(&harvest, rank, &out),
        Stage::Draw {
            law,
            rows,
            seed,
            principal,
            frame,
            out,
        } => draw(&law, rows, seed, &principal, &frame, &out),
        Stage::Forward {
            harvest,
            draw_dir,
            executed,
            rows,
            out,
        } => forward_check(&harvest, &draw_dir, &executed, rows, &out),
        Stage::Mc {
            draw_dir,
            executed,
            out,
        } => monte_carlo(&draw_dir, &executed, &out),
        Stage::Analytic {
            harvest,
            law,
            draw_dir,
            executed,
            rows,
            out,
        } => analytic(&harvest, &law, &draw_dir, &executed, rows, &out),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("[finite_response_teacher_2946] error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn declare(harvest: &Path, rank: usize, out: &Path) -> Result<(), String> {
    let started = Instant::now();
    let meta = read_json(&harvest.join("meta.json"))?;
    let rows = read_rows::<f32>(&harvest.join("post_norm.npy"), None)?;
    let read_seconds = started.elapsed().as_secs_f64();
    let (n, d) = rows.dim();
    if n < 2 {
        return Err(format!("{n} post-norm rows cannot carry a covariance"));
    }
    let h0 = rows
        .mean_axis(Axis(0))
        .ok_or("post_norm.npy holds no rows")?;
    let centred = (&rows - &h0) * (1.0 / ((n - 1) as f64).sqrt());
    let centred_seconds = started.elapsed().as_secs_f64();
    let factors = centred
        .svd(false, true)
        .map_err(|err| format!("thin SVD of the centred post-norm rows: {err}"))?;
    let svd_seconds = started.elapsed().as_secs_f64();
    let singular = factors.1;
    let vt = factors.2.ok_or("the SVD returned no right singular vectors")?;
    let mut order: Vec<usize> = (0..singular.len()).collect();
    order.sort_by(|a, b| singular[*b].total_cmp(&singular[*a]));
    if rank == 0 || rank > order.len() {
        return Err(format!(
            "--rank {rank} outside 1..={} (rows {n}, hidden {d})",
            order.len()
        ));
    }
    if singular[order[rank - 1]] == 0.0 {
        return Err(format!("the post-norm rows have rank below {rank}"));
    }
    let mut factor = Array2::<f64>::zeros((d, rank));
    for (column, &index) in order[..rank].iter().enumerate() {
        factor
            .column_mut(column)
            .assign(&(&vt.row(index) * singular[index]));
    }
    let variances: Vec<f64> = order.iter().map(|&i| singular[i] * singular[i]).collect();
    let total: f64 = variances.iter().sum();
    let retained: f64 = variances[..rank].iter().sum();

    std::fs::create_dir_all(out).map_err(|err| format!("create {}: {err}", out.display()))?;
    write_npy(&out.join("h0.npy"), &[d as u64], h0.iter().copied())?;
    write_npy(
        &out.join("L.npy"),
        &[d as u64, rank as u64],
        factor.iter().copied(),
    )?;
    write_json(
        &out.join("law.json"),
        &json!({
            "stage": "declare",
            "harvest": harvest.display().to_string(),
            "model": meta.get("model").cloned().unwrap_or(Value::Null),
            "revision": meta.get("revision").cloned().unwrap_or(Value::Null),
            "model_type": meta.get("model_type").cloned().unwrap_or(Value::Null),
            "hidden_act": meta.get("hidden_act").cloned().unwrap_or(Value::Null),
            "layer": meta.get("layer").cloned().unwrap_or(Value::Null),
            "context_set": meta.get("context_set").cloned().unwrap_or(Value::Null),
            "declared": "h = h0 + L Z, Z ~ N(0, I_k); h0 = mean of the post-norm rows; L = V_k S_k / sqrt(n-1) from the thin SVD of the centred rows",
            "rows": n,
            "hidden": d,
            "rank": rank,
            "total_variance": total,
            "retained_variance": retained,
            "retained_fraction": retained / total,
            "principal_variances": variances,
            "seconds": {
                "read": read_seconds,
                "centre": centred_seconds - read_seconds,
                "svd": svd_seconds - centred_seconds,
                "factor_and_write": started.elapsed().as_secs_f64() - svd_seconds,
            },
        }),
    )?;
    println!(
        "[declare] rows={n} hidden={d} rank={rank} retained variance {retained:.6e} of {total:.6e} ({:.4}); seconds read={read_seconds:.1} centre={:.1} svd={:.1} total={:.1}",
        retained / total,
        centred_seconds - read_seconds,
        svd_seconds - centred_seconds,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// A retained subspace of the latent `Z ~ N(0, I_k)`.
enum Frame {
    /// The leading `count` principal coordinates.
    Principal(usize),
    /// The span of an orthonormal `(k x q)` matrix.
    Span(Array2<f64>),
}

fn draw(
    law: &Path,
    rows: usize,
    seed: u64,
    principal: &[String],
    frames: &[String],
    out: &Path,
) -> Result<(), String> {
    let h0 = read_vector(&law.join("h0.npy"))?;
    let factor = read_rows::<f64>(&law.join("L.npy"), None)?;
    let (d, k) = factor.dim();
    if h0.len() != d {
        return Err(format!("h0 has {} entries but L has {d} rows", h0.len()));
    }
    if rows < 2 {
        return Err(format!("--rows {rows} cannot carry a standard error"));
    }

    let mut named: Vec<(String, Frame, Value)> = Vec::new();
    for spec in principal {
        let (name, count) = spec
            .split_once(':')
            .ok_or_else(|| format!("--principal {spec}: expected NAME:COUNT"))?;
        let count: usize = count
            .parse()
            .map_err(|err| format!("--principal {spec}: {err}"))?;
        if count == 0 || count > k {
            return Err(format!("--principal {spec}: COUNT outside 1..={k}"));
        }
        named.push((
            name.to_string(),
            Frame::Principal(count),
            json!({ "kind": "principal", "retained": count }),
        ));
    }
    for spec in frames {
        let (name, path) = spec
            .split_once(':')
            .ok_or_else(|| format!("--frame {spec}: expected NAME:PATH"))?;
        let q = read_rows::<f64>(Path::new(path), None)?;
        if q.nrows() != k || q.ncols() == 0 || q.ncols() > k {
            return Err(format!(
                "--frame {spec}: shape {:?} is not (k={k}) x (1..={k})",
                q.dim()
            ));
        }
        let gram = fast_atb(&q, &q);
        let defect = gram
            .indexed_iter()
            .map(|((i, j), &v)| (v - if i == j { 1.0 } else { 0.0 }).abs())
            .fold(0.0_f64, f64::max);
        let info = json!({
            "kind": "span",
            "path": path,
            "retained": q.ncols(),
            "orthonormality_defect": defect,
        });
        named.push((name.to_string(), Frame::Span(q), info));
    }
    for (index, (name, ..)) in named.iter().enumerate() {
        if name == "Z" || name == "Z~" || named[..index].iter().any(|other| &other.0 == name) {
            return Err(format!("frame name {name:?} is reserved or repeated"));
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let z = gaussian_rows(&mut rng, rows, k);
    let z_tilde = gaussian_rows(&mut rng, rows, k);

    std::fs::create_dir_all(out).map_err(|err| format!("create {}: {err}", out.display()))?;
    write_npy(
        &out.join("z.npy"),
        &[rows as u64, k as u64],
        z.iter().copied(),
    )?;
    write_npy(
        &out.join("z_tilde.npy"),
        &[rows as u64, k as u64],
        z_tilde.iter().copied(),
    )?;

    let blocks = 2 + named.len();
    let h_path = out.join("h.npy");
    let file = File::create(&h_path).map_err(|err| format!("create {}: {err}", h_path.display()))?;
    let mut writer = npyz::WriteOptions::<f64>::new()
        .default_dtype()
        .shape(&[(rows * blocks) as u64, d as u64])
        .writer(BufWriter::new(file))
        .begin_nd()
        .map_err(|err| format!("begin {}: {err}", h_path.display()))?;
    let mut layout = vec![
        json!({ "block": 0, "name": "Z", "first_row": 0 }),
        json!({ "block": 1, "name": "Z~", "first_row": rows }),
    ];
    writer
        .extend(intervention_rows(&h0, &factor, &z).iter().copied())
        .map_err(|err| format!("write {}: {err}", h_path.display()))?;
    writer
        .extend(intervention_rows(&h0, &factor, &z_tilde).iter().copied())
        .map_err(|err| format!("write {}: {err}", h_path.display()))?;
    for (offset, (name, frame, info)) in named.iter().enumerate() {
        let coupled = coupled_latent(frame, &z, &z_tilde);
        writer
            .extend(intervention_rows(&h0, &factor, &coupled).iter().copied())
            .map_err(|err| format!("write {}: {err}", h_path.display()))?;
        layout.push(json!({
            "block": offset + 2,
            "name": name,
            "first_row": (offset + 2) * rows,
            "frame": info,
        }));
    }
    writer
        .finish()
        .map_err(|err| format!("finish {}: {err}", h_path.display()))?;

    write_json(
        &out.join("draw.json"),
        &json!({
            "stage": "draw",
            "law": law.display().to_string(),
            "seed": seed,
            "sampling": "StdRng::seed_from_u64(seed); rand_distr StandardNormal, Z then Z~, row-major",
            "coupling": "Z' = PZ + (I-P)Z~",
            "rows": rows,
            "rank": k,
            "hidden": d,
            "blocks": layout,
        }),
    )?;
    println!(
        "[draw] seed={seed} rows={rows} rank={k} hidden={d} blocks={blocks} -> {}",
        h_path.display()
    );
    Ok(())
}

fn gaussian_rows(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || {
        Distribution::<f64>::sample(&StandardNormal, &mut *rng)
    })
}

fn coupled_latent(frame: &Frame, z: &Array2<f64>, z_tilde: &Array2<f64>) -> Array2<f64> {
    match frame {
        Frame::Principal(count) => {
            let mut coupled = z_tilde.clone();
            coupled
                .slice_mut(s![.., ..*count])
                .assign(&z.slice(s![.., ..*count]));
            coupled
        }
        Frame::Span(q) => {
            let difference = z - z_tilde;
            let retained = fast_atb(&difference.t(), q);
            z_tilde + &fast_atb(&retained.t(), &q.t())
        }
    }
}

/// `h = h0 + L z` for every latent row.
fn intervention_rows(h0: &Array1<f64>, factor: &Array2<f64>, latent: &Array2<f64>) -> Array2<f64> {
    fast_atb(&latent.t(), &factor.t()) + h0
}

#[derive(Clone, Copy)]
enum Activation {
    /// Exact GELU `t Phi(t)`, HF `hidden_act = "gelu"`.
    Gelu,
    /// SiLU `t logistic(t)`, HF `hidden_act = "silu"`.
    Silu,
}

impl Activation {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "gelu" => Ok(Self::Gelu),
            "silu" => Ok(Self::Silu),
            other => Err(format!(
                "hidden_act {other:?} is not a declared #2946 primitive (exact gelu or silu)"
            )),
        }
    }

    fn value(self, t: f64) -> f64 {
        match self {
            Self::Gelu => t * 0.5 * erfc(-t / SQRT_2),
            Self::Silu => t * logistic(t),
        }
    }

    fn slope(self, t: f64) -> f64 {
        match self {
            Self::Gelu => 0.5 * erfc(-t / SQRT_2) + t * (-0.5 * t * t).exp() / (2.0 * PI).sqrt(),
            Self::Silu => {
                let p = logistic(t);
                p * (1.0 + t * (1.0 - p))
            }
        }
    }
}

fn logistic(t: f64) -> f64 {
    if t >= 0.0 {
        1.0 / (1.0 + (-t).exp())
    } else {
        let e = t.exp();
        e / (1.0 + e)
    }
}

enum Block {
    /// GPT-NeoX: `F(h) = W_out act(W_in h + b_in) + b_out`.
    Dense {
        w_in: Array2<f64>,
        b_in: Array1<f64>,
        w_out: Array2<f64>,
        b_out: Array1<f64>,
    },
    /// Qwen3: `F(h) = W_down (act(W_gate h) * W_up h)`.
    Gated {
        gate: Array2<f64>,
        up: Array2<f64>,
        down: Array2<f64>,
    },
}

impl Block {
    fn load(harvest: &Path) -> Result<Self, String> {
        let param = |name: &str| harvest.join(format!("{name}.npy"));
        if param("dense_h_to_4h.weight").exists() {
            Ok(Self::Dense {
                w_in: read_rows::<f64>(&param("dense_h_to_4h.weight"), None)?,
                b_in: read_vector(&param("dense_h_to_4h.bias"))?,
                w_out: read_rows::<f64>(&param("dense_4h_to_h.weight"), None)?,
                b_out: read_vector(&param("dense_4h_to_h.bias"))?,
            })
        } else if param("gate_proj.weight").exists() {
            Ok(Self::Gated {
                gate: read_rows::<f64>(&param("gate_proj.weight"), None)?,
                up: read_rows::<f64>(&param("up_proj.weight"), None)?,
                down: read_rows::<f64>(&param("down_proj.weight"), None)?,
            })
        } else {
            Err(format!(
                "{} holds neither a GPT-NeoX nor a Qwen3 MLP",
                harvest.display()
            ))
        }
    }

    /// `F(h)` for rows `h`, and the first-order rounding bound of this evaluation.
    fn forward(&self, h: &Array2<f64>, activation: Activation) -> (Array2<f64>, Array2<f64>) {
        let eps = f64::EPSILON;
        let reader_terms = (h.ncols() + 1) as f64 * eps;
        let h_abs = h.mapv(f64::abs);
        let reader_sum = |w: &Array2<f64>| {
            let value = fast_atb(&h.t(), &w.t());
            let rounding = fast_atb(&h_abs.t(), &w.mapv(f64::abs).t()) * reader_terms;
            (value, rounding)
        };
        let (units, unit_rounding, w_out, b_out) = match self {
            Self::Dense {
                w_in,
                b_in,
                w_out,
                b_out,
            } => {
                let (pre, pre_rounding) = reader_sum(w_in);
                let pre = pre + b_in;
                let pre_rounding = pre_rounding + &(b_in.mapv(f64::abs) * reader_terms);
                let units = pre.mapv(|t| activation.value(t));
                let rounding = &pre.mapv(|t| activation.slope(t).abs()) * &pre_rounding
                    + &(units.mapv(f64::abs) * eps);
                (units, rounding, w_out, Some(b_out))
            }
            Self::Gated { gate, up, down } => {
                let (pre, pre_rounding) = reader_sum(gate);
                let (linear, linear_rounding) = reader_sum(up);
                let gates = pre.mapv(|t| activation.value(t));
                let units = &gates * &linear;
                let rounding = &(&pre.mapv(|t| activation.slope(t).abs()) * &linear.mapv(f64::abs))
                    * &pre_rounding
                    + &(&gates.mapv(f64::abs) * &linear_rounding)
                    + &(units.mapv(f64::abs) * eps);
                (units, rounding, down, None)
            }
        };
        let unit_terms = (units.ncols() + 1) as f64 * eps;
        let w_out_abs = w_out.mapv(f64::abs);
        let value = fast_atb(&units.t(), &w_out.t());
        let rounding = fast_atb(&units.mapv(f64::abs).t(), &w_out_abs.t()) * unit_terms
            + &fast_atb(&unit_rounding.t(), &w_out_abs.t());
        match b_out {
            Some(bias) => (value + bias, rounding + &(bias.mapv(f64::abs) * unit_terms)),
            None => (value, rounding),
        }
    }
}

fn forward_check(
    harvest: &Path,
    draw_dir: &Path,
    executed: &Path,
    rows: usize,
    out: &Path,
) -> Result<(), String> {
    let meta = read_json(&harvest.join("meta.json"))?;
    let activation = Activation::parse(
        meta.get("hidden_act")
            .and_then(Value::as_str)
            .ok_or("meta.json names no hidden_act")?,
    )?;
    let block = Block::load(harvest)?;
    let h = read_rows::<f64>(&draw_dir.join("h.npy"), Some(rows))?;
    let torch = read_rows::<f64>(executed, Some(rows))?;
    if h.dim() != torch.dim() {
        return Err(format!(
            "h rows {:?} and executed rows {:?} disagree",
            h.dim(),
            torch.dim()
        ));
    }
    let (value, rounding) = block.forward(&h, activation);
    let mut max_gap = 0.0_f64;
    let mut max_ratio = 0.0_f64;
    let mut beyond_twice_bound = 0_usize;
    let mut gap_sq = 0.0_f64;
    let mut torch_sq = 0.0_f64;
    for ((rust, executed_value), bound) in value.iter().zip(torch.iter()).zip(rounding.iter()) {
        let gap = (rust - executed_value).abs();
        max_gap = max_gap.max(gap);
        max_ratio = max_ratio.max(gap / bound);
        if gap > 2.0 * bound {
            beyond_twice_bound += 1;
        }
        gap_sq += gap * gap;
        torch_sq += executed_value * executed_value;
    }
    let report = json!({
        "stage": "forward",
        "rows": h.nrows(),
        "outputs": h.ncols(),
        "max_abs_gap": max_gap,
        "max_gap_over_bound": max_ratio,
        "entries_beyond_twice_bound": beyond_twice_bound,
        "relative_frobenius_gap": (gap_sq / torch_sq).sqrt(),
        "bound": "first-order rounding of one evaluation: (n+1) eps sum|terms| per reader and output sum, propagated through |act'|; activation implementation accuracy not bounded",
    });
    write_json(out, &report)?;
    println!(
        "[forward] rows={} max|gap|={max_gap:.3e} max gap/bound={max_ratio:.3e} beyond 2x bound={beyond_twice_bound} rel frob gap={:.3e}",
        h.nrows(),
        (gap_sq / torch_sq).sqrt()
    );
    Ok(())
}

fn monte_carlo(draw_dir: &Path, executed: &Path, out: &Path) -> Result<(), String> {
    let layout = read_json(&draw_dir.join("draw.json"))?;
    let rows = layout
        .get("rows")
        .and_then(Value::as_u64)
        .ok_or("draw.json names no rows")? as usize;
    let blocks = layout
        .get("blocks")
        .and_then(Value::as_array)
        .ok_or("draw.json names no blocks")?;
    let y = read_rows::<f64>(executed, None)?;
    if y.nrows() != rows * blocks.len() {
        return Err(format!(
            "executed rows {} != rows {rows} x blocks {}",
            y.nrows(),
            blocks.len()
        ));
    }
    let base = y.slice(s![0..rows, ..]);
    let independent = squared_distances(base, y.slice(s![rows..2 * rows, ..]));
    let (variance, variance_se) = half_mean_with_se(&independent);
    let mean_output = base.mean_axis(Axis(0)).ok_or("no executed rows")?;
    let mean_norm = mean_output.dot(&mean_output).sqrt();

    let mut frames = Vec::new();
    for (index, block) in blocks.iter().enumerate().skip(2) {
        let coupled = squared_distances(base, y.slice(s![index * rows..(index + 1) * rows, ..]));
        let (error, error_se) = half_mean_with_se(&coupled);
        let paired: Vec<f64> = independent
            .iter()
            .zip(&coupled)
            .map(|(a, b)| a - b)
            .collect();
        let (explained, explained_se) = half_mean_with_se(&paired);
        let name = block.get("name").cloned().unwrap_or(Value::Null);
        println!(
            "[mc] frame={name} E(P)={error:.6e} +- {error_se:.3e} (E/se={:.2}) V(P)={explained:.6e} +- {explained_se:.3e} E/V(I)={:.4}",
            error / error_se,
            error / variance
        );
        frames.push(json!({
            "name": name,
            "frame": block.get("frame").cloned().unwrap_or(Value::Null),
            "E": { "estimate": error, "se": error_se },
            "V": { "estimate": explained, "se": explained_se },
            "E_over_V_I": error / variance,
        }));
    }
    let executed_meta = read_json(Path::new(&format!("{}.json", executed.display())))?;
    write_json(
        out,
        &json!({
            "stage": "mc",
            "metric": "M = I",
            "estimators": "V(I) = mean|F(Z)-F(Z~)|^2/2, E(P) = mean|F(Z)-F(Z')|^2/2, V(P) paired, se = sd/(2 sqrt n)",
            "rows": rows,
            "draw": layout,
            "executed": executed_meta,
            "mean_output_norm": mean_norm,
            "V_I": { "estimate": variance, "se": variance_se },
            "frames": frames,
        }),
    )?;
    println!("[mc] rows={rows} V(I)={variance:.6e} +- {variance_se:.3e} |m|={mean_norm:.6e}");
    Ok(())
}

fn analytic(
    harvest: &Path,
    law: &Path,
    draw_dir: &Path,
    executed: &Path,
    rows: usize,
    out: &Path,
) -> Result<(), String> {
    let started = Instant::now();
    let meta = read_json(&harvest.join("meta.json"))?;
    let hidden_act = meta
        .get("hidden_act")
        .and_then(Value::as_str)
        .ok_or("meta.json names no hidden_act")?;
    let mlp_class = meta.get("mlp_class").and_then(Value::as_str);
    if mlp_class != Some("Qwen3MLP") {
        return Err(format!(
            "the analytic receipt here is A6, the gated Qwen3 MLP; meta.json names {hidden_act:?} {mlp_class:?}"
        ));
    }
    let parameters = torch_parameters(harvest, &meta)?;
    let output_dim = parameters
        .get("down_proj.weight")
        .map(|weight| weight.shape()[0])
        .ok_or("the harvest holds no down_proj.weight")?;
    let h0 = read_vector(&law.join("h0.npy"))?;
    let factor = read_rows::<f64>(&law.join("L.npy"), None)?;
    let rank = factor.ncols();
    // The torch layout and the absorption of the declared law `h = h0 + L z` (gate and up readers `W L`, `A L`,
    // biases `W h0`, `A h0`) have one owner in gam-sae.
    let block = UnabsorbedGatedBlock::from_torch_parameters(parameters, hidden_act, Array2::<f64>::eye(output_dim))
        .map_err(|err| format!("torch block: {err}"))?
        .absorb(h0.view(), factor.view())
        .map_err(|err| format!("absorbed block: {err}"))?;
    let absorb_seconds = started.elapsed().as_secs_f64();

    let layout = read_json(&draw_dir.join("draw.json"))?;
    let draw_rows = layout
        .get("rows")
        .and_then(Value::as_u64)
        .ok_or("draw.json names no rows")? as usize;
    let blocks = layout
        .get("blocks")
        .and_then(Value::as_array)
        .ok_or("draw.json names no blocks")?;
    let monte_carlo = read_json(&draw_dir.join("receipt.json"))?;
    let mc_frames = monte_carlo
        .get("frames")
        .and_then(Value::as_array)
        .ok_or("receipt.json names no frames")?;
    if rows < 2 || rows > draw_rows {
        return Err(format!("--rows {rows} outside 2..={draw_rows}"));
    }
    let z = read_rows::<f64>(&draw_dir.join("z.npy"), Some(rows))?;
    let y = read_rows::<f64>(executed, None)?;
    if z.ncols() != rank || y.nrows() != draw_rows * blocks.len() {
        return Err(format!(
            "draws {:?} and executed rows {:?} disagree with rank {rank} x blocks {}",
            z.dim(),
            y.dim(),
            blocks.len()
        ));
    }
    let base = y.slice(s![0..rows, ..]);
    let full_frame = Array2::<f64>::eye(rank);

    let mut frames = Vec::new();
    for (index, entry) in blocks.iter().enumerate().skip(2) {
        let frame_started = Instant::now();
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .ok_or("a draw block names no frame")?;
        if entry["frame"]["kind"] != "principal" {
            return Err(format!("frame {name} is not a principal frame"));
        }
        let retained = entry["frame"]
            .get("retained")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("frame {name} names no retained count"))? as usize;
        let mc = mc_frames
            .iter()
            .find(|frame| frame.get("name").and_then(Value::as_str) == Some(name))
            .ok_or_else(|| format!("receipt.json has no frame {name}"))?;
        let (mc_discarded, mc_discarded_se) = estimate(&mc["E"])?;

        let mut frame = Array2::<f64>::zeros((rank, retained));
        for column in 0..retained {
            frame[[column, column]] = 1.0;
        }
        let response = block
            .retained_response(frame.view(), z.view())
            .map_err(|err| format!("F-bar({name}): {err}"))?;
        // The positive control `g0 = F(Pz)` ignores the discarded input: the full frame's response at the retained
        // point, whose discarded variances are zero.
        let mut retained_points = z.clone();
        retained_points.slice_mut(s![.., retained..]).fill(0.0);
        let unsmoothed = block
            .retained_response(full_frame.view(), retained_points.view())
            .map_err(|err| format!("F(Pz) ({name}): {err}"))?;
        let coupled = y.slice(s![index * draw_rows..index * draw_rows + rows, ..]);
        let half_coupled: Vec<f64> = squared_distances(base, coupled)
            .into_iter()
            .map(|value| 0.5 * value)
            .collect();
        let (gap, gap_se) = paired_gap(base, response.values.view(), &half_coupled);
        let (control, control_se) = paired_gap(base, unsmoothed.values.view(), &half_coupled);
        // `E D = E|F̄_P − g|²`, and the computed `g` departs from `F̄_P` by at most the quadrature band per entry, so
        // a correct conditional mean leaves `E D` at most the mean squared band.
        let quadrature_bias = response
            .quadrature_band
            .outer_iter()
            .map(|row| row.dot(&row))
            .sum::<f64>()
            / rows as f64;
        let largest_band = response.quadrature_band.iter().fold(0.0_f64, |largest, &band| largest.max(band));
        let frame_seconds = frame_started.elapsed().as_secs_f64();
        println!(
            "[analytic] frame={name} retained={retained} E|F-bar - g|^2: Stein gated mean {gap:.3e} +- {gap_se:.3e} (z={:.2}, quadrature bias <= {quadrature_bias:.3e}), unsmoothed F(Pz) {control:.3e} +- {control_se:.3e} (z={:.2}); executed E(P)={mc_discarded:.6e} +- {mc_discarded_se:.3e}; {frame_seconds:.1}s",
            gap / gap_se,
            control / control_se,
        );
        frames.push(json!({
            "name": name,
            "retained": retained,
            "stein_gated_mean_gap": { "mean": gap, "se": gap_se, "z": gap / gap_se },
            "quadrature_bias_bound": quadrature_bias,
            "largest_quadrature_band": largest_band,
            "unsmoothed_control_gap": { "mean": control, "se": control_se, "z": control / control_se },
            "executed_E": { "estimate": mc_discarded, "se": mc_discarded_se },
            "seconds": frame_seconds,
        }));
    }
    write_json(
        out,
        &json!({
            "stage": "analytic",
            "acceptance": "#2946 A6: the Stein gated mean matches Monte Carlo on the executed Qwen3 MLP under the declared law",
            "harvest": harvest.display().to_string(),
            "law": law.display().to_string(),
            "draw": draw_dir.display().to_string(),
            "activation": "silu gate (SwiGLU)",
            "absorption": "W = W_gate L, b = W_gate h0, A = W_up L, c = W_up h0, U = W_down, M = I",
            "gap_statistic": "D = |F(Z) - g(PZ)|^2 - |F(Z) - F(Z')|^2 / 2, E D = E|F-bar_P - g|^2 (R2); mean +- sd/sqrt n",
            "rows_used": rows,
            "V_I_executed": monte_carlo["V_I"].clone(),
            "absorb_seconds": absorb_seconds,
            "frames": frames,
        }),
    )
}

/// Every MLP parameter `meta.json` lists, read from `<torch name>.npy` into the map the torch-layout parser takes.
fn torch_parameters(harvest: &Path, meta: &Value) -> Result<BTreeMap<String, ArrayD<f64>>, String> {
    let listed = meta
        .get("mlp_params")
        .and_then(Value::as_object)
        .ok_or("meta.json lists no mlp_params")?;
    let mut parameters = BTreeMap::new();
    for name in listed.keys() {
        let path = harvest.join(format!("{name}.npy"));
        let npy = open_npy(&path)?;
        let shape = npy
            .shape()
            .iter()
            .map(|&extent| usize::try_from(extent).map_err(|err| format!("{}: {err}", path.display())))
            .collect::<Result<Vec<usize>, String>>()?;
        let values = npy
            .try_data::<f64>()
            .map_err(|npy| format!("{} has dtype {}", path.display(), npy.dtype().descr()))?
            .collect::<std::io::Result<Vec<f64>>>()
            .map_err(|err| format!("read {}: {err}", path.display()))?;
        let array = ArrayD::from_shape_vec(IxDyn(&shape), values)
            .map_err(|err| format!("{} has an invalid shape: {err}", path.display()))?;
        parameters.insert(name.clone(), array);
    }
    Ok(parameters)
}

/// The `estimate` and `se` of one `mc` receipt entry.
fn estimate(entry: &Value) -> Result<(f64, f64), String> {
    match (
        entry.get("estimate").and_then(Value::as_f64),
        entry.get("se").and_then(Value::as_f64),
    ) {
        (Some(value), Some(se)) => Ok((value, se)),
        _ => Err(format!("receipt entry {entry} carries no estimate and se")),
    }
}

/// Mean and standard error of `D_i = |F(Z_i) - g_i|^2 - |F(Z_i) - F(Z'_i)|^2 / 2`.
fn paired_gap(
    executed: ArrayView2<'_, f64>,
    response: ArrayView2<'_, f64>,
    half_coupled: &[f64],
) -> (f64, f64) {
    let samples: Vec<f64> = squared_distances(executed, response)
        .iter()
        .zip(half_coupled)
        .map(|(gap, half)| gap - half)
        .collect();
    mean_with_se(&samples)
}

/// The sample mean and its standard error `sd / sqrt n`.
fn mean_with_se(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

fn squared_distances(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Vec<f64> {
    a.outer_iter()
        .zip(b.outer_iter())
        .map(|(x, y)| {
            x.iter()
                .zip(y.iter())
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
        })
        .collect()
}

/// Half the sample mean and its standard error `sd / (2 sqrt n)`.
fn half_mean_with_se(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (0.5 * mean, 0.5 * (variance / n).sqrt())
}

fn open_npy(path: &Path) -> Result<NpyFile<BufReader<File>>, String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let npy = NpyFile::new(BufReader::new(file))
        .map_err(|err| format!("read .npy header {}: {err}", path.display()))?;
    if let Order::Fortran = npy.order() {
        return Err(format!("{} is Fortran-ordered", path.display()));
    }
    Ok(npy)
}

/// The leading `max_rows` rows (all rows when `None`) of a 2-D .npy, as float64.
fn read_rows<T: npyz::Deserialize + Into<f64>>(
    path: &Path,
    max_rows: Option<usize>,
) -> Result<Array2<f64>, String> {
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
    Array2::from_shape_vec((take, cols), values)
        .map_err(|err| format!("{} has an invalid shape: {err}", path.display()))
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

fn write_npy(
    path: &Path,
    shape: &[u64],
    values: impl IntoIterator<Item = f64>,
) -> Result<(), String> {
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

fn read_json(path: &Path) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|err| format!("encode {}: {err}", path.display()))?;
    std::fs::write(path, text).map_err(|err| format!("write {}: {err}", path.display()))
}
