//! The VPD paper's 4-layer Pile target executed in Rust from its checkpoint (#2951).
//!
//! `t-9d2b8f02` (goodfire/spd) is a LlamaSimpleMLP: 4 layers, `d = 768`, 6 heads of 128, a
//! tanh-GELU MLP of 3072, rotate-half RoPE, RMSNorm, the embedding tied to the unembedding and a
//! vocabulary of 50,277. This example reads `model_step_99999.safetensors` and
//! `model_config.yaml` into [`LlamaSimpleMlp`], executes it on Pile rows, and writes the
//! next-token loss of each row with its certified radius. For the first `--dump` rows it also
//! writes the logits as float64 `.npy`, which `bench/mpd_vpd_target_forward_2951.py` compares
//! against the torch forward.
//!
//! The rows are an `int32`/`int64` `.npy` of shape `rows × (seq + 1)`, e.g. the first rows of
//! `danbraunai/pile-uncopyrighted-tok-shuffled`'s val split: the model reads `row[..seq]` and is
//! scored against `row[1..=seq]`, as the source's training loop scores it.
//!
//! ```text
//! cargo run --release -p gam-mpd --example mpd_vpd_target_forward_2951 -- \
//!     --run RUN_DIR --tokens ROWS.npy --rows 8 --seq 512 --dump 2 --out OUT_DIR
//! ```

use gam_runtime::resource::MemoryGovernor;
use gam_mpd::attention::ProjectedRows;
use gam_mpd::llama_simple_mlp::{LlamaSimpleMlp, LlamaSimpleMlpConfig};
use serde_json::json;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("[mpd_vpd_target_forward_2951] error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|pair| pair[0] == name).map(|pair| pair[1].clone())
}

fn required(args: &[String], name: &str) -> Result<String, String> {
    flag(args, name).ok_or_else(|| format!("missing {name}"))
}

fn count(args: &[String], name: &str, default: usize) -> Result<usize, String> {
    flag(args, name).map_or(Ok(default), |value| value.parse().map_err(|error| format!("{name}: {error}")))
}

/// The source's `model_config.yaml`: one flat `key: value` per line.
fn read_config(path: &Path) -> Result<LlamaSimpleMlpConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let fields: BTreeMap<&str, &str> = text
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim(), value.trim()))
        .collect();
    let field = |key: &str| fields.get(key).copied().ok_or_else(|| format!("{}: no {key}", path.display()));
    let integer = |key: &str| field(key)?.parse::<usize>().map_err(|error| format!("{key}: {error}"));
    let real = |key: &str| field(key)?.parse::<f64>().map_err(|error| format!("{key}: {error}"));
    if field("model_type")? != "LlamaSimpleMLP" {
        return Err(format!("model_type {} is not LlamaSimpleMLP", field("model_type")?));
    }
    for (key, expected) in [("attn_bias", "false"), ("mlp_bias", "false"), ("rotary_adjacent_pairs", "false")] {
        if field(key)? != expected {
            return Err(format!("{key} = {}, the forward executes {expected}", field(key)?));
        }
    }
    let config = LlamaSimpleMlpConfig {
        vocab: integer("vocab_size")?,
        model_dim: integer("n_embd")?,
        layers: integer("n_layer")?,
        heads: integer("n_head")?,
        hidden: integer("n_intermediate")?,
        rotary_base: real("rotary_base")?,
        epsilon: real("rms_norm_eps")?,
    };
    // The source rotates the whole head and gives every query head its own key/value head.
    if integer("rotary_dim")? != config.head_dim() || integer("n_key_value_heads")? != config.heads {
        return Err("rotary_dim must equal the head width and n_key_value_heads the head count".into());
    }
    Ok(config)
}

/// A little-endian, C-order `<i4` or `<i8` `.npy` of two axes, as `u32` token ids.
fn read_token_rows(path: &Path) -> Result<(usize, usize, Vec<u32>), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not a .npy file", path.display()));
    }
    let (length, start) = if bytes[6] >= 2 {
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    } else {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    };
    let header = std::str::from_utf8(&bytes[start..start + length]).map_err(|error| error.to_string())?;
    let width = if header.contains("'<i4'") {
        4
    } else if header.contains("'<i8'") {
        8
    } else {
        return Err(format!("{}: tokens must be <i4 or <i8: {header}", path.display()));
    };
    if header.contains("'fortran_order': True") {
        return Err(format!("{}: Fortran order", path.display()));
    }
    let shape: Vec<usize> = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or_else(|| format!("{}: no shape", path.display()))?
        .split(',')
        .filter(|axis| !axis.trim().is_empty())
        .map(|axis| axis.trim().parse().map_err(|error| format!("shape: {error}")))
        .collect::<Result<_, _>>()?;
    let [rows, cols] = shape[..] else { return Err(format!("{}: shape {shape:?} is not two axes", path.display())) };
    let data = &bytes[start + length..];
    if data.len() != rows * cols * width {
        return Err(format!("{}: {} data bytes for {rows}×{cols}", path.display(), data.len()));
    }
    let tokens = data
        .chunks_exact(width)
        .map(|raw| {
            let value = if width == 4 {
                i64::from(i32::from_le_bytes(raw.try_into().unwrap()))
            } else {
                i64::from_le_bytes(raw.try_into().unwrap())
            };
            u32::try_from(value).map_err(|_| format!("token {value} is not a vocabulary index"))
        })
        .collect::<Result<_, _>>()?;
    Ok((rows, cols, tokens))
}

/// A float64 `.npy` of two axes.
fn write_npy(path: &Path, rows: usize, cols: usize, values: impl Iterator<Item = f64>) -> Result<(), String> {
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let file = std::fs::File::create(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut out = std::io::BufWriter::new(file);
    let io = |error: std::io::Error| format!("{}: {error}", path.display());
    out.write_all(b"\x93NUMPY\x01\x00").map_err(io)?;
    out.write_all(&(header.len() as u16).to_le_bytes()).map_err(io)?;
    out.write_all(header.as_bytes()).map_err(io)?;
    for value in values {
        out.write_all(&value.to_le_bytes()).map_err(io)?;
    }
    out.flush().map_err(io)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let run_dir = PathBuf::from(required(&args, "--run")?);
    let token_path = PathBuf::from(required(&args, "--tokens")?);
    let out = PathBuf::from(required(&args, "--out")?);
    let (rows, seq, dump, tile) =
        (count(&args, "--rows", 8)?, count(&args, "--seq", 512)?, count(&args, "--dump", 0)?, count(&args, "--tile", 64)?);
    std::fs::create_dir_all(&out).map_err(|error| format!("{}: {error}", out.display()))?;
    let governor = MemoryGovernor::global();
    let config = read_config(&run_dir.join("model_config.yaml"))?;
    let started = Instant::now();
    let model = LlamaSimpleMlp::from_safetensors(governor, &run_dir.join("model_step_99999.safetensors"), config)
        .map_err(|error| error.to_string())?;
    eprintln!("loaded {config:?} in {:.1}s", started.elapsed().as_secs_f64());
    let (available, width, tokens) = read_token_rows(&token_path)?;
    if width < seq + 1 || available < rows {
        return Err(format!("{available}×{width} rows cannot give {rows} rows of {seq} inputs and targets"));
    }
    let mut report_rows = Vec::new();
    let (mut total, mut total_radius, mut scored) = (0.0, 0.0, 0usize);
    for r in 0..rows {
        let started = Instant::now();
        let row = &tokens[r * width..r * width + seq + 1];
        let (inputs, targets) = (&row[..seq], &row[1..]);
        if r < dump {
            let rows_out = model.final_rows(governor, inputs).map_err(|error| error.to_string())?;
            let logits = model
                .logits(governor, ProjectedRows { values: rows_out.values.view(), radius: rows_out.radius.view() })
                .map_err(|error| error.to_string())?;
            write_npy(&out.join(format!("logits_row{r}.npy")), seq, config.vocab, logits.values.iter().copied())?;
            write_npy(&out.join(format!("logit_radius_row{r}.npy")), seq, config.vocab, logits.radius.iter().copied())?;
        }
        let losses = model.next_token_loss(governor, inputs, targets, tile).map_err(|error| error.to_string())?;
        let loss: f64 = losses.iter().map(|l| l.loss).sum();
        let radius: f64 = losses.iter().map(|l| l.radius).sum();
        let worst = losses.iter().map(|l| l.radius).fold(0.0, f64::max);
        let correct = losses.iter().zip(targets).filter(|(l, t)| l.argmax == **t as usize).count();
        total += loss;
        total_radius += radius;
        scored += losses.len();
        eprintln!(
            "row {r}: CE {:.6} (radius of the mean {:.3e}, largest token radius {worst:.3e}), top-1 {:.4} [{:.1}s]",
            loss / losses.len() as f64,
            radius / losses.len() as f64,
            correct as f64 / losses.len() as f64,
            started.elapsed().as_secs_f64()
        );
        report_rows.push(json!({
            "row": r,
            "mean_ce": loss / losses.len() as f64,
            "mean_ce_radius": radius / losses.len() as f64,
            "largest_token_ce_radius": worst,
            "top1": correct as f64 / losses.len() as f64,
            "token_ce": losses.iter().map(|l| l.loss).collect::<Vec<_>>(),
            "seconds": started.elapsed().as_secs_f64(),
        }));
    }
    let report = json!({
        "run": run_dir,
        "tokens": token_path,
        "seq": seq,
        "rows": rows,
        "mean_ce": total / scored as f64,
        "mean_ce_radius": total_radius / scored as f64,
        "per_row": report_rows,
    });
    let path = out.join("report.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("{}", json!({"mean_ce": total / scored as f64, "mean_ce_radius": total_radius / scored as f64, "tokens": scored}));
    Ok(())
}
