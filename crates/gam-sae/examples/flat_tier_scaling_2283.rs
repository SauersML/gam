//! #2283 — flat-tier scaling points on the acceptance cell's own training rows.
//!
//! The theorem-faithful hybrid's flat checkpoint (`run_faithful_flat_checkpoint.sbatch`)
//! fits `sparse_dictionary_fit(K=32672, active=28, minibatch=8192, score_mode="required",
//! max_epochs=200)` on all 96,000 training rows of creditscope L30 `residual_post`. One
//! attempt at that shape runs for hours, so this example runs the same one-shot trainer
//! (`fit_sparse_dictionary`: the seeded inner alternation and the shared-rho REML schedule)
//! on the first `n` rows of that split. It reports the trainer's per-epoch heartbeat and
//! one summary line.
//!
//! The rows come from `experiments/1026_close/prep_flat_subsets.py`, which calls the
//! driver's own `load_chunk_dir` / `make_split` (seed 0, 120,000 rows, 20% test) and writes
//! the first `n` training rows as `<f4` `.npy`.
//!
//! ```text
//! cargo run --release -p gam-sae --example flat_tier_scaling_2283 -- \
//!   --npy=creditscope_l30_train_first4096.f32.npy --score-mode=required
//! ```
//!
//! | flag | default | meaning |
//! |---|---|---|
//! | `--npy` | required | `<f4` or `<f2` `.npy` of shape `n × p` |
//! | `--atoms` | 32672 | dictionary width `K` (the cell's `k_flat`) |
//! | `--active` | 28 | routing sparsity `s` (the cell's `k_lin`) |
//! | `--minibatch` | 8192 | route minibatch (the cell's `--sparse-minibatch`) |
//! | `--epochs` | 200 | inner epoch cap (the cell's `--max-epochs`) |
//! | `--score-mode` | required | `off` / `auto` / `required` |
//! | `--log` | warn | `warn` / `info` / `debug` |
//! | `--seed-out` | none | seed only: write the decoder seed as `<f4` `.npy` and exit |
//! | `--seed-in` | none | fit from a seed written by `--seed-out` for the same rows and `--atoms` |
//!
//! Seeding is `O(K·N·P)` host work, over 41 minutes at the acceptance shape, so a device
//! run seeds on a CPU allocation with `--seed-out` and fits on the GPU with `--seed-in`.
//!
//! One machine-readable `[2283-flat-scaling]` line is printed on stdout, for a fit or a
//! typed refusal alike, and one `[2283-flat-seed]` line for a seed-only run.

use std::path::Path;
use std::time::Instant;

use gam_sae::sparse_dict::{
    SparseDictConfig, fit_sparse_dictionary, fit_sparse_dictionary_from_seed,
    seed_sparse_dictionary_decoder,
};
use ndarray::Array2;

#[path = "support/f16.rs"]
mod f16;
#[path = "support/npy_header.rs"]
mod npy_header;

/// Value of `--<flag>=<value>` from the command line, or `None`. The last occurrence
/// wins, as in `curved_tier_scaling_2283`.
fn arg_value(flag: &str) -> Option<String> {
    let prefix = format!("--{flag}=");
    std::env::args()
        .filter_map(|arg| arg.strip_prefix(&prefix).map(str::to_string))
        .next_back()
}

fn arg_usize(flag: &str, fallback: usize) -> Result<usize, String> {
    match arg_value(flag) {
        Some(raw) => raw
            .trim()
            .parse::<usize>()
            .map_err(|error| format!("--{flag} must parse as usize: {error}")),
        None => Ok(fallback),
    }
}

/// Read a whole little-endian `<f4` or `<f2` `.npy` matrix.
fn read_npy(path: &Path) -> Result<Array2<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let (n, p, elem, is_f4, data_off) = npy_header::parse_npy_header(&bytes, path)?;
    let end = data_off + n * p * elem;
    if bytes.len() < end {
        return Err(format!(
            "{}: {} bytes, but the header describes {end}",
            path.display(),
            bytes.len()
        ));
    }
    let body = &bytes[data_off..end];
    let values: Vec<f32> = if is_f4 {
        body.chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    } else {
        body.chunks_exact(2)
            .map(|word| f16::f16_to_f32(u16::from_le_bytes([word[0], word[1]])))
            .collect()
    };
    Array2::from_shape_vec((n, p), values).map_err(|error| format!("{}: {error}", path.display()))
}

/// Write a little-endian `<f4` `.npy` matrix (format version 1.0).
fn write_npy(path: &Path, matrix: &Array2<f32>) -> Result<(), String> {
    let (rows, cols) = matrix.dim();
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // The magic, version and length take 10 bytes; the newline-terminated header pads
    // the data offset to a multiple of 64, as the format specifies.
    let data_offset = (10 + header.len() + 1).div_ceil(64) * 64;
    header.push_str(&" ".repeat(data_offset - 10 - header.len() - 1));
    header.push('\n');
    let header_len = u16::try_from(header.len())
        .map_err(|error| format!("{}: npy header length: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(data_offset + rows * cols * 4);
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in matrix.iter() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let npy = arg_value("npy").ok_or("--npy=<path> is required")?;
    let atoms = arg_usize("atoms", 32_672)?;
    let active = arg_usize("active", 28)?;
    let minibatch = arg_usize("minibatch", 8_192)?;
    let epochs = arg_usize("epochs", 200)?;
    let mode = arg_value("score-mode").unwrap_or_else(|| "required".to_string());
    let score_mode = gam_gpu::GpuPolicy::parse(&mode)
        .ok_or_else(|| format!("--score-mode must be off|auto|required, got {mode}"))?;
    let level = match arg_value("log").as_deref() {
        Some("trace") => log::LevelFilter::Trace,
        Some("debug") => log::LevelFilter::Debug,
        _ => log::LevelFilter::Warn,
    };
    gam_solve::progress_log::init_logging_at(level);
    gam_runtime::process_monitor::start();

    let load_started = Instant::now();
    let x = read_npy(Path::new(&npy))?;
    let (n, p) = x.dim();
    let load_secs = load_started.elapsed().as_secs_f64();

    let mut config = SparseDictConfig::new(atoms);
    config.active = active;
    config.minibatch = minibatch;
    config.max_epochs = epochs;
    config.score_mode = score_mode;

    if let Some(out) = arg_value("seed-out") {
        let seed_started = Instant::now();
        let seed =
            seed_sparse_dictionary_decoder(x.view(), &config).map_err(|error| error.to_string())?;
        let seed_secs = seed_started.elapsed().as_secs_f64();
        write_npy(Path::new(&out), &seed)?;
        println!(
            "[2283-flat-seed] n={n} p={p} atoms={atoms} load_s={load_secs:.1} \
             seed_s={seed_secs:.1} out={out}"
        );
        return Ok(());
    }
    let seed_in = arg_value("seed-in");
    let seed_load_started = Instant::now();
    let seed = seed_in.as_deref().map(|path| read_npy(Path::new(path))).transpose()?;
    let seed_load_secs = seed_load_started.elapsed().as_secs_f64();

    let fit_started = Instant::now();
    let outcome = match seed {
        Some(seed) => fit_sparse_dictionary_from_seed(x.view(), &config, seed),
        None => fit_sparse_dictionary(x.view(), &config),
    };
    let fit_secs = fit_started.elapsed().as_secs_f64();
    let shape = format!(
        "n={n} p={p} atoms={atoms} active={active} minibatch={minibatch} epochs_cap={epochs} \
         score_mode={mode} seed_in={} load_s={load_secs:.1} seed_load_s={seed_load_secs:.1} \
         fit_s={fit_secs:.1}",
        seed_in.as_deref().unwrap_or("none")
    );
    match outcome {
        Ok(fit) => println!(
            "[2283-flat-scaling] {shape} outcome=certified ev={:.6} inner_epochs={} \
             outer_iterations={} selected_rho={:.6e} routing_residual={:.3e} minibatches={} \
             device_minibatches={} cpu_minibatches={}",
            fit.explained_variance,
            fit.epochs,
            fit.convergence.outer_iterations,
            fit.convergence.selected_rho,
            fit.convergence.routing_residual,
            fit.score_route_stats.minibatches,
            fit.score_route_stats.device_minibatches,
            fit.score_route_stats.cpu_minibatches,
        ),
        Err(error) => println!("[2283-flat-scaling] {shape} outcome=refused error={error}"),
    }
    Ok(())
}
