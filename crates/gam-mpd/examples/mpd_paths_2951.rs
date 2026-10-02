//! Exact paths of an imported model's output against its own trace and against torch (#2951).
//!
//! `mpd_paths_2951 EXPORT_DIR EXPANSIONS [TORCH_DIR]`
//!
//! `EXPORT_DIR` is an engine export (`bench/mpd_engine_export_2951.py`; a rotary language model is
//! read at its first sequence of 16 tokens). The program is executed once and its output decomposed
//! by [`gam_mpd::paths::decompose`] with `EXPANSIONS` best-first expansions. Checked:
//! - the paths and remainders sum to the traced output within the decomposition's rounding band;
//! - with `TORCH_DIR` (`bench/mpd_paths_torch_2951.py`: `logits.f64`, the model's float64 logits on
//!   every `(a, b)` of a modular-addition run, row `a·p + b`), the paths also sum to torch's logits
//!   within twice the band.
//!
//! Printed: `max |path sum − value|` against both, the band, the path and remainder counts, the
//! certified remainder mass, and how many items carry 99% of the mass; then the heaviest paths.
//! The exit status is non-zero when a check fails.

use gam_mpd::import::{import, import_language_model, is_language_model};
use gam_mpd::operator_program::{Node, OperatorProgram, SlotValues};
use gam_mpd::paths::{PathOptions, decompose};
use gam_runtime::resource::MemoryGovernor;
use ndarray::Array2;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("mpd_paths_2951: {error}");
            ExitCode::FAILURE
        }
    }
}

fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn frobenius(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn label(program: &OperatorProgram, node: usize) -> String {
    let kind = match &program.nodes[node] {
        Node::Feature { slot, .. } => return format!("{node}:feature[{slot}]"),
        Node::Affine { terms, .. } => {
            let names: Vec<&str> = terms.iter().map(|(_, op)| program.operators[*op].name.as_str()).collect();
            return format!("{node}:affine[{}]", names.join("+"));
        }
        Node::Raw { .. } => "raw",
        Node::Constant { .. } => "constant",
        Node::Bilinear { .. } => "bilinear",
        Node::Softmax { .. } => "softmax",
        Node::Mix { .. } => "mix",
        Node::Pointwise { .. } => "pointwise",
        Node::Hadamard { .. } => "hadamard",
        Node::Readout { .. } => "readout",
        Node::Outer { .. } => "outer",
        Node::Concat { .. } => "concat",
        Node::Param { .. } => "param",
        Node::Call { .. } => "call",
        Node::Gain { .. } => "gain",
        Node::Attend { .. } => "attend",
        Node::RmsNorm { .. } => "rms_norm",
        Node::Transposed { .. } => "transposed",
    };
    format!("{node}:{kind}")
}

/// Raw little-endian float64 rows.
fn read_f64(path: &std::path::Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % (8 * cols) != 0 {
        return Err(format!("{}: {} bytes is not rows of {cols} float64", path.display(), bytes.len()));
    }
    let data: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((data.len() / cols, cols), data).map_err(|e| e.to_string())
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_paths_2951 EXPORT_DIR EXPANSIONS [TORCH_DIR]";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let expansions: usize = args.get(2).ok_or(usage)?.parse().map_err(|e| format!("EXPANSIONS: {e}"))?;
    let imported = if is_language_model(&dir)? { import_language_model(&dir, 1, 16)? } else { import(&dir)? };
    let program = &imported.program;
    let inputs = &imported.contract.family;
    let started = std::time::Instant::now();
    let trace = program.execute(inputs, false).map_err(|e| e.to_string())?;
    let result = decompose(MemoryGovernor::global(), program, inputs, &trace, program.output, &PathOptions::expansions(expansions))
        .map_err(|e| e.to_string())?;
    let items = result.items_sum();
    let own_gap = frobenius(&(&items - &result.value));
    let mut holds = own_gap <= result.rounding_band;
    println!(
        "{} ({}, {}): {} rows x {} outputs, {} paths, {} remainders (mass <= {:.3e}), {:.1}s",
        imported.name,
        imported.kind,
        imported.record["source"].as_str().unwrap_or("export"),
        result.value.nrows(),
        result.value.ncols(),
        result.paths.len(),
        result.remainders.len(),
        result.remainder_mass_bound,
        started.elapsed().as_secs_f64()
    );
    println!(
        "max |path sum - traced value| {:.3e} (Frobenius {:.3e}, band {:.3e}) {}",
        max_abs(&items, &result.value),
        own_gap,
        result.rounding_band,
        if own_gap <= result.rounding_band { "ok" } else { "FAIL" }
    );
    if let Some(torch_dir) = args.get(3) {
        let torch = read_f64(&PathBuf::from(torch_dir).join("logits.f64"), result.value.ncols())?;
        let p = result.value.ncols();
        let (SlotValues::Tokens(a), SlotValues::Tokens(b)) = (&inputs.slots[0], &inputs.slots[1]) else {
            return Err("a torch comparison needs two token slots (a, b)".to_string());
        };
        let rows: Vec<usize> = a.iter().zip(b).map(|(&a, &b)| a as usize * p + b as usize).collect();
        if rows.iter().any(|&row| row >= torch.nrows()) {
            return Err(format!("torch logits have {} rows", torch.nrows()));
        }
        let reference = torch.select(ndarray::Axis(0), &rows);
        let gap = frobenius(&(&items - &reference));
        println!(
            "max |path sum - torch logit| {:.3e} (Frobenius {:.3e}, 2 x band {:.3e}) {}",
            max_abs(&items, &reference),
            gap,
            2.0 * result.rounding_band,
            if gap <= 2.0 * result.rounding_band { "ok" } else { "FAIL" }
        );
        holds &= gap <= 2.0 * result.rounding_band;
    }
    let total: f64 = result.paths.iter().map(|p| p.mass()).sum::<f64>()
        + result.remainders.iter().map(|r| frobenius(&r.net)).sum::<f64>();
    println!(
        "items carrying 99% of the mass: {} of {} (90%: {}, 50%: {}); opaque sources {}; conditions {}",
        result.mass_count(0.99),
        result.paths.len() + result.remainders.len(),
        result.mass_count(0.9),
        result.mass_count(0.5),
        result.sources.iter().filter(|(_, kind)| matches!(kind, gam_mpd::paths::SourceKind::Opaque(_))).count(),
        result.conditions.len()
    );
    let mut order: Vec<usize> = (0..result.paths.len()).collect();
    order.sort_by(|&x, &y| result.paths[y].mass().total_cmp(&result.paths[x].mass()));
    for &index in order.iter().take(12) {
        let path = &result.paths[index];
        let nodes: Vec<String> = path.nodes.iter().map(|&node| label(program, node)).collect();
        println!("  mass {:.3e} ({:.1}%) share {:+.4}  {}", path.mass(), 100.0 * path.mass() / total, path.share, nodes.join(" -> "));
    }
    Ok(holds)
}
