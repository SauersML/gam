//! REAL-DATA routability / interference-floor audit on banked LLM activations.
//!
//! Run with the banked activation root as the first positional argument:
//!
//! ```text
//! cargo run -p gam-sae --example routability_real -- <data-root>
//! ```
//!
//! For each layer this fits a fixed-`K` sparse dictionary, forms the
//! reconstruction residual `R = X - Xhat`, and reports the residual routability
//! floor, empirical cross-gate quantiles, coherence excess, and unroutable-energy
//! fraction.

use gam_sae::routability::{routability_audit, routability_floor};
use gam_sae::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
use ndarray::Array2;
use std::path::Path;
use std::process::ExitCode;

const LAYERS: &[(&str, &str)] = &[
    (
        "qwen3_8b_wikitext_resid_L18",
        "harvest_out/qwen3_8b_wikitext/resid_L18.npy",
    ),
    ("qwen3_35b_msae_L17_train", "msae_l17/L17_train.f32.npy"),
];

const N_MAX_ROWS: usize = 20_000;
const N_AUDIT_ROWS: usize = 8_192;
const K_ATOMS: usize = 8_192;
const ACTIVE: usize = 32;
const EPOCHS: usize = 20;
const DELTA: f64 = 0.01;

fn main() -> ExitCode {
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "routability_real".to_string());
    let Some(root_arg) = args.next() else {
        eprintln!("usage: {program} <data-root>");
        return ExitCode::from(2);
    };
    if let Err(err) = run(Path::new(&root_arg)) {
        eprintln!("[routability_real] error: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(root: &Path) -> Result<(), String> {
    for (name, rel) in LAYERS {
        let path = root.join(rel);
        if !path.exists() {
            eprintln!(
                "[routability_real] SKIP {name}: {} not present",
                path.display()
            );
            continue;
        }

        let (n_full, p, x) = read_npy_subsample(&path, N_MAX_ROWS)?;
        let n = x.nrows();

        let mut cfg = SparseDictConfig::new(K_ATOMS.min(n.saturating_sub(1).max(1)));
        cfg.active = ACTIVE;
        cfg.max_epochs = EPOCHS;
        cfg.minibatch = 2_048;
        cfg.score_tile = 4_096;
        cfg.score_mode = gam_sae::gpu::GpuPolicy::Off;
        let fit = match fit_sparse_dictionary(x.view(), &cfg) {
            Ok(fit) => fit,
            Err(err) => {
                eprintln!("[routability_real] {name}: fit failed: {err}");
                continue;
            }
        };
        let k = fit.decoder.nrows();

        let xhat = fit.reconstruct();
        let mut residual = x.clone();
        residual -= &xhat;

        let audit_rows = N_AUDIT_ROWS.min(residual.nrows());
        let r_sample = residual.slice(ndarray::s![0..audit_rows, ..]);
        let audit = routability_audit(fit.decoder.view(), r_sample, 1, DELTA, &[0.5, 0.9, 0.99])
            .map_err(|err| format!("routability_audit on {name}: {err}"))?;
        let floor = routability_floor(p, k, 1, DELTA);

        let (unroutable_energy, total_energy) =
            unroutable_energy_fraction(fit.decoder.view(), r_sample, floor.floor);
        let unroutable_frac = if total_energy > 0.0 {
            unroutable_energy / total_energy
        } else {
            0.0
        };

        let resid_share = 1.0 - fit.explained_variance;

        println!("==================== {name} ====================");
        println!(
            "  data: N_full={n_full} used N={n} p={p} | dict K={k} active={ACTIVE} epochs_run={} EV={:.4} (resid share {:.4})",
            fit.epochs, fit.explained_variance, resid_share
        );
        println!(
            "  floor(delta={DELTA}) = sqrt(1/p)+sqrt(2 ln(K/delta)/p) = {:.5}",
            floor.floor
        );
        println!(
            "  empirical max-cross-gate quantiles: q50={:.5} q90={:.5} q99={:.5} max={:.5} mean={:.5}",
            audit.quantiles[0].1,
            audit.quantiles[1].1,
            audit.quantiles[2].1,
            audit.empirical_max,
            audit.empirical_mean
        );
        println!(
            "  coherence_excess (emp (1-delta)-quantile / floor) = {:.4}",
            audit.coherence_excess
        );
        println!(
            "  fraction of residual ROWS below floor (count)  = {:.4}",
            audit.fraction_below_floor
        );
        println!(
            "  UNROUTABLE-ENERGY FRACTION (energy below floor) = {:.4}",
            unroutable_frac
        );
    }
    Ok(())
}

fn unroutable_energy_fraction(
    decoder: ndarray::ArrayView2<'_, f32>,
    residuals: ndarray::ArrayView2<'_, f32>,
    floor: f64,
) -> (f64, f64) {
    let k = decoder.nrows();
    let mut unroutable = 0.0f64;
    let mut total = 0.0f64;
    for r in residuals.outer_iter() {
        let mut norm2 = 0.0f64;
        for &v in &r {
            norm2 += v as f64 * v as f64;
        }
        total += norm2;
        if norm2 <= 1.0e-24 {
            continue;
        }
        let norm = norm2.sqrt();
        let mut best = 0.0f64;
        for atom_idx in 0..k {
            let atom = decoder.row(atom_idx);
            let mut dot = 0.0f64;
            for (rv, av) in r.iter().zip(atom.iter()) {
                dot += *rv as f64 * *av as f64;
            }
            let gate = dot.abs();
            if gate > best {
                best = gate;
            }
        }
        if best / norm <= floor {
            unroutable += norm2;
        }
    }
    (unroutable, total)
}

fn parse_npy_header(
    head: &[u8],
    path: &Path,
) -> Result<(usize, usize, usize, bool, usize), String> {
    if head.len() <= 12 || &head[0..6] != b"\x93NUMPY" {
        return Err(format!("{}: not a .npy file", path.display()));
    }
    let major = head[6];
    let (header_len, data_off) = if major >= 2 {
        let header_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        (header_len, 12 + header_len)
    } else {
        let header_len = u16::from_le_bytes([head[8], head[9]]) as usize;
        (header_len, 10 + header_len)
    };
    if data_off > head.len() {
        return Err(format!(
            "{}: header exceeds initial read buffer",
            path.display()
        ));
    }
    let header = std::str::from_utf8(&head[data_off - header_len..data_off])
        .map_err(|err| format!("{}: header is not utf8: {err}", path.display()))?;
    let is_f4 = header.contains("'<f4'") || header.contains("\"<f4\"");
    let is_f2 = header.contains("'<f2'") || header.contains("\"<f2\"");
    if !(is_f4 || is_f2) {
        return Err(format!(
            "{}: expected little-endian <f4 or <f2; header: {header}",
            path.display()
        ));
    }
    if !(header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false")) {
        return Err(format!(
            "{}: expected C-order; header: {header}",
            path.display()
        ));
    }
    let shape_start = header
        .find("'shape':")
        .ok_or_else(|| format!("{}: missing shape key", path.display()))?
        + "'shape':".len();
    let paren_open = header[shape_start..]
        .find('(')
        .ok_or_else(|| format!("{}: missing shape open paren", path.display()))?
        + shape_start
        + 1;
    let paren_close = header[paren_open..]
        .find(')')
        .ok_or_else(|| format!("{}: missing shape close paren", path.display()))?
        + paren_open;
    let dims: Vec<usize> = header[paren_open..paren_close]
        .split(',')
        .filter_map(|token| token.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 2 {
        return Err(format!(
            "{}: expected a 2-D array, got {dims:?}",
            path.display()
        ));
    }
    let elem = if is_f4 { 4 } else { 2 };
    Ok((dims[0], dims[1], elem, is_f4, data_off))
}

fn read_npy_subsample(path: &Path, cap: usize) -> Result<(usize, usize, Array2<f32>), String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file =
        std::fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut head = vec![0u8; 256];
    file.read_exact(&mut head)
        .map_err(|err| format!("read header {}: {err}", path.display()))?;
    let (n_full, p, elem, is_f4, data_off) = parse_npy_header(&head, path)?;
    let take = cap.min(n_full);
    let stride = (n_full / take).max(1);
    let row_bytes = p * elem;
    let mut buf = vec![0u8; row_bytes];
    let mut out = Array2::<f32>::zeros((take, p));
    for i in 0..take {
        let src_row = i * stride;
        let off = data_off as u64 + (src_row as u64) * (row_bytes as u64);
        file.seek(SeekFrom::Start(off))
            .map_err(|err| format!("seek {}: {err}", path.display()))?;
        file.read_exact(&mut buf)
            .map_err(|err| format!("read row {src_row} {}: {err}", path.display()))?;
        if is_f4 {
            for c in 0..p {
                let b = c * 4;
                out[[i, c]] = f32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]]);
            }
        } else {
            for c in 0..p {
                let b = c * 2;
                out[[i, c]] = f16_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]]));
            }
        }
    }
    Ok((n_full, p, out))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let f = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            let mut m = mant as u32;
            let mut e: i32 = -1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (127 - 15 + 1 + e) as u32;
            ((sign as u32) << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        ((sign as u32) << 31) | (exp32 << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(f)
}
