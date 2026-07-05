//! REAL-DATA routability / interference-floor audit on banked LLM activations.
//!
//! This is a measurement harness, not a pass/fail gate, so it is compiled ONLY
//! under the off-by-default `real_data_harness` cargo feature (SPEC rule 15 bans
//! the ignored-test / XFAIL pattern; a feature gate is the conditional-compilation
//! equivalent that never runs in normal CI). Run explicitly on MSI (where the
//! banked activations live):
//!
//! ```text
//!   cargo test -p gam-sae --test routability_real --features real_data_harness -- --nocapture
//! ```
//!
//! For each layer this fits a fixed-`K` sparse dictionary (the collapsed linear
//! lane), forms the reconstruction residual `R = X − X̂`, and runs
//! [`gam_sae::routability::routability_audit`] on the residual rows against the
//! fitted decoder. It reports, per layer:
//!   * the closed-form routability floor `√(1/p)+√(2 ln(K/δ)/p)` at δ = 0.01;
//!   * the empirical max-cross-gate quantiles of the residual;
//!   * `coherence_excess` = empirical (1−δ)-quantile / floor (how much a trained
//!     dictionary's residual coherence exceeds the generic-position model);
//!   * the **unroutable-energy fraction** — the share of residual ℓ₂ energy that
//!     sits in rows whose best atom gate is at or below the floor, i.e. residual
//!     directions no width-`p` router can ever separate. That is the routing-side
//!     dark-matter fraction of the layer.

#![cfg(feature = "real_data_harness")]

use gam_sae::routability::{routability_audit, routability_floor};
use gam_sae::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
use ndarray::Array2;

/// Banked activation slices as `(label, path-relative-to-data-root)`. The data
/// root is supplied at runtime via the `ROUTABILITY_DATA_DIR` environment
/// variable — NO absolute host/cluster path is baked into the source (build.rs
/// bans committed cluster paths). A run without that env var (or without the
/// slices) simply skips each layer with a message. Invoke on the cluster as:
///   ROUTABILITY_DATA_DIR=<data-root> \
///     cargo test -p gam-sae --test routability_real --features real_data_harness -- --nocapture
const LAYERS: &[(&str, &str)] = &[
    (
        "qwen3_8b_wikitext_resid_L18",
        "harvest_out/qwen3_8b_wikitext/resid_L18.npy",
    ),
    ("qwen3_35b_msae_L17_train", "msae_l17/L17_train.f32.npy"),
];

/// Data root for the banked slices, from `$ROUTABILITY_DATA_DIR` (no default —
/// the harness skips when unset so no cluster path is ever hardcoded).
fn data_root() -> Option<String> {
    std::env::var("ROUTABILITY_DATA_DIR")
        .ok()
        .filter(|s| !s.is_empty())
}

// Measurement knobs (shared across layers). Kept modest so a full pass finishes
// in a couple of minutes on a fast node while still using a realistic expansion.
const N_MAX_ROWS: usize = 20_000; // subsample cap for the fit
const N_AUDIT_ROWS: usize = 8_192; // residual rows scored in the audit
const K_ATOMS: usize = 8_192; // dictionary width (expansion over p)
const ACTIVE: usize = 32; // per-row active budget s
const EPOCHS: usize = 20;
const DELTA: f64 = 0.01;

#[test]
fn real_qwen_routability_audit() {
    let root = match data_root() {
        Some(r) => r,
        None => {
            eprintln!(
                "[routability_real] SKIP all layers: set ROUTABILITY_DATA_DIR to the banked-slice root"
            );
            return;
        }
    };
    for (name, rel) in LAYERS {
        let path = format!("{root}/{rel}");
        let path = path.as_str();
        if !std::path::Path::new(path).exists() {
            eprintln!("[routability_real] SKIP {name}: {path} not present");
            continue;
        }
        // Seek-and-read only a strided subsample of rows — the banked slices are
        // multi-GB, so never load the whole file (SPEC: must not run out of memory).
        let (n_full, p, x) = read_npy_subsample(path, N_MAX_ROWS);
        let n = x.nrows();

        // Fit the collapsed-linear-lane dictionary.
        let mut cfg = SparseDictConfig::new(K_ATOMS.min(n.saturating_sub(1).max(1)));
        cfg.active = ACTIVE;
        cfg.max_epochs = EPOCHS;
        cfg.minibatch = 2_048;
        cfg.score_tile = 4_096;
        cfg.score_mode = gam_sae::gpu::GpuMode::Off;
        let fit = match fit_sparse_dictionary(x.view(), &cfg) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[routability_real] {name}: fit failed: {e}");
                continue;
            }
        };
        let k = fit.decoder.nrows();

        // Residual R = X − X̂ over the fitted (in-sample) rows.
        let xhat = fit.reconstruct();
        let mut residual = x.clone();
        residual -= &xhat;

        // Audit a sample of residual rows against the fitted decoder (block_size=1
        // linear lane). f64 accumulation inside the audit.
        let audit_rows = N_AUDIT_ROWS.min(residual.nrows());
        let r_sample = residual.slice(ndarray::s![0..audit_rows, ..]);
        let audit = routability_audit(
            fit.decoder.view(),
            r_sample,
            1,
            DELTA,
            &[0.5, 0.9, 0.99],
        )
        .expect("routability_audit on residual rows");
        let floor = routability_floor(p, k, 1, DELTA);

        // Energy-weighted unroutable fraction: share of residual ℓ₂ energy in rows
        // whose max cross-gate ≤ floor (no atom can route them). f64.
        let (unroutable_energy, total_energy) =
            unroutable_energy_fraction(fit.decoder.view(), r_sample, floor.floor);
        let unroutable_frac = if total_energy > 0.0 {
            unroutable_energy / total_energy
        } else {
            0.0
        };

        // Residual energy share of the layer (1 − EV), for context.
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
            "  UNROUTABLE-ENERGY FRACTION (energy below floor) = {:.4}  <-- routing-side dark matter",
            unroutable_frac
        );
    }
}

/// Energy-weighted unroutable fraction: sum of `‖r‖²` over residual rows whose max
/// cross-gate `max_k |⟨r,d_k⟩|/‖r‖ ≤ floor`, and the total residual energy. f64.
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
        for &v in r.iter() {
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

/// Parse a NumPy `.npy` (v1.0/v2.0) header for a C-order, little-endian `<f4` or
/// `<f2` 2-D array. Returns `(n, p, elem_bytes, is_f4, data_off)`. Panics with a
/// clear message on anything else — a measurement helper, not a general parser.
fn parse_npy_header(head: &[u8], path: &str) -> (usize, usize, usize, bool, usize) {
    assert!(
        head.len() > 12 && &head[0..6] == b"\x93NUMPY",
        "{path}: not a .npy file"
    );
    let major = head[6];
    let (header_len, data_off) = if major >= 2 {
        let hl = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        (hl, 12 + hl)
    } else {
        let hl = u16::from_le_bytes([head[8], head[9]]) as usize;
        (hl, 10 + hl)
    };
    let header = std::str::from_utf8(&head[data_off - header_len..data_off]).expect("utf8 header");
    let is_f4 = header.contains("'<f4'") || header.contains("\"<f4\"");
    let is_f2 = header.contains("'<f2'") || header.contains("\"<f2\"");
    assert!(
        is_f4 || is_f2,
        "{path}: expected little-endian <f4 or <f2; header: {header}"
    );
    assert!(
        header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false"),
        "{path}: expected C-order; header: {header}"
    );
    let shape_start = header.find("'shape':").expect("shape key") + "'shape':".len();
    let paren_open = header[shape_start..].find('(').expect("shape (") + shape_start + 1;
    let paren_close = header[paren_open..].find(')').expect("shape )") + paren_open;
    let dims: Vec<usize> = header[paren_open..paren_close]
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    assert_eq!(dims.len(), 2, "{path}: expected a 2-D array, got {dims:?}");
    let elem = if is_f4 { 4 } else { 2 };
    (dims[0], dims[1], elem, is_f4, data_off)
}

/// Seek-and-read a deterministic stride subsample of at most `cap` rows from a
/// banked `.npy` slice, converting to `f32`. Never loads the whole (multi-GB)
/// file: only the header plus `min(cap, n)` rows of `p·elem` bytes are read.
/// Returns `(n_full, p, rows)`.
fn read_npy_subsample(path: &str, cap: usize) -> (usize, usize, Array2<f32>) {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut head = vec![0u8; 256];
    f.read_exact(&mut head)
        .unwrap_or_else(|e| panic!("read header {path}: {e}"));
    let (n_full, p, elem, is_f4, data_off) = parse_npy_header(&head, path);
    let take = cap.min(n_full);
    let stride = (n_full / take).max(1);
    let row_bytes = p * elem;
    let mut buf = vec![0u8; row_bytes];
    let mut out = Array2::<f32>::zeros((take, p));
    for i in 0..take {
        let src_row = i * stride;
        let off = data_off as u64 + (src_row as u64) * (row_bytes as u64);
        f.seek(SeekFrom::Start(off))
            .unwrap_or_else(|e| panic!("seek {path}: {e}"));
        f.read_exact(&mut buf)
            .unwrap_or_else(|e| panic!("read row {src_row} {path}: {e}"));
        if is_f4 {
            for c in 0..p {
                let b = c * 4;
                out[[i, c]] =
                    f32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]]);
            }
        } else {
            for c in 0..p {
                let b = c * 2;
                out[[i, c]] = f16_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]]));
            }
        }
    }
    (n_full, p, out)
}

/// IEEE-754 half → single precision (no external crate; handles subnormals,
/// inf/NaN). Used only for `<f2`-banked activations.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let f = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            // Subnormal: normalise.
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
