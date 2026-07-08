//! M1 — the curved CROSSCODER, the first client of the block-generic multi-block
//! REML fit ([`SaeManifoldTerm::run_multiblock_reml_fit`]).
//!
//! # What a crosscoder is here
//!
//! A crosscoder learns a shared dictionary that reconstructs SEVERAL layers'
//! activations at once. The curved crosscoder makes that dictionary a single
//! low-dimensional manifold atom: ONE shared latent coordinate `t` per token is
//! decoded through per-layer decoder blocks into each layer's activations. The
//! anchor layer `Z` (`n × p_x`) is block 0; every FURTHER layer `Y_ℓ`
//! (`n × p_ℓ`) is an [`OutputBlock`], and the augmented target is
//! `Z̃ = [Z | √λ_1·Y_1 | … | √λ_{K-1}·Y_{K-1}]`. Because all blocks decode from
//! the same per-row basis `Φ(t_i)` and gate `a_i`, the latent is shared BY
//! CONSTRUCTION — the ordinary arrow-Schur joint fit reconstructs every layer,
//! and the per-layer relevance weight `λ_ℓ` is REML-selected (the closed-form
//! variance ratio `λ_ℓ = (R_x/p_x)/(R_ℓ/p_ℓ)`), never hand-tuned.
//! [`OutputBlock::split_honest_decoder`] un-does the `√λ_ℓ`, so each layer's
//! decoder is recovered in that layer's own units.
//!
//! # Follow-up client (not built here)
//!
//! Swap the *layer* axis for the *checkpoint* axis and the exact same driver is a
//! DEVELOPMENT-CODER: block `ℓ` is a later training checkpoint's activations at
//! the same tokens, the shared `t` is a feature that persists across training,
//! and `λ_ℓ` REML-selects which checkpoints the feature is legible in — a
//! curved probe of feature formation over training. Nothing in the driver
//! changes; only what the block targets mean.
//!
//! # Usage
//!
//! ```text
//! cargo run -p gam-sae --release --example curved_crosscoder -- \
//!     <anchor_layer.npy> <block_layer_1.npy> [block_layer_2.npy ...] \
//!     [--max-rows N] [--pca D] [--harmonics H] [--sweeps S]
//! ```
//!
//! Each `.npy` is a row-aligned `(N, hidden)` little-endian `<f4`/`<f2` matrix of
//! per-token residual-stream activations for one layer (SAME tokens, SAME order,
//! SAME N across files). Each layer is PCA-reduced to `D` dims (default 8) before
//! the fit. Output: the REML-selected `λ_ℓ` and per-layer explained variance,
//! plus a shared-latent baseline (anchor-only fit EV) for context.

use std::path::Path;
use std::sync::Arc;

use gam_sae::identifiability::thin_svd_scores;
use gam_sae::manifold::{
    AssignmentMode, LatentManifold, OutputBlock, PeriodicHarmonicEvaluator, SaeAssignment,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    TwoBlockRemlControls, stack_augmented_target,
};
use ndarray::{Array1, Array2};

struct Args {
    layers: Vec<String>,
    max_rows: usize,
    pca: usize,
    harmonics: usize,
    sweeps: usize,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    let program = raw.first().cloned().unwrap_or_else(|| "curved_crosscoder".to_string());
    let mut layers = Vec::new();
    let mut max_rows = 20000usize;
    let mut pca = 8usize;
    let mut harmonics = 3usize;
    let mut sweeps = 25usize;
    let mut i = 1usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--max-rows" => {
                i += 1;
                max_rows = raw.get(i).ok_or("--max-rows needs a value")?.parse().map_err(|e| format!("--max-rows: {e}"))?;
            }
            "--pca" => {
                i += 1;
                pca = raw.get(i).ok_or("--pca needs a value")?.parse().map_err(|e| format!("--pca: {e}"))?;
            }
            "--harmonics" => {
                i += 1;
                harmonics = raw.get(i).ok_or("--harmonics needs a value")?.parse().map_err(|e| format!("--harmonics: {e}"))?;
            }
            "--sweeps" => {
                i += 1;
                sweeps = raw.get(i).ok_or("--sweeps needs a value")?.parse().map_err(|e| format!("--sweeps: {e}"))?;
            }
            other => layers.push(other.to_string()),
        }
        i += 1;
    }
    if layers.len() < 2 {
        return Err(format!(
            "usage: {program} <anchor_layer.npy> <block_layer.npy> [more_layers.npy ...] \
             [--max-rows N] [--pca D] [--harmonics H] [--sweeps S]\n\
             need an anchor layer plus at least one further layer"
        ));
    }
    if pca < 2 {
        return Err(format!("--pca must be ≥ 2 (need two PCs for the circle seed); got {pca}"));
    }
    Ok(Args { layers, max_rows, pca, harmonics, sweeps })
}

/// Explained variance of `fitted` against `target` (same shape); scale-invariant,
/// so measuring the SCALED augmented block equals measuring the honest-units one.
fn explained_variance(target: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let (n, p) = target.dim();
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for j in 0..p {
        let mean = (0..n).map(|i| target[[i, j]]).sum::<f64>() / n as f64;
        for i in 0..n {
            let r = target[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            let c = target[[i, j]] - mean;
            sst += c * c;
        }
    }
    if sst > 0.0 { 1.0 - ssr / sst } else { f64::NAN }
}

fn main() -> Result<(), String> {
    let args = parse_args()?;

    // Load + PCA-reduce every layer, holding N identical across files so the
    // per-layer subsamples stay row-aligned (same stride ⇒ same tokens).
    let mut reduced: Vec<Array2<f64>> = Vec::with_capacity(args.layers.len());
    let mut n_full_ref: Option<usize> = None;
    for path in &args.layers {
        let (n_full, p_raw, raw) = read_npy_subsample_f64(Path::new(path), args.max_rows)?;
        match n_full_ref {
            None => n_full_ref = Some(n_full),
            Some(n0) if n0 != n_full => {
                return Err(format!(
                    "row misalignment: '{}' has N={n_full} but the anchor has N={n0}; a crosscoder \
                     needs the SAME tokens (same N, same order) in every layer file",
                    path
                ));
            }
            Some(_) => {}
        }
        let keep = args.pca.min(p_raw).min(raw.nrows());
        let scores = thin_svd_scores(raw.view(), keep)?;
        println!(
            "layer '{}': N={} rows fit, raw hidden p={p_raw} → PCA D={keep}",
            layer_name(path),
            raw.nrows()
        );
        reduced.push(scores);
    }

    let anchor = reduced[0].clone();
    let (n, p_x) = anchor.dim();
    let block_layers = &reduced[1..];
    let p_tot = p_x + block_layers.iter().map(|b| b.ncols()).sum::<usize>();

    // Shared-latent circle seed from the anchor's two leading PCs (phase). The
    // joint fit refines the coordinate; this is only the deterministic start.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(2 * args.harmonics + 1)?);
    let mut coords = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        let a = anchor[[i, 0]];
        let b = anchor[[i, 1]];
        coords[[i, 0]] = (b.atan2(a) / std::f64::consts::TAU).rem_euclid(1.0);
    }

    // Build the crosscoder blocks (each further layer at initial log λ_ℓ = 0).
    let mut blocks: Vec<OutputBlock> = Vec::with_capacity(block_layers.len());
    for (idx, layer) in block_layers.iter().enumerate() {
        blocks.push(OutputBlock::new(
            layer_name(&args.layers[idx + 1]),
            layer.clone(),
            0.0,
        )?);
    }

    // --- Curved crosscoder fit: one shared t, per-layer decoders, λ_ℓ by REML.
    let (mut term, mut rho) = circle_term(&evaluator, &coords, p_tot)?;
    term.set_guards_enabled(false);
    let report = term.run_multiblock_reml_fit(
        anchor.view(),
        &mut blocks,
        &mut rho,
        None,
        TwoBlockRemlControls {
            max_sweeps: args.sweeps,
            inner_max_iter: 60,
            step_size: 1.0,
            ridge_ext_coord: 1e-6,
            ridge_beta: 1e-6,
            log_lambda_tol: 1e-3,
        },
    )?;

    println!(
        "\ncurved crosscoder: {} blocks, {} sweeps, converged={}",
        report.blocks.len(),
        report.sweeps,
        report.converged
    );

    // Per-layer EV, measured on the SCALED augmented block against the fitted
    // reconstruction (EV is scale-invariant, so this equals honest-units EV).
    let augmented = stack_augmented_target(anchor.view(), &blocks)?;
    let fitted = term.try_fitted_for_rho(&rho)?;
    let anchor_ev = explained_variance(
        &augmented.slice(ndarray::s![.., ..p_x]).to_owned(),
        &fitted.slice(ndarray::s![.., ..p_x]).to_owned(),
    );
    println!("  anchor layer '{}': EV = {anchor_ev:.4}  (λ_x ≡ 1 reference)", layer_name(&args.layers[0]));
    let mut offset = p_x;
    for (idx, block) in blocks.iter().enumerate() {
        let pl = block.block_dim();
        let ev = explained_variance(
            &augmented.slice(ndarray::s![.., offset..offset + pl]).to_owned(),
            &fitted.slice(ndarray::s![.., offset..offset + pl]).to_owned(),
        );
        let outcome = &report.blocks[idx];
        println!(
            "  block layer '{}': EV = {ev:.4}, REML log λ_ℓ = {:.4} (λ_ℓ = {:.4}), identifiable = {}",
            block.label,
            outcome.log_lambda,
            outcome.log_lambda.exp(),
            outcome.identifiable
        );
        offset += pl;
    }

    // --- Shared-latent baseline: fit the anchor ALONE and report its EV, so the
    // crosscoder's anchor EV can be read against what one layer alone supports.
    let (mut term_a, mut rho_a) = circle_term(&evaluator, &coords, p_x)?;
    term_a.set_guards_enabled(false);
    term_a.run_joint_fit_arrow_schur(anchor.view(), &mut rho_a, None, 60, 1.0, 1e-6, 1e-6)?;
    let fitted_a = term_a.try_fitted_for_rho(&rho_a)?;
    println!(
        "\nanchor-only baseline: anchor EV = {:.4} (single-layer fit, no cross-layer sharing)",
        explained_variance(&anchor, &fitted_a)
    );
    Ok(())
}

/// A short display name for a layer file (its stem).
fn layer_name(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

/// Build a K=1 circle term at augmented output width `p_tot`, seeded at `coords`.
fn circle_term(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new(
        "crosscoder",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )?
    .with_basis_second_jet(evaluator.clone());
    let n = coords.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )?;
    let term = SaeManifoldTerm::new(vec![atom], assignment)?;
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    Ok((term, rho))
}

// ---- Minimal row-aligned .npy reader (little-endian <f4 / <f2, C-order, 2-D).

fn read_npy_subsample_f64(path: &Path, cap: usize) -> Result<(usize, usize, Array2<f64>), String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut head = vec![0u8; 512];
    file.read_exact(&mut head).map_err(|err| format!("read header {}: {err}", path.display()))?;
    let (n_full, p, elem, is_f4, data_off) = parse_npy_header(&head, path)?;
    let take = cap.min(n_full).max(1);
    let stride = (n_full / take).max(1);
    let row_bytes = p.checked_mul(elem).ok_or_else(|| format!("{}: row byte overflow", path.display()))?;
    let mut buf = vec![0u8; row_bytes];
    let mut out = Array2::<f64>::zeros((take, p));
    for i in 0..take {
        let off = data_off as u64 + (i * stride) as u64 * row_bytes as u64;
        file.seek(SeekFrom::Start(off)).map_err(|err| format!("seek {}: {err}", path.display()))?;
        file.read_exact(&mut buf).map_err(|err| format!("read row {}: {err}", path.display()))?;
        if is_f4 {
            for c in 0..p {
                let b = c * 4;
                out[[i, c]] = f32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]]) as f64;
            }
        } else {
            for c in 0..p {
                let b = c * 2;
                out[[i, c]] = f16_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]])) as f64;
            }
        }
    }
    Ok((n_full, p, out))
}

fn parse_npy_header(head: &[u8], path: &Path) -> Result<(usize, usize, usize, bool, usize), String> {
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
        return Err(format!("{}: header exceeds initial read buffer", path.display()));
    }
    let header = std::str::from_utf8(&head[data_off - header_len..data_off])
        .map_err(|err| format!("{}: header not utf8: {err}", path.display()))?;
    let is_f4 = header.contains("'<f4'") || header.contains("\"<f4\"");
    let is_f2 = header.contains("'<f2'") || header.contains("\"<f2\"");
    if !(is_f4 || is_f2) {
        return Err(format!("{}: expected <f4 or <f2; header: {header}", path.display()));
    }
    if !(header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false")) {
        return Err(format!("{}: expected C-order; header: {header}", path.display()));
    }
    let shape_start = header.find("'shape':").ok_or_else(|| format!("{}: missing shape key", path.display()))?
        + "'shape':".len();
    let paren_open = header[shape_start..].find('(').ok_or_else(|| format!("{}: missing shape open paren", path.display()))?
        + shape_start + 1;
    let paren_close = header[paren_open..].find(')').ok_or_else(|| format!("{}: missing shape close paren", path.display()))?
        + paren_open;
    let dims: Vec<usize> = header[paren_open..paren_close]
        .split(',')
        .filter_map(|token| token.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 2 {
        return Err(format!("{}: expected a 2-D array, got {dims:?}", path.display()));
    }
    let elem = if is_f4 { 4 } else { 2 };
    Ok((dims[0], dims[1], elem, is_f4, data_off))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
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
    f32::from_bits(bits)
}
