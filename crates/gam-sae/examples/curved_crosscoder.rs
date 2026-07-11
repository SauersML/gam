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
    out_dir: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    let program = raw
        .first()
        .cloned()
        .unwrap_or_else(|| "curved_crosscoder".to_string());
    let mut layers = Vec::new();
    let mut max_rows = 20000usize;
    let mut pca = 8usize;
    let mut harmonics = 3usize;
    let mut sweeps = 25usize;
    let mut out_dir: Option<String> = None;
    let mut i = 1usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--out-dir" => {
                i += 1;
                out_dir = Some(raw.get(i).ok_or("--out-dir needs a value")?.clone());
            }
            "--max-rows" => {
                i += 1;
                max_rows = raw
                    .get(i)
                    .ok_or("--max-rows needs a value")?
                    .parse()
                    .map_err(|e| format!("--max-rows: {e}"))?;
            }
            "--pca" => {
                i += 1;
                pca = raw
                    .get(i)
                    .ok_or("--pca needs a value")?
                    .parse()
                    .map_err(|e| format!("--pca: {e}"))?;
            }
            "--harmonics" => {
                i += 1;
                harmonics = raw
                    .get(i)
                    .ok_or("--harmonics needs a value")?
                    .parse()
                    .map_err(|e| format!("--harmonics: {e}"))?;
            }
            "--sweeps" => {
                i += 1;
                sweeps = raw
                    .get(i)
                    .ok_or("--sweeps needs a value")?
                    .parse()
                    .map_err(|e| format!("--sweeps: {e}"))?;
            }
            other => layers.push(other.to_string()),
        }
        i += 1;
    }
    if layers.len() < 2 {
        return Err(format!(
            "usage: {program} <anchor_layer.npy> <block_layer.npy> [more_layers.npy ...] \
             [--max-rows N] [--pca D] [--harmonics H] [--sweeps S] [--out-dir DIR]\n\
             need an anchor layer plus at least one further layer"
        ));
    }
    if pca < 2 {
        return Err(format!(
            "--pca must be ≥ 2 (need two PCs for the circle seed); got {pca}"
        ));
    }
    Ok(Args {
        layers,
        max_rows,
        pca,
        harmonics,
        sweeps,
        out_dir,
    })
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
    // The anchor is the REML REFERENCE block: the variance ratios λ_ℓ are
    // measured relative to it, so it has NO fitted weight of its own — it is the
    // fixed denominator, not a λ=1 result. Reported as such (never fabricated).
    println!(
        "  anchor layer '{}': EV = {anchor_ev:.4}  (REML reference block — no fitted λ; the λ_ℓ below are ratios to this layer's residual variance)",
        layer_name(&args.layers[0])
    );
    let mut block_ev = Vec::with_capacity(blocks.len());
    let mut offset = p_x;
    for (idx, block) in blocks.iter().enumerate() {
        let pl = block.block_dim();
        let ev = explained_variance(
            &augmented
                .slice(ndarray::s![.., offset..offset + pl])
                .to_owned(),
            &fitted
                .slice(ndarray::s![.., offset..offset + pl])
                .to_owned(),
        );
        block_ev.push(ev);
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

    // --- Save the RAW circle geometry so the shared-latent circle can be plotted
    // directly: the token scatter in the circle's own plane, the recovered
    // per-token angle, the densely-sampled fitted curve γ(t), and the HONEST per
    // layer λ (anchor = NaN = reference, never a fabricated 1.0).
    if let Some(dir) = &args.out_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {dir}: {e}"))?;

        // Recovered per-token angle t_i ∈ [0, 1) and the anchor's fitted decoder.
        let t = term.assignment.coords[0]
            .as_matrix()
            .column(0)
            .mapv(|v| v.rem_euclid(1.0));
        let m = term.atoms[0].decoder_coefficients.nrows();
        let b_anchor = term.atoms[0]
            .decoder_coefficients
            .slice(ndarray::s![.., ..p_x])
            .to_owned(); // (M × p_x)

        // The circle's fundamental plane in anchor space is spanned by the
        // first-harmonic decoder rows: Φ column 1 = sin(2πt), column 2 = cos(2πt).
        // Orthonormalise those two anchor-space vectors (Gram–Schmidt) → E (p_x×2).
        let e = fundamental_plane(&b_anchor);

        // Dense curve γ(t) = Φ(t)·B_anchor over a fine grid, projected into E.
        let grid = 512usize;
        let tg = Array2::<f64>::from_shape_fn((grid, 1), |(g, _)| g as f64 / grid as f64);
        let (phi_g, _) = evaluator.evaluate(tg.view())?;
        assert_eq!(phi_g.ncols(), m);
        let gamma = phi_g.dot(&b_anchor); // (grid × p_x)
        let curve_2d = gamma.dot(&e); // (grid × 2)

        // Token scatter: the raw anchor activations projected into the SAME plane.
        let scatter_2d = anchor.dot(&e); // (n × 2)

        // Honest per-layer λ: anchor = NaN (reference), blocks = fitted λ_ℓ.
        let n_layers = args.layers.len();
        let mut layer_lambda = Array1::<f64>::from_elem(n_layers, f64::NAN);
        let mut layer_ev = Array1::<f64>::zeros(n_layers);
        let mut is_anchor = Array1::<f64>::zeros(n_layers);
        is_anchor[0] = 1.0;
        layer_ev[0] = anchor_ev;
        for (idx, outcome) in report.blocks.iter().enumerate() {
            layer_lambda[idx + 1] = outcome.log_lambda.exp();
            layer_ev[idx + 1] = block_ev[idx];
        }

        write_npy_2d(&format!("{dir}/circle_scatter.npy"), scatter_2d.view())?;
        write_npy_2d(&format!("{dir}/circle_curve.npy"), curve_2d.view())?;
        write_npy_1d(&format!("{dir}/circle_angle.npy"), t.view())?;
        write_npy_1d(&format!("{dir}/circle_lambda.npy"), layer_lambda.view())?;
        write_npy_1d(&format!("{dir}/circle_layer_ev.npy"), layer_ev.view())?;
        write_npy_1d(&format!("{dir}/circle_is_anchor.npy"), is_anchor.view())?;
        // Layer labels + a note on the anchor honesty, for the plotter.
        let labels: Vec<String> = args.layers.iter().map(|p| layer_name(p)).collect();
        std::fs::write(
            format!("{dir}/circle_labels.txt"),
            format!(
                "layers: {}\nanchor: {} (reference block; circle_lambda[0]=NaN, NOT a fitted λ)\n\
                 circle_scatter.npy: (n×2) anchor tokens in the fundamental circle plane\n\
                 circle_curve.npy: (512×2) fitted γ(t) in the same plane\n\
                 circle_angle.npy: (n,) recovered per-token t_i ∈ [0,1)\n\
                 circle_lambda.npy: (K,) REML λ per layer; index 0 = anchor = NaN (reference)\n\
                 circle_layer_ev.npy: (K,) per-layer explained variance\n",
                labels.join(", "),
                labels[0]
            ),
        )
        .map_err(|e| format!("write labels: {e}"))?;
        println!(
            "\nsaved raw circle geometry to {dir}/circle_*.npy (scatter {}×2, curve {grid}×2, angle {}, λ per layer)",
            scatter_2d.nrows(),
            t.len()
        );
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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

/// Orthonormal basis `E` (`p_x × 2`) of the circle's fundamental plane in anchor
/// space: the plane spanned by the first-harmonic decoder rows (`Φ` column 1 =
/// `sin 2πt`, column 2 = `cos 2πt`), Gram–Schmidt-orthonormalised. Falls back to
/// the first two coordinate axes if either fundamental direction is degenerate.
fn fundamental_plane(b_anchor: &Array2<f64>) -> Array2<f64> {
    let p_x = b_anchor.ncols();
    let mut e0 = b_anchor.row(1).to_owned(); // sin coefficients
    let mut e1 = b_anchor.row(2).to_owned(); // cos coefficients
    let n0 = e0.dot(&e0).sqrt();
    if n0 > 1e-12 {
        e0.mapv_inplace(|v| v / n0);
    } else {
        e0.fill(0.0);
        if p_x > 0 {
            e0[0] = 1.0;
        }
    }
    let proj = e1.dot(&e0);
    e1 = &e1 - &(&e0 * proj);
    let n1 = e1.dot(&e1).sqrt();
    if n1 > 1e-12 {
        e1.mapv_inplace(|v| v / n1);
    } else {
        e1.fill(0.0);
        if p_x > 1 {
            e1[1] = 1.0;
        }
    }
    let mut e = Array2::<f64>::zeros((p_x, 2));
    for r in 0..p_x {
        e[[r, 0]] = e0[r];
        e[[r, 1]] = e1[r];
    }
    e
}

/// Write a 2-D `f64` array as a little-endian C-order `.npy` (`<f8`, version 1.0).
fn write_npy_2d(path: &str, a: ndarray::ArrayView2<'_, f64>) -> Result<(), String> {
    let (n, m) = a.dim();
    let mut data = Vec::with_capacity(n * m);
    for row in a.rows() {
        for &v in row.iter() {
            data.push(v);
        }
    }
    write_npy(path, &format!("({n}, {m})"), &data)
}

/// Write a 1-D `f64` array as a little-endian C-order `.npy` (`<f8`, version 1.0).
fn write_npy_1d(path: &str, a: ndarray::ArrayView1<'_, f64>) -> Result<(), String> {
    let data: Vec<f64> = a.iter().copied().collect();
    write_npy(path, &format!("({},)", a.len()), &data)
}

/// Core `.npy` v1.0 writer: header dict padded to a 64-byte boundary, then the
/// raw little-endian `f64` payload.
fn write_npy(path: &str, shape: &str, data: &[f64]) -> Result<(), String> {
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}");
    // Pad so that 10 (magic+version+len) + header.len() is a multiple of 64, and
    // the header ends with a newline.
    let unpadded = 10 + header.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut buf: Vec<u8> = Vec::with_capacity(10 + header.len() + data.len() * 8);
    buf.extend_from_slice(b"\x93NUMPY\x01\x00");
    let hlen = header.len() as u16;
    buf.extend_from_slice(&hlen.to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &buf).map_err(|e| format!("write {path}: {e}"))
}

// ---- Minimal row-aligned .npy reader (little-endian <f4 / <f2, C-order, 2-D).

fn read_npy_subsample_f64(path: &Path, cap: usize) -> Result<(usize, usize, Array2<f64>), String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file =
        std::fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut head = vec![0u8; 512];
    file.read_exact(&mut head)
        .map_err(|err| format!("read header {}: {err}", path.display()))?;
    let (n_full, p, elem, is_f4, data_off) = parse_npy_header(&head, path)?;
    let take = cap.min(n_full).max(1);
    let stride = (n_full / take).max(1);
    let row_bytes = p
        .checked_mul(elem)
        .ok_or_else(|| format!("{}: row byte overflow", path.display()))?;
    let mut buf = vec![0u8; row_bytes];
    let mut out = Array2::<f64>::zeros((take, p));
    for i in 0..take {
        let off = data_off as u64 + (i * stride) as u64 * row_bytes as u64;
        file.seek(SeekFrom::Start(off))
            .map_err(|err| format!("seek {}: {err}", path.display()))?;
        file.read_exact(&mut buf)
            .map_err(|err| format!("read row {}: {err}", path.display()))?;
        if is_f4 {
            for c in 0..p {
                let b = c * 4;
                out[[i, c]] =
                    f32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]]) as f64;
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
        .map_err(|err| format!("{}: header not utf8: {err}", path.display()))?;
    let is_f4 = header.contains("'<f4'") || header.contains("\"<f4\"");
    let is_f2 = header.contains("'<f2'") || header.contains("\"<f2\"");
    if !(is_f4 || is_f2) {
        return Err(format!(
            "{}: expected <f4 or <f2; header: {header}",
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
