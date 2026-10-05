//! Two measurements that decide changes to the library fit's optimizer (#2951).
//!
//! EXPORT SETTINGS.json native|checkpoint:PATH OUT host|gpu DRAWS
//!
//! SETTINGS.json is the library fit's (`mpd_library_mdl_2951`); FROM is the library's start at `M`
//! or a fit's checkpoint. Each of the DRAWS draws takes one training batch of the fit's experiments
//! (`interchange::sample`, `M`'s fixed read directions) and one weight sample `θ₊ = μ + σ ⊙ ε` with
//! its antithetic partner `θ₋ = μ − σ ⊙ ε`, and computes in every trainable operator:
//!
//! * `g₊` and `g₋`, the gradients of the batch's data term `Σ KL(M_e ‖ P_e)` at `θ₊` and `θ₋`;
//! * `c`, the gradient at `θ₊` of the layer-local error `½ Σ_b ‖s_P,b − s_M,b‖²`, where `s_P,b` and
//!   `s_M,b` are the streams after block `b` (a layer's attention or its MLP) run by `P` and by `M`
//!   from the same entering stream. The entering streams come from `P`'s own run of the batch's base
//!   sequences (`own`) or from `M`'s (`native`).
//!
//! The first measurement decides whether a control variate built from the local gradient can
//! replace most full gradients: `ĝ = β c + (I/ρ)(g₊ − β c)`, with `I` drawn true with probability
//! `ρ`, is unbiased, and its excess second moment is `(1 − ρ)/ρ · E‖g₊ − β c‖²`. Per operator the
//! report gives `r = E‖g₊ − β c‖² / E‖g₊‖²` at the best fixed `β` (expectations over draws), and the
//! seconds of the local and the full gradient.
//!
//! The second decides how the fit estimates each weight's curvature `h_j = E_q[∂²D/∂θ_j²]`. Stein's
//! estimate `g₊ ε / σ` contains the batch's gradient at the mean times `ε`, noise that grows with the
//! training tokens per batch token; the antithetic estimate `(g₊ − g₋) ε / (2σ)` has the same
//! expectation without that term. Both also contain `Σ_k H_jk σ_k ε_k ε_j / σ_j`, every other
//! weight's noise through its Hessian coupling. The third estimate is the square of `u`, the
//! gradient at `θ₊` of `Σ_t log P(y_t)` over the base sequences run by `P` alone with each `y_t`
//! drawn from `P` (`sampled_label_gradient`): it estimates the Gauss–Newton diagonal, which equals
//! the Hessian of `KL(M ‖ P)` where `P`'s predictions equal `M`'s. Per operator the report gives,
//! over its entries, the median ratio of Stein's and the antithetic estimates' variances over draws
//! and the median `|mean| / sd` over draws of each of the three estimates.
use gam_gpu::{
    GpuPolicy,
    tensor::{Device, Op, Tensor},
};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    interchange::{self, Batch, BlockEngine, Interchange},
    library_mdl,
    operator_program::{Node, OperatorProgram, SlotValues},
    run_check::{layer_nodes, split_sites},
};
use ndarray::{Array2, Zip};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, ops::Range, path::Path, time::Instant};

const USAGE: &str = "EXPORT SETTINGS.json native|checkpoint:PATH OUT host|gpu DRAWS";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    training_sequences: usize,
    context: usize,
    held_out: [usize; 2],
    fit: library_mdl::Settings,
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Standard normal draws (Box–Muller) of shape `dim`.
fn standard_normal(rng: &mut StdRng, dim: (usize, usize)) -> Array2<f64> {
    let n = dim.0 * dim.1;
    let mut values = Vec::with_capacity(n + 1);
    while values.len() < n {
        let u = 1.0 - rng.random::<f64>();
        let angle = std::f64::consts::TAU * rng.random::<f64>();
        let radius = (-2.0 * u.ln()).sqrt();
        values.push(radius * angle.cos());
        values.push(radius * angle.sin());
    }
    values.truncate(n);
    Array2::from_shape_vec(dim, values).expect("shape of the drawn values")
}

/// Per trainable operator (in `trainable` order) its gradient from `gradient`, on the host, zero
/// where the computation did not reach it.
fn downloaded(d: &Device, trainable: &[usize], shapes: &[(usize, usize)], gradient: &BTreeMap<usize, Tensor>) -> Result<Vec<Array2<f64>>, String> {
    trainable
        .iter()
        .zip(shapes)
        .map(|(op, shape)| match gradient.get(op) {
            Some(g) => d.download(g).map_err(error),
            None => Ok(Array2::zeros(*shape)),
        })
        .collect()
}

/// The gradient of `½ Σ_b ‖s_P,b − s_M,b‖²` in `P`'s trainable operators on the sequences
/// `tokens`, each block run by `P` and by `M` from the same entering stream: `P`'s own run's when
/// `own`, else `M`'s.
fn local_gradient(ic: &Interchange, tokens: &[&[u32]], own: bool) -> Result<BTreeMap<usize, Tensor>, String> {
    let (m, p) = ic.models();
    let d = BlockEngine::device(&p);
    let length = tokens.first().map_or(0, |t| t.len());
    let ranges: Vec<Range<usize>> = (0..tokens.len()).map(|i| i * length..(i + 1) * length).collect();
    let mut stream = d.zeros(tokens.len() * length, BlockEngine::width(&p)).map_err(error)?;
    let mut gradient = BTreeMap::new();
    for b in 0..BlockEngine::blocks(&p) {
        let mut after_p = d.copy(&stream).map_err(error)?;
        let tape = p.forward(b, &mut after_p, &ranges, tokens, None, true)?.ok_or("the block kept no tape")?;
        let mut after_m = d.copy(&stream).map_err(error)?;
        m.forward(b, &mut after_m, &ranges, tokens, None, false)?;
        let mut cotangent = d.copy(&after_p).map_err(error)?;
        d.axpy(&mut cotangent, -1.0, &after_m).map_err(error)?;
        p.reverse(b, tape, &mut cotangent, &ranges, None, &mut gradient)?;
        stream = if own { after_p } else { after_m };
    }
    Ok(gradient)
}

/// The head's matrix `E` (classes × hidden, logits `h Eᵀ`) of the flat program `flat`, whose output
/// is a bias-free linear head on its hidden node of width `width`.
fn head_matrix(flat: &OperatorProgram, width: usize) -> Result<Array2<f64>, String> {
    let logits = match &flat.nodes[flat.output] {
        Node::Readout { input, .. } => *input,
        _ => flat.output,
    };
    let operator = match &flat.nodes[logits] {
        Node::Transposed { operator, .. } => *operator,
        Node::Affine { terms, bias: None } if terms.len() == 1 => terms[0].1,
        _ => return Err("the output is not a bias-free linear head".into()),
    };
    let matrix = flat.operators[operator].matrix();
    match (matrix.nrows() == width, matrix.ncols() == width) {
        (false, true) => Ok(matrix),
        (true, false) => Ok(matrix.t().to_owned()),
        _ => Err("the head's orientation is ambiguous or its width is not the hidden width".into()),
    }
}

/// The gradient of `Σ_t log P(y_t)` in `P`'s trainable operators on the sequences `tokens` run by
/// `P` alone, each `y_t` drawn from `P`'s own next-token distribution at position `t`: its square,
/// entry by entry, is an unbiased estimate of the diagonal of the Gauss–Newton matrix
/// `Σ_t J_tᵀ F_t J_t` (`F_t` the Fisher matrix of `P`'s softmax at `t`).
fn sampled_label_gradient(ic: &Interchange, head: &Tensor, tokens: &[&[u32]], rng: &mut StdRng) -> Result<BTreeMap<usize, Tensor>, String> {
    let (_, p) = ic.models();
    let (d, width, arithmetic) = (BlockEngine::device(&p), BlockEngine::width(&p), BlockEngine::arithmetic(&p));
    let length = tokens.first().map_or(0, |t| t.len());
    let ranges: Vec<Range<usize>> = (0..tokens.len()).map(|i| i * length..(i + 1) * length).collect();
    let rows = tokens.len() * length;
    let mut stream = d.zeros(rows, width).map_err(error)?;
    let mut tapes = Vec::with_capacity(BlockEngine::blocks(&p));
    for b in 0..BlockEngine::blocks(&p) {
        tapes.push(p.forward(b, &mut stream, &ranges, tokens, None, true)?.ok_or("the block kept no tape")?);
    }
    // The cotangent of the final normed stream, a tile of rows at a time: the logits' sampled
    // cotangent pulled back through the head.
    let tile = (1 << 28) / (4 * head.rows()).max(1);
    let mut cotangent = d.zeros(rows, width).map_err(error)?;
    for start in (0..rows).step_by(tile.max(1)) {
        let n = tile.max(1).min(rows - start);
        let h = d.rows_of(&stream, start, n).map_err(error)?;
        let mut logits = d.zeros(n, head.rows()).map_err(error)?;
        d.gemm(&mut logits, 1.0, &h, Op::N, head, Op::T, 0.0, arithmetic).map_err(error)?;
        let uniforms = d.upload_vec(n, 1, (0..n).map(|_| rng.random::<f64>()).collect()).map_err(error)?;
        d.sampled_cotangent(&mut logits, &uniforms, None).map_err(error)?;
        let mut seed = d.zeros(n, width).map_err(error)?;
        d.gemm(&mut seed, 1.0, &logits, Op::N, head, Op::N, 0.0, arithmetic).map_err(error)?;
        d.set_rows(&mut cotangent, start, &seed).map_err(error)?;
    }
    let mut gradient = BTreeMap::new();
    for b in (0..tapes.len()).rev() {
        let tape = tapes.pop().ok_or("a block without its tape")?;
        p.reverse(b, tape, &mut cotangent, &ranges, None, &mut gradient)?;
    }
    Ok(gradient)
}

/// Sums over draws for one operator.
struct Totals {
    full: f64,
    local: [(f64, f64); 2],
    /// Per entry, the sums of each curvature estimate and of its square: Stein's, the antithetic
    /// one, and the sampled-label Gauss–Newton diagonal.
    sums: [Array2<f64>; 6],
}

fn dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    Zip::from(a).and(b).fold(0.0, |s, x, y| s + x * y)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, f64::total_cmp);
    Some(values[mid])
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, from, out, mode, draws] = &args[..] else {
        return Err(USAGE.into());
    };
    let (export, settings_path, out) = (Path::new(export), Path::new(settings_path), Path::new(out));
    let draws: usize = draws.parse().map_err(error)?;
    if draws < 2 {
        return Err("variances over draws need at least two draws".into());
    }
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    if first >= end || settings.training_sequences < 2 {
        return Err("held-out sequences must be a nonempty range, and a source needs another training sequence".into());
    }
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("a GPU required")?,
        _ => return Err(USAGE.into()),
    };
    let rows = end.max(settings.training_sequences + if settings.training_sequences > first { end - first } else { 0 });
    let imported = import_language_model(export, rows, settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
    let train: Vec<Vec<u32>> = sequences[..first].iter().chain(&sequences[end..]).take(settings.training_sequences).cloned().collect();
    if train.len() != settings.training_sequences {
        return Err("the export holds fewer training sequences than asked for".into());
    }
    let explanation = library_mdl::explanation(&native, &layers)?;
    let posterior = match from.split_once(':') {
        None if from == "native" => library_mdl::Posterior::new(&explanation, 2 * train.len() * settings.context)?,
        Some(("checkpoint", path)) => library_mdl::checkpoint_posterior(&explanation, Path::new(path))?,
        _ => return Err(USAGE.into()),
    };
    let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
    let reads = interchange::library_reads(&explanation.artifact.program, sites.len())?;
    let trainable = explanation.trainable.clone();
    let mut ic = Interchange::new(&device, &native, &sites, &explanation.artifact, &trainable, reads, settings.fit.numeric_bytes, settings.fit.head_tile_rows)?;
    let variables = ic.variables().to_vec();
    let (flat, _, _) = interchange::sites(&explanation.artifact, &sites)?;
    let head = {
        let (_, p) = ic.models();
        BlockEngine::device(&p).upload(head_matrix(&flat, BlockEngine::width(&p))?.view()).map_err(error)?
    };
    // The experiments' fixed directions are M's reads, the library's start.
    let starting: Vec<Array2<f64>> = trainable.iter().map(|op| explanation.artifact.program.operators[*op].matrix()).collect();
    let shapes: Vec<(usize, usize)> = starting.iter().map(Array2::dim).collect();
    let sd: Vec<Array2<f64>> = posterior.log_sd.iter().map(|s| s.mapv(f64::exp)).collect();
    let mut totals: Vec<Totals> = shapes
        .iter()
        .map(|dim| Totals { full: 0.0, local: [(0.0, 0.0); 2], sums: std::array::from_fn(|_| Array2::zeros(*dim)) })
        .collect();
    let (mut full_seconds, mut local_seconds) = (0.0, [0.0; 2]);
    let mut rng = StdRng::seed_from_u64(settings.fit.seed);
    for k in 0..draws {
        // One batch of distinct bases, each with a source among the other training sequences.
        let count = settings.fit.batch_sequences.min(train.len());
        let bases: Vec<usize> = interchange::hybrid_of(&mut rng, train.len(), count).iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i).collect();
        let sources: Vec<usize> = bases
            .iter()
            .map(|&i| {
                let j = rng.random_range(0..train.len() - 1);
                if j >= i { j + 1 } else { j }
            })
            .collect();
        let pick = |indices: &[usize]| indices.iter().map(|i| train[*i].clone()).collect::<Vec<_>>();
        let batch = Batch::new(pick(&bases), pick(&sources))?;
        let experiments = interchange::sample(&mut rng, bases.len(), &variables, 2 * layer_count, settings.context)?;
        let design = ic.design_at(&variables, &experiments, &starting)?;
        let targets = ic.targets(&batch, &experiments, &design)?;
        let noise: Vec<Array2<f64>> = shapes.iter().map(|dim| standard_normal(&mut rng, *dim)).collect();
        let sample = |sign: f64| -> Vec<Array2<f64>> {
            posterior.mean.iter().zip(&sd).zip(&noise).map(|((mu, s), e)| Zip::from(mu).and(s).and(e).map_collect(|m, s, e| m + sign * s * e)).collect()
        };
        let d = ic.models().1.program.device().clone();
        ic.load(&sample(1.0))?;
        let clock = Instant::now();
        let plus = downloaded(&d, &trainable, &shapes, &ic.evaluate_resident(&batch, &experiments, &design, &targets, true)?.gradient)?;
        full_seconds += clock.elapsed().as_secs_f64();
        let base_tokens: Vec<&[u32]> = batch.base.iter().map(Vec::as_slice).collect();
        let mut local = Vec::with_capacity(2);
        for (i, own) in [true, false].into_iter().enumerate() {
            let clock = Instant::now();
            local.push(downloaded(&d, &trainable, &shapes, &local_gradient(&ic, &base_tokens, own)?)?);
            local_seconds[i] += clock.elapsed().as_secs_f64();
        }
        let factor = downloaded(&d, &trainable, &shapes, &sampled_label_gradient(&ic, &head, &base_tokens, &mut rng)?)?;
        ic.load(&sample(-1.0))?;
        let minus = downloaded(&d, &trainable, &shapes, &ic.evaluate_resident(&batch, &experiments, &design, &targets, true)?.gradient)?;
        for (i, total) in totals.iter_mut().enumerate() {
            total.full += dot(&plus[i], &plus[i]);
            for (kind, c) in local.iter().enumerate() {
                total.local[kind].0 += dot(&plus[i], &c[i]);
                total.local[kind].1 += dot(&c[i], &c[i]);
            }
            let [stein, stein_sq, anti, anti_sq, newton, newton_sq] = &mut total.sums;
            for ((a, a2), u) in newton.iter_mut().zip(newton_sq.iter_mut()).zip(&factor[i]) {
                *a += u * u;
                *a2 += u.powi(4);
            }
            let sums = stein.iter_mut().zip(stein_sq.iter_mut()).zip(anti.iter_mut()).zip(anti_sq.iter_mut());
            let values = plus[i].iter().zip(&minus[i]).zip(noise[i].iter().zip(&sd[i]));
            for ((((a, a2), b), b2), ((gp, gm), (e, s))) in sums.zip(values) {
                if *s > 0.0 {
                    let plain = gp * e / s;
                    let antithetic = (gp - gm) * e / (2.0 * s);
                    *a += plain;
                    *a2 += plain * plain;
                    *b += antithetic;
                    *b2 += antithetic * antithetic;
                }
            }
        }
        log::info!("draw {}/{draws}", k + 1);
    }
    let n = draws as f64;
    let program = &explanation.artifact.program;
    let mut operators = Vec::with_capacity(trainable.len());
    let mut kinds: BTreeMap<&'static str, [f64; 3]> = BTreeMap::new();
    for ((op, total), s) in trainable.iter().zip(&totals).zip(&sd) {
        let name = &program.operators[*op].name;
        let kind = if name.contains(".mlp") { "mlp" } else { "attention" };
        // E‖g − β c‖² at the best β, per kind of local gradient, as a share of E‖g‖².
        let left: Vec<f64> = total.local.iter().map(|(cross, square)| if *square > 0.0 { total.full - cross * cross / square } else { total.full }).collect();
        let entry = kinds.entry(kind).or_insert([0.0; 3]);
        entry[0] += total.full;
        entry[1] += left[0];
        entry[2] += left[1];
        let [stein, stein_sq, anti, anti_sq, newton, newton_sq] = &total.sums;
        let (mut ratio, mut stein_snr, mut anti_snr, mut newton_snr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let variance = |sum: f64, square: f64| (square - sum * sum / n) / (n - 1.0);
        for ((((((a, a2), b), b2), c), c2), sd) in stein.iter().zip(stein_sq).zip(anti).zip(anti_sq).zip(newton).zip(newton_sq).zip(s) {
            if *sd <= 0.0 {
                continue;
            }
            let (va, vb, vc) = (variance(*a, *a2), variance(*b, *b2), variance(*c, *c2));
            if va > 0.0 && vb > 0.0 {
                ratio.push(va / vb);
                stein_snr.push((a / n).abs() / va.sqrt());
                anti_snr.push((b / n).abs() / vb.sqrt());
            }
            if vc > 0.0 {
                newton_snr.push((c / n) / vc.sqrt());
            }
        }
        operators.push(json!({
            "name": name,
            "entries": s.len(),
            "residual_own": if total.full > 0.0 { left[0] / total.full } else { f64::NAN },
            "residual_native": if total.full > 0.0 { left[1] / total.full } else { f64::NAN },
            "curvature_entries": ratio.len(),
            "median_variance_ratio": median(ratio),
            "median_snr_stein": median(stein_snr),
            "median_snr_antithetic": median(anti_snr),
            "median_snr_gauss_newton": median(newton_snr),
        }));
    }
    let by_kind: BTreeMap<&str, Value> = kinds
        .iter()
        .map(|(kind, [full, own, native])| (*kind, json!({"residual_own": own / full, "residual_native": native / full})))
        .collect();
    let (full, own, native) = kinds.values().fold((0.0, 0.0, 0.0), |acc, v| (acc.0 + v[0], acc.1 + v[1], acc.2 + v[2]));
    let report = json!({
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "from": from,
        "device": device.name(),
        "draws": draws,
        "seconds_per_draw": {"full_gradient": full_seconds / n, "local_own": local_seconds[0] / n, "local_native": local_seconds[1] / n},
        "residual": {"own": own / full, "native": native / full},
        "by_kind": by_kind,
        "operators": operators,
    });
    std::fs::create_dir_all(out).map_err(error)?;
    std::fs::write(out.join("REPORT.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    println!("{}", serde_json::to_string_pretty(&json!({"seconds_per_draw": report["seconds_per_draw"], "residual": report["residual"], "by_kind": report["by_kind"]})).map_err(error)?);
    Ok(())
}
