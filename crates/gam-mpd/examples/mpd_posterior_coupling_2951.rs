//! What the factorized posterior costs the code length on an MLP's groups (#2951).
//!
//! EXPORT SETTINGS.json native|checkpoint:PATH OUT host|gpu
//!
//! The library's posterior is a product of independent Gaussians, one per parameter. For a group
//! whose data term is locally `½ N δθᵀ H δθ` and whose prior variance is `v`, with `Λ = N H + I/v`,
//! the best factorized posterior costs `½ Σ_j log Λ_jj` nats beyond the best Gaussian with any
//! covariance, which costs `½ log det Λ`: the gap `½ (Σ_j log Λ_jj − log det Λ) ≥ 0` (Hadamard's
//! inequality) is what correlations inside the group would save. This measures it for the MLPs'
//! output vectors `u_i` and gate rows `g_i`, with `H` the Gauss–Newton matrix of `Σ KL(M ‖ P)` per
//! token in its Kronecker factorization:
//!
//! * `H(u_i) ≈ ā_i Ḡ`: `ā_i` the mean over tokens of the function's squared activation, `Ḡ` the mean
//!   of `δ δᵀ` with `δ` the cotangent of the MLP's output from the Fisher probe at `P`'s own
//!   prediction at every token (`Device::fisher_probe_cotangent`; its expectation over the probe's
//!   signs is `Σ_t J_tᵀ F_t J_t` at that output);
//! * `H(g_i) ≈ w̄_i A`: `A` the mean of `x xᵀ` over the MLP's inputs `x` (with a constant 1 where the
//!   gate has a bias), `w̄_i` the mean of `φ'(z_i)² (u_i · δ)²`.
//!
//! `P` runs alone at the posterior mean on the first two batches of training sequences. The last
//! layer's MLP feeds the final norm rather than the stream and is left out. Per layer and in total
//! the report gives, for each token count `N` of a list, the factorized and the correlated
//! precision cost `½ Σ log(v Λ_jj)` and `½ log det(v Λ)` in bits, and their difference.
use gam_gpu::{
    GpuPolicy,
    tensor::{Device, Op, Tensor},
};
use gam_linalg::{decompose::eigh, roundoff::SymmetricAssembly};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    interchange::{self, BlockEngine, Interchange},
    library_mdl::{self, Posterior},
    operator_program::{Node, OperatorProgram, SlotValues},
    run_check::{layer_nodes, split_sites},
};
use ndarray::{Array1, Array2, Axis};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, f64::consts::LN_2, ops::Range, path::Path};

const USAGE: &str = "EXPORT SETTINGS.json native|checkpoint:PATH OUT host|gpu";

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

fn operator(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    program.operators.iter().position(|o| o.name == name).ok_or_else(|| format!("no operator {name}"))
}

/// The node applying `op` as a term of a bias-free or biased affine node, and that term's input.
fn applying(program: &OperatorProgram, op: usize) -> Result<(usize, usize), String> {
    program
        .nodes
        .iter()
        .enumerate()
        .find_map(|(n, node)| match node {
            Node::Affine { terms, .. } => terms.iter().find(|(_, o)| *o == op).map(|(input, _)| (n, *input)),
            _ => None,
        })
        .ok_or_else(|| format!("no node applies {}", program.operators[op].name))
}

/// One MLP's nodes in the flat program and its operators' places among the trainable ones.
struct Mlp {
    /// The MLP's input `x`, its gate's pre-activation `z` and its law, and its activations `h`.
    input: usize,
    gate_node: usize,
    law: gam_mpd::operator_program::Law,
    activations: usize,
    /// The output operator (an index into `Explanation::trainable`), and whether the gate has a
    /// bias.
    output: usize,
    biased: bool,
}

/// Running sums over tokens for one MLP.
struct Sums {
    rows: f64,
    /// `Σ δ δᵀ`, `Σ x̃ x̃ᵀ`, and over the functions `Σ h hᵀ` and `Σ s sᵀ` with
    /// `s_i = φ'(z_i) (u_i · δ)`: their diagonals are each function's own factors.
    cotangent: Array2<f64>,
    input: Array2<f64>,
    activation: Array2<f64>,
    weight: Array2<f64>,
}

/// `½ (Σ_j log(v Λ_jj), log det(v Λ))` in nats for `Λ = c S + I/v`, `S` given by its diagonal and
/// eigenvalues.
fn costs(c: f64, v: f64, diagonal: &Array1<f64>, eigenvalues: &Array1<f64>) -> (f64, f64) {
    let factorized = diagonal.iter().map(|s| (1.0 + v * c * s.max(0.0)).ln()).sum::<f64>() / 2.0;
    let correlated = eigenvalues.iter().map(|s| (1.0 + v * c * s.max(0.0)).ln()).sum::<f64>() / 2.0;
    (factorized, correlated)
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, from, out, mode] = &args[..] else {
        return Err(USAGE.into());
    };
    let (export, settings_path, out) = (Path::new(export), Path::new(settings_path), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
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
    let explanation = library_mdl::explanation(&native, &layers)?;
    let posterior: Posterior = match from.split_once(':') {
        None if from == "native" => Posterior::new(&explanation, 2 * train.len() * settings.context)?,
        // The factorized posterior along the operators' own axes, whose coupling gap is measured.
        Some(("checkpoint", path)) => library_mdl::checkpoint_posterior(&explanation, Path::new(path))?,
        _ => return Err(USAGE.into()),
    };
    let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
    let reads = interchange::reads(&native, &sites)?;
    let mut ic = Interchange::new(&device, &native, &sites, &explanation.artifact, &explanation.trainable, reads, settings.fit.numeric_bytes, settings.fit.head_tile_rows)?;
    let (flat, _, _) = interchange::sites(&explanation.artifact, &sites)?;
    let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
    let mut mlps = Vec::new();
    for l in 0..layer_count - 1 {
        let (gate, out) = (operator(&flat, &format!("library.l{l}.mlp.gate"))?, operator(&flat, &format!("library.l{l}.mlp.out"))?);
        let (gate_node, input) = applying(&flat, gate)?;
        let (_, activations) = applying(&flat, out)?;
        let law = flat
            .nodes
            .iter()
            .find_map(|n| match n {
                Node::Pointwise { input, laws } if *input == gate_node => laws.first().copied(),
                _ => None,
            })
            .ok_or("no activation law on the gate")?;
        if activations != flat.nodes.iter().position(|n| matches!(n, Node::Pointwise { input, .. } if *input == gate_node)).ok_or("no activation node")? {
            return Err(format!("layer {l}: a gated MLP (its output reads more than the gate's law)"));
        }
        let biased = operator(&flat, &format!("library.l{l}.mlp.gate_bias")).is_ok();
        mlps.push(Mlp { input, gate_node, law, activations, output: position[&out], biased });
    }
    ic.load(&posterior.mean)?;
    let head = {
        let logits = match &flat.nodes[flat.output] {
            Node::Readout { input, .. } => *input,
            _ => flat.output,
        };
        let op = match &flat.nodes[logits] {
            Node::Transposed { operator, .. } => *operator,
            Node::Affine { terms, bias: None } if terms.len() == 1 => terms[0].1,
            _ => return Err("the output is not a bias-free linear head".into()),
        };
        let matrix = flat.operators[op].matrix();
        let width = BlockEngine::width(&ic.models().1);
        let e = if matrix.ncols() == width { matrix } else { matrix.t().to_owned() };
        device.upload(e.view()).map_err(error)?
    };
    let width = BlockEngine::width(&ic.models().1);
    let mut sums: Vec<Sums> = mlps
        .iter()
        .map(|m| {
            let units = posterior.mean[m.output].ncols();
            let inputs = width + usize::from(m.biased);
            Sums { rows: 0.0, cotangent: Array2::zeros((width, width)), input: Array2::zeros((inputs, inputs)), activation: Array2::zeros((units, units)), weight: Array2::zeros((units, units)) }
        })
        .collect();
    let mut rng = StdRng::seed_from_u64(settings.fit.seed);
    let batch = settings.fit.batch_sequences;
    for chunk in train.chunks(batch).take(2) {
        let tokens: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let (_, p) = ic.models();
        let d = BlockEngine::device(&p);
        let arithmetic = BlockEngine::arithmetic(&p);
        // The values: one forward pass of the whole program.
        let family = library_mdl::sequence_family(&tokens)?;
        let trace = p.program.forward(&family)?;
        // The cotangents at each MLP's output: P alone block by block, the Fisher probe at the head.
        let length = tokens[0].len();
        let ranges: Vec<Range<usize>> = (0..tokens.len()).map(|i| i * length..(i + 1) * length).collect();
        let total = tokens.len() * length;
        let mut stream = d.zeros(total, width).map_err(error)?;
        let mut tapes = Vec::new();
        for b in 0..BlockEngine::blocks(&p) {
            tapes.push(p.forward(b, &mut stream, &ranges, &tokens, None, true)?.ok_or("the block kept no tape")?);
        }
        let mut cotangent = d.zeros(total, width).map_err(error)?;
        let tile = (1 << 28) / (4 * head.rows()).max(1);
        let key = rng.random::<u64>();
        for start in (0..total).step_by(tile.max(1)) {
            let n = tile.max(1).min(total - start);
            let h = d.rows_of(&stream, start, n).map_err(error)?;
            let mut logits = d.zeros(n, head.rows()).map_err(error)?;
            d.gemm(&mut logits, 1.0, &h, Op::N, &head, Op::T, 0.0, arithmetic).map_err(error)?;
            d.softmax_rows(&mut logits, false).map_err(error)?;
            d.fisher_probe_cotangent(&mut logits, (key, start), None).map_err(error)?;
            let mut seed = d.zeros(n, width).map_err(error)?;
            d.gemm(&mut seed, 1.0, &logits, Op::N, &head, Op::N, 0.0, arithmetic).map_err(error)?;
            d.set_rows(&mut cotangent, start, &seed).map_err(error)?;
        }
        let mut gradient: BTreeMap<usize, Tensor> = BTreeMap::new();
        let mut at_outputs: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        for b in (0..tapes.len()).rev() {
            // Block 2l + 1 is layer l's MLP; the cotangent of the stream after it is its output's.
            if b % 2 == 1 && b / 2 < mlps.len() {
                at_outputs.insert(b / 2, d.download(&cotangent).map_err(error)?);
            }
            let tape = tapes.pop().ok_or("a block without its tape")?;
            p.reverse(b, &tape, &mut cotangent, &ranges, None, (&mut gradient, p.arithmetic()))?;
        }
        for (l, (m, s)) in mlps.iter().zip(&mut sums).enumerate() {
            let delta = &at_outputs[&l];
            let x = d.download(trace.value(m.input)?).map_err(error)?;
            let z = d.download(trace.value(m.gate_node)?).map_err(error)?;
            let h = d.download(trace.value(m.activations)?).map_err(error)?;
            let x = if m.biased { ndarray::concatenate(Axis(1), &[x.view(), Array2::ones((x.nrows(), 1)).view()]).map_err(error)? } else { x };
            s.cotangent += &delta.t().dot(delta);
            s.input += &x.t().dot(&x);
            s.activation += &h.t().dot(&h);
            // `s_i = φ'(z_i) (u_i · δ)` for every token and function: δ U with U the output map
            // (d × units).
            let weighted = &z.mapv(|t| m.law.derivative(t)) * &delta.dot(&*posterior.mean[m.output]);
            s.weight += &weighted.t().dot(&weighted);
            s.rows += delta.nrows() as f64;
        }
        log::info!("batch of {} sequences done", chunk.len());
    }
    // Each group's prior variance: the mean of μ² + σ² over its entries.
    let variance = |group: usize| -> f64 {
        let (mut n, mut total) = (0.0, 0.0);
        for cell in &explanation.groups[group].cells {
            let at = position[&cell.operator];
            for &r in &cell.rows {
                for c in cell.cols.clone() {
                    let (mu, s) = (posterior.mean[at][[r, c]], posterior.log_sd[at][[r, c]]);
                    total += mu * mu + (2.0 * s).exp();
                    n += 1.0;
                }
            }
        }
        total / n
    };
    let counts: Vec<f64> = [17, 20, 24, 27, 30].iter().map(|e| 2f64.powi(*e)).collect();
    let mut by_layer = Vec::new();
    let mut totals = vec![[0.0f64; 4]; counts.len()];
    let mirrored = |m: &Array2<f64>| Array2::from_shape_fn(m.dim(), |(i, j)| if i >= j { m[[i, j]] } else { m[[j, i]] });
    let spectrum = |m: &Array2<f64>| -> Result<Array1<f64>, String> { Ok(eigh(m.view(), SymmetricAssembly::Mirrored, None).map_err(error)?.values) };
    // Whole operators: `H ≈ W ⊗ A` for the gate map, `Aₕ ⊗ Ḡ` for the output map, at the mean prior
    // variance of the operator's groups; the gap now also counts the couplings between functions.
    let mut operators = vec![[0.0f64; 4]; counts.len()];
    for (l, s) in sums.iter().enumerate() {
        let (g, a) = (mirrored(&(&s.cotangent / s.rows)), mirrored(&(&s.input / s.rows)));
        let (h, w) = (mirrored(&(&s.activation / s.rows)), mirrored(&(&s.weight / s.rows)));
        let (ge, ae, he, we) = (spectrum(&g)?, spectrum(&a)?, spectrum(&h)?, spectrum(&w)?);
        let functions = &explanation.layers[l].functions;
        let live: Vec<usize> = (0..functions.len()).filter(|i| functions[*i].iter().all(|g| posterior.active[*g])).collect();
        let mean_variance = |part: usize| live.iter().map(|i| variance(if part == 0 { functions[*i][0] } else { *functions[*i].last().unwrap_or(&functions[*i][0]) })).sum::<f64>() / live.len().max(1) as f64;
        let (vg, vo) = (mean_variance(0), mean_variance(1));
        for (k, n) in counts.iter().enumerate() {
            let pair = |left: &Array1<f64>, right: &Array1<f64>, v: f64| -> f64 { left.iter().map(|x| right.iter().map(|y| (1.0 + v * n * (x * y).max(0.0)).ln()).sum::<f64>()).sum::<f64>() / 2.0 };
            let (hd, wd, gd, ad) = (h.diag().to_owned(), w.diag().to_owned(), g.diag().to_owned(), a.diag().to_owned());
            operators[k][0] += pair(&hd, &gd, vo);
            operators[k][1] += pair(&he, &ge, vo);
            operators[k][2] += pair(&wd, &ad, vg);
            operators[k][3] += pair(&we, &ae, vg);
        }
    }
    for (l, s) in sums.iter().enumerate() {
        let g = mirrored(&(&s.cotangent / s.rows));
        let a = mirrored(&(&s.input / s.rows));
        let (ge, ae) = (spectrum(&g)?, spectrum(&a)?);
        let (gd, ad) = (g.diag().to_owned(), a.diag().to_owned());
        let functions = &explanation.layers[l].functions;
        let mut rows = Vec::new();
        for (k, n) in counts.iter().enumerate() {
            let mut layer = [0.0f64; 4];
            for (i, groups) in functions.iter().enumerate() {
                if !groups.iter().all(|g| posterior.active[*g]) {
                    continue;
                }
                let (gate_group, output_group) = (groups[0], *groups.last().ok_or("a function without groups")?);
                let (f, c) = costs(n * s.activation[[i, i]] / s.rows, variance(output_group), &gd, &ge);
                layer[0] += f;
                layer[1] += c;
                let (f, c) = costs(n * s.weight[[i, i]] / s.rows, variance(gate_group), &ad, &ae);
                layer[2] += f;
                layer[3] += c;
            }
            for (t, v) in totals[k].iter_mut().zip(layer) {
                *t += v;
            }
            let bits = |v: f64| v / LN_2;
            rows.push(json!({
                "tokens": n,
                "outputs": {"factorized_bits": bits(layer[0]), "correlated_bits": bits(layer[1]), "gap_bits": bits(layer[0] - layer[1])},
                "gates": {"factorized_bits": bits(layer[2]), "correlated_bits": bits(layer[3]), "gap_bits": bits(layer[2] - layer[3])},
            }));
        }
        let condition = |e: &Array1<f64>| e.iter().cloned().fold(f64::MIN, f64::max) / e.iter().cloned().filter(|v| *v > 0.0).fold(f64::MAX, f64::min);
        by_layer.push(json!({"layer": l, "cotangent_condition": condition(&ge), "input_condition": condition(&ae), "by_tokens": rows}));
    }
    let summary: Vec<Value> = counts
        .iter()
        .zip(&totals)
        .zip(&operators)
        .map(|((n, t), o)| {
            let bits = |v: f64| v / LN_2;
            json!({
                "tokens": n,
                "within_groups": {"factorized_bits": bits(t[0] + t[2]), "correlated_bits": bits(t[1] + t[3]), "gap_bits": bits(t[0] + t[2] - t[1] - t[3]), "outputs_gap_bits": bits(t[0] - t[1]), "gates_gap_bits": bits(t[2] - t[3])},
                "whole_operators": {"factorized_bits": bits(o[0] + o[2]), "correlated_bits": bits(o[1] + o[3]), "gap_bits": bits(o[0] + o[2] - o[1] - o[3]), "outputs_gap_bits": bits(o[0] - o[1]), "gates_gap_bits": bits(o[2] - o[3])},
            })
        })
        .collect();
    let report = json!({"export_sha256": settings.export_sha256, "from": from, "device": device.name(), "summary": summary, "layers": by_layer});
    std::fs::create_dir_all(out).map_err(error)?;
    std::fs::write(out.join("REPORT.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    println!("{}", serde_json::to_string_pretty(&report["summary"]).map_err(error)?);
    Ok(())
}
