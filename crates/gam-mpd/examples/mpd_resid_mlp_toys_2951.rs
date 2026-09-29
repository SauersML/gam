//! The residual-MLP toys of APD/SPD (#2951) under the path code of `parameter_decomposition::mlp_paths`.
//!
//! Trains the three targets of Braun et al. 2025 / Bushnaq et al. 2025 with their recipe
//! (`param_decomp/experiments/resid_mlp/train_resid_mlp.py`): a fixed random unit-row embedding
//! `W_E` (`n × 1000`) with `W_U = W_Eᵀ`, `L` ReLU blocks without biases (1×50, 2×25, 3×17
//! neurons; 100, 100, 102 features), inputs `x_i ~ U(−1, 1)` independently with probability 0.01,
//! labels `x + relu(x)`, AdamW (weight decay 0.01) at 3e-3 with cosine decay to 0, batch 2048, for
//! 1000, 1000 and 10000 steps, `nn.Linear`'s default uniform initialisation. The forward and
//! backward run on the reassociated products (`W_E W_inᵀ`, `W_outᵀ W_inᵀ`, `W_outᵀ W_U`); the
//! gradient is the same function of the parameters.
//!
//! The behaviour is the target's output on `2^16` draws of its input law, a Gaussian with the
//! target's own mean squared error to its labels as the variance, explained once per training
//! draw (`steps × 2048`). Everything is reported on fresh draws: the faithfulness MSE to the
//! target, averaged per input over outputs and over inputs with an active feature (APD's
//! histogram), the scrubbed and anti-scrubbed MSE (half of the components ablated, sparing or
//! including the active ones), the active components per input, one-hot component counts at
//! `x_i = 0.75` (SPD's causal-importance probe) and each feature's neuron contributions against the
//! target's, `(W_U[i] W_out) ⊙ (W_in W_E[i])`.
//!
//! `cargo run --release -p gam-mpd --example mpd_resid_mlp_toys_2951 [layers...]`

use gam_mpd::mlp_paths::{
    GaussianBehaviour, MlpBlock, PathProgram, ResidualMlp, SparseRow, fit,
};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

const D_EMBED: usize = 1000;
const BATCH: usize = 2048;
const PROBABILITY: f64 = 0.01;
const LR: f64 = 3e-3;
const WEIGHT_DECAY: f64 = 0.01;
const FIT_ROWS: usize = 1 << 16;
const EVAL_ROWS: usize = 1 << 16;
const SCRUB_DRAWS: usize = 4;

struct Toy {
    layers: usize,
    features: usize,
    width: usize,
    steps: usize,
}

const TOYS: [Toy; 3] = [
    Toy { layers: 1, features: 100, width: 50, steps: 1000 },
    Toy { layers: 2, features: 100, width: 25, steps: 1000 },
    Toy { layers: 3, features: 102, width: 17, steps: 10000 },
];

fn normal(rng: &mut StdRng) -> f64 {
    let u: f64 = 1.0 - rng.random::<f64>();
    let v: f64 = rng.random::<f64>();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn uniform_matrix(rng: &mut StdRng, rows: usize, cols: usize, bound: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| bound * (2.0 * rng.random::<f64>() - 1.0))
}

fn draw(rng: &mut StdRng, rows: usize, n: usize) -> Vec<SparseRow> {
    (0..rows)
        .map(|_| {
            (0..n)
                .filter_map(|i| (rng.random::<f64>() < PROBABILITY).then(|| (i, 2.0 * rng.random::<f64>() - 1.0)))
                .filter(|(_, v)| *v != 0.0)
                .collect()
        })
        .collect()
}

fn dense(rows: &[SparseRow], n: usize) -> Array2<f64> {
    let mut x = Array2::zeros((rows.len(), n));
    for (r, row) in rows.iter().enumerate() {
        for &(i, v) in row {
            x[[r, i]] = v;
        }
    }
    x
}

fn labels(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v + v.max(0.0))
}

struct Adam {
    m: Array2<f64>,
    v: Array2<f64>,
}

impl Adam {
    fn new(shape: (usize, usize)) -> Self {
        Self { m: Array2::zeros(shape), v: Array2::zeros(shape) }
    }

    /// torch.optim.AdamW, betas (0.9, 0.999), eps 1e-8.
    fn step(&mut self, param: &mut Array2<f64>, grad: &Array2<f64>, lr: f64, t: i32) {
        param.mapv_inplace(|p| p * (1.0 - lr * WEIGHT_DECAY));
        self.m.zip_mut_with(grad, |m, g| *m = 0.9 * *m + 0.1 * g);
        self.v.zip_mut_with(grad, |v, g| *v = 0.999 * *v + 0.001 * g * g);
        let (c1, c2) = (1.0 - 0.9f64.powi(t), 1.0 - 0.999f64.powi(t));
        ndarray::Zip::from(param).and(&self.m).and(&self.v).for_each(|p, m, v| {
            *p -= lr * (m / c1) / ((v / c2).sqrt() + 1e-8);
        });
    }
}

/// Train one toy with the reference recipe.
fn train(toy: &Toy, rng: &mut StdRng) -> ResidualMlp {
    let n = toy.features;
    let mut embed = Array2::from_shape_fn((n, D_EMBED), |_| normal(rng));
    for mut row in embed.outer_iter_mut() {
        let norm = row.dot(&row).sqrt();
        row.mapv_inplace(|v| v / norm);
    }
    let unembed = embed.t().to_owned();
    let direct = embed.dot(&unembed);
    let mut w_in: Vec<Array2<f64>> =
        (0..toy.layers).map(|_| uniform_matrix(rng, toy.width, D_EMBED, 1.0 / (D_EMBED as f64).sqrt())).collect();
    let mut w_out: Vec<Array2<f64>> =
        (0..toy.layers).map(|_| uniform_matrix(rng, D_EMBED, toy.width, 1.0 / (toy.width as f64).sqrt())).collect();
    let mut adam_in: Vec<Adam> = (0..toy.layers).map(|_| Adam::new((toy.width, D_EMBED))).collect();
    let mut adam_out: Vec<Adam> = (0..toy.layers).map(|_| Adam::new((D_EMBED, toy.width))).collect();
    let layers = toy.layers;
    for step in 0..toy.steps {
        let progress = step as f64 / (toy.steps - 1) as f64;
        let lr = LR * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        let x = dense(&draw(rng, BATCH, n), n);
        let target = labels(&x);
        let p: Vec<Array2<f64>> = w_in.iter().map(|w| embed.dot(&w.t())).collect();
        let v: Vec<Array2<f64>> = w_out.iter().map(|w| w.t().dot(&unembed)).collect();
        let q = |k: usize, l: usize| w_out[k].t().dot(&w_in[l].t());
        let mut pre = Vec::with_capacity(layers);
        let mut act: Vec<Array2<f64>> = Vec::with_capacity(layers);
        for l in 0..layers {
            let mut h = x.dot(&p[l]);
            for k in 0..l {
                h += &act[k].dot(&q(k, l));
            }
            act.push(h.mapv(|t| t.max(0.0)));
            pre.push(h);
        }
        let mut y = x.dot(&direct);
        for k in 0..layers {
            y += &act[k].dot(&v[k]);
        }
        let dy = (y - &target) * (2.0 / (BATCH * n) as f64);
        let mut dh: Vec<Array2<f64>> = vec![Array2::zeros((0, 0)); layers];
        let mut grad_in: Vec<Array2<f64>> = w_in.iter().map(|w| Array2::zeros(w.dim())).collect();
        let mut grad_out: Vec<Array2<f64>> = w_out.iter().map(|w| Array2::zeros(w.dim())).collect();
        for k in (0..layers).rev() {
            let mut da = dy.dot(&v[k].t());
            for l in k + 1..layers {
                let qkl = q(k, l);
                da += &dh[l].dot(&qkl.t());
                let dq = act[k].t().dot(&dh[l]);
                grad_out[k] += &w_in[l].t().dot(&dq.t());
                grad_in[l] += &dq.t().dot(&w_out[k].t());
            }
            let dv = act[k].t().dot(&dy);
            grad_out[k] += &unembed.dot(&dv.t());
            let mut d = da;
            d.zip_mut_with(&pre[k], |g, h| {
                if *h <= 0.0 {
                    *g = 0.0;
                }
            });
            let dp = x.t().dot(&d);
            grad_in[k] += &dp.t().dot(&embed);
            dh[k] = d;
        }
        let t = step as i32 + 1;
        for l in 0..layers {
            adam_in[l].step(&mut w_in[l], &grad_in[l], lr, t);
            adam_out[l].step(&mut w_out[l], &grad_out[l], lr, t);
        }
    }
    let blocks = w_in.into_iter().zip(w_out).map(|(w_in, w_out)| MlpBlock { w_in, w_out }).collect();
    ResidualMlp { embed, unembed, blocks }
}

fn mean_row_mse(a: &Array2<f64>, b: &Array2<f64>, rows: &[usize]) -> f64 {
    let n = a.ncols() as f64;
    rows.iter().map(|&r| (&a.row(r) - &b.row(r)).mapv(|e| e * e).sum() / n).sum::<f64>() / rows.len() as f64
}

fn run(toy: &Toy, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let model = train(toy, &mut rng);
    let train_seconds = started.elapsed().as_secs_f64();
    let n = toy.features;

    // The behaviour: the target's residual variance on its task, the fit draws, n = training draws.
    let probe = draw(&mut rng, EVAL_ROWS, n);
    let probe_out = model.forward(&probe).unwrap();
    let variance = (&probe_out - &labels(&dense(&probe, n))).mapv(|e| e * e).mean().unwrap();
    let fit_inputs = draw(&mut rng, FIT_ROWS, n);
    let fit_outputs = model.forward(&fit_inputs).unwrap();
    let behaviour = GaussianBehaviour {
        inputs: fit_inputs,
        outputs: fit_outputs,
        variance,
        observations: (toy.steps * BATCH) as f64,
    };
    let reference = PathProgram::from_model(&model).unwrap();
    let eval = draw(&mut rng, EVAL_ROWS, n);
    let target = model.forward(&eval).unwrap();
    let rewrite_gap = (&reference.forward(&eval) - &target).iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    let started = Instant::now();
    let fitted = fit(&reference, &behaviour).unwrap();
    let fit_seconds = started.elapsed().as_secs_f64();
    let program = &fitted.program;

    // Faithfulness on fresh draws, over inputs with an active feature.
    let out = program.forward(&eval);
    let active_rows: Vec<usize> = (0..eval.len()).filter(|&r| !eval[r].is_empty()).collect();
    let mse = mean_row_mse(&out, &target, &active_rows);
    let label_mse_target = mean_row_mse(&target, &labels(&dense(&eval, n)), &active_rows);
    let label_mse_program = mean_row_mse(&out, &labels(&dense(&eval, n)), &active_rows);

    // Components: sources. Active = nonzero source with a present row.
    let sources = program.source_offset(program.blocks.len());
    let present: Vec<bool> = (0..sources)
        .map(|s| {
            let reads = program.blocks.iter().any(|b| s < b.read.nrows() && b.read.row(s).iter().any(|v| *v != 0.0));
            let write = program.source_neuron(s).is_some_and(|(l, j)| program.blocks[l].write.row(j).iter().any(|v| *v != 0.0));
            reads || write
        })
        .collect();
    let mut scrubbed = Array2::<f64>::zeros(out.dim());
    let mut anti = Array2::<f64>::zeros(out.dim());
    let (mut scrub_mse, mut anti_mse) = (0.0, 0.0);
    let (mut l0_inputs, mut l0_neurons) = (0.0, 0.0);
    for draw_index in 0..SCRUB_DRAWS {
        for &r in &active_rows {
            let trace = program.trace_row(&eval[r]);
            let active: Vec<bool> = (0..sources)
                .map(|s| {
                    present[s]
                        && match program.source_neuron(s) {
                            Some((l, j)) => trace.act[l][j] > 0.0,
                            None => eval[r].iter().any(|(i, _)| *i == s),
                        }
                })
                .collect();
            if draw_index == 0 {
                l0_inputs += active[..n].iter().filter(|a| **a).count() as f64;
                l0_neurons += active[n..].iter().filter(|a| **a).count() as f64;
            }
            // Half of the components ablated: sparing the active ones, or among all of them.
            let mut order: Vec<usize> = (0..sources).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.random_range(0..=i));
            }
            let mut spare = vec![false; sources];
            let inactive: Vec<usize> = order.iter().copied().filter(|&s| !active[s]).collect();
            for &s in inactive.iter().take(sources / 2) {
                spare[s] = true;
            }
            let mut all = vec![false; sources];
            for &s in order.iter().take(sources / 2) {
                all[s] = true;
            }
            scrubbed.row_mut(r).assign(&program.trace_row_ablated(&eval[r], Some(spare.as_slice())).out);
            anti.row_mut(r).assign(&program.trace_row_ablated(&eval[r], Some(all.as_slice())).out);
        }
        scrub_mse += mean_row_mse(&scrubbed, &target, &active_rows) / SCRUB_DRAWS as f64;
        anti_mse += mean_row_mse(&anti, &target, &active_rows) / SCRUB_DRAWS as f64;
    }
    let scrub_identical = active_rows.iter().all(|&r| scrubbed.row(r) == out.row(r));
    l0_inputs /= active_rows.len() as f64;
    l0_neurons /= active_rows.len() as f64;

    // One-hot probes at 0.75 and neuron contributions per feature.
    let mut onehot_in = vec![0.0; toy.layers];
    let mut onehot_out = vec![0.0; toy.layers];
    let (mut cos_sum, mut cos_min, mut ratio_sum, mut support) = (0.0, f64::INFINITY, 0.0, 0.0);
    for i in 0..n {
        let trace = program.trace_row(&[(i, 0.75)]);
        for l in 0..toy.layers {
            let offset = program.source_offset(l);
            let reads_input = program.blocks[l].read.row(i).iter().any(|v| *v != 0.0);
            let reads_neurons = (n..offset)
                .filter(|&s| {
                    let (k, j) = program.source_neuron(s).unwrap();
                    trace.act[k][j] > 0.0 && program.blocks[l].read.row(s).iter().any(|v| *v != 0.0)
                })
                .count();
            onehot_in[l] += (usize::from(reads_input) + reads_neurons) as f64 / n as f64;
            onehot_out[l] += (0..program.blocks[l].width())
                .filter(|&j| trace.act[l][j] > 0.0 && program.blocks[l].write.row(j).iter().any(|v| *v != 0.0))
                .count() as f64
                / n as f64;
        }
        let (mut dot, mut model_sq, mut ours_sq) = (0.0, 0.0, 0.0);
        for l in 0..toy.layers {
            for k in 0..program.blocks[l].width() {
                let model_nc = reference.blocks[l].read[[i, k]] * reference.blocks[l].write[[k, i]];
                let ours_nc = program.blocks[l].read[[i, k]] * program.blocks[l].write[[k, i]];
                dot += model_nc * ours_nc;
                model_sq += model_nc * model_nc;
                ours_sq += ours_nc * ours_nc;
                support += f64::from(u8::from(program.blocks[l].read[[i, k]] != 0.0)) / n as f64;
            }
        }
        let cos = dot / (model_sq * ours_sq).sqrt();
        cos_sum += cos / n as f64;
        cos_min = cos_min.min(cos);
        ratio_sum += (ours_sq / model_sq).sqrt() / n as f64;
    }

    let dense_reals = 2 * D_EMBED * toy.width * toy.layers;
    let rewrite_reals: usize = reference.blocks.iter().map(|b| b.read.len() + b.write.len()).sum();
    println!(
        "{{\"layers\":{},\"features\":{n},\"width\":{},\"train_s\":{train_seconds:.1},\"fit_s\":{fit_seconds:.1},\
\"variance\":{variance:.4e},\"observations\":{},\"rewrite_gap\":{rewrite_gap:.2e},\"dense_reals\":{dense_reals},\
\"rewrite_reals\":{rewrite_reals},\"reals\":{},\"code_bits\":{},\"data_bits\":{:.1},\"sweeps\":{:?},\
\"mse_to_target\":{mse:.3e},\"scrubbed_mse\":{scrub_mse:.3e},\"scrubbed_bit_identical\":{scrub_identical},\
\"anti_scrubbed_mse\":{anti_mse:.3e},\"label_mse_target\":{label_mse_target:.3e},\"label_mse_program\":{label_mse_program:.3e},\
\"l0_input_components\":{l0_inputs:.3},\"l0_neuron_components\":{l0_neurons:.3},\"onehot_win_components\":{onehot_in:?},\
\"onehot_wout_components\":{onehot_out:?},\"neurons_per_feature\":{support:.2},\"nc_cosine_mean\":{cos_sum:.5},\
\"nc_cosine_min\":{cos_min:.5},\"nc_l2_ratio_mean\":{ratio_sum:.4}}}",
        toy.layers,
        toy.width * toy.layers,
        behaviour.observations,
        program.real_count(),
        fitted.code_bits,
        fitted.data_bits,
        fitted.sweeps,
    );
}

fn main() {
    let chosen: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    for toy in TOYS.iter().filter(|t| chosen.is_empty() || chosen.contains(&t.layers)) {
        run(toy, toy.layers as u64);
    }
}
