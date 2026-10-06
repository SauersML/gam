//! Ground-truth regression gate for the library's code length `F` (#2951) on toy models whose
//! mechanisms are known (APD §3, SPD §3): compressed computation (100 features through a 50-neuron
//! MLP in a 1000-wide residual stream, whose output map is one shared rank-50 map), and toy models
//! of superposition (TMS 40→10, and 5→2 with an identity inserted in the hidden layer). Each target
//! `M` is trained here; toys only gate the method, they are never results.
//!
//! An account of `M` parameterizes `M`'s weights by blocks `θ` (a weight is a block, the product
//! `θ_i θ_jᵀ` of two blocks, or a block times fixed native reads) and partitions the blocks'
//! entries into prior groups. Its code length is `library_mdl`'s, scored by its own posterior
//! ([`library_mdl::Posterior::from_parts`], `description`) and fitted by its own IVON step
//! ([`DevicePosterior`] on the host) with the Gauss–Newton curvature from a sampled-label factor:
//!
//! `F = E_q[D(θ)] + Σ_G KL(q_G ‖ p_G) + Σ_G (½ ln |G| + L_scale) + L_subset`.
//!
//! The toys' outputs are real vectors, so the divergence per token is between Gaussian predictive
//! distributions of one variance `s²` centered on `M`'s and the explanation's outputs,
//! `‖y_M − y_P‖² / (2 s²)`, `s²` being `M`'s own mean squared error per output against its task's
//! labels (the predictive variance of a regression trained by squared error); the explanation is
//! scored against `M`'s outputs, never the labels. A label drawn from the explanation's own
//! prediction is `y_P + s ξ`, so the Gauss–Newton factor of a batch is the reverse pass of the
//! cotangent `−ξ / s`. As in `library_mdl::fit`, the `N` training tokens are fixed batches visited
//! once per epoch, one weight sample per batch, until an epoch's paired improvement of the
//! per-batch estimate of `F` is below its standard error.
//!
//! `mpd_toy_gate_2951 [compressed|tms] [LOG2_N…]` (both toys at `N = 2^14` by default).
use gam_gpu::tensor::Device;
use gam_linalg::decompose::solve;
use gam_mpd::{
    device_posterior::{DevicePosterior, Ivon, Parts},
    library_mdl::Posterior,
};
use ndarray::{Array2, Axis, Zip};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{collections::BTreeMap, f64::consts::LN_2};

/// Tokens per batch.
const BATCH: usize = 1024;

fn normal(rng: &mut StdRng, dim: (usize, usize)) -> Array2<f64> {
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
    Array2::from_shape_vec(dim, values).expect("shape")
}

/// `t` inputs (columns) of `n` sparse features: each active with probability `p`, uniform on
/// `[low, 1]` when active.
fn features(rng: &mut StdRng, n: usize, t: usize, p: f64, low: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, t), |_| if rng.random::<f64>() < p { low + (1.0 - low) * rng.random::<f64>() } else { 0.0 })
}

fn relu(v: f64) -> f64 {
    v.max(0.0)
}

// ------------------------------------------------------------------------------------- targets

/// A toy network as a function of its weights.
trait Target {
    /// The outputs (outputs × tokens) and what the reverse pass reads.
    fn forward(&self, w: &[Array2<f64>], x: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>);
    /// The derivatives in `w` of `Σ_t δ_t · y_t` for the output cotangents `delta`.
    fn backward(&self, w: &[Array2<f64>], x: &Array2<f64>, cache: &[Array2<f64>], delta: &Array2<f64>) -> Vec<Array2<f64>>;
}

/// TMS: `y = ReLU(Wᵀ I W x + b)`; weights `[W, b]`, or `[W, I, b]` with the identity inserted.
struct Tms {
    identity: bool,
}

impl Target for Tms {
    fn forward(&self, w: &[Array2<f64>], x: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>) {
        let h = w[0].dot(x);
        let h2 = if self.identity { w[1].dot(&h) } else { h.clone() };
        let pre = w[0].t().dot(&h2) + &w[w.len() - 1];
        (pre.mapv(relu), vec![h, h2, pre])
    }

    fn backward(&self, w: &[Array2<f64>], x: &Array2<f64>, cache: &[Array2<f64>], delta: &Array2<f64>) -> Vec<Array2<f64>> {
        let (h, h2, pre) = (&cache[0], &cache[1], &cache[2]);
        let mut dpre = delta.clone();
        Zip::from(&mut dpre).and(pre).for_each(|g, p| {
            if *p <= 0.0 {
                *g = 0.0;
            }
        });
        let dh2 = w[0].dot(&dpre);
        let mut dw = h2.dot(&dpre.t());
        let mut grads = Vec::with_capacity(w.len());
        if self.identity {
            dw += &w[1].t().dot(&dh2).dot(&x.t());
            grads.push(dw);
            grads.push(dh2.dot(&h.t()));
        } else {
            dw += &dh2.dot(&x.t());
            grads.push(dw);
        }
        grads.push(dpre.sum_axis(Axis(1)).insert_axis(Axis(1)));
        grads
    }
}

/// Compressed computation: `y = Eᵀ (E x + W_out ReLU(W_in E x))`, `E` the fixed embedding;
/// weights `[W_in, W_out]`.
struct Compressed {
    e: Array2<f64>,
    gram: Array2<f64>,
}

impl Target for Compressed {
    fn forward(&self, w: &[Array2<f64>], x: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>) {
        let pre = w[0].dot(&self.e).dot(x);
        let y = self.gram.dot(x) + self.e.t().dot(&w[1]).dot(&pre.mapv(relu));
        (y, vec![pre])
    }

    fn backward(&self, w: &[Array2<f64>], x: &Array2<f64>, cache: &[Array2<f64>], delta: &Array2<f64>) -> Vec<Array2<f64>> {
        let (pre, read) = (&cache[0], self.e.t().dot(&w[1]));
        let dread = delta.dot(&pre.mapv(relu).t());
        let mut dpre = read.t().dot(delta);
        Zip::from(&mut dpre).and(pre).for_each(|g, p| {
            if *p <= 0.0 {
                *g = 0.0;
            }
        });
        vec![dpre.dot(&x.t()).dot(&self.e.t()), self.e.dot(&dread)]
    }
}

/// Train `M` by Adam on the squared error against `labels(x)`, the rate decaying linearly to zero.
fn train(target: &dyn Target, w: &mut [Array2<f64>], sample: &dyn Fn(&mut StdRng) -> Array2<f64>, labels: &dyn Fn(&Array2<f64>) -> Array2<f64>, steps: usize, rate: f64, seed: u64) {
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m: Vec<Array2<f64>> = w.iter().map(|a| Array2::zeros(a.dim())).collect();
    let mut v = m.clone();
    for step in 1..=steps {
        let x = sample(&mut rng);
        let (y, cache) = target.forward(w, &x);
        let grads = target.backward(w, &x, &cache, &((y - labels(&x)) / x.ncols() as f64));
        let lr = rate * (1.0 - (step - 1) as f64 / steps as f64);
        let (c1, c2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
        for k in 0..w.len() {
            Zip::from(&mut w[k]).and(&grads[k]).and(&mut m[k]).and(&mut v[k]).for_each(|w, g, m, v| {
                *m = b1 * *m + (1.0 - b1) * g;
                *v = b2 * *v + (1.0 - b2) * g * g;
                *w -= lr * (*m / c1) / ((*v / c2).sqrt() + eps);
            });
        }
    }
}

/// `s²`: `M`'s mean squared error per output against its labels.
fn predictive_variance(target: &dyn Target, w: &[Array2<f64>], sample: &dyn Fn(&mut StdRng) -> Array2<f64>, labels: &dyn Fn(&Array2<f64>) -> Array2<f64>, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut sum, mut count) = (0.0, 0.0);
    for _ in 0..16 {
        let x = sample(&mut rng);
        let r = target.forward(w, &x).0 - labels(&x);
        sum += r.iter().map(|v| v * v).sum::<f64>();
        count += r.len() as f64;
    }
    sum / count
}

// ------------------------------------------------------------------------------------- accounts

/// How one of `M`'s weights is formed from the account's blocks.
enum Form {
    Block(usize),
    /// `θ_i θ_jᵀ`.
    Product(usize, usize),
    /// `θ_i Rᵀ` with native reads `R`.
    Reads(usize, Array2<f64>),
}

struct Account {
    name: &'static str,
    forms: Vec<Form>,
    start: Vec<Array2<f64>>,
    membership: Vec<Array2<u32>>,
    groups: usize,
}

impl Account {
    fn assemble(&self, theta: &[Array2<f64>]) -> Vec<Array2<f64>> {
        self.forms
            .iter()
            .map(|f| match f {
                Form::Block(i) => theta[*i].clone(),
                Form::Product(i, j) => theta[*i].dot(&theta[*j].t()),
                Form::Reads(i, r) => theta[*i].dot(&r.t()),
            })
            .collect()
    }

    fn pullback(&self, theta: &[Array2<f64>], dw: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let mut out: Vec<Array2<f64>> = theta.iter().map(|t| Array2::zeros(t.dim())).collect();
        for (f, g) in self.forms.iter().zip(dw) {
            match f {
                Form::Block(i) => out[*i] += g,
                Form::Product(i, j) => {
                    out[*i] += &g.dot(&theta[*j]);
                    out[*j] += &g.t().dot(&theta[*i]);
                }
                Form::Reads(i, r) => out[*i] += &g.dot(r),
            }
        }
        out
    }
}

/// Group ids: the whole block one group.
fn whole(dim: (usize, usize), g: u32) -> Array2<u32> {
    Array2::from_elem(dim, g)
}

/// Group ids: one group per column, from `g`.
fn columns(dim: (usize, usize), g: u32) -> Array2<u32> {
    Array2::from_shape_fn(dim, |(_, c)| g + c as u32)
}

/// Group ids: one group per row, from `g`.
fn rows(dim: (usize, usize), g: u32) -> Array2<u32> {
    Array2::from_shape_fn(dim, |(r, _)| g + r as u32)
}

/// Per group of `groups`, the reference variance its scale is sent against, by
/// `library_mdl::mean_squares`'s rule on the account's start `start` (made from `M`, so `M` fixes
/// it): the mean square of its starting values; for a group starting at zero, the mean square of
/// its blocks' entries in the groups that do not, and `1` where those are all zero too.
fn references(start: &[Array2<f64>], membership: &[Array2<u32>], groups: usize) -> Vec<f64> {
    let mut own = vec![(0.0, 0.0); groups];
    for (values, ids) in start.iter().zip(membership) {
        for (value, g) in values.iter().zip(ids) {
            let entry = &mut own[*g as usize];
            entry.0 += 1.0;
            entry.1 += value * value;
        }
    }
    // Per block, the count and sum of squares of its entries in groups not all zero.
    let blocks: Vec<(f64, f64)> = start
        .iter()
        .zip(membership)
        .map(|(values, ids)| values.iter().zip(ids).filter(|(_, g)| own[**g as usize].1 > 0.0).fold((0.0, 0.0), |(count, sum), (value, _)| (count + 1.0, sum + value * value)))
        .collect();
    (0..groups)
        .map(|g| {
            let (count, sum) = own[g];
            if sum > 0.0 {
                return sum / count;
            }
            let (count, sum) = membership.iter().zip(&blocks).filter(|(ids, _)| ids.iter().any(|h| *h as usize == g)).fold((0.0, 0.0), |(n, total), (_, (c, s))| (n + *c, total + *s));
            if sum > 0.0 { sum / count } else { 1.0 }
        })
        .collect()
}

// ------------------------------------------------------------------------------------- the fit

/// A converged fit: `F` and its parts in bits, and the epochs taken.
struct Fitted {
    objective: f64,
    data: f64,
    description: f64,
    epochs: usize,
}

/// The weight noise's key for batch `b` of `epoch`.
fn key(seed: u64, epoch: usize, b: usize) -> u64 {
    seed ^ (epoch as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (b as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

/// Fit `account` of `M` (`target` at weights `native`) to the training `batches` of inputs by
/// `library_mdl`'s posterior and IVON step, to the paired-epoch convergence of its estimate of `F`.
fn fit(target: &dyn Target, native: &[Array2<f64>], account: &Account, batches: &[Array2<f64>], s2: f64, seed: u64) -> Result<Fitted, String> {
    let host = Device::host();
    let tokens = batches.len() * BATCH;
    let outputs: Vec<Array2<f64>> = batches.iter().map(|x| target.forward(native, x).0).collect();
    let reference = references(&account.start, &account.membership, account.groups);
    let mut posterior = Posterior::from_parts(account.start.clone(), account.membership.clone(), reference, tokens)?;
    let operators: Vec<usize> = (0..account.start.len()).collect();
    let groups: Vec<Vec<u32>> = account.membership.iter().map(|m| m.iter().copied().collect()).collect();
    let parts = Parts { operators: &operators, mean: &posterior.mean, log_sd: &posterior.log_sd, groups: &groups, count: account.groups, reference: Some(posterior.references()) };
    let mut device = DevicePosterior::from_parts(&host, &parts, tokens as f64, None, 0)?;
    let count = batches.len() as f64;
    let ivon = Ivon { beta1: 0.9, beta2: 1.0 - 1.0 / count };
    let upload = |a: &Array2<f64>| host.upload(a.view()).map_err(|e| e.to_string());
    let mut previous: Option<Vec<f64>> = None;
    let mut epoch = 0;
    loop {
        let (mut estimates, mut data_sum, mut description_sum) = (Vec::with_capacity(batches.len()), 0.0, 0.0);
        for (b, (x, y)) in batches.iter().zip(&outputs).enumerate() {
            let key = key(seed, epoch, b);
            // The sample `θ = μ + σ ε` of the posterior as it stands, and its description.
            let mut theta = Vec::with_capacity(operators.len());
            for i in &operators {
                let (mean, log_sd) = device.values(*i)?;
                let mut sample = host.zeros(mean.nrows(), mean.ncols()).map_err(|e| e.to_string())?;
                host.reparameterize(&mut sample, (&upload(&mean)?, &upload(&log_sd)?), (key, *i as u64)).map_err(|e| e.to_string())?;
                theta.push(host.download(&sample).map_err(|e| e.to_string())?);
                posterior.mean[*i] = mean.into();
                posterior.log_sd[*i] = log_sd.into();
            }
            let description = posterior.description();
            let w = account.assemble(&theta);
            let (out, cache) = target.forward(&w, x);
            let dim = out.dim();
            let residual = out - y;
            let data = residual.iter().map(|r| r * r).sum::<f64>() / (2.0 * s2);
            let gradient = account.pullback(&theta, &target.backward(&w, x, &cache, &(residual / s2)));
            // The Gauss–Newton factor: labels `y_P + s ξ` from the explanation's own prediction.
            let xi = normal(&mut StdRng::seed_from_u64(key ^ 0x5851_F42D_4C95_7F2D), dim);
            let factor = account.pullback(&theta, &target.backward(&w, x, &cache, &(xi * (-1.0 / s2.sqrt()))));
            estimates.push(count * data + description);
            data_sum += count * data;
            description_sum += description;
            let gradient: BTreeMap<usize, _> = operators.iter().map(|i| Ok((*i, upload(&gradient[*i])?))).collect::<Result<_, String>>()?;
            let factor: BTreeMap<usize, _> = operators.iter().map(|i| Ok((*i, upload(&factor[*i])?))).collect::<Result<_, String>>()?;
            device.step(&gradient, 1.0 / BATCH as f64, (&factor, 1.0 / BATCH as f64), &BTreeMap::new(), &ivon)?;
        }
        let converged = previous.as_ref().is_some_and(|before| {
            let differences: Vec<f64> = before.iter().zip(&estimates).map(|(a, b)| a - b).collect();
            let mean = differences.iter().sum::<f64>() / count;
            let variance = differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (count - 1.0);
            mean < (variance / count).sqrt()
        });
        if converged {
            return Ok(Fitted {
                objective: estimates.iter().sum::<f64>() / count / LN_2,
                data: data_sum / count / LN_2,
                description: description_sum / count / LN_2,
                epochs: epoch + 1,
            });
        }
        previous = Some(estimates);
        epoch += 1;
    }
}

fn report(target: &dyn Target, native: &[Array2<f64>], account: &Account, batches: &[Array2<f64>], s2: f64) -> Result<f64, String> {
    let started = std::time::Instant::now();
    let fitted = fit(target, native, account, batches, s2, 11)?;
    eprintln!(
        "  {:<34} N=2^{:<2} F {:>10.1} bits  data {:>9.1}  description {:>10.1}  groups {:>3}  epochs {:>2}  {:.0} s",
        account.name,
        ((batches.len() * BATCH) as f64).log2().round(),
        fitted.objective,
        fitted.data,
        fitted.description,
        account.groups,
        fitted.epochs,
        started.elapsed().as_secs_f64()
    );
    Ok(fitted.objective)
}

// ------------------------------------------------------------------------------------- the toys

/// Compressed computation (SPD §3.3): which account of `W_in` and `W_out` has the shortest `F`.
fn compressed(scales: &[i32]) -> Result<(), String> {
    let (n, k, d, p) = (100usize, 50usize, 1000usize, 0.01);
    let mut rng = StdRng::seed_from_u64(5);
    let mut e = normal(&mut rng, (d, n));
    for mut c in e.columns_mut() {
        let norm = c.dot(&c).sqrt();
        c /= norm;
    }
    let gram = e.t().dot(&e);
    let sample = move |rng: &mut StdRng| features(rng, n, BATCH, p, -1.0);
    let labels = |x: &Array2<f64>| x + &x.mapv(relu);
    let target = Compressed { e: e.clone(), gram: gram.clone() };
    let mut w = vec![normal(&mut rng, (k, d)) * (1.0 / (d as f64).sqrt()), normal(&mut rng, (d, k)) * (1.0 / (k as f64).sqrt())];
    train(&target, &mut w, &sample, &labels, 4000, 3e-3, 6);
    // The dual reads `D = E (EᵀE)⁻¹`: `Dᵀ E = I`; `W_in` keeps only what the data see, `W_in E Dᵀ`.
    let dual = solve(gram.view(), e.t()).map_err(|e| e.to_string())?.t().to_owned();
    let a = w[0].dot(&e);
    w[0] = a.dot(&dual.t());
    let s2 = predictive_variance(&target, &w, &sample, &labels, 8);
    let probe = Array2::from_shape_fn((n, 1), |(i, _)| if i == 0 { 1.0 } else { 0.0 });
    eprintln!("compressed computation: s² {s2:.3e}, M(e_0)_0 = {:.3} (label 2)", target.forward(&w, &probe).0[[0, 0]]);
    // APD's split of W_out: `W_out = O Bᵀ`, `B = W_in E` (feature c's neuron pattern), `O = W_out (B Bᵀ)⁻¹ B`.
    let split = solve(a.dot(&a.t()).view(), a.view()).map_err(|e| e.to_string())?;
    let o = w[1].dot(&split);
    let (nu, ku) = (n as u32, k as u32);
    let accounts = [
        Account { name: "one W_in, one W_out", forms: vec![Form::Block(0), Form::Block(1)], start: w.clone(), membership: vec![whole((k, d), 0), whole((d, k), 1)], groups: 2 },
        Account { name: "neurons (library family)", forms: vec![Form::Block(0), Form::Block(1)], start: w.clone(), membership: vec![rows((k, d), 0), columns((d, k), ku)], groups: 2 * k },
        Account {
            name: "SPD: W_in per feature, one W_out",
            forms: vec![Form::Product(0, 1), Form::Block(2)],
            start: vec![a.clone(), dual.clone(), w[1].clone()],
            membership: vec![columns((k, n), 0), columns((d, n), nu), whole((d, k), 2 * nu)],
            groups: 2 * n + 1,
        },
        Account {
            name: "APD: W_out split per feature too",
            forms: vec![Form::Product(0, 1), Form::Product(3, 2)],
            start: vec![a.clone(), dual.clone(), a.clone(), o],
            membership: vec![columns((k, n), 0), columns((d, n), nu), columns((k, n), 2 * nu), columns((d, n), 3 * nu)],
            groups: 4 * n,
        },
        Account {
            name: "native reads: W_in per feature",
            forms: vec![Form::Reads(0, dual.clone()), Form::Block(1)],
            start: vec![a.clone(), w[1].clone()],
            membership: vec![columns((k, n), 0), whole((d, k), nu)],
            groups: n + 1,
        },
        Account {
            name: "native reads: one W_in",
            forms: vec![Form::Reads(0, dual.clone()), Form::Block(1)],
            start: vec![a, w[1].clone()],
            membership: vec![whole((k, n), 0), whole((d, k), 1)],
            groups: 2,
        },
    ];
    for &scale in scales {
        let mut rng = StdRng::seed_from_u64(9);
        let batches: Vec<Array2<f64>> = (0..(1usize << scale) / BATCH).map(|_| sample(&mut rng)).collect();
        for account in &accounts {
            report(&target, &w, account, &batches, s2)?;
        }
    }
    Ok(())
}

/// TMS: features as columns of `W` against `W` as one group (40→10), and the inserted identity as
/// one map against `m` rank-one maps (5→2).
fn tms(scales: &[i32]) -> Result<(), String> {
    for (n, m) in [(40usize, 10usize), (5, 2)] {
        let sample = move |rng: &mut StdRng| features(rng, n, BATCH, 0.05, 0.0);
        let labels = |x: &Array2<f64>| x.clone();
        let mut rng = StdRng::seed_from_u64(3);
        let mut w = vec![normal(&mut rng, (m, n)) * (1.0 / (m as f64).sqrt()), Array2::zeros((n, 1))];
        train(&Tms { identity: false }, &mut w, &sample, &labels, 6000, 1e-2, 4);
        let s2 = predictive_variance(&Tms { identity: false }, &w, &sample, &labels, 5);
        let norms: Vec<f64> = (0..n).map(|c| (w[0].column(c).dot(&w[0].column(c)).sqrt() * 100.0).round() / 100.0).collect();
        eprintln!("TMS {n}→{m}: s² {s2:.3e}, feature norms {norms:?}");
        let (nu, mu) = (n as u32, m as u32);
        let wi = vec![w[0].clone(), Array2::eye(m), w[1].clone()];
        let accounts = if n == 5 {
            vec![
                Account {
                    name: "identity one map",
                    forms: vec![Form::Block(0), Form::Block(1), Form::Block(2)],
                    start: wi.clone(),
                    membership: vec![columns((m, n), 0), whole((m, m), nu), whole((n, 1), nu + 1)],
                    groups: n + 2,
                },
                Account {
                    name: "identity as m rank-one maps",
                    forms: vec![Form::Block(0), Form::Product(1, 2), Form::Block(3)],
                    start: vec![w[0].clone(), Array2::eye(m), Array2::eye(m), w[1].clone()],
                    membership: vec![columns((m, n), 0), columns((m, m), nu), columns((m, m), nu + mu), whole((n, 1), nu + 2 * mu)],
                    groups: n + 2 * m + 1,
                },
            ]
        } else {
            vec![
                Account {
                    name: "per-feature columns of W",
                    forms: vec![Form::Block(0), Form::Block(1)],
                    start: w.clone(),
                    membership: vec![columns((m, n), 0), whole((n, 1), nu)],
                    groups: n + 1,
                },
                Account { name: "W one group", forms: vec![Form::Block(0), Form::Block(1)], start: w.clone(), membership: vec![whole((m, n), 0), whole((n, 1), 1)], groups: 2 },
            ]
        };
        let (target, native) = if n == 5 { (Tms { identity: true }, wi) } else { (Tms { identity: false }, w) };
        for &scale in scales {
            let mut rng = StdRng::seed_from_u64(9);
            let batches: Vec<Array2<f64>> = (0..(1usize << scale) / BATCH).map(|_| sample(&mut rng)).collect();
            for account in &accounts {
                report(&target, &native, account, &batches, s2)?;
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = args.first().map(String::as_str).filter(|a| a.parse::<i32>().is_err());
    let scales: Vec<i32> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let scales = if scales.is_empty() { vec![14] } else { scales };
    if scales.iter().any(|s| !(10..=24).contains(s)) {
        return Err("LOG2_N between 10 and 24 (whole batches of 1024 tokens)".into());
    }
    let started = std::time::Instant::now();
    if which.is_none_or(|w| w == "compressed") {
        compressed(&scales)?;
    }
    if which.is_none_or(|w| w == "tms") {
        tms(&scales)?;
    }
    eprintln!("{:.1} s", started.elapsed().as_secs_f64());
    Ok(())
}
