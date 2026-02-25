use gam::estimate::{FitOptions, fit_gam, predict_gam};
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView1};
use std::time::Instant;

#[derive(Clone, Copy)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        (x as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn parse_arg(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_seed(args: &[String], default: u64) -> u64 {
    args.windows(2)
        .find(|w| w[0] == "--seed")
        .and_then(|w| w[1].parse::<u64>().ok())
        .unwrap_or(default)
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn auc_score(y: ArrayView1<'_, f64>, p: ArrayView1<'_, f64>) -> f64 {
    let mut pairs: Vec<(f64, f64)> = y.iter().copied().zip(p.iter().copied()).collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut rank_sum_pos = 0.0;
    let mut n_pos = 0.0;
    let mut n_neg = 0.0;
    for (idx, (yy, _)) in pairs.iter().enumerate() {
        if *yy > 0.5 {
            rank_sum_pos += (idx + 1) as f64;
            n_pos += 1.0;
        } else {
            n_neg += 1.0;
        }
    }
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

fn brier_score(y: ArrayView1<'_, f64>, p: ArrayView1<'_, f64>) -> f64 {
    let n = y.len() as f64;
    y.iter()
        .copied()
        .zip(p.iter().copied())
        .map(|(yy, pp)| {
            let d = yy - pp;
            d * d
        })
        .sum::<f64>()
        / n
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = parse_arg(&args, "--n", 50_000);
    let p = parse_arg(&args, "--p", 50).max(3);
    let seed = parse_seed(&args, 42);

    let mut rng = LcgRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.normal();
        }
    }

    let mut beta_true = Array1::<f64>::zeros(p);
    beta_true[0] = -0.25;
    beta_true[1] = 1.1;
    beta_true[2] = -0.9;
    if p > 3 {
        beta_true[3] = 0.6;
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eta = x.row(i).dot(&beta_true) + 0.2 * x[[i, 1]].sin();
        let pr = sigmoid(eta);
        y[i] = if rng.next_f64() < pr { 1.0 } else { 0.0 };
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    let s_list = vec![s];

    let opts = FitOptions {
        max_iter: 100,
        tol: 1e-6,
        nullspace_dims: vec![1],
    };

    let fit_start = Instant::now();
    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialLogit,
        &opts,
    )
    .expect("fit_gam failed");
    let fit_sec = fit_start.elapsed().as_secs_f64();

    let pred_start = Instant::now();
    let pred = predict_gam(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialLogit,
    )
    .expect("predict_gam failed");
    let pred_sec = pred_start.elapsed().as_secs_f64();

    let auc = auc_score(y.view(), pred.mean.view());
    let brier = brier_score(y.view(), pred.mean.view());

    println!(
        "{{\"engine\":\"gam\",\"scenario\":{{\"n\":{},\"p\":{}}},\"fit_sec\":{:.6},\"predict_sec\":{:.6},\"auc\":{:.6},\"brier\":{:.6},\"edf_total\":{:.6}}}",
        n, p, fit_sec, pred_sec, auc, brier, fit.edf_total
    );
}
