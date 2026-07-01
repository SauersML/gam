//! EXPLORATORY probe (temporary) for #1765: measure the real observation-band
//! coverage at high edf/n on a low-noise Gaussian fit, comparing the mgcv
//! `RSS/(n-edf)` scale against the unbiased `RSS/(n-edf2)` scale.

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}
fn ks_vs_uniform(mut u: Vec<f64>) -> f64 {
    u.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = u.len() as f64;
    let mut d = 0.0f64;
    for (i, &ui) in u.iter().enumerate() {
        let lo = i as f64 / n;
        let hi = (i as f64 + 1.0) / n;
        d = d.max((ui - lo).abs()).max((hi - ui).abs());
    }
    d
}

fn box_muller(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn run_probe(n: usize, p: usize, noise_sd: f64, seed: u64) {
    let z95 = 1.959964_f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    let mut mean_true = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        for j in 1..p {
            let freq = ((j + 1) / 2) as f64;
            let arg = std::f64::consts::PI * freq * t;
            x[[i, j]] = if j % 2 == 1 { arg.sin() } else { arg.cos() };
        }
        // A spectrally rich but smooth truth.
        let mut m = 0.0;
        for k in 1..=8 {
            m += (1.0 / k as f64) * (std::f64::consts::PI * k as f64 * t).sin();
        }
        mean_true[i] = m;
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        y[i] = mean_true[i] + box_muller(&mut rng) * noise_sd;
    }
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        s[[j, j]] = 1.0;
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let penalty = BlockwisePenalty::new(0..p, s.clone());
    let opts = FitOptions {
        compute_inference: true,
        max_iter: 200,
        tol: 1e-11,
        nullspace_dims: vec![0],
        ..FitOptions::default()
    };
    let fit = fit_gam(
        x.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        &[penalty],
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &opts,
    )
    .expect("fit");

    let edf = fit.edf_total().expect("edf");
    let sigma = fit.standard_deviation;
    let fitted: Array1<f64> = x.dot(&fit.beta);
    let rss: f64 = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| (yi - fi).powi(2))
        .sum();
    let nf = n as f64;

    // tr(F^2) from coefficient influence F = H^{-1} X'WX.
    let f_infl = fit.coefficient_influence().expect("influence");
    let mut tr_f2 = 0.0;
    for a in 0..p {
        for b in 0..p {
            tr_f2 += f_infl[[a, b]] * f_infl[[b, a]];
        }
    }
    let edf2 = 2.0 * edf - tr_f2;

    let sig2_edf = rss / (nf - edf);
    let sig2_edf2 = rss / (nf - edf2);
    let sig2_mle = rss / nf;

    // Per-row etavar from Vp (corrected) and Vb (conditional).
    let vp = fit.beta_covariance_corrected();
    let vb = fit.beta_covariance();
    let etavar = |cov: &Array2<f64>, row: usize| -> f64 {
        let mut acc = 0.0;
        for a in 0..p {
            let xa = x[[row, a]];
            if xa == 0.0 {
                continue;
            }
            let mut inner = 0.0;
            for b in 0..p {
                inner += cov[[a, b]] * x[[row, b]];
            }
            acc += xa * inner;
        }
        acc
    };

    let mut mean_etavar_vp = 0.0;
    let mut mean_etavar_vb = 0.0;
    // Fresh-noise held-out coverage at the same design rows.
    let mut cov_band_vp = 0usize; // sqrt(sigma^2 + etavar_vp)
    let mut cov_noise_only = 0usize; // sqrt(sigma^2) only
    let mut cov_edf2_vp = 0usize; // sqrt(sig2_edf2 + etavar_vp)
    let mut pit_band_vp = Vec::with_capacity(n);
    let mut pit_edf2_vp = Vec::with_capacity(n);
    for i in 0..n {
        let y_new = mean_true[i] + box_muller(&mut rng) * noise_sd;
        let ev_vp = vp.map(|c| etavar(c, i)).unwrap_or(0.0);
        let ev_vb = vb.map(|c| etavar(c, i)).unwrap_or(0.0);
        mean_etavar_vp += ev_vp;
        mean_etavar_vb += ev_vb;
        let mu = fitted[i];
        let sd_band = (sigma * sigma + ev_vp).max(0.0).sqrt();
        let sd_noise = sigma;
        let sd_edf2 = (sig2_edf2 + ev_vp).max(0.0).sqrt();
        if (y_new - mu).abs() <= z95 * sd_band {
            cov_band_vp += 1;
        }
        if (y_new - mu).abs() <= z95 * sd_noise {
            cov_noise_only += 1;
        }
        if (y_new - mu).abs() <= z95 * sd_edf2 {
            cov_edf2_vp += 1;
        }
        pit_band_vp.push(normal_cdf((y_new - mu) / sd_band));
        pit_edf2_vp.push(normal_cdf((y_new - mu) / sd_edf2));
    }
    mean_etavar_vp /= nf;
    mean_etavar_vb /= nf;

    eprintln!("==== PROBE n={n} p={p} noise={noise_sd} seed={seed} ====");
    eprintln!(
        "edf={edf:.2} edf2={edf2:.2} tr_f2={tr_f2:.2} edf/n={:.3}",
        edf / nf
    );
    eprintln!(
        "sigma_hat={sigma:.6} (true {noise_sd}) | sig2_edf={sig2_edf:.8} sig2_edf2={sig2_edf2:.8} sig2_mle={sig2_mle:.8}",
    );
    eprintln!(
        "sqrt(sig2_edf)={:.6} sqrt(sig2_edf2)={:.6}",
        sig2_edf.sqrt(),
        sig2_edf2.sqrt()
    );
    eprintln!(
        "mean_etavar_vp={mean_etavar_vp:.8} mean_etavar_vb={mean_etavar_vb:.8} (sigma^2={:.8})",
        sigma * sigma
    );
    eprintln!(
        "COVERAGE  band_vp(edf-scale)={:.3}  noise_only={:.3}  edf2_scale={:.3}",
        cov_band_vp as f64 / nf,
        cov_noise_only as f64 / nf,
        cov_edf2_vp as f64 / nf
    );
    eprintln!(
        "PIT_KS    band_vp(edf-scale)={:.4}  edf2_scale={:.4}",
        ks_vs_uniform(pit_band_vp),
        ks_vs_uniform(pit_edf2_vp)
    );
}

#[test]
fn obs_coverage_probe_sweep_1765() {
    run_probe(400, 160, 0.01, 11);
    run_probe(400, 200, 0.02, 12);
    run_probe(600, 250, 0.01, 13);
    run_probe(300, 200, 0.03, 14);
    run_probe(800, 120, 0.05, 15);
}
