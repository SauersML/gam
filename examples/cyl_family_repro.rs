//! #1263 repro: binomial/poisson cylinder rate-surface recovery diagnostic.
//! Run: cargo run --release --example cyl_family_repro

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use std::f64::consts::TAU;

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    fn next_unit(&mut self) -> f64 {
        (self.next_u32() as f64 + 1.0) / ((u32::MAX as f64) + 1.0)
    }
}

fn encode_columns(headers: &[&str], columns: &[&[f64]]) -> gam::data::EncodedDataset {
    let n = columns[0].len();
    let hdrs: Vec<String> = headers.iter().map(|s| (*s).to_string()).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        rows.push(StringRecord::from(row));
    }
    encode_recordswith_inferred_schema(hdrs, rows).expect("encode dataset")
}

fn predict_matrix(n_cols: usize, columns_in_order: &[&[f64]]) -> Array2<f64> {
    let n_rows = columns_in_order[0].len();
    let mut m = Array2::<f64>::zeros((n_rows, n_cols));
    for (j, col) in columns_in_order.iter().enumerate() {
        for i in 0..n_rows {
            m[[i, j]] = col[i];
        }
    }
    m
}

fn fit_and_predict_eta(
    formula: &str,
    data: &gam::data::EncodedDataset,
    cfg: &FitConfig,
    test_rows: &Array2<f64>,
) -> Array1<f64> {
    let result = fit_from_formula(formula, data, cfg).expect("fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    eprintln!(
        "  beta(len={}) min={:.4} max={:.4} mean={:.4}",
        fit.fit.beta.len(),
        fit.fit.beta.iter().cloned().fold(f64::INFINITY, f64::min),
        fit.fit
            .beta
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        fit.fit.beta.iter().sum::<f64>() / fit.fit.beta.len() as f64,
    );
    eprintln!("  lambdas={:?}", fit.fit.lambdas);
    let test_design = build_term_collection_design(test_rows.view(), &fit.resolvedspec)
        .expect("rebuild prediction design");
    test_design.design.apply(&fit.fit.beta)
}

/// #1263 lane-discrimination diagnostic for the Poisson cylinder fit.
///
/// Pinpoints WHERE the ~0.38 log-level offset enters by dumping, at the
/// converged optimum on the SAME design gam used:
///   (1) the free intercept beta_0 vs true ln(mean rate), AND the GLM
///       intercept score identity Σμ vs Σy (must hold to machine eps if the
///       unpenalized intercept truly converged its score equation);
///   (2) per-margin fitted lambdas (is the cyclic-theta margin over-penalized?);
///   (3) an INDEPENDENT from-scratch UNPENALIZED Poisson IRLS on the identical
///       design matrix — if the unpenalized hand-fit recovers the level/structure
///       but gam (penalized REML) does not, the offset is penalty/λ-induced
///       (outer-REML lane); if the unpenalized hand-fit ALSO mis-levels, the
///       basis/working-response/weights are the culprit (IRLS/family lane).
fn poisson_level_diagnostic(
    formula: &str,
    data: &gam::data::EncodedDataset,
    train_rows: &Array2<f64>,
    y: &[f64],
    rate_true: &[f64],
) {
    let pcfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &pcfg).expect("poisson fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    let beta = &fit.fit.beta;
    let n = y.len() as f64;

    // --- gam's fitted training mean from its converged beta on the SAME design ---
    let train_design = build_term_collection_design(train_rows.view(), &fit.resolvedspec)
        .expect("rebuild training design");
    let eta_gam = train_design.design.apply(beta);
    let mu_gam: Array1<f64> = eta_gam.mapv(f64::exp);
    let sum_mu = mu_gam.sum();
    let sum_y: f64 = y.iter().sum();
    let mean_y = sum_y / n;

    eprintln!("---- #1263 POISSON LANE DIAGNOSTIC ----");
    eprintln!(
        "  (1) free intercept beta_0={:.5} | ln(mean_y)={:.5} | diff={:+.5}  (true level base=1.5)",
        beta[0],
        mean_y.ln(),
        beta[0] - mean_y.ln()
    );
    eprintln!(
        "      GLM intercept score identity: Σμ={:.4}  Σy={:.4}  rel_gap={:+.3e}  (≈0 ⇒ unpenalized intercept score CONVERGED)",
        sum_mu,
        sum_y,
        (sum_mu - sum_y) / sum_y
    );
    // Level locus: smooth columns are sum-to-zero, so Σ over rows of the
    // non-intercept contribution should be ~0; any drift means the smooth is
    // carrying level the intercept should own.
    let n_cols = train_design.design.ncols();
    let mut beta_no_intercept = beta.clone();
    beta_no_intercept[0] = 0.0;
    let smooth_part = train_design.design.apply(&beta_no_intercept);
    eprintln!(
        "      smooth-only Σ(rows)/n={:+.5} (sum-to-zero ⇒ ~0; nonzero ⇒ smooth leaks LEVEL) | design ncols={}",
        smooth_part.sum() / n,
        n_cols
    );

    // --- (2) per-margin lambdas ---
    eprintln!("  (2) fitted lambdas (per penalty block) = {:?}", fit.fit.lambdas);
    eprintln!(
        "      deviance={:.4}  reml_score={:.4}  outer_converged={}",
        fit.fit.deviance, fit.fit.reml_score, fit.fit.outer_converged
    );

    // --- (3) independent UNPENALIZED Poisson IRLS on the identical design ---
    // Materialize the dense design once (n×p) and run textbook Fisher-scoring
    // Poisson IRLS: W=μ, z=η+(y-μ)/μ, solve (XᵀWX)β = XᵀWz with a tiny ridge
    // for numerical safety only (1e-8, far below any smoothing scale).
    let xdense = train_design.design.to_dense();
    let (nn, p) = (xdense.nrows(), xdense.ncols());
    let mut b = Array1::<f64>::zeros(p);
    b[0] = mean_y.max(1e-6).ln(); // same intercept warm start gam uses
    let yv = Array1::from(y.to_vec());
    for _it in 0..200 {
        let eta = xdense.dot(&b);
        let mu = eta.mapv(|e| e.exp().clamp(1e-10, 1e10));
        let w = &mu; // Poisson canonical: W = μ
        let z = &eta + &((&yv - &mu) / &mu);
        // Build XᵀWX and XᵀWz
        let mut ata = Array2::<f64>::zeros((p, p));
        let mut atz = Array1::<f64>::zeros(p);
        for i in 0..nn {
            let row = xdense.row(i);
            let wi = w[i];
            let zi = z[i];
            for a in 0..p {
                let xa = row[a] * wi;
                atz[a] += xa * zi;
                for c in a..p {
                    ata[[a, c]] += xa * row[c];
                }
            }
        }
        for a in 0..p {
            ata[[a, a]] += 1e-8;
            for c in (a + 1)..p {
                ata[[c, a]] = ata[[a, c]];
            }
        }
        let bnew = solve_spd(&ata, &atz).unwrap_or_else(|| b.clone());
        let delta = (&bnew - &b).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
        b = bnew;
        if delta < 1e-9 {
            break;
        }
    }
    let eta_hand = xdense.dot(&b);
    let mu_hand: Array1<f64> = eta_hand.mapv(f64::exp);
    let rate_true_arr = Array1::from(rate_true.to_vec());
    eprintln!(
        "  (3) UNPENALIZED hand Poisson IRLS on identical design: beta_0={:.5} | Σμ={:.4} (Σy={:.4}) | MSE(rate vs truth)={:.4e}",
        b[0],
        mu_hand.sum(),
        sum_y,
        mse(&mu_hand, &rate_true_arr)
    );
    eprintln!(
        "      gam penalized fit MSE(rate vs truth, TRAIN)={:.4e}",
        mse(&mu_gam, &rate_true_arr)
    );
    eprintln!(
        "      VERDICT: if hand-IRLS MSE ≈ truth (small) but gam MSE large ⇒ PENALTY/λ lane (outer-REML).\n               if hand-IRLS ALSO large ⇒ working-response/weights/basis lane (IRLS/family)."
    );
    eprintln!("---------------------------------------");
}

/// Minimal SPD solve via Cholesky (lower). Returns None if not PD.
fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // forward solve L y = b
    let mut yv = b.clone();
    for i in 0..n {
        let mut s = yv[i];
        for k in 0..i {
            s -= l[[i, k]] * yv[k];
        }
        yv[i] = s / l[[i, i]];
    }
    // back solve Lᵀ x = y
    let mut x = yv.clone();
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }
    Some(x)
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn mse(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

fn main() {
    init_parallelism();
    // ---- binomial cylinder ----
    let n_theta = 30usize;
    let n_h = 20usize;
    let n = n_theta * n_h;
    let mut rng = Lcg::new(0xB1B1_C0C0_u64);
    let (mut theta, mut h, mut p_true, mut y) = (vec![], vec![], vec![], vec![]);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hh = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            let eta = 0.55 * t.cos() - 0.25 * (2.0 * t).sin() + 0.3 * hh;
            let p = logistic(eta);
            let u = rng.next_unit();
            theta.push(t);
            h.push(hh);
            p_true.push(p);
            y.push(if u < p { 1.0 } else { 0.0 });
        }
    }
    let data = encode_columns(&["theta", "h", "y"], &[&theta, &h, &y]);
    let test = predict_matrix(3, &[&theta, &h, &vec![0.0; n]]);
    let p_true_arr = Array1::from(p_true.clone());
    let bcfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    eprintln!("== BINOMIAL te k=4 ==");
    let eta_pred = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &bcfg,
        &test,
    );
    let p_pred = eta_pred.mapv(logistic);
    eprintln!(
        "  eta_pred min={:.3} max={:.3} | MSE(p)={:.4e} (tol 0.02)",
        eta_pred.iter().cloned().fold(f64::INFINITY, f64::min),
        eta_pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mse(&p_pred, &p_true_arr)
    );

    // control: same data, GAUSSIAN family fit to y (just to see surface shape)
    eprintln!("== BINOMIAL-data GAUSSIAN te k=4 (control) ==");
    let gcfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let _ = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &gcfg,
        &test,
    );

    // control: binomial k=8 (more capacity)
    eprintln!("== BINOMIAL te k=8 (more capacity) ==");
    let eta8 = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=8)",
        &data,
        &bcfg,
        &test,
    );
    eprintln!(
        "  MSE(p) k=8 = {:.4e}",
        mse(&eta8.mapv(logistic), &p_true_arr)
    );

    // ---- poisson cylinder ----
    let mut rng = Lcg::new(0xDEADC0DE_u64);
    let (mut theta, mut h, mut rate_true, mut y) = (vec![], vec![], vec![], vec![]);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hh = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            let eta = 1.5 + 0.4 * t.cos() + 0.2 * hh;
            let lam = eta.exp();
            let mut k = 0u32;
            let mut s = 0.0_f64;
            loop {
                s += -rng.next_unit().ln();
                if s > lam {
                    break;
                }
                k += 1;
                if k > 100 {
                    break;
                }
            }
            theta.push(t);
            h.push(hh);
            rate_true.push(lam);
            y.push(k as f64);
        }
    }
    let data = encode_columns(&["theta", "h", "y"], &[&theta, &h, &y]);
    let test = predict_matrix(3, &[&theta, &h, &vec![0.0; n]]);
    let rate_true_arr = Array1::from(rate_true.clone());
    let pcfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    eprintln!("== POISSON te k=4 ==");
    let eta_pred = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &pcfg,
        &test,
    );
    let rate_pred = eta_pred.mapv(f64::exp);
    eprintln!(
        "  eta_pred min={:.3} max={:.3} | MSE(rate)={:.4e} (tol 0.5) | mean_rate_true={:.3}",
        eta_pred.iter().cloned().fold(f64::INFINITY, f64::min),
        eta_pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mse(&rate_pred, &rate_true_arr),
        rate_true.iter().sum::<f64>() / n as f64
    );

    // The training design is the same rows as `test` here (test reuses theta,h
    // with a dummy y column), so reuse `test` as the train design rows.
    poisson_level_diagnostic(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &test,
        &y,
        &rate_true,
    );
}
