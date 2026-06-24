// TEMP R-free recovery check for #1266's escape vs #1476 concurvity collapse.
// Exact #1476 DGP; asserts the PRIMARY gam recovery (R-free part) — the escape
// must NOT collapse a supported concurvity null space. Delete after diagnosis.
use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const N: usize = 400;
const RHO: f64 = 0.90;
const NOISE: f64 = 0.30;
const SEED: u64 = 1_476_011;
const TAU: f64 = std::f64::consts::TAU;

fn rank_to_unit(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap());
    let mut rank = vec![0usize; n];
    for (r, &i) in idx.iter().enumerate() {
        rank[i] = r;
    }
    rank.iter().map(|&r| (r as f64 + 0.5) / n as f64).collect()
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

#[test]
fn diag_1476_rfree_recovery() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(SEED);
    let nrm = Normal::new(0.0, 1.0).unwrap();
    let (mut g0, mut g1) = (Vec::new(), Vec::new());
    for _ in 0..N {
        let z0: f64 = nrm.sample(&mut rng);
        let z1: f64 = nrm.sample(&mut rng);
        g0.push(z0);
        g1.push(RHO * z0 + (1.0 - RHO * RHO).sqrt() * z1);
    }
    let x1 = rank_to_unit(&g0);
    let x2 = rank_to_unit(&g1);
    let f2: Vec<f64> = x2.iter().map(|&v| v * v).collect();
    let f2m = f2.iter().sum::<f64>() / N as f64;
    let mu: Vec<f64> = (0..N).map(|i| (TAU * x1[i]).sin() + (f2[i] - f2m)).collect();
    let y: Vec<f64> = mu.iter().map(|&m| m + NOISE * nrm.sample(&mut rng)).collect();

    let headers: Vec<String> = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|r| StringRecord::from(vec![x1[r].to_string(), x2[r].to_string(), y[r].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).unwrap();
    let col = ds.column_map();
    let (i1, i2) = (col["x1"], col["x2"]);
    let width = ds.headers.len();
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let FitResult::Standard(fit) = fit_from_formula("y ~ s(x1) + s(x2)", &ds, &cfg).unwrap() else {
        panic!("std");
    };
    let edf = fit.fit.edf_total().unwrap();
    let mut pts = Array2::<f64>::zeros((N, width));
    for r in 0..N {
        pts[[r, i1]] = x1[r];
        pts[[r, i2]] = x2[r];
    }
    let design = build_term_collection_design(pts.view(), &fit.resolvedspec).unwrap();
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let err = rmse(&gam_fitted, &mu);
    println!("DIAG1476 gam_rmse_vs_truth={err:.5} edf_total={edf:.3} (PRIMARY bar < 0.09)");
    assert!(err < 0.09, "CONCURVITY COLLAPSE: gam_rmse_vs_truth={err:.5} >= 0.09 (#1476 regressed)");
}
