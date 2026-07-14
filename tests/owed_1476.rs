//! R-free owed-work regression gate for #1476: a textbook additive model
//! `y ~ s(x1) + s(x2)` (gaussian) on two MODERATELY CORRELATED covariates
//! (corr ≈ 0.90) must recover the known component functions under the DEFAULT
//! double-penalty basis — it must NOT collapse one supported smooth.
//!
//! BUG: under moderate concurvity the double-penalty null-space selection ridge
//! mis-allocates. The two smooths' null-space (linear/constant) directions are
//! near-collinear, so the joint REML objective is essentially flat along the
//! "transfer the shared linear signal between the two smooths" ridge. With NO
//! prior curvature on the null-space coordinate (it is left fully `Flat` when the
//! fit is well-determined), one smooth's `λ_nullspace` rails to the ρ bound,
//! annihilating that smooth's genuine linear signal to `EDF ≈ 0` while the other
//! absorbs it. mgcv's `select=TRUE` vs `select=FALSE` leaves a supported smooth
//! alone (1.00×); gam's `double_penalty=True` over-shrinks ~3.3× vs
//! `double_penalty=False` on the SAME data.
//!
//! This gate is R-FREE: it contrasts gam's OWN `double_penalty=True` against
//! `double_penalty=False` on byte-identical rows (the issue's isolating
//! measurement). It asserts (a) the double-penalty fit recovers the additive
//! truth within a small factor of the single-penalty fit on the SAME data, and
//! (b) NEITHER smooth collapses (each per-smooth EDF > 1) under concurvity.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::TAU;

const N: usize = 400;
const RHO: f64 = 0.90;
const NOISE: f64 = 0.30;
const SEED: u64 = 1_476_011;

/// Deterministic SplitMix64 + Box–Muller standard-normal stream (no external RNG).
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

fn encode(cols: &[(&str, &[f64])]) -> EncodedDataset {
    let n = cols[0].1.len();
    let headers: Vec<String> = cols.iter().map(|(h, _)| (*h).to_string()).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(
                cols.iter()
                    .map(|(_, c)| c[i].to_string())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode concurvity dataset")
}

/// Rank-transform to exact Uniform[0,1] marginals: o = argsort(argsort(a)).
fn rank_to_unit(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).expect("finite covariate"));
    let mut rank = vec![0usize; n];
    for (r, &i) in idx.iter().enumerate() {
        rank[i] = r;
    }
    rank.iter().map(|&r| (r as f64 + 0.5) / n as f64).collect()
}

/// Per-smooth EDF for the term whose name contains `needle`.
fn term_edf(fit: &FitResult, needle: &str) -> f64 {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected standard fit");
    };
    let design = &std_fit.design;
    let unified = &std_fit.fit;
    let mut penalty_cursor = 0usize;
    for (_n, _r) in &design.random_effect_ranges {
        penalty_cursor += 1;
    }
    for term in &design.smooth.terms {
        let k = term.active_penalties.len();
        if term.name.contains(needle) {
            return unified.per_term_edf(term.coeff_range.clone(), penalty_cursor, k);
        }
        penalty_cursor += k;
    }
    panic!("no term {needle}");
}

struct Fit {
    fitted: Vec<f64>,
    edf1: f64,
    edf2: f64,
}

/// Fit `y ~ s(x1) + s(x2)` (gaussian/identity, REML) at the chosen double-penalty
/// setting on byte-identical rows; return the fitted mean and per-smooth EDF.
fn fit_concurvity(
    x1: &[f64],
    x2: &[f64],
    y: &[f64],
    width: usize,
    i1: usize,
    i2: usize,
    double_penalty: bool,
) -> Fit {
    let ds = encode(&[("x1", x1), ("x2", x2), ("y", y)]);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let dp = if double_penalty { "true" } else { "false" };
    let formula = format!("y ~ s(x1, double_penalty={dp}) + s(x2, double_penalty={dp})");
    let result = fit_from_formula(&formula, &ds, &cfg)
        .unwrap_or_else(|e| panic!("gam concurvity fit (dp={dp}) failed: {e:?}"));
    let edf1 = term_edf(&result, "x1");
    let edf2 = term_edf(&result, "x2");
    let FitResult::Standard(fit) = &result else {
        panic!("expected a standard GAM fit");
    };
    let mut pts = Array2::<f64>::zeros((N, width));
    for r in 0..N {
        pts[[r, i1]] = x1[r];
        pts[[r, i2]] = x2[r];
    }
    let design = build_term_collection_design(pts.view(), &fit.resolvedspec)
        .expect("rebuild concurvity design at training points");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    Fit { fitted, edf1, edf2 }
}

/// #1476: the DEFAULT double-penalty `s(x1)+s(x2)` must recover the additive
/// truth under concurvity, not collapse a supported smooth. We contrast gam's own
/// `double_penalty=True` against `double_penalty=False` on identical rows.
#[test]
fn gam_double_penalty_does_not_overshrink_supported_smooth_under_concurvity_1476() {
    init_parallelism();

    // ---- correlated design (Gaussian copula, corr ≈ RHO) on Uniform[0,1] -----
    let mut rng = SplitMix64::new(SEED);
    let mut g0 = Vec::with_capacity(N);
    let mut g1 = Vec::with_capacity(N);
    for _ in 0..N {
        let z0 = rng.next_normal();
        let z1 = rng.next_normal();
        g0.push(z0);
        g1.push(RHO * z0 + (1.0 - RHO * RHO).sqrt() * z1);
    }
    let x1 = rank_to_unit(&g0);
    let x2 = rank_to_unit(&g1);

    // truth: f1 = sin(2π x1), f2 = x2² (centered); mu = f1 + f2c. A low-order
    // quadratic is nullspace-heavy on [0,1] — exactly what the ridge over-shrinks.
    let f2: Vec<f64> = x2.iter().map(|&v| v * v).collect();
    let f2_mean = f2.iter().sum::<f64>() / N as f64;
    let mu: Vec<f64> = (0..N)
        .map(|i| (TAU * x1[i]).sin() + (f2[i] - f2_mean))
        .collect();
    let y: Vec<f64> = mu.iter().map(|&m| m + NOISE * rng.next_normal()).collect();

    let ds = encode(&[("x1", &x1), ("x2", &x2), ("y", &y)]);
    let col = ds.column_map();
    let (i1, i2) = (col["x1"], col["x2"]);
    let width = ds.headers.len();

    let dp_true = fit_concurvity(&x1, &x2, &y, width, i1, i2, true);
    let dp_false = fit_concurvity(&x1, &x2, &y, width, i1, i2, false);

    let err_true = rmse(&dp_true.fitted, &mu);
    let err_false = rmse(&dp_false.fitted, &mu);

    eprintln!(
        "[#1476] concurvity (seed {SEED}, corr≈{RHO}): n={N} \
         dp=TRUE  rmse_vs_truth={err_true:.5} edf[s(x1)={:.2}, s(x2)={:.2}] | \
         dp=FALSE rmse_vs_truth={err_false:.5} edf[s(x1)={:.2}, s(x2)={:.2}] | ratio={:.2}",
        dp_true.edf1,
        dp_true.edf2,
        dp_false.edf1,
        dp_false.edf2,
        err_true / err_false.max(1e-12)
    );

    // PRIMARY (#1476): the default double-penalty fit must recover the additive
    // truth within a small factor of the single-penalty fit on the SAME data.
    // mgcv's select=TRUE vs select=FALSE is 1.00×; the bug ships ≈3.3× over-shrink.
    // A factor of 1.6 catches the collapse and tolerates the legitimate small
    // regularization difference between the two penalty conventions.
    assert!(
        err_true <= 1.6 * err_false + 1e-9,
        "double_penalty=True over-shrinks under concurvity: rmse_vs_truth {err_true:.5} \
         > 1.6 * double_penalty=False rmse {err_false:.5} (ratio {:.2}); a supported \
         smooth's null-space ridge is over-selected by the select-out prior (#1476).",
        err_true / err_false.max(1e-12)
    );

    // NO COLLAPSE (#1476): under concurvity BOTH supported smooths must keep more
    // than a single degree of freedom. The bug rails one smooth's nullspace λ to
    // ≈1e13, flattening it to EDF ≈ 0 (well below 1).
    assert!(
        dp_true.edf1 > 1.0 && dp_true.edf2 > 1.0,
        "double_penalty=True collapses a supported smooth under concurvity: \
         edf[s(x1)]={:.3}, edf[s(x2)]={:.3} (each must exceed 1.0) — the null-space \
         select-out prior over-shrinks a genuinely-supported smooth (#1476).",
        dp_true.edf1,
        dp_true.edf2
    );
}
