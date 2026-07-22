//! zz_measure DIAGNOSTIC (#1561): per-group oracle λ sweep for the Gaussian
//! location-scale BY-GROUP loser
//! (`families::quality_vs_gamlss_gaussian_location_scale_by_group`).
//!
//! Why this test exists
//! --------------------
//! The by-group location-scale quality test fails on GROUP A in BOTH channels
//! (μ-RMSE 0.1258 vs gamlss 0.0410; logσ-RMSE 0.5719 vs 0.1500) while group B is
//! essentially fine (0.0733/0.1757 vs 0.0487/0.0959). The mechanism is disputed
//! between OVER-smoothing (λ̂ above the accuracy optimum — group-A curves
//! flattened, residuals inflated, σ̂ biased) and UNDER-smoothing (λ̂ below it —
//! group-A curves chase noise). The two point at OPPOSITE fixes, and the failure
//! log carries no λ/edf, so the sign is unknown. This test pins it.
//!
//! Structure (mirrors tests/zz_measure_1561_mu_oracle_lambda_sweep.rs, commit
//! 254d78399, whose ρ=0 self-reconstruction is the non-negotiable validity guard)
//! ---------------------------------------------------------------------------
//! `s(x, by=group)` expands to one ROW-GATED per-level smooth per group (muA
//! zero on B rows, muB zero on A rows) plus a shared treatment intercept, so the
//! μ block is exactly linear-Gaussian GIVEN σ: at the converged fit β̂_μ solves
//!
//!     (XᵀWX + Σ_k λ̂_k S_k) β̂_μ = XᵀW(y − offset),   W = diag(1/σ̂²),
//!
//! over ALL 200 rows jointly. We rebuild that GLOBAL system, freeze the converged
//! σ̂ into W, and sweep ONLY the group-A μ-penalty scale exp(ρ)·λ̂_{μ,A} while the
//! group-B penalty and the (unpenalized) intercept stay at their production
//! values — so ρ=0 reproduces the production μ solve BY CONSTRUCTION (asserted),
//! and the argmin ρ*_A reads how much extra/less penalty group-A ACCURACY wants
//! relative to REML's joint choice. Because the group smooths are row-gated, the
//! group-A μ-RMSE responds almost entirely to its own penalty; the identical
//! sweep on group B is run alongside as the near-optimal control.
//!
//! Sign of ρ*_A:  ρ*_A ≪ 0 ⇒ REML λ̂ ABOVE the accuracy optimum ⇒ OVER-smoothing
//! (fix = widen/inspect the per-block ρ box, the #2356 class). ρ*_A ≫ 0 ⇒ REML λ̂
//! BELOW it ⇒ UNDER-smoothing (fix = leverage-corrected σ-score / criterion).
//!
//! A second μ sweep freezes the TRUE σ(x) into W (isolating whether gam's biased
//! σ̂ weights, not the criterion, drive the μ mis-selection: if ρ*_A(σ_true) ≈
//! ρ*_A(σ̂) the weights are exonerated).
//!
//! The σ arm freezes μ̂ and runs the analogous oracle on the log-σ block: given
//! the frozen residuals r̂ = y − μ̂, β̂_σ is the stationary point of the
//! conditional penalized σ-MLE (the joint σ-score depends on β_μ only through r̂),
//! so a penalized Fisher-scoring on σ reproduces β̂_σ at ρ=0 (asserted) and the
//! group-A σ-penalty sweep localizes whether logσ_A's 0.57 is a λ-selection
//! defect or downstream of μ_A.
//!
//! zz_measure discipline (feedback_zz_measure_diagnostic_tests): numbers are
//! eprintln'd under a stable `[zz1561:bygroup]` prefix; the ONLY hard asserts are
//! finiteness and the ρ=0 reconstruction fidelity that makes each oracle valid.

use gam::data::EncodedDataset;
use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::{TermCollectionDesign, TermCollectionSpec, build_term_collection_design};
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

/// gam's location-scale noise link floor: raw σ = response_scale·FLOOR + exp(η_σ)
/// (the response-relative soft floor is part of the saved fit contract; matches
/// the sibling μ-oracle test and probe_1561_locscale_lambda.rs).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

const N_PER_GROUP: usize = 100;
const GRID_POINTS: usize = 50;

// ===========================================================================
// Data — bit-identical to
// tests/quality/families/quality_vs_gamlss_gaussian_location_scale_by_group.rs
// (StdRng seed 321, group A rows then B rows; A=0.0, B=1.0 first-seen codes).
// ===========================================================================

fn mean_a(x: f64) -> f64 {
    (2.0 * PI * x).sin()
}
fn sigma_a(x: f64) -> f64 {
    0.10 + 0.10 * (PI * x).sin()
}
fn mean_b(x: f64) -> f64 {
    0.5 + 0.3 * (3.0 * PI * x).sin()
}
fn sigma_b(x: f64) -> f64 {
    0.12 + 0.08 * x
}

/// `seq(a, b, length.out = GRID_POINTS)`, matching the quality test's grid.
fn linspace(a: f64, b: f64) -> Vec<f64> {
    (0..GRID_POINTS)
        .map(|i| a + (b - a) * (i as f64) / ((GRID_POINTS - 1) as f64))
        .collect()
}

/// The fitted dataset plus everything the oracle needs, all derived from the
/// exact quality-test data stream.
struct ByGroupData {
    /// Per-row x, response, and group code (0.0 = A for rows 0..100, 1.0 = B).
    xs: Vec<f64>,
    ys: Vec<f64>,
    codes: Vec<f64>,
    /// Per-group dense grids over each group's observed x-range + truths on them.
    grid_a: Vec<f64>,
    grid_b: Vec<f64>,
    truth_mu_a: Vec<f64>,
    truth_mu_b: Vec<f64>,
    truth_logsig_a: Vec<f64>,
    truth_logsig_b: Vec<f64>,
    /// Encoded dataset ready to fit, and its column indices.
    ds: EncodedDataset,
    x_idx: usize,
    group_idx: usize,
}

fn build_bygroup_data() -> ByGroupData {
    let mut rng = StdRng::seed_from_u64(321);
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x");
    let std_normal = Normal::new(0.0_f64, 1.0_f64).expect("standard normal");

    let headers = vec!["y".to_string(), "x".to_string(), "group".to_string()];
    let mut rows: Vec<csv::StringRecord> = Vec::with_capacity(2 * N_PER_GROUP);
    let mut xs = Vec::with_capacity(2 * N_PER_GROUP);
    let mut ys = Vec::with_capacity(2 * N_PER_GROUP);
    let mut codes = Vec::with_capacity(2 * N_PER_GROUP);
    let mut x_a = Vec::with_capacity(N_PER_GROUP);
    let mut x_b = Vec::with_capacity(N_PER_GROUP);

    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_a(x) + sigma_a(x) * std_normal.sample(&mut rng);
        x_a.push(x);
        xs.push(x);
        ys.push(y);
        codes.push(0.0);
        rows.push(csv::StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "A".to_string(),
        ]));
    }
    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_b(x) + sigma_b(x) * std_normal.sample(&mut rng);
        x_b.push(x);
        xs.push(x);
        ys.push(y);
        codes.push(1.0);
        rows.push(csv::StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "B".to_string(),
        ]));
    }

    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode by-group data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let group_idx = col["group"];

    let (a_lo, a_hi) = x_a
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let (b_lo, b_hi) = x_b
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let grid_a = linspace(a_lo, a_hi);
    let grid_b = linspace(b_lo, b_hi);
    let truth_mu_a: Vec<f64> = grid_a.iter().map(|&x| mean_a(x)).collect();
    let truth_mu_b: Vec<f64> = grid_b.iter().map(|&x| mean_b(x)).collect();
    let truth_logsig_a: Vec<f64> = grid_a.iter().map(|&x| sigma_a(x).ln()).collect();
    let truth_logsig_b: Vec<f64> = grid_b.iter().map(|&x| sigma_b(x).ln()).collect();

    ByGroupData {
        xs,
        ys,
        codes,
        grid_a,
        grid_b,
        truth_mu_a,
        truth_mu_b,
        truth_logsig_a,
        truth_logsig_b,
        ds,
        x_idx,
        group_idx,
    }
}

// ===========================================================================
// Small dense SPD linear algebra (systems are p ≈ 20; hand-rolled to keep the
// diagnostic dependency-free, identical in spirit to the sibling test).
// ===========================================================================

fn chol_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let p = a.nrows();
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
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
    Some(l)
}

fn chol_solve_vec(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let p = l.nrows();
    let mut y = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = s / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }
    x
}

fn chol_with_jitter(a: &Array2<f64>) -> Array2<f64> {
    if let Some(l) = chol_lower(a) {
        return l;
    }
    let p = a.nrows();
    let scale: f64 = (0..p).map(|i| a[[i, i]].abs()).sum::<f64>() / (p as f64);
    let mut jitter = scale.max(1.0) * 1e-12;
    for _ in 0..40 {
        let mut aj = a.clone();
        for i in 0..p {
            aj[[i, i]] += jitter;
        }
        if let Some(l) = chol_lower(&aj) {
            return l;
        }
        jitter *= 10.0;
    }
    panic!("penalized normal matrix not factorizable even with jitter");
}

/// Σ_{j ∈ cols} [A⁻¹ B]_{jj} — the effective-df contribution of a column set
/// (whole set = tr(A⁻¹ B)).
fn ainv_b_diag_sum(l: &Array2<f64>, b: &Array2<f64>, cols: &[usize]) -> f64 {
    let mut acc = 0.0;
    for &j in cols {
        let col = b.column(j).to_owned();
        let z = chol_solve_vec(l, &col);
        acc += z[j];
    }
    acc
}

// ===========================================================================
// Design materialization helpers.
// ===========================================================================

/// Materialize the dense design matrix (rows × p) of a term collection at the
/// given design points, alongside its affine offset and the design itself (for
/// the penalty blocks).
fn materialize_design(
    points: &Array2<f64>,
    spec: &TermCollectionSpec,
) -> (Array2<f64>, Array1<f64>, TermCollectionDesign) {
    let design =
        build_term_collection_design(points.view(), spec).expect("rebuild term-collection design");
    let m = points.nrows();
    let p = design.design.ncols();
    let mut x = Array2::<f64>::zeros((m, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        let col_j = design.design.apply(&e);
        x.column_mut(j).assign(&col_j);
    }
    let offset = design.affine_offset.clone();
    (x, offset, design)
}

/// Materialize just the dense design matrix and affine offset (for grids, where
/// the penalty blocks are not needed).
fn materialize_xoff(points: &Array2<f64>, spec: &TermCollectionSpec) -> (Array2<f64>, Array1<f64>) {
    let md = materialize_design(points, spec);
    (md.0, md.1)
}

/// Embed each penalty block into a full p×p matrix (local block at col_range),
/// returning the embedded matrices in penalty order.
fn embed_penalties(design: &TermCollectionDesign, p: usize) -> Vec<Array2<f64>> {
    design
        .penalties
        .iter()
        .map(|bp| {
            let mut s = Array2::<f64>::zeros((p, p));
            for (a, gi) in bp.col_range.clone().enumerate() {
                for (b, gj) in bp.col_range.clone().enumerate() {
                    s[[gi, gj]] = bp.local[[a, b]];
                }
            }
            s
        })
        .collect()
}

/// For each penalty, decide whether its columns carry their design mass on the
/// group-A rows (code≈0) or the group-B rows (code≈1), via Σ|X[i,j]| over each
/// group. Returns (index_of_group_A_penalty, index_of_group_B_penalty). Assumes
/// exactly the two row-gated per-level smooths (the treatment intercept is
/// unpenalized, so it contributes no penalty block).
fn identify_group_penalties(
    design: &TermCollectionDesign,
    x_train: &Array2<f64>,
    codes: &[f64],
) -> (usize, usize) {
    let n = x_train.nrows();
    let mut a_idx: Option<usize> = None;
    let mut b_idx: Option<usize> = None;
    for (k, bp) in design.penalties.iter().enumerate() {
        let cols: Vec<usize> = bp.col_range.clone().collect();
        let mut mass_a = 0.0;
        let mut mass_b = 0.0;
        for i in 0..n {
            let m: f64 = cols.iter().map(|&j| x_train[[i, j]].abs()).sum();
            if codes[i] < 0.5 {
                mass_a += m;
            } else {
                mass_b += m;
            }
        }
        // A row-gated per-level smooth has (essentially) all its design mass on
        // its own group's rows; classify by the dominant side.
        if mass_a >= mass_b {
            a_idx = Some(k);
        } else {
            b_idx = Some(k);
        }
    }
    (
        a_idx.expect("a group-A-gated penalty"),
        b_idx.expect("a group-B-gated penalty"),
    )
}

fn max_abs_rel_diff(recon: &Array1<f64>, target: &Array1<f64>) -> (f64, f64) {
    let linf = recon
        .iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let scale = target.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-12);
    (linf, linf / scale)
}

// ===========================================================================
// The experiment.
// ===========================================================================

#[test]
fn zz_measure_1561_bygroup_oracle_lambda_sweep() {
    init_parallelism();
    eprintln!(
        "=== #1561 BY-GROUP DECISIVE EXPERIMENT: is group-A over- or under-smoothed? ===\n\
         Fit y ~ s(x,bs='tp',by=group) with noise s(x,bs='tp',by=group) on the exact\n\
         quality-test data (seed 321, n=200). Per-group oracle sweeps of λ_{{μ,A}}/λ_{{σ,A}}\n\
         about the REML choice pin the sign: ρ*<0 ⇒ over-smoothing, ρ*>0 ⇒ under-smoothing.\n\
         ρ=0 reproduces the production solve (asserted) so the oracle is faithful."
    );

    let d = build_bygroup_data();
    let n = d.xs.len();
    let ncols = d.ds.headers.len();

    // ---- fit exactly as the quality test does --------------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x, bs='tp', by=group)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp', by=group)", &d.ds, &cfg)
        .expect("gam by-group location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        ..
    }) = result
    else {
        panic!("expected a GaussianLocationScale fit");
    };
    let c = response_scale;

    let loc_block = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location block");
    let scale_block = fit.fit.block_by_role(BlockRole::Scale).expect("scale block");
    let beta_mu = loc_block.beta.clone();
    let beta_sigma = scale_block.beta.clone();
    let lambda_mu: Vec<f64> = loc_block.lambdas.to_vec();
    let lambda_sigma: Vec<f64> = scale_block.lambdas.to_vec();
    let edf_mu_total = loc_block.edf;
    let edf_sigma_total = scale_block.edf;

    // ---- rebuild μ & σ designs at the TRAINING points (x AND group set) ------
    let mut train_pts = Array2::<f64>::zeros((n, ncols));
    for i in 0..n {
        train_pts[[i, d.x_idx]] = d.xs[i];
        train_pts[[i, d.group_idx]] = d.codes[i];
    }
    let (x_mu, mu_offset, mu_design) = materialize_design(&train_pts, &fit.meanspec_resolved);
    let (x_sig, sig_offset, sig_design) = materialize_design(&train_pts, &fit.noisespec_resolved);
    let p = x_mu.ncols();
    let q = x_sig.ncols();
    assert_eq!(p, beta_mu.len(), "μ design width vs β̂_μ count");
    assert_eq!(q, beta_sigma.len(), "σ design width vs β̂_σ count");
    assert_eq!(
        mu_design.penalties.len(),
        lambda_mu.len(),
        "rebuilt μ penalty count vs μ-block λ count (λ̂→penalty mapping)"
    );
    assert_eq!(
        sig_design.penalties.len(),
        lambda_sigma.len(),
        "rebuilt σ penalty count vs σ-block λ count (λ̂→penalty mapping)"
    );

    let embedded_mu = embed_penalties(&mu_design, p);
    let embedded_sig = embed_penalties(&sig_design, q);
    let (mu_a_idx, mu_b_idx) = identify_group_penalties(&mu_design, &x_mu, &d.codes);
    let (sig_a_idx, sig_b_idx) = identify_group_penalties(&sig_design, &x_sig, &d.codes);

    // Group column sets of the μ smooths (for per-group edf leverage sums).
    let mu_a_cols: Vec<usize> = mu_design.penalties[mu_a_idx].col_range.clone().collect();
    let mu_b_cols: Vec<usize> = mu_design.penalties[mu_b_idx].col_range.clone().collect();

    // ---- per-group evaluation designs ----------------------------------------
    let make_grid_design = |grid: &[f64], code: f64| -> (Array2<f64>, Array1<f64>) {
        let mut pts = Array2::<f64>::zeros((grid.len(), ncols));
        for (i, &gx) in grid.iter().enumerate() {
            pts[[i, d.x_idx]] = gx;
            pts[[i, d.group_idx]] = code;
        }
        materialize_xoff(&pts, &fit.meanspec_resolved)
    };
    let make_grid_sig_design = |grid: &[f64], code: f64| -> (Array2<f64>, Array1<f64>) {
        let mut pts = Array2::<f64>::zeros((grid.len(), ncols));
        for (i, &gx) in grid.iter().enumerate() {
            pts[[i, d.x_idx]] = gx;
            pts[[i, d.group_idx]] = code;
        }
        materialize_xoff(&pts, &fit.noisespec_resolved)
    };
    let (x_grid_mu_a, off_grid_mu_a) = make_grid_design(&d.grid_a, 0.0);
    let (x_grid_mu_b, off_grid_mu_b) = make_grid_design(&d.grid_b, 1.0);
    let (x_grid_sig_a, off_grid_sig_a) = make_grid_sig_design(&d.grid_a, 0.0);
    let (x_grid_sig_b, off_grid_sig_b) = make_grid_sig_design(&d.grid_b, 1.0);

    // Grid μ-RMSE-to-truth for a μ coefficient vector, per group.
    let grid_mu_rmse_a =
        |beta: &Array1<f64>| rmse(&(x_grid_mu_a.dot(beta) + &off_grid_mu_a).to_vec(), &d.truth_mu_a);
    let grid_mu_rmse_b =
        |beta: &Array1<f64>| rmse(&(x_grid_mu_b.dot(beta) + &off_grid_mu_b).to_vec(), &d.truth_mu_b);
    // Grid logσ-RMSE-to-truth for a σ coefficient vector, per group.
    let logsig = |x: &Array2<f64>, off: &Array1<f64>, beta: &Array1<f64>| -> Vec<f64> {
        let eta = x.dot(beta) + off;
        eta.iter().map(|&e| (c * LOGB_SIGMA_FLOOR + e.exp()).ln()).collect()
    };
    let grid_logsig_rmse_a =
        |beta: &Array1<f64>| rmse(&logsig(&x_grid_sig_a, &off_grid_sig_a, beta), &d.truth_logsig_a);
    let grid_logsig_rmse_b =
        |beta: &Array1<f64>| rmse(&logsig(&x_grid_sig_b, &off_grid_sig_b, beta), &d.truth_logsig_b);

    // ---- frozen σ̂ and truth-σ weights (raw units) ---------------------------
    let eta_sigma = x_sig.dot(&beta_sigma) + &sig_offset;
    let sigma_hat: Vec<f64> = eta_sigma.iter().map(|&e| c * LOGB_SIGMA_FLOOR + e.exp()).collect();
    let w_hat: Vec<f64> = sigma_hat.iter().map(|&s| 1.0 / (s * s)).collect();
    let w_true: Vec<f64> = (0..n)
        .map(|i| {
            let st = if d.codes[i] < 0.5 {
                sigma_a(d.xs[i])
            } else {
                sigma_b(d.xs[i])
            };
            1.0 / st.max(1e-3).powi(2)
        })
        .collect();

    // ===== §1: production per-block readout ===================================
    // Base μ penalty in raw space reproducing β̂_μ at ρ=0: Σ_k (λ̂_k/c²)·scale_k·S_k
    // (production selects λ̂ on the c-standardized response; β̂_μ solves the
    // raw-space penalized WLS with penalty λ̂_k/c² — the c² factor is validated by
    // the ρ=0 reconstruction below, exactly as in the sibling test).
    let mu_penalty = |scale: &[f64]| -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((p, p));
        for (k, sk) in embedded_mu.iter().enumerate() {
            s.scaled_add((lambda_mu[k] / (c * c)) * scale[k], sk);
        }
        s
    };
    let build_xtwx_xtwr = |w: &[f64]| -> (Array2<f64>, Array1<f64>) {
        let mut xtwx = Array2::<f64>::zeros((p, p));
        let mut xtwr = Array1::<f64>::zeros(p);
        for i in 0..n {
            let wi = w[i];
            let ri = d.ys[i] - mu_offset[i];
            for a in 0..p {
                let xa = x_mu[[i, a]];
                xtwr[a] += wi * xa * ri;
                for b in 0..p {
                    xtwx[[a, b]] += wi * xa * x_mu[[i, b]];
                }
            }
        }
        (xtwx, xtwr)
    };
    let mu_solve_at = |xtwx: &Array2<f64>, xtwr: &Array1<f64>, scale: &[f64]| -> Array1<f64> {
        let mut a = xtwx.clone();
        a += &mu_penalty(scale);
        let l = chol_with_jitter(&a);
        chol_solve_vec(&l, xtwr)
    };

    let (xtwx_hat, xtwr_hat) = build_xtwx_xtwr(&w_hat);
    // Per-group edf via the leverage decomposition of the reconstruction system.
    let a_hat = {
        let mut a = xtwx_hat.clone();
        a += &mu_penalty(&vec![1.0; lambda_mu.len()]);
        chol_with_jitter(&a)
    };
    let edf_mu_a = ainv_b_diag_sum(&a_hat, &xtwx_hat, &mu_a_cols);
    let edf_mu_b = ainv_b_diag_sum(&a_hat, &xtwx_hat, &mu_b_cols);

    // Production per-group errors (should reproduce the quality-test failure).
    let prod_mu_rmse_a = grid_mu_rmse_a(&beta_mu);
    let prod_mu_rmse_b = grid_mu_rmse_b(&beta_mu);
    let prod_logsig_rmse_a = grid_logsig_rmse_a(&beta_sigma);
    let prod_logsig_rmse_b = grid_logsig_rmse_b(&beta_sigma);

    eprintln!(
        "[zz1561:bygroup] fit: n={n} p_mu={p} q_sig={q} response_scale(c)={c:.6} outer_iters={}",
        fit.fit.outer_iterations
    );
    eprintln!(
        "[zz1561:bygroup] μ blocks: λ̂_{{μ,A}}={:.4e} (ρ={:+.3}) λ̂_{{μ,B}}={:.4e} (ρ={:+.3}) \
         | edf_μ,A={edf_mu_a:.3} edf_μ,B={edf_mu_b:.3} edf_μ,total(fit)={edf_mu_total:.3}",
        lambda_mu[mu_a_idx],
        lambda_mu[mu_a_idx].ln(),
        lambda_mu[mu_b_idx],
        lambda_mu[mu_b_idx].ln()
    );
    eprintln!(
        "[zz1561:bygroup] σ blocks: λ̂_{{σ,A}}={:.4e} (ρ={:+.3}) λ̂_{{σ,B}}={:.4e} (ρ={:+.3}) \
         | edf_σ,total(fit)={edf_sigma_total:.3}",
        lambda_sigma[sig_a_idx],
        lambda_sigma[sig_a_idx].ln(),
        lambda_sigma[sig_b_idx],
        lambda_sigma[sig_b_idx].ln()
    );
    eprintln!(
        "[zz1561:bygroup] production truth-recovery (reproduces quality test): \
         A μ-RMSE={prod_mu_rmse_a:.4} logσ-RMSE={prod_logsig_rmse_a:.4} | \
         B μ-RMSE={prod_mu_rmse_b:.4} logσ-RMSE={prod_logsig_rmse_b:.4}"
    );

    // ===== §2/§3: μ oracle — sweep group-A (and group-B) penalty ==============
    // ρ=0 reconstruction of the production μ solve (frozen-σ̂ weights).
    let unit_scale = vec![1.0; lambda_mu.len()];
    let beta_mu_recon = mu_solve_at(&xtwx_hat, &xtwr_hat, &unit_scale);
    let (mu_recon_linf, mu_recon_rel) = max_abs_rel_diff(&beta_mu_recon, &beta_mu);
    eprintln!(
        "[zz1561:bygroup] μ ρ=0 reconstruction: ‖β_μ(0)−β̂_μ‖∞={mu_recon_linf:.3e} \
         (rel {mu_recon_rel:.3e})  rmse_A(0)={:.4} vs prod {prod_mu_rmse_a:.4}  \
         rmse_B(0)={:.4} vs prod {prod_mu_rmse_b:.4}",
        grid_mu_rmse_a(&beta_mu_recon),
        grid_mu_rmse_b(&beta_mu_recon)
    );

    let (xtwx_true, xtwr_true) = build_xtwx_xtwr(&w_true);

    // Sweep one group's μ penalty (index `sweep_idx`) over ρ∈[-8,8], holding the
    // other group + intercept at production; measure THAT group's μ-RMSE under
    // both frozen-σ̂ and truth-σ weights.
    let sweep_mu = |group: &str,
                    sweep_idx: usize,
                    rmse_of: &dyn Fn(&Array1<f64>) -> f64|
     -> (f64, f64, f64, f64) {
        let n_grid = 65usize;
        let (rho_lo, rho_hi) = (-8.0, 8.0);
        let mut best_hat = (f64::INFINITY, 0.0f64);
        let mut best_true = (f64::INFINITY, 0.0f64);
        eprintln!(
            "[zz1561:bygroup] μ sweep group {group} (λ_{{μ,{group}}} scaled):  rho | frozen-σ̂ rmse | truth-σ rmse"
        );
        for g in 0..n_grid {
            let rho = rho_lo + (rho_hi - rho_lo) * (g as f64) / ((n_grid - 1) as f64);
            let mut scale = vec![1.0; lambda_mu.len()];
            scale[sweep_idx] = rho.exp();
            let beta_h = mu_solve_at(&xtwx_hat, &xtwr_hat, &scale);
            let beta_t = mu_solve_at(&xtwx_true, &xtwr_true, &scale);
            let rh = rmse_of(&beta_h);
            let rt = rmse_of(&beta_t);
            assert!(rh.is_finite() && rt.is_finite(), "non-finite μ-RMSE at rho={rho}");
            if rh < best_hat.0 {
                best_hat = (rh, rho);
            }
            if rt < best_true.0 {
                best_true = (rt, rho);
            }
            if g % 4 == 0 || g == n_grid - 1 {
                eprintln!("[zz1561:bygroup]   {group} {rho:6.2} |  {rh:.5}  |  {rt:.5}");
            }
        }
        (best_hat.1, best_hat.0, best_true.1, best_true.0)
    };

    let (rho_a_hat, rmse_a_hat, rho_a_true, rmse_a_true) =
        sweep_mu("A", mu_a_idx, &grid_mu_rmse_a);
    let (rho_b_hat, rmse_b_hat, rho_b_true, rmse_b_true) =
        sweep_mu("B", mu_b_idx, &grid_mu_rmse_b);

    let verdict_mu = |rho_star: f64, gain_rel: f64| -> &'static str {
        if rho_star.abs() < 0.75 || gain_rel < 0.05 {
            "μ EXONERATED (REML λ̂ ≈ accuracy optimum)"
        } else if rho_star <= -0.75 {
            "μ OVER-SMOOTHING (REML λ̂ ABOVE optimum — accuracy wants LESS penalty)"
        } else {
            "μ UNDER-SMOOTHING (REML λ̂ BELOW optimum — accuracy wants MORE penalty)"
        }
    };
    let gain_a = if prod_mu_rmse_a > 0.0 {
        (prod_mu_rmse_a - rmse_a_hat) / prod_mu_rmse_a
    } else {
        0.0
    };
    let gain_b = if prod_mu_rmse_b > 0.0 {
        (prod_mu_rmse_b - rmse_b_hat) / prod_mu_rmse_b
    } else {
        0.0
    };
    eprintln!(
        "[zz1561:bygroup] μ ARGMIN A: frozen-σ̂ ρ*={rho_a_hat:+.2} rmse*={rmse_a_hat:.4} \
         (gain {:.1}% of prod {prod_mu_rmse_a:.4}) | truth-σ ρ*={rho_a_true:+.2} rmse*={rmse_a_true:.4}",
        100.0 * gain_a
    );
    eprintln!(
        "[zz1561:bygroup] μ ARGMIN B: frozen-σ̂ ρ*={rho_b_hat:+.2} rmse*={rmse_b_hat:.4} \
         (gain {:.1}% of prod {prod_mu_rmse_b:.4}) | truth-σ ρ*={rho_b_true:+.2} rmse*={rmse_b_true:.4}",
        100.0 * gain_b
    );
    eprintln!("[zz1561:bygroup] μ VERDICT A: {}", verdict_mu(rho_a_hat, gain_a));
    eprintln!("[zz1561:bygroup] μ VERDICT B: {}", verdict_mu(rho_b_hat, gain_b));
    let h2_a = if (rho_a_hat - rho_a_true).abs() < 0.75 {
        "H2 UNLIKELY (σ̂ vs σ_true μ-optima agree — biased weights not the driver)"
    } else {
        "H2 LIVE (σ̂ vs σ_true μ-optima disagree — biased weights contribute)"
    };
    eprintln!("[zz1561:bygroup] μ H2 read (group A): {h2_a}");

    // ===== §4: σ oracle — freeze μ̂, sweep group-A (and B) σ penalty ==========
    // Frozen residuals from the production μ̂ (raw units).
    let mu_hat_train = x_mu.dot(&beta_mu) + &mu_offset;
    let resid: Vec<f64> = (0..n).map(|i| d.ys[i] - mu_hat_train[i]).collect();

    // Raw-space σ penalty reproducing β̂_σ at ρ=0: Σ_k λ̂_{σ,k}·scale_k·S_k (NO c²
    // factor — the standardized and raw σ-log-likelihoods differ only by the
    // additive constant log c, so the σ-score, hence the penalty that balances
    // it, is scale-invariant; validated by the ρ=0 reconstruction below).
    let sig_penalty = |scale: &[f64]| -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((q, q));
        for (k, sk) in embedded_sig.iter().enumerate() {
            s.scaled_add(lambda_sigma[k] * scale[k], sk);
        }
        s
    };
    // Penalized Fisher-scoring on the conditional σ-MLE at penalty scale `scale`,
    // started from β̂_σ. Gaussian scale channel with σ = c·FLOOR + exp(η):
    //   score_i = (−1/σ + r²/σ³)·g,   fisher_i = 2 g²/σ²,   g = exp(η) = σ − c·FLOOR.
    let sig_solve_at = |scale: &[f64]| -> Array1<f64> {
        let s_pen = sig_penalty(scale);
        let mut beta = beta_sigma.clone();
        for _ in 0..100 {
            let eta = x_sig.dot(&beta) + &sig_offset;
            let mut grad = Array1::<f64>::zeros(q);
            let mut xtax = Array2::<f64>::zeros((q, q));
            for i in 0..n {
                let sg = c * LOGB_SIGMA_FLOOR + eta[i].exp();
                let g = sg - c * LOGB_SIGMA_FLOOR;
                let ri = resid[i];
                let sc = (-1.0 / sg + ri * ri / (sg * sg * sg)) * g;
                let fisher = 2.0 * g * g / (sg * sg);
                for a in 0..q {
                    let xa = x_sig[[i, a]];
                    grad[a] += xa * sc;
                    for b in 0..q {
                        xtax[[a, b]] += fisher * xa * x_sig[[i, b]];
                    }
                }
            }
            // ∇F = Xᵀ score − S_pen β ;  −∇²F ≈ Xᵀ diag(fisher) X + S_pen (PD).
            let g_pen = s_pen.dot(&beta);
            let rhs = &grad - &g_pen;
            let mut a = xtax;
            a += &s_pen;
            let l = chol_with_jitter(&a);
            let step = chol_solve_vec(&l, &rhs);
            let step_inf = step.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
            beta = &beta + &step;
            if step_inf < 1e-11 {
                break;
            }
        }
        beta
    };

    let unit_sig_scale = vec![1.0; lambda_sigma.len()];
    let beta_sig_recon = sig_solve_at(&unit_sig_scale);
    let (sig_recon_linf, sig_recon_rel) = max_abs_rel_diff(&beta_sig_recon, &beta_sigma);
    eprintln!(
        "[zz1561:bygroup] σ ρ=0 reconstruction: ‖β_σ(0)−β̂_σ‖∞={sig_recon_linf:.3e} \
         (rel {sig_recon_rel:.3e})  logσ-rmse_A(0)={:.4} vs prod {prod_logsig_rmse_a:.4}  \
         logσ-rmse_B(0)={:.4} vs prod {prod_logsig_rmse_b:.4}",
        grid_logsig_rmse_a(&beta_sig_recon),
        grid_logsig_rmse_b(&beta_sig_recon)
    );

    let sweep_sig = |group: &str,
                     sweep_idx: usize,
                     rmse_of: &dyn Fn(&Array1<f64>) -> f64|
     -> (f64, f64) {
        let n_grid = 65usize;
        let (rho_lo, rho_hi) = (-8.0, 8.0);
        let mut best = (f64::INFINITY, 0.0f64);
        eprintln!(
            "[zz1561:bygroup] σ sweep group {group} (λ_{{σ,{group}}} scaled):  rho | logσ-rmse"
        );
        for g in 0..n_grid {
            let rho = rho_lo + (rho_hi - rho_lo) * (g as f64) / ((n_grid - 1) as f64);
            let mut scale = vec![1.0; lambda_sigma.len()];
            scale[sweep_idx] = rho.exp();
            let beta = sig_solve_at(&scale);
            let r = rmse_of(&beta);
            assert!(r.is_finite(), "non-finite logσ-RMSE at rho={rho}");
            if r < best.0 {
                best = (r, rho);
            }
            if g % 4 == 0 || g == n_grid - 1 {
                eprintln!("[zz1561:bygroup]   {group} {rho:6.2} |  {r:.5}");
            }
        }
        (best.1, best.0)
    };

    let (rho_sig_a, rmse_sig_a) = sweep_sig("A", sig_a_idx, &grid_logsig_rmse_a);
    let (rho_sig_b, rmse_sig_b) = sweep_sig("B", sig_b_idx, &grid_logsig_rmse_b);
    let gain_sig_a = if prod_logsig_rmse_a > 0.0 {
        (prod_logsig_rmse_a - rmse_sig_a) / prod_logsig_rmse_a
    } else {
        0.0
    };
    let gain_sig_b = if prod_logsig_rmse_b > 0.0 {
        (prod_logsig_rmse_b - rmse_sig_b) / prod_logsig_rmse_b
    } else {
        0.0
    };
    eprintln!(
        "[zz1561:bygroup] σ ARGMIN A: ρ*={rho_sig_a:+.2} logσ-rmse*={rmse_sig_a:.4} \
         (gain {:.1}% of prod {prod_logsig_rmse_a:.4})",
        100.0 * gain_sig_a
    );
    eprintln!(
        "[zz1561:bygroup] σ ARGMIN B: ρ*={rho_sig_b:+.2} logσ-rmse*={rmse_sig_b:.4} \
         (gain {:.1}% of prod {prod_logsig_rmse_b:.4})",
        100.0 * gain_sig_b
    );
    let verdict_sig_a = if rho_sig_a.abs() < 0.75 || gain_sig_a < 0.05 {
        "σ_A largely EXONERATED at fixed μ̂ ⇒ logσ_A error is DOWNSTREAM of μ_A (couple the fixes)"
    } else if rho_sig_a <= -0.75 {
        "σ_A OVER-SMOOTHING (REML λ̂_{σ,A} above optimum — accuracy wants LESS penalty)"
    } else {
        "σ_A UNDER-SMOOTHING (REML λ̂_{σ,A} below optimum — accuracy wants MORE penalty)"
    };
    eprintln!("[zz1561:bygroup] σ VERDICT A: {verdict_sig_a}");

    // ---- the ONLY hard asserts: the two ρ=0 reconstructions ------------------
    // If either fails, the corresponding oracle's design/penalty/weight/λ̂ mapping
    // is wrong and its sweep is meaningless. Loose enough for solver/floor
    // tolerance, tight enough to catch a wrong basis, penalty, or λ̂→block map.
    assert!(
        mu_recon_rel < 5e-3,
        "[zz1561:bygroup] μ ρ=0 reconstruction did not reproduce β̂_μ (rel ‖·‖∞={mu_recon_rel:.3e}) \
         — μ oracle INVALID"
    );
    assert!(
        sig_recon_rel < 5e-3,
        "[zz1561:bygroup] σ ρ=0 reconstruction did not reproduce β̂_σ (rel ‖·‖∞={sig_recon_rel:.3e}) \
         — σ oracle INVALID"
    );
    assert!(
        prod_mu_rmse_a.is_finite()
            && prod_logsig_rmse_a.is_finite()
            && rmse_a_hat.is_finite()
            && rmse_sig_a.is_finite(),
        "[zz1561:bygroup] non-finite production/oracle error"
    );
}
