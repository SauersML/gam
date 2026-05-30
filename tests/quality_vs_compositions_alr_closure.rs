//! End-to-end quality: gam's compositional-geometry primitive
//! (`gam::geometry::simplex::simplex_frechet_mean`) is the load-bearing
//! Aitchison-geometry operation — the closed weighted geometric mean (Aitchison
//! center) of a data cloud on the simplex. This test asserts OBJECTIVE quality
//! of that primitive, never "gam reproduces R `compositions`":
//!
//!   (A) GROUND-TRUTH RECOVERY (correctness vs analytic ground truth). The
//!       Aitchison center has an *exact closed form*: `clo(exp(sum_i w_i·log c_i))`
//!       on the closed rows `c_i`. We compute that closed form independently in
//!       plain Rust and assert gam reproduces it to f64 round-off. Matching an
//!       exact mathematical quantity is an objective accuracy claim.
//!
//!   (B) FRÉCHET OPTIMALITY (the defining variational property). The Aitchison
//!       center is by definition the unique minimizer of the weighted Fréchet
//!       functional `F(m) = sum_i w_i · d_A(x_i, m)^2` over the open simplex,
//!       where `d_A` is the Aitchison distance. We assert that gam's output is a
//!       stationary point (the analytic Fréchet gradient in the clr tangent
//!       space vanishes) AND that it strictly beats a battery of deterministic
//!       perturbed competitors on `F`. This is gam's own quality, computed on
//!       gam's own output — it proves gam returns the *optimal* center, which is
//!       a strictly stronger claim than agreeing with any peer tool.
//!
//!   (C) SIMPLEX-MEMBERSHIP STRUCTURE (constraint satisfaction). gam's output
//!       must lie on the simplex: sum to 1 (to round-off) and be strictly
//!       positive. Asserted directly as a property of the output.
//!
//! The mature R `compositions` package (`mean(acomp(.))`) is retained only as a
//! BASELINE-TO-MATCH-OR-BEAT on the objective Fréchet functional `F`: gam's `F`
//! must be <= compositions' `F` (up to round-off). We also print the rel-L2
//! between the two centers with `eprintln!` for context, but agreement with
//! `compositions` is NOT a pass criterion.

use gam::geometry::simplex::simplex_frechet_mean;
use gam::test_support::reference::{Column, relative_l2, run_r};
use ndarray::{Array1, Array2};

/// Closure (sum-to-1 projection) of a single positive vector.
fn close(v: &[f64]) -> Vec<f64> {
    let s: f64 = v.iter().sum();
    v.iter().map(|x| x / s).collect()
}

/// centred-log-ratio of a closed, strictly-positive composition.
/// `clr(c)_j = log c_j - (1/d) sum_k log c_k`. The clr maps the simplex
/// isometrically into the hyperplane `sum = 0` of R^d with the ordinary
/// Euclidean metric, so Aitchison geometry becomes plain linear algebra there.
fn clr(c: &[f64]) -> Vec<f64> {
    let logs: Vec<f64> = c.iter().map(|x| x.ln()).collect();
    let mean = logs.iter().sum::<f64>() / logs.len() as f64;
    logs.iter().map(|l| l - mean).collect()
}

/// Squared Aitchison distance `d_A(x, m)^2 = || clr(x) - clr(m) ||^2`.
fn aitchison_sq(x: &[f64], m: &[f64]) -> f64 {
    let cx = clr(x);
    let cm = clr(m);
    cx.iter()
        .zip(&cm)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
}

/// Weighted Fréchet functional `F(m) = sum_i w_i d_A(closed_row_i, m)^2`,
/// evaluated on the closed (sum-to-1) rows of `raw` with normalized weights `w`.
fn frechet_functional(raw: &Array2<f64>, w: &[f64], m: &[f64]) -> f64 {
    let n = raw.nrows();
    let wsum: f64 = w.iter().sum();
    let mut f = 0.0;
    for i in 0..n {
        let row: Vec<f64> = close(&raw.row(i).to_vec());
        f += (w[i] / wsum) * aitchison_sq(&row, m);
    }
    f
}

/// Analytic Fréchet gradient of `F` in the clr tangent space (the hyperplane
/// `sum = 0`). For `F(m) = sum_i w_i || clr(c_i) - clr(m) ||^2`, the gradient
/// w.r.t. the clr coordinates of `m` is `2 * (clr(m) - sum_i w_i clr(c_i))`.
/// At the Aitchison center this is exactly zero, so its norm is a scale-free
/// stationarity certificate.
fn frechet_grad_norm(raw: &Array2<f64>, w: &[f64], m: &[f64]) -> f64 {
    let n = raw.nrows();
    let wsum: f64 = w.iter().sum();
    let d = m.len();
    let mut bary = vec![0.0; d];
    for i in 0..n {
        let row: Vec<f64> = close(&raw.row(i).to_vec());
        let cr = clr(&row);
        for j in 0..d {
            bary[j] += (w[i] / wsum) * cr[j];
        }
    }
    let cm = clr(m);
    cm.iter()
        .zip(&bary)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Exact closed-form Aitchison center: `clo(exp(sum_i w_i log c_i))` over the
/// closed rows. This IS the analytic ground truth gam must reproduce.
fn closed_geometric_mean(raw: &Array2<f64>, w: &[f64]) -> Vec<f64> {
    let n = raw.nrows();
    let d = raw.ncols();
    let wsum: f64 = w.iter().sum();
    let mut mean_log = vec![0.0; d];
    for i in 0..n {
        let row: Vec<f64> = close(&raw.row(i).to_vec());
        for j in 0..d {
            mean_log[j] += (w[i] / wsum) * row[j].ln();
        }
    }
    let g: Vec<f64> = mean_log.iter().map(|l| l.exp()).collect();
    close(&g)
}

#[test]
fn simplex_frechet_mean_is_the_optimal_aitchison_center() {
    // ---- fixed-seed synthetic compositional cloud: 50 rows x 4 parts -------
    // Deterministic positive "counts" (no RNG dependency): a smooth, reproducible
    // recipe with a non-degenerate spread across all 4 parts. Raw (unclosed) and
    // strictly positive, so closure and the log-geometry are well defined.
    let n = 50usize;
    let d = 4usize;
    let mut raw = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        let t = (i as f64 + 1.0) / (n as f64); // (0, 1]
        let c0 = 2.0 + 1.5 * (std::f64::consts::TAU * t).sin().abs() + 0.3 * t;
        let c1 = 1.0 + 0.8 * (3.0 * t + 0.5).cos().abs() + 0.7 * t * t;
        let c2 = 0.5 + 1.2 * ((i % 7) as f64) / 7.0 + 0.4 * (1.0 - t);
        let c3 = 0.25 + 0.9 * ((i % 5) as f64 + 1.0) / 5.0 + 0.2 * t;
        raw[[i, 0]] = c0;
        raw[[i, 1]] = c1;
        raw[[i, 2]] = c2;
        raw[[i, 3]] = c3;
    }
    for v in raw.iter() {
        assert!(*v > 0.0, "synthetic composition must be strictly positive");
    }

    // Fixed, reproducible non-uniform weights (unnormalized; gam normalizes).
    let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.05).collect();
    let uniform: Vec<f64> = vec![1.0; n];

    // ---- gam outputs -------------------------------------------------------
    let gam_mean = simplex_frechet_mean(raw.view(), None).expect("gam simplex_frechet_mean");
    assert_eq!(gam_mean.len(), d, "gam mean must have d components");
    let w_arr = Array1::from(weights.clone());
    let gam_wmean =
        simplex_frechet_mean(raw.view(), Some(w_arr.view())).expect("gam weighted frechet mean");

    // ===================================================================== //
    // (A) GROUND-TRUTH RECOVERY: match the exact closed-form Aitchison center
    //     computed independently in plain Rust. This is correctness vs an
    //     analytic mathematical quantity, not "same as a peer tool".
    // ===================================================================== //
    let truth_mean = closed_geometric_mean(&raw, &uniform);
    let truth_wmean = closed_geometric_mean(&raw, &weights);
    let rec_unif = relative_l2(&gam_mean, &truth_mean);
    let rec_wt = relative_l2(&gam_wmean, &truth_wmean);

    // ===================================================================== //
    // (B) FRÉCHET OPTIMALITY: stationarity + strictly-beats perturbations.
    // ===================================================================== //
    let grad_unif = frechet_grad_norm(&raw, &uniform, &gam_mean);
    let grad_wt = frechet_grad_norm(&raw, &weights, &gam_wmean);

    let f_gam_unif = frechet_functional(&raw, &uniform, &gam_mean);
    let f_gam_wt = frechet_functional(&raw, &weights, &gam_wmean);

    // Deterministic perturbed competitors on the open simplex. Each must have a
    // STRICTLY larger Fréchet functional than gam's center (it is the unique
    // minimizer). We perturb in the clr tangent directions and re-close.
    let perturb = |center: &[f64], dir: usize, eps: f64| -> Vec<f64> {
        // multiply two coordinates up/down then re-close: a genuine move on the
        // simplex away from `center`.
        let mut v = center.to_vec();
        v[dir % d] *= 1.0 + eps;
        v[(dir + 1) % d] *= 1.0 - 0.5 * eps;
        close(&v)
    };
    let mut worst_margin_unif = f64::INFINITY; // min over competitors of (F_comp - F_gam)
    let mut worst_margin_wt = f64::INFINITY;
    for dir in 0..d {
        for &eps in &[0.05_f64, 0.15, -0.05, -0.15, 0.30] {
            let comp_u = perturb(&gam_mean, dir, eps);
            let comp_w = perturb(&gam_wmean, dir, eps);
            let fu = frechet_functional(&raw, &uniform, &comp_u);
            let fw = frechet_functional(&raw, &weights, &comp_w);
            worst_margin_unif = worst_margin_unif.min(fu - f_gam_unif);
            worst_margin_wt = worst_margin_wt.min(fw - f_gam_wt);
        }
    }

    // ===================================================================== //
    // (C) SIMPLEX-MEMBERSHIP STRUCTURE: sum-to-1 + strict positivity.
    // ===================================================================== //
    let sum_dev = (gam_mean.iter().sum::<f64>() - 1.0).abs();
    let wsum_dev = (gam_wmean.iter().sum::<f64>() - 1.0).abs();
    let min_entry = gam_mean.iter().cloned().fold(f64::INFINITY, f64::min);
    let wmin_entry = gam_wmean.iter().cloned().fold(f64::INFINITY, f64::min);

    // ---- mature BASELINE-TO-BEAT: R `compositions` -------------------------
    // `mean(acomp(X))` returns the closed geometric mean. We retain it ONLY as a
    // baseline on the OBJECTIVE Fréchet functional `F` (gam must be as-good-or-
    // better), and print the center rel-L2 for context. Agreement with the
    // reference center is NOT a pass criterion.
    let part0: Vec<f64> = raw.column(0).to_vec();
    let part1: Vec<f64> = raw.column(1).to_vec();
    let part2: Vec<f64> = raw.column(2).to_vec();
    let part3: Vec<f64> = raw.column(3).to_vec();
    let r = run_r(
        &[
            Column::new("p0", &part0),
            Column::new("p1", &part1),
            Column::new("p2", &part2),
            Column::new("p3", &part3),
        ],
        r#"
        suppressPackageStartupMessages(library(compositions))
        X <- as.matrix(df[, c("p0", "p1", "p2", "p3")])
        ctr <- as.numeric(mean(acomp(X)))      # closed geometric mean, sums to 1
        emit("center", ctr)
        "#,
    );
    let ref_center = r.vector("center");
    assert_eq!(ref_center.len(), d, "compositions center length mismatch");
    let ref_center_closed = close(ref_center); // guard against tiny round-off
    let f_ref_unif = frechet_functional(&raw, &uniform, &ref_center_closed);
    let ref_vs_truth = relative_l2(ref_center, &truth_mean);
    let gam_vs_ref = relative_l2(&gam_mean, ref_center);

    eprintln!(
        "aitchison center: n={n} d={d}\n  \
         (A) recovery rel_l2 vs exact closed-form: unif={rec_unif:.3e} weighted={rec_wt:.3e}\n  \
         (B) frechet grad norm: unif={grad_unif:.3e} weighted={grad_wt:.3e}; \
         F(gam) unif={f_gam_unif:.6e} weighted={f_gam_wt:.6e}; \
         worst competitor margin unif={worst_margin_unif:.3e} weighted={worst_margin_wt:.3e}\n  \
         (C) sum_dev unif={sum_dev:.3e} weighted={wsum_dev:.3e}; \
         min_entry unif={min_entry:.6} weighted={wmin_entry:.6}\n  \
         BASELINE compositions: F(ref)={f_ref_unif:.6e}; \
         ref_vs_truth_rel_l2={ref_vs_truth:.3e} gam_vs_ref_rel_l2={gam_vs_ref:.3e}"
    );

    // ---------------------------------------------------------------------- //
    // ASSERTIONS — objective quality only.
    // ---------------------------------------------------------------------- //

    // (A) gam reproduces the EXACT closed-form Aitchison center. The two paths
    // are the same exact arithmetic up to f64 round-off and gam's max-subtraction
    // log-sum-exp stabilization; 1e-12 relative is the genuine round-off floor.
    assert!(
        rec_unif < 1e-12,
        "gam center must equal the exact closed geometric mean: rel_l2={rec_unif:.3e}"
    );
    assert!(
        rec_wt < 1e-12,
        "gam weighted center must equal the exact weighted closed geometric mean: \
         rel_l2={rec_wt:.3e}"
    );

    // (B) gam's center is a Fréchet STATIONARY point: the analytic gradient in
    // the clr tangent space vanishes to round-off (scale-free certificate).
    assert!(
        grad_unif < 1e-12,
        "gam center is not Fréchet-stationary (uniform): grad_norm={grad_unif:.3e}"
    );
    assert!(
        grad_wt < 1e-12,
        "gam center is not Fréchet-stationary (weighted): grad_norm={grad_wt:.3e}"
    );
    // ... and it is the MINIMIZER: every perturbed competitor has strictly larger
    // Fréchet cost. (Margins are O(eps^2 * curvature) — comfortably > round-off.)
    assert!(
        worst_margin_unif > 1e-9,
        "gam center does not strictly minimize the Fréchet functional (uniform): \
         smallest competitor margin={worst_margin_unif:.3e}"
    );
    assert!(
        worst_margin_wt > 1e-9,
        "gam center does not strictly minimize the Fréchet functional (weighted): \
         smallest competitor margin={worst_margin_wt:.3e}"
    );

    // (C) gam's output lies on the open simplex.
    assert!(
        sum_dev < 1e-12 && wsum_dev < 1e-12,
        "gam center must sum to 1: unif sum_dev={sum_dev:.3e} weighted sum_dev={wsum_dev:.3e}"
    );
    assert!(
        min_entry > 0.0 && wmin_entry > 0.0,
        "gam center must be strictly positive: unif min={min_entry:.6} weighted min={wmin_entry:.6}"
    );

    // (BASELINE) match-or-beat the mature tool on the OBJECTIVE Fréchet
    // functional: gam's optimal center must cost no more than compositions'
    // center (allowing only round-off slack). This is NOT "match the reference
    // output" — it is "be at least as optimal as the reference".
    assert!(
        f_gam_unif <= f_ref_unif + 1e-12,
        "gam center must be as-good-or-better than compositions on the Fréchet \
         functional: F(gam)={f_gam_unif:.6e} F(ref)={f_ref_unif:.6e}"
    );
}
