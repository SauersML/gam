//! End-to-end quality on a REAL compositional dataset: the AFM (alkali / iron-
//! oxide / magnesium-oxide) percentages of 23 aphyric Skye lavas — the textbook
//! ternary geochemistry example from Aitchison, *The Statistical Analysis of
//! Compositional Data* (1986). Each lava is a 3-part composition that sums to a
//! constant whole (100%), so it lives on the 2-simplex and Aitchison geometry
//! (CLR / closed geometric mean) is exactly the right tool.
//!
//! Source (raw, no auth): MASS::Skye via Rdatasets —
//!   https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Skye.csv
//! The identical 23 rows, in identical order, are fed to gam and to the mature R
//! `compositions` package; columns are A, F, M.
//!
//! REALISTIC USE-CASE: a petrologist asks for the *representative* composition of
//! a lava suite. The arithmetic mean of percentages is biased on the simplex
//! (it ignores the constant-sum constraint and the relative scale of the parts);
//! the Aitchison center — gam's `simplex_frechet_mean`, the closed weighted
//! geometric mean — is the correct centre of a compositional cloud. This test
//! asserts gam computes that centre correctly and that the CLR coordinates gam's
//! Aitchison geometry implies match the mature reference to machine precision.
//!
//! OBJECTIVE QUALITY ASSERTED (never "gam reproduces a peer tool" as the bar):
//!
//!   (A) GROUND-TRUTH RECOVERY. The Aitchison center has an EXACT closed form,
//!       `clo(exp(mean_i log c_i))` on the closed rows `c_i`. We compute it
//!       independently in plain Rust on the real data and assert gam reproduces
//!       it to f64 round-off. Matching an exact analytic quantity is correctness.
//!
//!   (B) FRÉCHET OPTIMALITY (the defining variational property). The Aitchison
//!       center is the unique minimizer of `F(m) = mean_i d_A(c_i, m)^2` over the
//!       open simplex (d_A = Aitchison distance). We assert gam's output is a
//!       stationary point (the analytic CLR-tangent Fréchet gradient vanishes)
//!       AND strictly beats a battery of deterministic perturbed competitors on
//!       `F`. This is gam's own optimality, computed on gam's own output.
//!
//!   (C) SIMPLEX-MEMBERSHIP STRUCTURE. gam's output must lie on the open simplex:
//!       sum to 1 (round-off) and be strictly positive. Asserted on the output.
//!
//!   (D) CLR ROUND-TRIP vs GROUND TRUTH. The centered-log-ratio is a deterministic
//!       analytic map; `compositions::clr` is its mature reference implementation.
//!       We assert gam's Aitchison-geometry CLR of the recovered center matches
//!       `compositions::clr` to machine precision, and that the CLR round-trips
//!       back through the inverse (closed softmax) to the center to round-off.
//!
//! The mature R `compositions` package is retained as (i) GROUND TRUTH for the
//! exact CLR map (matching it IS accuracy of the transform) and (ii) a BASELINE
//! on the objective Fréchet functional: gam's `F` must be <= compositions' `F`
//! (up to round-off). Agreement with the reference *center* is printed for
//! context but is NOT itself a pass criterion.

use gam::geometry::simplex::simplex_frechet_mean;
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_r};
use ndarray::Array2;
use std::path::Path;

const SKYE_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/skye_afm_lavas.csv"
);

/// Closure (sum-to-1 projection) of a single strictly-positive vector.
fn close(v: &[f64]) -> Vec<f64> {
    let s: f64 = v.iter().sum();
    v.iter().map(|x| x / s).collect()
}

/// Centred-log-ratio of a closed strictly-positive composition:
/// `clr(c)_j = log c_j - (1/d) sum_k log c_k`. The CLR maps the simplex
/// isometrically into the hyperplane `sum = 0` of R^d with the Euclidean metric,
/// turning Aitchison geometry into plain linear algebra.
fn clr(c: &[f64]) -> Vec<f64> {
    let logs: Vec<f64> = c.iter().map(|x| x.ln()).collect();
    let mean = logs.iter().sum::<f64>() / logs.len() as f64;
    logs.iter().map(|l| l - mean).collect()
}

/// Inverse CLR (closed softmax of a centered log-vector): maps a `sum = 0` CLR
/// point back onto the open simplex.
fn clr_inv(z: &[f64]) -> Vec<f64> {
    let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
    close(&exps)
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

/// Uniform-weight Fréchet functional `F(m) = (1/n) sum_i d_A(closed_row_i, m)^2`
/// over the closed rows of `raw`.
fn frechet_functional(raw: &Array2<f64>, m: &[f64]) -> f64 {
    let n = raw.nrows();
    let mut f = 0.0;
    for i in 0..n {
        let row = close(&raw.row(i).to_vec());
        f += aitchison_sq(&row, m);
    }
    f / n as f64
}

/// Analytic Fréchet gradient norm of `F` in the CLR tangent space (hyperplane
/// `sum = 0`). For `F(m) = mean_i || clr(c_i) - clr(m) ||^2` the gradient w.r.t.
/// the CLR coordinates of `m` is `2 (clr(m) - mean_i clr(c_i))`; at the Aitchison
/// center it is exactly zero, so its norm is a scale-free stationarity certificate.
fn frechet_grad_norm(raw: &Array2<f64>, m: &[f64]) -> f64 {
    let n = raw.nrows();
    let d = m.len();
    let mut bary = vec![0.0; d];
    for i in 0..n {
        let cr = clr(&close(&raw.row(i).to_vec()));
        for j in 0..d {
            bary[j] += cr[j] / n as f64;
        }
    }
    let cm = clr(m);
    cm.iter()
        .zip(&bary)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Exact closed-form Aitchison center: `clo(exp(mean_i log c_i))` over the closed
/// rows. This IS the analytic ground truth gam must reproduce.
fn closed_geometric_mean(raw: &Array2<f64>) -> Vec<f64> {
    let n = raw.nrows();
    let d = raw.ncols();
    let mut mean_log = vec![0.0; d];
    for i in 0..n {
        let row = close(&raw.row(i).to_vec());
        for j in 0..d {
            mean_log[j] += row[j].ln() / n as f64;
        }
    }
    let g: Vec<f64> = mean_log.iter().map(|l| l.exp()).collect();
    close(&g)
}

/// Minimal CSV loader for the 4-column Skye file (`rownames,A,F,M`). Returns the
/// (A, F, M) percentage columns in file order. No external schema machinery so
/// the exact rows and order handed to gam and to R are guaranteed identical.
fn load_skye(path: &Path) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let text = std::fs::read_to_string(path).expect("read skye_afm_lavas.csv");
    let mut a = Vec::new();
    let mut f = Vec::new();
    let mut m = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            assert_eq!(
                line.trim(),
                "rownames,A,F,M",
                "unexpected Skye header: {line:?}"
            );
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        assert_eq!(cols.len(), 4, "Skye row must have 4 fields: {line:?}");
        a.push(cols[1].trim().parse::<f64>().expect("parse A"));
        f.push(cols[2].trim().parse::<f64>().expect("parse F"));
        m.push(cols[3].trim().parse::<f64>().expect("parse M"));
    }
    (a, f, m)
}

#[test]
fn gam_aitchison_center_of_skye_afm_lavas_is_optimal_and_clr_exact() {
    // ---- load the real Skye AFM compositional dataset (A, F, M percentages) ---
    let (a, f, m) = load_skye(Path::new(SKYE_CSV));
    let n = a.len();
    let d = 3usize;
    assert_eq!(n, 23, "Skye AFM dataset should have 23 lavas, got {n}");

    // Verify the constant-sum (compositional) structure and strict positivity on
    // the real data — these are the properties that make Aitchison geometry the
    // correct tool. The raw percentages already sum to 100.
    let mut raw = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        let s = a[i] + f[i] + m[i];
        assert!(
            (s - 100.0).abs() < 1e-9,
            "Skye row {i} parts must sum to 100%, got {s}"
        );
        assert!(
            a[i] > 0.0 && f[i] > 0.0 && m[i] > 0.0,
            "Skye row {i} parts must be strictly positive: A={} F={} M={}",
            a[i],
            f[i],
            m[i]
        );
        raw[[i, 0]] = a[i];
        raw[[i, 1]] = f[i];
        raw[[i, 2]] = m[i];
    }

    // ---- gam output: the Aitchison center of the lava suite -------------------
    let gam_mean = simplex_frechet_mean(raw.view(), None).expect("gam simplex_frechet_mean");
    assert_eq!(
        gam_mean.len(),
        d,
        "gam center must have 3 components (A,F,M)"
    );

    // ===================================================================== //
    // (A) GROUND-TRUTH RECOVERY: match the exact closed-form Aitchison center
    //     computed independently in plain Rust on the real data.
    // ===================================================================== //
    let truth_mean = closed_geometric_mean(&raw);
    let rec_rel = relative_l2(&gam_mean, &truth_mean);

    // ===================================================================== //
    // (B) FRÉCHET OPTIMALITY: stationarity + strictly-beats perturbations.
    // ===================================================================== //
    let grad = frechet_grad_norm(&raw, &gam_mean);
    let f_gam = frechet_functional(&raw, &gam_mean);

    // Deterministic perturbed competitors on the open simplex: scale two parts
    // up/down and re-close, a genuine move away from the center. Each must have a
    // STRICTLY larger Fréchet functional (the center is the unique minimizer).
    let perturb = |center: &[f64], dir: usize, eps: f64| -> Vec<f64> {
        let mut v = center.to_vec();
        v[dir % d] *= 1.0 + eps;
        v[(dir + 1) % d] *= 1.0 - 0.5 * eps;
        close(&v)
    };
    let mut worst_margin = f64::INFINITY; // min over competitors of (F_comp - F_gam)
    for dir in 0..d {
        for &eps in &[0.05_f64, 0.15, -0.05, -0.15, 0.30] {
            let comp = perturb(&gam_mean, dir, eps);
            worst_margin = worst_margin.min(frechet_functional(&raw, &comp) - f_gam);
        }
    }

    // ===================================================================== //
    // (C) SIMPLEX-MEMBERSHIP STRUCTURE: sum-to-1 + strict positivity.
    // ===================================================================== //
    let sum_dev = (gam_mean.iter().sum::<f64>() - 1.0).abs();
    let min_entry = gam_mean.iter().cloned().fold(f64::INFINITY, f64::min);

    // ===================================================================== //
    // (D) CLR ROUND-TRIP: gam's Aitchison CLR of the center round-trips back to
    //     the center through the inverse (closed softmax), to round-off.
    // ===================================================================== //
    let gam_clr = clr(&gam_mean);
    let clr_sum = gam_clr.iter().sum::<f64>().abs(); // CLR lives on sum = 0
    let roundtrip = clr_inv(&gam_clr);
    let roundtrip_err = max_abs_diff(&roundtrip, &gam_mean);

    // ---- mature reference: R `compositions` -----------------------------------
    // `mean(acomp(X))` is the closed geometric-mean center (objective Fréchet
    // BASELINE), and `clr(.)` is the exact analytic CLR map (GROUND TRUTH for the
    // transform). Identical 23 rows, identical order.
    let r = run_r(
        &[
            Column::new("A", &a),
            Column::new("F", &f),
            Column::new("M", &m),
        ],
        r#"
        suppressPackageStartupMessages(library(compositions))
        X <- as.matrix(df[, c("A", "F", "M")])
        ac <- acomp(X)                               # closed Aitchison compositions
        ctr <- as.numeric(mean(ac))                  # closed geometric-mean center
        ctr <- ctr / sum(ctr)                        # unit closure for comparison
        emit("center", ctr)
        # Exact CLR of the center (analytic ground truth for the transform).
        clr_ctr <- as.numeric(clr(acomp(matrix(ctr, nrow = 1))))
        emit("clr_center", clr_ctr)
        "#,
    );
    let ref_center = r.vector("center");
    let ref_clr = r.vector("clr_center");
    assert_eq!(ref_center.len(), d, "compositions center length mismatch");
    assert_eq!(ref_clr.len(), d, "compositions clr length mismatch");

    let ref_center_closed = close(ref_center); // guard tiny round-off
    let f_ref = frechet_functional(&raw, &ref_center_closed);
    let clr_err = max_abs_diff(&gam_clr, ref_clr);
    let gam_vs_ref = relative_l2(&gam_mean, ref_center);

    eprintln!(
        "Skye AFM Aitchison center: n={n} d={d}\n  \
         gam center A,F,M = [{:.6}, {:.6}, {:.6}]\n  \
         (A) recovery rel_l2 vs exact closed-form: {rec_rel:.3e}\n  \
         (B) frechet grad norm: {grad:.3e}; F(gam)={f_gam:.6e}; \
         worst competitor margin={worst_margin:.3e}\n  \
         (C) sum_dev={sum_dev:.3e}; min_entry={min_entry:.6}\n  \
         (D) clr_sum(|.|)={clr_sum:.3e}; clr round-trip max_abs={roundtrip_err:.3e}; \
         clr_err vs compositions={clr_err:.3e}\n  \
         BASELINE compositions: F(ref)={f_ref:.6e}; gam_vs_ref_rel_l2={gam_vs_ref:.3e}",
        gam_mean[0], gam_mean[1], gam_mean[2]
    );

    // ---------------------------------------------------------------------- //
    // ASSERTIONS — objective quality only.
    // ---------------------------------------------------------------------- //

    // (A) gam reproduces the EXACT closed-form Aitchison center on the real data.
    // Same exact arithmetic up to f64 round-off and gam's log-sum-exp max-shift.
    assert!(
        rec_rel < 1e-12,
        "gam center must equal the exact closed geometric mean of the Skye lavas: \
         rel_l2={rec_rel:.3e}"
    );

    // (B) gam's center is Fréchet-stationary (CLR-tangent gradient vanishes) ...
    assert!(
        grad < 1e-12,
        "gam center is not Fréchet-stationary on Skye AFM: grad_norm={grad:.3e}"
    );
    // ... and is the MINIMIZER: every perturbed competitor costs strictly more.
    // Margins are O(eps^2 * curvature), comfortably above round-off.
    assert!(
        worst_margin > 1e-9,
        "gam center does not strictly minimize the Fréchet functional on Skye AFM: \
         smallest competitor margin={worst_margin:.3e}"
    );

    // (C) gam's output lies on the open simplex.
    assert!(
        sum_dev < 1e-12,
        "gam center must sum to 1: sum_dev={sum_dev:.3e}"
    );
    assert!(
        min_entry > 0.0,
        "gam center must be strictly positive: min_entry={min_entry:.6}"
    );

    // (D) gam's Aitchison CLR is the exact analytic CLR map: it matches the mature
    // `compositions::clr` to machine precision, sits on the sum = 0 hyperplane,
    // and round-trips back to the center through the closed softmax inverse.
    assert!(
        clr_err < 1e-10,
        "gam CLR of the center is not the exact analytic CLR (vs compositions): \
         max_abs={clr_err:.3e}"
    );
    assert!(
        clr_sum < 1e-12,
        "gam CLR must lie on the sum = 0 hyperplane: |sum|={clr_sum:.3e}"
    );
    assert!(
        roundtrip_err < 1e-12,
        "CLR round-trip (center -> clr -> closed softmax) must recover the center: \
         max_abs={roundtrip_err:.3e}"
    );

    // (BASELINE) match-or-beat the mature tool on the OBJECTIVE Fréchet functional:
    // gam's optimal center must cost no more than compositions' center (round-off
    // slack only). This is "be at least as optimal", not "match the reference".
    assert!(
        f_gam <= f_ref + 1e-12,
        "gam center must be as-good-or-better than compositions on the Fréchet \
         functional: F(gam)={f_gam:.6e} F(ref)={f_ref:.6e}"
    );
}
