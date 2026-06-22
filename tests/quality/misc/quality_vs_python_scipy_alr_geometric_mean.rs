//! Compositional (Aitchison-geometry) primitive quality test: gam's closed
//! geometric / Fréchet mean on the simplex.
//!
//! OBJECTIVE METRIC ASSERTED (primary): the Fréchet *variational optimality* of
//! gam's centroid. The compositional geometric mean is, by definition, the
//! minimiser of the Aitchison-Fréchet functional
//!
//!     F(m) = (1/n) Σ_i d_A(m, x_i)^2 ,
//!
//! where `d_A` is the Aitchison distance (Euclidean distance between
//! centred-log-ratio coordinates). "Quality" here is therefore not "looks like
//! some tool's output" — it is *does gam's point actually minimise the defining
//! objective*. We assert this two ways, both computed on gam's own returned
//! point against the identical sample:
//!   (P1) STATIONARITY / GLOBAL MINIMALITY: F(gam_mean) is a true minimum. The
//!        Fréchet functional on the simplex is strictly convex in clr space, so
//!        the minimiser is unique and any displacement must raise F. We assert
//!        F(gam_mean) <= F(candidate) for a battery of perturbed candidates
//!        (each observation, the arithmetic centroid, random simplex points),
//!        and that the analytic Fréchet gradient at gam_mean is ~0.
//!   (P2) MATCH-OR-BEAT THE REFERENCE ON THE OBJECTIVE: SciPy's closed-form
//!        `closure(gmean(X))` is fed the byte-identical sample; we assert
//!        F(gam_mean) <= F(scipy_mean) * (1 + tiny). gam's centroid is at least
//!        as good a minimiser of the true objective as the mature tool's — an
//!        accuracy claim about the objective, never "the coordinates match".
//!
//! Because the Aitchison-Fréchet minimiser admits the exact closed form
//! `closure(exp(mean_j log x))`, SciPy's `gmean` is *exact mathematical ground
//! truth* for the location of that minimiser. We additionally report (eprintln,
//! NOT asserted as the pass criterion) the coordinate deviation vs SciPy for
//! context, but the gate is the objective F, not coordinate closeness.
//!
//! INTRINSIC STRUCTURE also asserted (any correct compositional centroid must
//! satisfy these regardless of the optimiser used):
//!   (S1) closure validity — strictly positive parts summing to one;
//!   (S2) Aitchison perturbation-equivariance — perturbing every observation by
//!        a fixed composition `q` shifts the centroid by exactly `q`, the
//!        defining group-invariance of the Fréchet centre.
//!
//! Both engines are fed byte-identical data.

use gam::geometry::simplex::simplex_frechet_mean;
use gam::load_csvwith_inferred_schema;
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use ndarray::Array2;
use std::path::Path;

/// AFM volcanic-rock compositions from the Isle of Skye (Aitchison 1986, the
/// canonical reference dataset for the additive log-ratio; shipped as
/// `skye_afm_lavas.csv`). Each row is an (A=alkali, F=FeO, M=MgO) composition.
const SKYE_AFM_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/skye_afm_lavas.csv"
);

/// Additive log-ratio (ALR) coordinates of a composition with the LAST part as
/// the common reference: `alr(x)_j = ln(x_j / x_D)` for `j = 0..D-1`. This is
/// the bijection from the (D-1)-simplex to `R^{D-1}` underlying compositional
/// regression; squared Euclidean distance in ALR space is the (oblique) ALR
/// metric the geometric-mean identity is expressed in.
fn alr(x: &[f64]) -> Vec<f64> {
    let d = x.len();
    let denom = x[d - 1].ln();
    x[..d - 1].iter().map(|v| v.ln() - denom).collect()
}

/// Invert ALR back to a closed composition: append a 0 reference coordinate,
/// exponentiate, and close. `alr_inv(alr(x)) == x` for any strictly positive
/// composition `x`.
fn alr_inv(coords: &[f64]) -> Vec<f64> {
    let mut raw: Vec<f64> = coords.iter().map(|c| c.exp()).collect();
    raw.push(1.0);
    let total: f64 = raw.iter().sum();
    raw.iter().map(|v| v / total).collect()
}

/// Closure of a single composition row: divide by its total so it sums to 1.
fn close_row(parts: &[f64]) -> Vec<f64> {
    let total: f64 = parts.iter().sum();
    parts.iter().map(|p| p / total).collect()
}

/// Aitchison perturbation `x ⊕ q` = closure(x .* q): the simplex group
/// operation. Used to probe equivariance of the Fréchet centre.
fn perturb(x: &[f64], q: &[f64]) -> Vec<f64> {
    let prod: Vec<f64> = x.iter().zip(q).map(|(a, b)| a * b).collect();
    close_row(&prod)
}

/// Centred-log-ratio coordinates of a composition: `clr(x)_j = ln x_j - mean_k ln x_k`.
/// This is the isometry that maps Aitchison geometry into ordinary Euclidean
/// space, so squared Aitchison distance is just squared Euclidean distance here.
fn clr(x: &[f64]) -> Vec<f64> {
    let logs: Vec<f64> = x.iter().map(|v| v.ln()).collect();
    let mean = logs.iter().sum::<f64>() / logs.len() as f64;
    logs.iter().map(|l| l - mean).collect()
}

/// Squared Aitchison distance between two compositions = ||clr(a) - clr(b)||^2.
fn aitchison_sq(a: &[f64], b: &[f64]) -> f64 {
    let ca = clr(a);
    let cb = clr(b);
    ca.iter().zip(&cb).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Aitchison-Fréchet functional F(m) = mean_i d_A(m, x_i)^2 for candidate `m`.
fn frechet_objective(m: &[f64], rows: &[[f64; 4]]) -> f64 {
    let s: f64 = rows.iter().map(|r| aitchison_sq(m, r)).sum();
    s / rows.len() as f64
}

#[test]
fn simplex_geometric_mean_minimizes_aitchison_frechet_objective() {
    // ---- synthetic 4-part compositions, identical to both engines ---------
    // Rows are random [a, b, c, 1-a-b-c] with a,b,c ~ U(0.01, 0.98) drawn from
    // a fixed-seed deterministic generator, rejecting rows whose 4th part is
    // not strictly positive (so every part is strictly positive — required for
    // log stability and for the geometric mean to be defined). A SplitMix64
    // stream keeps the data reproducible and dependency-free.
    let nparts = 4usize;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_u01 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // map to (0,1)
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let lo = 0.01_f64;
    let hi = 0.98_f64;
    let target_rows = 200usize;

    let mut rows: Vec<[f64; 4]> = Vec::with_capacity(target_rows);
    while rows.len() < target_rows {
        let a = lo + (hi - lo) * next_u01();
        let b = lo + (hi - lo) * next_u01();
        let c = lo + (hi - lo) * next_u01();
        let d = 1.0 - a - b - c;
        // strict positivity of every part (log stability); reject otherwise.
        if d > 1e-6 && a > 0.0 && b > 0.0 && c > 0.0 {
            rows.push([a, b, c, d]);
        }
    }
    let n = rows.len();

    // Flatten into a row-major (n x 4) matrix for gam, and per-part column
    // vectors for the reference harness — both carry the SAME numbers.
    let mut mat = Array2::<f64>::zeros((n, nparts));
    let mut p0 = Vec::with_capacity(n);
    let mut p1 = Vec::with_capacity(n);
    let mut p2 = Vec::with_capacity(n);
    let mut p3 = Vec::with_capacity(n);
    for (i, r) in rows.iter().enumerate() {
        for j in 0..nparts {
            mat[[i, j]] = r[j];
        }
        p0.push(r[0]);
        p1.push(r[1]);
        p2.push(r[2]);
        p3.push(r[3]);
    }

    // ---- gam: closed geometric (Fréchet) mean on the simplex --------------
    let gam_mean = simplex_frechet_mean(mat.view(), None).expect("gam simplex Fréchet mean");
    assert_eq!(
        gam_mean.len(),
        nparts,
        "Fréchet mean must have one part per column"
    );

    // ---- SciPy closed-form reference: closure(gmean(X, axis=0)) -----------
    // scipy.stats.gmean is the column-wise geometric mean; closing it yields the
    // compositional geometric mean — the exact closed form of the
    // Aitchison-Fréchet minimiser. Used here as a BASELINE on the objective F.
    let r = run_python(
        &[
            Column::new("p0", &p0),
            Column::new("p1", &p1),
            Column::new("p2", &p2),
            Column::new("p3", &p3),
        ],
        r#"
from scipy.stats import gmean
X = np.column_stack([df["p0"], df["p1"], df["p2"], df["p3"]])
g = gmean(X, axis=0)          # exact column-wise geometric mean
g = g / g.sum()               # closure -> valid composition (sums to 1)
emit("mean", g)
"#,
    );
    let scipy_mean = r.vector("mean");
    assert_eq!(scipy_mean.len(), nparts, "scipy mean length mismatch");

    // ---- (P1) the centroid is the MINIMISER of the Fréchet objective ------
    // The defining quality of a Fréchet/geometric mean: it minimises the mean
    // squared Aitchison distance to the sample. We assert gam's point is at
    // least as good as a battery of competing candidates, and that no nearby
    // displacement lowers the objective (global minimality of a strictly
    // convex problem). This asserts gam's OBJECTIVE quality, not its likeness
    // to any tool.
    let f_gam = frechet_objective(&gam_mean, &rows);
    assert!(
        f_gam.is_finite() && f_gam >= 0.0,
        "Fréchet objective at gam mean is not a valid value: {f_gam}"
    );

    // Candidate set: every observation (the centroid must beat any single
    // member), the arithmetic centroid (closed mean of raw parts — a natural
    // but WRONG centre for Aitchison geometry), and a spread of random simplex
    // points. None may achieve a lower objective than gam's centroid.
    let mut arith = vec![0.0_f64; nparts];
    for r in &rows {
        for j in 0..nparts {
            arith[j] += r[j];
        }
    }
    let arith = close_row(&arith);

    let mut candidates: Vec<Vec<f64>> = Vec::new();
    for r in &rows {
        candidates.push(r.to_vec());
    }
    candidates.push(arith.clone());
    for _ in 0..64 {
        let raw = [
            lo + (hi - lo) * next_u01(),
            lo + (hi - lo) * next_u01(),
            lo + (hi - lo) * next_u01(),
            lo + (hi - lo) * next_u01(),
        ];
        candidates.push(close_row(&raw));
    }

    let mut worst_competitor = f64::INFINITY;
    for cand in &candidates {
        let f = frechet_objective(cand, &rows);
        if f < worst_competitor {
            worst_competitor = f;
        }
        // gam's centroid must not be beaten by any candidate (allow a tiny
        // floating-point slack only).
        assert!(
            f_gam <= f + 1e-12,
            "gam Fréchet mean is NOT the minimiser: F(gam)={f_gam:.12e} > F(candidate)={f:.12e}"
        );
    }
    // The arithmetic centroid is the wrong centre for Aitchison geometry; it
    // must score strictly worse, confirming the test discriminates the correct
    // optimum from a plausible-but-wrong one.
    let f_arith = frechet_objective(&arith, &rows);
    assert!(
        f_arith > f_gam + 1e-9,
        "arithmetic centroid did not score worse than the Aitchison centroid \
         (test cannot discriminate the correct optimum): F(arith)={f_arith:.12e} F(gam)={f_gam:.12e}"
    );

    // First-order stationarity: the gradient of F in clr coordinates at the
    // minimiser is 2*(clr(m) - mean_i clr(x_i)). gam's point must annihilate it.
    let clr_gam = clr(&gam_mean);
    let mut clr_bar = vec![0.0_f64; nparts];
    for r in &rows {
        let c = clr(r);
        for j in 0..nparts {
            clr_bar[j] += c[j];
        }
    }
    for v in clr_bar.iter_mut() {
        *v /= n as f64;
    }
    let grad_norm = clr_gam
        .iter()
        .zip(&clr_bar)
        .map(|(g, b)| (2.0 * (g - b)).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        grad_norm < 1e-10,
        "Fréchet gradient at gam mean is non-zero (not a stationary point): |grad|={grad_norm:.3e}"
    );

    // ---- (P2) MATCH-OR-BEAT the reference on the objective ----------------
    let f_scipy = frechet_objective(scipy_mean, &rows);
    let coord_dev = max_abs_diff(&gam_mean, scipy_mean);
    eprintln!(
        "simplex Fréchet: n={n} parts={nparts} F(gam)={f_gam:.12e} F(scipy)={f_scipy:.12e} \
         F(arith)={f_arith:.12e} best_competitor={worst_competitor:.12e} \
         |grad|={grad_norm:.3e} coord_dev_vs_scipy={coord_dev:.3e}"
    );
    // gam must be at least as good a minimiser of the TRUE objective as SciPy's
    // mature closed form. (Both sit at the analytic optimum, so the two F values
    // coincide to floating point; the 1e-9 relative slack guards rounding only
    // and would catch any genuine optimisation shortfall in gam.)
    assert!(
        f_gam <= f_scipy * (1.0 + 1e-9) + 1e-15,
        "gam Fréchet objective is worse than the SciPy baseline: \
         F(gam)={f_gam:.12e} > F(scipy)={f_scipy:.12e}"
    );

    // ---- (S1) intrinsic property: closure validity ------------------------
    let gam_sum: f64 = gam_mean.iter().sum();
    assert!(
        (gam_sum - 1.0).abs() < 1e-13,
        "gam Fréchet mean is not a closed composition: sum={gam_sum:.16}"
    );
    for (j, &v) in gam_mean.iter().enumerate() {
        assert!(
            v > 0.0 && v.is_finite(),
            "gam Fréchet mean part {j} is not strictly positive: {v}"
        );
    }

    // ---- (S2) intrinsic property: Aitchison perturbation-equivariance -----
    // The Fréchet centre is equivariant under the simplex group operation:
    //   mean(x_i ⊕ q) = mean(x_i) ⊕ q   for any composition q.
    // This is the defining invariance of the geometric mean as the Aitchison
    // centroid; a centroid that violated it would not be the Fréchet mean.
    let q = close_row(&[0.4, 0.1, 0.3, 0.2]);
    let mut mat_pert = Array2::<f64>::zeros((n, nparts));
    for (i, r) in rows.iter().enumerate() {
        let pr = perturb(r, &q);
        for j in 0..nparts {
            mat_pert[[i, j]] = pr[j];
        }
    }
    let gam_mean_pert =
        simplex_frechet_mean(mat_pert.view(), None).expect("gam Fréchet mean of perturbed sample");
    let expected_pert = perturb(&gam_mean, &q);
    let equiv_dev = max_abs_diff(&gam_mean_pert, &expected_pert);
    eprintln!(
        "perturbation-equivariance: q={q:?} mean_of_perturbed={gam_mean_pert:?} \
         perturbed_mean={expected_pert:?} dev={equiv_dev:.3e}"
    );
    // Exact algebraic identity up to floating point; 1e-12 catches any real
    // break in the group-equivariance of the centroid.
    assert!(
        equiv_dev < 1e-12,
        "gam Fréchet mean violates Aitchison perturbation-equivariance: dev={equiv_dev:.3e}"
    );
}

/// Real-data arm: gam's compositional geometric mean on the canonical Skye AFM
/// lavas (Aitchison 1986), exercising the SAME `simplex_frechet_mean` capability
/// the synthetic test above proves on known-truth data. Real data => the true
/// centre is unknown, so we assert OBJECTIVE held-out quality instead of recovery:
///
///   PRIMARY (ALR-transform correctness + geometric-mean identity, tool-free):
///     gam's centroid on the TRAIN rows must satisfy the defining ALR identity —
///     its additive-log-ratio coordinates equal the column-wise mean of the train
///     rows' ALR coordinates — and round-trip exactly through `alr`/`alr_inv`.
///     This is an exact mathematical property of the compositional geometric mean.
///
///   PREDICTIVE (objective, tool-free): treating gam's TRAIN centroid as a
///     constant predictor of held-out compositions, its mean squared Aitchison
///     (clr) distance to the TEST rows must clear an absolute bar AND beat the
///     arithmetic (closure-of-mean-parts) centroid — the natural but
///     geometrically WRONG centre — on that SAME held-out metric.
///
///   BASELINE (match-or-beat): SciPy's `closure(gmean(train))` is fed the
///     byte-identical TRAIN rows; gam's held-out mean-sq-Aitchison risk must be no
///     worse than SciPy's * (1 + tiny). SciPy is a baseline to match-or-beat on
///     the objective, never a fitted target to reproduce.
#[test]
fn simplex_geometric_mean_minimizes_aitchison_frechet_objective_on_real_data() {
    // ---- load the canonical Skye AFM lavas (A, F, M strictly positive) -----
    let ds =
        load_csvwith_inferred_schema(Path::new(SKYE_AFM_CSV)).expect("load skye_afm_lavas.csv");
    let col = ds.column_map();
    let a_idx = col["A"];
    let f_idx = col["F"];
    let m_idx = col["M"];
    let a_col: Vec<f64> = ds.values.column(a_idx).to_vec();
    let f_col: Vec<f64> = ds.values.column(f_idx).to_vec();
    let m_col: Vec<f64> = ds.values.column(m_idx).to_vec();
    let n = a_col.len();
    assert!(n >= 20, "skye AFM should have ~23 rows, got {n}");

    let nparts = 3usize;
    // Closed AFM compositions (parts sum to 1), one [A, F, M] row each. Every
    // part is strictly positive in this dataset, so the log-ratios are defined.
    let rows: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let raw = [a_col[i], f_col[i], m_col[i]];
            let total: f64 = raw.iter().sum();
            assert!(
                raw.iter().all(|&v| v > 0.0) && total > 0.0,
                "AFM row {i} is not strictly positive: {raw:?}"
            );
            [raw[0] / total, raw[1] / total, raw[2] / total]
        })
        .collect();

    // ---- deterministic train/test split: every 3rd row held out -----------
    let is_test = |i: usize| i % 3 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() >= 12 && test_rows.len() >= 6,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    // Train compositions: matrix for gam, and per-part columns (TRAIN length) for
    // the reference harness — byte-identical numbers, same order.
    let mut train_mat = Array2::<f64>::zeros((train_rows.len(), nparts));
    let mut a_tr = Vec::with_capacity(train_rows.len());
    let mut f_tr = Vec::with_capacity(train_rows.len());
    let mut m_tr = Vec::with_capacity(train_rows.len());
    for (out_row, &src) in train_rows.iter().enumerate() {
        let r = rows[src];
        for j in 0..nparts {
            train_mat[[out_row, j]] = r[j];
        }
        a_tr.push(r[0]);
        f_tr.push(r[1]);
        m_tr.push(r[2]);
    }

    // ---- gam: compositional geometric (Fréchet) mean on TRAIN -------------
    let gam_mean = simplex_frechet_mean(train_mat.view(), None).expect("gam simplex Fréchet mean");
    assert_eq!(
        gam_mean.len(),
        nparts,
        "Fréchet mean must have one part per column"
    );

    // ---- (PRIMARY) ALR-transform correctness + geometric-mean identity -----
    // The compositional geometric mean's ALR coordinates equal the arithmetic
    // mean of the train rows' ALR coordinates: alr(g) = mean_i alr(x_i). This is
    // an exact identity (alr is a log-linear map and g is the closure of the
    // column-wise geometric mean). It both validates the alr transform and
    // certifies gam returned the geometric mean.
    let mut alr_bar = vec![0.0_f64; nparts - 1];
    for &src in &train_rows {
        let coords = alr(&rows[src]);
        for j in 0..nparts - 1 {
            alr_bar[j] += coords[j];
        }
    }
    for v in alr_bar.iter_mut() {
        *v /= train_rows.len() as f64;
    }
    let alr_gam = alr(&gam_mean);
    let alr_identity_dev = max_abs_diff(&alr_gam, &alr_bar);
    assert!(
        alr_identity_dev < 1e-12,
        "ALR geometric-mean identity broken: alr(gam_mean)={alr_gam:?} vs \
         mean_i alr(x_i)={alr_bar:?} dev={alr_identity_dev:.3e}"
    );
    // alr round-trip: alr_inv(alr_bar) reconstructs gam's centroid exactly.
    let reconstructed = alr_inv(&alr_bar);
    let roundtrip_dev = max_abs_diff(&reconstructed, &gam_mean);
    assert!(
        roundtrip_dev < 1e-12,
        "ALR inverse failed to reconstruct the geometric mean: dev={roundtrip_dev:.3e}"
    );

    // ---- held-out predictive risk of the TRAIN centroid -------------------
    // The centroid predicts every held-out composition; its quality is the mean
    // squared Aitchison (clr) distance to the TEST rows. Computed in plain Rust.
    let test_comps: Vec<[f64; 3]> = test_rows.iter().map(|&i| rows[i]).collect();
    let heldout_risk = |centre: &[f64]| -> f64 {
        let s: f64 = test_comps.iter().map(|r| aitchison_sq(centre, r)).sum();
        s / test_comps.len() as f64
    };
    let gam_heldout = heldout_risk(&gam_mean);

    // Arithmetic centroid of the TRAIN parts: the natural but geometrically WRONG
    // centre for Aitchison data. gam's geometric mean must predict the held-out
    // compositions strictly better than it.
    let mut arith = vec![0.0_f64; nparts];
    for &src in &train_rows {
        for j in 0..nparts {
            arith[j] += rows[src][j];
        }
    }
    let arith = close_row(&arith);
    let arith_heldout = heldout_risk(&arith);

    // ---- SciPy baseline: closure(gmean(TRAIN)) on byte-identical rows ------
    let r = run_python(
        &[
            Column::new("A", &a_tr),
            Column::new("F", &f_tr),
            Column::new("M", &m_tr),
        ],
        r#"
from scipy.stats import gmean
X = np.column_stack([df["A"], df["F"], df["M"]])
g = gmean(X, axis=0)          # exact column-wise geometric mean of TRAIN
g = g / g.sum()               # closure -> valid composition (sums to 1)
emit("mean", g)
"#,
    );
    let scipy_mean = r.vector("mean");
    assert_eq!(scipy_mean.len(), nparts, "scipy mean length mismatch");
    let scipy_heldout = heldout_risk(scipy_mean);
    let coord_dev = max_abs_diff(&gam_mean, scipy_mean);

    eprintln!(
        "skye AFM held-out: n={n} n_train={} n_test={} gam_mean={gam_mean:?} \
         alr_identity_dev={alr_identity_dev:.3e} roundtrip_dev={roundtrip_dev:.3e} \
         gam_heldout={gam_heldout:.6e} arith_heldout={arith_heldout:.6e} \
         scipy_heldout={scipy_heldout:.6e} coord_dev_vs_scipy={coord_dev:.3e}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- ABSOLUTE objective bar -------------------------------------------
    // The AFM trend is real and compact in Aitchison space; the geometric centre
    // sits within mean squared clr-distance 1.0 of the held-out lavas (the parts
    // span well under one e-fold of spread). A degenerate centre would blow past
    // this.
    assert!(
        gam_heldout.is_finite() && gam_heldout >= 0.0,
        "gam held-out Aitchison risk is not a valid value: {gam_heldout}"
    );
    assert!(
        gam_heldout < 1.0,
        "gam held-out Aitchison risk too high: {gam_heldout:.6e} (>= 1.0)"
    );

    // ---- DISCRIMINATION: beat the geometrically wrong arithmetic centre ----
    assert!(
        gam_heldout < arith_heldout,
        "geometric mean did not predict held-out lavas better than the arithmetic \
         centroid (test cannot discriminate the correct centre): \
         gam={gam_heldout:.6e} arith={arith_heldout:.6e}"
    );

    // ---- BASELINE (match-or-beat) on the SAME held-out metric -------------
    // gam must be at least as good a held-out predictor as SciPy's mature closed
    // form. Both sit at the analytic geometric mean, so the risks coincide to
    // floating point; the slack guards rounding and would catch any genuine
    // shortfall in gam's centroid.
    assert!(
        gam_heldout <= scipy_heldout * (1.0 + 1e-9) + 1e-15,
        "gam held-out Aitchison risk worse than the SciPy baseline: \
         gam={gam_heldout:.6e} > scipy={scipy_heldout:.6e}"
    );
}
