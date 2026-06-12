//! Oracle: the streaming band-assembled K×K tensor-product B-spline smoother
//! must reproduce, to near machine precision, the SAME penalized system built
//! independently in-test — self-constructed truth (#904): naive O(n·p²) dense
//! normal-equation accumulation, naive dense Gauss–Legendre penalty assembly,
//! and in-test Gaussian elimination. Agreement at 1e-8 proves the scatter-add
//! and band assembly are exact (no approximation tolerance budget — both
//! paths compute the same finite-dimensional Gaussian).
//!
//! Three arms:
//! 1. exactness — coefficients, fitted means and prediction variances match
//!    the dense oracle at fixed (λ, σ²);
//! 2. truth recovery — REML-selected fit on a smooth surface + fixed
//!    quasi-random noise beats the noise floor sanely;
//! 3. penalty correctness — for f = x1² + x1·x2 + x2² (constant second
//!    derivatives) the assembled J(f) equals the closed-form integral
//!    (4a1² + 2a1a2 + 4a2²)·Area to 1e-8, which a dropped mixed term
//!    (the axis-wise P-spline shortcut) would miss by exactly 2a1a2·Area.

use gam::solver::grid_spline_2d::{GridSpline2dDesign, fit_grid_spline_2d_at};

// ───────────────────────── in-test dense linear algebra ─────────────────────

/// Dense in-test Gaussian elimination solve A·X = B (partial pivoting).
fn dense_solve(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend_from_slice(&b[i]);
            row
        })
        .collect();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap();
        aug.swap(col, piv);
        let p = aug[col][col];
        assert!(p.abs() > 1e-300, "dense oracle: singular pivot");
        for i in 0..n {
            if i == col {
                continue;
            }
            let f = aug[i][col] / p;
            if f == 0.0 {
                continue;
            }
            for k in col..n + m {
                aug[i][k] -= f * aug[col][k];
            }
        }
    }
    (0..n)
        .map(|i| (0..m).map(|j| aug[i][n + j] / aug[i][i]).collect())
        .collect()
}

// ─────────────── in-test re-statement of the basis definition ───────────────
// The basis (uniform extended knots, cardinal cubic segments) is part of the
// model definition shared by both paths; what the oracle re-derives
// INDEPENDENTLY is the assembly (dense accumulation vs streaming band
// scatter-add) and the solve (Gaussian elimination vs Cholesky).

fn bval(u: f64) -> [f64; 4] {
    let v = 1.0 - u;
    [
        v * v * v / 6.0,
        (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
        (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0,
        u * u * u / 6.0,
    ]
}

fn bd1(u: f64) -> [f64; 4] {
    let v = 1.0 - u;
    [
        -0.5 * v * v,
        0.5 * (3.0 * u * u - 4.0 * u),
        0.5 * (-3.0 * u * u + 2.0 * u + 1.0),
        0.5 * u * u,
    ]
}

fn bd2(u: f64) -> [f64; 4] {
    [1.0 - u, 3.0 * u - 2.0, 1.0 - 3.0 * u, u]
}

fn locate(lo: f64, h: f64, cells: usize, x: f64) -> (usize, f64) {
    let t = (x - lo) / h;
    let cell = (t.floor().max(0.0) as usize).min(cells - 1);
    (cell, t - cell as f64)
}

/// Full dense p-length basis row at (x1, x2).
fn dense_row(lo: [f64; 2], h: [f64; 2], k: usize, x1: f64, x2: f64) -> Vec<f64> {
    let m = k + 3;
    let (c1, u1) = locate(lo[0], h[0], k, x1);
    let (c2, u2) = locate(lo[1], h[1], k, x2);
    let (b1, b2) = (bval(u1), bval(u2));
    let mut row = vec![0.0_f64; m * m];
    for i in 0..4 {
        for j in 0..4 {
            row[(c1 + i) * m + (c2 + j)] = b1[i] * b2[j];
        }
    }
    row
}

const GL4_NODES: [f64; 4] = [
    -0.861_136_311_594_052_6,
    -0.339_981_043_584_856_26,
    0.339_981_043_584_856_26,
    0.861_136_311_594_052_6,
];
const GL4_WEIGHTS: [f64; 4] = [
    0.347_854_845_137_453_85,
    0.652_145_154_862_546_2,
    0.652_145_154_862_546_2,
    0.347_854_845_137_453_85,
];

/// Naive dense assembly of S = ∫ a1²·B11 B11ᵀ + 2a1a2·B12 B12ᵀ + a2²·B22 B22ᵀ
/// by 4-point Gauss–Legendre per axis per cell (exact: per-axis integrand
/// degree ≤ 6 < 8 — see the engine module header).
fn dense_penalty(h: [f64; 2], k: usize, a: [f64; 2]) -> Vec<Vec<f64>> {
    let m = k + 3;
    let p = m * m;
    let (s11, s12, s22) = (a[0] * a[0], 2.0 * a[0] * a[1], a[1] * a[1]);
    let mut s = vec![vec![0.0_f64; p]; p];
    for c1 in 0..k {
        for c2 in 0..k {
            for q1 in 0..4 {
                for q2 in 0..4 {
                    let u1 = 0.5 * (1.0 + GL4_NODES[q1]);
                    let u2 = 0.5 * (1.0 + GL4_NODES[q2]);
                    let wq = 0.25 * h[0] * h[1] * GL4_WEIGHTS[q1] * GL4_WEIGHTS[q2];
                    let (v1, d1, dd1) = (bval(u1), bd1(u1), bd2(u1));
                    let (v2, d2, dd2) = (bval(u2), bd1(u2), bd2(u2));
                    let mut g11 = vec![0.0_f64; p];
                    let mut g12 = vec![0.0_f64; p];
                    let mut g22 = vec![0.0_f64; p];
                    for i in 0..4 {
                        for j in 0..4 {
                            let g = (c1 + i) * m + (c2 + j);
                            g11[g] = dd1[i] / (h[0] * h[0]) * v2[j];
                            g12[g] = d1[i] / h[0] * d2[j] / h[1];
                            g22[g] = v1[i] * dd2[j] / (h[1] * h[1]);
                        }
                    }
                    for r in 0..p {
                        if g11[r] == 0.0 && g12[r] == 0.0 && g22[r] == 0.0 {
                            continue;
                        }
                        for c in 0..p {
                            s[r][c] += wq
                                * (s11 * g11[r] * g11[c]
                                    + s12 * g12[r] * g12[c]
                                    + s22 * g22[r] * g22[c]);
                        }
                    }
                }
            }
        }
    }
    s
}

// ───────────────────────────── deterministic data ───────────────────────────

/// 2-D Kronecker (plastic-constant) low-discrepancy points on a box, smooth
/// truth + fixed golden-ratio noise. No RNG anywhere.
fn test_data(n: usize, lo: [f64; 2], hi: [f64; 2], noise_amp: f64) -> Vec<[f64; 4]> {
    let a1 = 0.754_877_666_246_692_7; // 1/ρ, ρ³ = ρ + 1
    let a2 = 0.569_840_290_998_053_2; // 1/ρ²
    (0..n)
        .map(|i| {
            let u1 = ((i + 1) as f64 * a1).fract();
            let u2 = ((i + 1) as f64 * a2).fract();
            let x1 = lo[0] + (hi[0] - lo[0]) * u1;
            let x2 = lo[1] + (hi[1] - lo[1]) * u2;
            let truth = (3.0 * x1).sin() * (2.0 * x2).cos() + 0.4 * x1 * x2;
            let noise = ((i as f64 * 0.618_033_988_749_894_9).fract() - 0.5) * 2.0 * noise_amp;
            let w = 1.0 + 0.5 * ((i % 4) as f64);
            [x1, x2, truth + noise, w]
        })
        .collect()
}

fn split(rows: &[[f64; 4]]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        rows.iter().map(|r| r[0]).collect(),
        rows.iter().map(|r| r[1]).collect(),
        rows.iter().map(|r| r[2]).collect(),
        rows.iter().map(|r| r[3]).collect(),
    )
}

// ──────────────────────────────── arm 1: exactness ──────────────────────────

#[test]
fn streaming_band_assembly_matches_dense_oracle() {
    let k = 4usize;
    let metric = [1.3_f64, 0.7];
    let rows = test_data(160, [0.2, -0.5], [1.7, 0.9], 0.2);
    let (x1, x2, y, w) = split(&rows);
    let design = GridSpline2dDesign::build(&x1, &x2, &y, &w, k, metric).expect("design build");

    // Independent bounding box — must agree with the engine's convention.
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for r in &rows {
        for ax in 0..2 {
            lo[ax] = lo[ax].min(r[ax]);
            hi[ax] = hi[ax].max(r[ax]);
        }
    }
    let h = [(hi[0] - lo[0]) / k as f64, (hi[1] - lo[1]) / k as f64];
    let eng_lo = design.lower_corner();
    let eng_h = design.cell_widths();
    for ax in 0..2 {
        assert!(
            (eng_lo[ax] - lo[ax]).abs() < 1e-14 && (eng_h[ax] - h[ax]).abs() < 1e-14,
            "bounding-box convention drift on axis {ax}"
        );
    }
    let p = design.num_coeffs();
    assert_eq!(p, (k + 3) * (k + 3));

    // Naive dense normal equations: O(n·p²), the slow truth.
    let mut gram = vec![vec![0.0_f64; p]; p];
    let mut rhs: Vec<Vec<f64>> = vec![vec![0.0_f64]; p];
    for r in &rows {
        let row = dense_row(lo, h, k, r[0], r[1]);
        for i in 0..p {
            if row[i] == 0.0 {
                continue;
            }
            rhs[i][0] += r[3] * r[2] * row[i];
            for j in 0..p {
                gram[i][j] += r[3] * row[i] * row[j];
            }
        }
    }
    let s = dense_penalty(h, k, metric);

    // Off-data check points: cell interiors, knot lines, and one point OUTSIDE
    // the box (both paths extend the boundary-cell polynomial identically).
    let mut checks: Vec<[f64; 2]> = (0..12)
        .map(|i| {
            let t = i as f64 / 11.0;
            [
                lo[0] + (hi[0] - lo[0]) * t,
                lo[1] + (hi[1] - lo[1]) * (1.0 - t),
            ]
        })
        .collect();
    checks.push([lo[0] - 0.05, hi[1] + 0.03]);

    for &log_lambda in &[-1.0_f64, 2.5] {
        let lambda = log_lambda.exp();
        let mut a: Vec<Vec<f64>> = (0..p)
            .map(|i| (0..p).map(|j| gram[i][j] + lambda * s[i][j]).collect())
            .collect();
        // Symmetrize against the naive accumulation's roundoff asymmetry.
        for i in 0..p {
            for j in 0..i {
                let v = 0.5 * (a[i][j] + a[j][i]);
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        let c_oracle = dense_solve(&a, &rhs);
        let fit = fit_grid_spline_2d_at(&x1, &x2, &y, &w, k, metric, log_lambda, Some(1.0))
            .expect("engine fit at fixed lambda");
        let c_scale = c_oracle
            .iter()
            .map(|r| r[0].abs())
            .fold(0.0_f64, f64::max)
            .max(1e-3);
        for g in 0..p {
            assert!(
                (fit.coeffs[0][g] - c_oracle[g][0]).abs() <= 1e-8 * c_scale,
                "coefficient mismatch at {g} (logλ={log_lambda}): engine={} dense={}",
                fit.coeffs[0][g],
                c_oracle[g][0]
            );
        }
        // Fitted means and prediction variances at the check points: variance
        // is σ²·row'A⁻¹row with σ² = 1 supplied on both paths.
        for (ci, pt) in checks.iter().enumerate() {
            let row = dense_row(lo, h, k, pt[0], pt[1]);
            let mean_oracle: f64 = (0..p).map(|g| row[g] * c_oracle[g][0]).sum();
            let z = dense_solve(&a, &row.iter().map(|&v| vec![v]).collect::<Vec<_>>());
            let var_oracle: f64 = (0..p).map(|g| row[g] * z[g][0]).sum();
            let (mean, var) = fit.predict(0, pt[0], pt[1]).expect("engine predict");
            assert!(
                (mean - mean_oracle).abs() <= 1e-8 * mean_oracle.abs().max(1e-3),
                "fitted mean mismatch at check {ci} (logλ={log_lambda}): engine={mean} dense={mean_oracle}"
            );
            assert!(
                (var - var_oracle).abs() <= 1e-8 * var_oracle.max(1e-12),
                "variance mismatch at check {ci} (logλ={log_lambda}): engine={var} dense={var_oracle}"
            );
        }
    }
}

// ─────────────────────────── arm 2: truth recovery ──────────────────────────

#[test]
fn reml_fit_beats_the_noise_floor() {
    let noise_amp = 0.25;
    let rows = test_data(520, [0.0, 0.0], [1.0, 1.0], noise_amp);
    let (x1, x2, y, w) = split(&rows);
    let design = GridSpline2dDesign::build(&x1, &x2, &y, &w, 6, [1.0, 1.0]).expect("design");
    let fit = design.fit_reml().expect("REML-selected fit");

    let mut mse_fit = 0.0;
    let mut mse_noise = 0.0;
    for r in &rows {
        let truth = (3.0 * r[0]).sin() * (2.0 * r[1]).cos() + 0.4 * r[0] * r[1];
        let (mean, var) = fit.predict(0, r[0], r[1]).expect("predict at data point");
        assert!(var > 0.0, "posterior variance must be positive");
        mse_fit += (mean - truth) * (mean - truth);
        mse_noise += (r[2] - truth) * (r[2] - truth);
    }
    mse_fit /= rows.len() as f64;
    mse_noise /= rows.len() as f64;
    assert!(
        mse_fit < 0.5 * mse_noise,
        "REML fit must beat the noise floor sanely: mse_fit={mse_fit}, noise floor={mse_noise}"
    );
    // Profiled σ̂² should sit near the true noise variance (uniform ±0.25).
    let true_var = noise_amp * noise_amp / 3.0;
    assert!(
        fit.sigma2[0] > 0.4 * true_var && fit.sigma2[0] < 2.5 * true_var,
        "profiled sigma2 {} far from true noise variance {true_var}",
        fit.sigma2[0]
    );
}

// ───────────────── arm 3: penalty correctness (mixed term) ──────────────────

/// Greville abscissa of cubic basis j on a uniform extended knot axis:
/// the spline with these coefficients reproduces f(x) = x.
fn greville(lo: f64, h: f64, j: usize) -> f64 {
    lo + (j as f64 - 1.0) * h
}

/// Cubic blossom of x² at basis j's interior knots (t_{j+1}, t_{j+2}, t_{j+3}):
/// coefficients reproducing f(x) = x² (Marsden's identity).
fn blossom_sq(lo: f64, h: f64, j: usize) -> f64 {
    let t1 = lo + (j as f64 - 2.0) * h;
    let t2 = lo + (j as f64 - 1.0) * h;
    let t3 = lo + j as f64 * h;
    (t1 * t2 + t1 * t3 + t2 * t3) / 3.0
}

#[test]
fn assembled_penalty_matches_closed_form_quadratic_energy() {
    // Data on an exact box (corners included) — y is irrelevant to S.
    let lo = [0.2_f64, -0.5];
    let hi = [1.7_f64, 0.9];
    let k = 5usize;
    let a = [1.3_f64, 0.7];
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            let v1 = lo[0] + (hi[0] - lo[0]) * i as f64 / 5.0;
            let v2 = lo[1] + (hi[1] - lo[1]) * j as f64 / 5.0;
            x1.push(v1);
            x2.push(v2);
            y.push((v1 - v2).sin());
            w.push(1.0);
        }
    }
    let design = GridSpline2dDesign::build(&x1, &x2, &y, &w, k, a).expect("design");
    let glo = design.lower_corner();
    let gh = design.cell_widths();
    let m = design.basis_per_axis();

    // Coefficients of f = x1² + x1·x2 + x2² by polynomial reproduction:
    // c_{j1,j2} = blossom₂(j1) + ξ(j1)·ξ(j2) + blossom₂(j2).
    let mut coeff = vec![0.0_f64; m * m];
    for j1 in 0..m {
        for j2 in 0..m {
            coeff[j1 * m + j2] = blossom_sq(glo[0], gh[0], j1)
                + greville(glo[0], gh[0], j1) * greville(glo[1], gh[1], j2)
                + blossom_sq(glo[1], gh[1], j2);
        }
    }
    // Sanity: the coefficient vector really encodes f (catches any basis or
    // Greville/blossom convention drift before the energy assertion).
    for t in 0..7 {
        let p1 = lo[0] + (hi[0] - lo[0]) * (0.07 + 0.13 * t as f64);
        let p2 = lo[1] + (hi[1] - lo[1]) * (0.91 - 0.11 * t as f64);
        let row = dense_row(glo, gh, k, p1, p2);
        let val: f64 = (0..m * m).map(|g| row[g] * coeff[g]).sum();
        let truth = p1 * p1 + p1 * p2 + p2 * p2;
        assert!(
            (val - truth).abs() <= 1e-9 * truth.abs().max(1.0),
            "polynomial reproduction failed at ({p1}, {p2}): spline={val} truth={truth}"
        );
    }

    // f has CONSTANT second derivatives f11 = 2, f12 = 1, f22 = 2, so
    //   J(f) = (a1²·4 + 2·a1·a2·1 + a2²·4) · Area
    // exactly. A dropped mixed term misses this by exactly 2·a1·a2·Area.
    let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
    let j_closed = (4.0 * a[0] * a[0] + 2.0 * a[0] * a[1] + 4.0 * a[1] * a[1]) * area;
    let j_assembled = design
        .penalty_value(&coeff)
        .expect("penalty quadratic form");
    assert!(
        (j_assembled - j_closed).abs() <= 1e-8 * j_closed,
        "assembled biharmonic energy {j_assembled} != closed form {j_closed} \
         (mixed-term share is {})",
        2.0 * a[0] * a[1] * area
    );
}
