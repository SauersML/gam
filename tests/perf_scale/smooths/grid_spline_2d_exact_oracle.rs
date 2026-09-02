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

use gam::terms::grid_spline_2d::GridSpline2dDesign;

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

// ─────────────────────────── arm 2: truth recovery ──────────────────────────

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

