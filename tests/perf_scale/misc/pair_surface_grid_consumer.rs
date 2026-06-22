//! #1031 consumer-level oracle for `fit_pair_surface`, THE first-class
//! pair-component estimator backed by the streaming 2-D grid engine.
//!
//! Self-constructed truth (#904): an independent dense re-statement of the
//! SAME estimator — naive O(n·p²) normal equations on the uniform cubic
//! B-spline tensor basis, naive dense Gauss–Legendre assembly of the FULL
//! anisotropic biharmonic penalty at the consumer's pinned metric
//! `a_i = span_i²`, in-test Gaussian elimination — must reproduce every
//! carve-facing object the consumer hands out (coefficients, scale-free
//! coefficient covariance, EDF, residual cross-covariance, predictions) to
//! near machine precision, and the REML-selected λ must be a maximizer of
//! the independently computed pooled restricted criterion.
//!
//! Three arms:
//! 1. dense-oracle exactness of the consumer surface (small grid);
//! 2. carve integration — a planted ADDITIVE surface fissions losslessly,
//!    a planted BOUND surface is rejected and stays whole (one empirical
//!    measure end to end: the bases the consumer returns are the bases the
//!    carve centers against);
//! 3. e2e truth recovery at large gridded n (320×320 lattice, two response
//!    dimensions sharing one REML λ) through the streaming path, with
//!    posterior predictions from the consumer's own honest API.

use std::time::Instant;

use gam::inference::smooth_test::SmoothTestScale;
use gam::terms::structure::anova_atom::{
    BindingNotion, CarveInput, FISSION_MAX_INTERACTION_FRACTION, PairSurfaceBackend, basis_means,
    carve, fit_pair_surface,
};
use ndarray::Array2;

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

/// Naive in-test Cholesky log-determinant of a dense SPD matrix.
fn dense_chol_logdet(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    let mut logdet = 0.0;
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for t in 0..j {
                s -= l[i][t] * l[j][t];
            }
            if i == j {
                assert!(s > 0.0, "dense oracle: not positive definite at {i}");
                l[i][i] = s.sqrt();
                logdet += 2.0 * l[i][i].ln();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    logdet
}

// ─────────────── in-test re-statement of the basis definition ───────────────

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

/// Naive dense assembly of the full anisotropic biharmonic penalty.
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

/// Deterministic noise in (−amp, amp): golden-ratio rotation, no RNG.
fn noise(i: usize, amp: f64) -> f64 {
    ((i as f64 * 0.618_033_988_749_894_9).fract() - 0.5) * 2.0 * amp
}

/// `nx × ny` lattice over the box — GRIDDED data, corners included.
fn lattice(nx: usize, ny: usize, lo: [f64; 2], hi: [f64; 2]) -> (Vec<f64>, Vec<f64>) {
    let mut x1 = Vec::with_capacity(nx * ny);
    let mut x2 = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            x1.push(lo[0] + (hi[0] - lo[0]) * i as f64 / (nx - 1) as f64);
            x2.push(lo[1] + (hi[1] - lo[1]) * j as f64 / (ny - 1) as f64);
        }
    }
    (x1, x2)
}

// ─────────────────────── arm 1: dense-oracle exactness ──────────────────────

#[test]
fn pair_surface_grid_backend_matches_dense_oracle() {
    let lo = [0.2_f64, -0.5];
    let hi = [1.7_f64, 0.9];
    let (x1, x2) = lattice(19, 19, lo, hi);
    let n = x1.len(); // 361 ⇒ K = ⌈n^(1/3)⌉ = 8, m = 11, p = 121
    let mut responses = Array2::<f64>::zeros((n, 2));
    for r in 0..n {
        responses[[r, 0]] =
            (3.0 * x1[r]).sin() * (2.0 * x2[r]).cos() + 0.4 * x1[r] * x2[r] + noise(r, 0.1);
        responses[[r, 1]] =
            (2.0 * x1[r]).cos() + (3.0 * x2[r]).sin() - 0.3 * x1[r] * x2[r] + noise(r + 7, 0.1);
    }
    let fit = fit_pair_surface(&x1, &x2, responses.view()).expect("pair surface fit");
    assert_eq!(fit.backend, PairSurfaceBackend::GridExact);
    let m = fit.phi_a.ncols();
    assert_eq!(m, 11, "auto-chosen K must be ⌈361^(1/3)⌉ = 8 ⇒ m = 11");
    let k = m - 3;
    let p = m * m;

    // The consumer's pinned conventions, re-derived independently: bounding
    // box from the data, metric a_i = span_i².
    let h = [(hi[0] - lo[0]) / k as f64, (hi[1] - lo[1]) / k as f64];
    let metric = [
        (hi[0] - lo[0]) * (hi[0] - lo[0]),
        (hi[1] - lo[1]) * (hi[1] - lo[1]),
    ];
    for ax in 0..2 {
        assert!(
            (fit.lower_corner[ax] - lo[ax]).abs() < 1e-14
                && (fit.cell_widths[ax] - h[ax]).abs() < 1e-14,
            "bounding-box convention drift on axis {ax}"
        );
    }

    // Naive dense normal equations + penalty: the slow truth.
    let mut gram = vec![vec![0.0_f64; p]; p];
    let mut rhs = vec![vec![0.0_f64; 2]; p];
    for r in 0..n {
        let row = dense_row(lo, h, k, x1[r], x2[r]);
        for i in 0..p {
            if row[i] == 0.0 {
                continue;
            }
            for d in 0..2 {
                rhs[i][d] += responses[[r, d]] * row[i];
            }
            for j in 0..p {
                gram[i][j] += row[i] * row[j];
            }
        }
    }
    let s = dense_penalty(h, k, metric);
    let lambda = fit.surface.lambda;
    let system_at = |lam: f64| -> Vec<Vec<f64>> {
        let mut a: Vec<Vec<f64>> = (0..p)
            .map(|i| (0..p).map(|j| gram[i][j] + lam * s[i][j]).collect())
            .collect();
        for i in 0..p {
            for j in 0..i {
                let v = 0.5 * (a[i][j] + a[j][i]);
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        a
    };
    let a = system_at(lambda);
    let c_oracle = dense_solve(&a, &rhs); // p × 2

    // Coefficients (row-major vec convention j·m + k).
    let c_scale = c_oracle
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1e-3);
    for d in 0..2 {
        for j in 0..m {
            for kk in 0..m {
                let eng = fit.surface.coeffs[d][[j, kk]];
                let den = c_oracle[j * m + kk][d];
                assert!(
                    (eng - den).abs() <= 1e-8 * c_scale,
                    "coefficient mismatch dim {d} ({j},{kk}): consumer={eng} dense={den}"
                );
            }
        }
    }

    // Scale-free coefficient covariance U = (X'X + λS)⁻¹.
    let mut eye = vec![vec![0.0_f64; p]; p];
    for g in 0..p {
        eye[g][g] = 1.0;
    }
    let a_inv = dense_solve(&a, &eye);
    let u_scale = a_inv
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    for i in 0..p {
        for j in 0..p {
            assert!(
                (fit.surface.unit_covariance[[i, j]] - a_inv[i][j]).abs() <= 1e-8 * u_scale,
                "unit covariance mismatch at ({i},{j})"
            );
        }
    }

    // EDF = tr(A⁻¹·X'X), residual df, residual cross-covariance.
    let mut edf = 0.0;
    for g in 0..p {
        for t in 0..p {
            edf += a_inv[g][t] * gram[t][g];
        }
    }
    assert!(
        (fit.surface.edf - edf).abs() <= 1e-7 * edf,
        "edf mismatch: consumer={} dense={edf}",
        fit.surface.edf
    );
    let residual_df = n as f64 - edf;
    assert!((fit.surface.residual_df - residual_df).abs() <= 1e-7 * residual_df);
    let mut rcc = [[0.0_f64; 2]; 2];
    for r in 0..n {
        let row = dense_row(lo, h, k, x1[r], x2[r]);
        let mut res = [0.0_f64; 2];
        for d in 0..2 {
            let fitted: f64 = (0..p).map(|g| row[g] * c_oracle[g][d]).sum();
            res[d] = responses[[r, d]] - fitted;
        }
        for d in 0..2 {
            for e in 0..2 {
                rcc[d][e] += res[d] * res[e];
            }
        }
    }
    for d in 0..2 {
        for e in 0..2 {
            let den = rcc[d][e] / residual_df;
            assert!(
                (fit.surface.residual_cross_cov[[d, e]] - den).abs() <= 1e-7 * den.abs().max(1e-12),
                "residual cross-covariance mismatch at ({d},{e}): consumer={} dense={den}",
                fit.surface.residual_cross_cov[[d, e]]
            );
        }
    }

    // The honest prediction API: mean and posterior variance at off-grid
    // checkpoints (plus one point OUTSIDE the box) against the dense truth,
    // and consistency with the returned sample bases (one measure end to
    // end: φ-row ⊗ φ-row dotted with vec(C) is the fitted value).
    let mut checks: Vec<[f64; 2]> = (0..9)
        .map(|i| {
            let t = (i as f64 + 0.37) / 9.0;
            [lo[0] + (hi[0] - lo[0]) * t, hi[1] - (hi[1] - lo[1]) * t]
        })
        .collect();
    checks.push([lo[0] - 0.04, hi[1] + 0.06]);
    for (ci, pt) in checks.iter().enumerate() {
        let row = dense_row(lo, h, k, pt[0], pt[1]);
        let z = dense_solve(&a, &row.iter().map(|&v| vec![v]).collect::<Vec<_>>());
        let quad: f64 = (0..p).map(|g| row[g] * z[g][0]).sum();
        for d in 0..2 {
            let mean_oracle: f64 = (0..p).map(|g| row[g] * c_oracle[g][d]).sum();
            let var_oracle = (rcc[d][d] / residual_df) * quad;
            let (mean, var) = fit.predict(d, pt[0], pt[1]).expect("consumer predict");
            assert!(
                (mean - mean_oracle).abs() <= 1e-8 * mean_oracle.abs().max(1e-3),
                "prediction mean mismatch at check {ci}, dim {d}: {mean} vs {mean_oracle}"
            );
            assert!(
                (var - var_oracle).abs() <= 1e-8 * var_oracle.max(1e-12),
                "prediction variance mismatch at check {ci}, dim {d}: {var} vs {var_oracle}"
            );
        }
    }
    for r in (0..n).step_by(41) {
        let mut surf = 0.0;
        for j in 0..m {
            for kk in 0..m {
                surf += fit.phi_a[[r, j]] * fit.surface.coeffs[0][[j, kk]] * fit.phi_b[[r, kk]];
            }
        }
        let (mean, _) = fit.predict(0, x1[r], x2[r]).expect("predict at sample row");
        assert!(
            (surf - mean).abs() <= 1e-10 * mean.abs().max(1e-3),
            "returned bases disagree with the prediction surface at row {r}"
        );
    }

    // REML optimality: the selected λ must (weakly) beat its neighbors on
    // the independently computed pooled restricted criterion
    //   ℓ_R(λ) = −½ Σ_d [log|X'X+λS| − (p−3)·log λ + (n−3)·log σ̂²_d(λ)].
    let crit = |lam: f64| -> f64 {
        let a = system_at(lam);
        let logdet = dense_chol_logdet(&a);
        let c = dense_solve(&a, &rhs);
        let dof = (n - 3) as f64;
        let r_pen = (p - 3) as f64;
        let mut v = 0.0;
        for d in 0..2 {
            let yty: f64 = (0..n).map(|r| responses[[r, d]] * responses[[r, d]]).sum();
            let quad: f64 = (0..p).map(|g| rhs[g][d] * c[g][d]).sum();
            let rss_pen = yty - quad;
            assert!(rss_pen > 0.0);
            v += logdet - r_pen * lam.ln() + dof * (rss_pen / dof).ln();
        }
        -0.5 * v
    };
    let at_star = crit(lambda);
    for shift in [-0.4_f64, 0.4] {
        let neighbor = crit(lambda * shift.exp());
        assert!(
            at_star >= neighbor - 1e-7 * at_star.abs().max(1.0),
            "REML-selected λ is not a maximizer: crit(λ*)={at_star}, \
             crit(λ*·e^{shift})={neighbor}"
        );
    }
}

// ───────────────────────── arm 2: carve integration ─────────────────────────

#[test]
fn pair_surface_feeds_carve_additive_splits_bound_refuses() {
    let (x1, x2) = lattice(19, 19, [0.0, 0.0], [1.0, 1.0]);
    let n = x1.len();

    // ADDITIVE truth: the fitted surface's interaction block carries only
    // noise + spline approximation leakage; the carve must fission and the
    // children must reassemble the parent.
    let mut add = Array2::<f64>::zeros((n, 1));
    for r in 0..n {
        add[[r, 0]] = (2.0 * x1[r]).sin() + (3.0 * x2[r]).cos() + noise(r, 1e-3);
    }
    let fit = fit_pair_surface(&x1, &x2, add.view()).expect("additive pair surface");
    assert_eq!(fit.backend, PairSurfaceBackend::GridExact);
    let input = CarveInput {
        phi_a: fit.phi_a.view(),
        phi_b: fit.phi_b.view(),
        coeffs: &fit.surface.coeffs,
        coeff_covariance: Some(&fit.surface.coeff_covariance),
        joint_coeff_covariance: None,
        kernel_a: None,
        kernel_b: None,
        edf: None,
        residual_df: fit.surface.residual_df,
        scale: SmoothTestScale::Estimated,
        notion: BindingNotion::Representational,
    };
    let report = carve(&input, 0.05).expect("carve additive");
    assert!(
        report.interaction_fraction < FISSION_MAX_INTERACTION_FRACTION,
        "additive surface must carry negligible interaction energy \
         (fraction = {})",
        report.interaction_fraction
    );
    let plan = report
        .fission
        .as_ref()
        .expect("additive surface must fission");
    // Lossless reassembly: childA + childB reproduce the fitted surface on
    // the sample (defect is declared, and tiny here).
    let mean_a = basis_means(fit.phi_a.view());
    let mean_b = basis_means(fit.phi_b.view());
    let m = fit.phi_a.ncols();
    let mut max_gap = 0.0_f64;
    for r in 0..n {
        let mut parent = 0.0;
        let mut fa = plan.child_a[0].constant;
        let mut fb = plan.child_b[0].constant;
        for j in 0..m {
            fa += (fit.phi_a[[r, j]] - mean_a[j]) * plan.child_a[0].centered_coeffs[j];
            fb += (fit.phi_b[[r, j]] - mean_b[j]) * plan.child_b[0].centered_coeffs[j];
            for kk in 0..m {
                parent += fit.phi_a[[r, j]] * fit.surface.coeffs[0][[j, kk]] * fit.phi_b[[r, kk]];
            }
        }
        max_gap = max_gap.max((parent - (fa + fb)).abs());
    }
    assert!(
        max_gap < 1e-2,
        "fission children must reassemble the additive parent (max gap {max_gap})"
    );

    // BOUND truth: a multiplicative surface — the binding test must reject
    // and the carve must keep the atom whole.
    let mut bound = Array2::<f64>::zeros((n, 1));
    for r in 0..n {
        bound[[r, 0]] = (2.0 * x1[r]).sin() * (3.0 * x2[r]).cos() + noise(r, 0.05);
    }
    let fit_b = fit_pair_surface(&x1, &x2, bound.view()).expect("bound pair surface");
    assert_eq!(fit_b.backend, PairSurfaceBackend::GridExact);
    let input_b = CarveInput {
        phi_a: fit_b.phi_a.view(),
        phi_b: fit_b.phi_b.view(),
        coeffs: &fit_b.surface.coeffs,
        coeff_covariance: Some(&fit_b.surface.coeff_covariance),
        joint_coeff_covariance: None,
        kernel_a: None,
        kernel_b: None,
        edf: None,
        residual_df: fit_b.surface.residual_df,
        scale: SmoothTestScale::Estimated,
        notion: BindingNotion::Representational,
    };
    let report_b = carve(&input_b, 0.05).expect("carve bound");
    let p_val = report_b.edge_p_value.expect("binding test must run");
    assert!(p_val < 1e-3, "planted binding must reject, p = {p_val}");
    assert!(report_b.fission.is_none(), "bound surface must not fission");
    assert!(report_b.interaction_fraction > 0.05);
}

// ──────────────────── arm 3: large gridded n, end to end ────────────────────

#[test]
fn pair_surface_large_gridded_n_recovers_truth_end_to_end() {
    const SIDE: usize = 320; // n = 102_400 gridded rows ⇒ K hits the cap (32)
    const NOISE_AMP: f64 = 0.3;
    let truth0 = |a: f64, b: f64| (3.0 * a).sin() * (2.0 * b).cos() + 0.4 * a * b;
    let truth1 = |a: f64, b: f64| (2.0 * a).sin() + (3.0 * b).cos();

    let (x1, x2) = lattice(SIDE, SIDE, [0.0, 0.0], [1.0, 1.0]);
    let n = x1.len();
    let mut responses = Array2::<f64>::zeros((n, 2));
    for r in 0..n {
        responses[[r, 0]] = truth0(x1[r], x2[r]) + noise(r, NOISE_AMP);
        responses[[r, 1]] = truth1(x1[r], x2[r]) + noise(r + 13, NOISE_AMP);
    }

    let start = Instant::now();
    let fit = fit_pair_surface(&x1, &x2, responses.view()).expect("large pair surface");
    println!(
        "fit_pair_surface n={n} (D=2) seconds: {:.3}",
        start.elapsed().as_secs_f64()
    );
    assert_eq!(fit.backend, PairSurfaceBackend::GridExact);
    assert_eq!(
        fit.phi_a.ncols(),
        35,
        "auto-chosen K must hit the engine's sizing cap (32 ⇒ m = 35)"
    );

    // Truth recovery at OFF-grid checkpoints through the consumer's own
    // prediction API, both response dimensions.
    const CHECKS: usize = 25;
    let true_var = NOISE_AMP * NOISE_AMP / 3.0;
    for (d, truth) in [(0usize, &truth0 as &dyn Fn(f64, f64) -> f64), (1, &truth1)] {
        let mut mse = 0.0;
        for i in 0..CHECKS {
            let px1 = (i as f64 + 0.43) / CHECKS as f64;
            for j in 0..CHECKS {
                let px2 = (j as f64 + 0.61) / CHECKS as f64;
                let (mean, var) = fit.predict(d, px1, px2).expect("predict checkpoint");
                assert!(var > 0.0, "posterior variance must be positive");
                let err = mean - truth(px1, px2);
                mse += err * err;
            }
        }
        mse /= (CHECKS * CHECKS) as f64;
        assert!(
            mse < 0.5 * true_var,
            "dim {d}: REML fit must recover truth well below the noise floor: \
             mse={mse}, noise var={true_var}"
        );
        let s2 = fit.surface.residual_cross_cov[[d, d]];
        assert!(
            s2 > 0.4 * true_var && s2 < 2.5 * true_var,
            "dim {d}: residual variance {s2} far from true noise variance {true_var}"
        );
    }
    // The two response dimensions carry INDEPENDENT noise streams: the
    // residual cross-covariance must be near-diagonal.
    let off = fit.surface.residual_cross_cov[[0, 1]].abs();
    assert!(
        off < 0.2 * true_var,
        "independent noise streams must not correlate (off-diagonal {off})"
    );
}
