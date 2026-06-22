//! Regression for #805: `SoftmaxAssignmentSparsityPenalty`'s PSD majorizer
//! dominates the *dense* per-row entropy Hessian in the Loewner order, not just
//! entry-by-entry.
//!
//! The bug-hunt test only checks the diagonal contract on one near-uniform row.
//! The deeper requirement — and the whole point of fixing the "diagonal applied
//! to a dense operator" error — is that the advertised diagonal majorizer `D`
//! satisfies `D ⪰ H` as matrices, where `H` is the dense per-row Hessian. This
//! test reconstructs `H` from the exact `hvp` (the source of truth for the dense
//! operator) and verifies, over many configurations and probe directions, that
//!   * `vᵀ D v ≥ 0`            (B ⪰ 0),
//!   * `vᵀ D v ≥ vᵀ H v`       (B ⪰ ∂²P, Loewner order), and
//!   * the exact `H` is genuinely indefinite somewhere (the bug was real),
//!   * `psd_majorizer_diag` equals the diagonal of the reconstructed `H`'s
//!     absolute row sums and `psd_majorizer_hvp` is `D ⊙ v`.

use gam::terms::analytic_penalties::{AnalyticPenalty, SoftmaxAssignmentSparsityPenalty};
use ndarray::{Array1, Array2, ArrayView1};

/// Reconstruct the full (block-diagonal) Hessian by applying the exact `hvp` to
/// each standard basis vector — `hvp` is the documented source of truth for the
/// dense operator.
fn dense_hessian(
    pen: &SoftmaxAssignmentSparsityPenalty,
    target: ArrayView1<'_, f64>,
    rho: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let n = target.len();
    let mut h = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        let col = pen.hvp(target, rho, e.view());
        for i in 0..n {
            h[[i, j]] = col[i];
        }
    }
    h
}

#[test]
fn majorizer_dominates_dense_hessian_in_loewner_order() {
    let configs: &[(usize, f64)] = &[(2, 1.0), (4, 0.5), (4, 2.0), (6, 1.3)];
    // Per-row logit patterns: near-uniform, confident (peaked), mixed, signed.
    let row_patterns: &[fn(usize) -> Vec<f64>] = &[
        |k| vec![0.0; k],
        |k| (0..k).map(|i| if i == 0 { 6.0 } else { 0.0 }).collect(),
        |k| (0..k).map(|i| i as f64 * 0.7).collect(),
        |k| {
            (0..k)
                .map(|i| ((i as f64) - (k as f64) / 2.0) * 1.1)
                .collect()
        },
    ];

    for &(k, tau) in configs {
        for &rho_val in &[0.0_f64, 0.8, -0.5] {
            let pen = SoftmaxAssignmentSparsityPenalty::new(k, tau);
            let rho = Array1::from(vec![rho_val]);

            // Build a multi-row target spanning all patterns.
            let mut t = Vec::new();
            for p in row_patterns {
                t.extend(p(k));
            }
            let target = Array1::from(t);
            let n = target.len();

            let maj = pen
                .psd_majorizer_diag(target.view(), rho.view())
                .expect("diagonal majorizer exists");
            let hess_diag = pen.hessian_diag(target.view(), rho.view()).unwrap();
            let h = dense_hessian(&pen, target.view(), rho.view());

            // Sanity: reconstructed dense diagonal matches the analytic
            // hessian_diag, and H is symmetric.
            for i in 0..n {
                assert!(
                    (h[[i, i]] - hess_diag[i]).abs() < 1e-9,
                    "k={k} tau={tau} rho={rho_val}: dense H diag {} != hessian_diag {}",
                    h[[i, i]],
                    hess_diag[i]
                );
                for j in 0..n {
                    assert!(
                        (h[[i, j]] - h[[j, i]]).abs() < 1e-9,
                        "H not symmetric at ({i},{j})"
                    );
                }
                // B ⪰ 0 entrywise on the diagonal (necessary for PSD).
                assert!(maj[i] >= -1e-12, "majorizer entry {} < 0", maj[i]);
                // The majorizer diagonal must equal the absolute row sum of H
                // (the Gershgorin construction), confirming it accounts for the
                // off-diagonal coupling rather than ignoring it.
                let abs_row_sum: f64 = (0..n).map(|j| h[[i, j]].abs()).sum();
                assert!(
                    (maj[i] - abs_row_sum).abs() < 1e-9,
                    "k={k} tau={tau} rho={rho_val}: majorizer[{i}]={} != |row sum|={}",
                    maj[i],
                    abs_row_sum
                );
            }

            // Loewner domination + PSD over many probe directions.
            let qf_d = |v: &Array1<f64>| -> f64 { (0..n).map(|i| maj[i] * v[i] * v[i]).sum() };
            let qf_h = |v: &Array1<f64>| -> f64 {
                let mut acc = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        acc += v[i] * h[[i, j]] * v[j];
                    }
                }
                acc
            };

            let mut min_h_qf = f64::INFINITY;
            for seed in 0..40 {
                let v: Array1<f64> = Array1::from(
                    (0..n)
                        .map(|i| {
                            // Deterministic pseudo-random in [-1, 1].
                            let x = ((i * 131 + seed * 977 + 17) % 1000) as f64 / 500.0 - 1.0;
                            x
                        })
                        .collect::<Vec<_>>(),
                );
                let dq = qf_d(&v);
                let hq = qf_h(&v);
                min_h_qf = min_h_qf.min(hq);
                assert!(
                    dq >= -1e-9,
                    "k={k} tau={tau} rho={rho_val} seed={seed}: vᵀDv = {dq} < 0 (B ⪰ 0 violated)"
                );
                assert!(
                    dq >= hq - 1e-7,
                    "k={k} tau={tau} rho={rho_val} seed={seed}: vᵀDv {dq} < vᵀHv {hq} (B ⪰ ∂²P violated)"
                );
            }
            // The dense Hessian is genuinely indefinite (some direction has
            // negative curvature) — confirms the majorizer is non-trivially
            // correcting an indefinite operator, not a convex one.
            assert!(
                min_h_qf < -1e-6,
                "k={k} tau={tau} rho={rho_val}: expected indefinite H, min qf = {min_h_qf}"
            );

            // psd_majorizer_hvp is the diagonal applied to v.
            let v: Array1<f64> =
                Array1::from((0..n).map(|i| (i as f64 % 3.0) - 1.0).collect::<Vec<_>>());
            let hv = pen.psd_majorizer_hvp(target.view(), rho.view(), v.view());
            for i in 0..n {
                assert!(
                    (hv[i] - maj[i] * v[i]).abs() < 1e-12,
                    "psd_majorizer_hvp[{i}] {} != D⊙v {}",
                    hv[i],
                    maj[i] * v[i]
                );
            }
        }
    }
}
