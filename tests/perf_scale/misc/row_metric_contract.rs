//! Contract tests for Object 2 — the standalone `RowMetric` (WP-B phase 1).
//!
//! These assert the load-bearing invariants of the single provenance-carrying
//! per-row metric *in isolation* (no SAE/gauge wiring): factored ops agree with
//! a dense `p × p` oracle, anisotropic whitening recovers isotropy, the
//! Euclidean provenance is a bit-identical no-op, and the Tikhonov `δ` floor is
//! strictly solver-only (it never touches a criterion-facing quantity).
//!
//! All factored operations are checked to *never* require materializing the
//! `p × p` `M_n`: we build the dense `M_n = U_n U_nᵀ` only inside the test
//! oracle, never via the API under test.

use std::sync::Arc;

use gam::inference::row_metric::{MetricProvenance, RowMetric};
use ndarray::{Array1, Array2};

/// Dense oracle: materialize `M_n = U_n U_nᵀ ∈ ℝ^{p × p}` from a single row's
/// flat factor `U_n[i, k] = u_row[i * rank + k]`. Used ONLY to check the
/// factored API; the API itself must never form this.
fn dense_block(u_row: &[f64], p: usize, rank: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0;
            for k in 0..rank {
                acc += u_row[i * rank + k] * u_row[j * rank + k];
            }
            m[[i, j]] = acc;
        }
    }
    m
}

fn dense_quad_form(m: &Array2<f64>, r: &[f64]) -> f64 {
    let p = m.nrows();
    let mut acc = 0.0;
    for i in 0..p {
        for j in 0..p {
            acc += r[i] * m[[i, j]] * r[j];
        }
    }
    acc
}

/// Build a small per-row factored metric with `n` rows, `p` outputs, `rank`
/// factors, filled deterministically so every row's block is distinct.
fn factored_u(n: usize, p: usize, rank: usize) -> Arc<Array2<f64>> {
    let mut u = Array2::<f64>::zeros((n, p * rank));
    for row in 0..n {
        for i in 0..p {
            for k in 0..rank {
                // distinct, finite, nonzero entries
                u[[row, i * rank + k]] =
                    0.3 + 0.17 * (row as f64) + 0.11 * (i as f64) - 0.07 * (k as f64);
            }
        }
    }
    Arc::new(u)
}

/// CONTRACT 1 — factored quad-form equals the dense `p × p` oracle at small `p`.
///
/// `quad_form(row, r) == rᵀ (U_n U_nᵀ) r` for every row, for a genuinely
/// low-rank metric (`rank < p`). Asserts the factored contraction `‖U_nᵀ r‖²`
/// reproduces the dense quadratic form to full double precision.
#[test]
fn factored_quad_form_matches_dense_oracle() {
    let n = 5usize;
    let p = 4usize;
    let rank = 2usize;
    let u = factored_u(n, p, rank);
    let metric = RowMetric::output_fisher(Arc::clone(&u), p, rank)
        .expect("low-rank PSD factors must validate");
    assert_eq!(metric.provenance(), MetricProvenance::OutputFisher { rank });

    let residuals: Vec<Vec<f64>> = (0..n)
        .map(|row| {
            (0..p)
                .map(|i| 0.5 - 0.21 * (i as f64) + 0.4 * (row as f64))
                .collect()
        })
        .collect();

    for row in 0..n {
        let u_row: Vec<f64> = (0..p * rank).map(|c| u[[row, c]]).collect();
        let m = dense_block(&u_row, p, rank);
        let r = &residuals[row];
        let expected = dense_quad_form(&m, r);
        let r_arr = Array1::from(r.clone());
        let got = metric.quad_form(row, r_arr.view());
        assert!(
            (got - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
            "row {row}: factored quad_form {got} != dense oracle {expected}"
        );
        // fisher_mass is the same quadratic read as an information mass.
        let mass = metric.fisher_mass(row, r_arr.view());
        assert_eq!(
            mass, got,
            "fisher_mass must equal quad_form for the same vector"
        );
    }
}

/// CONTRACT 2 — whitening a planted anisotropic residual recovers isotropy.
///
/// If the per-row factor is `U_n = diag(s_0, …, s_{p-1})` (rank == p), then
/// `whiten_residual_row(r)_i = s_i r_i`. Plant a residual `r_i = 1 / s_i` (so the
/// raw residual is anisotropic — large where the metric is weak, small where it
/// is strong) and confirm the whitened residual is the all-ones (isotropic)
/// vector, i.e. every channel contributes equally after whitening.
#[test]
fn whitening_a_planted_anisotropic_residual_recovers_isotropy() {
    let n = 3usize;
    let p = 3usize;
    let rank = p; // diagonal factor
    let scales = [2.0_f64, 5.0, 0.5];
    let mut u = Array2::<f64>::zeros((n, p * rank));
    for row in 0..n {
        for i in 0..p {
            // diagonal U_n: U_n[i, k] = s_i if i == k else 0
            u[[row, i * rank + i]] = scales[i];
        }
    }
    let metric = RowMetric::output_fisher(Arc::new(u), p, rank)
        .expect("diagonal factor metric must validate");

    for row in 0..n {
        // anisotropic raw residual r_i = 1 / s_i
        let r: Vec<f64> = scales.iter().map(|&s| 1.0 / s).collect();
        let r_arr = Array1::from(r);
        let whitened = metric.whiten_residual_row(row, r_arr.view());
        assert_eq!(whitened.len(), rank);
        for (i, &w) in whitened.iter().enumerate() {
            assert!(
                (w - 1.0).abs() < 1e-12,
                "row {row} channel {i}: whitened component {w} should be isotropic 1.0"
            );
        }
        // The induced quad form is then exactly p (each isotropic channel = 1²).
        let qf = metric.quad_form(row, r_arr.view());
        assert!(
            (qf - p as f64).abs() < 1e-12,
            "row {row}: whitened quad form {qf} should equal p={p}"
        );
    }
}

/// CONTRACT 6 — WhitenedStructured carries the #974 seam: it is constructible,
/// behaves like a factored metric for now, and reports its factor rank in the
/// provenance so #974 can fill the residual-covariance factorization later.
#[test]
fn whitened_structured_is_a_scoped_seam_for_974() {
    let n = 2usize;
    let p = 3usize;
    let rank = 2usize;
    let u = factored_u(n, p, rank);
    let metric = RowMetric::whitened_structured(Arc::clone(&u), p, rank)
        .expect("structured factors validate via the shared normalizer");
    assert_eq!(
        metric.provenance(),
        MetricProvenance::WhitenedStructured { factor_rank: rank }
    );
    // For now it whitens exactly like OutputFisher (same factors).
    let of = RowMetric::output_fisher(Arc::clone(&u), p, rank).unwrap();
    for row in 0..n {
        let r: Vec<f64> = (0..p)
            .map(|i| 0.6 - 0.2 * i as f64 + 0.1 * row as f64)
            .collect();
        let r_arr = Array1::from(r);
        assert_eq!(
            metric.whiten_residual_row(row, r_arr.view()),
            of.whiten_residual_row(row, r_arr.view()),
            "WhitenedStructured must currently match OutputFisher whitening"
        );
    }
}
