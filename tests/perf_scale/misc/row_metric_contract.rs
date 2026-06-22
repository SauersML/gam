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
use ndarray::{Array1, Array2, ArrayView1};

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

/// CONTRACT 3 — the δ floor is strictly solver-only (#747 invariant).
///
/// Two metrics with the SAME factors, one with `δ = 0` and one with a large
/// `δ`, must produce IDENTICAL criterion-facing quantities: `quad_form`,
/// `whiten_residual_row`, `fisher_mass`, and the criterion-facing `row_traces()`. Only
/// `solver_floor()` differs. This proves `δ` cannot bias the evidence criterion.
#[test]
fn delta_floor_is_solver_only_and_never_enters_the_criterion() {
    let n = 4usize;
    let p = 3usize;
    let rank = 1usize; // rank-deficient: U_n U_nᵀ is singular, so a solver would need δ
    let u = factored_u(n, p, rank);

    let no_floor = RowMetric::output_fisher(Arc::clone(&u), p, rank)
        .expect("rank-1 factors are PSD and must validate");
    let with_floor = RowMetric::output_fisher_with_solver_floor(Arc::clone(&u), p, rank, 7.5)
        .expect("a positive solver floor must be accepted");

    // The floor is recorded only on the solver side.
    assert_eq!(no_floor.solver_floor(), 0.0);
    assert_eq!(with_floor.solver_floor(), 7.5);

    // Provenance is identical (the floor is not part of provenance).
    assert_eq!(no_floor.provenance(), with_floor.provenance());

    // Every criterion-facing quantity is bit-identical between the two.
    for row in 0..n {
        let r: Vec<f64> = (0..p)
            .map(|i| 0.9 - 0.3 * (i as f64) + 0.2 * (row as f64))
            .collect();
        let r_arr = Array1::from(r);
        let qf0 = no_floor.quad_form(row, r_arr.view());
        let qf1 = with_floor.quad_form(row, r_arr.view());
        assert_eq!(qf0, qf1, "row {row}: δ floor must not change quad_form");

        let w0 = no_floor.whiten_residual_row(row, r_arr.view());
        let w1 = with_floor.whiten_residual_row(row, r_arr.view());
        assert_eq!(
            w0, w1,
            "row {row}: δ floor must not change whitened residual"
        );

        let m0 = no_floor.fisher_mass(row, r_arr.view());
        let m1 = with_floor.fisher_mass(row, r_arr.view());
        assert_eq!(m0, m1, "row {row}: δ floor must not change fisher_mass");
    }

    // The criterion-facing per-row traces are δ-free too.
    assert_eq!(
        no_floor.row_traces(),
        with_floor.row_traces(),
        "δ floor must not be baked into the criterion-facing metric traces"
    );
}

/// CONTRACT 4 — Euclidean provenance is a bit-identical no-op fast path.
///
/// `whiten_residual_row` returns the residual unchanged, `quad_form` equals the
/// bare `Σ r²` (computed by the exact same fold so equality is bit-for-bit), and
/// `whiten_jacobian` returns `J_n` reshaped unchanged. This is the guarantee
/// that installing a Euclidean `RowMetric` is behavior-preserving against the
/// historical isotropic path.
#[test]
fn euclidean_provenance_is_bit_identical_no_op() {
    let n = 6usize;
    let p = 3usize;
    let metric = RowMetric::euclidean(n, p).expect("Euclidean metric must build");
    assert_eq!(metric.provenance(), MetricProvenance::Euclidean);
    assert_eq!(metric.solver_floor(), 0.0);

    for row in 0..n {
        let r: Vec<f64> = (0..p)
            .map(|i| 0.4 - 0.13 * (i as f64) + 0.25 * (row as f64))
            .collect();
        let r_arr = Array1::from(r.clone());

        // whitening is the identity (same values, same order)
        let whitened = metric.whiten_residual_row(row, r_arr.view());
        assert_eq!(whitened, r, "Euclidean whitening must return r unchanged");

        // quad_form is bit-for-bit Σ r² (identical fold)
        let view: ArrayView1<'_, f64> = r_arr.view();
        let bare: f64 = view.iter().map(|&v| v * v).sum();
        assert_eq!(
            metric.quad_form(row, r_arr.view()),
            bare,
            "Euclidean quad_form must be bit-for-bit Σ r²"
        );
    }

    // whiten_jacobian under Euclidean returns J_n reshaped (p, d) unchanged.
    let d = 2usize;
    let row = 2usize;
    // J_n[i, a] = j_row[i*d + a]
    let j_row: Vec<f64> = (0..p * d).map(|c| 1.0 + 0.5 * c as f64).collect();
    let whitened_jac = metric.whiten_jacobian(row, &j_row, d);
    assert_eq!(whitened_jac.dim(), (p, d));
    for i in 0..p {
        for a in 0..d {
            assert_eq!(
                whitened_jac[[i, a]],
                j_row[i * d + a],
                "Euclidean whiten_jacobian must reshape J_n unchanged"
            );
        }
    }
}

/// CONTRACT 5 — whiten_jacobian factored equals the dense pullback oracle.
///
/// `whiten_jacobian(J_n)ᵀ whiten_jacobian(J_n) == J_nᵀ (U_n U_nᵀ) J_n`, i.e. the
/// factored whitening reproduces the dense pullback `g_n` without forming the
/// `p × p` `M_n`.
#[test]
fn whiten_jacobian_factored_matches_dense_pullback() {
    let n = 3usize;
    let p = 4usize;
    let rank = 2usize;
    let d = 3usize;
    let u = factored_u(n, p, rank);
    let metric = RowMetric::output_fisher(Arc::clone(&u), p, rank).expect("factors validate");

    for row in 0..n {
        let j_row: Vec<f64> = (0..p * d)
            .map(|c| 0.2 + 0.31 * c as f64 - 0.05 * row as f64)
            .collect();

        // factored: M = U_nᵀ J_n (rank × d), g = MᵀM (d × d)
        let m = metric.whiten_jacobian(row, &j_row, d);
        assert_eq!(m.dim(), (rank, d));
        let mut g_factored = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut acc = 0.0;
                for k in 0..rank {
                    acc += m[[k, a]] * m[[k, b]];
                }
                g_factored[[a, b]] = acc;
            }
        }

        // dense oracle: g = J_nᵀ (U_n U_nᵀ) J_n
        let u_row: Vec<f64> = (0..p * rank).map(|c| u[[row, c]]).collect();
        let mblock = dense_block(&u_row, p, rank);
        let mut g_dense = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut acc = 0.0;
                for i in 0..p {
                    for j in 0..p {
                        acc += j_row[i * d + a] * mblock[[i, j]] * j_row[j * d + b];
                    }
                }
                g_dense[[a, b]] = acc;
            }
        }

        for a in 0..d {
            for b in 0..d {
                let gf = g_factored[[a, b]];
                let gd = g_dense[[a, b]];
                assert!(
                    (gf - gd).abs() <= 1e-10 * (1.0 + gd.abs()),
                    "row {row} g[{a},{b}]: factored {gf} != dense {gd}"
                );
            }
        }
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
