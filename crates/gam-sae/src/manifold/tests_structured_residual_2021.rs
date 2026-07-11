//! #2021 WAVE-2 acceptance — the `WhitenedStructured` residual metric, once
//! installed on a [`SaeManifoldTerm`] via [`SaeManifoldTerm::set_row_metric`],
//! actually whitens the reconstruction likelihood: it reports
//! `whitens_likelihood()`, and BOTH the objective value (`loss_scaled`'s
//! `data_fit`) and the assembled Newton gradient differ from the isotropic (iid)
//! path on a fixture with structured, heteroscedastic residual correlation —
//! i.e. the whitening is active, not a silent no-op. The metric-independent
//! penalty terms (assignment sparsity, ARD, smoothness) stay bit-identical, so
//! the ONLY channel that moves is the data-fit — exactly the #974 seam
//! `loss_scaled` documents.
//!
//! The producer is the hoisted `StructuredResidualModel::row_metric`
//! (`gam_solve::inference::residual_factor`), the first real emitter of
//! [`gam_problem::MetricProvenance::WhitenedStructured`]. This test drives that
//! producer → `set_row_metric` install → `loss`/`assemble_arrow_schur` consume
//! path end-to-end.

use crate::assignment::{AssignmentMode, SaeAssignment};
use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm};
use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, Array3};

// Deterministic standard-normal draws (Box–Muller over an LCG) so the fixture —
// and therefore the fitted factor / diagonal — is reproducible bit-for-bit.
fn lcg_uniform(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_uniform(s).max(1e-12);
    let u2 = lcg_uniform(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// Total assembled gradient energy `Σ_i ‖g_t^(i)‖² + ‖g_β‖²` — the norm the inner
// solve's convergence gate measures; a robust, layout-independent scalar to
// compare the iid and whitened assemblies.
fn grad_norm_sq(sys: &gam_solve::arrow_schur::ArrowSchurSystem) -> f64 {
    let gt: f64 = sys
        .rows
        .iter()
        .map(|r| r.gt.iter().map(|&v| v * v).sum::<f64>())
        .sum();
    let gb: f64 = sys.gb.iter().map(|&v| v * v).sum::<f64>();
    gt + gb
}

fn build_term(n: usize, p: usize, k: usize) -> SaeManifoldTerm {
    // Euclidean atoms (identity geometry), width-2 basis, one latent axis. Each
    // atom carries a distinct nonzero decoder so the reconstruction — and hence
    // the residual `Z − R` the metric whitens — is genuinely nonzero.
    let atoms: Vec<SaeManifoldAtom> = (0..k)
        .map(|i| {
            let f = (i as f64) + 1.0;
            let decoder = Array2::<f64>::from_shape_fn((2, p), |(m, c)| {
                0.1 * f * ((m + 1) as f64) - 0.05 * (c as f64) + 0.02 * f
            });
            SaeManifoldAtom::new(
                format!("atom{i}"),
                SaeAtomBasisKind::EuclideanPatch,
                1,
                Array2::<f64>::from_elem((n, 2), 1.0),
                Array3::<f64>::zeros((n, 2, 1)),
                decoder,
                Array2::<f64>::eye(2),
            )
            .unwrap()
        })
        .collect();
    let coords: Vec<Array2<f64>> = (0..k)
        .map(|_| Array2::<f64>::from_shape_fn((n, 1), |(r, _)| 0.05 * (r as f64)))
        .collect();
    let manifolds = vec![LatentManifold::Euclidean; k];
    let logits =
        Array2::<f64>::from_shape_fn((n, k), |(r, c)| 0.3 * (c as f64) - 0.1 * (r as f64) + 0.2);
    // ordered Beta--Bernoulli-MAP (fixed alpha): small K ⇒ dense layout, so the iid and whitened
    // assemblies share the exact same row structure and only the metric differs.
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// A `WhitenedStructured` per-row precision (rank-≥1 factor + heteroscedastic
/// diagonal) fitted over `(n, p)` correlated residuals.
fn fit_structured_metric(n: usize, p: usize) -> gam_problem::RowMetric {
    // A single, STRONG shared rank-1 interference direction (factor loadings that
    // dominate the per-output idiosyncratic noise: |λ_i| ≈ 1 vs d_scale ≈ 0.2–0.3,
    // an ≈25:1 factor-to-idiosyncratic variance ratio on the leading channel) plus
    // per-output heteroscedastic idiosyncratic noise ⇒ Σ_n is genuinely anisotropic
    // (D non-uniform), so whitening cannot collapse to a scalar rescale (which would
    // leave the loss ratio-invariant). The dominant off-diagonal correlation makes a
    // rank-1 factor MATHEMATICALLY JUSTIFIED, not marginal: the evidence ladder's
    // ½·p·log n Occam penalty is cleared with room to spare at the fixture's `n`
    // (see the caller's n rationale), so `fit` selects rank 1 rather than the
    // pure-diagonal model. (The prior loadings — |λ| ≤ 1 with d_scale up to 0.95 —
    // were a marginal factor whose BIC advantage was swamped by the penalty at small
    // n: the ladder correctly picked rank 0 there, vacuously skipping the whitening
    // asserts below. That is a fixture regime bug, #2107 — rank selection itself is
    // sound, as the at-scale `evidence_ladder_prefers_planted_rank_one` /
    // `factor_recovers_planted_interference_subspace` tests in `residual_factor.rs`
    // confirm — so the fix is to plant a genuinely-justified factor, not to touch
    // the estimator.)
    let lam = [1.5_f64, -1.2, 0.9, 1.3, -1.0];
    let dscale = [0.25_f64, 0.30, 0.20, 0.28, 0.22];
    let mut seed = 0x2021_00D5_1234_ABCDu64;
    let mut residuals = Array2::<f64>::zeros((n, p));
    let mut activity = Array1::<f64>::zeros(n);
    for row in 0..n {
        let common = lcg_normal(&mut seed);
        // Uneven occupancy so the occupancy-normalized per-row scale varies.
        activity[row] = 0.25 + (row as f64) / (n as f64);
        let amp = activity[row].sqrt();
        for i in 0..p {
            residuals[[row, i]] = amp * lam[i % lam.len()] * common
                + dscale[i % dscale.len()] * lcg_normal(&mut seed);
        }
    }
    let model = StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residuals.view(),
        activity: activity.view(),
        max_factor_rank: 2,
    })
    .expect("StructuredResidualModel::fit");
    assert!(
        model.factor_rank() >= 1,
        "fixture must induce a non-trivial factor (Σ_n ≠ diagonal)"
    );
    let metric = model.row_metric(n).expect("row_metric");
    assert_eq!(metric.n_rows(), n);
    assert_eq!(metric.p_out(), p);
    metric
}

/// LOAD-BEARING #2021 acceptance: installing the `WhitenedStructured` metric
/// (a) reports `whitens_likelihood()`, (b) moves the data-fit value, (c) moves
/// the assembled gradient, and (d) leaves the metric-independent penalties
/// untouched.
#[test]
fn structured_residual_metric_whitens_loss_and_gradient_2021() {
    // n=48 sits on the verified stable rank-1 plateau for the strong planted factor
    // (`fit_structured_metric` selects rank 1 at n = 48, 96, 192, 384; #2107). It is
    // comfortably past the small-sample regime where BIC's ½·p·log n Occam penalty
    // legitimately dominates a weak factor (the original n=6 fixture selected rank 0
    // — a correct BIC verdict for THAT weak factor, which vacuously skipped the
    // whitening asserts). With the strong factor a rank-1 structure is justified with
    // margin here, so `fit` genuinely emits a whitening (anisotropic) metric and the
    // assertions below actually execute. n is kept modest so the O(n) loss/gradient
    // assembly stays a cheap unit test.
    let (n, p, k) = (48usize, 3usize, 3usize);
    let mut term = build_term(n, p, k);
    let target = Array2::<f64>::from_shape_fn((n, p), |(r, c)| {
        0.4 - 0.15 * (r as f64) + 0.25 * (c as f64) + 0.05 * ((r * p + c) as f64)
    });
    let rho = SaeManifoldRho::new(-1.0, -6.0, vec![Array1::<f64>::from_elem(1, 0.0); k]);

    // ---- iid (Euclidean) baseline: no metric installed. ----
    assert!(
        term.row_metric().is_none(),
        "precondition: no metric ⇒ isotropic path"
    );
    let loss_iid = term.loss(target.view(), &rho).unwrap();
    let sys_iid = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let g_iid = grad_norm_sq(&sys_iid);

    // ---- install the WhitenedStructured metric and re-evaluate. ----
    let metric = fit_structured_metric(n, p);
    assert!(
        metric.whitens_likelihood(),
        "the fitted StructuredResidualModel metric must whiten the likelihood"
    );
    term.set_row_metric(metric).unwrap();
    assert!(
        term.row_metric().is_some_and(|m| m.whitens_likelihood()),
        "installed metric must report whitens_likelihood()"
    );

    let loss_str = term.loss(target.view(), &rho).unwrap();
    let sys_str = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let g_str = grad_norm_sq(&sys_str);

    // (b) data-fit VALUE moved materially (whitening is active, not a no-op).
    let df_rel = (loss_str.data_fit - loss_iid.data_fit).abs() / (1.0 + loss_iid.data_fit.abs());
    assert!(
        df_rel > 1.0e-3,
        "whitened data_fit ({}) must differ from iid ({}); rel={df_rel:e}",
        loss_str.data_fit,
        loss_iid.data_fit
    );
    assert!(loss_str.data_fit.is_finite() && loss_iid.data_fit.is_finite());

    // (c) assembled GRADIENT moved (the data-fit Jacobian is whitened by M_n).
    let g_rel = (g_str - g_iid).abs() / (1.0 + g_iid.abs());
    assert!(
        g_rel > 1.0e-3,
        "whitened gradient energy ({g_str}) must differ from iid ({g_iid}); rel={g_rel:e}"
    );

    // (d) metric-INDEPENDENT penalties are byte-identical — the metric touches
    // ONLY the reconstruction data-fit (the #974 seam), nothing else.
    assert!(
        (loss_str.assignment_sparsity - loss_iid.assignment_sparsity).abs() < 1.0e-12,
        "assignment-sparsity penalty must not depend on the row metric"
    );
    assert!(
        (loss_str.ard - loss_iid.ard).abs() < 1.0e-12,
        "ARD penalty must not depend on the row metric"
    );
    assert!(
        (loss_str.smoothness - loss_iid.smoothness).abs() < 1.0e-12,
        "decoder-smoothness penalty must not depend on the row metric"
    );
}

/// `fit_row_metric` is the one-shot install seam and must equal
/// `fit(..).row_metric(n)`, and installing it must whiten a term identically —
/// pinning that the production convenience path is a faithful shortcut.
#[test]
fn fit_row_metric_one_shot_matches_fit_then_row_metric_2021() {
    let (n, p) = (6usize, 3usize);
    let lam = [1.0_f64, -0.7, 0.4];
    let dscale = [0.10_f64, 0.55, 0.95];
    let mut seed = 0x2021_FEED_5678_1111u64;
    let mut residuals = Array2::<f64>::zeros((n, p));
    let mut activity = Array1::<f64>::zeros(n);
    for row in 0..n {
        let common = lcg_normal(&mut seed);
        activity[row] = 0.25 + (row as f64) / (n as f64);
        let amp = activity[row].sqrt();
        for i in 0..p {
            residuals[[row, i]] = amp * lam[i] * common + dscale[i] * lcg_normal(&mut seed);
        }
    }
    let input = || ResidualFactorInput {
        residuals: residuals.view(),
        activity: activity.view(),
        max_factor_rank: 2,
    };
    let two_step = StructuredResidualModel::fit(input())
        .unwrap()
        .row_metric(n)
        .unwrap();
    let one_shot = StructuredResidualModel::fit_row_metric(input()).unwrap();
    assert!(two_step.whitens_likelihood() && one_shot.whitens_likelihood());
    let v = Array1::<f64>::from_vec(vec![0.7, -1.3, 0.4]);
    for &row in &[0usize, n / 2, n - 1] {
        let q1 = two_step.quad_form(row, v.view());
        let q2 = one_shot.quad_form(row, v.view());
        assert!(
            (q1 - q2).abs() <= 1.0e-12 * (1.0 + q1.abs()),
            "row {row}: fit_row_metric {q2} must equal fit().row_metric() {q1}"
        );
    }
}
