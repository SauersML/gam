//! Encode-side proof (#2021 / whitened-GLS): once a non-diagonal
//! [`MetricProvenance::WhitenedStructured`](gam_problem::MetricProvenance::WhitenedStructured)
//! metric `M_n = Σ_n^{-1}` is installed on a [`SaeManifoldTerm`], the frozen-decoder
//! coordinate READ (the encode) lands at the `M`-metric projection of the target
//! onto the atom's image — i.e. the normalized-GLS estimate
//! `argmin_t (x − f(t))ᵀ M (x − f(t))` — and that estimate is materially DIFFERENT
//! from the isotropic (naive `argmin_t ‖x − f(t)‖²`) read whenever `M` is
//! anisotropic. This is the executable guard that the assembly's
//! `htt = J M Jᵀ` / `gt = J M r` normalization (construction.rs:4842-4892) is
//! actually engaged in the coordinate solve, not silently dropped.
//!
//! Unlike `tests_structured_residual_2021` (which *fits* the metric via
//! `StructuredResidualModel` and can be defeated by a rank-0 residual selection on
//! a small fixture), this test HAND-BUILDS the precision `M = U Uᵀ` from a rotated
//! anisotropic spectrum, so the whitening is guaranteed non-trivial and the guard
//! cannot be knocked out by rank-selection drift.

use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_problem::RowMetric;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Signed wrapped distance between two Circle coordinates of period 1.0.
fn wrap_dist(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(1.0);
    (if d > 0.5 { d - 1.0 } else { d }).abs()
}

/// A 3×3 rotated-anisotropic precision `M = R Λ Rᵀ` (SPD, non-diagonal) and its
/// factor `U = R Λ^{1/2}` (so `U Uᵀ = M`, the layout `whitened_structured` wants).
/// The rotation lives in the e0–e1 plane (the atom's image plane), so the
/// anisotropy genuinely warps the read direction; e2 is decoupled.
fn rotated_anisotropic_factor(beta: f64, lambda: [f64; 3]) -> (Array2<f64>, Array2<f64>) {
    let (c, s) = (beta.cos(), beta.sin());
    // R embeds a 2D rotation in the e0–e1 block.
    let r = ndarray::array![[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let mut u = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for k in 0..3 {
            u[[i, k]] = r[[i, k]] * lambda[k].sqrt();
        }
    }
    // M = U Uᵀ.
    let mut m = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += u[[i, k]] * u[[j, k]];
            }
            m[[i, j]] = acc;
        }
    }
    (u, m)
}

/// Build a fresh K=1 periodic (circle) term whose decoder puts cos→e0, sin→e1
/// (const→0), over `n` identical rows, coordinates seeded at `t_start`. Priors are
/// nulled (log α ≈ −50 ⇒ von-Mises/Gaussian energy ≈ 0) so the read is pure GLS.
fn build_circle_term(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    n: usize,
    p: usize,
    t_start: f64,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let coords = Array2::<f64>::from_elem((n, 1), t_start);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, 0]] = 1.0; // cos → e0
    decoder[[2, 1]] = 1.0; // sin → e1
    let atom = SaeManifoldAtom::new(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    term.set_guards_enabled(false);
    // Null the coordinate prior so the frozen-decoder read is pure (metric-only) GLS.
    let rho = SaeManifoldRho::new(-50.0, -50.0, vec![Array1::<f64>::from_elem(1, -50.0)]);
    (term, rho)
}

/// The circle image `f(t) = Φ(t) · decoder ∈ ℝ^p` on a fine grid, from the SAME
/// evaluator the solver uses (so the reference makes no basis-convention
/// assumption). Returns `(grid_t, f)` with `f` shape `(G, p)`.
fn image_grid(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    decoder: &Array2<f64>,
    g: usize,
    p: usize,
) -> (Vec<f64>, Array2<f64>) {
    let grid_t: Vec<f64> = (0..g).map(|i| i as f64 / g as f64).collect();
    let coords = Array2::<f64>::from_shape_fn((g, 1), |(i, _)| grid_t[i]);
    let (phi, _jet) = evaluator.evaluate(coords.view()).unwrap();
    // f = phi (G,3) · decoder (3,p).
    let mut f = Array2::<f64>::zeros((g, p));
    for row in 0..g {
        for out in 0..p {
            let mut acc = 0.0;
            for b in 0..3 {
                acc += phi[[row, b]] * decoder[[b, out]];
            }
            f[[row, out]] = acc;
        }
    }
    (grid_t, f)
}

/// Grid argmin over the circle of the `M`-metric residual `(x−f)ᵀ M (x−f)`
/// (`M = None` ⇒ identity / naive).
fn argmin_metric_projection(
    grid_t: &[f64],
    f: &Array2<f64>,
    x: &Array1<f64>,
    m: Option<&Array2<f64>>,
    p: usize,
) -> f64 {
    let mut best_t = 0.0;
    let mut best = f64::INFINITY;
    for (i, &t) in grid_t.iter().enumerate() {
        let mut r = Array1::<f64>::zeros(p);
        for out in 0..p {
            r[out] = x[out] - f[[i, out]];
        }
        let obj = match m {
            None => r.dot(&r),
            Some(mat) => {
                // rᵀ M r.
                let mut acc = 0.0;
                for a in 0..p {
                    let mut mr = 0.0;
                    for b in 0..p {
                        mr += mat[[a, b]] * r[b];
                    }
                    acc += r[a] * mr;
                }
                acc
            }
        };
        if obj < best {
            best = obj;
            best_t = t;
        }
    }
    best_t
}

/// LOAD-BEARING encode guard: an installed non-diagonal `WhitenedStructured`
/// metric makes the frozen-decoder coordinate read land at the `M`-metric
/// projection (normalized GLS), which is materially DIFFERENT from the isotropic
/// read. Proves `(BᵀΣ⁻¹B)⁻¹BᵀΣ⁻¹x` is engaged in the coordinate solve, not dropped.
#[test]
fn whitened_metric_engages_normalized_gls_coordinate_read_2021() {
    let n = 8usize;
    let p = 3usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());

    // Target: an in-plane point off the unit image so the projection ANGLE is
    // well-defined and metric-sensitive. All rows identical ⇒ one shared read.
    let x = ndarray::array![1.0, 0.7, 0.0];
    let target = Array2::<f64>::from_shape_fn((n, p), |(_, c)| x[c]);

    // Non-diagonal precision M (rotated 12×/1×/1× anisotropy in the image plane).
    let (u_mat, m_mat) = rotated_anisotropic_factor(0.9, [12.0, 1.0, 1.0]);
    // Per-row factor stack u[row, i*rank + k] = U[i,k]; rank = p; identical rows.
    let u = Array2::<f64>::from_shape_fn((n, p * p), |(_, col)| {
        let i = col / p;
        let k = col % p;
        u_mat[[i, k]]
    });
    let metric = RowMetric::whitened_structured(Arc::new(u), p, p).unwrap();
    assert!(metric.whitens_likelihood(), "hand-built metric must whiten");

    // Analytic references from the SAME evaluator basis.
    let decoder = {
        let mut d = Array2::<f64>::zeros((3, p));
        d[[1, 0]] = 1.0;
        d[[2, 1]] = 1.0;
        d
    };
    let g = 200_000usize;
    let (grid_t, f_grid) = image_grid(&evaluator, &decoder, g, p);
    let t_ref_m = argmin_metric_projection(&grid_t, &f_grid, &x, Some(&m_mat), p);
    let t_ref_i = argmin_metric_projection(&grid_t, &f_grid, &x, None, p);

    // The metric must MATERIALLY move the read (else it is a scalar/no-op metric
    // and this test would not distinguish whiten from naive).
    let separation = wrap_dist(t_ref_m, t_ref_i);
    assert!(
        separation > 0.02,
        "fixture must make the M-projection differ from the naive read (else the \
         test cannot bite); separation = {separation:.5} (t_ref_M={t_ref_m:.5}, \
         t_ref_I={t_ref_i:.5})"
    );

    // --- Whitened solve: install M, run the frozen-decoder coordinate read. ---
    let t_start = 0.02; // inside the min basin, away from both references
    let (mut term_w, mut rho_w) = build_circle_term(&evaluator, n, p, t_start);
    term_w.set_row_metric(metric).unwrap();
    term_w
        .run_fixed_decoder_arrow_schur(target.view(), &mut rho_w, None, 300, 1.0, 1e-9)
        .expect("whitened frozen-decoder read");
    let t_white = term_w.assignment.coords[0].row(0)[0].rem_euclid(1.0);

    // --- Naive solve: no metric installed (isotropic path). ---
    let (mut term_i, mut rho_i) = build_circle_term(&evaluator, n, p, t_start);
    term_i
        .run_fixed_decoder_arrow_schur(target.view(), &mut rho_i, None, 300, 1.0, 1e-9)
        .expect("naive frozen-decoder read");
    let t_naive = term_i.assignment.coords[0].row(0)[0].rem_euclid(1.0);

    // The whitened read matches the M-metric projection (normalized GLS)...
    assert!(
        wrap_dist(t_white, t_ref_m) < 2.0e-3,
        "whitened coordinate read {t_white:.6} must land at the M-projection \
         {t_ref_m:.6} (Δ={:.2e})",
        wrap_dist(t_white, t_ref_m)
    );
    // ...the naive read matches the isotropic projection...
    assert!(
        wrap_dist(t_naive, t_ref_i) < 2.0e-3,
        "naive coordinate read {t_naive:.6} must land at the isotropic projection \
         {t_ref_i:.6} (Δ={:.2e})",
        wrap_dist(t_naive, t_ref_i)
    );
    // ...and the two reads genuinely differ (the metric changed the encode).
    assert!(
        wrap_dist(t_white, t_naive) > 0.02,
        "the whitened read {t_white:.6} must differ materially from the naive read \
         {t_naive:.6} — the (BᵀΣ⁻¹B)⁻¹BᵀΣ⁻¹ normalization is engaged"
    );
}
