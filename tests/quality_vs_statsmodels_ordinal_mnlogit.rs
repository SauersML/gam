//! End-to-end quality: gam's penalized categorical-response solver must predict
//! the same per-class probability simplex as the mature Python reference for
//! multinomial-logit regression —
//! `statsmodels.discrete.discrete_model.MNLogit` — on identical data and a
//! byte-identical feature design.
//!
//! ## Why MNLogit is the right mature comparator
//!
//! gam's only categorical-response solver is the penalized multinomial-logit
//! driver `gam::families::multinomial::fit_penalized_multinomial`: a *nominal*
//! softmax with `K-1` independent active linear predictors and a reference class
//! `η_{K-1} ≡ 0`. The matching textbook implementation is statsmodels'
//! `MNLogit` — the same nominal softmax likelihood, fit by maximum likelihood
//! (its default reference/baseline is class 0, but the fitted `P(Y=j|x)` simplex
//! is invariant to which class is taken as baseline, so the per-row
//! probabilities are directly comparable). This is the canonical mature standard
//! for unregularized multinomial-logit and is the model gam actually implements.
//!
//! An earlier draft compared against statsmodels' *ordinal* proportional-odds
//! `OrderedModel`. That is the wrong comparator: the proportional-odds model is a
//! strictly *constrained* submodel (all `K-1` slope vectors tied to a single
//! shared slope plus `K-1` cutpoints), so its MLE minimizes a different
//! likelihood over a smaller parameter space than gam's unrestricted nominal
//! softmax. At finite `n` the two fitted simplices need not — and in general do
//! not — coincide to any tight tolerance, even on PO-generated data. There is no
//! theorem forcing agreement, so a tight bound there would be unjustified and a
//! loose one would assert nothing. Comparing identical models (gam multinomial
//! vs. statsmodels `MNLogit`) is the principled choice and admits a genuinely
//! tight, meaningful bound.
//!
//! ## Identical inputs to both engines
//!
//! We build gam's design once — intercept + linear `x1` + cyclic cubic spline
//! basis of `x2` (`s(x2, bs="cc")`) — via the real formula → design path, then:
//!   * feed that dense design (and the smooth's block penalty) to
//!     `fit_penalized_multinomial`, and
//!   * hand the *same* dense design columns (INCLUDING the intercept column,
//!     since `MNLogit` does not add its own) to `MNLogit`.
//! Both therefore see byte-identical features and the identical integer response.
//! We fit gam with a near-zero ridge (`lambda=1e-3`) so it is effectively the
//! unpenalized multinomial MLE on those basis columns, matching `MNLogit`'s
//! unregularized MLE; the residual penalty is the only intended difference and is
//! negligible at this λ.

use gam::families::multinomial::{MultinomialFitInputs, fit_penalized_multinomial};
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 200;
const J: usize = 5; // number of categorical levels {0,1,2,3,4}
// Latent-variable thresholds that carve the standard-logistic-ish latent into J
// ordinal bins. Y = #{c : latent > cut_c}, so there are J-1 = 4 interior cuts.
// (The generating mechanism is irrelevant to the comparison — both engines fit
// the same nominal multinomial-logit; we only need a well-populated J-level
// categorical response with smooth covariate dependence.)
const CUTS: [f64; 4] = [-1.0, 0.0, 1.0, 2.0];

#[test]
fn gam_multinomial_matches_statsmodels_mnlogit() {
    init_parallelism();

    // ---- synthetic J-level categorical data with smooth covariate effects ---
    // latent = 0.6*x1 + smooth(x2) + N(0,1); smooth is periodic on [0,1) to suit
    // the cyclic basis. Discretize via the fixed CUTS into J levels.
    let mut rng = StdRng::seed_from_u64(20240529);
    let ux = Uniform::new(0.0, 1.0).expect("uniform x");
    let noise = Normal::new(0.0, 1.0).expect("normal noise");
    let mut x1 = vec![0.0f64; N];
    let mut x2 = vec![0.0f64; N];
    let mut y = vec![0.0f64; N];
    for i in 0..N {
        let a = ux.sample(&mut rng);
        let b = ux.sample(&mut rng);
        // periodic smooth in x2: amplitude-1 sinusoid, continuous across the seam
        let smooth = (2.0 * std::f64::consts::PI * b).sin();
        let latent = 0.6 * a + smooth + noise.sample(&mut rng);
        let level = CUTS.iter().filter(|&&c| latent > c).count();
        x1[i] = a;
        x2[i] = b;
        y[i] = level as f64;
    }

    // ---- build gam's design from the formula (intercept + x1 + cc(x2)) -------
    // We need a frozen TermCollectionSpec + materialized design at these rows.
    // The categorical driver takes a raw design + penalty, so we obtain the
    // design by fitting a throwaway Gaussian on the level-coded response with
    // the SAME formula; that yields gam's real cyclic-spline basis columns.
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let rows = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![
                y[i].to_string(),
                x1[i].to_string(),
                x2[i].to_string(),
            ])
        })
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x1 + s(x2, bs=\"cc\")", &ds, &cfg)
        .expect("gam builds the x1 + cc(x2) design");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit to expose the design");
    };

    // Dense design X (N x P): column 0 is the intercept, then x1, then the
    // cyclic-spline basis of x2.
    let design = fit
        .design
        .design
        .try_to_dense_by_chunks("multinomial design")
        .expect("materialize gam design");
    let n = design.nrows();
    let p = design.ncols();
    assert_eq!(n, N, "design row count");
    assert!(p >= 3, "expect intercept + x1 + >=1 smooth column, got P={p}");

    // Shared P x P smooth penalty: assemble the smooth's blockwise local penalty
    // into the global coordinate frame (intercept + x1 blocks are unpenalized).
    let mut penalty = Array2::<f64>::zeros((p, p));
    for blk in &fit.design.penalties {
        let r = blk.col_range.clone();
        penalty.slice_mut(s![r.clone(), r.clone()]).assign(&blk.local);
    }

    // One-hot response Y (N x J).
    let mut y_one_hot = Array2::<f64>::zeros((n, J));
    for i in 0..n {
        let lvl = y[i] as usize;
        assert!(lvl < J, "level out of range");
        y_one_hot[[i, lvl]] = 1.0;
    }

    // ---- fit gam's penalized multinomial-logit at near-zero ridge -----------
    // lambda small => effectively the multinomial MLE on these exact columns,
    // matching the unregularized MNLogit MLE.
    let lambdas = Array1::from_elem(J - 1, 1e-3);
    let out = fit_penalized_multinomial(MultinomialFitInputs {
        design: design.view(),
        y_one_hot: y_one_hot.view(),
        penalty: penalty.view(),
        lambdas: lambdas.view(),
        row_weights: None,
        fisher_w_override: None,
        max_iter: 200,
        tol: 1e-10,
    })
    .expect("gam multinomial fit converges");
    assert!(out.converged, "gam multinomial did not converge");
    let gam_probs = out.fitted_probabilities; // (N, J)
    assert_eq!(gam_probs.dim(), (n, J));

    // ---- fit the SAME model on the SAME features with statsmodels MNLogit ---
    // Pass ALL P design columns (including the intercept at index 0); MNLogit
    // does NOT prepend its own constant, so passing gam's full design makes the
    // two feature spaces byte-identical. The returned (N, J) matrix is the
    // per-row class-probability simplex, row-major.
    let mut cols: Vec<Column<'_>> = Vec::with_capacity(p + 1);
    cols.push(Column::new("y", &y));
    // Materialize every design column (including the intercept) as a named col.
    let design_cols: Vec<Vec<f64>> = (0..p).map(|j| design.column(j).to_vec()).collect();
    let col_names: Vec<String> = (0..p).map(|j| format!("d{j}")).collect();
    for (name, data) in col_names.iter().zip(design_cols.iter()) {
        cols.push(Column::new(name.as_str(), data));
    }

    let py_body = format!(
        r#"
import numpy as np
from statsmodels.discrete.discrete_model import MNLogit

names = {names:?}
X = np.column_stack([np.asarray(df[c], dtype=float) for c in names])
yv = np.asarray(df["y"], dtype=float).astype(int)

# Nominal multinomial-logit (same model gam fits) on the identical features.
# X already contains gam's intercept column, so we do NOT add a constant.
m = MNLogit(yv, X)
res = m.fit(method="newton", maxiter=2000, gtol=1e-10, disp=False)
probs = res.predict(X)                 # (N, J) per-class probabilities, row-major
emit("probs", np.asarray(probs, dtype=float).reshape(-1))
emit("nclass", [probs.shape[1]])
"#,
        names = col_names
    );
    let r = run_python(&cols, &py_body);
    let nclass = r.scalar("nclass") as usize;
    assert_eq!(nclass, J, "statsmodels recovered {nclass} classes, expected {J}");
    let ref_flat = r.vector("probs");
    assert_eq!(ref_flat.len(), n * J, "reference prob matrix size mismatch");

    // ---- compare predicted class probabilities & cumulative probabilities ---
    // Flatten gam's (N,J) row-major to align with statsmodels' row-major flatten.
    // Both index the simplex by class label j, so columns align directly.
    let mut gam_flat = Vec::<f64>::with_capacity(n * J);
    for i in 0..n {
        for j in 0..J {
            gam_flat.push(gam_probs[[i, j]]);
        }
    }

    // Per-row cumulative probabilities P(Y<=j), j = 0..J-2 (the last is 1).
    let mut max_cum_diff_per_level = vec![0.0f64; J - 1];
    for i in 0..n {
        let mut g = 0.0;
        let mut s = 0.0;
        for j in 0..(J - 1) {
            g += gam_probs[[i, j]];
            s += ref_flat[i * J + j];
            max_cum_diff_per_level[j] = max_cum_diff_per_level[j].max((g - s).abs());
        }
    }

    // L2-relative and max-abs on the full class-probability simplex.
    let class_rel_l2 = relative_l2(&gam_flat, ref_flat);
    let class_max_abs = max_abs_diff(&gam_flat, ref_flat);
    let worst_cum = max_cum_diff_per_level.iter().cloned().fold(0.0f64, f64::max);

    eprintln!(
        "multinomial vs statsmodels MNLogit: n={n} J={J} P={p} \
         class_rel_l2={class_rel_l2:.5} class_max_abs={class_max_abs:.5} \
         cum_max_abs_per_level={max_cum_diff_per_level:?} worst_cum={worst_cum:.5} \
         gam_iters={iters} gam_dev={dev:.3}",
        iters = out.iterations,
        dev = out.deviance
    );

    // ---- principled agreement bounds ----------------------------------------
    // gam and MNLogit fit the IDENTICAL nominal multinomial-logit likelihood on
    // the IDENTICAL feature columns. gam adds only a λ=1e-3 ridge on the spline
    // coefficients; both otherwise solve the same convex MLE. The optimizers
    // therefore reach essentially the same β̂, so the fitted per-row simplices
    // must coincide up to that tiny ridge perturbation plus optimizer tolerance.
    // A 1e-2 ridge-scale plus solver-tolerance budget gives:
    //   * full-simplex relative L2 < 5e-3, and max-abs per cell < 5e-3,
    //   * cumulative P(Y<=j) max-abs per level < 5e-3.
    // These are tight enough that any real divergence in gam's softmax solve
    // (wrong reference coding, mis-assembled Fisher curvature, bad penalty
    // embedding) fails the test. A genuine divergence here is a real signal —
    // do not loosen.
    for (level, &d) in max_cum_diff_per_level.iter().enumerate() {
        assert!(
            d < 5e-3,
            "cumulative P(Y<={level}) diverges from statsmodels MNLogit: max-abs={d:.5} (bound 5e-3)"
        );
    }
    assert!(
        class_max_abs < 5e-3,
        "class probabilities diverge from statsmodels MNLogit: max-abs={class_max_abs:.5} (bound 5e-3)"
    );
    assert!(
        class_rel_l2 < 5e-3,
        "class probabilities diverge from statsmodels MNLogit: rel_l2={class_rel_l2:.5} (bound 5e-3)"
    );
}
