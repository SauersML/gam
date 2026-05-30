//! End-to-end OBJECTIVE quality: gam's penalized categorical-response solver must
//! recover the TRUE per-class probability simplex of the data-generating process,
//! measured as RMSE against the analytic ground-truth probabilities. statsmodels'
//! `MNLogit` is fit on the identical data and design only as a BASELINE TO BEAT on
//! that same truth-recovery metric — never as a target to reproduce.
//!
//! ## The objective metric
//!
//! The synthetic data has a KNOWN generating mechanism, so the per-row class
//! probabilities are an exact analytic quantity (see below). The pass/fail claim
//! is therefore truth recovery, not tool-agreement:
//!
//!   * PRIMARY: `RMSE(gam_simplex, truth_simplex)` is below a principled bar. The
//!     softmax/multinomial-logit family is *misspecified* relative to the ordered-
//!     probit generator, so the recoverable simplex carries irreducible
//!     approximation error; the bar is set from the signal scale, not from any
//!     reference tool's output.
//!   * MATCH-OR-BEAT: gam's truth-recovery RMSE is no worse than statsmodels'
//!     MNLogit RMSE times 1.10. Both fit the identical nominal-softmax likelihood
//!     on byte-identical features, so gam must be at least as accurate at recovering
//!     the truth as the mature reference. (We additionally print the gam-vs-MNLogit
//!     simplex relative-L2 with `eprintln!` purely for context.)
//!
//! Matching MNLogit's fitted numbers is explicitly NOT the criterion: two
//! maximum-likelihood fits of the same misspecified model could agree closely while
//! both being a poor approximation of the truth. The truth-recovery RMSE measures
//! the only thing that matters — how close the predicted simplex is to reality.
//!
//! ## Analytic ground-truth simplex
//!
//! The latent score is `latent = m(x) + ε`, `ε ~ N(0,1)`, with systematic part
//! `m(x) = 0.6*x1 + sin(2π x2)`. The response is the count of exceeded cutpoints,
//! `Y = #{c : latent > CUTS[c]}`. Hence, for the J-1 cuts in ascending order,
//!   `P(Y >= k | x) = P(ε > CUTS[k-1] - m(x)) = Φ(m(x) - CUTS[k-1])`,
//! and the class probabilities are the successive differences
//!   `P(Y = 0) = 1 - Φ(m - CUTS[0])`,
//!   `P(Y = j) = Φ(m - CUTS[j-1]) - Φ(m - CUTS[j])`, `1 <= j <= J-2`,
//!   `P(Y = J-1) = Φ(m - CUTS[J-2])`.
//! This is the exact ordered-probit simplex the data were drawn from; we use it as
//! ground truth for both engines.
//!
//! ## Identical inputs to both engines
//!
//! We build gam's design once — intercept + linear `x1` + cyclic cubic spline basis
//! of `x2` (`s(x2, bs="cc")`) — via the real formula → design path, then feed that
//! dense design (and the smooth's block penalty) to `fit_penalized_multinomial`,
//! and hand the *same* dense design columns (including the intercept) to `MNLogit`.
//! Both see byte-identical features and the identical integer response. gam uses a
//! near-zero ridge (`lambda=1e-3`) so it is effectively the unpenalized multinomial
//! MLE on those basis columns, matching `MNLogit`'s unregularized MLE.

use gam::families::multinomial::{MultinomialFitInputs, fit_penalized_multinomial};
use gam::test_support::reference::{Column, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 200;
const J: usize = 5; // number of categorical levels {0,1,2,3,4}
// Latent-variable thresholds that carve the standard-normal-noise latent into J
// ordinal bins. Y = #{c : latent > cut_c}, so there are J-1 = 4 interior cuts.
const CUTS: [f64; 4] = [-1.0, 0.0, 1.0, 2.0];

/// Standard normal CDF Φ via the error function identity Φ(z) = ½(1 + erf(z/√2)),
/// with an Abramowitz–Stegun 7.1.26 rational approximation of erf (|err| < 1.5e-7).
/// Used to evaluate the exact ordered-probit ground-truth simplex; precision far
/// exceeds the multinomial approximation error the test actually measures.
fn norm_cdf(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign * (1.0 - poly * (-ax * ax).exp());
    0.5 * (1.0 + erf)
}

/// Exact ground-truth class probabilities for the ordered-probit generator at a
/// systematic value `m = 0.6*x1 + sin(2π x2)`. Returns the length-J simplex.
fn truth_simplex(m: f64) -> [f64; J] {
    let surv = |k: usize| norm_cdf(m - CUTS[k]); // P(Y >= k+1) = Φ(m - CUTS[k])
    let mut p = [0.0f64; J];
    p[0] = 1.0 - surv(0);
    for j in 1..(J - 1) {
        p[j] = surv(j - 1) - surv(j);
    }
    p[J - 1] = surv(J - 2);
    p
}

#[test]
fn gam_multinomial_recovers_true_class_simplex() {
    init_parallelism();

    // ---- synthetic J-level categorical data with smooth covariate effects ---
    // latent = 0.6*x1 + sin(2π x2) + N(0,1); the sinusoid is periodic on [0,1) to
    // suit the cyclic basis. Discretize via the fixed CUTS into J levels. We also
    // record the systematic part m(x) per row to evaluate the analytic truth.
    let mut rng = StdRng::seed_from_u64(20240529);
    let ux = Uniform::new(0.0, 1.0).expect("uniform x");
    let noise = Normal::new(0.0, 1.0).expect("normal noise");
    let mut x1 = vec![0.0f64; N];
    let mut x2 = vec![0.0f64; N];
    let mut y = vec![0.0f64; N];
    let mut m_sys = vec![0.0f64; N];
    for i in 0..N {
        let a = ux.sample(&mut rng);
        let b = ux.sample(&mut rng);
        let smooth = (2.0 * std::f64::consts::PI * b).sin();
        let m = 0.6 * a + smooth;
        let latent = m + noise.sample(&mut rng);
        let level = CUTS.iter().filter(|&&c| latent > c).count();
        x1[i] = a;
        x2[i] = b;
        y[i] = level as f64;
        m_sys[i] = m;
    }

    // Analytic ground-truth simplex, row-major (N, J).
    let mut truth_flat = Vec::<f64>::with_capacity(N * J);
    for i in 0..N {
        let p = truth_simplex(m_sys[i]);
        for j in 0..J {
            truth_flat.push(p[j]);
        }
    }

    // ---- build gam's design from the formula (intercept + x1 + cc(x2)) -------
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let rows = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![y[i].to_string(), x1[i].to_string(), x2[i].to_string()])
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

    // Dense design X (N x P): intercept, x1, then the cyclic-spline basis of x2.
    let design = fit
        .design
        .design
        .try_to_dense_by_chunks("multinomial design")
        .expect("materialize gam design");
    let n = design.nrows();
    let p = design.ncols();
    assert_eq!(n, N, "design row count");
    assert!(
        p >= 3,
        "expect intercept + x1 + >=1 smooth column, got P={p}"
    );

    // Shared P x P smooth penalty (intercept + x1 blocks are unpenalized).
    let mut penalty = Array2::<f64>::zeros((p, p));
    for blk in &fit.design.penalties {
        let r = blk.col_range.clone();
        penalty
            .slice_mut(s![r.clone(), r.clone()])
            .assign(&blk.local);
    }

    // One-hot response Y (N x J).
    let mut y_one_hot = Array2::<f64>::zeros((n, J));
    for i in 0..n {
        let lvl = y[i] as usize;
        assert!(lvl < J, "level out of range");
        y_one_hot[[i, lvl]] = 1.0;
    }

    // ---- fit gam's penalized multinomial-logit at near-zero ridge -----------
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

    // gam's simplex flattened row-major to align with truth_flat.
    let mut gam_flat = Vec::<f64>::with_capacity(n * J);
    for i in 0..n {
        for j in 0..J {
            gam_flat.push(gam_probs[[i, j]]);
        }
    }

    // ---- fit the SAME model on the SAME features with statsmodels MNLogit ----
    // Baseline only: its predicted simplex is scored against the SAME ground truth.
    let mut cols: Vec<Column<'_>> = Vec::with_capacity(p + 1);
    cols.push(Column::new("y", &y));
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
    assert_eq!(
        nclass, J,
        "statsmodels recovered {nclass} classes, expected {J}"
    );
    let ref_flat = r.vector("probs");
    assert_eq!(ref_flat.len(), n * J, "reference prob matrix size mismatch");

    // ---- OBJECTIVE metric: RMSE of each fitted simplex vs analytic truth ----
    let gam_truth_rmse = rmse(&gam_flat, &truth_flat);
    let ref_truth_rmse = rmse(ref_flat, &truth_flat);

    // For context only (NOT a pass criterion): how close the two fits are to
    // each other on the simplex. Two MLEs of the same misspecified model can
    // agree closely yet both miss the truth; this number is informational.
    let gam_vs_ref_rel_l2 = relative_l2(&gam_flat, ref_flat);

    eprintln!(
        "multinomial truth recovery: n={n} J={J} P={p} \
         gam_truth_rmse={gam_truth_rmse:.5} ref_truth_rmse={ref_truth_rmse:.5} \
         gam_vs_ref_rel_l2={gam_vs_ref_rel_l2:.5} \
         gam_iters={iters} gam_dev={dev:.3}",
        iters = out.iterations,
        dev = out.deviance
    );

    // PRIMARY claim: gam recovers the true class simplex. The softmax family is
    // misspecified relative to the ordered-probit generator, so some
    // approximation error is irreducible; the bar 0.06 is roughly a quarter of
    // the typical per-class probability mass (~1/J = 0.2) and well below the
    // signal it must capture. A genuinely broken softmax solve (wrong reference
    // coding, mis-assembled Fisher curvature, bad penalty embedding) blows past
    // this and fails — do not loosen.
    assert!(
        gam_truth_rmse < 0.06,
        "gam fails to recover the true class simplex: RMSE-vs-truth={gam_truth_rmse:.5} (bound 0.06)"
    );

    // MATCH-OR-BEAT: gam is at least as accurate at recovering the truth as the
    // mature MNLogit baseline, fit on identical data/features. gam carries only a
    // negligible λ=1e-3 ridge, so it must not be meaningfully worse.
    assert!(
        gam_truth_rmse <= ref_truth_rmse * 1.10,
        "gam is less accurate than statsmodels MNLogit at recovering the truth: \
         gam_rmse={gam_truth_rmse:.5} > 1.10 * ref_rmse={ref_truth_rmse:.5}"
    );
}
