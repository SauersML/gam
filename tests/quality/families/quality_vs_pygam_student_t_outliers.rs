//! End-to-end quality: the scaled Student-t response family on a smooth signal
//! contaminated by heavy-tailed outliers (the pyGAM audit's `outlier_n300`),
//! judged on **truth MSE** — the held-out squared error against the noise-free
//! mean `sin(4πx)` — with pyGAM's default `LinearGAM(s(0))` as the baseline to
//! match-or-beat on identical folds.
//!
//! The data, the folds and pyGAM's out-of-fold predictions are produced by the
//! reference interpreter itself (`numpy.random.default_rng(n + 3)` and
//! `KFold(5, shuffle=True, random_state=0)`), so gam and pyGAM see bit-identical
//! rows. A Gaussian REML fit on this data is dragged to a straight line on some
//! folds by the 5% Cauchy-like contamination; the Student-t fit estimates its
//! scale and degrees of freedom by LAML jointly with the smoothing parameter,
//! downweights the outliers, and must keep the curvature.
//!
//! Asserted per fold: the fit keeps more effective degrees of freedom than the
//! intercept-plus-linear null space of `s(x)` (a collapse to linear is exactly
//! that null space) and reports a finite heavy-tailed `ν̂`. Asserted overall: the
//! mean truth MSE is no worse than pyGAM's.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, run_python};
use gam::types::ResponseFamily;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const N: usize = 300;
const FOLDS: usize = 5;
/// Intercept plus the linear direction the second-derivative penalty of `s(x)`
/// leaves unpenalized: the effective dimension of a fit collapsed to a line.
const LINEAR_NULL_SPACE_EDF: f64 = 2.0;

#[test]
fn student_t_matches_or_beats_pygam_truth_mse_on_outlier_contaminated_smooth() {
    init_parallelism();

    let row: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let py = run_python(
        &[Column::new("row", &row)],
        r#"
from pygam import LinearGAM, s
from sklearn.model_selection import KFold
n = len(df["row"])
rng = np.random.default_rng(n + 3)
x = rng.uniform(0, 1, n)
mu = np.sin(2 * np.pi * 2 * x)
y = mu + rng.normal(0, 0.3, n)
idx = rng.choice(n, n // 20, replace=False)
y[idx] += rng.standard_t(1.5, len(idx)) * 5
fold = np.zeros(n)
pred = np.zeros(n)
for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(x)):
    fold[te] = k
    pred[te] = LinearGAM(s(0)).fit(x[tr, None], y[tr]).predict(x[te, None])
emit("x", x)
emit("y", y)
emit("mu", mu)
emit("fold", fold)
emit("pygam_pred", pred)
"#,
    );
    let x = py.vector("x");
    let y = py.vector("y");
    let mu = py.vector("mu");
    let fold = py.vector("fold");
    let pygam_pred = py.vector("pygam_pred");

    let cfg = FitConfig {
        family: Some("student-t".to_string()),
        ..FitConfig::default()
    };
    let mut gam_sq = Vec::with_capacity(N);
    let mut pygam_sq = Vec::with_capacity(N);
    for k in 0..FOLDS {
        let held_out = |i: &usize| fold[*i] as usize == k;
        let train: Vec<usize> = (0..N).filter(|i| !held_out(i)).collect();
        let test: Vec<usize> = (0..N).filter(held_out).collect();

        let records: Vec<StringRecord> = train
            .iter()
            .map(|&i| StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
            .collect();
        let train_ds = encode_recordswith_inferred_schema(vec!["x".into(), "y".into()], records)
            .expect("encode training fold");
        let x_col = train_ds.column_map()["x"];
        let result =
            fit_from_formula("y ~ s(x)", &train_ds, &cfg).expect("Student-t fit on training fold");
        let FitResult::Standard(fit) = result else {
            panic!("a Student-t location smooth must be a standard GAM fit");
        };

        let edf = fit
            .fit
            .edf_total()
            .expect("a converged fit reports its EDF");
        let (sigma, nu) = match fit.fit.likelihood_family.as_ref().map(|f| &f.response) {
            Some(ResponseFamily::StudentT { sigma, nu }) => (*sigma, *nu),
            other => panic!("fold {k}: Student-t fit must carry (sigma, nu); got {other:?}"),
        };

        let mut grid = Array2::<f64>::zeros((test.len(), train_ds.headers.len()));
        for (r, &i) in test.iter().enumerate() {
            grid[[r, x_col]] = x[i];
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild design at held-out rows");
        let pred = design.design.apply(&fit.fit.beta);
        let fold_mse = test
            .iter()
            .zip(pred.iter())
            .map(|(&i, p)| (p - mu[i]).powi(2))
            .sum::<f64>()
            / test.len() as f64;
        eprintln!(
            "outlier_n300 fold {k}: edf={edf:.3} sigma={sigma:.4} nu={nu:.3} truth_mse={fold_mse:.5}"
        );

        assert!(
            edf > LINEAR_NULL_SPACE_EDF + 1.0,
            "fold {k}: the Student-t fit collapsed toward a line (edf = {edf:.3})"
        );
        assert!(
            nu.is_finite() && sigma.is_finite() && sigma > 0.0,
            "fold {k}: fitted (sigma, nu) = ({sigma}, {nu}) must be finite"
        );
        for (&i, p) in test.iter().zip(pred.iter()) {
            gam_sq.push((p - mu[i]).powi(2));
            pygam_sq.push((pygam_pred[i] - mu[i]).powi(2));
        }
    }

    let gam_mse = gam_sq.iter().sum::<f64>() / gam_sq.len() as f64;
    let pygam_mse = pygam_sq.iter().sum::<f64>() / pygam_sq.len() as f64;
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_pygam_student_t_outliers",
            "truth_mse",
            gam_mse,
            "pygam",
            pygam_mse,
        )
        .line()
    );
    assert!(
        gam_mse <= pygam_mse,
        "Student-t truth MSE {gam_mse:.5} is worse than pyGAM's {pygam_mse:.5} on outlier_n300"
    );
}
