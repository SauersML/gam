//! End-to-end quality: gam's *custom-family* machinery must reproduce a mature
//! GLM reference (`statsmodels.GLM`) on a **non-canonical link**.
//!
//! The capability under test is `fit_custom_family` driving a hand-written
//! `CustomFamily` that encodes a Poisson likelihood with the **identity** link
//! `μ = η = X·β`. Identity-link Poisson is the canonical stress test for a
//! custom family: the variance-stabilizing transform (`sqrt`) and the link
//! (`identity`) disagree, so the family must linearize the mean correctly
//! through the IRLS pseudo-response `z = η + (y-μ)/μ'` and Fisher weight
//! `w = (μ')²/V(μ) = 1/μ` rather than borrowing the canonical log-link weights
//! `w = μ`. A wrong gradient/weight derivation still "runs" but converges to a
//! different coefficient vector — exactly what this test catches.
//!
//! Both engines fit the **same** dense design matrix (intercept plus a centered
//! cubic basis of a synthetic smooth covariate) with **no penalty**, so each
//! solves the identical unpenalized Poisson-identity maximum-likelihood problem.
//! Because that objective is concave on `{β : Xβ > 0}` with a unique optimum,
//! close agreement is the correct expectation and any real divergence is a real
//! bug in gam's custom-family linearization. We assert:
//!   1. fitted coefficients agree (relative L2 ≤ 0.01),
//!   2. fitted means are near-identical (Pearson ≥ 0.9999), and
//!   3. the Poisson deviance agrees within 2%.
//!
//! The gam-side deviance is recomputed from the fitted means via the standard
//! GLM formula `D = 2·Σ[y·log(y/μ) − (y−μ)]`, and statsmodels reports its own
//! `res.deviance`, which uses that identical Poisson deviance definition. The
//! comparison is therefore independent of gam's internal `-2ℓ` bookkeeping
//! convention.

use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState,
};
use gam::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::init_parallelism;
use ndarray::{Array1, Array2};

/// Poisson likelihood with the identity link `μ = η`.
///
/// `eta = X·β + offset` is supplied by the solver in `block_states[0].eta`.
/// For the identity link `μ = η`, `μ' = dμ/dη = 1`, so the IRLS working set is
/// the textbook Fisher-scoring GLM iteration:
///   * pseudo-response `z = η + (y − μ)/μ' = η + (y − μ)`
///   * working weight  `w = (μ')² / V(μ) = 1/μ`
/// which is precisely the iteration `statsmodels.GLM(...).fit()` performs, so a
/// correct family converges to the same MLE on the same design.
#[derive(Clone)]
struct PoissonIdentityFamily {
    /// Observed counts, one per row.
    y: Array1<f64>,
}

/// Floor for `μ` so the IRLS weight `1/μ` and `log μ` stay finite while the
/// inner Newton iterates cross the feasibility boundary; the converged mode of
/// this dataset has `μ` comfortably above this floor, so it does not perturb
/// the optimum (it only guards transient infeasible trial points). statsmodels'
/// own identity-link Poisson IRLS uses an equivalent positivity guard.
const MU_FLOOR: f64 = 1e-8;

impl CustomFamily for PoissonIdentityFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let eta = &block_states[0].eta;
        let n = eta.len();
        if n != self.y.len() {
            return Err(format!(
                "PoissonIdentityFamily: eta len {n} != response len {}",
                self.y.len()
            ));
        }
        let mut ll = 0.0;
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu = eta[i].max(MU_FLOOR);
            let yi = self.y[i];
            // Poisson log-likelihood (dropping the response-only log(y!) constant,
            // which does not affect the MLE): ℓ_i = y·log μ − μ.
            ll += yi * mu.ln() - mu;
            // Identity link ⇒ μ' = 1: z = η + (y − μ), w = 1/μ.
            z[i] = eta[i] + (yi - mu);
            w[i] = 1.0 / mu;
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: z,
                working_weights: w,
            }],
        })
    }
}

/// Standard GLM Poisson deviance `D = 2·Σ[y·log(y/μ) − (y−μ)]`, with the
/// `y·log(y/μ)` term taken as 0 when `y = 0` (its limit). Identical definition
/// on both engines so the comparison is convention-free.
fn poisson_deviance(y: &[f64], mu: &[f64]) -> f64 {
    assert_eq!(y.len(), mu.len(), "poisson_deviance length mismatch");
    let mut d = 0.0;
    for (&yi, &mui) in y.iter().zip(mu.iter()) {
        let term = if yi > 0.0 { yi * (yi / mui).ln() } else { 0.0 };
        d += 2.0 * (term - (yi - mui));
    }
    d
}

#[test]
fn custom_poisson_identity_link_matches_statsmodels() {
    init_parallelism();

    // ---- synthetic data: n=200, X ~ U(0,10), Y ~ Poisson(exp(0.5+0.3·s(X))) --
    // Deterministic LCG + Knuth Poisson sampler so the test is reproducible and
    // both engines consume byte-identical data via the shared CSV harness.
    let n = 200usize;
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next_u01 = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // top 53 bits -> (0,1)
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = 10.0 * next_u01();
        // A genuinely smooth nonlinear mean on the *log* scale (so counts are
        // always positive); the fitted model is identity-link, which is the
        // link/variance mismatch we are stress-testing.
        let s = (0.6 * xi).sin() + 0.15 * xi;
        let mean = (0.5 + 0.3 * s).exp();
        // Knuth's Poisson sampler.
        let l = (-mean).exp();
        let mut k = 0.0;
        let mut p = 1.0;
        loop {
            p *= next_u01();
            if p <= l {
                break;
            }
            k += 1.0;
        }
        x.push(xi);
        y.push(k);
    }

    // ---- build ONE dense design shared by both engines ---------------------
    // Intercept + centered cubic basis of a standardized covariate. Raw numeric
    // columns are handed verbatim to statsmodels (no spline reimplementation),
    // guaranteeing identical-data, identical-design comparison.
    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let var_x: f64 = x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / n as f64;
    let sd_x = var_x.sqrt().max(1e-12);
    let z: Vec<f64> = x.iter().map(|&v| (v - mean_x) / sd_x).collect();
    let z2_raw: Vec<f64> = z.iter().map(|v| v * v).collect();
    let z3_raw: Vec<f64> = z.iter().map(|v| v * v * v).collect();
    let m2: f64 = z2_raw.iter().sum::<f64>() / n as f64;
    let m3: f64 = z3_raw.iter().sum::<f64>() / n as f64;
    let z2: Vec<f64> = z2_raw.iter().map(|v| v - m2).collect();
    let z3: Vec<f64> = z3_raw.iter().map(|v| v - m3).collect();

    let p = 4usize; // [1, z, z2, z3]
    let mut xmat = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        xmat[[i, 0]] = 1.0;
        xmat[[i, 1]] = z[i];
        xmat[[i, 2]] = z2[i];
        xmat[[i, 3]] = z3[i];
    }

    // ---- fit with gam's custom-family engine (unpenalized Poisson-identity) --
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
    let y_arr = Array1::from(y.clone());
    let family = PoissonIdentityFamily { y: y_arr };
    // Intercept-only positive start so μ = Xβ > 0 everywhere initially; the
    // other coefficients start at 0. The optimum is unique, so the start only
    // governs feasibility, not the answer.
    let mut beta0 = Array1::<f64>::zeros(p);
    beta0[0] = mean_y.max(1.0);
    let spec = ParameterBlockSpec {
        name: "poisson_identity".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(xmat.clone())),
        offset: Array1::<f64>::zeros(n),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: Some(beta0),
        ..ParameterBlockSpec::defaults()
    };
    let options = BlockwiseFitOptions::default();
    let result =
        gam::custom_family::fit_custom_family(&family, std::slice::from_ref(&spec), &options)
            .expect("gam custom Poisson-identity fit");
    let beta_gam: Vec<f64> = result.blocks[0].beta.to_vec();
    assert_eq!(beta_gam.len(), p, "gam returned {} coeffs, expected {p}", beta_gam.len());

    // Fitted means under the identity link: μ = X·β.
    let beta_gam_arr = Array1::from(beta_gam.clone());
    let fitted_gam: Vec<f64> = xmat.dot(&beta_gam_arr).to_vec();
    let dev_gam = poisson_deviance(&y, &fitted_gam);

    // ---- fit the SAME design with statsmodels (the mature GLM reference) ----
    // The CSV carries the response and every design column, so statsmodels sees
    // byte-identical data and the identical design (exog = the 4 columns, no
    // added intercept since column c0 already is the intercept).
    let r = run_python(
        &[
            Column::new("y", &y),
            Column::new("c0", &xmat.column(0).to_vec()),
            Column::new("c1", &xmat.column(1).to_vec()),
            Column::new("c2", &xmat.column(2).to_vec()),
            Column::new("c3", &xmat.column(3).to_vec()),
        ],
        r#"
import statsmodels.api as sm
yv = np.asarray(df["y"], dtype=float)
X = np.column_stack([
    np.asarray(df["c0"], dtype=float),
    np.asarray(df["c1"], dtype=float),
    np.asarray(df["c2"], dtype=float),
    np.asarray(df["c3"], dtype=float),
])
fam = sm.families.Poisson(link=sm.families.links.Identity())
model = sm.GLM(yv, X, family=fam)
res = model.fit(maxiter=300)
emit("beta", np.asarray(res.params, dtype=float))
emit("fitted", np.asarray(res.fittedvalues, dtype=float))
emit("deviance", [float(res.deviance)])
"#,
    );
    let beta_sm = r.vector("beta");
    let fitted_sm = r.vector("fitted");
    let dev_sm = r.scalar("deviance");

    assert_eq!(beta_sm.len(), p, "statsmodels returned {} coeffs", beta_sm.len());
    assert_eq!(fitted_sm.len(), n, "statsmodels fitted length mismatch");

    // ---- compare -----------------------------------------------------------
    let beta_rel = relative_l2(&beta_gam, beta_sm);
    let fitted_corr = pearson(&fitted_gam, fitted_sm);
    let dev_rel = (dev_gam - dev_sm).abs() / dev_sm.abs().max(1e-12);

    eprintln!(
        "poisson-identity custom family: n={n} p={p} \
         beta_rel_l2={beta_rel:.5} fitted_pearson={fitted_corr:.6} \
         dev_gam={dev_gam:.4} dev_sm={dev_sm:.4} dev_rel={dev_rel:.5}"
    );
    eprintln!("beta_gam = {beta_gam:?}");
    eprintln!("beta_sm  = {beta_sm:?}");

    // Both engines solve the identical unpenalized concave Poisson-identity MLE
    // on the identical design, so they must reach the same optimum. These are
    // the spec's principled bounds: a custom family that mislinearizes the
    // non-canonical mean gradient/weight would shift β and fail relative-L2 long
    // before pushing fitted-mean correlation below 0.9999.
    assert!(
        beta_rel <= 0.01,
        "fitted coefficients diverge from statsmodels: rel_l2={beta_rel:.5}"
    );
    assert!(
        fitted_corr >= 0.9999,
        "fitted means diverge from statsmodels: pearson={fitted_corr:.6}"
    );
    assert!(
        dev_rel <= 0.02,
        "Poisson deviance disagrees with statsmodels: gam={dev_gam:.4} sm={dev_sm:.4} (rel={dev_rel:.5})"
    );
}
