//! End-to-end quality: gam's ALO sandwich standard error on the linear
//! predictor must be **well CALIBRATED** — the 95% confidence interval it builds
//! around each in-sample linear predictor must cover the *known true* linear
//! predictor `eta_true_i = x_i' beta_true` at (close to) the nominal 95% rate.
//!
//! OBJECTIVE METRIC ASSERTED: empirical coverage of `eta_true` by the
//! gam-derived 95% CI `eta_hat_i +/- t_{0.975, n-p} * SE(eta_i)`, aggregated over
//! many independent simulation replicates drawn from a fixed known parameter
//! vector. A standard-error estimator is *good* exactly when its intervals cover
//! the truth at their stated rate; this is the property a practitioner actually
//! relies on. The pass criterion is
//!
//!   | empirical_coverage(gam) - 0.95 | <= 0.02
//!
//! i.e. gam's intervals are neither anti-conservative (too narrow) nor wasteful
//! (too wide) against ground truth. This is a TRUTH-RELATIVE calibration claim,
//! not a "reproduce a peer tool" claim: the SEs are judged against the true data
//! generating process, not against another fitted output.
//!
//! BASELINE TO MATCH-OR-BEAT: Python `statsmodels` (OLS / Gaussian GLM), the
//! standard regression stack, fits the identical data and builds its own
//! textbook OLS prediction interval `SE(eta_i) = sqrt(sigma^2 x_i'(X'X)^{-1} x_i)`,
//! `sigma^2 = RSS/(n-p)`. We compute statsmodels' own empirical coverage on the
//! same replicates and require gam to be at least as well calibrated:
//!
//!   | coverage(gam) - 0.95 | <= | coverage(statsmodels) - 0.95 | + 0.01
//!
//! For an unpenalized Gaussian linear model both engines are estimating the same
//! closed-form prediction variance, so gam should track the OLS optimum closely;
//! the calibration bar is the real quality claim and the baseline is only a sanity
//! floor. We still print the per-engine coverage and the worst-case SE magnitude
//! with `eprintln!` for context.

use csv::StringRecord;
use gam::inference::alo::compute_alo_diagnostics_from_fit;
use gam::test_support::reference::{Column, run_python};
use gam::types::LinkFunction;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use statrs::distribution::{ContinuousCDF, StudentsT};

#[test]
fn gam_alo_sandwich_ci_covers_true_linear_predictor_at_nominal_rate() {
    init_parallelism();

    // ---- simulation design -------------------------------------------------
    // Known truth: eta_true_i = intercept + sum_j beta_true[j] * x_ij. Gaussian
    // noise with sigma. Many independent replicates so the empirical coverage of
    // the 95% CI is estimated with a tight Monte-Carlo error.
    const N: usize = 80;
    const P: usize = 5;
    // Wall-clock is dominated by REPS sequential gam fits (the statsmodels
    // baseline is a single subprocess that loops the same replicates). 200 reps
    // overran the 360 s reference-quality budget; 60 reps still pools 60×80 = 4800
    // coverage indicators, whose Monte-Carlo SE on a coverage proportion is
    // ≈ sqrt(0.95·0.05/4800) ≈ 0.0031 — comfortably inside the ±0.02 calibration
    // band asserted below, so the coverage claim keeps its statistical power.
    const REPS: usize = 60;
    const SIGMA: f64 = 0.7;
    let beta_true = [1.3_f64, -0.8, 0.5, 2.1, -1.4];
    let intercept_true = 0.4_f64;

    // Exact t critical value for a 95% two-sided interval with n-p residual df.
    // (n - p) counts the intercept + P slopes => residual df = N - (P + 1).
    let resid_df = (N - (P + 1)) as f64;
    let tdist = StudentsT::new(0.0, 1.0, resid_df).expect("students-t");
    let tcrit = tdist.inverse_cdf(0.975);

    let headers: Vec<String> = (0..P)
        .map(|j| format!("x{}", j + 1))
        .chain(std::iter::once("y".to_string()))
        .collect();

    // gam coverage accumulators (true eta inside the gam CI?).
    let mut gam_covered: u64 = 0;
    let mut gam_total: u64 = 0;
    // Worst-case gam prediction SE expressed in units of the noise sigma, used
    // only as a finiteness / non-explosion sanity check (printed for context).
    let mut worst_se_over_sigma: f64 = 0.0;

    // Columns handed to the single statsmodels run: every replicate's data is
    // stacked side-by-side under replicate-indexed names so the reference sees
    // byte-identical numbers and we pay one subprocess spawn instead of REPS.
    let mut ref_storage: Vec<(String, Vec<f64>)> = Vec::new();

    for rep in 0..REPS {
        // Distinct, deterministic seed per replicate (fixed across runs).
        let seed = 0x5EED_0000_0000_0000_u64 ^ (rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = StdRng::seed_from_u64(seed);
        let xdist = Normal::new(0.0_f64, 1.0).expect("normal x");
        let edist = Normal::new(0.0_f64, SIGMA).expect("normal eps");

        let mut xcols: Vec<Vec<f64>> = (0..P).map(|_| Vec::with_capacity(N)).collect();
        let mut yvec: Vec<f64> = Vec::with_capacity(N);
        let mut eta_true: Vec<f64> = Vec::with_capacity(N);
        for _ in 0..N {
            let mut eta = intercept_true;
            let mut xrow = [0.0_f64; P];
            for j in 0..P {
                let xij = xdist.sample(&mut rng);
                xrow[j] = xij;
                eta += beta_true[j] * xij;
            }
            for j in 0..P {
                xcols[j].push(xrow[j]);
            }
            eta_true.push(eta);
            yvec.push(eta + edist.sample(&mut rng));
        }

        // ---- build gam dataset (identical numbers) -------------------------
        let mut records: Vec<StringRecord> = Vec::with_capacity(N);
        for i in 0..N {
            let mut fields: Vec<String> = (0..P).map(|j| format!("{:.17e}", xcols[j][i])).collect();
            fields.push(format!("{:.17e}", yvec[i]));
            records.push(StringRecord::from(fields));
        }
        let data =
            encode_recordswith_inferred_schema(headers.clone(), records).expect("encode dataset");

        // ---- fit gam: y ~ x1 + ... + x5, Gaussian-identity, NO smooth ------
        let formula = "y ~ x1 + x2 + x3 + x4 + x5";
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula(formula, &data, &cfg).expect("gam fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for an unpenalized linear model");
        };

        let yview = data.values.column(data.column_map()["y"]);
        let alo = compute_alo_diagnostics_from_fit(&fit.fit, yview, LinkFunction::Identity)
            .expect("gam ALO diagnostics");
        let eta_hat = alo.pred_identity.to_vec();
        let se = alo.se_sandwich.to_vec();
        assert_eq!(eta_hat.len(), N, "ALO eta length mismatch");
        assert_eq!(se.len(), N, "ALO se length mismatch");

        // gam coverage of the KNOWN true linear predictor.
        for i in 0..N {
            let half = tcrit * se[i];
            if (eta_true[i] - eta_hat[i]).abs() <= half {
                gam_covered += 1;
            }
            gam_total += 1;

            // Textbook OLS prediction SE for context (computed in the Python run);
            // here we only track gam's own SE magnitude relative to the residual
            // scale to flag pathological widths.
            let rel = (se[i] / SIGMA).abs();
            worst_se_over_sigma = worst_se_over_sigma.max(rel);
        }

        // stash this replicate's data for the single statsmodels baseline run.
        for j in 0..P {
            ref_storage.push((format!("x{}_r{rep}", j + 1), xcols[j].clone()));
        }
        ref_storage.push((format!("y_r{rep}"), yvec.clone()));
        ref_storage.push((format!("etrue_r{rep}"), eta_true));
    }

    let gam_coverage = gam_covered as f64 / gam_total as f64;

    // ---- baseline: statsmodels OLS coverage on the identical replicates ----
    let columns: Vec<Column> = ref_storage
        .iter()
        .map(|(name, data)| Column::new(name.as_str(), data))
        .collect();
    let r = run_python(
        &columns,
        &format!(
            r#"
import numpy as np
import statsmodels.api as sm
from scipy import stats
REPS = {REPS}
P = {P}
covered = 0
total = 0
for rep in range(REPS):
    Xcols = [np.asarray(df["x%d_r%d" % (j + 1, rep)], dtype=float) for j in range(P)]
    X = np.column_stack(Xcols)
    y = np.asarray(df["y_r%d" % rep], dtype=float)
    etrue = np.asarray(df["etrue_r%d" % rep], dtype=float)
    Xd = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xd).fit()
    cov = np.asarray(model.cov_params(), dtype=float)      # sigma^2 (X'X)^-1
    eta_hat = Xd @ np.asarray(model.params, dtype=float)
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xd, cov, Xd))    # textbook prediction SE
    dfres = int(model.df_resid)
    tcrit = stats.t.ppf(0.975, dfres)
    half = tcrit * se
    covered += int(np.sum(np.abs(etrue - eta_hat) <= half))
    total += int(etrue.size)
emit("ref_coverage", [covered / total])
"#
        ),
    );
    let ref_coverage = r.scalar("ref_coverage");

    // ---- report + assertions ----------------------------------------------
    let gam_dev = (gam_coverage - 0.95).abs();
    let ref_dev = (ref_coverage - 0.95).abs();
    eprintln!(
        "ALO CI calibration vs truth: n={N} p={P} reps={REPS} tcrit={tcrit:.4} \
         gam_coverage={gam_coverage:.4} (|dev|={gam_dev:.4}) \
         statsmodels_coverage={ref_coverage:.4} (|dev|={ref_dev:.4}) \
         worst_se/sigma={worst_se_over_sigma:.3}"
    );

    // PRIMARY claim: gam's 95% interval covers the TRUE linear predictor at the
    // nominal rate. Monte-Carlo SE of the coverage estimate over 60*80 = 4800
    // replicate*row indicators is ~0.0031, so a +/-0.02 band is well outside the
    // noise yet still a genuine calibration test.
    assert!(
        gam_dev <= 0.02,
        "gam 95% CI miscalibrated vs known truth: coverage={gam_coverage:.4} \
         (nominal 0.95, |dev|={gam_dev:.4} > 0.02)"
    );

    // MATCH-OR-BEAT: gam must be at least as well calibrated as the standard OLS
    // stack on the same data (small slack absorbs Monte-Carlo noise).
    assert!(
        gam_dev <= ref_dev + 0.01,
        "gam calibration worse than statsmodels baseline: \
         gam |dev|={gam_dev:.4} vs statsmodels |dev|={ref_dev:.4} (+0.01 slack)"
    );

    // Sanity: SEs must be finite and on the residual scale, not exploded.
    assert!(
        worst_se_over_sigma.is_finite() && worst_se_over_sigma < 5.0,
        "gam prediction SE pathological relative to noise sigma: \
         worst se/sigma={worst_se_over_sigma:.3}"
    );
}
