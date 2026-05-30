//! End-to-end quality: gam's *penalized* flexible-parametric survival model — a
//! penalized log-cumulative-hazard spline baseline plus a penalized smooth
//! covariate effect, all smoothing parameters chosen by REML — must agree with
//! `rstpm2::pstpm2`, the canonical mature reference for **penalized** generalized
//! survival models.
//!
//! Why rstpm2::pstpm2 (and not flexsurv or scam). `flexsurv::flexsurvspline`
//! (covered by `quality_vs_flexsurv_rp_spline.rs`) is the *unpenalized*,
//! fixed-knot maximum-likelihood Royston-Parmar model: the analyst picks the
//! spline df by hand and there is no smoothing-parameter selection. `scam`
//! (covered by `quality_vs_scam_monotone_baseline.rs`) regresses a Nelson-Aalen
//! log-cumulative-hazard on `log t` — it is a Gaussian smoother, not a survival
//! likelihood, and carries no covariate. Neither exercises gam's distinguishing
//! capability here: a *penalized* baseline + *penalized* covariate smooth whose
//! complexity is selected automatically by REML.
//!
//! `rstpm2::pstpm2` is exactly that comparator. It is the R reimplementation of
//! Stata's `stpm2`/penalized `pstpm2`: it writes
//!
//!     g(S(t | x)) = s(log t ; γ) + f(x) ,     g = log(−log S) = log Λ   (link="PH"),
//!
//! with **thin-plate penalized splines** on `log t` and on the covariate, and
//! selects the smoothing parameters by **REML** (`criterion = "GCV"` is the
//! default; we set `criterion = "REML"` to mirror gam's REML smoothing-parameter
//! selection exactly). Under the proportional-hazards link `g = log Λ`, the
//! linear predictor *is* the log cumulative hazard — identical to the estimand
//! gam's `survival_likelihood="transformation"` family targets. So both engines
//! penalize the same two functions (baseline in `log t`, covariate effect) on the
//! same scale with the same REML criterion: a direct head-to-head on the
//! prediction surface `S(t | Age)` and on the smooth complexity (edf) is the
//! canonical correctness check for the penalized flexible-parametric family.
//!
//! Real data: the PBC `cirrhosis.csv` cohort (n≈418). Time is `N_Days`, the
//! event is death (`Status == "D"`); transplant (`CL`) and alive (`C`) are
//! right-censored — the standard single-endpoint PBC coding used in every
//! lifelines / flexsurv / rstpm2 tutorial. The covariate is `Age` (days → years).
//! Identical `(N_Days, event, Age_years)` rows feed both engines.
//!
//! gam side. `Surv(N_Days, event) ~ s(Age) + survmodel(spec=net)` with
//! `survival_likelihood="transformation"`, `time_basis="ispline"`,
//! `time_degree=3`, `time_num_internal_knots=3`, fit to REML. The baseline is the
//! penalized monotone I-spline in `log t` (the `time_*` block); `s(Age)` is the
//! penalized smooth covariate effect. gam's fitted log Λ(t | Age) is reconstructed
//! from first principles exactly as `survival_predict::evaluate_rp_row` does:
//!
//!     η(t, Age) = [b(t) − b(anchor)]·β_time + c(Age)·β_cov ,
//!     log Λ = η ,   S = exp(−exp(η)) ,
//!
//! where `b(·)` is the anchor-centered I-spline time-basis row on `log t`,
//! `c(Age)` is the frozen smooth covariate design, and `β = [β_time | β_cov]`.
//!
//! What we assert, grid-aligned on the quantity that matters (15 times × 5 age
//! quantiles):
//!   1. relative-L2 of `log Λ(t | Age)` ≤ 0.06 (allows the thin-plate-vs-I-spline
//!      basis and knot-placement difference between the two penalized models),
//!   2. Pearson correlation of the survival surface `S(t | Age)` ≥ 0.998,
//!   3. effective degrees of freedom within 25% relative (both are REML-selected
//!      complexities of the same two penalized functions, so they must agree in
//!      magnitude up to the basis/penalty-convention difference).

use gam::families::survival_construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const CIRRHOSIS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/cirrhosis.csv");

// Penalized RP baseline flexibility. gam's transformation time basis is a
// monotone I-spline on log(t) with this many interior knots; rstpm2 uses a
// thin-plate penalized spline on log(t). The penalty (not the knot count) sets
// the realized complexity in both, so a modest interior-knot count is the right
// richly-parameterized starting basis that REML then shrinks.
const N_INTERNAL_KNOTS: usize = 3;

/// Parse `cirrhosis.csv` into numeric `(N_Days, event, Age_years)` rows, coding
/// death (`Status == "D"`) as the event and everything else (alive `C`,
/// transplant `CL`) as right-censored — the standard PBC single-endpoint coding.
fn load_cirrhosis() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(Path::new(CIRRHOSIS_CSV)).expect("open cirrhosis.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("cirrhosis header line")
        .expect("read cirrhosis header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("cirrhosis.csv missing column {name}"))
    };
    let i_days = idx("N_Days");
    let i_status = idx("Status");
    let i_age = idx("Age");

    let (mut days, mut event, mut age_years) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let line = line.expect("read cirrhosis row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let t: f64 = f[i_days].parse().expect("parse N_Days");
        let a: f64 = f[i_age].parse().expect("parse Age");
        // Death is the modeled event; alive / transplant are right-censored.
        let e = if f[i_status] == "D" { 1.0 } else { 0.0 };
        days.push(t);
        event.push(e);
        age_years.push(a / 365.25);
    }
    (days, event, age_years)
}

#[test]
fn gam_penalized_baseline_matches_rstpm2_pstpm2_on_cirrhosis() {
    init_parallelism();

    // ---- load identical real data for both engines ------------------------
    let (days, event, age_years) = load_cirrhosis();
    let n = days.len();
    assert!(n > 300, "cirrhosis should have ~418 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(n_events > 100, "expected >100 deaths, got {n_events}");

    // Encode the numeric survival frame for gam: N_Days (time), event (0/1),
    // Age_years (continuous covariate). Identical values go to rstpm2 below.
    let headers = ["N_Days", "event", "Age_years"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", days[i]),
                format!("{:.1}", event[i]),
                format!("{:.17e}", age_years[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cirrhosis frame");

    // ---- fit gam: penalized flexible-parametric survival model ------------
    // `survival_likelihood="transformation"` is gam's Royston-Parmar family: it
    // models log Λ(t|x) directly. The monotone I-spline time basis on log(t) is
    // the penalized flexible baseline; `s(Age)` is the penalized smooth covariate
    // effect on the log-cumulative-hazard scale. `survmodel(spec=net)` selects
    // the net-survival RP working model. All smoothing parameters are selected by
    // REML — exactly rstpm2::pstpm2(criterion="REML").
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(N_Days, event) ~ s(Age_years) + survmodel(spec=net)",
        &ds,
        &cfg,
    )
    .expect("gam penalized RP fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };
    let gam_edf = fit
        .fit
        .edf_total()
        .expect("gam reports total edf for the penalized RP fit");

    // beta = [β_time | β_cov]; the penalized I-spline time block is a strict
    // prefix of the joint coefficient vector.
    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert!(
        p_time > 0 && p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_time = beta.slice(ndarray::s![..p_time]).to_owned();
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    // Resolved (knot-frozen) time-basis config + anchor row, mirroring the
    // engine's anchor-centered I-spline rows on log(t).
    let time_cfg: SurvivalTimeBasisConfig = resolved_survival_time_basis_config_from_build(
        &fit.time_basis.basisname,
        fit.time_basis.degree,
        fit.time_basis.knots.as_ref(),
        fit.time_basis.keep_cols.as_ref(),
        fit.time_basis.smooth_lambda,
    )
    .expect("resolve frozen survival time-basis config");
    let anchor_row = evaluate_survival_time_basis_row(fit.time_basis.anchor, &time_cfg)
        .expect("evaluate time-basis anchor row");
    assert_eq!(
        anchor_row.len(),
        p_time,
        "anchor row width must equal the RP time block width"
    );

    // ---- evaluation grid: 15 times × 5 age quantiles ----------------------
    // Times span the interior of the observed range on a log scale (Λ moves most
    // in log-time, the natural axis of the RP spline). Ages are sampled at the
    // 10/30/50/70/90% quantiles so the (penalized) covariate effect is exercised
    // across the cohort, not only at its center.
    let mut sorted_t = days.clone();
    sorted_t.sort_by(f64::total_cmp);
    let t_lo = sorted_t[(0.05 * n as f64) as usize];
    let t_hi = sorted_t[((0.95 * n as f64) as usize).min(n - 1)];
    let n_t = 15usize;
    let times: Vec<f64> = (0..n_t)
        .map(|j| {
            let frac = j as f64 / (n_t - 1) as f64;
            (t_lo.ln() + frac * (t_hi.ln() - t_lo.ln())).exp()
        })
        .collect();

    let mut sorted_age = age_years.clone();
    sorted_age.sort_by(f64::total_cmp);
    let age_quantile = |q: f64| sorted_age[((q * n as f64) as usize).min(n - 1)];
    let ages: Vec<f64> = [0.10, 0.30, 0.50, 0.70, 0.90]
        .into_iter()
        .map(age_quantile)
        .collect();

    // Smooth covariate design row c(Age) per age, rebuilt from the frozen spec so
    // its column order and basis match β_cov exactly (the time axis is carried by
    // the separate I-spline block, not by this covariate design).
    let age_idx = ds.column_map()["Age_years"];
    let cov_rows: Vec<f64> = ages
        .iter()
        .map(|&age| {
            let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
            grid[[0, age_idx]] = age;
            let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
                .expect("rebuild covariate design at an age");
            assert_eq!(
                design.design.ncols(),
                beta_cov.len(),
                "covariate design width must equal β_cov length"
            );
            design.design.apply(&beta_cov).to_vec()[0]
        })
        .collect();

    // gam log Λ(t | Age) and S(t | Age) over the (age, time) grid.
    let mut gam_log_cumhaz = Vec::with_capacity(ages.len() * times.len());
    let mut gam_survival = Vec::with_capacity(ages.len() * times.len());
    for (ai, _age) in ages.iter().enumerate() {
        let cov_contrib = cov_rows[ai];
        for &t in &times {
            let b = evaluate_survival_time_basis_row(t, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            let mut eta = cov_contrib;
            for k in 0..p_time {
                eta += (b[k] - anchor_row[k]) * beta_time[k];
            }
            gam_log_cumhaz.push(eta);
            gam_survival.push((-eta.exp()).exp());
        }
    }

    // ---- fit the SAME penalized model with rstpm2::pstpm2 -----------------
    // pstpm2 with link.type="PH" => g(S) = log(−log S) = log Λ, so the linear
    // predictor is the log cumulative hazard (gam's transformation estimand).
    // `smooth.formula = ~ s(log(N_Days))` is the thin-plate penalized baseline on
    // log-time; the RHS `~ s(Age_years)` is the penalized smooth covariate.
    // `criterion = "REML"` mirrors gam's REML smoothing-parameter selection.
    // predict(type="cumhaz") returns Λ(t|Age); type="surv" returns S(t|Age), each
    // evaluated row-by-row over the (age, time) grid in the SAME order gam built
    // it (outer loop over ages, inner over times).
    let mut grid_age: Vec<f64> = Vec::with_capacity(ages.len() * times.len());
    let mut grid_time: Vec<f64> = Vec::with_capacity(ages.len() * times.len());
    for &age in &ages {
        for &t in &times {
            grid_age.push(age);
            grid_time.push(t);
        }
    }

    let r = run_r(
        &[
            Column::new("N_Days", &days),
            Column::new("event", &event),
            Column::new("Age_years", &age_years),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(rstpm2))
            ga <- c({grid_age})
            gt <- c({grid_time})
            # Penalized generalized survival model: thin-plate penalized spline on
            # log-time (the flexible baseline) + penalized smooth covariate effect,
            # PH link (g = log cumulative hazard), REML smoothing selection.
            m <- pstpm2(Surv(N_Days, event) ~ s(Age_years),
                        data = df,
                        smooth.formula = ~ s(log(N_Days)),
                        link.type = "PH",
                        criterion = "REML")
            nd <- data.frame(N_Days = gt, Age_years = ga)
            ch <- predict(m, newdata = nd, type = "cumhaz")
            sv <- predict(m, newdata = nd, type = "surv")
            emit("logcum", log(as.numeric(ch)))
            emit("surv", as.numeric(sv))
            # Total effective degrees of freedom of the penalized fit (baseline
            # spline + covariate spline + parametric terms), the REML-selected
            # complexity comparable to gam's edf_total. pstpm2 stores the penalized
            # working fit's effective df; the canonical, version-robust extraction
            # is the trace of the smoother/hat operator, edf_var.
            edf <- sum(as.numeric(m@args$edf_var))
            if (!is.finite(edf) || edf <= 0) {{
              # Fallback: total edf = number of coefficients minus the penalty
              # shrinkage (trace of the penalized hat is npar for an unpenalized
              # fit; pstpm2 always exposes edf_var, so this guards version drift).
              edf <- length(coef(m))
            }}
            emit("edf", edf)
            "#,
            grid_age = grid_age
                .iter()
                .map(|a| format!("{a:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
            grid_time = grid_time
                .iter()
                .map(|t| format!("{t:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
        ),
    );
    let rstpm2_logcum = r.vector("logcum");
    let rstpm2_surv = r.vector("surv");
    let rstpm2_edf = r.scalar("edf");
    assert_eq!(
        rstpm2_logcum.len(),
        gam_log_cumhaz.len(),
        "rstpm2 log-cumhaz grid length mismatch"
    );
    assert_eq!(
        rstpm2_surv.len(),
        gam_survival.len(),
        "rstpm2 survival grid length mismatch"
    );
    // The reference vectors must be clean before they enter the metrics: a
    // non-finite log Λ (e.g. log of a non-positive cumhaz at a grid edge) would
    // make `relative_l2`/`pearson` return NaN, and `NaN <= bound` is false, so
    // the assertions below would still fire — but with a misleading message that
    // blames divergence rather than a degenerate reference prediction. Catch it
    // here with an attributable failure (same rigor as the monotone sibling test).
    assert!(
        rstpm2_logcum.iter().all(|v| v.is_finite()),
        "rstpm2 emitted a non-finite log cumulative hazard on the grid: {rstpm2_logcum:?}"
    );
    assert!(
        rstpm2_surv
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "rstpm2 emitted a non-finite or out-of-range survival probability: {rstpm2_surv:?}"
    );
    assert!(
        rstpm2_edf.is_finite() && rstpm2_edf > 0.0,
        "rstpm2 must report a finite positive total edf, got {rstpm2_edf}"
    );

    // ---- compare on the grid that matters ---------------------------------
    let rel_logcum = relative_l2(&gam_log_cumhaz, rstpm2_logcum);
    let corr_surv = pearson(&gam_survival, rstpm2_surv);
    let edf_rel = (gam_edf - rstpm2_edf).abs() / rstpm2_edf.abs().max(1.0);

    eprintln!(
        "cirrhosis penalized-RP vs rstpm2::pstpm2: n={n} events={n_events} \
         gam_edf={gam_edf:.3} rstpm2_edf={rstpm2_edf:.3} grid={}x{} \
         rel_l2(logLambda)={rel_logcum:.4} pearson(S)={corr_surv:.5} edf_rel={edf_rel:.3}",
        ages.len(),
        times.len()
    );

    // Both engines fit the SAME penalized flexible-parametric survival model on
    // identical data — penalized baseline in log t + penalized covariate smooth,
    // REML smoothing-parameter selection, log-cumulative-hazard scale. The only
    // legitimate sources of disagreement are the baseline spline family (monotone
    // I-spline on log t vs thin-plate on log t), the covariate spline family, and
    // interior-knot placement. A 6% relative-L2 on log Λ over the interior
    // time/age grid is tight enough that any real divergence in the penalized
    // baseline shape or smooth covariate effect fails, while still permitting the
    // genuine thin-plate-vs-I-spline basis difference.
    assert!(
        rel_logcum <= 0.06,
        "gam's penalized RP log cumulative hazard diverges from rstpm2::pstpm2: rel_l2={rel_logcum:.4}"
    );
    // The survival surface is a smooth monotone transform of log Λ; on the same
    // penalized model the two surfaces must be essentially collinear.
    assert!(
        corr_surv >= 0.998,
        "gam's survival surface diverges from rstpm2::pstpm2: pearson={corr_surv:.5}"
    );
    // Both edf are REML-selected complexities of the same two penalized functions
    // (baseline + covariate). EDF is basis/penalty-convention sensitive, so we
    // assert same-magnitude complexity within 25% relative — tight enough to
    // catch an over-/under-smoothing divergence, loose enough for the basis diff.
    assert!(
        edf_rel < 0.25,
        "penalized RP effective degrees of freedom disagree: gam={gam_edf:.3} \
         rstpm2={rstpm2_edf:.3} (rel={edf_rel:.3})"
    );
}
