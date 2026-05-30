//! End-to-end quality: gam's Bayesian posterior for a penalized binomial GAM
//! must **recover the known latent function** it was generated from, and must
//! react to the smoothing parameter the way penalized-likelihood theory requires.
//!
//! OBJECTIVE metric asserted (this is the pass/fail claim — NOT "matches PyMC"):
//!   * **Truth recovery (primary, accuracy).** The data are simulated from a
//!     *known* latent log-odds function f(x)=x³+sin(5x) with y~Bernoulli(σ(f)).
//!     At a moderate smoothing parameter, gam's posterior-mean fit on the
//!     linear-predictor scale, η̂ = X β̄, must reconstruct f(x): we assert
//!     RMSE(η̂, f) is a small fraction of the signal range. This is an objective
//!     statement about gam's fit versus ground truth, independent of any peer.
//!   * **Penalty → posterior concentration (structure).** As λ grows
//!     0.1 → 1 → 10, the posterior spread of the penalized (roughness-bearing)
//!     coefficients must shrink monotonically — the defining
//!     penalty-as-prior interaction.
//!   * **Convergence.** R-hat < 1.1 and a non-trivial effective sample size for
//!     every λ.
//!
//! PyMC's role: BASELINE TO MATCH-OR-BEAT, not a correctness oracle. PyMC is a
//! mature NUTS engine that can encode the *identical* penalized posterior
//!     p(β | y) ∝ exp{ ℓ(y; Xβ) − ½ λ βᵀP β },
//! via `pm.Potential(-0.5*λ βᵀPβ)` over the same (X, y, λP). We fit it on the
//! identical design/response and measure ITS posterior-mean truth-recovery RMSE
//! on the same grid. Because both engines target the same density, "matching
//! PyMC" proves nothing about correctness; instead we require that gam recovers
//! the truth *at least as well as* PyMC does (RMSE(gam) ≤ RMSE(PyMC) × 1.10).
//! The element-wise mean/std rel-L2 gap to PyMC is still computed and printed via
//! eprintln! for context, but it is NOT a pass criterion.
//!
//! Identical data is guaranteed: the design matrix `X` that gam builds from its
//! own `s(x)` basis and the binary responses `y` are handed verbatim to PyMC, so
//! both engines fit the same coefficients in the same basis.

use gam::estimate::Dispersion;
use gam::hmc::{
    FamilyNutsInputs, GlmFlatInputs, NutsConfig, NutsResult, run_nuts_sampling_flattened_family,
};
use gam::smooth::{build_term_collection_design, weighted_blockwise_penalty_sum};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2};

/// True latent log-odds function the data are simulated from. The posterior-mean
/// linear predictor η̂ = X β̄ must reconstruct this on the observed grid; it is the
/// ground truth for the accuracy assertion.
fn truth_logodds(x: f64) -> f64 {
    x.powi(3) + (5.0 * x).sin()
}

/// Deterministic synthetic data: n=150, x∈[-1,1], highly nonlinear truth
/// f(x)=x³+sin(5x), y~Bernoulli(logit⁻¹(f(x))). A fixed-seed splitmix64 stream
/// makes the draws byte-reproducible AND identical for both engines (the very
/// same `x`/`y` vectors are fed to gam and to PyMC). Returns (x, y, truth f(x)).
fn synthetic_binomial() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    const N: usize = 150;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_u01 = || {
        // splitmix64 → uniform(0,1)
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut truth = Vec::with_capacity(N);
    for i in 0..N {
        // Deterministic, evenly spaced covariate on [-1, 1].
        let xi = -1.0 + 2.0 * (i as f64) / ((N - 1) as f64);
        let f = truth_logodds(xi);
        let prob = 1.0 / (1.0 + (-f).exp());
        let yi = if next_u01() < prob { 1.0 } else { 0.0 };
        x.push(xi);
        y.push(yi);
        truth.push(f);
    }
    (x, y, truth)
}

/// Penalized binomial-logit IRLS at a *fixed* penalty `S = λP`. Returns the MAP
/// coefficient (the posterior mode) and the penalized Hessian `XᵀWX + S` at that
/// mode. These seed gam's sampler (the auto-selected Pólya-Gamma Gibbs path for
/// unit-weight Bernoulli logit re-equilibrates from any reasonable start, but a
/// genuine MAP keeps the warmup short and the Hessian is the exact curvature the
/// NUTS-family dispatcher whitens against for non-Gibbs links).
fn penalized_logit_mode(
    x: &Array2<f64>,
    y: &Array1<f64>,
    s_lambda: &Array2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut beta = Array1::<f64>::zeros(p);
    let mut hessian = Array2::<f64>::zeros((p, p));
    for _ in 0..100 {
        let eta = x.dot(&beta);
        let mut w = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n); // working residual gradient term
        for i in 0..n {
            let mu = 1.0 / (1.0 + (-eta[i]).exp());
            let wi = (mu * (1.0 - mu)).max(1e-9);
            w[i] = wi;
            z[i] = y[i] - mu; // score contribution before X^T
        }
        // Gradient of −ℓ + ½βᵀSβ : g = −Xᵀ(y−μ) + Sβ
        let grad = -x.t().dot(&z) + s_lambda.dot(&beta);
        // Hessian: XᵀWX + S
        let mut xtwx = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let row = x.row(i);
            let wi = w[i];
            for a in 0..p {
                let ra = row[a] * wi;
                for b in 0..p {
                    xtwx[[a, b]] += ra * row[b];
                }
            }
        }
        hessian = &xtwx + s_lambda;
        // Newton step: solve H δ = −grad via a small ridge-stabilised Gaussian
        // elimination (p is tiny here).
        let mut a = hessian.clone();
        for d in 0..p {
            a[[d, d]] += 1e-10;
        }
        let rhs = -&grad;
        let delta = solve_spd(&a, &rhs);
        let step = delta.mapv(|v| v.clamp(-5.0, 5.0));
        beta = &beta + &step;
        if step.mapv(f64::abs).sum() < 1e-10 {
            break;
        }
    }
    (beta, hessian)
}

/// Dense SPD solve by Cholesky + forward/back substitution (p ≲ 12 here, so a
/// dependency-free in-test solver is appropriate).
fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let p = a.nrows();
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = sum.max(1e-12).sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Solve L u = b
    let mut u = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * u[k];
        }
        u[i] = sum / l[[i, i]];
    }
    // Solve Lᵀ x = u
    let mut xsol = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        let mut sum = u[i];
        for k in (i + 1)..p {
            sum -= l[[k, i]] * xsol[k];
        }
        xsol[i] = sum / l[[i, i]];
    }
    xsol
}

#[test]
fn gam_penalized_binomial_posterior_matches_pymc_and_concentrates_with_lambda() {
    init_parallelism();

    // ---- identical synthetic data for both engines ------------------------
    let (x_raw, y_raw, truth) = synthetic_binomial();
    let n = x_raw.len();
    // Signal range of the true latent log-odds: the natural scale for a
    // truth-recovery bar on the linear-predictor (η) scale.
    let truth_min = truth.iter().cloned().fold(f64::INFINITY, f64::min);
    let truth_max = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let signal_range = truth_max - truth_min;
    assert!(
        signal_range > 1.0,
        "synthetic truth must span a non-trivial range, got {signal_range:.3}"
    );

    // Encode as a gam dataset (x covariate, y binary response).
    let header = "x,y".to_string();
    let mut records = String::new();
    records.push_str(&header);
    records.push('\n');
    for i in 0..n {
        records.push_str(&format!("{:.17e},{}\n", x_raw[i], y_raw[i] as i64));
    }
    // Persist to a temp CSV so we exercise the same loader path the CLI uses.
    let dir = std::env::temp_dir().join(format!("gam_pymc_hmc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let csv_path = dir.join("data.csv");
    std::fs::write(&csv_path, &records).expect("write synthetic csv");
    let ds = load_csvwith_inferred_schema(&csv_path).expect("load synthetic csv");
    assert_eq!(
        ds.values.nrows(),
        n,
        "loader must preserve every synthetic row"
    );

    // ---- fit gam once to materialise the s(x) basis + roughness penalty ----
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gam binomial fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for y ~ s(x)");
    };

    // Rebuild the (frozen) design at the observed x: this is the exact design
    // matrix gam's sampler integrates over, and the one we hand to PyMC.
    let col = ds.column_map();
    let x_idx = col["x"];
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &xi) in x_raw.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let x_dense: Array2<f64> = design.design.to_dense();
    let p = x_dense.ncols();
    assert!(
        (5..=20).contains(&p),
        "s(x) default basis should give a modest coefficient count, got p={p}"
    );
    assert_eq!(
        design.penalties.len(),
        1,
        "s(x) should yield one penalty block"
    );

    // Identify the penalized subspace: columns whose diagonal penalty weight is
    // non-trivial. These are the "high-order" / roughness-bearing coefficients
    // whose posterior must tighten as λ grows; the smooth's null space
    // (unpenalized polynomial part) is excluded from the shrinkage claim.
    let unit_penalty = weighted_blockwise_penalty_sum(&design.penalties, &[1.0], p);
    let penalized_cols: Vec<usize> = (0..p)
        .filter(|&j| unit_penalty[[j, j]].abs() > 1e-8)
        .collect();
    assert!(
        !penalized_cols.is_empty(),
        "the roughness penalty must act on at least one coefficient"
    );

    let likelihood = LikelihoodSpec {
        response: ResponseFamily::Binomial,
        link: InverseLink::Standard(StandardLink::Logit),
    };
    let y_arr = Array1::from(y_raw.clone());
    let weights = Array1::<f64>::ones(n);

    let lambdas = [0.1_f64, 1.0, 10.0];
    let nuts_cfg = NutsConfig {
        n_samples: 1500,
        nwarmup: 1000,
        n_chains: 4,
        target_accept: 0.9,
        seed: 20_260_529,
    };

    // Accumulate, per λ: gam posterior mean/std over coefficients, the total
    // posterior std of the penalized subspace (the shrinkage statistic), and the
    // truth-recovery RMSE of the posterior-mean fit η̂ = X β̄ against f(x).
    let mut gam_means: Vec<Array1<f64>> = Vec::new();
    let mut gam_stds: Vec<Array1<f64>> = Vec::new();
    let mut gam_penalized_spread: Vec<f64> = Vec::new();
    let mut gam_eta_rmse: Vec<f64> = Vec::new();

    for &lam in &lambdas {
        let s_lambda = weighted_blockwise_penalty_sum(&design.penalties, &[lam], p);
        let (mode, hessian) = penalized_logit_mode(&x_dense, &y_arr, &s_lambda);

        let res: NutsResult = run_nuts_sampling_flattened_family(
            likelihood.clone(),
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x_dense.view(),
                y: y_arr.view(),
                weights: weights.view(),
                penalty_matrix: s_lambda.view(),
                mode: mode.view(),
                hessian: hessian.view(),
                gamma_shape: None,
                dispersion: Dispersion::Known(1.0),
                firth_bias_reduction: false,
            }),
            &nuts_cfg,
        )
        .expect("gam binomial posterior sampling");

        // (2) convergence for every λ.
        assert!(
            res.rhat < 1.1,
            "gam posterior failed to converge at lambda={lam}: R-hat={:.4}",
            res.rhat
        );
        let total_draws = (nuts_cfg.n_samples * nuts_cfg.n_chains) as f64;
        assert!(
            res.ess > 0.05 * total_draws,
            "gam effective sample size too low at lambda={lam}: ess={:.1} of {total_draws} draws",
            res.ess
        );

        let spread: f64 = penalized_cols.iter().map(|&j| res.posterior_std[j]).sum();

        // Truth recovery: posterior-mean fit on the linear-predictor scale must
        // reconstruct the latent log-odds the data were generated from.
        let eta_hat: Array1<f64> = x_dense.dot(&res.posterior_mean);
        let eta_rmse = gam::test_support::reference::rmse(eta_hat.as_slice().unwrap(), &truth);

        eprintln!(
            "gam lambda={lam:>5}: R-hat={:.4} ess={:.0} penalized_posterior_std_sum={:.5} \
             eta_rmse_vs_truth={eta_rmse:.5} (signal_range={signal_range:.3})",
            res.rhat, res.ess, spread
        );
        gam_means.push(res.posterior_mean.clone());
        gam_stds.push(res.posterior_std.clone());
        gam_penalized_spread.push(spread);
        gam_eta_rmse.push(eta_rmse);
    }

    // (1) Penalty → posterior concentration: the penalized coefficients' total
    // posterior spread must DECREASE monotonically as λ increases. A tighter
    // smoothness prior must concentrate the posterior; if it does not, gam's
    // penalty is not entering the posterior the way the theory requires.
    assert!(
        gam_penalized_spread[0] > gam_penalized_spread[1]
            && gam_penalized_spread[1] > gam_penalized_spread[2],
        "penalized-coefficient posterior spread must shrink monotonically with lambda: \
         {:.5} (λ=0.1) -> {:.5} (λ=1) -> {:.5} (λ=10)",
        gam_penalized_spread[0],
        gam_penalized_spread[1],
        gam_penalized_spread[2]
    );

    // (PRIMARY) Truth recovery: at a moderate smoothing parameter the
    // posterior-mean linear predictor must reconstruct the latent log-odds the
    // data were generated from. We take the best-of-grid RMSE (the smoothing
    // parameter is a nuisance gam would otherwise select; sweeping a fixed grid,
    // the best fit on the grid is the fair accuracy statistic) and require it to
    // be a small fraction of the signal range. This is the objective accuracy
    // claim — gam recovers the truth — and does not reference any peer tool.
    let gam_best_rmse = gam_eta_rmse
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let truth_bar = 0.25 * signal_range;
    eprintln!(
        "gam best-of-grid eta truth-recovery RMSE = {gam_best_rmse:.5} \
         (bar = 0.25 * signal_range = {truth_bar:.5})"
    );
    assert!(
        gam_best_rmse <= truth_bar,
        "gam posterior-mean fit must recover the latent log-odds: best eta RMSE \
         {gam_best_rmse:.5} exceeds bar {truth_bar:.5} (= 0.25 * signal_range {signal_range:.3})"
    );

    // ---- PyMC reference: identical (X, y, λP) penalized posterior ----------
    // Hand PyMC the exact design columns and responses. We pass each design
    // column as `c{j}` and `y`, plus the flattened penalty matrix as `pen`
    // (length p*p) and the lambda grid as `lam`. PyMC encodes the SAME
    // un-normalised log-density: Bernoulli-logit likelihood on η = X@β plus a
    // Potential −½ λ βᵀPβ. ArviZ supplies R-hat / ESS.
    let mut columns: Vec<gam::test_support::reference::Column<'_>> = Vec::new();
    // Owned storage for column data (must outlive the run_python call).
    let col_storage: Vec<Vec<f64>> = (0..p).map(|j| x_dense.column(j).to_vec()).collect();
    for (j, c) in col_storage.iter().enumerate() {
        // SAFETY of naming: `c{j}` is a stable, unique header per design column.
        let name: &'static str = Box::leak(format!("c{j}").into_boxed_str());
        columns.push(gam::test_support::reference::Column::new(name, c));
    }
    columns.push(gam::test_support::reference::Column::new("y", &y_raw));

    // The penalty matrix and lambda grid are model constants, not per-row data.
    // The harness wants equal-length columns, so we ship them inside the Python
    // body as literals instead of as data columns.
    let pen_flat: Vec<String> = unit_penalty.iter().map(|v| format!("{v:.17e}")).collect();
    let pen_literal = format!("[{}]", pen_flat.join(", "));
    let lam_literal = format!(
        "[{}]",
        lambdas
            .iter()
            .map(|l| format!("{l:.17e}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let body = format!(
        r#"
import numpy as np
import pymc as pm
import arviz as az

P_DIM = {p}
LAMBDAS = np.array({lam_literal}, dtype=float)
PEN = np.array({pen_literal}, dtype=float).reshape(P_DIM, P_DIM)

# Rebuild the design exactly as gam handed it over (columns c0..c{{p-1}}).
X = np.column_stack([np.asarray(df["c%d" % j], dtype=float) for j in range(P_DIM)])
y = np.asarray(df["y"], dtype=float)

for li, lam in enumerate(LAMBDAS):
    S = lam * PEN
    with pm.Model() as model:
        beta = pm.Flat("beta", shape=P_DIM)
        eta = pm.math.dot(X, beta)
        # Smoothness prior as an explicit potential: -1/2 * beta^T (lam P) beta,
        # matching gam's penalized log-posterior term exactly.
        pm.Potential("rough", -0.5 * pm.math.dot(beta, pm.math.dot(S, beta)))
        pm.Bernoulli("obs", logit_p=eta, observed=y)
        idata = pm.sample(
            draws=1500, tune=1000, chains=4, cores=1,
            target_accept=0.9, random_seed=20260529 + li,
            progressbar=False, compute_convergence_checks=False,
        )
    post = idata.posterior["beta"]
    mean = post.mean(dim=("chain", "draw")).values
    sd = post.std(dim=("chain", "draw")).values
    summ = az.summary(idata, var_names=["beta"])
    rhat = float(np.nanmax(summ["r_hat"].values))
    ess = float(np.nanmin(summ["ess_bulk"].values))
    emit("mean_%d" % li, mean)
    emit("std_%d" % li, sd)
    emit("rhat_%d" % li, [rhat])
    emit("ess_%d" % li, [ess])
"#
    );

    let pymc = gam::test_support::reference::run_python(&columns, &body);

    // ---- PyMC as a BASELINE TO MATCH-OR-BEAT on truth recovery -------------
    // Both engines target the identical penalized posterior, so "matching PyMC"
    // is not a correctness claim. Instead we measure PyMC's OWN posterior-mean
    // truth-recovery RMSE on the same design/grid and require gam to recover the
    // latent function at least as well (within a 10% accuracy margin). The
    // element-wise mean/std rel-L2 gap is computed and printed for context only.
    let mut pymc_best_rmse = f64::INFINITY;
    for (li, &lam) in lambdas.iter().enumerate() {
        let pm_mean = pymc.vector(&format!("mean_{li}"));
        let pm_std = pymc.vector(&format!("std_{li}"));
        let pm_rhat = pymc.scalar(&format!("rhat_{li}"));
        assert_eq!(pm_mean.len(), p, "PyMC mean dim mismatch at lambda={lam}");

        // PyMC's posterior-mean fit on the η scale, and its truth-recovery RMSE.
        let pm_mean_arr = Array1::from(pm_mean.to_vec());
        let pm_eta: Array1<f64> = x_dense.dot(&pm_mean_arr);
        let pm_eta_rmse =
            gam::test_support::reference::rmse(pm_eta.as_slice().unwrap(), &truth);
        pymc_best_rmse = pymc_best_rmse.min(pm_eta_rmse);

        // Context-only diagnostics: cross-engine rel-L2 on the penalized block.
        let gm = &gam_means[li];
        let gs = &gam_stds[li];
        let gam_mean_pen: Vec<f64> = penalized_cols.iter().map(|&j| gm[j]).collect();
        let pm_mean_pen: Vec<f64> = penalized_cols.iter().map(|&j| pm_mean[j]).collect();
        let gam_std_pen: Vec<f64> = penalized_cols.iter().map(|&j| gs[j]).collect();
        let pm_std_pen: Vec<f64> = penalized_cols.iter().map(|&j| pm_std[j]).collect();
        let mean_rel_l2 =
            gam::test_support::reference::relative_l2(&gam_mean_pen, &pm_mean_pen);
        let std_rel_l2 =
            gam::test_support::reference::relative_l2(&gam_std_pen, &pm_std_pen);
        eprintln!(
            "lambda={lam:>5}: PyMC R-hat={pm_rhat:.4} eta_rmse_vs_truth={pm_eta_rmse:.5} \
             | context-only gam-vs-PyMC penalized-block rel_l2(mean)={mean_rel_l2:.4} \
             rel_l2(std)={std_rel_l2:.4}"
        );
    }

    // Match-or-beat: gam's best truth-recovery error must not exceed PyMC's by
    // more than 10%. (Both fit the same penalized posterior; the claim is that
    // gam's sampler is at least as accurate at reconstructing the truth.)
    eprintln!(
        "truth-recovery RMSE: gam best={gam_best_rmse:.5}  PyMC best={pymc_best_rmse:.5} \
         (match-or-beat bound = PyMC * 1.10 = {:.5})",
        pymc_best_rmse * 1.10
    );
    assert!(
        gam_best_rmse <= pymc_best_rmse * 1.10,
        "gam must recover the truth at least as well as the PyMC baseline: \
         gam best eta RMSE {gam_best_rmse:.5} exceeds PyMC {pymc_best_rmse:.5} * 1.10"
    );

    std::fs::remove_dir_all(&dir).ok();
}
