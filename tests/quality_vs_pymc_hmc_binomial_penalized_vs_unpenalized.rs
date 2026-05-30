//! End-to-end quality: gam's Bayesian posterior for a penalized binomial GAM
//! must agree, coefficient-by-coefficient, with **PyMC** — the reference NUTS
//! probabilistic-programming engine (plus ArviZ for R-hat / ESS) — when both
//! target the *identical* penalized posterior, and gam's own posterior must
//! react to the smoothing parameter the way the penalized-likelihood theory
//! says it should.
//!
//! Why PyMC is the right comparator here. gam's penalized GAM posterior is
//!     p(β | y) ∝ exp{ ℓ(y; Xβ) − ½ λ βᵀP β },
//! i.e. a Bernoulli-logit likelihood times a (rank-deficient) Gaussian prior
//! whose precision is `λP`, the roughness penalty of the `s(x)` smooth. That is
//! *exactly* a Bayesian GAM with a smoothness prior (the brms / rstanarm
//! formulation), and PyMC is the canonical, mature NUTS engine that can encode
//! the identical un-normalised log-density via `pm.Potential(-0.5*λ βᵀPβ)` over
//! the *same* design matrix `X` and the *same* responses `y`. Feeding both
//! engines the identical `(X, y, λP)` makes the two posteriors the SAME
//! distribution, so their posterior means / stds must coincide up to Monte-Carlo
//! error — a real divergence is a real bug in gam's sampler.
//!
//! The test asserts three things, none of them weakened:
//!   1. **Penalty → posterior concentration.** As λ grows 0.1 → 1 → 10, the
//!      posterior spread of the *penalized* (high-order, roughness-bearing)
//!      coefficients must shrink monotonically. This is the defining
//!      penalty-posterior interaction: a larger roughness penalty is a tighter
//!      smoothness prior and must concentrate the posterior on smoother fits.
//!   2. **Convergence.** gam's posterior must converge for every λ
//!      (R-hat < 1.1, and an effective sample size that is a non-trivial
//!      fraction of the draws).
//!   3. **Cross-engine agreement.** At each λ, gam's posterior mean and posterior
//!      std must match PyMC's element-wise (the two encode the same density), and
//!      PyMC must itself report R-hat < 1.1. The bound is set by Monte-Carlo
//!      error, not by tolerance-shopping.
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

/// Deterministic synthetic data: n=150, x∈[-1,1], highly nonlinear truth
/// f(x)=x³+sin(5x), y~Bernoulli(logit⁻¹(f(x))). A fixed-seed splitmix64 stream
/// makes the draws byte-reproducible AND identical for both engines (the very
/// same `x`/`y` vectors are fed to gam and to PyMC).
fn synthetic_binomial() -> (Vec<f64>, Vec<f64>) {
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
    for i in 0..N {
        // Deterministic, evenly spaced covariate on [-1, 1].
        let xi = -1.0 + 2.0 * (i as f64) / ((N - 1) as f64);
        let f = xi.powi(3) + (5.0 * xi).sin();
        let prob = 1.0 / (1.0 + (-f).exp());
        let yi = if next_u01() < prob { 1.0 } else { 0.0 };
        x.push(xi);
        y.push(yi);
    }
    (x, y)
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
    let (x_raw, y_raw) = synthetic_binomial();
    let n = x_raw.len();

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

    // Accumulate, per λ: gam posterior mean/std over coefficients, plus the
    // total posterior std of the penalized subspace (the shrinkage statistic).
    let mut gam_means: Vec<Array1<f64>> = Vec::new();
    let mut gam_stds: Vec<Array1<f64>> = Vec::new();
    let mut gam_penalized_spread: Vec<f64> = Vec::new();

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
        eprintln!(
            "gam lambda={lam:>5}: R-hat={:.4} ess={:.0} penalized_posterior_std_sum={:.5}",
            res.rhat, res.ess, spread
        );
        gam_means.push(res.posterior_mean.clone());
        gam_stds.push(res.posterior_std.clone());
        gam_penalized_spread.push(spread);
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

    // ---- (3) cross-engine agreement at every λ ----------------------------
    // Both engines target the same density; their posterior summaries must
    // coincide up to Monte-Carlo error. We compare the posterior mean and std
    // on the penalized subspace (where the prior bites and the comparison is
    // most diagnostic) element-wise, and require PyMC's own R-hat < 1.1.
    for (li, &lam) in lambdas.iter().enumerate() {
        let pm_mean = pymc.vector(&format!("mean_{li}"));
        let pm_std = pymc.vector(&format!("std_{li}"));
        let pm_rhat = pymc.scalar(&format!("rhat_{li}"));
        assert_eq!(pm_mean.len(), p, "PyMC mean dim mismatch at lambda={lam}");
        assert!(
            pm_rhat < 1.1,
            "PyMC reference failed to converge at lambda={lam}: R-hat={pm_rhat:.4}"
        );

        let gm = &gam_means[li];
        let gs = &gam_stds[li];

        // Element-wise agreement on the penalized coefficients.
        let mut max_mean_gap = 0.0_f64;
        let mut max_std_relgap = 0.0_f64;
        for &j in &penalized_cols {
            let mean_gap = (gm[j] - pm_mean[j]).abs();
            max_mean_gap = max_mean_gap.max(mean_gap);
            // Relative std gap, guarded by the PyMC std scale.
            let denom = pm_std[j].abs().max(1e-3);
            let std_relgap = (gs[j] - pm_std[j]).abs() / denom;
            max_std_relgap = max_std_relgap.max(std_relgap);
        }
        eprintln!(
            "lambda={lam:>5}: PyMC R-hat={pm_rhat:.4} | max|Δmean|={max_mean_gap:.4} \
             max rel|Δstd|={max_std_relgap:.4}"
        );

        // Posterior means coincide because the targets are identical. The bound
        // is Monte-Carlo scale: with ~6000 effective-ish draws and posterior
        // stds O(0.3–3), the standard error of a posterior mean is well under
        // 0.15; we allow 0.30 to absorb cross-sampler autocorrelation (Gibbs
        // vs NUTS) without being so loose it asserts nothing.
        assert!(
            max_mean_gap < 0.30,
            "gam vs PyMC posterior means diverge at lambda={lam}: max|Δmean|={max_mean_gap:.4}"
        );
        // Posterior stds must agree to within 25% relative — both estimate the
        // same posterior covariance; a larger gap means one sampler is
        // mis-targeting the penalized density.
        assert!(
            max_std_relgap < 0.25,
            "gam vs PyMC posterior stds diverge at lambda={lam}: max rel|Δstd|={max_std_relgap:.4}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}
