use gam::construction::CanonicalPenalty;
use gam::estimate::{ExternalOptimOptions, PenaltySpec, evaluate_externalcost_andridge};
use gam::pirls::{PenaltyConfig, PirlsConfig, PirlsProblem, fit_model_for_fixed_rho};
use gam::smooth::BlockwisePenalty;
use gam::types::{
    GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LogSmoothingParamsView, ResponseFamily,
    StandardLink,
};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn canonicalize_test_penalties(s_list: &[Array2<f64>]) -> Vec<CanonicalPenalty> {
    let p = s_list[0].nrows();
    s_list
        .iter()
        .enumerate()
        .filter_map(|(idx, s)| {
            gam::construction::canonicalize_penalty_spec(
                &PenaltySpec::Dense(s.clone()),
                p,
                idx,
                "test",
            )
            .expect("canonicalize test penalty")
        })
        .collect()
}

fn make_problem(
    seed: u64,
) -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array2<f64>,
    Vec<BlockwisePenalty>,
) {
    let n = 100;
    let p = 10;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }
    let beta = Array1::from_shape_fn(p, |j| if j == 0 { -0.1 } else { 0.2 / j as f64 });
    let eta = x.dot(&beta);
    let y = eta.mapv(|e| {
        let prob = 1.0 / (1.0 + (-e).exp());
        if rng.random::<f64>() < prob { 1.0 } else { 0.0 }
    });
    let w = Array1::<f64>::ones(n);
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    (x, y, w, s.clone(), vec![BlockwisePenalty::new(0..p, s)])
}

fn binomial_logit_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    )
}

fn fit_beta_norm(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    penalties: &[CanonicalPenalty],
    rho: f64,
    firth: bool,
) -> f64 {
    let p = x.ncols();
    let cfg = PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        link_kind: InverseLink::Standard(StandardLink::Logit),
        max_iterations: 500,
        convergence_tolerance: 1e-10,
        firth_bias_reduction: firth,
        initial_lm_lambda: None,
        arrow_schur: None,
    };
    let offset = Array1::<f64>::zeros(y.len());
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(array![rho].view())
            .expect("test fixture smoothing parameters satisfy the closed domain"),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: w.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: penalties,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &cfg,
        None,
    )
    .expect("fit");
    fit.beta_transformed
        .dot(fit.beta_transformed.as_ref())
        .sqrt()
}

#[test]
fn firthfd_step_size_sensitivity() {
    let (x, y, w, s_dense, s_list) = make_problem(31);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: binomial_logit_likelihood(),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        tol: 1e-10,
        max_iter: 500,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let base_rho = 12.0;
    let cost_at = |rho: f64| -> f64 {
        evaluate_externalcost_andridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &array![rho],
        )
        .map(|(c, _)| c)
        .expect("cost")
    };
    assert!(s_dense.iter().all(|v| v.is_finite()));
    let wide_trend = cost_at(base_rho + 1.0) - cost_at(base_rho - 1.0);
    let trend_sign = wide_trend > 0.0;
    let step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005];
    let mut consistent_count = 0;
    for &h in &step_sizes {
        let fd = (cost_at(base_rho + h) - cost_at(base_rho - h)) / (2.0 * h);
        if (fd > 0.0) == trend_sign {
            consistent_count += 1;
        }
    }
    assert!(consistent_count >= step_sizes.len() / 2);
}

/// Completely-separable logistic fixture: the response is a perfect step
/// function of the second design column (`y == 1  ⇔  x[:,1] > 0`), so the
/// unpenalised logistic MLE diverges (β → ∞ along that column). This is the
/// canonical regime in which Firth's Jeffreys-prior bias reduction earns its
/// keep — it returns finite, shrunk estimates where the plain MLE runs away.
/// The response is a deterministic threshold (no Bernoulli draw), so the
/// separation is *exact* and the fits are reproducible on every platform.
fn separable_problem(n: usize, p: usize) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(31);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }
    let y = Array1::from_shape_fn(n, |i| if x[[i, 1]] > 0.0 { 1.0 } else { 0.0 });
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    (x, y, s)
}

// ---------------------------------------------------------------------------
// Issue #1554. The two tests below replace a pair of "oscillation count"
// asserts that were self-masking in two compounding ways:
//
//   * they compared the *wrong* direction (`no_firth <= firth`) with a vacuous
//     `|| <= N` escape hatch that passed for any value of the other count, and
//   * — confirmed by direct measurement (MSI) — the metric they used had **no
//     signal at all**: at the heavily-penalised operating point (ρ = 12) both
//     the cost and the ‖β‖ profiles are perfectly monotone in ρ for *both*
//     Firth and no-Firth, so every count was 0 and `0 <= 0` could never fail.
//
// A well-converged penalised fit is a smooth function of ρ (implicit function
// theorem), so "Firth reduces oscillation of the converged ρ-profile" is not an
// observable property of this solver. The property the tests were *reaching
// for* — Firth stabilises a degenerate fit — is real and robust, but it lives
// under complete separation at weak penalty, not at ρ = 12. The replacements
// assert that genuine bias-reduction behaviour directly, in both coordinate and
// objective space, with >10× margins.
// ---------------------------------------------------------------------------

/// Coordinate-space view: under complete separation the unpenalised logistic
/// MLE diverges. As the ridge is relaxed (ρ → −∞, λ → 0) the no-Firth
/// coefficient norm runs away without bound, while Firth's Jeffreys penalty
/// pins it to a finite limit. The damping — the ratio ‖β‖_noFirth / ‖β‖_Firth —
/// therefore grows monotonically as the problem degenerates. This is the
/// property the old signal-free "oscillation count" assert was a proxy for.
#[test]
fn firth_dampens_separation_runaway_in_coefficients() {
    let n = 40;
    let p = 6;
    let (x, y, s) = separable_problem(n, p);
    let w = Array1::<f64>::ones(n);
    let penalties = canonicalize_test_penalties(&[s.clone()]);
    // ρ from weak penalty (deep separation, λ → 0) up to a balanced regime.
    let rhos: Vec<f64> = (-16..=0).map(|i| i as f64).collect();
    let bf: Vec<f64> = rhos
        .iter()
        .map(|&r| fit_beta_norm(&x, &y, &w, &penalties, r, true))
        .collect();
    let bnf: Vec<f64> = rhos
        .iter()
        .map(|&r| fit_beta_norm(&x, &y, &w, &penalties, r, false))
        .collect();
    let ratio: Vec<f64> = bf.iter().zip(&bnf).map(|(&f, &nf)| nf / f).collect();
    eprintln!("rhos          = {rhos:?}");
    eprintln!("||b||_firth   = {bf:?}");
    eprintln!("||b||_nofirth = {bnf:?}");
    eprintln!("ratio nf/f    = {ratio:?}");

    // (1) Firth shrinks: ‖β‖_Firth ≤ ‖β‖_noFirth at every ρ.
    for (i, (&f, &nf)) in bf.iter().zip(&bnf).enumerate() {
        assert!(
            f <= nf + 1e-9,
            "Firth must not inflate ‖β‖ vs no-Firth at ρ={}: firth={f} nofirth={nf}",
            rhos[i]
        );
    }

    // (2) No-Firth runaway: at the weakest penalty the plain MLE has blown up,
    //     dwarfing its own balanced-penalty value.
    let bnf_weak = bnf[0]; // ρ = -16
    let bnf_balanced = *bnf.last().unwrap(); // ρ = 0
    assert!(
        bnf_weak > 50.0,
        "no-Firth ‖β‖ should run away under separation as λ→0, got {bnf_weak}"
    );
    assert!(
        bnf_weak > 10.0 * bnf_balanced,
        "no-Firth ‖β‖ should explode as λ→0: weak={bnf_weak} balanced={bnf_balanced}"
    );

    // (3) Firth boundedness: the Firth estimate converges to a finite limit as
    //     λ → 0 — it barely moves across the deep-separation plateau and never
    //     approaches the no-Firth runaway.
    let bf_weak = bf[0]; // ρ = -16
    let bf_plateau = bf[6]; // ρ = -10
    assert!(
        bf_weak < 0.5 * bnf_weak,
        "Firth must stay far below the no-Firth runaway: firth={bf_weak} nofirth={bnf_weak}"
    );
    assert!(
        (bf_weak - bf_plateau).abs() / bf_plateau < 0.05,
        "Firth ‖β‖ should plateau to a finite limit as λ→0: ρ=-16 {bf_weak} vs ρ=-10 {bf_plateau}"
    );

    // (4) Monotone damping: the ratio strictly increases as the penalty weakens
    //     (ρ decreases) — i.e. it is strictly decreasing in ρ — and the damping
    //     is large, not marginal, at λ → 0.
    for i in 0..ratio.len() - 1 {
        assert!(
            ratio[i] > ratio[i + 1],
            "damping ratio must strengthen as λ→0: ratio[ρ={}]={} !> ratio[ρ={}]={}",
            rhos[i],
            ratio[i],
            rhos[i + 1],
            ratio[i + 1]
        );
    }
    assert!(
        ratio[0] > 5.0,
        "expected strong damping at λ→0, ratio={}",
        ratio[0]
    );
}

/// Objective-space view: under complete separation the no-Firth model drives
/// its deviance toward 0 — a perfect, degenerate fit bought by inflating β — as
/// λ → 0, while Firth keeps the deviance bounded above a positive floor. Firth
/// *refuses* the degenerate fit in exchange for finite estimates, so at weak
/// penalty its deviance EXCEEDS no-Firth's (the opposite of the unregularised
/// intuition). A robust witness that the Jeffreys term is active in the inner
/// objective, not merely bolted on after the solve.
#[test]
fn firth_refuses_degenerate_separation_fit_in_deviance() {
    let n = 40;
    let p = 6;
    let (x, y, s) = separable_problem(n, p);
    let w = Array1::<f64>::ones(n);
    let penalties = canonicalize_test_penalties(&[s.clone()]);
    let rhos: Vec<f64> = (-16..=0).map(|i| i as f64).collect();
    let df: Vec<f64> = rhos
        .iter()
        .map(|&r| fit_deviance(&x, &y, &w, &penalties, r, true))
        .collect();
    let dnf: Vec<f64> = rhos
        .iter()
        .map(|&r| fit_deviance(&x, &y, &w, &penalties, r, false))
        .collect();
    eprintln!("rhos        = {rhos:?}");
    eprintln!("dev_firth   = {df:?}");
    eprintln!("dev_nofirth = {dnf:?}");

    // (1) No-Firth collapses to a degenerate perfect fit as λ → 0.
    let dnf_weak = dnf[0]; // ρ = -16
    assert!(
        dnf_weak < 1e-2,
        "no-Firth deviance should collapse toward 0 under separation, got {dnf_weak}"
    );

    // (2) Firth refuses it: deviance bounded above a positive floor and ~flat
    //     across the deep-separation plateau (the finite-estimate limit).
    let df_weak = df[0]; // ρ = -16
    let df_plateau = df[6]; // ρ = -10
    assert!(
        df_weak > 1.0,
        "Firth deviance must stay bounded away from 0 under separation, got {df_weak}"
    );
    assert!(
        (df_weak - df_plateau).abs() / df_plateau < 0.05,
        "Firth deviance should plateau as λ→0: ρ=-16 {df_weak} vs ρ=-10 {df_plateau}"
    );

    // (3) Sign flip: at weak penalty Firth deviance exceeds no-Firth by a wide
    //     margin — it trades fit quality for stability.
    assert!(
        df_weak > 100.0 * dnf_weak,
        "Firth must refuse the degenerate fit: firth={df_weak} nofirth={dnf_weak}"
    );

    // (4) No-Firth deviance falls monotonically toward 0 as λ → 0: it keeps
    //     buying a better separation fit with larger β as the penalty relaxes.
    //     ρ increases left→right, so deviance must increase with ρ.
    for i in 0..dnf.len() - 1 {
        assert!(
            dnf[i] < dnf[i + 1],
            "no-Firth deviance should fall as λ→0: dev[ρ={}]={} !< dev[ρ={}]={}",
            rhos[i],
            dnf[i],
            rhos[i + 1],
            dnf[i + 1]
        );
    }
}

fn fit_deviance(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    penalties: &[CanonicalPenalty],
    rho: f64,
    firth: bool,
) -> f64 {
    let p = x.ncols();
    let cfg = PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        link_kind: InverseLink::Standard(StandardLink::Logit),
        max_iterations: 500,
        convergence_tolerance: 1e-10,
        firth_bias_reduction: firth,
        initial_lm_lambda: None,
        arrow_schur: None,
    };
    let offset = Array1::<f64>::zeros(y.len());
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(array![rho].view())
            .expect("test fixture smoothing parameters satisfy the closed domain"),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: w.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: penalties,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &cfg,
        None,
    )
    .expect("fit");
    fit.deviance
}
