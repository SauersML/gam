use gam::ResourcePolicy;
use gam::custom_family::{CustomFamily, ParameterBlockSpec, ParameterBlockState};
use gam::families::gamlss::{
    BinomialLocationScaleFamily, BinomialMeanWiggleFamily, GammaLogFamily,
    GaussianLocationScaleFamily, PoissonLogFamily,
};
use gam::families::sigma_link::LOGB_SIGMA_FLOOR;
use gam::matrix::DesignMatrix;
use gam::types::{InverseLink, StandardLink};
use ndarray::{Array1, Array2, array};

fn spec(name: &str, x: &Array2<f64>) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::from(x.clone()),
        offset: Array1::zeros(x.nrows()),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn fd_grad<F: Fn(&Array1<f64>) -> f64>(f: &F, b: &Array1<f64>, i: usize, h: f64) -> f64 {
    let mut bp = b.clone();
    let mut bm = b.clone();
    bp[i] += h;
    bm[i] -= h;
    (f(&bp) - f(&bm)) / (2.0 * h)
}
fn fd_hess_diag<F: Fn(&Array1<f64>) -> f64>(f: &F, b: &Array1<f64>, i: usize, h: f64) -> f64 {
    let mut bp = b.clone();
    let mut bm = b.clone();
    bp[i] += h;
    bm[i] -= h;
    (f(&bp) - 2.0 * f(b) + f(&bm)) / (h * h)
}
fn fd_cross<F: Fn(&Array1<f64>) -> f64>(f: &F, b: &Array1<f64>, i: usize, j: usize, h: f64) -> f64 {
    let mut bpp = b.clone();
    let mut bpm = b.clone();
    let mut bmp = b.clone();
    let mut bmm = b.clone();
    bpp[i] += h;
    bpp[j] += h;
    bpm[i] += h;
    bpm[j] -= h;
    bmp[i] -= h;
    bmp[j] += h;
    bmm[i] -= h;
    bmm[j] -= h;
    (f(&bpp) - f(&bpm) - f(&bmp) + f(&bmm)) / (4.0 * h * h)
}

#[test]
fn gamlss_joint_derivatives_match_finite_difference() {
    let x = array![[1.0], [0.4], [-0.7], [1.3], [-1.1]];
    let z = array![[0.2], [1.1], [-0.5], [0.7], [-1.4]];
    let y_g = array![0.5, -0.1, 0.8, 1.6, -0.4];
    let y_b = array![1.0, 0.0, 1.0, 0.0, 1.0];
    let y_p = array![2.0, 0.0, 1.0, 3.0, 4.0];
    let y_ga = array![1.2, 0.8, 2.0, 1.5, 0.6];
    let w = Array1::ones(5);

    // The (log σ, log σ) block of the Gaussian location-scale joint Hessian is
    // the Fisher/expected information E[H_{ls,ls}] = Σ 2κ²a·z² (gam#566), NOT
    // the observed log-likelihood curvature 2κ²n + κ'(a−n) that finite
    // differencing the log-likelihood recovers. The Fisher form is what feeds
    // the REML determinant/EDF (Fisher scoring, as gamlss/mgcv gaulss do); the
    // score stays the exact observed gradient so the stationary point is
    // unchanged. So for the Gaussian ls block we assert against the analytic
    // Fisher value instead of FD; every other diagonal/cross block and every
    // score component still matches FD of the actual log-likelihood exactly.
    type LsFisherOverride = Option<(usize, Box<dyn Fn(&Array1<f64>) -> f64>)>;
    let z_for_fisher = z.clone();
    let w_for_fisher = w.clone();
    let gaussian_ls_fisher: LsFisherOverride = Some((
        1usize,
        Box::new(move |beta: &Array1<f64>| {
            // η_ls = z·β_ls; σ = LOGB_SIGMA_FLOOR + exp(η_ls); κ = (dσ/dη)/σ.
            (0..z_for_fisher.nrows())
                .map(|i| {
                    let eta_ls = z_for_fisher[[i, 0]] * beta[1];
                    let dsigma = eta_ls.exp();
                    let sigma = LOGB_SIGMA_FLOOR + dsigma;
                    let kappa = dsigma / sigma;
                    2.0 * kappa * kappa * w_for_fisher[i] * z_for_fisher[[i, 0]].powi(2)
                })
                .sum()
        }),
    ));

    let families: Vec<(
        Box<dyn CustomFamily>,
        Vec<ParameterBlockSpec>,
        Array1<f64>,
        Option<(usize, usize)>,
        LsFisherOverride,
    )> = vec![
        (
            Box::new(GaussianLocationScaleFamily {
                y: y_g.clone(),
                weights: w.clone(),
                mu_design: Some(DesignMatrix::from(x.clone())),
                log_sigma_design: Some(DesignMatrix::from(z.clone())),
                policy: ResourcePolicy::default_library(),
                cached_row_scalars: std::sync::RwLock::new(None),
            }),
            vec![spec("mu", &x), spec("log_sigma", &z)],
            array![0.3, -0.2],
            Some((0, 1)),
            gaussian_ls_fisher,
        ),
        (
            Box::new(BinomialLocationScaleFamily {
                y: y_b.clone(),
                weights: w.clone(),
                link_kind: InverseLink::Standard(StandardLink::Logit),
                threshold_design: Some(DesignMatrix::from(x.clone())),
                log_sigma_design: Some(DesignMatrix::from(z.clone())),
                policy: ResourcePolicy::default_library(),
            }),
            vec![spec("threshold", &x), spec("log_sigma", &z)],
            array![0.1, 0.15],
            Some((0, 1)),
            None,
        ),
        (
            // The wiggle warp is pinned to a frozen design (`frozen_warp_design`)
            // so `∂q/∂q0 = 1` and no live degree-3 spline basis is rebuilt from
            // `wiggle_knots` (that dynamic basis needs ≥ 8 knots and a β_w whose
            // length matches its column count — inconsistent with this test's
            // uniform scalar-per-block layout). The frozen design equals `z`, so
            // the wiggle block η_w = z·β_w matches the shared block-1 state
            // construction below exactly (identical convention to the log-σ
            // block of the other location-scale families), keeping the analytic
            // score and joint Hessian consistent with the finite differences.
            Box::new(BinomialMeanWiggleFamily {
                y: y_b.clone(),
                weights: w.clone(),
                link_kind: InverseLink::Standard(StandardLink::Logit),
                wiggle_knots: array![-1.0, -0.3, 0.4, 1.1],
                wiggle_degree: 3,
                policy: ResourcePolicy::default_library(),
                frozen_warp_design: Some(std::sync::Arc::new(z.clone())),
            }),
            vec![spec("eta", &x), spec("wiggle", &z)],
            array![0.05, 0.02],
            Some((0, 1)),
            None,
        ),
        (
            Box::new(PoissonLogFamily {
                y: y_p.clone(),
                weights: w.clone(),
            }),
            vec![spec("eta", &x)],
            array![0.25],
            None,
            None,
        ),
        (
            Box::new(GammaLogFamily {
                y: y_ga.clone(),
                weights: w.clone(),
                shape: 2.4,
            }),
            vec![spec("eta", &x)],
            array![0.2],
            None,
            None,
        ),
    ];

    for (fam, specs, beta0, cross_pair, ls_fisher_override) in families {
        let f = |b: &Array1<f64>| {
            let states = if specs.len() == 2 {
                vec![
                    ParameterBlockState {
                        beta: array![b[0]],
                        eta: x.column(0).to_owned() * b[0],
                    },
                    ParameterBlockState {
                        beta: array![b[1]],
                        eta: z.column(0).to_owned() * b[1],
                    },
                ]
            } else {
                vec![ParameterBlockState {
                    beta: array![b[0]],
                    eta: x.column(0).to_owned() * b[0],
                }]
            };
            fam.evaluate(&states).unwrap().log_likelihood
        };
        let states = if specs.len() == 2 {
            vec![
                ParameterBlockState {
                    beta: array![beta0[0]],
                    eta: x.column(0).to_owned() * beta0[0],
                },
                ParameterBlockState {
                    beta: array![beta0[1]],
                    eta: z.column(0).to_owned() * beta0[1],
                },
            ]
        } else {
            vec![ParameterBlockState {
                beta: array![beta0[0]],
                eta: x.column(0).to_owned() * beta0[0],
            }]
        };
        let analytic_grad = fam
            .exact_newton_joint_gradient_evaluation(&states, &specs)
            .unwrap()
            .unwrap()
            .gradient;
        let h_pos = fam
            .exact_newton_joint_hessian_with_specs(&states, &specs)
            .unwrap()
            .unwrap();
        let analytic_h = -&h_pos;
        for i in 0..beta0.len() {
            let g_fd = fd_grad(&f, &beta0, i, 1e-6);
            // The score (gradient) is always the exact observed gradient, so it
            // must match FD of the log-likelihood to machine precision — this is
            // what guarantees the joint Newton converges to the true MLE
            // stationary point even when the (ls,ls) curvature is Fisher.
            assert!(
                (analytic_grad[i] - g_fd).abs() <= 1e-7,
                "grad mismatch i={i}: analytic={} fd={}",
                analytic_grad[i],
                g_fd
            );
            if let Some((ls_idx, fisher_fn)) = ls_fisher_override.as_ref()
                && *ls_idx == i
            {
                // Fisher (expected) (ls,ls) information feeds the REML
                // determinant (gam#566); assert the analytic block equals the
                // analytic Fisher value Σ 2κ²a·z², not the observed FD curvature.
                // `analytic_h = -h_pos`, so the log-likelihood-space entry is
                // the negative of the (positive-definite) Fisher information.
                let fisher = fisher_fn(&beta0);
                assert!(
                    (analytic_h[[i, i]] + fisher).abs() <= 1e-9 * (1.0 + fisher.abs()),
                    "ls,ls Fisher-info mismatch i={i}: analytic={} expected_fisher_negated={}",
                    analytic_h[[i, i]],
                    -fisher
                );
                // Confirm the Fisher form genuinely differs from the observed FD
                // curvature on this (off-truth) data point — otherwise the
                // assertion above would be vacuously also-FD and the #566 change
                // untested.
                let h_fd = fd_hess_diag(&f, &beta0, i, 1e-5);
                assert!(
                    (analytic_h[[i, i]] - h_fd).abs() > 1e-6,
                    "ls,ls Fisher and observed FD coincide (test no longer exercises #566): analytic={} fd={}",
                    analytic_h[[i, i]],
                    h_fd
                );
                continue;
            }
            // Central second-difference step. The earlier h=1e-5 sat well below
            // the optimal ~ε^{1/4} (≈1.2e-4) for a second difference, so the
            // estimate was catastrophic-cancellation / roundoff limited: on the
            // Gaussian (μ,μ) block (where the log-likelihood is EXACTLY quadratic
            // in β_μ, so the truncation error is identically zero) the analytic
            // −Σx²/σ² and the FD disagreed by ~1.5e-5 purely from f-evaluation
            // roundoff amplified by 1/h². Using h=1e-4 (near the roundoff/
            // truncation optimum) shrinks that to <2e-7 while keeping every
            // non-quadratic block's O(h²) truncation far under the 1e-5 bar. The
            // analytic value is the one being trusted; this only sharpens the FD
            // yardstick, it does not relax the accept tolerance.
            let h_fd = fd_hess_diag(&f, &beta0, i, 1e-4);
            assert!(
                (analytic_h[[i, i]] - h_fd).abs() <= 1e-5,
                "hess diag mismatch i={i}: analytic={} fd={}",
                analytic_h[[i, i]],
                h_fd
            );
        }
        if let Some((i, j)) = cross_pair {
            let c_fd = fd_cross(&f, &beta0, i, j, 1e-4);
            if ls_fisher_override.is_some() {
                // The Gaussian location-scale joint Hessian's (μ, log σ) cross
                // block is the Fisher/expected information E[H_{μ,ls}] = 2κ·E[m]
                // = 2κ·E[r]·w/σ² = 0 — location and scale are information-
                // orthogonal (gam#684) — NOT the observed 2κm the FD of the
                // log-likelihood recovers. This is the SAME expected-curvature
                // choice that makes the (ls,ls) block Fisher above (Fisher
                // scoring, as gamlss/mgcv gaulss): the observed 2κm is mean-zero
                // noise that would inject spurious μ↔σ coupling into the REML
                // determinant via the Schur complement and over-smooth log σ.
                // So assert the analytic block is the exact analytic Fisher value
                // (0), and confirm it genuinely differs from the observed FD so
                // the #684 orthogonalization stays exercised rather than being
                // vacuously also-FD.
                assert!(
                    analytic_h[[i, j]].abs() <= 1e-12,
                    "μ,ls Fisher cross should be the orthogonal 0 (gam#684): analytic={}",
                    analytic_h[[i, j]]
                );
                assert!(
                    (analytic_h[[i, j]] - c_fd).abs() > 1e-6,
                    "μ,ls Fisher cross and observed FD coincide (test no longer exercises #684): analytic={} fd={}",
                    analytic_h[[i, j]],
                    c_fd
                );
            } else {
                assert!(
                    (analytic_h[[i, j]] - c_fd).abs() <= 1e-5,
                    "cross mismatch ({i},{j}): analytic={} fd={}",
                    analytic_h[[i, j]],
                    c_fd
                );
            }
        }
    }
}
