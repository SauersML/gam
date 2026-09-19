use gam::ResourcePolicy;
use gam::custom_family::{CustomFamily, ParameterBlockSpec, ParameterBlockState};
use gam::families::gamlss::{
    BinomialLocationScaleFamily, BinomialMeanWiggleFamily, GaussianLocationScaleFamily,
};
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
    let w = Array1::ones(5);

    // The Gaussian location-scale (log σ, log σ) diagonal and (μ, log σ) cross
    // block are compared against finite differences of the actual
    // log-likelihood, exactly like every other family here.
    //
    // This test used to carve both of them out. It asserted the (ls,ls) block
    // equalled the FISHER information Σ 2κ²a·z² (gam#566) and the (μ,ls) cross
    // equalled the information-orthogonal 0 (gam#684), and — to keep those
    // carve-outs honest — it additionally required each to DIFFER from the FD
    // value by more than 1e-6. Its comment said the Fisher form "is what feeds
    // the REML determinant/EDF".
    //
    // That is now backwards. Production ships the OBSERVED joint Hessian and
    // says so at the single source of truth,
    // `gaussian_locscale_observed_joint_row_coeffs` in
    // `crates/gam-models/src/gamlss/gaussian/joint_psi.rs`, whose doc block
    // retires the #684/#566 block-Fisher object by name: the LAML criterion
    // −½log|H+S| REQUIRES the observed penalized Hessian at β̂
    // (Wood–Pya–Säfken 2016), because zeroing `ml` and expecting `ll` drops the
    // cross-block Schur deficit and the fitted-residual shrinkage, which
    // overstate σ-block information and bias λ̂_σ upward (#1561: log-σ
    // over-smoothing). Ten production sites are committed to observed; this
    // test was the sole holdout.
    //
    // So the carve-outs asserted the opposite of shipped behaviour, and their
    // "must differ from FD" guards made that unavoidable: observed curvature IS
    // what FD recovers, so the guard fails precisely when production is right.
    // `ll = κ′(a−n) + 2κ²n` equals the Fisher `2κ²a` only at the truth (n→a),
    // and `ml = 2κm` is the exact cross block, zero only in expectation — this
    // fixture is deliberately off-truth, which is why the two disagree by 3.19×
    // rather than by roundoff.
    //
    // Removing the override does not weaken anything: both blocks are now held
    // to the same FD bar as every other block, which is a stronger claim than
    // "equals a closed form we also wrote here", and it is a bar that tracks
    // production instead of a retired object.
    let families: Vec<(
        Box<dyn CustomFamily>,
        Vec<ParameterBlockSpec>,
        Array1<f64>,
        Option<(usize, usize)>,
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
        ),
        (
            Box::new(BinomialLocationScaleFamily {
                y: y_b.clone(),
                weights: w.clone(),
                link_kind: InverseLink::Standard(StandardLink::Logit),
                threshold_design: Some(DesignMatrix::from(x.clone())),
                log_sigma_design: Some(DesignMatrix::from(z.clone())),
                policy: ResourcePolicy::default_library(),
                jeffreys_armed: false,
            }),
            vec![spec("threshold", &x), spec("log_sigma", &z)],
            array![0.1, 0.15],
            Some((0, 1)),
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
        ),
    ];

    for (fam, specs, beta0, cross_pair) in families {
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
            // The score (gradient) is the exact observed gradient, so it must
            // match FD of the log-likelihood to machine precision — this is what
            // guarantees the joint Newton converges to the true MLE stationary
            // point.
            assert!(
                (analytic_grad[i] - g_fd).abs() <= 1e-7,
                "grad mismatch i={i}: analytic={} fd={}",
                analytic_grad[i],
                g_fd
            );
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
            // Every cross block, including the Gaussian location-scale
            // (μ, log σ) one, is held to FD of the actual log-likelihood.
            //
            // This arm used to assert that block was the information-orthogonal
            // 0 (gam#684) AND that it differed from FD by more than 1e-6. The
            // shipped cross block is the OBSERVED `ml = 2κm`, which is zero only
            // in EXPECTATION — `2κ·E[r]·w/σ² = 0` holds at the truth, not at an
            // arbitrary β. On this deliberately off-truth fixture it is plainly
            // nonzero, so both halves of that carve-out contradicted production;
            // see the note on the fixture above for why the block-Fisher object
            // was retired.
            assert!(
                (analytic_h[[i, j]] - c_fd).abs() <= 1e-5,
                "cross mismatch ({i},{j}): analytic={} fd={}",
                analytic_h[[i, j]],
                c_fd
            );
        }
    }
}
