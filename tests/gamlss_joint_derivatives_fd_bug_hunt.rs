use gam::custom_family::{CustomFamily, ParameterBlockSpec, ParameterBlockState};
use gam::families::gamlss::{
    BinomialLocationScaleFamily, BinomialMeanWiggleFamily, GammaLogFamily,
    GaussianLocationScaleFamily, PoissonLogFamily,
};
use gam::matrix::DesignMatrix;
use gam::resource::ResourcePolicy;
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
        row_scaling: None,
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
            }),
            vec![spec("threshold", &x), spec("log_sigma", &z)],
            array![0.1, 0.15],
            Some((0, 1)),
        ),
        (
            Box::new(BinomialMeanWiggleFamily {
                y: y_b.clone(),
                weights: w.clone(),
                link_kind: InverseLink::Standard(StandardLink::Logit),
                wiggle_knots: array![-1.0, -0.3, 0.4, 1.1],
                wiggle_degree: 3,
                policy: ResourcePolicy::default_library(),
            }),
            vec![
                spec("eta", &x),
                spec("wiggle", &array![[1.0], [1.0], [1.0], [1.0], [1.0]]),
            ],
            array![0.05, 0.02],
            Some((0, 1)),
        ),
        (
            Box::new(PoissonLogFamily {
                y: y_p.clone(),
                weights: w.clone(),
            }),
            vec![spec("eta", &x)],
            array![0.25],
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
            let h_fd = fd_hess_diag(&f, &beta0, i, 1e-5);
            assert!(
                (analytic_grad[i] - g_fd).abs() <= 1e-7,
                "grad mismatch i={i}: analytic={} fd={}",
                analytic_grad[i],
                g_fd
            );
            assert!(
                (analytic_h[[i, i]] - h_fd).abs() <= 1e-5,
                "hess diag mismatch i={i}: analytic={} fd={}",
                analytic_h[[i, i]],
                h_fd
            );
        }
        if let Some((i, j)) = cross_pair {
            let c_fd = fd_cross(&f, &beta0, i, j, 1e-5);
            assert!(
                (analytic_h[[i, j]] - c_fd).abs() <= 1e-5,
                "cross mismatch ({i},{j}): analytic={} fd={}",
                analytic_h[[i, j]],
                c_fd
            );
        }
    }
}
