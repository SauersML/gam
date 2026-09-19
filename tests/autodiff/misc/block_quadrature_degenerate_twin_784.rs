//! #784: a flagged block direction whose eigenvalue is exactly doubled refuses the block quadrature correction
//! typed, at the eigenframe near-degeneracy stage, instead of splicing a silent zero.
//!
//! The design is two orthogonal, identical blocks. Every row appears once under level 1 (columns 0..4) and once under
//! level 2 (columns 4..8), with a cubic polynomial block per level and a ridge penalty per block. At a symmetric ρ the
//! penalized Hessian is block-diagonal with equal blocks, so every eigenvalue is doubled and a flagged direction has a
//! twin within its measured resolution. The asymmetric ρ separates the twins and is the control.
//!
//! Measured by the lane probe (job 1217955 at fb91ed1fed) on this fixture: at ρ = [−2, −2] the skewness verdict flags
//! m = 2 directions and the flagged eigenvalue's twin sits at gap 0 against a resolution of 6.3e-15. At ρ = [−2, −1]
//! the same directions are flagged and nothing declines.

use gam::estimate::{ExternalOptimOptions, evaluate_externalcost};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_problem::FailureCategory;
use gam_problem::estimation_error::{BlockQuadratureCorrectionStage, EstimationError};
use ndarray::{Array1, Array2, array};

fn logit_options() -> ExternalOptimOptions {
    ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![0, 0],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    }
}

/// 60 base rows on [−1, 1] with a deterministic logistic response, duplicated across two levels, and a cubic
/// polynomial block per level.
fn twin_block_design() -> (Array1<f64>, Array2<f64>) {
    let n0 = 60usize;
    let slope = 8.0;
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let base: Vec<(f64, f64)> = (0..n0)
        .map(|i| {
            let x = -1.0 + 2.0 * (i as f64) / ((n0 - 1) as f64);
            let mu = 1.0 / (1.0 + (-(slope * x - 0.5)).exp());
            (x, if uniform() < mu { 1.0 } else { 0.0 })
        })
        .collect();
    let mut y = Array1::<f64>::zeros(2 * n0);
    let mut x = Array2::<f64>::zeros((2 * n0, 8));
    for level in 0..2 {
        for (i, &(xi, yi)) in base.iter().enumerate() {
            let row = level * n0 + i;
            y[row] = yi;
            for power in 0..4 {
                x[(row, 4 * level + power)] = xi.powi(power as i32);
            }
        }
    }
    (y, x)
}

fn cost_at(rho: Array1<f64>) -> Result<f64, EstimationError> {
    let (y, x) = twin_block_design();
    let w = Array1::<f64>::ones(y.len());
    let offset = Array1::<f64>::zeros(y.len());
    let penalties = vec![
        BlockwisePenalty::new(0..4, Array2::eye(4)),
        BlockwisePenalty::new(4..8, Array2::eye(4)),
    ];
    evaluate_externalcost(
        y.view(),
        w.view(),
        x,
        offset.view(),
        &penalties,
        &logit_options(),
        &rho,
    )
}

#[test]
fn doubled_flagged_eigenvalue_refuses_the_block_correction_at_the_near_degeneracy_stage_784() {
    // Registers the Laplace marginal corrector; without it the correction declines before the diagnostic.
    gam::init_parallelism();
    let symmetric = cost_at(array![-2.0, -2.0]);
    let asymmetric = cost_at(array![-2.0, -1.0]);
    match &symmetric {
        Ok(cost) => eprintln!("[#784 twin pin] symmetric rho=[-2, -2]: Ok cost={cost:.12e}"),
        Err(error) => eprintln!("[#784 twin pin] symmetric rho=[-2, -2]: Err {error}"),
    }
    match &asymmetric {
        Ok(cost) => eprintln!("[#784 twin pin] asymmetric rho=[-2, -1]: Ok cost={cost:.12e}"),
        Err(error) => eprintln!("[#784 twin pin] asymmetric rho=[-2, -1]: Err {error}"),
    }

    let error = match symmetric {
        Err(error) => error,
        Ok(cost) => panic!("the symmetric twin spliced a cost ({cost:.12e}) instead of refusing typed"),
    };
    let EstimationError::BlockQuadratureCorrectionRefused {
        stage:
            BlockQuadratureCorrectionStage::EigenframeNearDegeneracy {
                block_eigenvalue,
                other_eigenvalue,
                gap,
                tolerance,
            },
    } = &error
    else {
        panic!("the symmetric twin refused at the wrong stage: {error}");
    };
    eprintln!(
        "[#784 twin pin] near-degeneracy: block_eigenvalue={block_eigenvalue:e} \
         other_eigenvalue={other_eigenvalue:e} gap={gap:e} tolerance={tolerance:e}"
    );
    assert!(
        gap <= tolerance,
        "the refusal must carry a gap inside its own resolution: gap={gap:e} tolerance={tolerance:e}"
    );
    assert!(
        error.is_trial_point_infeasible(),
        "a near-degenerate eigenframe is a fact about this ρ, so the outer search backs off it: {error}"
    );
    assert!(
        error.failure_category() == FailureCategory::Convergence,
        "a rho-local refusal is a convergence failure: {error}"
    );

    match asymmetric {
        Ok(cost) => assert!(cost.is_finite(), "the separated control must score finite, got {cost:e}"),
        Err(error) => panic!("the separated control must not refuse: {error}"),
    }
}
