//! A random-effect factor with many levels is fitted on the sparse exact path
//! (sparse `XᵀWX + S_λ` Cholesky, block-diagonal original-frame penalty
//! algebra, closed-form diagonal `log|S_λ|₊`). None of that structure may move
//! the REML objective: the same model handed over as a dense design takes the
//! dense spectral path, and both must land on the same optimum.
//!
//! The design is `intercept + smooth block + one-hot factor`. The smooth block
//! carries two penalties on one coefficient range (a wiggliness penalty and a
//! null-space ridge, as `s(x)` does), so the sparse path is exercised on shared
//! penalty ranges as well as on the diagonal random-effect penalty.

use faer::sparse::{SparseColMat, Triplet};
use gam_problem::LikelihoodSpec;
use gam_solve::estimate::{
    ExternalOptimOptions, ExternalOptimResult, optimize_external_designwith_heuristic_log_lambdas,
};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

const LEVELS: usize = 160;
const PER_LEVEL: usize = 15;
const ROWS: usize = LEVELS * PER_LEVEL;
const SMOOTH: usize = 4;
const COLUMNS: usize = 1 + SMOOTH + LEVELS;

struct Problem {
    y: Array1<f64>,
    dense: Array2<f64>,
    sparse: SparseColMat<usize, f64>,
    penalties: Vec<BlockwisePenalty>,
}

fn problem() -> Problem {
    let mut dense = Array2::<f64>::zeros((ROWS, COLUMNS));
    let mut y = Array1::<f64>::zeros(ROWS);
    for row in 0..ROWS {
        // Levels are interleaved so every level sees the whole covariate range.
        let level = row % LEVELS;
        let t = (row as f64 + 0.5) / ROWS as f64;
        dense[[row, 0]] = 1.0;
        for j in 0..SMOOTH {
            let harmonic = (j / 2 + 1) as f64;
            let angle = 2.0 * std::f64::consts::PI * harmonic * t;
            dense[[row, 1 + j]] = if j % 2 == 0 { angle.sin() } else { angle.cos() };
        }
        dense[[row, 1 + SMOOTH + level]] = 1.0;
        let level_effect = 0.8 * ((level as f64) * 2.399_963).sin();
        let noise = 0.3 * ((row as f64) * 12.989_8).sin() * ((row as f64) * 0.618_034).cos();
        y[row] = 1.0 + (6.0 * t).sin() + level_effect + noise;
    }
    let triplets: Vec<Triplet<usize, usize, f64>> = dense
        .indexed_iter()
        .filter(|&(_, &value)| value != 0.0)
        .map(|((row, col), &value)| Triplet::new(row, col, value))
        .collect();
    let sparse = SparseColMat::try_new_from_triplets(ROWS, COLUMNS, &triplets)
        .expect("the one-hot design is a valid sparse matrix");

    // Second-difference wiggliness penalty `DᵀD` (null space: constant and
    // linear coefficient sequences) and the projector onto that null space,
    // the double-penalty pair `s(x, select=True)` builds on one range.
    let mut difference = Array2::<f64>::zeros((SMOOTH - 2, SMOOTH));
    for row in 0..SMOOTH - 2 {
        difference[[row, row]] = 1.0;
        difference[[row, row + 1]] = -2.0;
        difference[[row, row + 2]] = 1.0;
    }
    let wiggle = difference.t().dot(&difference);
    let constant = Array1::<f64>::from_elem(SMOOTH, 0.5);
    let mean_index = (SMOOTH as f64 - 1.0) / 2.0;
    let linear = Array1::from_shape_fn(SMOOTH, |j| j as f64 - mean_index);
    let linear = &linear / linear.dot(&linear).sqrt();
    let ridge = Array2::from_shape_fn((SMOOTH, SMOOTH), |(i, j)| {
        constant[i] * constant[j] + linear[i] * linear[j]
    });
    let smooth_range = 1..1 + SMOOTH;
    let penalties = vec![
        BlockwisePenalty::new(smooth_range.clone(), wiggle),
        BlockwisePenalty::new(smooth_range, ridge),
        BlockwisePenalty::new(1 + SMOOTH..COLUMNS, Array2::<f64>::eye(LEVELS)),
    ];
    Problem {
        y,
        dense,
        sparse,
        penalties,
    }
}

fn fit(problem: &Problem, sparse: bool) -> ExternalOptimResult {
    let weights = Array1::<f64>::ones(ROWS);
    let offset = Array1::<f64>::zeros(ROWS);
    let options = ExternalOptimOptions {
        family: LikelihoodSpec::gaussian_identity(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 200,
        tol: 1e-9,
        nullspace_dims: vec![2, 2, 0],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    };
    let result = if sparse {
        optimize_external_designwith_heuristic_log_lambdas(
            problem.y.view(),
            weights.view(),
            problem.sparse.clone(),
            offset.view(),
            problem.penalties.clone(),
            None,
            &options,
        )
    } else {
        optimize_external_designwith_heuristic_log_lambdas(
            problem.y.view(),
            weights.view(),
            problem.dense.clone(),
            offset.view(),
            problem.penalties.clone(),
            None,
            &options,
        )
    };
    result.unwrap_or_else(|error| panic!("the random-effect fit must succeed: {error:?}"))
}

#[test]
fn many_level_random_effect_sparse_fit_matches_the_dense_fit() {
    let problem = problem();
    let dense = fit(&problem, false);
    let sparse = fit(&problem, true);

    assert!(dense.outer_converged, "the dense fit must converge");
    assert!(sparse.outer_converged, "the sparse fit must converge");

    let dense_score = dense
        .reml_score
        .expect("the dense fit has a finite REML score");
    let sparse_score = sparse
        .reml_score
        .expect("the sparse fit has a finite REML score");
    let score_gap = (sparse_score - dense_score).abs() / dense_score.abs().max(1.0);
    assert!(
        score_gap <= 1e-9,
        "REML optimum: sparse {sparse_score:.15e} vs dense {dense_score:.15e}"
    );

    for (k, (&got, &want)) in sparse
        .log_lambdas
        .iter()
        .zip(dense.log_lambdas.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() <= 1e-4,
            "log-lambda {k}: sparse {got:.12} vs dense {want:.12}"
        );
    }

    let beta_scale = dense.beta.iter().fold(1.0_f64, |m, b| m.max(b.abs()));
    for (j, (&got, &want)) in sparse.beta.iter().zip(dense.beta.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-6 * beta_scale,
            "coefficient {j}: sparse {got:.15e} vs dense {want:.15e}"
        );
    }
}
