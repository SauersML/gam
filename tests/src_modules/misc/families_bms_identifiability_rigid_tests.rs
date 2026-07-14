use super::exact_eval_cache::*;

use super::family::*;

use super::gradient_paths::*;

use super::hessian_paths::*;

use super::install_flex::validate_spec;

use super::*;

use super::exact_kernel::{
    DenestedCubicCell as ExactDenestedCubicCell, ExactCellBranch as ExactCellBranchShared,
    LocalSpanCubic, branch_cell as branch_exact_cell, build_denested_partition_cells,
    denested_cell_coefficient_partials as exact_denested_cell_coefficient_partials,
    global_cubic_from_local as exact_global_cubic_from_local,
    transformed_link_cubic as exact_transformed_link_cubic,
};

use crate::custom_family::{CustomFamily, ExactOuterDerivativeOrder};

use ndarray::array;

fn rigid_test_design_hyper_layout(
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
) -> crate::custom_family::CustomFamilyHyperLayout {
    let axis_count: usize = derivative_blocks.iter().map(Vec::len).sum();
    crate::custom_family::CustomFamilyHyperLayout::new(
        derivative_blocks.to_vec(),
        Vec::new(),
        Array1::zeros(axis_count),
    )
    .expect("rigid BMS test design-hyper layout")
}

/// Standard normal PDF's own derivative stack (as opposed to
/// [`unary_derivatives_normal_cdf`]'s CDF stack). No production row path
/// differentiates the bare PDF; this oracle test does.
fn unary_derivatives_normal_pdf(x: f64) -> [f64; 5] {
    let pdf = normal_pdf(x);
    [
        pdf,
        -x * pdf,
        (x * x - 1.0) * pdf,
        (-x.powi(3) + 3.0 * x) * pdf,
        (x.powi(4) - 6.0 * x * x + 3.0) * pdf,
    ]
}

#[inline]
fn bernoulli_marginal_slope_probit_link() -> InverseLink {
    InverseLink::Standard(StandardLink::Probit)
}

fn dense_design(matrix: Array2<f64>) -> DesignMatrix {
    DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(matrix))
}

fn test_family_with_dense_designs(
    y: Array1<f64>,
    weights: Array1<f64>,
    z: Array1<f64>,
    marginal_design: Array2<f64>,
    logslope_design: Array2<f64>,
) -> BernoulliMarginalSlopeFamily {
    BernoulliMarginalSlopeFamily {
        y: Arc::new(y),
        weights: Arc::new(weights),
        z: Arc::new(z),
        marginal_design: dense_design(marginal_design),
        logslope_design: dense_design(logslope_design),
        ..default_test_family()
    }
}

fn test_family_with_zero_primary_designs(
    y: Array1<f64>,
    weights: Array1<f64>,
    z: Array1<f64>,
) -> BernoulliMarginalSlopeFamily {
    let n = z.len();
    test_family_with_dense_designs(y, weights, z, Array2::zeros((n, 0)), Array2::zeros((n, 0)))
}

fn test_family_with_intercept_designs(
    y: Array1<f64>,
    weights: Array1<f64>,
    z: Array1<f64>,
) -> BernoulliMarginalSlopeFamily {
    let n = z.len();
    let ones_col = Array2::ones((n, 1));
    test_family_with_dense_designs(y, weights, z, ones_col.clone(), ones_col)
}

fn default_test_family() -> BernoulliMarginalSlopeFamily {
    let empty_design = dense_design(Array2::zeros((0, 0)));
    BernoulliMarginalSlopeFamily {
        y: Arc::new(Array1::zeros(0)),
        weights: Arc::new(Array1::zeros(0)),
        z: Arc::new(Array1::zeros(0)),
        marginal_design: empty_design.clone(),
        logslope_design: empty_design,
        latent_measure: LatentMeasureKind::StandardNormal,
        gaussian_frailty_sd: None,
        base_link: InverseLink::Standard(gam_spec::StandardLink::Probit),
        score_warp: None,
        link_dev: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cell_moment_lru: Arc::new(exact_kernel::CellMomentLruCache::new(1024)),
        cell_moment_cache_stats: Arc::new(exact_kernel::CellMomentCacheStats::default()),
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(std::sync::Mutex::new(None)),
    }
}

#[test]
fn bernoulli_marginal_slope_outer_seed_config_screens_glm_stability_anchors() {
    let config = default_test_family().outer_seed_config(6);
    assert_eq!(
        config.risk_profile,
        gam_solve::seeding::SeedRiskProfile::GeneralizedLinear
    );
    assert_eq!(config.seed_budget, 1);
    // The BMS marginal-slope startup screen caps inner iterations at the first
    // viable reachability floor (8). Two cycles sits below the observed KKT
    // reachability floor for these startup seeds: it rejects every candidate
    // and then immediately pays a second screening pass at cap=8, so the
    // production config (`outer_seed_config`) starts at 8 and lets the cascade
    // escalate only when needed. See d388d12e7.
    assert_eq!(config.screen_max_inner_iterations, 8);
    assert_eq!(config.max_seeds, 6);

    let seeds = gam_solve::seeding::generate_rho_candidates(6, None, &config);
    for anchor in [2.0, 4.0] {
        assert!(
            seeds
                .iter()
                .any(|seed| seed.iter().all(|rho| (*rho - anchor).abs() < 1e-12)),
            "missing symmetric GLM startup anchor rho={anchor}"
        );
    }
}

fn empty_termspec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    }
}

fn dummy_blockspec(p: usize, n_rows: usize) -> ParameterBlockSpec {
    // Block names must be unique across a single `validate_blockspecs`
    // call (the family layer keys coefficient labels by name). Tests
    // commonly build `vec![dummy_blockspec(...), dummy_blockspec(...)]`,
    // so the helper draws a fresh suffix from a per-process counter
    // rather than hard-coding "dummy".
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let idx = SEQ.fetch_add(1, Ordering::Relaxed);
    ParameterBlockSpec {
        name: format!("dummy_{idx}"),
        design: dense_design(Array2::<f64>::zeros((n_rows, p))),
        offset: Array1::zeros(n_rows),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(Array1::zeros(p)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn dummy_block_state(beta: Array1<f64>, n_rows: usize) -> ParameterBlockState {
    ParameterBlockState {
        beta,
        eta: Array1::zeros(n_rows),
    }
}

fn scalar_block_state(value: f64, n_rows: usize) -> ParameterBlockState {
    ParameterBlockState {
        beta: array![value],
        eta: Array1::from_elem(n_rows, value),
    }
}

fn flex_hessian_matvec_fixture(
    n: usize,
) -> Result<
    (
        BernoulliMarginalSlopeFamily,
        Vec<ParameterBlockState>,
        BernoulliMarginalSlopeExactEvalCache,
        Array1<f64>,
    ),
    String,
> {
    let z = Array1::from_iter((0..n).map(|i| {
        let t = (i as f64 + 0.5) / n as f64;
        (10.0 * t).sin() + 0.3 * (31.0 * t).cos()
    }));
    let y = Array1::from_iter((0..n).map(|i| if (i * 37 + 11) % 101 < 43 { 1.0 } else { 0.0 }));
    let weights = Array1::from_iter((0..n).map(|i| 0.75 + 0.5 * ((i % 7) as f64) / 6.0));
    let design = Array2::from_shape_fn((n, 2), |(row, col)| match col {
        0 => 1.0,
        1 => z[row],
        _ => unreachable!(),
    });
    let cfg = DeviationBlockConfig {
        num_internal_knots: 4,
        ..DeviationBlockConfig::default()
    };
    let score_prepared = build_score_warp_deviation_block_from_seed(&z, &cfg)?;
    let q_seed = Array1::from_iter(z.iter().map(|zi| 0.05 + 0.2 * zi));
    let link_seed = padded_deviation_seed(&q_seed, 1.0, 0.5);
    let link_prepared =
        build_link_deviation_block_from_knots_design_seed_and_weights(&link_seed, &q_seed, &cfg)?;
    let family = BernoulliMarginalSlopeFamily {
        score_warp: Some(score_prepared.runtime.clone()),
        link_dev: Some(link_prepared.runtime.clone()),
        ..test_family_with_dense_designs(y, weights, z.clone(), design.clone(), design)
    };
    let marginal_beta = array![0.05, 0.08];
    let logslope_beta = array![-0.15, 0.04];
    let marginal_eta =
        Array1::from_iter(z.iter().map(|zi| marginal_beta[0] + marginal_beta[1] * zi));
    let logslope_eta =
        Array1::from_iter(z.iter().map(|zi| logslope_beta[0] + logslope_beta[1] * zi));
    let states = vec![
        ParameterBlockState {
            beta: marginal_beta,
            eta: marginal_eta,
        },
        ParameterBlockState {
            beta: logslope_beta,
            eta: logslope_eta,
        },
        ParameterBlockState {
            beta: Array1::zeros(score_prepared.block.design.ncols()),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_prepared.block.design.ncols()),
            eta: Array1::zeros(n),
        },
    ];
    let mut cache = family.build_exact_eval_cache(&states)?;
    cache.row_primary_hessians = family.build_row_primary_hessian_cache(&states, &cache)?;
    let direction = Array1::from_iter((0..cache.slices.total).map(|j| {
        let x = j as f64 + 1.0;
        0.03 * x.sin() + 0.01 * (0.37 * x).cos()
    }));
    Ok((family, states, cache, direction))
}

fn dual_flex_exact_fixture() -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let z = array![-0.8, 0.2, 1.1];
    let y = array![0.0, 1.0, 1.0];
    let weights = array![1.0, 0.7, 1.3];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &z,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "score warp block", e));
    let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link block", e));
    let family = BernoulliMarginalSlopeFamily {
        score_warp: Some(score_prepared.runtime.clone()),
        link_dev: Some(link_prepared.runtime.clone()),
        ..test_family_with_intercept_designs(y.clone(), weights.clone(), z.clone())
    };
    let block_states = vec![
        scalar_block_state(0.25, z.len()),
        scalar_block_state(0.6, z.len()),
        ParameterBlockState {
            beta: Array1::from_iter(
                (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
            ),
            eta: Array1::zeros(z.len()),
        },
        ParameterBlockState {
            beta: Array1::from_iter(
                (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
            ),
            eta: Array1::zeros(z.len()),
        },
    ];
    (family, block_states)
}

fn h_only_exact_fixture() -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score-warp block", e));
    let score_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("score-warp initial beta")
        .len();
    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(
            Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
            seed.len(),
        ),
    ];
    let family = BernoulliMarginalSlopeFamily {
        score_warp: Some(prepared.runtime.clone()),
        ..test_family_with_zero_primary_designs(
            array![0.0, 1.0, 0.0, 1.0, 0.0],
            Array1::ones(seed.len()),
            seed.clone(),
        )
    };
    (family, block_states)
}

fn w_only_exact_fixture() -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_test_link_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link deviation block", e));
    let link_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("link initial beta")
        .len();
    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(
            Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
            seed.len(),
        ),
    ];
    let family = BernoulliMarginalSlopeFamily {
        link_dev: Some(prepared.runtime.clone()),
        ..test_family_with_zero_primary_designs(
            array![0.0, 1.0, 0.0, 1.0, 0.0],
            Array1::ones(seed.len()),
            seed.clone(),
        )
    };
    (family, block_states)
}

fn assert_allclose_relative(actual: &Array1<f64>, expected: &Array1<f64>, tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let denom = a.abs().max(e.abs()).max(1.0);
        let rel = (a - e).abs() / denom;
        assert!(
            rel <= tol,
            "entry {idx}: actual={a:.17e}, expected={e:.17e}, rel={rel:.3e}, tol={tol:.3e}"
        );
    }
}

#[test]
fn cross_block_identifiability_anchor_wider_than_candidate_no_false_alias() {
    // Old buggy algorithm conflated `null(AᵀC) == {0}` with
    // `span(C) ⊆ span(A)`. When A has more cols than C, AᵀC is
    // n_a × n_c with n_a > n_c, so the null space of AᵀC is generically
    // {0} even though (I − P_A) C is well-rank. The new algorithm uses
    // a weighted projection (I − P_A) C and accepts directions with
    // positive eigenvalues of C̃ᵀ W C̃, so this case keeps a strictly
    // positive number of directions.
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    let n = 64usize;
    let z = Array1::from_iter((0..n).map(|i| {
        let t = (i as f64) / (n as f64 - 1.0);
        -1.5 + 3.0 * t
    }));
    let weights = Array1::from_elem(n, 1.0);
    // Build an anchor with 20 columns whose entries are a deterministic
    // pseudo-random pattern.
    let mut anchor_dense = ndarray::Array2::<f64>::zeros((n, 20));
    for i in 0..n {
        for j in 0..20 {
            let x = (i as f64 * 0.13 + j as f64 * 0.91 + 1.0).sin();
            let y = (i as f64 * 0.07 - j as f64 * 0.31 + 2.0).cos();
            anchor_dense[[i, j]] = 0.5 * x + 0.5 * y;
        }
    }
    let anchor_design = DesignMatrix::Dense(DenseDesignMatrix::from(anchor_dense.clone()));
    let link_cfg = DeviationBlockConfig {
        num_internal_knots: 2,
        ..DeviationBlockConfig::default()
    };
    let q0_seed = Array1::from_iter(z.iter().map(|zi| 0.05 + 0.4 * zi));
    let link_seed = padded_deviation_seed(&q0_seed, 1.0, 0.5);
    let mut link_prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
        &link_seed, &q0_seed, &link_cfg,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link-dev fixture", e));
    let p_before = link_prepared.runtime.basis_dim();
    assert!(p_before > 0, "fixture must have positive basis_dim");
    use super::deviation_runtime::ParametricAnchorBlock;
    install_compiled_flex_block_into_runtime(
        &mut link_prepared,
        &q0_seed,
        &link_cfg,
        &[(&anchor_design, ParametricAnchorBlock::Marginal)],
        &[],
        &weights,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "wider anchor should not false-positive full alias", e
        )
    });
    let p_after = link_prepared.runtime.basis_dim();
    assert!(
        p_after > 0,
        "new algorithm must keep strictly positive basis_dim when (I-P_A)C has rank > 0"
    );
    // Verify orthogonality of the new design (with residual applied)
    // against the anchor: AᵀW · C̃ should be at the noise floor.
    let new_design = link_prepared
        .runtime
        .design_at_training_with_residual(&q0_seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "design after orthogonalisation", e));
    assert_eq!(new_design.nrows(), n);
    assert_eq!(new_design.ncols(), p_after);
    let cross = anchor_dense.t().dot(&new_design);
    let max_abs = cross.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let anchor_norm = anchor_dense.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cand_norm = new_design.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = (anchor_norm * cand_norm).max(1.0);
    assert!(
        max_abs <= 1.0e-9 * scale,
        "Aᵀ C̃ should be at noise floor; max|.|={max_abs:.3e}, scale={scale:.3e}",
    );
}

/// Case (1) from the bug analysis: textbook minimal counterexample
/// where the old `null(AᵀC)` test silently dropped a direction it
/// should have kept. With a single-column anchor that exactly equals
/// one column of the candidate at training rows (so `AᵀC` has a single
/// nonzero entry and a zero null space), the old algorithm would
/// declare "fully aliased". The (I−P_A)C theorem keeps everything
/// except the one anchored column.
#[test]
fn cross_block_identifiability_minimal_counterexample_keeps_orthogonal_complement() {
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    let n = 64usize;
    let z = Array1::from_iter((0..n).map(|i| {
        let t = (i as f64) / (n as f64 - 1.0);
        -1.5 + 3.0 * t
    }));
    let weights = Array1::from_elem(n, 1.0);
    let link_cfg = DeviationBlockConfig {
        num_internal_knots: 3,
        ..DeviationBlockConfig::default()
    };
    let q0_seed = Array1::from_iter(z.iter().map(|zi| 0.05 + 0.4 * zi));
    let link_seed = padded_deviation_seed(&q0_seed, 1.0, 0.5);
    let mut link_prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
        &link_seed, &q0_seed, &link_cfg,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link-dev fixture", e));
    let p_before = link_prepared.runtime.basis_dim();
    assert!(
        p_before >= 2,
        "fixture must have at least two basis columns"
    );
    // Build a 1-column anchor equal to the first column of the
    // candidate's training-row design — i.e. `A = C[:, 0]`. This is
    // exactly the textbook `A = e₁`, `C = [e₁ | e₂ | …]` shape: AᵀC
    // has a single nonzero leading entry, the old `null(AᵀC)` test
    // returns ∅ (since the first column of AᵀC is nonzero), but
    // `(I − P_A) C` keeps `p_before - 1` independent directions.
    let candidate_design = link_prepared
        .runtime
        .design(&q0_seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "candidate training-row design", e));
    let mut anchor_dense = ndarray::Array2::<f64>::zeros((n, 1));
    anchor_dense
        .column_mut(0)
        .assign(&candidate_design.column(0));
    let anchor_design = DesignMatrix::Dense(DenseDesignMatrix::from(anchor_dense.clone()));
    use super::deviation_runtime::ParametricAnchorBlock;
    install_compiled_flex_block_into_runtime(
        &mut link_prepared,
        &q0_seed,
        &link_cfg,
        &[(&anchor_design, ParametricAnchorBlock::Marginal)],
        &[],
        &weights,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "minimal counterexample must keep p_before - 1 directions, not collapse to 0", e
        )
    });
    let p_after = link_prepared.runtime.basis_dim();
    assert_eq!(
        p_after,
        p_before - 1,
        "(I−P_A)C should keep exactly {} directions when one column of C is reproduced by A; got {}",
        p_before - 1,
        p_after,
    );
    let new_design = link_prepared
        .runtime
        .design_at_training_with_residual(&q0_seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "design after orthogonalisation", e));
    let cross = anchor_dense.t().dot(&new_design);
    let max_abs = cross.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let anchor_norm = anchor_dense.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cand_norm = new_design.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = (anchor_norm * cand_norm).max(1.0);
    assert!(
        max_abs <= 1.0e-9 * scale,
        "AᵀC̃ should be at noise floor after residualisation; max|.|={max_abs:.3e}, scale={scale:.3e}",
    );
}

/// Case (3) from the bug analysis: true alias `span(C) ⊆ span(A)`.
/// P5 converts the `k_kept == 0` outcome from a hard error into a
/// structured `FullyAliased { reason }` that the production caller
/// drops with a logged warning rather than aborting the fit.
///
/// The threshold anchors against ‖c_sqw‖_F² (the input candidate
/// spectrum), not against `λ_max(C̃ᵀWC̃)`: when span(C) ⊆ span(A)
/// every residualised eigenvalue sits at FP noise so
/// `λ_max_c` itself collapses to noise, and a relative-only
/// threshold would keep noise. The Frobenius² anchor scales with
/// the *input* energy and bounds noise eigenvalues from above by
/// `~ eps² · ‖C‖_F²`, well below `drop_tol = ‖C‖_F² · 64·n·eps²`.
#[test]
fn cross_block_identifiability_true_alias_returns_fully_aliased_outcome() {
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    let n = 64usize;
    let z = Array1::from_iter((0..n).map(|i| {
        let t = (i as f64) / (n as f64 - 1.0);
        -1.5 + 3.0 * t
    }));
    let weights = Array1::from_elem(n, 1.0);
    let link_cfg = DeviationBlockConfig {
        num_internal_knots: 2,
        ..DeviationBlockConfig::default()
    };
    let q0_seed = Array1::from_iter(z.iter().map(|zi| 0.05 + 0.4 * zi));
    let link_seed = padded_deviation_seed(&q0_seed, 1.0, 0.5);
    let mut link_prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
        &link_seed, &q0_seed, &link_cfg,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link-dev fixture", e));
    let candidate_design = link_prepared
        .runtime
        .design(&q0_seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "candidate training-row design", e));
    let anchor_design = DesignMatrix::Dense(DenseDesignMatrix::from(candidate_design));
    use super::deviation_runtime::ParametricAnchorBlock;
    let outcome = install_compiled_flex_block_into_runtime(
        &mut link_prepared,
        &q0_seed,
        &link_cfg,
        &[(&anchor_design, ParametricAnchorBlock::Marginal)],
        &[],
        &weights,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "true alias must produce a structured FullyAliased outcome", e
        )
    });
    match outcome {
        FlexCompileOutcome::FullyAliased { reason } => {
            assert!(
                reason.contains("zero directions remaining"),
                "expected FullyAliased reason mentioning 'zero directions remaining', got: {reason}",
            );
        }
        FlexCompileOutcome::Reparameterised => {
            panic!("expected FullyAliased outcome but got Reparameterised");
        }
    }
}

/// Flex-anchor counterpart of the parametric true-alias test. Captures
/// the invariant that flex-evaluation anchors are real participants in
/// the cross-block W-orthogonalisation: when the flex-anchor's column
/// span fully covers the candidate's column span, the outcome must be
/// `FullyAliased`, not a silent `Reparameterised`. Regression guard for
/// the historical "FlexEvaluation silently skipped" bug.
#[test]
fn cross_block_identifiability_flex_anchor_true_alias_returns_fully_aliased() {
    let n = 64usize;
    let z = Array1::from_iter((0..n).map(|i| {
        let t = (i as f64) / (n as f64 - 1.0);
        -1.5 + 3.0 * t
    }));
    let weights = Array1::from_elem(n, 1.0);
    let link_cfg = DeviationBlockConfig {
        num_internal_knots: 2,
        ..DeviationBlockConfig::default()
    };
    let q0_seed = Array1::from_iter(z.iter().map(|zi| 0.05 + 0.4 * zi));
    let link_seed = padded_deviation_seed(&q0_seed, 1.0, 0.5);
    let mut link_prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
        &link_seed, &q0_seed, &link_cfg,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link-dev fixture", e));
    // Capture the candidate's training-row design and use it as the
    // flex-evaluation anchor. span(A) = span(C) by construction, so the
    // residualised candidate has zero surviving directions.
    let flex_anchor_design = link_prepared
        .runtime
        .design(&q0_seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "candidate training-row design", e));
    let outcome = install_compiled_flex_block_into_runtime(
        &mut link_prepared,
        &q0_seed,
        &link_cfg,
        &[],
        &[&flex_anchor_design],
        &weights,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "flex-anchor true alias must produce a structured FullyAliased outcome", e
        )
    });
    match outcome {
        FlexCompileOutcome::FullyAliased { reason } => {
            assert!(
                reason.contains("zero directions remaining"),
                "expected FullyAliased reason mentioning 'zero directions remaining', got: {reason}",
            );
        }
        FlexCompileOutcome::Reparameterised => {
            panic!(
                "FlexEvaluation anchor whose span covers the candidate must yield FullyAliased; \
                     got Reparameterised, indicating the FlexEvaluation arm was silently skipped",
            );
        }
    }
}

/// Direct match-arm coverage on the `FlexCompileOutcome`
/// API. Confirms that the production code's `match outcome` against
/// `FullyAliased { reason }` extracts the reason as documented. No
/// algorithm invocation; constructs the outcome value directly.
#[test]
fn cross_block_identifiability_outcome_fully_aliased_extracts_reason() {
    let outcome = FlexCompileOutcome::FullyAliased {
        reason: "candidate has zero directions remaining after residualisation".to_string(),
    };
    match outcome {
        FlexCompileOutcome::FullyAliased { reason } => {
            assert!(reason.contains("zero directions remaining"));
        }
        FlexCompileOutcome::Reparameterised => {
            panic!("constructed FullyAliased; cannot pattern-match as Reparameterised")
        }
    }
}

/// Case (4) from the bug analysis: partial alias. Build an anchor
/// whose first k_alias columns are the leading orthonormal basis
/// vectors of span(C) (so each lies entirely inside span(C)), plus
/// one extra column entirely outside span(C). The residualisation
/// theorem must keep exactly `effective_rank(C) − k_alias` directions
/// — neither over-keeping (old `null(AᵀC)` could pass noise through)
/// nor under-keeping. Using QR-orthonormal anchor columns avoids the
/// numerical-conditioning sensitivity that would arise from inverting
/// an ill-conditioned `AᵀWA` built from raw I-spline columns.
#[test]
fn cross_block_identifiability_partial_alias_keeps_residual_rank() {
    use gam_linalg::faer_ndarray::FaerQr;
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    let n = 96usize;
    let z = Array1::from_iter((0..n).map(|i| {
        let t = (i as f64) / (n as f64 - 1.0);
        -1.5 + 3.0 * t
    }));
    let weights = Array1::from_elem(n, 1.0);
    let link_cfg = DeviationBlockConfig {
        num_internal_knots: 4,
        ..DeviationBlockConfig::default()
    };
    let q0_seed = Array1::from_iter(z.iter().map(|zi| 0.03 + 0.42 * zi));
    let link_seed = padded_deviation_seed(&q0_seed, 1.0, 0.5);
    let mut link_prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
        &link_seed, &q0_seed, &link_cfg,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link-dev fixture", e));
    let candidate_design = link_prepared
        .runtime
        .design(&q0_seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "candidate training-row design", e));
    // QR gives Q (n × p_c) orthonormal with span(Q) = span(C). The
    // candidate's effective rank at training rows equals the rank of
    // R (its leading nonzero diagonal entries). For the row counts
    // and knot configs used here, p_c is small so we just take the
    // full Q and let the algorithm itself report the surviving rank.
    let (q, _r) = candidate_design
        .qr()
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "thin QR of candidate", e));
    let p_c = candidate_design.ncols();
    // k_alias < p_c so partial (not full) alias.
    let k_alias = (p_c / 2).max(1);
    assert!(
        k_alias < p_c,
        "partial-alias test needs p_c > k_alias + 1, got p_c={p_c}, k_alias={k_alias}",
    );
    let p_a = k_alias + 1;
    let mut anchor_dense = ndarray::Array2::<f64>::zeros((n, p_a));
    for j in 0..k_alias {
        anchor_dense.column_mut(j).assign(&q.column(j));
    }
    // Extra column: deterministic high-frequency pattern unrelated to
    // span(C). Apply (I − Q Qᵀ) to it so the extra column lies
    // entirely in the orthogonal complement of span(C); this makes
    // span(A) ∩ span(C) have rank exactly k_alias by construction.
    let mut extra = Array1::<f64>::zeros(n);
    for i in 0..n {
        extra[i] = ((i as f64 * 0.17).sin() + (i as f64 * 0.41).cos()) * 0.5;
    }
    let q_t_extra = q.t().dot(&extra);
    let extra_orth = &extra - &q.dot(&q_t_extra);
    anchor_dense.column_mut(k_alias).assign(&extra_orth);
    let anchor_design = DesignMatrix::Dense(DenseDesignMatrix::from(anchor_dense.clone()));
    use super::deviation_runtime::ParametricAnchorBlock;
    let p_before = link_prepared.runtime.basis_dim();
    install_compiled_flex_block_into_runtime(
        &mut link_prepared,
        &q0_seed,
        &link_cfg,
        &[(&anchor_design, ParametricAnchorBlock::Marginal)],
        &[],
        &weights,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "partial alias must keep the surviving rank", e
        )
    });
    let p_after = link_prepared.runtime.basis_dim();
    assert_eq!(
        p_after,
        p_before - k_alias,
        "partial alias should drop exactly the {} aliased directions; got {} -> {}",
        k_alias,
        p_before,
        p_after,
    );
    let new_design = link_prepared
        .runtime
        .design_at_training_with_residual(&q0_seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "design after orthogonalisation", e));
    let cross = anchor_dense.t().dot(&new_design);
    let max_abs = cross.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let anchor_norm = anchor_dense.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cand_norm = new_design.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = (anchor_norm * cand_norm).max(1.0);
    assert!(
        max_abs <= 1.0e-9 * scale,
        "AᵀC̃ should be at noise floor; max|.|={max_abs:.3e}, scale={scale:.3e}",
    );
}

#[test]
fn flex_hessian_matvec_matches_dense_hessian() {
    let (family, states, cache, direction) = flex_hessian_matvec_fixture(96)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "flex Hv fixture", e));
    let dense = family
        .exact_newton_joint_hessian_dense_from_cache(&states, &cache)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "dense Hessian", e));
    let from_matvec = family
        .exact_newton_joint_hessian_matvec_from_cache(&direction, &states, &cache)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parallel chunked Hv", e));
    let from_dense = dense.dot(&direction);
    assert_allclose_relative(&from_matvec, &from_dense, 1.0e-13);
}

#[test]
fn row_primary_third_trace_many_matches_single_direction_contracts() {
    let (family, states, cache, direction_a) = flex_hessian_matvec_fixture(24)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "flex trace fixture", e));
    let direction_b = Array1::from_iter((0..cache.slices.total).map(|j| {
        let x = j as f64 + 0.5;
        0.02 * (0.7 * x).sin() - 0.015 * (0.31 * x).cos()
    }));
    let r = cache.primary.total;
    let gram = (0..r * r)
        .map(|idx| {
            let x = idx as f64 + 1.0;
            0.03 * x.sin() + 0.01 * (0.17 * x).cos()
        })
        .collect::<Vec<_>>();
    for row in 0..6 {
        let row_ctx = BernoulliMarginalSlopeFamily::row_ctx(&cache, row);
        let row_dirs = vec![
            family
                .row_primary_direction_from_flat(row, &cache.slices, &cache.primary, &direction_a)
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "row direction a", e)),
            family
                .row_primary_direction_from_flat(row, &cache.slices, &cache.primary, &direction_b)
                .expect("row direction b"),
        ];
        let many = family
            .row_primary_third_trace_many_with_moments(
                row, &states, &cache, row_ctx, &row_dirs, &gram,
            )
            .expect("many-direction row trace");
        for (dir_idx, row_dir) in row_dirs.iter().enumerate() {
            let third = family
                .row_primary_third_contracted(row, &states, &cache, row_ctx, row_dir)
                .unwrap_or_else(|e| {
                    panic!("{} failed: {:?}", "single-direction third contraction", e)
                });
            let single = BernoulliMarginalSlopeFamily::row_primary_trace_contract(&third, &gram);
            let denom = single.abs().max(many[dir_idx].abs()).max(1.0);
            let rel = (single - many[dir_idx]).abs() / denom;
            assert!(
                rel <= 1.0e-12,
                "row {row} dir {dir_idx}: many={:.17e} single={:.17e} rel={:.3e}",
                many[dir_idx],
                single,
                rel
            );
        }
    }
}

#[test]
fn bernoulli_margslope_warm_start_cache_persists_across_eval_cache_builds() {
    let n = 256usize;
    let z = Array1::from_iter((0..n).map(|i| {
        let t = (i as f64 + 0.5) / n as f64;
        (12.0 * t).sin() + 0.25 * (37.0 * t).cos()
    }));
    let y = Array1::from_iter((0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::ones(n);
    let cfg = DeviationBlockConfig {
        num_internal_knots: 4,
        ..DeviationBlockConfig::default()
    };
    let score_prepared = build_score_warp_deviation_block_from_seed(&z, &cfg)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "score-warp deviation block", e));
    let q_seed = Array1::from_iter(z.iter().map(|zi| 0.1 + 0.25 * zi));
    let link_seed = padded_deviation_seed(&q_seed, 1.0, 0.5);
    let link_prepared =
        build_link_deviation_block_from_knots_design_seed_and_weights(&link_seed, &q_seed, &cfg)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "link-wiggle deviation block", e));

    let cache = new_intercept_warm_start_cache(n);
    let make_family =
        |cache: Option<Arc<BernoulliInterceptWarmStartCache>>| BernoulliMarginalSlopeFamily {
            y: Arc::new(y.clone()),
            weights: Arc::new(weights.clone()),
            z: Arc::new(z.clone()),
            marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::ones((n, 1)),
            )),
            logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::ones((n, 1)),
            )),
            score_warp: Some(score_prepared.runtime.clone()),
            link_dev: Some(link_prepared.runtime.clone()),
            intercept_warm_starts: cache,
            ..default_test_family()
        };
    let marginal_eta = Array1::from_iter((0..n).map(|i| 0.15 * ((i as f64) * 0.001).sin()));
    let slope_eta = Array1::from_iter((0..n).map(|i| 0.35 + 0.02 * ((i as f64) * 0.003).cos()));
    let states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: marginal_eta,
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: slope_eta,
        },
        ParameterBlockState {
            beta: Array1::zeros(score_prepared.block.design.ncols()),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_prepared.block.design.ncols()),
            eta: Array1::zeros(n),
        },
    ];

    let warm_family = make_family(Some(Arc::clone(&cache)));
    let first = warm_family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "first warm eval cache", e));
    let nan_bits = f64::NAN.to_bits();
    for slot in cache.intercept_value.iter() {
        let bits = slot.load(Ordering::Relaxed);
        let v = f64::from_bits(bits);
        assert!(
            v.is_finite(),
            "cache slot should be populated with converged intercept after first build"
        );
        assert_ne!(bits, nan_bits);
    }

    let second = warm_family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "second warm eval cache", e));

    let cold_family = make_family(None);
    let cold = cold_family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "cold reference eval cache", e));
    for ((warm_a, warm_b), cold_ctx) in first
        .row_contexts
        .iter()
        .zip(second.row_contexts.iter())
        .zip(cold.row_contexts.iter())
    {
        assert!((warm_a.intercept - cold_ctx.intercept).abs() < 1e-9);
        assert!((warm_b.intercept - cold_ctx.intercept).abs() < 1e-9);
    }
}

#[test]
fn bernoulli_margslope_flex_ll_early_exit_is_exact_or_provably_rejected() {
    let n = 24_000usize;
    let z = Array1::from_iter((0..n).map(|i| {
        let t = (i as f64 + 0.5) / n as f64;
        (10.0 * t).sin() + 0.2 * (29.0 * t).cos()
    }));
    let y = Array1::from_iter((0..n).map(|i| if (i * 17 + 3) % 7 >= 4 { 1.0 } else { 0.0 }));
    let weights = Array1::ones(n);
    let cfg = DeviationBlockConfig {
        num_internal_knots: 4,
        ..DeviationBlockConfig::default()
    };
    let score_prepared = build_score_warp_deviation_block_from_seed(&z, &cfg)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "score-warp deviation block", e));
    let q_seed = Array1::from_iter(z.iter().map(|zi| 0.1 + 0.2 * zi));
    let link_seed = padded_deviation_seed(&q_seed, 1.0, 0.5);
    let link_prepared =
        build_link_deviation_block_from_knots_design_seed_and_weights(&link_seed, &q_seed, &cfg)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "link-wiggle deviation block", e));
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(y),
        weights: Arc::new(weights),
        z: Arc::new(z),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        score_warp: Some(score_prepared.runtime.clone()),
        link_dev: Some(link_prepared.runtime.clone()),
        ..default_test_family()
    };
    let states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: Array1::from_iter((0..n).map(|i| 0.08 * ((i as f64) * 0.002).sin())),
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: Array1::from_iter((0..n).map(|i| 0.30 + 0.03 * ((i as f64) * 0.003).cos())),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_prepared.block.design.ncols()),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_prepared.block.design.ncols()),
            eta: Array1::zeros(n),
        },
    ];

    let exact = family
        .log_likelihood_only_with_options(&states, &BlockwiseFitOptions::default())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact full FLEX ll", e));
    let mut permissive = BlockwiseFitOptions::default();
    permissive.early_exit_threshold = Some((-exact) + 1.0);
    let accepted = family
        .log_likelihood_only_with_options(&states, &permissive)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "permissive threshold should compute full FLEX ll", e
            )
        });
    let rel = ((accepted - exact) / exact.abs().max(1.0)).abs();
    assert!(
        rel < 1e-10,
        "accepted early-exit-enabled FLEX LL {accepted} differs from exact {exact} by rel {rel}"
    );

    let mut rejecting = BlockwiseFitOptions::default();
    rejecting.early_exit_threshold = Some(1e-6);
    let err = family
        .log_likelihood_only_with_options(&states, &rejecting)
        .expect_err("tight threshold should reject before the full FLEX row sweep");
    assert!(
        err.contains("line-search rejected early"),
        "unexpected early-exit error: {err}"
    );
}

#[test]
fn row_primary_fourth_contracted_rejects_bad_direction_lengths() {
    let family = test_family_with_zero_primary_designs(array![1.0], array![1.0], array![0.25]);
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.15],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.2],
        },
    ];
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let row_ctx = family
        .build_row_exact_context_with_stats_and_cell_cache(0, &block_states, None, true)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "row context", e));
    let bad_dir = array![1.0];
    let good_dir = array![0.0, 1.0];

    let err = family
        .row_primary_fourth_contracted_ordered(
            0,
            &block_states,
            &cache,
            &row_ctx,
            &bad_dir,
            &good_dir,
        )
        .expect_err("bad direction length should be rejected before indexing");

    assert!(
        err.contains("direction lengths (1,2) != 2"),
        "unexpected error: {err}"
    );
}

fn base_spec(
    y: Array1<f64>,
    weights: Array1<f64>,
    z: Array1<f64>,
) -> BernoulliMarginalSlopeTermSpec {
    let n = y.len();
    BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: bernoulli_marginal_slope_probit_link(),
        marginalspec: empty_termspec(),
        logslopespec: empty_termspec(),
        marginal_offset: Array1::zeros(n),
        logslope_offset: Array1::zeros(n),
        frailty: FrailtySpec::None,
        score_warp: None,
        link_dev: None,
        latent_z_policy: LatentZPolicy::default(),
        score_influence_jacobian: None,
    }
}

#[test]
fn bernoulli_marginal_link_map_zeroes_derivatives_on_clamped_tails() {
    let link = bernoulli_marginal_slope_probit_link();
    let lower = bernoulli_marginal_link_map(&link, -8.0)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "lower tail map", e));
    let upper = bernoulli_marginal_link_map(&link, 8.0)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "upper tail map", e));
    let lower_q = standard_normal_quantile(BERNOULLI_LINK_PROBABILITY_EPS).unwrap();
    let upper_q = standard_normal_quantile(1.0 - BERNOULLI_LINK_PROBABILITY_EPS).unwrap();

    assert_eq!(lower.mu, BERNOULLI_LINK_PROBABILITY_EPS);
    assert_eq!(upper.mu, 1.0 - BERNOULLI_LINK_PROBABILITY_EPS);
    assert!((lower.q - lower_q).abs() < 1e-12);
    assert!((upper.q - upper_q).abs() < 1e-12);
    assert_eq!([lower.mu1, lower.mu2, lower.mu3, lower.mu4], [0.0; 4]);
    assert_eq!([upper.mu1, upper.mu2, upper.mu3, upper.mu4], [0.0; 4]);
    assert_eq!([lower.q1, lower.q2, lower.q3, lower.q4], [0.0; 4]);
    assert_eq!([upper.q1, upper.q2, upper.q3, upper.q4], [0.0; 4]);
}

#[test]
fn rigid_transformed_gradient_matches_negative_log_likelihood_derivative() {
    let link = bernoulli_marginal_slope_probit_link();
    let eta = 0.25;
    let g = -0.15;
    let z = 0.7;
    let y = 1.0;
    let weight = 1.3;
    let probit_scale = 1.0;
    let objective = |eta_value: f64, g_value: f64| {
        let marginal = bernoulli_marginal_link_map(&link, eta_value).unwrap();
        rigid_standard_normal_neglog_only(marginal.q, g_value, z, y, weight, probit_scale).unwrap()
    };
    let marginal = bernoulli_marginal_link_map(&link, eta)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "marginal map", e));
    let grad = rigid_standard_normal_row_kernel(marginal, g, z, y, weight, probit_scale)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "kernel", e))
        .1;
    let step = 1e-6;
    let finite_eta = (objective(eta + step, g) - objective(eta - step, g)) / (2.0 * step);
    let finite_g = (objective(eta, g + step) - objective(eta, g - step)) / (2.0 * step);

    assert!(
        (grad[0] - finite_eta).abs() < 1e-7,
        "eta gradient {} != finite difference {}",
        grad[0],
        finite_eta
    );
    assert!(
        (grad[1] - finite_g).abs() < 1e-7,
        "g gradient {} != finite difference {}",
        grad[1],
        finite_g
    );
    assert!(
        grad[0] < 0.0,
        "y=1 probit nll should decrease as eta increases"
    );
}

struct HandRigidProbitKernel {
    logcdf: f64,
    u1: f64,
    u2: f64,
    u3: f64,
    u4: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    eta_q: f64,
    eta_g: f64,
}

impl HandRigidProbitKernel {
    #[inline]
    fn new(q: f64, g: f64, z: f64, y: f64, w: f64, probit_scale: f64) -> Result<Self, String> {
        let s = 2.0 * y - 1.0;
        let observed_logslope = rigid_observed_logslope(g, probit_scale);
        let g2 = observed_logslope * observed_logslope;
        let c = (1.0 + g2).sqrt();
        let c1 = probit_scale * observed_logslope / c;
        let c_inv3 = 1.0 / (c * c * c);
        let c_inv5 = c_inv3 / (c * c);
        let c_inv7 = c_inv5 / (c * c);
        let eta = marginal_slope_standard_normal_scalar_eta(q, g, z, probit_scale);
        let m = s * eta;
        let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(m);
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w)?;
        Ok(Self {
            logcdf,
            u1: s * k1,
            u2: k2,
            u3: s * k3,
            u4: k4,
            c1,
            c2: probit_scale * probit_scale * c_inv3,
            c3: -3.0 * probit_scale.powi(3) * observed_logslope * c_inv5,
            c4: probit_scale.powi(4) * (12.0 * g2 - 3.0) * c_inv7,
            eta_q: c,
            eta_g: q * c1 + probit_scale * z,
        })
    }

    #[inline]
    fn primary_hessian(&self, q: f64) -> [[f64; 2]; 2] {
        let h00 = self.u2 * self.eta_q * self.eta_q;
        let h01 = self.u2 * self.eta_q * self.eta_g + self.u1 * self.c1;
        let h11 = self.u2 * self.eta_g * self.eta_g + self.u1 * q * self.c2;
        [[h00, h01], [h01, h11]]
    }

    #[inline]
    fn third_contracted(&self, q: f64, dq: f64, dg: f64) -> [[f64; 2]; 2] {
        let dd = self.eta_q * dq + self.eta_g * dg;
        let dd_q = self.c1 * dg;
        let dd_g = self.c1 * dq + q * self.c2 * dg;
        let dd_qg = self.c2 * dg;
        let dd_gg = self.c2 * dq + q * self.c3 * dg;
        let t00 = self.u3 * self.eta_q * self.eta_q * dd + self.u2 * 2.0 * self.eta_q * dd_q;
        let t01 = self.u3 * self.eta_q * self.eta_g * dd
            + self.u2 * (self.c1 * dd + self.eta_q * dd_g + self.eta_g * dd_q)
            + self.u1 * dd_qg;
        let t11 = self.u3 * self.eta_g * self.eta_g * dd
            + self.u2 * (q * self.c2 * dd + 2.0 * self.eta_g * dd_g)
            + self.u1 * dd_gg;
        [[t00, t01], [t01, t11]]
    }

    #[inline]
    fn fourth_contracted(&self, q: f64, uq: f64, ug: f64, vq: f64, vg: f64) -> [[f64; 2]; 2] {
        let du = self.eta_q * uq + self.eta_g * ug;
        let dv = self.eta_q * vq + self.eta_g * vg;
        let du_a = [self.c1 * ug, self.c1 * uq + q * self.c2 * ug];
        let dv_a = [self.c1 * vg, self.c1 * vq + q * self.c2 * vg];
        let du_ab = [
            [0.0, self.c2 * ug],
            [self.c2 * ug, self.c2 * uq + q * self.c3 * ug],
        ];
        let dv_ab = [
            [0.0, self.c2 * vg],
            [self.c2 * vg, self.c2 * vq + q * self.c3 * vg],
        ];
        let dduv = self.c1 * (uq * vg + ug * vq) + q * self.c2 * ug * vg;
        let dduv_a = [
            self.c2 * ug * vg,
            self.c2 * (uq * vg + ug * vq) + q * self.c3 * ug * vg,
        ];
        let dduv_ab = [
            [0.0, self.c3 * ug * vg],
            [
                self.c3 * ug * vg,
                self.c3 * (uq * vg + ug * vq) + q * self.c4 * ug * vg,
            ],
        ];
        let eta_a = [self.eta_q, self.eta_g];
        let eta_ab = [[0.0, self.c1], [self.c1, q * self.c2]];
        let mut f = [[0.0f64; 2]; 2];
        for a in 0..2 {
            for b in a..2 {
                let val = self.u4 * eta_a[a] * eta_a[b] * du * dv
                    + self.u3
                        * (eta_ab[a][b] * du * dv
                            + du_a[a] * eta_a[b] * dv
                            + dv_a[a] * eta_a[b] * du
                            + du_a[b] * eta_a[a] * dv
                            + dv_a[b] * eta_a[a] * du
                            + dduv * eta_a[a] * eta_a[b])
                    + self.u2
                        * (eta_ab[a][b] * dduv
                            + du_a[a] * dv_a[b]
                            + dv_a[a] * du_a[b]
                            + du_ab[a][b] * dv
                            + dv_ab[a][b] * du
                            + eta_a[b] * dduv_a[a]
                            + eta_a[a] * dduv_a[b])
                    + self.u1 * dduv_ab[a][b];
                f[a][b] = val;
                f[b][a] = val;
            }
        }
        f
    }
}

#[inline]
fn hand_rigid_transformed_gradient(
    marginal: BernoulliMarginalLinkMap,
    kernel: &HandRigidProbitKernel,
) -> [f64; 2] {
    [
        kernel.u1 * kernel.eta_q * marginal.q1,
        kernel.u1 * kernel.eta_g,
    ]
}

#[inline]
fn hand_rigid_transformed_hessian(
    marginal: BernoulliMarginalLinkMap,
    kernel: &HandRigidProbitKernel,
) -> [[f64; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    let grad_q = kernel.u1 * kernel.eta_q;
    [
        [
            h_q[0][0] * marginal.q1 * marginal.q1 + grad_q * marginal.q2,
            h_q[0][1] * marginal.q1,
        ],
        [h_q[1][0] * marginal.q1, h_q[1][1]],
    ]
}

#[inline]
fn hand_rigid_internal_third_components(
    marginal: BernoulliMarginalLinkMap,
    kernel: &HandRigidProbitKernel,
) -> (f64, f64, f64, f64) {
    let q_dir = kernel.third_contracted(marginal.q, 1.0, 0.0);
    let g_dir = kernel.third_contracted(marginal.q, 0.0, 1.0);
    (q_dir[0][0], q_dir[0][1], q_dir[1][1], g_dir[1][1])
}

#[inline]
fn hand_rigid_transformed_third_full(
    marginal: BernoulliMarginalLinkMap,
    kernel: &HandRigidProbitKernel,
) -> [[[f64; 2]; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    let grad_q = kernel.u1 * kernel.eta_q;
    let (f_qqq, f_qqg, f_qgg, f_ggg) = hand_rigid_internal_third_components(marginal, kernel);
    let q1_sq = marginal.q1 * marginal.q1;
    let q1_cu = q1_sq * marginal.q1;
    let f_etaetaeta =
        f_qqq * q1_cu + 3.0 * h_q[0][0] * marginal.q1 * marginal.q2 + grad_q * marginal.q3;
    let f_etaetag = f_qqg * q1_sq + h_q[0][1] * marginal.q2;
    let f_etagg = f_qgg * marginal.q1;
    hand_third_full_from_symmetric_components(f_etaetaeta, f_etaetag, f_etagg, f_ggg)
}

#[inline]
fn hand_third_full_from_symmetric_components(
    t_qqq: f64,
    t_qqg: f64,
    t_qgg: f64,
    t_ggg: f64,
) -> [[[f64; 2]; 2]; 2] {
    let mut t = [[[0.0; 2]; 2]; 2];
    t[0][0][0] = t_qqq;
    t[0][0][1] = t_qqg;
    t[0][1][0] = t_qqg;
    t[1][0][0] = t_qqg;
    t[0][1][1] = t_qgg;
    t[1][0][1] = t_qgg;
    t[1][1][0] = t_qgg;
    t[1][1][1] = t_ggg;
    t
}

#[inline]
fn hand_rigid_transformed_fourth_full(
    marginal: BernoulliMarginalLinkMap,
    kernel: &HandRigidProbitKernel,
) -> [[[[f64; 2]; 2]; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    let grad_q = kernel.u1 * kernel.eta_q;
    let (f_qqq, f_qqg, f_qgg, _) = hand_rigid_internal_third_components(marginal, kernel);
    let qq = kernel.fourth_contracted(marginal.q, 1.0, 0.0, 1.0, 0.0);
    let qg = kernel.fourth_contracted(marginal.q, 1.0, 0.0, 0.0, 1.0);
    let gg = kernel.fourth_contracted(marginal.q, 0.0, 1.0, 0.0, 1.0);
    let f_qqqq = qq[0][0];
    let f_qqqg = qq[0][1];
    let f_qqgg = qq[1][1];
    let f_qggg = qg[1][1];
    let f_gggg = gg[1][1];
    let q1_sq = marginal.q1 * marginal.q1;
    let q1_cu = q1_sq * marginal.q1;
    let q1_q = q1_sq * q1_sq;
    let f_eta4 = f_qqqq * q1_q
        + 6.0 * f_qqq * q1_sq * marginal.q2
        + 3.0 * h_q[0][0] * marginal.q2 * marginal.q2
        + 4.0 * h_q[0][0] * marginal.q1 * marginal.q3
        + grad_q * marginal.q4;
    let f_eta3g =
        f_qqqg * q1_cu + 3.0 * f_qqg * marginal.q1 * marginal.q2 + h_q[0][1] * marginal.q3;
    let f_eta2g2 = f_qqgg * q1_sq + f_qgg * marginal.q2;
    let f_etag3 = f_qggg * marginal.q1;
    hand_fourth_full_from_symmetric_components(f_eta4, f_eta3g, f_eta2g2, f_etag3, f_gggg)
}

#[inline]
fn hand_fourth_full_from_symmetric_components(
    t_qqqq: f64,
    t_qqqg: f64,
    t_qqgg: f64,
    t_qggg: f64,
    t_gggg: f64,
) -> [[[[f64; 2]; 2]; 2]; 2] {
    let mut t = [[[[0.0; 2]; 2]; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                for d in 0..2 {
                    let g_count = a + b + c + d;
                    t[a][b][c][d] = match g_count {
                        0 => t_qqqq,
                        1 => t_qqqg,
                        2 => t_qqgg,
                        3 => t_qggg,
                        _ => t_gggg,
                    };
                }
            }
        }
    }
    t
}

fn hand_rigid_standard_normal_row_kernel(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
    let kernel = HandRigidProbitKernel::new(marginal.q, g, z, y, w, probit_scale)?;
    Ok((
        -w * kernel.logcdf,
        hand_rigid_transformed_gradient(marginal, &kernel),
        hand_rigid_transformed_hessian(marginal, &kernel),
    ))
}

fn hand_rigid_standard_normal_third_full(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<[[[f64; 2]; 2]; 2], String> {
    let kernel = HandRigidProbitKernel::new(marginal.q, g, z, y, w, probit_scale)?;
    Ok(hand_rigid_transformed_third_full(marginal, &kernel))
}

fn hand_rigid_standard_normal_fourth_full(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<[[[[f64; 2]; 2]; 2]; 2], String> {
    let kernel = HandRigidProbitKernel::new(marginal.q, g, z, y, w, probit_scale)?;
    Ok(hand_rigid_transformed_fourth_full(marginal, &kernel))
}

fn assert_scalar_close(actual: f64, expected: f64, tol: f64, context: &str) {
    let denom = actual.abs().max(expected.abs()).max(1.0);
    let rel = (actual - expected).abs() / denom;
    assert!(
        rel <= tol,
        "{context}: actual={actual:.17e}, expected={expected:.17e}, rel={rel:.3e}, tol={tol:.3e}"
    );
}

#[test]
fn rigid_standard_normal_tower_path_matches_hand_chain_witness() {
    let link = bernoulli_marginal_slope_probit_link();
    let eta_grid: [f64; 9] = [-8.0, -6.5, -2.0, -0.4, 0.0, 0.75, 2.25, 6.0, 8.0];
    let g_grid = [-1.4, -0.55, 0.0, 0.8, 1.7];
    let z_grid = [-2.25, -0.35, 0.4, 2.1];
    // Per-regime bounds (same construction as the CTN endpoint oracle): the
    // 3rd/4th-order chains are ill-conditioned EXPRESSIONS in the saturation
    // tail — at |eta| >= 6 the signed argument reaches |m| ~ 9-13 where the
    // Mills-ratio derivative stack has kappa ~ 1e7-1e9, so two exact paths
    // with different contraction orders (hand chain vs tower; arm64 vs
    // x86_64 FMA) can only agree to ~kappa*eps. The single worst-conditioned
    // interior component is the mixed `fourth[q,q,q,g]` (= f_eta3g) entry,
    // whose Faa-di-Bruno chain carries a q1^3 Mills-ratio-cubed factor; on
    // some FMA/contraction-order combinations it reassociates up to ~8.5e-10
    // (observed with compensated-FMA GEMV kernels in 92d25e154 — pure roundoff,
    // the value/gradient/Hessian/third channels all stay pinned tighter). Far
    // x86_64 tails (|eta|>=6) reach ~1.2e-7 due to extreme kappa. Both regimes
    // remain >=3 orders of magnitude below the >=1e-6 dropped-term signal this
    // witness exists to catch, so these bounds cover the worst-conditioned exact
    // reassociation with headroom without ever masking a missing/incorrect term.
    let interior_tol = 2.0e-9;
    let tail_tol = 5.0e-7;

    for eta in eta_grid {
        let tol = if eta.abs() >= 6.0 {
            tail_tol
        } else {
            interior_tol
        };
        let marginal = bernoulli_marginal_link_map(&link, eta)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "marginal map", e));
        for g in g_grid {
            for z in z_grid {
                for y in [0.0, 1.0] {
                    for weight in [1.0, 1.3] {
                        for probit_scale in [1.0, 0.7] {
                            let production = rigid_standard_normal_row_kernel(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| {
                                panic!("{} failed: {:?}", "production row kernel", e)
                            });
                            let hand = hand_rigid_standard_normal_row_kernel(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| panic!("{} failed: {:?}", "hand row kernel", e));
                            assert_scalar_close(
                                production.0,
                                hand.0,
                                tol,
                                &format!(
                                    "value eta={eta} g={g} z={z} y={y} w={weight} scale={probit_scale}"
                                ),
                            );
                            for i in 0..2 {
                                assert_scalar_close(
                                    production.1[i],
                                    hand.1[i],
                                    tol,
                                    &format!(
                                        "gradient[{i}] eta={eta} g={g} z={z} y={y} w={weight} scale={probit_scale}"
                                    ),
                                );
                                for j in 0..2 {
                                    assert_scalar_close(
                                        production.2[i][j],
                                        hand.2[i][j],
                                        tol,
                                        &format!(
                                            "hessian[{i},{j}] eta={eta} g={g} z={z} y={y} w={weight} scale={probit_scale}"
                                        ),
                                    );
                                }
                            }

                            let production_third = rigid_standard_normal_third_full(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| panic!("{} failed: {:?}", "production third", e));
                            let hand_third = hand_rigid_standard_normal_third_full(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| panic!("{} failed: {:?}", "hand third", e));
                            for a in 0..2 {
                                for b in 0..2 {
                                    for c in 0..2 {
                                        assert_scalar_close(
                                            production_third[a][b][c],
                                            hand_third[a][b][c],
                                            tol,
                                            &format!(
                                                "third[{a},{b},{c}] eta={eta} g={g} z={z} y={y} w={weight} scale={probit_scale}"
                                            ),
                                        );
                                    }
                                }
                            }

                            let production_fourth = rigid_standard_normal_fourth_full(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| panic!("{} failed: {:?}", "production fourth", e));
                            let hand_fourth = hand_rigid_standard_normal_fourth_full(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| panic!("{} failed: {:?}", "hand fourth", e));
                            for a in 0..2 {
                                for b in 0..2 {
                                    for c in 0..2 {
                                        for d in 0..2 {
                                            assert_scalar_close(
                                                production_fourth[a][b][c][d],
                                                hand_fourth[a][b][c][d],
                                                tol,
                                                &format!(
                                                    "fourth[{a},{b},{c},{d}] eta={eta} g={g} z={z} y={y} w={weight} scale={probit_scale}"
                                                ),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// #932 single-sourcing contract for the rigid standard-normal RowKernel<2>:
/// the production uncontracted third- AND fourth-order primary tensors must be
/// the `.t3`/`.t4` channels of the SAME `Tower4<2>` row jet that delivers value /
/// gradient / Hessian — i.e. the entire derivative tower is mechanically derived
/// from one row expression, with no parallel hand-written chain-rule engine.
///
/// Before the cutover the fourth tensor was assembled by a separate
/// `RigidStandardNormalChain` (hand Faà-di-Bruno q-chain) that could silently
/// drift from the tower (the #736/#947 desync genus). This test pins
/// `rigid_standard_normal_fourth_full == tower.t4` (and the already-cutover
/// `rigid_standard_normal_third_full == tower.t3`) to bit identity, so any
/// reintroduction of a divergent non-tower fourth-order path fails here. The
/// independent-witness arm (`rigid_standard_normal_tower_path_matches_hand_chain_witness`)
/// then certifies the tower channels themselves against a fully separate
/// derivation; together they give prod == tower == hand-witness.
#[test]
fn rigid_standard_normal_third_fourth_full_are_single_sourced_from_tower_932() {
    let link = bernoulli_marginal_slope_probit_link();
    let eta_grid: [f64; 7] = [-6.0, -2.0, -0.4, 0.0, 0.75, 2.25, 6.0];
    let g_grid = [-1.4, -0.55, 0.0, 0.8, 1.7];
    let z_grid = [-2.25, -0.35, 0.4, 2.1];
    for eta in eta_grid {
        let marginal = bernoulli_marginal_link_map(&link, eta)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "marginal map", e));
        for g in g_grid {
            for z in z_grid {
                for y in [0.0, 1.0] {
                    for weight in [1.0, 1.3] {
                        for probit_scale in [1.0, 0.7] {
                            let tower = rigid_standard_normal_tower(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| panic!("{} failed: {:?}", "tower", e));
                            let third = rigid_standard_normal_third_full(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| panic!("{} failed: {:?}", "third", e));
                            let fourth = rigid_standard_normal_fourth_full(
                                marginal,
                                g,
                                z,
                                y,
                                weight,
                                probit_scale,
                            )
                            .unwrap_or_else(|e| panic!("{} failed: {:?}", "fourth", e));
                            for a in 0..2 {
                                for b in 0..2 {
                                    for c in 0..2 {
                                        assert_eq!(
                                            third[a][b][c], tower.t3[a][b][c],
                                            "third[{a}][{b}][{c}] not single-sourced from tower.t3 \
                                             (eta={eta} g={g} z={z} y={y} w={weight} scale={probit_scale})"
                                        );
                                        for d in 0..2 {
                                            assert_eq!(
                                                fourth[a][b][c][d], tower.t4[a][b][c][d],
                                                "fourth[{a}][{b}][{c}][{d}] not single-sourced from \
                                                 tower.t4 (eta={eta} g={g} z={z} y={y} w={weight} \
                                                 scale={probit_scale})"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn pair_distance(lhs: (f64, f64), rhs: (f64, f64)) -> f64 {
    (lhs.0 - rhs.0).abs() + (lhs.1 - rhs.1).abs()
}

/// Build a tiny synthetic rigid-probit family with `n` rows. The marginal
/// and log-slope blocks each carry a single all-ones column so the
/// per-row eta is simply the scalar block beta. No flex deviations are
/// active, so `log_likelihood_only` takes the closed-form rigid path.
fn make_rigid_test_family(n: usize) -> BernoulliMarginalSlopeFamily {
    // Pseudo-random labels and weights uncorrelated with row parity, so a
    // half-data subsample over even rows is a representative subsample
    // for the Horvitz-Thompson rescaling check below.
    let y: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 7) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.5 + 3.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
    );
    let ones_col = Array2::from_shape_fn((n, 1), |_| 1.0);
    test_family_with_dense_designs(y, weights, z, ones_col.clone(), ones_col)
}

fn rigid_block_states(
    family: &BernoulliMarginalSlopeFamily,
    q: f64,
    b: f64,
) -> Vec<ParameterBlockState> {
    let n = family.y.len();
    vec![scalar_block_state(q, n), scalar_block_state(b, n)]
}

#[test]
fn bernoulli_log_likelihood_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_rigid_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);

    let baseline = family
        .log_likelihood_only(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline ll (no subsample)", e));

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full_mask = family
        .log_likelihood_only_with_options(&states, &opts_full)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll with mask=full", e));

    let rel = ((with_full_mask - baseline) / baseline.abs().max(1.0)).abs();
    assert!(
        rel < 1e-12,
        "subsample(mask=full) {} differs from baseline {} by rel {}",
        with_full_mask,
        baseline,
        rel
    );
}

#[test]
fn bernoulli_log_likelihood_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_rigid_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);

    // Even rows only: weight_scale = n / (n/2) = 2.0
    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();
    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .log_likelihood_only_with_options(&states, &opts_half)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll with mask=even", e));

    // Compute the unscaled even-row sum directly via a full-mask
    // call on a reduced family (just check identity 2 * Σ_even ≈ scaled).
    // Construct an "even-only" full call by building a custom mask
    // covering the same rows but with weight_scale = 1.0.
    let mut opts_even_unscaled = BlockwiseFitOptions::default();
    opts_even_unscaled.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::with_uniform_weight(even_mask, m, 0, 1.0),
    ));
    let raw_even_sum = family
        .log_likelihood_only_with_options(&states, &opts_even_unscaled)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "raw even-row ll sum", e));

    let expected_scaled = (n as f64 / m as f64) * raw_even_sum;
    let rel = ((scaled - expected_scaled) / expected_scaled.abs().max(1.0)).abs();
    assert!(
        rel < 1e-12,
        "scaled {} != 2*even_sum {} (rel {})",
        scaled,
        expected_scaled,
        rel
    );

    // Horvitz-Thompson: 2 * Σ_even should approximate the full-data sum.
    // For a smooth integrand on a moderately fine grid, ~5% relative
    // accuracy is plenty.
    let baseline = family
        .log_likelihood_only(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline ll", e));
    let ht_rel = ((scaled - baseline) / baseline.abs().max(1.0)).abs();
    assert!(
        ht_rel < 0.05,
        "Horvitz-Thompson scaled {} not near baseline {} (rel {})",
        scaled,
        baseline,
        ht_rel
    );
}

/// Same shape as `make_rigid_test_family` but with a non-zero
/// gaussian_frailty_sd so the sigma-aware joint psi paths fire.
fn make_sigma_aware_test_family(n: usize) -> BernoulliMarginalSlopeFamily {
    let mut family = make_rigid_test_family(n);
    family.gaussian_frailty_sd = Some(0.7);
    family
}

fn rel_diff_array1(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut max = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs() / b[i].abs().max(1.0);
        if d > max {
            max = d;
        }
    }
    max
}

fn rel_diff_array2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let mut max = 0.0f64;
    for ((i, j), &av) in a.indexed_iter() {
        let bv = b[[i, j]];
        let d = (av - bv).abs() / bv.abs().max(1.0);
        if d > max {
            max = d;
        }
    }
    max
}

#[test]
fn bernoulli_sigma_psi_terms_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);
    let specs = vec![dummy_blockspec(1, n), dummy_blockspec(1, n)];

    let baseline = family
        .sigma_exact_joint_psi_terms(&states, &specs)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline psi terms", e))
        .expect("baseline some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_full)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "psi terms with full mask", e))
        .expect("some");

    let obj_rel = ((with_full.objective_psi - baseline.objective_psi)
        / baseline.objective_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let score_rel = rel_diff_array1(&with_full.score_psi, &baseline.score_psi);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
    let h_full = with_full
        .hessian_psi_operator
        .as_ref()
        .expect("op")
        .to_dense();
    let h_baseline = baseline
        .hessian_psi_operator
        .as_ref()
        .expect("op")
        .to_dense();
    let h_rel = rel_diff_array2(&h_full, &h_baseline);
    assert!(h_rel < 1e-12, "hessian rel {}", h_rel);
}

#[test]
fn bernoulli_sigma_psi_terms_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);
    let specs = vec![dummy_blockspec(1, n), dummy_blockspec(1, n)];

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_half)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "scaled", e))
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_raw)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "raw", e))
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi;
    let obj_rel = ((scaled.objective_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let exp_score = &raw.score_psi * factor;
    let score_rel = rel_diff_array1(&scaled.score_psi, &exp_score);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
    let h_scaled = scaled
        .hessian_psi_operator
        .as_ref()
        .expect("op failed")
        .to_dense();
    let h_raw = raw
        .hessian_psi_operator
        .as_ref()
        .expect("op failed")
        .to_dense();
    let h_exp = &h_raw * factor;
    let h_rel = rel_diff_array2(&h_scaled, &h_exp);
    assert!(h_rel < 1e-12, "hessian rel {}", h_rel);
}

#[test]
fn bernoulli_sigma_psi_second_order_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);

    let baseline = family
        .sigma_exact_joint_psisecond_order_terms(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline", e))
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_full)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "with full mask", e))
        .expect("some");

    let obj_rel = ((with_full.objective_psi_psi - baseline.objective_psi_psi)
        / baseline.objective_psi_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let score_rel = rel_diff_array1(&with_full.score_psi_psi, &baseline.score_psi_psi);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn bernoulli_sigma_psi_second_order_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_half)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "scaled", e))
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_raw)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "raw", e))
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi_psi;
    let obj_rel = ((scaled.objective_psi_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let exp_score = &raw.score_psi_psi * factor;
    let score_rel = rel_diff_array1(&scaled.score_psi_psi, &exp_score);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn bernoulli_sigma_psihessian_directional_derivative_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);
    let dir = array![0.1, -0.2];

    let baseline = family
        .sigma_exact_joint_psihessian_directional_derivative(&states, &dir)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline", e))
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(&states, &dir, &opts_full)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "with full", e))
        .expect("some");

    let rel = rel_diff_array2(&with_full, &baseline);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn bernoulli_sigma_psihessian_directional_derivative_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);
    let dir = array![0.1, -0.2];

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(&states, &dir, &opts_half)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "scaled", e))
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(&states, &dir, &opts_raw)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "raw", e))
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp = &raw * factor;
    let rel = rel_diff_array2(&scaled, &exp);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn bernoulli_psi_workspace_with_options_threads_subsample_to_first_order() {
    use crate::custom_family::CustomFamilyBlockPsiDerivative;
    use crate::outer_subsample::OuterScoreSubsample;

    // Build a sigma-aware family at n=200 with the sigma-aux derivative
    // entry on the logslope block (last block).
    let n = 200usize;
    let family = make_sigma_aware_test_family(n);
    let states = rigid_block_states(&family, 0.3, 0.4);
    let specs = vec![dummy_blockspec(1, n), dummy_blockspec(1, n)];
    let derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>> = vec![
        Vec::new(),
        vec![CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            None,
            None,
            None,
            None,
        )],
    ];
    let hyper_layout = rigid_test_design_hyper_layout(&derivative_blocks);

    // Build a half-mask of even rows (factor = 2.0).
    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();
    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xBEEF_CAFE),
    ));

    // Reference: call the family-level subsample-aware sigma path directly.
    let direct = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_half)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "direct sigma terms with options", e))
        .expect("direct some");

    // Workspace path: build via the new outer-aware trait method and call
    // first_order_terms at the sigma-aux psi index. The subsample must
    // arrive intact through the workspace boundary.
    let ws = family
        .exact_newton_joint_psi_workspace_with_options(
            &states,
            &specs,
            &hyper_layout,
            &opts_half,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "workspace with options", e))
        .expect("workspace some");
    let psi_total: usize = derivative_blocks.iter().map(Vec::len).sum();
    let sigma_psi = psi_total - 1;
    let via_ws = ws
        .first_order_terms(sigma_psi)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "ws first_order_terms", e))
        .expect("some");

    // Bit-for-bit equality is the right contract here: both paths take the
    // same masked rows, the same Horvitz-Thompson rescaling, and the same
    // fixed RNG seed-derived weight scale.
    assert_eq!(via_ws.objective_psi, direct.objective_psi);
    let score_rel = rel_diff_array1(&via_ws.score_psi, &direct.score_psi);
    assert!(score_rel == 0.0, "score_psi diverged: rel {}", score_rel);
    let h_ws = via_ws
        .hessian_psi_operator
        .as_ref()
        .expect("ws hessian op")
        .to_dense();
    let h_direct = direct
        .hessian_psi_operator
        .as_ref()
        .expect("direct hessian op")
        .to_dense();
    let h_rel = rel_diff_array2(&h_ws, &h_direct);
    assert!(h_rel == 0.0, "hessian diverged: rel {}", h_rel);

    // Sanity: confirm the half-mask path is actually scaled relative to
    // the unsampled (full-data) workspace, so we know the subsample took
    // effect rather than silently being ignored.
    let ws_full = family
        .exact_newton_joint_psi_workspace_with_options(
            &states,
            &specs,
            &hyper_layout,
            &BlockwiseFitOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "workspace full", e))
        .expect("some");
    let via_ws_full = ws_full
        .first_order_terms(sigma_psi)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "ws_full first_order_terms", e))
        .expect("some");
    // The half-mask subsample should give a different (rescaled) value
    // than the full-data path; if it matched bit-for-bit, the subsample
    // never made it through.
    assert!(
        (via_ws.objective_psi - via_ws_full.objective_psi).abs() > 1e-9,
        "subsample objective {} too close to full-data {} (subsample not threaded?)",
        via_ws.objective_psi,
        via_ws_full.objective_psi
    );
    let m_f = m as f64;
    let n_f = n as f64;
    // The rescaling magnitude should be of order n/m relative to half-mask
    // raw row sum; we simply require the subsample-aware result to be
    // within a factor of (n/m)*2 of the full-data result, as a coarse
    // sanity check that we haven't accidentally double-scaled.
    let ratio_bound = (n_f / m_f) * 2.0 + 1.0;
    let ratio = (via_ws.objective_psi.abs() + 1.0) / (via_ws_full.objective_psi.abs() + 1.0);
    assert!(
        ratio < ratio_bound && (1.0 / ratio) < ratio_bound,
        "subsample/full ratio {} outside coarse bound {}",
        ratio,
        ratio_bound
    );
}

fn build_test_link_deviation_block_from_seed(
    seed: &Array1<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<DeviationPrepared, String> {
    build_link_deviation_block_from_knots_design_seed_and_weights(seed, seed, cfg)
}

#[test]
fn score_warp_basis_smoothness_penalty_is_full_rank() {
    // Replaces the old data-distribution moment-anchor test. After the
    // smoothness-null-space drop (β-independent identifiability), the
    // integrated derivative penalty at the configured max derivative
    // order must be full rank on the transformed basis — the
    // {constants, linears, ...} null space is structurally absent
    // from the parameterization. This is the property that makes
    // joint H+S well-conditioned during PIRLS regardless of how β
    // shifts the linear-predictor distribution.
    let seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let cfg = DeviationBlockConfig {
        num_internal_knots: 5,
        ..DeviationBlockConfig::default()
    };
    let prepared = build_score_warp_deviation_block_from_seed(&seed, &cfg).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "build smoothness-null-space-drop score-warp", e
        )
    });
    let max_order = cfg
        .penalty_orders
        .iter()
        .copied()
        .max()
        .or(Some(cfg.penalty_order))
        .filter(|o| *o > 0)
        .expect("test config has a positive penalty order");
    let (penalty, nullity) = prepared
        .runtime
        .integrated_derivative_penalty_with_nullity(max_order)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "integrated penalty on transformed basis", e
            )
        });
    assert_eq!(
        nullity, 0,
        "smoothness-null-space-drop basis must have zero nullity at the max configured \
             derivative order; got {nullity}"
    );
    // Cross-check via eigendecomposition that all eigenvalues exceed the
    // numerical positive-eigenvalue threshold — confirms full rank
    // beyond the rank-reporting heuristic.
    use gam_linalg::faer_ndarray::FaerEigh;
    let (evals, _) = penalty.eigh(faer::Side::Lower).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "eigendecomposition of transformed-basis penalty", e
        )
    });
    let evals_slice = evals
        .as_slice()
        .expect("contiguous transformed-basis penalty eigenvalues");
    let threshold =
        gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(evals_slice);
    let smallest = evals_slice.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        smallest > threshold,
        "smallest eigenvalue {smallest} of transformed-basis penalty must exceed positive \
             threshold {threshold} — null space was not fully dropped"
    );
}

#[test]
fn link_deviation_basis_smoothness_penalty_is_full_rank() {
    // Mirror of `score_warp_basis_smoothness_penalty_is_full_rank` for
    // the link-deviation entry point. Same property, same proof: full
    // rank of the integrated derivative penalty on the transformed
    // basis.
    let q = array![-2.0, -0.8, -0.1, 0.4, 1.3, 2.1];
    let cfg = DeviationBlockConfig {
        num_internal_knots: 5,
        ..DeviationBlockConfig::default()
    };
    let prepared = build_link_deviation_block_from_knots_design_seed_and_weights(&q, &q, &cfg)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "build smoothness-null-space-drop link-deviation", e
            )
        });
    let max_order = cfg
        .penalty_orders
        .iter()
        .copied()
        .max()
        .or(Some(cfg.penalty_order))
        .filter(|o| *o > 0)
        .expect("test config has a positive penalty order");
    let (penalty, nullity) = prepared
        .runtime
        .integrated_derivative_penalty_with_nullity(max_order)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "integrated penalty on transformed basis", e
            )
        });
    assert_eq!(
        nullity, 0,
        "smoothness-null-space-drop basis must have zero nullity at the max configured \
             derivative order; got {nullity}"
    );
    use gam_linalg::faer_ndarray::FaerEigh;
    let (evals, _) = penalty.eigh(faer::Side::Lower).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "eigendecomposition of transformed-basis penalty", e
        )
    });
    let evals_slice = evals
        .as_slice()
        .expect("contiguous transformed-basis penalty eigenvalues");
    let threshold =
        gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(evals_slice);
    let smallest = evals_slice.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        smallest > threshold,
        "smallest eigenvalue {smallest} of transformed-basis penalty must exceed positive \
             threshold {threshold} — null space was not fully dropped"
    );
}

#[test]
fn bernoulli_marginal_slope_rejects_nonprobit_base_link() {
    let y = array![0.0, 1.0];
    let weights = array![1.0, 1.0];
    let z = array![-0.4, 0.9];
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
        [1.0],
        [1.0]
    ]));
    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: InverseLink::Standard(StandardLink::Logit),
        marginalspec: empty_termspec(),
        logslopespec: empty_termspec(),
        marginal_offset: Array1::zeros(2),
        logslope_offset: Array1::zeros(2),
        frailty: FrailtySpec::None,
        score_warp: None,
        link_dev: None,
        latent_z_policy: LatentZPolicy::default(),
        score_influence_jacobian: None,
    };
    let err = validate_spec(design.to_dense().view(), &spec)
        .expect_err("non-probit marginal-slope link should be rejected");
    assert!(err.contains("requires link(type=probit)"));
    let err = bernoulli_marginal_slope_eta_from_probability(
        &InverseLink::Standard(StandardLink::Logit),
        0.5,
        "test logit inverse",
    )
    .expect_err("non-probit marginal-slope inverse should be rejected");
    assert!(err.contains("requires link(type=probit)"));
}

fn expand_integer_weight_rows(
    y: &Array1<f64>,
    z: &Array1<f64>,
    weights: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let mut y_expanded = Vec::new();
    let mut z_expanded = Vec::new();
    for i in 0..y.len() {
        let reps = weights[i] as usize;
        assert!(
            (weights[i] - reps as f64).abs() < 1e-12,
            "test helper expects integer weights, got {}",
            weights[i]
        );
        for _ in 0..reps {
            y_expanded.push(y[i]);
            z_expanded.push(z[i]);
        }
    }
    (Array1::from_vec(y_expanded), Array1::from_vec(z_expanded))
}

#[test]
fn link_dev_without_score_warp_exposes_structural_derivative_lower_bounds() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_test_link_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link deviation block", e));
    let link_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("link block initial beta")
        .len();
    let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.1 * (idx as f64 + 1.0)));
    let family = BernoulliMarginalSlopeFamily {
        link_dev: Some(prepared.runtime.clone()),
        ..test_family_with_zero_primary_designs(
            Array1::zeros(seed.len()),
            Array1::ones(seed.len()),
            seed.clone(),
        )
    };
    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(beta_link.clone(), seed.len()),
    ];

    let slices = block_slices(&family);
    assert!(slices.h.is_none(), "score-warp slice should be absent");
    let link_slice = slices.w.as_ref().expect("link slice");
    assert_eq!(
        slices.marginal.len(),
        0,
        "zero-column marginal design should not contribute coefficient coordinates"
    );
    assert_eq!(
        slices.logslope.len(),
        0,
        "zero-column logslope design should not contribute coefficient coordinates"
    );
    assert_eq!(
        link_slice.start, 0,
        "link-only coefficients should start at 0"
    );
    assert_eq!(link_slice.len(), link_dim);

    let primary = primary_slices(&slices);
    assert!(primary.h.is_none(), "primary h slice should be absent");
    let primary_w = primary.w.as_ref().expect("primary link slice");
    assert_eq!(primary_w.start, 2, "primary link slice should start at 2");
    assert_eq!(primary.total, 2 + link_dim);

    // Verify that the analytic IFT path produces finite gradient/Hessian
    // for a link-dev-only family.
    family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "eval cache", e));
    let row_ctx = family
        .build_row_exact_context_with_stats_and_cell_cache(0, &block_states, None, true)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "row context", e));
    let (nll, grad, hess) = family
        .compute_row_primary_gradient_hessian(0, &block_states, &primary, &row_ctx)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "analytic flex eval", e));
    assert!(nll.is_finite(), "neglog should be finite for link-dev-only");
    assert!(
        grad.iter().all(|v| v.is_finite()),
        "gradient should be finite"
    );
    assert!(
        hess.iter().all(|v| v.is_finite()),
        "Hessian should be finite"
    );

    let dummy_spec = dummy_blockspec(link_dim, seed.len());
    assert!(
        family
            .block_linear_constraints(&block_states, 1, &dummy_spec)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "non-link constraint lookup", e))
            .is_none(),
        "non-link block should not expose auxiliary monotonicity constraints"
    );
    let constraints = family
        .block_linear_constraints(&block_states, 2, &dummy_spec)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "link constraint lookup", e))
        .expect("link constraints");
    assert_eq!(constraints.a.ncols(), link_dim);
    assert_eq!(constraints.b.len(), constraints.a.nrows());
    assert!(
        constraints.a.nrows() >= link_dim,
        "anchored link constraints should be expressed in raw derivative-control rows"
    );
    assert_eq!(
        constraints.b,
        Array1::<f64>::from_elem(
            constraints.a.nrows(),
            prepared.runtime.monotonicity_eps() - 1.0
        )
    );
}

#[test]
fn zero_deviation_intercept_fast_path_matches_denested_calibration() {
    let seed = Array1::from_iter((0..25).map(|i| -2.4 + 4.8 * i as f64 / 24.0));
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 8,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score-warp block", e));
    let link_prepared = build_test_link_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 8,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link-deviation block", e));
    let n = seed.len();
    let family = BernoulliMarginalSlopeFamily {
        gaussian_frailty_sd: Some(0.65),
        score_warp: Some(score_prepared.runtime.clone()),
        link_dev: Some(link_prepared.runtime.clone()),
        intercept_warm_starts: Some(new_intercept_warm_start_cache(n)),
        ..test_family_with_zero_primary_designs(Array1::zeros(n), Array1::ones(n), seed.clone())
    };
    let marginal_eta = 0.35;
    let slope = -0.8;
    let beta_h = Array1::zeros(score_prepared.runtime.basis_dim());
    let beta_w = Array1::zeros(link_prepared.runtime.basis_dim());

    let marginal = family
        .marginal_link_map(marginal_eta)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "marginal map", e));
    let scale = family.probit_frailty_scale();
    let rigid_a = rigid_prescale_intercept_from_marginal(marginal.q, slope, scale);
    let (f_rigid, f_a_rigid, _) = family
        .evaluate_denested_calibration_newton(
            rigid_a,
            marginal_eta,
            slope,
            Some(&beta_h),
            Some(&beta_w),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "denested zero-deviation calibration", e));
    assert!(
        f_rigid.abs() <= 5e-13,
        "closed-form rigid intercept residual should be at machine epsilon, got {f_rigid}"
    );
    let analytic_deriv = rigid_prescale_intercept_derivative_abs(marginal.q, slope, scale);
    assert!(
        (f_a_rigid - analytic_deriv).abs() <= 5e-13,
        "denested derivative {f_a_rigid} != analytic {analytic_deriv}"
    );

    let (a_fast, deriv_fast, fast_path) = family
        .solve_row_intercept_base(0, marginal_eta, slope, Some(&beta_h), Some(&beta_w), None)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "zero-deviation solve", e));
    assert!(
        fast_path,
        "zero coefficients should take analytic fast path"
    );
    assert_eq!(a_fast, rigid_a);
    assert_eq!(deriv_fast, analytic_deriv);
}

#[test]
fn exact_layout_ignores_dummy_beta_widths_for_empty_design_blocks() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score-warp block", e));
    let link_prepared = build_test_link_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link deviation block", e));
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(Array1::zeros(seed.len())),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        score_warp: Some(score_prepared.runtime.clone()),
        link_dev: Some(link_prepared.runtime.clone()),
        ..default_test_family()
    };
    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(
            Array1::zeros(score_prepared.runtime.basis_dim()),
            seed.len(),
        ),
        dummy_block_state(Array1::zeros(link_prepared.runtime.basis_dim()), seed.len()),
    ];

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    assert_eq!(cache.slices.marginal.len(), 0);
    assert_eq!(cache.slices.logslope.len(), 0);
    assert_eq!(cache.slices.h.as_ref().expect("h slice").start, 0);
    assert_eq!(
        cache.slices.w.as_ref().expect("w slice").start,
        score_prepared.runtime.basis_dim()
    );
    assert_eq!(
        cache.slices.total,
        score_prepared.runtime.basis_dim() + link_prepared.runtime.basis_dim()
    );
    assert_eq!(cache.primary.q, 0);
    assert_eq!(cache.primary.logslope, 1);
    assert_eq!(cache.primary.h.as_ref().expect("primary h").start, 2);
    assert_eq!(
        cache.primary.w.as_ref().expect("primary w").start,
        2 + score_prepared.runtime.basis_dim()
    );
}

#[test]
fn score_warp_block_exposes_structural_derivative_lower_bounds() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score-warp block", e));
    let score_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("score-warp initial beta")
        .len();
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(Array1::zeros(seed.len())),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        score_warp: Some(prepared.runtime.clone()),
        ..default_test_family()
    };
    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(Array1::zeros(score_dim), seed.len()),
    ];

    let dummy_spec = dummy_blockspec(score_dim, seed.len());
    let constraints = family
        .block_linear_constraints(&block_states, 2, &dummy_spec)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "constraint lookup", e))
        .expect("score-warp constraints");
    assert_eq!(constraints.a.ncols(), score_dim);
    assert_eq!(constraints.b.len(), constraints.a.nrows());
    assert!(
        constraints.a.nrows() >= score_dim,
        "anchored score-warp constraints should be expressed in raw derivative-control rows"
    );
    assert_eq!(
        constraints.b,
        Array1::<f64>::from_elem(
            constraints.a.nrows(),
            prepared.runtime.monotonicity_eps() - 1.0
        )
    );
}

#[test]
fn post_update_block_beta_rejects_infeasible_score_warp_step() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score-warp block", e));
    let score_dim = prepared.block.design.ncols();
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(Array1::zeros(seed.len())),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        score_warp: Some(prepared.runtime.clone()),
        ..default_test_family()
    };
    let current = Array1::<f64>::zeros(score_dim);
    let mut proposed = current.clone();
    proposed[0] = -128.0;
    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(current.clone(), seed.len()),
    ];
    let spec = dummy_blockspec(score_dim, seed.len());
    let err = family
        .post_update_block_beta(&block_states, 2, &spec, proposed.clone())
        .expect_err("post-update must not repair infeasible constrained beta");
    assert!(
        err.contains("structural monotonicity") || err.contains("exact monotonicity"),
        "unexpected error: {err}"
    );
}

#[test]
fn structural_deviation_runtime_is_piecewise_cubic() {
    let seed = array![-1.0, 0.0, 1.0];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "structural deviation basis", e));
    assert_eq!(prepared.runtime.degree(), 3);
    assert_eq!(prepared.runtime.value_span_degree(), 3);
    let has_cubic_curvature = prepared
        .runtime
        .span_c3()
        .iter()
        .any(|value| value.abs() > 1e-12);
    assert!(
        has_cubic_curvature,
        "structural deviation basis must expose true cubic span coefficients"
    );
}

#[test]
fn structural_deviation_runtime_is_c2_at_internal_breakpoints() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "structural deviation basis", e));
    let dim = prepared.block.design.ncols();
    let beta = Array1::from_iter((0..dim).map(|idx| 0.015 * (idx as f64 + 1.0)));
    let n_spans = prepared.runtime.breakpoints().len().saturating_sub(1);
    for span_idx in 1..n_spans {
        let left_cubic = prepared
            .runtime
            .local_cubic_on_span(beta.view(), span_idx - 1)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "left span cubic", e));
        let right_cubic = prepared
            .runtime
            .local_cubic_on_span(beta.view(), span_idx)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "right span cubic", e));
        let knot = prepared.runtime.breakpoints()[span_idx];
        assert!(
            (left_cubic.evaluate(knot) - right_cubic.evaluate(knot)).abs() <= 1e-10,
            "deviation value should be continuous at breakpoint {span_idx}"
        );
        assert!(
            (left_cubic.first_derivative(knot) - right_cubic.first_derivative(knot)).abs() <= 1e-10,
            "deviation first derivative should be continuous at breakpoint {span_idx}"
        );
        assert!(
            (left_cubic.second_derivative(knot) - right_cubic.second_derivative(knot)).abs()
                <= 1e-10,
            "deviation second derivative should be continuous at breakpoint {span_idx}"
        );
    }
}

#[test]
fn structural_deviation_rejects_noncubic_degree() {
    let seed = array![-1.0, 0.0, 1.0];
    let err = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            degree: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .expect_err("structural deviation block should reject non-cubic degree");
    assert!(err.contains("degree must be 3"));
}

#[test]
fn deviation_runtime_replays_exact_training_design() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build deviation block", e));
    let replayed = prepared
        .runtime
        .design(&seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "replayed design", e));
    let trained = prepared.block.design.to_dense();
    assert_eq!(replayed.dim(), trained.dim());
    for i in 0..replayed.nrows() {
        for j in 0..replayed.ncols() {
            assert!(
                (replayed[[i, j]] - trained[[i, j]]).abs() <= 1e-10,
                "training-basis replay mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn structural_constraints_match_exact_monotonicity_guard() {
    let seed = array![-1.0, 0.0, 1.0, 2.0];
    let prepared =
        build_score_warp_deviation_block_from_seed(&seed, &DeviationBlockConfig::default())
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "build deviation block", e));
    let constraints = prepared.runtime.structural_monotonicity_constraints();
    let dim = constraints.a.ncols();
    assert_eq!(dim, prepared.runtime.basis_dim());
    assert_eq!(
        constraints.a.nrows(),
        3 * prepared.runtime.breakpoints().len().saturating_sub(1)
    );
    assert_eq!(
        constraints.b,
        Array1::<f64>::from_elem(
            constraints.a.nrows(),
            prepared.runtime.monotonicity_eps() - 1.0
        )
    );
    let feasible = Array1::<f64>::zeros(dim);
    prepared
        .runtime
        .monotonicity_feasible(&feasible, "feasible structural beta")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "zero deviation should be feasible", e));
    let d1_design = prepared
        .runtime
        .first_derivative_design(&seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "derivative design", e));
    let row_idx = (0..d1_design.nrows())
        .find(|&idx| d1_design.row(idx).dot(&d1_design.row(idx)) > 0.0)
        .expect("derivative design should include a nonzero row");
    let derivative_row = d1_design.row(row_idx).to_owned();
    let row_norm_sq = derivative_row.dot(&derivative_row);
    let infeasible = derivative_row.mapv(|value| -2.0 * value / row_norm_sq);
    assert!(
        prepared
            .runtime
            .monotonicity_feasible(&infeasible, "infeasible structural beta")
            .is_err()
    );
}

#[test]
fn structural_constraints_are_quadratic_derivative_bernstein_controls() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build deviation block", e));
    let constraints = prepared.runtime.structural_monotonicity_constraints();
    let beta = Array1::from_iter((0..prepared.runtime.basis_dim()).map(|idx| {
        let centered = idx as f64 - 0.5 * (prepared.runtime.basis_dim() as f64 - 1.0);
        0.025 * centered
    }));
    let controls = constraints.a.dot(&beta);
    let n_spans = prepared.runtime.breakpoints().len().saturating_sub(1);
    for span_idx in 0..n_spans {
        let cubic = prepared
            .runtime
            .local_cubic_on_span(beta.view(), span_idx)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "local cubic", e));
        let left = cubic.left;
        let right = cubic.right;
        let mid = 0.5 * (left + right);
        let b0 = controls[3 * span_idx];
        let b1 = controls[3 * span_idx + 1];
        let b2 = controls[3 * span_idx + 2];
        assert!(
            (b0 - cubic.first_derivative(left)).abs() <= 1e-10,
            "left Bernstein control should equal derivative at span start"
        );
        assert!(
            (b2 - cubic.first_derivative(right)).abs() <= 1e-10,
            "right Bernstein control should equal derivative at span end"
        );
        let midpoint_from_bernstein = 0.25 * b0 + 0.5 * b1 + 0.25 * b2;
        assert!(
            (midpoint_from_bernstein - cubic.first_derivative(mid)).abs() <= 1e-10,
            "quadratic Bernstein controls should reconstruct derivative at span midpoint"
        );
    }
}

#[test]
fn deviation_penalties_are_integrated_function_penalties() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            penalty_order: 2,
            penalty_orders: vec![1, 2, 3],
            double_penalty: true,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build deviation block", e));

    let expected_orders = [1, 2, 3, 0];
    assert_eq!(prepared.block.penalties.len(), expected_orders.len());

    for ((penalty, &nullity), &order) in prepared
        .block
        .penalties
        .iter()
        .zip(prepared.block.nullspace_dims.iter())
        .zip(expected_orders.iter())
    {
        let gam_solve::estimate::PenaltySpec::Dense(actual) = penalty else {
            panic!("deviation penalties should be dense local Gram matrices");
        };
        let (expected, expected_nullity) = prepared
            .runtime
            .integrated_derivative_penalty_with_nullity(order)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "integrated function penalty", e));
        assert_eq!(nullity, expected_nullity);
        assert_eq!(actual.dim(), expected.dim());
        for i in 0..actual.nrows() {
            for j in 0..actual.ncols() {
                assert!(
                    (actual[[i, j]] - expected[[i, j]]).abs() <= 1e-10,
                    "penalty order {order} mismatch at ({i},{j}): got {}, expected {}",
                    actual[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    let gam_solve::estimate::PenaltySpec::Dense(l2_penalty) = &prepared.block.penalties[1] else {
        panic!("deviation double penalty should be dense");
    };
    let mut max_identity_diff = 0.0_f64;
    for i in 0..l2_penalty.nrows() {
        for j in 0..l2_penalty.ncols() {
            let identity = if i == j { 1.0 } else { 0.0 };
            max_identity_diff = max_identity_diff.max((l2_penalty[[i, j]] - identity).abs());
        }
    }
    assert!(
        max_identity_diff > 1e-6,
        "deviation double penalty must be integrated L2, not coefficient identity"
    );
}

#[test]
fn local_cubic_span_reconstructs_deviation_exactly() {
    // Score-warp deviation runtime: C² piecewise-cubic basis.
    //
    // Continuity across interior breakpoints:
    //   value  (d0) — C⁰ continuous (matches on both sides)
    //   slope  (d1) — C¹ continuous (matches on both sides)
    //   curvature (d2) — C² continuous (matches on both sides)
    //
    // `evaluate_span_polynomial_design` resolves the two-sided ambiguity at
    // an interior breakpoint x == endpoint_points[k] (0 < k < last) by
    // biasing to the LEFT span (span_idx = k - 1). For a C² cubic this is
    // numerically the same value through d2; only d3 is span-local.
    //
    // This test reconstructs each span's polynomial from design rows.
    // For d0/d1/d2 the expected value matches the selected span at every
    // sample point.
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build deviation block", e));
    let dim = prepared.block.design.ncols();
    let beta = Array1::from_iter((0..dim).map(|idx| 0.025 * (idx as f64 + 1.0)));
    let n_spans = prepared.runtime.breakpoints().len().saturating_sub(1);
    let support_left = prepared.runtime.breakpoints()[0];
    let support_right = prepared.runtime.breakpoints()[prepared.runtime.breakpoints().len() - 1];

    for span_idx in 0..n_spans {
        let cubic = prepared
            .runtime
            .local_cubic_on_span(beta.view(), span_idx)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "local cubic coefficients", e));
        let left = cubic.left;
        let right = cubic.right;
        let x_eval = array![left, 0.5 * (left + right), right];
        let value_design = prepared
            .runtime
            .design(&x_eval)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "value design", e));
        let d1_design = prepared
            .runtime
            .first_derivative_design(&x_eval)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "first derivative design", e));
        let d2_design = prepared
            .runtime
            .second_derivative_design(&x_eval)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "second derivative design", e));
        let expected = value_design.dot(&beta);
        let expected_d1 = d1_design.dot(&beta);
        let expected_d2 = d2_design.dot(&beta);
        for i in 0..x_eval.len() {
            let x = x_eval[i];
            assert!(
                (cubic.evaluate(x) - expected[i]).abs() < 1e-10,
                "span {span_idx}, x={x:.6}: cubic value mismatch"
            );
            assert!(
                (cubic.first_derivative(x) - expected_d1[i]).abs() < 1e-10,
                "span {span_idx}, x={x:.6}: cubic first-derivative mismatch"
            );
            assert!(
                (cubic.second_derivative(x) - expected_d2[i]).abs() < 1e-10,
                "span {span_idx}, x={x:.6}: cubic second-derivative mismatch"
            );
            let selected = prepared
                .runtime
                .local_cubic_at(beta.view(), x)
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "lookup cubic at x", e));
            if x < support_left || x > support_right {
                // Strictly outside support: tail saturation — constant
                // value, zero slope and curvature.
                assert!(selected.c1.abs() < 1e-12);
                assert!(selected.c2.abs() < 1e-12);
                assert!(selected.c3.abs() < 1e-12);
                assert!((selected.evaluate(x) - expected[i]).abs() < 1e-10);
            } else {
                // Interior or exact boundary point: uses the same
                // left-biased span convention as derivative designs.
                let expected_span_idx = if i == 0 && span_idx > 0 {
                    span_idx - 1
                } else {
                    span_idx
                };
                let expected_cubic = prepared
                    .runtime
                    .local_cubic_on_span(beta.view(), expected_span_idx)
                    .unwrap_or_else(|e| {
                        panic!("{} failed: {:?}", "expected lookup cubic on span", e)
                    });
                assert_eq!(selected, expected_cubic);
            }
        }
    }
}

#[test]
fn basis_span_cubic_reconstructs_basis_column_exactly() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build deviation block", e));
    let basis_idx = 0usize;
    let support_left = prepared.runtime.breakpoints()[0];
    let support_right = prepared.runtime.breakpoints()[prepared.runtime.breakpoints().len() - 1];
    let cubic = prepared
        .runtime
        .basis_span_cubic(0, basis_idx)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "basis span cubic", e));
    let x_eval = array![cubic.left, 0.5 * (cubic.left + cubic.right), cubic.right];
    let design = prepared
        .runtime
        .design(&x_eval)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "basis design", e));
    let d1 = prepared
        .runtime
        .first_derivative_design(&x_eval)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "basis d1 design", e));
    for i in 0..x_eval.len() {
        let x = x_eval[i];
        assert!((cubic.evaluate(x) - design[[i, basis_idx]]).abs() < 1e-10);
        assert!((cubic.first_derivative(x) - d1[[i, basis_idx]]).abs() < 1e-10);
        let selected = prepared
            .runtime
            .basis_cubic_at(basis_idx, x)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "basis cubic at x", e));
        if x < support_left || x > support_right {
            // Strictly outside support: tail saturation.
            assert!(selected.c1.abs() < 1e-12);
            assert!(selected.c2.abs() < 1e-12);
            assert!(selected.c3.abs() < 1e-12);
            assert!((selected.evaluate(x) - design[[i, basis_idx]]).abs() < 1e-10);
        } else {
            let expected_span_idx = 0;
            let expected_cubic = prepared
                .runtime
                .basis_span_cubic(expected_span_idx, basis_idx)
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "expected basis span cubic", e));
            assert_eq!(selected, expected_cubic);
        }
    }
}

#[test]
fn deviation_runtime_saturates_outside_support() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build deviation block", e));
    let dim = prepared.block.design.ncols();
    let beta = Array1::from_iter((0..dim).map(|idx| 0.02 * (idx as f64 + 1.0)));
    let left = prepared.runtime.breakpoints()[0];
    let right = prepared.runtime.breakpoints()[prepared.runtime.breakpoints().len() - 1];

    let left_tail_near = prepared
        .runtime
        .local_cubic_at(beta.view(), left - 0.25)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "left tail", e));
    let left_tail_far = prepared
        .runtime
        .local_cubic_at(beta.view(), left - 3.0)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "left far tail", e));
    let right_tail_near = prepared
        .runtime
        .local_cubic_at(beta.view(), right + 0.25)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "right tail", e));
    let right_tail_far = prepared
        .runtime
        .local_cubic_at(beta.view(), right + 3.0)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "right far tail", e));

    for cubic in [
        left_tail_near,
        left_tail_far,
        right_tail_near,
        right_tail_far,
    ] {
        assert!(cubic.c1.abs() < 1e-12);
        assert!(cubic.c2.abs() < 1e-12);
        assert!(cubic.c3.abs() < 1e-12);
    }
    assert!((left_tail_near.c0 - left_tail_far.c0).abs() < 1e-12);
    assert!((right_tail_near.c0 - right_tail_far.c0).abs() < 1e-12);
}

#[test]
fn deviation_runtime_replays_the_exact_training_basis() {
    let seed = array![-2.0, -1.0, -0.25, 0.25, 1.0, 2.0];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build deviation block", e));

    let replayed = prepared
        .runtime
        .design(&seed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "replay anchored deviation design", e));
    let trained = prepared.block.design.to_dense();
    assert_eq!(replayed.dim(), trained.dim());
    for i in 0..replayed.nrows() {
        for j in 0..replayed.ncols() {
            assert!(
                (replayed[[i, j]] - trained[[i, j]]).abs() <= 1e-10,
                "replayed anchored deviation design mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn denested_microcells_follow_score_and_link_breaks() {
    let score_seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let link_seed = array![-1.5, -0.5, 0.5, 1.5];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &score_seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score warp block", e));
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link deviation block", e));
    let beta_h = Array1::from_iter(
        (0..score_prepared.block.design.ncols()).map(|idx| 0.02 * (idx as f64 + 1.0)),
    );
    let beta_w = Array1::from_iter(
        (0..link_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
    );

    let exact_cells_a0 = build_denested_partition_cells(
        0.25,
        0.9,
        score_prepared
            .runtime
            .breakpoints()
            .as_slice()
            .expect("score breaks"),
        link_prepared
            .runtime
            .breakpoints()
            .as_slice()
            .expect("link breaks"),
        |z| score_prepared.runtime.local_cubic_at(beta_h.view(), z),
        |u| link_prepared.runtime.local_cubic_at(beta_w.view(), u),
    )
    .expect("exact module microcells for a=0.25");
    let exact_cells_a1 = build_denested_partition_cells(
        0.55,
        0.9,
        score_prepared
            .runtime
            .breakpoints()
            .as_slice()
            .expect("score breaks"),
        link_prepared
            .runtime
            .breakpoints()
            .as_slice()
            .expect("link breaks"),
        |z| score_prepared.runtime.local_cubic_at(beta_h.view(), z),
        |u| link_prepared.runtime.local_cubic_at(beta_w.view(), u),
    )
    .expect("exact module microcells for a=0.55");

    assert!(
        exact_cells_a0.len() >= score_prepared.runtime.breakpoints().len().saturating_sub(1),
        "microcell partition should refine the score spans"
    );
    assert!(
        exact_cells_a0
            .windows(2)
            .all(|w| (w[0].cell.right - w[1].cell.left).abs() <= 1e-12),
        "microcells should tile the partition contiguously"
    );
    assert!(exact_cells_a0.first().unwrap().cell.left.is_infinite());
    assert!(exact_cells_a0.last().unwrap().cell.right.is_infinite());
    assert!(
        exact_cells_a0
            .iter()
            .zip(exact_cells_a1.iter())
            .any(|(lhs, rhs)| (lhs.cell.left - rhs.cell.left).abs() > 1e-10),
        "changing the intercept should move at least one link-induced breakpoint"
    );
}

#[test]
fn denested_microcell_eta_matches_direct_denested_formula() {
    let score_seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let link_seed = array![-1.5, -0.5, 0.5, 1.5];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &score_seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score warp block", e));
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link deviation block", e));
    let beta_h = Array1::from_iter(
        (0..score_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
    );
    let beta_w = Array1::from_iter(
        (0..link_prepared.block.design.ncols()).map(|idx| 0.02 * (idx as f64 + 1.0)),
    );

    let a = 0.35;
    let b = -0.7;
    let cells = build_denested_partition_cells(
        a,
        b,
        score_prepared
            .runtime
            .breakpoints()
            .as_slice()
            .expect("score breaks"),
        link_prepared
            .runtime
            .breakpoints()
            .as_slice()
            .expect("link breaks"),
        |z| score_prepared.runtime.local_cubic_at(beta_h.view(), z),
        |u| link_prepared.runtime.local_cubic_at(beta_w.view(), u),
    )
    .expect("microcells");

    for cell in &cells {
        let z = exact_kernel::interval_probe_point(cell.cell.left, cell.cell.right)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "finite microcell probe", e));
        let h = score_prepared
            .runtime
            .design(&array![z])
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "score design", e))
            .row(0)
            .dot(&beta_h);
        let link = link_prepared
            .runtime
            .design(&array![a + b * z])
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "link design", e))
            .row(0)
            .dot(&beta_w);
        let expected = a + b * z + b * h + link;
        assert!(
            (cell.cell.eta(z) - expected).abs() < 1e-10,
            "microcell eta should equal the direct de-nested predictor at z={z:.6}"
        );
    }
}

#[test]
fn local_cubic_global_transform_reconstructs_same_function() {
    let cubic = exact_kernel::LocalSpanCubic {
        left: -1.3,
        right: 0.7,
        c0: 0.4,
        c1: -0.2,
        c2: 0.15,
        c3: -0.05,
    };
    let (g0, g1, g2, g3) = exact_global_cubic_from_local(LocalSpanCubic {
        left: cubic.left,
        right: cubic.right,
        c0: cubic.c0,
        c1: cubic.c1,
        c2: cubic.c2,
        c3: cubic.c3,
    });
    for &x in &[-1.3, -0.8, -0.1, 0.5, 0.7] {
        let direct = cubic.evaluate(x);
        let global = g0 + g1 * x + g2 * x * x + g3 * x * x * x;
        assert!(
            (direct - global).abs() < 1e-12,
            "global cubic transform should preserve the span polynomial at x={x}"
        );
    }
}

#[test]
fn denested_branch_selection_uses_normalized_cell_coefficients() {
    let affine = ExactDenestedCubicCell {
        left: -1.0,
        right: 1.0,
        c0: 0.2,
        c1: -0.4,
        c2: 1e-13,
        c3: -1e-13,
    };
    let quartic = ExactDenestedCubicCell {
        c2: 2e-4,
        c3: 1e-13,
        ..affine
    };
    let sextic = ExactDenestedCubicCell {
        c2: 2e-4,
        c3: 5e-3,
        ..affine
    };
    assert_eq!(
        branch_exact_cell(affine).unwrap_or_else(|e| panic!("{} failed: {:?}", "affine branch", e)),
        ExactCellBranchShared::Affine
    );
    assert_eq!(
        branch_exact_cell(quartic)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "quartic branch", e)),
        ExactCellBranchShared::Quartic
    );
    assert_eq!(
        branch_exact_cell(sextic).unwrap_or_else(|e| panic!("{} failed: {:?}", "sextic branch", e)),
        ExactCellBranchShared::Sextic
    );
}

#[test]
fn denested_cell_coefficient_partials_match_finite_differences() {
    let score_span = exact_kernel::LocalSpanCubic {
        left: -0.75,
        right: 0.25,
        c0: 0.08,
        c1: -0.03,
        c2: 0.02,
        c3: -0.01,
    };
    let link_span = exact_kernel::LocalSpanCubic {
        left: -0.6,
        right: 0.9,
        c0: -0.05,
        c1: 0.04,
        c2: -0.02,
        c3: 0.015,
    };
    let a = 0.3;
    let b = -0.7;
    let eps = 1e-6;

    let coeffs = |aa: f64, bb: f64| {
        let (h0, h1, h2, h3) = exact_global_cubic_from_local(LocalSpanCubic {
            left: score_span.left,
            right: score_span.right,
            c0: score_span.c0,
            c1: score_span.c1,
            c2: score_span.c2,
            c3: score_span.c3,
        });
        let (d0, d1, d2, d3) = exact_transformed_link_cubic(
            LocalSpanCubic {
                left: link_span.left,
                right: link_span.right,
                c0: link_span.c0,
                c1: link_span.c1,
                c2: link_span.c2,
                c3: link_span.c3,
            },
            aa,
            bb,
        );
        [
            aa + bb * h0 + d0,
            bb + bb * h1 + d1,
            bb * h2 + d2,
            bb * h3 + d3,
        ]
    };
    let (dc_da, dc_db) = exact_denested_cell_coefficient_partials(
        LocalSpanCubic {
            left: score_span.left,
            right: score_span.right,
            c0: score_span.c0,
            c1: score_span.c1,
            c2: score_span.c2,
            c3: score_span.c3,
        },
        LocalSpanCubic {
            left: link_span.left,
            right: link_span.right,
            c0: link_span.c0,
            c1: link_span.c1,
            c2: link_span.c2,
            c3: link_span.c3,
        },
        a,
        b,
    );
    let plus_a = coeffs(a + eps, b);
    let minus_a = coeffs(a - eps, b);
    let plus_b = coeffs(a, b + eps);
    let minus_b = coeffs(a, b - eps);
    for j in 0..4 {
        let fd_a = (plus_a[j] - minus_a[j]) / (2.0 * eps);
        let fd_b = (plus_b[j] - minus_b[j]) / (2.0 * eps);
        assert!(
            (dc_da[j] - fd_a).abs() < 1e-6,
            "dc/da mismatch at coefficient {j}: analytic={}, fd={fd_a}",
            dc_da[j]
        );
        assert!(
            (dc_db[j] - fd_b).abs() < 1e-6,
            "dc/db mismatch at coefficient {j}: analytic={}, fd={fd_b}",
            dc_db[j]
        );
    }
}

#[test]
fn observed_denested_partials_include_third_a_derivative_for_piecewise_cubic_link() {
    let z = array![-0.8, 0.2, 1.1];
    let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link block", e));
    let beta_w = Array1::from_iter(
        (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
    );
    let family =
        BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 1.0]),
            weights: Arc::new(array![1.0, 0.7, 1.3]),
            z: Arc::new(z.clone()),
            marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            link_dev: Some(link_prepared.runtime.clone()),
            ..default_test_family()
        };

    let a = 0.35;
    let b = 0.6;
    let row = 1usize;
    let obs = family
        .observed_denested_cell_partials(row, a, b, None, Some(&beta_w))
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "observed denested partials", e));
    let u_obs = a + b * z[row];
    let link_span = link_prepared
        .runtime
        .local_cubic_at(beta_w.view(), u_obs)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "local cubic at observed point", e));
    let expected_daaa = exact_kernel::denested_cell_third_partials(link_span).0;

    assert_eq!(obs.dc_daaa, expected_daaa);
    assert!(
        eval_coeff4_at(&obs.dc_daaa, z[row]).abs() > 1e-12,
        "piecewise-cubic link spans should contribute a third a-derivative"
    );
}

#[test]
fn pooled_probit_baseline_matches_expanded_integer_weight_fit() {
    let y = array![0.0, 1.0, 0.0, 1.0];
    let z = array![-1.5, -0.2, 0.4, 1.4];
    let weights = array![25.0, 2.0, 1.0, 20.0];
    let weighted = pooled_probit_baseline(&y, &z, &weights)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "weighted baseline", e));
    let unweighted = pooled_probit_baseline(&y, &z, &Array1::ones(y.len()))
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "unweighted baseline", e));
    let (y_expanded, z_expanded) = expand_integer_weight_rows(&y, &z, &weights);
    let expanded =
        pooled_probit_baseline(&y_expanded, &z_expanded, &Array1::ones(y_expanded.len()))
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "expanded baseline", e));

    assert!(
        pair_distance(expanded, unweighted) > 1e-2,
        "test data should distinguish weighted from unweighted seeding"
    );
    assert!(
        pair_distance(weighted, expanded) < 1e-8,
        "weighted pilot baseline should match the expanded integer-weight fit"
    );
}

#[test]
fn validate_spec_rejects_nonfinite_or_negative_weights() {
    let data = Array2::<f64>::zeros((3, 0));
    let y = array![0.0, 1.0, 0.0];
    let z = array![-1.0, 0.0, 1.0];

    let err = validate_spec(
        data.view(),
        &base_spec(y.clone(), array![1.0, f64::NAN, 1.0], z.clone()),
    )
    .expect_err("non-finite weights should be rejected");
    assert!(err.contains("finite non-negative weights"));

    let err = validate_spec(data.view(), &base_spec(y, array![1.0, -0.5, 1.0], z))
        .expect_err("negative weights should be rejected");
    assert!(err.contains("finite non-negative weights"));
}

#[test]
fn validate_spec_rejects_nonfinite_z_values() {
    let data = Array2::<f64>::zeros((3, 0));
    let err = validate_spec(
        data.view(),
        &base_spec(
            array![0.0, 1.0, 0.0],
            array![1.0, 1.0, 1.0],
            array![-1.0, f64::INFINITY, 1.0],
        ),
    )
    .expect_err("non-finite z should be rejected");
    assert!(err.contains("finite z values"));
}

#[test]
fn validate_spec_accepts_learnable_gaussian_shift_sigma() {
    let data = Array2::<f64>::zeros((3, 0));
    let mut spec = base_spec(
        array![0.0, 1.0, 0.0],
        array![1.0, 1.0, 1.0],
        array![-1.0, 0.0, 1.0],
    );
    spec.frailty = FrailtySpec::GaussianShift {
        scale: FrailtyScale::Learned { initial_sigma: 0.5 },
    };

    validate_spec(data.view(), &spec).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "learnable GaussianShift sigma should validate", e
        )
    });
}

#[test]
fn signed_probit_helpers_handle_nonfinite_boundaries_explicitly() {
    let (logcdf_pos, lambda_pos) = signed_probit_logcdf_and_mills_ratio(f64::INFINITY);
    assert_eq!(logcdf_pos, 0.0);
    assert_eq!(lambda_pos, 0.0);

    let (logcdf_neg, lambda_neg) = signed_probit_logcdf_and_mills_ratio(f64::NEG_INFINITY);
    assert_eq!(logcdf_neg, f64::NEG_INFINITY);
    assert_eq!(lambda_neg, f64::INFINITY);

    let (logcdf_nan, lambda_nan) = signed_probit_logcdf_and_mills_ratio(f64::NAN);
    assert!(logcdf_nan.is_nan());
    assert!(lambda_nan.is_nan());
}

#[test]
fn signed_probit_exact_derivative_helper_rejects_invalid_nonfinite_margins() {
    assert_eq!(
        signed_probit_neglog_derivatives_up_to_fourth(f64::INFINITY, 2.5)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "+inf should use the zero tail", e)),
        (0.0, 0.0, 0.0, 0.0)
    );

    let neg_inf_err = signed_probit_neglog_derivatives_up_to_fourth(f64::NEG_INFINITY, 2.5)
        .expect_err("-inf should be rejected in the exact derivative path");
    assert!(neg_inf_err.contains("non-finite signed margin"));

    let nan_err = signed_probit_neglog_derivatives_up_to_fourth(f64::NAN, 2.5)
        .expect_err("NaN should be rejected in the exact derivative path");
    assert!(nan_err.contains("non-finite signed margin"));
}

#[test]
fn unary_neglog_phi_preserves_negative_infinity_and_nan_boundaries() {
    assert_eq!(
        unary_derivatives_neglog_phi(f64::INFINITY, 1.75),
        [0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        unary_derivatives_neglog_phi(f64::NEG_INFINITY, 1.75),
        [f64::INFINITY, f64::NEG_INFINITY, 1.75, 0.0, 0.0]
    );
    let nan_terms = unary_derivatives_neglog_phi(f64::NAN, 1.75);
    assert!(nan_terms.iter().all(|value| value.is_nan()));
}

#[test]
fn flexible_family_routes_outer_derivatives_by_scale() {
    let seed = array![-1.0, 0.0, 1.0];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "score warp block", e));
    let family =
        BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(3)),
            z: Arc::new(seed.clone()),
            marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            score_warp: Some(score_prepared.runtime.clone()),
            ..default_test_family()
        };
    let specs = vec![
        dummy_blockspec(1, 3),
        dummy_blockspec(1, 3),
        dummy_blockspec(2, 3),
    ];
    // Flex (`score_warp` active) now advertises the EXACT profiled outer θ-HVP:
    // `outer_hyper_hessian_hvp_available` returns true when the realized specs
    // match the family row count, and `exact_outer_derivative_order` falls
    // through to the HVP-aware order (`Second`) instead of demoting flex to a
    // first-order BFGS outer loop. The operator is the matrix-free exact
    // analytic θ-HVP assembled from the family's directional-derivative kernels,
    // so the REML optimum is unchanged; only the outer optimizer geometry
    // (ARC / exact Newton instead of BFGS) improves.
    assert!(family.outer_hyper_hessian_hvp_available(&specs));
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
    assert!(family.exact_newton_joint_psi_workspace_for_first_order_terms());

    let n_large = 50_000;
    let mut large_flex_family = family.clone();
    large_flex_family.y = Arc::new(Array1::zeros(n_large));
    large_flex_family.weights = Arc::new(Array1::ones(n_large));
    large_flex_family.z = Arc::new(Array1::zeros(n_large));
    let large_flex_specs = vec![
        dummy_blockspec(1, n_large),
        dummy_blockspec(1, n_large),
        dummy_blockspec(2, n_large),
    ];
    // Large-scale flex stays on the exact second-order path through the
    // matrix-free θ-HVP operator: even at n=50k the operator reuses the flex
    // row stream once per Hv (near-gradient cost) and PCG converges in a few
    // iters, so the outer Newton converges in ≤ a couple of iterations rather
    // than several full-inner-resolve BFGS line searches.
    assert!(large_flex_family.outer_hyper_hessian_hvp_available(&large_flex_specs));
    assert_eq!(
        large_flex_family
            .exact_outer_derivative_order(&large_flex_specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
    let (large_flex_gradient, large_flex_hessian) = custom_family_outer_derivatives(
        &large_flex_family,
        &large_flex_specs,
        &BlockwiseFitOptions::default(),
    );
    assert_eq!(large_flex_gradient, gam_problem::Derivative::Analytic);
    assert_eq!(large_flex_hessian, gam_problem::DeclaredHessianForm::Either);

    let mut large_rigid_family = large_flex_family.clone();
    large_rigid_family.score_warp = None;
    let large_rigid_specs = vec![dummy_blockspec(1, n_large), dummy_blockspec(1, n_large)];
    assert_eq!(
        large_rigid_family
            .exact_outer_derivative_order(&large_rigid_specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
}

#[test]
fn bms_advertises_exact_outer_hvp_and_plans_arc_outer_newton() {
    // Regression for the BMS outer-loop lever: the Bernoulli-marginal-slope
    // family used to report `outer_hvp_available=false` (trait default), which
    // forced the outer ρ-optimization onto BFGS — several outer iterations,
    // each a Value + ValueAndGradient + multiple StrongWolfe line-search
    // probes, every probe a full inner re-solve. BMS now mirrors its survival
    // twin and advertises the EXACT matrix-free profiled outer θ-HVP, lifting
    // the outer-derivative order to `Second` so the planner selects ARC /
    // exact outer Newton and converges the outer loop in ≤ a couple of iters.
    //
    // The advertised operator is the exact analytic θ-HVP assembled from the
    // family's `exact_newton_joint_hessian_directional_derivative` /
    // `...second_directional_derivative` / `exact_newton_joint_psi_terms`
    // kernels by the generic custom-family joint-hyper assembler — never a
    // finite-difference or quasi-Newton surrogate — so the REML/LAML optimum
    // is bit-identical to the first-order path.
    use gam_problem::{DeclaredHessianForm, Derivative};
    use gam_solve::rho_optimizer::{OuterCapability, Solver, plan};

    let arc_plan = |hessian: DeclaredHessianForm, n_params: usize, psi_dim: usize| {
        // `(Analytic, Either)` is matched as `(Analytic, Analytic)` by the
        // planner's `declared_hessian_for_planning`, which is the FIRST match
        // arm — so it routes to ARC regardless of EFS/HybridEfs eligibility
        // (the flex case carries ψ coords and would otherwise reach HybridEfs).
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian,
            n_params,
            psi_dim,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        plan(&cap).solver
    };

    // Rigid two-block path (no ψ coords).
    let rigid_family = make_rigid_test_family(64);
    let rigid_specs = vec![dummy_blockspec(1, 64), dummy_blockspec(1, 64)];
    assert!(rigid_family.outer_hyper_hessian_hvp_available(&rigid_specs));
    assert_eq!(
        rigid_family.exact_outer_derivative_order(&rigid_specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
    let (rigid_gradient, rigid_hessian) = custom_family_outer_derivatives(
        &rigid_family,
        &rigid_specs,
        &BlockwiseFitOptions::default(),
    );
    assert_eq!(rigid_gradient, Derivative::Analytic);
    assert_eq!(rigid_hessian, DeclaredHessianForm::Either);
    assert_eq!(arc_plan(rigid_hessian, rigid_specs.len(), 0), Solver::Arc);

    // Flex path (score-warp deviation block adds ψ coords): the prior behavior
    // demoted this to BFGS; it now plans ARC on the exact θ-HVP.
    let seed = array![-1.0, 0.0, 1.0];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "score warp block", e));
    let flex_family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![0.0, 1.0, 0.0]),
        weights: Arc::new(Array1::ones(3)),
        z: Arc::new(seed.clone()),
        marginal_design: dense_design(array![[1.0], [1.0], [1.0]]),
        logslope_design: dense_design(array![[1.0], [1.0], [1.0]]),
        score_warp: Some(score_prepared.runtime.clone()),
        ..default_test_family()
    };
    let flex_specs = vec![
        dummy_blockspec(1, 3),
        dummy_blockspec(1, 3),
        dummy_blockspec(2, 3),
    ];
    assert!(flex_family.outer_hyper_hessian_hvp_available(&flex_specs));
    assert_eq!(
        flex_family.exact_outer_derivative_order(&flex_specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
    let (flex_gradient, flex_hessian) =
        custom_family_outer_derivatives(&flex_family, &flex_specs, &BlockwiseFitOptions::default());
    assert_eq!(flex_gradient, Derivative::Analytic);
    assert_eq!(flex_hessian, DeclaredHessianForm::Either);
    // The flex deviation block carries ψ coords; assert ARC wins over the
    // HybridEfs lane that would otherwise be eligible for a ψ-bearing problem.
    assert_eq!(arc_plan(flex_hessian, flex_specs.len(), 1), Solver::Arc);
}

#[test]
fn exact_outer_order_stays_second_order_at_large_scale_work_scale() {
    use crate::custom_family::{
        default_coefficient_hessian_cost, exact_outer_order_from_capability,
    };
    use gam_linalg::matrix::DesignMatrix;
    use ndarray::Array2;

    // Small problem (K=4, n=10, p=8): combined cost ≪ threshold → Second.
    let small_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::zeros((10, 8)),
    ));
    let small_specs: Vec<ParameterBlockSpec> = (0..2)
        .map(|i| ParameterBlockSpec {
            name: format!("block_{i}"),
            design: small_design.clone(),
            offset: ndarray::Array1::zeros(10),
            penalties: (0..2)
                .map(|_| crate::custom_family::PenaltyMatrix::Dense(Array2::zeros((8, 8))))
                .collect(),
            nullspace_dims: vec![0; 2],
            initial_log_lambdas: ndarray::Array1::zeros(2),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
        .collect();
    let small_cost = default_coefficient_hessian_cost(&small_specs);
    assert_eq!(
        exact_outer_order_from_capability(&small_specs, small_cost),
        ExactOuterDerivativeOrder::Second
    );

    // Large-scale problem. This used to demote to first-order BFGS based
    // on a K²·n·p² work estimate. The new contract keeps exact analytic
    // Hessians exposed; representation and runtime cost are handled by the
    // Hessian operator layer.
    let big_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::zeros((5_000, 500)),
    ));
    let big_specs: Vec<ParameterBlockSpec> = (0..2)
        .map(|i| ParameterBlockSpec {
            name: format!("block_{i}"),
            design: big_design.clone(),
            offset: ndarray::Array1::zeros(5_000),
            penalties: (0..10)
                .map(|_| crate::custom_family::PenaltyMatrix::Dense(Array2::zeros((500, 500))))
                .collect(),
            nullspace_dims: vec![0; 10],
            initial_log_lambdas: ndarray::Array1::zeros(10),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
        .collect();
    let big_cost = default_coefficient_hessian_cost(&big_specs);
    assert_eq!(
        exact_outer_order_from_capability(&big_specs, big_cost),
        ExactOuterDerivativeOrder::Second
    );

    // Production CTN-on-PC scenario: K=22 (ρ_dim=6 + log-κ_dim=16), n≈4·10⁵
    // observations with p_total = p_resp·p_cov ≈ 300 covers a Khatri–Rao
    // joint Hessian whose dense build is n·p² ≈ 3.6·10¹⁰ flops. This must
    // still advertise second-order calculus; CTN supplies operator-form
    // Hessian geometry so no first-order downgrade is acceptable.
    //
    // Pass the production coefficient cost directly (the gate only reads
    // K from specs; allocating a 400k×300 zero matrix would burn ~1 GB
    // for no reason).
    let ctn_specs = vec![ParameterBlockSpec {
        name: "ctn".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (1, 1),
        ))),
        offset: ndarray::Array1::zeros(1),
        penalties: (0..22)
            .map(|_| crate::custom_family::PenaltyMatrix::Dense(Array2::zeros((1, 1))))
            .collect(),
        nullspace_dims: vec![0; 22],
        initial_log_lambdas: ndarray::Array1::zeros(22),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let ctn_cost: u64 = 400_000u64 * 300 * 300;
    assert_eq!(
        exact_outer_order_from_capability(&ctn_specs, ctn_cost),
        ExactOuterDerivativeOrder::Second
    );

    // The declaration helper no longer uses cost or HVP presence as a
    // downgrade policy. Callers check whether any analytic second-order
    // representation exists before calling it.
    use crate::custom_family::exact_outer_order_with_outer_hvp;
    assert_eq!(
        exact_outer_order_with_outer_hvp(&ctn_specs, ctn_cost, false),
        ExactOuterDerivativeOrder::Second,
        "exact outer order must not be cost-demoted"
    );
    let huge_k_specs = vec![ParameterBlockSpec {
        name: "k".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (1, 1),
        ))),
        offset: ndarray::Array1::zeros(1),
        penalties: (0..5_000)
            .map(|_| crate::custom_family::PenaltyMatrix::Dense(Array2::zeros((1, 1))))
            .collect(),
        nullspace_dims: vec![0; 5_000],
        initial_log_lambdas: ndarray::Array1::zeros(5_000),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    assert_eq!(
        exact_outer_order_with_outer_hvp(&huge_k_specs, 0, true),
        ExactOuterDerivativeOrder::Second,
        "outer HVP support keeps the exact second-order declaration regardless of K"
    );
}

#[test]
fn bernoulli_marginal_slope_coefficient_cost_uses_joint_coupled_formula() {
    use crate::custom_family::default_coefficient_hessian_cost;
    use gam_linalg::matrix::DesignMatrix;

    // Two-block rigid marginal-slope shape: marginal p=20, log-slope p=8,
    // n=1000. The default block-diagonal formula gives Σ n·p_b² =
    // 1000·(400 + 64) = 464_000. The joint-coupled override must add
    // the cross-block outer-product fill 2·n·p_marg·p_log = 320_000,
    // producing n·(p_marg + p_log)² = 1000·784 = 784_000.
    let n = 1000usize;
    let p_marg = 20usize;
    let p_log = 8usize;
    let marg_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::zeros((n, p_marg)),
    ));
    let log_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::zeros((n, p_log)),
    ));
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(Array1::zeros(n)),
        weights: Arc::new(Array1::from_elem(n, 1.0)),
        z: Arc::new(Array1::zeros(n)),
        marginal_design: marg_design.clone(),
        logslope_design: log_design.clone(),
        ..default_test_family()
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "marginal".to_string(),
            design: marg_design,
            offset: Array1::zeros(n),
            penalties: vec![crate::custom_family::PenaltyMatrix::Dense(Array2::zeros((
                p_marg, p_marg,
            )))],
            nullspace_dims: vec![0],
            initial_log_lambdas: Array1::zeros(1),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "logslope".to_string(),
            design: log_design,
            offset: Array1::zeros(n),
            penalties: vec![crate::custom_family::PenaltyMatrix::Dense(Array2::zeros((
                p_log, p_log,
            )))],
            nullspace_dims: vec![0],
            initial_log_lambdas: Array1::zeros(1),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];

    let p_total = (p_marg + p_log) as u64;
    let expected_joint = (n as u64) * p_total * p_total;
    let expected_block_diag = (n as u64) * ((p_marg * p_marg + p_log * p_log) as u64);
    assert_eq!(family.coefficient_hessian_cost(&specs), expected_joint);
    assert_eq!(
        default_coefficient_hessian_cost(&specs),
        expected_block_diag
    );
    assert!(family.coefficient_hessian_cost(&specs) > default_coefficient_hessian_cost(&specs));
}

#[test]
fn rigid_fast_path_matches_loglik_finite_differences() {
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![1.0]),
        weights: Arc::new(array![1.2]),
        z: Arc::new(array![0.3]),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0]
        ])),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0]
        ])),
        ..default_test_family()
    };
    let states_at = |q: f64, g: f64| {
        vec![
            ParameterBlockState {
                beta: array![q],
                eta: array![q],
            },
            ParameterBlockState {
                beta: array![g],
                eta: array![g],
            },
        ]
    };
    let q = 0.4;
    let g = 0.7;
    let block_states = states_at(q, g);
    let eval = family
        .evaluate(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "rigid family evaluation", e));
    let grad_q = match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        BlockWorkingSet::Diagonal { .. } => {
            panic!("expected exact-newton marginal block")
        }
    };
    let grad_g = match &eval.blockworking_sets[1] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        BlockWorkingSet::Diagonal { .. } => {
            panic!("expected exact-newton log-slope block")
        }
    };
    let hess_qq = match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { hessian, .. } => match hessian {
            SymmetricMatrix::Dense(h) => h[[0, 0]],
            _ => panic!("expected dense marginal Hessian"),
        },
        BlockWorkingSet::Diagonal { .. } => {
            panic!("expected exact-newton marginal block")
        }
    };
    let hess_gg = match &eval.blockworking_sets[1] {
        BlockWorkingSet::ExactNewton { hessian, .. } => match hessian {
            SymmetricMatrix::Dense(h) => h[[0, 0]],
            _ => panic!("expected dense log-slope Hessian"),
        },
        BlockWorkingSet::Diagonal { .. } => {
            panic!("expected exact-newton log-slope block")
        }
    };

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "rigid exact eval cache", e));
    let row_ctx = family
        .build_row_exact_context_with_stats_and_cell_cache(0, &block_states, None, true)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "rigid row context", e));
    let (_, primary_grad, primary_hess) = family
        .compute_row_primary_gradient_hessian(0, &block_states, &cache.primary, &row_ctx)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "rigid exact row derivatives", e));
    let expected_score_q =
        BernoulliMarginalSlopeFamily::exact_newton_score_component_from_objective_gradient(
            primary_grad[0],
        );
    let expected_score_g =
        BernoulliMarginalSlopeFamily::exact_newton_score_component_from_objective_gradient(
            primary_grad[1],
        );

    assert!(
        (grad_q - expected_score_q).abs() < 1e-10,
        "marginal gradient mismatch: fast={grad_q:.12e}, exact={expected_score_q:.12e}"
    );
    assert!(
        (grad_g - expected_score_g).abs() < 1e-10,
        "logslope gradient mismatch: fast={grad_g:.12e}, exact={expected_score_g:.12e}"
    );
    assert!(
        (hess_qq - primary_hess[[0, 0]]).abs() < 1e-10,
        "marginal Hessian mismatch: fast={hess_qq:.12e}, exact={:.12e}",
        primary_hess[[0, 0]]
    );
    assert!(
        (hess_gg - primary_hess[[1, 1]]).abs() < 1e-10,
        "logslope Hessian mismatch: fast={hess_gg:.12e}, exact={:.12e}",
        primary_hess[[1, 1]]
    );
}

/// Exercises the w-only (link_dev without score_warp) layout through the
/// full gradient + Hessian path, verifying that:
///   (a) no index-out-of-bounds panic occurs,
///   (b) all outputs are finite,
///   (c) the Hessian is symmetric.
///
/// This guards against arity bookkeeping bugs where the directional-jet or
/// block-slice code assumes both h and w blocks are present.
#[test]
fn w_only_gradient_hessian_finite_and_symmetric() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_test_link_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link deviation block", e));
    let link_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("link initial beta")
        .len();
    // Non-trivial link coefficients to exercise all jet branches.
    let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0)));

    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        link_dev: Some(prepared.runtime.clone()),
        ..default_test_family()
    };

    // Three blocks: marginal (dim 0), logslope (dim 0), link_dev.
    // eta is irrelevant for zero-column designs; use zeros.
    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(beta_link.clone(), seed.len()),
    ];

    let slices = block_slices(&family);
    assert!(slices.h.is_none(), "score-warp absent → no h slice");
    let primary = primary_slices(&slices);
    assert!(primary.h.is_none(), "primary h absent");
    assert_eq!(primary.total, 2 + link_dim);

    // Exercise every row — different z values exercise different link
    // regimes (negative tail, near zero, positive tail).
    for row in 0..seed.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));

        let (_, grad, hess) = family
            .compute_row_primary_gradient_hessian(row, &block_states, &primary, &row_ctx)
            .unwrap_or_else(|e| {
                panic!("row {row}: compute_row_primary_gradient_hessian failed: {e}")
            });

        assert_eq!(
            grad.len(),
            primary.total,
            "row {row}: gradient length mismatch"
        );
        assert_eq!(
            hess.dim(),
            (primary.total, primary.total),
            "row {row}: hessian shape mismatch"
        );
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "row {row}: non-finite gradient entry: {grad:?}"
        );
        assert!(
            hess.iter().all(|v| v.is_finite()),
            "row {row}: non-finite hessian entry"
        );

        // Symmetry check.
        for a in 0..primary.total {
            for b in 0..a {
                let diff = (hess[[a, b]] - hess[[b, a]]).abs();
                assert!(
                    diff < 1e-10,
                    "row {row}: hessian asymmetry at ({a},{b}): \
                         H[{a},{b}]={:.6e} vs H[{b},{a}]={:.6e}, diff={diff:.3e}",
                    hess[[a, b]],
                    hess[[b, a]]
                );
            }
        }
    }
}

#[test]
fn h_only_gradient_hessian_finite_and_symmetric() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score-warp block", e));
    let score_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("score-warp initial beta")
        .len();
    let beta_score = Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0)));

    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        score_warp: Some(prepared.runtime.clone()),
        ..default_test_family()
    };

    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(beta_score, seed.len()),
    ];

    let slices = block_slices(&family);
    assert!(slices.w.is_none(), "link-dev absent → no w slice");
    let primary = primary_slices(&slices);
    assert!(primary.w.is_none(), "primary w absent");
    assert_eq!(primary.total, 2 + score_dim);

    for row in 0..seed.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));

        let (_, grad, hess) = family
            .compute_row_primary_gradient_hessian(row, &block_states, &primary, &row_ctx)
            .unwrap_or_else(|e| {
                panic!("row {row}: compute_row_primary_gradient_hessian failed: {e}")
            });

        assert_eq!(
            grad.len(),
            primary.total,
            "row {row}: gradient length mismatch"
        );
        assert_eq!(
            hess.dim(),
            (primary.total, primary.total),
            "row {row}: hessian shape mismatch"
        );
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "row {row}: non-finite gradient entry: {grad:?}"
        );
        assert!(
            hess.iter().all(|v| v.is_finite()),
            "row {row}: non-finite hessian entry"
        );

        for a in 0..primary.total {
            for b in 0..a {
                let diff = (hess[[a, b]] - hess[[b, a]]).abs();
                assert!(
                    diff < 1e-10,
                    "row {row}: hessian asymmetry at ({a},{b}): \
                         H[{a},{b}]={:.6e} vs H[{b},{a}]={:.6e}, diff={diff:.3e}",
                    hess[[a, b]],
                    hess[[b, a]]
                );
            }
        }
    }
}

#[test]
fn w_only_exact_outer_directional_derivatives_are_present_and_finite() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_test_link_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link deviation block", e));
    let link_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("link initial beta")
        .len();
    let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0)));

    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        link_dev: Some(prepared.runtime.clone()),
        ..default_test_family()
    };
    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(beta_link, seed.len()),
    ];

    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    let w_range = slices.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.15;
    if w_range.len() > 1 {
        dir_u[w_range.start + 1] = -0.07;
    }

    dir_v[w_range.start] = 0.09;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = 0.03;
    }

    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "w-only third directional derivative", e))
        .expect("w-only third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "w-only fourth directional derivative", e))
        .expect("w-only fourth directional derivative matrix");

    assert_eq!(third.dim(), (total, total));
    assert_eq!(fourth.dim(), (total, total));
    assert!(third.iter().all(|value| value.is_finite()));
    assert!(fourth.iter().all(|value| value.is_finite()));
    let max_abs_third = third
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_abs_fourth = fourth
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        max_abs_third > 1e-10,
        "expected nonzero w-only third directional derivative"
    );
    assert!(
        max_abs_fourth > 1e-10,
        "expected nonzero w-only fourth directional derivative"
    );

    for i in 0..total {
        for j in 0..i {
            assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
            assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
        }
    }
}

#[test]
fn h_only_exact_outer_directional_derivatives_are_present_and_finite() {
    // ROOT CAUSE (pre-4250aa07 vs current).  The pre-refactor `block_slices`
    // sized its marginal/logslope slices from `states[block_idx].beta.len()`
    // (1 each from `dummy_block_state(array![0.0], ...)`), giving a 3-slot
    // block-space {marginal, logslope, h...} even though the design matrices
    // were zero-width.  After the refactor, `block_slices` sizes slices from
    // `design.ncols()`, so zero-width designs collapse marginal and logslope
    // to empty ranges.  Any write to `dir_u[slices.marginal.start]` then
    // aliases `dir_u[slices.logslope.start]` and `dir_u[h_range.start]`,
    // which is why the refactor dropped those writes — but that also
    // dropped the only path by which `exact_newton_joint_hessian_directional_derivative`
    // (which maps block-space direction → primary-space direction via
    // `marginal_design·β_marginal` → `row_dir[primary.q]` and
    // `logslope_design·β_logslope` → `row_dir[primary.logslope]`) could
    // inject nonzero q / logslope components.
    //
    // At the current state (b ≡ block_states[1].eta[row] = 0 and pure
    // h-only block-space direction), the third directional derivative is
    // structurally zero:
    //
    //   In the flex score-warp path, the h-block contribution to the
    //   observed cell coefficient is
    //       coeff_u[idx_h] = s · score_basis_cell_coefficients(basis_span, b)
    //                      = s · [b·h0, b·h1, b·h2, b·h3]   (cubic_cell_kernel.rs:815)
    //   At b = 0 this vanishes, and the only other h-index term —
    //   coeff_bu[idx_h] = s · [h0, h1, h2, h3] — is reached via
    //   `param_directional_from_b_family` only when `dir[primary.logslope] ≠ 0`
    //   (bernoulli_marginal_slope.rs:1799-1815).  With row_dir[q] and
    //   row_dir[logslope] both zero, every directional contraction
    //   (coeff_dir, coeff_a_dir, coeff_aa_dir, coeff_u_dir[u], coeff_au_dir[u],
    //   pair_directional_from_bb_family(...)) collapses to zero.
    //   Therefore `max_abs_third > 1e-10` is analytically impossible.
    //
    // PRINCIPLED FIX.  Restore the block-space geometry the test is
    // actually designed to exercise: give marginal_design and logslope_design
    // single columns of ones (the canonical "scalar" parameterisation used
    // by the sigma FD test at rs:13436), set block_states so row-wise
    // q_internal and b are at typical nondegenerate values, and populate
    // dir_u / dir_v on all three blocks.  The test name was misleading:
    // "h-only" originally referred to the score-warp (h) BLOCK being the
    // only flex block configured (link_dev: None) — not to the direction
    // being supported solely on h.  This fix preserves that original
    // semantic and exercises the block→primary direction map end-to-end.
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score-warp block", e));
    let score_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("score-warp initial beta")
        .len();
    let beta_score = Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0)));
    let scalar_design = || {
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::from_elem((seed.len(), 1), 1.0),
        ))
    };

    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: scalar_design(),
        logslope_design: scalar_design(),
        score_warp: Some(prepared.runtime.clone()),
        ..default_test_family()
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.25],
            eta: Array1::from_elem(seed.len(), 0.25),
        },
        ParameterBlockState {
            beta: array![0.15],
            eta: Array1::from_elem(seed.len(), 0.15),
        },
        ParameterBlockState {
            beta: beta_score,
            eta: Array1::zeros(seed.len()),
        },
    ];

    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[slices.marginal.start] = -0.35;
    dir_u[slices.logslope.start] = 0.28;
    let h_range = slices.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.12;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.06;
    }

    dir_v[slices.marginal.start] = 0.18;
    dir_v[slices.logslope.start] = -0.22;
    dir_v[h_range.start] = 0.07;
    if h_range.len() > 1 {
        dir_v[h_range.start + 1] = 0.05;
    }

    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "h-only third directional derivative", e))
        .expect("h-only third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "h-only fourth directional derivative", e))
        .expect("h-only fourth directional derivative matrix");

    assert_eq!(third.dim(), (total, total));
    assert_eq!(fourth.dim(), (total, total));
    assert!(third.iter().all(|value| value.is_finite()));
    assert!(fourth.iter().all(|value| value.is_finite()));
    let max_abs_third = third
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_abs_fourth = fourth
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        max_abs_third > 1e-10,
        "expected nonzero h-only third directional derivative"
    );
    assert!(
        max_abs_fourth > 1e-10,
        "expected nonzero h-only fourth directional derivative"
    );

    for i in 0..total {
        for j in 0..i {
            assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
            assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
        }
    }
}

#[test]
fn h_only_row_primary_higher_order_contractions_are_finite_and_symmetric() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_score_warp_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build score-warp block", e));
    let score_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("score-warp initial beta")
        .len();
    let beta_score = Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0)));

    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        score_warp: Some(prepared.runtime.clone()),
        ..default_test_family()
    };

    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(beta_score, seed.len()),
    ];

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = -0.35;
    dir_u[cache.primary.logslope] = 0.28;
    let h_range = cache.primary.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.12;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.06;
    }

    dir_v[cache.primary.q] = 0.18;
    dir_v[cache.primary.logslope] = -0.22;
    dir_v[h_range.start] = 0.07;
    if h_range.len() > 1 {
        dir_v[h_range.start + 1] = 0.05;
    }

    let mut max_abs_third = 0.0_f64;
    let mut max_abs_fourth = 0.0_f64;
    for row in 0..seed.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_third_contracted failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_u, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_fourth_contracted failed: {e}"));

        assert_eq!(third.dim(), (total, total));
        assert_eq!(fourth.dim(), (total, total));
        assert!(third.iter().all(|value| value.is_finite()));
        assert!(fourth.iter().all(|value| value.is_finite()));
        max_abs_third = max_abs_third.max(
            third
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        );
        max_abs_fourth = max_abs_fourth.max(
            fourth
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        );

        for i in 0..total {
            for j in 0..i {
                assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
            }
        }
    }
    assert!(
        max_abs_third > 1e-10,
        "expected nonzero h-only third contraction"
    );
    assert!(
        max_abs_fourth > 1e-10,
        "expected nonzero h-only fourth contraction"
    );
}

#[test]
fn w_only_row_primary_higher_order_contractions_are_finite_and_symmetric() {
    let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
    let prepared = build_test_link_deviation_block_from_seed(
        &seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build link deviation block", e));
    let link_dim = prepared
        .block
        .initial_beta
        .as_ref()
        .expect("link initial beta")
        .len();
    let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0)));

    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
        weights: Arc::new(Array1::ones(seed.len())),
        z: Arc::new(seed.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::zeros((seed.len(), 0)),
        )),
        link_dev: Some(prepared.runtime.clone()),
        ..default_test_family()
    };

    let block_states = vec![
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(array![0.0], seed.len()),
        dummy_block_state(beta_link, seed.len()),
    ];

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = 0.4;
    dir_u[cache.primary.logslope] = -0.3;
    let w_range = cache.primary.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.15;
    if w_range.len() > 1 {
        dir_u[w_range.start + 1] = -0.07;
    }

    dir_v[cache.primary.q] = -0.2;
    dir_v[cache.primary.logslope] = 0.25;
    dir_v[w_range.start] = 0.09;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = 0.03;
    }

    let mut max_abs_third = 0.0_f64;
    let mut max_abs_fourth = 0.0_f64;
    for row in 0..seed.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_third_contracted failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_u, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_fourth_contracted failed: {e}"));

        assert_eq!(third.dim(), (total, total));
        assert_eq!(fourth.dim(), (total, total));
        assert!(third.iter().all(|value| value.is_finite()));
        assert!(fourth.iter().all(|value| value.is_finite()));
        max_abs_third = max_abs_third.max(
            third
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        );
        max_abs_fourth = max_abs_fourth.max(
            fourth
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        );

        for i in 0..total {
            for j in 0..i {
                assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
            }
        }
    }
    assert!(
        max_abs_third > 1e-10,
        "expected nonzero w-only third contraction"
    );
    assert!(
        max_abs_fourth > 1e-10,
        "expected nonzero w-only fourth contraction"
    );
}

#[test]
fn dual_flex_row_primary_higher_order_contractions_are_finite_and_symmetric() {
    let (family, block_states) = dual_flex_exact_fixture();

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = 0.7;
    dir_u[cache.primary.logslope] = -0.2;
    let h_range = cache.primary.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = cache.primary.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[cache.primary.q] = -0.4;
    dir_v[cache.primary.logslope] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    let mut max_abs_third = 0.0_f64;
    let mut max_abs_fourth = 0.0_f64;
    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_third_contracted failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_u, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_fourth_contracted failed: {e}"));

        assert_eq!(third.dim(), (total, total));
        assert_eq!(fourth.dim(), (total, total));
        assert!(third.iter().all(|value| value.is_finite()));
        assert!(fourth.iter().all(|value| value.is_finite()));
        max_abs_third = max_abs_third.max(
            third
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        );
        max_abs_fourth = max_abs_fourth.max(
            fourth
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        );

        for i in 0..total {
            for j in 0..i {
                assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
            }
        }
    }
    assert!(
        max_abs_third > 1e-10,
        "expected nonzero dual-flex third contraction"
    );
    assert!(
        max_abs_fourth > 1e-10,
        "expected nonzero dual-flex fourth contraction"
    );
}

#[test]
fn dual_flex_row_primary_higher_order_zero_direction_returns_zero() {
    let (family, block_states) = dual_flex_exact_fixture();

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let zero = Array1::<f64>::zeros(cache.primary.total);
    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &zero)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_third_contracted failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &zero, &zero)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_fourth_contracted failed: {e}"));

        assert!(
            third.iter().all(|value| value.abs() <= 0.0),
            "row {row}: expected zero third contraction for zero direction"
        );
        assert!(
            fourth.iter().all(|value| value.abs() <= 0.0),
            "row {row}: expected zero fourth contraction for zero directions"
        );
    }
}

#[test]
fn h_only_row_primary_higher_order_zero_direction_returns_zero() {
    let (family, block_states) = h_only_exact_fixture();
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let zero = Array1::<f64>::zeros(cache.primary.total);
    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &zero)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_third_contracted failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &zero, &zero)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_fourth_contracted failed: {e}"));

        assert!(
            third.iter().all(|value| value.abs() <= 0.0),
            "row {row}: expected zero h-only third contraction for zero direction"
        );
        assert!(
            fourth.iter().all(|value| value.abs() <= 0.0),
            "row {row}: expected zero h-only fourth contraction for zero directions"
        );
    }
}

#[test]
fn w_only_row_primary_higher_order_zero_direction_returns_zero() {
    let (family, block_states) = w_only_exact_fixture();
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let zero = Array1::<f64>::zeros(cache.primary.total);
    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &zero)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_third_contracted failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &zero, &zero)
            .unwrap_or_else(|e| panic!("row {row}: row_primary_fourth_contracted failed: {e}"));

        assert!(
            third.iter().all(|value| value.abs() <= 0.0),
            "row {row}: expected zero w-only third contraction for zero direction"
        );
        assert!(
            fourth.iter().all(|value| value.abs() <= 0.0),
            "row {row}: expected zero w-only fourth contraction for zero directions"
        );
    }
}

#[test]
fn dual_flex_exact_outer_zero_direction_returns_zero() {
    let (family, block_states) = dual_flex_exact_fixture();

    let slices = block_slices(&family);
    let zero = Array1::<f64>::zeros(slices.total);
    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &zero)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex third directional derivative", e
            )
        })
        .expect("dual-flex third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &zero, &zero)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex fourth directional derivative", e
            )
        })
        .expect("dual-flex fourth directional derivative matrix");

    assert!(
        third.iter().all(|value| value.abs() <= 0.0),
        "expected zero dual-flex third directional derivative for zero direction"
    );
    assert!(
        fourth.iter().all(|value| value.abs() <= 0.0),
        "expected zero dual-flex fourth directional derivative for zero directions"
    );
}

#[test]
fn dual_flex_exact_outer_fourth_direction_swap_is_symmetric() {
    let (family, block_states) = dual_flex_exact_fixture();

    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[slices.marginal.start] = 0.7;
    dir_u[slices.logslope.start] = -0.2;
    let h_range = slices.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = slices.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[slices.marginal.start] = -0.4;
    dir_v[slices.logslope.start] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    let forward = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex fourth directional derivative", e
            )
        })
        .expect("dual-flex fourth directional derivative matrix");
    let swapped = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_v, &dir_u)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex swapped fourth directional derivative", e
            )
        })
        .expect("dual-flex swapped fourth directional derivative matrix");

    assert_eq!(forward.dim(), (total, total));
    assert_eq!(swapped.dim(), (total, total));
    for i in 0..total {
        for j in 0..total {
            assert!(
                (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                "fourth directional derivative should be symmetric in direction arguments at ({i},{j})"
            );
        }
    }
}

#[test]
fn dual_flex_row_primary_fourth_direction_swap_is_symmetric() {
    let (family, block_states) = dual_flex_exact_fixture();

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = 0.7;
    dir_u[cache.primary.logslope] = -0.2;
    let h_range = cache.primary.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = cache.primary.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[cache.primary.q] = -0.4;
    dir_v[cache.primary.logslope] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let forward = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_u, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: forward fourth contraction failed: {e}"));
        let swapped = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_v, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: swapped fourth contraction failed: {e}"));

        assert_eq!(forward.dim(), (total, total));
        assert_eq!(swapped.dim(), (total, total));
        for i in 0..total {
            for j in 0..total {
                assert!(
                    (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                    "row {row}: fourth contraction should be symmetric in direction arguments at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn dual_flex_row_primary_higher_order_direction_sign_rules_hold() {
    let (family, block_states) = dual_flex_exact_fixture();

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = 0.7;
    dir_u[cache.primary.logslope] = -0.2;
    let h_range = cache.primary.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = cache.primary.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[cache.primary.q] = -0.4;
    dir_v[cache.primary.logslope] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    let neg_dir_u = dir_u.mapv(|value| -value);
    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: third contraction failed: {e}"));
        let third_neg = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &neg_dir_u)
            .unwrap_or_else(|e| panic!("row {row}: negated third contraction failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_u, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: fourth contraction failed: {e}"));
        let fourth_neg_u = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &neg_dir_u, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: negated-u fourth contraction failed: {e}"));

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                    "row {row}: third contraction should be odd in its direction at ({i},{j})"
                );
                assert!(
                    (fourth_neg_u[[i, j]] + fourth[[i, j]]).abs() < 1e-8,
                    "row {row}: fourth contraction should be linear in dir_u sign at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn h_only_row_primary_fourth_direction_swap_is_symmetric() {
    let (family, block_states) = h_only_exact_fixture();
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = -0.35;
    dir_u[cache.primary.logslope] = 0.28;
    let h_range = cache.primary.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.12;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.06;
    }

    dir_v[cache.primary.q] = 0.18;
    dir_v[cache.primary.logslope] = -0.22;
    dir_v[h_range.start] = 0.07;
    if h_range.len() > 1 {
        dir_v[h_range.start + 1] = 0.05;
    }

    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let forward = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_u, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: forward fourth contraction failed: {e}"));
        let swapped = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_v, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: swapped fourth contraction failed: {e}"));

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                    "row {row}: h-only fourth contraction should be symmetric in direction arguments at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn w_only_row_primary_fourth_direction_swap_is_symmetric() {
    let (family, block_states) = w_only_exact_fixture();
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = 0.4;
    dir_u[cache.primary.logslope] = -0.3;
    let w_range = cache.primary.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.15;
    if w_range.len() > 1 {
        dir_u[w_range.start + 1] = -0.07;
    }

    dir_v[cache.primary.q] = -0.2;
    dir_v[cache.primary.logslope] = 0.25;
    dir_v[w_range.start] = 0.09;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = 0.03;
    }

    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let forward = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_u, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: forward fourth contraction failed: {e}"));
        let swapped = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir_v, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: swapped fourth contraction failed: {e}"));

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                    "row {row}: w-only fourth contraction should be symmetric in direction arguments at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn h_only_row_primary_higher_order_direction_sign_rules_hold() {
    let (family, block_states) = h_only_exact_fixture();
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir = Array1::<f64>::zeros(total);
    dir[cache.primary.q] = -0.35;
    dir[cache.primary.logslope] = 0.28;
    let h_range = cache.primary.h.as_ref().expect("h slice");
    dir[h_range.start] = 0.12;
    if h_range.len() > 1 {
        dir[h_range.start + 1] = -0.06;
    }
    let neg_dir = dir.mapv(|value| -value);

    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir)
            .unwrap_or_else(|e| panic!("row {row}: third contraction failed: {e}"));
        let third_neg = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &neg_dir)
            .unwrap_or_else(|e| panic!("row {row}: negated third contraction failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir, &dir)
            .unwrap_or_else(|e| panic!("row {row}: fourth contraction failed: {e}"));
        let fourth_neg = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &neg_dir, &neg_dir)
            .unwrap_or_else(|e| panic!("row {row}: doubly-negated fourth contraction failed: {e}"));

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                    "row {row}: h-only third contraction should be odd at ({i},{j})"
                );
                assert!(
                    (fourth_neg[[i, j]] - fourth[[i, j]]).abs() < 1e-8,
                    "row {row}: h-only fourth contraction should be invariant under flipping both directions at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn w_only_row_primary_higher_order_direction_sign_rules_hold() {
    let (family, block_states) = w_only_exact_fixture();
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir = Array1::<f64>::zeros(total);
    dir[cache.primary.q] = 0.4;
    dir[cache.primary.logslope] = -0.3;
    let w_range = cache.primary.w.as_ref().expect("w slice");
    dir[w_range.start] = 0.15;
    if w_range.len() > 1 {
        dir[w_range.start + 1] = -0.07;
    }
    let neg_dir = dir.mapv(|value| -value);

    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir)
            .unwrap_or_else(|e| panic!("row {row}: third contraction failed: {e}"));
        let third_neg = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &neg_dir)
            .unwrap_or_else(|e| panic!("row {row}: negated third contraction failed: {e}"));
        let fourth = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &dir, &dir)
            .unwrap_or_else(|e| panic!("row {row}: fourth contraction failed: {e}"));
        let fourth_neg = family
            .row_primary_fourth_contracted(row, &block_states, &cache, &row_ctx, &neg_dir, &neg_dir)
            .unwrap_or_else(|e| panic!("row {row}: doubly-negated fourth contraction failed: {e}"));

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                    "row {row}: w-only third contraction should be odd at ({i},{j})"
                );
                assert!(
                    (fourth_neg[[i, j]] - fourth[[i, j]]).abs() < 1e-8,
                    "row {row}: w-only fourth contraction should be invariant under flipping both directions at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn dual_flex_exact_outer_direction_sign_rules_hold() {
    let (family, block_states) = dual_flex_exact_fixture();

    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[slices.marginal.start] = 0.7;
    dir_u[slices.logslope.start] = -0.2;
    let h_range = slices.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = slices.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[slices.marginal.start] = -0.4;
    dir_v[slices.logslope.start] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    let neg_dir_u = dir_u.mapv(|value| -value);
    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex third directional derivative", e
            )
        })
        .expect("dual-flex third directional derivative matrix");
    let third_neg = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &neg_dir_u)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex negated third directional derivative", e
            )
        })
        .expect("dual-flex negated third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex fourth directional derivative", e
            )
        })
        .expect("dual-flex fourth directional derivative matrix");
    let fourth_neg_u = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &neg_dir_u, &dir_v)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex negated-u fourth directional derivative", e
            )
        })
        .expect("dual-flex negated-u fourth directional derivative matrix");

    assert_eq!(third.dim(), (total, total));
    assert_eq!(third_neg.dim(), (total, total));
    assert_eq!(fourth.dim(), (total, total));
    assert_eq!(fourth_neg_u.dim(), (total, total));
    for i in 0..total {
        for j in 0..total {
            assert!(
                (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                "third directional derivative should be odd in its direction at ({i},{j})"
            );
            assert!(
                (fourth_neg_u[[i, j]] + fourth[[i, j]]).abs() < 1e-8,
                "fourth directional derivative should be linear in dir_u sign at ({i},{j})"
            );
        }
    }
}

#[test]
fn dual_flex_exact_outer_fourth_double_sign_flip_is_invariant() {
    let (family, block_states) = dual_flex_exact_fixture();

    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[slices.marginal.start] = 0.7;
    dir_u[slices.logslope.start] = -0.2;
    let h_range = slices.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = slices.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[slices.marginal.start] = -0.4;
    dir_v[slices.logslope.start] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    let neg_dir_u = dir_u.mapv(|value| -value);
    let neg_dir_v = dir_v.mapv(|value| -value);
    let forward = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex fourth directional derivative", e
            )
        })
        .expect("dual-flex fourth directional derivative matrix");
    let flipped = family
        .exact_newton_joint_hessiansecond_directional_derivative(
            &block_states,
            &neg_dir_u,
            &neg_dir_v,
        )
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex doubly-negated fourth directional derivative", e
            )
        })
        .expect("dual-flex doubly-negated fourth directional derivative matrix");

    assert_eq!(forward.dim(), (total, total));
    assert_eq!(flipped.dim(), (total, total));
    for i in 0..total {
        for j in 0..total {
            assert!(
                (forward[[i, j]] - flipped[[i, j]]).abs() < 1e-8,
                "fourth directional derivative should be invariant under flipping both directions at ({i},{j})"
            );
        }
    }
}

#[test]
fn dual_flex_exact_outer_third_direction_is_linear() {
    let (family, block_states) = dual_flex_exact_fixture();

    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[slices.marginal.start] = 0.7;
    dir_u[slices.logslope.start] = -0.2;
    let h_range = slices.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = slices.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[slices.marginal.start] = -0.4;
    dir_v[slices.logslope.start] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    let dir_sum = &dir_u + &dir_v;
    let third_u = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex third directional derivative u", e
            )
        })
        .expect("dual-flex third directional derivative u matrix");
    let third_v = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_v)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex third directional derivative v", e
            )
        })
        .expect("dual-flex third directional derivative v matrix");
    let third_sum = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_sum)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex third directional derivative sum", e
            )
        })
        .expect("dual-flex third directional derivative sum matrix");

    for i in 0..total {
        for j in 0..total {
            let expected = third_u[[i, j]] + third_v[[i, j]];
            assert!(
                (third_sum[[i, j]] - expected).abs() < 1e-8,
                "third directional derivative should be linear in its direction at ({i},{j})"
            );
        }
    }
}

#[test]
fn dual_flex_row_primary_third_direction_is_linear() {
    let (family, block_states) = dual_flex_exact_fixture();

    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = 0.7;
    dir_u[cache.primary.logslope] = -0.2;
    let h_range = cache.primary.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = cache.primary.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[cache.primary.q] = -0.4;
    dir_v[cache.primary.logslope] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    let dir_sum = &dir_u + &dir_v;
    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third_u = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: third contraction u failed: {e}"));
        let third_v = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: third contraction v failed: {e}"));
        let third_sum = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_sum)
            .unwrap_or_else(|e| panic!("row {row}: third contraction sum failed: {e}"));

        for i in 0..total {
            for j in 0..total {
                let expected = third_u[[i, j]] + third_v[[i, j]];
                assert!(
                    (third_sum[[i, j]] - expected).abs() < 1e-8,
                    "row {row}: third contraction should be linear in its direction at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn h_only_row_primary_third_direction_is_linear() {
    let (family, block_states) = h_only_exact_fixture();
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = -0.35;
    dir_u[cache.primary.logslope] = 0.28;
    let h_range = cache.primary.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.12;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.06;
    }

    dir_v[cache.primary.q] = 0.18;
    dir_v[cache.primary.logslope] = -0.22;
    dir_v[h_range.start] = 0.07;
    if h_range.len() > 1 {
        dir_v[h_range.start + 1] = 0.05;
    }

    let dir_sum = &dir_u + &dir_v;
    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third_u = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: third contraction u failed: {e}"));
        let third_v = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: third contraction v failed: {e}"));
        let third_sum = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_sum)
            .unwrap_or_else(|e| panic!("row {row}: third contraction sum failed: {e}"));

        for i in 0..total {
            for j in 0..total {
                let expected = third_u[[i, j]] + third_v[[i, j]];
                assert!(
                    (third_sum[[i, j]] - expected).abs() < 1e-8,
                    "row {row}: h-only third contraction should be linear at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn w_only_row_primary_third_direction_is_linear() {
    let (family, block_states) = w_only_exact_fixture();
    let cache = family
        .build_exact_eval_cache(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let total = cache.primary.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[cache.primary.q] = 0.4;
    dir_u[cache.primary.logslope] = -0.3;
    let w_range = cache.primary.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.15;
    if w_range.len() > 1 {
        dir_u[w_range.start + 1] = -0.07;
    }

    dir_v[cache.primary.q] = -0.2;
    dir_v[cache.primary.logslope] = 0.25;
    dir_v[w_range.start] = 0.09;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = 0.03;
    }

    let dir_sum = &dir_u + &dir_v;
    for row in 0..family.z.len() {
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &block_states, None, true)
            .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
        let third_u = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_u)
            .unwrap_or_else(|e| panic!("row {row}: third contraction u failed: {e}"));
        let third_v = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_v)
            .unwrap_or_else(|e| panic!("row {row}: third contraction v failed: {e}"));
        let third_sum = family
            .row_primary_third_contracted(row, &block_states, &cache, &row_ctx, &dir_sum)
            .unwrap_or_else(|e| panic!("row {row}: third contraction sum failed: {e}"));

        for i in 0..total {
            for j in 0..total {
                let expected = third_u[[i, j]] + third_v[[i, j]];
                assert!(
                    (third_sum[[i, j]] - expected).abs() < 1e-8,
                    "row {row}: w-only third contraction should be linear at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn dual_flex_exact_outer_fourth_first_direction_is_linear() {
    let (family, block_states) = dual_flex_exact_fixture();

    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    let mut dir_w = Array1::<f64>::zeros(total);
    dir_u[slices.marginal.start] = 0.7;
    dir_u[slices.logslope.start] = -0.2;
    let h_range = slices.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.1;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.05;
    }
    let w_range = slices.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.08;

    dir_v[slices.marginal.start] = -0.4;
    dir_v[slices.logslope.start] = 0.3;
    dir_v[h_range.start] = -0.03;
    dir_v[w_range.start] = 0.06;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = -0.02;
    }

    dir_w[slices.marginal.start] = 0.11;
    dir_w[slices.logslope.start] = -0.09;
    dir_w[h_range.start] = 0.04;
    dir_w[w_range.start] = -0.05;

    let dir_sum = &dir_u + &dir_v;
    let fourth_u = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_w)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex fourth directional derivative u,w", e
            )
        })
        .expect("dual-flex fourth directional derivative u,w matrix");
    let fourth_v = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_v, &dir_w)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex fourth directional derivative v,w", e
            )
        })
        .expect("dual-flex fourth directional derivative v,w matrix");
    let fourth_sum = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_sum, &dir_w)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "dual-flex fourth directional derivative (u+v),w", e
            )
        })
        .expect("dual-flex fourth directional derivative (u+v),w matrix");

    for i in 0..total {
        for j in 0..total {
            let expected = fourth_u[[i, j]] + fourth_v[[i, j]];
            assert!(
                (fourth_sum[[i, j]] - expected).abs() < 1e-8,
                "fourth directional derivative should be linear in its first direction at ({i},{j})"
            );
        }
    }
}

#[test]
fn h_only_exact_outer_third_direction_is_linear() {
    let (family, block_states) = h_only_exact_fixture();
    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    let h_range = slices.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.12;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.06;
    }

    dir_v[h_range.start] = 0.07;
    if h_range.len() > 1 {
        dir_v[h_range.start + 1] = 0.05;
    }

    let dir_sum = &dir_u + &dir_v;
    let third_u = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "h-only third directional derivative u", e
            )
        })
        .expect("h-only third directional derivative u matrix");
    let third_v = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_v)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "h-only third directional derivative v", e
            )
        })
        .expect("h-only third directional derivative v matrix");
    let third_sum = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_sum)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "h-only third directional derivative sum", e
            )
        })
        .expect("h-only third directional derivative sum matrix");

    for i in 0..total {
        for j in 0..total {
            let expected = third_u[[i, j]] + third_v[[i, j]];
            assert!(
                (third_sum[[i, j]] - expected).abs() < 1e-8,
                "h-only third directional derivative should be linear at ({i},{j})"
            );
        }
    }
}

#[test]
fn w_only_exact_outer_third_direction_is_linear() {
    let (family, block_states) = w_only_exact_fixture();
    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    let w_range = slices.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.15;
    if w_range.len() > 1 {
        dir_u[w_range.start + 1] = -0.07;
    }

    dir_v[w_range.start] = 0.09;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = 0.03;
    }

    let dir_sum = &dir_u + &dir_v;
    let third_u = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "w-only third directional derivative u", e
            )
        })
        .expect("w-only third directional derivative u matrix");
    let third_v = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_v)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "w-only third directional derivative v", e
            )
        })
        .expect("w-only third directional derivative v matrix");
    let third_sum = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_sum)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "w-only third directional derivative sum", e
            )
        })
        .expect("w-only third directional derivative sum matrix");

    for i in 0..total {
        for j in 0..total {
            let expected = third_u[[i, j]] + third_v[[i, j]];
            assert!(
                (third_sum[[i, j]] - expected).abs() < 1e-8,
                "w-only third directional derivative should be linear at ({i},{j})"
            );
        }
    }
}

#[test]
fn h_only_exact_outer_direction_sign_rules_hold() {
    let (family, block_states) = h_only_exact_fixture();
    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir = Array1::<f64>::zeros(total);
    let h_range = slices.h.as_ref().expect("h slice");
    dir[h_range.start] = 0.12;
    if h_range.len() > 1 {
        dir[h_range.start + 1] = -0.06;
    }
    let neg_dir = dir.mapv(|value| -value);

    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "h-only third directional derivative", e))
        .expect("h-only third directional derivative matrix");
    let third_neg = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &neg_dir)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "h-only negated third directional derivative", e
            )
        })
        .expect("h-only negated third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir, &dir)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "h-only fourth directional derivative", e))
        .expect("h-only fourth directional derivative matrix");
    let fourth_neg = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &neg_dir, &neg_dir)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "h-only doubly-negated fourth directional derivative", e
            )
        })
        .expect("h-only doubly-negated fourth directional derivative matrix");

    for i in 0..total {
        for j in 0..total {
            assert!(
                (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                "h-only third directional derivative should be odd at ({i},{j})"
            );
            assert!(
                (fourth_neg[[i, j]] - fourth[[i, j]]).abs() < 1e-8,
                "h-only fourth directional derivative should be invariant under flipping both directions at ({i},{j})"
            );
        }
    }
}

#[test]
fn w_only_exact_outer_direction_sign_rules_hold() {
    let (family, block_states) = w_only_exact_fixture();
    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir = Array1::<f64>::zeros(total);
    let w_range = slices.w.as_ref().expect("w slice");
    dir[w_range.start] = 0.15;
    if w_range.len() > 1 {
        dir[w_range.start + 1] = -0.07;
    }
    let neg_dir = dir.mapv(|value| -value);

    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "w-only third directional derivative", e))
        .expect("w-only third directional derivative matrix");
    let third_neg = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &neg_dir)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "w-only negated third directional derivative", e
            )
        })
        .expect("w-only negated third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir, &dir)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "w-only fourth directional derivative", e))
        .expect("w-only fourth directional derivative matrix");
    let fourth_neg = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &neg_dir, &neg_dir)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "w-only doubly-negated fourth directional derivative", e
            )
        })
        .expect("w-only doubly-negated fourth directional derivative matrix");

    for i in 0..total {
        for j in 0..total {
            assert!(
                (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                "w-only third directional derivative should be odd at ({i},{j})"
            );
            assert!(
                (fourth_neg[[i, j]] - fourth[[i, j]]).abs() < 1e-8,
                "w-only fourth directional derivative should be invariant under flipping both directions at ({i},{j})"
            );
        }
    }
}

#[test]
fn h_only_exact_outer_fourth_direction_swap_is_symmetric() {
    let (family, block_states) = h_only_exact_fixture();
    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    let h_range = slices.h.as_ref().expect("h slice");
    dir_u[h_range.start] = 0.12;
    if h_range.len() > 1 {
        dir_u[h_range.start + 1] = -0.06;
    }

    dir_v[h_range.start] = 0.07;
    if h_range.len() > 1 {
        dir_v[h_range.start + 1] = 0.05;
    }

    let forward = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "h-only fourth directional derivative", e))
        .expect("h-only fourth directional derivative matrix");
    let swapped = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_v, &dir_u)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "h-only swapped fourth directional derivative", e
            )
        })
        .expect("h-only swapped fourth directional derivative matrix");

    for i in 0..total {
        for j in 0..total {
            assert!(
                (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                "h-only fourth directional derivative should be symmetric in direction arguments at ({i},{j})"
            );
        }
    }
}

#[test]
fn w_only_exact_outer_fourth_direction_swap_is_symmetric() {
    let (family, block_states) = w_only_exact_fixture();
    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    let w_range = slices.w.as_ref().expect("w slice");
    dir_u[w_range.start] = 0.15;
    if w_range.len() > 1 {
        dir_u[w_range.start + 1] = -0.07;
    }

    dir_v[w_range.start] = 0.09;
    if w_range.len() > 1 {
        dir_v[w_range.start + 1] = 0.03;
    }

    let forward = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "w-only fourth directional derivative", e))
        .expect("w-only fourth directional derivative matrix");
    let swapped = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_v, &dir_u)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "w-only swapped fourth directional derivative", e
            )
        })
        .expect("w-only swapped fourth directional derivative matrix");

    for i in 0..total {
        for j in 0..total {
            assert!(
                (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                "w-only fourth directional derivative should be symmetric in direction arguments at ({i},{j})"
            );
        }
    }
}

#[test]
fn h_only_exact_outer_zero_direction_returns_zero() {
    let (family, block_states) = h_only_exact_fixture();
    let slices = block_slices(&family);
    let zero = Array1::<f64>::zeros(slices.total);
    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &zero)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "h-only third directional derivative", e))
        .expect("h-only third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &zero, &zero)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "h-only fourth directional derivative", e))
        .expect("h-only fourth directional derivative matrix");

    assert!(
        third.iter().all(|value| value.abs() <= 0.0),
        "expected zero h-only third directional derivative for zero direction"
    );
    assert!(
        fourth.iter().all(|value| value.abs() <= 0.0),
        "expected zero h-only fourth directional derivative for zero directions"
    );
}

#[test]
fn w_only_exact_outer_zero_direction_returns_zero() {
    let (family, block_states) = w_only_exact_fixture();
    let slices = block_slices(&family);
    let zero = Array1::<f64>::zeros(slices.total);
    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &zero)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "w-only third directional derivative", e))
        .expect("w-only third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &zero, &zero)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "w-only fourth directional derivative", e))
        .expect("w-only fourth directional derivative matrix");

    assert!(
        third.iter().all(|value| value.abs() <= 0.0),
        "expected zero w-only third directional derivative for zero direction"
    );
    assert!(
        fourth.iter().all(|value| value.abs() <= 0.0),
        "expected zero w-only fourth directional derivative for zero directions"
    );
}

#[test]
fn w_only_gradient_matches_loglik_finite_differences() {
    let z = array![-0.8, 0.2, 1.1];
    let y = array![0.0, 1.0, 1.0];
    let weights = array![1.0, 0.7, 1.3];
    let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link block", e));
    let family =
        BernoulliMarginalSlopeFamily {
            y: Arc::new(y.clone()),
            weights: Arc::new(weights.clone()),
            z: Arc::new(z.clone()),
            marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            link_dev: Some(link_prepared.runtime.clone()),
            ..default_test_family()
        };
    let beta_w = Array1::from_iter(
        (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
    );
    let states_at = |q: f64, b: f64, bw: Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: array![q],
                eta: Array1::from_elem(z.len(), q),
            },
            ParameterBlockState {
                beta: array![b],
                eta: Array1::from_elem(z.len(), b),
            },
            ParameterBlockState {
                beta: bw,
                eta: Array1::zeros(z.len()),
            },
        ]
    };

    let q0 = 0.25;
    let b0 = 0.6;
    let block_states = states_at(q0, b0, beta_w.clone());
    let eval = family
        .evaluate(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "family evaluation", e));
    let grad_q = match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton q block"),
    };
    let grad_b = match &eval.blockworking_sets[1] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton b block"),
    };
    let grad_w0 = match &eval.blockworking_sets[2] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton w block"),
    };

    let fd = |which: &str, eps: f64| match which {
        "q" => {
            let plus = family
                .log_likelihood_only(&states_at(q0 + eps, b0, beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus q", e));
            let minus = family
                .log_likelihood_only(&states_at(q0 - eps, b0, beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus q", e));
            (plus - minus) / (2.0 * eps)
        }
        "b" => {
            let plus = family
                .log_likelihood_only(&states_at(q0, b0 + eps, beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus b", e));
            let minus = family
                .log_likelihood_only(&states_at(q0, b0 - eps, beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus b", e));
            (plus - minus) / (2.0 * eps)
        }
        "w0" => {
            let mut plus_w = beta_w.clone();
            plus_w[0] += eps;
            let mut minus_w = beta_w.clone();
            minus_w[0] -= eps;
            let plus = family
                .log_likelihood_only(&states_at(q0, b0, plus_w))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus w", e));
            let minus = family
                .log_likelihood_only(&states_at(q0, b0, minus_w))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus w", e));
            (plus - minus) / (2.0 * eps)
        }
        _ => panic!("unknown derivative target"),
    };

    let eps = 1e-5;
    assert!((grad_q - fd("q", eps)).abs() < 2e-4);
    assert!((grad_b - fd("b", eps)).abs() < 2e-4);
    assert!((grad_w0 - fd("w0", eps)).abs() < 2e-4);
}

#[test]
fn h_only_gradient_matches_loglik_finite_differences() {
    let z = array![-0.8, 0.2, 1.1];
    let y = array![0.0, 1.0, 1.0];
    let weights = array![1.0, 0.7, 1.3];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &z,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "score warp block", e));
    let family =
        BernoulliMarginalSlopeFamily {
            y: Arc::new(y.clone()),
            weights: Arc::new(weights.clone()),
            z: Arc::new(z.clone()),
            marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            score_warp: Some(score_prepared.runtime.clone()),
            ..default_test_family()
        };
    let beta_h = Array1::from_iter(
        (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
    );
    let states_at = |q: f64, b: f64, bh: Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: array![q],
                eta: Array1::from_elem(z.len(), q),
            },
            ParameterBlockState {
                beta: array![b],
                eta: Array1::from_elem(z.len(), b),
            },
            ParameterBlockState {
                beta: bh,
                eta: Array1::zeros(z.len()),
            },
        ]
    };

    let q0 = 0.25;
    let b0 = 0.6;
    let block_states = states_at(q0, b0, beta_h.clone());
    let eval = family
        .evaluate(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "family evaluation", e));
    let grad_q = match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton q block"),
    };
    let grad_b = match &eval.blockworking_sets[1] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton b block"),
    };
    let grad_h0 = match &eval.blockworking_sets[2] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton h block"),
    };

    let fd = |which: &str, eps: f64| match which {
        "q" => {
            let plus = family
                .log_likelihood_only(&states_at(q0 + eps, b0, beta_h.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus q", e));
            let minus = family
                .log_likelihood_only(&states_at(q0 - eps, b0, beta_h.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus q", e));
            (plus - minus) / (2.0 * eps)
        }
        "b" => {
            let plus = family
                .log_likelihood_only(&states_at(q0, b0 + eps, beta_h.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus b", e));
            let minus = family
                .log_likelihood_only(&states_at(q0, b0 - eps, beta_h.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus b", e));
            (plus - minus) / (2.0 * eps)
        }
        "h0" => {
            let mut plus_h = beta_h.clone();
            plus_h[0] += eps;
            let mut minus_h = beta_h.clone();
            minus_h[0] -= eps;
            let plus = family
                .log_likelihood_only(&states_at(q0, b0, plus_h))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus h", e));
            let minus = family
                .log_likelihood_only(&states_at(q0, b0, minus_h))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus h", e));
            (plus - minus) / (2.0 * eps)
        }
        _ => panic!("unknown derivative target"),
    };

    let eps = 1e-5;
    assert!((grad_q - fd("q", eps)).abs() < 2e-4);
    assert!((grad_b - fd("b", eps)).abs() < 2e-4);
    assert!((grad_h0 - fd("h0", eps)).abs() < 2e-4);
}

#[test]
fn flexible_denested_gradient_matches_loglik_finite_differences() {
    let z = array![-0.8, 0.2, 1.1];
    let y = array![0.0, 1.0, 1.0];
    let weights = array![1.0, 0.7, 1.3];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &z,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "score warp block", e));
    let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link block", e));
    let family =
        BernoulliMarginalSlopeFamily {
            y: Arc::new(y.clone()),
            weights: Arc::new(weights.clone()),
            z: Arc::new(z.clone()),
            marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            score_warp: Some(score_prepared.runtime.clone()),
            link_dev: Some(link_prepared.runtime.clone()),
            ..default_test_family()
        };
    let beta_h = Array1::from_iter(
        (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
    );
    let beta_w = Array1::from_iter(
        (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
    );
    let states_at = |q: f64, b: f64, bh: Array1<f64>, bw: Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: array![q],
                eta: Array1::from_elem(z.len(), q),
            },
            ParameterBlockState {
                beta: array![b],
                eta: Array1::from_elem(z.len(), b),
            },
            ParameterBlockState {
                beta: bh,
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: bw,
                eta: Array1::zeros(z.len()),
            },
        ]
    };

    let q0 = 0.25;
    let b0 = 0.6;
    let block_states = states_at(q0, b0, beta_h.clone(), beta_w.clone());
    let eval = family
        .evaluate(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "family evaluation", e));
    let grad_q = match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton q block"),
    };
    let grad_b = match &eval.blockworking_sets[1] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton b block"),
    };
    let grad_h0 = match &eval.blockworking_sets[2] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton h block"),
    };
    let grad_w0 = match &eval.blockworking_sets[3] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("expected exact-newton w block"),
    };

    let fd = |which: &str, eps: f64| match which {
        "q" => {
            let plus = family
                .log_likelihood_only(&states_at(q0 + eps, b0, beta_h.clone(), beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus q", e));
            let minus = family
                .log_likelihood_only(&states_at(q0 - eps, b0, beta_h.clone(), beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus q", e));
            (plus - minus) / (2.0 * eps)
        }
        "b" => {
            let plus = family
                .log_likelihood_only(&states_at(q0, b0 + eps, beta_h.clone(), beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus b", e));
            let minus = family
                .log_likelihood_only(&states_at(q0, b0 - eps, beta_h.clone(), beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus b", e));
            (plus - minus) / (2.0 * eps)
        }
        "h0" => {
            let mut plus_h = beta_h.clone();
            plus_h[0] += eps;
            let mut minus_h = beta_h.clone();
            minus_h[0] -= eps;
            let plus = family
                .log_likelihood_only(&states_at(q0, b0, plus_h, beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus h", e));
            let minus = family
                .log_likelihood_only(&states_at(q0, b0, minus_h, beta_w.clone()))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus h", e));
            (plus - minus) / (2.0 * eps)
        }
        "w0" => {
            let mut plus_w = beta_w.clone();
            plus_w[0] += eps;
            let mut minus_w = beta_w.clone();
            minus_w[0] -= eps;
            let plus = family
                .log_likelihood_only(&states_at(q0, b0, beta_h.clone(), plus_w))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus w", e));
            let minus = family
                .log_likelihood_only(&states_at(q0, b0, beta_h.clone(), minus_w))
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus w", e));
            (plus - minus) / (2.0 * eps)
        }
        _ => panic!("unknown derivative target"),
    };

    let eps = 1e-5;
    assert!((grad_q - fd("q", eps)).abs() < 2e-4);
    assert!((grad_b - fd("b", eps)).abs() < 2e-4);
    assert!((grad_h0 - fd("h0", eps)).abs() < 2e-4);
    assert!((grad_w0 - fd("w0", eps)).abs() < 2e-4);
}

#[test]
fn flexible_exact_outer_directional_derivatives_are_present_and_finite() {
    let z = array![-0.8, 0.2, 1.1];
    let y = array![0.0, 1.0, 1.0];
    let weights = array![1.0, 0.7, 1.3];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &z,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "score warp block", e));
    let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link block", e));
    let family =
        BernoulliMarginalSlopeFamily {
            y: Arc::new(y.clone()),
            weights: Arc::new(weights.clone()),
            z: Arc::new(z.clone()),
            marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                array![[1.0], [1.0], [1.0]],
            )),
            score_warp: Some(score_prepared.runtime.clone()),
            link_dev: Some(link_prepared.runtime.clone()),
            ..default_test_family()
        };
    let beta_h = Array1::from_iter(
        (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
    );
    let beta_w = Array1::from_iter(
        (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
    );
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.25],
            eta: Array1::from_elem(z.len(), 0.25),
        },
        ParameterBlockState {
            beta: array![0.6],
            eta: Array1::from_elem(z.len(), 0.6),
        },
        ParameterBlockState {
            beta: beta_h.clone(),
            eta: Array1::zeros(z.len()),
        },
        ParameterBlockState {
            beta: beta_w.clone(),
            eta: Array1::zeros(z.len()),
        },
    ];

    let slices = block_slices(&family);
    let total = slices.total;
    let mut dir_u = Array1::<f64>::zeros(total);
    let mut dir_v = Array1::<f64>::zeros(total);
    dir_u[slices.marginal.start] = 0.7;
    dir_u[slices.logslope.start] = -0.2;
    if let Some(h_range) = slices.h.as_ref() {
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
    }
    if let Some(w_range) = slices.w.as_ref() {
        dir_u[w_range.start] = 0.08;
    }

    dir_v[slices.marginal.start] = -0.4;
    dir_v[slices.logslope.start] = 0.3;
    if let Some(h_range) = slices.h.as_ref() {
        dir_v[h_range.start] = -0.03;
    }
    if let Some(w_range) = slices.w.as_ref() {
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }
    }

    let third = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "flex third directional derivative", e))
        .expect("flex third directional derivative matrix");
    let fourth = family
        .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "flex fourth directional derivative", e))
        .expect("flex fourth directional derivative matrix");

    assert_eq!(third.dim(), (total, total));
    assert_eq!(fourth.dim(), (total, total));
    assert!(third.iter().all(|value| value.is_finite()));
    assert!(fourth.iter().all(|value| value.is_finite()));
    let max_abs_third = third
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_abs_fourth = fourth
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        max_abs_third > 1e-10,
        "expected nonzero dual-flex third directional derivative"
    );
    assert!(
        max_abs_fourth > 1e-10,
        "expected nonzero dual-flex fourth directional derivative"
    );

    for i in 0..total {
        for j in 0..i {
            assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
            assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
        }
    }
}

#[test]
fn flexible_evaluate_block_diagonals_match_joint_exact_oracle() {
    let z = array![-1.1, -0.25, 0.35, 1.2];
    let y = array![0.0, 1.0, 0.0, 1.0];
    let weights = array![1.0, 0.8, 1.3, 0.7];
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &z,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "score warp block", e));
    let link_seed = array![-1.8, -0.7, 0.1, 0.9, 1.7];
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 4,
            ..DeviationBlockConfig::default()
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "link block", e));
    let marginal_x = array![[1.0, -0.4], [1.0, 0.2], [1.0, 0.7], [1.0, 1.1]];
    let logslope_x = array![[1.0, 0.3], [1.0, -0.6], [1.0, 0.5], [1.0, -1.0]];
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(y.clone()),
        weights: Arc::new(weights.clone()),
        z: Arc::new(z.clone()),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            marginal_x.clone(),
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            logslope_x.clone(),
        )),
        score_warp: Some(score_prepared.runtime.clone()),
        link_dev: Some(link_prepared.runtime.clone()),
        ..default_test_family()
    };
    let beta_m = array![0.18, -0.07];
    let beta_g = array![0.42, 0.05];
    let beta_h = Array1::from_iter(
        (0..score_prepared.block.design.ncols()).map(|idx| 0.006 * (idx as f64 + 1.0)),
    );
    let beta_w = Array1::from_iter(
        (0..link_prepared.block.design.ncols()).map(|idx| -0.004 * (idx as f64 + 1.0)),
    );
    let block_states = vec![
        ParameterBlockState {
            beta: beta_m.clone(),
            eta: marginal_x.dot(&beta_m),
        },
        ParameterBlockState {
            beta: beta_g.clone(),
            eta: logslope_x.dot(&beta_g),
        },
        ParameterBlockState {
            beta: beta_h,
            eta: Array1::zeros(z.len()),
        },
        ParameterBlockState {
            beta: beta_w,
            eta: Array1::zeros(z.len()),
        },
    ];
    let eval = family
        .evaluate(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "block evaluation", e));
    let joint_hessian = family
        .exact_newton_joint_hessian(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "joint hessian result", e))
        .expect("dense joint hessian");
    let joint_gradient = family
        .exact_newton_joint_gradient_evaluation(&block_states, &[])
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "joint gradient result", e))
        .expect("joint gradient");
    let slices = block_slices(&family);
    let ranges = [
        slices.marginal.clone(),
        slices.logslope.clone(),
        slices.h.clone().expect("score-warp block"),
        slices.w.clone().expect("link-deviation block"),
    ];

    assert!((eval.log_likelihood - joint_gradient.log_likelihood).abs() < 2e-12);
    assert!(
        (eval.log_likelihood
            - family
                .log_likelihood_only(&block_states)
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "log likelihood only", e)))
        .abs()
            < 2e-12
    );

    for (block_idx, range) in ranges.iter().enumerate() {
        let (gradient, hessian) = match &eval.blockworking_sets[block_idx] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => (gradient, hessian),
            _ => panic!("expected exact-newton block {block_idx}"),
        };
        let expected_gradient = joint_gradient.gradient.slice(s![range.clone()]);
        for idx in 0..gradient.len() {
            assert!(
                (gradient[idx] - expected_gradient[idx]).abs() < 2e-10,
                "gradient mismatch block {block_idx} idx {idx}: got {} expected {}",
                gradient[idx],
                expected_gradient[idx]
            );
        }
        let dense_hessian = match hessian {
            SymmetricMatrix::Dense(h) => h,
            _ => panic!("expected dense hessian block {block_idx}"),
        };
        let expected_hessian = joint_hessian.slice(s![range.clone(), range.clone()]);
        for i in 0..dense_hessian.nrows() {
            for j in 0..dense_hessian.ncols() {
                assert!(
                    (dense_hessian[[i, j]] - expected_hessian[[i, j]]).abs() < 2e-9,
                    "hessian mismatch block {block_idx} ({i},{j}): got {} expected {}",
                    dense_hessian[[i, j]],
                    expected_hessian[[i, j]]
                );
            }
        }
    }
}

#[test]
fn latent_z_normalization_accepts_finite_sample_gaussian_scores() {
    let z = array![
        -0.85, -0.12, 0.31, 1.04, -1.21, 0.56, 0.77, -0.44, 1.33, -0.09, 0.28, -0.67
    ];
    let weights = Array1::from_elem(12, 1.0);
    let (standardized, normalization) = standardize_latent_z_with_policy(
        &z,
        &weights,
        "bernoulli-marginal-slope",
        &LatentZPolicy::exploratory_fit_weighted(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "normalize z", e));
    let replayed = normalization
        .apply(&z, "bernoulli-marginal-slope replay")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "replay normalized z", e));
    let mean = standardized.sum() / standardized.len() as f64;
    let var = standardized.iter().map(|v| v * v).sum::<f64>() / standardized.len() as f64;
    assert_eq!(replayed, standardized);
    assert!(mean.abs() < 1e-12);
    assert!((var.sqrt() - 1.0).abs() < 1e-12);
}

#[test]
fn latent_z_normalization_rejects_extreme_non_gaussian_scores() {
    let z = array![0.0, 0.0, 0.0, 0.0, 10.0, -10.0];
    let weights = Array1::from_elem(6, 1.0);
    let strict_policy = LatentZPolicy {
        check_mode: LatentZCheckMode::Strict,
        ..LatentZPolicy::default()
    };
    let err =
        standardize_latent_z_with_policy(&z, &weights, "bernoulli-marginal-slope", &strict_policy)
            .expect_err("expected non-gaussian rejection");
    assert!(err.contains("approximately latent N(0,1)"));
}

/// Under P4, bad-normal but continuous latent z routes through a
/// rank-INT calibration (weighted mid-distribution ranks → `Φ⁻¹`)
/// instead of the legacy local-/global-empirical grid. Rank-INT is a
/// modeling choice (the affine rigid model is respecified on the
/// calibrated axis), and the standard-normal closed-form kernel is
/// admitted only because the calibrated sample itself passes the
/// standard-normal adequacy gate — which a distinct-valued sample's
/// mid-rank normal scores do. This test pins that routing.
#[test]
fn auto_latent_measure_uses_rank_int_calibration_for_bad_normal_diagnostics() {
    // Exponential quantiles: strongly skewed (the raw pooled gate fails on
    // mean/skew), but all values are distinct, so the calibrated mid-rank
    // normal scores pass the gate and keep the standard-normal kernel.
    let n = 100usize;
    let z = Array1::from_iter((1..=n).map(|i| -f64::ln(1.0 - (i as f64 - 0.5) / n as f64)));
    let weights = Array1::from_elem(n, 1.0);
    let policy = LatentZPolicy {
        check_mode: LatentZCheckMode::Off,
        normalization: LatentZNormalizationMode::None,
        latent_measure: LatentMeasureSpec::Auto { grid_size: 5 },
        ..LatentZPolicy::default()
    };
    let (measure, calibration) = build_latent_measure_with_geometry(&z, &weights, &policy, None)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "auto latent measure", e));
    assert!(
        matches!(measure, LatentMeasureKind::StandardNormal),
        "bad-normal latent z must route through rank-INT to the standard-normal kernel"
    );
    match calibration {
        LatentMeasureCalibration::RankInverseNormal(cal) => {
            // Knot table is non-empty and the transformed sample is
            // (approximately) N(0,1) by construction.
            assert!(
                !cal.sorted_z.is_empty(),
                "rank-INT knot table must be non-empty"
            );
            assert_eq!(cal.sorted_z.len(), cal.weighted_cdf.len());
            // Strictly increasing knots and strictly increasing CDF.
            for w in cal.sorted_z.windows(2) {
                assert!(w[0] < w[1], "sorted_z must be strictly increasing");
            }
            for w in cal.weighted_cdf.windows(2) {
                assert!(
                    w[0] <= w[1],
                    "weighted_cdf must be non-decreasing (got {} -> {})",
                    w[0],
                    w[1]
                );
            }
            // Post-mean near 0, post-sd not wildly off 1.
            assert!(
                cal.post_mean.abs() < 0.5,
                "rank-INT post-mean too far from 0: {}",
                cal.post_mean
            );
            assert!(
                cal.post_sd > 0.0 && cal.post_sd.is_finite(),
                "rank-INT post-sd must be positive finite, got {}",
                cal.post_sd
            );
        }
        LatentMeasureCalibration::None => {
            panic!("bad-normal latent z must produce a RankInverseNormal calibration")
        }
        LatentMeasureCalibration::ConditionalLocationScale(_) => {
            panic!(
                "conditioning=None cannot fire the E[z|C]/Var(z|C) Rao gate, so the \
                 calibration must never be ConditionalLocationScale here"
            )
        }
    }
}

/// Rank-INT cannot Gaussianize a heavily tied sample: the calibrated law
/// stays discrete (here four of six observations share one atom), so the
/// standard-normal adequacy re-check on the calibrated scale fails and the
/// Auto path must fall back to the mathematically exact global-empirical
/// latent measure instead of running the closed-form standard-normal
/// kernel on a non-Gaussian score.
#[test]
fn auto_latent_measure_falls_back_to_empirical_when_rank_int_cannot_gaussianize() {
    let z = array![0.0, 0.0, 0.0, 0.0, 10.0, -10.0];
    let weights = Array1::from_elem(6, 1.0);
    let policy = LatentZPolicy {
        check_mode: LatentZCheckMode::Off,
        normalization: LatentZNormalizationMode::None,
        latent_measure: LatentMeasureSpec::Auto { grid_size: 5 },
        ..LatentZPolicy::default()
    };
    let (measure, calibration) = build_latent_measure_with_geometry(&z, &weights, &policy, None)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "auto latent measure", e));
    assert!(
        matches!(measure, LatentMeasureKind::GlobalEmpirical { .. }),
        "tied latent z whose rank-INT output fails the adequacy gate must use the \
         global-empirical latent measure, got {measure:?}"
    );
    assert!(
        matches!(calibration, LatentMeasureCalibration::None),
        "the empirical fallback models the raw score directly; no rank-INT map applies"
    );
}

/// The weighted mid-distribution rank-INT depends only on *relative*
/// weights: rescaling every weight by a common factor must leave the
/// calibrated scores unchanged, and a two-point sample must map to
/// symmetric non-degenerate normal scores (the scale-dependent Blom
/// formula collapsed both to zero at total weight one).
#[test]
fn rank_int_calibration_is_invariant_to_common_weight_rescaling() {
    let z = array![-1.0, 1.0];
    for &scale in &[0.5_f64, 1.0, 100.0] {
        let weights = Array1::from_elem(2, scale * 0.5);
        let cal = LatentZRankIntCalibration::fit(&z, &weights)
            .unwrap_or_else(|e| panic!("rank-INT fit failed at weight scale {scale}: {e:?}"));
        // Mid-ranks are (0.25, 0.75) at every common weight scale.
        assert!(
            (cal.weighted_cdf[0] - 0.25).abs() < 1e-12
                && (cal.weighted_cdf[1] - 0.75).abs() < 1e-12,
            "mid-distribution ranks must be (0.25, 0.75) independent of weight scale, got {:?} at scale {scale}",
            cal.weighted_cdf
        );
        let lo = cal.apply_at_predict(-1.0);
        let hi = cal.apply_at_predict(1.0);
        assert!(
            (lo + hi).abs() < 1e-12 && hi > 0.5,
            "calibrated two-point scores must be symmetric and non-degenerate, got ({lo}, {hi}) at scale {scale}"
        );
    }
}

/// Zero-weight observations carry no probability mass: they must not become
/// knots of the weighted empirical distribution, and tie groups get the
/// midpoint of their pooled mass.
#[test]
fn rank_int_calibration_drops_zero_mass_knots_and_pools_tie_mass() {
    let z = array![-2.0, 0.0, 0.0, 5.0, 3.0];
    let weights = array![1.0, 1.0, 1.0, 0.0, 1.0];
    let cal = LatentZRankIntCalibration::fit(&z, &weights)
        .unwrap_or_else(|e| panic!("rank-INT fit failed: {e:?}"));
    assert_eq!(
        cal.sorted_z,
        vec![-2.0, 0.0, 3.0],
        "the zero-weight z=5 row must not become a knot"
    );
    // W=4: p(-2)=0.5/4, p(0)=(1+1)/4 (tie group of mass 2 centered at its
    // midpoint), p(3)=(3+0.5)/4.
    let expect = [0.125, 0.5, 0.875];
    for (got, want) in cal.weighted_cdf.iter().zip(expect.iter()) {
        assert!(
            (got - want).abs() < 1e-12,
            "mid-distribution ranks must be {expect:?}, got {:?}",
            cal.weighted_cdf
        );
    }
}

#[test]
fn empirical_intercept_calibrates_marginal_probability() {
    let nodes = vec![-2.0, -0.25, 0.5, 3.0];
    let weights = vec![0.2, 0.3, 0.1, 0.4];
    let target_q = -0.35;
    let target_mu = normal_cdf(target_q);
    let slope = 0.8;
    let scale = 0.9;
    let intercept = empirical_intercept_from_marginal(
        target_mu, target_q, slope, scale, &nodes, &weights, None,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "empirical intercept", e));
    let calibrated = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&node, &weight)| weight * normal_cdf(intercept + scale * slope * node))
        .sum::<f64>();
    assert!((calibrated - target_mu).abs() <= 1e-10);
}

#[test]
fn skewed_rigid_empirical_grid_calibrates_marginal_probability() {
    let z = array![-1.15, -1.05, -0.95, -0.8, -0.65, -0.45, 0.1, 0.9, 2.4, 4.7];
    let weights = array![1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.5, 0.35, 0.2, 0.1];
    let grid = build_empirical_z_grid(&z, &weights, 7, "test skewed grid")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "grid", e));
    let target_q = 0.25;
    let target_mu = normal_cdf(target_q);
    let slope = 1.35;
    let scale = 0.82;

    let intercept = empirical_intercept_from_marginal(
        target_mu,
        target_q,
        slope,
        scale,
        &grid.nodes,
        &grid.weights,
        None,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "empirical intercept", e));
    let calibrated = grid
        .nodes
        .iter()
        .zip(grid.weights.iter())
        .map(|(&node, &weight)| weight * normal_cdf(intercept + scale * slope * node))
        .sum::<f64>();

    assert!((calibrated - target_mu).abs() <= 1e-10);
}

/// Builds a 3-row family on a global empirical latent grid plus a closure
/// returning the closed-form `(neglog, grad, hess)` as a function of
/// `(m = marginal_eta, g = slope)`. Shared by the finite-difference
/// validation tests below.
fn empirical_rigid_fd_fixture() -> (
    BernoulliMarginalSlopeFamily,
    EmpiricalZGrid,
    [f64; 3],
    [f64; 3],
) {
    let grid_nodes = array![-1.15, -1.05, -0.95, -0.8, -0.65, -0.45, 0.1, 0.9, 2.4, 4.7];
    let grid_w = array![1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.5, 0.35, 0.2, 0.1];
    let grid = build_empirical_z_grid(&grid_nodes, &grid_w, 7, "cf vs fd grid")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "grid", e));
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(array![1.0, 0.0, 1.0]),
        weights: Arc::new(array![1.0, 0.7, 1.3]),
        z: Arc::new(array![0.3, -0.8, 1.6]),
        latent_measure: LatentMeasureKind::GlobalEmpirical { grid: grid.clone() },
        gaussian_frailty_sd: Some(0.82),
        ..default_test_family()
    };
    (family, grid, [0.25, -0.4, 0.7], [1.35, 0.9, 1.1])
}

/// Validates the §2 implicit-function-theorem closed form for the rigid
/// empirical-grid kernel against central finite differences of its own
/// exact row negative log-likelihood (ground truth, not a derivative): FD
/// of `nll` reproduces the analytic gradient, and FD of the gradient the
/// Hessian. An independent check sharing no derivative-formula code — a
/// sign or algebra slip in any intercept derivative breaks it.
#[test]
fn empirical_rigid_grad_hess_match_finite_differences() {
    let (family, grid, marginal_etas, slopes) = empirical_rigid_fd_fixture();
    let cf = |row: usize, m: f64, g: f64| {
        let marginal = family
            .marginal_link_map(m)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "link map", e));
        family
            .empirical_rigid_primary_grad_hess_closed_form(
                row,
                marginal,
                g,
                &grid.nodes,
                &grid.weights,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "closed form", e))
    };
    let h = 1e-4;
    for row in 0..3 {
        let (m, g) = (marginal_etas[row], slopes[row]);
        let (_, grad_cf, hess_cf) = cf(row, m, g);

        // gradient vs FD of nll
        let grad_fd = [
            (cf(row, m + h, g).0 - cf(row, m - h, g).0) / (2.0 * h),
            (cf(row, m, g + h).0 - cf(row, m, g - h).0) / (2.0 * h),
        ];
        for k in 0..2 {
            assert!(
                (grad_cf[k] - grad_fd[k]).abs() <= 1e-5 + 1e-4 * grad_cf[k].abs(),
                "row {row}: grad[{k}] closed-form {} vs fd {}",
                grad_cf[k],
                grad_fd[k]
            );
        }
        // Hessian vs FD of gradient
        let (gpm, gmm) = (cf(row, m + h, g).1, cf(row, m - h, g).1);
        let (gpg, gmg) = (cf(row, m, g + h).1, cf(row, m, g - h).1);
        for k in 0..2 {
            let fd0 = (gpm[k] - gmm[k]) / (2.0 * h);
            let fd1 = (gpg[k] - gmg[k]) / (2.0 * h);
            assert!(
                (hess_cf[0][k] - fd0).abs() <= 1e-5 + 1e-4 * hess_cf[0][k].abs(),
                "row {row}: hess[0][{k}] closed-form {} vs fd {}",
                hess_cf[0][k],
                fd0
            );
            assert!(
                (hess_cf[1][k] - fd1).abs() <= 1e-5 + 1e-4 * hess_cf[1][k].abs(),
                "row {row}: hess[1][{k}] closed-form {} vs fd {}",
                hess_cf[1][k],
                fd1
            );
        }
    }
}

/// Higher-order analogue: the §2 IFT closed forms for the rigid
/// empirical-grid **third** and **fourth** derivative tensors must equal the
/// central finite differences of the analytic level below — FD of the
/// Hessian reproduces the third tensor, FD of the third the fourth. Every
/// level is thus tied transitively to the exact negative log-likelihood
/// (validated in `empirical_rigid_grad_hess_match_finite_differences`); a
/// sign or algebra slip in any single intercept derivative localises to one
/// component assertion.
#[test]
fn empirical_rigid_higher_order_match_finite_differences() {
    let (family, grid, marginal_etas, slopes) = empirical_rigid_fd_fixture();
    let hess = |row: usize, m: f64, g: f64| {
        let marginal = family
            .marginal_link_map(m)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "link map", e));
        family
            .empirical_rigid_primary_grad_hess_closed_form(
                row,
                marginal,
                g,
                &grid.nodes,
                &grid.weights,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "closed form", e))
            .2
    };
    let third = |row: usize, m: f64, g: f64| {
        let marginal = family.marginal_link_map(m).expect("link map");
        family
            .empirical_rigid_third_full_closed_form(row, marginal, g, &grid.nodes, &grid.weights)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "third closed form", e))
    };
    let fourth = |row: usize, m: f64, g: f64| {
        let marginal = family.marginal_link_map(m).expect("link map");
        family
            .empirical_rigid_fourth_full_closed_form(row, marginal, g, &grid.nodes, &grid.weights)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "fourth closed form", e))
    };
    let h = 1e-4;
    for row in 0..3 {
        let (m, g) = (marginal_etas[row], slopes[row]);

        // third[i][j][k] vs ∂_i of hess[j][k]
        let t_cf = third(row, m, g);
        let (hpm, hmm) = (hess(row, m + h, g), hess(row, m - h, g));
        let (hpg, hmg) = (hess(row, m, g + h), hess(row, m, g - h));
        for j in 0..2 {
            for k in 0..2 {
                let fd0 = (hpm[j][k] - hmm[j][k]) / (2.0 * h);
                let fd1 = (hpg[j][k] - hmg[j][k]) / (2.0 * h);
                assert!(
                    (t_cf[0][j][k] - fd0).abs() <= 1e-4 + 1e-4 * t_cf[0][j][k].abs(),
                    "row {row}: third[0][{j}][{k}] cf {} vs fd {}",
                    t_cf[0][j][k],
                    fd0
                );
                assert!(
                    (t_cf[1][j][k] - fd1).abs() <= 1e-4 + 1e-4 * t_cf[1][j][k].abs(),
                    "row {row}: third[1][{j}][{k}] cf {} vs fd {}",
                    t_cf[1][j][k],
                    fd1
                );
            }
        }

        // fourth[i][j][k][l] vs ∂_i of third[j][k][l]
        let f_cf = fourth(row, m, g);
        let (tpm, tmm) = (third(row, m + h, g), third(row, m - h, g));
        let (tpg, tmg) = (third(row, m, g + h), third(row, m, g - h));
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    let fd0 = (tpm[j][k][l] - tmm[j][k][l]) / (2.0 * h);
                    let fd1 = (tpg[j][k][l] - tmg[j][k][l]) / (2.0 * h);
                    assert!(
                        (f_cf[0][j][k][l] - fd0).abs() <= 1e-3 + 1e-3 * f_cf[0][j][k][l].abs(),
                        "row {row}: fourth[0][{j}][{k}][{l}] cf {} vs fd {}",
                        f_cf[0][j][k][l],
                        fd0
                    );
                    assert!(
                        (f_cf[1][j][k][l] - fd1).abs() <= 1e-3 + 1e-3 * f_cf[1][j][k][l].abs(),
                        "row {row}: fourth[1][{j}][{k}][{l}] cf {} vs fd {}",
                        f_cf[1][j][k][l],
                        fd1
                    );
                }
            }
        }
    }
}

/// #932 jet-program oracle for the rigid **empirical-grid** branch of the
/// production `BernoulliRigidRowKernel` (`RowKernel<2>`).
///
/// The standard-normal branch of that kernel is already a jet program
/// (`rigid_standard_normal_tower` evaluates the whole value/grad/Hess/3rd/4th
/// tower over `Tower4` `compose_unary` — a true cutover, no hand calculus),
/// and is guarded by `rigid_standard_normal_*_full` hand-witness tests above.
/// The empirical-grid branch is the one production tower still derived by
/// HAND: `empirical_rigid_primary_grad_hess_closed_form` /
/// `empirical_rigid_third_full_closed_form` /
/// `empirical_rigid_fourth_full_closed_form` write the implicit-function-theorem
/// intercept chain `a_m, a_g, a_mm, … a_gggg` and the Faà-di-Bruno
/// `ℓ_ijkl = u4·η⁴ + …` cross blocks term by term. That hand chain is exactly
/// the #736/#833 desync genus (the comment at `empirical_rigid_fourth_full_closed_form`
/// records #833 — a dropped `g_aa·a_ggg` term that left the 4th-order block
/// ~1.8% short, previously caught only by finite differences).
///
/// Until now that branch was guarded only by the FINITE-DIFFERENCE oracles
/// (`empirical_rigid_grad_hess_match_finite_differences`,
/// `empirical_rigid_higher_order_match_finite_differences`). This test adds the
/// #932 universal-oracle form the issue asks for: it writes the empirical row
/// NLL ONCE over the generic row-program scalar and asserts the hand kernel's value / gradient /
/// Hessian / `row_third_contracted` / `row_fourth_contracted` channels equal the
/// single-expression tower truth EXACTLY (no FD truncation tolerance). The
/// intercept jet `a(m, g)` is obtained by JET-NEWTON on the implicit
/// calibration `Σ_k w_k·Φ(a + s·g·x_k) = μ(m)` — independent code that never
/// writes a single `a_{(i,j)}` formula — so a sign/algebra slip in any
/// intercept derivative or ℓ cross block is loud here, term by term.
///
/// The program expresses `μ(m)` via the link map's own `(mu, mu1..mu4)` jet
/// seed (the marginal link is family data, not the intercept calculus under
/// test) and the signed-probit NLL chain via the shared
/// `signed_probit_neglog_unary_stack` — the same single scalar source of truth
/// the hand kernel consumes, so agreement isolates the implicit-intercept and
/// Faà-di-Bruno assembly the hand path maintains.
struct EmpiricalRigidNllProgram {
    /// Per-row converged intercept scalar root (the jet-Newton seed/anchor).
    a_root: Vec<f64>,
    /// Per-row marginal link map (mu + mu1..mu4 for the μ(m) jet seed).
    marginal: Vec<BernoulliMarginalLinkMap>,
    /// Per-row slope g and latent score z and weight w and ±1 sign.
    slope: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    sign: Vec<f64>,
    /// Probit frailty scale s (shared across rows).
    s: f64,
    /// The empirical calibration grid (shared global grid).
    nodes: Vec<f64>,
    grid_w: Vec<f64>,
}

impl gam_math::jet_tower::RowProgram<2> for EmpiricalRigidNllProgram {
    fn n_rows(&self) -> usize {
        self.a_root.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        // Primary 0 = marginal eta m; primary 1 = slope g.
        Ok([self.marginal[row].q, self.slope[row]])
    }

    fn eval<S: gam_math::jet_scalar::JetScalar<2>>(
        &self,
        row: usize,
        p: &[S; 2],
    ) -> Result<S, String> {
        // p[0] is the seeded m-jet, p[1] the seeded g-jet. The marginal link
        // value μ(m) and its m-derivatives enter as a constant-in-g jet seeded
        // from the link map (μ depends only on m = primary 0).
        let g = p[1];
        let lm = self.marginal[row];
        let mu = p[0].compose_unary([lm.mu, lm.mu1, lm.mu2, lm.mu3, lm.mu4]);

        let s = self.s;
        // observed_slope = s·g, a jet in g.
        let observed_slope = g.scale(s);

        // Jet-Newton for the intercept jet a(m, g) solving the implicit
        // calibration G(a, m, g) = Σ_k w_k·Φ(a + s·g·x_k) − μ(m) = 0.
        // Seed at the converged scalar root; Newton in the jet ring converges
        // every derivative order quadratically, so a fixed handful of steps
        // pins all of a_{(i,j)} for i+j ≤ 4 to full precision. This is the
        // fully-independent derivation: no a_m/a_mm/… formula is written.
        let mut a = S::constant(self.a_root[row]);
        for _ in 0..8 {
            // G(a) and G_a(a) accumulated in one grid pass over the jet algebra.
            let mut gsum = S::constant(0.0).sub(&mu); // − μ(m)
            let mut ga = S::constant(0.0);
            for (&node, &weight) in self.nodes.iter().zip(self.grid_w.iter()) {
                // η_k = a + (s·g)·x_k, a jet in (m via a, g).
                let eta_k = a.add(&observed_slope.scale(node));
                let cdf = eta_k.compose_unary(unary_derivatives_normal_cdf(eta_k.value()));
                let pdf = eta_k.compose_unary(unary_derivatives_normal_pdf(eta_k.value()));
                gsum = gsum.add(&cdf.scale(weight));
                ga = ga.add(&pdf.scale(weight));
            }
            // Newton update a ← a − G / G_a in the jet ring.
            a = a.sub(&gsum.mul(&ga.recip()));
        }

        // Observed index η = a + (s·g)·z; z is this row's data constant.
        let z = self.z[row];
        let eta = a.add(&observed_slope.scale(z));
        let signed = eta.scale(self.sign[row]);
        // ONE transcendental: the shared signed-probit NLL stack, the same
        // single scalar source of truth the hand kernel consumes.
        if !(signed.value().is_finite() || signed.value() == f64::INFINITY) {
            return Err(format!(
                "empirical rigid nll program: non-finite signed margin {} at row {row}",
                signed.value()
            ));
        }
        let stack = signed_probit_neglog_unary_stack(signed.value(), self.w[row]);
        if !stack[0].is_finite() {
            return Err(format!(
                "empirical rigid nll program: non-finite log Φ at row {row}"
            ));
        }
        Ok(signed.compose_unary(stack))
    }
}

#[test]
fn empirical_rigid_row_kernel_agrees_with_jet_tower_program_all_channels() {
    use gam_math::jet_tower::{KernelChannels, program_full_tower, verify_kernel_channels};

    let (family, grid, marginal_etas, slopes) = empirical_rigid_fd_fixture();
    let s = family.probit_frailty_scale();

    // Build the independent jet program over the same grid / marginals /
    // converged intercept roots the hand kernel uses.
    let mut a_root = Vec::new();
    let mut marginal = Vec::new();
    let mut slope = Vec::new();
    let mut zv = Vec::new();
    let mut wv = Vec::new();
    let mut signv = Vec::new();
    for row in 0..3 {
        let (m, g) = (marginal_etas[row], slopes[row]);
        let lm = family
            .marginal_link_map(m)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "link map", e));
        let root = family
            .empirical_rigid_intercept_for_row(row, lm, g, &grid.nodes, &grid.weights)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "intercept root", e));
        a_root.push(root);
        marginal.push(lm);
        slope.push(g);
        zv.push(family.z[row]);
        wv.push(family.weights[row]);
        signv.push(2.0 * family.y[row] - 1.0);
    }
    let program = EmpiricalRigidNllProgram {
        a_root,
        marginal,
        slope,
        z: zv,
        w: wv,
        sign: signv,
        s,
        nodes: grid.nodes.to_vec(),
        grid_w: grid.weights.to_vec(),
    };

    // Several deterministic direction vectors so every (m, g) cross block of
    // the third/fourth tensors participates in the contraction.
    let dirs: [[f64; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [0.7, -1.3], [-0.4, 0.9]];

    for row in 0..3 {
        let (m, g) = (marginal_etas[row], slopes[row]);
        let lm = family
            .marginal_link_map(m)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "link map", e));

        // First, confirm the program VALUE channel equals the production NLL
        // exactly (the program is only a valid oracle if its value path is the
        // production NLL). Then audit every derivative channel.
        let (hand_v, hand_grad, hand_hess) = family
            .empirical_rigid_primary_grad_hess_closed_form(row, lm, g, &grid.nodes, &grid.weights)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "primary closed form", e));

        // Hand third / fourth full tensors, contracted along each direction.
        let hand_third_full = family
            .empirical_rigid_third_full_closed_form(row, lm, g, &grid.nodes, &grid.weights)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "third closed form", e));
        let hand_fourth_full = family
            .empirical_rigid_fourth_full_closed_form(row, lm, g, &grid.nodes, &grid.weights)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "fourth closed form", e));

        let third: Vec<([f64; 2], [[f64; 2]; 2])> = dirs
            .iter()
            .map(|d| (*d, contract_third_full(&hand_third_full, d[0], d[1])))
            .collect();
        let fourth: Vec<([f64; 2], [f64; 2], [[f64; 2]; 2])> = dirs
            .iter()
            .flat_map(|u| {
                dirs.iter().map(move |v| {
                    (
                        *u,
                        *v,
                        contract_fourth_full(&hand_fourth_full, u[0], u[1], v[0], v[1]),
                    )
                })
            })
            .collect();

        let claims = KernelChannels::<2> {
            value: hand_v,
            gradient: hand_grad,
            hessian: hand_hess,
            third,
            fourth,
        };

        let tower = program_full_tower(&program, row)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "program tower", e));
        verify_kernel_channels(&tower, &claims, 1e-7).unwrap_or_else(|e| {
            panic!("empirical rigid jet-tower oracle mismatch at row {row}: {e}")
        });
    }
}

#[test]
fn gaussian_rigid_intercept_miscalibrates_skewed_empirical_law() {
    let nodes = vec![-0.95, -0.7, -0.45, -0.2, 0.4, 1.3, 3.1];
    let weights = vec![0.28, 0.22, 0.17, 0.13, 0.1, 0.07, 0.03];
    let target_q = -0.15;
    let target_mu = normal_cdf(target_q);
    let slope = 1.4;
    let scale = 0.9;

    let gaussian_intercept = rigid_intercept_from_marginal(target_q, slope, scale);
    let gaussian_mu = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&node, &weight)| weight * normal_cdf(gaussian_intercept + scale * slope * node))
        .sum::<f64>();
    assert!(
        (gaussian_mu - target_mu).abs() > 1e-3,
        "skewed empirical law should not be calibrated by Gaussian identity"
    );

    let empirical_intercept = empirical_intercept_from_marginal(
        target_mu, target_q, slope, scale, &nodes, &weights, None,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "empirical intercept", e));
    let empirical_mu = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&node, &weight)| weight * normal_cdf(empirical_intercept + scale * slope * node))
        .sum::<f64>();
    assert!((empirical_mu - target_mu).abs() <= 1e-10);
}

/// Pathological warm-start: a stale cached intercept buried so deep in
/// the left tail that every quadrature node `ηᵢ = a + b·zᵢ` underflows
/// `φ(ηᵢ)` and `Φ(ηᵢ)` to exactly zero in linear arithmetic. The legacy
/// linear-space evaluator computed `f' = Σ wᵢ φᵢ = 0.0` and rejected the
/// state as invalid; the log-space reformulation evaluates `log Φ(η)` via
/// `erfcx` with full precision in any tail and produces a globally
/// monotone, finite-derivative residual that the monotone-root solver
/// can drive back to the basin from any seed. This exercises that path
/// directly so a regression to the linear-space formulation breaks here.
#[test]
fn empirical_intercept_recovers_from_deep_tail_warm_start() {
    let nodes = vec![-2.0, -0.25, 0.5, 3.0];
    let weights = vec![0.2, 0.3, 0.1, 0.4];
    let target_q = -0.35;
    let target_mu = normal_cdf(target_q);
    let slope = 0.8;
    let scale = 0.9;
    // Warm start at a = -100: every η = -100 + 0.72·z lies in the deep
    // left tail, where linear-space φ and Φ both round to 0.0 in IEEE-754.
    let stale_warm_start = Some(-100.0_f64);
    let intercept = empirical_intercept_from_marginal(
        target_mu,
        target_q,
        slope,
        scale,
        &nodes,
        &weights,
        stale_warm_start,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "empirical intercept must recover from deep-tail warm start", e
        )
    });
    let calibrated = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&node, &weight)| weight * normal_cdf(intercept + scale * slope * node))
        .sum::<f64>();
    assert!(
        (calibrated - target_mu).abs() <= 1e-10,
        "calibrated mu={calibrated} should match target_mu={target_mu} from any seed"
    );
}

/// Same recovery contract on the right tail (a +∞-side stale warm start),
/// where it is `1 − Φ` that underflows in linear arithmetic. The
/// log-space residual is bounded above by `−log μ★ < ∞`, and the natural
/// Newton step in this region is small (`F'(a) → 0` slowly), so the
/// solver still drives `F → 0` from any seed.
#[test]
fn empirical_intercept_recovers_from_far_right_warm_start() {
    let nodes = vec![-2.0, -0.25, 0.5, 3.0];
    let weights = vec![0.2, 0.3, 0.1, 0.4];
    let target_q = -0.35;
    let target_mu = normal_cdf(target_q);
    let slope = 0.8;
    let scale = 0.9;
    let stale_warm_start = Some(50.0_f64);
    let intercept = empirical_intercept_from_marginal(
        target_mu,
        target_q,
        slope,
        scale,
        &nodes,
        &weights,
        stale_warm_start,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "empirical intercept must recover from far-right warm start", e
        )
    });
    let calibrated = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&node, &weight)| weight * normal_cdf(intercept + scale * slope * node))
        .sum::<f64>();
    assert!(
        (calibrated - target_mu).abs() <= 1e-10,
        "calibrated mu={calibrated} should match target_mu={target_mu} from any seed"
    );
}

/// Weighted covariance of two equal-length vectors. Test helper for the
/// conditional latent-z calibration tests below.
fn weighted_cov_for_test(a: &Array1<f64>, b: &Array1<f64>, w: &Array1<f64>) -> f64 {
    let sw = w.sum();
    let ma = a.iter().zip(w.iter()).map(|(&x, &wi)| wi * x).sum::<f64>() / sw;
    let mb = b.iter().zip(w.iter()).map(|(&x, &wi)| wi * x).sum::<f64>() / sw;
    a.iter()
        .zip(b.iter())
        .zip(w.iter())
        .map(|((&x, &y), &wi)| wi * (x - ma) * (y - mb))
        .sum::<f64>()
        / sw
}

/// #905: the conditional `E[z|C]`/`Var(z|C)` Rao gate must fire when the
/// latent score has a conditional mean shift on the marginal-index span — the
/// `b(C)·m(C)` leakage the pooled-marginal gate cannot see — and the resulting
/// `ζ = (z − m(C))/√v(C)` must be conditionally centered (the leakage removed).
#[test]
fn conditional_latent_gate_detects_and_removes_conditional_mean_shift() {
    let n = 400usize;
    // Conditioning covariate c (the "PC"): a centered linspace.
    let c = Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64 - 1.0) * 2.0 - 1.0));
    // z = 0.8·c + mean-zero noise decorrelated from c. Marginally this carries
    // a conditional mean shift E[z|c] = 0.8·c ≠ 0.
    let z = Array1::from_iter((0..n).map(|i| 0.8 * c[i] + if i % 2 == 0 { 0.35 } else { -0.35 }));
    let weights = Array1::ones(n);
    let a_block = c.clone().insert_axis(ndarray::Axis(1));

    // Pre-correction: z is strongly correlated with c.
    let cov_before = weighted_cov_for_test(&c, &z, &weights);
    assert!(
        cov_before.abs() > 0.1,
        "synthetic z must carry a conditional mean shift on c (cov={cov_before})"
    );

    let cal = fit_conditional_latent_calibration_if_needed(&z, &weights, a_block.view())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "conditional gate must not error", e))
        .expect("conditional Rao gate must fire on a clear conditional mean shift");
    assert_eq!(cal.basis_ncols, 1);

    let zeta = cal
        .apply(z.view(), a_block.view())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "conditional calibration applies", e));
    let cov_after = weighted_cov_for_test(&c, &zeta, &weights);
    assert!(
        cov_after.abs() < 1.0e-6,
        "ζ must be conditionally centered on c (cov_before={cov_before}, cov_after={cov_after})"
    );
    // The post-correction sanity-check moments are recorded.
    assert!(cal.post_mean.abs() < 1.0e-6, "post_mean={}", cal.post_mean);
}

/// Regression test on `weighted_ridge_sandwich_cov` directly: the HC0 sandwich
/// must be FINITE on a numerically rank-deficient normal matrix, the smallest
/// failure mode behind the "conditional latent calibration sandwich covariance
/// is non-finite" production error. Two identical informative columns make
/// `AᵀWA` exactly rank 1 inside a 2-D system; even with a diagonal-relative
/// `1e-8` Tikhonov the explicit `M⁻¹` form blows up the sandwich, but the SPD
/// pseudo-inverse path projects out the non-identified direction and the
/// returned covariance is finite and PSD on the identifiable span.
#[test]
fn weighted_ridge_sandwich_cov_is_finite_on_rank_deficient_normal_matrix() {
    let n = 1_024usize;
    // Two perfectly collinear basis columns: `AᵀA` is rank 1 in a 2-D system.
    let mut basis = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x = (i as f64) / (n as f64 - 1.0) * 2.0 - 1.0;
        basis[[i, 0]] = x;
        basis[[i, 1]] = x;
    }
    let weights = Array1::ones(n);
    // Residuals carry signal in the (rank-1) identifiable direction.
    let residuals: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let total = basis.column(0).iter().map(|x| x * x).sum::<f64>();
    // Match the caller's recipe: `M = AᵀWA + λ · diag(AᵀWA)` with `λ = 1e-8`.
    // Both diagonals equal `Σ x²`, both off-diagonals equal `Σ x²`, so M has
    // eigenvalues `(2 + 1e-8)·total` and `1e-8·total` — the second is barely
    // above zero, the ratio is 2/1e-8 = 2e8 cond.
    let mut normal_matrix = Array2::<f64>::from_elem((2, 2), total);
    normal_matrix[[0, 0]] *= 1.0 + AUTO_Z_CONDITIONAL_RIDGE_REL;
    normal_matrix[[1, 1]] *= 1.0 + AUTO_Z_CONDITIONAL_RIDGE_REL;

    let cov = weighted_ridge_sandwich_cov(basis.view(), &residuals, weights.view(), &normal_matrix)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "rank-deficient normal matrix must yield a finite sandwich via pseudo-inverse", e
            )
        });

    assert_eq!(cov.dim(), (2, 2));
    assert!(
        cov.iter().all(|v| v.is_finite()),
        "sandwich covariance must be finite; got {:?}",
        cov
    );
    // Symmetric.
    assert!(
        (cov[[0, 1]] - cov[[1, 0]]).abs() < 1.0e-9,
        "sandwich covariance must be symmetric"
    );
    // The identifiable direction is `(1, 1)/√2`; its variance must be positive.
    let one = ndarray::array![1.0, 1.0];
    let projected = one.dot(&cov.dot(&one)) * 0.5;
    assert!(
        projected.is_finite() && projected > 0.0,
        "variance in the identifiable direction must be finite positive: {projected}"
    );
}

/// Regression test: the Murphy–Topel first-stage sandwich must be FINITE on a
/// wide rank-deficient conditioning basis. The large-scale benchmarks tripped
/// "conditional latent calibration sandwich covariance is non-finite" whenever
/// the marginal design (smooth + 16-D duchon + linkwiggle) contributed a
/// near-null direction to the conditioning span — the explicit `M⁻¹` of the
/// near-singular normal matrix blew the sandwich through f64 range. Replacing
/// it with the SPD pseudo-inverse `M⁺` makes the non-identified directions
/// contribute zero variance (the correct asymptotic answer for parameters not
/// identified by the first-stage data) and the sandwich finite.
#[test]
fn conditional_latent_gate_handles_rank_deficient_conditioning_basis() {
    let n = 2_000usize;
    let p_widen = 8usize;
    let informative: Vec<f64> = (0..n)
        .map(|i| (i as f64) / (n as f64 - 1.0) * 2.0 - 1.0)
        .collect();
    // Build a deliberately rank-deficient wide conditioning basis: one
    // informative column carrying the conditional mean shift, four duplicates
    // (perfect collinearity), and three constant-1 columns (collinear with the
    // prepended intercept). The weighted Gram of `[1 | a(C)]` therefore has
    // effective rank 2 inside a 9-dimensional system, mirroring the structural
    // collinearity the freeze + identifiability pipeline can leave in a wide
    // marginal design.
    let mut a_block = Array2::<f64>::zeros((n, p_widen));
    for j in 0..p_widen {
        for i in 0..n {
            a_block[[i, j]] = if j < p_widen / 2 { informative[i] } else { 1.0 };
        }
    }
    // z carries a conditional mean shift on the informative column so the gate
    // fires and the sandwich path is exercised end-to-end.
    let z = Array1::from_iter(
        (0..n).map(|i| 0.8 * informative[i] + if i % 2 == 0 { 0.35 } else { -0.35 }),
    );
    let weights = Array1::ones(n);

    let cal = fit_conditional_latent_calibration_if_needed(&z, &weights, a_block.view())
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "conditional gate must not error on a rank-deficient conditioning basis", e
            )
        })
        .expect("conditional Rao gate must fire on a clear conditional mean shift");

    assert!(
        cal.mean_cov.iter().all(|v| v.is_finite()),
        "mean sandwich covariance must be finite on rank-deficient conditioning"
    );
    assert!(
        cal.var_cov.iter().all(|v| v.is_finite()),
        "variance sandwich covariance must be finite on rank-deficient conditioning"
    );

    // The covariance is a valid PSD matrix in the identifiable subspace — its
    // diagonal entries must be non-negative.
    for j in 0..cal.mean_cov.nrows() {
        assert!(
            cal.mean_cov[[j, j]] >= -1.0e-12,
            "mean sandwich diagonal must be PSD: cov[{j},{j}]={}",
            cal.mean_cov[[j, j]]
        );
    }

    // The calibration must still remove the conditional mean shift even when
    // some directions of the basis are non-identified: the Rao gate uses a
    // pseudo-inverse internally, and the fitted m̂(C) lives in the identifiable
    // span of the basis, so ζ = (z − m̂(C))/√v̂(C) must be conditionally
    // centered on the informative direction.
    let zeta = cal.apply(z.view(), a_block.view()).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "conditional calibration applies on rank-deficient basis", e
        )
    });
    let informative_arr = Array1::from(informative);
    let cov_after = weighted_cov_for_test(&informative_arr, &zeta, &weights);
    assert!(
        cov_after.abs() < 1.0e-4,
        "ζ must be conditionally centered on the informative direction (cov={cov_after})"
    );
}

/// #905: with NO conditional structure (z independent of the conditioning
/// span), the Rao gate must NOT fire — the conditional correction is reserved
/// for genuine conditional shifts, leaving the pooled-marginal gate in charge.
#[test]
fn conditional_latent_gate_silent_without_conditional_structure() {
    let n = 400usize;
    let c = Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64 - 1.0) * 2.0 - 1.0));
    // z alternates sign independently of the smooth c: cov(c, z) ≈ 0 and the
    // squared residual is constant, so neither the mean nor the variance Rao
    // block has anything to detect.
    let z = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }));
    let weights = Array1::ones(n);
    let a_block = c.insert_axis(ndarray::Axis(1));

    let result = fit_conditional_latent_calibration_if_needed(&z, &weights, a_block.view())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "conditional gate must not error", e));
    assert!(
        result.is_none(),
        "conditional gate must stay silent when z carries no conditional structure"
    );
}

/// #905: the Auto path routes a conditional shift to the conditional
/// location-scale calibration when a conditioning span is supplied, but falls
/// back to the pooled-marginal path when none is (the no-CTN raw-z gap the
/// issue names is closed only when the marginal-index span is available).
#[test]
fn auto_latent_measure_routes_conditional_shift_to_location_scale() {
    let n = 400usize;
    let c = Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64 - 1.0) * 2.0 - 1.0));
    let z = Array1::from_iter((0..n).map(|i| 0.8 * c[i] + if i % 2 == 0 { 0.35 } else { -0.35 }));
    let weights = Array1::ones(n);
    let a_block = c.insert_axis(ndarray::Axis(1));
    let policy = LatentZPolicy {
        check_mode: LatentZCheckMode::Off,
        normalization: LatentZNormalizationMode::None,
        latent_measure: LatentMeasureSpec::Auto { grid_size: 5 },
        ..LatentZPolicy::default()
    };

    let (measure, calibration) =
        build_latent_measure_with_geometry(&z, &weights, &policy, Some(a_block.view()))
            .unwrap_or_else(|e| {
                panic!(
                    "{} failed: {:?}",
                    "auto latent measure with conditioning", e
                )
            });
    assert!(matches!(measure, LatentMeasureKind::StandardNormal));
    assert!(
        matches!(
            calibration,
            LatentMeasureCalibration::ConditionalLocationScale(_)
        ),
        "a conditional shift with a conditioning span must route to the conditional correction"
    );

    // Without the conditioning span the Auto path cannot see the conditional
    // shift and falls back to the pooled-marginal decision (here: no
    // calibration, since this z passes the pooled-normal diagnostics well
    // enough — exactly the leak the issue describes).
    let (_measure_blind, calibration_blind) =
        build_latent_measure_with_geometry(&z, &weights, &policy, None).unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "auto latent measure without conditioning", e
            )
        });
    assert!(
        !matches!(
            calibration_blind,
            LatentMeasureCalibration::ConditionalLocationScale(_)
        ),
        "without a conditioning span the conditional correction cannot be selected"
    );
}

#[test]
fn auto_latent_measure_preserves_standard_normal_fast_path() {
    let n = 2001usize;
    let z = Array1::from_iter((0..n).map(|idx| {
        let p = (idx as f64 + 0.5) / n as f64;
        standard_normal_quantile(p)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "normal quantile", e))
    }));
    let weights = Array1::ones(n);
    let policy = LatentZPolicy {
        latent_measure: LatentMeasureSpec::Auto { grid_size: 17 },
        ..LatentZPolicy::default()
    };
    let (measure, calibration) = build_latent_measure_with_geometry(&z, &weights, &policy, None)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "measure", e));

    assert!(matches!(measure, LatentMeasureKind::StandardNormal));
    assert!(
        matches!(calibration, LatentMeasureCalibration::None),
        "well-conditioned standard-normal z must skip rank-INT calibration"
    );
    let slope = 0.7;
    let scale = 0.85;
    let target_q = 0.4;
    assert_eq!(
        rigid_intercept_from_marginal(target_q, slope, scale),
        target_q * (1.0 + (scale * slope).powi(2)).sqrt()
    );
}

#[test]
fn flexible_family_exposes_exact_newton_workspaces() {
    let (family, block_states) = dual_flex_exact_fixture();
    let n = family.z.len();
    let specs = vec![
        dummy_blockspec(1, n),
        dummy_blockspec(1, n),
        dummy_blockspec(block_states[2].beta.len(), n),
        dummy_blockspec(block_states[3].beta.len(), n),
    ];
    let derivative_blocks = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let hyper_layout = rigid_test_design_hyper_layout(&derivative_blocks);

    assert!(
        family
            .exact_newton_joint_hessian_workspace(&block_states, &specs)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "flex hessian workspace", e))
            .is_some()
    );
    assert!(
        family
            .exact_newton_joint_psi_workspace(&block_states, &specs, &hyper_layout)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "flex psi workspace", e))
            .is_some()
    );
}

#[test]
fn sigma_exact_joint_psi_terms_returns_analytic_terms() {
    let z = array![-0.8, 0.2, 1.1];
    let y = array![0.0, 1.0, 1.0];
    let weights = array![1.0, 0.7, 1.3];
    let sigma = 0.7;
    let make_family = |sigma: f64| BernoulliMarginalSlopeFamily {
        gaussian_frailty_sd: Some(sigma),
        ..test_family_with_intercept_designs(y.clone(), weights.clone(), z.clone())
    };
    let family = make_family(sigma);
    let block_states = vec![
        scalar_block_state(0.25, z.len()),
        scalar_block_state(0.6, z.len()),
    ];
    let specs = vec![dummy_blockspec(1, z.len()), dummy_blockspec(1, z.len())];

    let terms = family
        .sigma_exact_joint_psi_terms(&block_states, &specs)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "analytic sigma psi terms", e))
        .expect("sigma terms present");
    assert!(terms.objective_psi.is_finite());
    assert_eq!(terms.score_psi.len(), 2);
    assert!(terms.score_psi.iter().all(|value| value.is_finite()));
    assert_eq!(
        terms
            .hessian_psi_operator
            .as_ref()
            .expect("sigma Hessian operator")
            .to_dense()
            .dim(),
        (2, 2)
    );

    let second = family
        .sigma_exact_joint_psisecond_order_terms(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "analytic second sigma terms", e))
        .expect("second sigma terms present");
    assert!(second.objective_psi_psi.is_finite());
    assert_eq!(second.score_psi_psi.len(), 2);

    let drift = family
        .sigma_exact_joint_psihessian_directional_derivative(&block_states, &array![0.1, -0.2])
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "analytic sigma Hessian directional derivative", e
            )
        })
        .expect("sigma drift present");
    assert_eq!(drift.dim(), (2, 2));
    assert!(drift.iter().all(|value| value.is_finite()));

    let tau = sigma.ln();
    let eps = 1e-5;
    let ll_plus = make_family((tau + eps).exp())
        .log_likelihood_only(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll plus sigma", e));
    let ll_minus = make_family((tau - eps).exp())
        .log_likelihood_only(&block_states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "ll minus sigma", e));
    let objective_fd = -(ll_plus - ll_minus) / (2.0 * eps);
    assert!((terms.objective_psi - objective_fd).abs() < 1e-5);
}

/// Multi-row rigid-probit family with non-trivial marginal/logslope
/// designs (per-row eta varies), so half-mask Horvitz-Thompson rescaling
/// over even rows is a representative subsample for the block-path
/// `exact_newton_joint_psi_terms_from_cache` and its second-order sibling.
fn make_block_psi_test_family(n: usize) -> BernoulliMarginalSlopeFamily {
    let y: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 7) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.5 + 3.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
    );
    let marginal_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.3 + 0.4 * (((i * 29 + 11) % n) as f64) / (n as f64)
    });
    let logslope_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.2 + 0.5 * (((i * 37 + 9) % n) as f64) / (n as f64)
    });
    BernoulliMarginalSlopeFamily {
        y: Arc::new(y),
        weights: Arc::new(weights),
        z: Arc::new(z),
        marginal_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            marginal_design,
        )),
        logslope_design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            logslope_design,
        )),
        ..default_test_family()
    }
}

fn block_psi_test_block_states(
    family: &BernoulliMarginalSlopeFamily,
    m_beta: f64,
    g_beta: f64,
) -> Vec<ParameterBlockState> {
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family.logslope_design.to_dense().to_owned();
    let m_eta = m_design.dot(&array![m_beta]);
    let g_eta = g_design.dot(&array![g_beta]);
    vec![
        ParameterBlockState {
            beta: array![m_beta],
            eta: m_eta,
        },
        ParameterBlockState {
            beta: array![g_beta],
            eta: g_eta,
        },
    ]
}

fn block_psi_test_marginal_derivative_blocks(
    n: usize,
) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
    let x_psi = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.4 + 0.3 * (((i * 41 + 13) % n) as f64) / (n as f64)
    });
    vec![
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ]
}

fn block_psi_test_dual_derivative_blocks(
    n: usize,
) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
    let x_psi_m = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.4 + 0.3 * (((i * 41 + 13) % n) as f64) / (n as f64)
    });
    let x_psi_g = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.2 + 0.5 * (((i * 43 + 17) % n) as f64) / (n as f64)
    });
    vec![
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi_m,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi_g,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
    ]
}

#[test]
fn bernoulli_psi_terms_from_cache_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let cache = family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));

    let baseline = family
        .exact_newton_joint_psi_terms_from_cache(&states, &derivative_blocks, 0, &cache)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline psi terms", e))
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .exact_newton_joint_psi_terms_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &cache,
            &opts_full,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "with full", e))
        .expect("some");

    let obj_rel = ((with_full.objective_psi - baseline.objective_psi)
        / baseline.objective_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let score_rel = rel_diff_array1(&with_full.score_psi, &baseline.score_psi);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn bernoulli_psi_terms_from_cache_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let cache = family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .exact_newton_joint_psi_terms_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &cache,
            &opts_half,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "scaled", e))
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .exact_newton_joint_psi_terms_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &cache,
            &opts_raw,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "raw", e))
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi;
    let obj_rel = ((scaled.objective_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let exp_score = &raw.score_psi * factor;
    let score_rel = rel_diff_array1(&scaled.score_psi, &exp_score);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn bernoulli_psi_second_order_terms_from_cache_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);
    let cache = family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));

    let baseline = family
        .exact_newton_joint_psisecond_order_terms_from_cache(
            &states,
            &derivative_blocks,
            0,
            1,
            &cache,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline psi second-order", e))
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .exact_newton_joint_psisecond_order_terms_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            &cache,
            &opts_full,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "with full", e))
        .expect("some");

    let obj_rel = ((with_full.objective_psi_psi - baseline.objective_psi_psi)
        / baseline.objective_psi_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let score_rel = rel_diff_array1(&with_full.score_psi_psi, &baseline.score_psi_psi);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn bernoulli_psi_second_order_terms_from_cache_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);
    let cache = family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .exact_newton_joint_psisecond_order_terms_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            &cache,
            &opts_half,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "scaled", e))
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .exact_newton_joint_psisecond_order_terms_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            &cache,
            &opts_raw,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "raw", e))
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi_psi;
    let obj_rel = ((scaled.objective_psi_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let exp_score = &raw.score_psi_psi * factor;
    let score_rel = rel_diff_array1(&scaled.score_psi_psi, &exp_score);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn bernoulli_psihessian_directional_derivative_from_cache_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let cache = family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let slices = &cache.slices;
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let baseline = family
        .exact_newton_joint_psihessian_directional_derivative_from_cache(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &cache,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline", e))
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .exact_newton_joint_psihessian_directional_derivative_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &cache,
            &opts_full,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "with full", e))
        .expect("some");

    let rel = rel_diff_array2(&with_full, &baseline);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn bernoulli_psihessian_directional_derivative_from_cache_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let cache = family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let slices = &cache.slices;
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .exact_newton_joint_psihessian_directional_derivative_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &cache,
            &opts_half,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "scaled", e))
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .exact_newton_joint_psihessian_directional_derivative_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &cache,
            &opts_raw,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "raw", e))
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp = &raw * factor;
    let rel = rel_diff_array2(&scaled, &exp);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn bernoulli_psihessian_operator_from_cache_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let cache = family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let slices = &cache.slices;
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let baseline = family
        .exact_newton_joint_psihessian_directional_derivative_operator_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &cache,
            &BlockwiseFitOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline operator", e))
        .expect("some");
    let baseline_dense = baseline.to_dense();

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .exact_newton_joint_psihessian_directional_derivative_operator_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &cache,
            &opts_full,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "with full", e))
        .expect("some");
    let with_full_dense = with_full.to_dense();

    let rel = rel_diff_array2(&with_full_dense, &baseline_dense);
    assert!(rel < 1e-12, "operator drift rel {}", rel);
}

#[test]
fn bernoulli_psihessian_operator_from_cache_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let cache = family
        .build_exact_eval_cache(&states)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact eval cache", e));
    let slices = &cache.slices;
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .exact_newton_joint_psihessian_directional_derivative_operator_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &cache,
            &opts_half,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "scaled", e))
        .expect("some");
    let scaled_dense = scaled.to_dense();

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .exact_newton_joint_psihessian_directional_derivative_operator_from_cache_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &cache,
            &opts_raw,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "raw", e))
        .expect("some");
    let raw_dense = raw.to_dense();

    let factor = n as f64 / m as f64;
    let exp = &raw_dense * factor;
    let rel = rel_diff_array2(&scaled_dense, &exp);
    assert!(rel < 1e-12, "operator drift rel {}", rel);
}
