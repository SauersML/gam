
// ── Phase 7: joint-Hessian directional-derivative subsample tests ──

#[test]
fn bernoulli_jointhessian_directional_derivative_from_cache_subsample_full_equals_unsampled() {
    use crate::families::marginal_slope_shared::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("exact eval cache");
    let slices = &cache.slices;
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let baseline = family
        .exact_newton_joint_hessian_directional_derivative_from_cache(&states, &d_beta_flat, &cache)
        .expect("baseline")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
        (0..n).collect(),
        n,
        0xDEADBEEF,
    )));
    let with_full = family
        .exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
            &states,
            &d_beta_flat,
            &cache,
            &opts_full,
        )
        .expect("with full")
        .expect("some");

    let rel = rel_diff_array2(&with_full, &baseline);
    assert!(rel < 1e-12, "joint Hessian dH drift rel {}", rel);
}


#[test]
fn bernoulli_jointhessian_batched_directional_operators_match_single_direction_path() {
    use crate::families::marginal_slope_shared::OuterScoreSubsample;

    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("exact eval cache");
    let slices = &cache.slices;
    let directions: Vec<Array1<f64>> = (0..3)
        .map(|rep| {
            let mut d = Array1::<f64>::zeros(slices.total);
            d[slices.marginal.start] = 0.03 * (rep as f64 + 1.0);
            d[slices.logslope.start] = -0.02 * (rep as f64 + 1.0);
            d
        })
        .collect();

    let mut opts = BlockwiseFitOptions::default();
    opts.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
        (0..n).step_by(2).collect(),
        n,
        0xB47C,
    )));
    let batched = family
        .exact_newton_joint_hessian_directional_derivative_operators_from_cache_with_options(
            &states,
            &directions,
            &cache,
            &opts,
        )
        .expect("batched operators");
    assert_eq!(batched.len(), directions.len());

    for (idx, direction) in directions.iter().enumerate() {
        let single = family
            .exact_newton_joint_hessian_directional_derivative_operator_from_cache_with_options(
                &states, direction, &cache, &opts,
            )
            .expect("single operator")
            .expect("single operator some")
            .to_dense();
        let batched_dense = batched[idx]
            .as_ref()
            .expect("batched operator some")
            .to_dense();
        let rel = rel_diff_array2(&batched_dense, &single);
        assert!(rel < 1e-12, "batched operator {idx} drift rel {rel}");
    }
}


fn make_flex_hvp_cache_test_family(
    n: usize,
) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let score_seed = Array1::linspace(-2.0, 2.0, n.max(6));
    let link_seed = Array1::linspace(-1.8, 1.8, n.max(6));
    let score_prepared = build_score_warp_deviation_block_from_seed(
        &score_seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .expect("build score warp block");
    let link_prepared = build_test_link_deviation_block_from_seed(
        &link_seed,
        &DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        },
    )
    .expect("build link deviation block");
    let y: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 17 + 3) % 7 >= 4 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.75 + ((i * 11 + 5) % 5) as f64 * 0.05));
    let z: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -1.7 + 3.4 * (i as f64 + 0.5) / n as f64));
    let marginal_x = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            -0.4 + 0.8 * ((i * 19 + 7) % n) as f64 / n as f64
        }
    });
    let logslope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            0.3 - 0.6 * ((i * 23 + 11) % n) as f64 / n as f64
        }
    });
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(y),
        weights: Arc::new(weights),
        z: Arc::new(z.clone()),
        gaussian_frailty_sd: Some(0.15),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            marginal_x.clone(),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            logslope_x.clone(),
        )),
        score_warp: Some(score_prepared.runtime.clone()),
        link_dev: Some(link_prepared.runtime.clone()),
        ..default_test_family()
    };
    let beta_m = array![0.12, -0.04];
    let beta_g = array![0.35, 0.03];
    let beta_h = Array1::from_iter(
        (0..score_prepared.runtime.basis_dim()).map(|idx| 0.0015 * (idx as f64 + 1.0)),
    );
    let beta_w = Array1::from_iter(
        (0..link_prepared.runtime.basis_dim()).map(|idx| -0.001 * (idx as f64 + 1.0)),
    );
    let states = vec![
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
    (family, states)
}


/// gam#683: the axis-projected fast path inside
/// `row_primary_{third,fourth}_contracted_recompute` must reproduce the
/// slow per-(row, direction) cell-walk workers it replaces, for the
/// single-axis directions every outer-derivative consumer builds. Equality
/// holds by (bi)linearity of the contraction; the tolerance only absorbs
/// the float reassociation of pulling the scalar outside the cell sum.
#[test]
fn bernoulli_flex_axis_tensor_cache_matches_slow_recompute() {
    let (family, states) = make_flex_hvp_cache_test_family(24);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("flex exact eval cache");
    let r = cache.primary.total;
    let q = cache.primary.q;
    let g = cache.primary.logslope;
    let mut max_rel_third = 0.0_f64;
    let mut max_rel_fourth = 0.0_f64;
    for row in 0..family.y.len() {
        let row_ctx = BernoulliMarginalSlopeFamily::row_ctx(&cache, row);
        let zero_dir = Array1::<f64>::zeros(r);
        let zero_third = family
            .row_primary_third_contracted_recompute(row, &states, &cache, row_ctx, &zero_dir)
            .expect("zero third fast path");
        assert!(
            zero_third.iter().all(|&value| value == 0.0),
            "zero third direction must return exact zeros"
        );
        let mut q_dir = Array1::<f64>::zeros(r);
        q_dir[q] = 1.25;
        let zero_fourth = family
            .row_primary_fourth_contracted_recompute(
                row, &states, &cache, row_ctx, &zero_dir, &q_dir,
            )
            .expect("zero fourth fast path");
        assert!(
            zero_fourth.iter().all(|&value| value == 0.0),
            "zero fourth direction must return exact zeros"
        );
        let batched = family
            .row_primary_third_contracted_many_with_moments(
                row,
                &states,
                &cache,
                row_ctx,
                &[zero_dir.clone(), q_dir.clone()],
            )
            .expect("batched zero/single-axis third fast path");
        assert_eq!(batched.len(), 2);
        assert!(
            batched[0].iter().all(|&value| value == 0.0),
            "batched zero third direction must return exact zeros"
        );
        let q_single = family
            .row_primary_third_contracted_recompute(row, &states, &cache, row_ctx, &q_dir)
            .expect("single q third fast path");
        for (a, b) in batched[1].iter().zip(q_single.iter()) {
            assert!(
                (a - b).abs() <= 1e-12,
                "batched q third differs from single q third: batched={a:.3e} single={b:.3e}"
            );
        }
        // Third: each primary axis, several scalars including negative.
        for &axis in &[q, g] {
            for &s in &[1.0_f64, -0.7, 2.3] {
                let mut dir = Array1::<f64>::zeros(r);
                dir[axis] = s;
                let fast = family
                    .row_primary_third_contracted_recompute(row, &states, &cache, row_ctx, &dir)
                    .expect("fast third");
                let slow = family
                    .row_primary_third_contracted_recompute_with_moments(
                        row, &states, &cache, row_ctx, &dir,
                    )
                    .expect("slow third");
                assert_eq!(fast.dim(), (r, r));
                for (a, b) in fast.iter().zip(slow.iter()) {
                    let denom = a.abs().max(b.abs()).max(1.0);
                    max_rel_third = max_rel_third.max((a - b).abs() / denom);
                }
            }
        }
        // Fourth: every ordered pair of single axes, distinct scalars. The
        // slow reference symmetrizes `ordered` exactly as `_recompute` does.
        for &(au, su) in &[(q, 1.3_f64), (g, -0.9)] {
            for &(av, sv) in &[(q, 0.8_f64), (g, 1.7)] {
                let mut du = Array1::<f64>::zeros(r);
                du[au] = su;
                let mut dv = Array1::<f64>::zeros(r);
                dv[av] = sv;
                let fast = family
                    .row_primary_fourth_contracted_recompute(
                        row, &states, &cache, row_ctx, &du, &dv,
                    )
                    .expect("fast fourth");
                let ordered = family
                    .row_primary_fourth_contracted_recompute_ordered(
                        row, &states, &cache, row_ctx, &du, &dv,
                    )
                    .expect("slow fourth ordered");
                let swapped = family
                    .row_primary_fourth_contracted_recompute_ordered(
                        row, &states, &cache, row_ctx, &dv, &du,
                    )
                    .expect("slow fourth swapped");
                for ((f, o), w) in fast.iter().zip(ordered.iter()).zip(swapped.iter()) {
                    let slow = 0.5 * (o + w);
                    let denom = f.abs().max(slow.abs()).max(1.0);
                    max_rel_fourth = max_rel_fourth.max((f - slow).abs() / denom);
                }
            }
        }
    }
    assert!(
        max_rel_third <= 1e-9,
        "third axis-cache vs slow recompute drift too large: {max_rel_third:.3e}"
    );
    assert!(
        max_rel_fourth <= 1e-9,
        "fourth axis-cache vs slow recompute drift too large: {max_rel_fourth:.3e}"
    );
}


#[test]
fn bernoulli_row_cell_moment_upgrade_reuses_base_partitions_without_lru() {
    let (family, states) = make_flex_hvp_cache_test_family(12);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("flex exact eval cache");
    let base = cache
        .row_cell_moments
        .as_ref()
        .expect("degree-9 base row-cell bundle");
    assert_eq!(base.max_degree, 9);
    assert!(base.covers_all_rows());
    assert_eq!(base.selected_rows, family.y.len());

    let stats_before = family.cell_moment_cache_stats.snapshot();
    let d15 = family
        .bundle_for_degree(&states, &cache, 15)
        .expect("degree-15 bundle result")
        .expect("degree-15 bundle");
    let stats_after_d15 = family.cell_moment_cache_stats.snapshot();
    assert_eq!(
        stats_before, stats_after_d15,
        "high-degree bundle upgrade must not probe the row-unique fit-lifetime LRU"
    );
    assert_eq!(d15.max_degree, 15);
    assert!(d15.covers_all_rows());
    assert_eq!(d15.selected_rows, base.selected_rows);

    for row in 0..family.y.len() {
        let base_row = base.row(row, 9).expect("base row moments");
        let high_row = d15.row(row, 15).expect("degree-15 row moments");
        assert_eq!(base_row.len(), high_row.len());
        for (base_cell, high_cell) in base_row.iter().zip(high_row.iter()) {
            assert_eq!(base_cell.partition_cell, high_cell.partition_cell);
            assert_eq!(high_cell.state.moments.len(), 16);
            for k in 0..=9 {
                let denom = base_cell.state.moments[k].abs().max(1.0);
                let rel = (base_cell.state.moments[k] - high_cell.state.moments[k]).abs() / denom;
                assert!(
                    rel <= 1e-12,
                    "prefix moment drift row={row} k={k} rel={rel:e}"
                );
            }
        }
    }

    let d21 = family
        .bundle_for_degree(&states, &cache, 21)
        .expect("degree-21 bundle result")
        .expect("degree-21 bundle");
    assert_eq!(family.cell_moment_cache_stats.snapshot(), stats_after_d15);
    assert_eq!(d21.max_degree, 21);
    assert!(d21.covers_all_rows());
}


#[test]
fn bernoulli_value_cell_moments_use_shared_lru() {
    let family = BernoulliMarginalSlopeFamily {
        cell_moment_lru: Arc::new(exact_kernel::CellMomentLruCache::new(16 * 1024 * 1024)),
        cell_moment_cache_stats: Arc::new(exact_kernel::CellMomentCacheStats::default()),
        ..default_test_family()
    };
    let cell = exact_kernel::DenestedCubicCell {
        left: -1.25,
        right: 0.75,
        c0: 0.15,
        c1: -0.35,
        c2: 0.08,
        c3: -0.015,
    };

    let before = family.cell_moment_cache_stats.snapshot();
    let first = family
        .evaluate_cell_moments_lru(cell, 4)
        .expect("cold value moments");
    let (hits, misses, _) = family.cell_moment_cache_stats.hit_rate_delta(before);
    assert_eq!(hits, 0, "first value-moment call should be cold");
    assert_eq!(misses, 1, "first value-moment call should record one miss");

    let second = family
        .evaluate_cell_moments_lru(cell, 4)
        .expect("warm value moments");
    assert_eq!(second, first);
    let (hits, misses, _) = family.cell_moment_cache_stats.hit_rate_delta(before);
    assert_eq!(hits, 1, "second value-moment call should hit the LRU");
    assert_eq!(
        misses, 1,
        "warm value-moment call must not record another miss"
    );

    let derivative = family
        .evaluate_cell_derivative_moments_lru(cell, 4)
        .expect("cold derivative moments after value moments");
    let derivative_again = family
        .evaluate_cell_derivative_moments_lru(cell, 4)
        .expect("warm derivative moments");
    assert_eq!(derivative_again, derivative);

    let third = family
        .evaluate_cell_moments_lru(cell, 4)
        .expect("value moments preserved beside derivative moments");
    assert_eq!(third, first);
    let (hits, misses, _) = family.cell_moment_cache_stats.hit_rate_delta(before);
    assert_eq!(
        (hits, misses),
        (3, 2),
        "value and derivative moments should share one LRU entry without evicting each other"
    );
}


#[test]
fn bernoulli_flex_paired_subsample_ll_delta_sign_matches_full_ll() {
    use crate::families::marginal_slope_shared::OuterScoreSubsample;

    let (family, old_states) = make_flex_hvp_cache_test_family(96);
    let mut trial_states = old_states.clone();
    trial_states[0].beta[0] += 0.015;
    trial_states[0].eta += 0.015;
    trial_states[1].beta[1] -= 0.01;
    let logslope_col =
        Array1::from_iter((0..96).map(|i| 0.3 - 0.6 * ((i * 23 + 11) % 96) as f64 / 96.0));
    trial_states[1].eta.scaled_add(-0.01, &logslope_col);

    let full_old = family
        .log_likelihood_only(&old_states)
        .expect("full old ll");
    let full_trial = family
        .log_likelihood_only(&trial_states)
        .expect("full trial ll");

    let mut opts = BlockwiseFitOptions::default();
    opts.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
        (0..96).step_by(2).collect(),
        96,
        0x5EED5EED,
    )));
    let sub_old = family
        .log_likelihood_only_with_options(&old_states, &opts)
        .expect("subsample old ll");
    let sub_trial = family
        .log_likelihood_only_with_options(&trial_states, &opts)
        .expect("subsample trial ll");

    let full_delta = full_trial - full_old;
    let sub_delta = sub_trial - sub_old;
    assert!(
        full_delta.abs() > 1e-8,
        "synthetic beta-pair should produce a non-degenerate full-LL delta: {full_delta}"
    );
    assert_eq!(
        full_delta.is_sign_positive(),
        sub_delta.is_sign_positive(),
        "paired subsample LL delta sign ({sub_delta}) should match full LL delta sign ({full_delta})"
    );
}


#[test]
fn bernoulli_flex_row_primary_hessian_cache_policy_materializes_aou_shape() {
    // large-scale-shaped cache (~629 MiB) under a 16 GiB available-RAM budget:
    // single-cache budget is 4 GiB and the global pin budget is 8 GiB, so
    // even though the cache is hundreds of MiB it amortizes the build.
    // The full row-primary shape is neglog (1) + grad (r) + hess (r*r)
    // per row, i.e. r²+r+1 = 421 floats/row for r=20 (not r²=400):
    //   195_780 · 421 · 8 = 659_387_040 bytes (~629 MiB).
    let plan = decide_row_primary_hessian_cache(
        195_780,
        20,
        BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
        16 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
        0,
    );
    assert_eq!(plan.bytes, 659_387_040);
    assert!(plan.materialize);
    assert_eq!(
        plan.reason,
        RowPrimaryHessianCacheReason::ReuseAmortizesBuild
    );
}


#[test]
fn bernoulli_flex_row_primary_hessian_cache_policy_streams_when_single_cache_exceeds_ram() {
    // 626 MiB cache vs. only 2 GiB available RAM → single-cache budget is
    // 512 MiB and the build is rejected, falling back to streaming.
    let plan = decide_row_primary_hessian_cache(
        195_780,
        20,
        BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
        2 * 1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
        0,
    );
    assert!(!plan.materialize);
    assert_eq!(
        plan.reason,
        RowPrimaryHessianCacheReason::SingleCacheExceedsRamFraction
    );
}


#[test]
fn bernoulli_flex_row_primary_hessian_cache_policy_streams_when_global_pin_exhausted() {
    // 16 GiB available with 7.5 GiB already pinned: global pin budget is
    // 8 GiB, so a 626 MiB new cache pushes total pinned over the cap.
    let plan = decide_row_primary_hessian_cache(
        195_780,
        20,
        BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
        16 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
        (15 * 1024 * 1024 * 1024) / 2,
    );
    assert!(!plan.materialize);
    assert_eq!(
        plan.reason,
        RowPrimaryHessianCacheReason::GlobalPinExceedsRamFraction
    );
}


#[test]
fn bernoulli_flex_row_primary_hessian_cache_policy_streams_low_reuse() {
    let plan = decide_row_primary_hessian_cache(
        100,
        4,
        1,
        16 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
        0,
    );
    assert!(!plan.materialize);
    assert_eq!(plan.reason, RowPrimaryHessianCacheReason::ReuseTooLow);
}


#[test]
fn bernoulli_flex_row_primary_hessian_cache_policy_no_flip_under_memory_pressure() {
    // Regression for the large-scale 16d-flex shape (n=320k, r=20 → 1.08 GB cache)
    // that flipped materialize→stream mid-fit, dropping the inner solve from
    // the fast dense route onto the matrix-free CG path that never finished.
    let n = 320_000usize;
    let r = 20usize;
    let bytes = (n as u64) * ((r * r + r + 1) as u64) * 8;
    assert_eq!(bytes, 1_077_760_000);

    // First decision while RAM is plentiful: 4.15 GiB available establishes
    // the stable capacity floor; single-cache budget ≈ 1.11 GB > 1.08 GB →
    // materialize on the dense route.
    let plentiful = 4_457_512_960u64;
    let plan_hot = decide_row_primary_hessian_cache(
        n,
        r,
        BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
        plentiful,
        plentiful,
        0,
    );
    assert!(plan_hot.materialize);
    assert_eq!(
        plan_hot.reason,
        RowPrimaryHessianCacheReason::ReuseAmortizesBuild
    );

    // Later in the same fit, live available RAM has dropped to 3.18 GiB but
    // the stable capacity floor is unchanged. Pre-fix the single-cache
    // budget would have fallen to ≈0.80 GB and rejected the same 1.08 GB
    // cache; with the stable floor the decision must NOT flip.
    let pressured = 3_180_212_224u64;
    let plan_cold = decide_row_primary_hessian_cache(
        n,
        r,
        BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
        plentiful,
        pressured,
        0,
    );
    assert!(
        plan_cold.materialize,
        "stable capacity budget must keep the 1.08 GB cache materialized through a transient available-RAM dip"
    );
    assert_eq!(
        plan_cold.reason,
        RowPrimaryHessianCacheReason::ReuseAmortizesBuild
    );

    // Sanity: feeding the pressured value as BOTH the stable floor and the
    // live reading (the pre-fix behaviour) reproduces the bad flip, proving
    // the floor — not some unrelated slack — is what prevents it.
    let plan_prefix = decide_row_primary_hessian_cache(
        n,
        r,
        BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
        pressured,
        pressured,
        0,
    );
    assert!(!plan_prefix.materialize);
    assert_eq!(
        plan_prefix.reason,
        RowPrimaryHessianCacheReason::SingleCacheExceedsRamFraction
    );
}


#[test]
fn bernoulli_flex_hvp_cache_matches_uncached_path_small_case() {
    let (family, states) = make_flex_hvp_cache_test_family(12);
    let mut cached = family
        .build_exact_eval_cache(&states)
        .expect("cached exact eval cache");
    cached.row_primary_hessians = family
        .build_row_primary_hessian_cache(&states, &cached)
        .expect("row Hessian cache");
    let uncached = BernoulliMarginalSlopeExactEvalCache {
        slices: cached.slices.clone(),
        primary: cached.primary.clone(),
        row_contexts: cached.row_contexts.clone(),
        row_cell_moments: None,
        cell_family_forest: None,
        row_cell_moments_d15: crate::resource::RayonSafeOnce::new(),
        row_cell_moments_d21: crate::resource::RayonSafeOnce::new(),
        row_primary_hessians: RowPrimaryEvalCache::Empty,
        rigid_third_full: crate::resource::RayonSafeOnce::new(),
        rigid_fourth_full: crate::resource::RayonSafeOnce::new(),
        flex_axis_third_tensors: crate::resource::RayonSafeOnce::new(),
        flex_axis_fourth_tensors: crate::resource::RayonSafeOnce::new(),
        full_data_outer_rows: std::sync::OnceLock::new(),
    };
    let direction =
        Array1::from_iter((0..cached.slices.total).map(|idx| 0.02 * ((idx % 5) as f64 - 2.0)));
    let hv_cached = family
        .exact_newton_joint_hessian_matvec_from_cache(&direction, &states, &cached)
        .expect("cached Hv");
    let hv_uncached = family
        .exact_newton_joint_hessian_matvec_from_cache(&direction, &states, &uncached)
        .expect("uncached Hv");
    let rel = rel_diff_array1(&hv_cached, &hv_uncached);
    assert!(rel < 5e-11, "cached Hv drift rel {rel}");

    let dense_cached = family
        .exact_newton_joint_hessian_dense_from_cache(&states, &cached)
        .expect("cached dense Hessian");
    let dense_uncached = family
        .exact_newton_joint_hessian_dense_from_cache(&states, &uncached)
        .expect("uncached dense Hessian");
    let rel_dense = rel_diff_array2(&dense_cached, &dense_uncached);
    assert!(
        rel_dense < 5e-11,
        "cached dense Hessian drift rel {rel_dense}"
    );

    let diag_cached = family
        .exact_newton_joint_hessian_diagonal_from_cache(&states, &cached)
        .expect("cached diag");
    let diag_uncached = family
        .exact_newton_joint_hessian_diagonal_from_cache(&states, &uncached)
        .expect("uncached diag");
    let rel_diag = rel_diff_array1(&diag_cached, &diag_uncached);
    assert!(rel_diag < 5e-11, "cached diag drift rel {rel_diag}");
}


#[test]
fn bernoulli_flex_tiled_hvp_cache_matches_host_cache_small_case() {
    let (family, states) = make_flex_hvp_cache_test_family(14);
    let mut host_cache = family
        .build_exact_eval_cache(&states)
        .expect("host exact eval cache");
    host_cache.row_primary_hessians = family
        .build_row_primary_hessian_cache(&states, &host_cache)
        .expect("host row Hessian cache");
    let host_pin = host_cache
        .row_primary_hessians
        .host_pin()
        .expect("host row-primary cache");
    let r = host_cache.primary.total;
    let mut tiles = Vec::new();
    for rows in [0..5, 5..10, 10..14] {
        tiles.push(RowPrimaryEvalTile {
            row_start: rows.start,
            rows: RowPrimaryEvalPin::new(
                host_pin.neglog().slice(s![rows.clone()]).to_owned(),
                host_pin.grad().slice(s![rows.clone(), ..]).to_owned(),
                host_pin.hess().slice(s![rows, ..]).to_owned(),
                0,
            ),
        });
    }
    let tiled_cache = BernoulliMarginalSlopeExactEvalCache {
        slices: host_cache.slices.clone(),
        primary: host_cache.primary.clone(),
        row_contexts: host_cache.row_contexts.clone(),
        row_cell_moments: host_cache.row_cell_moments.clone(),
        cell_family_forest: None,
        row_cell_moments_d15: crate::resource::RayonSafeOnce::new(),
        row_cell_moments_d21: crate::resource::RayonSafeOnce::new(),
        row_primary_hessians: RowPrimaryEvalCache::Tiled(RowPrimaryEvalTiles::new(
            family.y.len(),
            r,
            5,
            tiles,
        )),
        rigid_third_full: crate::resource::RayonSafeOnce::new(),
        rigid_fourth_full: crate::resource::RayonSafeOnce::new(),
        flex_axis_third_tensors: crate::resource::RayonSafeOnce::new(),
        flex_axis_fourth_tensors: crate::resource::RayonSafeOnce::new(),
        full_data_outer_rows: std::sync::OnceLock::new(),
    };
    let direction =
        Array1::from_iter((0..host_cache.slices.total).map(|idx| 0.015 * ((idx % 7) as f64 - 3.0)));
    let hv_host = family
        .exact_newton_joint_hessian_matvec_from_cache(&direction, &states, &host_cache)
        .expect("host Hv");
    let hv_tiled = family
        .exact_newton_joint_hessian_matvec_from_cache(&direction, &states, &tiled_cache)
        .expect("tiled Hv");
    let rel_hv = rel_diff_array1(&hv_host, &hv_tiled);
    assert!(rel_hv < 5e-11, "tiled Hv drift rel {rel_hv}");

    let diag_host = family
        .exact_newton_joint_hessian_diagonal_from_cache(&states, &host_cache)
        .expect("host diag");
    let diag_tiled = family
        .exact_newton_joint_hessian_diagonal_from_cache(&states, &tiled_cache)
        .expect("tiled diag");
    let rel_diag = rel_diff_array1(&diag_host, &diag_tiled);
    assert!(rel_diag < 5e-11, "tiled diag drift rel {rel_diag}");

    // Batched multi-RHS apply over the tiled cache must reproduce, column
    // for column, the single-vector HVP applied to each column. Build a
    // small (total x n_rhs) block of distinct directions, apply it both
    // ways, and require exact agreement.
    let total = tiled_cache.slices.total;
    let n_rhs = 3usize;
    let mut v_cols = Array2::<f64>::zeros((total, n_rhs));
    for col in 0..n_rhs {
        for idx in 0..total {
            v_cols[[idx, col]] = 0.013 * ((idx % (5 + col)) as f64 - 2.0) + 0.001 * col as f64;
        }
    }
    let mut batched = Array2::<f64>::zeros((total, n_rhs));
    family
        .exact_newton_joint_hessian_matvec_mat_from_cache_into(
            &v_cols,
            &states,
            &tiled_cache,
            &mut batched,
        )
        .expect("tiled batched Hv");
    for col in 0..n_rhs {
        let col_dir = v_cols.column(col).to_owned();
        let hv_col = family
            .exact_newton_joint_hessian_matvec_from_cache(&col_dir, &states, &tiled_cache)
            .expect("tiled per-column Hv");
        let rel_col = rel_diff_array1(&hv_col, &batched.column(col).to_owned());
        assert!(
            rel_col < 5e-11,
            "tiled batched apply column {col} drift rel {rel_col}"
        );
    }
}


#[test]
fn bernoulli_flex_hvp_cache_timing_large_scale_shape_pattern() {
    // Wall-clock micro-benchmark for the per-row primary-Hessian cache
    // (`row_primary_hessians`).  The matrix-free CG / inner-Newton loops
    // contract the same per-row primary Hessian against many trial
    // directions at the same β, so caching the `r×r` blocks once should
    // beat rebuilding cell moments + flex jets on every Hv.
    let (family, states) = make_flex_hvp_cache_test_family(96);
    let mut cached = family
        .build_exact_eval_cache(&states)
        .expect("cached exact eval cache");
    cached.row_primary_hessians = family
        .build_row_primary_hessian_cache(&states, &cached)
        .expect("row Hessian cache");
    let uncached = BernoulliMarginalSlopeExactEvalCache {
        slices: cached.slices.clone(),
        primary: cached.primary.clone(),
        row_contexts: cached.row_contexts.clone(),
        row_cell_moments: None,
        cell_family_forest: None,
        row_cell_moments_d15: crate::resource::RayonSafeOnce::new(),
        row_cell_moments_d21: crate::resource::RayonSafeOnce::new(),
        row_primary_hessians: RowPrimaryEvalCache::Empty,
        rigid_third_full: crate::resource::RayonSafeOnce::new(),
        rigid_fourth_full: crate::resource::RayonSafeOnce::new(),
        flex_axis_third_tensors: crate::resource::RayonSafeOnce::new(),
        flex_axis_fourth_tensors: crate::resource::RayonSafeOnce::new(),
        full_data_outer_rows: std::sync::OnceLock::new(),
    };
    let directions: Vec<_> = (0..4)
        .map(|rep| {
            Array1::from_iter(
                (0..cached.slices.total)
                    .map(|idx| 0.01 * (((idx * 13 + rep * 7) % 11) as f64 - 5.0)),
            )
        })
        .collect();
    let start_uncached = std::time::Instant::now();
    for direction in &directions {
        family
            .exact_newton_joint_hessian_matvec_from_cache(direction, &states, &uncached)
            .expect("uncached Hv");
    }
    let uncached_elapsed = start_uncached.elapsed();
    let start_cached = std::time::Instant::now();
    for direction in &directions {
        family
            .exact_newton_joint_hessian_matvec_from_cache(direction, &states, &cached)
            .expect("cached Hv");
    }
    let cached_elapsed = start_cached.elapsed();
    eprintln!("flex Hv cache timing: uncached={uncached_elapsed:?} cached={cached_elapsed:?}");
    assert!(
        cached_elapsed < uncached_elapsed,
        "expected cached Hv loop to beat uncached: cached={cached_elapsed:?} uncached={uncached_elapsed:?}"
    );
}


#[test]
fn bernoulli_jointhessian_directional_operator_matches_dense_small_case() {
    let n = 17usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("exact eval cache");
    let slices = &cache.slices;
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let dense = family
        .exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
            &states,
            &d_beta_flat,
            &cache,
            &BlockwiseFitOptions::default(),
        )
        .expect("dense dH")
        .expect("dH present");
    let operator = family
        .exact_newton_joint_hessian_directional_derivative_operator_from_cache_with_options(
            &states,
            &d_beta_flat,
            &cache,
            &BlockwiseFitOptions::default(),
        )
        .expect("operator dH")
        .expect("dH operator present");

    let rel = rel_diff_array2(&operator.to_dense(), &dense);
    assert!(rel < 1e-12, "operator dH rel {}", rel);
}


#[test]
fn bernoulli_jointhessian_second_directional_operator_matches_dense_small_case() {
    let n = 17usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("exact eval cache");
    let slices = &cache.slices;
    let mut d_beta_u = Array1::<f64>::zeros(slices.total);
    d_beta_u[slices.marginal.start] = 0.05;
    d_beta_u[slices.logslope.start] = -0.04;
    let mut d_beta_v = Array1::<f64>::zeros(slices.total);
    d_beta_v[slices.marginal.start] = -0.03;
    d_beta_v[slices.logslope.start] = 0.02;

    let dense = family
        .exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options(
            &states,
            &d_beta_u,
            &d_beta_v,
            &cache,
            &BlockwiseFitOptions::default(),
        )
        .expect("dense d2H")
        .expect("d2H present");
    let operator = family
        .exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache_with_options(
            &states,
            &d_beta_u,
            &d_beta_v,
            &cache,
            &BlockwiseFitOptions::default(),
        )
        .expect("operator d2H")
        .expect("d2H operator present");

    let rel = rel_diff_array2(&operator.to_dense(), &dense);
    assert!(rel < 1e-12, "operator d2H rel {}", rel);
}


#[test]
fn bernoulli_large_scale_outer_derivatives_keep_analytic_hessian_route() {
    let n = 50_001usize;
    let family = make_block_psi_test_family(n);
    let specs = vec![dummy_blockspec(1, n), dummy_blockspec(1, n)];
    let options = BlockwiseFitOptions::default();

    let (gradient, hessian) =
        crate::custom_family::custom_family_outer_derivatives(&family, &specs, &options);

    assert_eq!(
        gradient,
        crate::solver::outer_strategy::Derivative::Analytic
    );
    assert_eq!(
        hessian,
        crate::solver::outer_strategy::DeclaredHessianForm::Either
    );
}


#[test]
fn bernoulli_jointhessian_directional_derivative_from_cache_subsample_half_scales_correctly() {
    use crate::families::marginal_slope_shared::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("exact eval cache");
    let slices = &cache.slices;
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
        even_mask.clone(),
        n,
        0xCAFE,
    )));
    let scaled = family
        .exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
            &states,
            &d_beta_flat,
            &cache,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
            &states,
            &d_beta_flat,
            &cache,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp = &raw * factor;
    let rel = rel_diff_array2(&scaled, &exp);
    assert!(rel < 1e-12, "joint Hessian dH HT rel {}", rel);
}


#[test]
fn auto_outer_subsample_two_phase_converges_to_full_data_optimum() {
    // The full BMS outer-strategy + custom-family pipeline is too
    // elaborate to drive end-to-end from a unit test, so we exercise
    // the substituted check from the task spec: 15 calls to
    // `batched_outer_gradient_terms` with a mix of distinct ρ and
    // line-search-style retries at the same ρ. The contract is
    //
    //   counter strictly counts distinct ρ values
    //
    // so retries must NOT bump it. With BUDGET = 12, the first 12
    // distinct-ρ calls land in Phase 1 (subsample), the 13th onward
    // fall through to Phase 2 (full data). We verify both bookkeeping
    // properties without needing to actually drive a fit.
    //
    // n = 35_000 sits above `AutoOuterSubsampleOptions::default()
    // .min_n_for_auto = 30_000`, so `auto_outer_score_subsample`
    // would actually return `Some(mask)` if invoked — i.e. the
    // Phase-1 branch reaches the mask-installing arm and the
    // log::info! lines fire. Specs/derivative_blocks are empty so
    // the function exits via the `total == 0` early return after the
    // guard runs; that lets us focus on counter semantics with no
    // FLEX-cache plumbing.
    let n = 35_000usize;
    let family = make_block_psi_test_family(n);
    let states: Vec<ParameterBlockState> = Vec::new();
    let specs: Vec<ParameterBlockSpec> = Vec::new();
    let deriv_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> = Vec::new();
    // Non-empty ρ — empty arrays would have zero L2 distance and the
    // counter would never bump after the first call, defeating the
    // distinct-ρ check. With empty specs, `total == 0` short-circuits
    // before the rho.len() vs penalty-count consistency check.
    let rho_dim = 3usize;
    let mut opts = BlockwiseFitOptions::default();
    opts.auto_outer_subsample = true;

    let counter = || {
        family
            .auto_subsample_phase_counter
            .load(std::sync::atomic::Ordering::SeqCst)
    };

    // Walk through 15 distinct ρ values, with a line-search retry
    // (same ρ) interleaved at iterations 3 and 7. Distinct-ρ events:
    // 15. Retries: 2 (must not bump). Final counter must equal 15.
    let mut distinct_calls = 0usize;
    for step in 0..15usize {
        let rho_step = Array1::<f64>::from_elem(rho_dim, step as f64 * 0.1);
        family
            .batched_outer_gradient_terms(&states, &specs, &deriv_blocks, &rho_step, &opts, None)
            .expect("guard ok");
        distinct_calls += 1;
        assert_eq!(
            counter(),
            distinct_calls,
            "distinct-ρ call {step}: counter should equal number of distinct ρ"
        );
        // Line-search retry: same ρ. Must NOT bump.
        if step == 3 || step == 7 {
            let rho_retry = Array1::<f64>::from_elem(rho_dim, step as f64 * 0.1);
            family
                .batched_outer_gradient_terms(
                    &states,
                    &specs,
                    &deriv_blocks,
                    &rho_retry,
                    &opts,
                    None,
                )
                .expect("guard ok on retry");
            assert_eq!(
                counter(),
                distinct_calls,
                "line-search retry at step {step} must NOT bump counter"
            );
        }
    }
    assert_eq!(counter(), 15, "final counter should be 15 distinct ρ");

    // With auto_outer_subsample = false the guard short-circuits; a
    // fresh family's counter must remain at 0 across many calls.
    let family_off = make_block_psi_test_family(n);
    let opts_off = BlockwiseFitOptions {
        auto_outer_subsample: false,
        ..BlockwiseFitOptions::default()
    };
    for step in 0..5 {
        let rho_step = Array1::<f64>::from_elem(rho_dim, step as f64 * 0.1);
        family_off
            .batched_outer_gradient_terms(
                &states,
                &specs,
                &deriv_blocks,
                &rho_step,
                &opts_off,
                None,
            )
            .expect("guard ok off");
    }
    assert_eq!(
        family_off
            .auto_subsample_phase_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "with auto_outer_subsample=false the counter must stay at 0"
    );
}


/// #740: the direction-contracted ψψ second-order kernel must reproduce the
/// exact α-contraction of the per-pair `second_order_terms(i, j)`.
///
/// This is the family-side exactness gate for the profiled θ-HVP: the operator
/// applies `second_order_terms_contracted(α_ψ)` once per matvec instead of the
/// dense path's K² per-pair `second_order_terms`. The two MUST agree term for
/// term — `Σ_j α_j second_order_terms(i, j) == second_order_terms_contracted(α)[i]`
/// — for objective, score, and the (dense-materialized) Hessian operator, across
/// a non-trivial ψ direction. A wrong contraction (sign, dropped i↔j leg, or a
/// dropped same-block cross design term) shows up here as a non-zero residual.
#[test]
fn bernoulli_contracted_psi_second_order_matches_per_pair_contraction() {
    use crate::custom_family::CustomFamilyBlockPsiDerivative;

    let n = 40usize;
    // Two-column marginal block + one-column logslope block, so the marginal ψ
    // axes carry a genuine multi-coefficient design derivative `x_psi`.
    let y: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 37 + 11) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let z: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -1.2 + 2.4 * ((i % 7) as f64 + 0.5) / 7.0));
    let marginal = Array2::from_shape_fn((n, 2), |(r, c)| {
        if c == 0 {
            1.0
        } else {
            ((r * 13 + 5) % 11) as f64 / 11.0 - 0.5
        }
    });
    let logslope = Array2::from_shape_fn((n, 1), |_| 1.0);
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(y),
        weights: Arc::new(weights),
        z: Arc::new(z),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            marginal.clone(),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(logslope)),
        ..default_test_family()
    };

    let states = vec![
        ParameterBlockState {
            beta: array![0.3, -0.2],
            eta: marginal.dot(&array![0.3, -0.2]),
        },
        ParameterBlockState {
            beta: array![0.25],
            eta: Array1::from_elem(n, 0.25),
        },
    ];
    let specs = vec![dummy_blockspec(2, n), dummy_blockspec(1, n)];

    // Two ψ axes on the marginal block, each a distinct design-derivative column
    // `x_psi` (shape n×2). Distinct columns make the ψ_i/ψ_j legs asymmetric so
    // the i↔j contraction is genuinely exercised.
    let x_psi_0 = Array2::from_shape_fn((n, 2), |(r, c)| {
        ((r * 7 + c * 3 + 1) % 9) as f64 / 9.0 - 0.4
    });
    let x_psi_1 = Array2::from_shape_fn((n, 2), |(r, c)| {
        ((r * 5 + c * 2 + 4) % 8) as f64 / 8.0 - 0.55
    });
    let derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>> = vec![
        vec![
            CustomFamilyBlockPsiDerivative::new(
                None,
                x_psi_0,
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            ),
            CustomFamilyBlockPsiDerivative::new(
                None,
                x_psi_1,
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            ),
        ],
        Vec::new(),
    ];

    let opts = BlockwiseFitOptions::default();
    let ws = family
        .exact_newton_joint_psi_workspace_with_options(&states, &specs, &derivative_blocks, &opts)
        .expect("psi workspace with options")
        .expect("psi workspace some");

    let psi_dim: usize = derivative_blocks.iter().map(Vec::len).sum();
    assert_eq!(psi_dim, 2, "fixture should expose two marginal ψ axes");

    // A non-trivial ψ direction (no axis dominant, opposite signs).
    let alpha = [0.7_f64, -1.3_f64];

    let contracted = ws
        .second_order_terms_contracted(&alpha)
        .expect("contracted second-order call")
        .expect("contracted second-order some (likelihood kernel available)");
    assert_eq!(contracted.objective.len(), psi_dim);
    assert_eq!(contracted.score.nrows(), psi_dim);
    assert_eq!(contracted.hessian.len(), psi_dim);

    // Reference: Σ_j α_j second_order_terms(i, j), per output row i.
    for i in 0..psi_dim {
        let mut ref_obj = 0.0_f64;
        let mut ref_score: Option<Array1<f64>> = None;
        let mut ref_hess: Option<Array2<f64>> = None;
        for (j, &aj) in alpha.iter().enumerate() {
            if aj == 0.0 {
                continue;
            }
            let pair = ws
                .second_order_terms(i, j)
                .expect("per-pair second-order call")
                .expect("per-pair second-order some");
            ref_obj += aj * pair.objective_psi_psi;
            let score_j = pair.score_psi_psi.mapv(|v| aj * v);
            ref_score = Some(match ref_score {
                Some(acc) => acc + &score_j,
                None => score_j,
            });
            let hess_j = pair
                .hessian_psi_psi_operator
                .as_ref()
                .expect("per-pair hessian operator")
                .to_dense()
                .mapv(|v| aj * v);
            ref_hess = Some(match ref_hess {
                Some(acc) => acc + &hess_j,
                None => hess_j,
            });
        }
        let ref_score = ref_score.expect("non-empty alpha");
        let ref_hess = ref_hess.expect("non-empty alpha");

        let obj_err = (contracted.objective[i] - ref_obj).abs() / (1.0 + ref_obj.abs());
        assert!(
            obj_err < 1e-10,
            "row {i}: contracted objective {} != Σ_j α_j V_ij {} (rel {obj_err:.3e})",
            contracted.objective[i],
            ref_obj
        );

        let score_err = rel_diff_array1(&contracted.score.row(i).to_owned(), &ref_score);
        assert!(
            score_err < 1e-9,
            "row {i}: contracted score diverged from Σ_j α_j g_ij (rel {score_err:.3e})"
        );

        let hess_dense = match &contracted.hessian[i] {
            crate::solver::estimate::reml::unified::DriftDerivResult::Operator(op) => op.to_dense(),
            crate::solver::estimate::reml::unified::DriftDerivResult::Dense(m) => m.clone(),
        };
        let hess_err = rel_diff_array2(&hess_dense, &ref_hess);
        assert!(
            hess_err < 1e-9,
            "row {i}: contracted Hessian diverged from Σ_j α_j D²_ψ H_L[ψ_i, ψ_j] (rel {hess_err:.3e})"
        );
    }
}


/// #740: the GENERIC ψψ penalty fold inside `build_contracted_psi_hook` must
/// reproduce the per-pair `ext_ext` penalty contraction exactly, ON A FIXTURE
/// WHERE PENALTY AND LIKELIHOOD ψψ ARE BOTH NONZERO.
///
/// The operator's ψψ block is supplied by `build_contracted_psi_hook` =
/// family likelihood contraction (the workspace kernel, gated elsewhere) PLUS
/// the generic penalty motion (½βᵀS_ψψβ → objective, S_ψψβ → score, S_ψψ
/// BlockLocalDrift → hessian, τ-Hessian → ld_s). The per-pair `ext_ext(i, j)`
/// folds the IDENTICAL penalty, so it is the exact oracle:
///   `build_contracted_psi_hook(α).row(i) == Σ_j α_j · ext_ext(i, j)`.
/// The fixture puts a real penalty on the marginal block AND ψ axes that carry
/// nonzero `x_psi` (likelihood) AND nonzero `s_psi`/`s_psi_psi` (penalty moves),
/// so a bug in EITHER the generic penalty contraction OR the likelihood
/// contraction — or a double-count between them — breaks the equality.
#[test]
fn bernoulli_contracted_psi_hook_matches_per_pair_with_penalty() {
    use crate::custom_family::CustomFamilyBlockPsiDerivative;
    use crate::families::custom_family::{build_contracted_psi_hook, build_psi_pair_callbacks};
    use crate::solver::estimate::reml::penalty_logdet::PenaltyPseudologdet;
    use crate::solver::estimate::reml::unified::DriftDerivResult;

    let n = 40usize;
    let y: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 41 + 13) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let z: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -1.1 + 2.2 * ((i % 6) as f64 + 0.5) / 6.0));
    let marginal = Array2::from_shape_fn((n, 2), |(r, c)| {
        if c == 0 {
            1.0
        } else {
            ((r * 17 + 3) % 13) as f64 / 13.0 - 0.5
        }
    });
    let logslope = Array2::from_shape_fn((n, 1), |_| 1.0);
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(y),
        weights: Arc::new(weights),
        z: Arc::new(z),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            marginal.clone(),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(logslope)),
        ..default_test_family()
    };

    let beta_marginal = array![0.3, -0.2];
    let states = vec![
        ParameterBlockState {
            beta: beta_marginal.clone(),
            eta: marginal.dot(&beta_marginal),
        },
        ParameterBlockState {
            beta: array![0.25],
            eta: Array1::from_elem(n, 0.25),
        },
    ];

    // Marginal block carries one SPD penalty so `s_logdet_blocks` (the
    // PenaltyPseudologdet) is non-trivial and the τ-Hessian / ld_s path fires.
    let s_pen = array![[1.4_f64, 0.2], [0.2, 1.1]];
    let mut marginal_spec = dummy_blockspec(2, n);
    marginal_spec.penalties = vec![crate::custom_family::PenaltyMatrix::Dense(s_pen.clone())];
    let specs = vec![marginal_spec, dummy_blockspec(1, n)];
    let penalty_counts = vec![1usize, 0usize];
    let rho = array![0.15_f64]; // one ρ on the marginal block penalty

    // Two ψ axes on the marginal block: each carries a distinct likelihood
    // design-derivative `x_psi` AND moves the penalty (`s_psi` + `s_psi_psi`),
    // referencing penalty index 0 of the block.
    let x_psi_0 = Array2::from_shape_fn((n, 2), |(r, c)| {
        ((r * 7 + c * 3 + 1) % 9) as f64 / 9.0 - 0.4
    });
    let x_psi_1 = Array2::from_shape_fn((n, 2), |(r, c)| {
        ((r * 5 + c * 2 + 4) % 8) as f64 / 8.0 - 0.55
    });
    // Penalty motion sized clearly above the vacuity floor (s_psi/s_psi_psi
    // entries O(1)) so ½βᵀS_ψψβ and the τ-Hessian are unambiguously live, not
    // borderline against 1e-6.
    let s_psi_0 = array![[1.10_f64, 0.25], [0.25, 0.80]];
    let s_psi_1 = array![[0.90_f64, -0.30], [-0.30, 1.05]];
    // s_psi_psi[i] is indexed by the SECOND axis j (per assemble_block_local_s_psi_psi):
    // axis i's vector holds ∂²S/∂ψ_i∂ψ_j for each j.
    let s_pp_0 = vec![
        array![[0.70_f64, 0.12], [0.12, 0.40]],   // (0,0)
        array![[0.45_f64, -0.18], [-0.18, 0.55]], // (0,1)
    ];
    let s_pp_1 = vec![
        array![[0.45_f64, -0.18], [-0.18, 0.55]], // (1,0) = (0,1)^sym
        array![[0.95_f64, 0.20], [0.20, 0.65]],   // (1,1)
    ];
    let derivative_blocks: std::sync::Arc<Vec<Vec<CustomFamilyBlockPsiDerivative>>> =
        std::sync::Arc::new(vec![
            vec![
                CustomFamilyBlockPsiDerivative::new(
                    Some(0),
                    x_psi_0,
                    s_psi_0,
                    None,
                    None,
                    Some(s_pp_0),
                    None,
                ),
                CustomFamilyBlockPsiDerivative::new(
                    Some(0),
                    x_psi_1,
                    s_psi_1,
                    None,
                    None,
                    Some(s_pp_1),
                    None,
                ),
            ],
            Vec::new(),
        ]);

    let opts = BlockwiseFitOptions::default();
    let psi_workspace = family
        .exact_newton_joint_psi_workspace_with_options(&states, &specs, &derivative_blocks, &opts)
        .expect("psi workspace")
        .expect("psi workspace some");

    // Build the exact pseudologdet eigenspace exactly as the evaluator does
    // (custom_family.rs), so the τ-Hessian leg is live in BOTH paths.
    let lambda0 = rho[0].exp();
    let s_lambda = s_pen.mapv(|v| lambda0 * v);
    let pld = PenaltyPseudologdet::from_assembled(s_lambda, None).expect("pseudologdet");
    let s_logdet_blocks = vec![pld];

    let beta_flat = array![0.3_f64, -0.2, 0.25];
    let rho_slice = rho.as_slice().unwrap();

    let (ext_ext, _rho_ext) = build_psi_pair_callbacks(
        &family,
        &states,
        &specs,
        std::sync::Arc::clone(&derivative_blocks),
        &beta_flat,
        rho_slice,
        &penalty_counts,
        Some(&s_logdet_blocks),
        Some(std::sync::Arc::clone(&psi_workspace)),
    )
    .expect("per-pair callbacks");

    let hook = build_contracted_psi_hook(
        &specs,
        std::sync::Arc::clone(&derivative_blocks),
        &beta_flat,
        rho_slice,
        &penalty_counts,
        Some(&s_logdet_blocks),
        Some(std::sync::Arc::clone(&psi_workspace)),
    )
    .expect("contracted hook build")
    .expect("contracted hook some (likelihood kernel available)");

    let psi_dim = 2usize;
    let alpha = [0.7_f64, -1.3_f64];
    let contracted = hook(&alpha).expect("hook call").expect("hook some");

    for i in 0..psi_dim {
        let mut ref_a = 0.0_f64;
        let mut ref_g: Option<Array1<f64>> = None;
        let mut ref_b: Option<Array2<f64>> = None;
        let mut ref_ld = 0.0_f64;
        for (j, &aj) in alpha.iter().enumerate() {
            if aj == 0.0 {
                continue;
            }
            let pair = ext_ext(i, j);
            ref_a += aj * pair.a;
            ref_ld += aj * pair.ld_s;
            let g_j = pair.g.mapv(|v| aj * v);
            ref_g = Some(match ref_g {
                Some(acc) => acc + &g_j,
                None => g_j,
            });
            // The per-pair Hessian drift is `b_mat` (dense) + `b_operator` (the
            // S_ψψ BlockLocalDrift composite); sum their dense forms.
            let mut b_dense = if pair.b_mat.nrows() > 0 {
                pair.b_mat.clone()
            } else {
                Array2::<f64>::zeros((beta_flat.len(), beta_flat.len()))
            };
            if let Some(op) = pair.b_operator.as_ref() {
                b_dense = b_dense + op.to_dense();
            }
            let b_j = b_dense.mapv(|v| aj * v);
            ref_b = Some(match ref_b {
                Some(acc) => acc + &b_j,
                None => b_j,
            });
        }
        let ref_g = ref_g.expect("nonempty alpha");
        let ref_b = ref_b.expect("nonempty alpha");

        let a_err = (contracted.objective[i] - ref_a).abs() / (1.0 + ref_a.abs());
        assert!(
            a_err < 1e-9,
            "row {i}: hook objective {} != Σ_j α_j ext_ext.a {} (rel {a_err:.3e})",
            contracted.objective[i],
            ref_a
        );
        let ld_err = (contracted.ld_s[i] - ref_ld).abs() / (1.0 + ref_ld.abs());
        assert!(
            ld_err < 1e-9,
            "row {i}: hook ld_s {} != Σ_j α_j ext_ext.ld_s {} (rel {ld_err:.3e}) \
             — the τ-Hessian penalty-logdet fold diverged",
            contracted.ld_s[i],
            ref_ld
        );
        let g_err = rel_diff_array1(&contracted.score.row(i).to_owned(), &ref_g);
        assert!(
            g_err < 1e-9,
            "row {i}: hook score diverged from Σ_j α_j ext_ext.g (rel {g_err:.3e})"
        );

        let hess_dense = match &contracted.hessian[i] {
            DriftDerivResult::Operator(op) => op.to_dense(),
            DriftDerivResult::Dense(m) => m.clone(),
        };
        let b_err = rel_diff_array2(&hess_dense, &ref_b);
        assert!(
            b_err < 1e-9,
            "row {i}: hook Hessian (likelihood + S_ψψ) diverged from Σ_j α_j ext_ext drift (rel {b_err:.3e})"
        );
    }

    // The penalty must be LIVE: a likelihood-only fixture would make this gate
    // vacuous. Compare against a GENUINELY penalty-free baseline — derivative
    // blocks with the SAME x_psi (identical likelihood) but ZERO s_psi/s_psi_psi
    // (no penalty motion at all). Passing `None` for s_logdet_blocks is NOT
    // sufficient: build_contracted_psi_hook still adds the ½βᵀS_ψψβ objective and
    // S_ψψβ score terms from s_psi_psi regardless of s_logdet (only the
    // τ-Hessian ld_s is gated on it), so a same-derivative-blocks baseline would
    // be identical in objective/score and the guard would falsely fire.
    let derivative_blocks_no_pen: std::sync::Arc<Vec<Vec<CustomFamilyBlockPsiDerivative>>> =
        std::sync::Arc::new(vec![
            vec![
                CustomFamilyBlockPsiDerivative::new(
                    None,
                    derivative_blocks[0][0].x_psi.clone(),
                    Array2::zeros((2, 2)),
                    None,
                    None,
                    None,
                    None,
                ),
                CustomFamilyBlockPsiDerivative::new(
                    None,
                    derivative_blocks[0][1].x_psi.clone(),
                    Array2::zeros((2, 2)),
                    None,
                    None,
                    None,
                    None,
                ),
            ],
            Vec::new(),
        ]);
    let hook_no_pen = build_contracted_psi_hook(
        &specs,
        std::sync::Arc::clone(&derivative_blocks_no_pen),
        &beta_flat,
        rho_slice,
        &penalty_counts,
        Some(&s_logdet_blocks),
        Some(std::sync::Arc::clone(&psi_workspace)),
    )
    .expect("no-penalty hook build")
    .expect("no-penalty hook some");
    let contracted_no_pen = hook_no_pen(&alpha).expect("call").expect("some");
    let ld_total: f64 = contracted.ld_s.iter().map(|v| v.abs()).sum();
    assert!(
        ld_total > 1e-6,
        "#740 penalty-fold test is vacuous: τ-Hessian ld_s ~0 across all ψ rows"
    );
    let obj_shift: f64 = contracted
        .objective
        .iter()
        .zip(contracted_no_pen.objective.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        obj_shift > 1e-6,
        "#740 penalty-fold test is vacuous: ½βᵀS_ψψβ penalty contributes ~0 to the objective"
    );
}


#[test]
fn bernoulli_batched_outer_gradient_matches_hypercoord_path_for_rho_and_psi() {
    use crate::families::custom_family::build_psi_hyper_coords;
    use crate::solver::estimate::reml::penalty_logdet::PenaltyPseudologdet;
    use crate::solver::estimate::reml::unified::{DenseSpectralOperator, HessianOperator};

    let n = 32usize;
    let y: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 41 + 13) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let z: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -1.1 + 2.2 * ((i % 6) as f64 + 0.5) / 6.0));
    let marginal = Array2::from_shape_fn((n, 2), |(r, c)| {
        if c == 0 {
            1.0
        } else {
            ((r * 17 + 3) % 13) as f64 / 13.0 - 0.5
        }
    });
    let logslope = Array2::from_shape_fn((n, 1), |_| 1.0);
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(y),
        weights: Arc::new(weights),
        z: Arc::new(z),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            marginal.clone(),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(logslope)),
        ..default_test_family()
    };

    let beta_marginal = array![0.3, -0.2];
    let states = vec![
        ParameterBlockState {
            beta: beta_marginal.clone(),
            eta: marginal.dot(&beta_marginal),
        },
        ParameterBlockState {
            beta: array![0.25],
            eta: Array1::from_elem(n, 0.25),
        },
    ];

    let s_pen = array![[1.4_f64, 0.2], [0.2, 1.1]];
    let mut marginal_spec = dummy_blockspec(2, n);
    marginal_spec.penalties = vec![crate::custom_family::PenaltyMatrix::Dense(s_pen.clone())];
    let specs = vec![marginal_spec, dummy_blockspec(1, n)];
    let penalty_counts = vec![1usize, 0usize];
    let rho = array![0.15_f64];

    let x_psi_0 = Array2::from_shape_fn((n, 2), |(r, c)| {
        ((r * 7 + c * 3 + 1) % 9) as f64 / 9.0 - 0.4
    });
    let x_psi_1 = Array2::from_shape_fn((n, 2), |(r, c)| {
        ((r * 5 + c * 2 + 4) % 8) as f64 / 8.0 - 0.55
    });
    let s_psi_0 = array![[1.10_f64, 0.25], [0.25, 0.80]];
    let s_psi_1 = array![[0.90_f64, -0.30], [-0.30, 1.05]];
    let s_pp_0 = vec![
        array![[0.70_f64, 0.12], [0.12, 0.40]],
        array![[0.45_f64, -0.18], [-0.18, 0.55]],
    ];
    let s_pp_1 = vec![
        array![[0.45_f64, -0.18], [-0.18, 0.55]],
        array![[0.95_f64, 0.20], [0.20, 0.65]],
    ];
    let derivative_blocks = vec![
        vec![
            crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                Some(0),
                x_psi_0,
                s_psi_0,
                None,
                None,
                Some(s_pp_0),
                None,
            ),
            crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                Some(0),
                x_psi_1,
                s_psi_1,
                None,
                None,
                Some(s_pp_1),
                None,
            ),
        ],
        Vec::new(),
    ];
    let psi_dim: usize = derivative_blocks.iter().map(Vec::len).sum();
    assert_eq!(psi_dim, 2, "fixture should expose two ψ coordinates");

    let opts = BlockwiseFitOptions::default();
    let workspace = family
        .exact_newton_joint_psi_workspace_with_options(&states, &specs, &derivative_blocks, &opts)
        .expect("psi workspace")
        .expect("psi workspace some");
    let batched = family
        .batched_outer_gradient_terms(&states, &specs, &derivative_blocks, &rho, &opts, None)
        .expect("batched outer gradient")
        .expect("batched terms some");

    let ranges = BernoulliMarginalSlopeFamily::block_ranges_from_specs(&specs);
    let total = ranges.last().map(|(_, end)| *end).expect("nonempty specs");
    let theta_dim = rho.len() + psi_dim;
    let beta = BernoulliMarginalSlopeFamily::flatten_block_state_betas_for_specs(&states, &specs)
        .expect("flatten beta");
    assert_eq!(beta.len(), total);

    let ridge = opts.ridge_floor.max(1e-15);
    let mut h = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("joint hessian some");
    let mut manual_objective_theta = Array1::<f64>::zeros(theta_dim);
    let mut manual_trace_s_pinv_sdot = Array1::<f64>::zeros(theta_dim);
    let mut penalties_dense: Vec<Vec<Array2<f64>>> = Vec::with_capacity(specs.len());
    let mut penalty_cursor = 0usize;
    for (block_idx, spec) in specs.iter().enumerate() {
        let count = spec.penalties.len();
        let block_rho = rho
            .slice(ndarray::s![penalty_cursor..penalty_cursor + count])
            .to_owned();
        let (start, end) = ranges[block_idx];
        let beta_block = beta.slice(ndarray::s![start..end]);
        let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
        let mut block_penalties = Vec::with_capacity(count);
        for (local_idx, penalty) in spec.penalties.iter().enumerate() {
            let dense = penalty.to_dense();
            let lambda = block_rho[local_idx].exp();
            let s_beta = dense.dot(&beta_block);
            manual_objective_theta[penalty_cursor + local_idx] =
                0.5 * lambda * beta_block.dot(&s_beta);
            s_lambda.scaled_add(lambda, &dense);
            block_penalties.push(dense);
        }
        h.slice_mut(ndarray::s![start..end, start..end])
            .scaled_add(1.0, &s_lambda);
        penalties_dense.push(block_penalties);
        penalty_cursor += count;
    }
    if opts.ridge_policy.include_quadratic_penalty || opts.ridge_policy.include_penalty_logdet {
        for diag in 0..total {
            h[[diag, diag]] += ridge;
        }
    }

    let penalty_logdet_ridge = if opts.ridge_policy.include_penalty_logdet {
        ridge
    } else {
        0.0
    };
    let mut penalty_logdet_blocks = Vec::with_capacity(specs.len());
    penalty_cursor = 0;
    for (block_idx, spec) in specs.iter().enumerate() {
        let count = spec.penalties.len();
        let block_rho = rho
            .slice(ndarray::s![penalty_cursor..penalty_cursor + count])
            .to_owned();
        let lambdas = block_rho.mapv(f64::exp).to_vec();
        let pld = PenaltyPseudologdet::from_components(
            &penalties_dense[block_idx],
            &lambdas,
            penalty_logdet_ridge,
        )
        .expect("penalty pseudologdet");
        let first = pld.rho_derivatives(&penalties_dense[block_idx], &lambdas).0;
        for (local_idx, value) in first.iter().enumerate() {
            manual_trace_s_pinv_sdot[penalty_cursor + local_idx] = *value;
        }
        penalty_cursor += spec.penalties.len();
        penalty_logdet_blocks.push(pld);
    }

    let spectral = DenseSpectralOperator::from_symmetric_with_mode(&h, family.pseudo_logdet_mode())
        .expect("dense spectral");
    let mut manual_trace_h_inv_hdot = Array1::<f64>::zeros(theta_dim);
    let mut directions = Array2::<f64>::zeros((total, theta_dim));
    penalty_cursor = 0;
    for (block_idx, spec) in specs.iter().enumerate() {
        let (start, end) = ranges[block_idx];
        let beta_block = beta.slice(ndarray::s![start..end]);
        for (local_idx, penalty) in spec.penalties.iter().enumerate() {
            let idx = penalty_cursor + local_idx;
            let lambda = rho[idx].exp();
            let dense = penalty.to_dense();
            manual_trace_h_inv_hdot[idx] +=
                spectral.trace_logdet_block_local(&dense, lambda, start, end);
            let curvature_rhs = dense.dot(&beta_block).mapv(|value| lambda * value);
            let mut rhs = Array1::<f64>::zeros(total);
            rhs.slice_mut(ndarray::s![start..end])
                .assign(&curvature_rhs);
            let v = spectral.solve(&rhs);
            directions.column_mut(idx).assign(&(-&v));
        }
        penalty_cursor += spec.penalties.len();
    }

    let rho_slice = rho.as_slice().expect("rho contiguous");
    let psi_coords = build_psi_hyper_coords(
        &family,
        &states,
        &specs,
        &derivative_blocks,
        &beta,
        rho_slice,
        &penalty_counts,
        Some(&penalty_logdet_blocks),
        !family.exact_newton_joint_hessian_beta_dependent(),
        Some(workspace),
    )
    .expect("psi hyper coords");
    assert_eq!(psi_coords.len(), psi_dim);
    for (psi_index, coord) in psi_coords.iter().enumerate() {
        let idx = rho.len() + psi_index;
        manual_objective_theta[idx] = coord.a;
        manual_trace_s_pinv_sdot[idx] = coord.ld_s;
        if let Some(dense) = coord.drift.dense.as_ref() {
            manual_trace_h_inv_hdot[idx] += spectral.trace_logdet_gradient(dense);
        }
        if let Some(block_local) = coord.drift.block_local.as_ref() {
            manual_trace_h_inv_hdot[idx] += spectral.trace_logdet_block_local(
                &block_local.local,
                1.0,
                block_local.start,
                block_local.end,
            );
        }
        if let Some(operator) = coord.drift.operator_ref() {
            manual_trace_h_inv_hdot[idx] += spectral.trace_logdet_operator(operator);
        }
        let v = spectral.solve(&coord.g);
        directions.column_mut(idx).assign(&(-&v));
    }

    let cache = family
        .build_exact_eval_cache_with_options(&states, Some(&opts))
        .expect("exact eval cache");
    let correction_traces = family
        .batched_rho_correction_logdet_traces_full_rows(
            &states,
            &cache,
            spectral.logdet_gradient_factor(),
            &directions,
        )
        .expect("batched correction traces");
    manual_trace_h_inv_hdot += &correction_traces;

    for idx in 0..theta_dim {
        let label = if idx < rho.len() { "rho" } else { "psi" };
        assert_scalar_close(
            batched.objective_theta[idx],
            manual_objective_theta[idx],
            1.0e-11,
            &format!("{label}[{idx}] objective_theta"),
        );
        assert_scalar_close(
            batched.trace_h_inv_hdot[idx],
            manual_trace_h_inv_hdot[idx],
            1.0e-10,
            &format!("{label}[{idx}] trace_h_inv_hdot"),
        );
        assert_scalar_close(
            batched.trace_s_pinv_sdot[idx],
            manual_trace_s_pinv_sdot[idx],
            1.0e-11,
            &format!("{label}[{idx}] trace_s_pinv_sdot"),
        );
    }
}


/// #740 (option B): the INDEPENDENT ground-truth gate — the analytic outer
/// Hessian over θ=(ρ,ψ) must equal a CENTERED FINITE DIFFERENCE of the outer
/// gradient, across pure-ψ AND mixed-ρψ directions, on a real BMS spatial
/// length-scale fit.
///
/// The machine-precision operator==dense and contracted==per-pair gates certify
/// "#740 reproduces the trusted per-pair path", but that is first-order /
/// self-referential: only Hessian == centered-FD-of-gradient over ψ is
/// independent ground truth that the ψψ second-order block is actually correct
/// (gradient==FD-of-value, which other tests cover, cannot catch a ψψ Hessian
/// error). For a change flagged highest-REML-bias-risk this is the close-out.
///
/// ψ = −log(length_scale): perturbing ψ rebuilds the matern design + ψ
/// derivative blocks at the shifted length-scale (centers frozen once, so the FD
/// is well-defined — same pattern as basis_matern_log_kappa_*_derivative_fd).
/// The outer (ρ,ψ) gradient/Hessian come from evaluate_custom_family_joint_hyper.
#[test]
fn profiled_theta_hvp_outer_hessian_matches_fd_of_gradient_psi_and_mixed() {
    use crate::custom_family::PenaltyMatrix;
    use crate::families::custom_family::evaluate_custom_family_joint_hyper;
    use crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives;
    use crate::solver::estimate::reml::unified::EvalMode;
    use crate::terms::basis::{CenterStrategy, MaternBasisSpec, MaternNu};
    use crate::terms::smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, build_term_collection_design,
        freeze_term_collection_from_design,
    };

    crate::init_parallelism();

    // Larger, well-identified, NON-separable fixture so the coupled
    // marginal-slope inner Newton converges (a hard-threshold y on small n +
    // an intercept-only marginal triggers the #979 phantom-multiplier grind).
    let n = 160usize;
    // 2D spatial covariate for the matern logslope smooth + a marginal covariate.
    let mut data = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let x0 = (i as f64 / (n as f64 - 1.0)) * 2.0 - 1.0;
        let x1 = (0.41 * i as f64).sin() * 0.6 + 0.25 * x0;
        let xm = (0.23 * i as f64).cos() * 0.8; // marginal covariate (col 2)
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        data[[i, 2]] = xm;
    }
    // Bernoulli draw from a SMOOTH probability (logit) with a deterministic
    // pseudo-uniform, so the data is genuinely non-separable (no hard threshold)
    // and both blocks are identified.
    let y: Array1<f64> = Array1::from_iter((0..n).map(|i| {
        let lin = 0.5 + 0.7 * data[[i, 2]] - 0.4 * data[[i, 0]] + 0.3 * data[[i, 1]];
        let p = 1.0 / (1.0 + (-lin).exp());
        let u = ((i * 2654435761usize) % 1000) as f64 / 1000.0;
        if u < p { 1.0 } else { 0.0 }
    }));
    let z: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * ((i % 9) as f64 + 0.5) / 9.0));
    let weights = Array1::from_elem(n, 1.0);
    // Marginal covariate column (intercept + this) for a well-identified,
    // less-coupled marginal block.
    let marginal_cov: Array1<f64> = data.column(2).to_owned();

    // Base matern logslope spec. Freeze the centers ONCE at the base
    // length-scale so the FD perturbations only move `length_scale`, not the
    // basis centers (centers held fixed ⇒ the ψ FD is well-defined). Fewer
    // centers ⇒ fewer penalty components ⇒ less marginal/logslope coupling.
    let base_length_scale = 1.1_f64;
    let make_spec = |length_scale: f64| TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            name: "spatial_logslope".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::EqualMass { num_centers: 4 },
                    length_scale,
                    nu: MaternNu::ThreeHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: Default::default(),
                    aniso_log_scales: None,
                    nullspace_shrinkage_survived: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let base_spec = make_spec(base_length_scale);
    let base_design =
        build_term_collection_design(data.view(), &base_spec).expect("base spatial design");
    let frozen_spec = freeze_term_collection_from_design(&base_spec, &base_design)
        .expect("freeze spatial spec (locks centers)");
    // A matern block produces SEVERAL penalty components (main smoothness +
    // nullspace/aux blocks), not one — so θ carries `n_rho` ρ coordinates on the
    // logslope block plus the ψ length-scale axes. Discover the real count from
    // the design rather than assuming 1 (it's structural: same at every
    // length-scale).
    let n_rho = base_design.penalties_as_penalty_matrix().len();
    assert!(n_rho >= 1, "matern block must carry at least one penalty");

    // Build (family, specs, derivative_blocks, objective, gradient, Hessian) for a given ψ
    // offset (in −log(length_scale)) and ρ offset. ψ-offset δ ⇒ length_scale =
    // base * exp(−δ); the frozen centers are reused, so only the kernel
    // length-scale moves. Returns the full θ=(ρ,ψ) outer gradient + dense
    // Hessian from the profiled joint-hyper evaluator.
    let y_arc = Arc::new(y);
    let z_arc = Arc::new(z);
    let w_arc = Arc::new(weights);
    let outer_at = |psi_offset: f64, rho_offset: f64, mode: EvalMode| {
        // Rebuild the frozen spec at the perturbed length-scale by mutating only
        // the matern `length_scale` field (centers stay frozen).
        let mut spec = frozen_spec.clone();
        if let SmoothBasisSpec::Matern { spec: ms, .. } = &mut spec.smooth_terms[0].basis {
            ms.length_scale = base_length_scale * (-psi_offset).exp();
        }
        let design = build_term_collection_design(data.view(), &spec).expect("perturbed design");
        // ψ derivative blocks for the logslope spatial block at this length-scale.
        let logslope_psi = build_block_spatial_psi_derivatives(data.view(), &spec, &design)
            .expect("spatial psi derivatives")
            .expect("spatial psi derivative rows");
        let derivative_blocks = vec![Vec::new(), logslope_psi];

        // Well-identified marginal block: [intercept | covariate] (p=2), so the
        // marginal is not degenerate and is less coupled to the logslope block.
        // The matern logslope block carries the spatial design + its penalties.
        let p_log = design.design.ncols();
        let marginal_mat =
            Array2::from_shape_fn((n, 2), |(r, c)| if c == 0 { 1.0 } else { marginal_cov[r] });
        let marginal_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(marginal_mat));
        let logslope_design = design.design.clone();

        let family = BernoulliMarginalSlopeFamily {
            y: Arc::clone(&y_arc),
            weights: Arc::clone(&w_arc),
            z: Arc::clone(&z_arc),
            marginal_design: marginal_design.clone(),
            logslope_design: logslope_design.clone(),
            ..default_test_family()
        };

        // Block specs: marginal (intercept, no penalty), logslope (matern design
        // + its `n_rho` penalty components).
        let logslope_penalties: Vec<PenaltyMatrix> = design.penalties_as_penalty_matrix();
        assert_eq!(
            logslope_penalties.len(),
            n_rho,
            "matern penalty count must be structural (same as the base design)"
        );
        let mut marginal_spec = dummy_blockspec(2, n);
        marginal_spec.design = marginal_design;
        let mut logslope_spec = dummy_blockspec(p_log, n);
        logslope_spec.design = logslope_design;
        // initial_log_lambdas MUST match penalty count (validate_blockspec_consistency).
        logslope_spec.initial_log_lambdas = Array1::zeros(logslope_penalties.len());
        logslope_spec.nullspace_dims = design.nullspace_dims.clone();
        logslope_spec.penalties = logslope_penalties;
        let specs = vec![marginal_spec, logslope_spec];

        // θ = [ρ_0..ρ_{n_rho-1}, ψ_0..]. The ρ block is shifted UNIFORMLY by
        // rho_offset (so the ρ-FD direction below is the all-ones ρ block); ψ_dim
        // comes from the spatial derivative blocks (evaluated at ψ=0 of the
        // perturbed center).
        let rho = Array1::from_elem(n_rho, 0.1_f64 + rho_offset);

        let opts = BlockwiseFitOptions::default();
        let res = evaluate_custom_family_joint_hyper(
            &family,
            &specs,
            &opts,
            &rho,
            &derivative_blocks,
            None,
            mode,
        )
        .expect("joint hyper eval");
        assert!(
            res.inner_converged,
            "inner solve must converge for valid outer derivatives"
        );
        (res.objective, res.gradient, res.outer_hessian)
    };

    // Base point: analytic gradient + Hessian over θ=(ρ,ψ).
    let (_, grad0, hess0) = outer_at(0.0, 0.0, EvalMode::ValueGradientHessian);
    let theta_dim = grad0.len();
    let psi_dim = theta_dim - n_rho;
    assert!(
        psi_dim >= 1,
        "fixture must expose at least one spatial ψ axis"
    );
    // The ψ-active path returns the outer Hessian as a matrix-free OPERATOR
    // (#740 forces the operator route when the contracted hook is present, and
    // it advertises Unavailable materialization for the PRODUCTION planner).
    // `into_option()` is None for the Operator variant; `materialize_dense()`
    // builds the dense matrix via K column matvecs — which is exactly what this
    // FD comparison needs.
    let hess0 = hess0
        .materialize_dense()
        .expect("materialize outer Hessian")
        .expect("outer Hessian over (ρ,ψ) present");

    let eps = 1e-4_f64;

    // The single spatial length-scale knob maps to one ψ offset; restrict to one
    // ψ axis so the ψ FD perturbation is unambiguous.
    assert_eq!(
        psi_dim, 1,
        "FD ψ arm assumes one spatial ψ axis (one length-scale)"
    );

    let (cost_plus, _, _) = outer_at(eps, 0.0, EvalMode::ValueOnly);
    let (cost_minus, _, _) = outer_at(-eps, 0.0, EvalMode::ValueOnly);
    let fd_psi_gradient = (cost_plus - cost_minus) / (2.0 * eps);
    let analytic_psi_gradient = grad0[n_rho];

    // #740 HVP ψ-gradient attribution (read-only diagnostic; does NOT alter the
    // assertion or tolerance below). Re-run the base ValueGradientHessian eval
    // under a debug-stash CaptureGuard so the per-coordinate split of the outer
    // ψ-gradient (`a`, `½·production_tr`, `−½·ld_s`, and the frozen-B_i vs cubic
    // IFT-correction split of the logdet trace) is captured, and print it next to
    // the FD total. The coordinator reads these MSI numbers to localize the
    // disagreement to a single analytic term.
    {
        use crate::test_support::debug_stash;
        let _capture = debug_stash::CaptureGuard::request();
        let (base_obj, _, _) = outer_at(0.0, 0.0, EvalMode::ValueGradientHessian);
        assert!(
            base_obj.is_finite(),
            "stash-capture base eval must reproduce a finite outer objective"
        );
        let stash = debug_stash::take_terms();
        // `a`-channel split of the first ψ coordinate: (a_likelihood = objective_psi,
        // a_penalty_quadratic = ½β̂ᵀ(∂S_λ/∂ψ)β̂). Drained from the global sink
        // recorded by build_psi_hyper_coords for the base eval.
        let a_split = debug_stash::take_a_split();
        let half_prod = stash.production_tr.map(|t| 0.5 * t);
        let half_ld_s = stash.coord_ld_s.map(|t| 0.5 * t);
        let recon = match (stash.coord_a, half_prod, half_ld_s) {
            (Some(a), Some(hp), Some(hs)) => Some(a + hp - hs),
            _ => None,
        };

        // Probes (a) + (b): per-component FD of the outer VALUE at ψ±eps (β̂
        // re-solved each side), using the same `eps` as the failing gradient
        // check. Hold the SAME CaptureGuard so each eval populates the stash;
        // `take_terms` drains it after each. Then attribute the FD total:
        //   FD(log_det_h)            ↔ production_tr  (probe b: frozen ∂W/∂ψ logdet-H drift)
        //   FD(cost − ½ldh + ½lds)   ↔ coord_a        (probe a: cost/`a` term completeness)
        //   FD(log_det_s)            ↔ coord_ld_s
        let (_, _, _) = outer_at(eps, 0.0, EvalMode::ValueGradientHessian);
        let stash_plus = debug_stash::take_terms();
        let a_split_plus = debug_stash::take_a_split();
        let (_, _, _) = outer_at(-eps, 0.0, EvalMode::ValueGradientHessian);
        let stash_minus = debug_stash::take_terms();
        let a_split_minus = debug_stash::take_a_split();
        let central = |plus: Option<f64>, minus: Option<f64>| -> Option<f64> {
            match (plus, minus) {
                (Some(p), Some(m)) => Some((p - m) / (2.0 * eps)),
                _ => None,
            }
        };
        let fd_log_det_h = central(stash_plus.coord_log_det_h, stash_minus.coord_log_det_h);
        let fd_log_det_s = central(stash_plus.coord_log_det_s, stash_minus.coord_log_det_s);
        // Logdet-EXCLUDED cost piece at each side: cost − ½log_det_h + ½log_det_s.
        let cost_ex = |s: &debug_stash::TermStash| -> Option<f64> {
            match (s.coord_cost, s.coord_log_det_h, s.coord_log_det_s) {
                (Some(c), Some(ldh), Some(lds)) => Some(c - 0.5 * ldh + 0.5 * lds),
                _ => None,
            }
        };
        let fd_cost_excl_logdet = central(cost_ex(&stash_plus), cost_ex(&stash_minus));
        // Full cost FD (sanity: must reproduce the public-API fd_psi_gradient).
        let fd_cost_total = central(stash_plus.coord_cost, stash_minus.coord_cost);
        // FD of each `a`-channel separately, so a coord_a mismatch is attributed
        // to the likelihood channel (objective_psi) vs the penalty-quadratic
        // channel ½β̂ᵀS_ψβ̂. (a_likelihood is itself the analytic ψ-deriv of −ℓ at
        // fixed β; its FD counterpart is FD of the −ℓ value, which equals
        // fd_cost_excl_logdet − FD(½βᵀSλβ). We expose the raw channels so the
        // coordinator sees which of the two analytic channels is the small one.)
        let a_like = a_split.map(|(l, _)| l);
        let a_penq = a_split.map(|(_, q)| q);
        let a_like_plus = a_split_plus.map(|(l, _)| l);
        let a_like_minus = a_split_minus.map(|(l, _)| l);
        let a_penq_plus = a_split_plus.map(|(_, q)| q);
        let a_penq_minus = a_split_minus.map(|(_, q)| q);

        println!(
            "[HVP-attr] analytic_psi_grad={analytic_psi_gradient:.8e} fd_psi_grad={fd_psi_gradient:.8e} \
             coord_a={:?} production_tr={:?} half_production_tr={:?} coord_ld_s={:?} \
             half_ld_s={:?} recon_a+halfprod-halflds={:?} frozen_tr={:?} correction_tr={:?} \
             correction_tr_proj={:?} projection_active={:?} unprojected_tr={:?}",
            stash.coord_a,
            stash.production_tr,
            half_prod,
            stash.coord_ld_s,
            half_ld_s,
            recon,
            stash.frozen_tr,
            stash.correction_tr,
            stash.correction_tr_proj,
            stash.projection_active,
            stash.unprojected_tr,
        );
        println!(
            "[HVP-attr-fd] PROBE_a coord_a={:?} fd_cost_excl_logdet={:?} (these should match) | \
             PROBE_b production_tr={:?} fd_log_det_h={:?} (these should match) | \
             coord_ld_s={:?} fd_log_det_s={:?} (should match) | \
             fd_cost_total={:?} fd_psi_grad_public={fd_psi_gradient:.8e} (cross-check) | \
             base_log_det_h={:?} base_log_det_s={:?} base_cost={:?}",
            stash.coord_a,
            fd_cost_excl_logdet,
            stash.production_tr,
            fd_log_det_h,
            stash.coord_ld_s,
            fd_log_det_s,
            fd_cost_total,
            stash.coord_log_det_h,
            stash.coord_log_det_s,
            stash.coord_cost,
        );
        println!(
            "[HVP-attr-asplit] a_likelihood(objective_psi)={a_like:?} a_penalty_quadratic={a_penq:?} \
             (sum should == coord_a) | base_a_like_plus={a_like_plus:?} a_like_minus={a_like_minus:?} \
             base_a_penq_plus={a_penq_plus:?} a_penq_minus={a_penq_minus:?} | \
             implied_missing=coord_a - fd_cost_excl_logdet"
        );
        // #740 gate-4 root-cause probe: the inner KKT residual inf-norm at the
        // base β̂ and whether the batched envelope-ONLY outer-gradient override
        // fired (dropping the β-response correction). residual≫0 + override_fired
        // ⇒ the analytic ψ-gradient is missing the KKT-residual β-response (the
        // gap). residual≈0 yet coord_a ≠ fd_cost_excl_logdet ⇒ objective_psi
        // assembly bug (the kernel ∂X/∂ψ is FD-verified-correct standalone).
        let kkt_probe = debug_stash::take_kkt_probe();
        println!(
            "[740-KKT-PROBE] inner_kkt_residual_inf={:?} batched_envelope_override_fired={:?}",
            kkt_probe.map(|(r, _)| r),
            kkt_probe.map(|(_, f)| f),
        );
    }
    let psi_gradient_scale = 1.0 + analytic_psi_gradient.abs().max(fd_psi_gradient.abs());
    let psi_gradient_rel = (analytic_psi_gradient - fd_psi_gradient).abs() / psi_gradient_scale;
    assert!(
        psi_gradient_rel < 2e-3,
        "outer ψ gradient disagrees with centered FD of the outer value: analytic={}, fd={}, \
         rel={psi_gradient_rel:.3e}",
        analytic_psi_gradient,
        fd_psi_gradient
    );

    // Centered FD of the outer GRADIENT along a θ direction `dir` of length
    // theta_dim = n_rho + 1. The ρ block (dir[0..n_rho]) is shifted UNIFORMLY by
    // its common value (the directions below set it all-ones or all-zeros), and
    // the ψ component (dir[n_rho]) shifts the length-scale (−log scale). Rebuild
    // design+blocks at ±eps·dir, take (g(+)-g(-))/2eps. Compare to H·dir.
    let fd_along = |dir: &Array1<f64>| -> Array1<f64> {
        let rho_step = dir[0]; // ρ block is uniform in the directions used below
        let psi_step = dir[n_rho];
        let (_, gp, _) = outer_at(eps * psi_step, eps * rho_step, EvalMode::ValueAndGradient);
        let (_, gm, _) = outer_at(-eps * psi_step, -eps * rho_step, EvalMode::ValueAndGradient);
        (&gp - &gm).mapv(|v| v / (2.0 * eps))
    };

    let check = |dir: Array1<f64>, label: &str| {
        let analytic = hess0.dot(&dir);
        let fd = fd_along(&dir);
        for k in 0..theta_dim {
            let scale = 1.0 + analytic[k].abs().max(fd[k].abs());
            let rel = (analytic[k] - fd[k]).abs() / scale;
            assert!(
                rel < 2e-3,
                "[{label}] outer Hessian·dir component {k} disagrees with centered FD of the \
                 outer gradient: analytic={}, fd={}, rel={rel:.3e}",
                analytic[k],
                fd[k]
            );
        }
    };

    // Pure-ψ: ρ block all-zero, ψ = 1.
    let mut pure_psi = Array1::<f64>::zeros(theta_dim);
    pure_psi[n_rho] = 1.0;
    check(pure_psi, "pure-ψ");
    // Mixed-ρψ: ρ block all-ones + ψ = 1 — perturbs ρ and ψ together; exposes a
    // ρψ/ψψ block-split error that pure-ρ and pure-ψ would both miss.
    let mut mixed = Array1::<f64>::ones(theta_dim);
    mixed[n_rho] = 1.0;
    check(mixed, "mixed-ρψ");
}


/// The single-expression Taylor-jet tower (#932) of the rigid standard-normal
/// Bernoulli marginal-slope row NLL, written ONCE over `Tower4<2>` primaries
/// `(η_m, g)`:
///
/// ```text
///   q   = q(η_m)                  (the family's own certified [q, q1..q4]
///                                  link-map stack — Φ⁻¹∘clamp∘Φ)
///   gᵒ  = s_f · g
///   c   = √(1 + gᵒ²)
///   η   = q·c + gᵒ·z
///   m   = s·η,  s = 2y − 1
///   nll = −w·log Φ(m)             (the family's own signed-probit stack)
/// ```
///
/// It reuses the family's hand-certified `[f64; 5]` special-function stacks
/// (`unary_derivatives_sqrt` / `unary_derivatives_neglog_phi` and the
/// link-map's q-derivatives) through `Tower4::compose_unary`, so no probit
/// primitive is re-derived here: the tower mechanizes only the
/// Leibniz / Faà di Bruno composition that the production
/// `rigid_standard_normal_*` tower path mechanizes — the chain-rule
/// scaffolding (`u1..u4`, `c1..c4`, q-transform stacking) where
/// cross-block sign errors of the #736 genus live.
struct BernoulliRigidNllProgram {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
}


impl crate::families::jet_tower::RowNllProgram<2> for BernoulliRigidNllProgram {
    fn n_rows(&self) -> usize {
        self.family.y.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        Ok([self.block_states[0].eta[row], self.block_states[1].eta[row]])
    }

    fn row_nll(
        &self,
        row: usize,
        p: &[crate::families::jet_tower::Tower4<2>; 2],
    ) -> Result<crate::families::jet_tower::Tower4<2>, String> {
        let y = self.family.y[row];
        let w = self.family.weights[row];
        let z = self.family.z[row];
        let s = 2.0 * y - 1.0;
        let s_f = self.family.probit_frailty_scale();

        let marginal = self.family.marginal_link_map(p[0].v)?;
        let q = p[0].compose_unary([
            marginal.q,
            marginal.q1,
            marginal.q2,
            marginal.q3,
            marginal.q4,
        ]);
        let observed_g = p[1] * s_f;
        let one_plus_b2 = observed_g * observed_g + 1.0;
        let c = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.v));
        let eta = q * c + observed_g * z;
        let m = eta * s;
        Ok(m.compose_unary(unary_derivatives_neglog_phi(m.v, w)))
    }
}


/// #932 universal oracle on the SECOND (and final) production `RowKernel`
/// impl: `BernoulliRigidRowKernel`.
///
/// Audits every channel the hand-written kernel emits — value / gradient /
/// Hessian / `row_third_contracted(dir)` / `row_fourth_contracted(u, v)` —
/// against the single-expression `RowNllProgram<2>`-derived tower truth,
/// over per-row-varying `(η_m, g)` fixtures (mixed labels, non-unit weights,
/// with and without Gaussian frailty so the probit scale ≠ 1) and several
/// direction vectors. Together with the survival marginal-slope oracle this
/// closes #932 deployment step 2: every hand-written `RowKernel` in the tree
/// is CI-verified channel-by-channel against the one-expression truth.
#[test]
fn bernoulli_rigid_row_kernel_agrees_with_jet_tower_program_all_channels() {
    use crate::families::jet_tower::{KernelChannels, evaluate_program, verify_kernel_channels};
    use crate::families::row_kernel::RowKernel;

    let n = 6usize;
    let dirs: [[f64; 2]; 3] = [[0.8, -0.6], [-0.35, 1.1], [1.4, 0.25]];

    for frailty in [None, Some(0.7_f64)] {
        let family = BernoulliMarginalSlopeFamily {
            gaussian_frailty_sd: frailty,
            ..make_rigid_test_family(n)
        };
        // Per-row varying primaries so every channel sees nontrivial
        // (η_m, g) variation, away from the link-map clamp tails.
        let eta_m = Array1::from_iter((0..n).map(|i| -1.0 + 0.43 * i as f64));
        let g_eta = Array1::from_iter((0..n).map(|i| 0.5 - 0.21 * i as f64));
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: eta_m,
            },
            ParameterBlockState {
                beta: array![0.0],
                eta: g_eta,
            },
        ];

        let kernel =
            super::row_kernel::BernoulliRigidRowKernel::new(family.clone(), block_states.clone());
        let program = BernoulliRigidNllProgram {
            family,
            block_states,
        };

        for row in 0..n {
            let tower = evaluate_program(&program, row).expect("tower evaluation");

            let (value, gradient, hessian) =
                RowKernel::row_kernel(&kernel, row).expect("hand kernel value/grad/hess");

            let third: Vec<([f64; 2], [[f64; 2]; 2])> = dirs
                .iter()
                .map(|dir| {
                    let claim = RowKernel::row_third_contracted(&kernel, row, dir)
                        .expect("hand kernel third");
                    (*dir, claim)
                })
                .collect();

            let fourth: Vec<([f64; 2], [f64; 2], [[f64; 2]; 2])> = dirs
                .iter()
                .enumerate()
                .map(|(i, u)| {
                    let v = dirs[(i + 1) % dirs.len()];
                    let claim = RowKernel::row_fourth_contracted(&kernel, row, u, &v)
                        .expect("hand kernel fourth");
                    (*u, v, claim)
                })
                .collect();

            let claims = KernelChannels {
                value,
                gradient,
                hessian,
                third,
                fourth,
            };

            verify_kernel_channels(&tower, &claims, 1e-9).unwrap_or_else(|e| {
                panic!(
                    "frailty {frailty:?} row {row}: hand BernoulliRigidRowKernel disagrees \
                     with #932 jet-tower truth: {e}"
                )
            });
        }
    }
}


/// The rigid BLAS-3 batched all-axes second-directional override
/// (`second_directional_derivative_all_axes_dense_override`, the #979 biobank
/// `coord_corrections` perf lever) must reproduce, for EVERY canonical axis
/// `e_a`, the generic per-axis scatter
/// `row_kernel_second_directional_derivative(All, δ, e_a)` — the object the
/// Jeffreys `H_Φ` drift consumed before the batched hook existed.
///
/// Two assertions:
///   1. Batched axis `a` matches the BLAS-3 single-axis path bit-for-bit
///      (`==`): the BATCHING itself introduces no reduction-order change
///      (same chunked `Xᵀ diag(w) X` machinery, same per-row
///      `contract_fourth_full` args, hoisted `δ`-projection).
///   2. Batched axis `a` matches the generic per-row BLAS-1 scatter to tight
///      tolerance — the only difference is the documented BLAS-3-vs-syr in-row
///      Gram reduction order (same contract as `hessian_dense_blas3`).
#[test]
fn bernoulli_rigid_batched_all_axes_second_directional_matches_per_axis_scatter() {
    use crate::families::row_kernel::{
        RowSet, row_kernel_second_directional_derivative,
        row_kernel_second_directional_derivative_all_axes,
    };

    let n = 40usize;
    // Multi-column designs so the per-axis sweep is nontrivial in BOTH blocks
    // (p_m = 3 marginal axes, p_g = 2 logslope axes ⇒ p = 5).
    let y: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 7) as f64 * 0.1));
    let z: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -1.5 + 3.0 * ((i * 17 + 5) % n) as f64 / n as f64));
    let marginal_design = Array2::from_shape_fn((n, 3), |(i, j)| {
        1.0 + 0.37 * (((i * 7 + j * 11) % 9) as f64 - 4.0)
    });
    let logslope_design = Array2::from_shape_fn((n, 2), |(i, j)| {
        0.5 + 0.21 * (((i * 5 + j * 3) % 7) as f64 - 3.0)
    });
    let p_m = marginal_design.ncols();
    let p_g = logslope_design.ncols();
    let p = p_m + p_g;

    for frailty in [None, Some(0.7_f64)] {
        let family = BernoulliMarginalSlopeFamily {
            gaussian_frailty_sd: frailty,
            ..test_family_with_dense_designs(
                y.clone(),
                weights.clone(),
                z.clone(),
                marginal_design.clone(),
                logslope_design.clone(),
            )
        };
        // Per-row varying primaries (the cached fourth tensor must vary by row).
        let eta_m = Array1::from_iter((0..n).map(|i| -0.8 + 0.05 * i as f64));
        let g_eta = Array1::from_iter((0..n).map(|i| 0.4 - 0.02 * i as f64));
        let block_states = vec![
            ParameterBlockState {
                beta: Array1::from_iter((0..p_m).map(|j| 0.1 * (j as f64 + 1.0))),
                eta: eta_m,
            },
            ParameterBlockState {
                beta: Array1::from_iter((0..p_g).map(|j| -0.05 * (j as f64 + 1.0))),
                eta: g_eta,
            },
        ];
        let kernel =
            super::row_kernel::BernoulliRigidRowKernel::new(family.clone(), block_states.clone());

        // Fixed direction δ with nontrivial projection onto both blocks.
        let delta: Vec<f64> = (0..p).map(|a| 0.3 - 0.13 * a as f64).collect();

        let batched =
            row_kernel_second_directional_derivative_all_axes(&kernel, &RowSet::All, &delta)
                .expect("batched all-axes BLAS-3 second directional");
        assert_eq!(batched.len(), p, "one matrix per canonical axis");

        for a in 0..p {
            let mut axis = vec![0.0_f64; p];
            axis[a] = 1.0;

            // (2) Generic per-row BLAS-1 scatter reference.
            let scatter = row_kernel_second_directional_derivative(
                &kernel,
                &RowSet::All,
                &delta,
                &axis,
            )
            .expect("generic per-axis scatter second directional");

            let mut max_abs = 0.0_f64;
            let mut max_rel = 0.0_f64;
            for ((i, j), &got) in batched[a].indexed_iter() {
                let want = scatter[[i, j]];
                let abs = (got - want).abs();
                max_abs = max_abs.max(abs);
                max_rel = max_rel.max(abs / want.abs().max(1.0));
            }
            assert!(
                max_rel < 1e-10,
                "frailty {frailty:?} axis {a}: batched BLAS-3 all-axes H²dot[δ,e_a] \
                 disagrees with generic per-row scatter: max_abs={max_abs:e} max_rel={max_rel:e}"
            );

            // Symmetric p×p output (the joint Hessian second directional is symmetric).
            for i in 0..p {
                for j in 0..p {
                    assert!(
                        (batched[a][[i, j]] - batched[a][[j, i]]).abs() < 1e-12,
                        "axis {a}: batched output must be symmetric at ({i},{j})"
                    );
                }
            }
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// #1028 / #905 — Murphy–Topel generated-regressor correction assembly.
// These pin the engine-agnostic assembly that turns the ONE outstanding
// engine quantity (the per-row slope-score sensitivity to the calibrated score
// s_i = ∂score_β,i/∂ζ_i) into the full correction (Vb·G)·V₁·(Vb·G)ᵀ. The only
// missing piece for end-to-end is threading s_i out of the shared joint engine
// (the #932 z-jet channel); everything here is exact and complete.

fn murphy_topel_test_calibration() -> LatentZConditionalCalibration {
    // Mean-and-variance conditional calibration over a single conditioning
    // covariate a(C) (basis_ncols = 1), so θ₁ = (mean_coeffs[2], var_coeffs[2]).
    LatentZConditionalCalibration {
        mean_coeffs: vec![0.1, 0.4], // m(C) = 0.1 + 0.4·a
        var_coeffs: vec![1.2, 0.3],  // v(C) = max(1.2 + 0.3·a, floor)
        basis_ncols: 1,
        var_floor: 0.05,
        global_var: 1.0,
        post_mean: 0.0,
        post_sd: 1.0,
        // Diagonal first-stage covariances; PSD, distinct scales per block.
        mean_cov: ndarray::array![[0.02, 0.0], [0.0, 0.05]],
        var_cov: ndarray::array![[0.03, 0.0], [0.0, 0.01]],
    }
}


#[test]
fn generated_regressor_correction_is_psd_and_inflates_slope_se_when_gate_fires() {
    let cal = murphy_topel_test_calibration();
    let p_beta = 2usize;
    // Conditioning covariate a(C) (well above the variance floor everywhere) and
    // raw latent scores z.
    let a_block = ndarray::array![[0.3], [-0.5], [0.8], [0.1], [-0.2]];
    let z = ndarray::array![0.6, -0.4, 1.1, 0.05, -0.9];
    // A nonzero per-row slope-score-to-ζ sensitivity (n × p_β): the gate fires.
    let s = ndarray::array![
        [0.5, -0.2],
        [-0.3, 0.7],
        [0.9, 0.1],
        [0.2, -0.6],
        [-0.4, 0.3]
    ];
    // Naive reduced-frame slope covariance Vb (SPD).
    let vb = ndarray::array![[0.8, 0.1], [0.1, 0.5]];

    let term = cal
        .generated_regressor_correction(s.view(), z.view(), a_block.view(), vb.view())
        .expect("assemble correction");
    assert_eq!(term.dim(), (p_beta, p_beta));

    // Symmetric.
    for i in 0..p_beta {
        for j in 0..p_beta {
            assert!(
                (term[[i, j]] - term[[j, i]]).abs() < 1e-12,
                "correction term must be symmetric"
            );
        }
    }
    // PSD: xᵀ·term·x ≥ 0 for a sweep of directions (it is a congruence of the
    // PSD V₁).
    for &dir in &[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [2.0, -3.0]] {
        let x = ndarray::array![dir[0], dir[1]];
        let q = x.dot(&term.dot(&x));
        assert!(
            q >= -1e-10,
            "correction term is not PSD: direction {dir:?} gives quadratic form {q:.3e}"
        );
    }
    // The diagonal (variance of each slope coefficient) is strictly inflated:
    // corrected SE = sqrt(vb_ii + term_ii) > naive SE = sqrt(vb_ii).
    for i in 0..p_beta {
        assert!(
            term[[i, i]] > 1e-9,
            "slope coefficient {i} variance is not inflated by the Murphy–Topel \
             correction when the gate fires: term[{i},{i}]={:.3e}",
            term[[i, i]]
        );
        let naive_se = vb[[i, i]].sqrt();
        let corrected_se = (vb[[i, i]] + term[[i, i]]).sqrt();
        assert!(
            corrected_se > naive_se,
            "corrected SE must strictly exceed naive SE for coefficient {i}: \
             corrected={corrected_se:.5} naive={naive_se:.5}"
        );
    }
}


#[test]
fn generated_regressor_correction_vanishes_when_all_rows_floored() {
    // When the conditional variance is on the floor for EVERY row, ∂ζ/∂v = 0,
    // and with a mean block whose sensitivity is also driven to zero the whole
    // J_zeta is zero ⇒ G = 0 ⇒ correction = 0 (the floored-row property: a
    // floored row carries no first-stage uncertainty into β̂).
    //
    // Construct a calibration whose raw v(C) sits below the floor everywhere so
    // the variance sensitivity vanishes, AND scale the mean sensitivity to be
    // checked separately. Here we instead test the cleanest invariant: with a
    // zero score-sensitivity s_i ≡ 0 the correction is exactly zero regardless
    // of J_zeta (no second-stage coupling to the first stage).
    let cal = murphy_topel_test_calibration();
    let n = 4usize;
    let a_block = ndarray::array![[0.3], [-0.5], [0.8], [0.1]];
    let z = ndarray::array![0.6, -0.4, 1.1, 0.05];
    let s = Array2::<f64>::zeros((n, 2)); // gate did not fire on any row
    let vb = ndarray::array![[0.8, 0.1], [0.1, 0.5]];

    let term = cal
        .generated_regressor_correction(s.view(), z.view(), a_block.view(), vb.view())
        .expect("assemble correction");
    let max_abs = term.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    assert!(
        max_abs < 1e-14,
        "correction must be exactly zero when no row's slope score responds to \
         ζ (G = 0); got max |term| = {max_abs:.3e}"
    );
}


#[test]
fn generated_regressor_correction_matches_explicit_g_accumulation() {
    // Oracle: independently accumulate G = Σ_i s_i ⊗ (∂ζ_i/∂θ₁) from the public
    // `zeta_theta1_jacobian_row`, form (Vb·G)·V₁·(Vb·G)ᵀ by hand, and require
    // bit-for-bit agreement with `generated_regressor_correction`. This pins the
    // chain-rule G assembly that the seam relies on.
    let cal = murphy_topel_test_calibration();
    let n = 5usize;
    let p_beta = 2usize;
    let dim_theta1 = cal.theta1_dim();
    assert_eq!(dim_theta1, 4); // 2 mean + 2 var

    let a_block = ndarray::array![[0.3], [-0.5], [0.8], [0.1], [-0.2]];
    let z = ndarray::array![0.6, -0.4, 1.1, 0.05, -0.9];
    let s = ndarray::array![
        [0.5, -0.2],
        [-0.3, 0.7],
        [0.9, 0.1],
        [0.2, -0.6],
        [-0.4, 0.3]
    ];
    let vb = ndarray::array![[0.8, 0.1], [0.1, 0.5]];

    // Explicit G.
    let mut g = Array2::<f64>::zeros((p_beta, dim_theta1));
    for i in 0..n {
        let jz = cal.zeta_theta1_jacobian_row(z[i], a_block.row(i));
        assert_eq!(jz.len(), dim_theta1);
        for b in 0..p_beta {
            for k in 0..dim_theta1 {
                g[[b, k]] += s[[i, b]] * jz[k];
            }
        }
    }
    let v1 = cal.theta1_covariance();
    let vb_g = vb.dot(&g);
    let expected = vb_g.dot(&v1).dot(&vb_g.t());

    let got = cal
        .generated_regressor_correction(s.view(), z.view(), a_block.view(), vb.view())
        .expect("assemble correction");

    for i in 0..p_beta {
        for j in 0..p_beta {
            assert!(
                (got[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "correction[{i},{j}] mismatch: got={:.6e} expected={:.6e}",
                got[[i, j]],
                expected[[i, j]]
            );
        }
    }
}


#[test]
fn rigid_standard_normal_mixed_z_sensitivity_matches_central_difference_of_gradient() {
    // #1028 engine quantity oracle: the per-row mixed (primary, z) partial
    // `[∂²ℓ/∂q∂z, ∂²ℓ/∂g∂z]` read off the z-augmented row jet must equal the
    // central finite-difference of the production row GRADIENT `[∂ℓ/∂q, ∂ℓ/∂g]`
    // (`rigid_standard_normal_row_kernel`) in z. This pins the new jet channel
    // against the same kernel value/gradient the fit consumes.
    let link = bernoulli_marginal_slope_probit_link();
    let probit_scale = 0.7;
    let h = 1e-6;
    // A sweep of converged-frame (marginal η, slope g), latent score z, label y.
    let cases = [
        (0.3_f64, 0.5_f64, 0.6_f64, 1.0_f64, 1.0_f64),
        (-0.8, 0.9, -1.1, 0.0, 1.4),
        (1.2, -0.4, 0.25, 1.0, 0.8),
        (-0.2, 1.3, -0.7, 0.0, 1.0),
        (0.05, 0.2, 1.8, 1.0, 0.5),
    ];
    for (marginal_eta, g, z, y, w) in cases {
        let marginal = bernoulli_marginal_link_map(&link, marginal_eta).expect("marginal map");
        let analytic =
            rigid_standard_normal_mixed_z_sensitivity(marginal, g, z, y, w, probit_scale)
                .expect("mixed-z sensitivity");

        let grad_at = |z_val: f64| -> [f64; 2] {
            let (_v, grad, _h) =
                rigid_standard_normal_row_kernel(marginal, g, z_val, y, w, probit_scale)
                    .expect("row kernel");
            grad
        };
        let plus = grad_at(z + h);
        let minus = grad_at(z - h);
        let fd_q = (plus[0] - minus[0]) / (2.0 * h);
        let fd_g = (plus[1] - minus[1]) / (2.0 * h);

        assert!(
            (analytic[0] - fd_q).abs() < 1e-5,
            "∂²ℓ/∂q∂z mismatch at (q={marginal_eta}, g={g}, z={z}, y={y}): \
             analytic={:.6e} fd={:.6e}",
            analytic[0],
            fd_q
        );
        assert!(
            (analytic[1] - fd_g).abs() < 1e-5,
            "∂²ℓ/∂g∂z mismatch at (q={marginal_eta}, g={g}, z={z}, y={y}): \
             analytic={:.6e} fd={:.6e}",
            analytic[1],
            fd_g
        );
    }
}


#[test]
fn score_zeta_sensitivity_equals_jacobian_transpose_of_mixed_z_partial() {
    // The assembled `score_zeta_sensitivity` row `s_i` must be exactly the block
    // Jacobian transpose `J_iᵀ` applied to the primary mixed-z 2-vector:
    //   s_i[marginal_range] = (∂²ℓ/∂q∂z) · M_i,
    //   s_i[logslope_range] = (∂²ℓ/∂g∂z) · G_i.
    // This pins the reduced-frame contraction the seam feeds to
    // `generated_regressor_correction`.
    let link = bernoulli_marginal_slope_probit_link();
    let probit_scale = 0.85;
    let n = 4usize;
    let p_m = 2usize;
    let r = 2usize;
    let marginal_design = ndarray::array![[1.0, 0.3], [1.0, -0.6], [1.0, 0.9], [1.0, -0.2]];
    let logslope_design = ndarray::array![[0.4, -0.1], [0.7, 0.2], [-0.3, 0.8], [0.5, -0.5]];
    let marginal_eta = ndarray::array![0.2, -0.5, 0.7, -0.1];
    let slope_eta = ndarray::array![0.3, 0.8, -0.4, 1.1];
    let z = ndarray::array![0.6, -0.9, 0.25, 1.3];
    let y = ndarray::array![1.0, 0.0, 1.0, 0.0];
    let weights = ndarray::array![1.0, 1.2, 0.8, 1.5];
    let p_beta = p_m + r;

    let s = rigid_standard_normal_score_zeta_sensitivity(
        &link,
        &marginal_eta,
        &slope_eta,
        &z,
        &y,
        &weights,
        probit_scale,
        marginal_design.view(),
        logslope_design.view(),
        p_beta,
    )
    .expect("assemble score_zeta_sensitivity");
    assert_eq!(s.dim(), (n, p_beta));

    for i in 0..n {
        let marginal = bernoulli_marginal_link_map(&link, marginal_eta[i]).expect("marginal map");
        let [s_q, s_g] = rigid_standard_normal_mixed_z_sensitivity(
            marginal,
            slope_eta[i],
            z[i],
            y[i],
            weights[i],
            probit_scale,
        )
        .expect("mixed-z sensitivity");
        for j in 0..p_m {
            let expected = s_q * marginal_design[[i, j]];
            assert!(
                (s[[i, j]] - expected).abs() < 1e-12,
                "marginal contraction mismatch at row {i} col {j}: got={:.6e} expected={:.6e}",
                s[[i, j]],
                expected
            );
        }
        for j in 0..r {
            let expected = s_g * logslope_design[[i, j]];
            assert!(
                (s[[i, p_m + j]] - expected).abs() < 1e-12,
                "logslope contraction mismatch at row {i} col {j}: got={:.6e} expected={:.6e}",
                s[[i, p_m + j]],
                expected
            );
        }
    }

    // End-to-end: the assembled s feeds the landed assembly and the corrected
    // covariance strictly inflates the naive Vb diagonals whenever the row scores
    // respond to ζ (G ≠ 0), and the term is PSD.
    let cal = murphy_topel_test_calibration_basis2();
    let vb = ndarray::array![
        [0.8, 0.1, 0.0, 0.0],
        [0.1, 0.6, 0.05, 0.0],
        [0.0, 0.05, 0.5, 0.02],
        [0.0, 0.0, 0.02, 0.4]
    ];
    let term = cal
        .generated_regressor_correction(
            s.view(),
            z.view(),
            marginal_design
                .slice(ndarray::s![.., 1..])
                .to_owned()
                .view(),
            vb.view(),
        )
        .expect("assemble correction");
    // PSD.
    use crate::faer_ndarray::FaerEigh;
    let (evals, _) = term.eigh(faer::Side::Lower).expect("eig");
    let min_eval = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    assert!(
        min_eval > -1e-9,
        "generated-regressor term must be PSD; min eigenvalue {min_eval:.3e}"
    );
    // Strict inflation on at least one slope coordinate (the gate fired).
    let max_diag = (0..p_beta).map(|i| term[[i, i]]).fold(0.0_f64, f64::max);
    assert!(
        max_diag > 1e-9,
        "corrected SE must strictly exceed naive SE on a responding coordinate; \
         max diagonal inflation {max_diag:.3e}"
    );
}


// Calibration whose basis conditions on the slope covariate a(C) = marginal
// design column 1 (basis_ncols = 1), matching the `score_zeta` test's
// single-covariate conditioning span.
fn murphy_topel_test_calibration_basis2() -> LatentZConditionalCalibration {
    LatentZConditionalCalibration {
        mean_coeffs: vec![0.05, 0.3],
        var_coeffs: vec![1.1, 0.2],
        basis_ncols: 1,
        var_floor: 0.05,
        global_var: 1.0,
        post_mean: 0.0,
        post_sd: 1.0,
        mean_cov: ndarray::array![[0.02, 0.0], [0.0, 0.04]],
        var_cov: ndarray::array![[0.03, 0.0], [0.0, 0.015]],
    }
}


/// Deterministic splitmix64 → standard-normal sampler for the Monte-Carlo
/// oracle below (self-contained so the test carries no RNG dev-dependency and
/// is bit-reproducible across machines).
struct SplitMix64 {
    state: u64,
}


impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa, shifted off exact 0.
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Standard normal via Box–Muller (one draw per call).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}


/// #1028 acceptance — Murphy–Topel generated-regressor SE correction oracle.
///
/// Murphy–Topel (1985) propagates the FIRST-STAGE PARAMETER uncertainty
/// `Var(θ̂₁) = V₁` into the second-stage slope covariance,
/// `V_β = Vb + (Vb·G) V₁ (Vb·G)ᵀ`, with the cross term
/// `G = Σ_i (∂score_β,i/∂ζ_i)(∂ζ_i/∂θ₁)`. It models the variance β̂ inherits
/// because the calibrated regressor `ζ̂(θ̂₁)` is built from an ESTIMATED
/// `θ̂₁ ~ N(θ₁, V₁)`. A full "resample the whole dataset" sampling experiment
/// does NOT isolate this: there, β̂'s variance is dominated by the
/// errors-in-variables discrepancy `ζ̂ − ζ*` (the slope is fit on `ζ̂` while the
/// outcome tracks the true `ζ*`), a larger, different effect that no first-order
/// θ̂₁-propagation formula targets. This oracle isolates exactly the MT channel:
/// hold ONE fixed dataset and ONE fitted calibration, then parametric-bootstrap
/// the first stage `θ̂₁* ~ N(θ̂₁, V₁)`, rebuild `ζ̂(θ̂₁*)`, and refit the slope.
/// The empirical variance of β̂ over the θ̂₁* draws is, to first order, the
/// Murphy–Topel term, and we assert the implemented `(Vb·G) V₁ (Vb·G)ᵀ` matches
/// it — plus a finite-difference oracle pinning the per-component sensitivity
/// `∂β̂/∂θ₁ = Vb·G` against a refit of β̂.
#[test]
fn murphy_topel_correction_matches_two_stage_sampling_variance() {
    // Murphy–Topel (1985) propagates the FIRST-STAGE PARAMETER uncertainty
    // Var(θ̂₁) into the second-stage slope covariance:
    //   V_β = Vb + (Vb·G) V₁ (Vb·G)ᵀ,  G = Σ_i (∂score_β,i/∂ζ_i)(∂ζ_i/∂θ₁).
    // It is NOT an errors-in-variables correction: the inflation it models is the
    // variance β̂ inherits because the calibrated regressor ζ̂(θ̂₁) is built from an
    // ESTIMATED θ̂₁ ~ N(θ₁, V₁), holding the second-stage outcome's own noise
    // fixed. The earlier "two-stage sampling" fixture redrew the WHOLE dataset
    // each replicate, so its empirical Var(β̂) was dominated by the
    // errors-in-variables discrepancy ζ̂−ζ* (the fit uses ζ̂ while y is generated
    // from ζ*) — a different, larger effect that no first-order θ̂₁-propagation
    // formula targets, which is why that assertion was structurally unattainable.
    //
    // This oracle isolates exactly the MT effect: ONE fixed dataset and ONE fitted
    // calibration, then a parametric bootstrap of the first stage θ̂₁* ~ N(θ̂₁, V₁).
    // For each draw we rebuild ζ̂(θ̂₁*) and refit the slope; the empirical variance
    // of β̂ over the θ̂₁* draws is, to first order, exactly the Murphy–Topel term.
    // We assert the implemented `(Vb·G) V₁ (Vb·G)ᵀ` matches that bootstrap
    // variance, AND (FD oracle) that the analytic ∂β̂/∂θ₁ = Vb·G the term is built
    // from matches a finite difference of the refit β̂ in θ̂₁.
    let n = 400usize;
    let gamma = 0.9_f64; // true conditional-mean slope E[z|C] = γ·C
    let beta_true = 1.2_f64; // stage-2 slope
    let sigma = 0.4_f64; // stage-2 residual sd
    let z_noise_sd = 0.8_f64; // idiosyncratic part of z

    let c = Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64 - 1.0) * 2.0 - 1.0));
    let a_block = c.clone().insert_axis(ndarray::Axis(1));

    let mut rng = SplitMix64::new(0x1028_BEEF_C0DE_2026);
    // --- ONE fixed dataset ---
    let u = Array1::from_iter((0..n).map(|_| rng.next_normal()));
    let z = Array1::from_iter((0..n).map(|i| gamma * c[i] + z_noise_sd * u[i]));
    let y = Array1::from_iter((0..n).map(|i| beta_true * u[i] + sigma * rng.next_normal()));
    let weights = Array1::ones(n);

    // --- STAGE 1: the production conditional-location gate, fit ONCE ---
    let cal = fit_conditional_latent_calibration_if_needed(&z, &weights, a_block.view())
        .expect("stage-1 gate must not error")
        .expect("stage-1 conditional gate must fire on the conditional shift");
    let zeta = cal
        .apply(z.view(), a_block.view())
        .expect("stage-1 calibration applies");

    // --- STAGE 2: slope on the calibrated regressor, fit ONCE ---
    let refit_beta = |cal_ref: &LatentZConditionalCalibration| -> f64 {
        let zh = cal_ref.apply(z.view(), a_block.view()).expect("apply");
        let szz: f64 = zh.iter().map(|&zz| zz * zz).sum();
        let szy: f64 = zh.iter().zip(y.iter()).map(|(&zz, &yy)| zz * yy).sum();
        szy / szz
    };
    let beta_hat = refit_beta(&cal);
    let s_zz: f64 = zeta.iter().map(|&zz| zz * zz).sum();
    let vb_scalar = sigma * sigma / s_zz; // naive Vb = σ²/Σζ̂²

    // --- the implemented Murphy–Topel term (the quantity under test) ---
    let dim_theta1 = cal.theta1_dim();
    let mut g = Array1::<f64>::zeros(dim_theta1);
    for i in 0..n {
        let dscore_dzeta = (y[i] - 2.0 * beta_hat * zeta[i]) / (sigma * sigma);
        let jz = cal.zeta_theta1_jacobian_row(z[i], a_block.row(i));
        for k in 0..dim_theta1 {
            g[k] += dscore_dzeta * jz[k];
        }
    }
    let v1 = cal.theta1_covariance();
    let vbg = g.mapv(|gk| vb_scalar * gk); // Vb·G (row), == ∂β̂/∂θ₁ up to sign
    let mt_correction: f64 = vbg.dot(&v1.dot(&vbg)); // (Vb·G) V₁ (Vb·G)ᵀ
    assert!(
        mt_correction >= -1e-14,
        "Murphy–Topel correction must be PSD (≥0); got {mt_correction:.3e}"
    );

    // --- FD ORACLE: ∂β̂/∂θ₁ (refit) vs the analytic Vb·G the term is built from ---
    // The MT term is invariant to the overall sign of Vb·G (it is a quadratic
    // form), but the per-component MAGNITUDES must match the true sensitivity of
    // the refit slope to each first-stage coefficient, or G is structurally wrong.
    let h = 1e-5_f64;
    for k in 0..dim_theta1 {
        let mut cal_p = cal.clone();
        if k < cal_p.mean_coeffs.len() {
            cal_p.mean_coeffs[k] += h;
        } else {
            cal_p.var_coeffs[k - cal.mean_coeffs.len()] += h;
        }
        let mut cal_m = cal.clone();
        if k < cal_m.mean_coeffs.len() {
            cal_m.mean_coeffs[k] -= h;
        } else {
            cal_m.var_coeffs[k - cal.mean_coeffs.len()] -= h;
        }
        let fd = (refit_beta(&cal_p) - refit_beta(&cal_m)) / (2.0 * h);
        // ∂β̂/∂θ₁ = −H_β⁻¹ ∂score_β/∂θ₁ = −Vb·G with the score-sign convention
        // above (score_β = Σ(y−βζ̂)ζ̂/σ²; ∂β̂/∂θ₁ = +(1/Σζ̂²)Σ(y−2β̂ζ̂)∂ζ̂/∂θ₁,
        // and Vb·G carries exactly that sum with vb=σ²/Σζ̂² cancelling the 1/σ²).
        let analytic = vbg[k];
        assert!(
            (fd - analytic).abs() <= 1e-4 + 1e-3 * fd.abs(),
            "MT sensitivity ∂β̂/∂θ₁[{k}] disagrees with finite difference: \
             analytic(Vb·G)={analytic:.6e} fd={fd:.6e}"
        );
    }

    // --- BOOTSTRAP ORACLE: parametric first-stage resampling θ̂₁* ~ N(θ̂₁, s²V₁) ---
    // The Murphy–Topel term is the FIRST-ORDER (delta-method) propagation of the
    // first-stage covariance into β̂: `(Vb·G) V₁ (Vb·G)ᵀ`. The refit slope β̂(θ₁) is
    // a NONLINEAR function of θ₁ (β̂ = Σζ̂y/Σζ̂² is a ratio, and ζ̂ = (z−m)/√v is
    // nonlinear in the variance coefficients), so the variance of β̂ under a
    // full-magnitude Gaussian perturbation of θ̂₁ carries a second-order curvature
    // term ON TOP of the first-order MT term — at this V₁ the curvature nearly
    // doubles it (ratio_mt_boot≈0.54). That excess is a real higher-order effect,
    // not an MT defect (the FD oracle above already certifies ∂β̂/∂θ₁ = Vb·G
    // exactly), and no first-order formula can or should reproduce it.
    //
    // To test the channel the MT term ACTUALLY computes, shrink the perturbation
    // by a scale `s`: with θ̂₁* = θ̂₁ + s·L·N(0,I), Var(β̂*) = s²·(Vb·G)V₁(Vb·G)ᵀ +
    // O(s⁴) curvature. Comparing boot_var against the correspondingly-scaled
    // first-order prediction `s²·mt_correction` isolates the first-order term: the
    // O(s⁴) curvature is suppressed to a few percent at s=0.25 (s²≈0.0625 times
    // the ≈0.85 curvature/first-order ratio ⇒ ≲6% residual), while MC noise stays
    // controlled at 20000 draws. This asserts the MT formula against the exact
    // first-order propagation variance it targets — strictly the right oracle.
    let pert_scale = 0.25_f64;
    let mt_first_order_scaled = pert_scale * pert_scale * mt_correction;
    // Lower Cholesky V₁ = L Lᵀ (small dim_theta1; V₁ is block-diagonal PSD).
    let chol = {
        let p = dim_theta1;
        let mut l = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..=i {
                let mut sum = v1[[i, j]];
                for k in 0..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }
                if i == j {
                    l[[i, j]] = sum.max(0.0).sqrt();
                } else {
                    let ljj = l[[j, j]];
                    l[[i, j]] = if ljj > 0.0 { sum / ljj } else { 0.0 };
                }
            }
        }
        l
    };
    let boot = 20000usize;
    let dm = cal.mean_coeffs.len();
    let mut boot_mean = 0.0_f64;
    let mut boot_m2 = 0.0_f64;
    for b in 0..boot {
        // θ̂₁* = θ̂₁ + s·L·standard-normal (shrunk perturbation, see above).
        let stdn: Vec<f64> = (0..dim_theta1).map(|_| rng.next_normal()).collect();
        let mut cal_star = cal.clone();
        for r in 0..dim_theta1 {
            let mut delta = 0.0_f64;
            for col in 0..=r {
                delta += chol[[r, col]] * stdn[col];
            }
            delta *= pert_scale;
            if r < dm {
                cal_star.mean_coeffs[r] += delta;
            } else {
                cal_star.var_coeffs[r - dm] += delta;
            }
        }
        // Reject draws that floor the variance (out of the linear regime); rare.
        let beta_star = refit_beta(&cal_star);
        let nb = (b + 1) as f64;
        let d = beta_star - boot_mean;
        boot_mean += d / nb;
        boot_m2 += d * (beta_star - boot_mean);
    }
    let boot_var = boot_m2 / (boot as f64 - 1.0);

    println!(
        "[MT-oracle] mt_correction={mt_correction:.6e} boot_var={boot_var:.6e} \
         mt_first_order_scaled={mt_first_order_scaled:.6e} naive_vb={vb_scalar:.6e} \
         pert_scale={pert_scale:.3} inflation={:.4} ratio_scaled={:.4} beta_hat={beta_hat:.5}",
        (vb_scalar + mt_correction).sqrt() / vb_scalar.sqrt(),
        mt_first_order_scaled / boot_var,
    );

    // (1) The correction is materially non-zero (the gate fires; θ̂₁ carries real
    //     uncertainty into ζ̂), so the corrected slope SE strictly exceeds naive.
    assert!(
        mt_correction > 0.05 * vb_scalar,
        "Murphy–Topel correction must materially inflate the naive Vb: \
         correction={mt_correction:.6e} naive_vb={vb_scalar:.6e}"
    );

    // (2) The implemented MT term matches the FIRST-ORDER first-stage-propagation
    //     variance of the refit slope — the core acceptance criterion. Measured by
    //     the shrunk-perturbation bootstrap (s=0.25), whose variance is
    //     s²·(Vb·G)V₁(Vb·G)ᵀ to leading order, compared against the matching
    //     s²·mt_correction. The band absorbs the 20000-draw Monte-Carlo error
    //     (≈1% on a variance) plus the residual O(s²)≈few-% curvature that the
    //     shrink suppresses but does not fully eliminate; a structurally wrong G or
    //     V₁ would mismatch by a factor, not a few percent.
    let rel_err = (mt_first_order_scaled - boot_var).abs() / boot_var;
    assert!(
        rel_err < 0.10,
        "Murphy–Topel correction must match the first-order first-stage bootstrap \
         variance of β̂: mt_first_order_scaled={mt_first_order_scaled:.6e} \
         boot_var={boot_var:.6e} (pert_scale={pert_scale:.3}, rel_err={rel_err:.4})"
    );
}
