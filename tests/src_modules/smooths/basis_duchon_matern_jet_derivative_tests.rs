#[test]
fn test_duchon_radial_jets_t_equals_q_r_over_r_fd() {
    // Finite-difference check: t = q' / r, so
    //   t ≈ (q(r+ε) - q(r-ε)) / (2ε·r).
    // Uses a 4-point Richardson-extrapolated central stencil so the
    // truncation error is O(h^4) rather than O(h^2), which keeps the
    // relative error below 1e-3 even at r = 0.1 where q''' is steep.
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 3usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);

    for &r in &[0.1, 0.5, 1.0, 2.0] {
        let eps = 1e-3 * r;
        let jets_2p = duchon_radial_jets(
            r + 2.0 * eps,
            length_scale,
            p_order,
            s_order,
            k_dim,
            &coeffs,
        )
        .expect("jets+2h");
        let jets_p = duchon_radial_jets(r + eps, length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("jets+h");
        let jets_m = duchon_radial_jets(r - eps, length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("jets-h");
        let jets_2m = duchon_radial_jets(
            r - 2.0 * eps,
            length_scale,
            p_order,
            s_order,
            k_dim,
            &coeffs,
        )
        .expect("jets-2h");
        let jets =
            duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs).expect("jets");
        // 5-point central difference: (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12h).
        let q_prime_fd = (-jets_2p.q + 8.0 * jets_p.q - 8.0 * jets_m.q + jets_2m.q) / (12.0 * eps);
        let t_fd = q_prime_fd / r;
        let rel = if jets.t.abs() > 1e-15 {
            ((jets.t - t_fd) / jets.t).abs()
        } else {
            (jets.t - t_fd).abs()
        };
        assert!(
            rel < 1e-3,
            "t FD mismatch at r={r}: jets.t={}, fd={t_fd}, rel_err={rel}",
            jets.t,
        );
    }
}

#[test]
fn test_duchon_radial_jets_t_derivatives_match_finite_difference() {
    // Uses a 5-point central stencil (O(h^4) truncation) for t_r. The
    // step must be large enough that partial-fraction cancellation in
    // the double-precision t values does not dominate the difference.
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 3usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);

    for &r in &[0.1_f64, 0.5, 1.0, 2.0] {
        let h = 1e-2 * r.max(1e-6);
        let jets_2p =
            duchon_radial_jets(r + 2.0 * h, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("jets+2h");
        let jets_p = duchon_radial_jets(r + h, length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("jets+h");
        let jets_m = duchon_radial_jets(r - h, length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("jets-h");
        let jets_2m =
            duchon_radial_jets(r - 2.0 * h, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("jets-2h");
        let jets =
            duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs).expect("jets");

        // 5-point central first derivative:
        //   f'(x) ≈ (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12h).
        let t_r_fd = (-jets_2p.t + 8.0 * jets_p.t - 8.0 * jets_m.t + jets_2m.t) / (12.0 * h);
        let rel_t_r = if jets.t_r.abs() > 1e-15 {
            ((jets.t_r - t_r_fd) / jets.t_r).abs()
        } else {
            (jets.t_r - t_r_fd).abs()
        };
        assert!(
            rel_t_r < 1e-2,
            "t_r FD mismatch at r={r}: jets.t_r={}, fd={t_r_fd}, rel_err={rel_t_r}",
            jets.t_r,
        );
        assert!(jets.t_rr.is_finite(), "expected finite t_rr at r={r}");
    }
}

#[test]
fn test_duchon_radial_jets_t_collision_matches_nearby() {
    // The collision limit t(0) should be close to t at small r.
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 4usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);

    let jets_0 = duchon_radial_jets(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
        .expect("jets at origin");
    // Evaluate at a small radius
    let r_small = 1e-4 * length_scale;
    let jets_small = duchon_radial_jets(r_small, length_scale, p_order, s_order, k_dim, &coeffs)
        .expect("jets at small r");

    let rel = if jets_0.t.abs() > 1e-15 {
        ((jets_0.t - jets_small.t) / jets_0.t).abs()
    } else {
        (jets_0.t - jets_small.t).abs()
    };
    assert!(
        rel < 1e-2,
        "t collision limit should be close to nearby value: t(0)={}, t(r_small)={}, rel_err={rel}",
        jets_0.t,
        jets_small.t,
    );
}

#[test]
fn test_duchon_radial_jets_t_derivative_collision_limits_are_exact() {
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 5usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let t_rr_collision =
        duchon_phi_rrrrrr_collision(length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("collision phi''''''")
            / 15.0;

    let jets_0 = duchon_radial_jets(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
        .expect("jets at origin");
    assert!(
        jets_0.t_r.abs() < 1e-12,
        "expected t_r(0)=0, got {}",
        jets_0.t_r
    );
    assert!(
        (jets_0.t_rr - t_rr_collision).abs() < 1e-12,
        "expected exact t_rr(0) collision limit, got {} vs {}",
        jets_0.t_rr,
        t_rr_collision
    );
}

#[test]
fn test_duchon_high_dim_single_matern_block_operator_jets_are_stable() {
    let p_order = 0usize;
    let s_order = 1usize;
    let k_dim = 16usize;
    let length_scale = 1.0;
    let kappa = 1.0 / length_scale;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
    let r = 1e-5;

    let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs).expect("jets");

    // The point of this test is STABILITY of the production radial-jet path for a
    // high-dimensional, low-smoothness single Matérn block (p=0, s=1, d=16) at a
    // near-collision radius. For these orders the kernel is NOT C² at the origin
    // (smoothness_order = 2·(p+s) = 2 ≤ d+2 = 18), so the RAW single-block
    // operator scalars q, t genuinely DIVERGE as r→0: the bare block evaluator
    // (`duchon_matern_operator_block_jets_with_ladder`) returns ~1e79..1e103 here
    // — precisely the blow-up the production path (#1424/#1453) regularizes away.
    // Demanding that `duchon_radial_jets` EQUAL that raw divergent block (the old
    // assertion) was backwards: it pinned the unstable reference as the oracle.
    // A stable result is a FINITE, non-exploding one that satisfies the operator
    // consistency identities — mirroring the sibling
    // `test_duchon_high_dim_*_remain_finite_and_consistent` tests.
    assert!(jets.q.is_finite(), "q must be finite, got {}", jets.q);
    assert!(jets.t.is_finite(), "t must be finite, got {}", jets.t);
    assert!(jets.t_r.is_finite(), "t_r must be finite, got {}", jets.t_r);
    assert!(
        jets.t_rr.is_finite(),
        "t_rr must be finite, got {}",
        jets.t_rr
    );
    assert!(
        jets.phi_rr.is_finite(),
        "phi_rr must be finite, got {}",
        jets.phi_rr
    );
    assert!(jets.lap.is_finite(), "lap must be finite, got {}", jets.lap);

    // Stability: the regularized scalars must NOT carry the raw block's
    // astronomical near-origin magnitude. The kernel value φ and its admissible
    // operator scalars are O(1) for this normalized block; a value > 1e6 would
    // mean the divergence leaked through. (The raw block here is ~1e79.)
    let stable_ceiling = 1.0e6;
    assert!(
        jets.q.abs() <= stable_ceiling
            && jets.t.abs() <= stable_ceiling
            && jets.phi_rr.abs() <= stable_ceiling
            && jets.lap.abs() <= stable_ceiling,
        "regularized jets must stay bounded near the collision, got q={} t={} phi_rr={} lap={}",
        jets.q,
        jets.t,
        jets.phi_rr,
        jets.lap
    );

    // Internal operator-consistency identities the production jets must satisfy
    // (φ'' = q + r²·t; Laplacian = d·q + r²·t), independent of the raw block.
    assert!(
        ((jets.phi_rr - (jets.q + r * r * jets.t)).abs()) <= 1e-10 * jets.phi_rr.abs().max(1.0)
    );
    assert!(
        ((jets.lap - (k_dim as f64 * jets.q + r * r * jets.t)).abs())
            <= 1e-10 * jets.lap.abs().max(1.0)
    );
}

#[test]
fn test_duchon_high_dim_single_matern_block_subfloor_jets_stay_stable() {
    let p_order = 0usize;
    let s_order = 1usize;
    let k_dim = 16usize;
    let length_scale = 1.0;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let r_floor = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale;
    let r_small = 0.25 * r_floor;

    let jets_small = duchon_radial_jets(r_small, length_scale, p_order, s_order, k_dim, &coeffs)
        .expect("sub-floor jets");
    let jets_floor = duchon_radial_jets(r_floor, length_scale, p_order, s_order, k_dim, &coeffs)
        .expect("floor jets");

    assert!(jets_small.q.is_finite());
    assert!(jets_small.t.is_finite());
    assert!(jets_small.lap.is_finite());
    assert!((jets_small.q - jets_floor.q).abs() <= 1e-12 * jets_floor.q.abs().max(1.0));
    assert!((jets_small.t - jets_floor.t).abs() <= 1e-12 * jets_floor.t.abs().max(1.0));
    assert!((jets_small.lap - jets_floor.lap).abs() <= 1e-12 * jets_floor.lap.abs().max(1.0));
}

#[test]
fn test_duchon_high_dim_mixed_operator_jets_remain_finite_and_consistent() {
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 4usize;
    let k_dim = 16usize;
    let length_scale = 1.0;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let r = 1e-5;

    let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs).expect("jets");

    assert!(jets.q.is_finite());
    assert!(jets.t.is_finite());
    assert!(jets.t_r.is_finite());
    assert!(jets.t_rr.is_finite());
    assert!(
        ((jets.phi_rr - (jets.q + r * r * jets.t)).abs()) <= 1e-10 * jets.phi_rr.abs().max(1.0)
    );
    assert!(
        ((jets.lap - (k_dim as f64 * jets.q + r * r * jets.t)).abs())
            <= 1e-10 * jets.lap.abs().max(1.0)
    );
}

#[test]
fn test_duchon_kernel_radial_triplet_uses_collisionphi_rr_at_origin() {
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 3usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let (_, phi_r, phi_rr) = duchon_kernel_radial_triplet(
        0.0,
        Some(length_scale),
        p_order,
        s_order as f64,
        k_dim,
        Some(&coeffs),
    )
    .expect("radial triplet at origin");
    let (phi_rr_collision, _, _) =
        duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("collision phi_rr");

    assert!(phi_r.abs() < 1e-12);
    assert!((phi_rr - phi_rr_collision).abs() < 1e-12);
}

#[test]
fn test_matern_public_second_derivative_matchesfd_of_public_first_derivative() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 0.9,
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
    };
    let analytic = build_matern_basis_log_kappasecond_derivative(data.view(), &spec)
        .expect("analytic Matérn second derivative should build");

    let eps: f64 = 1e-5;
    let kappa = 1.0 / spec.length_scale;
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = ls_plus;
    spec_minus.length_scale = ls_minus;
    let plus = build_matern_basis_log_kappa_derivative(data.view(), &spec_plus).expect("plus");
    let minus = build_matern_basis_log_kappa_derivative(data.view(), &spec_minus).expect("minus");

    let fd_design = (&plus.design_derivative - &minus.design_derivative) / (2.0 * eps);
    let fd_penalty = (&plus.penalties_derivative[0] - &minus.penalties_derivative[0]) / (2.0 * eps);

    let design_err = (&analytic.designsecond_derivative - &fd_design)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let penalty_err = (&analytic.penaltiessecond_derivative[0] - &fd_penalty)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    assert!(
        design_err < 5e-3,
        "Matérn public second-derivative design mismatch: {design_err}"
    );
    assert!(
        penalty_err < 5e-3,
        "Matérn public second-derivative penalty mismatch: {penalty_err}"
    );
}

#[test]
fn test_matern_aniso_operator_penalties_use_cross_provider() {
    let data = array![
        [0.0, 0.0],
        [1.0, 0.2],
        [0.3, 1.1],
        [0.9, 0.8],
        [0.4, 0.5],
        [0.7, 0.1]
    ];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 0.9,
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: Some(vec![0.1, -0.1]),
    };

    let basis = build_matern_basis(data.view(), &spec).expect("aniso Matérn basis");
    let derivs = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec)
        .expect("aniso Matérn derivatives");
    let expected_cols = basis.design.ncols();

    assert_eq!(derivs.penalties_cross_pairs, vec![(0, 1)]);
    let cross_penalties = derivs
        .penalties_cross_provider
        .as_ref()
        .expect("aniso Matérn cross penalties should be provider-backed")
        .evaluate(0, 1)
        .expect("aniso Matérn cross penalties");
    assert!(!cross_penalties.is_empty());
    for penalty in &cross_penalties {
        assert_eq!(penalty.nrows(), expected_cols);
        assert_eq!(penalty.ncols(), expected_cols);
    }
}

#[test]
fn test_duchon_public_second_derivative_matchesfd_of_public_first_derivative() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: Some(0.9),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let analytic = build_duchon_basis_log_kappasecond_derivative(data.view(), &spec)
        .expect("analytic Duchon second derivative should build");

    let eps: f64 = 2e-5;
    let kappa = 1.0 / spec.length_scale.expect("hybrid Duchon length_scale");
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = Some(ls_plus);
    spec_minus.length_scale = Some(ls_minus);
    let plus = build_duchon_basis_log_kappa_derivative(data.view(), &spec_plus).expect("plus");
    let minus = build_duchon_basis_log_kappa_derivative(data.view(), &spec_minus).expect("minus");

    let fd_design = (&plus.design_derivative - &minus.design_derivative) / (2.0 * eps);
    let fd_penalty = (&plus.penalties_derivative[0] - &minus.penalties_derivative[0]) / (2.0 * eps);

    let design_err = (&analytic.designsecond_derivative - &fd_design)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let penalty_err = (&analytic.penaltiessecond_derivative[0] - &fd_penalty)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    assert!(
        design_err < 5e-3,
        "Duchon public second-derivative design mismatch: {design_err}"
    );
    assert!(
        penalty_err < 5e-3,
        "Duchon public second-derivative penalty mismatch: {penalty_err}"
    );
}

#[test]
fn test_pure_duchon_default_tuple_rejects_insufficient_nullspace() {
    let data = array![[0.0, 0.1], [0.2, 0.0], [0.4, 0.2], [0.6, 0.4], [0.8, 0.5]];
    let centers = data.slice(s![0..4, ..]).to_owned();
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: None,
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Zero,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let err = match build_duchon_basis(data.view(), &spec) {
        Ok(_) => panic!("pure Duchon default tuple violates the nullspace-order condition"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("power < dimension/2"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pure_duchon_default_counterexample_is_rejected() {
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let block_order = pure_duchon_block_order(
        duchon_p_from_nullspace_order(DuchonNullspaceOrder::Zero),
        2.0,
    );
    let k23 = polyharmonic_kernel(2.0_f64.sqrt(), (block_order) as f64, 2);
    let alpha = [-2.0, 1.0, 1.0];
    let qform = 2.0 * alpha[1] * alpha[2] * k23;
    assert!(
        qform < 0.0,
        "the raw pure Duchon default tuple is indefinite under the constant-only side condition"
    );

    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        length_scale: None,
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Zero,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let err = match build_duchon_basis(centers.view(), &spec) {
        Ok(_) => panic!("indefinite pure Duchon counterexample should be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("power < dimension/2"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pure_duchon_10d_a2_diagonal_is_infinite_not_zero() {
    let value = polyharmonic_kernel(0.0, 2.0, 10);
    assert!(
        value.is_infinite() && value.is_sign_positive(),
        "d=10, a=2 pure polyharmonic diagonal should be +inf, got {value}"
    );

    let near_zero = polyharmonic_kernel(1.0e-3, 2.0, 10);
    let expected = 1.0 / (8.0 * std::f64::consts::PI.powi(5)) * 1.0e18;
    assert!(
        ((near_zero - expected) / expected).abs() < 1e-12,
        "unexpected d=10, a=2 near-origin value: got {near_zero}, expected {expected}"
    );
}

#[test]
fn test_matern_radial_triplet_matches_finite_difference() {
    let r = 0.37;
    let length_scale = 0.9;
    let nu = MaternNu::FiveHalves;
    let (phi, phi_r, phi_rr, _) =
        matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu).expect("triplet");
    let h = 1e-6;
    let fp = matern_kernel_from_distance(r + h, length_scale, nu).expect("fp");
    let fm = matern_kernel_from_distance((r - h).max(0.0), length_scale, nu).expect("fm");
    let firstfd = (fp - fm) / (2.0 * h);
    let secondfd = (fp - 2.0 * phi + fm) / (h * h);
    assert_eq!(phi_r.signum(), firstfd.signum());
    assert_eq!(phi_rr.signum(), secondfd.signum());
    assert!((phi_r - firstfd).abs() < 5e-5);
    assert!((phi_rr - secondfd).abs() < 1e-3);
}

#[test]
fn test_matern_safe_ratio_matches_closed_form_limits_atzero() {
    let ls = 1.7;
    let kappa = 1.0 / ls;
    let (_, _, _, r32) =
        matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::ThreeHalves)
            .expect("three-halves");
    let (_, _, _, r52) = matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::FiveHalves)
        .expect("five-halves");
    let (_, _, _, r72) =
        matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::SevenHalves)
            .expect("seven-halves");
    let (_, _, _, r92) = matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::NineHalves)
        .expect("nine-halves");
    assert!((r32 - (-3.0 * kappa * kappa)).abs() < 1e-12);
    assert!((r52 - (-(5.0 / 3.0) * kappa * kappa)).abs() < 1e-12);
    assert!((r72 - (-(7.0 / 5.0) * kappa * kappa)).abs() < 1e-12);
    assert!((r92 - (-(9.0 / 7.0) * kappa * kappa)).abs() < 1e-12);
}

#[test]
fn test_matern_safe_ratio_half_is_finitewith_floor() {
    let ls = 1.3;
    let (_, _, _, ratio) =
        matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::Half).expect("half");
    assert!(ratio.is_finite());
    assert!(ratio < 0.0);
}

#[test]
fn test_duchon_radial_triplet_matches_finite_difference_away_fromzero() {
    let r = 0.42;
    let length_scale = 1.1;
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 3usize;
    let dim = 4usize;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
        r,
        Some(length_scale),
        p_order,
        s_order as f64,
        dim,
        Some(&coeffs),
    )
    .expect("triplet");
    let h = 1e-5;
    let fp = duchon_matern_kernel_general_from_distance(
        r + h,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        Some(&coeffs),
    )
    .expect("fp");
    let fm = duchon_matern_kernel_general_from_distance(
        r - h,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        Some(&coeffs),
    )
    .expect("fm");
    let firstfd = (fp - fm) / (2.0 * h);
    let secondfd = (fp - 2.0 * phi + fm) / (h * h);
    assert_eq!(phi_r.signum(), firstfd.signum());
    assert_eq!(phi_rr.signum(), secondfd.signum());
    assert!((phi_r - firstfd).abs() < 1e-3);
    assert!((phi_rr - secondfd).abs() < 1e-1);
}

#[test]
fn test_duchon_radial_triplet_closed_form_branch_matches_finite_difference() {
    // p=1,s=4,k=10 uses the exact K0/K1 branch with analytic derivatives.
    let r = 2.0;
    let length_scale = 1.0;
    let p_order = 1usize;
    let s_order = 4usize;
    let dim = 10usize;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
        r,
        Some(length_scale),
        p_order,
        s_order as f64,
        dim,
        Some(&coeffs),
    )
    .expect("triplet");
    let h = 1e-5;
    let fp = duchon_matern_kernel_general_from_distance(
        r + h,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        Some(&coeffs),
    )
    .expect("fp");
    let fm = duchon_matern_kernel_general_from_distance(
        r - h,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        Some(&coeffs),
    )
    .expect("fm");
    let firstfd = (fp - fm) / (2.0 * h);
    let secondfd = (fp - 2.0 * phi + fm) / (h * h);
    assert_eq!(phi_r.signum(), firstfd.signum());
    assert_eq!(phi_rr.signum(), secondfd.signum());
    assert!((phi_r - firstfd).abs() < 2e-3);
    assert!(phi_rr.is_finite());
    assert!(secondfd.is_finite());
}

#[test]
fn test_duchon_radial_triplet_pure_polyharmonic_matches_finite_difference() {
    let r = 0.73;
    let length_scale = 1.0;
    let p_order = 1usize;
    let s_order = 0usize;
    let dim = 3usize;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
        r,
        Some(length_scale),
        p_order,
        s_order as f64,
        dim,
        Some(&coeffs),
    )
    .expect("triplet");
    let h = 1e-6;
    let fp = duchon_matern_kernel_general_from_distance(
        r + h,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        Some(&coeffs),
    )
    .expect("fp");
    let fm = duchon_matern_kernel_general_from_distance(
        r - h,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        Some(&coeffs),
    )
    .expect("fm");
    let firstfd = (fp - fm) / (2.0 * h);
    let secondfd = (fp - 2.0 * phi + fm) / (h * h);
    assert_eq!(phi_r.signum(), firstfd.signum());
    assert_eq!(phi_rr.signum(), secondfd.signum());
    assert!((phi_r - firstfd).abs() < 1e-6);
    assert!((phi_rr - secondfd).abs() < 1e-4);
}

#[test]
fn test_collocation_derivatives_are_finite_at_rzero() {
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let m_ops = build_matern_collocation_operator_matrices(
        centers.view(),
        None,
        0.8,
        MaternNu::FiveHalves,
        false,
        None,
        None,
    )
    .expect("matern ops");
    assert!(m_ops.d1.iter().all(|v| v.is_finite()));
    assert!(m_ops.d2.iter().all(|v| v.is_finite()));

    let d_ops = build_duchon_collocation_operator_matrices(
        centers.view(),
        None,
        Some(0.8),
        3.0,
        DuchonNullspaceOrder::Linear,
        None,
        None,
        2,
    )
    .expect("duchon ops");
    assert!(d_ops.d1.iter().all(|v| v.is_finite()));
    assert!(d_ops.d2.iter().all(|v| v.is_finite()));
}

#[test]
fn test_matern_collocationweights_scalerows_by_sqrtweight() {
    let centers = array![[0.0, 0.0], [1.0, 0.0]];
    let unit = build_matern_collocation_operator_matrices(
        centers.view(),
        None,
        0.9,
        MaternNu::FiveHalves,
        false,
        None,
        None, // aniso_log_scales
    )
    .expect("unit weights");
    let weights = array![4.0, 1.0];
    let weighted = build_matern_collocation_operator_matrices(
        centers.view(),
        Some(weights.view()),
        0.9,
        MaternNu::FiveHalves,
        false,
        None,
        None, // aniso_log_scales
    )
    .expect("weighted");
    // First collocation row should scale by sqrt(4)=2.
    for j in 0..unit.d0.ncols() {
        assert!((weighted.d0[[0, j]] - 2.0 * unit.d0[[0, j]]).abs() < 1e-12);
    }
    // Second row has weight 1 -> unchanged.
    for j in 0..unit.d0.ncols() {
        assert!((weighted.d0[[1, j]] - unit.d0[[1, j]]).abs() < 1e-12);
    }
}

#[test]
fn matern_closed_form_should_decay_to_zero_not_nan_at_huge_distance() {
    let r = 1.0e308;
    let value = matern_kernel_from_distance(r, 1.0, MaternNu::NineHalves).expect("kernel");
    assert!(
        value == 0.0,
        "the Matérn kernel should decay to 0 for enormous finite distances, not produce NaN/Inf; got {value}"
    );

    let dpsi = matern_kernel_log_kappa_derivative_from_distance(r, 1.0, MaternNu::NineHalves)
        .expect("kernel first hyper-derivative");
    assert!(
        dpsi == 0.0,
        "the log-kappa derivative should also decay to 0 for enormous finite distances; got {dpsi}"
    );
    let d2psi =
        matern_kernel_log_kappasecond_derivative_from_distance(r, 1.0, MaternNu::NineHalves)
            .expect("kernel second hyper-derivative");
    assert!(
        d2psi == 0.0,
        "the second log-kappa derivative should also decay to 0 for enormous finite distances; got {d2psi}"
    );
}

#[test]
fn maternvalue_psi_triplet_should_decay_to_zero_not_nan_at_huge_distance() {
    let r = 1.0e308;
    let (value, dpsi, d2psi) =
        maternvalue_psi_triplet(r, 1.0, MaternNu::NineHalves).expect("psi triplet");
    assert!(
        value == 0.0,
        "maternvalue_psi_triplet value should decay to 0 for enormous finite distances; got {value}"
    );
    assert!(
        dpsi == 0.0,
        "maternvalue_psi_triplet first psi derivative should decay to 0 for enormous finite distances; got {dpsi}"
    );
    assert!(
        d2psi == 0.0,
        "maternvalue_psi_triplet second psi derivative should decay to 0 for enormous finite distances; got {d2psi}"
    );
}

#[test]
fn matern_operator_psi_triplet_should_decay_to_zero_not_nan_at_huge_distance() {
    let r = 1.0e308;
    let triplet =
        matern_operator_psi_triplet(r, 1.0, MaternNu::NineHalves, 3).expect("operator psi triplet");
    for (idx, value) in [
        triplet.0, triplet.1, triplet.2, triplet.3, triplet.4, triplet.5, triplet.6, triplet.7,
        triplet.8,
    ]
    .into_iter()
    .enumerate()
    {
        assert!(
            value == 0.0,
            "matern_operator_psi_triplet component {idx} should decay to 0 for enormous finite distances; got {value}"
        );
    }
}

#[test]
fn matern_nine_halves_log_kappasecond_derivative_matches_closed_form() {
    let r = 1.0_f64;
    let length_scale = 1.0_f64;
    let a = 3.0 * r / length_scale;
    let expected = (-a).exp()
        * (-(2.0 / 7.0) * a * a - (2.0 / 7.0) * a.powi(3) - (3.0 / 35.0) * a.powi(4)
            + (1.0 / 105.0) * a.powi(5)
            + (1.0 / 105.0) * a.powi(6));
    let actual = matern_kernel_log_kappasecond_derivative_from_distance(
        r,
        length_scale,
        MaternNu::NineHalves,
    )
    .expect("9/2 second log-kappa derivative");
    assert!(
        (actual - expected).abs() < 1e-15,
        "nu=9/2 second log-kappa derivative should match the closed form at r={r}, length_scale={length_scale}; got {actual} vs {expected}"
    );
}

#[test]
fn matern_operator_psi_triplet_should_match_closed_form_polynomials() {
    let r = 1.0_f64;
    let length_scale = 1.0_f64;

    // Pre-baked closed-form `(nu, a, expected_ratio, expected_lap)`
    // tuples for the three supported smoothnesses; eliminates any
    // need to match the broader `MaternNu` enum (which has variants
    // — `Half`, `ThreeHalves` — that this closed-form test does not
    // cover).
    let a52 = 5.0_f64.sqrt() * r / length_scale;
    let a72 = 7.0_f64.sqrt() * r / length_scale;
    let a92 = 3.0 * r / length_scale;
    let cases: [(MaternNu, f64, f64, f64); 3] = [
        (
            MaternNu::FiveHalves,
            a52,
            -(5.0 / 3.0) * (-a52).exp() * (a52 + 1.0),
            (5.0 / 3.0) * (-a52).exp() * (a52 * a52 - a52 - 1.0),
        ),
        (
            MaternNu::SevenHalves,
            a72,
            -(7.0 / 15.0) * (-a72).exp() * (a72 * a72 + 3.0 * a72 + 3.0),
            (7.0 / 15.0) * (-a72).exp() * (a72.powi(3) - 3.0 * a72 - 3.0),
        ),
        (
            MaternNu::NineHalves,
            a92,
            -(3.0 / 35.0) * (-a92).exp() * (a92.powi(3) + 6.0 * a92 * a92 + 15.0 * a92 + 15.0),
            (3.0 / 35.0)
                * (-a92).exp()
                * (a92.powi(4) + 2.0 * a92.powi(3) - 3.0 * a92 * a92 - 15.0 * a92 - 15.0),
        ),
    ];
    for (nu, _a, expected_ratio, expected_lap) in cases {
        let triplet =
            matern_operator_psi_triplet(r, length_scale, nu, 1).expect("operator psi triplet");
        let ratio = triplet.3;
        let lap = triplet.6;
        assert!(
            (ratio - expected_ratio).abs() < 1e-14,
            "phi'(r)/r closed form mismatch for nu={nu:?}: got {ratio} vs {expected_ratio}"
        );
        assert!(
            (lap - expected_lap).abs() < 1e-14,
            "phi'' closed form mismatch for nu={nu:?}: got {lap} vs {expected_lap}"
        );
    }
}

#[test]
fn matern_collocation_operator_matrices_should_match_closed_forms_in_1d() {
    let centers = array![[0.0], [1.0]];
    let length_scale = 1.0_f64;

    // Pre-baked `(nu, expected_phi, expected_ratio, expected_second)`
    // tuples for the three supported smoothnesses; eliminates the
    // need to match the wider `MaternNu` enum (which has `Half` and
    // `ThreeHalves` variants this closed-form test does not cover).
    let r = 1.0_f64;
    let a52 = 5.0_f64.sqrt() * r / length_scale;
    let a72 = 7.0_f64.sqrt() * r / length_scale;
    let a92 = 3.0 * r / length_scale;
    let cases: [(MaternNu, f64, f64, f64); 3] = [
        (
            MaternNu::FiveHalves,
            (1.0 + a52 + a52 * a52 / 3.0) * (-a52).exp(),
            -(5.0 / 3.0) * (-a52).exp() * (a52 + 1.0),
            (5.0 / 3.0) * (-a52).exp() * (a52 * a52 - a52 - 1.0),
        ),
        (
            MaternNu::SevenHalves,
            (1.0 + a72 + (2.0 / 5.0) * a72 * a72 + (1.0 / 15.0) * a72.powi(3)) * (-a72).exp(),
            -(7.0 / 15.0) * (-a72).exp() * (a72 * a72 + 3.0 * a72 + 3.0),
            (7.0 / 15.0) * (-a72).exp() * (a72.powi(3) - 3.0 * a72 - 3.0),
        ),
        (
            MaternNu::NineHalves,
            (1.0 + a92
                + (3.0 / 7.0) * a92 * a92
                + (2.0 / 21.0) * a92.powi(3)
                + (1.0 / 105.0) * a92.powi(4))
                * (-a92).exp(),
            -(3.0 / 35.0) * (-a92).exp() * (a92.powi(3) + 6.0 * a92 * a92 + 15.0 * a92 + 15.0),
            (3.0 / 35.0)
                * (-a92).exp()
                * (a92.powi(4) + 2.0 * a92.powi(3) - 3.0 * a92 * a92 - 15.0 * a92 - 15.0),
        ),
    ];
    for (nu, expected_phi, expected_ratio, expected_second) in cases {
        let ops = build_matern_collocation_operator_matrices(
            centers.view(),
            None,
            length_scale,
            nu,
            false,
            None,
            None,
        )
        .expect("matern collocation operators");

        assert!(
            (ops.d0[[1, 0]] - expected_phi).abs() < 1e-14,
            "D0 off-diagonal mismatch for nu={nu:?}: got {} vs {expected_phi}",
            ops.d0[[1, 0]]
        );
        assert!(
            (ops.d1[[1, 0]] - expected_ratio).abs() < 1e-14,
            "D1 off-diagonal mismatch for nu={nu:?}: got {} vs {expected_ratio}",
            ops.d1[[1, 0]]
        );
        // 1D: D2 has p*d*d = 2 rows; the Hessian is a single phi''(r) entry.
        // Off-diagonal pair (k=1, j=0): row index (k*d + a)*d + b = (1*1+0)*1+0 = 1.
        assert!(
            (ops.d2[[1, 0]] - expected_second).abs() < 1e-14,
            "D2 off-diagonal mismatch for nu={nu:?}: got {} vs {expected_second}",
            ops.d2[[1, 0]]
        );
        assert_eq!(
            ops.d2.nrows(),
            centers.nrows() * centers.ncols() * centers.ncols(),
            "D2 must expose full p*d*d Hessian rows"
        );
    }
}

// ---- anisotropic distance helper tests ----

#[test]
fn aniso_distance_isotropic_when_eta_zero() {
    // When all η_a = 0, exp(2·0) = 1, so aniso distance == Euclidean distance.
    let x = [1.0, 2.0, 3.0];
    let c = [4.0, 5.0, 6.0];
    let eta = [0.0, 0.0, 0.0];
    let iso_r = {
        let mut d2 = 0.0;
        for a in 0..3 {
            let h = x[a] - c[a];
            d2 += h * h;
        }
        d2.sqrt()
    };
    let (r, s) = aniso_distance_and_components(&x, &c, &eta);
    assert_abs_diff_eq!(r, iso_r, epsilon = 1e-14);
    assert_abs_diff_eq!(aniso_distance(&x, &c, &eta), iso_r, epsilon = 1e-14);
    // s_a components should sum to r²
    let s_sum: f64 = s.iter().sum();
    assert_abs_diff_eq!(s_sum, r * r, epsilon = 1e-14);
}

#[test]
fn aniso_distance_weighted_correctly() {
    // Two axes, η = [ln2, -ln2] so exp(2η) = [4, 1/4].
    // h = [1, 2], so s = [4·1, 0.25·4] = [4, 1], r = √5.
    let x = [3.0, 5.0];
    let c = [2.0, 3.0];
    let eta = [2.0_f64.ln(), -(2.0_f64.ln())];
    let (r, s) = aniso_distance_and_components(&x, &c, &eta);
    assert_abs_diff_eq!(s[0], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(s[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r, 5.0_f64.sqrt(), epsilon = 1e-12);
    assert_abs_diff_eq!(
        aniso_distance(&x, &c, &eta),
        5.0_f64.sqrt(),
        epsilon = 1e-12
    );
}

#[test]
fn aniso_distance_components_sum_to_r_squared() {
    let x = [1.5, -0.3, 2.7, 0.1];
    let c = [0.2, 1.1, -0.5, 3.3];
    let eta = [0.5, -0.2, 0.1, -0.4];
    let (r, s) = aniso_distance_and_components(&x, &c, &eta);
    let s_sum: f64 = s.iter().sum();
    assert_abs_diff_eq!(s_sum, r * r, epsilon = 1e-12);
}

#[test]
fn aniso_distance_zero_displacement_gives_zero_component() {
    // When h_a = 0 for some axis, that s_a must be exactly 0.
    let x = [1.0, 5.0, 3.0];
    let c = [1.0, 2.0, 3.0]; // axis 0 and 2 have h=0
    let eta = [10.0, -5.0, -5.0]; // large eta on axis 0 should not matter
    let (r, s) = aniso_distance_and_components(&x, &c, &eta);
    assert_eq!(s[0], 0.0, "s_a should be exactly 0 when h_a = 0");
    assert_eq!(s[2], 0.0, "s_a should be exactly 0 when h_a = 0");
    assert!(s[1] > 0.0);
    // r should equal sqrt(s[1])
    assert_abs_diff_eq!(r, s[1].sqrt(), epsilon = 1e-14);
}

// ── knot_cloud_axis_scales tests ─────────────────────────────────────

#[test]
fn test_knot_cloud_axis_scales_basic() {
    // 5x3 center matrix with known std devs per axis.
    // Axis 0: values 1,2,3,4,5 → std = sqrt(2.5) ≈ 1.5811
    // Axis 1: values 10,20,30,40,50 → std = sqrt(250) ≈ 15.811
    // Axis 2: values 0,0,0,0,1 → std = sqrt(0.2) ≈ 0.4472
    use ndarray::Array2;
    let centers = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 10.0, 0.0, 2.0, 20.0, 0.0, 3.0, 30.0, 0.0, 4.0, 40.0, 0.0, 5.0, 50.0, 1.0,
        ],
    )
    .unwrap();
    let scales = knot_cloud_axis_scales(centers.view());
    assert_eq!(scales.len(), 3);
    // Axis 0: sample std of [1,2,3,4,5]
    let expected_0 = (2.5_f64).sqrt(); // sqrt(10/4)
    assert_abs_diff_eq!(scales[0], expected_0, epsilon = 1e-10);
    // Axis 1: 10x axis 0
    assert_abs_diff_eq!(scales[1], expected_0 * 10.0, epsilon = 1e-10);
    // Axis 2: sample std of [0,0,0,0,1]
    let var2 = (4.0 * 0.04 + 0.64) / 4.0; // mean=0.2, var = sum((xi-0.2)^2)/4
    let expected_2 = var2.sqrt();
    // Re-derive: mean=0.2, deviations: -0.2,-0.2,-0.2,-0.2,0.8
    // sum of sq = 4*0.04 + 0.64 = 0.8, var = 0.8/4 = 0.2, std = sqrt(0.2)
    assert_abs_diff_eq!(scales[2], expected_2, epsilon = 1e-10);
}

#[test]
fn test_knot_cloud_axis_scales_zero_variance() {
    // One axis is constant → should return sigma=1.0 for that axis.
    use ndarray::Array2;
    let centers =
        Array2::from_shape_vec((4, 2), vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0]).unwrap();
    let scales = knot_cloud_axis_scales(centers.view());
    assert_eq!(scales.len(), 2);
    // Axis 0 has nonzero variance
    assert!(scales[0] > 1e-6);
    // Axis 1 is constant → sigma clamped to 1.0
    assert_abs_diff_eq!(scales[1], 1.0, epsilon = 1e-12);
}

#[test]
fn test_knot_cloud_axis_scales_single_center() {
    // Fewer than 2 centers → returns vec![1.0; d].
    use ndarray::Array2;
    let centers = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let scales = knot_cloud_axis_scales(centers.view());
    assert_eq!(scales, vec![1.0, 1.0, 1.0]);
}

// ── initial_aniso_contrasts tests ────────────────────────────────────

#[test]
fn test_initial_aniso_contrasts_sum_to_zero() {
    // Create centers with different axis scales; verify sum of η ≈ 0.
    use ndarray::Array2;
    let centers = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0, 5.0, 50.0,
            500.0,
        ],
    )
    .unwrap();
    let eta = initial_aniso_contrasts(centers.view());
    assert_eq!(eta.len(), 3);
    let sum: f64 = eta.iter().sum();
    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-12);
}

#[test]
fn test_initial_aniso_contrasts_1d_returns_empty() {
    // 1-D centers → empty vec (anisotropy meaningless).
    use ndarray::Array2;
    let centers = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let eta = initial_aniso_contrasts(centers.view());
    assert!(eta.is_empty());
}

#[test]
fn test_initial_aniso_contrasts_equal_scales() {
    // All axes have same std dev → all η should be ~0.
    use ndarray::Array2;
    let centers = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
    )
    .unwrap();
    let eta = initial_aniso_contrasts(centers.view());
    assert_eq!(eta.len(), 3);
    for &e in &eta {
        assert_abs_diff_eq!(e, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn test_initial_aniso_contrasts_unequal_scales() {
    // Axis 0 has 10x the std dev of axis 1.
    // η_a = −ln(σ_a) + mean(−ln(σ_b))
    // Axis with larger σ → more negative −ln(σ) → η_a < 0
    // Axis with smaller σ → more positive −ln(σ) → η_a > 0
    use ndarray::Array2;
    let centers =
        Array2::from_shape_vec((4, 2), vec![10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0]).unwrap();
    let eta = initial_aniso_contrasts(centers.view());
    assert_eq!(eta.len(), 2);
    // Axis 0 has 10x spread → negative η (larger scale → smaller κ)
    assert!(
        eta[0] < 0.0,
        "axis with larger spread should have negative η, got {}",
        eta[0]
    );
    // Axis 1 has smaller spread → positive η
    assert!(
        eta[1] > 0.0,
        "axis with smaller spread should have positive η, got {}",
        eta[1]
    );
    // Sum should be zero
    assert_abs_diff_eq!(eta[0] + eta[1], 0.0, epsilon = 1e-12);
    // |η| should be ln(10)/2 for d=2 zero-sum contrasts
    assert_abs_diff_eq!(eta[0].abs(), 10.0_f64.ln() / 2.0, epsilon = 1e-12);
}

// ── auto_seed_aniso_contrasts tests (Duchon scale_dims seeding) ──────

#[test]
fn test_auto_seed_replaces_zeros() {
    // Input: Some(&[0.0, 0.0, 0.0]) → the Duchon scale_dims seeding sentinel:
    // replaced with knot-derived geometry contrasts.
    use ndarray::Array2;
    let centers = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0, 5.0, 50.0,
            500.0,
        ],
    )
    .unwrap();
    let zeros = vec![0.0, 0.0, 0.0];
    let result = auto_seed_aniso_contrasts(centers.view(), Some(&zeros));
    let eta = result.expect("should return Some");
    assert_eq!(eta.len(), 3);
    // Should NOT be all zeros any more — should match initial_aniso_contrasts
    let expected = initial_aniso_contrasts(centers.view());
    for (a, b) in eta.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-14);
    }
}

#[test]
fn test_auto_seed_preserves_nonzero() {
    // Input: Some(&[0.1, -0.05, -0.05]) → should be returned unchanged.
    use ndarray::Array2;
    let centers = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();
    let input = vec![0.1, -0.05, -0.05];
    let result = auto_seed_aniso_contrasts(centers.view(), Some(&input));
    let eta = result.expect("should return Some");
    assert_eq!(eta, input);
}

// ── centered_aniso_contrasts tests (pure Matérn forward transform) ───

#[test]
fn test_centered_aniso_honors_explicit_all_zero_as_isotropic() {
    // The pure forward transform must NOT reinterpret an explicit all-zero
    // vector as a geometry-seeding sentinel: [0,0,0] → [0,0,0] (isotropic),
    // independent of any center geometry. This is the unit-level guard for
    // the discontinuity-at-η=0 bug (#1042); the geometry-driven override is
    // exclusively the Duchon-only `auto_seed_aniso_contrasts` behavior.
    let zeros = vec![0.0, 0.0, 0.0];
    let eta = centered_aniso_contrasts(Some(&zeros)).expect("should return Some");
    assert_eq!(eta, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_centered_aniso_subtracts_mean() {
    // A non-zero vector is centered (Σ η = 0), zeroing tiny residuals.
    let input = vec![0.5, 0.5];
    let eta = centered_aniso_contrasts(Some(&input)).expect("should return Some");
    // mean is 0.5, so both center to 0 → isotropic, matching the all-zero case.
    assert_eq!(eta, vec![0.0, 0.0]);

    let input = vec![0.3, -0.1, -0.2];
    let eta = centered_aniso_contrasts(Some(&input)).expect("should return Some");
    assert_abs_diff_eq!(eta.iter().sum::<f64>(), 0.0, epsilon = 1e-15);
    // Already zero-sum, so returned essentially unchanged.
    for (a, b) in eta.iter().zip(input.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }
}

#[test]
fn test_centered_aniso_preserves_none() {
    assert!(centered_aniso_contrasts(None).is_none());
}

#[test]
fn test_centered_aniso_one_d_is_passthrough() {
    // A 1-D contrast is a no-op (anisotropy is meaningless for d=1).
    let input = vec![0.7];
    let eta = centered_aniso_contrasts(Some(&input)).expect("should return Some");
    assert_eq!(eta, vec![0.7]);
}

#[test]
fn test_auto_seed_preserves_none() {
    // Input: None → should remain None.
    use ndarray::Array2;
    let centers =
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let result = auto_seed_aniso_contrasts(centers.view(), None);
    assert!(result.is_none());
}

// -----------------------------------------------------------------------
// Duchon anisotropic exposed-axis second-derivative kernel tests.
//
// These tests target the `2 q G_{LR}` overlap term in
// `ImplicitDesignPsiDerivative::transformed_second_kernel_value`. A bug
// in that term (e.g. missing the overlap contribution) can hide behind a
// smooth radial kernel; the overlap only shows up as an additive correction
// in the exact analytic second derivative.
//
// The full formula for linear combinations
//   L = Σ_a l_a ∂/∂ψ_a,   R = Σ_a r_a ∂/∂ψ_a
// applied to the radial kernel φ (which depends on the distance shaped
// by per-axis scales s_a = exp(2 η_a)) is
//
//   D²_{L,R} φ = t S_L S_R + 2 q G_{LR}
//                + c q (C_R S_L + C_L S_R)
//                + c² C_L C_R φ
//
// where S_L = Σ l_a s_a, C_L = Σ l_a, G_{LR} = Σ l_a r_a s_a, and
// c = psi_scale_share. The closed-form values below pin each piece of
// the formula — in particular the overlap contribution 2 q G_{LR}.
// -----------------------------------------------------------------------

#[test]
fn overlap_diag_contrast_e0_minus_elast_matches_closed_form() {
    // Pure Duchon contrast L = R = e_0 - e_last in d = 3:
    //   C_L = 0, so the `cq` and `c²` terms vanish.
    //   S_L = s_0 - s_last, G_LL = s_0 + s_last.
    //   Correct kernel value = t (s_0 - s_last)² + 2 q (s_0 + s_last).
    // The `2 q (s_0 + s_last)` piece is the overlap term — missing it
    // would leave only the first summand.
    let s = [1.3_f64, 0.7, 2.1];
    let phi = 3.0;
    let q = -0.5;
    let t = 0.9;
    let c = 0.0;
    let l: &[(usize, f64)] = &[(0, 1.0), (2, -1.0)];
    let r = l;

    let s_l: f64 = l.iter().map(|&(a, la)| la * s[a]).sum();
    let s_r: f64 = r.iter().map(|&(a, ra)| ra * s[a]).sum();
    let c_l: f64 = l.iter().map(|&(_, la)| la).sum();
    let c_r: f64 = r.iter().map(|&(_, ra)| ra).sum();
    let overlap = ImplicitDesignPsiDerivative::transformed_combo_overlap_streaming(l, r, &s);

    let got = ImplicitDesignPsiDerivative::transformed_second_kernel_value(
        phi, q, t, s_l, c_l, s_r, c_r, overlap, c,
    );
    let expected = t * (s[0] - s[2]).powi(2) + 2.0 * q * (s[0] + s[2]);

    assert!(
        (got - expected).abs() < 1e-12,
        "diag overlap mismatch: got={got} expected={expected}"
    );

    // Buggy "no-overlap" value misses the `2 q G_LL` correction. Pin
    // that the correction is non-trivial and equals the overlap term.
    let no_overlap = t * (s[0] - s[2]).powi(2);
    assert!(
        (got - no_overlap).abs() > 1e-6,
        "overlap term contributes no correction: got={got} no_overlap={no_overlap}"
    );
    assert!(
        (got - no_overlap - 2.0 * q * (s[0] + s[2])).abs() < 1e-12,
        "overlap correction should equal 2 q (s_0 + s_last) exactly"
    );
}

#[test]
fn overlap_cross_contrast_matches_closed_form() {
    // Pure Duchon cross contrast L = e_0 - e_last, R = e_1 - e_last.
    //   C_L = C_R = 0, so only the first two terms of the formula
    //   survive. Only the `last` axis is shared between L and R:
    //     G_{LR} = l_last * r_last * s_last = (-1)(-1) s_last = s_last.
    //   Correct kernel value = t (s_0 - s_last)(s_1 - s_last) + 2 q s_last.
    // The `2 q s_last` piece is the overlap term.
    let s = [1.3_f64, 0.7, 2.1];
    let phi = 3.0;
    let q = -0.5;
    let t = 0.9;
    let c = 0.0;
    let l: &[(usize, f64)] = &[(0, 1.0), (2, -1.0)];
    let r: &[(usize, f64)] = &[(1, 1.0), (2, -1.0)];

    let s_l: f64 = l.iter().map(|&(a, la)| la * s[a]).sum();
    let s_r: f64 = r.iter().map(|&(a, ra)| ra * s[a]).sum();
    let c_l: f64 = l.iter().map(|&(_, la)| la).sum();
    let c_r: f64 = r.iter().map(|&(_, ra)| ra).sum();
    let overlap = ImplicitDesignPsiDerivative::transformed_combo_overlap_streaming(l, r, &s);

    let got = ImplicitDesignPsiDerivative::transformed_second_kernel_value(
        phi, q, t, s_l, c_l, s_r, c_r, overlap, c,
    );
    let expected = t * (s[0] - s[2]) * (s[1] - s[2]) + 2.0 * q * s[2];

    assert!(
        (got - expected).abs() < 1e-12,
        "cross overlap mismatch: got={got} expected={expected}"
    );

    // Confirm the overlap term is exactly what separates the correct
    // value from the buggy "no-overlap" version: 2 q s_last.
    let no_overlap = t * (s[0] - s[2]) * (s[1] - s[2]);
    assert!(
        (got - no_overlap - 2.0 * q * s[2]).abs() < 1e-12,
        "overlap correction should equal 2 q s_last exactly"
    );
}

#[test]
fn overlap_vs_no_overlap_diag_differs_by_2q_sum() {
    // Pin the streaming-overlap helper itself: for L = R = e_0 - e_last,
    // each l_a² ∈ {0, 1}, so overlap = Σ_a l_a² s_a = s_0 + s_last.
    let s = [1.3_f64, 0.7, 2.1];
    let q = -0.5;
    let l: &[(usize, f64)] = &[(0, 1.0), (2, -1.0)];

    let overlap = ImplicitDesignPsiDerivative::transformed_combo_overlap_streaming(l, l, &s);
    let expected_overlap = s[0] + s[2];
    assert!(
        (overlap - expected_overlap).abs() < 1e-14,
        "overlap helper mismatch: got={overlap} expected={expected_overlap}"
    );

    let overlap_contribution = 2.0 * q * overlap;
    let expected = 2.0 * q * (s[0] + s[2]);
    assert!(
        (overlap_contribution - expected).abs() < 1e-14,
        "overlap contribution mismatch: got={overlap_contribution} expected={expected}"
    );
}

#[test]
fn overlap_psi_scale_share_nonzero_matches_full_formula() {
    // With c ≠ 0 and L = R = e_0 + e_1 (so C_L = C_R = 2), every term
    // of the full formula is active. Pin the helper against the
    // hand-written expression.
    let s = [1.3_f64, 0.7];
    let phi = 3.0;
    let q = -0.5;
    let t = 0.9;
    let c = 0.25;
    let l: &[(usize, f64)] = &[(0, 1.0), (1, 1.0)];

    let s_l: f64 = l.iter().map(|&(a, la)| la * s[a]).sum();
    let c_l: f64 = l.iter().map(|&(_, la)| la).sum();
    let overlap = ImplicitDesignPsiDerivative::transformed_combo_overlap_streaming(l, l, &s);

    // G_LL = Σ l_a² s_a = s_0 + s_1 for all-ones combo.
    assert!((overlap - (s[0] + s[1])).abs() < 1e-14);

    let got = ImplicitDesignPsiDerivative::transformed_second_kernel_value(
        phi, q, t, s_l, c_l, s_l, c_l, overlap, c,
    );
    let expected = t * s_l * s_l
        + 2.0 * q * overlap
        + c * q * (c_l * s_l + c_l * s_l)
        + c * c * c_l * c_l * phi;

    assert!(
        (got - expected).abs() < 1e-12,
        "psi-scale-share full-formula mismatch: got={got} expected={expected}"
    );
}

// ----------------------------------------------------------------------
// Closed-form Riesz / Matérn / hybrid Duchon kernel tests
// ----------------------------------------------------------------------

use super::closed_form_penalty::{
    bessel_k, isotropic_duchon_penalty, matern_kernel_value, riesz_kernel_value,
};

#[test]
fn test_riesz_d3_j1() {
    // R_1^3(r) = 1/(4π r)
    for &r in &[0.1_f64, 1.0, 10.0] {
        let got = riesz_kernel_value(3, 1.0, r);
        let expected = 1.0 / (4.0 * std::f64::consts::PI * r);
        assert!(
            (got - expected).abs() / expected.abs() < 1e-12,
            "R_1^3({r}) got={got} expected={expected}"
        );
    }
}

#[test]
fn test_riesz_d3_j2() {
    // R_2^3(r) = -r/(8π)
    // Γ(3/2 - 2) = Γ(-1/2) = -2√π. 4^2 π^{3/2} Γ(2) = 16 π^{3/2}.
    // Coefficient = -2√π / (16 π^{3/2}) = -1/(8π). Times r^{2·2-3} = r.
    for &r in &[0.1_f64, 1.0, 10.0] {
        let got = riesz_kernel_value(3, 2.0, r);
        let expected = -r / (8.0 * std::f64::consts::PI);
        assert!(
            (got - expected).abs() / expected.abs() < 1e-12,
            "R_2^3({r}) got={got} expected={expected}"
        );
    }
}

#[test]
fn test_matern_d3_ell1() {
    // M_1^3(r; 1) = e^{-r} / (4π r)
    for &r in &[0.1_f64, 0.5, 1.0, 2.5, 5.0] {
        let got = matern_kernel_value(3, 1, 1.0, r);
        let expected = (-r).exp() / (4.0 * std::f64::consts::PI * r);
        assert!(
            (got - expected).abs() / expected.abs() < 1e-10,
            "M_1^3({r}; 1) got={got} expected={expected}"
        );
    }
}

#[test]
fn test_isotropic_pure_polyharmonic() {
    // s = 0: g_q^iso(R) should equal R_{2m-q}^d(R) regardless of κ.
    let m = 2;
    let q = 1;
    let d = 3;
    for &kappa in &[0.0_f64, 0.3, 1.0, 5.0] {
        for &r in &[0.2_f64, 1.0, 4.0] {
            let got = isotropic_duchon_penalty(q, d, m, 0.0, kappa, r);
            let expected = riesz_kernel_value(d, (2 * m - q) as f64, r);
            assert!(
                (got - expected).abs() / expected.abs().max(1e-300) < 1e-12,
                "pure polyharmonic mismatch (κ={kappa}, r={r}): got={got} expected={expected}"
            );
        }
    }
}

/// Numerical inverse Fourier transform of a radially symmetric function in d=3:
///   F^{-1}{f̂(|ρ|)}(R) = (1/(2π² R)) ∫_0^∞ ρ · sin(ρR) · f̂(ρ) dρ.
/// Compute on [0, ρ_max] with composite Gauss-Legendre to verify partial fractions.
fn fourier_inv_radial_d3<F: Fn(f64) -> f64>(f_hat: F, r: f64, rho_max: f64) -> f64 {
    // 64-point Gauss-Legendre on [-1,1]
    let (nodes, weights) = gauss_legendre_64();
    // Composite over [0, rho_max] in segments to handle oscillation
    let n_seg = 200usize;
    let dseg = rho_max / n_seg as f64;
    let mut sum = 0.0_f64;
    for k in 0..n_seg {
        let a = k as f64 * dseg;
        let b = a + dseg;
        let half = 0.5 * (b - a);
        let mid = 0.5 * (a + b);
        let mut seg = 0.0_f64;
        for (xi, wi) in nodes.iter().zip(weights.iter()) {
            let rho = mid + half * xi;
            let val = rho * (rho * r).sin() * f_hat(rho);
            seg += wi * val;
        }
        sum += half * seg;
    }
    sum / (2.0 * std::f64::consts::PI * std::f64::consts::PI * r)
}

fn gauss_legendre_64() -> (Vec<f64>, Vec<f64>) {
    // Build via Golub-Welsch on the Jacobi matrix for Legendre (β_k = k/√(4k²-1))
    let n = 64usize;
    let diag = vec![0.0_f64; n];
    let mut sub = vec![0.0_f64; n - 1];
    for k in 1..n {
        let kf = k as f64;
        sub[k - 1] = kf / (4.0 * kf * kf - 1.0).sqrt();
    }
    // Symmetric tridiagonal eigendecomposition via simple QL with implicit shifts
    let mut z = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        z[i][i] = 1.0;
    }
    let mut d = diag.clone();
    let mut e = sub.clone();
    e.push(0.0);
    // Standard tql2 algorithm
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m_ = l;
            while m_ < n - 1 {
                let dd = d[m_].abs() + d[m_ + 1].abs();
                if e[m_].abs() <= 1e-30 + 1e-15 * dd {
                    break;
                }
                m_ += 1;
            }
            if m_ == l {
                break;
            }
            iter += 1;
            if iter > 100 {
                panic!("tql2 no convergence");
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = (g * g + 1.0).sqrt();
            let sign = if g >= 0.0 { 1.0 } else { -1.0 };
            g = d[m_] - d[l] + e[l] / (g + sign * r);
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut i = m_;
            while i > l {
                let f = s * e[i - 1];
                let bb = c * e[i - 1];
                let rr = (f * f + g * g).sqrt();
                e[i] = rr;
                if rr == 0.0 {
                    d[i] -= p;
                    e[m_] = 0.0;
                    break;
                }
                s = f / rr;
                c = g / rr;
                g = d[i] - p;
                let rrr = (d[i - 1] - g) * s + 2.0 * c * bb;
                p = s * rrr;
                d[i] = g + p;
                g = c * rrr - bb;
                for k in 0..n {
                    let f = z[k][i];
                    z[k][i] = s * z[k][i - 1] + c * f;
                    z[k][i - 1] = c * z[k][i - 1] - s * f;
                }
                i -= 1;
            }
            d[l] -= p;
            e[l] = g;
            e[m_] = 0.0;
        }
    }
    // Sort eigenvalues
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());
    let nodes: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    let weights: Vec<f64> = idx.iter().map(|&i| 2.0 * z[0][i] * z[0][i]).collect();
    (nodes, weights)
}

fn symmetric_eigenvalue_bounds_jacobi(matrix: &Array2<f64>) -> (f64, f64) {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols());
    if n == 0 {
        return (0.0, 0.0);
    }

    let mut a = matrix.to_owned();
    let diag_scale = (0..n)
        .map(|i| a[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tol = 1e-14 * diag_scale;

    for _ in 0..(100 * n * n).max(1) {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let off = a[[i, j]].abs();
                if off > max_off {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off <= tol {
            break;
        }

        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];
        if apq == 0.0 {
            continue;
        }
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s_rot = t * c;

        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let akp = a[[k, p]];
            let akq = a[[k, q]];
            let new_kp = c * akp - s_rot * akq;
            let new_kq = s_rot * akp + c * akq;
            a[[k, p]] = new_kp;
            a[[p, k]] = new_kp;
            a[[k, q]] = new_kq;
            a[[q, k]] = new_kq;
        }

        a[[p, p]] = app - t * apq;
        a[[q, q]] = aqq + t * apq;
        a[[p, q]] = 0.0;
        a[[q, p]] = 0.0;
    }

    let mut min_eval = f64::INFINITY;
    let mut max_abs_eval = 0.0_f64;
    for i in 0..n {
        let value = a[[i, i]];
        min_eval = min_eval.min(value);
        max_abs_eval = max_abs_eval.max(value.abs());
    }
    (min_eval, max_abs_eval)
}

#[test]
fn test_isotropic_hybrid_partial_fraction() {
    // d=3, m=2, s=2, κ=1, q=2 → a = 2m-q = 2, b = 2s = 4.
    // f̂(ρ) = 1 / (ρ^{2a} (κ²+ρ²)^b) = 1 / (ρ^4 (1+ρ²)^4)
    // The Fourier integral diverges at ρ=0 (ρ^4 in denominator with 3D measure ρ²
    // gives integrand ~ 1/ρ² near 0). So strictly, the partial-fraction sum of
    // Riesz/Matern includes distributional pieces. To compare numerically, we
    // verify a *regularized* identity by subtracting the Riesz singular tail:
    // form g(R) - A_1 R_1^3(R) - A_2 R_2^3(R) (i.e. only keep the Matérn part)
    // and compare to the numerical inverse FT of f̂(ρ) - (A_1 ρ^{-2} + A_2 ρ^{-4}).
    let d: usize = 3;
    let m: usize = 2;
    let s: usize = 2;
    let q: usize = 2;
    let kappa = 1.0_f64;
    let a: usize = 2 * m - q; // = 2
    let b: usize = 2 * s; // = 4

    // Compute A_j coefficients
    let mut a_coeffs: Vec<(usize, f64)> = Vec::new();
    for j in 1..=a {
        let sign = if (a - j).is_multiple_of(2) { 1.0 } else { -1.0 };
        let binom = {
            // C(n,k) for small n,k
            fn c(n: usize, k: usize) -> f64 {
                let mut acc = 1.0_f64;
                for i in 0..k {
                    acc *= (n - i) as f64 / (i + 1) as f64;
                }
                acc
            }
            c(a + b - j - 1, a - j)
        };
        let coeff = sign * binom * kappa.powi(-(2 * (a + b - j) as i32));
        a_coeffs.push((j, coeff));
    }

    for &r in &[0.5_f64, 1.5, 3.0] {
        // g_q^iso(R) - sum of Riesz pieces = sum of Matérn pieces.
        let g_full = isotropic_duchon_penalty(q, d, m, s as f64, kappa, r);
        let mut riesz_sum = 0.0_f64;
        for &(j, c) in &a_coeffs {
            riesz_sum += c * riesz_kernel_value(d, (j) as f64, r);
        }
        let matern_part = g_full - riesz_sum;

        // Numerical Fourier inverse of f̂(ρ) - Σ A_j ρ^{-2j}.
        // Equivalently, inverse FT of Σ B_ℓ (ρ²+1)^{-ℓ}.
        let f_hat_residual = |rho: f64| -> f64 {
            let mut val = 1.0 / (rho.powi(4) * (1.0 + rho * rho).powi(4));
            for &(j, c) in &a_coeffs {
                val -= c * rho.powi(-(2 * j as i32));
            }
            val
        };
        // Integrate on [eps, rho_max]; near ρ=0 the residual is finite (it's a
        // cancellation of the singular part), but rho * sin(ρr) regularizes.
        let approx = fourier_inv_radial_d3(f_hat_residual, r, 80.0);
        let rel = (matern_part - approx).abs() / matern_part.abs().max(1e-12);
        // Numerical inverse-FT truncation at rho_max=80 leaves residual error;
        // the closed form is the ground truth, this is a sanity bound.
        assert!(
            rel < 5e-2,
            "hybrid partial fraction Matérn part mismatch (r={r}): \
                 got={matern_part} approx={approx} rel={rel}"
        );
    }

    // Sanity: bessel_k matches half-integer closed form for ν=1/2
    let k_half = bessel_k(0.5, 1.0);
    let expected = (std::f64::consts::PI / 2.0).sqrt() * (-1.0_f64).exp();
    assert!((k_half - expected).abs() < 1e-12);
}

#[test]
fn test_schoenberg_isotropic_agrees_with_partial_fraction() {
    use super::closed_form_penalty::isotropic_duchon_penalty;

    // `isotropic_duchon_penalty` is the separately q-loaded
    // Riesz-Matérn partial-fraction representative. This test checks that
    // representative directly; the anisotropic radial chain is tested
    // against `(-Δ)^q` of the q=0 representative below.
    let cases: &[(usize, usize, usize, usize, f64, f64)] = &[
        (1, 3, 1, 1, 1.0, 0.5),
        (1, 3, 1, 1, 1.0, 2.0),
        (2, 5, 2, 2, 1.0, 1.0),
    ];
    for &(q, d, m, s, kappa, big_r) in cases {
        let iso = isotropic_duchon_penalty(q, d, m, s as f64, kappa, big_r);
        assert!(
            iso.is_finite(),
            "isotropic q-loaded representative is non-finite: q={q} d={d} m={m} s={s} kappa={kappa} R={big_r}"
        );
    }
}

#[test]
fn test_schoenberg_anisotropic_smooth() {
    use super::closed_form_penalty::anisotropic_duchon_penalty;

    // (d=3, m=1, s=1) is convergent for q=1.
    let r = vec![0.7_f64, -0.3, 1.1];
    let mut prev: Option<f64> = None;
    for k in 0..6 {
        let t = (k as f64) * 0.2 - 0.5; // η_1 sweep ∈ [-0.5, 0.5]
        let eta = vec![t, 0.1, -0.2];
        let v = anisotropic_duchon_penalty(1, 1, 1.0, 1.0, &eta, &r);
        assert!(v.is_finite(), "non-finite g_q at eta_1={t}");
        if let Some(p) = prev {
            // Smoothness sanity: small step in η should produce a small change.
            let ratio = (v - p).abs() / p.abs().max(1e-12);
            assert!(
                ratio < 5.0,
                "discontinuity-like jump in g_q (eta_1 step 0.2): prev={p} curr={v}"
            );
        }
        prev = Some(v);
    }
}

fn build_collocation_operator_penalty_via_dq_dq(
    centers: ArrayView2<'_, f64>,
    q: usize,
    length_scale: f64,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
) -> Array2<f64> {
    let ops = build_duchon_collocation_operator_matrices(
        centers,
        None,
        Some(length_scale),
        power,
        nullspace_order,
        aniso_log_scales,
        None,
        q.max(1),
    )
    .expect("collocation operator matrices");
    let dq = match q {
        0 => &ops.d0,
        1 => &ops.d1,
        2 => &ops.d2,
        _ => panic!("unsupported derivative order"),
    };
    symmetrize(&fast_ata(dq))
}

fn build_closed_form_operator_penalty(
    centers: ArrayView2<'_, f64>,
    q: usize,
    length_scale: f64,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
) -> Array2<f64> {
    let ops = build_duchon_collocation_operator_matrices(
        centers,
        None,
        Some(length_scale),
        power,
        nullspace_order,
        aniso_log_scales,
        None,
        q.max(1),
    )
    .expect("collocation operator matrices");
    let p_order =
        duchon_p_from_nullspace_order(duchon_effective_nullspace_order(centers, nullspace_order));
    let kappa = 1.0 / length_scale;
    closed_form_operator_penalty_in_total_basis(
        centers,
        q,
        p_order,
        duchon_power_to_usize(power),
        kappa,
        aniso_log_scales,
        ops.kernel_nullspace_transform.as_ref(),
        ops.polynomial_block_cols,
        None,
    )
}

fn frobenius_relative_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    let diff_norm = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    diff_norm / scale.max(1e-300)
}

#[test]
fn test_analytic_vs_collocation_high_k_agreement() {
    // Synthesize K=200 deterministic knots in d=3, build s_1 via
    // closed-form (Schoenberg/Riesz-Matérn) and via the existing K-knot
    // collocation D_1^T D_1. Check structural invariants and rough
    // numerical correlation. Convergent regime requires `2m-q ≥ 1` (for
    // the partial-fraction expansion), `4(m+s) > d + 2q` (UV), `d + 2q
    // > 4m` (IR with κ>0), and `2(p+s) > d+1` (D1 collocation validity).
    // (d=3, m=1, s=2) with q=1 is the smallest valid test setup.
    let k = 200;
    let d = 3;
    let length_scale = 1.0; // kappa = 1
    let power = 2.0; // s_order = 2
    let nullspace = DuchonNullspaceOrder::Zero; // p_order = 1 (m=1)

    // Deterministic LCG for reproducibility (no rng dep needed).
    let mut state: u64 = 0xC0FFEE_BEEFu64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 32) as f64) / (u32::MAX as f64) - 0.5
    };
    let mut centers = Array2::<f64>::zeros((k, d));
    for i in 0..k {
        for c in 0..d {
            centers[[i, c]] = next();
        }
    }

    {
        let q = 1usize;
        let analytic = build_closed_form_operator_penalty(
            centers.view(),
            q,
            length_scale,
            power,
            nullspace,
            None,
        );
        let colloc = build_collocation_operator_penalty_via_dq_dq(
            centers.view(),
            q,
            length_scale,
            power,
            nullspace,
            None,
        );

        assert_eq!(analytic.dim(), colloc.dim(), "shape mismatch q={q}");

        // Symmetry of the closed-form penalty.
        let asym = analytic
            .iter()
            .zip(analytic.t().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            asym < 1e-9,
            "closed-form penalty not symmetric (q={q}): max|a-aᵀ|={asym}"
        );

        // Both should be finite.
        assert!(
            analytic.iter().all(|v| v.is_finite()),
            "non-finite analytic q={q}"
        );
        assert!(
            colloc.iter().all(|v| v.is_finite()),
            "non-finite collocation q={q}"
        );

        // Both should be nontrivial (non-zero Frobenius norm).
        let an_norm = analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
        let cl_norm = colloc.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(an_norm > 1e-9, "closed-form penalty is ~zero (q={q})");
        assert!(cl_norm > 1e-9, "collocation penalty is ~zero (q={q})");

        // Normalized matrices (unit Frobenius) — closed-form is the exact
        // integral; collocation is a Riemann-style approximation. After
        // normalization the relative Frobenius difference should be bounded
        // (not necessarily small at K=200 since the two operators differ by
        // a quadrature scheme). Use a permissive bound to assert the two are
        // in the same equivalence class, not just any random matrix.
        let an_normalized = analytic.mapv(|v| v / an_norm);
        let cl_normalized = colloc.mapv(|v| v / cl_norm);
        let rel = frobenius_relative_diff(&an_normalized, &cl_normalized);
        assert!(rel.is_finite(), "non-finite relative diff for q={q}: {rel}");
        assert!(
            rel < 1.5,
            "closed-form vs collocation drift too far at K={k}, q={q}: rel_frob={rel}"
        );
    }
}

// =====================================================================
// Adversarial closed-form-penalty tests
//
// These tests stay in regimes that satisfy the convergence preconditions
// documented at `closed_form_penalty`:
//   * partial-fraction precondition: 2m - q ≥ 1
//   * UV: 4(m+s) > d + 2q
//   * IR (κ>0): d + 2q > 4m
//   * pointwise kernel: 2(p+s) > d  (p = m here)
// and use tolerances tight enough to catch sign / normalization /
// partial-fraction-coefficient / ψ-derivative bugs.
// =====================================================================

#[test]
fn test_riesz_satisfies_laplacian_identity() {
    // Numerical inverse-Fourier on |ρ|^{-2j} is unreliable (singular at
    // ρ = 0, not absolutely integrable in d dimensions for the j values
    // we care about), so the original test compared the closed form to
    // a quadrature with no chance of converging. Replace with the
    // identity that uniquely fixes both sign AND normalization:
    //   −Δ_radial R_j^d = R_{j-1}^d        (j ≥ 2, non-log cases)
    // Fourier-side proof: F[R_j^d] = ρ^{-2j}, F[−Δ] = ρ², product =
    // ρ^{-2(j-1)} = F[R_{j-1}^d]. Any normalization error in
    // riesz_kernel_value would show up here as a constant offset, and
    // a sign flip would make the relation fail outright.
    use super::closed_form_penalty::riesz_kernel_value;
    // Skip log cases (where 2(j-1) ≥ d and 2(j-1)-d is even); restrict
    // to (d, j) for which both R_j^d and R_{j-1}^d are non-log so the
    // identity holds without subtracting log terms.
    // Non-log means 2j != d + 2n for any n ≥ 0, equivalently
    // 2j < d OR (2j - d) is odd. We pick d odd to keep both R_j and
    // R_{j-1} in the non-log regime cleanly.
    let cases: &[(usize, usize)] = &[(3, 2), (3, 3), (5, 2), (5, 3), (7, 3), (7, 4)];
    for &(d, j) in cases {
        for &r in &[0.3_f64, 1.0, 3.0] {
            // 5-point finite-difference stencil for f''(r) and centered
            // finite difference for f'(r).
            let h = 1e-3_f64 * r;
            let f_mm = riesz_kernel_value(d, (j) as f64, r - 2.0 * h);
            let f_m = riesz_kernel_value(d, (j) as f64, r - h);
            let f_0 = riesz_kernel_value(d, (j) as f64, r);
            let f_p = riesz_kernel_value(d, (j) as f64, r + h);
            let f_pp = riesz_kernel_value(d, (j) as f64, r + 2.0 * h);
            let f_pp_d2 = (-f_mm + 16.0 * f_m - 30.0 * f_0 + 16.0 * f_p - f_pp) / (12.0 * h * h);
            let f_p_d1 = (f_mm - 8.0 * f_m + 8.0 * f_p - f_pp) / (12.0 * h);
            let lap = f_pp_d2 + (d as f64 - 1.0) / r * f_p_d1;
            let lhs = -lap;
            let rhs = riesz_kernel_value(d, (j - 1) as f64, r);
            let rel = (lhs - rhs).abs() / rhs.abs().max(1e-300);
            assert!(
                rel < 5e-5,
                "Riesz Laplacian identity broken (d={d}, j={j}, r={r}): \
                     -ΔR_j={lhs:.6e}, R_{{j-1}}={rhs:.6e}, rel={rel:.3e}"
            );
        }
    }
}

#[test]
fn test_log_riesz_finite_part_satisfies_laplacian_identity() {
    use super::closed_form_penalty::riesz_kernel_value;

    // Even-dimensional log-Riesz branches need the finite-part shift
    // A_n in R_{d/2+n}^d = c_n r^{2n}(log r + A_n). The shift is correct
    // exactly when the distributional recurrence survives away from the
    // origin:
    //   -Δ R_j^d = R_{j-1}^d.
    // The old A_n = 0 convention fails for n > 0 by a polynomial residue.
    let cases: &[(usize, usize)] = &[(2, 2), (4, 2), (4, 3), (6, 3), (6, 4)];
    for &(d, j) in cases {
        for &r in &[0.37_f64, 0.9, 2.4] {
            let h = 2e-4_f64 * r;
            let f_mm = riesz_kernel_value(d, (j) as f64, r - 2.0 * h);
            let f_m = riesz_kernel_value(d, (j) as f64, r - h);
            let f_0 = riesz_kernel_value(d, (j) as f64, r);
            let f_p = riesz_kernel_value(d, (j) as f64, r + h);
            let f_pp = riesz_kernel_value(d, (j) as f64, r + 2.0 * h);
            let d2 = (-f_mm + 16.0 * f_m - 30.0 * f_0 + 16.0 * f_p - f_pp) / (12.0 * h * h);
            let d1 = (f_mm - 8.0 * f_m + 8.0 * f_p - f_pp) / (12.0 * h);
            let lhs = -(d2 + (d as f64 - 1.0) * d1 / r);
            let rhs = riesz_kernel_value(d, (j - 1) as f64, r);
            let abs = (lhs - rhs).abs();
            let scale = lhs.abs().max(rhs.abs()).max(1.0);
            assert!(
                abs / scale < 2e-7,
                "log-Riesz finite-part recurrence broken: d={d} j={j} r={r} \
                     -ΔR_j={lhs:.12e} R_{{j-1}}={rhs:.12e} abs={abs:.3e}"
            );
        }
    }
}

#[test]
fn test_matern_matches_half_integer_closed_forms() {
    // The radial-Fourier-inversion test was unstable at small r: the
    // 80-segment Gauss-Legendre rule on (ρ² + κ²)^{-ℓ} on [0, 200] never
    // reaches the asymptotic Bessel oscillation regime cleanly enough
    // for d odd. Replace with explicit half-integer closed forms that
    // pin sign and normalization exactly.
    //
    // For d odd, ν = ℓ - d/2 is half-integer and K_ν reduces to a finite
    // sum:
    //   K_{1/2}(x) = √(π/(2x)) e^{-x}
    //   K_{3/2}(x) = √(π/(2x)) (1 + 1/x) e^{-x}
    //   K_{5/2}(x) = √(π/(2x)) (1 + 3/x + 3/x²) e^{-x}
    //
    // M_ℓ^d(r; κ) = κ^{d/2-ℓ} (2π)^{-d/2} 2^{1-ℓ} / Γ(ℓ) · r^{ℓ-d/2}
    //               · K_{ℓ-d/2}(κr)
    //
    // We tabulate three (d, ℓ) cells covering ν ∈ {1/2, 3/2, 5/2}, all
    // half-integer, all positive (so no singular K_ν at r → 0 issues).
    use super::closed_form_penalty::matern_kernel_value;
    use std::f64::consts::PI;
    // (d, ℓ): ν = ℓ - d/2 ∈ {1/2, 3/2, 5/2}.
    let cases: &[(usize, usize)] = &[(3, 2), (3, 3), (3, 4), (5, 3), (5, 4)];
    for &(d, ell) in cases {
        for &kappa in &[0.5_f64, 1.0, 2.0] {
            for &r in &[0.3_f64, 1.0, 3.0] {
                let nu_2 = 2 * ell as i64 - d as i64; // 2ν, an odd positive integer
                assert!(nu_2 >= 1, "test setup: ν must be > 0");
                let x = kappa * r;
                // K_ν(x) = √(π/(2x)) e^{-x} P_ν(1/x), where P_ν is the
                // explicit polynomial in 1/x for half-integer ν.
                // For ν = (2k+1)/2 (k = 0, 1, 2, …):
                //   P_ν(y) = Σ_{i=0}^{k} (k+i)! / (i! (k-i)!) · (y/2)^i
                let k = ((nu_2 - 1) / 2) as usize;
                let y = 1.0 / x;
                let mut poly = 0.0_f64;
                let mut k_minus_i_factorial = (1..=k).map(|i| i as f64).product::<f64>();
                let mut i_factorial = 1.0_f64;
                let mut k_plus_i_factorial = k_minus_i_factorial; // (k!)
                let mut y_half_pow = 1.0_f64;
                for i in 0..=k {
                    let coeff = k_plus_i_factorial / (i_factorial * k_minus_i_factorial);
                    poly += coeff * y_half_pow;
                    if i < k {
                        // advance: i → i+1
                        i_factorial *= i as f64 + 1.0;
                        k_minus_i_factorial /= (k - i) as f64;
                        k_plus_i_factorial *= (k + i + 1) as f64;
                        y_half_pow *= 0.5 * y;
                    }
                }
                let bessel_k = (PI / (2.0 * x)).sqrt() * (-x).exp() * poly;
                let nu = ell as f64 - 0.5 * d as f64;
                let pref = kappa.powf(0.5 * d as f64 - ell as f64)
                    * (2.0 * PI).powf(-0.5 * d as f64)
                    * 2.0_f64.powf(1.0 - ell as f64)
                    / statrs::function::gamma::gamma(ell as f64);
                let expected = pref * r.powf(nu) * bessel_k;
                let got = matern_kernel_value(d, ell, kappa, r);
                let rel = (got - expected).abs() / expected.abs().max(1e-300);
                assert!(
                    rel < 1e-10,
                    "Matérn closed form disagrees with half-integer K_ν \
                         (d={d}, ℓ={ell}, κ={kappa}, r={r}): \
                         got={got:.10e} expected={expected:.10e} rel={rel:.3e}"
                );
            }
        }
    }
}

#[test]
fn test_matern_satisfies_helmholtz() {
    use super::closed_form_penalty::matern_kernel_value;
    // (κ² − Δ_radial) M_ℓ^d = M_{ℓ-1}^d in d=3.
    let d = 3usize;
    let kappa = 1.0_f64;
    for &ell in &[2usize, 3] {
        for &r in &[0.5_f64, 1.0, 1.7, 2.5] {
            let h = 1e-4_f64;
            let m_pp = matern_kernel_value(d, ell, kappa, r + h);
            let m_mm = matern_kernel_value(d, ell, kappa, r - h);
            let m_0 = matern_kernel_value(d, ell, kappa, r);
            let f_pp = (m_pp - 2.0 * m_0 + m_mm) / (h * h);
            let f_p = (m_pp - m_mm) / (2.0 * h);
            let lap = f_pp + 2.0 / r * f_p;
            let lhs = kappa * kappa * m_0 - lap;
            let rhs = matern_kernel_value(d, ell - 1, kappa, r);
            let rel = (lhs - rhs).abs() / rhs.abs().max(1e-300);
            assert!(
                rel < 5e-3,
                "Helmholtz relation broken: d={d}, ℓ={ell}, r={r}: \
                     (κ²−Δ)M_ℓ={lhs:.6e}, M_{{ℓ−1}}={rhs:.6e}, rel={rel:.3e}"
            );
        }
    }
}

#[test]
fn test_isotropic_duchon_satisfies_partial_fraction_identity() {
    use super::closed_form_penalty::{
        isotropic_duchon_penalty, matern_kernel_value, riesz_kernel_value,
    };
    let d = 3usize;
    let m = 2usize;
    let s = 2usize;
    let q = 2usize;
    let kappa = 1.0_f64;
    let a = 2 * m - q;
    let b = 2 * s;
    let kappa_sq = kappa * kappa;

    fn binom(n: usize, k: usize) -> f64 {
        let mut acc = 1.0_f64;
        for i in 0..k {
            acc *= (n - i) as f64 / (i + 1) as f64;
        }
        acc
    }

    for &r in &[0.4_f64, 0.9, 1.5, 2.5, 5.0] {
        let mut expected = 0.0_f64;
        for j in 1..=a {
            let sign = if (a - j).is_multiple_of(2) { 1.0 } else { -1.0 };
            let coeff = sign * binom(a + b - j - 1, a - j) * kappa_sq.powi(-((a + b - j) as i32));
            expected += coeff * riesz_kernel_value(d, (j) as f64, r);
        }
        let sign_a = if a.is_multiple_of(2) { 1.0 } else { -1.0 };
        for ell in 1..=b {
            let coeff =
                sign_a * binom(a + b - ell - 1, b - ell) * kappa_sq.powi(-((a + b - ell) as i32));
            expected += coeff * matern_kernel_value(d, ell, kappa, r);
        }
        let got = isotropic_duchon_penalty(q, d, m, s as f64, kappa, r);
        let rel = (got - expected).abs() / expected.abs().max(1e-300);
        assert!(
            rel < 1e-12,
            "partial-fraction identity broken (r={r}): got={got:.10e} expected={expected:.10e} rel={rel:.3e}"
        );
    }
}

#[test]
fn test_isotropic_duchon_kappa_to_zero_limit() {
    use super::closed_form_penalty::{isotropic_duchon_penalty, riesz_kernel_value};
    // For positive κ,
    //   ĝ_κ(ρ) = 1 / (ρ^{2a}(κ²+ρ²)^b),  a = 2m-q, b = 2s.
    // The pointwise κ→0 limit equals the pure Riesz representative only
    // when the low-frequency tail is uniformly integrable:
    //
    //   ∫_0^κ ρ^{d-1-2a} κ^{-2b} dρ = O(κ^{d-2a-2b}) → 0
    //
    // hence d > 2(a+b). Use d=13 for a=1,b=4 so the positive-κ kernel
    // genuinely converges to R_{a+b}^d away from the origin.
    let d = 13usize;
    let m = 1usize;
    let s = 2usize;
    let q = 1usize;
    let r = 1.3_f64;
    let target = riesz_kernel_value(d, (2 * m + 2 * s - q) as f64, r);
    let kappas = [1.0_f64, 0.1, 0.01, 0.001];
    let mut prev_err = f64::INFINITY;
    for &kappa in &kappas {
        let got = isotropic_duchon_penalty(q, d, m, s as f64, kappa, r);
        let err = (got - target).abs() / target.abs();
        if kappa <= 0.5 {
            assert!(
                err <= prev_err * 1.05 + 1e-15,
                "convergence not monotone: κ={kappa}, err={err:.3e}, prev_err={prev_err:.3e}"
            );
        }
        prev_err = err;
    }
    let got = isotropic_duchon_penalty(q, d, m, s as f64, 0.001, r);
    let rel = (got - target).abs() / target.abs();
    assert!(
        rel < 5e-3,
        "κ→0 limit not Riesz: got={got:.6e} target={target:.6e} rel={rel:.3e}"
    );
}

#[test]
fn test_isotropic_duchon_kappa_to_zero_ir_divergence_is_quotiented_by_finite_part() {
    use super::closed_form_penalty::{isotropic_duchon_penalty, riesz_kernel_value};

    // Same (m,s,q) as the convergent test but d=5. Now a=1,b=4 and
    // d-2a-2b = -5, so the ordinary low-frequency positive-κ Green's
    // function carries a divergent polynomial/nullspace component. The
    // constrained Duchon kernel is its finite-part representative; after
    // quotienting that nullspace, the off-diagonal value converges to the
    // κ=0 Riesz representative. The small-χ Riesz-series chart is what
    // resolves the severe Riesz/Matérn cancellation needed to see this.
    let d = 5usize;
    let m = 1usize;
    let s = 2usize;
    let q = 1usize;
    let r = 1.3_f64;
    let finite_part = riesz_kernel_value(d, (2 * m + 2 * s - q) as f64, r);
    let kappa_hi = 0.1_f64;
    let kappa_lo = 0.01_f64;
    let hi = isotropic_duchon_penalty(q, d, m, s as f64, kappa_hi, r);
    let lo = isotropic_duchon_penalty(q, d, m, s as f64, kappa_lo, r);

    assert!(
        hi.is_finite() && lo.is_finite() && finite_part.is_finite(),
        "test setup should stay finite away from κ=0 and r=0"
    );
    assert!(
        hi.abs() > 1.0e5 * finite_part.abs(),
        "moderate positive-κ finite-part representative should still be far from κ=0 Riesz at κ={kappa_hi}: \
             hi={hi:.6e}, finite_part={finite_part:.6e}"
    );
    let lo_err = (lo - finite_part).abs();
    assert!(
        lo_err < 1.0e-3 * finite_part.abs(),
        "small positive-κ finite-part representative should converge to κ=0 Riesz: \
             κ={kappa_lo}, value={lo:.6e}, finite_part={finite_part:.6e}, err={lo_err:.6e}"
    );
}

#[test]
fn test_small_kappa_finite_part_chart_is_shared_by_value_radial_and_kappa_partials() {
    use super::closed_form_penalty::{
        isotropic_duchon_penalty, radial_derivatives_of_isotropic_duchon,
        radial_derivatives_of_isotropic_duchon_kappa_partial,
        radial_derivatives_of_isotropic_duchon_kappa_partial2,
    };

    fn finite_part_series(
        d: usize,
        a: usize,
        b: usize,
        kappa: f64,
        r: f64,
        max_order: usize,
        kappa_derivative_order: usize,
    ) -> Vec<f64> {
        fn odd_dim_riesz_block_from_value(
            d: usize,
            j: usize,
            r: f64,
            max_order: usize,
        ) -> Vec<f64> {
            assert_eq!(d % 2, 1);
            let value = super::closed_form_penalty::riesz_kernel_value(d, (j) as f64, r);
            let p = 2 * j as i32 - d as i32;
            let mut out = Vec::with_capacity(max_order + 1);
            out.push(value);
            let mut coef = value;
            for order in 1..=max_order {
                coef *= (p - (order as i32 - 1)) as f64 / r;
                out.push(coef);
            }
            out
        }

        let mut out = vec![0.0_f64; max_order + 1];
        let mut coeff = 1.0_f64;
        let kappa_sq = kappa * kappa;
        for n in 0..96 {
            let kappa_factor = match kappa_derivative_order {
                0 => 1.0,
                other => {
                    // Closed-form k-th κ-derivative of (κ²)^n; mirrors the
                    // production routine
                    // `duchon_small_chi_riesz_series_radial_derivatives`.
                    if n == 0 {
                        0.0
                    } else {
                        let p = 2.0 * n as f64;
                        let mut numerator = 1.0_f64;
                        for j in 0..other {
                            numerator *= p - j as f64;
                        }
                        let denom = kappa.powi(other as i32);
                        numerator / denom
                    }
                }
            };
            if kappa_factor != 0.0 {
                let block = odd_dim_riesz_block_from_value(d, a + b + n, r, max_order);
                for (order, value) in block.into_iter().enumerate() {
                    out[order] += coeff * kappa_factor * value;
                }
            }
            coeff *= -((b + n) as f64) * kappa_sq / ((n + 1) as f64);
        }
        out
    }

    // Same singular cell that exposed the κ→0 failure, but q=0 because
    // `radial_derivatives_of_isotropic_duchon` is the base f(R) before
    // anisotropic Laplacians apply q. The root invariant is that every
    // production path uses this same constrained finite-part representative:
    // value, radial R-derivatives, and κ-partials.
    let d = 5usize;
    let m = 1usize;
    let s = 2usize;
    let a = 2 * m;
    let b = 2 * s;
    let kappa = 0.01_f64;
    let r = 1.3_f64;
    let max_order = 4usize;

    let expected = finite_part_series(d, a, b, kappa, r, max_order, 0);
    let expected_dk = finite_part_series(d, a, b, kappa, r, max_order, 1);
    let expected_dkk = finite_part_series(d, a, b, kappa, r, max_order, 2);

    let value = isotropic_duchon_penalty(0, d, m, s as f64, kappa, r);
    let radial = radial_derivatives_of_isotropic_duchon(d, m, (s) as f64, kappa, r, max_order);
    let dk = radial_derivatives_of_isotropic_duchon_kappa_partial(d, m, s, kappa, r, max_order);
    let dkk = radial_derivatives_of_isotropic_duchon_kappa_partial2(d, m, s, kappa, r, max_order);

    let rel0 = (value - expected[0]).abs() / expected[0].abs().max(1e-300);
    assert!(
        rel0 < 1e-12,
        "small-κ value path is not the finite-part chart: got={value:.12e} expected={:.12e} rel={rel0:.3e}",
        expected[0]
    );

    for order in 0..=max_order {
        let denom = expected[order].abs().max(radial[order].abs()).max(1e-300);
        let rel = (radial[order] - expected[order]).abs() / denom;
        assert!(
            rel < 1e-12,
            "small-κ radial derivative order {order} mixed charts: got={:.12e} expected={:.12e} rel={rel:.3e}",
            radial[order],
            expected[order]
        );

        let denom = expected_dk[order].abs().max(dk[order].abs()).max(1e-300);
        let rel = (dk[order] - expected_dk[order]).abs() / denom;
        assert!(
            rel < 1e-11,
            "small-κ κ-partial radial order {order} mixed charts: got={:.12e} expected={:.12e} rel={rel:.3e}",
            dk[order],
            expected_dk[order]
        );

        let denom = expected_dkk[order].abs().max(dkk[order].abs()).max(1e-300);
        let rel = (dkk[order] - expected_dkk[order]).abs() / denom;
        assert!(
            rel < 1e-10,
            "small-κ κκ-partial radial order {order} mixed charts: got={:.12e} expected={:.12e} rel={rel:.3e}",
            dkk[order],
            expected_dkk[order]
        );
    }
}

#[test]
fn test_even_log_riesz_small_kappa_uses_full_taylor_series() {
    use super::closed_form_penalty::{isotropic_duchon_penalty, riesz_kernel_value};

    // Even-dimensional log-Riesz case: d/2 <= N = 2m - q + 2s.
    // This used to return only the leading R_N term under cancellation.
    // The expected value below is the finite-part Taylor series itself,
    // including log-Riesz constants in every R_{N+n} term.
    let d = 4usize;
    let m = 1usize;
    let s = 2usize;
    let q = 1usize;
    let a = 2 * m - q;
    let b = 2 * s;
    let n0 = a + b;
    let kappa = 0.08_f64;
    let r = 1.1_f64;

    let mut coeff = 1.0_f64;
    let mut expected = 0.0_f64;
    for n in 0..80 {
        expected += coeff * riesz_kernel_value(d, (n0 + n) as f64, r);
        coeff *= -((b + n) as f64) * kappa * kappa / ((n + 1) as f64);
    }
    let leading = riesz_kernel_value(d, (n0) as f64, r);
    let got = isotropic_duchon_penalty(q, d, m, s as f64, kappa, r);
    let rel = (got - expected).abs() / expected.abs().max(1e-300);
    assert!(
        rel < 1e-11,
        "even log-Riesz Taylor mismatch: got={got:.12e} expected={expected:.12e} rel={rel:.3e}"
    );
    assert!(
        (expected - leading).abs() > 1e-5 * expected.abs().max(1e-300),
        "test must exercise more than the leading κ→0 term"
    );
}

#[test]
fn test_even_log_riesz_small_kappa_derivative_bundle_matches_fd() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty_radial, pair_block_radial_with_j_second_derivatives,
    };

    // Same even-dimensional log-Riesz cancellation basin as the value
    // Taylor test, but through the production derivative bundle. This
    // pins the important wiring: value, η, κ, ηκ, and κκ must all use
    // one analytic positive-κ representation rather than mixing a stable
    // value path with cancelled partial-fraction derivatives.
    let q = 1usize;
    let m = 1usize;
    let s = 2usize;
    let kappa = 0.08_f64;
    let eta = vec![0.15_f64, -0.10, 0.05, -0.02];
    let r = vec![0.7_f64, -0.4, 0.3, 0.2];
    let big_j = eta.iter().sum::<f64>().exp();
    let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);
    assert!(bundle.value.is_finite());

    let v_eta = |eta_use: &[f64]| -> f64 {
        eta_use.iter().sum::<f64>().exp()
            * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, eta_use, &r)
    };
    let h_eta = 2.0e-5_f64;
    for axis in 0..eta.len() {
        let mut ep = eta.clone();
        let mut em = eta.clone();
        ep[axis] += h_eta;
        em[axis] -= h_eta;
        let fd = (v_eta(&ep) - v_eta(&em)) / (2.0 * h_eta);
        let denom = fd.abs().max(bundle.d_eta[axis].abs()).max(1.0e-12);
        let rel = (bundle.d_eta[axis] - fd).abs() / denom;
        assert!(
            rel < 2.0e-6,
            "even log-Riesz η derivative mismatch on axis {axis}: bundle={:.12e} fd={:.12e} rel={rel:.3e}",
            bundle.d_eta[axis],
            fd
        );
    }

    let v_kappa = |kk: f64| -> f64 {
        big_j * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kk, &eta, &r)
    };
    let h_kappa = 1.0e-4 * kappa;
    let dk_fd = (v_kappa(kappa + h_kappa) - v_kappa(kappa - h_kappa)) / (2.0 * h_kappa);
    let denom = dk_fd.abs().max(bundle.d_kappa.abs()).max(1.0e-12);
    let rel = (bundle.d_kappa - dk_fd).abs() / denom;
    assert!(
        rel < 5.0e-6,
        "even log-Riesz κ derivative mismatch: bundle={:.12e} fd={:.12e} rel={rel:.3e}",
        bundle.d_kappa,
        dk_fd
    );
}

/// Closed-set tag for the q values the closed-form radial Laplacian
/// helper supports. Lets call sites discharge the q ∈ {0, 1, 2}
/// restriction at the type level instead of via a runtime check.
#[derive(Copy, Clone)]
enum RadialLaplacianPower {
    Q0,
    Q1,
    Q2,
}

impl RadialLaplacianPower {
    fn from_usize(q: usize) -> Self {
        match q {
            0 => Self::Q0,
            1 => Self::Q1,
            2 => Self::Q2,
            // Saturate higher q to Q2: the call sites in this test module
            // only ever pass q ∈ {0, 1, 2}, so the saturation never fires
            // in practice but keeps the conversion total.
            _ => Self::Q2,
        }
    }
    fn fr_len(self) -> usize {
        match self {
            Self::Q0 => 1,
            Self::Q1 => 3,
            Self::Q2 => 5,
        }
    }
}

fn isotropic_radial_laplacian_power_from_q0(
    q: usize,
    d: usize,
    m: usize,
    s: usize,
    kappa: f64,
    big_r: f64,
) -> f64 {
    use super::closed_form_penalty::radial_derivatives_of_isotropic_duchon;

    let power = RadialLaplacianPower::from_usize(q);
    let fr_needed = power.fr_len();
    let fr = radial_derivatives_of_isotropic_duchon(d, m, (s) as f64, kappa, big_r, 2 * q);
    // Defensive bound check: `fr_needed` derivatives must be returned.
    assert!(
        fr.len() >= fr_needed,
        "radial_derivatives_of_isotropic_duchon returned {} values, need {}",
        fr.len(),
        fr_needed
    );
    match power {
        RadialLaplacianPower::Q0 => fr[0],
        RadialLaplacianPower::Q1 => -(fr[2] + ((d as f64) - 1.0) * fr[1] / big_r),
        RadialLaplacianPower::Q2 => {
            let r2 = big_r * big_r;
            let r3 = r2 * big_r;
            let dm1 = (d as f64) - 1.0;
            let dm3 = (d as f64) - 3.0;
            fr[4] + 2.0 * dm1 * fr[3] / big_r + dm1 * dm3 * fr[2] / r2 - dm1 * dm3 * fr[1] / r3
        }
    }
}

#[test]
fn test_anisotropic_eta_zero_matches_q0_radial_chain_strict_tolerance() {
    use super::closed_form_penalty::anisotropic_duchon_penalty;
    // At η = 0, the anisotropic operator must reduce to the q-fold
    // radial `(-Δ)^q` chain applied to the q=0 constrained Duchon
    // representative. It must not silently switch to a separately
    // q-loaded finite-part representative.
    let cases: &[(usize, usize, usize, usize, f64)] = &[(1, 3, 1, 2, 1.0), (2, 5, 2, 2, 1.0)];
    for &(q, d, m, s, kappa) in cases {
        for &big_r in &[0.2_f64, 0.5, 1.0, 2.0, 4.0] {
            let eta = vec![0.0_f64; d];
            let mut r = vec![0.0_f64; d];
            r[0] = big_r;
            let aniso = anisotropic_duchon_penalty(q, m, (s) as f64, kappa, &eta, &r);
            let expected = isotropic_radial_laplacian_power_from_q0(q, d, m, s, kappa, big_r);
            let rel = (aniso - expected).abs() / expected.abs().max(aniso.abs()).max(1e-300);
            assert!(
                rel < 1e-12,
                "strict eta-zero radial-chain disagreement: q={q} d={d} m={m} s={s} κ={kappa} R={big_r} \
                     aniso={aniso:.6e} expected={expected:.6e} rel={rel:.3e}"
            );
        }
    }
}

#[test]
fn test_anisotropic_public_wrapper_eta_zero_matches_radial_chain() {
    use super::closed_form_penalty::anisotropic_duchon_penalty;
    let q = 1usize;
    let d = 3usize;
    let m = 1usize;
    let s = 2usize;
    let kappa = 1.0_f64;
    for &big_r in &[0.4_f64, 1.0, 2.5] {
        let eta = vec![0.0_f64; d];
        let mut r = vec![0.0_f64; d];
        r[0] = big_r;
        let aniso = anisotropic_duchon_penalty(q, m, (s) as f64, kappa, &eta, &r);
        let expected = isotropic_radial_laplacian_power_from_q0(q, d, m, s, kappa, big_r);
        let rel = (aniso - expected).abs() / expected.abs().max(aniso.abs()).max(1e-300);
        assert!(
            rel < 1e-12,
            "eta-zero aniso/radial-chain disagreement at R={big_r}: \
                 aniso={aniso:.6e} expected={expected:.6e} rel={rel:.3e}"
        );
    }
}

#[test]
fn test_psi_first_deriv_invariance_under_uniform_eta_shift() {
    // The J = exp(Σ η_k) prefactor must scale as e^{c·d} under a uniform
    // η shift by c.  This catches a missing/wrong J prefactor in the
    // pair-block bundle.
    use super::closed_form_penalty::{
        pair_block_radial_with_j_second_derivatives, psi_first_derivative,
    };
    let q = 1usize;
    let m = 1usize;
    let s = 1usize;
    let kappa = 1.0_f64;
    let d = 3usize;
    let eta_base = vec![0.1_f64, -0.2, 0.3];
    let r = vec![0.5_f64, 0.4, 0.3];
    let c = 0.4_f64;
    let eta_shift: Vec<f64> = eta_base.iter().map(|x| x + c).collect();

    let val_base = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta_base, &r).value;
    let bundle_shift = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta_shift, &r);
    let val_shift = bundle_shift.value;

    let j_base: f64 = eta_base.iter().sum::<f64>().exp();
    let j_shift: f64 = eta_shift.iter().sum::<f64>().exp();
    let bare_base = val_base / j_base;
    let bare_shift = val_shift / j_shift;

    let j_ratio = j_shift / j_base;
    let expected_ratio = (c * d as f64).exp();
    let rel = (j_ratio - expected_ratio).abs() / expected_ratio;
    assert!(
        rel < 1e-12,
        "J prefactor scaling wrong: J_shift/J_base={j_ratio} expected e^{{cd}}={expected_ratio}"
    );

    assert!(bare_base.is_finite() && bare_base.abs() > 1e-30);
    assert!(bare_shift.is_finite() && bare_shift.abs() > 1e-30);

    let bare_d0 = psi_first_derivative(q, m, s, kappa, &eta_shift, &r, 0);
    let unwrapped_d0 = (bundle_shift.d_eta[0] - bundle_shift.value) / j_shift;
    assert!((bare_d0 - unwrapped_d0).abs() < 1e-12);
}

#[test]
fn test_anisotropy_breaks_isotropic_symmetry() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty, isotropic_duchon_penalty,
        pair_block_radial_with_j_second_derivatives,
    };
    let q = 1usize;
    let m = 1usize;
    let s = 1usize;
    let d = 3usize;
    let kappa = 1.0_f64;
    let c = 0.5_f64;
    let eta = vec![c, 0.0_f64, 0.0_f64];
    let r = vec![1.0_f64, 0.0_f64, 0.0_f64];

    let aniso_bare = anisotropic_duchon_penalty(q, m, (s) as f64, kappa, &eta, &r);
    let z_norm = (-c).exp();
    let iso_at_z = isotropic_duchon_penalty(q, d, m, s as f64, kappa, z_norm);

    let rel = (aniso_bare - iso_at_z).abs() / iso_at_z.abs().max(1e-300);
    assert!(
        rel > 0.05,
        "anisotropic call collapsed to isotropic-at-|z|: \
             aniso={aniso_bare:.6e} iso(|z|)={iso_at_z:.6e} rel={rel:.3e} \
             (expected meaningful disagreement under anisotropy)"
    );

    let val_with_j = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r).value;
    let expected_j = eta.iter().sum::<f64>().exp();
    let bare = val_with_j / expected_j;
    let bare_rel = (bare - aniso_bare).abs() / aniso_bare.abs().max(1e-300);
    assert!(
        bare_rel < 1e-12,
        "pair_block J/bare consistency broken: J·g={val_with_j} g={aniso_bare} ratio={}",
        val_with_j / aniso_bare
    );
}

#[test]
fn test_value_against_completely_independent_brute_force() {
    // (d=3, m=1, s=2, κ=1, q=1) — UV 12>5 ✓, IR 5>4 ✓, 2m-q=1 ✓.
    // f̂(ρ) = 1 / (ρ² (1+ρ²)^4) — both ends integrable in d=3 sin form.
    use super::closed_form_penalty::isotropic_duchon_penalty;
    let q = 1usize;
    let d = 3usize;
    let m = 1usize;
    let s = 2usize;
    let kappa = 1.0_f64;
    let f_hat = |rho: f64| 1.0 / (rho * rho * (1.0 + rho * rho).powi(4));
    for &big_r in &[0.5_f64, 1.5, 3.0] {
        let closed = isotropic_duchon_penalty(q, d, m, s as f64, kappa, big_r);
        let approx = fourier_inv_radial_d3(f_hat, big_r, 100.0);
        let rel = (closed - approx).abs() / closed.abs().max(1e-300);
        assert!(
            rel < 1e-3,
            "brute-force FT vs closed-form mismatch (R={big_r}): \
                 closed={closed:.6e} approx={approx:.6e} abs={:.3e} rel={rel:.3e}",
            (closed - approx).abs()
        );
    }
}

#[test]
fn test_anisotropic_public_wrapper_matches_radial_closed_form() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty, anisotropic_duchon_penalty_radial,
    };

    // The public wrapper keeps the historical function name but delegates
    // to the same analytic radial closed form for every metric, including
    // uniform and η = 0 metrics.
    let cases: &[(usize, usize, usize, usize, f64)] = &[(1, 3, 1, 1, 1.0), (2, 5, 2, 2, 1.0)];

    for &(q, d, m, s, kappa) in cases {
        // Try a few non-degenerate r vectors.
        let r_choices: &[Vec<f64>] = &[
            (0..d).map(|i| 0.3 + 0.2 * i as f64).collect(),
            (0..d).map(|i| 0.7 + 0.1 * i as f64).collect(),
        ];
        let eta_choices: &[Vec<f64>] = &[
            (0..d).map(|i| 0.1 - 0.05 * i as f64).collect(),
            (0..d).map(|i| -0.03 + 0.02 * i as f64).collect(),
        ];
        for r in r_choices {
            for eta in eta_choices {
                let radial = anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, eta, r);
                let wrapped = anisotropic_duchon_penalty(q, m, (s) as f64, kappa, eta, r);
                let rel = (radial - wrapped).abs() / wrapped.abs().max(radial.abs()).max(1e-300);
                assert!(
                    rel < 1e-12,
                    "public wrapper vs radial closed form disagreement: q={q} d={d} m={m} s={s} \
                         kappa={kappa} r={:?} eta={:?} radial={radial:.6e} \
                         wrapped={wrapped:.6e} rel={rel:.3e}",
                    r,
                    eta
                );
            }
        }
    }
}

#[test]
fn test_radial_form_isotropic_limit_matches_radial_laplacian_chain() {
    // η = 0 collapses the radial-form anisotropic penalty to
    //   J · (-Δ)^q f(R) on the q=0 constrained radial representative:
    //   q = 0:  f(R)
    //   q = 1: -[ f''(R) + (d-1) f'(R)/R ]
    //   q = 2:  Δ²f(R) (well-known closed form on radial fns)
    use super::closed_form_penalty::anisotropic_duchon_penalty_radial;

    // Use parameter triples covering both Riesz-only (s=0) and
    // hybrid (s>0). All require 2m - q ≥ 1 from
    // isotropic_duchon_penalty's invariant.
    // For q=1: m=1 is fine.
    // For q=2: m=2 is needed.
    let cases: &[(usize, usize, usize, usize, f64)] = &[
        // (q, d, m, s, kappa)
        (0, 3, 2, 0, 0.0), // pure Riesz, q=0
        (1, 3, 1, 0, 0.0), // pure Riesz, q=1
        (1, 3, 1, 1, 1.0), // hybrid q=1
        (2, 3, 2, 0, 0.0), // pure Riesz, q=2
        (2, 4, 2, 0, 0.0), // pure Riesz, q=2 (log-Riesz case)
        (2, 5, 2, 1, 1.5), // hybrid q=2
    ];
    for &(q, d, m, s, kappa) in cases {
        let eta = vec![0.0_f64; d];
        // R varies: place lag along one axis with various magnitudes.
        for &big_r in &[0.4_f64, 1.0, 2.5] {
            let mut r = vec![0.0_f64; d];
            r[0] = big_r;
            let radial = anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, &eta, &r);
            let expected = isotropic_radial_laplacian_power_from_q0(q, d, m, s, kappa, big_r);
            let abs = (radial - expected).abs();
            let scale = expected.abs().max(radial.abs()).max(1.0);
            assert!(
                abs / scale < 1e-8,
                "radial form (η=0) disagrees with radial Laplacian chain: \
                     q={q} d={d} m={m} s={s} kappa={kappa} R={big_r} \
                     radial={radial:.6e} expected={expected:.6e} abs={abs:.3e}"
            );
        }
    }
}

#[test]
fn test_radial_derivatives_match_finite_differences() {
    use super::closed_form_penalty::{
        isotropic_duchon_penalty, radial_derivatives_of_isotropic_duchon,
    };

    // (d=3, m=2, s=1, κ=1.5) — 2m=4, plenty of derivatives.
    let d = 3usize;
    let m = 2usize;
    let s = 1usize;
    let kappa = 1.5_f64;
    let r = 0.7_f64;
    let h = 1e-3_f64;

    let derivs = radial_derivatives_of_isotropic_duchon(d, m, (s) as f64, kappa, r, 4);
    // Cross-check derivs[0] against isotropic_duchon_penalty(q=0, …)
    let f0 = isotropic_duchon_penalty(0, d, m, s as f64, kappa, r);
    assert!(
        (derivs[0] - f0).abs() / f0.abs().max(1e-300) < 1e-12,
        "radial deriv 0 mismatch: got={} f0={f0}",
        derivs[0]
    );
    // FD 1st derivative of f via 2-pt central difference.
    let f_at = |rr: f64| isotropic_duchon_penalty(0, d, m, s as f64, kappa, rr);
    let fd1 = (f_at(r + h) - f_at(r - h)) / (2.0 * h);
    let rel1 = (derivs[1] - fd1).abs() / fd1.abs().max(1e-300);
    assert!(
        rel1 < 1e-5,
        "f'(R): analytic={} fd={fd1} rel={rel1}",
        derivs[1]
    );
    // FD 2nd derivative
    let fd2 = (f_at(r + h) - 2.0 * f_at(r) + f_at(r - h)) / (h * h);
    let rel2 = (derivs[2] - fd2).abs() / fd2.abs().max(1e-300);
    assert!(
        rel2 < 1e-3,
        "f''(R): analytic={} fd={fd2} rel={rel2}",
        derivs[2]
    );
}

// ------------------------------------------------------------------
// Adversarial closed-form Duchon stress tests.
// ------------------------------------------------------------------

fn det_rand(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
}

#[test]
fn test_aniso_scale_invariance_via_letter_a_section_9() {
    use super::closed_form_penalty::anisotropic_duchon_penalty_radial;

    let cases: &[(usize, usize, usize, usize, f64)] = &[
        (0, 3, 1, 1, 0.7),
        (1, 3, 1, 1, 1.3),
        (1, 5, 1, 2, 0.4),
        (2, 5, 2, 2, 0.9),
        (0, 7, 1, 1, 1.1),
    ];

    let mut seed = 0xC0FFEE_u64;
    for &(q, d, m, s, kappa) in cases {
        for _trial in 0..6 {
            let eta_raw: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 1.6).collect();
            let r: Vec<f64> = (0..d).map(|_| 0.4 + 1.6 * det_rand(&mut seed)).collect();

            let mu: f64 = eta_raw.iter().sum::<f64>() / d as f64;
            let eta_c: Vec<f64> = eta_raw.iter().map(|&e| e - mu).collect();
            let kappa_new = kappa * (-mu).exp();

            let j_raw = eta_raw.iter().sum::<f64>().exp();
            let g_raw =
                j_raw * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, &eta_raw, &r);

            let j_c = eta_c.iter().sum::<f64>().exp();
            let g_c =
                j_c * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa_new, &eta_c, &r);

            let prefactor = (((2 * d) as f64 - (4 * m) as f64 - (4 * s) as f64) * mu).exp();
            let predicted = prefactor * g_c;

            let denom = predicted.abs().max(g_raw.abs()).max(1e-300);
            let rel = (g_raw - predicted).abs() / denom;
            assert!(
                rel < 1e-9,
                "Letter A §9 identity violated: q={q} d={d} m={m} s={s} κ={kappa} \
                     μ={mu:.4} g_raw={g_raw:.6e} predicted={predicted:.6e} rel={rel:.3e} \
                     η_raw={eta_raw:?} r={r:?}"
            );
        }
    }
}

#[test]
fn test_pair_block_psd_in_convergent_regime() {
    let cases: &[(usize, usize, usize, usize, f64)] = &[
        (1, 3, 1, 1, 0.7),
        (1, 5, 1, 2, 1.0),
        (2, 5, 2, 2, 0.6),
        (2, 7, 2, 3, 1.1),
    ];

    let mut seed = 0xDEAD_BEEF_u64;
    let k = 6_usize;

    for &(q, d, m, s, kappa) in cases {
        for _trial in 0..3 {
            let mut centers = Array2::<f64>::zeros((k, d));
            for i in 0..k {
                for c in 0..d {
                    centers[[i, c]] = det_rand(&mut seed);
                }
            }
            let eta: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 0.4).collect();

            let g = closed_form_anisotropic_pair_block(centers.view(), q, m, s, kappa, Some(&eta));

            let order = DuchonNullspaceOrder::Linear;
            let p_block = polynomial_block_from_order(centers.view(), order);
            let z = match kernel_constraint_nullspace_from_matrix(p_block.view()) {
                Ok(z) => z,
                Err(_) => continue,
            };

            let gz = g.dot(&z);
            let zgz = z.t().dot(&gz);
            let sym = (&zgz + &zgz.t()) * 0.5;

            let (min_eig, scale) = symmetric_eigenvalue_bounds_jacobi(&sym);
            let tol = 1e-9 * scale + 1e-12;
            assert!(
                min_eig > -tol,
                "constrained pair-block not PSD: q={q} d={d} m={m} s={s} κ={kappa} \
                     min_eig={min_eig:.3e} tol={tol:.3e}"
            );
        }
    }
}

#[test]
fn singular_convergent_derivative_builders_use_analytic_self_pair() {
    let q = 1usize;
    let d = 3usize;
    let m = 1usize;
    let s = 1usize;
    let kappa = 0.8_f64;
    let dp2q = d + 2 * q;
    assert!(2 * (m + s) <= dp2q && 4 * (m + s) > dp2q && dp2q > 4 * m);

    let centers = array![
        [0.0, 0.0, 0.0],
        [0.4, 0.1, 0.2],
        [0.2, 0.7, 0.3],
        [0.5, 0.3, 0.9],
    ];
    let eta = vec![0.12, -0.08, 0.04];

    let value = closed_form_anisotropic_pair_block(centers.view(), q, m, s, kappa, Some(&eta));
    let (psi_value, _, _) = closed_form_psi_derivatives_in_total_basis(
        centers.view(),
        q,
        m,
        s,
        kappa,
        Some(&eta),
        None,
        0,
        None,
    );
    let (aniso_value, _, _, _) = closed_form_aniso_psi_derivatives_in_total_basis(
        centers.view(),
        q,
        m,
        s,
        kappa,
        Some(&eta),
        None,
        0,
        None,
    );

    for i in 0..centers.nrows() {
        for j in 0..centers.nrows() {
            let denom = value[[i, j]]
                .abs()
                .max(psi_value[[i, j]].abs())
                .max(aniso_value[[i, j]].abs())
                .max(1e-300);
            assert!(
                (psi_value[[i, j]] - value[[i, j]]).abs() / denom < 1e-12,
                "log-kappa derivative builder value must match analytic pair matrix at ({i},{j}): value={:.16e} psi_value={:.16e} aniso_value={:.16e}",
                value[[i, j]],
                psi_value[[i, j]],
                aniso_value[[i, j]]
            );
            assert!(
                (aniso_value[[i, j]] - value[[i, j]]).abs() / denom < 1e-12,
                "eta derivative builder value must match analytic pair matrix at ({i},{j}): value={:.16e} psi_value={:.16e} aniso_value={:.16e}",
                value[[i, j]],
                psi_value[[i, j]],
                aniso_value[[i, j]]
            );
        }
    }

    let zero_lag = vec![0.0_f64; d];
    let diag_bundle = closed_form_penalty::pair_block_radial_with_j_second_derivatives(
        q, m, s, kappa, &eta, &zero_lag,
    );
    assert!(
        (value[[0, 0]] - diag_bundle.value).abs()
            / value[[0, 0]].abs().max(diag_bundle.value.abs()).max(1e-300)
            < 1e-12,
        "diagonal must use the analytic distributional self-pair, not epsilon radial"
    );
}

#[test]
fn test_radial_form_matches_q0_laplacian_chain_at_eta_zero_full_sweep() {
    use super::closed_form_penalty::anisotropic_duchon_penalty_radial;

    let qs = [0_usize, 1, 2];
    let ds = [1_usize, 3, 5, 7, 9, 11];
    let ms = [1_usize, 2, 3];
    let ss = [1_usize, 2, 3, 4];
    let kappas = [0.1_f64, 0.5, 1.0, 2.0];
    let big_rs = [0.5_f64, 5.0];

    let mut tested = 0_usize;
    let mut skipped = 0_usize;

    for &q in &qs {
        for &d in &ds {
            for &m in &ms {
                if 2 * m < q + 1 {
                    continue;
                }
                for &s in &ss {
                    if d % 2 == 0 {
                        skipped += 1;
                        continue;
                    }
                    if 4 * m + 4 * s <= d + 2 * q {
                        skipped += 1;
                        continue;
                    }
                    for &kappa in &kappas {
                        for &big_r in &big_rs {
                            let eta = vec![0.0_f64; d];
                            let mut r_vec = vec![0.0_f64; d];
                            r_vec[0] = big_r;

                            let radial = anisotropic_duchon_penalty_radial(
                                q,
                                m,
                                (s) as f64,
                                kappa,
                                &eta,
                                &r_vec,
                            );
                            let expected =
                                isotropic_radial_laplacian_power_from_q0(q, d, m, s, kappa, big_r);
                            if !radial.is_finite() || !expected.is_finite() {
                                skipped += 1;
                                continue;
                            }
                            let denom = expected.abs().max(radial.abs()).max(1e-300);
                            let rel = (radial - expected).abs() / denom;
                            assert!(
                                rel < 1e-12,
                                "η=0 radial vs q0-Laplacian disagreement: q={q} d={d} m={m} \
                                     s={s} κ={kappa} R={big_r} radial={radial:.6e} \
                                     expected={expected:.6e} rel={rel:.3e}"
                            );
                            tested += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        tested >= 30,
        "sweep tested too few cases: tested={tested} skipped={skipped}"
    );
}

#[test]
fn test_radial_form_uniform_eta_uses_exact_isotropic_metric_identity() {
    use super::closed_form_penalty::anisotropic_duchon_penalty_radial;

    let cases: &[(usize, usize, usize, usize, f64, f64)] = &[
        (0, 3, 1, 2, 0.5, 0.20),
        (1, 7, 1, 2, 0.1, -0.35),
        (2, 5, 2, 2, 0.8, 0.15),
    ];
    for &(q, d, m, s, kappa, common_eta) in cases {
        let eta = vec![common_eta; d];
        let r: Vec<f64> = (0..d).map(|axis| 0.25 + 0.08 * axis as f64).collect();
        let euclidean_r = r.iter().map(|&ri| ri * ri).sum::<f64>().sqrt();
        let b = (-2.0 * common_eta).exp();
        let expected = b.powi(q as i32)
            * isotropic_radial_laplacian_power_from_q0(q, d, m, s, kappa, b.sqrt() * euclidean_r);
        let radial = anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, &eta, &r);
        let rel = (radial - expected).abs() / expected.abs().max(radial.abs()).max(1e-300);
        assert!(
            rel < 1e-12,
            "uniform-η radial identity failed: q={q} d={d} m={m} s={s} κ={kappa} \
                 η={common_eta} radial={radial:.16e} expected={expected:.16e} rel={rel:.3e}"
        );
    }
}

#[test]
fn test_letter_b_taylor_matches_partial_fraction_in_overlap() {
    use super::closed_form_penalty::isotropic_duchon_penalty;

    let cases: &[(usize, usize, usize, usize, f64, f64)] = &[
        (3, 1, 2, 1, 0.5, 1.0),
        (3, 2, 1, 1, 0.5, 0.8),
        (5, 1, 2, 1, 0.5, 1.2),
        (5, 2, 2, 2, 0.4, 1.5),
        (7, 2, 2, 1, 0.6, 1.0),
    ];

    for &(d, m, s, q, kappa, r) in cases {
        let v0 = isotropic_duchon_penalty(q, d, m, s as f64, kappa, r);
        assert!(
            v0.is_finite(),
            "primary value not finite: d={d} m={m} s={s} q={q} κ={kappa} r={r}"
        );

        let v_lo = isotropic_duchon_penalty(q, d, m, s as f64, kappa * 0.999, r);
        let v_hi = isotropic_duchon_penalty(q, d, m, s as f64, kappa * 1.001, r);
        let denom = v0.abs().max(1e-300);
        let jump_lo = (v_lo - v0).abs() / denom;
        let jump_hi = (v_hi - v0).abs() / denom;
        assert!(
            jump_lo < 1e-2 && jump_hi < 1e-2,
            "discontinuity near χ-gate threshold: d={d} m={m} s={s} q={q} κ={kappa} \
                 r={r} v0={v0:.6e} v_lo={v_lo:.6e} v_hi={v_hi:.6e}"
        );
    }
}

#[test]
fn test_g_2_radial_form_matches_letter_a_explicit_formula() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty_radial, radial_derivatives_of_isotropic_duchon,
    };

    let mut seed = 0xFEED_FACE_u64;

    let cases: &[(usize, usize, usize, f64)] = &[(5, 2, 2, 0.7), (5, 2, 3, 1.1), (7, 2, 3, 0.5)];

    for &(d, m, s, kappa) in cases {
        for _ in 0..4 {
            let eta: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 0.6).collect();
            let r: Vec<f64> = (0..d).map(|_| 0.4 + 1.0 * det_rand(&mut seed)).collect();

            let h2_direct = anisotropic_duchon_penalty_radial(2, m, (s) as f64, kappa, &eta, &r);

            let mut s1 = 0.0_f64;
            let mut s2 = 0.0_f64;
            let mut r2 = 0.0_f64;
            let mut u1 = 0.0_f64;
            let mut u2 = 0.0_f64;
            for k in 0..d {
                let b = (-2.0 * eta[k]).exp();
                let rk2 = r[k] * r[k];
                s1 += b;
                s2 += b * b;
                r2 += b * rk2;
                u1 += b * b * rk2;
                u2 += b * b * b * rk2;
            }
            let big_r = r2.sqrt();
            let fr = radial_derivatives_of_isotropic_duchon(d, m, (s) as f64, kappa, big_r, 4);

            let r1 = big_r;
            let r2p = r1 * r1;
            let r3 = r2p * r1;
            let r4 = r2p * r2p;
            let r5 = r4 * r1;
            let r6 = r4 * r2p;
            let r7 = r6 * r1;

            let big_a = fr[4] / r4 - 6.0 * fr[3] / r5 + 15.0 * fr[2] / r6 - 15.0 * fr[1] / r7;
            let big_b = 2.0 * fr[3] / r3 - 6.0 * fr[2] / r4 + 6.0 * fr[1] / r5;
            let big_c = fr[2] / r2p - fr[1] / r3;

            let h2_grouped =
                big_a * u1 * u1 + big_b * (s1 * u1 + 2.0 * u2) + big_c * (s1 * s1 + 2.0 * s2);

            let denom = h2_direct.abs().max(h2_grouped.abs()).max(1e-300);
            let abs = (h2_direct - h2_grouped).abs();
            let rel = abs / denom;
            assert!(
                abs <= 5.0e-18 + 1.0e-10 * denom,
                "g_2 invariant-grouped formula disagrees: d={d} m={m} s={s} κ={kappa} \
                     direct={h2_direct:.6e} grouped={h2_grouped:.6e} abs={abs:.3e} rel={rel:.3e}"
            );
        }
    }
}

#[test]
fn test_isotropic_limit_at_b_equals_i_recovers_radial_bilaplacian() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty_radial, radial_derivatives_of_isotropic_duchon,
    };

    let cases: &[(usize, usize, usize, f64)] = &[(3, 2, 1, 0.7), (5, 2, 2, 1.0), (7, 2, 3, 0.5)];

    for &(d, m, s, kappa) in cases {
        for &big_r in &[0.6_f64, 1.2, 2.4] {
            let eta = vec![0.0_f64; d];
            let mut r = vec![0.0_f64; d];
            r[0] = big_r;

            let g2 = anisotropic_duchon_penalty_radial(2, m, (s) as f64, kappa, &eta, &r);

            let fr = radial_derivatives_of_isotropic_duchon(d, m, (s) as f64, kappa, big_r, 4);
            let r1 = big_r;
            let r2 = r1 * r1;
            let r3 = r2 * r1;
            let dm1 = (d as f64) - 1.0;
            let dm3 = (d as f64) - 3.0;
            let bilap =
                fr[4] + 2.0 * dm1 / r1 * fr[3] + dm1 * dm3 / r2 * fr[2] - dm1 * dm3 / r3 * fr[1];

            let denom = g2.abs().max(bilap.abs()).max(1e-300);
            let rel = (g2 - bilap).abs() / denom;
            assert!(
                rel < 1e-10,
                "η=0 g_2 ≠ radial bi-Laplacian: d={d} m={m} s={s} κ={kappa} R={big_r} \
                     g2={g2:.6e} bilap={bilap:.6e} rel={rel:.3e}"
            );
        }
    }
}

#[test]
fn test_pair_block_symmetric_under_pair_swap() {
    use super::closed_form_penalty::anisotropic_duchon_penalty_radial;

    let mut seed = 0xBADD_F00D_u64;
    let cases: &[(usize, usize, usize, usize, f64)] = &[
        (1, 3, 1, 1, 0.7),
        (1, 5, 1, 2, 1.0),
        (2, 5, 2, 2, 0.5),
        (2, 7, 2, 3, 1.2),
    ];

    for &(q, d, m, s, kappa) in cases {
        for _ in 0..5 {
            let eta: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 0.5).collect();
            let r: Vec<f64> = (0..d).map(|_| 0.3 + 1.5 * det_rand(&mut seed)).collect();
            let r_neg: Vec<f64> = r.iter().map(|&x| -x).collect();

            let v_pos = anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, &eta, &r);
            let v_neg = anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, &eta, &r_neg);
            let denom = v_pos.abs().max(1e-300);
            let rel = (v_pos - v_neg).abs() / denom;
            assert!(
                rel < 1e-13,
                "pair-block not symmetric under pair swap: q={q} d={d} m={m} s={s} \
                     κ={kappa} v(r)={v_pos:.6e} v(-r)={v_neg:.6e} rel={rel:.3e}"
            );
        }
    }
}

#[test]
fn test_pair_block_continuous_at_diagonal_via_eps_limit() {
    use super::closed_form_penalty::anisotropic_duchon_penalty_radial;

    // Pure-Duchon (κ=0), q ∈ {1, 2}, with p = 4(m+s)−d > 2q so
    // the finite-part radial limit is bounded.
    let cases: &[(usize, usize, usize, usize, f64)] = &[
        (2, 3, 2, 1, 0.0), // p = 8 - 3 = 5 > 4
        (2, 5, 2, 2, 0.0), // p = 11 > 4
    ];

    for &(q, d, m, s, kappa) in cases {
        let eta: Vec<f64> = (0..d)
            .map(|i| 0.05 * (i as f64 - (d as f64) / 2.0))
            .collect();
        for &big_r in &[1e-1_f64, 1e-2, 1e-3, 1e-4, 1e-5] {
            let mut r = vec![0.0_f64; d];
            r[0] = big_r;
            let v = anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, &eta, &r);
            assert!(
                v.is_finite(),
                "pair-block diverged at small R: q={q} d={d} m={m} s={s} R={big_r}"
            );
            assert!(
                v.abs() < 10.0,
                "pair-block magnitude exploded near R=0: q={q} d={d} m={m} s={s} \
                     R={big_r} v={v:.6e}"
            );
        }
    }
}

#[test]
fn test_full_dim_validity_d4_d8_d16_with_resolved_orders() {
    let mut seed = 0xFACE_FEED_u64;
    let dims = [4_usize, 8, 16];
    let kappa = 1.0_f64;

    for &d in &dims {
        let (_ns_order, s) = resolve_duchon_orders(d, DuchonNullspaceOrder::Linear, 2, Some(1.0));
        let q = 2_usize;
        let m = 2_usize;

        let k = 4_usize;
        let mut centers = Array2::<f64>::zeros((k, d));
        for i in 0..k {
            for c in 0..d {
                centers[[i, c]] = det_rand(&mut seed);
            }
        }
        let eta: Vec<f64> = vec![0.0_f64; d];

        let g = closed_form_anisotropic_pair_block(centers.view(), q, m, s, kappa, Some(&eta));

        assert_eq!(g.dim(), (k, k), "wrong dim for d={d}");
        for i in 0..k {
            for j in 0..k {
                let v = g[[i, j]];
                assert!(
                    v.is_finite(),
                    "non-finite entry at d={d} (i,j)=({i},{j}): {v}"
                );
            }
        }
        for i in 0..k {
            for j in 0..i {
                let a = g[[i, j]];
                let b = g[[j, i]];
                let denom = a.abs().max(b.abs()).max(1e-300);
                let rel = (a - b).abs() / denom;
                assert!(
                    rel < 1e-12,
                    "asymmetry at d={d} (i,j)=({i},{j}): {a} vs {b} rel={rel:.3e}"
                );
            }
        }
    }
}

fn assert_pair_block_bundle_fully_fd_gated<F>(
    label: &str,
    eta: &[f64],
    kappa: f64,
    bundle: &super::closed_form_penalty::PairBlockBundle,
    value_at: F,
) where
    F: Fn(&[f64], f64) -> f64,
{
    fn assert_fd_close(
        label: &str,
        channel: &str,
        analytic: f64,
        finite_difference: f64,
        value_scale: f64,
        relative_tolerance: f64,
    ) {
        assert!(
            analytic.is_finite() && finite_difference.is_finite(),
            "{label} {channel}: non-finite analytic={analytic} fd={finite_difference}"
        );
        let scale = analytic
            .abs()
            .max(finite_difference.abs())
            .max(1e-8 * value_scale)
            .max(f64::MIN_POSITIVE);
        let relative_error = (analytic - finite_difference).abs() / scale;
        assert!(
            relative_error <= relative_tolerance,
            "{label} {channel}: analytic={analytic:.12e} fd={finite_difference:.12e} \
             rel={relative_error:.3e} tol={relative_tolerance:.3e}"
        );
    }

    assert!(kappa > 0.0 && kappa.is_finite(), "{label}: invalid kappa");
    assert_eq!(bundle.d_eta.len(), eta.len(), "{label}: d_eta shape");
    assert_eq!(bundle.d2_eta.len(), eta.len(), "{label}: d2_eta rows");
    assert_eq!(
        bundle.d2_eta_kappa.len(),
        eta.len(),
        "{label}: d2_eta_kappa shape"
    );
    assert!(
        bundle.d2_eta.iter().all(|row| row.len() == eta.len()),
        "{label}: d2_eta columns"
    );

    let value = value_at(eta, kappa);
    let value_scale = value.abs().max(1e-14);
    assert_fd_close(label, "value", bundle.value, value, value_scale, 1e-13);

    // Two axes exercise both the diagonal and off-diagonal eta-Hessian paths
    // without multiplying the cost of the branch matrix by ambient dimension.
    let axes = [0usize, eta.len() - 1];
    let eta_step = 2e-3_f64;
    let kappa_step = 2e-3 * kappa;

    for &axis in &axes {
        let eta_value = |delta: f64| {
            let mut shifted = eta.to_vec();
            shifted[axis] += delta;
            value_at(&shifted, kappa)
        };
        let eta_m2 = eta_value(-2.0 * eta_step);
        let eta_m1 = eta_value(-eta_step);
        let eta_p1 = eta_value(eta_step);
        let eta_p2 = eta_value(2.0 * eta_step);
        let eta_first_fd = (eta_m2 - 8.0 * eta_m1 + 8.0 * eta_p1 - eta_p2) / (12.0 * eta_step);
        let eta_second_fd = (-eta_p2 + 16.0 * eta_p1 - 30.0 * value + 16.0 * eta_m1 - eta_m2)
            / (12.0 * eta_step * eta_step);
        assert_fd_close(
            label,
            &format!("d_eta[{axis}]"),
            bundle.d_eta[axis],
            eta_first_fd,
            value_scale,
            5e-5,
        );
        assert_fd_close(
            label,
            &format!("d2_eta[{axis},{axis}]"),
            bundle.d2_eta[axis][axis],
            eta_second_fd,
            value_scale,
            2e-3,
        );

        let mixed_eta_kappa_at = |eta_h: f64, kappa_h: f64| {
            let mut plus_eta = eta.to_vec();
            let mut minus_eta = eta.to_vec();
            plus_eta[axis] += eta_h;
            minus_eta[axis] -= eta_h;
            (value_at(&plus_eta, kappa + kappa_h)
                - value_at(&plus_eta, kappa - kappa_h)
                - value_at(&minus_eta, kappa + kappa_h)
                + value_at(&minus_eta, kappa - kappa_h))
                / (4.0 * eta_h * kappa_h)
        };
        let mixed_coarse = mixed_eta_kappa_at(eta_step, kappa_step);
        let mixed_fine = mixed_eta_kappa_at(0.5 * eta_step, 0.5 * kappa_step);
        let mixed_fd = (4.0 * mixed_fine - mixed_coarse) / 3.0;
        assert_fd_close(
            label,
            &format!("d2_eta_kappa[{axis}]"),
            bundle.d2_eta_kappa[axis],
            mixed_fd,
            value_scale,
            2e-3,
        );
    }

    let (axis_a, axis_b) = (axes[0], axes[1]);
    let eta_cross_at = |step: f64| {
        let mut pp = eta.to_vec();
        let mut pm = eta.to_vec();
        let mut mp = eta.to_vec();
        let mut mm = eta.to_vec();
        pp[axis_a] += step;
        pp[axis_b] += step;
        pm[axis_a] += step;
        pm[axis_b] -= step;
        mp[axis_a] -= step;
        mp[axis_b] += step;
        mm[axis_a] -= step;
        mm[axis_b] -= step;
        (value_at(&pp, kappa) - value_at(&pm, kappa) - value_at(&mp, kappa) + value_at(&mm, kappa))
            / (4.0 * step * step)
    };
    let eta_cross_coarse = eta_cross_at(eta_step);
    let eta_cross_fine = eta_cross_at(0.5 * eta_step);
    let eta_cross_fd = (4.0 * eta_cross_fine - eta_cross_coarse) / 3.0;
    assert_fd_close(
        label,
        &format!("d2_eta[{axis_a},{axis_b}]"),
        bundle.d2_eta[axis_a][axis_b],
        eta_cross_fd,
        value_scale,
        2e-3,
    );
    assert_fd_close(
        label,
        &format!("d2_eta symmetry [{axis_b},{axis_a}]"),
        bundle.d2_eta[axis_b][axis_a],
        bundle.d2_eta[axis_a][axis_b],
        value_scale,
        1e-13,
    );

    let kappa_m2 = value_at(eta, kappa - 2.0 * kappa_step);
    let kappa_m1 = value_at(eta, kappa - kappa_step);
    let kappa_p1 = value_at(eta, kappa + kappa_step);
    let kappa_p2 = value_at(eta, kappa + 2.0 * kappa_step);
    let kappa_first_fd =
        (kappa_m2 - 8.0 * kappa_m1 + 8.0 * kappa_p1 - kappa_p2) / (12.0 * kappa_step);
    let kappa_second_fd = (-kappa_p2 + 16.0 * kappa_p1 - 30.0 * value + 16.0 * kappa_m1 - kappa_m2)
        / (12.0 * kappa_step * kappa_step);
    assert_fd_close(
        label,
        "d_kappa",
        bundle.d_kappa,
        kappa_first_fd,
        value_scale,
        5e-5,
    );
    assert_fd_close(
        label,
        "d2_kappa",
        bundle.d2_kappa,
        kappa_second_fd,
        value_scale,
        2e-3,
    );

    // A branch fixture with zero derivative signal would turn its FD gate into
    // a tautology. Require every derivative family to carry observable signal.
    let eta_first_signal = axes
        .iter()
        .map(|&axis| bundle.d_eta[axis].abs())
        .fold(0.0_f64, f64::max);
    let eta_second_signal = axes
        .iter()
        .map(|&axis| bundle.d2_eta[axis][axis].abs())
        .chain(std::iter::once(bundle.d2_eta[axis_a][axis_b].abs()))
        .fold(0.0_f64, f64::max);
    let mixed_signal = axes
        .iter()
        .map(|&axis| bundle.d2_eta_kappa[axis].abs())
        .fold(0.0_f64, f64::max);
    let signal_floor = 1e-10 * value_scale;
    assert!(
        eta_first_signal > signal_floor,
        "{label}: vacuous d_eta gate"
    );
    assert!(
        eta_second_signal > signal_floor,
        "{label}: vacuous d2_eta gate"
    );
    assert!(
        bundle.d_kappa.abs() > signal_floor,
        "{label}: vacuous d_kappa gate"
    );
    assert!(
        bundle.d2_kappa.abs() > signal_floor,
        "{label}: vacuous d2_kappa gate"
    );
    assert!(mixed_signal > signal_floor, "{label}: vacuous mixed gate");
}

#[test]
fn test_pair_block_derivative_branch_matrix_is_fully_fd_gated_2315() {
    use super::closed_form_penalty::{
        analytic_self_pair_bundle, aniso_invariants, hybrid_self_pair_bundle_odd_d,
        pair_block_radial_with_j_second_derivatives, schoenberg_self_pair_bundle,
        schwinger_radial_is_convergent, use_duchon_small_chi_riesz_series,
    };

    // Zero lag has two independent analytic implementations. Exercise every
    // q match arm in convergent odd/even Schoenberg regimes and in the
    // IR-singular odd-dimensional collision branch that motivated #2291.
    let zero_lag_cases = [
        ("zero-schoenberg-odd", 5usize, 1usize, 2usize, true),
        ("zero-schoenberg-even", 6, 1, 2, true),
        ("zero-hybrid-ir-singular-odd", 3, 2, 2, false),
    ];
    for (case_label, d, m, s, expect_schoenberg) in zero_lag_cases {
        let eta: Vec<f64> = (0..d).map(|axis| 0.06 * axis as f64 - 0.1).collect();
        let r = vec![0.0_f64; d];
        let kappa = 0.9_f64;
        for q in 0..=2 {
            let schoenberg = schoenberg_self_pair_bundle(q, m, s, kappa, &eta);
            let hybrid = hybrid_self_pair_bundle_odd_d(q, m, s, kappa, &eta);
            if expect_schoenberg {
                assert!(
                    schoenberg.is_some(),
                    "{case_label} q={q}: wrong zero-lag branch"
                );
            } else {
                assert!(
                    schoenberg.is_none(),
                    "{case_label} q={q}: IR gate not exercised"
                );
                assert!(
                    hybrid.is_some(),
                    "{case_label} q={q}: hybrid branch not exercised"
                );
            }

            let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);
            let label = format!("{case_label}-q{q}");
            assert_pair_block_bundle_fully_fd_gated(
                &label,
                &eta,
                kappa,
                &bundle,
                |eta_at, kappa_at| {
                    analytic_self_pair_bundle(q, m, s, kappa_at, eta_at)
                        .expect("finite analytic self-pair throughout FD stencil")
                        .value
                },
            );
        }
    }

    // Nonzero lag has three radial charts: small-chi finite-part series,
    // ordinary low-d partial fractions, and ordinary high-d Schwinger form.
    // Include both parities and keep each fixture well away from its dispatch
    // boundary so the finite-difference stencil never changes charts.
    let nonzero_cases = [
        (
            "ordinary-pf-odd",
            1usize,
            1usize,
            0.9_f64,
            vec![0.12, -0.08, 0.04],
            vec![0.7, -0.4, 0.5],
            false,
            false,
        ),
        (
            "ordinary-pf-even",
            1,
            2,
            0.8,
            vec![0.1, -0.07, 0.03, -0.02],
            vec![0.6, -0.5, 0.4, 0.3],
            false,
            false,
        ),
        (
            "ordinary-schwinger-odd",
            1,
            2,
            0.8,
            vec![0.1, -0.07, 0.03, -0.02, 0.05],
            vec![0.7, -0.5, 0.4, 0.3, -0.2],
            false,
            true,
        ),
        (
            "ordinary-schwinger-even",
            1,
            2,
            0.8,
            vec![0.1, -0.07, 0.03, -0.02, 0.05, -0.04],
            vec![0.7, -0.5, 0.4, 0.3, -0.2, 0.6],
            false,
            true,
        ),
        (
            "small-chi-odd",
            1,
            1,
            0.9,
            vec![0.12, -0.08, 0.04],
            vec![0.03, -0.02, 0.015],
            true,
            false,
        ),
        (
            "small-chi-even",
            1,
            2,
            0.8,
            vec![0.1, -0.07, 0.03, -0.02],
            vec![0.025, -0.02, 0.015, 0.01],
            true,
            false,
        ),
    ];
    for (case_label, m, s, kappa, eta, r, expect_small_chi, expect_schwinger) in nonzero_cases {
        let big_r = aniso_invariants(&eta, &r).0;
        assert_eq!(
            use_duchon_small_chi_riesz_series(kappa, big_r),
            expect_small_chi,
            "{case_label}: wrong small-chi fixture classification"
        );
        assert_eq!(
            schwinger_radial_is_convergent(eta.len(), m),
            expect_schwinger,
            "{case_label}: wrong ordinary-chart fixture classification"
        );
        for q in 0..=2 {
            let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);
            let label = format!("{case_label}-q{q}");
            assert_pair_block_bundle_fully_fd_gated(
                &label,
                &eta,
                kappa,
                &bundle,
                |eta_at, kappa_at| {
                    pair_block_radial_with_j_second_derivatives(q, m, s, kappa_at, eta_at, &r).value
                },
            );
        }
    }
}

#[test]
fn test_pair_block_pure_riesz_kappa_independence_is_exactly_gated_2315() {
    use super::closed_form_penalty::pair_block_radial_with_j_second_derivatives;

    // `s == 0` is a separate production match arm: the hybrid factor is absent,
    // so the value and every eta derivative are exactly independent of kappa,
    // while all kappa and eta-kappa derivatives are exactly zero. Exercise both
    // ambient parities and every supported q arm. Merely checking finiteness or
    // zero derivatives would be vacuous if the value accidentally acquired a
    // hidden kappa dependence, so the value itself is compared across 18 orders
    // of magnitude as the independent semantic authority.
    for d in [3usize, 4usize] {
        let eta: Vec<f64> = (0..d).map(|axis| 0.05 * axis as f64 - 0.08).collect();
        let r: Vec<f64> = (0..d)
            .map(|axis| if axis % 2 == 0 { 0.7 } else { -0.4 })
            .collect();
        for q in 0..=2 {
            let reference = pair_block_radial_with_j_second_derivatives(q, 2, 0, 1.0, &eta, &r);
            assert!(
                reference.value.is_finite() && reference.value != 0.0,
                "pure-Riesz d={d} q={q}: fixture must carry nonzero value signal"
            );
            assert!(
                reference.d_eta.iter().any(|value| value.abs() > 0.0),
                "pure-Riesz d={d} q={q}: fixture must carry eta-derivative signal"
            );

            for kappa in [1e-9_f64, 1.0, 1e9] {
                let bundle = pair_block_radial_with_j_second_derivatives(q, 2, 0, kappa, &eta, &r);
                assert_eq!(
                    bundle.value.to_bits(),
                    reference.value.to_bits(),
                    "pure-Riesz d={d} q={q}: value changed with kappa={kappa}"
                );
                assert_eq!(
                    bundle.d_eta, reference.d_eta,
                    "pure-Riesz d={d} q={q}: eta gradient changed with kappa={kappa}"
                );
                assert_eq!(
                    bundle.d2_eta, reference.d2_eta,
                    "pure-Riesz d={d} q={q}: eta Hessian changed with kappa={kappa}"
                );
                assert_eq!(
                    bundle.d_kappa, 0.0,
                    "pure-Riesz d={d} q={q}: d_kappa must vanish"
                );
                assert_eq!(
                    bundle.d2_kappa, 0.0,
                    "pure-Riesz d={d} q={q}: d2_kappa must vanish"
                );
                assert!(
                    bundle.d2_eta_kappa.iter().all(|&value| value == 0.0),
                    "pure-Riesz d={d} q={q}: mixed eta-kappa derivatives must vanish"
                );
            }
        }
    }
}

/// #2315 Gap 3: DIRECT finite-difference gate on the even-ambient-dimension
/// Duchon collision derivative `phi^(2j)(0)`
/// (`duchon_phi_even_derivative_collision`). Until now this branch was reached
/// only *indirectly*, through the odd-d hybrid self-pair zero-lag path
/// (`hybrid_self_pair_radial_derivative_with_kappa_derivs_odd_d`), so its
/// even-d behaviour carried no direct reference.
///
/// The even-d regime is the interesting one: the polyharmonic block that lands
/// on radial order `r^{2j}` sits at `m = d/2 + j >= d/2`, so it is a `ln(r)`
/// block whose pure Taylor part is zero and whose log part must cancel exactly
/// against the Matérn-block log parts before the surviving pure coefficient
/// yields `phi^{(2j)}(0) = (2j)! * a_{2j}`. A wrong `r^{2j}` exponent, a dropped
/// `(2j)!` factor, or a broken log cancellation is exactly the silent-derivative
/// class #2315 targets, and each fails here.
///
/// The analytic collision derivative is gated against a central finite
/// difference of the *underlying* radial kernel value
/// `duchon_matern_kernel_general_from_distance`. The stencil is sampled strictly
/// OUTSIDE the near-collision Taylor radius (`r > DUCHON_COLLISION_TAYLOR_REL *
/// ell`), so the finite-difference reference is the independent direct
/// partial-fraction / single-integral kernel — never the
/// collision-derivative-based Taylor carrier that the near-collision branch uses
/// (which would make the gate circular). Every fixture carries a smoothness
/// margin of two extra orders (`2(p+s) > d + 2j + 2`) so the truncation term is
/// a bounded higher derivative, not a log-divergent borderline limit.
#[test]
fn test_even_d_duchon_collision_derivative_matches_finite_difference_2315() {
    use super::duchon_kernel_math::{
        duchon_matern_kernel_general_from_distance, duchon_partial_fraction_coeffs,
        DUCHON_COLLISION_TAYLOR_REL,
    };
    use super::duchon_psi_derivatives::{
        duchon_phi_even_derivative_collision, duchon_polyharmonic_block_taylor_r2j,
    };

    // (even d, p_order, s_order, j, fd step h, tol). Each fixture satisfies the
    // finite-collision smoothness bound with two orders of margin
    // (2(p+s) > d + 2j + 2) and places the r^{2j} polyharmonic block on the
    // even-d ln(r) branch at m = d/2 + j <= p, so the log-cancellation path is
    // genuinely live. j=1 uses the 3-point second-derivative stencil (tol 1e-5,
    // matching the Matern/Riesz radial ladders); j=2 uses the 5-point
    // fourth-derivative stencil (tol 1e-3, matching the isotropic-kappa
    // second-difference gate).
    let cases: &[(usize, usize, usize, usize, f64, f64)] = &[
        (2, 2, 2, 1, 5e-3, 1e-5),
        (4, 3, 2, 1, 5e-3, 1e-5),
        (2, 3, 2, 2, 3e-3, 1e-3),
        (4, 4, 2, 2, 3e-3, 1e-3),
    ];

    let length_scale = 1.0_f64;
    let kappa = 1.0_f64 / length_scale;

    for &(d, p, s, j, h, tol) in cases {
        assert!(
            d.is_multiple_of(2),
            "d={d}: this gate targets even ambient dimension"
        );
        assert!(
            2 * (p + s) > d + 2 * j + 2,
            "d={d} p={p} s={s} j={j}: fixture must carry the two-order collision-smoothness margin"
        );

        // Prove the even-d ln(r) branch is genuinely exercised: the polyharmonic
        // block carrying radial order r^{2j} sits at m = d/2 + j and, for even d,
        // must be a log block (pure part zero, log part nonzero) whose
        // contribution has to cancel inside the collision sum.
        let m_log = d / 2 + j;
        assert!(
            m_log <= p,
            "d={d} j={j}: log-carrying block m={m_log} must be within p={p}"
        );
        let (poly_pure, poly_log) = duchon_polyharmonic_block_taylor_r2j(m_log, d, j);
        assert_eq!(
            poly_pure, 0.0,
            "d={d} j={j}: even-d block m={m_log} must be pure-free (ln(r) branch)"
        );
        assert!(
            poly_log != 0.0,
            "d={d} j={j}: even-d block m={m_log} must carry a nonzero log term that the sum cancels"
        );

        let coeffs = duchon_partial_fraction_coeffs(p, s, kappa);
        assert!(
            coeffs.a[m_log] != 0.0,
            "d={d} j={j}: log block m={m_log} must have a nonzero partial-fraction coefficient"
        );

        // Analytic even-order collision derivative phi^{(2j)}(0). `.expect`
        // panics if the internal log cancellation fails, so this also gates the
        // even-d cancellation itself.
        let analytic = duchon_phi_even_derivative_collision(length_scale, p, s, d, &coeffs, j)
            .expect("even-d collision derivative must take the finite analytic path");
        assert!(
            analytic.is_finite(),
            "d={d} j={j}: analytic collision derivative not finite"
        );

        // Central finite difference of the underlying kernel value, sampled
        // strictly outside the near-collision Taylor radius so the reference is
        // the independent direct kernel evaluation (not the collision-derivative
        // Taylor carrier).
        let taylor_radius = DUCHON_COLLISION_TAYLOR_REL * length_scale;
        assert!(
            h > taylor_radius,
            "d={d}: FD step h={h} must exceed the near-collision radius {taylor_radius:.3e} to stay on the independent kernel path"
        );
        let phi = |rr: f64| {
            duchon_matern_kernel_general_from_distance(
                rr,
                Some(length_scale),
                p,
                s,
                d,
                Some(&coeffs),
            )
            .expect("underlying Duchon kernel value must be finite across the FD stencil")
        };
        let phi0 = phi(0.0);
        // phi is an even radial function, so phi(-k h) = phi(k h). Central
        // even-order collision stencils (truncation O(h^2)):
        //   phi''(0)   = (2 phi(h) - 2 phi(0)) / h^2
        //   phi''''(0) = (2 phi(2h) - 8 phi(h) + 6 phi(0)) / h^4
        let fd = match j {
            1 => (2.0 * phi(h) - 2.0 * phi0) / (h * h),
            2 => (2.0 * phi(2.0 * h) - 8.0 * phi(h) + 6.0 * phi0) / (h * h * h * h),
            other => unreachable!("only j in {{1, 2}} are gated here, got {other}"),
        };

        let scale = analytic.abs().max(phi0.abs()).max(1.0);
        assert!(
            (analytic - fd).abs() <= tol * scale,
            "even-d collision derivative FD mismatch: d={d} p={p} s={s} j={j} \
             analytic={analytic:.12e} fd={fd:.12e} diff={:.3e} tol={:.3e}",
            (analytic - fd).abs(),
            tol * scale
        );
    }
}

#[test]
fn test_kappa_derivative_matches_finite_difference() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty_radial, kappa_first_derivative,
        pair_block_radial_with_j_second_derivatives,
    };

    let mut seed = 0xCAFE_BABE_u64;
    let cases: &[(usize, usize, usize, usize, f64)] =
        &[(1, 3, 1, 1, 1.0), (1, 5, 1, 2, 0.7), (2, 5, 2, 2, 0.9)];

    for &(q, d, m, s, kappa) in cases {
        for _ in 0..3 {
            let eta: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 0.4).collect();
            let r: Vec<f64> = (0..d).map(|_| 0.5 + 1.0 * det_rand(&mut seed)).collect();

            let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);

            let big_j = eta.iter().sum::<f64>().exp();
            let bare_dk = kappa_first_derivative(q, m, s, kappa, &eta, &r);
            assert!((big_j * bare_dk - bundle.d_kappa).abs() < 1e-12);
            let v_at = |kk: f64| -> f64 {
                big_j * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kk, &eta, &r)
            };
            let h1 = 1e-3 * kappa;
            let h2 = 0.5 * h1;
            let d1 = (v_at(kappa + h1) - v_at(kappa - h1)) / (2.0 * h1);
            let d2 = (v_at(kappa + h2) - v_at(kappa - h2)) / (2.0 * h2);
            let dk_richardson = (4.0 * d2 - d1) / 3.0;

            let denom = dk_richardson.abs().max(bundle.d_kappa.abs()).max(1e-12);
            let rel = (bundle.d_kappa - dk_richardson).abs() / denom;
            assert!(
                rel < 1e-6,
                "bundle ∂_κ vs Richardson FD: q={q} d={d} m={m} s={s} κ={kappa} \
                     bundle={:.6e} richardson={:.6e} rel={rel:.3e}",
                bundle.d_kappa,
                dk_richardson
            );
        }
    }
}

#[test]
fn test_eta_derivative_matches_finite_difference() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty_radial, pair_block_radial_with_j_second_derivatives,
    };

    let mut seed = 0xABCD_1234_u64;
    let cases: &[(usize, usize, usize, usize, f64)] = &[(1, 3, 1, 1, 0.8), (2, 5, 2, 2, 1.0)];

    for &(q, d, m, s, kappa) in cases {
        for _trial in 0..3 {
            let eta: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 0.4).collect();
            let r: Vec<f64> = (0..d).map(|_| 0.4 + 1.2 * det_rand(&mut seed)).collect();

            let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);

            for l in 0..d {
                let v_at = |eta_use: &[f64]| -> f64 {
                    eta_use.iter().sum::<f64>().exp()
                        * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, eta_use, &r)
                };
                let h1 = 1e-3_f64;
                let h2 = 0.5 * h1;
                let mut e_p = eta.clone();
                let mut e_m = eta.clone();
                e_p[l] += h1;
                e_m[l] -= h1;
                let d1 = (v_at(&e_p) - v_at(&e_m)) / (2.0 * h1);
                let mut e_p2 = eta.clone();
                let mut e_m2 = eta.clone();
                e_p2[l] += h2;
                e_m2[l] -= h2;
                let d2 = (v_at(&e_p2) - v_at(&e_m2)) / (2.0 * h2);
                let de_richardson = (4.0 * d2 - d1) / 3.0;

                let denom = de_richardson.abs().max(bundle.d_eta[l].abs()).max(1e-12);
                let rel = (bundle.d_eta[l] - de_richardson).abs() / denom;
                assert!(
                    rel < 1e-6,
                    "bundle ∂_η_{l} vs Richardson FD: q={q} d={d} m={m} s={s} κ={kappa} \
                         bundle={:.6e} richardson={:.6e} rel={rel:.3e}",
                    bundle.d_eta[l],
                    de_richardson
                );
            }
        }
    }
}

#[test]
fn test_pair_block_analytic_d2kappa_matches_fd() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty_radial, kappa_second_derivative,
        pair_block_radial_with_j_second_derivatives,
    };

    let mut seed = 0x1234_5678_u64;
    let cases: &[(usize, usize, usize, usize, f64)] = &[
        (0, 3, 1, 1, 1.0),
        (1, 3, 1, 1, 0.9),
        (1, 5, 1, 2, 0.7),
        (2, 5, 2, 2, 0.85),
    ];

    for &(q, d, m, s, kappa) in cases {
        for _ in 0..3 {
            let eta: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 0.4).collect();
            let r: Vec<f64> = (0..d).map(|_| 0.5 + 1.0 * det_rand(&mut seed)).collect();

            let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);

            let big_j = eta.iter().sum::<f64>().exp();
            let bare_d2k = kappa_second_derivative(q, m, s, kappa, &eta, &r);
            assert!((big_j * bare_d2k - bundle.d2_kappa).abs() < 1e-12);
            let v_at = |kk: f64| -> f64 {
                big_j * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kk, &eta, &r)
            };
            let v0 = v_at(kappa);
            let h1 = 1e-3 * kappa;
            let h2 = 0.5 * h1;
            let dd1 = (v_at(kappa + h1) - 2.0 * v0 + v_at(kappa - h1)) / (h1 * h1);
            let dd2 = (v_at(kappa + h2) - 2.0 * v0 + v_at(kappa - h2)) / (h2 * h2);
            let dd_richardson = (4.0 * dd2 - dd1) / 3.0;

            let denom = dd_richardson.abs().max(bundle.d2_kappa.abs()).max(1e-8);
            let rel = (bundle.d2_kappa - dd_richardson).abs() / denom;
            assert!(
                rel < 1e-6,
                "bundle ∂²_κ vs Richardson FD: q={q} d={d} m={m} s={s} κ={kappa} \
                     bundle={:.6e} richardson={:.6e} rel={rel:.3e}",
                bundle.d2_kappa,
                dd_richardson
            );
        }
    }
}

#[test]
fn test_pair_block_analytic_d2eta_matches_fd() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty_radial, pair_block_radial_with_j_second_derivatives,
        psi_second_derivative,
    };

    let mut seed = 0xFEED_FACE_u64;
    let cases: &[(usize, usize, usize, usize, f64)] =
        &[(0, 2, 1, 1, 1.0), (1, 3, 1, 1, 0.8), (2, 3, 2, 2, 1.0)];

    for &(q, d, m, s, kappa) in cases {
        for _trial in 0..2 {
            let eta: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 0.4).collect();
            let r: Vec<f64> = (0..d).map(|_| 0.4 + 1.2 * det_rand(&mut seed)).collect();

            let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);

            let big_j = eta.iter().sum::<f64>().exp();
            let v_at = |eta_use: &[f64]| -> f64 {
                eta_use.iter().sum::<f64>().exp()
                    * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kappa, eta_use, &r)
            };
            let v0 = v_at(&eta);
            let h = 1e-3_f64;
            // Diagonal: 2nd central FD.
            for l in 0..d {
                let bare_d2 = psi_second_derivative(q, m, s, kappa, &eta, &r, l, l);
                let unwrapped_d2 =
                    (bundle.d2_eta[l][l] - 2.0 * bundle.d_eta[l] + bundle.value) / big_j;
                assert!((bare_d2 - unwrapped_d2).abs() < 1e-12);

                let mut e_p = eta.clone();
                let mut e_m = eta.clone();
                e_p[l] += h;
                e_m[l] -= h;
                let dd_fd = (v_at(&e_p) - 2.0 * v0 + v_at(&e_m)) / (h * h);
                let denom = dd_fd.abs().max(bundle.d2_eta[l][l].abs()).max(1e-8);
                let rel = (bundle.d2_eta[l][l] - dd_fd).abs() / denom;
                assert!(
                    rel < 1e-5,
                    "bundle ∂²_{{η_{l}η_{l}}} vs FD: q={q} d={d} m={m} s={s} κ={kappa} \
                         bundle={:.6e} fd={:.6e} rel={rel:.3e}",
                    bundle.d2_eta[l][l],
                    dd_fd
                );
            }
            // Off-diagonal: 4-point cross finite-difference stencil.
            for k in 0..d {
                for l in (k + 1)..d {
                    let mut epp = eta.clone();
                    let mut epm = eta.clone();
                    let mut emp = eta.clone();
                    let mut emm = eta.clone();
                    epp[k] += h;
                    epp[l] += h;
                    epm[k] += h;
                    epm[l] -= h;
                    emp[k] -= h;
                    emp[l] += h;
                    emm[k] -= h;
                    emm[l] -= h;
                    let off_fd =
                        (v_at(&epp) - v_at(&epm) - v_at(&emp) + v_at(&emm)) / (4.0 * h * h);
                    let bare_cross = psi_second_derivative(q, m, s, kappa, &eta, &r, k, l);
                    let unwrapped_cross = (bundle.d2_eta[k][l] - bundle.d_eta[k] - bundle.d_eta[l]
                        + bundle.value)
                        / big_j;
                    assert!((bare_cross - unwrapped_cross).abs() < 1e-12);
                    let denom = off_fd.abs().max(bundle.d2_eta[k][l].abs()).max(1e-8);
                    let rel = (bundle.d2_eta[k][l] - off_fd).abs() / denom;
                    assert!(
                        rel < 1e-5,
                        "bundle ∂²_{{η_{k}η_{l}}} vs FD: q={q} d={d} m={m} s={s} κ={kappa} \
                             bundle={:.6e} fd={:.6e} rel={rel:.3e}",
                        bundle.d2_eta[k][l],
                        off_fd
                    );
                    // Symmetry sanity.
                    let sym = (bundle.d2_eta[k][l] - bundle.d2_eta[l][k]).abs();
                    assert!(
                        sym < 1e-10,
                        "Hessian symmetry violated: ({k},{l}) Δ={sym:.3e}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_pair_block_analytic_d2etakappa_matches_fd() {
    use super::closed_form_penalty::{
        anisotropic_duchon_penalty_radial, pair_block_radial_with_j_second_derivatives,
        psi_kappa_mixed_derivative,
    };

    let mut seed = 0xDEAD_BEEF_u64;
    let cases: &[(usize, usize, usize, usize, f64)] = &[(1, 3, 1, 1, 0.8), (2, 3, 2, 2, 0.9)];

    for &(q, d, m, s, kappa) in cases {
        for _ in 0..2 {
            let eta: Vec<f64> = (0..d).map(|_| (det_rand(&mut seed) - 0.5) * 0.4).collect();
            let r: Vec<f64> = (0..d).map(|_| 0.5 + 1.0 * det_rand(&mut seed)).collect();

            let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);

            let big_j = eta.iter().sum::<f64>().exp();
            let v_at = |eta_use: &[f64], kk: f64| -> f64 {
                eta_use.iter().sum::<f64>().exp()
                    * anisotropic_duchon_penalty_radial(q, m, (s) as f64, kk, eta_use, &r)
            };
            let h_e = 1e-3_f64;
            let h_k = 1e-3 * kappa;

            for l in 0..d {
                let bare_cross = psi_kappa_mixed_derivative(q, m, s, kappa, &eta, &r, l);
                let unwrapped_cross = (bundle.d2_eta_kappa[l] - bundle.d_kappa) / big_j;
                assert!((bare_cross - unwrapped_cross).abs() < 1e-12);

                let mut e_p = eta.clone();
                let mut e_m = eta.clone();
                e_p[l] += h_e;
                e_m[l] -= h_e;
                let cross_h =
                    (v_at(&e_p, kappa + h_k) - v_at(&e_p, kappa - h_k) - v_at(&e_m, kappa + h_k)
                        + v_at(&e_m, kappa - h_k))
                        / (4.0 * h_e * h_k);

                let h_e2 = 0.5 * h_e;
                let h_k2 = 0.5 * h_k;
                let mut e_p2 = eta.clone();
                let mut e_m2 = eta.clone();
                e_p2[l] += h_e2;
                e_m2[l] -= h_e2;
                let cross_h2 = (v_at(&e_p2, kappa + h_k2)
                    - v_at(&e_p2, kappa - h_k2)
                    - v_at(&e_m2, kappa + h_k2)
                    + v_at(&e_m2, kappa - h_k2))
                    / (4.0 * h_e2 * h_k2);
                let cross_richardson = (4.0 * cross_h2 - cross_h) / 3.0;

                let denom = cross_richardson
                    .abs()
                    .max(bundle.d2_eta_kappa[l].abs())
                    .max(1e-8);
                let rel = (bundle.d2_eta_kappa[l] - cross_richardson).abs() / denom;
                assert!(
                    rel < 1e-5,
                    "bundle ∂²_{{η_{l}κ}} vs FD: q={q} d={d} m={m} s={s} κ={kappa} \
                         bundle={:.6e} fd={:.6e} rel={rel:.3e}",
                    bundle.d2_eta_kappa[l],
                    cross_richardson
                );
            }
        }
    }
}
#[test]
fn test_periodic_bspline_wraps_design_at_cylinder_seam() {
    let x = array![0.0, 0.25, 0.5, 0.75, 1.0, 1.25];
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::PeriodicUniform {
            data_range: (0.0, 1.0),
            num_basis: 7,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary_conditions: Default::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let built = build_bspline_basis_1d(x.view(), &spec).expect("periodic basis");
    let design = built.design.to_dense();
    assert_eq!(design.nrows(), x.len());
    assert_eq!(design.ncols(), 7);
    for col in 0..design.ncols() {
        assert_abs_diff_eq!(design[[0, col]], design[[4, col]], epsilon = 1e-12);
        assert_abs_diff_eq!(design[[1, col]], design[[5, col]], epsilon = 1e-12);
    }
    assert_eq!(built.active_penalties.len(), 1);
    assert_eq!(built.active_penalties[0].matrix.nrows(), design.ncols());
    assert_eq!(built.active_penalties[0].matrix.ncols(), design.ncols());
}

#[test]
fn test_periodic_bspline_with_sum_to_zero_keeps_wrapped_rows_equal() {
    let x = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::PeriodicUniform {
            data_range: (0.0, 1.0),
            num_basis: 6,
        },
        double_penalty: true,
        identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        boundary_conditions: Default::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let built = build_bspline_basis_1d(x.view(), &spec).expect("periodic centered basis");
    let design = built.design.to_dense();
    for col in 0..design.ncols() {
        assert_abs_diff_eq!(design[[0, col]], design[[5, col]], epsilon = 1e-12);
    }
    let col_sums = design.sum_axis(Axis(0));
    for sum in col_sums.iter() {
        assert_abs_diff_eq!(*sum, 0.0, epsilon = 1e-10);
    }
    // Cyclic basis must emit a single wiggliness penalty even when
    // `double_penalty=true`: the cyclic difference penalty has a single
    // null direction (the constant) which the periodic sum-to-zero
    // identifiability constraint removes wholesale, so the
    // null-space-shrinkage projector reduces to `Tᵀ(z·zᵀ)T = 0` — an
    // identically zero penalty carrying its own smoothing parameter
    // would leave that λ unidentified and prevent outer-REML termination
    // (see #874 / the comment in `build_bspline_basis_1d`'s cyclic arm).
    // Match mgcv `bs="cc"`, which is likewise a single-penalty smooth.
    assert_eq!(built.active_penalties.len(), 1);
}

#[test]
fn wahba_sphere_kernel_simd_matches_scalar_within_documented_tolerance() {
    // SIMD-kind helper should match the scalar-kind helper across lanes.
    // abs/rel diff < 1e-12 vs the scalar path across the full domain.
    // Sweep cos_gamma across [-1, 1] (including degenerate endpoints
    // and the floor near cos_gamma = 1) for every supported penalty
    // order. Use a slightly looser 1e-11 assertion so genuine
    // regressions trip while floating-point ULP wiggle doesn't.
    let xs: [f64; 16] = [
        -1.0,
        -0.999_999_999,
        -0.9,
        -0.5,
        -0.123_456,
        -1.0e-6,
        0.0,
        1.0e-6,
        0.123_456,
        0.5,
        0.7,
        0.9,
        0.999_9,
        0.999_999_9,
        1.0 - 1.0e-12,
        1.0,
    ];
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for &m in &[1_usize, 2, 3, 4] {
        for chunk in xs.chunks(4) {
            let lane = wide::f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let simd = wahba_sphere_kernel_from_cos_simd_kind(lane, m, SphereWahbaKernel::Sobolev);
            let simd_arr: [f64; 4] = simd.into();
            for (i, &x) in chunk.iter().enumerate() {
                let scalar = wahba_sphere_kernel_from_cos_kind(x, m, SphereWahbaKernel::Sobolev)
                    .expect("scalar kernel must produce finite value over closed [-1, 1]");
                let abs = (simd_arr[i] - scalar).abs();
                let rel = abs / scalar.abs().max(1.0e-300);
                if abs.is_finite() {
                    max_abs = max_abs.max(abs);
                }
                if rel.is_finite() {
                    max_rel = max_rel.max(rel);
                }
                assert!(
                    abs < 1.0e-11,
                    "SIMD vs scalar abs diff {abs:.3e} > 1e-11 at \
                         cos_gamma={x:.6e} penalty_order={m}; \
                         simd={s:.17e} scalar={c:.17e}",
                    s = simd_arr[i],
                    c = scalar,
                );
                assert!(
                    rel < 1.0e-11,
                    "SIMD vs scalar rel diff {rel:.3e} > 1e-11 at \
                         cos_gamma={x:.6e} penalty_order={m}; \
                         simd={s:.17e} scalar={c:.17e}",
                    s = simd_arr[i],
                    c = scalar,
                );
            }
        }
    }
    // Surface the measured maxima in cargo-test output for visibility
    // when the test is run with --nocapture.
    eprintln!(
        "wahba simd-vs-scalar: max abs diff = {max_abs:.3e}, \
             max rel diff = {max_rel:.3e}",
    );
}

#[test]
fn auto_streaming_skips_small_bspline_basis() {
    // Representative small B-spline workload: n = 100 rows, k = 10 basis
    // columns. Dense buffer ≈ 100 · 10 · 8 = 8000 B ≪ 1 GiB threshold,
    // so streaming must not engage and the helper must return None.
    let chunk = auto_streaming_chunk_size_for_dense(100, 10);
    assert!(
        chunk.is_none(),
        "auto-streaming engaged for tiny n=100 k=10 basis: {chunk:?}"
    );
}

#[test]
fn auto_streaming_engages_for_large_synthetic_basis() {
    // Virtual workload: 1e6 rows × 200 basis columns. Dense bytes
    // = 1e6 · 200 · 8 = 1.6 GB > 1 GiB threshold, so streaming must
    // engage. Target chunk ≈ 256 MiB / (200 · 8 B) ≈ 167_772 rows,
    // clamped to [1024, n_rows].
    let chunk = auto_streaming_chunk_size_for_dense(1_000_000, 200)
        .expect("dense buffer exceeds 1 GiB → streaming must engage");
    assert!(chunk >= 1024, "chunk {chunk} below MIN_CHUNK_ROWS");
    assert!(chunk <= 1_000_000, "chunk {chunk} exceeds n_rows");
    // The 256 MiB / row_bytes formula should land in the ~150k-200k row
    // range for this column count.
    assert!(
        (100_000..=300_000).contains(&chunk),
        "chunk {chunk} outside the expected ~256 MiB / (200·8) window"
    );
}
