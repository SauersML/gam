//! Unit gates for the exact penalty-map invariance and the judged subspace
//! (#2676).

use super::*;
use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2, array};

fn penalty(local: Array2<f64>, total_dim: usize) -> CanonicalPenalty {
    let block = local.nrows();
    let (eigenvalues, eigenvectors) = {
        use gam_linalg::faer_ndarray::FaerEigh;
        local.eigh(faer::Side::Lower).expect("symmetric eigh")
    };
    let positive: Vec<f64> = eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > 1e-12)
        .collect();
    let rank = positive.len();
    let mut root = Array2::<f64>::zeros((rank, block));
    let mut next = 0usize;
    for index in 0..block {
        if eigenvalues[index] > 1e-12 {
            let scale = eigenvalues[index].sqrt();
            for column in 0..block {
                root[[next, column]] = scale * eigenvectors[[column, index]];
            }
            next += 1;
        }
    }
    CanonicalPenalty {
        root,
        col_range: 0..block,
        total_dim,
        nullity: block - rank,
        local,
        prior_mean: Array1::zeros(block),
        positive_eigenvalues: positive,
        op: None,
    }
}

/// The property that makes this a defect rather than a tolerance question:
/// along the lifted invariance the rho-curvature is EXACTLY the chain-rule
/// term, so `|t' H_rho t| == sum_k |g_k| t_k^2` whenever the gradient shares a
/// sign on the direction's support. Built from an `H_lambda` that is genuinely
/// positive definite on the identified block, so there is no negative curvature
/// anywhere in the construction.
#[test]
fn lifted_invariance_carries_only_the_chain_rule_term_2676() {
    // V depends on lambda through Lambda = lambda_0 + c*lambda_2 and lambda_1.
    let scale = 0.75_f64;
    let lambdas = Array1::from(vec![1.5_f64, 4.0, 2.25]);
    // A positive definite reduced Hessian in (Lambda, lambda_1).
    let reduced = array![[0.8_f64, -0.2], [-0.2, 0.5]];
    // Pull back: dLambda/dlambda = (1, 0, c), dlambda_1/dlambda = (0, 1, 0).
    let jacobian = array![[1.0_f64, 0.0], [0.0, 1.0], [scale, 0.0]];
    let h_lambda = jacobian.dot(&reduced).dot(&jacobian.t());
    // A gradient that is a genuine differential of the same reduced criterion,
    // so g_lambda = J g_reduced and g_rho = diag(lambda) g_lambda.
    let g_reduced = Array1::from(vec![-3.0e-5_f64, 7.0e-6]);
    let g_lambda = jacobian.dot(&g_reduced);
    let g_rho = Array1::from(vec![
        lambdas[0] * g_lambda[0],
        lambdas[1] * g_lambda[1],
        lambdas[2] * g_lambda[2],
    ]);
    let mut h_rho = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            h_rho[[i, j]] = lambdas[i] * h_lambda[[i, j]] * lambdas[j];
        }
        h_rho[[i, i]] += g_rho[i];
    }

    let s0 = array![[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let s2 = s0.mapv(|value| scale * value);
    let bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    let lifted = invariance
        .theta_directions(&lambdas, 3, 0)
        .expect("one lifted direction");
    assert_eq!(lifted.ncols(), 1);
    let t = lifted.column(0).to_owned();

    let rayleigh = t.dot(&h_rho.dot(&t));
    let chain_rule: f64 = (0..3).map(|k| g_rho[k] * t[k] * t[k]).sum();
    let floor: f64 = (0..3).map(|k| g_rho[k].abs() * t[k] * t[k]).sum();
    // `t' (diag(l) H_lambda diag(l)) t` is zero as an IDENTITY, so what is left
    // of it in floating point is round-off of the matrix's own scale — not of
    // the curvature being reported. Judge it against that scale, which is the
    // whole asymmetry: the residual is `eps * ||H||`, and the quantity the gate
    // decides on is `10^11` times smaller than `||H||`.
    let matrix_scale = h_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let intrinsic = rayleigh - chain_rule;
    assert!(
        intrinsic.abs() <= 64.0 * f64::EPSILON * matrix_scale,
        "the intrinsic part {intrinsic:.6e} must be round-off of the matrix scale \
         {matrix_scale:.6e}, i.e. the reparameterisation identity holds"
    );
    // The two gradient components on this direction's support share a sign
    // (both are `g_reduced[0]` times a positive number), so the identity is
    // tight: the quantity the gate compares IS the bound it compares against,
    // and the ONLY thing separating them is that round-off.
    let relative_gap = (rayleigh.abs() - floor).abs() / floor;
    assert!(
        relative_gap <= 1e-9,
        "|t'H t| = {:.17e} and the gradient floor {floor:.17e} must agree to at least 9 \
         digits (gap {relative_gap:.3e}); the gate's verdict is the SIGN of that gap",
        rayleigh.abs()
    );
    // And the whole matrix IS indefinite in rho even though H_lambda is PSD —
    // the negative eigenvalue is manufactured by the reparameterisation.
    use gam_linalg::faer_ndarray::FaerEigh;
    let (eigenvalues, _) = h_rho.eigh(faer::Side::Lower).expect("eigh");
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        minimum < 0.0,
        "the rho Hessian must carry the manufactured negative eigenvalue (got {minimum:.6e})"
    );
    // Deflating it leaves a strictly positive definite judged block.
    let judged = judged_subspace_basis(3, &[], Some(&lifted)).expect("complement is 2-dimensional");
    assert_eq!(judged.ncols(), 2);
    let compressed = compress_to_judged_subspace(&h_rho, &judged);
    let (compressed_eigenvalues, _) = compressed.eigh(faer::Side::Lower).expect("eigh");
    let compressed_minimum = compressed_eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        compressed_minimum > 0.0,
        "the judged complement must be positive definite (got {compressed_minimum:.6e})"
    );
}

/// With nothing to deflate the judged subspace is the interior indicator basis,
/// so `Z' H Z` is the historical sub-block BIT FOR BIT. This is what makes the
/// change inert on every model without a redundant penalty map.
#[test]
fn empty_deflation_reproduces_the_interior_sub_block_bitwise_2676() {
    let h = array![
        [4.0_f64, 1.0, -0.5, 0.25],
        [1.0, 3.0, 0.75, -0.125],
        [-0.5, 0.75, 2.5, 0.5],
        [0.25, -0.125, 0.5, 1.75],
    ];
    let excluded = [1usize, 3];
    let basis = judged_subspace_basis(4, &excluded, None).expect("interior is non-empty");
    let compressed = compress_to_judged_subspace(&h, &basis);
    assert_eq!(compressed.dim(), (2, 2));
    let expected = array![[4.0_f64, -0.5], [-0.5, 2.5]];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(
                compressed[[i, j]].to_bits(),
                expected[[i, j]].to_bits(),
                "entry ({i},{j}) must be bit-identical to the sub-block"
            );
        }
    }
}

/// A penalty map with no redundancy must produce no invariance at all — the
/// deflation must not fire on ordinary models.
#[test]
fn independent_penalties_have_no_invariance_2676() {
    let s0 = array![[2.0_f64, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0_f64, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let s2 = array![[1.0_f64, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]];
    let bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    assert_eq!(invariance.dimension(), 0);
    let lambdas = Array1::from(vec![1.0_f64, 2.0, 3.0]);
    assert!(invariance.theta_directions(&lambdas, 3, 0).is_none());
}

/// A nonzero prior mean can BREAK a redundancy that the quadratic parts alone
/// would show: `S_2 = c S_0` but `mu_2 != mu_0` leaves the linear part of the
/// penalty varying along `(c, 0, -1)`, so the criterion is not invariant and
/// nothing may be deflated. The augmented Gram is what sees this.
#[test]
fn a_prior_mean_that_breaks_proportionality_removes_the_invariance_2676() {
    let s0 = array![[2.0_f64, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0_f64, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let scale = 0.75_f64;
    let s2 = s0.mapv(|value| scale * value);
    let mut bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];
    assert_eq!(
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3)
            .expect("gram decomposes")
            .dimension(),
        1,
        "with zero prior means the pair is redundant"
    );
    bundle[2].prior_mean = Array1::from(vec![0.4_f64, -0.3, 0.0]);
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    assert_eq!(
        invariance.dimension(),
        0,
        "a differing prior mean makes the criterion genuinely depend on the direction"
    );
}

/// The lift is `diag(lambda)^{-1} w`, not the raw `w`: at unequal lambdas the
/// two are different directions, and only the lifted one annihilates
/// `diag(lambda) H_lambda diag(lambda)`.
#[test]
fn the_lift_is_the_lambda_weighted_tangent_not_the_index_pattern_2676() {
    let scale = 1.0_f64;
    let s0 = array![[1.0_f64, 0.0], [0.0, 0.0]];
    let s2 = s0.mapv(|value| scale * value);
    let bundle = vec![penalty(s0, 2), penalty(s2, 2)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 2).expect("gram decomposes");
    assert_eq!(invariance.dimension(), 1);
    let lambdas = Array1::from(vec![0.5_f64, 8.0]);
    let lifted = invariance
        .theta_directions(&lambdas, 2, 0)
        .expect("lift exists");
    let t = lifted.column(0).to_owned();
    // w ∝ (1, -1); t ∝ (1/0.5, -1/8) = (2, -0.125), which is 16:1, not 1:1.
    let ratio = (t[0] / t[1]).abs();
    assert!(
        (ratio - 16.0).abs() < 1e-9,
        "the lifted tangent must be lambda-weighted (|t0/t1| = {ratio:.6}, expected 16)"
    );
}

/// Embedding into a wider theta (the exact-joint `(rho, psi)` route) leaves the
/// non-rho coordinates at exactly zero, which is what makes the chain-rule
/// identity survive the embedding.
#[test]
fn embedding_into_a_wider_theta_zeroes_the_non_rho_block_2676() {
    let s0 = array![[1.0_f64, 0.0], [0.0, 0.0]];
    let s1 = s0.clone();
    let bundle = vec![penalty(s0, 2), penalty(s1, 2)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 2).expect("gram decomposes");
    let lambdas = Array1::from(vec![2.0_f64, 3.0]);
    let lifted = invariance
        .theta_directions(&lambdas, 5, 0)
        .expect("lift exists");
    assert_eq!(lifted.dim(), (5, 1));
    for row in 2..5 {
        assert_eq!(lifted[[row, 0]], 0.0, "psi coordinate {row} must be zero");
    }
}

/// The disjoint-supports fast path is a THEOREM, and it must agree with the
/// Gram it skips. An ordinary additive model — one penalty per smooth, each on
/// its own coefficient block — has no invariance, and the shortcut must reach
/// that answer for the reason stated (positive-definite diagonal plus a rank-1
/// PSD border) and not by accident.
#[test]
fn disjoint_penalty_supports_have_no_invariance_2676() {
    // Three blocks on 0..2, 2..5, 5..7 — the additive-model layout.
    let s0 = array![[2.0_f64, 0.5], [0.5, 1.0]];
    let s1 = array![[3.0_f64, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.5]];
    let s2 = array![[4.0_f64, -1.0], [-1.0, 2.0]];
    let mut bundle = vec![penalty(s0, 7), penalty(s1, 7), penalty(s2, 7)];
    bundle[1].col_range = 2..5;
    bundle[2].col_range = 5..7;
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 7).expect("gram decomposes");
    assert_eq!(invariance.dimension(), 0);

    // And the shortcut is not hiding a disagreement: force the general path by
    // overlapping the FIRST TWO ranges by one column with a penalty that is
    // zero on the shared column, so the Gram is still block-diagonal and the
    // answer must be unchanged.
    let mut overlapping = bundle.clone();
    let mut widened = Array2::<f64>::zeros((3, 3));
    widened
        .slice_mut(ndarray::s![..2, ..2])
        .assign(&overlapping[0].local);
    overlapping[0].local = widened;
    overlapping[0].col_range = 0..3;
    overlapping[0].prior_mean = Array1::zeros(3);
    let general =
        PenaltyMapInvariance::from_canonical_penalties(&overlapping, 7).expect("gram decomposes");
    assert_eq!(
        general.dimension(),
        0,
        "the general Gram path must agree with the disjoint-support shortcut"
    );
}

/// The instrument (#2748): on the certified invariance the residual of
/// `T'H_ρT = T'diag(g_ρ)T` is EXACTLY zero in exact arithmetic, so whatever
/// comes back is error and only error. Built on the same reduced-criterion
/// fixture as the chain-rule gate above, so the (H_ρ, g_ρ) pair is genuinely
/// consistent by construction rather than by assertion.
#[test]
fn the_invariance_residual_is_zero_on_a_consistent_assembly_2748() {
    let (h_rho, g_rho, lifted) = reduced_criterion_fixture();
    let residual =
        invariance_residual_2norm(&h_rho, &g_rho, &lifted).expect("one certified direction");
    let matrix_scale = h_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    assert!(
        residual <= 64.0 * f64::EPSILON * matrix_scale,
        "a consistent (H_rho, g_rho) pair must measure {residual:.6e} <= round-off of the \
         matrix scale {matrix_scale:.6e}"
    );
}

/// And it recovers an injected error exactly. `H_ρ + η·tt'` moves the identity
/// by `η` and by nothing else, so the instrument reads `η` — that is what makes
/// it a MEASUREMENT of `‖δH‖₂` rather than an estimate of it.
#[test]
fn the_invariance_residual_recovers_an_injected_assembly_error_2748() {
    let (mut h_rho, g_rho, lifted) = reduced_criterion_fixture();
    let t = lifted.column(0).to_owned();
    let injected = 3.5e-7_f64;
    for row in 0..3 {
        for col in 0..3 {
            h_rho[[row, col]] += injected * t[row] * t[col];
        }
    }
    let residual =
        invariance_residual_2norm(&h_rho, &g_rho, &lifted).expect("one certified direction");
    // The tolerance is the arithmetic the instrument itself runs at — one
    // rounding of the matrix scale — not a chosen relative fraction of the
    // injected value, which would tighten as the injection shrinks and say
    // nothing about the instrument.
    let matrix_scale = h_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    assert!(
        (residual - injected).abs() <= 64.0 * f64::EPSILON * matrix_scale,
        "the instrument must read the injected {injected:.17e}; got {residual:.17e}"
    );
}

/// The control that says the instrument measures the error WHERE THE IDENTITY
/// IS, not everywhere: a perturbation confined to the judged complement leaves
/// it at zero. Without this the residual could be read as a global matrix-error
/// estimate, which it is not — it is a certified lower bound obtained from one
/// subspace.
#[test]
fn a_perturbation_off_the_invariance_does_not_move_the_residual_2748() {
    let (mut h_rho, g_rho, lifted) = reduced_criterion_fixture();
    let t = lifted.column(0).to_owned();
    let baseline =
        invariance_residual_2norm(&h_rho, &g_rho, &lifted).expect("one certified direction");
    // Any direction orthogonal to `t`, made so explicitly rather than assumed.
    let mut u = Array1::from(vec![0.31_f64, -0.87, 0.42]);
    let projection = u.dot(&t);
    u.scaled_add(-projection, &t);
    let norm = u.dot(&u).sqrt();
    u.mapv_inplace(|value| value / norm);
    for row in 0..3 {
        for col in 0..3 {
            h_rho[[row, col]] += 1.0e-3 * u[row] * u[col];
        }
    }
    let perturbed =
        invariance_residual_2norm(&h_rho, &g_rho, &lifted).expect("one certified direction");
    let matrix_scale = h_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    assert!(
        (perturbed - baseline).abs() <= 64.0 * f64::EPSILON * matrix_scale,
        "a 1e-3 perturbation confined to the judged complement moved the invariance \
         residual from {baseline:.6e} to {perturbed:.6e}; the instrument must be blind to it"
    );
}

/// THE REGRESSION (#2676): a penalty map whose operators are `1.2e-8` from
/// dependent must NOT be certified as exactly dependent.
///
/// `1.2e-8` is the measured `geo_disease_matern` (centers=24, n=4000) figure.
/// It matters because the Gram carries the defect SQUARED —
/// `lambda_min(G) = delta^2 = 1.5e-16` — so a rank test taken on `G` at `G`'s
/// own `eps` certified it, deflated the direction, and then reported the
/// criterion's genuine curvature there as the assembly's error.
///
/// Three assertions, and the second and third are what make the first
/// non-vacuous: the fixture IS a near-dependency at exactly that scale, and its
/// Gram eigenvalue IS at machine epsilon, i.e. the old test could not have
/// distinguished it from an exact one.
#[test]
fn a_map_1p2e_minus_8_from_dependent_is_not_certified_dependent_2676() {
    const DEFECT: f64 = 1.238_259e-8;
    let s0 = array![[2.0_f64, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0_f64, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    // A perturbation orthogonal to `s0` in the Frobenius inner product, so the
    // defect is exactly its relative size and none of it is absorbed into the
    // best proportionality constant.
    let base_norm = s0.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut direction = array![[0.0_f64, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let overlap =
        direction.iter().zip(s0.iter()).map(|(a, b)| a * b).sum::<f64>() / (base_norm * base_norm);
    direction = &direction - &(overlap * &s0);
    let direction_norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    let s2 = &s0 + &(direction * (DEFECT * base_norm / direction_norm));

    let bundle = vec![penalty(s0.clone(), 3), penalty(s1, 3), penalty(s2.clone(), 3)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    assert_eq!(
        invariance.dimension(),
        0,
        "operators {DEFECT:.3e} from dependent are MEASURABLY independent and must not be \
         certified; the criterion carries genuine curvature of order defect^2 along that \
         direction and the certificate has to judge it"
    );
    assert!(
        invariance.resolution() < DEFECT,
        "the reported defect floor {:.3e} must be BELOW the defect it refused, or the refusal \
         is an accident",
        invariance.resolution()
    );

    // Non-vacuity 1: the fixture really is a near-dependency at that scale.
    let residual = (&s2 - &s0).iter().map(|v| v * v).sum::<f64>().sqrt() / base_norm;
    assert!(
        (residual - DEFECT).abs() <= 1e-3 * DEFECT,
        "the fixture must sit at {DEFECT:.3e}; got {residual:.3e}"
    );

    // Non-vacuity 2: the Gram route could not have seen it. The smallest
    // eigenvalue of the 2x2 Gram of the near-dependent pair is `delta^2/2` times
    // the operator scale, which is AT machine epsilon on this fixture — so a
    // rank test denominated in the Gram is deciding on its own round-off.
    let gram_eigenvalue = 0.5 * DEFECT * DEFECT * base_norm * base_norm;
    assert!(
        gram_eigenvalue <= 16.0 * f64::EPSILON * base_norm * base_norm,
        "the fixture must be one the Gram cannot resolve: lambda_min(G) = {gram_eigenvalue:.3e} \
         against an eps-scaled Gram norm of {:.3e}",
        f64::EPSILON * base_norm * base_norm
    );
}

/// The other side of the same boundary: operators that ARE equal to the
/// arithmetic that formed them stay certified. Without this the fix above could
/// be "certify nothing", which would silently retire the whole deflation.
#[test]
fn operators_equal_to_their_own_arithmetic_stay_certified_2676() {
    let s0 = array![[2.0_f64, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0_f64, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    // One ulp of perturbation on the largest entry — the most a bit-identical
    // pair can differ by and still be the same operator.
    let mut s2 = s0.mapv(|value| 0.75 * value);
    s2[[0, 0]] = f64::from_bits(s2[[0, 0]].to_bits() + 1);
    let bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    assert_eq!(
        invariance.dimension(),
        1,
        "a one-ulp difference is the operators' own representation and must stay certified"
    );
}

/// The arithmetic the rank boundary rests on, gated directly: the double-double
/// accumulation must resolve a Gram entry whose value is `eps^2`-scale, which is
/// precisely what an `f64` accumulation cannot do.
#[test]
fn the_gram_accumulation_resolves_below_machine_epsilon_2676() {
    // `sum_i a_i b_i` over terms of size 1 whose exact total is 1e-24. An `f64`
    // accumulation of the same terms returns 0 or an eps-scale artefact.
    let a = [1.0_f64, 1.0, 1e-24];
    let b = [1.0_f64, -1.0, 1.0];
    let mut compensated = DoubleDouble::ZERO;
    let mut naive = 0.0_f64;
    for index in 0..a.len() {
        compensated = compensated.add(DoubleDouble::from_product(a[index], b[index]));
        naive += a[index] * b[index];
    }
    assert_eq!(
        compensated.to_f64(),
        1e-24,
        "the compensated accumulation must return the exact total"
    );
    // The naive one happens to survive THIS ordering, so reorder it into the one
    // a Gram loop would produce and show the loss.
    let mut reordered = 0.0_f64;
    for index in [0usize, 2, 1] {
        reordered += a[index] * b[index];
    }
    assert_eq!(
        reordered, 0.0,
        "the f64 accumulation loses the entire quantity under a different ordering \
         (got {reordered:.3e}), which is why the rank boundary cannot rest on it"
    );
    assert_eq!(naive, 1e-24);
    // And the two-product residual is exact where a rounded product is not.
    let (product, residual) = DoubleDouble::two_product(1.0 + f64::EPSILON, 1.0 + f64::EPSILON);
    assert_eq!(product, 1.0 + 2.0 * f64::EPSILON);
    assert_eq!(residual, f64::EPSILON * f64::EPSILON);
}

/// The rank boundary is a statement about the operators, so it must not move
/// when every operator is rescaled by a common factor — the map's null space is
/// scale-free and the floor is denominated in the operators' own norm.
///
/// Run at `1e-8` and `1e8`, i.e. sixteen orders apart, on both a dependent and
/// an independent bundle. This is the edge case the `max(1.0)` clamp the floor
/// used to carry would have failed: it made the floor loose in exact proportion
/// to how small the operators were.
#[test]
fn the_rank_verdict_is_invariant_to_a_uniform_rescaling_2676() {
    let s0 = array![[2.0_f64, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0_f64, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let dependent = s0.mapv(|value| 0.75 * value);
    let independent = array![[1.0_f64, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]];
    for factor in [1.0_f64, 1e-8, 1e8] {
        for (third, expected) in [(&dependent, 1usize), (&independent, 0usize)] {
            let bundle = vec![
                penalty(s0.mapv(|value| factor * value), 3),
                penalty(s1.mapv(|value| factor * value), 3),
                penalty(third.mapv(|value| factor * value), 3),
            ];
            let invariance = PenaltyMapInvariance::from_canonical_penalties(&bundle, 3)
                .expect("gram decomposes");
            assert_eq!(
                invariance.dimension(),
                expected,
                "the verdict must not depend on a uniform rescaling (factor {factor:.1e})"
            );
        }
    }
}

/// The degenerate end: a penalty map that is identically zero constrains
/// nothing, so every direction of `lambda` leaves the criterion unchanged and
/// the whole space is the invariance. The floor is zero there — the operators'
/// own scale is zero — so no pivot can clear it, which is the right answer
/// reached for the right reason rather than by a special case.
#[test]
fn an_all_zero_penalty_map_is_entirely_null_2676() {
    let zero = Array2::<f64>::zeros((3, 3));
    let bundle = vec![penalty(zero.clone(), 3), penalty(zero, 3)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    assert_eq!(
        invariance.dimension(),
        2,
        "an identically zero penalty map is flat in every lambda direction"
    );
    assert_eq!(invariance.resolution(), 0.0);
}

/// The `(H_ρ, g_ρ, lifted invariance)` triple used by the #2748 instrument
/// gates: a criterion that depends on `λ` only through `Λ = λ₀ + c·λ₂` and
/// `λ₁`, assembled through the chain rule so the reparameterisation identity
/// holds by construction.
fn reduced_criterion_fixture() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let scale = 0.75_f64;
    let lambdas = Array1::from(vec![1.5_f64, 4.0, 2.25]);
    let reduced = array![[0.8_f64, -0.2], [-0.2, 0.5]];
    let jacobian = array![[1.0_f64, 0.0], [0.0, 1.0], [scale, 0.0]];
    let h_lambda = jacobian.dot(&reduced).dot(&jacobian.t());
    let g_reduced = Array1::from(vec![-3.0e-5_f64, 7.0e-6]);
    let g_lambda = jacobian.dot(&g_reduced);
    let g_rho = Array1::from(vec![
        lambdas[0] * g_lambda[0],
        lambdas[1] * g_lambda[1],
        lambdas[2] * g_lambda[2],
    ]);
    let mut h_rho = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            h_rho[[i, j]] = lambdas[i] * h_lambda[[i, j]] * lambdas[j];
        }
        h_rho[[i, i]] += g_rho[i];
    }
    let s0 = array![[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let s2 = s0.mapv(|value| scale * value);
    let bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    let lifted = invariance
        .theta_directions(&lambdas, 3, 0)
        .expect("one lifted direction");
    (h_rho, g_rho, lifted)
}
