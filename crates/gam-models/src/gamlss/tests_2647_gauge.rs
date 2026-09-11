// gam#2647 — the joint penalized Hessian of the binomial location-scale wiggle
// model must be non-singular, and that has to be the PENALTY's doing.
//
// The measurement instruments that found this (a budget ladder, a design-level
// alias ladder, an orbit walk) are recorded on the issue; what survives here is
// the property they established, asserted rather than printed.
#![cfg(test)]

use super::*;
use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;

/// The exact joint LIKELIHOOD Hessian of this model is structurally
/// rank-deficient inside the wiggle block, and the penalty is the only thing
/// that makes the joint system solvable.
///
/// The warp design is dynamic: `B(q₀)` is rebuilt from the current index every
/// inner cycle, and an anchored I-spline saturates outside its knot hull (every
/// column `0` below `left`, one row-independent constant vector above `right` —
/// `create_ispline_dense`). Whenever `q₀` fails to spread across the hull the
/// rows of `B(q₀)` collapse and `H_ww = BᵀWB` loses rank with them. At `β = 0`
/// every row has `q₀ = 0`, so the design is rank one and `H_L` carries `p_w − 1`
/// exact zeros. That is measured below and is NOT a defect: a likelihood cannot
/// be asked to identify a basis the data does not exercise.
///
/// What IS a defect — and what gam#2647 was — is the penalty leaving a direction
/// of that null space free. `null(H_L + S) = ker(H_L) ∩ ker(S)`, so with an
/// order-2 roughness and `double_penalty = false` the linear warp (which is in
/// `ker(S_w)`, and which is a rescale of the index the warp is composed onto)
/// was exactly flat: the solve walked it with `‖β‖∞` up 230× while `½βᵀSβ` fell
/// like `‖β‖⁻²` and `−loglik` stayed flat to `8e-4`.
///
/// So the assertion is a pair, and both halves are load-bearing: `H_L` alone is
/// exactly singular in the wiggle block, and `H_L + S` is not singular at all.
/// A test that only checked the second could be satisfied by a design change
/// that never touched the penalty; a test that only checked the first would pin
/// the precondition and miss the fix.
#[test]
pub(crate) fn joint_penalized_hessian_is_nonsingular_where_the_likelihood_alone_is_not_2647() {
    let n = 30usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.5 * std::f64::consts::PI * t).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 9 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let q_seed = Array1::linspace(-1.5, 1.5, n);

    let thresholdspec = simple_matern_term_collection(&[0, 1], 0.45);
    let threshold_collection =
        build_term_collection_design(data.view(), &thresholdspec).expect("threshold design");
    let threshold_design = threshold_collection.design.clone();
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::<f64>::zeros((n, 0)),
    ));
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 4, 2, false)
            .expect("wiggle block");
    let p_t = threshold_design.ncols();
    let p_w = wiggle_block.design.ncols();
    assert!(p_t >= 2 && p_w >= 2, "degenerate fixture: p_t={p_t} p_w={p_w}");

    let family = BinomialLocationScaleWiggleFamily {
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let block = |name: &str, design: DesignMatrix| ParameterBlockSpec {
        name: name.to_string(),
        design,
        offset: Array1::zeros(n),
        ..ParameterBlockSpec::defaults()
    };
    let specs = vec![
        block("threshold", threshold_design.clone()),
        block("log_sigma", log_sigma_design),
        block("wiggle", wiggle_block.design.clone()),
    ];
    let states = vec![
        ParameterBlockState {
            eta: Array1::zeros(n),
            beta: Array1::zeros(p_t),
        },
        ParameterBlockState {
            eta: Array1::zeros(n),
            beta: Array1::zeros(0),
        },
        ParameterBlockState {
            eta: Array1::zeros(n),
            beta: Array1::zeros(p_w),
        },
    ];

    let h_l = family
        .exact_newton_joint_hessian_with_specs(&states, &specs)
        .expect("exact joint hessian")
        .expect("exact joint hessian available");
    let total = p_t + p_w;
    assert_eq!(h_l.dim(), (total, total));

    let (lvals, lvecs) = h_l.eigh(Side::Lower).expect("likelihood hessian eigh");
    let l_scale = lvals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let exact_zero_in_wiggle = (0..total)
        .filter(|&k| lvals[k].abs() <= 1e-12 * l_scale)
        .filter(|&k| {
            let v = lvecs.column(k);
            let mass_w: f64 = (p_t..total).map(|i| v[i] * v[i]).sum();
            mass_w > 0.99
        })
        .count();
    assert!(
        exact_zero_in_wiggle >= 2,
        "precondition lost: the dynamic warp design no longer collapses at β = 0, so this \
         fixture stopped exercising the null space gam#2647 is about (found \
         {exact_zero_in_wiggle} wiggle-local exact zeros in H_L, λ_max = {l_scale:.6e})"
    );

    // The fixture's own seed is ρ = 0, i.e. λ = 1 on every block.
    let mut h_pen = h_l.clone();
    let s_t: Array2<f64> = threshold_collection
        .penalties_as_penalty_matrix()
        .first()
        .map(|p| p.to_dense())
        .unwrap_or_else(|| Array2::zeros((p_t, p_t)));
    for i in 0..p_t {
        for j in 0..p_t {
            h_pen[[i, j]] += s_t[[i, j]];
        }
    }
    for penalty in &wiggle_block.penalties {
        let s_w = match penalty {
            gam_terms::penalty_spec::PenaltySpec::Dense(m) => m.clone(),
            gam_terms::penalty_spec::PenaltySpec::DenseWithMean { matrix, .. } => matrix.clone(),
            other => panic!("unexpected warp penalty representation: {other:?}"),
        };
        for i in 0..p_w {
            for j in 0..p_w {
                h_pen[[p_t + i, p_t + j]] += s_w[[i, j]];
            }
        }
    }

    let (pvals, _) = h_pen.eigh(Side::Lower).expect("penalized hessian eigh");
    let p_min = pvals.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
    let p_max = pvals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    // Before the gauge closure this was the ridge floor -- the family's own
    // source comment records `σ_min ≈ ridge_floor ≈ 1e-10`. It is now O(1)
    // (measured 7.254550e-1 against λ_max ≈ 7.2e1). The bar is set six orders
    // above the old value and five below the measured one, so it separates the
    // two regimes without pinning a number that honest reconditioning can move.
    assert!(
        p_min > 1e-4 * p_max,
        "the joint penalized Hessian is still effectively singular (gam#2647): |λ|min = \
         {p_min:.6e} against |λ|max = {p_max:.6e}. Some direction of the warp block's design \
         null space is unpenalized, and the penalized criterion is unbounded below along it."
    );
}
