#![cfg(test)]
//! #2946 pins for context-dependent declared laws.
//!
//! The planted and gradient identities run on dyadic data, so the operator's covariances are formed exactly and
//! identities between two routes hold bit for bit. The finite-difference arm accepts within a Richardson estimate
//! of its own truncation plus the derived rounding of the values it differences. The sampled arm accepts within a
//! standard-error multiple derived from a declared false-alarm rate. Each arm carries a positive control the test
//! shows it rejects.

use super::{ContextAccess, ContextBlocks, ContextDeclaredLaw, ContextLaw, ContextResponseError};
use crate::response::raw_block::UnabsorbedBlock;
use crate::response::subspace::KnownBlock;
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::GaussianActivation;
use gam_math::probability::standard_normal_quantile;
use ndarray::{Array1, Array2, ArrayView2, array, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::{LN_2, PI};

/// The declared probability that a correct evaluation fails one Monte Carlo test for a fresh seed, split evenly over
/// the test's two-sided agreement arms (Bonferroni).
const FAMILY_FALSE_ALARM: f64 = 1.0e-6;

fn standard_error_multiple(arms: usize) -> f64 {
    standard_normal_quantile(1.0 - FAMILY_FALSE_ALARM / (2.0 * arms as f64))
        .expect("the false-alarm split lies inside (0, 1)")
}

fn identity(dim: usize) -> Array2<f64> {
    Array2::from_shape_fn((dim, dim), |(row, column)| if row == column { 1.0 } else { 0.0 })
}

/// `H = I − ½·11ᵀ` in four dimensions: symmetric, orthogonal, every entry `±½`.
fn half_reflector() -> Array2<f64> {
    Array2::from_shape_fn((4, 4), |(row, column)| if row == column { 0.5 } else { -0.5 })
}

/// One zero-bias ReLU unit reading the first ambient coordinate, written to one output with output bias ½ and metric 2.
fn single_unit_block() -> UnabsorbedBlock {
    UnabsorbedBlock {
        readers: array![[1.0, 0.0, 0.0]],
        biases: Array1::zeros(1),
        writers: array![[1.0]],
        output_bias: array![0.5],
        metric: array![[2.0]],
        activation: GaussianActivation::Relu,
    }
}

/// Context A reads `z₁` and context B reads `z₂`: B's loading swaps the first two coordinates. Both baselines are
/// orthogonal to the reader, so every absorbed bias is zero while the baselines themselves differ.
fn planted_contexts() -> ContextBlocks {
    let swap = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    ContextBlocks::new(
        single_unit_block(),
        vec![
            ContextDeclaredLaw {
                baseline: array![0.0, 0.0, 3.0],
                loading: identity(3),
            },
            ContextDeclaredLaw {
                baseline: array![0.0, -2.0, 0.0],
                loading: swap,
            },
        ],
    )
    .expect("two finite contexts sharing the intervention dimension")
}

/// fr-subspace's overlapping readers, zero biases, nonnegative writers and a coupled dyadic metric.
fn overlapping_block() -> UnabsorbedBlock {
    UnabsorbedBlock {
        readers: array![
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, -1.0],
        ],
        biases: Array1::zeros(4),
        writers: array![
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ],
        output_bias: Array1::zeros(3),
        metric: array![[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
        activation: GaussianActivation::Relu,
    }
}

/// The overlapping block under three signed-permutation loadings (the identity, a signed permutation and a cyclic
/// shift of the columns), with zero baselines, since these readers span the ambient space. The frame of the tests
/// below spans `{(1, −1, 0, 0), (0, 0, 1, 1)}`, whose vectors have the form `(a, −a, b, b)`. No absorbed reader has
/// that form, so no reader lies in the frame, every pair's correlation sits strictly inside `(−1, 1)` at the frame,
/// and the kernel is smooth there.
fn overlapping_laws() -> Vec<ContextDeclaredLaw> {
    let signed_permutation = array![
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.0],
    ];
    let cyclic_shift = array![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    [identity(4), signed_permutation, cyclic_shift]
        .into_iter()
        .map(|loading| ContextDeclaredLaw {
            baseline: Array1::zeros(4),
            loading,
        })
        .collect()
}

fn overlapping_contexts(laws: Vec<ContextDeclaredLaw>) -> ContextBlocks {
    ContextBlocks::new(overlapping_block(), laws).expect("finite contexts sharing the intervention dimension")
}

fn frame_along(axis: usize, dim: usize) -> Array2<f64> {
    Array2::from_shape_fn((dim, 1), |(row, column)| if row == axis && column == 0 { 1.0 } else { 0.0 })
}

#[test]
fn two_planted_contexts_report_their_own_errors_beside_the_average_frame() {
    let contexts = planted_contexts();
    let law = ContextLaw::Enumerated {
        masses: vec![3.0, 1.0],
    };
    let frame = frame_along(0, 3);
    let report = contexts
        .evaluate(frame.view(), &law, &ContextAccess::ContextSpecific)
        .expect("a declared law over two contexts");

    // Context A's reader lies in the frame, so V_A(P) and V_A(I) feed the kernel identical arguments.
    assert_eq!(report.rows[0].error, 0.0, "the planted context must discard nothing, got {}", report.rows[0].error);
    // Each row is the operator's own error for that context's absorbed block, built here by hand: readers `W L(c)`
    // and biases `b + W h₀(c) = 0`.
    let by_hand = KnownBlock::new(
        array![[0.0, 1.0, 0.0]],
        Array1::zeros(1),
        array![[1.0]],
        array![0.5],
        array![[2.0]].view(),
        GaussianActivation::Relu,
    )
    .expect("context B's absorbed block");
    let context_b_error = by_hand.discarded_error(frame.view()).expect("the frame");
    assert_eq!(report.rows[1].error, context_b_error.value);
    assert_eq!(report.rows[1].total_variance, by_hand.total_variance().value);
    // The one absorption owner builds that same block from context B's declared law.
    let absorbed = single_unit_block()
        .absorb(array![0.0, -2.0, 0.0].view(), array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]].view())
        .expect("context B's declared law");
    assert_eq!(absorbed.readers(), by_hand.readers());
    assert_eq!(absorbed.biases(), by_hand.biases());
    assert_eq!(absorbed.discarded_error(frame.view()).expect("the frame"), context_b_error);
    // The output bias reaches every context's block, which carries it for responses at points; no error reads it.
    for context in 0..contexts.context_count() {
        assert_eq!(contexts.block(context).output_bias(), array![0.5].view());
    }
    // Independent anchor: B's reader is orthogonal to the frame, so B discards its whole variance,
    // `D·v·(½ − 1/(2π)) = 1 − 1/π`. The band counts the rounded operations of the two kernel evaluations (ρ = 1 and
    // ρ = 0: projection, arccos, sine, quotients, products), the smoothing and the differences, at most 48 on any path,
    // over the magnitudes `D·(½ + 3/(2π))` they accumulate.
    let closed_form = 1.0 - 1.0 / PI;
    let band = accumulation_growth(48) * 2.0 * (0.5 + 3.0 / (2.0 * PI));
    assert!(
        (report.rows[1].error - closed_form).abs() <= band,
        "context B must discard 1 − 1/π = {closed_form}, got {} (band {band})",
        report.rows[1].error,
    );
    assert!(
        (report.rows[1].error - report.rows[1].total_variance).abs() <= band,
        "the frame discards all of the rare context's variance: error {} vs V_B(I) {}",
        report.rows[1].error,
        report.rows[1].total_variance,
    );

    // The average weighs B by 1/4, so it is exactly a quarter of B's row: the average hides the rare context's loss.
    assert_eq!(report.error, 0.25 * report.rows[1].error);
    assert_eq!(report.worst_context, 1);
    assert!(report.error < report.rows[report.worst_context].error);
    assert!(report.standard_error.is_none(), "a declared law is exact");

    // The frame is stationary for the average: A's gradient is horizontal to its own reader, and B's reader has no
    // coordinate in the frame. Both vanish exactly on dyadic data.
    assert!(
        report.horizontal_gradient.iter().all(|entry| *entry == 0.0),
        "the planted frame must be stationary, got {:?}",
        report.horizontal_gradient,
    );
    // And it beats the rare context's own frame for the average, which instead discards A with weight 3/4.
    let rare_frame = frame_along(1, 3);
    let rare_report = contexts
        .evaluate(rare_frame.view(), &law, &ContextAccess::ContextSpecific)
        .expect("the rare context's frame");
    assert_eq!(rare_report.rows[1].error, 0.0);
    assert_eq!(rare_report.worst_context, 0);
    assert!(
        report.error < rare_report.error,
        "the heavy context's frame must be better on average: {} vs {}",
        report.error,
        rare_report.error,
    );
}

/// The four horizontal directions `Q⊥ A` with `A` in `{I, diag(1, −1), [[0, 1], [1, 0]], [[0, −1], [1, 0]]}`. They
/// span the tangent space at `Q`, have orthonormal columns and satisfy `Qᵀ Δ = 0`, so `(Q + tΔ)/√(1 + t²)` is
/// orthonormal for every `t`.
fn horizontal_directions(complement: ArrayView2<'_, f64>) -> Vec<Array2<f64>> {
    [
        array![[1.0, 0.0], [0.0, 1.0]],
        array![[1.0, 0.0], [0.0, -1.0]],
        array![[0.0, 1.0], [1.0, 0.0]],
        array![[0.0, -1.0], [1.0, 0.0]],
    ]
    .into_iter()
    .map(|mixing| complement.dot(&mixing))
    .collect()
}

#[test]
fn the_gradient_of_the_average_is_the_sum_of_the_per_context_gradients() {
    let contexts = overlapping_contexts(overlapping_laws());
    let masses = vec![2.0, 1.0, 1.0];
    let law = ContextLaw::Enumerated {
        masses: masses.clone(),
    };
    let reflector = half_reflector();
    let frame = reflector.slice(s![.., ..2]).to_owned();
    let report = contexts
        .evaluate(frame.view(), &law, &ContextAccess::ContextSpecific)
        .expect("a declared law");

    // The sum of the per-context gradients, `−Σ_c π_c ∇_Q V_c`, from each context's own block.
    let mut per_context_sum = Array2::<f64>::zeros(frame.dim());
    let mut partial_sum = Array2::<f64>::zeros(frame.dim());
    for context in 0..contexts.context_count() {
        let weight = masses[context] / 4.0;
        let gradient = contexts
            .block(context)
            .explained_variance_gradient(frame.view())
            .expect("the frame")
            .horizontal_gradient;
        per_context_sum.scaled_add(-weight, &gradient);
        if context != 2 {
            partial_sum.scaled_add(-weight, &gradient);
        }
    }
    assert_eq!(report.horizontal_gradient, per_context_sum);

    // It is the gradient of the average: along each horizontal direction, the central difference of the averaged
    // error at `t = ±h` and `±h/2` brackets `⟨∇, Δ⟩`. `D_h − f′ = f‴h²/6 + O(h⁴)`, so the error of `D_{h/2}` is a
    // third of `|D_h − D_{h/2}|` to leading order; accepting within the whole difference leaves room for the next
    // order. Each value carries its rounding `δ ≤ γ_N Σ|terms|`: every unit pair contributes `|D_jk|·√(v_j v_k)·
    // (½ + 1/(2π))` to `V(I)` and to `V(P)`, and `N` counts at most 48 rounded operations per pair term (the retracted
    // frame's entries included) plus the additions over three contexts. The four values of one bracket then move it by
    // at most `4δ/h`.
    let block = overlapping_block();
    let metric_writers = block.metric.dot(&block.writers);
    let gram = block.writers.t().dot(&metric_writers);
    let variances: Vec<f64> = block.readers.rows().into_iter().map(|row| row.dot(&row)).collect();
    let mut absolute_terms = 0.0;
    for j in 0..4 {
        for k in 0..4 {
            absolute_terms += gram[[j, k]].abs() * (variances[j] * variances[k]).sqrt();
        }
    }
    let operations = 3 * 16 * 48 + 3 * 2 * 16;
    let value_rounding = accumulation_growth(operations) * 2.0 * absolute_terms * (0.5 + 1.0 / (2.0 * PI));
    let step = 0.125;
    let complement = reflector.slice(s![.., 2..]).to_owned();
    let averaged_error = |t: f64, direction: &Array2<f64>| {
        let moved = (&frame + &(direction * t)) / (1.0 + t * t).sqrt();
        contexts
            .evaluate(moved.view(), &law, &ContextAccess::ContextSpecific)
            .expect("a retracted frame")
            .error
    };
    let mut control_rejected = false;
    for direction in horizontal_directions(complement.view()) {
        let coarse = (averaged_error(step, &direction) - averaged_error(-step, &direction)) / (2.0 * step);
        let fine = (averaged_error(step / 2.0, &direction) - averaged_error(-step / 2.0, &direction)) / step;
        let tolerance = (coarse - fine).abs() + 4.0 * value_rounding / step;
        let analytic = (&report.horizontal_gradient * &direction).sum();
        assert!(
            (analytic - fine).abs() <= tolerance,
            "⟨∇, Δ⟩ = {analytic} vs central difference {fine} (Richardson {coarse}, tolerance {tolerance})",
        );
        let without_context = (&partial_sum * &direction).sum();
        control_rejected |= (without_context - fine).abs() > tolerance;
    }
    assert!(control_rejected, "a gradient missing one context's term must be rejected along some direction");
}

#[test]
fn an_unpriced_context_index_is_refused_and_a_priced_one_matches_the_context_specific_error() {
    let contexts = planted_contexts();
    let law = ContextLaw::Enumerated {
        masses: vec![1.0, 1.0],
    };
    let frame = frame_along(0, 3);

    let unpriced = contexts.evaluate(frame.view(), &law, &ContextAccess::PricedIndex { nats_per_row: 0.0 });
    let Err(ContextResponseError::UnderpricedContextIndex {
        nats_per_row,
        entropy_nats,
        band,
    }) = unpriced
    else {
        panic!("an unpriced index over two contexts must be refused, got {unpriced:?}");
    };
    assert_eq!(nats_per_row, 0.0);
    assert!((entropy_nats - LN_2).abs() <= band, "H(½, ½) = ln 2, got {entropy_nats} (band {band})");
    let underpriced = contexts.evaluate(frame.view(), &law, &ContextAccess::PricedIndex { nats_per_row: 0.5 * LN_2 });
    assert!(
        matches!(underpriced, Err(ContextResponseError::UnderpricedContextIndex { .. })),
        "a price below the entropy must be refused, got {underpriced:?}",
    );

    // Positive control: the refusal is the entropy floor, not a blanket refusal of the index.
    let priced = contexts
        .evaluate(frame.view(), &law, &ContextAccess::PricedIndex { nats_per_row: LN_2 })
        .expect("an index priced at the law's entropy");
    let specific = contexts
        .evaluate(frame.view(), &law, &ContextAccess::ContextSpecific)
        .expect("a context-specific explanation");
    assert_eq!(priced.error, specific.error);
    assert_eq!(priced.rows, specific.rows);
    assert_eq!(priced.access, ContextAccess::PricedIndex { nats_per_row: LN_2 });
    assert_eq!(priced.index_nats_per_row, Some(LN_2), "the report must carry the index's price");
    assert_eq!(specific.index_nats_per_row, None);
    assert_eq!(priced.weights, vec![0.5, 0.5]);
    let single = ContextBlocks::new(
        single_unit_block(),
        vec![ContextDeclaredLaw {
            baseline: Array1::zeros(3),
            loading: identity(3),
        }],
    )
    .expect("one context");
    let free = single.evaluate(
        frame.view(),
        &ContextLaw::Enumerated { masses: vec![1.0] },
        &ContextAccess::PricedIndex { nats_per_row: 0.0 },
    );
    assert!(free.is_ok(), "a one-context law has nothing to know, so its index is free: {free:?}");

    // A sampled population declares no law to price against.
    let sampled =
        contexts.evaluate(frame.view(), &ContextLaw::Sampled, &ContextAccess::PricedIndex { nats_per_row: 10.0 });
    assert!(
        matches!(sampled, Err(ContextResponseError::IndexPriceWithoutDeclaredLaw)),
        "an index price under a sampled law must be refused, got {sampled:?}",
    );
}

#[test]
fn sampled_contexts_estimate_the_declared_average_within_their_standard_error() {
    let population = overlapping_laws();
    let reflector = half_reflector();
    let frame = reflector.slice(s![.., ..2]).to_owned();
    let exact = overlapping_contexts(population.clone())
        .evaluate(
            frame.view(),
            &ContextLaw::Enumerated {
                masses: vec![6.0, 1.0, 1.0],
            },
            &ContextAccess::ContextSpecific,
        )
        .expect("the declared law");
    let uniform = overlapping_contexts(population.clone())
        .evaluate(
            frame.view(),
            &ContextLaw::Enumerated {
                masses: vec![1.0, 1.0, 1.0],
            },
            &ContextAccess::ContextSpecific,
        )
        .expect("the uniform law");

    // Draws from the declared law: slots 0..=5 of eight are context 0, slot 6 context 1 and slot 7 context 2.
    let draws = 1 << 14;
    let mut rng = StdRng::seed_from_u64(0x2946_c7);
    let sampled_laws: Vec<ContextDeclaredLaw> = std::iter::repeat_with(|| {
        let slot: u32 = rng.random_range(0..8);
        let context = match slot {
            0..=5 => 0,
            6 => 1,
            _ => 2,
        };
        population[context].clone()
    })
    .take(draws)
    .collect();
    let sampled = overlapping_contexts(sampled_laws)
        .evaluate(frame.view(), &ContextLaw::Sampled, &ContextAccess::ContextSpecific)
        .expect("independent draws");
    let standard_error = sampled.standard_error.expect("a sampled law reports its standard error");
    assert!(standard_error > 0.0, "the three contexts must discard different errors");
    let multiple = standard_error_multiple(1);
    assert!(
        (sampled.error - exact.error).abs() <= multiple * standard_error,
        "sampled {} ± {standard_error} vs declared {} (multiple {multiple})",
        sampled.error,
        exact.error,
    );
    // Positive control: the same contexts averaged under the wrong (uniform) law are rejected.
    assert!(
        (sampled.error - uniform.error).abs() > multiple * standard_error,
        "the uniform law's average {} must be rejected by sampled {} ± {standard_error}",
        uniform.error,
        sampled.error,
    );

    let one_draw = overlapping_contexts(population[..1].to_vec()).evaluate(
        frame.view(),
        &ContextLaw::Sampled,
        &ContextAccess::ContextSpecific,
    );
    assert!(
        matches!(one_draw, Err(ContextResponseError::TooFewDraws { draws: 1, required: 2 })),
        "one draw has no standard error, got {one_draw:?}",
    );
}
