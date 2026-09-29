//! Joint operators against executed rotate-half scoring, against dense Frobenius Grams, under
//! canonical gauge elements, and against the planted "same subspace, different law" pair.

use super::*;
use crate::parameter_decomposition::attention::{AffineProjection, RotaryPairing};
use crate::parameter_decomposition::canonical::{DecoderLayer, LayerRmsNorm, TensorDefect};
use crate::parameter_decomposition::gated_rewrite::rms_normalizers;
use crate::parameter_decomposition::gauge::SwigluUnits;
use crate::parameter_decomposition::state::resolve_stacked_factor;
use crate::parameter_decomposition::test_support::planted_toys::{ROUTING_HEADS, RoutingToy};
use crate::parameter_decomposition::test_support::test_governor;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn uniform(rng: &mut StdRng, rows: usize, cols: usize, low: f64, high: f64) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || rng.random_range(low..high))
}

/// Dyadic rationals `k/8` in `[−1, 1]`: every product and short sum a fixture forms is exact.
fn dyadic(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || f64::from(rng.random_range(-8_i32..=8)) / 8.0)
}

fn dyadic_gain(rng: &mut StdRng, len: usize) -> Array1<f64> {
    Array1::from_shape_simple_fn(len, || f64::from(rng.random_range(4_i32..=12)) / 8.0)
}

fn affine(weight: Array2<f64>, bias: Array1<f64>) -> AffineProjection {
    AffineProjection { weight, bias }
}

/// A GPT-NeoX-shaped block: biases, partial rotary (two planes of a six-wide head, two
/// pass-through coordinates), two query heads on one key/value head, `α ≠ 1`.
fn neox_block(pairing: RotaryPairing, seed: u64) -> NativeAttention {
    let mut rng = StdRng::seed_from_u64(seed);
    let g = AttentionGeometry {
        model_dim: 5,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 6,
    };
    let mut bias = |len| Array1::from_shape_simple_fn(len, || rng.random_range(-0.5..0.5));
    let (qb, kb, vb, ob) = (bias(12), bias(6), bias(6), bias(5));
    let mut rng = StdRng::seed_from_u64(seed + 1);
    NativeAttention::new(
        g,
        RotaryEmbedding {
            pairing,
            inverse_frequencies: vec![1.0, 0.03],
            attention_scaling: 1.25,
        },
        0.4,
        affine(uniform(&mut rng, 12, 5, -1.0, 1.0), qb),
        affine(uniform(&mut rng, 6, 5, -1.0, 1.0), kb),
        affine(uniform(&mut rng, 6, 5, -1.0, 1.0), vb),
        affine(uniform(&mut rng, 5, 12, -1.0, 1.0), ob),
    )
    .expect("fixture block")
}

fn homogeneous(row: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut out = row.to_vec();
    out.push(1.0);
    Array1::from(out)
}

/// With no query/key norm the executed score is exactly `σα² Σ_j (cos A + sin B) + σ Π` of the
/// homogeneous inputs at `Δ = p_s − p_t`: rotate-half and rotate-every-two alike.
#[test]
fn scores_match_executed_rotary_attention() {
    for (pairing, seed) in [(RotaryPairing::HalfSplit, 1), (RotaryPairing::Interleaved, 2)] {
        let native = neox_block(pairing, seed);
        let operators = query_key_operators(&native, None).expect("operators");
        assert!(operators.pass_through(0).is_some());
        assert_eq!(operators.cosine(0, 0).width(), 6, "homogeneous factors");
        let mut rng = StdRng::seed_from_u64(seed + 10);
        let x = uniform(&mut rng, 4, 5, -1.0, 1.0);
        let positions = [3_i64, 5, 6, 11];
        let executed = native.execute(test_governor(), x.view(), &positions).expect("executes");
        for head in 0..2 {
            for t in 0..4 {
                for s in 0..=t {
                    let (value, band) = operators
                        .score(head, homogeneous(x.row(t)).view(), homogeneous(x.row(s)).view(), positions[s] - positions[t])
                        .expect("score");
                    let (direct, radius) = (executed.scores[[head, t, s]], executed.score_radius[[head, t, s]]);
                    assert!((value - direct).abs() <= band + radius, "{pairing:?} head {head} ({t},{s}): {value} vs {direct}");
                    assert!(band + radius < 1e-12, "vacuous band {}", band + radius);
                }
            }
        }
    }
}

/// Behind Qwen3's query/key norm, with the input norm's gain folded into the columns, the
/// executed score equals `ν_q ν_k` times the operators' score at the gain-free core `z`
/// (`x = γ ⊙ z`). The dyadic fixture makes `x` and the projections `u = W x` exact, so `ν̂` is
/// within `γ_{d+4}` of `ν(u)` (`rms_normalizers`).
#[test]
fn normed_scores_exclude_only_the_normalizers() {
    let mut rng = StdRng::seed_from_u64(21);
    let g = AttentionGeometry {
        model_dim: 6,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
    };
    let native = NativeAttention::new(
        g,
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![0.5, 0.0625],
            attention_scaling: 1.0,
        },
        0.5,
        affine(dyadic(&mut rng, 16, 6), Array1::zeros(16)),
        affine(dyadic(&mut rng, 8, 6), Array1::zeros(8)),
        affine(dyadic(&mut rng, 8, 6), Array1::zeros(8)),
        affine(dyadic(&mut rng, 6, 16), Array1::zeros(6)),
    )
    .expect("block")
    .with_query_key_norm(0.015625, dyadic_gain(&mut rng, 4), dyadic_gain(&mut rng, 4))
    .expect("norm");
    let input_gain = dyadic_gain(&mut rng, 6);
    let core = dyadic(&mut rng, 3, 6);
    let x = &core * &input_gain;
    let positions = [0_i64, 2, 3];
    let executed = native.execute(test_governor(), x.view(), &positions).expect("executes");
    let operators = query_key_operators(&native, Some(input_gain.view())).expect("operators");
    let normalizers = |weight: &Array2<f64>, head: usize, row: usize| {
        let rows = weight.slice(s![head * 4..(head + 1) * 4, ..]);
        let u = rows.dot(&x.row(row));
        rms_normalizers(u.view().insert_axis(Axis(0)), 0.015625).expect("normalizer")[0]
    };
    let nu = accumulation_growth(4 + 4);
    for head in 0..4 {
        for t in 0..3 {
            for s in 0..=t {
                let scale = normalizers(&native.query().weight, head, t) * normalizers(&native.key().weight, g.key_value_head(head), s);
                let (value, band) = operators.score(head, core.row(t), core.row(s), positions[s] - positions[t]).expect("score");
                let joint = scale * value;
                let (direct, radius) = (executed.scores[[head, t, s]], executed.score_radius[[head, t, s]]);
                let allowance = scale * band + ((1.0 + nu) * (1.0 + nu) * (1.0 + accumulation_growth(2)) - 1.0) * joint.abs();
                assert!((joint - direct).abs() <= radius + allowance, "head {head} ({t},{s}): {joint} vs {direct}");
            }
        }
    }
}

/// The factored Gram against dense `d × d` operators formed in the test.
#[test]
fn gram_from_factors_matches_dense_operators() {
    let native = neox_block(RotaryPairing::HalfSplit, 31);
    let operators = query_key_operators(&native, None).expect("operators");
    let value_output = value_output_operators(&native, None).expect("operators");
    let family: Vec<&FactoredOperator> = vec![
        operators.cosine(0, 0),
        operators.sine(0, 0),
        operators.cosine(1, 1),
        operators.pass_through(1).expect("pass-through"),
        value_output.head(0),
        value_output.group(0),
    ];
    let gram = family_gram(test_governor(), &family).expect("gram");
    assert!(gram.reserved_bytes() > 0);
    let dense: Vec<Array2<f64>> = family.iter().map(|m| m.left().dot(&m.right().t())).collect();
    for i in 0..family.len() {
        for j in 0..family.len() {
            let direct = (&dense[i] * &dense[j]).sum();
            // The dense route's own rounding: each entry a rank-term product, then a width²-term sum.
            let magnitude = |m: &FactoredOperator| m.left().mapv(f64::abs).dot(&m.right().mapv(f64::abs).t());
            let reach = (&magnitude(family[i]) * &magnitude(family[j])).sum();
            let dense_band = accumulation_growth(2 * family[i].rank().max(family[j].rank()) + 36 + 2) * reach;
            assert!(
                (gram.gram[[i, j]] - direct).abs() <= gram.band[[i, j]] + dense_band,
                "({i},{j}): {} vs {direct}",
                gram.gram[[i, j]]
            );
        }
    }
}

/// Energies over a declared context: a plane whose wavelength exceeds it is content, the pass-
/// through coordinates are content, and the split sums to the total.
#[test]
fn energies_split_by_the_declared_context() {
    let native = neox_block(RotaryPairing::HalfSplit, 41);
    let operators = query_key_operators(&native, None).expect("operators");
    let energies = operators.energies(100.0).expect("declared context");
    for (head, energy) in energies.iter().enumerate() {
        // 2π/1 < 100 < 2π/0.03.
        assert_eq!(energy.slow_planes, vec![1]);
        let (a, b) = (frobenius_squared(operators.cosine(head, 1)).0, frobenius_squared(operators.sine(head, 1)).0);
        assert!((energy.content - (a + b)).abs() <= energy.band);
        assert!(energy.positional > 0.0 && energy.pass_through > 0.0);
    }
    for length in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(matches!(operators.energies(length), Err(JointRefusal::ContextLength { .. })));
    }
}

fn normed_layer(seed: u64) -> DecoderLayer {
    let mut rng = StdRng::seed_from_u64(seed);
    let g = AttentionGeometry {
        model_dim: 8,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
    };
    let mut gains = |len: usize| Array1::from_shape_simple_fn(len, || rng.random_range(0.5..1.5));
    let (query_gain, key_gain, input_gain, post_gain) = (gains(4), gains(4), gains(8), gains(8));
    let mut rng = StdRng::seed_from_u64(seed + 1);
    let attention = NativeAttention::new(
        g,
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![1.0, 0.1],
            attention_scaling: 1.0,
        },
        0.5,
        affine(uniform(&mut rng, 16, 8, -1.0, 1.0), Array1::zeros(16)),
        affine(uniform(&mut rng, 8, 8, -1.0, 1.0), Array1::zeros(8)),
        affine(uniform(&mut rng, 8, 8, -1.0, 1.0), Array1::zeros(8)),
        affine(uniform(&mut rng, 8, 16, -1.0, 1.0), Array1::zeros(8)),
    )
    .expect("block")
    .with_query_key_norm(1e-6, query_gain, key_gain)
    .expect("norm");
    let mlp = SwigluUnits::new(
        uniform(&mut rng, 6, 8, -1.0, 1.0),
        uniform(&mut rng, 6, 8, -1.0, 1.0),
        uniform(&mut rng, 8, 6, -1.0, 1.0),
    )
    .expect("mlp");
    DecoderLayer::native(LayerRmsNorm::native(1e-6, input_gain), &attention, LayerRmsNorm::native(1e-6, post_gain), &mlp).expect("layer")
}

fn relative_defect(defect: &TensorDefect) -> f64 {
    match defect {
        TensorDefect::Relative(relative) => *relative,
        TensorDefect::Exact => 0.0,
        TensorDefect::Entrywise(_) => f64::INFINITY,
    }
}

/// The canonical layer's joint operators equal the native layer's within the representation
/// defects of both; a non-gauge change moves one beyond them.
#[test]
fn joint_operators_are_gauge_invariant() {
    let native = normed_layer(51);
    let canonical = native.canonical().expect("canonical").layer;
    let native_block = native.attention().expect("block");
    let canonical_block = canonical.attention().expect("block");
    let before = query_key_operators(&native_block, Some(native.input_norm.gain.view())).expect("native operators");
    let after = query_key_operators(&canonical_block, None).expect("canonical operators");
    // Canonical factors: the folded rows (γ_1), the balanced gains (γ_1) and the factor's own
    // product (γ_1), `(1 + γ_1)³ ≤ 1 + γ_3`. Native factors: the gain and input-gain products (γ_2).
    let norm = canonical.query_key_norm.as_ref().expect("normed");
    assert_eq!(relative_defect(&canonical.query.weight_defect), accumulation_growth(1));
    assert_eq!(norm.query.gain_defect, accumulation_growth(1));
    assert_eq!(after.formation_defect(), accumulation_growth(1));
    let canonical_rows = accumulation_growth(3);
    for head in 0..4 {
        for plane in 0..2 {
            for (first, second) in [(before.cosine(head, plane), after.cosine(head, plane)), (before.sine(head, plane), after.sine(head, plane))] {
                let allowance = first.relative_defect_reach(before.formation_defect(), before.formation_defect())
                    + second.relative_defect_reach(canonical_rows, canonical_rows);
                let comparison = compare_operators(test_governor(), first, second).expect("comparison");
                let lower = comparison.difference.lower_bound().expect("exact");
                assert!(lower <= allowance, "head {head} plane {plane}: {lower:e} beyond {allowance:e}");
                let norm = comparison.first_norm.lower_bound().expect("exact");
                assert!(allowance < 1e-12 * norm, "allowance {allowance:e} is vacuous");
            }
        }
    }
    let (before_ov, after_ov) = (
        value_output_operators(&native_block, Some(native.input_norm.gain.view())).expect("native OV"),
        value_output_operators(&canonical_block, None).expect("canonical OV"),
    );
    let bound = |defect: &TensorDefect| match defect {
        TensorDefect::Entrywise(bound) => upper_frobenius(bound.view()),
        other => panic!("canonical OV defects are entrywise, not {other:?}"),
    };
    for head in 0..4 {
        let (first, second) = (before_ov.head(head), after_ov.head(head));
        let group = head / 2;
        let value_defect = match &canonical.value.weight_defect {
            TensorDefect::Entrywise(bound) => upper_frobenius(bound.slice(s![group * 4..(group + 1) * 4, ..])),
            other => panic!("canonical value defect is entrywise, not {other:?}"),
        };
        let output_defect = match &canonical.output.weight_defect {
            TensorDefect::Entrywise(bound) => upper_frobenius(bound.slice(s![.., head * 4..(head + 1) * 4])),
            other => panic!("canonical output defect is entrywise, not {other:?}"),
        };
        assert!(bound(&canonical.output.weight_defect) >= output_defect);
        let allowance = first.relative_defect_reach(0.0, before_ov.formation_defect())
            + entrywise_defect_reach(upper_frobenius(second.left()), upper_frobenius(second.right()), output_defect, value_defect);
        let comparison = compare_operators(test_governor(), first, second).expect("comparison");
        let lower = comparison.difference.lower_bound().expect("exact");
        assert!(lower <= allowance, "OV head {head}: {lower:e} beyond {allowance:e}");
        assert!(allowance < 1e-10 * comparison.first_norm.lower_bound().expect("exact"));
    }
    // Control: the query gain scaled on plane 0 without the key gain is not a gauge.
    let mut moved = canonical.clone();
    let (a, b) = native.rotary().plane(0);
    let gains = moved.query_key_norm.as_mut().expect("normed");
    gains.query.gain[a] *= 1.5;
    gains.query.gain[b] *= 1.5;
    let moved_operators = query_key_operators(&moved.attention().expect("block"), None).expect("operators");
    let comparison = compare_operators(test_governor(), after.cosine(0, 0), moved_operators.cosine(0, 0)).expect("comparison");
    assert!(comparison.proven_distinct());
    assert!((comparison.scale - 1.0 / 1.5).abs() < 1e-12, "a plane-uniform scale is proportional: {}", comparison.scale);
}

/// The planted toys' finding: two operators with one range and a factor 2 between them share
/// every subspace measure, yet they are distinct laws. The comparison certifies them distinct
/// and proportional with scale 1/2; an identical copy is not certified distinct.
#[test]
fn equal_subspace_is_not_equal_law() {
    let mut rng = StdRng::seed_from_u64(61);
    let first = FactoredOperator::new(uniform(&mut rng, 9, 2, -1.0, 1.0), uniform(&mut rng, 9, 2, -1.0, 1.0)).expect("operator");
    let doubled = FactoredOperator::new(first.left().mapv(|entry| 2.0 * entry), first.right().to_owned()).expect("operator");
    let comparison = compare_operators(test_governor(), &first, &doubled).expect("comparison");
    assert!(comparison.proven_distinct());
    assert!((comparison.scale - 0.5).abs() <= 1e-14);
    // The residual is not resolved from zero, and its band is small against the operator.
    let residual = &comparison.proportionality_residual;
    assert!(residual.lower_bound().expect("exact") <= 0.0);
    let upper = residual.upper_bound().expect("exact");
    assert!(upper <= 1e-10 * comparison.first_norm.lower_bound().expect("exact"), "residual band {upper:e}");
    let same = compare_operators(test_governor(), &first, &first.clone()).expect("comparison");
    assert!(!same.proven_distinct());
    // A different law with the same column space on both sides is distinct and not proportional.
    let mixed = FactoredOperator::new(first.left().dot(&Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.0, 1.0]).expect("shape")), first.right().to_owned())
        .expect("operator");
    let comparison = compare_operators(test_governor(), &first, &mixed).expect("comparison");
    assert!(comparison.proven_distinct());
    assert!(comparison.proportionality_residual.lower_bound().expect("exact") > 0.0);
}

/// A family whose Gram does not fit the governor's budget is refused before allocation.
#[test]
fn family_gram_is_governed() {
    let governor = MemoryGovernor::with_budget_bytes(64);
    let operator = FactoredOperator::new(Array2::ones((8, 2)), Array2::ones((8, 2))).expect("operator");
    assert!(matches!(family_gram(&governor, &[&operator, &operator]), Err(JointRefusal::Memory(_))));
}

/// Toy 5, the routing toy: heads 1 and 2 share query and key, head 3 reads query `2 Q₁`.
/// Every query/key operator of head 3 is head 1's times 2, so a subspace measure (the
/// Frobenius cosine `⟨M₁, M₃⟩/‖M₁‖‖M₃‖`, a captured energy, a top principal cosine) reads
/// all three heads as one pattern. Only operator equality separates them: heads 1 and 2
/// are not proven distinct on any plane, head 3 is proven distinct from both on every
/// plane and proportional with scale `1/2`. The routing laws are `{1, 2}` and `{3}`, and
/// the first law's transport `C₁ + C₂` has rank 4, above either head's 2.
#[test]
fn toy5_operator_equality_separates_laws_that_share_every_subspace() {
    let toy = RoutingToy::new(2951);
    let native = toy.native(&toy.value, &toy.output);
    let operators = query_key_operators(&native, None).expect("operators");
    let planes = operators.planes();
    let operators = &operators;
    let family: Vec<&FactoredOperator> = (0..ROUTING_HEADS)
        .flat_map(|head| (0..planes).flat_map(move |plane| [operators.cosine(head, plane), operators.sine(head, plane)]))
        .collect();
    let gram = family_gram(test_governor(), &family).expect("gram");
    let per_head = 2 * planes;
    for other in [1, 2] {
        for index in 0..per_head {
            let (i, j) = (index, other * per_head + index);
            let cosine = gram.gram[[i, j]] / (gram.gram[[i, i]] * gram.gram[[j, j]]).sqrt();
            // Each Gram entry is within its band; the cosine's first-order reach is their
            // sum over the smaller norm, and the square root and division round twice.
            let band = (gram.band[[i, j]] + gram.band[[i, i]] + gram.band[[j, j]]) / gram.gram[[i, i]].min(gram.gram[[j, j]])
                + accumulation_growth(4);
            assert!((1.0 - cosine).abs() <= band, "head {}: subspace cosine {cosine} of operator {index}", other + 1);
        }
    }

    let proven_distinct = |first: &FactoredOperator, second: &FactoredOperator| {
        compare_operators(test_governor(), first, second).expect("comparison").proven_distinct()
    };
    let mut laws: Vec<Vec<usize>> = vec![vec![0]];
    for head in 1..ROUTING_HEADS {
        let joined = laws.iter_mut().find(|law| {
            (0..planes).all(|plane| {
                !proven_distinct(operators.cosine(law[0], plane), operators.cosine(head, plane))
                    && !proven_distinct(operators.sine(law[0], plane), operators.sine(head, plane))
            })
        });
        match joined {
            Some(law) => law.push(head),
            None => laws.push(vec![head]),
        }
    }
    let truth = RoutingToy::truth();
    assert_eq!(laws, truth.laws);
    for plane in 0..planes {
        for (first, second) in [
            (operators.cosine(0, plane), operators.cosine(2, plane)),
            (operators.sine(0, plane), operators.sine(2, plane)),
        ] {
            let comparison = compare_operators(test_governor(), first, second).expect("comparison");
            assert!(comparison.proven_distinct());
            assert_eq!(comparison.scale, 1.0 / truth.third_head_scale, "dyadic factors make the least-squares scale exact");
            assert!(comparison.proportionality_residual.lower_bound().expect("exact") <= 0.0);
        }
    }

    let value_output = value_output_operators(&native, None).expect("value/output operators");
    let rank = |heads: &[usize]| {
        for &head in heads {
            let executed = value_output.head(head);
            let dense = executed.left().dot(&executed.right().t());
            assert_eq!(dense, RoutingToy::transport(&toy.value, &toy.output, &[head]), "the owner's C_h is O_h V_h");
        }
        let transport = RoutingToy::transport(&toy.value, &toy.output, heads);
        resolve_stacked_factor(test_governor(), &transport, 0.0).expect("rank").resolved_rank
    };
    assert_eq!(rank(&[0]), 2);
    assert_eq!(rank(&[1]), 2);
    assert_eq!(vec![rank(&truth.laws[0]), rank(&truth.laws[1])], truth.law_transport_ranks);
}
