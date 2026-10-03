//! Joint operators against executed rotate-half scoring, against dense Frobenius Grams, and
//! against the planted "same subspace, different law" pair.

use super::*;
use crate::attention::{AffineProjection, RotaryPairing};
use crate::gated_rewrite::rms_normalizers;
use gam_linalg::roundoff::resolved_singular_count;
use crate::test_support::planted_toys::{ROUTING_HEADS, RoutingToy};
use crate::test_support::test_governor;
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

    let truth = RoutingToy::truth();
    let laws = routing_laws(test_governor(), &native, operators).expect("routing laws");
    assert_eq!(laws.laws, truth.laws);
    assert_eq!(laws.law_of(1), Some(0));
    assert_eq!(laws.law_of(2), Some(1));
    assert!(!compare_heads(test_governor(), operators, 0, 1).expect("heads").proven_distinct());
    // One pattern at two temperatures: head 3's score operator is head 1's times 2.
    let [relation] = laws.relations.as_slice() else {
        panic!("expected one relation between two laws, got {:?}", laws.relations);
    };
    assert!(relation.comparison.proven_distinct() && relation.proportional());
    assert_eq!(relation.comparison.scale, truth.third_head_scale, "dyadic factors make the common scale exact");
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
        let singular_values = crate::dense::svd(transport.view(), false).expect("svd").singular_values;
        resolved_singular_count(&singular_values, transport.nrows(), transport.ncols(), 0.0)
    };
    assert_eq!(rank(&[0]), 2);
    assert_eq!(rank(&[1]), 2);
    assert_eq!(vec![rank(&truth.laws[0]), rank(&truth.laws[1])], truth.law_transport_ranks);
}

/// The routing-law letters of the toy: one per law, `C₁ + C₂` and `C₃`, formed from the
/// stored factors within their band; a key change that separates heads 1 and 2 splits the
/// first law, and a query/key norm adds the normalizers to the law test.
#[test]
fn attention_letters_are_one_summed_transport_per_law() {
    let toy = RoutingToy::new(2951);
    let native = toy.native(&toy.value, &toy.output);
    let letters = attention_letters(test_governor(), &native, None, None).expect("letters");
    assert_eq!(letters.laws.laws, RoutingToy::truth().laws);
    assert_eq!(letters.transports().len(), 2);
    for (law, (transport, &band)) in letters.laws.laws.iter().zip(letters.transports().iter().zip(letters.bands())) {
        let exact = RoutingToy::transport(&toy.value, &toy.output, law);
        let distance = (transport - &exact).iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!(distance <= band, "law {law:?}: {distance:e} beyond {band:e}");
    }

    // A post-norm output gain `ω` scales each law's transport rows: `diag(ω) T`, within the
    // bands, and leaves the laws alone. Dyadic `ω` keeps the scaled rows exact.
    let omega = Array1::from_shape_fn(toy.value[0].ncols(), |i| 0.5 + (i % 4) as f64 / 4.0);
    let scaled = attention_letters(test_governor(), &native, None, Some(omega.view())).expect("letters");
    assert_eq!(scaled.laws, letters.laws);
    for (law, (transport, &band)) in letters.laws.laws.iter().zip(scaled.transports().iter().zip(scaled.bands())) {
        let exact = &RoutingToy::transport(&toy.value, &toy.output, law) * &omega.view().insert_axis(Axis(1));
        let distance = (transport - &exact).iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!(distance <= band, "law {law:?}: {distance:e} beyond {band:e}");
    }
    assert!(matches!(
        attention_letters(test_governor(), &native, None, Some(Array1::ones(3).view())),
        Err(JointRefusal::Shape { what: "output gain", .. })
    ));

    let mut split = toy.clone();
    split.key[1][[0, 0]] += 0.25;
    let native = split.native(&split.value, &split.output);
    let letters = attention_letters(test_governor(), &native, None, None).expect("letters");
    assert_eq!(letters.laws.laws, vec![vec![0], vec![1], vec![2]]);
    assert!(letters.laws.relations.iter().filter(|relation| relation.law == 2 && relation.earlier == 0).all(|relation| relation.proportional()));

    // A plane scale `S = 2` on plane 0 of the second head's key with `S⁻ᵀ` on its query is a
    // rotary gauge: the operators, and so the law, do not move. Behind a query/key norm the raw query
    // rows now differ, so the normalizers differ and the heads route differently.
    let mut scaled = toy.clone();
    for row in [0, 2] {
        scaled.query[1].row_mut(row).mapv_inplace(|entry| entry * 0.5);
        scaled.key[1].row_mut(row).mapv_inplace(|entry| entry * 2.0);
    }
    let plain = attention_letters(test_governor(), &scaled.native(&scaled.value, &scaled.output), None, None).expect("letters");
    assert_eq!(plain.laws.laws, RoutingToy::truth().laws);
    let normed = scaled
        .native(&scaled.value, &scaled.output)
        .with_query_key_norm(1e-6, Array1::ones(4), Array1::ones(4))
        .expect("normed");
    let letters = attention_letters(test_governor(), &normed, None, None).expect("letters");
    assert_eq!(letters.laws.laws, vec![vec![0], vec![1], vec![2]]);
}
