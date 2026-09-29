//! Canonical forms against executed layers: function equality within derived radii, the
//! stated form of every canonical tensor, orbit invariance, refusals, and positive controls
//! whose non-gauge changes are refused by the gauge owner and detected by execution.

use super::*;
use crate::parameter_decomposition::attention::RotaryPairing;
use crate::parameter_decomposition::gated_rewrite::MaskedNorm;
use crate::parameter_decomposition::gauge::HiddenUnits;
use crate::parameter_decomposition::test_support::test_governor;
use gam_math::gaussian_activation::GaussianActivation;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const WIDTH: usize = 8;
const HIDDEN: usize = 6;
const POSITIONS: [i64; 5] = [0, 1, 2, 4, 7];

pub(super) fn uniform(rng: &mut StdRng, rows: usize, cols: usize, low: f64, high: f64) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || rng.random_range(low..high))
}

/// Gains in `[1/2, 3/2]` with coordinate `negative` sign-flipped.
fn gains(rng: &mut StdRng, len: usize, negative: usize) -> Array1<f64> {
    let mut gain = Array1::from_shape_simple_fn(len, || rng.random_range(0.5..1.5));
    gain[negative] = -gain[negative];
    gain
}

pub(super) fn geometry() -> AttentionGeometry {
    AttentionGeometry {
        model_dim: WIDTH,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
    }
}

/// A Qwen3-shaped layer: GQA with two query heads per key/value head, two rotary planes of
/// distinct frequency per head, an optional per-head query/key norm, bias-free projections.
pub(super) fn native_attention(rng: &mut StdRng, normed: bool) -> NativeAttention {
    let g = geometry();
    let affine = |weight: Array2<f64>| AffineProjection {
        bias: Array1::zeros(weight.nrows()),
        weight,
    };
    let native = NativeAttention::new(
        g,
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![1.0, 0.1],
            attention_scaling: 1.0,
        },
        0.5,
        affine(uniform(rng, g.query_dim(), WIDTH, -1.0, 1.0)),
        affine(uniform(rng, g.key_value_dim(), WIDTH, -1.0, 1.0)),
        affine(uniform(rng, g.key_value_dim(), WIDTH, -1.0, 1.0)),
        affine(uniform(rng, WIDTH, g.query_dim(), -1.0, 1.0)),
    )
    .expect("fixture tensors match the geometry");
    if normed {
        let (query_gain, key_gain) = (gains(rng, g.head_dim, 1), gains(rng, g.head_dim, 2));
        native
            .with_query_key_norm(1e-6, query_gain, key_gain)
            .expect("fixture gains match the head")
    } else {
        native
    }
}

fn layer(seed: u64, normed: bool) -> (DecoderLayer, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let attention = native_attention(&mut rng, normed);
    let mlp = SwigluUnits::new(
        uniform(&mut rng, HIDDEN, WIDTH, -1.0, 1.0),
        uniform(&mut rng, HIDDEN, WIDTH, -1.0, 1.0),
        uniform(&mut rng, WIDTH, HIDDEN, -1.0, 1.0),
    )
    .expect("fixture MLP shapes agree");
    let layer = DecoderLayer::native(
        LayerRmsNorm::native(1e-6, gains(&mut rng, WIDTH, 3)),
        &attention,
        LayerRmsNorm::native(1e-6, gains(&mut rng, WIDTH, 5)),
        &mlp,
    )
    .expect("fixture layer is consistent");
    let residual = uniform(&mut rng, POSITIONS.len(), WIDTH, -2.0, 2.0);
    (layer, residual)
}

fn execute(layer: &DecoderLayer, residual: &Array2<f64>) -> LayerExecution {
    layer
        .execute(test_governor(), ProjectedRows::exact(residual.view()), &POSITIONS)
        .expect("fixture layer executes")
}

fn largest(matrix: &Array2<f64>) -> f64 {
    matrix.iter().fold(0.0_f64, |best, &entry| best.max(entry.abs()))
}

/// Native and canonical layers agree within the sum of their radii, and the radii are tiny
/// against the output, so the agreement is not vacuous.
#[test]
fn canonical_layer_executes_the_native_function() {
    for (seed, normed) in [(11, true), (12, false), (13, true)] {
        let (native, residual) = layer(seed, normed);
        let canonical = native.canonical().expect("native fixture has a canonical form");
        let (before, after) = (execute(&native, &residual), execute(&canonical.layer, &residual));
        let (excess, at) = before.largest_excess(&after);
        assert!(excess <= 0.0, "seed {seed}: canonical output leaves the band by {excess:e} at {at:?}");
        let scale = largest(&before.output);
        for radius in [&before.output_radius, &after.output_radius] {
            assert!(largest(radius) < 1e-10 * scale, "seed {seed}: radius {} is vacuous", largest(radius));
        }
        assert_eq!(canonical.elements.query_key.is_some(), normed);
        assert!(canonical.elements.value_output_separated.iter().all(|&separated| separated));
    }
}

/// Every canonical tensor has the stated form, within its derived band.
#[test]
fn canonical_tensors_have_the_stated_form() {
    let (native, _) = layer(21, true);
    let canonical = native.canonical().expect("native fixture has a canonical form");
    let layer = &canonical.layer;
    // (a) unit gains, stored exactly; the element's inverse is the native gain.
    assert!(layer.input_norm.gain.iter().all(|&gain| gain == 1.0));
    assert!(layer.post_norm.gain.iter().all(|&gain| gain == 1.0));
    assert_eq!(canonical.elements.input_norm, native.input_norm.gain);
    // (b) ‖up_n‖ = ‖down_n‖ within the balance band.
    let units = layer.swiglu().expect("canonical MLP is consistent");
    let balance = balance_swiglu(&native.swiglu().expect("native MLP")).expect("balances");
    for unit in 0..HIDDEN {
        let up_norm = units.up().row(unit).dot(&units.up().row(unit)).sqrt();
        let down_norm = units.down().column(unit).dot(&units.down().column(unit)).sqrt();
        assert!((up_norm / down_norm - 1.0).abs() <= balance.balance_band(), "unit {unit}: {up_norm} vs {down_norm}");
        let lead = leading_index(units.up().row(unit));
        assert!(units.up()[[unit, lead]] > 0.0);
    }
    // (c) orthonormal value rows per group. Householder QR and the SVD each deliver an
    // orthonormal factor within `γ_{c m n}` (Higham, Thm 19.4); the band takes c = 4 per factor.
    let g = geometry();
    for group in 0..g.n_kv_heads {
        let rows = layer.value.weight.slice(s![group * g.head_dim..(group + 1) * g.head_dim, ..]);
        let gram = fast_abt(&rows, &rows) - Array2::<f64>::eye(g.head_dim);
        let band = 2.0 * accumulation_growth(4 * WIDTH * g.head_dim);
        assert!(largest(&gram) <= band, "group {group}: orthonormality defect {:e}", largest(&gram));
    }
    // (d) positive gains, and one geometric mean per plane for the query and key pairs.
    let norm = layer.query_key_norm.as_ref().expect("normed fixture");
    assert!(norm.query.gain.iter().chain(norm.key.gain.iter()).all(|&gain| gain > 0.0));
    for plane in 0..2 {
        let (a, b) = native.rotary().plane(plane);
        let query = norm.query.gain[a] * norm.query.gain[b];
        let key = norm.key.gain[a] * norm.key.gain[b];
        // Ratio, two square roots, the scale's two products per side and the check's products.
        assert!((query / key - 1.0).abs() <= accumulation_growth(12), "plane {plane}: {query} vs {key}");
    }
}

/// Moving the native layer along its declared orbits does not move its canonical form.
#[test]
fn canonical_form_is_orbit_invariant() {
    let (native, _) = layer(31, true);
    let mut rng = StdRng::seed_from_u64(32);
    let mut moved = native.clone();
    // Input norm: (w, W_k) ↦ (D w, W_k D⁻¹).
    let scales = Array1::from_shape_simple_fn(WIDTH, || rng.random_range(0.5..2.0));
    let norm = NormGain::new(
        native.input_norm.gain.clone(),
        None,
        vec![native.query.weight.clone(), native.key.weight.clone(), native.value.weight.clone()],
    )
    .expect("fixture norm")
    .apply(scales.view())
    .expect("nonzero scales are in the group");
    moved.input_norm.gain = norm.gain().to_owned();
    moved.query.weight = norm.reads()[0].clone();
    moved.key.weight = norm.reads()[1].clone();
    moved.value.weight = norm.reads()[2].clone();
    // SwiGLU: signed unit scales.
    let change = UnitGaugeChange {
        permutation: (0..HIDDEN).collect(),
        scales: Array1::from_shape_fn(HIDDEN, |unit| if unit % 2 == 0 { 1.7 } else { -0.3 }),
    };
    let units = native.swiglu().expect("native MLP").apply(&change).expect("nonzero scales");
    moved.up.weight = units.up().to_owned();
    moved.down.weight = units.down().to_owned();
    // Query/key: plane scales and a key gain flip.
    let block = NormedRotaryQueryKey::new(&moved.attention().expect("moved block")).expect("normed family");
    let turned = block
        .apply(&NormedRotaryChange {
            plane_scales: Array1::from(vec![1.3, 0.6]),
            quarter_turns: Array2::zeros((2, 2)),
            query_gain_flips: vec![false; 4],
            key_gain_flips: vec![true, false, false, false],
        })
        .expect("the element is in the group");
    moved.query.weight = turned.query_weight().to_owned();
    moved.key.weight = turned.key_weight().to_owned();
    let norm = moved.query_key_norm.as_mut().expect("normed fixture");
    norm.query.gain = turned.query_gain().to_owned();
    norm.key.gain = turned.key_gain().to_owned();

    let (first, second) = (
        native.canonical().expect("native canonical").layer,
        moved.canonical().expect("moved canonical").layer,
    );
    let relative = |left: &Array2<f64>, right: &Array2<f64>, band: f64, what: &str| {
        for ((index, &a), &b) in left.indexed_iter().zip(right.iter()) {
            assert!((a - b).abs() <= band * a.abs().max(b.abs()), "{what} {index:?}: {a} vs {b}");
        }
    };
    // Folded reads: `W diag(w)` against `(W / D)(D w)`, each within a few roundings.
    relative(&first.query.weight, &second.query.weight, accumulation_growth(6), "query");
    relative(&first.key.weight, &second.key.weight, accumulation_growth(6), "key");
    relative(&first.gate.weight, &second.gate.weight, accumulation_growth(2), "gate");
    let balance = accumulation_growth(4 * (WIDTH + WIDTH) + 40);
    relative(&first.up.weight, &second.up.weight, balance, "up");
    relative(&first.down.weight, &second.down.weight, balance, "down");
    let (first_norm, second_norm) = (
        first.query_key_norm.as_ref().expect("normed"),
        second.query_key_norm.as_ref().expect("normed"),
    );
    for (a, b) in first_norm.query.gain.iter().zip(second_norm.query.gain.iter()) {
        assert!((a - b).abs() <= accumulation_growth(16) * a.abs(), "query gain {a} vs {b}");
    }
}

/// A gate row scale under SiLU is not a gauge: the gauge owner refuses it, and the executed
/// layer moves beyond both radii.
#[test]
fn gate_row_scale_is_refused_and_detected() {
    let (native, residual) = layer(41, true);
    let units = HiddenUnits::new(
        native.gate.weight.clone(),
        Array1::zeros(HIDDEN),
        native.down.weight.clone(),
        GaussianActivation::Silu,
    )
    .expect("fixture units");
    let mut scales = Array1::ones(HIDDEN);
    scales[0] = 2.0;
    let refused = units.apply(&UnitGaugeChange {
        permutation: (0..HIDDEN).collect(),
        scales,
    });
    assert!(matches!(refused, Err(GaugeRefusal::NotInGroup { what: "hidden unit scale", index: 0, .. })));
    let mut moved = native.clone();
    moved.gate.weight.row_mut(0).mapv_inplace(|entry| 2.0 * entry);
    moved.down.weight.column_mut(0).mapv_inplace(|entry| entry / 2.0);
    let (excess, at) = execute(&native, &residual).largest_excess(&execute(&moved, &residual));
    assert!(excess > 0.0, "the non-gauge change was not detected ({excess:e} at {at:?})");
}

/// Non-gauge variants of the other three transforms are detected by execution: a gain folded
/// into its reads but also kept, a value block scaled without its output columns, and a query
/// gain scaled on one plane without the key gain.
#[test]
fn non_gauge_variants_are_detected() {
    let (native, residual) = layer(51, true);
    let reference = execute(&native, &residual);
    let mut doubled = native.clone();
    doubled.query.weight = &doubled.query.weight * &native.input_norm.gain;
    let mut value = native.clone();
    value.value.weight.slice_mut(s![0..4, ..]).mapv_inplace(|entry| 2.0 * entry);
    let mut plane = native.clone();
    let norm = plane.query_key_norm.as_mut().expect("normed fixture");
    let (a, b) = native.rotary().plane(0);
    norm.query.gain[a] *= 1.5;
    norm.query.gain[b] *= 1.5;
    for (what, moved) in [("kept gain", doubled), ("value only", value), ("query plane only", plane)] {
        let (excess, at) = reference.largest_excess(&execute(&moved, &residual));
        assert!(excess > 0.0, "{what}: not detected ({excess:e} at {at:?})");
    }
    // Quarter turns of both parities on a generic plane are outside the normed group.
    let block = NormedRotaryQueryKey::new(&native.attention().expect("block")).expect("normed family");
    let refused = block.apply(&NormedRotaryChange {
        plane_scales: Array1::ones(2),
        quarter_turns: Array2::from_shape_vec((2, 2), vec![1, 0, 0, 0]).expect("shape"),
        query_gain_flips: vec![false; 4],
        key_gain_flips: vec![false; 4],
    });
    assert!(matches!(refused, Err(GaugeRefusal::NotInGroup { .. })));
}

/// (a) on a GPT-NeoX LayerNorm with bias. The normalized core `z = fl(c ν)` is computed before
/// the gain, identically for the native and the unit gain, so both reads start from one `z`:
/// native `W fl(fl(w z) + β)` represents `W (w z + β)` within `γ_2 (|w z| + |β|)` on its
/// input, and canonical `W′ fl(z + β′)` represents `W diag(w) (z + β ⊘ w)`, the same map, with
/// the defects of `W′` and `β′`.
#[test]
fn layer_norm_fold_executes_the_native_read() {
    let mut rng = StdRng::seed_from_u64(61);
    let gain = gains(&mut rng, WIDTH, 2);
    let bias = Array1::from_shape_simple_fn(WIDTH, || rng.random_range(-1.0..1.0));
    let read = uniform(&mut rng, 5, WIDTH, -1.0, 1.0);
    let residual = uniform(&mut rng, 4, WIDTH, -2.0, 2.0);
    let folded = fold_norm_gain(&NormGain::new(gain.clone(), Some(bias.clone()), vec![read.clone()]).expect("norm"))
        .expect("nonzero gains fold");
    let canonical = folded.canonical();
    assert!(canonical.gain().iter().all(|&entry| entry == 1.0));
    let zeros = Array1::zeros(WIDTH);
    let ones = Array1::ones(WIDTH);
    let core = MaskedNorm::Layer {
        epsilon: 1e-5,
        gain: ones.view(),
        bias: zeros.view(),
    }
    .apply(residual.view())
    .expect("finite rows");
    let native_rows = MaskedNorm::Layer {
        epsilon: 1e-5,
        gain: gain.view(),
        bias: bias.view(),
    }
    .apply(residual.view())
    .expect("finite rows");
    let two = accumulation_growth(2);
    let native_radius = Zip::from(&core)
        .and(&gain.broadcast(core.raw_dim()).expect("row gain"))
        .and(&bias.broadcast(core.raw_dim()).expect("row bias"))
        .map_collect(|&z, &w, &beta| up(two * up(up((w * z).abs()) + beta.abs())));
    let shifted_bias = canonical.bias().expect("LayerNorm bias").to_owned();
    let canonical_rows = &core + &shifted_bias;
    let one = accumulation_growth(1);
    let canonical_radius = Zip::from(&canonical_rows)
        .and(&shifted_bias.broadcast(core.raw_dim()).expect("row bias"))
        .map_collect(|&y, &beta| up(up(one * y.abs()) + up(folded.defect() * beta.abs())));
    let (native_rows_copy, native_radius_copy) = (native_rows.clone(), native_radius.clone());
    let native_read = LayerProjection::bias_free(read.clone(), TensorDefect::Exact);
    let canonical_read = LayerProjection::bias_free(canonical.reads()[0].clone(), TensorDefect::Relative(folded.defect()));
    let before = native_read
        .read(test_governor(), &Banded { values: native_rows, radius: native_radius })
        .expect("read");
    let after = canonical_read
        .read(test_governor(), &Banded { values: canonical_rows, radius: canonical_radius })
        .expect("read");
    for (((&a, &b), &ra), &rb) in before.values.iter().zip(after.values.iter()).zip(before.radius.iter()).zip(after.radius.iter()) {
        assert!((a - b).abs() <= ra + rb, "{a} vs {b} beyond {}", ra + rb);
    }
    // Control: the folded read of the native rows counts the gain twice.
    let wrong = LayerProjection::bias_free(canonical.reads()[0].clone(), TensorDefect::Exact)
        .read(test_governor(), &Banded { values: native_rows_copy, radius: native_radius_copy })
        .expect("read");
    let separated = wrong
        .values
        .iter()
        .zip(before.values.iter())
        .zip(wrong.radius.iter().zip(before.radius.iter()))
        .any(|((&a, &b), (&ra, &rb))| (a - b).abs() > ra + rb);
    assert!(separated, "a twice-applied gain was not detected");
}

/// Typed refusals: a zero gain, a layer already carrying defects, a value block without full
/// row rank.
#[test]
fn canonical_forms_refuse_what_they_do_not_derive() {
    let (native, _) = layer(71, true);
    let mut zero = native.input_norm.gain.clone();
    zero[4] = 0.0;
    let refused = fold_norm_gain(&NormGain::new(zero, None, vec![native.query.weight.clone()]).expect("norm"));
    assert!(matches!(refused, Err(CanonicalRefusal::ZeroNormGain { coordinate: 4 })));
    let canonical = native.canonical().expect("canonical").layer;
    assert!(matches!(canonical.canonical(), Err(CanonicalRefusal::NotNative { .. })));
    let mut deficient = native.value.affine();
    deficient.weight.row_mut(5).fill(0.0);
    let refused = canonical_value_output(geometry(), &deficient, &native.output.affine(), &TensorDefect::Exact);
    assert!(matches!(refused, Err(CanonicalRefusal::ValueRank { group: 1, .. })));
}

/// The canonical value/output form of the routing toy's rank-2 heads, one head per group,
/// and the per-group transports `O′_h V′_h` it determines, each within the band its defects
/// derive: `|O′ − O T| |V′| + (|O′| + δ_O) |V′ − T⁻¹V| + γ_r |O′| |V′|`.
fn routing_canonical_transports(value: &[Array2<f64>], output: &[Array2<f64>]) -> Vec<(Array2<f64>, Array2<f64>)> {
    use crate::parameter_decomposition::test_support::planted_toys::{ROUTING_HEADS, ROUTING_RANK, ROUTING_WIDTH, RoutingToy};
    let (d, r) = (ROUTING_WIDTH, ROUTING_RANK);
    let geometry = AttentionGeometry {
        model_dim: d,
        n_heads: ROUTING_HEADS,
        n_kv_heads: ROUTING_HEADS,
        head_dim: r,
    };
    let stacked_value = concatenate(Axis(0), &value.iter().map(|block| block.view()).collect::<Vec<_>>()).expect("stack");
    let stacked_output = concatenate(Axis(1), &output.iter().map(|block| block.view()).collect::<Vec<_>>()).expect("stack");
    let affine = |weight: Array2<f64>| AffineProjection {
        bias: Array1::zeros(weight.nrows()),
        weight,
    };
    let canonical = canonical_value_output(geometry, &affine(stacked_value), &affine(stacked_output), &TensorDefect::Exact)
        .expect("full-rank heads have a canonical form");
    assert!(canonical.separated.iter().all(|&separated| separated));
    (0..ROUTING_HEADS)
        .map(|head| {
            let rows = head * r..(head + 1) * r;
            let value_prime = canonical.value.weight.slice(s![rows.clone(), ..]);
            let output_prime = canonical.output.weight.slice(s![.., rows.clone()]);
            let (value_defect, output_defect) =
                (canonical.value_defect.slice(s![rows.clone(), ..]), canonical.output_defect.slice(s![.., rows]));
            let transport = output_prime.dot(&value_prime);
            let absolute_value = value_prime.mapv(f64::abs);
            let absolute_output = output_prime.mapv(f64::abs);
            let band = output_defect.dot(&absolute_value)
                + (&absolute_output + &output_defect).dot(&value_defect)
                + absolute_output.dot(&absolute_value).mapv(|entry| accumulation_growth(r + 2) * entry);
            let exact = RoutingToy::transport(value, output, &[head]);
            for ((index, &computed), (&exact, &band)) in transport.indexed_iter().zip(exact.iter().zip(band.iter())) {
                assert!((computed - exact).abs() <= band, "head {head} {index:?}: {computed} vs {exact} beyond {band:e}");
            }
            (transport, band)
        })
        .collect()
}

/// Toy 5's value/output gauges against the canonical form, which quotients the declared
/// per-group `GL(r)` and nothing more (module docs, *Scope*).
/// - A per-head `S_h` moves every head and no function: the executed block agrees within
///   its radii, and the canonical form carries the same per-group transports.
/// - The cross-head `GL(4)` of heads 1 and 2, whose patterns are identical, also moves no
///   function (their summed transport is unchanged exactly), but it moves each head's own
///   transport, which the canonical form determines. So one function has two canonical
///   forms: equal canonical forms imply equal functions, not the converse.
#[test]
fn toy5_cross_head_transport_gauge_is_outside_the_canonical_form() {
    use crate::parameter_decomposition::test_support::planted_toys::{ROUTING_HEADS, RoutingToy};
    let toy = RoutingToy::new(2951);
    let positions = RoutingToy::positions();
    let base = routing_canonical_transports(&toy.value, &toy.output);
    let native = toy.native(&toy.value, &toy.output);
    let (per_head_value, per_head_output) = toy.per_head_gauge(5);
    let (cross_value, cross_output) = toy.cross_head_gauge(6);
    for (value, output, what) in [(&per_head_value, &per_head_output, "per-head"), (&cross_value, &cross_output, "cross-head")] {
        let moved = toy.native(value, output);
        for x in &toy.sequences {
            let before = native.execute(test_governor(), x.view(), &positions).expect("executes");
            let after = moved.execute(test_governor(), x.view(), &positions).expect("executes");
            for (((&a, &b), &ra), &rb) in
                before.output.iter().zip(after.output.iter()).zip(before.output_radius.iter()).zip(after.output_radius.iter())
            {
                assert!((a - b).abs() <= ra + rb, "{what}: {a} vs {b} beyond {}", ra + rb);
            }
        }
        assert_eq!(
            RoutingToy::transport(value, output, &[0, 1]),
            RoutingToy::transport(&toy.value, &toy.output, &[0, 1]),
            "{what}: the law's transport is carried exactly"
        );
    }
    // Two canonical transports agree when every entry is within the sum of their bands, and
    // are certified distinct when some entry is beyond it.
    let distinct = |first: &(Array2<f64>, Array2<f64>), second: &(Array2<f64>, Array2<f64>)| {
        first.0.iter().zip(second.0.iter()).zip(first.1.iter().zip(second.1.iter())).any(|((&a, &b), (&ra, &rb))| (a - b).abs() > ra + rb)
    };
    let per_head = routing_canonical_transports(&per_head_value, &per_head_output);
    let cross = routing_canonical_transports(&cross_value, &cross_output);
    for head in 0..ROUTING_HEADS {
        assert!(!distinct(&per_head[head], &base[head]), "head {head}: the per-head gauge is quotiented");
    }
    assert!(!distinct(&cross[2], &base[2]), "head 3 is untouched");
    for head in 0..2 {
        assert!(distinct(&cross[head], &base[head]), "head {head}: the cross-head gauge moves its canonical transport");
    }
}
