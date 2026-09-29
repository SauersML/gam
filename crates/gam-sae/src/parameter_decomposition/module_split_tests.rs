#![cfg(test)]
//! Known-answer toys for the additive module split (toys 1–4 and 7 of the planted
//! suite, receipt `experiments/issue-2951/receipts/opfirst_toys_planted.json`, and
//! planted blocks under dense integer mixing) and its controls.

use super::*;
use crate::parameter_decomposition::state::{ObservabilityStep, WeightedObservability};
use crate::parameter_decomposition::fibre::parameter_fibre;
use crate::parameter_decomposition::test_support::planted_toys::{
    MlpToy, RANDOM_NULL_SEED, cross_edge, hadamard, hadamard_modules, hadamard_modules_truth, paired_copy, paired_copy_truth,
    random_mlp, rotation_toy,
};
use crate::parameter_decomposition::test_support::test_governor;
use crate::response::interaction::connected_components;
use ndarray::{Array1, Array2, Axis, array, concatenate, s};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

struct Block {
    w_in: Array2<f64>,
    b_in: Array1<f64>,
    w_out: Array2<f64>,
    b_out: Array1<f64>,
}

impl Block {
    fn evaluate(&self, activation: GaussianActivation, x: ArrayView1<'_, f64>) -> Array1<f64> {
        let hidden = (self.w_in.dot(&x) + &self.b_in).mapv(|t| activation_value(activation, t).expect("σ"));
        self.w_out.dot(&hidden) + &self.b_out
    }

    fn normal_form(&self, activation: GaussianActivation, skip: Option<&Array2<f64>>) -> MlpNormalForm {
        MlpNormalForm::new(
            activation,
            self.w_in.view(),
            self.b_in.view(),
            self.w_out.view(),
            self.b_out.view(),
            skip.map(|skip| skip.view()),
        )
        .expect("normal form")
    }

    fn of(toy: &MlpToy) -> Self {
        Self {
            w_in: toy.w_in.clone(),
            b_in: toy.b_in.clone(),
            w_out: toy.w_out.clone(),
            b_out: toy.b_out.clone(),
        }
    }

    fn with_reads(&self, w_in: Array2<f64>) -> Self {
        Self {
            w_in,
            b_in: self.b_in.clone(),
            w_out: self.w_out.clone(),
            b_out: self.b_out.clone(),
        }
    }
}

fn exact_parts(status: &EvidenceStatus<(), SplitDomain>) -> (f64, f64) {
    match status {
        EvidenceStatus::Exact {
            value,
            numerical_error,
            ..
        } => (*value, *numerical_error),
        other => panic!("expected an exact algebraic value, got {other:?}"),
    }
}

/// A unimodular dense integer mixing: a product of integer shears, so every
/// mixed read stays an exact integer vector.
fn unimodular(width: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut mixing = Array2::<f64>::eye(width);
    for _ in 0..3 {
        let mut upper = Array2::<f64>::eye(width);
        let mut lower = Array2::<f64>::eye(width);
        for i in 0..width {
            for j in 0..width {
                let draw = rng.random_range(-1_i32..=1) as f64;
                if i < j {
                    upper[[i, j]] = draw;
                } else if i > j {
                    lower[[i, j]] = draw;
                }
            }
        }
        mixing = mixing.dot(&upper).dot(&lower);
    }
    mixing
}

/// Planted blocks of read dimensions 2, 2 and 1 on `ℝ⁵`: three generic integer
/// reads in each plane (a connected triple) and two parallel reads with
/// different biases on the line, then a dense unimodular mixing. Returns the
/// block and every unit's block.
fn planted_blocks(seed: u64, bridge: bool) -> (Block, Vec<usize>) {
    let plane = [[1.0, 2.0], [3.0, -1.0], [1.0, 1.0]];
    let mut reads: Vec<(Array1<f64>, usize)> = Vec::new();
    for (block, offset) in [(0, 0), (1, 2)] {
        for pair in plane {
            let mut read = Array1::<f64>::zeros(5);
            read[offset] = pair[0];
            read[offset + 1] = pair[1] + block as f64;
            reads.push((read, block));
        }
    }
    for scale in [1.0, 2.0] {
        let mut read = Array1::<f64>::zeros(5);
        read[4] = scale;
        reads.push((read, 2));
    }
    if bridge {
        reads.push((array![1.0, 0.0, 1.0, 0.0, 0.0], 0));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    reads.shuffle(&mut rng);
    let mixing = unimodular(5, seed);
    let hidden = reads.len();
    let mut w_in = Array2::<f64>::zeros((hidden, 5));
    let mut truth = Vec::with_capacity(hidden);
    for (unit, (read, block)) in reads.into_iter().enumerate() {
        w_in.row_mut(unit).assign(&read.dot(&mixing));
        truth.push(block);
    }
    let block = Block {
        w_in,
        b_in: Array1::from_shape_fn(hidden, |unit| (unit as f64 - 4.0) / 8.0),
        w_out: Array2::from_shape_simple_fn((3, hidden), || rng.random_range(-4_i32..=4) as f64 / 4.0),
        b_out: array![0.25, -0.5, 0.125],
    };
    (block, truth)
}

fn partition_of(truth: &[usize]) -> Vec<Vec<usize>> {
    let mut blocks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (unit, &block) in truth.iter().enumerate() {
        blocks.entry(block).or_default().push(unit);
    }
    let mut partition: Vec<Vec<usize>> = blocks.into_values().collect();
    partition.sort();
    partition
}

fn sorted(mut partition: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    partition.sort();
    partition
}

fn random_block(seed: u64, hidden: usize, width: usize) -> Block {
    let mut rng = StdRng::seed_from_u64(seed);
    Block {
        w_in: Array2::from_shape_simple_fn((hidden, width), || rng.random_range(-1.0..1.0)),
        b_in: Array1::from_shape_simple_fn(hidden, || rng.random_range(-1.0..1.0)),
        w_out: Array2::from_shape_simple_fn((3, hidden), || rng.random_range(-1.0..1.0)),
        b_out: Array1::from_shape_simple_fn(3, || rng.random_range(-1.0..1.0)),
    }
}

/// A rounding level for evaluating a block at `‖h‖ ≤ radius`: the evaluation's
/// depth times its absolute terms.
fn evaluation_level(form: &MlpNormalForm, reads: &Array2<f64>, radius: f64) -> f64 {
    let (units, input) = reads.dim();
    let pre: f64 = reads.rows().into_iter().map(|row| norm(row) * radius).sum::<f64>()
        + form.biases.iter().map(|value| value.abs()).sum::<f64>();
    let writes: f64 = form.writes.iter().map(|value| value.abs()).sum();
    accumulation_growth(4 * (units + input + 4)) * (1.0 + pre) * (1.0 + writes)
}

#[test]
fn lipschitz_constant_comes_from_the_activation_owner() {
    let sqrt2 = std::f64::consts::SQRT_2;
    let (cdf, pdf) = normal_cdf_and_pdf(sqrt2);
    let gelu = slope_bound(GaussianActivation::ExactGelu).expect("gelu");
    let closed = cdf + sqrt2 * pdf;
    assert!(gelu >= closed && gelu - closed <= accumulation_growth(16) * closed, "{gelu} vs {closed}");
    let relu = slope_bound(GaussianActivation::Relu).expect("relu");
    assert!((1.0..=1.0 + accumulation_growth(4)).contains(&relu));
    assert!(matches!(
        slope_bound(GaussianActivation::Silu),
        Err(ModuleSplitError::Activation { .. })
    ));
}

/// Opposite forms merge by adding writes and moving `−v wᵀ` into `L`;
/// duplicates add writes; the merged form reproduces the block.
#[test]
fn normal_form_merges_opposite_forms_into_the_linear_part() {
    let mut block = random_block(29, 5, 4);
    let (negated, bias) = (block.w_in.row(0).mapv(|value| -value), -block.b_in[0]);
    block.w_in.row_mut(3).assign(&negated);
    block.b_in[3] = bias;
    let duplicate = block.w_in.row(1).to_owned();
    block.w_in.row_mut(4).assign(&duplicate);
    block.b_in[4] = block.b_in[1];
    let mut rng = StdRng::seed_from_u64(30);
    for activation in [GaussianActivation::ExactGelu, GaussianActivation::Relu] {
        let form = block.normal_form(activation, None);
        assert_eq!(form.reads.nrows(), 3);
        let merged = form
            .sources
            .iter()
            .position(|sources| sources.iter().any(|source| source.unit == 3))
            .expect("unit 3 merged");
        let expected = &block.w_out.column(0) + &block.w_out.column(3);
        assert_eq!(form.writes.row(merged), expected);
        assert!(form.sources[merged].iter().any(|source| source.negated));
        for _ in 0..20 {
            let x = Array1::from_shape_simple_fn(4, || rng.random_range(-3.0..3.0));
            let direct = block.evaluate(activation, x.view());
            let normal = form.evaluate(x.view()).expect("evaluate");
            let level = evaluation_level(&form, &block.w_in, 3.0 * 2.0);
            for (&left, &right) in direct.iter().zip(normal.iter()) {
                assert!((left - right).abs() <= level, "{activation:?}: {left} vs {right}");
            }
        }
    }
}

/// Toy 1: `F(h) = σ(h) − σ(−h) = h`. Every pair merges, `L = I` exactly, every
/// write cancels, no unit is left: a linear block, split by any partition, so
/// there is nothing to identify. Merged, a one-row readout observes one
/// direction; the unmerged units would claim all `d`.
///
/// Design rule: sign duplicates merge before any unit graph is read. The raw reads
/// `[I; −I]` have the projector `½ [[I, −I], [−I, I]]`, whose pattern falsely splits the
/// block into `d` modules `{i, d + i}`, each an exact split at zero cost; the merged
/// form has no unit, and the module split refuses it.
#[test]
fn toy1_paired_copy_is_a_linear_block() {
    let width = 8;
    let truth = paired_copy_truth(width);
    let block = Block::of(&paired_copy(width));
    let raw_projector = block.w_in.dot(&block.w_in.t()) * 0.5;
    let raw_pairs = (0..2 * width)
        .flat_map(|i| ((i + 1)..2 * width).map(move |j| (i, j)))
        .filter(|&(i, j)| raw_projector[[i, j]] != 0.0);
    assert_eq!(connected_components(2 * width, raw_pairs).len(), width, "the unmerged graph claims d modules");
    let form = block.normal_form(GaussianActivation::ExactGelu, None);
    assert_eq!(form.reads.nrows(), truth.merged_units);
    assert_eq!(form.cancelled.len(), 2 * width);
    assert_eq!(form.linear, truth.linear);
    assert!(matches!(form.additive_blocks(test_governor()), Err(ModuleSplitError::NoUnits)));

    let readout = array![[0.3, -1.1, 0.4, 0.9, -0.2, 0.7, 0.05, -0.6]];
    let merged = WeightedObservability::pull_back(
        test_governor(),
        &[ObservabilityStep {
            letters: form.observability_letters(),
            readouts: vec![readout.view()],
        }],
    )
    .expect("pull back");
    assert_eq!(merged.spectrum().resolved_rank, 1);
    let raw_writes = block.w_out.t().to_owned();
    let unmerged = WeightedObservability::pull_back(
        test_governor(),
        &[ObservabilityStep {
            letters: vec![
                ObservabilityLetter::Linear(form.linear.view()),
                ObservabilityLetter::Units {
                    reads: block.w_in.view(),
                    writes: raw_writes.view(),
                },
            ],
            readouts: vec![readout.view()],
        }],
    )
    .expect("pull back");
    assert_eq!(unmerged.spectrum().resolved_rank, width);
}

/// Planted blocks 2, 2, 1 under dense integer mixing are the finest
/// components; each one's optimal split is exact within its derived error, and
/// the Laplacian annihilates each block's indicator.
#[test]
fn planted_blocks_are_the_finest_components() {
    for activation in [GaussianActivation::ExactGelu, GaussianActivation::Relu] {
        let (block, truth) = planted_blocks(2951, false);
        let form = block.normal_form(activation, None);
        let blocks = form.additive_blocks(test_governor()).expect("blocks");
        assert_eq!(sorted(blocks.finest.clone()), partition_of(&truth), "{activation:?}");
        assert!(matches!(blocks.rank, EvidenceStatus::Exact { value, .. } if value == 5.0));
        assert_eq!(blocks.free_input_dimension, 0);
        assert_eq!(blocks.unresolved_joins.len(), 3);
        for join in &blocks.unresolved_joins {
            assert!(join.largest <= join.largest_band);
        }
        for component in &blocks.components {
            let (loss, error) = exact_parts(&component.loss);
            assert!(loss <= error, "E* = {loss:e} beyond its error {error:e}");
            let (cut, cut_error) = exact_parts(&component.cut);
            assert!(cut <= cut_error);
            let indicator = Array1::from_shape_fn(truth.len(), |unit| {
                if component.subset.contains(&unit) { 1.0 } else { 0.0 }
            });
            let applied = blocks.laplacian_apply(indicator.view()).expect("laplacian");
            let tolerance = 4.0 * blocks.projector_band * truth.len() as f64;
            assert!(applied.iter().all(|value| value.abs() <= tolerance), "{applied:?}");
        }
    }
}

/// A read across two planted planes merges their blocks and nothing else.
#[test]
fn a_bridge_read_merges_two_blocks() {
    let (block, truth) = planted_blocks(7, true);
    let form = block.normal_form(GaussianActivation::ExactGelu, None);
    let blocks = form.additive_blocks(test_governor()).expect("blocks");
    let merged: Vec<usize> = truth.iter().map(|&block| if block == 2 { 1 } else { 0 }).collect();
    assert_eq!(sorted(blocks.finest.clone()), partition_of(&merged));
}

/// `Π`, the components and the split quantities do not move under eight random
/// shears of the input, `W → W S`, and `Ŵ` moves equivariantly, `Ŵ → Ŵ S`.
#[test]
fn blocks_are_invariant_and_splits_equivariant_under_shears() {
    let (block, _) = planted_blocks(11, false);
    let form = block.normal_form(GaussianActivation::ExactGelu, None);
    let base = form.additive_blocks(test_governor()).expect("blocks");
    let subset = [0, 3, 5, 6];
    let reference = form.optimal_split(test_governor(), &base, &subset).expect("split");
    let projector = base.frame.dot(&base.frame.t());
    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..8 {
        let mut shear = Array2::<f64>::eye(5);
        for i in 0..5 {
            for j in (i + 1)..5 {
                shear[[i, j]] = rng.random_range(-1.0..1.0);
            }
        }
        let moved_form = block.with_reads(block.w_in.dot(&shear)).normal_form(GaussianActivation::ExactGelu, None);
        let moved = moved_form.additive_blocks(test_governor()).expect("blocks");
        assert_eq!(sorted(moved.finest.clone()), sorted(base.finest.clone()));
        let moved_projector = moved.frame.dot(&moved.frame.t());
        let band = 2.0 * (base.projector_band + moved.projector_band);
        assert!((&moved_projector - &projector).iter().all(|value| value.abs() <= band));

        let split = moved_form.optimal_split(test_governor(), &moved, &subset).expect("split");
        let (loss, loss_error) = exact_parts(&split.loss);
        let (base_loss, base_error) = exact_parts(&reference.loss);
        assert!((loss - base_loss).abs() <= loss_error + base_error);
        let (cut, cut_error) = exact_parts(&split.cut);
        let (base_cut, base_cut_error) = exact_parts(&reference.cut);
        assert!((cut - base_cut).abs() <= cut_error + base_cut_error);
        let carried = reference.approximate_reads.dot(&shear);
        let scale: f64 = carried.iter().map(|value| value.abs()).sum::<f64>() + 1.0;
        let tolerance = (loss_error + base_error).sqrt() * scale + band * scale;
        assert!(
            (&split.approximate_reads - &carried).iter().all(|value| value.abs() <= tolerance),
            "Ŵ is not carried by the shear"
        );
    }
}

/// On a dense random block: `E*` equals the directly computed loss of `Û`,
/// `χ` equals the directly summed cut and bounds `E*/2`, no other projector of
/// the same rank does better, and the native bound holds on sampled ball
/// points.
#[test]
fn optimal_split_matches_the_direct_loss_and_bounds_the_native_error() {
    let block = random_block(41, 10, 4);
    let form = block.normal_form(GaussianActivation::ExactGelu, None);
    let blocks = form.additive_blocks(test_governor()).expect("blocks");
    assert_eq!(blocks.finest.len(), 1);
    let subset = [1, 2, 4, 7, 8];
    let split = form.optimal_split(test_governor(), &blocks, &subset).expect("split");
    let frame = &blocks.frame;
    let inside = |unit: usize| subset.contains(&unit);
    let loss_of = |basis: &Array2<f64>| {
        let projected = frame.dot(&basis.t()).dot(basis);
        (0..frame.nrows())
            .map(|unit| {
                let residual = if inside(unit) {
                    &frame.row(unit) - &projected.row(unit)
                } else {
                    projected.row(unit).to_owned()
                };
                residual.dot(&residual)
            })
            .sum::<f64>()
    };
    let (loss, loss_error) = exact_parts(&split.loss);
    let direct = loss_of(&split.projector_basis);
    assert!((direct - loss).abs() <= loss_error, "direct {direct} vs E* {loss}");
    let projector = frame.dot(&frame.t());
    let direct_cut: f64 = (0..frame.nrows())
        .flat_map(|i| (0..frame.nrows()).map(move |j| (i, j)))
        .filter(|&(i, j)| inside(i) && !inside(j))
        .map(|(i, j)| projector[[i, j]].powi(2))
        .sum();
    let (cut, cut_error) = exact_parts(&split.cut);
    assert!((direct_cut - cut).abs() <= cut_error);
    assert!(cut + cut_error >= loss / 2.0);
    let mut rng = StdRng::seed_from_u64(42);
    let rank = frame.ncols();
    let kept = split.projector_basis.nrows();
    for _ in 0..50 {
        let draw = Array2::from_shape_simple_fn((kept, rank), || rng.random_range(-1.0..1.0));
        let (_, _, right) = draw.svd(false, true).expect("svd");
        let basis = right.expect("vt").slice(s![..kept, ..]).to_owned();
        assert!(loss_of(&basis) >= loss - loss_error);
    }

    let radius = 2.0;
    let native = split.native_bound.upper_bound().expect("bound") * radius;
    assert!(native <= split.cut_bound * radius * (1.0 + accumulation_growth(8)) || split.cut_bound >= native / radius);
    let level = evaluation_level(&form, &form.reads, radius) + evaluation_level(&form, &split.approximate_reads, radius);
    for _ in 0..200 {
        let direction = Array1::from_shape_simple_fn(4, || rng.random_range(-1.0..1.0));
        let length = rng.random_range(0.0..radius) / norm(direction.view());
        let point = direction * length;
        let exact = form.evaluate(point.view()).expect("F");
        let approximate = form
            .evaluate_with_reads(split.approximate_reads.view(), point.view())
            .expect("F̂");
        let gap = norm((&exact - &approximate).view());
        assert!(gap <= native + level, "‖F − F̂‖ = {gap} above {native}");
        assert!(gap <= split.cut_bound * radius + level);
    }
}

/// Toy 2: two planted modules of a Hadamard frame (read dimensions 3 and 5,
/// 6 and 10 units). The finest components are the modules, and weighted
/// observability of module 1's outputs pulls back to module 1's reads, rank 3.
/// Toy 3 adds the skip `ε r₁ s₂ᵀ`: a linear term is additive under every
/// partition, so the blocks do not move, while observability gains `s₂`
/// (rank 4).
#[test]
fn toys_two_and_three_modules_and_a_linear_cross_edge() {
    let (toy, truth) = hadamard_modules(2951);
    let planted = hadamard_modules_truth(&truth);
    let block = Block::of(&toy);
    let outputs = hadamard().slice(s![.., ..;-1]).to_owned();
    let module_one_outputs = outputs.slice(s![..3, ..]).to_owned();
    for epsilon in [0.0, 1e-6, 1e-3, 1e-1] {
        let skip = cross_edge(epsilon);
        let form = block.normal_form(GaussianActivation::ExactGelu, Some(&skip));
        let live: Vec<usize> = form.sources.iter().map(|sources| truth[sources[0].unit]).collect();
        let blocks = form.additive_blocks(test_governor()).expect("blocks");
        assert_eq!(sorted(blocks.finest.clone()), partition_of(&live), "ε = {epsilon}");
        let observed = WeightedObservability::pull_back(
            test_governor(),
            &[ObservabilityStep {
                letters: form.observability_letters(),
                readouts: vec![module_one_outputs.view()],
            }],
        )
        .expect("pull back");
        let expected = if epsilon == 0.0 { planted.observable_rank } else { planted.observable_rank_with_cross_edge };
        assert_eq!(observed.spectrum().resolved_rank, expected, "ε = {epsilon}");
    }
}

/// Control: a dense random block is one component with every pair certified.
#[test]
fn a_random_dense_block_is_one_component() {
    let block = random_block(5, 16, 8);
    let form = block.normal_form(GaussianActivation::ExactGelu, None);
    let blocks = form.additive_blocks(test_governor()).expect("blocks");
    assert_eq!(blocks.finest.len(), 1);
    assert_eq!(blocks.certified_pairs, 16 * 15 / 2);
    assert!(blocks.unresolved_joins.is_empty());
}

#[test]
fn laplacian_matches_the_dense_form_and_subsets_are_checked() {
    let block = random_block(8, 9, 4);
    let form = block.normal_form(GaussianActivation::Relu, None);
    let blocks = form.additive_blocks(test_governor()).expect("blocks");
    let projector = blocks.frame.dot(&blocks.frame.t());
    let q = Array1::from_shape_fn(9, |unit| (unit as f64).sin());
    let dense = Array1::from_shape_fn(9, |i| {
        projector[[i, i]] * q[i] - (0..9).map(|j| projector[[i, j]].powi(2) * q[j]).sum::<f64>()
    });
    let applied = blocks.laplacian_apply(q.view()).expect("laplacian");
    let tolerance = accumulation_growth(64) * 9.0;
    assert!((&applied - &dense).iter().all(|value| value.abs() <= tolerance));
    assert!(matches!(
        form.optimal_split(test_governor(), &blocks, &[1, 1]),
        Err(ModuleSplitError::InvalidSubset { unit: 1, .. })
    ));
}

/// Toy 4 as a block: the paired copy of a rotation `R` (angles `0.3, 0.3, 1.1` in a hidden
/// basis), `R σ(h) − R σ(−h) = R h`. Every write cancels and the merged linear part is `R`
/// bitwise, so the split refuses: a linear block is additive under every partition. Under
/// a projector contract its exact splits are the invariant subspace pairs of `R`, a
/// continuous family whenever the commutant `{X : X R = R X}` exceeds `d`: the rotation
/// commutant has dimension `2·2² + 2·1² = 10 > 6`, and the fibre oracle on
/// `X ↦ X R − R X` bounds the computed one by exactly that.
#[test]
fn toy4_paired_rotation_is_a_linear_block_with_a_continuous_commutant() {
    let width = 6;
    let toy = rotation_toy();
    let planted = &toy.planted;
    let rotation = &planted.matrix;
    let mut w_in = Array2::<f64>::zeros((2 * width, width));
    w_in.slice_mut(s![..width, ..]).assign(&Array2::<f64>::eye(width));
    w_in.slice_mut(s![width.., ..]).assign(&(-Array2::<f64>::eye(width)));
    let w_out = concatenate![Axis(1), *rotation, -rotation];
    let block = Block {
        w_in,
        b_in: Array1::zeros(2 * width),
        w_out,
        b_out: Array1::zeros(width),
    };
    let form = block.normal_form(GaussianActivation::ExactGelu, None);
    assert_eq!(form.reads.nrows(), 0);
    assert_eq!(&form.linear, rotation);
    assert!(matches!(form.additive_blocks(test_governor()), Err(ModuleSplitError::NoUnits)));

    // `vec(X R − R X) = (Rᵀ ⊗ I − I ⊗ R) vec(X)`, column-major `vec`.
    let order = width * width;
    let commutator = Array2::from_shape_fn((order, order), |(row, col)| {
        let (i, j) = (row % width, row / width);
        let (k, l) = (col % width, col / width);
        let right = if i == k { rotation[[l, j]] } else { 0.0 };
        let left = if j == l { rotation[[i, k]] } else { 0.0 };
        right - left
    });
    // `R` is within `matrix_defect` of an exact rotation, and `‖E ⊗ I‖₂ = ‖I ⊗ E‖₂ = ‖E‖₂`;
    // each entry is one rounded difference.
    let rounding = accumulation_growth(1) * commutator.iter().map(|value| value * value).sum::<f64>().sqrt();
    let fibre = parameter_fibre(test_governor(), &commutator, 2.0 * planted.matrix_defect + rounding).expect("fibre");
    assert_eq!(fibre.nullity_at_most(), toy.commutant_dimension);
    assert!(fibre.nullity_at_most() > width, "a continuous family of splits");
}

/// Toy 7, the random null: a dense random GELU block of 64 units on `ℝ¹⁶` is one
/// component with no unresolved join, so no module is claimed.
#[test]
fn toy7_random_block_claims_no_module() {
    let block = Block::of(&random_mlp(RANDOM_NULL_SEED, 64, 16));
    let form = block.normal_form(GaussianActivation::ExactGelu, None);
    let blocks = form.additive_blocks(test_governor()).expect("blocks");
    assert_eq!(blocks.finest.len(), 1);
    assert!(blocks.unresolved_joins.is_empty());
    assert!(matches!(blocks.rank, EvidenceStatus::Exact { value, .. } if value == 16.0));
}
