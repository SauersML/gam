//! Tests for the state-block decomposition (#2946, the proposal's §9.2).
//!
//! Each test plants a family whose isotypic structure is known in closed form,
//! then asserts the reported components, block dimensions, multiplicities and
//! division algebras. It checks the commutant dimension against an independent
//! dense oracle: the nullity of `X ↦ ([X, A_a], [X, A_aᵀ])_a` on all `d²`
//! unknowns, read by the owned factor rank predicate. Every test also carries a
//! control whose structure differs, and which its assertions tell apart.

use super::{BlockBasis, DivisionAlgebra, StateBlocks};
use gam_linalg::roundoff::factor_rank_partition;
use ndarray::{Array1, Array2, Axis, array, s};

fn rotation(angle: f64) -> Array2<f64> {
    let (sine, cosine) = angle.sin_cos();
    array![[cosine, -sine], [sine, cosine]]
}

fn direct_sum(blocks: &[Array2<f64>]) -> Array2<f64> {
    let dimension = blocks.iter().map(|block| block.nrows()).sum();
    let mut sum = Array2::<f64>::zeros((dimension, dimension));
    let mut offset = 0;
    for block in blocks {
        let size = block.nrows();
        sum.slice_mut(s![offset..offset + size, offset..offset + size])
            .assign(block);
        offset += size;
    }
    sum
}

/// A fixed orthogonal frame: Givens rotations over every coordinate pair at
/// distinct angles, so no planted block lies along a coordinate axis.
fn planted_frame(dimension: usize) -> Array2<f64> {
    let mut frame = Array2::<f64>::eye(dimension);
    let mut angle = 0.37_f64;
    for first in 0..dimension {
        for second in (first + 1)..dimension {
            let (sine, cosine) = angle.sin_cos();
            let mut givens = Array2::<f64>::eye(dimension);
            givens[[first, first]] = cosine;
            givens[[second, second]] = cosine;
            givens[[first, second]] = -sine;
            givens[[second, first]] = sine;
            frame = frame.dot(&givens);
            angle += 0.61;
        }
    }
    frame
}

fn conjugate(frame: &Array2<f64>, map: &Array2<f64>) -> Array2<f64> {
    frame.dot(map).dot(&frame.t())
}

/// Nullity of the commutator map on all `d²` unknowns, from the rank of its
/// dense factor: a `d² × d²` oracle that is affordable only at test widths.
fn dense_commutant_dimension(maps: &[Array2<f64>]) -> usize {
    let dimension = maps[0].nrows();
    let unknowns = dimension * dimension;
    let mut factor = Array2::<f64>::zeros((2 * maps.len() * unknowns, unknowns));
    for unknown in 0..unknowns {
        let mut unit = Array2::<f64>::zeros((dimension, dimension));
        unit[[unknown / dimension, unknown % dimension]] = 1.0;
        for (index, map) in maps.iter().enumerate() {
            let forward = unit.dot(map) - map.dot(&unit);
            let transposed = unit.dot(&map.t()) - map.t().dot(&unit);
            let offset = 2 * index * unknowns;
            factor
                .slice_mut(s![offset..offset + unknowns, unknown])
                .assign(&Array1::from_iter(forward.iter().copied()));
            factor
                .slice_mut(s![offset + unknowns..offset + 2 * unknowns, unknown])
                .assign(&Array1::from_iter(transposed.iter().copied()));
        }
    }
    let partition = factor_rank_partition(&factor).expect("dense commutator factor SVD");
    unknowns - partition.rank
}

fn projector_distance(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    (left.dot(&left.t()) - right.dot(&right.t()))
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

/// A plane rotation by `δ ∉ {0, π}` leaves no line invariant: one irreducible
/// two-dimensional block with complex intertwiners. Control: at `δ = π` the map
/// is `−I`, every line is invariant, and the plane is two undetermined lines.
#[test]
fn plane_rotation_is_one_irreducible_plane_with_no_invariant_line() {
    let turn = rotation(0.9);
    let blocks = StateBlocks::decompose(vec![turn.clone()]).expect("rotation decomposes");
    assert_eq!(blocks.components.len(), 1);
    let plane = &blocks.components[0];
    assert_eq!(plane.dimension(), 2);
    assert_eq!(
        plane.block_dimension, 2,
        "a rotation by 0.9 must leave no line invariant"
    );
    assert_eq!(plane.multiplicity, 1);
    assert_eq!(plane.division_algebra, DivisionAlgebra::Complex);
    assert_eq!(plane.block_basis(), BlockBasis::Determined);
    assert_eq!(blocks.commutant_dimension(), 2);
    assert_eq!(dense_commutant_dimension(&[turn]), 2);

    // The half turn is written exactly: `rotation(π)` carries `sin π ≈ 1.2e-16`,
    // which the decomposition leaves unresolved at the map's scale, while the
    // oracle's factor predicate is relative to the commutator factor's own
    // largest singular value (that same `1e-16`) and would resolve it.
    let half_turn = -Array2::<f64>::eye(2);
    let control = StateBlocks::decompose(vec![half_turn.clone()]).expect("half turn decomposes");
    assert_eq!(control.components.len(), 1);
    let lines = &control.components[0];
    assert_eq!(lines.block_dimension, 1, "−I leaves every line invariant");
    assert_eq!(lines.multiplicity, 2);
    assert_eq!(lines.division_algebra, DivisionAlgebra::Real);
    assert_eq!(control.commutant_dimension(), 4);
    assert_eq!(dense_commutant_dimension(&[half_turn]), 4);
}

/// The identity leaves every subspace invariant: one isotypic component of
/// `d` equivalent lines whose block basis is undetermined, and no basis of lines
/// may be reported. Control: a scaling with distinct factors determines every
/// coordinate line.
#[test]
fn identity_is_one_isotypic_component_with_an_undetermined_block_basis() {
    let dimension = 4;
    let identity = Array2::<f64>::eye(dimension);
    let blocks = StateBlocks::decompose(vec![identity.clone()]).expect("identity decomposes");
    assert_eq!(
        blocks.components.len(),
        1,
        "the identity must not be reported as a chosen basis of lines"
    );
    let component = &blocks.components[0];
    assert_eq!(component.dimension(), dimension);
    assert_eq!(component.block_dimension, 1);
    assert_eq!(component.multiplicity, dimension);
    assert_eq!(component.division_algebra, DivisionAlgebra::Real);
    assert_eq!(
        component.block_basis(),
        BlockBasis::Undetermined {
            multiplicity: 4,
            ambiguity_dimension: 6,
        }
    );
    let span = factor_rank_partition(&component.basis).expect("component basis SVD");
    assert_eq!(span.rank, dimension, "the component is the whole state");
    assert_eq!(blocks.commutant_dimension(), 16);
    assert_eq!(dense_commutant_dimension(&[identity]), 16);

    let scaling = Array2::from_diag(&array![0.5, 1.5, 2.5, 3.5]);
    let control = StateBlocks::decompose(vec![scaling.clone()]).expect("scaling decomposes");
    assert_eq!(control.components.len(), dimension);
    assert!(
        control
            .components
            .iter()
            .all(|line| line.dimension() == 1 && line.block_basis() == BlockBasis::Determined),
        "distinct scale factors must determine every line"
    );
    assert_eq!(control.commutant_dimension(), dimension);
    assert_eq!(dense_commutant_dimension(&[scaling]), dimension);
}

/// A rotation plane and a scaling line planted in a generic frame are recovered
/// as two determined components. Their spans match the planted ones within the
/// reported angle bound, and a plane that mixes the two does not.
#[test]
fn planted_direct_sum_of_a_rotation_and_a_scaling_is_recovered() {
    let frame = planted_frame(3);
    let family = vec![
        conjugate(&frame, &direct_sum(&[rotation(0.7), array![[1.6]]])),
        conjugate(&frame, &direct_sum(&[rotation(-2.1), array![[0.4]]])),
    ];
    let blocks = StateBlocks::decompose(family.clone()).expect("planted family decomposes");
    assert_eq!(blocks.components.len(), 2);
    let plane = blocks
        .components
        .iter()
        .find(|component| component.dimension() == 2)
        .expect("a planar component");
    let line = blocks
        .components
        .iter()
        .find(|component| component.dimension() == 1)
        .expect("a line component");
    assert_eq!(
        (plane.block_dimension, plane.multiplicity, plane.division_algebra),
        (2, 1, DivisionAlgebra::Complex)
    );
    assert_eq!(
        (line.block_dimension, line.multiplicity, line.division_algebra),
        (1, 1, DivisionAlgebra::Real)
    );
    // `‖P − P*‖_F ≤ √(2k)·sin θ_max` for `k`-dimensional spans.
    let plane_bound = 2.0 * plane.angle_bound;
    let line_bound = 2.0_f64.sqrt() * line.angle_bound;
    let planted_plane = frame.slice(s![.., 0..2]).to_owned();
    let planted_line = frame.slice(s![.., 2..3]).to_owned();
    let plane_distance = projector_distance(&plane.basis, &planted_plane);
    let line_distance = projector_distance(&line.basis, &planted_line);
    assert!(
        plane_distance <= plane_bound,
        "plane projector distance {plane_distance:e} exceeds its bound {plane_bound:e}"
    );
    assert!(
        line_distance <= line_bound,
        "line projector distance {line_distance:e} exceeds its bound {line_bound:e}"
    );
    let mixed_plane = ndarray::concatenate(
        Axis(1),
        &[frame.slice(s![.., 0..1]), frame.slice(s![.., 2..3])],
    )
    .expect("mixed plane columns");
    assert!(
        projector_distance(&plane.basis, &mixed_plane) > plane_bound,
        "a plane mixing the planted line in must fall outside the bound"
    );
    assert_eq!(blocks.commutant_dimension(), 3);
    assert_eq!(dense_commutant_dimension(&family), 3);
}

/// Two equal rotation planes form one isotypic component with two equivalent
/// complex blocks: the component is recovered and its block basis is
/// undetermined, with a two-dimensional manifold of decompositions (`ℂP¹`).
/// Control: unequal angles in the two planes make the blocks inequivalent, so
/// three determined components appear.
#[test]
fn repeated_equivalent_rotation_blocks_report_their_mixing_ambiguity() {
    let frame = planted_frame(5);
    let family = vec![
        conjugate(
            &frame,
            &direct_sum(&[rotation(0.7), rotation(0.7), array![[1.6]]]),
        ),
        conjugate(
            &frame,
            &direct_sum(&[rotation(2.3), rotation(2.3), array![[0.4]]]),
        ),
    ];
    let blocks = StateBlocks::decompose(family.clone()).expect("repeated family decomposes");
    assert_eq!(blocks.components.len(), 2);
    let pair = blocks
        .components
        .iter()
        .find(|component| component.dimension() == 4)
        .expect("the four-dimensional isotypic component");
    assert_eq!(
        (pair.block_dimension, pair.multiplicity, pair.division_algebra),
        (2, 2, DivisionAlgebra::Complex)
    );
    assert_eq!(
        pair.block_basis(),
        BlockBasis::Undetermined {
            multiplicity: 2,
            ambiguity_dimension: 2,
        }
    );
    let pair_bound = 8.0_f64.sqrt() * pair.angle_bound;
    let pair_distance = projector_distance(&pair.basis, &frame.slice(s![.., 0..4]).to_owned());
    assert!(
        pair_distance <= pair_bound,
        "component projector distance {pair_distance:e} exceeds its bound {pair_bound:e}"
    );
    assert_eq!(blocks.commutant_dimension(), 9);
    assert_eq!(dense_commutant_dimension(&family), 9);

    let control_family = vec![conjugate(
        &frame,
        &direct_sum(&[rotation(0.7), rotation(1.9), array![[1.6]]]),
    )];
    let control =
        StateBlocks::decompose(control_family.clone()).expect("control family decomposes");
    assert_eq!(control.components.len(), 3);
    assert!(
        control
            .components
            .iter()
            .all(|component| component.block_basis() == BlockBasis::Determined),
        "inequivalent planes must leave nothing to mix"
    );
    assert_eq!(control.commutant_dimension(), 5);
    assert_eq!(dense_commutant_dimension(&control_family), 5);
}

/// A shear `λI + N` is not normal, and its `*`-algebra is all of `M₂(ℝ)`: one
/// irreducible real plane, found by joining two one-dimensional clusters through
/// a nonzero block. Control: without `N` the plane is `λI`, two undetermined
/// lines.
#[test]
fn a_planted_shear_is_one_irreducible_real_plane() {
    let frame = planted_frame(3);
    let shear = conjugate(
        &frame,
        &direct_sum(&[array![[0.7, 1.0], [0.0, 0.7]], array![[2.0]]]),
    );
    let blocks = StateBlocks::decompose(vec![shear.clone()]).expect("shear decomposes");
    assert_eq!(blocks.components.len(), 2);
    let plane = blocks
        .components
        .iter()
        .find(|component| component.dimension() == 2)
        .expect("a planar component");
    assert_eq!(
        (plane.block_dimension, plane.multiplicity, plane.division_algebra),
        (2, 1, DivisionAlgebra::Real)
    );
    let plane_bound = 2.0 * plane.angle_bound;
    let plane_distance = projector_distance(&plane.basis, &frame.slice(s![.., 0..2]).to_owned());
    assert!(
        plane_distance <= plane_bound,
        "shear plane projector distance {plane_distance:e} exceeds its bound {plane_bound:e}"
    );
    assert_eq!(blocks.commutant_dimension(), 2);
    assert_eq!(dense_commutant_dimension(&[shear]), 2);

    let scalar_plane = conjugate(
        &frame,
        &direct_sum(&[array![[0.7, 0.0], [0.0, 0.7]], array![[2.0]]]),
    );
    let control = StateBlocks::decompose(vec![scalar_plane.clone()]).expect("control decomposes");
    assert_eq!(control.components.len(), 2);
    let lines = control
        .components
        .iter()
        .find(|component| component.dimension() == 2)
        .expect("the scalar plane component");
    assert_eq!((lines.block_dimension, lines.multiplicity), (1, 2));
    assert_eq!(
        lines.block_basis(),
        BlockBasis::Undetermined {
            multiplicity: 2,
            ambiguity_dimension: 1,
        }
    );
    assert_eq!(control.commutant_dimension(), 5);
    assert_eq!(dense_commutant_dimension(&[scalar_plane]), 5);
}
