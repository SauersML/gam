#![cfg(test)]
//! Planted groups, discovery toys and nulls against the symmetry owner (#2951): the isotypic
//! decomposition of cyclic, dihedral, symmetric and quaternion actions against their known
//! irreps; discovery of a planted cyclic (abelian) and `S_3` (non-abelian, 2-dimensional irreps
//! of multiplicity 2) symmetry from token tables with no declaration, certified on a planted
//! equivariant function; a Gaussian null that must report no exact symmetry; the operator commutant of a
//! planted block family against a generic one, densely and past the dense limit from rank-two
//! factors; and candidate subdomains with planted twins and a planted cyclic cell.

use crate::joint_operators::FactoredOperator;
use crate::symmetry::{
    CommutantRoute, IrrepKind, Permutation, PermutationGroup, TokenFunction, TokenOperators,
    candidate_subdomains, certify_on_function, commutator_norm, discover_permutations,
    generator_code_bits, isotypic_basis, linear_assignment, operator_commutant,
};
use crate::test_support::hidden_basis;
use gam_linalg::faer_ndarray::{FaerEigh, fast_ab, fast_abt, fast_atb};
use faer::Side;
use ndarray::{Array2, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::TAU;

fn shift(n: usize, by: usize) -> Permutation {
    (0..n).map(|x| ((x + by) % n) as u32).collect()
}

fn reflection(n: usize) -> Permutation {
    (0..n).map(|x| ((n - x) % n) as u32).collect()
}

/// `S_3` as permutations of `{0,1,2}`, indexed; the left-regular action on 6 points.
fn s3_elements() -> Vec<[usize; 3]> {
    vec![[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0], [2, 0, 1]]
}

fn s3_index(p: [usize; 3]) -> usize {
    s3_elements().iter().position(|q| *q == p).expect("an element of S_3")
}

fn s3_compose(a: [usize; 3], b: [usize; 3]) -> [usize; 3] {
    [a[b[0]], a[b[1]], a[b[2]]]
}

fn s3_left_regular(g: [usize; 3]) -> Permutation {
    s3_elements().into_iter().map(|h| s3_index(s3_compose(g, h)) as u32).collect()
}

/// The projector onto frequency `k`'s plane of `Z_p` in the token order.
fn fourier_projector(p: usize, k: usize) -> Array2<f64> {
    Array2::from_shape_fn((p, p), |(a, b)| {
        let angle = TAU * (k * ((a + p - b) % p)) as f64 / p as f64;
        2.0 * angle.cos() / p as f64
    })
}

fn block_projector(basis: &crate::symmetry::IsotypicBasis, block: usize) -> Array2<f64> {
    let columns = basis.block_columns(block).to_owned();
    fast_abt(&columns, &columns)
}

fn spectral_distance(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let difference = a - b;
    let (values, _) = difference.eigh(Side::Lower).expect("eigh");
    values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

#[test]
fn schreier_sims_orders_and_membership() {
    let p = 13;
    let cyclic = PermutationGroup::new(p, vec![shift(p, 1)]).expect("group");
    assert_eq!(cyclic.order(), Some(p as u128));
    assert!(cyclic.is_abelian());
    assert!(cyclic.contains(&shift(p, 5)));
    assert!(!cyclic.contains(&reflection(p)));
    let dihedral = PermutationGroup::new(p, vec![shift(p, 3), reflection(p)]).expect("group");
    assert_eq!(dihedral.order(), Some(2 * p as u128));
    assert!(!dihedral.is_abelian());
    assert!(dihedral.contains(&shift(p, 1)));
    let symmetric = PermutationGroup::new(6, vec![vec![1, 0, 2, 3, 4, 5], vec![1, 2, 3, 4, 5, 0]]).expect("group");
    assert_eq!(symmetric.order(), Some(720));
    let regular = PermutationGroup::new(6, vec![s3_left_regular([1, 0, 2]), s3_left_regular([1, 2, 0])]).expect("group");
    assert_eq!(regular.order(), Some(6));
    assert_eq!(regular.elements(36).expect("small").len(), 6);
    // The canonical cycle of a group holding the unit shift is the unit shift.
    assert_eq!(dihedral.regular_cycle(), Some(shift(p, 1)));
    let positions = dihedral.cycle_positions().expect("positions");
    assert_eq!(positions, (0..p as u32).map(Some).collect::<Vec<_>>());
}

#[test]
fn assignment_matches_brute_force() {
    let mut rng = StdRng::seed_from_u64(7);
    for n in 1..=6 {
        let cost = Array2::from_shape_fn((n, n), |_| rng.random_range(0.0..1.0));
        let matching = linear_assignment(cost.view()).expect("assignment");
        let value: f64 = matching.iter().enumerate().map(|(i, &j)| cost[[i, j]]).sum();
        let mut best = f64::INFINITY;
        let mut perm: Vec<usize> = (0..n).collect();
        permutations(&mut perm, 0, &mut |q| {
            best = best.min(q.iter().enumerate().map(|(i, &j)| cost[[i, j]]).sum());
        });
        assert!((value - best).abs() <= 1e-12 * n as f64, "n = {n}: {value} vs {best}");
    }
}

fn permutations(items: &mut Vec<usize>, k: usize, visit: &mut dyn FnMut(&[usize])) {
    if k == items.len() {
        visit(items);
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permutations(items, k + 1, visit);
        items.swap(k, i);
    }
}

/// `Z_p` on `p` points: the trivial line and `(p−1)/2` complex-type planes whose projectors are
/// the Fourier projectors and whose characters at the unit shift are `2 cos(2πk/p)`.
#[test]
fn cyclic_isotypic_blocks_are_the_fourier_planes() {
    let p = 11;
    let group = PermutationGroup::new(p, vec![shift(p, 1)]).expect("group");
    let basis = isotypic_basis(&group).expect("decomposition");
    assert_eq!(basis.blocks.len(), 1 + (p - 1) / 2, "{:?}", basis.blocks);
    let orthogonality = fast_atb(&basis.basis, &basis.basis) - Array2::<f64>::eye(p);
    assert!(orthogonality.iter().all(|v| v.abs() < 1e-12));
    let trivial = &basis.blocks[0];
    assert_eq!((trivial.irrep_dim, trivial.multiplicity, trivial.kind), (1, 1, IrrepKind::Real));
    for (index, block) in basis.blocks.iter().enumerate().skip(1) {
        assert_eq!((block.irrep_dim, block.multiplicity, block.kind), (2, 1, IrrepKind::Complex));
        let cosine = block.character[0] / 2.0;
        let k = (1..=(p - 1) / 2)
            .min_by(|&a, &b| {
                let da = ((TAU * a as f64 / p as f64).cos() - cosine).abs();
                let db = ((TAU * b as f64 / p as f64).cos() - cosine).abs();
                da.total_cmp(&db)
            })
            .expect("a frequency");
        assert!(((TAU * k as f64 / p as f64).cos() - cosine).abs() < 1e-10);
        let distance = spectral_distance(&block_projector(&basis, index), &fourier_projector(p, k));
        assert!(distance <= block.projector_error + 1e-10, "plane {k}: {distance}");
    }
    let bits = generator_code_bits(&group).expect("bits");
    assert!(bits > 0);
}

/// `D_p`: the same planes, now of real type.
#[test]
fn dihedral_planes_are_real_type() {
    let p = 9;
    let group = PermutationGroup::new(p, vec![shift(p, 1), reflection(p)]).expect("group");
    let basis = isotypic_basis(&group).expect("decomposition");
    assert_eq!(basis.blocks.len(), 1 + (p - 1) / 2);
    for block in basis.blocks.iter().skip(1) {
        assert_eq!((block.irrep_dim, block.multiplicity, block.kind), (2, 1, IrrepKind::Real));
        assert!(block.character[1].abs() < 1e-10, "a reflection has trace 0 on a plane");
    }
}

/// The regular action of `S_3`: trivial, sign, and the 2-dimensional standard irrep with
/// multiplicity 2, whose character on a transposition is 0 and on a 3-cycle is −1.
#[test]
fn s3_regular_has_the_standard_irrep_twice() {
    let group = PermutationGroup::new(6, vec![s3_left_regular([1, 0, 2]), s3_left_regular([1, 2, 0])]).expect("group");
    let basis = isotypic_basis(&group).expect("decomposition");
    let shapes: Vec<(usize, usize, IrrepKind)> = basis.blocks.iter().map(|b| (b.irrep_dim, b.multiplicity, b.kind)).collect();
    assert_eq!(shapes, vec![(1, 1, IrrepKind::Real), (1, 1, IrrepKind::Real), (2, 2, IrrepKind::Real)]);
    assert!(basis.blocks[0].character.iter().all(|v| (v - 1.0).abs() < 1e-10), "{:?}", basis.blocks[0].character);
    let standard = &basis.blocks[2];
    assert!(standard.character[0].abs() < 1e-10 && (standard.character[1] + 1.0).abs() < 1e-10, "{:?}", standard.character);
    let sign = &basis.blocks[1];
    assert!((sign.character[0] + 1.0).abs() < 1e-10 && (sign.character[1] - 1.0).abs() < 1e-10);
}

/// The quaternion group `Q_8` acting regularly: four real characters and one 4-dimensional
/// irreducible of quaternionic type.
#[test]
fn quaternion_group_has_a_quaternionic_irrep() {
    // Elements ±1, ±i, ±j, ±k as (sign, unit) with unit 0..4 = 1, i, j, k.
    let table = [[0usize, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]];
    let signs = [[1i32, 1, 1, 1], [1, -1, 1, -1], [1, -1, -1, 1], [1, 1, -1, -1]];
    let index = |sign: i32, unit: usize| unit + if sign < 0 { 4 } else { 0 };
    let multiply = |a: usize, b: usize| {
        let (sa, ua) = (if a >= 4 { -1 } else { 1 }, a % 4);
        let (sb, ub) = (if b >= 4 { -1 } else { 1 }, b % 4);
        index(sa * sb * signs[ua][ub], table[ua][ub])
    };
    let left = |g: usize| -> Permutation { (0..8).map(|h| multiply(g, h) as u32).collect() };
    let group = PermutationGroup::new(8, vec![left(1), left(2)]).expect("group");
    assert_eq!(group.order(), Some(8));
    let basis = isotypic_basis(&group).expect("decomposition");
    let shapes: Vec<(usize, usize, IrrepKind)> = basis.blocks.iter().map(|b| (b.irrep_dim, b.multiplicity, b.kind)).collect();
    assert_eq!(shapes.iter().filter(|s| **s == (1, 1, IrrepKind::Real)).count(), 4, "{shapes:?}");
    assert!(shapes.contains(&(4, 1, IrrepKind::Quaternionic)), "{shapes:?}");
}

/// A token table whose Gram is `Σ_k c_k² Π_k` for a planted relabelling of `Z_p`, rotated by a
/// hidden orthogonal map, so its symmetry is exact and nothing names it.
fn planted_cyclic_table(p: usize, labelling: &[usize], d: usize, seed: u64) -> Array2<f64> {
    let mut features = Array2::<f64>::zeros((p, p));
    for t in 0..p {
        let a = labelling[t];
        features[[t, 0]] = 0.7;
        for k in 1..=(p - 1) / 2 {
            let angle = TAU * (k * a) as f64 / p as f64;
            features[[t, 2 * k - 1]] = angle.cos();
            features[[t, 2 * k]] = angle.sin();
        }
    }
    let weights: Vec<f64> = (0..p).map(|c| if c == 0 { 1.0 } else { 3.0 / (1.0 + ((c + 1) / 2) as f64) }).collect();
    for (c, w) in weights.iter().enumerate() {
        features.column_mut(c).mapv_inplace(|v| v * w);
    }
    let hidden = hidden_basis(d, seed ^ 0xABCD);
    fast_ab(&features, &hidden.slice(s![..p, ..]).to_owned())
}

/// The planted cyclic table under a scrambled labelling: the search finds a group of order `2p`
/// (shifts and reflections of the hidden labelling), exact within band, with the orbit of every
/// base point certified by its gap, and the isotypic planes are the planted labelling's Fourier
/// planes.
#[test]
fn discovers_a_scrambled_cyclic_symmetry() {
    let p = 13;
    let labelling: Vec<usize> = (0..p).map(|t| (5 * t + 3) % p).collect();
    let table = planted_cyclic_table(p, &labelling, 16, 11);
    let ops = TokenOperators::from_tables(&[table.view()]).expect("operators");
    let discovery = discover_permutations(&ops).expect("discovery");
    assert!(discovery.exact(), "{:?} vs band {}", discovery.generator_defects, discovery.exact_band);
    assert_eq!(discovery.group.order(), Some(2 * p as u128), "{:?}", discovery.levels);
    // The planted shift, written in token ids.
    let mut token_of = vec![0usize; p];
    for (t, &a) in labelling.iter().enumerate() {
        token_of[a] = t;
    }
    let planted: Permutation = (0..p).map(|t| token_of[(labelling[t] + 1) % p] as u32).collect();
    assert!(discovery.group.contains(&planted));
    let basis = isotypic_basis(&discovery.group).expect("decomposition");
    for (index, block) in basis.blocks.iter().enumerate().skip(1) {
        let cosine = block.character.iter().copied().fold(f64::NEG_INFINITY, f64::max) / 2.0;
        let projector = block_projector(&basis, index);
        let matched = (1..=(p - 1) / 2).any(|k| {
            let planted_projector = Array2::from_shape_fn((p, p), |(x, y)| fourier_projector(p, k)[[labelling[x], labelling[y]]]);
            spectral_distance(&projector, &planted_projector) <= block.projector_error + 1e-9
        });
        assert!(matched, "block {index} (cos {cosine}) is not a planted plane");
    }
}

/// The left-regular `S_3` action planted in a Gram `H = f(x⁻¹y)` with a generic symmetric `f`
/// that is not a class function, and an equivariant function on token pairs. Discovery finds
/// exactly `S_3` (non-abelian, the standard irrep twice), the function certifies it exactly with
/// the output permutation discovered, and a transposition of two tokens is refused.
#[test]
fn discovers_a_non_abelian_symmetry_and_certifies_it_on_the_function() {
    let elements = s3_elements();
    let inverse_index: Vec<usize> =
        elements.iter().map(|g| (0..6).find(|&h| s3_compose(*g, elements[h]) == [0, 1, 2]).expect("inverse")).collect();
    let mut rng = StdRng::seed_from_u64(3);
    let mut f = vec![0.0; 6];
    for g in 0..6 {
        if f[g] == 0.0 {
            let value = rng.random_range(0.2..1.0);
            f[g] = value;
            f[inverse_index[g]] = value;
        }
    }
    f[0] = 3.0;
    let gram = Array2::from_shape_fn((6, 6), |(x, y)| f[s3_index(s3_compose(elements[inverse_index[x]], elements[y]))]);
    let ops = TokenOperators::new(vec![gram.clone()], vec![0.0]).expect("operators");
    let discovery = discover_permutations(&ops).expect("discovery");
    assert_eq!(discovery.group.order(), Some(6), "{:?}", discovery.levels);
    assert!(!discovery.group.is_abelian());
    assert!(discovery.exact());
    let basis = isotypic_basis(&discovery.group).expect("decomposition");
    assert!(basis.blocks.iter().any(|b| (b.irrep_dim, b.multiplicity) == (2, 2)), "{:?}", basis.blocks);
    // Schur: the invariant Gram is block-diagonal in the forced basis.
    assert!(basis.off_block_fraction(gram.view()).expect("fraction") < 1e-20);
    // An equivariant function on pairs: ℓ(x, y)[c] = H[x, c] + 2 H[y, c]² (entrywise).
    let mut tokens = Array2::<u32>::zeros((36, 2));
    let mut logits = Array2::<f64>::zeros((36, 6));
    for x in 0..6 {
        for y in 0..6 {
            let r = 6 * x + y;
            tokens[[r, 0]] = x as u32;
            tokens[[r, 1]] = y as u32;
            for c in 0..6 {
                logits[[r, c]] = gram[[x, c]] + 2.0 * gram[[y, c]].powi(2);
            }
        }
    }
    let function = TokenFunction { tokens: tokens.view(), acted: &[true, true], logits: logits.view(), radii: None };
    for g in discovery.group.generators() {
        let certificate = certify_on_function(g, &function).expect("certificate");
        assert!(certificate.exact(), "{certificate:?}");
        assert_eq!(certificate.output_permutation, *g);
        assert_eq!(certificate.argmax_agreement, 36);
    }
    let transposition: Permutation = vec![1, 0, 2, 3, 4, 5];
    let refused = certify_on_function(&transposition, &function).expect("certificate");
    assert!(!refused.exact() && refused.kl_bits > 0.0);
}

/// A Gaussian table has no exact symmetry: whatever the gap-certified search proposes on it is
/// far outside the exactness band (a lone approximate involution at most), unlike the planted
/// tables above.
#[test]
fn gaussian_null_reports_no_exact_symmetry() {
    for seed in 0..4 {
        let mut rng = StdRng::seed_from_u64(100 + seed);
        let table = Array2::from_shape_fn((15, 10), |_| rng.random_range(-1.0..1.0));
        let ops = TokenOperators::from_tables(&[table.view()]).expect("operators");
        let discovery = discover_permutations(&ops).expect("discovery");
        let order = discovery.group.order().expect("small");
        assert!(order <= 2, "seed {seed}: {:?}", discovery.levels);
        if order > 1 {
            assert!(!discovery.exact());
            assert!(discovery.generator_defects.iter().all(|&d| d > 1e6 * discovery.exact_band), "seed {seed}: {:?}", discovery.generator_defects);
        }
    }
}

/// Dense operators as factors `A Iᵀ`.
fn dense_members(members: &[Array2<f64>]) -> Vec<FactoredOperator> {
    members.iter().map(|a| FactoredOperator::new(a.clone(), Array2::eye(a.nrows())).expect("factors")).collect()
}

/// A planted block family `Q (X_i ⊗ I_2 ⊕ Y_i) Qᵀ`: the commutant holds `I ⊗ M_2 ⊕ I`, dimension
/// 5, and the isotypic blocks are a 3-dimensional irrep of multiplicity 2 and a 2-dimensional
/// one; a generic family commutes with the scalars only.
#[test]
fn operator_commutant_of_a_planted_block_family() {
    let mut rng = StdRng::seed_from_u64(21);
    let n = 8;
    let q = hidden_basis(n, 5);
    let mut members = Vec::new();
    for _ in 0..3 {
        let x = Array2::from_shape_fn((3, 3), |_| rng.random_range(-1.0..1.0));
        let y = Array2::from_shape_fn((2, 2), |_| rng.random_range(-1.0..1.0));
        let mut block = Array2::<f64>::zeros((n, n));
        for a in 0..3 {
            for b in 0..3 {
                block[[a, b]] = x[[a, b]];
                block[[3 + a, 3 + b]] = x[[a, b]];
            }
        }
        block.slice_mut(s![6..8, 6..8]).assign(&y);
        members.push(fast_abt(&fast_ab(&q, &block), &q));
    }
    let factored = dense_members(&members);
    let commutant = operator_commutant(&factored.iter().collect::<Vec<_>>(), 3).expect("commutant");
    assert_eq!(commutant.exact_dim, 5, "{commutant:?}");
    for t in &commutant.basis {
        for a in &members {
            assert!(commutator_norm(t.view(), a.view()) < 1e-9);
        }
    }
    let shapes: Vec<(usize, usize)> = commutant.isotypic.blocks.iter().map(|b| (b.irrep_dim, b.multiplicity)).collect();
    assert_eq!(shapes, vec![(2, 1), (3, 2)]);
    for a in &members {
        assert!(commutant.isotypic.off_block_fraction(a.view()).expect("fraction") < 1e-20);
    }
    let generic: Vec<Array2<f64>> = (0..2).map(|_| Array2::from_shape_fn((n, n), |_| rng.random_range(-1.0..1.0))).collect();
    let factored = dense_members(&generic);
    let scalars = operator_commutant(&factored.iter().collect::<Vec<_>>(), 3).expect("commutant");
    assert_eq!(scalars.exact_dim, 1);
    assert!(scalars.blocks[0].approximate_defects[0] > 1e-3);
}

/// Rank-two members supported on a hidden split `ℝ^60 ⊕ ℝ^36` of `ℝ^96`, past the dense limit:
/// the algebra route finds the two blocks, each with the scalars as commutant (the 60-block
/// certified from its simple eigenvalues, the 36-block read as a dense kernel), holds every
/// member block-diagonal, and keeps the cross-block coupling inside its bound. The same members
/// with the split mixed by one full-rank member give one block and the scalars.
#[test]
fn operator_commutant_splits_a_planted_family_past_the_dense_limit() {
    let mut rng = StdRng::seed_from_u64(31);
    let n = 96;
    let q = hidden_basis(n, 9);
    let mut members = Vec::new();
    for (range, count) in [(0..60, 20), (60..96, 12)] {
        for _ in 0..count {
            let mut left = Array2::<f64>::zeros((n, 2));
            let mut right = Array2::<f64>::zeros((n, 2));
            for row in range.clone() {
                for c in 0..2 {
                    left[[row, c]] = rng.random_range(-1.0..1.0);
                    right[[row, c]] = rng.random_range(-1.0..1.0);
                }
            }
            members.push(FactoredOperator::new(fast_ab(&q, &left), fast_ab(&q, &right)).expect("factors"));
        }
    }
    let commutant = operator_commutant(&members.iter().collect::<Vec<_>>(), 0).expect("commutant");
    let shapes: Vec<(usize, usize, CommutantRoute)> = commutant.blocks.iter().map(|b| (b.dim, b.commutant_dim, b.route)).collect();
    assert_eq!(shapes, vec![(36, 1, CommutantRoute::Dense), (60, 1, CommutantRoute::Simple)]);
    assert_eq!(commutant.exact_dim, 2);
    assert!(commutant.split_coupling <= commutant.split_bound, "{} > {}", commutant.split_coupling, commutant.split_bound);
    for member in &members {
        let dense = fast_abt(&member.left().to_owned(), &member.right().to_owned());
        assert!(commutant.isotypic.off_block_fraction(dense.view()).expect("fraction") < 1e-20);
    }
    for t in &commutant.basis {
        for member in &members {
            let dense = fast_abt(&member.left().to_owned(), &member.right().to_owned());
            assert!(commutator_norm(t.view(), dense.view()) < 1e-9);
        }
    }
    let mixing = FactoredOperator::new(Array2::from_shape_fn((n, n), |_| rng.random_range(-1.0..1.0)), Array2::eye(n)).expect("factors");
    let mut mixed: Vec<&FactoredOperator> = members.iter().collect();
    mixed.push(&mixing);
    let scalars = operator_commutant(&mixed, 0).expect("commutant");
    assert_eq!(scalars.exact_dim, 1);
    assert_eq!(scalars.blocks.len(), 1);
    assert!(scalars.blocks[0].bottleneck_coupling.expect("connected") > 0.0);
}

/// A Gaussian table with two planted duplicate classes: the refinement isolates them, each class
/// is a twin class, and every other point is a singleton; a Gaussian table has no cell at all.
#[test]
fn candidate_subdomains_find_planted_twins() {
    let mut rng = StdRng::seed_from_u64(41);
    let mut table = Array2::from_shape_fn((40, 8), |_| rng.random_range(-1.0..1.0));
    for copy in [5, 17, 30] {
        let row = table.row(2).to_owned();
        table.row_mut(copy).assign(&row);
    }
    let row = table.row(11).to_owned();
    table.row_mut(12).assign(&row);
    let subdomains = candidate_subdomains(&[table.view()]).expect("subdomains");
    assert_eq!(subdomains.twins, vec![vec![2, 5, 17, 30], vec![11, 12]]);
    assert_eq!(subdomains.cells, subdomains.twins);
    assert!(subdomains.open_cells().is_empty());
    assert_eq!(subdomains.quotient_size(), 36);
    assert!((subdomains.log2_twin_order() - 48f64.log2()).abs() < 1e-12);
    let null = Array2::from_shape_fn((40, 8), |_| rng.random_range(-1.0..1.0));
    let none = candidate_subdomains(&[null.view()]).expect("subdomains");
    assert!(none.cells.is_empty() && none.twins.is_empty());
}

/// The planted cyclic table: its tokens share every invariant, so they form one open cell with
/// no twins, which is where the permutation search acts.
#[test]
fn candidate_subdomains_keep_a_planted_orbit_open() {
    let p = 13;
    let labelling: Vec<usize> = (0..p).map(|t| (5 * t + 3) % p).collect();
    let table = planted_cyclic_table(p, &labelling, 16, 11);
    let subdomains = candidate_subdomains(&[table.view()]).expect("subdomains");
    assert_eq!(subdomains.cells, vec![(0..p).collect::<Vec<_>>()]);
    assert!(subdomains.twins.is_empty());
    assert_eq!(subdomains.open_cells().len(), 1);
}

/// A cycle drawn on an ellipse with an offset (`c + u cos θ_a + v sin θ_a`, `|u| ≠ |v|`,
/// `u·v ≠ 0`, scrambled labelling): its Gram has no cyclic symmetry, the projector onto its
/// resolved rank-2 leading subspace has the dihedral one, exactly within band, and the canonical cycle's positions are
/// an affine relabelling of the planted `a ↦ a + 1`.
#[test]
fn the_leading_subspace_whitens_an_elliptic_cycle() {
    let p = 9;
    let labelling: Vec<usize> = (0..p).map(|t| (4 * t + 2) % p).collect();
    let (c, u, v) = ([0.25, -0.5, 0.125, 0.75], [1.0, 0.5, -0.75, 0.25], [-0.25, 1.0, 0.5, -0.5]);
    let table = Array2::from_shape_fn((p, 4), |(t, i)| {
        let angle = TAU * labelling[t] as f64 / p as f64;
        c[i] + u[i] * angle.cos() + v[i] * angle.sin()
    });
    let gram = discover_permutations(&TokenOperators::from_tables(&[table.view()]).expect("operators")).expect("discovery");
    assert!(gram.group.order().expect("small") < 2 * p as u128 || !gram.exact(), "{:?}", gram.levels);
    let subspaces = TokenOperators::leading_subspaces(table.view(), 0.0).expect("subspaces");
    let ranks: Vec<usize> = subspaces.iter().map(|(rank, _)| *rank).collect();
    assert_eq!(ranks, vec![1, 2], "the plane and its leading line are resolved; rank 3 is inside the zero cluster");
    let ops = &subspaces[1].1;
    let discovery = discover_permutations(ops).expect("discovery");
    assert_eq!(discovery.group.order(), Some(2 * p as u128), "{:?}", discovery.levels);
    assert!(discovery.exact(), "{:?} vs {}", discovery.generator_defects, discovery.exact_band);
    let positions = discovery.group.cycle_positions().expect("an odd cycle");
    let position = |t: usize| positions[t].expect("on the cycle") as usize;
    let affine = (1..p).any(|k| (0..p).all(|t| position(t) == (k * (labelling[t] + p - labelling[0])) % p));
    assert!(affine, "{positions:?}");
}
