//! #2023 — the co-collapse reseed draws from the residual's data rows or its graph
//! harmonics, never from its principal components.
//!
//! #2023's architecture rule is "dead-atom resampling draws from high-residual data
//! rows (k-SVD replacement rule), never from PCs". The reseed used to fall through
//! to the principal-component seed for every set a data-row lever did not cover, and
//! that lever was hard-coded off at every production entry. The reseed now reads the
//! graph-harmonic seed for circle, torus and sphere sets and the worst-reconstructed
//! rows for everything else. These tests pin the coordinates each arm must produce,
//! on residuals where a principal-component seed produces different ones.

use super::tests::{small_two_atom_periodic_term, trivial_k1_euclidean_term};
use super::*;

/// A flat atom resamples from the WORST-RECONSTRUCTED data rows at every retry, and
/// the retry index walks down that ranking.
///
/// The fixture's decoder is zero, so the reconstruction residual is `-target`, whose
/// row energies are 0.25, 0.64, 4.0 and 1.44: the order is rows [2, 3, 1, 0]. The
/// rows are axis-aligned, so an anchor row's projection is non-zero only on its own
/// axis and the anchor lands at exactly `+0.5`.
#[test]
fn flat_atoms_reseed_from_the_worst_reconstructed_rows_at_every_retry_2023() {
    let mut term = trivial_k1_euclidean_term();
    assert_eq!(
        (term.n_obs(), term.output_dim()),
        (4, 3),
        "#2023: the planted residual below is written for this fixture's shape"
    );
    let target = ndarray::array![
        [0.5, 0.0, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 2.0],
        [1.2, 0.0, 0.0],
    ];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);

    term.reseed_atoms_from_residual(&[0], target.view(), &rho, 0)
        .expect("#2023: the first reseed must succeed");
    let coords = term.assignment.coords[0].as_matrix();
    assert_eq!(
        coords[[2, 0]],
        0.5,
        "#2023: retry 0 must anchor at the worst-reconstructed row, row 2: {coords:?}"
    );

    term.reseed_atoms_from_residual(&[0], target.view(), &rho, 1)
        .expect("#2023: the second reseed must succeed");
    let coords = term.assignment.coords[0].as_matrix();
    assert_eq!(
        coords[[3, 0]],
        0.5,
        "#2023: retry 1 must anchor at the second-worst row, row 3, instead of \
         re-anchoring on row 2: {coords:?}"
    );
}

/// On a residual whose worst row and leading principal direction disagree, a flat
/// reseed follows the worst row.
///
/// Rows 0..=2 of the target spread along the first output axis, which carries most of
/// the centred residual's variance (covariance `[[2.75, -0.375], [-0.375, 1.6875]]`
/// on the first two axes), while row 3 is the single worst-reconstructed row and lies
/// on the second axis. Anchored at row 3, rows 0..=2 project to zero and read `-0.5`,
/// and row 3 reads `+0.5`. The principal-component seed of the same residual is
/// computed as the control: it separates row 0 from row 1, so a reseed that read
/// principal components would fail the first assertion.
#[test]
fn flat_reseed_follows_the_worst_row_where_principal_components_disagree_2023() {
    let mut term = trivial_k1_euclidean_term();
    let target = ndarray::array![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.5, 0.0],
    ];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    let residual = term
        .reconstruction_residual(target.view(), &rho)
        .expect("#2023: the fixture's residual must evaluate");

    term.reseed_atoms_from_residual(&[0], target.view(), &rho, 0)
        .expect("#2023: the reseed must succeed");
    let coords = term.assignment.coords[0].as_matrix();
    assert_eq!(
        (coords[[0, 0]], coords[[1, 0]], coords[[2, 0]], coords[[3, 0]]),
        (-0.5, -0.5, -0.5, 0.5),
        "#2023: the reseed must anchor at the worst row (row 3), not along the leading \
         principal direction: {coords:?}"
    );

    let principal = sae_pca_seed_initial_coords_with_pc_offset(
        residual.view(),
        &[SaeAtomBasisKind::EuclideanPatch],
        &[1],
        0,
    )
    .expect("#2023: the control principal-component seed must evaluate");
    assert!(
        (principal[[0, 0, 0]] - principal[[0, 1, 0]]).abs() > 0.5,
        "#2023: the control must discriminate — the principal-component seed separates \
         rows 0 and 1 on this residual: {principal:?}"
    );
}

/// Circle atoms reseed from the residual graph's harmonics: the reseeded coordinates
/// are exactly the harmonic seed of the residual at the same retry.
#[test]
fn circle_atoms_reseed_from_the_residual_graph_harmonics_2023() {
    let (mut term, target, rho) = small_two_atom_periodic_term();
    let residual = term
        .reconstruction_residual(target.view(), &rho)
        .expect("#2023: the fixture's residual must evaluate");
    let kinds = vec![
        term.atoms[0].basis_kind().clone(),
        term.atoms[1].basis_kind().clone(),
    ];
    let dims = vec![term.atoms[0].latent_dim(), term.atoms[1].latent_dim()];
    let expected = topology_curved_seed_initial_coords(residual.view(), &kinds, &dims, 1)
        .expect("#2023: the harmonic seed must evaluate")
        .expect("#2023: five rows admit the residual's kNN graph");

    term.reseed_atoms_from_residual(&[0, 1], target.view(), &rho, 1)
        .expect("#2023: the periodic fixture must reseed");
    for atom in 0..2 {
        let coords = term.assignment.coords[atom].as_matrix();
        for row in 0..term.n_obs() {
            assert_eq!(
                coords[[row, 0]],
                expected[[atom, row, 0]],
                "#2023: atom {atom} row {row} must carry the harmonic seed"
            );
        }
    }
}

/// A chart the harmonic seed does not cover (RP²) reseeds from data rows: every row is
/// a unit ambient 3-vector, and the worst-reconstructed row points along the first
/// frame direction, because its projections on the later Gram–Schmidt directions are
/// zero.
#[test]
fn projective_plane_reseed_rows_are_unit_vectors_anchored_at_the_worst_row_2023() {
    let n = 6usize;
    let p = 4usize;
    let residual = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        ((row * 5 + col * 3) as f64 * 0.7).sin() + 0.1 * (row as f64)
    });
    let worst = (0..n)
        .max_by(|&a, &b| {
            let ea: f64 = residual.row(a).iter().map(|v| v * v).sum();
            let eb: f64 = residual.row(b).iter().map(|v| v * v).sum();
            ea.total_cmp(&eb).then_with(|| b.cmp(&a))
        })
        .expect("#2023: the residual has rows");

    let seed = sae_data_row_anchored_coords(
        residual.view(),
        &[SaeAtomBasisKind::ProjectivePlane],
        &[2],
        0,
    )
    .expect("#2023: the RP² data-row seed must evaluate");
    assert_eq!(seed.dim(), (1, n, 3), "#2023: RP² stores its ambient 3-vector");
    for row in 0..n {
        let norm = (0..3).map(|axis| seed[[0, row, axis]].powi(2)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() <= 1.0e-12,
            "#2023: row {row} must lie on the unit sphere, norm {norm}"
        );
    }
    let anchor = (seed[[0, worst, 0]], seed[[0, worst, 1]], seed[[0, worst, 2]]);
    assert!(
        (anchor.0 - 1.0).abs() <= 1.0e-12 && anchor.1.abs() <= 1.0e-12 && anchor.2.abs() <= 1.0e-12,
        "#2023: the worst row {worst} must read the first frame direction, got {anchor:?}"
    );
}
