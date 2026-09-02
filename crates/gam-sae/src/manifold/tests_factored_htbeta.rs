//! Factored-Htβ / low-rank-decoder REML tests, split out of `tests.rs` to keep
//! that file under the #780 10k-line gate. These exercise the factored border
//! dimension, the projected beta-penalty curvature, the native row-Htβ solve,
//! and the factored evidence / Occam term against the dense full-B baseline.

use super::*;
use approx::assert_abs_diff_eq;
use ndarray::array;

pub(crate) fn materialize_row_htbeta_for_test(
    sys: &ArrowSchurSystem,
    row_idx: usize,
) -> Array2<f64> {
    let di = sys.row_dims[row_idx];
    let k = sys.k;
    let row = &sys.rows[row_idx];
    let use_dense = sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none();
    let mut out = if use_dense && row.htbeta.dim() == (di, k) {
        row.htbeta.clone()
    } else {
        Array2::<f64>::zeros((di, k))
    };
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        let mut basis = Array1::<f64>::zeros(k);
        let mut col = Array1::<f64>::zeros(di);
        for beta_col in 0..k {
            basis.fill(0.0);
            basis[beta_col] = 1.0;
            col.fill(0.0);
            op(row_idx, basis.view(), &mut col);
            for row_col in 0..di {
                out[[row_col, beta_col]] += col[row_col];
            }
        }
    }
    out
}

pub(crate) fn project_row_htbeta_to_factored_for_test(
    term: &SaeManifoldTerm,
    htbeta_b: ArrayView2<'_, f64>,
) -> Array2<f64> {
    FrameProjection::new(term).project_rows(htbeta_b)
}

pub(crate) fn low_rank_factored_htbeta_term(
    k_atoms: usize,
    m: usize,
    p: usize,
    frame_rank: usize,
    latent_dim: usize,
    n_obs: usize,
) -> SaeManifoldTerm {
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let coords = Array2::from_shape_fn((n_obs, latent_dim), |(row, axis)| {
            let phase = (row + 1) as f64 * (axis + 2) as f64 + 0.37 * (atom_idx + 1) as f64;
            0.2 * phase.sin() + 0.1 * (0.17 * phase).cos()
        });
        let mut phi = Array2::<f64>::zeros((n_obs, m));
        let mut jet = Array3::<f64>::zeros((n_obs, m, latent_dim));
        for row in 0..n_obs {
            for basis_col in 0..m {
                let base = (row + 1) as f64 * (basis_col + 1) as f64;
                phi[[row, basis_col]] = if basis_col == 0 { 1.0 } else { 0.0 }
                    + 0.01 * (base + 3.0 * atom_idx as f64).sin();
                for axis in 0..latent_dim {
                    jet[[row, basis_col, axis]] =
                        0.005 * ((base * (axis + 1) as f64) + atom_idx as f64).cos();
                }
            }
        }
        let mut frame = Array2::<f64>::zeros((p, frame_rank));
        for frame_col in 0..frame_rank {
            frame[[(atom_idx * frame_rank + frame_col) % p, frame_col]] = 1.0;
        }
        let coords_c = Array2::from_shape_fn((m, frame_rank), |(basis_col, frame_col)| {
            0.2 + 0.03 * (basis_col + 2 * frame_col + atom_idx) as f64
        });
        let decoder = coords_c.dot(&frame.t());
        let mut atom = SaeManifoldAtom::new_with_provided_function_gram(
            "factored_htbeta_shape",
            SaeAtomBasisKind::EuclideanPatch,
            latent_dim,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .expect("atom fixture: basis, jet, decoder and Gram shapes agree by construction");
        atom.maybe_activate_decoder_frame()
            .expect("frame activation")
            .expect("low-rank atom should activate a frame");
        atoms.push(atom);
        coord_blocks.push(coords);
    }
    let logits = Array2::<f64>::from_shape_fn((n_obs, k_atoms), |(row, atom)| {
        0.03 * ((row + 1) as f64 * (atom + 2) as f64).sin()
    });
    let manifolds =
        vec![LatentManifold::Product(vec![LatentManifold::Euclidean; latent_dim]); k_atoms];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::softmax(0.9),
    )
    .expect(
        "assignment fixture: logits, coordinate blocks and manifolds agree in block count and rows",
    );
    SaeManifoldTerm::new(atoms, assignment)
        .expect("term fixture: every atom's row count matches the assignment's")
}

pub(crate) fn factored_htbeta_rho(k_atoms: usize, latent_dim: usize) -> SaeManifoldRho {
    SaeManifoldRho::new(0.0, -0.2, vec![Array1::<f64>::zeros(latent_dim); k_atoms])
}

#[test]
pub(crate) fn factored_row_htbeta_d2_matches_dense_full_b_then_project() {
    let k_atoms = 3usize;
    let m = 5usize;
    let p = 32usize;
    let frame_rank = 2usize;
    let latent_dim = 2usize;
    let n_obs = 6usize;
    let mut factored_term =
        low_rank_factored_htbeta_term(k_atoms, m, p, frame_rank, latent_dim, n_obs);
    assert!(factored_term.frames_active());
    assert_eq!(
        factored_term.factored_border_dim(),
        k_atoms * m * frame_rank
    );
    assert!(factored_term.factored_border_dim() < factored_term.beta_dim());

    let mut full_term = factored_term.clone();
    for atom in &mut full_term.atoms {
        atom.deactivate_decoder_frame();
    }
    let rho = factored_htbeta_rho(k_atoms, latent_dim);
    let target = Array2::<f64>::from_shape_fn((n_obs, p), |(row, col)| {
        0.01 * (row + 1) as f64 - 0.002 * (col + 1) as f64
    });

    let native_sys = factored_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let full_sys = full_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let mut projected_sys = factored_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    projected_sys.htbeta_matvec = None;
    projected_sys.htbeta_transpose_matvec = None;
    projected_sys.htbeta_dense_supplement = false;
    for row_idx in 0..n_obs {
        let htbeta_b = materialize_row_htbeta_for_test(&full_sys, row_idx);
        projected_sys.rows[row_idx].htbeta =
            project_row_htbeta_to_factored_for_test(&factored_term, htbeta_b.view());
    }
    projected_sys.refresh_row_hessian_fingerprint();

    let ridge_t = 5.0e-1;
    let (native_dt, native_db, _) = native_sys.solve(ridge_t, 1.0e-8).unwrap();
    let (projected_dt, projected_db, _) = projected_sys.solve(ridge_t, 1.0e-8).unwrap();
    assert_eq!(native_dt.len(), projected_dt.len());
    assert_eq!(native_db.len(), projected_db.len());
    for idx in 0..native_dt.len() {
        assert_abs_diff_eq!(native_dt[idx], projected_dt[idx], epsilon = 1.0e-10);
    }
    for idx in 0..native_db.len() {
        assert_abs_diff_eq!(native_db[idx], projected_db[idx], epsilon = 1.0e-10);
    }
}

#[test]
pub(crate) fn qwen_shape_d2_factored_htbeta_assembly_stays_below_8gib() {
    const K_ATOMS: usize = 8;
    const M: usize = 10;
    const P: usize = 2048;
    const FRAME_RANK: usize = 2;
    const LATENT_DIM: usize = 2;
    const N_OBS: usize = 2000;
    const EIGHT_GIB: usize = 8 * 1024 * 1024 * 1024;

    let mut term = low_rank_factored_htbeta_term(K_ATOMS, M, P, FRAME_RANK, LATENT_DIM, N_OBS);
    assert!(term.frames_active());
    assert_eq!(term.beta_dim(), K_ATOMS * M * P);
    assert_eq!(term.factored_border_dim(), K_ATOMS * M * FRAME_RANK);
    assert!(term.factored_border_dim() < term.beta_dim());

    let rho = factored_htbeta_rho(K_ATOMS, LATENT_DIM);
    let target = Array2::<f64>::from_shape_fn((N_OBS, P), |(row, col)| {
        1.0e-4 * ((row + 1) as f64 * (col + 3) as f64).sin()
    });
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();

    assert_eq!(sys.k, term.factored_border_dim());
    assert!(sys.htbeta_matvec.is_none());
    assert!(sys.htbeta_transpose_matvec.is_none());
    let actual_row_dim = sys.row_dims[0];
    assert!(actual_row_dim > 0);
    assert!(sys.row_dims.iter().all(|&dim| dim == actual_row_dim));
    for row in &sys.rows {
        assert_eq!(row.htbeta.ncols(), term.factored_border_dim());
        assert_eq!(row.htbeta.nrows(), actual_row_dim);
    }

    let htbeta_bytes: usize = sys
        .rows
        .iter()
        .map(|row| row.htbeta.len() * std::mem::size_of::<f64>())
        .sum();
    let assembled_dense_bytes = htbeta_bytes
        + sys.hbb.len() * std::mem::size_of::<f64>()
        + sys.gb.len() * std::mem::size_of::<f64>();
    let old_full_b_htbeta_bytes = N_OBS
        .saturating_mul(actual_row_dim)
        .saturating_mul(term.beta_dim())
        .saturating_mul(std::mem::size_of::<f64>());

    assert!(
        old_full_b_htbeta_bytes > EIGHT_GIB,
        "test shape must reproduce the old p-wide H_tbeta memory wall"
    );
    assert!(
        assembled_dense_bytes < EIGHT_GIB,
        "qwen-shaped factored assembly stored {assembled_dense_bytes} bytes, \
             exceeding the 8 GiB gate"
    );
}

/// A full-rank small-`p` decoder must NOT activate a frame: the factored
/// border equals the full `M_k·p`, the Grassmann evidence dimension is `0`,
/// and the Occam normalizer is bit-for-bit the historical
/// `½·p·rank(S)·log λ` — the small-`p` evidence-equality contract.
#[test]
pub(crate) fn factored_evidence_matches_full_b_at_small_p() {
    let m = 5usize;
    let p = 2usize;
    // Full-rank decoder (rank 2 == p): no border saving, frame must stay off.
    let mut decoder = Array2::<f64>::zeros((m, p));
    for mu in 0..m {
        decoder[[mu, 0]] = 1.0 + mu as f64;
        decoder[[mu, 1]] = (mu as f64) - 2.0;
    }
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        jet[[mu, mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new_with_provided_function_gram(
        "fullrank",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap();
    let activated = atom.maybe_activate_decoder_frame().expect("activate");
    assert_eq!(
        activated, None,
        "full-rank small-p must stay on full-B path"
    );
    assert!(atom.decoder_frame.is_none());
    assert_eq!(atom.border_frame_rank(), p);
    assert_eq!(atom.frame_manifold_dimension(), 0);

    let mut term = SaeManifoldTerm::new(
        vec![atom],
        SaeAssignment::from_blocks_with_mode(
            Array2::<f64>::zeros((m, 1)),
            vec![Array2::<f64>::zeros((m, 1))],
            AssignmentMode::softmax(0.7),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(!term.frames_active());
    assert_eq!(term.factored_border_dim(), term.beta_dim());
    assert_eq!(term.grassmann_evidence_dimension(), 0);
    let activated_n = term.auto_activate_decoder_frames().expect("auto");
    assert_eq!(activated_n, 0, "small-p auto-activation must be a no-op");

    // Occam normalizer equals the historical ½·p·rank(S)·log λ exactly.
    let rho = SaeManifoldRho::new(0.0, 0.37, vec![array![0.0_f64]]);
    let occam = term.reml_occam_term(&rho).expect("occam");
    let rank_s = SaeManifoldTerm::symmetric_rank(term.atoms[0].smooth_penalty()).unwrap();
    let expected = 0.5 * (p as f64) * (rank_s as f64) * rho.log_lambda_smooth[0];
    assert_abs_diff_eq!(occam, expected, epsilon = 1.0e-12);
}
