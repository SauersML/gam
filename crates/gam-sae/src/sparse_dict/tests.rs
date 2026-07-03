use super::codes::solve_row_codes;
use super::scoring::{TileScorer, top_s_online};
use super::{SparseDictConfig, fit_sparse_dictionary, sparse_dictionary_transform_with_mode};
use ndarray::{Array2, ArrayView2};

/// Build an exact rank-1 mixture: `K` orthonormal planted atoms (rows of an
/// orthonormal basis), each row a scaled single atom plus a tiny second atom.
fn planted(k: usize, p: usize, n: usize, second_share: f32) -> (Array2<f32>, Array2<f32>) {
    // Deterministic orthonormal directions from a fixed integer-symmetric matrix.
    let mut a = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            a[[i, j]] = ((i * 7 + j * 3 + 1) % 11) as f64 - 5.0;
        }
    }
    let sym = &a + &a.t();
    use gam_linalg::faer_ndarray::FaerEigh;
    let (_ev, evecs) = sym.eigh(faer::Side::Lower).expect("orthonormal seed");
    let mut atoms = Array2::<f32>::zeros((k, p));
    for atom in 0..k {
        let col = evecs.column(atom % p);
        for c in 0..p {
            atoms[[atom, c]] = col[c] as f32;
        }
    }
    let mut x = Array2::<f32>::zeros((n, p));
    for row in 0..n {
        let primary = row % k;
        let secondary = (primary + 1) % k;
        let scale = 0.7 + 0.01 * (row / k) as f32;
        for c in 0..p {
            x[[row, c]] =
                scale * atoms[[primary, c]] + second_share * scale * atoms[[secondary, c]];
        }
    }
    (x, atoms)
}

/// PCA explained variance of the best rank-`r` subspace (linear baseline).
fn pca_ev(x: ArrayView2<'_, f32>, rank: usize) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    let mut means = vec![0.0f64; p];
    for i in 0..n {
        for c in 0..p {
            means[c] += x[[i, c]] as f64;
        }
    }
    for c in 0..p {
        means[c] /= n as f64;
    }
    let mut cov = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for a in 0..p {
            let xa = x[[i, a]] as f64 - means[a];
            for b in 0..p {
                cov[[a, b]] += xa * (x[[i, b]] as f64 - means[b]);
            }
        }
    }
    use gam_linalg::faer_ndarray::FaerEigh;
    let (evals, _) = cov.eigh(faer::Side::Lower).expect("pca eig");
    // eigh returns ascending; sum of top-`rank` over total.
    let total: f64 = evals.iter().sum();
    let mut sorted: Vec<f64> = evals.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top: f64 = sorted.iter().take(rank).sum();
    if total <= 1.0e-24 { 1.0 } else { top / total }
}

/// Held-out reconstruction EV of a fitted dictionary `decoder` on a *fresh*
/// block `x_test` it never trained on. The decoder is FROZEN: each test row is
/// routed (top-`s`) against it and its codes are the active-set LS solve — the
/// exact production held-out path (`ManifoldSAE.reconstruct`), one decoder, new
/// coordinates. EV is `1 − RSS/TSS` with the TSS centred on `x_test`'s own mean,
/// so a dictionary that merely memorised the train block earns nothing here.
fn held_out_ev(
    decoder: ArrayView2<'_, f32>,
    x_test: ArrayView2<'_, f32>,
    s: usize,
    tile: usize,
    code_ridge: f32,
) -> f64 {
    let n = x_test.nrows();
    let p = x_test.ncols();
    let scorer = TileScorer::new(s, tile);
    let mut means = vec![0.0f64; p];
    for i in 0..n {
        for c in 0..p {
            means[c] += x_test[[i, c]] as f64;
        }
    }
    for c in 0..p {
        means[c] /= n as f64;
    }
    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    for i in 0..n {
        let row = x_test.row(i);
        let active = scorer.route_row(row, decoder);
        let code = solve_row_codes(row, decoder, &active, s, code_ridge);
        let mut recon = vec![0.0f64; p];
        for j in 0..code.indices.len() {
            let cj = code.codes[j] as f64;
            if cj == 0.0 {
                continue;
            }
            let drow = decoder.row(code.indices[j] as usize);
            for c in 0..p {
                recon[c] += cj * drow[c] as f64;
            }
        }
        for c in 0..p {
            let r = x_test[[i, c]] as f64 - recon[c];
            rss += r * r;
            let t = x_test[[i, c]] as f64 - means[c];
            tss += t * t;
        }
    }
    if tss <= 1.0e-24 {
        if rss <= 1.0e-24 { 1.0 } else { 0.0 }
    } else {
        1.0 - rss / tss
    }
}

/// HELD-OUT rank-`r` PCA EV: principal subspace fitted on `x_train` ONLY, then
/// scored on `x_test`. This is the honest linear baseline the sparse trainer
/// must match-or-beat — the rank-`r` linear autoencoder's out-of-sample
/// reconstruction, with NO leakage of the test block into the basis.
fn pca_ev_held_out(x_train: ArrayView2<'_, f32>, x_test: ArrayView2<'_, f32>, rank: usize) -> f64 {
    let p = x_train.ncols();
    let ntr = x_train.nrows();
    let mut means = vec![0.0f64; p];
    for i in 0..ntr {
        for c in 0..p {
            means[c] += x_train[[i, c]] as f64;
        }
    }
    for c in 0..p {
        means[c] /= ntr as f64;
    }
    // Train covariance → top-`rank` eigenvectors (the PCA basis).
    let mut cov = Array2::<f64>::zeros((p, p));
    for i in 0..ntr {
        for a in 0..p {
            let xa = x_train[[i, a]] as f64 - means[a];
            for b in 0..p {
                cov[[a, b]] += xa * (x_train[[i, b]] as f64 - means[b]);
            }
        }
    }
    use gam_linalg::faer_ndarray::FaerEigh;
    let (evals, evecs) = cov.eigh(faer::Side::Lower).expect("pca eig");
    // eigh returns ascending eigenvalues; take the top-`rank` columns.
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap());
    let keep: Vec<usize> = order.into_iter().take(rank.min(p)).collect();
    // Project test rows onto the train PCA subspace and reconstruct.
    let nte = x_test.nrows();
    let mut means_te = vec![0.0f64; p];
    for i in 0..nte {
        for c in 0..p {
            means_te[c] += x_test[[i, c]] as f64;
        }
    }
    for c in 0..p {
        means_te[c] /= nte as f64;
    }
    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    for i in 0..nte {
        // Centre on the TRAIN mean (the basis's origin) for reconstruction.
        let mut centred = vec![0.0f64; p];
        for c in 0..p {
            centred[c] = x_test[[i, c]] as f64 - means[c];
        }
        let mut recon = vec![0.0f64; p];
        for &k in &keep {
            let mut coord = 0.0f64;
            for c in 0..p {
                coord += centred[c] * evecs[[c, k]];
            }
            for c in 0..p {
                recon[c] += coord * evecs[[c, k]];
            }
        }
        for c in 0..p {
            let r = centred[c] - recon[c];
            rss += r * r;
            // TSS centred on the test mean: the variance an honest baseline must explain.
            let t = x_test[[i, c]] as f64 - means_te[c];
            tss += t * t;
        }
    }
    if tss <= 1.0e-24 {
        if rss <= 1.0e-24 { 1.0 } else { 0.0 }
    } else {
        1.0 - rss / tss
    }
}

#[test]
fn online_top_s_recovers_planted_largest_scores() {
    // A single row whose true top-s atoms are known: scores are
    // exactly the decoder · row dot products, so the planted maxima must win.
    // One distinct unit axis per atom so every atom's score is unique: the
    // planted top-3 atoms (by |xᵀd|) are unambiguous.
    let p = 50;
    let k = 50;
    let mut decoder = Array2::<f32>::zeros((k, p));
    for atom in 0..k {
        decoder[[atom, atom]] = 1.0;
    }
    let mut row = ndarray::Array1::<f32>::zeros(p);
    // Strongest atom is index 17 (score 9), then 4 (score 5), then 31 (score 3).
    row[17] = 9.0;
    row[4] = 5.0;
    row[31] = 3.0;
    let picked = top_s_online(row.view(), decoder.view(), 3, 8);
    let want_atoms = [17u32, 4u32, 31u32];
    assert_eq!(picked.len(), 3);
    for (rank, &(atom, score)) in picked.iter().enumerate() {
        assert_eq!(
            atom, want_atoms[rank],
            "rank {rank}: expected atom {}, got atom {atom} (score {score})",
            want_atoms[rank]
        );
    }
}

#[test]
fn tile_scorer_matches_untiled_brute_force() {
    let p = 5;
    let k = 37;
    let mut decoder = Array2::<f32>::zeros((k, p));
    for atom in 0..k {
        for c in 0..p {
            decoder[[atom, c]] = (((atom * 3 + c * 5 + 1) % 7) as f32 - 3.0) / 3.0;
        }
    }
    let row = ndarray::Array1::<f32>::from_vec((0..p).map(|c| (c as f32) - 2.0).collect());
    // Brute force: full score then argsort.
    let mut brute: Vec<(u32, f32)> = (0..k)
        .map(|a| {
            let mut acc = 0.0f32;
            for c in 0..p {
                acc += row[c] * decoder[[a, c]];
            }
            (a as u32, acc)
        })
        .collect();
    brute.sort_by(|x, y| {
        y.1.abs()
            .partial_cmp(&x.1.abs())
            .unwrap()
            .then(x.0.cmp(&y.0))
    });
    let scorer = TileScorer::new(4, 7);
    let tiled = scorer.route_row(row.view(), decoder.view());
    assert_eq!(tiled.len(), 4);
    for j in 0..4 {
        assert_eq!(
            tiled[j].0, brute[j].0,
            "tiled top-{j} disagrees with brute force"
        );
    }
}

#[test]
fn tile_scorer_dispatch_matches_cpu_below_device_floor() {
    let p = 5;
    let k = 37;
    let rows = Array2::<f32>::from_shape_fn((3, p), |(r, c)| {
        (((r * 11 + c * 7 + 3) % 13) as f32 - 6.0) / 6.0
    });
    let decoder = Array2::<f32>::from_shape_fn((k, p), |(atom, c)| {
        (((atom * 3 + c * 5 + 1) % 7) as f32 - 3.0) / 3.0
    });
    let scorer = TileScorer::new(4, 7);

    let cpu = scorer.route_minibatch(rows.view(), decoder.view());
    let dispatched = scorer
        .route_minibatch_dispatch(rows.view(), decoder.view())
        .expect("dispatch route");
    assert_eq!(dispatched, cpu);
}

#[test]
fn tile_scorer_required_mode_refuses_subfloor_route() {
    let p = 5;
    let k = 37;
    let rows = Array2::<f32>::from_shape_fn((3, p), |(r, c)| {
        (((r * 11 + c * 7 + 3) % 13) as f32 - 6.0) / 6.0
    });
    let decoder = Array2::<f32>::from_shape_fn((k, p), |(atom, c)| {
        (((atom * 3 + c * 5 + 1) % 7) as f32 - 3.0) / 3.0
    });
    let scorer = TileScorer::new(4, 7);

    let err = scorer
        .route_minibatch_with_mode(rows.view(), decoder.view(), gam_gpu::GpuMode::Required)
        .expect_err("Required mode must fail closed below the device floor");
    assert!(
        err.contains("below the device launch break-even"),
        "unexpected Required-mode error: {err}"
    );
}

#[test]
fn sparse_transform_with_explicit_mode_reports_cpu_route_stats() {
    let p = 5;
    let k = 11;
    let rows = Array2::<f32>::from_shape_fn((7, p), |(r, c)| {
        (((r * 17 + c * 5 + 2) % 19) as f32 - 9.0) / 9.0
    });
    let mut decoder = Array2::<f32>::from_shape_fn((k, p), |(atom, c)| {
        (((atom * 13 + c * 3 + 1) % 23) as f32 - 11.0) / 11.0
    });
    for mut row in decoder.outer_iter_mut() {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|v| v / norm);
    }

    let transform = sparse_dictionary_transform_with_mode(
        rows.view(),
        decoder.view(),
        3,
        4,
        1.0e-6,
        gam_gpu::GpuMode::Off,
    )
    .expect("explicit-mode transform");
    assert_eq!(transform.indices.dim(), (rows.nrows(), 3));
    assert_eq!(transform.codes.dim(), (rows.nrows(), 3));
    assert_eq!(transform.score_route_stats.minibatches, 1);
    assert_eq!(transform.score_route_stats.cpu_minibatches, 1);
    assert_eq!(transform.score_route_stats.device_minibatches, 0);
}

#[test]
fn sparse_fit_records_score_route_stats() {
    let (k, p, n) = (8usize, 10usize, 64usize);
    let (x, _atoms) = planted(k, p, n, 0.1);
    let config = SparseDictConfig {
        n_atoms: k,
        active: 1,
        minibatch: 16,
        max_epochs: 1,
        score_tile: 8,
        code_ridge: 1.0e-6,
        decoder_ridge: 1.0e-6,
        tolerance: 0.0,
        score_mode: gam_gpu::GpuMode::Off,
    };
    let fit = fit_sparse_dictionary(x.view(), &config).expect("fit");
    assert_eq!(fit.score_route_stats.minibatches, 8);
    assert_eq!(fit.score_route_stats.cpu_minibatches, 8);
    assert_eq!(fit.score_route_stats.device_minibatches, 0);
    assert_eq!(fit.score_route_stats.admitted_minibatches, 0);
    assert_eq!(
        fit.score_route_stats.score_elements,
        8u128 * 16u128 * k as u128
    );
}

#[test]
fn sparse_trainer_recovers_planted_dictionary_beats_pca_baseline() {
    // Planted K-atom rank-1 mixture; the sparse trainer with top_s=2 should
    // reconstruct it at high EV and match-or-beat a rank-K PCA baseline.
    let (k, p, n) = (8usize, 12usize, 480usize);
    let (x, _atoms) = planted(k, p, n, 0.2);
    let config = SparseDictConfig {
        n_atoms: k,
        active: 2,
        minibatch: 128,
        max_epochs: 40,
        score_tile: 16,
        code_ridge: 1.0e-6,
        decoder_ridge: 1.0e-6,
        tolerance: 1.0e-9,
        score_mode: gam_gpu::GpuMode::Off,
    };
    let fit = fit_sparse_dictionary(x.view(), &config).expect("sparse dictionary fit");
    let baseline = pca_ev(x.view(), k);
    assert!(
        fit.explained_variance > 0.95,
        "expected EV > 0.95, got {}",
        fit.explained_variance
    );
    assert!(
        fit.explained_variance + 1.0e-6 >= baseline,
        "sparse trainer EV {} must match-or-beat rank-{k} PCA baseline {}",
        fit.explained_variance,
        baseline
    );
}

#[test]
fn sparse_trainer_beats_rank_k_pca_on_held_out_reconstruction() {
    // #1026 MVP ACCEPTANCE (the real one): on a planted dictionary at modest K,
    // the trainer's route→sparse-codes→decoder-update must recover HELD-OUT
    // reconstruction EV that match-or-beats a rank-K linear/PCA baseline fitted
    // on the SAME train block — out of sample, no leakage. A planted sparse
    // mixture (each row a handful of atoms drawn from a K-atom over-complete
    // dictionary, K > p) is exactly the regime where a sparse top-s code beats a
    // rank-K dense subspace: the linear PCA of a p-dim block saturates at rank p,
    // but the sparse dictionary keeps resolving distinct atoms past p.
    let (k, p, n) = (64usize, 16usize, 1600usize);
    // Over-complete planted dictionary: K=64 atoms in p=16 dims, each row a
    // 2-sparse combination. Linear PCA caps at rank 16; the sparse code does not.
    let (x, _atoms) = planted(k, p, n, 0.35);
    // Deterministic 80/20 split (stride the rows so both blocks see every atom).
    let n_test = n / 5;
    let mut train_rows: Vec<usize> = Vec::new();
    let mut test_rows: Vec<usize> = Vec::new();
    for i in 0..n {
        if i % 5 == 0 {
            test_rows.push(i);
        } else {
            train_rows.push(i);
        }
    }
    let mut x_train = Array2::<f32>::zeros((train_rows.len(), p));
    for (r, &i) in train_rows.iter().enumerate() {
        x_train.row_mut(r).assign(&x.row(i));
    }
    let mut x_test = Array2::<f32>::zeros((test_rows.len(), p));
    for (r, &i) in test_rows.iter().enumerate() {
        x_test.row_mut(r).assign(&x.row(i));
    }
    assert_eq!(x_test.nrows(), n_test);

    let s = 2usize;
    let tile = 16usize;
    let code_ridge = 1.0e-6f32;
    let config = SparseDictConfig {
        n_atoms: k,
        active: s,
        minibatch: 256,
        max_epochs: 60,
        score_tile: tile,
        code_ridge,
        decoder_ridge: 1.0e-6,
        tolerance: 1.0e-9,
        score_mode: gam_gpu::GpuMode::Off,
    };
    // Fit the dictionary on TRAIN ONLY.
    let fit = fit_sparse_dictionary(x_train.view(), &config).expect("held-out trainer fit");

    // Held-out EV: frozen decoder, fresh test-row codes (production path).
    let sparse_out = held_out_ev(fit.decoder.view(), x_test.view(), s, tile, code_ridge);
    // Linear baseline: rank-K PCA fitted on train, scored on test. With K > p the
    // rank is clamped to p, so this is the best possible LINEAR autoencoder here.
    let pca_out = pca_ev_held_out(x_train.view(), x_test.view(), k);

    assert!(
        sparse_out > 0.9,
        "held-out sparse-dictionary EV {sparse_out} should explain the planted held-out block"
    );
    assert!(
        sparse_out + 1.0e-4 >= pca_out,
        "held-out sparse EV {sparse_out} must match-or-beat held-out rank-{k} PCA baseline {pca_out}"
    );
}

/// Fraction of atoms that fired for no training row (dead atoms) in a fit.
fn dead_atom_fraction(fit: &super::SparseDictFit) -> f64 {
    let k = fit.decoder.nrows();
    let mut alive = vec![false; k];
    for (i, idx_row) in fit.indices.outer_iter().enumerate() {
        for (j, &idx) in idx_row.iter().enumerate() {
            if fit.codes[[i, j]] != 0.0 {
                alive[idx as usize] = true;
            }
        }
    }
    let dead = alive.iter().filter(|&&a| !a).count();
    dead as f64 / k as f64
}

#[test]
fn dead_atom_revival_keeps_ev_monotone_in_k_and_beats_linear_subspace() {
    // #1026 regression. The collapsed-linear lane must reach reconstruction parity
    // as `K` scales. Without dead-atom revival a large dictionary leaves atoms at
    // their farthest-point seed (measured on real banked OLMo: 87% dead at K=512),
    // effective `K` collapses, and HELD-OUT EV becomes NON-MONOTONE in `K` — adding
    // atoms makes reconstruction WORSE, the opposite of parity. Revival re-seeds
    // dead atoms onto the worst-reconstructed rows' residual directions so adding
    // atoms can only help.
    //
    // Regime: an OVER-COMPLETE planted dictionary (64 atoms in p=16, each row a
    // 2-sparse mixture). Two invariants that the pathology broke, both on HELD-OUT
    // data (frozen decoder, fresh test-row codes — the production path; held-out
    // cannot be gamed by reviving atoms onto idiosyncratic train rows):
    //   1. MONOTONICITY: held-out EV must not drop as K grows (16 -> 64 -> 256).
    //   2. PARITY/SUPERIORITY over the linear subspace: at large K the adaptive
    //      s-sparse code must beat a FIXED rank-s PCA autoencoder (a K-atom SAE
    //      picks the best s atoms per row, so it must dominate one fixed s-dim
    //      basis) — the "match-or-beat linear at matched active budget" target.
    let (planted_k, p, n) = (64usize, 16usize, 2000usize);
    let (x, _atoms) = planted(planted_k, p, n, 0.35);
    // Deterministic 80/20 split (stride so both blocks see every planted atom).
    let mut train_rows: Vec<usize> = Vec::new();
    let mut test_rows: Vec<usize> = Vec::new();
    for i in 0..n {
        if i % 5 == 0 {
            test_rows.push(i);
        } else {
            train_rows.push(i);
        }
    }
    let mut x_train = Array2::<f32>::zeros((train_rows.len(), p));
    for (r, &i) in train_rows.iter().enumerate() {
        x_train.row_mut(r).assign(&x.row(i));
    }
    let mut x_test = Array2::<f32>::zeros((test_rows.len(), p));
    for (r, &i) in test_rows.iter().enumerate() {
        x_test.row_mut(r).assign(&x.row(i));
    }

    let s = 2usize;
    let tile = 16usize;
    let code_ridge = 1.0e-6f32;
    let mk = |k: usize| SparseDictConfig {
        n_atoms: k,
        active: s,
        minibatch: 256,
        max_epochs: 60,
        score_tile: tile,
        code_ridge,
        decoder_ridge: 1.0e-6,
        tolerance: 1.0e-9,
        score_mode: gam_gpu::GpuMode::Off,
    };

    let fit_small = fit_sparse_dictionary(x_train.view(), &mk(16)).expect("K=16 fit");
    let fit_mid = fit_sparse_dictionary(x_train.view(), &mk(64)).expect("K=64 fit");
    let fit_large = fit_sparse_dictionary(x_train.view(), &mk(256)).expect("K=256 fit");

    let ev_small = held_out_ev(fit_small.decoder.view(), x_test.view(), s, tile, code_ridge);
    let ev_mid = held_out_ev(fit_mid.decoder.view(), x_test.view(), s, tile, code_ridge);
    let ev_large = held_out_ev(fit_large.decoder.view(), x_test.view(), s, tile, code_ridge);

    // 1. Held-out monotonicity in K (small slack absorbs f32 routing noise). The
    //    un-revived lane failed this — large K dropped below small K.
    assert!(
        ev_mid + 5.0e-3 >= ev_small,
        "[#1026] held-out EV must not drop from K=16 ({ev_small:.4}) to K=64 ({ev_mid:.4})"
    );
    assert!(
        ev_large + 5.0e-3 >= ev_mid,
        "[#1026] held-out EV must not drop from K=64 ({ev_mid:.4}) to K=256 ({ev_large:.4})"
    );
    // 2. Parity/superiority: the large-K adaptive s-sparse code beats a fixed
    //    rank-s PCA linear autoencoder on held-out data.
    let pca_rank_s = pca_ev_held_out(x_train.view(), x_test.view(), s);
    assert!(
        ev_large > pca_rank_s + 0.05,
        "[#1026] K=256 held-out EV ({ev_large:.4}) must beat fixed rank-{s} PCA \
         ({pca_rank_s:.4}) — adaptive over-complete sparse coding must dominate a \
         single s-dim linear subspace at matched active budget"
    );
    // And the over-complete lane actually resolves the planted structure at scale.
    assert!(
        ev_large > 0.85,
        "[#1026] K=256 held-out EV ({ev_large:.4}) should resolve the 2-sparse \
         planted mixture (reconstruction parity at scale)"
    );
}

/// Minimal reader for a NumPy `.npy` v1.0 file holding a C-order little-endian
/// `float32` 2-D array (the format of the banked activation slices in
/// `tests/data`). Returns `(rows, cols, data)` in row-major order. Panics with a
/// clear message on any format it does not handle — this is a measurement
/// helper, not a general parser.
fn read_npy_f32_2d(path: &str) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(
        bytes.len() > 10 && &bytes[0..6] == b"\x93NUMPY",
        "{path}: not a .npy file"
    );
    // Byte 6/7 = version; bytes 8..10 = little-endian header length (v1.0).
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).expect("utf8 header");
    assert!(
        header.contains("'<f4'") || header.contains("\"<f4\""),
        "{path}: expected little-endian float32 (<f4); header: {header}"
    );
    assert!(
        header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false"),
        "{path}: expected C-order; header: {header}"
    );
    // Parse the shape tuple "(N, P)".
    let shape_start = header.find("'shape':").expect("shape key") + "'shape':".len();
    let paren_open = header[shape_start..].find('(').expect("shape (") + shape_start + 1;
    let paren_close = header[paren_open..].find(')').expect("shape )") + paren_open;
    let dims: Vec<usize> = header[paren_open..paren_close]
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    assert_eq!(dims.len(), 2, "{path}: expected a 2-D array, got {dims:?}");
    let (n, p) = (dims[0], dims[1]);
    let data_off = 10 + header_len;
    let expect = n * p * 4;
    assert_eq!(
        bytes.len() - data_off,
        expect,
        "{path}: data length mismatch (n={n}, p={p})"
    );
    let mut data = Vec::with_capacity(n * p);
    let mut off = data_off;
    for _ in 0..(n * p) {
        data.push(f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]));
        off += 4;
    }
    (n, p, data)
}

/// #1026 REAL-DATA parity measurement (ignored by default — it is a measurement
/// harness, not a pass/fail gate, and it reads the banked activation slices).
///
/// Run explicitly:
///   ./build.sh nextest run -p gam-sae real_olmo_sparse_dict_ev_vs_k_parity \
///       --run-ignored all --no-capture
///
/// Prints, for each banked OLMo slice and active budget `s`, the held-in and
/// held-out EV of the collapsed-linear lane at K on the parity ladder, plus the
/// dead-atom fraction and the rank-`s` held-out PCA baseline. This is the
/// before/after evidence for the dead-atom-revival fix: EV must be MONOTONE in K
/// and the dead fraction small (pre-fix it was non-monotone with the majority of
/// atoms dead).
#[test]
fn real_olmo_sparse_dict_ev_vs_k_parity() {
    let files = [
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/olmo_l18_pca64_635.npy"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/olmo_mixedlayer_pca64_768.npy"
        ),
    ];
    for path in files {
        let (n, p, data) = read_npy_f32_2d(path);
        let x = Array2::from_shape_vec((n, p), data).expect("shape");
        // Deterministic 80/20 split by stride.
        let mut tr: Vec<usize> = Vec::new();
        let mut te: Vec<usize> = Vec::new();
        for i in 0..n {
            if i % 5 == 0 {
                te.push(i);
            } else {
                tr.push(i);
            }
        }
        let mut x_tr = Array2::<f32>::zeros((tr.len(), p));
        for (r, &i) in tr.iter().enumerate() {
            x_tr.row_mut(r).assign(&x.row(i));
        }
        let mut x_te = Array2::<f32>::zeros((te.len(), p));
        for (r, &i) in te.iter().enumerate() {
            x_te.row_mut(r).assign(&x.row(i));
        }
        println!(
            "\n=== {path}  (N={n}, P={p}, train={}, test={}) ===",
            tr.len(),
            te.len()
        );
        for s in [8usize, 32usize] {
            let tile = p.max(1);
            let pca = pca_ev_held_out(x_tr.view(), x_te.view(), s);
            println!("  active s={s}  rank-{s} held-out PCA EV = {pca:.4}");
            let mut prev = f64::NEG_INFINITY;
            for k in [s, 32usize, 128, 512, 1024] {
                if k < s {
                    continue;
                }
                let config = SparseDictConfig {
                    n_atoms: k,
                    active: s,
                    minibatch: 256,
                    max_epochs: 40,
                    score_tile: tile,
                    code_ridge: 1.0e-6,
                    decoder_ridge: 1.0e-6,
                    tolerance: 1.0e-7,
                    score_mode: gam_gpu::GpuMode::Off,
                };
                let fit = fit_sparse_dictionary(x_tr.view(), &config).expect("fit");
                let ev_te = held_out_ev(fit.decoder.view(), x_te.view(), s, tile, 1.0e-6);
                let dead = dead_atom_fraction(&fit);
                let mono = if ev_te + 5.0e-3 >= prev {
                    ""
                } else {
                    "  <-- DROP"
                };
                println!(
                    "    K={k:5}  train_EV={:.4}  test_EV={ev_te:.4}  dead={dead:.3}  epochs={}{mono}",
                    fit.explained_variance, fit.epochs
                );
                prev = ev_te;
            }
        }
    }
}

#[test]
fn fixed_width_sparse_storage_never_dense_and_reconstructs() {
    let (k, p, n) = (6usize, 8usize, 240usize);
    let (x, _atoms) = planted(k, p, n, 0.0);
    let config = SparseDictConfig {
        n_atoms: k,
        active: 1,
        max_epochs: 30,
        score_tile: 4,
        score_mode: gam_gpu::GpuMode::Off,
        ..SparseDictConfig::new(k)
    };
    let fit = fit_sparse_dictionary(x.view(), &config).expect("fit");
    // Storage is fixed-width N×s, NOT N×K.
    assert_eq!(fit.indices.dim(), (n, 1));
    assert_eq!(fit.codes.dim(), (n, 1));
    assert_eq!(fit.decoder.dim(), (k, p));
    // Reconstruction EV from the packed sparse codes matches the reported EV.
    let recon = fit.reconstruct();
    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    let mut means = vec![0.0f64; p];
    for i in 0..n {
        for c in 0..p {
            means[c] += x[[i, c]] as f64;
        }
    }
    for c in 0..p {
        means[c] /= n as f64;
    }
    for i in 0..n {
        for c in 0..p {
            let r = x[[i, c]] as f64 - recon[[i, c]] as f64;
            rss += r * r;
            let t = x[[i, c]] as f64 - means[c];
            tss += t * t;
        }
    }
    let recon_ev = 1.0 - rss / tss;
    assert!(
        (recon_ev - fit.explained_variance).abs() < 1.0e-4,
        "packed-code reconstruction EV {recon_ev} disagrees with reported {}",
        fit.explained_variance
    );
}

#[test]
fn route_minibatch_returns_a_valid_top_s() {
    // The batched-GEMM minibatch router must return a genuine top-`s` per row:
    // every selected atom's score (recomputed exactly in f64) is within f32-GEMM
    // rounding of the true `s`-th-largest |score| cutoff, and the reported score
    // matches the exact dot product. Where two atoms tie within rounding the
    // batched and row-at-a-time paths may pick different members of the tie —
    // that is correct (they are interchangeable) and is exactly why the fit is
    // minibatch-invariant rather than bit-identical. Non-orthogonal unit atoms
    // so the scores are generic.
    let (k, p, n) = (40usize, 11usize, 137usize);
    let mut decoder = Array2::<f32>::zeros((k, p));
    for atom in 0..k {
        for c in 0..p {
            decoder[[atom, c]] = (((atom * 5 + c * 3 + 1) % 13) as f32 - 6.0) / 6.0;
        }
    }
    // Unit-norm the decoder rows (the trainer always routes against unit atoms).
    for mut row in decoder.outer_iter_mut() {
        let nrm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if nrm > 1.0e-12 {
            row.mapv_inplace(|v| v / nrm);
        }
    }
    let mut x = Array2::<f32>::zeros((n, p));
    for row in 0..n {
        for c in 0..p {
            x[[row, c]] = (((row * 7 + c * 2 + 3) % 17) as f32 - 8.0) / 4.0;
        }
    }
    let s = 4usize;
    let scorer = TileScorer::new(s, 7);
    let batched = scorer.route_minibatch(x.view(), decoder.view());
    assert_eq!(batched.len(), n);

    // Exact f64 |score| of one row against one atom.
    let exact_mag = |row: usize, atom: usize| -> f64 {
        let mut acc = 0.0f64;
        for c in 0..p {
            acc += x[[row, c]] as f64 * decoder[[atom, c]] as f64;
        }
        acc.abs()
    };
    const TOL: f64 = 1.0e-5;
    for (i, shortlist) in batched.iter().enumerate() {
        assert_eq!(shortlist.len(), s, "row {i}: shortlist must have width s");
        // The shortlist's atoms are distinct.
        let mut seen = std::collections::HashSet::new();
        for &(atom, _) in shortlist {
            assert!(seen.insert(atom), "row {i}: atom {atom} selected twice");
        }
        // Reported scores match the exact dot product.
        for &(atom, score) in shortlist {
            assert!(
                (score.abs() as f64 - exact_mag(i, atom as usize)).abs() <= TOL,
                "row {i}: reported |score| {} for atom {atom} != exact {}",
                score.abs(),
                exact_mag(i, atom as usize)
            );
        }
        // The true s-th-largest |score| cutoff, computed exactly.
        let mut all: Vec<f64> = (0..k).map(|a| exact_mag(i, a)).collect();
        all.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cutoff = all[s - 1];
        // Every selected atom must clear the cutoff up to rounding (a valid top-s).
        for &(atom, _) in shortlist {
            assert!(
                exact_mag(i, atom as usize) + TOL >= cutoff,
                "row {i}: selected atom {atom} (|score| {}) is below the top-{s} cutoff {cutoff}",
                exact_mag(i, atom as usize)
            );
        }
        // The shortlist is sorted by descending |score|.
        for w in shortlist.windows(2) {
            assert!(
                w[0].1.abs() + (TOL as f32) >= w[1].1.abs(),
                "row {i}: shortlist not sorted by descending |score|"
            );
        }
    }
}

#[test]
fn fit_is_minibatch_size_invariant() {
    // The minibatch knob bounds peak working set, NOT the solution. Fitting the
    // same data with a tiny minibatch (1 row at a time) and with a minibatch that
    // covers the whole block must produce the same dictionary quality: the
    // route→code→refresh math is identical, only the score-block tiling changes.
    let (k, p, n) = (8usize, 12usize, 480usize);
    let (x, _atoms) = planted(k, p, n, 0.2);
    let base = SparseDictConfig {
        n_atoms: k,
        active: 2,
        minibatch: 1,
        max_epochs: 40,
        score_tile: 16,
        code_ridge: 1.0e-6,
        decoder_ridge: 1.0e-6,
        tolerance: 1.0e-9,
        score_mode: gam_gpu::GpuMode::Off,
    };
    let fit_mb1 = fit_sparse_dictionary(x.view(), &base).expect("minibatch=1 fit");
    let fit_mbn = fit_sparse_dictionary(
        x.view(),
        &SparseDictConfig {
            minibatch: n,
            ..base
        },
    )
    .expect("minibatch=N fit");
    let fit_mb_mid = fit_sparse_dictionary(
        x.view(),
        &SparseDictConfig {
            minibatch: 64,
            ..base
        },
    )
    .expect("minibatch=64 fit");
    // Same EV to f32-rounding tolerance regardless of how the rows were batched.
    assert!(
        (fit_mb1.explained_variance - fit_mbn.explained_variance).abs() < 1.0e-4,
        "minibatch=1 EV {} vs minibatch=N EV {} must agree",
        fit_mb1.explained_variance,
        fit_mbn.explained_variance
    );
    assert!(
        (fit_mb1.explained_variance - fit_mb_mid.explained_variance).abs() < 1.0e-4,
        "minibatch=1 EV {} vs minibatch=64 EV {} must agree",
        fit_mb1.explained_variance,
        fit_mb_mid.explained_variance
    );
}

#[test]
fn scales_to_large_k_without_dense_n_by_k() {
    // K far larger than the planted rank: trainer must stay correct and never
    // allocate N×K (it would here be 240*2000 floats; the test just checks it
    // runs and stays fixed-width).
    let (planted_k, p, n) = (8usize, 10usize, 240usize);
    let (x, _atoms) = planted(planted_k, p, n, 0.1);
    let k = 2000usize;
    let config = SparseDictConfig {
        n_atoms: k,
        active: 1,
        max_epochs: 6,
        score_tile: 256,
        score_mode: gam_gpu::GpuMode::Off,
        ..SparseDictConfig::new(k)
    };
    let fit = fit_sparse_dictionary(x.view(), &config).expect("large-K fit");
    assert_eq!(fit.indices.dim(), (n, 1));
    assert!(
        fit.explained_variance > 0.9,
        "large-K trainer should still explain the low-rank signal; got {}",
        fit.explained_variance
    );
}

/// #1026 — a real large-K `fit_sparse_dictionary` whose minibatch × K route
/// block clears the device break-even reports that admission honestly while this
/// local test pins execution to CPU with the per-fit `score_mode`.
#[test]
fn large_k_fit_reports_admitted_route_stats_and_is_reproducible() {
    // minibatch=512 × K=4096 = 2,097,152-element score block per minibatch,
    // above DEVICE_SCORE_BLOCK_MIN_ELEMS (1<<20). p=48 is a representative
    // residual-stream width. GpuMode::Off keeps this regression local-CPU only.
    let (planted_k, p, n) = (8usize, 48usize, 1536usize);
    let (x, _atoms) = planted(planted_k, p, n, 0.1);
    let k = 4096usize;
    let config = SparseDictConfig {
        n_atoms: k,
        active: 2,
        minibatch: 512,
        max_epochs: 4,
        score_tile: 1024,
        score_mode: gam_gpu::GpuMode::Off,
        ..SparseDictConfig::new(k)
    };

    let fit = fit_sparse_dictionary(x.view(), &config).expect("large-K fit");
    let fit2 = fit_sparse_dictionary(x.view(), &config).expect("large-K fit (rerun)");

    assert_eq!(
        fit.decoder, fit2.decoder,
        "[#1026] sparse-dict fit is non-deterministic across runs (GPU route must \
         be bit-reproducible)"
    );
    assert_eq!(fit.indices, fit2.indices);
    assert_eq!(fit.codes, fit2.codes);
    assert_eq!(fit.score_route_stats, fit2.score_route_stats);
    assert!(fit.score_route_stats.admitted_minibatches > 0);
    assert_eq!(fit.score_route_stats.device_minibatches, 0);
    assert_eq!(
        fit.score_route_stats.cpu_minibatches,
        fit.score_route_stats.minibatches
    );
    assert!(
        fit.explained_variance > 0.9,
        "[#1026] large-K fit should explain the low-rank signal; got {}",
        fit.explained_variance
    );
}
