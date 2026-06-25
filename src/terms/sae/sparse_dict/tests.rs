use super::scoring::{TileScorer, top_s_online};
use super::{SparseDictConfig, fit_sparse_dictionary};
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
    use crate::faer_ndarray::FaerEigh;
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
            x[[row, c]] = scale * atoms[[primary, c]] + second_share * scale * atoms[[secondary, c]];
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
    use crate::faer_ndarray::FaerEigh;
    let (evals, _) = cov.eigh(faer::Side::Lower).expect("pca eig");
    // eigh returns ascending; sum of top-`rank` over total.
    let total: f64 = evals.iter().sum();
    let mut sorted: Vec<f64> = evals.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top: f64 = sorted.iter().take(rank).sum();
    if total <= 1.0e-24 { 1.0 } else { top / total }
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
    brute.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap().then(x.0.cmp(&y.0)));
    let scorer = TileScorer::new(4, 7);
    let tiled = scorer.route_row(row.view(), decoder.view());
    assert_eq!(tiled.len(), 4);
    for j in 0..4 {
        assert_eq!(tiled[j].0, brute[j].0, "tiled top-{j} disagrees with brute force");
    }
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
fn fixed_width_sparse_storage_never_dense_and_reconstructs() {
    let (k, p, n) = (6usize, 8usize, 240usize);
    let (x, _atoms) = planted(k, p, n, 0.0);
    let config = SparseDictConfig {
        n_atoms: k,
        active: 1,
        max_epochs: 30,
        score_tile: 4,
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
