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
    sorted.sort_by(|a, b| {
        b.partial_cmp(a)
            .expect("eigenvalues of a finite covariance are finite, so the order is total")
    });
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
        .route_minibatch_with_mode(rows.view(), decoder.view(), gam_gpu::GpuPolicy::Required)
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
        gam_gpu::GpuPolicy::Off,
    )
    .expect("explicit-mode transform");
    assert_eq!(transform.indices.dim(), (rows.nrows(), 3));
    assert_eq!(transform.codes.dim(), (rows.nrows(), 3));
    assert_eq!(transform.score_route_stats.minibatches, 1);
    assert_eq!(transform.score_route_stats.cpu_minibatches, 1);
    assert_eq!(transform.score_route_stats.device_minibatches, 0);
    assert_eq!(transform.score_route_stats.device_dtoh_bytes, 0);
    assert_eq!(
        transform.score_route_stats.unfused_score_dtoh_bytes_avoided,
        0
    );
}

#[test]
fn sparse_fit_default_auto_uses_cpu_below_device_floor() {
    let (k, p, n) = (8usize, 10usize, 64usize);
    let (x, _atoms) = planted(k, p, n, 0.1);
    let config = SparseDictConfig {
        n_atoms: k,
        active: 1,
        minibatch: 16,
        max_epochs: 1,
        score_tile: 8,
        ..SparseDictConfig::new(k)
    };

    let fit = fit_sparse_dictionary(x.view(), &config)
        .expect("default Auto mode must not require a subfloor CUDA route");

    assert_eq!(config.score_mode, gam_gpu::GpuPolicy::Auto);
    assert_eq!(fit.score_route_stats.minibatches, 8);
    assert_eq!(fit.score_route_stats.cpu_minibatches, 8);
    assert_eq!(fit.score_route_stats.device_minibatches, 0);
    assert_eq!(fit.score_route_stats.admitted_minibatches, 0);
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
        score_mode: gam_gpu::GpuPolicy::Off,
    };
    let fit = fit_sparse_dictionary(x.view(), &config).expect("fit");
    assert_eq!(fit.score_route_stats.minibatches, 8);
    assert_eq!(fit.score_route_stats.cpu_minibatches, 8);
    assert_eq!(fit.score_route_stats.device_minibatches, 0);
    assert_eq!(fit.score_route_stats.admitted_minibatches, 0);
    assert_eq!(fit.score_route_stats.device_dtoh_bytes, 0);
    assert_eq!(fit.score_route_stats.unfused_score_dtoh_bytes_avoided, 0);
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
        score_mode: gam_gpu::GpuPolicy::Off,
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

/// #2275 trichotomy on the `fit_sparse_dictionary` trainer path: at `K` far above
/// the intrinsic rank the objective (EV) plateaus at ~1 while the discrete top-s
/// routing keeps churning (spurious support directions rotate freely in the
/// equivalent-optima manifold), so the routing fixed-point residual legitimately
/// cannot close. The trainer must return that objective-converged iterate as a
/// **best-effort OPEN certificate** (`certified = false`, routing residual honestly
/// above tolerance) rather than collapsing it to a hard `InnerNonConvergence` — the
/// same contract inc1 restored on `fit_block_sparse_dictionary`. NO tolerance is
/// softened: the open residual is reported as-is.
#[test]
fn over_complete_trainer_returns_best_effort_open_certificate_2275() {
    // K=64 atoms in p=16 dims (K >> rank), 2-sparse rows — the over-complete regime
    // whose routing oscillates while EV saturates.
    let (k, p, n) = (64usize, 16usize, 1600usize);
    let (x, _atoms) = planted(k, p, n, 0.35);
    let config = SparseDictConfig {
        n_atoms: k,
        active: 2,
        minibatch: 256,
        max_epochs: 60,
        score_tile: 16,
        code_ridge: 1.0e-6,
        decoder_ridge: 1.0e-6,
        tolerance: 1.0e-9,
        score_mode: gam_gpu::GpuPolicy::Off,
    };
    let fit = fit_sparse_dictionary(x.view(), &config)
        .expect("#2275: over-complete trainer fit must RETURN best-effort, not error");

    // Objective converged: EV at its achievable plateau.
    assert!(
        fit.explained_variance > 0.9,
        "over-complete EV should saturate the planted structure; got {}",
        fit.explained_variance
    );
    // Best-effort OPEN, not certified: the absolute routing fixed point did not close.
    assert!(
        !fit.convergence.certified,
        "K >> rank fit must carry an OPEN (best-effort) certificate; got certified=true \
         (routing_residual={}, tol={})",
        fit.convergence.routing_residual, fit.convergence.routing_tolerance
    );
    // No tolerance softening: the openness is reported honestly, above the SAME tol.
    assert!(
        fit.convergence.routing_residual > fit.convergence.routing_tolerance,
        "an open certificate must record routing_residual above tolerance; got {} <= {}",
        fit.convergence.routing_residual,
        fit.convergence.routing_tolerance
    );
    assert!(
        fit.convergence.inner_ev_residual.is_finite(),
        "the plateaued objective residual must be recorded (finite); got {}",
        fit.convergence.inner_ev_residual
    );
}

/// `N` distinct unit directions spread over the sphere in `P` dimensions, from a
/// deterministic integer hash (no RNG, no float seeds). Unlike [`planted`] — whose
/// rows collapse onto `rank` distinct directions, so any `K ≥ rank` dictionary
/// interpolates them exactly — every row here needs its own direction, so a
/// dictionary with `K < N` leaves residual on every row no matter how it routes.
fn spread_directions(n: usize, p: usize) -> Array2<f32> {
    let mut x = Array2::<f32>::zeros((n, p));
    for row in 0..n {
        let mut norm2 = 0.0f32;
        for col in 0..p {
            // The `row·col` cross term keeps the per-row pattern from being a
            // pure arithmetic progression, so directions stay distinct well past
            // the modulus rather than repeating every 1009 rows.
            let hash = (row * 131 + col * 71 + row * col * 17) % 1009;
            let value = (hash as f32) / 504.0 - 1.0;
            x[[row, col]] = value;
            norm2 += value * value;
        }
        let norm = norm2.sqrt();
        if norm > 0.0 {
            for col in 0..p {
                x[[row, col]] /= norm;
            }
        }
    }
    x
}

/// #2400 — the saturated-support birth-swap arm, at the production entry.
///
/// The open arm admits a fit whose final transition still accepted residual-row
/// births, but ONLY when live-support cardinality has stopped setting new highs:
/// those births are replacements on a fixed-cardinality support manifold, not
/// structure the fit is still recruiting. `LiveSupportGrowth`'s own unit test
/// pins the state machine in isolation; this pins that `run` actually reaches
/// that branch and that nothing else can mint a model while births are live.
///
/// The invariant is asserted for every shape and is premise-free: a certified fit
/// or a still-growing support with live births would be a model minted from
/// churning structure, whatever the data. The sweep also requires that at least
/// one shape genuinely reach the OPEN arm, so it cannot degenerate into a set of
/// trivially-certified fits that assert nothing.
///
/// The POSITIVE birth-swap arm is proven separately and deterministically by
/// `churning_births_are_admitted_only_after_the_support_saturates_2400`, which
/// drives `open_round_is_stationary` against a live `LiveSupportGrowth`. That is
/// deliberate: across the regimes below — `K` above and below `N`, `s` from 1 to
/// 8, low-rank and continuum data — the trainer's births always stop before the
/// plateau confirms, because the two conditions fight each other. Births need a
/// substantially unexplained row AND a dead atom, and spare capacity (`K > N`,
/// which produces the dead atoms) is exactly what lets the fit explain every row.
/// The production arm therefore stays exercised by the real-activation lane; the
/// sweep here records the regimes so that stays visible rather than assumed.
#[test]
fn open_fit_with_positive_births_is_certified_only_by_saturated_support_2400() {
    // (K, s, N, P). Rows are `N` distinct directions spread over the sphere (see
    // `spread_directions`) rather than a `planted` mixture, whose `rank` distinct
    // directions every one of these shapes interpolates to EV = 1 in four epochs.
    // The list spans both sides of the capacity frontier: `K < N` (no spare
    // capacity, every atom stays live) and `K > N` with a wide active set (spare
    // capacity, so dead atoms exist and a revived atom can win one of `s` slots).
    let shapes = [
        (96usize, 1usize, 600usize, 12usize),
        (64, 2, 600, 12),
        (200, 1, 900, 16),
        (512, 4, 256, 16),
        (768, 8, 384, 32),
    ];

    let mut observed = String::new();
    let mut open_arms = 0usize;
    for (k, s, n, p) in shapes {
        let x = spread_directions(n, p);
        let config = SparseDictConfig {
            n_atoms: k,
            active: s,
            minibatch: 256,
            max_epochs: 12,
            score_tile: 256,
            tolerance: 1.0e-9,
            score_mode: gam_gpu::GpuPolicy::Off,
            ..SparseDictConfig::new(k)
        };
        let Ok(fit) = fit_sparse_dictionary(x.view(), &config) else {
            // A shape that never confirms a plateau inside its budget is a typed
            // non-convergence, which is the honest outcome and not this test's
            // subject. Record it and move on.
            observed.push_str(&format!("K={k} s={s}: typed non-convergence\n"));
            continue;
        };
        let convergence = fit.convergence;
        observed.push_str(&format!(
            "K={k} s={s}: births={} live_high_water={} saturated={} certified={} \
             epochs={} ev={:.6}\n",
            convergence.accepted_births,
            convergence.live_atom_high_water,
            convergence.support_saturated,
            convergence.certified,
            fit.epochs,
            fit.explained_variance,
        ));

        assert!(
            convergence.live_atom_high_water <= k,
            "live support cannot exceed the dictionary width: {} > {k}",
            convergence.live_atom_high_water
        );
        if !convergence.certified {
            open_arms += 1;
        }
        if convergence.accepted_births > 0 {
            assert!(
                convergence.support_saturated,
                "a fit returned with {} live births must have saturated its support \
                 (K={k}, s={s}); returning one whose support is still growing mints a \
                 model from structure the fit has not finished recruiting",
                convergence.accepted_births
            );
            assert!(
                !convergence.certified,
                "births in the final transition are incompatible with an ABSOLUTE \
                 fixed-point certificate (K={k}, s={s}); only the open arm admits them"
            );
        }
    }

    assert!(
        open_arms > 0,
        "every shape certified an absolute fixed point, so the sweep never reached \
         the open arm whose invariant it is asserting; observed:\n{observed}"
    );
}

/// #2275 no-weakening guard: best-effort is NOT handed out on a whim. Genuine
/// non-convergence — the objective plateau never CONFIRMED within the epoch budget
/// — must still surface the typed `InnerNonConvergence` error. With a budget below
/// the sustained-plateau horizon the same over-complete fit (which can never certify
/// its routing) has no confirmed plateau to return, so it must error, exactly as
/// before this change.
#[test]
fn over_complete_trainer_still_errors_without_confirmed_plateau_2275() {
    let (k, p, n) = (64usize, 16usize, 1600usize);
    let (x, _atoms) = planted(k, p, n, 0.35);
    let config = SparseDictConfig {
        n_atoms: k,
        active: 2,
        minibatch: 256,
        // Below the sustained-plateau confirmation horizon: too short to certify OR
        // to confirm a best-effort plateau, so this is genuine non-convergence.
        max_epochs: 3,
        score_tile: 16,
        code_ridge: 1.0e-6,
        decoder_ridge: 1.0e-6,
        tolerance: 1.0e-9,
        score_mode: gam_gpu::GpuPolicy::Off,
    };
    let err = fit_sparse_dictionary(x.view(), &config)
        .expect_err("an unconfirmed plateau must remain a typed non-convergence error");
    assert!(
        err.to_string().contains("did not converge"),
        "expected the typed InnerNonConvergence, got: {err}"
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
        score_mode: gam_gpu::GpuPolicy::Off,
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
        score_mode: gam_gpu::GpuPolicy::Off,
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
        score_mode: gam_gpu::GpuPolicy::Off,
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
    // residual-stream width. GpuPolicy::Off keeps this regression local-CPU only.
    let (planted_k, p, n) = (8usize, 48usize, 1536usize);
    let (x, _atoms) = planted(planted_k, p, n, 0.1);
    let k = 4096usize;
    let config = SparseDictConfig {
        n_atoms: k,
        active: 2,
        minibatch: 512,
        max_epochs: 4,
        score_tile: 1024,
        score_mode: gam_gpu::GpuPolicy::Off,
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

/// The host score route must be invariant to minibatch/chunk granularity: each
/// routed shortlist and its active-set codes depend only on their own row, so
/// splitting the rows into many parallel minibatch chunks (the multi-core host
/// path) must return the same routing as routing the whole block at once —
/// exact-equal atom shortlists, and codes equal to f32 tolerance (the per-row
/// s×s solve is independent of how the rows are grouped). This pins the parallel
/// `route_and_code_all` host fast path (the fix for the single-threaded
/// `matrixmultiply` score wall) against a single-block route.
#[test]
fn host_route_is_invariant_to_minibatch_chunking() {
    use super::update::route_and_code_all;

    // Tiny shape (K=64, N=512, P=32), K < N so the scorer runs the real GEMM
    // route rather than the K>N wrap. GpuPolicy::Off pins the host path.
    let (x, decoder) = planted(64, 32, 512, 0.15);
    let s = 4usize;
    let tile = 13usize; // deliberately not a divisor of K, to exercise ragged tiles
    let code_ridge = 1.0e-6f32;
    let scorer = TileScorer::new(s, tile);

    // Whole block in one route (probe block covers all N; no parallel chunks).
    let single = route_and_code_all(
        x.view(),
        decoder.view(),
        &scorer,
        s,
        code_ridge,
        x.nrows(),
        gam_gpu::GpuPolicy::Off,
        None,
    )
    .expect("single-block host route");

    // Many small parallel chunks (probe block of 7, then chunked remainder).
    let chunked = route_and_code_all(
        x.view(),
        decoder.view(),
        &scorer,
        s,
        code_ridge,
        7,
        gam_gpu::GpuPolicy::Off,
        None,
    )
    .expect("chunked host route");

    assert_eq!(single.len(), x.nrows());
    assert_eq!(single.len(), chunked.len());
    for (row, (a, b)) in single.iter().zip(chunked.iter()).enumerate() {
        assert_eq!(
            a.indices, b.indices,
            "row {row}: chunking changed the routed atom shortlist"
        );
        assert_eq!(a.codes.len(), b.codes.len());
        for (j, (&ca, &cb)) in a.codes.iter().zip(b.codes.iter()).enumerate() {
            assert!(
                (ca - cb).abs() <= 1.0e-6,
                "row {row} code {j}: chunking changed the code ({ca} vs {cb})"
            );
        }
    }
}
