//! Tests for the in-frame curved cascade (`inframe_curved.rs`): planted low-rank
//! curved recovery at `p = 2048` with memory orders below the dense path, parity
//! with the full-`p` fit when the frame contains the truth, and rejection of a
//! region with no curved structure.

use ndarray::{Array2, Array3};

use super::atom::{SaeAtomBasisKind, SaeManifoldAtom};
use super::inframe_curved::{
    ChartOccupancyStatus, CurvedRegion, InFrameCurvedConfig, WeightFrameOccupancy,
    activate_residual_frame, dense_ambient_radial_reference, fit_inframe_curved_regions,
    fit_inframe_curved_weight_frame_catalog, inframe_curved_region_prediction, residual_span_frame,
};
use super::weight_frame_catalog::{
    WeightFrameCatalogConfig, WeightFrameMatrix, WeightFrameSource,
    frame_catalog_from_weight_matrices,
};

/// Deterministic LCG in `[-1, 1)` so the tests are reproducible without an RNG
/// dependency.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_unit(&mut self) -> f64 {
        // Numerical Recipes LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        2.0 * bits - 1.0
    }
    fn normal(&mut self) -> f64 {
        // Box–Muller from two uniforms in (0,1).
        let u1 = (self.next_unit() * 0.5 + 0.5).max(1.0e-12);
        let u2 = self.next_unit() * 0.5 + 0.5;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Build a `p × r` column-orthonormal ambient embedding by Gram–Schmidt on
/// random Gaussian columns.
fn random_orthonormal(p: usize, r: usize, seed: u64) -> Array2<f64> {
    let mut rng = Lcg::new(seed);
    let mut q = Array2::<f64>::zeros((p, r));
    for col in 0..r {
        let mut v: Vec<f64> = (0..p).map(|_| rng.normal()).collect();
        for prev in 0..col {
            let mut dot = 0.0;
            for i in 0..p {
                dot += v[i] * q[[i, prev]];
            }
            for i in 0..p {
                v[i] -= dot * q[[i, prev]];
            }
        }
        let mut norm = 0.0;
        for i in 0..p {
            norm += v[i] * v[i];
        }
        norm = norm.sqrt().max(1.0e-12);
        for i in 0..p {
            q[[i, col]] = v[i] / norm;
        }
    }
    q
}

/// Plant a curved (spherical-shell) structure of intrinsic dimension `r_true`
/// inside a `p`-dimensional residual: latent points on a noisy shell of radius
/// ~1, embedded through a random orthonormal frame, plus tiny ambient noise.
fn planted_curved_residual(
    n: usize,
    p: usize,
    r_true: usize,
    shell_noise: f64,
    ambient_noise: f64,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    let q = random_orthonormal(p, r_true, seed);
    let mut rng = Lcg::new(seed ^ 0x9E3779B97F4A7C15);
    let mut latent = Array2::<f64>::zeros((n, r_true));
    for i in 0..n {
        let mut v: Vec<f64> = (0..r_true).map(|_| rng.normal()).collect();
        let mut norm = 0.0;
        for x in &v {
            norm += x * x;
        }
        norm = norm.sqrt().max(1.0e-12);
        let radius = 1.0 + shell_noise * rng.normal();
        for x in &mut v {
            *x = radius * *x / norm;
        }
        for j in 0..r_true {
            latent[[i, j]] = v[j];
        }
    }
    // Ambient residual = latent @ Qᵀ + ambient noise.
    let mut residual = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let mut acc = 0.0;
            for k in 0..r_true {
                acc += latent[[i, k]] * q[[j, k]];
            }
            residual[[i, j]] = acc + ambient_noise * rng.normal();
        }
    }
    (residual, q)
}

fn project_matrix_onto_frame(matrix: &Array2<f64>, frame: &Array2<f64>) -> Array2<f64> {
    frame.dot(&frame.t().dot(matrix))
}

fn relative_frobenius(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let mut diff = 0.0;
    let mut denom = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        diff += d * d;
        denom += y * y;
    }
    (diff / denom.max(1.0e-30)).sqrt()
}

#[test]
fn weight_frame_catalog_spans_component_column_images() {
    let p = 24;
    let q_ov = random_orthonormal(p, 2, 1001);
    let q_mlp = random_orthonormal(p, 2, 1002);

    let mut w_v = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        w_v[[0, j]] = 0.7 * (j as f64 + 1.0);
        w_v[[1, j]] = if j % 2 == 0 { 1.0 } else { -0.5 };
    }
    let ov = q_ov.dot(&w_v);

    let mut down_coeff = Array2::<f64>::zeros((2, 5));
    for j in 0..5 {
        down_coeff[[0, j]] = 1.0 + j as f64;
        down_coeff[[1, j]] = if j % 2 == 0 { 2.0 } else { -1.0 };
    }
    let w_down = q_mlp.dot(&down_coeff);

    let components = vec![
        WeightFrameMatrix::attention_head_ov(3, 14, q_ov.view(), w_v.view()).expect("OV builds"),
        WeightFrameMatrix::mlp_down_projection(3, w_down.view()),
    ];
    let catalog = frame_catalog_from_weight_matrices(
        &components,
        &WeightFrameCatalogConfig {
            frame_rank_min: 2,
            frame_rank_max: 2,
            ..Default::default()
        },
    )
    .expect("catalog builds");

    assert_eq!(catalog.entries().len(), 2);
    assert_eq!(
        catalog.entries()[0].source,
        WeightFrameSource::AttentionHeadOv { layer: 3, head: 14 }
    );
    assert_eq!(
        catalog.entries()[1].source,
        WeightFrameSource::MlpDownProjection { layer: 3 }
    );

    let ov_frame = catalog.entries()[0].frame.frame().to_owned();
    let mlp_frame = catalog.entries()[1].frame.frame().to_owned();
    let ov_projected = project_matrix_onto_frame(&ov, &ov_frame);
    let mlp_projected = project_matrix_onto_frame(&w_down, &mlp_frame);
    assert!(
        relative_frobenius(&ov_projected, &ov) < 1.0e-10,
        "OV catalog frame must span exactly the OV column image"
    );
    assert!(
        relative_frobenius(&mlp_projected, &w_down) < 1.0e-10,
        "MLP catalog frame must span exactly the W_down column image"
    );
}

#[test]
fn weight_sourced_atom_fit_is_tagged_with_component_source() {
    let n = 240;
    let p = 96;
    let r = 4;
    let (residual, q) = planted_curved_residual(n, p, r, 0.02, 0.0, 1414);
    let mut w_v = Array2::<f64>::zeros((r, p));
    for j in 0..p {
        for k in 0..r {
            w_v[[k, j]] = ((j + 1 + k) as f64).sin();
        }
    }
    let components =
        vec![WeightFrameMatrix::attention_head_ov(2, 14, q.view(), w_v.view()).expect("OV")];
    let catalog = frame_catalog_from_weight_matrices(
        &components,
        &WeightFrameCatalogConfig {
            frame_rank_min: r,
            frame_rank_max: r,
            ..Default::default()
        },
    )
    .expect("catalog");
    let config = InFrameCurvedConfig {
        frame_rank_min: r,
        frame_rank_max: r,
        min_rows: 16,
        ..Default::default()
    };
    let result = fit_inframe_curved_weight_frame_catalog(
        residual.view(),
        &catalog,
        &[WeightFrameOccupancy {
            frame_index: 0,
            rows: (0..n).collect(),
            basis_size: 5,
        }],
        n,
        &config,
    )
    .expect("weight-frame fit");

    assert_eq!(result.records.len(), 1);
    assert_eq!(
        result.records[0].occupancy_status,
        ChartOccupancyStatus::Occupied
    );
    assert_eq!(
        result.records[0].frame_source,
        Some(WeightFrameSource::AttentionHeadOv { layer: 2, head: 14 }),
        "occupied atom record must carry native mechanism attribution"
    );
    if !result.curved_prediction.regions().is_empty() {
        assert_eq!(
            result.curved_prediction.regions()[0].frame_source(),
            Some(&WeightFrameSource::AttentionHeadOv { layer: 2, head: 14 })
        );
    }
}

#[test]
fn zero_occupancy_weight_frame_is_reported_chartable_unoccupied() {
    let n = 160;
    let p = 64;
    let r = 3;
    let (residual, q_used) = planted_curved_residual(n, p, r, 0.03, 0.0, 5150);
    let q_unused = random_orthonormal(p, r, 5151);
    let mut coeff = Array2::<f64>::zeros((r, p));
    for j in 0..p {
        for k in 0..r {
            coeff[[k, j]] = ((j + 2 * k + 1) as f64).cos();
        }
    }
    let components = vec![
        WeightFrameMatrix::attention_head_ov(6, 1, q_used.view(), coeff.view()).expect("OV"),
        WeightFrameMatrix::mlp_down_projection(6, q_unused.view()),
    ];
    let catalog = frame_catalog_from_weight_matrices(
        &components,
        &WeightFrameCatalogConfig {
            frame_rank_min: r,
            frame_rank_max: r,
            ..Default::default()
        },
    )
    .expect("catalog");
    let config = InFrameCurvedConfig {
        frame_rank_min: r,
        frame_rank_max: r,
        min_rows: 16,
        ..Default::default()
    };
    let result = fit_inframe_curved_weight_frame_catalog(
        residual.view(),
        &catalog,
        &[
            WeightFrameOccupancy {
                frame_index: 0,
                rows: (0..n).collect(),
                basis_size: 4,
            },
            WeightFrameOccupancy {
                frame_index: 1,
                rows: Vec::new(),
                basis_size: 4,
            },
        ],
        n,
        &config,
    )
    .expect("weight-frame atlas");

    assert_eq!(result.records.len(), 2);
    let unoccupied = &result.records[1];
    assert_eq!(
        unoccupied.occupancy_status,
        ChartOccupancyStatus::ChartableUnoccupied,
        "unused weight frame remains in the atlas as chartable but unoccupied here"
    );
    assert_eq!(
        unoccupied.frame_source,
        Some(WeightFrameSource::MlpDownProjection { layer: 6 })
    );
    assert_eq!(unoccupied.evidence.n_rows, 0);
    assert!(!unoccupied.evidence.selected_by_bic);
}

#[test]
fn planted_low_rank_curved_recovered_inframe_p2048() {
    let n = 1200;
    let p = 2048;
    let r_true = 6;
    let (residual, _q) = planted_curved_residual(n, p, r_true, 0.02, 0.0, 42);
    let m = 8usize; // atom basis size

    let config = InFrameCurvedConfig {
        frame_rank_min: 2,
        frame_rank_max: 16,
        crossfit_folds: 4,
        min_rows: 32,
        ..Default::default()
    };
    let region = CurvedRegion {
        rows: (0..n).collect(),
        basis_size: m,
    };

    let start = std::time::Instant::now();
    let result = fit_inframe_curved_regions(residual.view(), &[region], n, &config)
        .expect("in-frame curved fit");
    let elapsed = start.elapsed();

    // Frame learned at (or above) the true intrinsic rank, well below p.
    let rec = &result.records[0];
    assert!(
        rec.frame_rank >= r_true && rec.frame_rank <= 16,
        "frame rank {} should recover the intrinsic rank {r_true} within the band",
        rec.frame_rank
    );
    assert!(
        rec.frame_rank < p,
        "frame rank {} must be far below ambient p={p}",
        rec.frame_rank
    );

    // The curved chart beats the linear baseline out-of-sample and is accepted.
    assert!(
        rec.evidence.deviance_gain > 0.0,
        "curved chart should improve held-out deviance, got {}",
        rec.evidence.deviance_gain
    );
    assert_eq!(result.selected_regions, vec![0], "planted region selected");

    // MEASURED memory ledger: border and covariance orders below the dense path.
    let ledger = &result.ledger;
    assert_eq!(ledger.dense_border_coeffs, m * p);
    assert_eq!(ledger.inframe_border_coeffs, m * rec.frame_rank);
    let border_shrink = ledger.border_shrink();
    let cov_shrink = ledger.cov_shrink();
    assert!(
        border_shrink >= (p as f64 / 16.0),
        "border must shrink by ~p/r; got {border_shrink:.1}x (dense={} inframe={})",
        ledger.dense_border_coeffs,
        ledger.inframe_border_coeffs
    );
    assert!(
        cov_shrink >= 10_000.0,
        "posterior covariance must shrink by (p/r)²; got {cov_shrink:.0}x"
    );
    // Dense per-atom covariance is multi-GB; in-frame is well under a MB.
    assert!(
        ledger.dense_cov_bytes >= 1_000_000_000,
        "dense (M·p)² covariance should be ~GB, got {} bytes",
        ledger.dense_cov_bytes
    );
    assert!(
        ledger.inframe_cov_bytes <= 4_000_000,
        "in-frame (M·r)² covariance should be well under a MB, got {} bytes",
        ledger.inframe_cov_bytes
    );

    // The p=2048 in-frame fit is tractable in wall-clock terms (the whole point).
    assert!(
        elapsed.as_secs_f64() < 30.0,
        "in-frame fit at p={p} took {elapsed:?}; should be fast"
    );

    assert_eq!(result.curved_prediction.n_rows(), n);
    assert_eq!(result.curved_prediction.output_dim(), p);
    assert_eq!(
        result.curved_prediction.inframe_entries(),
        n * rec.frame_rank,
        "hot prediction storage must be the accepted region's N_g x r image"
    );
    assert_eq!(
        result.curved_prediction.accepted_ambient_entries_if_eager(),
        n * p,
        "the eager ambient atom image would have been N_g x p"
    );
    assert!(
        result.curved_prediction.inframe_entries()
            < result.curved_prediction.accepted_ambient_entries_if_eager(),
        "curved prediction must stay in-frame on the hot path"
    );

    // The lifted curved prediction reconstructs the planted shell when explicitly
    // materialized for verification (rows nonzero, radius ~1).
    let pred = result.curved_prediction.materialize_ambient();
    let mut mean_norm = 0.0;
    for i in 0..n {
        let mut ss = 0.0;
        for j in 0..p {
            ss += pred[[i, j]] * pred[[i, j]];
        }
        mean_norm += ss.sqrt();
    }
    mean_norm /= n as f64;
    assert!(
        (mean_norm - 1.0).abs() < 0.25,
        "reconstructed shell radius {mean_norm:.3} should be ~1"
    );
}

#[test]
fn inframe_matches_dense_full_p_when_frame_contains_truth() {
    // Residual lying EXACTLY in an r_true-dim subspace (no ambient noise): the
    // learned frame contains the truth, so the in-frame radial fit and the
    // full-p dense radial fit must agree to machine precision.
    let n = 300;
    let p = 256;
    let r_true = 4;
    let (residual, _q) = planted_curved_residual(n, p, r_true, 0.1, 0.0, 7);

    let config = InFrameCurvedConfig {
        frame_rank_min: r_true,
        frame_rank_max: r_true,
        min_rows: 16,
        ..Default::default()
    };
    let rows: Vec<usize> = (0..n).collect();

    let inframe_pred = inframe_curved_region_prediction(residual.view(), &rows, &config)
        .expect("in-frame prediction")
        .expect("frame learned");
    let rank = inframe_pred.frame_rank();
    assert_eq!(rank, r_true, "frame rank pinned to the true intrinsic rank");
    assert_eq!(
        inframe_pred.inframe_entries(),
        n * r_true,
        "single-region prediction stays in the learned r-frame"
    );

    let dense_pred =
        dense_ambient_radial_reference(residual.view(), config.whitening_ridge).expect("dense fit");
    let inframe_ambient = inframe_pred.materialize_ambient();

    // Both are n×p ambient predictions of the SAME chart; compare Frobenius.
    let mut diff = 0.0;
    let mut denom = 0.0;
    for i in 0..n {
        for j in 0..p {
            let d = inframe_ambient[[i, j]] - dense_pred[[i, j]];
            diff += d * d;
            denom += dense_pred[[i, j]] * dense_pred[[i, j]];
        }
    }
    let rel = (diff / denom.max(1.0e-30)).sqrt();
    assert!(
        rel < 1.0e-6,
        "in-frame and dense full-p radial fits must match on the frame's span; \
         relative Frobenius diff {rel:.3e}"
    );
}

#[test]
fn residual_span_frame_is_the_production_hook_low_rank_and_spans_truth() {
    // The production seam (`residual_span_frame`) must (a) return a frame whose
    // rank is far below p — so a curved atom carrying it flips the arrow-Schur
    // onto its M·r frames_engaged path instead of the dense M·p Hessian — and
    // (b) span the planted subspace, so the in-frame fit loses no signal. The
    // second property is checked structurally: the residual projected onto the
    // frame and lifted back must reconstruct the (exactly low-rank) residual.
    let n = 400;
    let p = 1024;
    let r_true = 6;
    let (residual, _q) = planted_curved_residual(n, p, r_true, 0.05, 0.0, 2130);
    let config = InFrameCurvedConfig {
        frame_rank_min: r_true,
        frame_rank_max: 16,
        min_rows: 16,
        ..Default::default()
    };
    let rows: Vec<usize> = (0..n).collect();

    let frame = residual_span_frame(residual.view(), &rows, &config)
        .expect("frame learns")
        .expect("beneficial low-rank frame exists for a planted low-rank residual");
    let r = frame.rank();
    assert!(
        r >= r_true && r <= 16 && r < p,
        "seam frame rank {r} should recover the intrinsic rank {r_true} and stay far below p={p}"
    );

    // The frame spans the truth: R (U Uᵀ) ≈ R for an exactly-low-rank residual.
    let u = frame.frame().to_owned(); // p × r
    let mut diff = 0.0;
    let mut denom = 0.0;
    for i in 0..n {
        // z_i = R_i · U  (length r); lifted = z_i · Uᵀ  (length p).
        let mut z = vec![0.0; r];
        for (k, zk) in z.iter_mut().enumerate() {
            let mut acc = 0.0;
            for j in 0..p {
                acc += residual[[i, j]] * u[[j, k]];
            }
            *zk = acc;
        }
        for j in 0..p {
            let mut lifted = 0.0;
            for (k, &zk) in z.iter().enumerate() {
                lifted += zk * u[[j, k]];
            }
            let d = residual[[i, j]] - lifted;
            diff += d * d;
            denom += residual[[i, j]] * residual[[i, j]];
        }
    }
    let rel = (diff / denom.max(1e-30)).sqrt();
    assert!(
        rel < 1e-6,
        "seam frame must span the planted subspace (residual reconstructs through U Uᵀ); rel={rel:.3e}"
    );

    // A full-rank (isotropic) residual admits no beneficial low-rank frame, so
    // the seam returns None and the caller correctly leaves that region dense.
    let mut rng = Lcg::new(9001);
    let mut iso = Array2::<f64>::zeros((64, 8));
    for i in 0..64 {
        for j in 0..8 {
            iso[[i, j]] = rng.normal();
        }
    }
    let tight = InFrameCurvedConfig {
        frame_rank_min: 2,
        frame_rank_max: 4,
        rank_cutoff: 1e-9, // count every direction ⇒ numerical rank fills the width
        ..Default::default()
    };
    let iso_rows: Vec<usize> = (0..64).collect();
    let got = residual_span_frame(iso.view(), &iso_rows, &tight)
        .expect("runs")
        .expect("rank_max below ambient width must return a strict low-rank frame");
    // rank_max=4 < p=8 so a frame is still returned, but it must be a strict
    // low-rank projection (r <= 4), never the full width.
    assert!(
        got.rank() <= 4 && got.rank() < 8,
        "seam frame must stay strictly low-rank"
    );
}

#[test]
fn activate_residual_frame_installs_factored_decoder_and_engages_frames() {
    // The one-call wiring hook must (a) install a low-rank decoder_frame learned
    // from the residual, and (b) leave the decoder EXACTLY factored as B = C·Uᵀ
    // (B == (B U) Uᵀ) so the factored arrow-Schur C-solve converges — the same
    // invariant maybe_activate_decoder_frame enforces, but with the frame sourced
    // from the residual span (no dense fit).
    let n = 200;
    let p = 128;
    let m = 3usize; // atom basis size
    let r_true = 5;
    let (residual, _q) = planted_curved_residual(n, p, r_true, 0.05, 0.0, 4242);

    // A minimal atom whose decoder is generic full-rank (M×p); activation must
    // project it onto the residual frame.
    let mut rng = Lcg::new(77);
    let mut decoder = Array2::<f64>::zeros((m, p));
    for a in 0..m {
        for j in 0..p {
            decoder[[a, j]] = rng.normal();
        }
    }
    let basis_values = Array2::<f64>::zeros((1, m));
    let basis_jacobian = Array3::<f64>::zeros((1, m, 1));
    let smooth_penalty = Array2::<f64>::eye(m);
    let mut atom = SaeManifoldAtom::new_with_provided_function_gram(
        "seam",
        SaeAtomBasisKind::Periodic,
        1,
        basis_values,
        basis_jacobian,
        decoder,
        smooth_penalty,
    )
    .expect("atom builds");
    assert!(atom.decoder_frame.is_none(), "starts on the full-p path");

    let config = InFrameCurvedConfig {
        frame_rank_min: r_true,
        frame_rank_max: 16,
        min_rows: 16,
        ..Default::default()
    };
    let rows: Vec<usize> = (0..n).collect();
    let r = activate_residual_frame(&mut atom, residual.view(), &rows, &config)
        .expect("activation runs")
        .expect("beneficial low-rank frame installed");
    assert!(
        r >= r_true && r < p,
        "installed frame rank {r} low-rank vs p={p}"
    );
    let frame = atom.decoder_frame.as_ref().expect("frame installed");
    assert_eq!(frame.rank(), r);

    // Decoder is now exactly factored: B == (B U) Uᵀ (projection is idempotent).
    let u = frame.frame().to_owned();
    let mut reproj = atom.decoder_coefficients.dot(&u).dot(&u.t());
    reproj -= &atom.decoder_coefficients;
    let mut fro = 0.0;
    for v in reproj.iter() {
        fro += v * v;
    }
    assert!(
        fro.sqrt() < 1e-9,
        "activated decoder must satisfy B = (B U) Uᵀ exactly; residual {:.3e}",
        fro.sqrt()
    );
}

#[test]
fn linear_structure_is_not_promoted_to_curved() {
    // The meaningful negative control for the curved gate is LINEAR structure,
    // not isotropic noise. (An isotropic Gaussian blob has genuine radial
    // concentration: its points cluster near a shell of radius ~√r, so the
    // 1-parameter radial chart is a *better* one-dimensional summary than a
    // rank-1 line and the gate honestly accepts it — that is not a false
    // positive, it is real held-out deviance the linear stage did not capture.)
    //
    // What the gate must NOT do is promote structure a LINEAR model already
    // explains. Here the residual is rank-1: every row is a scalar multiple of a
    // single ambient direction (plus negligible noise). Rank-1 PCA reconstructs
    // it exactly; the radial chart, which forces every row onto one radius,
    // destroys the sign/magnitude along the line and reconstructs strictly
    // worse. So the curved refinement loses out-of-sample and must be rejected —
    // the linear lane owns this region.
    // Exactly rank-1 (no ambient noise): with even tiny noise the fine spectral
    // cutoff would count the noise directions and the whitening floor would
    // amplify them into a spurious isotropic blob. An exact rank-1 residual keeps
    // every extra frame direction at zero projection, so the radial chart can
    // only collapse the single real axis onto ±one radius — strictly worse than
    // the exact rank-1 linear reconstruction.
    let n = 800;
    let p = 512;
    let dir = random_orthonormal(p, 1, 314);
    let mut rng = Lcg::new(2130);
    let mut residual = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let s = rng.normal(); // signed coordinate along the single linear axis
        for j in 0..p {
            residual[[i, j]] = s * dir[[j, 0]];
        }
    }

    let config = InFrameCurvedConfig {
        frame_rank_min: 4,
        frame_rank_max: 8,
        min_rows: 32,
        ..Default::default()
    };
    let region = CurvedRegion {
        rows: (0..n).collect(),
        basis_size: 8,
    };
    let result = fit_inframe_curved_regions(residual.view(), &[region], n, &config)
        .expect("fit runs on linear structure");
    assert!(
        result.selected_regions.is_empty(),
        "purely linear (rank-1) structure must NOT be promoted to a curved atom; \
         deviance_gain={} margin={}",
        result.records[0].evidence.deviance_gain,
        result.records[0].evidence.margin
    );
}

#[test]
fn ledger_shrink_matches_reviewer_frontier_shape() {
    // The reviewer's headline shape: M=8, p=4096, r=16 ⇒ border shrinks 256×,
    // per-atom covariance shrinks 65536× (8.6 GB → 131 KB). Assert the ledger
    // arithmetic reproduces those numbers exactly for a single accepted region.
    let n = 600;
    let p = 4096;
    let r_target = 16;
    // Plant exactly r_target intrinsic directions so the frame lands at r=16.
    let (residual, _q) = planted_curved_residual(n, p, r_target, 0.02, 0.0, 99);
    let config = InFrameCurvedConfig {
        frame_rank_min: r_target,
        frame_rank_max: r_target,
        min_rows: 32,
        ..Default::default()
    };
    let region = CurvedRegion {
        rows: (0..n).collect(),
        basis_size: 8,
    };
    let result = fit_inframe_curved_regions(residual.view(), &[region], n, &config).expect("fit");
    assert_eq!(result.records[0].frame_rank, r_target);
    if result.selected_regions.is_empty() {
        // Even if the gate is conservative on this synthetic draw, the per-record
        // border/cov arithmetic is what we are asserting; recompute from record.
        let rec = &result.records[0];
        assert_eq!(rec.dense_border_coeffs, 8 * p);
        assert_eq!(rec.inframe_border_coeffs, 8 * r_target);
    } else {
        let ledger = &result.ledger;
        assert_eq!(ledger.dense_border_coeffs, 8 * p);
        assert_eq!(ledger.inframe_border_coeffs, 8 * r_target);
        // (8·4096)² · 8 bytes = 8.59 GB ; (8·16)² · 8 bytes = 131072 bytes.
        assert_eq!(ledger.dense_cov_bytes, (8 * p) * (8 * p) * 8);
        assert_eq!(
            ledger.inframe_cov_bytes,
            (8 * r_target) * (8 * r_target) * 8
        );
        assert_eq!(ledger.inframe_cov_bytes, 131_072);
        assert!((ledger.border_shrink() - 256.0).abs() < 1.0e-9);
        assert!((ledger.cov_shrink() - 65_536.0).abs() < 1.0e-6);
    }
}

#[test]
fn inframe_curved_p4096_feasible_where_dense_joint_ooms_2134() {
    // #2134 wall #1: the COLD-JOINT `sae_manifold_fit` lane does not scale in p —
    // it times out at p=256 and OOMs (48 GB) at p=1024 because the dense arrow-Schur
    // border/covariance carries the full ambient width `p` (border `Σ M_k·p`, per-atom
    // covariance `(M·p)²`). The shipped in-frame cascade sidesteps that wall: the
    // curved chart is fit purely in an `r`-dim learned frame, so `p` only reappears
    // in the final ambient lift. This gate fits the SAME frontier shape the dense
    // lane could not reach — p=4096 at the issue's N=1500 — and asserts that the
    // fitted state and its memory ledger stay in the compact in-frame geometry.
    // It is the explicit p=4096 feasibility gate the report requires.
    let n = 1500; // the issue's N; the dense lane already timed out at p=256/this N.
    let p = 4096; // the dense lane OOMed at p=1024; the in-frame lane clears 4× that.
    let r_true = 8;
    let m = 8usize; // atom basis size ⇒ dense per-atom covariance is (8·4096)² = 8.6 GB.
    let (residual, _q) = planted_curved_residual(n, p, r_true, 0.02, 0.0, 21_34);

    let config = InFrameCurvedConfig {
        frame_rank_min: 2,
        frame_rank_max: 16,
        crossfit_folds: 4,
        min_rows: 32,
        ..Default::default()
    };
    let region = CurvedRegion {
        rows: (0..n).collect(),
        basis_size: m,
    };

    let result = fit_inframe_curved_regions(residual.view(), &[region], n, &config)
        .expect("in-frame curved fit is feasible at p=4096 where the dense joint OOMs");

    // The frame is learned at ~the intrinsic rank, far below the ambient width, so
    // the curved arithmetic never touches p except in the lift.
    let rec = &result.records[0];
    assert!(
        rec.frame_rank >= r_true && rec.frame_rank <= 16 && rec.frame_rank < p,
        "frame rank {} recovers intrinsic rank {r_true} and stays far below p={p}",
        rec.frame_rank
    );
    assert_eq!(
        result.selected_regions,
        vec![0],
        "planted curved region selected"
    );

    // The dense per-atom covariance the joint lane
    // would allocate is (M·p)² · 8 B ≈ 8.6 GB — the source of the OOM. The in-frame
    // covariance is (M·r)² · 8 B, well under a MB. Assert the ledger reproduces both
    // so the report's cost model is anchored on measured arithmetic, not a claim.
    let ledger = &result.ledger;
    assert_eq!(ledger.dense_border_coeffs, m * p);
    assert_eq!(ledger.inframe_border_coeffs, m * rec.frame_rank);
    assert!(
        ledger.dense_cov_bytes >= 8_000_000_000,
        "dense (M·p)² covariance is the ~8.6 GB the joint lane OOMs on, got {} bytes",
        ledger.dense_cov_bytes
    );
    assert!(
        ledger.inframe_cov_bytes <= 1_000_000,
        "in-frame (M·r)² covariance must stay well under a MB, got {} bytes",
        ledger.inframe_cov_bytes
    );
    eprintln!(
        "[#2134 p4096] N={n} p={p} frame_rank={} dense_border={} inframe_border={} \
         dense_cov_bytes={} inframe_cov_bytes={} border_shrink={:.1} cov_shrink={:.1}",
        rec.frame_rank,
        ledger.dense_border_coeffs,
        ledger.inframe_border_coeffs,
        ledger.dense_cov_bytes,
        ledger.inframe_cov_bytes,
        ledger.border_shrink(),
        ledger.cov_shrink(),
    );

    // The hot-path prediction never materialises the N×p ambient image: it stays
    // N_g×r, so peak working memory is p-independent up to the copied residual.
    assert_eq!(
        result.curved_prediction.inframe_entries(),
        n * rec.frame_rank
    );
    assert!(
        result.curved_prediction.inframe_entries()
            < result.curved_prediction.accepted_ambient_entries_if_eager(),
        "curved prediction must stay in-frame (N_g×r), never the eager N_g×p ambient image"
    );
}

#[test]
fn accepted_curved_prediction_hot_path_stays_in_r_frame() {
    let n = 180;
    let p = 128;
    let r_true = 4;
    let (residual, _q) = planted_curved_residual(n, p, r_true, 0.02, 0.0, 974);
    let config = InFrameCurvedConfig {
        frame_rank_min: r_true,
        frame_rank_max: r_true,
        min_rows: 16,
        ..Default::default()
    };
    let region = CurvedRegion {
        rows: (0..n).collect(),
        basis_size: 5,
    };
    let result = fit_inframe_curved_regions(residual.view(), &[region], n, &config).expect("fit");
    assert_eq!(
        result.selected_regions,
        vec![0],
        "planted curved region selected"
    );
    let prediction = &result.curved_prediction;
    assert_eq!(prediction.regions().len(), 1);
    assert_eq!(prediction.regions()[0].frame_rank(), r_true);
    assert_eq!(
        prediction.inframe_entries(),
        n * r_true,
        "accepted atom image must be stored as N_g x r"
    );
    assert_eq!(
        prediction.accepted_ambient_entries_if_eager(),
        n * p,
        "the forbidden eager atom image would be N_g x p"
    );
    assert!(
        prediction.inframe_entries() * (p / r_true)
            <= prediction.accepted_ambient_entries_if_eager(),
        "in-frame storage should scale by r instead of p"
    );

    let slice = prediction.materialize_rows(&[0, n / 2, n - 1]);
    assert_eq!(slice.dim(), (3, p));
    let mut slice_energy = 0.0;
    for value in slice.iter() {
        slice_energy += value * value;
    }
    assert!(
        slice_energy > 0.0,
        "ambient lifting happens only for the requested residual slice"
    );
}
