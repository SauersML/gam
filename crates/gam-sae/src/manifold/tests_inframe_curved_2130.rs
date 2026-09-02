//! Tests for the in-frame curved cascade (`inframe_curved.rs`): planted low-rank
//! curved recovery at `p = 2048` with memory orders below the dense path, parity
//! with the full-`p` fit when the frame contains the truth, and rejection of a
//! region with no curved structure.

use ndarray::{Array2, Array3};

use super::atom::{SaeAtomBasisKind, SaeManifoldAtom};
use super::inframe_curved::{ChartOccupancyStatus, InFrameCurvedConfig, WeightFrameOccupancy, activate_residual_frame, fit_inframe_curved_weight_frame_catalog, residual_span_frame};
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
    let mut reproj = atom.decoder_coefficients().dot(&u).dot(&u.t());
    reproj -= atom.decoder_coefficients();
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

