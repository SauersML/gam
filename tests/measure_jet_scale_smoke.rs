//! Measure-jet frame acceptance gate 5 (docs/measure_jet_frame.md §7.5): scale smoke —
//! "fast" as a CI gate, not a vibe. Gates the measure-jet BUILD path (no
//! REML fit): masses O(n·m·d), design O(n·m·d), energy O(m²·d·L). The point
//! is catching an O(n²) regression anywhere in that path, not benchmarking.
//!
//! Row-count note. FarthestPoint center selection is NOT the bottleneck:
//! `select_thin_plate_knots` keeps a maintained min-distance array, so each
//! added center costs one rayon-parallel O(n·d) sweep — O(n·m·d) total,
//! linear in n. What forces the drop from the charter's 10⁶ rows to
//! n = 200_000 is the constraint-transform GEMM inside the build
//! (`raw_design · z`, an (n×m)·(m×(m−1)) product): it is O(n·m²) — already
//! the asymptotically dominant term over the documented O(n·m·d) passes —
//! and CI executes tests at opt-level 0 (no [profile.*] opt override in
//! Cargo.toml; test.yml runs `cargo test --config profile.dev.debug=0`),
//! where the ≈1.8e11-flop product at n = 10⁶ alone takes minutes. At
//! n = 200_000 the whole build sits well inside the bound while an O(n²)
//! row-pairwise regression (≥ 4e10 ops) would still blow it.
//!
//! Memory: the data matrix is built once (200_000 × 8 f64 ≈ 13 MB); the
//! transient peak is the raw n×m representer design plus its constrained
//! copy (≈ 0.5 GB each) — sane for CI.

use std::time::Instant;

use gam::basis::{BasisMetadata, CenterStrategy, MeasureJetBasisSpec, build_measure_jet_basis};
use ndarray::Array2;

const N_ROWS: usize = 200_000;
const N_DIMS: usize = 8;
const N_CENTERS: usize = 300;
/// Generous-but-real wall-clock bound: slow CI machines pass with margin,
/// an O(n²) regression in the build path cannot.
const WALL_CLOCK_BOUND_SECS: f64 = 120.0;
/// Deterministic pseudo-noise amplitude off the filament backbone.
const JITTER: f64 = 1e-3;

/// SplitMix64 finalizer mapped to [0, 1): deterministic per-index jitter
/// with no RNG state and no seed dependence between rows.
fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Three deterministic 1-D strands (trig curves of the row index) embedded
/// in 8-D, separated by per-strand offset vectors, with hashed sub-resolution
/// jitter so the filament has honest thickness.
fn filament_coordinate(row: usize, dim: usize) -> f64 {
    let strand = (row % 3) as f64;
    let t = (row / 3) as f64 / (N_ROWS / 3) as f64;
    let k = dim as f64;
    let freq = 1.0 + 0.45 * k + 0.6 * strand;
    let phase = 0.8 * strand + 0.37 * k;
    let amp = 1.0 / (1.0 + 0.25 * k);
    let drift = (0.6 - 0.15 * k) * (strand - 1.0);
    let backbone = amp * (std::f64::consts::TAU * 0.35 * freq * t + phase).sin()
        + drift * t
        + 1.7 * strand * (0.9 * k).cos();
    let jitter = JITTER * (2.0 * hashed_unit((row * N_DIMS + dim) as u64) - 1.0);
    backbone + jitter
}

#[test]
fn measure_jet_build_scale_smoke_200k_rows() {
    let data = Array2::<f64>::from_shape_fn((N_ROWS, N_DIMS), |(i, k)| filament_coordinate(i, k));

    // Multiscale (per-scale spectral split) is an explicit opt-in (#1116);
    // this smoke exercises the spectral build path, so it opts in.
    let spec = MeasureJetBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: N_CENTERS,
        },
        multiscale: true,
        ..Default::default()
    };

    let started = Instant::now();
    // (a) the build must succeed at filament scale.
    let built = build_measure_jet_basis(data.view(), &spec)
        .expect("measure-jet build must succeed on a 200k-row 8-D filament");
    let elapsed = started.elapsed().as_secs_f64();
    let rate = N_ROWS as f64 / elapsed;
    println!(
        "measure-jet scale smoke: n={N_ROWS} d={N_DIMS} m={N_CENTERS} \
         build={elapsed:.2}s rate={rate:.0} rows/s"
    );

    // (b) wall-clock gate: an O(n²) pass over the rows cannot fit in here.
    assert!(
        elapsed < WALL_CLOCK_BOUND_SECS,
        "measure-jet build took {elapsed:.1}s for n={N_ROWS} \
         (bound {WALL_CLOCK_BOUND_SECS}s) — an O(n²) regression in the build path?"
    );

    // (c) per-level (spectral) mode under the multiscale opt-in (#1116): one
    // penalty candidate per band scale plus the double-penalty ridge.
    let BasisMetadata::MeasureJet {
        eps_band, order_s, ..
    } = &built.metadata
    else {
        panic!("measure-jet build must carry MeasureJet metadata");
    };
    assert_eq!(
        *order_s, 0.0,
        "default order keeps the auto (spectral) sentinel"
    );
    assert!(
        !eps_band.is_empty(),
        "realized scale band must be non-empty"
    );
    assert_eq!(
        built.penalties.len(),
        eps_band.len() + 1,
        "per-level candidate count must be band length ({}) + 1 ridge",
        eps_band.len()
    );

    // Shape sanity: every row designed against the m−1 sum-to-zero columns.
    assert_eq!(built.design.nrows(), N_ROWS);
    assert_eq!(built.design.ncols(), N_CENTERS - 1);
}
