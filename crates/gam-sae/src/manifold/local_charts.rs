//! Atlas-first local-chart primitive for manifold discovery (#2280).
//!
//! The intrinsic-metric seeder ([`super::intrinsic_seed`]) produces ONE GLOBAL
//! Landmark-Isomap embedding of the ambient rows, and the topology readout
//! ([`super::chart_atlas`] / [`crate::inference::atlas_nerve`]) labels that global
//! embedding. Neither is the atlas-first primitive: a manifold is not a single
//! chart but a COLLECTION of overlapping local charts, each an injective map from
//! a neighborhood into `R^d`, glued by transition maps on their overlaps. This
//! module builds exactly that object — a [`LocalAtlas`] — as a pure, deterministic
//! construction with per-chart injectivity checks and observed transition
//! geometry.
//!
//! The construction is the classical local-PCA atlas:
//!
//!   1. deterministic farthest-point CENTERS over the ambient rows, greedy in a
//!      metric normalized by each center's own local sample spacing (see
//!      `occupancy_normalized_centers`), so the patches tile the manifold with
//!      sublinearly many charts and each Voronoi cell holds about the mean
//!      occupancy that the patch budget is a multiple of;
//!   2. one PATCH per center — its nearest ambient rows that stay connected to the
//!      center through the atlas's neighbourhood graph, at most `patch_size` of
//!      them — sized so neighboring patches OVERLAP (controlled by
//!      [`LocalAtlasConfig`]). The neighborhood is the LARGEST prefix of that
//!      connected distance order (see `connected_distance_order`) that yields a
//!      certified chart: a local-PCA frame is a
//!      TANGENT plane only while the patch stays inside the local curvature scale,
//!      so a patch that outgrows it is shrunk (dropping its farthest row) until it
//!      certifies, rather than being handed a frame that is not tangent;
//!   3. one CHART per patch — local PCA of the centered neighborhood (SVD of the
//!      `m × p` centered block), whose leading `d` right singular vectors are an
//!      orthonormal frame and whose chart map is the injective projection
//!      `x ↦ Fᵀ(x − μ)`. The frame is put in a CANONICAL SIGN GAUGE (each axis's
//!      largest-magnitude component is made positive), since a singular vector is
//!      only defined up to sign. Each chart carries a [`ChartCertificate`]: a rank
//!      gate (the `d`-th captured singular value clears a floor) and an injectivity
//!      gate (no two neighborhood rows collapse to the same chart coordinate). A
//!      patch that cannot certify at any admissible size is rejected with a typed
//!      [`LocalChartError`];
//!   4. one TRANSITION per overlapping patch pair. Its observed orientation is
//!      the fitted chart Jacobian's determinant: on the overlap the chart change is
//!      `c_to = F_toᵀ(μ_from − μ_to) + (F_toᵀ F_from) c_from + O(curvature)`, so the
//!      handedness relation of the two charts is `sign = sgn det(F_toᵀ F_from)` —
//!      a well-conditioned frame quantity (`|det| = ∏ cos θ_k` over the principal
//!      angles), not a fit to the overlap point cloud. The transition also carries
//!      the orthogonal Procrustes map `R ∈ O(d)` best aligning the two charts on
//!      their shared support WITHIN that handedness class, together with its
//!      translation, so `det R = sign` by construction and reflections are recorded
//!      rather than forced to `+1`.
//!
//! # Statistical authority boundary
//!
//! These charts and transitions are fitted to the observed rows. Numerical rank,
//! injectivity, and conditioning checks establish that the fitted geometry is
//! well posed; they do not turn its orientation signs into population claims.
//! Consequently this module exposes an explicitly observed sign-cocycle readout,
//! never an [`crate::inference::atlas_holonomy::AtlasSignedEdge`] or an exact
//! holonomy certificate. A promotable noisy-topology result must instead go
//! through [`crate::inference::atlas_holonomy::AtlasHolonomyCertificate::gaussian_pca`],
//! whose independent pilot/inference rows, population bounds, familywise level,
//! and typed refusals supply the finite-sample authority. For a one-dimensional
//! chart the orthogonal Procrustes factor is still exactly a `±1` scalar, so the
//! observed transition uses the same orientation convention as
//! [`super::chart_atlas::UnitSpeedChartTransition`].
//!
//! # Determinism
//!
//! Fleet law: no RNG anywhere. Centers are farthest-point (first-wins ties),
//! neighborhoods are prefixes of a TOTAL distance order (distance, then row index —
//! so the ordering does not depend on the sort's stability), patch members and
//! shared supports are sorted by row index, chart frames are put in a canonical sign
//! gauge rather than inheriting the solver's arbitrary singular-vector signs, and
//! every eigen/SVD is faer's deterministic solver. No hashing, no parallel reduction,
//! no float-order ambiguity: same input ⇒ bit-identical atlas run-to-run.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use faer::sparse::{SparseColMat, Triplet};
use gam_linalg::faer_ndarray::FaerSvd;
use gam_linalg::sparse_exact::{factorize_sparse_spd, solve_sparse_spd};

use super::intrinsic_seed::{intrinsic_geodesic_embedding_on_graph, intrinsic_knn_graph};
use super::{AtlasOrientability, GraphCompressionKind};

#[cfg(test)]
#[path = "local_chart_recovery_tests.rs"]
mod recovered_transition_tests;

/// The developing map's placement (see `LocalAtlas::developed_coordinates`): each chart's
/// rigid motion `G_a(c) = Q_a c + s_a` into the root chart's frame, and whether each
/// transition is one the spanning tree glued through.
struct DevelopingMap {
    rotation: Vec<Array2<f64>>,
    offset: Vec<Array1<f64>>,
    in_tree: Vec<bool>,
}

/// A non-tree transition's holonomy in the developed frame, `h(x) = M x + translation`,
/// and whether `M` reverses orientation (see `LocalAtlas::holonomy_quotient_coordinates`).
struct DevelopedHolonomy {
    translation: Array1<f64>,
    reversing: bool,
}

/// A non-tree transition's mean seam displacement in the developed frame, and whether its
/// holonomy reverses orientation (see `LocalAtlas::seam_displacements`).
struct SeamDisplacement {
    displacement: Array1<f64>,
    reversing: bool,
}

/// The angular resolution of one chart frame, returned as a SINE, derived from the
/// patch's own captured-variance certificate. Not a tolerance and not a knob.
///
/// The frame is the `d`-plane that minimises the neighborhood's off-plane residual
/// energy, and [`ChartCertificate::captured_variance_fraction`] `= 1 − ε` says the
/// fitted plane leaves exactly `ε` of that energy off-plane. Tilt the plane by an
/// angle `φ`: the tilt carries `sin²φ` of the CAPTURED energy off-plane, so the
/// tilted plane's residual is at least `sin²φ·(1 − ε)` against the fitted plane's
/// `ε`. The neighborhood therefore cannot prefer the fitted plane over any tilt with
///
/// ```text
///     sin²φ · (1 − ε) ≤ ε        ⟺        sin φ ≤ √(ε / (1 − ε)).
/// ```
///
/// That angle is the frame's OWN estimation resolution — the tilt at which this
/// neighborhood stops distinguishing one tangent plane from another. A perfectly
/// flat patch (`ε = 0`) resolves its plane exactly; a patch whose off-plane energy
/// has reached its in-plane energy (`ε ≥ ½`) resolves nothing at all, and the
/// resolution saturates at `1` so that no orientation read off it is admitted.
fn frame_angular_resolution(certificate: &ChartCertificate) -> f64 {
    let captured = certificate.captured_variance_fraction;
    if !captured.is_finite() || captured <= 0.5 {
        return 1.0;
    }
    let captured = captured.min(1.0);
    let epsilon = 1.0 - captured;
    let sine = (epsilon / captured).sqrt();
    if sine.is_finite() { sine.min(1.0) } else { 1.0 }
}

fn distance_order(z: ArrayView2<'_, f64>, center: usize) -> Vec<usize> {
    let mut scored: Vec<(f64, usize)> = (0..z.nrows())
        .map(|row| (sq_distance(z, center, row), row))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, row)| row).collect()
}

/// Rows in the order they join the center's piece of the neighbourhood graph as the
/// ambient ball around the center grows (#2911).
///
/// A plain prefix of the ambient distance order admits every row inside its radius,
/// including rows of another part of the manifold that the embedding folds close by. On
/// a taller swiss roll a patch at a sparsely sampled outer winding reached across the
/// winding gap, and the rows it borrowed from the next winding gave the nerve 1-cycles
/// the sheet does not have. So walk the ambient order and admit a row only once
/// neighbourhood edges among rows already inside the ball join it to the center. A row
/// that enters the ball unjoined waits; when a later row joins it, every waiting row it
/// reaches joins too, in ambient order. Where each row entering the ball already has a
/// neighbour on the center's side, as on a densely sampled patch that stays on one
/// sheet, this is exactly the ambient order.
///
/// The neighbourhood graph is bridged to one component, so every row joins eventually
/// and the order is a permutation of the rows. Deterministic: the ambient order breaks
/// ties by row index, and a joining wave is sorted by ambient rank.
fn connected_distance_order(
    z: ArrayView2<'_, f64>,
    neighbours: &[Vec<(usize, f64)>],
    center: usize,
) -> Vec<usize> {
    let ambient = distance_order(z, center);
    let n = ambient.len();
    let mut rank = vec![0usize; n];
    for (position, &row) in ambient.iter().enumerate() {
        rank[row] = position;
    }
    let mut in_ball = vec![false; n];
    let mut joined = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for &row in &ambient {
        in_ball[row] = true;
        if !(row == center || neighbours[row].iter().any(|edge| joined[edge.0])) {
            continue;
        }
        joined[row] = true;
        let mut wave = vec![row];
        let mut cursor = 0usize;
        while cursor < wave.len() {
            let current = wave[cursor];
            cursor += 1;
            for edge in &neighbours[current] {
                if in_ball[edge.0] && !joined[edge.0] {
                    joined[edge.0] = true;
                    wave.push(edge.0);
                }
            }
        }
        wave.sort_unstable_by_key(|&member| rank[member]);
        order.extend(wave);
    }
    order
}

/// The two frames' COMBINED angular resolution, as a sine: `sin(φ_a + φ_b)`.
///
/// Principal angles between subspaces are 1-Lipschitz in the geodesic subspace
/// metric, so replacing each fitted plane by any plane inside its own resolution
/// moves every principal angle `θ_k` by at most `φ_a + φ_b`. This is the budget an
/// observed `cos θ_k` must clear before its sign is a fact about the data rather
/// than about the two local PCAs' estimation error.
///
/// Beyond `φ_a + φ_b = π/2` the sine turns back down, which would read as a
/// SMALLER budget for a WORSE pair of charts; past that point no angle is resolved
/// at all, so the budget saturates at `1` instead.
fn combined_frame_resolution(sin_a: f64, sin_b: f64) -> f64 {
    if !(sin_a.is_finite() && sin_b.is_finite()) || sin_a >= 1.0 || sin_b >= 1.0 {
        return 1.0;
    }
    let cos_a = (1.0 - sin_a * sin_a).max(0.0).sqrt();
    let cos_b = (1.0 - sin_b * sin_b).max(0.0).sqrt();
    // `φ_a + φ_b > π/2` ⟺ `sin φ_a > cos φ_b`.
    if sin_a > cos_b {
        return 1.0;
    }
    (sin_a * cos_b + sin_b * cos_a).min(1.0)
}

/// Coverage multiplier for the number of farthest-point patch centers, `⌈c·√n⌉`.
/// `√n` is the standard covering-number scaling for a fixed-radius net; `2`
/// places ~2√n charts over the sample — dense enough that neighboring patches
/// overlap once each is grown to [`LocalAtlasConfig::patch_size`], sparse enough
/// that the atlas stays sublinear in `n`.
const PATCH_COUNT_COVERAGE_MULTIPLIER: f64 = 2.0;

/// Floor on the overlap multiplier for the default patch size, `⌈c·n/patch_count⌉`.
/// `n/count` is the average Voronoi-cell occupancy of one center; growing each
/// patch to `c` times that occupancy forces neighboring cells to share rows, which
/// is what produces the overlaps a transition cocycle needs. `3` gives a robust
/// overlap — over-determining both the chart's local PCA and every overlap's
/// Procrustes — without the patches degenerating into the whole sample.
const PATCH_SIZE_OVERLAP_FLOOR: f64 = 3.0;

/// The mean cover multiplicity a `d`-dimensional atlas needs, `max(3, 2^d)`.
///
/// The fitting floor above is not the only requirement. For the NERVE of the cover
/// to carry the manifold's homology (#2280), every `d`-simplex of the centers'
/// Delaunay complex must be witnessed by a row shared by all its patches, and that
/// requires each patch to reach the vertices of its neighbors' cells — roughly
/// twice its own cell radius. Doubling a radius multiplies the covered volume by
/// `2^d`, so the cover's mean multiplicity must be at least `2^d`. Both are floors
/// on the same quantity, so the larger wins: `d = 1 → 3` (fitting binds),
/// `d = 2 → 4`, `d = 3 → 8`.
///
/// Measured on a flat lattice: at mean multiplicity 3 the nerve of the sheet
/// carries two spurious 1-cycles and `χ = −1`; at 4 it is exactly a disk
/// (`b₁ = 0`, `b₂ = 0`, `χ = 1`).
fn patch_size_overlap_multiplier(intrinsic_dim: usize) -> f64 {
    let doubling = 2.0_f64.powi(i32::try_from(intrinsic_dim.max(1)).unwrap_or(i32::MAX));
    PATCH_SIZE_OVERLAP_FLOOR.max(doubling)
}

/// Farthest-point patch centers in a metric normalized by each center's local
/// sample spacing, so that every Voronoi cell holds about `occupancy_rank` rows.
///
/// [`LocalAtlasConfig::balanced`] denominates the patch budget in ROWS, a multiple
/// of the mean cell occupancy `n / count`, and `patch_size_overlap_multiplier`
/// derives that multiple from each patch reaching twice its own cell radius. Both
/// are statements about a cell holding the MEAN occupancy. A plain farthest-point
/// net equalizes cell AREA in the ambient metric instead, so on a non-uniformly
/// sampled manifold the densely sampled cells hold several times the mean, a patch
/// there reaches barely past its own cell, and the nerve loses intersections that
/// neighboring patches genuinely share (#2280). On `swiss_roll(80, 16)`, whose
/// along-roll spacing grows as `√(1 + t²)`, the four inner-tip centers carrying the
/// spurious `b₁ = 1` class owned 33–63 rows against a mean of 17.8, and their
/// patches reached 0.90–1.47 cell radii rather than 2.
///
/// The repair is to the net, not to the budget. Each chosen center measures squared
/// distances in units of its own squared distance to its `occupancy_rank`-th nearest
/// row, and the greedy step takes the row farthest from every center in those
/// units. A center in a dense region is small in its own units and one in a sparse
/// region is large, so centers crowd where rows crowd and each cell holds about
/// `occupancy_rank` rows — the premise the row budget was derived under. Growing the
/// budget of crowded cells instead is not equivalent: it pushes dense patches past
/// small features (a spherical band's narrow rims) that the fixed budget keeps them
/// inside.
///
/// Deterministic like the intrinsic seeder's farthest-point landmarks: row 0 first,
/// first-wins ties, and a stop once every remaining row coincides with a center. A
/// center with at least `occupancy_rank` coincident rows has no spacing to normalize
/// by, so it covers exactly the rows it coincides with.
fn occupancy_normalized_centers(
    z: ArrayView2<'_, f64>,
    count: usize,
    occupancy_rank: usize,
) -> Vec<usize> {
    let n = z.nrows();
    if n == 0 {
        return Vec::new();
    }
    let target = count.max(1).min(n);
    let rank = occupancy_rank.min(n - 1);
    let mut normalized_nearest = vec![f64::INFINITY; n];
    let mut chosen: Vec<usize> = Vec::with_capacity(target);
    let mut next = 0usize;
    loop {
        chosen.push(next);
        let distances: Vec<f64> = (0..n).map(|row| sq_distance(z, next, row)).collect();
        let mut ordered = distances.clone();
        ordered.select_nth_unstable_by(rank, f64::total_cmp);
        let spacing = ordered[rank];
        for (row, &distance) in distances.iter().enumerate() {
            let normalized = if distance <= 0.0 {
                0.0
            } else if spacing > 0.0 {
                distance / spacing
            } else {
                continue;
            };
            if normalized < normalized_nearest[row] {
                normalized_nearest[row] = normalized;
            }
        }
        if chosen.len() == target {
            break;
        }
        let mut best = 0usize;
        let mut best_distance = -1.0;
        for (row, &value) in normalized_nearest.iter().enumerate() {
            if value > best_distance {
                best_distance = value;
                best = row;
            }
        }
        if best_distance <= 0.0 {
            break;
        }
        next = best;
    }
    chosen
}

/// Minimum fraction of the ambient rows that certified charts must cover for the
/// atlas to be built at all. On real, noisy activations a handful of centers can
/// land on unchartable neighborhoods (a locally rank-deficient blob, a sampling
/// hole) and are dropped individually rather than aborting the whole atlas; but a
/// sub-atlas whose surviving charts cover only a minority of the sample no longer
/// describes the data it was asked to chart.
///
/// Half is the principled cut, not a tunable knob: at exactly one half the charted
/// and omitted row sets are the same size, so below it the charted rows are the
/// strict minority and any downstream seeding built on this atlas would be fit to a
/// non-representative subsample — silently biasing the model toward whichever region
/// happened to certify. A higher bar (e.g. 0.7) would reject usable atlases that
/// merely shed a few boundary centers; a lower one (e.g. 0.3) would bless an atlas
/// fit to a minority of the data. This is a covering property of the certified
/// charts, not a statistical confidence statement.
const MIN_ATLAS_ROW_COVERAGE: f64 = 0.5;

/// Construction parameters for a [`LocalAtlas`].
///
/// The defaults ([`LocalAtlasConfig::balanced`]) are derived from `(n, d)` by the
/// same principled covering-number rules the intrinsic seeder uses; the explicit
/// fields let a caller (or a fixture) pin an exact tiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalAtlasConfig {
    /// Target chart dimension `d` (the local-PCA rank).
    pub intrinsic_dim: usize,
    /// Number of patch centers, placed farthest-point in the occupancy-normalized
    /// metric so each cell holds about `n / patch_count` rows.
    pub patch_count: usize,
    /// Upper bound on the number of nearest rows in each patch (including its
    /// center). A patch that cannot certify a tangent chart at this size is shrunk
    /// toward `2(d + 1)`, the over-determination floor, so the realized
    /// [`LocalPatch::members`] length may be smaller.
    pub patch_size: usize,
    /// Minimum shared rows for two patches to register a transition. Must be at
    /// least `d + 1` so the `d`-dimensional Procrustes alignment is
    /// over-determined; the default uses `d + 2`.
    pub min_overlap: usize,
}

impl LocalAtlasConfig {
    /// Principled defaults for `n` rows and chart dimension `d`: `⌈2√n⌉` centers,
    /// each grown to `⌈max(3, 2^d)·n/count⌉` (floored at `2(d+1)` so a chart's PCA
    /// and every overlap's Procrustes are over-determined), with
    /// `min_overlap = d + 2`. See `patch_size_overlap_multiplier` for why the
    /// overlap multiplier depends on `d`.
    #[must_use]
    pub fn balanced(n_points: usize, intrinsic_dim: usize) -> Self {
        let d = intrinsic_dim.max(1);
        let n = n_points.max(1);
        let patch_count = ((PATCH_COUNT_COVERAGE_MULTIPLIER * (n as f64).sqrt()).ceil() as usize)
            .max(d + 2)
            .min(n);
        let occupancy = (n as f64 / patch_count as f64).max(1.0);
        let patch_size = ((patch_size_overlap_multiplier(d) * occupancy).ceil() as usize)
            .max(2 * (d + 1))
            .min(n);
        Self {
            intrinsic_dim,
            patch_count,
            patch_size,
            min_overlap: d + 2,
        }
    }
}

/// Why a patch failed to yield a certified injective `d`-chart, or why the atlas
/// could not be built.
#[derive(Clone, Debug, PartialEq)]
pub enum LocalChartError {
    /// The ambient block is empty.
    EmptyInput,
    /// Fewer rows than one patch needs.
    InsufficientRows { have: usize, need: usize },
    /// A non-finite ambient coordinate.
    NonFiniteAmbient { row: usize, col: usize, value: f64 },
    /// The chart dimension exceeds what any neighborhood could span.
    IntrinsicDimTooLarge {
        intrinsic_dim: usize,
        ambient_dim: usize,
    },
    /// The `d`-th captured singular value did not clear the rank floor at ANY
    /// admissible neighborhood size: the neighborhood spans fewer than `d`
    /// directions however far it is shrunk.
    DegeneratePatch {
        center: usize,
        intrinsic_dim: usize,
        smallest_captured_singular: f64,
        leading_singular: f64,
    },
    /// The chart projection maps two neighborhood rows the ambient separates to
    /// coordinates inside their rounding band of each other — it is not injective on
    /// its own support — and stayed non-injective down to the smallest admissible
    /// neighborhood. The distances are the collapsed pair's.
    NonInjectiveChart {
        center: usize,
        projected_sq_distance: f64,
        ambient_sq_distance: f64,
    },
    /// The SVD backing a chart or a transition failed to converge.
    SvdFailure { center: usize, detail: String },
    /// The independent intrinsic realization needed to audit the cover failed.
    IntrinsicMetricFailure { detail: String },
    /// Individually chartable centers were dropped (see [`LocalAtlas::rejected_centers`]),
    /// but the certified charts that remain cover fewer than `MIN_ATLAS_ROW_COVERAGE`
    /// of the rows — the surviving sub-atlas describes only a minority of the sample,
    /// so the build refuses rather than returning a partial atlas.
    AtlasCoverageTooLow {
        /// Number of centers that yielded a certified chart.
        certified: usize,
        /// Number of centers attempted.
        requested: usize,
        /// Rows lying in at least one certified chart.
        covered_rows: usize,
        /// Total ambient rows.
        total_rows: usize,
        /// The fraction floor that was not met.
        min_row_coverage: f64,
    },
}

impl fmt::Display for LocalChartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "local_charts: ambient block is empty"),
            Self::InsufficientRows { have, need } => write!(
                f,
                "local_charts: need at least {need} rows for one patch, got {have}"
            ),
            Self::NonFiniteAmbient { row, col, value } => write!(
                f,
                "local_charts: ambient Z must be finite; Z[{row}, {col}] = {value}"
            ),
            Self::IntrinsicDimTooLarge {
                intrinsic_dim,
                ambient_dim,
            } => write!(
                f,
                "local_charts: chart dimension {intrinsic_dim} exceeds ambient dimension {ambient_dim}"
            ),
            Self::DegeneratePatch {
                center,
                intrinsic_dim,
                smallest_captured_singular,
                leading_singular,
            } => write!(
                f,
                "local_charts: patch at row {center} does not span {intrinsic_dim} dimensions \
                 (smallest captured singular value {smallest_captured_singular:.3e} vs leading \
                 {leading_singular:.3e})"
            ),
            Self::NonInjectiveChart {
                center,
                projected_sq_distance,
                ambient_sq_distance,
            } => write!(
                f,
                "local_charts: chart at row {center} is not injective on its neighborhood \
                 (a pair at ambient sq distance {ambient_sq_distance:.3e} projects to sq \
                 distance {projected_sq_distance:.3e}, inside its rounding band)"
            ),
            Self::SvdFailure { center, detail } => {
                write!(
                    f,
                    "local_charts: SVD failed for patch at row {center}: {detail}"
                )
            }
            Self::IntrinsicMetricFailure { detail } => {
                write!(f, "local_charts: intrinsic cover audit failed: {detail}")
            }
            Self::AtlasCoverageTooLow {
                certified,
                requested,
                covered_rows,
                total_rows,
                min_row_coverage,
            } => write!(
                f,
                "local_charts: only {certified}/{requested} centers certified, covering \
                 {covered_rows}/{total_rows} rows (below the {min_row_coverage:.2} floor); \
                 the surviving charts describe a minority of the sample"
            ),
        }
    }
}

impl std::error::Error for LocalChartError {}

/// Per-chart injectivity / conditioning certificate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartCertificate {
    /// `σ₁ / σ_d` of the local-PCA block — the chart frame's condition number.
    pub condition: f64,
    /// The leading captured singular value `σ₁`.
    pub leading_singular: f64,
    /// The smallest captured singular value `σ_d` (the rank gate quantity).
    pub smallest_captured_singular: f64,
    /// Fraction of the neighborhood's total variance captured by the `d`-frame,
    /// `Σ_{i≤d} σ_i² / Σ_i σ_i²` — the extrinsic flatness of the patch.
    pub captured_variance_fraction: f64,
    /// The smallest bi-Lipschitz LOWER stretch of the chart map over the neighborhood
    /// pairs the ambient separates beyond their rounding band,
    /// `min ‖c_p − c_q‖ / ‖x_p − x_q‖ ∈ (0, 1]`, and `1` when no pair is separated.
    /// Every such pair keeps its projected distance above its own band, so the chart
    /// is injective on the support; near `1` certifies a near-isometric chart.
    pub min_projection_stretch: f64,
}

/// The combinatorial patch: a center row and the sorted neighborhood it charts.
///
/// `members` is the largest prefix of the center's distance order that certified a
/// tangent chart, so its length is at most [`LocalAtlasConfig::patch_size`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalPatch {
    /// The farthest-point center row of this patch.
    pub center: usize,
    /// The neighborhood rows, sorted ascending (includes `center`).
    pub members: Vec<usize>,
}

/// One injective local `d`-chart: the PCA frame of a patch plus its certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalChart {
    /// The patch center this chart is built around (matches [`LocalPatch::center`]).
    pub center: usize,
    /// Ambient centroid `μ` of the neighborhood, length `p`.
    pub mean: Array1<f64>,
    /// Orthonormal chart frame `F`, shape `(p, d)`: the leading `d` right singular
    /// vectors of the centered neighborhood, each in the canonical sign gauge (its
    /// largest-magnitude component is positive, lowest index winning a tie). The
    /// chart map is `x ↦ Fᵀ(x − μ)`. Singular vectors are only defined up to sign,
    /// so without this gauge the chart's coordinate axes — and every `det` read off
    /// them — would carry an arbitrary solver-chosen orientation.
    pub frame: Array2<f64>,
    /// Captured singular values `σ₁ ≥ … ≥ σ_d`, length `d`.
    pub singular_values: Array1<f64>,
    /// Chart coordinates of the patch members, shape `(m, d)`, row-aligned with
    /// the owning [`LocalPatch::members`].
    pub coords: Array2<f64>,
    /// Injectivity / conditioning certificate.
    pub certificate: ChartCertificate,
}

impl LocalChart {
    /// Apply the injective chart map to an ambient point: `Fᵀ(x − μ)`.
    #[must_use]
    pub fn project(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        let d = self.frame.ncols();
        let mut out = Array1::<f64>::zeros(d);
        for ax in 0..d {
            let mut acc = 0.0;
            for c in 0..self.frame.nrows() {
                acc += self.frame[[c, ax]] * (x[c] - self.mean[c]);
            }
            out[ax] = acc;
        }
        out
    }
}

/// Numerical conditioning of an observed fitted transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionConditioning {
    /// Every principal angle between the two tangent planes is RESOLVABLY below
    /// `π/2` — `σ_min(F_toᵀ F_from)` clears the two frames' combined angular
    /// resolution, so the chart change is a local diffeomorphism and its handedness
    /// is a fact about the data rather than about the local PCAs' estimation error
    /// — AND the shared-support cross-covariance is well conditioned (the alignment
    /// is well posed). This is not a statistical confidence statement about the
    /// population transition.
    WellConditioned,
    /// Either the two tangent planes are orthogonal to within their own estimation
    /// resolution (so no handedness is determined), or the shared support did not
    /// span all `d` chart directions (so the alignment is ambiguous). The edge is
    /// retained as geometry but excluded from the observed sign cocycle.
    Degenerate,
}

/// The chart-to-chart map on an overlap: an orthogonal Procrustes rotation, a
/// translation, and an orientation sign.
///
/// Convention: for a shared ambient point with chart coordinates `c_from` in the
/// `from` chart and `c_to` in the `to` chart, `c_to ≈ R · c_from + t`.
///
/// The orientation is read from the fitted transition Jacobian, `sign =
/// sgn det(F_toᵀ F_from)`, and `R` is the orthogonal Procrustes factor restricted to
/// that handedness class (`U diag(1, …, 1, ±1) Vᵀ`), so `det R = sign` and a genuine
/// reflection is recorded rather than forced to `+1`. Reading the sign off the
/// frames rather than off the fitted `U Vᵀ` matters: on a small or elongated shared
/// support the free Procrustes factor is reflection-ambiguous (a reflection fits the
/// overlap just as well as a rotation, at identical residual), and one such spurious
/// flip anywhere in the atlas would slander an orientable manifold as non-orientable.
/// The frame determinant has no such ambiguity — it is a property of the two tangent
/// planes, not of the handful of points they happen to share.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartTransition {
    /// Source patch index (`< to_patch`, the canonical undirected orientation).
    pub from_patch: usize,
    /// Target patch index.
    pub to_patch: usize,
    /// Running index of this overlap component.
    pub overlap_id: usize,
    /// The shared support rows, sorted ascending.
    pub shared_rows: Vec<usize>,
    /// Orthogonal Procrustes rotation `R`, shape `(d, d)`: `c_to ≈ R c_from + t`,
    /// restricted to the handedness class of `sign` (so `det R = sign`).
    pub rotation: Array2<f64>,
    /// Translation `t`, length `d`.
    pub translation: Array1<f64>,
    /// Orientation sign `sgn det(F_toᵀ F_from) = det R ∈ {±1}`.
    pub sign: i8,
    /// Relative Procrustes residual `‖C_to − R C_from‖_F / ‖C_to‖_F` — how coherent
    /// the two charts are on the overlap (`0` = perfectly co-oriented planes).
    pub residual: f64,
    /// `σ_min(F_toᵀ F_from) = min_k cos θ_k`, the cosine of the LARGEST principal
    /// angle between the two tangent planes. This is the quantity whose crossing of
    /// zero flips [`ChartTransition::sign`].
    pub smallest_principal_cosine: f64,
    /// `sin(φ_to + φ_from)`, the two frames' combined angular resolution derived
    /// from their captured-variance certificates. `sign` is admitted into the
    /// observed cocycle only while `smallest_principal_cosine` exceeds it; the pair
    /// is reported so a consumer can see the margin the verdict rests on rather
    /// than only the boolean.
    pub sign_resolution_budget: f64,
    /// Whether the observed fitted transition is numerically well conditioned.
    pub conditioning: TransitionConditioning,
}

impl ChartTransition {
    /// Apply the transition to a `from`-chart coordinate: `R c + t`.
    #[must_use]
    pub fn apply(&self, coordinate: ArrayView1<'_, f64>) -> Array1<f64> {
        let d = self.rotation.nrows();
        let mut out = Array1::<f64>::zeros(d);
        for i in 0..d {
            let mut acc = self.translation[i];
            for j in 0..d {
                acc += self.rotation[[i, j]] * coordinate[j];
            }
            out[i] = acc;
        }
        out
    }
}

/// A farthest-point center that could not be charted, kept as a diagnostic so a
/// dropped center is recorded rather than silently vanishing.
///
/// One non-certifiable center no longer aborts the whole atlas (real activations
/// have locally rank-deficient neighborhoods); the center is skipped, its typed
/// reason retained here, and the build continues on the remaining centers. The
/// atlas as a whole still refuses honestly — via [`LocalChartError::AtlasCoverageTooLow`]
/// — when too much of the sample is left uncharted.
#[derive(Clone, Debug, PartialEq)]
pub struct RejectedCenter {
    /// The farthest-point center row that failed to certify.
    pub center: usize,
    /// The typed per-center reason it was dropped (a [`LocalChartError::DegeneratePatch`],
    /// [`LocalChartError::NonInjectiveChart`], or [`LocalChartError::SvdFailure`]).
    pub reason: LocalChartError,
}

impl fmt::Display for RejectedCenter {
    /// One user-facing line naming the dropped center and why, so a caller
    /// surfacing the atlas renders a legible diagnostic rather than an opaque row
    /// index — a dropped center should be something the user can see happened.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dropped center at row {}: {}", self.center, self.reason)
    }
}

/// A collection of overlapping injective local charts glued by a signed
/// transition cocycle — the atlas-first manifold primitive (#2280).
#[derive(Clone, Debug, PartialEq)]
pub struct LocalAtlas {
    intrinsic_dim: usize,
    ambient_dim: usize,
    patches: Vec<LocalPatch>,
    charts: Vec<LocalChart>,
    transitions: Vec<ChartTransition>,
    rejected_centers: Vec<RejectedCenter>,
    intrinsic_cover_multiplicity: (usize, f64),
    intrinsic_coordinates: Array2<f64>,
    /// The deterministic kNN graph `build` grew patch membership through, symmetric and
    /// bridged to one component, row-aligned with the rows the atlas was built on.
    neighbourhood: Vec<Vec<(usize, f64)>>,
}

impl LocalAtlas {
    /// Construct the local-chart atlas from ambient rows `z` under `config`.
    ///
    /// Pure, deterministic construction: occupancy-normalized farthest-point
    /// centers, nearest-row patches, local-PCA charts (each certified injective or
    /// rejected with a typed error), and orthogonal Procrustes transitions on every
    /// overlap.
    pub fn build(
        z: ArrayView2<'_, f64>,
        config: LocalAtlasConfig,
    ) -> Result<Self, LocalChartError> {
        let (n, p) = z.dim();
        if n == 0 || p == 0 {
            return Err(LocalChartError::EmptyInput);
        }
        for ((row, col), &value) in z.indexed_iter() {
            if !value.is_finite() {
                return Err(LocalChartError::NonFiniteAmbient { row, col, value });
            }
        }
        let d = config.intrinsic_dim.max(1);
        if d > p {
            return Err(LocalChartError::IntrinsicDimTooLarge {
                intrinsic_dim: d,
                ambient_dim: p,
            });
        }
        let patch_size = config.patch_size.min(n).max(d + 1);
        if n < patch_size {
            return Err(LocalChartError::InsufficientRows {
                have: n,
                need: patch_size,
            });
        }
        let min_overlap = config.min_overlap.max(d + 1);

        // One neighbourhood graph serves the intrinsic realization and patch membership,
        // which grows only through it (see `connected_distance_order`).
        let neighbours = intrinsic_knn_graph(z, d);
        let membership_coords = intrinsic_geodesic_embedding_on_graph(z, d, &neighbours)
            .map_err(|detail| LocalChartError::IntrinsicMetricFailure { detail })?;
        // (1) deterministic farthest-point centers in the occupancy-normalized metric,
        // so every cell holds about the mean occupancy `patch_size` is a multiple of.
        let requested_centers = config.patch_count.max(1).min(n);
        let centers =
            occupancy_normalized_centers(z, requested_centers, n.div_ceil(requested_centers));
        // (2)+(3) one certified local-PCA chart per patch, on the largest
        // neighborhood that certifies (see `certified_neighborhood_chart`). A center
        // whose neighborhood cannot certify at any admissible size is DROPPED with its
        // typed reason rather than aborting the whole atlas — real activations have
        // locally rank-deficient neighborhoods, and one such center should not blind the
        // rest of the manifold. The honest whole-atlas refusal is deferred to the
        // coverage gate below (and, when nothing certified at all, to the per-center
        // reason itself, which is the true story in that case).
        let mut patches: Vec<LocalPatch> = Vec::with_capacity(centers.len());
        let mut charts: Vec<LocalChart> = Vec::with_capacity(centers.len());
        let mut rejected_centers: Vec<RejectedCenter> = Vec::new();
        for &center in &centers {
            match certified_neighborhood_chart(z, &neighbours, center, patch_size, d) {
                Ok((members, chart)) => {
                    patches.push(LocalPatch { center, members });
                    charts.push(chart);
                }
                // Per-center failures (each carries its own `center`): drop this center
                // and keep building. Any other variant is a global precondition already
                // screened above and is propagated unchanged.
                Err(
                    reason @ (LocalChartError::DegeneratePatch { .. }
                    | LocalChartError::NonInjectiveChart { .. }
                    | LocalChartError::SvdFailure { .. }),
                ) => rejected_centers.push(RejectedCenter { center, reason }),
                Err(other) => return Err(other),
            }
        }

        // Honest whole-atlas refusal. When NOTHING certified, the per-center reason is
        // the true diagnosis (e.g. data that is `d`-degenerate everywhere), so surface
        // it directly. When a partial atlas exists but its certified charts cover fewer
        // than `MIN_ATLAS_ROW_COVERAGE` of the rows, the survivors describe a minority of
        // the sample — refuse rather than return a misleadingly partial atlas.
        if charts.is_empty() {
            return Err(rejected_centers
                .into_iter()
                .next()
                .map_or(LocalChartError::EmptyInput, |rejected| rejected.reason));
        }
        let covered_rows = {
            let mut covered: BTreeSet<usize> = BTreeSet::new();
            for patch in &patches {
                covered.extend(patch.members.iter().copied());
            }
            covered.len()
        };
        if (covered_rows as f64) < MIN_ATLAS_ROW_COVERAGE * n as f64 {
            return Err(LocalChartError::AtlasCoverageTooLow {
                certified: charts.len(),
                requested: centers.len(),
                covered_rows,
                total_rows: n,
                min_row_coverage: MIN_ATLAS_ROW_COVERAGE,
            });
        }

        // Independently realize the same cover in intrinsic coordinates: the same
        // budget AND the same net, the ambient cover's own occupancy-normalized net.
        // A fold can make the ambient cover look regular while its intrinsic
        // realization piles many charts onto one region; topology may not be
        // promoted from a cover that fails either coordinate-system certificate. An
        // area-equalizing net here would pile rows up wherever the intrinsic
        // coordinate is sparsely sampled (a 1-D realization of a closed curve is
        // folded, so its density is non-uniform by construction) and refuse honest
        // covers for a property of the net rather than of the data.
        let intrinsic_centers = occupancy_normalized_centers(
            membership_coords.view(),
            requested_centers,
            n.div_ceil(requested_centers),
        );
        let mut intrinsic_counts = vec![0usize; n];
        for center in intrinsic_centers {
            for row in distance_order(membership_coords.view(), center)
                .into_iter()
                .take(patch_size)
            {
                intrinsic_counts[row] += 1;
            }
        }
        let intrinsic_max = intrinsic_counts.iter().copied().max().unwrap_or(0);
        let intrinsic_mean = intrinsic_counts.iter().sum::<usize>() as f64 / n as f64;

        // (4) orthogonal Procrustes transition per overlapping patch pair.
        let mut transitions: Vec<ChartTransition> = Vec::new();
        let mut overlap_id = 0usize;
        for i in 0..patches.len() {
            for j in (i + 1)..patches.len() {
                let shared = sorted_intersection(&patches[i].members, &patches[j].members);
                if shared.len() < min_overlap {
                    continue;
                }
                let transition = build_transition(&charts, &patches, i, j, overlap_id, &shared);
                transitions.push(transition);
                overlap_id += 1;
            }
        }

        Ok(Self {
            intrinsic_dim: d,
            ambient_dim: p,
            patches,
            charts,
            transitions,
            rejected_centers,
            intrinsic_cover_multiplicity: (intrinsic_max, intrinsic_mean),
            intrinsic_coordinates: membership_coords,
            neighbourhood: neighbours,
        })
    }

    /// Chart dimension `d`.
    #[must_use]
    pub fn intrinsic_dim(&self) -> usize {
        self.intrinsic_dim
    }

    /// Ambient dimension `p`.
    #[must_use]
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    /// The combinatorial patches, indexed the same as [`Self::charts`].
    #[must_use]
    pub fn patches(&self) -> &[LocalPatch] {
        &self.patches
    }

    /// The injective local charts.
    #[must_use]
    pub fn charts(&self) -> &[LocalChart] {
        &self.charts
    }

    /// The overlap transitions.
    #[must_use]
    pub fn transitions(&self) -> &[ChartTransition] {
        &self.transitions
    }

    /// Number of charts (= number of patches).
    #[must_use]
    pub fn chart_count(&self) -> usize {
        self.charts.len()
    }

    /// Centers that were dropped because their neighborhood could not certify a
    /// chart at any admissible size. Empty on a clean build; non-empty records
    /// which farthest-point centers the atlas skipped and why, so a caller sees the
    /// dropped structure rather than a silent hole. Ordered by center-iteration
    /// (deterministic).
    #[must_use]
    pub fn rejected_centers(&self) -> &[RejectedCenter] {
        &self.rejected_centers
    }

    pub(super) fn intrinsic_cover_multiplicity(&self) -> (usize, f64) {
        self.intrinsic_cover_multiplicity
    }

    /// The Landmark-Isomap realization of the atlas's rows at chart rank `d` that
    /// `build` audited its cover against, row-aligned with the rows it was built on.
    pub(crate) fn intrinsic_coordinates(&self) -> &Array2<f64> {
        &self.intrinsic_coordinates
    }

    /// The neighbourhood graph patch membership grew through: `neighbourhood()[i]` lists
    /// row `i`'s neighbours with their ambient distances, symmetric and bridged to one
    /// component.
    pub(crate) fn neighbourhood(&self) -> &[Vec<(usize, f64)>] {
        &self.neighbourhood
    }

    /// The developing map (#2280): one `d`-coordinate realization of the rows `z` the
    /// atlas was built on, glued from its local charts along a spanning tree of the
    /// well-conditioned transitions.
    ///
    /// Every chart `a` receives a rigid motion `G_a(c) = Q_a c + s_a` into the frame of
    /// chart `0`, the root, whose `G_0` is the identity. A tree edge carries its
    /// transition `c_to ≈ R c_from + t` across in either direction: `G_to(c) =
    /// G_from(Rᵀ(c − t))` and `G_from(c) = G_to(R c + t)`. The tree is Kruskal's least
    /// total residual (`ChartTransition::residual`, `overlap_id` breaking ties), so the
    /// gluing crosses the overlaps on which two charts agree best. Only well-conditioned
    /// transitions are eligible: a degenerate one carries no resolved handedness, and
    /// composing through a wrongly handed `R` would mirror everything beyond it.
    ///
    /// A row takes its image in its home chart, the patch that contains it with the
    /// nearest center (lowest index on a tie). A row no patch contains takes the nearest
    /// center's chart, whose map `Fᵀ(x − μ)` is defined at every ambient point. The
    /// result is centered, which fixes the translation the root chart leaves free.
    ///
    /// This is a global chart only when the atlas's holonomy is trivial, which is what
    /// the topology readout certifies when it names a contractible manifold. Over a
    /// cover with a non-trivial class the non-tree edges cut seams into it. Refused when
    /// the well-conditioned transitions leave the charts in more than one component,
    /// because then no single chart develops them.
    pub(crate) fn developed_coordinates(
        &self,
        z: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let map = self.developing_map()?;
        let mut coords = self.develop_rows(z, &map)?;
        let mean = coords
            .mean_axis(ndarray::Axis(0))
            .ok_or_else(|| "developed_coordinates: there are no rows to develop".to_string())?;
        coords -= &mean;
        Ok(coords)
    }

    /// The quotient coordinates the atlas's holonomy dictates for a manifold its readout
    /// names (#2906), in the convention the matching seed uses.
    ///
    /// Developing a shared row through the `from` chart of a non-tree transition
    /// `c_to ≈ R c_from + t` and through its `to` chart gives two images related by the
    /// holonomy `h(x) = M x + v`, with `M = Q_to R Q_fromᵀ` and `v = Q_to t + s_to − M s_from`.
    /// Over a flat quotient every holonomy is a deck transformation of the developed chart,
    /// so the quotient is read off them instead of off a principal projection:
    ///
    /// * a circle (chart rank 1): a non-trivial holonomy translates the developed line by a
    ///   multiple of the loop length, and a transition crossing the seam once carries one
    ///   length. The period is the largest orientation-preserving translation, and the
    ///   coordinate is the fraction of that period in `[0, 1)`, the periodic seed's convention.
    /// * a cylinder (chart rank 2): every orientation-preserving non-tree transition moves its
    ///   shared rows, developed through its two charts, by a whole number of loops, counted
    ///   against the longest such displacement. The loop's axis and length are the mean of those
    ///   displacements, each divided by its count. A displacement is measured where the two
    ///   sheets of the development meet, so a holonomy carrying a small rotation about a far
    ///   pivot cannot tilt the axis as its translation at the root chart would. The coordinates
    ///   are the fraction of the loop along that axis and the height across it, centered and
    ///   scaled to unit spread, the convention the cylinder seed uses.
    ///
    /// Both reads project onto a straight line, so they need the loop to develop straight: a
    /// 1-manifold always does, and a cylinder does when its loop is a geodesic of the surface.
    /// A Möbius band's loop does not. On the standard embedding the centerline
    /// `(2 cos u, 2 sin u, 0)` has geodesic curvature `−cos(u/2)/2` against the width ruling,
    /// so its developed tangent turns by `−2 sin(u/2)`, two radians at `u = π`, and the
    /// developed centerline ends `4π·(J₀(2), −H₀(2)) ≈ (2.81, −9.94)` from its start in the
    /// seam tangent's frame. The reversing holonomy reflects across that tangent, and the
    /// centerline's projection onto it runs backwards wherever `sin(u/2) > π/4`, so no
    /// projection onto the glide axis is a loop coordinate. The band (chart rank 2) is read on
    /// the rows' neighbourhood graph instead, through the charts' handedness; see
    /// [`Self::mobius_quotient_coordinates`].
    ///
    /// Refused for any other manifold or chart rank, for a cover with no holonomy of the
    /// class the read needs, and for a degenerate period or height.
    pub(crate) fn holonomy_quotient_coordinates(
        &self,
        z: ArrayView2<'_, f64>,
        manifold: GraphCompressionKind,
    ) -> Result<Array2<f64>, String> {
        let map = self.developing_map()?;
        let developed = self.develop_rows(z, &map)?;
        let holonomies = self.non_tree_holonomies(&map);
        let n = developed.nrows();
        match (manifold, self.intrinsic_dim) {
            (GraphCompressionKind::Circle, 1) => {
                let period = holonomies
                    .iter()
                    .filter(|holonomy| !holonomy.reversing)
                    .map(|holonomy| holonomy.translation[0].abs())
                    .fold(0.0_f64, f64::max);
                if !(period > 0.0 && period.is_finite()) {
                    return Err(format!(
                        "holonomy_quotient_coordinates: no non-tree transition translates the developed line (largest translation {period:.3e})"
                    ));
                }
                let mut coords = Array2::<f64>::zeros((n, 1));
                for row in 0..n {
                    let fraction = developed[[row, 0]] / period;
                    coords[[row, 0]] = fraction - fraction.floor();
                }
                Ok(coords)
            }
            (GraphCompressionKind::Cylinder, 2) => {
                let displacements: Vec<Array1<f64>> = self
                    .seam_displacements(z, &map)
                    .into_iter()
                    .filter(|seam| !seam.reversing)
                    .map(|seam| seam.displacement)
                    .collect();
                let reference = displacements
                    .iter()
                    .max_by(|left, right| left.dot(*left).total_cmp(&right.dot(*right)))
                    .ok_or_else(|| {
                        "holonomy_quotient_coordinates: no orientation-preserving non-tree transition, so no loop translation to read"
                            .to_string()
                    })?;
                let longest_sq = reference.dot(reference);
                if !(longest_sq > 0.0 && longest_sq.is_finite()) {
                    return Err(format!(
                        "holonomy_quotient_coordinates: the loop translation has no length (squared {longest_sq:.3e})"
                    ));
                }
                let mut loop_sum = [0.0_f64; 2];
                let mut crossings = 0usize;
                for displacement in &displacements {
                    let windings = (displacement.dot(reference) / longest_sq).round();
                    if windings == 0.0 {
                        continue;
                    }
                    loop_sum[0] += displacement[0] / windings;
                    loop_sum[1] += displacement[1] / windings;
                    crossings += 1;
                }
                let generator = [
                    loop_sum[0] / crossings as f64,
                    loop_sum[1] / crossings as f64,
                ];
                let period = generator[0].hypot(generator[1]);
                if !(period > 0.0 && period.is_finite()) {
                    return Err(format!(
                        "holonomy_quotient_coordinates: the loop translation has no length ({period:.3e})"
                    ));
                }
                let axis = [generator[0] / period, generator[1] / period];
                let normal = [-axis[1], axis[0]];
                let mut coords = Array2::<f64>::zeros((n, 2));
                for row in 0..n {
                    let (x0, x1) = (developed[[row, 0]], developed[[row, 1]]);
                    let fraction = (x0 * axis[0] + x1 * axis[1]) / period;
                    coords[[row, 0]] = fraction - fraction.floor();
                    coords[[row, 1]] = x0 * normal[0] + x1 * normal[1];
                }
                let count = n.max(1) as f64;
                let mean = coords.column(1).sum() / count;
                let spread = (coords
                    .column(1)
                    .iter()
                    .map(|height| (height - mean) * (height - mean))
                    .sum::<f64>()
                    / count)
                    .sqrt();
                if !(spread > 0.0 && spread.is_finite()) {
                    return Err(format!(
                        "holonomy_quotient_coordinates: the cylinder height is degenerate (spread {spread:.3e})"
                    ));
                }
                for row in 0..n {
                    coords[[row, 1]] = (coords[[row, 1]] - mean) / spread;
                }
                Ok(coords)
            }
            (GraphCompressionKind::MobiusStrip, 2) => {
                self.mobius_quotient_coordinates(z, &map, &developed)
            }
            (other, rank) => Err(format!(
                "holonomy_quotient_coordinates: no quotient read for {other:?} at chart rank {rank}"
            )),
        }
    }

    /// A Möbius band's quotient coordinates (#2906): the loop fraction `s ∈ [0, 1)` and the
    /// signed width `w ∈ [−1, 1]` on the period-two double cover the Möbius seed races on, whose
    /// deck twin of `(s, w)` is `(s + 1, −w)`.
    ///
    /// Both coordinates are solved on the atlas's neighbourhood graph, and the width is carried
    /// through the charts' handedness, so the read exists only where the cover is twisted:
    ///
    /// 1. A graph edge `(i, j)` whose rows' home charts `a`, `b` resolve a handedness carries
    ///    `τ_ij = sgn det(F_bᵀF_a)` ([`frame_handedness`], the sign the transitions carry), and
    ///    `+1` inside one chart. When `τ` is a coboundary on the graph, a coherent row
    ///    orientation exists, the cover carries no orientation-reversing holonomy, and the read
    ///    is refused.
    /// 2. The loop generator is the longest mean seam displacement over the well-conditioned
    ///    non-tree transitions, either orientation. An edge's winding `k_ij` is the nearest
    ///    integer to its developed gap's coefficient along that generator: `0` inside one sheet
    ///    of the development, `±1` across a seam.
    /// 3. The loop coordinate `θ` minimizes `Σ (θ_j − θ_i − k_ij)²` with row `0` pinned, and
    ///    `s = θ − ⌊θ⌋`.
    /// 4. The width increment along a resolved edge is its chart displacement across the loop,
    ///    `η_ij = ⟨n_a, φ_a(x_j) − φ_a(x_i)⟩`, with `n_a` the unit normal to chart `a`'s gradient
    ///    of `θ`. Carrying the width through the handedness gives `w_j = τ_ij·(w_i + η_ij)`. The
    ///    least-squares system of those equations has matrix `D − τ∘A`, which is positive
    ///    definite exactly when `τ` is not a coboundary, the twist step 1 found.
    /// 5. The width is oriented over the band cut at `s = 0`, propagated along the resolved edges
    ///    that do not cross that cut, so rows on either side of the cut are each other's deck
    ///    twins. It is scaled by its largest magnitude.
    ///
    /// Refused for an orientable cover, for a cover with no non-tree seam, for a chart whose
    /// rows resolve no gradient of `θ`, for a width system that does not factor, and for rows
    /// the cut band does not reach.
    fn mobius_quotient_coordinates(
        &self,
        z: ArrayView2<'_, f64>,
        map: &DevelopingMap,
        developed: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let n = developed.nrows();
        if self.neighbourhood.len() != n {
            return Err(format!(
                "holonomy_quotient_coordinates: the atlas's neighbourhood graph has {} rows but {n} were supplied",
                self.neighbourhood.len()
            ));
        }
        let home = self.home_charts(z)?;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (i, neighbours) in self.neighbourhood.iter().enumerate() {
            for &(j, _) in neighbours {
                if j > i {
                    edges.push((i, j));
                }
            }
        }

        // (1) The handedness of every edge whose home charts resolve one, and whether it is a
        // coboundary.
        let mut chart_pair_sign: BTreeMap<(usize, usize), Option<i8>> = BTreeMap::new();
        let mut signed: Vec<(usize, usize, i8)> = Vec::with_capacity(edges.len());
        for &(i, j) in &edges {
            let (a, b) = (home[i].min(home[j]), home[i].max(home[j]));
            let sign = if a == b {
                Some(1)
            } else {
                *chart_pair_sign.entry((a, b)).or_insert_with(|| {
                    frame_handedness(&self.charts[a], &self.charts[b]).resolved_sign()
                })
            };
            if let Some(sign) = sign {
                signed.push((i, j, sign));
            }
        }
        let mut signed_adjacency: Vec<Vec<(usize, i8)>> = vec![Vec::new(); n];
        for &(i, j, sign) in &signed {
            signed_adjacency[i].push((j, sign));
            signed_adjacency[j].push((i, sign));
        }
        let mut orientation = vec![0i8; n];
        let mut contradicted = false;
        for root in 0..n {
            if orientation[root] != 0 {
                continue;
            }
            orientation[root] = 1;
            let mut queue = std::collections::VecDeque::from([root]);
            while let Some(row) = queue.pop_front() {
                for &(next, sign) in &signed_adjacency[row] {
                    let required = orientation[row] * sign;
                    if orientation[next] == 0 {
                        orientation[next] = required;
                        queue.push_back(next);
                    } else if orientation[next] != required {
                        contradicted = true;
                    }
                }
            }
        }
        if !contradicted {
            return Err(
                "holonomy_quotient_coordinates: the charts' handedness is a coboundary on the rows' neighbourhood graph, so the cover carries no orientation-reversing holonomy and there is no band twist to read"
                    .to_string(),
            );
        }

        // (2) The loop generator.
        let generator = self
            .seam_displacements(z, map)
            .into_iter()
            .map(|seam| seam.displacement)
            .max_by(|left, right| left.dot(left).total_cmp(&right.dot(right)))
            .ok_or_else(|| {
                "holonomy_quotient_coordinates: no well-conditioned non-tree transition, so no loop translation to read"
                    .to_string()
            })?;
        let generator_sq = generator.dot(&generator);
        if !(generator_sq > 0.0 && generator_sq.is_finite()) {
            return Err(format!(
                "holonomy_quotient_coordinates: the loop translation has no length (squared {generator_sq:.3e})"
            ));
        }

        // (3) The harmonic loop coordinate, row 0 pinned.
        let mut laplacian: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut loop_rhs = Array1::<f64>::zeros(n);
        for &(i, j) in &edges {
            let gap = &developed.row(j) - &developed.row(i);
            let winding = (gap.dot(&generator) / generator_sq).round();
            loop_rhs[j] += winding;
            loop_rhs[i] -= winding;
            *laplacian.entry((i, i)).or_insert(0.0) += 1.0;
            *laplacian.entry((j, j)).or_insert(0.0) += 1.0;
            *laplacian.entry((i, j)).or_insert(0.0) -= 1.0;
        }
        let mut theta = Array1::<f64>::zeros(n);
        if n > 1 {
            let pinned: BTreeMap<(usize, usize), f64> = laplacian
                .iter()
                .filter(|&(&(row, col), _)| row > 0 && col > 0)
                .map(|(&(row, col), &value)| ((row - 1, col - 1), value))
                .collect();
            let solved = solve_sparse_symmetric(n - 1, &pinned, loop_rhs.slice(s![1..]))
                .map_err(|error| {
                    format!("holonomy_quotient_coordinates: the loop Laplacian did not factor: {error}")
                })?;
            theta.slice_mut(s![1..]).assign(&solved);
        }
        let loop_fraction = theta.mapv(|value| value - value.floor());

        // (4) Each chart's unit normal to its gradient of θ, then the twisted width system.
        let mut normals: Vec<[f64; 2]> = Vec::with_capacity(self.charts.len());
        for (chart_index, (chart, patch)) in self.charts.iter().zip(self.patches.iter()).enumerate()
        {
            let center = chart.project(z.row(patch.center));
            let mut gram = [[0.0_f64; 2]; 2];
            let mut moment = [0.0_f64; 2];
            for &member in &patch.members {
                let image = chart.project(z.row(member));
                let offset = [image[0] - center[0], image[1] - center[1]];
                let lift = theta[member] - theta[patch.center];
                let lift = lift - lift.round();
                for r in 0..2 {
                    moment[r] += offset[r] * lift;
                    for c in 0..2 {
                        gram[r][c] += offset[r] * offset[c];
                    }
                }
            }
            let det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
            let gradient = [
                (gram[1][1] * moment[0] - gram[0][1] * moment[1]) / det,
                (gram[0][0] * moment[1] - gram[1][0] * moment[0]) / det,
            ];
            let norm = gradient[0].hypot(gradient[1]);
            if !(det > 0.0 && norm > 0.0 && norm.is_finite()) {
                return Err(format!(
                    "holonomy_quotient_coordinates: chart {chart_index} resolves no gradient of the loop coordinate (determinant {det:.3e}, gradient norm {norm:.3e})"
                ));
            }
            normals.push([-gradient[1] / norm, gradient[0] / norm]);
        }
        let mut twisted: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut width_rhs = Array1::<f64>::zeros(n);
        for &(i, j, sign) in &signed {
            let chart = home[i];
            let from = self.charts[chart].project(z.row(i));
            let to = self.charts[chart].project(z.row(j));
            let increment =
                normals[chart][0] * (to[0] - from[0]) + normals[chart][1] * (to[1] - from[1]);
            let tau = f64::from(sign);
            *twisted.entry((i, i)).or_insert(0.0) += 1.0;
            *twisted.entry((j, j)).or_insert(0.0) += 1.0;
            *twisted.entry((i, j)).or_insert(0.0) -= tau;
            width_rhs[j] += tau * increment;
            width_rhs[i] -= increment;
        }
        let width = solve_sparse_symmetric(n, &twisted, width_rhs.view()).map_err(|error| {
            format!("holonomy_quotient_coordinates: the twisted width system did not factor: {error}")
        })?;

        // (5) Orient the width over the band cut at s = 0.
        let mut sheet = vec![0i8; n];
        sheet[0] = 1;
        let mut queue = std::collections::VecDeque::from([0usize]);
        while let Some(row) = queue.pop_front() {
            for &(next, sign) in &signed_adjacency[row] {
                if sheet[next] == 0 && (loop_fraction[next] - loop_fraction[row]).abs() <= 0.5 {
                    sheet[next] = sheet[row] * sign;
                    queue.push_back(next);
                }
            }
        }
        let unreached = sheet.iter().filter(|&&side| side == 0).count();
        if unreached > 0 {
            return Err(format!(
                "holonomy_quotient_coordinates: {unreached} rows are not reached across the band cut at s = 0, so their width has no orientation"
            ));
        }
        let scale = width.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(format!(
                "holonomy_quotient_coordinates: the band width is degenerate (largest magnitude {scale:.3e})"
            ));
        }
        let mut coords = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            coords[[row, 0]] = loop_fraction[row];
            coords[[row, 1]] = f64::from(sheet[row]) * width[row] / scale;
        }
        Ok(coords)
    }

    /// The developing map's placement (#2280): each chart's rigid motion
    /// `G_a(c) = Q_a c + s_a` into the root chart's frame, and which transitions the
    /// spanning tree glued through.
    fn developing_map(&self) -> Result<DevelopingMap, String> {
        fn find_root(parent: &mut [usize], mut node: usize) -> usize {
            while parent[node] != node {
                parent[node] = parent[parent[node]];
                node = parent[node];
            }
            node
        }

        let d = self.intrinsic_dim;
        let m = self.charts.len();
        let mut order: Vec<usize> = (0..self.transitions.len())
            .filter(|&index| {
                matches!(
                    self.transitions[index].conditioning,
                    TransitionConditioning::WellConditioned
                )
            })
            .collect();
        order.sort_by(|&left, &right| {
            let (left, right) = (&self.transitions[left], &self.transitions[right]);
            left.residual
                .total_cmp(&right.residual)
                .then(left.overlap_id.cmp(&right.overlap_id))
        });
        let mut parent: Vec<usize> = (0..m).collect();
        let mut tree: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut in_tree = vec![false; self.transitions.len()];
        let mut tree_edges = 0usize;
        for edge in order {
            let transition = &self.transitions[edge];
            let from = find_root(&mut parent, transition.from_patch);
            let to = find_root(&mut parent, transition.to_patch);
            if from != to {
                parent[from.max(to)] = from.min(to);
                tree[transition.from_patch].push(edge);
                tree[transition.to_patch].push(edge);
                in_tree[edge] = true;
                tree_edges += 1;
            }
        }
        if tree_edges + 1 != m {
            return Err(format!(
                "developing map: the well-conditioned transitions leave the {m} charts in {} components",
                m - tree_edges
            ));
        }

        let mut rotation: Vec<Array2<f64>> = vec![Array2::<f64>::eye(d); m];
        let mut offset: Vec<Array1<f64>> = vec![Array1::<f64>::zeros(d); m];
        let mut placed = vec![false; m];
        placed[0] = true;
        let mut queue = std::collections::VecDeque::from([0usize]);
        while let Some(chart) = queue.pop_front() {
            for &edge in &tree[chart] {
                let transition = &self.transitions[edge];
                let forward = transition.from_patch == chart;
                let next = if forward {
                    transition.to_patch
                } else {
                    transition.from_patch
                };
                if placed[next] {
                    continue;
                }
                let (composed, shifted) = if forward {
                    // G_to(c) = G_from(Rᵀ(c − t)) = Q Rᵀ c + s − Q Rᵀ t.
                    let composed = rotation[chart].dot(&transition.rotation.t());
                    let shifted = &offset[chart] - &composed.dot(&transition.translation);
                    (composed, shifted)
                } else {
                    // G_from(c) = G_to(R c + t) = Q R c + Q t + s.
                    let composed = rotation[chart].dot(&transition.rotation);
                    let shifted = rotation[chart].dot(&transition.translation) + &offset[chart];
                    (composed, shifted)
                };
                rotation[next] = composed;
                offset[next] = shifted;
                placed[next] = true;
                queue.push_back(next);
            }
        }
        Ok(DevelopingMap {
            rotation,
            offset,
            in_tree,
        })
    }

    /// Every row of `z` through its home chart's placement, uncentered. A row's home is the
    /// patch that contains it with the nearest center (lowest index on a tie); a row no patch
    /// contains takes the nearest center's chart, whose map `Fᵀ(x − μ)` is defined at every
    /// ambient point.
    fn develop_rows(
        &self,
        z: ArrayView2<'_, f64>,
        map: &DevelopingMap,
    ) -> Result<Array2<f64>, String> {
        let home = self.home_charts(z)?;
        let mut coords = Array2::<f64>::zeros((z.nrows(), self.intrinsic_dim));
        for (row, &chart) in home.iter().enumerate() {
            let image = map.rotation[chart].dot(&self.charts[chart].project(z.row(row)))
                + &map.offset[chart];
            coords.row_mut(row).assign(&image);
        }
        Ok(coords)
    }

    /// Each row's home chart: the patch that contains it with the nearest center (lowest index
    /// on a tie), or for a row no patch contains, the nearest center's chart.
    fn home_charts(&self, z: ArrayView2<'_, f64>) -> Result<Vec<usize>, String> {
        let (n, p) = z.dim();
        if p != self.ambient_dim {
            return Err(format!(
                "developing map: rows have {p} columns but the atlas was built in {} dimensions",
                self.ambient_dim
            ));
        }
        let largest_member = self
            .patches
            .iter()
            .flat_map(|patch| patch.members.iter().copied())
            .max()
            .unwrap_or(0);
        if n <= largest_member {
            return Err(format!(
                "developing map: the atlas charts row {largest_member} but only {n} rows were supplied"
            ));
        }
        let mut home: Vec<Option<(usize, f64)>> = vec![None; n];
        for (chart, patch) in self.patches.iter().enumerate() {
            for &row in &patch.members {
                let distance = sq_distance(z, row, patch.center);
                let nearer = match home[row] {
                    Some((_, nearest)) => distance < nearest,
                    None => true,
                };
                if nearer {
                    home[row] = Some((chart, distance));
                }
            }
        }
        Ok(home
            .into_iter()
            .enumerate()
            .map(|(row, owner)| match owner {
                Some((chart, _)) => chart,
                None => {
                    let mut nearest_chart = 0usize;
                    let mut nearest = f64::INFINITY;
                    for (chart, patch) in self.patches.iter().enumerate() {
                        let distance = sq_distance(z, row, patch.center);
                        if distance < nearest {
                            nearest = distance;
                            nearest_chart = chart;
                        }
                    }
                    nearest_chart
                }
            })
            .collect())
    }

    /// The holonomy of every well-conditioned transition the spanning tree did not use, in
    /// the developed frame (see `holonomy_quotient_coordinates`).
    fn non_tree_holonomies(&self, map: &DevelopingMap) -> Vec<DevelopedHolonomy> {
        let mut holonomies = Vec::new();
        for (index, transition) in self.transitions.iter().enumerate() {
            if map.in_tree[index]
                || !matches!(
                    transition.conditioning,
                    TransitionConditioning::WellConditioned
                )
            {
                continue;
            }
            let (from, to) = (transition.from_patch, transition.to_patch);
            let linear = map.rotation[to]
                .dot(&transition.rotation)
                .dot(&map.rotation[from].t());
            let translation = map.rotation[to].dot(&transition.translation) + &map.offset[to]
                - linear.dot(&map.offset[from]);
            let reversing = determinant(&linear) < 0.0;
            holonomies.push(DevelopedHolonomy {
                translation,
                reversing,
            });
        }
        holonomies
    }

    /// For every well-conditioned transition the spanning tree did not use, the mean over its
    /// shared rows of a row's image through the `to` chart's placement minus its image through
    /// the `from` chart's placement: the holonomy's translation where the two sheets of the
    /// development meet, and whether that holonomy reverses orientation (see
    /// `holonomy_quotient_coordinates`).
    fn seam_displacements(
        &self,
        z: ArrayView2<'_, f64>,
        map: &DevelopingMap,
    ) -> Vec<SeamDisplacement> {
        let mut displacements = Vec::new();
        for (index, transition) in self.transitions.iter().enumerate() {
            if map.in_tree[index]
                || !matches!(
                    transition.conditioning,
                    TransitionConditioning::WellConditioned
                )
            {
                continue;
            }
            let (from, to) = (transition.from_patch, transition.to_patch);
            let linear = map.rotation[to]
                .dot(&transition.rotation)
                .dot(&map.rotation[from].t());
            let reversing = determinant(&linear) < 0.0;
            let shared =
                sorted_intersection(&self.patches[from].members, &self.patches[to].members);
            if shared.is_empty() {
                continue;
            }
            let mut displacement = Array1::<f64>::zeros(self.intrinsic_dim);
            for &row in &shared {
                let through_from = map.rotation[from].dot(&self.charts[from].project(z.row(row)))
                    + &map.offset[from];
                let through_to =
                    map.rotation[to].dot(&self.charts[to].project(z.row(row))) + &map.offset[to];
                displacement += &(through_to - through_from);
            }
            displacement /= shared.len() as f64;
            displacements.push(SeamDisplacement {
                displacement,
                reversing,
            });
        }
        displacements
    }

    /// Numerically well-conditioned observed transition signs as
    /// `(a, b, overlap, sign)`.
    ///
    /// This is a diagnostic of the fitted atlas, not an exact or finite-sample
    /// sign certificate. It deliberately returns ordinary tuples that no
    /// authoritative holonomy constructor accepts as analytic provenance.
    #[must_use]
    pub(crate) fn observed_signed_edges(&self) -> Vec<(usize, usize, usize, i8)> {
        self.transitions
            .iter()
            .filter(|transition| {
                matches!(
                    transition.conditioning,
                    TransitionConditioning::WellConditioned
                )
            })
            .map(|t| (t.from_patch, t.to_patch, t.overlap_id, t.sign))
            .collect()
    }

    /// Point readout from the numerically well-conditioned portion of the fitted
    /// sign cocycle.
    ///
    /// This method is intentionally named `observed_*`: it has no sampling model,
    /// error probability, or authority to promote topology. Use the Gaussian-PCA
    /// holonomy certificate for a population claim. A contradictory revisit is a
    /// negative observed-holonomy cycle; disconnected components are resolved
    /// independently.
    #[must_use]
    pub fn observed_orientability(&self) -> AtlasOrientability {
        let mut orientation: BTreeMap<usize, i8> = BTreeMap::new();
        // Adjacency over numerically well-conditioned observed edges only.
        let mut adj: BTreeMap<usize, Vec<(usize, i8)>> = BTreeMap::new();
        for (a, b, _, sign) in self.observed_signed_edges() {
            adj.entry(a).or_default().push((b, sign));
            adj.entry(b).or_default().push((a, sign));
        }
        for root in 0..self.charts.len() {
            if orientation.contains_key(&root) {
                continue;
            }
            orientation.insert(root, 1);
            let mut queue = std::collections::VecDeque::from([root]);
            while let Some(chart) = queue.pop_front() {
                let here = orientation[&chart];
                if let Some(neighbors) = adj.get(&chart) {
                    for &(next, sign) in neighbors {
                        let required = here * sign;
                        match orientation.get(&next) {
                            Some(&existing) if existing != required => {
                                return AtlasOrientability::NonOrientable;
                            }
                            Some(_) => {}
                            None => {
                                orientation.insert(next, required);
                                queue.push_back(next);
                            }
                        }
                    }
                }
            }
        }
        AtlasOrientability::Orientable
    }
}

/// The chart of the LARGEST certifiable neighborhood of `center`, at most
/// `patch_size` rows.
///
/// A local-PCA frame is the patch's TANGENT plane only while the neighborhood stays
/// inside the local curvature scale. Past it the leading `d` singular directions
/// stop being tangent: on a strongly curved, anisotropically sampled patch (a coarse
/// swiss roll, where the along-roll sample spacing is many times the across-roll
/// spacing) the top two directions are both spent resolving the arc's BEND, the
/// across-roll direction is demoted out of the frame, and rows separated only across
/// the roll project onto the SAME chart coordinate — the chart is not injective and
/// the patch is honestly rejected. The cure is not to relax the certificate but to
/// use a neighborhood the tangent plane actually fits: drop the last member to join
/// and retry, down to the `2(d + 1)` over-determination floor. Prefixes of the
/// connected distance order are nested, so this walks a deterministic chain of
/// shrinking neighborhoods and returns the first — hence largest — one that
/// certifies. If none does, the last (smallest-neighborhood) typed error is returned:
/// genuinely `d`-degenerate data (a line charted at `d = 2`) still fails, at every
/// size, with `DegeneratePatch`.
fn certified_neighborhood_chart(
    z: ArrayView2<'_, f64>,
    neighbours: &[Vec<(usize, f64)>],
    center: usize,
    patch_size: usize,
    d: usize,
) -> Result<(Vec<usize>, LocalChart), LocalChartError> {
    let order = connected_distance_order(z, neighbours, center);
    // Over-determination floor: a `d`-frame and a `d`-dimensional Procrustes both
    // want at least `2(d + 1)` rows, but never demand more rows than the caller's
    // patch budget, and never fewer than the `d + 1` a `d`-chart strictly needs.
    let floor = (2 * (d + 1)).min(patch_size).max(d + 1);
    let mut size = patch_size;
    loop {
        let mut members: Vec<usize> = order.iter().take(size).copied().collect();
        members.sort_unstable();
        match build_local_chart(z, center, &members, d) {
            Ok(chart) => return Ok((members, chart)),
            Err(_) if size > floor => size -= 1,
            Err(err) => return Err(err),
        }
    }
}

fn sq_distance(z: ArrayView2<'_, f64>, a: usize, b: usize) -> f64 {
    (0..z.ncols())
        .map(|column| {
            let difference = z[[a, column]] - z[[b, column]];
            difference * difference
        })
        .sum()
}

/// Intersection of two ascending-sorted row lists, ascending.
pub(super) fn sorted_intersection(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Local PCA chart of a neighborhood, with rank + injectivity certification.
fn build_local_chart(
    z: ArrayView2<'_, f64>,
    center: usize,
    members: &[usize],
    d: usize,
) -> Result<LocalChart, LocalChartError> {
    let m = members.len();
    let p = z.ncols();
    // Centered neighborhood block (m × p).
    let mut mean = Array1::<f64>::zeros(p);
    for &row in members {
        for c in 0..p {
            mean[c] += z[[row, c]];
        }
    }
    mean.mapv_inplace(|v| v / m as f64);
    let mut centered = Array2::<f64>::zeros((m, p));
    for (r, &row) in members.iter().enumerate() {
        for c in 0..p {
            centered[[r, c]] = z[[row, c]] - mean[c];
        }
    }

    let (_, svals, vt) = centered
        .svd(false, true)
        .map_err(|err| LocalChartError::SvdFailure {
            center,
            detail: format!("{err:?}"),
        })?;
    let vt = vt.expect("svd(_, true) returns Vᵀ");
    let rank = svals.len();
    if rank < d {
        return Err(LocalChartError::DegeneratePatch {
            center,
            intrinsic_dim: d,
            smallest_captured_singular: 0.0,
            leading_singular: svals.first().copied().unwrap_or(0.0),
        });
    }
    let leading = svals[0];
    let smallest_captured = svals[d - 1];
    // Numerical rank of the centered `m × p` block. A backward-stable SVD returns the
    // exact singular values of a block within `max(m, p)·ε·σ₁` of this one, so by
    // Weyl's inequality a `σ_d` at or below that tolerance cannot be told apart from an
    // exactly rank-deficient neighborhood: the patch spans fewer than `d` directions,
    // its frame's `d`-th axis is roundoff, and it is rejected rather than given one.
    let rank_tolerance = leading * (m.max(p) as f64) * f64::EPSILON;
    if !(leading > 0.0) || smallest_captured <= rank_tolerance {
        return Err(LocalChartError::DegeneratePatch {
            center,
            intrinsic_dim: d,
            smallest_captured_singular: smallest_captured,
            leading_singular: leading,
        });
    }

    // Frame = leading d right singular vectors (rows of Vᵀ), stored as (p × d), each
    // put in the CANONICAL SIGN GAUGE: a singular vector is only defined up to sign,
    // so the solver's arbitrary choice is replaced by a deterministic, data-derived
    // one — make each axis's largest-magnitude component positive (lowest index wins
    // a tie). Flipping an axis is an orthogonal change of chart coordinates, so it
    // leaves every distance, the injectivity certificate and the captured variance
    // untouched; what it buys is that a `det` read off these frames measures a
    // GEOMETRIC relation between two patches rather than a pair of coin flips.
    let mut frame = Array2::<f64>::zeros((p, d));
    for ax in 0..d {
        for c in 0..p {
            frame[[c, ax]] = vt[[ax, c]];
        }
        let mut pivot = 0usize;
        let mut best = frame[[0, ax]].abs();
        for c in 1..p {
            let v = frame[[c, ax]].abs();
            if v > best {
                best = v;
                pivot = c;
            }
        }
        if frame[[pivot, ax]] < 0.0 {
            for c in 0..p {
                frame[[c, ax]] = -frame[[c, ax]];
            }
        }
    }
    // Chart coordinates of every member: centered · frame  (m × d).
    let coords = centered.dot(&frame);

    // Injectivity certificate. The orthogonal split
    // `‖x_a − x_b‖² = ‖c_a − c_b‖² + ‖r_a − r_b‖²` makes the projected squared
    // distance exact, so the chart is injective on its neighborhood iff every pair the
    // ambient separates stays separated in the chart. Each pair is decided at the
    // computation's own resolution, with `s = ‖x_a‖ + ‖x_b‖` over the centered rows:
    // - an ambient coordinate difference takes two roundings of centered entries, so
    //   coincident rows read an ambient squared distance of at most `(γ₂·s)²`;
    // - a chart coordinate is a `p`-term inner product against a unit frame axis, whose
    //   summand absolute sum is at most the row norm (Cauchy–Schwarz), so a coordinate
    //   difference rounds within `γ_{p+1}·s` and a collapsed pair reads a projected
    //   squared distance of at most `d·(γ_{p+1}·s)²`.
    // A pair whose ambient distance clears its band while its projected distance does
    // not is a collapse the chart cannot tell from a coincidence, so the chart is
    // refused. Deciding pair by pair keeps a near-duplicate pair from masking a
    // collapsed one, which comparing the smallest projected distance against the
    // smallest ambient distance (two different pairs) does not. The smallest lower
    // stretch over the separated pairs goes on the certificate for consumers that want
    // a geometric bar.
    let row_norms: Vec<f64> = (0..m)
        .map(|r| {
            (0..p)
                .map(|c| centered[[r, c]] * centered[[r, c]])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let ambient_growth = gam_linalg::roundoff::accumulation_growth(2);
    let projected_growth = gam_linalg::roundoff::accumulation_growth(p + 1);
    let mut min_stretch = f64::INFINITY;
    for a in 0..m {
        for b in (a + 1)..m {
            let mut amb = 0.0;
            for c in 0..p {
                let diff = centered[[a, c]] - centered[[b, c]];
                amb += diff * diff;
            }
            let scale = row_norms[a] + row_norms[b];
            if !(amb > (ambient_growth * scale).powi(2)) {
                continue;
            }
            let mut proj = 0.0;
            for ax in 0..d {
                let diff = coords[[a, ax]] - coords[[b, ax]];
                proj += diff * diff;
            }
            if proj <= d as f64 * (projected_growth * scale).powi(2) {
                return Err(LocalChartError::NonInjectiveChart {
                    center,
                    projected_sq_distance: proj,
                    ambient_sq_distance: amb,
                });
            }
            let stretch = (proj / amb).sqrt();
            if stretch < min_stretch {
                min_stretch = stretch;
            }
        }
    }

    let total_variance: f64 = svals.iter().map(|s| s * s).sum();
    let captured: f64 = svals.iter().take(d).map(|s| s * s).sum();
    let captured_variance_fraction = if total_variance > 0.0 {
        captured / total_variance
    } else {
        0.0
    };

    let singular_values = Array1::from_iter(svals.iter().take(d).copied());
    let certificate = ChartCertificate {
        condition: leading / smallest_captured,
        leading_singular: leading,
        smallest_captured_singular: smallest_captured,
        captured_variance_fraction,
        min_projection_stretch: if min_stretch.is_finite() {
            min_stretch
        } else {
            1.0
        },
    };

    Ok(LocalChart {
        center,
        mean,
        frame,
        singular_values,
        coords,
        certificate,
    })
}

/// Orthogonal Procrustes transition between two charts on their shared support.
///
/// The patch indices select both the chart and its row-aligned, sorted member
/// list from the same atlas. Deriving those coupled inputs here makes it
/// impossible for callers to pair a chart with another patch's membership.
/// `shared` is a subset of both selected member lists.
fn build_transition(
    charts: &[LocalChart],
    patches: &[LocalPatch],
    from_patch: usize,
    to_patch: usize,
    overlap_id: usize,
    shared: &[usize],
) -> ChartTransition {
    let chart_i = &charts[from_patch];
    let chart_j = &charts[to_patch];
    let members_i = &patches[from_patch].members;
    let members_j = &patches[to_patch].members;
    let d = chart_i.frame.ncols();
    let s = shared.len();
    // Shared-support coordinates in each chart, (d × s).
    let mut c_from = Array2::<f64>::zeros((d, s));
    let mut c_to = Array2::<f64>::zeros((d, s));
    for (col, &row) in shared.iter().enumerate() {
        let li = members_i
            .binary_search(&row)
            .expect("shared row is a member of patch i");
        let lj = members_j
            .binary_search(&row)
            .expect("shared row is a member of patch j");
        for ax in 0..d {
            c_from[[ax, col]] = chart_i.coords[[li, ax]];
            c_to[[ax, col]] = chart_j.coords[[lj, ax]];
        }
    }
    // Center each set of shared coordinates.
    let mut mean_from = Array1::<f64>::zeros(d);
    let mut mean_to = Array1::<f64>::zeros(d);
    for ax in 0..d {
        let mut sf = 0.0;
        let mut st = 0.0;
        for col in 0..s {
            sf += c_from[[ax, col]];
            st += c_to[[ax, col]];
        }
        mean_from[ax] = sf / s as f64;
        mean_to[ax] = st / s as f64;
    }
    for ax in 0..d {
        for col in 0..s {
            c_from[[ax, col]] -= mean_from[ax];
            c_to[[ax, col]] -= mean_to[ax];
        }
    }

    let handedness = frame_handedness(chart_i, chart_j);
    let sign = handedness.sign;
    let frame_nondegenerate = handedness.resolved_sign().is_some();

    // ALIGNMENT: orthogonal Procrustes, minimize ‖C_to − R C_from‖_F over the
    // handedness class {R ∈ O(d) : det R = sign}. M = C_to C_fromᵀ (d × d);
    // SVD M = U S Vᵀ; the free optimum is U Vᵀ, and the constrained optimum flips the
    // LAST (weakest) singular direction when the free one lands in the wrong class:
    // R = U diag(1, …, 1, ±1) Vᵀ. So det R = sign by construction, and reflections are
    // recorded rather than forced to +1.
    let m_mat = c_to.dot(&c_from.t());
    let (rotation, conditioning) = match m_mat.svd(true, true) {
        Ok((Some(u), sv, Some(vt))) => {
            let mut r = u.dot(&vt);
            if (determinant(&r) >= 0.0) != (sign >= 0) {
                let mut flipped = u;
                let last = d - 1;
                for row in 0..d {
                    flipped[[row, last]] = -flipped[[row, last]];
                }
                r = flipped.dot(&vt);
            }
            let leading = sv.first().copied().unwrap_or(0.0);
            let smallest = sv.get(d - 1).copied().unwrap_or(0.0);
            // The Procrustes ALIGNMENT is well posed only when `M = C_to C_fromᵀ` has
            // full numerical rank; otherwise the shared support does not span all `d`
            // chart directions and the fitted rotation is undetermined in the unspanned
            // one. Each entry of `M` sums `s` products, and on an overlap the two charts
            // genuinely share `C_to ≈ R·C_from`, so `σ₁(M) ≈ ‖C_to‖₂‖C_from‖₂` and the
            // accumulated error is at most `s·ε·σ₁`; the `d × d` SVD adds `d·ε·σ₁`. Below
            // `max(s, d)·ε·σ₁` the smallest direction is roundoff. Such an edge is kept
            // as geometry but marked [`TransitionConditioning::Degenerate`] and held out
            // of the sign cocycle, mirroring the analytic-vs-fitted split in
            // [`super::chart_atlas`].
            let rank_tolerance = leading * (s.max(d) as f64) * f64::EPSILON;
            let well_posed = leading > 0.0 && smallest > rank_tolerance;
            let conditioning = if well_posed && frame_nondegenerate {
                TransitionConditioning::WellConditioned
            } else {
                TransitionConditioning::Degenerate
            };
            (r, conditioning)
        }
        // A failed / rank-empty SVD leaves the alignment unresolved: the identity of
        // the right handedness class, degenerate conditioning (excluded from the
        // observed sign cocycle).
        _ => (signed_identity(d, sign), TransitionConditioning::Degenerate),
    };

    // Residual ‖C_to − R C_from‖_F / ‖C_to‖_F.
    let rc = rotation.dot(&c_from);
    let mut num = 0.0;
    let mut den = 0.0;
    for ax in 0..d {
        for col in 0..s {
            let diff = c_to[[ax, col]] - rc[[ax, col]];
            num += diff * diff;
            den += c_to[[ax, col]] * c_to[[ax, col]];
        }
    }
    let residual = if den > 0.0 { (num / den).sqrt() } else { 0.0 };

    // Translation t = mean_to − R mean_from.
    let mut translation = mean_to.clone();
    for i in 0..d {
        let mut acc = 0.0;
        for j in 0..d {
            acc += rotation[[i, j]] * mean_from[j];
        }
        translation[i] -= acc;
    }

    ChartTransition {
        from_patch,
        to_patch,
        overlap_id,
        shared_rows: shared.to_vec(),
        rotation,
        translation,
        sign,
        residual,
        smallest_principal_cosine: handedness.smallest_principal_cosine,
        sign_resolution_budget: handedness.sign_resolution_budget,
        conditioning,
    }
}

/// The handedness of the chart change between two charts, with what it rests on.
struct FrameHandedness {
    /// `sgn det(F_toᵀ F_from)`.
    sign: i8,
    /// `σ_min(F_toᵀ F_from)`, the cosine of the largest principal angle between the planes.
    smallest_principal_cosine: f64,
    /// The two frames' combined angular resolution ([`combined_frame_resolution`]).
    sign_resolution_budget: f64,
}

impl FrameHandedness {
    /// The sign, while the smallest principal cosine clears the frames' combined resolution.
    fn resolved_sign(&self) -> Option<i8> {
        (self.smallest_principal_cosine > self.sign_resolution_budget).then_some(self.sign)
    }
}

/// The handedness of the chart change `from → to` (#2280), read off the two frames.
///
/// ORIENTATION: the exact transition Jacobian. On an overlap the chart change is
///     c_to = F_toᵀ(μ_from − μ_to) + A · c_from + O(curvature),   A = F_toᵀ F_from,
/// so the handedness relation of the two charts is sgn det A, with
/// |det A| = ∏_k cos θ_k over the principal angles between the tangent planes.
/// This is a property of the two FRAMES: unlike a Procrustes fit to the shared
/// point cloud, it does not degrade when the overlap is small or elongated (where
/// a reflection fits the points exactly as well as a rotation, at the same
/// residual, and the fitted det is a coin flip). `det A = det Aᵀ`, so the sign does
/// not depend on which chart is `from`.
///
/// RESOLVABILITY of that sign. The singular values of `A` are the cosines of the
/// principal angles, so `det A` changes sign exactly when some `θ_k` crosses
/// `π/2` — the handedness is a fact about the manifold only while every
/// `cos θ_k` clears what the two frames' own estimation error could manufacture.
/// `σ_min(A) = min_k cos θ_k` is the sharp quantity: `|det A| = ∏_k cos θ_k ≤
/// σ_min(A)`, so gating the DETERMINANT against the same budget is the
/// conservative relaxation, and at `d > 1` it refuses edges whose every angle is
/// individually well resolved merely because the product of several cosines is
/// small. The budget itself is derived from the two charts' certificates
/// ([`frame_angular_resolution`]) rather than being a numerical floor: a floor
/// asks "is this determinant distinguishable from zero in f64", which is a
/// question about arithmetic, and the question the sign needs answered is
/// whether it is distinguishable from zero given how well the two local PCAs
/// pinned their tangent planes.
fn frame_handedness(from: &LocalChart, to: &LocalChart) -> FrameHandedness {
    let a_mat = frame_overlap(&to.frame, &from.frame);
    let sign: i8 = if determinant(&a_mat) >= 0.0 { 1 } else { -1 };
    let sign_resolution_budget = combined_frame_resolution(
        frame_angular_resolution(&from.certificate),
        frame_angular_resolution(&to.certificate),
    );
    let smallest_principal_cosine = match a_mat.svd(false, false) {
        Ok((_, sv, _)) => sv.iter().copied().fold(f64::INFINITY, f64::min),
        // An unresolved SVD leaves the principal angles unknown, so the edge carries
        // no handedness rather than a guessed one.
        Err(_) => 0.0,
    };
    let smallest_principal_cosine = if smallest_principal_cosine.is_finite() {
        smallest_principal_cosine
    } else {
        0.0
    };
    FrameHandedness {
        sign,
        smallest_principal_cosine,
        sign_resolution_budget,
    }
}

/// Solve `H·x = rhs` for a symmetric positive definite `H` given by its upper-triangle
/// entries, through one sparse Cholesky factor.
fn solve_sparse_symmetric(
    dim: usize,
    upper: &BTreeMap<(usize, usize), f64>,
    rhs: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let triplets: Vec<Triplet<usize, usize, f64>> = upper
        .iter()
        .map(|(&(row, col), &value)| Triplet::new(row, col, value))
        .collect();
    let matrix = SparseColMat::try_new_from_triplets(dim, dim, &triplets)
        .map_err(|error| format!("the sparse system did not assemble: {error:?}"))?;
    let factor = factorize_sparse_spd(&matrix).map_err(|error| error.to_string())?;
    solve_sparse_spd(&factor, &rhs).map_err(|error| error.to_string())
}

/// The `d × d` frame overlap `F_toᵀ F_from` — the Jacobian of the fitted chart
/// transition `c_from ↦ c_to` on the two patches' common tangent plane. Its entries
/// are the pairwise inner products of the two orthonormal chart frames, its singular
/// values are the cosines of the principal angles between the tangent planes, and
/// its determinant's sign is the charts' handedness relation.
fn frame_overlap(frame_to: &Array2<f64>, frame_from: &Array2<f64>) -> Array2<f64> {
    frame_to.t().dot(frame_from)
}

/// The `d × d` identity of the handedness class `sign`: `diag(1, …, 1, sign)`.
fn signed_identity(d: usize, sign: i8) -> Array2<f64> {
    let mut m = Array2::<f64>::eye(d);
    if sign < 0 && d > 0 {
        m[[d - 1, d - 1]] = -1.0;
    }
    m
}

/// Determinant of a small square matrix by Gaussian elimination with partial
/// pivoting. Used only to read the SIGN of an orthogonal Procrustes factor, where
/// `|det| = 1`.
fn determinant(m: &Array2<f64>) -> f64 {
    let n = m.nrows();
    let mut a = m.clone();
    let mut det = 1.0;
    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        let mut best = a[[col, col]].abs();
        for r in (col + 1)..n {
            let v = a[[r, col]].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best == 0.0 {
            return 0.0;
        }
        if pivot != col {
            for c in 0..n {
                a.swap([col, c], [pivot, c]);
            }
            det = -det;
        }
        det *= a[[col, col]];
        for r in (col + 1)..n {
            let factor = a[[r, col]] / a[[col, col]];
            for c in col..n {
                let sub = factor * a[[col, c]];
                a[[r, c]] -= sub;
            }
        }
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::manifold::intrinsic_seed::farthest_point_landmarks;
    use crate::manifold::tests_topology_fixtures::{
        cylinder_strip, embedded_plane, mobius_strip, spherical_band, swiss_roll, torus,
    };

    /// Rows owned by each center: nearest center, lowest index on a tie.
    fn cell_occupancy(z: ArrayView2<'_, f64>, centers: &[usize]) -> Vec<usize> {
        let mut occupancy = vec![0usize; centers.len()];
        for row in 0..z.nrows() {
            let mut owner = 0usize;
            let mut nearest = f64::INFINITY;
            for (slot, &center) in centers.iter().enumerate() {
                let distance = sq_distance(z, center, row);
                if distance < nearest {
                    nearest = distance;
                    owner = slot;
                }
            }
            occupancy[owner] += 1;
        }
        occupancy
    }

    /// #2280 — the patch budget is a multiple of the MEAN cell occupancy, so the
    /// centers must deliver cells that hold about the mean. `swiss_roll(80, 16)`'s
    /// along-roll spacing grows as `√(1 + t²)`, so an ambient farthest-point net
    /// crowds rows into the inner-tip cells; the occupancy-normalized net must not.
    ///
    /// Bar: no cell holds more than twice the mean occupancy. Under locally uniform
    /// sampling a patch of `2^d = 4` mean occupancies then reaches at least `√2` of its
    /// cell radii. The plain ambient net on the SAME fixture is the positive control:
    /// it must fail the bar, or the fixture does not exercise the defect.
    #[test]
    fn patch_centers_hold_the_mean_occupancy_on_a_nonuniform_sample_2280() {
        let z = swiss_roll(80, 16);
        let n = z.nrows();
        let config = LocalAtlasConfig::balanced(n, 2);
        let mean = n as f64 / config.patch_count as f64;
        let plain = cell_occupancy(
            z.view(),
            &farthest_point_landmarks(z.view(), config.patch_count),
        );
        let normalized = cell_occupancy(
            z.view(),
            &occupancy_normalized_centers(
                z.view(),
                config.patch_count,
                n.div_ceil(config.patch_count),
            ),
        );
        assert_eq!(plain.len(), config.patch_count, "the plain net must place every center");
        assert_eq!(
            normalized.len(),
            config.patch_count,
            "the normalized net must place every center"
        );
        let plain_max = plain.iter().copied().max().unwrap_or(0);
        let normalized_max = normalized.iter().copied().max().unwrap_or(0);
        eprintln!(
            "#2280 cell occupancy on swiss_roll(80,16): mean={mean:.2} \
             plain_max={plain_max} normalized_max={normalized_max}"
        );
        assert!(
            plain_max as f64 > 2.0 * mean,
            "positive control: the ambient net must crowd a cell past twice the mean \
             ({plain_max} vs mean {mean:.2}), or this fixture does not exercise #2280"
        );
        assert!(
            normalized_max as f64 <= 2.0 * mean,
            "no occupancy-normalized cell may hold more than twice the mean \
             ({normalized_max} vs mean {mean:.2})"
        );
    }

    /// The frame resolution is the tilt at which a neighborhood stops preferring its
    /// own fitted plane, and it is read off the certificate rather than chosen.
    ///
    /// The three anchors are the derivation itself, not sampled behaviour: a plane
    /// that captures everything is pinned exactly; a plane whose off-plane energy has
    /// reached its in-plane energy is pinned not at all; and in between the resolution
    /// is `√(ε / (1 − ε))`, whose defining property is that a tilt of that size leaks
    /// exactly the residual the fit already carries.
    #[test]
    fn frame_resolution_is_the_tilt_that_leaks_the_measured_residual_2280() {
        let pinned = ChartCertificate {
            condition: 1.0,
            leading_singular: 1.0,
            smallest_captured_singular: 1.0,
            captured_variance_fraction: 1.0,
            min_projection_stretch: 1.0,
        };
        assert!(frame_angular_resolution(&pinned).abs() < 1e-15);

        let unpinned = ChartCertificate {
            captured_variance_fraction: 0.5,
            ..pinned
        };
        assert!((frame_angular_resolution(&unpinned) - 1.0).abs() < 1e-15);

        // A patch capturing 0.99 leaves ε = 0.01 off-plane, so a tilt of
        // sin φ = √(0.01/0.99) leaks exactly that much of the captured energy back.
        let good = ChartCertificate {
            captured_variance_fraction: 0.99,
            ..pinned
        };
        let sine = frame_angular_resolution(&good);
        let leaked = sine * sine * 0.99;
        assert!(
            (leaked - 0.01).abs() < 1e-12,
            "a tilt of the resolution must leak the measured residual: {leaked}"
        );

        // Two frames compose by angle, not by sine: the budget is sin(φ_a + φ_b),
        // strictly above either alone, and it saturates rather than folding back once
        // the pair has no resolution left.
        let pair = combined_frame_resolution(sine, sine);
        assert!(pair > sine && pair < 1.0, "{pair}");
        let expected = 2.0 * sine * (1.0 - sine * sine).sqrt();
        assert!((pair - expected).abs() < 1e-12, "{pair} vs {expected}");
        assert!((combined_frame_resolution(0.9, 0.9) - 1.0).abs() < 1e-15);
    }

    /// The gate this replaced was a numerical floor, and the difference is not
    /// cosmetic: on a torus whose patches span a real fraction of the minor circle,
    /// `|det| > 1e-6` admits a handedness from tangent planes that are within a
    /// fraction of a degree of orthogonal.
    ///
    /// The assertion is the non-vacuity of the change — there EXIST transitions the
    /// retired floor admits and the derived budget refuses — plus the reason: their
    /// largest principal angle is far closer to `π/2` than the two charts' own
    /// estimation error can resolve.
    #[test]
    fn the_derived_sign_budget_refuses_edges_the_numerical_floor_admitted_2280() {
        let z = torus(48, 20, 2.0, 0.8);
        let config = LocalAtlasConfig::balanced(z.nrows(), 2);
        let atlas = LocalAtlas::build(z.view(), config).expect("torus atlas builds");

        let refused: Vec<&ChartTransition> = atlas
            .transitions()
            .iter()
            .filter(|t| matches!(t.conditioning, TransitionConditioning::Degenerate))
            .collect();
        assert!(
            !refused.is_empty(),
            "the derived budget must bite on a torus cover of this coarseness"
        );
        for t in &refused {
            assert!(
                t.smallest_principal_cosine > 1.0e-6,
                "every refused edge must be one the retired 1e-6 floor would have \
                 admitted, else the change is invisible: {}",
                t.smallest_principal_cosine
            );
            assert!(
                t.smallest_principal_cosine <= t.sign_resolution_budget,
                "a refused edge must be refused BY the budget: {} vs {}",
                t.smallest_principal_cosine,
                t.sign_resolution_budget
            );
        }
        assert!(
            atlas.observed_signed_edges().len() < atlas.transitions().len(),
            "the observed cocycle must be a strict subcomplex here"
        );
    }

    // --- tests ------------------------------------------------------------

    /// The orientation cocycle recovers the Möbius/cylinder distinction: the
    /// cylinder atlas is orientable, the Möbius atlas is not.
    #[test]
    fn orientation_sign_recovers_mobius_vs_cylinder_2280() {
        let cyl = cylinder_strip(60, 5);
        let cyl_atlas = LocalAtlas::build(cyl.view(), LocalAtlasConfig::balanced(cyl.nrows(), 2))
            .expect("cylinder atlas must build");
        assert_eq!(
            cyl_atlas.observed_orientability(),
            AtlasOrientability::Orientable,
            "a cylinder is orientable"
        );

        let mob = mobius_strip(60, 5);
        let mob_atlas = LocalAtlas::build(mob.view(), LocalAtlasConfig::balanced(mob.nrows(), 2))
            .expect("mobius atlas must build");
        assert_eq!(
            mob_atlas.observed_orientability(),
            AtlasOrientability::NonOrientable,
            "a Möbius strip is non-orientable: the sign cocycle has a negative-holonomy loop"
        );
    }

    /// A neighborhood that spans fewer than `d` directions (all rows on a line, but
    /// `d = 2`) is rejected with the typed degenerate-patch error.
    #[test]
    fn degenerate_patch_rejected_with_typed_error_2280() {
        // 30 collinear points in 3-D: intrinsic dimension 1.
        let n = 30usize;
        let mut z = Array2::<f64>::zeros((n, 3));
        for r in 0..n {
            let t = r as f64;
            z[[r, 0]] = t;
            z[[r, 1]] = 2.0 * t;
            z[[r, 2]] = -t;
        }
        let config = LocalAtlasConfig::balanced(n, 2);
        let err = LocalAtlas::build(z.view(), config).unwrap_err();
        assert!(
            matches!(
                err,
                LocalChartError::DegeneratePatch {
                    intrinsic_dim: 2,
                    ..
                }
            ),
            "collinear data cannot yield a 2-chart; got {err}"
        );
    }

    /// A far, tightly collinear cluster placed alongside a healthy plane: FPS elects
    /// a center inside it (it is the most isolated region), that center cannot chart
    /// a 2-plane at any size — and the atlas DROPS it rather than aborting, keeping
    /// the plane's charts. The dropped center is recorded, not silently swallowed.
    #[test]
    fn atlas_drops_a_degenerate_center_and_keeps_the_rest_2280() {
        // Healthy bulk: a clean 12×12 embedded plane (144 rows).
        let plane = embedded_plane(12, 12);
        let plane_n = plane.nrows();
        // A far collinear blob along ambient axis 0, isolated at x ≈ 200, so a
        // center inside it charts only its own rank-1 neighborhood. The blob must
        // hold MORE rows than one patch, or the blob center's neighborhood would
        // reach back into the plane and certify; the premise is asserted below
        // rather than assumed, so a change to the derived patch size fails loudly
        // instead of silently voiding the test.
        let blob_n = 40usize;
        let n = plane_n + blob_n;
        let mut z = Array2::<f64>::zeros((n, 4));
        for r in 0..plane_n {
            for c in 0..4 {
                z[[r, c]] = plane[[r, c]];
            }
        }
        for t in 0..blob_n {
            z[[plane_n + t, 0]] = 200.0 + 0.02 * t as f64;
        }

        let config = LocalAtlasConfig::balanced(n, 2);
        assert!(
            config.patch_size < blob_n,
            "the fixture's premise: one patch ({}) must fit inside the blob ({blob_n}), or the \
             blob center's neighborhood reaches the plane and certifies",
            config.patch_size
        );
        let atlas =
            LocalAtlas::build(z.view(), config).expect("a mostly-healthy sample must still build");

        assert!(
            !atlas.rejected_centers().is_empty(),
            "the degenerate blob center must be recorded as dropped, not aborted"
        );
        for rejected in atlas.rejected_centers() {
            assert!(
                rejected.center >= plane_n,
                "only blob rows ({plane_n}..) are unchartable; dropped center {} is on the plane",
                rejected.center
            );
            assert!(
                matches!(rejected.reason, LocalChartError::DegeneratePatch { .. }),
                "a collinear neighborhood drops with DegeneratePatch; got {}",
                rejected.reason
            );
        }
        // The drop is user-visible: each dropped center renders a legible diagnostic
        // line, not an opaque index buried in an internal list.
        let rendered = format!("{}", atlas.rejected_centers()[0]);
        assert!(
            rendered.contains("dropped center at row") && rendered.contains("does not span"),
            "a dropped center must render a legible reason; got {rendered:?}"
        );
        assert!(
            atlas.chart_count() >= 1,
            "the plane's charts survive the dropped blob center"
        );
        // The certified charts still cover the plane bulk (well above the floor).
        let covered: BTreeSet<usize> = atlas
            .patches()
            .iter()
            .flat_map(|p| p.members.iter().copied())
            .collect();
        assert!(
            covered.len() as f64 >= MIN_ATLAS_ROW_COVERAGE * n as f64,
            "surviving charts must clear the coverage floor: {} of {n}",
            covered.len()
        );
        // Bit-identical run-to-run, dropped centers included.
        let again = LocalAtlas::build(z.view(), config).unwrap();
        assert_eq!(atlas, again, "skip-and-continue must be deterministic");
    }

    /// The honest whole-atlas refusal: when the certified charts cover only a
    /// minority of the sample (a tiny plane drowned by a large degenerate blob), the
    /// build refuses with `AtlasCoverageTooLow` rather than returning a partial atlas
    /// that silently omits most of the data.
    #[test]
    fn atlas_refuses_when_certified_coverage_falls_below_floor_2280() {
        // Tiny healthy plane: 4×4 = 16 rows.
        let plane = embedded_plane(4, 4);
        let plane_n = plane.nrows();
        // A large collinear blob (100 rows) far away along ambient axis 1: its centers
        // cannot certify a 2-chart, so the survivors cover only the 16 plane rows.
        let blob_n = 100usize;
        let n = plane_n + blob_n;
        let mut z = Array2::<f64>::zeros((n, 4));
        for r in 0..plane_n {
            for c in 0..4 {
                z[[r, c]] = plane[[r, c]];
            }
        }
        for t in 0..blob_n {
            z[[plane_n + t, 0]] = 1000.0;
            z[[plane_n + t, 1]] = t as f64;
        }

        let config = LocalAtlasConfig::balanced(n, 2);
        let err = LocalAtlas::build(z.view(), config).unwrap_err();
        match err {
            LocalChartError::AtlasCoverageTooLow {
                certified,
                covered_rows,
                total_rows,
                ..
            } => {
                assert!(certified >= 1, "the plane still certified some charts");
                assert!(
                    (covered_rows as f64) < 0.5 * total_rows as f64,
                    "coverage {covered_rows}/{total_rows} must be below the floor to refuse"
                );
            }
            other => panic!(
                "a minority-coverage sub-atlas must refuse via AtlasCoverageTooLow; got {other}"
            ),
        }
    }

    /// Regression guard: a clean sample drops nothing, so the new rejected-centers
    /// bookkeeping leaves the happy path exactly as it was.
    #[test]
    fn clean_atlas_drops_no_centers_2280() {
        let z = embedded_plane(10, 10);
        let atlas = LocalAtlas::build(z.view(), LocalAtlasConfig::balanced(z.nrows(), 2)).unwrap();
        assert!(
            atlas.rejected_centers().is_empty(),
            "a clean plane certifies every center; nothing should be dropped"
        );
        assert!(atlas.chart_count() > 0, "a clean plane yields charts");
    }

    /// Determinism doctrine: the atlas is bit-identical run-to-run.
    #[test]
    fn atlas_is_bit_identical_run_to_run_2280() {
        let z = swiss_roll(30, 6);
        let config = LocalAtlasConfig::balanced(z.nrows(), 2);
        let a = LocalAtlas::build(z.view(), config).unwrap();
        let b = LocalAtlas::build(z.view(), config).unwrap();
        assert_eq!(a, b, "local atlas must be bit-identical run-to-run");
    }

    /// The observed signed-edge diagnostic is canonical without claiming analytic
    /// or finite-sample provenance.
    #[test]
    fn observed_signed_edges_are_canonical_but_not_certificates_2280() {
        let z = spherical_band(12, 16);
        let atlas = LocalAtlas::build(z.view(), LocalAtlasConfig::balanced(z.nrows(), 2)).unwrap();
        let edges = atlas.observed_signed_edges();
        assert!(!edges.is_empty(), "a covered sphere has overlaps");
        let mut seen_overlaps = std::collections::BTreeSet::new();
        for (a, b, overlap, sign) in edges {
            assert!(a < b, "canonical undirected edge must have a < b");
            assert!(matches!(sign, -1 | 1), "sign must be ±1, got {sign}");
            assert!(a < atlas.chart_count() && b < atlas.chart_count());
            assert!(seen_overlaps.insert(overlap), "overlap ids must be unique");
        }
    }

    /// The 1-D orthogonal Procrustes factor is exactly `±1`, so a `d = 1` chart
    /// transition speaks the same orientation language as a
    /// `UnitSpeedChartTransition` sign.
    #[test]
    fn one_dimensional_transition_sign_is_plus_or_minus_one_2280() {
        // Points on a smooth 1-D curve (a helix) in 3-D.
        let n = 60usize;
        let mut z = Array2::<f64>::zeros((n, 3));
        for r in 0..n {
            let t = 0.2 * r as f64;
            z[[r, 0]] = t.cos();
            z[[r, 1]] = t.sin();
            z[[r, 2]] = 0.1 * t;
        }
        let atlas = LocalAtlas::build(z.view(), LocalAtlasConfig::balanced(n, 1)).unwrap();
        assert_eq!(atlas.intrinsic_dim(), 1);
        for t in atlas.transitions() {
            assert_eq!(t.rotation.dim(), (1, 1));
            assert!(
                (t.rotation[[0, 0]].abs() - 1.0).abs() < 1e-9,
                "a 1-D orthogonal factor is ±1, got {}",
                t.rotation[[0, 0]]
            );
            assert_eq!(t.sign as f64, t.rotation[[0, 0]].signum());
        }
    }

    #[test]
    fn determinant_reads_orthogonal_sign() {
        let mut reflection = Array2::<f64>::eye(3);
        reflection[[2, 2]] = -1.0;
        assert!((determinant(&reflection) + 1.0).abs() < 1e-12);
        let rotation = Array2::<f64>::eye(3);
        assert!((determinant(&rotation) - 1.0).abs() < 1e-12);
    }

    /// #2280 — the developing map of an exact plane is an isometry of the planted
    /// lattice. Every transition between two charts of one plane is an exact rigid
    /// motion, so gluing along the tree accumulates only rounding and every pairwise
    /// distance survives, at the rounding bound the plane's cocycle test uses. The map
    /// is also bit-identical run to run.
    #[test]
    fn developed_plane_is_an_isometry_of_the_planted_lattice_2280() {
        let points = embedded_plane(12, 12);
        let develop = || {
            LocalAtlas::build(points.view(), LocalAtlasConfig::balanced(points.nrows(), 2))
                .expect("the plane atlas builds")
                .developed_coordinates(points.view())
                .expect("a connected plane atlas develops")
        };
        let developed = develop();
        let again = develop();
        assert_eq!(developed.dim(), (points.nrows(), 2));
        for (left, right) in developed.iter().zip(again.iter()) {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "the developing map is deterministic"
            );
        }
        let mut worst = 0.0_f64;
        for a in 0..points.nrows() {
            for b in (a + 1)..points.nrows() {
                let ambient = sq_distance(points.view(), a, b).sqrt();
                let chart = sq_distance(developed.view(), a, b).sqrt();
                worst = worst.max((chart - ambient).abs());
            }
        }
        assert!(
            worst < 1e-8,
            "the developed plane must preserve every pairwise distance: worst discrepancy {worst:.3e}"
        );
    }

    /// #2280 — two copies of a sheet displaced far beyond any patch radius share no
    /// transition, so no single chart develops them, and the map refuses rather than
    /// placing the second copy arbitrarily.
    #[test]
    fn developed_coordinates_refuse_a_disconnected_cover_2280() {
        let sheet = embedded_plane(14, 14);
        let n = sheet.nrows();
        let mut z = Array2::<f64>::zeros((2 * n, 4));
        for row in 0..n {
            for column in 0..4 {
                z[[row, column]] = sheet[[row, column]];
                z[[n + row, column]] = sheet[[row, column]] + if column == 0 { 1.0e4 } else { 0.0 };
            }
        }
        let atlas = LocalAtlas::build(z.view(), LocalAtlasConfig::balanced(z.nrows(), 2))
            .expect("each copy certifies its own charts");
        let refusal = atlas
            .developed_coordinates(z.view())
            .expect_err("a two-component cover has no single developed chart");
        assert!(refusal.contains("components"), "{refusal}");
    }

    /// Worst circular distance from each row's phase to its planted loop fraction, after the
    /// best rotation and reflection of the loop. The rotation is the circular mean of the
    /// phase differences, a closed form, for each of the two reflections.
    fn worst_loop_residual(phase: ArrayView1<'_, f64>, planted: &[f64]) -> f64 {
        let tau = std::f64::consts::TAU;
        [1.0_f64, -1.0]
            .into_iter()
            .map(|orientation| {
                let (mut sine, mut cosine) = (0.0_f64, 0.0_f64);
                for (row, &truth) in planted.iter().enumerate() {
                    let angle = tau * (phase[row] - orientation * truth);
                    sine += angle.sin();
                    cosine += angle.cos();
                }
                let shift = sine.atan2(cosine) / tau;
                planted
                    .iter()
                    .enumerate()
                    .map(|(row, &truth)| {
                        let offset = phase[row] - orientation * truth - shift;
                        (offset - offset.round()).abs()
                    })
                    .fold(0.0_f64, f64::max)
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// #2906 — the holonomy of a planted circle and of a planted cylinder dictates their
    /// quotient coordinates. Every row's loop coordinate must land within half a lattice
    /// step of its planted position after the best rotation and reflection, so the loop
    /// order is recovered exactly. On the cylinder the height must also keep the planted
    /// width order in every loop column.
    #[test]
    fn holonomy_quotient_recovers_the_planted_loop_2906() {
        let n = 400usize;
        let points = crate::manifold::tests_topology_fixtures::circle(n, 2.0);
        let atlas = LocalAtlas::build(points.view(), LocalAtlasConfig::balanced(n, 1))
            .expect("the planted circle's atlas builds");
        let phase = atlas
            .holonomy_quotient_coordinates(points.view(), GraphCompressionKind::Circle)
            .expect("the planted circle's holonomy reads a period");
        let planted: Vec<f64> = (0..n).map(|row| row as f64 / n as f64).collect();
        let worst = worst_loop_residual(phase.column(0), &planted);
        assert!(
            worst < 0.5 / n as f64,
            "every circle row must land within half a lattice step of its planted phase: worst {worst:.3e}"
        );

        let (n_u, n_v) = (60usize, 14usize);
        let cylinder = cylinder_strip(n_u, n_v);
        let atlas =
            LocalAtlas::build(cylinder.view(), LocalAtlasConfig::balanced(cylinder.nrows(), 2))
                .expect("the planted cylinder's atlas builds");
        let quotient = atlas
            .holonomy_quotient_coordinates(cylinder.view(), GraphCompressionKind::Cylinder)
            .expect("the planted cylinder's holonomy reads a loop translation");
        let planted: Vec<f64> = (0..cylinder.nrows())
            .map(|row| (row / n_v) as f64 / n_u as f64)
            .collect();
        let worst = worst_loop_residual(quotient.column(0), &planted);
        assert!(
            worst < 0.5 / n_u as f64,
            "every cylinder row must land within half a loop step of its planted angle: worst {worst:.3e}"
        );
        // The cylinder's width axis is fixed, so within every loop column its height must be
        // monotone in the planted width, in one direction for the whole cylinder.
        let rising = quotient[[n_v - 1, 1]] > quotient[[0, 1]];
        let mut unordered = Vec::new();
        for column in 0..n_u {
            for iv in 1..n_v {
                let below = quotient[[column * n_v + iv - 1, 1]];
                let above = quotient[[column * n_v + iv, 1]];
                if (above > below) != rising {
                    unordered.push(format!(
                        "column {column} rows {} and {iv}: {below:.3e} then {above:.3e}",
                        iv - 1
                    ));
                }
            }
        }
        assert!(
            unordered.is_empty(),
            "the cylinder's height must recover the planted width order: {}",
            unordered.join("; ")
        );
    }

    /// Fraction of rows whose loop coordinate lands within half a lattice step (`step / 2`) of
    /// its planted loop fraction, after the best rotation and reflection of the loop. The
    /// rotation is the circular mean of the phase differences for each reflection, as in
    /// `worst_loop_residual`.
    fn loop_recovery_rate(phase: ArrayView1<'_, f64>, planted: &[f64], step: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        [1.0_f64, -1.0]
            .into_iter()
            .map(|orientation| {
                let (mut sine, mut cosine) = (0.0_f64, 0.0_f64);
                for (row, &truth) in planted.iter().enumerate() {
                    let angle = tau * (phase[row] - orientation * truth);
                    sine += angle.sin();
                    cosine += angle.cos();
                }
                let shift = sine.atan2(cosine) / tau;
                let recovered = planted
                    .iter()
                    .enumerate()
                    .filter(|&(row, &truth)| {
                        let offset = phase[row] - orientation * truth - shift;
                        (offset - offset.round()).abs() < 0.5 * step
                    })
                    .count();
                recovered as f64 / planted.len().max(1) as f64
            })
            .fold(0.0_f64, f64::max)
    }

    /// The leading `components` principal projections of the centered rows of `z`.
    fn principal_projection(z: &Array2<f64>, components: usize) -> Array2<f64> {
        let mean = z
            .mean_axis(ndarray::Axis(0))
            .expect("a planted fixture has rows");
        let centered = z - &mean;
        let (_, _, vt) = centered
            .svd(false, true)
            .expect("a planted fixture has a principal frame");
        let vt = vt.expect("the SVD returns the right frame it was asked for");
        centered.dot(&vt.slice(ndarray::s![0..components, ..]).t())
    }

    /// The phase of the leading principal pair as a fraction of the loop, the principal seed's
    /// circle and cylinder angle.
    fn principal_phase(projection: &Array2<f64>) -> Array1<f64> {
        Array1::from_iter((0..projection.nrows()).map(|row| {
            let fraction =
                projection[[row, 1]].atan2(projection[[row, 0]]) / std::f64::consts::TAU;
            fraction - fraction.floor()
        }))
    }

    /// #2906 acceptance — circle and cylinder atoms seeded from the atlas's holonomy recover
    /// their planted loop parameter at least as well as the principal-projection seed on the
    /// same fixture. Recovery is the fraction of rows whose loop coordinate lands within half a
    /// lattice step of its planted position after the best rotation and reflection. The
    /// principal seeds are the ones discovery races: the leading pair's phase, and the
    /// cylinder's phase with the third component as height. The Möbius band's read is pinned by
    /// `holonomy_seeded_mobius_band_reads_its_twist_2906`.
    #[test]
    fn holonomy_seeded_loops_recover_their_planted_parameter_2906() {
        let mut shortfalls: Vec<String> = Vec::new();
        let mut compare = |label: &str, atlas_rate: f64, principal_rate: f64| {
            eprintln!(
                "[2906-acceptance] {label}: holonomy seed {atlas_rate:.4}, principal seed {principal_rate:.4}"
            );
            if atlas_rate < principal_rate {
                shortfalls.push(format!(
                    "{label}: holonomy seed {atlas_rate:.4} below principal seed {principal_rate:.4}"
                ));
            }
        };

        let n = 400usize;
        let points = crate::manifold::tests_topology_fixtures::circle(n, 2.0);
        let planted: Vec<f64> = (0..n).map(|row| row as f64 / n as f64).collect();
        let step = 1.0 / n as f64;
        let holonomy = LocalAtlas::build(points.view(), LocalAtlasConfig::balanced(n, 1))
            .expect("the planted circle's atlas builds")
            .holonomy_quotient_coordinates(points.view(), GraphCompressionKind::Circle)
            .expect("the planted circle's holonomy reads a period");
        let principal = principal_phase(&principal_projection(&points, 2));
        compare(
            "circle(400, 2) loop",
            loop_recovery_rate(holonomy.column(0), &planted, step),
            loop_recovery_rate(principal.view(), &planted, step),
        );

        let (n_u, n_v) = (60usize, 14usize);
        let planted: Vec<f64> = (0..n_u * n_v)
            .map(|row| (row / n_v) as f64 / n_u as f64)
            .collect();
        let step = 1.0 / n_u as f64;

        let cylinder = cylinder_strip(n_u, n_v);
        let holonomy =
            LocalAtlas::build(cylinder.view(), LocalAtlasConfig::balanced(cylinder.nrows(), 2))
                .expect("the planted cylinder's atlas builds")
                .holonomy_quotient_coordinates(cylinder.view(), GraphCompressionKind::Cylinder)
                .expect("the planted cylinder's holonomy reads a loop translation");
        let principal = principal_phase(&principal_projection(&cylinder, 3));
        compare(
            "cylinder_strip(60, 14) loop",
            loop_recovery_rate(holonomy.column(0), &planted, step),
            loop_recovery_rate(principal.view(), &planted, step),
        );

        assert!(
            shortfalls.is_empty(),
            "holonomy-seeded loops must recover their planted parameter at least as well as the principal seed:\n{}",
            shortfalls.join("\n")
        );
    }

    /// #2906 — a Möbius band seeded from the atlas reads its twist through the charts'
    /// handedness, not only its base.
    ///
    /// On `mobius_strip(60, 14)` the loop coordinate must recover the planted base at least as
    /// well as the principal double-cover seed discovery races, measured as in
    /// `holonomy_seeded_loops_recover_their_planted_parameter_2906`. Within every loop column the
    /// signed width must be monotone in the planted width, read on one sheet of the double
    /// cover. The seed's coordinates identify `(s, w)` with its deck twin `(s + 1, −w)`, so a row
    /// the cut at `s = 0` separates from its column's first row carries the other
    /// representative, and its width is read negated. Job 1198326 measured column 0 of this band
    /// straddling the cut: row 0 at `(0, 0.5595)`, its neighbours at `s ≈ 0.99` with the
    /// opposite sign. The same read on
    /// `cylinder_strip(60, 14)`, whose handedness is a coboundary, must refuse. A seed that read
    /// only the base would accept the cylinder, since its loop reads exactly.
    #[test]
    fn holonomy_seeded_mobius_band_reads_its_twist_2906() {
        let (n_u, n_v) = (60usize, 14usize);
        let band = crate::manifold::tests_topology_fixtures::mobius_strip(n_u, n_v);
        let atlas = LocalAtlas::build(band.view(), LocalAtlasConfig::balanced(band.nrows(), 2))
            .expect("the planted band's atlas builds");
        let quotient = atlas
            .holonomy_quotient_coordinates(band.view(), GraphCompressionKind::MobiusStrip)
            .expect("the planted band's handedness reads a twist");
        let planted: Vec<f64> = (0..band.nrows())
            .map(|row| (row / n_v) as f64 / n_u as f64)
            .collect();
        let step = 1.0 / n_u as f64;
        let rows: Vec<usize> = (0..band.nrows()).collect();
        let principal = crate::manifold::mobius_double_cover_coords_from_projection(
            principal_projection(&band, 3).view(),
            &rows,
        )
        .expect("the principal double cover reads the planted band");
        let atlas_rate = loop_recovery_rate(quotient.column(0), &planted, step);
        let principal_rate = loop_recovery_rate(principal.column(0), &planted, step);
        let mut unordered = Vec::new();
        for column in 0..n_u {
            let reference = quotient[[column * n_v, 0]];
            let widths: Vec<f64> = (0..n_v)
                .map(|iv| {
                    let row = column * n_v + iv;
                    if (quotient[[row, 0]] - reference).abs() > 0.5 {
                        -quotient[[row, 1]]
                    } else {
                        quotient[[row, 1]]
                    }
                })
                .collect();
            let rising = widths.windows(2).all(|pair| pair[1] > pair[0]);
            let falling = widths.windows(2).all(|pair| pair[1] < pair[0]);
            if !(rising || falling) {
                unordered.push(format!("column {column}: {widths:?}"));
            }
        }
        let cylinder = cylinder_strip(n_u, n_v);
        let refusal =
            LocalAtlas::build(cylinder.view(), LocalAtlasConfig::balanced(cylinder.nrows(), 2))
                .expect("the planted cylinder's atlas builds")
                .holonomy_quotient_coordinates(cylinder.view(), GraphCompressionKind::MobiusStrip)
                .expect_err("an orientable cover has no twist to read");
        eprintln!(
            "[2906-mobius] base: holonomy seed {atlas_rate:.4}, principal double cover \
             {principal_rate:.4}; width columns out of order {}/{n_u}; cylinder: {refusal}",
            unordered.len()
        );
        assert!(
            atlas_rate >= principal_rate,
            "the band's holonomy seed must recover its base at least as well as the principal \
             double cover: {atlas_rate:.4} against {principal_rate:.4}"
        );
        assert!(
            unordered.is_empty(),
            "the band's signed width must be monotone in the planted width in every loop \
             column:\n{}",
            unordered.join("\n")
        );
        assert!(
            refusal.contains("coboundary"),
            "the cylinder must be refused for carrying no twist, not for another reason: {refusal}"
        );
    }
}
