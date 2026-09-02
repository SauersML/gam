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
//!   1. deterministic farthest-point CENTERS over the ambient rows (reusing
//!      [`super::intrinsic_seed::farthest_point_landmarks`], the same greedy
//!      coverage pattern the intrinsic seeder uses), so the patches tile the
//!      manifold with sublinearly many charts;
//!   2. one PATCH per center — its nearest ambient rows, at most `patch_size` of
//!      them — sized so neighboring patches OVERLAP (controlled by
//!      [`LocalAtlasConfig`]). The neighborhood is the LARGEST prefix of the
//!      distance order that yields a certified chart: a local-PCA frame is a
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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use gam_linalg::faer_ndarray::FaerSvd;

use super::AtlasOrientability;
use super::intrinsic_seed::farthest_point_landmarks;

/// The `d`-th captured singular value must exceed this fraction of the leading
/// singular value for the patch to count as a genuine `d`-dimensional chart.
/// Below it the neighborhood spans fewer than `d` directions and the local-PCA
/// frame's `d`-th axis is numerical noise; the patch is rejected rather than
/// given a rank-deficient frame. `1e-8` sits well above the f64 SVD noise on a
/// normalized block and far below any axis carrying real neighborhood spread.
const CHART_RANK_FLOOR_FRAC: f64 = 1.0e-8;

/// A chart is injective on its neighborhood iff no two distinct rows project to
/// the same coordinate. The orthogonal split
/// `‖x_p − x_q‖² = ‖c_p − c_q‖² + ‖r_p − r_q‖²` makes the projected squared
/// distance exact, so injectivity is `min_{p≠q} ‖c_p − c_q‖² > 0`. To reject a
/// chart that all but collapses a pair (the off-frame residual erasing almost the
/// whole ambient separation), require the smallest projected squared distance to
/// retain at least this fraction of the smallest ambient squared distance. `1e-6`
/// is a numerical "did not collapse" floor, not a quality bar — the realized
/// stretch is surfaced on [`ChartCertificate::min_projection_stretch`] for
/// consumers that want a sharper geometric threshold.
const CHART_INJECTIVITY_FLOOR_FRAC: f64 = 1.0e-6;

/// A transition's Procrustes ALIGNMENT is well posed only when the cross-covariance
/// `M = C_to C_fromᵀ` is well conditioned: its smallest singular value clears this
/// fraction of its largest. A near-singular `M` means the shared support does not
/// span all `d` chart directions, so the rotation that best fits the overlap point
/// cloud is ambiguous in the unspanned direction. Such an edge is retained as
/// geometry but marked [`TransitionConditioning::Degenerate`] and kept out of the
/// sign cocycle, mirroring the analytic-vs-fitted split in [`super::chart_atlas`].
const TRANSITION_CONDITION_FLOOR_FRAC: f64 = 1.0e-6;

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
    /// Number of farthest-point patch centers.
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
    /// The chart projection collapses two distinct neighborhood rows onto (nearly)
    /// the same coordinate — it is not injective on its own support — and stayed
    /// non-injective down to the smallest admissible neighborhood.
    NonInjectiveChart {
        center: usize,
        min_projected_sq_distance: f64,
        min_ambient_sq_distance: f64,
    },
    /// The SVD backing a chart or a transition failed to converge.
    SvdFailure { center: usize, detail: String },
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
                min_projected_sq_distance,
                min_ambient_sq_distance,
            } => write!(
                f,
                "local_charts: chart at row {center} is not injective on its neighborhood \
                 (min projected sq distance {min_projected_sq_distance:.3e} vs min ambient \
                 {min_ambient_sq_distance:.3e})"
            ),
            Self::SvdFailure { center, detail } => {
                write!(
                    f,
                    "local_charts: SVD failed for patch at row {center}: {detail}"
                )
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
    /// The smallest bi-Lipschitz LOWER stretch of the chart map over neighborhood
    /// pairs, `min_{p≠q} ‖c_p − c_q‖ / ‖x_p − x_q‖ ∈ (0, 1]`. Strictly positive
    /// certifies injectivity on the support; near `1` certifies a near-isometric
    /// chart.
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

/// A pair of charts flagged as a co-collapse candidate: they cover nearly the
/// same region AND glue by one coherent transition, so they represent ONE
/// manifold redundantly rather than two distinct structures.
///
/// This is the detection half of the #2280 gluing-consistency mechanism, which is
/// also the #2080 model-level anti-collapse readout: "two atoms decoding the same
/// region either agree on a transition (one manifold → merge) or provably do not
/// (genuinely distinct structure)." A confinement/ownership force needs this
/// detector — you cannot confine what you cannot name — and the atlas already
/// carries both signals (shared support and the fitted transition residual), so
/// the readout is a pure function of the built atlas with no fit in the loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoCollapseCandidate {
    /// Source patch index (`< to_patch`, the canonical undirected orientation).
    pub from_patch: usize,
    /// Target patch index.
    pub to_patch: usize,
    /// The overlap component id of the underlying transition.
    pub overlap_id: usize,
    /// `|shared| / min(|members_from|, |members_to|) ∈ (0, 1]`: how completely the
    /// two patches cover the same rows. Near `1` ⇒ duplicate coverage; a healthy
    /// adjacent overlap shares only a band and stays well below `1`.
    pub mutual_coverage: f64,
    /// The transition's Procrustes residual on the shared support. Near `0` ⇒ the
    /// two charts glue by one exact isometry ⇒ the same manifold; a large residual
    /// is genuinely distinct structure sharing the region, NOT a collapse.
    pub transition_residual: f64,
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
}

impl LocalAtlas {
    /// Construct the local-chart atlas from ambient rows `z` under `config`.
    ///
    /// Pure, deterministic construction: farthest-point centers, nearest-row
    /// patches, local-PCA charts (each certified injective or rejected with a
    /// typed error), and orthogonal Procrustes transitions on every overlap.
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

        // (1) deterministic farthest-point centers (reused intrinsic-seed machinery).
        let centers = farthest_point_landmarks(z, config.patch_count.max(1).min(n));

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
            match certified_neighborhood_chart(z, center, patch_size, d) {
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

    /// Numerically well-conditioned observed transition signs as
    /// `(a, b, overlap, sign)`.
    ///
    /// This is a diagnostic of the fitted atlas, not an exact or finite-sample
    /// sign certificate. It deliberately returns ordinary tuples that no
    /// authoritative holonomy constructor accepts as analytic provenance.
    #[must_use]
    pub fn observed_signed_edges(&self) -> Vec<(usize, usize, usize, i8)> {
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

/// Every ambient row ordered by ascending Euclidean distance to `center`, ties
/// broken by ascending row index (a TOTAL order, so the ordering is independent of
/// the sort's stability and identical run-to-run). `center` itself is first.
fn distance_order(z: ArrayView2<'_, f64>, center: usize) -> Vec<usize> {
    let n = z.nrows();
    let mut scored: Vec<(f64, usize)> = (0..n).map(|r| (sq_distance(z, center, r), r)).collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, r)| r).collect()
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
/// use a neighborhood the tangent plane actually fits: drop the farthest member and
/// retry, down to the `2(d + 1)` over-determination floor. Prefixes of the distance
/// order are nested, so this walks a deterministic chain of shrinking balls and
/// returns the first — hence largest — one that certifies. If none does, the last
/// (smallest-neighborhood) typed error is returned: genuinely `d`-degenerate data
/// (a line charted at `d = 2`) still fails, at every size, with `DegeneratePatch`.
fn certified_neighborhood_chart(
    z: ArrayView2<'_, f64>,
    center: usize,
    patch_size: usize,
    d: usize,
) -> Result<(Vec<usize>, LocalChart), LocalChartError> {
    let order = distance_order(z, center);
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
    let mut acc = 0.0;
    for c in 0..z.ncols() {
        let diff = z[[a, c]] - z[[b, c]];
        acc += diff * diff;
    }
    acc
}

/// Intersection of two ascending-sorted row lists, ascending.
fn sorted_intersection(a: &[usize], b: &[usize]) -> Vec<usize> {
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
    if !(leading > 0.0) || smallest_captured <= CHART_RANK_FLOOR_FRAC * leading {
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

    // Injectivity certificate: smallest projected pairwise sq distance vs smallest
    // ambient pairwise sq distance, and the smallest bi-Lipschitz lower stretch.
    let mut min_proj_sq = f64::INFINITY;
    let mut min_amb_sq = f64::INFINITY;
    let mut min_stretch = f64::INFINITY;
    for a in 0..m {
        for b in (a + 1)..m {
            let mut amb = 0.0;
            for c in 0..p {
                let diff = centered[[a, c]] - centered[[b, c]];
                amb += diff * diff;
            }
            let mut proj = 0.0;
            for ax in 0..d {
                let diff = coords[[a, ax]] - coords[[b, ax]];
                proj += diff * diff;
            }
            if amb < min_amb_sq {
                min_amb_sq = amb;
            }
            if proj < min_proj_sq {
                min_proj_sq = proj;
            }
            if amb > 0.0 {
                let stretch = (proj / amb).sqrt();
                if stretch < min_stretch {
                    min_stretch = stretch;
                }
            }
        }
    }
    if !min_amb_sq.is_finite() {
        // A single-row patch cannot certify injectivity; the patch-size floor
        // (≥ d + 1) prevents this, but guard defensively.
        min_amb_sq = 0.0;
        min_proj_sq = 0.0;
        min_stretch = 1.0;
    }
    if min_proj_sq <= CHART_INJECTIVITY_FLOOR_FRAC * min_amb_sq && min_amb_sq > 0.0 {
        return Err(LocalChartError::NonInjectiveChart {
            center,
            min_projected_sq_distance: min_proj_sq,
            min_ambient_sq_distance: min_amb_sq,
        });
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

    // ORIENTATION: the exact transition Jacobian. On the overlap the chart change is
    //     c_to = F_toᵀ(μ_from − μ_to) + A · c_from + O(curvature),   A = F_toᵀ F_from,
    // so the handedness relation of the two charts is sgn det A, with
    // |det A| = ∏_k cos θ_k over the principal angles between the tangent planes.
    // This is a property of the two FRAMES: unlike a Procrustes fit to the shared
    // point cloud, it does not degrade when the overlap is small or elongated (where
    // a reflection fits the points exactly as well as a rotation, at the same
    // residual, and the fitted det is a coin flip).
    let a_mat = frame_overlap(&chart_j.frame, &chart_i.frame);
    let det_a = determinant(&a_mat);
    let sign: i8 = if det_a >= 0.0 { 1 } else { -1 };

    // RESOLVABILITY of that sign. The singular values of `A` are the cosines of the
    // principal angles, so `det A` changes sign exactly when some `θ_k` crosses
    // `π/2` — the handedness is a fact about the manifold only while every
    // `cos θ_k` clears what the two frames' own estimation error could manufacture.
    // `σ_min(A) = min_k cos θ_k` is the sharp quantity: `|det A| = ∏_k cos θ_k ≤
    // σ_min(A)`, so gating the DETERMINANT against the same budget is the
    // conservative relaxation, and at `d > 1` it refuses edges whose every angle is
    // individually well resolved merely because the product of several cosines is
    // small. The budget itself is derived from the two charts' certificates
    // ([`frame_angular_resolution`]) rather than being a numerical floor: a floor
    // asks "is this determinant distinguishable from zero in f64", which is a
    // question about arithmetic, and the question the sign needs answered is
    // whether it is distinguishable from zero given how well the two local PCAs
    // pinned their tangent planes.
    let sign_resolution_budget = combined_frame_resolution(
        frame_angular_resolution(&chart_i.certificate),
        frame_angular_resolution(&chart_j.certificate),
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
    let frame_nondegenerate = smallest_principal_cosine > sign_resolution_budget;

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
            let well_posed = leading > 0.0 && smallest > TRANSITION_CONDITION_FLOOR_FRAC * leading;
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
        smallest_principal_cosine,
        sign_resolution_budget,
        conditioning,
    }
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

    use crate::manifold::tests_topology_fixtures::{
        cylinder_strip, embedded_plane, mobius_strip, spherical_band, swiss_roll, torus,
    };

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

    /// Find any triple of patches that pairwise share a registered transition and
    /// have a non-empty triple intersection (a genuine triple overlap).
    fn genuine_triple(atlas: &LocalAtlas) -> Option<(usize, usize, usize)> {
        let k = atlas.chart_count();
        for a in 0..k {
            for b in (a + 1)..k {
                if atlas.directed_rotation(a, b).is_none() {
                    continue;
                }
                for c in (b + 1)..k {
                    if atlas.directed_rotation(b, c).is_some()
                        && atlas.directed_rotation(a, c).is_some()
                    {
                        let ab = sorted_intersection(
                            &atlas.patches()[a].members,
                            &atlas.patches()[b].members,
                        );
                        let triple = sorted_intersection(&ab, &atlas.patches()[c].members);
                        if !triple.is_empty() {
                            return Some((a, b, c));
                        }
                    }
                }
            }
        }
        None
    }

    // --- tests ------------------------------------------------------------

    /// Every chart on a swiss roll is injective on its neighborhood (build
    /// succeeds and every certificate has a strictly positive lower stretch), and
    /// the transition cocycle closes on a genuine triple overlap.
    #[test]
    fn swiss_roll_charts_injective_and_cocycle_closes_2280() {
        let z = swiss_roll(40, 8);
        let config = LocalAtlasConfig::balanced(z.nrows(), 2);
        let atlas = LocalAtlas::build(z.view(), config).expect("swiss roll atlas must build");
        assert!(atlas.chart_count() >= 3, "need several charts to overlap");
        for chart in atlas.charts() {
            assert!(
                chart.certificate.min_projection_stretch > 0.0,
                "chart at row {} must be injective (positive lower stretch)",
                chart.center
            );
            assert!(
                chart.certificate.captured_variance_fraction > 0.7,
                "a swiss-roll patch is mostly planar (above the 2/3 isotropic baseline); \
                 captured var {} too low",
                chart.certificate.captured_variance_fraction
            );
        }
        let (a, b, c) = genuine_triple(&atlas).expect("swiss roll must have a triple overlap");
        let defect = atlas.triangle_cocycle_defect(a, b, c).unwrap();
        assert!(
            defect < 0.5,
            "flat (contractible) triple cocycle must nearly close, defect {defect:.3e}"
        );
        assert_eq!(
            atlas.triangle_sign_product(a, b, c),
            Some(1),
            "an orientable triple has sign product +1"
        );
    }

    /// On an isometrically embedded flat plane the charts are exact and the
    /// transition cocycle closes to rounding — the sharp closure assertion.
    #[test]
    fn embedded_plane_cocycle_closes_to_rounding_2280() {
        let z = embedded_plane(12, 12);
        let config = LocalAtlasConfig::balanced(z.nrows(), 2);
        let atlas = LocalAtlas::build(z.view(), config).expect("plane atlas must build");
        for chart in atlas.charts() {
            assert!(
                chart.certificate.captured_variance_fraction > 1.0 - 1e-9,
                "an exact plane patch captures all variance"
            );
            assert!(
                (chart.certificate.min_projection_stretch - 1.0).abs() < 1e-6,
                "an isometric chart has unit lower stretch"
            );
        }
        let (a, b, c) = genuine_triple(&atlas).expect("plane must have a triple overlap");
        let defect = atlas.triangle_cocycle_defect(a, b, c).unwrap();
        assert!(
            defect < 1e-8,
            "exact-plane triple cocycle must close to rounding, defect {defect:.3e}"
        );
        assert_eq!(
            atlas.observed_orientability(),
            AtlasOrientability::Orientable
        );
    }

    /// On a sphere every chart is an injective local tangent map and the atlas is
    /// orientable with a closing triple cocycle.
    #[test]
    fn sphere_charts_injective_and_orientable_2280() {
        let z = spherical_band(14, 20);
        let config = LocalAtlasConfig::balanced(z.nrows(), 2);
        let atlas = LocalAtlas::build(z.view(), config).expect("sphere atlas must build");
        for chart in atlas.charts() {
            assert!(
                chart.certificate.min_projection_stretch > 0.0,
                "sphere chart at row {} must be injective",
                chart.center
            );
        }
        assert_eq!(
            atlas.observed_orientability(),
            AtlasOrientability::Orientable,
            "the sphere is orientable"
        );
        let (a, b, c) = genuine_triple(&atlas).expect("sphere must have a triple overlap");
        assert_eq!(
            atlas.triangle_sign_product(a, b, c),
            Some(1),
            "an orientable triple has sign product +1"
        );
        let defect = atlas.triangle_cocycle_defect(a, b, c).unwrap();
        assert!(
            defect < 0.75,
            "a small sphere triple cocycle nearly closes, defect {defect:.3e}"
        );
    }

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

    /// #2280/#2080 co-collapse detector: two charts covering the SAME region and
    /// gluing by one exact isometry are flagged as one redundant manifold. Forced
    /// by an explicit config whose patches each grow to the whole flat cloud, so
    /// the two charts share every row (mutual coverage 1) at ~zero residual.
    #[test]
    fn co_collapse_flags_duplicate_charts_2280() {
        let z = embedded_plane(6, 6); // 36 exactly-planar points
        let config = LocalAtlasConfig {
            intrinsic_dim: 2,
            patch_count: 2,
            patch_size: z.nrows(),
            min_overlap: 3,
        };
        let atlas = LocalAtlas::build(z.view(), config).expect("duplicate-patch atlas must build");
        assert_eq!(
            atlas.chart_count(),
            2,
            "config pins two whole-cloud patches"
        );
        let candidates = atlas.co_collapse_candidates(0.9, 1.0e-6);
        assert_eq!(
            candidates.len(),
            1,
            "the two identical-support charts are one co-collapse candidate"
        );
        let candidate = candidates[0];
        assert!(
            candidate.mutual_coverage > 0.99,
            "duplicate charts must have near-total mutual coverage, got {}",
            candidate.mutual_coverage
        );
        assert!(
            candidate.transition_residual < 1.0e-6,
            "an exact-isometry glue must have ~zero transition residual, got {}",
            candidate.transition_residual
        );
    }

    /// The detector honors BOTH thresholds monotonically: a coverage bar above 1
    /// admits nothing (coverage is at most 1), while the fully permissive query
    /// flags exactly the well-conditioned transitions — the deterministic bracket
    /// on the gate logic.
    #[test]
    fn co_collapse_thresholds_bracket_the_gate_2280() {
        let z = embedded_plane(6, 6);
        let config = LocalAtlasConfig {
            intrinsic_dim: 2,
            patch_count: 2,
            patch_size: z.nrows(),
            min_overlap: 3,
        };
        let atlas = LocalAtlas::build(z.view(), config).unwrap();
        assert!(
            atlas.co_collapse_candidates(1.5, f64::INFINITY).is_empty(),
            "no pair can exceed a coverage bar above 1"
        );
        let permissive = atlas.co_collapse_candidates(0.0, f64::INFINITY);
        let well_conditioned = atlas.observed_signed_edges().len();
        assert_eq!(
            permissive.len(),
            well_conditioned,
            "the fully permissive query flags exactly the well-conditioned transitions"
        );
    }

    /// On a genuine 2-D swiss roll the detector is SELECTIVE, not a rubber stamp on
    /// every overlap: tightening either the coverage bar (healthy adjacency shares
    /// only a band, well below duplicate) or the residual bar (curved patches glue
    /// with a nonzero sagitta residual) strictly shrinks the flagged set. This is
    /// the discrimination the anti-collapse force rests on — redundant coverage is
    /// flagged, honest adjacency and distinct co-located structure are spared.
    #[test]
    fn co_collapse_spares_healthy_swiss_roll_atlas_2280() {
        let z = swiss_roll(40, 8);
        let atlas = LocalAtlas::build(z.view(), LocalAtlasConfig::balanced(z.nrows(), 2)).unwrap();
        let permissive = atlas.co_collapse_candidates(0.0, f64::INFINITY).len();
        assert!(
            permissive > 0,
            "the swiss-roll atlas has overlapping charts"
        );
        // Coverage gate discriminates: not every overlap is near-duplicate.
        let strict_coverage = atlas.co_collapse_candidates(0.98, f64::INFINITY).len();
        assert!(
            strict_coverage < permissive,
            "a strict coverage bar must spare healthy partial-overlap adjacency \
             (strict {strict_coverage} vs permissive {permissive})"
        );
        // Residual gate discriminates: curved patches do not glue at exactly zero.
        let strict_residual = atlas.co_collapse_candidates(0.0, 0.0).len();
        assert!(
            strict_residual < permissive,
            "a strict residual bar must spare curved (nonzero-residual) glues \
             (strict {strict_residual} vs permissive {permissive})"
        );
    }
}
