//! Atlas-first local-chart primitive for manifold discovery (#2280).
//!
//! The intrinsic-metric seeder ([`super::intrinsic_seed`]) produces ONE GLOBAL
//! Landmark-Isomap embedding of the ambient rows, and the topology readout
//! ([`super::chart_atlas`] / [`crate::inference::atlas_nerve`]) labels that global
//! embedding. Neither is the atlas-first primitive: a manifold is not a single
//! chart but a COLLECTION of overlapping local charts, each an injective map from
//! a neighborhood into `R^d`, glued by transition maps on their overlaps. This
//! module builds exactly that object — a [`LocalAtlas`] — as a pure, deterministic
//! construction with per-chart injectivity certificates and a signed transition
//! cocycle.
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
//!   4. one TRANSITION per overlapping patch pair. Its orientation is the EXACT
//!      transition Jacobian's determinant: on the overlap the chart change is
//!      `c_to = F_toᵀ(μ_from − μ_to) + (F_toᵀ F_from) c_from + O(curvature)`, so the
//!      handedness relation of the two charts is `sign = sgn det(F_toᵀ F_from)` —
//!      a well-conditioned frame quantity (`|det| = ∏ cos θ_k` over the principal
//!      angles), not a fit to the overlap point cloud. The transition also carries
//!      the orthogonal Procrustes map `R ∈ O(d)` best aligning the two charts on
//!      their shared support WITHIN that handedness class, together with its
//!      translation, so `det R = sign` by construction and reflections are recorded
//!      rather than forced to `+1`.
//!
//! # Transition cocycle interface
//!
//! The signed transition edges are the exact substrate the #2311 holonomy readout
//! ([`crate::inference::atlas_holonomy`]) and the #2310 quotient census consume.
//! [`LocalAtlas::signed_edges`] yields `(a, b, overlap, sign)` tuples that map
//! one-to-one onto [`crate::inference::atlas_holonomy::AtlasSignedEdge::new_analytic`],
//! and [`LocalAtlas::orientability`] reproduces the SAME sign-cocycle propagation
//! as [`super::chart_atlas::ManifoldChartAtlas::orientability`], returning the
//! shared [`super::AtlasOrientability`] verdict. For a one-dimensional chart the
//! orthogonal Procrustes factor is exactly a `±1` scalar, so this `sign` coincides
//! with the `sign` field of [`super::chart_atlas::UnitSpeedChartTransition`] — the
//! one-dimensional and the general-`d` transition speak the same orientation
//! language.
//!
//! # Determinism
//!
//! Fleet law: no RNG anywhere. Centers are farthest-point (first-wins ties),
//! neighborhoods are nearest-by-distance (index tie-break), patch members and
//! shared supports are sorted by row index, and every eigen/SVD is faer's
//! deterministic solver. Same input ⇒ bit-identical atlas run-to-run.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::BTreeMap;
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
/// geometry but marked [`TransitionConfidence::Degenerate`] and kept out of the
/// sign cocycle, mirroring the analytic-vs-fitted split in [`super::chart_atlas`].
const TRANSITION_CONDITION_FLOOR_FRAC: f64 = 1.0e-6;

/// A transition's ORIENTATION is `sgn det(F_toᵀ F_from)`, and that is meaningful
/// only while the chart change is a local diffeomorphism, i.e. while the two
/// tangent planes are not orthogonal. `|det(F_toᵀ F_from)| = ∏_k cos θ_k` over the
/// principal angles between the frames, so the determinant vanishes exactly when
/// some chart direction of one patch is invisible to the other. `1e-6` is the
/// numerical "the planes are not orthogonal" floor (overlapping patches on a
/// manifold sit far above it — the sphere/Möbius/cylinder fixtures run at `|det|`
/// of order `0.5` to `1`), below which the edge is [`TransitionConfidence::Degenerate`]
/// and contributes no sign.
const FRAME_OVERLAP_DETERMINANT_FLOOR: f64 = 1.0e-6;

/// Coverage multiplier for the number of farthest-point patch centers, `⌈c·√n⌉`.
/// `√n` is the standard covering-number scaling for a fixed-radius net; `2`
/// places ~2√n charts over the sample — dense enough that neighboring patches
/// overlap once each is grown to [`LocalAtlasConfig::patch_size`], sparse enough
/// that the atlas stays sublinear in `n`.
const PATCH_COUNT_COVERAGE_MULTIPLIER: f64 = 2.0;

/// Overlap multiplier for the default patch size, `⌈c·n/patch_count⌉`. `n/count`
/// is the average Voronoi-cell occupancy of one center; growing each patch to `c`
/// times that occupancy forces neighboring cells to share rows, which is what
/// produces the overlaps a transition cocycle needs. `3` gives a robust overlap
/// without the patches degenerating into the whole sample.
const PATCH_SIZE_OVERLAP_MULTIPLIER: f64 = 3.0;

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
    /// each grown to `⌈3·n/count⌉` (floored at `2(d+1)` so a chart's PCA and every
    /// overlap's Procrustes are over-determined), with `min_overlap = d + 2`.
    #[must_use]
    pub fn balanced(n_points: usize, intrinsic_dim: usize) -> Self {
        let d = intrinsic_dim.max(1);
        let n = n_points.max(1);
        let patch_count = ((PATCH_COUNT_COVERAGE_MULTIPLIER * (n as f64).sqrt()).ceil() as usize)
            .max(d + 2)
            .min(n);
        let occupancy = (n as f64 / patch_count as f64).max(1.0);
        let patch_size = ((PATCH_SIZE_OVERLAP_MULTIPLIER * occupancy).ceil() as usize)
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
    IntrinsicDimTooLarge { intrinsic_dim: usize, ambient_dim: usize },
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
                write!(f, "local_charts: SVD failed for patch at row {center}: {detail}")
            }
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

/// Whether a transition's orientation sign is trustworthy enough to enter the
/// exact sign cocycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionConfidence {
    /// The frames' overlap determinant is non-degenerate (the chart change is a
    /// local diffeomorphism) AND the shared-support cross-covariance is well
    /// conditioned (the alignment is well posed); `sign` is an exact handedness.
    Certified,
    /// Either the two tangent planes are nearly orthogonal (so the chart change is
    /// not a diffeomorphism and no handedness exists), or the shared support did
    /// not span all `d` chart directions (so the alignment is ambiguous). The edge
    /// is geometry only and is excluded from [`LocalAtlas::signed_edges`].
    Degenerate,
}

/// The chart-to-chart map on an overlap: an orthogonal Procrustes rotation, a
/// translation, and an orientation sign.
///
/// Convention: for a shared ambient point with chart coordinates `c_from` in the
/// `from` chart and `c_to` in the `to` chart, `c_to ≈ R · c_from + t`.
///
/// The orientation is read from the EXACT transition Jacobian, `sign =
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
    /// Running index of this overlap, matching the `overlap` argument of the
    /// holonomy consumer's `AtlasSignedEdge::new_analytic` / `AtlasHolonomyEdgeId`.
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
    /// Whether `sign` is exact enough to enter the sign cocycle.
    pub confidence: TransitionConfidence,
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

/// A collection of overlapping injective local charts glued by a signed
/// transition cocycle — the atlas-first manifold primitive (#2280).
#[derive(Clone, Debug, PartialEq)]
pub struct LocalAtlas {
    intrinsic_dim: usize,
    ambient_dim: usize,
    patches: Vec<LocalPatch>,
    charts: Vec<LocalChart>,
    transitions: Vec<ChartTransition>,
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
        // neighborhood that certifies (see `certified_neighborhood_chart`).
        let mut patches: Vec<LocalPatch> = Vec::with_capacity(centers.len());
        let mut charts: Vec<LocalChart> = Vec::with_capacity(centers.len());
        for &center in &centers {
            let (members, chart) = certified_neighborhood_chart(z, center, patch_size, d)?;
            patches.push(LocalPatch { center, members });
            charts.push(chart);
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
                let transition =
                    build_transition(&charts, &patches, i, j, overlap_id, &shared);
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

    /// The certified signed transition edges as `(a, b, overlap, sign)`, ready for
    /// the #2311 holonomy consumer's
    /// [`crate::inference::atlas_holonomy::AtlasSignedEdge::new_analytic`].
    /// Degenerate transitions are intentionally absent, exactly as fitted sphere
    /// seams are absent from [`super::chart_atlas`]'s exact cocycle.
    #[must_use]
    pub fn signed_edges(&self) -> Vec<(usize, usize, usize, i8)> {
        self.transitions
            .iter()
            .filter(|t| matches!(t.confidence, TransitionConfidence::Certified))
            .map(|t| (t.from_patch, t.to_patch, t.overlap_id, t.sign))
            .collect()
    }

    /// Read orientability from the certified sign cocycle, reusing the SAME
    /// verdict type and propagation as
    /// [`super::chart_atlas::ManifoldChartAtlas::orientability`]: propagate a local
    /// orientation across certified edges; a contradictory revisit is a
    /// negative-holonomy cycle (the Möbius obstruction). Disconnected components
    /// are each resolved independently. Always `Some` here because every certified
    /// edge carries an exact sign.
    #[must_use]
    pub fn orientability(&self) -> Option<AtlasOrientability> {
        let mut orientation: BTreeMap<usize, i8> = BTreeMap::new();
        // Adjacency over certified signed edges only.
        let mut adj: BTreeMap<usize, Vec<(usize, i8)>> = BTreeMap::new();
        for (a, b, _, sign) in self.signed_edges() {
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
                                return Some(AtlasOrientability::NonOrientable);
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
        Some(AtlasOrientability::Orientable)
    }

    /// The directed rotation `from → to` (`Some` iff the two patches share a
    /// registered transition), together with its orientation sign. Stored
    /// transitions are canonical (`from < to`); the reverse direction returns the
    /// inverse rotation `Rᵀ` (an orthogonal inverse), whose determinant — and thus
    /// the sign — is unchanged.
    #[must_use]
    pub fn directed_rotation(&self, from: usize, to: usize) -> Option<(Array2<f64>, i8)> {
        self.transitions.iter().find_map(|t| {
            if t.from_patch == from && t.to_patch == to {
                Some((t.rotation.clone(), t.sign))
            } else if t.from_patch == to && t.to_patch == from {
                Some((transpose(&t.rotation), t.sign))
            } else {
                None
            }
        })
    }

    /// Frobenius defect of the transition cocycle around the triangle `a→b→c→a`:
    /// `‖R_ca · R_bc · R_ab − I‖_F`. `Some` iff all three pairwise transitions
    /// exist. A closed cocycle (contractible loop, coherent frames) returns ~0.
    #[must_use]
    pub fn triangle_cocycle_defect(&self, a: usize, b: usize, c: usize) -> Option<f64> {
        let (r_ab, _) = self.directed_rotation(a, b)?;
        let (r_bc, _) = self.directed_rotation(b, c)?;
        let (r_ca, _) = self.directed_rotation(c, a)?;
        let product = matmul(&r_ca, &matmul(&r_bc, &r_ab));
        let d = product.nrows();
        let mut acc = 0.0;
        for i in 0..d {
            for j in 0..d {
                let target = if i == j { 1.0 } else { 0.0 };
                let diff = product[[i, j]] - target;
                acc += diff * diff;
            }
        }
        Some(acc.sqrt())
    }

    /// Product of the orientation signs around the triangle `a→b→c→a`. `Some` iff
    /// all three pairwise transitions exist. Gauge-invariant: `+1` on an orientable
    /// loop, `-1` on a Möbius loop.
    #[must_use]
    pub fn triangle_sign_product(&self, a: usize, b: usize, c: usize) -> Option<i8> {
        let (_, s_ab) = self.directed_rotation(a, b)?;
        let (_, s_bc) = self.directed_rotation(b, c)?;
        let (_, s_ca) = self.directed_rotation(c, a)?;
        Some(s_ab * s_bc * s_ca)
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
        min_projection_stretch: if min_stretch.is_finite() { min_stretch } else { 1.0 },
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

    // Orthogonal Procrustes: minimize ‖C_to − R C_from‖_F over R ∈ O(d).
    // M = C_to C_fromᵀ (d × d); SVD M = U S Vᵀ; R = U Vᵀ (reflections ALLOWED, so
    // det R records a genuine handedness flip).
    let m_mat = c_to.dot(&c_from.t());
    let (rotation, sign, confidence) = match m_mat.svd(true, true) {
        Ok((Some(u), sv, Some(vt))) => {
            let r = u.dot(&vt);
            let sign_val: i8 = if determinant(&r) >= 0.0 { 1 } else { -1 };
            let leading = sv.first().copied().unwrap_or(0.0);
            let smallest = sv.get(d.saturating_sub(1)).copied().unwrap_or(0.0);
            let confidence = if leading > 0.0
                && smallest > TRANSITION_CONDITION_FLOOR_FRAC * leading
            {
                TransitionConfidence::Certified
            } else {
                TransitionConfidence::Degenerate
            };
            (r, sign_val, confidence)
        }
        // A failed / rank-empty SVD leaves the alignment unresolved: identity
        // rotation, degenerate confidence (excluded from the sign cocycle).
        _ => (
            Array2::<f64>::eye(d),
            1,
            TransitionConfidence::Degenerate,
        ),
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
        confidence,
    }
}

/// General square-matrix multiply `A · B`.
fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

/// Transpose of a square matrix.
fn transpose(a: &Array2<f64>) -> Array2<f64> {
    a.t().to_owned()
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

    // --- fixtures ---------------------------------------------------------

    /// A swiss roll: a flat 2-D sheet `(t, h)` rolled into ambient 3-D. Folded in
    /// the ambient metric (ambient-near points can be geodesically far), yet
    /// intrinsically flat, so the transition cocycle around a contractible triangle
    /// must close.
    fn swiss_roll(n_t: usize, n_h: usize) -> Array2<f64> {
        let n = n_t * n_h;
        let mut z = Array2::<f64>::zeros((n, 3));
        let mut r = 0usize;
        for it in 0..n_t {
            // t over ~1.5 turns.
            let t = 1.0 + 3.0 * std::f64::consts::PI * (it as f64) / (n_t as f64 - 1.0);
            for ih in 0..n_h {
                let h = 2.0 * (ih as f64) / (n_h as f64 - 1.0);
                z[[r, 0]] = t * t.cos();
                z[[r, 1]] = t * t.sin();
                z[[r, 2]] = h;
                r += 1;
            }
        }
        z
    }

    /// A flat 2-D lattice embedded isometrically into 4-D by a fixed orthonormal
    /// pair of ambient directions. Local PCA recovers the exact plane, so every
    /// transition is an exact isometry and the cocycle closes to rounding — the
    /// sharp cocycle-closure fixture.
    fn embedded_plane(n_x: usize, n_y: usize) -> Array2<f64> {
        // Two orthonormal ambient directions in R^4.
        let u = [0.5, 0.5, 0.5, 0.5];
        let v = [0.5, -0.5, 0.5, -0.5];
        let n = n_x * n_y;
        let mut z = Array2::<f64>::zeros((n, 4));
        let mut r = 0usize;
        for ix in 0..n_x {
            for iy in 0..n_y {
                let a = ix as f64;
                let b = iy as f64;
                for c in 0..4 {
                    z[[r, c]] = a * u[c] + b * v[c];
                }
                r += 1;
            }
        }
        z
    }

    /// Points on the unit 2-sphere on a lat/lon grid (poles excluded).
    fn sphere(n_lat: usize, n_lon: usize) -> Array2<f64> {
        let n = n_lat * n_lon;
        let mut z = Array2::<f64>::zeros((n, 3));
        let mut r = 0usize;
        for i in 0..n_lat {
            let lat = -1.2 + 2.4 * (i as f64) / (n_lat as f64 - 1.0); // in (−π/2, π/2)
            for j in 0..n_lon {
                let lon = std::f64::consts::TAU * (j as f64) / (n_lon as f64);
                z[[r, 0]] = lat.cos() * lon.cos();
                z[[r, 1]] = lat.cos() * lon.sin();
                z[[r, 2]] = lat.sin();
                r += 1;
            }
        }
        z
    }

    /// A cylinder strip: loop coordinate `u`, width `v` on a FIXED ambient axis, so
    /// the width frame never flips — orientable.
    fn cylinder_strip(n_u: usize, n_v: usize) -> Array2<f64> {
        let n = n_u * n_v;
        let mut z = Array2::<f64>::zeros((n, 3));
        let mut r = 0usize;
        for iu in 0..n_u {
            let u = std::f64::consts::TAU * (iu as f64) / (n_u as f64);
            for iv in 0..n_v {
                let v = -0.4 + 0.8 * (iv as f64) / (n_v as f64 - 1.0);
                z[[r, 0]] = 2.0 * u.cos();
                z[[r, 1]] = 2.0 * u.sin();
                z[[r, 2]] = v;
                r += 1;
            }
        }
        z
    }

    /// A Möbius strip: the standard half-twist embedding, so the width frame
    /// reverses once around the loop — non-orientable.
    fn mobius_strip(n_u: usize, n_v: usize) -> Array2<f64> {
        let n = n_u * n_v;
        let mut z = Array2::<f64>::zeros((n, 3));
        let mut r = 0usize;
        for iu in 0..n_u {
            let u = std::f64::consts::TAU * (iu as f64) / (n_u as f64);
            for iv in 0..n_v {
                let v = -0.4 + 0.8 * (iv as f64) / (n_v as f64 - 1.0);
                let radial = 2.0 + v * (u / 2.0).cos();
                z[[r, 0]] = radial * u.cos();
                z[[r, 1]] = radial * u.sin();
                z[[r, 2]] = v * (u / 2.0).sin();
                r += 1;
            }
        }
        z
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
        assert_eq!(atlas.orientability(), Some(AtlasOrientability::Orientable));
    }

    /// On a sphere every chart is an injective local tangent map and the atlas is
    /// orientable with a closing triple cocycle.
    #[test]
    fn sphere_charts_injective_and_orientable_2280() {
        let z = sphere(14, 20);
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
            atlas.orientability(),
            Some(AtlasOrientability::Orientable),
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
            cyl_atlas.orientability(),
            Some(AtlasOrientability::Orientable),
            "a cylinder is orientable"
        );

        let mob = mobius_strip(60, 5);
        let mob_atlas = LocalAtlas::build(mob.view(), LocalAtlasConfig::balanced(mob.nrows(), 2))
            .expect("mobius atlas must build");
        assert_eq!(
            mob_atlas.orientability(),
            Some(AtlasOrientability::NonOrientable),
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
            matches!(err, LocalChartError::DegeneratePatch { intrinsic_dim: 2, .. }),
            "collinear data cannot yield a 2-chart; got {err}"
        );
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

    /// The signed-edge export matches the holonomy consumer's contract: `(a, b,
    /// overlap, sign)` with `a < b`, unique overlap ids, and `sign ∈ {±1}`.
    #[test]
    fn signed_edges_match_holonomy_consumer_contract_2280() {
        let z = sphere(12, 16);
        let atlas =
            LocalAtlas::build(z.view(), LocalAtlasConfig::balanced(z.nrows(), 2)).unwrap();
        let edges = atlas.signed_edges();
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
        let atlas =
            LocalAtlas::build(z.view(), LocalAtlasConfig::balanced(n, 1)).unwrap();
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
}
