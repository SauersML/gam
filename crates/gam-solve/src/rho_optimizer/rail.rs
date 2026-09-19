//! Which outer coordinates are railed on their box faces, and the facts a
//! certificate or a refusal records about each (#2465, #2530, #2954): moved out
//! of `run.rs` whole.

use super::newton_polish::rail_face_kind;
use super::run::OuterConfig;
use super::{coordinate_rail_margin, native_coordinate};
use crate::model_types::RailedCoordinateFact;
use ndarray::Array1;

/// Is outer coordinate `k` pinned within [`coordinate_rail_margin`] of either
/// of its own box bounds?
///
/// Factored out so the λ-block REPORT and the θ-wide certificate FACE below
/// cannot drift apart about what "railed" means for the same coordinate: they
/// differ only in which coordinates they scan, never in the test.
///
/// The margin is the shared width-capped one, so this is the exact-bound test on
/// [`rail_relaxed_bounds`]' relaxed endpoints. It used to be a flat
/// [`CERTIFICATE_RAIL_MARGIN`] while the residual projector capped the same
/// constant at a quarter-width, and a box narrower than `2 ×
/// CERTIFICATE_RAIL_MARGIN` — the raw-κ chart window on any standardised
/// feature set — was then covered end to end by its own two margin bands: every
/// κ read railed, flat κ = 0 included, and the certificate's reduced Hessian
/// lost every row it was supposed to judge (#2462).
///
/// Comparing against the relaxed endpoints rather than `|θ_k − bound|` also
/// keeps an infeasible coordinate railed. Under the absolute-value form a point
/// *outside* the box by more than the margin reported interior, which is the one
/// reading that can never be right.
pub(crate) fn outer_coordinate_is_railed(
    theta: &Array1<f64>,
    k: usize,
    config: &OuterConfig,
) -> bool {
    RailTest::evaluate(theta, k, config).is_railed()
}

/// One coordinate's rail test: the verdict together with the interval and the
/// margin it was decided against (#2465).
///
/// The predicate above computed `(lo, hi)`, derived a margin from them, compared
/// against the relaxed endpoints, and returned a bare `bool` — so `railed=[3]`
/// reached the reader with everything that produced it already destroyed, and
/// recovering the interval on #2462 took a thirteen-point seeding sweep.
///
/// #2462 made carrying it necessary rather than merely useful: the margin is now
/// [`coordinate_rail_margin`], **width-capped per coordinate**, so two
/// coordinates in the same fit can be judged railed against different margins.
/// `railed=[1, 3]` is no longer even one statement, and the flag alone cannot
/// say which band either coordinate met.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RailTest {
    /// Index into the θ vector.
    pub(crate) index: usize,
    /// The coordinate's value at the judged point.
    pub(crate) theta: f64,
    /// The interval it was tested against. `None` means the configured box does
    /// not cover this coordinate at all — which is itself the reason the verdict
    /// is `false`, a distinction the bare bool erased.
    pub(crate) box_bounds: Option<(f64, f64)>,
    /// The width-capped margin in force for THIS coordinate, from
    /// [`coordinate_rail_margin`]. Zero when the box does not cover it.
    pub(crate) margin: f64,
    /// The coordinate `index` names in the caller's native order, which is what a
    /// refusal prints: a test taken inside a canonical run is rendered through
    /// [`native_coordinate`] (#2817).
    pub(crate) native_index: usize,
}

impl RailTest {
    pub(crate) fn evaluate(theta: &Array1<f64>, k: usize, config: &OuterConfig) -> Self {
        let box_bounds = match config.model_domain_bounds.as_ref() {
            Some((lo, hi)) if k < lo.len() && k < hi.len() => Some((lo[k], hi[k])),
            Some(_) => None,
            None => Some((gam_problem::LOG_STRENGTH_MIN, gam_problem::LOG_STRENGTH_MAX)),
        };
        Self {
            // Indexed, not `get`-ed: every caller scans an index range derived
            // from `theta.len()`, so an out-of-range k is a caller bug and the
            // predicate this replaced panicked on it. Softening that to a silent
            // `false` would turn a bug into a coordinate quietly reported
            // un-railed.
            theta: theta[k],
            index: k,
            native_index: native_coordinate(config.native_coordinate_order.as_deref(), k),
            margin: box_bounds.map_or(0.0, |(lo, hi)| coordinate_rail_margin(lo, hi)),
            box_bounds,
        }
    }

    /// Pinned at or past either relaxed endpoint. This is the ONLY definition of
    /// railed in this file; both the λ-block report and the θ-wide certificate
    /// face route through it, and the relaxed-endpoint form (rather than
    /// `|θ_k − bound|`) is what keeps an infeasible coordinate railed (#2462).
    pub(crate) fn is_railed(self) -> bool {
        match self.box_bounds {
            Some((lo, hi)) => self.theta <= lo + self.margin || self.theta >= hi - self.margin,
            None => false,
        }
    }
}

impl std::fmt::Display for RailTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.box_bounds {
            Some((lo, hi)) => write!(
                f,
                "#{} theta={:.6e} box=[{:.6e}, {:.6e}] margin={:.3e} railed_at=(<={:.6e} or >={:.6e})",
                self.native_index,
                self.theta,
                lo,
                hi,
                self.margin,
                lo + self.margin,
                hi - self.margin,
            ),
            None => write!(
                f,
                "#{} theta={:.6e} box=NOT-COVERED-BY-CONFIGURED-BOUNDS",
                self.native_index, self.theta,
            ),
        }
    }
}

/// The certificate-facing facts for `indices`: the interval and margin each
/// railed coordinate was judged against (#2530).
///
/// Built from [`RailTest`], which is the single definition of railed, so the
/// certificate reports what the predicate actually decided rather than a second
/// derivation of it. A coordinate the configured box does not cover contributes
/// nothing: it was not judged against an interval, so there is no interval to
/// report.
pub(crate) fn railed_coordinate_facts(
    theta: &Array1<f64>,
    indices: &[usize],
    config: &OuterConfig,
) -> Vec<RailedCoordinateFact> {
    indices
        .iter()
        .filter_map(|&k| {
            let test = RailTest::evaluate(theta, k, config);
            test.box_bounds.map(|(lower, upper)| RailedCoordinateFact {
                index: test.index,
                theta: test.theta,
                lower,
                upper,
                margin: test.margin,
                face: rail_face_kind(config, k, test.theta >= upper - test.margin),
            })
        })
        .collect()
}

/// Render the rail tests for `indices`, so a refusal naming railed coordinates
/// also states the interval and margin each was judged against (#2465).
pub(crate) fn rail_test_summary(
    theta: &Array1<f64>,
    indices: &[usize],
    config: &OuterConfig,
) -> String {
    indices
        .iter()
        .map(|&k| RailTest::evaluate(theta, k, config).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Smoothing coordinates (leading ρ block) railed against the outer box.
///
/// This is the **report**. `OuterCriterionCertificate::lambdas_railed` indexes
/// *smoothing parameters*, and every consumer reads it that way — gam-report's
/// "λ railed" warning, the pyffi certificate surface, `is_clean`. It is
/// deliberately NOT the set the certificate reasons with; see
/// [`certificate_railed_coordinates`].
pub(crate) fn certificate_railed_lambdas(
    rho: &Array1<f64>,
    rho_dim: usize,
    config: &OuterConfig,
) -> Vec<usize> {
    (0..rho_dim.min(rho.len()))
        .filter(|&k| outer_coordinate_is_railed(rho, k, config))
        .collect()
}

/// **Every** outer coordinate railed against its own box bound — the ρ block
/// *and* the trailing non-ρ blocks that a joint search carries in the same θ
/// vector under the same box: the spatial log-κ ψ coordinates and the
/// auxiliary coordinates.
///
/// This is the set every *decision* in [`certify_outer_optimality`] must use,
/// and using the λ-only report there instead was a real defect. The active-set
/// reasoning at a box-constrained optimum is a statement about coordinates, not
/// about what a coordinate happens to parameterize:
///
/// * the second-order condition only has to hold on the **feasible tangent
///   subspace**, so `certificate_hessian_is_psd_off_railed` deletes the railed
///   rows and columns. `layout.rho_dim()` is `n_params − psi_dim`, so a ψ
///   coordinate pinned on its data-derived κ window stayed *inside* that
///   sub-block and its saturated (and, near a window edge, routinely negative)
///   curvature row decided `hessian_psd = false` for the whole fit. That is
///   precisely the failure the block's own comment says must not happen — "a
///   rail-caused indefiniteness would disable the very certificate that exists
///   to certify a railed optimum (#2299)" — it was simply never extended past
///   the ρ block;
/// * the same omission hid the coordinate from the asymptote-rail mint
///   (#2348 Inc 1), from tail-snap, from `interior_curvature_floor_clearance`,
///   from the #2155 saddle escape, and from the #2392 wrong-rail pull-back and
///   active-set reduction. A κ search whose rail is a ψ coordinate could
///   therefore never be certified *by construction*, no matter how correct its
///   gradient was;
/// * and it made the refusal message actively misleading, because
///   `project_gradient_vector` DOES project every coordinate against its own
///   bound. So `|g|` could collapse to a small `|Pg|` while `railed=[]`
///   reported nothing responsible for the collapse — the exact disagreement
///   #979 measured from the other side and could not explain.
///
/// Relaxing nothing: a coordinate near a bound whose gradient still points
/// *into* the box keeps its feasible-descent component in `|Pg|`
/// ([`project_gradient_vector`] zeros only the outward half), so a genuinely
/// unconverged interior direction is still refused.
pub(crate) fn certificate_railed_coordinates(
    theta: &Array1<f64>,
    config: &OuterConfig,
) -> Vec<usize> {
    (0..theta.len())
        .filter(|&k| outer_coordinate_is_railed(theta, k, config))
        .collect()
}
