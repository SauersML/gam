//! Structured capture of outer-objective evidence for integration tests.
//!
//! Two channels serve tests that grade the outer criterion's derivatives. Both
//! are disabled by default and thread-local, so a parallel integration test can
//! neither consume nor overwrite another test's evidence, and an ordinary fit
//! pays one thread-local read per publication site.
//!
//! - The outer-seed probe (#2460, #2765). A test registers an observer with
//!   [`observe_next_outer_seed`]. At the first seed with enough ψ axes the
//!   generic outer runner lends it an [`OuterSeedProbe`], which evaluates the
//!   real objective at any θ from that seed's own inner start. Every evaluation
//!   returns analytic evidence only: the criterion value, its analytic gradient,
//!   the scalar criterion components, and the selected coefficient mode with its
//!   analytic mode response. A test that compares that evidence with a finite
//!   difference forms the difference itself, so the production tree differences
//!   nothing (SPEC rule 2, #2901).
//! - The ρ-block audit (#2454), below.
//!
//! Tests consume typed arrays rather than scraping formatted production logs.

use crate::estimate::EstimationError;
use ndarray::{Array1, Array2};
use std::cell::RefCell;

/// Which derivative order one probe evaluation returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterSeedOrder {
    /// The criterion value alone.
    Value,
    /// The criterion value and its analytic θ-gradient.
    ValueAndGradient,
}

/// The seed a probe is lent at, and the box every evaluation it makes has to
/// stay inside.
///
/// `seed` is the complete outer coordinate `θ = (ρ ‖ ψ)`, and `rho_dim` locates
/// the ψ block inside it.
#[derive(Clone, Debug)]
pub struct OuterSeedLayout {
    pub seed: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub rho_dim: usize,
    pub psi_dim: usize,
}

/// Analytic evidence from one probe evaluation at one θ.
#[derive(Clone, Debug)]
pub struct OuterSeedEvaluation {
    /// The criterion value.
    pub cost: f64,
    /// The analytic θ-gradient, present exactly when the evaluation asked for
    /// [`OuterSeedOrder::ValueAndGradient`].
    pub gradient: Option<Array1<f64>>,
    /// `(cost, [fixed_beta, logdet_h, logdet_s, kkt])` as the evaluator
    /// published them.
    ///
    /// `None` where the criterion is not assembled from REML atoms. The
    /// constant-curvature fair profile computes its value and derivative in
    /// closed form and publishes neither this nor a selected mode.
    pub criterion_components: Option<(f64, [f64; 4])>,
    /// The selected coefficient mode `β̂` and, on gradient evaluations, the
    /// analytic extended-coordinate response columns `v_i`, with
    /// `dβ̂/dψ_i = −v_i`.
    pub selected_mode: Option<(Array1<f64>, Option<Array2<f64>>)>,
}

/// Evaluation access to the real outer objective at one runner seed.
pub trait OuterSeedProbe {
    /// The seed this probe was lent at, and its box.
    fn layout(&self) -> &OuterSeedLayout;

    /// Evaluate the criterion at `theta` from the seed's own inner start.
    ///
    /// Every call resets the objective first, so two calls at one θ are the
    /// same evaluation, and a displaced θ is solved from the starting point the
    /// seed itself is solved from.
    fn evaluate(
        &mut self,
        theta: &Array1<f64>,
        order: OuterSeedOrder,
    ) -> Result<OuterSeedEvaluation, EstimationError>;
}

/// A test's inspection of the outer objective at one seed.
///
/// The runner logs an error it returns and proceeds with the seed cascade
/// unchanged: an observer must not decide the fit it observes.
pub type OuterSeedObserver =
    Box<dyn FnOnce(&mut dyn OuterSeedProbe) -> Result<(), EstimationError>>;

/// What the evaluator published during one probe evaluation.
#[derive(Default)]
pub(crate) struct OuterSeedCapture {
    criterion_components: Option<(f64, [f64; 4])>,
    selected_mode: Option<(Array1<f64>, Option<Array2<f64>>)>,
}

impl OuterSeedCapture {
    pub(crate) fn into_evaluation(
        self,
        cost: f64,
        gradient: Option<Array1<f64>>,
    ) -> OuterSeedEvaluation {
        OuterSeedEvaluation {
            cost,
            gradient,
            criterion_components: self.criterion_components,
            selected_mode: self.selected_mode,
        }
    }
}

thread_local! {
    static SEED_OBSERVER: RefCell<Option<(usize, OuterSeedObserver)>> = const { RefCell::new(None) };
    static SEED_CAPTURE: RefCell<Option<OuterSeedCapture>> = const { RefCell::new(None) };
}

/// Lend `observer` a probe at the next outer seed on this thread that has at
/// least `min_psi_dim` ψ axes. Replaces an observer that has not run yet.
pub fn observe_next_outer_seed(min_psi_dim: usize, observer: OuterSeedObserver) {
    SEED_OBSERVER.with(|slot| *slot.borrow_mut() = Some((min_psi_dim, observer)));
}

/// Take this thread's observer when a seed with `psi_dim` ψ axes satisfies it.
pub(crate) fn take_outer_seed_observer(psi_dim: usize) -> Option<OuterSeedObserver> {
    SEED_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot
            .as_ref()
            .is_some_and(|(min_psi_dim, _)| psi_dim >= *min_psi_dim)
        {
            slot.take().map(|(_, observer)| observer)
        } else {
            None
        }
    })
}

/// Open the publication window for one probe evaluation.
pub(crate) fn begin_outer_seed_capture() {
    SEED_CAPTURE.with(|capture| *capture.borrow_mut() = Some(OuterSeedCapture::default()));
}

/// Close the publication window and take what the evaluation published.
pub(crate) fn end_outer_seed_capture() -> OuterSeedCapture {
    SEED_CAPTURE.with(|capture| capture.borrow_mut().take().unwrap_or_default())
}

/// Whether a probe evaluation is in flight on this thread.
///
/// Emitters consult this before building the evidence they would hand to
/// [`record_outer_selected_mode`], so an ordinary fit pays a thread-local read
/// rather than a coefficient-vector clone on every outer evaluation.
pub(crate) fn outer_seed_capture_armed() -> bool {
    SEED_CAPTURE.with(|capture| capture.borrow().is_some())
}

/// Publish the selected scalar-criterion decomposition to an in-flight probe
/// evaluation.
///
/// Public so sibling workspace evaluators can report through the same typed
/// sink after their own nonconvex mode selection. No-op outside a probe
/// evaluation.
pub fn record_outer_criterion_components(cost: f64, components: [f64; 4]) {
    SEED_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut() {
            state.criterion_components = Some((cost, components));
        }
    });
}

/// Publish the selected coefficient mode and its analytic extended-coordinate
/// response columns to an in-flight probe evaluation.
///
/// Sibling workspace evaluators call this only after nonconvex candidate
/// selection, beside [`record_outer_criterion_components`]. Value-only
/// evaluations pass no response columns but still publish their selected
/// coefficients. No-op outside a probe evaluation.
pub fn record_outer_selected_mode(
    beta: Array1<f64>,
    ext_mode_response_cols: Option<Array2<f64>>,
) {
    SEED_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut() {
            state.selected_mode = Some((beta, ext_mode_response_cols));
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
//  ρ-block outer audit (#2454)
// ═══════════════════════════════════════════════════════════════════════════
//
// The seed probe has carried typed analytic ψ evidence since #2460; the ρ
// block had none, so every large-λ smoothing-gradient investigation had to
// scrape `log::trace!` lines or bolt an environment-gated instrument onto the
// evaluator. This channel closes that asymmetry: it emits, per outer
// evaluation, the SAME four-way additive decomposition the criterion VALUE
// carries (`RemlCriterionComponents`) but for each ρ coordinate's analytic
// gradient — so a caller can finite-difference each criterion component and
// grade the gradient part that owns it, instead of grading only their sum.
//
// Thread-local and disabled by default, matching the seed probe: a parallel
// integration test can neither consume nor overwrite another test's audit.

/// One ρ coordinate's analytic gradient, split into the additive parts that
/// match the criterion-value components of `RemlCriterionComponents`.
///
/// `fixed_beta + logdet_h + logdet_s` is the envelope gradient entry as
/// assembled; `total` additionally carries any IFT/KKT correction folded in
/// afterwards, so `total − (fixed_beta + logdet_h + logdet_s)` is the `kkt`
/// part. `lambda` and `block_quadratic` are the two raw inputs the
/// `fixed_beta` part is built from (`½·λ_k·q_k`, scaled by the dispersion
/// channel), retained because a defect that is proportional to `λ_k` is only
/// diagnosable against the `λ_k` it was multiplied by.
#[derive(Clone, Copy, Debug)]
pub struct RhoGradientParts {
    pub index: usize,
    pub lambda: f64,
    pub block_quadratic: f64,
    /// `rank(S_k)` as the outer penalty coordinate represents it (rows of its
    /// root), and the ambient dimension it acts on.
    pub rank: usize,
    pub dim: usize,
    pub fixed_beta: f64,
    pub logdet_h: f64,
    /// `logdet_h` split at the drift: `½ tr(K · λ_k S_k)`, the half that does
    /// not read the coefficient mode response. Both `K` and `S_k` are PSD, so a
    /// NEGATIVE value here is a defect with no oracle required.
    pub frozen_logdet_h: f64,
    /// The other half, `½ tr(K · D_β H[v_k])`.
    pub mode_response_logdet_h: f64,
    pub logdet_s: f64,
    pub total: f64,
}

/// The two floating-point spellings of the penalty energy `β̂ᵀS(λ)β̂` that the
/// criterion and its ρ-gradient respectively read, plus the profiled-Gaussian
/// scalars that connect them to `fixed_beta`.
///
/// `stable` is the inner solve's stable-basis emission (what the criterion
/// VALUE uses); `block_sum` is `Σ_k λ_k q_k` rebuilt from the outer penalty
/// coordinates (what `½λ_k q_k` — the gradient's `fixed_beta` channel — is a
/// per-block projection of). They are the same mathematical quantity, so any
/// disagreement is a floating-point one; the ρ-derivative multiplies it by
/// `λ_k`, which is why it must be measured rather than assumed small.
///
/// Recorded on BOTH dispersion arms (#2644). The channel reconstruction
/// `dp_cgrad · (½λ_k q_k) / phi` is what the three scalars are for, and it is
/// kept true on both: the profiled-Gaussian arm supplies the smooth
/// deviance-floor chain factor and the profiled scale, while fixed dispersion
/// — where the channel is bare `½λ_k q_k` — supplies `dp_cgrad = phi = 1.0`.
/// `dp_raw`/`dp_floored` are the penalized deviance; on the fixed arm no
/// criterion term reads them and they are equal.
#[derive(Clone, Copy, Debug)]
pub struct PenaltyEnergyAudit {
    pub stable: f64,
    pub block_sum: f64,
    pub dp_raw: f64,
    pub dp_floored: f64,
    pub dp_cgrad: f64,
    pub phi: f64,
}

/// The same penalty energy spelled from the ORIGINAL-frame canonical penalty
/// roots and from the TRANSFORMED-frame (post-`Qs`) ones, both evaluated at the
/// coefficient vector the outer evaluator will actually use.
///
/// Recorded at the assembly site, where both root sets and the inner solve's
/// own `stable_penalty_term` are simultaneously in scope.
#[derive(Clone, Debug)]
pub struct PenaltyFrameAudit {
    pub stable_penalty_term: f64,
    pub original_frame_blocks: Vec<f64>,
    pub transformed_frame_blocks: Vec<f64>,
    /// `‖Qs − I‖_max`; zero exactly when the reparameterization is the identity.
    pub qs_deviation_from_identity: f64,
    /// Which coefficient frame the inner solve reports `beta` in.
    pub coordinate_frame: &'static str,
    /// `βᵀ S_transformed β` from the reparameterization's rebuilt (rank-truncated)
    /// penalty, at `β` as handed to the outer evaluator and at `Qsᵀβ`.
    pub s_transformed_quadratic: f64,
    pub s_transformed_quadratic_rotated: f64,
    /// `‖E_transformed β‖²` and `‖E_transformed Qsᵀβ‖²`.
    pub e_transformed_quadratic: f64,
    pub e_transformed_quadratic_rotated: f64,
    /// `p`, the reconstruction's row count (`structural_rank`), and the
    /// dimension of the λ-invariant DECLARED-NULL subspace the split excludes.
    pub p: usize,
    pub e_rows: usize,
    pub null_dim: usize,
    /// `‖U_⊥ᵀ β_t‖²` — how much of β̂ lives in the declared-null subspace.
    pub beta_null_energy: f64,
    /// Per-block `(Πβ_t)ᵀ S_k^t (Πβ_t)` with `Π = I − U_⊥U_⊥ᵀ`: the block
    /// quadratic restricted to the subspace the criterion actually penalizes.
    pub projected_frame_blocks: Vec<f64>,
    /// The rank the criterion's own `−½ log|S(λ)|₊` term ranges over, and its
    /// value. This is the OTHER half of the same-penalty question (#2454): the
    /// `fixed_beta` channel and `H` both carry the split-projected `S̃`, whose
    /// rank is `e_rows`, while `log|S|₊` is taken on `Σ_k λ_k S_k` and can
    /// therefore charge MORE directions than `½log|H|` will ever inflate. The
    /// asymptotic slope of the criterion in ρ is `½(rank(S̃) − penalty_rank)`,
    /// so any gap between these two integers is a linear-in-ρ ramp with no
    /// interior optimum.
    pub penalty_logdet_rank: usize,
    pub penalty_logdet_value: f64,
}

/// One outer evaluation's #784 block-local quadrature record: the
/// spliced value `Δ_b`, the block the splice selected, and the four gradient
/// channels PER ρ COORDINATE exactly as the assembly formed them (#2623).
///
/// Every field is in the corrector's own `Δ_b`-side convention, i.e. the sign the
/// producer emits, NOT the cost-side sign. `delta_b` is `+Δ_b` (the criterion
/// carries `−Δ_b`) and `explicit_a` is the raw quadrature gradient. Recording
/// the raw values is the whole point: the sign question this decides is which
/// side of `d(cost)/dρ = −d(Δ_b)/dρ` each channel already lives on, and a record
/// that pre-applied a sign would assume the answer.
///
/// `spliced` is the entry the assembly actually adds to the cost gradient, so
/// `spliced` vs `−(explicit_a + trace_bc + mode_d)` is the disagreement itself,
/// readable without re-deriving it.
#[derive(Clone, Debug)]
pub struct QuadratureMarginalAudit {
    /// `Δ_b` as the corrector reports it: added to the block marginal
    /// log-likelihood, SUBTRACTED from the criterion.
    pub delta_b: f64,
    /// Absolute fine/coarse quadrature-rule difference on `delta_b`.
    pub quadrature_error: f64,
    /// Number of nodes in the fine rule.
    pub node_count: usize,
    /// The activation evidence: `max|γ_r|` over curvature directions and the
    /// threshold `τ(n_eff)` it had to exceed.
    pub max_abs_skewness: f64,
    pub skewness_threshold: f64,
    /// Which `H` eigenvector indices form the integrated block, ascending.
    ///
    /// An FD stencil must compare this ACROSS its points. The block is selected
    /// by a threshold on a per-direction diagnostic, so a stencil that changes
    /// block membership is differencing two different functions and its
    /// quotient is not a derivative of either.
    pub block_cols: Vec<usize>,
    /// Channel (a), `∂Δ_b/∂ρ_j` — the corrector's explicit penalty-score channel,
    /// raw.
    pub explicit_a: Vec<f64>,
    /// Channels (b)+(c) together, `tr(Ḣ_j · (Q_b + Q_c))`.
    pub trace_bc: Vec<f64>,
    /// Channel (d), `g_dᵀ · dβ̂/dρ_j`.
    pub mode_d: Vec<f64>,
    /// The gradient entry the assembly writes into the cost gradient.
    pub spliced: Vec<f64>,
}

/// One outer evaluation's ρ-block audit: the criterion value decomposition and
/// the per-coordinate analytic gradient decomposition that pairs with it.
#[derive(Clone, Debug, Default)]
pub struct RhoOuterAudit {
    /// `(cost, [fixed_beta, logdet_h, logdet_s, kkt])` for the criterion VALUE.
    pub criterion: Option<(f64, [f64; 4])>,
    /// Per-ρ-coordinate analytic gradient parts, in coordinate order.
    pub parts: Vec<RhoGradientParts>,
    /// The penalty-energy spellings behind the `fixed_beta` channel.
    pub penalty_energy: Option<PenaltyEnergyAudit>,
    /// The original-frame vs transformed-frame penalty roots at the assembly
    /// site.
    pub penalty_frame: Option<PenaltyFrameAudit>,
    /// Whether the #784 block-local quadrature ENGAGED on this
    /// evaluation (#2623).
    ///
    /// False means the splice DECLINED, so gradient channels (b), (c) and (d)
    /// were never formed. A finite-difference comparison of those channels is
    /// then vacuous rather than passing: it is the shape where a guard is
    /// satisfied by an absence. Any FD row that means to exercise them must
    /// ASSERT this true before comparing, or it silently degenerates into the
    /// well-behaved regime where the splice never runs.
    pub quadrature_marginal_engaged: bool,
    /// The engaged splice's value, block and per-coordinate channel split, or
    /// `None` when it declined (#2623).
    ///
    /// Present exactly when `quadrature_marginal_engaged` is true. Kept beside the
    /// flag rather than behind a separate accessor so a reader cannot assert
    /// engagement without having the channels in hand, nor read the channels
    /// without having checked engagement.
    pub quadrature_marginal: Option<QuadratureMarginalAudit>,
}

thread_local! {
    static RHO_AUDIT: RefCell<Option<RhoOuterAudit>> = const { RefCell::new(None) };
}

/// Arm the ρ-block audit on this thread, discarding any previous window.
///
/// Every subsequent outer evaluation on this thread overwrites the window, so
/// the caller reads the audit for the LAST evaluation it triggered — which is
/// the contract a probe wants when it evaluates at one θ at a time.
pub fn enable_rho_outer_audit() {
    RHO_AUDIT.with(|audit| *audit.borrow_mut() = Some(RhoOuterAudit::default()));
}

/// Disarm the ρ-block audit and take the last evaluation's window.
pub fn take_rho_outer_audit() -> Option<RhoOuterAudit> {
    RHO_AUDIT.with(|audit| audit.borrow_mut().take())
}

pub(crate) fn rho_outer_audit_enabled() -> bool {
    RHO_AUDIT.with(|audit| audit.borrow().is_some())
}

/// Start a fresh window for one outer evaluation (no-op when disarmed).
///
/// Clears the criterion and per-coordinate gradient slots, which the evaluator
/// refills on this evaluation. The penalty-frame slot is deliberately NOT
/// cleared: it is written at the assembly site, which runs BEFORE the evaluator
/// for the same evaluation, so clearing it here would discard the record the
/// caller asked for.
pub(crate) fn begin_rho_outer_audit_eval() {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            state.criterion = None;
            state.parts = Vec::new();
            // Engagement is decided INSIDE the evaluation, so it is cleared
            // here and set again if the splice runs. Latching it across
            // evaluations would let one engaged eval vouch for a later
            // declined one (#2623).
            state.quadrature_marginal_engaged = false;
            state.quadrature_marginal = None;
        }
    });
}

/// Record that the #784 quadrature splice engaged on this
/// evaluation, together with the channels it formed (#2623). No-op when the
/// audit is disarmed.
pub(crate) fn record_quadrature_marginal(record: QuadratureMarginalAudit) {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            state.quadrature_marginal_engaged = true;
            state.quadrature_marginal = Some(record);
        }
    });
}

/// The record written by the last engaged splice on this thread, if the audit is
/// armed and one has been written since the window began.
///
/// The correction is computed once per inner solution and cached on the eval
/// bundle, while the audit window is cleared at the START of every assemble call
/// sharing that bundle — and one ρ drives two or three of them (value,
/// value+gradient, value+gradient+Hessian). So the assemble that computes the
/// splice records it and the next one clears the record and then hits the cache,
/// which would report a genuinely engaged evaluation as declined. The cache
/// carries this record forward and re-publishes it, which is what this reader is
/// for (#2623).
pub(crate) fn last_quadrature_marginal_record() -> Option<QuadratureMarginalAudit> {
    RHO_AUDIT.with(|audit| {
        audit
            .borrow()
            .as_ref()
            .and_then(|state| state.quadrature_marginal.clone())
    })
}

pub(crate) fn record_rho_outer_criterion(cost: f64, components: [f64; 4]) {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            state.criterion = Some((cost, components));
        }
    });
}

pub(crate) fn record_rho_penalty_frame(frame: PenaltyFrameAudit) {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            state.penalty_frame = Some(frame);
        }
    });
}

pub(crate) fn record_rho_penalty_energy(energy: PenaltyEnergyAudit) {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            state.penalty_energy = Some(energy);
        }
    });
}

pub(crate) fn record_rho_gradient_parts(parts: Vec<RhoGradientParts>) {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            state.parts = parts;
        }
    });
}
