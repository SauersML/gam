//! Structured capture of outer-objective evidence for integration tests.
//!
//! The raw-evaluation window serves flexible-link measurements (#1876). The
//! finite-difference record serves end-to-end gradient gates (#2460): when
//! explicitly enabled, the generic outer runner compares the analytic gradient
//! at its first bounded seed with a finite difference of that same objective.
//! Tests consume typed arrays rather than scraping formatted production logs.
//!
//! Both channels are disabled by default. The raw window is process-global
//! because its flexible-link measurements intentionally span helper calls. The
//! finite-difference request is thread-local: a parallel integration test can
//! neither consume nor overwrite another test's one-shot audit.

use ndarray::{Array1, Array2};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// One captured outer evaluation: the outer coordinate `theta = (ρ ‖ link)`, the
/// scalar cost, and the analytic outer gradient in the same layout.
#[derive(Clone, Debug)]
pub struct OuterEvalRecord {
    pub theta: Array1<f64>,
    pub cost: f64,
    pub gradient: Array1<f64>,
}

/// Analytic-vs-finite-difference evidence for the ψ block at one real outer
/// seed.
///
/// `theta` retains the complete outer seed and `rho_dim` locates the ψ block in
/// that seed. Every gradient and scalar-stencil array contains exactly
/// `psi_dim` entries in ψ-local order. Smoothing-parameter ρ coordinates are
/// deliberately excluded: the κ/geometry gates that request this record do not
/// grade them, and each unnecessary finite-difference coordinate costs two
/// complete inner profiles.
#[derive(Clone, Debug)]
pub struct OuterGradientFdRecord {
    pub theta: Array1<f64>,
    pub rho_dim: usize,
    pub psi_dim: usize,
    pub cost: f64,
    pub analytic_psi_gradient: Array1<f64>,
    pub finite_difference_psi_gradient: Array1<f64>,
    pub psi_steps: Array1<f64>,
    /// Ridders' estimate of each ψ finite difference's OWN error, and the
    /// truncation order of the accepted extrapolant (`2` is a raw stencil, `4`
    /// one Richardson stage, …).
    ///
    /// Present because a finite difference is an estimator: without it a
    /// consumer cannot tell an analytic-gradient defect from its own oracle's
    /// truncation, and has to grade at whatever tolerance the worst step
    /// happens to need — which is how these gates ended up at `5e-2` (#2461).
    /// `f64::INFINITY` marks a coordinate the ladder could not resolve; such a
    /// component says nothing about the analytic gradient.
    pub psi_fd_uncertainty: Array1<f64>,
    pub psi_fd_orders: Vec<usize>,

    /// Max-abs of the `#1033b` psi-Gram anchor correction applied to the
    /// criterion's VALUE at this seed, as `(gram_delta, rhs_delta)`.
    ///
    /// `joint_hyper` pins the n-free tensor to the exactly streamed statistics
    /// by adding a constant offset measured at one reference psi, then installs
    /// the derivative of the UNCORRECTED tensor. A constant removes nothing from
    /// a derivative, so a non-zero value here is the tensor's own value error at
    /// this seed and its SLOPE error is loose in the gradient lane (#2464).
    ///
    /// `None` means the correction never ran on this seed -- NOT that it ran and
    /// was zero. The distinction is the whole point of the field: a probe that
    /// reports `0.0` for "never fired" is unfalsifiable, and reading an absent
    /// emission as a measured zero is exactly how this quantity was first
    /// mis-measured.
    pub psi_gram_anchor_deltas: Option<(f64, f64)>,
    /// The per-atom breakdown of the same comparison, when the objective's
    /// criterion is assembled from atoms at all.
    pub decomposition: OuterGradientFdDecomposition,
    /// The SAME comparison for the ρ (log-smoothing) block, when the caller
    /// armed [`enable_outer_gradient_fd_capture_over_theta`].
    ///
    /// `None` is the default and means the ρ coordinates were not differenced —
    /// NOT that they agreed. The two blocks are separate fields rather than one
    /// θ-indexed array because grading ρ is a deliberate, expensive opt-in: each
    /// coordinate costs a full Ridders ladder of inner profiles, and the shipped
    /// κ/geometry gates that consume the ψ block do not grade ρ.
    pub rho: Option<OuterGradientFdRhoBlock>,
    /// The curvature the `logdet_h` atom is taken on, differenced as a MATRIX
    /// against the analytic drift the same evaluation published (#2765).
    ///
    /// `None` when the backend cannot hand out a dense `H`. See
    /// [`OuterCurvatureDriftAudit`] for why a scalar-only audit cannot decide
    /// what this decides.
    pub curvature: Option<OuterCurvatureDriftAudit>,
}

/// The criterion's own curvature and its per-θ-coordinate analytic drift, as
/// MATRICES, at one evaluation point (#2765).
///
/// The scalar audit above compares `½ tr(K·Ḣ_i)` against a finite difference of
/// `½ log|H|`. When those disagree, three objects could be at fault and the
/// scalar cannot separate them: the curvature `H` itself, the drift `Ḣ_i`, or
/// the trace kernel `K` the cost's log-determinant pairs with. Differencing `H`
/// as a matrix and comparing it to `Ḣ_i` entry by entry answers the middle
/// question outright, and answers it in a form that names WHICH block of the
/// joint coefficient space is wrong rather than reporting one contracted number.
#[derive(Clone, Debug)]
pub struct OuterCurvatureSnapshot {
    /// The dense curvature whose log-determinant the `logdet_h` atom reports.
    pub hessian: Array2<f64>,
    /// The orthonormal tangent basis `Z` of `null(A_act)` when the inner solve
    /// returned on an active inequality face and the criterion is therefore the
    /// TANGENT-projected `½log|ZᵀHZ|` (#2765).
    ///
    /// Recorded because it is the coordinate system the whole comparison lives
    /// in: a drift stated in `p`-space says nothing about a determinant taken on
    /// an `m`-dimensional face, and a face that MOVES between two displaced θ is
    /// not a differentiable criterion at all — a distinction the contracted
    /// scalar cannot express and the previous audit silently averaged over.
    pub tangent_basis: Option<Array2<f64>>,
    /// `log|H|` as the criterion consumes it, i.e. the operator's own
    /// log-determinant plus any uniform-rescale correction. Recorded so a gap
    /// against `½ log det(hessian)` — a value/kernel disagreement rather than a
    /// derivative one — is visible instead of assumed absent.
    pub logdet: f64,
    /// Total analytic `Ḣ_i` per θ coordinate, ρ block first then ψ, densified.
    pub drifts: Vec<Array2<f64>>,
}

/// Per-θ-coordinate evidence that the analytic drift IS the derivative of the
/// curvature the criterion's log-determinant is taken on (#2765).
#[derive(Clone, Debug)]
pub struct OuterCurvatureDriftAudit {
    /// `‖Ḣ_i^analytic − (H(θ+h) − H(θ−h))/2h‖_max` per θ coordinate.
    pub drift_max_abs_error: Array1<f64>,
    /// The same, relative to `‖Ḣ_i‖_max` of the two.
    pub drift_relative_error: Array1<f64>,
    /// `(row, col)` of the entry carrying `drift_max_abs_error`.
    pub drift_worst_entry: Vec<(usize, usize)>,
    /// `‖Z₊Z₊ᵀ − Z₋Z₋ᵀ‖_max` per θ coordinate: how far the ACTIVE FACE the
    /// criterion's determinant is taken on moves between the two displaced
    /// evaluations. A non-zero entry means the finite difference straddles two
    /// different criteria, so the analytic derivative is not wrong there — the
    /// question is not well posed there.
    pub face_drift_max_abs: Array1<f64>,
    /// Tangent dimension at the base point, and at each coordinate's `θ ± h`.
    pub tangent_dim: Option<usize>,
    pub displaced_tangent_dim: Vec<(Option<usize>, Option<usize>)>,
    /// The two objects the comparison is between, retained per θ coordinate:
    /// the analytic `ZᵀḢ_iZ` and the measured `d(ZᵀHZ)/dθ_i`. A max-abs number
    /// says a drift is wrong; these say HOW — whether the error is a multiple
    /// of the face curvature, a rank-one leak, or one block's own.
    pub analytic_face_drift: Vec<Array2<f64>>,
    pub measured_face_drift: Vec<Array2<f64>>,
    /// `ZᵀHZ` itself, the curvature whose determinant the atom reports.
    pub face_curvature: Array2<f64>,
    /// `‖Ḣ_i^analytic‖_max`, so a relative error can be read against a scale.
    pub analytic_drift_max_abs: Array1<f64>,
    /// `½ log det(H)` recomputed from the captured dense matrix, against the
    /// `logdet` the criterion actually consumed. A gap here means the operator's
    /// log-determinant is not the plain one of this matrix (spectral
    /// regularization, a rank mask, a uniform rescale), which changes what the
    /// trace kernel has to be.
    pub dense_half_logdet: f64,
    pub criterion_half_logdet: f64,
}

/// Analytic-vs-finite-difference evidence for the ρ (log-smoothing) block at the
/// same seed, in ρ-local order (#2765).
///
/// The ψ block above and this one are graded by the SAME Ridders ladder against
/// the SAME criterion, which is the point: the two blocks share the criterion's
/// moving-Hessian machinery (the trace kernel `K` and the mode-response drift
/// `D_β H[v]`) but have structurally different frozen drifts — `λ_k S_k`, a
/// known exact matrix, for ρ, and the family's own `∂_ψ H|_β` for ψ. So a
/// defect that shows in BOTH blocks lives in the shared machinery and a defect
/// that shows in only one lives in that block's own drift. Without this the
/// bisection could not be made, and a ψ-only audit could only report that
/// *something* in the chain was wrong.
///
/// The analytic parts come from the ρ-block audit channel
/// ([`RhoGradientParts`]), which the capture arms for itself; the
/// finite-difference parts come from the same criterion-component stencils the
/// ψ block uses.
#[derive(Clone, Debug)]
pub struct OuterGradientFdRhoBlock {
    pub analytic_gradient: Array1<f64>,
    pub finite_difference_gradient: Array1<f64>,
    pub steps: Array1<f64>,
    pub fd_uncertainty: Array1<f64>,
    pub fd_orders: Vec<usize>,
    /// The analytic entry as the ρ-block audit reports it, before the prior
    /// gradient and the canonical-face KKT projection the assembly applies
    /// afterwards. Equal to `analytic_gradient` on every model that carries
    /// neither; recorded separately so a gap between them is visible rather
    /// than charged to the derivative.
    pub analytic_audit_total: Array1<f64>,
    pub analytic_fixed_beta: Array1<f64>,
    pub analytic_logdet_h: Array1<f64>,
    /// `analytic_logdet_h` split at the drift: the half that does not read the
    /// coefficient mode response, `½ tr(K · λ_k S_k)`, and the half that does,
    /// `½ tr(K · D_β H[v_k])`. The first is PSD-by-construction, so a negative
    /// entry is a defect that needs no oracle at all.
    pub analytic_frozen_logdet_h: Array1<f64>,
    pub analytic_mode_response_logdet_h: Array1<f64>,
    pub analytic_logdet_s: Array1<f64>,
    /// `audit_total − (fixed_beta + logdet_h + logdet_s)`: the IFT/KKT fold.
    pub analytic_kkt: Array1<f64>,
    pub finite_difference_fixed_beta: Array1<f64>,
    pub finite_difference_logdet_h: Array1<f64>,
    pub finite_difference_logdet_s: Array1<f64>,
    pub finite_difference_kkt: Array1<f64>,
}

/// Whether the audited criterion decomposes into REML atoms, and the evidence
/// either way (#2460).
///
/// The comparison above — one analytic ψ gradient against one Ridders-certified
/// finite difference of the same objective — is available from any outer
/// objective that declares a ψ block, because it needs only `eval_cost` and
/// `eval_with_order`. The breakdown below is not: it exists where the criterion
/// is `fixed-β likelihood + ½log|H| − ½log|S|₊ + KKT residual` and the evaluator
/// publishes those atoms as it assembles them.
///
/// Routes that evaluate a criterion directly — the constant-curvature fair
/// profile computes its value and derivative in closed form and never enters a
/// REML assembly — have no atoms to publish and no selected coefficient mode to
/// difference. Making the breakdown a PRECONDITION of the measurement is what
/// left those routes with no audit at all, which is the wrong way round: a
/// hand-derived derivative on a bespoke profile is the one that most wants
/// checking.
#[derive(Clone, Debug)]
pub enum OuterGradientFdDecomposition {
    /// The evaluator published every atom, and each is differenced at the step
    /// the Ridders ladder accepted for the total.
    Decomposed(Box<OuterGradientFdAtoms>),
    /// The evaluator published no atoms, no scalar criterion components and no
    /// selected coefficient mode. `reason` names the objective so a consumer
    /// reports which route it got rather than an empty array.
    ///
    /// A PARTIAL publication is never reported here — it is a defect in an
    /// evaluator that means to decompose, and the capture still fails loudly.
    NotDecomposed { reason: String },
}

impl OuterGradientFdDecomposition {
    /// The atoms, or `None` where the criterion does not decompose.
    pub fn atoms(&self) -> Option<&OuterGradientFdAtoms> {
        match self {
            Self::Decomposed(atoms) => Some(atoms),
            Self::NotDecomposed { .. } => None,
        }
    }
}

/// Per-atom analytic-vs-finite-difference evidence, in ψ-local order.
///
/// This is what localizes a total mismatch to a term: the survival marginal-slope
/// gate reads it to separate an agreeing fixed-β atom from a disagreeing
/// moving-Hessian chain, which is a different bug report from "the gradient is
/// wrong".
#[derive(Clone, Debug)]
pub struct OuterGradientFdAtoms {
    pub fixed_beta_psi_gradient: Array1<f64>,
    pub logdet_h_psi_gradient: Array1<f64>,
    pub frozen_logdet_h_psi_gradient: Array1<f64>,
    pub mode_response_logdet_h_psi_gradient: Array1<f64>,
    pub analytic_mode_response_norm: Array1<f64>,
    pub finite_difference_mode_response_norm: Array1<f64>,
    pub mode_response_relative_error: Array1<f64>,
    pub mode_response_max_abs_error: Array1<f64>,
    pub logdet_s_psi_gradient: Array1<f64>,
    pub kkt_psi_gradient: Array1<f64>,
    pub finite_difference_fixed_beta_psi_gradient: Array1<f64>,
    pub finite_difference_logdet_h_psi_gradient: Array1<f64>,
    pub finite_difference_logdet_s_psi_gradient: Array1<f64>,
    pub finite_difference_kkt_psi_gradient: Array1<f64>,
}

/// Maximum evaluations retained per capture window (opening iterates only).
const MAX_CAPTURED: usize = 8;

static ENABLED: AtomicBool = AtomicBool::new(false);

struct OuterGradientFdCapture {
    min_psi_dim: usize,
    grade_rho: bool,
    record: Option<OuterGradientFdRecord>,
    components: Vec<(f64, f64, f64, f64, f64, f64)>,
    criterion_components: Option<(f64, [f64; 4])>,
    psi_gram_anchor_deltas: Option<(f64, f64)>,
    selected_mode: Option<(Array1<f64>, Option<Array2<f64>>)>,
    curvature: Option<OuterCurvatureSnapshot>,
    tangent_basis: Option<Array2<f64>>,
}

thread_local! {
    static FD_CAPTURE: RefCell<Option<OuterGradientFdCapture>> = const { RefCell::new(None) };
}

fn buffer() -> &'static Mutex<Vec<OuterEvalRecord>> {
    static BUFFER: OnceLock<Mutex<Vec<OuterEvalRecord>>> = OnceLock::new();
    BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

/// Request one structured audit over the WHOLE θ vector — the ψ block and the
/// ρ (log-smoothing) block — at the next outer seed with enough ψ axes.
///
/// Opt-in rather than the default because grading ρ costs a full Ridders ladder
/// of inner profiles per coordinate, which is the same price the ψ block pays
/// and which the shipped κ/geometry gates have no use for. What it buys is the
/// bisection described on [`OuterGradientFdRhoBlock`]: the two blocks share the
/// criterion's moving-Hessian machinery and differ in their frozen drift, so a
/// disagreement present in one and absent in the other localizes the defect
/// without any new instrumentation inside the evaluator.
pub fn enable_outer_gradient_fd_capture_over_theta(min_psi_dim: usize) {
    arm_outer_gradient_fd_capture(min_psi_dim, true);
}

fn arm_outer_gradient_fd_capture(min_psi_dim: usize, grade_rho: bool) {
    FD_CAPTURE.with(|capture| {
        *capture.borrow_mut() = Some(OuterGradientFdCapture {
            min_psi_dim,
            grade_rho,
            record: None,
            components: Vec::new(),
            criterion_components: None,
            psi_gram_anchor_deltas: None,
            selected_mode: None,
            curvature: None,
            tangent_basis: None,
        });
    });
}

/// Whether the armed audit was asked to difference the ρ block too.
pub(crate) fn outer_gradient_fd_capture_grades_rho() -> bool {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()
            .is_some_and(|state| state.record.is_none() && state.grade_rho)
    })
}

pub(crate) fn begin_outer_gradient_component_capture() {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut() {
            state.components.clear();
        }
    });
}

pub(crate) fn outer_gradient_component_capture_enabled() -> bool {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()
            .is_some_and(|state| state.record.is_none())
    })
}

pub(crate) fn record_outer_gradient_component(
    fixed_beta: f64,
    logdet_h: f64,
    frozen_logdet_h: f64,
    mode_response_logdet_h: f64,
    logdet_s: f64,
    kkt: f64,
) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.components.push((
                fixed_beta,
                logdet_h,
                frozen_logdet_h,
                mode_response_logdet_h,
                logdet_s,
                kkt,
            ));
        }
    });
}

pub(crate) fn take_outer_gradient_components() -> Vec<(f64, f64, f64, f64, f64, f64)> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .map_or_else(Vec::new, |state| std::mem::take(&mut state.components))
    })
}

pub(crate) fn begin_outer_criterion_component_capture() {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut() {
            state.criterion_components = None;
            state.psi_gram_anchor_deltas = None;
            state.selected_mode = None;
            state.curvature = None;
            state.tangent_basis = None;
        }
    });
}

pub(crate) fn peek_outer_tangent_basis() -> Option<Array2<f64>> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()
            .and_then(|state| state.tangent_basis.clone())
    })
}

/// Retain the criterion's dense curvature and its analytic drifts for an armed
/// audit (#2765).
///
/// Called from the REML/LAML assembly, which is the only place that holds both
/// the operator the log-determinant is taken on and the per-coordinate drift
/// that claims to be its derivative. No-op unless a finite-difference audit
/// armed this thread, so an ordinary fit pays one thread-local read.
pub fn record_outer_curvature_snapshot(snapshot: OuterCurvatureSnapshot) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.curvature = Some(snapshot);
        }
    });
}

pub(crate) fn take_outer_curvature_snapshot() -> Option<OuterCurvatureSnapshot> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .and_then(|state| state.curvature.take())
    })
}

/// Retain the final selected scalar-criterion decomposition for an armed
/// outer-gradient audit.
///
/// This is public only so sibling workspace evaluators can report through the
/// same typed sink after their own nonconvex mode selection. It is a no-op
/// unless [`enable_outer_gradient_fd_capture`] armed the calling thread.
pub fn record_outer_criterion_components(cost: f64, components: [f64; 4]) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.criterion_components = Some((cost, components));
        }
    });
}

/// Report the psi-Gram anchor correction's magnitude for an armed audit.
///
/// Public for the same reason as [`record_outer_criterion_components`]: the
/// correction is applied in the evaluator, not in the outer runner that builds
/// the record. No-op unless [`enable_outer_gradient_fd_capture`] armed this
/// thread. Called on every application, so the LAST application before the
/// record is finalized is the one reported -- the seed the audit grades.
pub fn record_psi_gram_anchor_deltas(gram_delta_max_abs: f64, rhs_delta_max_abs: f64) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.psi_gram_anchor_deltas = Some((gram_delta_max_abs, rhs_delta_max_abs));
        }
    });
}

pub(crate) fn take_psi_gram_anchor_deltas() -> Option<(f64, f64)> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .and_then(|state| state.psi_gram_anchor_deltas.take())
    })
}

pub(crate) fn take_outer_criterion_components() -> Option<(f64, [f64; 4])> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .and_then(|state| state.criterion_components.take())
    })
}

/// Retain the selected coefficient mode and its analytic extended-coordinate
/// response columns for an armed finite-difference audit.
///
/// Sibling workspace evaluators call this only after nonconvex candidate
/// selection, beside [`record_outer_criterion_components`]. Value-only
/// evaluations pass no response columns but still retain their selected
/// coefficients for the scalar stencil.
pub fn record_outer_selected_mode(
    beta: Array1<f64>,
    ext_mode_response_cols: Option<Array2<f64>>,
) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.selected_mode = Some((beta, ext_mode_response_cols));
        }
    });
}

/// Whether a finite-difference audit is armed on this thread at all.
///
/// Emitters consult this before building the evidence they would hand to
/// [`record_outer_selected_mode`], so an unarmed fit pays a thread-local read
/// rather than a coefficient-vector clone on every outer evaluation.
pub fn outer_gradient_audit_capture_armed() -> bool {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()
            .is_some_and(|state| state.record.is_none())
    })
}

pub(crate) fn take_outer_selected_mode() -> Option<(Array1<f64>, Option<Array2<f64>>)> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .and_then(|state| state.selected_mode.take())
    })
}

/// Stop the audit window and take its single record.
pub fn take_outer_gradient_fd_capture() -> Option<OuterGradientFdRecord> {
    FD_CAPTURE.with(|capture| capture.borrow_mut().take().and_then(|state| state.record))
}

pub(crate) fn outer_gradient_fd_capture_enabled(psi_dim: usize) -> bool {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()
            .is_some_and(|state| state.record.is_none() && psi_dim >= state.min_psi_dim)
    })
}

pub(crate) fn record_outer_gradient_fd(record: OuterGradientFdRecord) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
            && record.psi_dim >= state.min_psi_dim
        {
            state.record = Some(record);
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
//  ρ-block outer audit (#2454)
// ═══════════════════════════════════════════════════════════════════════════
//
// The ψ block has carried a typed analytic-vs-FD record since #2460; the ρ
// block had none, so every large-λ smoothing-gradient investigation had to
// scrape `log::trace!` lines or bolt an environment-gated instrument onto the
// evaluator. This channel closes that asymmetry: it emits, per outer
// evaluation, the SAME four-way additive decomposition the criterion VALUE
// carries (`RemlCriterionComponents`) but for each ρ coordinate's analytic
// gradient — so a caller can finite-difference each criterion component and
// grade the gradient part that owns it, instead of grading only their sum.
//
// Thread-local and disabled by default, matching the ψ channel: a parallel
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

/// Re-arm the ρ-block audit with a window taken earlier.
///
/// Exists so a nested consumer — the outer-gradient FD capture, which arms this
/// channel for itself to read the analytic ρ atoms — can hand a caller's window
/// back untouched instead of silently disarming an audit it did not open.
pub fn restore_rho_outer_audit(window: RhoOuterAudit) {
    RHO_AUDIT.with(|audit| *audit.borrow_mut() = Some(window));
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

/// Record one outer evaluation when capture is enabled (no-op otherwise). Only
/// the first [`MAX_CAPTURED`] evaluations of a window are retained.
pub(crate) fn record_outer_eval(theta: &Array1<f64>, cost: f64, gradient: &Array1<f64>) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let mut b = buffer().lock().expect("outer-eval capture buffer");
    if b.len() < MAX_CAPTURED {
        b.push(OuterEvalRecord {
            theta: theta.clone(),
            cost,
            gradient: gradient.clone(),
        });
    }
}
