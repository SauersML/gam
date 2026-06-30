use super::*;
pub use gam_problem::WeightField;

// ---------------------------------------------------------------------------
// Isometry penalty
// ---------------------------------------------------------------------------

/// Choice of reference Riemannian metric `g^ref(t)` on the latent manifold.
///
/// `Euclidean` is the natural default: the reference metric is `I_d`, so the
/// penalty pulls the decoder toward locally-isometric (length-preserving)
/// behavior. `UserSupplied` lets the caller hand in a `(n_obs, d, d)` jet of
/// per-row reference metrics (useful for warm-starting from a chart of a
/// pre-fit GP-LVM).
#[derive(Clone)]
pub enum IsometryReference {
    Euclidean,
    UserSupplied(Arc<Array2<f64>>), // (n_obs, d*d) row-major flattened
}

impl std::fmt::Debug for IsometryReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsometryReference::Euclidean => f.write_str("Euclidean"),
            IsometryReference::UserSupplied(a) => f
                .debug_tuple("UserSupplied")
                .field(&format_args!("{}×{}", a.nrows(), a.ncols()))
                .finish(),
        }
    }
}

/// Radial Duchon decoder metadata used to materialize
/// `∂J_n[i, a] / ∂t_{n, c}` from `φ'(r)` and `φ''(r)` on demand.
///
/// `radial_coefficients[k, i]` is the decoder coefficient that maps radial
/// basis column `k` into output channel `i`. Polynomial-tail columns are not
/// represented here; callers whose decoder contains a non-linear polynomial
/// tail should provide `jacobian_second_cache` directly.
#[derive(Debug, Clone)]
pub struct IsometryDuchonRadialSource {
    pub centers: Arc<Array2<f64>>,
    pub radial_coefficients: Arc<Array2<f64>>,
    pub length_scale: Option<f64>,
    pub nullspace_order: DuchonNullspaceOrder,
    /// Forward hybrid spectral order `s = spec.power`. The Cartesian
    /// derivative engine must resolve the same `(p, s, κ)` the forward
    /// `build_duchon_basis` used, so it differentiates the exact resolved
    /// hybrid Green's function `φ_{p,s,κ}` rather than a hard-coded `s = 0`
    /// surrogate (issue #440).
    pub power: usize,
}

/// Isometry-to-reference penalty (canonical-coordinate gauge term).
///
/// Lives on ext-coords: the target slice is a row of the `LatentCoordValues` flat
/// vector (row-major `n_obs × d`). Owns one ρ-axis (`log μ_iso`).
///
/// Penalizes `½ μ Σ_n ‖g_n(t) − g^ref(t_n)‖²_F`, where the pullback metric
/// at row `n` is
///
/// ```text
///   g_n = J_n^T W_n J_n,    J_n ∈ ℝ^{p × d}
/// ```
///
/// and `W_n` is a per-row low-rank PSD behavioral metric stored as
/// `W_n = U_n U_n^T` with `U_n ∈ ℝ^{p × r}`. The canonical-coordinate
/// statement is "one unit of motion in `t` ↦ one unit of behavioral change",
/// so the `W_n` weighting is load-bearing.
///
/// In the SAE objective this is the extension-coordinate gauge fix: it prevents
/// the latent chart from absorbing arbitrary smooth reparameterizations of the
/// decoder manifold. ARD, sparsity, or rank penalties can then select axes or
/// structure in a chart whose metric scale is pinned.
///
/// **Contraction order invariant.** Every place this struct touches `W_n`,
/// the contraction is `(J^T U_n)(U_n^T J)` — never `J^T W_n J` with `W_n`
/// materialized as `p × p`. Concretely we form `M_n = U_n^T J_n ∈ ℝ^{r × d}`
/// once and then `g_n = M_n^T M_n` (`d × d`). Cost per row:
/// `O(p · r · d + r · d²)`, independent of `p²`.
///
/// **When to use.** Whenever a `LatentCoord` block is in play without an
/// auxiliary variable (`AuxPrior`) to break the diffeomorphism gauge. Fixes
/// the audit finding that ARD is not a standalone gauge fix. With a Euclidean
/// reference, the penalty pulls the decoder toward a local isometry, which is
/// enough to make the inner Hessian on `t` full-rank and the IFT well-defined.
///
/// **Math.** Let `J_n ∈ ℝ^{p × d}` be the local decoder Jacobian. Then
/// `g_n = J_n^T W_n J_n` and the penalty is
/// `½ μ Σ_n ‖J_n^T W_n J_n − g^ref_n‖²_F`. Analytic gradient w.r.t. `t_n`:
///
/// ```text
///   ∂P/∂t_{n,c}
///     = μ Σ_{a,b} (g_n − g^ref_n)_{ab}
///         [ H_{n,:,a,c}^T W_n J_{n,:,b}
///           + J_{n,:,a}^T W_n H_{n,:,b,c} ],
///   H_{n,i,a,c} = ∂J_{n,i,a}/∂t_{n,c}.
/// ```
///
/// Gotchas:
///
/// * The value path returns the configured missing-cache default when the
///   first-jet cache is absent; gradient/HVP paths need the first and second
///   decoder jets and return zeros when the analytic jet source is unavailable.
/// * The exact Hessian includes a residual-curvature term requiring the third
///   decoder jet. REML/PIRLS curvature should prefer the Gauss-Newton PSD
///   majorizer when a positive curvature block is required.
/// * `W_n` is a metric weight, not a scalar confidence. Changing it changes the
///   canonical units of latent motion.
///
/// The per-row Jacobian `J_n` is exactly the radial-derivative jet
/// `design_gradient_wrt_t` already computes for `LatentCoordValues`; the
/// second derivative `∂J/∂t` is built by the shared
/// [`crate::basis::radial_basis_cartesian_derivative`] engine from the
/// radial Hessian identity. A finite-difference oracle for the docstring is
/// to central-difference `value(t ± h e_j)` against `grad_target(t)[j]`;
/// the analytic value follows the oracle until finite-difference
/// cancellation dominates. No autograd needed.
///
/// `μ = exp(ρ_iso)` is REML-selectable as one extra ρ axis.
///
/// `jacobian_cache_slot` and `jacobian_second_cache_slot` are interior-mutable
/// (`RwLock<Option<Arc<…>>>`) so the SAE outer loop can refresh them in place
/// each step without needing `&mut self` on the registry-held penalty (see
/// `refresh_caches` and [`crate::terms::sae::manifold::refresh_isometry_caches_from_atom`]).
/// Readers go through the [`Self::jacobian_cache`] / [`Self::jacobian_second_cache`]
/// accessors, which take the read lock briefly and clone the inner `Arc`
/// (refcount bump — no payload copy). Writers go through [`Self::refresh_caches`].
#[derive(Debug)]
pub struct IsometryPenalty {
    pub target: PsiSlice,
    pub reference: IsometryReference,
    /// Index of this penalty's strength `log μ_iso` inside the *local* rho
    /// view this penalty receives. Always `0` for now (single owned axis).
    pub rho_index: usize,
    /// Cached Jacobian `J ∈ ℝ^{n_obs × p × d}`, flattened row-major
    /// `(n_obs, p*d)`. The owning driver refreshes this each IFT outer step
    /// before invoking `value` / `grad_target`; in operator-only call sites
    /// (Hessian-vector products) the cache must be live. Access through
    /// [`Self::jacobian_cache`] / [`Self::set_jacobian_cache`].
    pub jacobian_cache_slot: RwLock<Option<Arc<Array2<f64>>>>,
    /// Optional cached per-row Jacobian *second derivative*
    /// `H_n ∈ ℝ^{p × d × d}`, flattened row-major as `(n_obs, p*d*d)`.
    /// `H_n[i, a, c] = ∂J_n[i, a] / ∂t_{n, c}`. Either this cache or
    /// `duchon_radial_source` must be present for exact isometry
    /// gradient/HVP calls. Access through [`Self::jacobian_second_cache`] /
    /// [`Self::set_jacobian_second_cache`].
    pub jacobian_second_cache_slot: RwLock<Option<Arc<Array2<f64>>>>,
    /// Optional radial-Duchon source used to build `jacobian_second_cache`
    /// analytically from `φ'(r)` and the public `φ''(r)` jet helper. This is
    /// the exact chain-rule path for callers that do not pre-cache `∂J/∂t`.
    pub duchon_radial_source: Option<Arc<IsometryDuchonRadialSource>>,
    /// Optional cached per-row Jacobian *third derivative*
    /// `K_n ∈ ℝ^{p × d × d × d}`, stored as an `Array3` with shape
    /// `(n_obs, p, d * d * d)` where the third axis packs `(a, c, d)` in
    /// row-major order `((a * d) + c) * d + dd`. `hvp` uses the full
    /// residual-curvature Hessian (proposal §4(b)):
    ///   B_{ab,cd} = K_{a,cd}^T W J_b + H_{a,c}^T W H_{b,d}
    ///             + H_{a,d}^T W H_{b,c} + J_a^T W K_{b,cd}.
    /// Either this cache or `duchon_radial_source` must be present for
    /// analytic `hvp` calls. Interior-mutable (mirrors
    /// `jacobian_second_cache_slot`) so the SAE outer loop can refresh `K` in
    /// place each step. Access through [`Self::third_decoder_derivative`] /
    /// [`Self::set_third_decoder_derivative`].
    pub third_decoder_derivative_slot: RwLock<Option<Arc<ndarray::Array3<f64>>>>,
    /// Output dimensionality `p` (column count of each per-row Jacobian).
    pub p_out: usize,
    /// Per-row behavioral metric in low-rank factored form. Defaults to
    /// `Identity` (the unweighted `J^T J` pullback). When `Factored`, all
    /// `g_n` contractions are done via `M_n = U_n^T J_n` (`r × d`), keeping
    /// memory and FLOPs scaling at `O(p · r · d)` per row instead of
    /// `O(p²)` per row.
    pub weight: WeightField,
    pub scalar_weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

pub(crate) struct IsometryHvpState<'a> {
    d: usize,
    n_obs: usize,
    p: usize,
    jac2: CowArray<'a, f64, Ix2>,
    jac3: CowArray<'a, f64, Ix3>,
    metric: IsometryMetricState,
    wj_rows: Vec<Array2<f64>>,
}

#[derive(Debug, Clone)]
struct IsometryMetricState {
    g: Array2<f64>,
    residual: Array2<f64>,
    metric_grad: Array2<f64>,
    normalizer: f64,
    trace_denominator: f64,
    residual_dot_g: f64,
}

impl IsometryMetricState {
    fn residual_direction(&self, delta_g: ArrayView2<'_, f64>, d: usize) -> (Array2<f64>, f64) {
        let n_obs = self.g.nrows();
        let dd = d * d;
        let mut delta_trace_sum = 0.0;
        for n in 0..n_obs {
            for a in 0..d {
                delta_trace_sum += delta_g[[n, a * d + a]];
            }
        }
        let delta_normalizer = delta_trace_sum / self.trace_denominator;
        let inv_norm = 1.0 / self.normalizer;
        let inv_norm_sq = inv_norm * inv_norm;
        let mut delta_residual = Array2::<f64>::zeros((n_obs, dd));
        for n in 0..n_obs {
            for k in 0..dd {
                delta_residual[[n, k]] =
                    delta_g[[n, k]] * inv_norm - self.g[[n, k]] * delta_normalizer * inv_norm_sq;
            }
        }
        (delta_residual, delta_normalizer)
    }

    fn metric_grad_direction(&self, delta_g: ArrayView2<'_, f64>, d: usize) -> Array2<f64> {
        let n_obs = self.g.nrows();
        let dd = d * d;
        let (delta_residual, delta_normalizer) = self.residual_direction(delta_g, d);
        let mut delta_residual_dot_g = 0.0;
        for n in 0..n_obs {
            for k in 0..dd {
                delta_residual_dot_g += delta_residual[[n, k]] * self.g[[n, k]];
                delta_residual_dot_g += self.residual[[n, k]] * delta_g[[n, k]];
            }
        }
        let inv_norm = 1.0 / self.normalizer;
        let inv_norm_sq = inv_norm * inv_norm;
        let delta_trace_coeff = delta_residual_dot_g * inv_norm_sq / self.trace_denominator
            - 2.0 * self.residual_dot_g * delta_normalizer * inv_norm_sq * inv_norm
                / self.trace_denominator;
        let mut out = Array2::<f64>::zeros((n_obs, dd));
        for n in 0..n_obs {
            for a in 0..d {
                for b in 0..d {
                    let k = a * d + b;
                    let mut value = delta_residual[[n, k]] * inv_norm
                        - self.residual[[n, k]] * delta_normalizer * inv_norm_sq;
                    if a == b {
                        value -= delta_trace_coeff;
                    }
                    out[[n, k]] = value;
                }
            }
        }
        out
    }
}

fn isometry_dg_entry(
    jac2: ArrayView2<'_, f64>,
    wj: ArrayView2<'_, f64>,
    n: usize,
    d: usize,
    p: usize,
    a: usize,
    b: usize,
    c: usize,
) -> f64 {
    let mut s = 0.0;
    for i in 0..p {
        s += jac2[[n, (i * d + a) * d + c]] * wj[[i, b]];
        s += wj[[i, a]] * jac2[[n, (i * d + b) * d + c]];
    }
    s
}

fn isometry_row_delta_g(
    jac2: ArrayView2<'_, f64>,
    wj: ArrayView2<'_, f64>,
    v: ArrayView1<'_, f64>,
    n: usize,
    d: usize,
    p: usize,
) -> Array2<f64> {
    let mut delta_g = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for b in 0..d {
            let mut s = 0.0;
            for c in 0..d {
                s += isometry_dg_entry(jac2, wj, n, d, p, a, b, c) * v[n * d + c];
            }
            delta_g[[a, b]] = s;
        }
    }
    delta_g
}

impl IsometryPenalty {
    pub const DEFAULT_VALUE_ON_MISSING_CACHE: f64 = 0.0;

    #[must_use]
    pub fn new_euclidean(target: PsiSlice, p_out: usize) -> Self {
        Self {
            target,
            reference: IsometryReference::Euclidean,
            rho_index: 0,
            jacobian_cache_slot: RwLock::new(None),
            jacobian_second_cache_slot: RwLock::new(None),
            duchon_radial_source: None,
            third_decoder_derivative_slot: RwLock::new(None),
            p_out,
            weight: WeightField::Identity,
            scalar_weight: 1.0,
            weight_schedule: None,
        }
    }

    /// Read-side accessor: takes the read lock briefly and clones the inner
    /// `Arc` (refcount bump only; no payload copy). Returns `None` when the
    /// cache has not been refreshed yet. Internally panics on poisoned lock
    /// — the lock only wraps an `Option<Arc<…>>`, so the write side cannot
    /// leave it in an invariant-violating state.
    #[must_use]
    pub fn jacobian_cache(&self) -> Option<Arc<Array2<f64>>> {
        self.jacobian_cache_slot
            .read()
            .expect("IsometryPenalty::jacobian_cache_slot poisoned")
            .clone()
    }

    /// Read-side accessor for the per-row Jacobian second derivative.
    /// Mirrors [`Self::jacobian_cache`].
    #[must_use]
    pub fn jacobian_second_cache(&self) -> Option<Arc<Array2<f64>>> {
        self.jacobian_second_cache_slot
            .read()
            .expect("IsometryPenalty::jacobian_second_cache_slot poisoned")
            .clone()
    }

    /// Per-step refresh entry point. Takes `&self` (no `&mut`) so the SAE
    /// outer loop can install fresh caches on an `Arc<IsometryPenalty>` held
    /// in the analytic-penalty registry without disturbing the surrounding
    /// dispatcher. Pass `None` for either argument to clear that cache (the
    /// dispatcher will then either fall back to the Duchon radial source if
    /// available, or return the zero safe default).
    pub fn refresh_caches(&self, jac: Option<Arc<Array2<f64>>>, jac2: Option<Arc<Array2<f64>>>) {
        *self
            .jacobian_cache_slot
            .write()
            .expect("IsometryPenalty::jacobian_cache_slot poisoned") = jac;
        *self
            .jacobian_second_cache_slot
            .write()
            .expect("IsometryPenalty::jacobian_second_cache_slot poisoned") = jac2;
    }

    /// In-place writer for just the Jacobian cache (used by callers that
    /// already own the radial Duchon source and only want to refresh `J`).
    pub fn set_jacobian_cache(&self, jac: Option<Arc<Array2<f64>>>) {
        *self
            .jacobian_cache_slot
            .write()
            .expect("IsometryPenalty::jacobian_cache_slot poisoned") = jac;
    }

    /// In-place writer for just the Jacobian second-derivative cache.
    pub fn set_jacobian_second_cache(&self, jac2: Option<Arc<Array2<f64>>>) {
        *self
            .jacobian_second_cache_slot
            .write()
            .expect("IsometryPenalty::jacobian_second_cache_slot poisoned") = jac2;
    }

    /// Read-side accessor for the per-row Jacobian third derivative `K`.
    /// Mirrors [`Self::jacobian_second_cache`].
    #[must_use]
    pub fn third_decoder_derivative(&self) -> Option<Arc<ndarray::Array3<f64>>> {
        self.third_decoder_derivative_slot
            .read()
            .expect("IsometryPenalty::third_decoder_derivative_slot poisoned")
            .clone()
    }

    /// In-place writer for just the Jacobian third-derivative cache `K`.
    pub fn set_third_decoder_derivative(&self, jac3: Option<Arc<ndarray::Array3<f64>>>) {
        *self
            .third_decoder_derivative_slot
            .write()
            .expect("IsometryPenalty::third_decoder_derivative_slot poisoned") = jac3;
    }
}

impl Clone for IsometryPenalty {
    fn clone(&self) -> Self {
        Self {
            target: self.target.clone(),
            reference: self.reference.clone(),
            rho_index: self.rho_index,
            jacobian_cache_slot: RwLock::new(self.jacobian_cache()),
            jacobian_second_cache_slot: RwLock::new(self.jacobian_second_cache()),
            duchon_radial_source: self.duchon_radial_source.clone(),
            third_decoder_derivative_slot: RwLock::new(self.third_decoder_derivative()),
            p_out: self.p_out,
            weight: self.weight.clone(),
            scalar_weight: self.scalar_weight,
            weight_schedule: self.weight_schedule.clone(),
        }
    }
}

impl IsometryPenalty {
    /// Attach a cached third decoder derivative
    /// `K_n[i, a, c, d] = ∂²J_n[i, a] / ∂t_{n, c} ∂t_{n, d}`, flattened
    /// row-major as `(n_obs, p * d * d * d)`. The Hessian-vector product
    /// uses the full residual-curvature term in addition to the metric
    /// Gauss-Newton piece.
    #[must_use]
    pub fn with_third_decoder_derivative(self, k: Arc<ndarray::Array3<f64>>) -> Self {
        self.set_third_decoder_derivative(Some(k));
        self
    }

    #[must_use]
    pub fn with_reference(mut self, reference: IsometryReference) -> Self {
        self.reference = reference;
        self
    }

    #[must_use]
    pub fn with_jacobian_cache(self, j: Arc<Array2<f64>>) -> Self {
        self.set_jacobian_cache(Some(j));
        self
    }

    #[must_use]
    pub fn with_jacobian_second_cache(self, h: Arc<Array2<f64>>) -> Self {
        self.set_jacobian_second_cache(Some(h));
        self
    }

    /// Attach radial Duchon decoder metadata so the exact `∂J/∂t` tensor can
    /// be rebuilt from the current target coordinates. A doc-test oracle for
    /// this path is: build `J(t)` from `duchon_radial_first_derivative_nd`,
    /// evaluate `grad_target(t)`, then central-difference `value(t ± h e_j)`;
    /// the analytic component should agree to finite-difference tolerance as
    /// `h` is refined before cancellation dominates.
    #[must_use]
    pub fn with_duchon_radial_source(mut self, source: Arc<IsometryDuchonRadialSource>) -> Self {
        self.duchon_radial_source = Some(source);
        self
    }

    /// Attach the gauge metric **from the single
    /// [`RowMetric`](gam_problem::RowMetric)** that also drives
    /// the reconstruction likelihood. This is the only way an `IsometryPenalty`
    /// acquires a non-identity behavioral metric: the independent
    /// `WeightField` setter has been removed so a gauge-metric ≠
    /// likelihood-metric state is structurally unrepresentable. The
    /// contraction-order invariant (`M_n = U_n^T J_n`, never materializing the
    /// `p × p` `W_n`) is preserved by the [`WeightField::Factored`] layout the
    /// metric emits.
    ///
    /// `p_out` is taken from the metric so the gauge's output dimension is
    /// pinned to the metric's.
    #[must_use]
    pub fn with_row_metric(mut self, metric: &gam_problem::RowMetric) -> Self {
        // Only a metric that drives the gauge installs a non-identity pullback
        // weight. A Euclidean metric reduces the gauge pullback to the bare
        // `J_nᵀ J_n`, so its `to_weight_field()` is `Identity` and the existing
        // (default-Identity) weight is left exactly as is — bit-for-bit the
        // pre-metric isotropic gauge. The output dimension is pinned to the
        // metric's regardless, so the gauge and likelihood agree on `p_out`.
        if metric.drives_gauge() {
            self.weight = metric.to_weight_field();
        }
        self.p_out = metric.p_out();
        self
    }

    impl_with_weight_schedule!(scalar_weight);

    fn missing_cache_default(&self, method: &str, detail: &str) {
        log::warn!(
            "IsometryPenalty::{method} missing required derivative state: {detail}; \
             returning the zero safe default"
        );
    }

    fn has_jacobian_cache(&self, method: &str) -> bool {
        if self.jacobian_cache().is_some() {
            true
        } else {
            self.missing_cache_default(method, "jacobian_cache is None");
            false
        }
    }

    fn has_jacobian_second_source(&self, method: &str) -> bool {
        if self.jacobian_second_cache().is_some() || self.duchon_radial_source.is_some() {
            true
        } else {
            self.missing_cache_default(
                method,
                "both jacobian_second_cache and duchon_radial_source are None",
            );
            false
        }
    }

    fn has_jacobian_third_source(&self, method: &str) -> bool {
        if self.third_decoder_derivative().is_some() || self.duchon_radial_source.is_some() {
            true
        } else {
            self.missing_cache_default(
                method,
                "both third_decoder_derivative cache and duchon_radial_source are None",
            );
            false
        }
    }

    /// Build `M_n = U_n^T J_n ∈ ℝ^{r_n × d}` for row `n`. For
    /// `WeightField::Identity`, `r_n = p` and `M_n = J_n`.
    ///
    /// This is the single contraction site where `W_n` (or its `U_n` factor)
    /// is consumed. Every value/grad/hvp path funnels through here, so the
    /// `(J^T U)(U^T J)` ordering invariant cannot be violated by accident.
    fn projected_jacobian_row(&self, n: usize, d: usize) -> Option<Array2<f64>> {
        let Some(jac) = self.jacobian_cache() else {
            self.missing_cache_default("projected_jacobian_row", "jacobian_cache is None");
            return None;
        };
        let jac_row = jac.row(n);
        let jac_slice = jac_row
            .as_slice()
            .expect("jacobian cache must be in standard row-major layout");
        match &self.weight {
            WeightField::Identity => {
                let p = self.p_out;
                let mut m = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        m[[i, a]] = jac_slice[i * d + a];
                    }
                }
                Some(m)
            }
            WeightField::Factored { u, rank, p_out } => {
                let u_row = u.row(n);
                let u_slice = u_row
                    .as_slice()
                    .expect("weight factor U must be in standard row-major layout");
                Some(WeightField::project_jac_row_with_u(
                    u_slice, jac_slice, *p_out, *rank, d,
                ))
            }
        }
    }

    /// Form `W_n J_n` without materializing `W_n`.
    fn weighted_jacobian_row(&self, n: usize, d: usize) -> Option<Array2<f64>> {
        let Some(jac) = self.jacobian_cache() else {
            self.missing_cache_default("weighted_jacobian_row", "jacobian_cache is None");
            return None;
        };
        let p = self.p_out;
        match &self.weight {
            WeightField::Identity => {
                let mut out = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        out[[i, a]] = jac[[n, i * d + a]];
                    }
                }
                Some(out)
            }
            WeightField::Factored { u, rank, p_out } => {
                assert_eq!(p, *p_out);
                let r = *rank;
                let m_n = self.projected_jacobian_row(n, d)?;
                let mut out = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        let mut s = 0.0;
                        for k in 0..r {
                            s += u[[n, i * r + k]] * m_n[[k, a]];
                        }
                        out[[i, a]] = s;
                    }
                }
                Some(out)
            }
        }
    }

    fn weighted_dot_decoder_vectors<F, G>(&self, n: usize, p: usize, x: F, y: G) -> f64
    where
        F: Fn(usize) -> f64,
        G: Fn(usize) -> f64,
    {
        match &self.weight {
            WeightField::Identity => {
                let mut s = 0.0;
                for i in 0..p {
                    s += x(i) * y(i);
                }
                s
            }
            WeightField::Factored { u, rank, p_out } => {
                assert_eq!(p, *p_out);
                let r = *rank;
                let mut s = 0.0;
                for k in 0..r {
                    let mut ux = 0.0;
                    let mut uy = 0.0;
                    for i in 0..p {
                        let uik = u[[n, i * r + k]];
                        ux += uik * x(i);
                        uy += uik * y(i);
                    }
                    s += ux * uy;
                }
                s
            }
        }
    }

    fn target_matrix(target: ArrayView1<'_, f64>, n_obs: usize, d: usize) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((n_obs, d));
        for n in 0..n_obs {
            for a in 0..d {
                out[[n, a]] = target[n * d + a];
            }
        }
        out
    }

    /// Second-order input-location derivative tensor of the Duchon decoder,
    /// flattened to `(n_obs, p_out · d²)` with column layout
    /// `i·d² + (a·d + c)`.
    ///
    /// Thin adapter over the shared [`radial_basis_cartesian_derivative`]
    /// engine: it owns the radial-jet evaluation and the radial→Cartesian map;
    /// here we only forward the source geometry.
    fn duchon_radial_jacobian_second(
        &self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
        source: &IsometryDuchonRadialSource,
    ) -> Result<Array2<f64>, BasisError> {
        assert_eq!(source.centers.ncols(), d);
        assert_eq!(source.radial_coefficients.nrows(), source.centers.nrows());
        assert_eq!(source.radial_coefficients.ncols(), self.p_out);
        let t = Self::target_matrix(target, n_obs, d);
        radial_basis_cartesian_derivative(
            2,
            t.view(),
            source.centers.view(),
            source.radial_coefficients.view(),
            source.length_scale,
            source.nullspace_order,
            source.power,
        )
    }

    /// Third-order input-location derivative tensor of the Duchon decoder,
    /// shaped `(n_obs, p_out, d³)` with last-axis layout `(a·d + c)·d + e`.
    ///
    /// Thin adapter over the shared [`radial_basis_cartesian_derivative`]
    /// engine; the flat `(n_obs, p_out · d³)` result is reshaped to the
    /// `Array3` consumed by the HVP path (row-major flatten of `(p_out, d³)`
    /// is exactly `i·d³ + idx`).
    fn duchon_radial_jacobian_third(
        &self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
        source: &IsometryDuchonRadialSource,
    ) -> Result<ndarray::Array3<f64>, BasisError> {
        assert_eq!(source.centers.ncols(), d);
        assert_eq!(source.radial_coefficients.nrows(), source.centers.nrows());
        assert_eq!(source.radial_coefficients.ncols(), self.p_out);
        let t = Self::target_matrix(target, n_obs, d);
        let flat = radial_basis_cartesian_derivative(
            3,
            t.view(),
            source.centers.view(),
            source.radial_coefficients.view(),
            source.length_scale,
            source.nullspace_order,
            source.power,
        )?;
        Ok(flat
            .into_shape_with_order((n_obs, self.p_out, d * d * d))
            .expect("radial_basis_cartesian_derivative order-3 output reshapes to (n_obs, p, d³)"))
    }

    fn jacobian_second<'a>(
        &'a self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
    ) -> Option<CowArray<'a, f64, Ix2>> {
        if let Some(jac2) = self.jacobian_second_cache() {
            // Clone the underlying Array2 to detach from the Arc — the
            // CowArray needs to outlive the temporary Arc returned by the
            // accessor. The clone is `n_obs × p·d²` floats, paid once per
            // grad_target / hvp_state invocation; same per-step cost as the
            // pre-refactor code path which also took ownership via
            // `jac2.view().to_owned()` semantics implicitly.
            return Some(CowArray::from((*jac2).clone()));
        }
        let source = self.duchon_radial_source.as_ref()?;
        match self.duchon_radial_jacobian_second(target, n_obs, d, source) {
            Ok(jac2) => Some(CowArray::from(jac2)),
            Err(err) => {
                self.missing_cache_default(
                    "jacobian_second",
                    &format!("failed to materialize Duchon radial second derivative: {err}"),
                );
                None
            }
        }
    }

    fn jacobian_third<'a>(
        &'a self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
    ) -> Option<CowArray<'a, f64, Ix3>> {
        if let Some(jac3) = self.third_decoder_derivative() {
            return Some(CowArray::from(jac3.as_ref().clone()));
        }
        let source = self.duchon_radial_source.as_ref()?;
        match self.duchon_radial_jacobian_third(target, n_obs, d, source) {
            Ok(jac3) => Some(CowArray::from(jac3)),
            Err(err) => {
                self.missing_cache_default(
                    "jacobian_third",
                    &format!("failed to materialize Duchon radial third derivative: {err}"),
                );
                None
            }
        }
    }

    pub(crate) fn hvp_state<'a>(
        &'a self,
        target: ArrayView1<'_, f64>,
    ) -> Option<IsometryHvpState<'a>> {
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        if !self.has_jacobian_cache("hvp")
            || !self.has_jacobian_second_source("hvp")
            || !self.has_jacobian_third_source("hvp")
        {
            return None;
        }
        let p = self.p_out;
        let jac2 = self.jacobian_second(target.view(), n_obs, d)?;
        let jac3 = self.jacobian_third(target.view(), n_obs, d)?;
        let g = self.pullback_metric(d)?;
        let metric = self.normalized_metric_state(g, n_obs, d)?;
        let mut wj_rows = Vec::with_capacity(n_obs);
        for n in 0..n_obs {
            wj_rows.push(self.weighted_jacobian_row(n, d)?);
        }
        Some(IsometryHvpState {
            d,
            n_obs,
            p,
            jac2,
            jac3,
            metric,
            wj_rows,
        })
    }

    pub(crate) fn hvp_with_precomputed_state(
        &self,
        state: &IsometryHvpState<'_>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        let d = state.d;
        let n_obs = state.n_obs;
        let p = state.p;
        let jac2 = &state.jac2;
        let jac3 = &state.jac3;
        let metric = &state.metric;
        let mut out = Array1::<f64>::zeros(v.len());
        let mut delta_g = Array2::<f64>::zeros((n_obs, d * d));
        for n in 0..n_obs {
            let wj = &state.wj_rows[n];
            let row_delta = isometry_row_delta_g(jac2.view(), wj.view(), v, n, d, p);
            for a in 0..d {
                for b in 0..d {
                    delta_g[[n, a * d + b]] = row_delta[[a, b]];
                }
            }
        }
        let delta_metric_grad = metric.metric_grad_direction(delta_g.view(), d);

        for n in 0..n_obs {
            let wj = &state.wj_rows[n];
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let dg = isometry_dg_entry(jac2.view(), wj.view(), n, d, p, a, b, c);
                        acc += dg * delta_metric_grad[[n, a * d + b]];
                    }
                }
                out[n * d + c] = mu * acc;
            }

            for c in 0..d {
                let mut acc_res = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let metric_grad = metric.metric_grad[[n, a * d + b]];
                        if metric_grad == 0.0 {
                            continue;
                        }
                        let mut bv = 0.0;
                        for dd in 0..d {
                            let vd = v[n * d + dd];
                            if vd == 0.0 {
                                continue;
                            }
                            let mut k_a_cd_w_j_b = 0.0;
                            for i in 0..p {
                                k_a_cd_w_j_b += jac3[[n, i, ((a * d) + c) * d + dd]] * wj[[i, b]];
                            }
                            let h_a_c_w_h_b_d = self.weighted_dot_decoder_vectors(
                                n,
                                p,
                                |i| jac2[[n, (i * d + a) * d + c]],
                                |i| jac2[[n, (i * d + b) * d + dd]],
                            );
                            let h_a_d_w_h_b_c = self.weighted_dot_decoder_vectors(
                                n,
                                p,
                                |i| jac2[[n, (i * d + a) * d + dd]],
                                |i| jac2[[n, (i * d + b) * d + c]],
                            );
                            let mut j_a_w_k_b_cd = 0.0;
                            for i in 0..p {
                                j_a_w_k_b_cd += wj[[i, a]] * jac3[[n, i, ((b * d) + c) * d + dd]];
                            }
                            bv +=
                                (k_a_cd_w_j_b + h_a_c_w_h_b_d + h_a_d_w_h_b_c + j_a_w_k_b_cd) * vd;
                        }
                        acc_res += metric_grad * bv;
                    }
                }
                out[n * d + c] += mu * acc_res;
            }
        }
        out
    }

    /// Per-row pullback metric `g_n = J_n^T W_n J_n = M_n^T M_n` with
    /// `M_n = U_n^T J_n ∈ ℝ^{r_n × d}`. Returns `(n_obs, d, d)` flattened
    /// row-major as `(n_obs, d*d)`.
    ///
    /// Cost per row: `O(p · r · d)` for the `M_n` build (single pass over
    /// `U_n` and `J_n`) plus `O(r · d²)` for `M_n^T M_n`. The `p × p` weight
    /// `W_n` is never materialized.
    pub fn pullback_metric(&self, latent_dim: usize) -> Option<Array2<f64>> {
        let Some(jac) = self.jacobian_cache() else {
            self.missing_cache_default("pullback_metric", "jacobian_cache is None");
            return None;
        };
        let n_obs = jac.nrows();
        let p = self.p_out;
        assert_eq!(jac.ncols(), p * latent_dim);
        let mut g_all = Array2::<f64>::zeros((n_obs, latent_dim * latent_dim));
        for n in 0..n_obs {
            // M_n = U_n^T J_n  (or J_n itself when W = I).
            let m = self.projected_jacobian_row(n, latent_dim)?;
            let r = m.nrows();
            // g_n = M_n^T M_n: (d × d) result, contracting r.
            for a in 0..latent_dim {
                for b in 0..latent_dim {
                    let mut s = 0.0;
                    for k in 0..r {
                        s += m[[k, a]] * m[[k, b]];
                    }
                    g_all[[n, a * latent_dim + b]] = s;
                }
            }
        }
        Some(g_all)
    }

    /// The scale normalizer `gbar = (1 / (N d)) Σ_n tr(g_n)` of the cached
    /// pullback metric — the single shared denominator the scale-invariant
    /// gauge divides every per-row metric by.
    ///
    /// `value` / `grad_*` / `hvp` consume this implicitly through
    /// [`Self::normalized_metric_state`]; the SAE arrow-Schur assembly cannot
    /// (it builds explicit per-row `htt` / `htbeta` / `hbb` curvature blocks
    /// from the raw pullback `g_n`, not through the trait operators), so it
    /// reads `gbar` here and folds `1/gbar²` into its Gauss-Newton curvature.
    /// That `1/gbar²` factor is exactly the frozen-normalizer Gauss-Newton
    /// block of the normalized residual `R_n = g_n/gbar − g^ref_n`: the raw
    /// block (the GN block of the *un-normalized* `½μ‖g_n − g^ref‖²`) scales
    /// ∝‖B‖⁴ in the decoder magnitude while the normalized gradient is
    /// scale-free, so without the factor the joint Newton step collapses and
    /// the proximal ridge saturates at 1e15 (#795). It stays PSD (a positive
    /// scalar on an already-PSD Gram block), so the Schur complement is
    /// unaffected. `None` when the metric is unavailable or degenerate, mirror-
    /// ing `normalized_metric_state`'s non-positive-normalizer guard.
    pub fn metric_normalizer(&self, latent_dim: usize) -> Option<f64> {
        let g = self.pullback_metric(latent_dim)?;
        let n_obs = g.nrows();
        let trace_denominator = (n_obs * latent_dim) as f64;
        let mut trace_sum = 0.0;
        for n in 0..n_obs {
            for a in 0..latent_dim {
                trace_sum += g[[n, a * latent_dim + a]];
            }
        }
        let normalizer = trace_sum / trace_denominator;
        (normalizer.is_finite() && normalizer > f64::MIN_POSITIVE).then_some(normalizer)
    }

    /// Reference metric per row for the normalized pullback metric, `(n_obs, d*d)`.
    fn reference_metric(&self, n_obs: usize, d: usize) -> CowArray<'_, f64, Ix2> {
        match &self.reference {
            IsometryReference::Euclidean => {
                let mut out = Array2::<f64>::zeros((n_obs, d * d));
                for n in 0..n_obs {
                    for a in 0..d {
                        out[[n, a * d + a]] = 1.0;
                    }
                }
                CowArray::from(out)
            }
            IsometryReference::UserSupplied(a) => {
                assert_eq!(a.nrows(), n_obs);
                assert_eq!(a.ncols(), d * d);
                CowArray::from(a.view())
            }
        }
    }

    /// Shared normalized metric state for the scale-invariant isometry gauge.
    ///
    /// The residual is `R_n = g_n / gbar - g_ref,n`, with
    /// `gbar = (1 / (N d)) Σ_n tr(g_n)`. The metric-gradient is the exact
    /// derivative of `0.5 Σ ||R_n||²` with respect to the raw pullback metrics:
    ///
    /// `A_n = R_n / gbar - (Σ_l R_l:g_l) I / (gbar² N d)`.
    ///
    /// All value, gradient, and HVP paths consume this state so the global
    /// normalizer's derivative is never detached.
    fn normalized_metric_state(
        &self,
        g: Array2<f64>,
        n_obs: usize,
        d: usize,
    ) -> Option<IsometryMetricState> {
        let dd = d * d;
        let trace_denominator = (n_obs * d) as f64;
        let mut trace_sum = 0.0;
        for n in 0..n_obs {
            for a in 0..d {
                trace_sum += g[[n, a * d + a]];
            }
        }
        let normalizer = trace_sum / trace_denominator;
        if !(normalizer.is_finite() && normalizer > f64::MIN_POSITIVE) {
            self.missing_cache_default(
                "normalized_metric_state",
                &format!(
                    "unit-average-speed normalizer is non-positive or non-finite: {normalizer}"
                ),
            );
            return None;
        }
        let g_ref = self.reference_metric(n_obs, d);
        let mut residual = Array2::<f64>::zeros((n_obs, dd));
        let inv_norm = 1.0 / normalizer;
        for n in 0..n_obs {
            for k in 0..dd {
                residual[[n, k]] = g[[n, k]] * inv_norm - g_ref[[n, k]];
            }
        }
        let mut residual_dot_g = 0.0;
        for n in 0..n_obs {
            for k in 0..dd {
                residual_dot_g += residual[[n, k]] * g[[n, k]];
            }
        }
        let trace_coeff = residual_dot_g / (normalizer * normalizer * trace_denominator);
        let mut metric_grad = Array2::<f64>::zeros((n_obs, dd));
        for n in 0..n_obs {
            for a in 0..d {
                for b in 0..d {
                    let k = a * d + b;
                    let mut value = residual[[n, k]] * inv_norm;
                    if a == b {
                        value -= trace_coeff;
                    }
                    metric_grad[[n, k]] = value;
                }
            }
        }
        Some(IsometryMetricState {
            g,
            residual,
            metric_grad,
            normalizer,
            trace_denominator,
            residual_dot_g,
        })
    }

    /// Exact closed-form gradient of the isometry penalty with respect to the
    /// cached decoder Jacobian `J ∈ ℝ^{n_obs × p × d}` (the autograd input that
    /// torch's `_IsometryPenaltyFn` differentiates). Returns the flattened
    /// `(n_obs, p*d)` layout that matches the Jacobian cache.
    ///
    /// Derivation (W-aware, reference-aware, weight-aware):
    ///
    ///   P        = ½ μ Σ_n ‖R_n‖²_F,
    ///   R_n      = g_n / gbar − g^ref_n,
    ///   gbar     = (1 / (N d)) Σ_n tr(g_n)
    ///   A_n      = ∂(P/μ)/∂g_n
    ///   ∂g_{ab}/∂J_{i,c}
    ///            = δ_{ca}(W J)_{i,b} + δ_{cb}(W J)_{i,a}   (W symmetric)
    ///   ∂P/∂J_{i,c}
    ///            = μ Σ_{a,b} A_{ab} ∂g_{ab}/∂J_{i,c}
    ///            = 2 μ Σ_b A_{cb} (W J)_{i,b}
    ///            = 2 μ ((W J) A)_{i,c}
    ///
    /// where `A` includes the exact derivative of the shared `gbar` normalizer.
    pub fn grad_jacobian(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        let p = self.p_out;
        let mut grad = Array2::<f64>::zeros((n_obs, p * d));
        if !self.has_jacobian_cache("grad_jacobian") {
            return grad;
        }
        let Some(g) = self.pullback_metric(d) else {
            return grad;
        };
        let Some(metric) = self.normalized_metric_state(g, n_obs, d) else {
            return grad;
        };
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        for n in 0..n_obs {
            let Some(wj) = self.weighted_jacobian_row(n, d) else {
                return Array2::<f64>::zeros((n_obs, p * d));
            };
            for i in 0..p {
                for c in 0..d {
                    let mut acc = 0.0;
                    for b in 0..d {
                        acc += metric.metric_grad[[n, c * d + b]] * wj[[i, b]];
                    }
                    grad[[n, i * d + c]] = 2.0 * mu * acc;
                }
            }
        }
        grad
    }
}

impl AnalyticPenalty for IsometryPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        if !self.has_jacobian_cache("value") {
            return Self::DEFAULT_VALUE_ON_MISSING_CACHE;
        }
        let Some(g) = self.pullback_metric(d) else {
            return Self::DEFAULT_VALUE_ON_MISSING_CACHE;
        };
        let Some(metric) = self.normalized_metric_state(g, n_obs, d) else {
            return Self::DEFAULT_VALUE_ON_MISSING_CACHE;
        };
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        let mut acc = 0.0;
        for n in 0..n_obs {
            for k in 0..(d * d) {
                let diff = metric.residual[[n, k]];
                acc += diff * diff;
            }
        }
        0.5 * mu * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Exact closed-form gradient, W-aware:
        //
        //   P     = ½ μ Σ_n ‖R_n‖²_F,   R_n = g_n / gbar − g^ref_n
        //   g_n   = J_n^T W_n J_n,      W_n = U_n U_n^T
        //   A_n   = ∂(P/μ)/∂g_n, including the exact derivative of
        //           gbar = (1 / (N d)) Σ_n tr(g_n)
        //   ∂g_{ab}/∂t_c
        //         = (H_{:,a,c})^T (W J)_{:,b}  +  (J_{:,a})^T W H_{:,b,c}
        //   ∂P/∂t_c
        //         = μ Σ_{a,b} A_{a,b} · ∂g_{ab}/∂t_c
        //
        // `H = ∂J/∂t` comes either from the live cache or from the radial
        // Duchon `φ''(r)` helper. The sign is positive: differentiating
        // `t - c` with respect to `t` contributes `+I`.
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        if !self.has_jacobian_cache("grad_target")
            || !self.has_jacobian_second_source("grad_target")
        {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(g) = self.pullback_metric(d) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(metric) = self.normalized_metric_state(g, n_obs, d) else {
            return Array1::<f64>::zeros(target.len());
        };
        let p = self.p_out;
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        let mut grad = Array1::<f64>::zeros(target.len());
        let Some(jac2) = self.jacobian_second(target, n_obs, d) else {
            return grad;
        };
        assert_eq!(jac2.ncols(), p * d * d);

        for n in 0..n_obs {
            let Some(wj) = self.weighted_jacobian_row(n, d) else {
                return grad;
            };
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let mut dg = 0.0;
                        for i in 0..p {
                            dg += jac2[[n, (i * d + a) * d + c]] * wj[[i, b]];
                            dg += wj[[i, a]] * jac2[[n, (i * d + b) * d + c]];
                        }
                        acc += metric.metric_grad[[n, a * d + b]] * dg;
                    }
                }
                grad[n * d + c] = mu * acc;
            }
        }
        grad
    }

    /// Fully analytic - wired through `radial_basis_cartesian_derivative`.
    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // Fully analytic isometry Hessian-vector product wired through the
        // shared `radial_basis_cartesian_derivative` engine when no
        // third-derivative cache is supplied.
        //
        // The full Hessian of P_iso = (μ/2) Σ_n ||J^T W J / gbar - G_ref||²_F
        // (per proposal §4(b)) is
        //   μ [Dgᵀ · ∂²(0.5||R||²)/∂g² · Dg + A · ∂²g],
        // where R = g/gbar - G_ref and A = ∂(0.5||R||²)/∂g includes the global
        // gbar derivative.
        //   B_{ab,cd} = K_{a,cd}^T W J_b + H_{a,c}^T W H_{b,d}
        //             + H_{a,d}^T W H_{b,c} + J_a^T W K_{b,cd},
        // where K is the third decoder derivative and H is the second.
        let Some(state) = self.hvp_state(target) else {
            return Array1::<f64>::zeros(v.len());
        };
        self.hvp_with_precomputed_state(&state, rho, v)
    }

    /// PSD majorizer-vector product `B_GN(target; ρ) v` for the **nonconvex**
    /// isometry penalty.
    ///
    /// The Gauss-Newton block differentiates the normalized residual
    /// `R = g/gbar - G_ref` itself and returns `μ DRᵀ DR v`. This is PSD by
    /// construction and includes the shared-normalizer derivative exactly;
    /// using only `∂g` would reintroduce scale coupling and would not be the
    /// Gauss-Newton operator of the objective being minimized.
    fn psd_majorizer_hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        if !self.has_jacobian_cache("psd_majorizer_hvp")
            || !self.has_jacobian_second_source("psd_majorizer_hvp")
        {
            return Array1::<f64>::zeros(v.len());
        }
        let Some(jac2) = self.jacobian_second(target, n_obs, d) else {
            return Array1::<f64>::zeros(v.len());
        };
        let Some(g) = self.pullback_metric(d) else {
            return Array1::<f64>::zeros(v.len());
        };
        let Some(metric) = self.normalized_metric_state(g, n_obs, d) else {
            return Array1::<f64>::zeros(v.len());
        };
        let p = self.p_out;
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        let mut out = Array1::<f64>::zeros(v.len());
        let mut wj_rows = Vec::with_capacity(n_obs);
        for n in 0..n_obs {
            let Some(wj) = self.weighted_jacobian_row(n, d) else {
                return Array1::<f64>::zeros(v.len());
            };
            wj_rows.push(wj);
        }
        let mut delta_g = Array2::<f64>::zeros((n_obs, d * d));
        for n in 0..n_obs {
            let row_delta = isometry_row_delta_g(jac2.view(), wj_rows[n].view(), v, n, d, p);
            for a in 0..d {
                for b in 0..d {
                    delta_g[[n, a * d + b]] = row_delta[[a, b]];
                }
            }
        }
        let (delta_residual, _delta_normalizer) = metric.residual_direction(delta_g.view(), d);
        let mut g_dot_delta_residual = 0.0;
        for n in 0..n_obs {
            for k in 0..(d * d) {
                g_dot_delta_residual += metric.g[[n, k]] * delta_residual[[n, k]];
            }
        }
        let inv_norm = 1.0 / metric.normalizer;
        let inv_norm_sq = inv_norm * inv_norm;
        for n in 0..n_obs {
            let wj = &wj_rows[n];
            for c in 0..d {
                let mut trace_dg = 0.0;
                for a in 0..d {
                    trace_dg += isometry_dg_entry(jac2.view(), wj.view(), n, d, p, a, a, c);
                }
                let delta_normalizer_c = trace_dg / metric.trace_denominator;
                let mut acc = -delta_normalizer_c * inv_norm_sq * g_dot_delta_residual;
                for a in 0..d {
                    for b in 0..d {
                        let dg = isometry_dg_entry(jac2.view(), wj.view(), n, d, p, a, b, c);
                        acc += dg * inv_norm * delta_residual[[n, a * d + b]];
                    }
                }
                out[n * d + c] = mu * acc;
            }
        }
        out
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // P(ρ) = ½ μ · S, where S is the (ρ-independent) Frobenius sum and
        // μ = exp(ρ_iso). So ∂P/∂ρ_iso = P.
        let mut out = Array1::<f64>::zeros(self.rho_count());
        out[self.rho_index] = self.value(target, rho);
        out
    }

    fn rho_count(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "isometry"
    }

    impl_scalar_apply_schedule!(scalar_weight);
}
