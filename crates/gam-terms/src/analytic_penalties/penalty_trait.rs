use super::*;

pub(crate) use gam_problem::{LOG_STRENGTH_MAX, LOG_STRENGTH_MIN, checked_exp_log_strength};

// ---------------------------------------------------------------------------
// Common trait
// ---------------------------------------------------------------------------

/// Whether a penalty's target is a slice of `β` (decoder coefficients), a
/// slice of extension coordinates (per-observation latent field, e.g.
/// `LatentCoordValues`),
/// or a slice of `ρ` (a hyperparameter sub-block — rare, used by hyperpriors
/// that we don't yet ship analytically).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyTier {
    Beta,
    Psi,
    Rho,
}

/// The unit one analytic-penalty ρ coordinate is measured in (#4266).
///
/// A consumer that intersects a published face with a search domain of its own
/// needs this. The resolvability face
/// [`precision_box`](gam_problem::precision_box) is stated in e-folds, so it
/// says something about a coordinate that enters through `exp` and nothing at
/// all about a coordinate carrying the units of a data column; applying it to
/// the second is exactly the hand-supplied box #4266 is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhoCoordinateKind {
    /// An e-fold offset on a positive strength: the coordinate reaches the
    /// penalty only through `exp`, or through a base weight's logarithm.
    LogStrength,
    /// An ordinary additive real coordinate — a location, a regression offset,
    /// or the argument of a map that is not a logarithm. It carries the units
    /// of whatever it is added to, so its published face is the only face it
    /// has.
    Location,
}

/// Reference for the column / coordinate range a penalty operates over.
///
/// Mirrors `BlockwisePenalty::col_range` for the β tier and is the natural
/// per-observation flat index for the extension-coordinate tier (matching the
/// `LatentCoordValues` row-major flat layout: `n * d + a`).
#[derive(Debug, Clone)]
pub struct PsiSlice {
    /// Inclusive-start, exclusive-end flat range into the underlying ext-coordinate vector.
    pub range: std::ops::Range<usize>,
    /// For latent-coordinate slices: the latent dimensionality, used to
    /// reshape the flat slice into per-row `(n_obs, d)` blocks.
    pub latent_dim: Option<usize>,
}

impl PsiSlice {
    #[must_use]
    pub fn full(len: usize, latent_dim: Option<usize>) -> Self {
        Self {
            range: 0..len,
            latent_dim,
        }
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
}

/// Resolve the exact learnable strength `base_weight · exp(rho)` in log space.
///
/// The effective log-strength `ln|base_weight| + rho` must lie in the closed
/// [`LOG_STRENGTH_MIN`, `LOG_STRENGTH_MAX`] domain. Values outside that domain
/// are rejected instead of saturated: a plateau would make the evaluated
/// value constant while analytic `rho` derivatives remain nonzero. Computing
/// the product as `sign(base_weight) · exp(ln|base_weight| + rho)` also avoids
/// an overflowing intermediate `exp(rho)` when a very small base permits a
/// large legal coordinate.
pub fn resolve_learnable_weight(base_weight: f64, rho: f64) -> Result<f64, String> {
    if base_weight == 0.0 {
        return Err(
            "a multiplicatively learnable weight requires a nonzero base; zero would make its rho coordinate structurally dead"
                .to_string(),
        );
    }
    if !(base_weight.is_finite() && rho.is_finite()) {
        return Err(format!(
            "learnable weight requires finite base and coordinate; got base_weight={base_weight}, rho={rho}"
        ));
    }
    let log_base = base_weight.abs().ln();
    let (lower, upper) = (LOG_STRENGTH_MIN - log_base, LOG_STRENGTH_MAX - log_base);
    if !(lower..=upper).contains(&rho) {
        return Err(format!(
            "learnable coordinate must be in [{lower}, {upper}] so its effective log strength is in [{LOG_STRENGTH_MIN}, {LOG_STRENGTH_MAX}]; got {rho}"
        ));
    }
    // Map the two emitted faces back to their mathematical effective values
    // exactly. This is not saturation: values beyond either face were refused
    // above. It only removes one subtraction/addition roundoff at a legal face.
    let log_strength = if rho == lower {
        LOG_STRENGTH_MIN
    } else if rho == upper {
        LOG_STRENGTH_MAX
    } else {
        log_base + rho
    };
    Ok(checked_exp_log_strength(log_strength)
        .map_err(|error| error.to_string())?
        .copysign(base_weight))
}
pub fn learnable_weight_coordinate_domain(base_weight: f64) -> Result<Option<(f64, f64)>, String> {
    if base_weight == 0.0 {
        return Ok(None);
    }
    if !base_weight.is_finite() {
        return Err(format!(
            "learnable weight domain requires a finite base; got {base_weight}"
        ));
    }
    let log_base = base_weight.abs().ln();
    Ok(Some((
        LOG_STRENGTH_MIN - log_base,
        LOG_STRENGTH_MAX - log_base,
    )))
}

/// Exact strength for trait methods whose owning evaluation seam has already
/// called `AnalyticPenalty::validate_rho`. Keeping this preconditioned helper
/// private prevents an unchecked public plateau/error path.
pub(crate) fn validated_learnable_weight(base_weight: f64, rho: f64) -> f64 {
    resolve_learnable_weight(base_weight, rho)
        .expect("analytic-penalty rho must be validated before strength evaluation")
}

pub(crate) fn validated_exp_log_strength(log_strength: f64) -> f64 {
    checked_exp_log_strength(log_strength)
        .expect("analytic-penalty rho must be validated before precision evaluation")
}

/// Uniform interface implemented by every analytic penalty in this module.
///
/// `target` is the relevant slice of the β or extension-coordinate vector, viewed as
/// a flat `ArrayView1`. The owning REML driver is responsible for slicing the
/// global parameter vector before calling, and for routing the returned
/// gradient back into the correct global indices.
pub trait AnalyticPenalty: Send + Sync {
    /// Tier the target lives in (β or ext-coord).
    fn tier(&self) -> PenaltyTier;

    /// Validate the penalty-local outer-rho vector before any value or
    /// derivative method consumes it. Implementations with a multiplicative
    /// base weight override this to validate `ln|base| + rho`; the default is
    /// the unit-base log-strength domain.
    fn validate_rho(&self, rho: ArrayView1<'_, f64>) -> Result<(), String> {
        if rho.len() != self.rho_count() {
            return Err(format!(
                "analytic penalty `{}` rho length {} != declared {}",
                self.name(),
                rho.len(),
                self.rho_count()
            ));
        }
        for (axis, &value) in rho.iter().enumerate() {
            checked_exp_log_strength(value).map_err(|error| {
                format!(
                    "analytic penalty `{}` rho axis {axis}: {error}",
                    self.name()
                )
            })?;
        }
        Ok(())
    }

    /// Per-local-coordinate legal intervals, one per ρ coordinate.
    ///
    /// Every face is finite (#4266). A penalty owns the units of its own
    /// coordinates, so it is the only place a face can be derived; a consumer
    /// handed an infinite face has to invent a finite one, and the invented
    /// one is a hand-supplied box. A coordinate whose own arithmetic bounds it
    /// nowhere still has the face where the quantity it feeds stops being a
    /// representable strength, which is what [`LOG_STRENGTH_MIN`] and
    /// [`LOG_STRENGTH_MAX`] state for the default log-strength coordinate.
    fn rho_coordinate_domains(&self) -> Result<Vec<(f64, f64)>, String> {
        Ok(vec![(LOG_STRENGTH_MIN, LOG_STRENGTH_MAX); self.rho_count()])
    }

    /// The unit of each local ρ coordinate, in the order
    /// [`Self::rho_coordinate_domains`] publishes their faces (#4266). The
    /// default is [`RhoCoordinateKind::LogStrength`] throughout, matching the
    /// default face `[LOG_STRENGTH_MIN, LOG_STRENGTH_MAX]`.
    fn rho_coordinate_kinds(&self) -> Vec<RhoCoordinateKind> {
        vec![RhoCoordinateKind::LogStrength; self.rho_count()]
    }

    /// Scalar penalty contribution `P(target; ρ)`. The strength factor
    /// `exp(ρ)` (or whatever parameterization the penalty uses) is folded in.
    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64;

    /// Gradient `∂P/∂target`, same length as `target`.
    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64>;

    /// Diagonal of the Hessian `diag(∂²P/∂target²)` when the Hessian is
    /// block-diagonal. Returns `None` for penalties whose Hessian is dense
    /// (Isometry); those implement [`Self::hvp`] instead. The default
    /// signals "no closed-form diagonal" by returning `None` for any
    /// non-empty target — concrete penalties either override with their
    /// own analytic diagonal or rely on the matrix-free `hvp` path.
    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        assert!(
            rho.iter().all(|value| value.is_finite()),
            "analytic-penalty rho must be finite"
        );
        if target.is_empty() {
            Some(Array1::zeros(0))
        } else {
            None
        }
    }

    /// Hessian-vector product `H v = (∂²P/∂target²) v`, in closed form.
    ///
    /// The default covers every penalty whose Hessian is diagonal: it reads the
    /// analytic [`Self::hessian_diag`] and forms `diag ⊙ v`. Penalties with a
    /// dense (non-diagonal) Hessian — e.g. `IsometryPenalty`,
    /// `SheafConsistencyPenalty`, the orthogonality / nuclear-norm family —
    /// return `None` from `hessian_diag` and supply their own analytic `hvp`
    /// override (Laplacian/Gram-vector products). There is no finite-difference
    /// path: a penalty that reaches the default without a closed-form diagonal
    /// is a programming error and panics rather than silently differencing its
    /// own gradient (SPEC: finite differences are never used outside tests).
    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let diag = self.hessian_diag(target, rho).unwrap_or_else(|| {
            // SAFETY: programming-error invariant, never a runtime/data condition.
            // A penalty whose Hessian is non-diagonal MUST override `hvp` with its
            // closed-form Hessian-vector product; reaching this default means the
            // impl is missing that override. SPEC forbids a finite-difference
            // fallback outside tests, so there is no recoverable path — failing
            // loud here is the contract.
            panic!(
                "AnalyticPenalty::hvp default reached for `{}`, whose Hessian is \
                 not diagonal (hessian_diag returned None). Such a penalty must \
                 override `hvp` with its closed-form Hessian-vector product; the \
                 default never finite-differences.",
                self.name()
            )
        });
        assert_eq!(diag.len(), v.len(), "hvp dimension mismatch");
        let mut out = Array1::<f64>::zeros(v.len());
        for i in 0..v.len() {
            out[i] = diag[i] * v[i];
        }
        out
    }

    /// Diagonal of a **PSD majorizer** of the Hessian — the positive
    /// re-weighted-ℓ₂ / MM surrogate `diag(B(target; ρ))` with
    /// `B ⪰ ∂²P/∂target²` everywhere and `B ⪰ 0`. This is a *different*
    /// operator from [`Self::hessian_diag`]: for nonconvex penalties (log
    /// sparsity, smooth-threshold) the exact Hessian is indefinite, but the inner
    /// Newton / PIRLS solve and the log-det / preconditioner pipeline require
    /// a PSD curvature block. For convex penalties the majorizer coincides
    /// with the exact Hessian, so the default simply delegates to
    /// [`Self::hessian_diag`]; nonconvex penalties override.
    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        self.hessian_diag(target, rho)
    }

    /// Matrix-vector product against the **PSD majorizer** `B(target; ρ) v`
    /// (see [`Self::psd_majorizer_diag`]). For convex penalties this is the
    /// exact Hessian-vector product, so the default delegates to
    /// [`Self::hvp`]; nonconvex penalties override to return their PSD
    /// surrogate instead of the indefinite true Hessian.
    fn psd_majorizer_hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        if let Some(diag) = self.psd_majorizer_diag(target, rho) {
            assert_eq!(diag.len(), v.len(), "psd_majorizer_hvp dimension mismatch");
            let mut out = Array1::<f64>::zeros(v.len());
            for i in 0..v.len() {
                out[i] = diag[i] * v[i];
            }
            return out;
        }
        self.hvp(target, rho, v)
    }

    /// Gradient of the penalty value w.r.t. each owned ρ-axis. Length equals
    /// [`Self::rho_count`].
    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64>;

    /// Number of REML-selectable hyperparameter axes this penalty contributes
    /// to the outer ρ vector.
    fn rho_count(&self) -> usize;

    /// Human-readable identifier for diagnostics / logging.
    fn name(&self) -> &str;
}

/// Emit the standard learnable-scalar-weight [`AnalyticPenalty::grad_rho`] for a
/// penalty whose single owned ρ-axis is the (optionally learnable) log-weight at
/// `self.rho_index`, gated by `self.learnable_weight`. Invoke inside the `impl
/// AnalyticPenalty for …` block.
macro_rules! impl_learnable_weight_grad_rho {
    () => {
        fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
            if !self.learnable_weight {
                return Array1::<f64>::zeros(0);
            }
            let mut out = Array1::<f64>::zeros(1);
            out[self.rho_index] = self.value(target, rho);
            out
        }
    };
}

/// Emit the standard learnable-scalar-weight [`AnalyticPenalty::rho_count`]:
/// one ρ-axis when the weight is learnable, none otherwise. Invoke inside the
/// `impl AnalyticPenalty for …` block.
macro_rules! impl_learnable_weight_rho_count {
    () => {
        fn rho_count(&self) -> usize {
            usize::from(self.learnable_weight)
        }
    };
}

macro_rules! impl_learnable_weight_domain {
    ($field:ident) => {
        fn validate_rho(&self, rho: ArrayView1<'_, f64>) -> Result<(), String> {
            if rho.len() != self.rho_count() {
                return Err(format!(
                    "analytic penalty `{}` rho length {} != declared {}",
                    self.name(),
                    rho.len(),
                    self.rho_count()
                ));
            }
            if self.learnable_weight {
                resolve_learnable_weight(self.$field, rho[self.rho_index]).map_err(|error| {
                    format!("analytic penalty `{}`: {error}", self.name())
                })?;
            }
            Ok(())
        }

        fn rho_coordinate_domains(&self) -> Result<Vec<(f64, f64)>, String> {
            if !self.learnable_weight {
                return Ok(Vec::new());
            }
            let domain = learnable_weight_coordinate_domain(self.$field)?.ok_or_else(|| {
                format!(
                    "analytic penalty `{}` cannot expose a learnable coordinate with zero base weight",
                    self.name()
                )
            })?;
            Ok(vec![domain])
        }
    };
}
