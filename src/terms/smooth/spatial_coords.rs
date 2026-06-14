//! Spatial length-scale / log-kappa / anisotropy / measure-jet / constant-
//! curvature *coordinate* machinery: the `SpatialLogKappaCoords` carrier, the
//! psi <-> length-scale/kappa maps, anisotropy log-scale get/set, the spatial
//! optimization-options type, and the pilot anisotropy initializer. Pure
//! coordinate policy shared by the design, fit, and optimization arms.

use super::*;

pub(crate) struct SpatialPsiDerivative {
    // These are derivatives with respect to psi = log(kappa), not log(length_scale).
    pub penalty_index: usize,
    pub penalty_indices: Vec<usize>,
    pub global_range: Range<usize>,
    pub total_p: usize,
    pub x_psi_local: Array2<f64>,
    pub s_psi_components_local: Vec<Array2<f64>>,
    pub x_psi_psi_local: Array2<f64>,
    pub s_psi_psi_components_local: Vec<Array2<f64>>,
    pub aniso_group_id: Option<usize>,
    /// Pre-computed cross-derivative design matrices for other axes
    /// in the same aniso group: Vec of (axis_offset_in_group, matrix).
    pub aniso_cross_designs: Option<Vec<(usize, Array2<f64>)>>,
    /// On-demand cross-penalty second derivatives ∂²S_m/∂ψ_a∂ψ_b for axes in
    /// the same anisotropy group. The input is the other axis offset in the
    /// group, and the output is one local penalty matrix per active penalty.
    pub aniso_cross_penalty_provider: Option<
        std::sync::Arc<
            dyn Fn(usize) -> Result<Vec<Array2<f64>>, EstimationError> + Send + Sync + 'static,
        >,
    >,
    /// Optional implicit design-derivative operator (shared across all axes
    /// in the same aniso group). When present, `x_psi_local` and
    /// `x_psi_psi_local` may be zero-sized, and design-derivative matvecs
    /// should go through this operator using `implicit_axis` as the axis index.
    pub implicit_operator: Option<std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>>,
    /// Which axis in the implicit operator this entry corresponds to.
    pub implicit_axis: usize,
}


#[derive(Debug, Clone)]
pub(crate) struct SpatialLogKappaCoords {
    /// Flattened ψ values. For isotropic terms, one entry per term.
    /// For anisotropic terms, d entries per term (one ψ_a per axis).
    values: Array1<f64>,
    /// Dimensionality of each term: 1 for isotropic, d for anisotropic.
    dims_per_term: Vec<usize>,
}


/// Which end of the ψ bound the shared `aniso_bounds_from_data` helper is
/// computing. The lower end uses `-max_length_scale.ln()` as the pure-Duchon
/// fallback and the `.0` element of `spatial_term_psi_bounds`; the upper end
/// uses `-min_length_scale.ln()` and `.1`. Everything else is identical.
#[derive(Clone, Copy)]
enum AnisoBoundEnd {
    Lower,
    Upper,
}


impl SpatialLogKappaCoords {
    /// Construct from an explicit dims layout plus values.
    pub(crate) fn new_with_dims(values: Array1<f64>, dims_per_term: Vec<usize>) -> Self {
        assert_eq!(
            values.len(),
            dims_per_term.iter().sum::<usize>(),
            "SpatialLogKappaCoords: values length {} != sum of dims_per_term {}",
            values.len(),
            dims_per_term.iter().sum::<usize>(),
        );
        Self {
            values,
            dims_per_term,
        }
    }

    /// Isotropic initialization (backward-compatible path).
    pub(crate) fn from_length_scales(
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut out = Array1::<f64>::zeros(term_indices.len());
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            // Constant-curvature: the single ψ slot is the raw signed κ, seeded
            // from the spec (default κ = 0). The −ln(length_scale) convention is
            // log-κ semantics and must not touch the raw-κ coordinate; the κ
            // window projection happens later via `clamp_to_bounds`. Mirrors the
            // aniso constructor's κ branch.
            if let Some(cc) = constant_curvature_term_spec(spec, term_idx) {
                out[slot] = cc.kappa;
                continue;
            }
            let length_scale = get_spatial_length_scale(spec, term_idx)
                .unwrap_or(options.min_length_scale)
                .clamp(options.min_length_scale, options.max_length_scale);
            out[slot] = -length_scale.ln();
        }
        Self {
            values: out,
            dims_per_term: vec![1; term_indices.len()],
        }
    }

    /// Anisotropic-aware initialization.
    ///
    /// Initialization strategy (per math team recommendation): standardize the
    /// knot cloud axiswise, then run the existing isotropic κ initializer in
    /// the standardized space. This reuses the trusted isotropic initializer
    /// and gives initial η_a = −ln(σ_a) + mean(ln(σ_a)), which satisfies
    /// Ση_a = 0 by construction.
    ///
    /// For each term, checks whether it has `aniso_log_scales` set on its basis spec.
    /// - If isotropic (no aniso_log_scales, or 1-D): 1 entry = −ln(length_scale).
    /// - If anisotropic with a scalar length scale: d entries, one ψ_a per axis.
    ///   Initialized as ψ_a = −ln(length_scale) + η_a  where η_a are the existing
    ///   aniso_log_scales (which sum to zero). Multi-dimensional terms without
    ///   explicit anisotropy stay scalar here so the seed dimensionality matches
    ///   `spatial_dims_per_term`.
    /// - If pure Duchon anisotropic: d - 1 free entries store the leading η_a
    ///   values directly; the final axis is reconstructed to keep Ση_a = 0.
    pub(crate) fn from_length_scales_aniso(
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut vals = Vec::new();
        let mut dims = Vec::new();
        for &term_idx in term_indices {
            // Measure-jet: dial coordinates seeded directly from the term's
            // realized (α, τ[, s]); the −ln(length_scale) convention below is
            // κ-semantics and never applies to dials.
            if let Some(mj) = measure_jet_term_spec(spec, term_idx) {
                let seed = measure_jet_psi_seed(mj);
                dims.push(seed.len());
                vals.extend(seed);
                continue;
            }
            // Constant-curvature: one signed κ slot seeded from the spec's κ
            // (clamped feasible). The −ln(length_scale) convention below is
            // log-κ semantics and must not touch the raw-κ coordinate. Bounds
            // are unavailable here (no data view), so this is the raw spec κ;
            // `reseed_from_data` / `clamp_to_bounds` later project it feasible.
            if let Some(cc) = constant_curvature_term_spec(spec, term_idx) {
                vals.push(cc.kappa);
                dims.push(1);
                continue;
            }
            let length_scale = get_spatial_length_scale(spec, term_idx)
                .unwrap_or(options.min_length_scale)
                .clamp(options.min_length_scale, options.max_length_scale);
            let psi_bar = -length_scale.ln(); // global scale = −ln(length_scale)

            if spatial_term_uses_per_axis_psi(spec, term_idx) {
                // Per-axis anisotropy is enrolled in the joint outer vector:
                // ψ_a = ψ̄ + η_a, one slot per axis. The hyper_dirs builder
                // produces matching per-axis derivatives in
                // `try_build_spatial_term_log_kappa_aniso_derivativeinfos`.
                let d = get_spatial_feature_dim(spec, term_idx).unwrap_or(1);
                let eta_raw = get_spatial_aniso_log_scales(spec, term_idx)
                    .expect("predicate guarantees aniso_log_scales is Some");
                let eta = center_aniso_log_scales(&eta_raw);
                for &eta_a in &eta {
                    vals.push(psi_bar + eta_a);
                }
                dims.push(d);
            } else {
                // Isotropic enrollment — either a 1-D term, a multi-D term
                // without explicit anisotropy, or a basis (e.g. Duchon) whose
                // η is a fixed geometry parameter rather than a REML hyper
                // axis. Exactly one ψ̄ slot, matching the single
                // `SpatialPsiDerivative` produced by
                // `try_build_spatial_term_log_kappa_derivativeinfo`.
                vals.push(psi_bar);
                dims.push(1);
            }
        }
        Self {
            values: Array1::from_vec(vals),
            dims_per_term: dims,
        }
    }

    /// Isotropic lower bounds derived from per-term data geometry.
    /// Each entry gets the ψ_lo bound returned by `spatial_term_psi_bounds`
    /// for the corresponding term, intersected with the options window.
    pub(crate) fn lower_bounds_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut values = Array1::<f64>::zeros(term_indices.len());
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            values[slot] = spatial_term_psi_bounds(data, spec, term_idx, options).0;
        }
        Self {
            values,
            dims_per_term: vec![1; term_indices.len()],
        }
    }

    /// Isotropic upper bounds derived from per-term data geometry.
    pub(crate) fn upper_bounds_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut values = Array1::<f64>::zeros(term_indices.len());
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            values[slot] = spatial_term_psi_bounds(data, spec, term_idx, options).1;
        }
        Self {
            values,
            dims_per_term: vec![1; term_indices.len()],
        }
    }

    /// Anisotropic-aware lower bounds derived from per-term data geometry.
    /// For hybrid anisotropic terms the scalar ψ_lo bound applies to the
    /// mean `ψ̄`, not directly to every raw axis coordinate `ψ_a = ψ̄ + η_a`.
    /// Shift each axis by the current centered `η_a` so projecting/clamping
    /// the seed moves only the global scale direction and does not silently
    /// shrink anisotropy that is already consistent with the current
    /// `length_scale`.
    ///
    /// Pure Duchon anisotropy is structurally different: its stored
    /// coordinates are (d-1) free η_a values representing log axis-scale
    /// ratios, NOT log-κ. For those terms the κ-range geometry bound is
    /// over-restrictive (η_a = ±5 is normal, but that corresponds to 7+
    /// orders of magnitude in κ-space and would be rejected by the data
    /// window). Fall back to the options window `[-ln(max_ls), -ln(min_ls)]`
    /// for those coordinates — that's the same bound the pre-data-geometry
    /// code used, which is calibrated to allow legitimate anisotropy.
    pub(crate) fn lower_bounds_aniso_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        dims_per_term: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        Self::aniso_bounds_from_data(
            data,
            spec,
            term_indices,
            dims_per_term,
            options,
            AnisoBoundEnd::Lower,
        )
    }

    /// Anisotropic-aware upper bounds derived from per-term data geometry.
    /// See `lower_bounds_aniso_from_data` for the hybrid-aniso offsetting and
    /// pure-Duchon dispatch rationale.
    pub(crate) fn upper_bounds_aniso_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        dims_per_term: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        Self::aniso_bounds_from_data(
            data,
            spec,
            term_indices,
            dims_per_term,
            options,
            AnisoBoundEnd::Upper,
        )
    }

    /// Shared implementation for the lower/upper aniso bounds. The bound end
    /// only changes which options scale (`max_length_scale` vs
    /// `min_length_scale`) becomes the pure-Duchon fallback bound and which
    /// element of the `(lo, hi)` data-geometry tuple is consumed; the
    /// per-term cursor walk and aniso-offset handling are identical.
    fn aniso_bounds_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        dims_per_term: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
        end: AnisoBoundEnd,
    ) -> Self {
        assert_eq!(term_indices.len(), dims_per_term.len());
        let total: usize = dims_per_term.iter().sum();
        let mut values = Array1::<f64>::zeros(total);
        let mut cursor = 0;
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            let d = dims_per_term[slot];
            // Measure-jet: per-coordinate dial boxes, never κ-window geometry
            // (which would reject legitimate dial values outright).
            if let Some(mj) = measure_jet_term_spec(spec, term_idx) {
                let bounds = measure_jet_psi_bound_values(mj, matches!(end, AnisoBoundEnd::Upper));
                for (offset, bound) in bounds.into_iter().enumerate() {
                    if offset < d {
                        values[cursor + offset] = bound;
                    }
                }
                cursor += d;
                continue;
            }
            // Constant-curvature: the single signed-κ box from the data chart
            // window (symmetric about κ = 0), never a κ = log-scale window.
            if constant_curvature_term_spec(spec, term_idx).is_some() {
                let (lo, hi) = constant_curvature_kappa_bounds(data, spec, term_idx);
                if d >= 1 {
                    values[cursor] = match end {
                        AnisoBoundEnd::Lower => lo,
                        AnisoBoundEnd::Upper => hi,
                    };
                }
                cursor += d;
                continue;
            }
            let psi_bound = {
                let (lo, hi) = spatial_term_psi_bounds(data, spec, term_idx, options);
                match end {
                    AnisoBoundEnd::Lower => lo,
                    AnisoBoundEnd::Upper => hi,
                }
            };
            let axis_offsets = if d <= 1 {
                vec![0.0; d]
            } else {
                get_spatial_aniso_log_scales(spec, term_idx)
                    .filter(|eta| eta.len() == d)
                    .map(|eta| center_aniso_log_scales(&eta))
                    .unwrap_or_else(|| vec![0.0; d])
            };
            for offset in 0..d {
                values[cursor + offset] = psi_bound + axis_offsets[offset];
            }
            cursor += d;
        }
        Self {
            values,
            dims_per_term: dims_per_term.to_vec(),
        }
    }

    /// Rewrite any ψ entries whose originating term lacks an explicit
    /// `length_scale` so they sit at the midpoint of the per-term data-derived
    /// ψ window. Used so the outer optimizer starts inside the physically
    /// meaningful region instead of at an arbitrary `options.max_length_scale`
    /// derived seed. For terms with an explicit length_scale, the user's
    /// choice is respected. Anisotropy offsets η_a (those stored by
    /// `from_length_scales_aniso`) are preserved: we re-center around the new
    /// ψ̄, keeping Ση_a = 0.
    pub(crate) fn reseed_from_data(
        mut self,
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        assert_eq!(term_indices.len(), self.dims_per_term.len());
        let mut cursor = 0;
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            let d = self.dims_per_term[slot];
            // Measure-jet dials are seeded from the realized spec and must
            // not be recentered into a κ data window.
            if measure_jet_term_spec(spec, term_idx).is_some() {
                cursor += d;
                continue;
            }
            // Constant-curvature κ is seeded from the spec (the user's curvature
            // hint, default κ = 0); `clamp_to_bounds` projects it feasible. It
            // is not a log-scale, so the log-κ recenter below never applies.
            if constant_curvature_term_spec(spec, term_idx).is_some() {
                cursor += d;
                continue;
            }
            let Some(psi_bar_new) = spatial_term_psi_seed(data, spec, term_idx, options) else {
                cursor += d;
                continue;
            };
            if d == 0 {
                continue;
            }
            let current: Vec<f64> = self.values.slice(s![cursor..cursor + d]).to_vec();
            let psi_bar_old = current.iter().sum::<f64>() / d as f64;
            for (offset, &old_value) in current.iter().enumerate() {
                self.values[cursor + offset] = psi_bar_new + (old_value - psi_bar_old);
            }
            cursor += d;
        }
        self
    }

    /// Project ψ values into `[lower, upper]` element-wise. Used after
    /// `from_length_scales*` + `reseed_from_data` when a user-supplied
    /// `spec.length_scale` falls outside the data-derived ψ window set by
    /// `{lower,upper}_bounds*_from_data`. BFGS requires theta0 ∈ [lower,
    /// upper]; projecting is the unique closest feasible seed. The user's
    /// length_scale was always a hint for the outer optimizer (the optimizer
    /// is authoritative for κ), not a hard constraint — so clipping preserves
    /// their intent as far as the geometry allows. Emits `log::info!` when
    /// any coordinate moves, so the outside-window case is diagnostically
    /// visible (not silent).
    pub(crate) fn clamp_to_bounds(
        mut self,
        lower: &SpatialLogKappaCoords,
        upper: &SpatialLogKappaCoords,
    ) -> Self {
        assert_eq!(self.values.len(), lower.values.len());
        assert_eq!(self.values.len(), upper.values.len());
        let mut n_projected = 0usize;
        let mut worst_delta = 0.0_f64;
        for idx in 0..self.values.len() {
            let lo = lower.values[idx];
            let hi = upper.values[idx];
            if !(lo.is_finite() && hi.is_finite()) {
                continue;
            }
            let v = self.values[idx];
            if v < lo {
                worst_delta = worst_delta.max(lo - v);
                self.values[idx] = lo;
                n_projected += 1;
            } else if v > hi {
                worst_delta = worst_delta.max(v - hi);
                self.values[idx] = hi;
                n_projected += 1;
            }
        }
        if n_projected > 0 {
            log::info!(
                "[spatial-kappa] projected {n_projected}/{} ψ seed coords into data-derived bounds \
                 (worst excess={worst_delta:.3} log units); user length_scale falls outside \
                 [{KERNEL_RANGE_MIN_DIAMETER_FRACTION}/r_max, {KERNEL_RANGE_MAX_SPACING_MULTIPLE}/r_min] geometry window",
                self.values.len()
            );
        }
        self
    }

    /// Reconstruct from theta tail with known dimensionality layout.
    pub(crate) fn from_theta_tail_with_dims(
        theta: &Array1<f64>,
        start: usize,
        dims_per_term: Vec<usize>,
    ) -> Self {
        let total: usize = dims_per_term.iter().sum();
        Self {
            values: theta.slice(s![start..start + total]).to_owned(),
            dims_per_term,
        }
    }

    /// Total number of ψ values in the flat array (= sum of dims_per_term).
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// Dimensionality layout: how many ψ values each term contributes.
    pub(crate) fn dims_per_term(&self) -> &[usize] {
        &self.dims_per_term
    }

    /// Get the offset into the flat array for logical term i.
    fn term_offset(&self, term_idx: usize) -> usize {
        self.dims_per_term[..term_idx].iter().sum()
    }

    /// Get the slice of ψ values for logical term i.
    fn term_slice(&self, term_idx: usize) -> &[f64] {
        let offset = self.term_offset(term_idx);
        let d = self.dims_per_term[term_idx];
        &self.values.as_slice().unwrap()[offset..offset + d]
    }

    pub(crate) fn as_array(&self) -> &Array1<f64> {
        &self.values
    }

    /// Split at a logical-term boundary. `mid` is the number of terms in the
    /// first half (not a flat-array index).
    pub(crate) fn split_at(&self, mid: usize) -> (Self, Self) {
        let flat_mid: usize = self.dims_per_term[..mid].iter().sum();
        (
            Self {
                values: self.values.slice(s![0..flat_mid]).to_owned(),
                dims_per_term: self.dims_per_term[..mid].to_vec(),
            },
            Self {
                values: self.values.slice(s![flat_mid..]).to_owned(),
                dims_per_term: self.dims_per_term[mid..].to_vec(),
            },
        )
    }

    /// Apply optimized ψ values back to the spec.
    ///
    /// For isotropic terms (dims=1): sets scalar length_scale = exp(−ψ).
    /// For anisotropic terms (dims=d): hybrid/isotropic families set
    /// length_scale = exp(−ψ̄) with centered η_a = ψ_a − ψ̄, while pure Duchon
    /// writes only centered η_a and leaves length_scale = None.
    pub(crate) fn apply_tospec(
        &self,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
    ) -> Result<TermCollectionSpec, EstimationError> {
        if term_indices.len() != self.dims_per_term.len() {
            crate::bail_invalid_estim!(
                "SpatialLogKappaCoords::apply_tospec: term count mismatch: \
                 term_indices={} dims_per_term={}",
                term_indices.len(),
                self.dims_per_term.len()
            );
        }
        let mut updated = spec.clone();
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            let psi = self.term_slice(slot);
            let d = self.dims_per_term[slot];
            // Measure-jet: write the dial coordinates straight back; the
            // κ-translation below would misread them as log-scales.
            if measure_jet_term_spec(&updated, term_idx).is_some() {
                set_measure_jet_psi_dials(&mut updated, term_idx, psi)?;
                continue;
            }
            // Constant-curvature: write the optimized signed κ straight back;
            // the −exp(ψ) length-scale translation below is log-κ semantics and
            // would misread the raw curvature.
            if constant_curvature_term_spec(&updated, term_idx).is_some() {
                set_constant_curvature_kappa(&mut updated, term_idx, psi)?;
                continue;
            }
            let (next_length_scale, next_aniso) = spatial_term_psi_to_length_scale_and_aniso(psi);
            if (d == 1 || next_length_scale.is_some())
                && let Some(length_scale) = next_length_scale
            {
                set_spatial_length_scale(&mut updated, term_idx, length_scale)?;
            }
            if let Some(eta) = next_aniso {
                set_spatial_aniso_log_scales(&mut updated, term_idx, eta)?;
            }
        }
        Ok(updated)
    }
}


pub(crate) fn center_aniso_log_scales(eta: &[f64]) -> Vec<f64> {
    if eta.len() <= 1 {
        return eta.to_vec();
    }
    let mean = eta.iter().sum::<f64>() / eta.len() as f64;
    eta.iter()
        .map(|&v| {
            let centered = v - mean;
            if centered.abs() <= 1e-15 {
                0.0
            } else {
                centered
            }
        })
        .collect()
}


pub(crate) fn spatial_term_supports_hyper_optimization(spec: &TermCollectionSpec, term_idx: usize) -> bool {
    // Ordinary penalized thin-plate regression splines do not have an
    // identifiable kernel scale once REML is already learning the smoothing
    // penalty. Treat the resolved length scale as fixed geometry; enrolling a
    // scalar TPS kappa axis creates the flat ρ/κ valleys reported in #718,
    // #721, #731, and #732.
    if let Some(term) = spec.smooth_terms.get(term_idx)
        && let SmoothBasisSpec::ThinPlate { .. } = &term.basis
    {
        return false;
    }

    // Duchon anisotropy η is a FIXED, geometry-derived basis parameter, NOT a
    // REML hyper axis: the metric is estimated once from the knot-cloud spread
    // (`auto_seed_aniso_contrasts`, applied on every Duchon basis build) and the
    // Hilbert-scale λ's carry all learned smoothness. So a pure Duchon (no κ)
    // contributes no outer optimization axis even when `scale_dims` is on —
    // "standardize the geometry, then learn the smoothness." Only an explicit
    // kernel length scale κ (the Matérn / hybrid path) is optimized here.
    //
    // ISOTROPIC Matérn: the *default* `matern(x1, x2)` is isotropic
    // (`scale_dims=false` → `aniso_log_scales = None`). It contributes exactly
    // ONE κ optimization axis — its scalar log-κ. The shared GAMLSS /
    // location-scale exact-joint ψ engine and the spatial-κ joint outer solver
    // both require an isotropic Matérn block to expose this single isotropic κ
    // axis (#822/#851); without it the per-block ψ-derivative lists are empty
    // and the joint-ψ hooks degenerate to `None`. The isotropic κ is the lone
    // kernel hyper axis here, mirroring the per-axis ψ ARD that the anisotropic
    // path exposes (just collapsed to one dimension).
    //
    // ANISOTROPIC Matérn (`scale_dims=true` → `aniso_log_scales = Some`) keeps
    // its per-axis kernel-η ARD: the d-dimensional ψ search is the *point* of
    // the anisotropic request ("Matérn keeps its kernel-η ARD").
    //
    // Either way a Matérn term always enrolls a κ/ψ axis (1 isotropic, or d
    // anisotropic), so `spatial_dims_per_term` reports the correct count.
    if let Some(term) = spec.smooth_terms.get(term_idx)
        && let SmoothBasisSpec::Matern { .. } = &term.basis
    {
        return true;
    }

    // Measure-jet geometry dials are outer ψ coordinates; enrollment is
    // owned by `measure_jet_enrolls_psi`.
    if let Some(mj) = measure_jet_term_spec(spec, term_idx) {
        return measure_jet_enrolls_psi(mj);
    }

    // Constant-curvature smooths always enroll their single signed curvature κ
    // as an outer ψ-coordinate (#944 stage 3): κ̂ is the headline estimand, so
    // unlike a fixed-ℓ kernel it is fitted by default, not gated on a
    // user-supplied scale. The coordinate is raw κ (interior κ = 0), and its
    // exact design/penalty κ-derivatives come from
    // `build_constant_curvature_basis_kappa_derivatives`.
    if constant_curvature_term_spec(spec, term_idx).is_some() {
        return true;
    }

    get_spatial_length_scale(spec, term_idx).is_some()
}


/// The measure-jet term's spec, when `term_idx` is a measure-jet smooth.
/// Single accessor for every dial-plumbing dispatch below.
pub(crate) fn measure_jet_term_spec(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<&crate::basis::MeasureJetBasisSpec> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec, .. } => Some(spec),
            _ => None,
        })
}


/// Single source for measure-jet outer-ψ enrollment: the lnτ dial is
/// undefined in the τ = 0 pseudo-inverse oracle mode (see
/// `build_measure_jet_basis_psi_derivatives`), so only a positive ridge
/// enrolls the dial group. `spatial_term_supports_hyper_optimization` and
/// `spatial_term_uses_per_axis_psi` both defer here so the θ-layout
/// sources cannot disagree.
pub(crate) fn measure_jet_enrolls_psi(mj: &crate::basis::MeasureJetBasisSpec) -> bool {
    // ψ dials ride multiscale mode only: the per-scale spectral split and the
    // (α, lnτ) dials are enabled together by the explicit `multiscale` opt-in
    // (single source: `measure_jet_multiscale_mode`, #1116). Single-scale-mode
    // terms (the default at any center count) stay at one fused penalty with
    // fixed dials — Duchon/Matérn's outer footprint — so they never inflate the
    // family's O(n) per-row evaluation count (#1039). The lnτ channel
    // additionally needs a positive ridge.
    mj.tau0 > 0.0 && crate::basis::measure_jet_multiscale_mode(mj)
}

/// Measure-jet ψ dial boxes. The dials are NOT log-kernel-scales, so the
/// κ-window machinery never applies: `s` stays inside the admissible order
/// interval of the affine-jet energy, `α` spans density-weighted (0) through
/// past-Coifman–Lafon (>1) normalization, and `lnτ` covers the ridge from
/// numerically-exact-projection to heavy noise-floor damping.
const MEASURE_JET_PSI_S_BOUNDS: (f64, f64) = (0.05, 1.95);

const MEASURE_JET_PSI_ALPHA_BOUNDS: (f64, f64) = (-1.0, 3.0);

const MEASURE_JET_PSI_LN_TAU_BOUNDS: (f64, f64) = (-18.420680743952367, 4.605170185988092);


/// Is this measure-jet term in fused (pinned-order) mode? The `order_s`
/// sentinel is the spectral/fused mode marker (see the basis module docs).
/// Only consulted for terms that enroll ψ (multiscale mode), where `order_s == 0`.
fn measure_jet_is_fused(mj: &crate::basis::MeasureJetBasisSpec) -> bool {
    mj.order_s > 0.0
}


/// ψ dimension of a measure-jet term: fused mode carries (s, α, lnτ);
/// per-level (spectral) mode carries (α, lnτ) — the order is absorbed by the
/// REML-learned per-scale amplitudes. MUST agree with the coordinate layout
/// of `build_measure_jet_basis_psi_derivatives`.
pub(crate) fn measure_jet_psi_dim(mj: &crate::basis::MeasureJetBasisSpec) -> usize {
    if measure_jet_is_fused(mj) { 3 } else { 2 }
}


/// Seed ψ from the term's realized dials, in producer coordinate order.
fn measure_jet_psi_seed(mj: &crate::basis::MeasureJetBasisSpec) -> Vec<f64> {
    let ln_tau = mj.tau0.max(f64::MIN_POSITIVE).ln();
    if measure_jet_is_fused(mj) {
        vec![mj.order_s, mj.alpha, ln_tau]
    } else {
        vec![mj.alpha, ln_tau]
    }
}


/// One end of the per-coordinate dial boxes, in producer coordinate order.
fn measure_jet_psi_bound_values(mj: &crate::basis::MeasureJetBasisSpec, upper: bool) -> Vec<f64> {
    let pick = |b: (f64, f64)| if upper { b.1 } else { b.0 };
    if measure_jet_is_fused(mj) {
        vec![
            pick(MEASURE_JET_PSI_S_BOUNDS),
            pick(MEASURE_JET_PSI_ALPHA_BOUNDS),
            pick(MEASURE_JET_PSI_LN_TAU_BOUNDS),
        ]
    } else {
        vec![
            pick(MEASURE_JET_PSI_ALPHA_BOUNDS),
            pick(MEASURE_JET_PSI_LN_TAU_BOUNDS),
        ]
    }
}


/// Write optimized ψ dials back into a measure-jet spec. Returns `true` when
/// any dial actually moved. The geometry (centers, masses, band, ℓ, z) is
/// ψ-FIXED by contract — only the dials change, so frozen-quadrature
/// rebuilds reproduce the identical penalty layout at the new dials.
fn apply_measure_jet_psi(
    mj: &mut crate::basis::MeasureJetBasisSpec,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    if psi.len() != measure_jet_psi_dim(mj) {
        crate::bail_invalid_estim!(
            "measure-jet ψ write-back dimension mismatch: got {} values for a {}-dial term",
            psi.len(),
            measure_jet_psi_dim(mj)
        );
    }
    let (next_s, next_alpha, next_tau) = if measure_jet_is_fused(mj) {
        (Some(psi[0]), psi[1], psi[2].exp())
    } else {
        (None, psi[0], psi[1].exp())
    };
    if !(next_alpha.is_finite() && next_tau.is_finite() && next_tau > 0.0) {
        crate::bail_invalid_estim!(
            "measure-jet ψ write-back produced non-finite dials (alpha={next_alpha}, tau={next_tau})"
        );
    }
    let mut changed = false;
    if let Some(s) = next_s
        && s != mj.order_s
    {
        mj.order_s = s;
        changed = true;
    }
    if next_alpha != mj.alpha {
        mj.alpha = next_alpha;
        changed = true;
    }
    if next_tau != mj.tau0 {
        mj.tau0 = next_tau;
        changed = true;
    }
    Ok(changed)
}


/// Collection-level measure-jet dial write-back (the `apply_tospec` /
/// realizer-side entry). Returns whether anything moved.
pub(crate) fn set_measure_jet_psi_dials(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!("measure-jet ψ write-back: term index {term_idx} out of range");
    };
    set_single_term_measure_jet_psi_dials(term, psi)
}


/// Single-term dial write-back: the shared match+apply core, also used
/// directly on the cached per-trial build spec (whose caller has already
/// change-checked at the collection level and rebuilds regardless of the
/// moved flag).
fn set_single_term_measure_jet_psi_dials(
    term: &mut SmoothTermSpec,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &mut term.basis else {
        crate::bail_invalid_estim!("measure-jet ψ write-back targeted a non-measure-jet term");
    };
    apply_measure_jet_psi(mj, psi)
}


/// The constant-curvature smooth's spec, when `term_idx` is one. Single
/// accessor for every κ-ψ dispatch below, mirroring `measure_jet_term_spec`.
pub(crate) fn constant_curvature_term_spec(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<&crate::basis::ConstantCurvatureBasisSpec> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::ConstantCurvature { spec, .. } => Some(spec),
            _ => None,
        })
}


/// Hard positive cap on |κ| relative to the data's inverse squared chart
/// radius. The κ-stereographic chart is valid for `1 + κ‖x‖² > 0`; at
/// `|κ| = 1/R²` (R² = max squared chart radius) the gauge `1 + κ‖x‖²` reaches
/// the chart edge for the farthest data point, so the optimizer is boxed to a
/// safe fraction of that scale on both sides. κ = 0 (flat) is the centre of
/// the window, an interior point of the `S^d ← ℝ^d → H^d` family — exactly the
/// reachability the raw-κ (not log-κ) coordinate exists to preserve.
const CONSTANT_CURVATURE_KAPPA_CHART_FRACTION: f64 = 0.5;

/// Floor on the data's squared chart radius used to scale the κ window, so a
/// degenerate (near-origin) point cloud still yields a finite, usable bracket
/// rather than an unbounded one.
const CONSTANT_CURVATURE_MIN_CHART_RADIUS2: f64 = 1e-8;


/// `(κ_min, κ_max)` outer-optimization window for a constant-curvature term,
/// derived from the data's maximum squared chart radius `R²` so the κ-jets
/// never leave the κ-stereographic chart. Symmetric about κ = 0:
/// `±CONSTANT_CURVATURE_KAPPA_CHART_FRACTION / R²`.
pub(crate) fn constant_curvature_kappa_bounds(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> (f64, f64) {
    let feature_cols = match spec.smooth_terms.get(term_idx).map(|t| &t.basis) {
        Some(SmoothBasisSpec::ConstantCurvature { feature_cols, .. }) => feature_cols,
        _ => return (-1.0, 1.0),
    };
    let mut max_r2 = CONSTANT_CURVATURE_MIN_CHART_RADIUS2;
    for row in data.outer_iter() {
        let mut r2 = 0.0_f64;
        for &c in feature_cols.iter() {
            if let Some(&v) = row.get(c)
                && v.is_finite()
            {
                r2 += v * v;
            }
        }
        if r2 > max_r2 {
            max_r2 = r2;
        }
    }
    let half = CONSTANT_CURVATURE_KAPPA_CHART_FRACTION / max_r2;
    (-half, half)
}


/// Write the optimized κ back into a constant-curvature term spec. Returns
/// `true` when κ moved. Centers, ℓ, and the constraint transform `z` are
/// κ-FIXED by the basis κ-contract, so only `kappa` changes.
pub(crate) fn set_constant_curvature_kappa(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!(
            "constant-curvature κ write-back: term index {term_idx} out of range"
        );
    };
    set_single_term_constant_curvature_kappa(term, psi)
}


/// Single-term κ write-back: the shared validate+apply core, also used directly
/// on the cached per-trial build spec in the incremental realizer (whose caller
/// has already change-checked at the collection level and rebuilds regardless
/// of the moved flag). Mirrors [`set_single_term_measure_jet_psi_dials`].
fn set_single_term_constant_curvature_kappa(
    term: &mut SmoothTermSpec,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    if psi.len() != 1 {
        crate::bail_invalid_estim!(
            "constant-curvature κ write-back expects exactly one value, got {}",
            psi.len()
        );
    }
    let next_kappa = psi[0];
    if !next_kappa.is_finite() {
        crate::bail_invalid_estim!(
            "constant-curvature κ write-back produced a non-finite κ = {next_kappa}"
        );
    }
    let SmoothBasisSpec::ConstantCurvature { spec: cc, .. } = &mut term.basis else {
        crate::bail_invalid_estim!(
            "constant-curvature κ write-back targeted a non-constant-curvature term"
        );
    };
    if cc.kappa != next_kappa {
        cc.kappa = next_kappa;
        Ok(true)
    } else {
        Ok(false)
    }
}


/// Returns `true` when a spatial term has NO outer optimization axes — i.e.
/// the user provided an explicit `length_scale` and the term does not enroll
/// REML-side per-axis ψ contrasts, so both the scalar κ and any fixed geometry
/// anisotropy are anchored.
///
/// This is the per-term predicate that distinguishes "fixed kernel scale"
/// from "optimize the kernel scale" within the family entry points that
/// want to honor an explicit user-supplied scale (e.g. Bernoulli
/// marginal-slope, where the joint-spatial outer solver otherwise spends
/// ~80 iters stalled on the user's chosen ρ at high gradient).
pub fn spatial_term_has_locked_kappa(spec: &TermCollectionSpec, term_idx: usize) -> bool {
    get_spatial_length_scale(spec, term_idx).is_some()
        && !spatial_term_uses_per_axis_psi(spec, term_idx)
}


/// Per-term data-derived ψ = log κ bounds.
///
/// Uses the same safe operating range documented in
/// [`crate::basis::build_matern_basis`] / [`crate::basis::build_duchon_basis`]:
///   κ ∈ [2 / r_max, 1e2 / r_min]
/// where (r_min, r_max) are pairwise-distance extrema of the term's resolved
/// centers (post-fit) or the standardized feature data columns (pre-fit).
/// Lower edge of the data-derived kernel-range window, as a fraction of the
/// maximum pairwise distance `r_max`: length scales below `2/r_max` resolve
/// structure finer than the closest center pair, so the kernel range floor is
/// set at twice the maximum spacing.
const KERNEL_RANGE_MIN_DIAMETER_FRACTION: f64 = 2.0;

/// Upper edge of the data-derived kernel-range window, as a multiple of the
/// minimum pairwise distance `r_min`: beyond `100/r_min` the radial columns go
/// nearly collinear with the polynomial nullspace, so the kernel range is
/// capped here to keep the basis geometry well-conditioned.
const KERNEL_RANGE_MAX_SPACING_MULTIPLE: f64 = 1e2;


/// Returns ψ-space bounds (ψ_lo = ln(κ_lo), ψ_hi = ln(κ_hi)).
///
/// When geometry is unavailable (e.g., fewer than 2 distinct points), falls
/// back to the scalar `options.min_length_scale` / `options.max_length_scale`
/// window so the outer optimizer never sees NaN bounds.
///
/// The returned window is intersected with the options window so user-set
/// `min_length_scale` / `max_length_scale` remain hard limits.
fn spatial_term_psi_bounds(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    term_idx: usize,
    options: &SpatialLengthScaleOptimizationOptions,
) -> (f64, f64) {
    let fallback = (
        -options.max_length_scale.ln(),
        -options.min_length_scale.ln(),
    );
    // Constant-curvature: the ψ coordinate is the raw signed κ, so its window is
    // the chart-feasible κ bracket, NOT a log-ℓ window. Mirrors the aniso bounds
    // path's `constant_curvature_kappa_bounds` branch so the isotropic
    // (non-aniso) seed clamp projects κ into the right interval.
    if constant_curvature_term_spec(spec, term_idx).is_some() {
        return constant_curvature_kappa_bounds(data, spec, term_idx);
    }
    let Some(term) = spec.smooth_terms.get(term_idx) else {
        return fallback;
    };
    // Prefer resolved centers (post-fit) since they live in the same standardized
    // space the kernel actually sees. Centers are capped at `default_num_centers`
    // (<=2000), so exact pairwise bounds are cheap (<4M ops). If centers are
    // not yet UserProvided, fall back to the standardized feature data columns
    // with the capped-sample path (O(K²·d), K=1024) — the sample is
    // conservative for κ bounds (see `pairwise_distance_bounds_sampled`
    // docs): it never excludes a feasible κ the exact method would include.
    //
    // Under anisotropy the kernel metric is y-space (y_a = exp(η_a) x_a),
    // so r_min/r_max must be y-space distances. This matters only when the
    // spec already carries calibrated η_a at setup time (e.g., warm-start
    // or refit paths); for fresh optimization η_a starts at 0 and y = x.
    let aniso = get_spatial_aniso_log_scales(spec, term_idx);
    let r_bounds = match spatial_term_center_strategy(term) {
        Some(CenterStrategy::UserProvided(centers)) if centers.nrows() >= 2 => {
            match aniso.as_deref() {
                Some(eta) if eta.len() == centers.ncols() => {
                    let y = points_in_aniso_y_space(centers.view(), eta);
                    pairwise_distance_bounds(y.view())
                }
                _ => pairwise_distance_bounds(centers.view()),
            }
        }
        _ => standardized_spatial_term_data(data, term)
            .ok()
            .and_then(|x| match aniso.as_deref() {
                Some(eta) if eta.len() == x.ncols() => {
                    let y = points_in_aniso_y_space(x.view(), eta);
                    pairwise_distance_bounds_sampled(y.view())
                }
                _ => pairwise_distance_bounds_sampled(x.view()),
            }),
    };
    let Some((r_min, r_max)) = r_bounds else {
        return fallback;
    };
    // Length scales substantially larger than the data diameter make radial
    // TPS/Matern columns nearly collinear with their polynomial nullspace.
    // The nullspace already carries constant/linear low-frequency structure,
    // so cap the kernel range at the diameter scale instead of letting the
    // optimizer enter a numerically degenerate basis geometry.
    let psi_lo_data = (KERNEL_RANGE_MIN_DIAMETER_FRACTION / r_max).ln();
    let psi_hi_data = (KERNEL_RANGE_MAX_SPACING_MULTIPLE / r_min).ln();
    // Intersect with the options window so min/max_length_scale remain hard caps.
    let psi_lo = psi_lo_data.max(fallback.0);
    let psi_hi = psi_hi_data.min(fallback.1);
    if psi_lo >= psi_hi {
        // Degenerate intersection — fall back to the options window to keep the
        // outer optimizer from collapsing to a point.
        return fallback;
    }
    (psi_lo, psi_hi)
}


/// Data-derived ψ seed for a spatial term when the user has not set an
/// explicit length_scale on its basis spec. Uses the geometric mean of the
/// data-informed kappa range (i.e., the midpoint of the ψ window).
fn spatial_term_psi_seed(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    term_idx: usize,
    options: &SpatialLengthScaleOptimizationOptions,
) -> Option<f64> {
    if get_spatial_length_scale(spec, term_idx).is_some() {
        return None; // user/spec-provided length_scale wins
    }
    let (psi_lo, psi_hi) = spatial_term_psi_bounds(data, spec, term_idx, options);
    Some(0.5 * (psi_lo + psi_hi))
}


pub(crate) fn spatial_term_psi_to_length_scale_and_aniso(psi: &[f64]) -> (Option<f64>, Option<Vec<f64>>) {
    if psi.len() <= 1 {
        (Some((-psi.first().copied().unwrap_or(0.0)).exp()), None)
    } else {
        let psi_bar = psi.iter().sum::<f64>() / psi.len() as f64;
        (
            Some((-psi_bar).exp()),
            Some(psi.iter().map(|&value| value - psi_bar).collect()),
        )
    }
}


/// Get the `aniso_log_scales` from a spatial term, if present.
pub fn get_spatial_aniso_log_scales(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<Vec<f64>> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.aniso_log_scales.clone(),
            SmoothBasisSpec::Duchon { spec, .. } => spec.aniso_log_scales.clone(),
            _ => None,
        })
}


/// Get the number of feature columns (spatial dimensionality) for a spatial term.
pub(crate) fn get_spatial_feature_dim(spec: &TermCollectionSpec, term_idx: usize) -> Option<usize> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::ThinPlate { feature_cols, .. } => Some(feature_cols.len()),
            SmoothBasisSpec::Matern { feature_cols, .. } => Some(feature_cols.len()),
            SmoothBasisSpec::Duchon { feature_cols, .. } => Some(feature_cols.len()),
            _ => None,
        })
}


/// Log the learned per-axis spatial anisotropy for all spatial terms that
/// have `aniso_log_scales` set after optimization.
///
/// For scalar-scale families this reports eta, effective per-axis length
/// scales, and per-axis kappa values. For pure Duchon it reports the centered
/// eta contrasts only.
pub fn log_spatial_aniso_scales(spec: &TermCollectionSpec) {
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let (aniso, length_scale) = match &term.basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                (spec.aniso_log_scales.as_ref(), Some(spec.length_scale))
            }
            SmoothBasisSpec::Duchon { spec, .. } => {
                (spec.aniso_log_scales.as_ref(), spec.length_scale)
            }
            _ => (None, None),
        };
        let Some(eta) = aniso else { continue };
        if eta.is_empty() {
            continue;
        }
        let mut lines = match length_scale {
            Some(ls) => format!(
                "[spatial-kappa] term {} (\"{}\"): anisotropic length scales optimized (global length_scale={:.4})",
                term_idx, term.name, ls
            ),
            None => format!(
                "[spatial-kappa] term {} (\"{}\"): pure Duchon shape anisotropy optimized",
                term_idx, term.name
            ),
        };
        for (a, &eta_a) in eta.iter().enumerate() {
            if let Some(ls) = length_scale {
                let length_a = ls * (-eta_a).exp();
                let kappa_a = (1.0 / ls) * eta_a.exp();
                lines.push_str(&format!(
                    "\n  axis {}: eta={:+.4}, length={:.4}, kappa={:.4}",
                    a, eta_a, length_a, kappa_a
                ));
            } else {
                lines.push_str(&format!("\n  axis {}: eta={:+.4}", a, eta_a));
            }
        }
        log::info!("{}", lines);
    }
}


/// Set `aniso_log_scales` on a spatial term's basis spec.
pub(crate) fn set_spatial_aniso_log_scales(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    eta: Vec<f64>,
) -> Result<(), EstimationError> {
    let eta = center_aniso_log_scales(&eta);
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!("spatial aniso_log_scales term index {term_idx} out of range");
    };
    match &mut term.basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not support aniso_log_scales",
            term.name
        ))),
    }
}


/// Sync knot-cloud-derived anisotropy contrasts from basis metadata back into
/// the mutable spec so the optimizer starts from the correct eta values.
///
/// Call this after building the smooth design but before initializing the
/// optimizer's psi coordinates. For each spatial term whose metadata contains
/// computed `aniso_log_scales`, this writes them into the spec.
pub(crate) fn sync_aniso_contrasts_from_metadata(
    spec: &mut TermCollectionSpec,
    design: &SmoothDesign,
) {
    for (term_idx, term) in design.terms.iter().enumerate() {
        let meta_aniso = match &term.metadata {
            BasisMetadata::Matern {
                aniso_log_scales, ..
            } => aniso_log_scales.clone(),
            BasisMetadata::Duchon {
                aniso_log_scales, ..
            } => aniso_log_scales.clone(),
            _ => None,
        };
        if let Some(eta) = meta_aniso
            && eta.len() > 1
        {
            set_spatial_aniso_log_scales(spec, term_idx, eta).ok();
        }
    }
}


#[derive(Debug, Clone)]
pub struct SpatialLengthScaleOptimizationOptions {
    /// Enable outer-loop optimization over spatial κ (= 1 / length_scale)
    /// for supported radial-kernel smooths.
    /// This applies to ThinPlate, Matérn, and Duchon terms.
    pub enabled: bool,
    /// Maximum number of outer iterations in the exact joint [rho, psi] solve.
    pub max_outer_iter: usize,
    /// Relative improvement threshold for terminating the outer solve.
    pub rel_tol: f64,
    /// Initial log(length_scale) perturbation used for seed construction.
    pub log_step: f64,
    /// Minimum allowed length_scale during κ search.
    pub min_length_scale: f64,
    /// Maximum allowed length_scale during κ search.
    pub max_length_scale: f64,
    /// Automatic geometry-initializer threshold for large-scale spatial fits.
    ///
    /// When n exceeds twice this value, the fitter uses a spatially stratified
    /// subsample only to seed κ/anisotropy geometry: centers are resolved,
    /// axis contrasts are initialized from center/data spread, and one or two
    /// cheap ψ reseeding updates are applied. It never runs PIRLS, REML, ARC,
    /// BFGS, or any recursive optimizer on the pilot.
    ///
    /// The final coefficients, smoothing parameters, and spatial geometry are
    /// always optimized on the full dataset.
    ///
    /// Set to 0 to skip the pilot geometry initializer.
    pub pilot_subsample_threshold: usize,
}


impl Default for SpatialLengthScaleOptimizationOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_outer_iter: 80,
            rel_tol: 1e-4,
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-3,
            max_length_scale: 1e3,
            pilot_subsample_threshold: 10_000,
        }
    }
}


impl SpatialLengthScaleOptimizationOptions {
    /// Validate the struct's invariants. Callers that construct these options
    /// from external input (CLI, config, Python API) should call this before
    /// passing the options into the fitter. Returns `Err` with a descriptive
    /// message when an invariant is violated; the fitter then panics or
    /// returns `EstimationError` at its own boundary.
    ///
    /// Invariants:
    ///   * `min_length_scale > 0`, finite
    ///   * `max_length_scale > 0`, finite
    ///   * `min_length_scale < max_length_scale`
    ///   * `rel_tol > 0`, finite
    ///   * `log_step > 0`, finite
    ///
    /// These invariants are what the downstream κ-bound and ψ-window code
    /// assumes (`-log(max_ls)` must be finite, `(min,max)` must not be
    /// inverted, etc.). Without validation, invalid options produce silent
    /// NaN-propagation inside the outer optimizer.
    pub fn validate(&self) -> Result<(), String> {
        if !self.min_length_scale.is_finite() || self.min_length_scale <= 0.0 {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions::min_length_scale must be > 0 and finite, got {}",
                self.min_length_scale
            ))
            .into());
        }
        if !self.max_length_scale.is_finite() || self.max_length_scale <= 0.0 {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions::max_length_scale must be > 0 and finite, got {}",
                self.max_length_scale
            ))
            .into());
        }
        if self.min_length_scale >= self.max_length_scale {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions requires min_length_scale < max_length_scale, got min={} max={}",
                self.min_length_scale, self.max_length_scale
            ))
            .into());
        }
        if !self.rel_tol.is_finite() || self.rel_tol <= 0.0 {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions::rel_tol must be > 0 and finite, got {}",
                self.rel_tol
            ))
            .into());
        }
        if !self.log_step.is_finite() || self.log_step <= 0.0 {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions::log_step must be > 0 and finite, got {}",
                self.log_step
            ))
            .into());
        }
        Ok(())
    }
}


#[derive(Debug, Clone)]
fn spatial_term_user_centers(term: &SmoothTermSpec) -> Option<ArrayView2<'_, f64>> {
    match spatial_term_center_strategy(term) {
        Some(CenterStrategy::UserProvided(centers)) => Some(centers.view()),
        _ => None,
    }
}


fn finite_centered_axis_contrasts(values: &[f64], expected_dim: usize) -> Option<Vec<f64>> {
    if values.len() != expected_dim || expected_dim <= 1 {
        return None;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(center_aniso_log_scales(values))
}


fn blended_pilot_axis_contrasts(
    pilot_data: ArrayView2<'_, f64>,
    term: &SmoothTermSpec,
    centers: ArrayView2<'_, f64>,
) -> Option<Vec<f64>> {
    let d = centers.ncols();
    if d <= 1 {
        return None;
    }
    let center_eta = initial_aniso_contrasts(centers);
    let data_eta = standardized_spatial_term_data(pilot_data, term)
        .ok()
        .and_then(|x| finite_centered_axis_contrasts(&initial_aniso_contrasts(x.view()), d));
    let center_eta = finite_centered_axis_contrasts(&center_eta, d)?;
    let blended = match data_eta {
        Some(data_eta) => center_eta
            .iter()
            .zip(data_eta.iter())
            .map(|(&from_centers, &from_data)| 0.5 * (from_centers + from_data))
            .collect::<Vec<_>>(),
        None => center_eta,
    };
    finite_centered_axis_contrasts(&blended, d)
}


fn apply_pilot_spatial_psi_reseed(
    pilot_data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    spatial_terms: &[usize],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<TermCollectionSpec, EstimationError> {
    let dims_per_term = spatial_dims_per_term(spec, spatial_terms);
    let use_aniso = has_aniso_terms(spec, spatial_terms);
    let log_kappa0 = if use_aniso {
        SpatialLogKappaCoords::from_length_scales_aniso(spec, spatial_terms, kappa_options)
    } else {
        SpatialLogKappaCoords::from_length_scales(spec, spatial_terms, kappa_options)
    };
    let log_kappa0 = log_kappa0.reseed_from_data(pilot_data, spec, spatial_terms, kappa_options);
    let log_kappa_lower = if use_aniso {
        SpatialLogKappaCoords::lower_bounds_aniso_from_data(
            pilot_data,
            spec,
            spatial_terms,
            &dims_per_term,
            kappa_options,
        )
    } else {
        SpatialLogKappaCoords::lower_bounds_from_data(
            pilot_data,
            spec,
            spatial_terms,
            kappa_options,
        )
    };
    let log_kappa_upper = if use_aniso {
        SpatialLogKappaCoords::upper_bounds_aniso_from_data(
            pilot_data,
            spec,
            spatial_terms,
            &dims_per_term,
            kappa_options,
        )
    } else {
        SpatialLogKappaCoords::upper_bounds_from_data(
            pilot_data,
            spec,
            spatial_terms,
            kappa_options,
        )
    };
    log_kappa0
        .clamp_to_bounds(&log_kappa_lower, &log_kappa_upper)
        .apply_tospec(spec, spatial_terms)
}


pub(crate) fn apply_spatial_anisotropy_pilot_initializer(
    data: ArrayView2<'_, f64>,
    spec: &mut TermCollectionSpec,
    spatial_terms: &[usize],
    target_size: usize,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> usize {
    if target_size == 0 || data.nrows() <= target_size.saturating_mul(2) || spatial_terms.is_empty()
    {
        return 0;
    }
    if !has_aniso_terms(spec, spatial_terms) {
        return 0;
    }
    let indices = stratified_spatial_subsample(data, spec, target_size);
    let pilot_data = sampled_rows(data, &indices);
    let mut working = spec.clone();
    let mut updated_terms = 0usize;
    const GEOMETRY_UPDATES: usize = 2;

    for pass in 0..GEOMETRY_UPDATES {
        let planned_terms = match plan_joint_spatial_centers_for_term_blocks(
            pilot_data.view(),
            &[working.smooth_terms.clone()],
        )
        .and_then(|mut blocks| {
            blocks.pop().ok_or_else(|| {
                BasisError::InvalidInput(
                    "pilot geometry initializer produced no smooth-term block".to_string(),
                )
            })
        }) {
            Ok(terms) => terms,
            Err(err) => {
                log::warn!(
                    "[spatial-kappa] pilot geometry initializer skipped after center planning failed: {err}"
                );
                return updated_terms;
            }
        };

        for &term_idx in spatial_terms {
            let Some(current_eta) = get_spatial_aniso_log_scales(&working, term_idx) else {
                continue;
            };
            let Some(d) = get_spatial_feature_dim(&working, term_idx) else {
                continue;
            };
            if d <= 1 || current_eta.len() != d {
                continue;
            }
            let Some(planned_term) = planned_terms.get(term_idx) else {
                continue;
            };
            let Some(centers) = spatial_term_user_centers(planned_term) else {
                continue;
            };
            let Some(eta) = blended_pilot_axis_contrasts(pilot_data.view(), planned_term, centers)
            else {
                continue;
            };
            if set_spatial_aniso_log_scales(&mut working, term_idx, eta).is_ok() {
                updated_terms += usize::from(pass == 0);
            }
        }

        match apply_pilot_spatial_psi_reseed(
            pilot_data.view(),
            &working,
            spatial_terms,
            kappa_options,
        ) {
            Ok(updated) => {
                working = updated;
            }
            Err(err) => {
                log::warn!(
                    "[spatial-kappa] pilot geometry ψ reseed skipped after deterministic initializer error: {err}"
                );
                break;
            }
        }
    }

    if updated_terms > 0 {
        log::info!(
            "[spatial-kappa] initialized anisotropy from {}-row pilot geometry for {} spatial term(s); proceeding to full-data optimization",
            indices.len(),
            updated_terms
        );
        *spec = working;
    }
    updated_terms
}


pub(crate) fn spatial_length_scale_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    spec.smooth_terms
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| spatial_term_supports_hyper_optimization(spec, idx).then_some(idx))
        .collect()
}


/// Returns `true` when every spatial term in `spec` has a locked kernel
/// scale (explicit `length_scale=X` without anisotropy) and therefore
/// contributes no outer ψ/κ optimization axis. Empty term collections
/// also return `true` — there are no kappas to optimize.
///
/// Used by family entry points that want to honor a user-supplied scalar
/// length scale exactly: when all spatial terms are locked the n-block
/// joint-spatial outer solver has nothing to optimize, and routing
/// through it merely spends ~80 outer iters chasing a stalled ARC at the
/// user's chosen ρ. Skipping straight to the rho-only path avoids that
/// waste and respects the user's explicit kernel-scale input.
pub fn all_spatial_terms_kappa_fixed(spec: &TermCollectionSpec) -> bool {
    spec.smooth_terms.iter().enumerate().all(|(idx, _)| {
        !spatial_term_supports_hyper_optimization(spec, idx)
            || spatial_term_has_locked_kappa(spec, idx)
    })
}


pub(crate) fn spatial_term_uses_per_axis_psi(resolvedspec: &TermCollectionSpec, term_idx: usize) -> bool {
    // Measure-jet enrolls a multi-coordinate DIAL group (α, lnτ[, s]) —
    // grouped like per-axis anisotropy in the θ layout, but the coordinates
    // are geometry dials, not axis scales. Enrollment is owned by
    // `measure_jet_enrolls_psi`.
    if let Some(mj) = measure_jet_term_spec(resolvedspec, term_idx) {
        return measure_jet_enrolls_psi(mj);
    }
    let Some(d) = get_spatial_feature_dim(resolvedspec, term_idx) else {
        return false;
    };
    if d <= 1 {
        return false;
    }
    let Some(eta) = get_spatial_aniso_log_scales(resolvedspec, term_idx) else {
        return false;
    };
    if eta.len() != d {
        return false;
    }
    !matches!(
        resolvedspec.smooth_terms.get(term_idx).map(|t| &t.basis),
        Some(SmoothBasisSpec::Duchon { .. })
    )
}


/// Compute `dims_per_term` for a list of spatial term indices.
///
/// Returns a vector where entry i is the number of stored ψ values for
/// spatial term i: `d` for terms that enroll per-axis anisotropy in the
/// REML joint vector (`spatial_term_uses_per_axis_psi`), `1` otherwise.
pub(crate) fn spatial_dims_per_term(
    resolvedspec: &TermCollectionSpec,
    spatial_terms: &[usize],
) -> Vec<usize> {
    spatial_terms
        .iter()
        .map(|&term_idx| {
            if let Some(mj) = measure_jet_term_spec(resolvedspec, term_idx) {
                // Dial group, not per-axis anisotropy; layout owned by
                // `measure_jet_psi_dim`.
                measure_jet_psi_dim(mj)
            } else if spatial_term_uses_per_axis_psi(resolvedspec, term_idx) {
                get_spatial_feature_dim(resolvedspec, term_idx).unwrap_or(1)
            } else {
                1
            }
        })
        .collect()
}


/// Check whether any spatial terms enroll per-axis anisotropic ψ in the joint
/// outer vector. Mirrors the hyper_dirs builder's enrollment predicate so the
/// outer θ-layout cannot drift from the inner evaluator's ψ count.
pub(crate) fn has_aniso_terms(resolvedspec: &TermCollectionSpec, spatial_terms: &[usize]) -> bool {
    spatial_terms
        .iter()
        .any(|&term_idx| spatial_term_uses_per_axis_psi(resolvedspec, term_idx))
}


/// Emits the `theta`-keyed memoization accessors shared verbatim by the
/// single-block and n-block exact-joint design caches. Both carry the same
/// `current_theta` / `last_cost` / `last_eval` fields, so the cost/eval
/// lookups and the `store_eval` writer are identical; this macro is the single
/// source so the two inherent impls cannot drift.
fn set_spatial_length_scale(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    length_scale: f64,
) -> Result<(), EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!("spatial length-scale term index {term_idx} out of range");
    };
    match &mut term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.length_scale = Some(length_scale);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not expose a spatial length scale",
            term.name
        ))),
    }
}


/// Apply a length scale to a single `SmoothTermSpec` (independent of any
/// outer `TermCollectionSpec`). Mirrors `set_spatial_length_scale` but on a
/// term in isolation; used by the incremental realizer's cached planned spec.
pub(crate) fn set_single_term_spatial_length_scale(
    term: &mut SmoothTermSpec,
    length_scale: f64,
) -> Result<(), EstimationError> {
    match &mut term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.length_scale = Some(length_scale);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not expose a spatial length scale",
            term.name
        ))),
    }
}


/// Apply anisotropy contrasts to a single `SmoothTermSpec`. Mirrors
/// `set_spatial_aniso_log_scales` but on a term in isolation; used by the
/// incremental realizer's cached planned spec.
pub(crate) fn set_single_term_spatial_aniso_log_scales(
    term: &mut SmoothTermSpec,
    eta: Vec<f64>,
) -> Result<(), EstimationError> {
    let eta = center_aniso_log_scales(&eta);
    match &mut term.basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not support aniso_log_scales",
            term.name
        ))),
    }
}


pub fn get_spatial_length_scale(spec: &TermCollectionSpec, term_idx: usize) -> Option<f64> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::ThinPlate { spec, .. } => Some(spec.length_scale),
            SmoothBasisSpec::Matern { spec, .. } => Some(spec.length_scale),
            SmoothBasisSpec::Duchon { spec, .. } => spec.length_scale,
            _ => None,
        })
}


/// The signed sectional curvature κ of a constant-curvature smooth at
/// `term_idx`, or `None` if that term is not a `curv(...)` smooth. After a fit
/// with κ-optimization enabled this reads the **fitted κ̂** out of the resolved
/// spec (`freeze_term_collection_from_design` writes the optimized κ back into
/// the spec, and `BasisMetadata::ConstantCurvature.kappa` carries the same
/// value). This is the headline #944 estimand accessor — the κ̂ in
/// "κ̂ = −1.8 (95% CI …)". Mirrors [`get_spatial_length_scale`].
pub fn get_constant_curvature_kappa(spec: &TermCollectionSpec, term_idx: usize) -> Option<f64> {
    constant_curvature_term_spec(spec, term_idx).map(|cc| cc.kappa)
}


/// Indices of every constant-curvature (`curv(...)`) smooth term in `spec`.
pub fn constant_curvature_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    (0..spec.smooth_terms.len())
        .filter(|&idx| constant_curvature_term_spec(spec, idx).is_some())
        .collect()
}


/// Freeze a `TermCollectionSpec` by baking in the concrete knots, centers,
/// identifiability transforms, and random-effect levels that were resolved
/// during design-matrix construction.  The result passes `validate_frozen`
/// and is safe to serialize for prediction.
///
/// This is the single canonical freezer — every model-save path should call
/// this rather than rolling ad-hoc freezing logic.
/// Freeze a smooth basis spec from its fit-time metadata so that predict-time
/// rebuilds reproduce the exact fitted geometry instead of recomputing any
/// data-dependent construction (knot selection, radial reparameterization,
/// eigen-truncation, identifiability constraint) on the prediction rows.
///
/// This is the SINGLE source of truth for freezing, shared by stand-alone
/// terms and by `by=`-wrapped / factor-sum-to-zero inner smooths. The wrapper
/// arms recurse into this same function, so every inner basis kind is frozen
/// with identical logic. A previous split implementation froze only B-spline
/// inners, leaving spatial inner bases (`bs='tp'`/`matern`/`duchon`/`sos`)
/// unfrozen and recomputed on the prediction grid (#704).
