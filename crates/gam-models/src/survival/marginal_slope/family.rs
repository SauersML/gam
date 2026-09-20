//! The `SurvivalMarginalSlopeFamily` data container itself: its fields, the
//! intercept warm-start cache, per-fit hint state, and the small accessor /
//! flex-block-routing methods that read the family's own configuration
//! (which optional blocks are active, where each block's coefficients live).

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurvivalMarginalSlopeFamilyHyperAxis {
    Baseline(usize),
    LogSigma,
}

/// Explicit local role map for family-owned hyperparameter coordinates.
///
/// Baseline coordinates are first and log-sigma, when learned, is last. Fixed
/// frailty scale has no log-sigma coordinate. The role map is stored on every
/// realized family so callbacks never infer semantics from empty derivative
/// matrices or floating-point values.
#[derive(Clone, Debug, Default)]
pub(crate) struct SurvivalMarginalSlopeFamilyHyperState {
    baseline_axis_count: usize,
    log_sigma_axis: Option<usize>,
    /// Exact family-coordinate values used to realize this family instance.
    /// Kept bitwise aligned with the family tail of `CustomFamilyHyperLayout`
    /// so a workspace cannot accidentally reuse row geometry from a
    /// neighbouring outer probe.
    family_values: Array1<f64>,
    pub(crate) baseline_geometry:
        Option<Arc<crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry>>,
}

impl SurvivalMarginalSlopeFamilyHyperState {
    pub(crate) fn new(
        baseline_geometry: Option<
            Arc<crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry>,
        >,
        learned_log_sigma: Option<f64>,
    ) -> Result<Self, String> {
        let baseline_axis_count = baseline_geometry
            .as_ref()
            .map_or(0, |geometry| geometry.theta.len());
        let mut family_values = baseline_geometry
            .as_ref()
            .map_or_else(Vec::new, |geometry| geometry.theta.to_vec());
        if let Some(log_sigma) = learned_log_sigma {
            if !log_sigma.is_finite() {
                return Err(
                    "survival marginal-slope learned log-sigma coordinate must be finite"
                        .to_string(),
                );
            }
            family_values.push(log_sigma);
        }
        if family_values.iter().any(|value| !value.is_finite()) {
            return Err(
                "survival marginal-slope baseline family coordinates must be finite".to_string(),
            );
        }
        Ok(Self {
            baseline_axis_count,
            log_sigma_axis: learned_log_sigma.map(|_| baseline_axis_count),
            family_values: Array1::from_vec(family_values),
            baseline_geometry,
        })
    }

    /// Absorb the family-owned hyper coordinates into a persistent warm-start
    /// key (#3697): the role map, the realized coordinate values and the whole
    /// baseline offset geometry. The geometry's `baseline_config` is the recipe
    /// its arrays were evaluated from at `theta`; it reaches the family only
    /// through those arrays, which are hashed value by value.
    pub(crate) fn fingerprint_into(&self, hasher: &mut gam_runtime::warm_start::Fingerprinter) {
        let Self {
            baseline_axis_count,
            log_sigma_axis,
            family_values,
            baseline_geometry,
        } = self;
        hasher.write_usize(*baseline_axis_count);
        match log_sigma_axis {
            None => hasher.write_bool(false),
            Some(axis) => {
                hasher.write_bool(true);
                hasher.write_usize(*axis);
            }
        }
        hasher.write_f64_array1(family_values);
        let Some(geometry) = baseline_geometry else {
            hasher.write_bool(false);
            return;
        };
        hasher.write_bool(true);
        let crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry {
            baseline_config: _,
            theta,
            offset_entry,
            offset_exit,
            derivative_offset_exit,
            offset_entry_theta_first,
            offset_exit_theta_first,
            derivative_offset_exit_theta_first,
            offset_entry_theta_second,
            offset_exit_theta_second,
            derivative_offset_exit_theta_second,
        } = geometry.as_ref();
        hasher.write_f64_array1(theta);
        hasher.write_f64_array1(offset_entry);
        hasher.write_f64_array1(offset_exit);
        hasher.write_f64_array1(derivative_offset_exit);
        hasher.write_f64_array2(offset_entry_theta_first);
        hasher.write_f64_array2(offset_exit_theta_first);
        hasher.write_f64_array2(derivative_offset_exit_theta_first);
        for second in [
            offset_entry_theta_second,
            offset_exit_theta_second,
            derivative_offset_exit_theta_second,
        ] {
            for &extent in second.shape() {
                hasher.write_usize(extent);
            }
            for &value in second.iter() {
                hasher.write_f64(value);
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.baseline_axis_count + usize::from(self.log_sigma_axis.is_some())
    }

    pub(crate) fn role(
        &self,
        family_axis: usize,
    ) -> Option<SurvivalMarginalSlopeFamilyHyperAxis> {
        if family_axis < self.baseline_axis_count {
            Some(SurvivalMarginalSlopeFamilyHyperAxis::Baseline(
                family_axis,
            ))
        } else if self.log_sigma_axis == Some(family_axis) {
            Some(SurvivalMarginalSlopeFamilyHyperAxis::LogSigma)
        } else {
            None
        }
    }

    pub(crate) fn validate_layout(
        &self,
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
    ) -> Result<(), String> {
        if hyper_layout.family_axis_count() != self.len() {
            return Err(format!(
                "SurvivalMarginalSlopeFamily declares {} family hyper axes, manifest carries {}",
                self.len(),
                hyper_layout.family_axis_count(),
            ));
        }
        let manifest_family_values = hyper_layout
            .values()
            .slice(s![hyper_layout.design_axis_count()..]);
        if manifest_family_values.len() != self.family_values.len()
            || manifest_family_values
                .iter()
                .zip(self.family_values.iter())
                .any(|(manifest, realized)| manifest.to_bits() != realized.to_bits())
        {
            return Err(
                "SurvivalMarginalSlopeFamily row geometry does not bitwise match the family-coordinate manifest"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// The time block has one beta vector but THREE design matrices (entry, exit,
/// derivative-at-exit). The ParameterBlockSpec uses the exit design as its
/// "official" design, so block_states[0].eta = design_exit @ beta + offset_exit.
/// This eta is NOT used in the likelihood computation — row_neglog_directional
/// recomputes all 3 linear predictors from beta_time directly. The exit-design
/// eta exists only to satisfy the CustomFamily/PIRLS interface; ExactNewton
/// blocks do not use eta for working response/weights.
#[derive(Clone)]
pub(crate) struct SurvivalMarginalSlopeFamily {
    pub(crate) n: usize,
    pub(crate) event: Arc<Array1<f64>>,
    pub(crate) weights: Arc<Array1<f64>>,
    pub(crate) z: Arc<Array2<f64>>,
    /// The score covariance the row program consumes, indexed BY ROW.
    ///
    /// `Σ` is `Var(z | a)` in the identity this family is defined by, so it is a
    /// function of the marginal-index span and not a single matrix; the field
    /// carries the pooled object when the conditional gate did not fire and a
    /// materialised `Σ(a_i)` per row when it did (gam#2766).
    pub(crate) score_covariance: ScoreCovarianceField,
    pub(crate) gaussian_frailty_sd: Option<f64>,
    pub(crate) family_hyper: SurvivalMarginalSlopeFamilyHyperState,
    pub(crate) derivative_guard: f64,
    /// Time block: 3 designs sharing one beta vector.
    /// Stored as DesignMatrix to support sparse local-support bases at
    /// large scale (B-spline/I-spline rows have only degree+1 nonzeros).
    pub(crate) design_entry: DesignMatrix,
    pub(crate) design_exit: DesignMatrix,
    pub(crate) design_derivative_exit: DesignMatrix,
    pub(crate) offset_entry: Arc<Array1<f64>>,
    pub(crate) offset_exit: Arc<Array1<f64>>,
    pub(crate) derivative_offset_exit: Arc<Array1<f64>>,
    /// Rows that enter at the time origin (`age_entry ≤ ENTRY_AT_ORIGIN_THRESHOLD`,
    /// the predicate `survival::base` uses). Such a row is not left-truncated:
    /// `S(0) = 1`, so its likelihood carries no `log Φ(−η₀)` entry factor. Its
    /// entry time is floored to `SURVIVAL_TIME_FLOOR` and its entry design row
    /// is the I-spline's left-boundary row, so the stand-in `η₀` is the fitted
    /// index at the first exit time, not the `η₀ → −∞` limit. Conditioning on it
    /// rewarded raising that index for every landmarked row: the fitted curve
    /// went flat below the first exits with `S(0⁺) ≈ 0.97` (gnomon#2336).
    pub(crate) entry_at_origin: Arc<Array1<bool>>,
    /// Baseline covariate block: contributes additively to q0 and q1, but not qd1.
    pub(crate) marginal_design: DesignMatrix,
    /// The slope coefficient design, its physical channels in current
    /// coordinates, and their baseline + smooth offsets. This is the sole
    /// source of truth for both scalar and per-score slope geometry.
    pub(crate) slope_layout: SlopeLayout,
    pub(crate) score_warp: Option<DeviationRuntime>,
    pub(crate) link_dev: Option<DeviationRuntime>,
    /// Absorbed Stage-1 influence columns `Z̃_infl` at the training rows
    /// (`n × p₁`), residualized against the marginal location span in the
    /// rigid-pilot row metric (#461, design §3). When `Some`, the family hosts a
    /// dedicated additive absorber block whose coefficient `γ` shifts the
    /// de-nested observed index `η₁` by `+Z̃_infl[row,:]·γ` (sibling of the
    /// per-row calibration intercept — un-`c(g)`-scaled, unlike the marginal
    /// block which enters the time-quantile location through `q·c(g)`). The
    /// block carries a fixed small ridge and is dropped at predict. `None` ⇒ raw
    /// `z` with no CTN Stage-1; the free-warp `score_warp` is the fallback basis.
    pub(crate) influence_absorber: Option<Array2<f64>>,
    pub(crate) time_linear_constraints: Option<LinearInequalityConstraints>,
    pub(crate) time_wiggle_knots: Option<Array1<f64>>,
    pub(crate) time_wiggle_degree: Option<usize>,
    pub(crate) time_wiggle_ncols: usize,
    /// Per-row cache of the previous PIRLS iter's converged intercepts. Two
    /// slots per row: `[entry_q0, exit_q1]`. Across consecutive PIRLS
    /// iterations β changes only a little, so the previously-converged `a` is
    /// an excellent initial guess for the calibration root and typically lets
    /// the solver finish in ~1–2 iterations versus the rigid closed-form seed
    /// which can be many bracket-expansion steps away. Slots are initialised
    /// to `NaN` (sentinel for "not yet solved") and overwritten with the
    /// converged intercept on every successful call.
    ///
    /// Set to `None` for unit-test fixtures that build a
    /// `SurvivalMarginalSlopeFamily` directly without running the full fit
    /// pipeline; production paths go through `make_family` which initialises
    /// the cache to length-`n`. When `None`, the solver behaves exactly as it
    /// did before the warm-start machinery was added (closed-form rigid seed).
    pub(crate) intercept_warm_starts: Option<Arc<SurvivalInterceptWarmStartCache>>,
    /// The fit's pool of runtime-sized FLEX jet arenas, drawn on by the
    /// third-order directional contractions and the order-≤2 timepoint builder.
    /// An idle arena stays only while the MemoryGovernor admits the bytes it
    /// retains, and the pool ends with the fit's families (gam#3266).
    pub(crate) flex_jet_arenas: Arc<FlexJetArenaPool>,
    /// Whether this member's Jeffreys/Firth prior is armed. A fit arms it only
    /// on the unarmed fit's own evidence, through
    /// `fit_custom_family_arming_on_evidence` (#979).
    pub(crate) jeffreys_armed: bool,
    /// The declared latent law the row index is anchored on (gam#2923), when
    /// the fit runs the anchored frame instead of the Gaussian closed form.
    /// `None` is the standard-normal law, on which the closed form is exact and
    /// every model built before this existed takes exactly the path it did.
    pub(crate) latent_law: Option<Arc<SurvivalLatentLaw>>,
}

impl SurvivalMarginalSlopeFamily {
    /// Fingerprint of everything that defines this family's penalized inner
    /// objective, for the persistent warm-start key (#3697). A key hit returns
    /// the cached mode, log-likelihood and log-determinants without
    /// re-evaluating them, so any field that changes the likelihood and is
    /// missing here serves one model's evidence to another.
    ///
    /// The family is destructured exhaustively: a field added later is a
    /// compile error here until it is either hashed or named as not part of
    /// the likelihood. The two fields ignored are solver scratch, not model:
    /// the intercept warm starts only seed the calibration root, which is
    /// solved to convergence regardless, and the jet arenas are memory pools.
    pub(crate) fn likelihood_fingerprint(&self) -> Result<String, String> {
        use gam_custom_family::hash_cf_design_matrix;
        let Self {
            n,
            event,
            weights,
            z,
            score_covariance,
            gaussian_frailty_sd,
            family_hyper,
            derivative_guard,
            design_entry,
            design_exit,
            design_derivative_exit,
            offset_entry,
            offset_exit,
            derivative_offset_exit,
            entry_at_origin,
            marginal_design,
            slope_layout,
            score_warp,
            link_dev,
            influence_absorber,
            time_linear_constraints,
            time_wiggle_knots,
            time_wiggle_degree,
            time_wiggle_ncols,
            intercept_warm_starts: _,
            flex_jet_arenas: _,
            jeffreys_armed,
            latent_law,
        } = self;
        let mut hasher = gam_runtime::warm_start::Fingerprinter::new();
        hasher.write_str("survival-marginal-slope-family");
        hasher.write_usize(*n);
        hasher.write_f64_array1(event);
        hasher.write_f64_array1(weights);
        hasher.write_f64_array2(z);
        score_covariance.fingerprint_into(&mut hasher)?;
        match gaussian_frailty_sd {
            Some(value) => {
                hasher.write_bool(true);
                hasher.write_f64(*value);
            }
            None => hasher.write_bool(false),
        }
        family_hyper.fingerprint_into(&mut hasher);
        hasher.write_f64(*derivative_guard);
        hash_cf_design_matrix(&mut hasher, design_entry)?;
        hash_cf_design_matrix(&mut hasher, design_exit)?;
        hash_cf_design_matrix(&mut hasher, design_derivative_exit)?;
        hasher.write_f64_array1(offset_entry);
        hasher.write_f64_array1(offset_exit);
        hasher.write_f64_array1(derivative_offset_exit);
        hasher.write_usize(entry_at_origin.len());
        for &at_origin in entry_at_origin.iter() {
            hasher.write_bool(at_origin);
        }
        hash_cf_design_matrix(&mut hasher, marginal_design)?;
        slope_layout.fingerprint_into(&mut hasher)?;
        for deviation in [score_warp, link_dev] {
            match deviation {
                Some(runtime) => {
                    hasher.write_bool(true);
                    runtime.fingerprint_into(&mut hasher);
                }
                None => hasher.write_bool(false),
            }
        }
        match influence_absorber {
            Some(columns) => {
                hasher.write_bool(true);
                hasher.write_f64_array2(columns);
            }
            None => hasher.write_bool(false),
        }
        match time_linear_constraints {
            Some(LinearInequalityConstraints { a, b }) => {
                hasher.write_bool(true);
                hasher.write_f64_array2(a);
                hasher.write_f64_array1(b);
            }
            None => hasher.write_bool(false),
        }
        match time_wiggle_knots {
            Some(knots) => {
                hasher.write_bool(true);
                hasher.write_f64_array1(knots);
            }
            None => hasher.write_bool(false),
        }
        match time_wiggle_degree {
            Some(degree) => {
                hasher.write_bool(true);
                hasher.write_usize(*degree);
            }
            None => hasher.write_bool(false),
        }
        hasher.write_usize(*time_wiggle_ncols);
        hasher.write_bool(*jeffreys_armed);
        match latent_law {
            Some(law) => {
                hasher.write_bool(true);
                law.fingerprint_into(&mut hasher)?;
            }
            None => hasher.write_bool(false),
        }
        Ok(hasher.finish_hex())
    }

    /// The weight of the row's entry survival factor `log Φ(−η₀)`: the row's
    /// prior weight for a delayed entry, `0` for an entry at the time origin.
    #[inline]
    pub(crate) fn entry_weight(&self, row: usize) -> f64 {
        if self.entry_at_origin[row] {
            0.0
        } else {
            self.weights[row]
        }
    }

    /// The row's slope index on its three follow-up channels (gam#2765).
    ///
    /// The block's own linear predictor is already the EXIT-time slope — the
    /// slope block's `ParameterBlockSpec` design is the exit design, exactly
    /// as the time block's is. A time-constant slope is the degenerate case
    /// `g₀ = g₁`, `ġ₁ = 0`; a follow-up-varying one reads the other two channels
    /// off the layout's entry / exit-derivative designs at the same
    /// coefficients.
    #[inline]
    pub(crate) fn row_slope_channels(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SlopeRowChannels, String> {
        let state = block_states.get(2).ok_or_else(|| {
            "survival marginal-slope row slope channels require the slope block state"
                .to_string()
        })?;
        self.slope_layout
            .row_channels(row, &state.beta, state.eta[row])
    }

    /// Whether this family's slope moves along the follow-up axis. Selects the
    /// six-primary row frame over the four-primary one.
    #[inline]
    pub(crate) fn slope_is_follow_up_varying(&self) -> bool {
        self.slope_layout.is_follow_up_varying()
    }

    /// Whether this family anchors its index on a declared latent law
    /// (gam#2923). Selects the anchored four-primary frame over the Gaussian
    /// closed-form one.
    #[inline]
    pub(crate) fn anchored_law_active(&self) -> bool {
        self.latent_law.is_some()
    }

    /// The joint latent law of the score vector a per-score `K ≥ 2` fit
    /// anchors on (gam#2929), when it runs the anchored vector program.
    #[inline]
    pub(crate) fn joint_latent_law(&self) -> Option<&JointLatentLawRuntime> {
        self.latent_law.as_ref().and_then(|law| law.joint())
    }

    /// How many primaries the family's CORE (non-flex) row frame carries. The
    /// runtime counterpart of the `SlopeRowGeometry` const parameter, for the
    /// `ndarray`-shaped surfaces that carry the frame dynamically.
    #[inline]
    pub(crate) fn core_primary_dimension(&self) -> usize {
        if self.slope_is_follow_up_varying() {
            DYNAMIC_SLOPE_PRIMARIES
        } else {
            STATIC_SLOPE_PRIMARIES
        }
    }

    /// Whether the ψ workspace installs a second-order pair calculus for every pair of this
    /// family's ψ coordinates beside `design_axes` design axes; see
    /// [`second_order_psi_pairs_served`] (gam#2765).
    pub(crate) fn psi_second_order_pairs_served(&self, design_axes: usize) -> bool {
        second_order_psi_pairs_served(
            self.family_hyper.baseline_axis_count,
            self.family_hyper.log_sigma_axis.is_some(),
            design_axes,
            !self.per_z_slope_active() && (self.flex_active() || self.flex_timewiggle_active()),
        )
    }

    /// Whether the rigid frame serves every ψ-mixed third information derivative an armed
    /// Jeffreys objective's exact outer Hessian reads. Design and baseline-chart axes have
    /// closed forms there; a learned log σ has none along a coefficient direction, and the ψ
    /// workspace refuses it (gam#2765).
    pub(crate) fn rigid_psi_jeffreys_third_served(&self) -> bool {
        self.rigid_third_information_available() && self.family_hyper.log_sigma_axis.is_none()
    }

    /// Whether the ζ composition of `timewiggle_third` serves every ψ-mixed third information
    /// derivative an armed Jeffreys objective's exact outer Hessian reads under a time wiggle.
    /// Design axes (gam#2893) and baseline-chart axes (gam#3061) have ζ closed forms; a learned
    /// log σ has none.
    pub(crate) fn timewiggle_psi_jeffreys_third_served(&self) -> bool {
        self.timewiggle_zeta_fifth_available() && self.family_hyper.log_sigma_axis.is_none()
    }

    /// Memoize the dense form of each operator-backed covariate design the
    /// rigid row kernel reads one row at a time (gnomon#2337).
    ///
    /// Every gradient, Hessian-vector product and Hessian diagonal pass reads
    /// the marginal and time-constant slope designs row by row
    /// (`jacobian_action`, `jacobian_transpose_action`,
    /// `add_diagonal_quadratic`). A gauged Duchon design is a coefficient
    /// transform over a stacked block operator, so each of those reads streams
    /// one row through the operator stack — allocations and a one-row GEMM per
    /// row, per design, per pass — and the allocator serializes the row-parallel
    /// passes. The governed memo keeps one ledger-charged dense copy shared by
    /// every clone of the design, so this family and every workspace built from
    /// it read rows from that copy. A refusal (construction policy, byte cap or
    /// joint-ledger pressure) keeps the streamed storage the design already had.
    pub(crate) fn memoize_operator_backed_designs(&self) {
        let designs = std::iter::once(("marginal", &self.marginal_design)).chain(
            self.slope_layout
                .static_coefficient_design()
                .map(|design| ("slope", design)),
        );
        for (label, design) in designs {
            if design.is_sparse() || design.as_dense_ref().is_some() {
                continue;
            }
            if let Err(reason) =
                design.try_to_dense_arc("survival marginal-slope row-kernel design memo")
            {
                log::debug!(
                    "[survival-marginal-slope] {label} design stays streamed ({}x{}), so every \
                     row-kernel pass reads it one row at a time through its operator: {reason}",
                    design.nrows(),
                    design.ncols(),
                );
            }
        }
    }
}

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn family_hyper_role(
        &self,
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        global_axis: usize,
    ) -> Result<Option<SurvivalMarginalSlopeFamilyHyperAxis>, String> {
        self.family_hyper.validate_layout(hyper_layout)?;
        match hyper_layout.axis(global_axis) {
            Some(crate::custom_family::CustomFamilyHyperAxis::DesignPenalty { .. }) => Ok(None),
            Some(crate::custom_family::CustomFamilyHyperAxis::Family { family_axis }) => self
                .family_hyper
                .role(family_axis)
                .map(Some)
                .ok_or_else(|| {
                    format!(
                        "SurvivalMarginalSlopeFamily has no local family hyper axis {family_axis}"
                    )
                }),
            None => Err(format!(
                "SurvivalMarginalSlopeFamily hyper axis {global_axis} is out of range for {} axes",
                hyper_layout.len()
            )),
        }
    }
}

/// Discriminates the two intercept slots per row: the entry-time intercept
/// (solved against `q0`) and the exit-time intercept (solved against `q1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurvivalInterceptSlotKind {
    Entry = 0,
    Exit = 1,
}

/// Per-row warm-start storage for the survival calibration root solver.
///
/// Two slots per row (entry intercept against `q0`, exit intercept against
/// `q1`). Each slot stores the converged intercept `a` alongside a
/// `beta_tag: u64` — a 64-bit hash of the joint coefficient vector at the
/// time of write. Reads return `Some(a)` only when the caller's tag matches
/// the stored tag AND the stored value is finite. This makes the cache
/// transactional with respect to trust-region trials and subsampled probes:
/// a rejected trial at β_A and an accepted full-data eval at β_B key under
/// distinct tags, so writes from one cannot poison reads from the other.
///
/// The "never written" sentinel is `beta_tag == 0`. Callers compute their
/// tag with `hash_intercept_warm_start_key` and remap `0` to `1` so that the
/// sentinel can never collide with a real key. Two consecutive evaluations
/// at the same β share the same tag and reuse the cached root.
///
/// Memory ordering: the writer stores `value` with `Relaxed` and then `tag`
/// with `Release`. The reader loads `tag` with `Acquire`, reads `value`
/// with `Relaxed`, and re-checks `tag` with `Acquire`. The double-check
/// detects a torn read where another thread interleaved a tag bump between
/// the value read and the second tag load.
pub(crate) struct SurvivalInterceptWarmStartCache {
    pub(crate) entry_value: Vec<std::sync::atomic::AtomicU64>,
    pub(crate) entry_tag: Vec<std::sync::atomic::AtomicU64>,
    pub(crate) exit_value: Vec<std::sync::atomic::AtomicU64>,
    pub(crate) exit_tag: Vec<std::sync::atomic::AtomicU64>,
}

impl SurvivalInterceptWarmStartCache {
    #[inline]
    pub(crate) fn slots_for(
        &self,
        kind: SurvivalInterceptSlotKind,
    ) -> (
        &[std::sync::atomic::AtomicU64],
        &[std::sync::atomic::AtomicU64],
    ) {
        match kind {
            SurvivalInterceptSlotKind::Entry => (&self.entry_value, &self.entry_tag),
            SurvivalInterceptSlotKind::Exit => (&self.exit_value, &self.exit_tag),
        }
    }

    /// Return the cached intercept iff the slot's stored `beta_tag` matches
    /// the caller's `beta_tag` and the stored value is finite. Otherwise
    /// returns `None` (cache miss — caller falls back to closed-form seed).
    #[inline]
    pub(crate) fn load(
        &self,
        row: usize,
        kind: SurvivalInterceptSlotKind,
        beta_tag: u64,
    ) -> Option<f64> {
        let (values, tags) = self.slots_for(kind);
        let value_slot = values.get(row)?;
        let tag_slot = tags.get(row)?;
        let tag_before = tag_slot.load(std::sync::atomic::Ordering::Acquire);
        if tag_before != beta_tag {
            return None;
        }
        let bits = value_slot.load(std::sync::atomic::Ordering::Relaxed);
        let tag_after = tag_slot.load(std::sync::atomic::Ordering::Acquire);
        if tag_after != beta_tag {
            return None;
        }
        let value = f64::from_bits(bits);
        value.is_finite().then_some(value)
    }

    /// Stamp the slot with the converged intercept under `beta_tag`. Concurrent
    /// writers from different trials race; the last writer wins, which is fine
    /// because every reader gates on its own tag and only accepts a match.
    #[inline]
    pub(crate) fn store(&self, row: usize, kind: SurvivalInterceptSlotKind, a: f64, beta_tag: u64) {
        let (values, tags) = self.slots_for(kind);
        if let (Some(value_slot), Some(tag_slot)) = (values.get(row), tags.get(row)) {
            // Invalidate before writing the new value so an interleaved
            // reader cannot see the new tag paired with the old value.
            tag_slot.store(0, std::sync::atomic::Ordering::Release);
            value_slot.store(a.to_bits(), std::sync::atomic::Ordering::Relaxed);
            tag_slot.store(beta_tag, std::sync::atomic::Ordering::Release);
        }
    }
}

pub(crate) fn new_intercept_warm_start_cache(n: usize) -> Arc<SurvivalInterceptWarmStartCache> {
    Arc::new(SurvivalInterceptWarmStartCache {
        entry_value: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()))
            .collect(),
        entry_tag: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect(),
        exit_value: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()))
            .collect(),
        exit_tag: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect(),
    })
}

/// Reusable FLEX jet arenas whose retained tapes are on the governor's ledger
/// (gam#3266).
pub(crate) type FlexJetArenaPool =
    gam_runtime::resource::GovernedScratchPool<gam_math::jet_scalar::DynamicJetArena>;

/// A fit's FLEX jet arena pool on the process governor. Each arena is charged
/// at the one chunk its reset keeps, the high-water mark of its tape.
pub(crate) fn new_flex_jet_arena_pool() -> Arc<FlexJetArenaPool> {
    Arc::new(gam_runtime::resource::GovernedScratchPool::new(
        gam_runtime::resource::MemoryGovernor::global().clone(),
        "survival marginal-slope flex jet arena",
        gam_math::jet_scalar::DynamicJetArena::new,
        gam_math::jet_scalar::DynamicJetArena::allocated_bytes,
    ))
}

/// FNV-1a 64-bit hash of the joint coefficient slices `(beta_h, beta_w)`.
/// Returned tag is guaranteed non-zero (zero is remapped to one) so that
/// the cache's "never written" sentinel cannot collide with a real key.
/// At 64 bits, false collisions across distinct β are astronomically rare;
/// on a miss we just re-solve from the closed-form seed.
#[inline]
pub(crate) fn hash_intercept_warm_start_key(
    beta_h: Option<&Array1<f64>>,
    beta_w: Option<&Array1<f64>>,
) -> u64 {
    let mut hash = Fnv1a::new();
    hash.mix_opt_beta(0xa1, beta_h);
    hash.mix_opt_beta(0xa2, beta_w);
    hash.finish_nonzero()
}

#[derive(Clone, Default)]
pub(crate) struct ThetaHints {
    pub(crate) time_beta: Option<Array1<f64>>,
    pub(crate) marginal_beta: Option<Array1<f64>>,
    pub(crate) slope_beta: Option<Array1<f64>>,
    pub(crate) score_warp_beta: Option<Array1<f64>>,
    pub(crate) link_dev_beta: Option<Array1<f64>>,
    pub(crate) influence_beta: Option<Array1<f64>>,
}

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn time_derivative_lower_bound(&self) -> f64 {
        assert!(
            self.derivative_guard.is_finite() && self.derivative_guard > 0.0,
            "survival marginal-slope derivative guard must be finite and positive: derivative_guard={}",
            self.derivative_guard
        );
        self.derivative_guard
    }

    pub(crate) fn flex_active(&self) -> bool {
        // The absorbed influence block (#461) rides the dynamic-Q primary-jet
        // path (it adds the `o_infl` primary coordinate), so it counts as "flex"
        // for dispatch purposes even when no score_warp / link_dev is present —
        // the rigid closed-form row kernel has no `o_infl` channel.
        self.score_warp.is_some() || self.link_dev.is_some() || self.influence_absorber.is_some()
    }

    pub(crate) fn effective_flex_active(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<bool, String> {
        if self.score_warp.is_some() && self.flex_score_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival score-warp block state".to_string(),
            }
            .into());
        }
        if self.link_dev.is_some() && self.flex_link_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival link-deviation block state".to_string(),
            }
            .into());
        }
        if self.influence_absorber.is_some() && self.flex_influence_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival influence-absorber block state".to_string(),
            }
            .into());
        }
        Ok(self.flex_active())
    }

    pub(crate) fn flex_score_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.score_warp.is_none() {
            return Ok(None);
        }
        block_states
            .get(3)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival score-warp block state".to_string())
    }

    pub(crate) fn flex_link_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.link_dev.is_none() {
            return Ok(None);
        }
        let idx = if self.score_warp.is_some() { 4 } else { 3 };
        block_states
            .get(idx)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival link-deviation block state".to_string())
    }

    /// Coefficient `γ` of the absorbed Stage-1 influence block (#461). The
    /// absorber is the trailing block, so its index is `3 + score_warp? +
    /// link_dev?`. `None` when no influence Jacobian was installed.
    pub(crate) fn flex_influence_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.influence_absorber.is_none() {
            return Ok(None);
        }
        let idx = 3 + usize::from(self.score_warp.is_some()) + usize::from(self.link_dev.is_some());
        block_states
            .get(idx)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival influence-absorber block state".to_string())
    }

    /// Per-row absorbed-influence index offset `o_infl[row] = Z̃_infl[row,:]·γ`.
    /// Returns `0.0` when no absorber is installed (the additive shift vanishes),
    /// so callers can fold it unconditionally into the de-nested observed `η₁`.
    pub(crate) fn influence_index_offset(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        let (Some(z_tilde), Some(gamma)) = (
            self.influence_absorber.as_ref(),
            self.flex_influence_beta(block_states)?,
        ) else {
            return Ok(0.0);
        };
        if gamma.len() != z_tilde.ncols() {
            return Err(format!(
                "survival influence-absorber β length {} != Z̃_infl columns {}",
                gamma.len(),
                z_tilde.ncols()
            ));
        }
        Ok(z_tilde.row(row).dot(gamma))
    }
}

/// Whether the survival marginal-slope ψ workspace installs a second-order pair calculus for
/// every pair of a θ with these ψ coordinates. `second_order_terms` serves (design, design),
/// (log σ, log σ) and (baseline chart, baseline chart) pairs, a (chart, design) pair only
/// through the FLEX family program, and nothing for a learned log σ beside any other axis. An
/// exact outer Hessian over such a θ reads every pair, so declaring one there would refuse
/// every trial point that asks for curvature (gam#2765).
pub(crate) fn second_order_psi_pairs_served(
    baseline_axes: usize,
    learned_log_sigma: bool,
    design_axes: usize,
    baseline_design_through_flex: bool,
) -> bool {
    !(learned_log_sigma && baseline_axes + design_axes > 0)
        && (baseline_axes == 0 || design_axes == 0 || baseline_design_through_flex)
}

#[cfg(test)]
mod psi_curvature_declaration_tests {
    use super::*;

    /// gam#2765: the pairs `second_order_terms` installs. Before this predicate the outer
    /// Hessian declaration accepted every refused row below on the rigid frame, whatever the
    /// ψ roles were.
    #[test]
    fn second_order_psi_pairs_served_matches_the_installed_pair_calculus_2765() {
        // (baseline chart axes, learned log σ, design axes, chart×design through FLEX)
        let served = [
            ((2, false, 0, false), true), // Weibull chart alone: the #2765 recovery fixture
            ((0, false, 3, false), true), // design axes alone
            ((0, true, 0, false), true),  // learned log σ alone
            ((2, false, 2, true), true),  // chart beside design through the FLEX program
        ];
        let refused = [
            ((2, true, 0, false), false),  // log σ beside a chart axis
            ((0, true, 2, false), false),  // log σ beside a design axis
            ((2, false, 2, false), false), // rigid chart beside a design axis
            ((2, true, 2, true), false),   // log σ refuses through FLEX too
        ];
        for ((baseline, sigma, design, through_flex), expected) in served.into_iter().chain(refused)
        {
            assert_eq!(
                second_order_psi_pairs_served(baseline, sigma, design, through_flex),
                expected,
                "baseline={baseline} log_sigma={sigma} design={design} through_flex={through_flex}"
            );
        }
    }
}
