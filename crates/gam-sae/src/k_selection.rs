//! # Automatic dictionary-size (`K`) selection from the EV-vs-`K` frontier (#1026).
//!
//! The manifold-SAE fit takes a dictionary size `K` (the atom count). Today `K`
//! is user-specified; this module turns the **EV-vs-`K` frontier** measured by
//! the OLMo research battery (`tests/sae/olmo_research_battery.py`, the #1026
//! data) into a principled automatic choice.
//!
//! ## Why a knee/MDL criterion (and not REML) here
//!
//! `K` is a **discrete structure** choice — like the topology race in
//! [`gam_solve::structure_search`] picks between manifold topologies, not a
//! continuous smoothing parameter. The REML-always / no-GCV-BIC policy governs
//! *fitting* (the continuous `ρ`/`λ` smoothing tier); discrete structure
//! selection legitimately uses a knee/penalized-fit criterion (the topology
//! search already uses `score='bic'` for the same reason). So the dictionary
//! size is selected by either:
//!
//! * an **elbow / kneedle** criterion — pick the `K` at the saturation knee of
//!   the explained-variance curve, where the marginal EV gain per added atom
//!   first drops below a principled fraction of the early (steep-regime) slope,
//!   or equivalently the point of maximum curvature of the normalized curve; or
//! * a **penalized-EV / MDL** stop — maximize `EV(K) − γ · (K / K_max)`, a
//!   description-length-style trade of reconstruction gain against dictionary
//!   complexity.
//!
//! ## The manifold-vs-linear advantage
//!
//! The frontier carries *two* curves: the manifold-SAE EV-vs-`K` and a
//! linear-SAE baseline EV-vs-`K`. [`ManifoldVsLinearAdvantage`] quantifies the
//! parameter-efficiency win: manifold reaches a target EV at `K_m` atoms vs
//! linear at `K_l`, and the compression factor is the DECODER-PARAMETER ratio
//! `K_l·p_l / K_m·p_m > 1`, NOT the atom ratio `K_l / K_m`. A manifold atom
//! stores more scalars than a linear atom (a `d`-chart with an `M`-wide basis
//! over `p` channels costs `M·p`, a linear atom `p`), so counting atoms would
//! over-credit the manifold; the parameter ratio is the honest win.
//!
//! ## Degenerate curves
//!
//! Real frontiers are not always knee-shaped. The selector classifies the
//! curve and reports the verdict through [`KSelectionFlag`] so callers can act
//! on it rather than silently trusting a spurious knee:
//!
//! * **`Knee`** — a clear saturation knee was found.
//! * **`NoKnee`** — the curve keeps climbing (still steep at `K_max`): the
//!   selector returns the largest `K`, flagged, because more atoms would still
//!   help.
//! * **`Linear`** — EV grows ~linearly in `K` with no curvature: no knee
//!   exists; the largest `K` is returned, flagged.
//! * **`Flat`** — EV is already saturated at the smallest `K`: the smallest
//!   `K` is returned, flagged.

use ndarray::ArrayView2;

/// One `(K, EV)` sample on the EV-vs-`K` frontier: dictionary size `k` and the
/// explained variance / R² of the reconstruction at that size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvVsKPoint {
    /// Dictionary size (atom count) `K ≥ 1`.
    pub k: usize,
    /// Explained variance (R²) of the reconstruction at this `K`, in `(-∞, 1]`.
    pub ev: f64,
}

impl EvVsKPoint {
    pub fn new(k: usize, ev: f64) -> Self {
        Self { k, ev }
    }
}

/// An EV-vs-`K` frontier: a set of `(K, EV)` samples, e.g. from a fit sweep or
/// the OLMo research battery.
///
/// Construction sorts by `K` ascending and rejects duplicate / empty / non-
/// finite input so downstream slope math is well-defined.
#[derive(Debug, Clone)]
pub struct EvVsKCurve {
    points: Vec<EvVsKPoint>,
}

impl EvVsKCurve {
    /// Build from `(K, EV)` samples. Errors on an empty curve, a non-positive
    /// `K`, a non-finite `EV`, or duplicate `K` values.
    pub fn new(mut points: Vec<EvVsKPoint>) -> Result<Self, String> {
        if points.is_empty() {
            return Err("EvVsKCurve::new: at least one (K, EV) sample required".into());
        }
        for p in &points {
            if p.k == 0 {
                return Err("EvVsKCurve::new: K must be >= 1".into());
            }
            if !p.ev.is_finite() {
                return Err(format!("EvVsKCurve::new: non-finite EV at K={}", p.k));
            }
        }
        points.sort_by_key(|p| p.k);
        for w in points.windows(2) {
            if w[0].k == w[1].k {
                return Err(format!("EvVsKCurve::new: duplicate K={}", w[0].k));
            }
        }
        Ok(Self { points })
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn points(&self) -> &[EvVsKPoint] {
        &self.points
    }

    /// Smallest `K` on the curve.
    pub fn k_min(&self) -> usize {
        self.points[0].k
    }

    /// Largest `K` on the curve.
    pub fn k_max(&self) -> usize {
        self.points[self.points.len() - 1].k
    }

    /// EV at the smallest `K` for which `EV(K) >= target`, if any. The curve is
    /// scanned in ascending `K` order, so this is the *cheapest* dictionary
    /// reaching the target. Returns `None` when the target is never reached.
    pub fn k_reaching(&self, target_ev: f64) -> Option<usize> {
        self.points.iter().find(|p| p.ev >= target_ev).map(|p| p.k)
    }
}

/// How the dictionary size is selected from the frontier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KSelectionMode {
    /// Kneedle-style: pick the `K` of maximum curvature of the min-max
    /// normalized curve (the saturation knee), accepting it only when the
    /// post-knee marginal slope has decayed below `knee_slope_fraction` of the
    /// initial (steep-regime) slope.
    Kneedle,
    /// Penalized-EV / MDL: maximize `EV(K) − complexity_penalty · (K / K_max)`.
    /// The `complexity_penalty` `γ` here is a TUNED dial — see
    /// [`KSelectionMode::MeasuredMdl`] for the derived, tuning-free replacement.
    PenalizedMdl,
    /// Measured MDL stopping rule (Theorem 4 of the "Superposed Geometry" memo):
    /// the tuning-free replacement for `PenalizedMdl`'s `γ`. Stop at the first
    /// `K` where the marginal EV gain per atom drops below the residual fraction
    /// times one atom's storage nats over the total coded-scalar count:
    /// ```text
    ///   ∂EV/∂K  <  (1 − EV(K)) · ( d_eff,atom · ln n_eff ) / ( N · k̄ · d̄ ).
    /// ```
    /// Every quantity on the right is MEASURED by the fit (supplied through
    /// [`KSelectionConfig::measured_coding`]); there is no free parameter. This
    /// is the RECOMMENDED path. It reproduces [`KSelectionFlag::NoKnee`] exactly:
    /// when the inequality never binds up to `K_max`, the largest `K` is
    /// returned, flagged `NoKnee`.
    MeasuredMdl,
}

impl KSelectionMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "kneedle" | "knee" | "elbow" => Ok(Self::Kneedle),
            "mdl" | "penalized" | "penalized_mdl" => Ok(Self::PenalizedMdl),
            "measured" | "measured_mdl" | "theorem4" | "theorem_4" => Ok(Self::MeasuredMdl),
            other => Err(format!(
                "K-selection mode must be 'kneedle', 'mdl', or 'measured'; got {other:?}"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Kneedle => "kneedle",
            Self::PenalizedMdl => "mdl",
            Self::MeasuredMdl => "measured",
        }
    }
}

/// The fit-measured coding ingredients that make the Theorem-4 stopping rule
/// ([`KSelectionMode::MeasuredMdl`]) tuning-free. Every field is read off the
/// fit at its operating point — nothing is tuned.
///
/// # Persistence is bits (Theorem D)
///
/// Adding an atom lowers the code term by the reconstruction it buys and raises
/// the dictionary term by exactly one atom's storage, `d_eff,atom · ln n_eff`
/// nats. That storage is `2·` the per-atom rank charge `½·d_eff·ln n_eff` the
/// WBIC / rank-charge accounting prices (the factor of two is the
/// storage-vs-evidence convention). One nat of log-persistence per active row
/// buys one nat of model evidence per unit codimension — persistence IS bits,
/// and the LOG-length is the unique parameterisation in which storage is
/// additive with the likelihood.
#[derive(Debug, Clone, Copy)]
pub struct MeasuredCoding {
    /// Effective dof of the marginal atom, `d_eff,atom = rank_eff · basis_edf`
    /// (the same product the rank charge prices).
    pub d_eff_atom: f64,
    /// Occupancy-corrected effective sample size `n_eff = Σ_row a²` — the SAME
    /// quantity the MP edge / rank charge use, NOT the global row count. Keeps
    /// the rule inert-row invariant.
    pub n_eff: f64,
    /// Number of coded rows `N`.
    pub n_rows: f64,
    /// Mean active atoms per row `k̄` (mean `L0` / occupancy).
    pub k_bar: f64,
    /// Mean coded scalars per active atom `d̄` (per-atom coordinate dimension).
    pub d_bar: f64,
}

impl MeasuredCoding {
    /// One atom's storage cost in nats, `d_eff,atom · ln n_eff` — the numerator
    /// of the Theorem-4 right-hand side. Twice the per-atom rank charge
    /// `½·d_eff·ln n_eff` (storage-vs-evidence convention).
    pub fn atom_storage_nats(&self) -> f64 {
        self.d_eff_atom * self.n_eff.max(1.0).ln()
    }

    /// Total count of coded scalars `N · k̄ · d̄` over which a marginal residual
    /// saving is amortised — the denominator of the Theorem-4 right-hand side.
    pub fn coded_scalar_count(&self) -> f64 {
        self.n_rows * self.k_bar * self.d_bar
    }

    /// The measured Theorem-4 right-hand side at explained variance `ev`:
    /// `(1 − ev) · atom_storage_nats / coded_scalar_count`. This is the smallest
    /// marginal EV gain per atom that still pays for the atom's storage.
    pub fn stop_threshold(&self, ev: f64) -> f64 {
        let denom = self.coded_scalar_count();
        if !(denom > 0.0) {
            return f64::INFINITY;
        }
        (1.0 - ev).max(0.0) * self.atom_storage_nats() / denom
    }
}

/// Tuning for [`select_k`].
#[derive(Debug, Clone, Copy)]
pub struct KSelectionConfig {
    pub mode: KSelectionMode,
    /// Kneedle: the post-knee marginal slope must fall below this fraction of
    /// the initial slope for a point to count as a saturation knee. A curve
    /// whose slope never decays this far is classified [`KSelectionFlag::NoKnee`]
    /// (still climbing) or [`KSelectionFlag::Linear`].
    pub knee_slope_fraction: f64,
    /// MDL: the complexity weight `γ` on the normalized size `K / K_max`.
    pub complexity_penalty: f64,
    /// Below this total EV span (`max EV − min EV` across the curve) the curve
    /// is treated as already saturated ([`KSelectionFlag::Flat`]) and the
    /// smallest `K` is returned.
    pub flat_span_tol: f64,
    /// The fit-measured coding ingredients for [`KSelectionMode::MeasuredMdl`].
    /// `None` on the `Kneedle` / `PenalizedMdl` paths (they do not need it). When
    /// the mode is `MeasuredMdl` but this is `None`, [`select_k`] falls back to
    /// `Kneedle` (nothing to measure with).
    pub measured_coding: Option<MeasuredCoding>,
}

impl Default for KSelectionConfig {
    fn default() -> Self {
        Self {
            mode: KSelectionMode::Kneedle,
            // Knee = where marginal gain has decayed to 10% of the steep slope.
            knee_slope_fraction: 0.10,
            complexity_penalty: 0.05,
            flat_span_tol: 1.0e-6,
            measured_coding: None,
        }
    }
}

impl KSelectionConfig {
    /// The recommended tuning-free config: the Theorem-4 measured MDL stopping
    /// rule fed the fit's own coding ingredients. Prefer this over a tuned
    /// `PenalizedMdl` `γ`.
    pub fn measured(coding: MeasuredCoding) -> Self {
        Self {
            mode: KSelectionMode::MeasuredMdl,
            measured_coding: Some(coding),
            ..Self::default()
        }
    }
}

/// Classification of the chosen `K` / the curve shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KSelectionFlag {
    /// A clear saturation knee was found; `K` is the knee.
    Knee,
    /// The curve is still climbing at `K_max`; returned `K = K_max`.
    NoKnee,
    /// EV grows ~linearly in `K` (no curvature); returned `K = K_max`.
    Linear,
    /// EV is already saturated at the smallest `K`; returned `K = K_min`.
    Flat,
}

impl KSelectionFlag {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Knee => "knee",
            Self::NoKnee => "no_knee",
            Self::Linear => "linear",
            Self::Flat => "flat",
        }
    }

    /// Whether the returned `K` is a genuine saturation knee (vs a fallback to
    /// an endpoint because no knee exists).
    pub const fn is_knee(self) -> bool {
        matches!(self, Self::Knee)
    }
}

/// Result of [`select_k`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KSelection {
    /// The selected dictionary size.
    pub k: usize,
    /// The explained variance at the selected `K`.
    pub ev: f64,
    /// Curve-shape classification of the selection.
    pub flag: KSelectionFlag,
    /// The strength score driving the choice: for Kneedle this is the post-knee
    /// slope-decay fraction (smaller = sharper knee); for MDL it is the
    /// penalized objective value at the selected `K`.
    pub score: f64,
}

/// Select the dictionary size `K` at the saturation knee of an EV-vs-`K`
/// frontier.
///
/// The discrete analogue of the topology race in
/// [`gam_solve::structure_search`]: a knee/MDL criterion over a discrete
/// structure parameter, *not* a REML smoothing choice (see module docs).
///
/// On a curve with no usable knee the largest (or, when already saturated,
/// smallest) `K` is returned with the corresponding [`KSelectionFlag`] so the
/// caller can decide whether to widen the sweep rather than trusting a spurious
/// elbow.
pub fn select_k(curve: &EvVsKCurve, config: &KSelectionConfig) -> KSelection {
    let pts = curve.points();
    let n = pts.len();

    // Single point: nothing to choose.
    if n == 1 {
        return KSelection {
            k: pts[0].k,
            ev: pts[0].ev,
            flag: KSelectionFlag::Flat,
            score: 0.0,
        };
    }

    // Already-saturated curve (EV barely moves): smallest K wins.
    let ev_min = pts.iter().map(|p| p.ev).fold(f64::INFINITY, f64::min);
    let ev_max = pts.iter().map(|p| p.ev).fold(f64::NEG_INFINITY, f64::max);
    let span = ev_max - ev_min;
    if span <= config.flat_span_tol {
        return KSelection {
            k: pts[0].k,
            ev: pts[0].ev,
            flag: KSelectionFlag::Flat,
            score: span,
        };
    }

    match config.mode {
        KSelectionMode::Kneedle => select_kneedle(curve, config, span),
        KSelectionMode::PenalizedMdl => select_mdl(curve, config),
        KSelectionMode::MeasuredMdl => match config.measured_coding {
            Some(coding) => select_measured(curve, &coding),
            // No coding ingredients to measure with: fall back to the knee.
            None => select_kneedle(curve, config, span),
        },
    }
}

/// The Theorem-4 measured MDL stopping rule ([`KSelectionMode::MeasuredMdl`]).
///
/// Walks the frontier and stops at the first `K` whose marginal EV gain PER ATOM
/// (`(EV_i − EV_{i−1}) / (K_i − K_{i−1})`) falls below the measured
/// [`MeasuredCoding::stop_threshold`] at that `K`. The selected `K` is the last
/// atom that still paid for itself — the sample before the first binding one. If
/// no atom fails the test, the largest `K` is returned flagged
/// [`KSelectionFlag::NoKnee`], reproducing the tuned-path NoKnee behaviour
/// exactly. The `score` carries the marginal-vs-threshold ratio at the stop
/// (`< 1` when the rule bound, `+∞` for the NoKnee case).
///
/// This is MODEL-FREE: it compares the MEASURED marginal against the MEASURED
/// right-hand side at each `K`, assuming no functional form for `EV(K)`.
fn select_measured(curve: &EvVsKCurve, coding: &MeasuredCoding) -> KSelection {
    let pts = curve.points();
    let n = pts.len();
    for i in 1..n {
        let dk = (pts[i].k - pts[i - 1].k) as f64;
        let marginal = (pts[i].ev - pts[i - 1].ev) / dk.max(MIN_DENOM);
        let threshold = coding.stop_threshold(pts[i].ev);
        if marginal < threshold {
            // Atom `i` failed to pay for its storage: stop growing. The accepted
            // dictionary is everything strictly before it.
            let ratio = if threshold > 0.0 {
                marginal / threshold
            } else {
                f64::INFINITY
            };
            return KSelection {
                k: pts[i - 1].k,
                ev: pts[i - 1].ev,
                flag: KSelectionFlag::Knee,
                score: ratio,
            };
        }
    }
    // Never bound: the residual saving still outpays storage at K_max.
    KSelection {
        k: curve.k_max(),
        ev: pts[n - 1].ev,
        flag: KSelectionFlag::NoKnee,
        score: f64::INFINITY,
    }
}

/// Per-segment marginal EV gain *per added atom*: `(EV_{i+1} − EV_i) / (K_{i+1} − K_i)`.
fn marginal_slopes(pts: &[EvVsKPoint]) -> Vec<f64> {
    pts.windows(2)
        .map(|w| {
            let dk = (w[1].k - w[0].k) as f64;
            (w[1].ev - w[0].ev) / dk
        })
        .collect()
}

fn select_kneedle(curve: &EvVsKCurve, config: &KSelectionConfig, span: f64) -> KSelection {
    let pts = curve.points();
    let n = pts.len();
    let slopes = marginal_slopes(pts);

    // Initial (steep-regime) slope: the first positive segment slope. If the
    // curve never rises, treat it as flat (handled by the span gate upstream,
    // but guard anyway).
    let init_slope = slopes.iter().copied().find(|s| *s > 0.0).unwrap_or(0.0);
    if init_slope <= 0.0 {
        return KSelection {
            k: pts[0].k,
            ev: pts[0].ev,
            flag: KSelectionFlag::Flat,
            score: 0.0,
        };
    }

    // Linearity test: a straight EV-vs-K line has near-constant marginal slope.
    // Compare the slope range to the mean slope; a small relative spread means
    // there is no curvature, hence no knee.
    let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
    let slope_lo = slopes.iter().copied().fold(f64::INFINITY, f64::min);
    let slope_hi = slopes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let slope_spread = slope_hi - slope_lo;
    if mean_slope > 0.0 && slope_spread <= LINEARITY_SLOPE_REL_TOL * mean_slope {
        return KSelection {
            k: curve.k_max(),
            ev: pts[n - 1].ev,
            flag: KSelectionFlag::Linear,
            score: slope_spread / mean_slope.max(MIN_DENOM),
        };
    }

    // Kneedle: on the min-max normalized curve, the knee is the point of
    // greatest drop of the curve below the chord from first to last point.
    // Equivalently the point maximizing the normalized-EV minus normalized-K
    // difference d_i = ev_hat_i − k_hat_i. We locate that candidate, then
    // accept it only if the *post-knee* marginal slope has decayed below
    // `knee_slope_fraction` of the initial slope (the saturation test).
    let k_first = pts[0].k as f64;
    let k_last = pts[n - 1].k as f64;
    let k_range = (k_last - k_first).max(MIN_DENOM);

    let mut best_idx = 0usize;
    let mut best_diff = f64::NEG_INFINITY;
    for (i, p) in pts.iter().enumerate() {
        let ev_hat = (p.ev - pts[0].ev) / span;
        let k_hat = (p.k as f64 - k_first) / k_range;
        let diff = ev_hat - k_hat;
        if diff > best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }

    // Saturation acceptance: the marginal slope of the segment *after* the
    // candidate knee must have decayed below the fraction of the initial slope.
    // `best_idx` indexes a point; the post-knee slope is segment `best_idx`
    // (between best_idx and best_idx+1) when it exists.
    let post_slope = if best_idx < slopes.len() {
        slopes[best_idx]
    } else {
        // Knee at the very last point => everything saturated up to here.
        0.0
    };
    let decay_fraction = (post_slope / init_slope).max(0.0);

    if decay_fraction <= config.knee_slope_fraction {
        KSelection {
            k: pts[best_idx].k,
            ev: pts[best_idx].ev,
            flag: KSelectionFlag::Knee,
            score: decay_fraction,
        }
    } else {
        // The "knee" candidate is still on a steep stretch: the curve has not
        // saturated within the sampled range. Return the largest K, flagged.
        KSelection {
            k: curve.k_max(),
            ev: pts[n - 1].ev,
            flag: KSelectionFlag::NoKnee,
            score: decay_fraction,
        }
    }
}

fn select_mdl(curve: &EvVsKCurve, config: &KSelectionConfig) -> KSelection {
    let pts = curve.points();
    let k_max = curve.k_max() as f64;
    let gamma = config.complexity_penalty;

    let mut best_idx = 0usize;
    let mut best_obj = f64::NEG_INFINITY;
    for (i, p) in pts.iter().enumerate() {
        let obj = p.ev - gamma * (p.k as f64 / k_max.max(MIN_DENOM));
        if obj > best_obj {
            best_obj = obj;
            best_idx = i;
        }
    }

    // Classify the MDL pick the same way the Kneedle path reports its endpoints
    // so callers get a consistent flag vocabulary: interior pick => Knee,
    // endpoint picks => the endpoint reason.
    let flag = if best_idx == 0 {
        KSelectionFlag::Flat
    } else if best_idx == pts.len() - 1 {
        KSelectionFlag::NoKnee
    } else {
        KSelectionFlag::Knee
    };

    KSelection {
        k: pts[best_idx].k,
        ev: pts[best_idx].ev,
        flag,
        score: best_obj,
    }
}

/// The manifold-vs-linear parameter-efficiency advantage at a target EV.
///
/// Manifold reaches `target_ev` at `k_manifold` atoms, linear at `k_linear`
/// atoms. Compression is measured in DECODER PARAMETERS, not atom count: a
/// manifold atom stores more scalars than a linear atom (a `d`-chart with an
/// `M`-wide basis over `p` output channels costs `M·p` decoder scalars, a linear
/// atom `p`), so `k_linear / k_manifold` would over-credit the manifold — one
/// manifold atom worth ten linear atoms in EV can cost ten linear atoms' worth
/// of parameters and buy no compression at all. The honest ratio is
/// `linear_params / manifold_params`, and it exceeds 1 only when the manifold
/// truly reaches the target with fewer stored scalars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldVsLinearAdvantage {
    /// Target explained variance both representations are compared at.
    pub target_ev: f64,
    /// Smallest manifold `K` reaching `target_ev`, if any.
    pub k_manifold: Option<usize>,
    /// Smallest linear `K` reaching `target_ev`, if any.
    pub k_linear: Option<usize>,
    /// Total decoder parameters the manifold spends to reach `target_ev`
    /// (`k_manifold · manifold_params_per_atom`), if reached.
    pub manifold_params: Option<f64>,
    /// Total decoder parameters the linear baseline spends to reach `target_ev`
    /// (`k_linear · linear_params_per_atom`), if reached.
    pub linear_params: Option<f64>,
    /// `linear_params / manifold_params` when both reach the target, else `None`
    /// — the parameter-efficiency ratio (`> 1` ⇒ the manifold reaches the target
    /// with fewer stored scalars).
    pub compression_ratio: Option<f64>,
}

impl ManifoldVsLinearAdvantage {
    /// True iff the manifold representation reaches the target EV with strictly
    /// FEWER decoder parameters than the linear baseline (`manifold_params <
    /// linear_params`) — parameter efficiency, not atom-count efficiency.
    pub fn manifold_dominates(&self) -> bool {
        match (self.manifold_params, self.linear_params) {
            (Some(pm), Some(pl)) => pm < pl,
            _ => false,
        }
    }
}

/// Compute the manifold-vs-linear advantage at `target_ev`: the cheapest `K`
/// each representation needs to reach the target and their PARAMETER ratio.
///
/// `manifold_params_per_atom` / `linear_params_per_atom` are the decoder scalar
/// counts a single atom of each representation stores (manifold: `M·p` for an
/// `M`-wide basis over `p` output channels; linear: `p`). The compression ratio
/// is `linear_params / manifold_params`, so a manifold atom that reaches the
/// target with a large basis is charged its true parameter cost instead of being
/// credited one atom's worth.
pub fn manifold_vs_linear_advantage(
    manifold: &EvVsKCurve,
    linear: &EvVsKCurve,
    target_ev: f64,
    manifold_params_per_atom: f64,
    linear_params_per_atom: f64,
) -> ManifoldVsLinearAdvantage {
    let k_manifold = manifold.k_reaching(target_ev);
    let k_linear = linear.k_reaching(target_ev);
    let manifold_params = k_manifold.map(|k| k as f64 * manifold_params_per_atom);
    let linear_params = k_linear.map(|k| k as f64 * linear_params_per_atom);
    let compression_ratio = match (manifold_params, linear_params) {
        (Some(pm), Some(pl)) if pm > 0.0 => Some(pl / pm),
        _ => None,
    };
    ManifoldVsLinearAdvantage {
        target_ev,
        k_manifold,
        k_linear,
        manifold_params,
        linear_params,
        compression_ratio,
    }
}

/// The full auto-`K` recommendation the OLMo research battery / hillclimb
/// driver consumes: the knee-selected dictionary size on the manifold frontier,
/// plus the manifold-vs-linear advantage measured at the EV the auto-`K` choice
/// reaches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoKRecommendation {
    /// Knee-selected dictionary size on the manifold frontier.
    pub selection: KSelection,
    /// Manifold-vs-linear advantage at the auto-`K` operating EV.
    pub advantage: ManifoldVsLinearAdvantage,
}

/// One-call auto-`K`: knee-select `K` on the manifold EV-vs-`K` frontier, then
/// report how many linear atoms would be needed to match the EV the selected
/// `K` achieves.
///
/// This is the intended battery entry point: feed it the manifold and linear
/// EV-vs-`K` sweeps measured on real OLMo L25 activations and the returned
/// [`AutoKRecommendation::selection`] is the engine's auto-`K`, ready to be
/// checked against the human-chosen `K`.
pub fn recommend_auto_k(
    manifold: &EvVsKCurve,
    linear: &EvVsKCurve,
    config: &KSelectionConfig,
    manifold_params_per_atom: f64,
    linear_params_per_atom: f64,
) -> AutoKRecommendation {
    let selection = select_k(manifold, config);
    let advantage = manifold_vs_linear_advantage(
        manifold,
        linear,
        selection.ev,
        manifold_params_per_atom,
        linear_params_per_atom,
    );
    AutoKRecommendation {
        selection,
        advantage,
    }
}

/// Build an [`EvVsKCurve`] from explicit `(K, EV)` pairs, e.g. the columns the
/// OLMo research battery emits per sweep. Convenience wrapper over
/// [`EvVsKCurve::new`].
pub fn curve_from_pairs(pairs: &[(usize, f64)]) -> Result<EvVsKCurve, String> {
    EvVsKCurve::new(
        pairs
            .iter()
            .map(|&(k, ev)| EvVsKPoint::new(k, ev))
            .collect(),
    )
}

/// Explained variance (R²) of `fitted` against `x`, total-SS normalized and
/// column-mean centered. Shared definition with the linear-dictionary lane so
/// the manifold and linear EV-vs-`K` curves are on the same scale.
pub fn explained_variance(x: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    assert_eq!(
        x.dim(),
        fitted.dim(),
        "explained_variance: x {:?} != fitted {:?}",
        x.dim(),
        fitted.dim()
    );
    let n = x.nrows();
    if n == 0 || x.ncols() == 0 {
        return 0.0;
    }
    let mut rss = 0.0;
    for row in 0..n {
        for col in 0..x.ncols() {
            let r = x[[row, col]] - fitted[[row, col]];
            rss += r * r;
        }
    }
    let means = x
        .mean_axis(ndarray::Axis(0))
        .expect("non-empty input has means");
    let mut tss = 0.0;
    for row in 0..n {
        for col in 0..x.ncols() {
            let c = x[[row, col]] - means[col];
            tss += c * c;
        }
    }
    if tss <= MIN_DENOM {
        if rss <= MIN_DENOM { 1.0 } else { 0.0 }
    } else {
        1.0 - rss / tss
    }
}

/// Relative slope spread below which an EV-vs-K curve is deemed straight
/// (no curvature => no knee).
const LINEARITY_SLOPE_REL_TOL: f64 = 0.05;

/// Floor for denominators that could otherwise be zero.
const MIN_DENOM: f64 = 1.0e-12;

#[cfg(test)]
mod k_selection_tests {
    use super::*;
    use ndarray::array;

    fn knee_curve() -> EvVsKCurve {
        // Steep rise to K=4 (EV ~0.9), then near-flat saturation. Knee at K=4.
        curve_from_pairs(&[
            (1, 0.40),
            (2, 0.65),
            (3, 0.82),
            (4, 0.90),
            (8, 0.915),
            (16, 0.92),
            (32, 0.922),
        ])
        .expect("knee curve")
    }

    #[test]
    fn kneedle_picks_the_elbow() {
        let curve = knee_curve();
        let sel = select_k(&curve, &KSelectionConfig::default());
        assert_eq!(sel.flag, KSelectionFlag::Knee);
        assert_eq!(sel.k, 4, "knee should sit at the saturation corner K=4");
        assert!((sel.ev - 0.90).abs() < 1e-9);
    }

    #[test]
    fn linear_curve_returns_full_k_with_flag() {
        // EV grows exactly linearly in K: no curvature, no knee.
        let curve = curve_from_pairs(&[(1, 0.10), (2, 0.20), (3, 0.30), (4, 0.40), (5, 0.50)])
            .expect("linear curve");
        let sel = select_k(&curve, &KSelectionConfig::default());
        assert_eq!(sel.flag, KSelectionFlag::Linear);
        assert_eq!(sel.k, 5, "linear curve returns the largest K");
    }

    #[test]
    fn still_climbing_curve_returns_full_k_no_knee() {
        // Concave but still steeply rising at K_max: not yet saturated.
        let curve = curve_from_pairs(&[(1, 0.10), (2, 0.30), (3, 0.48), (4, 0.64), (5, 0.78)])
            .expect("climbing curve");
        let cfg = KSelectionConfig {
            // demand a sharp saturation (1% of initial slope) it cannot meet
            knee_slope_fraction: 0.01,
            ..KSelectionConfig::default()
        };
        let sel = select_k(&curve, &cfg);
        assert_eq!(sel.flag, KSelectionFlag::NoKnee);
        assert_eq!(sel.k, 5);
    }

    #[test]
    fn flat_curve_returns_smallest_k() {
        let curve =
            curve_from_pairs(&[(1, 0.900), (2, 0.9000001), (4, 0.9000002)]).expect("flat curve");
        let sel = select_k(&curve, &KSelectionConfig::default());
        assert_eq!(sel.flag, KSelectionFlag::Flat);
        assert_eq!(sel.k, 1, "already-saturated curve returns smallest K");
    }

    #[test]
    fn single_point_curve_is_flat() {
        let curve = curve_from_pairs(&[(7, 0.5)]).expect("single point");
        let sel = select_k(&curve, &KSelectionConfig::default());
        assert_eq!(sel.flag, KSelectionFlag::Flat);
        assert_eq!(sel.k, 7);
    }

    #[test]
    fn mdl_picks_interior_knee_on_saturating_curve() {
        let curve = knee_curve();
        let cfg = KSelectionConfig {
            mode: KSelectionMode::PenalizedMdl,
            complexity_penalty: 0.05,
            ..KSelectionConfig::default()
        };
        let sel = select_k(&curve, &cfg);
        // MDL trades EV gain against K/K_max=K/32. Past the knee the EV gain is
        // tiny while the size penalty keeps growing, so the optimum is interior.
        assert!(
            sel.k <= 8,
            "MDL should not chase the saturated tail, got {}",
            sel.k
        );
        assert!(
            sel.k >= 3,
            "MDL should not under-fit the steep rise, got {}",
            sel.k
        );
    }

    #[test]
    fn mdl_penalty_zero_takes_full_k() {
        let curve = knee_curve();
        let cfg = KSelectionConfig {
            mode: KSelectionMode::PenalizedMdl,
            complexity_penalty: 0.0,
            ..KSelectionConfig::default()
        };
        let sel = select_k(&curve, &cfg);
        // With no complexity penalty, max EV (largest K) wins.
        assert_eq!(sel.k, 32);
    }

    #[test]
    fn advantage_metric_rewards_manifold_compression() {
        // Manifold reaches EV 0.90 at K=4; linear needs K=16 for the same. With
        // equal per-atom parameter cost the parameter ratio is 16·p / 4·p = 4x.
        let manifold = curve_from_pairs(&[(1, 0.40), (2, 0.65), (4, 0.90), (8, 0.93), (16, 0.94)])
            .expect("manifold curve");
        let linear = curve_from_pairs(&[(1, 0.20), (2, 0.35), (4, 0.55), (8, 0.78), (16, 0.91)])
            .expect("linear curve");
        let adv = manifold_vs_linear_advantage(&manifold, &linear, 0.90, 10.0, 10.0);
        assert_eq!(adv.k_manifold, Some(4));
        assert_eq!(adv.k_linear, Some(16));
        assert_eq!(adv.manifold_params, Some(40.0));
        assert_eq!(adv.linear_params, Some(160.0));
        assert!(adv.manifold_dominates());
        let ratio = adv.compression_ratio.expect("both reach target");
        assert!(
            (ratio - 4.0).abs() < 1e-12,
            "expected 160/40 = 4x parameter compression, got {ratio}"
        );
    }

    #[test]
    fn advantage_metric_counts_parameters_not_atoms() {
        // The reviewer's counterexample: one heavy manifold atom (100 decoder
        // scalars) reaching the target vs many light linear atoms (1 scalar each).
        // An atom-count ratio would report a spurious win; the parameter ratio
        // shows the manifold spends MORE scalars and does NOT dominate.
        let manifold = curve_from_pairs(&[(1, 0.90), (2, 0.95)]).expect("manifold curve");
        let linear =
            curve_from_pairs(&[(2, 0.30), (5, 0.60), (10, 0.90)]).expect("linear curve");
        let adv = manifold_vs_linear_advantage(&manifold, &linear, 0.90, 100.0, 1.0);
        assert_eq!(adv.k_manifold, Some(1));
        assert_eq!(adv.k_linear, Some(10));
        assert_eq!(adv.manifold_params, Some(100.0));
        assert_eq!(adv.linear_params, Some(10.0));
        // 1 atom < 10 atoms, but 100 params > 10 params: no parameter compression.
        assert!(!adv.manifold_dominates());
        let ratio = adv.compression_ratio.expect("both reach target");
        assert!((ratio - 0.1).abs() < 1e-12, "10/100 = 0.1x, got {ratio}");
    }

    #[test]
    fn advantage_metric_handles_unreached_target() {
        let manifold = curve_from_pairs(&[(1, 0.40), (2, 0.55)]).expect("manifold curve");
        let linear = curve_from_pairs(&[(1, 0.20), (2, 0.35)]).expect("linear curve");
        let adv = manifold_vs_linear_advantage(&manifold, &linear, 0.90, 10.0, 10.0);
        assert_eq!(adv.k_manifold, None);
        assert_eq!(adv.k_linear, None);
        assert!(adv.manifold_params.is_none());
        assert!(adv.linear_params.is_none());
        assert!(adv.compression_ratio.is_none());
        assert!(!adv.manifold_dominates());
    }

    #[test]
    fn recommend_auto_k_combines_knee_and_advantage() {
        // Manifold knees at K=4 (EV 0.90); linear needs K=16 for EV 0.90.
        let manifold = curve_from_pairs(&[
            (1, 0.40),
            (2, 0.65),
            (3, 0.82),
            (4, 0.90),
            (8, 0.915),
            (16, 0.92),
            (32, 0.922),
        ])
        .expect("manifold curve");
        let linear = curve_from_pairs(&[
            (1, 0.20),
            (2, 0.35),
            (4, 0.55),
            (8, 0.78),
            (16, 0.91),
            (32, 0.93),
        ])
        .expect("linear curve");
        let rec = recommend_auto_k(&manifold, &linear, &KSelectionConfig::default(), 10.0, 10.0);
        assert_eq!(rec.selection.k, 4);
        assert_eq!(rec.selection.flag, KSelectionFlag::Knee);
        // At the auto-K EV (0.90) linear needs K=16; equal per-atom parameter
        // cost => 160/40 = 4x parameter compression.
        assert_eq!(rec.advantage.k_manifold, Some(4));
        assert_eq!(rec.advantage.k_linear, Some(16));
        assert!(rec.advantage.manifold_dominates());
        let ratio = rec.advantage.compression_ratio.expect("both reach EV");
        assert!((ratio - 4.0).abs() < 1e-12);
    }

    #[test]
    fn explained_variance_matches_perfect_and_mean_baselines() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        // Perfect reconstruction => EV 1.
        let ev_perfect = explained_variance(x.view(), x.view());
        assert!((ev_perfect - 1.0).abs() < 1e-12);
        // Mean-only reconstruction => EV 0.
        let means = x.mean_axis(ndarray::Axis(0)).expect("means");
        let mean_fit = array![
            [means[0], means[1]],
            [means[0], means[1]],
            [means[0], means[1]]
        ];
        let ev_mean = explained_variance(x.view(), mean_fit.view());
        assert!(
            ev_mean.abs() < 1e-12,
            "mean baseline EV should be 0, got {ev_mean}"
        );
    }

    #[test]
    fn curve_rejects_bad_input() {
        assert!(EvVsKCurve::new(vec![]).is_err());
        assert!(curve_from_pairs(&[(0, 0.5)]).is_err());
        assert!(curve_from_pairs(&[(1, f64::NAN)]).is_err());
        assert!(curve_from_pairs(&[(2, 0.5), (2, 0.6)]).is_err());
    }

    #[test]
    fn curve_sorts_by_k() {
        let curve = curve_from_pairs(&[(8, 0.9), (1, 0.4), (4, 0.8)]).expect("curve");
        assert_eq!(curve.k_min(), 1);
        assert_eq!(curve.k_max(), 8);
        assert_eq!(curve.points()[0].k, 1);
        assert_eq!(curve.points()[2].k, 8);
    }

    #[test]
    fn mode_parse_roundtrips() {
        assert_eq!(
            KSelectionMode::parse("elbow").expect("parse"),
            KSelectionMode::Kneedle
        );
        assert_eq!(
            KSelectionMode::parse("MDL").expect("parse"),
            KSelectionMode::PenalizedMdl
        );
        assert_eq!(
            KSelectionMode::parse("measured").expect("parse"),
            KSelectionMode::MeasuredMdl
        );
        assert_eq!(
            KSelectionMode::parse("theorem4").expect("parse"),
            KSelectionMode::MeasuredMdl
        );
        assert_eq!(KSelectionMode::Kneedle.as_str(), "kneedle");
        assert_eq!(KSelectionMode::MeasuredMdl.as_str(), "measured");
        assert!(KSelectionMode::parse("nonsense").is_err());
    }

    /// Build a saturating EV curve `EV(K) = K/(K+τ)` with constant coding
    /// ingredients, so the Theorem-4 right-hand side is a constant `c` and the
    /// stopping `K` has a closed form we can check against.
    ///
    /// For `EV(K)=K/(K+τ)`: residual `1−EV = τ/(K+τ)`, marginal
    /// `EV(K)−EV(K−1) = τ/((K+τ)(K−1+τ))`, ratio marginal/residual `= 1/(K−1+τ)`.
    /// The stop condition marginal `< (1−EV)·c` is `1/(K−1+τ) < c`, i.e.
    /// `K > 1/c − τ + 1`. So the FIRST binding K is `⌈1/c − τ + 1⌉` (strict) and
    /// the selected K (last atom that paid) is that minus one.
    fn saturating_coding(c: f64) -> (Vec<EvVsKPoint>, MeasuredCoding) {
        // Set atom_storage_nats / coded_scalar_count == c. Pick n_eff so ln = 2,
        // d_eff_atom = 1 → storage = 2; then coded = 2/c.
        let d_eff_atom = 1.0;
        let n_eff = std::f64::consts::E.powf(2.0);
        let storage = d_eff_atom * n_eff.ln();
        let coding = MeasuredCoding {
            d_eff_atom,
            n_eff,
            n_rows: storage / c,
            k_bar: 1.0,
            d_bar: 1.0,
        };
        (Vec::new(), coding)
    }

    fn saturating_curve(tau: f64, k_max: usize) -> EvVsKCurve {
        curve_from_pairs(
            &(1..=k_max)
                .map(|k| {
                    let kf = k as f64;
                    (k, kf / (kf + tau))
                })
                .collect::<Vec<_>>(),
        )
        .expect("saturating curve")
    }

    #[test]
    fn measured_rule_stops_at_theory_predicted_k() {
        // 1/c = 15.5, τ = 5 → K > 15.5 − 5 + 1 = 11.5 → first binding K = 12,
        // selected K = 11.
        let (_p, coding) = saturating_coding(1.0 / 15.5);
        let curve = saturating_curve(5.0, 40);
        let sel = select_k(&curve, &KSelectionConfig::measured(coding));
        assert_eq!(sel.flag, KSelectionFlag::Knee, "measured rule should bind");
        assert_eq!(sel.k, 11, "selected K = last atom that paid for itself");
        assert!(sel.score < 1.0, "marginal below threshold at the stop");
    }

    #[test]
    fn measured_rule_returns_k_max_when_never_binding() {
        // Tiny c → threshold ≈ 0 → every atom always pays → NoKnee at K_max.
        let (_p, coding) = saturating_coding(1e-9);
        let curve = saturating_curve(5.0, 25);
        let sel = select_k(&curve, &KSelectionConfig::measured(coding));
        assert_eq!(sel.flag, KSelectionFlag::NoKnee, "must reproduce NoKnee");
        assert_eq!(sel.k, 25, "NoKnee returns K_max");
    }

    #[test]
    fn measured_rule_stops_earlier_when_storage_expensive() {
        let curve = saturating_curve(5.0, 40);
        let (_pe, expensive) = saturating_coding(1.0 / 6.5); // K > 2.5 → selected 2
        let (_pc, cheap) = saturating_coding(1.0 / 15.5); // selected 11
        let k_exp = select_k(&curve, &KSelectionConfig::measured(expensive)).k;
        let k_cheap = select_k(&curve, &KSelectionConfig::measured(cheap)).k;
        assert_eq!(k_exp, 2);
        assert!(
            k_exp < k_cheap,
            "expensive storage must stop earlier: {k_exp} vs {k_cheap}"
        );
    }

    #[test]
    fn measured_mode_without_coding_falls_back_to_knee() {
        let curve = knee_curve();
        let cfg = KSelectionConfig {
            mode: KSelectionMode::MeasuredMdl,
            measured_coding: None,
            ..KSelectionConfig::default()
        };
        let sel = select_k(&curve, &cfg);
        // Falls back to Kneedle, which knees this curve at K=4.
        assert_eq!(sel.k, 4);
        assert_eq!(sel.flag, KSelectionFlag::Knee);
    }

    #[test]
    fn stop_threshold_matches_hand_computation() {
        let coding = MeasuredCoding {
            d_eff_atom: 2.0,
            n_eff: std::f64::consts::E, // ln = 1
            n_rows: 100.0,
            k_bar: 2.0,
            d_bar: 5.0,
        };
        // storage = 2·1 = 2; coded = 100·2·5 = 1000; residual at ev=0.75 = 0.25.
        let expected = 0.25 * 2.0 / 1000.0;
        assert!((coding.stop_threshold(0.75) - expected).abs() < 1e-15);
    }
}
