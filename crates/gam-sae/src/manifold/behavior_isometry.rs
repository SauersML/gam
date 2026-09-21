//! #2015 — the per-atom **representation–behavior isometry defect**: the reported
//! statistic that turns the steering paper's isometry *assumption* into a
//! *measured* per-atom quantity of the two-block fit.
//!
//! # What it measures
//!
//! A Rung-2 two-block atom decodes ONE shared latent coordinate `t` into both an
//! activation image `x(t) = Φ_k(t) B_k` and a nats-unit behavior image
//! `y(t) = Φ_k(t) C_k` (the `√λ_y` un-done, [`BehaviorBlock::split_decoder`]). The
//! steering paper's headline claim is that the representation manifold and the
//! behavior manifold are **scaled-isometric** on natural data: moving along the
//! shared coordinate changes activation and behavior in lock-step, up to one
//! global scale. In the induced 1-D geometry that is exactly the statement
//!
//! ```text
//!   s_x(t) / s_y(t) = const across t,     s_x = ‖dx/dt‖,  s_y = ‖dy/dt‖ .
//! ```
//!
//! The two induced metrics are proportional iff their speed ratio is constant.
//! This module reports, per atom, the **speed ratio** `r(t) = s_x(t)/s_y(t)`:
//! its (support-weighted) mean is the isometry **scale**, and its coefficient of
//! variation is the isometry **defect** — `0` ⟺ an exact scaled isometry, large
//! ⟺ the correspondence between representation and behavior bends along the atom.
//! An atom with a high defect is *flagged*, not silently distorted (issue #2015,
//! point 1).
//!
//! # Why the defect is gauge-invariant (no chart canonicalization needed)
//!
//! Both speeds are taken with respect to the *same* latent `t`. Under any
//! reparameterization `t ↦ u(t)` both `s_x` and `s_y` are multiplied by the same
//! Jacobian `|dt/du|`, so their **ratio is invariant**. The defect therefore does
//! not depend on the residual `Diff(S¹)` / `Diff([0,1])` chart gauge the fit
//! happened to land in — unlike the within-chart arc-length defect
//! ([`crate::manifold::atom_coordinate_fidelity`]), which measures a property of
//! the parameterization itself. The two are complementary: arc-length defect asks
//! "is the chart honest?"; isometry defect asks "does the honest chart carry the
//! SAME geometry in activation and in behavior?".
//!
//! # Calibration readout
//!
//! `s_y(t)²` is the behavioral dose in nats per unit `t²`
//! ([`SphereTangentEmbedding::predicted_nats`]): a latent step `Δt` costs
//! `s_y(t)²·Δt²` nats. Its mean is reported as [`AtomBehaviorIsometry::nats_per_unit_t`]
//! — the raw calibration quantity the unit-speed re-gauge (issue #2015, point 3;
//! the `= 2` kill test on #1942) drives to a constant. This module reports it;
//! pinning it to `2` is the follow-up arc-length re-gauge of the behavior block.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::chart_canonicalization::curve_speeds;

use super::{SaeBasisEvaluator, SaeManifoldTerm, SupportMeasure};

/// The per-atom representation–behavior isometry certificate of a fitted Rung-2
/// two-block atom. Produced by [`atom_behavior_isometry`]; `None` for atoms
/// without a `d = 1` chart or when no behavior block is installed.
#[derive(Debug, Clone)]
pub struct AtomBehaviorIsometry {
    /// The atom this certificate is for.
    pub atom_idx: usize,
    /// Number of positive-support rows the speeds were read at.
    pub n_rows: usize,
    /// Total occupancy mass `Σ_i w_i` from the shared atom support measure.
    pub support_mass: f64,
    /// `true` iff the behavior image actually moves along the latent (some
    /// supported row has a resolved behavior speed). A behaviorally inert atom (`C_k ≈ 0`,
    /// constant behavior) has `false` here and a `NaN` defect/scale — there is no
    /// correspondence to certify, reported honestly rather than as a defect.
    pub behavior_engaged: bool,
    /// Number of supported rows where the behavioral metric collapses
    /// (`s_y` is numerically zero). A scaled isometry requires both induced
    /// metrics to be non-degenerate on the same support, so any positive count
    /// invalidates the isometry claim instead of silently dropping those rows.
    pub behavior_metric_collapse_rows: usize,
    /// Support-weighted RMS of the activation induced speed `s_x = ‖dx/dt‖`.
    pub activation_speed_rms: f64,
    /// Support-weighted RMS of the behavior induced speed `s_y = √(ẏᵀ G(y) ẏ)`, in
    /// the behavior chart's Fisher metric.
    pub behavior_speed_rms: f64,
    /// The isometry **scale**: support-weighted mean of `r = s_x/s_y` over the
    /// rows where behavior moves. Activation length per unit behavior length.
    /// `NaN` when `behavior_engaged == false` or the behavioral metric collapses
    /// on any supported row.
    pub scale: f64,
    /// The isometry **defect**: support-weighted coefficient of variation of
    /// `r = s_x/s_y`, `std(r)/mean(r)`. `0` ⟺ an exact scaled isometry; grows as
    /// the representation–behavior correspondence bends along the atom.
    /// `+∞` when the behavioral metric collapses anywhere on support while the
    /// atom is otherwise engaged; `NaN` only when behavior is globally inert.
    pub defect_cv: f64,
    /// `min r / scale` over the rows — how much slower (relative to its mean) the
    /// activation moves per unit behavior at its most compressed row. `NaN` when
    /// not engaged.
    pub min_ratio_over_scale: f64,
    /// `max r / scale` over the rows — the most stretched row. `NaN` when not
    /// engaged.
    pub max_ratio_over_scale: f64,
    /// Support-weighted mean of `s_y²` — the behavioral dose in **nats per unit
    /// `t²`** ([`super::SphereTangentEmbedding::predicted_nats`]). The unit-speed
    /// re-gauge drives this to a constant (`≈ 2` in the calibrated gauge); this is
    /// the raw readout. `NaN` when not engaged.
    pub nats_per_unit_t: f64,
    /// Behavior-pinned canonical chart coordinate.  This is the arc length of
    /// the fitted behavior image, divided by `sqrt(2)`, with its origin pinned
    /// at the point closest to the behavior embedding's Frechet basepoint and
    /// its orientation pinned by the first non-zero tangent component there.
    /// Consequently a displacement `du` has the universal local calibration
    /// `KL = 2 du^2 + O(du^3)`, independently of the atom, layer, or model.
    /// `None` when behavior is inert or the behavior image is degenerate.
    pub behavior_pinned_chart: Option<BehaviorPinnedChart>,
}

/// The behavior block's canonical representative of a one-dimensional chart.
///
/// This is a quotient coordinate, not a second fitted latent: activation and
/// behavior still share the fit's single `t`, while `coords` records the exact
/// post-fit reparameterization of that shared chart into behavior arc length.
/// It therefore remains valid when a finite basis cannot recompose an active
/// nonlinear reparameterization without approximation (the same read-only
/// quotient discipline used by the activation `coords_u_arc` certificate).
#[derive(Debug, Clone)]
pub struct BehaviorPinnedChart {
    /// Per-row behavior-arc coordinate in `sqrt(nats / 2)` units.
    pub coords: Array1<f64>,
    /// Fitted row nearest the behavior embedding basepoint; its coordinate is
    /// exactly zero (modulo `period` for a circle).
    pub anchor_row: usize,
    /// `+1` or `-1`, selected deterministically from the behavior tangent at
    /// the pinned origin.
    pub orientation: i8,
    /// Total behavior-image length in the behavior chart's Fisher metric.
    pub behavior_length: f64,
    /// Period of `coords` for a circular chart (`behavior_length / sqrt(2)`),
    /// or `None` for an interval.
    pub period: Option<f64>,
    /// Universal calibration of this quotient coordinate.  Equal to exactly
    /// `2.0` by construction, not an empirical average.
    pub nats_per_unit_coordinate: f64,
}

/// Build the representation–behavior isometry certificate for one fitted atom, or
/// `None` when the atom has no `d = 1` chart (the induced-speed construction is
/// 1-D) or when no behavior block is installed on the term (there is no y-block to
/// compare the activation geometry against).
///
/// The row set mirrors the other per-atom diagnostics: all of the atom's fitted
/// coordinate rows, support-weighted by the shared atom support measure so
/// low-occupancy rows do not dominate the ratio statistics.
pub fn atom_behavior_isometry(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<Option<AtomBehaviorIsometry>, String> {
    // A behavior block is required: without a y-block there is no second geometry.
    let Some(block) = term.behavior_block().cloned() else {
        return Ok(None);
    };
    // The induced-speed construction (curve_speeds) is a 1-D latent property; use
    // the same d = 1 gate the in-loop unit-speed retraction uses.
    let Some(topology) = term.d1_unit_speed_topology(atom_idx) else {
        return Ok(None);
    };
    let atom = &term.atoms[atom_idx];
    let evaluator = atom
        .basis_evaluator
        .as_ref()
        .ok_or_else(|| format!("atom_behavior_isometry: atom {atom_idx} has no basis evaluator"))?;

    // Split the fitted augmented decoder [B_k | √λ_y C_k] into the activation
    // decoder B_k and the nats-unit behavior decoder C_k (the √λ_y un-done).
    // Tier-0 column standardization/equilibration (#2015) is a fit-internal
    // frame: undo it first so b_k/c_k are in raw activation / nats units,
    // matching the raw-frame quantities the isometry ratio is quoted against.
    let mut augmented = atom.decoder_coefficients().to_owned();
    if let Some(scale) = term.tier0_scale() {
        if scale.len() != augmented.ncols() {
            return Err(format!(
                "atom_behavior_isometry: tier0 scale length {} != decoder width {}",
                scale.len(),
                augmented.ncols()
            ));
        }
        for (col, &s) in scale.iter().enumerate() {
            augmented.column_mut(col).mapv_inplace(|v| v * s);
        }
    }
    let (b_k, c_k) = block.split_decoder(augmented.view())?;
    let coords = term.assignment.coords[atom_idx].as_matrix().to_owned();
    if coords.ncols() != 1 {
        return Ok(None);
    }

    // One jet evaluation at the fitted rows, contracted with each decoder block:
    // s_x = ‖Φ'(t) B_k‖, and s_y = √(ẏᵀ G(y) ẏ) with ẏ = Φ'(t) C_k at y = Φ(t) C_k,
    // in the behavior chart's Fisher metric.
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let s_x = curve_speeds(&jet, b_k.view())?;
    let behavior_points = phi.dot(&c_k);
    let s_y = behavior_curve_speeds(&jet, c_k.view(), behavior_points.view())?;
    if s_x.len() != coords.nrows() || s_y.len() != coords.nrows() {
        return Err(format!(
            "atom_behavior_isometry: speed profiles have lengths {}/{} but atom has {} rows",
            s_x.len(),
            s_y.len(),
            coords.nrows()
        ));
    }

    let support = SupportMeasure::from_assignment(&term.assignment, atom_idx)?;
    let weights = support.weights();
    if weights.len() != s_x.len() {
        return Err(format!(
            "atom_behavior_isometry: support has {} rows but atom has {}",
            weights.len(),
            s_x.len()
        ));
    }

    let mut report = assemble(atom_idx, &s_x, &s_y, weights, support.mass());
    if report.behavior_engaged && report.behavior_metric_collapse_rows == 0 {
        report.behavior_pinned_chart =
            behavior_pinned_chart(evaluator.as_ref(), c_k.view(), coords.column(0), &topology)?;
    }
    Ok(Some(report))
}

/// Behavior induced speeds `s_y = √(ẏᵀ G(y) ẏ)` of the curve `y(t) = Φ(t) C`, in
/// the sphere-tangent chart's Fisher metric
/// ([`super::SphereTangentEmbedding::predicted_nats`]), so `s_y²·Δt²` is the
/// second-order KL of a latent step wherever the behavior image sits, not only at
/// the chart basepoint. `points` holds the decoded behavior coordinates `Φ(t) C`
/// at the jet's rows.
fn behavior_curve_speeds(
    jet: &ndarray::Array3<f64>,
    decoder: ArrayView2<'_, f64>,
    points: ArrayView2<'_, f64>,
) -> Result<Vec<f64>, String> {
    let (rows, m, d) = jet.dim();
    if d != 1 || decoder.nrows() != m || points.dim() != (rows, decoder.ncols()) {
        return Err(format!(
            "behavior_curve_speeds: jet {:?}, decoder {:?} and points {:?} do not describe \
             one 1-D behavior curve",
            jet.dim(),
            decoder.dim(),
            points.dim()
        ));
    }
    let mut velocity = Array1::<f64>::zeros(decoder.ncols());
    let mut velocity_absolute_sum = Array1::<f64>::zeros(decoder.ncols());
    let mut speeds = Vec::with_capacity(rows);
    for row in 0..rows {
        velocity.fill(0.0);
        velocity_absolute_sum.fill(0.0);
        for basis in 0..m {
            let dphi = jet[[row, basis, 0]];
            for ((slot, absolute), &coefficient) in velocity
                .iter_mut()
                .zip(velocity_absolute_sum.iter_mut())
                .zip(decoder.row(basis))
            {
                let term = dphi * coefficient;
                *slot += term;
                *absolute += term.abs();
            }
        }
        let dose = super::SphereTangentEmbedding::predicted_nats(points.row(row), velocity.view())
            .map_err(|error| format!("behavior_curve_speeds: row {row}: {error}"))?;
        // `G(y) ⪰ I`, so `s_y = 0` exactly when `ẏ = 0`. A velocity whose every
        // component lies within the rounding band of its `m`-term accumulation is
        // indistinguishable from zero, and its speed is reported as exactly 0.
        let resolved = velocity
            .iter()
            .zip(velocity_absolute_sum.iter())
            .any(|(&component, &absolute)| {
                component.abs() > gam_linalg::roundoff::accumulation_band(m, absolute)
            });
        speeds.push(if resolved { dose.sqrt() } else { 0.0 });
    }
    Ok(speeds)
}

/// Construct the behavior-pinned arc-length representative on the activation
/// chart canonicalizer's arc-length grid, then read the fitted rows into that
/// coordinate. The two quotient reads share one numerical contract
/// ([`crate::chart_canonicalization::ARC_LENGTH_GRID_CELLS`]): the behavior
/// speed is sampled at every cell's node, midpoint and next node, the cumulative
/// arc length is the composite-Simpson sum (per-cell error `O(Δu⁵)`), a row
/// between nodes reads the exact integral of its cell's quadratic speed
/// interpolant ([`crate::chart_canonicalization::partial_cell_arc`]), and an
/// interval chart takes the same resolvable domain
/// ([`crate::chart_canonicalization::arc_length_grid_resolves`]).
fn behavior_pinned_chart(
    evaluator: &dyn SaeBasisEvaluator,
    behavior_decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView1<'_, f64>,
    topology: &crate::chart_canonicalization::CanonicalChartTopology,
) -> Result<Option<BehaviorPinnedChart>, String> {
    use crate::chart_canonicalization::{
        ARC_LENGTH_GRID_CELLS, CanonicalChartTopology, arc_length_grid_resolves,
        partial_cell_arc,
    };

    if row_coords.is_empty() {
        return Ok(None);
    }
    let (lo, hi, circular) = match topology {
        CanonicalChartTopology::Circle { period } => {
            if !period.is_finite() || *period <= 0.0 {
                return Ok(None);
            }
            (0.0, *period, true)
        }
        CanonicalChartTopology::Interval => {
            let lo = row_coords.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = row_coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !(lo.is_finite() && hi.is_finite() && arc_length_grid_resolves(lo, hi)) {
                return Ok(None);
            }
            (lo, hi, false)
        }
    };
    // Simpson grid: node `j` at sample `2j`, the midpoint of cell `j` at `2j + 1`.
    let cells = ARC_LENGTH_GRID_CELLS;
    let step = (hi - lo) / cells as f64;
    let mut grid = Array2::<f64>::zeros((2 * cells + 1, 1));
    for j in 0..=cells {
        grid[[2 * j, 0]] = lo + j as f64 * step;
        if j < cells {
            grid[[2 * j + 1, 0]] = lo + (j as f64 + 0.5) * step;
        }
    }
    let (phi, jet) = evaluator.evaluate(grid.view())?;
    let behavior_points = phi.dot(&behavior_decoder);
    let speeds = behavior_curve_speeds(&jet, behavior_decoder, behavior_points.view())?;
    if speeds.len() != 2 * cells + 1
        || speeds
            .iter()
            .any(|speed| !speed.is_finite() || *speed < 0.0)
    {
        return Ok(None);
    }
    let mut cumulative = Array1::<f64>::zeros(cells + 1);
    for j in 0..cells {
        let (f0, fm, f1) = (speeds[2 * j], speeds[2 * j + 1], speeds[2 * j + 2]);
        cumulative[j + 1] = cumulative[j] + step * (f0 + 4.0 * fm + f1) / 6.0;
    }
    let behavior_length = cumulative[cells];
    if !(behavior_length.is_finite() && behavior_length > 0.0) {
        return Ok(None);
    }

    // Origin: closest decoded behavior tangent point to the embedding's
    // Frechet basepoint (the zero vector in tangent coordinates), searched on
    // the grid nodes, where the cumulative arc length is tabulated.
    let mut anchor_node = 0usize;
    let mut anchor_norm_sq = f64::INFINITY;
    for j in 0..=cells {
        let point = behavior_points.row(2 * j);
        let norm_sq = point.dot(&point);
        if norm_sq < anchor_norm_sq {
            anchor_norm_sq = norm_sq;
            anchor_node = j;
        }
    }
    let anchor_arc = cumulative[anchor_node];

    // Orientation: lexicographic sign of the physical behavior tangent at the
    // pinned origin.  If the exact anchor is stationary, use the nearest grid
    // point with a resolved tangent; a globally stationary image was rejected
    // by the positive-length gate above.
    // A component's sign is read only once it clears the rounding band of its
    // accumulation, the same resolution rule `behavior_curve_speeds` applies.
    let mut orientation = 0_i8;
    for radius in 0..=cells {
        for node in [
            anchor_node.saturating_sub(radius),
            (anchor_node + radius).min(cells),
        ] {
            let idx = 2 * node;
            if speeds[idx] == 0.0 {
                continue;
            }
            for out in 0..behavior_decoder.ncols() {
                let mut derivative = 0.0_f64;
                let mut absolute = 0.0_f64;
                for basis in 0..behavior_decoder.nrows() {
                    let term = jet[[idx, basis, 0]] * behavior_decoder[[basis, out]];
                    derivative += term;
                    absolute += term.abs();
                }
                if derivative.abs()
                    > gam_linalg::roundoff::accumulation_band(behavior_decoder.nrows(), absolute)
                {
                    orientation = if derivative > 0.0 { 1 } else { -1 };
                    break;
                }
            }
            if orientation != 0 {
                break;
            }
        }
        if orientation != 0 {
            break;
        }
    }
    if orientation == 0 {
        return Ok(None);
    }

    // Arc length from `lo` to a row: the tabulated nodes plus the exact integral
    // of the row's cell-local quadratic speed interpolant, the read
    // `unit_speed_reparameterization` applies to the activation chart.
    let interpolate_arc = |coord: f64| -> f64 {
        let local = if circular {
            (coord - lo).rem_euclid(hi - lo)
        } else {
            (coord - lo).clamp(0.0, hi - lo)
        };
        let cell = ((local / step).floor() as usize).min(cells - 1);
        let x = local - cell as f64 * step;
        cumulative[cell]
            + partial_cell_arc(
                speeds[2 * cell],
                speeds[2 * cell + 1],
                speeds[2 * cell + 2],
                step,
                x,
            )
    };
    let coordinate_period = behavior_length * std::f64::consts::FRAC_1_SQRT_2;
    let mut canonical = Array1::<f64>::zeros(row_coords.len());
    for (row, &coord) in row_coords.iter().enumerate() {
        let arc = interpolate_arc(coord);
        let signed = orientation as f64 * (arc - anchor_arc);
        canonical[row] = if circular {
            (signed * std::f64::consts::FRAC_1_SQRT_2).rem_euclid(coordinate_period)
        } else {
            signed * std::f64::consts::FRAC_1_SQRT_2
        };
    }
    let anchor_row = canonical
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let distance = |value: f64| {
                if circular {
                    value.min(coordinate_period - value)
                } else {
                    value.abs()
                }
            };
            distance(**a)
                .partial_cmp(&distance(**b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(row, _)| row)
        .unwrap_or(0);
    // Pin the reported fitted row exactly at the origin.  This is a constant
    // rotation of the already arc-length coordinate and changes no distances.
    let row_shift = canonical[anchor_row];
    for value in canonical.iter_mut() {
        *value = if circular {
            (*value - row_shift).rem_euclid(coordinate_period)
        } else {
            *value - row_shift
        };
    }

    Ok(Some(BehaviorPinnedChart {
        coords: canonical,
        anchor_row,
        orientation,
        behavior_length,
        period: circular.then_some(coordinate_period),
        nats_per_unit_coordinate: 2.0,
    }))
}

/// The representation–behavior isometry certificate for every atom of the term
/// (in atom order); each entry is `None` when [`atom_behavior_isometry`] returns
/// `None` for that atom (non-`d = 1`, or no behavior block installed).
pub(crate) fn behavior_isometry_report(
    term: &SaeManifoldTerm,
) -> Result<Vec<Option<AtomBehaviorIsometry>>, String> {
    (0..term.atoms.len())
        .map(|k| atom_behavior_isometry(term, k))
        .collect()
}

/// Assemble the certificate from the two speed profiles and the support weights.
/// Split out so the arithmetic is unit-testable without a fitted term.
fn assemble(
    atom_idx: usize,
    s_x: &[f64],
    s_y: &[f64],
    weights: ArrayView1<'_, f64>,
    support_mass: f64,
) -> AtomBehaviorIsometry {
    // Weighted RMS of each speed over positive-support rows.
    let mut mass = 0.0_f64;
    let mut sx_sq = 0.0_f64;
    let mut sy_sq = 0.0_f64;
    let mut sy_max = 0.0_f64;
    let mut n_rows = 0usize;
    for i in 0..s_x.len() {
        let w = weights[i];
        if !(w > 0.0) {
            continue;
        }
        n_rows += 1;
        mass += w;
        sx_sq += w * s_x[i] * s_x[i];
        sy_sq += w * s_y[i] * s_y[i];
        sy_max = sy_max.max(s_y[i]);
    }
    let activation_speed_rms = if mass > 0.0 {
        (sx_sq / mass).sqrt()
    } else {
        f64::NAN
    };
    let behavior_speed_rms = if mass > 0.0 {
        (sy_sq / mass).sqrt()
    } else {
        f64::NAN
    };
    let nats_per_unit_t = if mass > 0.0 { sy_sq / mass } else { f64::NAN };

    // A behaviorally inert atom (C_k ≈ 0) has no behavior geometry to match.
    let behavior_engaged = mass > 0.0 && sy_max > 0.0;
    if !behavior_engaged {
        return AtomBehaviorIsometry {
            atom_idx,
            n_rows,
            support_mass,
            behavior_engaged: false,
            behavior_metric_collapse_rows: n_rows,
            activation_speed_rms,
            behavior_speed_rms,
            scale: f64::NAN,
            defect_cv: f64::NAN,
            min_ratio_over_scale: f64::NAN,
            max_ratio_over_scale: f64::NAN,
            nats_per_unit_t,
            behavior_pinned_chart: None,
        };
    }

    // A scaled isometry is a pointwise statement on the whole support. Do not
    // condition the statistic on `s_y > 0`: doing so dropped precisely the
    // rows where the behavioral metric collapsed and could turn `[1,0]` into a
    // perfect match for `[1,1]`. `behavior_curve_speeds` reports an unresolved
    // velocity as exactly 0, so exact zero is the collapse test.
    let collapse_rows = (0..s_y.len())
        .filter(|&i| weights[i] > 0.0 && s_y[i] == 0.0)
        .count();
    if collapse_rows > 0 {
        return AtomBehaviorIsometry {
            atom_idx,
            n_rows,
            support_mass,
            behavior_engaged: true,
            behavior_metric_collapse_rows: collapse_rows,
            activation_speed_rms,
            behavior_speed_rms,
            scale: f64::NAN,
            defect_cv: f64::INFINITY,
            min_ratio_over_scale: f64::NAN,
            max_ratio_over_scale: f64::INFINITY,
            nats_per_unit_t,
            behavior_pinned_chart: None,
        };
    }

    // Weighted mean and variance of r = s_x/s_y on the complete support. The
    // support mass is `mass` (> 0 by the engagement check above).
    let mut r_mean = 0.0_f64;
    let mut r_min = f64::INFINITY;
    let mut r_max = f64::NEG_INFINITY;
    let mut ratios: Vec<(f64, f64)> = Vec::with_capacity(s_x.len());
    for i in 0..s_x.len() {
        let w = weights[i];
        if !(w > 0.0) {
            continue;
        }
        let r = s_x[i] / s_y[i];
        r_mean += w * r;
        r_min = r_min.min(r);
        r_max = r_max.max(r);
        ratios.push((r, w));
    }
    r_mean /= mass;
    let mut r_var = 0.0_f64;
    for (r, w) in &ratios {
        let d = r - r_mean;
        r_var += w * d * d;
    }
    r_var /= mass;
    let defect_cv = if r_mean != 0.0 {
        r_var.sqrt() / r_mean.abs()
    } else {
        f64::NAN
    };

    AtomBehaviorIsometry {
        atom_idx,
        n_rows,
        support_mass,
        behavior_engaged: true,
        behavior_metric_collapse_rows: 0,
        activation_speed_rms,
        behavior_speed_rms,
        scale: r_mean,
        defect_cv,
        min_ratio_over_scale: r_min / r_mean,
        max_ratio_over_scale: r_max / r_mean,
        nats_per_unit_t,
        behavior_pinned_chart: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// A constant speed ratio (scaled isometry) reports zero defect and a scale
    /// equal to the constant ratio, and the calibration reads the behavior speed².
    #[test]
    fn constant_ratio_is_zero_defect() {
        let n = 32usize;
        // s_y varies across rows; s_x = 3·s_y exactly ⇒ r ≡ 3, defect 0.
        let s_y: Vec<f64> = (0..n).map(|i| 0.5 + 0.4 * (i as f64 / n as f64)).collect();
        let s_x: Vec<f64> = s_y.iter().map(|&v| 3.0 * v).collect();
        let weights = Array1::<f64>::ones(n);
        let cert = assemble(0, &s_x, &s_y, weights.view(), n as f64);
        assert!(cert.behavior_engaged);
        assert_eq!(cert.behavior_metric_collapse_rows, 0);
        assert!((cert.scale - 3.0).abs() < 1e-12, "scale {}", cert.scale);
        assert!(cert.defect_cv < 1e-12, "defect {}", cert.defect_cv);
        // nats/unit t = weighted mean of s_y².
        let want: f64 = s_y.iter().map(|v| v * v).sum::<f64>() / n as f64;
        assert!((cert.nats_per_unit_t - want).abs() < 1e-12);
    }

    #[test]
    fn local_behavior_metric_collapse_invalidates_isometry() {
        let s_x = [1.0, 1.0];
        let s_y = [1.0, 0.0];
        let weights = Array1::<f64>::ones(2);
        let cert = assemble(0, &s_x, &s_y, weights.view(), 2.0);
        assert!(cert.behavior_engaged);
        assert_eq!(cert.behavior_metric_collapse_rows, 1);
        assert!(cert.scale.is_nan());
        assert!(cert.defect_cv.is_infinite());
    }

    /// A varying speed ratio (broken isometry) reports a strictly positive defect
    /// whose value matches the CV of the planted ratio.
    #[test]
    fn varying_ratio_reports_positive_defect() {
        let n = 40usize;
        let s_y: Vec<f64> = vec![1.0; n];
        // r_i = 1 + 0.5·cos(2π i/n): mean 1, so CV = std = 0.5/√2.
        let s_x: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos())
            .collect();
        let weights = Array1::<f64>::ones(n);
        let cert = assemble(0, &s_x, &s_y, weights.view(), n as f64);
        let want_cv = (0.5_f64 * 0.5 / 2.0).sqrt(); // std of 0.5·cos over a full period
        assert!(
            (cert.defect_cv - want_cv).abs() < 1e-2,
            "defect {} vs expected {want_cv}",
            cert.defect_cv
        );
        assert!(cert.defect_cv > 0.2);
    }

    /// Composite-Simpson behavior arc length over `[a, b]` with `cells` cells of
    /// its own (a grid not aligned with the production chart grid).
    fn reference_behavior_arc(
        evaluator: &dyn SaeBasisEvaluator,
        decoder: ArrayView2<'_, f64>,
        a: f64,
        b: f64,
        cells: usize,
    ) -> f64 {
        if b <= a {
            return 0.0;
        }
        let h = (b - a) / cells as f64;
        let mut grid = Array2::<f64>::zeros((2 * cells + 1, 1));
        for j in 0..=cells {
            grid[[2 * j, 0]] = a + j as f64 * h;
            if j < cells {
                grid[[2 * j + 1, 0]] = a + (j as f64 + 0.5) * h;
            }
        }
        let (phi, jet) = evaluator.evaluate(grid.view()).expect("evaluate");
        let points = phi.dot(&decoder);
        let speeds = behavior_curve_speeds(&jet, decoder, points.view()).expect("speeds");
        (0..cells)
            .map(|j| h * (speeds[2 * j] + 4.0 * speeds[2 * j + 1] + speeds[2 * j + 2]) / 6.0)
            .sum()
    }

    /// #4317: the pinned behavior chart integrates behavior arc length under the
    /// activation canonicalizer's Simpson contract. A smooth two-harmonic
    /// behavior arc on the interval chart `[0.1, 0.4]` has Simpson per-cell error
    /// `h⁵ max|s⁗|/2880` with `h = 0.3/2048`, so the chart length and every
    /// pairwise pinned distance `|u_i − u_j| = |s(t_i) − s(t_j)|/√2` agree with an
    /// independent 8192-cell Simpson reference to rounding (1e-11 of the length).
    /// Positive control: the trapezoid-plus-linear-interpolation read on the
    /// same 2048 nodes misses both by more than 1e-9 of the length.
    #[test]
    fn behavior_pinned_chart_arc_length_matches_the_simpson_contract() {
        use crate::basis::PeriodicHarmonicEvaluator;
        use crate::chart_canonicalization::{ARC_LENGTH_GRID_CELLS, CanonicalChartTopology};

        let evaluator = PeriodicHarmonicEvaluator::new(5).expect("evaluator");
        // Rows of the decoder are the basis columns [1, s1, c1, s2, c2].
        let decoder = ndarray::array![[0.05, 0.0], [0.0, 0.3], [0.3, 0.0], [0.1, 0.0], [0.0, 0.08]];
        let rows = Array1::from(vec![0.1, 0.123_456_7, 0.2, 0.271_828, 0.314_159, 0.377_7, 0.4]);
        let (lo, hi) = (0.1, 0.4);

        let chart = behavior_pinned_chart(
            &evaluator,
            decoder.view(),
            rows.view(),
            &CanonicalChartTopology::Interval,
        )
        .expect("chart")
        .expect("a smooth non-stationary behavior arc has a pinned chart");
        assert_eq!(chart.period, None);

        let reference_cells = 8192;
        let reference_length =
            reference_behavior_arc(&evaluator, decoder.view(), lo, hi, reference_cells);
        let reference_arc: Vec<f64> = rows
            .iter()
            .map(|&t| reference_behavior_arc(&evaluator, decoder.view(), lo, t, reference_cells))
            .collect();
        let tolerance = 1e-11 * reference_length;
        let length_error = (chart.behavior_length - reference_length).abs();
        assert!(
            length_error <= tolerance,
            "behavior_length {} vs reference {reference_length}: error {length_error:e}",
            chart.behavior_length
        );
        let mut worst_distance_error = 0.0_f64;
        for i in 0..rows.len() {
            for j in 0..i {
                let got = (chart.coords[i] - chart.coords[j]).abs();
                let want = (reference_arc[i] - reference_arc[j]).abs() * std::f64::consts::FRAC_1_SQRT_2;
                worst_distance_error = worst_distance_error.max((got - want).abs());
            }
        }
        assert!(
            worst_distance_error <= tolerance,
            "pinned pairwise distances miss the reference by {worst_distance_error:e} \
             (tolerance {tolerance:e})"
        );

        // Positive control: the node-only trapezoid read the chart used before.
        let cells = ARC_LENGTH_GRID_CELLS;
        let step = (hi - lo) / cells as f64;
        let mut nodes = Array2::<f64>::zeros((cells + 1, 1));
        for i in 0..=cells {
            nodes[[i, 0]] = lo + step * i as f64;
        }
        let (phi, jet) = evaluator.evaluate(nodes.view()).expect("evaluate");
        let points = phi.dot(&decoder);
        let speeds = behavior_curve_speeds(&jet, decoder.view(), points.view()).expect("speeds");
        let mut trapezoid = vec![0.0_f64; cells + 1];
        for i in 1..=cells {
            trapezoid[i] = trapezoid[i - 1] + 0.5 * step * (speeds[i - 1] + speeds[i]);
        }
        let trapezoid_arc = |t: f64| {
            let pos = ((t - lo) / step).clamp(0.0, cells as f64);
            let left = (pos.floor() as usize).min(cells - 1);
            let frac = pos - left as f64;
            trapezoid[left] + frac * (trapezoid[left + 1] - trapezoid[left])
        };
        assert!(
            (trapezoid[cells] - reference_length).abs() > 1e-9 * reference_length,
            "positive control: the trapezoid length should miss the reference"
        );
        let trapezoid_worst_row = rows
            .iter()
            .zip(&reference_arc)
            .map(|(&t, &want)| (trapezoid_arc(t) - want).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            trapezoid_worst_row > 1e-9 * reference_length,
            "positive control: the trapezoid row read should miss the reference, got {trapezoid_worst_row:e}"
        );
    }

    /// An inert behavior block (all behavior speeds zero) is reported as not
    /// engaged with a NaN defect — no correspondence to certify, not a defect.
    #[test]
    fn inert_behavior_is_not_engaged() {
        let n = 16usize;
        let s_x: Vec<f64> = vec![1.0; n];
        let s_y: Vec<f64> = vec![0.0; n];
        let weights = Array1::<f64>::ones(n);
        let cert = assemble(0, &s_x, &s_y, weights.view(), n as f64);
        assert!(!cert.behavior_engaged);
        assert!(cert.defect_cv.is_nan());
        assert!(cert.scale.is_nan());
    }
}
