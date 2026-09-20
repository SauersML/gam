//! #2691 — the chart's OWN dispersion, and the refusal that makes a collapsed
//! chart impossible to return silently.
//!
//! `sae_manifold_fit` could converge, certify, report a healthy REML trajectory
//! and hand back a `d_atom = 1` chart coordinate that was a single value to
//! fourteen decimal places. Every consumer downstream (steering, #2234's E1/E2)
//! then measured exact zeros, because a chart with one point has no
//! displacements in it. Nothing in the returned object said so.
//!
//! Two properties of that defect fix the shape of this module:
//!
//! 1. **It must not be denominated in reconstruction quality.** The #2691
//!    ledger measured the fully collapsed arm at `fit_ev = 0.0581`, BELOW a
//!    partially collapsed arm at `0.0883`, while the only arm that recovered
//!    anything had the HIGHEST EV (`0.3179`). A chart compressed into a small
//!    arc still lets the harmonic decoder trace the ring — the decoder simply
//!    rescales — so EV keeps reporting a fine reconstruction while the
//!    coordinate has stopped being a coordinate. Every quantity here is a
//!    property of the coordinate alone; the target is never consulted.
//!
//! 2. **It must be measured in the chart's own manifold.** On a period-`P`
//!    circle `t` and `t + P` are the SAME point, so the raw `f64` standard
//!    deviation of the coordinate is not its dispersion: the #2691 chart-
//!    dimension scan reported `coord_std ≈ 4.98e-1` with two distinct values at
//!    `pca-dim` 3, 8 and 16 and read them as healthy, when `{0.0, 1.0}` is one
//!    point of the circle and every one of those dimensions was collapsed. A
//!    periodic axis is therefore measured by its circular variance
//!    `1 − |mean exp(i κ t)|`, a Euclidean axis by its standard deviation.

use ndarray::{ArrayView1, ArrayView2};

use super::SaeManifoldTerm;

/// Which atoms are LOAD-BEARING for the reconstruction, decided at the
/// representation limit rather than by a tuned activity threshold.
///
/// Row `i` is reconstructed as `Σ_k a[i,k] · decode_k(t_ik)`. An atom whose
/// weight on every row is below `ε · max_j a[i,j]` cannot change that sum at
/// binary64 — adding it to the dominant term is the identity — so it does not
/// participate in the fit and its chart is unobserved. Every other atom does
/// participate: whatever its chart says is part of the answer the caller gets.
///
/// This is the same kind of quantity as the per-axis `floor` above (the
/// resolution at which two values are the same f64 point at that axis's own
/// magnitude), and for the same reason: the question "does this object affect
/// the returned numbers?" has a representation answer, not a policy answer.
pub(crate) fn load_bearing_atoms(assignments: ArrayView2<'_, f64>) -> Vec<bool> {
    let k = assignments.ncols();
    let mut load_bearing = vec![false; k];
    for row in assignments.rows() {
        let dominant = row.iter().fold(0.0_f64, |m, &a| m.max(a.abs()));
        if !(dominant > 0.0) {
            // No atom carries this row at all; it distinguishes nothing.
            continue;
        }
        let representable = f64::EPSILON * dominant;
        for (atom, &weight) in row.iter().enumerate() {
            if weight.abs() > representable {
                load_bearing[atom] = true;
            }
        }
    }
    load_bearing
}

/// The dispersion of one atom's chart axis, measured in that axis's own
/// manifold, together with the floor below which the axis carries no
/// coordinate at all.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartAxisDispersion {
    pub atom: usize,
    pub atom_name: String,
    pub axis: usize,
    /// The axis's period when it is periodic; `None` for a Euclidean axis.
    pub period: Option<f64>,
    /// Periodic axis: the circular variance `1 − |mean exp(i κ t)| ∈ [0, 1]`.
    /// Euclidean axis: the coordinate's standard deviation.
    pub dispersion: f64,
    /// The dispersion `n` rows would show if they were the SAME chart point up
    /// to floating-point representation at this axis's own magnitude. Below it,
    /// the axis is a constant and the chart has one point along it.
    pub floor: f64,
    /// Number of chart points the axis resolves, after wrapping a periodic axis
    /// into one period. `1` is a collapsed axis by construction.
    pub resolved_points: usize,
}

impl ChartAxisDispersion {
    /// Whether this axis has stopped being a coordinate: its rows are one
    /// point of its manifold, up to floating-point representation.
    pub fn degenerate(&self) -> bool {
        self.resolved_points <= 1 || !(self.dispersion > self.floor)
    }
}

/// Every chart axis of a fitted dictionary, in `(atom, axis)` order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartDegeneracyReport {
    pub axes: Vec<ChartAxisDispersion>,
    /// Number of atoms in the dictionary the report was taken from.
    pub atom_count: usize,
}

impl ChartDegeneracyReport {
    pub(crate) fn degenerate_axes(&self) -> impl Iterator<Item = &ChartAxisDispersion> {
        self.axes.iter().filter(|axis| axis.degenerate())
    }


    /// Whether some atom has lost its ENTIRE chart — every one of its axes is a
    /// single point, so the "manifold atom" decodes to one point of the ambient
    /// space and carries no displacements. This is the #2691 condition.
    pub fn atoms_without_a_chart(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for atom in 0..self.atom_count {
            let mut axes = self.axes.iter().filter(|entry| entry.atom == atom).peekable();
            if axes.peek().is_none() {
                continue;
            }
            if axes.all(ChartAxisDispersion::degenerate) {
                out.push(atom);
            }
        }
        out
    }

    /// The atoms that have lost their entire chart AND are load-bearing for the
    /// reconstruction — the atoms whose collapse the caller actually receives.
    ///
    /// `atoms_without_a_chart` alone is not the refusal condition at `K ≥ 2`
    /// for opposite reasons in the two directions: an atom that carries no
    /// representable assignment mass has an unobserved chart and refusing on it
    /// would refuse fits that are fine, while an atom that DOES carry mass and
    /// has no chart is a point masquerading as a manifold inside an otherwise
    /// healthy dictionary — the case where one atom's collapse hides behind
    /// another's, which a fit-level aggregate cannot see.
    pub(crate) fn chart_less_load_bearing_atoms(&self, assignments: ArrayView2<'_, f64>) -> Vec<usize> {
        let load_bearing = load_bearing_atoms(assignments);
        self.atoms_without_a_chart()
            .into_iter()
            .filter(|atom| load_bearing.get(*atom).copied().unwrap_or(true))
            .collect()
    }


    /// One line per named atom, for a refusal message, plus the surviving atoms'
    /// dispersions — so a partial collapse reads as "atom 0 is a point WHILE
    /// atom 1 is a chart", which is the state a fit-level aggregate hides.
    pub fn atom_evidence(&self, atoms: &[usize]) -> String {
        let named = atoms
            .iter()
            .copied()
            .map(|atom| {
                let detail = self
                    .axes
                    .iter()
                    .filter(|entry| entry.atom == atom)
                    .map(|entry| {
                        let kind = match entry.period {
                            Some(period) => format!("periodic(P={period:.6e}) circular variance"),
                            None => "euclidean standard deviation".to_string(),
                        };
                        format!(
                            "axis {} {kind} {:.6e} <= floor {:.6e} ({} resolved chart point(s) \
                             over the rows)",
                            entry.axis, entry.dispersion, entry.floor, entry.resolved_points
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                let name = self
                    .axes
                    .iter()
                    .find(|entry| entry.atom == atom)
                    .map(|entry| entry.atom_name.clone())
                    .unwrap_or_default();
                format!("atom {atom} ('{name}'): {detail}")
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let survivors = self
            .axes
            .iter()
            .filter(|entry| !atoms.contains(&entry.atom) && !entry.degenerate())
            .map(|entry| {
                format!(
                    "atom {} axis {} dispersion {:.6e} ({} chart point(s))",
                    entry.atom, entry.axis, entry.dispersion, entry.resolved_points
                )
            })
            .collect::<Vec<_>>();
        if survivors.is_empty() {
            named
        } else {
            format!(
                "{named} || the chart(s) that did NOT collapse, which is why no fit-level \
                 aggregate can see this: {}",
                survivors.join("; ")
            )
        }
    }
}

impl SaeManifoldTerm {
    /// Measure every chart axis's dispersion in its own manifold. Pure read of
    /// the fitted coordinates — no target, no reconstruction, no EV.
    pub(crate) fn chart_degeneracy_report(&self) -> ChartDegeneracyReport {
        let mut axes = Vec::new();
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let periods = coord.effective_axis_periods();
            let matrix = coord.as_matrix();
            let n = matrix.nrows();
            if n == 0 {
                continue;
            }
            let atom_name = self
                .atoms
                .get(atom_idx)
                .map(|atom| atom.name.clone())
                .unwrap_or_default();
            for axis in 0..coord.latent_dim() {
                let (dispersion, floor, resolved_points) =
                    axis_dispersion(matrix.column(axis), periods[axis]);
                axes.push(ChartAxisDispersion {
                    atom: atom_idx,
                    atom_name: atom_name.clone(),
                    axis,
                    period: periods[axis],
                    dispersion,
                    floor,
                    resolved_points,
                });
            }
        }
        ChartDegeneracyReport {
            axes,
            atom_count: self.k_atoms(),
        }
    }
}

/// `(dispersion, floor, resolved_points)` of one chart axis, measured in its
/// own manifold (see [`ChartAxisDispersion`]).
///
/// The floor sits at the representation limit (`ε · max|t|` in `t`, i.e.
/// `½ (κ ε max|t|)²` in circular variance), so the dispersion must be computed
/// with an error that scales with the SPREAD of the rows, not with their
/// magnitude. Neither textbook formula does that. `1 − |mean exp(iκt)|`
/// subtracts a resultant `≈ 1` from `1`, which has absolute error `O(ε)`
/// against a floor of `O(ε²)`: `{0.3, 100.3}` on a period-1 circle (one point
/// of it, up to the representation of `100.3`) reads `1.1e-16`, fourteen
/// orders above its floor, and is certified as a chart, while the two genuine
/// points `{0.25, 0.25 + 1e-9}` (circular variance `4.9e-18`) read exactly `0`
/// and are refused. Likewise a Euclidean `mean(t)` carries summation error of
/// order `n ε |t|` that centering on it then reports as spread: 10⁵ rows split
/// over the two adjacent floats `0.1` and `0.1 + ulp` read a standard
/// deviation of `1.9e-13`, four orders above both their true `6.9e-18` and
/// their floor.
///
/// Both are removed by measuring every row relative to the first. That offset
/// rounds by at most half an ulp of `max|t|`, below the floor's own
/// resolution, and is exact (Sterbenz) whenever the rows are close as floats.
/// On a periodic axis the offset is reduced into
/// `[−P/2, P/2]` exactly (`%` is exact, and the `± P` correction is again a
/// Sterbenz subtraction), and the circular variance is evaluated as
/// `mean 2 sin²((φᵢ − μ)/2) = 1 − mean cos(φᵢ − μ)` about the mean direction
/// `μ`. At the true `μ` that is exactly `1 − R`, and an error in `μ` enters
/// only at second order because `∂/∂μ mean cos(φᵢ − μ) = 0` there; when
/// `R = 0` it is `1` for every `μ`.
fn axis_dispersion(column: ArrayView1<'_, f64>, period: Option<f64>) -> (f64, f64, usize) {
    let n = column.len();
    let magnitude = column.iter().fold(0.0_f64, |m, &t| m.max(t.abs()));
    // The resolution at which two coordinate values on this axis are the SAME
    // f64 point, at the axis's own magnitude. This is a representation limit,
    // not a tuned tolerance.
    let resolution = f64::EPSILON * magnitude;
    let anchor = column[0];
    match period {
        Some(period) if period > 0.0 => {
            let kappa = std::f64::consts::TAU / period;
            let half = 0.5 * period;
            let phases: Vec<f64> = column
                .iter()
                .map(|&t| {
                    let offset = (t - anchor) % period;
                    let offset = if offset > half {
                        offset - period
                    } else if offset < -half {
                        offset + period
                    } else {
                        offset
                    };
                    kappa * offset
                })
                .collect();
            let (sin_sum, cos_sum) = phases
                .iter()
                .fold((0.0_f64, 0.0_f64), |(s, c), &phi| (s + phi.sin(), c + phi.cos()));
            let mean_direction = sin_sum.atan2(cos_sum);
            let dispersion = phases
                .iter()
                .map(|&phi| {
                    let half_gap = (0.5 * (phi - mean_direction)).sin();
                    2.0 * half_gap * half_gap
                })
                .sum::<f64>()
                / n as f64;
            // Rows separated by `resolution` in `t` are separated by
            // `kappa * resolution` in phase; the circular variance of a spread
            // that small is `½ (κ·resolution)²` to leading order. That is the
            // dispersion an axis shows when every row is the same chart point.
            let phase_resolution = kappa * resolution;
            let floor = 0.5 * phase_resolution * phase_resolution;
            let mut wrapped: Vec<i64> = column
                .iter()
                .map(|&t| {
                    let unit = t.rem_euclid(period) / period;
                    (unit / f64::EPSILON).round() as i64
                })
                .collect();
            wrapped.sort_unstable();
            wrapped.dedup();
            (dispersion, floor, wrapped.len())
        }
        _ => {
            let mean_offset = column.iter().map(|&t| t - anchor).sum::<f64>() / n as f64;
            let variance = column
                .iter()
                .map(|&t| {
                    let centred = (t - anchor) - mean_offset;
                    centred * centred
                })
                .sum::<f64>()
                / n as f64;
            let mut distinct: Vec<u64> = column.iter().map(|t| t.to_bits()).collect();
            distinct.sort_unstable();
            distinct.dedup();
            (variance.sqrt(), resolution, distinct.len())
        }
    }
}

/// Certificate wrapper for [`ChartDegeneracyReport`], so a fit's ledger carries
/// "the chart is still a coordinate" as an explicit claim rather than leaving
/// it to a caller to notice its absence.
#[derive(Clone, Debug)]
pub(crate) struct ChartNondegeneracyCertificate {
    axes: usize,
    degenerate_axes: usize,
    collapsed_atoms: usize,
    atom_count: usize,
    /// Smallest per-axis ratio `dispersion / floor` over all axes. `< 1` on any
    /// axis means that axis is a constant up to floating-point representation.
    min_dispersion_over_floor: f64,
    min_resolved_points: usize,
}

impl ChartNondegeneracyCertificate {
    pub fn new(report: &ChartDegeneracyReport) -> Self {
        let min_dispersion_over_floor = report
            .axes
            .iter()
            .map(|axis| {
                if axis.floor > 0.0 {
                    axis.dispersion / axis.floor
                } else if axis.dispersion > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                }
            })
            .fold(f64::INFINITY, f64::min);
        Self {
            axes: report.axes.len(),
            degenerate_axes: report.degenerate_axes().count(),
            collapsed_atoms: report.atoms_without_a_chart().len(),
            atom_count: report.atom_count,
            min_dispersion_over_floor,
            min_resolved_points: report
                .axes
                .iter()
                .map(|axis| axis.resolved_points)
                .min()
                .unwrap_or(0),
        }
    }
}

impl gam_problem::topology_certificates::Certificate for ChartNondegeneracyCertificate {
    fn claim(&self) -> gam_problem::topology_certificates::Claim {
        gam_problem::topology_certificates::Claim::new(
            "chart-nondegeneracy",
            "every fitted chart axis still separates rows in its OWN manifold \
             (circular variance on a periodic axis, standard deviation on a \
             Euclidean one) by more than floating-point representation at that \
             axis's magnitude; a chart that has collapsed to one point is \
             reported here rather than certified as a fit. This claim is \
             deliberately independent of reconstruction quality: #2691 measured \
             a fully collapsed chart with a HIGHER explained variance than a \
             partially collapsed one",
        )
    }

    fn evidence(&self) -> gam_problem::topology_certificates::Evidence {
        let mut evidence = gam_problem::topology_certificates::Evidence::new();
        evidence.insert("chart_axes", self.axes.into());
        evidence.insert("degenerate_axes", self.degenerate_axes.into());
        evidence.insert("atoms_without_a_chart", self.collapsed_atoms.into());
        evidence.insert("atoms", self.atom_count.into());
        evidence.insert("min_resolved_chart_points", self.min_resolved_points.into());
        if self.min_dispersion_over_floor.is_finite() {
            evidence.insert(
                "min_dispersion_over_floor",
                self.min_dispersion_over_floor.into(),
            );
        } else {
            evidence.insert("min_dispersion_over_floor", "n/a".into());
        }
        evidence
    }

    fn verdict(&self) -> gam_problem::topology_certificates::Verdict {
        use gam_problem::topology_certificates::Verdict;
        if self.axes == 0 {
            Verdict::Unavailable
        } else if self.degenerate_axes == 0 {
            Verdict::Certified
        } else {
            Verdict::Insufficient
        }
    }
}

#[cfg(test)]
mod tests {
    use super::axis_dispersion;
    use ndarray::Array1;

    fn measure(values: &[f64], period: Option<f64>) -> (f64, f64, usize) {
        axis_dispersion(Array1::from(values.to_vec()).view(), period)
    }

    fn is_degenerate((dispersion, floor, resolved_points): (f64, f64, usize)) -> bool {
        resolved_points <= 1 || !(dispersion > floor)
    }

    /// `100.3 − 100` is `0.3` only up to the representation of `100.3`, so the
    /// two rows are ONE point of the period-1 circle that wraps into two
    /// buckets. `1 − |mean exp(iκt)|` read `1.1e-16` here — fourteen orders
    /// above the `9.8e-27` floor — and certified the axis as a chart.
    #[test]
    fn one_circle_point_modulo_the_period_is_degenerate() {
        let measured = measure(&[0.3, 100.3], Some(1.0));
        assert_eq!(measured.2, 2, "the rows must wrap into two buckets, so the dispersion decides");
        assert!(
            is_degenerate(measured),
            "one circle point must be degenerate: dispersion {:.3e}, floor {:.3e}",
            measured.0,
            measured.1
        );
    }

    /// Two genuine points `1e-9` apart on the period-1 circle have circular
    /// variance `2 sin²(κδ/4) ≈ 4.9e-18`, fourteen orders above the floor.
    /// `1 − R` rounded that to exactly `0` and refused the chart.
    #[test]
    fn two_circle_points_a_nanoperiod_apart_are_a_chart() {
        let (a, b) = (0.25, 0.25 + 1e-9);
        let measured = measure(&[a, b], Some(1.0));
        assert!(!is_degenerate(measured), "dispersion {:.3e}, floor {:.3e}", measured.0, measured.1);
        let quarter = std::f64::consts::TAU * (b - a) / 4.0;
        let exact = 2.0 * quarter.sin() * quarter.sin();
        // Every step (the Sterbenz-exact offset, one sin, one product, a mean
        // of two) is correctly rounded or exact, so a few ulps bound it.
        assert!(
            (measured.0 - exact).abs() <= 4.0 * f64::EPSILON * exact,
            "circular variance {:.17e} vs exact {:.17e}",
            measured.0,
            exact
        );
    }

    /// A cluster straddling the seam `t ≡ t + 1` must be measured across it.
    #[test]
    fn a_cluster_straddling_the_period_seam_is_a_chart() {
        let measured = measure(&[0.0, 0.999_999_999, 1e-9], Some(1.0));
        assert!(!is_degenerate(measured), "dispersion {:.3e}, floor {:.3e}", measured.0, measured.1);
        // Small-angle circular variance `½ var(φ)`; its truncation error is
        // `O(φ²) ≈ 4e-17` relative, far inside the ulp-level bound below.
        let phases = [0.0, 0.999_999_999 - 1.0, 1e-9].map(|o| std::f64::consts::TAU * o);
        let mean = phases.iter().sum::<f64>() / 3.0;
        let small_angle = 0.5 * phases.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / 3.0;
        assert!(
            (measured.0 - small_angle).abs() <= 16.0 * f64::EPSILON * small_angle,
            "circular variance {:.17e} vs small-angle {:.17e}",
            measured.0,
            small_angle
        );
    }

    /// 10⁵ rows on the two adjacent floats `0.1` and `0.1 + ulp` have standard
    /// deviation `ulp/2 = 6.9e-18`, below the `ε·0.1 = 2.2e-17` floor. Centering
    /// on `mean(t)` read `1.9e-13` of summation error as spread and certified it.
    #[test]
    fn rows_on_two_adjacent_floats_are_one_euclidean_point() {
        let next = f64::from_bits(0.1_f64.to_bits() + 1);
        let values: Vec<f64> = (0..100_000).map(|i| if i % 2 == 0 { 0.1 } else { next }).collect();
        let measured = measure(&values, None);
        assert_eq!(measured.2, 2);
        assert!(
            is_degenerate(measured),
            "two adjacent floats must be degenerate: std {:.3e}, floor {:.3e}",
            measured.0,
            measured.1
        );
        let half_ulp = 0.5 * (next - 0.1);
        assert!(
            (measured.0 - half_ulp).abs() <= 4.0 * f64::EPSILON * half_ulp,
            "std {:.17e} vs half an ulp {:.17e}",
            measured.0,
            half_ulp
        );
    }

    /// Spread charts are unchanged: seven equally spaced circle points have
    /// resultant `0`, i.e. circular variance `1`.
    #[test]
    fn an_evenly_spread_circle_has_unit_circular_variance() {
        let values: Vec<f64> = (0..7).map(|i| i as f64 / 7.0).collect();
        let measured = measure(&values, Some(1.0));
        assert_eq!(measured.2, 7);
        assert!((measured.0 - 1.0).abs() <= 16.0 * f64::EPSILON, "circular variance {:.17e}", measured.0);
    }
}
