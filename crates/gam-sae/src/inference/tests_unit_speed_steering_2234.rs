//! #2234 / #2263 item 3 — **is a requested chart displacement realized as the
//! INTRINSIC displacement it is documented to be?**
//!
//! [`crate::manifold::SaeManifoldTerm::steer_rows`] documents its `δ` as
//! "radians / fraction-of-period, per the atom's manifold" and implements it as
//! `LatentManifold::retract(t, δ)` — a raw offset of the FITTED chart parameter.
//! Those are the same object only when the fitted parameter is arc length. The
//! chart parameterization is a gauge freedom (#2022 / #1019 / #2081): the fit
//! lands on one point of the `Diff(S¹)` orbit, and
//! [`crate::chart_canonicalization::chart_arclength_coordinates`] says the
//! consequence in as many words — *"any downstream consumer reading
//! angle/dose/adjacency off the raw `t_i` is reading a gauge-ARBITRARY
//! quantity"*. Steering reads exactly that.
//!
//! The in-loop unit-speed retraction (#2022,
//! `SaeManifoldTerm::retract_unit_speed_charts_in_loop`) exists to close the gap
//! by re-gauging the fitted chart to arc length — but it is allowed to HONESTLY
//! SKIP: `unit_speed_reparameterization` returns `Ok(None)` when the basis
//! family cannot re-express the reparameterized curve inside
//! `CHART_RECOMPOSITION_REL_TOL`, and a finite harmonic basis generically cannot
//! carry the arc-length reparameterization of a non-round ring. So whether `δ`
//! means arc length depends on whether an internal gate passed, and nothing
//! tells the caller which.
//!
//! ## Why the existing E4 pin cannot see this
//!
//! `manifold::tests_steering_e4` plants a ROUND circle. A round ring is exactly
//! rank-2, and a degree-`≤H` trigonometric-polynomial map onto a round circle
//! must be `t ↦ e^{±2πi n t}` up to a constant, so the harmonic basis PINS the
//! chart to be affine in the planted angle. On that fixture the chart is
//! unit-speed by construction and the raw offset IS the arc-length offset — the
//! defect is invisible there, not absent.
//!
//! This module plants the same object with one property changed: a
//! **1 : 0.3 ellipse**. Still one closed loop, still a rank-2 "circle feature",
//! still exactly harmonic-1 representable — but arc length is no longer affine
//! in the chart parameter. The round ring is kept as the control arm, where the
//! raw and canonical readings must agree.
//!
//! ## Ground truth is external, and the angle recovery is calibrated
//!
//! The planted ellipse's cumulative arc length `S(θ) = ∫₀^θ √(a²sin²+b²cos²)` is
//! integrated here on an independent fine grid; it is never read from the code
//! under test.
//!
//! A steered row's realized displacement is `S(θ̂(steered)) − S(θ̂(unsteered))`.
//! `θ̂` is NOT read through the planted ambient frame: the fitted decode may sit
//! in any invertible affine image of it (a Tier-0 column scaling, the decoder's
//! own scale gauge, a frame rotation), and reading through the planted frame
//! would silently price that gauge as steering error. Instead the recovery is
//! CALIBRATED on the unsteered decode by regressing it on `[1, cos θ, sin θ]`
//! — exact in closed form because the planted `θ` grid is equally spaced over a
//! full period, so the design is orthogonal.
//!
//! One bar, [`SEPARATION_REL_TOL`], SEPARATES the two surfaces and is asserted in
//! both directions; the round ring is the measured null ([`NULL_REL_TOL`]) that
//! says what the instrument can actually resolve.

#![cfg(test)]

use crate::chart_canonicalization::{CanonicalChartTopology, chart_arclength_coordinates};
use crate::manifold::{
    SaeFitAssignmentKind, SaeFitConfig, SaeFitSeedReport, SaeFitSeedRequest, SaeManifoldTerm,
    SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed, build_sae_minimal_seed,
};
use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use ndarray::{Array1, Array2, array};

/// Rows in every fixture. Enough that the fitted chart resolves the ring densely
/// (the #2263 month harness showed a 12-row fixture collapsing to four distinct
/// coordinates) and small enough for a K=1 fit in seconds.
const N_ROWS: usize = 192;
/// Ambient width of the planted cloud.
const P_OUT: usize = 6;
/// Semi-minor axis of the ellipse arm, with the semi-major fixed at 1. The
/// decoder-curve speed of an `(a, b)` ellipse ranges over `[b, a]`, so this is
/// exactly the planted speed ratio a faithful chart must show.
const ELLIPSE_MINOR: f64 = 0.3;
/// Cells in the independent planted-arc-length quadrature. Trapezoid on a smooth
/// positive integrand, then a within-cell linear read; at `h = 2π/2¹⁸` both are
/// good to `~1e-10` of the perimeter, orders below the chart-fidelity floor this
/// fixture actually measures.
const PLANTED_ARC_CELLS: usize = 1 << 18;

/// One planted closed ring `m(θ) = a·cos θ·u + b·sin θ·v` on an orthonormal
/// ambient frame, sampled on a deterministic equal-`θ` grid. `a = b` is the
/// round control; `a > b` is the discriminator.
struct PlantedRing {
    /// Ambient rows, shape `(N_ROWS, P_OUT)`.
    z: Array2<f64>,
    /// Planted intrinsic angle of each row, radians.
    theta: Array1<f64>,
    /// Cumulative planted arc length at `θ_j = 2π j / PLANTED_ARC_CELLS`.
    arc_table: Vec<f64>,
}

impl PlantedRing {
    fn new(major: f64, minor: f64) -> Self {
        let mut u = Array1::<f64>::zeros(P_OUT);
        let mut v = Array1::<f64>::zeros(P_OUT);
        for j in 0..P_OUT {
            u[j] = ((j as f64 + 1.0) * 0.7).sin();
            v[j] = ((j as f64 + 1.0) * 0.7).cos();
        }
        let un = u.dot(&u).sqrt();
        u.mapv_inplace(|x| x / un);
        let uv = u.dot(&v);
        for j in 0..P_OUT {
            v[j] -= uv * u[j];
        }
        let vn = v.dot(&v).sqrt();
        v.mapv_inplace(|x| x / vn);

        let theta = Array1::<f64>::from_shape_fn(N_ROWS, |i| {
            std::f64::consts::TAU * (i as f64 + 0.5) / N_ROWS as f64
        });
        let mut z = Array2::<f64>::zeros((N_ROWS, P_OUT));
        for i in 0..N_ROWS {
            let (c, s) = (theta[i].cos(), theta[i].sin());
            for j in 0..P_OUT {
                z[[i, j]] = major * c * u[j] + minor * s * v[j];
            }
        }

        // Independent cumulative arc length of the planted curve. `‖m'(θ)‖ =
        // √(a² sin²θ + b² cos²θ)` because (u, v) are orthonormal.
        let h = std::f64::consts::TAU / PLANTED_ARC_CELLS as f64;
        let speed = |t: f64| -> f64 {
            let (s, c) = (t.sin(), t.cos());
            (major * major * s * s + minor * minor * c * c).sqrt()
        };
        let mut arc_table = vec![0.0_f64; PLANTED_ARC_CELLS + 1];
        for j in 0..PLANTED_ARC_CELLS {
            let t0 = j as f64 * h;
            arc_table[j + 1] = arc_table[j] + 0.5 * h * (speed(t0) + speed(t0 + h));
        }

        Self {
            z,
            theta,
            arc_table,
        }
    }

    /// Total planted perimeter.
    fn perimeter(&self) -> f64 {
        self.arc_table[PLANTED_ARC_CELLS]
    }

    /// Cumulative planted arc length at an arbitrary angle, extended over the
    /// whole line by `S(θ + 2π) = S(θ) + L`.
    fn arc(&self, theta: f64) -> f64 {
        let turns = (theta / std::f64::consts::TAU).floor();
        let local = theta - turns * std::f64::consts::TAU;
        let x = local / std::f64::consts::TAU * PLANTED_ARC_CELLS as f64;
        let cell = (x.floor() as usize).min(PLANTED_ARC_CELLS - 1);
        let frac = x - cell as f64;
        let base = self.arc_table[cell] + frac * (self.arc_table[cell + 1] - self.arc_table[cell]);
        base + turns * self.perimeter()
    }
}

/// Shortest signed difference of two angles, in `(−π, π]`.
fn wrap_pi(x: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let y = x - tau * (x / tau).round();
    if y <= -std::f64::consts::PI {
        y + tau
    } else {
        y
    }
}

/// The calibrated map from an ambient decode back to the planted angle.
///
/// The unsteered decode is regressed on `[1, cos θ, sin θ]`; because the planted
/// `θ` grid is equally spaced over a full period the design is exactly
/// orthogonal, so the fit is the closed form below and needs no solver. Any
/// invertible affine image of the planted ring (Tier-0 column scaling, decoder
/// scale gauge, frame rotation, the ellipse's own axes) is absorbed into
/// `(offset, u_axis, v_axis)`, so the recovered angle is the PLANTED angle and
/// not a gauge artifact.
struct AngleRecovery {
    offset: Array1<f64>,
    u_axis: Array1<f64>,
    v_axis: Array1<f64>,
    /// Gram of `(u_axis, v_axis)` — the recovery solves a 2×2 system against it,
    /// which returns exactly `(cos θ, sin θ)` on a point of the fitted ring.
    guu: f64,
    guv: f64,
    gvv: f64,
}

impl AngleRecovery {
    fn calibrate(decode: &Array2<f64>, theta: &Array1<f64>) -> Self {
        let n = decode.nrows();
        let mut offset = Array1::<f64>::zeros(P_OUT);
        let mut u_axis = Array1::<f64>::zeros(P_OUT);
        let mut v_axis = Array1::<f64>::zeros(P_OUT);
        for i in 0..n {
            let (c, s) = (theta[i].cos(), theta[i].sin());
            for j in 0..P_OUT {
                offset[j] += decode[[i, j]];
                u_axis[j] += c * decode[[i, j]];
                v_axis[j] += s * decode[[i, j]];
            }
        }
        offset.mapv_inplace(|x| x / n as f64);
        u_axis.mapv_inplace(|x| 2.0 * x / n as f64);
        v_axis.mapv_inplace(|x| 2.0 * x / n as f64);
        let guu = u_axis.dot(&u_axis);
        let guv = u_axis.dot(&v_axis);
        let gvv = v_axis.dot(&v_axis);
        Self {
            offset,
            u_axis,
            v_axis,
            guu,
            guv,
            gvv,
        }
    }

    fn theta_of(&self, point: &[f64]) -> f64 {
        let mut bu = 0.0_f64;
        let mut bv = 0.0_f64;
        for j in 0..P_OUT {
            let r = point[j] - self.offset[j];
            bu += r * self.u_axis[j];
            bv += r * self.v_axis[j];
        }
        let det = self.guu * self.gvv - self.guv * self.guv;
        let c = (self.gvv * bu - self.guv * bv) / det;
        let s = (self.guu * bv - self.guv * bu) / det;
        s.atan2(c)
    }
}

/// Fit a converged K=1 periodic (circle) atom to a planted ring, through the
/// public dense-certification seed builders and the real arrow–Schur joint fit
/// (so the #2022 in-loop unit-speed retraction gets its chance at this chart).
fn fit_ring(ring: &PlantedRing) -> SaeManifoldTerm {
    let assignment_kind = SaeFitAssignmentKind::Softmax;
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: ring.z.view(),
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![1],
        assignment_kind,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 0,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed on the planted ring");
    let SaeMinimalSeedReport {
        geometry_plans,
        basis_values,
        basis_jacobian,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        refine_routing,
    } = minimal;

    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: ring.z.view(),
        geometry_plans: &geometry_plans,
        basis_values: basis_values.view(),
        basis_jacobian: basis_jacobian.view(),
        decoder_coefficients: decoder_coefficients.view(),
        smooth_penalties: smooth_penalties.view(),
        initial_logits: initial_logits.view(),
        initial_coords: initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: refine_routing,
        seed_refine_random_state: 0,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed on the planted ring");
    let SaeFitSeedReport {
        base_term: mut term,
        initial_rho: mut rho,
        ..
    } = seed;
    term.run_joint_fit_arrow_schur(ring.z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6)
        .expect("K=1 circle joint fit must run e2e on the planted ring");
    term
}

/// Global explained variance of `fitted` against `truth` (column-mean baseline).
fn explained_variance(truth: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let (n, p) = truth.dim();
    let mut sse = 0.0_f64;
    let mut sst = 0.0_f64;
    for j in 0..p {
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += truth[[i, j]];
        }
        mean /= n as f64;
        for i in 0..n {
            let r = truth[[i, j]] - fitted[[i, j]];
            sse += r * r;
            let d = truth[[i, j]] - mean;
            sst += d * d;
        }
    }
    if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN }
}

/// Everything one fixture reports about its own chart before any steering number
/// may be read off it.
struct ChartReport {
    ev: f64,
    span_turns: f64,
    distinct: usize,
    speed_cv: f64,
    speed_ratio: f64,
    /// Max over rows of `|S(θ̂(g(t_i))) − S(θ_i)|`: how far, in planted arc
    /// length, the fitted curve's own point for row `i` sits from where row `i`
    /// was planted.
    fidelity_arc_floor: f64,
    /// `+1` when advancing the chart parameter advances the planted angle, `−1`
    /// when it retards it. The fit recovers `t = ±θ/2π + φ`; which sign it lands
    /// on is a gauge, so it is measured once here (on a step far from the
    /// antipode, where the sign of a displacement IS determined) and reused for
    /// every requested displacement.
    orientation: f64,
    recovery: AngleRecovery,
    /// The unsteered decode, kept so every steered decode is differenced against
    /// the same object the recovery was calibrated on.
    base_decode: Array2<f64>,
}

/// Realized planted-arc displacement of one row, read as the difference of two
/// calibrated angle recoveries through the SAME decode path.
fn realized_row_displacement(
    term: &SaeManifoldTerm,
    ring: &PlantedRing,
    report: &ChartReport,
    row: usize,
    raw_step: f64,
) -> f64 {
    let steered = term
        .steer_decode(0, &[row], array![raw_step].view())
        .expect("steered decode");
    let from: Vec<f64> = (0..P_OUT).map(|j| report.base_decode[[row, j]]).collect();
    let to: Vec<f64> = (0..P_OUT).map(|j| steered[[0, j]]).collect();
    let theta_from = report.recovery.theta_of(&from);
    let theta_to = report.recovery.theta_of(&to);
    let step = wrap_pi(theta_to - theta_from);
    ring.arc(theta_from + step) - ring.arc(theta_from)
}

fn chart_report(term: &SaeManifoldTerm, ring: &PlantedRing) -> ChartReport {
    let fitted = term.try_fitted().expect("fitted reconstruction");
    let ev = explained_variance(&ring.z, &fitted);

    let coords = term.assignment.coords[0].as_matrix();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut values: Vec<f64> = Vec::with_capacity(N_ROWS);
    for row in 0..N_ROWS {
        lo = lo.min(coords[[row, 0]]);
        hi = hi.max(coords[[row, 0]]);
        values.push(coords[[row, 0]]);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-9);

    // The chart's own speed field, read through the module that DEFINES the
    // canonical chart — single source of truth for what "unit-speed" means here.
    let evaluator = term.atoms[0]
        .basis_evaluator
        .as_ref()
        .expect("periodic atom carries an installed evaluator")
        .clone();
    let topology = CanonicalChartTopology::Circle { period: 1.0 };
    let reading = chart_arclength_coordinates(
        evaluator.as_ref(),
        term.atoms[0].decoder_coefficients().view(),
        coords.column(0),
        &topology,
    )
    .expect("arc-length reading")
    .expect("non-degenerate chart");

    let rows: Vec<usize> = (0..N_ROWS).collect();
    let base_decode = term
        .steer_decode(0, &rows, array![0.0].view())
        .expect("unsteered decode");
    let recovery = AngleRecovery::calibrate(&base_decode, &ring.theta);

    let mut fidelity = 0.0_f64;
    for row in 0..N_ROWS {
        let point: Vec<f64> = (0..P_OUT).map(|j| base_decode[[row, j]]).collect();
        let gap = wrap_pi(recovery.theta_of(&point) - ring.theta[row]);
        let arc_gap = (ring.arc(ring.theta[row] + gap) - ring.arc(ring.theta[row])).abs();
        fidelity = fidelity.max(arc_gap);
    }

    let mut report = ChartReport {
        ev,
        span_turns: hi - lo,
        distinct: values.len(),
        speed_cv: reading.speed_cv,
        speed_ratio: reading.max_speed_over_mean / reading.min_speed_over_mean,
        fidelity_arc_floor: fidelity,
        orientation: 1.0,
        recovery,
        base_decode,
    };
    // Orientation probe: an eighth of a period is far from the antipode on every
    // fixture here, so the SIGN of the realized displacement is determined.
    let mut probe: Vec<f64> = (0..N_ROWS)
        .map(|row| realized_row_displacement(term, ring, &report, row, 0.125))
        .collect();
    probe.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if probe[N_ROWS / 2] < 0.0 {
        report.orientation = -1.0;
    }
    report
}

/// One requested displacement, measured over every row.
struct RealizedDisplacement {
    /// Per-row realized displacement, aligned to the representative closest to
    /// the request. A displacement on a closed ring lives in `R / L·Z`, so the
    /// ERROR has to be measured there too: at exactly half a turn the sign is not
    /// determined by the data (gh#2263 measured that split at `+6` of 12 and
    /// noted it reads as "the +6 steer is unstable" when it is the metric that is
    /// degenerate). Aligning removes the artifact without hiding a real miss —
    /// a row that genuinely failed to move is still a half-perimeter error.
    mean: f64,
    median: f64,
    lo: f64,
    hi: f64,
    worst: f64,
    worst_row: usize,
}

impl RealizedDisplacement {
    fn measure(
        term: &SaeManifoldTerm,
        ring: &PlantedRing,
        report: &ChartReport,
        raw_steps: &[f64],
        requested: f64,
    ) -> Self {
        let perimeter = ring.perimeter();
        let mut aligned = Vec::with_capacity(N_ROWS);
        let mut worst = 0.0_f64;
        let mut worst_row = 0usize;
        for row in 0..N_ROWS {
            let signed = report.orientation
                * realized_row_displacement(term, ring, report, row, raw_steps[row]);
            let gap = signed - requested;
            let wrapped = gap - perimeter * (gap / perimeter).round();
            aligned.push(requested + wrapped);
            if wrapped.abs() > worst {
                worst = wrapped.abs();
                worst_row = row;
            }
        }
        let mut sorted = aligned.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = aligned.iter().sum::<f64>() / N_ROWS as f64;
        Self {
            mean,
            median: sorted[N_ROWS / 2],
            lo: sorted[0],
            hi: sorted[N_ROWS - 1],
            worst,
            worst_row,
        }
    }
}

/// Requested displacements, as twelfths of the ring — the `+1..+6` range gh#2263
/// item 3 states its overshoot in, transplanted to a fixture where the realized
/// displacement has an external ground truth and no model is needed.
const TWELFTHS: [usize; 6] = [1, 2, 3, 4, 5, 6];

/// The three fixtures, by semi-minor axis (semi-major fixed at 1): the round
/// control, the discriminator, and a more anisotropic ring that shows the defect
/// tracking the chart's own speed ratio rather than being a property of one
/// fixture.
const RING_MINORS: [f64; 3] = [1.0, 0.3, 0.1];

/// The bar that SEPARATES a landed request from a gauge-arbitrary one: the worst
/// row's realized displacement must match the request to this fraction of the
/// request, and a raw-parameter steer on a non-unit-speed chart must FAIL it.
/// One constant, both directions asserted.
///
/// **Where the two sides sit.** Measured on the fixtures below: the canonical
/// arm's worst miss is `1.1e-7` (chart speed ratio 3.33) and `1.4e-5` (ratio 10),
/// i.e. 72× BELOW this bar at its worst; the raw arm's worst miss is `5.2e-1` and
/// `7.3e-1`, i.e. 520× ABOVE it. The bar sits in the middle of the five-order gap
/// rather than perched on either side of it.
///
/// **What stops the canonical arm going lower**, and why that is the instrument
/// rather than the steer: at `+6/12` the two arms apply the SAME step (half a
/// period is half the arc on a centrally symmetric ring, so the canonical and the
/// raw step coincide there) and report the SAME miss — `1.380e-7` on the 3.33
/// chart, `1.399e-5` on the 10 chart. A number both arms produce from one
/// identical step is a property of the measurement, not of either surface, and it
/// tracks each fixture's `fidelity_arc_floor`, printed beside it.
const SEPARATION_REL_TOL: f64 = 1.0e-3;

/// The instrument's null bar, applied only to the ROUND fixture, whose chart is
/// unit-speed to `speed_cv ≈ 7e-14`. There a requested displacement must come
/// back exactly, and the measured miss is `1.6e-12` of the request — so this bar
/// sits about 600× above what the instrument actually resolves.
const NULL_REL_TOL: f64 = 1.0e-9;

/// One `(fixture, step source)` sweep over the requested displacements.
struct Sweep {
    /// `(requested, mean, median, min, max, worst_abs, worst_row)`. Every
    /// displacement is in planted arc length.
    rows: Vec<(f64, f64, f64, f64, f64, f64, usize)>,
    worst_rel: f64,
    worst_at: usize,
}

fn sweep(
    term: &SaeManifoldTerm,
    ring: &PlantedRing,
    report: &ChartReport,
    canonical: bool,
) -> Sweep {
    let rows_idx: Vec<usize> = (0..N_ROWS).collect();
    let mut rows = Vec::with_capacity(TWELFTHS.len());
    let mut worst_rel = 0.0_f64;
    let mut worst_at = 0usize;
    for &k in &TWELFTHS {
        let fraction = k as f64 / 12.0;
        let requested = fraction * ring.perimeter();
        let raw_steps: Vec<f64> = if canonical {
            // The canonical surface picks the STEP; the ambient move is still
            // produced by the same group action, so the two arms differ in
            // nothing but the step.
            crate::inference::steering::steer_rows_unit_speed(term, 0, &rows_idx, fraction)
                .expect("canonical steer")
                .raw_steps
        } else {
            // What a caller passes today: the requested fraction of the period,
            // straight into the raw chart parameter.
            vec![fraction; N_ROWS]
        };
        let realized = RealizedDisplacement::measure(term, ring, report, &raw_steps, requested);
        let rel = realized.worst / requested;
        if rel > worst_rel {
            worst_rel = rel;
            worst_at = k;
        }
        rows.push((
            requested,
            realized.mean,
            realized.median,
            realized.lo,
            realized.hi,
            realized.worst,
            realized.worst_row,
        ));
    }
    Sweep {
        rows,
        worst_rel,
        worst_at,
    }
}

fn print_sweep(name: &str, arm: &str, s: &Sweep) {
    eprintln!("[#2234 {name}/{arm}] requested | realized mean | median | min | max | worst |err|");
    for (k, (requested, mean, median, lo, hi, worst, worst_row)) in
        TWELFTHS.iter().zip(s.rows.iter())
    {
        eprintln!(
            "[#2234 {name}/{arm}] +{k}/12 = {requested:.6} | {mean:.6} | {median:.6} | \
             {lo:.6} | {hi:.6} | {worst:.3e} ({:.3}%, row {worst_row})",
            100.0 * worst / requested
        );
    }
    eprintln!(
        "[#2234 {name}/{arm}] WORST over the sweep: {:.3e} of the request (at +{}/12)",
        s.worst_rel, s.worst_at
    );
}

fn print_chart(name: &str, ring: &PlantedRing, term: &SaeManifoldTerm, report: &ChartReport) {
    eprintln!(
        "[#2234 {name}] EV={:.6} span={:.4} turns, {} distinct coords, speed_cv={:.4e}, \
         speed max/min={:.4}, fidelity floor={:.3e} arc, perimeter={:.6}, orientation={:+.0}, \
         in-loop retraction committed={}",
        report.ev,
        report.span_turns,
        report.distinct,
        report.speed_cv,
        report.speed_ratio,
        report.fidelity_arc_floor,
        ring.perimeter(),
        report.orientation,
        term.atoms[0].chart_canonicalized
    );
}

/// **Fixture precondition, not a result** (round control arm): the chart must
/// recover the planted ring, span it, and — because the ring is round — be
/// unit-speed. If this arm's chart were NOT unit-speed, the measurement below
/// would have no null.
#[test]
fn round_ring_chart_is_unit_speed_and_faithful() {
    let ring = PlantedRing::new(1.0, 1.0);
    let term = fit_ring(&ring);
    let report = chart_report(&term, &ring);
    print_chart("round", &ring, &term, &report);
    assert!(
        report.ev > 0.99,
        "the round-ring fit must recover the planted circle before anything is measured on it \
         (EV={:.6})",
        report.ev
    );
    assert!(
        report.span_turns >= 0.8 && report.distinct >= N_ROWS,
        "the round-ring chart must span the circle and resolve every row (span={:.4} turns, \
         {} distinct coordinates for {N_ROWS} rows)",
        report.span_turns,
        report.distinct
    );
    assert!(
        report.speed_ratio < 1.001,
        "the round ring's chart is not unit-speed (speed max/min={:.6}); it is the null the \
         displacement measurement is read against",
        report.speed_ratio
    );
}

/// **Fixture precondition, not a result** (discriminator arm): the ellipse chart
/// must recover the planted ring AND still be non-unit-speed after the fit, or
/// the discriminator is vacuous. The bar is read off the planted geometry: a
/// `(1, 0.3)` ellipse has a decoder-curve speed ratio of `3.33`, so a chart that
/// carried the planted anisotropy shows a ratio near that; requiring `> 2`
/// refuses a fixture that lost most of it.
#[test]
fn ellipse_ring_chart_is_not_unit_speed() {
    let ring = PlantedRing::new(1.0, ELLIPSE_MINOR);
    let term = fit_ring(&ring);
    let report = chart_report(&term, &ring);
    print_chart("ellipse", &ring, &term, &report);
    assert!(
        report.ev > 0.99,
        "the ellipse fit must recover the planted ring before anything is measured on it \
         (EV={:.6})",
        report.ev
    );
    assert!(
        report.span_turns >= 0.8 && report.distinct >= N_ROWS,
        "the ellipse chart must span the ring and resolve every row (span={:.4} turns, \
         {} distinct coordinates for {N_ROWS} rows)",
        report.span_turns,
        report.distinct
    );
    assert!(
        report.speed_ratio > 2.0,
        "the ellipse chart lost the planted 1:{ELLIPSE_MINOR} anisotropy (speed max/min={:.4}); \
         a unit-speed chart here would make the displacement discriminator vacuous",
        report.speed_ratio
    );
}

/// **THE MEASUREMENT.** A caller asks to advance a row by `k/12` of the way
/// around its feature — the request gh#2263 item 3 is stated in — and passes that
/// fraction as the chart step, which is exactly what
/// [`crate::manifold::SaeManifoldTerm::steer_rows`] documents it as. Does the row
/// move `k/12` of the ring?
///
/// Every arm on every fixture is measured and PRINTED before any assertion
/// fires, so a failure never suppresses the numbers that explain it. Three
/// fixtures × two step sources:
///
/// * round / raw — the measured null. The chart is unit-speed, so this arm must
///   land the request, and what it misses by IS the instrument's resolution.
/// * round / canonical — the no-op control. On a unit-speed chart the canonical
///   surface must reproduce the raw one, or its success elsewhere could be a
///   second gauge rather than a repair.
/// * ellipse (and the more anisotropic ring) / raw — the defect, with its size
///   tracking the chart's measured speed ratio rather than the fixture.
/// * ellipse / canonical — the repair.
#[test]
fn requested_ring_fraction_is_realized_only_in_the_canonical_chart() {
    let mut ladder: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for &minor in &RING_MINORS {
        let name = format!("minor{minor}");
        let ring = PlantedRing::new(1.0, minor);
        let term = fit_ring(&ring);
        let report = chart_report(&term, &ring);
        let raw = sweep(&term, &ring, &report, false);
        let canonical = sweep(&term, &ring, &report, true);
        print_chart(&name, &ring, &term, &report);
        print_sweep(&name, "raw", &raw);
        print_sweep(&name, "canonical", &canonical);
        // Read the fixture's own preconditions AFTER printing everything, so a
        // degenerate fit is diagnosed rather than silently scored.
        assert!(
            report.ev > 0.99 && report.span_turns >= 0.8 && report.distinct >= N_ROWS,
            "[{name}] the fit is not a usable chart: EV={:.6}, span={:.4} turns, \
             {} distinct coordinates for {N_ROWS} rows",
            report.ev,
            report.span_turns,
            report.distinct
        );
        ladder.push((
            minor,
            report.speed_cv,
            report.speed_ratio,
            raw.worst_rel,
            canonical.worst_rel,
        ));
    }

    eprintln!(
        "[#2234 ladder] minor | chart speed_cv | speed max/min | raw worst | canonical worst"
    );
    for (minor, cv, ratio, raw, canonical) in &ladder {
        eprintln!(
            "[#2234 ladder] {minor:.2} | {cv:.4e} | {ratio:.4} | {raw:.3e} | {canonical:.3e}"
        );
    }

    let (_, _, _, round_raw, round_canonical) = ladder[0];
    // The measured null: on a provably unit-speed chart the raw step IS the
    // canonical step, so this arm bounds what the instrument can resolve. If it
    // failed, no other number in this test would mean anything.
    assert!(
        round_raw <= NULL_REL_TOL,
        "the instrument's own null failed: on a unit-speed chart a requested displacement missed \
         by {round_raw:.3e} of the request, above the {NULL_REL_TOL:e} bar. Nothing else measured \
         here is readable until this passes"
    );
    // No-op control, denominated in the SAME fixture's raw arm rather than in a
    // constant: the canonical surface must not disturb an already canonical
    // chart, or its success on a curved chart could be a second gauge rather than
    // a repair.
    assert!(
        round_canonical <= 10.0 * round_raw,
        "the canonical surface DISTURBED an already unit-speed chart: it missed by \
         {round_canonical:.3e} of the request where the raw surface on the same fixture missed by \
         {round_raw:.3e}"
    );

    for &(minor, cv, ratio, raw, canonical) in &ladder {
        // THE RESULT: the same request, landed on every chart in the ladder.
        assert!(
            canonical <= SEPARATION_REL_TOL,
            "[minor={minor}] the canonical surface did not land the requested displacement: \
             worst row missed by {canonical:.3e} of the request, above the \
             {SEPARATION_REL_TOL:e} bar. Chart speed_cv={cv:.4e}, speed max/min={ratio:.4}; the \
             round null on the same instrument is {round_raw:.3e}"
        );
        if minor == 1.0 {
            continue;
        }
        // Anti-no-op control, and the other side of the SAME bar. If
        // `steer_rows_unit_speed` were secretly returning the raw step, every
        // assertion above would still pass on the round arm; a non-unit-speed
        // chart must show the two surfaces genuinely disagreeing across the bar.
        assert!(
            raw > SEPARATION_REL_TOL,
            "[minor={minor}] the fixture stopped discriminating: a RAW-parameter steer missed the \
             requested displacement by only {raw:.3e} of the request, below the \
             {SEPARATION_REL_TOL:e} bar the canonical arm is held to, so the canonical arm's \
             success is not evidence of a repair. Chart speed max/min={ratio:.4}"
        );
    }
}
