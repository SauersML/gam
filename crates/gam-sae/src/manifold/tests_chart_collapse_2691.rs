//! #2691 — the chart-collapse boundary sweep.
//!
//! `tests_steering_e4.rs` recovers a planted circle at R² ≈ 1 with
//! `n=240, p=6`, noise-free. The #2691 reproducer collapses to a CONSTANT
//! `d_atom=1` coordinate (std 1.06e-14, one distinct value over 70 rows) at
//! `n=70, p=8` with isotropic noise — and certifies it. This module locates the
//! boundary between those two regimes in the pure-Rust harness (no model, no
//! wheel, no GPU) so any hypothesis about the cause has a seconds-scale
//! reproducer.
//!
//! The measured quantity is the FITTED CHART COORDINATE itself, never `fit_ev`:
//! the issue measures the collapsed arm's EV (0.0581) BELOW a partially
//! collapsed arm's (0.0883), so an EV-denominated diagnostic provably cannot
//! see this defect.

#![cfg(test)]

use super::tests::{deterministic_circle_noise, global_ev};
use super::tests_startup_validation_1782::{Topo, build_term};
use super::*;
use ndarray::{Array1, Array2, array};

/// A planted circle of radius `radius` in a generic 2-plane of `R^p`, plus
/// isotropic noise of RMS `sigma`, `n` rows. This is the #2691 reproducer's
/// generator expressed in the Rust harness.
pub(crate) struct PlantedCircle {
    pub(crate) z: Array2<f64>,
    pub(crate) theta: Array1<f64>,
}

/// The deterministic orthonormal 2-frame in `R^p` the ring is planted in
/// (Gram-Schmidt), so the planted ring is an exact circle and the sweep needs no
/// RNG. Shared by the generator and by anything that has to read a fit back out
/// in the planted plane, so the two cannot drift apart.
pub(crate) fn planted_frame(p: usize) -> (Array1<f64>, Array1<f64>) {
    let mut u = Array1::<f64>::zeros(p);
    let mut v = Array1::<f64>::zeros(p);
    for j in 0..p {
        u[j] = ((j as f64 + 1.0) * 0.7).sin();
        v[j] = ((j as f64 + 1.0) * 0.7).cos();
    }
    let un = u.dot(&u).sqrt();
    u.mapv_inplace(|x| x / un);
    let uv = u.dot(&v);
    for j in 0..p {
        v[j] -= uv * u[j];
    }
    let vn = v.dot(&v).sqrt();
    v.mapv_inplace(|x| x / vn);
    (u, v)
}

pub(crate) fn planted_circle_cloud(n: usize, p: usize, radius: f64, sigma: f64) -> PlantedCircle {
    let (u, v) = planted_frame(p);

    let theta =
        Array1::<f64>::from_shape_fn(n, |i| std::f64::consts::TAU * (i as f64 + 0.5) / n as f64);
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let (c, s) = (theta[i].cos(), theta[i].sin());
        for j in 0..p {
            z[[i, j]] =
                radius * (c * u[j] + s * v[j]) + sigma * deterministic_circle_noise(i, j);
        }
    }
    PlantedCircle { z, theta }
}

/// Squared circular correlation between a fitted period-1 chart coordinate and
/// the known generating phase, maximised over both orientations — the #2691
/// "recovery R²". It cannot be low for a units or a sign reason.
pub(crate) fn circular_recovery_r2(coord: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let n = coord.len();
    assert_eq!(n, theta.len());
    let mut best = 0.0_f64;
    for orientation in [1.0_f64, -1.0_f64] {
        // Regress (cos θ, sin θ) on (cos 2πt, sin 2πt) — equivalently, the
        // squared modulus of the mean of exp(i(θ ∓ 2πt)) is the circular
        // correlation of the two phases up to an unknown offset φ.
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let phase = theta[i] - orientation * std::f64::consts::TAU * coord[i];
            re += phase.cos();
            im += phase.sin();
        }
        re /= n as f64;
        im /= n as f64;
        best = best.max(re * re + im * im);
    }
    best
}

pub(crate) struct ChartOutcome {
    pub(crate) ev: f64,
    /// Raw f64 standard deviation of the coordinate — what the #2691 ledger
    /// reports. On a period-1 circle this is NOT the chart's dispersion: `0.0`
    /// and `1.0` are the SAME point, so a fully collapsed chart can read
    /// `std ≈ 0.5` here.
    pub(crate) coord_std: f64,
    /// Circular variance `1 − |mean exp(2πi t)|` — the chart's dispersion IN
    /// ITS OWN MANIFOLD. `0` ⇒ every row sits on one point of the circle.
    pub(crate) circular_variance: f64,
    pub(crate) distinct: usize,
    /// Distinct coordinate values AFTER wrapping to `[0, 1)` and rounding to
    /// 1e-9 — the number of genuinely distinct chart points.
    pub(crate) distinct_wrapped: usize,
    pub(crate) recovery_r2: f64,
    /// Final per-atom ARD log-precision on the chart axis (the von-Mises
    /// coordinate prior's precision, `atom.rs::ArdAxisPrior`).
    pub(crate) log_ard: f64,
    pub(crate) refusal: Option<String>,
}

/// Fit the #2691 call shape (`K=1`, `d_atom=1`, circle, softmax) in the Rust
/// harness and report the CHART's own health, never the reconstruction EV.
pub(crate) fn fit_and_measure_chart(cloud: &PlantedCircle, n_iter: usize) -> ChartOutcome {
    let z = &cloud.z;
    let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
    let mut rho = SaeManifoldRho::new(
        1.0e-3_f64.ln(),
        1.0e-3_f64.ln(),
        vec![array![1.0e-3_f64.ln()]; 1],
    );
    if let Err(error) =
        term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, n_iter, 1.0, 1.0e-6, 1.0e-6)
    {
        return ChartOutcome {
            ev: f64::NAN,
            coord_std: f64::NAN,
            circular_variance: f64::NAN,
            distinct: 0,
            distinct_wrapped: 0,
            recovery_r2: f64::NAN,
            log_ard: rho.log_ard[0][0],
            refusal: Some(format!("{error}")),
        };
    }
    let ev = term
        .try_fitted()
        .map(|fitted| global_ev(z.view(), fitted.view()))
        .unwrap_or(f64::NAN);
    let coords = term.assignment.coords[0].as_matrix();
    let coord: Array1<f64> = coords.column(0).to_owned();
    let n = coord.len() as f64;
    let mean = coord.iter().sum::<f64>() / n;
    let coord_std = (coord.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
    // Circular dispersion in the chart's OWN manifold: `t` and `t + 1` are the
    // same point of a period-1 circle, so a raw std cannot decide whether the
    // chart is degenerate.
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in coord.iter() {
        let phase = std::f64::consts::TAU * t;
        re += phase.cos();
        im += phase.sin();
    }
    re /= n;
    im /= n;
    let circular_variance = 1.0 - (re * re + im * im).sqrt();
    let mut sorted: Vec<f64> = coord.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    let distinct = sorted.len();
    let mut wrapped: Vec<i64> = coord
        .iter()
        .map(|&t| (t.rem_euclid(1.0) * 1.0e9).round() as i64 % 1_000_000_000)
        .collect();
    wrapped.sort_unstable();
    wrapped.dedup();
    let distinct_wrapped = wrapped.len();
    let recovery_r2 = circular_recovery_r2(&coord, &cloud.theta);
    ChartOutcome {
        ev,
        coord_std,
        circular_variance,
        distinct,
        distinct_wrapped,
        recovery_r2,
        log_ard: rho.log_ard[0][0],
        refusal: None,
    }
}

/// The 2-D (n × p) × noise boundary sweep. Diagnostic — it prints a table and
/// asserts nothing about the interior, because the interior is the thing under
/// investigation. Run with `--nocapture` to read the table.
#[test]
fn zz_2691_chart_collapse_boundary_sweep() {
    // The whole grid, fixed: 48 fits in 2.4 s. `radius` and `sigma` are the
    // #2691 reproducer's own generator constants (circle of radius 2.086,
    // isotropic noise RMS 0.352).
    let ns: Vec<usize> = vec![60, 70, 90, 120, 180, 240];
    let ps: Vec<usize> = vec![4, 6, 8, 12];
    let sigmas: Vec<f64> = vec![0.0, 0.352];
    let radius: f64 = 2.086;

    eprintln!(
        "[2691-sweep] n\tp\tsigma\tEV\tcoord_std\tcirc_var\tdistinct\tdistinct_wrapped\t\
         recovery_R2\tlog_ard\tsecs\trefusal"
    );
    let mut worst: Option<(usize, usize, f64, f64)> = None;
    for &n in &ns {
        for &p in &ps {
            for &sigma in &sigmas {
                let cloud = planted_circle_cloud(n, p, radius, sigma);
                let start = std::time::Instant::now();
                let outcome = fit_and_measure_chart(&cloud, 40);
                let secs = start.elapsed().as_secs_f64();
                match &outcome.refusal {
                    Some(message) => {
                        let first = message.lines().next().unwrap_or("").to_string();
                        eprintln!(
                            "[2691-sweep] {n}\t{p}\t{sigma:.3}\t-\t-\t-\t-\t-\t-\t{:.3}\t\
                             {secs:.1}\tREFUSED: {first}",
                            outcome.log_ard
                        );
                    }
                    None => eprintln!(
                        "[2691-sweep] {n}\t{p}\t{sigma:.3}\t{:.4}\t{:.3e}\t{:.3e}\t{}/{n}\t{}/{n}\t\
                         {:.4}\t{:.3}\t{secs:.1}\t-",
                        outcome.ev,
                        outcome.coord_std,
                        outcome.circular_variance,
                        outcome.distinct,
                        outcome.distinct_wrapped,
                        outcome.recovery_r2,
                        outcome.log_ard
                    ),
                }
                if outcome.refusal.is_none()
                    && worst.is_none_or(|(_, _, _, r2)| outcome.recovery_r2 < r2)
                {
                    worst = Some((n, p, sigma, outcome.recovery_r2));
                }
            }
        }
    }
    // The bar the sweep earns: at a FIXED, seeded ρ the inner joint fit
    // recovers the planted circle EVERYWHERE on this grid — measured
    // 0.9793..1.0000 across n ∈ [60, 240] × p ∈ [4, 12] × σ ∈ {0, 0.352}. There
    // is no n × p boundary between E4's regime and #2691's, so a future
    // regression that reintroduces one fails here.
    let (n, p, sigma, r2) = worst.expect("the sweep must fit at least one cell");
    assert!(
        r2 > 0.9,
        "#2691: the inner joint fit must recover the planted circle at a fixed ρ;          worst cell n={n} p={p} sigma={sigma} recovery R²={r2:.4}"
    );
}

/// A planted ordinal LINE in a generic direction of `R^p` plus isotropic noise
/// — the companion #2691 fixture for `atom_topology="euclidean"`, `d_atom=1`,
/// which the issue reports refusing at every chart dimension with
/// `dense exact-stationarity pseudoinverse failed certification`.
pub(crate) fn planted_line_cloud(n: usize, p: usize, span: f64, sigma: f64) -> PlantedCircle {
    let mut u = Array1::<f64>::zeros(p);
    for j in 0..p {
        u[j] = ((j as f64 + 1.0) * 0.7).sin();
    }
    let un = u.dot(&u).sqrt();
    u.mapv_inplace(|x| x / un);
    // `theta` here carries the planted ordinal position, not an angle.
    let theta = Array1::<f64>::from_shape_fn(n, |i| (i as f64 + 0.5) / n as f64 - 0.5);
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] = span * theta[i] * u[j] + sigma * deterministic_circle_noise(i, j);
        }
    }
    PlantedCircle { z, theta }
}

/// The euclidean/ordinal companion sweep. Reports whether the fit refuses and
/// with what, plus the linear recovery R² of the fitted chart coordinate
/// against the planted ordinal position. It grades nothing about the table; it
/// asserts only that every row is readable (a refusal names its reason, a fit
/// carries a finite EV and chart, and R² is a squared correlation).
#[test]
fn zz_2691_euclidean_line_refusal_sweep() {
    // Every chart dimension the issue reports the euclidean arm refusing at.
    let ns: Vec<usize> = vec![70];
    let ps: Vec<usize> = vec![3, 4, 6, 8, 12, 16];
    let sigmas: Vec<f64> = vec![0.0, 0.352];

    eprintln!("[2691-line] n\tp\tsigma\tEV\tcoord_std\trecovery_R2\tsecs\trefusal");
    for &n in &ns {
        for &p in &ps {
            for &sigma in &sigmas {
                let cloud = planted_line_cloud(n, p, 4.0, sigma);
                let z = &cloud.z;
                let (mut term, _disp) =
                    build_term(z.view(), 1, Topo::Euclidean, AssignmentMode::softmax(1.0));
                let mut rho = SaeManifoldRho::new(
                    1.0e-3_f64.ln(),
                    1.0e-3_f64.ln(),
                    vec![array![1.0e-3_f64.ln()]; 1],
                );
                let start = std::time::Instant::now();
                let result = term.run_joint_fit_arrow_schur(
                    z.view(),
                    &mut rho,
                    None,
                    40,
                    1.0,
                    1.0e-6,
                    1.0e-6,
                );
                let secs = start.elapsed().as_secs_f64();
                match result {
                    Err(error) => {
                        let text = format!("{error}").replace('\n', " ");
                        assert!(
                            !text.trim().is_empty(),
                            "a refusal row must name its reason (n={n}, p={p}, sigma={sigma})"
                        );
                        eprintln!("[2691-line] {n}\t{p}\t{sigma:.3}\t-\t-\t-\t{secs:.1}\tREFUSED: {text}");
                    }
                    Ok(_) => {
                        let ev = term
                            .try_fitted()
                            .map(|fitted| global_ev(z.view(), fitted.view()))
                            .unwrap_or(f64::NAN);
                        let coords = term.assignment.coords[0].as_matrix();
                        let coord: Array1<f64> = coords.column(0).to_owned();
                        let nn = coord.len() as f64;
                        let cm = coord.iter().sum::<f64>() / nn;
                        let tm = cloud.theta.iter().sum::<f64>() / nn;
                        let (mut sxy, mut sxx, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
                        for i in 0..coord.len() {
                            let a = coord[i] - cm;
                            let b = cloud.theta[i] - tm;
                            sxy += a * b;
                            sxx += a * a;
                            syy += b * b;
                        }
                        let r2 = if sxx > 0.0 && syy > 0.0 {
                            sxy * sxy / (sxx * syy)
                        } else {
                            0.0
                        };
                        let std = (sxx / nn).sqrt();
                        assert!(
                            ev.is_finite() && coord.iter().all(|value| value.is_finite()),
                            "a fit that returned Ok must carry a finite EV and chart \
                             (n={n}, p={p}, sigma={sigma}, ev={ev})"
                        );
                        assert!(
                            (0.0..=1.0 + 1.0e-12).contains(&r2),
                            "the recovery R² is a squared correlation \
                             (n={n}, p={p}, sigma={sigma}, r2={r2})"
                        );
                        eprintln!(
                            "[2691-line] {n}\t{p}\t{sigma:.3}\t{ev:.4}\t{std:.3e}\t{r2:.4}\t{secs:.1}\t-"
                        );
                    }
                }
            }
        }
    }
}

/// The FULL production entry, the way `gamfit.sae_manifold_fit` reaches it:
/// minimal seed → fit seed → `run_sae_manifold_fit` with
/// `run_outer_rho_search: true`. This is the only structural difference from
/// [`fit_and_measure_chart`], which drives the inner joint fit alone at a FIXED
/// ρ — and the inner-only sweep recovers the planted circle at R² ≥ 0.98
/// everywhere on `n ∈ [60, 240] × p ∈ [4, 12] × σ ∈ {0, 0.352}`. So whatever
/// #2691 is, it lives in the outer ρ search or the production seed, not in the
/// joint Newton.
#[test]
fn zz_2691_outer_path_chart_collapse_sweep() {
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitRequest, SaeFitSeedReport, SaeFitSeedRequest,
        SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed, build_sae_minimal_seed,
        run_sae_manifold_fit,
    };
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;

    // ONE cell: the outer path costs tens of seconds a fit, and the point this
    // test carries is that ρ moves `log_ard` at all (measured +1.341 here, and
    // -11.729 one chart dimension away) — not a grid. The σ ≥ 1.5 cells that
    // rail it to the outer box are on the issue with their wall times; they are
    // 300-500 s apiece and do not belong in a routine suite.
    let ns: Vec<usize> = vec![70];
    let ps: Vec<usize> = vec![8];
    let sigmas: Vec<f64> = vec![0.352];
    let radius: f64 = 2.086;

    eprintln!(
        "[2691-outer] n\tp\tsigma\tR2rec\tcoord_std\tcirc_var\tdistinct_wrapped\trecovery_R2\t\
         log_ard\tsecs\trefusal"
    );
    for &n in &ns {
        for &p in &ps {
            for &sigma in &sigmas {
                let cloud = planted_circle_cloud(n, p, radius, sigma);
                let target = cloud.z.clone();
                let start = std::time::Instant::now();
                let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
                    target: target.view(),
                    atom_basis: vec!["periodic".to_string()],
                    atom_dim: vec![1],
                    assignment_kind: SaeFitAssignmentKind::Softmax,
                    alpha: 1.0,
                    tau: 1.0,
                    threshold: 0.0,
                    top_k: None,
                    random_state: 20260731,
                    initial_logits: None,
                    initial_coords: None,
                })
                .expect("minimal seed");
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
                    target: target.view(),
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
                    assignment_kind: SaeFitAssignmentKind::Softmax,
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
                    seed_refine_random_state: 20260731,
                    data_row_reseed: false,
                    fit_config: SaeFitConfig::default(),
                    temperature_schedule: None,
                    fisher_metric: None,
                    row_loss_weights: None,
                    registry: &registry,
                })
                .expect("fit seed");
                let SaeFitSeedReport {
                    base_term,
                    initial_rho,
                    isometry_pin_active,
                    metric_provenance,
                } = seed;
                let outcome = run_sae_manifold_fit(SaeFitRequest {
                    reconstruction_optimism_folds: None,
                    base_term,
                    target: target.clone(),
                    registry,
                    initial_rho,
                    max_iter: 40,
                    learning_rate: 1.0,
                    ridge_ext_coord: 1.0e-6,
                    ridge_beta: 1.0e-6,
                    alpha: 1.0,
                    isometry_pin_active,
                    metric_provenance,
                    promote_from_residual: false,
                    run_structure_search: false,
                    run_outer_rho_search: true,
                    structured_residual_passes: 0,
                    cancel: None,
                });
                let secs = start.elapsed().as_secs_f64();
                let report = match outcome
                    .map_err(|error| format!("{error}"))
                    .and_then(|o| o.manifold_or_error())
                {
                    Ok(report) => report,
                    Err(error) => {
                        let text = error.replace('\n', " ");
                        eprintln!(
                            "[2691-outer] {n}\t{p}\t{sigma:.3}\t-\t-\t-\t-\t-\t-\t{secs:.1}\t\
                             REFUSED: {text}"
                        );
                        continue;
                    }
                };
                let coords = report.term.assignment.coords[0].as_matrix();
                let coord: Array1<f64> = coords.column(0).to_owned();
                let nn = coord.len() as f64;
                let mean = coord.iter().sum::<f64>() / nn;
                let coord_std =
                    (coord.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / nn).sqrt();
                let (mut re, mut im) = (0.0_f64, 0.0_f64);
                for &t in coord.iter() {
                    let phase = std::f64::consts::TAU * t;
                    re += phase.cos();
                    im += phase.sin();
                }
                re /= nn;
                im /= nn;
                let circular_variance = 1.0 - (re * re + im * im).sqrt();
                let mut wrapped: Vec<i64> = coord
                    .iter()
                    .map(|&t| (t.rem_euclid(1.0) * 1.0e9).round() as i64 % 1_000_000_000)
                    .collect();
                wrapped.sort_unstable();
                wrapped.dedup();
                let recovery_r2 = circular_recovery_r2(&coord, &cloud.theta);
                eprintln!(
                    "[2691-outer] {n}\t{p}\t{sigma:.3}\t{:.4}\t{coord_std:.3e}\t\
                     {circular_variance:.3e}\t{}/{n}\t{recovery_r2:.4}\t{:.3}\t{secs:.1}\t-",
                    report.reconstruction_r2,
                    wrapped.len(),
                    report.rho.log_ard[0][0]
                );
            }
        }
    }
}

/// TEST A CONSTANT BY WHETHER IT VARIES. The inner sweep above holds the ARD
/// log-precision fixed at `ln 1e-3` and never collapses. The von-Mises
/// coordinate prior (`atom.rs::ArdAxisPrior`) is `V(t) = (α/κ²)(1 − cos κt)`,
/// whose UNIQUE minimum is the chart origin `t = 0`; its precision `α` is an
/// outer coordinate the ρ search is free to raise. This ladder asks the only
/// question that matters: does raising `α` alone, at a fixture the inner solver
/// recovers perfectly, drive the chart to a constant?
#[test]
fn zz_2691_ard_precision_ladder_collapses_the_chart() {
    let n: usize = 70;
    let p: usize = 8;
    let cloud = planted_circle_cloud(n, p, 2.086, 0.352);
    let z = &cloud.z;
    let ladder: Vec<f64> = vec![
        1.0e-3_f64.ln(),
        1.0_f64.ln(),
        1.0e1_f64.ln(),
        1.0e2_f64.ln(),
        1.0e3_f64.ln(),
        1.0e4_f64.ln(),
        1.0e6_f64.ln(),
        1.0e9_f64.ln(),
    ];
    eprintln!("[2691-ard] log_ard\talpha\tEV\tcoord_std\tcirc_var\tdistinct_wrapped\trecovery_R2");
    let mut rungs: Vec<(f64, f64, usize)> = Vec::new();
    for &log_ard in &ladder {
        let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let mut rho =
            SaeManifoldRho::new(1.0e-3_f64.ln(), 1.0e-3_f64.ln(), vec![array![log_ard]; 1]);
        let fit =
            term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6);
        if let Err(error) = fit {
            let text = format!("{error}").replace('\n', " ");
            eprintln!("[2691-ard] {log_ard:.3}\t{:.3e}\tREFUSED: {text}", log_ard.exp());
            continue;
        }
        let ev = term
            .try_fitted()
            .map(|fitted| global_ev(z.view(), fitted.view()))
            .unwrap_or(f64::NAN);
        let coords = term.assignment.coords[0].as_matrix();
        let coord: Array1<f64> = coords.column(0).to_owned();
        let nn = coord.len() as f64;
        let mean = coord.iter().sum::<f64>() / nn;
        let coord_std = (coord.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / nn).sqrt();
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for &t in coord.iter() {
            let phase = std::f64::consts::TAU * t;
            re += phase.cos();
            im += phase.sin();
        }
        re /= nn;
        im /= nn;
        let circular_variance = 1.0 - (re * re + im * im).sqrt();
        let mut wrapped: Vec<i64> = coord
            .iter()
            .map(|&t| (t.rem_euclid(1.0) * 1.0e9).round() as i64 % 1_000_000_000)
            .collect();
        wrapped.sort_unstable();
        wrapped.dedup();
        let recovery_r2 = circular_recovery_r2(&coord, &cloud.theta);
        eprintln!(
            "[2691-ard] {log_ard:.3}\t{:.3e}\t{ev:.4}\t{coord_std:.3e}\t{circular_variance:.3e}\t\
             {}/{n}\t{recovery_r2:.4}",
            log_ard.exp(),
            wrapped.len()
        );
        rungs.push((log_ard.exp(), circular_variance, wrapped.len()));
    }
    // The mechanism, pinned. Nothing but α moved between these two rungs.
    if let (Some(low), Some(high)) = (
        rungs.iter().find(|(alpha, _, _)| *alpha <= 1.0e-3),
        rungs.iter().find(|(alpha, _, _)| *alpha >= 1.0e9),
    ) {
        assert!(
            low.1 > 0.5,
            "#2691: at α=1e-3 the chart must still be spread over the circle              (circular variance {:.3e})",
            low.1
        );
        assert!(
            high.1 < 1.0e-6 && high.2 <= 1,
            "#2691: raising ONLY the von-Mises coordinate precision to α=1e9 must collapse the              chart to one point — this is the mechanism the guard exists for              (circular variance {:.3e}, {} resolved chart points)",
            high.1,
            high.2
        );
    }
}

/// #2691 REGRESSION — the production entry must REFUSE a chart that has
/// collapsed to a single point of its own manifold, instead of returning a
/// certified fit whose coordinate is a constant.
///
/// The fixture is the ARD ladder's terminal rung driven through the REAL entry
/// (`run_sae_manifold_fit`, the same function `gamfit.sae_manifold_fit` calls),
/// at a FIXED ρ so the collapse is placed by construction rather than waited
/// for: `α = 1e9` on the periodic chart axis. At the parent commit this call
/// returns `Ok` with `coord_std = 5.000e-1`, one distinct chart point over 70
/// rows, and a healthy-looking report — the exact #2691 `ideal` arm. Here it
/// must be a typed `SaeFitError::DegenerateChart`.
///
/// The bar is deliberately NOT denominated in explained variance: #2691
/// measured the fully collapsed arm's EV (0.0581) BELOW a partially collapsed
/// arm's (0.0883), so an EV gate passes every one of these states.
#[test]
fn zz_2691_collapsed_chart_is_refused_by_the_production_entry() {
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitError, SaeFitRequest, SaeFitSeedReport,
        SaeFitSeedRequest, SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed,
        build_sae_minimal_seed, run_sae_manifold_fit,
    };
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;

    let cloud = planted_circle_cloud(70, 8, 2.086, 0.352);
    let target = cloud.z.clone();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: target.view(),
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![1],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 20260731,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
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
        target: target.view(),
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
        assignment_kind: SaeFitAssignmentKind::Softmax,
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
        seed_refine_random_state: 20260731,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    let SaeFitSeedReport {
        base_term,
        mut initial_rho,
        isometry_pin_active,
        metric_provenance,
    } = seed;
    // The collapse lever, and the ONLY thing changed from a healthy call: the
    // origin-anchored von-Mises coordinate prior's precision on the chart axis.
    for axis in initial_rho.log_ard.iter_mut() {
        axis.fill(1.0e9_f64.ln());
    }

    let outcome = run_sae_manifold_fit(SaeFitRequest {
        reconstruction_optimism_folds: None,
        base_term,
        target,
        registry,
        initial_rho,
        max_iter: 40,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        alpha: 1.0,
        isometry_pin_active,
        metric_provenance,
        promote_from_residual: false,
        run_structure_search: false,
        // Fixed ρ: the collapse is PLACED, not waited for, so this reproducer
        // runs in seconds and cannot be a flaky search outcome.
        run_outer_rho_search: false,
        structured_residual_passes: 0,
        cancel: None,
    });

    match outcome {
        Err(SaeFitError::DegenerateChart {
            atoms,
            evidence,
            report,
        }) => {
            eprintln!("[2691-regression] refused as required: atoms={atoms:?} {evidence}");
            assert_eq!(
                atoms,
                vec![0],
                "the single K=1 atom must be the one named as chart-less"
            );
            assert!(
                report.axes.iter().all(|axis| axis.resolved_points <= 1),
                "the collapsed chart axis must resolve at most one chart point; got {:?}",
                report
                    .axes
                    .iter()
                    .map(|axis| axis.resolved_points)
                    .collect::<Vec<_>>()
            );
        }
        Err(other) => panic!(
            "#2691: a chart collapsed to one point must refuse as DegenerateChart, got: {other}"
        ),
        Ok(outcome) => {
            let report = outcome
                .manifold_or_error()
                .expect("outcome carried a manifold");
            let coords = report.term.assignment.coords[0].as_matrix();
            let coord: Array1<f64> = coords.column(0).to_owned();
            let chart = report.term.chart_degeneracy_report();
            panic!(
                "#2691 REGRESSION: the production entry certified a collapsed chart. \
                 reconstruction_r2={:.4}, chart dispersion={:?}, resolved chart points={:?}, \
                 first coordinates={:?}",
                report.reconstruction_r2,
                chart
                    .axes
                    .iter()
                    .map(|axis| axis.dispersion)
                    .collect::<Vec<_>>(),
                chart
                    .axes
                    .iter()
                    .map(|axis| axis.resolved_points)
                    .collect::<Vec<_>>(),
                coord.iter().take(5).collect::<Vec<_>>()
            );
        }
    }
}


// #2691 — the euclidean/ordinal arm through the FULL production entry is NOT a
// test here: one point of it (n=70, p=3, σ=0.352, `run_outer_rho_search: true`)
// costs 833 s, which does not belong in a routine suite. It was measured once
// and the result is on the issue: the fitted chart came back with standard
// deviation EXACTLY 0.000000e0 and ONE resolved point over 70 rows — the same
// collapse as the circle, via the Euclidean coordinate prior `½αt²`, whose
// minimum is also the chart origin. The cheap inner-path arm above
// (`zz_2691_euclidean_line_refusal_sweep`, 0.1 s a cell) stands as the standing
// witness that the inner solver is not the one that loses the chart.

// #2691 — the CAUSE-side fix (an evidence rail on the outer search's ARD
// chart-coordinate box, so ρ cannot drive the von-Mises precision past the
// point at which the prior owns the coordinate) is NOT in this commit. A first
// attempt was written and measured: the rail it computes was never invoked on
// the σ=1.5 witness (`ard_evidence_domain_upper` emitted zero entries, so its
// own debug line never fired), which means the hook it was placed on is not the
// one the SAE outer search consults. A constraint whose only evidence is that
// something ELSE caught the case is not a fix, so it is held rather than landed.
// What ships here is the reproducer, the mechanism, and the refusal that makes
// the collapse impossible to return silently.
//
// FOLLOW-UP: the cause-side face now exists —
// `outer_objective::periodic_ard_chart_resolution_upper`, installed on BOTH
// exits of `outer_domain_upper_bound`, which IS the hook the SAE outer search
// consults (`fit_outer_stage_to_boundary` → `OuterProblem::run` →
// `rho_optimizer::run::install_objective_domain`). Three tests below gate it:
// the MECHANISM (the face's value moves with the data geometry it is derived
// from), the PROPERTY (the face separates precisions this fixture measurably
// survives from ones it does not), and a BOUNDED σ witness that returns an
// answer at every σ instead of nothing at the σ that matters.

/// Build the #2691 K=1 periodic objective for a planted circle and return
/// `(installed ARD face, seeded ARD entry, chart period)`. The face is READ BACK
/// OUT of `outer_domain_upper_bound()`, so it is what the search is handed, not
/// a number recomputed alongside it.
fn ard_face_for(n: usize, p: usize, radius: f64, sigma: f64) -> (f64, f64, f64) {
    use gam_solve::rho_optimizer::OuterObjective;
    let cloud = planted_circle_cloud(n, p, radius, sigma);
    let z = &cloud.z;
    let (term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
    let period = term.assignment.coords[0].effective_axis_periods()[0]
        .expect("the circle chart axis must carry a period");
    // Canonicalize the flat layout against the term's assignment family FIRST —
    // exactly as `fit_outer_stage_to_boundary` does — so the flat index read here
    // is the one the objective uses (a K=1 softmax dictionary has no
    // sparse/router coordinate, which shifts the layout).
    let rho = SaeManifoldRho::new(
        1.0e-3_f64.ln(),
        1.0e-3_f64.ln(),
        vec![array![1.0e-3_f64.ln()]; 1],
    )
    .for_assignment(term.assignment.mode);
    let ard_index = rho.ard_flat_index(0, 0);
    let seed = rho.to_flat()[ard_index];
    let objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, rho, 40, 1.0, 1.0e-6, 1.0e-6);
    let upper = objective
        .outer_domain_upper_bound()
        .expect("the SAE outer domain face must be constructible")
        .expect("the SAE outer objective declares a typed upper face");
    (upper[ard_index], seed, period)
}

/// #2691 MECHANISM — the ARD chart coordinate's domain face must be denominated
/// in this fixture's own geometry, not in binary64's normal range.
///
/// At the parent of `669d59532` the face was
/// `gam_problem::LOG_STRENGTH_MAX = 700`, a floating-point representability
/// policy by its own module doc, and the generic `RHO_BOUND = 30` was the only
/// thing standing between the ρ search and `α = e^700`. The installed face is
/// the MINIMUM of two derived quantities: the chart-resolution limit
/// `2·(ln 2n − ln P)` and the data-curvature-matching precision
/// `ln(max_row w·‖∂(gated decode)/∂t‖²)`.
///
/// The half of this that cannot be faked is the INVARIANCE. The binding face on
/// this fixture is the curvature one, and the observed latent curvature scales
/// as `‖∂z/∂t‖² ∝ scale²` — so **doubling the planted cloud must raise the face
/// by exactly `2·ln 2`**, and nothing else about the call changes. A hand-picked constant cannot move; a constant "derived" from
/// another constant cannot move with the DATA. That assertion is the one that
/// would fail first if anyone replaced this with a literal.
///
/// Both one-sided halves are pinned too: the face must bind (strictly below
/// `RHO_BOUND` and `LOG_STRENGTH_MAX`) and must still admit the seeded entry.
#[test]
fn zz_2691_the_ard_domain_face_moves_with_the_data_not_with_binary64() {
    let n: usize = 70;
    let (face_r1, seed, period) = ard_face_for(n, 8, 2.086, 0.352);
    // The WHOLE cloud is scaled, signal and noise together: the decoder is a
    // ridge-LSQ fit of `z`, linear in `z` at fixed basis, and the PCA-seeded
    // chart coordinate is scale-invariant — so doubling `z` doubles the decoder
    // exactly. Scaling the radius alone would not double `z` and the invariance
    // below would be approximate for a reason that has nothing to do with the
    // face.
    let (face_r2, _, period_2) = ard_face_for(n, 8, 2.0 * 2.086, 2.0 * 0.352);
    assert_eq!(period, period_2, "the chart period must not depend on scale");

    let resolution_face = 2.0 * ((2.0 * n as f64) / period).ln();
    eprintln!(
        "[2691-face] period={period} seed_log_ard={seed:.4} face(r=2.086)={face_r1:.6} \
         face(r=4.172)={face_r2:.6} resolution_face={resolution_face:.6} RHO_BOUND={} \
         LOG_STRENGTH_MAX={}",
        gam_solve::estimate::RHO_BOUND,
        gam_problem::LOG_STRENGTH_MAX,
    );

    // REJECT half — it binds, and the representability constant is gone.
    assert!(
        face_r1 < gam_solve::estimate::RHO_BOUND,
        "#2691: a face at or above the generic ρ box ({}) constrains nothing; got {face_r1:.6}",
        gam_solve::estimate::RHO_BOUND
    );
    assert!(
        face_r1 < gam_problem::LOG_STRENGTH_MAX,
        "#2691: the ARD face must not be the binary64 representability policy ({}); got {face_r1:.6}",
        gam_problem::LOG_STRENGTH_MAX
    );
    // The face is the MINIMUM of the two derived faces, so it can never exceed
    // the resolution one.
    assert!(
        face_r1 <= resolution_face + 1.0e-12,
        "#2691: the installed face {face_r1:.6} must not exceed the chart-resolution face {resolution_face:.6}"
    );

    // ACCEPT half — the box is not degenerate at the seeded entry.
    assert!(
        seed < face_r1,
        "#2691: the face must still admit the seeded ARD entry {seed:.6}; got {face_r1:.6}"
    );

    // THE INVARIANCE. Observed latent curvature ∝ radius², so log-face moves by
    // exactly 2·ln 2 when the planted radius doubles.
    let delta = face_r2 - face_r1;
    let expected = 2.0 * std::f64::consts::LN_2;
    assert!(
        (delta - expected).abs() <= 1.0e-6 * expected,
        "#2691: doubling the planted cloud must raise the data-curvature face by 2·ln 2 = {expected:.6}; got {delta:.6}. A face that does not move with the data is a constant, not a derivation (and a face still pinned to the chart-resolution limit would not move either — resolution_face={resolution_face:.6})"
    );
}

/// #2691 PROPERTY — every precision the face EXCLUDES must be one this
/// fixture's chart measurably cannot carry, and the healthiest precision must be
/// ADMITTED. Plus the residual gap, located rather than asserted away.
///
/// This is deliberately the bar a NECESSARY face can carry, and no more. The
/// first version of this fix used only the chart-resolution face and the
/// fixture's ladder refuted it — circular recovery R² fell `0.969 → 0.037`
/// between `α = 1e1` and `α = 1e2`, two orders below that face. The
/// data-curvature face added since is strictly tighter, and this test pins the
/// two directions that are actually claimable:
///
/// * REJECT — no rung OUTSIDE the face may be healthy (`R² > 0.9`). A face that
///   excludes a working precision is too tight and this catches it.
/// * ACCEPT — the healthiest rung on the ladder must be INSIDE the face.
///
/// It does NOT assert that everything inside is healthy, because measured, it is
/// not: see the printed `[2691-gap]` line, which locates the transition on a
/// fine ladder and reports how far the installed face sits above it. That gap is
/// recorded as a number in the test output rather than hidden behind a bar
/// chosen to pass, and closing it needs a quantity the system owns and that
/// nobody has identified yet — inventing one is the laundered-literal failure
/// this issue already documented.
///
/// Note on instrument choice, which #2691 is itself about: `distinct_wrapped`
/// stays `70/70` out to `α = 1e4` while circular variance is `6.4e-4`, so a
/// "more than one resolved point" bar passes a dead chart. A COUNT is not a
/// resolution measure. Recovery R² is.
#[test]
fn zz_2691_the_face_excludes_every_precision_this_chart_cannot_carry() {
    let n: usize = 70;
    let (face, seed, period) = ard_face_for(n, 8, 2.086, 0.352);
    let resolution_face = 2.0 * ((2.0 * n as f64) / period).ln();
    let cloud = planted_circle_cloud(n, 8, 2.086, 0.352);
    let z = &cloud.z;

    let recovery_at = |log_ard: f64| -> Option<f64> {
        let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let mut rho =
            SaeManifoldRho::new(1.0e-3_f64.ln(), 1.0e-3_f64.ln(), vec![array![log_ard]; 1]);
        term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6)
            .ok()?;
        let coords = term.assignment.coords[0].as_matrix();
        let coord: Array1<f64> = coords.column(0).to_owned();
        Some(circular_recovery_r2(&coord, &cloud.theta))
    };

    // A FINE ladder through the measured transition, so the gap is located
    // rather than bracketed by two decades.
    let ladder: Vec<f64> = vec![
        1.0e-3, 1.0e0, 1.0e1, 2.0e1, 4.0e1, 8.0e1, 1.6e2, 3.2e2, 1.0e3, 1.0e4, 1.0e6, 1.0e9,
    ];
    let mut healthy_outside: Vec<f64> = Vec::new();
    let mut best: Option<(f64, f64)> = None;
    let mut last_healthy: Option<f64> = None;
    let mut first_dead: Option<f64> = None;
    eprintln!(
        "[2691-sep] face={face:.4} resolution_face={resolution_face:.4} seed={seed:.4}"
    );
    for alpha in ladder {
        let log_ard = alpha.ln();
        let Some(r2) = recovery_at(log_ard) else {
            eprintln!("[2691-sep] alpha={alpha:.3e} log_ard={log_ard:.3} REFUSED");
            continue;
        };
        let inside = log_ard <= face;
        eprintln!(
            "[2691-sep] alpha={alpha:.3e} log_ard={log_ard:.3} inside={inside} recovery_R2={r2:.4}"
        );
        if best.is_none_or(|(_, b)| r2 > b) {
            best = Some((log_ard, r2));
        }
        if r2 > 0.9 {
            last_healthy = Some(log_ard);
            if !inside {
                healthy_outside.push(log_ard);
            }
        } else if r2 < 0.5 && first_dead.is_none() {
            first_dead = Some(log_ard);
        }
    }

    let (best_log_ard, best_r2) = best.expect("the ladder must fit at least one rung");
    let transition = first_dead.expect(
        "the ladder must reach a dead rung (R² < 0.5) or it cannot say where the chart fails",
    );
    eprintln!(
        "[2691-gap] last_healthy_log_ard={:?} first_dead_log_ard={transition:.4} \
         installed_face={face:.4} face_minus_first_dead={:.4} \
         resolution_face_minus_first_dead={:.4}",
        last_healthy,
        face - transition,
        resolution_face - transition
    );

    // ACCEPT — the healthiest precision must be reachable.
    assert!(
        best_log_ard <= face,
        "#2691 ACCEPT half: the face {face:.4} excludes the healthiest precision on the ladder \
         (log_ard {best_log_ard:.4}, recovery R² {best_r2:.4}) — the face is too tight"
    );
    assert!(
        seed < face,
        "#2691: the face must still admit the seeded ARD entry {seed:.6}; got {face:.6}"
    );

    // REJECT — nothing the face throws away may be a working chart.
    assert!(
        healthy_outside.is_empty(),
        "#2691 REJECT half: the face {face:.4} excludes precisions this chart SURVIVES \
         (recovery R² > 0.9) at log_ard {healthy_outside:?} — the face is too tight"
    );

    // The installed face must be strictly tighter than the chart-resolution face
    // alone, which the ladder measured admitting three dead rungs. This is the
    // part of the gap the data-curvature face actually closes; the rest is
    // reported on the `[2691-gap]` line above and is NOT claimed here.
    assert!(
        face < resolution_face,
        "#2691: the data-curvature face must bind on this fixture — installed {face:.4} is not \
         below the chart-resolution face {resolution_face:.4}, so the two-order gap the ladder \
         measured is untouched"
    );
}

/// #2691 — the σ witness, BOUNDED BY ITERATIONS rather than by the clock.
///
/// The first attempt at this drove the full production entry at σ = 1.5 with an
/// unbounded outer budget. Measured: base refused `DegenerateChart` at 659 s;
/// with the box corrected it was still running at 3600 s and was killed with no
/// `test result:` line. **A timeout measured nothing**, and leaving it there
/// would mean the regime with the worst behaviour is the one with no instrument
/// — a fix and a hang produce the same absence.
///
/// So the budget is `with_max_iter`, not a wall clock: every σ returns a
/// terminal ρ whether or not the search converged, and the row is printed either
/// way. `OuterProblem::run` is the same seam that installs the objective domain
/// (`rho_optimizer::run::install_objective_domain`), so the face under test is
/// the one in force.
///
/// The assertion is the one this issue was filed on and it is now checkable at
/// every σ: the terminal ARD log-precision must lie inside the installed face,
/// and must NOT sit on the generic `RHO_BOUND = 30` the issue measured it
/// railing to. The printed `moved` column is the mute-fixture guard: if the
/// search never left the seed at any σ, the run proved nothing and the test says
/// so rather than passing quietly.
#[test]
fn zz_2691_bounded_sigma_witness_returns_an_answer_at_every_sigma() {
    let n: usize = 70;
    let p: usize = 8;
    let radius: f64 = 2.086;
    let sigmas: Vec<f64> = vec![0.352, 1.0, 1.5];
    let mut any_moved = false;
    let mut on_generic_box: Vec<f64> = Vec::new();
    let mut outside_face: Vec<(f64, f64, f64)> = Vec::new();

    eprintln!("[2691-sigma] sigma\tface\tseed\tterminal_log_ard\tmoved\tconverged\tsecs");
    for &sigma in &sigmas {
        let (face, seed, _period) = ard_face_for(n, p, radius, sigma);
        let cloud = planted_circle_cloud(n, p, radius, sigma);
        let z = &cloud.z;
        let (term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let rho = SaeManifoldRho::new(
            1.0e-3_f64.ln(),
            1.0e-3_f64.ln(),
            vec![array![1.0e-3_f64.ln()]; 1],
        )
        .for_assignment(term.assignment.mode);
        let ard_index = rho.ard_flat_index(0, 0);
        let rho_flat = rho.to_flat();
        let n_params = rho_flat.len();
        let mut objective =
            SaeManifoldOuterObjective::new(term, z.clone(), None, rho, 40, 1.0, 1.0e-6, 1.0e-6);
        let start = std::time::Instant::now();
        // The BUDGET, and the whole point of this test: an iteration cap returns
        // a terminal ρ at every σ. A clock cap returns nothing at the σ that
        // matters.
        let result = gam_solve::rho_optimizer::OuterProblem::new(n_params)
            .with_initial_rho(rho_flat.clone())
            .with_max_iter(8)
            .run(&mut objective, "SAE #2691 bounded σ witness");
        let secs = start.elapsed().as_secs_f64();
        let (terminal, converged) = match &result {
            Ok(outcome) => (outcome.rho[ard_index], outcome.converged()),
            Err(error) => {
                eprintln!(
                    "[2691-sigma] {sigma:.3}\t{face:.4}\t{seed:.4}\t-\t-\t-\t{secs:.1}\tREFUSED: {}",
                    format!("{error}").replace('\n', " ")
                );
                continue;
            }
        };
        let moved = (terminal - seed).abs() > 1.0e-9;
        any_moved |= moved;
        eprintln!(
            "[2691-sigma] {sigma:.3}\t{face:.4}\t{seed:.4}\t{terminal:.4}\t{moved}\t{converged}\t{secs:.1}"
        );
        if (terminal - gam_solve::estimate::RHO_BOUND).abs() <= 1.0e-6 {
            on_generic_box.push(sigma);
        }
        if terminal > face + 1.0e-9 {
            outside_face.push((sigma, terminal, face));
        }
    }

    assert!(
        any_moved,
        "#2691: the bounded σ witness never moved the ARD coordinate off its seed at any σ, so \
         it cannot say anything about railing — raise the iteration budget or the fixture is mute"
    );
    assert!(
        on_generic_box.is_empty(),
        "#2691: the terminal ARD log-precision sat on the generic ρ box ({}) at σ {on_generic_box:?} \
         — that is the rail this issue was filed on",
        gam_solve::estimate::RHO_BOUND
    );
    assert!(
        outside_face.is_empty(),
        "#2691: the terminal ARD log-precision escaped its installed face at (σ, terminal, face) \
         {outside_face:?}"
    );
}

/// #2691 — is the residual gap a SCALE error or a dimensionless factor?
///
/// The installed face is `ln(observed GN curvature)` and it is measurably loose:
/// on this fixture the face sits at `log α = 5.179` while the chart's recovery
/// dies between `log α = 2.996` (α=20, R² 0.939) and `3.689` (α=40, R² 0.060) —
/// `1.49` log units of slack. Before anyone proposes a tighter face, the shape
/// of that slack has to be settled, and there are only two possibilities:
///
/// * the curvature is the WRONG denominator, and the transition scales
///   differently from `‖∂z/∂t‖² ∝ scale²`; or
/// * the curvature is the RIGHT denominator up to a dimensionless constant
///   `c = curvature / α*`, in which case the transition moves by exactly the
///   same `scale²` the face does and `c` is scale-invariant.
///
/// Those two have opposite consequences and the difference is one measurement:
/// run the SAME absolute ladder at two cloud scales and compare the bracket on
/// `c`. The ladder is deliberately NOT pre-scaled by `scale²`, because scaling
/// it would assume the answer.
///
/// This test asserts the invariance rather than any particular value of `c`:
/// naming a `c` from one fixture is exactly the laundered-literal failure this
/// issue documented, and a bracket that holds across scales is the *input* a
/// future derivation has to match, not the derivation.
#[test]
fn zz_2691_is_the_residual_gap_scale_invariant() {
    let n: usize = 70;
    let p: usize = 8;
    // One absolute ladder for both scales, so the comparison cannot be circular.
    // Quarter-octave steps: a factor-of-two bracket on `c` only CONSTRAINS the
    // candidates (2*pi, e^2, 8, pi^2 all sit inside one), so the ladder is fine
    // enough to leave a bracket that can SELECT. It still spans both scales'
    // transitions without being re-centred on either.
    let ladder: Vec<f64> = (0..57).map(|k| 2.0_f64.powf(k as f64 / 4.0)).collect();

    let mut brackets: Vec<(f64, f64, f64, f64)> = Vec::new();
    for &scale in &[1.0_f64, 2.0_f64] {
        let radius = 2.086 * scale;
        let sigma = 0.352 * scale;
        let (face, _seed, _period) = ard_face_for(n, p, radius, sigma);
        let curvature = face.exp();
        let cloud = planted_circle_cloud(n, p, radius, sigma);
        let z = &cloud.z;
        let mut last_healthy = f64::NAN;
        let mut first_dead = f64::NAN;
        for &alpha in &ladder {
            let log_ard = alpha.ln();
            let (mut term, _disp) =
                build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
            let mut rho =
                SaeManifoldRho::new(1.0e-3_f64.ln(), 1.0e-3_f64.ln(), vec![array![log_ard]; 1]);
            if term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6)
                .is_err()
            {
                continue;
            }
            let coords = term.assignment.coords[0].as_matrix();
            let coord: Array1<f64> = coords.column(0).to_owned();
            let r2 = circular_recovery_r2(&coord, &cloud.theta);
            eprintln!("[2691-scale] scale={scale} alpha={alpha:.3} recovery_R2={r2:.4}");
            if r2 > 0.9 {
                last_healthy = alpha;
            } else if r2 < 0.5 && first_dead.is_nan() {
                first_dead = alpha;
            }
        }
        assert!(
            last_healthy.is_finite() && first_dead.is_finite(),
            "#2691: the ladder must bracket the transition at scale {scale}; got \
             last_healthy={last_healthy} first_dead={first_dead}"
        );
        // `c` is bracketed from both sides: the transition lies in
        // (last_healthy, first_dead], so c lies in [curv/first_dead, curv/last_healthy).
        let c_lo = curvature / first_dead;
        let c_hi = curvature / last_healthy;
        eprintln!(
            "[2691-scale] scale={scale} curvature={curvature:.4} face={face:.4} \
             transition_in=({last_healthy:.3},{first_dead:.3}] c_in=[{c_lo:.4},{c_hi:.4})"
        );
        brackets.push((scale, curvature, last_healthy, first_dead));
    }

    let (s1, curv1, lh1, fd1) = brackets[0];
    let (s2, curv2, lh2, fd2) = brackets[1];
    let (c1_lo, c1_hi) = (curv1 / fd1, curv1 / lh1);
    let (c2_lo, c2_hi) = (curv2 / fd2, curv2 / lh2);
    eprintln!(
        "[2691-scale-verdict] curvature {curv1:.4}->{curv2:.4} (ratio {:.4}, expected 4) \
         transition ({lh1:.3},{fd1:.3}]->({lh2:.3},{fd2:.3}] (ratio {:.4}) \
         c_bracket scale={s1}: [{c1_lo:.3},{c1_hi:.3})  scale={s2}: [{c2_lo:.3},{c2_hi:.3})",
        curv2 / curv1,
        fd2 / fd1
    );

    // The curvature itself must scale as `scale^2` — this is the same invariance
    // `zz_2691_the_ard_domain_face_moves_with_the_data_not_with_binary64` pins,
    // restated here so a failure below cannot be blamed on the face.
    assert!(
        (curv2 / curv1 - 4.0).abs() <= 1.0e-6 * 4.0,
        "#2691: the observed curvature must scale as scale^2; got ratio {:.6}",
        curv2 / curv1
    );

    // THE QUESTION. Overlapping brackets ⇒ `c` is scale-invariant ⇒ the
    // curvature is the right denominator and the residual slack is one
    // dimensionless number. Disjoint brackets ⇒ it is not, and any face built on
    // `curvature / c` is wrong in a way this fixture can already see.
    assert!(
        c1_lo < c2_hi && c2_lo < c1_hi,
        "#2691: the dimensionless gap c = curvature/alpha* is NOT scale-invariant — \
         [{c1_lo:.3},{c1_hi:.3}) at scale {s1} does not overlap [{c2_lo:.3},{c2_hi:.3}) at \
         scale {s2}. The observed Gauss--Newton curvature is then the WRONG denominator for \
         the residual gap, and a tighter face must be derived from something else."
    );
}

/// #2691 — is the dimensionless `c = curvature / α*` a UNIVERSAL constant?
///
/// ## Retraction of the experiment this file previously proposed
///
/// `zz_2691_is_the_residual_gap_scale_invariant` established that `c` is
/// scale-invariant, leaving a quarter-octave bracket `[5.5458, 6.5951)` in which
/// `2π = 6.2832` is the only simple constant. The follow-up proposed on the
/// issue was "vary the chart PERIOD: `c = 2π` would not move, `c = κ = 2π/P`
/// would move as `1/P`". **That experiment is mute by construction and must not
/// be run as a discriminator.** Under a consistent convention change
/// `t' = t/P`:
///
/// ```text
///   basis:   same functions of u = t/P  =>  ‖∂z/∂t'‖² = P²·curvature
///   prior:   V = (α/κ²)(1 − cos κt), κ = 2π/P; the SAME energy in t' units
///            (κ' = 2π) requires α' = α·P²
///   =>       α*' = P²·α*   and   c' = P²·curvature / (P²·α*) = c
/// ```
///
/// `c` is a reparametrization invariant, and `κ` expressed in the chart's own
/// normalized coordinate is `2π` for EVERY `P`. So "`c = 2π`" and "`c = κ`" are
/// not two hypotheses — they are one statement in two unit systems, and no
/// value of `P` separates them. (Nor is a period-`P` chart even constructible
/// here: `PeriodicHarmonicEvaluator` hardwires `2π·h·t`, and
/// `LatentManifold::Circle { period }` is a *convention* that must match it.)
///
/// ## What can discriminate
///
/// If `c` is a universal pure number it must be invariant to every axis that is
/// NOT a reparametrization. Those are cheap and legal: `n`, `p`, `σ`. Any
/// dependence kills "universal constant" and names the axis the real law is
/// written in.
///
/// ## Pre-registered, before the run
///
/// Bracket width is `2^(1/4) = 1.1892`, and each axis has a `4×` lever, so any
/// power law with `|slope| > ln(1.1892)/ln(4) = 0.125` is resolvable.
///
/// | hypothesis | prediction |
/// |---|---|
/// | `H0`: `c` universal | slope `0` on all three axes; all cells' brackets share a common value |
/// | `c ∝ √n` | `c(160)/c(40) = 2.000` — brackets disjoint |
/// | `c ∝ p` | `c(16)/c(4) = 4.000` — grossly disjoint |
/// | `c ∝ ln n` | `c(160)/c(40) = 1.3758` — disjoint |
/// | `c ∝ 1/σ` | `c(0.176)/c(0.704) = 4.000` — disjoint |
///
/// ## RESULT: `H0` was tested once and REFUTED. Recorded, with what it does and
/// does not license.
///
/// ```text
/// cell                    c_lo     c_hi    c_mid   width  rungs
/// n=40  p=8  sigma=0.352  4.000    5.657   4.757   1.414      2
/// n=70  p=8  sigma=0.352  5.657    6.727   6.169   1.189      1
/// n=160 p=8  sigma=0.352  4.757    5.657   5.187   1.189      1
/// n=70  p=4  sigma=0.352  5.657    6.727   6.169   1.189      1
/// n=70  p=16 sigma=0.352  4.757    5.657   5.187   1.189      1
/// n=70  p=8  sigma=0.176  4.757    6.727   5.657   1.414      2
/// n=70  p=8  sigma=0.704  8.000   26.909  14.672   3.364      7
/// intersection over all cells = [8.0000, 5.6569)  -> EMPTY
/// ```
///
/// **`H0` is refuted as stated — and NONE of the pre-registered alternatives is
/// supported either.** They predicted `2.000` / `4.000` / `1.3758` / `4.000`.
///
/// ### Re-measured with the OFF-GRID estimator, which corrects a claim I made
/// from the grid brackets
///
/// The interpolated `0.7`-crossing point estimates (`c_star`), with the
/// `0.9 -> 0.5` decay span that says whether the crossing has a referent:
///
/// ```text
/// cell                    c_star   decay_span  resolved
/// n=40  p=8  sigma=0.352  4.5789   1.2018      yes
/// n=70  p=8  sigma=0.352  6.4224   1.0934      yes
/// n=160 p=8  sigma=0.352  5.4258   1.0842      yes
/// n=70  p=4  sigma=0.352  6.4359   1.0839      yes
/// n=70  p=16 sigma=0.352  5.3967   1.0829      yes
/// n=70  p=8  sigma=0.176  5.4315   1.1031      yes
/// n=70  p=8  sigma=0.704  9.2305   2.7325      NO  (non-measurement)
/// spread over the six resolved cells: [4.5789, 6.4359] = 1.4056 = 1.96 rungs
/// ```
///
/// **I previously wrote that the six sharp cells "agree to within one
/// quarter-octave". That was an artifact of reading grid-quantized brackets and
/// it is wrong.** Off the grid they span a factor of `1.4056`. The interpolated
/// crossing is resolved far better than one grid step — every resolved cell's
/// whole `0.9 -> 0.5` decay fits inside ~one step — so `1.4056` is a REAL
/// spread, not quantization.
///
/// What the spread is NOT is a law. It is non-monotonic in `n`
/// (`4.579 -> 6.422 -> 5.426` over `40 -> 70 -> 160`), flat in `p` from 4 to 8
/// (`6.436` vs `6.422`, 0.2%) and then down at 16 (`5.397`), and up with `sigma`
/// (`5.432 -> 6.422`). So `c` genuinely moves by ~40% across this design and
/// does so along NO single axis varied here — which refutes `H0` for a better
/// reason than the grid tie did, and still supports none of the alternatives.
///
/// **Caveat that bounds all of the above: there are NO REPLICATES.**
/// `deterministic_circle_noise` takes no seed, so changing `n` also changes the
/// noise draw, and seed-to-seed variation of `c_star` is UNMEASURED. Part of the
/// `1.4056` could be it. So `1.4056` is an upper bound on the systematic
/// variation, not an estimate of it.
///
/// **The actionable conclusion is robust to that caveat, which is why the
/// replicate work is deliberately NOT done here.** The caveat has exactly two
/// branches and they agree on the only thing anyone would act on:
///
/// * if the spread is SYSTEMATIC, `c` is not a constant;
/// * if the spread is SEED VARIANCE, `c` is not *measurable* to better than
///   ~40% by this instrument.
///
/// Both give **"a face `curvature / c` with a fixed `c` is not installable"**.
/// Nothing is landing on `c`, and proof is owed by the landing — so pinning `c`
/// rather than bounding it buys nothing at present. The follow-up is SPECIFIED
/// rather than performed: it needs a seed parameter on the fixture generator and
/// `>= 3` draws per cell, and it is owed by whoever wants to install such a
/// face, not by this diagnostic.
///
/// Two things actually break the intersection, and neither is a law:
///
/// 1. **A resolution tie, and it is an ESTIMATOR DEFECT.** `n=40` gives
///    `[4.000, 5.6569)` and `n=70` gives `[5.6569, 6.7272)` — ADJACENT half-open
///    brackets touching at exactly the same grid point. `c` is quantized to
///    `2^(k/4)` by construction, so **two cells whose true `c` are IDENTICAL
///    produce an empty intersection whenever that value lands on a grid point.**
///    `intersection-non-empty` is therefore not a test of agreement; it is a
///    test of agreement AND grid alignment, and the second conjunct is a
///    property of the instrument. That is what produced the headline `EMPTY`
///    beside six cells agreeing to within one quarter-octave. The repair, applied
///    below: estimate the crossing by log-linear INTERPOLATION to get a point
///    estimate off the grid, and report `c_star` with the width that bounds it.
/// 2. **One cell that is a NON-MEASUREMENT, not a disagreeing measurement.** At
///    `sigma = 0.704` the transition is `7` rungs wide: recovery decays
///    GRADUALLY, so **there is no sharp `alpha*` for any estimator to have as a
///    referent**, and `[8.0, 26.9)` is an artifact of forcing a two-threshold
///    bracket onto a gradual curve. It is excluded because the quantity does not
///    exist there, NOT because it disagreed — those two read identically in a
///    table and only one of them is legitimate, so the distinction is stated
///    here and the cell is flagged in the output rather than silently dropped.
///
/// The actionable consequence is the same either way and it is the point of this
/// test: **a face of the form `curvature / c` with a fixed `c` is not
/// installable.** `c` moves by ~40% across `n`, `p` and `sigma` with no monotone
/// dependence on any of them, and in the high-noise regime there is no sharp
/// transition for it to denominate at all.
///
/// ## Axes NOT varied — the claim is invariance on three axes, not universality
///
/// **`h`, the harmonic order, is untested, and it is the axis most likely to
/// move a constant that equals `2 pi`.** `PeriodicHarmonicEvaluator` hardwires
/// `2 pi * h * t`, so if `c ~ 2 pi` because the BASIS frequency carries that
/// factor, `h` changes the frequency content without changing units — `c`
/// invariant in `h` would be much stronger evidence, `c` moving with `h` would
/// mean the constant is a property of the basis rather than of the geometry.
/// It is not cheap here and that is why it is absent, stated precisely so nobody
/// reads its absence as a null: `build_term` hardcodes
/// `PeriodicHarmonicEvaluator::new(3)` (`H = 1`), and merely widening the basis
/// against `w = 1` planted data is a weak probe because the extra harmonics fit
/// only noise and barely move the curvature. The strong version needs a planted
/// WINDING number `w > 1`, a basis with `H >= w`, and a winding-aware
/// replacement for `circular_recovery_r2` — a new fixture and a new statistic.
///
/// Also unvaried: the chart topology, the assignment mode, `K`, and the row
/// metric.
///
/// ## THE PRE-COMMITMENT LEDGER — the full set, what was asserted, and why each
/// exclusion
///
/// Selecting a subset of pre-commitments AFTER seeing data can still bias, if a
/// different result would have led to asserting a different member. So the whole
/// set is enumerated here with a reason per exclusion, and each reason is marked
/// as data-dependent or not. A reader can then check the exclusions themselves.
///
/// | # | pre-committed assertion | status | reason |
/// |---|---|---|---|
/// | A1 | the curvature face binds in every cell (`face < resolution_face`), so `exp(face)` IS the curvature | **ASSERTED** | — |
/// | A2 | every cell brackets its transition inside its own curvature-centred ladder (`last_healthy` and `first_dead` both found) | **ASSERTED** | — |
/// | A3 | `H0`: the intersection of all cells' `c` brackets is non-empty | **DROPPED — REFUTED** | **DATA-DEPENDENT.** Reported as a refutation above with its numbers, not omitted. Re-running it as a gate after seeing its answer would be a bar fitted to its own data; asserting its negation would gate on the two artifacts named above. |
///
/// **A correction to `9e34d7853`'s commit message.** It said the two `c`-unit
/// assertions below "were both commitments BEFORE this run". That is not
/// accurate and this comment is the correction of record: they were written
/// AFTER the run. They are not new bars — `c_lo > 1 && c_hi < 32` and
/// `c_lo > 1` are A2 restated in `c` units, since the ladder spans
/// `curvature/32 .. curvature` and a bracketed transition already implies them.
/// They are kept for legibility and labelled here as restatements, not as
/// independent pre-commitments.
///
/// So exactly ONE commitment was dropped, its reason IS data-dependent, and it
/// is reported as a failure rather than omitted — which is the case the ledger
/// exists to make visible.
#[test]
fn zz_2691_is_the_dimensionless_gap_a_universal_constant() {
    let default_n: usize = 70;
    let default_p: usize = 8;
    let default_sigma: f64 = 0.352;
    let radius: f64 = 2.086;

    // (label, n, p, sigma). Three values on each axis so a slope is FITTED, not
    // inferred from two points.
    let cells: Vec<(&str, usize, usize, f64)> = vec![
        ("n", 40, default_p, default_sigma),
        ("n", default_n, default_p, default_sigma),
        ("n", 160, default_p, default_sigma),
        ("p", default_n, 4, default_sigma),
        ("p", default_n, 16, default_sigma),
        ("sigma", default_n, default_p, 0.176),
        ("sigma", default_n, default_p, 0.704),
    ];

    let mut results: Vec<(String, usize, usize, f64, f64, f64, f64, bool)> = Vec::new();
    for (axis, n, p, sigma) in cells {
        let (face, _seed, _period) = ard_face_for(n, p, radius, sigma);
        let resolution_face = 2.0 * ((2.0 * n as f64) / 1.0_f64).ln();
        // `exp(face)` is the curvature only while the curvature face binds.
        assert!(
            face < resolution_face,
            "#2691: cell ({axis} n={n} p={p} sigma={sigma}) has the RESOLUTION face binding \
             ({face:.4} vs {resolution_face:.4}), so exp(face) is not the curvature and this \
             cell cannot report c"
        );
        let curvature = face.exp();
        let cloud = planted_circle_cloud(n, p, radius, sigma);
        let z = &cloud.z;
        // Quarter-octave ladder spanning curvature/32 .. curvature, centred on a
        // quantity MEASURED in this cell rather than on the answer.
        let mut curve: Vec<(f64, f64)> = Vec::new();
        let mut last_healthy = f64::NAN;
        let mut first_dead = f64::NAN;
        for k in 0..=20 {
            let alpha = curvature * 2.0_f64.powf(-5.0 + k as f64 / 4.0);
            let log_ard = alpha.ln();
            let (mut term, _disp) =
                build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
            let mut rho =
                SaeManifoldRho::new(1.0e-3_f64.ln(), 1.0e-3_f64.ln(), vec![array![log_ard]; 1]);
            if term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6)
                .is_err()
            {
                continue;
            }
            let coords = term.assignment.coords[0].as_matrix();
            let coord: Array1<f64> = coords.column(0).to_owned();
            let r2 = circular_recovery_r2(&coord, &cloud.theta);
            curve.push((alpha, r2));
            if r2 > 0.9 {
                last_healthy = alpha;
            } else if r2 < 0.5 && first_dead.is_nan() {
                first_dead = alpha;
            }
        }
        // The half-open grid bracket cannot express "these two cells agree at a
        // grid point", so the location is estimated OFF the grid: the first
        // downward crossing of a recovery level, log-linearly interpolated in
        // `alpha`. `c_star` is the 0.7 crossing; the 0.9 and 0.5 crossings bound
        // how sharp the transition is and therefore whether `c_star` has a
        // referent at all.
        let crossing = |level: f64| -> f64 {
            for pair in curve.windows(2) {
                let (a0, r0) = pair[0];
                let (a1, r1) = pair[1];
                if r0 >= level && r1 < level {
                    let t = (r0 - level) / (r0 - r1);
                    return (a0.ln() + t * (a1.ln() - a0.ln())).exp();
                }
            }
            f64::NAN
        };
        let (a90, a70, a50) = (crossing(0.9), crossing(0.7), crossing(0.5));
        assert!(
            last_healthy.is_finite() && first_dead.is_finite(),
            "#2691: cell ({axis} n={n} p={p} sigma={sigma}) did not bracket the transition \
             (last_healthy={last_healthy} first_dead={first_dead}); the ladder does not cover it"
        );
        let c_lo = curvature / first_dead;
        let c_hi = curvature / last_healthy;
        // Sharpness rule, stated in the instrument's own units and INDEPENDENT of
        // the value: the 0.9 -> 0.5 decay must happen inside one octave (four
        // grid steps) for a point crossing to name a location. A cell that fails
        // it is a NON-MEASUREMENT -- the quantity has no referent there -- not a
        // measurement that disagrees.
        let decay_span = if a90.is_finite() && a50.is_finite() {
            a50 / a90
        } else {
            f64::INFINITY
        };
        let resolved = decay_span <= 2.0;
        let c_star = if a70.is_finite() {
            curvature / a70
        } else {
            f64::NAN
        };
        eprintln!(
            "[2691-univ] axis={axis} n={n} p={p} sigma={sigma} curvature={curvature:.4} \
             transition=({last_healthy:.4},{first_dead:.4}] c=[{c_lo:.4},{c_hi:.4}) \
             c_star={c_star:.4} decay_span={decay_span:.4} resolved={resolved}{}",
            if resolved {
                ""
            } else {
                "  <- NON-MEASUREMENT: the 0.9->0.5 decay exceeds one octave, so no sharp \
                 alpha* exists for c_star to locate; excluded because the quantity has no \
                 referent here, NOT because it disagreed"
            }
        );
        results.push((axis.to_string(), n, p, sigma, c_lo, c_hi, c_star, resolved));
    }

    // REPORTED, not asserted: the H0 grid-bracket intersection (kept only so the
    // refuted hypothesis stays visible), and the OFF-GRID point estimates that
    // replace it.
    let lo = results.iter().fold(f64::NEG_INFINITY, |a, r| a.max(r.4));
    let hi = results.iter().fold(f64::INFINITY, |a, r| a.min(r.5));
    eprintln!(
        "[2691-univ-verdict] H0 grid-bracket intersection = [{lo:.4}, {hi:.4}) -> {} \
         (NOTE: half-open brackets on a fixed 2^(k/4) grid cannot express agreement AT a grid \
         point, so this statistic tests agreement AND grid alignment; the c_star spread below \
         is the one that measures agreement)",
        if lo < hi { "NON-EMPTY" } else { "EMPTY (H0 refuted)" }
    );
    for (axis, n, p, sigma, c_lo, c_hi, c_star, resolved) in &results {
        let width = c_hi / c_lo;
        let rungs = (width.log2() * 4.0).round() as i64;
        eprintln!(
            "[2691-univ-cell] {axis} n={n} p={p} sigma={sigma} c=[{c_lo:.4},{c_hi:.4}) \
             width={width:.4} rungs={rungs} c_star={c_star:.4} resolved={resolved}"
        );
    }

    // The agreement statistic, over the cells where the quantity HAS a referent.
    let sharp: Vec<f64> = results
        .iter()
        .filter(|r| r.7 && r.6.is_finite())
        .map(|r| r.6)
        .collect();
    assert!(
        sharp.len() >= 2,
        "#2691: fewer than two cells produced a resolved transition, so no agreement \
         statistic exists; got {} of {}",
        sharp.len(),
        results.len()
    );
    let c_min = sharp.iter().copied().fold(f64::INFINITY, f64::min);
    let c_max = sharp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let grid_step = 2.0_f64.powf(0.25);
    eprintln!(
        "[2691-univ-spread] resolved cells={} c_star in [{c_min:.4}, {c_max:.4}] spread={:.4} \
         grid_step={grid_step:.4} spread_in_rungs={:.2}",
        sharp.len(),
        c_max / c_min,
        (c_max / c_min).log2() * 4.0
    );

    // ASSERTION A1 and A2 are the pre-committed pair and are asserted above
    // (curvature face binds per cell; the transition is bracketed inside the
    // cell's own ladder). The two checks below are A2 RESTATED in `c` units --
    // the ladder spans curvature/32 .. curvature, so a bracketed transition
    // already implies them. They were written AFTER the run and are kept for
    // legibility; they are not independent pre-commitments and the doc comment
    // says so.
    for (axis, n, p, sigma, c_lo, c_hi, _c_star, _resolved) in &results {
        assert!(
            *c_lo > 1.0 && *c_hi < 32.0,
            "#2691: cell ({axis} n={n} p={p} sigma={sigma}) has c=[{c_lo:.4},{c_hi:.4}) at or \
             outside the ladder edge -- the transition escaped a 32x window below the measured \
             curvature, so the curvature is no longer the right order for it here"
        );
    }
}


/// #2691 REGRESSION, the `K >= 2` half — a dictionary in which ONE atom's chart
/// has collapsed to a point while its sibling's is healthy must be NAMED, and
/// the guard's own population filter must not name an atom nobody uses.
///
/// This is the case the first guard left open: it refused only when EVERY atom
/// had lost its chart (`collapsed_atoms.len() == atom_count`), so a `K = 2` fit
/// with one point-atom and one real chart passed. The reconstruction is carried
/// by the healthy atom, so nothing denominated in reconstruction quality can
/// order these states either — the same reason the `K = 1` witness above is not
/// denominated in EV.
///
/// The collapse is PLACED (`set_flat` writes one point of the circle onto atom
/// 0's chart) rather than searched for, so the witness measures the DECISION and
/// cannot be a flaky optimizer outcome. The entry-level half — that
/// `run_sae_manifold_fit` refuses on exactly this predicate — is held by
/// `zz_2691_collapsed_chart_is_refused_by_the_production_entry` above.
///
/// MEASURED, and the reason this witness is at the predicate rather than at the
/// entry: driving the same `K = 2` state through `run_sae_manifold_fit` at a
/// fixed rho (alpha = 1e9 on atom 0, 1e-3 on atom 1, planted circle n=70 p=8
/// sigma=0.352) never reaches the chart guard at all — the inner solve refuses
/// first, at both the parent and this commit, with "inner solve did not converge
/// at fixed rho ... refusing to rank an off-optimum Laplace criterion"
/// (‖g‖=4.152980e-1 vs tolerance 2.897286e-4 after 5760 inner iterations). That
/// is a refusal stack, not a second defect: no fit is minted either way. It does
/// mean an entry-level `K = 2` fixture would have measured the inner solver, not
/// this guard.
#[test]
fn zz_2691_a_collapsed_atom_beside_a_healthy_one_is_named() {
    use ndarray::Array2;

    let cloud = planted_circle_cloud(70, 8, 2.086, 0.352);
    let (mut term, _disp) =
        build_term(cloud.z.view(), 2, Topo::Circle, AssignmentMode::softmax(1.0));
    let n = cloud.z.nrows();

    // Place the collapse on atom 0 ONLY: every row sits on the chart origin,
    // one point of the period-1 circle. Atom 1 keeps the seeded chart.
    term.assignment.coords[0].set_flat(Array1::<f64>::zeros(n).view());

    let report = term.chart_degeneracy_report();
    let assignments = term.assignment.assignments();
    for axis in report.axes.iter() {
        eprintln!(
            "[2691-k2] atom {} axis {} dispersion {:.6e} floor {:.6e} resolved_points {} \
             degenerate {}",
            axis.atom,
            axis.axis,
            axis.dispersion,
            axis.floor,
            axis.resolved_points,
            axis.degenerate()
        );
    }
    assert_eq!(
        report.atom_count, 2,
        "the fixture must be the K=2 case; got {} atoms",
        report.atom_count
    );
    assert!(
        report.axes.iter().any(|axis| axis.atom == 1 && !axis.degenerate()),
        "#2691: this witness is only the partial-collapse case if atom 1's chart SURVIVED — \
         otherwise it is the already-covered all-atoms collapse. Axes: {:?}",
        report.axes
    );
    assert_eq!(
        report.atoms_without_a_chart(),
        vec![0],
        "the placed collapse must be seen on atom 0 and only atom 0"
    );

    // THE DEFECT, stated as the predicate that used to decide the refusal: the
    // fit-level aggregate reads "not every atom collapsed" and lets this state
    // through.
    assert!(
        report.atoms_without_a_chart().len() != report.atom_count,
        "#2691: the fit-level `all atoms collapsed` condition must be FALSE here — if it were \
         true this fixture would not be the partial collapse the guard is being extended for"
    );

    // THE FIX: the per-atom condition names the collapsed atom, because it is
    // load-bearing — it carries assignment mass that is representable against
    // the dominant atom on some row.
    let load_bearing = load_bearing_atoms(assignments.view());
    eprintln!("[2691-k2] load_bearing={load_bearing:?}");
    assert_eq!(
        report.chart_less_load_bearing_atoms(assignments.view()),
        vec![0],
        "#2691: a collapsed atom that carries assignment mass must be named; assignments \
         column sums = {:?}",
        (0..assignments.ncols())
            .map(|k| assignments.column(k).sum())
            .collect::<Vec<_>>()
    );
    let evidence = report.atom_evidence(&[0]);
    eprintln!("[2691-k2] evidence: {evidence}");
    assert!(
        evidence.contains("did NOT collapse"),
        "#2691: the refusal evidence must show the surviving chart beside the collapsed one, \
         because that pairing is what a fit-level aggregate hides; got: {evidence}"
    );

    // THE CONTROL on the population filter: an atom nobody uses has an
    // UNOBSERVED chart, not a collapsed one, and must NOT be named — otherwise
    // the guard would refuse fits that are fine.
    let mut unused = Array2::<f64>::zeros(assignments.raw_dim());
    for row in 0..unused.nrows() {
        unused[[row, 1]] = 1.0;
    }
    assert_eq!(
        load_bearing_atoms(unused.view()),
        vec![false, true],
        "an atom with zero assignment mass on every row cannot change the reconstruction at f64"
    );
    assert!(
        report.chart_less_load_bearing_atoms(unused.view()).is_empty(),
        "#2691: a collapsed atom that carries NO representable assignment mass must not be \
         named — its chart is unobserved, and refusing on it would refuse fits that are fine"
    );
}
/// #2691 — the recovered ring against the planted one, printed.
///
/// Diagnostic: it renders the fit in the plane the circle was planted in, so the
/// collapse and the recovery are legible as pictures rather than as a scalar.
/// Two arms, one variable (the ARD chart-coordinate precision): the healthy arm
/// must trace the planted ring, the collapsed arm must be a single point.
#[test]
fn zz_2691_recovered_versus_planted_ring() {
    let (n, p, sigma) = (70_usize, 8_usize, 0.352_f64);
    let cloud = planted_circle_cloud(n, p, 2.086, sigma);
    let (u, v) = planted_frame(p);
    let z = &cloud.z;

    for (label, alpha) in [("healthy", 1.0e-3_f64), ("collapsed", 1.0e9_f64)] {
        let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let mut rho =
            SaeManifoldRho::new(1.0e-3_f64.ln(), 1.0e-3_f64.ln(), vec![array![alpha.ln()]; 1]);
        term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6)
            .expect("inner joint fit");
        let fitted = term.try_fitted().expect("fitted reconstruction");
        let coord: Array1<f64> = term.assignment.coords[0].as_matrix().column(0).to_owned();
        let chart = term.chart_degeneracy_report();

        // Project both clouds onto the planted 2-frame. Only the plane is shared
        // between them; nothing about the fit is used to choose it.
        let project = |rows: ndarray::ArrayView2<'_, f64>| -> Vec<(f64, f64)> {
            (0..rows.nrows())
                .map(|i| {
                    let row = rows.row(i);
                    (row.dot(&u), row.dot(&v))
                })
                .collect()
        };
        let planted = project(z.view());
        let recovered = project(fitted.view());
        let extent = planted
            .iter()
            .chain(recovered.iter())
            .fold(0.0_f64, |m, (a, b)| m.max(a.abs()).max(b.abs()))
            .max(f64::MIN_POSITIVE);

        // 25x49 character grid over [-extent, extent]^2; '.' planted, '#'
        // recovered, '@' both.
        const ROWS: usize = 25;
        const COLS: usize = 49;
        let mut grid = vec![vec![' '; COLS]; ROWS];
        let mut mark = |points: &[(f64, f64)], glyph: char| {
            for &(a, b) in points {
                let col = (((a / extent) * 0.5 + 0.5) * (COLS - 1) as f64).round() as isize;
                let row = ((0.5 - (b / extent) * 0.5) * (ROWS - 1) as f64).round() as isize;
                if (0..COLS as isize).contains(&col) && (0..ROWS as isize).contains(&row) {
                    let cell = &mut grid[row as usize][col as usize];
                    *cell = if *cell == ' ' || *cell == glyph { glyph } else { '@' };
                }
            }
        };
        mark(&planted, '.');
        mark(&recovered, '#');

        let dispersion = chart.axes[0].dispersion;
        let points = chart.axes[0].resolved_points;
        let recovery = circular_recovery_r2(&coord, &cloud.theta);
        eprintln!(
            "[2691-ring] arm={label} alpha={alpha:.1e} circular_variance={dispersion:.6e} \
             resolved_chart_points={points}/{n} recovery_R2={recovery:.4} extent={extent:.4} \
             ('.' planted, '#' recovered, '@' both)"
        );
        for row in &grid {
            eprintln!("[2691-ring] {label:>9} |{}|", row.iter().collect::<String>());
        }
        // The chart coordinate against the phase that generated it, so the
        // picture above has the numbers beside it.
        let sample: Vec<String> = (0..n)
            .step_by(n / 10)
            .map(|i| format!("({:.3},{:.4})", cloud.theta[i], coord[i]))
            .collect();
        eprintln!("[2691-ring] {label} (theta, chart t) = {}", sample.join(" "));

        match label {
            "healthy" => assert!(
                dispersion > 0.5 && points == n && recovery > 0.9,
                "#2691: the healthy arm must trace the planted ring (circular variance \
                 {dispersion:.3e}, {points}/{n} chart points, recovery R2 {recovery:.4})"
            ),
            _ => assert!(
                chart.axes[0].degenerate(),
                "#2691: the alpha=1e9 arm must read as degenerate in the chart's own metric \
                 (circular variance {dispersion:.3e}, {points}/{n} chart points)"
            ),
        }
    }
}
