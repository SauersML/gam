//! #2933 F07 — the exact-`A` null classification is a property of the pencil `(A, Φ)`,
//! not of the coordinates it is written in.
//!
//! Under a nonsingular change of coordinates `θ → Lθ` the pencil transforms by congruence,
//! `A → LᵀAL`, `Φ → LᵀΦL`, `B_raw → LᵀB_rawL`, `E → LᵀEL`, a right-hand side maps as
//! `rhs → Lᵀrhs` and a solution as `x → L⁻¹x`. So the retained, in-band and negative counts,
//! the band's dual projection, the covariant pseudoinverse and `log|A|_reg − 2 log|det L|`
//! must all be unchanged. Every coordinate change here is NONORTHOGONAL: a rotation or a
//! common scale maps an ordinary eigenbasis to an ordinary eigenbasis, and could not
//! witness the defect.

use super::*;
use ndarray::{Array1, Array2, array};

/// A dense `Φ` with its own Cholesky factor, for the independent pencil fixtures
/// (#2933 F07).
pub(crate) struct DensePencilMetric {
    metric: Array2<f64>,
    substituted: Array2<f64>,
    lower: Array2<f64>,
}

impl DensePencilMetric {
    /// `metric` is `Φ`, `raw` the majorizer `B_raw` it conditions.
    pub(crate) fn new(metric: Array2<f64>, raw: &Array2<f64>) -> Result<Self, String> {
        use gam_linalg::faer_ndarray::FaerCholesky;
        let lower = metric
            .cholesky(Side::Lower)
            .map_err(|error| format!("DensePencilMetric: Φ is not positive definite: {error:?}"))?
            .lower_triangular();
        Ok(Self {
            substituted: &metric - raw,
            metric,
            lower,
        })
    }
}

impl ExactAPencilMetric for DensePencilMetric {
    fn dim(&self) -> usize {
        self.metric.nrows()
    }

    fn apply(&self, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        Ok(self.metric.dot(&v))
    }

    fn substituted_image(&self, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        Ok(self.substituted.dot(&v))
    }

    fn lower_solve(&self, v: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        Ok(gam_linalg::triangular::forward_substitution_lower_matrix(&self.lower, v))
    }

    fn lower_transpose_solve(&self, v: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        Ok(gam_linalg::triangular::back_substitution_lower_transpose_matrix(&self.lower, v))
    }

    fn log_det(&self) -> Result<f64, String> {
        Ok((0..self.lower.nrows())
            .map(|index| 2.0 * self.lower[[index, index]].ln())
            .sum())
    }

    fn frobenius_norm(&self) -> Result<f64, String> {
        Ok(self.metric.iter().map(|value| value * value).sum::<f64>().sqrt())
    }
}

/// Every coordinate on the border, so the whole pencil is one dense block.
fn border(v: &Array1<f64>) -> SaeArrowVector {
    SaeArrowVector {
        t: Array1::zeros(0),
        beta: v.clone(),
    }
}

fn flatten(v: &SaeArrowVector) -> Array1<f64> {
    Array1::from_iter(v.t.iter().chain(v.beta.iter()).copied())
}

fn outer(left: &Array1<f64>, right: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn((left.len(), right.len()), |(i, j)| left[i] * right[j])
}

fn congruence(m: &Array2<f64>, t: &Array2<f64>) -> Array2<f64> {
    let moved = t.t().dot(m).dot(t);
    (&moved + &moved.t()) * 0.5
}

/// `L⁻ᵀX` column by column, by back substitution written out here.
fn lower_transpose_solve(lower: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
    let n = lower.nrows();
    let mut out = Array2::<f64>::zeros(x.raw_dim());
    for column in 0..x.ncols() {
        for i in (0..n).rev() {
            let mut sum = x[[i, column]];
            for k in (i + 1)..n {
                sum -= lower[[k, i]] * out[[k, column]];
            }
            out[[i, column]] = sum / lower[[i, i]];
        }
    }
    out
}

fn householder(dim: usize, step: f64) -> Array2<f64> {
    let v = Array1::from_iter((0..dim).map(|i| 1.0 + step * i as f64));
    Array2::eye(dim) - Array2::from_shape_fn((dim, dim), |(i, j)| 2.0 * v[i] * v[j] / v.dot(&v))
}

/// Production's pencil geometry for a dense `Φ` and the majorizer `B_raw` it conditions.
fn pencil_block(a: &Array2<f64>, metric: &Array2<f64>, raw: &Array2<f64>) -> ExactHessianSpectralBlock {
    let metric = DensePencilMetric::new(metric.clone(), raw).expect("positive-definite metric");
    SaeManifoldTerm::exact_hessian_spectral_block(a.clone(), &metric).expect("pencil geometry")
}

/// `(retained positive, in band, resolved negative)`.
fn census(block: &ExactHessianSpectralBlock) -> (usize, usize, usize) {
    let mut out = (0, 0, 0);
    for index in 0..block.eigenvalues.len() {
        let mu = block.eigenvalues[index];
        let edge = block.rank_floor(index);
        if mu > edge {
            out.0 += 1;
        } else if mu < -edge {
            out.2 += 1;
        } else {
            out.1 += 1;
        }
    }
    assert_eq!(out.1, block.band.len(), "the band carrier must list exactly the in-band directions");
    out
}

/// The part of `rhs` the band removes, `ΦW_Z W_Zᵀ rhs`.
fn band_projection(block: &ExactHessianSpectralBlock, rhs: &Array1<f64>) -> Array1<f64> {
    let coefficients = Array1::from_iter(
        block
            .band
            .iter()
            .map(|&index| block.eigenvectors.column(index).dot(rhs)),
    );
    block.band_metric_images.dot(&coefficients)
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// The audit's congruence: `A = diag(1e-10, 1)`, `B = I` and
/// `L = diag(1e5, 1)·Q·diag(1, √2)` with `Q` the 45° rotation.
fn audit_congruence() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let small = 1.0e-10_f64;
    let a = Array2::from_diag(&array![small, 1.0]);
    let rotation = array![[1.0, -1.0], [1.0, 1.0]] / 2.0_f64.sqrt();
    let l = Array2::from_diag(&array![1.0 / small.sqrt(), 1.0])
        .dot(&rotation)
        .dot(&Array2::from_diag(&array![1.0, 2.0_f64.sqrt()]));
    (a, Array2::<f64>::eye(2), l)
}

/// The audit's counterexample. `LᵀAL = diag(1, 2)` while `LᵀBL` is of order `1e10`. The
/// per-direction Euclidean rule this replaced compared the ordinary eigenvalues `1` and `2`
/// with `√ε·vᵀBv ≈ 74` and `≈ 149` and retained nothing after the change of coordinates,
/// having retained one direction before it.
#[test]
fn the_audit_congruence_keeps_the_retained_subspace_the_solve_and_the_value_2933() {
    let (a, identity, l) = audit_congruence();
    let small = a[[0, 0]];
    let a_moved = congruence(&a, &l);
    let b_moved = congruence(&identity, &l);
    assert!(
        (a_moved[[0, 0]] - 1.0).abs() <= 1.0e-9
            && (a_moved[[1, 1]] - 2.0).abs() <= 1.0e-9
            && a_moved[[0, 1]].abs() <= 1.0e-9,
        "the fixture must be the audit's transformed operator diag(1, 2), got {a_moved:?}"
    );
    assert!(
        b_moved[[0, 0]] > 1.0e9 && b_moved[[1, 1]] > 1.0e9,
        "the fixture's transformed metric must be of order 1e10, got {b_moved:?}"
    );
    let plain = pencil_block(&a, &identity, &identity);
    let moved = pencil_block(&a_moved, &b_moved, &b_moved);
    assert_eq!(census(&plain), (1, 1, 0), "the plain pencil retains the unit direction");
    assert_eq!(
        census(&moved),
        (1, 1, 0),
        "#2933 F07: the same pencil in nonorthogonal coordinates must retain the same subspace \
         (spectrum {:?}, edges {:?})",
        moved.eigenvalues,
        (0..2).map(|index| moved.rank_floor(index)).collect::<Vec<_>>()
    );

    let rhs = array![0.3, -0.7];
    let x = flatten(&plain.solve_stationarity(&border(&rhs)).expect("plain solve").step);
    let x_moved = flatten(
        &moved
            .solve_stationarity(&border(&l.t().dot(&rhs)))
            .expect("transformed solve")
            .step,
    );
    assert!(max_abs(&x) > 0.5, "non-vacuity: the retained direction must carry the response, got {x:?}");
    let back = l.dot(&x_moved);
    assert!(
        (&back - &x).iter().all(|d| d.abs() <= 1.0e-8 * (1.0 + max_abs(&x))),
        "#2933 F07: the transformed solve must be L⁻¹x: L·x'={back:?}, x={x:?}"
    );

    let none = Array1::<f64>::zeros(0);
    let value = SaeManifoldTerm::classify_exact_hessian_basin(&plain, &none, None, 0, "plain", None)
        .expect("plain value")
        .log_det;
    let value_moved =
        SaeManifoldTerm::classify_exact_hessian_basin(&moved, &none, None, 0, "moved", None)
            .expect("transformed value")
            .log_det;
    let jacobian = 2.0 * (1.0 / small.sqrt()).ln() + 2.0_f64.ln();
    assert!(
        (value_moved - value - jacobian).abs() <= 1.0e-8 * (1.0 + jacobian),
        "#2933 F07: the priced log-determinant must move by exactly 2·log|det L| = {jacobian:.12e}, \
         got {value:.12e} → {value_moved:.12e}"
    );
}

/// #2228 contract. A non-empty band is not a failed solve: the solve steps on the resolvable
/// complement and lists each direction it held out with its `|μ|` and edge. The root
/// refinement counts the hold and takes that complement step, skips where the band holds
/// every direction, and takes no step where the solve resolves a negative curvature, since
/// `−A⁺g` moves along it toward a saddle.
#[test]
fn a_band_direction_yields_the_complement_step_and_is_counted_by_the_root_refinement_2933() {
    let identity = Array2::<f64>::eye(3);
    let held = 1.0e-13_f64;
    let block = pencil_block(&Array2::from_diag(&array![3.0, held, 2.0]), &identity, &identity);
    assert_eq!(census(&block), (2, 1, 0), "the fixture must hold exactly one direction in the band");

    let rhs = array![6.0, 5.0, 4.0];
    let solve = block
        .solve_stationarity(&border(&rhs))
        .expect("#2228: a non-empty band is not a failed solve");
    let step = flatten(&solve.step);
    assert!(
        (step[0] - 2.0).abs() <= 1.0e-12 && step[1].abs() <= 1.0e-12 && (step[2] - 2.0).abs() <= 1.0e-12,
        "the step must be A⁺rhs on the resolvable complement, got {step:?}"
    );
    assert_eq!(solve.retained_rank, 2, "the complement retains both positive directions");
    assert_eq!(solve.negative_curvature, None, "the fixture resolves no negative curvature");
    assert_eq!(solve.band.len(), 1, "the band list must name the held direction");
    // `dim·ε·‖A‖₂` is the eigensolver's backward error on this fixture.
    assert!(
        (solve.band[0].magnitude - held).abs() <= 9.0 * f64::EPSILON
            && solve.band[0].edge == sae_exact_a_pencil_floor(),
        "the held direction must carry its own |μ| and the pencil floor as its edge, got {:?}",
        solve.band[0]
    );

    // `(band holds, band skips, solve failures, negative-curvature no-steps)`.
    let counts = |term: &SaeManifoldTerm| {
        let counts = term.evidence_root_telemetry.counts();
        (
            counts.band_holds,
            counts.band_skips,
            counts.solve_failures,
            counts.negative_curvature_no_steps,
        )
    };
    let term = crate::manifold::tests::trivial_k1_euclidean_term();
    assert_eq!(counts(&term), (0, 0, 0, 0), "a fresh term starts with an empty ledger");
    let root_step = flatten(
        &term
            .evidence_root_step_from_pencil(Ok(solve))
            .expect("the resolvable complement carries a root step"),
    );
    assert_eq!(root_step, -&step, "the root step is −A⁺g on the resolvable complement");
    assert_eq!(counts(&term), (1, 0, 0, 0), "a band hold that still steps is a hold");

    let resolved = pencil_block(&Array2::from_diag(&array![3.0, 1.0, 2.0]), &identity, &identity);
    let resolved_solve = resolved
        .solve_stationarity(&border(&rhs))
        .expect("a full-rank solve");
    assert!(resolved_solve.band.is_empty(), "the resolved fixture must have an empty band");
    term.evidence_root_step_from_pencil(Ok(resolved_solve))
        .expect("a full-rank solve carries a root step");
    assert_eq!(counts(&term), (1, 0, 0, 0), "an empty band must not be counted");

    // A resolved negative curvature: `A⁺` keeps it with `1/μ < 0`, so `−A⁺g` would climb it.
    let saddle = pencil_block(&Array2::from_diag(&array![3.0, 1.0, -2.0]), &identity, &identity);
    let saddle_solve = saddle
        .solve_stationarity(&border(&rhs))
        .expect("an indefinite full-rank solve is not a failed solve");
    let negative = saddle_solve
        .negative_curvature
        .expect("the solve must report the resolved negative curvature");
    // `dim·ε·‖A‖₂` is the eigensolver's backward error on this fixture.
    assert!(
        negative.directions == 1
            && (negative.min_curvature + 2.0).abs() <= 9.0 * f64::EPSILON
            && negative.edge == sae_exact_a_pencil_floor(),
        "the report must carry the one negative direction, its μ and its edge, got {negative:?}"
    );
    assert!(
        term.evidence_root_step_from_pencil(Ok(saddle_solve)).is_none(),
        "a resolved negative curvature leaves no root step"
    );
    assert_eq!(counts(&term), (1, 0, 0, 1), "a negative curvature is its own no-step");

    // `−held` sits inside the band, so no direction here is a resolved negative curvature.
    let flat = pencil_block(&Array2::from_diag(&array![held, 2.0 * held, -held]), &identity, &identity);
    let flat_solve = flat
        .solve_stationarity(&border(&rhs))
        .expect("#2228: a band holding every direction is not a failed solve");
    assert_eq!((flat_solve.retained_rank, flat_solve.band.len()), (0, 3));
    assert_eq!(flat_solve.negative_curvature, None, "an in-band negative μ is not resolved");
    assert!(
        term.evidence_root_step_from_pencil(Ok(flat_solve)).is_none(),
        "a band holding every direction leaves no root step"
    );
    assert_eq!(counts(&term), (1, 1, 0, 1), "a band holding every direction is a skip, not a hold");

    // The outer objective restores a saved clone after a value probe; the ledger is shared, so
    // an outcome recorded on the clone that runs the probe survives the restore.
    let probe = term.clone();
    assert!(
        probe
            .evidence_root_step_from_pencil(Err("a failed certificate".to_string()))
            .is_none(),
        "a failed solve leaves no root step"
    );
    assert_eq!(counts(&term), (1, 1, 1, 1), "a failed solve is counted on the shared ledger");
}

/// A declared gauge direction keeps its unit pin in both `Φ` and `A` (the raw restoration skips
/// gauge-only pins), so a residual-driven curvature along it moves `μ` off 1 by the residual:
/// the direction is retained and never priced as a basin. The same curvature along an
/// undeclared direction, whose metric carries only its measured raw curvature, is a resolved
/// negative direction, and without a clamp the basin refuses it. An exact symmetry is declared,
/// not inferred from its sign. The curvature is the one routes' diagnostic 1149614 read on the
/// support fixture's frozen evidence leg (`h = 1e−5`, isotropic plane ARD `[1.0, 1.0]`), where
/// an undeclared plane rotation refused as a saddle and `[1.0, 1.5]` removed the null.
#[test]
fn a_declared_gauge_pin_with_residual_curvature_is_retained_not_refused_2933() {
    let residual_curvature = -4.000404e-6_f64;
    let none = Array1::<f64>::zeros(0);

    let declared_metric = Array2::from_diag(&array![2.0, 1.0]);
    let declared_a = Array2::from_diag(&array![2.0, 1.0 + residual_curvature]);
    let declared = pencil_block(&declared_a, &declared_metric, &declared_metric);
    assert_eq!(
        census(&declared),
        (2, 0, 0),
        "a declared gauge pin must be retained (spectrum {:?})",
        declared.eigenvalues
    );
    let value = SaeManifoldTerm::classify_exact_hessian_basin(&declared, &none, None, 0, "declared", None)
        .expect("a declared gauge direction is not a saddle")
        .log_det;
    // `log|Φ| + Σ ln μ` with `μ = (1, 1 + r)`; the reduction's backward error is `dim·ε·κ(Φ)`.
    let expected = 2.0_f64.ln() + residual_curvature.ln_1p();
    assert!(
        (value - expected).abs() <= 2.0 * 2.0 * f64::EPSILON * (1.0 + expected.abs()),
        "the pin prices within the residual of log 1: {value:.16e} vs {expected:.16e}"
    );

    let measured = 1.0e-4_f64;
    let undeclared_metric = Array2::from_diag(&array![2.0, measured]);
    let undeclared_a = Array2::from_diag(&array![2.0, residual_curvature]);
    let undeclared = pencil_block(&undeclared_a, &undeclared_metric, &undeclared_metric);
    assert_eq!(
        census(&undeclared),
        (1, 0, 1),
        "an undeclared residual curvature is a resolved negative direction (spectrum {:?}, edges {:?})",
        undeclared.eigenvalues,
        (0..2).map(|index| undeclared.rank_floor(index)).collect::<Vec<_>>()
    );
    assert!(
        SaeManifoldTerm::classify_exact_hessian_basin(&undeclared, &none, None, 0, "undeclared", None)
            .is_err(),
        "without a declaration or a clamp, a resolved negative direction is a saddle"
    );
}

/// The audit's congruence through production's own metric. A border-only evidence factor
/// whose reduced Schur is the metric itself, so `ArrowMetric::Joint` whitens with the
/// cache's Schur Cholesky factor, as it does on every production state. The transformed
/// metric has condition number `≈ 1e10`, over the Newton step's Schur stability bound, so
/// the step takes the production Newton--Schur Tikhonov floor; that floor never enters the
/// evidence factor, which is the strict Cholesky of the metric.
#[test]
fn the_audit_congruence_keeps_the_retained_count_through_the_evidence_factor_2933() {
    let (a, identity, l) = audit_congruence();
    let mut censuses = Vec::new();
    for (operator, metric) in [
        (a.clone(), identity.clone()),
        (congruence(&a, &l), congruence(&identity, &l)),
    ] {
        let mut system = ArrowSchurSystem::new(0, 0, 2);
        system.hbb = metric;
        let (_, _, cache) = solve_arrow_newton_step_with_options(
            &system,
            0.0,
            0.0,
            &ArrowSolveOptions::direct()
                .with_positive_definite_evidence()
                .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR),
        )
        .expect("border-only evidence factor");
        let prepared = ArrowMetric::Joint(&cache)
            .prepare()
            .expect("prepared evidence metric");
        let block = SaeManifoldTerm::exact_hessian_spectral_block(operator, &prepared)
            .expect("pencil geometry through the evidence factor");
        censuses.push(census(&block));
    }
    assert_eq!(
        censuses,
        vec![(1, 1, 0), (1, 1, 0)],
        "#2933 F07: the evidence factor's pencil must retain one direction in both coordinate \
         systems (retained, in band, negative)"
    );
}

/// A pencil with every stratum a consumer reads: three retained positive directions (one of
/// them at `3√ε`), an in-band null, a direction only the evidence factor's substituted
/// stiffness puts in the band, and a resolved negative direction the clamp remainder
/// explains. Three nonorthogonal coordinate changes `T = shear·scale·reflection`, whose
/// scales span two decades, must move none of the counts, the band's dual projection, the
/// solve or the value beyond the Jacobian.
#[test]
fn nonorthogonal_congruences_move_neither_the_band_nor_the_solve_nor_the_value_2933() {
    let dim = 6;
    let floor = sae_exact_a_pencil_floor();
    let lower = Array2::from_shape_fn((dim, dim), |(i, j)| {
        if i == j {
            1.0 + 0.3 * i as f64
        } else if i > j {
            0.2 * ((i + 2 * j) as f64).sin()
        } else {
            0.0
        }
    });
    let frame = householder(dim, 0.37);
    let mu = array![-3.0, 5.0e-11, 0.2, 3.0 * floor, 2.5, 7.0];
    let metric = lower.dot(&lower.t());
    let a = {
        let operator = lower
            .dot(&frame)
            .dot(&Array2::from_diag(&mu))
            .dot(&frame.t())
            .dot(&lower.t());
        (&operator + &operator.t()) * 0.5
    };
    let basis = lower_transpose_solve(&lower, &frame);
    // The evidence factor substituted 0.9 of the unit metric along the 0.2 direction.
    let pinned = metric.dot(&basis.column(2));
    let raw = &metric - &(0.9 * outer(&pinned, &pinned));
    // A clamp remainder explaining the negative direction: its basin curvature is −3 + 4 = 1.
    let clamped = metric.dot(&basis.column(0));
    let clamp = 4.0 * outer(&clamped, &clamped);
    let none = Array1::<f64>::zeros(0);
    let rhs = Array1::from_iter((0..dim).map(|i| 0.5 - 0.17 * i as f64));
    use gam_linalg::faer_ndarray::FaerEigh;
    let condition = |m: &Array2<f64>| -> f64 {
        let (values, _) = m.eigh(Side::Lower).expect("symmetric eigendecomposition");
        let (low, high) = values
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(low, high), &value| (low.min(value.abs()), high.max(value.abs())));
        high / low
    };

    let plain = pencil_block(&a, &metric, &raw);
    let plain_census = census(&plain);
    assert_eq!(
        plain_census,
        (3, 2, 1),
        "the fixture must populate every stratum (spectrum {:?})",
        plain.eigenvalues
    );
    // The `3√ε` direction exercises the edge of the counts, but a computed eigenvector there is
    // resolved only to `ε‖Ã‖/gap` against its in-band neighbour, and every consumer that inverts
    // or takes the logarithm of that curvature amplifies the round-off by `1/μ ≈ 2e7` in any
    // coordinates. The band's projection, the solve and the value are therefore read off the same
    // pencil with that direction resolved, and each bar is the first-order forward error of the
    // whitened reduction: `dim·ε·κ(Φ')` backward, times the retained condition where a curvature
    // is inverted, times `κ(T)` where the result is mapped back.
    let mut mu_resolved = mu.clone();
    mu_resolved[3] = 1.3;
    let a_resolved = {
        let operator = lower
            .dot(&frame)
            .dot(&Array2::from_diag(&mu_resolved))
            .dot(&frame.t())
            .dot(&lower.t());
        (&operator + &operator.t()) * 0.5
    };
    let resolved = pencil_block(&a_resolved, &metric, &raw);
    assert_eq!(census(&resolved), (3, 2, 1), "the resolved pencil must keep every stratum");
    let (retained_low, retained_high) = (0..dim)
        .filter(|index| !resolved.band.contains(index))
        .map(|index| resolved.eigenvalues[index].abs())
        .fold((f64::INFINITY, 0.0_f64), |(low, high), value| (low.min(value), high.max(value)));
    let retained_condition = retained_high / retained_low;
    let x = flatten(&resolved.solve_stationarity(&border(&rhs)).expect("plain solve").step);
    assert!(max_abs(&x) > 1.0e-2, "non-vacuity: the solve must carry a response, got {x:?}");
    let projection = band_projection(&resolved, &rhs);
    assert!(max_abs(&projection) > 1.0e-3, "non-vacuity: the band must remove a component");
    let value = SaeManifoldTerm::classify_exact_hessian_basin(&resolved, &none, Some(&clamp), 0, "plain", None)
        .expect("the clamp explains the negative direction")
        .log_det;

    for trial in 0..3 {
        let shear = Array2::from_shape_fn((dim, dim), |(i, j)| {
            if i == j {
                1.0
            } else if j > i {
                0.6 * (((trial + 1) * (i + 3 * j)) as f64).cos()
            } else {
                0.0
            }
        });
        let scales = Array1::from_iter((0..dim).map(|i| {
            10.0_f64.powf(-1.0 + 2.0 * ((i + trial) % dim) as f64 / (dim - 1) as f64)
        }));
        let reflection = householder(dim, 0.11 + 0.2 * trial as f64);
        let t = shear.dot(&Array2::from_diag(&scales)).dot(&reflection);
        let jacobian = 2.0 * scales.iter().map(|s| s.ln()).sum::<f64>();

        let moved = pencil_block(
            &congruence(&a, &t),
            &congruence(&metric, &t),
            &congruence(&raw, &t),
        );
        assert_eq!(
            census(&moved),
            plain_census,
            "trial {trial}: a nonorthogonal change of coordinates moved the classification \
             (spectrum {:?})",
            moved.eigenvalues
        );
        let metric_moved = congruence(&metric, &t);
        let moved_resolved =
            pencil_block(&congruence(&a_resolved, &t), &metric_moved, &congruence(&raw, &t));
        assert_eq!(
            census(&moved_resolved),
            (3, 2, 1),
            "trial {trial}: a nonorthogonal change of coordinates moved the resolved classification"
        );
        let metric_condition = condition(&metric_moved);
        let backward = dim as f64 * f64::EPSILON * metric_condition;
        let coordinate_condition = condition(&t.t().dot(&t)).sqrt();
        // A band direction's eigenvector mixes with the retained range at `‖Ã‖/gap`.
        let band_high = resolved
            .band
            .iter()
            .map(|&index| resolved.eigenvalues[index].abs())
            .fold(0.0_f64, f64::max);
        let amplification = retained_high / (retained_low - band_high).min(retained_low);

        let moved_rhs = t.t().dot(&rhs);
        let x_moved = flatten(
            &moved_resolved
                .solve_stationarity(&border(&moved_rhs))
                .expect("moved solve")
                .step,
        );
        let solve_error = max_abs(&(&t.dot(&x_moved) - &x));
        let solve_bar =
            backward * (retained_condition + amplification) * coordinate_condition * (1.0 + max_abs(&x));

        let expected_projection = t.t().dot(&projection);
        let projection_error =
            max_abs(&(&band_projection(&moved_resolved, &moved_rhs) - &expected_projection));
        let projection_bar =
            backward * amplification * coordinate_condition * (1.0 + max_abs(&expected_projection));

        let value_moved = SaeManifoldTerm::classify_exact_hessian_basin(
            &moved_resolved,
            &none,
            Some(&congruence(&clamp, &t)),
            0,
            "moved",
            None,
        )
        .expect("the moved clamp explains the moved negative direction")
        .log_det;
        let value_error = (value_moved - value - jacobian).abs();
        let value_bar = dim as f64 * backward * retained_condition * (1.0 + value.abs() + jacobian.abs());

        eprintln!(
            "#2933 F07 congruence trial {trial}: κ(Φ')={metric_condition:.3e} κ(T)={coordinate_condition:.3e} \
             solve {solve_error:.3e}/{solve_bar:.3e} projection {projection_error:.3e}/{projection_bar:.3e} \
             value {value_error:.3e}/{value_bar:.3e}"
        );
        assert!(
            solve_error <= solve_bar,
            "trial {trial}: the transformed solve must be T⁻¹x (error {solve_error:.3e}, bar \
             {solve_bar:.3e}): x'={x_moved:?}, x={x:?}"
        );
        assert!(
            projection_error <= projection_bar,
            "trial {trial}: the band's dual projection must transform as Tᵀ (error \
             {projection_error:.3e}, bar {projection_bar:.3e})"
        );
        assert!(
            value_error <= value_bar,
            "trial {trial}: the priced log-determinant must move by 2·log|det T| = {jacobian:.12e} \
             (error {value_error:.3e}, bar {value_bar:.3e}), got {value:.12e} → {value_moved:.12e}"
        );
    }
}

/// The value prices an in-band direction at the metric's own curvature, so on its stratum it
/// moves with `Φ` there, and the differential carries a `dΦ` weight. Along a path that holds
/// the in-band curvature at `1e-10` while every operand moves, a central difference of the
/// priced value must equal `⟨X_A, dA⟩ + ⟨X_Φ, dΦ⟩ + ⟨X_E, dE⟩`, and the `dΦ` part must be live.
#[test]
fn the_priced_value_differentiates_along_the_metric_as_well_as_the_operator_2933() {
    let dim = 7;
    let lower = Array2::from_shape_fn((dim, dim), |(i, j)| {
        if i == j {
            1.2 + 0.25 * i as f64
        } else if i > j {
            0.15 * ((3 * i + j) as f64).cos()
        } else {
            0.0
        }
    });
    let metric0 = lower.dot(&lower.t());
    let frame = householder(dim, 0.29);
    let mu0 = array![-0.3, 1.0e-10, 0.4, 1.3, 2.0, 5.0, 9.0];
    let dmu = array![0.7, 0.0, -0.2, 0.9, 0.4, -1.1, 0.3];
    let e0 = Array1::from_iter((0..dim).map(|i| 1.0 + i as f64 / dim as f64));
    let e1 = Array1::from_iter((0..dim).map(|i| (i as f64 + 0.5).sin()));
    let root = |m: &Array2<f64>| {
        let (values, vectors) = m.eigh(Side::Lower).expect("symmetric metric");
        vectors.dot(&Array2::from_diag(&values.mapv(f64::sqrt))).dot(&vectors.t())
    };
    // A metric motion with a component along the in-band direction `w_z = Φ^{-1/2}f_1`, which
    // only the dΦ weight reads: `w_zᵀ(Φ^{1/2}f_1 f_1ᵀΦ^{1/2})w_z = 1`.
    let band_image = root(&metric0).dot(&frame.column(1));
    let metric1 = Array2::from_shape_fn((dim, dim), |(i, j)| 0.05 * ((i + j + 1) as f64 * 0.7).cos())
        + outer(&band_image, &band_image);
    let state = |s: f64| {
        let metric = &metric0 + &(s * &metric1);
        let half = root(&metric);
        let operator = half
            .dot(&frame)
            .dot(&Array2::from_diag(&(&mu0 + &(s * &dmu))))
            .dot(&frame.t())
            .dot(&half);
        let operator = (&operator + &operator.t()) * 0.5;
        (operator, metric, Array2::from_diag(&(&e0 + &(s * &e1))))
    };
    let none = Array1::<f64>::zeros(0);
    let value = |s: f64| {
        let (operator, metric, clamp) = state(s);
        let block = pencil_block(&operator, &metric, &metric);
        SaeManifoldTerm::classify_exact_hessian_basin(&block, &none, Some(&clamp), 0, "path", None)
            .expect("priced basin along the path")
            .log_det
    };

    let (operator, metric, clamp) = state(0.0);
    let block = pencil_block(&operator, &metric, &metric);
    assert_eq!(census(&block), (5, 1, 1), "the path must hold one in-band and one negative direction");
    let basin =
        SaeManifoldTerm::classify_exact_hessian_basin(&block, &none, Some(&clamp), 0, "path", None)
            .expect("priced basin");
    assert!(
        basin.inverse_values.iter().any(|&inverse| inverse > 0.0),
        "the clamp must price the negative direction"
    );
    let pricing = SaeManifoldTerm::exact_hessian_basin_differential(&block, &none, Some(&clamp), 0, &basin)
        .expect("stratum differential");

    let h = 1.0e-5;
    let (plus_operator, plus_metric, plus_clamp) = state(h);
    let (minus_operator, minus_metric, minus_clamp) = state(-h);
    let d_operator = (&plus_operator - &minus_operator) / (2.0 * h);
    let d_metric = (&plus_metric - &minus_metric) / (2.0 * h);
    let d_clamp = (&plus_clamp - &minus_clamp) / (2.0 * h);
    let fd = (value(h) - value(-h)) / (2.0 * h);
    let from_operator = (&pricing.a_derivative * &d_operator).sum();
    let from_metric = (&pricing.metric_derivative * &d_metric).sum();
    let from_clamp = (&pricing.clamp_border_derivative * &d_clamp).sum();
    let analytic = from_operator + from_metric + from_clamp;
    assert!(
        from_metric.abs() > 1.0e-2 * (1.0 + fd.abs()),
        "non-vacuity: the dΦ weight must carry signal on this path (⟨X_Φ, dΦ⟩={from_metric:e}, fd={fd:e})"
    );
    assert!(
        (fd - analytic).abs() <= 1.0e-6 * (1.0 + fd.abs()),
        "#2933 F07: the stratum differential must match the priced value: fd={fd:.12e}, analytic \
         {analytic:.12e} (dA {from_operator:e}, dΦ {from_metric:e}, dE {from_clamp:e})"
    );
}

/// `PreparedArrowMetric` is one factorization of the metric it applies: on a state whose
/// evidence factor conditions rows, and on a lifted border, `L⁻¹ΦL⁻ᵀ = I`, `log|Φ|` is the
/// log-determinant of the applied `Φ`, and the substituted image is `Φ − B_raw`.
#[test]
fn the_prepared_metric_factors_the_metric_it_applies_2933() {
    let (mut term, target, rho) =
        crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.0, -6.0);
    term.penalized_quasi_laplace_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
        .expect("the Patch-D fixture must converge to its own mode");
    let system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("fixed-state assembly");
    let (_, _, cache) =
        solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &term.evidence_factor_options())
            .expect("fixed-state evidence factor");
    let total_t = cache.delta_t_len();
    let k = cache.k;
    assert!(k > 1, "the fixture must carry a border to lift");
    let joint_dim = total_t + k;
    let mut raw = Array2::<f64>::zeros((joint_dim, joint_dim));
    let mut unit = Array1::<f64>::zeros(joint_dim);
    for column in 0..joint_dim {
        unit[column] = 1.0;
        let applied = crate::manifold::arrow_solver::apply_raw_cached_arrow_hessian(
            &cache,
            unit.slice(s![..total_t]),
            unit.slice(s![total_t..]),
        )
        .expect("raw majorizer apply");
        raw.column_mut(column).assign(&flatten(&applied));
        unit[column] = 0.0;
    }
    let width = k - 1;
    let lift = Array2::from_shape_fn((k, width), |(row, column)| {
        if row == column {
            1.0
        } else {
            0.3 * (((row + 2 * column) as f64) * 0.7).sin()
        }
    });
    for (label, metric, embedding) in [
        ("joint", ArrowMetric::Joint(&cache), Array2::<f64>::eye(joint_dim)),
        (
            "lifted",
            ArrowMetric::JointLifted {
                cache: &cache,
                lift: &lift,
            },
            {
                let mut embedding = Array2::<f64>::zeros((joint_dim, total_t + width));
                for index in 0..total_t {
                    embedding[[index, index]] = 1.0;
                }
                embedding.slice_mut(s![total_t.., total_t..]).assign(&lift);
                embedding
            },
        ),
    ] {
        let prepared = metric.prepare().expect("prepared metric");
        let dim = prepared.dim();
        let mut dense = Array2::<f64>::zeros((dim, dim));
        let mut substituted = Array2::<f64>::zeros((dim, dim));
        let half = prepared
            .lower_transpose_solve(Array2::<f64>::eye(dim).view())
            .expect("transpose triangular solve");
        let mut image = Array2::<f64>::zeros((dim, dim));
        let mut unit = Array1::<f64>::zeros(dim);
        for column in 0..dim {
            unit[column] = 1.0;
            dense
                .column_mut(column)
                .assign(&prepared.apply(unit.view()).expect("metric apply"));
            substituted
                .column_mut(column)
                .assign(&prepared.substituted_image(unit.view()).expect("substituted image"));
            image
                .column_mut(column)
                .assign(&prepared.apply(half.column(column)).expect("metric apply"));
            unit[column] = 0.0;
        }
        let whitened = prepared.lower_solve(image.view()).expect("triangular solve");
        let identity_error = (&whitened - &Array2::<f64>::eye(dim))
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(
            identity_error <= f64::EPSILON.sqrt(),
            "{label}: L⁻¹ΦL⁻ᵀ must be the identity, max error {identity_error:.3e}"
        );
        let (values, _) = ((&dense + &dense.t()) * 0.5)
            .eigh(Side::Lower)
            .expect("metric eigendecomposition");
        let dense_log_det: f64 = values.iter().map(|value| value.ln()).sum();
        let factored_log_det = prepared.log_det().expect("metric log-determinant");
        assert!(
            (factored_log_det - dense_log_det).abs() <= f64::EPSILON.sqrt() * (1.0 + dense_log_det.abs()),
            "{label}: log|Φ| from the factor {factored_log_det:.12e} != dense {dense_log_det:.12e}"
        );
        // #2267: the block builder reads ‖Φ‖_F off the metric's entries, not `dim` applies.
        let dense_frobenius = dense.iter().map(|value| value * value).sum::<f64>().sqrt();
        let entries_frobenius = prepared.frobenius_norm().expect("metric Frobenius norm");
        assert!(
            (entries_frobenius - dense_frobenius).abs() <= f64::EPSILON.sqrt() * dense_frobenius,
            "{label}: ‖Φ‖_F from the entries {entries_frobenius:.12e} != dense {dense_frobenius:.12e}"
        );
        let expected = &dense - &embedding.t().dot(&raw).dot(&embedding);
        let scale = dense.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let substitution_error = (&substituted - &expected)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(
            substitution_error <= f64::EPSILON.sqrt() * scale,
            "{label}: the substituted image must be Φ − B_raw, max error {substitution_error:.3e} \
             against scale {scale:.3e}"
        );

        // #2933 F36/F39: the whitening solves a whole block of right-hand sides at once, so
        // its sums run in another order than one column at a time, and the result is compared
        // to rounding, not to the bit. A triangular solve is exact for `L + ΔL`,
        // `|ΔL| ≤ γ_n|L|` (Higham, Thm 8.5), so each side of one whitening moves `Ã = L⁻¹AL⁻ᵀ`
        // by at most `γ_n·cond(L)·‖Ã‖₂`, with `cond(L) = ‖|L⁻¹||L|‖ ≤ n·κ₂(Φ)^½`. Two sides
        // of two whitenings give `4γ_n·n·κ₂(Φ)^½·‖Ã‖₂`.
        let operator = {
            let pulled_back = embedding.t().dot(&raw).dot(&embedding);
            (&pulled_back + &pulled_back.t()) * 0.5
        };
        let unit_roundoff = 0.5 * f64::EPSILON;
        let gamma = dim as f64 * unit_roundoff / (1.0 - dim as f64 * unit_roundoff);
        let condition = values[dim - 1] / values[0];
        let solve_band = |whitened_norm: f64| {
            4.0 * gamma * dim as f64 * condition.sqrt() * whitened_norm
        };
        let block = prepared
            .lower_solve(
                prepared
                    .lower_solve(operator.view())
                    .expect("block triangular solve")
                    .t(),
            )
            .expect("block triangular solve");
        let columnwise = |rhs: ArrayView2<'_, f64>| -> Array2<f64> {
            let mut out = Array2::<f64>::zeros(rhs.raw_dim());
            for column in 0..rhs.ncols() {
                out.column_mut(column).assign(
                    &prepared
                        .lower_solve(rhs.slice(s![.., column..column + 1]))
                        .expect("one-column triangular solve")
                        .column(0),
                );
            }
            out
        };
        let by_columns = columnwise(columnwise(operator.view()).t());
        let symmetrized = (&block + &block.t()) * 0.5;
        let (block_spectrum, _) = symmetrized.eigh(Side::Lower).expect("whitened eigendecomposition");
        let whitened_norm = block_spectrum.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(
            whitened_norm > 0.0,
            "{label}: non-vacuity, the pulled-back majorizer must whiten to a nonzero operator"
        );
        let order_error = (&block - &by_columns).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        eprintln!(
            "[2933-F39] {label}: dim={dim} κ₂(Φ)={condition:.3e} ‖Ã‖₂={whitened_norm:.3e}: block vs \
             column max error {order_error:.3e}, band {:.3e}",
            solve_band(whitened_norm)
        );
        assert!(
            order_error <= solve_band(whitened_norm),
            "{label}: the block whitening must be the column-by-column whitening to rounding: \
             max error {order_error:.3e} against the band {:.3e}",
            solve_band(whitened_norm)
        );

        // The pencil spectrum against an independent whitening by the dense Cholesky factor of
        // the applied `Φ`. A factor with `L⁻¹ΦL⁻ᵀ = I + F` whitens the pencil of `L·Lᵀ`, whose
        // eigenvalues lie within a factor `[1/(1+‖F‖), 1/(1−‖F‖)]` of the pencil `(A, Φ)`'s
        // (Ostrowski), so each factor contributes `‖F‖/(1−‖F‖)·|λᵢ|` and its solves the band above.
        use gam_linalg::faer_ndarray::FaerCholesky;
        use gam_linalg::triangular::forward_substitution_lower_matrix;
        let symmetric_metric = (&dense + &dense.t()) * 0.5;
        let reference_lower = symmetric_metric
            .cholesky(Side::Lower)
            .expect("dense Cholesky of the applied metric")
            .lower_triangular();
        let reference_whiten = |matrix: &Array2<f64>| {
            forward_substitution_lower_matrix(
                &reference_lower,
                forward_substitution_lower_matrix(&reference_lower, matrix).t(),
            )
        };
        let frobenius = |matrix: &Array2<f64>| matrix.iter().map(|x| x * x).sum::<f64>().sqrt();
        let block_residual = frobenius(&(&whitened - &Array2::<f64>::eye(dim)));
        let reference_residual =
            frobenius(&(&reference_whiten(&symmetric_metric) - &Array2::<f64>::eye(dim)));
        let reference = reference_whiten(&operator);
        let (reference_spectrum, _) = ((&reference + &reference.t()) * 0.5)
            .eigh(Side::Lower)
            .expect("reference whitened eigendecomposition");
        let mut worst = (0.0_f64, 0);
        for index in 0..dim {
            let (ours, theirs) = (block_spectrum[index], reference_spectrum[index]);
            let band = block_residual / (1.0 - block_residual) * ours.abs()
                + reference_residual / (1.0 - reference_residual) * theirs.abs()
                + solve_band(whitened_norm);
            if (ours - theirs).abs() / band > worst.0 {
                worst = ((ours - theirs).abs() / band, index);
            }
            assert!(
                (ours - theirs).abs() <= band,
                "{label}: pencil eigenvalue {index} of the block whitening {ours:.12e} is not the \
                 dense metric's {theirs:.12e} within the band {band:.3e} (residuals ‖F‖ = \
                 {block_residual:.3e}, {reference_residual:.3e})"
            );
        }
        eprintln!(
            "[2933-F39] {label}: residuals ‖F‖ = {block_residual:.3e} (block), \
             {reference_residual:.3e} (dense); worst eigenvalue deviation {:.3e} of its band at \
             index {}",
            worst.0, worst.1
        );
    }
}

/// The metric weight is contracted against `dB_raw` through [`SaeManifoldTerm::evidence_metric_raw_weight`],
/// which folds the evidence factor's row pins and reduced-Schur conditioning into it. A
/// central difference of `⟨X, Φ(ρ)⟩`, with `Φ` materialized off a fresh evidence factor at
/// each endpoint and the state held fixed, must equal the analytic trace for every outer
/// coordinate the fixture carries.
#[test]
fn the_metric_derivative_differentiates_the_conditioned_evidence_factor_2933() {
    let (mut term, target, rho) =
        crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.0, -6.0);
    term.penalized_quasi_laplace_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
        .expect("the Patch-D fixture must converge to its own mode");
    let options = term.evidence_factor_options();
    let factor = |term: &mut SaeManifoldTerm, at: &SaeManifoldRho| -> ArrowFactorCache {
        let system = term
            .assemble_arrow_schur(target.view(), at, None)
            .expect("fixed-state assembly");
        let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("fixed-state evidence factor");
        cache
    };
    let cache = factor(&mut term, &rho);
    eprintln!(
        "#2933 F07 metric ρ-derivative: {} spectrally conditioned rows, reduced-Schur \
         conditioning recorded={}",
        cache.deflation_row_spectra.iter().filter(|spectrum| spectrum.is_some()).count(),
        cache.beta_schur_conditioning.is_some(),
    );
    let dim = cache.delta_t_len() + cache.k;
    let weight = Array2::from_shape_fn((dim, dim), |(i, j)| {
        ((i + j + 1) as f64 * 0.23).cos() / (1.0 + (i as f64 - j as f64).abs())
    });
    let (trace, _) = term
        .evidence_metric_derivative_channels(&rho, target.view(), &cache, &weight)
        .expect("metric derivative channels");

    let mut coordinates: Vec<(usize, Box<dyn Fn(&mut SaeManifoldRho, f64)>)> = Vec::new();
    if let Some(flat) = rho.sparse_flat_index() {
        coordinates.push((flat, Box::new(|at: &mut SaeManifoldRho, step| at.log_lambda_sparse += step)));
    }
    for atom in 0..rho.log_lambda_smooth.len() {
        coordinates.push((
            rho.smooth_flat_index(atom),
            Box::new(move |at: &mut SaeManifoldRho, step| at.log_lambda_smooth[atom] += step),
        ));
    }
    for atom in 0..rho.log_ard.len() {
        for axis in 0..rho.log_ard[atom].len() {
            coordinates.push((
                rho.ard_flat_index(atom, axis),
                Box::new(move |at: &mut SaeManifoldRho, step| at.log_ard[atom][axis] += step),
            ));
        }
    }
    let h = 1.0e-6;
    let mut live = 0usize;
    for (flat, perturb) in &coordinates {
        let mut plus = rho.clone();
        perturb(&mut plus, h);
        let mut minus = rho.clone();
        perturb(&mut minus, -h);
        let plus_metric = crate::manifold::tests::dense_evidence_metric(&factor(&mut term, &plus));
        let minus_metric = crate::manifold::tests::dense_evidence_metric(&factor(&mut term, &minus));
        let fd = ((&weight * &plus_metric).sum() - (&weight * &minus_metric).sum()) / (2.0 * h);
        let analytic = 2.0 * trace[*flat];
        assert!(
            (fd - analytic).abs() <= 1.0e-5 * (1.0 + analytic.abs()),
            "#2933 F07: ρ[{flat}]: analytic ⟨X, dΦ⟩={analytic:e}, fd={fd:e}"
        );
        live += usize::from(analytic.abs() > 1.0e-4);
    }
    assert!(live > 0, "non-vacuity: the metric trace must carry signal on some coordinate");
}

/// The θ half of the metric channel: along a joint state direction `d`, a central difference
/// of `⟨X, Φ(θ)⟩`, with a fresh evidence factor at each endpoint and ρ held fixed, must equal
/// `⟨Γ_Φ, d⟩`. Softmax carries the entropy majorizer leg, and ordered Beta--Bernoulli the
/// majorized prior diagonal's row-local and shared-mass legs. The step's own error is read off
/// two step sizes, and the round-off of a `dim²`-term contraction is added to it.
#[test]
fn the_metric_derivative_differentiates_the_evidence_factor_along_the_state_2933() {
    let softmax = {
        let (mut term, target, mut rho) =
            crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();
        rho.log_lambda_sparse = 0.0;
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for value in axis.iter_mut() {
                *value = -1.0;
            }
        }
        term.penalized_quasi_laplace_criterion_with_cache(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .expect("the moderate-penalty softmax basin converges");
        ("softmax", term, target, rho)
    };
    let ordered = {
        let (mut term, target, rho) =
            crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.0, -6.0);
        term.penalized_quasi_laplace_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
            .expect("the Patch-D fixture must converge to its own mode");
        ("ordered Beta--Bernoulli", term, target, rho)
    };
    for (label, mut term, target, rho) in [softmax, ordered] {
        let options = term.evidence_factor_options();
        let evidence_factor = |state: &mut SaeManifoldTerm| -> ArrowFactorCache {
            let system = state
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("fixed-ρ assembly");
            let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
                .expect("fixed-ρ evidence factor");
            cache
        };
        let cache = evidence_factor(&mut term);
        // A clone drops the collapse-prevention gates and would read live routing in the
        // barriers, so every endpoint declares the gates this state's assembly installed.
        let gates = term.collapse_prevention_gates();
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let weight = Array2::from_shape_fn((dim, dim), |(i, j)| {
            ((i + 2 * j + 1) as f64 * 0.19).sin() + ((2 * i + j + 1) as f64 * 0.19).sin()
        });
        let (_, gamma) = term
            .evidence_metric_derivative_channels(&rho, target.view(), &cache, &weight)
            .expect("metric derivative channels");
        let dt = Array1::from_shape_fn(total_t, |i| ((i + 1) as f64 * 0.61).sin());
        let db = Array1::from_shape_fn(k, |j| ((j + 2) as f64 * 0.43).cos());
        let analytic = gamma.t.dot(&dt) + gamma.beta.dot(&db);
        let contraction = |step: f64| -> (f64, f64) {
            let mut state = term.clone();
            state.declare_collapse_prevention_gates(&gates);
            // The step length is positive, so the backward endpoint walks the negated direction.
            let (direction_t, direction_beta) = if step > 0.0 {
                (dt.clone(), db.clone())
            } else {
                (-&dt, -&db)
            };
            state
                .apply_newton_step(direction_t.view(), direction_beta.view(), step.abs())
                .expect("perturbed state");
            let moved = evidence_factor(&mut state);
            assert_eq!(
                (moved.delta_t_len(), moved.k),
                (total_t, k),
                "{label}: the perturbation must keep the row layout"
            );
            let metric = crate::manifold::tests::dense_evidence_metric(&moved);
            (
                (&weight * &metric).sum(),
                weight
                    .iter()
                    .zip(metric.iter())
                    .map(|(x, m)| (x * m).abs())
                    .sum::<f64>(),
            )
        };
        let difference = |h: f64| -> (f64, f64) {
            let (plus, plus_scale) = contraction(h);
            let (minus, minus_scale) = contraction(-h);
            ((plus - minus) / (2.0 * h), plus_scale.max(minus_scale) / h)
        };
        let h = 1.0e-5;
        let (coarse, _) = difference(h);
        let (fine, roundoff_scale) = difference(0.5 * h);
        let tolerance = (coarse - fine).abs() + dim as f64 * f64::EPSILON * roundoff_scale;
        eprintln!(
            "#2933 F07 metric θ-derivative ({label}): analytic={analytic:.12e} fd(h)={coarse:.12e} \
             fd(h/2)={fine:.12e} tolerance={tolerance:.3e}"
        );
        assert!(
            analytic.abs() > 1.0e-4,
            "{label}: non-vacuity: the metric θ-adjoint must carry signal along the direction"
        );
        assert!(
            (fine - analytic).abs() <= tolerance,
            "#2933 F07 ({label}): ⟨Γ_Φ, d⟩={analytic:.12e} but the central difference of ⟨X, Φ(θ)⟩ \
             is {fine:.12e} (tolerance {tolerance:.3e})"
        );
    }
}
