//! #2515 measurement scaffold — what a bundle route handed the EXACT-`A`
//! geometry actually produces, channel by channel, against the dense exact-`A`
//! authority.
//!
//! The open half of #2515 is that route selection changes the ranked criterion:
//! the dense direct-logdet route prices `½log|A|` for `A = ∇²_θθ L`, while the
//! bundle / matrix-free route differentiates the Gauss--Newton majorizer `B`.
//! The repair has two independent halves, and this module exists to measure
//! which of them each ρ-coordinate needs before either is written:
//!
//! * the WRONG INVERSE — the from-probes channels reconstruct `(H⁻¹)_tt` as
//!   `A_i⁻¹ + G_i S⁻¹ G_iᵀ` off the factor cache they are handed, so handing
//!   them `B`'s cache contracts `B⁻¹` no matter which operator's probes arrive;
//! * the WRONG OPERATOR — `∂A/∂ρ = ∂B/∂ρ + ∂ΔC/∂ρ`, and `∂ΔC/∂ρ` is nonzero on
//!   exactly the coordinates `exact_stationarity_penalty_derivative_delta_by_flat`
//!   keys (ARD log-precision, and the softmax sparse log-strength at `K ≥ 2`).
//!
//! The probe below fixes the state, builds the exact-`A` evidence system, factors
//! it, forms the FULL-BASIS probe set `√k·e_j` with exact `S_A⁻¹ e_j` off that
//! factorization, and reports every channel both ways. Full-basis probes remove
//! stochastic approximation, so a residual is an authority defect and not probe
//! noise.

use super::construction::{ArrowMetric, sae_exact_a_direction_floor};
use super::outer_objective::sae_surrogate_lane_config;
use super::tests::{TestPeriodicEvaluator, periodic_basis};
use super::*;
use approx::assert_abs_diff_eq;
use ndarray::array;
use std::sync::Arc;

/// The shared #2515 witness state: one periodic atom on a unit-period circle at
/// `α = 250`, which puts `cos κt < 0` over a third of the rows so the periodic
/// ARD concave clamp is genuinely ACTIVE — that is what makes `ΔC = A − B`
/// nonzero and the two routes distinguishable at all.
pub(crate) struct ExactAWitness2515 {
    pub(crate) term: SaeManifoldTerm,
    pub(crate) target: Array2<f64>,
    pub(crate) rho: SaeManifoldRho,
}

pub(crate) fn exact_a_witness_2515() -> ExactAWitness2515 {
    exact_a_witness_2515_at_alpha(250.0)
}

/// The same state at a caller-chosen ARD precision. `α ≤ 10` keeps `A = B + ΔC`
/// and its reduced Schur positive definite, so BOTH evidence routes are admitted
/// and route parity is statable; the historical `α = 250` witness is far past
/// that boundary (see `zz_scan_exact_a_admitted_alpha_2515`).
pub(crate) fn exact_a_witness_2515_at_alpha(alpha: f64) -> ExactAWitness2515 {
    let n = 24usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let decoder = array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]];
    assert_eq!(decoder.ncols(), p);
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
        target[[row, 1]] += 1.0e-3 * (0.29 * row as f64).cos();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .expect("the #2515 witness atom is built from a well-formed periodic chart")
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("the #2515 witness assignment has one block and one manifold");
    let term = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("one atom against a one-block assignment is a valid term");
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![alpha.ln()]]);
    ExactAWitness2515 { term, target, rho }
}

/// SCAN — find an `α` where the exact observed information `A = B + ΔC` is still
/// per-row PD (so the streaming/bundle evidence route is ADMITTED) while the
/// periodic ARD concave clamp is genuinely ACTIVE (so `ΔC ≠ 0` and the two routes
/// are distinguishable at all). The named #2515 witness sits at `α = 250`, where
/// `A`'s worst per-row eigenvalue is `−1.93e2` and the streaming route refuses —
/// so route PARITY cannot be stated there, only the operator gap.
#[test]
fn zz_scan_exact_a_admitted_alpha_2515() {
    for log10_alpha in [-0.5_f64, 0.0, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.4] {
        let alpha = 10.0_f64.powf(log10_alpha);
        let ExactAWitness2515 {
            mut term, target, ..
        } = exact_a_witness_2515();
        let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![alpha.ln()]]);
        let sys = match term.assemble_arrow_schur(target.view(), &rho, None) {
            Ok(sys) => sys,
            Err(err) => {
                println!("[#2515 SCAN] alpha={alpha:.4e}: assembly refused: {err}");
                continue;
            }
        };
        let a_sys = match term.exact_a_evidence_system(target.view(), &rho, &sys) {
            Ok(sys) => sys,
            Err(err) => {
                println!("[#2515 SCAN] alpha={alpha:.4e}: exact-A assembly refused: {err}");
                continue;
            }
        };
        let mut worst_b = f64::INFINITY;
        let mut worst_a = f64::INFINITY;
        for (s, worst) in [(&sys, &mut worst_b), (&a_sys, &mut worst_a)] {
            for row in &s.rows {
                let (eigs, _) =
                    gam_linalg::faer_ndarray::FaerEigh::eigh(&row.htt, faer::Side::Lower).unwrap();
                *worst = worst.min(eigs.iter().cloned().fold(f64::INFINITY, f64::min));
            }
        }
        // How many rows carry a live `ΔC` on the ARD coordinate?
        let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
        let clamped = match solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options) {
            Ok((_, _, cache)) => term
                .exact_stationarity_penalty_derivative_delta_by_flat(&rho, &cache)
                .map(|d| {
                    d.get(&rho.ard_flat_index(0, 0))
                        .map(|m| m.diag().iter().filter(|v| **v != 0.0).count())
                        .unwrap_or(0)
                })
                .unwrap_or(0),
            Err(_) => usize::MAX,
        };
        let mut lane = SurrogateLaneState::new(sae_surrogate_lane_config());
        lane.request_logdet_derivative_bundle();
        let evidence_options = ArrowSolveOptions::direct()
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let streaming = gam_solve::arrow_schur::matrix_free_arrow_evidence_evaluation(
            &a_sys,
            0.0,
            0.0,
            &evidence_options,
            8,
            16,
            0xA51_5u64,
            &mut lane,
        );
        let dense_a = solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &options).is_ok();
        println!(
            "[#2515 SCAN] alpha={alpha:.4e}  min_eig(B)={worst_b:.4e}  min_eig(A)={worst_a:.4e}  \
             clamped_rows={clamped}  dense_A_factors={dense_a}  streaming_A={}",
            match &streaming {
                Ok(e) => format!("OK logdet={:.6e}", e.log_det()),
                Err(err) => format!("REFUSED {err:?}"),
            }
        );
    }
}

/// Full-basis probe set `√k·e_j` with exact `S⁻¹ e_j` off `cache`. At this probe
/// set the Hutchinson umbrella `(1/m)Σ_j (S⁻¹z_j)ᵀ M z_j` is EXACTLY
/// `tr(S⁻¹ M)`, so every from-probes channel is deterministic.
pub(crate) fn full_basis_probe_bundle(
    cache: &ArrowFactorCache,
) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| {
            cache
                .schur_inverse_apply(v.view())
                .expect("the #2515 probe directions are in the cached Schur complement's domain")
        })
        .collect();
    (probes, sinv)
}

/// #2515/#2712 — ROUTE PARITY ON A CACHE THAT ACTUALLY DEFLATES. This is the
/// measurement that lifted `penalized_quasi_laplace_streaming_outer_evaluation`'s
/// spectral-deflation refusal, and the one that holds the line now that it is
/// gone.
///
/// The refusal's own note named its lift condition: "#2515 first, then a measured
/// parity gate on a streaming-lane evaluation whose cache actually deflates.
/// Argument alone is what put the wrong reason on it in the first place." Then
/// `ac66e624d` measured this comparison on #2712's certified deflated anchor and
/// found the two routes 1.8 RELATIVE apart:
///
/// ```text
/// majorizer cache deflated rows = 10, exact-A cache deflated rows = 10
/// complete gradient max|Δ| = 9.131537e0 against ‖g‖∞ = 5.004339e0
/// ```
///
/// and named the reconciliation that would close it: the dense route floored the
/// spectrum of the materialized `A` against an ABSOLUTE band while the arrow route
/// conditioned each row block relative to its own largest eigenvalue — two floors,
/// two metrics, one `A`, the #2673 genus.
///
/// #2673 then did exactly that reconciliation (`00c1fe139`, `758c9d336`): the
/// absolute floor is deleted, and every direction is classified by its curvature in
/// the majorizer metric, `max(dim·ε·‖A‖₂, √ε·vᵀBv)`, at the value site and the
/// gradient site alike. Nobody re-measured THIS comparison against it. Re-measured:
///
/// ```text
/// majorizer cache deflated rows = 10, exact-A cache deflated rows = 10
/// complete gradient max|Δ| = 2.798722e-8 against ‖g‖∞ = 1.726754e1
///                          = 1.62e-9 RELATIVE
/// ```
///
/// so the same anchor that carried the refusal now carries the parity, and this
/// test asserts the parity. The residual is not machine precision (`1.57e-14` on a
/// non-deflating state, [`laplace_value_and_gradient_are_route_invariant_2515`])
/// and is not left unexplained: see
/// [`dense_exact_a_prices_a_b_deflated_direction_as_one_plus_delta_c_2515`], which
/// pins it to the one remaining ordering difference — whether `ΔC` is added before
/// or after a `B`-deflated direction is unit-pinned.
///
/// The non-vacuity arms below are load-bearing in the same way they were when the
/// assertion pointed the other way: this anchor must really deflate, and the
/// gradient must be non-trivial, or a parity claim about a deflating cache is a
/// claim about nothing.
#[test]
fn exact_a_route_parity_holds_on_a_deflated_cache_2515() {
    let (mut term, rho, target, b_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 exact-A parity on the #2712 deflated anchor",
        );
    let deflated_rows = b_cache
        .deflated_row_directions
        .iter()
        .filter(|d| !d.is_empty())
        .count();
    println!("[#2515 DEFLATED] majorizer cache deflated rows = {deflated_rows}");
    // Every step below is `expect`, not an early `return`. This gate asserts a
    // parity; a gate that quietly returns when one of the two routes declines is a
    // gate that reports green for the exact failure it exists to catch, which is
    // the "fails early, withdraws its coverage" trap this issue has already paid
    // for once.
    let loss = term
        .loss(target.view(), &rho)
        .expect("#2515: the certified deflated anchor evaluates its loss");
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("#2515: the certified deflated anchor assembles");
    let a_sys = term
        .exact_a_evidence_system(target.view(), &rho, &sys)
        .expect("#2515: the certified deflated anchor builds its exact-A evidence system");
    // The PD-evidence policy refuses an indefinite reduced Schur outright. The
    // PRODUCTION evidence policy is unit deflation at the canonical spectral
    // floor, which pseudo-inverts across the null/negative directions instead —
    // that is the policy the streaming lane runs and therefore the one a parity
    // statement about it has to use.
    let mut a_cache = None;
    for (label, options) in [
        (
            "positive-definite",
            ArrowSolveOptions::direct().with_positive_definite_evidence(),
        ),
        (
            "unit-deflation (production evidence policy)",
            ArrowSolveOptions::direct()
                .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
                .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR),
        ),
    ] {
        match solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &options) {
            Ok((_, _, cache)) => {
                println!("[#2515 DEFLATED] exact-A arrow factorization OK under {label}");
                a_cache = Some(cache);
                break;
            }
            Err(err) => println!(
                "[#2515 DEFLATED] exact-A arrow factorization refused under {label}: {err:?}"
            ),
        }
    }
    let a_cache = a_cache.expect(
        "#2515: neither evidence policy factored the exact-A system on the deflated \
         anchor, so there is no arrow route to state parity against",
    );
    println!(
        "[#2515 DEFLATED] exact-A cache deflated rows = {}  fingerprints differ = {}",
        a_cache
            .deflated_row_directions
            .iter()
            .filter(|d| !d.is_empty())
            .count(),
        a_cache.row_hessian_fingerprint != b_cache.row_hessian_fingerprint,
    );
    let b_solver = DeflatedArrowSolver::plain(&b_cache);
    let dense = term
        .analytic_outer_rho_gradient_components(target.view(), &rho, &loss, &b_cache, &b_solver)
        .expect("#2515: the dense exact-A gradient is the authority this parity is against")
        .gradient();
    let (a_probes, a_sinv) = full_basis_probe_bundle(&a_cache);
    let bundled = term
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &b_cache,
            &b_solver,
            Some(BundleEvidenceGeometry {
                operator: EvidenceOperator::ExactObservedInformation,
                cache: &a_cache,
                probes: &a_probes,
                sinv: &a_sinv,
            }),
            None,
        )
        .expect(
            "#2515: the bundle route must PRODUCE a gradient on the exact-A geometry of a \
         deflating cache -- that it does is what lifted the streaming refusal",
        )
        .gradient();
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..dense.len() {
        worst = worst.max((dense[i] - bundled[i]).abs());
        scale = scale.max(dense[i].abs());
        println!(
            "[#2515 DEFLATED] coord {i}: dense={:+.10e} bundle={:+.10e} |Δ|={:.3e}",
            dense[i],
            bundled[i],
            (dense[i] - bundled[i]).abs()
        );
    }
    println!(
        "[#2515 DEFLATED] complete gradient max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e} \
         (relative {:.6e})",
        worst / scale.max(f64::MIN_POSITIVE)
    );
    assert!(
        deflated_rows > 0,
        "#2515/#2712: this anchor must actually deflate, or the measurement is about \
         nothing; certified regime promised SomeRowDeflates"
    );
    assert!(
        scale.is_finite() && scale > 1.0e-6,
        "#2515/#2712: the gradient must be non-trivial for a relative gap to mean \
         anything; ‖g‖∞={scale:.6e}"
    );
    // The bar is `1e-6` RELATIVE, and it is the same bar
    // `production_objective_forced_streaming_value_gradient_matches_dense` holds the
    // forced-streaming production gradient to — not a number fitted to what this
    // fixture happens to produce. The measured residual is `1.62e-9` relative
    // (`2.798722e-8` against `‖g‖∞ = 1.726754e1`), three decades inside it, and its
    // cause is pinned separately by
    // `dense_exact_a_prices_a_b_deflated_direction_as_one_plus_delta_c_2515` rather
    // than absorbed here.
    //
    // The history this replaces: `ac66e624d` measured `9.131537e0` against
    // `‖g‖∞ = 5.004339e0` — 1.8 RELATIVE — and asserted the gap was still large so
    // that the production refusal it justified could not decay into folklore. #2673
    // then deleted the absolute floor and put both classifications in the majorizer
    // metric (`00c1fe139`, `758c9d336`), which is precisely the reconciliation that
    // refusal named as its lift condition, so this gate is now the parity statement
    // and the refusal is gone.
    assert!(
        worst <= 1.0e-6 * scale,
        "#2515/#2712: the exact-A bundle route and the dense exact-A route disagree on \
         a deflating cache (max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e}, relative \
         {:.6e}). They agreed to 1.62e-9 relative when this was written, on this \
         anchor, after #2673 put both classifications in the majorizer metric. A \
         regression here means the streaming lane is steering a deflating fit with \
         the derivative of an operator the dense criterion does not rank — the \
         defect `penalized_quasi_laplace_streaming_outer_evaluation` refused for, \
         before that refusal was lifted on this measurement.",
        worst / scale.max(f64::MIN_POSITIVE)
    );
}

/// MEASUREMENT — attribute the deflating-anchor route gap to the
/// CLASSIFICATION, not to a channel.
///
/// `exact_a_route_parity_still_fails_on_a_deflated_cache_2515` establishes that
/// the two routes disagree by `9.13` against `‖g‖∞ = 5.00` once a cache
/// deflates, and the production streaming gate refuses on that number. It does
/// not say WHICH directions the two routes classify differently, and the repair
/// is a different one for each answer. This probe prints both classifications
/// direction by direction.
///
/// The two rules under comparison, both on the SAME `A`:
///
/// * DENSE (`ExactHessianSpectralBlock::rank_floor`, #2673) — the null band of
///   an eigendirection `v` is `max(dim·ε·‖A‖₂, √ε·vᵀBv)`, i.e. the pencil
///   curvature against the majorizer metric the gradient path also uses; a
///   negative direction is priced at its ARD-clamp basin `λ+vᵀEv` (#2336) and
///   only a basin below `−floor` refuses.
/// * ARROW (`factor_spectral_deflated_criterion_row`,
///   `factor_evidence_unit_deflated_schur`) — the band is
///   `SPECTRAL_DEFLATION_REL_FLOOR·max|λ|` of the block ALONE, which sees
///   neither `B` nor the rest of the operator, and EVERY non-positive
///   eigenvalue is unit-pinned regardless of what the clamp explains.
///
/// So there are two independent disagreements — the BAND and the SIGN — and
/// which of them carries the `9.13` decides whether the repair is a metric or a
/// pricing rule.
///
/// HISTORICAL, AND DELIBERATELY SO. This probe factors the exact-`A` system under
/// `with_evidence_unit_deflation`, which is the MAJORIZER's policy and no longer
/// what production uses on this operator (see
/// `with_indefinite_refusing_evidence_unit_deflation`). That is the point: it is
/// the negative control for the repair, and re-pointing it at the production
/// policy would delete the only in-tree record of what the one-sided band did.
#[test]
fn zz_attribute_deflated_route_classification_2515() {
    let (mut term, rho, target, b_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 deflation-pricing attribution",
        );
    let total_t = b_cache.delta_t_len();
    let k = b_cache.k;
    println!(
        "[#2515 CLASSIFY] total_t={total_t} k={k} rows={}",
        b_cache.n_rows()
    );

    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &b_cache)
        .expect("the anchor's exact Hessian materializes");
    let e_diag = term
        .materialize_ard_concave_clamp_diagonal(&rho, &b_cache)
        .expect("the anchor's ARD concave clamp diagonal is available");

    for (label, block, metric) in [
        ("joint", a.clone(), ArrowMetric::Joint(&b_cache)),
        (
            "coordinate",
            a.slice(s![..total_t, ..total_t]).to_owned(),
            ArrowMetric::Coordinate(&b_cache),
        ),
    ] {
        let (evals, evecs) = gam_linalg::faer_ndarray::FaerEigh::eigh(&block, faer::Side::Lower)
            .expect("a symmetric block diagonalizes");
        let spectral_norm = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let mut pinned = 0usize;
        let mut priced_negative = 0usize;
        let mut log_det = 0.0_f64;
        for idx in 0..evals.len() {
            let lambda = evals[idx];
            let v = evecs.column(idx);
            let bvv = metric
                .quadratic_form(v)
                .expect("the majorizer metric is defined on every direction");
            let floor = sae_exact_a_direction_floor(evals.len(), spectral_norm, bvv);
            let e_v: f64 = (0..total_t.min(v.len()))
                .map(|j| e_diag[j] * v[j] * v[j])
                .sum();
            let priced = if lambda < -floor {
                priced_negative += 1;
                lambda + e_v
            } else {
                lambda
            };
            if priced > floor {
                log_det += priced.ln();
            } else {
                pinned += 1;
                println!(
                    "[#2515 CLASSIFY] dense {label} dir {idx}: lambda={lambda:+.6e} \
                     floor={floor:.6e} v'Bv={bvv:.6e} v'Ev={e_v:.6e} priced={priced:+.6e} PINNED"
                );
            }
        }
        println!(
            "[#2515 CLASSIFY] dense {label}: dim={} ||A||2={spectral_norm:.6e} pinned={pinned} \
             clamp-priced-negative={priced_negative} log_det={log_det:.10e}",
            evals.len()
        );
    }

    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("the anchor assembles");
    let a_sys = term
        .exact_a_evidence_system(target.view(), &rho, &sys)
        .expect("the anchor's exact-A evidence system builds");
    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
    let (_, _, a_cache) = solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &options)
        .expect("the production evidence policy factors the anchor's exact-A system");

    let mut arrow_row_log_det = 0.0_f64;
    let mut arrow_pinned = 0usize;
    for row in 0..a_cache.n_rows() {
        let Some(spectrum) = a_cache
            .deflation_row_spectra
            .get(row)
            .and_then(Option::as_ref)
        else {
            continue;
        };
        let a_row = &a_sys.rows[row].htt;
        let b_row = &sys.rows[row].htt;
        let norm = spectrum
            .raw_evals
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        for idx in 0..spectrum.raw_evals.len() {
            let lambda = spectrum.raw_evals[idx];
            let v = spectrum.evecs.column(idx);
            let bvv = v.dot(&b_row.dot(&v));
            let avv = v.dot(&a_row.dot(&v));
            let pencil_floor = sae_exact_a_direction_floor(spectrum.raw_evals.len(), norm, bvv);
            let base = a_cache.row_offsets[row];
            let e_v: f64 = (0..v.len()).map(|j| e_diag[base + j] * v[j] * v[j]).sum();
            let arrow_floor = gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR * norm;
            let conditioned = spectrum.cond_evals[idx];
            if conditioned > 0.0 {
                arrow_row_log_det += conditioned.ln();
            }
            let deflated = matches!(
                spectrum.conditioning[idx],
                gam_solve::arrow_schur::RowSpectralConditioning::UnitDeflated
            );
            if deflated {
                arrow_pinned += 1;
            }
            println!(
                "[#2515 CLASSIFY] arrow row {row} dir {idx}: lambda={lambda:+.6e} \
                 cond={conditioned:+.6e} arrow_floor={arrow_floor:.6e} \
                 pencil_floor={pencil_floor:.6e} v'Bv={bvv:.6e} v'Av={avv:+.6e} \
                 v'Ev={e_v:.6e} basin={:+.6e} deflated={deflated} \
                 pencil_would_pin={}",
                lambda + e_v,
                (if lambda < -pencil_floor {
                    lambda + e_v
                } else {
                    lambda
                }) <= pencil_floor
            );
        }
    }
    println!(
        "[#2515 CLASSIFY] arrow rows: pinned={arrow_pinned} \
         sum_log_cond_row={arrow_row_log_det:.10e}"
    );
    match a_cache.beta_schur_deflation.as_ref() {
        Some(spectrum) => {
            for idx in 0..spectrum.raw_evals.len() {
                println!(
                    "[#2515 CLASSIFY] arrow schur dir {idx}: lambda={:+.6e} cond={:+.6e} \
                     deflated={}",
                    spectrum.raw_evals[idx], spectrum.cond_evals[idx], spectrum.deflated[idx]
                );
            }
        }
        None => println!("[#2515 CLASSIFY] arrow schur: no deflation recorded"),
    }
}

/// #2515 — THE ONE ORDERING DIFFERENCE LEFT BETWEEN THE TWO EXACT-`A` OPERATORS,
/// and it is not a floor, a metric or a channel: it is whether `ΔC` is added
/// before or after a `B`-deflated direction is unit-pinned.
///
/// After #2673 put both classifications in the majorizer metric, the complete
/// gradients agree to `1.62e-9` relative on #2712's certified deflated anchor
/// ([`exact_a_route_parity_holds_on_a_deflated_cache_2515`]) — three decades
/// inside the production bar, but six decades short of the `1.57e-14` the same
/// comparison reaches on a NON-deflating state. That gap has one cause, and this
/// gate is what stops it from being folklore:
///
/// * the DENSE route materializes `A` through `apply_cached_arrow_hessian`, which
///   applies `L Lᵀ` of the row's UNDAMPED FACTOR — the majorizer already
///   unit-pinned by its own factorization. So the dense exact-`A` row block is
///   `B̃ + ΔC`, and a direction `B` declared null enters it as `1 + vᵀΔCv`;
/// * the ARROW route assembles `B_raw + ΔC` (`exact_a_evidence_system` folds `ΔC`
///   into the untouched majorizer blocks) and unit-pins the RESULT, so the same
///   direction is exactly `1`.
///
/// Both honour "a unit-deflated direction contributes `log 1 = 0` with zero ρ/θ
/// dependence". Only one of them honours it EXACTLY: `vᵀΔCv` is a function of ρ,
/// so the dense route prices a ρ-dependent `log(1 + vᵀΔCv)` on a direction whose
/// whole point is to be ρ-independent. It is `~1e-8` here, which is why the two
/// routes agree to nine digits rather than to fifteen.
///
/// The assertion is an IDENTITY, not a tolerance on a difference of two large
/// numbers: the two exact-`A` row blocks differ by the majorizer's own
/// conditioning increment `B̃ − B_raw` and by NOTHING ELSE. If some other
/// disagreement ever appears between the two assemblers, this goes red on it
/// specifically, rather than being absorbed into the parity bar next door.
#[test]
fn dense_exact_a_prices_a_b_deflated_direction_as_one_plus_delta_c_2515() {
    let (mut term, rho, target, b_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 the B-conditioning increment is the whole residual",
        );
    let a_dense = term
        .materialize_exact_hessian_dense(&rho, target.view(), &b_cache)
        .expect("#2515: the deflated anchor's exact Hessian materializes");
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("#2515: the deflated anchor assembles");
    let a_sys = term
        .exact_a_evidence_system(target.view(), &rho, &sys)
        .expect("#2515: the deflated anchor builds its exact-A evidence system");

    let mut worst_identity = 0.0_f64;
    let mut worst_block_scale = 0.0_f64;
    let mut deflated_directions = 0usize;
    let mut worst_pinned_delta_c = 0.0_f64;
    for row in 0..b_cache.n_rows() {
        let q = b_cache.row_dims[row];
        let base = b_cache.row_offsets[row];
        let factor = b_cache.undamped_factor(row);
        // `B̃` — what the dense route's operator apply actually applies.
        let b_conditioned = factor.dot(&factor.t());
        // `B_raw` — what the arrow route's evidence system was folded into.
        let b_raw = &sys.rows[row].htt;
        let a_dense_block = a_dense.slice(s![base..base + q, base..base + q]).to_owned();
        let a_arrow_block = &a_sys.rows[row].htt;
        for a in 0..q {
            for b in 0..q {
                let residual = (a_dense_block[[a, b]] - a_arrow_block[[a, b]])
                    - (b_conditioned[[a, b]] - b_raw[[a, b]]);
                worst_identity = worst_identity.max(residual.abs());
                worst_block_scale = worst_block_scale.max(a_dense_block[[a, b]].abs());
            }
        }
        // On each direction the majorizer factorization declared null, report what
        // each route's exact-`A` actually prices there.
        for direction in b_cache
            .deflated_row_directions
            .get(row)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            if direction.len() != q {
                continue;
            }
            deflated_directions += 1;
            let dense_curvature = direction.dot(&a_dense_block.dot(direction));
            let arrow_curvature = direction.dot(&a_arrow_block.dot(direction));
            let b_raw_curvature = direction.dot(&b_raw.dot(direction));
            // `vᵀΔCv` on this direction, from the arrow side where `B` is untouched.
            let delta_c = arrow_curvature - b_raw_curvature;
            worst_pinned_delta_c = worst_pinned_delta_c.max(delta_c.abs());
            println!(
                "[#2515 ORDERING] row {row}: v'B_raw v={b_raw_curvature:.6e} \
                 v'(B_raw+ΔC)v={arrow_curvature:.6e} v'ΔCv={delta_c:+.6e} \
                 dense v'(B̃+ΔC)v={dense_curvature:.10e} (arrow pins this direction to 1)"
            );
        }
    }
    println!(
        "[#2515 ORDERING] |(A_dense − A_arrow) − (B̃ − B_raw)|∞ = {worst_identity:.6e} \
         over block scale {worst_block_scale:.6e}; deflated directions inspected = \
         {deflated_directions}; worst |v'ΔCv| on a pinned direction = \
         {worst_pinned_delta_c:.6e}"
    );

    assert!(
        deflated_directions > 0,
        "#2515: this anchor must carry at least one majorizer-deflated direction, or \
         the ordering claim is about nothing"
    );
    assert!(
        worst_block_scale.is_finite() && worst_block_scale > 1.0e-6,
        "#2515: the exact-A row blocks must be non-trivial for the identity to mean \
         anything; block scale {worst_block_scale:.6e}"
    );
    assert!(
        worst_identity <= 1.0e-12 * worst_block_scale,
        "#2515: the dense and arrow exact-A row blocks differ by something OTHER than \
         the majorizer's own conditioning increment (|(A_dense − A_arrow) − (B̃ − \
         B_raw)|∞ = {worst_identity:.6e} over block scale {worst_block_scale:.6e}). \
         The residual in `exact_a_route_parity_holds_on_a_deflated_cache_2515` is \
         attributed to that increment and to nothing else; if this is red, that \
         attribution is wrong and the parity bar next door is absorbing a second \
         cause."
    );
}

/// #2515 — THE LIFTED GATE, END TO END: a state whose evidence factorization
/// spectrally deflates now gets a streaming outer gradient instead of a typed
/// refusal, and that gradient is the dense one.
///
/// The two tests either side of this measure the ASSEMBLERS at a fixed state.
/// This one drives the production entry point — `evaluate_outer_criterion_route`
/// with `direct_logdet_admitted = false`, the branch the memory planner selects
/// at production `p` — so a regression that re-armed the refusal, or that
/// admitted it while silently returning a `B`-rooted gradient, is caught where a
/// fit would actually meet it.
///
/// Before the lift this returned
/// `"streaming outer derivative is not admitted: the … evidence factorization
/// spectrally deflates row R in N direction(s)"`, so on a deflating state the
/// streaming lane had no answer at all — and at production `p` the streaming lane
/// is the only lane there is. That is the residual route-dependence this issue
/// was left with once the operator halves were closed: not a wrong criterion on
/// one route, but a criterion on one route and nothing on the other.
#[test]
fn forced_streaming_admits_a_deflating_state_and_matches_dense_2515() {
    let (term, rho, target, b_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 the lifted deflation gate, end to end",
        );
    let anchor_deflated_rows = b_cache
        .deflated_row_directions
        .iter()
        .filter(|directions| !directions.is_empty())
        .count();
    assert!(
        anchor_deflated_rows > 0,
        "#2515: the anchor must deflate, or this exercises the ordinary lane"
    );

    let mut dense = SaeManifoldOuterObjective::new(
        term.clone(),
        target.clone(),
        None,
        rho.clone(),
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    let mut streaming =
        SaeManifoldOuterObjective::new(term, target, None, rho.clone(), 40, 0.4, 1.0e-6, 1.0e-6);
    let rho_flat = dense.baseline_rho.to_flat();
    let route_rho = streaming
        .baseline_rho
        .from_flat(rho_flat.view())
        .expect("#2515: both objectives own the same typed rho layout");

    let dense_artifact = dense
        .evaluate_outer_criterion_route(&route_rho, true, false)
        .expect("#2515: the dense route is the authority this parity is against");
    let dense_gradient = dense
        .analytic_gradient_for_outer_evaluation(&route_rho, &dense_artifact)
        .expect("#2515: the dense route's analytic gradient");

    let streaming_artifact = streaming
        .evaluate_outer_criterion_route(&route_rho, false, false)
        .expect(
            "#2515: the forced streaming route must ADMIT a deflating state. A typed \
             `streaming outer derivative is not admitted: … spectrally deflates row …` \
             here means the lifted refusal has been re-armed",
        );
    let streaming_gradient = streaming
        .analytic_gradient_for_outer_evaluation(&route_rho, &streaming_artifact)
        .expect("#2515: the forced streaming route's analytic gradient");

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (coordinate, (&streamed, &direct)) in streaming_gradient
        .iter()
        .zip(dense_gradient.iter())
        .enumerate()
    {
        assert!(
            streamed.is_finite() && direct.is_finite(),
            "#2515: gradient coordinate {coordinate} is non-finite \
             (streaming={streamed}, dense={direct})"
        );
        worst = worst.max((streamed - direct).abs());
        scale = scale.max(direct.abs());
    }
    println!(
        "[#2515 LIFTED] anchor deflated rows={anchor_deflated_rows} \
         cost dense={:.10e} streaming={:.10e} \
         gradient max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e}",
        dense_artifact.cost, streaming_artifact.cost
    );

    assert_eq!(
        streaming_gradient.len(),
        dense_gradient.len(),
        "#2515: the two routes must own the same outer coordinate layout"
    );
    assert!(
        scale.is_finite() && scale > 1.0e-9,
        "#2515: route parity must exercise a nonzero analytic gradient; ‖g‖∞={scale:.6e}"
    );
    assert_abs_diff_eq!(
        streaming_artifact.cost,
        dense_artifact.cost,
        epsilon = 1.0e-7
    );
    assert!(
        worst <= 1.0e-6 * scale.max(1.0),
        "#2515: the forced streaming gradient departs from the dense one \
         (max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e}). The streaming lane is steering \
         a fit with the derivative of an operator the dense criterion does not rank — \
         the defect the spectral-deflation refusal used to hide behind."
    );
}

/// #2515 — ROUTE PARITY ON A LADDER OF DEFLATING STATES, not on one anchor.
///
/// [`exact_a_route_parity_holds_on_a_deflated_cache_2515`] states the parity at
/// the single ρ #2712 certified. That is the state the lifted refusal's own
/// justification was measured on, so it has to be there — but one state is thin
/// evidence for removing a production refusal, and a parity that happens to hold
/// at one ρ is exactly the kind of number this issue has been burned by before.
///
/// This walks a ρ ladder around that anchor at the SAME inner state (θ̂, β̂ frozen
/// at the certified fixed point), which is what a route-parity statement needs:
/// both assemblers reading one state, differing only in which geometry they
/// contract. Every rung whose majorizer factorization actually deflates is
/// compared; rungs where nothing deflates are reported and skipped, because they
/// are already covered at `1.57e-14` by
/// [`laplace_value_and_gradient_are_route_invariant_2515`].
#[test]
fn exact_a_route_parity_holds_across_a_deflating_rho_ladder_2515() {
    let (mut term, anchor_rho, target, _anchor_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 route parity across a deflating rho ladder",
        );
    // The PRODUCTION exact-`A` evidence policy since #2515: the null band is
    // unit-pinned and a RESOLVED negative direction is refused rather than pinned.
    // Factoring the ladder under `with_evidence_unit_deflation` instead would test
    // a policy production no longer uses on this operator — and would restore
    // exactly the silent saddle-as-null pricing this gate exists to keep out.
    let evidence_options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(
            gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR,
        );
    let majorizer_options = ArrowSolveOptions::direct().with_positive_definite_evidence();

    let mut deflating_states = 0usize;
    let mut indefinite_states = 0usize;
    let mut worst_relative = 0.0_f64;
    let mut worst_label = String::new();
    // Rungs are excursions AROUND the certified anchor `(-0.5, -1.0, -0.5)`, not a
    // sweep across the whole box: far from it the dense route refuses outright
    // (`IndefiniteObservedInformation` at an off-optimum inner state), and a rung
    // where only one route has an answer states nothing about parity. Those rungs
    // are reported and skipped, and the `deflating_states` floor below is what stops
    // the whole ladder from silently degenerating into them.
    for (sparse, smooth, ard) in [
        (-0.5_f64, -1.0_f64, -0.5_f64),
        (-0.5, -1.0, -0.2),
        (-0.5, -1.0, -0.8),
        (-0.5, -0.9, -0.5),
        (-0.5, -1.1, -0.5),
        (-0.35, -1.0, -0.5),
        (-0.65, -1.0, -0.5),
        (-0.65, -0.9, -0.35),
        (-0.8, -0.8, -0.3),
    ] {
        let mut rho = anchor_rho.clone();
        rho.log_lambda_sparse = sparse;
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = smooth;
        }
        for axis in rho.log_ard.iter_mut() {
            for value in axis.iter_mut() {
                *value = ard;
            }
        }
        let label = format!("sparse={sparse:.1} smooth={smooth:.1} ard={ard:.1}");

        let Ok(sys) = term.assemble_arrow_schur(target.view(), &rho, None) else {
            println!("[#2515 LADDER] {label}: majorizer assembly refused");
            continue;
        };
        let Ok((_, _, b_cache)) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &majorizer_options)
        else {
            println!("[#2515 LADDER] {label}: majorizer factorization refused");
            continue;
        };
        let deflated_rows = b_cache
            .deflated_row_directions
            .iter()
            .filter(|directions| !directions.is_empty())
            .count();
        if deflated_rows == 0 {
            println!("[#2515 LADDER] {label}: no row deflates — covered by the undeflated gate");
            continue;
        }
        let Ok(a_sys) = term.exact_a_evidence_system(target.view(), &rho, &sys) else {
            println!("[#2515 LADDER] {label}: exact-A evidence system refused");
            continue;
        };
        let a_cache = match solve_arrow_newton_step_with_options(
            &a_sys,
            0.0,
            0.0,
            &evidence_options,
        ) {
            Ok((_, _, cache)) => cache,
            Err(err) => {
                let rendered = err.to_string();
                // A RESOLVED-INDEFINITE refusal is not a skip: it is the arrow
                // route reaching the dense route's own verdict, and the assertion
                // below requires the dense route to agree that this state's exact
                // `A` really is indefinite. Any OTHER refusal is a skip.
                if gam_solve::arrow_schur::ArrowSchurError::rendered_is_indefinite_evidence(
                    &rendered,
                ) {
                    let min_eig = dense_exact_a_min_eigenvalue_2515(&mut term, &rho, &target, &b_cache);
                    println!(
                        "[#2515 LADDER] {label}: exact-A arrow factorization REFUSED as \
                         indefinite; dense min eig(A) = {min_eig:+.6e}"
                    );
                    assert!(
                        min_eig < 0.0,
                        "#2515: the arrow route refused [{label}] as indefinite while the \
                         DENSE spectrum of the same exact A has min eigenvalue \
                         {min_eig:+.6e} >= 0. The two routes must reach the same \
                         mode-or-saddle verdict; a refusal the dense route does not \
                         corroborate is an over-refusal, and it costs the streaming lane \
                         states it can legitimately rank."
                    );
                    indefinite_states += 1;
                } else {
                    println!("[#2515 LADDER] {label}: exact-A arrow factorization refused: {rendered}");
                }
                continue;
            }
        };
        let Ok(loss) = term.loss(target.view(), &rho) else {
            println!("[#2515 LADDER] {label}: loss unavailable");
            continue;
        };
        let solver = DeflatedArrowSolver::plain(&b_cache);
        let Ok(dense) = term.analytic_outer_rho_gradient_components(
            target.view(),
            &rho,
            &loss,
            &b_cache,
            &solver,
        ) else {
            println!("[#2515 LADDER] {label}: dense exact-A gradient refused");
            continue;
        };
        let (probes, sinv) = full_basis_probe_bundle(&a_cache);
        let bundled = term
            .analytic_outer_rho_gradient_components_with_bundle(
                target.view(),
                &rho,
                &loss,
                &b_cache,
                &solver,
                Some(BundleEvidenceGeometry {
                    operator: EvidenceOperator::ExactObservedInformation,
                    cache: &a_cache,
                    probes: &probes,
                    sinv: &sinv,
                }),
                None,
            )
            .expect(
                "#2515: once the dense route has produced a gradient at this state, the \
                 bundle route on the exact-A geometry must produce one too",
            );
        let dense = dense.gradient();
        let bundled = bundled.gradient();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for coordinate in 0..dense.len() {
            worst = worst.max((dense[coordinate] - bundled[coordinate]).abs());
            scale = scale.max(dense[coordinate].abs());
        }
        let relative = worst / scale.max(f64::MIN_POSITIVE);
        println!(
            "[#2515 LADDER] {label}: deflated rows={deflated_rows} max|Δ|={worst:.6e} \
             ‖g‖∞={scale:.6e} relative={relative:.6e}"
        );
        deflating_states += 1;
        if relative > worst_relative {
            worst_relative = relative;
            worst_label = label;
        }
    }

    println!(
        "[#2515 LADDER] deflating states compared = {deflating_states}, indefinite states \
         refused = {indefinite_states}, worst relative gap = {worst_relative:.6e} at \
         [{worst_label}]"
    );
    assert!(
        deflating_states >= 6,
        "#2515: this ladder must reach at least six genuinely deflating states, or it \
         is a parity claim about the undeflated regime the gate next door already \
         covers; got {deflating_states}"
    );
    assert!(
        worst_relative <= 1.0e-6,
        "#2515: the two routes disagree on a deflating state (worst relative gap \
         {worst_relative:.6e} at [{worst_label}]). One anchor is not the contract — the \
         streaming lane's spectral-deflation refusal was lifted on the claim that the \
         classification is shared across the deflating regime, not at one ρ."
    );
    assert!(
        indefinite_states > 0,
        "#2515: this ladder must also reach at least one state whose exact A is genuinely \
         INDEFINITE, or it says nothing about the half of the classification that used to \
         disagree by 1.009 relative — the arrow route pinning a resolved negative direction \
         to +1 while the dense route priced it as #2336 clamp-attributable curvature. That \
         is the regime the refusal now covers, and an assertion that never enters it is an \
         assertion about the easy half."
    );
}

/// The smallest eigenvalue of the DENSE exact observed information at one state —
/// the dense route's own mode-or-saddle verdict, in one number, for the ladder to
/// corroborate an arrow-route refusal against.
fn dense_exact_a_min_eigenvalue_2515(
    term: &mut SaeManifoldTerm,
    rho: &SaeManifoldRho,
    target: &Array2<f64>,
    cache: &ArrowFactorCache,
) -> f64 {
    let Ok(a_dense) = term.materialize_exact_hessian_dense(rho, target.view(), cache) else {
        return f64::NAN;
    };
    let Ok((evals, _)) = gam_linalg::faer_ndarray::FaerEigh::eigh(&a_dense, faer::Side::Lower)
    else {
        return f64::NAN;
    };
    evals.iter().copied().fold(f64::INFINITY, f64::min)
}

/// MEASUREMENT — the ρ rung where the ladder breaks, taken apart.
///
/// [`exact_a_route_parity_holds_across_a_deflating_rho_ladder_2515`] widened from
/// one anchor to nine rungs and immediately found one where the two routes do NOT
/// agree — `log λ_smooth = −1.1`, one tenth of a decade from the certified anchor:
///
/// ```text
/// smooth=−1.0  max|Δ|=3.267663e-8  ‖g‖∞=1.726754e1  relative=1.89e-9
/// smooth=−1.1  max|Δ|=4.411584e2   ‖g‖∞=4.371555e2  relative=1.01e0
/// smooth=−0.9  max|Δ|=2.909077e-8  ‖g‖∞=7.659687e-1 relative=3.80e-8
/// ```
///
/// `‖g‖∞` itself moves `0.77 → 17.3 → 437` across those three rungs, so the state
/// is doing something violent of its own accord. This probe prints, for each of
/// the three, every classification either route makes — per-row deflation counts
/// on both caches, β-Schur deflation on the exact-`A` cache, the dense joint and
/// coordinate pinned counts and clamp-priced-negative counts — plus the
/// per-coordinate gradient split, so the `1.01` is attributed to one of them
/// rather than guessed at.
///
/// HISTORICAL, AND DELIBERATELY SO. Like its sibling above, this factors the
/// exact-`A` system under the MAJORIZER's `with_evidence_unit_deflation`, which
/// production no longer uses on this operator. That is what lets it still PRINT
/// the mechanism it found — `S_A dir 0: raw=-7.997610e-3 cond=+1.000000e0`, a
/// resolved negative direction priced as the ρ-independent null — where the
/// production policy now refuses instead. The parity gates assert the repair; this
/// keeps the disease under glass.
#[test]
fn zz_attribute_the_broken_ladder_rung_2515() {
    let (mut term, anchor_rho, target, _anchor_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 the broken ladder rung",
        );
    let evidence_options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
    let majorizer_options = ArrowSolveOptions::direct().with_positive_definite_evidence();

    for smooth in [-0.9_f64, -1.0, -1.05, -1.1, -1.2] {
        let mut rho = anchor_rho.clone();
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = smooth;
        }
        let Ok(sys) = term.assemble_arrow_schur(target.view(), &rho, None) else {
            println!("[#2515 RUNG] smooth={smooth:.2}: majorizer assembly refused");
            continue;
        };
        let Ok((_, _, b_cache)) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &majorizer_options)
        else {
            println!("[#2515 RUNG] smooth={smooth:.2}: majorizer factorization refused");
            continue;
        };
        let Ok(a_sys) = term.exact_a_evidence_system(target.view(), &rho, &sys) else {
            println!("[#2515 RUNG] smooth={smooth:.2}: exact-A evidence system refused");
            continue;
        };
        let Ok((_, _, a_cache)) =
            solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &evidence_options)
        else {
            println!("[#2515 RUNG] smooth={smooth:.2}: exact-A arrow factorization refused");
            continue;
        };
        let b_deflated = b_cache
            .deflated_row_directions
            .iter()
            .filter(|d| !d.is_empty())
            .count();
        let a_deflated = a_cache
            .deflated_row_directions
            .iter()
            .filter(|d| !d.is_empty())
            .count();
        let a_beta_deflated = a_cache
            .beta_schur_deflation
            .as_ref()
            .map(|spectrum| spectrum.deflated.iter().filter(|d| **d).count());
        let b_beta_deflated = b_cache
            .beta_schur_deflation
            .as_ref()
            .map(|spectrum| spectrum.deflated.iter().filter(|d| **d).count());

        // The dense classification on the same state.
        let a_dense = term
            .materialize_exact_hessian_dense(&rho, target.view(), &b_cache)
            .expect("#2515: the rung's exact Hessian materializes");
        let e_diag = term
            .materialize_ard_concave_clamp_diagonal(&rho, &b_cache)
            .expect("#2515: the rung's clamp diagonal");
        let total_t = b_cache.delta_t_len();
        let mut dense_report = String::new();
        for (label, block, metric) in [
            ("joint", a_dense.clone(), ArrowMetric::Joint(&b_cache)),
            (
                "coord",
                a_dense.slice(s![..total_t, ..total_t]).to_owned(),
                ArrowMetric::Coordinate(&b_cache),
            ),
        ] {
            let (evals, evecs) =
                gam_linalg::faer_ndarray::FaerEigh::eigh(&block, faer::Side::Lower)
                    .expect("a symmetric block diagonalizes");
            let norm = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            let mut pinned = 0usize;
            let mut priced_negative = 0usize;
            let mut min_eig = f64::INFINITY;
            for idx in 0..evals.len() {
                let lambda = evals[idx];
                min_eig = min_eig.min(lambda);
                let v = evecs.column(idx);
                let bvv = metric.quadratic_form(v).unwrap_or(0.0);
                let floor = sae_exact_a_direction_floor(evals.len(), norm, bvv);
                let e_v: f64 = (0..total_t.min(v.len()))
                    .map(|j| e_diag[j] * v[j] * v[j])
                    .sum();
                let priced = if lambda < -floor {
                    priced_negative += 1;
                    lambda + e_v
                } else {
                    lambda
                };
                if !(priced > floor) {
                    pinned += 1;
                }
            }
            dense_report.push_str(&format!(
                " dense-{label}(min_eig={min_eig:+.3e} pinned={pinned} negpriced={priced_negative})"
            ));
        }

        let Ok(loss) = term.loss(target.view(), &rho) else {
            println!("[#2515 RUNG] smooth={smooth:.2}: loss unavailable");
            continue;
        };
        let solver = DeflatedArrowSolver::plain(&b_cache);
        let dense = match term.analytic_outer_rho_gradient_components(
            target.view(),
            &rho,
            &loss,
            &b_cache,
            &solver,
        ) {
            Ok(components) => components,
            Err(err) => {
                println!("[#2515 RUNG] smooth={smooth:.2}: dense gradient refused: {err}");
                continue;
            }
        };
        let (probes, sinv) = full_basis_probe_bundle(&a_cache);
        let bundled = match term.analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &b_cache,
            &solver,
            Some(BundleEvidenceGeometry {
                operator: EvidenceOperator::ExactObservedInformation,
                cache: &a_cache,
                probes: &probes,
                sinv: &sinv,
            }),
            None,
        ) {
            Ok(components) => components,
            Err(err) => {
                println!("[#2515 RUNG] smooth={smooth:.2}: bundle gradient refused: {err}");
                continue;
            }
        };
        println!(
            "[#2515 RUNG] smooth={smooth:.2}: b_rows_deflated={b_deflated} \
             a_rows_deflated={a_deflated} b_beta_deflated={b_beta_deflated:?} \
             a_beta_deflated={a_beta_deflated:?}{dense_report}"
        );
        if let Some(spectrum) = a_cache.beta_schur_deflation.as_ref() {
            let norm = spectrum
                .raw_evals
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            for idx in 0..spectrum.raw_evals.len() {
                if spectrum.deflated[idx] {
                    println!(
                        "[#2515 RUNG] smooth={smooth:.2} S_A dir {idx}: raw={:+.6e} \
                         cond={:+.6e} relative={:+.6e} (floor {:.6e})",
                        spectrum.raw_evals[idx],
                        spectrum.cond_evals[idx],
                        spectrum.raw_evals[idx] / norm.max(f64::MIN_POSITIVE),
                        gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR
                    );
                }
            }
        }
        for row in 0..a_cache.n_rows() {
            if let Some(spectrum) = a_cache.deflation_row_spectra.get(row).and_then(Option::as_ref)
            {
                let norm = spectrum
                    .raw_evals
                    .iter()
                    .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
                for idx in 0..spectrum.raw_evals.len() {
                    let raw = spectrum.raw_evals[idx];
                    if raw < -gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR * norm {
                        println!(
                            "[#2515 RUNG] smooth={smooth:.2} A_tt row {row} dir {idx}: RESOLVED \
                             NEGATIVE raw={raw:+.6e} cond={:+.6e} relative={:+.6e}",
                            spectrum.cond_evals[idx],
                            raw / norm.max(f64::MIN_POSITIVE)
                        );
                    }
                }
            }
        }
        let dense_g = dense.gradient();
        let bundled_g = bundled.gradient();
        for i in 0..dense_g.len() {
            println!(
                "[#2515 RUNG] smooth={smooth:.2} coord {i}: dense={:+.10e} bundle={:+.10e} \
                 |Δ|={:.6e}",
                dense_g[i],
                bundled_g[i],
                (dense_g[i] - bundled_g[i]).abs()
            );
        }
        println!(
            "[#2515 RUNG] smooth={smooth:.2} logdet_trace dense={:?}",
            dense.logdet_trace.to_vec()
        );
        println!(
            "[#2515 RUNG] smooth={smooth:.2} logdet_trace bundle={:?}",
            bundled.logdet_trace.to_vec()
        );
    }
}

/// MEASUREMENT — the same ρ sweep, through PRODUCTION, both routes.
///
/// [`zz_attribute_the_broken_ladder_rung_2515`] compares the two gradient
/// ASSEMBLERS at a frozen state and finds the break where the exact `A` goes
/// indefinite. What production does there is a separate question and a more
/// important one: the dense value route returns the typed
/// `IndefiniteObservedInformation` refusal at an exact-`A` saddle, which makes
/// that ρ INFEASIBLE (`+inf`) and steers the outer solver away, while the
/// streaming route's evidence conditioning unit-pins every non-positive
/// direction and therefore has no way to reach that verdict at all.
///
/// If that is what happens, route selection — a memory-planner decision — decides
/// whether a saddle is a saddle, which is the #2486 genus and the substantive
/// half of this issue.
#[test]
fn zz_attribute_the_broken_rung_through_production_2515() {
    let (term, anchor_rho, target, _cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 the broken rung through production",
        );
    for smooth in [-0.9_f64, -1.0, -1.05, -1.1, -1.2, -1.4] {
        let mut rho = anchor_rho.clone();
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = smooth;
        }
        let mut dense = SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        let mut streaming = SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        let rho_flat = dense.baseline_rho.to_flat();
        let route_rho = match streaming.baseline_rho.from_flat(rho_flat.view()) {
            Ok(rho) => rho,
            Err(err) => {
                println!("[#2515 PROD] smooth={smooth:.2}: rho layout: {err}");
                continue;
            }
        };
        let dense_report = match dense.evaluate_outer_criterion_route(&route_rho, true, false) {
            Ok(artifact) => {
                match dense.analytic_gradient_for_outer_evaluation(&route_rho, &artifact) {
                    Ok(gradient) => format!(
                        "cost={:.10e} ‖g‖∞={:.6e}",
                        artifact.cost,
                        gradient.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                    ),
                    Err(err) => format!("cost={:.10e} GRADIENT REFUSED: {err}", artifact.cost),
                }
            }
            Err(err) => format!("VALUE REFUSED: {err}"),
        };
        let streaming_report =
            match streaming.evaluate_outer_criterion_route(&route_rho, false, false) {
                Ok(artifact) => {
                    match streaming.analytic_gradient_for_outer_evaluation(&route_rho, &artifact) {
                        Ok(gradient) => format!(
                            "cost={:.10e} ‖g‖∞={:.6e}",
                            artifact.cost,
                            gradient.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                        ),
                        Err(err) => format!("cost={:.10e} GRADIENT REFUSED: {err}", artifact.cost),
                    }
                }
                Err(err) => format!("VALUE REFUSED: {err}"),
            };
        println!("[#2515 PROD] smooth={smooth:.2} dense    : {dense_report}");
        println!("[#2515 PROD] smooth={smooth:.2} streaming: {streaming_report}");
    }
}

/// #2515 — THE STREAMING OUTER GRADIENT EXISTS ON EVERY STATE THE DENSE ROUTE
/// DIFFERENTIATES, and this is the gate that says so at the production entry
/// point.
///
/// Before the gate freeze was scoped to the whole criterion evaluation, the
/// streaming lane returned a VALUE and then refused its GRADIENT across a whole
/// band of smoothing strengths:
///
/// ```text
/// smooth=-1.10  dense     cost=1.8195496423e1  ‖g‖∞=1.580471e1
///               streaming cost=1.8195496415e1  GRADIENT REFUSED: … refuses a stale
///                         matrix-free system/cache pair (row fingerprints DIFFER,
///                         manifold fingerprints EQUAL)
/// smooth=-1.20  same
/// smooth=-1.40  same
/// ```
///
/// The cache was factored inside `converge_inner_for_undamped_logdet`'s frozen
/// window and the system was assembled after that window closed, so
/// `assemble_arrow_schur_scaled` re-refreshed all three collapse-prevention gates
/// from the moved state and the pair described two operators. Equal manifold
/// fingerprints with unequal row ones is exactly the signature
/// `evidence_assembly_row_fingerprint_sources_2515` attributes to the gate state.
///
/// This walks the same band and requires BOTH routes to produce a `(value,
/// gradient)` pair and to agree on it. A refusal on either side fails, and so does
/// a disagreement — the two are the same defect seen from opposite ends, and a
/// gate that accepted a refusal as "well, it declined safely" would have passed
/// throughout the era this fixes.
#[test]
fn forced_streaming_has_a_gradient_wherever_the_dense_route_does_2515() {
    let (term, anchor_rho, target, _cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 the streaming gradient exists wherever the dense one does",
        );
    let mut compared = 0usize;
    for smooth in [-0.9_f64, -1.05, -1.1, -1.4] {
        let mut rho = anchor_rho.clone();
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = smooth;
        }
        let mut dense = SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        let mut streaming = SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        let rho_flat = dense.baseline_rho.to_flat();
        let route_rho = streaming
            .baseline_rho
            .from_flat(rho_flat.view())
            .expect("#2515: both objectives own the same typed rho layout");

        // The dense route decides whether this state is rankable at all. Where it
        // declines (an exact-A saddle, say), there is nothing for the streaming
        // lane to match and the rung is not a counter-example to anything.
        let Ok(dense_artifact) = dense.evaluate_outer_criterion_route(&route_rho, true, false)
        else {
            println!("[#2515 EXISTS] smooth={smooth:.2}: dense route declines this state");
            continue;
        };
        let Ok(dense_gradient) =
            dense.analytic_gradient_for_outer_evaluation(&route_rho, &dense_artifact)
        else {
            println!("[#2515 EXISTS] smooth={smooth:.2}: dense gradient declines this state");
            continue;
        };

        let streaming_artifact = streaming
            .evaluate_outer_criterion_route(&route_rho, false, false)
            .unwrap_or_else(|err| {
                panic!(
                    "#2515: the dense route ranks smooth={smooth:.2} and the forced streaming \
                     route must too. Got: {err}"
                )
            });
        let streaming_gradient = streaming
            .analytic_gradient_for_outer_evaluation(&route_rho, &streaming_artifact)
            .unwrap_or_else(|err| {
                panic!(
                    "#2515: the dense route DIFFERENTIATES smooth={smooth:.2} and the forced \
                     streaming route refused. A `stale matrix-free system/cache pair` here is \
                     the gate freeze stopping one call short of the evidence assembly that \
                     prices the criterion; anything else is a new defect at the same seam. \
                     Got: {err}"
                )
            });

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for (streamed, direct) in streaming_gradient.iter().zip(dense_gradient.iter()) {
            worst = worst.max((streamed - direct).abs());
            scale = scale.max(direct.abs());
        }
        println!(
            "[#2515 EXISTS] smooth={smooth:.2}: cost dense={:.10e} streaming={:.10e} \
             gradient max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e}",
            dense_artifact.cost, streaming_artifact.cost
        );
        assert_abs_diff_eq!(
            streaming_artifact.cost,
            dense_artifact.cost,
            epsilon = 1.0e-7
        );
        assert!(
            worst <= 1.0e-6 * scale.max(1.0),
            "#2515: the two routes both produced a gradient at smooth={smooth:.2} and they \
             disagree (max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e})"
        );
        compared += 1;
    }
    assert!(
        compared >= 3,
        "#2515: at least three rungs must be rankable by the dense route, or this gate is \
         about a band the dense route also declines; got {compared}"
    );
}
