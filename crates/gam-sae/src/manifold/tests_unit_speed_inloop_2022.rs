//! #2022 GATE — the in-loop unit-speed (arc-length) chart retraction NEVER
//! corrupts the fit. This is the repurposed gate: because a non-trivial d=1
//! arc-length reparam is NONLINEAR, its recomposition against a finite basis
//! cannot meet the 1e-9 image-freeze tolerance, so `unit_speed_reparameterization`
//! HONEST-SKIPS and the in-loop retraction is a safe near-no-op for typical
//! bases. We therefore gate the three RELIABLY-reachable contracts:
//!
//!  * GRADIENT CONSISTENCY — the assembled latent/decoder gradient matches a
//!    finite difference of the objective at the chart the fit sits on, INCLUDING
//!    the ARD coordinate gradient (the term an errant re-gauge would desync).
//!  * (A) NO-OP SAFETY — on a chart the basis can't reparam-close,
//!    `canonicalize_atom_unit_speed_chart` returns `Ok(false)` and the loss
//!    (incl. ARD), coords, and decoder are byte-unchanged; the in-loop hook
//!    `retract_unit_speed_charts_in_loop` re-gauges 0 atoms.
//!  * (B) EARLY-OUT — an already-unit-speed chart short-circuits
//!    (`unit_speed_retraction` returns `None`, defect < tol) before any refit.
//!
//! The active-retraction "ARD-moves" invariance is deferred to a full-fit
//! integration test (the active path is rarely fired for finite bases).

use crate::assignment::{AssignmentMode, SaeAssignment};
use crate::basis::SaeBasisEvaluator;
use crate::chart_canonicalization::{CanonicalChartTopology, unit_speed_retraction};
use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm};
use gam_terms::latent::LatentManifold;
use ndarray::{Array2, Array3, Array4, Array5, ArrayView2, array};
use std::sync::Arc;

use super::tests::{TestPeriodicEvaluator, periodic_basis};

fn build_circle_term(coords_col: &Array2<f64>, decoder: &Array2<f64>) -> SaeManifoldTerm {
    let n = coords_col.nrows();
    let (phi, jet) = periodic_basis(coords_col);
    let m = phi.ncols();
    assert_eq!(decoder.nrows(), m, "decoder rows must equal basis width");
    let atom = SaeManifoldAtom::new(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let logits = Array2::<f64>::from_elem((n, 1), 2.0); // gated ON (logit 2 > threshold 0)
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords_col.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::threshold_gate(1.0, 0.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

#[test]
fn unit_speed_hook_gradient_consistent_and_noop_safe_2022() {
    let period = 1.0_f64;
    // Uneven circle coordinates ⇒ non-uniform curve speed; row 3 well off the
    // origin so the ARD coordinate gradient there is non-trivial.
    let coords_col = array![
        [0.02_f64],
        [0.10],
        [0.17],
        [0.31],
        [0.55],
        [0.66],
        [0.80],
        [0.95]
    ];
    let n = coords_col.nrows();
    let p = 3usize;
    let (phi0, _) = periodic_basis(&coords_col);
    let m = phi0.ncols();
    let decoder = Array2::<f64>::from_shape_fn((m, p), |(a, b)| {
        0.3 * ((a + 1) as f64) - 0.15 * (b as f64) + 0.05 * ((a * p + b) as f64)
    });

    let mut term = build_circle_term(&coords_col, &decoder);
    let target =
        Array2::<f64>::from_shape_fn((n, p), |(r, c)| 0.2 - 0.05 * (r as f64) + 0.1 * (c as f64));
    let rho = SaeManifoldRho::new(0.0, -4.0, vec![array![0.0]]); // λ_sparse=1, α=1 ARD

    // ================= GRADIENT CONSISTENCY (incl. ARD coord grad) =================
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let h = 1.0e-6_f64;

    // t-coordinate FD — guards ∂(data_fit + ARD)/∂t. A t-perturbation is invisible
    // unless the cached basis Φ is refreshed. Row block = [logit, coord] ⇒ gt[1].
    let row = 3usize;
    let base_flat = term.assignment.coords[0].as_matrix().column(0).to_owned();
    let mut fp = base_flat.clone();
    fp[row] = (fp[row] + h).rem_euclid(period);
    term.assignment.coords[0].set_flat(fp.view());
    term.refresh_basis_from_current_coords().unwrap();
    let lp = term.loss(target.view(), &rho).unwrap().total();
    let mut fm = base_flat.clone();
    fm[row] = (fm[row] - h).rem_euclid(period);
    term.assignment.coords[0].set_flat(fm.view());
    term.refresh_basis_from_current_coords().unwrap();
    let lm = term.loss(target.view(), &rho).unwrap().total();
    term.assignment.coords[0].set_flat(base_flat.view());
    term.refresh_basis_from_current_coords().unwrap();
    let gt_fd = (lp - lm) / (2.0 * h);
    let gt_analytic = sys.rows[row].gt[1];
    assert!(
        gt_fd.abs() > 1.0e-3,
        "coord gradient must be non-trivial so the ARD term is genuinely guarded (got {gt_fd})"
    );
    assert!(
        (gt_analytic - gt_fd).abs() <= 1.0e-4 * (1.0 + gt_fd.abs()),
        "assembled coord gradient {gt_analytic} must match FD {gt_fd} (ARD/coord desync guard)"
    );

    // β FD — ≥2 entries, tight tol (β is Euclidean; no refresh needed).
    for &(bm, bp) in &[(0usize, 0usize), (1usize, 1usize)] {
        let beta_idx = bm * p + bp; // single atom: β flattened row-major (m × p)
        let g_analytic = sys.gb[beta_idx];
        let base = term.atoms[0].decoder_coefficients[[bm, bp]];
        term.atoms[0].decoder_coefficients[[bm, bp]] = base + h;
        let lpp = term.loss(target.view(), &rho).unwrap().total();
        term.atoms[0].decoder_coefficients[[bm, bp]] = base - h;
        let lmm = term.loss(target.view(), &rho).unwrap().total();
        term.atoms[0].decoder_coefficients[[bm, bp]] = base; // restore
        let g_fd = (lpp - lmm) / (2.0 * h);
        assert!(
            (g_analytic - g_fd).abs() <= 1.0e-6 * (1.0 + g_fd.abs()),
            "β-gradient[{bm},{bp}] {g_analytic} must match FD {g_fd}"
        );
    }

    // ============================ (A) NO-OP SAFETY ============================
    let l0 = term.loss(target.view(), &rho).unwrap();
    let coords0 = term.assignment.coords[0].as_matrix().column(0).to_owned();
    let decoder0 = term.atoms[0].decoder_coefficients.clone();
    let topo = CanonicalChartTopology::Circle { period };
    // A non-uniform harmonic chart cannot recompose to 1e-9 ⇒ honest-skip.
    let applied = term.canonicalize_atom_unit_speed_chart(0, &topo).unwrap();
    assert!(
        !applied,
        "a non-uniform harmonic d=1 chart cannot meet the 1e-9 recomposition gate ⇒ must honest-skip"
    );
    let l1 = term.loss(target.view(), &rho).unwrap();
    assert!(
        (l1.data_fit - l0.data_fit).abs() < 1.0e-12,
        "honest-skip must not move data_fit"
    );
    assert!(
        (l1.smoothness - l0.smoothness).abs() < 1.0e-12,
        "honest-skip must not move smoothness"
    );
    assert!(
        (l1.ard - l0.ard).abs() < 1.0e-12,
        "honest-skip must not move ARD"
    );
    assert!(
        (l1.assignment_sparsity - l0.assignment_sparsity).abs() < 1.0e-12,
        "honest-skip must not move the assignment prior"
    );
    let coords1 = term.assignment.coords[0].as_matrix().column(0).to_owned();
    let cdrift = coords0
        .iter()
        .zip(coords1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        cdrift == 0.0,
        "honest-skip must leave coords byte-unchanged; drift {cdrift}"
    );
    let ddrift = decoder0
        .iter()
        .zip(term.atoms[0].decoder_coefficients.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        ddrift == 0.0,
        "honest-skip must leave the decoder byte-unchanged; drift {ddrift}"
    );
    // The in-loop hook re-gauges 0 atoms on this term and is a strict no-op.
    let n_retracted = term.retract_unit_speed_charts_in_loop().unwrap();
    assert_eq!(
        n_retracted, 0,
        "in-loop hook must re-gauge 0 atoms when nothing can reparam-close"
    );
    let l2 = term.loss(target.view(), &rho).unwrap();
    assert!((l2.data_fit - l0.data_fit).abs() < 1.0e-12 && (l2.ard - l0.ard).abs() < 1.0e-12);

    // ============================ (B) EARLY-OUT ============================
    // A constant-speed circle chart γ(x) = (sin 2πx, cos 2πx) via the harmonic
    // basis [1, sin, cos] (decoder rows const→0, sin→e0, cos→e1) has ‖γ'‖ = 2π
    // everywhere, so the speed-uniformity defect is 0 < tol and the retraction
    // early-outs (None) before any recomposition.
    let uniform_decoder =
        Array2::<f64>::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
    let early = unit_speed_retraction(
        &TestPeriodicEvaluator,
        uniform_decoder.view(),
        coords_col.column(0),
        &topo,
    )
    .unwrap();
    assert!(
        early.is_none(),
        "an already-unit-speed chart must early-out (defect < UNIT_SPEED_INLOOP_DEFECT_TOL)"
    );
}

/// #2022 GATE (inner scale-quotient) — the joint (δB, δs) SCALE-gauge step
/// projection applied per Newton trial in `apply_decoder_step_from_flat`:
///
///  * **Gauge-orthogonality (exact):** the applied joint step
///    `Δ = (B₁ − B₀, s₁ − s₀)` has zero component along the exact scale-gauge
///    nullvector `v = (vec B₀, −1)` — `⟨ΔB, B₀⟩_F − Δs = 0` to round-off — so
///    the inner walk cannot slide along the unobservable scale↔amplitude
///    trade-off (the direction the PD-floor / deflation budget floor).
///  * **First-order image exactness (the non-detonation law):** the projected
///    step's physical atom image `exp(s₁)·Φ·B₁` differs from the RAW
///    (quotient-off) step's image by `O(step²)` only. This is what
///    distinguishes the projection from the #2100 mid-solve ‖B‖≡1 peel: no
///    parameter is rewritten, the iterate path is continuous in `step_size`,
///    and a healthy fit's line search sees the same first-order model.
///  * **Opt-out byte-identity:** with `quotient_scale` disabled the applied
///    step is exactly `B += step·δB`, `s` untouched.
///  * **Snapshot round-trip:** `restore_mutable_state` returns `s` to its
///    banked value (the log_amplitude leg of the keep-best snapshot).
#[test]
fn joint_step_projection_gauge_orthogonal_first_order_exact_2022() {
    let coords_col = array![
        [0.02_f64],
        [0.10],
        [0.17],
        [0.31],
        [0.55],
        [0.66],
        [0.80],
        [0.95]
    ];
    let n = coords_col.nrows();
    let p = 3usize;
    let (phi0, _) = periodic_basis(&coords_col);
    let m = phi0.ncols();
    let decoder = Array2::<f64>::from_shape_fn((m, p), |(a, b)| {
        0.3 * ((a + 1) as f64) - 0.15 * (b as f64) + 0.05 * ((a * p + b) as f64)
    });
    let mut term = build_circle_term(&coords_col, &decoder);
    assert!(
        term.quotient_scale(),
        "quotient_scale is default-on (#2228); this gate exercises the default path"
    );
    // Give the atom a non-trivial starting amplitude so the s-channel is live.
    term.atoms[0].log_amplitude = 0.35;

    let q = term.assignment.row_block_dim();
    let beta_dim = term.beta_dim();
    assert_eq!(beta_dim, m * p);
    // A deliberately radial-heavy decoder step: δB = 0.6·B₀ + (a structured
    // tangent part), so the projection has real work to do.
    let b0 = term.atoms[0].decoder_coefficients.clone();
    let s0 = term.atoms[0].log_amplitude;
    let mut delta_beta = ndarray::Array1::<f64>::zeros(beta_dim);
    for a in 0..m {
        for b in 0..p {
            delta_beta[a * p + b] =
                0.6 * b0[[a, b]] + 0.05 * ((a + 2 * b) as f64) - 0.03 * (b as f64);
        }
    }
    let delta_t = ndarray::Array1::<f64>::zeros(n * q);
    let eps = 1.0e-4_f64;

    // Bank the state, then apply the projected step on the default path.
    let snapshot = term.snapshot_mutable_state();
    term.apply_newton_step(delta_t.view(), delta_beta.view(), eps)
        .unwrap();
    let b1 = term.atoms[0].decoder_coefficients.clone();
    let s1 = term.atoms[0].log_amplitude;

    // (1) Exact gauge-orthogonality of the applied joint step against v = (B₀, −1).
    let db_b: f64 = b1
        .iter()
        .zip(b0.iter())
        .map(|(after, before)| (after - before) * before)
        .sum();
    let ds = s1 - s0;
    let ortho = db_b - ds;
    let scale = eps * (1.0 + b0.iter().map(|v| v * v).sum::<f64>());
    assert!(
        ortho.abs() <= 1.0e-12 * scale.max(1.0),
        "applied joint step must be exactly orthogonal to the scale-gauge nullvector: \
         ⟨ΔB,B₀⟩−Δs = {ortho:.3e}"
    );
    assert!(
        ds.abs() > 0.0,
        "a radial-heavy δB must genuinely move the s-channel (Δs = {ds:.3e})"
    );

    // (2) First-order image exactness vs the RAW quotient-off step.
    let mut raw = build_circle_term(&coords_col, &decoder);
    raw.atoms[0].log_amplitude = s0;
    raw.set_quotient_scale(false);
    raw.apply_newton_step(delta_t.view(), delta_beta.view(), eps)
        .unwrap();
    let raw_b1 = raw.atoms[0].decoder_coefficients.clone();
    assert_eq!(
        raw.atoms[0].log_amplitude, s0,
        "quotient-off apply must not touch log_amplitude"
    );
    // Opt-out byte-identity: B += eps·δB exactly.
    for a in 0..m {
        for b in 0..p {
            let expect = b0[[a, b]] + eps * delta_beta[a * p + b];
            assert!(
                (raw_b1[[a, b]] - expect).abs() <= 1.0e-15 * (1.0 + expect.abs()),
                "quotient-off decoder step must be the plain += (entry {a},{b})"
            );
        }
    }
    // Physical atom images: exp(s)·Φ·B (same Φ both sides — coords untouched).
    let img_proj = term.atoms[0]
        .basis_values
        .dot(&term.atoms[0].decoder_coefficients)
        .mapv(|v| v * s1.exp());
    let img_raw = raw.atoms[0]
        .basis_values
        .dot(&raw.atoms[0].decoder_coefficients)
        .mapv(|v| v * s0.exp());
    let img_scale = img_raw.iter().fold(1.0_f64, |acc, &v| acc.max(v.abs()));
    let img_diff = img_proj
        .iter()
        .zip(img_raw.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    // Second-order bound: the two applies agree to O(eps²) — allow a generous
    // curvature constant, still ~6 orders below a first-order mismatch (eps·scale).
    assert!(
        img_diff <= 100.0 * eps * eps * img_scale,
        "projected and raw steps must agree to first order in the physical image: \
         max diff {img_diff:.3e} vs bound {:.3e}",
        100.0 * eps * eps * img_scale
    );

    // (3) Snapshot round-trip restores the s-channel exactly.
    term.restore_mutable_state(&snapshot).unwrap();
    assert_eq!(
        term.atoms[0].log_amplitude.to_bits(),
        s0.to_bits(),
        "restore_mutable_state must return log_amplitude to the banked value"
    );
    let b_restored = &term.atoms[0].decoder_coefficients;
    let bdrift = b_restored
        .iter()
        .zip(b0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(bdrift == 0.0, "restore must return the decoder exactly");
}

/// #1939 GATE — the empirical-Bayes amplitude prior (SCAD + evidence-ARD on
/// the explicit amplitudes) is PRICED BY THE FIT-LEVEL REFEREE. The boundary
/// amplitude solve persists its converged `(α, λ_k, σ̂²)` on the term, and
/// `penalized_objective_total` must include the prior energy in data-fit units:
/// otherwise the keep-best incumbent ranks states on the bare data-fit and
/// silently vetoes the prior's accepted shrinkage (the #2230 desync class on
/// the amplitude channel).
#[test]
fn amplitude_prior_priced_in_fit_referee_1939() {
    let coords_col = array![
        [0.02_f64],
        [0.10],
        [0.17],
        [0.31],
        [0.55],
        [0.66],
        [0.80],
        [0.95]
    ];
    let n = coords_col.nrows();
    let p = 3usize;
    let (phi0, _) = periodic_basis(&coords_col);
    let m = phi0.ncols();
    let decoder = Array2::<f64>::from_shape_fn((m, p), |(a, b)| {
        0.3 * ((a + 1) as f64) - 0.15 * (b as f64) + 0.05 * ((a * p + b) as f64)
    });
    let mut term = build_circle_term(&coords_col, &decoder);
    let target =
        Array2::<f64>::from_shape_fn((n, p), |(r, c)| 0.2 - 0.05 * (r as f64) + 0.1 * (c as f64));
    let rho = SaeManifoldRho::new(0.0, -4.0, vec![array![0.0]]);

    assert!(
        term.amplitude_prior.is_none() && term.amplitude_prior_value(1.0) == 0.0,
        "no amplitude-prior energy before the first boundary solve (historical objective)"
    );
    term.optimize_log_amplitudes_closed_form(target.view(), &rho)
        .unwrap();
    assert!(
        term.amplitude_prior.is_some(),
        "the boundary amplitude solve must persist its converged prior state"
    );
    let v = term.amplitude_prior_value(1.0);
    assert!(
        v.is_finite() && v >= 0.0,
        "amplitude-prior energy must be finite and non-negative; got {v}"
    );

    // The referee objective includes EXACTLY the persisted prior energy.
    let with_prior = term
        .penalized_objective_total(target.view(), &rho, None, 1.0)
        .unwrap();
    let saved = term.amplitude_prior.take();
    let without_prior = term
        .penalized_objective_total(target.view(), &rho, None, 1.0)
        .unwrap();
    term.amplitude_prior = saved;
    assert!(
        (with_prior - without_prior - v).abs() <= 1.0e-12 * (1.0 + with_prior.abs()),
        "penalized_objective_total must add the amplitude-prior energy exactly: \
         with {with_prior}, without {without_prior}, prior {v}"
    );

    // Stale-K guard: a λ vector that no longer matches the dictionary is skipped.
    if let Some(prior) = term.amplitude_prior.as_mut() {
        prior.scad_lambda.push(1.0);
    }
    assert_eq!(
        term.amplitude_prior_value(1.0),
        0.0,
        "a K-mismatched (stale) prior state must be skipped, not mispriced"
    );
}

/// #2099 GATE (fit-level decoder-scale invariance, the acceptance pin the
/// issue's close-audit says is "coupled to the amplitude-gauge fix"): two
/// representations of the SAME fitted model that differ by the pure scale gauge
/// `B ↦ c·B`, `s ↦ s − ln c` (identical physical dictionary `exp(s)·Φ·B`) must
/// yield
///  1. the same data-fit as-is (the physical image is identical),
///  2. the IDENTICAL canonical parameter state after the in-loop scale
///     retraction that runs at every accepted iterate (`B/‖B‖`, `s + ln‖B‖`),
///  3. the same criterion at that canonical state, and
///  4. the same subsequent inner-Newton trajectory (one solve+apply step from
///     each representation lands on the same physical image).
/// Pre-canonicalization the RAW smoothness quadratic `½λ tr(BᵀSB)` is allowed
/// to differ (it reads the representative, scaling as c²) — that residual gauge
/// dependence is exactly why the boundary retraction + the #2022 step
/// projection exist, and this gate pins that they remove it at fit level.
#[test]
fn fit_level_decoder_rescale_invariance_2099() {
    let coords_col = array![
        [0.02_f64],
        [0.10],
        [0.17],
        [0.31],
        [0.55],
        [0.66],
        [0.80],
        [0.95]
    ];
    let n = coords_col.nrows();
    let p = 3usize;
    let (phi0, _) = periodic_basis(&coords_col);
    let m = phi0.ncols();
    let decoder = Array2::<f64>::from_shape_fn((m, p), |(a, b)| {
        0.3 * ((a + 1) as f64) - 0.15 * (b as f64) + 0.05 * ((a * p + b) as f64)
    });
    let c = 3.7_f64; // the gauge scale relating the two representations
    let decoder_scaled = decoder.mapv(|v| c * v);

    let mut rep_a = build_circle_term(&coords_col, &decoder);
    rep_a.atoms[0].log_amplitude = 0.2;
    let mut rep_b = build_circle_term(&coords_col, &decoder_scaled);
    rep_b.atoms[0].log_amplitude = 0.2 - c.ln();
    assert!(rep_a.quotient_scale() && rep_b.quotient_scale());

    let target =
        Array2::<f64>::from_shape_fn((n, p), |(r, cc)| 0.2 - 0.05 * (r as f64) + 0.1 * (cc as f64));
    let rho = SaeManifoldRho::new(0.0, -4.0, vec![array![0.0]]);

    // (1) The physical image is identical ⇒ data-fit invariant as-is.
    let la = rep_a.loss(target.view(), &rho).unwrap();
    let lb = rep_b.loss(target.view(), &rho).unwrap();
    assert!(
        (la.data_fit - lb.data_fit).abs() <= 1.0e-10 * (1.0 + la.data_fit.abs()),
        "data-fit must be decoder-rescale invariant: {} vs {}",
        la.data_fit,
        lb.data_fit
    );

    // (2) The in-loop scale retraction lands both on the SAME canonical state.
    rep_a.retract_decoder_gauge_in_loop();
    rep_b.retract_decoder_gauge_in_loop();
    let s_a = rep_a.atoms[0].log_amplitude;
    let s_b = rep_b.atoms[0].log_amplitude;
    assert!(
        (s_a - s_b).abs() <= 1.0e-12 * (1.0 + s_a.abs()),
        "canonical log-amplitudes must agree: {s_a} vs {s_b}"
    );
    let frame_drift = rep_a.atoms[0]
        .decoder_coefficients
        .iter()
        .zip(rep_b.atoms[0].decoder_coefficients.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        frame_drift <= 1.0e-13,
        "canonical unit-Frobenius frames must agree; max drift {frame_drift}"
    );

    // (3) Criterion invariance at the canonical state (all components now read
    // the same representative).
    let ca = rep_a.loss(target.view(), &rho).unwrap();
    let cb = rep_b.loss(target.view(), &rho).unwrap();
    assert!(
        (ca.total() - cb.total()).abs() <= 1.0e-10 * (1.0 + ca.total().abs()),
        "canonical-state criterion must be invariant: {} vs {}",
        ca.total(),
        cb.total()
    );

    // (4) Trajectory invariance: one inner Newton solve+apply from each
    // canonical representation lands on the same physical image (the #2022
    // projected apply keeps the walk on the quotient).
    let (dta, dba) = rep_a
        .solve_newton_step(target.view(), &rho, None, 1.0e-4, 1.0e-4)
        .unwrap();
    let (dtb, dbb) = rep_b
        .solve_newton_step(target.view(), &rho, None, 1.0e-4, 1.0e-4)
        .unwrap();
    rep_a.apply_newton_step(dta.view(), dba.view(), 0.5).unwrap();
    rep_b.apply_newton_step(dtb.view(), dbb.view(), 0.5).unwrap();
    let img_a = rep_a.atoms[0]
        .basis_values
        .dot(&rep_a.atoms[0].decoder_coefficients)
        .mapv(|v| v * rep_a.atoms[0].log_amplitude.exp());
    let img_b = rep_b.atoms[0]
        .basis_values
        .dot(&rep_b.atoms[0].decoder_coefficients)
        .mapv(|v| v * rep_b.atoms[0].log_amplitude.exp());
    let img_scale = img_a.iter().fold(1.0_f64, |acc, &v| acc.max(v.abs()));
    let img_drift = img_a
        .iter()
        .zip(img_b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        img_drift <= 1.0e-8 * img_scale,
        "one Newton step from the two representations must land on the same \
         physical image; max drift {img_drift:.3e} (scale {img_scale:.3e})"
    );
}

/// A faithful monomial line basis `Φ(t) = [1, t, t²]` with its exact jet
/// `∂Φ/∂t = [0, 1, 2t]`. `d = 1`, Euclidean (Interval) chart. It exists so the
/// #2070 active-retraction gate below can DRIVE the arc-length retraction to
/// `applied = true` on a FAITHFUL basis (jet == dΦ/dt), which the harmonic
/// `[1, sin, cos]` chart cannot: a non-uniform arc-length reparam is nonlinear,
/// but when the reparameterized decoded IMAGE is affine in the new arc-length
/// coordinate (as it is for the `γ(t) = t²` decoder below — see the test) the
/// `{1, t, t²}` basis reproduces it EXACTLY, so the image-freeze recomposition
/// clears the `1e-9` gate and the retraction fires for real.
#[derive(Debug)]
struct MonomialLineEvaluator;

impl SaeBasisEvaluator for MonomialLineEvaluator {
    /// `∂²Φ/∂t² = [0, 0, 2]` — the exact second jet of `[1, t, t²]`.
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        if coords.ncols() != 1 {
            return Some(Err(format!(
                "MonomialLineEvaluator::second_jet_dyn: expected latent_dim 1, got {}",
                coords.ncols()
            )));
        }
        let n = coords.nrows();
        let mut h = Array4::<f64>::zeros((n, 3, 1, 1));
        for row in 0..n {
            h[[row, 2, 0, 0]] = 2.0;
        }
        Some(Ok(h))
    }

    /// `∂³Φ/∂t³ = 0` — `[1, t, t²]` is quadratic, so every third derivative vanishes.
    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        if coords.ncols() != 1 {
            return Some(Err(format!(
                "MonomialLineEvaluator::third_jet_dyn: expected latent_dim 1, got {}",
                coords.ncols()
            )));
        }
        Some(Ok(Array5::<f64>::zeros((coords.nrows(), 3, 1, 1, 1))))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != 1 {
            return Err(format!(
                "MonomialLineEvaluator: expected latent_dim 1, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, 3));
        let mut jet = Array3::<f64>::zeros((n, 3, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = t;
            phi[[row, 2]] = t * t;
            // ∂Φ/∂t = [0, 1, 2t] — the true jet, so the curve speed the retraction
            // reads is the honest |γ'(t)|.
            jet[[row, 1, 0]] = 1.0;
            jet[[row, 2, 0]] = 2.0 * t;
        }
        Ok((phi, jet))
    }
}

fn build_line_term(coords_col: &Array2<f64>, decoder: &Array2<f64>) -> SaeManifoldTerm {
    let n = coords_col.nrows();
    let (phi, jet) = MonomialLineEvaluator
        .evaluate(coords_col.view())
        .expect("monomial basis evaluates");
    let m = phi.ncols();
    assert_eq!(decoder.nrows(), m, "decoder rows must equal basis width");
    let atom = SaeManifoldAtom::new(
        "line",
        SaeAtomBasisKind::Linear,
        1,
        phi,
        jet,
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(MonomialLineEvaluator));
    let logits = Array2::<f64>::from_elem((n, 1), 2.0); // gated ON (logit 2 > threshold 0)
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords_col.clone()],
        vec![LatentManifold::Euclidean],
        AssignmentMode::threshold_gate(1.0, 0.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

/// #2070 GATE (active path) — DRIVE the in-loop unit-speed retraction to
/// `applied = true` and assert it moves ONLY the identifying ARD coordinate
/// prior. This is the un-weakened companion to
/// `unit_speed_hook_gradient_consistent_and_noop_safe_2022`: that one pins the
/// honest-skip / no-op contract for a chart the basis cannot reparam-close; this
/// one pins the ACTIVE transport for a chart it CAN.
///
/// Fixture: monomial line chart `Φ = [1, t, t²]` with a decoder that traces the
/// 1-D image `γ(t) = t²`. The curve speed `|γ'| = 2t` is NON-uniform (so the
/// retraction does not early-out), yet the arc-length-reparameterized image is
/// AFFINE in the new coordinate — `t² = t_min² + (t_max² − t_min²)·t̃` — so the
/// `{1, t, t²}` basis reproduces it with zero image drift and the retraction
/// FIRES. Because the reparam is image-frozen, data-fit + intrinsic smoothness
/// are invariant and the ENTIRE loss change is the ARD coordinate prior
/// re-evaluated at the re-gauged `t̃` (Euclidean `½α Σ t²`), which must equal the
/// closed-form energy delta exactly.
#[test]
fn unit_speed_active_retraction_moves_only_ard_2070() {
    // Fitted coordinates on the interval, bounded away from 0 so `|γ'| = 2t > 0`
    // and the chart is non-degenerate; unevenly spaced so `t̃ = t²`-normalized
    // genuinely re-gauges every row.
    let coords_col = array![[0.15_f64], [0.30], [0.42], [0.58], [0.71], [0.88]];
    let n = coords_col.nrows();
    let p = 1usize;
    // Decoder selects the t² monomial ⇒ γ(t) = t² (single output channel).
    let decoder = Array2::<f64>::from_shape_vec((3, p), vec![0.0, 0.0, 1.0]).unwrap();
    let mut term = build_line_term(&coords_col, &decoder);
    let target = Array2::<f64>::from_shape_fn((n, p), |(r, _)| 0.10 + 0.05 * r as f64);
    // α = exp(0) = 1 Euclidean ARD precision; modest smoothness.
    let rho = SaeManifoldRho::new(0.0, -4.0, vec![array![0.0]]);

    let l0 = term.loss(target.view(), &rho).unwrap();
    let coords0 = term.assignment.coords[0].as_matrix().column(0).to_owned();

    // ---- DRIVE the active retraction through the in-loop apply-path ----
    let topo = CanonicalChartTopology::Interval;
    let applied = term.canonicalize_atom_unit_speed_chart(0, &topo).unwrap();
    assert!(
        applied,
        "the monomial line chart's image is affine in arc-length ⇒ the ACTIVE arc-length \
         retraction MUST fire (this is the reachable faithful d=1 active path)"
    );

    let l1 = term.loss(target.view(), &rho).unwrap();
    let coords1 = term.assignment.coords[0].as_matrix().column(0).to_owned();

    // The retraction genuinely re-gauged the coordinates (not a silent no-op).
    let cdrift = coords0
        .iter()
        .zip(coords1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        cdrift > 1.0e-3,
        "the active retraction must MOVE the coordinates; max drift {cdrift}"
    );

    // (data-fit) invariant — the reparam is image-frozen (Φ(t̃)B̃ ≈ Φ(t)B).
    assert!(
        (l1.data_fit - l0.data_fit).abs() <= 1.0e-6 * (1.0 + l0.data_fit.abs()),
        "data_fit must be invariant under the image-frozen retraction: {} vs {}",
        l1.data_fit,
        l0.data_fit
    );
    // (smoothness) invariant — the congruence transport preserves BᵀSB.
    assert!(
        (l1.smoothness - l0.smoothness).abs() <= 1.0e-6 * (1.0 + l0.smoothness.abs()),
        "smoothness must be invariant (transport preserves BᵀSB): {} vs {}",
        l1.smoothness,
        l0.smoothness
    );
    // (assignment prior) invariant — routing logits are untouched by the reparam.
    assert!(
        (l1.assignment_sparsity - l0.assignment_sparsity).abs()
            <= 1.0e-9 * (1.0 + l0.assignment_sparsity.abs()),
        "the assignment prior must be invariant under the retraction"
    );

    // (ARD) the identifying coordinate prior actually MOVED under the ACTIVE path.
    let ard_delta = l1.ard - l0.ard;
    assert!(
        ard_delta.abs() > 1.0e-6,
        "the ARD coordinate prior MUST move under the active retraction (it pins the \
         residual gauge); delta {ard_delta}"
    );

    // (moves ONLY ARD) every non-ARD penalized-loss component is invariant, so the
    // entire loss delta is carried by the ARD term alone.
    let total_delta = l1.total() - l0.total();
    assert!(
        (total_delta - ard_delta).abs() <= 1.0e-6 * (1.0 + ard_delta.abs()),
        "the retraction must move ONLY the ARD prior; total Δ {total_delta} vs ARD Δ {ard_delta}"
    );

    // (ARD is EXACTLY the reparam effect) the Euclidean coordinate energy
    // `½α Σ t²` re-evaluated at the re-gauged coords equals the ARD change; the
    // precision normalizer depends only on (α, n) and cancels in the delta.
    let alpha = SaeManifoldRho::stable_exp_strength(0.0);
    let expected: f64 = (0..n)
        .map(|i| 0.5 * alpha * (coords1[i] * coords1[i] - coords0[i] * coords0[i]))
        .sum();
    assert!(
        (ard_delta - expected).abs() <= 1.0e-9 * (1.0 + expected.abs()),
        "ARD delta {ard_delta} must equal the Euclidean coordinate-energy delta {expected} \
         at the reparam'd coords (confirms it is the reparam effect, not a bookkeeping bug)"
    );

    // The in-loop hook itself runs safely on this reachable-active-path atom.
    let n_retracted = term.retract_unit_speed_charts_in_loop().unwrap();
    // The chart is now arc-length (just retracted) so the hook re-gauges 0 atoms
    // on this second call — idempotent, never perturbing a settled fit.
    assert_eq!(
        n_retracted, 0,
        "a freshly-retracted chart is already unit-speed ⇒ the hook is idempotent (0 re-gauges)"
    );
}
