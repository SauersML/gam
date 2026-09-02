//! #2712 — the from-probes selected-inverse cluster on deflated rows.
//!
//! Two things live here: the reconstruction identity on a SPECTRALLY deflated
//! row (the load-bearing claim), and the measurement that decides which fixture
//! a non-vacuous parity gate can even be stated on.
//!
//! The spectral branch is the one worth pinning for the reconstruction: the
//! correction there is `Σ_{a,b} W[a,b]·M[a,b]·(1 − F[a,b])` with
//! `W = Uᵀ inv_vv U`, so it reads every OFF-DIAGONAL entry of the row's
//! selected-inverse block through the Daleckii–Krein rotation coefficients
//! `(λₘ − 1)/(λₘ − λᵢ)` that couple the kept and deflated subspaces. A
//! reconstruction that recovered only the diagonal passes a diagonal comparison
//! and fails there.
//!
//! # The separation is a property of the fixture, and it had to be measured
//!
//! The issue's own acceptance note is the sharp one: agreement is not evidence
//! unless the deflation-aware and deflation-blind operators provably separate on
//! the fixture, because they coincide wherever the deflation is inactive.
//! `zz_measure_deflation_correction_size_2712` measures exactly that separation
//! on the tree's deflating fixtures, and the numbers are NOT interchangeable —
//! on the ordered Beta–Bernoulli anchor the correction moves `Γ` by `8.5e-8`
//! against `‖Γ‖∞ = 98.9`, because that fixture's deflated direction is a
//! near-null the raw derivative barely touches. The gates below therefore state
//! non-vacuity as a RESOLUTION RATIO against the measured separation rather than
//! as an absolute threshold copied from a sibling gate that was separating two
//! entirely different operators.

use super::tests::{gamma_fd_tiny_fixture, small_two_atom_periodic_term};
use super::tests_logdet_adjoint_780::deflation_blind_cache;
use super::tests_recovery_split_780::{
    FdAnchorRegime, certified_fd_anchor, rho_ladder_family, sparse_lift_ladder,
};
use super::*;

/// The `log λ_sparse` ladder that reaches the deflating regime on the ordered
/// Beta–Bernoulli tiny fixture (shared with the sibling gates in
/// `tests_logdet_adjoint_780`).
const DEFLATING_SPARSE_LIFTS: [f64; 10] = [2.4, 1.8, 1.3, 0.9, 0.5, 0.2, 0.0, -0.3, -0.6, -1.0];

/// The ordered Beta–Bernoulli tiny fixture at its certified deflating anchor.
fn obb_deflated_anchor(label: &str) -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>, ArrowFactorCache)
{
    let (mut term, target, rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    let anchor = certified_fd_anchor(
        label,
        &target,
        FdAnchorRegime::deflated(),
        rho_ladder_family(&term, sparse_lift_ladder(&rho, &DEFLATING_SPARSE_LIFTS), 5),
    );
    (anchor.term, anchor.rho, target, anchor.cache)
}

/// The #2330 residual-excited two-atom circle at a certified deflating anchor.
///
/// This is the SOFTMAX deflating fixture, and it is not interchangeable with the
/// ordered Beta–Bernoulli one: there the deflated direction lives in the LOGIT
/// subspace (the assignment penalty is what drives it), here it lives in the
/// over-parametrized CHART — the coordinate slots. Which subspace it occupies
/// decides which channel's deflation correction is non-zero at all, measured:
/// the ARD log-precision correction contracts `D = hess·eₛeₛᵀ` at a COORDINATE
/// slot `s`, so on the ordered Beta–Bernoulli anchor it evaluates to exactly
/// zero (the deflated direction is orthogonal to every ARD slot) and no parity
/// gate stated there can be non-vacuous.
///
/// The ρ ladder is the one `sae_logdet_theta_adjoint_matches_fd_on_deflated_fixture_2330`
/// declares: #2398 measured that the historical single lift now lands on an
/// exact-`A` saddle where the deflated-PD state does not exist, so the ladder
/// walks the lift down until a deflated maximum certifies.
pub(crate) fn residual_excited_deflated_anchor(
    label: &str,
) -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>, ArrowFactorCache) {
    let (mut term, mut target, mut rho) = gamma_fd_tiny_fixture();
    let (n, p) = (target.nrows(), target.ncols());
    for row in 0..n {
        for col in 0..p {
            let phase = (row as f64 + 0.35) / n as f64;
            let theta = std::f64::consts::TAU * phase;
            target[[row, col]] += 0.6 * (3.0 * theta + 0.5 * col as f64).sin();
        }
    }
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    )
    .expect("off-manifold fixture converges with both atoms alive");

    let eval_rho_ladder: Vec<(String, SaeManifoldRho)> = [
        (0.5_f64, -2.0_f64, -1.2_f64, -1.0_f64),
        (0.5, -1.5, -1.2, -1.0),
        (0.2, -2.0, -1.2, -1.0),
        (0.2, -1.5, -1.0, -0.8),
        (0.0, -1.5, -1.0, -0.8),
        (-0.2, -1.2, -0.8, -0.6),
        (-0.5, -1.0, -0.5, -0.5),
    ]
    .iter()
    .map(|&(sparse, smooth, ard0, ard1)| {
        let mut candidate = rho.clone();
        candidate.log_lambda_sparse = sparse;
        for value in candidate.log_lambda_smooth.iter_mut() {
            *value = smooth;
        }
        candidate.log_ard = vec![ndarray::array![ard0], ndarray::array![ard1]];
        (
            format!("eval rho (sparse={sparse:.1}, smooth={smooth:.1}, ard=[{ard0:.1}, {ard1:.1}])"),
            candidate,
        )
    })
    .collect();
    let anchor = certified_fd_anchor(
        label,
        &target,
        FdAnchorRegime::deflated(),
        rho_ladder_family(&term, eval_rho_ladder, 0),
    );
    (anchor.term, anchor.rho, target, anchor.cache)
}

/// The cold, genuinely indefinite two-atom softmax state, where
/// `factor_spectral_deflated_criterion_row` (#1117) records a real
/// `RowDeflationSpectrum`.
fn spectrally_deflated_cold_state() -> (SaeManifoldTerm, SaeManifoldRho, ArrowFactorCache) {
    let (mut term, target, rho) = small_two_atom_periodic_term();
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("cold arrow assembly");
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("the cold undamped factor is spectrally conditioned (#1117), not refused");
    let spectral_rows = cache
        .deflation_row_spectra
        .iter()
        .filter(|spectrum| spectrum.is_some())
        .count();
    assert!(
        spectral_rows > 0,
        "#2712 premise: this gate needs a row whose deflation carries a RECORDED \
         SPECTRUM (the Daleckii–Krein branch that reads the off-diagonal block). \
         Got {spectral_rows} spectral row(s) and {} gauge direction(s).",
        cache.gauge_deflated_directions
    );
    assert!(
        cache.k > 0,
        "#2712 premise: the fixture must carry a border, or `S⁻¹` is not in play at all"
    );
    (term, rho, cache)
}

/// The exact `(z_j, S⁻¹ z_j)` bundle at full-basis probes `√k·e_j`, where the
/// Hutchinson outer products are algebraically exact.
fn full_basis_bundle(cache: &ArrowFactorCache) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
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
                .expect("schur_inverse_apply")
        })
        .collect();
    (probes, sinv)
}

fn sup_difference(a: &SaeArrowVector, b: &SaeArrowVector) -> f64 {
    a.t.iter()
        .zip(b.t.iter())
        .chain(a.beta.iter().zip(b.beta.iter()))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn sup_norm(a: &SaeArrowVector) -> f64 {
    a.t.iter()
        .chain(a.beta.iter())
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max)
}

/// The shared non-vacuity claim, stated once.
///
/// `parity` is how far the from-probes route is from the dense route;
/// `separation` is how far the DEFLATION-BLIND operator is from the dense route
/// — that is, what a port which silently dropped the Daleckii–Krein correction
/// would score on the same comparison. The gate is meaningful exactly when
/// `parity` is much smaller than `separation`, and the ratio is the margin by
/// which such a port would be caught.
///
/// Deliberately NOT an absolute threshold. The correction's size is a property
/// of the fixture, so an absolute floor copied from a sibling gate would either
/// reject an honest fixture or — much worse — pass one on which the two
/// operators are numerically indistinguishable and the parity assertion proves
/// nothing at all.
fn assert_deflation_resolved(what: &str, parity: f64, separation: f64) {
    assert!(
        separation.is_finite() && separation > 0.0,
        "{what}: the deflation-aware and deflation-blind operators do not separate at \
         all on this fixture (separation {separation:.6e}), so agreement between the \
         dense and from-probes routes says nothing about the Daleckii–Krein \
         correction."
    );
    let margin = separation / parity.max(f64::MIN_POSITIVE);
    assert!(
        parity * 1.0e3 <= separation,
        "{what}: from-probes parity error {parity:.6e} is not small enough against the \
         {separation:.6e} distance to the deflation-blind operator. The gate can only \
         claim the correction is reconstructed if a port that dropped it would be \
         caught by a wide margin; the measured margin is {margin:.3e}x."
    );
    eprintln!(
        "#2712 {what}: parity {parity:.6e}, deflation-blind separation \
         {separation:.6e} — a port that dropped the correction would be caught by \
         {margin:.3e}x"
    );
}

/// A cache whose per-row deflation record is REDIRECTED onto the eigendirection
/// with the largest support at local slot `slot`, keeping every factor, the
/// reduced Schur and the recorded eigenbasis untouched.
///
/// #2712 non-vacuity instrument for the RANK-ONE channels, and the reason one is
/// needed. The ARD log-precision correction contracts `D = hess·eₛeₛᵀ` at a
/// single coordinate slot, so `M = Uᵀ D U` has entries `hess·U[s,a]·U[s,b]` and
/// the whole correction carries a factor `U[s, d]` for the deflated index `d`.
/// Every deflating fixture in the tree happens to deflate a direction with
/// (numerically) no support on the ARD slots — measured at `< 1 ulp` of the trace
/// on both the ordered Beta–Bernoulli and the residual-excited anchors — so a
/// parity gate on that channel is vacuous there no matter how tight its
/// tolerance: it cannot distinguish a route that applies the correction from one
/// that drops it.
///
/// Redirecting the RECORD, not the factor, is deliberate. The claim under test is
/// that the two ROUTES compute the same functional of
/// `(inv_vv, D, dirs, spectrum)`, and both read those four from the same place;
/// the physical consistency of the factor with the record is irrelevant to that
/// claim and would only limit which inputs can be exercised. The from-probes
/// route still has to reconstruct the DEFLATED `inv_vv` from the bundle to agree,
/// because the spectral branch reads `W = Uᵀ inv_vv U` — including its
/// off-diagonal entries.
fn deflation_redirected_to_slot(cache: &ArrowFactorCache, slot: usize) -> ArrowFactorCache {
    let mut redirected = cache.clone();
    let rows = cache.deflation_row_spectra.len();
    let mut dirs: Vec<Vec<Array1<f64>>> = vec![Vec::new(); rows];
    let mut spectra: Vec<Option<RowDeflationSpectrum>> = vec![None; rows];
    for row in 0..rows {
        let Some(spectrum) = cache.deflation_row_spectra[row].as_ref() else {
            continue;
        };
        let q = spectrum.evecs.nrows();
        if slot >= q {
            continue;
        }
        // The eigendirection this slot actually loads onto.
        let mut best = 0usize;
        let mut best_weight = -1.0_f64;
        for column in 0..spectrum.evecs.ncols() {
            let weight = spectrum.evecs[[slot, column]].abs();
            if weight > best_weight {
                best_weight = weight;
                best = column;
            }
        }
        let mut conditioning: Vec<RowSpectralConditioning> =
            spectrum.conditioning.iter().copied().collect();
        let mut cond_evals = spectrum.cond_evals.clone();
        for (index, decision) in conditioning.iter_mut().enumerate() {
            if index == best {
                *decision = RowSpectralConditioning::UnitDeflated;
                cond_evals[index] = 1.0;
            } else {
                *decision = RowSpectralConditioning::Raw;
                cond_evals[index] = spectrum.raw_evals[index];
            }
        }
        dirs[row] = vec![spectrum.evecs.column(best).to_owned()];
        spectra[row] = Some(RowDeflationSpectrum {
            evecs: spectrum.evecs.clone(),
            raw_evals: spectrum.raw_evals.clone(),
            cond_evals,
            conditioning: conditioning.into(),
        });
    }
    redirected.deflated_row_directions = std::sync::Arc::from(dirs);
    redirected.deflation_row_spectra = std::sync::Arc::from(spectra);
    redirected
}

/// How large the Daleckii–Krein correction actually is, per deflating fixture,
/// together with the conditioning decisions that produce it.
///
/// This exists because the size of the correction is the resolution any parity
/// gate on it can possibly have, and that is a property of the fixture which has
/// to be measured rather than assumed. Reported, never asserted on: which
/// eigenvalue a fixture's seed lands on is a measurement about that fixture, and
/// pinning it would turn a diagnostic into a gate on someone else's numerics.
#[test]
fn zz_measure_deflation_correction_size_2712() {
    fn report(
        label: &str,
        term: &SaeManifoldTerm,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) {
        let mut unit_deflated = 0usize;
        let mut floor_clamped = 0usize;
        let mut raw_kept = 0usize;
        let mut rows_with_dirs = 0usize;
        let mut rows_with_spectrum = 0usize;
        for row in 0..cache.row_dims.len() {
            if cache
                .deflated_row_directions
                .get(row)
                .is_some_and(|d| !d.is_empty())
            {
                rows_with_dirs += 1;
            }
            if let Some(spectrum) = cache.deflation_row_spectra.get(row).and_then(Option::as_ref) {
                rows_with_spectrum += 1;
                for decision in spectrum.conditioning.iter() {
                    match decision {
                        RowSpectralConditioning::UnitDeflated => unit_deflated += 1,
                        RowSpectralConditioning::FloorClamped => floor_clamped += 1,
                        RowSpectralConditioning::Raw => raw_kept += 1,
                    }
                }
            }
        }
        let solver = DeflatedArrowSolver::plain(cache);
        let dense = term.logdet_theta_adjoint(rho, cache, &solver);
        let blind_cache = deflation_blind_cache(cache);
        let blind_solver = DeflatedArrowSolver::plain(&blind_cache);
        let blind = term.logdet_theta_adjoint(rho, &blind_cache, &blind_solver);
        let (scale, separation) = match (&dense, &blind) {
            (Ok(d), Ok(b)) => (sup_norm(d), sup_difference(d, b)),
            _ => (f64::NAN, f64::NAN),
        };
        eprintln!(
            "[#2712 correction size] {label}: rows(dirs)={rows_with_dirs} \
             rows(spectrum)={rows_with_spectrum} conditioning(unit={unit_deflated}, \
             floor={floor_clamped}, raw={raw_kept}) ‖Γ‖∞={scale:.6e} \
             ‖Γ_dense − Γ_deflation-blind‖∞={separation:.6e} relative={:.6e}",
            separation / (1.0 + scale)
        );
    }

    let (term, rho, _target, cache) =
        obb_deflated_anchor("#2712 correction size: ordered Beta--Bernoulli");
    report(
        "ordered Beta--Bernoulli tiny (certified deflated anchor)",
        &term,
        &rho,
        &cache,
    );

    let (term, rho, cache) = spectrally_deflated_cold_state();
    report("two-atom softmax cold seed (spectral)", &term, &rho, &cache);

    let (term, rho, _target, cache) =
        residual_excited_deflated_anchor("#2712 correction size: residual-excited softmax");
    report(
        "residual-excited two-atom softmax (certified deflated anchor)",
        &term,
        &rho,
        &cache,
    );
}

/// The reconstruction identity on a SPECTRALLY deflated row, including the
/// off-diagonal entries the Daleckii–Krein rotation term reads.
///
/// This gate needs no separation argument: it compares the two ROUTES for the
/// same block directly, so a route that reconstructed something other than the
/// deflated block differs here whether or not any downstream correction is
/// numerically large on this fixture.
#[test]
fn row_selected_inverse_from_probes_matches_dense_on_spectrally_deflated_rows_2712() {
    let (_term, _rho, cache) = spectrally_deflated_cold_state();
    let (probes, sinv) = full_basis_bundle(&cache);
    let solver = DeflatedArrowSolver::plain(&cache);
    let beta_inv = solver.beta_inv().expect("beta_inv");

    let mut rows = 0usize;
    let mut worst_diagonal = 0.0_f64;
    let mut worst_off_diagonal = 0.0_f64;
    let mut worst_border = 0.0_f64;
    let mut off_diagonal_mass = 0.0_f64;
    let mut block_scale = 0.0_f64;
    for row in 0..cache.row_dims.len() {
        if cache
            .deflation_row_spectra
            .get(row)
            .and_then(Option::as_ref)
            .is_none()
        {
            continue;
        }
        rows += 1;
        let q = cache.row_dims[row];
        let (dense_vv, dense_vbeta) = solver
            .selected_inverse_row_blocks(row, &beta_inv)
            .expect("dense selected inverse row blocks");
        let (probe_vv, probe_vbeta) = row_selected_inverse_from_probes(
            &cache,
            row,
            &probes,
            &sinv,
            true,
            "#2712 spectral reconstruction gate",
        )
        .expect("from-probes selected inverse row blocks");
        for a in 0..q {
            for b in 0..q {
                let err = (dense_vv[[a, b]] - probe_vv[[a, b]]).abs();
                if a == b {
                    worst_diagonal = worst_diagonal.max(err);
                } else {
                    worst_off_diagonal = worst_off_diagonal.max(err);
                    off_diagonal_mass = off_diagonal_mass.max(dense_vv[[a, b]].abs());
                }
                block_scale = block_scale.max(dense_vv[[a, b]].abs());
            }
        }
        for (d, p) in dense_vbeta.iter().zip(probe_vbeta.iter()) {
            worst_border = worst_border.max((d - p).abs());
            block_scale = block_scale.max(d.abs());
        }
    }
    eprintln!(
        "#2712 spectral reconstruction: {rows} spectrally deflated row(s); \
         worst diagonal error {worst_diagonal:.3e}, worst off-diagonal error \
         {worst_off_diagonal:.3e} (off-diagonal magnitude {off_diagonal_mass:.3e}), \
         worst t–β error {worst_border:.3e}, block magnitude {block_scale:.3e}"
    );
    assert!(
        rows > 0,
        "the premise promised a spectrally deflated row and the loop found none"
    );
    // A reconstruction that only got the DIAGONAL right would pass a
    // diagonal-only comparison; the off-diagonal mass is what makes the
    // off-diagonal assertion non-vacuous.
    assert!(
        off_diagonal_mass > 1.0e-6 * (1.0 + block_scale),
        "the deflated selected-inverse block must carry real off-diagonal mass for \
         the Daleckii–Krein rotation term to be under test; got \
         {off_diagonal_mass:.3e} against block magnitude {block_scale:.3e}"
    );
    // RELATIVE: a kept near-null eigendirection legitimately inflates `inv_vv`.
    let tol = 1.0e-11 * (1.0 + block_scale);
    assert!(
        worst_diagonal <= tol && worst_off_diagonal <= tol && worst_border <= tol,
        "from-probes reconstruction must equal the dense selected inverse on a \
         spectrally deflated row: diag {worst_diagonal:.3e}, off-diag \
         {worst_off_diagonal:.3e}, t–β {worst_border:.3e} against tolerance {tol:.3e}"
    );
}

/// Parity for the ARD log-precision Hessian trace on a deflated cache, and — on
/// a deflation record redirected onto an ARD slot — the proof that the parity is
/// sensitive to the Daleckii–Krein correction at all.
///
/// Two claims, because on a real fixture only the first is available:
///
/// 1. On the fixture's OWN deflation, dense and from-probes agree. Reported
///    separation included, so the reader sees that this half is a reconstruction
///    check, not a correction check.
/// 2. On the same cache with the deflation record redirected onto the ARD slot
///    (see [`deflation_redirected_to_slot`]), the correction becomes large and
///    the two routes must still agree by a wide margin against the
///    deflation-blind operator. This is the half that would catch a route which
///    dropped the correction.
#[test]
fn ard_log_precision_hessian_trace_from_probes_matches_dense_on_deflated_rows_2712() {
    let (term, rho, _target, cache) =
        residual_excited_deflated_anchor("#2712 deflated ARD trace parity");
    let (probes, sinv) = full_basis_bundle(&cache);

    let compare = |label: &str, cache: &ArrowFactorCache| -> (f64, f64, f64, usize) {
        let solver = DeflatedArrowSolver::plain(cache);
        let dense = term
            .ard_log_precision_hessian_trace(&rho, cache, &solver, EvidenceOperator::Majorizer)
            .expect("dense ARD trace");
        let blind_cache = deflation_blind_cache(cache);
        let blind_solver = DeflatedArrowSolver::plain(&blind_cache);
        let blind = term
            .ard_log_precision_hessian_trace(
                &rho,
                &blind_cache,
                &blind_solver,
                EvidenceOperator::Majorizer,
            )
            .expect("deflation-blind dense ARD trace");
        let from_probes = term
            .ard_log_precision_hessian_trace_from_probes(
                &rho,
                cache,
                &probes,
                &sinv,
                EvidenceOperator::Majorizer,
            )
            .expect("the from-probes ARD trace must PRICE a deflated cache, not refuse it");
        let mut separation = 0.0_f64;
        let mut parity = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut entries = 0usize;
        for ((d, b), m) in dense.iter().zip(blind.iter()).zip(from_probes.iter()) {
            assert_eq!(d.len(), m.len());
            assert_eq!(d.len(), b.len());
            for ((dv, bv), mv) in d.iter().zip(b.iter()).zip(m.iter()) {
                separation = separation.max((dv - bv).abs());
                parity = parity.max((dv - mv).abs());
                scale = scale.max(dv.abs());
                entries += 1;
            }
        }
        eprintln!(
            "#2712 ARD trace [{label}] over {entries} (atom, axis) entries: magnitude \
             {scale:.6e}, parity {parity:.6e}, deflation-blind separation {separation:.6e}"
        );
        (parity, separation, scale, entries)
    };

    let (parity, _separation, scale, entries) = compare("fixture deflation", &cache);
    assert!(
        entries > 0,
        "the fixture must carry at least one live ARD axis for this gate to mean anything"
    );
    assert!(
        parity <= 1.0e-11 * (1.0 + scale),
        "from-probes ARD trace must equal the dense trace on a deflated cache: \
         {parity:.6e} against trace magnitude {scale:.6e}"
    );

    // WHICH local slots are ARD coordinate slots is a row-layout fact, and this
    // gate should not assume it: sweep every slot of the row block, require
    // parity at each, and require that at least one of them makes the correction
    // decisive. On this fixture slot 0 is a logit slot, so redirecting there
    // leaves the ARD correction at the rounding floor exactly as the fixture's
    // own deflation does.
    let q_max = cache.row_dims.iter().copied().max().unwrap_or(0);
    assert!(q_max > 0, "the fixture must have a non-empty row block");
    let mut best_separation = 0.0_f64;
    let mut best_parity = 0.0_f64;
    for slot in 0..q_max {
        let redirected = deflation_redirected_to_slot(&cache, slot);
        let (parity, separation, scale, _entries) =
            compare(&format!("deflation redirected to slot {slot}"), &redirected);
        assert!(
            parity <= 1.0e-11 * (1.0 + scale),
            "from-probes ARD trace must equal the dense trace on the record redirected \
             to slot {slot}: {parity:.6e} against trace magnitude {scale:.6e}"
        );
        if separation > best_separation {
            best_separation = separation;
            best_parity = parity;
        }
    }
    assert_deflation_resolved("ARD log-precision trace", best_parity, best_separation);
}

/// Separation + parity for the assignment-strength Hessian trace on a deflated
/// cache.
#[test]
fn assignment_log_strength_hessian_trace_from_probes_matches_dense_on_deflated_rows_2712() {
    let (term, rho, _target, cache) =
        obb_deflated_anchor("#2712 deflated assignment-strength trace parity");
    let (probes, sinv) = full_basis_bundle(&cache);
    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("dense assignment-strength trace");

    let blind_cache = deflation_blind_cache(&cache);
    let blind_solver = DeflatedArrowSolver::plain(&blind_cache);
    let blind = term
        .assignment_log_strength_hessian_trace(&rho, &blind_cache, &blind_solver)
        .expect("deflation-blind dense assignment-strength trace");

    let from_probes = term
        .assignment_log_strength_hessian_trace_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("the from-probes assignment trace must PRICE a deflated cache, not refuse it");

    let separation = (dense - blind).abs();
    let parity = (dense - from_probes).abs();
    eprintln!(
        "#2712 assignment trace: dense {dense:.10e}, deflation-blind {blind:.10e}, \
         from-probes {from_probes:.10e}"
    );
    assert!(
        parity <= 1.0e-11 * (1.0 + dense.abs()),
        "from-probes assignment trace must equal the dense trace on a deflated \
         cache: {parity:.6e}"
    );
    assert_deflation_resolved("assignment-strength trace", parity, separation);
}

