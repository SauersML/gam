//! #2228 / #2132 / #1095 / #1893 — the DEFINITIVE gate-decoder CO-COLLAPSE
//! RECOVERY gate. This is the proof-of-cure the cluster closes on: an IBP-gated
//! atom, at the smoothing strength REML actually SELECTS (material, NOT the
//! negligible `log λ = -6` the other co-collapse fixtures pin), reconstructs its
//! planted signal ONLY when the amplitude scale-gauge pin is engaged.
//!
//! THE BUG (memory `project_massive_k_manifold_is_silently_linear` /
//! `project_1095_2228_inner_certificate_gated_on_undamped_factor`): the decoder
//! DATA-fit β-Hessian is gate-weighted by `a_k²` (the row gate enters the design
//! as `a_k·Φ·B`), but the decoder SMOOTHNESS penalty `S_k` is scaled by
//! `λ_smooth` ALONE (`rho.log_lambda_smooth`, ungated — see
//! `SaeManifoldRho::log_lambda_smooth`). The decoder therefore sees an EFFECTIVE
//! shrinkage `λ_smooth / a_k²`. Under the ordered IBP-MAP prior a K=1 atom's gate
//! is CAPPED at `a_0 = σ(logit)·π_0 = σ(logit)·(α/(α+1)) ≈ 0.5` (α = 1), so the
//! penalty over-shrinks the decoder magnitude by `≈ 1/0.5² = 4×`. At a MATERIAL
//! `λ_smooth` (the regime REML lands in on noisy data) that 4× over-shrink crushes
//! the signal-bearing fundamental and the gated reconstruction `a_0·Φ·B` cannot
//! reach the target — R² collapses to `≈ 1 − 0.75² = 0.44` (the borderline
//! `top_k = 1` figure quoted at `fit_drivers.rs`'s pin site).
//!
//! THE FIX (`gauge.rs::pin_scale_gauge`, engaged by `quotient_scale = true`, now
//! default-on): PEEL each atom's `‖B_k‖_F` out of the PENALIZED decoder and into
//! the UNPENALIZED log-amplitude `s_k`, leaving the penalty to shape only a
//! unit-Frobenius frame `B̂_k`. The magnitude then lives in `s_k` (which the
//! smoothness penalty never touches) and is re-homed by the data-optimal amplitude
//! solve to `exp(s_k) ≈ 1/a_k`, so the gated contribution reaches the target
//! REGARDLESS of the `π_0` gate cap.
//!
//! WHY THE EXISTING FIXTURES ARE NOT A RECOVERY PROOF: every other co-collapse
//! test (`tests_cocollapse_disjoint_2027`, `quotient_scale_on_does_not_crash_k1`)
//! runs at `log λ_smooth = -6` — a NEGLIGIBLE penalty, where `λ_smooth/a_k²` is
//! still tiny, so the decoder is NOT over-shrunk and the pin is (correctly) inert.
//! Those tests prove the pin does no HARM; they do not exercise the regime where
//! it is LOAD-BEARING. The co-collapse only bites once `λ_smooth` is material
//! relative to the data term — i.e. exactly the level REML selects on real (noisy)
//! activations. This gate puts the fixture in that regime and demands the cure.
//!
//! ACCEPTANCE (the honest recovery bar):
//!   (0) REML genuinely SELECTS a material smoothing `λ*` on this data — its
//!       evidence at `λ*` strictly beats the negligible `log λ = -6`, and `λ*` is
//!       well above the tiny floor (so this is not a cherry-picked `λ`).
//!   (1) WITHOUT the pin (`quotient_scale = false`), at `λ*`, the IBP-gated fit
//!       co-collapses: reconstruction R² < 0.5.
//!   (2) WITH the pin (`quotient_scale = true`), at the SAME `λ*`, the fit
//!       recovers the planted circle: R² > 0.9.
//!
//! zz_measure discipline: every EV / λ / criterion is `eprintln`'d so a byte-
//! identical re-measure on MSI CI reveals the actual regime, not a silent pass.

use super::tests::deterministic_circle_noise;
use super::*;

/// A single clean planted circle (the fundamental `cos θ, sin θ` on output
/// columns 0 and 1) plus isotropic deterministic noise across ALL `p` columns.
/// The circle is the ONLY structure, so a unit-Frobenius periodic frame whose
/// magnitude is free reconstructs it to near-unit EV; the noise on the higher
/// output columns is what makes REML want a MATERIAL smoothing strength (it has
/// `m ≫ 3` harmonic coefficients, most of which carry only noise).
fn noisy_circle_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let theta = std::f64::consts::TAU * (row as f64) / (n as f64);
        z[[row, 0]] = theta.cos();
        z[[row, 1]] = theta.sin();
        for col in 0..p {
            z[[row, col]] += sigma * deterministic_circle_noise(row, col + 11);
        }
    }
    z
}

/// Build a fresh K=1 IBP-MAP-gated periodic (circle) atom on `target`, decoder
/// cold at zero, `m` harmonics, identity smoothness penalty, PCA-seeded coords,
/// and per-row logits high enough that the gate saturates to its `π_0 ≈ 0.5`
/// IBP cap (`σ(6) ≈ 0.9975`). `quotient_on` toggles the amplitude scale-gauge pin.
fn ibp_circle_k1_term(target: ArrayView2<'_, f64>, m: usize, quotient_on: bool) -> SaeManifoldTerm {
    let n = target.nrows();
    let p = target.ncols();
    let d = 1usize;
    let k = 1usize;
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let dims = vec![d; k];
    let seed = sae_pca_seed_initial_coords(target, &basis_kinds, &dims).unwrap();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());

    let mut basis_values = Array3::<f64>::zeros((k, n, m));
    let mut basis_jacobian = Array4::<f64>::zeros((k, n, m, d));
    let decoder = Array3::<f64>::zeros((k, m, p));
    let mut penalties = Array3::<f64>::zeros((k, m, m));
    let mut coords_vec: Vec<Array2<f64>> = Vec::new();
    for atom in 0..k {
        let coords = seed.slice(s![atom, .., 0..d]).to_owned();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        basis_values.slice_mut(s![atom, .., ..]).assign(&phi);
        basis_jacobian.slice_mut(s![atom, .., .., ..]).assign(&jet);
        penalties
            .slice_mut(s![atom, .., ..])
            .assign(&Array2::<f64>::eye(m));
        coords_vec.push(coords);
    }
    // Strong per-row logits: the gate saturates to its IBP cap a_0 = σ(6)·π_0 ≈ 0.5
    // (α = 1 ⇒ π_0 = 0.5), the regime where the ungated λ_smooth/a_0² ≈ 4·λ_smooth
    // over-shrink bites.
    let logits = Array2::<f64>::from_elem((n, k), 6.0);
    let mut evaluators: Vec<Option<Arc<dyn SaeBasisSecondJet>>> = Vec::new();
    for _ in 0..k {
        evaluators.push(Some(evaluator.clone()));
    }
    let mut term = term_from_padded_blocks_with_mode(
        n,
        p,
        &basis_kinds,
        basis_values.view(),
        basis_jacobian.view(),
        &vec![m; k],
        &dims,
        decoder.view(),
        penalties.view(),
        logits.view(),
        &coords_vec,
        AssignmentMode::ibp_map(1.0, 1.0, false),
        &evaluators,
    )
    .unwrap();
    term.set_quotient_scale(quotient_on);
    term
}

/// #2228 recovery gate — the definitive proof the scale-gauge pin cures the
/// gate-decoder co-collapse at a REML-selected (material) smoothing level.
#[test]
fn zz_ibp_k1_recovery_gate_pin_rescues_material_smoothing_2228() {
    let n = 120usize;
    let p = 6usize;
    let m = 11usize; // [1, sin θ, cos θ, …, sin 5θ, cos 5θ] — 8 non-fundamental harmonics carry only noise.
    let sigma = 0.10;
    let target = noisy_circle_target(n, p, sigma);

    // --- (0) REML model selection over the single smoothing scalar, run under the
    // production (pin-ON) fit. This is a 1-D REML line search — exactly what the
    // outer optimizer does for this coordinate — over a log-λ grid spanning the
    // negligible floor (-6) to strong smoothing (+3). The argmin is λ*, the level
    // REML operates at; its evidence at -6 is the "tiny" contrast.
    let grid = [-6.0_f64, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let mut crits = Vec::with_capacity(grid.len());
    for &log_lambda in grid.iter() {
        let mut term = ibp_circle_k1_term(target.view(), m, true);
        let rho = SaeManifoldRho::new(0.0, log_lambda, vec![Array1::<f64>::zeros(1)]);
        let (crit, _loss) = term
            .reml_criterion(target.view(), &rho, None, 60, 0.05, 1.0e-3, 1.0e-3)
            .unwrap_or_else(|e| panic!("REML criterion must evaluate under the pin at log λ={log_lambda}: {e}"));
        eprintln!("[#2228 recovery] REML criterion @ log λ_smooth={log_lambda:+.1} -> {crit:.6}");
        crits.push(crit);
    }
    let (argmin, &crit_star) = crits
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let log_lambda_star = grid[argmin];
    let crit_tiny = crits[0]; // log λ = -6, the negligible floor.
    eprintln!(
        "[#2228 recovery] REML SELECTS log λ*={log_lambda_star:+.1} (crit {crit_star:.6}); \
         tiny-floor log λ=-6 crit {crit_tiny:.6}"
    );

    // REML must genuinely PREFER material smoothing to the negligible floor, and the
    // selected λ* must sit well above the tiny regime the other fixtures pin. If it
    // did not, the recovery contrast below would be a cherry-picked λ, not the
    // operating point.
    assert!(
        crit_star < crit_tiny,
        "#2228: REML must prefer a material smoothing over the negligible floor: \
         crit(λ*)={crit_star:.6} vs crit(log λ=-6)={crit_tiny:.6} — the co-collapse regime \
         is only honest if REML actually selects it"
    );
    assert!(
        log_lambda_star > -4.0,
        "#2228: REML-selected smoothing log λ*={log_lambda_star} is at the negligible floor; \
         the recovery gate needs a MATERIAL λ (the level real noisy activations select), \
         not the tiny λ the do-no-harm fixtures use"
    );

    // --- (1) WITHOUT the pin, at the REML-selected λ*: the ungated λ_smooth/a_0²≈4λ*
    // over-shrink co-collapses the gated decoder.
    let mut off = ibp_circle_k1_term(target.view(), m, false);
    let mut rho_off = SaeManifoldRho::new(0.0, log_lambda_star, vec![Array1::<f64>::zeros(1)]);
    let loss_off = off
        .run_joint_fit_arrow_schur(target.view(), &mut rho_off, None, 60, 0.05, 1.0e-3, 1.0e-3)
        .expect("OFF joint fit must return (a co-collapsed but finite fit)");
    assert!(loss_off.total().is_finite(), "OFF loss must stay finite");
    let ev_off = off
        .dictionary_reconstruction_ev(target.view(), &rho_off)
        .unwrap();

    // --- (2) WITH the pin, at the SAME λ*: the peel frees the magnitude and the
    // circle is recovered.
    let mut on = ibp_circle_k1_term(target.view(), m, true);
    let mut rho_on = SaeManifoldRho::new(0.0, log_lambda_star, vec![Array1::<f64>::zeros(1)]);
    let loss_on = on
        .run_joint_fit_arrow_schur(target.view(), &mut rho_on, None, 60, 0.05, 1.0e-3, 1.0e-3)
        .expect("ON joint fit must converge");
    assert!(loss_on.total().is_finite(), "ON loss must stay finite");
    let ev_on = on
        .dictionary_reconstruction_ev(target.view(), &rho_on)
        .unwrap();

    let s_off = off.atoms[0].log_amplitude;
    let s_on = on.atoms[0].log_amplitude;
    let bnorm_off: f64 = off.atoms[0]
        .decoder_coefficients
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let bnorm_on: f64 = on.atoms[0]
        .decoder_coefficients
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    eprintln!(
        "[#2228 recovery] @ log λ*={log_lambda_star:+.1}: EV_off={ev_off:.4} (s={s_off:.3}, ‖B‖={bnorm_off:.3}) \
         | EV_on={ev_on:.4} (s={s_on:.3}, ‖B‖={bnorm_on:.3})"
    );

    assert!(
        ev_off < 0.5,
        "#2228: WITHOUT the pin the IBP-gated decoder must co-collapse at the REML-selected \
         smoothing (EV_off={ev_off:.4}, expected < 0.5 — the ungated λ_smooth/a_0²≈4λ over-shrink)"
    );
    assert!(
        ev_on > 0.9,
        "#2228: WITH the pin the SAME fit must recover the planted circle (EV_on={ev_on:.4}, \
         expected > 0.9 — the peel homes ‖B‖ into the unpenalized s_k so the gate cap no longer \
         over-shrinks the magnitude)"
    );
}
