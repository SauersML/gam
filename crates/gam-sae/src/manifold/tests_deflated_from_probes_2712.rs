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

use super::tests::small_two_atom_periodic_term;
use super::*;

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

