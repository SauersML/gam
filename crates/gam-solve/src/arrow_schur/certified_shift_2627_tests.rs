//! #2627 — the proximal rungs an arrow system's declared bounds certify.
//!
//! `ArrowShiftCertificate` reads Gershgorin edges of every `H_tt^(i)`, the declared
//! cross-block norms and the shared block's majorant, and certifies a rung when every
//! factorization guard provably passes there. The pins:
//!
//! * every certified rung of a randomized indefinite system is positive definite by
//!   a dense eigendecomposition of the shifted joint Hessian, the production Direct
//!   solve factors it, and the curvature floor lies below its smallest eigenvalue;
//!   some rungs are refused, so the check is not vacuous;
//! * the coupling term `Σ_i s_i²/(lo_i − e_i)` is what keeps the certificate sound: a
//!   rung where the row guards and `ρ_β − N_β > 0` hold while the joint Hessian is
//!   indefinite is refused, and the next decade is certified;
//! * a matrix-free cross block declares the same bounds as its dense twin, so the
//!   certificate is bit-identical, and an operator installed without a declaration is
//!   refused for that reason.

#![cfg(test)]

use super::*;
use crate::arrow_schur::certified_shift::ArrowShiftCertificate;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::roundoff::symmetric_spectrum_rounding_band;

/// Deterministic entries in `[-1, 1)`.
fn uniform_entries(seed: u64, count: usize) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407 ^ index as u64);
        entries.push(((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0);
    }
    entries
}

/// `n` rows of dimension `d` and a border of `k`, with symmetric indefinite `H_tt`
/// and `H_ββ` and a cross block twice their scale, so small rungs refuse.
fn indefinite_arrow_system(seed: u64, n: usize, d: usize, k: usize) -> ArrowSchurSystem {
    let mut entries = uniform_entries(seed, n * (d * d + d * k + d) + k * k + k).into_iter();
    let mut next = move || entries.next().expect("enough fixture entries");
    let mut sys = ArrowSchurSystem::new(n, d, k);
    for row in sys.rows.iter_mut() {
        for a in 0..d {
            for b in a..d {
                let value = next();
                row.htt[[a, b]] = value;
                row.htt[[b, a]] = value;
            }
        }
        for a in 0..d {
            for c in 0..k {
                row.htbeta[[a, c]] = 2.0 * next();
            }
        }
        for a in 0..d {
            row.gt[a] = next();
        }
    }
    for a in 0..k {
        for b in a..k {
            let value = next();
            sys.hbb[[a, b]] = value;
            sys.hbb[[b, a]] = value;
        }
    }
    for a in 0..k {
        sys.gb[a] = next();
    }
    sys
}

/// The smallest eigenvalue of the dense joint Hessian with `ridge` on both blocks, and
/// the eigensolver's rounding band.
fn shifted_spectrum_floor(sys: &ArrowSchurSystem, ridge: f64) -> (f64, f64) {
    let latent: usize = sys.rows.iter().map(|row| row.htt.nrows()).sum();
    let size = latent + sys.k;
    let mut dense = Array2::<f64>::zeros((size, size));
    let mut offset = 0usize;
    for row in &sys.rows {
        let d = row.htt.nrows();
        for a in 0..d {
            for b in 0..d {
                dense[[offset + a, offset + b]] = row.htt[[a, b]];
            }
            dense[[offset + a, offset + a]] += ridge;
            for c in 0..sys.k {
                dense[[offset + a, latent + c]] = row.htbeta[[a, c]];
                dense[[latent + c, offset + a]] = row.htbeta[[a, c]];
            }
        }
        offset += d;
    }
    for a in 0..sys.k {
        for b in 0..sys.k {
            dense[[latent + a, latent + b]] = sys.hbb[[a, b]];
        }
        dense[[latent + a, latent + a]] += ridge;
    }
    let eigenvalues = dense.eigh(Side::Lower).expect("joint Hessian EVD").0;
    let floor = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    (
        floor,
        symmetric_spectrum_rounding_band(eigenvalues.as_slice().expect("contiguous")),
    )
}

/// Rung `0` and the decades `1e-8 … 1e16`.
fn rungs() -> Vec<f64> {
    std::iter::once(0.0)
        .chain((0..=24).map(|decade| 1.0e-8 * 10.0_f64.powi(decade)))
        .collect()
}

#[test]
fn a_certified_rung_is_positive_definite_and_factors_2627() {
    let options = ArrowSolveOptions::direct();
    let mut certified = 0usize;
    let mut refused = 0usize;
    for seed in 0..4u64 {
        let sys = indefinite_arrow_system(seed, 5, 2, 3);
        let certificate =
            ArrowShiftCertificate::from_system(&sys, 0.0, 0.0).expect("a finite fixture");
        for ridge in rungs() {
            if !certificate.certifies_factorable(ridge, true) {
                refused += 1;
                continue;
            }
            certified += 1;
            let (floor, band) = shifted_spectrum_floor(&sys, ridge);
            assert!(
                floor > band,
                "seed {seed}, rung {ridge:e}: certified, but the shifted joint Hessian has \
                 λ_min {floor:e} within its band {band:e}"
            );
            let step = solve_arrow_newton_step_core(&sys, ridge, ridge, &options);
            assert!(
                step.is_ok(),
                "seed {seed}, rung {ridge:e}: certified, but the Direct solve refused: {:?}",
                step.err()
            );
            let curvature = certificate
                .damped_curvature_lower_bound(ridge)
                .expect("a certified rung certifies positive curvature");
            assert!(
                curvature <= floor + band,
                "seed {seed}, rung {ridge:e}: the curvature floor {curvature:e} exceeds λ_min \
                 {floor:e} (band {band:e})"
            );
        }
    }
    assert!(
        certified > 0 && refused > 0,
        "both arms must fire: {certified} certified and {refused} refused rungs"
    );
}

/// `H = [[1, 10], [10, −1]]`: one latent row, one border column. At `μ = 10` the row
/// guard holds (`lo = 11`) and `ρ_β − N_β = 9 > 0`, but `λ_min = 10 − √101 < 0`; only
/// the coupling term `s²/lo ≈ 9.09` refuses it. At `μ = 100`, `λ_min = 100 − √101`.
#[test]
fn the_coupling_term_refuses_an_indefinite_rung_the_block_bounds_admit_2627() {
    let mut sys = ArrowSchurSystem::new(1, 1, 1);
    sys.rows[0].htt[[0, 0]] = 1.0;
    sys.rows[0].htbeta[[0, 0]] = 10.0;
    sys.hbb[[0, 0]] = -1.0;
    let certificate = ArrowShiftCertificate::from_system(&sys, 0.0, 0.0).expect("finite fixture");

    let (indefinite_floor, indefinite_band) = shifted_spectrum_floor(&sys, 10.0);
    assert!(
        indefinite_floor < -indefinite_band,
        "positive control: the joint Hessian at μ = 10 must be indefinite, got λ_min \
         {indefinite_floor:e}"
    );
    let mut border_row_sum = [0.0_f64];
    sys.shared_block_abs_majorant_matvec(&[1.0], &mut border_row_sum);
    assert!(
        10.0 - border_row_sum[0] > 0.0,
        "premise: the border bound alone, ρ_β − N_β = {:e}, admits μ = 10",
        10.0 - border_row_sum[0]
    );
    assert!(
        !certificate.certifies_factorable(10.0, true),
        "the certificate must refuse the indefinite rung μ = 10"
    );

    assert!(
        certificate.certifies_factorable(100.0, true),
        "the certificate must admit μ = 100"
    );
    let (definite_floor, definite_band) = shifted_spectrum_floor(&sys, 100.0);
    assert!(
        definite_floor > definite_band,
        "the certified rung μ = 100 must be positive definite, got λ_min {definite_floor:e} \
         (band {definite_band:e})"
    );
    let curvature = certificate
        .damped_curvature_lower_bound(100.0)
        .expect("μ = 100 certifies positive curvature");
    assert!(
        curvature > 0.0 && curvature <= definite_floor + definite_band,
        "the curvature floor {curvature:e} must lie in (0, λ_min = {definite_floor:e}]"
    );
}

#[test]
fn a_declared_matrix_free_cross_block_certifies_like_its_dense_twin_2627() {
    let dense = indefinite_arrow_system(7, 4, 2, 3);
    let slabs: Vec<Array2<f64>> = dense.rows.iter().map(|row| row.htbeta.clone()).collect();
    let row_norm_bounds: Arc<[f64]> = slabs
        .iter()
        .map(|slab| frobenius_norm_upper_bound(slab.iter().copied()))
        .collect();
    let forward_slabs = slabs.clone();
    let transpose_slabs = slabs;
    let mut matrix_free = dense.clone();
    matrix_free.set_row_htbeta_operator(
        move |row: usize, x: ArrayView1<'_, f64>, out: &mut Array1<f64>| {
            out.assign(&forward_slabs[row].dot(&x));
        },
        move |row: usize, v: ArrayView1<'_, f64>, out: &mut Array1<f64>| {
            *out += &transpose_slabs[row].t().dot(&v);
        },
        // The forward accumulates `k = 3` terms per coordinate and the transpose `d = 2`,
        // one more for the addition into `out`.
        RowHtbetaDeclaration {
            row_norm_bounds,
            apply_depth: 4,
        },
    );
    let dense_certificate =
        ArrowShiftCertificate::from_system(&dense, 0.0, 0.0).expect("dense fixture");
    let matrix_free_certificate =
        ArrowShiftCertificate::from_system(&matrix_free, 0.0, 0.0).expect("declared operator");
    let mut certified = 0usize;
    for ridge in rungs() {
        let dense_verdict = dense_certificate.certifies_factorable(ridge, false);
        assert_eq!(
            dense_verdict,
            matrix_free_certificate.certifies_factorable(ridge, false),
            "rung {ridge:e}: the declared operator and its dense twin must certify alike"
        );
        certified += usize::from(dense_verdict);
        assert_eq!(
            dense_certificate
                .damped_curvature_lower_bound(ridge)
                .map(f64::to_bits),
            matrix_free_certificate
                .damped_curvature_lower_bound(ridge)
                .map(f64::to_bits),
            "rung {ridge:e}: the two curvature floors must be bit-identical"
        );
    }
    assert!(certified > 0, "the twin comparison is vacuous unless some rung certifies");

    let mut undeclared = matrix_free.clone();
    undeclared.htbeta_declaration = None;
    let refusal = ArrowShiftCertificate::from_system(&undeclared, 0.0, 0.0)
        .expect_err("an operator without a declaration must be refused");
    assert!(
        refusal.contains("without its declaration"),
        "the refusal must name the missing declaration, got: {refusal}"
    );
}

/// A border of one coordinate with `H_ββ = −4e15`, `g = 1`, `f = 1`. At `μ = 4.6e15`
/// the damped curvature is `6e14`, so the model still promises `1/(2·6e14) ≈ 8e-16`,
/// above half the float spacing at 1 (`1.1e-16`), while `‖g‖²/μ` would already
/// declare it unrepresentable. At `μ = 1e16` the curvature is `6e15` and the promise
/// `8e-17` is below it.
#[test]
fn the_armijo_stop_reads_certified_curvature_not_the_ridge_2627() {
    let mut sys = ArrowSchurSystem::new(0, 1, 1);
    sys.hbb[[0, 0]] = -4.0e15;
    sys.gb[0] = 1.0;
    let certificate = ArrowShiftCertificate::from_system(&sys, 0.0, 0.0).expect("finite fixture");
    let spacing = 1.0_f64.next_up() - 1.0;

    let probe = 4.6e15;
    assert!(
        1.0 / probe < spacing,
        "premise: the ridge-only mutant ‖g‖²/μ = {:e} declares the decrease unrepresentable",
        1.0 / probe
    );
    let curvature = certificate
        .damped_curvature_lower_bound(probe)
        .expect("μ = 4.6e15 certifies positive curvature");
    assert!(
        curvature <= probe - 4.0e15,
        "the curvature floor {curvature:e} must not exceed λ_min = {:e}",
        probe - 4.0e15
    );
    assert!(
        !certificate.promises_unrepresentable_decrease(probe, 1.0),
        "at μ = 4.6e15 the model still promises a representable decrease"
    );
    assert!(
        certificate.promises_unrepresentable_decrease(1.0e16, 1.0),
        "at μ = 1e16 the model promises less than half a float spacing"
    );
    assert!(
        !certificate.promises_unrepresentable_decrease(1.0e15, 1.0),
        "below the border deficit no curvature is certified, so the stop cannot fire"
    );
}

/// One latent row and one border coordinate whose shared block is `−5·s`, with no
/// coupling. The border factors once the proximal ridge passes `5·s`: at `μ = 10` for
/// `s = 1` (rung 10) and at `μ = 1e15` for `s = 1e14` (rung 24). A fixed count of 22
/// rungs reaches `1e13` and refuses the scaled system; the structural ladder reaches
/// it, one rung per decade the scale moved.
#[test]
fn the_ladder_reaches_a_border_deficit_past_any_fixed_count_2627() {
    let options = ArrowSolveOptions::direct();
    let escalations = |scale: f64| -> Result<usize, String> {
        let mut sys = ArrowSchurSystem::new(1, 1, 1);
        sys.rows[0].htt[[0, 0]] = 1.0;
        sys.rows[0].gt[0] = 0.5;
        sys.hbb[[0, 0]] = -5.0 * scale;
        sys.gb[0] = 1.0;
        let (delta_t, delta_beta, diagnostics) =
            solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
                .map_err(|error| format!("scale {scale:e}: {error}"))?;
        if delta_t.iter().chain(delta_beta.iter()).all(|value| value.is_finite()) {
            Ok(diagnostics.ridge_escalations)
        } else {
            Err(format!("scale {scale:e}: the accepted step is not finite"))
        }
    };
    let base = escalations(1.0);
    assert_eq!(base, Ok(10), "the unscaled deficit clears at μ = 10, rung 10");
    let scaled = escalations(1.0e14);
    assert_eq!(scaled, Ok(24), "the deficit scaled by 1e14 clears at μ = 1e15, rung 24");
}

/// A positive definite system whose explicit InexactPCG request is priced at zero
/// products: its dense route costs less than one reduced-Schur product, and a request
/// that asked for InexactPCG refuses a miss instead of handing it to Direct (#2900 row
/// 6.15). So every rung refuses with `PcgBudgetExhausted`. The ladder must stop at the
/// first rung the declared bounds certify, carry that refusal as the cause, and climb
/// no further; a fixed count refused after 22 rungs with a bare PCG refusal.
#[test]
fn a_refusal_at_a_certified_rung_is_typed_and_ends_the_ladder_2627() {
    let n = 2;
    let k = 1;
    let mut sys = ArrowSchurSystem::new(n, 1, k);
    let entries = uniform_entries(11, n * k);
    for (row_index, row) in sys.rows.iter_mut().enumerate() {
        row.htt[[0, 0]] = 2.0;
        row.gt[0] = 1.0;
        for c in 0..k {
            row.htbeta[[0, c]] = 0.05 * entries[row_index * k + c];
        }
    }
    for a in 0..k {
        sys.hbb[[a, a]] = 3.0;
        sys.gb[a] = 1.0;
    }
    let mut options = ArrowSolveOptions::inexact_pcg();
    options.gpu_matvec = None;
    let budget = resolve_arrow_route(&sys, &options).pcg_budget;
    assert_eq!(
        budget,
        Some(ArrowPcgBudget::dense_route_priced(0, false)),
        "premise: the explicit InexactPCG request is priced at zero products, and a miss \
         refuses instead of falling back to Direct"
    );

    let certificate = ArrowShiftCertificate::from_system(&sys, 0.0, 0.0).expect("finite fixture");
    let mut certified_ridge = 0.0_f64;
    while !certificate.certifies_factorable(certified_ridge, false) {
        certified_ridge = next_proximal_ridge(certified_ridge, DEFAULT_PROXIMAL_RIDGE_GROWTH);
    }
    assert!(
        certified_ridge > 0.0,
        "premise: the border bound refuses rung 0, so the ladder must climb before it stops"
    );

    let refusal = solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
        .expect_err("a PCG priced at zero products refuses at every rung");
    let ArrowSchurError::RefusedAtCertifiedShift {
        proximal_ridge,
        cause,
    } = refusal
    else {
        panic!("the ladder must end with RefusedAtCertifiedShift, got {refusal:?}");
    };
    assert_eq!(
        proximal_ridge.to_bits(),
        certified_ridge.to_bits(),
        "the ladder must stop at the first certified rung {certified_ridge:e}, got {proximal_ridge:e}"
    );
    assert!(
        matches!(
            *cause,
            ArrowSchurError::PcgBudgetExhausted {
                products_spent: 0,
                ..
            }
        ),
        "the cause must be the budget refusal, got {cause:?}"
    );
}

/// A scalar row `H = 2`, `g = 1`, `f = 1`, whose trial objective rises by at least two
/// float spacings at every rung, so no rung is accepted. The damped model promises
/// `1/(2(2 + μ))`, above half the spacing at 1 through `μ = 1e15` and below it at
/// `μ = 1e16`, so the correction tries the rungs `1e-8 … 1e15` (24 attempts) and then
/// converges in place; a fixed count stopped after 22.
#[test]
fn the_proximal_correction_stops_where_no_decrease_is_representable_2627() {
    let mut sys = ArrowSchurSystem::new(1, 1, 0);
    sys.rows[0].htt[[0, 0]] = 2.0;
    sys.rows[0].gt[0] = 1.0;
    let accepted = solve_arrow_newton_step_with_proximal_correction(
        &sys,
        0.0,
        0.0,
        1.0,
        &ArrowSolveOptions::direct(),
        &ArrowProximalCorrectionOptions::default(),
        |delta_t, _| 1.0 + 2.0 * f64::EPSILON * (1.0 + delta_t[0].abs()),
    )
    .expect("a rise within the objective resolution converges in place");
    assert_eq!(
        accepted.attempts, 24,
        "rungs 1e-8 … 1e15 promise a representable decrease and 1e16 does not"
    );
    assert!(
        accepted.delta_t.iter().all(|&value| value == 0.0),
        "no rung decreased the objective, so the correction converges in place"
    );
    assert_eq!(accepted.trial_objective_value, 1.0);
}
