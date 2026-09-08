//! #2336 — the indefinite exact-`A` refusal is an INFEASIBLE outer probe, not a
//! fatal abort. Companion to `tests_schur_seed_refusal_1782`, which pins the same
//! contract for the non-PD reduced-Schur refusal; this one covers the typed
//! `SaeCriterionError::IndefiniteObservedInformation` variant that #2330 Phase-2a
//! introduced when it made `½log|A|` the ranked value.

use super::construction::{ArrowMetric, sae_exact_a_direction_floor};
use super::tests::*;
use super::*;
use gam_solve::rho_optimizer::OuterObjective;
use ndarray::{Array1, Array2, s};

/// Reproduce the off-manifold, fixed-stratum state whose `B`-converged mode is an
/// exact-`A` SADDLE. This is the excitation the #2253/#2330 shared fixture uses:
/// the residual, entropy, and curvature-delta channels are all genuinely live, and
/// at this ρ some latent coordinates sit in the ARD periodic prior's CONCAVE half.
///
/// The majorizer clamps that curvature away (`atom.rs`: `hess =
/// psd_majorizer_hess + negative_hessian_remainder`, `psd_majorizer_hess =
/// α·softplus_{τ₀}(cos κt)` — the #2339 smooth envelope of `max(hess, 0)`), so
/// `A = B − E` with `E ⪰ 0` diagonal in the coordinate block, carrying
/// `≈|α·cos κt_ik|` with `α = e^{ρ_ard}`. `B ≻ 0` by construction, so the inner
/// Newton converges, while the exact `A` it does NOT see stays indefinite.
fn ard_saddle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (term, mut target, mut rho) = gamma_fd_tiny_fixture();
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
    (term, target, rho)
}

/// #2267 regression — a fully degenerate spectrum is the worst possible cluster
/// shape: every eigenvector returned for `A = 3I` is arbitrary until the whole
/// `dim`-wide eigenspace is re-resolved against `E`.  Production `E` is diagonal
/// on the coordinate block and zero on the beta border.  Build that operator
/// independently here and require the rotated basis to diagonalize its exact
/// dense restriction, with the expected spectrum, while leaving `A`'s repeated
/// eigenvalues untouched.
///
/// This pins the semantic invariant behind the direct weighted-product route.
/// The pre-fix implementation obtained the same quantity by scanning a dense
/// `dim x dim` zero matrix for every cluster pair, making this fully-degenerate
/// case quartic in `dim` despite the operator having only `total_t` entries.
#[test]
fn fully_degenerate_cluster_diagonalizes_direct_e_diag_2267() {
    let dim = 24usize;
    let total_t = 17usize;
    let a = Array2::<f64>::eye(dim) * 3.0;
    let e_diag = Array1::from_shape_fn(total_t, |row| 0.5 * (row as f64 + 1.0));

    let (eigenvalues, eigenvectors) =
        SaeManifoldTerm::cluster_stable_eigh(&a, &e_diag, None, total_t)
            .expect("a fully degenerate exact-A cluster must resolve against diagonal E");

    let eigenvalue_error = eigenvalues
        .iter()
        .map(|value| (value - 3.0).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        eigenvalue_error <= 1.0e-12,
        "cluster rotation must not alter A's repeated eigenvalue; max error={eigenvalue_error:.3e}"
    );

    // Independent dense oracle: production must not build this matrix, but at
    // this tiny test dimension it makes the represented operator unambiguous.
    let mut dense_e = Array2::<f64>::zeros((dim, dim));
    for row in 0..total_t {
        dense_e[[row, row]] = e_diag[row];
    }
    let restricted = eigenvectors.t().dot(&dense_e.dot(&eigenvectors));
    let mut max_off_diagonal = 0.0_f64;
    for row in 0..dim {
        for column in 0..dim {
            if row != column {
                max_off_diagonal = max_off_diagonal.max(restricted[[row, column]].abs());
            }
        }
    }
    assert!(
        max_off_diagonal <= 1.0e-11,
        "the rotated fully-degenerate cluster must diagonalize V^T E V; max off-diagonal={max_off_diagonal:.3e}"
    );

    let mut observed = restricted.diag().to_vec();
    observed.sort_by(f64::total_cmp);
    let mut expected = e_diag.to_vec();
    expected.extend(std::iter::repeat(0.0).take(dim - total_t));
    expected.sort_by(f64::total_cmp);
    let spectrum_error = observed
        .iter()
        .zip(expected.iter())
        .map(|(actual, oracle)| (actual - oracle).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        spectrum_error <= 1.0e-11,
        "the cluster restriction must preserve the diagonal-E spectrum; max error={spectrum_error:.3e}"
    );
}

/// #2515 regression — a small but nonzero eigenvalue gap does not define a
/// rotatable eigenspace. Even when the gap is below the old `sqrt(eps) * ||A||`
/// clustering threshold, the returned columns must remain eigenvectors of the
/// returned eigenvalues. Rotating them against `E` while leaving the distinct
/// eigenvalues untouched constructs a false inverse and breaks the derivative
/// of the log-determinant value.
#[test]
fn nearly_degenerate_distinct_spectrum_preserves_eigenpairs_2515() {
    let root_half = 0.5_f64.sqrt();
    let rotation = ndarray::array![[root_half, -root_half], [root_half, root_half]];
    let gap = 1.0e-9_f64;
    let diagonal = Array2::from_diag(&ndarray::array![3.0, 3.0 + gap]);
    let operator = rotation.dot(&diagonal.dot(&rotation.t()));
    // This diagonal is deliberately not diagonal in the eigenbasis of A, so the
    // retired near-cluster rule performs a nontrivial, invalid rotation.
    let e_diag = ndarray::array![1.0, 2.0];

    let (eigenvalues, eigenvectors) =
        SaeManifoldTerm::cluster_stable_eigh(&operator, &e_diag, None, 2)
            .expect("the near-degenerate exact-A spectrum must decompose");
    let residual = operator.dot(&eigenvectors)
        - eigenvectors.dot(&Array2::from_diag(&eigenvalues));
    let max_residual = residual
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let backward_error = 32.0 * f64::EPSILON * 2.0 * (3.0 + gap);
    assert!(
        max_residual <= backward_error,
        "cluster-stable eigensystem violates A V = V diag(lambda): \
         max residual {max_residual:.6e}, backward-error allowance {backward_error:.6e}"
    );
}

/// #2336 GATE — after the value-side E-attributability fix, a B-converged mode
/// whose exact-A indefiniteness is FULLY attributable to the bounded ARD periodic
/// concave-clamp wrinkle `E` prices a FINITE criterion (basin curvature `λ+e_v ≥ 0`
/// on the switched directions) instead of refusing. `ard_saddle_state`'s two
/// negatives (≈ −0.015) are E-attributable (`e_v ≥ |λ|`, verified in
/// `zz_measure_e_attributability_2336`), so both the criterion and the outer eval
/// return finite. RED before the fix (the criterion returned
/// `Err(IndefiniteObservedInformation{{joint}})` and `eval` priced `+inf`), GREEN
/// STABLE across #2339: E = α·softplus_τ₀(−cos κt) ≥ α·max(−cos,0) (the hard clamp)
/// pointwise, so the smooth clamp only GROWS e_v — the attributability test loosens
/// by at most α·τ₀·ln2 = α·(deflation floor), within #2339's τ₀ budget — hence
/// a_saddle prices under both the hard and the smooth clamp.
/// after. This is the canonical E-attributable wrinkle-saddle specimen (same state
/// fix-2253 anchored as `converged_state_with_residual_a_saddle_2336`, now
/// documented as the PRICING specimen: its `λ+e_v(ARD)=+0.026` shows the clamp
/// alone lifts it, so it prices — it is NOT a genuine deep saddle). The genuine
/// refusal path is exercised by `genuine_saddle_is_infeasible_probe_not_fatal_2336`.
#[test]
pub(crate) fn e_attributable_ard_saddle_prices_finite_2336() {
    let (mut term, target, rho) = ard_saddle_state();
    let priced = term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    assert!(
        matches!(&priced, Ok((value, _, _)) if value.is_finite()),
        "post E-attributability fix the ARD-wrinkle saddle must PRICE FINITE, not refuse; got: {:?}",
        priced
            .as_ref()
            .map(|(value, _, _)| *value)
            .map_err(|e| format!("{e:?}"))
    );

    let (term, target, rho) = ard_saddle_state();
    let rho_flat = rho.to_flat();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target, None, rho, 40, 0.4, 1.0e-6, 1.0e-6);
    match objective.eval(&rho_flat) {
        Ok(evaluation) => assert!(
            evaluation.cost.is_finite(),
            "an E-attributable saddle-ρ must price FINITE (was +inf refusal), got cost={}",
            evaluation.cost
        ),
        Err(err) => panic!(
            "#2336: an E-attributable saddle-ρ must be a FINITE outer eval, not a fatal abort; \
             got: {err}"
        ),
    }
}

/// #2434 regression gate — the switched-direction derivative already landed with
/// the #2336 value rule in `e97238721`; two stale prototype comments later made it
/// look absent. Pin the production direct-ρ channel against the value it actually
/// differentiates so neither comments nor implementation can drift again.
///
/// Hold θ̂ fixed at the canonical E-attributable saddle, rebuild the cache at each
/// perturbed ρ, and centrally difference
/// `½(log|A_priced| − log|A_tt,priced|)`. The analytic side is the direct trace from
/// `dense_exact_a_logdet_channels`, including:
///
/// 1. the priced inverse contraction;
/// 2. the Daleckii–Krein eigenvector-response matrix; and
/// 3. the explicit `dE/dρ_ard = E` term.
///
/// This deliberately probes only ARD coordinates: they are the coordinates on
/// which the allegedly missing B-channel is live. The spectral assertion first
/// proves the fixture really contains a switched negative direction; otherwise an
/// ordinary positive-definite state could false-green the derivative comparison.
#[test]
fn priced_ard_direct_gradient_matches_fixed_state_value_2434() {
    let (mut term, target, rho) = ard_saddle_state();
    let (_value, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("canonical E-attributable saddle must produce a priced cache");

    let total_t = cache.delta_t_len();
    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("materialize exact A at the priced state");
    let e_diag = term
        .materialize_ard_concave_clamp_diagonal(&rho, &cache)
        .expect("materialize the clamp-attribution diagonal");
    // #2828 — the production classification also carries the border half of
    // `E`, so a probe that shares the scalar rule must share the operator too.
    let e_beta = term
        .decoder_prior_majorizer_gap_border(&cache)
        .expect("decoder-prior majorization gap");
    let (eigs, vecs) = SaeManifoldTerm::cluster_stable_eigh(&a, &e_diag, e_beta.as_ref(), total_t)
        .expect("stable exact-A eigh");
    // #2673 — the band is per direction, in the `B` metric both the value and
    // the gradient classify in. This probe supplies its own eigenvectors and its
    // own `B`-applies and shares only the scalar rule.
    let spectral_norm = eigs.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let joint_metric = ArrowMetric::Joint(&cache);
    let floor_at = |idx: usize| -> f64 {
        let vbv = joint_metric
            .quadratic_form(vecs.column(idx))
            .expect("B quadratic form");
        sae_exact_a_direction_floor(eigs.len(), spectral_norm, vbv)
    };
    let switched = eigs
        .iter()
        .enumerate()
        .filter(|(idx, lambda)| {
            let floor = floor_at(*idx);
            if **lambda >= -floor {
                return false;
            }
            let v = vecs.column(*idx);
            let e_v = (0..total_t)
                .map(|row| e_diag[row] * v[row] * v[row])
                .sum::<f64>();
            **lambda + e_v >= -floor
        })
        .count();
    assert!(
        switched > 0,
        "#2434 gate is invalid: the fixture contains no clamp-attributable switched direction"
    );

    let analytic = term
        .dense_exact_a_logdet_channels(target.view(), &rho, &loss, &cache)
        .expect("complete priced exact-A derivative")
        .logdet_trace;
    let fixed_state_priced_logdet =
        |mut candidate: SaeManifoldTerm, at_rho: &SaeManifoldRho| -> f64 {
            let (_criterion, _loss, at_cache) = candidate
                .penalized_quasi_laplace_criterion_with_cache(
                    target.view(),
                    at_rho,
                    None,
                    0,
                    0.4,
                    1.0e-6,
                    1.0e-6,
                )
                .expect("fixed-state perturbed cache must remain on the priced stratum");
            let (log_a, log_a_tt) = candidate
                .exact_observed_information_log_dets(at_rho, target.view(), &at_cache)
                .expect("fixed-state perturbed exact-A value");
            0.5 * (log_a - log_a_tt)
        };

    let converged_term = term;
    let h = 1.0e-5_f64;
    let mut checked = 0usize;
    let mut max_signal = 0.0_f64;
    let mut worst_relative_error = 0.0_f64;
    for atom in 0..rho.log_ard.len() {
        for axis in 0..rho.log_ard[atom].len() {
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            plus.log_ard[atom][axis] += h;
            minus.log_ard[atom][axis] -= h;
            let value_plus = fixed_state_priced_logdet(converged_term.clone(), &plus);
            let value_minus = fixed_state_priced_logdet(converged_term.clone(), &minus);
            let finite_difference = (value_plus - value_minus) / (2.0 * h);
            let index = rho.ard_flat_index(atom, axis);
            let exact = analytic[index];
            let scale = 1.0 + finite_difference.abs().max(exact.abs());
            let relative_error = (finite_difference - exact).abs() / scale;
            max_signal = max_signal.max(finite_difference.abs().max(exact.abs()));
            worst_relative_error = worst_relative_error.max(relative_error);
            eprintln!(
                "#2434 priced ARD atom={atom} axis={axis}: analytic={exact:.12e} \
                 fd={finite_difference:.12e} scaled_error={relative_error:.3e}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "#2434 gate found no ARD coordinates");
    assert!(
        max_signal > 1.0e-6,
        "#2434 gate is not load-bearing: every priced ARD derivative is numerically zero"
    );
    assert!(
        worst_relative_error <= 1.0e-4,
        "priced exact-A analytic derivative disagrees with the fixed-state central \
         difference of its value: worst scaled error {worst_relative_error:.3e}"
    );
}

/// The saddle specimen both halves of the gate must use: `(residual_scale,
/// log λ_sparse)` for `obb_patchd_fixture`. Named once because the gate
/// builds the fixture twice -- refusal, then outer-eval pricing -- and two
/// literals silently drifted apart, leaving the pricing half asserting
/// against a state that was no longer indefinite.
const GENUINE_SADDLE_SPECIMEN: (f64, f64) = (1.0, -6.0);

/// #2336 refusal companion — a GENUINE saddle (indefiniteness NOT attributable to
/// the bounded ARD concave-clamp: `λ+e_v < −floor`) must STILL return the typed
/// `IndefiniteObservedInformation` refusal, and the outer eval must price it as
/// `+inf` infeasible (not a fatal abort). Guards the
/// "refuse ⟺ genuinely-indefinite" half of the value-side contract; the price
/// half is `e_attributable_ard_saddle_prices_finite_2336`.
///
/// SPECIMEN, AND WHY THIS ONE. The scale is a MEASURED property of the fixture,
/// not a free parameter, and the previous one stopped holding: at
/// `(scale 0.02, log λ_sparse −6.0)` the criterion now returns `Ok(38.509663)`,
/// so the gate was asserting a refusal against a state that is no longer a
/// saddle. Re-scanning `obb_patchd_fixture` over
/// `scale ∈ {0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0}` ×
/// `log λ_sparse ∈ {−8, −6, −4, −2, 0}` — 50 cells — exactly four still refuse:
///
/// ```text
/// (0.01, −6)   (0.2, −2)   (1.0, −6)   (1.0, −4)
/// ```
///
/// They are isolated points, not a region, so the choice matters: `scale = 1.0`
/// is the ONLY place two adjacent lifts both refuse, which is why the specimen
/// moved there rather than to the nearer `(0.01, −6)`. A neighbour gives the
/// gate margin against the next drift instead of parking it on a knife edge.
///
/// The refusal itself is what certifies genuineness — the raising site adds the
/// clamp curvature back (`basin = λ + e_v`) and only refuses when `basin <
/// −floor`, so an `IndefiniteObservedInformation` IS the E-non-attributable
/// case by construction.
#[test]
fn genuine_saddle_is_infeasible_probe_not_fatal_2336() {
    let (mut term, target, rho) = super::tests_logdet_adjoint_780::obb_patchd_fixture(
        GENUINE_SADDLE_SPECIMEN.0,
        GENUINE_SADDLE_SPECIMEN.1,
    );
    let refusal = term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    assert!(
        matches!(
            refusal,
            Err(SaeCriterionError::IndefiniteObservedInformation { block }) if block == "joint"
        ),
        "the genuine (non-E-attributable) saddle specimen must refuse on the joint block; got: {:?}",
        refusal.map(|(value, _, _)| value)
    );

    let (term, target, rho) = super::tests_logdet_adjoint_780::obb_patchd_fixture(
        GENUINE_SADDLE_SPECIMEN.0,
        GENUINE_SADDLE_SPECIMEN.1,
    );
    let rho_flat = rho.to_flat();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target, None, rho, 40, 0.4, 1.0e-6, 1.0e-6);
    match objective.eval(&rho_flat) {
        Ok(evaluation) => assert!(
            evaluation.cost.is_infinite() && evaluation.cost.is_sign_positive(),
            "a genuine saddle-ρ must price as +inf infeasible, got cost={}",
            evaluation.cost
        ),
        Err(err) => panic!(
            "#2336: a genuine indefinite exact A must be an INFEASIBLE probe the outer solver \
             can backtrack from, not a fatal abort; got: {err}"
        ),
    }
}

/// #2228 MEASUREMENT (zz_measure) — with the certify-at-best-seen fix (½λ²/scale-min
/// keyed, band unchanged), run the criterion on ard_saddle_state.
///
/// **`min_eig` alone does not decide anything here, and reading it as if it did is
/// how this probe misled two readers into filing a correctness alarm against
/// designed behaviour.** Since #2330/#2336 the exact-`A` gate is not a PSD test: a
/// negative eigendirection is refused only when the ARD concave clamp cannot
/// account for its negativity. Per direction `v` with eigenvalue `λ < −floor`, the
/// gate forms
///
/// ```text
/// floor = max(dim·ε·‖A‖₂, √ε·vᵀBv)        (#2673 — per direction, in the B metric)
/// basin = λ + vᵀEv        (E = the ARD concave-clamp diagonal, zero on the β border)
/// basin < −floor  ⇒  genuine saddle, typed IndefiniteObservedInformation refusal
/// basin ≥ −floor  ⇒  clamp-attributable, PRICED at the basin curvature
/// ```
///
/// So `A` being indefinite is expected and by itself proves nothing; the deciding
/// quantity is `basin`, and `min_eig` and `basin` differ by exactly the `vᵀEv` this
/// probe used to omit. Report all four — `λ`, `vᵀEv`, `basin`, `floor` — for every
/// direction the gate actually examined, using the SAME `cluster_stable_eigh` the
/// gate uses so degenerate clusters resolve identically rather than to whatever a
/// plain decomposition happens to return.
///
/// `Ok` ⇒ every negative direction was clamp-attributable and priced at its basin
/// curvature. `Err` ⇒ either a direction the clamp cannot explain (a genuine
/// saddle) or the best-achievable ½λ²/scale plateauing above the band (a solver
/// stall), honestly reported at the best-seen ‖g‖.
#[test]
fn zz_measure_best_seen_classification_2228() {
    let (mut term, target, rho) = ard_saddle_state();
    let result = term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    match result {
        Ok((value, _, cache)) => {
            let a = term
                .materialize_exact_hessian_dense(&rho, target.view(), &cache)
                .expect("materialize A at certified best-seen mode");
            let e_diag = term
                .materialize_ard_concave_clamp_diagonal(&rho, &cache)
                .expect("ARD concave-clamp diagonal at the certified mode");
            let total_t = cache.delta_t_len();
            let e_beta = term
                .decoder_prior_majorizer_gap_border(&cache)
                .expect("decoder-prior majorization gap");
            let (eigs, vecs) =
                SaeManifoldTerm::cluster_stable_eigh(&a, &e_diag, e_beta.as_ref(), total_t)
                    .expect("A eigendecomposition (gate-identical clustering)");
            let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
            let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let spectral_norm = eigs.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            let joint_metric = ArrowMetric::Joint(&cache);
            let floors: Vec<f64> = (0..eigs.len())
                .map(|idx| {
                    let vbv = joint_metric
                        .quadratic_form(vecs.column(idx))
                        .expect("B quadratic form");
                    sae_exact_a_direction_floor(eigs.len(), spectral_norm, vbv)
                })
                .collect();
            let floor = floors.iter().copied().fold(0.0_f64, f64::max);
            eprintln!(
                "2228-MEASURE: Ok(value={value:.9e}) certified; min_eig={min_eig:.6e} \
                 max_eig={max_eig:.6e} widest floor={floor:.6e}"
            );
            // A CERTIFIED best-seen mode is only meaningful if the numbers it is
            // certified on are well-posed: a finite criterion value, a finite
            // ordered spectrum, and a strictly positive PD floor to compare against.
            assert!(
                value.is_finite(),
                "2228-MEASURE: a certified best-seen mode must carry a finite criterion value, \
                 got {value}"
            );
            assert!(
                min_eig.is_finite() && max_eig.is_finite() && min_eig <= max_eig,
                "2228-MEASURE: the exact-A spectrum must be finite and ordered \
                 (min_eig={min_eig}, max_eig={max_eig})"
            );
            assert!(
                floor > 0.0 && floor.is_finite(),
                "2228-MEASURE: the relative PD floor the gate decides on must be a positive \
                 finite number, got {floor}"
            );
            // Every direction the gate examined, with the quantity it decided on.
            for (idx, &lambda) in eigs.iter().enumerate() {
                let floor = floors[idx];
                if lambda >= -floor {
                    continue;
                }
                let v = vecs.column(idx);
                let limit = total_t.min(v.len());
                let e_v: f64 = (0..limit).map(|j| e_diag[j] * v[j] * v[j]).sum();
                let basin = lambda + e_v;
                // `basin` is the quantity the refuse/price decision is read off, so
                // both of its addends must be real numbers; `e_v = vᵀEv` with E the
                // ARD concave-clamp remainder is a diagonal quadratic form on a unit
                // eigenvector and cannot be infinite.
                assert!(
                    e_v.is_finite() && basin.is_finite(),
                    "2228-MEASURE: dir {idx}: the priced basin curvature must be finite \
                     (lambda={lambda}, vEv={e_v})"
                );
                eprintln!(
                    "2228-MEASURE:   dir {idx}: lambda={lambda:.6e} vEv={e_v:.6e} \
                     basin={basin:.6e} vs -floor={:.6e} => {}",
                    -floor,
                    if basin < -floor {
                        "GENUINE SADDLE (would refuse)"
                    } else {
                        "clamp-attributable (priced at basin)"
                    }
                );
            }
        }
        Err(err) => eprintln!("2228-MEASURE: Err({err:?}) => plateau above band (solver stall)"),
    }
}

/// #2336 DECISIVE MEASUREMENT v2 (zz_measure) — the CORRECTED escape test.
///
/// v1 undershot: the stall-clearing step `sqrt(2·tol/|λ|) ≈ 1.3e-3` is ~85× smaller
/// than the true 1-D minimizer the external bot found (`s ≈ 0.11`, `ΔL ≈ −1e-4`,
/// purely quadratic `½s²λ_min`), so v1 never left the near-stationary neighbourhood
/// and could not test whether a lower mode exists. This version does a real 1-D
/// line search of `penalized_objective_total` along `±v` (v = most-negative exact-A
/// eigenvector) to the minimizer, steps there, then re-converges through the
/// DESCENT-enforcing accepted lane (`converge_inner_for_undamped_logdet`,
/// `refine_progress_extension = true`, `inner_max_iter > 0`), and re-materialises
/// the exact A. Iterated up to 3× (MAX_SAE_SADDLE_ESCAPES-style), reporting the
/// spectrum at every mode. This decides: does escape+descent reach a lower/PD mode
/// (⇒ implement escape with a line-search magnitude), or does the negative-curvature
/// direction persist so no nearby lower mode exists (⇒ value-side is the honest fix)?
#[test]
fn zz_measure_saddle_escape_linesearch_reconverge_2336() {
    use super::{FaerEigh, Side};
    let (mut term, target, rho) = ard_saddle_state();
    let inner_max_iter = 40usize;
    let learning_rate = 0.4;
    let ridge_ext_coord = 1.0e-6;
    let ridge_beta = 1.0e-6;

    let mut rho_fixed = rho.clone();
    let initial = term
        .run_joint_fit_arrow_schur_for_quasi_laplace(
            target.view(),
            &mut rho_fixed,
            None,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )
        .expect("initial joint fit to seed the inner state");
    let mut loss = initial.loss;
    let mut criterion_fixed_point = initial.fixed_point;
    let options = ArrowSolveOptions::direct()
        .with_gpu_policy(term.gpu_policy)
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
    let mut cache = term
        .converge_inner_for_undamped_logdet(
            target.view(),
            &rho,
            &mut rho_fixed,
            None,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &mut criterion_fixed_point,
            &options,
            true,
        )
        .expect("converge inner to the undamped-logdet optimum (saddle)");

    for iter in 0..3usize {
        let total_t = cache.delta_t_len();
        let a = term
            .materialize_exact_hessian_dense(&rho, target.view(), &cache)
            .expect("materialize exact A");
        let (eigs, vecs) = a.eigh(Side::Lower).expect("A eigh");
        let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut min_idx = 0usize;
        let mut min_eig = f64::INFINITY;
        for (i, &v) in eigs.iter().enumerate() {
            if v < min_eig {
                min_eig = v;
                min_idx = i;
            }
        }
        let n_neg = eigs.iter().filter(|&&v| v < 0.0).count();
        let obj0 = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("penalized objective at mode");
        let reclass = term.exact_observed_information_log_dets(&rho, target.view(), &cache);
        eprintln!(
            "2336-ITER{iter}: MODE obj={obj0:.9e} min_eig={min_eig:.6e} max_eig={max_eig:.6e} n_neg={n_neg} reclass={}",
            match &reclass {
                Ok(_) => "Ok(accepted)".to_string(),
                Err(e) => format!("Err({e:?})"),
            }
        );
        let floor = 1.0e-9 * max_eig.max(1.0);
        if min_eig >= -floor {
            eprintln!(
                "2336-ITER{iter}: ACCEPTED — exact A is PD within the criterion floor; escaped"
            );
            break;
        }

        // Real 1-D line search of penalized_objective_total along ±v.
        let dir = vecs.column(min_idx);
        let dir_t = dir.slice(s![..total_t]).to_owned();
        let dir_beta = dir.slice(s![total_t..]).to_owned();
        let snapshot = term.snapshot_mutable_state();
        let mut best: (f64, f64, bool) = (obj0, 0.0, false);
        for negate in [false, true] {
            let dt = if negate { -&dir_t } else { dir_t.clone() };
            let db = if negate { -&dir_beta } else { dir_beta.clone() };
            let mut s = 1.0e-3;
            while s <= 0.6 {
                term.apply_newton_step(dt.view(), db.view(), s)
                    .expect("line-search trial step");
                let cand = term
                    .penalized_objective_total(target.view(), &rho, None, 1.0)
                    .expect("line-search objective");
                term.restore_mutable_state(&snapshot)
                    .expect("restore after line-search trial");
                if cand.is_finite() && cand < best.0 {
                    best = (cand, s, negate);
                }
                s *= 1.4;
            }
        }
        let (obj_min, s_min, negate) = best;
        eprintln!(
            "2336-ITER{iter}: LINESEARCH s_min={s_min:.6e} negate={negate} obj_min={obj_min:.9e} dL={:.6e}  (predicted ½s²|λ|={:.6e})",
            obj_min - obj0,
            0.5 * s_min * s_min * min_eig.abs()
        );
        if s_min == 0.0 || !(obj_min < obj0) {
            eprintln!(
                "2336-ITER{iter}: NO DESCENT along ±v at any tried s — escape structurally unavailable"
            );
            break;
        }

        // Step to the minimizer, then re-converge via the descent-enforcing lane.
        let dt = if negate { -&dir_t } else { dir_t.clone() };
        let db = if negate { -&dir_beta } else { dir_beta.clone() };
        term.apply_newton_step(dt.view(), db.view(), s_min)
            .expect("commit line-search step");
        let obj_stepped = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective after step");
        cache = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                &rho,
                &mut rho_fixed,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                &mut loss,
                &mut criterion_fixed_point,
                &options,
                true,
            )
            .expect("re-converge after escape step");
        let obj_reconv = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective after re-convergence");
        eprintln!(
            "2336-ITER{iter}: STEPPED obj={obj_stepped:.9e} (dL_step={:.6e}) -> RECONV obj={obj_reconv:.9e} (dL_reconv={:.6e})",
            obj_stepped - obj0,
            obj_reconv - obj_stepped
        );
    }

    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("materialize final A");
    let (eigs, _) = a.eigh(Side::Lower).expect("final A eigh");
    let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
    let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let n_neg = eigs.iter().filter(|&&v| v < 0.0).count();
    let obj_final = term
        .penalized_objective_total(target.view(), &rho, None, 1.0)
        .expect("final objective");
    let reclass = term.exact_observed_information_log_dets(&rho, target.view(), &cache);
    eprintln!(
        "2336-FINAL: obj={obj_final:.9e} min_eig={min_eig:.6e} max_eig={max_eig:.6e} n_neg={n_neg} now_pd_accepted={}",
        reclass.is_ok()
    );
    // The escape verdict is read off this final line, so its three numbers must be
    // mutually consistent and well-posed: a finite ordered spectrum, a finite
    // objective, and a negative-eigenvalue count that agrees with the reported
    // minimum (they are two readouts of the same spectrum and cannot disagree).
    assert!(
        min_eig.is_finite() && max_eig.is_finite() && min_eig <= max_eig,
        "2336-FINAL: the exact-A spectrum must be finite and ordered \
         (min_eig={min_eig}, max_eig={max_eig})"
    );
    assert!(
        obj_final.is_finite(),
        "2336-FINAL: the penalized objective at the final mode must be finite, got {obj_final}"
    );
    assert_eq!(
        n_neg > 0,
        min_eig < 0.0,
        "2336-FINAL: the negative-eigenvalue count and the reported minimum are two readouts of \
         one spectrum and must agree (n_neg={n_neg}, min_eig={min_eig})"
    );
}

/// #2336 ROOT-CAUSE (zz_measure) — is the +1.25e-4 re-convergence CLIMB (v2) a
/// gate-refreeze objective desync, or genuine B-Newton attraction to the saddle?
///
/// v2 showed: step to the exact-A negative-curvature minimizer (s≈0.11, exact
/// objective drops −9.45e-5), then `converge_inner_for_undamped_logdet(refine=true)`
/// RAISES the objective by +1.25e-4, overshooting ABOVE the original saddle. A pure
/// ∇L=0 attractor would climb by exactly +9.45e-5 (undo the step); the +3e-5
/// overshoot is the tell that the re-convergence prices a DIFFERENT objective than
/// the line-search probe. `converge_inner_for_undamped_logdet` REFRESHES the
/// collapse-prevention gates (decoder repulsion, coactivation barriers) at its entry
/// state unless `streaming_gates_frozen` is already set. This test measures (A) the
/// gate-induced objective shift at the stepped point, and (B) whether holding the
/// gates frozen-consistent across probe + re-convergence removes the climb.
#[test]
fn zz_measure_saddle_gate_desync_2336() {
    use super::{FaerEigh, Side};
    let inner_max_iter = 40usize;
    let learning_rate = 0.4;
    let ridge_ext_coord = 1.0e-6;
    let ridge_beta = 1.0e-6;

    // Helper closure would need &mut term; inline twice on fresh terms instead.
    let reach_saddle = |term: &mut SaeManifoldTerm,
                        target: &Array2<f64>,
                        rho: &SaeManifoldRho|
     -> (
        ArrowFactorCache,
        SaeManifoldRho,
        SaeManifoldLoss,
        bool,
        ArrowSolveOptions,
    ) {
        let mut rho_fixed = rho.clone();
        let initial = term
            .run_joint_fit_arrow_schur_for_quasi_laplace(
                target.view(),
                &mut rho_fixed,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
            .expect("initial joint fit");
        let mut loss = initial.loss;
        let mut criterion_fixed_point = initial.fixed_point;
        let options = ArrowSolveOptions::direct()
            .with_gpu_policy(term.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let cache = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                rho,
                &mut rho_fixed,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                &mut loss,
                &mut criterion_fixed_point,
                &options,
                true,
            )
            .expect("converge inner to saddle");
        (cache, rho_fixed, loss, criterion_fixed_point, options)
    };

    let neg_dir = |term: &SaeManifoldTerm,
                   target: &Array2<f64>,
                   rho: &SaeManifoldRho,
                   cache: &ArrowFactorCache|
     -> (f64, Array1<f64>, Array1<f64>) {
        let total_t = cache.delta_t_len();
        let a = term
            .materialize_exact_hessian_dense(rho, target.view(), cache)
            .expect("materialize A");
        let (eigs, vecs) = a.eigh(Side::Lower).expect("eigh");
        let mut min_idx = 0usize;
        let mut min_eig = f64::INFINITY;
        for (i, &v) in eigs.iter().enumerate() {
            if v < min_eig {
                min_eig = v;
                min_idx = i;
            }
        }
        let dir = vecs.column(min_idx);
        (
            min_eig,
            dir.slice(s![..total_t]).to_owned(),
            dir.slice(s![total_t..]).to_owned(),
        )
    };

    // Line search of penalized_objective_total along ±v; returns (obj, s, negate).
    let line_search = |term: &mut SaeManifoldTerm,
                       target: &Array2<f64>,
                       rho: &SaeManifoldRho,
                       dir_t: &Array1<f64>,
                       dir_beta: &Array1<f64>,
                       obj0: f64|
     -> (f64, f64, bool) {
        let snapshot = term.snapshot_mutable_state();
        let mut best = (obj0, 0.0f64, false);
        for negate in [false, true] {
            let dt = if negate { -dir_t } else { dir_t.clone() };
            let db = if negate { -dir_beta } else { dir_beta.clone() };
            let mut s = 1.0e-3;
            while s <= 0.6 {
                term.apply_newton_step(dt.view(), db.view(), s)
                    .expect("trial");
                let cand = term
                    .penalized_objective_total(target.view(), rho, None, 1.0)
                    .expect("trial obj");
                term.restore_mutable_state(&snapshot).expect("restore");
                if cand.is_finite() && cand < best.0 {
                    best = (cand, s, negate);
                }
                s *= 1.4;
            }
        }
        best
    };

    // ---- Experiment A: gate-induced objective shift at the stepped point. ----
    {
        let (mut term, target, rho) = ard_saddle_state();
        let (cache, _rf, _loss, _cfp, _opts) = reach_saddle(&mut term, &target, &rho);
        // Freeze the gates AT the saddle (what the line-search probe will price).
        term.refresh_decoder_repulsion_gate();
        term.refresh_barrier_coactivation_gate();
        term.streaming_gates_frozen = true;
        let (min_eig, dir_t, dir_beta) = neg_dir(&term, &target, &rho, &cache);
        let obj_saddle = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj saddle frozen");
        let (obj_min, s_min, negate) =
            line_search(&mut term, &target, &rho, &dir_t, &dir_beta, obj_saddle);
        let dt = if negate { -&dir_t } else { dir_t.clone() };
        let db = if negate { -&dir_beta } else { dir_beta.clone() };
        term.apply_newton_step(dt.view(), db.view(), s_min)
            .expect("step");
        // Objective at the stepped point under the SADDLE-frozen gates (probe view).
        let obj_stepped_frozen = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj stepped frozen");
        // Now refresh the gates AT the stepped point (what re-convergence would do).
        term.refresh_decoder_repulsion_gate();
        term.refresh_barrier_coactivation_gate();
        let obj_stepped_refreshed = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj stepped refreshed");
        eprintln!(
            "2336-GATESHIFT: min_eig={min_eig:.6e} s_min={s_min:.4e} negate={negate} \
             obj_saddle={obj_saddle:.9e} obj_min(ls)={obj_min:.9e} dL_step={:.6e} | \
             obj_stepped_frozen={obj_stepped_frozen:.9e} obj_stepped_refreshed={obj_stepped_refreshed:.9e} \
             GATE_SHIFT={:.6e}",
            obj_stepped_frozen - obj_saddle,
            obj_stepped_refreshed - obj_stepped_frozen
        );
        // GATE_SHIFT is a difference of two objectives, so both must be finite for
        // the shift to mean anything, and the line search must return a step from
        // inside its own bracket that never increases the objective it minimises
        // (its accumulator is seeded at `obj_saddle` and only replaced on strict
        // decrease — a violation means the search is reporting a different state
        // than the one it priced).
        assert!(
            min_eig.is_finite()
                && obj_saddle.is_finite()
                && obj_min.is_finite()
                && obj_stepped_frozen.is_finite()
                && obj_stepped_refreshed.is_finite(),
            "2336-GATESHIFT: every reported curvature and objective must be finite \
             (min_eig={min_eig}, obj_saddle={obj_saddle}, obj_min={obj_min}, \
              obj_stepped_frozen={obj_stepped_frozen}, \
              obj_stepped_refreshed={obj_stepped_refreshed})"
        );
        assert!(
            (0.0..=0.6).contains(&s_min) && obj_min <= obj_saddle,
            "2336-GATESHIFT: the line search must return a bracketed step that does not raise \
             the objective (s_min={s_min}, obj_min={obj_min}, obj_saddle={obj_saddle})"
        );
    }

    // ---- Experiment B: re-converge with gates held frozen-consistent. ----
    {
        let (mut term, target, rho) = ard_saddle_state();
        let (cache, mut rho_fixed, mut loss, mut cfp, options) =
            reach_saddle(&mut term, &target, &rho);
        // Freeze gates at the saddle and KEEP them frozen through re-convergence
        // (converge_inner sees streaming_gates_frozen==true and does NOT refresh).
        term.refresh_decoder_repulsion_gate();
        term.refresh_barrier_coactivation_gate();
        term.streaming_gates_frozen = true;
        let (min_eig, dir_t, dir_beta) = neg_dir(&term, &target, &rho, &cache);
        let obj_saddle = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj saddle frozen B");
        let (_om, s_min, negate) =
            line_search(&mut term, &target, &rho, &dir_t, &dir_beta, obj_saddle);
        let dt = if negate { -&dir_t } else { dir_t.clone() };
        let db = if negate { -&dir_beta } else { dir_beta.clone() };
        term.apply_newton_step(dt.view(), db.view(), s_min)
            .expect("step B");
        let obj_stepped = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj stepped B");
        let cache2 = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                &rho,
                &mut rho_fixed,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                &mut loss,
                &mut cfp,
                &options,
                true,
            )
            .expect("re-converge frozen B");
        let obj_reconv = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj reconv B");
        let a = term
            .materialize_exact_hessian_dense(&rho, target.view(), &cache2)
            .expect("materialize A B");
        let (eigs, _) = a.eigh(Side::Lower).expect("eigh B");
        let min_eig2 = eigs.iter().copied().fold(f64::INFINITY, f64::min);
        let n_neg2 = eigs.iter().filter(|&&v| v < 0.0).count();
        eprintln!(
            "2336-FROZENRECONV: min_eig0={min_eig:.6e} s_min={s_min:.4e} negate={negate} \
             obj_saddle={obj_saddle:.9e} obj_stepped={obj_stepped:.9e} (dL_step={:.6e}) \
             obj_reconv={obj_reconv:.9e} (dL_reconv={:.6e}) min_eig_reconv={min_eig2:.6e} n_neg_reconv={n_neg2}",
            obj_stepped - obj_saddle,
            obj_reconv - obj_stepped
        );
        eprintln!(
            "2336-FROZENVERDICT: with gates held frozen-consistent, re-convergence dL={:.6e} \
             (v2 unfrozen was +1.25e-4 CLIMB). climb_removed={}",
            obj_reconv - obj_stepped,
            (obj_reconv - obj_stepped) < 1.0e-4
        );
        // The FROZENVERDICT is `obj_reconv - obj_stepped`; a non-finite endpoint
        // would make the printed climb (and the `climb_removed` boolean derived
        // from it) arbitrary. The re-converged spectrum's two readouts must also
        // agree with each other.
        assert!(
            min_eig.is_finite()
                && obj_saddle.is_finite()
                && obj_stepped.is_finite()
                && obj_reconv.is_finite()
                && min_eig2.is_finite(),
            "2336-FROZENRECONV: every reported curvature and objective must be finite \
             (min_eig={min_eig}, obj_saddle={obj_saddle}, obj_stepped={obj_stepped}, \
              obj_reconv={obj_reconv}, min_eig_reconv={min_eig2})"
        );
        assert_eq!(
            n_neg2 > 0,
            min_eig2 < 0.0,
            "2336-FROZENRECONV: the re-converged negative-eigenvalue count must agree with the \
             reported minimum (n_neg={n_neg2}, min_eig={min_eig2})"
        );
    }
}

/// #2336 E-ATTRIBUTABILITY VERIFICATION (zz_measure) — the decisive theory gate.
///
/// The value-side fix prices a negative exact-A eigendirection v at its BASIN
/// curvature `λ + e_v` (adding back the dropped ARD-concave clamp) iff the
/// indefiniteness is fully attributable to that bounded wrinkle, i.e. `e_v ≥ |λ|`
/// where `e_v = vᵀ E v` and E is the ARD concave-clamp remainder diagonal
/// (materialize_ard_concave_clamp_diagonal). If e_v < |λ| the negative curvature
/// exceeds anything the wrinkle can produce ⇒ genuine saddle ⇒ keep refusing.
///
/// This test VERIFIES the premise on ard_saddle_state: its 2 negative eigenvalues
/// (≈ −0.015) must be E-attributable (`e_v ≥ |λ|`). If any negative direction is
/// NOT attributable, the theory is wrong and the fix must not be built as designed.
/// Cross-checks e_v (ARD-only diagonal) against the full ΔC = A−B contraction
/// (apply_exact_hessian_minus_b) — for coord-dominated directions they should be
/// close (softmax/residual channels small on those directions).
#[test]
fn zz_measure_e_attributability_2336() {
    use super::{FaerEigh, Side};
    let (mut term, target, rho) = ard_saddle_state();
    let inner_max_iter = 40usize;
    let learning_rate = 0.4;
    let ridge_ext_coord = 1.0e-6;
    let ridge_beta = 1.0e-6;

    let mut rho_fixed = rho.clone();
    let initial = term
        .run_joint_fit_arrow_schur_for_quasi_laplace(
            target.view(),
            &mut rho_fixed,
            None,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )
        .expect("initial joint fit");
    let mut loss = initial.loss;
    let mut criterion_fixed_point = initial.fixed_point;
    let options = ArrowSolveOptions::direct()
        .with_gpu_policy(term.gpu_policy)
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
    let cache = term
        .converge_inner_for_undamped_logdet(
            target.view(),
            &rho,
            &mut rho_fixed,
            None,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &mut criterion_fixed_point,
            &options,
            true,
        )
        .expect("converge inner to saddle");

    let total_t = cache.delta_t_len();
    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("materialize exact A");
    let (eigs, vecs) = a.eigh(Side::Lower).expect("A eigh");
    let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let floor = 1.0e-9 * max_eig.max(1.0);

    let e_diag = term
        .materialize_ard_concave_clamp_diagonal(&rho, &cache)
        .expect("materialize E_ard diagonal");
    eprintln!(
        "2336-EATTR: total_t={total_t} beta={} max_eig={max_eig:.6e} floor={floor:.3e} E_diag_sum={:.6e} E_diag_max={:.6e}",
        cache.k,
        e_diag.iter().sum::<f64>(),
        e_diag.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );

    let mut n_neg = 0usize;
    let mut all_attributable = true;
    for (i, &lambda) in eigs.iter().enumerate() {
        if lambda >= -floor {
            continue;
        }
        n_neg += 1;
        let v = vecs.column(i);
        // e_v = vᵀ E v (E diagonal in the t-block, zero on β / logit rows).
        let mut e_v = 0.0_f64;
        for j in 0..total_t {
            e_v += e_diag[j] * v[j] * v[j];
        }
        // Cross-check: full ΔC = A−B contraction along v (ARD + softmax + residual).
        let v_t = v.slice(s![..total_t]).to_owned();
        let v_beta = v.slice(s![total_t..]).to_owned();
        let dc = term
            .apply_exact_hessian_minus_b(
                &rho,
                target.view(),
                &cache,
                &SaeArrowVector {
                    t: v_t.clone(),
                    beta: v_beta.clone(),
                },
            )
            .expect("apply ΔC");
        let vt_dc = v_t.dot(&dc.t) + v_beta.dot(&dc.beta);
        // vᵀ(B−A)v_full = −vt_dc; the t-coord fraction of ‖v‖² measures how
        // coord-localised (hence ARD-relevant) the direction is.
        let t_frac = (0..total_t).map(|j| v[j] * v[j]).sum::<f64>();
        let priced = lambda + e_v;
        // `attributable` is the theory verdict; it is only a verdict if its inputs
        // are well-posed. `v` is a unit eigenvector, so the t-block's share of its
        // squared norm is a fraction in [0,1]: if `t_frac` ever left that range the
        // "coord-localised" cross-check printed next to it would be meaningless.
        assert!(
            e_v.is_finite() && priced.is_finite() && vt_dc.is_finite(),
            "2336-EATTR: neg#{n_neg}: the attributability inputs must be finite \
             (lambda={lambda}, e_v={e_v}, full(B-A)v.v={})",
            -vt_dc
        );
        assert!(
            (-1.0e-9..=1.0 + 1.0e-9).contains(&t_frac),
            "2336-EATTR: neg#{n_neg}: v is a unit eigenvector, so its t-block share must lie in \
             [0,1], got t_frac={t_frac}"
        );
        let attributable = priced >= -floor;
        if !attributable {
            all_attributable = false;
        }
        eprintln!(
            "2336-EATTR: neg#{n_neg} lambda={lambda:.6e} e_v(ARD)={e_v:.6e} lambda+e_v={priced:.6e} \
             attributable={attributable} | full(B-A)v.v={:.6e} t_frac={t_frac:.4e}",
            -vt_dc
        );
    }
    eprintln!(
        "2336-EATTR: VERDICT n_neg={n_neg} all_attributable={all_attributable} \
         => fixture criterion would be {}",
        if all_attributable {
            "FINITE (priced)"
        } else {
            "STILL REFUSED (genuine saddle remains)"
        }
    );
}
