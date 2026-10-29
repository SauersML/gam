//! #3355 — the noise-null expectation of the vanish proof's signal bound,
//! its Monte Carlo calibration, and the dead/healthy discrimination receipt.
//!
//! The 9-30 diagnosis on #3355 directs a per-stratum redesign: fit the decoder
//! constrained to rank r for every ρ, optimize each stratum to its own
//! stationary point, then SELECT across strata by criterion value. The 10-08
//! receipt (`tests_dead_basin_3355`, fork branch) measured that value selection
//! alone picks the DEAD stratum on this fixture (V(dead)=19.8 vs V(healthy)
//! =35.3 at the 10-19 main; re-measured 65.0 below), and the 10-12 census
//! (`tests_twilight_census_3355`, fork branch) measured why: both strata sit
//! on the #2258-promoted rank-1 branch, so rank classification cannot separate
//! them; the separating evidence is the atom's total gated signal versus the
//! residual noise scale. That census priced the evidence predicate `S_k ≤ R̄`
//! with scale 1.0 and flagged the honest gap: ONE scale to justify.
//!
//! This module derives that scale from the fitted state itself — no tuned
//! constant — calibrated to the statistic the vanish proof actually bounds.
//!
//! ## The statistic (construction.rs:1758)
//!
//! The vanish proof bounds every atom's gated signal by
//! `S_k = tr(G_k)·‖B_k‖²_F/(n·p)` with `G_k = Φ_kᵀ diag(a_k²) Φ_k`, at the
//! decoder's realised coefficients `B_k` (a read-only bound; it does not
//! assume B was fitted against this target). The proof's own boundary `S_*`
//! is a ROUNDING scale (the residual reduction's backward-error floor). The
//! MP edge `e` is a NOISE scale, but only for the per-direction
//! reconstruction energies `μ_j = σ_j(G^½ B)²/n_eff` — a different statistic
//! with a different normalizer.
//!
//! ## The conditional noise-only law of S_k (the derivation this module owns)
//!
//! Fix the coordinates, gates and Gram G, and let the decoder be ridge-fitted
//! against a noise-only target `E` with independent `N(0, R̄)` entries:
//! `B̂ = (G + λS)⁻¹ Φᵀ diag(a) E`. With `A = (G + λS)⁻¹` (symmetric) and
//! `M = Φᵀ diag(a) E` (`m×p`): `M`'s `p` columns are independent, each with
//! covariance `E[M_{·k}M_{·k}ᵀ] = Φᵀdiag(a²)Φ·R̄ = G·R̄`, so `E[M Mᵀ] = p·G·R̄`
//! and `E‖B̂‖²_F = p·R̄·tr(A G A)`, giving (the p cancels)
//!
//! ```text
//! E_null[S_k] = tr(G) · tr(A G A) · R̄ / n
//! ```
//!
//! (The Monte Carlo receipt below is what caught a dropped `p` in the first
//! draft of this derivation: the MC mean sat at 2.94× the law — `p = 3` —
//! exactly the column multiplicity of `E[MMᵀ]`.) This is NOT the
//! reconstruction-energy law of `tests_wbic_rank_charge_2933::
//! conditional_noise_null` (`E[Σ_j μ_j] = tr(C)·tr Σ/n_eff`,
//! `C = G^½ A G A G^½`): that law is calibrated to `Σμ_j`, whose scale is
//! `tr Σ/n_eff`, while `S_k`'s scale is `tr(G)·R̄/n`. Conflating the two
//! miscalibrates any gate built on `S_k`.
//!
//! ## The gate's shape (multiplicative, decoder-free threshold)
//!
//! The selection-layer evidence gate the census proposed refuses a stratum
//! whose promoted atom carries no signal beyond its own noise null:
//! `refuse ⇔ S_k ≤ E_null[S_k]`. `E_null[S_k]` is decoder-free: it prices
//! the noise-only expectation of the SAME bound the vanish proof uses, at the
//! SAME fitted coordinates/gates the stratum was priced at. No tuned multiple
//! is applied — the calibrated claim is the null expectation itself, measured
//! against Monte Carlo on the noise-only law it names; a multiple, if the
//! discrimination margin demands one, belongs to the 2933 battery's
//! calibration, not to this module. The veto direction is fail-closed: `S_k`
//! is an UPPER bound, so the gate can only refuse an atom whose PROVABLE
//! signal is under its own null expectation.
//!
//! On the receipt fixture the derived null also RETROSPECTIVELY CALIBRATES
//! the 10-12 census's scale-1.0 predicate: at the healthy ρ the derived null
//! is 0.211 ≈ 0.84·R̄ (a near-orthonormal gated basis), which is why
//! `S_k ≤ R̄` discriminated there — the ad hoc scale was secretly the null
//! expectation on that geometry. Off orthonormal geometry (railed λ, diffuse
//! gates) the two scales diverge, and only the derived null carries the MC
//! calibration.
//!
//! ## What this module is NOT
//!
//! No production gate fires from here. The value path
//! (construction_quasi_laplace.rs:944 `rank_dof_from_grams`) prices every atom
//! through the shared stratum producer; the gate's wiring site is the
//! selection layer of the per-stratum redesign, which has not landed. The
//! derived law lives in test scope for exactly the reason
//! `conditional_noise_null` does: no production consumer exists yet.

// The derived law itself (no production consumer yet — same shape as
// `conditional_noise_null` in `tests_wbic_rank_charge_2933.rs`): moves to
// production when the per-stratum selection layer (#3355 9-30 direction)
// wires it.
use super::Side;
use super::construction::VanishedAtomsProof;
use super::construction::certified_psd_spectrum;
use super::outer_objective::SaeManifoldOuterObjective;
use super::rho::SaeManifoldRho;
use super::tests::{PlantedCircleAssignmentMode, planted_circle_data, planted_circle_seed_term};
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use ndarray::Array2;

/// The noise-null law of one atom's vanish-proof signal bound `S_k`
/// (construction.rs:1758): `E_null[S_k] = tr(G)·tr(A G A)·R̄/n`.
///
/// All inputs are read from the fitted state the stratum was priced at. The
/// decoder is NOT an input — the expectation is decoder-free by construction.
#[derive(Clone, Debug)]
pub(crate) struct SignalNullLaw {
    /// `tr(G)`, the bound's numerator trace.
    gram_trace: f64,
    /// `tr(A G A)` with `A = (G + λS)⁻¹` — the ridge-fit noise trace
    /// `E‖B̂‖²_F / (p·R̄)`.
    ridge_noise_trace: f64,
    /// `n`, the sample count the bound's null expectation divides by.
    sample_count: f64,
}

impl SignalNullLaw {
    /// Derive the noise-null expectation of the vanish proof's signal bound
    /// at one fitted atom state: gated basis Gram `gram = Φᵀdiag(a²)Φ`
    /// (`m×m`), smoothing `(lam_smooth, smooth_penalty)`, scalar counts
    /// `(n, p)`. Errors mirror the stratum producer's contract violations
    /// (`validate_rank_charge_problem`-shaped): non-finite inputs, a non-PSD
    /// Gram, or a non-PD `G + λS`. An empty basis carries no law (all-zero).
    pub(crate) fn new(
        gram: &Array2<f64>,
        lam_smooth: f64,
        smooth_penalty: Option<&Array2<f64>>,
        n: usize,
        p: usize,
    ) -> Result<Self, String> {
        let m = gram.nrows();
        if gram.ncols() != m {
            return Err(format!(
                "signal null law: the gated Gram must be square; got {m}×{}",
                gram.ncols()
            ));
        }
        if gram.iter().any(|value| !value.is_finite()) {
            return Err("signal null law: the gated Gram must be finite".to_string());
        }
        if !lam_smooth.is_finite() || lam_smooth < 0.0 {
            return Err(format!(
                "signal null law: lam_smooth must be finite and non-negative; got {lam_smooth}"
            ));
        }
        if let Some(penalty) = smooth_penalty {
            if penalty.dim() != (m, m) {
                return Err(format!(
                    "signal null law: smooth penalty must be {m}×{m}; got {}×{}",
                    penalty.nrows(),
                    penalty.ncols()
                ));
            }
            if penalty.iter().any(|value| !value.is_finite()) {
                return Err("signal null law: smooth penalty must be finite".to_string());
            }
        }
        if n == 0 || p == 0 {
            return Err("signal null law: the scalar count n·p must be positive".to_string());
        }
        if m == 0 {
            return Ok(Self {
                gram_trace: 0.0,
                ridge_noise_trace: 0.0,
                sample_count: n as f64,
            });
        }
        let (evals, _) = gram
            .eigh(Side::Lower)
            .map_err(|e| format!("signal null law: eigh(G): {e}"))?;
        let evals = certified_psd_spectrum(evals.view(), "signal null law Gram")?;
        let gram_trace: f64 = evals.iter().sum();
        let mut penalized = gram.clone();
        if let Some(penalty) = smooth_penalty {
            penalized.scaled_add(lam_smooth, penalty);
        }
        let factor = penalized
            .cholesky(Side::Lower)
            .map_err(|e| format!("signal null law: G + lambda*S is not positive definite: {e}"))?;
        // tr(A G A) = tr(A² G) (cyclic, A symmetric): X = A·G, Y = A·X,
        // then Σ diag(Y). Two Cholesky solves, no factor indexing.
        let x = factor.solve_mat(gram);
        let y = factor.solve_mat(&x);
        let ridge_noise_trace: f64 = (0..m).map(|i| y[[i, i]]).sum();
        if !(gram_trace.is_finite() && ridge_noise_trace.is_finite()) {
            return Err(format!(
                "signal null law: trace quantities are not finite (tr(G)={gram_trace}, \
                 tr(AGA)={ridge_noise_trace})"
            ));
        }
        Ok(Self {
            gram_trace,
            ridge_noise_trace,
            sample_count: n as f64,
        })
    }

    /// `E_null[S_k] = tr(G)·tr(A G A)·R̄/n` — the noise-only expectation of
    /// the vanish proof's signal bound at this atom's fitted
    /// coordinates/gates, at residual mean square `R̄`.
    pub(crate) fn expected_signal_bound(&self, residual_mean_square: f64) -> f64 {
        self.gram_trace * self.ridge_noise_trace * residual_mean_square / self.sample_count
    }

    /// `tr(G)` — the gated Gram's trace (the bound's numerator).
    pub(crate) fn gram_trace(&self) -> f64 {
        self.gram_trace
    }

    /// `tr(A G A)` — the ridge-fit noise trace `E‖B̂‖²_F / (p·R̄)`.
    pub(crate) fn ridge_noise_trace(&self) -> f64 {
        self.ridge_noise_trace
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(f64::MIN_POSITIVE);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[test]
fn noise_only_monte_carlo_matches_the_derived_expectation_3355() {
    // Synthetic gated basis: n=40 rows, m=5 columns, p=3 outputs, gates in
    // (0,1], ridge λ=0.7 against S=I. Small enough for 400 exact solves.
    let n = 40_usize;
    let m = 5_usize;
    let p = 3_usize;
    let lam = 0.7_f64;
    let r_bar = 0.25_f64;
    let draws = 400_usize;
    let mut rng = 0xABCD_EF01_2345_6789_u64;

    let mut phi = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            phi[[i, j]] = standard_normal(&mut rng);
        }
    }
    let gates: Vec<f64> = (0..n).map(|_| 0.5 + 0.5 * uniform01(&mut rng)).collect();
    // G = Φᵀ diag(a²) Φ
    let mut gram = Array2::<f64>::zeros((m, m));
    for i in 0..n {
        let w = gates[i] * gates[i];
        for j in 0..m {
            for k in 0..m {
                gram[[j, k]] += w * phi[[i, j]] * phi[[i, k]];
            }
        }
    }
    let penalty = Array2::<f64>::eye(m);

    let law =
        SignalNullLaw::new(&gram, lam, Some(&penalty), n, p).expect("synthetic law must derive");
    let expected = law.expected_signal_bound(r_bar);
    assert!(
        expected > 0.0 && expected.is_finite(),
        "the derived null expectation must be positive and finite, got {expected}"
    );

    // A = (G + λS)⁻¹ via the same factorization shape the law uses.
    let mut penalized = gram.clone();
    penalized.scaled_add(lam, &penalty);
    let factor = penalized
        .cholesky(Side::Lower)
        .expect("G + lambda*S must be PD on the synthetic basis");

    // T = Φᵀ diag(a) E (m×p), B̂ = A T. Statistic: tr(G)‖B̂‖²/(n·p).
    let gram_trace: f64 = (0..m).map(|j| gram[[j, j]]).sum();
    let mut statistics = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut e = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                e[[i, j]] = r_bar.sqrt() * standard_normal(&mut rng);
            }
        }
        let mut t = Array2::<f64>::zeros((m, p));
        for i in 0..n {
            for j in 0..m {
                let a_phi = gates[i] * phi[[i, j]];
                for k in 0..p {
                    t[[j, k]] += a_phi * e[[i, k]];
                }
            }
        }
        let b_hat = factor.solve_mat(&t);
        let frobenius: f64 = b_hat.iter().map(|v| v * v).sum();
        statistics.push(gram_trace * frobenius / (n * p) as f64);
    }
    let mean: f64 = statistics.iter().sum::<f64>() / draws as f64;
    let variance: f64 = statistics
        .iter()
        .map(|s| (s - mean) * (s - mean))
        .sum::<f64>()
        / (draws - 1) as f64;
    let standard_error = (variance / draws as f64).sqrt();

    eprintln!(
        "[#3355-null-gate] MC n={n} m={m} p={p} λ={lam} R̄={r_bar} draws={draws}: \
         mean={mean:.6e} law={expected:.6e} se={standard_error:.2e} \
         ratio={:.4} tr(G)={:.4} tr(AGA)={:.6e}",
        mean / expected,
        law.gram_trace(),
        law.ridge_noise_trace(),
    );
    assert!(
        standard_error < 0.05 * expected,
        "MC standard error {standard_error:.2e} must sit under 5% of the law {expected:.6e}"
    );
    assert!(
        (mean - expected).abs() <= 4.0 * standard_error,
        "MC mean {mean:.6e} must agree with the derived expectation {expected:.6e} \
         within 4 standard errors ({:.2e})",
        4.0 * standard_error
    );
}

/// Evaluate the seeded fixture at an explicit ρ (exactly as
/// `tests_twilight_census_3355::census_at` does), then read the vanish
/// proof's `S_k` and the derived null expectation `E_null[S_k]` per atom.
fn margins_at(
    log_lambda_sparse: f64,
    log_lambda_smooth: f64,
    log_ard: f64,
) -> Result<(f64, Vec<(f64, f64, f64)>), String> {
    let z = planted_circle_data(40, 0.02);
    let (term, _seed_dispersion) =
        planted_circle_seed_term(z.view(), PlantedCircleAssignmentMode::Softmax);
    let init_rho = SaeManifoldRho::new(
        log_lambda_sparse,
        log_lambda_smooth,
        vec![ndarray::array![log_ard]],
    )
    .for_assignment(&term.assignment);
    let flat = init_rho.to_flat(&term.assignment)?;
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 50, 0.04, 1.0e-6, 1.0e-6);
    let (value, _beta) = objective.evaluate_authoritative_criterion(flat.view())?;
    let rho = objective.current_rho.clone();

    let residual = objective.term.reconstruction_residual(z.view(), &rho)?;
    let mut grams = objective.term.empty_decoder_gram_accumulator();
    objective.term.accumulate_decoder_gram(&mut grams)?;
    let n_eff = objective.term.per_atom_effective_sample_size();
    let residual_energy = objective
        .term
        .residual_energy_for_vanishing(residual.view())?;
    let rms = residual_energy.mean_square();
    let proof = objective
        .term
        .vanished_atoms_from_signal_upper_bound(&grams, &n_eff, rms)?;
    let bounds = match &proof {
        VanishedAtomsProof::Certified {
            signal_upper_bounds,
            ..
        } => signal_upper_bounds.clone(),
        VanishedAtomsProof::Unavailable { reason } => {
            return Err(format!(
                "null gate receipt: vanish proof unavailable: {reason}"
            ));
        }
    };
    let lam = rho.lambda_smooth_vec()?;
    let n = z.nrows();
    let p = objective.term.output_dim();
    let mut rows = Vec::with_capacity(objective.term.k_atoms());
    for k in 0..objective.term.k_atoms() {
        let atom = &objective.term.atoms[k];
        let law = SignalNullLaw::new(&grams[k], lam[k], Some(atom.smooth_penalty()), n, p)?;
        rows.push((n_eff[k], bounds[k], law.expected_signal_bound(rms)));
    }
    Ok((value, rows))
}

#[test]
fn the_gate_refuses_the_dead_stratum_and_keeps_the_healthy_one_3355() {
    gam_runtime::test_support::install_diagnostic_logger();
    // The receipts' ρ (cell 1: softmax, n=40, σ=0.02).
    let (dead_v, dead) =
        margins_at(-3.912, 21.309, 3.681).expect("the dead rho must evaluate (10-08 receipt)");
    let (healthy_v, healthy) =
        margins_at(2.507, -1.061, 3.681).expect("the healthy rho must evaluate");

    for (label, rows) in [("dead", &dead), ("healthy", &healthy)] {
        for (k, (n_eff_k, s_k, tau_null)) in rows.iter().enumerate() {
            eprintln!(
                "[#3355-null-gate] {label} atom {k}: n_eff={n_eff_k:.3} \
                 S_k={s_k:.6e} E_null[S_k]={tau_null:.6e} ratio={:.3e}",
                s_k / tau_null
            );
        }
    }
    eprintln!(
        "[#3355-null-gate] V(dead)={dead_v:.4} V(healthy)={healthy_v:.4} \
         (value selection alone prefers dead — the 10-08 receipt)"
    );

    // The dead stratum: every occupied atom's PROVABLE signal sits under its
    // own decoder-free null expectation — the gate refuses.
    for (k, (n_eff_k, s_k, tau_null)) in dead.iter().enumerate() {
        if *n_eff_k > 0.0 {
            assert!(
                s_k <= tau_null,
                "dead atom {k}: S_k={s_k:.6e} must sit under its null expectation \
                 {tau_null:.6e} (the evidence gate's refuse direction)"
            );
        }
    }
    // The healthy stratum: the planted circle's signal clears its null
    // expectation — the gate keeps it.
    let mut kept = 0_usize;
    for (k, (n_eff_k, s_k, tau_null)) in healthy.iter().enumerate() {
        if *n_eff_k > 0.0 {
            assert!(
                s_k > tau_null,
                "healthy atom {k}: S_k={s_k:.6e} must clear its null expectation \
                 {tau_null:.6e} (the planted circle carries real gated signal)"
            );
            kept += 1;
        }
    }
    assert!(kept > 0, "the healthy stratum must keep at least one atom");
}
