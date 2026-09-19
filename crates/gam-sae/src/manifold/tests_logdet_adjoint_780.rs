#![cfg(test)]
//! Stationary-cache `∂log|H|/∂θ` adjoint regression tests (#1416),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet_sample` are sourced from the sibling
//! `tests` module.

#![cfg(test)]

use super::*;
use super::tests::gamma_fd_tiny_fixture;
use super::tests_behavioral_fisher_rung1::pack_probe_factors;
use super::tests_recovery_split_780::{
    FdAnchorRegime, FdBranchRegime, FiniteDifferenceStratumCertificate, FixedStateLogdetSample,
    certified_branch_stable_central_difference, certified_fd_anchor, fixed_state_logdet_sample,
    rho_ladder_family, rho_ladder_family_with_tolerance, sparse_lift_ladder,
};

/// The exact `(z_j, S⁻¹ z_j)` bundle at full-basis probes `√k·e_j`, where the
/// Hutchinson outer products `logdet_theta_adjoint_from_probes` contracts are
/// algebraically exact.
fn full_basis_probe_bundle(cache: &ArrowFactorCache) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut probe = Array1::<f64>::zeros(k);
            probe[j] = sqrt_k;
            probe
        })
        .collect();
    let sinv = probes
        .iter()
        .map(|probe| {
            cache
                .schur_inverse_apply(probe.view())
                .expect("exact reduced-Schur solve at a full-basis probe")
        })
        .collect();
    (probes, sinv)
}

#[derive(Clone, Copy)]
struct TinyComplex {
    re: f64,
    im: f64,
}

impl TinyComplex {
    fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }

    fn exp(self) -> Self {
        let e = self.re.exp();
        Self {
            re: e * self.im.cos(),
            im: e * self.im.sin(),
        }
    }
}

fn real_softmax(logits: &[f64], tau: f64) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = logits
        .iter()
        .map(|&z| ((z - max_logit) / tau).exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    for weight in weights.iter_mut() {
        *weight /= sum;
    }
    weights
}

fn complex_softmax_weight_product_derivative(
    logits: &[f64],
    tau: f64,
    atom_a: usize,
    atom_b: usize,
    atom_w: usize,
    block_inner: f64,
) -> f64 {
    let h = 1.0e-30;
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut denom = TinyComplex::real(0.0);
    let mut numer_a = TinyComplex::real(0.0);
    let mut numer_b = TinyComplex::real(0.0);
    for (atom, &logit) in logits.iter().enumerate() {
        let z = TinyComplex {
            re: (logit - max_logit) / tau,
            im: if atom == atom_w { h / tau } else { 0.0 },
        };
        let exp_z = z.exp();
        denom = denom.add(exp_z);
        if atom == atom_a {
            numer_a = exp_z;
        }
        if atom == atom_b {
            numer_b = exp_z;
        }
    }
    let a = numer_a.div(denom);
    let b = numer_b.div(denom);
    a.mul(b).mul(TinyComplex::real(block_inner)).im / h
}

#[test]
pub(crate) fn softmax_tt_weight_product_logit_adjoint_hits_both_factors_2156() {
    let logits = [0.31_f64, -0.27, 0.14, -0.08];
    let tau = 0.73_f64;
    let inv_tau = 1.0 / tau;
    let assignments = real_softmax(&logits, tau);
    let block_inner = 1.417_f64;

    for (atom_a, atom_b, atom_w) in [(0usize, 2usize, 1usize), (2usize, 2usize, 2usize)] {
        let h_ab = assignments[atom_a] * assignments[atom_b] * block_inner;
        let one_factor =
            h_ab * (if atom_w == atom_a { 1.0 } else { 0.0 } - assignments[atom_w]) * inv_tau;
        let fixed = h_ab
            * SaeManifoldTerm::softmax_data_weight_product_logit_factor(
                &assignments,
                atom_a,
                atom_b,
                atom_w,
                inv_tau,
            );
        let complex_step = complex_softmax_weight_product_derivative(
            &logits,
            tau,
            atom_a,
            atom_b,
            atom_w,
            block_inner,
        );
        let ratio = fixed / one_factor;
        assert!(
            (ratio - 2.0).abs() <= 1.0e-12,
            "one-factor softmax product derivative must be 2x low: got ratio {ratio:.12}"
        );
        assert!(
            (fixed - complex_step).abs() <= 1.0e-6 * (1.0 + complex_step.abs()),
            "fixed softmax product derivative must match complex-step: fixed={fixed:.12e}, complex={complex_step:.12e}"
        );
    }
}

// #2330 Patch D — fixed-θ EXACT-A logdet for the θ-adjoint FD arbiter: rebuild
// the fixed-θ̂ cache at the (perturbed) state and return log|A| (not log|B|).
// `None` when the criterion refuses or A is indefinite there (so the FD probe
// can report that instead of panicking).
fn fixed_state_exact_a_logdet(
    mut term: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Option<f64> {
    let (_v, _l, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .ok()?;
    term.exact_observed_information_log_dets(rho, target.view(), &cache)
        .ok()
}

// #2330 Patch D — an ordered-Beta--Bernoulli fixture whose target is generated
// with the SAME independent-logistic gates the model applies. The shared
// `gamma_fd_tiny_fixture` builds its target from NORMALIZED softmax weights, so
// simply flipping that fixture's mode to ordered Beta--Bernoulli leaves a target
// the model cannot reach: the resulting large residual drives the dropped
// residual curvature `ΔC = ⟨error_metric, ∂²f⟩` big enough to push the exact
// `A = B + ΔC` indefinite, and the Phase-2a criterion then refuses at
// construction. `residual_scale` adds a deterministic model-unreachable
// component on top of the reachable target, so `ΔC` — the object Patch D
// differentiates — is nonzero and tunable rather than either zero (a fixture
// that would false-green the arbiter) or saddle-inducing.
pub(crate) fn obb_patchd_fixture(
    residual_scale: f64,
    log_lambda_sparse: f64,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10usize;
    let p = 3usize;
    let k_atoms = 2usize;
    let m = 3usize;
    let tau = 0.7_f64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let weights = [
        [
            [0.10, -0.05, 0.03],
            [0.35, -0.20, 0.12],
            [-0.16, 0.18, 0.08],
        ],
        [
            [-0.08, 0.04, 0.06],
            [0.22, 0.10, -0.18],
            [0.11, -0.24, 0.15],
        ],
    ];
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let phase = (row as f64 + 0.35) / n as f64;
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.21).fract();
        logits[[row, 0]] = if row % 2 == 0 { 0.8 } else { -0.6 };
        logits[[row, 1]] = if row % 3 == 0 { -0.4 } else { 0.5 };
        for atom in 0..k_atoms {
            // Ordered Beta--Bernoulli gate: independent per-atom logistic, NOT a
            // normalized simplex weight.
            let gate = 1.0 / (1.0 + (-logits[[row, atom]] / tau).exp());
            let theta = std::f64::consts::TAU * coords[atom][[row, 0]];
            let basis = [1.0, theta.sin(), theta.cos()];
            for out_col in 0..p {
                for basis_col in 0..m {
                    target[[row, out_col]] +=
                        gate * basis[basis_col] * weights[atom][basis_col][out_col];
                }
            }
        }
        for out_col in 0..p {
            target[[row, out_col]] +=
                residual_scale * (((row * 7 + out_col * 3) as f64) * 0.7).sin();
        }
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let (phi, jet) = evaluator.evaluate(coords[atom].view()).unwrap();
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            weights[atom][basis_col][out_col]
        });
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("patchd_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone()),
        );
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        AssignmentMode::ordered_beta_bernoulli(tau, 0.9, false),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(
        log_lambda_sparse,
        -6.0,
        vec![Array1::from_vec(vec![-6.0]), Array1::from_vec(vec![-6.0])],
    );
    (term, target, rho)
}

// #2330 Patch D prerequisite — map the residual scale at which the converged
// exact `A` stops being positive definite, and how big the residual-curvature
// block `ΔC` is inside that window. This decides whether the Patch-D FD arbiter
// can be anchored on a PD fixture at all, and separates "the shared fixture
// manufactured a saddle" from "every converged mode is an A-saddle" (the latter
// would gate #2330 behind #2336's saddle escape rather than behind Patch D).
#[test]
fn sae_exact_a_pd_window_scan_2330_patchd() {
    for &scale in &[0.0_f64, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4] {
        let (mut term, target, rho) = obb_patchd_fixture(scale, -6.0);
        let built = term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        match built {
            Ok((_value, _loss, cache)) => {
                match term.exact_a_spectrum_summary(&rho, target.view(), &cache) {
                    Ok((min_eig, max_eig, n_neg, dc_frob, a_frob)) => {
                        eprintln!(
                            "PATCHD_WINDOW scale={scale:.4} PD_OK min_eig={min_eig:.6e} \
                             max_eig={max_eig:.6e} n_neg={n_neg} dc_frob={dc_frob:.6e} \
                             a_frob={a_frob:.6e} dc_rel={:.6e}",
                            dc_frob / a_frob.max(1.0e-300)
                        );
                        // This window scan decides where the Patch-D arbiter may be
                        // anchored, so each summary row must be a real spectrum
                        // summary: ordered finite extremes, Frobenius norms that are
                        // finite non-negative magnitudes, and a negative count that
                        // agrees with the reported minimum. A silently-NaN row would
                        // otherwise read as "PD_OK".
                        assert!(
                            min_eig.is_finite() && max_eig.is_finite() && min_eig <= max_eig,
                            "scale={scale}: the exact-A spectrum must be finite and ordered \
                             (min_eig={min_eig}, max_eig={max_eig})"
                        );
                        assert!(
                            dc_frob.is_finite()
                                && dc_frob >= 0.0
                                && a_frob.is_finite()
                                && a_frob >= 0.0,
                            "scale={scale}: Frobenius norms must be finite non-negative \
                             magnitudes (dc_frob={dc_frob}, a_frob={a_frob})"
                        );
                        // `n_neg` counts eigenvalues below the relative PD floor, so
                        // it may be 0 while `min_eig` is a hair negative — but it
                        // can never be positive unless the minimum is genuinely
                        // negative. That one-way implication is exact.
                        assert!(
                            n_neg == 0 || min_eig < 0.0,
                            "scale={scale}: {n_neg} eigenvalue(s) below the PD floor but the \
                             reported minimum is min_eig={min_eig} ≥ 0 — the count and the \
                             minimum are two readouts of one spectrum"
                        );
                    }
                    Err(e) => eprintln!("PATCHD_WINDOW scale={scale:.4} SPECTRUM_ERR {e}"),
                }
            }
            Err(e) => eprintln!("PATCHD_WINDOW scale={scale:.4} CRITERION_REFUSED {e:?}"),
        }
    }
}

// #2330 Patch D FD ARBITER — the per-coordinate gap between the analytic
// exact-A θ-adjoint `Γ_A,w = tr(A⁺ ∂A/∂θ_w)` and a CENTRAL DIFFERENCE of
// `exact_observed_information_log_dets(...).0 = log|A|` over frozen θ̂ with the
// cache REBUILT at each perturbed state (a frozen cache would false-green the
// gate). At baseline — before the Patch-D `∂ΔC/∂θ` legs land — the residual
// here IS the missing term, coordinate by coordinate.
//
// Anchored on `obb_patchd_fixture`, whose exact A is positive definite at the
// converged mode (see `sae_exact_a_pd_window_scan_2330_patchd`); the shared
// softmax fixture is not OBB-reachable and lands on an A-saddle where the
// criterion refuses outright.
#[test]
fn sae_exact_a_theta_adjoint_gap_measure_2330_patchd() {
    let (mut term, target, rho) = obb_patchd_fixture(0.0, -6.0);
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("PD converged cache");
    let log_a = term
        .exact_observed_information_log_dets(&rho, target.view(), &cache)
        .expect("exact-A log det at the converged mode");
    eprintln!("PATCHD base log|A|={log_a:.9e}");
    let gamma = term
        .exact_a_theta_adjoint_joint(&rho, target.view(), &cache)
        .expect("analytic exact-A joint theta adjoint");

    // Probe slots read off the ACTUAL cache layout rather than hardcoded, so a
    // layout change cannot silently repoint the probes at the wrong variables.
    let mut probes: Vec<(usize, usize, SaeLocalRowVar)> = Vec::new();
    for row in 0..3usize {
        let vars = term
            .row_vars_for_cache_row(row, &cache)
            .expect("row vars for probe layout");
        for (local, var) in vars.iter().enumerate() {
            probes.push((row, local, *var));
        }
    }
    probes.truncate(8);

    // This state has a narrow admitted basin: a 1e-3 coordinate perturbation
    // leaves it outright (every endpoint refuses). Verify convergence of the
    // scalar oracle itself before comparing it with the analytic derivative.
    // Richardson removes its leading O(h²) term.
    //
    // #2828 — the coordinate step is 4e-5, MEASURED, not chosen. The oracle
    // differences `log|A|` through a rebuilt cache and a fresh symmetric
    // eigendecomposition, so its noise floor is `O(eps·‖A‖)` per eigenvalue;
    // at this mode `A` carries a cluster of six eigenvalues near 2.7e-8 against
    // a spectral norm of 3.0e1, and those directions dominate
    // `tr(A⁺ ∂A/∂θ)`. Below the floor the central difference gets WORSE as `h`
    // shrinks, which is the opposite of the regime Richardson assumes. Scanned
    // over the four coordinate probes at this fixture (`max_rel` is the
    // analytic-vs-FD gap, `max_oracle` the Richardson non-convergence the gate
    // charges to the same budget):
    //
    //   h = 1e-3  every endpoint REFUSED (outside the admitted basin)
    //   h = 3e-4  max_rel 9.926e-4  max_oracle 1.827e-2
    //   h = 1e-4  max_rel 2.347e-5  max_oracle 1.419e-4
    //   h = 4e-5  max_rel 5.231e-5  max_oracle 6.780e-5   <- used
    //   h = 1e-5  max_rel 3.350e-4  max_oracle 7.927e-4
    //   h = 3e-6  max_rel 2.859e-3  max_oracle 5.052e-3
    //
    // The U shape is the signature of a roundoff-limited oracle, and it is the
    // measurement that says the analytic θ-adjoint is RIGHT here: at the
    // oracle's own optimum the two agree to 2e-5..5e-5, an order of magnitude
    // inside `COORD_TOL`. The former 1e-5 sat a decade below the floor and was
    // reading its own noise.
    const COORD_TOL: f64 = 1.0e-3;
    const LOGIT_TOL: f64 = 3.0e-2;
    let mut max_coord_rel = 0.0_f64;
    let mut max_logit_rel = 0.0_f64;
    for &(row, local, var) in &probes {
        // Logits move on the gate-temperature scale. Their response is O(1),
        // so tiny coordinate-sized steps amplify the scalar eigensolve's
        // rounding error. Coordinate responses here reach O(1e4), and need
        // the smaller step to remain inside the admitted basin.
        let coarse_step = match var {
            SaeLocalRowVar::Logit { .. } => match term.assignment.mode {
                AssignmentMode::OrderedBetaBernoulli { temperature, .. } => 1.0e-3 * temperature,
                _ => unreachable!("OBB fixture"),
            },
            SaeLocalRowVar::Coord { .. } => 4.0e-5,
        };
        let mut estimates = [0.0_f64; 3];
        for (step_index, divisor) in [1.0_f64, 2.0, 4.0].into_iter().enumerate() {
            let h = coarse_step / divisor;
            let mut plus = term.clone();
            let mut minus = term.clone();
            // Clone drops these lagged operands. Preserve the same frozen
            // Newton objective whose Hessian/third derivative is contracted.
            for endpoint in [&mut plus, &mut minus] {
                endpoint.decoder_repulsion_gate = term.decoder_repulsion_gate.clone();
                endpoint.barrier_coactivation_gate = term.barrier_coactivation_gate.clone();
                endpoint.amplitude_barrier_gate = term.amplitude_barrier_gate;
                endpoint.streaming_gates_frozen = true;
            }
            match var {
                SaeLocalRowVar::Logit { atom } => {
                    plus.assignment.logits[[row, atom]] += h;
                    minus.assignment.logits[[row, atom]] -= h;
                }
                SaeLocalRowVar::Coord { atom, axis } => {
                    let mut fp = plus.assignment.coords[atom].as_flat().clone();
                    let mut fm = minus.assignment.coords[atom].as_flat().clone();
                    let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                    fp[idx] += h;
                    fm[idx] -= h;
                    plus.assignment.coords[atom].set_flat(fp.view());
                    minus.assignment.coords[atom].set_flat(fm.view());
                }
            }
            let a = fixed_state_exact_a_logdet(plus, &target, &rho)
                .expect("every positive finite-difference endpoint must be admitted");
            let b = fixed_state_exact_a_logdet(minus, &target, &rho)
                .expect("every negative finite-difference endpoint must be admitted");
            estimates[step_index] = (a - b) / (2.0 * h);
        }
        let coarse = (4.0 * estimates[1] - estimates[0]) / 3.0;
        let fine = (4.0 * estimates[2] - estimates[1]) / 3.0;
        let tolerance = match var {
            SaeLocalRowVar::Coord { .. } => COORD_TOL,
            SaeLocalRowVar::Logit { .. } => LOGIT_TOL,
        };
        let oracle_error = (fine - coarse).abs() / (1.0 + fine.abs().max(coarse.abs()));
        assert!(
            oracle_error < tolerance,
            "row={row} var={var:?}: scalar FD oracle has not converged: estimates={estimates:?}, Richardson error={oracle_error:e}"
        );
        let analytic = gamma.t[cache.row_offsets[row] + local];
        let rel = (fine - analytic).abs() / (1.0 + fine.abs().max(analytic.abs()));
        // Charge the oracle's remaining uncertainty to the same error budget
        // instead of allowing it in addition to the derivative tolerance.
        let accounted_error = rel + oracle_error;
        match var {
            SaeLocalRowVar::Coord { .. } => max_coord_rel = max_coord_rel.max(accounted_error),
            SaeLocalRowVar::Logit { .. } => max_logit_rel = max_logit_rel.max(accounted_error),
        }
        eprintln!(
            "PATCHD_GAP row={row} local={local} var={var:?} fd={fine:.9e} analytic={analytic:.9e} rel={rel:.3e} oracle_error={oracle_error:.3e}"
        );
    }
    eprintln!("PATCHD_ARBITER max_coord_rel={max_coord_rel:.3e} max_logit_rel={max_logit_rel:.3e}");
    assert!(
        max_coord_rel < COORD_TOL,
        "exact-A theta-adjoint coordinate channel must match FD: max_coord_rel={max_coord_rel:.3e} >= {COORD_TOL:.1e}"
    );
    assert!(
        max_logit_rel < LOGIT_TOL,
        "exact-A theta-adjoint logit channel must match FD: max_logit_rel={max_logit_rel:.3e} >= {LOGIT_TOL:.1e}"
    );
}

// #2330 Patch D — channel-2 exercise gate. The main arbiter fixture sets
// log_lambda_sparse=-6 (OBB prior weight e^{-6}≈0.0025), so the ordered-BB prior
// curvature channel-2 (∂ΔC_obb/∂logit) is nearly inert there — correct-in-form
// but numerically ~0, which would let a channel-2 SIGN error ship silently.
// This variant raises the prior weight so channel-2 carries measurable weight;
// the logit slots staying FD-consistent here is what actually exercises its sign.
#[test]
fn sae_exact_a_theta_adjoint_gap_measure_2330_patchd_weighted() {
    // Scan a few sparse weights at residual_scale 0; report PD + the logit gaps so
    // a channel-2 sign error shows up as a blown logit slot.
    for &(rs, lls) in &[
        (0.005_f64, -4.0_f64),
        (0.005, -3.0),
        (0.01, -3.0),
        (0.02, -2.0),
    ] {
        let (mut term, target, rho) = obb_patchd_fixture(rs, lls);
        let built = term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        let cache = match built {
            Ok((_v, _l, c)) => c,
            Err(e) => {
                eprintln!("PATCHD_W lls={lls:.2} CRITERION_REFUSED {e:?}");
                continue;
            }
        };
        let gamma = match term.exact_a_theta_adjoint_joint(&rho, target.view(), &cache) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("PATCHD_W lls={lls:.2} GAMMA_ERR {e}");
                continue;
            }
        };
        let h = 1.0e-5;
        for row in 0..2usize {
            let vars = term.row_vars_for_cache_row(row, &cache).expect("vars");
            for (local, var) in vars.iter().enumerate() {
                let mut plus = term.clone();
                let mut minus = term.clone();
                match *var {
                    SaeLocalRowVar::Logit { atom } => {
                        plus.assignment.logits[[row, atom]] += h;
                        minus.assignment.logits[[row, atom]] -= h;
                    }
                    SaeLocalRowVar::Coord { atom, axis } => {
                        let mut fp = plus.assignment.coords[atom].as_flat().clone();
                        let mut fm = minus.assignment.coords[atom].as_flat().clone();
                        let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                        fp[idx] += h;
                        fm[idx] -= h;
                        plus.assignment.coords[atom].set_flat(fp.view());
                        minus.assignment.coords[atom].set_flat(fm.view());
                    }
                }
                let analytic = gamma.t[cache.row_offsets[row] + local];
                match (
                    fixed_state_exact_a_logdet(plus, &target, &rho),
                    fixed_state_exact_a_logdet(minus, &target, &rho),
                ) {
                    (Some(a), Some(b)) => {
                        let fd = (a - b) / (2.0 * h);
                        let rel = (fd - analytic).abs() / (1.0 + fd.abs().max(analytic.abs()));
                        eprintln!(
                            "PATCHD_W rs={rs:.3} lls={lls:.2} row={row} var={var:?} fd={fd:.6e} \
                             analytic={analytic:.6e} rel={rel:.3e}"
                        );
                    }
                    _ => {
                        eprintln!("PATCHD_W rs={rs:.3} lls={lls:.2} row={row} var={var:?} refused")
                    }
                }
            }
        }
    }
}

// ─── #2080 / #2712 from-probes θ-adjoint parity, restored (#2818) ───────────
//
// `c0a21b554` deleted this gate because it no longer compiled, and it no longer
// compiled because `d484a091a` had deleted the FD-anchor scaffolding it stood
// on — `certified_fd_anchor`, `FdAnchorRegime`, `FdAnchorCandidate`,
// `rho_ladder_family`, `sparse_lift_ladder`, `deflation_blind_cache`. Every one
// of those was `#[cfg(test)]`, where the sweep's criterion ("no production
// artifact links this function") is true of everything by construction.
//
// The three production entry points the gate actually grades —
// `SaeManifoldTerm::logdet_theta_adjoint`,
// `SaeManifoldTerm::logdet_theta_adjoint_from_probes`, and
// `ArrowFactorCache::schur_inverse_apply` — were untouched, so this is a
// rebuild against them directly. Everything the anchor machinery did for the
// `any_maximum()` regime this gate declared is inlined as closures: walk the
// declared `log λ_sparse` ladder, converge each member's own inner mode, freeze
// it at `inner_max_iter = 0`, and accept the first member the criterion prices
// finitely. Nothing in that acceptance can see the finite difference or the
// analytic value, so it still cannot converge on "whatever agrees".

/// #2080 θ-adjoint from-probes — SOFTMAX fixture. Exercises the softmax entropy
/// dense off-diagonal channel + the core t–t / t–β / β–β selected-inverse folds.
///
/// The matrix-free θ-adjoint reconstructed from the FULL-BASIS probe bundle
/// (`z_j = √k·e_j`, exact dense `S⁻¹` via `cache.schur_inverse_apply`) must
/// reproduce the dense selected-inverse θ-adjoint. This isolates the from-probes
/// reconstruction; the dense adjoint is already FD-validated against `log|H|`
/// elsewhere.
///
/// On a DEFLATED cache the two have to be told apart from the deflation-blind
/// operator before agreement means anything: the deflated and undeflated
/// θ-adjoints coincide wherever the deflation is inactive, so machine-precision
/// agreement is ALSO what a port that ignored deflation would produce. The gate
/// measures `‖Γ_dense − Γ_deflation-blind‖∞` first and refuses to read parity as
/// evidence unless the two provably separate.
#[test]
fn sae_logdet_theta_adjoint_from_probes_matches_dense_softmax_2080() {
    // The declared `log λ_sparse` ladder. The assignment-strength penalty is the
    // dial that moves a state between the deflating and non-deflating regimes,
    // so it is the natural declared axis for a regime the gate needs but cannot
    // control directly. Ordered by lift; the accepted member is reported.
    const SPARSE_LIFTS: [f64; 9] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];

    let (base_term, target, base_rho) =
        crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();

    // The `any_maximum()` anchor regime, inlined: the frozen state must merely
    // BE a maximum the criterion will price. Everything the gate then
    // differentiates is defined there; nothing further is asserted about it.
    let mut rejections: Vec<String> = Vec::with_capacity(SPARSE_LIFTS.len());
    let mut certified: Option<(String, SaeManifoldTerm, SaeManifoldRho, ArrowFactorCache)> = None;
    for &lift in &SPARSE_LIFTS {
        let description = format!("log_lambda_sparse={lift:.2}");
        let mut rho = base_rho.clone();
        rho.log_lambda_sparse = lift;
        let mut term = base_term.clone();
        // Reach this member's own mode first. A solve that refuses is a
        // rejection of this member, not a panic.
        if let Err(error) = term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        ) {
            rejections.push(format!("  {description}: inner solve refused: {error}"));
            continue;
        }
        // `inner_max_iter = 0` freezes θ̂ where the ladder put it: the anchor is
        // the point the gate declares, not wherever the solve would wander.
        match term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        ) {
            Ok((value, loss, cache)) if value.is_finite() && loss.total().is_finite() => {
                certified = Some((description, term, rho, cache));
                break;
            }
            Ok((value, loss, _)) => rejections.push(format!(
                "  {description}: frozen state priced non-finitely (value={value}, loss={})",
                loss.total()
            )),
            Err(error) => rejections.push(format!(
                "  {description}: criterion refused the frozen state: {error}"
            )),
        }
    }
    let (accepted, term, rho, cache) = certified.unwrap_or_else(|| {
        panic!(
            "#2080 from-probes softmax parity: no member of the declared ladder is a maximum \
             the criterion will price. Widening the ladder is a fixture decision and dropping \
             the regime would change what is proved. Rejections:\n{}",
            rejections.join("\n")
        )
    });
    eprintln!("#2080 from-probes softmax parity: anchor certified at {accepted}");

    let solver = DeflatedArrowSolver::plain(&cache);
    let inverse = term
        .materialize_joint_inverse(&cache, &solver)
        .expect("dense joint inverse");
    let dense = term
        .logdet_theta_adjoint_dense(&rho, &cache, &inverse, false, false, None)
        .expect("dense theta-adjoint");

    let deflated_rows = cache
        .deflated_row_directions
        .iter()
        .filter(|d| !d.is_empty())
        .count();
    // The deflation-blind operator: the production dense adjoint against the
    // same cache with ONLY the deflation metadata stripped — the per-row
    // Cholesky factors and the reduced Schur are untouched. That is exactly what
    // a from-probes port which silently dropped the Daleckii–Krein correction
    // would return, so the distance to it is the resolution this gate has. It is
    // a REFERENCE, never a route.
    let separation = if deflated_rows == 0 {
        0.0
    } else {
        let mut blind = cache.clone();
        let rows = cache.deflated_row_directions.len();
        blind.deflated_row_directions = std::sync::Arc::from(vec![Vec::new(); rows]);
        blind.deflation_row_spectra = std::sync::Arc::from(vec![None; rows]);
        let blind_solver = DeflatedArrowSolver::plain(&blind);
        let blind_inverse = term
            .materialize_joint_inverse(&blind, &blind_solver)
            .expect("deflation-blind joint inverse");
        let blind_gamma = term
            .logdet_theta_adjoint_dense(&rho, &blind, &blind_inverse, false, false, None)
            .expect("deflation-blind dense theta-adjoint");
        dense
            .t
            .iter()
            .zip(blind_gamma.t.iter())
            .chain(dense.beta.iter().zip(blind_gamma.beta.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    };

    let k = cache.k;
    assert!(
        k > 0,
        "fixture must have a non-empty border to exercise S⁻¹ folds"
    );
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
    let mf = term
        .logdet_theta_adjoint_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("matrix-free theta-adjoint");

    assert_eq!(dense.t.len(), mf.t.len());
    assert_eq!(dense.beta.len(), mf.beta.len());
    let mut max_abs = 0.0_f64;
    let mut parity = 0.0_f64;
    for (d, m) in dense
        .t
        .iter()
        .zip(mf.t.iter())
        .chain(dense.beta.iter().zip(mf.beta.iter()))
    {
        parity = parity.max((d - m).abs());
        max_abs = max_abs.max(d.abs());
    }
    eprintln!(
        "#2080/#2712 from-probes θ-adjoint gate: {deflated_rows} deflated row(s), \
         ‖Γ_dense‖∞ = {max_abs:.6e}, ‖Γ_dense − Γ_from-probes‖∞ = {parity:.6e}, \
         ‖Γ_dense − Γ_deflation-blind‖∞ = {separation:.6e}"
    );
    // #2712: `1e-8` per entry is the historical undeflated bar; on a deflated
    // cache it has to be finer than the correction it is supposed to be
    // sensitive to, which the assertion at the end checks against the measured
    // separation.
    let relative_tolerance = if deflated_rows > 0 { 1.0e-10 } else { 1.0e-8 };
    for (i, (d, m)) in dense.t.iter().zip(mf.t.iter()).enumerate() {
        assert!(
            (d - m).abs() <= relative_tolerance * (1.0 + d.abs()),
            "theta-adjoint gamma_t[{i}] mismatch: dense={d:.10e}, from_probes={m:.10e}"
        );
    }
    for (i, (d, m)) in dense.beta.iter().zip(mf.beta.iter()).enumerate() {
        assert!(
            (d - m).abs() <= relative_tolerance * (1.0 + d.abs()),
            "theta-adjoint gamma_beta[{i}] mismatch: dense={d:.10e}, from_probes={m:.10e}"
        );
    }
    assert!(
        max_abs > 0.0 && max_abs.is_finite(),
        "the theta-adjoint must be non-trivial to make the parity check meaningful"
    );
    if deflated_rows > 0 {
        // Non-vacuity, stated as a RATIO against the MEASURED separation rather
        // than as an absolute threshold. On the historical fixture the
        // correction moves Γ by 8.5e-8 against ‖Γ‖∞ = 98.9 — the deflated
        // direction is a near-null the raw derivative barely touches — so an
        // absolute floor would reject an honest fixture, while the per-entry
        // `1e-8·(1+|Γ|)` parity tolerance ALONE would admit a deflation-blind
        // port (8.5e-8 < 1e-6). The ratio is the margin by which such a port is
        // actually caught.
        assert!(
            separation > 0.0 && parity * 1.0e3 <= separation,
            "the deflated and deflation-blind θ-adjoints must SEPARATE before parity \
             is evidence of anything: a port that dropped the Daleckii–Krein \
             correction would also agree here. Measured separation {separation:.6e} \
             against parity error {parity:.6e} on {deflated_rows} deflated row(s)."
        );
        // ...and the parity tolerance the loops above applied must itself be
        // finer than the separation, or a deflation-blind port would slip
        // through them even though the ratio above holds.
        let loop_tolerance = relative_tolerance * (1.0 + max_abs);
        assert!(
            loop_tolerance < separation,
            "the per-entry parity tolerance {loop_tolerance:.6e} is coarser than the \
             {separation:.6e} distance to the deflation-blind operator, so the \
             element-wise assertions above would pass a port that dropped the \
             Daleckii–Krein correction. Tighten them or pick a fixture on which the \
             correction is larger."
        );
    }
}

/// #2333 — the dense and from-probes θ-adjoints must carry the same logit legs on a
/// Softmax row with more than one free logit, under both operators.
///
/// `f39b76a831` rewrote the from-probes logit×logit block and dropped the #2080
/// simplex Jacobian third derivative, while the dense tower kept it. The K=2 parity
/// above has one free logit, so it exercises only the diagonal of that leg. With
/// K=3 the chart holds two free logits, and every off-diagonal triple `(a, b, w)` of
/// `simplex_gate_logit_jacobian_third` and of the entropy derivative enters.
///
/// The state is one directly factored majorizer system, so no ladder decides which
/// state is compared. Full-basis probes make the from-probes reconstruction
/// algebraically exact, so the towers differ only by accumulation order. The
/// separation is the simplex leg's own size, rebuilt from the dense inverse over the
/// logit slots: a tower that drops the leg is short by exactly that amount on a
/// logit entry.
#[test]
fn softmax_theta_adjoint_logit_legs_agree_across_towers_with_two_free_logits_2333() {
    let n = 12usize;
    let p = 3usize;
    let k_atoms = 3usize;
    let m = 3usize;
    let temperature = 0.9_f64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let weights = [
        [
            [0.10, -0.05, 0.03],
            [0.35, -0.20, 0.12],
            [-0.16, 0.18, 0.08],
        ],
        [
            [-0.08, 0.04, 0.06],
            [0.22, 0.10, -0.18],
            [0.11, -0.24, 0.15],
        ],
        [
            [0.05, 0.09, -0.07],
            [-0.14, 0.27, 0.10],
            [0.19, 0.06, -0.21],
        ],
    ];
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords: Vec<Array2<f64>> = (0..k_atoms)
        .map(|_| Array2::<f64>::zeros((n, 1)))
        .collect();
    for row in 0..n {
        let phase = (row as f64 + 0.35) / n as f64;
        for (atom, block) in coords.iter_mut().enumerate() {
            block[[row, 0]] = (phase + 0.29 * atom as f64).fract();
        }
        // Unequal gates on both free logits; the reference logit `K − 1` stays at zero.
        logits[[row, 0]] = 0.6 * (0.7 * row as f64).cos();
        logits[[row, 1]] = -0.4 + 0.5 * (0.9 * row as f64).sin();
    }
    let bases: Vec<(Array2<f64>, Array3<f64>)> = coords
        .iter()
        .map(|block| evaluator.evaluate(block.view()).unwrap())
        .collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        AssignmentMode::softmax(temperature),
    )
    .unwrap();
    let mut target = Array2::<f64>::zeros((n, p));
    let mut gates = vec![0.0_f64; k_atoms];
    for row in 0..n {
        assignment
            .try_assignments_row_into(row, &mut gates)
            .expect("finite softmax row");
        for (atom, (phi, _)) in bases.iter().enumerate() {
            for out_col in 0..p {
                for basis_col in 0..m {
                    target[[row, out_col]] +=
                        gates[atom] * phi[[row, basis_col]] * weights[atom][basis_col][out_col];
                }
            }
        }
        // A small residual, so the fixture is not an exact interpolant.
        target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
        target[[row, 2]] += 1.0e-3 * (0.29 * row as f64).cos();
    }
    let atoms: Vec<SaeManifoldAtom> = bases
        .into_iter()
        .enumerate()
        .map(|(atom, (phi, jet))| {
            let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
                weights[atom][basis_col][out_col]
            });
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("free_logits_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone())
        })
        .collect();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(
        0.8_f64.ln(),
        0.0,
        (0..k_atoms).map(|_| Array1::from_vec(vec![50.0_f64.ln()])).collect(),
    );
    let system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("majorizer arrow system");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let cache = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .expect("positive-definite majorizer factor")
        .2;

    // Premise: some row must carry two free logit slots, or no off-diagonal leg enters.
    let row_vars: Vec<Vec<SaeLocalRowVar>> = (0..n)
        .map(|row| term.row_vars_for_cache_row(row, &cache).expect("row variables"))
        .collect();
    let most_free_logits = row_vars
        .iter()
        .map(|vars| {
            vars.iter()
                .filter(|var| matches!(var, SaeLocalRowVar::Logit { .. }))
                .count()
        })
        .max()
        .unwrap_or(0);
    assert!(
        most_free_logits >= 2,
        "#2333 premise: a K=3 Softmax row must carry two free logit slots, got {most_free_logits}"
    );

    let solver = DeflatedArrowSolver::plain(&cache);
    let inverse = term
        .materialize_joint_inverse(&cache, &solver)
        .expect("dense joint inverse");
    let (probes, sinv) = full_basis_probe_bundle(&cache);

    // The simplex leg `Σ_{a,b logit} inv_ab · ∂³J/∂ℓ_a∂ℓ_b∂ℓ_w` on every logit entry,
    // relative to `1 + |Γ_w|` of the dense majorizer adjoint (unit row weights).
    let count = crate::assignment::simplex_gate_free_count(&term.assignment)
        .expect("a K=3 Softmax assignment has free logits");
    let dense_majorizer = term
        .logdet_theta_adjoint_dense(&rho, &cache, &inverse, false, false, None)
        .expect("dense majorizer theta-adjoint");
    let mut simplex_separation = 0.0_f64;
    for (row, vars) in row_vars.iter().enumerate() {
        let base = cache.row_offsets[row];
        term.assignment
            .try_assignments_row_into(row, &mut gates)
            .expect("finite softmax row");
        for (w, var_w) in vars.iter().enumerate() {
            let SaeLocalRowVar::Logit { atom: atom_w } = *var_w else {
                continue;
            };
            let mut leg = 0.0_f64;
            for (a, var_a) in vars.iter().enumerate() {
                let SaeLocalRowVar::Logit { atom: atom_a } = *var_a else {
                    continue;
                };
                for (b, var_b) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: atom_b } = *var_b else {
                        continue;
                    };
                    leg += inverse[[base + b, base + a]]
                        * crate::assignment::simplex_gate_logit_jacobian_third(
                            &gates,
                            atom_a,
                            atom_b,
                            atom_w,
                            count,
                            temperature.recip(),
                        );
                }
            }
            simplex_separation =
                simplex_separation.max(leg.abs() / (1.0 + dense_majorizer.t[base + w].abs()));
        }
    }

    let mut report = Vec::new();
    let mut worst_parity = 0.0_f64;
    for (label, exact_a, operator) in [
        ("majorizer", false, EvidenceOperator::Majorizer),
        ("exact", true, EvidenceOperator::ExactObservedInformation),
    ] {
        let dense = term
            .logdet_theta_adjoint_dense(&rho, &cache, &inverse, false, exact_a, None)
            .expect("dense theta-adjoint");
        let from_probes = term
            .logdet_theta_adjoint_from_probes(&rho, &cache, &probes, &sinv, operator, None)
            .expect("from-probes theta-adjoint");
        assert_eq!(dense.t.len(), from_probes.t.len());
        assert_eq!(dense.beta.len(), from_probes.beta.len());
        let mut parity = 0.0_f64;
        let mut worst_entry = String::from("none");
        for (row, vars) in row_vars.iter().enumerate() {
            let base = cache.row_offsets[row];
            for (w, var_w) in vars.iter().enumerate() {
                let (reference, observed) = (dense.t[base + w], from_probes.t[base + w]);
                let gap = (reference - observed).abs() / (1.0 + reference.abs());
                if gap > parity {
                    parity = gap;
                    worst_entry = format!(
                        "row {row} {var_w:?}: dense={reference:.12e} from_probes={observed:.12e}"
                    );
                }
            }
        }
        for (reference, observed) in dense.beta.iter().zip(from_probes.beta.iter()) {
            parity = parity.max((reference - observed).abs() / (1.0 + reference.abs()));
        }
        report.push(format!("{label}: parity={parity:.6e} worst {worst_entry}"));
        worst_parity = worst_parity.max(parity);
    }
    eprintln!(
        "#2333 SOFTMAX_FREE_LOGIT_TOWER_PARITY free_logits={most_free_logits} \
         simplex_separation={simplex_separation:.6e}\n  {}",
        report.join("\n  ")
    );
    assert!(
        worst_parity <= 1.0e-10,
        "#2333: the from-probes θ-adjoint must reproduce the dense tower on every entry of a \
         Softmax state with two free logits:\n  {}",
        report.join("\n  ")
    );
    assert!(
        simplex_separation > 1.0e-6 && simplex_separation > 1.0e3 * worst_parity,
        "#2333: the parity bar must reject a tower missing the simplex Jacobian leg: \
         separation {simplex_separation:e}, parity {worst_parity:e}"
    );
}

/// A fixed-state log-det sample at the anchor's collapse-prevention gates.
///
/// The θ-adjoint differentiates `log|H|` with the gates held: the outer objective declares one gate
/// set for its whole solve, and every derivative reads it as a constant (#2933 F05). An endpoint that
/// re-derives the gates at its own state differentiates their motion as well. On the ordered
/// Beta--Bernoulli tiny fixtures the separation barrier's per-atom `N_eff,k = Σ_i a_ik²` moves with an
/// interior gate, and on atom 1's logit that motion read 0.33 to 0.81, while the same endpoints at held
/// gates matched `Γ` at the stencil's resolution (job 1163634, #2080). `SaeManifoldTerm::clone` drops a
/// declaration, so the gates are declared on the clone that is evaluated, and the evaluated term must
/// still carry them.
fn held_gate_logdet_sample(
    term: &SaeManifoldTerm,
    gates: &super::penalties::CollapsePreventionGates,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    perturb: impl FnOnce(&mut SaeManifoldTerm),
) -> FixedStateLogdetSample {
    let mut state = term.clone();
    state.declare_collapse_prevention_gates(gates);
    perturb(&mut state);
    let cache = state
        .penalized_quasi_laplace_criterion_with_cache(target.view(), rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
        .expect("fixed-state cache at the anchor's gates")
        .2;
    assert!(
        state.collapse_prevention_gates() == *gates,
        "the endpoint must be evaluated at the anchor's declared gates, not re-derive them"
    );
    FixedStateLogdetSample {
        value: cache
            .arrow_log_det()
            .expect("fixed-state authoritative joint logdet"),
        stratum: FiniteDifferenceStratumCertificate::from_arrow_cache(&cache),
    }
}

/// The central difference of a held-gate log-det at `h/2`, with the tolerance the #2933 F07 metric gate
/// derives from its own stencil. `|fd(h) − fd(h/2)|` is three times `fd(h/2)`'s `O(h²)` truncation
/// (`fd(h) − fd(h/2) = ¾·c·h²` against `c·h²/4`), and `dim·ε·max|log|H||/(h/2)` bounds the
/// factorization's roundoff in the quotient. Every endpoint stays in the center's stratum.
fn held_gate_central_difference(
    label: &str,
    center: &FiniteDifferenceStratumCertificate,
    dim: usize,
    h: f64,
    sample: impl Fn(f64) -> FixedStateLogdetSample,
) -> (f64, f64) {
    let mut quotients = [0.0_f64; 2];
    let mut roundoff_scale = 0.0_f64;
    for (slot, step) in [h, 0.5 * h].into_iter().enumerate() {
        let plus = sample(step);
        let minus = sample(-step);
        center.assert_same_stratum(&format!("{label} (+{step:e})"), &plus.stratum);
        center.assert_same_stratum(&format!("{label} (-{step:e})"), &minus.stratum);
        quotients[slot] = (plus.value - minus.value) / (2.0 * step);
        roundoff_scale = plus.value.abs().max(minus.value.abs()) / step;
    }
    let tolerance =
        (quotients[0] - quotients[1]).abs() + dim as f64 * f64::EPSILON * roundoff_scale;
    (quotients[1], tolerance)
}

/// The ordered Beta--Bernoulli prior's logit leg in `Γ` at one probe, read off the production third
/// channels: `E_rr·∂D_r/∂ℓ_r` at fixed active mass plus the column-mass channel
/// `u_r·Σ_i E_ii·∂D_i/∂M`. Every term carries the gate slope `u = z(1 − z)/τ`, so at a saturated gate
/// the leg falls below any finite-difference resolution and a logit probe verifies nothing: before
/// 19ce8785f3 these fixtures settled atom 1 at `z ≈ 1e-8` (#2080). The tiny fixtures carry no row
/// weights.
fn obb_logit_prior_leg(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    row: usize,
    atom: usize,
) -> f64 {
    let channels = crate::assignment::ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
        &term.assignment,
        rho,
        None,
    )
    .expect("ordered Beta--Bernoulli third channels")
    .expect("an ordered Beta--Bernoulli assignment");
    let inverse = term
        .materialize_joint_inverse(cache, &DeflatedArrowSolver::plain(cache))
        .expect("dense joint inverse");
    let site = |i: usize| {
        term.row_vars_for_cache_row(i, cache)
            .expect("row variables")
            .iter()
            .position(|var| matches!(var, SaeLocalRowVar::Logit { atom: slot } if *slot == atom))
            .map(|position| cache.row_offsets[i] + position)
    };
    let k = channels.k_max;
    let column_mass: f64 = (0..term.n_obs())
        .filter_map(|i| site(i).map(|index| inverse[[index, index]] * channels.m_channel[i * k + atom]))
        .sum();
    let own = site(row).expect("the probed logit is a free slot");
    inverse[[own, own]] * channels.local_logit_third[row * k + atom]
        + channels.z_jac[row * k + atom] * column_mass
}

/// gam#2144 — the log-det row jets must be whitened whenever the metric
/// `whitens_likelihood()` at ANY rank, not only when rank-deficient. The
/// arrow-Schur assembly builds the likelihood Hessian from whitened Jacobians
/// (`Jᵀ U Uᵀ J`) under any whitening factor, so a FULL-RANK non-identity factor
/// (here `diag(1, 2, 1.5)`, `rank == p == 3`) rescales the output-space
/// derivatives just like a low-rank sketch does. The pre-fix code gated jet
/// whitening on `ordered_beta_bernoulli_low_rank_whiten()` (`whitens_likelihood && rank < p`), so
/// full-rank whitening left the row jets in RAW output space — differentiating
/// `JᵀJ` against an assembled `Jᵀ U Uᵀ J`. This pins the production
/// `logdet_theta_adjoint` against a fixed-state central difference of the
/// authoritative whitened joint `log|H|`; the unpatched (identity-on-the-jet)
/// path fails it.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_full_rank_whitening_2144() {
    use gam_problem::RowMetric;
    use std::sync::Arc;
    let (mut term, target, rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    let n = term.n_obs();
    let p = term.output_dim();
    // Full-rank (rank == p) DIAGONAL non-identity whitening factor U = diag(d).
    // M_n = U Uᵀ = diag(d²) is genuinely non-identity, so the whitened Jacobian
    // Jᵀ U Uᵀ J ≠ JᵀJ, yet the metric has NO null space — whitening engages with
    // no rank-deficiency in play.
    let d = [1.0_f64, 2.0, 1.5];
    assert_eq!(p, d.len(), "diagonal whitening factor width must equal p");
    let s = p;
    let mut u = Array2::<f64>::zeros((n, p * s));
    for row in 0..n {
        for i in 0..p {
            u[[row, i * s + i]] = d[i];
        }
    }
    term.set_row_metric(RowMetric::behavioral_fisher(Arc::new(u), p, s).unwrap())
        .unwrap();
    assert!(
        term.whiten_logdet_row_jets(),
        "full-rank whitening metric must whiten the log-det row jets"
    );
    assert!(
        term.row_metric().is_some_and(|m| m.metric_rank() == p),
        "rank-{s} == p={p} metric must be genuinely full-rank (this test discriminates \
         jet whitening from rank-deficiency handling)"
    );
    // #2144/#1038: the ordered Beta--Bernoulli PSD majorization is now UNCONDITIONAL (any rank, any
    // metric), so the joint Hessian here is the majorized operator too — the
    // historical #1416 non-PD landscape at `log_lambda_sparse = 0.5` no longer
    // exists. Keep the historical PD-island level `−0.8` for continuity (the
    // discriminating property of this test is unchanged either way:
    // `Jᵀ U Uᵀ J ≠ JᵀJ` separates whitened row jets from raw ones, which is
    // what the fixed-state FD comparison pins).
    let anchor = certified_fd_anchor(
        "#2144 full-rank whitened theta adjoint",
        &target,
        FdAnchorRegime::any_maximum(),
        rho_ladder_family(
            &term,
            sparse_lift_ladder(&rho, &[-0.8, -0.4, 0.0, 0.4, 0.8, 1.2]),
            200,
        ),
    );
    let term = anchor.term;
    let rho = anchor.rho;
    let cache = anchor.cache;
    // #2144 — the majorized θ-adjoint production contracts, at full-basis probes
    // where its Hutchinson outer products are exact. The dense
    // `logdet_theta_adjoint_dense(.., exact_a = false, ..)` arm carries no ordered
    // Beta--Bernoulli prior-majorizer channel (neither the local direct-z entry nor
    // the shared-mass column pass), so under a rank-deficient whitening metric it
    // read `analytic = 2.15e-14` against `fd = 1.4285` on atom 1's logit.
    let (probes, sinv) = full_basis_probe_bundle(&cache);
    let gamma = term
        .logdet_theta_adjoint_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("Gamma");
    let h = 1.0e-5;
    let fd_stratum = anchor.stratum;
    let gates = term.collapse_prevention_gates();
    let dim = cache.delta_t_len() + cache.k;
    let probes_idx = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
        (1usize, 2usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
        (6usize, 3usize, SaeLocalRowVar::Coord { atom: 1, axis: 0 }),
    ];
    for (row, local_pos, var) in probes_idx {
        let label = format!("full-rank whitened Gamma row={row} local_pos={local_pos}");
        let (fd, tolerance) = held_gate_central_difference(&label, &fd_stratum, dim, h, |step| {
            held_gate_logdet_sample(&term, &gates, &target, &rho, |state| match var {
                SaeLocalRowVar::Logit { atom } => state.assignment.logits[[row, atom]] += step,
                SaeLocalRowVar::Coord { atom, axis } => {
                    let mut flat = state.assignment.coords[atom].as_flat().clone();
                    let idx = row * state.assignment.coords[atom].latent_dim() + axis;
                    flat[idx] += step;
                    state.assignment.coords[atom].set_flat(flat.view());
                }
            })
        });
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        if let SaeLocalRowVar::Logit { atom } = var {
            let prior_leg = obb_logit_prior_leg(&term, &rho, &cache, row, atom);
            assert!(
                prior_leg.abs() > tolerance,
                "{label}: regime: the ordered Beta--Bernoulli logit leg {prior_leg:.3e} is not \
                 resolved by the finite difference (tolerance {tolerance:.3e}), so the gate is \
                 saturated and this probe verifies nothing about the logit legs"
            );
            eprintln!("{label}: logit prior leg {prior_leg:.6e}");
        }
        eprintln!(
            "{label}: fd={fd:.10e} analytic={analytic:.10e} gap={:.3e} tolerance={tolerance:.3e}",
            fd - analytic
        );
        // #2330: `local_pos` above is HARDCODED, i.e. this loop asserts every row is
        // packed `[Logit0, Logit1, Coord0, Coord1]`. If the whitened layout orders its
        // local block differently, `analytic` is a different variable's derivative and
        // the comparison is meaningless rather than merely out of tolerance. Report the
        // resolved block width so a failure distinguishes the two: a width of 4 is
        // consistent with the assumed packing, whereas a width equal to the coordinate
        // count alone says the logit probes are indexing coordinate slots.
        let block_width = cache.row_offsets[row + 1] - cache.row_offsets[row];
        assert!(
            (fd - analytic).abs() <= tolerance,
            "{label}: fd={fd:.8e}, analytic={analytic:.8e}, tolerance={tolerance:.3e} \
             (var={var:?}, row block width={block_width}, gamma.t len={}, row_offsets[{row}]={})",
            gamma.t.len(),
            cache.row_offsets[row],
        );
    }
}

/// Learnable-alpha ordered Beta--Bernoulli logit theta-adjoint.
/// `learnable_alpha = true`, a path the fixed-alpha `..._ordered_beta_bernoulli` sibling never
/// exercises. Under learnable α the resolved weight convention flips (`weight`
/// stays 1.0 and `log_lambda_sparse` drives `α` via `resolve_learnable_weight`
/// instead of scaling the prior), so a single logit perturbation holds alpha
/// fixed and moves only `M_k` and the local sigmoid gate.
///
/// The comparison point must EXIST and be STATIONARY: like the indefinite-basin
/// diagnosis driving the whole #1625 fix, the analytic
/// `Γ = tr(H⁻¹ ∂H/∂θ)` equals the fixed-state central difference of `log|H|`
/// only at a CONVERGED inner cache. A short inner budget (e.g. `iter = 5`) leaves
/// (t, β) non-stationary, and `held_gate_logdet_sample` (which re-solves with
/// `iter = 0`) then differences `log|H|` about a different state, manufacturing a
/// spurious O(several-%) mismatch that does NOT shrink with the FD step — the
/// tell that it is a state desync, not truncation. Converging the inner solve
/// (`iter = 200`, tol `1e-8`) makes Γ and the FD share one stationary state, and
/// the learnable-α logit adjoint then matches to ≈6 digits.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ordered_beta_bernoulli_learnable_alpha_1625()
 {
    let (mut term, target, rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    // The historical `ρ₀ = 0.6` was itself the result of a hand sweep for a
    // level that drives a PD learnable-α cache. The sweep is the ladder; the
    // certificate is what the sweep was looking for.
    let anchor = certified_fd_anchor(
        "#1625 learnable-alpha ordered Beta--Bernoulli theta adjoint",
        &target,
        FdAnchorRegime::any_maximum(),
        rho_ladder_family_with_tolerance(
            &term,
            sparse_lift_ladder(&rho, &[0.6, 0.9, 1.2, 0.3, 0.0, 1.5]),
            200,
            1.0e-8,
        ),
    );
    let term = anchor.term;
    let rho = anchor.rho;
    let cache = anchor.cache;
    // #2144 — the majorized θ-adjoint production contracts, at full-basis probes
    // where its Hutchinson outer products are exact. The dense
    // `logdet_theta_adjoint_dense(.., exact_a = false, ..)` arm carries no ordered
    // Beta--Bernoulli prior-majorizer channel (neither the local direct-z entry nor
    // the shared-mass column pass), so under a rank-deficient whitening metric it
    // read `analytic = 2.15e-14` against `fd = 1.4285` on atom 1's logit.
    let (probes, sinv) = full_basis_probe_bundle(&cache);
    let gamma = term
        .logdet_theta_adjoint_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("Gamma");
    let h = 1.0e-5;
    let fd_stratum = anchor.stratum;
    let gates = term.collapse_prevention_gates();
    let dim = cache.delta_t_len() + cache.k;
    // Probe both atoms across distinct rows so the shared-mass channel is
    // exercised on both columns under learnable alpha.
    let probes = [
        (0usize, 0usize, 0usize),
        (4usize, 1usize, 1usize),
        (7usize, 0usize, 0usize),
    ];
    for (row, local_pos, atom) in probes {
        let label =
            format!("learnable-α ordered Beta--Bernoulli Gamma row={row} local_pos={local_pos}");
        let (fd, tolerance) = held_gate_central_difference(&label, &fd_stratum, dim, h, |step| {
            held_gate_logdet_sample(&term, &gates, &target, &rho, |state| {
                state.assignment.logits[[row, atom]] += step;
            })
        });
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let prior_leg = obb_logit_prior_leg(&term, &rho, &cache, row, atom);
        assert!(
            prior_leg.abs() > tolerance,
            "{label}: regime: the ordered Beta--Bernoulli logit leg {prior_leg:.3e} is not \
             resolved by the finite difference (tolerance {tolerance:.3e}), so the gate is \
             saturated and this probe verifies nothing about the logit legs"
        );
        eprintln!(
            "{label}: fd={fd:.10e} analytic={analytic:.10e} gap={:.3e} tolerance={tolerance:.3e} \
             logit prior leg {prior_leg:.6e}",
            fd - analytic
        );
        assert!(
            (fd - analytic).abs() <= tolerance,
            "{label}: fd={fd:.8e}, analytic={analytic:.8e}, tolerance={tolerance:.3e}"
        );
    }
}

/// The assembly PSD-majorizes the ordered Beta--Bernoulli curvature
/// unconditionally, so the
/// θ-adjoint must differentiate that SAME majorized operator. This is the
/// metric-first analogue of `..._ordered_beta_bernoulli`: install a rank-2 BehavioralFisher
/// metric (`s = 2 < p = 3`, a genuinely rank-deficient whitening) on the ordered Beta--Bernoulli tiny
/// fixture and check the analytic `Γ` matches the fixed-state dense FD of `log|H|`
/// — both flow through the majorized assembly (`held_gate_logdet_sample` rebuilds the
/// SAME majorized `H` at the anchor's gates). This guards the majorized θ-adjoint channels against the
/// majorized criterion log-det in the whitened+rank-deficient regime, where the
/// whitened data curvature cannot dominate the raw indefinite prior pieces.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ordered_beta_bernoulli_low_rank_metric_2144()
 {
    use gam_problem::RowMetric;
    use std::sync::Arc;
    let (mut term, target, rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    let n = term.n_obs();
    let p = term.output_dim();
    let s = 2usize;
    // Deterministic rank-2 output-Fisher sketch, directional (not a scalar × I) so
    // the metric genuinely whitens with a nontrivial null space.
    let mut seed = 0x2144_ABCD_u64;
    let probes = Array3::<f64>::from_shape_fn((n, p, s), |(_, i, kk)| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let base = if kk == 0 && i == 0 {
            1.2
        } else if kk == 1 && i + 1 == p {
            1.0
        } else {
            0.0
        };
        base + 0.15 * (((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5)
    });
    let u = pack_probe_factors(probes.view());
    term.set_row_metric(RowMetric::behavioral_fisher(Arc::new(u), p, s).unwrap())
        .unwrap();
    assert!(
        term.row_metric()
            .is_some_and(|m| m.whitens_likelihood() && m.metric_rank() < p),
        "rank-{s} metric on p={p} must be a genuinely rank-deficient whitening metric"
    );
    let anchor = certified_fd_anchor(
        "#2144 low-rank-metric ordered Beta--Bernoulli theta adjoint",
        &target,
        FdAnchorRegime::any_maximum(),
        rho_ladder_family(&term, sparse_lift_ladder(&rho, &PD_BASIN_SPARSE_LIFTS), 200),
    );
    let term = anchor.term;
    let rho = anchor.rho;
    let cache = anchor.cache;
    // #2144 — the majorized θ-adjoint production contracts, at full-basis probes
    // where its Hutchinson outer products are exact. The dense
    // `logdet_theta_adjoint_dense(.., exact_a = false, ..)` arm carries no ordered
    // Beta--Bernoulli prior-majorizer channel (neither the local direct-z entry nor
    // the shared-mass column pass), so under a rank-deficient whitening metric it
    // read `analytic = 2.15e-14` against `fd = 1.4285` on atom 1's logit.
    let (probes, sinv) = full_basis_probe_bundle(&cache);
    let gamma = term
        .logdet_theta_adjoint_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("Gamma");
    let h = 1.0e-5;
    let fd_stratum = anchor.stratum;
    let gates = term.collapse_prevention_gates();
    let dim = cache.delta_t_len() + cache.k;
    let probes_idx = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
        (1usize, 2usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
        (6usize, 3usize, SaeLocalRowVar::Coord { atom: 1, axis: 0 }),
    ];
    for (row, local_pos, var) in probes_idx {
        let label = format!("majorized ordered Beta--Bernoulli Gamma row={row} local_pos={local_pos}");
        let (fd, tolerance) = held_gate_central_difference(&label, &fd_stratum, dim, h, |step| {
            held_gate_logdet_sample(&term, &gates, &target, &rho, |state| match var {
                SaeLocalRowVar::Logit { atom } => state.assignment.logits[[row, atom]] += step,
                SaeLocalRowVar::Coord { atom, axis } => {
                    let mut flat = state.assignment.coords[atom].as_flat().clone();
                    let idx = row * state.assignment.coords[atom].latent_dim() + axis;
                    flat[idx] += step;
                    state.assignment.coords[atom].set_flat(flat.view());
                }
            })
        });
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        if let SaeLocalRowVar::Logit { atom } = var {
            let prior_leg = obb_logit_prior_leg(&term, &rho, &cache, row, atom);
            assert!(
                prior_leg.abs() > tolerance,
                "{label}: regime: the ordered Beta--Bernoulli logit leg {prior_leg:.3e} is not \
                 resolved by the finite difference (tolerance {tolerance:.3e}), so the gate is \
                 saturated and this probe verifies nothing about the logit legs"
            );
            eprintln!("{label}: logit prior leg {prior_leg:.6e}");
        }
        eprintln!(
            "{label}: fd={fd:.10e} analytic={analytic:.10e} gap={:.3e} tolerance={tolerance:.3e}",
            fd - analytic
        );
        assert!(
            (fd - analytic).abs() <= tolerance,
            "{label}: fd={fd:.8e}, analytic={analytic:.8e}, tolerance={tolerance:.3e}"
        );
    }
}

/// #2330 — the DEFLATED-fixture arbiter for the #1006 envelope adjoint.
///
/// The tiny-fixture test above converges to a well-conditioned PD state with NO
/// per-row deflation, so it never exercises the Daleckii–Krein
/// [`SaeManifoldTerm::deflation_block_correction`] path. This fixture (the
/// residual-excited two-atom circle, lifted ρ, its softmax on the ladder of
/// [`super::tests_deflated_from_probes_2712::deflating_gate_temperatures`])
/// deflates every row's logit slot by construction, so `logdet_theta_adjoint` here goes
/// through the DK correction. `Γ_joint = tr(H⁻¹ ∂H/∂θ)` must equal the fixed-θ̂
/// central difference of the criterion's authoritative `arrow_log_det()` — the
/// SAME operator the DK comment claims to differentiate. The #2253 full-set
/// Hessian gate proved the assembled outer gradient is non-conservative in the
/// smooth↔ARD cross with the deflated `Γ_joint` as the dominant carrier (bisected
/// to `asym=1.97e-2`); #2330 tracks that as a defect in the deflated θ-adjoint
/// itself, which this test isolates independently of ρ and of the CH5 builder.
/// Its green unblocks the #2253 full-set gate and the capability→Dense flip. FD
/// is skipped on the ARD majorizer kink (`|cos κt| < 0.2`), where
/// `max(α cos κt, 0)` is non-smooth.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_fd_on_deflated_fixture_2330() {
    // The anchor is the shared residual-excited deflated anchor: evaluation ρ
    // with θ̂ frozen, lifted off the floor so the deflated legs sit above
    // finite-difference noise, and certified to be a MAXIMUM there.
    let (term, rho, target, cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2330 deflated-fixture theta adjoint",
        );
    // #2144 — the majorized θ-adjoint production contracts, at full-basis probes
    // where its Hutchinson outer products are exact, as the three pins above.
    let (probes, sinv) = full_basis_probe_bundle(&cache);
    let gamma = term
        .logdet_theta_adjoint_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("Gamma_joint");

    let h = 1.0e-5;
    let fd_stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(&cache);
    let mut checked = 0usize;
    let mut worst = 0.0_f64;
    // #2366 branch-guard tally. Reported rather than asserted: which regime a
    // stencil lands in is a measurement about this fixture at this step, and
    // pinning it would turn an honest "the guard cannot see here" into a gate
    // on the roundoff floor.
    let mut branch_smooth = 0usize;
    let mut branch_roundoff = 0usize;
    let mut worst_branch_ratio = 0.0_f64;
    for row in 0..term.n_obs() {
        let vars = term
            .row_vars_for_cache_row(row, &cache)
            .expect("row vars for deflated fixture");
        for (local_pos, var) in vars.iter().enumerate() {
            // Probe the ARD coordinate slots (the deflated t-block); skip the
            // majorizer kink where the fixed-θ̂ central difference is invalid.
            let SaeLocalRowVar::Coord { atom, axis } = *var else {
                continue;
            };
            let t_val = term.assignment.coords[atom].row(row)[axis];
            let cos_kt = (std::f64::consts::TAU * t_val).cos();
            if cos_kt.abs() < 0.2 {
                continue;
            }
            let at = |dt: f64| {
                let mut t = term.clone();
                let mut flat = t.assignment.coords[atom].as_flat().clone();
                let idx = row * t.assignment.coords[atom].latent_dim() + axis;
                flat[idx] += dt;
                t.assignment.coords[atom].set_flat(flat.view());
                fixed_state_logdet_sample(t, &target, &rho)
            };
            // The deflated fixture is where classifier-invisible nonsmoothness
            // is likeliest, so this gate carries the value-free branch guard on
            // top of the structural stratum certificate (#2366).
            let (fd, branch) = certified_branch_stable_central_difference(
                &format!("deflated Gamma_joint row={row} pos={local_pos} atom={atom} axis={axis}"),
                &fd_stratum,
                h,
                at,
            );
            match branch {
                FdBranchRegime::Smooth { ratio } => {
                    branch_smooth += 1;
                    worst_branch_ratio = worst_branch_ratio.max(ratio);
                }
                FdBranchRegime::RoundoffDominated { coarse_gap, floor } => {
                    branch_roundoff += 1;
                    eprintln!(
                        "deflated Gamma_joint row={row} pos={local_pos}: branch guard \
                         inapplicable — coarse gap {coarse_gap:.3e} is at the roundoff \
                         floor {floor:.3e}"
                    );
                }
            }
            let analytic = gamma.t[cache.row_offsets[row] + local_pos];
            let err = (fd - analytic).abs();
            worst = worst.max(err);
            eprintln!(
                "deflated Gamma_joint row={row} pos={local_pos} atom={atom} axis={axis} \
                 cos_kt={cos_kt:.3} fd={fd:.8e} analytic={analytic:.8e} err={err:.3e}"
            );
            let tol = 2.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
            assert!(
                err <= tol,
                "deflated Gamma_joint mismatch row={row} pos={local_pos}: \
                 fd={fd:.8e}, analytic={analytic:.8e} (the deflated log-det θ-adjoint \
                 does not match ∂arrow_log_det/∂θ — DK correction defect)"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "deflated-fixture adjoint test probed no interior ARD coordinate (worst so far {worst:.3e})"
    );
    eprintln!(
        "#2366 branch guard on the deflated fixture: {branch_smooth} stencil(s) certified \
         smooth (worst h² gap ratio {worst_branch_ratio:.4}, predicted 0.25), \
         {branch_roundoff} roundoff-dominated and therefore not concluded"
    );
    assert_eq!(
        branch_smooth + branch_roundoff,
        checked,
        "every probed stencil must reach one of the two branch-guard regimes"
    );
}

/// #2712 — the identity the whole fix rests on, asserted directly rather than
/// inferred from a downstream trace: at FULL-BASIS probes the from-probes
/// reconstruction of a row's selected-inverse blocks equals the dense
/// [`DeflatedArrowSolver::selected_inverse_row_blocks`] ON A DEFLATED ROW.
///
/// If `cache.undamped_factor(i)` factorized the RAW `H_tt^(i)` — the reading the
/// issue's refusal was written from — this would fail on exactly the deflated
/// rows, because `A_i⁻¹` would then be the undeflated block. It does not, because
/// the factor carries the CONDITIONED spectrum: the gate also checks
/// `A_i v = v` on each deflated direction, which is the unit-stiffness pin
/// itself. The anchor is the shared ordered Beta–Bernoulli deflated anchor,
/// [`super::tests_deflated_from_probes_2712::obb_deflated_anchor`], where every
/// row's logit slots deflate by construction.
#[test]
fn sae_row_selected_inverse_from_probes_is_the_deflated_block_2712() {
    let cache = super::tests_deflated_from_probes_2712::obb_deflated_anchor(
        "#2712 deflated selected-inverse reconstruction",
    )
    .3;
    let k = cache.k;
    assert!(k > 0, "the fixture must have a border for S⁻¹ to matter");
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<ndarray::Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = ndarray::Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<ndarray::Array1<f64>> = probes
        .iter()
        .map(|v| {
            cache
                .schur_inverse_apply(v.view())
                .expect("schur_inverse_apply")
        })
        .collect();
    let solver = DeflatedArrowSolver::plain(&cache);
    let beta_inv = solver.beta_inv().expect("beta_inv");

    let mut deflated_rows = 0usize;
    let mut max_block_error = 0.0_f64;
    let mut block_scale = 0.0_f64;
    let mut max_unit_pin_error = 0.0_f64;
    for row in 0..cache.row_dims.len() {
        let dirs = cache
            .deflated_row_directions
            .get(row)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if dirs.is_empty() {
            continue;
        }
        deflated_rows += 1;

        // The unit-stiffness pin, read straight off the cached factor: rebuild
        // `A_i = L Lᵀ` and check `A_i vᵢ = vᵢ`.
        let q = cache.row_dims[row];
        let l = cache.undamped_factor(row);
        let mut a_block = Array2::<f64>::zeros((q, q));
        for i in 0..q {
            for j in 0..q {
                let mut acc = 0.0_f64;
                for t in 0..=i.min(j) {
                    acc += l[[i, t]] * l[[j, t]];
                }
                a_block[[i, j]] = acc;
            }
        }
        for v in dirs {
            let av = a_block.dot(v);
            for slot in 0..q {
                max_unit_pin_error = max_unit_pin_error.max((av[slot] - v[slot]).abs());
            }
        }

        let (dense_vv, dense_vbeta) = solver
            .selected_inverse_row_blocks(row, &beta_inv)
            .expect("dense selected inverse row blocks");
        let (probe_vv, probe_vbeta) = row_selected_inverse_from_probes(
            &cache,
            row,
            &probes,
            &sinv,
            true,
            "#2712 reconstruction gate",
        )
        .expect("from-probes selected inverse row blocks");
        for (d, p) in dense_vv.iter().zip(probe_vv.iter()) {
            max_block_error = max_block_error.max((d - p).abs());
            block_scale = block_scale.max(d.abs());
        }
        for (d, p) in dense_vbeta.iter().zip(probe_vbeta.iter()) {
            max_block_error = max_block_error.max((d - p).abs());
            block_scale = block_scale.max(d.abs());
        }
    }
    assert!(
        deflated_rows > 0,
        "the certified anchor promised a deflated row and delivered none"
    );
    eprintln!(
        "#2712 reconstruction gate: {deflated_rows} deflated row(s), \
         max|A_i v - v| = {max_unit_pin_error:.6e}, \
         max|selected-inverse block difference| = {max_block_error:.6e} \
         against block magnitude {block_scale:.6e}"
    );
    assert!(
        max_unit_pin_error <= 1.0e-9,
        "`undamped_factor` must carry the CONDITIONED spectrum (A_i v = v on a \
         deflated direction); got max|A_i v - v| = {max_unit_pin_error:.6e}"
    );
    // RELATIVE, deliberately: a kept near-null eigendirection makes `inv_vv`
    // legitimately huge (measured `1.7e7` on this fixture), so an absolute bar
    // here would be a statement about the conditioning, not about the
    // reconstruction.
    assert!(
        max_block_error <= 1.0e-11 * (1.0 + block_scale),
        "the from-probes reconstruction must equal the dense selected inverse on a \
         DEFLATED row at full-basis probes; got {max_block_error:.6e} against block \
         magnitude {block_scale:.6e}"
    );
}

/// The declared `log λ_sparse` ladder for gates that need any state the
/// criterion will price as a maximum. `0.5` is the level these gates hard-coded
/// after the #1625 indefinite-basin diagnosis, so a tree on which that level
/// still works reproduces the historical anchor exactly; the rest climbs out of
/// the low-`ρ_sparse` basin the same diagnosis identified, then drops below it
/// for the fixtures whose maximum lies the other way.
const PD_BASIN_SPARSE_LIFTS: [f64; 8] = [0.5, 0.9, 1.3, 1.8, 2.4, 0.2, -0.2, -0.6];
