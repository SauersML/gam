//! Rung-2 two-block joint fit: behavior as a jointly-fitted output block that
//! shares the latent coordinate `t` and the gate `a` with the activation block.
//!
//! These tests exercise the REAL arrow-Schur joint fit on the AUGMENTED output
//! `[Z | √λ_y · Y]` built by [`BehaviorBlock`]. No new solver path is needed:
//! the augmented output shares `t` and `a` by construction, so the ordinary
//! joint fit reconstructs both blocks. The tests assert (1) the plumbing round
//! trips — the augmented target fits and `split_decoder` recovers the true
//! `[B_k | C_k]`; (2) both blocks are well reconstructed under one shared
//! coordinate; and (3) selection-for-mattering: an activation pattern whose
//! behavior does not vary earns a ≈ 0 behavior decoder while keeping its
//! activation decoder.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, BehaviorBlock, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    SphereTangentEmbedding, TwoBlockRemlControls, reconstruction_explained_variance,
};

const ON: f64 = 6.0;

/// Softmax of a logit vector (numerically stable) — the planted behavior law.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Explained variance of a column block `[c0, c1)` of a fitted augmented output
/// against the corresponding block of the (unscaled) target.
fn block_ev(target: &Array2<f64>, fitted: &Array2<f64>, c0: usize, c1: usize) -> f64 {
    let t = target.slice(ndarray::s![.., c0..c1]).to_owned();
    let f = fitted.slice(ndarray::s![.., c0..c1]).to_owned();
    reconstruction_explained_variance(t.view(), f.view())
        .unwrap_or_else(|| panic!("EV undefined for block [{c0},{c1})"))
}

/// Build a K=1 periodic (circle) atom at the AUGMENTED output width `p_tot`,
/// with cold (zero) decoders. Returns the atom and the shared coordinate block.
fn augmented_circle_atom(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> (SaeManifoldAtom, Array2<f64>) {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "b0",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    (atom, coords.clone())
}

/// Assemble a K=1 softmax term (single always-on atom) at augmented width.
fn build_k1(atom: SaeManifoldAtom, coord_block: Array2<f64>) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = coord_block.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), ON);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coord_block],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

/// Both blocks — activation AND behavior — are driven by ONE shared circle
/// coordinate, so a single-atom two-block fit must reconstruct BOTH, and
/// `split_decoder` must recover a behavior decoder that decodes back to the
/// planted distributions.
#[test]
fn two_block_joint_fit_reconstructs_activation_and_behavior() {
    let n = 60usize;
    let p_x = 4usize;
    let vocab = 4usize; // behavior tangent dim p_y = 3
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    // Planted activation: two channels are a clean cos/sin of the circle angle,
    // the rest zero — a genuine curved circle image in activation space.
    let mut z = Array2::<f64>::zeros((n, p_x));
    // Planted behavior distributions: a softmax whose logits rotate with θ, so
    // the next-token law moves smoothly along the SAME coordinate.
    let mut probs = Array2::<f64>::zeros((n, vocab));
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = (2.0 * theta).cos();
        let law = softmax(&[
            1.5 * theta.cos(),
            1.5 * theta.sin(),
            0.3 * (2.0 * theta).cos(),
            0.0,
        ]);
        for j in 0..vocab {
            probs[[i, j]] = law[j];
        }
    }

    let block = BehaviorBlock::fit(probs.view(), p_x, 0.0).unwrap();
    let p_y = block.behavior_dim();
    assert_eq!(p_y, vocab - 1);
    let p_tot = p_x + p_y;

    let augmented = block.augmented_target(z.view()).unwrap();
    assert_eq!(augmented.dim(), (n, p_tot));

    let (atom, cb) = augmented_circle_atom(&evaluator, &coords, p_tot);
    let (mut term, mut rho) = build_k1(atom, cb);
    term.set_behavior_block(block.clone()).unwrap();
    assert_eq!(term.activation_output_dim(), p_x);
    assert_eq!(term.behavior_output_range(), Some(p_x..p_tot));

    term.set_guards_enabled(false);
    term.run_joint_fit_arrow_schur(augmented.view(), &mut rho, None, 48, 1.0, 1e-6, 1e-6)
        .expect("two-block joint fit must complete");

    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    // Compare each block on the SAME (scaled) augmented target the fit saw.
    let ev_act = block_ev(&augmented, &fitted, 0, p_x);
    let ev_beh = block_ev(&augmented, &fitted, p_x, p_tot);
    assert!(ev_act > 0.9, "activation block EV too low: {ev_act}");
    assert!(ev_beh > 0.9, "behavior block EV too low: {ev_beh}");

    // split_decoder recovers a NON-trivial behavior decoder, and decoding the
    // fitted behavior reconstruction returns to the planted distributions.
    let (b_k, c_k) = block
        .split_decoder(term.atoms[0].decoder_coefficients.view())
        .unwrap();
    assert_eq!(b_k.dim().1, p_x);
    assert_eq!(c_k.dim().1, p_y);
    let c_norm = c_k.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        c_norm > 0.1,
        "behavior decoder collapsed to ~0 despite real behavior: {c_norm}"
    );

    // Decode the fitted behavior tangent at a few rows and compare KL to planted.
    let mut worst_kl = 0.0_f64;
    for &i in &[0usize, n / 4, n / 2, 3 * n / 4] {
        // Fitted behavior tangent in nats units = (augmented fitted behavior cols)/√λ_y.
        let inv = 1.0 / block.sqrt_lambda_y();
        let y_hat = Array1::from_shape_fn(p_y, |j| fitted[[i, p_x + j]] * inv);
        let p_hat = block.embedding.decode(y_hat.view()).unwrap();
        let kl =
            crate::manifold::SphereTangentEmbedding::exact_kl(probs.row(i), p_hat.view()).unwrap();
        worst_kl = worst_kl.max(kl);
    }
    assert!(
        worst_kl < 0.02,
        "decoded behavior diverges from planted (worst KL {worst_kl} nats)"
    );
}

/// Selection-for-mattering: when the behavior does NOT vary across rows (the
/// activation pattern has no behavioral correlate), the behavior target is
/// identically zero, so the fitted behavior decoder is ≈ 0 — the behavior block
/// contributes nothing — while the activation block is still reconstructed.
#[test]
fn constant_behavior_yields_zero_behavior_decoder() {
    let n = 48usize;
    let p_x = 3usize;
    let vocab = 4usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut probs = Array2::<f64>::zeros((n, vocab));
    let flat = softmax(&[0.2, 0.1, -0.1, 0.0]); // SAME law on every row
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        for j in 0..vocab {
            probs[[i, j]] = flat[j];
        }
    }

    let block = BehaviorBlock::fit(probs.view(), p_x, 0.0).unwrap();
    // Constant behavior ⇒ zero tangent target by construction.
    assert!(block.target.iter().all(|v| v.abs() < 1e-10));
    let p_y = block.behavior_dim();
    let p_tot = p_x + p_y;

    let augmented = block.augmented_target(z.view()).unwrap();
    let (atom, cb) = augmented_circle_atom(&evaluator, &coords, p_tot);
    let (mut term, mut rho) = build_k1(atom, cb);
    term.set_behavior_block(block.clone()).unwrap();
    term.set_guards_enabled(false);
    term.run_joint_fit_arrow_schur(augmented.view(), &mut rho, None, 48, 1.0, 1e-6, 1e-6)
        .expect("fit must complete");

    let (b_k, c_k) = block
        .split_decoder(term.atoms[0].decoder_coefficients.view())
        .unwrap();
    let b_norm = b_k.iter().map(|v| v * v).sum::<f64>().sqrt();
    let c_norm = c_k.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        b_norm > 0.1,
        "activation decoder should still fit the circle: {b_norm}"
    );
    assert!(
        c_norm < 1e-6,
        "constant behavior must earn a ~0 behavior decoder; got {c_norm}"
    );
}

/// Deterministic xorshift noise in `[-1, 1]` so the planted blocks carry a
/// known per-block noise scale without pulling in an RNG dependency.
fn noise_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map the top 53 bits to [-1, 1).
        ((state >> 11) as f64 / (1u64 << 52) as f64) - 1.0
    }
}

/// REML λ_y selection: plant a two-block circle whose activation block carries
/// noise σ_x and whose behavior tangent block carries noise σ_y (σ_x ≠ σ_y).
/// The variance-ratio fixed point must (1) converge, (2) land near the planted
/// ratio λ_y = σ_x²/σ_y², and (3) be invariant to the starting weight — the
/// weight is selected by the data, never a knob.
#[test]
fn reml_selects_lambda_y_at_planted_variance_ratio() {
    let n = 96usize;
    let p_x = 4usize;
    let vocab = 5usize; // p_y = 4
    let sigma_x = 0.20_f64;
    let sigma_y = 0.05_f64; // planted λ_y = (0.2/0.05)² = 16, log λ ≈ 2.77
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut probs = Array2::<f64>::zeros((n, vocab));
    let mut noise_x = noise_stream(0x5eed_0001);
    let mut noise_q = noise_stream(0x5eed_0002);
    // Build the CLEAN behavior first, embed it to get a chart, then add the
    // σ_y noise IN TANGENT COORDINATES (where the block's Gaussian model lives)
    // and decode back to distributions, so the planted behavior noise scale is
    // exactly σ_y in the units the fit sees.
    let mut clean_probs = Array2::<f64>::zeros((n, vocab));
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos() + sigma_x * noise_x();
        z[[i, 1]] = theta.sin() + sigma_x * noise_x();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos() + sigma_x * noise_x();
        z[[i, 3]] = sigma_x * noise_x();
        let law = softmax(&[
            1.2 * theta.cos(),
            1.2 * theta.sin(),
            0.4 * (2.0 * theta).sin(),
            0.2,
            0.0,
        ]);
        for j in 0..vocab {
            clean_probs[[i, j]] = law[j];
        }
    }
    let (clean_chart, clean_y) = SphereTangentEmbedding::fit(clean_probs.view()).unwrap();
    for i in 0..n {
        let mut y_row = clean_y.row(i).to_owned();
        for value in y_row.iter_mut() {
            *value += sigma_y * noise_q();
        }
        let p_noisy = clean_chart.decode(y_row.view()).unwrap();
        for j in 0..vocab {
            probs[[i, j]] = p_noisy[j];
        }
    }

    // Uniform noise on [-1,1] has variance 1/3 in BOTH blocks, so the planted
    // residual-variance ratio is σ_x²/σ_y² regardless of the common factor.
    let planted_log_lambda = (sigma_x / sigma_y).powi(2).ln();

    let mut selected = Vec::new();
    for &start_log_lambda in &[0.0_f64, 3.5] {
        let block = BehaviorBlock::fit(probs.view(), p_x, start_log_lambda).unwrap();
        let p_tot = p_x + block.behavior_dim();
        let (atom, cb) = augmented_circle_atom(&evaluator, &coords, p_tot);
        let (mut term, mut rho) = build_k1(atom, cb);
        term.set_behavior_block(block).unwrap();
        term.set_guards_enabled(false);
        let report = term
            .run_two_block_reml_fit(
                z.view(),
                &mut rho,
                None,
                TwoBlockRemlControls {
                    max_sweeps: 20,
                    inner_max_iter: 48,
                    step_size: 1.0,
                    ridge_ext_coord: 1e-6,
                    ridge_beta: 1e-6,
                    log_lambda_tol: 1e-3,
                },
            )
            .expect("two-block REML fit must complete");
        assert!(
            report.converged,
            "λ_y fixed point did not converge from start {start_log_lambda}: trajectory {:?}",
            report.log_lambda_trajectory
        );
        assert!(report.lambda_identifiable);
        selected.push(report.log_lambda_y);
    }
    // Starting-point invariance: both starts land on the same fixed point.
    assert!(
        (selected[0] - selected[1]).abs() < 0.05,
        "λ_y depends on its start: {selected:?}"
    );
    // And that fixed point is the planted variance ratio (log-scale tolerance
    // generous to finite-sample residual noise, tight enough to reject the
    // starts at 0.0 / 3.5 and any off-by-a-factor error).
    assert!(
        (selected[0] - planted_log_lambda).abs() < 0.6,
        "REML log λ_y = {} but planted ratio is {planted_log_lambda}",
        selected[0]
    );
}

/// The flagship identification payoff: gauge fixed BY DATA. Plant a circle
/// whose ACTIVATION image is reflection-symmetric — every activation channel
/// is an even function of θ, so rows at θ and −θ are bit-identical and
/// activation-only fitting CANNOT distinguish the two arcs: its fitted output
/// at a mirror pair is provably identical, and hence NO readout of that fit
/// can reproduce a behavior that differs across the mirror. The planted
/// BEHAVIOR carries an odd `sin θ` logit, so the mirror pairs are behaviorally
/// far apart. The two-block fit shares one latent `t` between the blocks, and
/// the behavior evidence must pin the reflection the activation left free.
#[test]
fn behavior_block_pins_reflection_gauge_that_activation_alone_cannot() {
    let n = 64usize;
    let p_x = 3usize;
    let vocab = 4usize; // p_y = 3
    // Order-4 harmonic basis (num_basis = 1 + 2·4 = 9). The planted behavior is a
    // softmax of harmonic-1 logits, and a softmax is NONLINEAR: its sphere-tangent
    // coordinates carry the full Bessel tail (harmonics 2,3,4,… with relative
    // amplitudes I_k(1.4)/I_0(1.4) ≈ 0.29, 0.044, 0.0080, …). An order-2 basis
    // (harmonics 1,2) truncates at harmonic 3, leaving an IRREDUCIBLE per-row KL
    // floor (~0.17 at the peak row) that no coupling weight λ_y can beat — the fit
    // would converge honestly yet miss the strict 10×-finer identification bar
    // purely for lack of decoder capacity. Order 4 captures the tail through
    // harmonic 4 (residual starts at harmonic 5, relative amplitude ≈ 0.0011), so
    // the representable reconstruction clears `worst_kl < 0.1·planted_sep` with
    // orders of magnitude to spare. NOTE the reflection-PINNING signal itself (the
    // odd `sin θ` logit) is harmonic 1 and was already representable at order 2 —
    // raising the order does not change the identification premise, only the
    // fidelity of the even Bessel harmonics the reconstruction is scored on.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(9).unwrap());
    // BOTH fits are seeded at the FOLDED coordinate min(i, n−i)/n — the best an
    // activation-only estimator could possibly recover, since it maps mirror
    // rows (identical activations) to one point. The two-block fit must then
    // pull the mirror pairs apart using behavior evidence alone.
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i.min(n - i) as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut probs = Array2::<f64>::zeros((n, vocab));
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        // Even functions of θ only: the activation cannot see the sign of θ.
        // z0..z2 are representable at order 4; the small `cos 7θ` term is a high
        // EVEN harmonic ABOVE the basis order (7 > 4) that the basis cannot absorb,
        // so it keeps the anchor residual R_x > 0 — hence the variance-ratio λ_y*
        // numerator stays positive and the coupling weight is not starved into the
        // near-degenerate fixed point a perfectly-representable anchor would force
        // (the same R_x→0 hazard the sibling isometry fixture dodges with its own
        // wiggle). `cos 7θ` is even, so mirror rows i and n−i stay BIT-identical
        // (the reflection-symmetry premise the test asserts), and harmonic 7 < the
        // Nyquist 32 on this 64-point grid, so it is a genuine unrepresented
        // residual, not an alias of a low harmonic.
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = (2.0 * theta).cos();
        z[[i, 2]] = 0.5 * (3.0 * theta).cos() + 0.05 * (7.0 * theta).cos();
        // The behavior DOES see it: an odd sin θ logit.
        let law = softmax(&[1.4 * theta.sin(), 0.8 * theta.cos(), 0.3, 0.0]);
        for j in 0..vocab {
            probs[[i, j]] = law[j];
        }
    }

    // Mirror pairs: rows i and n−i sit at θ and 2π−θ ≡ −θ with identical z.
    let mirror_pairs: Vec<(usize, usize)> = (1..n / 2).map(|i| (i, n - i)).collect();
    for &(a, b) in &mirror_pairs {
        for j in 0..p_x {
            assert!(
                (z[[a, j]] - z[[b, j]]).abs() < 1e-12,
                "planted mirror rows must have identical activations"
            );
        }
    }

    let block = BehaviorBlock::fit(probs.view(), p_x, 0.0).unwrap();
    let p_y = block.behavior_dim();
    let p_tot = p_x + p_y;

    // --- Activation-only fit: mirror rows are indistinguishable inputs, so the
    // fitted reconstruction collapses each pair — the misorientation is not a
    // solver artifact but an identifiability fact this fit exhibits.
    let (atom_a, cb_a) = augmented_circle_atom(&evaluator, &coords, p_x);
    let (mut term_a, mut rho_a) = build_k1(atom_a, cb_a);
    term_a.set_guards_enabled(false);
    term_a
        .run_joint_fit_arrow_schur(z.view(), &mut rho_a, None, 48, 1.0, 1e-6, 1e-6)
        .expect("activation-only fit must complete");
    let fitted_a = term_a.try_fitted_for_rho(&rho_a).unwrap();
    for &(a, b) in &mirror_pairs {
        for j in 0..p_x {
            assert!(
                (fitted_a[[a, j]] - fitted_a[[b, j]]).abs() < 1e-6,
                "activation-only fit should collapse mirror rows (identical inputs): rows \
                 {a}/{b} col {j} differ by {}",
                (fitted_a[[a, j]] - fitted_a[[b, j]]).abs()
            );
        }
    }

    // --- Two-block REML fit: behavior pins the reflection. ---
    let (atom_b, cb_b) = augmented_circle_atom(&evaluator, &coords, p_tot);
    let (mut term_b, mut rho_b) = build_k1(atom_b, cb_b);
    term_b.set_behavior_block(block).unwrap();
    term_b.set_guards_enabled(false);
    let report = term_b
        .run_two_block_reml_fit(
            z.view(),
            &mut rho_b,
            None,
            TwoBlockRemlControls {
                max_sweeps: 20,
                inner_max_iter: 48,
                step_size: 1.0,
                ridge_ext_coord: 1e-6,
                ridge_beta: 1e-6,
                log_lambda_tol: 1e-3,
            },
        )
        .expect("two-block fit must complete");
    assert!(report.lambda_identifiable);
    let block = term_b.behavior_block().unwrap().clone();

    let fitted_b = term_b.try_fitted_for_rho(&rho_b).unwrap();
    let inv = 1.0 / block.sqrt_lambda_y();

    // The planted mirror-pair behavioral separation, and the fitted model's
    // per-row behavioral error. The two-block fit must recover the ASYMMETRY
    // the activation-only fit provably cannot represent.
    let mut planted_sep_max = 0.0_f64;
    let mut worst_kl = 0.0_f64;
    let mut worst_row = 0usize;
    let mut sum_kl = 0.0_f64;
    let mut n_kl = 0usize;
    for &(a, b) in &mirror_pairs {
        let sep = SphereTangentEmbedding::exact_kl(probs.row(a), probs.row(b)).unwrap();
        planted_sep_max = planted_sep_max.max(sep);
        for &row in &[a, b] {
            let y_hat = Array1::from_shape_fn(p_y, |j| fitted_b[[row, p_x + j]] * inv);
            let p_hat = block.embedding.decode(y_hat.view()).unwrap();
            let kl = SphereTangentEmbedding::exact_kl(probs.row(row), p_hat.view()).unwrap();
            if kl > worst_kl {
                worst_kl = kl;
                worst_row = row;
            }
            sum_kl += kl;
            n_kl += 1;
        }
    }
    // Verification telemetry (#2015): the selected coupling weight and the
    // worst-row reconstruction. With the order-4 basis the representable behavior
    // clears the 10×-finer bar with large margin; a REML-selected finite λ_y and a
    // worst_kl far below `bar` is the expected line.
    eprintln!(
        "[#2015 reflection-gauge] log λ_y={:.6} (λ_y={:.4}), converged={}, sweeps={}, \
         worst_kl={worst_kl:.6} @row {worst_row}/{n}, mean_kl={:.6}, \
         planted_sep_max={planted_sep_max:.6}, bar=0.1·sep={:.6}",
        report.log_lambda_y,
        report.log_lambda_y.exp(),
        report.converged,
        report.sweeps,
        sum_kl / n_kl as f64,
        0.1 * planted_sep_max,
    );
    // The planted behavior really is strongly asymmetric across the mirror...
    assert!(
        planted_sep_max > 0.5,
        "planted mirror separation too weak to test anything: {planted_sep_max}"
    );
    // ...and the two-block fit reproduces per-row behavior far more finely than
    // that separation — i.e. it distinguishes the arcs the activation collapses.
    assert!(
        worst_kl < 0.1 * planted_sep_max,
        "two-block fit failed to pin the reflection: worst per-row KL {worst_kl} vs planted \
         mirror separation {planted_sep_max}"
    );
}
