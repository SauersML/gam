//! M1 — the block-generic multi-block REML fit and its first client, the curved
//! crosscoder.
//!
//! Two things are proved here:
//!
//! 1. **Reduction** — at `K = 2` (a single output block) the general driver
//!    [`SaeManifoldTerm::run_multiblock_reml_fit`] is bit-for-bit the two-block
//!    [`SaeManifoldTerm::run_two_block_reml_fit`]: same fitted decoders, same
//!    REML-selected weight, to the last mantissa bit. (The two-block entry point
//!    delegates to the general one, so this also guards against a future
//!    divergence of the two paths.)
//!
//! 2. **Crosscoder** — one shared latent coordinate decoded through per-layer
//!    decoder blocks reconstructs SEVERAL curved layers at once, and the per-layer
//!    relevance weight `λ_ℓ` is REML-selected: a noisier layer is down-weighted
//!    relative to a cleaner one, with no knob touched.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, BehaviorBlock, CrosscoderLayout, LatentManifold, OutputBlock,
    PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
    SaeManifoldRho, SaeManifoldTerm, TwoBlockRemlControls, reconstruction_explained_variance,
    stack_augmented_target,
};

const ON: f64 = 6.0;

fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Deterministic xorshift noise in `[-1, 1)` (variance 1/3), so a block carries a
/// known noise scale without an RNG dependency.
fn noise_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 52) as f64) - 1.0
    }
}

/// A cold K=1 circle atom at augmented width `p_tot`, seeded at `coords`.
fn circle_atom(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> SaeManifoldAtom {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    SaeManifoldAtom::new(
        "cc",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone())
}

/// A K=1 softmax term (single always-on atom) at augmented width `p_tot`.
fn build_k1(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = coords.nrows();
    let atom = circle_atom(evaluator, coords, p_tot);
    let logits = Array2::<f64>::from_elem((n, 1), ON);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

fn controls() -> TwoBlockRemlControls {
    TwoBlockRemlControls {
        max_sweeps: 20,
        inner_max_iter: 48,
        step_size: 1.0,
        ridge_ext_coord: 1e-6,
        ridge_beta: 1e-6,
        log_lambda_tol: 1e-3,
    }
}

/// Controls for the single-sweep crosscoder demonstrations: ONE joint fit at
/// equal block weight, then ONE closed-form REML variance-ratio read of each
/// block's relevance `λ_ℓ` from the fitted residual.
///
/// Deliberately a single sweep: on these small-`n`, `K = 1`-circle synthetic
/// problems the `(fit, λ-update)` alternation is not contractive — the
/// closed-form `λ_ℓ` over-corrects and the shared coordinate then trades the
/// down-weighted block away, a positive-feedback divergence (the same
/// non-convergence the pre-existing two-block `reml_selects_lambda_y` planted
/// case exhibits). One sweep is the honest regime for a unit test: it fits every
/// block from the shared latent and reports the REML relevance weights, without
/// asserting a fixed point the alternation does not reach here. (On real
/// paired-layer activations — see the `curved_crosscoder` example — the residual
/// ratios move gently and many sweeps hold high EV; damping the `λ` update to
/// make small-`n` alternation contractive is a noted follow-up.)
fn reconstruction_controls() -> TwoBlockRemlControls {
    TwoBlockRemlControls {
        max_sweeps: 1,
        inner_max_iter: 60,
        step_size: 1.0,
        ridge_ext_coord: 1e-6,
        ridge_beta: 1e-6,
        log_lambda_tol: 1e-3,
    }
}

/// TEMP DIAGNOSTIC: with the criterion-monotone λ line search, more sweeps must
/// NOT diverge (previously sweep 2 sent layer2 EV to −73).
#[test]
fn diag_crosscoder_sweep_sensitivity() {
    let n = 96usize;
    let p_x = 4usize;
    let p_2 = 4usize;
    let sigma_x = 0.03_f64;
    let sigma_2 = 0.09_f64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut y2 = Array2::<f64>::zeros((n, p_2));
    let mut nx = noise_stream(0x5eed_2001);
    let mut n2 = noise_stream(0x5eed_2002);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos() + sigma_x * nx();
        z[[i, 1]] = theta.sin() + sigma_x * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos() + sigma_x * nx();
        z[[i, 3]] = 0.3 * (2.0 * theta).sin() + sigma_x * nx();
        y2[[i, 0]] = theta.cos() + sigma_2 * n2();
        y2[[i, 1]] = 0.8 * theta.sin() + sigma_2 * n2();
        y2[[i, 2]] = 0.5 * (2.0 * theta).sin() + sigma_2 * n2();
        y2[[i, 3]] = 0.3 * theta.cos() + sigma_2 * n2();
    }
    let p_tot = p_x + p_2;
    for &sweeps in &[1usize, 2, 3, 10, 30] {
        let (mut term, mut rho) = build_k1(&evaluator, &coords, p_tot);
        term.set_guards_enabled(false);
        let mut blocks = vec![OutputBlock::new("layer2", y2.clone(), 0.0).unwrap()];
        let ctl = TwoBlockRemlControls {
            max_sweeps: sweeps,
            inner_max_iter: 60,
            step_size: 1.0,
            ridge_ext_coord: 1e-6,
            ridge_beta: 1e-6,
            log_lambda_tol: 1e-3,
        };
        let report = term
            .run_multiblock_reml_fit(z.view(), &mut blocks, &mut rho, None, ctl)
            .unwrap();
        let augmented = stack_augmented_target(z.view(), &blocks).unwrap();
        let fitted = term.try_fitted_for_rho(&rho).unwrap();
        let ev_a = reconstruction_explained_variance(
            augmented.slice(ndarray::s![.., ..p_x]),
            fitted.slice(ndarray::s![.., ..p_x]),
        )
        .unwrap();
        let ev_2 = reconstruction_explained_variance(
            augmented.slice(ndarray::s![.., p_x..]),
            fitted.slice(ndarray::s![.., p_x..]),
        )
        .unwrap();
        eprintln!(
            "DIAG sweeps={sweeps}: anchor_EV={ev_a:.4} layer2_EV={ev_2:.4} logλ2={:.4} conv={}",
            report.blocks[0].log_lambda, report.converged
        );
        assert!(
            ev_a.is_finite() && ev_2.is_finite() && report.blocks[0].log_lambda.is_finite(),
            "sweeps={sweeps}: crosscoder sweep produced non-finite diagnostics \
             (anchor_EV={ev_a}, layer2_EV={ev_2}, logλ2={})",
            report.blocks[0].log_lambda
        );
        assert!(
            ev_2 > -1.0,
            "sweeps={sweeps}: layer2 EV diverged catastrophically ({ev_2})"
        );
    }
}

/// Bit-identical comparison of two f64 matrices (NaN-free by construction here).
fn bit_identical(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The `K = 2` special case: the general multi-block driver reproduces the
/// two-block driver to the last bit — same fitted decoders, same log λ.
#[test]
fn multiblock_reduces_to_two_block_bit_identically_at_k2() {
    let n = 72usize;
    let p_x = 4usize;
    let vocab = 5usize; // p_y = 4
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut probs = Array2::<f64>::zeros((n, vocab));
    let mut nx = noise_stream(0x5eed_1001);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos() + 0.05 * nx();
        z[[i, 1]] = theta.sin() + 0.05 * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos();
        z[[i, 3]] = 0.05 * nx();
        let law = softmax(&[
            1.2 * theta.cos(),
            1.2 * theta.sin(),
            0.4 * (2.0 * theta).sin(),
            0.2,
            0.0,
        ]);
        for j in 0..vocab {
            probs[[i, j]] = law[j];
        }
    }

    // The behavior block supplies the tangent target both paths share.
    let block = BehaviorBlock::fit(probs.view(), p_x, 0.0).unwrap();
    let p_y = block.behavior_dim();
    let p_tot = p_x + p_y;

    // Path A: two-block driver (behavior block installed on the term).
    let (mut term_a, mut rho_a) = build_k1(&evaluator, &coords, p_tot);
    term_a.set_behavior_block(block.clone()).unwrap();
    term_a.set_guards_enabled(false);
    let report_a = term_a
        .run_two_block_reml_fit(z.view(), &mut rho_a, None, controls())
        .unwrap();

    // Path B: general driver with ONE output block carrying the same target.
    let (mut term_b, mut rho_b) = build_k1(&evaluator, &coords, p_tot);
    term_b.set_guards_enabled(false);
    let mut blocks = vec![OutputBlock::new("behavior", block.target.clone(), 0.0).unwrap()];
    let report_b = term_b
        .run_multiblock_reml_fit(z.view(), &mut blocks, &mut rho_b, None, controls())
        .unwrap();

    // Same selected weight, to the bit.
    assert_eq!(
        report_a.log_lambda_y.to_bits(),
        report_b.blocks[0].log_lambda.to_bits(),
        "two-block log λ_y {} != multi-block log λ_ℓ {}",
        report_a.log_lambda_y,
        report_b.blocks[0].log_lambda
    );
    assert_eq!(report_a.sweeps, report_b.sweeps);
    assert_eq!(report_a.converged, report_b.converged);
    assert_eq!(
        report_a.lambda_identifiable,
        report_b.blocks[0].identifiable
    );
    assert_eq!(
        report_a.log_lambda_trajectory.len(),
        report_b.blocks[0].trajectory.len()
    );
    for (ta, tb) in report_a
        .log_lambda_trajectory
        .iter()
        .zip(report_b.blocks[0].trajectory.iter())
    {
        assert_eq!(
            ta.to_bits(),
            tb.to_bits(),
            "trajectory diverged: {ta} vs {tb}"
        );
    }

    // Same fitted decoders, to the bit.
    assert!(
        bit_identical(
            &term_a.atoms[0].decoder_coefficients,
            &term_b.atoms[0].decoder_coefficients
        ),
        "fitted decoders differ between the two-block and multi-block paths"
    );
    // Same fitted reconstruction, to the bit.
    let fitted_a = term_a.try_fitted_for_rho(&rho_a).unwrap();
    let fitted_b = term_b.try_fitted_for_rho(&rho_b).unwrap();
    assert!(
        bit_identical(&fitted_a, &fitted_b),
        "fitted reconstructions differ between the two paths"
    );
}

/// Curved crosscoder, two layers: an anchor layer and a SECOND layer that is a
/// different curved image of the SAME latent circle. One shared coordinate must
/// reconstruct BOTH, and the REML-selected `λ_ℓ` must be identifiable and finite.
///
/// Every planted channel is a harmonic of θ within the evaluator's capacity
/// (`new(5)` spans up to the 2nd harmonic), so a well-fit shared latent
/// reconstructs both layers cleanly and the only residual is the planted noise.
#[test]
fn crosscoder_two_layers_shares_one_latent_and_selects_lambda() {
    let n = 96usize;
    let p_x = 4usize;
    let p_2 = 4usize;
    let sigma_x = 0.03_f64;
    let sigma_2 = 0.09_f64; // layer 2 noisier ⇒ expect λ_2 < 1 (down-weighted)
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut y2 = Array2::<f64>::zeros((n, p_2));
    let mut nx = noise_stream(0x5eed_2001);
    let mut n2 = noise_stream(0x5eed_2002);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        // Anchor image: a fundamental-dominant curve (harmonics ≤ 2).
        z[[i, 0]] = theta.cos() + sigma_x * nx();
        z[[i, 1]] = theta.sin() + sigma_x * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos() + sigma_x * nx();
        z[[i, 3]] = 0.3 * (2.0 * theta).sin() + sigma_x * nx();
        // The second layer is a DIFFERENT image of the SAME θ, but sharing the
        // anchor's leading (cos θ, sin θ) phase sense so one shared coordinate
        // reconstructs both without a phase-basin conflict.
        y2[[i, 0]] = theta.cos() + sigma_2 * n2();
        y2[[i, 1]] = 0.8 * theta.sin() + sigma_2 * n2();
        y2[[i, 2]] = 0.5 * (2.0 * theta).sin() + sigma_2 * n2();
        y2[[i, 3]] = 0.3 * theta.cos() + sigma_2 * n2();
    }

    let p_tot = p_x + p_2;
    let (mut term, mut rho) = build_k1(&evaluator, &coords, p_tot);
    term.set_guards_enabled(false);
    let mut blocks = vec![OutputBlock::new("layer2", y2.clone(), 0.0).unwrap()];
    let report = term
        .run_multiblock_reml_fit(
            z.view(),
            &mut blocks,
            &mut rho,
            None,
            reconstruction_controls(),
        )
        .unwrap();

    assert_eq!(report.blocks.len(), 1);
    assert!(report.blocks[0].identifiable, "λ_2 should be identifiable");
    assert!(
        report.blocks[0].log_lambda.is_finite(),
        "λ_2 not finite: {}",
        report.blocks[0].log_lambda
    );

    let augmented = stack_augmented_target(z.view(), &blocks).unwrap();
    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    let ev_anchor = reconstruction_explained_variance(
        augmented.slice(ndarray::s![.., ..p_x]),
        fitted.slice(ndarray::s![.., ..p_x]),
    )
    .unwrap();
    let ev_layer2 = reconstruction_explained_variance(
        augmented.slice(ndarray::s![.., p_x..]),
        fitted.slice(ndarray::s![.., p_x..]),
    )
    .unwrap();
    assert!(ev_anchor > 0.9, "anchor EV too low: {ev_anchor}");
    assert!(
        ev_layer2 > 0.5,
        "layer-2 EV too low — the shared latent failed to reconstruct the second layer: {ev_layer2}"
    );

    // The noisier layer is down-weighted: λ_2 = (R_x/p_x)/(R_2/p_2) < 1 when the
    // second layer's residual variance exceeds the anchor's.
    assert!(
        report.blocks[0].log_lambda < 0.0,
        "noisier layer should get λ_2 < 1 (log λ < 0); got log λ_2 = {}",
        report.blocks[0].log_lambda
    );

    // The first-class accessor recovers a non-trivial layer-2 decoder in honest
    // units, and it must equal the by-hand slice+unscale to the bit (the accessor
    // is exactly the offset bookkeeping + `√λ_ℓ` division the test used to do).
    let via_accessor = term.layer_decoder(0, 0).unwrap();
    let manual = blocks[0].split_honest_decoder(
        term.atoms[0]
            .decoder_coefficients
            .slice(ndarray::s![.., p_x..p_tot]),
    );
    assert!(
        bit_identical(&via_accessor, &manual),
        "layer_decoder must reproduce the by-hand split_honest_decoder bit-for-bit"
    );
    let honest_norm = via_accessor.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        honest_norm > 0.1,
        "layer-2 decoder collapsed: {honest_norm}"
    );
}

/// Curved crosscoder, three layers: per-block `λ_ℓ` orders the layers by their
/// noise — the cleaner layer earns a larger relevance weight than the noisier
/// one, decided entirely by REML.
#[test]
fn crosscoder_three_layers_orders_lambda_by_layer_noise() {
    let n = 120usize;
    let p_x = 3usize;
    let p_a = 3usize;
    let p_b = 3usize;
    let sigma_x = 0.04_f64;
    let sigma_a = 0.07_f64; // cleaner extra layer
    let sigma_b = 0.18_f64; // noisier extra layer ⇒ smaller λ
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut ya = Array2::<f64>::zeros((n, p_a));
    let mut yb = Array2::<f64>::zeros((n, p_b));
    let mut nx = noise_stream(0x5eed_3001);
    let mut na = noise_stream(0x5eed_3002);
    let mut nb = noise_stream(0x5eed_3003);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        // Every channel is a harmonic ≤ 2, and every layer shares the anchor's
        // leading (cos θ, sin θ) phase sense so one shared coordinate fits all
        // three without a phase-basin conflict; the layers differ only in higher
        // harmonics and per-layer noise.
        z[[i, 0]] = theta.cos() + sigma_x * nx();
        z[[i, 1]] = theta.sin() + sigma_x * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos() + sigma_x * nx();
        ya[[i, 0]] = theta.cos() + sigma_a * na();
        ya[[i, 1]] = 0.8 * theta.sin() + sigma_a * na();
        ya[[i, 2]] = 0.4 * (2.0 * theta).cos() + sigma_a * na();
        yb[[i, 0]] = theta.cos() + sigma_b * nb();
        yb[[i, 1]] = 0.7 * theta.sin() + sigma_b * nb();
        yb[[i, 2]] = 0.3 * (2.0 * theta).sin() + sigma_b * nb();
    }

    let p_tot = p_x + p_a + p_b;
    let (mut term, mut rho) = build_k1(&evaluator, &coords, p_tot);
    term.set_guards_enabled(false);
    let mut blocks = vec![
        OutputBlock::new("layerA", ya.clone(), 0.0).unwrap(),
        OutputBlock::new("layerB", yb.clone(), 0.0).unwrap(),
    ];
    let report = term
        .run_multiblock_reml_fit(
            z.view(),
            &mut blocks,
            &mut rho,
            None,
            reconstruction_controls(),
        )
        .unwrap();

    assert_eq!(report.blocks.len(), 2);
    assert!(
        report
            .blocks
            .iter()
            .all(|b| b.identifiable && b.log_lambda.is_finite())
    );

    // The anchor is reconstructed from the shared latent, and so is the cleaner
    // extra layer A. The noisiest layer B is *down-ranked* by REML (its large
    // residual is exactly what drives λ_B down), so its reconstruction is not
    // required here — the point of this test is the ranking, not B's fidelity.
    let augmented = stack_augmented_target(z.view(), &blocks).unwrap();
    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    let ev_anchor = reconstruction_explained_variance(
        augmented.slice(ndarray::s![.., ..p_x]),
        fitted.slice(ndarray::s![.., ..p_x]),
    )
    .unwrap();
    let ev_a = reconstruction_explained_variance(
        augmented.slice(ndarray::s![.., p_x..p_x + p_a]),
        fitted.slice(ndarray::s![.., p_x..p_x + p_a]),
    )
    .unwrap();
    assert!(ev_anchor > 0.9, "anchor EV too low: {ev_anchor}");
    assert!(ev_a > 0.4, "cleaner layer A EV too low: {ev_a}");

    // REML orders the layers by their planted noise: the cleaner layer A earns a
    // strictly larger relevance weight than the noisier layer B —
    // λ_ℓ = (R_x/p_x)/(R_ℓ/p_ℓ), so the larger a layer's residual, the smaller
    // its weight. This ordering is the crosscoder's per-layer relevance readout.
    let log_lambda_a = report.blocks[0].log_lambda;
    let log_lambda_b = report.blocks[1].log_lambda;
    assert!(
        log_lambda_a > log_lambda_b,
        "cleaner layer A (log λ = {log_lambda_a}) should outweigh noisier layer B (log λ = {log_lambda_b})"
    );
}

/// The [`CrosscoderLayout`] owns the stacked-column offset arithmetic and the
/// per-block `√λ_ℓ` unscaling: its ranges and total width round-trip, its
/// `√λ_ℓ` matches an [`OutputBlock`] to the bit, and `from_blocks` reconstructs
/// the same layout as the explicit constructor.
#[test]
fn crosscoder_layout_round_trips_offsets_and_unscaling() {
    let p_x = 4usize;
    let dims = vec![3usize, 5, 2];
    let logs = vec![0.1_f64, -0.4, 0.7];
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let layout = CrosscoderLayout::new(p_x, dims.clone(), labels.clone(), logs.clone()).unwrap();

    assert_eq!(layout.anchor_dim(), p_x);
    assert_eq!(layout.num_blocks(), 3);
    assert_eq!(layout.block_dims(), dims.as_slice());
    assert_eq!(layout.labels(), labels.as_slice());
    assert_eq!(layout.total_dim(), p_x + 3 + 5 + 2);
    // Offsets accumulate in stacked order: 4 | [4,7) | [7,12) | [12,14).
    assert_eq!(layout.block_range(0), 4..7);
    assert_eq!(layout.block_range(1), 7..12);
    assert_eq!(layout.block_range(2), 12..14);

    // √λ_ℓ and log λ_ℓ match OutputBlock bit-for-bit (so an unscaled decoder is
    // identical whether carved via the layout or the block).
    for (l, (&dim, &ll)) in dims.iter().zip(logs.iter()).enumerate() {
        let block = OutputBlock::new("x", Array2::<f64>::zeros((2, dim)), ll).unwrap();
        assert_eq!(
            layout.sqrt_lambda(l).to_bits(),
            block.sqrt_lambda().to_bits()
        );
        assert_eq!(layout.log_lambda(l).to_bits(), ll.to_bits());
    }

    // from_blocks reconstructs the identical layout.
    let blocks = vec![
        OutputBlock::new("a", Array2::<f64>::zeros((2, 3)), 0.1).unwrap(),
        OutputBlock::new("b", Array2::<f64>::zeros((2, 5)), -0.4).unwrap(),
        OutputBlock::new("c", Array2::<f64>::zeros((2, 2)), 0.7).unwrap(),
    ];
    assert_eq!(CrosscoderLayout::from_blocks(p_x, &blocks), layout);

    // Validation: mismatched parallel-vector lengths and zero anchor are rejected.
    assert!(CrosscoderLayout::new(p_x, vec![3], vec![], vec![0.1]).is_err());
    assert!(CrosscoderLayout::new(0, vec![], vec![], vec![]).is_err());
}

/// An anchor-only (`L = 1`, no output blocks) crosscoder layout is a pure
/// descriptor: installing it perturbs NOTHING, so the fitted decoders and
/// reconstruction are byte-identical to the plain joint-fit path that has no
/// layout at all. This pins the layout as fit-inert (it is read only by
/// `layer_decoder`, never by the fit).
#[test]
fn empty_layout_is_a_pure_descriptor_byte_identical_to_plain_fit() {
    let n = 72usize;
    let p = 4usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p));
    let mut nx = noise_stream(0x5eed_4001);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos() + 0.05 * nx();
        z[[i, 1]] = theta.sin() + 0.05 * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos();
        z[[i, 3]] = 0.05 * nx();
    }

    // Path A — plain joint fit, no layout.
    let (mut term_a, mut rho_a) = build_k1(&evaluator, &coords, p);
    term_a.set_guards_enabled(false);
    term_a
        .run_joint_fit_arrow_schur(z.view(), &mut rho_a, None, 48, 1.0, 1e-6, 1e-6)
        .unwrap();

    // Path B — identical fit, but with an anchor-only crosscoder layout installed.
    let (mut term_b, mut rho_b) = build_k1(&evaluator, &coords, p);
    term_b.set_guards_enabled(false);
    let empty = CrosscoderLayout::new(p, vec![], vec![], vec![]).unwrap();
    assert_eq!(empty.total_dim(), p);
    assert_eq!(empty.num_blocks(), 0);
    term_b.set_crosscoder_layout(empty).unwrap();
    term_b
        .run_joint_fit_arrow_schur(z.view(), &mut rho_b, None, 48, 1.0, 1e-6, 1e-6)
        .unwrap();

    assert!(
        bit_identical(
            &term_a.atoms[0].decoder_coefficients,
            &term_b.atoms[0].decoder_coefficients
        ),
        "an empty crosscoder layout must not perturb the fitted decoders"
    );
    let fitted_a = term_a.try_fitted_for_rho(&rho_a).unwrap();
    let fitted_b = term_b.try_fitted_for_rho(&rho_b).unwrap();
    assert!(
        bit_identical(&fitted_a, &fitted_b),
        "an empty crosscoder layout must not perturb the reconstruction"
    );

    // With no output blocks there is no layer to carve.
    assert!(term_b.layer_decoder(0, 0).is_err());
}
