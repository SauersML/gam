//! Owed-work regression gate for #1033 (n-independent SAE outer ρ-loop:
//! hyperparameter search must touch only k-dim objects).
//!
//! ## What #1033 asks, and what is actually achievable
//!
//! #1033 wants the SAE-manifold outer hyperparameter (ρ) search reformulated so
//! that each outer evaluation consumes only k-dim *sufficient statistics* (k =
//! the number of atoms / latent axes), with the n-sized (row-count) work built
//! ONCE in O(n) — making the per-evaluation cost independent of n.
//!
//! The honest architectural finding (the primary deliverable of #1033) is that
//! the STRICT form of that contract — "the entire ρ-search reads only k-dim
//! objects" — is NOT achievable for the current SAE architecture without
//! introducing an approximation the code does not currently make. Every O(n)
//! touch inside one outer evaluation is *ρ-dependent*, not ρ-invariant, because
//! the outer search drives the NONLINEAR inner joint fit (`reml_criterion*` →
//! `assemble_arrow_schur`, the inner Newton solve). For one ρ evaluation the
//! n-sized work is, and each piece DEPENDS on ρ:
//!
//!   * per-row assignment gates `a_k = a_k(ρ.lambda_sparse)` (softmax / JumpReLU
//!     / IBP-MAP temperatures), and hence the per-row active-set selection;
//!   * per-row reconstruction `fitted = Σ_k a_k · B_k φ_k(t_k)`;
//!   * per-row residuals and any whitening;
//!   * per-row logit + coordinate Jacobian rows;
//!   * the data-fit Gram accumulation `G` (sparsity pattern + magnitudes both
//!     scale with the ρ-dependent gates).
//!
//! All of these flow from the gates `a_k(ρ)` AND from the inner solve's
//! converged `(coords, logits, β)`, which themselves move with ρ. The ONLY
//! ρ-invariant O(n) quantities are the target matrix and the per-row loss
//! weights, and those are already constant across the search. So one cannot
//! build a k-dim sufficient statistic once in O(n) and then make the WHOLE
//! ρ-search read only k-dim objects: the inner solve is a nonlinear fit whose
//! per-row state genuinely changes with ρ. A flat "outer-eval wall-time is
//! independent of n" contract would therefore be FALSE, and this test does NOT
//! assert it.
//!
//! ## What genuinely IS n-independent, and is pinned here
//!
//! The half of #1033 that is mathematically sound is already in place: the
//! objects the outer search CONSUMES are k-dim, not n-dim. The outer-ρ gradient
//! — the per-evaluation payload the optimizer steps on — has length
//! `1 + K + Σ_k d_k` (the shared sparse log-strength, the K per-atom smoothness
//! log-strengths — `log_lambda_smooth[k]`, one per atom since #1556 — plus the
//! per-atom per-axis ARD precisions), independent of the row count n. Each of
//! its four analytic
//! channels (`explicit`, `logdet_trace`, `occam`, `third_order_correction`) is a
//! k-dim `Array1`. The k-dim CONSUMPTION is exactly what makes the search scale
//! in the hyperparameter dimension and not in n — even though BUILDING each
//! channel still costs O(n) per ρ (the part that is not removable here).
//!
//! This test pins that invariant: build the SAME two-atom term — identical
//! atoms, decoders, ARD layout, and ρ — over two DIFFERENT row counts, run the
//! public `reml_criterion_with_cache`, take the public
//! `analytic_outer_rho_gradient_at_converged`, and assert every gradient channel
//! has the k-dim length `1 + K + Σ_k d_k` for BOTH n, and that the assembled
//! gradient vector has that same n-invariant length. A regression that routed an
//! n-sized object into the outer-search payload (re-introducing an n-dimensional
//! coordinate the optimizer would have to walk) would change one of these
//! lengths and fail here. The ρ flat-coordinate round trip (`to_flat` /
//! `from_flat`) is pinned n-invariant for the same reason: the search space the
//! engine optimizes over is k-dim.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::terms::latent::LatentManifold;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};

const M: usize = 3; // periodic basis: [const, sin, cos]
const P: usize = 3; // output channels
const D: usize = 1; // latent dim per atom (circle)
const K: usize = 2; // atoms

/// k-dim outer-coordinate length: the shared sparse log-strength
/// (`log_lambda_sparse`), the K per-atom smoothness log-strengths
/// (`log_lambda_smooth[k]`, one per atom — #1556), plus one ARD precision per
/// atom per latent axis. Independent of the row count n — that is the property
/// under test.
const RHO_FLAT_LEN: usize = 1 + K + K * D;

/// Build a two-atom periodic circle SAE term over `n` rows, with a fixed decoder
/// and deterministic coordinates / logits / target. The ONLY thing that varies
/// across calls is `n` (the row count); the atom shapes, decoders, ARD layout,
/// and basis are held identical so the k-dim outer-coordinate structure is the
/// same and only the n-sized inner work changes.
fn build_term_with_n_rows(n: usize) -> (SaeManifoldTerm, Array2<f64>) {
    // Deterministic, distinct circle coordinates spread over the period.
    let coords0 = Array2::from_shape_fn((n, D), |(i, _)| {
        (0.05 + 0.0137 * (i as f64)).rem_euclid(1.0)
    });
    let coords1 = Array2::from_shape_fn((n, D), |(i, _)| {
        (0.15 + 0.0191 * (i as f64)).rem_euclid(1.0)
    });

    let eval = PeriodicHarmonicEvaluator::new(M).unwrap();
    let (phi0, jet0) = eval.evaluate(coords0.view()).unwrap();
    let (phi1, jet1) = eval.evaluate(coords1.view()).unwrap();

    // A non-degenerate decoder so the inner data-fit term is genuinely active.
    let mut dec0 = Array2::<f64>::zeros((M, P));
    dec0[[0, 0]] = 0.5; // const -> ch0
    dec0[[1, 1]] = 1.0; // sin   -> ch1
    dec0[[2, 2]] = 1.0; // cos   -> ch2
    let mut dec1 = Array2::<f64>::zeros((M, P));
    dec1[[0, 1]] = 0.5; // const -> ch1
    dec1[[1, 2]] = 1.0; // sin   -> ch2
    dec1[[2, 0]] = 1.0; // cos   -> ch0

    let make = |name: &str, phi: Array2<f64>, jet, dec: Array2<f64>| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            D,
            phi,
            jet,
            dec,
            Array2::<f64>::eye(M),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()))
    };
    let atom0 = make("circle_0", phi0, jet0, dec0);
    let atom1 = make("circle_1", phi1, jet1, dec1);

    // Mild, non-degenerate logits so both atoms are routed-on.
    let logits = Array2::from_shape_fn((n, K), |(i, k)| {
        0.3 + 0.05 * (i as f64).sin() - 0.05 * (k as f64)
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0.clone(), coords1.clone()],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.5, 1.0, false),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();

    // A reconstruction target that is itself the amplitude-1 decode of atom 0
    // plus mild structured noise, so the data-fit term is non-trivial and the
    // converged inner solve (and hence the outer gradient) is well defined.
    let target =
        Array2::from_shape_fn((n, P), |(i, c)| 0.1 * ((i as f64) * 0.3 + (c as f64)).sin());
    (term, target)
}

/// The ARD-enabled ρ: one log-precision per atom per latent axis, plus the two
/// shared log-strengths. Identical k-dim layout for every n.
fn ard_rho() -> SaeManifoldRho {
    SaeManifoldRho::new(
        (1.0e-2_f64).ln(),
        (1.0e-2_f64).ln(),
        vec![Array1::<f64>::from_elem(D, (1.0e-1_f64).ln()); K],
    )
}

/// The ρ flat-coordinate space the outer engine optimizes over is k-dim, not
/// n-dim: `to_flat` / `from_flat` round-trip at length `1 + K + Σ_k d_k`
/// independent of the row count. (Pure ρ-structure property — no inner solve.)
#[test]
fn rho_flat_coordinate_space_is_n_invariant_1033() {
    let rho = ard_rho();
    let flat = rho.to_flat();
    assert_eq!(
        flat.len(),
        RHO_FLAT_LEN,
        "ρ flat coordinate length must be the k-dim 1 + K + Σ d_k = {RHO_FLAT_LEN}, the outer \
         search space dimension; got {}",
        flat.len()
    );
    // Round-trip is the exact inverse and stays k-dim.
    let back = rho.from_flat(flat.view());
    assert_eq!(back.to_flat().len(), RHO_FLAT_LEN);
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "k-dim outer coordinate must be finite"
    );
}

/// The outer-ρ GRADIENT — the object the hyperparameter search consumes on every
/// outer evaluation — is k-dim (`1 + K + Σ_k d_k`) and that length is invariant to
/// the row count n. Each of its four analytic channels is a k-dim `Array1`. This
/// is the genuinely-achievable half of #1033: the search CONSUMES only k-dim
/// objects even though BUILDING them still costs O(n) per ρ (the inner solve is
/// nonlinear in ρ; see the module doc for why the n-sized build is not
/// removable). Building the SAME term at two different n and asserting identical
/// k-dim gradient shapes pins that no n-sized object leaks into the outer-search
/// payload.
#[test]
fn outer_rho_gradient_is_k_dim_and_n_invariant_1033() {
    // Two genuinely different row counts; everything else identical.
    let n_small = 24usize;
    let n_large = 240usize;

    let rho = ard_rho();
    // Inner-solve knobs mirroring the production outer objective's defaults.
    let inner_max_iter = 64usize;
    let learning_rate = 1.0;
    let ridge_ext_coord = 1.0e-8;
    let ridge_beta = 1.0e-8;

    let mut channel_lens: Vec<[usize; 4]> = Vec::with_capacity(2);
    let mut grad_lens: Vec<usize> = Vec::with_capacity(2);

    for &n in &[n_small, n_large] {
        let (mut term, target) = build_term_with_n_rows(n);
        assert_eq!(term.n_obs(), n, "term must carry exactly n={n} rows");
        assert_eq!(term.k_atoms(), K);

        // Public outer evaluation: run the inner joint fit once at this ρ and read
        // the converged loss + factor cache.
        let (cost, loss, cache) = term
            .reml_criterion_with_cache(
                target.view(),
                &rho,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
            .expect("reml_criterion_with_cache at the ARD ρ");
        assert!(
            cost.is_finite(),
            "outer REML cost at n={n} must be finite; got {cost}"
        );

        // The per-evaluation payload the outer optimizer steps on.
        let grad_components = term
            .analytic_outer_rho_gradient_at_converged(target.view(), &rho, &loss, &cache)
            .expect("analytic outer-ρ gradient at the converged inner solve");

        channel_lens.push([
            grad_components.explicit.len(),
            grad_components.logdet_trace.len(),
            grad_components.occam.len(),
            grad_components.third_order_correction.len(),
        ]);

        let grad = grad_components.gradient();
        // Every channel and the assembled gradient must be the k-dim outer length,
        // never anything that scales with n.
        assert_eq!(
            grad_components.explicit.len(),
            RHO_FLAT_LEN,
            "n={n}: explicit channel must be k-dim ({RHO_FLAT_LEN}), not n-dim; got {}",
            grad_components.explicit.len()
        );
        assert_eq!(
            grad_components.logdet_trace.len(),
            RHO_FLAT_LEN,
            "n={n}: logdet_trace channel must be k-dim ({RHO_FLAT_LEN}); got {}",
            grad_components.logdet_trace.len()
        );
        assert_eq!(
            grad_components.occam.len(),
            RHO_FLAT_LEN,
            "n={n}: occam channel must be k-dim ({RHO_FLAT_LEN}); got {}",
            grad_components.occam.len()
        );
        assert_eq!(
            grad_components.third_order_correction.len(),
            RHO_FLAT_LEN,
            "n={n}: third_order_correction channel must be k-dim ({RHO_FLAT_LEN}); got {}",
            grad_components.third_order_correction.len()
        );
        assert_eq!(
            grad.len(),
            RHO_FLAT_LEN,
            "n={n}: assembled outer-ρ gradient must be the k-dim search dimension \
             ({RHO_FLAT_LEN}), independent of the row count; got {}",
            grad.len()
        );
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "n={n}: outer-ρ gradient must be finite"
        );
        grad_lens.push(grad.len());
    }

    // The decisive n-invariance: the per-evaluation payload dimension is the SAME
    // for the small and large problems. Different n, identical k-dim consumption.
    assert_eq!(
        channel_lens[0], channel_lens[1],
        "outer-ρ gradient channel lengths must be identical across n_small={n_small} and \
         n_large={n_large} (k-dim consumption is n-invariant); got {:?} vs {:?}",
        channel_lens[0], channel_lens[1]
    );
    assert_eq!(
        grad_lens[0], grad_lens[1],
        "assembled outer-ρ gradient length must be n-invariant; got {} (n={n_small}) vs {} \
         (n={n_large})",
        grad_lens[0], grad_lens[1]
    );
    assert_eq!(
        grad_lens[0], RHO_FLAT_LEN,
        "and that n-invariant length must be exactly the k-dim outer coordinate count"
    );
}
