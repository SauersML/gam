use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::solver::arrow_schur::ArrowFactorCache;
use gam::terms::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldLoss, SaeManifoldRho, SaeManifoldTerm,
};

struct Fixture {
    term: SaeManifoldTerm,
    target: Array2<f64>,
    rho: SaeManifoldRho,
}

/// Build the K=2 periodic-harmonic SAE fixture under a given assignment `mode`.
///
/// `log_lambda_sparse` is exposed so the IBP-MAP arm can run its empirical-π
/// prior at a genuinely active weight (the fixed-`alpha` IBP penalty reads
/// `lambda_sparse` as its weight lever), which is what exercises the #1006
/// empirical-`M_k` third channel through the outer-ρ gradient.
fn fixture(mode: AssignmentMode, log_lambda_sparse: f64) -> Fixture {
    let n = 80usize;
    let p = 6usize;
    let k_atoms = 2usize;
    let m = 5usize;
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator");

    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let mut target = Array2::<f64>::zeros((n, p));
    let weights0 = [
        [0.20, -0.10, 0.06, 0.03, -0.04, 0.08],
        [0.70, -0.25, 0.40, 0.12, -0.35, 0.18],
        [0.15, 0.55, -0.25, 0.28, 0.08, -0.22],
        [0.08, -0.04, 0.03, -0.02, 0.01, 0.06],
        [-0.06, 0.02, 0.05, 0.04, -0.03, 0.01],
    ];
    let weights1 = [
        [-0.10, 0.05, 0.08, -0.02, 0.05, -0.03],
        [-0.30, 0.42, 0.12, -0.20, 0.16, 0.30],
        [0.48, 0.10, -0.32, 0.18, 0.26, -0.14],
        [0.04, 0.07, -0.02, 0.03, -0.05, 0.02],
        [0.03, -0.05, 0.04, 0.01, 0.02, -0.04],
    ];

    for row in 0..n {
        let phase = (row as f64 + 0.25) / n as f64;
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.18).fract();
        let route = if row < n / 2 { 1.7 } else { -1.7 };
        logits[[row, 0]] = route;
        // A genuine second active gate so the IBP / JumpReLU per-atom logits all
        // sit inside their optimization bands (softmax ignores the absolute
        // level, gate modes do not).
        logits[[row, 1]] = if row % 3 == 0 { 0.9 } else { 0.3 };
        let theta0 = std::f64::consts::TAU * coords[0][[row, 0]];
        let theta1 = std::f64::consts::TAU * coords[1][[row, 0]];
        let basis0 = [
            1.0,
            theta0.sin(),
            theta0.cos(),
            (2.0 * theta0).sin(),
            (2.0 * theta0).cos(),
        ];
        let basis1 = [
            1.0,
            theta1.sin(),
            theta1.cos(),
            (2.0 * theta1).sin(),
            (2.0 * theta1).cos(),
        ];
        let mix0 = 1.0 / (1.0 + (-route / 0.7).exp());
        let mix1 = 1.0 - mix0;
        for col in 0..p {
            let mut v0 = 0.0;
            let mut v1 = 0.0;
            for b in 0..m {
                v0 += basis0[b] * weights0[b][col];
                v1 += basis1[b] * weights1[b][col];
            }
            target[[row, col]] = mix0 * v0 + mix1 * v1;
        }
    }

    let mut atoms = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom_idx].view())
            .expect("periodic basis evaluation");
        let decoder = if atom_idx == 0 {
            Array2::from_shape_fn((m, p), |(row, col)| weights0[row][col])
        } else {
            Array2::from_shape_fn((m, p), |(row, col)| weights1[row][col])
        };
        let mut smooth = Array2::<f64>::eye(m);
        smooth[[0, 0]] = 0.0;
        let atom = SaeManifoldAtom::new(
            format!("circle_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            smooth,
        )
        .expect("circle atom")
        .with_basis_evaluator(Arc::new(
            PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator clone"),
        ) as Arc<dyn SaeBasisEvaluator>);
        atoms.push(atom);
    }

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    let rho = SaeManifoldRho::new(
        log_lambda_sparse,
        -8.0,
        vec![Array1::from_vec(vec![-8.0]), Array1::from_vec(vec![-8.0])],
    );
    Fixture { term, target, rho }
}

fn evaluate(
    start: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    inner_max_iter: usize,
) -> (SaeManifoldTerm, f64, SaeManifoldLoss, ArrowFactorCache) {
    let mut term = start.clone();
    let (value, loss, cache) = term
        .reml_criterion_with_cache(
            target.view(),
            rho,
            None,
            inner_max_iter,
            0.45,
            1.0e-6,
            1.0e-6,
        )
        .unwrap_or_else(|err| panic!("REML criterion failed: {err}"));
    (term, value, loss, cache)
}

fn centered_fd(
    start: &SaeManifoldTerm,
    target: &Array2<f64>,
    template: &SaeManifoldRho,
    coord: usize,
    inner_max_iter: usize,
) -> f64 {
    let h = 2.0e-4;
    let mut plus = template.to_flat();
    let mut minus = template.to_flat();
    plus[coord] += h;
    minus[coord] -= h;
    let rho_plus = template.from_flat(plus.view());
    let rho_minus = template.from_flat(minus.view());
    let (_, vp, _, _) = evaluate(start, target, &rho_plus, inner_max_iter);
    let (_, vm, _, _) = evaluate(start, target, &rho_minus, inner_max_iter);
    (vp - vm) / (2.0 * h)
}

/// The full analytic outer-ρ gradient — explicit + direct log-det traces +
/// Occam + the #1006 third-order implicit-state correction — must match a
/// centered finite difference of the actual REML criterion (inner problem
/// re-solved at each ρ, so the FD carries the envelope/IFT terms).
fn assert_full_gradient_matches_fd(label: &str, f: &Fixture) {
    let (converged, _value, loss, cache) = evaluate(&f.term, &f.target, &f.rho, 8);
    let components = converged
        .analytic_outer_rho_gradient_at_converged(f.target.view(), &f.rho, &loss, &cache)
        .expect("analytic components");
    let analytic = components.gradient();
    let n_params = f.rho.to_flat().len();

    for coord in 0..n_params {
        let fd = centered_fd(&converged, &f.target, &f.rho, coord, 8);
        let diff = (fd - analytic[coord]).abs();
        let tol = 2.5e-3 * (1.0 + fd.abs().max(analytic[coord].abs()));
        assert!(
            diff <= tol,
            "[{label}] full rho gradient coord {coord}: fd={fd:.8e}, analytic={:.8e}, diff={diff:.3e}, tol={tol:.3e}",
            analytic[coord]
        );
    }
}

/// K=2 fixture whose per-atom decoder design is RANK-DEFICIENT in the data —
/// the #1117 OLMo-circle degeneracy. The latent coordinate is squeezed into a
/// narrow phase band so the 2nd-harmonic basis pair `[sin 2θ, cos 2θ]` is
/// barely excited and the bare data Gram `G_k = D_kᵀ D_k` drops rank. Under K=2
/// the shared-row logit×coordinate Gauss-Newton cross term then drives a per-row
/// `H_tt` block genuinely indefinite at/near the stationary point, so the
/// undamped evidence factor must condition it by unit-stiffness SPECTRAL
/// deflation (eigenvalue → +1, ρ-independent log 1 = 0). This is precisely the
/// branch whose former ridge-damped fallback injected a ρ-dependent evidence
/// bias and desynced the outer value from the analytic ρ-gradient (#1117). The
/// certificate this test asserts — analytic ∂V/∂ρ ≈ centered FD of the actual
/// re-solved criterion — is exactly `grad·v ≈ fd·v`: it holds iff the value and
/// gradient ride the SAME deflated factorization.
fn rank_deficient_fixture(mode: AssignmentMode, log_lambda_sparse: f64) -> Fixture {
    let n = 80usize;
    let p = 6usize;
    let k_atoms = 2usize;
    let m = 5usize;
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator");

    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    // Squeeze both atoms' latent coordinate into a ±0.5%-wide band around 0.5:
    // the periodic 2nd-harmonic columns `[sin 2θ, cos 2θ]` are then nearly
    // unexcited, so the bare data Gram `G_k = D_kᵀ D_k` drops rank (the #1117
    // OLMo-circle degeneracy). Under K=2 the shared-row logit×coordinate
    // Gauss-Newton cross term drives a per-row `H_tt` block indefinite at/near
    // the optimum, forcing the undamped evidence factor down the spectral
    // unit-stiffness deflation path this fix installs.
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let phase = 0.5 + 0.005 * ((row as f64 / n as f64) - 0.5);
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = phase;
        let route = if row < n / 2 { 1.4 } else { -1.4 };
        logits[[row, 0]] = route;
        logits[[row, 1]] = if row % 3 == 0 { 0.9 } else { 0.3 };
        let theta = std::f64::consts::TAU * phase;
        let basis = [
            1.0,
            theta.sin(),
            theta.cos(),
            (2.0 * theta).sin(),
            (2.0 * theta).cos(),
        ];
        for col in 0..p {
            // Deterministic, finite target so the inner solve converges; the
            // exact values do not matter for the FD certificate.
            let mut v = 0.0;
            for (b, &bv) in basis.iter().enumerate() {
                v += bv * (0.1 + 0.03 * (b as f64) - 0.01 * (col as f64));
            }
            target[[row, col]] = v;
        }
    }

    let mut atoms = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom_idx].view())
            .expect("periodic basis evaluation");
        let decoder = Array2::from_shape_fn((m, p), |(r, c)| {
            0.1 + 0.05 * (r as f64) - 0.02 * (c as f64) + 0.01 * (atom_idx as f64)
        });
        let mut smooth = Array2::<f64>::eye(m);
        smooth[[0, 0]] = 0.0;
        let atom = SaeManifoldAtom::new(
            format!("circle_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            smooth,
        )
        .expect("circle atom")
        .with_basis_evaluator(Arc::new(
            PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator clone"),
        ) as Arc<dyn SaeBasisEvaluator>);
        atoms.push(atom);
    }

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    let rho = SaeManifoldRho::new(
        log_lambda_sparse,
        -8.0,
        vec![Array1::from_vec(vec![-8.0]), Array1::from_vec(vec![-8.0])],
    );
    Fixture { term, target, rho }
}

#[test]
fn sae_outer_rho_gradient_components_match_centered_fd_softmax() {
    let f = fixture(AssignmentMode::softmax(0.7), -8.0);
    assert_full_gradient_matches_fd("softmax", &f);
}

#[test]
fn sae_outer_rho_gradient_certificate_consistent_under_rank_deficient_k2() {
    // K=2 rank-deficient circle: the indefinite per-row H_tt must be spectral-
    // deflated at unit stiffness, NOT ridge-damped, so the outer REML value and
    // its analytic ρ-gradient stay consistent (grad·v ≈ fd·v). A ρ-dependent
    // ridge bias would break this certificate and is what stalled the outer BFGS
    // line-search for multi-atom fits (#1117).
    let f = rank_deficient_fixture(AssignmentMode::softmax(0.7), -8.0);
    assert_full_gradient_matches_fd("rank_deficient_k2_softmax", &f);
}

#[test]
fn sae_outer_rho_gradient_components_match_centered_fd_jumprelu() {
    // JumpReLU with the threshold below the active logits so both gates sit in
    // the optimization band and the sigmoid-sparsity third channel is live.
    let f = fixture(AssignmentMode::jumprelu(0.7, 0.0), -1.5);
    assert_full_gradient_matches_fd("jumprelu", &f);
}

#[test]
fn sae_outer_rho_gradient_components_match_centered_fd_ibp_map() {
    // IBP-MAP exercises the #1006 empirical-π third channel: pi_k(M_k) couples
    // every row in a column, so the outer-ρ gradient through log|H| depends on
    // the cross-row M_k channel of `logdet_theta_adjoint`. lambda_sparse is the
    // active prior weight here, so coord 0's FD directly stresses it.
    let f = fixture(AssignmentMode::ibp_map(0.7, 0.9, false), -1.5);
    assert_full_gradient_matches_fd("ibp_map", &f);
}

/// Centered finite difference of the FROZEN-θ penalized loss `loss.total()` at
/// `ρ̂ ± h` along one outer coordinate. [`SaeManifoldTerm::loss`] borrows `&self`
/// and never re-solves the inner (t, β) problem, so the inner state stays pinned
/// at the converged `θ̂` and this difference isolates the DIRECT ρ-derivative of
/// the data-fit + priors — exactly the analytic `explicit` channel — with no
/// envelope / IFT term. FD-OK: audit instrument only (SPEC), never a production
/// path; the analytic channel is authoritative.
fn frozen_explicit_fd(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    template: &SaeManifoldRho,
    coord: usize,
) -> f64 {
    let h = 2.0e-4;
    let mut plus = template.to_flat();
    let mut minus = template.to_flat();
    plus[coord] += h;
    minus[coord] -= h;
    let rho_plus = template.from_flat(plus.view());
    let rho_minus = template.from_flat(minus.view());
    let lp = term
        .loss(target.view(), &rho_plus)
        .expect("frozen-θ loss +h")
        .total();
    let lm = term
        .loss(target.view(), &rho_minus)
        .expect("frozen-θ loss -h")
        .total();
    (lp - lm) / (2.0 * h)
}

/// #2087 — per-CHANNEL FD decomposition of the outer-ρ gradient, so a
/// gradient↔objective desync is localized to a NAMED analytic channel rather
/// than only reported as "the summed gradient disagrees with the FD".
///
/// The full analytic gradient is
/// `explicit + logdet_trace + occam + third_order`. This test finite-differences
/// the two independently-recoverable value sub-paths and attributes each:
///
///   * `explicit` is the direct ρ-derivative of the frozen-θ penalized loss
///     `loss.total()` (inner state pinned at `θ̂`), recovered EXACTLY by
///     [`frozen_explicit_fd`]. A match rules the data-fit / prior channel OUT.
///   * the remaining `log|H|`-block `(logdet_trace + occam + third_order)` is the
///     envelope FD of the RE-SOLVED criterion MINUS the frozen-θ `explicit` FD.
///     A mismatch localizes the defect to the Hessian log-det derivative — the
///     direct trace `½ tr(H⁻¹ ∂H/∂ρ)` and/or the θ-adjoint `Γ` that feeds the
///     #1006 third-order correction (the cross-row IBP Woodbury / empirical-`M`
///     channel of `logdet_theta_adjoint`).
///
/// This is the durable localization artifact for the #1798/#1795/#2087 cross-row
/// IBP Woodbury logdet genus: a red run PRINTS the per-coordinate channel split
/// and its assertion message NAMES the culprit block.
fn assert_channel_decomposition(label: &str, f: &Fixture) {
    let (converged, _value, loss, cache) = evaluate(&f.term, &f.target, &f.rho, 8);
    let components = converged
        .analytic_outer_rho_gradient_at_converged(f.target.view(), &f.rho, &loss, &cache)
        .expect("analytic components");
    let n_params = f.rho.to_flat().len();

    for coord in 0..n_params {
        let explicit_a = components.explicit[coord];
        let logdet_a = components.logdet_trace[coord];
        let occam_a = components.occam[coord];
        let third_a = components.third_order_correction[coord];

        // Direct (frozen-θ) FD of loss.total() ≡ the analytic `explicit` channel.
        let explicit_fd = frozen_explicit_fd(&converged, &f.target, &f.rho, coord);
        // Envelope FD of the RE-SOLVED criterion ≡ the FULL analytic gradient.
        let total_fd = centered_fd(&converged, &f.target, &f.rho, coord, 8);
        // The log|H|-block is the envelope FD minus the frozen-θ explicit FD.
        let block_fd = total_fd - explicit_fd;
        let block_a = logdet_a + occam_a + third_a;

        eprintln!(
            "[{label}] coord {coord}: EXPLICIT fd={explicit_fd:.6e} an={explicit_a:.6e} | \
             logH-BLOCK fd={block_fd:.6e} an={block_a:.6e} \
             (= logdet_trace {logdet_a:.6e} + occam {occam_a:.6e} + third_order {third_a:.6e})"
        );

        let expl_tol = 2.5e-3 * (1.0 + explicit_fd.abs().max(explicit_a.abs()));
        assert!(
            (explicit_fd - explicit_a).abs() <= expl_tol,
            "[{label}] EXPLICIT channel coord {coord} desync: frozen-θ loss FD {explicit_fd:.8e} \
             vs analytic explicit {explicit_a:.8e} — the direct data-fit/prior ρ-derivative is wrong"
        );
        let block_tol = 2.5e-3 * (1.0 + block_fd.abs().max(block_a.abs()));
        assert!(
            (block_fd - block_a).abs() <= block_tol,
            "[{label}] log|H|-BLOCK coord {coord} desync: envelope-minus-explicit FD {block_fd:.8e} \
             vs analytic (logdet_trace {logdet_a:.8e} + occam {occam_a:.8e} + third_order {third_a:.8e}) \
             = {block_a:.8e}. The direct prior/data-fit channel is clean (asserted above), so the \
             defect is in the Hessian log-det derivative: the direct trace ½tr(H⁻¹∂H/∂ρ) and/or the \
             θ-adjoint Γ feeding the #1006 third-order (cross-row IBP Woodbury / empirical-M channel)."
        );
    }
}

#[test]
fn sae_outer_rho_gradient_channel_decomposition_softmax_2087() {
    // Control arm: softmax carries no cross-row IBP log-det channel, so BOTH the
    // explicit and the log|H|-block FD attributions must hold — this pins that the
    // decomposition instrument itself is sound before it is read on the IBP arm.
    let f = fixture(AssignmentMode::softmax(0.7), -8.0);
    assert_channel_decomposition("softmax", &f);
}

#[test]
fn sae_outer_rho_gradient_channel_decomposition_ibp_map_2087() {
    // Suspect arm: IBP-MAP drives the cross-row empirical-`M` / Woodbury log-det
    // channels of `logdet_theta_adjoint`. The decomposition pins whether a #2087
    // desync lives in the explicit prior channel or the Hessian log-det block —
    // the located artifact a build-capable seat consumes to fix the θ-adjoint.
    let f = fixture(AssignmentMode::ibp_map(0.7, 0.9, false), -1.5);
    assert_channel_decomposition("ibp_map", &f);
}

/// #2087 discriminator — is the `third_order` channel (−15.49 on the softmax
/// arm, larger than and cancelling the verified-correct `logdet_trace`) a REAL
/// envelope term the small-`h` FD misses, or spurious amplification through a
/// near-singular adjoint solve?
///
/// Two instruments, printed per step:
///
///   1. **Did the FD probes move θ̂ at all?** At `ρ ± h` the probe re-solve
///      warm-starts from `θ̂(ρ)`; if its stationarity gate already passes, it
///      returns the SAME state and the "envelope" FD is exactly the frozen-θ
///      difference — i.e. `explicit + logdet_trace`, NO third-order content.
///      Fingerprint: re-solved loss vs frozen-θ loss at the same `ρ + h`.
///   2. **h-sweep.** If the analytic third-order is real, the envelope FD must
///      drift from `logdet_trace` toward `logdet_trace + third_order` as `h`
///      grows past the inner tolerance; if it stays glued to `logdet_trace`,
///      the true θ̂-response term is ≈ 0 and the analytic third-order value is
///      the defect (Γ / θ̂_ρ amplification through weakly-identified inner
///      directions).
///
/// Diagnostic instrument for the #1795/#1798/#2087 genus: it asserts only
/// finiteness; its printed table is the artifact the fix consumes.
#[test]
fn zz_2087_third_order_envelope_discriminator_softmax() {
    let f = fixture(AssignmentMode::softmax(0.7), -8.0);
    let (converged, _value, loss, cache) = evaluate(&f.term, &f.target, &f.rho, 8);
    let components = converged
        .analytic_outer_rho_gradient_at_converged(f.target.view(), &f.rho, &loss, &cache)
        .expect("analytic components");
    let coord = 0usize;
    let explicit_a = components.explicit[coord];
    let logdet_a = components.logdet_trace[coord];
    let third_a = components.third_order_correction[coord];
    eprintln!(
        "[2087-diag] analytic: explicit={explicit_a:.6e} logdet_trace={logdet_a:.6e} \
         occam={:.6e} third_order={third_a:.6e}",
        components.occam[coord]
    );

    // (1) θ̂-motion fingerprint at the gate's own h.
    let h_small = 2.0e-4;
    let mut plus = f.rho.to_flat();
    plus[coord] += h_small;
    let rho_plus = f.rho.from_flat(plus.view());
    let frozen_plus = converged
        .loss(f.target.view(), &rho_plus)
        .expect("frozen-θ loss at rho+h")
        .total();
    let (_, _, resolved_plus, _) = evaluate(&converged, &f.target, &rho_plus, 8);
    let moved = (resolved_plus.total() - frozen_plus).abs();
    eprintln!(
        "[2087-diag] θ̂-motion fingerprint at h={h_small:.1e}: \
         |loss(θ̂(ρ+h)) − loss(θ̂(ρ))| = {moved:.6e} \
         ({} — a re-solve that returns the warm start gives exactly 0)",
        if moved == 0.0 { "θ̂ DID NOT MOVE" } else { "θ̂ moved" }
    );

    // (2) h-sweep with a deep probe budget so the inner response can express.
    for h in [2.0e-4_f64, 2.0e-3, 1.0e-2, 5.0e-2] {
        let mut p = f.rho.to_flat();
        let mut m_ = f.rho.to_flat();
        p[coord] += h;
        m_[coord] -= h;
        let rho_p = f.rho.from_flat(p.view());
        let rho_m = f.rho.from_flat(m_.view());
        let (_, vp, _, _) = evaluate(&converged, &f.target, &rho_p, 64);
        let (_, vm, _, _) = evaluate(&converged, &f.target, &rho_m, 64);
        let fd = (vp - vm) / (2.0 * h);
        let block_fd = fd - explicit_a;
        assert!(fd.is_finite(), "envelope FD non-finite at h={h:.1e}");
        eprintln!(
            "[2087-diag] h={h:.1e} (deep probes): envelope FD={fd:.6e} → block \
             (FD−explicit)={block_fd:.6e} | trace-only predicts {logdet_a:.6e}, \
             trace+third predicts {:.6e}",
            logdet_a + third_a
        );
    }
}
