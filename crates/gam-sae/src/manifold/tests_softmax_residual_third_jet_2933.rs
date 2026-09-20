// `manifold/mod.rs` declares this module only as
// `#[cfg(test)] mod tests_softmax_residual_third_jet_2933;`.
#![cfg(test)]
//! #2933 F01 — the θ-derivative of the exact observed information must carry the
//! residual THIRD derivative `⟨M r, ∂³f_abw⟩` of the whole reconstruction
//! `f = Σ_k a_k(ℓ)·γ_k(t_k)`, cross-atom gate terms included.
//!
//! `patchd_residual_third_leg` returned zero for any triple with mixed atom labels
//! and for any logit unless the gate was ordered Beta–Bernoulli, while the outer
//! gradient of a softmax fit reaches both production θ-adjoints
//! (`dense_exact_a_logdet_channels` and the bundle's
//! `logdet_theta_adjoint_from_probes`) with the data target that switches the leg
//! on. So `∂_ℓj ∂²_tk f = ∂a_k/∂ℓ_j·γ_k''`, nonzero for `j ≠ k`, never reached `dA`,
//! and a ThresholdGate fit never saw its own logit legs.
//!
//! The arbiters are central differences of the MATERIALIZED dense `A`, of
//! `log|det A|`, of the exact-A evidence factor's log-determinant, and of the
//! assembled KKT gradient, at endpoints moved by `apply_newton_step` with the
//! collapse gates held. They are free to disagree with every analytic leg. Each
//! derivative arbiter is paired with the same contraction run without the target
//! (`residual_target = None` skips the residual third legs) as a positive control,
//! which must miss the difference by far more than the analytic side is allowed to.

use crate::manifold::tests_dense_solver_oracles::DeflatedArrowSolver;
use super::*;
use gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR;
use ndarray::{Array1, Array2, s};

/// Gate temperature, deliberately not one.
const TAU: f64 = 0.7;
/// ThresholdGate threshold. Every fixture logit sits below it.
const THRESHOLD: f64 = 0.0;
/// Coarsest central-difference step; Richardson uses `h`, `h/2` and `h/4`.
const STEP: f64 = 1.0e-3;
/// Rows whose `t` slots are differenced.
const PROBE_ROWS: [usize; 2] = [0, 3];
/// Amplitude of the target's offset from the fixture's reconstruction.
const RESIDUAL_AMPLITUDE: f64 = 0.05;
/// Relative bar for every central-difference comparison. Richardson over `h, h/2, h/4`
/// at `STEP` leaves a truncation remainder of order `h⁴ ≈ 1e-12` and a rounding floor
/// of order `ε·‖A‖₂/h ≈ 1e-11` on this fixture (`‖A‖₂ ≈ 6`). The finer-minus-coarser
/// oracle is charged to each gap, so an unresolved difference cannot pass. The bar
/// sits four decades above that resolution.
const FD_TOL: f64 = 1.0e-6;
/// A positive control must miss by at least this many bars: the leg a gate certifies
/// is resolved two decades above the bar before its absence counts as visible.
const CONTROL_MARGIN: f64 = 100.0;

/// How the logits move the gates `a_k` of the fixture's reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureGate {
    /// `a = softmax(ℓ/τ)`: every free logit moves every gate.
    Softmax,
    /// `a_k = σ((ℓ_k − θ)/τ)`: a logit moves only its own atom's gate.
    ThresholdGate,
}

/// The target's offset from the fixture's reconstruction, so `M r ≠ 0`.
fn residual_offset(n: usize, p: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, p), |(row, out_col)| {
        RESIDUAL_AMPLITUDE * (((row * 7 + out_col * 3) as f64) * 0.7).sin()
    })
}

/// Four periodic atoms: free logits per row (three under softmax, so a triple of three
/// DISTINCT logits exists; four under the threshold gate), unequal gates, harmonic
/// curves whose third coordinate derivative does not vanish, a target offset from the
/// reconstruction (so `M r ≠ 0`), and live sparsity, smoothness and ARD priors.
///
/// Every row factor stays off the deflation stratum by construction:
/// - Every coordinate lies within a sixth of a period of the chart origin, where the
///   periodic ARD curvature `α·cos(2πt) ≥ α/2` is positive and its majorizer
///   `α·softplus_{τ₀}(cos 2πt)` equals it, so every coordinate direction carries
///   prior curvature whatever the data rank. On the concave half that majorizer is
///   numerically zero, and a row with more such coordinates than its data Jacobian
///   resolves has an exactly singular `H_tt`.
/// - The decoders' phase frequency in the output column depends on the basis
///   column, so the decoded curves do not all lie in one plane of the outputs, as
///   they would for a decoder that is a sum of two outer products.
/// - Threshold gates sit below one half, where the prior's logit curvature
///   `w·λ·s·(1 − 2a)/τ²` is positive and its majorizer equals it.
/// - The residual is small beside that curvature.
///
/// The premises in the tests check the outcome instead of assuming it.
fn residual_fixture(gate: FixtureGate) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 6usize;
    let p = 3usize;
    let k_atoms = 4usize;
    let m = 3usize;
    let evaluator = Arc::new(
        PeriodicHarmonicEvaluator::new(m)
            .expect("m=3 is odd and positive, the basis width this evaluator requires"),
    );
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords: Vec<Array2<f64>> =
        (0..k_atoms).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    for row in 0..n {
        for atom in 0..k_atoms {
            let t = (0.2 * (row as f64 + 0.35) / n as f64 + 0.04 * atom as f64 - 0.15)
                .rem_euclid(1.0);
            assert!(
                (2.0 * std::f64::consts::PI * t).cos() > 0.0,
                "#2933 F01 premise: coordinate {t} of atom {atom} in row {row} must sit on the \
                 convex half of the periodic ARD prior"
            );
            coords[atom][[row, 0]] = t;
            let phase = (1.3 * row as f64 + 0.8 * atom as f64 + 0.2).sin();
            logits[[row, atom]] = match gate {
                FixtureGate::Softmax => 0.9 * phase,
                FixtureGate::ThresholdGate => THRESHOLD - 0.4 - 0.5 * phase.abs(),
            };
        }
    }
    let gates = Array2::from_shape_fn((n, k_atoms), |(row, atom)| match gate {
        FixtureGate::Softmax => softmax_row(logits.row(row), TAU)[atom],
        FixtureGate::ThresholdGate => {
            1.0 / (1.0 + (-(logits[[row, atom]] - THRESHOLD) / TAU).exp())
        }
    });
    if gate == FixtureGate::ThresholdGate {
        let highest = gates.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            highest < 0.5,
            "#2933 F01 premise: every threshold gate must sit below one half, where the prior's \
             logit curvature is positive (highest gate {highest:.6})"
        );
    }
    let mut target = residual_offset(n, p);
    let mut atoms = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom].view())
            .expect("fixture coords are finite points of the unit-period circle chart");
        // The phase's frequency in `out_col` depends on `basis_col`, so the rows of
        // the decoder are not confined to one plane of the outputs.
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            let (basis_col, out_col) = (basis_col as f64, out_col as f64);
            0.4 * (1.7 * atom as f64 + 1.1 * basis_col + (0.6 + 0.8 * basis_col) * out_col + 0.3)
                .sin()
        });
        let decoded = phi.dot(&decoder);
        for row in 0..n {
            for out_col in 0..p {
                target[[row, out_col]] += gates[[row, atom]] * decoded[[row, out_col]];
            }
        }
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("residual_third_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .expect("decoder is (m, p) and phi/jet come from the same evaluator")
            .with_basis_second_jet(evaluator.clone()),
        );
    }
    let mode = match gate {
        FixtureGate::Softmax => AssignmentMode::softmax(TAU),
        FixtureGate::ThresholdGate => AssignmentMode::threshold_gate(TAU, THRESHOLD),
    };
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .expect("k_atoms coord blocks and k_atoms manifolds match the logits' k_atoms columns");
    let term = SaeManifoldTerm::new(atoms, assignment)
        .expect("the assignment was built with exactly one block per atom");
    let rho = SaeManifoldRho::new(-1.0, -1.0, vec![Array1::from_vec(vec![-1.0]); k_atoms])
        .for_assignment(&term.assignment);
    (term, target, rho)
}

/// The undamped `B` factor at the term's CURRENT state with no inner iteration: the
/// fixed-state operator every `A = B + ΔC` below is materialized against.
fn fixed_state_cache(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let sys = term
        .assemble_arrow_schur(target.view(), rho, None)
        .expect("arrow-Schur assembly at a fixture state");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("undamped B factor at a fixture state");
    cache
}

/// The anchor and its cache, with the three per-assembly lagged gates pinned
/// (#2828): the objective whose Hessian `A` is holds them fixed, so a difference
/// across endpoints that re-derived their own gates would belong to no operator.
fn anchored_state(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (SaeManifoldTerm, ArrowFactorCache) {
    let mut anchor = term.clone();
    let cache = fixed_state_cache(&mut anchor, target, rho);
    anchor.streaming_gates_frozen = true;
    (anchor, cache)
}

/// A clone of `anchor` holding the anchor's collapse-prevention gates.
/// `SaeManifoldTerm::clone` drops them, and an endpoint that re-derived its own would
/// differentiate a routing refresh that no operator here carries.
fn frozen_endpoint(anchor: &SaeManifoldTerm) -> SaeManifoldTerm {
    let mut endpoint = anchor.clone();
    endpoint.declare_collapse_prevention_gates(&anchor.collapse_prevention_gates());
    endpoint
}

fn deflated_direction_count(term: &SaeManifoldTerm, cache: &ArrowFactorCache) -> usize {
    (0..term.n_obs())
        .map(|row| cache.deflated_row_directions[row].len())
        .sum()
}

/// Asserts the anchor's `B` factor is deflation-free, printing the count it read.
fn assert_anchor_deflation_free(label: &str, term: &SaeManifoldTerm, cache: &ArrowFactorCache) {
    let deflated = deflated_direction_count(term, cache);
    eprintln!("[#2933 F01 {label}] anchor deflated directions={deflated}");
    assert_eq!(
        deflated, 0,
        "#2933 F01 premise: the anchor must be deflation-free"
    );
}

/// #2933 F01 premise — every row block of the materialized exact `A` is positive
/// definite, the property the fixture builds in. The deflation premises at the
/// anchor and at every finite-difference endpoint separately check that no row
/// factor changes stratum.
fn assert_exact_row_blocks_positive_definite(
    anchor: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) {
    let a = anchor
        .materialize_exact_hessian_dense(rho, target.view(), cache)
        .expect("dense exact A at the anchor");
    for row in 0..anchor.n_obs() {
        let q = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let block = a.slice(s![base..base + q, base..base + q]).to_owned();
        let (eigenvalues, _) = block
            .eigh(Side::Lower)
            .expect("a row block of the symmetric exact A");
        let smallest = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let largest = eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "[#2933 F01 premise] row={row} exact A_tt eigenvalues in [{smallest:.6e}, {largest:.6e}]"
        );
        assert!(
            smallest > 0.0,
            "#2933 F01 premise: row {row}'s exact A_tt must be positive definite by construction \
             (eigenvalues in [{smallest:.3e}, {largest:.3e}])"
        );
    }
}

fn log_abs_det(a: &Array2<f64>) -> f64 {
    let (eigenvalues, _) = a.eigh(Side::Lower).expect("the dense exact A is symmetric");
    eigenvalues.iter().map(|lambda| lambda.abs().ln()).sum()
}

/// The exact inverse of the materialized `A`. `log|det A|` is differentiable only where
/// `A` is nonsingular, and the inverse contraction must be resolved below the bar it
/// feeds. Its relative error is of order `κ(A)·ε`, which must sit two decades below
/// `FD_TOL`: `κ(A) ≤ FD_TOL/(CONTROL_MARGIN·ε)`, i.e.
/// `min|λ| ≥ ‖A‖₂·ε·CONTROL_MARGIN/FD_TOL`.
fn nonsingular_inverse(label: &str, a: &Array2<f64>) -> Array2<f64> {
    let (eigenvalues, eigenvectors) = a.eigh(Side::Lower).expect("the dense exact A is symmetric");
    let spectral_norm = eigenvalues.iter().map(|l| l.abs()).fold(0.0_f64, f64::max);
    let smallest = eigenvalues.iter().map(|l| l.abs()).fold(f64::INFINITY, f64::min);
    let negative = eigenvalues.iter().filter(|&&l| l < 0.0).count();
    let floor = spectral_norm * f64::EPSILON * CONTROL_MARGIN / FD_TOL;
    eprintln!(
        "[#2933 F01 {label}] dim={} ‖A‖₂={spectral_norm:.6e} min|λ|={smallest:.6e} \
         (floor ‖A‖₂·ε·CONTROL_MARGIN/FD_TOL={floor:.3e}) negative={negative}",
        a.nrows()
    );
    assert!(
        smallest >= floor,
        "#2933 F01 premise: A must be resolved below the arbiter's bar for its inverse \
         contraction to arbitrate (min|λ|={smallest:.3e} < ‖A‖₂·ε·CONTROL_MARGIN/FD_TOL={floor:.3e})"
    );
    eigenvectors
        .dot(&Array2::from_diag(&eigenvalues.mapv(|l| 1.0 / l)))
        .dot(&eigenvectors.t())
}

/// `anchor` moved by `signed_step` along the unit `t` slot `slot`, which is the variable
/// `var` of `row`. A displacement error `δ` biases a central difference by `δ/|h|`
/// relative, which must sit two decades below `FD_TOL`, so the step must move the named
/// variable to within `|h|·FD_TOL/CONTROL_MARGIN`.
fn moved_endpoint(
    anchor: &SaeManifoldTerm,
    anchor_cache: &ArrowFactorCache,
    row: usize,
    slot: usize,
    var: SaeLocalRowVar,
    signed_step: f64,
) -> SaeManifoldTerm {
    let mut endpoint = frozen_endpoint(anchor);
    // `apply_newton_step` refuses a non-positive step size, so the sign rides on
    // the direction.
    let mut direction = Array1::<f64>::zeros(anchor_cache.delta_t_len());
    direction[slot] = signed_step.signum();
    endpoint
        .apply_newton_step(
            direction.view(),
            Array1::<f64>::zeros(anchor_cache.k).view(),
            signed_step.abs(),
        )
        .expect("finite-difference endpoint step");
    let moved = match var {
        SaeLocalRowVar::Logit { atom } => {
            endpoint.assignment.logits[[row, atom]] - anchor.assignment.logits[[row, atom]]
        }
        SaeLocalRowVar::Coord { atom, axis } => {
            endpoint.assignment.coords[atom].row(row)[axis]
                - anchor.assignment.coords[atom].row(row)[axis]
        }
    };
    let displacement = (moved - signed_step).abs();
    let displacement_bar = signed_step.abs() * FD_TOL / CONTROL_MARGIN;
    eprintln!(
        "[#2933 F01 step] row={row} slot={slot} var={var:?} h={signed_step:e} \
         |moved − h|={displacement:.3e} (bar |h|·FD_TOL/CONTROL_MARGIN={displacement_bar:.3e})"
    );
    assert!(
        displacement <= displacement_bar,
        "#2933 F01 premise: slot {slot} of row {row} is {var:?}, but a step of {signed_step:e} \
         moved that variable by {moved:e} (|moved − h|={displacement:.3e} > {displacement_bar:.3e}); \
         the difference would not be in the variable the adjoint names"
    );
    endpoint
}

/// `anchor` moved by `signed_step` along joint coordinate `index`: a `t` slot below
/// `total_t`, a border coefficient from there on.
fn joint_endpoint(
    anchor: &SaeManifoldTerm,
    total_t: usize,
    k: usize,
    index: usize,
    signed_step: f64,
) -> SaeManifoldTerm {
    let mut endpoint = frozen_endpoint(anchor);
    let mut direction_t = Array1::<f64>::zeros(total_t);
    let mut direction_beta = Array1::<f64>::zeros(k);
    if index < total_t {
        direction_t[index] = signed_step.signum();
    } else {
        direction_beta[index - total_t] = signed_step.signum();
    }
    endpoint
        .apply_newton_step(direction_t.view(), direction_beta.view(), signed_step.abs())
        .expect("finite-difference endpoint step");
    endpoint
}

/// The `B` factor at a finite-difference endpoint, refused off the anchor's
/// deflation-free stratum.
fn endpoint_cache(
    endpoint: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let cache = fixed_state_cache(endpoint, target, rho);
    assert_eq!(
        deflated_direction_count(endpoint, &cache),
        0,
        "#2933 F01 premise: a finite-difference endpoint must stay deflation-free, or the two \
         endpoints straddle a discrete deflation change and their difference is no derivative"
    );
    cache
}

/// The dense exact `A` at the anchor moved along the unit `t` slot `slot`.
fn endpoint_exact_a(
    anchor: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    anchor_cache: &ArrowFactorCache,
    row: usize,
    slot: usize,
    var: SaeLocalRowVar,
    signed_step: f64,
) -> Array2<f64> {
    let mut endpoint = moved_endpoint(anchor, anchor_cache, row, slot, var, signed_step);
    let cache = endpoint_cache(&mut endpoint, target, rho);
    endpoint
        .materialize_exact_hessian_dense(rho, target.view(), &cache)
        .expect("dense exact A at a finite-difference endpoint")
}

/// The dense exact `A` at the anchor with decoder coefficient `(mu, out)` of `atom`
/// moved by `signed_step`.
fn endpoint_decoder_exact_a(
    anchor: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    atom: usize,
    mu: usize,
    out: usize,
    signed_step: f64,
) -> Array2<f64> {
    let mut endpoint = frozen_endpoint(anchor);
    let mut decoder = endpoint.atoms[atom].decoder_coefficients().clone();
    decoder[[mu, out]] += signed_step;
    endpoint.atoms[atom]
        .set_decoder_coefficients(decoder)
        .expect("a same-shape decoder");
    let cache = endpoint_cache(&mut endpoint, target, rho);
    endpoint
        .materialize_exact_hessian_dense(rho, target.view(), &cache)
        .expect("dense exact A at a finite-difference endpoint")
}

/// Central differences, Richardson-extrapolated over `h, h/2, h/4`, of the dense
/// exact `A` and of `log|det A|` along one `t` slot. Each oracle is the finer
/// extrapolation minus the coarser one, charged to the same budget as the gap.
struct SlotDifference {
    d_a: Array2<f64>,
    d_a_oracle: Array2<f64>,
    d_log_det: f64,
    d_log_det_oracle: f64,
}

fn slot_difference(
    anchor: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    row: usize,
    slot: usize,
    var: SaeLocalRowVar,
) -> SlotDifference {
    let mut d_a = Vec::with_capacity(3);
    let mut d_log_det = Vec::with_capacity(3);
    for divisor in [1.0_f64, 2.0, 4.0] {
        let h = STEP / divisor;
        let plus = endpoint_exact_a(anchor, target, rho, cache, row, slot, var, h);
        let minus = endpoint_exact_a(anchor, target, rho, cache, row, slot, var, -h);
        d_a.push((&plus - &minus) / (2.0 * h));
        d_log_det.push((log_abs_det(&plus) - log_abs_det(&minus)) / (2.0 * h));
    }
    let fine = (&d_a[2] * 4.0 - &d_a[1]) / 3.0;
    let coarse = (&d_a[1] * 4.0 - &d_a[0]) / 3.0;
    let fine_log_det = (4.0 * d_log_det[2] - d_log_det[1]) / 3.0;
    let coarse_log_det = (4.0 * d_log_det[1] - d_log_det[0]) / 3.0;
    SlotDifference {
        d_a_oracle: (&fine - &coarse).mapv(f64::abs),
        d_a: fine,
        d_log_det: fine_log_det,
        d_log_det_oracle: (fine_log_det - coarse_log_det).abs(),
    }
}

/// The Richardson central difference of `log|det A|` along decoder coefficient
/// `(mu, out)` of `atom`, with its finer-minus-coarser oracle.
fn decoder_log_det_difference(
    anchor: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    atom: usize,
    mu: usize,
    out: usize,
) -> (f64, f64) {
    let mut estimates = [0.0_f64; 3];
    for (position, divisor) in [1.0_f64, 2.0, 4.0].into_iter().enumerate() {
        let h = STEP / divisor;
        let plus = endpoint_decoder_exact_a(anchor, target, rho, atom, mu, out, h);
        let minus = endpoint_decoder_exact_a(anchor, target, rho, atom, mu, out, -h);
        estimates[position] = (log_abs_det(&plus) - log_abs_det(&minus)) / (2.0 * h);
    }
    let fine = (4.0 * estimates[2] - estimates[1]) / 3.0;
    let coarse = (4.0 * estimates[1] - estimates[0]) / 3.0;
    (fine, (fine - coarse).abs())
}

/// Full-basis probes of the reduced Schur and their exact solves: at these probes the
/// from-probes θ-adjoint reconstructs the cache's inverse exactly.
fn full_basis_bundle(cache: &ArrowFactorCache) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
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

/// The exact-A evidence factor the matrix-free outer gradient consumes
/// (`BundleEvidenceGeometry::cache`): the majorizer system corrected to `A = B + ΔC`,
/// carrying its classification geometry, factored under that lane's refusing
/// unit-deflation policy.
fn exact_a_evidence_factor(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let mut majorizer = term
        .assemble_arrow_schur(target.view(), rho, None)
        .expect("majorizer assembly for the exact-A evidence factor");
    SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut majorizer);
    let exact = term
        .exact_a_evidence_system(target.view(), rho, &majorizer, 1.0)
        .expect("exact-A evidence system");
    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR);
    let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&exact, 0.0, 0.0, &options)
        .expect("exact-A evidence factor");
    cache
}

/// A factor's discrete stratum: deflated directions, recorded row spectra (a clamp
/// basin or a repriced eigenvalue), and whether the reduced Schur was conditioned.
fn evidence_stratum(cache: &ArrowFactorCache) -> (usize, usize, bool) {
    (
        cache.deflated_row_directions.iter().map(Vec::len).sum(),
        cache
            .deflation_row_spectra
            .iter()
            .filter(|spectrum| spectrum.is_some())
            .count(),
        cache.beta_schur_conditioning.is_some(),
    )
}

/// The product-rule term of `∂³(a_k·γ_k)` a triple of row variables selects, and
/// whether a logit in it belongs to a different atom than its coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TripleKind {
    ThreeLogits { distinct: bool },
    TwoLogitsOneCoordinate { cross_atom: bool },
    OneLogitTwoCoordinates { cross_atom: bool },
    ThreeCoordinates,
    CoordinatesOfTwoAtoms,
}

fn triple_kind(vars: [SaeLocalRowVar; 3]) -> TripleKind {
    let mut logit_atoms = Vec::with_capacity(3);
    let mut coord_atoms = Vec::with_capacity(3);
    for var in vars {
        match var {
            SaeLocalRowVar::Logit { atom } => logit_atoms.push(atom),
            SaeLocalRowVar::Coord { atom, .. } => coord_atoms.push(atom),
        }
    }
    if coord_atoms.iter().any(|&atom| atom != coord_atoms[0]) {
        return TripleKind::CoordinatesOfTwoAtoms;
    }
    let cross_atom = coord_atoms
        .first()
        .is_some_and(|&owner| logit_atoms.iter().any(|&atom| atom != owner));
    match coord_atoms.len() {
        0 => TripleKind::ThreeLogits {
            distinct: logit_atoms[0] != logit_atoms[1]
                && logit_atoms[1] != logit_atoms[2]
                && logit_atoms[0] != logit_atoms[2],
        },
        1 => TripleKind::TwoLogitsOneCoordinate { cross_atom },
        2 => TripleKind::OneLogitTwoCoordinates { cross_atom },
        _ => TripleKind::ThreeCoordinates,
    }
}

struct CategoryReadout {
    count: usize,
    worst_gap: f64,
    worst_control_gap: f64,
    largest_difference: f64,
}

/// Every row-local entry `∂A_ab/∂θ_w` the dense exact-A θ-adjoint contracts, against a
/// central difference of the materialized `A`.
///
/// A unit inverse `e_b e_aᵀ` turns `Γ_w = Σ inv[b,a]·dh_ab(w)` into the single entry
/// `∂A_ab/∂θ_w`, so every triple `(a, b, w)` of a probe row is read separately and
/// reported in its product-rule category.
///
/// The categories whose residual leg must be visible differ by gate family. A softmax
/// logit moves every gate, so all seven categories carry the leg. An independent gate
/// moves only its own atom, so a triple mixing a logit with another atom reads an exact
/// zero leg, and only the same-atom categories are required to show it.
fn check_derivative_entries(gate: FixtureGate) {
    let (term, target, rho) = residual_fixture(gate);
    let (anchor, cache) = anchored_state(&term, &target, &rho);
    assert_anchor_deflation_free(&format!("dA {gate:?}"), &anchor, &cache);
    assert_exact_row_blocks_positive_definite(&anchor, &target, &rho, &cache);
    let dim = cache.delta_t_len() + cache.k;
    let mut readouts: Vec<(TripleKind, CategoryReadout)> = Vec::new();
    for row in PROBE_ROWS {
        let q = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let vars = anchor
            .row_vars_for_cache_row(row, &cache)
            .expect("row variables of a probe row");
        let differences: Vec<SlotDifference> = (0..q)
            .map(|w| slot_difference(&anchor, &target, &rho, &cache, row, base + w, vars[w]))
            .collect();
        for a in 0..q {
            for b in a..q {
                let mut unit = Array2::<f64>::zeros((dim, dim));
                unit[[base + b, base + a]] = 1.0;
                let with_target = anchor
                    .logdet_theta_adjoint_dense(&rho, &cache, &unit, true, true, Some(target.view()))
                    .expect("dense exact-A contraction with the data target");
                let without_target = anchor
                    .logdet_theta_adjoint_dense(&rho, &cache, &unit, true, true, None)
                    .expect("dense exact-A contraction without the data target");
                for w in 0..q {
                    let fd = differences[w].d_a[[base + a, base + b]];
                    let oracle = differences[w].d_a_oracle[[base + a, base + b]];
                    let analytic = with_target.t[base + w];
                    let control = without_target.t[base + w];
                    let scale = 1.0 + fd.abs();
                    let gap = ((analytic - fd).abs() + oracle) / scale;
                    let control_gap = (control - fd).abs() / scale;
                    let kind = triple_kind([vars[a], vars[b], vars[w]]);
                    eprintln!(
                        "[#2933 F01 dA {gate:?}] row={row} a={:?} b={:?} w={:?} kind={kind:?} \
                         fd={fd:.9e} analytic={analytic:.9e} without_target={control:.9e} \
                         gap={gap:.3e} control_gap={control_gap:.3e} oracle={oracle:.3e}",
                        vars[a], vars[b], vars[w],
                    );
                    let position = match readouts.iter().position(|(seen, _)| *seen == kind) {
                        Some(position) => position,
                        None => {
                            readouts.push((
                                kind,
                                CategoryReadout {
                                    count: 0,
                                    worst_gap: 0.0,
                                    worst_control_gap: 0.0,
                                    largest_difference: 0.0,
                                },
                            ));
                            readouts.len() - 1
                        }
                    };
                    let readout = &mut readouts[position].1;
                    readout.count += 1;
                    readout.worst_gap = readout.worst_gap.max(gap);
                    readout.worst_control_gap = readout.worst_control_gap.max(control_gap);
                    readout.largest_difference = readout.largest_difference.max(fd.abs());
                }
            }
        }
    }
    let summary: Vec<String> = readouts
        .iter()
        .map(|(kind, readout)| {
            format!(
                "{kind:?}: n={} worst_gap={:.3e} worst_control_gap={:.3e} max|fd|={:.3e}",
                readout.count,
                readout.worst_gap,
                readout.worst_control_gap,
                readout.largest_difference,
            )
        })
        .collect();
    for line in &summary {
        eprintln!("[#2933 F01 dA summary {gate:?}] {line}");
    }
    let worst_gap = readouts
        .iter()
        .map(|(_, readout)| readout.worst_gap)
        .fold(0.0_f64, f64::max);
    assert!(
        worst_gap <= FD_TOL,
        "#2933 F01 ({gate:?}): the dense exact-A θ-adjoint's dA entries do not match a central \
         difference of the materialized A (worst relative gap {worst_gap:.3e} > {FD_TOL:.1e}); \
         per category: {summary:#?}"
    );
    let required: &[TripleKind] = match gate {
        FixtureGate::Softmax => &[
            TripleKind::ThreeLogits { distinct: true },
            TripleKind::ThreeLogits { distinct: false },
            TripleKind::TwoLogitsOneCoordinate { cross_atom: true },
            TripleKind::TwoLogitsOneCoordinate { cross_atom: false },
            TripleKind::OneLogitTwoCoordinates { cross_atom: true },
            TripleKind::OneLogitTwoCoordinates { cross_atom: false },
            TripleKind::ThreeCoordinates,
        ],
        FixtureGate::ThresholdGate => &[
            TripleKind::ThreeLogits { distinct: false },
            TripleKind::TwoLogitsOneCoordinate { cross_atom: false },
            TripleKind::OneLogitTwoCoordinates { cross_atom: false },
            TripleKind::ThreeCoordinates,
        ],
    };
    for &required in required {
        let readout = readouts
            .iter()
            .find(|(kind, _)| *kind == required)
            .map(|(_, readout)| readout)
            .unwrap_or_else(|| panic!("#2933 F01 ({gate:?}): no triple of kind {required:?} was probed"));
        assert!(
            readout.worst_control_gap >= CONTROL_MARGIN * FD_TOL,
            "#2933 F01 positive control ({gate:?}): dropping the residual third leg must visibly \
             break the {required:?} entries, but the target-free contraction still matched to \
             {:.3e}; this gate cannot see that leg there. Per category: {summary:#?}",
            readout.worst_control_gap,
        );
    }
}

/// `Γ_w = tr(A⁻¹ ∂A/∂θ_w)`, the dense exact-A θ-adjoint contracted with the exact inverse
/// of the materialized `A`, against a central difference of `log|det A|`: the derivative
/// of the ranked `½log|A|` in each probe-row slot.
fn check_logdet_trace(gate: FixtureGate) {
    let (term, target, rho) = residual_fixture(gate);
    let (anchor, cache) = anchored_state(&term, &target, &rho);
    assert_anchor_deflation_free(&format!("tr {gate:?}"), &anchor, &cache);
    assert_exact_row_blocks_positive_definite(&anchor, &target, &rho, &cache);
    let a = anchor
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("dense exact A at the anchor");
    let inverse = nonsingular_inverse(&format!("tr {gate:?}"), &a);
    let with_target = anchor
        .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, Some(target.view()))
        .expect("dense exact-A θ-adjoint with the data target");
    let without_target = anchor
        .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, None)
        .expect("dense exact-A θ-adjoint without the data target");
    let mut worst_gap = 0.0_f64;
    let mut worst_logit_control_gap = 0.0_f64;
    for row in PROBE_ROWS {
        let q = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let vars = anchor
            .row_vars_for_cache_row(row, &cache)
            .expect("row variables of a probe row");
        for w in 0..q {
            let difference = slot_difference(&anchor, &target, &rho, &cache, row, base + w, vars[w]);
            let fd = difference.d_log_det;
            let analytic = with_target.t[base + w];
            let control = without_target.t[base + w];
            let scale = 1.0 + fd.abs();
            let gap = ((analytic - fd).abs() + difference.d_log_det_oracle) / scale;
            let control_gap = (control - fd).abs() / scale;
            eprintln!(
                "[#2933 F01 tr {gate:?}] row={row} w={:?} fd={fd:.9e} analytic={analytic:.9e} \
                 without_target={control:.9e} gap={gap:.3e} control_gap={control_gap:.3e} \
                 oracle={:.3e}",
                vars[w], difference.d_log_det_oracle,
            );
            worst_gap = worst_gap.max(gap);
            if matches!(vars[w], SaeLocalRowVar::Logit { .. }) {
                worst_logit_control_gap = worst_logit_control_gap.max(control_gap);
            }
        }
    }
    assert!(
        worst_gap <= FD_TOL,
        "#2933 F01 ({gate:?}): tr(A⁻¹ ∂A/∂θ) from the dense exact-A θ-adjoint does not match a \
         central difference of log|det A| (worst relative gap {worst_gap:.3e} > {FD_TOL:.1e})"
    );
    assert!(
        worst_logit_control_gap >= CONTROL_MARGIN * FD_TOL,
        "#2933 F01 positive control ({gate:?}): dropping the residual third leg must visibly move \
         the logit slots of tr(A⁻¹ ∂A/∂θ), but it still matched to {worst_logit_control_gap:.3e}"
    );
}

/// #2933 F01 — softmax arm of [`check_derivative_entries`].
#[test]
fn softmax_exact_a_derivative_entries_match_finite_difference_2933() {
    check_derivative_entries(FixtureGate::Softmax);
}

/// #2933 F01 — ThresholdGate arm of [`check_derivative_entries`]: the independent-gate
/// logit legs a threshold fit used to read as zero.
#[test]
fn threshold_gate_exact_a_derivative_entries_match_finite_difference_2933() {
    check_derivative_entries(FixtureGate::ThresholdGate);
}

/// #2933 F01 — softmax arm of [`check_logdet_trace`].
#[test]
fn softmax_exact_a_logdet_theta_adjoint_matches_finite_difference_2933() {
    check_logdet_trace(FixtureGate::Softmax);
}

/// #2933 F01 — ThresholdGate arm of [`check_logdet_trace`].
#[test]
fn threshold_gate_exact_a_logdet_theta_adjoint_matches_finite_difference_2933() {
    check_logdet_trace(FixtureGate::ThresholdGate);
}

/// #2933 F01 — the border slots of `tr(A⁻¹ ∂A/∂θ)`: every decoder coefficient against a
/// central difference of `log|det A|`. Their residual legs come from
/// `patchd_residual_third_leg_beta`, whose softmax gate factors no other gate checks at a
/// nonzero residual.
#[test]
fn softmax_exact_a_logdet_theta_adjoint_beta_slots_match_finite_difference_2933() {
    let (term, target, rho) = residual_fixture(FixtureGate::Softmax);
    let (anchor, cache) = anchored_state(&term, &target, &rho);
    assert_anchor_deflation_free("tr beta", &anchor, &cache);
    assert_exact_row_blocks_positive_definite(&anchor, &target, &rho, &cache);
    let a = anchor
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("dense exact A at the anchor");
    let inverse = nonsingular_inverse("tr beta", &a);
    let with_target = anchor
        .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, Some(target.view()))
        .expect("dense exact-A θ-adjoint with the data target");
    let without_target = anchor
        .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, None)
        .expect("dense exact-A θ-adjoint without the data target");
    let offsets = anchor.beta_offsets();
    let p = anchor.output_dim();
    let mut worst_gap = 0.0_f64;
    let mut worst_control_gap = 0.0_f64;
    let mut count = 0usize;
    for atom in 0..anchor.atoms.len() {
        let (basis, _) = anchor.atoms[atom].decoder_coefficients().dim();
        for mu in 0..basis {
            for out in 0..p {
                let index = offsets[atom] + mu * p + out;
                let (fd, oracle) = decoder_log_det_difference(&anchor, &target, &rho, atom, mu, out);
                let analytic = with_target.beta[index];
                let control = without_target.beta[index];
                let scale = 1.0 + fd.abs();
                let gap = ((analytic - fd).abs() + oracle) / scale;
                let control_gap = (control - fd).abs() / scale;
                eprintln!(
                    "[#2933 F01 tr beta] atom={atom} mu={mu} out={out} fd={fd:.9e} \
                     analytic={analytic:.9e} without_target={control:.9e} gap={gap:.3e} \
                     control_gap={control_gap:.3e} oracle={oracle:.3e}"
                );
                worst_gap = worst_gap.max(gap);
                worst_control_gap = worst_control_gap.max(control_gap);
                count += 1;
            }
        }
    }
    eprintln!(
        "[#2933 F01 tr beta summary] n={count} worst_gap={worst_gap:.3e} \
         worst_control_gap={worst_control_gap:.3e}"
    );
    assert!(
        worst_gap <= FD_TOL,
        "#2933 F01: the border slots of tr(A⁻¹ ∂A/∂θ) do not match a central difference of \
         log|det A| (worst relative gap {worst_gap:.3e} > {FD_TOL:.1e} over {count} coefficients)"
    );
    assert!(
        worst_control_gap >= CONTROL_MARGIN * FD_TOL,
        "#2933 F01 positive control: dropping the residual third legs must visibly move the border \
         slots, but the target-free adjoint still matched to {worst_control_gap:.3e}"
    );
}

/// #2933 F01 — the from-probes θ-adjoint on the exact-A evidence factor the matrix-free
/// outer gradient consumes, against a central difference of that factor's own
/// log-determinant at a nonzero residual.
///
/// The route-parity test below checks this route against the dense one through a shared
/// helper; this one arbitrates the route directly, so a leg missing from from-probes
/// alone, or from the helper both routes call, is visible here.
#[test]
fn from_probes_exact_a_logdet_theta_adjoint_matches_finite_difference_2933() {
    let (term, target, rho) = residual_fixture(FixtureGate::Softmax);
    let mut anchor = term.clone();
    let cache = exact_a_evidence_factor(&mut anchor, &target, &rho);
    anchor.streaming_gates_frozen = true;
    let stratum = evidence_stratum(&cache);
    eprintln!(
        "[#2933 F01 probes fd] anchor stratum: deflated directions={} recorded row spectra={} \
         reduced Schur conditioned={}",
        stratum.0, stratum.1, stratum.2
    );
    assert_eq!(
        stratum,
        (0, 0, false),
        "#2933 F01 premise: the exact-A evidence factor must be deflation-free, record no row \
         spectrum and leave the reduced Schur unconditioned, or its log-determinant is not one \
         smooth branch"
    );
    let (probes, sinv) = full_basis_bundle(&cache);
    let with_target = anchor
        .logdet_theta_adjoint_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::ExactObservedInformation,
            Some(target.view()),
        )
        .expect("from-probes exact-A θ-adjoint with the data target");
    let without_target = anchor
        .logdet_theta_adjoint_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::ExactObservedInformation,
            None,
        )
        .expect("from-probes exact-A θ-adjoint without the data target");
    let mut worst_gap = 0.0_f64;
    let mut worst_logit_control_gap = 0.0_f64;
    for row in PROBE_ROWS {
        let q = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let vars = anchor
            .row_vars_for_cache_row(row, &cache)
            .expect("row variables of a probe row");
        for w in 0..q {
            let log_det = |signed_step: f64| -> f64 {
                let mut endpoint =
                    moved_endpoint(&anchor, &cache, row, base + w, vars[w], signed_step);
                let endpoint_cache = exact_a_evidence_factor(&mut endpoint, &target, &rho);
                assert_eq!(
                    evidence_stratum(&endpoint_cache),
                    stratum,
                    "#2933 F01 premise: a finite-difference endpoint must sit on the anchor's stratum"
                );
                endpoint_cache
                    .arrow_log_det()
                    .expect("authoritative joint log-determinant of the exact-A evidence factor")
            };
            let mut estimates = [0.0_f64; 3];
            for (position, divisor) in [1.0_f64, 2.0, 4.0].into_iter().enumerate() {
                let h = STEP / divisor;
                estimates[position] = (log_det(h) - log_det(-h)) / (2.0 * h);
            }
            let fd = (4.0 * estimates[2] - estimates[1]) / 3.0;
            let oracle = (fd - (4.0 * estimates[1] - estimates[0]) / 3.0).abs();
            let analytic = with_target.t[base + w];
            let control = without_target.t[base + w];
            let scale = 1.0 + fd.abs();
            let gap = ((analytic - fd).abs() + oracle) / scale;
            let control_gap = (control - fd).abs() / scale;
            eprintln!(
                "[#2933 F01 probes fd] row={row} w={:?} fd={fd:.9e} analytic={analytic:.9e} \
                 without_target={control:.9e} gap={gap:.3e} control_gap={control_gap:.3e} \
                 oracle={oracle:.3e}",
                vars[w],
            );
            worst_gap = worst_gap.max(gap);
            if matches!(vars[w], SaeLocalRowVar::Logit { .. }) {
                worst_logit_control_gap = worst_logit_control_gap.max(control_gap);
            }
        }
    }
    assert!(
        worst_gap <= FD_TOL,
        "#2933 F01: the from-probes θ-adjoint does not match a central difference of the exact-A \
         evidence factor's log-determinant (worst relative gap {worst_gap:.3e} > {FD_TOL:.1e})"
    );
    assert!(
        worst_logit_control_gap >= CONTROL_MARGIN * FD_TOL,
        "#2933 F01 positive control: dropping the residual third leg must visibly move the logit \
         slots of the from-probes θ-adjoint, but it still matched to {worst_logit_control_gap:.3e}"
    );
}

/// #2933 F01 — the operator every test above differentiates must itself be the Hessian of
/// the objective the inner solve drives to stationarity, at a nonzero residual.
///
/// The gates above certify `∂(materialized A)/∂θ`; a term missing from `A` itself would be
/// missing on both sides of every one of them. Here `A` is compared column by column with
/// a central difference of the assembled KKT gradient `(gt, gb)`, after checking that
/// gradient against `penalized_objective_total`. The control is `A` materialized at the
/// target with the fixture's offset removed, the reconstruction the fixture drew it from,
/// where `A` carries no residual curvature.
#[test]
fn softmax_exact_a_is_the_derivative_of_the_kkt_gradient_2933() {
    let (term, target, rho) = residual_fixture(FixtureGate::Softmax);
    let (anchor, cache) = anchored_state(&term, &target, &rho);
    assert_anchor_deflation_free("kkt", &anchor, &cache);
    assert_exact_row_blocks_positive_definite(&anchor, &target, &rho, &cache);
    let a = anchor
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("dense exact A at the anchor");
    let total_t = cache.delta_t_len();
    let k = cache.k;
    let dim = total_t + k;
    assert_eq!(
        a.nrows(),
        dim,
        "#2933 F01 premise: A must be (total_t + k) square to be compared with (gt, gb)"
    );
    let gradient = |state: &mut SaeManifoldTerm| -> Array1<f64> {
        let sys = state
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("arrow-Schur assembly at a finite-difference endpoint");
        let mut g = Array1::<f64>::zeros(dim);
        let mut offset = 0usize;
        for row in &sys.rows {
            for (axis, &value) in row.gt.iter().enumerate() {
                g[offset + axis] = value;
            }
            offset += row.gt.len();
        }
        assert_eq!(
            offset, total_t,
            "#2933 F01 premise: the concatenated per-row gt width must equal cache.delta_t_len()"
        );
        assert_eq!(
            sys.gb.len(),
            k,
            "#2933 F01 premise: the border gradient width must equal cache.k"
        );
        for (axis, &value) in sys.gb.iter().enumerate() {
            g[total_t + axis] = value;
        }
        g
    };

    let g0 = gradient(&mut frozen_endpoint(&anchor));
    let mut premise_gap = 0.0_f64;
    let mut premise_scale = 0.0_f64;
    for index in 0..dim {
        let objective = |signed_step: f64| -> f64 {
            joint_endpoint(&anchor, total_t, k, index, signed_step)
                .penalized_objective_total(target.view(), &rho, None, 1.0)
                .expect("penalized objective at a finite-difference endpoint")
        };
        let mut estimates = [0.0_f64; 3];
        for (position, divisor) in [1.0_f64, 2.0, 4.0].into_iter().enumerate() {
            let h = STEP / divisor;
            estimates[position] = (objective(h) - objective(-h)) / (2.0 * h);
        }
        let fd = (4.0 * estimates[2] - estimates[1]) / 3.0;
        let oracle = (fd - (4.0 * estimates[1] - estimates[0]) / 3.0).abs();
        premise_gap = premise_gap.max(((g0[index] - fd).abs() + oracle) / (1.0 + fd.abs()));
        premise_scale = premise_scale.max(fd.abs());
    }
    eprintln!(
        "[#2933 F01 kkt] premise: worst relative gap of (gt, gb) against ∇penalized_objective_total \
         = {premise_gap:.3e}, max|∂P| = {premise_scale:.3e}"
    );
    assert!(
        premise_scale >= CONTROL_MARGIN * FD_TOL,
        "#2933 F01 premise: the fixture's gradient must be resolved above the bar \
         (max|∂P|={premise_scale:.3e}), or the gradient check is zero against zero"
    );
    assert!(
        premise_gap <= FD_TOL,
        "#2933 F01 premise: the assembled (gt, gb) is not the gradient of \
         penalized_objective_total (worst relative gap {premise_gap:.3e} > {FD_TOL:.1e}); a \
         central difference of a non-gradient is no Hessian"
    );

    let (n, p) = target.dim();
    let reconstruction_target = &target - &residual_offset(n, p);
    let a_without_residual = anchor
        .materialize_exact_hessian_dense(&rho, reconstruction_target.view(), &cache)
        .expect("dense exact A at the offset-free target");
    let mut worst_gap = 0.0_f64;
    let mut worst_control_gap = 0.0_f64;
    let mut largest = 0.0_f64;
    for column in 0..dim {
        let mut estimates: Vec<Array1<f64>> = Vec::with_capacity(3);
        for divisor in [1.0_f64, 2.0, 4.0] {
            let h = STEP / divisor;
            let plus = gradient(&mut joint_endpoint(&anchor, total_t, k, column, h));
            let minus = gradient(&mut joint_endpoint(&anchor, total_t, k, column, -h));
            estimates.push((&plus - &minus) / (2.0 * h));
        }
        let fine = (&estimates[2] * 4.0 - &estimates[1]) / 3.0;
        let coarse = (&estimates[1] * 4.0 - &estimates[0]) / 3.0;
        for row_index in 0..dim {
            let fd = fine[row_index];
            let scale = 1.0 + fd.abs();
            let gap = ((a[[row_index, column]] - fd).abs() + (fd - coarse[row_index]).abs()) / scale;
            let control_gap = (a_without_residual[[row_index, column]] - fd).abs() / scale;
            worst_gap = worst_gap.max(gap);
            worst_control_gap = worst_control_gap.max(control_gap);
            largest = largest.max(fd.abs());
        }
    }
    eprintln!(
        "[#2933 F01 kkt] columns={dim} worst relative gap={worst_gap:.3e} \
         worst control gap={worst_control_gap:.3e} max|∂g|={largest:.3e}"
    );
    assert!(
        worst_gap <= FD_TOL,
        "#2933 F01: the materialized exact A is not the central difference of the KKT gradient \
         (worst relative gap {worst_gap:.3e} > {FD_TOL:.1e})"
    );
    assert!(
        worst_control_gap >= CONTROL_MARGIN * FD_TOL,
        "#2933 F01 positive control: A without residual curvature must visibly miss the KKT \
         gradient's difference, but it still matched to {worst_control_gap:.3e}"
    );
}

/// #2933 F01 — the bundle / matrix-free θ-adjoint `logdet_theta_adjoint_from_probes`
/// receives the data target in production
/// (`analytic_outer_rho_gradient_components_with_bundle`) and must carry the same
/// residual third leg as the dense route. At full-basis probes it reconstructs `B⁻¹`
/// exactly, so the leg's contribution (the adjoint with the target minus the adjoint
/// without it) must equal the dense route's on the same inverse, and must be far
/// from zero.
///
/// This is route parity, not the defect's arbiter: both towers call one
/// `patchd_residual_third_leg`, so a leg missing from that helper is missing from
/// both and the parity still holds (it did at F01's parent). The central-difference
/// tests above arbitrate the leg; this test carries their verdict to the from-probes
/// route on the majorizer's inverse.
///
/// Bars: the leg must clear the magnitude floor every positive control uses,
/// `CONTROL_MARGIN·FD_TOL`, so the comparison is not zero against zero. Because the
/// parity carries a verdict resolved at `FD_TOL`, it must hold two decades tighter,
/// `(FD_TOL/CONTROL_MARGIN)·(1 + max|leg|)`.
#[test]
fn from_probes_residual_third_leg_matches_the_dense_route_2933() {
    let (term, target, rho) = residual_fixture(FixtureGate::Softmax);
    let (anchor, cache) = anchored_state(&term, &target, &rho);
    assert_anchor_deflation_free("probes parity", &anchor, &cache);
    assert_exact_row_blocks_positive_definite(&anchor, &target, &rho, &cache);
    assert!(
        cache.beta_schur_conditioning.is_none(),
        "#2933 F01 premise: a reduced-Schur basin reweights the from-probes row inverse, so the \
         two routes would contract the leg against different inverses"
    );
    let solver = DeflatedArrowSolver::plain(&cache);
    let inverse = anchor
        .materialize_joint_inverse(&cache, &solver)
        .expect("dense joint inverse of B");
    let (probes, sinv) = full_basis_bundle(&cache);
    let flat = |v: &SaeArrowVector| -> Vec<f64> { v.t.iter().chain(v.beta.iter()).copied().collect() };
    let leg = |with: &SaeArrowVector, without: &SaeArrowVector| -> Vec<f64> {
        flat(with)
            .into_iter()
            .zip(flat(without))
            .map(|(x, y)| x - y)
            .collect()
    };
    let dense_leg = leg(
        &anchor
            .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, Some(target.view()))
            .expect("dense exact-A θ-adjoint with the data target"),
        &anchor
            .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, None)
            .expect("dense exact-A θ-adjoint without the data target"),
    );
    let probes_leg = leg(
        &anchor
            .logdet_theta_adjoint_from_probes(
                &rho,
                &cache,
                &probes,
                &sinv,
                EvidenceOperator::ExactObservedInformation,
                Some(target.view()),
            )
            .expect("from-probes exact-A θ-adjoint with the data target"),
        &anchor
            .logdet_theta_adjoint_from_probes(
                &rho,
                &cache,
                &probes,
                &sinv,
                EvidenceOperator::ExactObservedInformation,
                None,
            )
            .expect("from-probes exact-A θ-adjoint without the data target"),
    );
    let largest = dense_leg.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let gap = dense_leg
        .iter()
        .zip(&probes_leg)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    let leg_floor = CONTROL_MARGIN * FD_TOL;
    let parity_bar = (FD_TOL / CONTROL_MARGIN) * (1.0 + largest);
    eprintln!(
        "[#2933 F01 probes] max|dense leg|={largest:.6e} (floor CONTROL_MARGIN·FD_TOL={leg_floor:.3e}) \
         max|probes − dense|={gap:.6e} (bar (FD_TOL/CONTROL_MARGIN)·(1+largest)={parity_bar:.3e})"
    );
    assert!(
        largest >= leg_floor,
        "#2933 F01: the residual third leg must carry real weight on this fixture \
         (max|leg|={largest:.3e} < {leg_floor:.3e}), or the parity below is a zero-vs-zero comparison"
    );
    assert!(
        gap <= parity_bar,
        "#2933 F01: the from-probes θ-adjoint's residual third leg differs from the dense route's \
         on the same inverse by {gap:.3e} (bar {parity_bar:.3e}, leg scale {largest:.3e})"
    );
}
