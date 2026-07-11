//! Penalized-objective finite-difference gradient contract tests, split verbatim
//! out of `tests.rs` to keep that tracked file under the #780 10k-line gate.
//! Declared as a sibling `#[cfg(test)] mod` in `mod.rs`.
//!
//! Covers the assembled-gradient-vs-central-FD checks for the analytic penalty
//! family (`sae_assembled_gradient_matches_penalized_objective_central_fd`,
//! `sae_d1_assembled_gradient_matches_loss_central_fd`,
//! `sae_reml_extra_penalty_energy_counts_live_isometry_once`) and their shared
//! `SaeFd*` / `sae_fd_*` / `sae_pen_*` fixtures. All production symbols are in
//! scope via `super::*`.

use super::*;
use approx::assert_abs_diff_eq;
use gam_terms::analytic_penalties::ARDPenalty;
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct SaeFdWorst {
    pub(crate) index: usize,
    pub(crate) analytic: f64,
    pub(crate) finite_difference: f64,
    pub(crate) absolute_error: f64,
    pub(crate) relative_error: f64,
}

impl SaeFdWorst {
    pub(crate) fn new() -> Self {
        Self {
            index: 0,
            analytic: 0.0,
            finite_difference: 0.0,
            absolute_error: 0.0,
            relative_error: 0.0,
        }
    }

    pub(crate) fn observe(&mut self, index: usize, analytic: f64, finite_difference: f64) {
        let absolute_error = (analytic - finite_difference).abs();
        let scale = analytic.abs().max(finite_difference.abs()).max(1.0e-9);
        let relative_error = absolute_error / scale;
        if relative_error > self.relative_error {
            self.index = index;
            self.analytic = analytic;
            self.finite_difference = finite_difference;
            self.absolute_error = absolute_error;
            self.relative_error = relative_error;
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SaeFdBlockReport {
    pub(crate) label: String,
    pub(crate) base_loss: f64,
    pub(crate) coord: SaeFdWorst,
    pub(crate) decoder: SaeFdWorst,
}

pub(crate) fn sae_fd_decoder(n_basis: usize, p_out: usize) -> Array2<f64> {
    let mut decoder = Array2::<f64>::zeros((n_basis, p_out));
    for basis in 0..n_basis {
        for out_col in 0..p_out {
            let phase = 0.73 * ((basis + 1) as f64) + 1.17 * ((out_col + 1) as f64);
            decoder[[basis, out_col]] = 0.16 * phase.sin() + 0.05 * (1.9 * phase).cos();
        }
    }
    decoder
}

pub(crate) fn sae_fd_target(n_obs: usize, p_out: usize) -> Array2<f64> {
    let mut target = Array2::<f64>::zeros((n_obs, p_out));
    for row in 0..n_obs {
        for out_col in 0..p_out {
            let x = (row as f64) + 1.0;
            let y = (out_col as f64) + 1.0;
            target[[row, out_col]] =
                0.21 * (0.31 * x + 0.47 * y).sin() - 0.13 * (0.19 * x * y).cos();
        }
    }
    target
}

pub(crate) fn sae_fd_coords(label: &str, n_obs: usize) -> Array2<f64> {
    let mut coords = Array2::<f64>::zeros((n_obs, 1));
    for row in 0..n_obs {
        let x = row as f64;
        coords[[row, 0]] = match label {
            "periodic_d1" => 0.07 + 0.043 * x + 0.004 * (1.3 * x).sin(),
            "euclidean_d1" => -0.46 + 0.048 * x + 0.006 * (1.7 * x).cos(),
            other => panic!("unknown SAE FD case label {other}"),
        };
    }
    coords
}

pub(crate) fn sae_fd_term(label: &str) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n_obs = 20usize;
    let p_out = 3usize;
    let coords = sae_fd_coords(label, n_obs);
    let (basis_kind, phi, jet, n_basis, atom) = match label {
        "periodic_d1" => {
            let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let n_basis = phi.ncols();
            let atom = SaeManifoldAtom::new(
                "periodic_d1",
                SaeAtomBasisKind::Periodic,
                1,
                phi.clone(),
                jet.clone(),
                sae_fd_decoder(n_basis, p_out),
                Array2::<f64>::eye(n_basis),
            )
            .unwrap()
            .with_basis_second_jet(evaluator);
            (SaeAtomBasisKind::Periodic, phi, jet, n_basis, atom)
        }
        "euclidean_d1" => {
            let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let n_basis = phi.ncols();
            let atom = SaeManifoldAtom::new(
                "euclidean_d1",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi.clone(),
                jet.clone(),
                sae_fd_decoder(n_basis, p_out),
                Array2::<f64>::eye(n_basis),
            )
            .unwrap()
            .with_basis_second_jet(evaluator);
            (SaeAtomBasisKind::EuclideanPatch, phi, jet, n_basis, atom)
        }
        other => panic!("unknown SAE FD case label {other}"),
    };
    assert_eq!(
        basis_kind.latent_manifold(1),
        atom.basis_kind.latent_manifold(1)
    );
    assert_eq!(phi.dim(), (n_obs, n_basis));
    assert_eq!(jet.dim(), (n_obs, n_basis, 1));

    let manifold = atom.basis_kind.latent_manifold(1);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n_obs, 1)),
        vec![coords],
        vec![manifold],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = sae_fd_target(n_obs, p_out);
    let rho = SaeManifoldRho::new(0.0, 1.0e-4_f64.ln(), vec![array![-30.0]]);
    (term, target, rho)
}

pub(crate) fn sae_fd_refresh(term: &mut SaeManifoldTerm) {
    let coords = term.assignment.coords[0].as_matrix();
    term.atoms[0].refresh_basis(coords.view()).unwrap();
}

pub(crate) fn sae_fd_set_coord(term: &mut SaeManifoldTerm, row: usize, value: f64) {
    let mut flat = term.assignment.coords[0].as_flat().clone();
    flat[row] = value;
    term.assignment.coords[0].set_flat(flat.view());
    sae_fd_refresh(term);
}

pub(crate) fn sae_fd_total_loss(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> f64 {
    term.loss(target.view(), rho).unwrap().total()
}

pub(crate) fn sae_fd_check_case(label: &str) -> SaeFdBlockReport {
    let epsilon = 1.0e-6;
    let (term, target, rho) = sae_fd_term(label);
    // Freeze the intrinsic bending Gram S̃ at the base (B, t) exactly as the
    // production assembly entry does (`assemble_arrow_schur` calls
    // `refresh_intrinsic_smooth_penalty` on every atom before assembling gb, the
    // #673 lagged-diffusivity chokepoint). Production's value path (Armijo/
    // criterion `loss()`) and its assembled gradient then read the SAME frozen
    // `atom.smooth_penalty` field, so they price one objective; the lag means gb
    // does NOT differentiate through S̃'s coord-dependence. This fixture was
    // written under the pre-3a9b9b40c seed-Gram convention: it FD-differenced
    // `loss()` on the un-refreshed `term` (Gram still the seed I) while `gb` came
    // from an `assembled` clone whose Gram had been refreshed to the ∫κ²ds
    // bending Gram (κ ≠ 0 for the d1 circle) — a fixture-only mismatch of ~λ(S̃−I)B.
    // Refreshing the base term aligns the fixture with the production ordering.
    let mut term = term;
    sae_fd_refresh(&mut term);
    for atom in &mut term.atoms {
        atom.refresh_intrinsic_smooth_penalty();
    }
    let term = term;
    let base_loss = sae_fd_total_loss(&term, &target, &rho);
    assert!(base_loss.is_finite(), "{label}: base loss is not finite");

    let mut assembled = term.clone();
    sae_fd_refresh(&mut assembled);
    let sys = assembled
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    assert_eq!(sys.rows.len(), term.n_obs());
    assert_eq!(sys.gb.len(), term.beta_dim());
    for row in 0..term.n_obs() {
        assert_eq!(
            sys.rows[row].gt.len(),
            1,
            "{label}: K=1 softmax d=1 should expose exactly one row coordinate gradient"
        );
    }

    let mut coord = SaeFdWorst::new();
    let base_coords = term.assignment.coords[0].as_flat().clone();
    for row in 0..term.n_obs() {
        let mut plus = term.clone();
        sae_fd_set_coord(&mut plus, row, base_coords[row] + epsilon);
        let loss_plus = sae_fd_total_loss(&plus, &target, &rho);

        let mut minus = term.clone();
        sae_fd_set_coord(&mut minus, row, base_coords[row] - epsilon);
        let loss_minus = sae_fd_total_loss(&minus, &target, &rho);

        let finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon);
        coord.observe(row, sys.rows[row].gt[0], finite_difference);
    }

    let mut decoder = SaeFdWorst::new();
    let beta = term.flatten_beta();
    for beta_idx in 0..beta.len() {
        let mut beta_plus = beta.clone();
        beta_plus[beta_idx] += epsilon;
        let mut plus = term.clone();
        plus.set_flat_beta(beta_plus.view()).unwrap();
        sae_fd_refresh(&mut plus);
        let loss_plus = sae_fd_total_loss(&plus, &target, &rho);

        let mut beta_minus = beta.clone();
        beta_minus[beta_idx] -= epsilon;
        let mut minus = term.clone();
        minus.set_flat_beta(beta_minus.view()).unwrap();
        sae_fd_refresh(&mut minus);
        let loss_minus = sae_fd_total_loss(&minus, &target, &rho);

        let finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon);
        decoder.observe(beta_idx, sys.gb[beta_idx], finite_difference);
    }

    SaeFdBlockReport {
        label: label.to_string(),
        base_loss,
        coord,
        decoder,
    }
}

/// Which manifold/basis a penalty-FD case runs on.
#[derive(Clone, Copy)]
pub(crate) enum SaePenCaseKind {
    EuclideanD1,
    PeriodicD1,
    EuclideanD2,
}

/// Which analytic penalty a penalty-FD case exercises.
#[derive(Clone, Copy)]
pub(crate) enum SaePenKind {
    Isometry,
    Ard,
    ScadMcp,
    NuclearNorm,
    DecoderIncoherence,
}

/// Single-atom SAE term on the requested manifold for the penalty-FD checks.
/// Mirrors `sae_fd_term` but exposes the analytic second jet the Isometry
/// penalty needs and allows a chosen latent dimension.
pub(crate) fn sae_pen_term(
    kind: SaePenCaseKind,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, PsiSlice) {
    let n_obs = 12usize;
    let p_out = 3usize;
    let (coords, latent_dim, atom): (Array2<f64>, usize, SaeManifoldAtom) = match kind {
        SaePenCaseKind::PeriodicD1 => {
            let mut coords = Array2::<f64>::zeros((n_obs, 1));
            for row in 0..n_obs {
                let x = row as f64;
                coords[[row, 0]] = 0.11 + 0.037 * x + 0.004 * (1.3 * x).sin();
            }
            let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let n_basis = phi.ncols();
            let atom = SaeManifoldAtom::new(
                "periodic_d1",
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                sae_fd_decoder(n_basis, p_out),
                Array2::<f64>::eye(n_basis),
            )
            .unwrap()
            .with_basis_second_jet(evaluator);
            (coords, 1, atom)
        }
        SaePenCaseKind::EuclideanD1 => {
            let mut coords = Array2::<f64>::zeros((n_obs, 1));
            for row in 0..n_obs {
                let x = row as f64;
                coords[[row, 0]] = -0.41 + 0.052 * x + 0.006 * (1.7 * x).cos();
            }
            let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let n_basis = phi.ncols();
            let atom = SaeManifoldAtom::new(
                "euclidean_d1",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi,
                jet,
                sae_fd_decoder(n_basis, p_out),
                Array2::<f64>::eye(n_basis),
            )
            .unwrap()
            .with_basis_second_jet(evaluator);
            (coords, 1, atom)
        }
        SaePenCaseKind::EuclideanD2 => {
            let mut coords = Array2::<f64>::zeros((n_obs, 2));
            for row in 0..n_obs {
                let x = row as f64;
                coords[[row, 0]] = -0.33 + 0.041 * x + 0.005 * (1.1 * x).cos();
                coords[[row, 1]] = 0.27 - 0.036 * x + 0.004 * (0.9 * x).sin();
            }
            let evaluator = Arc::new(EuclideanPatchEvaluator::new(2, 2).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let n_basis = phi.ncols();
            let atom = SaeManifoldAtom::new(
                "euclidean_d2",
                SaeAtomBasisKind::EuclideanPatch,
                2,
                phi,
                jet,
                sae_fd_decoder(n_basis, p_out),
                Array2::<f64>::eye(n_basis),
            )
            .unwrap()
            .with_basis_second_jet(evaluator);
            (coords, 2, atom)
        }
    };
    let manifold = atom.basis_kind.latent_manifold(latent_dim);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n_obs, 1)),
        vec![coords],
        vec![manifold],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = sae_fd_target(n_obs, p_out);
    // Suppress the built-in ARD / smoothness contributions so the registered
    // analytic penalty is the only penalty beyond data-fit + assignment prior.
    let log_ard = vec![Array1::from_elem(latent_dim, -30.0_f64)];
    let rho = SaeManifoldRho::new(0.0, 1.0e-4_f64.ln(), log_ard);
    let slice = PsiSlice {
        range: 0..n_obs * latent_dim,
        latent_dim: Some(latent_dim),
    };
    (term, target, rho, slice)
}

/// Two-atom K=2 SAE term for the DecoderIncoherence FD check. Both atoms are
/// d=1 euclidean patches so the β block is `[B_1 (M×p), B_2 (M×p)]`.
pub(crate) fn sae_pen_term_k2() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n_obs = 12usize;
    let p_out = 3usize;
    let mut atoms = Vec::with_capacity(2);
    let mut coord_blocks = Vec::with_capacity(2);
    for atom_idx in 0..2usize {
        let mut coords = Array2::<f64>::zeros((n_obs, 1));
        for row in 0..n_obs {
            let x = row as f64;
            coords[[row, 0]] = if atom_idx == 0 {
                -0.41 + 0.052 * x + 0.006 * (1.7 * x).cos()
            } else {
                0.18 + 0.039 * x + 0.005 * (1.1 * x).sin()
            };
        }
        let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let n_basis = phi.ncols();
        let mut decoder = sae_fd_decoder(n_basis, p_out);
        if atom_idx == 1 {
            for basis in 0..n_basis {
                for out_col in 0..p_out {
                    decoder[[basis, out_col]] += 0.07 * ((basis + out_col) as f64 + 1.0).cos();
                }
            }
        }
        let atom = SaeManifoldAtom::new(
            "euclidean_d1",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(n_basis),
        )
        .unwrap()
        .with_basis_second_jet(evaluator);
        atoms.push(atom);
        coord_blocks.push(coords);
    }
    let manifold = LatentManifold::Euclidean;
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n_obs, 2), 0.2),
        coord_blocks,
        vec![manifold.clone(), manifold],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let target = sae_fd_target(n_obs, p_out);
    let log_ard = vec![
        Array1::from_elem(1, -30.0_f64),
        Array1::from_elem(1, -30.0_f64),
    ];
    let rho = SaeManifoldRho::new(0.0, 1.0e-4_f64.ln(), log_ard);
    (term, target, rho)
}

/// Registry holding exactly one analytic penalty of the requested kind,
/// sized for `term`'s coord / β block.
pub(crate) fn sae_pen_registry(
    pen: SaePenKind,
    coord_slice: &PsiSlice,
    n_obs: usize,
    latent_dim: usize,
    beta_len: usize,
    p_out: usize,
) -> AnalyticPenaltyRegistry {
    use gam_terms::analytic_penalties::PenaltyConcavity;
    use gam_terms::analytic_penalties::ScadMcpPenalty;
    let mut registry = AnalyticPenaltyRegistry::new();
    match pen {
        SaePenKind::Isometry => {
            let penalty = IsometryPenalty::new_euclidean(coord_slice.clone(), latent_dim);
            registry.push(AnalyticPenaltyKind::Isometry(Arc::new(penalty)));
        }
        SaePenKind::Ard => {
            let penalty = ARDPenalty::new(coord_slice.clone(), latent_dim);
            registry.push(AnalyticPenaltyKind::Ard(Arc::new(penalty)));
        }
        SaePenKind::ScadMcp => {
            let penalty = ScadMcpPenalty::new(
                coord_slice.clone(),
                0.5,
                n_obs,
                3.0,
                1.0e-4,
                PenaltyConcavity::Mcp,
                false,
            )
            .unwrap();
            registry.push(AnalyticPenaltyKind::ScadMcp(Arc::new(penalty)));
        }
        SaePenKind::NuclearNorm => {
            let slice = PsiSlice {
                range: 0..beta_len,
                latent_dim: Some(beta_len / p_out),
            };
            let penalty = NuclearNormPenalty::new(slice, 0.7, p_out, 1.0e-4, None, false).unwrap();
            registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(penalty)));
        }
        SaePenKind::DecoderIncoherence => {
            let m_per = beta_len / (2 * p_out);
            let slice = PsiSlice {
                range: 0..beta_len,
                latent_dim: Some(beta_len / p_out),
            };
            let penalty = DecoderIncoherencePenalty::new(
                slice,
                vec![m_per, m_per],
                p_out,
                Array2::<f64>::from_elem((2, 2), 0.5),
                0.6,
                false,
            )
            .unwrap();
            registry.push(AnalyticPenaltyKind::DecoderIncoherence(Arc::new(penalty)));
        }
    }
    registry
}

/// FD-check the assembled gradient (`gt` / `gb`) against central differences
/// of `penalized_objective_total` with the registry's single analytic penalty
/// ACTIVE. Softmax mode always assembles the dense uniform row layout, so atom
/// `atom_idx`'s axis `a` for row `r` lives at `sys.rows[r].gt[off + a]` with
/// `off = coord_offsets()[atom_idx]` (a per-atom column offset, not a row
/// offset); the row index is the plain observation row.
pub(crate) fn sae_pen_fd_check(
    label: &str,
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    registry: &AnalyticPenaltyRegistry,
) -> SaeFdBlockReport {
    let epsilon = 1.0e-6;

    // #1610/#1625 lagged-diffusivity discipline: `assemble_arrow_schur` FREEZES
    // the decoder-repulsion and separation-barrier collinearity gates at assembly
    // entry and holds them fixed across the inner Newton line search, so the
    // assembled `gb` is the gradient of the penalized objective WITH those gates
    // frozen at the base decoders. In production the line search evaluates
    // `penalized_objective_total` on that same already-assembled term: it restores
    // only `SaeManifoldMutableState` (decoders, basis, logits, coords) per trial
    // and deliberately leaves the two frozen gates UNTOUCHED, so every line-search
    // trial reads the same frozen gate at its perturbed decoders.
    //
    // The FD probe must reproduce that exactly. It perturbs via `base.clone()`,
    // but `SaeManifoldTerm`'s custom `Clone` RESETS both frozen gates to `None`
    // (they are transient per-assembly state). With the gate gone,
    // `decoder_repulsion_value` — which, unlike the separation barrier, has NO
    // live fallback — returns 0 at every perturbed point, so the decoder-block FD
    // silently omits BOTH the frozen-gate repulsion AND barrier gradients that
    // `gb` legitimately carries (the K=2 near-collinear fixture engages them at
    // the ~5e-6 level). Freeze the gates ONCE on `base`, then RE-INSTALL that
    // frozen gate on every perturbation clone before evaluating the objective,
    // mirroring the optimizer's snapshot/restore (which never re-derives the
    // gate from the trial decoders).
    let mut base = term.clone();
    base.refresh_decoder_repulsion_gate();
    base.refresh_barrier_coactivation_gate();
    let base = base;
    let reinstall_frozen_gates = |t: &mut SaeManifoldTerm| {
        t.decoder_repulsion_gate = base.decoder_repulsion_gate.clone();
        t.barrier_coactivation_gate = base.barrier_coactivation_gate.clone();
    };

    let base_obj = base
        .penalized_objective_total(target.view(), rho, Some(registry), 1.0)
        .unwrap();
    assert!(base_obj.is_finite(), "{label}: base objective not finite");

    let mut assembled = base.clone();
    let sys = assembled
        .assemble_arrow_schur(target.view(), rho, Some(registry))
        .unwrap();

    // Coordinate-tier FD objective: data-fit + smoothness + ONLY the Psi-tier
    // (coordinate) analytic penalties. The β-tier decoder penalties
    // (DecoderIncoherence / MechanismSparsity / NuclearNorm) and the frozen-gate
    // decoder repulsion / separation barrier are coordinate-INDEPENDENT, so the
    // assembled coord gradient `rows[].gt` carries exactly zero of them. The
    // `decoder_incoherence_k2` fixture drives its incoherence VALUE to ~7e4;
    // folding that coord-constant into `penalized_objective_total` BEFORE the
    // central difference is catastrophic cancellation (the ~1e2 loss loses its
    // low bits when summed into the ~7e4 penalty, injecting ~3e-6 of FD roundoff
    // that the exact `gt` — correctly carrying no coord-gradient for a β-only
    // penalty — cannot match). Isolating the coord-tier value (all-penalty value
    // minus the β-tier decoder value) keeps the huge coord-constant OUT of the
    // differenced objective, so the coord FD sees only the terms `gt` truly
    // differentiates. Single-atom blocks (isometry/ard/scadmcp/nuclearnorm) are
    // unchanged: their β-tier value is coord-independent and cancels in the
    // central difference either way. (The decoder block below keeps the full
    // `penalized_objective_total` — the β penalties ARE what its gradient matches.)
    let coord_objective = |t: &SaeManifoldTerm| -> f64 {
        let loss = t.loss(target.view(), rho).expect("coord fd loss").total();
        let all_penalty = t
            .analytic_penalty_value_total(registry, 1.0)
            .expect("coord fd all-penalty value");
        let beta_penalty = t
            .analytic_decoder_penalty_value_total(registry)
            .expect("coord fd beta-penalty value");
        loss + (all_penalty - beta_penalty)
    };

    let mut coord = SaeFdWorst::new();
    let coord_offsets = base.assignment.coord_offsets();
    for atom_idx in 0..base.k_atoms() {
        let off = coord_offsets[atom_idx];
        let d = base.assignment.coords[atom_idx].latent_dim();
        let base_flat = base.assignment.coords[atom_idx].as_flat().clone();
        let n_atom = base_flat.len() / d;
        for row in 0..n_atom {
            for axis in 0..d {
                let lin = row * d + axis;
                let mut plus = base.clone();
                reinstall_frozen_gates(&mut plus);
                let mut flat_p = base_flat.clone();
                flat_p[lin] += epsilon;
                plus.assignment.coords[atom_idx].set_flat(flat_p.view());
                let coords_p = plus.assignment.coords[atom_idx].as_matrix();
                plus.atoms[atom_idx].refresh_basis(coords_p.view()).unwrap();
                let obj_p = coord_objective(&plus);

                let mut minus = base.clone();
                reinstall_frozen_gates(&mut minus);
                let mut flat_m = base_flat.clone();
                flat_m[lin] -= epsilon;
                minus.assignment.coords[atom_idx].set_flat(flat_m.view());
                let coords_m = minus.assignment.coords[atom_idx].as_matrix();
                minus.atoms[atom_idx]
                    .refresh_basis(coords_m.view())
                    .unwrap();
                let obj_m = coord_objective(&minus);

                let finite_difference = (obj_p - obj_m) / (2.0 * epsilon);
                coord.observe(
                    row * d + axis,
                    sys.rows[row].gt[off + axis],
                    finite_difference,
                );
            }
        }
    }

    let mut decoder = SaeFdWorst::new();
    let beta = base.flatten_beta();
    for beta_idx in 0..beta.len() {
        let mut beta_plus = beta.clone();
        beta_plus[beta_idx] += epsilon;
        let mut plus = base.clone();
        reinstall_frozen_gates(&mut plus);
        plus.set_flat_beta(beta_plus.view()).unwrap();
        let obj_p = plus
            .penalized_objective_total(target.view(), rho, Some(registry), 1.0)
            .unwrap();

        let mut beta_minus = beta.clone();
        beta_minus[beta_idx] -= epsilon;
        let mut minus = base.clone();
        reinstall_frozen_gates(&mut minus);
        minus.set_flat_beta(beta_minus.view()).unwrap();
        let obj_m = minus
            .penalized_objective_total(target.view(), rho, Some(registry), 1.0)
            .unwrap();

        let finite_difference = (obj_p - obj_m) / (2.0 * epsilon);
        decoder.observe(beta_idx, sys.gb[beta_idx], finite_difference);
    }

    SaeFdBlockReport {
        label: label.to_string(),
        base_loss: base_obj,
        coord,
        decoder,
    }
}

/// EXACT agreement between the SAE assembled gradient and the penalized
/// objective it claims to be the gradient of, per analytic penalty kind.
/// Central FD of `penalized_objective_total` (penalty ACTIVE) must match the
/// assembled coord `gt` and decoder `gb`. This pins the isometry decoder
/// gradient (`∂P/∂B`) that the value path counts but the gradient path used
/// to drop, alongside ARD, ScadMcp, NuclearNorm, and DecoderIncoherence.
#[test]
pub(crate) fn sae_assembled_gradient_matches_penalized_objective_central_fd() {
    let p_out = 3usize;
    let mut reports: Vec<SaeFdBlockReport> = Vec::new();

    let single_cases: &[(&str, SaePenCaseKind, SaePenKind)] = &[
        (
            "isometry_circle_d1",
            SaePenCaseKind::PeriodicD1,
            SaePenKind::Isometry,
        ),
        (
            "isometry_euclid_d2",
            SaePenCaseKind::EuclideanD2,
            SaePenKind::Isometry,
        ),
        ("ard_circle_d1", SaePenCaseKind::PeriodicD1, SaePenKind::Ard),
        (
            "scadmcp_euclid_d1",
            SaePenCaseKind::EuclideanD1,
            SaePenKind::ScadMcp,
        ),
        (
            "nuclearnorm_euclid_d1",
            SaePenCaseKind::EuclideanD1,
            SaePenKind::NuclearNorm,
        ),
    ];
    for (label, case_kind, pen_kind) in single_cases {
        let (term, target, rho, slice) = sae_pen_term(*case_kind);
        let n_obs = term.n_obs();
        let latent_dim = term.assignment.coords[0].latent_dim();
        let beta_len = term.beta_dim();
        let registry = sae_pen_registry(*pen_kind, &slice, n_obs, latent_dim, beta_len, p_out);
        term.validate_analytic_penalty_registry(&registry)
            .expect("penalty registry must validate for the SAE term");
        reports.push(sae_pen_fd_check(label, &term, &target, &rho, &registry));
    }

    {
        let (term, target, rho) = sae_pen_term_k2();
        let beta_len = term.beta_dim();
        let slice = PsiSlice {
            range: 0..beta_len,
            latent_dim: Some(beta_len / p_out),
        };
        let registry = sae_pen_registry(
            SaePenKind::DecoderIncoherence,
            &slice,
            term.n_obs(),
            1,
            beta_len,
            p_out,
        );
        term.validate_analytic_penalty_registry(&registry)
            .expect("DecoderIncoherence registry must validate for the K=2 SAE term");
        reports.push(sae_pen_fd_check(
            "decoder_incoherence_k2",
            &term,
            &target,
            &rho,
            &registry,
        ));
    }

    let relative_tolerance = 1.0e-5;
    let absolute_tolerance = 1.0e-7;
    let mut all_blocks_match = true;
    for report in &reports {
        let coord_ok = report.coord.relative_error <= relative_tolerance
            || report.coord.absolute_error <= absolute_tolerance;
        let decoder_ok = report.decoder.relative_error <= relative_tolerance
            || report.decoder.absolute_error <= absolute_tolerance;
        let metadata_ok = !report.label.is_empty() && report.base_loss.is_finite();
        all_blocks_match = all_blocks_match && metadata_ok && coord_ok && decoder_ok;
    }
    assert!(
        all_blocks_match,
        "SAE assembled gradient does not match central FD of the penalized objective: {reports:#?}"
    );
}

#[test]
pub(crate) fn sae_reml_extra_penalty_energy_counts_live_isometry_once() {
    let p_out = 3usize;
    let (term, _target, _rho, slice) = sae_pen_term(SaePenCaseKind::PeriodicD1);
    let registry = sae_pen_registry(
        SaePenKind::Isometry,
        &slice,
        term.n_obs(),
        term.assignment.coords[0].latent_dim(),
        term.beta_dim(),
        p_out,
    );

    let isometry_energy = term
        .isometry_penalty_value_total(&registry)
        .expect("live isometry value");
    assert!(
        isometry_energy > 0.0,
        "fixture must carry nonzero isometry energy"
    );

    let decoder_energy = term
        .analytic_decoder_penalty_value_total(&registry)
        .expect("decoder penalty value");
    assert_abs_diff_eq!(decoder_energy, 0.0, epsilon = 1.0e-12);

    // Full-objective contract: `extra` is exactly `penalized_objective_total −
    // loss.total()` — the complete registry value plus the decoder repulsion
    // conditioner and the Jeffreys separation barrier. On this single-atom
    // fixture repulsion and barrier are structurally zero (both are cross-atom
    // terms) and the registry carries only the isometry penalty, so the total
    // still equals the live isometry energy — asserted through the full
    // composition rather than the historical decoder+isometry-only pair.
    let repulsion_and_barrier =
        term.decoder_repulsion_value(1.0) + term.separation_barrier_value(1.0);
    let extra_energy = term
        .reml_extra_penalty_value_total(Some(&registry))
        .expect("REML extra penalty value");
    assert_abs_diff_eq!(
        extra_energy,
        isometry_energy + repulsion_and_barrier,
        epsilon = 1.0e-12
    );
}

#[test]
pub(crate) fn sae_d1_assembled_gradient_matches_loss_central_fd() {
    let reports = vec![
        sae_fd_check_case("euclidean_d1"),
        sae_fd_check_case("periodic_d1"),
    ];
    let relative_tolerance = 3.0e-5;
    let absolute_tolerance = 3.0e-7;
    let mut all_blocks_match = true;
    for report in &reports {
        let coord_ok = report.coord.relative_error <= relative_tolerance
            || report.coord.absolute_error <= absolute_tolerance;
        let decoder_ok = report.decoder.relative_error <= relative_tolerance
            || report.decoder.absolute_error <= absolute_tolerance;
        let metadata_ok = !report.label.is_empty() && report.base_loss.is_finite();
        all_blocks_match = all_blocks_match && metadata_ok && coord_ok && decoder_ok;
    }
    assert!(
        all_blocks_match,
        "SAE d=1 assembled gradient does not match central finite difference: {reports:#?}"
    );
}
