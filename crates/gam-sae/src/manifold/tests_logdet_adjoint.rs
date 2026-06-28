//! Finite-difference / dense-reference regressions for the SAE manifold
//! log-det machinery: the θ-adjoint of `log|H|`, the IBP-map and
//! learnable-α IBP log-det traces, and the row-jet program vs production
//! row-jets equivalence. Split out of `tests.rs` to keep that tracked file
//! under the #780 10k-line gate; these tests are self-contained (own tiny
//! fixtures, only `super::*` production items).

use super::*;

pub(crate) fn gamma_fd_tiny_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10usize;
    let p = 3usize;
    let k_atoms = 2usize;
    let m = 3usize;
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
        let assignments = softmax_row(logits.row(row), 0.9);
        for atom in 0..k_atoms {
            let theta = std::f64::consts::TAU * coords[atom][[row, 0]];
            let basis = [1.0, theta.sin(), theta.cos()];
            for out_col in 0..p {
                for basis_col in 0..m {
                    target[[row, out_col]] +=
                        assignments[atom] * basis[basis_col] * weights[atom][basis_col][out_col];
                }
            }
        }
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let (phi, jet) = evaluator.evaluate(coords[atom].view()).unwrap();
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            weights[atom][basis_col][out_col]
        });
        atoms.push(
            SaeManifoldAtom::new(
                format!("gamma_{atom}"),
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
        AssignmentMode::softmax(0.9),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(
        -6.0,
        -6.0,
        vec![Array1::from_vec(vec![-6.0]), Array1::from_vec(vec![-6.0])],
    );
    (term, target, rho)
}

pub(crate) fn fixed_state_logdet(
    mut term: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> f64 {
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
        .expect("fixed-state cache");
    let (tt, beta) = cache.arrow_log_det();
    tt + beta.expect("dense Schur logdet")
}

#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_on_tiny_fixture() {
    let (mut term, target, rho) = gamma_fd_tiny_fixture();
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    let probes = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (3usize, 1usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
    ];
    for (row, local_pos, var) in probes {
        let mut plus = term.clone();
        let mut minus = term.clone();
        match var {
            SaeLocalRowVar::Logit { atom } => {
                plus.assignment.logits[[row, atom]] += h;
                minus.assignment.logits[[row, atom]] -= h;
            }
            SaeLocalRowVar::Coord { atom, axis } => {
                let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                flat_p[idx] += h;
                flat_m[idx] -= h;
                plus.assignment.coords[atom].set_flat(flat_p.view());
                minus.assignment.coords[atom].set_flat(flat_m.view());
            }
        }
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 2.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ibp_map() {
    // The #1006 empirical-π third channel: under IBP-MAP, pi_k(M_k) couples
    // every row of column k, so perturbing one logit shifts EVERY row's
    // assembled `htt` diagonal in that column. `fixed_state_logdet` rebuilds
    // H at the perturbed state, so a single-logit FD captures both the
    // row-local direct-z channel and the global cross-row M_k channel that
    // `logdet_theta_adjoint` accumulates column-wise. lambda_sparse is the
    // active prior weight (fixed alpha), so the channel is genuinely live.
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    rho.log_lambda_sparse = -1.0;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    // Probe both atoms across distinct rows so the cross-row coupling
    // (different rows sharing a column) is exercised on both columns.
    let probes = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
        (7usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
    ];
    for (row, local_pos, var) in probes {
        let mut plus = term.clone();
        let mut minus = term.clone();
        match var {
            SaeLocalRowVar::Logit { atom } => {
                plus.assignment.logits[[row, atom]] += h;
                minus.assignment.logits[[row, atom]] -= h;
            }
            SaeLocalRowVar::Coord { atom, axis } => {
                let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                flat_p[idx] += h;
                flat_m[idx] -= h;
                plus.assignment.coords[atom].set_flat(flat_p.view());
                minus.assignment.coords[atom].set_flat(flat_m.view());
            }
        }
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "IBP Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// #1416 — the IBP fixed-alpha `ρ_sparse`-trace `½ tr(H⁻¹ ∂H_p/∂ρ_sparse)` must
/// include the FULL cross-row off-diagonal of the rank-one Woodbury source, not
/// just the diagonal. Under IBP-MAP the per-column empirical-mass `M_k` couples
/// every row of column `k` through `H_p = d·J Jᵀ + diag(s, c)`, and for fixed
/// alpha the entire IBP prior scales with `λ_sparse = eᵖ`, so
/// `∂H_p/∂ρ_sparse = H_p`. The analytic
/// `assignment_log_strength_hessian_trace` returns `½ ∂log|H|/∂ρ_sparse`; this
/// pins it against a fixed-state central difference of the joint `log|H|`. A
/// diagonal-only contraction (the pre-#1416 bug) would miss the
/// `½ d Σ_{i≠j}(H⁻¹)_{ij} J_i J_j` cross-row term and fail this FD.
#[test]
pub(crate) fn ibp_rho_sparse_logdet_trace_matches_dense_fd_1416() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // Fixed-alpha IBP-MAP with an active sparse prior so the cross-row Woodbury
    // source is genuinely live.
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    rho.log_lambda_sparse = -1.0;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let analytic = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("rho_sparse logdet trace");

    // Fixed-state central difference of log|H| w.r.t. ρ_sparse: vary λ_sparse,
    // hold (t, β) at the converged state (`fixed_state_logdet` re-assembles H
    // with inner_max_iter=0). The analytic trace is ½ ∂log|H|/∂ρ_sparse.
    let h = 1.0e-5;
    let mut rho_plus = rho.clone();
    let mut rho_minus = rho.clone();
    rho_plus.log_lambda_sparse += h;
    rho_minus.log_lambda_sparse -= h;
    let fd_half = 0.5
        * (fixed_state_logdet(term.clone(), &target, &rho_plus)
            - fixed_state_logdet(term.clone(), &target, &rho_minus))
        / (2.0 * h);
    let tol = 3.0e-3 * (1.0 + fd_half.abs().max(analytic.abs()));
    assert!(
        (fd_half - analytic).abs() <= tol,
        "IBP ρ_sparse logdet trace: fd(½∂log|H|/∂ρ)={fd_half:.8e}, \
         analytic={analytic:.8e}"
    );
}

/// #1417 — for LEARNABLE IBP alpha the joint Laplace `log|H|` depends on alpha
/// not only through the prior Hessian but EXPLICITLY through the data
/// Gauss-Newton blocks: `a_ik = σ(ℓ/τ)·π_k(α)`, so `H_ββ`, `H_tβ`, `H_tt` all
/// carry `α`. The complete `½ ∂log|H|/∂logα` is therefore the prior-Hessian
/// trace (`assignment_log_strength_hessian_trace`) PLUS the data trace
/// (`learnable_ibp_data_logdet_alpha_trace`, #1417). The learnable-alpha control
/// is `α(ρ₀) = α_base·e^{ρ₀}` (`resolve_learnable_weight`), so `∂logα/∂ρ₀ = 1`
/// and a fixed-state central difference of `log|H|` w.r.t. ρ₀ must equal twice
/// the SUM of both analytic traces. Omitting the data trace (the pre-#1417 bug)
/// would fail this FD.
#[test]
pub(crate) fn learnable_ibp_alpha_logdet_trace_matches_dense_fd_1417() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // Learnable-alpha IBP-MAP: ρ₀ (log_lambda_sparse) now drives alpha.
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    rho.log_lambda_sparse = 0.1;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    // The full ½ ∂log|H|/∂logα = prior trace + data trace, exactly as
    // `analytic_outer_rho_gradient_components` folds into `logdet_trace[0]`.
    let prior_trace = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("prior-Hessian alpha trace");
    let data_trace = term
        .learnable_ibp_data_logdet_alpha_trace(&rho, &cache, &solver)
        .expect("data-Hessian alpha trace");
    let analytic = prior_trace + data_trace;

    // Fixed-state central difference of log|H| w.r.t. ρ₀ (= log α offset).
    let h = 1.0e-5;
    let mut rho_plus = rho.clone();
    let mut rho_minus = rho.clone();
    rho_plus.log_lambda_sparse += h;
    rho_minus.log_lambda_sparse -= h;
    let fd_half = 0.5
        * (fixed_state_logdet(term.clone(), &target, &rho_plus)
            - fixed_state_logdet(term.clone(), &target, &rho_minus))
        / (2.0 * h);
    let tol = 3.0e-3 * (1.0 + fd_half.abs().max(analytic.abs()));
    assert!(
        (fd_half - analytic).abs() <= tol,
        "learnable-α logdet trace: fd(½∂log|H|/∂logα)={fd_half:.8e}, \
         analytic(prior+data)={analytic:.8e} (prior={prior_trace:.6e}, \
         data={data_trace:.6e})"
    );
    // The data trace must be a genuine, nonzero contribution (the #1417 term the
    // diagonal-only prior trace omitted) — otherwise the test would pass even if
    // `learnable_ibp_data_logdet_alpha_trace` returned 0.
    assert!(
        data_trace.abs() > 1.0e-9,
        "the #1417 data-Hessian alpha trace must be a live nonzero term; got \
         {data_trace:.3e}"
    );
}

/// #932 follow-up (the issue-comment cache-seam ask): the SAE row
/// jet-program oracle driven directly from a CONVERGED production
/// `ArrowFactorCache`, not a mirrored test layout.
///
/// For every row of the converged tiny fixture, the production
/// `row_jets_for_logdet` channels — the exact `first`/`second` tensors the
/// #1006 `logdet_theta_adjoint` contracts — are rebuilt as a
/// [`SaeReconstructionRowProgram`] from the SAME production inputs (the
/// term's basis value/jacobian tensors, `atom_second_jets`, decoder
/// blocks, gate logits/assignments, and the cache's own
/// `row_vars_for_cache_row` primary layout) and compared column by
/// column. The hand path sums sparse cross terms per (logit, coord)
/// variable pair; the tower derives them by Leibniz from one expression —
/// independent arithmetic, so agreement is a correctness proof of the
/// production packing on a real converged state. The `weighted` arm
/// exercises the #977 `set_row_loss_weights` √w seam, which scales every
/// production channel by `sqrt(w_row)`.
#[test]
pub(crate) fn sae_row_jet_program_matches_production_row_jets_on_converged_cache() {
    use crate::row_jet_program::{
        AtomRowBasisJet, RowGate, SaeReconstructionRowProgram,
    };

    // Tiny-fixture row arity: softmax gauges the last logit as the fixed
    // reference (assignment_coord_dim = k_atoms − 1 = 1 free logit), plus
    // 2 atoms × 1 latent coord.
    const K: usize = 3;
    for weighted in [false, true] {
        let (mut term, target, rho) = gamma_fd_tiny_fixture();
        if weighted {
            let weights: Vec<f64> = (0..term.n_obs())
                .map(|row| 0.5 + 0.17 * row as f64)
                .collect();
            term.set_row_loss_weights(weights)
                .expect("set row loss weights");
        }
        let (_value, _loss, cache) = term
            .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
            .expect("converged cache");
        let second_jets = term.atom_second_jets().expect("second jets");
        let border = term
            .border_channels_for_cache(&cache)
            .expect("border channels");
        let AssignmentMode::Softmax { temperature, .. } = term.assignment.mode else {
            panic!("gamma fixture is softmax-gated");
        };
        let inv_tau = 1.0 / temperature;
        let p = term.output_dim();
        let k_atoms = term.k_atoms();

        for row in 0..term.n_obs() {
            let vars = term.row_vars_for_cache_row(row, &cache).expect("row vars");
            assert_eq!(
                vars.len(),
                K,
                "tiny fixture rows carry 1 free softmax logit + 2 coords"
            );
            let assignments = term
                .assignment
                .try_assignments_row(row)
                .expect("assignments row");
            let jets = term
                .row_jets_for_logdet(
                    &rho,
                    row,
                    vars.clone(),
                    assignments.view(),
                    &second_jets,
                    &border,
                )
                .expect("production row jets");

            // Primary layout exactly as the cache rows it: slot positions
            // come from the production `row_vars_for_cache_row`, not a
            // re-derived convention.
            let mut logit_slot = vec![None; k_atoms];
            let mut coord_slot: Vec<Vec<usize>> = term
                .atoms
                .iter()
                .map(|atom| vec![usize::MAX; atom.latent_dim])
                .collect();
            for (pos, var) in vars.iter().enumerate() {
                match *var {
                    SaeLocalRowVar::Logit { atom } => logit_slot[atom] = Some(pos),
                    SaeLocalRowVar::Coord { atom, axis } => coord_slot[atom][axis] = pos,
                }
            }

            // Per-atom basis jets straight from the production tensors the
            // hand path consumes: basis_values / basis_jacobian /
            // atom_second_jets / decoder_coefficients.
            let atoms: Vec<AtomRowBasisJet> = term
                .atoms
                .iter()
                .enumerate()
                .map(|(k, atom)| {
                    let m = atom.basis_size();
                    let d = atom.latent_dim;
                    AtomRowBasisJet {
                        phi: (0..m).map(|b| atom.basis_values[[row, b]]).collect(),
                        d_phi: (0..m)
                            .map(|b| {
                                (0..d)
                                    .map(|axis| atom.basis_jacobian[[row, b, axis]])
                                    .collect()
                            })
                            .collect(),
                        d2_phi: (0..m)
                            .map(|b| {
                                (0..d)
                                    .map(|aa| {
                                        (0..d).map(|bb| second_jets[k][[row, b, aa, bb]]).collect()
                                    })
                                    .collect()
                            })
                            .collect(),
                        decoder: (0..m)
                            .map(|b| (0..p).map(|c| atom.decoder_coefficients[[b, c]]).collect())
                            .collect(),
                        latent_dim: d,
                    }
                })
                .collect();

            let prog = SaeReconstructionRowProgram {
                atoms,
                gate_value: assignments.to_vec(),
                logits: term.assignment.logits.row(row).to_vec(),
                gate_scale: vec![1.0; k_atoms],
                gate_shift: vec![0.0; k_atoms],
                gate: RowGate::Softmax { inv_tau },
                logit_slot,
                coord_slot,
                n_primaries: K,
            };
            // The production channels carry the √w row-loss weight (#977
            // single seam); the program is the unweighted reconstruction.
            let sqrt_row_w = term
                .row_loss_weights
                .as_deref()
                .map_or(1.0, |w| w[row].sqrt());
            if weighted {
                assert!(
                    (sqrt_row_w - 1.0).abs() > 1e-6,
                    "weighted arm must exercise a non-unit √w (row {row}, √w={sqrt_row_w})"
                );
            }

            for out_col in 0..p {
                let tower = prog.reconstruction_column::<K>(out_col);
                let g_floor = (0..K)
                    .map(|a| jets.first[a][out_col].abs())
                    .fold(1e-12_f64, f64::max);
                let h_floor = (0..K)
                    .flat_map(|a| (0..K).map(move |b| (a, b)))
                    .map(|(a, b)| jets.second[a][b][out_col].abs())
                    .fold(1e-12_f64, f64::max);
                for a in 0..K {
                    let want = sqrt_row_w * tower.g[a];
                    assert!(
                        (jets.first[a][out_col] - want).abs() <= 1e-9 * g_floor,
                        "weighted={weighted} row {row} col {out_col} first[{a}]: \
                             production {} vs tower {}",
                        jets.first[a][out_col],
                        want
                    );
                    for b in 0..K {
                        let want2 = sqrt_row_w * tower.h[a][b];
                        assert!(
                            (jets.second[a][b][out_col] - want2).abs() <= 1e-9 * h_floor,
                            "weighted={weighted} row {row} col {out_col} \
                                 second[{a}][{b}]: production {} vs tower {}",
                            jets.second[a][b][out_col],
                            want2
                        );
                    }
                }
            }

            // β BORDER CHANNELS (#932): the hand path packs `beta`
            // (value ∂ẑ_c/∂β = ζ_k·Φ_b·output_c) and `beta_deriv` /
            // `beta_l_deriv` (the mixed ∂²ẑ_c/∂β∂p_a = ∂(ζ_k·Φ_b)/∂p_a·output_c)
            // term by term in `row_jets_for_logdet`, with NO tower oracle
            // previously. The arrow β coefficient multiplies the channel's
            // (frame / identity) `output` vector — NOT the current decoder
            // matrix — so the local-variable dependence is exactly
            // s = ζ_k(ℓ)·Φ_b(t_k) = `beta_border_tower` (built from the SAME
            // gate_tower / basis_tower primitives as the reconstruction column);
            // production multiplies that scalar by `channel.output[c]·√w`. Pin
            // every β channel (value + both mixed-derivative arrays) to it at
            // ~1e-9.
            for (beta_pos, channel) in border.iter().enumerate() {
                // The β border channel's LOCAL-variable dependence is
                // s = ζ_k(ℓ)·Φ_b(t_k); the production packing multiplies that
                // scalar by the channel's (frame / identity) `output[c]` — NOT
                // the decoder matrix — and by √w.
                let s = prog.beta_border_tower::<K>(channel.atom, channel.basis_col);
                for out_col in 0..p {
                    let out_c = channel.output[out_col];
                    let want_v = sqrt_row_w * s.v * out_c;
                    let v_floor = want_v.abs().max(1e-12);
                    assert!(
                        (jets.beta[beta_pos][out_col] - want_v).abs() <= 1e-9 * v_floor,
                        "weighted={weighted} row {row} col {out_col} \
                         beta[{beta_pos}] (atom {} basis {}): production {} vs tower {}",
                        channel.atom,
                        channel.basis_col,
                        jets.beta[beta_pos][out_col],
                        want_v
                    );
                    for a in 0..K {
                        let want_d = sqrt_row_w * s.g[a] * out_c;
                        let d_floor = want_d.abs().max(1e-12);
                        // `beta_deriv` and `beta_l_deriv` are the SAME mixed
                        // ∂²ẑ_c/∂β∂p_a derivative the linear-in-β reconstruction
                        // produces (the hand path fills both identically); both
                        // must equal the tower's first-derivative channel × out_c.
                        assert!(
                            (jets.beta_deriv[a][beta_pos][out_col] - want_d).abs()
                                <= 1e-9 * d_floor,
                            "weighted={weighted} row {row} col {out_col} \
                             beta_deriv[{a}][{beta_pos}]: production {} vs tower {}",
                            jets.beta_deriv[a][beta_pos][out_col],
                            want_d
                        );
                        assert!(
                            (jets.beta_l_deriv[a][beta_pos][out_col] - want_d).abs()
                                <= 1e-9 * d_floor,
                            "weighted={weighted} row {row} col {out_col} \
                             beta_l_deriv[{a}][{beta_pos}]: production {} vs tower {}",
                            jets.beta_l_deriv[a][beta_pos][out_col],
                            want_d
                        );
                    }
                }
            }
        }
    }
}

#[test]
pub(crate) fn ibp_map_outer_objective_advertises_analytic_gradient() {
    // The IBP-MAP empirical-π third channel (including the cross-row M_k
    // coupling) is now assembled exactly in `logdet_theta_adjoint` (#1006),
    // so the outer objective advertises an analytic gradient like every
    // other assignment mode.
    let (mut term, target, rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.9, 1.0, false);

    let obj = SaeManifoldOuterObjective::new(term, target, None, rho, 5, 0.4, 1.0e-6, 1.0e-6);
    assert_eq!(obj.capability().gradient, Derivative::Analytic);
}
