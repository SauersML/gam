//! `sae_row_jet_program_matches_production_row_jets_on_converged_cache` and
//! `ibp_map_outer_objective_advertises_analytic_gradient`, split verbatim out
//! of `tests.rs` to keep that tracked file under the #780 10k-line gate.
//! Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; the shared
//! `gamma_fd_tiny_fixture` is sourced from the sibling `tests` module.

use super::tests::gamma_fd_tiny_fixture;
use super::*;

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
    use crate::row_jet_program::{AtomRowBasisJet, RowGate, SaeReconstructionRowProgram};

    // Tiny-fixture row arity: softmax gauges the last logit as the fixed
    // reference (assignment_coord_dim = k_atoms − 1 = 1 free logit), plus
    // 2 atoms × 1 latent coord.
    const K: usize = 3;
    for weighted in [false, true] {
        let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
        // #1625 — lift `log_lambda_sparse` off the fixture's `-6.0` floor into the
        // PD basin. At λ_sparse = e⁻⁶ ≈ 2.5e-3 the softmax assignment-prior
        // curvature is far too weak to regularize the rank-deficient 2-atom
        // periodic bilinear fit on these n=10 rows: the undamped (ridge=0) joint
        // Hessian has NO interior PD minimum, the inner KKT gradient floors ~600×
        // above tolerance while the undamped Newton step stays O(6), and the
        // `.expect("converged cache")` below panics on a genuinely unattainable
        // optimum. A fixed-state ρ-sweep of this exact softmax fixture shows the
        // inner solve converges for every `log_lambda_sparse ≥ -2`; `-1.0` sits
        // comfortably inside that PD region (the value the sibling #1416
        // IBP-ρ_sparse oracle already pins). This is a setup fix that makes a
        // genuine converged cache EXIST so the hand-vs-jet row-jet oracle below
        // can reach its real bit-identity assertions; it weakens no tolerance.
        rho.log_lambda_sparse = -1.0;
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
                .row_jets_for_logdet(row, vars.clone(), assignments.view(), &second_jets, &border)
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
                fixed_gate_value: Vec::new(),
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

/// #932 revert oracle: the 4-row SIMD jet batch (`row_jets_for_logdet_batch4`,
/// demoted off the production hot path) lane `i` must reproduce the production
/// scalar `row_jets_for_logdet` — which is now the HAND closed form for the
/// softmax gate — on row `i`, to ≤1e-9 across every channel
/// (`first`/`second`/`beta`/`beta_deriv`/`beta_l_deriv`). This keeps the jet
/// (the universal correctness oracle, retained not deleted) cross-checked
/// against the reinstated hand arithmetic on a real converged cache, so the
/// hand path stays guarded against the #736 forgotten-channel bug class.
#[test]
pub(crate) fn batch4_jet_lanes_match_scalar_hand_row_jets() {
    use ndarray::Array1;
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // #1625 — see `sae_row_jet_program_matches_production_row_jets_on_converged_cache`:
    // the fixture's `-6.0` sparse floor leaves the undamped joint Hessian without a
    // PD optimum, so `.expect("converged cache")` is unattainable. `-1.0` sits in
    // the PD basin (fixed-state ρ-sweep: converges for every `log_lambda_sparse ≥ -2`)
    // so the SIMD-lane vs scalar-hand oracle below reaches its real assertions.
    // Setup fix, no tolerance weakened.
    rho.log_lambda_sparse = -1.0;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let second_jets = term.atom_second_jets().expect("second jets");
    let border = term
        .border_channels_for_cache(&cache)
        .expect("border channels");
    assert!(
        term.n_obs() >= 4,
        "fixture must have ≥4 rows to exercise the 4-row batch"
    );

    let rows = [0usize, 1, 2, 3];
    let batch = term
        .row_jets_for_logdet_batch4(rows, &cache, &second_jets, &border)
        .expect("batch4 build")
        .expect("softmax-aligned fixture rows must batch");

    let maxabs = |xs: &[f64]| xs.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));

    for (lane, &row) in rows.iter().enumerate() {
        let vars = term.row_vars_for_cache_row(row, &cache).expect("row vars");
        let mut a = Array1::<f64>::zeros(term.k_atoms());
        term.assignment
            .try_assignments_row_into(
                row,
                a.as_slice_mut().expect("contiguous scratch"),
            )
            .expect("assignments row");
        let hand = term
            .row_jets_for_logdet(row, vars, a.view(), &second_jets, &border)
            .expect("hand scalar row jets");
        let jet = &batch[lane];

        for (a_idx, (hf, jf)) in hand.first.iter().zip(jet.first.iter()).enumerate() {
            let floor = maxabs(hf).max(1e-12);
            for (c, (h, j)) in hf.iter().zip(jf.iter()).enumerate() {
                assert!(
                    (h - j).abs() <= 1e-9 * floor,
                    "row {row} lane {lane} first[{a_idx}][{c}]: hand {h} vs jet {j}"
                );
            }
        }
        for (a_idx, (hrow, jrow)) in hand.second.iter().zip(jet.second.iter()).enumerate() {
            for (b_idx, (hf, jf)) in hrow.iter().zip(jrow.iter()).enumerate() {
                let floor = maxabs(hf).max(1e-12);
                for (c, (h, j)) in hf.iter().zip(jf.iter()).enumerate() {
                    assert!(
                        (h - j).abs() <= 1e-9 * floor,
                        "row {row} lane {lane} second[{a_idx}][{b_idx}][{c}]: hand {h} vs jet {j}"
                    );
                }
            }
        }
        for (bp, (hf, jf)) in hand.beta.iter().zip(jet.beta.iter()).enumerate() {
            let floor = maxabs(hf).max(1e-12);
            for (c, (h, j)) in hf.iter().zip(jf.iter()).enumerate() {
                assert!(
                    (h - j).abs() <= 1e-9 * floor,
                    "row {row} lane {lane} beta[{bp}][{c}]: hand {h} vs jet {j}"
                );
            }
        }
        for (pair, (hand_arr, jet_arr)) in [
            (&hand.beta_deriv, &jet.beta_deriv),
            (&hand.beta_l_deriv, &jet.beta_l_deriv),
        ]
        .iter()
        .enumerate()
        {
            for (a_idx, (hrow, jrow)) in hand_arr.iter().zip(jet_arr.iter()).enumerate() {
                for (bp, (hf, jf)) in hrow.iter().zip(jrow.iter()).enumerate() {
                    let floor = maxabs(hf).max(1e-12);
                    for (c, (h, j)) in hf.iter().zip(jf.iter()).enumerate() {
                        assert!(
                            (h - j).abs() <= 1e-9 * floor,
                            "row {row} lane {lane} beta_deriv(pair {pair})[{a_idx}][{bp}][{c}]: \
                             hand {h} vs jet {j}"
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

/// A trivial n=1, K=2, IBP-MAP term whose atoms carry a single basis function
/// with a KNOWN value / jacobian / decoder so the reconstruction row program's
/// value and first-derivative channels are hand-computable. Atom `k` has
/// `decoded_k = phi_k·dec_k`, `d(decoded_k)/dt = dphi_k·dec_k`. Used to pin the
/// #1026/#1033 fixed-gate handling in `reconstruction_row_program_for_logdet`.
fn fixed_gate_probe_term() -> (SaeManifoldTerm, SaeManifoldRho) {
    use ndarray::{Array1, Array2, Array3};
    let (n, m, p) = (1usize, 1usize, 1usize);
    let mk_atom = |name: &str, phi: f64, dphi: f64, dec: f64| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            Array2::from_shape_vec((n, m), vec![phi]).unwrap(),
            Array3::from_shape_vec((n, m, 1), vec![dphi]).unwrap(),
            Array2::from_shape_vec((m, p), vec![dec]).unwrap(),
            Array2::from_shape_vec((m, m), vec![1.0]).unwrap(),
        )
        .unwrap()
    };
    let atoms = vec![mk_atom("a0", 1.0, 2.0, 1.5), mk_atom("a1", 1.0, 0.5, -0.8)];
    // Free logits are set to an EXTREME value so any leakage of the free-logit
    // gate into the (frozen / ungated) fixed path is loud.
    let logits = Array2::from_shape_vec((n, 2), vec![5.0, -5.0]).unwrap();
    let coords = vec![
        Array2::from_shape_vec((n, 1), vec![0.15]).unwrap(),
        Array2::from_shape_vec((n, 1), vec![0.35]).unwrap(),
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.8, 1.8, false),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(
        0.0,
        0.0,
        vec![Array1::from_vec(vec![0.0]), Array1::from_vec(vec![0.0])],
    );
    (term, rho)
}

/// #1033 FROZEN-routing regression: with the free logits at one extreme and the
/// FROZEN (amortized) logits at the opposite extreme, the row reconstruction
/// program must gate on the FROZEN routing — every gate is pinned to the active
/// routing value with an EXACTLY-ZERO logit derivative — and its `logits` field
/// must read the frozen (not the free) logits. The coordinate derivatives use
/// the frozen gate, never the stale free-logit gate.
#[test]
pub(crate) fn frozen_ibp_row_program_gates_on_frozen_not_free_logit() {
    use ndarray::{Array1, Array4};
    let mut term = fixed_gate_probe_term().0;
    // Frozen routing = OPPOSITE extreme of the free logits [5, -5].
    let frozen = ndarray::Array2::from_shape_vec((1, 2), vec![-5.0, 5.0]).unwrap();
    term.assignment
        .set_frozen_routing_in_place(frozen.clone())
        .expect("install frozen routing");
    assert!(term.assignment.routing_is_frozen());
    for k in 0..2 {
        assert!(term.assignment.logit_is_fixed(k), "frozen ⇒ atom {k} fixed");
    }

    // The active routing gate values the assembly used (arbitrary but distinct;
    // deliberately NOT what either logit would produce, to prove the program
    // adopts them verbatim as constant gates).
    let assignments = Array1::from_vec(vec![0.2_f64, 0.9_f64]);
    // Layout: logit slots 0,1; coord slots 2,3.
    let vars = vec![
        SaeLocalRowVar::Logit { atom: 0 },
        SaeLocalRowVar::Logit { atom: 1 },
        SaeLocalRowVar::Coord { atom: 0, axis: 0 },
        SaeLocalRowVar::Coord { atom: 1, axis: 0 },
    ];
    let second_jets = vec![Array4::<f64>::zeros((1, 1, 1, 1)); 2];
    let prog = term
        .reconstruction_row_program_for_logdet(0, &vars, assignments.view(), &second_jets)
        .expect("row program");

    // The program reads the FROZEN logits, not the free ones.
    assert_eq!(prog.logits, vec![-5.0, 5.0], "must read frozen logits");
    // Every atom is a FIXED gate equal to the active routing value.
    assert_eq!(
        prog.fixed_gate_value,
        vec![Some(0.2), Some(0.9)],
        "frozen ⇒ all gates pinned to the active routing value"
    );

    let col = prog.reconstruction_column::<4>(0);
    // Logit-slot derivatives are exactly zero (gates are frozen constants).
    assert_eq!(col.g[0], 0.0, "frozen atom-0 logit derivative must be 0");
    assert_eq!(col.g[1], 0.0, "frozen atom-1 logit derivative must be 0");
    // Coordinate derivatives use the FROZEN gate value: g[coord_k] = a_k·dphi_k·dec_k.
    assert!(
        (col.g[2] - 0.2 * (2.0 * 1.5)).abs() < 1e-12,
        "atom-0 coord derivative must use frozen gate 0.2: {}",
        col.g[2]
    );
    assert!(
        (col.g[3] - 0.9 * (0.5 * -0.8)).abs() < 1e-12,
        "atom-1 coord derivative must use frozen gate 0.9: {}",
        col.g[3]
    );
    // Value = Σ a_k·phi_k·dec_k with the frozen gates.
    let expected_v = 0.2 * 1.5 + 0.9 * -0.8;
    assert!(
        (col.v - expected_v).abs() < 1e-12,
        "frozen reconstruction value"
    );
}

/// #1026 UNGATED-atom regression: an ungated atom's gate is pinned at 1.0 with a
/// zero logit derivative (its coordinate derivative uses gate 1.0), while a
/// sibling GATED atom keeps its free-logit gate and a nonzero logit derivative.
#[test]
pub(crate) fn ungated_ibp_row_program_gates_at_unit_with_zero_logit_derivative() {
    use ndarray::{Array1, Array4};
    let mut term = fixed_gate_probe_term().0;
    // Atom 0 ungated (dense background tier), atom 1 gated. Not frozen. IBP-MAP
    // accepts ungated atoms (only Softmax rejects them; see `with_ungated`).
    term.assignment.ungated = vec![true, false];
    assert!(!term.assignment.routing_is_frozen());
    assert!(term.assignment.logit_is_fixed(0), "ungated atom-0 is fixed");
    assert!(!term.assignment.logit_is_fixed(1), "gated atom-1 is free");

    // Ungated atom's active gate is pinned at 1.0; atom 1's is an arbitrary free
    // value (its exact number does not matter to this test — only that it moves).
    let assignments = Array1::from_vec(vec![1.0_f64, 0.6_f64]);
    let vars = vec![
        SaeLocalRowVar::Logit { atom: 0 },
        SaeLocalRowVar::Logit { atom: 1 },
        SaeLocalRowVar::Coord { atom: 0, axis: 0 },
        SaeLocalRowVar::Coord { atom: 1, axis: 0 },
    ];
    let second_jets = vec![Array4::<f64>::zeros((1, 1, 1, 1)); 2];
    let prog = term
        .reconstruction_row_program_for_logdet(0, &vars, assignments.view(), &second_jets)
        .expect("row program");

    // Not frozen ⇒ the program reads the FREE logits; only the ungated atom is a
    // fixed (unit) gate.
    assert_eq!(prog.logits, vec![5.0, -5.0], "unfrozen ⇒ reads free logits");
    assert_eq!(
        prog.fixed_gate_value,
        vec![Some(1.0), None],
        "only the ungated atom carries a fixed (unit) gate"
    );

    let col = prog.reconstruction_column::<4>(0);
    // Ungated atom: zero logit derivative, coord derivative at gate 1.0.
    assert_eq!(
        col.g[0], 0.0,
        "ungated atom logit derivative must be exactly 0"
    );
    assert!(
        (col.g[2] - 1.0 * (2.0 * 1.5)).abs() < 1e-12,
        "ungated atom coord derivative must use gate 1.0: {}",
        col.g[2]
    );
    // Gated sibling atom: its logit still moves the reconstruction.
    assert!(
        col.g[1].abs() > 1e-9,
        "gated atom-1 logit derivative must be nonzero: {}",
        col.g[1]
    );
}
