//! Finite-difference verification of `SaeDecoderBlockJacobian::effective_jacobian_at`.
//!
//! # What is tested
//!
//! For a synthetic SAE setup (K atoms, M basis functions per atom, p output
//! dimensions, N observations) we verify that
//!
//! ```text
//! J_k = effective_jacobian_at(state)   (shape n·p × M·p)
//! ```
//!
//! matches the finite-difference approximation
//!
//! ```text
//! J_k[:, j]  ≈  ( η(β_k + h·e_j) − η(β_k − h·e_j) ) / (2h)
//! ```
//!
//! where η_i = Σ_k a_ik · Φ_k(t_ik) · B_k is the SAE linear predictor and
//! the perturbation `h·e_j` touches the decoder coefficients of atom k only.
//!
//! # n_outputs verdict
//!
//! The test asserts `n_outputs() == p` (decoder output dimension), confirming
//! that T12's routing would send SAE decoder blocks through
//! `audit_identifiability_channel_aware` rather than the flat path.
//!
//! # Routing fix documentation
//!
//! Before this implementation SAE had no `ParameterBlockSpec` builder and was
//! never passed to `audit_identifiability` or `audit_identifiability_channel_aware`.
//! The decoder_parameter_block_specs() method introduced here wires in
//! `SaeDecoderBlockJacobian` (n_outputs = p > 1) so that any future audit
//! invocation routes through the channel-aware path, which correctly handles
//! cross-atom interactions via orthogonal output channels rather than treating
//! them as flat aliases.

use gam::families::custom_family::FamilyLinearizationState;
use gam::terms::sae_manifold::{
    AssignmentMode, SaeAssignment, SaeManifoldAtom, SaeManifoldTerm,
};
use ndarray::Array2;

const N: usize = 16;
const K_ATOMS: usize = 2;
const M_BASIS: usize = 3;
const P_OUT: usize = 2;
const LATENT_DIM: usize = 1;

fn make_atom(name: &str, phi: Array2<f64>) -> SaeManifoldAtom {
    let m = phi.ncols();
    let p = P_OUT;
    // Random-ish decoder and trivial penalty.
    let mut b = Array2::<f64>::zeros((m, p));
    for mm in 0..m {
        for pp in 0..p {
            b[[mm, pp]] = (mm as f64 + 1.0) * 0.3 + (pp as f64) * 0.7;
        }
    }
    let penalty = Array2::<f64>::eye(m);
    // basis_jacobian is unused for fitted()/effective_jacobian_at(), so zeros suffice.
    let jet = ndarray::Array3::<f64>::zeros((N, m, LATENT_DIM));
    SaeManifoldAtom::new(
        name,
        gam::terms::sae_manifold::SaeAtomBasisKind::EuclideanPatch,
        LATENT_DIM,
        phi,
        jet,
        b,
        penalty,
    )
    .unwrap()
}

fn build_term() -> SaeManifoldTerm {
    // Basis: [1, t, t²] evaluated at t_i = i/N.
    let mut phi0 = Array2::<f64>::zeros((N, M_BASIS));
    let mut phi1 = Array2::<f64>::zeros((N, M_BASIS));
    let mut coords0 = Array2::<f64>::zeros((N, LATENT_DIM));
    let mut coords1 = Array2::<f64>::zeros((N, LATENT_DIM));
    for i in 0..N {
        let t0 = (i as f64) / (N as f64);
        let t1 = 0.5 + (i as f64) / (2.0 * N as f64);
        phi0[[i, 0]] = 1.0;
        phi0[[i, 1]] = t0;
        phi0[[i, 2]] = t0 * t0;
        phi1[[i, 0]] = 1.0;
        phi1[[i, 1]] = t1;
        phi1[[i, 2]] = t1 * t1;
        coords0[[i, 0]] = t0;
        coords1[[i, 0]] = t1;
    }
    let atom0 = make_atom("atom0", phi0);
    let atom1 = make_atom("atom1", phi1);

    let logits = Array2::<f64>::zeros((N, K_ATOMS));
    let assignment = SaeAssignment::from_blocks_with_mode(
        logits,
        vec![coords0, coords1],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
}

/// Compute η for atom `k` only (the other atoms' contributions cancel out
/// when we diff w.r.t. B_k, so we compute full fitted η here and rely on
/// linearity).
fn compute_eta_full(term: &SaeManifoldTerm) -> Array2<f64> {
    term.fitted()
}

/// Perturb B_k[m, out] by delta, recompute η, restore.
fn eta_with_perturbed_decoder(
    term: &mut SaeManifoldTerm,
    atom_idx: usize,
    m: usize,
    out: usize,
    delta: f64,
) -> Array2<f64> {
    term.atoms[atom_idx].decoder_coefficients[[m, out]] += delta;
    let result = compute_eta_full(term);
    term.atoms[atom_idx].decoder_coefficients[[m, out]] -= delta;
    result
}

#[test]
fn sae_decoder_jacobian_n_outputs_equals_p() {
    let term = build_term();
    let specs = term.decoder_parameter_block_specs();
    assert_eq!(
        specs.len(),
        K_ATOMS,
        "expected one spec per atom; got {}",
        specs.len()
    );
    for (idx, spec) in specs.iter().enumerate() {
        let n_out = spec
            .jacobian_callback
            .as_ref()
            .map(|cb| cb.n_outputs())
            .unwrap_or(1);
        assert_eq!(
            n_out, P_OUT,
            "atom {idx} n_outputs must equal P_OUT={P_OUT}; got {n_out}"
        );
    }
}

#[test]
fn sae_decoder_jacobian_shape_matches_n_times_p_by_m_times_p() {
    let term = build_term();
    let specs = term.decoder_parameter_block_specs();
    let dummy_state = FamilyLinearizationState {
        beta: &[],
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    for (idx, spec) in specs.iter().enumerate() {
        let jac = spec
            .effective_jacobian_at("test", &dummy_state)
            .unwrap_or_else(|e| panic!("effective_jacobian_at failed for atom {idx}: {e}"));
        assert_eq!(
            jac.nrows(),
            N * P_OUT,
            "atom {idx} Jacobian nrows must be N*P={N}*{P_OUT}={}; got {}",
            N * P_OUT,
            jac.nrows(),
        );
        assert_eq!(
            jac.ncols(),
            M_BASIS * P_OUT,
            "atom {idx} Jacobian ncols must be M*P={M_BASIS}*{P_OUT}={}; got {}",
            M_BASIS * P_OUT,
            jac.ncols(),
        );
    }
}

#[test]
fn sae_decoder_jacobian_matches_finite_difference() {
    let mut term = build_term();
    let dummy_state = FamilyLinearizationState {
        beta: &[],
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let h = 1.0e-5;
    let tol_abs = 1.0e-7;
    let tol_rel = 1.0e-5;

    for atom_idx in 0..K_ATOMS {
        let specs = term.decoder_parameter_block_specs();
        let jac = specs[atom_idx]
            .effective_jacobian_at("test", &dummy_state)
            .unwrap_or_else(|e| panic!("effective_jacobian_at failed for atom {atom_idx}: {e}"));

        // Finite-difference column by column: perturb β_k[m*P_OUT + out].
        for m in 0..M_BASIS {
            for out in 0..P_OUT {
                let beta_col = m * P_OUT + out;
                let eta_plus = eta_with_perturbed_decoder(&mut term, atom_idx, m, out, h);
                let eta_minus = eta_with_perturbed_decoder(&mut term, atom_idx, m, out, -h);

                // FD column: (η_plus - η_minus) / (2h), flattened row-major.
                for row in 0..N {
                    for out2 in 0..P_OUT {
                        let fd = (eta_plus[[row, out2]] - eta_minus[[row, out2]]) / (2.0 * h);
                        let analytic = jac[[row * P_OUT + out2, beta_col]];
                        let err = (analytic - fd).abs();
                        let scale = tol_rel * fd.abs().max(analytic.abs()).max(1.0e-10);
                        assert!(
                            err <= tol_abs + scale,
                            "atom {atom_idx} β_col={beta_col} (m={m},out={out}) row={row} out2={out2}: \
                             analytic={analytic:.8e} fd={fd:.8e} err={err:.3e}",
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sae_decoder_jacobian_block_diagonal_in_output() {
    // Off-diagonal output entries (i·p + out vs m·p + out2, out ≠ out2)
    // must be exactly 0.
    let term = build_term();
    let dummy_state = FamilyLinearizationState {
        beta: &[],
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let specs = term.decoder_parameter_block_specs();
    for (atom_idx, spec) in specs.iter().enumerate() {
        let jac = spec
            .effective_jacobian_at("test", &dummy_state)
            .unwrap_or_else(|e| panic!("effective_jacobian_at failed for atom {atom_idx}: {e}"));
        for row in 0..N {
            for m in 0..M_BASIS {
                for out_row in 0..P_OUT {
                    for out_col in 0..P_OUT {
                        if out_row != out_col {
                            let v = jac[[row * P_OUT + out_row, m * P_OUT + out_col]];
                            assert_eq!(
                                v, 0.0,
                                "atom {atom_idx} row={row} m={m}: off-diagonal entry \
                                 J[{},{},{}] = {v} must be 0 (block-diagonal in output)",
                                row, out_row, out_col,
                            );
                        }
                    }
                }
            }
        }
    }
}
