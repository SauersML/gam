//! Regression gate for SAE arrow-Schur assembly and solve at large-scale-adjacent
//! scale (`K_atoms ∈ {8, 16, 32}`).
//!
//! Exercises the structural costs that are invisible at K≤2:
//!
//!   - dense `(q × beta_dim)` `H_tβ` cross-block where `q = K*(1 + d)`
//!   - dense `q × q` per-row `H_tt^(i)` block
//!   - dense `beta_dim × beta_dim` `sys.hbb` shared block (beta_dim = K*M*p)
//!   - both Softmax and JumpReLU assignment variants
//!   - Direct and InexactPCG solve modes
//!
//! Assertions are weak (finite, non-NaN) to survive future algorithmic
//! changes; the test is a timing and stability gate, not a numerics check.
//!
//! Sizing is chosen to fit in CI RAM (≤ 1 GB total):
//!   K=8,  M=4,  d=1, N=500  → q=16,  beta=64
//!   K=16, M=4,  d=1, N=500  → q=32,  beta=128
//!   K=32, M=8,  d=1, N=500  → q=64,  beta=512
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use ndarray::{Array1, Array2, Array3};

use gam::solver::arrow_schur::ArrowSolveOptions;
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm,
};

/// Deterministic pseudo-random f64 ∈ (-1, 1) via LCG.
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

struct Fixture {
    term: SaeManifoldTerm,
    target: Array2<f64>,
    rho: SaeManifoldRho,
}

fn build_fixture(
    k_atoms: usize,
    basis_size: usize,
    latent_dim: usize,
    n_obs: usize,
    p_out: usize,
    mode: AssignmentMode,
) -> Fixture {
    let m = basis_size;
    let d = latent_dim;
    let n = n_obs;
    let p = p_out;

    let mut rng: u64 = 0x1234_5678_9abc_def0u64
        .wrapping_add(k_atoms as u64 * 97)
        .wrapping_add(n as u64 * 7);

    let logits = Array2::from_shape_fn((n, k_atoms), |_| lcg_f64(&mut rng) * 0.5);
    let target = Array2::from_shape_fn((n, p), |_| lcg_f64(&mut rng));

    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(k_atoms);
    let mut coord_blocks: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);

    for _k in 0..k_atoms {
        let phi = Array2::from_shape_fn((n, m), |_| lcg_f64(&mut rng) * 0.1);
        let jet = Array3::from_shape_fn((n, m, d), |_| lcg_f64(&mut rng) * 0.01);
        let decoder = Array2::from_shape_fn((m, p), |_| lcg_f64(&mut rng) * 0.3);
        let mut smooth = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            // Diagonal-dominant → PD.
            smooth[[i, i]] = 0.1 + 0.01 * lcg_f64(&mut rng).abs();
        }
        let atom = SaeManifoldAtom::new(
            format!("atom_{_k}"),
            SaeAtomBasisKind::EuclideanPatch,
            d,
            phi,
            jet,
            decoder,
            smooth,
        )
        .unwrap_or_else(|e| panic!("SaeManifoldAtom::new failed: {e}"));
        atoms.push(atom);

        let coords = Array2::from_shape_fn((n, d), |_| lcg_f64(&mut rng) * 0.5);
        coord_blocks.push(coords);
    }

    let assignment = SaeAssignment::from_blocks_with_mode(logits, coord_blocks, mode)
        .unwrap_or_else(|e| panic!("SaeAssignment::from_blocks_with_mode failed: {e}"));

    let term = SaeManifoldTerm::new(atoms, assignment)
        .unwrap_or_else(|e| panic!("SaeManifoldTerm::new failed: {e}"));

    let log_ard: Vec<Array1<f64>> = (0..k_atoms)
        .map(|_| Array1::from_elem(latent_dim, 0.0_f64))
        .collect();
    let rho = SaeManifoldRho::new(0.0, -4.0, log_ard);

    Fixture { term, target, rho }
}

fn assert_finite_vec(v: &[f64], ctx: &str) {
    for (i, &x) in v.iter().enumerate() {
        assert!(x.is_finite(), "{ctx}: element [{i}] = {x} is not finite");
    }
}

// ---------------------------------------------------------------------------
// K=8 tests
// ---------------------------------------------------------------------------

#[test]
fn k8_softmax_assembly_is_finite() {
    let mut f = build_fixture(8, 4, 1, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    assert_finite_vec(sys.gb.as_slice().unwrap(), "k8_softmax gb");
    assert_finite_vec(sys.hbb.as_slice().unwrap(), "k8_softmax hbb");
}

#[test]
fn k8_jumprelu_assembly_is_finite() {
    let mut f = build_fixture(8, 4, 1, 500, 2, AssignmentMode::jumprelu(1.0, 0.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    assert_finite_vec(sys.gb.as_slice().unwrap(), "k8_jumprelu gb");
    assert_finite_vec(sys.hbb.as_slice().unwrap(), "k8_jumprelu hbb");
}

#[test]
fn k8_softmax_direct_solve_is_finite() {
    let mut f = build_fixture(8, 4, 1, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    let (delta_t, delta_beta, _diag) = sys
        .solve_with_options(1e-4, 1e-4, &ArrowSolveOptions::direct())
        .unwrap_or_else(|e| panic!("direct solve failed: {e}"));
    assert_finite_vec(delta_t.as_slice().unwrap(), "k8 direct delta_t");
    assert_finite_vec(delta_beta.as_slice().unwrap(), "k8 direct delta_beta");
}

#[test]
fn k8_softmax_pcg_solve_is_finite() {
    let mut f = build_fixture(8, 4, 1, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    let (delta_t, delta_beta, _diag) = sys
        .solve_with_options(1e-4, 1e-4, &ArrowSolveOptions::inexact_pcg())
        .unwrap_or_else(|e| panic!("PCG solve failed: {e}"));
    assert_finite_vec(delta_t.as_slice().unwrap(), "k8 pcg delta_t");
    assert_finite_vec(delta_beta.as_slice().unwrap(), "k8 pcg delta_beta");
}

// ---------------------------------------------------------------------------
// K=16 tests
// ---------------------------------------------------------------------------

#[test]
fn k16_softmax_assembly_is_finite() {
    let mut f = build_fixture(16, 4, 1, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    assert_finite_vec(sys.gb.as_slice().unwrap(), "k16_softmax gb");
    assert_finite_vec(sys.hbb.as_slice().unwrap(), "k16_softmax hbb");
}

#[test]
fn k16_jumprelu_assembly_is_finite() {
    let mut f = build_fixture(16, 4, 1, 500, 2, AssignmentMode::jumprelu(1.0, 0.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    assert_finite_vec(sys.gb.as_slice().unwrap(), "k16_jumprelu gb");
    assert_finite_vec(sys.hbb.as_slice().unwrap(), "k16_jumprelu hbb");
}

#[test]
fn k16_softmax_direct_solve_is_finite() {
    let mut f = build_fixture(16, 4, 1, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    let (delta_t, delta_beta, _diag) = sys
        .solve_with_options(1e-4, 1e-4, &ArrowSolveOptions::direct())
        .unwrap_or_else(|e| panic!("direct solve failed: {e}"));
    assert_finite_vec(delta_t.as_slice().unwrap(), "k16 direct delta_t");
    assert_finite_vec(delta_beta.as_slice().unwrap(), "k16 direct delta_beta");
}

// ---------------------------------------------------------------------------
// K=32 tests
// ---------------------------------------------------------------------------

#[test]
fn k32_softmax_assembly_is_finite() {
    let mut f = build_fixture(32, 8, 1, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    assert_finite_vec(sys.gb.as_slice().unwrap(), "k32_softmax gb");
    assert_finite_vec(sys.hbb.as_slice().unwrap(), "k32_softmax hbb");
}

#[test]
fn k32_jumprelu_assembly_is_finite() {
    let mut f = build_fixture(32, 8, 1, 500, 2, AssignmentMode::jumprelu(1.0, 0.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    assert_finite_vec(sys.gb.as_slice().unwrap(), "k32_jumprelu gb");
    assert_finite_vec(sys.hbb.as_slice().unwrap(), "k32_jumprelu hbb");
}

#[test]
fn k32_softmax_direct_solve_is_finite() {
    let mut f = build_fixture(32, 8, 1, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    let (delta_t, delta_beta, _diag) = sys
        .solve_with_options(1e-4, 1e-4, &ArrowSolveOptions::direct())
        .unwrap_or_else(|e| panic!("direct solve failed: {e}"));
    assert_finite_vec(delta_t.as_slice().unwrap(), "k32 direct delta_t");
    assert_finite_vec(delta_beta.as_slice().unwrap(), "k32 direct delta_beta");
}

#[test]
fn k32_softmax_pcg_solve_is_finite() {
    let mut f = build_fixture(32, 8, 1, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    let (delta_t, delta_beta, _diag) = sys
        .solve_with_options(1e-4, 1e-4, &ArrowSolveOptions::inexact_pcg())
        .unwrap_or_else(|e| panic!("PCG solve failed: {e}"));
    assert_finite_vec(delta_t.as_slice().unwrap(), "k32 pcg delta_t");
    assert_finite_vec(delta_beta.as_slice().unwrap(), "k32 pcg delta_beta");
}

// ---------------------------------------------------------------------------
// Full Newton step test at K=16 (assemble + solve in one call)
// ---------------------------------------------------------------------------

#[test]
fn k16_full_newton_step_is_finite() {
    let f = build_fixture(16, 4, 1, 500, 2, AssignmentMode::softmax(1.0));
    let mut term = f.term;
    let mut rho = f.rho;
    let target = f.target;

    let loss = term
        .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 1.0, 1e-4, 1e-4)
        .unwrap_or_else(|e| panic!("run_joint_fit_arrow_schur failed: {e}"));

    assert!(
        loss.total().is_finite(),
        "k16 Newton step loss is not finite: {}",
        loss.total()
    );
}

// ---------------------------------------------------------------------------
// hbb size probe: confirms the K*M*p × K*M*p block is actually constructed
// ---------------------------------------------------------------------------

#[test]
fn hbb_shape_matches_beta_dim() {
    let k_atoms = 16usize;
    let m = 4usize;
    let p = 2usize;
    let beta_dim = k_atoms * m * p; // 128

    let mut f = build_fixture(k_atoms, m, 1, 500, p, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));

    assert_eq!(
        sys.hbb.dim(),
        (beta_dim, beta_dim),
        "hbb shape mismatch: expected ({beta_dim}, {beta_dim}), got {:?}",
        sys.hbb.dim()
    );
    assert_eq!(
        sys.gb.len(),
        beta_dim,
        "gb length mismatch: expected {beta_dim}, got {}",
        sys.gb.len()
    );
}

// ---------------------------------------------------------------------------
// Block-diagonal data-fit β-Hessian reduction must match the dense reference.
//
// The data-fit Gauss-Newton β-Hessian `Jβᵀ Jβ` is block-diagonal across the
// `p` output channels and identical per channel, so the assembler now stores
// it as a single `(M_total × M_total)` block `G` installed as `G ⊗ I_p`
// (a `KroneckerPenaltyOp`) inside the composite penalty operator, rather than
// materialising the dense `(K·M·p)²` `sys.hbb`. This test pins that the
// structured operator's dense form exactly reproduces the dense reference
// (data-fit GN + decoder smoothness), assembled independently here.
// ---------------------------------------------------------------------------

#[test]
fn data_fit_beta_hessian_kronecker_matches_dense_reference() {
    let k_atoms = 8usize;
    let m = 4usize;
    let d = 1usize;
    let n = 500usize;
    let p = 2usize;
    let mut f = build_fixture(k_atoms, m, d, n, p, AssignmentMode::softmax(1.0));
    let lambda_smooth = f.rho.lambda_smooth();

    // Snapshot per-atom static data needed to build the dense reference.
    let beta_dim = f.term.beta_dim();
    let beta_offsets = f.term.beta_offsets();
    let assignments = f.term.assignment.assignments();
    let basis_values: Vec<ndarray::Array2<f64>> = f
        .term
        .atoms
        .iter()
        .map(|a| a.basis_values.clone())
        .collect();
    let smooth_penalties: Vec<ndarray::Array2<f64>> = f
        .term
        .atoms
        .iter()
        .map(|a| a.smooth_penalty.clone())
        .collect();

    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));

    // Dense reference β-Hessian = data-fit GN outer products + decoder
    // smoothness, indexed exactly as the flat β layout β[off + col*p + oc].
    let mut reference = Array2::<f64>::zeros((beta_dim, beta_dim));
    for row in 0..n {
        // Per-row support: (global β base, a_k·φ_k[col]).
        let mut a_phi: Vec<(usize, f64)> = Vec::new();
        for atom_idx in 0..k_atoms {
            let off = beta_offsets[atom_idx];
            let a_k = assignments[[row, atom_idx]];
            for col in 0..m {
                a_phi.push((off + col * p, a_k * basis_values[atom_idx][[row, col]]));
            }
        }
        for &(base_i, ji) in &a_phi {
            for &(base_j, jj) in &a_phi {
                for oc in 0..p {
                    reference[[base_i + oc, base_j + oc]] += ji * jj;
                }
            }
        }
    }
    for atom_idx in 0..k_atoms {
        let off = beta_offsets[atom_idx];
        let s = &smooth_penalties[atom_idx];
        for i in 0..m {
            for j in 0..m {
                let s_ij = 0.5 * (s[[i, j]] + s[[j, i]]) * lambda_smooth;
                for oc in 0..p {
                    reference[[off + i * p + oc, off + j * p + oc]] += s_ij;
                }
            }
        }
    }

    let actual = sys.effective_penalty_op().to_dense();
    assert_eq!(
        actual.dim(),
        (beta_dim, beta_dim),
        "penalty op dense shape mismatch"
    );
    let mut max_abs = 0.0_f64;
    for a in 0..beta_dim {
        for b in 0..beta_dim {
            max_abs = max_abs.max((actual[[a, b]] - reference[[a, b]]).abs());
        }
    }
    assert!(
        max_abs < 1e-9,
        "structured penalty op dense form deviates from dense reference: max|Δ|={max_abs:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Row block dimension probe: confirms q = K*(1+d) layout
// ---------------------------------------------------------------------------

#[test]
fn row_block_dim_matches_k_times_one_plus_d() {
    let k_atoms = 16usize;
    let d = 1usize;
    let expected_q = k_atoms * (1 + d); // 32

    let mut f = build_fixture(k_atoms, 4, d, 500, 2, AssignmentMode::softmax(1.0));
    let sys = f
        .term
        .assemble_arrow_schur(f.target.view(), &f.rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));

    let actual_q = sys.d;
    assert_eq!(
        actual_q, expected_q,
        "row block dim q mismatch: expected {expected_q}, got {actual_q}"
    );
    for (i, row) in sys.rows.iter().enumerate() {
        assert_eq!(
            row.htt.dim(),
            (expected_q, expected_q),
            "row[{i}].htt shape mismatch: expected ({expected_q}, {expected_q})"
        );
    }
}
