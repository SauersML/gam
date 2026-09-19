//! #2572: the PCG preconditioner ladder reads a dense `H_tβ` slab that a
//! matrix-free system does not have.
//!
//! The overcomplete support-sparse lane assembles its arrow system with
//! `htbeta_cols = 0` and installs `H_tβ` as a matvec pair
//! (`set_row_htbeta_operator`), then registers per-atom `block_offsets`. The
//! reduced-Schur kernels that were taught this convention route every cross
//! block through `sys_htbeta_apply_row` at the row's own width, and say why:
//!
//! > never a raw `row.htbeta` read at the global `sys.d`: matvec-backed rows
//! > carry absent/zero-sized slabs by contract (a raw read is wrong or panics)
//!
//! `BetaCouplingGraph::build` is the one place on the ladder that was not, and
//! it is what every escalated tier (ClusterJacobi, AdditiveSchwarz,
//! DiagAssembledSchwarz, BlockIncompleteCholesky, and the co-visibility group
//! builder) constructs first.
//!
//! ```text
//! cargo run -p gam-sae --release --example issue_2572_precond_probe
//! ```

use gam_sae::front_door::admit_topk_manifold;
use gam_sae::manifold::{
    SaeSupportSeedRequest, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, resolve_support_auto_atoms, sae_support_effective_atom_dims,
};
use gam_solve::arrow_schur::{
    AdditiveSchwarzPreconditioner, ArrowSchurSystem, BatchedBlockSolver, ClusterJacobiPreconditioner,
    CpuBatchedBlockSolver,
};
use ndarray::{Array2, Axis};

fn report(label: &str, run: impl FnOnce() -> Result<String, String>) {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(text)) => println!("{label:<34} Ok({text})"),
        Ok(Err(error)) => println!("{label:<34} Err({})", &error[..error.len().min(120)]),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            println!("{label:<34} PANIC({message})");
        }
    }
}

fn main() -> Result<(), String> {
    let (n_obs, p_out, k_atoms, support_k) = (240usize, 8usize, 24usize, 4usize);
    let mut target = Array2::<f64>::zeros((n_obs, p_out));
    for row in 0..n_obs {
        for col in 0..p_out {
            let t = (row * 7 + col * 13) as f64;
            target[[row, col]] = (0.1 * t).sin() + 0.3 * (0.03 * t).cos();
        }
    }
    let mut atom_basis = vec!["auto".to_string(); k_atoms];
    resolve_support_auto_atoms(&mut atom_basis);
    let atom_dim = vec![1usize; k_atoms];
    let effective = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission = admit_topk_manifold(n_obs, p_out, k_atoms, d_max, support_k)?;
    let mean = target.mean_axis(Axis(0)).ok_or("empty target")?;
    let centered = &target - &mean;
    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k,
        random_state: 0,
        admission,
    })?;
    let retained = seed.retained_atom_indices.clone();
    let term = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained.iter().map(|&a| atom_basis[a].clone()).collect(),
        atom_dim: retained.iter().map(|&a| atom_dim[a]).collect(),
        output_dim: p_out,
        random_state: 0,
    })?
    .term;
    let ard: Vec<Vec<f64>> = (0..term.k_atoms())
        .map(|atom| vec![1.0; term.assignment.atom_coord_dim(atom)])
        .collect();
    let lambda = vec![1.0_f64; term.k_atoms()];
    // A cold decoder leaves every `H_tt` singular (the latent Gauss-Newton block
    // is `J' J` with `J = 0`), so drive the inner fixed point until the row blocks
    // are PD and the factorization the ladder needs exists. The fixed point is the
    // means, not the measurement.
    let mut term = term;
    let inner = term
        .solve_fixed_point(centered.view(), &lambda, &ard, 1.0e-4, 1.0)
        .map(|report| format!("recurred in {}", report.iterations))
        .unwrap_or_else(|error| format!("did not recur: {}", &error[..error.len().min(60)]));
    println!("inner fixed point: {inner}");
    let system: ArrowSchurSystem = term.assemble_arrow_schur(centered.view(), &lambda, &ard)?;

    println!(
        "assembled: border k={}, rows={}, row dims {:?}, htbeta slab {:?}, block_offsets {}",
        system.k,
        system.rows.len(),
        system.rows[0].htt.dim(),
        system.rows[0].htbeta.dim(),
        system.block_offsets.len(),
    );

    let backend = CpuBatchedBlockSolver;
    let htt = backend
        .factor_blocks(&system.rows, 0.0, system.d, true)
        .map_err(|error| format!("row factorization: {error}"))?;

    // These are the escalated tiers of `steihaug_pcg_auto`'s preconditioner
    // ladder. Plain Jacobi (the first tier) reads the routed diagonal and is
    // fine; every tier above it starts by building the coupling graph.
    report("ClusterJacobi", || {
        ClusterJacobiPreconditioner::from_arrow_schur(&system, &htt, 0.0, &backend)
            .map(|_| "built".to_string())
            .map_err(|e| e.to_string())
    });
    report("AdditiveSchwarz{overlap:1}", || {
        AdditiveSchwarzPreconditioner::from_arrow_schur(&system, &htt, 0.0, &backend, 1)
            .map(|_| "built".to_string())
            .map_err(|e| e.to_string())
    });
    Ok(())
}
