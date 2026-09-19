//! #2572 repro: the overcomplete support-sparse (`assignment="topk"`, `K > P`)
//! fit through the exact stages `fit_support_sparse_manifold_sae` runs, in a
//! Python-free binary so the `ndarray: index out of bounds` panic gets a
//! symbolized frame.
//!
//! ```text
//! cargo run -p gam-sae --example issue_2572_repro -- \
//!     chart.bin <rows> <cols> <max_outer_iter> <k:s> [<k:s> ...]
//! ```
//!
//! Each `<k:s>` cell is run under `catch_unwind`, so one invocation sweeps the
//! shape ladder the issue reports and prints `PANIC` / `Err` / `Ok` per cell.

use gam_sae::front_door::{SaeFitLane, admit_topk_manifold};
use gam_sae::manifold::{
    SaeSupportOuterRequest, SaeSupportSeedRequest, SaeSupportTermSeedRequest,
    build_sae_support_seed, build_sae_support_term_seed, resolve_support_auto_atoms,
    run_sae_support_outer, sae_support_effective_atom_dims,
};
use ndarray::{Array2, Axis};

struct Budget {
    max_outer: usize,
}

fn one_cell(target: &Array2<f64>, k_atoms: usize, top_k: usize, budget: &Budget) -> String {
    let (rows, cols) = target.dim();
    // Exactly what the FFI front door does for `assignment="topk", K > P`.
    let mut atom_basis = vec!["auto".to_string(); k_atoms];
    resolve_support_auto_atoms(&mut atom_basis);
    let atom_dim = vec![1usize; k_atoms];
    let effective = match sae_support_effective_atom_dims(&atom_basis, &atom_dim) {
        Ok(dims) => dims,
        Err(error) => return format!("Err(dims): {error}"),
    };
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission = match admit_topk_manifold(rows, cols, k_atoms, d_max, top_k) {
        Ok(admission) => admission,
        Err(error) => return format!("Err(admission): {error}"),
    };
    if admission.lane != SaeFitLane::CurvedStreaming {
        return format!("Err(lane): {:?}", admission.lane);
    }
    let mean = match target.mean_axis(Axis(0)) {
        Some(mean) => mean,
        None => return "Err(mean)".to_string(),
    };
    let centered = target - &mean;

    let seed = match build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k: top_k,
        random_state: 0,
        admission,
    }) {
        Ok(seed) => seed,
        Err(error) => return format!("Err(seed): {error}"),
    };
    let retained = seed.retained_atom_indices.clone();
    let retained_basis: Vec<String> = retained.iter().map(|&a| atom_basis[a].clone()).collect();
    let retained_dim: Vec<usize> = retained.iter().map(|&a| atom_dim[a]).collect();
    let term_seed = match build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained_basis,
        atom_dim: retained_dim,
        output_dim: cols,
        random_state: 0,
    }) {
        Ok(seeded) => seeded,
        Err(error) => return format!("Err(term seed): {error}"),
    };
    let ard_precisions: Vec<Vec<f64>> = (0..term_seed.term.k_atoms())
        .map(|atom| vec![1.0; term_seed.term.assignment.atom_coord_dim(atom)])
        .collect();

    match run_sae_support_outer(SaeSupportOuterRequest {
        term: term_seed.term,
        target: centered,
        initial_smoothness: 1.0,
        ard_precisions,
        max_outer_iter: budget.max_outer,
        trust_radius: 1.0,
        random_state: 0,
    }) {
        Ok(report) => format!(
            "Ok: criterion={:.6e} outer_iters={} inner_iters={} retained={}",
            report.criterion.value(),
            report.outer_iterations,
            report.fixed_point.iterations,
            retained.len()
        ),
        Err(error) => format!("Err(outer): {error}"),
    }
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        return Err(
            "usage: issue_2572_repro <f64-le.bin> <rows> <cols> <max_outer_iter> <k:s>..."
                .into(),
        );
    }
    let rows: usize = args[2].parse().map_err(|e| format!("rows: {e}"))?;
    let cols: usize = args[3].parse().map_err(|e| format!("cols: {e}"))?;
    let max_outer: usize = args[4].parse().map_err(|e| format!("max_outer: {e}"))?;
    let budget = Budget { max_outer };

    let bytes = std::fs::read(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    if bytes.len() < rows * cols * 8 {
        return Err(format!("chart holds {} bytes < rows*cols*8", bytes.len()));
    }
    let data: Vec<f64> = bytes[..rows * cols * 8]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect();
    let target = Array2::from_shape_vec((rows, cols), data).map_err(|e| e.to_string())?;

    for cell in &args[5..] {
        let (k_text, s_text) = cell.split_once(':').ok_or("cell must be <k>:<s>")?;
        let k_atoms: usize = k_text.parse().map_err(|e| format!("k: {e}"))?;
        let top_k: usize = s_text.parse().map_err(|e| format!("top_k: {e}"))?;
        let started = std::time::Instant::now();
        let outcome = std::panic::catch_unwind(|| one_cell(&target, k_atoms, top_k, &budget));
        let elapsed = started.elapsed().as_secs_f64();
        match outcome {
            Ok(text) => println!("N={rows} P={cols} K={k_atoms} s={top_k} [{elapsed:.1}s] {text}"),
            Err(_) => println!("N={rows} P={cols} K={k_atoms} s={top_k} [{elapsed:.1}s] PANIC"),
        }
    }
    Ok(())
}
