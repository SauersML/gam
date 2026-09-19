//! Fast solver-iteration harness for the overcomplete hard-TopK support lane:
//! the real fit (auto-portfolio seed → inner fixed point → outer LAML) on a
//! dumped chart matrix, without building the Python wheel. Compiles with only
//! gam-sae's own dependency cone, so a solver edit iterates in ~1–2 minutes
//! instead of a full workspace wheel build.
//!
//! ```text
//! python: open("chart.bin","wb").write(np.ascontiguousarray(X, dtype="<f8").tobytes())
//! cargo run -p gam-sae --release --example support_real_chart -- \
//!     chart.bin <rows> <cols> <k_atoms> <top_k> <max_outer_iter>
//! ```
//!
//! `RUST_LOG=info` streams the per-cycle fixed-point telemetry.

use gam_sae::front_door::{SaeFitLane, admit_topk_manifold};
use gam_sae::manifold::{
    SaeSupportOuterRequest, SaeSupportSeedRequest, SaeSupportTermSeedRequest,
    build_sae_support_seed, build_sae_support_term_seed, resolve_support_auto_atoms,
    run_sae_support_outer, sae_support_effective_atom_dims,
};
use ndarray::{Array2, Axis};
use std::time::Instant;

fn main() -> Result<(), String> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: support_real_chart <f64-le.bin> <rows> <cols> <k_atoms> <top_k> <max_outer_iter>"
                .to_string(),
        );
    }
    let rows: usize = args[2].parse().map_err(|error| format!("rows: {error}"))?;
    let cols: usize = args[3].parse().map_err(|error| format!("cols: {error}"))?;
    let k_atoms: usize = args[4].parse().map_err(|error| format!("k_atoms: {error}"))?;
    let top_k: usize = args[5].parse().map_err(|error| format!("top_k: {error}"))?;
    // The outer search's budget. The inner fixed point has none: it stops on its
    // certificate or on a proven stall (#2576).
    let max_outer_iter: usize = args[6]
        .parse()
        .map_err(|error| format!("max_outer_iter: {error}"))?;
    let bytes = std::fs::read(&args[1]).map_err(|error| format!("{}: {error}", args[1]))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!(
            "{} holds {} bytes; rows*cols*8 = {}",
            args[1],
            bytes.len(),
            rows * cols * 8
        ));
    }
    let data: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
        .collect();
    let target = Array2::from_shape_vec((rows, cols), data).map_err(|error| format!("{error}"))?;

    let mut atom_basis = vec!["auto".to_string(); k_atoms];
    resolve_support_auto_atoms(&mut atom_basis);
    let atom_dim = vec![1usize; k_atoms];
    let effective = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission = admit_topk_manifold(rows, cols, k_atoms, d_max, top_k)?;
    if admission.lane != SaeFitLane::CurvedStreaming {
        return Err(format!("expected CurvedStreaming admission; got {:?}", admission.lane));
    }

    let mean = target.mean_axis(Axis(0)).ok_or("empty target")?;
    let centered = &target - &mean;

    let t_seed = Instant::now();
    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k: top_k,
        random_state: 0,
        admission,
    })?;
    let retained = seed.retained_atom_indices.clone();
    println!(
        "seed: {:.1}s, retained {} of {} atoms",
        t_seed.elapsed().as_secs_f64(),
        retained.len(),
        k_atoms
    );
    let term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained.iter().map(|&atom| atom_basis[atom].clone()).collect(),
        atom_dim: retained.iter().map(|&atom| atom_dim[atom]).collect(),
        output_dim: cols,
        random_state: 0,
    })?;
    let ard_precisions = (0..term_seed.term.k_atoms())
        .map(|atom| vec![1.0; term_seed.term.assignment.atom_coord_dim(atom)])
        .collect::<Vec<_>>();

    let t_fit = Instant::now();
    let outer = run_sae_support_outer(SaeSupportOuterRequest {
        term: term_seed.term,
        target: centered.clone(),
        initial_smoothness: 1.0,
        ard_precisions,
        max_outer_iter,
        trust_radius: 1.0,
        random_state: 0,
    })
    .map_err(|error| error.to_string())?;
    let fitted = outer.term.reconstruct()?;
    let residual_ss: f64 = centered
        .iter()
        .zip(fitted.iter())
        .map(|(truth, fit)| (truth - fit).powi(2))
        .sum();
    let total_ss: f64 = centered.iter().map(|value| value * value).sum();
    println!(
        "fit: {:.1}s, centered train EV = {:.4}",
        t_fit.elapsed().as_secs_f64(),
        1.0 - residual_ss / total_ss
    );
    Ok(())
}
