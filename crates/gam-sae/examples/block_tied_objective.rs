//! Measure fixed-code and recomputed tied-code loss on a three-row witness.
//! Both rank-one blocks are selected on every row; routing cannot change.

use gam_sae::sparse_dict::{BlockSparseConfig, BlockSparseStreamState};
use ndarray::{Array2, array};

fn residual_energy(x: &Array2<f64>, prediction: &Array2<f64>, gamma: f32) -> f64 {
    x.iter()
        .zip(prediction.iter())
        .map(|(&x, &p)| (x - gamma as f64 * p).powi(2))
        .sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[0.4_f32, 4.0], [0.3, -3.0], [-0.3, 3.0]];
    let mut decoder = array![[1.0_f32, -10.0], [10.0, 3.0]];
    for mut row in decoder.outer_iter_mut() {
        let norm = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        row.mapv_inplace(|value| (value as f64 / norm) as f32);
    }
    let mut config = BlockSparseConfig::new(2, 1);
    config.block_topk = 2;
    config.minibatch = x.nrows();
    config.frame_ridge = 0.0;
    config.aux_k = 0;
    let mut state = BlockSparseStreamState::new_with_decoder(decoder.clone(), &config)?;
    state.partial_fit(x.view())?;
    let first = state.end_epoch()?;
    let proposal = state.decoder().mapv(f64::from);
    let x64 = x.mapv(f64::from);
    let old = decoder.mapv(f64::from);
    let frozen_codes = x64.dot(&old.t());
    let before = residual_energy(&x64, &frozen_codes.dot(&old), first.gamma);
    let frozen_after = residual_energy(&x64, &frozen_codes.dot(&proposal), first.gamma);
    state.partial_fit(x.view())?;
    let second = state.end_epoch()?;
    let tied_after = residual_energy(&x64, &x64.dot(&proposal.t()).dot(&proposal), second.gamma);
    println!(
        "{}",
        serde_json::json!({
            "before_profiled_rss": before,
            "after_fixed_code_rss": frozen_after,
            "after_tied_profiled_rss": tied_after,
            "before_gamma": first.gamma,
            "after_gamma": second.gamma,
            "first_frame_residual": first.frame_residual,
            "second_frame_residual": second.frame_residual,
            "before_decoder": old,
            "proposed_decoder": proposal
        })
    );
    if tied_after > before {
        return Err("recomputed tied-code objective increased after the frame proposal".into());
    }
    Ok(())
}
