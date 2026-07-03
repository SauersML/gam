//! Rung-2 flagship demo: fit ONE weekday atom as a two-block (activation +
//! behavior) manifold term with the block weight λ_y selected by REML, and
//! report behaviorally-calibrated units — how many nats a latent step Δt is
//! worth, BY CONSTRUCTION of the unit-speed sphere-tangent behavior chart.
//!
//! Input: two CSV files (no headers unless `--header` rows are plain numbers):
//!   * activations `n × p_x` — per-token residual-stream activations (the
//!     probe subspace / PCA projection the weekday feature lives in);
//!   * probs `n × V` — per-token next-token probabilities over a RESTRICTED
//!     token set (e.g. the 7 weekday names + competitors), row-aligned with
//!     the activations. Rows need not be normalized.
//!
//! Usage:
//!   cargo run -p gam-sae --release --example two_block_weekday_demo -- \
//!       <activations.csv> <probs.csv> [harmonics]
//!
//! Output: REML-selected λ_y, per-block explained variance, and the behavioral
//! speed profile of the fitted circle — nats per unit Δt around the chart —
//! plus the same fit run activation-only for a side-by-side.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use gam_sae::manifold::{
    AssignmentMode, BehaviorBlock, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    SphereTangentEmbedding,
};

fn read_csv_matrix(path: &str) -> Result<Array2<f64>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: Result<Vec<f64>, _> = line
            .split(',')
            .map(|tok| tok.trim().parse::<f64>())
            .collect();
        match row {
            Ok(values) => rows.push(values),
            Err(_) if lineno == 0 => continue, // tolerate a header row
            Err(e) => return Err(format!("{path}:{}: {e}", lineno + 1)),
        }
    }
    let n = rows.len();
    if n == 0 {
        return Err(format!("{path}: no data rows"));
    }
    let p = rows[0].len();
    let mut out = Array2::<f64>::zeros((n, p));
    for (i, row) in rows.iter().enumerate() {
        if row.len() != p {
            return Err(format!(
                "{path}: row {} has {} fields, expected {p}",
                i + 1,
                row.len()
            ));
        }
        for (j, &v) in row.iter().enumerate() {
            out[[i, j]] = v;
        }
    }
    Ok(out)
}

/// Build a K=1 circle term at output width `p_tot`, seeded at `coords`.
fn circle_term(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new(
        "weekday",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )?
    .with_basis_second_jet(evaluator.clone());
    let n = coords.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )?;
    let term = SaeManifoldTerm::new(vec![atom], assignment)?;
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    Ok((term, rho))
}

fn explained_variance(target: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let (n, p) = target.dim();
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for j in 0..p {
        let mean = (0..n).map(|i| target[[i, j]]).sum::<f64>() / n as f64;
        for i in 0..n {
            let r = target[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            let c = target[[i, j]] - mean;
            sst += c * c;
        }
    }
    1.0 - ssr / sst
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(format!(
            "usage: {} <activations.csv> <probs.csv> [harmonics]",
            args[0]
        ));
    }
    let z = read_csv_matrix(&args[1])?;
    let probs = read_csv_matrix(&args[2])?;
    let harmonics: usize = args
        .get(3)
        .map(|s| s.parse().map_err(|e| format!("harmonics: {e}")))
        .transpose()?
        .unwrap_or(3);
    let (n, p_x) = z.dim();
    let vocab = probs.ncols();
    if probs.nrows() != n {
        return Err(format!(
            "row mismatch: activations n={n}, probs n={}",
            probs.nrows()
        ));
    }
    println!("n = {n} tokens, p_x = {p_x} activation dims, V = {vocab} restricted tokens");

    // Seed the circle coordinate by phase of the two leading activation PCs
    // (cheap deterministic seed; the fit refines it jointly).
    let evaluator = Arc::new(
        PeriodicHarmonicEvaluator::new(2 * harmonics + 1).map_err(|e| e.to_string())?,
    );
    let mut coords = Array2::<f64>::zeros((n, 1));
    // Column means for centering.
    let mean0 = z.column(0).sum() / n as f64;
    let mean1 = if p_x > 1 {
        z.column(1).sum() / n as f64
    } else {
        0.0
    };
    for i in 0..n {
        let a = z[[i, 0]] - mean0;
        let b = if p_x > 1 { z[[i, 1]] - mean1 } else { 0.0 };
        coords[[i, 0]] = (b.atan2(a) / std::f64::consts::TAU).rem_euclid(1.0);
    }

    // --- Two-block REML fit. ---
    let block = BehaviorBlock::fit(probs.view(), p_x, 0.0)?;
    let p_y = block.behavior_dim();
    let p_tot = p_x + p_y;
    let (mut term, mut rho) = circle_term(&evaluator, &coords, p_tot)?;
    term.set_behavior_block(block)?;
    term.set_guards_enabled(false);
    let report =
        term.run_two_block_reml_fit(z.view(), &mut rho, None, 30, 60, 1.0, 1e-6, 1e-6, 1e-3)?;
    let block = term
        .behavior_block()
        .expect("installed above")
        .clone();
    println!(
        "REML-selected log λ_y = {:.4} (λ_y = {:.4}); sweeps = {}, converged = {}, identifiable = {}",
        report.log_lambda_y,
        report.log_lambda_y.exp(),
        report.sweeps,
        report.converged,
        report.lambda_identifiable
    );

    let augmented = block.augmented_target(z.view())?;
    let fitted = term.try_fitted_for_rho(&rho)?;
    let act_t = augmented.slice(ndarray::s![.., ..p_x]).to_owned();
    let act_f = fitted.slice(ndarray::s![.., ..p_x]).to_owned();
    let beh_t = augmented.slice(ndarray::s![.., p_x..]).to_owned();
    let beh_f = fitted.slice(ndarray::s![.., p_x..]).to_owned();
    println!(
        "two-block fit: activation EV = {:.4}, behavior EV = {:.4}",
        explained_variance(&act_t, &act_f),
        explained_variance(&beh_t, &beh_f)
    );

    // --- Behavioral speed profile: nats per unit Δt around the fitted circle.
    // Decode the behavior block of the fitted curve at a fine grid of t and
    // measure exact KL between neighbouring decoded distributions — the
    // CALIBRATED SCALE on the dial. The chord version ‖Δy‖² is the flat
    // prediction; exact KL certifies it.
    let (_b_k, c_k) = block.split_decoder(term.atoms[0].decoder_coefficients.view())?;
    let grid = 64usize;
    let mut tg = Array2::<f64>::zeros((grid, 1));
    for g in 0..grid {
        tg[[g, 0]] = g as f64 / grid as f64;
    }
    let (phi_g, _) = evaluator.evaluate(tg.view()).map_err(|e| e.to_string())?;
    let y_curve = phi_g.dot(&c_k); // grid × p_y, nats-unit tangent coords
    let mut total_nats = 0.0_f64;
    let mut speeds = Vec::with_capacity(grid);
    for g in 0..grid {
        let g_next = (g + 1) % grid;
        let dy = &y_curve.row(g_next).to_owned() - &y_curve.row(g).to_owned();
        let flat = SphereTangentEmbedding::predicted_nats(dy.view());
        let p_a = block.embedding.decode(y_curve.row(g))?;
        let p_b = block.embedding.decode(y_curve.row(g_next))?;
        let kl = SphereTangentEmbedding::exact_kl(p_b.view(), p_a.view())?;
        total_nats += kl;
        speeds.push((tg[[g, 0]], flat, kl));
    }
    println!("behavioral circumference of the fitted weekday circle: {total_nats:.4} nats");
    println!("t, predicted_nats(chord), exact_kl (per 1/{grid} step):");
    for (t, flat, kl) in &speeds {
        println!("{t:.4}, {flat:.6}, {kl:.6}");
    }

    // --- Activation-only baseline for the side-by-side. ---
    let (mut term_a, mut rho_a) = circle_term(&evaluator, &coords, p_x)?;
    term_a.set_guards_enabled(false);
    term_a.run_joint_fit_arrow_schur(z.view(), &mut rho_a, None, 60, 1.0, 1e-6, 1e-6)?;
    let fitted_a = term_a.try_fitted_for_rho(&rho_a)?;
    println!(
        "activation-only baseline: activation EV = {:.4} (no behavioral units available)",
        explained_variance(&z, &fitted_a)
    );
    Ok(())
}
