//! Speed of the matrix-free structured-edit kernels at LLM width (#2951).
//!
//! `cargo run --release -p gam-sae --example mpd_apply_llm_width -- ROWS D_IN D_OUT TERMS ACTIVE REPS`
//!
//! Times, on deterministic fixtures of the declared shape:
//! * `native_linear`, the all-on path (`h W*ᵀ`), which is the floor;
//! * `apply_anchored_linear` with `m_Δ = 1` and `ACTIVE` of `TERMS` rank-one terms
//!   deleted (`s_k = −1`), whose excess over the floor is the edit's cost;
//! * `edit_frobenius_contractions` and `edit_factor_cotangents` against an output
//!   cotangent of the same row count.
//!
//! It also checks three rows of the apply against an explicit per-row reference
//! `W* h + Σ_k s_k u_k (v_kᵀ h)` within the derived roundoff band, and prints the
//! worst ratio of error to band. No `d_out × d_in` edit is formed anywhere.

use gam_linalg::roundoff::{accumulation_band, accumulation_growth};
use gam_runtime::resource::MemoryGovernor;
use gam_sae::parameter_decomposition::apply::{
    EditFootprint, FactoredEdit, apply_anchored_linear, edit_factor_cotangents,
    edit_frobenius_contractions, native_linear,
};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn fixture(rows: usize, cols: usize, phase: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((i as f64 + 1.0) * (0.37 + phase) + (j as f64 + 1.0) * (0.61 - 0.5 * phase)).sin()
            / (cols as f64).sqrt()
    })
}

fn parse(args: &[String], index: usize, name: &str) -> Result<usize, String> {
    args.get(index)
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| {
            format!(
                "usage: mpd_apply_llm_width ROWS D_IN D_OUT TERMS ACTIVE REPS (missing or invalid {name})"
            )
        })
}

fn main() -> Result<(), String> {
    let governor = MemoryGovernor::global();
    let args: Vec<String> = std::env::args().collect();
    let n_rows = parse(&args, 1, "ROWS")?.max(1);
    let input_dim = parse(&args, 2, "D_IN")?;
    let output_dim = parse(&args, 3, "D_OUT")?;
    let terms = parse(&args, 4, "TERMS")?;
    let active = parse(&args, 5, "ACTIVE")?.min(terms);
    let reps = parse(&args, 6, "REPS")?.max(1);
    println!(
        "shape rows={n_rows} d_in={input_dim} d_out={output_dim} terms={terms} active={active} reps={reps} threads={}",
        rayon::current_num_threads()
    );

    let started = Instant::now();
    let native = fixture(output_dim, input_dim, 0.11);
    let left = fixture(output_dim, terms, 0.29);
    let right = fixture(input_dim, terms, 0.47);
    let input = fixture(n_rows, input_dim, 0.83);
    let cotangent = fixture(n_rows, output_dim, 0.59);
    let edit = FactoredEdit::new(left, right).expect("factor shapes agree");
    let scales = Array1::from_shape_fn(terms, |k| if k < active { -1.0 } else { 0.0 });
    println!("fixtures_s={:.3}", started.elapsed().as_secs_f64());

    for (label, footprint) in [
        (
            "apply",
            EditFootprint::anchored_linear(n_rows, input_dim, output_dim, terms, active),
        ),
        (
            "contractions",
            EditFootprint::frobenius_contractions(n_rows, terms),
        ),
        (
            "cotangents",
            EditFootprint::factor_cotangents(n_rows, input_dim, output_dim, terms),
        ),
    ] {
        match footprint {
            Ok(footprint) => println!(
                "footprint {label} tile_rows={} result_mib={:.1} scratch_mib={:.1}",
                footprint.tile_rows,
                footprint.result_bytes as f64 / 1048576.0,
                footprint.scratch_bytes as f64 / 1048576.0
            ),
            Err(err) => println!("footprint {label} refused: {err}"),
        }
    }

    for rep in 0..reps {
        let started = Instant::now();
        let floor = native_linear(governor, native.view(), input.view());
        let native_s = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let edited = apply_anchored_linear(governor, native.view(), 1.0, edit.view(), scales.view(), input.view());
        let apply_s = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let contractions = edit_frobenius_contractions(governor, edit.view(), cotangent.view(), input.view());
        let contractions_s = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let cotangents = edit_factor_cotangents(governor, edit.view(), cotangent.view(), input.view());
        let cotangents_s = started.elapsed().as_secs_f64();
        println!(
            "rep={rep} native_s={native_s:.4} apply_s={apply_s:.4} edit_excess_s={:.4} contractions_s={contractions_s:.4} cotangents_s={cotangents_s:.4} ok={}/{}/{}/{}",
            apply_s - native_s,
            floor.is_ok(),
            edited.is_ok(),
            contractions.is_ok(),
            cotangents.is_ok()
        );
        if rep + 1 < reps {
            continue;
        }
        let edited = match edited {
            Ok(edited) => edited,
            Err(err) => {
                println!("apply refused: {err}");
                continue;
            }
        };
        // Row r, output o unfolds into d_in·(active + 1) products of at most four
        // factors: k = N + 2 over the computed absolute sum Â, and A ≤ Â/(1 − γ_k).
        let operations = input_dim * (active + 1) + 2;
        let growth = accumulation_growth(operations);
        let (left_factor, right_factor) = (edit.left(), edit.right());
        let mut worst_ratio = 0.0_f64;
        for row in [0, n_rows / 2, n_rows - 1] {
            let h = input.row(row);
            let absolute_h = h.mapv(f64::abs);
            let mut reference = native.dot(&h);
            let mut absolute = native.mapv(f64::abs).dot(&absolute_h);
            for k in 0..active {
                let (u, v) = (left_factor.column(k).to_owned(), right_factor.column(k));
                let projection = v.dot(&h);
                reference.scaled_add(scales[k] * projection, &u);
                absolute.scaled_add(
                    scales[k].abs() * v.mapv(f64::abs).dot(&absolute_h),
                    &u.mapv(f64::abs),
                );
            }
            for o in 0..output_dim {
                let band = 2.0 * accumulation_band(operations, absolute[o]) / (1.0 - growth);
                let error = (edited[[row, o]] - reference[o]).abs();
                worst_ratio = worst_ratio.max(error / band);
            }
        }
        println!("reference_check rows=3 worst_error_over_band={worst_ratio:.3e} within={}", worst_ratio <= 1.0);
    }
    Ok(())
}
