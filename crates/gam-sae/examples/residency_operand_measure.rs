//! #1017/#2230 residency measurement (host-pure, no GPU / no wheel / no logger).
//!
//! Assembles a wide-border SAE-manifold Arrow-Schur system via the SAME public
//! path a real fit takes (`SaeManifoldTerm::assemble_arrow_schur_scaled`, the
//! deferred/factored β-tier that `tests_device_engage_1783` exercises at small
//! scale), then prints `DeviceSaePcgData::operand_byte_report()` — the exact
//! per-category host→device operand bytes the matrix-free PCG re-uploads on EVERY
//! solve (hence every LM ridge-ladder trial), and which a base-resident frame
//! would remove. Everything here is host arithmetic, so it produces deterministic
//! increment-1 numbers today; the wheel-driven a100 run later confirms on real
//! data with real ladder counts.
//!
//! It also prints `reduced_schur_matvec_should_offload`'s verdict across a border
//! sweep INCLUDING the #2230 border (k≈21.5k) and a small K=2000-scale border, so
//! the "does the matrix-free device PCG even engage" question is answered in the
//! same run (the K=2000-vs-#2230 offload crossover).
//!
//! Defaults keep the assembled shape inside ~2 GiB host RAM (moderate `n`, real
//! wide `k`); the per-category bytes scale ~linearly in `n`, so the harness also
//! prints the linear extrapolation to the #2230 row count. Raise `--rows` /
//! `--harmonics` on a high-RAM box to measure closer to the true shape.
//!
//! Usage:
//!   cargo run --release -p gam-sae --example residency_operand_measure -- \
//!     [--rows N] [--harmonics H] [--p P] [--pcg-iters I]

use std::sync::Arc;

use gam_sae::gpu::GpuDispatchPolicy;
use gam_sae::manifold::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};

struct Args {
    rows: usize,
    harmonics: usize,
    p: usize,
    pcg_iters: usize,
}

fn parse_args() -> Result<Args, String> {
    // Defaults: moderate n, a real WIDE border. harmonics=500 -> basis M=1001;
    // a rank-`p` decoder gives frame rank r≈p, so the factored border is k≈M·p
    // (with p=16 → k≈16k, comfortably past the 4096 deferred-tier threshold and
    // the DEVICE_LOOP_MIN_P floor, and near the #2230 k≈21.5k). row_htbeta at
    // n=1024 is ~131 MiB, so the whole assembly fits ~2 GiB.
    let mut args = Args {
        rows: 1024,
        harmonics: 500,
        p: 16,
        pcg_iters: 40,
    };
    let mut it = std::env::args().skip(1);
    while let Some(key) = it.next() {
        let val = it
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        let n: usize = val
            .parse()
            .map_err(|_| format!("{key} expects a usize, got {val}"))?;
        match key.as_str() {
            "--rows" => args.rows = n,
            "--harmonics" => args.harmonics = n,
            "--p" => args.p = n,
            "--pcg-iters" => args.pcg_iters = n,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if args.rows == 0 || args.harmonics == 0 || args.p < 12 {
        return Err("rows>0, harmonics>0, p>=12 (SAE frame auto-activation floor)".to_string());
    }
    Ok(args)
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let n = args.rows;
    let p = args.p;

    // ---- Build the single wide-basis periodic circle atom (public path) -------
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let basis_size = 2 * args.harmonics + 1;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(basis_size)?);
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let m = phi.ncols();
    // Low-rank-in-p structured decoder (rank ≈ p): the cold decoder profiles out a
    // Grassmann frame of rank r≈p, so the factored border is k≈M·r — the wide
    // matrix-free border the #2230 fit runs on.
    let decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        (1.0 / (1.0 + b as f64)) * (((b as f64 + 1.0) * (c as f64 + 1.0)).cos())
    });
    let target = phi.dot(&decoder);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "residency_measure",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )?
    .with_basis_second_jet(evaluator.clone());
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )?;
    let mut term = SaeManifoldTerm::new(vec![atom], assignment)?;
    let activated = term.auto_activate_decoder_frames()?;
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);

    println!("=== #1017/#2230 residency operand measurement (host-pure) ===");
    println!(
        "shape: n={n} basis_M={m} p={p}  frames_activated={activated} frames_active={}",
        term.frames_active()
    );

    // ---- Assemble via the public deferred/factored β-tier -----------------------
    let sys = term.assemble_arrow_schur_scaled(target.view(), &rho, None, 1.0)?;
    let k = sys.k;
    let d = sys.d;
    println!(
        "assembled Arrow-Schur: border k={k}  per-row d={d}  rows={}",
        sys.rows.len()
    );

    // ---- The measurement: per-solve operand upload by category ------------------
    match sys.device_sae_pcg.as_ref() {
        Some(data) => {
            let report = data.operand_byte_report();
            println!("{report}");
            // Linear-in-n extrapolation to the #2230 row count (categories are all
            // O(n·…); k is already the real wide border here).
            const N_2230: usize = 60_000;
            if n > 0 {
                let scale = N_2230 as f64 / n as f64;
                println!(
                    "extrapolated per-solve upload at n={N_2230} (×{scale:.1}): {:.1} MiB \
                     (row_htbeta {:.1} MiB, a_phi {:.1} MiB, local_jac {:.1} MiB)",
                    report.total_bytes as f64 * scale / (1024.0 * 1024.0),
                    report.row_htbeta_bytes as f64 * scale / (1024.0 * 1024.0),
                    report.a_phi_bytes as f64 * scale / (1024.0 * 1024.0),
                    report.local_jac_bytes as f64 * scale / (1024.0 * 1024.0),
                );
            }
        }
        None => {
            println!(
                "device_sae_pcg = None: this shape did NOT install matrix-free device \
                 data (full-B / dense β-tier, or frames did not engage). The matrix-free \
                 residency frame does not apply here — rerun with p>=12 and k>4096 so the \
                 deferred factored tier engages."
            );
        }
    }

    // ---- Offload verdict sweep: the K=2000-vs-#2230 engagement question ---------
    let policy = GpuDispatchPolicy::default();
    println!(
        "--- reduced_schur_matvec_should_offload(n, k, d={d}, cg_iters={}) ---",
        args.pcg_iters
    );
    println!(
        "    DEVICE_LOOP_MIN_P floor = {}",
        GpuDispatchPolicy::DEVICE_LOOP_MIN_P
    );
    // Sweep the border from a K=2000-scale value up to the #2230 border, at both
    // the assembled n and the #2230 n, so the crossover is explicit.
    for &probe_n in &[n, 60_000usize] {
        for &probe_k in &[64usize, 512, 2_048, 8_192, 21_504, k] {
            let verdict = policy.reduced_schur_matvec_should_offload(
                probe_n,
                probe_k,
                d.max(1),
                args.pcg_iters,
            );
            println!(
                "    n={probe_n:>6} k={probe_k:>6} -> offload={}",
                if verdict { "YES" } else { "no" }
            );
        }
    }
    println!(
        "verdict at THIS assembled shape (n={n}, k={k}): offload={}",
        policy.reduced_schur_matvec_should_offload(n, k, d.max(1), args.pcg_iters)
    );
    println!("=== done ===");
    Ok(())
}
