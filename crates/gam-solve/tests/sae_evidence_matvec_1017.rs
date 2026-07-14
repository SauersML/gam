//! #1017 device-resident evidence-matvec gates (integration test).
//!
//! Runs the DETERMINISTIC framed reduced-Schur `S·v` — the operator that feeds
//! the SLQ/surrogate `log|S|` evidence lane — on a real CUDA device and checks
//! (1) it matches the bit-for-bit CPU oracle `sae_framed_schur_matvec_cpu` to
//! ≤1e-9, and (2) it is run-to-run bit-identical (the determinism contract the
//! shared atomic step-PCG matvec cannot satisfy). A separate utilization test
//! drives many applies through the resident builder (the SLQ apply loop) on a
//! large fixture so GPU utilization can be sampled during the run.
//!
//! This lives in `tests/` (not a `#[cfg(test)]` unit module) so it compiles
//! against only the `gam-solve` library and is insulated from unrelated
//! unit-test churn in the shared lib-test binary. It uses the public API only.
//! Off-device (CPU CI / non-Linux) the gates skip cleanly.

#![cfg(target_os = "linux")]

use ndarray::{Array1, Array2};

use gam_solve::arrow_schur::{
    ArrowSchurSystem, DeviceSaeFrameData, DeviceSaePcgData, DeviceSaeSmoothBlock,
    FactoredFrameGBlock,
};
use gam_solve::gpu_kernels::arrow_schur::{
    build_framed_resident_evidence_matvec, framed_reduced_schur_det_once_on_device,
    sae_framed_schur_matvec_cpu,
};

// Device presence is detected via the one-shot probe itself: for a well-formed
// framed fixture it returns `Err(Unavailable)` ONLY when CUDA is genuinely
// absent (it ignores the offload floor), and `Ok(..)` only after running on the
// GPU. So `Ok` ⇒ device present, `Err` ⇒ no device (clean off-device skip).
fn device_present(sys: &ArrowSchurSystem, data: &DeviceSaePcgData, rt: f64, rb: f64) -> bool {
    let x = Array1::<f64>::from_elem(data.beta_dim, 1.0);
    framed_reduced_schur_det_once_on_device(sys, data, rt, rb, &x).is_ok()
}

/// Build a framed SAE fixture (mix of framed `r<p` and identity-ride `r==p`
/// atoms, off-diagonal cross blocks, dense per-row `H_tβ`, `n` rows). `sys.hbb`
/// is zero — for a matrix-free framed system the β-Hessian lives entirely in the
/// `DeviceSaePcgData` penalty operator, which is exactly what both the device
/// matvec and the CPU oracle read, so the dense assembly is neither needed nor
/// built. Returns `(sys, data, ρ_t, ρ_β)`.
fn build_framed_fixture(
    n: usize,
    n_atoms: usize,
    p: usize,
    seed: u64,
) -> (ArrowSchurSystem, std::sync::Arc<DeviceSaePcgData>, f64, f64) {
    let ranks: Vec<usize> = (0..n_atoms)
        .map(|k| if k % 4 == 0 { p } else { 2 + (k % 3) })
        .collect();
    let basis_sizes: Vec<usize> = (0..n_atoms).map(|k| 3 + (k % 2)).collect();
    let mut border_offsets = Vec::with_capacity(n_atoms);
    let mut acc = 0usize;
    for k in 0..n_atoms {
        border_offsets.push(acc);
        acc += basis_sizes[k] * ranks[k];
    }
    let border_dim = acc;

    let mut state = seed;
    let mut sample = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
    };
    let mut frames: Vec<Array2<f64>> = Vec::new();
    for k in 0..n_atoms {
        let r = ranks[k];
        let mut u = Array2::<f64>::zeros((p, r));
        for i in 0..p {
            for j in 0..r {
                u[[i, j]] = if r == p && i == j {
                    1.0
                } else if r == p {
                    0.0
                } else {
                    sample()
                };
            }
        }
        frames.push(u);
    }
    let w_of = |i: usize, j: usize| {
        let (ui, uj) = (&frames[i], &frames[j]);
        let (ri, rj) = (ranks[i], ranks[j]);
        let mut w = Array2::<f64>::zeros((ri, rj));
        for a in 0..ri {
            for b in 0..rj {
                let mut s = 0.0;
                for c in 0..p {
                    s += ui[[c, a]] * uj[[c, b]];
                }
                w[[a, b]] = s;
            }
        }
        w
    };
    let mut pairs: Vec<(usize, usize)> = (0..n_atoms).map(|k| (k, k)).collect();
    for k in 0..n_atoms.saturating_sub(1) {
        if k % 3 == 0 {
            pairs.push((k, k + 1));
            pairs.push((k + 1, k));
        }
    }
    let mut frame_blocks = Vec::new();
    for &(i, j) in &pairs {
        let (mi, mj) = (basis_sizes[i], basis_sizes[j]);
        let mut g = Array2::<f64>::zeros((mi, mj));
        for r in 0..mi {
            for c in 0..mj {
                g[[r, c]] = 0.25 * sample();
            }
        }
        if i == j {
            for r in 0..mi.min(mj) {
                g[[r, r]] += mi as f64 + 2.0;
            }
        }
        frame_blocks.push(FactoredFrameGBlock {
            atom_i: i,
            atom_j: j,
            g,
            w: w_of(i, j),
        });
    }
    let mut smooth_blocks = Vec::new();
    let mut smooth_ranks = Vec::new();
    for k in 0..n_atoms {
        let m = basis_sizes[k];
        let mut a = Array2::<f64>::zeros((m, m));
        for r in 0..m {
            for c in 0..m {
                a[[r, c]] = 0.2 * sample();
            }
        }
        let mut s = a.t().dot(&a);
        for r in 0..m {
            s[[r, r]] += 1.0;
        }
        smooth_blocks.push(DeviceSaeSmoothBlock {
            global_offset: border_offsets[k],
            factor_a: s,
        });
        smooth_ranks.push(ranks[k]);
    }
    let q = 4usize;
    let mut sys = ArrowSchurSystem::new(n, q, border_dim);
    let mut row_htbeta = Vec::new();
    for i in 0..n {
        let mut a = Array2::<f64>::zeros((q, q));
        for r in 0..q {
            for c in 0..q {
                a[[r, c]] = sample();
            }
        }
        let mut htt = a.t().dot(&a);
        for r in 0..q {
            htt[[r, r]] += q as f64 + 1.0;
        }
        sys.rows[i].htt = htt;
        let mut slab = vec![0.0_f64; q * border_dim];
        for c in 0..q {
            for col in 0..border_dim {
                // Small cross entries so the reduced-Schur subtraction does not
                // overwhelm the PD penalty (keeps S well conditioned).
                let v = 0.02 * sample();
                slab[c * border_dim + col] = v;
                sys.rows[i].htbeta[[c, col]] = v;
            }
        }
        row_htbeta.push(slab);
    }
    // Matrix-free framed system: β-Hessian is the penalty operator, so hbb = 0.
    sys.hbb = Array2::<f64>::zeros((border_dim, border_dim));
    let data = DeviceSaePcgData {
        p,
        beta_dim: border_dim,
        a_phi: std::sync::Arc::from(Vec::new().into_boxed_slice()),
        local_jac: std::sync::Arc::from(Vec::new().into_boxed_slice()),
        smooth_blocks,
        sparse_g_blocks: Vec::new(),
        frame: Some(DeviceSaeFrameData {
            ranks,
            basis_sizes,
            border_offsets,
            frame_blocks,
            smooth_ranks,
            row_htbeta,
        }),
    };
    // Attach the resident operands to the system: the builder path
    // (`build_framed_resident_evidence_matvec`) reads `sys.device_sae_pcg`, while
    // the one-shot probe / CPU oracle take `data` explicitly. Keep an Arc handle
    // for those direct calls.
    let data = std::sync::Arc::new(data);
    sys.set_device_sae_pcg_allocation(data.clone());
    (sys, data, 1e-7, 1e-6)
}

fn cpu_oracle(
    sys: &ArrowSchurSystem,
    data: &DeviceSaePcgData,
    ridge_t: f64,
    ridge_beta: f64,
    x: &Array1<f64>,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; data.beta_dim];
    sae_framed_schur_matvec_cpu(
        sys,
        data,
        ridge_t,
        ridge_beta,
        x.as_slice().unwrap(),
        &mut out,
    )
    .expect("cpu oracle matvec");
    out
}

/// (1) parity vs the CPU oracle ≤1e-9 and (2) run-to-run bit-identical, via the
/// one-shot device probe. Fails loud if CUDA is present but the seam declines.
#[test]
fn evidence_matvec_deterministic_and_matches_cpu() {
    let (sys, data, ridge_t, ridge_beta) = build_framed_fixture(400, 8, 6, 0x1017_5111_0901_4c0d);
    let border_dim = data.beta_dim;
    let mut probes: Vec<Array1<f64>> = Vec::new();
    probes.push(Array1::from_shape_fn(border_dim, |a| {
        ((a as f64 + 1.0) * 0.41).cos()
    }));
    for axis in [0usize, border_dim / 2, border_dim - 1] {
        let mut e = Array1::<f64>::zeros(border_dim);
        e[axis] = 1.0;
        probes.push(e);
    }
    for (pi, x) in probes.iter().enumerate() {
        let dev1 =
            match framed_reduced_schur_det_once_on_device(&sys, &data, ridge_t, ridge_beta, x) {
                Ok(out) => out,
                // For this well-formed fixture the probe declines only when CUDA is
                // absent — a clean off-device skip.
                Err(_) => return,
            };
        let dev2 = framed_reduced_schur_det_once_on_device(&sys, &data, ridge_t, ridge_beta, x)
            .expect("second deterministic matvec");
        for a in 0..border_dim {
            assert_eq!(
                dev1[a].to_bits(),
                dev2[a].to_bits(),
                "#1017 determinism: probe {pi} coord {a} run-to-run differs: {:e} vs {:e}",
                dev1[a],
                dev2[a]
            );
        }
        let cpu = cpu_oracle(&sys, &data, ridge_t, ridge_beta, x);
        let scale = cpu.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
        for a in 0..border_dim {
            let rel = (dev1[a] - cpu[a]).abs() / scale;
            assert!(
                rel <= 1e-9,
                "#1017 parity: probe {pi} coord {a}: device={:e} cpu={:e} rel={rel:e} (>1e-9)",
                dev1[a],
                cpu[a],
            );
        }
    }
}

/// Drive the SLQ apply loop through the PRODUCTION resident builder on a large
/// fixture (upload once, apply many). Verifies parity on the first apply and
/// apply-to-apply determinism, then runs `SCHUR_SLQ_LOGDET_PROBES ×
/// LANCZOS_STEPS` = 2048 applies and prints wall-time + achieved f64 throughput
/// so GPU utilization can be sampled/attributed during the run.
#[test]
fn evidence_matvec_utilization_loop() {
    // Large enough that the O(n·d·k) device reduced-Schur term is the per-apply
    // cost; ~n=4000 rows, border ≈ Σ M_k·r_k with 64 atoms.
    let n = 4000usize;
    let n_atoms = 64usize;
    let p = 8usize;
    let (sys, data, ridge_t, ridge_beta) =
        build_framed_fixture(n, n_atoms, p, 0x1017_0000_beef_0007);
    let border_dim = data.beta_dim;
    let q = 4usize;
    let budget = 32usize * 64usize; // SCHUR_SLQ_LOGDET_PROBES × LANCZOS_STEPS

    let matvec = match build_framed_resident_evidence_matvec(&sys, ridge_t, ridge_beta, budget) {
        Ok(Some(mv)) => mv,
        Ok(None) => {
            // Distinguish "no device" (clean skip) from "device present but the
            // builder declined for a floor-clearing framed system" (a real bug):
            // the probe runs on any device regardless of the offload floor.
            assert!(
                !device_present(&sys, &data, ridge_t, ridge_beta),
                "#1017: CUDA present but resident evidence matvec builder declined for a \
                 framed system clearing the offload floor (n={n}, k={border_dim})"
            );
            return;
        }
        Err(failure) => {
            panic!("#1017: resident evidence matvec build faulted: {failure:?}")
        }
    };
    let apply = &matvec;

    let x = Array1::from_shape_fn(border_dim, |a| ((a as f64 + 1.0) * 0.29).sin());
    let mut out1 = Array1::<f64>::zeros(border_dim);
    let mut out2 = Array1::<f64>::zeros(border_dim);
    apply(&x, &mut out1);
    apply(&x, &mut out2);
    for a in 0..border_dim {
        assert_eq!(
            out1[a].to_bits(),
            out2[a].to_bits(),
            "#1017 resident determinism: coord {a} apply-to-apply differs"
        );
    }
    let cpu = cpu_oracle(&sys, &data, ridge_t, ridge_beta, &x);
    let scale = cpu.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
    let mut worst = 0.0_f64;
    for a in 0..border_dim {
        worst = worst.max((out1[a] - cpu[a]).abs() / scale);
    }
    assert!(worst <= 1e-9, "#1017 resident parity worst rel = {worst:e}");

    // The SLQ apply loop: 2048 device applies against the resident operands.
    let start = std::time::Instant::now();
    let mut acc = 0.0_f64;
    for it in 0..budget {
        let mut y = Array1::<f64>::zeros(border_dim);
        let xi = Array1::from_shape_fn(border_dim, |a| x[a] + (it as f64) * 1e-9);
        apply(&xi, &mut y);
        acc += y[it % border_dim];
    }
    let elapsed = start.elapsed();
    // Per-apply reduced-Schur f64 work ≈ 2·Σ_i (q·k)  [apply_h: k·q, scatter: k·q].
    let per_apply_flop = 2.0 * (n as f64) * (q as f64) * (border_dim as f64) * 2.0;
    let total_flop = per_apply_flop * (budget as f64);
    let secs = elapsed.as_secs_f64();
    eprintln!(
        "#1017-UTIL n={n} k={border_dim} q={q} applies={budget} wall={:.3}s \
         per_apply={:.3}ms gflop_total={:.2} gflops={:.2} (acc={acc:.3e})",
        secs,
        secs / (budget as f64) * 1e3,
        total_flop / 1e9,
        total_flop / 1e9 / secs.max(1e-9),
    );
}
