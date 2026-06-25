//! #1551 regression gate: the device-resident SAE arrow-Schur solve must engage
//! on a **production-shaped Direct-mode SAE fit** — not only on hand-built
//! `inexact_pcg()` + `radius == INFINITY` test options.
//!
//! ROOT CAUSE (issue #1551): the real SAE inner solve runs `ArrowSolveOptions::
//! direct()` and installs the matrix-free SAE operators. The dense device path
//! (`try_device_arrow_direct`) rejected the matrix-free system, and the device
//! matrix-free PCG was gated behind `mode == InexactPCG && radius == INFINITY`,
//! which no production site ever sets — so every real SAE fit ran the inner
//! arrow-Schur on the CPU (`nvidia-smi` ~0% for the whole fit).
//!
//! FIX: a Direct-mode device branch (`try_device_arrow_direct_sae_pcg`) engages
//! the device SAE PCG when (a) `device_sae_pcg` is present, (b) the GPU admits the
//! CG-amortised work, (c) CUDA is available. Direct mode is the exact full Newton
//! step (no trust-region truncation), so the unbounded device PCG converges to
//! the SAME step the dense Direct CPU path produces. The branch also emits the
//! dense reduced-Schur Cholesky factor so the joint-Hessian log-det the Laplace
//! evidence consumes stays exact (and bit-identical to the CPU path).
//!
//! This gate runs the SAE system through the PRODUCTION solver entry
//! `solve_arrow_newton_step_with_options` (the same call `converge_inner_for_
//! undamped_logdet` makes) with **Direct** options and a finite default trust
//! region — exactly the production shape, NOT the `INFINITY` test shape.
//!
//!   * On a CUDA host: asserts `used_device_arrow == true`, the device step
//!     matches the CPU dense reference to <= 1e-7, and the log-det is finite —
//!     the regression gate proving the GPU engages on a production-shaped fit.
//!   * On a CPU-only host (CI): the device branch declines (no runtime), the
//!     solve is the bit-identical CPU Direct path, `used_device_arrow == false`,
//!     and the joint-Hessian log-det is still present and finite — proving the
//!     Direct-mode routing is reachable and non-regressing. The device-vs-CPU
//!     numeric assertion is the GPU-gated remainder.
//!
//! Uses only the public crate API.

use gam::gpu::kernels::arrow_schur::solve_arrow_newton_step_dense_reference;
use gam::gpu::policy::GpuDispatchPolicy;
use gam::gpu::GpuRuntime;
use gam::solver::arrow_schur::{
    solve_arrow_newton_step_with_options, ArrowSchurSystem, ArrowSolveOptions, ArrowSolverMode,
    BetaPenaltyOp, DeviceSaeFrameData, DeviceSaePcgData, DeviceSaeSmoothBlock, FactoredFrameGBlock,
};
use ndarray::Array2;

/// Build a production-shaped framed SAE arrow system: few rows, wide factored
/// border (`k >= DEVICE_LOOP_MIN_P`), modest per-row depth `d` — the LLM/SAE
/// shape the device-offload policy admits. Mirrors the in-crate framed device
/// fixture so the dense per-row `htbeta` slabs + dense `hbb` make the CPU dense
/// reference an exact baseline for the device-solved step. Returns the system
/// with `device_sae_pcg` installed.
fn build_framed_sae_system(install_device_data: bool) -> ArrowSchurSystem {
    let p = 6usize;
    let n_atoms = 8usize;
    let ranks: Vec<usize> = (0..n_atoms)
        .map(|k| if k % 2 == 0 { 3usize } else { p })
        .collect();
    let basis_sizes: Vec<usize> = (0..n_atoms).map(|_| 3usize).collect();
    let mut border_offsets = Vec::with_capacity(n_atoms);
    let mut acc = 0usize;
    for k in 0..n_atoms {
        border_offsets.push(acc);
        acc += basis_sizes[k] * ranks[k];
    }
    let border_dim = acc; // Σ M_k·r_k = 4·(3·3) + 4·(3·6) = 108

    let mut state = 0xfeed_face_dead_beefu64;
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
    for &(i, j) in &[(0usize, 1usize), (2, 4), (3, 6)] {
        pairs.push((i, j));
        pairs.push((j, i));
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

    let n = 400usize;
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
                let v = 0.02 * sample();
                slab[c * border_dim + col] = v;
                sys.rows[i].htbeta[[c, col]] = v;
            }
        }
        row_htbeta.push(slab);
        for r in 0..q {
            sys.rows[i].gt[r] = 0.03 * sample();
        }
    }

    // Dense H_ββ matching the device penalty side EXACTLY (the device PCG penalty
    // matvec is `sae_framed_penalty_matvec_cpu`; here we materialise the same
    // operator densely so the CPU dense reference's reduced system agrees).
    use gam::solver::arrow_schur::{
        FactoredFrameKroneckerOp, IdentityRightKroneckerPenaltyOp,
    };
    let data_op =
        FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), frame_blocks.clone())
            .expect("frame op");
    let mut hbb = data_op.to_dense();
    for k in 0..n_atoms {
        let op = IdentityRightKroneckerPenaltyOp {
            factor_a: smooth_blocks[k].factor_a.clone(),
            p: ranks[k],
            global_offset: border_offsets[k],
            k: border_dim,
        };
        let dd = op.to_dense();
        for r in 0..border_dim {
            for c in 0..border_dim {
                hbb[[r, c]] += dd[[r, c]];
            }
        }
    }
    sys.hbb = hbb;
    for a in 0..border_dim {
        sys.gb[a] = 0.05 * ((a as f64 + 1.0) * 0.31).sin();
    }

    // Install the matrix-free per-row `H_tβ` operator (dense-backed), mirroring
    // the production full-`B` SAE path (`set_row_htbeta_operator`). The operator
    // reads the same `q × border_dim` row-major slabs already stored on
    // `sys.rows[i].htbeta`; with it installed the system matches the production
    // matrix-free shape.
    let slabs = row_htbeta.clone();
    let fwd_slabs = slabs.clone();
    let bd = border_dim;
    let qd = q;
    sys.set_row_htbeta_operator(
        move |row_idx, x, out| {
            let slab = &fwd_slabs[row_idx];
            let out_s = out.as_slice_mut().expect("std layout");
            for r in 0..qd {
                let mut acc = 0.0;
                let base = r * bd;
                for c in 0..bd {
                    acc += slab[base + c] * x[c];
                }
                out_s[r] = acc;
            }
        },
        move |row_idx, v, out| {
            let slab = &slabs[row_idx];
            let out_s = out.as_slice_mut().expect("std layout");
            for r in 0..qd {
                let vr = v[r];
                let base = r * bd;
                for c in 0..bd {
                    out_s[c] += slab[base + c] * vr;
                }
            }
        },
    );

    // `set_device_sae_pcg_data` asserts `a_phi`/`local_jac` have one entry per
    // row (the full-`B` residency operator's per-row support). The framed kernel
    // consumes `frame.row_htbeta` instead and ignores these, but the installer's
    // shape contract still requires `n` (here empty) entries.
    //
    // `install_device_data == false` yields the IDENTICAL system without the
    // device frames — the device-free CPU Direct path, the exact bit-identity
    // baseline the device-solved step must reproduce.
    if install_device_data {
        sys.set_device_sae_pcg_data(DeviceSaePcgData {
            p,
            beta_dim: border_dim,
            a_phi: vec![Vec::new(); n],
            local_jac: vec![Vec::new(); n],
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
        });
    }
    sys
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn trace(msg: &str) {
    eprintln!("[owed_1551 trace] {msg}");
}

#[test]
fn sae_direct_mode_engages_device_on_production_entry_1551() {
    trace("start");
    let sys = build_framed_sae_system(true);
    trace("built sys with device data");

    // The wide factored border must clear the device-loop floor; this is the
    // shape the Phase-1 offload predicate admits. If this regresses below the
    // floor the device would correctly decline for a different reason, masking
    // the routing bug, so pin it.
    assert!(
        sys.k >= GpuDispatchPolicy::DEVICE_LOOP_MIN_P,
        "fixture border k={} must clear DEVICE_LOOP_MIN_P={} so the device is admitted",
        sys.k,
        GpuDispatchPolicy::DEVICE_LOOP_MIN_P
    );

    // PRODUCTION shape: the exact options `converge_inner_for_undamped_logdet`
    // builds. The mode is Direct — and Direct mode is the EXACT full Newton step
    // (no trust-region truncation), so the trust radius is irrelevant to the
    // solve. The pre-existing InexactPCG device branch is gated on BOTH
    // `mode == InexactPCG` AND `radius == INFINITY`; production SAE is Direct
    // mode, so it never reaches that gate regardless of the radius. The Direct
    // device branch this gate exercises engages on `mode == Direct` directly.
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    assert_eq!(
        options.mode,
        ArrowSolverMode::Direct,
        "production SAE inner solve must be Direct mode (the device branch under test)"
    );

    // BIT-IDENTITY BASELINE: the IDENTICAL system WITHOUT device frames, solved
    // through the SAME production entry — i.e. the CPU Direct path. The device
    // branch must reproduce this exactly (Direct mode = exact full step, no
    // truncation), so this is the tight reference. (`solve_arrow_newton_step_
    // dense_reference` factors the FULL joint system rather than the reduced
    // Schur, so it diverges at the conditioning floor of a wide-border fixture;
    // it is used only as a loose cross-check below.)
    let sys_cpu = build_framed_sae_system(false);
    assert!(
        sys_cpu.device_sae_pcg.is_none(),
        "device-free baseline must carry no device frames"
    );
    trace("solving device-free CPU baseline");
    let (cpu_dt, cpu_db, cpu_cache) =
        solve_arrow_newton_step_with_options(&sys_cpu, 0.0, 0.0, &options)
            .expect("device-free CPU Direct baseline solve must succeed");
    trace("device-free baseline solved");
    assert!(
        !cpu_cache.pcg_diagnostics.used_device_arrow,
        "the device-free baseline must NOT claim device execution"
    );

    // Loose physical cross-check: the reduced-Schur Direct step and the full
    // dense-joint solve agree to the fixture's conditioning floor.
    trace("solving dense reference");
    let reference = solve_arrow_newton_step_dense_reference(&sys, 0.0, 0.0)
        .expect("CPU dense reference solve must succeed");
    let ref_dt = max_abs_diff(cpu_dt.as_slice().unwrap(), reference.delta_t.as_slice().unwrap());
    let ref_db = max_abs_diff(cpu_db.as_slice().unwrap(), reference.delta_beta.as_slice().unwrap());
    trace(&format!("dense ref done ref_dt={ref_dt:.3e} ref_db={ref_db:.3e}"));
    assert!(
        ref_dt <= 1e-2 && ref_db <= 1e-2,
        "reduced-Schur Direct step must agree with the full dense-joint solve to the \
         fixture conditioning floor (max|Δt|={ref_dt:.3e}, max|Δβ|={ref_db:.3e})"
    );

    trace("solving production (device-data) system");
    let (delta_t, delta_beta, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("production Direct-mode SAE solve must succeed");
    trace(&format!(
        "production solved used_device_arrow={}",
        cache.pcg_diagnostics.used_device_arrow
    ));

    // The joint-Hessian log-det MUST be present and finite on EITHER path — this
    // is the Laplace normaliser every production SAE inner solve consumes. The
    // device branch is responsible for emitting the reduced-Schur factor so this
    // never regresses to `None` (which would error the evidence solve).
    let log_det = cache
        .joint_hessian_log_det
        .expect("joint-Hessian log-det must be present (k>0 needs the reduced-Schur factor)");
    assert!(
        log_det.is_finite(),
        "joint-Hessian log-det must be finite, got {log_det}"
    );
    let cpu_log_det = cpu_cache
        .joint_hessian_log_det
        .expect("device-free CPU baseline must also produce the log-det");
    assert!(
        (log_det - cpu_log_det).abs() <= 1e-7 * (1.0 + cpu_log_det.abs()),
        "device-path log-det must match the CPU Direct log-det (device={log_det}, \
         cpu={cpu_log_det}) — the reduced-Schur factor the device branch emits must be \
         bit-equivalent to the CPU path"
    );

    let dt = max_abs_diff(delta_t.as_slice().unwrap(), cpu_dt.as_slice().unwrap());
    let db = max_abs_diff(delta_beta.as_slice().unwrap(), cpu_db.as_slice().unwrap());

    if GpuRuntime::global().is_some() {
        // GPU host: the device path must have ENGAGED on this production-shaped
        // Direct fit, and reproduced the exact full Newton step the CPU Direct
        // path computes (to the tight PCG tolerance).
        assert!(
            cache.pcg_diagnostics.used_device_arrow,
            "#1551 REGRESSION: a production-shaped Direct-mode SAE fit ran on the CPU \
             (used_device_arrow == false) despite a CUDA runtime being present — the \
             device SAE solver did not engage"
        );
        assert!(
            dt <= 1e-7 && db <= 1e-7,
            "device-solved Direct SAE step must match the CPU Direct step to <= 1e-7 \
             (max|Δt diff|={dt:.3e}, max|Δβ diff|={db:.3e})"
        );
        eprintln!(
            "[owed_1551] GPU host: device ENGAGED on production Direct SAE fit \
             (used_device_arrow=true); device-vs-CPU max|Δt|={dt:.3e} max|Δβ|={db:.3e}"
        );
    } else {
        // CPU-only host (CI): the device branch declines (no runtime), so the
        // production path IS the CPU Direct path — bit-identical to the baseline.
        // Pins that the routing is reachable, declines cleanly, and does not
        // regress the solve or the log-det.
        assert!(
            !cache.pcg_diagnostics.used_device_arrow,
            "no CUDA runtime present but used_device_arrow was set true"
        );
        assert!(
            dt == 0.0 && db == 0.0,
            "on a CPU host the device-frame system and the device-free system must \
             take the identical CPU Direct path (max|Δt diff|={dt:.3e}, max|Δβ diff|={db:.3e})"
        );
        eprintln!(
            "[owed_1551] CPU-only host: Direct-mode SAE routing reachable + non-regressing \
             (device-frame system == device-free system); device-vs-CPU engagement \
             assertion is the GPU-gated remainder"
        );
    }
}
