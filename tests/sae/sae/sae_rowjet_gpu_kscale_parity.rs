//! #1026 — GPU SAE reconstruction row-jet K-scale parity, PROVEN on the device.
//!
//! The headline #1026 ask is "scale K on the GPU and prove reconstruction
//! parity (CPU oracle == GPU)". The SAE reconstruction row-jet
//! (`gam_sae::gpu_kernels::sae_rowjet`) is the per-row order-≤2 derivative tower
//!
//! ```text
//!   ẑ_row,c = Σ_k ζ_k(ℓ) · decoded_{k,c},   first[a][c] = ∂ẑ_c/∂ℓ_a,
//!                                            second[a][b][c] = ∂²ẑ_c/∂ℓ_a∂ℓ_b
//! ```
//!
//! the arrow-Schur logdet consumer contracts into the Gauss-Newton curvature.
//! On the CPU it is the dense softmax gate Hessian — irreducibly `O(K³)` per
//! row; on the GPU the `n` rows are embarrassingly parallel and `exp` is
//! hardware, so the bottleneck collapses. The device kernel is a byte-faithful
//! NVRTC port of the host `Order2<K>` jet arithmetic, so device == CPU to
//! round-off.
//!
//! ## Why this test exists (the recurring failure mode it kills)
//!
//! The previously-shipped in-crate `device_matches_cpu_when_available` unit test
//! hedges: it runs `sae_row_jets_softmax` (which silently swallows any device
//! error and degrades to the CPU) and asserts "the contract on whichever path
//! ran". On a CUDA host where NVRTC silently declines to compile for this arch,
//! that test still PASSES on the CPU — a false green, exactly the #1026/#1551
//! "GPU 0%" failure mode.
//!
//! This test removes the hedge. On a CUDA host it drives the **fail-loud**
//! `sae_row_jets_softmax_required(.., GpuPolicy::Required)` entry point, which
//! returns `Err` instead of degrading. So:
//!
//!   * the device MUST compile the `sae_rowjet_softmax` kernel on THIS box's
//!     compute capability (sm_70 on the V100) and run it — a silent CPU fallback
//!     is a hard FAILURE here, not a skip-pass;
//!   * `SaeRowJetPath::Device` is asserted, proving the kernel ran on the GPU;
//!   * the device channels are locked to the CPU oracle to ≤ 1e-9 (measured
//!     ~1e-13 f64 on the V100) across the FULL supported K sweep K ∈ 1..=16 and
//!     a representative output width p — the K-scaling parity claim itself.
//!
//! On a CPU-only host (CI) the same `Required` call MUST return `Err` (the
//! fail-closed contract), and the `Auto` path MUST fall back to a CPU result
//! that is bit-identical to the oracle — proving the routing is reachable and
//! the CPU oracle is self-consistent.
//!
//! Uses only the public crate API. No `let _`, no `#[allow]`, no env vars, no
//! `#[cfg(feature = ...)]`.

use gam::terms::sae::gpu_kernels::sae_rowjet::{
    DEVICE_ROW_THRESHOLD, SaeRowJetPath, SaeSoftmaxRowInputs, sae_row_jets_cpu_softmax,
    sae_row_jets_softmax_required,
};

/// Deterministic logits/decoded fixture (no RNG crate, no `Math.random`-style
/// nondeterminism): the SAME shape the in-crate unit test uses, so the layout
/// is shared. `n` rows, `k` atoms, `p` output channels.
fn fixture(n: usize, k: usize, p: usize) -> Vec<SaeSoftmaxRowInputs> {
    (0..n)
        .map(|i| SaeSoftmaxRowInputs {
            logits: (0..k)
                .map(|j| 0.7 * ((i * 31 + j * 17) as f64 * 0.013).sin())
                .collect(),
            decoded: (0..k * p)
                .map(|t| ((i * 7 + t * 5) as f64 * 0.011).cos())
                .collect(),
        })
        .collect()
}

/// Max abs difference between two channel vectors of equal length.
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "channel length mismatch");
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

/// Whether a CUDA runtime is admitted on this Linux host.
#[cfg(target_os = "linux")]
fn cuda_available() -> bool {
    gam::gpu::GpuRuntime::global().is_some()
}

#[test]
fn sae_rowjet_gpu_kscale_parity() {
    // The inv_tau the production softmax assembly uses for the bottleneck shape.
    let inv_tau = 1.0 / 0.7;
    // One representative output width above the smallest meaningful p. p=16 is
    // the documented A100 calibration width and keeps the channel tensors small
    // enough that the full K sweep is seconds, not minutes.
    let p = 16;
    // A batch comfortably above the device launch break-even so the GPU path is
    // admitted (the +257 keeps it off the exact threshold boundary).
    let n = DEVICE_ROW_THRESHOLD + 257;

    #[cfg(target_os = "linux")]
    {
        if cuda_available() {
            // FAIL-LOUD device gate: under Required, every K in the supported
            // sweep MUST compile + run on the device and match the CPU oracle.
            let mut worst = 0.0_f64;
            for k in 1..=16usize {
                let rows = fixture(n, k, p);
                let cpu = sae_row_jets_cpu_softmax(&rows, k, p, inv_tau);
                let (dev, path) = sae_row_jets_softmax_required(
                    &rows,
                    k,
                    p,
                    inv_tau,
                    gam::gpu::GpuPolicy::Required,
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "[#1026] CUDA present but GpuPolicy::Required SAE row-jet \
                         FAILED at K={k}, p={p}, n={n}: {err}. A silent CPU \
                         fallback is a FAILURE — the device path did not engage \
                         (NVRTC compile/arch or launch fault on this GPU)."
                    )
                });
                assert_eq!(
                    path,
                    SaeRowJetPath::Device,
                    "[#1026] K={k}: Required succeeded but reported the CPU path — \
                     the device did not actually run (false GPU routing)."
                );
                assert_eq!(dev.n_rows, n, "K={k}: device row count");
                assert_eq!(dev.first.len(), n * k * p, "K={k}: device first len");
                assert_eq!(dev.second.len(), n * k * k * p, "K={k}: device second len");

                let d_first = max_abs_diff(&cpu.first, &dev.first);
                let d_second = max_abs_diff(&cpu.second, &dev.second);
                let kmax = d_first.max(d_second);
                worst = worst.max(kmax);
                assert!(
                    kmax <= 1e-9,
                    "[#1026] device vs CPU row-jet parity broke at K={k}, p={p}: \
                     first Δ={d_first:.3e}, second Δ={d_second:.3e} (> 1e-9)"
                );
            }
            eprintln!(
                "[#1026] GPU K-scale parity PROVEN on device for K∈1..=16, p={p}, \
                 n={n}: worst |device − CPU| = {worst:.3e} (≤ 1e-9)"
            );
            return;
        }
        eprintln!(
            "[#1026] Linux host without a CUDA runtime — asserting the \
             fail-closed Required contract + CPU oracle self-consistency."
        );
    }

    #[cfg(not(target_os = "linux"))]
    {
        eprintln!(
            "[#1026] non-Linux host — asserting the fail-closed Required \
             contract + CPU oracle self-consistency."
        );
    }

    // No device (CPU-only host or non-Linux): the Required contract MUST fail
    // closed, and the Auto/Off paths MUST produce a CPU result bit-identical to
    // the oracle. Run a small but representative K to keep CI cheap.
    let k = 8;
    let rows = fixture(n, k, p);
    let oracle = sae_row_jets_cpu_softmax(&rows, k, p, inv_tau);

    let required =
        sae_row_jets_softmax_required(&rows, k, p, inv_tau, gam::gpu::GpuPolicy::Required);
    assert!(
        required.is_err(),
        "[#1026] no device present, yet GpuPolicy::Required returned Ok — the \
         fail-closed contract was violated (a silent CPU fallback reported as a \
         device success is the exact #1551 false-routing bug)."
    );

    for mode in [gam::gpu::GpuPolicy::Auto, gam::gpu::GpuPolicy::Off] {
        let (channels, path) = sae_row_jets_softmax_required(&rows, k, p, inv_tau, mode)
            .expect("[#1026] Auto/Off must never error on a device-absent host");
        assert_eq!(
            path,
            SaeRowJetPath::Cpu,
            "[#1026] {mode:?}: device absent, so the CPU path must be reported"
        );
        assert_eq!(
            max_abs_diff(&oracle.first, &channels.first),
            0.0,
            "[#1026] {mode:?}: CPU first channel not bit-identical to the oracle"
        );
        assert_eq!(
            max_abs_diff(&oracle.second, &channels.second),
            0.0,
            "[#1026] {mode:?}: CPU second channel not bit-identical to the oracle"
        );
    }
}
