//! Owed-work regression gate for #1017 Phase 0 (process-parallel candidates).
//!
//! The SAE driver fits several INDEPENDENT candidates — topology candidates,
//! layer-ladder charts, checkpoint trajectories — but historically walked them
//! sequentially, leaving a multi-core box idle. Phase 0 is the zero-engine-code
//! win: fan the independent candidate fits out over rayon at the driver level,
//! capped so the heavy per-candidate solves don't oversubscribe. (The separate
//! device-residency leg of #1017 — the GPU routing seam — is pinned by
//! `tests/owed_1017.rs`; this file owns the CPU driver-parallelism leg.)
//!
//! This gate owns the LAYER-LADDER candidate loop:
//! `atom_transport_ladder_reports` (src/terms/sae/identifiability.rs). Each
//! atom's #1096 transport ladder is an independent, pure fit (read the shared
//! model by index, build that atom's canonical per-layer coordinates, run the
//! pure `transport_ladder` solve), so the reports now fan across rayon. The
//! parallelization must be OBSERVABLE (results bit-identical to the sequential
//! walk) and FAST (a real wall-clock speedup when several heavy ladders race on
//! a multi-core host).
//!
//! Two properties are pinned:
//!
//! 1. PARITY. The parallel dispatch (many inputs, called from outside any rayon
//!    worker) and the sequential dispatch (the SAME call forced down the
//!    sequential body by invoking it from INSIDE a rayon worker, where the
//!    function's `current_thread_index().is_none()` nested-rayon guard keeps it
//!    sequential) must produce byte-identical reports, in input order. The
//!    transport solve is deterministic, so the two `Debug` renderings must match
//!    exactly — a regression that reordered results or raced shared state would
//!    diverge here. First-by-index error recovery is also pinned: an invalid
//!    input at a known position surfaces the SAME error string from both paths,
//!    even though the parallel path evaluates later inputs that the sequential
//!    `?`-walk would have skipped (the per-atom body is pure and cannot panic on
//!    shape-valid input, so running them is harmless).
//!
//! 2. SPEEDUP. With several heavy ladders, the parallel path's wall-clock must
//!    be strictly less than the forced-sequential path's on a multi-core host.
//!    The inequality is asserted only when rayon actually has >1 worker thread
//!    (a single-core CI box cannot speed up and must not flake); parity is
//!    asserted unconditionally.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use std::time::Instant;

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use gam::inference::row_metric::RowMetric;
use gam::terms::sae::chart_canonicalization::CanonicalChartTopology;
use gam::terms::sae::identifiability::{
    AtomTopology, AtomTransportLadderInput, AtomTransportLadderReport, FittedAtom,
    FittedSaeManifold, atom_transport_ladder_reports,
};

/// A bare certificate-only fitted atom carrying just a name + Euclidean frame —
/// the transport-ladder leg only reads `atom.name` and the index, so the rest is
/// minimal valid scaffolding.
fn bare_atom(name: &str) -> FittedAtom {
    let mut frame = Array2::<f64>::zeros((1, 2));
    frame[[0, 0]] = 1.0;
    FittedAtom {
        name: name.to_string(),
        topology: AtomTopology::EuclideanPatch { latent_dim: 2 },
        frame,
        ard_variances: None,
        lowering_error: 0.0,
        chart_canonicalized: false,
        inner_fit: None,
    }
}

/// A `FittedSaeManifold` carrying `n_atoms` bare atoms named `atom_0..`.
fn model_with_atoms(n_atoms: usize) -> FittedSaeManifold {
    let atoms: Vec<FittedAtom> = (0..n_atoms)
        .map(|k| bare_atom(&format!("atom_{k}")))
        .collect();
    FittedSaeManifold {
        atoms,
        jacobian_rows: Vec::new(),
        isometry_penalty_root: Array2::<f64>::zeros((0, 0)),
        metric: RowMetric::euclidean(1, 1).expect("euclidean metric"),
    }
}

/// A deterministic, monotone three-layer interval ladder for atom `atom_index`.
/// Layer 0 is the base coordinate `t ∈ (0, 1)`; layers 1 and 2 are smooth
/// monotone warps of it (so each adjacent + two-hop transport map fits cleanly).
/// `n_rows` controls the per-ladder work — larger makes the per-candidate solve
/// heavier, which is what the parallel fan-out amortizes.
fn interval_ladder(atom_index: usize, n_rows: usize) -> AtomTransportLadderInput {
    let base = Array1::from_iter((0..n_rows).map(|i| (i as f64 + 0.5) / n_rows as f64));
    // Smooth monotone warps: u = t + 0.15 sin(2π t)/(2π), v = u + 0.10 sin(2π u)/(2π).
    let tau = std::f64::consts::TAU;
    let warp = |x: f64, amp: f64| x + amp * (tau * x).sin() / tau;
    let layer1 = base.mapv(|t| warp(t, 0.15));
    let layer2 = layer1.mapv(|u| warp(u, 0.10));
    AtomTransportLadderInput {
        atom_index,
        layers: vec![10, 11, 12],
        coords: vec![base, layer1, layer2],
        topologies: vec![
            CanonicalChartTopology::Interval,
            CanonicalChartTopology::Interval,
            CanonicalChartTopology::Interval,
        ],
    }
}

/// Render the reports for a byte-exact parity comparison. The transport solve is
/// deterministic, so identical inputs must yield identical `Debug` output
/// regardless of dispatch — any reordering or shared-state race shows up here.
fn render(reports: &[AtomTransportLadderReport]) -> String {
    format!("{reports:?}")
}

/// Run `atom_transport_ladder_reports` on a rayon worker so the function's
/// `current_thread_index().is_none()` nested-rayon guard is FALSE and it executes
/// its sequential body. This is how the test exercises the sequential path with
/// the identical public entry point the parallel path uses.
fn run_forced_sequential(
    model: &FittedSaeManifold,
    ladders: &[AtomTransportLadderInput],
) -> Result<Vec<AtomTransportLadderReport>, String> {
    rayon::iter::repeat_n((), 1)
        .map(|()| atom_transport_ladder_reports(model, ladders))
        .collect::<Vec<_>>()
        .into_iter()
        .next()
        .expect("exactly one forced-sequential result")
}

/// PARITY: the parallel dispatch (called outside any rayon worker) and the
/// sequential dispatch (forced by calling from inside a rayon worker, where the
/// nested-rayon guard keeps the body sequential) produce byte-identical reports
/// in input order.
#[test]
fn ladder_parallel_matches_sequential_1017() {
    let n_atoms = 8usize;
    let model = model_with_atoms(n_atoms);
    let ladders: Vec<AtomTransportLadderInput> =
        (0..n_atoms).map(|k| interval_ladder(k, 96)).collect();

    // Parallel path: outside any worker, len >= the parallel threshold.
    let parallel = atom_transport_ladder_reports(&model, &ladders).expect("parallel ladder fits");
    // Sequential path: the SAME call forced down the serial body on a worker.
    let sequential = run_forced_sequential(&model, &ladders).expect("sequential ladder fits");

    assert_eq!(
        parallel.len(),
        n_atoms,
        "every atom must produce a ladder report"
    );
    assert_eq!(
        parallel.len(),
        sequential.len(),
        "parallel and sequential must produce the same number of reports"
    );
    // Input-order preservation: report k is atom k in both paths.
    for (k, (par, seq)) in parallel.iter().zip(sequential.iter()).enumerate() {
        assert_eq!(par.atom_index, k, "parallel report {k} out of input order");
        assert_eq!(
            seq.atom_index, k,
            "sequential report {k} out of input order"
        );
        assert_eq!(par.atom_name, format!("atom_{k}"));
    }
    // Byte-exact parity of the full report payloads.
    assert_eq!(
        render(&parallel),
        render(&sequential),
        "parallel ladder reports must be byte-identical to the sequential walk"
    );
}

/// First-by-index error recovery is identical across dispatch: an out-of-range
/// atom index at a KNOWN position surfaces the same error from both paths, even
/// though the parallel path also evaluates the later (valid) inputs.
#[test]
fn ladder_first_error_is_deterministic_across_dispatch_1017() {
    let model = model_with_atoms(4);
    // Position 1 is invalid (atom index 99 out of range); positions 0, 2, 3 valid.
    let mut ladders: Vec<AtomTransportLadderInput> =
        (0..4).map(|k| interval_ladder(k, 48)).collect();
    ladders[1].atom_index = 99;

    let parallel_err = atom_transport_ladder_reports(&model, &ladders)
        .expect_err("an out-of-range atom index must error");
    let sequential_err =
        run_forced_sequential(&model, &ladders).expect_err("sequential must also error");

    assert_eq!(
        parallel_err, sequential_err,
        "the first-by-index error must be identical across parallel and sequential dispatch"
    );
    assert!(
        parallel_err.contains("out of range for 4 fitted atoms"),
        "the surfaced error must be the index-99 out-of-range one; got: {parallel_err}"
    );
}

/// SPEEDUP: with several heavy ladders the parallel path is strictly faster than
/// the forced-sequential path. Only asserted when rayon has >1 worker (a
/// single-core box cannot speed up and must not flake); parity above is
/// unconditional.
#[test]
fn ladder_parallel_is_faster_than_sequential_1017() {
    let n_threads = rayon::current_num_threads();
    let n_atoms = 16usize;
    let model = model_with_atoms(n_atoms);
    let ladders: Vec<AtomTransportLadderInput> =
        (0..n_atoms).map(|k| interval_ladder(k, 256)).collect();

    // Warm any one-time allocator/thread-pool spin-up so it is not charged to the
    // first measured call.
    let warm = atom_transport_ladder_reports(&model, &ladders).expect("warm");
    assert_eq!(warm.len(), n_atoms);

    // Forced-sequential timing: run inside a worker so the guard keeps it serial.
    let t_seq = {
        let start = Instant::now();
        let reports = run_forced_sequential(&model, &ladders).expect("sequential fits");
        let elapsed = start.elapsed();
        assert_eq!(reports.len(), n_atoms);
        elapsed
    };

    // Parallel timing: outside any worker, fans across rayon.
    let t_par = {
        let start = Instant::now();
        let reports = atom_transport_ladder_reports(&model, &ladders).expect("parallel fits");
        let elapsed = start.elapsed();
        assert_eq!(reports.len(), n_atoms);
        elapsed
    };

    if n_threads > 1 {
        assert!(
            t_par < t_seq,
            "parallel ladder fits must be faster than sequential on a {n_threads}-thread host: \
             parallel {t_par:?} vs sequential {t_seq:?}"
        );
    }
}
