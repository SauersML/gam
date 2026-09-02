#![cfg(test)]
//! `sae_row_jet_program_matches_production_row_jets_on_converged_cache` and
//! `ordered_beta_bernoulli_outer_objective_advertises_analytic_gradient`, split verbatim out
//! of `tests.rs` to keep that tracked file under the #780 10k-line gate.
//! Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; the shared
//! `gamma_fd_tiny_fixture` is sourced from the sibling `tests` module.
//!
//! That "declared as a `#[cfg(test)] mod`" claim is restated below as an inner
//! attribute so the compiler enforces it here, not just in `mod.rs`.
#![cfg(test)]

use super::tests::gamma_fd_tiny_fixture;
use super::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// Thread-scoped allocation ledger for the full-output schedule benchmark.  The
// allocator delegates every operation unchanged to `System`; counters are active
// only on the single libtest thread inside an explicitly measured region.
struct SaeRowJetCountingAllocator;

thread_local! {
    static TRACK_ROW_JET_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ROW_JET_ALLOCATION_CALLS: Cell<u64> = const { Cell::new(0) };
    static ROW_JET_ALLOCATED_BYTES: Cell<u64> = const { Cell::new(0) };
}

fn note_row_jet_allocation(size: usize) {
    if !TRACK_ROW_JET_ALLOCATIONS
        .try_with(Cell::get)
        .unwrap_or(false)
    {
        return;
    }
    ROW_JET_ALLOCATION_CALLS
        .try_with(|counter| counter.set(counter.get() + 1))
        .unwrap_or(());
    ROW_JET_ALLOCATED_BYTES
        .try_with(|counter| counter.set(counter.get() + size as u64))
        .unwrap_or(());
}

// SAFETY: every operation is delegated to `System` with its pointer/layout
// contract unchanged.  The const-initialized thread-local counters allocate
// nothing and cannot alter allocation ownership.
unsafe impl GlobalAlloc for SaeRowJetCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is valid by this method's `GlobalAlloc` contract.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            note_row_jet_allocation(layout.size());
        }
        pointer
    }

    // SAFETY: this preserves `GlobalAlloc::alloc_zeroed`'s layout contract and
    // delegates ownership unchanged to the system allocator.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is valid by this method's `GlobalAlloc` contract.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            note_row_jet_allocation(layout.size());
        }
        pointer
    }

    // SAFETY: callers must provide the live pointer and matching layout
    // required by `GlobalAlloc::dealloc`; both are forwarded unchanged.
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies the matching live `System` allocation.
        unsafe { System.dealloc(pointer, layout) }
    }

    // SAFETY: callers must satisfy `GlobalAlloc::realloc`'s live-allocation
    // contract; the pointer, layout, and requested size are forwarded unchanged.
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller supplies a live allocation and `new_size` is
        // forwarded unchanged, as required by `GlobalAlloc`.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            note_row_jet_allocation(new_size);
        }
        new_pointer
    }
}

#[global_allocator]
static SAE_ROW_JET_GLOBAL_ALLOCATOR: SaeRowJetCountingAllocator = SaeRowJetCountingAllocator;

fn begin_row_jet_allocation_measurement() {
    TRACK_ROW_JET_ALLOCATIONS.with(|tracking| tracking.set(false));
    ROW_JET_ALLOCATION_CALLS.with(|counter| counter.set(0));
    ROW_JET_ALLOCATED_BYTES.with(|counter| counter.set(0));
    TRACK_ROW_JET_ALLOCATIONS.with(|tracking| tracking.set(true));
}

fn end_row_jet_allocation_measurement() -> (u64, u64) {
    TRACK_ROW_JET_ALLOCATIONS.with(|tracking| tracking.set(false));
    (
        ROW_JET_ALLOCATION_CALLS.with(Cell::get),
        ROW_JET_ALLOCATED_BYTES.with(Cell::get),
    )
}

/// Build a one-row, full-channel softmax fixture for the #932 schedule benchmark.
/// Every atom has a live periodic coordinate jet and one beta-border channel, so
/// the timing covers reconstruction gradient/Hessian, coordinate and mixed blocks,
/// and beta / beta_deriv / beta_l_deriv rather than the gate-logit-only GPU subset.
fn schedule_perf_fixture(
    k_atoms: usize,
    p: usize,
    mode: AssignmentMode,
) -> (
    SaeManifoldTerm,
    Vec<SaeLocalRowVar>,
    Vec<Array4<f64>>,
    Vec<SaeBorderChannel>,
    Array1<f64>,
) {
    let n = 1usize;
    let m = 3usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let coordinate =
            Array2::from_shape_vec((n, 1), vec![((atom * 17 + 3) as f64 * 0.037).fract()]).unwrap();
        let (phi, jet) = evaluator.evaluate(coordinate.view()).unwrap();
        let decoder = Array2::from_shape_fn((m, p), |(basis, column)| {
            ((atom * 31 + basis * 11 + column * 7 + 1) as f64 * 0.019).sin()
        });
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("softmax_perf_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone()),
        );
        coord_blocks.push(coordinate);
    }
    let logits = Array2::from_shape_fn((n, k_atoms), |(_, atom)| {
        0.7 * ((atom * 13 + 2) as f64 * 0.17).cos() - 0.03 * atom as f64
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let mut vars = Vec::with_capacity(k_atoms.saturating_sub(1) + k_atoms);
    for atom in 0..k_atoms.saturating_sub(1) {
        vars.push(SaeLocalRowVar::Logit { atom });
    }
    for atom in 0..k_atoms {
        vars.push(SaeLocalRowVar::Coord { atom, axis: 0 });
    }
    let second_jets = term.atom_second_jets().unwrap();
    let border: Vec<SaeBorderChannel> = (0..k_atoms)
        .map(|atom| SaeBorderChannel {
            atom,
            basis_col: atom % m,
            index: atom,
            output: (0..p)
                .map(|column| ((atom * 5 + column * 3 + 1) as f64 * 0.23).cos())
                .collect(),
        })
        .collect();
    let assignments = term.assignment.try_assignments_row(0).unwrap();
    (term, vars, second_jets, border, assignments)
}

fn softmax_schedule_perf_fixture(
    k_atoms: usize,
    p: usize,
) -> (
    SaeManifoldTerm,
    Vec<SaeLocalRowVar>,
    Vec<Array4<f64>>,
    Vec<SaeBorderChannel>,
    Array1<f64>,
) {
    schedule_perf_fixture(k_atoms, p, AssignmentMode::softmax(0.9))
}

fn independent_schedule_perf_fixture(
    k_atoms: usize,
    p: usize,
) -> (
    SaeManifoldTerm,
    Vec<SaeLocalRowVar>,
    Vec<Array4<f64>>,
    Vec<SaeBorderChannel>,
    Array1<f64>,
) {
    schedule_perf_fixture(
        k_atoms,
        p,
        AssignmentMode::ordered_beta_bernoulli(0.9, 1.0, false),
    )
}

/// Exact nested-buffer shape returned by the pre-#932 hand implementation. It
/// deliberately remains test-local: production has one packed row allocation,
/// while this type preserves the historical allocation/performance baseline.
struct LegacySaeRowJets {
    vars: Vec<SaeLocalRowVar>,
    first: Vec<Vec<f64>>,
    second: Vec<Vec<Vec<f64>>>,
    beta: Vec<Vec<f64>>,
    beta_deriv: Vec<Vec<Vec<f64>>>,
    beta_l_deriv: Vec<Vec<Vec<f64>>>,
}

fn row_jets_for_logdet_hand_reference(
    term: &SaeManifoldTerm,
    row: usize,
    vars: Vec<SaeLocalRowVar>,
    assignments: ArrayView1<'_, f64>,
    second_jets: &[Array4<f64>],
    border: &[SaeBorderChannel],
) -> LegacySaeRowJets {
    let p = term.output_dim();
    let q = vars.len();
    let sqrt_row_w = term
        .row_loss_weights
        .as_deref()
        .map_or(1.0, |weights| weights[row].sqrt());
    let mut first = vec![vec![0.0_f64; p]; q];
    let mut second = vec![vec![vec![0.0_f64; p]; q]; q];
    let mut beta = vec![vec![0.0_f64; p]; border.len()];
    let mut beta_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
    let mut beta_l_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
    term.fill_row_jets_hand_reference(
        row,
        &vars,
        assignments,
        second_jets,
        border,
        sqrt_row_w,
        &mut first,
        &mut second,
        &mut beta,
        &mut beta_deriv,
        &mut beta_l_deriv,
    );
    LegacySaeRowJets {
        vars,
        first,
        second,
        beta,
        beta_deriv,
        beta_l_deriv,
    }
}

fn row_jet_channel_error(actual: &SaeRowJets, expected: &LegacySaeRowJets) -> (f64, f64) {
    assert_eq!(actual.vars.len(), expected.vars.len());
    let q = expected.vars.len();
    let p = expected.first.first().map_or(0, Vec::len);
    let n_beta = expected.beta.len();
    assert_eq!(actual.channels.q(), q);
    assert_eq!(actual.channels.p(), p);
    assert_eq!(actual.channels.n_beta(), n_beta);
    let mut max_abs = 0.0_f64;
    let mut scale = 1.0_f64;
    let mut visit = |a: f64, b: f64| {
        max_abs = max_abs.max((a - b).abs());
        scale = scale.max(a.abs()).max(b.abs());
    };
    for a in 0..q {
        for (&actual_value, &expected_value) in actual.first(a).iter().zip(&expected.first[a]) {
            visit(actual_value, expected_value);
        }
        for b in 0..q {
            for (&actual_value, &expected_value) in
                actual.second(a, b).iter().zip(&expected.second[a][b])
            {
                visit(actual_value, expected_value);
            }
        }
        for beta in 0..n_beta {
            for (&actual_value, &expected_value) in actual
                .beta_deriv(a, beta)
                .iter()
                .zip(&expected.beta_deriv[a][beta])
            {
                visit(actual_value, expected_value);
            }
            for (&actual_value, &expected_value) in actual
                .beta_l_deriv(a, beta)
                .iter()
                .zip(&expected.beta_l_deriv[a][beta])
            {
                visit(actual_value, expected_value);
            }
        }
    }
    for beta in 0..n_beta {
        for (&actual_value, &expected_value) in actual.beta(beta).iter().zip(&expected.beta[beta]) {
            visit(actual_value, expected_value);
        }
    }
    (max_abs, scale)
}

/// Full-output correctness, allocation, and release timing gate against the
/// exact historical non-abstracted hand reference. `fixture` selects the gate
/// graph; every reconstruction and beta-border channel is checked entry by
/// entry before either path is timed.
fn compiled_schedule_beats_hand_full_channels_932(
    gate_label: &str,
    fixture: fn(
        usize,
        usize,
    ) -> (
        SaeManifoldTerm,
        Vec<SaeLocalRowVar>,
        Vec<Array4<f64>>,
        Vec<SaeBorderChannel>,
        Array1<f64>,
    ),
) {
    use std::time::{Duration, Instant};

    fn compiled_checksum(jets: &SaeRowJets) -> f64 {
        let first = (!jets.vars.is_empty())
            .then(|| jets.first(0).first().copied())
            .flatten()
            .unwrap_or(0.0);
        let second = (!jets.vars.is_empty())
            .then(|| jets.second(0, 0).first().copied())
            .flatten()
            .unwrap_or(0.0);
        let beta = (jets.channels.n_beta() != 0)
            .then(|| jets.beta(0).first().copied())
            .flatten()
            .unwrap_or(0.0);
        first + second + beta
    }

    fn hand_checksum(jets: &LegacySaeRowJets) -> f64 {
        let first = jets
            .first
            .first()
            .and_then(|row| row.first())
            .copied()
            .unwrap_or(0.0);
        let second = jets
            .second
            .first()
            .and_then(|row| row.first())
            .and_then(|column| column.first())
            .copied()
            .unwrap_or(0.0);
        let beta = jets
            .beta
            .first()
            .and_then(|row| row.first())
            .copied()
            .unwrap_or(0.0);
        first + second + beta
    }

    for &k_atoms in &[1usize, 2, 8, 16, 32, 64] {
        let p = 16usize;
        let (term, vars, second_jets, border, assignments) = fixture(k_atoms, p);
        let compiled = term
            .row_jets_for_logdet(0, vars.clone(), assignments.view(), &second_jets, &border)
            .unwrap();
        let hand = row_jets_for_logdet_hand_reference(
            &term,
            0,
            vars.clone(),
            assignments.view(),
            &second_jets,
            &border,
        );
        let (max_abs, scale) = row_jet_channel_error(&compiled, &hand);
        assert!(
            max_abs <= 2.0e-12 * scale,
            "K={k_atoms} compiled vs hand full-channel max abs {max_abs:e}, scale {scale:e}"
        );

        begin_row_jet_allocation_measurement();
        let allocation_probe_compiled = term
            .row_jets_for_logdet(0, vars.clone(), assignments.view(), &second_jets, &border)
            .unwrap();
        let (compiled_allocations, compiled_bytes) = end_row_jet_allocation_measurement();
        begin_row_jet_allocation_measurement();
        let allocation_probe_hand = row_jets_for_logdet_hand_reference(
            &term,
            0,
            vars.clone(),
            assignments.view(),
            &second_jets,
            &border,
        );
        let (hand_allocations, hand_bytes) = end_row_jet_allocation_measurement();
        assert!(
            (compiled_checksum(&allocation_probe_compiled) + hand_checksum(&allocation_probe_hand))
                .is_finite(),
            "allocation probes must materialize finite full channels"
        );
        assert!(
            compiled_allocations <= hand_allocations && compiled_bytes <= hand_bytes,
            "K={k_atoms} compiled allocations {compiled_allocations}/{compiled_bytes}B must not \
             exceed hand {hand_allocations}/{hand_bytes}B"
        );
        assert_eq!(
            compiled_allocations, 2,
            "K={k_atoms} warmed full row must allocate only the owned vars and one packed channel buffer"
        );

        #[cfg(debug_assertions)]
        let (repetitions, trials) = (1usize, 1usize);
        #[cfg(not(debug_assertions))]
        let (repetitions, trials) = match k_atoms {
            1 => (20_000usize, 5usize),
            2 => (20_000usize, 5usize),
            8 => (2_000usize, 5usize),
            16 => (400usize, 5usize),
            32 => (80usize, 5usize),
            64 => (10usize, 5usize),
            _ => (1usize, 1usize),
        };

        let mut best_compiled = Duration::MAX;
        let mut best_hand = Duration::MAX;
        let mut accumulated = 0.0_f64;
        for trial in 0..trials {
            let run_compiled = || {
                let start = Instant::now();
                let mut sum = 0.0_f64;
                for _ in 0..repetitions {
                    let jets = term
                        .row_jets_for_logdet(
                            0,
                            vars.clone(),
                            assignments.view(),
                            &second_jets,
                            &border,
                        )
                        .unwrap();
                    sum += compiled_checksum(&jets);
                }
                (start.elapsed(), sum)
            };
            let run_hand = || {
                let start = Instant::now();
                let mut sum = 0.0_f64;
                for _ in 0..repetitions {
                    let jets = row_jets_for_logdet_hand_reference(
                        &term,
                        0,
                        vars.clone(),
                        assignments.view(),
                        &second_jets,
                        &border,
                    );
                    sum += hand_checksum(&jets);
                }
                (start.elapsed(), sum)
            };
            let ((compiled_time, compiled_sum), (hand_time, hand_sum)) = if trial % 2 == 0 {
                (run_compiled(), run_hand())
            } else {
                let hand_result = run_hand();
                let compiled_result = run_compiled();
                (compiled_result, hand_result)
            };
            best_compiled = best_compiled.min(compiled_time);
            best_hand = best_hand.min(hand_time);
            accumulated += compiled_sum + hand_sum;
        }
        assert!(
            accumulated.is_finite(),
            "timed channel checksum must be finite"
        );
        let compiled_ns = best_compiled.as_nanos() as f64 / repetitions as f64;
        let hand_ns = best_hand.as_nanos() as f64 / repetitions as f64;
        eprintln!(
            "[SAE-{gate_label}-932] K={k_atoms} P={p} hand={hand_ns:.1} ns/row \
             compiled={compiled_ns:.1} ns/row ratio={:.4}x max_abs={max_abs:.3e} \
             allocs hand={hand_allocations}/{hand_bytes}B \
             compiled={compiled_allocations}/{compiled_bytes}B",
            compiled_ns / hand_ns
        );
        #[cfg(not(debug_assertions))]
        assert!(
            compiled_ns <= hand_ns,
            "K={k_atoms} compiled schedule {compiled_ns:.1} ns/row must beat hand {hand_ns:.1} ns/row"
        );
    }
}

/// The softmax hand path materializes and contracts `d2z[L,L,K]`; the compiled
/// centered-moment schedule is output-optimal O(L²P).
#[test]
pub(crate) fn softmax_compiled_schedule_beats_hand_full_channels_932() {
    compiled_schedule_beats_hand_full_channels_932("SOFTMAX", softmax_schedule_perf_fixture);
}

/// Independent gates have diagonal logit Hessians. The compiled schedule must
/// exploit that structure directly and beat the historical hand assembly at
/// every tested width; the generic runtime jet is correctness-only.
#[test]
pub(crate) fn independent_compiled_schedule_beats_hand_full_channels_932() {
    compiled_schedule_beats_hand_full_channels_932(
        "INDEPENDENT",
        independent_schedule_perf_fixture,
    );
}

#[test]
pub(crate) fn ordered_beta_bernoulli_outer_objective_advertises_analytic_gradient() {
    // The ordered Beta--Bernoulli shared-mass third channel is assembled from
    // the exact integrated scalar in `logdet_theta_adjoint` (#1006),
    // so the outer objective advertises an analytic gradient like every
    // other assignment mode.
    let (mut term, target, rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.9, 1.0, false);

    let obj = SaeManifoldOuterObjective::new(term, target, None, rho, 5, 0.4, 1.0e-6, 1.0e-6);
    assert_eq!(obj.capability().gradient, Derivative::Analytic);
}

