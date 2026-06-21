//! Owed-work regression gate for GitHub issue #1388 (survival marginal-slope).
//!
//! ISSUE — two failures on the tiny `cirrhosis_survival` / `heart_failure_survival`
//! benchmarks, both rooted in the SAME degenerate geometry: once every
//! categorical level expands into its own column the joint marginal-slope design
//! becomes UNDER-DETERMINED (`p_joint > n`), so the unpenalized joint Jacobian
//! rank is capped at `min(n, p_joint) < p_joint`. In that regime:
//!
//!   1. "post-T construction rank invariant violated: rank(T)=53 != dim=31" — the
//!      post-T verifier ranked the BARE data Gram (no penalty augmentation, and a
//!      bespoke eigenvalue cutoff) while the channel-aware audit decides kept rank
//!      from the PENALTY-AUGMENTED joint Gram with its own `rank_of_gram`
//!      convention. A penalty-anchored direction the audit legitimately keeps was
//!      demoted by the verifier → a FALSE "post-T rank invariant violated" abort.
//!
//!   2. Stack overflow in `rust_gamlss_survival` right after `[STAGE] runtime
//!      threads` — the downstream canonicalisation fan-out over the many
//!      penalty/level blocks of an under-determined joint.
//!
//! FIX (landed on `origin/main` in `src/identifiability/canonical.rs`):
//!   * `1573681d3` / `2e2871e75` — the post-T rank invariant now ranks the
//!     PENALTY-AUGMENTED reduced design `J_can` with the audit's OWN
//!     `rank_of_gram` convention and certifies `rank(J_can) == p_total_red`
//!     (== audit_kept_rank), so an under-determined-but-penalty-identifiable
//!     `p_joint > n` joint no longer trips the invariant.
//!   * `eb51fa190` (#1391) — convention-robust post-T invariant
//!     `rank(J_can) == min(rank(J_pre), p_total_red)`.
//!   * `15ae5d389` (#1388) — preflight WARN surfacing the `p_joint > n`
//!     under-determination so the failure is diagnosable from the log.
//!
//! CERTIFICATE (public API only — `canonicalize_for_identifiability`):
//! reconstruct the issue's geometry at unit scale — a channel-aware survival
//! marginal-slope joint whose total column count EXCEEDS the row count
//! (`p_joint > n`) via categorical level expansion in a lower-priority block,
//! anchored by a geometry-owning `stacked_design` time block. This is the exact
//! regime that produced both the "post-T rank invariant violated" abort and the
//! canonicalisation fan-out overflow.
//!
//! The whole canonicalisation is driven on a DELIBERATELY BOUNDED-STACK worker
//! thread (1 MiB). A genuine UNBOUNDED recursion / data-proportional descent in
//! the fan-out would overflow that worker and abort the process (the `join`
//! below observes a panic-free completion), so the bounded stack is the active
//! guard for failure #2; the `Ok(..)` + post-T assertions are the guard for #1.
//! Were the rank-invariant fix reverted, `canonicalize_for_identifiability`
//! would return `Err(DimensionMismatch { reason: "... post-T rank invariant
//! violated ..." })` and the test would fail.

use std::sync::Arc;

use gam::families::custom_family::{AdditiveBlockJacobian, ParameterBlockSpec};
use gam::identifiability::canonical::canonicalize_for_identifiability;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

/// Number of survival output channels (η_entry, η_exit, ∂η) — routes the
/// canonicaliser through the channel-aware survival/marginal-slope path.
const K: usize = 3;

fn channel_spec(
    name: &str,
    design: Array2<f64>,
    own_channel: usize,
    priority: u8,
) -> ParameterBlockSpec {
    let n = design.nrows();
    let cb = Arc::new(AdditiveBlockJacobian {
        design: design.clone(),
        own_output: own_channel,
        n_family_outputs: K,
    });
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        gauge_priority: priority,
        jacobian_callback: Some(cb),
        ..ParameterBlockSpec::defaults()
    }
}

/// Build the under-determined `p_joint > n` survival-marginal-slope-style joint:
/// a geometry-owning `time` block (highest priority, `stacked_design` intact)
/// plus a `threshold` block whose `n_levels` one-hot categorical columns push
/// the total column count above `n` — exactly the cirrhosis/heart_failure regime
/// once every level becomes its own column.
fn build_underdetermined_specs(n: usize, n_levels: usize) -> Vec<ParameterBlockSpec> {
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();

    // time_transform: owns geometry via a 3·n-row [entry; exit; deriv] stack.
    let mut time_exit = Array2::<f64>::zeros((n, 2));
    let mut time_stacked = Array2::<f64>::zeros((3 * n, 2));
    for i in 0..n {
        time_exit[[i, 0]] = 1.0;
        time_exit[[i, 1]] = x[i];
        time_stacked[[i, 0]] = 1.0;
        time_stacked[[i, 1]] = 0.5 * x[i];
        time_stacked[[n + i, 0]] = 1.0;
        time_stacked[[n + i, 1]] = x[i];
        time_stacked[[2 * n + i, 0]] = 0.0;
        time_stacked[[2 * n + i, 1]] = 1.0;
    }
    let mut t_spec = channel_spec("time_transform", time_exit, 1, 200);
    t_spec.stacked_design = Some(DesignMatrix::Dense(DenseDesignMatrix::from(time_stacked)));
    t_spec.stacked_offset = Some(Array1::<f64>::zeros(3 * n));

    // threshold: intercept + one-hot expansion over `n_levels` categories. Sized
    // so 2 (time) + (1 + n_levels) (threshold) > n → under-determined joint.
    let p_th = 1 + n_levels;
    let mut threshold = Array2::<f64>::zeros((n, p_th));
    for i in 0..n {
        threshold[[i, 0]] = 1.0; // intercept (aliased with the time constant)
        let lvl = i % n_levels;
        threshold[[i, 1 + lvl]] = 1.0; // one-hot level membership
    }
    let th_spec = channel_spec("threshold", threshold, 0, 150);

    vec![t_spec, th_spec]
}

/// Drive `canonicalize_for_identifiability` on a deliberately bounded-stack
/// worker and assert it satisfies BOTH #1388 halves:
///   * does NOT overflow the bounded stack (a data-proportional / unbounded
///     fan-out would abort the worker — the `join` observes that), and
///   * does NOT falsely trip the post-T rank invariant (returns `Ok`, channel-
///     aware, geometry-owning time block preserved, non-fatal).
fn assert_underdetermined_joint_canonicalizes(n: usize, n_levels: usize, stack_bytes: usize) {
    let specs = build_underdetermined_specs(n, n_levels);
    let p_joint: usize = specs.iter().map(|s| s.design.ncols()).sum();
    assert!(
        p_joint > n,
        "fixture must be UNDER-DETERMINED to exercise the #1388 regime: \
         got p_joint={p_joint} <= n={n}",
    );

    // A genuine unbounded / data-proportional recursion in the canonicalisation
    // fan-out (failure #2) would overflow this bounded worker and abort it; a
    // bounded, iterative fan-out completes well within it. (Production gives
    // this path a 64 MiB Rayon worker / 512 MiB CLI stack; the tight bound here
    // is the active guard.)
    let worker = std::thread::Builder::new()
        .name("owed-1388-canon".to_string())
        .stack_size(stack_bytes)
        .spawn(move || canonicalize_for_identifiability(&specs))
        .expect("spawn bounded-stack canonicalisation worker");

    let result = worker
        .join()
        .expect("canonicalisation worker must NOT panic / overflow its stack (#1388 overflow)");

    // Failure #1: the under-determined joint must canonicalise without falsely
    // tripping the post-T rank invariant. Before 1573681d3 / 2e2871e75 /
    // eb51fa190 this returned Err(DimensionMismatch{reason: "...post-T rank
    // invariant violated..."}).
    let canon = match result {
        Ok(c) => c,
        Err(e) => panic!(
            "under-determined (p_joint={p_joint} > n={n}) survival marginal-slope joint \
             must canonicalise; got error (regression of the #1388 post-T rank \
             invariant fix): {e:?}",
        ),
    };

    // The channel-aware survival path must have been taken (the geometry under
    // test is the multi-channel marginal-slope one, not the flat audit).
    assert!(
        canon.used_channel_aware_audit,
        "multi-channel (K={K}) survival marginal-slope joint must route through the \
         channel-aware audit",
    );

    // The geometry-owning time block must survive at raw width with its stacked
    // operator intact (its constant is the gauge-owned baseline).
    let time = canon
        .reduced_specs
        .iter()
        .find(|s| s.name == "time_transform")
        .expect("time_transform survives canonicalisation");
    assert_eq!(
        time.design.ncols(),
        2,
        "geometry-owning time block keeps both raw columns",
    );
    assert!(
        time.stacked_design.is_some(),
        "geometry-owning time block keeps its stacked eta operator",
    );

    // The post-T invariant having held, the reduced width must be a genuine
    // subspace of the raw width and the audit must not be a hard fatal.
    let p_reduced: usize = canon.reduced_specs.iter().map(|s| s.design.ncols()).sum();
    assert!(
        p_reduced <= p_joint,
        "reduced width {p_reduced} must not exceed raw width {p_joint}",
    );
    assert!(
        !canon.audit.fatal,
        "under-determined penalty-identifiable joint must not hard-halt; audit: {}",
        canon.audit.summary,
    );
}

#[test]
fn owed_1388_underdetermined_survival_marginal_slope_canonicalizes_on_bounded_stack() {
    // p_joint = 2 + (1 + 24) = 27 > n = 20 → under-determined, the #1388 regime.
    assert_underdetermined_joint_canonicalizes(20, 24, 1 << 20);
}

#[test]
fn owed_1388_rank53_class_underdetermined_joint_canonicalizes_on_bounded_stack() {
    // Larger arm matching the heart_failure benchmark's reported geometry
    // ("rank(T)=53 != dim=31"): a wide categorical expansion that pushes
    // p_joint = 2 + (1 + 60) = 63 well past n = 40, exercising the post-T
    // fan-out at the scale where the rank-invariant fix actually bites. Same
    // bounded 1 MiB stack — if the fan-out had a column-proportional stack
    // descent, the WIDER block would overflow where the narrow one did not.
    assert_underdetermined_joint_canonicalizes(40, 60, 1 << 20);
}
