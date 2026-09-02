//! One gate for every GPU-conditional test, so an absent device is RECORDED
//! rather than reported as a pass (#2422).
//!
//! 41 tests could report `1 passed` having executed zero assertions, 37 of them
//! because a CPU-only runner takes a bare `return;` before the first assertion.
//! `sphere_gpu_end_to_end_fit_hill_climb_10x_vs_cpu` was green for its entire
//! life and had never appeared in `MASTER_FAILURES`; on a real A10 it misses its
//! `>=10x` gate at **0.51x**. A test that passes having verified nothing is
//! worse than a missing test, because a reader counts it as coverage.
//!
//! Five different skip idioms had grown up across fifteen files, and they
//! disagreed on the two things that matter:
//!
//! * **A device FAULT is not an absent device.** Twenty-one sites wrote
//!   `let Some(rt) = GpuRuntime::resolve(..) else { return; }`, and that `else`
//!   also swallows `Err(..)`. A machine WITH a GPU whose driver is broken
//!   therefore reported the same silent pass as a machine without one. This
//!   gate panics on `Err` unconditionally: a fault is a failure, always.
//! * **A lane that demanded a GPU must not skip.** [`crate::global_policy`]
//!   already carries that demand, so [`GpuPolicy::Required`] turns an absent
//!   device into a panic here. A GPU lane sets the policy once and every gated
//!   test in the process becomes mandatory — no per-test opt-in, and no
//!   environment variable (`env::var` is banned in this tree).
//!
//! What remains — `Auto`/`Off` with no device — is a real skip, and it is
//! counted. [`crate::test_gate::skipped_for_absent_device`] lets a suite assert how many gated
//! tests declined to run, so "37 passed" can no longer hide "37 verified
//! nothing".
use crate::device_runtime::GpuRuntime;

/// The one line a skipped GPU test prints, and the one string a CI ledger step
/// has to grep for.
///
/// Kept as a constant rather than spelled out at the `eprintln!` so the emitter
/// and any reader cannot drift apart — the failure mode #2593 recorded for a
/// different pair of substring ladders.
pub const SKIPPED_MARKER: &str = "SKIPPED(no-cuda):";

/// The outcome of asking for a device in a test.
#[derive(Debug)]
pub enum GpuTestGate {
    /// A runtime resolved; the test body must proceed and assert.
    Ready(&'static GpuRuntime),
    /// No device on this host, under a policy that permits running without
    /// one. Counted by [`skipped_for_absent_device`] and announced on stderr.
    AbsentDevice,
}

impl GpuTestGate {
    /// The runtime, or `None` when the host has no device.
    ///
    /// Deliberately NOT `Option`-shaped at the call site by default: a caller
    /// that writes `let Some(rt) = gate.runtime() else { return }` is back to
    /// the idiom this module exists to remove. Prefer matching on the gate so
    /// the absent arm is written out and visible in review.
    pub fn runtime(&self) -> Option<&'static GpuRuntime> {
        match self {
            Self::Ready(runtime) => Some(runtime),
            Self::AbsentDevice => None,
        }
    }
}

#[cfg(test)]
mod tests_gpu_test_gate_2422 {
    use super::*;

    /// The counter is the whole point: an absent device must leave a trace a
    /// test can assert on, not just a line on stderr that nothing reads.
    ///
    /// This test is itself device-conditional, and it says so honestly: on a
    /// host WITH a device it checks that nothing was counted as skipped, and on
    /// a host without one it checks that the skip was counted. Both arms
    /// assert, which is the property #2422 is about.
    #[test]
    fn an_absent_device_is_counted_not_silent_2422() {
        let before = skipped_for_absent_device();
        match gpu_for_test("gate self-test") {
            GpuTestGate::Ready(runtime) => {
                assert_eq!(
                    skipped_for_absent_device(),
                    before,
                    "a resolved runtime must not count as a skip"
                );
                assert!(
                    !runtime.selected_device().name.is_empty(),
                    "a Ready gate must carry a runtime with a selected device"
                );
            }
            GpuTestGate::AbsentDevice => {
                assert_absent_device_was_counted(before);
            }
        }
    }

    /// The skip marker is one fixed string, and it is the string the emitter
    /// actually prints.
    ///
    /// A ledger step greps for `SKIPPED_MARKER`; if the `eprintln!` were to be
    /// reworded independently the scrape would silently find nothing and report
    /// a clean inventory for a run full of skips — the same "parser with no
    /// producer" shape as #2617. Asserting the constant's spelling here keeps
    /// the two ends of that contract pinned together.
    #[test]
    fn the_skip_marker_is_one_greppable_string_2422() {
        assert_eq!(
            SKIPPED_MARKER, "SKIPPED(no-cuda):",
            "the marker a CI ledger greps for must not drift; update the ledger step \
             in the same commit if this ever changes"
        );
        assert!(
            !SKIPPED_MARKER.contains('{'),
            "the marker must be a literal, not a format template, or a grep cannot match it"
        );
    }

    /// A device-free host must not report a *faulted* device as absent, and a
    /// faulted one must never be counted as a skip.
    ///
    /// The distinction is the reason this module exists: twenty-one sites wrote
    /// `let Some(rt) = resolve(..) else { return }`, whose else-arm swallows
    /// `Err`. There is no way to fabricate a driver fault in-process here, so
    /// this asserts the reachable half — that the gate's two non-panicking
    /// outcomes are exactly `Ready` and `AbsentDevice`, and that the absent one
    /// is always accompanied by a count.
    #[test]
    fn the_gate_has_no_third_silent_outcome_2422() {
        let before = skipped_for_absent_device();
        match gpu_for_test("gate exhaustiveness self-test") {
            GpuTestGate::Ready(runtime) => {
                assert!(
                    runtime.selected_device().total_mem_bytes > 0,
                    "a Ready gate must carry a usable device, not a placeholder"
                );
            }
            GpuTestGate::AbsentDevice => {
                assert_absent_device_was_counted(before);
            }
        }
    }

    /// `Ready` carries a runtime and `AbsentDevice` does not — the projection
    /// every call site reads.
    #[test]
    fn the_gate_projects_to_an_option_only_at_the_call_site_2422() {
        let gate = gpu_for_test("gate projection self-test");
        assert_eq!(
            gate.runtime().is_some(),
            matches!(gate, GpuTestGate::Ready(_)),
            "runtime() must agree with the variant"
        );
    }
}
