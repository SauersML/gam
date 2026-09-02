//! Suite-level gate: a host with a device must not record absent-device skips.
//!
//! Every GPU-conditional test in this binary calls `gpu_gate(name)`, which
//! delegates to `gam::gpu::test_gate::gpu_for_test`. That gate counts a skip
//! when — and only when — the host genuinely has no CUDA device.
//!
//! The guarantee worth enforcing is the contrapositive: on a host WITH a
//! device, the absent-device counter must stay at zero. If it moves, the probe
//! is disagreeing with the hardware and every gated test in the binary skipped
//! while the suite reported all-green (#2422).
//!
//! ## Why this is checked against the counter, not `cuda_selected()`
//!
//! The previous version asserted `GpuRuntime::resolve(Auto).is_some() ->
//! cuda_selected()`. That is a true implication, but it is not this suite's skip
//! condition, and on a GPU host it was worse than unhelpful: `cuda_selected()`
//! reads the process-wide policy, `backend_status_and_policy_dispatch_are_consistent`
//! sets that policy to `Off` in this same binary, and the policy is a
//! first-writer-wins `OnceLock`. So on a GPU runner this test's verdict depended
//! on which of the two ran first — it would have failed for a reason unrelated
//! to any GPU test's correctness whenever it lost the race.
//!
//! Policy selection is now enforced where it can be attributed — inside
//! `gpu_gate`, per test, naming the sibling that set the policy — and this file
//! carries the part that is genuinely suite-wide.
//!
//! ## What this cannot do, stated plainly
//!
//! libtest has no end-of-suite hook, so this runs as an ordinary test among the
//! others and sees whatever the counter holds when it happens to run. On a
//! device-free host it is therefore permissive by construction: first it sees
//! zero skips, last it sees all of them, and neither is a failure because a
//! device-free host is allowed to skip. The assertion has teeth only in the
//! direction that matters — device present, count non-zero — and that direction
//! cannot be reached by ordering, because a device that resolves for one test
//! resolves for every test in the same process.
//!
//! The order-proof half of the inventory belongs in CI, which can read the whole
//! run: every skip prints `gam::gpu::test_gate::SKIPPED_MARKER`, so a ledger
//! step can count them and state "N tests skipped their subject" beside the pass
//! count. That step does not exist yet and is the remaining work on #2422; this
//! test is what can be enforced from inside the process.

use gam::gpu::test_gate::GpuTestGate;

#[test]
fn gpu_required_tests_did_not_skip() {
    // Ask the DEVICE, not the policy. `gpu_for_test` panics on a probe fault, so
    // a broken driver fails here instead of being read as absence.
    match gpu_for_test("gpu_required_tests_did_not_skip (suite gate)") {
        GpuTestGate::Ready(runtime) => {
            // A device resolved for THIS test, so it resolves for every gated
            // test in this process. Any absent-device skip already recorded is
            // therefore a probe that disagreed with the hardware.
            let skips = skipped_for_absent_device();
            assert_eq!(
                skips, 0,
                "this host has a CUDA device ({}), yet {skips} gated test(s) in this \
                 binary recorded an absent-device skip. Those tests reported `ok` having \
                 verified nothing on the device they exist to exercise (#2422). Grep the \
                 run for `SKIPPED(no-cuda):` to see which.",
                runtime.selected_device().name,
            );
        }
        GpuTestGate::AbsentDevice => {
            // Device-free host: skipping is legitimate, and this test's own skip
            // was counted by the gate. Assert the one thing still checkable and
            // non-vacuous here — that the counter is a live instrument rather
            // than a stub wired to nothing, which is what would make the
            // device-present assertion above meaningless.
            assert!(
                skipped_for_absent_device() >= 1,
                "the gate reported an absent device, so the shared skip counter must have \
                 recorded at least this test's own skip. A counter that never moves cannot \
                 support the device-present assertion above, and the suite would be back to \
                 counting silent passes as coverage (#2422)."
            );
        }
    }
}
