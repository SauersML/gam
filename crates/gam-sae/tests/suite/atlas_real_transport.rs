//! Atlas-Machine composition on REAL Qwen3 transport artifacts.
//!
//! Turns the abstract contract composer + loop-holonomy instrument into a
//! measurement on actual model internals. The fixture
//! `tests/data/qwen3_l11_l17_l23_theta.json` holds the per-row circular chart
//! angle `θ` — the angle in each layer's top-2 principal plane of the real
//! centered activations — at residual-stream layers L11 / L17 / L23 of
//! Qwen3.5-35B-A3B, from the MSI activation cache. Each layer's `θ` carries its
//! own arbitrary plane gauge/orientation; the composition triangle below is a
//! CLOSED loop, so every per-layer gauge cancels and the holonomy verdict is
//! gauge-invariant (no cross-layer alignment needed). Full provenance — and why
//! the top-2 plane angle rather than the K=1 SAE atom (its line search
//! live-locks at the MSI gamfit) — is in `tests/data/MSI_PROVENANCE.md`.
//!
//! From those angles this test fits the three inter-layer transports with the
//! crate's own REML machinery — the two forward hops `h: L11→L17`, `L17→L23`
//! and the direct map `L11→L23` — and then:
//!
//!  1. composes the two forward hops into an END-TO-END error bound
//!     ([`compose_contracts`]): each hop's target-space residual is the stage
//!     defect and its `sup|h′|` the metric expansion, so `total_defect` is the
//!     shadowing bound on the composed L11→L23 map built stage-by-stage;
//!  2. measures the LOOP HOLONOMY of the composition triangle
//!     `h_ab, h_bc, h_ac⁻¹` ([`loop_holonomy`]): each transport is classified as
//!     an `O(2)` element and the loop returns to L11, so a nontrivial net
//!     element is the obstruction to "the weekday circle is one global feature
//!     carried consistently around the loop" — equivalently, the measured
//!     failure of the composition law `h_ac = h_bc ∘ h_ab`, judged against the
//!     loop's own summed `O(2)` defects (measure-don't-latch).
//!
//! The printed numbers (visible with `--nocapture`) are the deliverable; the
//! assertions pin only data-agnostic structural invariants, never a fitted
//! value.

