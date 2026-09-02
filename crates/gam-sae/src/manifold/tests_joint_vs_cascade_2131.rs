//! JOINT-vs-CASCADE: what the cheap pairwise-κ ENERGY screen catches, and what
//! it cannot — the statistical half of the "blocks are linear objects; in-block
//! refinement is suboptimal under joint dependencies" investigation (#2131).
//!
//! The pairwise screen ([`super::pair_kappa::screen_pair`]) adjudicates a pair of
//! accepted atoms on the NORMALISED ENERGY CROSS-MOMENT
//! `ρ = E[r_A²·r_B²] / (E[r_A²]·E[r_B²])`, firing a MERGE only on POSITIVE binding
//! evidence `ρ > 1` (shared presence gate). That is a deliberate, sharp design:
//! `ρ = 1` is the independence null. These tests pin down, on the REAL shipped
//! screen and at frontier ambient width, the THREE distinct joint dependencies a
//! cascade can split across frames and which tail each lands in:
//!
//!   1. ONE circle whose 2-plane is SPLIT across two dense frames (each atom sees
//!      one diameter). The per-row energies are COMPLEMENTARY (`r_A²+r_B² = 1`),
//!      an ANTI-correlation ⇒ `ρ ≈ 1/2 < 1`. The presence-binding screen does NOT
//!      fire (it fires only on `ρ > 1`): the fragmentation of a single curved set
//!      into two linear frames lives in the LOWER tail, which the merge screen —
//!      by design — does not adjudicate. A DOCUMENTED GAP the terminal joint fit,
//!      not the screen, must close.
//!   2. Two circles, co-gated SHARED presence (a gated torus), independent angles.
//!      `ρ = 1/q > 1` ⇒ the screen FIRES. The screen's home tail.
//!   3. Two DENSE circles (`q = 1`) with CORRELATED phases (a torus density
//!      concentrated on the diagonal). Presence is constant, so each `r² ≡ 1`;
//!      the energy cross-moment is blind to the phase law ⇒ `ρ ≈ 1`, NO fire. The
//!      joint DENSITY (the interpretation) is invisible to a second-order ENERGY
//!      screen even though it is a genuine inter-atom dependence — recoverable
//!      only by a joint 2-D coordinate, at ZERO reconstruction cost (marginals
//!      already give full EV). The second documented gap.
//!
//! Together: the cheap screen catches exactly ONE of the three joint-dependence
//! regimes (shared-presence binding). The other two — energy complementarity of a
//! split single chart, and a phase law at dense presence — are structurally
//! outside an energy-cross-moment screen and are the province of the terminal
//! joint fit. Scale (`p ∈ {512, 2048}`) confirms the ρ anchors are ambient-width
//! invariant, so the claim is not a small-p artifact.

