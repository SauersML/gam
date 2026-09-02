//! #907 acceptance — **selection accuracy + calibrated BF/stacking magnitudes
//! under repeated draws** for the cross-class topology race.
//!
//! The in-tree acceptance criterion from the issue's rescope: planted truth
//! recovery (circle vs `k`-cluster at matched SNR) with *calibrated* decision
//! magnitudes — a large reported log-Bayes-factor or a decisive stacking
//! weight must actually be right, replicate after replicate. Selection
//! accuracy alone can hide overconfidence; this sweep pins both:
//!
//! * **Accuracy**: across `2 × N_REPLICATES` independent draws (half circle
//!   truth, half 7-cluster truth at matched SNR), the stacking-headline winner
//!   must match the planted generator every time.
//! * **Calibration of magnitudes**: every *decisive* call — held-out stacking
//!   weight above [`DECISIVE_STACKING_WEIGHT`] or rank-aware-evidence log-BF
//!   above [`DECISIVE_LOG_BF`] nats — must be correct. An overconfident
//!   adjudicator (big BF, wrong class) fails here even if some accuracy
//!   slack were allowed.
//! * **Direction of the corroborating evidence**: the Laplace-evidence
//!   difference must agree with the planted truth in the overwhelming
//!   majority of draws (it is the corroboration channel, allowed a small
//!   minority of near-tie inversions, never a systematic flip).
//!
//! Same generators and candidates as `tests/topology_mixture_rung.rs`
//! (matched SNR 12), swept over many more seeds.

// ---------------------------------------------------------------------------
// Deterministic RNG (fixed integer seed, no clock).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Planted generators at MATCHED SNR (identical to the mixture-rung tests).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Ring candidate (held-out density + rank-aware evidence), as in the planted
// races.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// One race; records the decision and its magnitudes.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The sweep.
// ---------------------------------------------------------------------------

