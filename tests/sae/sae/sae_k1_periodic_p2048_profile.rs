//! PROFILING harness (#1037 follow-up / K=1 perf lane): decompose the wall-time
//! of a K=1 periodic circle fit at real-bank scale (p=2048, multiple harmonics).
//!
//! The real banks (qwen p=2048, color p=5120) are K=1 PERIODIC circles. This
//! answers where a high-p K=1 periodic fit spends its time: the #1007 curvature-
//! homotopy walk (which for periodic only dials harmonics h>=2 and buys nothing
//! when the circle topology is already baked into the fundamental), versus the
//! full outer solve. Phase A times the walk alone; phase B times the full
//! production OuterProblem::run; B-A approximates the post-walk outer cost.
//! Prints a timed breakdown under `--nocapture`; the assertion only guards a
//! non-trivial fit (EV high), not timing.

// 7 cols = const + 3 harmonics: the fundamental traces the circle; h=2,3 are the
// curved columns the #1007 eta-dial actually scales, so the curvature walk does
// non-trivial work here (unlike the M=3 single-harmonic acceptance fixture).

