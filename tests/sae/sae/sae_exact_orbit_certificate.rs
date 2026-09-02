//! #998 — the full-resolution certificate: exact gauge orbits in the model's
//! own (decoder, coordinate) parameter space.
//!
//! Three planted facts, each the acceptance criterion it is named after:
//!
//! 1. **Compensated orbits are exact data-nulls** for bases closed under the
//!    group action: the harmonic circle's U(1) phase shift must come back
//!    unpinned with its compensation residual at the numerical noise floor —
//!    no lowering-error calibration involved. And **closure is computed, not
//!    declared**: the same machinery on a flat patch must certify the so(2)
//!    rotation exactly for a linear basis (closed) while genuinely pinning it
//!    for a quadratic basis (not closed) — the data honestly pins what the
//!    model class is honestly not symmetric under.
//! 2. **The penalty channel inverts the #995 falsifier**: with exact-orbit
//!    realisation the verdict on a true model-class symmetry must come from
//!    the penalty root alone — installing an [`OrbitPenaltyOperator`] that
//!    costs the orbit pins it; removing the operator flips it unpinned while
//!    the data rows stay present throughout (they are a null either way).
//! 3. **Merging**: exact within-atom verdicts replace the frame-path ones for
//!    viewed atoms (no double reporting), while unviewed atoms keep the
//!    calibrated frame path.

