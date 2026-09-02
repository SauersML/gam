//! #2023 C4 — Tier-0 shared mean (the manifold-tier analogue of
//! [`crate::tiered::Tier0Mean`]) tests: the shared mean de-means the target and
//! reconstructs exactly (round-trip), and — the headline — moving the global DC
//! into Tier-0 makes a DC-constant "zombie" atom EV-invisible BY CONSTRUCTION, so
//! the 6-circle fixture has ZERO zero-decoder survivors.
//!
//! A DC-constant zombie is an atom whose decoder loads ONLY the constant basis
//! column (a pure per-row constant, no manifold structure). Without Tier-0 the
//! atoms fit the RAW target, whose column mean is the global DC, so the zombie's
//! constant column loads that mean and the atom "survives" selection by carrying a
//! slice of it (the co-collapse-to-mean class, #10/#1893): removing it drops
//! explained variance, so its leave-one-atom-out ΔEV is positive and it is kept and
//! PC-reseeded. With the shared mean carried by Tier-0 the atoms fit the DE-MEANED
//! target `Z − μ`, whose column mean is zero, so the zombie's constant column loads
//! NOTHING — it decodes ≈0, earns essentially no explained variance, and its ΔEV is
//! non-positive: it is NOT a survivor. Genuine curved atoms (the 6 circles) earn
//! positive ΔEV in BOTH modes: Tier-0 removes the mean, not the structure. (The
//! failure mode Tier-0 exists to prevent — installing μ on top of a dictionary that
//! ALSO fit the raw mean into a decoder — is the DOUBLE-SUBTRACTION HAZARD below: it
//! biases every reconstruction by `+μ` and corrupts every atom's ΔEV, so the fixture
//! must fit the zombie against the same target Tier-0 leaves behind.)
//!
//! DOUBLE-SUBTRACTION HAZARD: exactly ONE stage owns the mean. `tier0_mean` must
//! stay `None` whenever an upstream data-prep step already centers the target
//! (e.g. the COMPOSE L17 driver's `tier0.json` mean/scale) — `None` is the correct
//! setting for already-centered data; only install a Tier-0 mean on RAW targets.
//! (Program follow-up: fold Tier-0 INTO the fitted artifact so encode/steer are
//! self-contained and the ownership question disappears — a default-flip once the
//! headline run is out.)

