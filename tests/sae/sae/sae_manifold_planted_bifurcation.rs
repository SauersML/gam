//! #1007 planted-bifurcation oracle: when the data admits TWO equally good
//! manifold completions, the certified curvature-homotopy walk must NOT
//! silently pick a branch and report a certified unique continuation.
//!
//! Fixture: a single circle atom (K = 1, d = 1, one harmonic) fit to the
//! exactly symmetric union of two planted circles living in mutually
//! orthogonal ambient planes — same radius, same angle ladder, same row
//! count. A one-atom dictionary must commit to ONE of the planes, and by
//! construction both choices have identical objective value: the η = 0
//! Eckart-Young anchor has a tied boundary singular pair (σ_r = σ_{r+1}),
//! so the global rank-2 relaxation is non-unique and no certified unique
//! branch to η = 1 exists.
//!
//! Contract under test (#1007 build-plan item 4): the walk must produce a
//! DETECTED event — either a recorded `CurvatureBifurcation` (pivot
//! collapse / tied-anchor detection) or a refusal to certify
//! (`arrived = false`, deferring to the documented seed cascade). Arriving
//! with `bifurcation: None` on this fixture is precisely the "silent branch
//! choice" failure mode the certificate exists to prevent.

