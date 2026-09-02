//! Unit gates for the #2629 floor classifier.
//!
//! Every case here is a SYNTHETIC ladder built from a known law, so the
//! classifier is tested against ground truth rather than against a fixture's
//! behaviour. The two families it must separate are:
//!
//! * `g(ρ) = w·a·tanh(a·ρ̃) + c·e^{−ρ}` — the criterion carries the barrier;
//! * `g(ρ) = c·e^{−ρ}` — it does not.
//!
//! The measured constants from the shipped fixtures are used as the `c` values
//! (`+87.5` from #2450's Matérn/Gaussian ladder, `−22.8` from #2629's
//! SAS/binomial one) so the synthetic ladders sit exactly where the real ones do.

use super::*;

/// #2450's measured face constant, on the Matérn/Gaussian fixture.
const FACE_C_POSITIVE: f64 = 87.512;
/// #2629's measured face constant, on the SAS/binomial fixture. Opposite sign:
/// the face may be approached from either side, and a classifier fed `|g|`
/// could not tell that from a floor.
const FACE_C_NEGATIVE: f64 = -22.82;

fn ladder_from(law: impl Fn(f64) -> f64) -> Vec<GuardLadderRung> {
    SATURATED_RHO_LADDER
        .iter()
        .map(|&rho| GuardLadderRung {
            rho,
            rho_gradient: law(rho),
        })
        .collect()
}

fn bare_face_ladder(c: f64) -> Vec<GuardLadderRung> {
    ladder_from(|rho| c * (-rho).exp())
}

