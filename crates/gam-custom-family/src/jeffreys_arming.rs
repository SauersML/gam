//! The Jeffreys/Firth arming lifecycle (#979, Jeffreys ruling (b)): fit a
//! family's unarmed objective first, and arm its prior once, only on typed
//! evidence from that fit.

use super::*;

use gam_problem::jeffreys_arming::JeffreysArmingEvidence;
use gam_solve::constrained_posterior::{ConePropernessEvidence, ConstrainedPosteriorMomentStatus};
use gam_solve::model_types::UnifiedFitResult;

/// A family whose Jeffreys/Firth prior arms on evidence rather than by
/// declaration.
///
/// [`CustomFamily::joint_jeffreys_term_required`] reports whether THIS instance
/// is armed. A family with a separation or under-identification regime
/// implements this trait so the lifecycle can build both members, and routes its
/// fit through [`fit_custom_family_arming_on_evidence`].
pub trait JeffreysArming: CustomFamily + Clone {
    /// This family with its Jeffreys/Firth prior disarmed (`None`), or armed on
    /// the typed evidence the unarmed fit refused or declined with. A family whose
    /// measured span depends on why it armed, such as the ray it was descending,
    /// reads that evidence here.
    fn with_jeffreys_armed(&self, evidence: Option<&JeffreysArmingEvidence>) -> Self;
}

/// Fit `family` unarmed, and refit it armed once, only when that fit's own
/// evidence says the unarmed objective has no finite stationary point with
/// positive-definite information on the identified span.
///
/// - An unarmed fit that certifies with no evidence is returned as it is, so a
///   clean fit IS the unarmed objective's fit.
/// - A refusal carrying [`CustomFamilyError::jeffreys_arming_evidence`] arms the
///   refit from the caller's specs. There is no certified mode to start from. A
///   refusal without evidence is returned unchanged.
/// - A certified fit whose cone-truncated posterior is proved improper arms the
///   refit, warm-started from the unarmed fit's coefficients and smoothing
///   strengths.
///
/// The armed fit publishes its evidence on
/// `FitArtifacts::jeffreys_arming_evidence`.
pub fn fit_custom_family_arming_on_evidence<F: JeffreysArming + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, CustomFamilyError> {
    fit_custom_family_arming_on_evidence_with_rho_prior(
        family,
        specs,
        options,
        gam_problem::RhoPrior::Flat,
    )
}

/// [`fit_custom_family_arming_on_evidence`] under a prior on the log smoothing
/// strengths. The unarmed fit and the armed refit select their strengths under
/// the same prior.
pub fn fit_custom_family_arming_on_evidence_with_rho_prior<
    F: JeffreysArming + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_prior: gam_problem::RhoPrior,
) -> Result<UnifiedFitResult, CustomFamilyError> {
    arm_on_evidence(
        |arming| match arming {
            Arming::Unarmed => fit_custom_family_with_rho_prior(
                &family.with_jeffreys_armed(None),
                specs,
                options,
                rho_prior.clone(),
            ),
            Arming::Armed { evidence, unarmed } => {
                let warm_specs = unarmed
                    .map(|fit| warm_started_specs(specs, fit))
                    .transpose()?;
                fit_custom_family_with_rho_prior(
                    &family.with_jeffreys_armed(Some(evidence)),
                    warm_specs.as_deref().unwrap_or(specs),
                    options,
                    rho_prior.clone(),
                )
            }
        },
        |fit| fit,
        CustomFamilyError::jeffreys_arming_evidence,
    )
}

/// Which member of a family's objective one run of [`arm_on_evidence`] fits.
pub enum Arming<'a, T> {
    /// The unarmed objective, which every lifecycle fits first.
    Unarmed,
    /// The armed objective, on the evidence the unarmed run produced. `unarmed`
    /// is that run's result when it certified (its cone posterior was proved
    /// improper), for a warm start; a refused unarmed run leaves it `None`.
    Armed {
        evidence: &'a JeffreysArmingEvidence,
        unarmed: Option<&'a T>,
    },
}

/// The arming lifecycle over a whole fitting route (#979, #3164).
///
/// `run` fits the route once under the member it is handed: every inner solve,
/// outer search and final fit of that run on the same member, so the route
/// solves one objective whichever driver path it takes. The unarmed run goes
/// first; it is refit armed once, only on typed evidence:
///
/// - an unarmed result that certifies with no evidence is returned as it is;
/// - a refusal whose `refusal_evidence` is `Some` arms the refit, with no
///   certified mode to start from; any other refusal is returned unchanged;
/// - a certified result whose cone-truncated posterior is proved improper arms
///   the refit, handed that result for a warm start.
///
/// `fit` reads a result's fitted model. The armed result publishes its evidence
/// on `FitArtifacts::jeffreys_arming_evidence`.
pub fn arm_on_evidence<T, E>(
    mut run: impl FnMut(Arming<'_, T>) -> Result<T, E>,
    fit: impl Fn(&mut T) -> &mut UnifiedFitResult,
    refusal_evidence: impl Fn(&E) -> Option<JeffreysArmingEvidence>,
) -> Result<T, E> {
    let (evidence, unarmed) = match run(Arming::Unarmed) {
        Ok(mut result) => match improper_cone_posterior_evidence(fit(&mut result)) {
            None => return Ok(result),
            Some(evidence) => (evidence, Some(result)),
        },
        Err(refusal) => match refusal_evidence(&refusal) {
            None => return Err(refusal),
            Some(evidence) => (evidence, None),
        },
    };
    log::debug!(
        "[custom-family] arming the Jeffreys/Firth prior on the unarmed fit's evidence: \
         {evidence:?}; warm start from the unarmed mode: {}",
        unarmed.is_some(),
    );
    let mut armed = run(Arming::Armed {
        evidence: &evidence,
        unarmed: unarmed.as_ref(),
    })?;
    fit(&mut armed).artifacts.jeffreys_arming_evidence = Some(evidence);
    Ok(armed)
}

/// The face evidence a certified fit carries: its constrained mode's
/// cone-truncated posterior proved improper (#979). A declined posterior and a
/// certified boundary-mode law both keep that evidence, so both arm.
fn improper_cone_posterior_evidence(fit: &UnifiedFitResult) -> Option<JeffreysArmingEvidence> {
    let geometry = fit.geometry.as_ref()?.constrained_posterior.as_ref()?;
    let decline = match &geometry.moment_status {
        ConstrainedPosteriorMomentStatus::Available => return None,
        ConstrainedPosteriorMomentStatus::Declined(decline)
        | ConstrainedPosteriorMomentStatus::BoundaryApproximation { decline, .. } => decline,
    };
    let ConePropernessEvidence::Certificate(certificate) = &decline.properness else {
        return None;
    };
    (certificate.is_proper() == Some(false)).then(|| {
        JeffreysArmingEvidence::ImproperConePosterior {
            ambient_negative: certificate.ambient_inertia.negative,
            reduced_negative: certificate.reduced_inertia.negative,
            lineality_negative: certificate.lineality_inertia.negative,
            copositive_minimum: certificate.copositive_minimum,
        }
    })
}

/// The caller's specs, seeded with a certified fit's raw-coordinate coefficients
/// and per-penalty smoothing strengths.
fn warm_started_specs(
    specs: &[ParameterBlockSpec],
    fit: &UnifiedFitResult,
) -> Result<Vec<ParameterBlockSpec>, CustomFamilyError> {
    if fit.blocks.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "Jeffreys arming warm start: the unarmed fit has {} blocks, the specs {}",
                fit.blocks.len(),
                specs.len()
            ),
        });
    }
    specs
        .iter()
        .zip(&fit.blocks)
        .map(|(spec, block)| {
            if block.beta.len() != spec.design.ncols() || block.lambdas.len() != spec.penalties.len()
            {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "Jeffreys arming warm start: block '{}' fitted {} coefficients and {} \
                         strengths, its spec has {} columns and {} penalties",
                        spec.name,
                        block.beta.len(),
                        block.lambdas.len(),
                        spec.design.ncols(),
                        spec.penalties.len()
                    ),
                });
            }
            let mut warm = spec.clone();
            warm.initial_beta = Some(block.beta.clone());
            warm.initial_log_lambdas = block.lambdas.mapv(f64::ln);
            Ok(warm)
        })
        .collect()
}
