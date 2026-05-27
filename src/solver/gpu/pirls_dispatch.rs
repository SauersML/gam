//! Stage 3.3 → CPU PIRLS bridge.
//!
//! `should_use_gpu_pirls_loop` (in [`crate::gpu::policy`]) decides *whether*
//! a fit should be routed to the device-resident `pirls_loop_on_stream`
//! entry from `pirls_gpu.rs`. This module decides *how* the CPU PIRLS entry
//! describes itself to that admission check: it carries the (response, link)
//! → `PirlsRowFamily` mapping, the curvature-mode mapping, and the strict
//! "is this fit even shaped like something the GPU loop knows how to do"
//! preconditions that gate the dispatch before any device call.
//!
//! Keeping the mapping here (rather than inlining it in
//! `fit_model_for_fixed_rho_with_adaptive_kkt`) lets the CPU PIRLS entry
//! treat the dispatch as a single yes/no decision and lets the parity tests
//! reach the mapper directly without standing up a full fit.

use crate::gpu::policy::{PirlsLoopAdmission, PirlsLoopCurvatureKind, PirlsLoopFamilyKind};
use crate::types::{InverseLink, LikelihoodSpec, LinkFunction, StandardLink, ResponseFamily};

/// Result of mapping the engine-level `(ResponseFamily, InverseLink)` pair
/// to the six built-in JIT-cached families the Stage 3.3 PIRLS loop can
/// evaluate without going through a Level-B raw-body NVRTC compile.
///
/// `None` means the fit must stay on the CPU LM loop: either the response /
/// link combination is one of the engine's custom variants (Sas, Mixture,
/// LatentCLogLog, BetaLogistic, Tweedie, NegativeBinomial, Beta,
/// RoystonParmar) for which Stage 3.3 has no built-in row kernel, or the
/// response is supported but the link does not match a built-in pairing
/// (e.g. Poisson with Identity link).
pub fn pirls_loop_family_for(spec: &LikelihoodSpec) -> Option<PirlsLoopFamilyKind> {
    let link = match &spec.link {
        InverseLink::Standard(lf) => *lf,
        // Custom / blended inverse links have no Stage 3.3 row kernel —
        // they require Stage 6 Level B JIT, which the CPU LM loop calls
        // through different machinery.
        _ => return None,
    };
    match (&spec.response, link) {
        (ResponseFamily::Binomial, StandardLink::Logit) => {
            Some(PirlsLoopFamilyKind::BernoulliLogit)
        }
        (ResponseFamily::Binomial, StandardLink::Probit) => {
            Some(PirlsLoopFamilyKind::BernoulliProbit)
        }
        (ResponseFamily::Binomial, StandardLink::CLogLog) => {
            Some(PirlsLoopFamilyKind::BernoulliCLogLog)
        }
        (ResponseFamily::Poisson, StandardLink::Log) => Some(PirlsLoopFamilyKind::PoissonLog),
        (ResponseFamily::Gaussian, StandardLink::Identity) => {
            Some(PirlsLoopFamilyKind::GaussianIdentity)
        }
        (ResponseFamily::Gamma, StandardLink::Log) => Some(PirlsLoopFamilyKind::GammaLog),
        // Every other pairing is either not in the JIT-cache set or is a
        // canonical-pair the row kernels do not currently support.
        _ => None,
    }
}

/// Curvature surface the GPU loop should use given the (family) mapping
/// and the CPU PIRLS loop's preferred curvature.
///
/// Stage 3.3 supports both Fisher and Observed kernels. The CPU LM loop
/// uses observed information by default for non-canonical Bernoulli /
/// Gamma-log families and Fisher elsewhere; we mirror that decision here.
pub fn pirls_loop_curvature_for(family: PirlsLoopFamilyKind) -> PirlsLoopCurvatureKind {
    match family {
        PirlsLoopFamilyKind::BernoulliProbit | PirlsLoopFamilyKind::BernoulliCLogLog => {
            PirlsLoopCurvatureKind::Observed
        }
        PirlsLoopFamilyKind::BernoulliLogit
        | PirlsLoopFamilyKind::PoissonLog
        | PirlsLoopFamilyKind::GaussianIdentity
        | PirlsLoopFamilyKind::GammaLog => PirlsLoopCurvatureKind::Fisher,
    }
}

/// Detect whether the CUDA runtime is initialised on this host. The probe
/// underneath returns `None` on every non-Linux target, so the function
/// works unconditionally — no `target_os` gate needed here.
pub fn gpu_runtime_available() -> bool {
    crate::gpu::runtime::GpuRuntime::is_available()
}

/// Strict admission shape for the Stage 3.3 PIRLS loop, computed from the
/// `(response, link)` spec and the active design shape `(n, p)`. Returns
/// `None` when the family / link is not in the JIT-cached set so the
/// caller skips both the GPU dispatch and the runtime probe.
pub fn admission_for(spec: &LikelihoodSpec, n: usize, p: usize) -> Option<PirlsLoopAdmission> {
    let family = pirls_loop_family_for(spec)?;
    let curvature = pirls_loop_curvature_for(family);
    Some(PirlsLoopAdmission {
        n,
        p,
        family: Some(family),
        curvature,
        gpu_available: gpu_runtime_available(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LikelihoodSpec, MixtureLinkState};
    use ndarray::Array1;

    fn dummy_mixture_state() -> MixtureLinkState {
        // K=2 components with the free logit at 0 — softmax weights are uniform.
        MixtureLinkState {
            components: vec![
                crate::types::LinkComponent::Logit,
                crate::types::LinkComponent::Probit,
            ],
            rho: Array1::from(vec![0.0_f64]),
            pi: Array1::from(vec![0.5_f64, 0.5_f64]),
        }
    }

    #[test]
    fn maps_six_canonical_built_in_pairings() {
        for (spec, want) in [
            (
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Logit),
                ),
                PirlsLoopFamilyKind::BernoulliLogit,
            ),
            (
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Probit),
                ),
                PirlsLoopFamilyKind::BernoulliProbit,
            ),
            (
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::CLogLog),
                ),
                PirlsLoopFamilyKind::BernoulliCLogLog,
            ),
            (
                LikelihoodSpec::new(
                    ResponseFamily::Poisson,
                    InverseLink::Standard(StandardLink::Log),
                ),
                PirlsLoopFamilyKind::PoissonLog,
            ),
            (
                LikelihoodSpec::gaussian_identity(),
                PirlsLoopFamilyKind::GaussianIdentity,
            ),
            (
                LikelihoodSpec::new(
                    ResponseFamily::Gamma,
                    InverseLink::Standard(StandardLink::Log),
                ),
                PirlsLoopFamilyKind::GammaLog,
            ),
        ] {
            assert_eq!(pirls_loop_family_for(&spec), Some(want), "for {:?}", spec);
        }
    }

    #[test]
    fn declines_unsupported_response_link_pairings() {
        // Custom / blended links → no Stage 3.3 kernel.
        let mixture_state = dummy_mixture_state();
        assert_eq!(
            pirls_loop_family_for(&LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Mixture(mixture_state),
            )),
            None
        );
        // Poisson + Identity is not in the JIT-cache set.
        assert_eq!(
            pirls_loop_family_for(&LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Identity),
            )),
            None
        );
        // Tweedie has no canonical pairing wired up.
        assert_eq!(
            pirls_loop_family_for(&LikelihoodSpec::new(
                ResponseFamily::Tweedie { p: 1.5 },
                InverseLink::Standard(StandardLink::Log),
            )),
            None
        );
    }

    #[test]
    fn non_canonical_bernoulli_links_request_observed_curvature() {
        assert_eq!(
            pirls_loop_curvature_for(PirlsLoopFamilyKind::BernoulliProbit),
            PirlsLoopCurvatureKind::Observed
        );
        assert_eq!(
            pirls_loop_curvature_for(PirlsLoopFamilyKind::BernoulliCLogLog),
            PirlsLoopCurvatureKind::Observed
        );
        assert_eq!(
            pirls_loop_curvature_for(PirlsLoopFamilyKind::BernoulliLogit),
            PirlsLoopCurvatureKind::Fisher
        );
    }

    #[test]
    fn admission_is_none_for_unmapped_family() {
        let mixture_state = dummy_mixture_state();
        let spec = LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Mixture(mixture_state),
        );
        assert!(admission_for(&spec, 80_000, 44).is_none());
    }
}
