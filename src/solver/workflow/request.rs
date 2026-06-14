use super::*;

#[derive(Clone, Debug)]
pub struct LinkWiggleConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
}


/// Configuration for the second-stage binomial-mean wiggle fit appended to a
/// standard pilot. The blockwise refit options live inside this struct so the
/// pilot config (`link_kind` + `wiggle`) and its required `refit_options` can
/// never disagree: either the whole standard-wiggle request is `Some`, or it
/// is `None`. The previous shape had two sibling `Option` fields on
/// `StandardFitRequest`, which allowed the materialize path to construct an
/// inconsistent state (#320: linkwiggle config without blockwise options).
#[derive(Clone)]
pub struct StandardBinomialWiggleConfig {
    pub link_kind: InverseLink,
    pub wiggle: LinkWiggleConfig,
    pub refit_options: BlockwiseFitOptions,
}


pub struct StandardFitRequest<'a> {
    pub data: Array2<f64>,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
    pub spec: TermCollectionSpec,
    pub family: LikelihoodSpec,
    pub options: FitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub wiggle: Option<StandardBinomialWiggleConfig>,
    pub coefficient_groups: Vec<CoefficientGroupSpec>,
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,
    pub latent_coord: Option<StandardLatentCoordConfig>,
    #[doc(hidden)]
    pub _marker: std::marker::PhantomData<&'a ()>,
}


pub struct GaussianLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: GaussianLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}


pub struct BinomialLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BinomialLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}


pub struct DispersionLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: DispersionGlmLocationScaleTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}


pub struct SurvivalLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub optimize_inverse_link: bool,
    /// See [`crate::families::custom_family::BlockwiseFitOptions::cache_session`].
    /// Threaded into the internally constructed `BlockwiseFitOptions` by
    /// `fit_survival_location_scale_model`.
    pub cache_session: Option<std::sync::Arc<crate::cache::Session>>,
}


pub struct SurvivalTransformationFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalTransformationTermSpec,
    /// See [`crate::families::custom_family::BlockwiseFitOptions::cache_session`].
    /// Threaded into the internally constructed `BlockwiseFitOptions` by
    /// `fit_survival_transformation_model`.
    pub cache_session: Option<std::sync::Arc<crate::cache::Session>>,
}


#[derive(Clone)]
pub struct SurvivalTransformationTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub covariate_spec: TermCollectionSpec,
    pub covariate_offset: Array1<f64>,
    pub baseline_cfg: crate::families::survival_construction::SurvivalBaselineConfig,
    pub likelihood_mode: crate::families::survival_construction::SurvivalLikelihoodMode,
    pub time_anchor: f64,
    pub time_build: crate::families::survival_construction::SurvivalTimeBuildOutput,
    pub timewiggle: Option<LinkWiggleFormulaSpec>,
    pub weibull_seed: Option<(f64, f64)>,
    pub ridge_lambda: f64,
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,
}
pub struct BernoulliMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BernoulliMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub policy: crate::resource::ResourcePolicy,
}


pub struct SurvivalMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}
pub struct LatentSurvivalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: LatentSurvivalTermSpec,
    pub frailty: FrailtySpec,
    pub options: BlockwiseFitOptions,
}


pub struct LatentBinaryFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: LatentBinaryTermSpec,
    pub frailty: FrailtySpec,
    pub options: BlockwiseFitOptions,
}


pub struct TransformationNormalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub response: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
    pub covariate_spec: TermCollectionSpec,
    pub config: TransformationNormalConfig,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub warm_start: Option<TransformationWarmStart>,
}
pub enum FitRequest<'a> {
    Standard(StandardFitRequest<'a>),
    GaussianLocationScale(GaussianLocationScaleFitRequest<'a>),
    BinomialLocationScale(BinomialLocationScaleFitRequest<'a>),
    DispersionLocationScale(DispersionLocationScaleFitRequest<'a>),
    SurvivalLocationScale(SurvivalLocationScaleFitRequest<'a>),
    SurvivalTransformation(SurvivalTransformationFitRequest<'a>),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest<'a>),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitRequest<'a>),
    LatentSurvival(LatentSurvivalFitRequest<'a>),
    LatentBinary(LatentBinaryFitRequest<'a>),
    TransformationNormal(TransformationNormalFitRequest<'a>),
}


/// Mechanical surface every `FitRequest` variant must expose. Centralising
/// these here makes the per-variant logic live with the variant's data and
/// turns the `FitRequest` impl block's case-analysis into a single dispatch
/// macro, so adding a new family is one struct + one trait impl + one
/// variant arm in [`family_dispatch!`]; the compiler then refuses to
/// silently leave a ladder unupdated.
pub trait FamilyFitRequest {
    /// Stable, short identifier ("standard", "survival-marginal-slope", …).
    /// Used as a `family={tag}` segment of the persistent warm-start key
    /// and as the structured tag in `[CACHE] …` log lines.
    const TAG: &'static str;

    /// Instance accessor for [`Self::TAG`] so callers holding a `&dyn` /
    /// generic reference (or going through [`family_dispatch!`]) can avoid
    /// the turbofish.
    fn tag(&self) -> &'static str {
        Self::TAG
    }

    /// Row count of the response / design (n).
    fn n_obs(&self) -> usize;

    /// Column count of the user-facing design matrix (p_data, before
    /// per-family basis expansion). Used in the cache-key `dims=N×D`
    /// segment for the shape-prefix lookup.
    fn n_cols(&self) -> usize;

    /// Family-shape hash inputs (link kind, baseline, frailty, etc.) —
    /// everything that changes the coefficient layout. Feeds the exact
    /// cache key.
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter);

    /// Data-independent seed-key hash inputs. Same structural identifiers
    /// as `write_shape_hash` but with the row count deliberately *excluded*
    /// so cross-validation folds / hyperparameter sweeps share a prefix.
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter);

    /// Attach the primary persistent warm-start session. Variants like
    /// `Standard` that open their own session inside the outer optimizer
    /// (see `solver/estimate.rs:2701`) implement this as a `drop(session)`
    /// no-op to avoid double-checkpointing on the same fingerprint.
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>);

    /// Attach a mirror session that receives a broadcast copy of the final
    /// `finalize` write under the seed-prefix keyspace. Variants without
    /// a mirror channel implement this as a `drop(mirror)` no-op.
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>);
}


/// Enumerates every `FitRequest` variant in **one** place. Use as
/// `family_dispatch!(expr, r => expr_using_r)`; expands to the full
/// 10-arm match. The compiler enforces exhaustiveness here too, so adding
/// a new variant produces a single hard error at this site (rather than
/// silently routing through an `_` arm of a hand-written ladder).
macro_rules! family_dispatch {
    ($scrutinee:expr, $req:ident => $body:expr) => {
        match $scrutinee {
            FitRequest::Standard($req) => $body,
            FitRequest::GaussianLocationScale($req) => $body,
            FitRequest::BinomialLocationScale($req) => $body,
            FitRequest::DispersionLocationScale($req) => $body,
            FitRequest::SurvivalLocationScale($req) => $body,
            FitRequest::SurvivalTransformation($req) => $body,
            FitRequest::BernoulliMarginalSlope($req) => $body,
            FitRequest::SurvivalMarginalSlope($req) => $body,
            FitRequest::LatentSurvival($req) => $body,
            FitRequest::LatentBinary($req) => $body,
            FitRequest::TransformationNormal($req) => $body,
        }
    };
}


impl<'a> FamilyFitRequest for StandardFitRequest<'a> {
    const TAG: &'static str = "standard";
    fn n_obs(&self) -> usize {
        self.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("standard");
        h.write_str(&format!("{:?}", self.family));
        h.write_usize(self.y.len());
        h.write_usize(self.data.ncols());
        // Topology identity (#869): raw `data.ncols()` is blind to the smooth
        // basis, so `s(..., type=AUTO)` candidates fit on the same data would
        // otherwise share one warm-start key and cross-seed each other with
        // incompatible β/ρ. Fold the term-collection structural shape in so
        // each candidate keys distinctly while same-topology refits still hit.
        self.spec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("standard-seed");
        h.write_str(&format!("{:?}", self.family));
        h.write_usize(self.data.ncols());
        // Same topology disambiguation for the data-independent seed key: a
        // sphere candidate must not seed a torus candidate across folds.
        self.spec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        // The standard REML path opens its own session inside the outer
        // optimizer via `reml_state.outer_cache_session()` (see
        // `solver/estimate.rs:2701`). Accepting another one here would
        // double-checkpoint the same fingerprint — drop the redundant arc.
        drop(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        // Same reasoning: the outer optimizer handles its own finalize.
        drop(mirror);
    }
}


impl<'a> FamilyFitRequest for GaussianLocationScaleFitRequest<'a> {
    const TAG: &'static str = "gaussian-location-scale";
    fn n_obs(&self) -> usize {
        self.spec.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("gauss-ls");
        h.write_usize(self.spec.y.len());
        h.write_usize(self.data.ncols());
        // Topology identity (#869, extended): the location-scale outer
        // cache session keys on this hash; an `ExactFinal` hit on it
        // short-circuits the whole fit (see
        // `outer_strategy::classify_cache_entry_for_outer`). Raw
        // `data.ncols()` is blind to the smooth basis, so two AUTO-topology
        // candidates (sphere vs euclidean) on the same data with the same
        // penalty count would collide on one key and one would return the
        // other's converged ρ as its own result. Fold each block's
        // term-collection structural shape in so each candidate keys
        // distinctly while same-topology refits still hit.
        self.spec.meanspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("gauss-ls-seed");
        h.write_usize(self.data.ncols());
        self.spec.meanspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FamilyFitRequest for BinomialLocationScaleFitRequest<'a> {
    const TAG: &'static str = "binomial-location-scale";
    fn n_obs(&self) -> usize {
        self.spec.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("binom-ls");
        h.write_usize(self.spec.y.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.link_kind));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.thresholdspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("binom-ls-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.link_kind));
        self.spec.thresholdspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FamilyFitRequest for DispersionLocationScaleFitRequest<'a> {
    const TAG: &'static str = "dispersion-location-scale";
    fn n_obs(&self) -> usize {
        self.spec.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("disp-ls");
        h.write_str(self.spec.kind.family_tag());
        h.write_usize(self.spec.y.len());
        h.write_usize(self.data.ncols());
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.meanspec.write_structural_shape_hash(h);
        self.spec.log_dispspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("disp-ls-seed");
        h.write_str(self.spec.kind.family_tag());
        h.write_usize(self.data.ncols());
        self.spec.meanspec.write_structural_shape_hash(h);
        self.spec.log_dispspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FamilyFitRequest for SurvivalLocationScaleFitRequest<'a> {
    const TAG: &'static str = "survival-location-scale";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-ls");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.inverse_link));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.thresholdspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-ls-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.inverse_link));
        self.spec.thresholdspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        // Request-level slot is mirrored into the spec slot so the family
        // fit fn sees the session when it constructs its internal
        // BlockwiseFitOptions.
        if self.cache_session.is_none() {
            self.cache_session = Some(session.clone());
        }
        if self.spec.cache_session.is_none() {
            self.spec.cache_session = Some(session);
        }
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.spec.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FamilyFitRequest for SurvivalTransformationFitRequest<'a> {
    const TAG: &'static str = "survival-transformation";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-tn");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.likelihood_mode));
        h.write_str(&self.spec.time_build.basisname);
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.covariate_spec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-tn-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.likelihood_mode));
        h.write_str(&self.spec.time_build.basisname);
        self.spec.covariate_spec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        // SurvivalTransformation uses WorkingModelPirlsOptions (different
        // mechanism) for its inner solve; the cache session is parked at
        // the request level so future wiring through
        // `persistent_survival_transformation_key` can be unified. For now
        // that path's own exact-match warm-start fires independently.
        self.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        // The path's own `persistent_survival_transformation_key`
        // mechanism handles exact-match warm-start; mirror finalize would
        // be a duplicate — drop the unused arc.
        drop(mirror);
    }
}


impl<'a> FamilyFitRequest for BernoulliMarginalSlopeFitRequest<'a> {
    const TAG: &'static str = "bernoulli-marginal-slope";
    fn n_obs(&self) -> usize {
        self.spec.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("bern-ms");
        h.write_usize(self.spec.y.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.base_link));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.marginalspec.write_structural_shape_hash(h);
        self.spec.logslopespec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("bern-ms-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.base_link));
        self.spec.marginalspec.write_structural_shape_hash(h);
        self.spec.logslopespec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FamilyFitRequest for SurvivalMarginalSlopeFitRequest<'a> {
    const TAG: &'static str = "survival-marginal-slope";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-ms");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.base_link));
        h.write_str(&format!("{:?}", self.spec.frailty));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.marginalspec.write_structural_shape_hash(h);
        self.spec.logslopespec.write_structural_shape_hash(h);
        match self.spec.logslopespecs.as_ref() {
            Some(specs) => {
                h.write_bool(true);
                h.write_usize(specs.len());
                for spec in specs {
                    spec.write_structural_shape_hash(h);
                }
            }
            None => h.write_bool(false),
        }
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-ms-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.base_link));
        h.write_str(&format!("{:?}", self.spec.frailty));
        self.spec.marginalspec.write_structural_shape_hash(h);
        self.spec.logslopespec.write_structural_shape_hash(h);
        match self.spec.logslopespecs.as_ref() {
            Some(specs) => {
                h.write_bool(true);
                h.write_usize(specs.len());
                for spec in specs {
                    spec.write_structural_shape_hash(h);
                }
            }
            None => h.write_bool(false),
        }
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FamilyFitRequest for LatentSurvivalFitRequest<'a> {
    const TAG: &'static str = "latent-survival";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("lat-surv");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.frailty));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.meanspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("lat-surv-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.frailty));
        self.spec.meanspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FamilyFitRequest for LatentBinaryFitRequest<'a> {
    const TAG: &'static str = "latent-binary";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("lat-bin");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.frailty));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.meanspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("lat-bin-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.frailty));
        self.spec.meanspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FamilyFitRequest for TransformationNormalFitRequest<'a> {
    const TAG: &'static str = "transformation-normal";
    fn n_obs(&self) -> usize {
        self.response.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("tn");
        h.write_usize(self.response.len());
        h.write_usize(self.data.ncols());
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.covariate_spec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("tn-seed");
        h.write_usize(self.data.ncols());
        self.covariate_spec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}


impl<'a> FitRequest<'a> {
    /// Short stable string identifying the family variant. Used as one
    /// segment of the warm-start cache key — every fit of this family
    /// shape shares the same family tag, so a hierarchical lookup can
    /// drop trailing key segments to find near-match seeds from prior
    /// fits of the same family on related data.
    pub fn family_tag(&self) -> &'static str {
        family_dispatch!(self, r => r.tag())
    }

    /// Deterministic warm-start cache key for this request.
    ///
    /// Layout (`/`-delimited so the suffix can be peeled for graceful
    /// degradation in future hierarchical lookups):
    ///   `v{lib}/family={tag}/dims=N×D/shape={family-shape-hash}`
    ///
    /// Inputs included: library version, family tag, dataset shape, and a
    /// family-specific shape hash covering anything that changes the basis
    /// dimension or coefficient layout (formula skeleton, baseline kind,
    /// link kind, etc.). Inputs deliberately *excluded*: anything that
    /// shifts the optimum but doesn't reshape the parameter vector
    /// (convergence tolerances, ridge, etc.) — those go in the
    /// finer-grained suffix so a near-match can still be a useful seed.
    pub fn cache_key(&self) -> String {
        // Family-shape hash: a 64-bit truncation of the canonical
        // SHA-256 `Fingerprinter` digest that
        // captures dimensions / formula structure unique to this family
        // variant. Two fits with identical family-shape hashes have
        // compatible θ shapes and can warm-start from each other.
        let mut shape = crate::cache::Fingerprinter::new();
        family_dispatch!(self, r => r.write_shape_hash(&mut shape));
        let shape_hash = shape.finish_hex();
        let (nrows, ncols) = family_dispatch!(self, r => (r.n_obs(), r.n_cols()));
        format!(
            "{}/family={}/dims={}x{}/shape={}",
            crate::solver::persistent_warm_start::cache_schema_tag(),
            self.family_tag(),
            nrows,
            ncols,
            shape_hash,
        )
    }

    /// Data-independent cache key suitable as a *seed* lookup when the
    /// exact-key cache misses. Two fits with the same family / link /
    /// baseline / column shape share this key even if their row counts
    /// differ (cross-validation folds, hyperparameter sweeps, anything
    /// re-fitting structurally identical models on related data).
    ///
    /// Save always goes to [`Self::cache_key`] (exact). The dispatcher
    /// looks up this prefix only when the exact key has no entry — so a
    /// near-match never silently overrides a perfect match.
    pub fn cache_seed_key(&self) -> String {
        let mut shape = crate::cache::Fingerprinter::new();
        family_dispatch!(self, r => r.write_seed_hash(&mut shape));
        format!(
            "{}/family={}/seed/{}",
            crate::solver::persistent_warm_start::cache_schema_tag(),
            self.family_tag(),
            shape.finish_hex(),
        )
    }

    /// Attach a mirror cache session that receives a broadcast copy of
    /// the final-result `finalize` write. Used by the dispatcher to write
    /// the converged ρ to the data-independent seed prefix keyspace so a
    /// future fit with related-but-not-identical structure can warm-start
    /// from this run. Checkpoints are mirrored through the same rate-limited
    /// session path as the primary cache, so interrupted runs can still seed
    /// later related fits instead of waiting for a successful final result.
    pub fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        family_dispatch!(self, r => <_ as FamilyFitRequest>::attach_cache_mirror(r, mirror))
    }

    /// Attach a warm-start cache session to this request.
    ///
    /// Threads the session into the variant's BlockwiseFitOptions
    /// (`request.options.cache_session`) or top-level `cache_session`
    /// field, whichever is the variant's natural slot. Idempotent —
    /// existing sessions are not overwritten so callers can pre-attach.
    pub fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        family_dispatch!(self, r => <_ as FamilyFitRequest>::attach_cache_session(r, session))
    }
}


pub struct StandardFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
    pub saved_link_state: FittedLinkState,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
}


pub struct SurvivalLocationScaleFitResult {
    pub fit: SurvivalLocationScaleTermFitResult,
    pub inverse_link: InverseLink,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
}


pub struct SurvivalTransformationFitResult {
    pub fit: UnifiedFitResult,
    pub resolvedspec: TermCollectionSpec,
    pub baseline_cfg: crate::families::survival_construction::SurvivalBaselineConfig,
    pub likelihood_mode: crate::families::survival_construction::SurvivalLikelihoodMode,
    /// Persistable snapshot of the time basis used during the fit. Replaces
    /// six previously flat fields (basisname / degree / knots / keep_cols /
    /// smooth_lambda / anchor) so the FFI save path consumes a single
    /// source-of-truth value rather than threading siblings independently.
    pub time_basis: crate::families::survival_construction::SavedSurvivalTimeBasis,
    pub time_base_ncols: usize,
    pub baseline_timewiggle: Option<TimeWiggleBlockInput>,
}

pub enum FitResult {
    Standard(StandardFitResult),
    GaussianLocationScale(GaussianLocationScaleFitResult),
    BinomialLocationScale(BinomialLocationScaleFitResult),
    DispersionLocationScale(DispersionLocationScaleFitResult),
    SurvivalLocationScale(SurvivalLocationScaleFitResult),
    SurvivalTransformation(SurvivalTransformationFitResult),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitResult),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitResult),
    LatentSurvival(LatentSurvivalTermFitResult),
    LatentBinary(LatentBinaryTermFitResult),
    TransformationNormal(TransformationNormalFitResult),
    /// Exact O(n) state-space cubic/linear/quintic smoothing-spline scan
    /// (#1030/#1034). A scan-bearing model IS a Gaussian-identity model with a
    /// different (exact) representation: rather than a dense design + coefficient
    /// vector it carries the Durbin–Koopman smoother posterior directly (knots,
    /// smoothed states, pointwise variances, σ², log λ, exact diffuse-REML EDF,
    /// and an exact per-row `predict`). Library callers that want the fitted
    /// posterior get it here without paying the dense O(n·k²)+O(k³) route; the
    /// CLI/FFI save paths build the persistence payload from the same
    /// `SplineScanFit` via `assemble_spline_scan_payload`.
    SplineScan(crate::solver::spline_scan::SplineScanFit),
    /// O(n log n) multiresolution residual-cascade smooth (#1032). UNLIKE the
    /// 1-D scan, the cascade is NOT the same posterior as the Duchon/Matérn term
    /// it stands in for (a different finite basis — the multilevel Wendland
    /// frame), so it is never a silent swap: this variant is produced only when
    /// the structural detector [`residual_cascade_fast_path`] fires on an
    /// eligible scattered-low-d Gaussian fit past the dense-kernel cliff AND the
    /// in-cascade quasi-uniformity guard certifies the metric; every other shape
    /// (and a rejected metric) falls through to the dense `fit_model` path. The
    /// cascade-bearing model carries the
    /// [`ResidualCascadeFit`](crate::solver::residual_cascade::ResidualCascadeFit)
    /// directly — knots-free nested geometry, coefficients, the factored
    /// precision, and an exact per-row `predict`; the CLI/FFI save paths build
    /// the persistence payload from its `to_state` snapshot.
    ResidualCascade(crate::solver::residual_cascade::ResidualCascadeFit),
}


/// Result of a dispersion-channel GAMLSS location-scale fit (#913). Wraps the
/// shared two-block [`BlockwiseTermFitResult`] (mean + log-precision designs
/// and coefficients) plus the family kind so the save path can stamp the right
/// likelihood. These families have no link-wiggle and no response
/// standardization, so the result is a thin wrapper.
pub struct DispersionLocationScaleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub kind: DispersionFamilyKind,
}

/// Out-of-fold Stage-1 latent score and its score-influence Jacobian for a
/// CTN → marginal-slope chain. `z_oof` (length n) replaces the in-sample `z`
/// the Stage-2 model consumes; `jac_oof` (n × p₁) is fed to the Stage-2 spec's
/// `score_influence_jacobian` so the joint solve absorbs the realized leakage
/// directions `Z_infl = diag(s_f·β̂₀)·J`.
pub struct CrossFitScoreCalibration {
    pub z_oof: Array1<f64>,
    pub jac_oof: Array2<f64>,
}


/// Internal recipe describing the CTN Stage-1 fit that produced a Stage-2 `z`
/// column. This is in-process plumbing — never a CLI flag, env var, or feature
/// gate. The orchestration layer populates [`FitConfig::ctn_stage1`] when (and
/// only when) the marginal-slope `z` was generated by a transformation-normal
/// Stage-1 fit; its presence is the sole auto-enable signal for cross-fitted
/// orthogonalization (design §5). When absent, Stage-2 falls back to the free
/// 1-D `score_warp` spline (which spans only the x-free leakage column).
#[derive(Clone, Debug)]
pub struct CtnStage1Recipe {
    /// Stage-1 response column name (the `y` the CTN transforms).
    pub response_column: String,
    /// Stage-1 covariate-side formula right-hand side (e.g. `"s(pc1) + s(pc2)"`),
    /// with no `~` and no response symbol. [`crossfit_score_calibration`] parses
    /// it and builds the CTN covariate basis exactly as
    /// `materialize_transformation_normal` does, then FREEZES that basis once on
    /// the full data and reuses the frozen spec for every fold's refit — so the
    /// rebuilt covariate design has an identical column geometry across folds,
    /// keeping `J`'s `p₁ = p_resp · p_cov` columns aligned (design §3).
    ///
    /// The recipe carries the formula RHS (a primitive string) rather than a
    /// resolved [`TermCollectionSpec`] because this struct is populated both via
    /// [`CtnStage1Recipe::new`] (set on [`FitConfig::ctn_stage1`], then
    /// [`fit_from_formula`]) and by the gamfit FFI marshaller
    /// (`gamfit/_calibrated_slope.py`), which can only serialize primitives over
    /// the JSON boundary — a `TermCollectionSpec` is not serializable. Freezing on
    /// the full Stage-2 data is equivalent to
    /// freezing on the Stage-1 data whenever the two stages share a frame (the
    /// calibrated-chain contract), so the column geometry still matches Stage-1.
    pub covariate_formula_rhs: String,
    /// Stage-1 CTN config (response basis degree / knot count / penalties).
    /// Its `response_num_internal_knots` is the FIXED response-basis size; the
    /// cross-fit pins it across folds so `p_resp` (and hence `p₁`) is
    /// fold-invariant (design §3).
    pub config: TransformationNormalConfig,
    /// Optional Stage-1 weight column name.
    pub weight_column: Option<String>,
    /// Optional Stage-1 offset column name.
    pub offset_column: Option<String>,
}


impl CtnStage1Recipe {
    /// Build a Stage-1 CTN recipe from the Stage-1 description. This is the public
    /// way to populate [`FitConfig::ctn_stage1`] — set it on a marginal-slope
    /// config and run [`fit_from_formula`] (the entry IS `fit_from_formula` with
    /// `ctn_stage1` set; there is no separate combined entry function). The
    /// materializer then cross-fits the CTN and installs the leakage-projection
    /// block; supplying the recipe *is* the request for orthogonalization.
    ///
    /// `response` is the Stage-1 CTN response column; `covariates` is the
    /// covariate-side formula right-hand side (e.g. `"s(pc1) + s(pc2)"` — no `~`,
    /// no response symbol). Validates both are non-empty and that `covariates`
    /// is an RHS only.
    pub fn new(
        response: &str,
        covariates: &str,
        config: TransformationNormalConfig,
        weight_column: Option<&str>,
        offset_column: Option<&str>,
    ) -> Result<Self, String> {
        let response_column = response.trim().to_string();
        if response_column.is_empty() {
            return Err("CtnStage1Recipe requires a non-empty Stage-1 response column".to_string());
        }
        let covariate_formula_rhs = covariates.trim().to_string();
        if covariate_formula_rhs.is_empty() {
            return Err(
                "CtnStage1Recipe requires a non-empty Stage-1 covariate formula RHS".to_string(),
            );
        }
        if covariate_formula_rhs.contains('~') {
            return Err(
                "CtnStage1Recipe covariates is a right-hand side only; pass 's(pc1) + s(pc2)', \
                 not 'score ~ s(pc1) + s(pc2)'"
                    .to_string(),
            );
        }
        Ok(Self {
            response_column,
            covariate_formula_rhs,
            config,
            weight_column: weight_column
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty()),
            offset_column: offset_column
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty()),
        })
    }
}
#[derive(Clone, Debug)]
pub struct FitConfig {
    /// Family: "gaussian", "binomial", "poisson", "negative-binomial",
    /// "gamma", "tweedie" (alias "tw"; variance power fixed at p = 1.5), or
    /// None for auto-detect.
    pub family: Option<String>,
    /// Fixed size/overdispersion parameter for `family="negative-binomial"`.
    pub negative_binomial_theta: Option<f64>,
    /// Link: "identity", "logit", "probit", "cloglog", "sas", "beta-logistic", or None.
    pub link: Option<String>,
    /// Whether to use flexible (wiggle-augmented) link.
    pub flexible_link: bool,
    /// Optional additive offset column for the primary linear predictor.
    pub offset_column: Option<String>,
    /// Optional additive offset column for the noise/log-scale predictor.
    pub noise_offset_column: Option<String>,
    /// Optional family-level frailty modifier.
    pub frailty: Option<FrailtySpec>,

    // Survival-specific
    /// Baseline target: "linear", "weibull", "gompertz", "gompertz-makeham".
    pub baseline_target: String,
    pub baseline_scale: Option<f64>,
    pub baseline_shape: Option<f64>,
    pub baseline_rate: Option<f64>,
    pub baseline_makeham: Option<f64>,
    /// Time basis: "ispline" or "none".
    pub time_basis: String,
    pub time_degree: usize,
    pub time_num_internal_knots: usize,
    pub time_smooth_lambda: f64,
    /// Survival likelihood mode: "location-scale", "transformation", "weibull",
    /// "marginal-slope", "latent", or "latent-binary".
    pub survival_likelihood: String,
    /// Residual distribution: "gaussian", "logistic", "gumbel".
    pub survival_distribution: String,
    pub threshold_time_k: Option<usize>,
    pub threshold_time_degree: usize,
    pub sigma_time_k: Option<usize>,
    pub sigma_time_degree: usize,

    // Location-scale (GAMLSS)
    /// If set, fit a location-scale model with this formula for the noise parameter.
    pub noise_formula: Option<String>,

    // Marginal-slope
    /// Formula for the log-slope model (survival marginal-slope or Bernoulli marginal-slope).
    pub logslope_formula: Option<String>,
    /// Column name for the z (exposure/dose) variable in marginal-slope models.
    pub z_column: Option<String>,
    /// Optional non-negative per-row training weights column.
    pub weight_column: Option<String>,
    /// Expectile asymmetry `τ ∈ (0, 1)` for `family = "expectile"`.
    ///
    /// When `family` resolves to `"expectile"` the fit minimizes the
    /// Newey–Powell asymmetric squared loss `Σ wᵢ(τ)·(yᵢ − μᵢ)²` with
    /// `wᵢ(τ) = τ` if `yᵢ > μᵢ` else `1 − τ`, tracing the conditional
    /// `τ`-expectile — the smooth analogue of the `τ`-quantile. `τ = 0.5`
    /// reduces exactly to the Gaussian-identity mean fit. The whole penalized
    /// smooth + REML `λ`-selection machinery is reused via a Least
    /// Asymmetrically Weighted Squares (LAWS) outer loop. `None` defaults to
    /// the median expectile `τ = 0.5` when the family is `"expectile"`; it is
    /// ignored for every other family. The asymmetry may also be written inline
    /// as `family = "expectile(0.9)"`, which fills this field at resolve time.
    pub expectile_tau: Option<f64>,
    /// Internal CTN Stage-1 provenance for the marginal-slope `z` column.
    ///
    /// When the marginal-slope `z` was generated by a transformation-normal
    /// Stage-1 fit, the orchestration layer fills this with the Stage-1 recipe.
    /// Its presence is the sole auto-enable signal for cross-fitted, Neyman-
    /// orthogonal score calibration (#461): the materializer cross-fits the CTN
    /// to produce out-of-fold `z` and the score-influence Jacobian `J`, replaces
    /// the raw `z` with `z_oof`, and absorbs `J` as a leakage-projection block in
    /// Stage-2. This is in-process plumbing only — there is no CLI flag, env var,
    /// or feature gate. `None` ⇒ raw `z` with the free-warp `score_warp`
    /// fallback. See [`CtnStage1Recipe`].
    pub ctn_stage1: Option<CtnStage1Recipe>,

    // Fitting options
    pub scale_dimensions: bool,
    /// Enable exact spatial adaptive regularization for standard formula fits.
    /// `None` uses the quality-first automatic policy. The current automatic
    /// policy leaves LAREG off unless explicitly requested because the
    /// optimizer's REML-selected local weights can over-regularize small
    /// high-yield spatial signals.
    pub adaptive_regularization: Option<bool>,
    pub ridge_lambda: f64,

    /// Route the fit through the transformation-normal family.  When set, the
    /// formula terms are treated as the covariate side of the transformation
    /// model and the response basis is built internally.  Incompatible with
    /// `noise_formula` and with `Surv(...)` responses.
    pub transformation_normal: bool,

    /// Enable Firth bias reduction for standard single-parameter families.
    pub firth: bool,
    /// Optional cap on the REML/LAML outer smoothing-parameter iterations for
    /// standard formula fits. `None` uses the production default.
    pub outer_max_iter: Option<usize>,

    /// GPU backend selection policy. `Auto` uses supported device kernels for
    /// large workloads, `Off` pins execution to CPU kernels, and `Force` fails
    /// loudly when a requested GPU kernel has no compiled backend.
    pub gpu_policy: crate::gpu::GpuPolicy,
    /// Optional override of the [`crate::resource::ResourcePolicy`] used when
    /// planning spatial bases (TPS / Matern / Duchon) during term construction.
    /// When `None`, the default-library policy is used.
    pub resource_policy: Option<crate::resource::ResourcePolicy>,

    /// Optional per-group metadata supplied by the caller. Fitting ignores this
    /// field; saved-model builders pass it through so deployment consumers can
    /// recover group provenance.
    pub group_metadata: Option<BTreeMap<String, JsonValue>>,

    /// Optional user-defined coefficient groups with separate precision
    /// parameters. Group-local priors, including catalog-metadata-informed
    /// Gamma precision hyperpriors, are resolved during design setup.
    pub coefficient_groups: Vec<CoefficientGroupSpec>,

    /// Optional per-existing-penalty-block Gamma(shape, rate) precision
    /// hyperpriors keyed by penalty-block label. This is the
    /// catalog-metadata-informed-prior hook for models that do not need a new
    /// user-defined coefficient group.
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,

    /// Python `gamfit.fit(..., latents={...})` configuration. This reaches
    /// the standard formula workflow as an owned latent-coordinate block:
    /// the named smooth's synthetic covariates are rebuilt from `t`, and
    /// joint REML optimizes `[rho, vec(t)]` through latent design hyper-dirs.
    pub latents: Option<JsonValue>,
    /// Python `gamfit.fit(..., penalties=[...])` analytic-penalty descriptors,
    /// validated against the declared latent-coordinate blocks before a
    /// standard latent fit starts.
    pub analytic_penalties: Option<JsonValue>,
    /// Formula-path latent topology selector descriptor. The selector itself
    /// fits candidates through the ordinary workflow; this slot lets callers
    /// request and validate that path from the same config registry.
    pub topology_auto_selector: Option<crate::solver::topology_selector::TopologyAutoSelector>,
    /// `gamfit.fit(..., smooths={...})` Python kwarg routed through the FFI
    /// bridge. JSON object keyed by formula symbol (single column name or
    /// comma-joined tuple) → smooth descriptor (`{"kind": "duchon",
    /// "centers": [[...], ...], ...}`). Applied as a post-processing step on
    /// the [`TermCollectionSpec`] produced by the formula DSL: each smooth
    /// term whose `feature_cols` match a registry key has its kind-specific
    /// tunables (centers, knots, kernel hyperparameters) overridden with the
    /// user-supplied values. The single canonical lowering path guarantees
    /// `smooths={"x": Duchon(centers=K)}` (integer) produces a bit-identical
    /// block spec to writing `duchon(x, centers=K)` in the formula; only
    /// explicit array-valued `centers=` differs, routing through
    /// `CenterStrategy::UserProvided` instead of `FarthestPoint`/`EqualMass`.
    pub smooth_overrides: Option<JsonValue>,
    /// Engage the cross-process ON-DISK persistent warm-start layer (#1082).
    ///
    /// Default `false`: only the always-on in-memory warm start runs, so a
    /// single fit and throwaway/replicate/CI-coverage loops pay zero disk I/O
    /// (no `WarmStartStore` dir/eviction scan, no record load/store). Set
    /// `true` to engage cross-process / repeat-fit resume: the flag threads
    /// `FitConfig → FitOptions → ExternalOptimOptions` down to the standard
    /// `RemlState`, which then calls `enable_persistent_warm_start_disk()`.
    pub persist_warm_start_disk: bool,
}


impl Default for FitConfig {
    fn default() -> Self {
        Self {
            family: None,
            negative_binomial_theta: None,
            link: None,
            flexible_link: false,
            offset_column: None,
            noise_offset_column: None,
            frailty: None,
            baseline_target: "linear".into(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".into(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            survival_likelihood: "location-scale".into(),
            survival_distribution: "gaussian".into(),
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            noise_formula: None,
            logslope_formula: None,
            z_column: None,
            weight_column: None,
            expectile_tau: None,
            ctn_stage1: None,
            scale_dimensions: false,
            adaptive_regularization: None,
            ridge_lambda: 1e-6,
            transformation_normal: false,
            firth: false,
            outer_max_iter: None,
            gpu_policy: crate::gpu::GpuPolicy::Auto,
            resource_policy: None,
            group_metadata: None,
            coefficient_groups: Vec::new(),
            penalty_block_gamma_priors: Vec::new(),
            latents: None,
            analytic_penalties: None,
            topology_auto_selector: None,
            smooth_overrides: None,
            persist_warm_start_disk: false,
        }
    }
}
/// The result of materializing a formula + config against a dataset.
pub struct MaterializedModel<'a> {
    pub request: FitRequest<'a>,
    pub inference_notes: Vec<String>,
}
pub struct SplineScanInputs {
    /// Abscissae of the single 1-D smooth (training rows of its feature column).
    pub x: Vec<f64>,
    /// Gaussian response.
    pub y: Vec<f64>,
    /// Observation weights (variance is `σ²/w`).
    pub w: Vec<f64>,
    /// Smoothing-spline order `m = penalty_order ∈ {1, 2, 3}`: `m = 1` the
    /// random-walk/linear smoother (penalty `λ∫f′²`), `m = 2` the cubic
    /// smoother (penalty `λ∫f″²`), `m = 3` the quintic smoother (penalty
    /// `λ∫(f‴)²`).
    pub order: usize,
}
pub struct ResidualCascadeInputs {
    /// One slice per coordinate axis (2 or 3) of the single scattered smooth.
    pub coords: Vec<Vec<f64>>,
    /// Gaussian response.
    pub y: Vec<f64>,
    /// Observation weights (variance is `σ²/w`).
    pub w: Vec<f64>,
    /// Per-axis positive metric scaling `diag(metric)` of `z = diag(metric)·x`.
    pub metric: Vec<f64>,
    /// Sobolev smoothness order `s` of the multilevel Wendland-(3,1) prior,
    /// clamped into the native-space window `(d/2, (d+3)/2]` (issue caveat 1).
    pub sobolev_s: f64,
}
