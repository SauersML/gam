// #1521: relocated DOWN from gam-models
// `fit_orchestration/drivers/spatial_optimization.rs`. `freeze_term_collection_from_design`
// (+ its private helper `freeze_smooth_basis_from_metadata`) is a pure spec→spec
// freezer over `TermCollectionSpec`/`TermCollectionDesign` — every type it touches
// lives in `gam-terms`/`gam-problem`, so it is a legal gam-terms resident and a
// shared home the future family sub-crates can call without depending on gam-models.
// Byte-identical to the original except `gam_terms::basis::` paths → `crate::basis::`.
use super::*;

// Basis identifiability / boundary / quadrature carriers the freeze logic matches
// on. In the pre-relocation home these arrived via drivers/mod.rs's
// `use gam_terms::basis::{…}`; here they resolve through `crate::basis` (the
// remaining basis specs — BSpline/Spatial/Duchon etc. — arrive via `super::*`).
use crate::basis::{
    ConstantCurvatureIdentifiability, MaternIdentifiability, MeasureJetFrozenQuadrature,
    MeasureJetIdentifiability, OneDimensionalBoundary, SphericalSplineIdentifiability,
};

/// Freeze a `TermCollectionSpec` by baking in the concrete knots, centers,
/// identifiability transforms, and random-effect levels that were resolved
/// during design-matrix construction.  The result passes `validate_frozen`
/// and is safe to serialize for prediction.
///
/// This is the single canonical freezer — every model-save path should call
/// this rather than rolling ad-hoc freezing logic.
/// Freeze a smooth basis spec from its fit-time metadata so that predict-time
/// rebuilds reproduce the exact fitted geometry instead of recomputing any
/// data-dependent construction (knot selection, radial reparameterization,
/// eigen-truncation, identifiability constraint) on the prediction rows.
///
/// This is the SINGLE source of truth for freezing, shared by stand-alone
/// terms and by `by=`-wrapped / factor-sum-to-zero inner smooths. The wrapper
/// arms recurse into this same function, so every inner basis kind is frozen
/// with identical logic. A previous split implementation froze only B-spline
/// inners, leaving spatial inner bases (`bs='tp'`/`matern`/`duchon`/`sos`)
/// unfrozen and recomputed on the prediction grid (#704).
fn freeze_smooth_basis_from_metadata(
    basis: &mut SmoothBasisSpec,
    metadata: &BasisMetadata,
    term_name: &str,
) -> Result<(), EstimationError> {
    match (&mut *basis, metadata) {
        (SmoothBasisSpec::ByVariable { inner, .. }, meta)
        | (SmoothBasisSpec::FactorSumToZero { inner, .. }, meta) => {
            freeze_smooth_basis_from_metadata(inner, meta, term_name)?;
        }
        (
            SmoothBasisSpec::BSpline1D { spec: s, .. },
            BasisMetadata::BSpline1D {
                knots,
                identifiability_transform,
                periodic,
                degree: meta_degree,
                ..
            },
        ) => {
            // Issue #340: bake the fit-time effective degree into the
            // frozen spec so reload sees a self-consistent
            // (degree, knots) pair.
            if let Some(d) = meta_degree {
                s.degree = *d;
            }
            s.knotspec = periodic
                .map(
                    |(domain_start, period, num_basis)| BSplineKnotSpec::PeriodicUniform {
                        data_range: (domain_start, domain_start + period),
                        num_basis,
                    },
                )
                .unwrap_or_else(|| BSplineKnotSpec::Provided(knots.clone()));
            s.identifiability = match identifiability_transform {
                Some(z) => BSplineIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => BSplineIdentifiability::None,
            };
            // Boundary projections are folded into `identifiability_transform`
            // by `build_bspline_basis_1d`. A frozen prediction spec rebuilds the
            // same raw knot basis and replays the captured `FrozenTransform`
            // exactly once; the builder now SKIPS re-deriving the boundary
            // nullspace transform whenever identifiability is `FrozenTransform`
            // (it is already baked into that transform), so re-projection can no
            // longer happen. We therefore KEEP the original
            // `boundary_conditions`: they are the single source of truth the
            // intercept-suppression decision reads
            // (`term_collection_has_one_sided_anchored_bspline`), and a
            // one-sided anchored smooth suppresses the global intercept at fit
            // time (#1238). Clearing them here flipped that decision at predict
            // and re-added a spurious intercept column → save→load→predict
            // 21-vs-22 design mismatch (#1265). Boundary conditions are left
            // exactly as fit.
        }
        (
            SmoothBasisSpec::BSpline1D { spec: s, .. },
            BasisMetadata::CubicRegression1D {
                knots,
                identifiability_transform,
            },
        ) => {
            // #1074: a natural cubic regression spline freezes to its
            // value-at-knot knot set plus the captured raw→constrained transform,
            // mirroring the `BSpline1D` arm above. The predict-time builder
            // reconstructs the cr geometry from `knots` and replays the
            // `FrozenTransform` exactly, so the design matches fit time.
            s.knotspec = BSplineKnotSpec::NaturalCubicRegression {
                knots: knots.clone(),
            };
            s.identifiability = match identifiability_transform {
                Some(z) => BSplineIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => BSplineIdentifiability::None,
            };
        }
        (
            SmoothBasisSpec::ThinPlate {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::ThinPlate {
                centers,
                length_scale,
                periodic: meta_periodic,
                identifiability_transform,
                input_scales: meta_scales,
                radial_reparam,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.length_scale = *length_scale;
            s.identifiability = match identifiability_transform {
                Some(z) => SpatialIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => match &s.identifiability {
                    SpatialIdentifiability::FrozenTransform { .. } => s.identifiability.clone(),
                    _ => SpatialIdentifiability::None,
                },
            };
            s.radial_reparam = radial_reparam.clone();
            s.periodic = meta_periodic.clone();
            *input_scales = meta_scales.clone();
        }
        (
            SmoothBasisSpec::ThinPlate { feature_cols, .. },
            BasisMetadata::Duchon {
                centers,
                length_scale,
                periodic: meta_periodic,
                power,
                nullspace_order,
                identifiability_transform,
                input_scales: meta_scales,
                aniso_log_scales: meta_aniso,
                radial_reparam: meta_radial_reparam,
                ..
            },
        ) => {
            // Auto-promotion path: the basis builder rewrote a canonical-TPS
            // request to a pure Duchon spline because k < polynomial-nullspace
            // size at this dimension. Bake the resolved Duchon parameters into
            // the spec so predict-time goes through the same Duchon code path.
            let identifiability = match identifiability_transform {
                Some(z) => SpatialIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => SpatialIdentifiability::None,
            };
            *basis = SmoothBasisSpec::Duchon {
                feature_cols: feature_cols.clone(),
                spec: DuchonBasisSpec {
                    periodic: meta_periodic.clone(),
                    center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                    length_scale: *length_scale,
                    power: *power,
                    nullspace_order: *nullspace_order,
                    identifiability,
                    aniso_log_scales: meta_aniso.clone(),
                    operator_penalties: Default::default(),
                    boundary: OneDimensionalBoundary::Open,
                    radial_reparam: meta_radial_reparam.clone(),
                },
                input_scales: meta_scales.clone(),
            };
        }
        (
            SmoothBasisSpec::Sphere { spec: s, .. },
            BasisMetadata::Sphere {
                centers,
                penalty_order,
                method,
                max_degree,
                wahba_kernel,
                constraint_transform,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.penalty_order = *penalty_order;
            s.method = *method;
            s.max_degree = *max_degree;
            s.wahba_kernel = *wahba_kernel;
            // #532: freeze the realized-design transform captured at fit time
            // so the predict-time rebuild reuses it verbatim instead of
            // dropping the parametric orthogonalization and resurrecting the
            // intercept collision. The Harmonic method never carries a
            // constraint transform.
            s.identifiability = match constraint_transform {
                Some(z) => SphericalSplineIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => SphericalSplineIdentifiability::CenterSumToZero,
            };
        }
        (
            SmoothBasisSpec::ConstantCurvature { spec: s, .. },
            BasisMetadata::ConstantCurvature {
                centers,
                kappa,
                length_scale,
                constraint_transform,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.kappa = *kappa;
            // Pin the REALIZED kernel range so a `0.0` auto-sentinel spec
            // replays the exact fit-time geometry at predict time (and at the
            // future ψ-channel per-trial rebuilds) instead of re-deriving ℓ
            // from whatever rows the rebuild sees.
            s.length_scale = *length_scale;
            // #532 pattern: freeze the composed `z · z_parametric` so the
            // rebuild replays the fit-time realized transform verbatim.
            s.identifiability = match constraint_transform {
                Some(z) => ConstantCurvatureIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => ConstantCurvatureIdentifiability::CenterSumToZero,
            };
        }
        (
            SmoothBasisSpec::MeasureJet {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::MeasureJet {
                centers,
                input_scales: meta_scales,
                length_scale,
                eps_band,
                order_s,
                alpha,
                tau0,
                masses,
                support_means,
                penalty_normalization_scales,
                raw_penalty_normalization_scales,
                fused_penalty_normalization_scale,
                constraint_transform,
                sigma_coord,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            // Pin the realized geometry so auto sentinels cannot re-derive it
            // from predict rows. Field semantics are owned elsewhere:
            // length_scale replays VERBATIM (build dispatch round-trip
            // contract); order_s is the sentinel-preserving mode marker and
            // masses + eps_band are the fit-data penalty quadrature (both
            // `BasisMetadata::MeasureJet`).
            s.length_scale = *length_scale;
            s.order_s = *order_s;
            s.alpha = *alpha;
            s.tau0 = *tau0;
            s.num_scales = eps_band.len();
            s.frozen_quadrature = Some(MeasureJetFrozenQuadrature {
                masses: masses.clone(),
                eps_band: eps_band.clone(),
                support_means: support_means.clone(),
                penalty_normalization_scales: penalty_normalization_scales.clone(),
                raw_penalty_normalization_scales: raw_penalty_normalization_scales.clone(),
                fused_penalty_normalization_scale: *fused_penalty_normalization_scale,
                sigma_coord: *sigma_coord,
            });
            // #532 pattern: freeze the composed `z · z_parametric` so the
            // rebuild replays the fit-time realized transform verbatim.
            s.identifiability = match constraint_transform {
                Some(z) => MeasureJetIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => MeasureJetIdentifiability::CenterSumToZero,
            };
            *input_scales = meta_scales.clone();
        }
        (
            SmoothBasisSpec::Matern {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::Matern {
                centers,
                length_scale,
                periodic: meta_periodic,
                nu,
                include_intercept,
                identifiability_transform,
                input_scales: meta_scales,
                aniso_log_scales: meta_aniso,
                nullspace_shrinkage_survived: meta_nullspace_survived,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.length_scale = *length_scale;
            s.nu = *nu;
            s.include_intercept = *include_intercept;
            // Pin the bootstrap-κ double-penalty nullspace-shrinkage decision into
            // the frozen transform so the κ-optimizer's per-trial design rebuilds
            // reproduce the SAME learned-penalty count (gam#787/#860); without
            // this the κ-dependent spectral test in `build_nullspace_shrinkage_penalty`
            // flips the count 6↔7 and the rebuilt ρ dimension disagrees with the
            // frozen joint setup ("joint hyper rho dimension mismatch"). When there
            // is no transform to freeze we keep `None` (unconstrained kernel needs
            // no replayed survival decision).
            s.identifiability = match identifiability_transform {
                Some(z) => MaternIdentifiability::FrozenTransform {
                    transform: z.clone(),
                    nullspace_shrinkage_survived: Some(*meta_nullspace_survived),
                },
                None => MaternIdentifiability::None,
            };
            s.aniso_log_scales = meta_aniso.clone();
            s.periodic = meta_periodic.clone();
            *input_scales = meta_scales.clone();
        }
        (
            SmoothBasisSpec::Duchon {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::Duchon {
                centers,
                length_scale,
                periodic: meta_periodic,
                power,
                nullspace_order,
                identifiability_transform,
                input_scales: meta_scales,
                aniso_log_scales: meta_aniso,
                radial_reparam,
                ..
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.length_scale = *length_scale;
            s.power = *power;
            s.nullspace_order = *nullspace_order;
            s.identifiability = match identifiability_transform {
                Some(z) => SpatialIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => match &s.identifiability {
                    // If the spec already carries a frozen transform but the
                    // metadata lost it (e.g. raw rebuild stripped it), keep
                    // the existing frozen transform rather than downgrading.
                    SpatialIdentifiability::FrozenTransform { .. } => s.identifiability.clone(),
                    _ => SpatialIdentifiability::None,
                },
            };
            s.aniso_log_scales = meta_aniso.clone();
            s.periodic = meta_periodic.clone();
            *input_scales = meta_scales.clone();
            // #1355: persist the frozen data-metric radial reparam so the
            // predict-time / κ-trial rebuild replays the EXACT fit-time rotated
            // radial basis (a fresh `V` from predict rows would differ).
            s.radial_reparam = radial_reparam.clone();
        }
        (
            SmoothBasisSpec::Sphere { spec: s, .. },
            BasisMetadata::SphereHarmonics {
                max_degree,
                radians,
            },
        ) => {
            s.max_degree = Some(*max_degree);
            s.radians = *radians;
        }
        (
            SmoothBasisSpec::TensorBSpline {
                feature_cols,
                spec: s,
            },
            BasisMetadata::TensorBSpline {
                feature_cols: fitted_cols,
                knots,
                degrees,
                periods,
                is_cr,
                identifiability_transform,
            },
        ) => {
            if s.marginalspecs.len() != knots.len() || s.marginalspecs.len() != degrees.len() {
                crate::bail_invalid_estim!(
                    "tensor freeze mismatch for '{}': marginalspecs={}, knots={}, degrees={}",
                    term_name,
                    s.marginalspecs.len(),
                    knots.len(),
                    degrees.len()
                );
            }
            *feature_cols = fitted_cols.clone();
            for i in 0..s.marginalspecs.len() {
                s.marginalspecs[i].degree = degrees[i];
                // A cr margin (#1074) freezes back to a `NaturalCubicRegression`
                // knotspec carrying the same k value-knots, so the predict-time
                // marginal cr design is rebuilt identically (not downgraded to
                // an open `Provided(knots)` B-spline). `is_cr` may be empty when
                // an older model is deserialized without the field — treat a
                // missing/short entry as `false` (legacy B-spline tensor).
                let margin_is_cr = is_cr.get(i).copied().unwrap_or(false);
                s.marginalspecs[i].knotspec = match (periods[i], knots[i].len()) {
                    (Some(period), num_basis) if num_basis >= 1 => {
                        // Periodic uniform reconstructs the open
                        // `[start, start + period)` data range from the
                        // first knot and the saved period. `knots[i].len()`
                        // is the periodic control-site count.
                        let domain_start = knots[i][0];
                        BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_start + period),
                            num_basis,
                        }
                    }
                    _ if margin_is_cr => BSplineKnotSpec::NaturalCubicRegression {
                        knots: knots[i].clone(),
                    },
                    _ => BSplineKnotSpec::Provided(knots[i].clone()),
                };
            }
            // Do NOT overwrite `s.periods` from `periods`: `frozen = spec.clone()`
            // already preserves the user's original `spec.periods`. The metadata
            // `periods` slot captures the effective per-margin period for
            // restoring `PeriodicUniform` knotspecs, but it may also reflect
            // periodicity implied by a `PeriodicUniform` knotspec on a margin
            // for which the user left `spec.periods` empty (intending the
            // periodic B-spline path, not the Fourier path). Overwriting
            // `s.periods` here would silently flip such a margin onto the
            // Fourier path at predict time and produce a basis distinct from
            // the one used at fit time.
            s.identifiability = match identifiability_transform {
                Some(z) => TensorBSplineIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => TensorBSplineIdentifiability::None,
            };
        }
        (
            SmoothBasisSpec::FactorSmooth { spec: s },
            BasisMetadata::FactorSmooth {
                knots,
                degree,
                periodic,
                group_levels,
                marginal_is_cr,
                ..
            },
        ) => {
            s.marginal.knotspec = if *marginal_is_cr {
                // A cubic regression spline marginal (mgcv's `bs="sz"` default,
                // #1074) stores its `k` value-knots, not a B-spline knot vector.
                // Restore the cr knotspec so the predict-time rebuild replays the
                // SAME marginal instead of misreading the value-knots as a
                // B-spline knot vector.
                BSplineKnotSpec::NaturalCubicRegression {
                    knots: knots.clone(),
                }
            } else {
                periodic
                    .map(
                        |(domain_start, period, num_basis)| BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_start + period),
                            num_basis,
                        },
                    )
                    .unwrap_or_else(|| BSplineKnotSpec::Provided(knots.clone()))
            };
            // Restore the FROZEN marginal degree (#555 predict-replay). With
            // a `Provided(knots)` knotspec the per-margin basis count is
            // `knots.len() - (degree + 1)`, so if fit-time auto-shrink
            // lowered the marginal degree (small per-group n: cubic →
            // quadratic/linear), rebuilding with the original spec degree
            // would yield a different per-level `p` and corrupt the
            // block-diagonal replay. Mirror the sibling BySmooth arm, which
            // restores `inner.degree = *degree` for exactly this reason.
            s.marginal.degree = *degree;
            s.group_frozen_levels = Some(group_levels.clone());
        }
        (
            SmoothBasisSpec::BySmooth { smooth, by_kind },
            BasisMetadata::FactorSmooth {
                knots,
                degree,
                periodic,
                group_levels,
                ..
            },
        ) => {
            if let ByVarKind::Factor { frozen_levels, .. } = by_kind {
                *frozen_levels = Some(group_levels.clone());
            }
            if let SmoothBasisSpec::BSpline1D { spec: inner, .. } = smooth.as_mut() {
                // Issue #340: FactorSmooth metadata records the per-axis
                // spline degree directly (not optional); reflect it on
                // the inner spec so reload sees a self-consistent
                // (degree, knots) pair after fit-time auto-shrink.
                inner.degree = *degree;
                inner.knotspec = periodic
                    .map(
                        |(domain_start, period, num_basis)| BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_start + period),
                            num_basis,
                        },
                    )
                    .unwrap_or_else(|| BSplineKnotSpec::Provided(knots.clone()));
                inner.identifiability = BSplineIdentifiability::None;
            }
        }
        (
            SmoothBasisSpec::BySmooth { smooth, by_kind },
            BasisMetadata::BySmooth { inner, levels, .. },
        ) => {
            // A `by=` smooth whose metadata is wrapped in `BySmooth`:
            // restore the frozen grouping levels, then recurse so the inner
            // basis is frozen with EXACTLY the same logic used for a
            // stand-alone term.
            if let ByVarKind::Factor { frozen_levels, .. } = by_kind
                && let Some(levels) = levels
            {
                *frozen_levels = Some(levels.clone());
            }
            freeze_smooth_basis_from_metadata(smooth, inner, term_name)?;
        }
        (SmoothBasisSpec::BySmooth { smooth, .. }, metadata) => {
            // `by=` wrapper carrying the inner basis metadata directly
            // (numeric `by`, or a factor `by` lowered to one gated block).
            // Recurse so a spatial inner basis (thin-plate, Matern, Duchon,
            // sphere, tensor, …) is frozen identically to a stand-alone
            // term. Previously only a B-spline inner was frozen here, so a
            // `s(x, by=g, bs='tp')` smooth left its data-dependent kernel
            // and eigen-truncation to be recomputed on the prediction grid,
            // crashing the predict-time design rebuild (#704).
            freeze_smooth_basis_from_metadata(smooth, metadata, term_name)?;
        }
        _ => {
            crate::bail_invalid_estim!(
                "smooth metadata/spec type mismatch while freezing term '{}'",
                term_name
            );
        }
    }
    Ok(())
}

pub fn freeze_term_collection_from_design(
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<TermCollectionSpec, EstimationError> {
    if spec.smooth_terms.len() != design.smooth.terms.len() {
        crate::bail_invalid_estim!(
            "freeze mismatch: smooth spec count {} != design smooth term count {}",
            spec.smooth_terms.len(),
            design.smooth.terms.len()
        );
    }
    if spec.random_effect_terms.len() != design.random_effect_levels.len() {
        crate::bail_invalid_estim!(
            "freeze mismatch: random-effect spec count {} != design random-effect term count {}",
            spec.random_effect_terms.len(),
            design.random_effect_levels.len()
        );
    }

    let mut frozen = spec.clone();

    // ── smooth terms ────────────────────────────────────────────────────
    for (term, fitted) in frozen
        .smooth_terms
        .iter_mut()
        .zip(design.smooth.terms.iter())
    {
        // Persist joint-null absorption rotation captured at fit time so
        // save → load → predict re-applies `X_new_raw · Q` identically to
        // in-memory prediction. `None` when the smooth had no joint null
        // space, or when rotation was suppressed (shape-constrained smooths
        // whose cone geometry would not survive arbitrary orthogonal
        // rotation). Without this propagation, models reloaded from disk
        // produce wrong η at predict-time for any smooth with `Some(Q)`.
        term.joint_null_rotation = fitted.joint_null_rotation.clone();
        freeze_smooth_basis_from_metadata(&mut term.basis, &fitted.metadata, &term.name)?;
        // Persist the global-orthogonality chart the metadata could not absorb
        // (factor-smooth kinds residualized against an overlapping owner
        // smooth, #978). Without this, save → load → predict rebuilds the
        // unresidualized full-width design and the fitted coefficients no
        // longer match it.
        if let Some(z) = fitted.unabsorbed_global_orthogonality.as_ref() {
            match &mut term.basis {
                SmoothBasisSpec::FactorSumToZero {
                    frozen_global_orthogonality,
                    ..
                } => *frozen_global_orthogonality = Some(z.clone()),
                SmoothBasisSpec::FactorSmooth { spec } => {
                    spec.frozen_global_orthogonality = Some(z.clone());
                }
                _ => {
                    crate::bail_invalid_estim!(
                        "freeze: term '{}' carries an unabsorbed global-orthogonality transform but its basis kind has no frozen carrier for it",
                        term.name
                    );
                }
            }
        }
    }

    // ── random-effect terms ─────────────────────────────────────────────
    for (idx, rt) in frozen.random_effect_terms.iter_mut().enumerate() {
        let (_, kept_levels) = &design.random_effect_levels[idx];
        rt.frozen_levels = Some(kept_levels.clone());
    }

    Ok(frozen)
}
