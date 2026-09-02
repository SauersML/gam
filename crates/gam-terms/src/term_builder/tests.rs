#![cfg(test)]
//! Unit tests for [`super`], the formula-term builder.
//!
//! Split out of `term_builder.rs` under #2791: the parent crossed the 10,000
//! line ceiling that `build.rs`'s `scan_for_oversized_tracked_files` enforces,
//! and production code versus its unit tests is the one seam in this file that
//! is genuinely cohesive on both sides. `use super::*` keeps every private item
//! of the parent in scope, so the tests are unchanged apart from indentation.
//!
//! The `#![cfg(test)]` above is redundant with the sole declaration of this
//! module (`#[cfg(test)] mod tests;` in the parent) and is there on purpose: it
//! states the test-only scope to the COMPILER rather than to the file name,
//! which is what `build.rs`'s scope tracker reads.

use super::*;
use crate::basis::{OperatorPenaltySpec, PenaltySource};
use crate::inference::formula_dsl::parse_formula;
use gam_data::{DataSchema, SchemaColumn};
use ndarray::{Array1, Array2};
use std::collections::BTreeMap;

/// #2293 regression: distinct-value counting for factor levels must route
/// through `gam_data::canonical_level_bits`, so `+0.0` / `-0.0` collapse to
/// one level and every NaN payload collapses to one level. The previous
/// ad-hoc `if x == 0.0 { 0.0 } else { x }.to_bits()` idiom collapsed signed
/// zero but left distinct NaN bit patterns as separate levels, over-counting
/// the cardinality that caps a factor/cr marginal's basis.
#[test]
fn unique_count_column_uses_canonical_level_bits() {
    // +0.0 and -0.0 are one level; two NaN payloads are one level.
    let signed_zero = Array1::from(vec![0.0, -0.0, 0.0]);
    assert_eq!(
        unique_count_column(signed_zero.view()),
        1,
        "+0.0 and -0.0 must collapse to a single level"
    );

    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0xfff8_0000_0000_dead);
    assert!(nan_a.is_nan() && nan_b.is_nan() && nan_a.to_bits() != nan_b.to_bits());
    let nans = Array1::from(vec![nan_a, nan_b]);
    assert_eq!(
        unique_count_column(nans.view()),
        1,
        "distinct NaN payloads must collapse to a single level"
    );

    // Ordinary finite values stay distinct.
    let finite = Array1::from(vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(unique_count_column(finite.view()), 3);
}

/// #1867 regression: on sparse 1-D data the generic conditioning cap in
/// [`default_num_centers`] (`n / COND_N_DIVISOR`) starves a radial
/// (matérn/duchon) basis BELOW the resolution the univariate B-spline
/// `s(x)` is handed on the SAME data — 7 vs 11 basis functions at n=30 —
/// so `matern(x)`/`duchon(x)` over-smooth oscillations that `s(x)`
/// recovers. The spline-equivalent floor threaded into the radial default
/// count must restore that resolution. Without the floor (the `0` argument,
/// i.e. the pre-fix behaviour) the radial default stays starved.
#[test]
fn radial_1d_default_not_starved_below_univariate_spline_resolution_1867() {
    let n = 30usize;
    let d = 1usize;
    // Raw radial default, starved by the n/COND_N_DIVISOR conditioning cap.
    let planned = default_num_centers(n, d);
    assert!(
        planned < 11,
        "precondition: conditioning cap starves the raw radial default (got {planned})"
    );
    // A well-resolved 1-D column of `n` distinct values asks for the
    // univariate spline basis dimension the competing `s(x)` gets.
    let col: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
    let univariate_floor =
        heuristic_knots_for_column(col.view()).saturating_add(DEFAULT_BSPLINE_DEGREE + 1);
    assert_eq!(univariate_floor, 11, "univariate spline resolution at n=30");

    // BEFORE (no floor): radial defaults inherit the starved count.
    assert_eq!(default_matern_center_count(n, d, planned, 0), planned);
    assert!(default_duchon_center_count(n, d, planned, 2, 0) <= planned);

    // AFTER (spline-equivalent floor): radial defaults are lifted to at
    // least the univariate spline resolution, so they are not dimensioned
    // coarser than `s(x)` on identical data.
    assert!(
        default_matern_center_count(n, d, planned, univariate_floor) >= univariate_floor,
        "matern 1-D default must not be starved below the spline resolution"
    );
    assert!(
        default_duchon_center_count(n, d, planned, 2, univariate_floor) >= univariate_floor,
        "duchon 1-D default must not be starved below the spline resolution"
    );

    // The floor is scoped to 1-D: a multivariate smooth passes 0 and keeps
    // the generic n-scaling plan unchanged.
    assert_eq!(default_matern_center_count(200, 2, 40, 0), 40);
}

/// #1757 regression: an omitted `k=`/`centers=` on a 2-D Duchon smooth must
/// remain a low-rank representer basis. The generic spatial planner grows
/// with `n` (125 centers at n=500), which makes the Duchon center-Gram
/// rotation and REML linear algebra scale as dense `O(k^3)` setup work
/// before the data-fit iterations even start. The Duchon-specific default
/// caps the implicit basis at the thin-plate/Duchon spline rank
/// `10 * 3^(d - 1)` (30 in 2-D) while explicit `k=`/`centers=` still bypass
/// this helper upstream.
#[test]
fn duchon_2d_default_is_low_rank_not_generic_spatial_width_1757() {
    let n = 500usize;
    let d = 2usize;
    let polynomial_cols = d + 1;
    let generic_plan = default_num_centers(n, d);
    let duchon_default = default_duchon_center_count(n, d, generic_plan, polynomial_cols, 0);
    let spline_rank = 10usize.saturating_mul(3usize.saturating_pow((d - 1) as u32));

    assert!(
        generic_plan > spline_rank,
        "precondition: generic spatial plan should be wider than the Duchon low-rank spline rank"
    );
    assert_eq!(
        duchon_default, spline_rank,
        "2-D Duchon default must use the low-rank spline representer size, not the generic spatial width"
    );
    assert!(
        duchon_default > polynomial_cols,
        "the capped default must still contain the affine polynomial null space"
    );
}

/// #2761 gate on the DEFAULT itself, not on a fixture.
///
/// The measure-jet representer range ℓ has now been default-on (`299c83ffc`,
/// which introduced it to remove a 13x deficit), default-off (`b1d94d1a5`,
/// one line, no measurement), and default-on again (#2761, after measuring
/// that the design's own span floor at a frozen ℓ *is* the 13.4x). Each flip
/// was invisible to the test suite until an accuracy fixture noticed months
/// later, because nothing asserted the default. This does.
///
/// It also pins the two overrides that make the default safe to hold:
/// a typed `length_scale=` is a request and pins ℓ, and an explicit
/// `learn_length_scale=` beats both.
#[test]
fn measure_jet_reml_selects_the_representer_range_by_default_2761() {
    let ds = continuous_dataset(
        &["y", "x1", "x2"],
        (0..40)
            .map(|i| {
                let t = i as f64 / 39.0;
                vec![(6.0 * t).sin(), t, 0.5 + 0.5 * (6.0 * t).cos()]
            })
            .collect(),
    );
    let col_map = ds.column_map();
    let learns = |body: &str| -> bool {
        let parsed = parse_formula(&format!("y ~ {body}")).expect("parse mjs formula");
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut Vec::new(),
            &gam_runtime::resource::ResourcePolicy::default_library(),
        )
        .expect("build mjs term");
        let SmoothBasisSpec::MeasureJet { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected a measure-jet smooth for '{body}'");
        };
        // Read through the SAME accessors the outer engine's θ-layout uses,
        // so a default that stops reaching ψ enrollment fails here too.
        let learns = crate::smooth::measure_jet_learns_length_scale(spec);
        assert_eq!(
            spec.learn_length_scale, learns,
            "'{body}': the ψ accessor and the spec field must not disagree"
        );
        assert_eq!(
            crate::smooth::measure_jet_psi_dim(spec),
            usize::from(learns),
            "'{body}': single-scale ψ dimension is exactly the ℓ coordinate"
        );
        assert_eq!(
            crate::smooth::measure_jet_enrolls_psi(spec),
            learns,
            "'{body}': single-scale enrollment is exactly the ℓ coordinate"
        );
        learns
    };

    assert!(
        learns("mjs(x1, x2, centers=8)"),
        "a plain measure-jet smooth must REML-select its representer range: λ shrinks \
             inside a span and cannot move one, so a frozen ℓ is an error no smoothing \
             parameter can repair (#2761 measured 13.4x held-out RMSE, with the design's \
             own least-squares span floor sitting AT the fitted value)"
    );
    assert!(
        !learns("mjs(x1, x2, centers=8, length_scale=0.3)"),
        "a typed length_scale= is a request, not a seed, and must pin ℓ — the same \
             short-circuit an explicitly-scaled Matérn gets"
    );
    assert!(
        !learns("mjs(x1, x2, centers=8, learn_length_scale=false)"),
        "an explicit opt-out must be honored"
    );
    assert!(
        learns("mjs(x1, x2, centers=8, length_scale=0.3, learn_length_scale=true)"),
        "an explicit opt-in must beat the length_scale= pin, so a caller can seed the \
             search at a range of their choosing"
    );
}

fn continuous_dataset(headers: &[&str], rows: Vec<Vec<f64>>) -> Dataset {
    let nrows = rows.len();
    let ncols = headers.len();
    let values = Array2::from_shape_vec(
        (nrows, ncols),
        rows.into_iter().flat_map(|row| row.into_iter()).collect(),
    )
    .expect("rectangular test data");
    Dataset {
        headers: headers.iter().map(|name| name.to_string()).collect(),
        values,
        schema: DataSchema {
            columns: headers
                .iter()
                .map(|name| SchemaColumn {
                    name: name.to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; ncols],
    }
}

#[test]
fn term_completeness_is_scoped_to_formula_columns() {
    let mut data = continuous_dataset(
        &["y", "x", "unused"],
        (0..12)
            .map(|row| vec![row as f64, row as f64 / 11.0, 1.0])
            .collect(),
    );
    data.values[[4, 2]] = f64::NAN;
    let parsed = parse_formula("y ~ x").expect("parse linear formula");
    build_termspec(
        &parsed.terms,
        &data,
        &data.column_map(),
        &mut Vec::new(),
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("a missing cell in an unreferenced column must be irrelevant");

    data.values[[4, 1]] = f64::NAN;
    let error = build_termspec(
        &parsed.terms,
        &data,
        &data.column_map(),
        &mut Vec::new(),
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect_err("a missing cell in a model term must fail before design construction");
    assert!(
        error
            .to_string()
            .contains("model term column 'x' contains a non-finite value at row 5")
    );
}

fn factor_dataset() -> Dataset {
    let rows = (0..24)
        .map(|i| {
            let x = i as f64 / 23.0;
            let g = (i % 2) as f64;
            vec![x + g, x, g]
        })
        .collect::<Vec<_>>();
    Dataset {
        headers: vec!["y".into(), "x".into(), "g".into()],
        values: Array2::from_shape_vec(
            (rows.len(), 3),
            rows.into_iter().flat_map(|row| row.into_iter()).collect(),
        )
        .expect("rectangular factor test data"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "g".into(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".into(), "b".into()],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
        ],
    }
}

fn build_two_dimensional_spatial_basis(
    ds: &Dataset,
    selector: &str,
    count_option: Option<&str>,
) -> SmoothBasisSpec {
    let mut options = BTreeMap::new();
    options.insert("bs".to_string(), selector.to_string());
    if let Some(option) = count_option {
        options.insert(option.to_string(), "7".to_string());
    }
    let mut notes = Vec::new();
    build_smooth_basis(
        SmoothKind::S,
        &["x".to_string(), "z".to_string()],
        &[1, 2],
        &options,
        ds,
        &mut notes,
        &ResourcePolicy::default_library(),
        1,
    )
    .unwrap_or_else(|error| {
        panic!("failed to build {selector} with count option {count_option:?}: {error}")
    })
}

fn curvature_or_measurejet_center_strategy(basis: &SmoothBasisSpec) -> &CenterStrategy {
    match basis {
        SmoothBasisSpec::ConstantCurvature { spec, .. } => &spec.center_strategy,
        SmoothBasisSpec::MeasureJet { spec, .. } => &spec.center_strategy,
        other => panic!("expected curvature or measure-jet basis, got {other:?}"),
    }
}

/// Build a `sphere(lat, lon)` term over columns 1 (lat) and 2 (lon) of `ds`.
fn build_sphere_over_lat_lon(ds: &Dataset) -> Result<SmoothBasisSpec, String> {
    let mut options = BTreeMap::new();
    options.insert("bs".to_string(), "sphere".to_string());
    options.insert("k".to_string(), "10".to_string());
    options.insert("kernel".to_string(), "sobolev".to_string());
    let mut notes = Vec::new();
    build_smooth_basis(
        SmoothKind::S,
        &["lat".to_string(), "lon".to_string()],
        &[1, 2],
        &options,
        ds,
        &mut notes,
        &ResourcePolicy::default_library(),
        1,
    )
}

/// A sphere/SOS smooth is intrinsically a function of BOTH angular
/// coordinates: a constant longitude puts every point on one meridian, an
/// unidentifiable 1-D slice of S² that must be rejected at term construction
/// with a coordinate-named error — not fit silently. Varying both angular
/// coordinates is accepted.
#[test]
fn sphere_rejects_constant_longitude_but_accepts_varying() {
    // lat varies across [-70, 70]; lon is pinned at 0 (a single meridian).
    let rows_const_lon: Vec<Vec<f64>> = (0..60)
        .map(|i| {
            let lat = -70.0 + 140.0 * (i as f64) / 59.0;
            vec![0.0, lat, 0.0] // y, lat, lon(const)
        })
        .collect();
    let ds_const = continuous_dataset(&["y", "lat", "lon"], rows_const_lon);
    let err = build_sphere_over_lat_lon(&ds_const)
        .expect_err("a constant-longitude sphere smooth must be rejected as degenerate");
    let lower = err.to_lowercase();
    assert!(
        (lower.contains("constant") || lower.contains("degenerate") || lower.contains("unique"))
            && lower.contains("lon"),
        "rejection must flag degeneracy and name the constant longitude coordinate: {err}"
    );

    // Both angular coordinates vary: a well-posed 2-sphere smooth builds.
    let rows_ok: Vec<Vec<f64>> = (0..60)
        .map(|i| {
            let lat = -70.0 + 140.0 * (i as f64) / 59.0;
            // A well-spread longitude (deterministic, no RNG) so the input
            // genuinely covers both angular axes.
            let lon = -170.0 + 340.0 * ((i * 17 % 60) as f64) / 59.0;
            vec![0.0, lat, lon]
        })
        .collect();
    let ds_ok = continuous_dataset(&["y", "lat", "lon"], rows_ok);
    build_sphere_over_lat_lon(&ds_ok)
        .expect("a sphere smooth over varying latitude and longitude must build");
}

#[test]
fn curvature_and_measurejet_omitted_counts_retain_auto_provenance() {
    let ds = continuous_dataset(
        &["y", "x", "z"],
        (0..64)
            .map(|i| {
                let x = i as f64 / 63.0;
                let z = ((i * 17) % 64) as f64 / 63.0;
                vec![x.sin() + z.cos(), x, z]
            })
            .collect(),
    );
    let expected = default_num_centers(ds.values.nrows(), 2);

    for selector in ["curv", "mjs"] {
        let basis = build_two_dimensional_spatial_basis(&ds, selector, None);
        let strategy = curvature_or_measurejet_center_strategy(&basis);
        assert!(
            matches!(strategy, CenterStrategy::Auto(_)),
            "an omitted count on {selector} must retain Auto provenance, got {strategy:?}",
        );
        assert_eq!(
            strategy.planned_num_centers(2),
            expected,
            "Auto provenance must preserve {selector}'s resolved default count",
        );
    }
}

#[test]
fn curvature_and_measurejet_explicit_count_aliases_remain_pinned() {
    let ds = continuous_dataset(
        &["y", "x", "z"],
        (0..32)
            .map(|i| {
                let x = i as f64 / 31.0;
                let z = ((i * 11) % 32) as f64 / 31.0;
                vec![x - z, x, z]
            })
            .collect(),
    );

    for selector in ["curv", "mjs"] {
        for alias in [
            "centers",
            "k",
            "basis_dim",
            "basis-dim",
            "basisdim",
            "knots",
        ] {
            let basis = build_two_dimensional_spatial_basis(&ds, selector, Some(alias));
            let strategy = curvature_or_measurejet_center_strategy(&basis);
            assert!(
                !matches!(strategy, CenterStrategy::Auto(_)),
                "explicit {alias}= on {selector} must remain pinned, got {strategy:?}",
            );
            assert_eq!(
                strategy.planned_num_centers(2),
                7,
                "explicit {alias}= must remain the exact {selector} center count",
            );
        }
    }
}

/// #1378: the DEFAULT univariate `s(x, bs="tp")` must build a *modest*
/// mgcv-sized basis, not the n-scaled spatial heuristic. The oversized
/// default basis left the two-penalty REML ρ-surface with a flat valley
/// whose optimizer landing point depended on row order, breaking
/// row-permutation invariance. Pin the default 1-D center count so a
/// regression that reinstates the n-scaled default trips here, fast, with
/// no fit/optimizer in the loop.
#[test]
fn default_univariate_thinplate_basis_dim_is_modest() {
    // n = 300 (the #1378 scenario): the n-scaled spatial heuristic would
    // request ~75 centers here. The modest default must stay near k = 10.
    let n = 300usize;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let x = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
            vec![x.sin(), x]
        })
        .collect();
    let ds = continuous_dataset(&["y", "x"], rows);

    let mut options = BTreeMap::new();
    options.insert("bs".to_string(), "tp".to_string());

    let mut notes = Vec::new();
    let basis = build_smooth_basis(
        SmoothKind::S,
        &["x".to_string()],
        &[1],
        &options,
        &ds,
        &mut notes,
        &ResourcePolicy::default_library(),
        1,
    )
    .expect("build default univariate tp smooth");

    let centers = match &basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => match &spec.center_strategy {
            CenterStrategy::Auto(inner) => match inner.as_ref() {
                CenterStrategy::FarthestPoint { num_centers }
                | CenterStrategy::EqualMass { num_centers }
                | CenterStrategy::EqualMassCovarRepresentative { num_centers }
                | CenterStrategy::KMeans { num_centers, .. } => *num_centers,
                other => panic!("unexpected auto inner center strategy: {other:?}"),
            },
            CenterStrategy::FarthestPoint { num_centers }
            | CenterStrategy::EqualMass { num_centers }
            | CenterStrategy::EqualMassCovarRepresentative { num_centers }
            | CenterStrategy::KMeans { num_centers, .. } => *num_centers,
            other => panic!("unexpected center strategy: {other:?}"),
        },
        other => panic!("expected ThinPlate basis, got {other:?}"),
    };

    // #1074: the mgcv-sized basis-dim ceiling assertion was removed with the
    // cap it tested. The default tp basis is now n-scaled; we only assert it
    // still builds a usable basis.
    assert!(
        centers >= 1,
        "default univariate tp must still build a usable basis (centers={centers})",
    );
}

/// gam#1629: a default 2-D `matern(x1, x2)` (no explicit `length_scale`)
/// must retain typed Auto ownership — NOT a baked-in data diameter — so the
/// planner's `auto_init_length_scale_in_place` seeds it on the
/// wiggly/resolving side (`max_range / sqrt(n)`), the same regime thin-plate
/// uses. This pins the corrected seed geometry without a fit/optimizer in
/// the loop.
#[test]
fn default_matern_2d_seeds_resolving_length_scale_not_overscaled_diameter() {
    // A fine multi-frequency 2-D grid (the #1629 reproduction shape): the
    // data diameter is O(1.4) in each axis; the resolving seed must be far
    // smaller than the diameter so high-frequency structure stays reachable.
    let side = 24usize; // n = 576
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(side * side);
    for i in 0..side {
        for j in 0..side {
            let x1 = i as f64 / (side - 1) as f64; // [0, 1]
            let x2 = j as f64 / (side - 1) as f64; // [0, 1]
            let y = (6.0 * x1).sin() * (6.0 * x2).cos();
            rows.push(vec![y, x1, x2]);
        }
    }
    let n = rows.len();
    let ds = continuous_dataset(&["y", "x1", "x2"], rows);

    let mut options = BTreeMap::new();
    options.insert("bs".to_string(), "gp".to_string()); // gp ⇒ Matérn
    let mut notes = Vec::new();
    let mut basis = build_smooth_basis(
        SmoothKind::S,
        &["x1".to_string(), "x2".to_string()],
        &[1, 2],
        &options,
        &ds,
        &mut notes,
        &ResourcePolicy::default_library(),
        1,
    )
    .expect("build default 2-D matern smooth");

    // (1) The builder must emit typed unresolved Auto provenance, not a
    // baked-in diameter or a magic numeric sentinel.
    let (feature_cols, seeded_length_scale) = match &basis {
        SmoothBasisSpec::Matern {
            feature_cols, spec, ..
        } => (feature_cols.clone(), spec.length_scale),
        other => panic!("expected Matern basis, got {other:?}"),
    };
    assert_eq!(seeded_length_scale, MaternLengthScale::auto());

    // (2) After the shared auto-init runs, the realized length-scale must
    // land in the resolving regime, far below the data diameter. This is
    // the seed the κ-optimizer starts REML from. Since #1731 the Matérn
    // seed is density-adaptive (`auto_initial_length_scale_for_centers`
    // with the requested center count) and since #2252 it uses the
    // rotation-invariant covariance extent `sqrt(12·λ_max)` instead of the
    // rotation-variant per-axis span, so the fitted basin is identical in
    // every rotated frame. Pin bit-equality against that production seed.
    crate::smooth::auto_init_length_scale_in_basis(ds.values.view(), &mut basis);
    let (realized, requested_centers) = match &basis {
        SmoothBasisSpec::Matern { spec, .. } => (
            spec.length_scale
                .resolved()
                .expect("auto-init must resolve Matérn length scale"),
            match &spec.center_strategy {
                CenterStrategy::FarthestPoint { num_centers }
                | CenterStrategy::EqualMass { num_centers }
                | CenterStrategy::EqualMassCovarRepresentative { num_centers }
                | CenterStrategy::KMeans { num_centers, .. } => *num_centers,
                CenterStrategy::Auto(inner) => match inner.as_ref() {
                    CenterStrategy::FarthestPoint { num_centers }
                    | CenterStrategy::EqualMass { num_centers }
                    | CenterStrategy::EqualMassCovarRepresentative { num_centers }
                    | CenterStrategy::KMeans { num_centers, .. } => *num_centers,
                    other => panic!("unexpected inner center strategy: {other:?}"),
                },
                other => panic!("unexpected center strategy: {other:?}"),
            },
        ),
        other => panic!("expected Matern basis after auto-init, got {other:?}"),
    };
    let expected = crate::smooth::auto_initial_length_scale_for_centers(
        ds.values.view(),
        &feature_cols,
        requested_centers,
    );
    assert!(
        (realized - expected).abs() <= 1e-12,
        "auto-init must seed the density-adaptive rotation-invariant \
             wiggly-side length scale (expected {expected}, got {realized})",
    );

    // Sanity: the resolving seed is well below the per-axis range (≈1.0).
    // Before the fix the seed was the full diameter (≈√2 ≈ 1.414); the
    // resolving seed here is ≈ 1.0 / sqrt(576) ≈ 0.042, ~30× smaller.
    let max_range = 1.0_f64; // each axis spans [0, 1]
    assert!(
        realized < max_range / 4.0,
        "matern seed length_scale {realized} must be in the resolving regime, \
             not the over-smoothed diameter corner (n={n}, max_range≈{max_range})",
    );
}

/// gam#979: the BMS entry point asks `all_spatial_terms_kappa_fixed` before
/// any design build. Omitted Matérn scales must therefore be distinguishable
/// from explicit scales both before and after Auto seed resolution.
#[test]
fn matern_length_scale_provenance_drives_prebuild_kappa_locking() {
    let ds = continuous_dataset(
        &["y", "x1", "x2"],
        vec![
            vec![0.0, -1.0, -0.5],
            vec![1.0, -0.2, 0.7],
            vec![0.0, 0.6, -0.8],
            vec![1.0, 1.1, 0.4],
        ],
    );
    let build = |length_scale: Option<&str>| {
        let mut options = BTreeMap::new();
        options.insert("bs".to_string(), "gp".to_string());
        if let Some(value) = length_scale {
            options.insert("length_scale".to_string(), value.to_string());
        }
        let mut notes = Vec::new();
        build_smooth_basis(
            SmoothKind::S,
            &["x1".to_string(), "x2".to_string()],
            &[1, 2],
            &options,
            &ds,
            &mut notes,
            &ResourcePolicy::default_library(),
            1,
        )
        .expect("build Matérn provenance fixture")
    };
    let collection = |basis| TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "spatial".to_string(),
            basis,
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let mut auto = collection(build(None));
    assert!(matches!(
        &auto.smooth_terms[0].basis,
        SmoothBasisSpec::Matern {
            spec: MaternBasisSpec {
                length_scale: MaternLengthScale::Auto { resolved: None },
                ..
            },
            ..
        }
    ));
    assert!(
        !crate::smooth::all_spatial_terms_kappa_fixed(&auto),
        "BMS pre-design query must enroll omitted Matérn κ"
    );
    crate::smooth::auto_init_length_scale_in_place(ds.values.view(), &mut auto.smooth_terms[0]);
    assert!(matches!(
        &auto.smooth_terms[0].basis,
        SmoothBasisSpec::Matern {
            spec: MaternBasisSpec {
                length_scale: MaternLengthScale::Auto {
                    resolved: Some(value)
                },
                ..
            },
            ..
        } if value.is_finite() && *value > 0.0
    ));
    assert!(
        !crate::smooth::all_spatial_terms_kappa_fixed(&auto),
        "resolved Auto Matérn κ must remain optimizer-owned"
    );

    for explicit in ["0.75", "0.0"] {
        let fixed = collection(build(Some(explicit)));
        assert!(matches!(
            &fixed.smooth_terms[0].basis,
            SmoothBasisSpec::Matern {
                spec: MaternBasisSpec {
                    length_scale: MaternLengthScale::Fixed(value),
                    ..
                },
                ..
            } if *value == explicit.parse::<f64>().unwrap()
        ));
        assert!(
            crate::smooth::all_spatial_terms_kappa_fixed(&fixed),
            "explicit Matérn length_scale={explicit} must lock κ before design build"
        );
    }
}

/// gam#1778: `matern(..., periodic=true)` and `thinplate(..., periodic=true)`
/// must be ACCEPTED. The squash-merge that wired periodic support into the
/// matern/thinplate basis specs forgot to add the periodic option keys to
/// those two builders' `validate_known_options` whitelists (only `duchon`
/// got both), so `periodic=`/`period=`/`cyclic=`/`period_start=`/`period_end=`
/// were rejected as unknown options even though the spec/builder consume them.
/// Before the whitelist fix this returned an "unknown option" error.
#[test]
fn matern_and_thinplate_accept_periodic_option() {
    let n = 200usize;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let x = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
            vec![x.sin(), x]
        })
        .collect();
    let ds = continuous_dataset(&["y", "x"], rows);

    // matern() with periodic=true must build without an unknown-option error.
    let mut matern_opts = BTreeMap::new();
    matern_opts.insert("bs".to_string(), "gp".to_string()); // gp ⇒ Matérn
    matern_opts.insert("periodic".to_string(), "true".to_string());
    let mut notes = Vec::new();
    let matern_basis = build_smooth_basis(
        SmoothKind::S,
        &["x".to_string()],
        &[1],
        &matern_opts,
        &ds,
        &mut notes,
        &ResourcePolicy::default_library(),
        1,
    )
    .expect("matern(x, periodic=true) must be accepted");
    match &matern_basis {
        SmoothBasisSpec::Matern { spec, .. } => assert!(
            spec.periodic.is_some(),
            "periodic=true must thread a Some(periodic) into the matern spec",
        ),
        other => panic!("expected Matern basis, got {other:?}"),
    }

    // thinplate()/tps() with periodic=true must likewise be accepted.
    let mut tps_opts = BTreeMap::new();
    tps_opts.insert("bs".to_string(), "tp".to_string());
    tps_opts.insert("periodic".to_string(), "true".to_string());
    let mut notes = Vec::new();
    let tps_basis = build_smooth_basis(
        SmoothKind::S,
        &["x".to_string()],
        &[1],
        &tps_opts,
        &ds,
        &mut notes,
        &ResourcePolicy::default_library(),
        1,
    )
    .expect("thinplate(x, periodic=true) must be accepted");
    match &tps_basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => assert!(
            spec.periodic.is_some(),
            "periodic=true must thread a Some(periodic) into the thinplate spec",
        ),
        other => panic!("expected ThinPlate basis, got {other:?}"),
    }
}

/// Regression: an explicit scalar `periodic=false` on a radial spatial smooth
/// must build a NON-periodic basis. The scalar-boolean shortcut used to emit
/// `Some(vec![None; dim])`, which the 1-D radial builders route on via
/// `spec.periodic.is_some()` (and the Duchon arm even back-fills the data
/// range into a lone `None`), so `periodic=false` silently produced a
/// *periodic* smooth — the opposite of what was asked. The spec's `periodic`
/// field must be `None` for every radial base (matern / thinplate / duchon),
/// matching the bracketed `[false]` form.
#[test]
fn scalar_periodic_false_builds_non_periodic_radial_smooth() {
    let n = 200usize;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let x = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
            vec![x.sin(), x]
        })
        .collect();
    let ds = continuous_dataset(&["y", "x"], rows);

    let build = |bs: &str| -> SmoothBasisSpec {
        let mut opts = BTreeMap::new();
        opts.insert("bs".to_string(), bs.to_string());
        opts.insert("periodic".to_string(), "false".to_string());
        let mut notes = Vec::new();
        build_smooth_basis(
            SmoothKind::S,
            &["x".to_string()],
            &[1],
            &opts,
            &ds,
            &mut notes,
            &ResourcePolicy::default_library(),
            1,
        )
        .unwrap_or_else(|e| panic!("s(x, bs={bs}, periodic=false) must be accepted: {e}"))
    };

    match &build("gp") {
        SmoothBasisSpec::Matern { spec, .. } => assert!(
            spec.periodic.is_none(),
            "periodic=false must leave the matern spec non-periodic, got {:?}",
            spec.periodic
        ),
        other => panic!("expected Matern basis, got {other:?}"),
    }
    match &build("tp") {
        SmoothBasisSpec::ThinPlate { spec, .. } => assert!(
            spec.periodic.is_none(),
            "periodic=false must leave the thinplate spec non-periodic, got {:?}",
            spec.periodic
        ),
        other => panic!("expected ThinPlate basis, got {other:?}"),
    }
    match &build("duchon") {
        SmoothBasisSpec::Duchon { spec, .. } => assert!(
            spec.periodic.is_none(),
            "periodic=false must leave the duchon spec non-periodic (no data-range \
                 back-fill), got {:?}",
            spec.periodic
        ),
        other => panic!("expected Duchon basis, got {other:?}"),
    }
}

fn inferred_tensor_basis_product(ds: &Dataset) -> usize {
    let parsed = parse_formula("y ~ te(theta, h)").expect("parse tensor formula");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        ds,
        &col_map,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("build tensor termspec");
    let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected tensor smooth");
    };
    spec.marginalspecs
        .iter()
        .map(|marginal| match marginal.knotspec {
            BSplineKnotSpec::Generate {
                num_internal_knots, ..
            } => num_internal_knots + marginal.degree + 1,
            BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
            BSplineKnotSpec::Automatic {
                num_internal_knots: Some(num_internal_knots),
                ..
            } => num_internal_knots + marginal.degree + 1,
            BSplineKnotSpec::Automatic {
                num_internal_knots: None,
                ..
            } => panic!("test helper cannot infer automatic knot count"),
            BSplineKnotSpec::Provided(ref knots) => knots.len().saturating_sub(marginal.degree + 1),
            // cr basis dimension equals the knot count (no degree offset).
            BSplineKnotSpec::NaturalCubicRegression { ref knots } => knots.len(),
        })
        .product()
}

fn tensor_margin_basis_sizes(ds: &Dataset, formula: &str) -> Vec<usize> {
    let parsed = parse_formula(formula).expect("parse tensor formula");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        ds,
        &col_map,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("build tensor termspec");
    let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected tensor smooth");
    };
    spec.marginalspecs
        .iter()
        .map(|marginal| match marginal.knotspec {
            BSplineKnotSpec::Generate {
                num_internal_knots, ..
            } => num_internal_knots + marginal.degree + 1,
            BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
            BSplineKnotSpec::Automatic {
                num_internal_knots: Some(num_internal_knots),
                ..
            } => num_internal_knots + marginal.degree + 1,
            BSplineKnotSpec::Automatic {
                num_internal_knots: None,
                ..
            } => panic!("test helper cannot infer automatic knot count"),
            BSplineKnotSpec::Provided(ref knots) => knots.len().saturating_sub(marginal.degree + 1),
            // cr basis dimension equals the knot count (no degree offset).
            BSplineKnotSpec::NaturalCubicRegression { ref knots } => knots.len(),
        })
        .collect()
}

#[test]
fn validate_known_options_lists_valid_option_names_for_unknown_parameter() {
    let mut options = BTreeMap::new();
    options.insert("lengt_scale".to_string(), "0.25".to_string());
    let err = validate_known_options(
        "matern",
        &options,
        &["type", "bs", "length_scale", "centers", "k", "nu"],
    )
    .expect_err("unknown smooth option should be rejected");
    assert!(
        err.contains("matern() does not accept option `lengt_scale`"),
        "error should name the invalid option, got: {err}"
    );
    assert!(
        err.contains("did you mean one of [length_scale]"),
        "error should suggest the closest valid option, got: {err}"
    );
    assert!(
        err.contains("Valid options: ["),
        "error should list valid option names, got: {err}"
    );
}

#[test]
fn validate_known_options_exempts_the_engine_injected_namespace() {
    // The pipeline injects `__by_col` (BySmooth) and `__secondary_center_cap`
    // (secondary-predictor parsimony) into a term's option map; neither is a
    // user key, so no arm's whitelist has to know them and none may refuse them.
    assert!(is_engine_option(SECONDARY_CENTER_CAP_OPTION));
    assert!(is_engine_option("__by_col"));
    assert!(!is_engine_option("by"));
    for (arm, known) in [
        ("thinplate", THINPLATE_SMOOTH_OPTION_KEYS),
        ("matern", MATERN_SMOOTH_OPTION_KEYS),
        ("bspline", BSPLINE_SMOOTH_OPTION_KEYS),
        ("cyclic", CYCLIC_SMOOTH_OPTION_KEYS),
    ] {
        let mut options = BTreeMap::new();
        options.insert("__by_col".to_string(), "2".to_string());
        options.insert(SECONDARY_CENTER_CAP_OPTION.to_string(), "12".to_string());
        validate_known_options(arm, &options, known)
            .unwrap_or_else(|err| panic!("{arm}: engine-injected options must pass: {err}"));
        assert!(
            !known.iter().any(|key| is_engine_option(key)),
            "{arm}: an engine-injected key is not part of a user-facing vocabulary: {known:?}"
        );

        // A genuine typo is still refused, and the vocabulary shown to the user
        // contains no engine key.
        options.insert("lengt_scale".to_string(), "0.25".to_string());
        let err = validate_known_options(arm, &options, known)
            .expect_err("a misspelled user option must still be rejected");
        assert!(err.contains("does not accept option `lengt_scale`"), "{err}");
        assert!(!err.contains("__"), "the valid-option list advertised an engine key: {err}");
    }
}

#[test]
fn tensor_k_accepts_square_bracket_per_margin_list() {
    let ds = continuous_dataset(
        &["y", "x", "z"],
        (0..40)
            .map(|i| {
                let x = i as f64 / 39.0;
                let z = ((i * 7) % 40) as f64 / 39.0;
                vec![x.sin() + z.cos(), x, z]
            })
            .collect(),
    );

    assert_eq!(
        tensor_margin_basis_sizes(&ds, "y ~ te(x, z, k=[5, 6])"),
        vec![5, 6],
        "square-bracket k lists should materialize the requested per-margin values"
    );
}

/// #1776 / #1752: a bare doubly-cyclic tensor `te(x, z, bs=c('cc','cc'))`
/// with NO explicit `period=` must build — each cyclic margin wraps on its
/// own observed `[min, max]` data span (mirroring mgcv's `bs="cc"` and the
/// 1-D cyclic fallback), instead of hard-erroring "periodic but requires an
/// explicit period". The periodic-radial refactor (c8c3192fa) replaced that
/// fallback with an unconditional `period=`-required error and orphaned the
/// `margin_is_cc` binding that drives it (the #1776 dead-binding `-D
/// warnings` build break). This pins the restored data-range derivation so a
/// regression that drops the `None if margin_is_cc` branch trips here, fast,
/// with no fit/optimizer in the loop.
#[test]
fn bare_doubly_cyclic_tensor_derives_period_from_data_range_1776() {
    let ds = continuous_dataset(
        &["y", "x", "z"],
        (0..40)
            .map(|i| {
                let x = i as f64 / 39.0;
                let z = ((i * 7) % 40) as f64 / 39.0;
                vec![x.sin() + z.cos(), x, z]
            })
            .collect(),
    );

    let parsed =
        parse_formula("y ~ te(x, z, bs=c('cc','cc'))").expect("parse doubly-cyclic tensor formula");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    // Must NOT hard-error: the bare cyclic margins derive their period from
    // the observed data range (the restored #1752 fallback).
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect(
        "bare cc-cc tensor must build via the data-range period fallback (#1776/#1752), \
             not hard-error on a missing explicit period",
    );
    let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected tensor smooth");
    };
    assert_eq!(
        spec.marginalspecs.len(),
        2,
        "te(x, z) builds exactly two tensor margins"
    );
    for (axis, marginal) in spec.marginalspecs.iter().enumerate() {
        assert!(
            matches!(marginal.knotspec, BSplineKnotSpec::PeriodicUniform { .. }),
            "cyclic margin {axis} must build a periodic (wrapped) knotspec from the \
                 data range, got {:?}",
            marginal.knotspec
        );
    }
}

#[test]
fn parse_cylinder_periodic_options_match_requested_forms() {
    let mut opts = BTreeMap::new();
    opts.insert("periodic".to_string(), "[0]".to_string());
    opts.insert("period".to_string(), "[2*pi, None]".to_string());
    let axes = parse_periodic_axes(&opts, 2).expect("axes");
    let periods = parse_periods(&opts, &axes).expect("periods");
    assert_eq!(axes, vec![true, false]);
    assert!((periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
    assert_eq!(periods[1], None);

    let mut boundary_opts = BTreeMap::new();
    boundary_opts.insert(
        "boundary".to_string(),
        "['periodic', 'natural']".to_string(),
    );
    boundary_opts.insert("period".to_string(), "[2*pi, None]".to_string());
    let boundary_axes = parse_periodic_axes(&boundary_opts, 2).expect("boundary axes");
    let boundary_periods = parse_periods(&boundary_opts, &boundary_axes).expect("boundary periods");
    assert_eq!(boundary_axes, vec![true, false]);
    assert!((boundary_periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
    assert_eq!(boundary_periods[1], None);

    let mut unicode_opts = BTreeMap::new();
    unicode_opts.insert("periodic".to_string(), "[0,1]".to_string());
    unicode_opts.insert("period".to_string(), "[2π, τ]".to_string());
    let unicode_axes = parse_periodic_axes(&unicode_opts, 2).expect("unicode axes");
    let unicode_periods = parse_periods(&unicode_opts, &unicode_axes).expect("unicode periods");
    assert_eq!(unicode_axes, vec![true, true]);
    assert!((unicode_periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
    assert!((unicode_periods[1].unwrap() - std::f64::consts::TAU).abs() < 1e-12);
}

/// The tensor boundary-token guard must ACCEPT `clamped`/`open` (the
/// B-spline-clamped, non-periodic margin spelling) alongside the periodic
/// selectors and the other inert non-periodic markers, and still REJECT a
/// genuine endpoint constraint like `anchored`. This locks the #415 /
/// cylinder fix (`te(theta, z, boundary=['periodic','clamped'])`, mgcv
/// `te(bs=c("cc","ps"))`) in the fast unit lane — the end-to-end cylinder
/// recovery test is R-gated (`run_r` + mgcv), so without this the guard
/// regressing back to rejecting `clamped` would slip through CPU CI.
#[test]
fn tensor_boundary_tokens_accept_clamped_open_reject_anchored() {
    fn boundary(raw: &str, dim: usize) -> Result<(), String> {
        let mut opts = BTreeMap::new();
        opts.insert("boundary".to_string(), raw.to_string());
        validate_tensor_boundary_tokens(&opts, dim)
    }

    // Mixed periodic + clamped (the cylinder) and its bare/case/quote
    // variants are all accepted.
    for raw in [
        "['periodic', 'clamped']",
        "['periodic', 'open']",
        "['cc', 'clamped']",
        "['clamped', 'natural']",
        "[Periodic, CLAMPED]",
        "c('cc', 'clamped')", // mgcv-style c(...) vector form round-trips
    ] {
        assert!(
            boundary(raw, 2).is_ok(),
            "boundary={raw:?} must be accepted (clamped/open/inert non-periodic markers)"
        );
    }

    // `bc=` is an accepted alias for `boundary=`.
    let mut bc_opts = BTreeMap::new();
    bc_opts.insert("bc".to_string(), "['periodic', 'clamped']".to_string());
    assert!(validate_tensor_boundary_tokens(&bc_opts, 2).is_ok());

    // A genuine endpoint constraint has no ordinary-margin meaning on a
    // tensor and must still be surfaced as a clean unsupported-feature error
    // rather than silently dropped.
    let err = boundary("['periodic', 'anchored']", 2)
        .expect_err("anchored endpoint constraint must be rejected on a tensor margin");
    assert!(
        err.contains("anchored") && err.contains("not supported"),
        "rejection must name the offending token and be an unsupported-feature error: {err}"
    );

    // Absent boundary/bc is a no-op success.
    assert!(validate_tensor_boundary_tokens(&BTreeMap::new(), 2).is_ok());
}

#[test]
fn parse_single_axis_periodic_zero_as_axis_not_false() {
    let mut opts = BTreeMap::new();
    opts.insert("periodic".to_string(), "[0]".to_string());
    opts.insert("period".to_string(), "2*pi".to_string());
    opts.insert("origin".to_string(), "0".to_string());
    let axes = parse_periodic_axes(&opts, 1).expect("axes");
    let periods = parse_periods(&opts, &axes).expect("periods");
    let origins = parse_period_origins(&opts, &axes).expect("origins");
    assert_eq!(axes, vec![true]);
    assert!((periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
    assert_eq!(origins[0], Some(0.0));
}

#[test]
fn one_dimensional_bspline_accepts_boundary_periodic() {
    let ds = continuous_dataset(
        &["y", "theta"],
        (0..16)
            .map(|i| {
                let theta = std::f64::consts::TAU * i as f64 / 16.0;
                vec![theta.sin(), theta]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ s(theta, boundary=periodic, period=2*pi, origin=0, k=8)")
        .expect("parse");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("periodic boundary should build");
    let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected 1D B-spline");
    };
    assert!(matches!(
        &spec.knotspec,
        BSplineKnotSpec::PeriodicUniform {
            data_range,
            num_basis: 8
        } if *data_range == (0.0, std::f64::consts::TAU)
    ));
}

#[test]
fn univariate_smooth_accepts_mgcv_cubic_regression_aliases() {
    let ds = continuous_dataset(
        &["y", "x"],
        (0..32)
            .map(|i| {
                let x = i as f64 / 31.0;
                vec![x * x, x]
            })
            .collect(),
    );
    let col_map = ds.column_map();

    for selector in ["cr", "cs"] {
        let formula = format!("y ~ s(x, bs='{selector}')");
        let parsed = parse_formula(&formula).expect("parse cr/cs smooth");
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &gam_runtime::resource::ResourcePolicy::default_library(),
        )
        .unwrap_or_else(|err| panic!("bs='{selector}' must build a 1-D smooth, got: {err:?}"));
        let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!(
                "bs='{selector}' must lower to a BSpline1D; got {:?}",
                terms.smooth_terms[0].basis
            );
        };
        assert!(
            spec.double_penalty,
            "bs='{selector}' must recover its null space by default"
        );

        let opt_out = format!("y ~ s(x, bs='{selector}', double_penalty=false)");
        let parsed = parse_formula(&opt_out).expect("parse explicit null-shrinkage opt-out");
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &gam_runtime::resource::ResourcePolicy::default_library(),
        )
        .expect("explicit cr/cs opt-out should build");
        let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("bs='{selector}' must lower to a BSpline1D");
        };
        assert!(!spec.double_penalty, "explicit opt-out must be preserved");
    }
}

#[test]
fn non_intercept_linear_effects_default_to_mle_with_explicit_null_recovery() {
    let ds = continuous_dataset(
        &["y", "x", "z"],
        (0..24)
            .map(|i| {
                let x = i as f64 / 23.0;
                let z = 1.0 - x;
                vec![x - z, x, z]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ x + z + x:z").expect("parse linear defaults");
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &ds.column_map(),
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build linear defaults");
    assert!(!terms.linear_terms.is_empty());
    assert!(
        terms.linear_terms.iter().all(|term| !term.double_penalty),
        "ordinary parametric effects must be unpenalized by default: {:?}",
        terms
            .linear_terms
            .iter()
            .map(|term| (&term.name, term.double_penalty))
            .collect::<Vec<_>>()
    );

    // `bounded()` is an exact interval transform and likewise defaults to
    // no shrinkage ridge. It also structurally rejects combining the
    // interval geometry with `double_penalty`.
    let bounded_parsed =
        parse_formula("y ~ bounded(z, min=-2, max=2)").expect("parse bounded defaults");
    let mut bounded_notes = Vec::new();
    let bounded_terms = build_termspec(
        &bounded_parsed.terms,
        &ds,
        &ds.column_map(),
        &mut bounded_notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build bounded defaults");
    assert_eq!(bounded_terms.linear_terms.len(), 1);
    assert!(
        !bounded_terms.linear_terms[0].double_penalty,
        "bounded() must default double_penalty=false since it cannot combine with the interval transform"
    );

    for formula in [
        "y ~ linear(x, double_penalty=true)",
        "y ~ linear(x:z, double_penalty=true)",
    ] {
        let parsed = parse_formula(formula).expect("parse explicit linear shrinkage");
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &ds.column_map(),
            &mut notes,
            &gam_runtime::resource::ResourcePolicy::default_library(),
        )
        .unwrap_or_else(|error| panic!("{formula} must build: {error}"));
        assert_eq!(terms.linear_terms.len(), 1, "{formula}");
        assert!(
            terms.linear_terms[0].double_penalty,
            "{formula} must preserve the explicit shrinkage opt-in"
        );
    }

    assert!(
        parse_formula("y ~ linear(x, double_penalty=ture)").is_err(),
        "a misspelled opt-in must be rejected instead of silently using the default"
    );
}

#[test]
fn tensor_smooths_default_to_joint_null_recovery_with_explicit_opt_out() {
    let ds = continuous_dataset(
        &["y", "x", "z"],
        (0..36)
            .map(|i| {
                let x = i as f64 / 35.0;
                let z = ((i * 11) % 36) as f64 / 35.0;
                vec![x * z, x, z]
            })
            .collect(),
    );
    let col_map = ds.column_map();
    for constructor in ["te", "ti", "t2"] {
        for (option, expected) in [("", true), (", double_penalty=false", false)] {
            let formula = format!("y ~ {constructor}(x, z{option})");
            let parsed = parse_formula(&formula).expect("parse tensor default");
            let mut notes = Vec::new();
            let terms = build_termspec(
                &parsed.terms,
                &ds,
                &col_map,
                &mut notes,
                &gam_runtime::resource::ResourcePolicy::default_library(),
            )
            .unwrap_or_else(|error| panic!("{formula} must build: {error}"));
            let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
                panic!("{formula} must lower to TensorBSpline");
            };
            assert_eq!(spec.double_penalty, expected, "{formula}");
        }
    }
}

#[test]
fn univariate_ps_small_k_degree_reduces_through_build(/* gam#1130 */) {
    // mgcv accepts `s(x, bs="ps", k=3)` (and the default cubic-regression
    // `s(x, k=3)`) by silently reducing the cubic basis to a quadratic.
    // The univariate ps/bspline build path used to reject this with
    // "k too small for degree 3"; it must now lower to a degree-2 basis
    // with zero internal knots (num_basis = k = 3), matching the te(...)
    // margin behaviour fixed in b75f55a91. Verified across the ps alias
    // and the default (cr) selector that both route through
    // parse_ps_internal_knots.
    let ds = continuous_dataset(
        &["y", "x"],
        (0..32)
            .map(|i| {
                let x = i as f64 / 31.0;
                vec![x * x, x]
            })
            .collect(),
    );
    let col_map = ds.column_map();

    for formula in ["y ~ s(x, bs='ps', k=3)", "y ~ s(x, k=3)"] {
        let parsed = parse_formula(formula).expect("parse small-k ps/cr smooth");
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &gam_runtime::resource::ResourcePolicy::default_library(),
        )
        .unwrap_or_else(|err| panic!("`{formula}` must degree-reduce, not error; got: {err:?}"));
        let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!(
                "`{formula}` must lower to a BSpline1D; got {:?}",
                terms.smooth_terms[0].basis
            );
        };
        assert_eq!(
            spec.degree, 2,
            "`{formula}` must drop the cubic default to a quadratic basis"
        );
        let num_internal = match &spec.knotspec {
            BSplineKnotSpec::Generate {
                num_internal_knots, ..
            } => *num_internal_knots,
            BSplineKnotSpec::Automatic {
                num_internal_knots: Some(n),
                ..
            } => *n,
            other => panic!("`{formula}` unexpected knotspec: {other:?}"),
        };
        assert_eq!(
            num_internal, 0,
            "`{formula}` must have zero internal knots (num_basis = k = 3)"
        );
        // Resulting basis dimension is num_internal + degree + 1 = 3 = k.
        assert!(
            spec.penalty_order >= 1 && spec.penalty_order <= spec.degree,
            "`{formula}` penalty_order {} must satisfy 1 <= order <= degree={}",
            spec.penalty_order,
            spec.degree
        );
    }
}

#[test]
fn formula_shape_constraint_round_trips_and_rejects_bogus() {
    let ds = continuous_dataset(
        &["y", "x"],
        (0..32)
            .map(|i| {
                let x = i as f64 / 31.0;
                vec![x * x, x]
            })
            .collect(),
    );
    let col_map = ds.column_map();

    let parsed =
        parse_formula("y ~ s(x, shape=monotone_increasing)").expect("parse monotone smooth");
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("monotone smooth should build");
    assert_eq!(
        terms.smooth_terms[0].shape,
        ShapeConstraint::MonotoneIncreasing
    );

    let parsed_bad = parse_formula("y ~ s(x, shape=bogus)").expect("parse bogus shape");
    let mut notes_bad = Vec::new();
    let err = build_termspec(
        &parsed_bad.terms,
        &ds,
        &col_map,
        &mut notes_bad,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect_err("bogus shape must error");
    assert!(
        format!("{err:?}").contains("unknown shape constraint"),
        "got: {err:?}"
    );
}

#[test]
fn default_sphere_smooth_uses_spherical_farthest_point_centers() {
    let ds = continuous_dataset(
        &["y", "lat", "lon"],
        (0..24)
            .map(|i| {
                let t = i as f64 / 24.0;
                let lat = -60.0 + 120.0 * t;
                let lon = -180.0 + 360.0 * ((7 * i) % 24) as f64 / 24.0;
                vec![lat.to_radians().sin(), lat, lon]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ sphere(lat, lon)").expect("parse");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build sphere termspec");
    let SmoothBasisSpec::Sphere { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected sphere term");
    };
    assert!(matches!(
        spec.center_strategy,
        CenterStrategy::FarthestPoint { .. }
    ));
}

#[test]
fn one_dimensional_duchon_defaults_to_scale_free_length_scale() {
    let ds = continuous_dataset(
        &["y", "x"],
        (0..32)
            .map(|i| {
                let x = i as f64 / 31.0;
                vec![(std::f64::consts::TAU * x).sin(), x]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ duchon(x)").expect("parse");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build default duchon termspec");
    let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected Duchon term");
    };
    assert_eq!(spec.length_scale, None);
    assert!(matches!(
        spec.center_strategy,
        CenterStrategy::Auto(ref inner)
            if matches!(
                inner.as_ref(),
                CenterStrategy::UniformGrid { .. }
            )
    ));
}

#[test]
fn formula_duchon_default_does_not_enable_collocation_operators() {
    let ds = continuous_dataset(
        &["y", "x", "z"],
        (0..40)
            .map(|i| {
                let x = (i as f64 / 39.0).fract();
                let z = ((7 * i) as f64 / 39.0).fract();
                vec![x + z, x, z]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ duchon(x, z)").expect("parse");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build default 2D duchon termspec");
    let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected Duchon term");
    };
    assert!(matches!(
        spec.operator_penalties.mass,
        OperatorPenaltySpec::Disabled
    ));
    assert!(matches!(
        spec.operator_penalties.tension,
        OperatorPenaltySpec::Disabled
    ));
    assert!(matches!(
        spec.operator_penalties.stiffness,
        OperatorPenaltySpec::Disabled
    ));
}

#[test]
fn one_dimensional_duchon_length_scale_opts_into_hybrid_mode() {
    let ds = continuous_dataset(
        &["y", "x"],
        (0..32)
            .map(|i| {
                let x = i as f64 / 31.0;
                vec![(std::f64::consts::TAU * x).sin(), x]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ duchon(x, length_scale=0.25)").expect("parse");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build hybrid duchon termspec");
    let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected Duchon term");
    };
    assert_eq!(spec.length_scale, Some(0.25));
}

#[test]
fn multidimensional_duchon_default_uses_low_rank_mgcv_sized_basis() {
    let ds = continuous_dataset(
        &["y", "x1", "x2"],
        (0..500)
            .map(|i| {
                let x1 = 2.0 * (i as f64 / 499.0) - 1.0;
                let x2 = (((37 * i) % 500) as f64 / 499.0) * 2.0 - 1.0;
                vec![(2.0 * x1).sin() + (1.5 * x2).cos(), x1, x2]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ duchon(x1, x2)").expect("parse");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build default 2D duchon termspec");
    let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected Duchon term");
    };
    let CenterStrategy::Auto(inner) = &spec.center_strategy else {
        panic!("expected auto center strategy");
    };
    assert!(matches!(
        inner.as_ref(),
        CenterStrategy::FarthestPoint { num_centers: 30 }
    ));
}

#[test]
fn spectral_duchon_reproduces_fixed_seed_uniform_landmarks() {
    let ds = continuous_dataset(
        &["y", "x1", "x2", "x3", "x4"],
        (0..64)
            .map(|i| {
                let x = i as f64 / 63.0;
                vec![
                    x.sin(),
                    x,
                    (3.0 * x).sin(),
                    (5.0 * x).cos(),
                    (7.0 * x).sin(),
                ]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ duchon(x1, x2, x3, x4, rank=6, order=0)").expect("parse");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build spectral Duchon termspec");
    let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected Duchon term");
    };
    let CenterStrategy::DuchonSpectral { knots, basis } = &spec.center_strategy else {
        panic!("expected spectral center strategy");
    };
    assert_eq!(basis.rank(), 6);
    let CenterStrategy::UserProvided(centers) = knots.as_ref() else {
        panic!("expected frozen sampled centers");
    };
    assert_eq!(centers.dim(), (64, 4));
}

#[test]
fn parse_matern_nu_accepts_equivalent_half_integer_forms() {
    let cases = [
        ("1/2", MaternNu::Half),
        (" 1 / 2 ", MaternNu::Half),
        (".5", MaternNu::Half),
        ("0.50", MaternNu::Half),
        ("half", MaternNu::Half),
        ("3 / 2", MaternNu::ThreeHalves),
        ("1.50", MaternNu::ThreeHalves),
        ("5 / 2", MaternNu::FiveHalves),
        ("2.500000000000", MaternNu::FiveHalves),
        ("7 / 2", MaternNu::SevenHalves),
        ("3.50", MaternNu::SevenHalves),
        ("9 / 2", MaternNu::NineHalves),
        ("4.50", MaternNu::NineHalves),
    ];
    for (raw, expected) in cases {
        let parsed = parse_matern_nu(raw).expect(raw);
        assert!(
            matches!(
                (parsed, expected),
                (MaternNu::Half, MaternNu::Half)
                    | (MaternNu::ThreeHalves, MaternNu::ThreeHalves)
                    | (MaternNu::FiveHalves, MaternNu::FiveHalves)
                    | (MaternNu::SevenHalves, MaternNu::SevenHalves)
                    | (MaternNu::NineHalves, MaternNu::NineHalves)
            ),
            "parsed {raw:?} as {parsed:?}, expected {expected:?}"
        );
    }
}

#[test]
fn parse_matern_nu_rejects_unsupported_or_invalid_values() {
    for raw in ["1", "2", "11/2", "1/0", "nan", "fast"] {
        let err = parse_matern_nu(raw).expect_err(raw);
        assert!(
            err.contains("supported half-integer values"),
            "unexpected error for {raw:?}: {err}"
        );
    }
}

#[test]
fn parse_ps_k_is_honoured_exactly_down_to_degree_plus_one() {
    // `k = internal_knots + degree + 1` for EVERY explicit `k`: the
    // four-function cubic `k=4` is zero internal knots, the same basis
    // `knots=0` names. A floor of two internal knots used to turn `k=4` and
    // `k=5` into the six-function basis silently.
    let mut opts = BTreeMap::new();
    opts.insert("k".to_string(), "4".to_string());
    let (internal, inferred, eff_degree) = parse_ps_internal_knots(&opts, 3, 20).expect("k=4");
    assert_eq!(internal, 0);
    assert_eq!(eff_degree, 3);
    assert!(!inferred);

    opts.insert("k".to_string(), "5".to_string());
    let (internal, inferred, eff_degree) = parse_ps_internal_knots(&opts, 3, 20).expect("k=5");
    assert_eq!(internal, 1);
    assert_eq!(eff_degree, 3);
    assert!(!inferred);

    opts.insert("k".to_string(), "6".to_string());
    let (internal, inferred, eff_degree) = parse_ps_internal_knots(&opts, 3, 20).expect("k=6");
    assert_eq!(internal, 2);
    assert_eq!(eff_degree, 3);
    assert!(!inferred);

    opts.insert("k".to_string(), "10".to_string());
    let (internal, inferred, eff_degree) = parse_ps_internal_knots(&opts, 3, 20).expect("k=10");
    assert_eq!(internal, 6);
    assert_eq!(eff_degree, 3);
    assert!(!inferred);
}

#[test]
fn parse_ps_internal_knots_drops_degree_for_small_k() {
    // mgcv's `s(x, bs="ps", k=3)` with the default cubic basis silently
    // reduces to a quadratic (`degree=2`) marginal. `k=3, degree=3`
    // should yield a quadratic basis with zero internal knots
    // (`num_basis = k = 3`).
    let mut opts = BTreeMap::new();
    opts.insert("k".to_string(), "3".to_string());
    let (internal, inferred, eff_degree) = parse_ps_internal_knots(&opts, 3, 20).expect("k=3");
    assert_eq!(eff_degree, 2);
    assert_eq!(internal, 0);
    assert!(!inferred);

    // `k=2` reduces to a linear (`degree=1`) marginal — the smallest
    // non-trivial spline basis.
    opts.insert("k".to_string(), "2".to_string());
    let (internal, inferred, eff_degree) = parse_ps_internal_knots(&opts, 3, 20).expect("k=2");
    assert_eq!(eff_degree, 1);
    assert_eq!(internal, 0);
    assert!(!inferred);

    // The under-2 case is structurally under-specified and rejected even
    // by the degree-reducing variant: no B-spline basis has fewer than
    // two functions.
    opts.insert("k".to_string(), "1".to_string());
    let err = parse_ps_internal_knots(&opts, 3, 20)
        .expect_err("k=1 is below the irreducible spline floor");
    assert!(err.contains("requires k >= 2"), "unexpected error: {err}");

    // `k = degree + 1` is the first un-reduced size: the full cubic degree
    // with zero internal knots, `num_basis = k = 4` exactly.
    opts.insert("k".to_string(), "4".to_string());
    let (internal, inferred, eff_degree) = parse_ps_internal_knots(&opts, 3, 20).expect("k=4");
    assert_eq!(eff_degree, 3);
    assert_eq!(internal, 0);
    assert!(!inferred);
}

#[test]
fn factor_smooth_marginal_degree_reduces_for_small_k() {
    let ds = factor_dataset();
    let col_map = ds.column_map();

    for (k, expected_degree) in [(3usize, 2usize), (2usize, 1usize)] {
        let parsed =
            parse_formula(&format!("y ~ s(x, g, bs=fs, k={k})")).expect("parse factor smooth");
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &gam_runtime::resource::ResourcePolicy::default_library(),
        )
        .unwrap_or_else(|err| panic!("fs k={k} should degree-reduce, got: {err:?}"));
        let SmoothBasisSpec::FactorSmooth { spec } = &terms.smooth_terms[0].basis else {
            panic!(
                "expected factor smooth, got {:?}",
                terms.smooth_terms[0].basis
            );
        };
        assert_eq!(spec.marginal.degree, expected_degree);
        assert!(
            spec.marginal.penalty_order <= spec.marginal.degree,
            "penalty_order {} must be clamped to degree {}",
            spec.marginal.penalty_order,
            spec.marginal.degree
        );
        let basis_size = match spec.marginal.knotspec {
            BSplineKnotSpec::Generate {
                num_internal_knots, ..
            } => num_internal_knots + spec.marginal.degree + 1,
            BSplineKnotSpec::Automatic {
                num_internal_knots: Some(num_internal_knots),
                ..
            } => num_internal_knots + spec.marginal.degree + 1,
            ref other => panic!("unexpected factor-smooth knotspec: {other:?}"),
        };
        assert_eq!(basis_size, k);
    }
}

/// Build a dataset with a ternary continuous covariate `x ∈ {0,1,2}` and a
/// 2-level categorical group `g`, for the low-cardinality cr-cap tests.
fn ternary_factor_dataset() -> Dataset {
    let rows = (0..120)
        .map(|i| {
            let x = (i % 3) as f64;
            let g = (i % 2) as f64;
            vec![x + g, x, g]
        })
        .collect::<Vec<_>>();
    Dataset {
        headers: vec!["y".into(), "x".into(), "g".into()],
        values: Array2::from_shape_vec(
            (rows.len(), 3),
            rows.into_iter().flat_map(|row| row.into_iter()).collect(),
        )
        .expect("rectangular ternary factor test data"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "g".into(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".into(), "b".into()],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
        ],
    }
}

#[test]
fn univariate_cr_smooth_caps_knots_to_data_support() {
    // #1541: `s(x, bs=cr, k=10)` on a ternary covariate (3 distinct values)
    // must NOT hard-fail in cr-knot selection ("cubic regression spline with
    // k=10 requires at least 10 distinct values, got 3"). The cr basis is
    // capped to the data support — exactly 3 value-knots at {0,1,2} — which
    // is full-rank for the data, so it can still represent any 3 group means.
    let ds = continuous_dataset(
        &["y", "x"],
        (0..90)
            .map(|i| vec![(i % 3) as f64, (i % 3) as f64])
            .collect(),
    );
    let col_map = ds.column_map();
    let parsed = parse_formula("y ~ s(x, bs=cr, k=10)").expect("parse cr smooth");
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("cr k=10 must cap to data support instead of erroring");
    let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected BSpline1D for s(x, bs=cr)");
    };
    let BSplineKnotSpec::NaturalCubicRegression { knots } = &spec.knotspec else {
        panic!("expected cr knotspec, got {:?}", spec.knotspec);
    };
    // Capped to exactly the 3 distinct covariate values.
    assert_eq!(knots.len(), 3, "cr basis not capped to 3 distinct values");
    assert_eq!(knots.as_slice().unwrap(), &[0.0, 1.0, 2.0]);
    // The reduction is surfaced to the user (mgcv warns in the same case).
    assert!(
        notes.iter().any(|n| n.contains("data-support cap")),
        "cap not reported in inference notes: {notes:?}"
    );
}

#[test]
fn univariate_cr_smooth_binary_covariate_degrades_to_bspline() {
    // #1541: a BINARY covariate has too few distinct values (2) for ANY cr
    // spline (needs >= 3 distinct). `s(x, bs=cr)` must degrade to a B-spline
    // marginal — the default basis the same data already fits — NOT hard-fail.
    let ds = continuous_dataset(
        &["y", "x"],
        (0..80)
            .map(|i| vec![(i % 2) as f64, (i % 2) as f64])
            .collect(),
    );
    let col_map = ds.column_map();
    let parsed = parse_formula("y ~ s(x, bs=cr, k=10)").expect("parse cr smooth");
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("binary cr must degrade to B-spline instead of erroring");
    let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected BSpline1D for s(x, bs=cr)");
    };
    assert!(
        !matches!(
            spec.knotspec,
            BSplineKnotSpec::NaturalCubicRegression { .. }
        ),
        "binary covariate must NOT build a cr basis, got {:?}",
        spec.knotspec
    );
    assert!(
        notes
            .iter()
            .any(|n| n.contains("Degraded to the linear B-spline")),
        "degradation not reported in inference notes: {notes:?}"
    );
}

/// #2783: `identifiability=` is parsed on the 1-D B-spline path, not
/// whitelisted-and-discarded. Walk the whole accepted vocabulary and the
/// three refusals in one place, so a future arm that forgets to call the
/// resolver cannot quietly reintroduce the inert option.
#[test]
fn one_dimensional_identifiability_option_is_parsed_and_validated() {
    let mut options = BTreeMap::new();

    // Absent: the caller's structural default is returned untouched.
    assert!(matches!(
        parse_bspline_identifiability(&options).expect("absent option parses"),
        None
    ));
    assert!(matches!(
        resolve_bspline_identifiability(
            &options,
            BSplineIdentifiability::None,
            BSplineIdentifiabilityContext::default(),
        )
        .expect("absent option keeps the structural default"),
        BSplineIdentifiability::None
    ));

    for token in ["none", "None", " NONE "] {
        options.insert("identifiability".to_string(), token.to_string());
        assert!(
            matches!(
                parse_bspline_identifiability(&options).expect("none parses"),
                Some(BSplineIdentifiability::None)
            ),
            "token {token:?} should select the unconstrained policy"
        );
    }
    for token in [
        "sum_tozero",
        "sum-to-zero",
        "sumtozero",
        "centered",
        "center_sum_tozero",
        "center-sum-to-zero",
    ] {
        options.insert("identifiability".to_string(), token.to_string());
        assert!(
            matches!(
                parse_bspline_identifiability(&options).expect("sum-to-zero parses"),
                Some(BSplineIdentifiability::WeightedSumToZero { weights: None })
            ),
            "token {token:?} should select sum-to-zero centering"
        );
    }
    for token in [
        "linear",
        "remove_linear_trend",
        "remove-linear-trend",
        "center_linear_orthogonal",
    ] {
        options.insert("identifiability".to_string(), token.to_string());
        assert!(
            matches!(
                parse_bspline_identifiability(&options).expect("linear parses"),
                Some(BSplineIdentifiability::RemoveLinearTrend)
            ),
            "token {token:?} should select the constant+linear removal"
        );
    }

    // An explicit token overrides the structural default in both directions.
    options.insert("identifiability".to_string(), "none".to_string());
    assert!(matches!(
        resolve_bspline_identifiability(
            &options,
            BSplineIdentifiability::default(),
            BSplineIdentifiabilityContext::default(),
        )
        .expect("explicit none overrides the centering default"),
        BSplineIdentifiability::None
    ));
    options.insert("identifiability".to_string(), "sum_tozero".to_string());
    assert!(matches!(
        resolve_bspline_identifiability(
            &options,
            BSplineIdentifiability::None,
            BSplineIdentifiabilityContext::default(),
        )
        .expect("explicit sum_tozero overrides an unconstrained default"),
        BSplineIdentifiability::WeightedSumToZero { weights: None }
    ));

    // Internal-only variants say so rather than pretending to be unknown.
    for token in ["frozen", "orthogonal"] {
        options.insert("identifiability".to_string(), token.to_string());
        let err = parse_bspline_identifiability(&options)
            .expect_err("internal-only policy must be refused");
        assert!(
            err.contains("internal-only"),
            "token {token:?} should be refused as internal-only, got: {err}"
        );
    }

    // An unknown token is refused, naming the option and the alternatives —
    // the behaviour every sibling smooth kind already had.
    options.insert("identifiability".to_string(), "totally_bogus".to_string());
    let err = parse_bspline_identifiability(&options)
        .expect_err("an unknown identifiability token must be refused");
    assert!(
        err.contains("totally_bogus") && err.contains("none, sum_tozero, linear"),
        "unknown-token error should name the token and the vocabulary, got: {err}"
    );

    // Refusal 1: an anchored endpoint already fixes the level.
    options.insert("identifiability".to_string(), "sum_tozero".to_string());
    let err = resolve_bspline_identifiability(
        &options,
        BSplineIdentifiability::None,
        BSplineIdentifiabilityContext {
            has_anchor: true,
            ..Default::default()
        },
    )
    .expect_err("anchor + centering is over-constrained");
    assert!(
        err.contains("anchored endpoint"),
        "anchor conflict should explain itself, got: {err}"
    );
    // ...but agreeing with the structural default is fine.
    options.insert("identifiability".to_string(), "none".to_string());
    assert!(matches!(
        resolve_bspline_identifiability(
            &options,
            BSplineIdentifiability::None,
            BSplineIdentifiabilityContext {
                has_anchor: true,
                ..Default::default()
            },
        )
        .expect("anchor + none agrees with the structural default"),
        BSplineIdentifiability::None
    ));

    // Refusal 2 and 3: `linear` needs open-knot B-spline geometry.
    options.insert("identifiability".to_string(), "linear".to_string());
    let err = resolve_bspline_identifiability(
        &options,
        BSplineIdentifiability::default(),
        BSplineIdentifiabilityContext {
            periodic: true,
            ..Default::default()
        },
    )
    .expect_err("a linear trend is not in the span of a cyclic basis");
    assert!(err.contains("periodic"), "got: {err}");
    let err = resolve_bspline_identifiability(
        &options,
        BSplineIdentifiability::default(),
        BSplineIdentifiabilityContext {
            natural_cubic_regression: true,
            ..Default::default()
        },
    )
    .expect_err("cr carries no Greville chart");
    assert!(err.contains("cr"), "got: {err}");
}

/// #2783: the option survives the whole formula → spec path, on both the
/// open and the cyclic 1-D arm — the two places that used to decide the
/// policy without reading it.
#[test]
fn one_dimensional_identifiability_option_reaches_the_built_spec() {
    let ds = continuous_dataset(
        &["y", "x"],
        (0..120)
            .map(|i| {
                let x = i as f64 / 119.0;
                vec![x.sin(), x]
            })
            .collect(),
    );
    let col_map = ds.column_map();
    let policy = gam_runtime::resource::ResourcePolicy::default_library();

    let built = |formula: &str| -> BSplineIdentifiability {
        let parsed = parse_formula(formula).expect("parse");
        let mut notes = Vec::new();
        let terms = build_termspec(&parsed.terms, &ds, &col_map, &mut notes, &policy)
            .unwrap_or_else(|e| panic!("{formula} should build: {e}"));
        let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected BSpline1D for {formula}");
        };
        spec.identifiability.clone()
    };

    assert!(matches!(
        built("y ~ s(x, k=8)"),
        BSplineIdentifiability::WeightedSumToZero { .. }
    ));
    assert!(matches!(
        built("y ~ s(x, k=8, identifiability='none')"),
        BSplineIdentifiability::None
    ));
    assert!(matches!(
        built("y ~ s(x, k=8, identifiability='linear')"),
        BSplineIdentifiability::RemoveLinearTrend
    ));
    assert!(matches!(
        built("y ~ cyclic(x, k=8, period=1)"),
        BSplineIdentifiability::WeightedSumToZero { .. }
    ));
    assert!(matches!(
        built("y ~ cyclic(x, k=8, period=1, identifiability='none')"),
        BSplineIdentifiability::None
    ));

    for formula in [
        "y ~ s(x, k=8, identifiability='totally_bogus')",
        "y ~ cyclic(x, k=8, period=1, identifiability='totally_bogus')",
        "y ~ cyclic(x, k=8, period=1, identifiability='linear')",
        "y ~ s(x, k=8, bc_left=anchored, anchor_left=0, identifiability='sum_tozero')",
    ] {
        let parsed = parse_formula(formula).expect("parse");
        let mut notes = Vec::new();
        build_termspec(&parsed.terms, &ds, &col_map, &mut notes, &policy)
            .expect_err(&format!("{formula} must be refused, not silently accepted"));
    }
}

/// #2781: a declared period makes its axis periodic on both resolvers, and
/// a declaration that names no axis is refused instead of dropped.
#[test]
fn a_declared_period_makes_its_axis_periodic() {
    let opts = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };

    // 1-D: the scalar period, and the half-open endpoint form, each declare
    // periodicity on their own.
    assert_eq!(
        parse_periodic_axes(&opts(&[("period", "24")]), 1).expect("period=24"),
        vec![true]
    );
    assert_eq!(
        parse_periodic_axes(&opts(&[("periods", "24")]), 1).expect("periods=24"),
        vec![true]
    );
    assert_eq!(
        parse_periodic_axes(&opts(&[("period_start", "0"), ("period_end", "24")]), 1)
            .expect("endpoint form"),
        vec![true]
    );
    // ...and no declaration still means aperiodic.
    assert_eq!(
        parse_periodic_axes(&opts(&[("k", "8")]), 1).expect("no declaration"),
        vec![false]
    );

    // Tensor: a per-margin list names exactly the margins that wrap.
    assert_eq!(
        parse_tensor_periodic_axes(&opts(&[("periods", "[2*pi, None]")]), 2)
            .expect("per-margin periods"),
        vec![true, false]
    );
    assert_eq!(
        parse_tensor_periodic_axes(&opts(&[("period", "[None, 24]")]), 2)
            .expect("per-margin period"),
        vec![false, true]
    );
    // A bare scalar on a multi-margin tensor names no margin, so it does not
    // flip one on; the arm-level guard below is what refuses it.
    assert_eq!(
        parse_tensor_periodic_axes(&opts(&[("period", "24")]), 2).expect("scalar on 2-D"),
        vec![false, false]
    );
    // A scalar boundary token broadcasts to every margin.
    assert_eq!(
        parse_tensor_periodic_axes(&opts(&[("bc", "periodic")]), 2).expect("scalar bc"),
        vec![true, true]
    );

    // `periodic=false` contradicts a period declaration rather than
    // outranking it silently.
    let err = parse_periodic_axes(&opts(&[("periodic", "false"), ("period", "24")]), 1)
        .expect_err("periodic=false + period= is a contradiction");
    assert!(err.contains("denies the periodicity"), "got: {err}");

    // Declarations that name no axis are refused, each by name.
    let err = reject_unconsumable_period_declaration(
        "tensor",
        &opts(&[("period", "24")]),
        &[false, false],
    )
    .expect_err("a scalar period on a 2-margin tensor names no margin");
    assert!(err.contains("does not say which"), "got: {err}");
    let err =
        reject_unconsumable_period_declaration("bspline", &opts(&[("origin", "0")]), &[false])
            .expect_err("an origin with no period is unconsumable");
    assert!(err.contains("declares no period"), "got: {err}");
    // ...and a genuine periodic axis consumes them.
    reject_unconsumable_period_declaration(
        "bspline",
        &opts(&[("period", "24"), ("origin", "0")]),
        &[true],
    )
    .expect("a periodic axis consumes its own declaration");
}

/// #2782: per-margin `degree=`/`penalty_order=` parse in both the scalar and
/// the list form, and an explicit `knot_placement` is distinguishable from
/// an unset one.
#[test]
fn tensor_per_axis_integer_options_parse_scalar_and_list_forms() {
    let mut options = BTreeMap::new();
    assert_eq!(
        parse_tensor_per_axis_usize(&options, "degree", 2).expect("absent"),
        vec![None, None]
    );

    options.insert("degree".to_string(), "2".to_string());
    assert_eq!(
        parse_tensor_per_axis_usize(&options, "degree", 3).expect("scalar broadcasts"),
        vec![Some(2), Some(2), Some(2)]
    );

    for spelling in ["[1, 3]", "c(1, 3)", "(1,3)"] {
        options.insert("degree".to_string(), spelling.to_string());
        assert_eq!(
            parse_tensor_per_axis_usize(&options, "degree", 2)
                .unwrap_or_else(|e| panic!("{spelling}: {e}")),
            vec![Some(1), Some(3)],
            "spelling {spelling} should parse per margin"
        );
    }

    options.insert("degree".to_string(), "[1, none]".to_string());
    assert_eq!(
        parse_tensor_per_axis_usize(&options, "degree", 2).expect("none keeps the default"),
        vec![Some(1), None]
    );

    options.insert("degree".to_string(), "[1, 2, 3]".to_string());
    let err = parse_tensor_per_axis_usize(&options, "degree", 2)
        .expect_err("a length mismatch must be refused");
    assert!(
        err.contains("3 entries") && err.contains("2 margins"),
        "got: {err}"
    );

    options.insert("degree".to_string(), "[1, banana]".to_string());
    let err = parse_tensor_per_axis_usize(&options, "degree", 2)
        .expect_err("a non-integer entry must be refused");
    assert!(err.contains("banana"), "got: {err}");

    let mut placement = BTreeMap::new();
    assert!(
        explicit_knot_placement(&placement)
            .expect("absent")
            .is_none()
    );
    placement.insert("knot_placement".to_string(), "uniform".to_string());
    assert_eq!(
        explicit_knot_placement(&placement).expect("explicit uniform"),
        Some(crate::basis::BSplineKnotPlacement::Uniform)
    );
}

/// #2782, end to end: the cr margin survives exactly when the caller asked
/// for what a cr margin IS, and moves to the B-spline branch otherwise.
#[test]
fn tensor_margin_leaves_cr_only_when_the_request_needs_a_bspline() {
    let ds = continuous_dataset(
        &["y", "x", "z"],
        (0..200)
            .map(|i| {
                let x = (i % 20) as f64 / 19.0;
                let z = (i / 20) as f64 / 9.0;
                vec![x + z, x, z]
            })
            .collect(),
    );
    let col_map = ds.column_map();
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let margins = |formula: &str| -> Vec<BSplineKnotSpec> {
        let parsed = parse_formula(formula).expect("parse");
        let mut notes = Vec::new();
        let terms = build_termspec(&parsed.terms, &ds, &col_map, &mut notes, &policy)
            .unwrap_or_else(|e| panic!("{formula} should build: {e}"));
        let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected a tensor spec for {formula}");
        };
        spec.marginalspecs
            .iter()
            .map(|m| m.knotspec.clone())
            .collect()
    };
    let is_cr = |k: &BSplineKnotSpec| matches!(k, BSplineKnotSpec::NaturalCubicRegression { .. });

    // Default and default-valued requests keep the cr margin, so naming an
    // option never changes a fit by itself.
    for formula in [
        "y ~ te(x, z, k=5)",
        "y ~ te(x, z, k=5, degree=3)",
        "y ~ te(x, z, k=5, penalty_order=2)",
        "y ~ te(x, z, k=5, degree=3, penalty_order=2)",
    ] {
        assert!(
            margins(formula).iter().all(is_cr),
            "{formula} must keep both cr margins"
        );
    }

    // A request the cr basis cannot carry moves that margin off it.
    for formula in [
        "y ~ te(x, z, k=5, degree=1)",
        "y ~ te(x, z, k=5, degree=4)",
        "y ~ te(x, z, k=5, penalty_order=1)",
        "y ~ te(x, z, k=5, penalty_order=3)",
        "y ~ te(x, z, k=5, knot_placement='uniform')",
        "y ~ te(x, z, k=5, knot_placement='quantile')",
    ] {
        assert!(
            margins(formula).iter().all(|k| !is_cr(k)),
            "{formula} must move both margins off the cr basis"
        );
    }

    // A per-margin list moves only the margin it names.
    let per_margin = margins("y ~ te(x, z, k=5, degree=[1, 3])");
    assert!(
        !is_cr(&per_margin[0]),
        "the degree=1 margin must be a B-spline"
    );
    assert!(is_cr(&per_margin[1]), "the degree=3 margin must stay cr");

    // #2781 on the same arm: a declared period makes the margin cyclic.
    let periodic = margins("y ~ te(x, z, k=5, periods=[1, None])");
    assert!(matches!(
        periodic[0],
        BSplineKnotSpec::PeriodicUniform { .. }
    ));
    assert!(is_cr(&periodic[1]));
}

/// #2781/#2782/#2783 guard: no whitelisted smooth option may be accepted
/// and inert.
///
/// [`validate_known_options`] answers "is this key spelled right?". All
/// three of those bugs lived in the gap between that question and the
/// different one, "does this key do anything?": the option was listed in an
/// arm's whitelist — which is precisely what stopped the unknown-option
/// refusal from firing — and then never read by that arm. The fit came back
/// bit-identical and nothing was reported.
///
/// This closes the gap mechanically instead of one option at a time. For
/// every smooth kind, each of that kind's whitelisted options is set to a
/// probe value and the built [`SmoothBasisSpec`] must CHANGE, or the formula
/// must be REFUSED. Silence is the one outcome that is not allowed.
///
/// A key that genuinely cannot change the spec belongs in
/// `structurally_inert` below WITH ITS REASON, so every exemption is a
/// reviewed statement rather than an oversight. Adding an option to a
/// whitelist without wiring it up now fails here.
#[test]
fn no_whitelisted_smooth_option_is_accepted_and_inert() {
    // Options that are real, but are consumed OUTSIDE the per-kind arm this
    // test drives, so probing them here would prove nothing about the arm.
    let structurally_inert = |kind: &str, key: &str| -> Option<&'static str> {
        match (kind, key) {
            // `type`/`bs` select which arm runs at all; changing them builds
            // a different smooth kind, which is what every other arm's row
            // in this table already covers.
            (_, "type" | "bs") => Some("selects the arm; covered by the other rows"),
            // `by=` is consumed by the `BySmooth` wrapper before the arm
            // dispatch (and `__by_col` is the engine-injected column index
            // that wrapper writes), so it never reaches the arm's options.
            (_, "by" | "__by_col") => Some("consumed by the BySmooth wrapper, not the arm"),
            // `ordered=` qualifies a FACTOR `by=` variable, so it is read by
            // the same wrapper.
            (_, "ordered") => Some("qualifies a factor by=, read by the BySmooth wrapper"),
            // `id=` is the smoothing-parameter-sharing tag: it groups terms
            // in the solver's rho vector and deliberately leaves each term's
            // basis untouched.
            (_, "id") => Some("shares a smoothing parameter; does not touch the basis"),
            // A centered cyclic basis has no free null space for the
            // double-penalty ridge to shrink: the cyclic wiggliness
            // penalty's only null direction is the constant, the periodic
            // sum-to-zero chart removes exactly that, and the ridge is
            // dropped as an identically zero block (#874). So there is no
            // second penalty for the flag to switch off. It becomes live
            // again under `identifiability='none'`, which is a different
            // baseline and is covered by the cyclic ridge tests.
            ("cyclic", "double_penalty") => {
                Some("no null space survives the periodic sum-to-zero chart (#874)")
            }
            // The Matérn cold build ships the ridge candidate unconditionally
            // and lets the bootstrap-κ spectral test decide at FIT time
            // whether it survives, pinning the outcome into the frozen
            // transform (gam#787/#860). The design this guard fingerprints is
            // the cold one, so the flag is invisible here by construction.
            ("matern", "double_penalty") => {
                Some("resolved by the fit-time bootstrap-κ spectral test, not the cold build")
            }
            // The constant-curvature RKHS Gram is full-rank positive
            // definite (#1464), so it has no null space for a shrinkage
            // ridge to act on: the candidate is built, comes out identically
            // zero, and `filter_penalty_candidates` drops it as
            // `ZeroMatrix`. The flag is therefore inert in BOTH directions
            // here — which is exactly why the arm defaults it off.
            ("curvature", "double_penalty") => {
                Some("the curvature Gram is full-rank PD, so the ridge is identically zero")
            }
            // Documented as a derivative-PLANNING hint for this family
            // (docs/formulas.md: "Thin-plate: inputs are automatically
            // standardized; scale_dims is not a learned anisotropy knob for
            // this family"). It reaches `plan_spatial_basis`, where it only
            // widens the dense-byte estimate that can trim the default center
            // count under a memory budget — so it is genuinely modelling-inert
            // by design, unlike the Matérn/Duchon anisotropy it shares a name
            // with.
            ("thinplate", "scale_dims") => {
                Some("a derivative-planning hint for TPS, not an anisotropy knob")
            }
            // Measure-jet Ψ (hyperparameter) switches: the representer
            // length-scale and the τ₀ multiscale threshold are read by the Ψ
            // learner during the fit, not by the design built at the spec's
            // own initial values.
            ("measurejet", "tau" | "learn_length_scale") => {
                Some("a Psi-learning switch read during the fit, not at design build")
            }
            _ => None,
        }
    };

    // Probe values per key, chosen away from that key's default. Count-like
    // keys carry TWO candidates: a single value can silently coincide with
    // the default on a particular fixture (the factor-smooth marginal's
    // default basis really is `k=6` on this dataset), which would report a
    // wired option as inert. An option passes when ANY candidate changes the
    // design, or when EVERY candidate is refused.
    let probe = |kind: &str, key: &str| -> &'static [&'static str] {
        match (kind, key) {
            // Periodicity: the tensor arm takes per-margin lists; the 1-D
            // arms take a scalar.
            ("tensor", "period" | "periods") => &["[1.0, None]"],
            (
                "tensor",
                "origin" | "origins" | "period_origin" | "period-origin" | "domain_origin",
            ) => &["[0.0, None]"],
            ("tensor", "periodic" | "cyclic") => &["[0]"],
            ("tensor", "boundary" | "bc") => &["['periodic', 'natural']"],
            (_, "periodic" | "cyclic") => &["true"],
            (_, "period" | "periods") => &["0.7"],
            (_, "period_start" | "start") => &["0.05"],
            (_, "period_end" | "end") => &["0.7"],
            (_, "origin" | "origins" | "period_origin" | "period-origin" | "domain_origin") => {
                &["0.1"]
            }
            (_, "boundary" | "bc" | "boundary_conditions") => &["clamped"],
            (_, "bc_left" | "left_bc" | "start_bc" | "bc_right" | "right_bc" | "end_bc") => {
                &["clamped"]
            }
            (_, "side") => &["left"],
            (
                _,
                "anchor" | "anchor_value" | "value" | "anchor_left" | "left_anchor"
                | "anchor_right" | "right_anchor",
            ) => &["0.0"],
            // Sizes and orders.
            (_, "k" | "basis_dim" | "basis-dim" | "basisdim") => &["6", "9"],
            (_, "centers") => &["6", "9"],
            (_, "knots") => &["13", "5"],
            (_, "knot_placement" | "knot-placement" | "knotplacement") => &["quantile"],
            (_, "degree") => &["2", "1"],
            (_, "penalty_order" | "m") => &["1", "3"],
            (_, "l" | "l_max" | "l-max" | "lmax" | "max_degree" | "max-degree") => &["2", "1"],
            (_, "rank") => &["5"],
            (_, "order" | "nullspace_order") => &["3", "0"],
            (_, "p" | "power") => &["1.5"],
            (_, "nu") => &["1.5"],
            (_, "kappa") => &["0.5"],
            (_, "alpha") => &["0.5"],
            (_, "tau") => &["0.5"],
            // `s` is the measure-jet's JET ORDER, admissible in (0, 2)
            // with 0.0 as the auto sentinel — not a count, and not the
            // same key as `scales` (the multiscale band count). 1.5 is
            // `MEASURE_JET_DEFAULT_ORDER_S`, so it would probe the default.
            (_, "s") => &["1.2"],
            (_, "scales") => &["3"],
            (_, "length_scale") => &["0.4"],
            (_, "chunk_size") => &["64"],
            // Flags and selectors.
            // Both polarities: the default is not the same on every arm
            // (`sz` defaults the null-space penalty OFF, `fs`/`s()` ON), and
            // a single polarity would probe the default on half of them.
            (_, "double_penalty") => &["false", "true"],
            (_, "identifiability") => &["none"],
            (_, "include_intercept") => &["true"],
            (_, "scale_dims") => &["true"],
            (_, "multiscale") => &["true"],
            (_, "learn_length_scale") => &["false"],
            (_, "centered") => &["false"],
            (_, "smooth_penalty") => &["false"],
            (_, "lazy_path") => &["true"],
            (_, "radians") => &["true"],
            (_, "units") => &["radians"],
            (_, "kernel") => &["pseudo"],
            (_, "method") => &["harmonic"],
            (_, "path" | "pca_basis_path") => &["'/nonexistent/pca.npy'"],
            other => panic!(
                "no probe value for {other:?}; add one (or an exemption with a \
                     reason) so the guard stays exhaustive"
            ),
        }
    };

    // `zbig` spans a very different range from `x` on purpose, so an
    // anisotropy option such as `scale_dims=` has something to change on the
    // radial arms; on two identically-scaled axes it is a true no-op and the
    // probe would prove nothing.
    let ds = continuous_dataset(
        &["y", "x", "z", "zbig", "lat", "lon", "g"],
        (0..240)
            .map(|i| {
                let t = i as f64;
                let x = (i % 24) as f64 / 23.0;
                let z = (i / 24) as f64 / 9.0;
                vec![
                    (t * 0.13).sin() + x + z,
                    x,
                    z,
                    500.0 * z + 3.0,
                    -80.0 + 160.0 * x,
                    -170.0 + 340.0 * z,
                    (i % 3) as f64,
                ]
            })
            .collect(),
    );
    // `g` is the categorical the factor-smooth arm needs; everything else
    // stays continuous.
    let ds = {
        let mut ds = ds;
        let g = ds
            .headers
            .iter()
            .position(|name| name == "g")
            .expect("guard dataset carries a `g` column");
        ds.schema.columns[g].kind = ColumnKindTag::Categorical;
        ds.schema.columns[g].levels = vec!["a".into(), "b".into(), "c".into()];
        ds.column_kinds[g] = ColumnKindTag::Categorical;
        ds
    };
    let col_map = ds.column_map();
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let build = |formula: &str| -> Result<String, String> {
        let parsed = parse_formula(formula)?;
        let mut notes = Vec::new();
        let spec = build_termspec(&parsed.terms, &ds, &col_map, &mut notes, &policy)
            .map_err(|err| err.to_string())?;
        // Fingerprint the BUILT DESIGN, not the spec. #2782 is exactly the
        // case a spec comparison misses: `degree=` was stored on the pushed
        // margin spec and then ignored by the cr basis builder, so the spec
        // differed while the model did not. It also has to skip
        // `SmoothTermSpec::name`, which is the term's source text and
        // therefore always differs once a probe option is appended — an
        // earlier draft compared whole specs and was vacuously green until
        // the reintroduce-the-bugs experiment caught it.
        let design = crate::smooth::build_term_collection_design(ds.values.view(), &spec)
            .map_err(|err| err.to_string())?;
        let dense = design.design.to_dense();
        let mut fingerprint = format!("design {}x{}", dense.nrows(), dense.ncols());
        for column in dense.columns() {
            let sum: f64 = column.iter().sum();
            let energy: f64 = column.iter().map(|v| v * v).sum();
            fingerprint.push_str(&format!(" |{sum:.10e},{energy:.10e}"));
        }
        for penalty in &design.smooth.penalties {
            let block = &penalty.local;
            let trace: f64 = (0..block.nrows()).map(|i| block[[i, i]]).sum();
            let energy: f64 = block.iter().map(|v| v * v).sum();
            fingerprint.push_str(&format!(
                " S[{}..{}]{}x{}:{trace:.10e},{energy:.10e}",
                penalty.col_range.start,
                penalty.col_range.end,
                block.nrows(),
                block.ncols()
            ));
        }
        Ok(fingerprint)
    };

    // (kind label, a term that reaches that arm, its whitelist). The term is
    // written so the probe below can be appended as one more option.
    let kinds: &[(&str, &str, &[&str])] = &[
        ("bspline", "s(x", BSPLINE_SMOOTH_OPTION_KEYS),
        ("cyclic", "cyclic(x", CYCLIC_SMOOTH_OPTION_KEYS),
        (
            "thinplate",
            "thinplate(x, zbig",
            THINPLATE_SMOOTH_OPTION_KEYS,
        ),
        ("matern", "matern(x, zbig", MATERN_SMOOTH_OPTION_KEYS),
        ("duchon", "duchon(x, zbig", DUCHON_SMOOTH_OPTION_KEYS),
        ("sphere", "sphere(lat, lon", SPHERE_SMOOTH_OPTION_KEYS),
        ("curvature", "curv(x, zbig", CURVATURE_SMOOTH_OPTION_KEYS),
        ("measurejet", "mjs(x, zbig", MEASURE_JET_SMOOTH_OPTION_KEYS),
        ("tensor", "te(x, z", TENSOR_SMOOTH_OPTION_KEYS),
        ("fs", "s(x, g, bs='fs'", FACTOR_SMOOTH_OPTION_KEYS),
        ("sz", "s(x, g, bs='sz'", FACTOR_SMOOTH_OPTION_KEYS),
        // `re` is swept against the SUPERSET on purpose: its own whitelist
        // (`RANDOM_EFFECT_SMOOTH_OPTION_KEYS`) is three structurally-inert
        // selectors, so sweeping it would probe nothing. Driving the
        // penalized flavours' list through the `re` arm is what pins the
        // #2791 property — every basis-shaping key is REFUSED there, which
        // this guard counts as a pass, rather than silently dropped.
        ("re", "s(x, g, bs='re'", FACTOR_SMOOTH_OPTION_KEYS),
    ];
    // `pca(...)` is the one arm not swept here: half its options name an
    // on-disk basis file (`path`, `pca_basis_path`, `lazy_path`), so probing
    // them means writing fixtures rather than building a term, which belongs
    // in that path's own integration tests.

    // Options that are accepted and inert TODAY, each with what is actually
    // wrong. They are expected failures, so the guard stays green while
    // still refusing any NEW one: this list may only shrink. Every entry is
    // a real defect of the same shape as #2781/#2782/#2783 — an option the
    // DSL validates and then throws away — found by this guard the first
    // time it ran with teeth. The reason strings ARE the bug reports; run
    // this test with an entry deleted to reproduce any one of them.
    // The ratchet is EMPTY: every option this guard found accepted-and-inert
    // when it first ran with teeth has since been wired up, refused, or
    // exempted in `structurally_inert` with a reason. Keep it that way —
    // an entry added here is a defect being deferred, and needs a reason
    // saying what is actually wrong.
    let known_inert: &[(&str, &str)] = &[];

    let mut inert = Vec::<String>::new();
    let mut honoured = 0usize;
    let mut refused = 0usize;
    for (kind, term, keys) in kinds {
        let baseline = match build(&format!("y ~ {term})")) {
            Ok(spec) => spec,
            Err(err) => panic!("baseline `y ~ {term})` must build, got: {err}"),
        };
        for key in *keys {
            if structurally_inert(kind, key).is_some() {
                continue;
            }
            let mut changed_any = false;
            let mut accepted_any = false;
            let mut last_formula = String::new();
            for value in probe(kind, key) {
                let formula = format!("y ~ {term}, {key}={value})");
                // Refused is a fine outcome — the option is not silently
                // dropped, which is the whole property under test — so a
                // rejected candidate simply does not vote.
                let Ok(spec) = build(&formula) else {
                    continue;
                };
                accepted_any = true;
                changed_any |= spec != baseline;
                last_formula = formula;
            }
            if changed_any {
                honoured += 1;
            } else if !accepted_any {
                refused += 1;
            } else {
                inert.push(last_formula);
            }
        }
    }

    // A guard whose probes all bounce off a parse error would pass while
    // proving nothing, so pin the shape of the sweep itself: most probes
    // must reach the builder and CHANGE the spec.
    let probed = honoured + refused + inert.len();
    assert!(
        probed >= 150,
        "the sweep should cover the whole option surface, only reached {probed} probes"
    );
    assert!(
        honoured * 2 > probed,
        "most probes should be HONOURED rather than refused, otherwise this \
             guard is testing error paths instead of option wiring \
             (honoured={honoured}, refused={refused}, inert={})",
        inert.len()
    );

    // The ratchet turns both ways: a NEW inert option fails here, and a
    // known one that has since been wired up must be deleted from the list
    // rather than left to rot into a lie about the engine.
    let fixed: Vec<&str> = known_inert
        .iter()
        .map(|(formula, _)| *formula)
        .filter(|formula| !inert.iter().any(|found| found == formula))
        .collect();
    assert!(
        fixed.is_empty(),
        "these options are listed in `known_inert` but are no longer inert — \
             delete their entries so the list keeps telling the truth:\n  {}",
        fixed.join("\n  ")
    );
    inert.retain(|formula| !known_inert.iter().any(|(known, _)| known == formula));

    assert!(
        inert.is_empty(),
        "these formula options were accepted and produced a bit-identical \
             smooth design — each is either unwired (wire it), unsatisfiable in \
             this configuration (refuse it), or genuinely inert (exempt it in \
             `structurally_inert` with a reason). If it is a defect you are not \
             fixing right now, add it to `known_inert` WITH ITS REASON so the \
             ratchet still holds:\n  {}",
        inert.join("\n  ")
    );
}

#[test]
fn sz_factor_smooth_low_cardinality_uses_bspline_marginal() {
    // #1605: the `sz` factor-smooth marginal is the SAME penalized B-spline
    // the `fs` sibling uses — NOT a natural cubic regression (`cr`) marginal,
    // whose hard natural boundary conditions f''=0 bias curved deviations
    // (a consistency failure). #1542 (the reason this test exists) is
    // subsumed: with a B-spline marginal a low-cardinality covariate no
    // longer needs a special cr data-support cap and can never hard-fail the
    // way the old cr-marginal `sz` spelling did — the build just succeeds,
    // exactly as `fs` already does on the identical data.
    let ds = ternary_factor_dataset();
    let col_map = ds.column_map();
    let parsed = parse_formula("y ~ s(x, g, bs=sz, k=10)").expect("parse sz factor smooth");
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("sz on a ternary covariate must build (B-spline marginal), not hard-fail");
    let SmoothBasisSpec::FactorSmooth { spec } = &terms.smooth_terms[0].basis else {
        panic!("expected FactorSmooth for s(x, g, bs=sz)");
    };
    assert!(
        !matches!(
            spec.marginal.knotspec,
            BSplineKnotSpec::NaturalCubicRegression { .. }
        ),
        "sz marginal must be a B-spline (curvature-capable), not the \
             natural-BC cr basis; got {:?}",
        spec.marginal.knotspec
    );
}

/// A dataset with a genuinely continuous covariate `x` (many distinct
/// values) and a `L`-level grouping factor `g`, suitable for building a
/// real factor-smooth marginal with a non-trivial {const, linear} null
/// space. `y` is unused by the structural penalty checks below.
fn continuous_x_factor_dataset(n: usize, n_groups: usize) -> Dataset {
    let rows = (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let g = (i % n_groups) as f64;
            vec![x + g, x, g]
        })
        .collect::<Vec<_>>();
    let levels: Vec<String> = (0..n_groups).map(|k| format!("g{k}")).collect();
    Dataset {
        headers: vec!["y".into(), "x".into(), "g".into()],
        values: Array2::from_shape_vec(
            (rows.len(), 3),
            rows.into_iter().flat_map(|row| row.into_iter()).collect(),
        )
        .expect("rectangular continuous-x factor data"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "g".into(),
                    kind: ColumnKindTag::Categorical,
                    levels,
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
        ],
    }
}

fn factor_smooth_spec_for(formula: &str, ds: &Dataset) -> FactorSmoothSpec {
    let col_map = ds.column_map();
    let parsed = parse_formula(formula).expect("parse factor smooth formula");
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build factor smooth term");
    let SmoothBasisSpec::FactorSmooth { spec } = &terms.smooth_terms[0].basis else {
        panic!("expected FactorSmooth basis for `{formula}`");
    };
    spec.clone()
}

/// #1605: the sum-to-zero factor smooth `s(x, g, bs="sz")` under-fit data
/// drawn from its own model class because its deviation blocks carried ONLY
/// the marginal wiggliness penalty — the {const, linear} null space of every
/// deviation curve was left completely unpenalized, so the single combined
/// wiggliness λ could not separate per-group intercept/slope variance from
/// curvature variance and REML parked it over-smoothed (same defect class as
/// the closed #700, more severe). mgcv's `bs="fs"` sibling avoids the gap by
/// adding a SEPARATE per-null-dimension ridge (one λ each), the
/// double-penalty `I_L ⊗ S_j` structure. The fix gives `sz` the same
/// null-space-ridge structure, mapped into the zero-sum CONTRAST space so the
/// constraint (and `sz`'s distinctness from `fs`) is preserved.
///
/// This pins the structural defect: after the fix the `sz` deviation build
/// must carry MORE than just its wiggliness penalty(s) — exactly one extra
/// null-space-ridge penalty per marginal null direction, matching the count
/// that `fs` carries — while keeping the narrower `(L-1)·p` zero-sum design
/// (NOT the `L·p` full-rank `fs` design). Before the fix `sz` carried only
/// the wiggliness penalties and this fails.
#[test]
fn sz_factor_smooth_carries_null_space_ridge_like_fs() {
    let ds = continuous_x_factor_dataset(180, 4);
    let mut workspace = crate::basis::BasisWorkspace::new();

    let sz_spec = factor_smooth_spec_for("y ~ s(x, g, bs=sz, k=8)", &ds);
    let sz_built =
        crate::smooth::build_factor_smooth(ds.values.view(), &sz_spec, "sz_term", &mut workspace)
            .expect("build sz factor smooth");

    let fs_spec = factor_smooth_spec_for("y ~ s(x, g, bs=fs, k=8)", &ds);
    let fs_built =
        crate::smooth::build_factor_smooth(ds.values.view(), &fs_spec, "fs_term", &mut workspace)
            .expect("build fs factor smooth");

    // Penalty structure (#1074 + #1605). `fs` is the exchangeable
    // random-effect smooth: all `L` level blocks share ONE wiggliness λ per
    // marginal penalty, plus one rank-1 null-space ridge per marginal null
    // direction (the #1605 double penalty). `sz` is the sum-to-zero factor
    // smooth and mgcv's `smooth.construct.sz` emits ONE penalty matrix PER
    // LEVEL — `L` independent curvature smoothing parameters — so REML can
    // shrink a low-amplitude group's deviation hard while leaving a busy
    // group nearly unpenalized. We mirror that: the single marginal
    // wiggliness penalty is split into its `L` independent zero-sum-contrast
    // summands (`L-1` free per-group blocks `(e_k e_kᵀ)⊗S` + the reference
    // coupling block `(11ᵀ)⊗S`), each carrying its own λ, and the null-space
    // ridges stay POOLED (the per-group intercept/slope shrinkage mgcv pools
    // under one variance even for `sz`).
    //
    // So with `nw` marginal wiggliness penalties and `nn` marginal null
    // directions: fs has `nw + nn` penalties; sz has `L·nw + nn`. sz must
    // therefore carry strictly MORE penalties than fs (the per-group split),
    // and the surplus must be exactly `(L-1)·nw`.
    let n_levels = sz_spec
        .group_frozen_levels
        .as_ref()
        .map(|l| l.len())
        .unwrap_or(4);
    assert!(n_levels >= 3, "test needs >=3 groups, got {n_levels}");

    // fs = nw + nn  ⇒  nn = fs_penalties - nw. The marginal has nw==1
    // wiggliness penalty (a single difference/curvature operator), so the
    // per-group split adds exactly (L-1)·nw = (L-1) extra penalties on top of
    // fs's count.
    let nw = 1usize; // one marginal wiggliness penalty for the B-spline marginal
    let expected_sz = fs_built.active_penalties.len() + (n_levels - 1) * nw;
    assert_eq!(
        sz_built.active_penalties.len(),
        expected_sz,
        "sz must split its wiggliness penalty per level (#1074): expected \
             fs_count {} + (L-1)·nw {} = {}, but sz had {}",
        fs_built.active_penalties.len(),
        (n_levels - 1) * nw,
        expected_sz,
        sz_built.active_penalties.len(),
    );
    assert!(
        sz_built.active_penalties.len() > fs_built.active_penalties.len(),
        "sz must carry strictly more penalties than fs after the per-group \
             split (sz={}, fs={})",
        sz_built.active_penalties.len(),
        fs_built.active_penalties.len(),
    );

    // The null-space ridges must still be present (the #1605 property that
    // keeps the deviation curvature un-over-smoothed). After removing the `L`
    // per-group wiggliness blocks, the remainder are the pooled null ridges,
    // and there must be at least one (a B-spline marginal has a non-empty
    // {const, linear} null space).
    let n_wiggliness = n_levels * nw; // L per-group blocks
    assert!(
        sz_built.active_penalties.len() > n_wiggliness,
        "sz deviation block carries no null-space ridge (penalties={}, \
             wiggliness blocks={}); the null space is unpenalized and REML \
             over-smooths the deviations",
        sz_built.active_penalties.len(),
        n_wiggliness,
    );

    // The zero-sum constraint must be preserved: the sz design must stay the
    // NARROWER `(L-1)·p` contrast design, strictly narrower than the fs
    // full-rank `L·p` design. This guards against "fixing" sz by making it
    // identical to fs (which would break identifiability / sum-to-zero).
    assert!(
        sz_built.dim < fs_built.dim,
        "sz design width {} must be strictly less than fs width {} \
             (zero-sum contrast drops one level block)",
        sz_built.dim,
        fs_built.dim,
    );

    for penalty in &sz_built.active_penalties {
        assert_eq!(
            penalty
                .null_eigenvectors
                .as_ref()
                .map_or(0, |basis| basis.ncols()),
            penalty.nullity
        );
    }
}

#[test]
fn sz_penalty_metadata_is_emitted_in_matrix_order_2289() {
    let ds = continuous_x_factor_dataset(180, 4);
    let mut workspace = crate::basis::BasisWorkspace::new();
    let spec = factor_smooth_spec_for("y ~ s(x, g, bs=sz, k=8, double_penalty=true)", &ds);
    let built = crate::smooth::build_factor_smooth(
        ds.values.view(),
        &spec,
        "sz_metadata_order",
        &mut workspace,
    )
    .expect("build multi-penalty sz smooth");
    let n_levels = spec.group_frozen_levels.as_ref().map(Vec::len).unwrap_or(4);

    assert!(built.active_penalties.len() >= 2 * n_levels);
    for (idx, penalty) in built.active_penalties.iter().enumerate() {
        let analysis = crate::basis::analyze_penalty_block(&penalty.matrix).expect("PSD penalty");
        assert_eq!(penalty.info.original_index, idx);
        assert_eq!(penalty.info.effective_rank, analysis.rank, "penalty {idx}");
        assert_eq!(penalty.nullity, analysis.nullity, "penalty {idx}");
    }
    assert!(
        built.active_penalties[..n_levels]
            .iter()
            .all(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
    );
    assert!(
        built.active_penalties[n_levels..2 * n_levels]
            .iter()
            .all(|penalty| matches!(penalty.info.source, PenaltySource::DoublePenaltyNullspace))
    );
}

/// #1457: `y ~ s(x, by=g) + g` with a BARE categorical `g` must NOT lower to
/// two `g` design blocks. The bare `+ g` is auto-promoted to a single
/// penalized random-effect block owning the factor's full level offsets; the
/// `by=` branch must then recognize that owner and skip adding its own
/// unpenalized treatment-coded main effect. Before the fix the dedup guard
/// recognized only explicit `group(g)` (a `ParsedTerm::RandomEffect`), so the
/// auto-promoted bare-`+ g` block slipped past and a spurious second `g`
/// block (plus an extra smoothing parameter) was added. Assert exactly ONE
/// `g` random/categorical block, and that adding the bare `+ g` introduces no
/// extra `g` blocks beyond `y ~ s(x, by=g)` alone.
fn factor_dataset_l3() -> Dataset {
    // `g` is categorical with THREE levels (encoded 0.0/1.0/2.0).
    let rows = (0..30)
        .map(|i| {
            let x = i as f64 / 29.0;
            let g = (i % 3) as f64;
            vec![x + g, x, g]
        })
        .collect::<Vec<_>>();
    Dataset {
        headers: vec!["y".into(), "x".into(), "g".into()],
        values: Array2::from_shape_vec(
            (rows.len(), 3),
            rows.into_iter().flat_map(|row| row.into_iter()).collect(),
        )
        .expect("rectangular L=3 factor test data"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "g".into(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".into(), "b".into(), "c".into()],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
        ],
    }
}

/// #2791: `bs='re'` is the PARAMETRIC random intercept + slope. It builds
/// no spline, so the basis-shaping options must be refused with a reason
/// rather than accepted and dropped — the whole `re` column of the arm's
/// shared whitelist used to be silently inert.
///
/// This approaches the property from the opposite side of the
/// `no_whitelisted_smooth_option_is_accepted_and_inert` sweep: that guard
/// asks "did the design move?", this one asks "did the user get told?", and
/// pins the paired positive — the same key on the same data, on `bs='fs'`,
/// still builds and still moves the design.
#[test]
fn random_effect_flavour_refuses_the_basis_options_it_cannot_honour_2791() {
    let ds = factor_dataset_l3();
    let col_map = ds.column_map();
    let policy = ResourcePolicy::default_library();
    let build = |formula: &str| -> Result<(), String> {
        let parsed = parse_formula(formula)?;
        let mut notes = Vec::new();
        build_termspec(&parsed.terms, &ds, &col_map, &mut notes, &policy)
            .map(|_| ())
            .map_err(|err| err.to_string())
    };

    build("y ~ s(x, g, bs='re')").expect("the bare random slope must still build");

    for (key, value) in [
        ("k", "9"),
        ("basis_dim", "9"),
        ("basisdim", "9"),
        ("knots", "5"),
        ("knot_placement", "quantile"),
        ("knotplacement", "quantile"),
        ("degree", "2"),
        ("penalty_order", "1"),
        ("m", "1"),
        ("double_penalty", "false"),
    ] {
        let err = build(&format!("y ~ s(x, g, bs='re', {key}={value})"))
            .expect_err("a basis-shaping option on bs='re' must be refused, not silently dropped");
        assert!(
            err.contains(key) && err.contains("bs='fs'"),
            "the refusal must name the offending key and the flavour that DOES \
                 honour it, got: {err}"
        );
    }

    // The hyphenated aliases `basis-dim`/`knot-placement` are in the
    // whitelist but are not reachable through the formula grammar (a bare
    // `-` inside an option name does not lex), so they are not probed here.
    //
    // The paired positive: `bs='fs'` is a real penalized smooth, so the same
    // keys build there. `knot_placement` is spelled three ways and `k` four;
    // one representative of each family is enough to prove the refusal above
    // is about the flavour, not the spelling.
    for formula in [
        "y ~ s(x, g, bs='fs', k=9)",
        "y ~ s(x, g, bs='fs', knots=5)",
        "y ~ s(x, g, bs='fs', knot_placement=quantile)",
        "y ~ s(x, g, bs='fs', degree=2)",
        "y ~ s(x, g, bs='fs', penalty_order=1)",
        "y ~ s(x, g, bs='fs', m=1)",
        "y ~ s(x, g, bs='fs', double_penalty=false)",
    ] {
        build(formula).unwrap_or_else(|err| panic!("`{formula}` must build, got: {err}"));
    }

    // A misspelling still gets the ordinary spelling error, so the new
    // refusal has not swallowed the typo path.
    let typo =
        build("y ~ s(x, g, bs='re', kk=9)").expect_err("an unknown option must still be refused");
    assert!(
        typo.contains("does not accept option `kk`"),
        "an unknown key must keep the spelling-check error, got: {typo}"
    );
}

#[test]
fn factor_by_smooth_plus_bare_categorical_does_not_duplicate_factor_block() {
    let ds = factor_dataset_l3();
    let col_map = ds.column_map();

    let g_blocks = |formula: &str| -> usize {
        let parsed = parse_formula(formula).expect("parse by-smooth formula");
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &ResourcePolicy::default_library(),
        )
        .unwrap_or_else(|err| panic!("`{formula}` must build, got: {err:?}"));
        terms
            .random_effect_terms
            .iter()
            .filter(|rt| rt.name == "g")
            .count()
    };

    // Baseline: the standalone factor-by smooth carries exactly ONE `g`
    // block (the unpenalized treatment-coded factor main effect added by the
    // `by=` branch).
    let by_only = g_blocks("y ~ s(x, by=g, k=10)");
    assert_eq!(
        by_only, 1,
        "`y ~ s(x, by=g)` must produce exactly one `g` design block"
    );

    // The bug: adding a bare `+ g` (auto-promoted to a penalized random
    // block owning the same level offsets) must NOT introduce a second `g`
    // block. Before the fix this was 2.
    let by_plus_bare = g_blocks("y ~ s(x, by=g, k=10) + g");
    assert_eq!(
        by_plus_bare, 1,
        "`y ~ s(x, by=g) + g` must collapse to ONE `g` block (#1457): the bare \
             `+ g` already owns the factor's level offsets, so the `by=` branch \
             must not add a second, treatment-coded main effect"
    );

    // The bare `+ g` adds no spurious extra `g` block versus the baseline.
    assert_eq!(
        by_plus_bare, by_only,
        "the bare `+ g` collision must add zero extra `g` blocks (#1457)"
    );
}

#[test]
fn factor_by_penalties_carry_full_expanded_null_geometry_2293() {
    let ds = factor_dataset_l3();
    let col_map = ds.column_map();
    // Leave the marginal null space unshrunk so every level-specific term
    // must carry a non-trivial joint-null chart. The production default is
    // double-penalized, whose primary and null-space ridge have a full-rank
    // joint sum and therefore correctly produce no joint-null rotation.
    let parsed =
        parse_formula("y ~ s(x, by=g, k=8, double_penalty=false)").expect("parse by smooth");
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("build by smooth spec");
    assert_eq!(terms.smooth_terms.len(), 3, "one smooth per factor level");

    // Formula construction represents an unordered factor-by smooth as one
    // explicit level-gated term per factor level. Validate the complete
    // realized expansion, rather than inspecting only its first level or
    // assuming the legacy monolithic BySmooth::Factor representation.
    for term in &terms.smooth_terms {
        assert!(matches!(
            &term.basis,
            SmoothBasisSpec::ByVariable {
                by: ByVariableSpec::Level { .. },
                ..
            }
        ));
        let mut workspace = crate::basis::BasisWorkspace::new();
        let built =
            crate::smooth::build_single_local_smooth_term(ds.values.view(), term, &mut workspace)
                .expect("build level-gated factor-by smooth");

        for (idx, penalty) in built.active_penalties.iter().enumerate() {
            let analysis = crate::basis::analyze_penalty_block(&penalty.matrix).expect("PSD block");
            assert_eq!(analysis.rank + penalty.nullity, built.dim, "penalty {idx}");
            assert_eq!(analysis.nullity, penalty.nullity, "penalty {idx}");
            assert_eq!(penalty.info.effective_rank, analysis.rank);
            let basis = penalty
                .null_eigenvectors
                .as_ref()
                .expect("nontrivial factor-level null basis");
            assert_eq!(basis.nrows(), built.dim);
            assert_eq!(basis.ncols(), penalty.nullity);
        }
        let joint = built
            .joint_null_rotation
            .as_ref()
            .expect("factor-level joint null geometry");
        assert!(joint.joint_nullity > 0);
        assert_eq!(joint.rotation.nrows(), built.dim);
        assert_eq!(joint.rotation.ncols(), built.dim);
    }
}

#[test]
fn parse_tensor_periods_and_origins_aliases() {
    let mut opts = BTreeMap::new();
    opts.insert(
        "boundary".to_string(),
        "['periodic', 'periodic']".to_string(),
    );
    opts.insert("periods".to_string(), "[7, 24]".to_string());
    opts.insert("origins".to_string(), "[0, -12]".to_string());
    let axes = parse_periodic_axes(&opts, 2).expect("axes");
    let periods = parse_periods(&opts, &axes).expect("periods");
    let origins = parse_period_origins(&opts, &axes).expect("origins");
    assert_eq!(axes, vec![true, true]);
    assert_eq!(periods, vec![Some(7.0), Some(24.0)]);
    assert_eq!(origins, vec![Some(0.0), Some(-12.0)]);
}

#[test]
fn tensor_smooth_honors_per_margin_k_list() {
    let ds = continuous_dataset(
        &["y", "theta", "h"],
        (0..20)
            .map(|i| {
                let theta = std::f64::consts::TAU * i as f64 / 20.0;
                let h = -1.0 + 2.0 * (i % 5) as f64 / 4.0;
                vec![theta.cos() + h, theta, h]
            })
            .collect(),
    );
    let parsed = parse_formula(
        "y ~ te(theta, h, periodic=[0], period=[2*pi, None], origin=[0, None], k=[9,5])",
    )
    .expect("parse tensor formula");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build tensor terms");
    let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected tensor B-spline");
    };
    let dims = spec
        .marginalspecs
        .iter()
        .map(|m| match m.knotspec {
            BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
            BSplineKnotSpec::Generate {
                num_internal_knots, ..
            } => num_internal_knots + m.degree + 1,
            // The mgcv-default `cr` margin (#1074) reports its basis size as
            // the number of value-knots placed.
            BSplineKnotSpec::NaturalCubicRegression { ref knots } => knots.len(),
            _ => panic!("unexpected tensor marginal knotspec"),
        })
        .collect::<Vec<_>>();
    assert_eq!(dims, vec![9, 5]);
}

#[test]
fn tensor_smooth_honors_per_margin_k_axis_aliases() {
    let ds = continuous_dataset(
        &["resp", "x", "y"],
        (0..12)
            .map(|i| {
                let t = i as f64 / 11.0;
                vec![t, t, 1.0 - t]
            })
            .collect(),
    );
    assert_eq!(
        tensor_margin_basis_sizes(&ds, "resp ~ te(x, y, k_x=9, k_y=5)"),
        vec![9, 5],
        "k_<margin> aliases should materialize requested per-margin values"
    );
}

#[test]
fn tensor_smooth_low_cardinality_axis_falls_back_to_lower_degree_basis() {
    // mgcv-style: `te(x, b, k=c(5, 2))` with a BINARY second margin (only
    // values {0, 1}) is a legitimate request — the binary axis can hold at
    // most a 2-function linear basis. We must NOT reject k=2 with a
    // "k too small for degree 3" config error; instead, drop the spline
    // degree on the binary axis to k_axis - 1 (here 1, linear) while
    // keeping the continuous margin at the requested degree=3, k=5.
    let ds = continuous_dataset(
        &["y", "x", "b"],
        (0..40)
            .map(|i| {
                let x = i as f64 / 39.0;
                let b = (i % 2) as f64;
                vec![x.sin() + 0.5 * b, x, b]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ te(x, b, k=[5, 2])").expect("parse tensor with k=[5,2]");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build tensor with binary margin");
    let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected tensor B-spline for te(x, b)");
    };
    // Continuous margin keeps requested degree=3 and k=5; binary margin
    // drops to degree=1 (linear) so the requested k=2 yields exactly two
    // basis functions before tensor-product identifiability is applied.
    let continuous = &spec.marginalspecs[0];
    let binary = &spec.marginalspecs[1];
    assert_eq!(continuous.degree, 3);
    assert_eq!(binary.degree, 1);
    assert!(
        binary.penalty_order >= 1 && binary.penalty_order <= binary.degree,
        "binary margin penalty_order {} must satisfy 1 <= order <= degree={}",
        binary.penalty_order,
        binary.degree
    );
    let basis_size = |m: &BSplineBasisSpec| match m.knotspec {
        BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
        BSplineKnotSpec::Generate {
            num_internal_knots, ..
        } => num_internal_knots + m.degree + 1,
        BSplineKnotSpec::Automatic {
            num_internal_knots: Some(n),
            ..
        } => n + m.degree + 1,
        // The mgcv-default `cr` margin (#1074) reports its basis size as the
        // number of value-knots placed.
        BSplineKnotSpec::NaturalCubicRegression { ref knots } => knots.len(),
        _ => panic!("unexpected tensor marginal knotspec"),
    };
    assert_eq!(basis_size(continuous), 5);
    assert_eq!(basis_size(binary), 2);
}

#[test]
fn tensor_smooth_uniform_k_is_capped_to_a_low_cardinality_margins_distinct_values() {
    // Regression: a SINGLE `k=5` applied to every axis of `te(x, b, k=5)`
    // with a BINARY second margin (`b ∈ {0, 1}`) must build a valid tensor,
    // NOT hard-fail in cr-knot selection ("cubic regression spline with k=5
    // requires at least 5 distinct values, got 2"). mgcv caps a margin's
    // basis to its data support; the binary axis becomes the 2-function
    // (linear) margin, while the continuous axis keeps the requested k=5.
    // This is the `te(age, badh, k=5)` real-data case that previously errored.
    let ds = continuous_dataset(
        &["y", "x", "b"],
        (0..40)
            .map(|i| {
                let x = i as f64 / 39.0;
                let b = (i % 2) as f64;
                vec![x.sin() + 0.5 * b, x, b]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ te(x, b, k=5)").expect("parse tensor with uniform k=5");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("uniform k=5 must auto-cap the binary margin instead of erroring");
    let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected tensor B-spline for te(x, b)");
    };
    let basis_size = |m: &BSplineBasisSpec| match &m.knotspec {
        BSplineKnotSpec::PeriodicUniform { num_basis, .. } => *num_basis,
        BSplineKnotSpec::Generate {
            num_internal_knots, ..
        } => num_internal_knots + m.degree + 1,
        BSplineKnotSpec::Automatic {
            num_internal_knots: Some(n),
            ..
        } => n + m.degree + 1,
        BSplineKnotSpec::NaturalCubicRegression { knots } => knots.len(),
        other => panic!("unexpected tensor marginal knotspec: {other:?}"),
    };
    let binary = &spec.marginalspecs[1];
    // Binary margin is reduced to the 2-function linear basis its data
    // supports (k capped from 5 to 2, degree dropped to 1).
    assert_eq!(basis_size(binary), 2);
    assert_eq!(binary.degree, 1);
    // The continuous margin is unaffected by the cap (40 distinct values).
    assert_eq!(basis_size(&spec.marginalspecs[0]), 5);
}

#[test]
fn tensor_all_tp_margins_with_per_margin_k_routes_to_bspline_tensor() {
    // `te(x1, x2, bs=c('tp','tp'), k=c(5,5))` is mgcv's per-margin tp tensor
    // with per-margin basis sizes — a tensor product of two 1-D bases, each
    // of dimension 5. The list-valued `k=c(5,5)` is honored by
    // `parse_tensor_k_list`, producing one penalized B-spline margin per axis
    // (each spanning the requested per-axis thin-plate function space). This
    // is the same anisotropic-tensor routing the scalar/no-`k` case takes —
    // a `te()` request is ALWAYS a tensor product, never a silent isotropic
    // thin-plate substitution.
    let ds = continuous_dataset(
        &["y", "x1", "x2"],
        (0..32)
            .map(|i| {
                let t = i as f64 / 31.0;
                vec![t.sin(), t, 1.0 - t]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ te(x1, x2, bs=c('tp','tp'), k=c(5,5))").expect("parse tensor");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build tensor terms with per-margin k");
    let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!(
            "expected B-spline tensor when k=c(5,5) is supplied with bs=c('tp','tp'), got {:?}",
            terms.smooth_terms[0].basis
        );
    };
    // Since #1074 a `tp` tensor margin (k >= 3) is realized as a
    // Lancaster–Salkauskas natural cubic-regression margin (cr basis
    // dimension == knot count), not an open `Generate` B-spline. It is
    // still a `TensorBSpline` spec with one penalized 1-D margin per axis,
    // so the routing assertion above still holds; only the per-margin
    // knotspec variant changed. The earlier `_ => panic!` arm pinned the
    // pre-#1074 `Generate`-only representation and is stale. Decode every
    // margin variant to its basis dimension (mirroring the
    // `tensor_margin_basis_sizes` helper).
    let dims = spec
        .marginalspecs
        .iter()
        .map(|m| match m.knotspec {
            BSplineKnotSpec::Generate {
                num_internal_knots, ..
            } => num_internal_knots + m.degree + 1,
            BSplineKnotSpec::Automatic {
                num_internal_knots: Some(num_internal_knots),
                ..
            } => num_internal_knots + m.degree + 1,
            BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
            BSplineKnotSpec::Provided(ref knots) => knots.len().saturating_sub(m.degree + 1),
            BSplineKnotSpec::NaturalCubicRegression { ref knots } => knots.len(),
            BSplineKnotSpec::Automatic {
                num_internal_knots: None,
                ..
            } => panic!("test cannot infer automatic knot count"),
        })
        .collect::<Vec<_>>();
    assert_eq!(dims, vec![5, 5]);
}

#[test]
fn tensor_all_tp_margins_without_per_margin_k_builds_anisotropic_tensor() {
    // `te(x1, x2, bs=c('tp','tp'))` is a tensor-product request and must
    // build a genuine anisotropic tensor product (one smoothing parameter
    // per margin), NOT a silently-substituted multi-D isotropic thin-plate
    // radial smooth — that would be a different model (`s(x1,x2,bs='tp')`).
    // The routing is now consistent whether or not `k` is list-valued: a tp
    // margin vector always realizes each axis as a 1-D penalized B-spline
    // margin spanning the same per-axis thin-plate function space (#1082).
    let ds = continuous_dataset(
        &["y", "x1", "x2"],
        (0..32)
            .map(|i| {
                let t = i as f64 / 31.0;
                vec![t.sin(), t, 1.0 - t]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ te(x1, x2, bs=c('tp','tp'))").expect("parse tensor");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build tensor terms without per-margin k");
    let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!(
            "te(...,bs=c('tp','tp')) must route to an anisotropic tensor product, not a \
                 silent isotropic thin-plate substitution; got {:?}",
            terms.smooth_terms[0].basis
        );
    };
    assert_eq!(
        spec.marginalspecs.len(),
        2,
        "tp tensor must carry one penalized B-spline margin per axis"
    );
}

#[test]
fn explicit_basis_sizes_are_not_small_n_clamped() {
    let ds = continuous_dataset(
        &["y", "x1", "x2", "x3", "x4", "x5"],
        (0..12)
            .map(|i| {
                let x = i as f64 / 11.0;
                vec![x.sin(), x, x * x, x + 0.1, 1.0 - x, (2.0 * x).sin()]
            })
            .collect(),
    );
    let parsed = parse_formula("y ~ s(x1, k=10) + s(x2) + s(x3) + s(x4) + s(x5)")
        .expect("parse multi-smooth formula");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build multi-smooth terms");
    let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected first smooth to be B-spline");
    };
    assert!(matches!(
        &spec.knotspec,
        BSplineKnotSpec::Generate {
            num_internal_knots: 6,
            ..
        }
    ));
}

#[test]
fn explicit_duchon_centers_are_not_small_n_bumped() {
    let ds = continuous_dataset(
        &["y", "x1", "x2", "x3", "x4", "x5"],
        (0..12)
            .map(|i| {
                let x = i as f64 / 11.0;
                vec![x.sin(), x, x * x, x + 0.1, 1.0 - x, (2.0 * x).sin()]
            })
            .collect(),
    );
    // Pure 1D Duchon at default options resolves the nullspace to Linear
    // (2s < d forces escalation), giving 2 polynomial nullspace columns;
    // the well-posedness gate requires num_centers > polynomial_cols, so
    // 3 is the smallest valid count. It is still well below the small-N
    // bump target of polynomial_cols + 4 = 6, so this exercises the
    // "explicit value is honored" path the test name advertises.
    let parsed = parse_formula("y ~ duchon(x1, centers=3) + s(x2) + s(x3) + s(x4) + s(x5)")
        .expect("parse multi-smooth formula");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("build multi-smooth terms");
    let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
        panic!("expected first smooth to be Duchon");
    };
    assert!(matches!(
        spec.center_strategy,
        CenterStrategy::UniformGrid { points_per_dim: 3 }
    ));
}

#[test]
fn inferred_tensor_basis_cap_uses_coordinate_support_not_duplicate_rows() {
    let mut unique_rows = Vec::new();
    for i in 0..50 {
        let theta = i as f64 / 50.0;
        for j in 0..16 {
            let h = -1.0 + 2.0 * (j as f64) / 15.0;
            let y = theta.cos() + h;
            unique_rows.push(vec![y, theta, h]);
        }
    }
    let mut repeated_rows = Vec::new();
    for _ in 0..12 {
        repeated_rows.extend(unique_rows.iter().cloned());
    }

    let unique = continuous_dataset(&["y", "theta", "h"], unique_rows);
    let repeated = continuous_dataset(&["y", "theta", "h"], repeated_rows);

    let unique_basis = inferred_tensor_basis_product(&unique);
    let repeated_basis = inferred_tensor_basis_product(&repeated);

    assert_eq!(
        unique_basis, repeated_basis,
        "duplicating existing tensor coordinates must not inflate inferred basis width"
    );
}

#[test]
fn inferred_three_dim_tensor_basis_stays_bounded_for_reml_selection() {
    // Regression for gam#813: the inferred per-margin k must be
    // dimension-aware so the 3-D tensor width p = ∏ k_d does not explode.
    // With the old 1-D-per-margin rule a 3-D `te` defaulted to 7³=343 at
    // small n and 20³=8000 at larger n, making the (non-Kronecker-factorable)
    // full-tensor sum-to-zero penalty's O(p³) REML reparameterization a
    // multi-minute stall. The dimension-aware budget keeps the product near
    // mgcv's te default (≈5³=125) regardless of n.
    let make = |n: usize| -> usize {
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f64 / n as f64;
            rows.push(vec![f.sin(), f, (2.0 * f).cos(), (3.0 * f) % 1.0]);
        }
        let ds = continuous_dataset(&["y", "x1", "x2", "x3"], rows);
        let parsed = parse_formula("y ~ te(x1, x2, x3)").expect("parse 3-D tensor");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &ResourcePolicy::default_library(),
        )
        .expect("build 3-D tensor termspec");
        let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected tensor smooth");
        };
        spec.marginalspecs
            .iter()
            .map(|m| match m.knotspec {
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                } => num_internal_knots + m.degree + 1,
                BSplineKnotSpec::Automatic {
                    num_internal_knots: Some(num_internal_knots),
                    ..
                } => num_internal_knots + m.degree + 1,
                // The mgcv-default `cr` margin (#1074) reports its basis size
                // as the number of value-knots placed.
                BSplineKnotSpec::NaturalCubicRegression { ref knots } => knots.len(),
                _ => panic!("unexpected tensor margin knotspec"),
            })
            .product()
    };

    // n=30 (the issue's data): was 7³=343, must now be modest.
    assert!(
        make(60) <= 216,
        "3-D te at small n must stay near the mgcv te default, got {}",
        make(60)
    );
    // Larger n must NOT grow the product toward n³ (was 20³=8000).
    assert!(
        make(2000) <= 216,
        "3-D te at large n must not blow ∏k toward the data size, got {}",
        make(2000)
    );
}

#[test]
fn parse_bspline_boundary_conditions_and_side_selector() {
    // The `side=left` filter routes the global `anchor=` value to the left
    // endpoint (not the right), preserving the non-zero value for the
    // affine boundary lift.
    let mut opts = BTreeMap::new();
    opts.insert("boundary_conditions".to_string(), "anchored".to_string());
    opts.insert("side".to_string(), "left".to_string());
    opts.insert("anchor".to_string(), "2.5".to_string());
    let parsed = parse_bspline_boundary_conditions(&opts).expect("left anchor parses");
    assert!(matches!(
        parsed.left,
        BSplineEndpointBoundaryCondition::Anchored { value } if value == 2.5
    ));
    assert!(matches!(
        parsed.right,
        BSplineEndpointBoundaryCondition::Free
    ));

    // Side-specific aliases (`start_bc`/`end_bc`) plus the side-specific
    // anchor key (`right_anchor`) must funnel the value onto the right
    // endpoint.
    let mut opts = BTreeMap::new();
    opts.insert("start_bc".to_string(), "clamped".to_string());
    opts.insert("end_bc".to_string(), "zero".to_string());
    opts.insert("right_anchor".to_string(), "-1.0".to_string());
    let parsed = parse_bspline_boundary_conditions(&opts).expect("right anchor parses");
    assert!(matches!(
        parsed.left,
        BSplineEndpointBoundaryCondition::Clamped
    ));
    assert!(matches!(
        parsed.right,
        BSplineEndpointBoundaryCondition::Anchored { value } if value == -1.0
    ));

    // With anchors at zero the basis builder accepts the configuration,
    // so the same alias plumbing yields a clean `Anchored { value: 0.0 }`
    // on the right and `Clamped` on the left.
    let mut opts = BTreeMap::new();
    opts.insert("start_bc".to_string(), "clamped".to_string());
    opts.insert("end_bc".to_string(), "zero".to_string());
    let parsed = parse_bspline_boundary_conditions(&opts).expect("boundary conditions");
    assert!(matches!(
        parsed.left,
        BSplineEndpointBoundaryCondition::Clamped
    ));
    assert!(matches!(
        parsed.right,
        BSplineEndpointBoundaryCondition::Anchored { value } if value.abs() < 1e-12
    ));
}

#[test]
fn one_sided_anchor_owns_level_without_sum_to_zero_constraint_1867() {
    let ds = continuous_dataset(
        &["y", "x"],
        (0..32)
            .map(|i| {
                let x = i as f64 / 31.0;
                vec![x * (1.0 - x), x]
            })
            .collect(),
    );
    let col_map = ds.column_map();

    let build = |formula: &str| {
        let parsed = parse_formula(formula).expect("parse anchored smooth");
        let mut notes = Vec::new();
        build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &ResourcePolicy::default_library(),
        )
        .expect("build anchored smooth")
    };

    let one_sided = build("y ~ s(x, bc_left=anchored, anchor_left=0, k=10)");
    let SmoothBasisSpec::BSpline1D { spec, .. } = &one_sided.smooth_terms[0].basis else {
        panic!("expected one-dimensional B-spline");
    };
    assert!(matches!(spec.identifiability, BSplineIdentifiability::None));

    // #2297: a two-sided anchor pins BOTH endpoint levels, which strips the
    // interior level as well — the smooth owns no free level at all, so
    // identifiability drops to `None` (drop-intercept/skip-centering), the
    // same ownership rule as the one-sided case above. The former
    // `WeightedSumToZero` expectation predates #2297 (2e90c51b7) and would
    // double-constrain the anchored level.
    let two_sided = build("y ~ s(x, bc_left=anchored, bc_right=anchored, k=10)");
    let SmoothBasisSpec::BSpline1D { spec, .. } = &two_sided.smooth_terms[0].basis else {
        panic!("expected one-dimensional B-spline");
    };
    assert!(matches!(spec.identifiability, BSplineIdentifiability::None));

    // Control: an un-anchored smooth keeps the default weighted sum-to-zero
    // constraint — #2297's anchor rule must not leak into plain smooths.
    let plain = build("y ~ s(x, k=10)");
    let SmoothBasisSpec::BSpline1D { spec, .. } = &plain.smooth_terms[0].basis else {
        panic!("expected one-dimensional B-spline");
    };
    assert!(matches!(
        spec.identifiability,
        BSplineIdentifiability::WeightedSumToZero { .. }
    ));
}

#[test]
fn categorical_by_categorical_interaction_expands_full_cross_cells() {
    // `y ~ f:g` is an INTERACTION-ONLY factor-by-factor model: neither `f`
    // nor `g` appears as a main effect, so neither marginal parent is
    // present and BOTH factors must be dummy-coded (gam#1159). The correct
    // design is the SATURATED cell-means model: the full cross of ALL levels
    // (3 * 2 = 6 cells) minus ONE reference cell (the lexicographically-first
    // level of every factor, here f0:g0) absorbed by the intercept — rank
    // 6-1 = 5 cell columns + intercept, column-space-identical to `f*g`.
    // Treatment-coding both factors (the old behaviour) kept only
    // (3-1)*(2-1) = 2 cells and collapsed the rest onto the intercept, a
    // rank-deficient fit; that is the bug this test now guards against.
    let n = 30usize;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let y = (i as f64).sin();
        let f = (i % 3) as f64; // 3 levels: 0,1,2
        let g = (i % 2) as f64; // 2 levels: 0,1
        rows.push(vec![y, f, g]);
    }
    let values = Array2::from_shape_vec(
        (n, 3),
        rows.into_iter().flat_map(|row| row.into_iter()).collect(),
    )
    .expect("rectangular cross-factor data");
    let ds = Dataset {
        headers: vec!["y".into(), "f".into(), "g".into()],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "f".into(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["f0".into(), "f1".into(), "f2".into()],
                },
                SchemaColumn {
                    name: "g".into(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["g0".into(), "g1".into()],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
            ColumnKindTag::Categorical,
        ],
    };

    let parsed = parse_formula("y ~ f:g").expect("parse `y ~ f:g`");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let terms = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("factor-by-factor `f:g` interaction must build, not error");

    assert_eq!(
        terms.linear_terms.len(),
        5,
        "saturated 3*2 = 6 cross cells minus one reference cell (f0:g0) = 5"
    );

    let f_col = *col_map.get("f").expect("f column");
    let g_col = *col_map.get("g").expect("g column");
    // The dropped reference cell pairs each factor's lexicographically-first
    // level: f0 (0.0) and g0 (0.0). It must NOT appear among the emitted
    // cells; every OTHER cross cell must.
    let f0 = 0.0_f64.to_bits();
    let g0 = 0.0_f64.to_bits();
    let mut emitted = std::collections::HashSet::new();
    for term in &terms.linear_terms {
        // No numeric operand: the realized column is a pure cell indicator.
        assert!(term.feature_cols.is_empty());
        assert_eq!(term.categorical_levels.len(), 2);
        let mut gates = std::collections::HashMap::new();
        for &(col, bits) in &term.categorical_levels {
            gates.insert(col, bits);
        }
        let f_bits = *gates.get(&f_col).expect("f gate present");
        let g_bits = *gates.get(&g_col).expect("g gate present");
        // The reference cell f0:g0 must have been dropped.
        assert!(
            !(f_bits == f0 && g_bits == g0),
            "the reference cell f0:g0 must be absorbed by the intercept, not emitted"
        );
        emitted.insert((f_bits, g_bits));

        let column = term
            .realized_design_column(ds.values.view())
            .expect("realize cross cell");
        for row in 0..n {
            let f = ds.values[[row, f_col]];
            let g = ds.values[[row, g_col]];
            let expected = if f.to_bits() == f_bits && g.to_bits() == g_bits {
                1.0
            } else {
                0.0
            };
            assert!(
                (column[row] - expected).abs() < 1e-12,
                "row {row}: expected {expected}, got {}",
                column[row]
            );
        }
        assert!(
            column.iter().any(|&v| v == 1.0),
            "each cross cell must be observed in the data"
        );
    }
    // Every non-reference cross cell is present exactly once: all 6 cells
    // except f0:g0.
    let f_levels = [0.0_f64.to_bits(), 1.0_f64.to_bits(), 2.0_f64.to_bits()];
    let g_levels = [0.0_f64.to_bits(), 1.0_f64.to_bits()];
    for &fb in &f_levels {
        for &gb in &g_levels {
            if fb == f0 && gb == g0 {
                continue;
            }
            assert!(
                emitted.contains(&(fb, gb)),
                "saturated cross cell must be present"
            );
        }
    }
}

/// #1561 by-group representation floor: a factor-by radial smooth's
/// per-level blocks each see only their level's rows, so the n-scaling
/// DEFAULT center count must size from the smallest level, not the pooled
/// row count (measured: pooled sizing gave ~50 centers per 100-row level
/// and an unconditionable mean block whose truth-recovery no λ could fix).
#[test]
fn by_level_thin_plate_sizes_default_centers_from_the_smallest_level() {
    let n_a = 60usize;
    let n_b = 180usize;
    let rows: Vec<Vec<f64>> = (0..(n_a + n_b))
        .map(|i| {
            let in_a = i < n_a;
            let x = if in_a {
                i as f64 / (n_a - 1) as f64
            } else {
                (i - n_a) as f64 / (n_b - 1) as f64
            };
            let g = if in_a { 0.0 } else { 1.0 };
            vec![x + g, x, g]
        })
        .collect();
    let ds = Dataset {
        headers: vec!["y".into(), "x".into(), "g".into()],
        values: Array2::from_shape_vec(
            (rows.len(), 3),
            rows.into_iter().flat_map(|row| row.into_iter()).collect(),
        )
        .expect("rectangular by-level test data"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "g".into(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".into(), "b".into()],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
        ],
    };
    let build_tp = |with_by: bool| -> SmoothBasisSpec {
        let mut options = BTreeMap::new();
        options.insert("bs".to_string(), "tps".to_string());
        if with_by {
            options.insert("by".to_string(), "g".to_string());
            options.insert("__by_col".to_string(), "2".to_string());
        }
        let mut notes = Vec::new();
        build_smooth_basis(
            SmoothKind::S,
            &["x".to_string()],
            &[1],
            &options,
            &ds,
            &mut notes,
            &ResourcePolicy::default_library(),
            1,
        )
        .expect("thin-plate basis builds")
    };
    let pooled = build_tp(false);
    let by_level = build_tp(true);
    let tp_centers = |basis: &SmoothBasisSpec| -> usize {
        match basis {
            SmoothBasisSpec::ThinPlate { spec, .. } => spec.center_strategy.planned_num_centers(1),
            SmoothBasisSpec::BySmooth { smooth, .. } => match smooth.as_ref() {
                SmoothBasisSpec::ThinPlate { spec, .. } => {
                    spec.center_strategy.planned_num_centers(1)
                }
                other => panic!("expected ThinPlate inside BySmooth, got {other:?}"),
            },
            other => panic!("expected ThinPlate, got {other:?}"),
        }
    };
    let pooled_centers = tp_centers(&pooled);
    let by_centers = tp_centers(&by_level);
    assert!(
        by_centers < pooled_centers,
        "by-level default centers must size from the smallest level: \
             by={by_centers} pooled={pooled_centers}"
    );
    // The by-level default must agree with a direct build on a dataset of
    // the smallest level's size (the block's true effective sample).
    let ds_small = continuous_dataset(
        &["y", "x"],
        (0..n_a)
            .map(|i| {
                let x = i as f64 / (n_a - 1) as f64;
                vec![x, x]
            })
            .collect(),
    );
    let mut small_options = BTreeMap::new();
    small_options.insert("bs".to_string(), "tps".to_string());
    let mut notes = Vec::new();
    let small = build_smooth_basis(
        SmoothKind::S,
        &["x".to_string()],
        &[1],
        &small_options,
        &ds_small,
        &mut notes,
        &ResourcePolicy::default_library(),
        1,
    )
    .expect("small-level thin-plate basis builds");
    assert_eq!(
        by_centers,
        tp_centers(&small),
        "by-level default must equal the smallest level's own default"
    );
}
