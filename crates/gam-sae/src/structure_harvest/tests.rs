// Behavior tests for structure_harvest, split out of structure_harvest.rs to
// keep that file under the #780 10k-line gate. `use super::*` resolves to the
// parent `structure_harvest` module exactly as the inlined `mod tests` did.
use super::*;
use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SAE_DEFAULT_TORUS_HARMONICS, SaeAssignment,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
};
use gam_solve::structure_search::{CollapseAction, CollapseEvent};
use gam_terms::latent::LatentManifold;
use ndarray::Array2;
use std::sync::Arc;

#[test]
fn intrinsic_primary_chart_is_cluster_local_2240() {
    let local_rows = 20usize;
    let other_rows = 20usize;
    let mut first = Array2::<f64>::zeros((local_rows + other_rows, 3));
    let mut second = Array2::<f64>::zeros((local_rows + other_rows, 3));
    for row in 0..local_rows {
        let x = (row % 5) as f64;
        let y = (row / 5) as f64;
        let z = 0.15 * x * y;
        for target in [&mut first, &mut second] {
            target[[row, 0]] = x;
            target[[row, 1]] = y;
            target[[row, 2]] = z;
        }
    }
    for offset in 0..other_rows {
        let row = local_rows + offset;
        let x = (offset % 5) as f64;
        let y = (offset / 5) as f64;
        first[[row, 0]] = 10.0 + x;
        first[[row, 1]] = y;
        second[[row, 0]] = 1.0e6 + 1.0e4 * x;
        second[[row, 1]] = -1.0e6 + 1.0e4 * y;
        second[[row, 2]] = 2.0e6;
    }
    let rows = (0..local_rows).collect::<Vec<_>>();
    let first_specs = build_intrinsic_primary_specs(first.view(), &rows, 2)
        .expect("first local embedding")
        .expect("realizable first local chart");
    let second_specs = build_intrinsic_primary_specs(second.view(), &rows, 2)
        .expect("second local embedding")
        .expect("realizable second local chart");
    let first_chart = &first_specs[0].coords;
    let second_chart = &second_specs[0].coords;

    for row in 0..local_rows {
        for col in 0..2 {
            assert_eq!(
                first_chart[[row, col]].to_bits(),
                second_chart[[row, col]].to_bits(),
                "another atom's observations must not alter cluster-local geodesics"
            );
        }
    }
    for row in local_rows..(local_rows + other_rows) {
        assert_eq!(first_chart[[row, 0]], 0.0);
        assert_eq!(first_chart[[row, 1]], 0.0);
        assert_eq!(second_chart[[row, 0]], 0.0);
        assert_eq!(second_chart[[row, 1]], 0.0);
    }
}

/// A parent nominated more than once (by different partners at different
/// significances) must collapse to EXACTLY ONE entry — the most-suspect
/// (lowest-significance) one — and distinct parents must all survive, in
/// most-suspect-first order. This is the regression for the
/// `dedup_by_key`-only-removes-adjacent-duplicates bug that used to let a
/// parent ride as several duplicate `Fission` proposals.
#[test]
fn dedup_most_suspect_keeps_one_per_parent() {
    // Atom 2 nominated three times (0.4, 0.1, 0.7); atom 5 twice (0.3, 0.9);
    // atom 1 once (0.6). Deliberately unsorted so a significance-first sort
    // would NOT place same-atom entries adjacently.
    let raw = vec![
        (2usize, 0.4_f64),
        (5, 0.9),
        (1, 0.6),
        (2, 0.1),
        (5, 0.3),
        (2, 0.7),
    ];
    let out = dedup_most_suspect_per_parent(raw);

    // Exactly one entry per distinct parent.
    assert_eq!(out.len(), 3, "one entry per distinct parent: {out:?}");
    let mut atoms: Vec<usize> = out.iter().map(|(a, _)| *a).collect();
    atoms.sort_unstable();
    assert_eq!(atoms, vec![1, 2, 5], "all distinct parents kept");

    // The kept significance per parent is the minimum (most-suspect).
    let sig = |atom: usize| out.iter().find(|(a, _)| *a == atom).unwrap().1;
    assert_eq!(sig(2), 0.1, "atom 2 keeps its most-suspect nomination");
    assert_eq!(sig(5), 0.3, "atom 5 keeps its most-suspect nomination");
    assert_eq!(sig(1), 0.6, "the singly-nominated atom is unchanged");

    // Most-suspect-first (significance ascending) — the order the downstream
    // `take(max_fissions)` and carve loop rely on.
    assert_eq!(
        out,
        vec![(2, 0.1), (5, 0.3), (1, 0.6)],
        "deterministic most-suspect-first order"
    );
}

/// A high active logit (atom routes strongly on the row) and a low one
/// (atom is dormant). With the `ACTIVE_SUPPORT_REL_FLOOR / K` threshold a
/// softmax of these separates the discrete support cleanly.
const ON: f64 = 6.0;
const OFF: f64 = -6.0;

/// #2238/#2239 — `auto` is an evidence-discovery request, not an alias for
/// the old periodic default. An undersupported cluster must fail loudly and
/// leave the caller's unresolved state untouched.
#[test]
fn auto_primary_topology_never_falls_back_on_race_failure_2238_2239() {
    let target = Array2::<f64>::zeros((15, 2));
    let labels = vec![0usize; 15];
    let mut basis = vec!["auto".to_string()];
    let mut dims = vec![2usize];

    let error = resolve_auto_primary_atoms(target.view(), &labels, &mut basis, &mut dims)
        .expect_err("an undersupported automatic race must be rejected");

    assert!(error.contains("auto atom 0"), "unexpected error: {error}");
    assert!(error.contains("at least 16"), "unexpected error: {error}");
    assert_eq!(basis, vec!["auto"], "failure must not install a fallback");
    assert_eq!(dims, vec![2], "failure must not rewrite latent dimension");
}

#[test]
fn quotient_surface_candidates_are_reachable_with_their_cover_geometry() {
    use gam_solve::AutoTopologyKind;

    let coords = Array2::<f64>::zeros((32, 2));
    let specs = topology_candidates_for_dim(coords.view(), 2).unwrap();
    let projective = specs
        .iter()
        .find(|spec| spec.kind == AutoTopologyKind::ProjectivePlane)
        .expect("RP2 must be enrolled in the two-dimensional evidence race");
    assert_eq!(
        projective.geometry.kind(),
        &SaeAtomBasisKind::ProjectivePlane
    );
    assert_eq!(projective.geometry.basis_size().unwrap(), 6);
    let klein = specs
        .iter()
        .find(|spec| spec.kind == AutoTopologyKind::KleinBottle)
        .expect("Klein bottle must be enrolled in the two-dimensional evidence race");
    assert_eq!(klein.geometry.kind(), &SaeAtomBasisKind::KleinBottle);
    assert_eq!(klein.geometry.basis_size().unwrap(), 13);
    assert_eq!(
        klein.manifold,
        LatentManifold::Product(vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ])
    );
}

#[test]
fn torus_race_persists_the_evidence_selected_reference_metric() {
    use ndarray::Array1;

    let side = 10usize;
    let n = side * side;
    let mut coords = Array2::<f64>::zeros((n, 2));
    for i in 0..side {
        for j in 0..side {
            let row = i * side + j;
            coords[[row, 0]] = i as f64 / side as f64;
            coords[[row, 1]] = j as f64 / side as f64;
        }
    }
    let geometry = SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Torus,
        2,
        SaeBasisResolution::TorusHarmonics { per_axis_order: 2 },
        SaeReferenceMetricPlan::FlatRectangularTorus { tau: 0.0 },
    )
    .unwrap();
    let bundle = geometry.evaluate_bundle(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((bundle.basis_values.ncols(), 3));
    for row in 0..decoder.nrows() {
        for col in 0..decoder.ncols() {
            decoder[[row, col]] = (((row + 1) * (col + 2)) as f64).sin() / (1.0 + row as f64);
        }
    }
    let mut target = bundle.basis_values.dot(&decoder);
    for row in 0..n {
        for col in 0..target.ncols() {
            target[[row, col]] += ((row + 3 * col + 1) as f64).cos() / n as f64;
        }
    }
    let weights = Array1::<f64>::ones(n);
    let difference_step = f64::EPSILON.cbrt();
    for family in [TorusMetricFamily::Flat, TorusMetricFamily::EmbeddedDonut] {
        let coordinate = 0.5;
        let analytic = evaluate_torus_metric_profile(
            bundle.basis_values.view(),
            target.view(),
            weights.view(),
            2,
            family,
            coordinate,
        )
        .unwrap();
        let plus = evaluate_torus_metric_profile(
            bundle.basis_values.view(),
            target.view(),
            weights.view(),
            2,
            family,
            coordinate + difference_step,
        )
        .unwrap();
        let minus = evaluate_torus_metric_profile(
            bundle.basis_values.view(),
            target.view(),
            weights.view(),
            2,
            family,
            coordinate - difference_step,
        )
        .unwrap();
        let refitted_direction = (plus.value - minus.value) / (2.0 * difference_step);
        let gap = (analytic.gradient[0] - refitted_direction).abs();
        assert!(
            gap <= difference_step * (1.0 + refitted_direction.abs()),
            "{family:?} coordinate gradient {} disagrees with refitted direction {refitted_direction} by {gap}",
            analytic.gradient[0]
        );
    }
    let spec = TopologyCandidateSpec::new(
        AutoTopologyKind::Torus,
        geometry,
        LatentManifold::Product(vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ]),
        coords,
    )
    .unwrap();
    let fit = fit_topology_candidate(&spec, target.view(), weights.view())
        .expect("torus metric family must reach a converged evidence winner")
        .fit_handle;
    match fit.geometry.reference_metric() {
        SaeReferenceMetricPlan::FlatRectangularTorus { tau } => {
            assert!(tau.is_finite() && *tau >= 0.0)
        }
        SaeReferenceMetricPlan::EmbeddedDonutTorus { tau } => {
            assert!(tau.is_finite() && *tau > 0.0)
        }
        other => panic!("torus race persisted a non-torus reference metric: {other:?}"),
    }
    let persisted_penalty = fit.geometry.build_reference_penalty().unwrap();
    let max_gap = persisted_penalty
        .iter()
        .zip(fit.penalty.iter())
        .fold(0.0_f64, |gap, (left, right)| gap.max((left - right).abs()));
    assert!(
        max_gap <= f64::EPSILON.sqrt(),
        "winning metric plan and installed penalty diverged by {max_gap}"
    );
}

#[test]
fn projective_plane_veronese_embedding_beats_the_unquotiented_sphere_chart() {
    use ndarray::Array1;

    let (n_latitude, n_longitude) = (10usize, 16usize);
    let n = n_latitude * n_longitude;
    let mut coords = Array2::<f64>::zeros((n, 2));
    let mut target = Array2::<f64>::zeros((n, 4));
    for latitude_index in 0..n_latitude {
        let latitude = -std::f64::consts::FRAC_PI_2
            + std::f64::consts::PI * (latitude_index as f64 + 0.5) / n_latitude as f64;
        for longitude_index in 0..n_longitude {
            let longitude = std::f64::consts::TAU * longitude_index as f64 / n_longitude as f64;
            let row = latitude_index * n_longitude + longitude_index;
            coords[[row, 0]] = latitude;
            coords[[row, 1]] = longitude;
            let x = latitude.cos() * longitude.cos();
            let y = latitude.cos() * longitude.sin();
            let z = latitude.sin();
            target[[row, 0]] = x * y;
            target[[row, 1]] = y * z;
            target[[row, 2]] = z * x;
            target[[row, 3]] = 0.5 * (x * x - y * y);
        }
    }
    let manifold = LatentManifold::Product(vec![
        LatentManifold::Interval {
            lo: -std::f64::consts::FRAC_PI_2,
            hi: std::f64::consts::FRAC_PI_2,
        },
        LatentManifold::Circle {
            period: std::f64::consts::TAU,
        },
    ]);
    let projective = TopologyCandidateSpec::new(
        AutoTopologyKind::ProjectivePlane,
        SaeAtomGeometryPlan::projective_plane(1).unwrap(),
        manifold.clone(),
        coords.clone(),
    )
    .unwrap();
    let sphere = TopologyCandidateSpec::new(
        AutoTopologyKind::Sphere,
        SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Sphere,
            2,
            SaeBasisResolution::SphereChart,
            SaeReferenceMetricPlan::SphereChart,
        )
        .unwrap(),
        manifold,
        coords,
    )
    .unwrap();
    let weights = Array1::<f64>::ones(n);
    let projective_fit = fit_topology_candidate(&projective, target.view(), weights.view())
        .expect("RP2 Veronese fit");
    let sphere_fit = fit_topology_candidate(&sphere, target.view(), weights.view())
        .expect("unquotiented sphere-chart fit");
    assert!(
        projective_fit.raw_reml < sphere_fit.raw_reml,
        "RP2 invariant basis must win its Veronese DGP: RP2={}, sphere={}",
        projective_fit.raw_reml,
        sphere_fit.raw_reml
    );
}

#[test]
fn klein_r4_embedding_beats_the_unrestricted_torus_cover() {
    use ndarray::Array1;

    let side = 14usize;
    let n = side * side;
    let mut coords = Array2::<f64>::zeros((n, 2));
    let mut target = Array2::<f64>::zeros((n, 4));
    for theta_index in 0..side {
        for phi_index in 0..side {
            let row = theta_index * side + phi_index;
            let theta_fraction = theta_index as f64 / side as f64;
            let phi_fraction = phi_index as f64 / side as f64;
            coords[[row, 0]] = theta_fraction;
            coords[[row, 1]] = phi_fraction;
            let theta = std::f64::consts::TAU * theta_fraction;
            let phi = std::f64::consts::TAU * phi_fraction;
            let radial = 2.0 + 0.5 * phi.cos();
            target[[row, 0]] = radial * (2.0 * theta).cos();
            target[[row, 1]] = radial * (2.0 * theta).sin();
            target[[row, 2]] = 0.5 * phi.sin() * theta.cos();
            target[[row, 3]] = 0.5 * phi.sin() * theta.sin();
            // A noiseless finite Fourier embedding is represented exactly
            // by both candidate frames and therefore has no finite
            // profiled Gaussian dispersion. Add a small deterministic
            // observation perturbation so this is an evidence comparison,
            // not an attempt to assign REML to a zero-residual sample.
            for output in 0..target.ncols() {
                target[[row, output]] += (((row + 1) * (output + 3)) as f64).sin() / n as f64;
            }
        }
    }
    let manifold = LatentManifold::Product(vec![
        LatentManifold::Circle { period: 1.0 },
        LatentManifold::Circle { period: 1.0 },
    ]);
    let klein = TopologyCandidateSpec::new(
        AutoTopologyKind::KleinBottle,
        SaeAtomGeometryPlan::klein_bottle(2).unwrap(),
        manifold.clone(),
        coords.clone(),
    )
    .unwrap();
    let torus = TopologyCandidateSpec::new(
        AutoTopologyKind::Torus,
        SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Torus,
            2,
            SaeBasisResolution::TorusHarmonics { per_axis_order: 2 },
            SaeReferenceMetricPlan::FlatRectangularTorus { tau: 0.0 },
        )
        .unwrap(),
        manifold,
        coords,
    )
    .unwrap();
    let weights = Array1::<f64>::ones(n);
    let klein_fit = fit_topology_candidate(&klein, target.view(), weights.view())
        .expect("Klein quotient fit");
    let torus_fit = fit_topology_candidate(&torus, target.view(), weights.view())
        .expect("unrestricted torus-cover fit");
    assert!(
        klein_fit.raw_reml < torus_fit.raw_reml,
        "Klein invariant basis must win its R4 DGP: Klein={}, torus={}",
        klein_fit.raw_reml,
        torus_fit.raw_reml
    );
}

/// #2238 — a genuinely two-dimensional primary factor must not be pinned to
/// the old one-dimensional circle. A full 8x8 planar grid is represented
/// exactly by the flat 2-D candidate, while phase alone discards radius.
#[test]
fn auto_primary_topology_selects_two_dimensional_factor_2238() {
    let side = 8usize;
    let target = Array2::<f64>::from_shape_fn((side * side, 2), |(row, col)| {
        let i = row / side;
        let j = row % side;
        if col == 0 {
            i as f64 - 0.5 * (side - 1) as f64
        } else {
            j as f64 - 0.5 * (side - 1) as f64
        }
    });
    let labels = vec![0usize; target.nrows()];
    let choices = discover_primary_atom_topologies(target.view(), &labels, 1, &[2])
        .expect("the supported planar race must produce a winner");

    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0].latent_dim, 2);
    assert_eq!(choices[0].basis_kind, SaeAtomBasisKind::EuclideanPatch);
}

/// #2238/#2239 — a genuinely CURVED 2-D primary factor (a 2-sphere) must be
/// discovered by the fit-entry evidence race as a d=2 sphere chart, not
/// pinned to the 1-D circle default. This is the manifold-zoo plateau and
/// its fix in one test: the circle chart is a function of longitude alone,
/// so it structurally discards latitude and caps the recovery near the
/// observed plateau, while the raced sphere chart reconstructs the planted
/// factor almost exactly. The race both SELECTS the curved chart (over the
/// circle and the flat patch) and, fitted, strictly BEATS the circle-pinned
/// recovery — the two claims the companion issues turn on.
#[test]
fn auto_primary_topology_selects_curved_sphere_and_beats_circle_2238_2239() {
    use gam_solve::AutoTopologyKind;
    use ndarray::Array1;

    // Deterministic (lat, lon) grid on the OPEN sphere (poles excluded so no
    // chart row is degenerate), embedded as the unit 2-sphere in R³.
    let (n_lat, n_lon) = (12usize, 14usize);
    let n = n_lat * n_lon;
    let mut lat = Vec::with_capacity(n);
    let mut lon = Vec::with_capacity(n);
    let mut target = Array2::<f64>::zeros((n, 3));
    for i in 0..n_lat {
        let theta = -std::f64::consts::FRAC_PI_2
            + std::f64::consts::PI * (i as f64 + 1.0) / (n_lat as f64 + 1.0);
        for j in 0..n_lon {
            let phi = std::f64::consts::TAU * j as f64 / n_lon as f64;
            let row = i * n_lon + j;
            target[[row, 0]] = theta.cos() * phi.cos();
            target[[row, 1]] = theta.cos() * phi.sin();
            target[[row, 2]] = theta.sin();
            lat.push(theta);
            lon.push(phi);
        }
    }

    // The primary-atom race (single cluster) must pick the d=2 sphere chart.
    let labels = vec![0usize; n];
    let choices = discover_primary_atom_topologies(target.view(), &labels, 1, &[2])
        .expect("the supported sphere race must produce a winner");
    assert_eq!(choices.len(), 1);
    assert_eq!(
        choices[0].basis_kind,
        SaeAtomBasisKind::Sphere,
        "the curved 2-sphere factor must be discovered as a sphere chart, not a circle/patch"
    );
    assert_eq!(choices[0].latent_dim, 2, "a sphere is intrinsically 2-D");

    // Reconstruction proof of "beats the circle-pinned recovery": fit the
    // circle chart (longitude only) and the sphere chart (lat, lon) to the
    // SAME planted factor through the same REML candidate fitter the race
    // uses, and compare the explained variance each achieves.
    let weights = Array1::<f64>::ones(n);
    let recon_r2 = |spec: &TopologyCandidateSpec| -> f64 {
        let fit = fit_topology_candidate(spec, target.view(), weights.view())
            .expect("candidate fit")
            .fit_handle;
        let recon = fit.phi.dot(&fit.decoder);
        let mut means = [0.0_f64; 3];
        for col in 0..3 {
            let mut acc = 0.0;
            for row in 0..n {
                acc += target[[row, col]];
            }
            means[col] = acc / n as f64;
        }
        let (mut ss_res, mut ss_tot) = (0.0_f64, 0.0_f64);
        for row in 0..n {
            for col in 0..3 {
                let r = target[[row, col]] - recon[[row, col]];
                ss_res += r * r;
                let c = target[[row, col]] - means[col];
                ss_tot += c * c;
            }
        }
        1.0 - ss_res / ss_tot.max(1e-12)
    };

    let mut circle_coords = Array2::<f64>::zeros((n, 1));
    for row in 0..n {
        circle_coords[[row, 0]] = lon[row] / std::f64::consts::TAU;
    }
    let circle_spec = TopologyCandidateSpec::new(
        AutoTopologyKind::Circle,
        SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Periodic,
            1,
            SaeBasisResolution::PeriodicHarmonics { order: 1 },
            SaeReferenceMetricPlan::UnitCircle,
        )
        .unwrap(),
        LatentManifold::Circle { period: 1.0 },
        circle_coords,
    )
    .unwrap();

    let mut sphere_coords = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        sphere_coords[[row, 0]] = lat[row];
        sphere_coords[[row, 1]] = lon[row];
    }
    let sphere_spec = TopologyCandidateSpec::new(
        AutoTopologyKind::Sphere,
        SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Sphere,
            2,
            SaeBasisResolution::SphereChart,
            SaeReferenceMetricPlan::SphereChart,
        )
        .unwrap(),
        LatentManifold::Product(vec![
            LatentManifold::Interval {
                lo: -std::f64::consts::FRAC_PI_2,
                hi: std::f64::consts::FRAC_PI_2,
            },
            LatentManifold::Circle {
                period: std::f64::consts::TAU,
            },
        ]),
        sphere_coords,
    )
    .unwrap();

    let circle_r2 = recon_r2(&circle_spec);
    let sphere_r2 = recon_r2(&sphere_spec);
    eprintln!(
        "[topology-2238] planted 2-sphere R²: circle-pinned={circle_r2:.4} sphere-chart={sphere_r2:.4}"
    );
    assert!(
        sphere_r2 > 0.9,
        "the discovered sphere chart must recover the planted 2-sphere (R²={sphere_r2:.4})"
    );
    assert!(
        circle_r2 < 0.75,
        "the 1-D circle default structurally caps the 2-sphere recovery (R²={circle_r2:.4})"
    );
    assert!(
        sphere_r2 > circle_r2 + 0.2,
        "discovery must strictly beat the circle-pinned recovery (sphere={sphere_r2:.4} vs circle={circle_r2:.4})"
    );
}

/// #2243 — a circle winner's harmonic RESOLUTION is grown by evidence, not
/// pinned to the historical fixed budget (2 harmonics at the default
/// `d_atom = 2`). The planted 1-D factor carries energy at the fundamental
/// AND the 4th harmonic with a GAP between (harmonics 2, 3 are absent), which
/// (a) the 2-harmonic default structurally cannot represent — half the energy
/// lives at 4f — and (b) defeats a naive "stop at the first non-improving
/// resolution" rule, exercising the global-argmin robustness of the selector.
#[test]
fn select_periodic_resolution_grows_past_default_over_harmonic_gap_2243() {
    use gam_solve::AutoTopologyKind;
    use ndarray::Array1;

    // Angular signal in R⁴: fundamental in (col0, col1), 4th harmonic in
    // (col2, col3), nothing at harmonics 2 or 3.
    let n = 240usize;
    let mut coords = Array2::<f64>::zeros((n, 1));
    let mut target = Array2::<f64>::zeros((n, 4));
    for row in 0..n {
        let t = row as f64 / n as f64;
        let angle = std::f64::consts::TAU * t;
        coords[[row, 0]] = t;
        target[[row, 0]] = angle.cos();
        target[[row, 1]] = angle.sin();
        target[[row, 2]] = (4.0 * angle).cos();
        target[[row, 3]] = (4.0 * angle).sin();
    }
    let weights = Array1::<f64>::ones(n);

    let selected = select_periodic_resolution(coords.view(), target.view(), weights.view(), n)
        .expect("resolution selection must succeed on a supported periodic signal");
    assert!(
        selected >= 4,
        "the 4th-harmonic content (past a gap) requires at least 4 harmonics; the fixed \
         2-harmonic default under-resolves it (selected={selected})"
    );

    // Reconstruction: the selected resolution recovers the whole signal while
    // the historical 2-harmonic default cannot touch the 4f half of the energy.
    // `h` is a harmonic ORDER (what `select_periodic_resolution` returns); the
    // periodic basis for order `h` is the `2h + 1` columns `{1, cos, sin, …,
    // cos hθ, sin hθ}`, and `PeriodicHarmonicEvaluator::new` takes that (odd)
    // width, not the order.
    let circle_r2 = |h: usize| -> f64 {
        let spec = TopologyCandidateSpec::new(
            AutoTopologyKind::Circle,
            SaeAtomGeometryPlan::new(
                SaeAtomBasisKind::Periodic,
                1,
                SaeBasisResolution::PeriodicHarmonics { order: h },
                SaeReferenceMetricPlan::UnitCircle,
            )
            .unwrap(),
            LatentManifold::Circle { period: 1.0 },
            coords.clone(),
        )
        .unwrap();
        let fit = fit_topology_candidate(&spec, target.view(), weights.view())
            .expect("candidate fit")
            .fit_handle;
        let recon = fit.phi.dot(&fit.decoder);
        let (mut ss_res, mut ss_tot) = (0.0_f64, 0.0_f64);
        for col in 0..4 {
            let mut mean = 0.0;
            for row in 0..n {
                mean += target[[row, col]];
            }
            mean /= n as f64;
            for row in 0..n {
                let r = target[[row, col]] - recon[[row, col]];
                ss_res += r * r;
                let c = target[[row, col]] - mean;
                ss_tot += c * c;
            }
        }
        1.0 - ss_res / ss_tot.max(1e-12)
    };
    let default_r2 = circle_r2(2);
    let selected_r2 = circle_r2(selected);
    eprintln!(
        "[resolution-2243] circle R²: default(2 harmonics)={default_r2:.4} selected({selected})={selected_r2:.4}"
    );
    assert!(
        selected_r2 > 0.99,
        "the evidence-selected resolution must recover the signal (R²={selected_r2:.4})"
    );
    assert!(
        default_r2 < 0.75,
        "the 2-harmonic default cannot represent the 4th-harmonic half of the energy (R²={default_r2:.4})"
    );
}

/// #2243 — a torus winner's per-axis harmonic ORDER is grown by evidence,
/// not pinned to the fixed `SAE_DEFAULT_TORUS_HARMONICS = 3` budget. The
/// planted toroidal factor carries the fundamental on axis 0 and the 5th
/// harmonic on axis 1 with a GAP (orders 2, 3, 4 absent on that axis), which
/// (a) the order-3 default structurally cannot represent — half the energy
/// lives at 5f — and (b) defeats a naive "stop at the first non-improving
/// order" rule, exercising the global-argmin robustness of the selector.
#[test]
fn select_torus_resolution_grows_past_default_over_harmonic_gap_2243() {
    use gam_solve::AutoTopologyKind;
    use ndarray::Array1;

    // Full g×g angular grid so the integer harmonics below Nyquist (g/2) are
    // exactly resolvable: fundamental on axis 0, 5th harmonic on axis 1.
    let g = 20usize;
    let n = g * g;
    let mut coords = Array2::<f64>::zeros((n, 2));
    let mut target = Array2::<f64>::zeros((n, 4));
    for i in 0..g {
        for j in 0..g {
            let row = i * g + j;
            let t0 = i as f64 / g as f64;
            let t1 = j as f64 / g as f64;
            coords[[row, 0]] = t0;
            coords[[row, 1]] = t1;
            let a0 = std::f64::consts::TAU * t0;
            let a1 = std::f64::consts::TAU * t1;
            target[[row, 0]] = a0.cos();
            target[[row, 1]] = a0.sin();
            target[[row, 2]] = (5.0 * a1).cos();
            target[[row, 3]] = (5.0 * a1).sin();
        }
    }
    let weights = Array1::<f64>::ones(n);

    let selected = select_torus_resolution(coords.view(), target.view(), weights.view(), n)
        .expect("resolution selection must succeed on a supported toroidal signal");
    assert!(
        selected >= 5,
        "the 5th-harmonic content (past a gap) requires at least order 5; the fixed \
         order-3 default under-resolves it (selected={selected})"
    );

    // Reconstruction: the selected order recovers the whole signal while the
    // fixed order-3 default cannot touch the 5f half of the energy.
    let torus_r2 = |h: usize| -> f64 {
        let spec = TopologyCandidateSpec::new(
            AutoTopologyKind::Torus,
            SaeAtomGeometryPlan::new(
                SaeAtomBasisKind::Torus,
                2,
                SaeBasisResolution::TorusHarmonics { per_axis_order: h },
                SaeReferenceMetricPlan::FlatRectangularTorus { tau: 0.0 },
            )
            .unwrap(),
            LatentManifold::Product(vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ]),
            coords.clone(),
        )
        .unwrap();
        let fit = fit_topology_candidate(&spec, target.view(), weights.view())
            .expect("candidate fit")
            .fit_handle;
        let recon = fit.phi.dot(&fit.decoder);
        let (mut ss_res, mut ss_tot) = (0.0_f64, 0.0_f64);
        for col in 0..4 {
            let mut mean = 0.0;
            for row in 0..n {
                mean += target[[row, col]];
            }
            mean /= n as f64;
            for row in 0..n {
                let r = target[[row, col]] - recon[[row, col]];
                ss_res += r * r;
                let c = target[[row, col]] - mean;
                ss_tot += c * c;
            }
        }
        1.0 - ss_res / ss_tot.max(1e-12)
    };
    let default_r2 = torus_r2(SAE_DEFAULT_TORUS_HARMONICS);
    let selected_r2 = torus_r2(selected);
    eprintln!(
        "[resolution-2243] torus R²: default(order {SAE_DEFAULT_TORUS_HARMONICS})={default_r2:.4} selected({selected})={selected_r2:.4}"
    );
    assert!(
        selected_r2 > 0.99,
        "the evidence-selected order must recover the signal (R²={selected_r2:.4})"
    );
    assert!(
        default_r2 < 0.75,
        "the order-3 default cannot represent the 5th-harmonic half of the energy (R²={default_r2:.4})"
    );
}

/// Deterministic low-discrepancy sequence on `[0, 1)` (van der Corput, base
/// 2) for RNG-free synthetic birth targets.
fn vdc(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let (mut x, mut denom, mut k) = (0.0_f64, 2.0_f64, i + 1);
            while k > 0 {
                x += (k & 1) as f64 / denom;
                denom *= 2.0;
                k >>= 1;
            }
            x
        })
        .collect()
}

/// F1 radial promotion: a `d = 1` birth whose per-row amplitude is a
/// CONTINUOUS spread (a disk, radius uniform in area ⇒ density ∝ r) must
/// enrich the race with the circle-vs-cylinder-vs-disk candidate set; a
/// present/absent (bimodal) birth must NOT promote.
#[test]
fn radial_promotion_fires_only_on_continuous_amplitude() {
    let n = 400;
    let coords = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    // Disk: place each row on a circle of radius r_i = sqrt(u_i) (area-uniform
    // radius, density ∝ r ⇒ Beta(2,1) ⇒ continuous), so the per-row amplitude
    // (row norm) is a continuous spread.
    let u = vdc(n);
    let disk = Array2::from_shape_fn((n, 2), |(i, j)| {
        let r = u[i].sqrt();
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        if j == 0 {
            r * theta.cos()
        } else {
            r * theta.sin()
        }
    });
    let promoted = radial_promoted_specs(coords.view(), disk.view(), 1)
        .expect("promotion decision")
        .expect("disk amplitude is continuous ⇒ promotion fires");
    let kinds: std::collections::HashSet<_> = promoted.iter().map(|s| s.kind).collect();
    assert!(kinds.contains(&AutoTopologyKind::Circle), "{kinds:?}");
    assert!(kinds.contains(&AutoTopologyKind::Cylinder), "{kinds:?}");
    assert!(kinds.contains(&AutoTopologyKind::Euclidean), "{kinds:?}");
    // No key collision: each promoted kind appears once.
    assert_eq!(kinds.len(), promoted.len());
    let expected_radial =
        standardized_log_birth_amplitudes(birth_row_amplitudes(disk.view()).view())
            .expect("disk log-amplitude spread");
    for spec in promoted.iter().filter(|spec| {
        matches!(
            spec.kind,
            AutoTopologyKind::Cylinder | AutoTopologyKind::Euclidean
        )
    }) {
        for row in 0..n {
            assert!(
                (spec.coords[[row, 1]] - expected_radial[row]).abs() < 1.0e-12,
                "promoted {:?} row {row} axis 1 must be standardized log-amplitude",
                spec.kind
            );
        }
    }

    // Present/absent circle: half the rows on the unit circle (amplitude 1),
    // half at the origin (amplitude 0) ⇒ bimodal ⇒ spike ⇒ NO promotion.
    let ring = Array2::from_shape_fn((n, 2), |(i, j)| {
        if i % 2 == 0 {
            0.0
        } else {
            let theta = std::f64::consts::TAU * (i as f64 / n as f64);
            if j == 0 { theta.cos() } else { theta.sin() }
        }
    });
    assert!(
        radial_promoted_specs(coords.view(), ring.view(), 1)
            .expect("promotion decision")
            .is_none(),
        "present/absent birth must not promote"
    );

    // A d != 1 birth never promotes (radial promotion is the d=1→2 lift).
    assert!(
        radial_promoted_specs(coords.view(), disk.view(), 2)
            .expect("promotion decision")
            .is_none()
    );
}

fn topology_fit_sse(fit: &TopologyRaceFit, target: ArrayView2<'_, f64>) -> f64 {
    let fitted = fit.phi.dot(&fit.decoder);
    let mut sse = 0.0_f64;
    for row in 0..target.nrows() {
        for col in 0..target.ncols() {
            let err = target[[row, col]] - fitted[[row, col]];
            sse += err * err;
        }
    }
    sse
}

#[test]
fn radial_promotion_seed_coordinate_expresses_annulus_radius() {
    let n_angles = 16;
    let n_radii = 25;
    let n = n_angles * n_radii;
    let radial = vdc(n_radii);
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| {
        let angle_idx = row / n_radii;
        angle_idx as f64 / n_angles as f64
    });
    let annulus = Array2::from_shape_fn((n, 2), |(row, col)| {
        let angle_idx = row / n_radii;
        let radius_idx = row % n_radii;
        let theta = std::f64::consts::TAU * (angle_idx as f64 / n_angles as f64);
        let radius = 0.3 + 0.7 * radial[radius_idx].sqrt();
        if col == 0 {
            radius * theta.cos()
        } else {
            radius * theta.sin()
        }
    });
    let promoted = radial_promoted_specs(coords.view(), annulus.view(), 1)
        .expect("promotion decision")
        .expect("annulus radius spread promotes a radial axis");
    let circle = promoted
        .iter()
        .find(|spec| spec.kind == AutoTopologyKind::Circle)
        .expect("promoted race includes the circle alternative");
    let cylinder = promoted
        .iter()
        .find(|spec| spec.kind == AutoTopologyKind::Cylinder)
        .expect("promoted race includes the cylinder alternative");
    let weights = Array1::<f64>::ones(n);
    let circle_fit =
        fit_topology_candidate(circle, annulus.view(), weights.view()).expect("circle fit");
    let cylinder_fit =
        fit_topology_candidate(cylinder, annulus.view(), weights.view()).expect("cylinder fit");
    let circle_sse = topology_fit_sse(&circle_fit.fit_handle, annulus.view());
    let cylinder_sse = topology_fit_sse(&cylinder_fit.fit_handle, annulus.view());
    assert!(
        cylinder_sse < 0.75 * circle_sse,
        "radial seed should let cylinder express radius variation: cylinder_sse={cylinder_sse}, circle_sse={circle_sse}"
    );
}

#[test]
fn birth_row_amplitudes_are_row_norms() {
    let y = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
    let a = birth_row_amplitudes(y.view());
    assert!((a[0] - 5.0).abs() < 1e-12);
    assert!((a[1]).abs() < 1e-12);
}

// ---- F2: finite-set (discrete anchor) atom ------------------------------

#[test]
fn finite_set_race_is_not_enrolled_by_default() {
    // Containment: the finite-set candidate is inert unless explicitly
    // enrolled, so the enum arm + evaluator can never affect a birth by
    // default.
    assert!(!finite_set_race_enrolled());
    set_finite_set_race_enrolled(true);
    assert!(finite_set_race_enrolled());
    set_finite_set_race_enrolled(false);
    assert!(!finite_set_race_enrolled());
}

#[test]
fn finite_set_candidate_fires_on_discrete_occupancy() {
    // Seven-point cyclic occupancy (weekdays): the coordinate collapses onto
    // 7 anchors, so the finite-set candidate builder returns 7 anchors and a
    // per-row integer index in [0, 7); the rank charge is anchors − 1 = 6.
    let per = 100;
    let mut rows = Vec::new();
    for i in 0..(7 * per) {
        // Sub-resolution embedding noise (±1e-3 over a span of 6 ⇒ ~1.7e-4
        // normalized, below the width floor) so the seven weekdays are a
        // genuine finite point set, not seven fuzzy blobs whose structured
        // noise the evidence could honestly resolve into more clusters.
        rows.push((i % 7) as f64 + 0.001 * ((i as f64).sin()));
    }
    let coords = Array2::from_shape_vec((7 * per, 1), rows).unwrap();
    let (anchors, idx) =
        finite_set_candidate_for_birth(coords.view()).expect("discrete ⇒ finite-set candidate");
    assert_eq!(anchors, 7, "anchors");
    assert_eq!(crate::manifold::finite_set_rank_charge(anchors), 6);
    // Every index is a valid anchor bin.
    assert!(
        idx.iter()
            .all(|&v| (0.0..=6.0).contains(&v) && v.fract() == 0.0)
    );

    // A uniformly-occupied coordinate is NOT a finite set — no candidate.
    let n = 400;
    let uni = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    assert!(finite_set_candidate_for_birth(uni.view()).is_none());
}

#[test]
fn anchor_indicator_evaluator_is_one_hot_with_zero_jets() {
    use crate::basis::{AnchorIndicatorEvaluator, SaeBasisEvaluator, SaeBasisSecondJet};
    let ev = AnchorIndicatorEvaluator::new(3).unwrap();
    // Coordinates snap to nearest anchor index; the design is one-hot.
    let coords = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 1.4]).unwrap();
    let (phi, jet) = ev.evaluate(coords.view()).unwrap();
    assert_eq!(phi.dim(), (4, 3));
    // Row sums are 1 (exactly one active anchor per row).
    for r in 0..4 {
        assert!((phi.row(r).sum() - 1.0).abs() < 1e-12);
    }
    assert!((phi[[0, 0]] - 1.0).abs() < 1e-12);
    assert!((phi[[1, 1]] - 1.0).abs() < 1e-12);
    assert!((phi[[2, 2]] - 1.0).abs() < 1e-12);
    assert!((phi[[3, 1]] - 1.0).abs() < 1e-12); // 1.4 rounds to anchor 1
    // The indicator is piecewise constant: all jets are zero.
    assert!(jet.iter().all(|&v| v == 0.0));
    let h = ev.second_jet(coords.view()).unwrap();
    assert!(h.iter().all(|&v| v == 0.0));
}

/// Build a `K`-atom periodic SAE term whose per-row routing is dictated by a
/// caller-supplied boolean activity matrix `active[(row, atom)]` (ON/OFF
/// logits). Every atom shares the same circle basis; only the routing (and,
/// for the birth template, the decoder) differs. Returns the term and a
/// matching ρ with native ARD enabled (one axis per atom).
fn planted_term(active: &[Vec<bool>]) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = active.len();
    let k = active[0].len();
    let p = 4usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut atoms = Vec::with_capacity(k);
    let mut coord_blocks = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let mut decoder = Array2::<f64>::zeros((3, p));
        // Give each atom a distinct decoder direction so reconstruction is
        // non-degenerate.
        decoder[[1, atom_idx % p]] = 1.0;
        decoder[[2, (atom_idx + 1) % p]] = 1.0;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("atom_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords.clone());
    }
    let mut logits = Array2::<f64>::zeros((n, k));
    for (row, atom_active) in active.iter().enumerate() {
        for (atom, &on) in atom_active.iter().enumerate() {
            logits[[row, atom]] = if on { ON } else { OFF };
        }
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
    (term, rho)
}

fn residuals_of(term: &SaeManifoldTerm) -> Array2<f64> {
    // A term scored against zero target gives R = −fitted; non-degenerate
    // residuals for the birth channel.
    let fitted = term.try_fitted().unwrap();
    -&fitted
}

/// #977 discovery oracle: with the production birth budget enabled, a fit
/// #1230 — `StructureSearchResult::structure_changed()` is the trigger the
/// FFI uses to decide whether the pre-search joint-Hessian shape bands are
/// stale and must be recomputed from the final post-search model.
///
/// It must report `true` iff at least one move LANDED and mutated the
/// returned `term`/`rho`: an `Accepted` move (certified birth / fission /
/// fusion + warm refit) or a `Demoted` death. It must report `false` when
/// every round was contested / vetoed (the term/rho are byte-for-byte the
/// pre-search fit, so the exact joint-Hessian bands stay valid), and when no
/// round ran at all. A false negative leaves seed atoms with stale bands
/// (the #1230 bug); a false positive needlessly discards exact bands.
#[test]
fn structure_changed_is_true_only_when_a_move_lands() {
    use gam_solve::structure_search::{MoveRecord, MoveVerdict};

    fn ledger_with(verdicts: Vec<MoveVerdict>) -> SearchLedger {
        SearchLedger {
            alpha: 0.05,
            moves: verdicts
                .into_iter()
                .enumerate()
                .map(|(i, verdict)| MoveRecord {
                    mv: StructureMove::Death { atom: i },
                    trigger: 0.0,
                    structure_hash: i as u64,
                    claim: ClaimKind::AtomExists { atom: i },
                    verdict,
                })
                .collect(),
            collapse_events: Vec::new(),
        }
    }

    // No rounds ran at all: nothing changed.
    let (term0, rho0) = planted_term(&[vec![true], vec![true]]);
    let empty = StructureSearchResult::from_rounds(term0.clone(), rho0.clone(), Vec::new());
    assert!(
        !empty.structure_changed(),
        "no rounds ⇒ the term/rho are the pre-search fit ⇒ structure_changed() must be false"
    );

    // Every move contested or vetoed: the dictionary is byte-for-byte the
    // pre-search fit, so the exact joint-Hessian bands remain valid.
    let no_landed = StructureSearchResult::from_rounds(
        term0.clone(),
        rho0.clone(),
        vec![ledger_with(vec![
            MoveVerdict::Contested { log_e: -1.0 },
            MoveVerdict::Vetoed { log_e: -2.0 },
        ])],
    );
    assert!(
        !no_landed.structure_changed(),
        "all-contested/vetoed rounds leave the model unchanged ⇒ structure_changed() must be false"
    );

    // An Accepted move landed (certified restructuring + warm refit): the
    // returned model differs from the pre-search fit ⇒ bands are stale.
    let accepted = StructureSearchResult::from_rounds(
        term0.clone(),
        rho0.clone(),
        vec![ledger_with(vec![
            MoveVerdict::Contested { log_e: -1.0 },
            MoveVerdict::Accepted { log_e: 3.0 },
        ])],
    );
    assert!(
        accepted.structure_changed(),
        "a landed Accepted move mutates term/rho ⇒ structure_changed() must be true (recompute bands)"
    );

    // A Demoted death is also a landed structure change.
    let demoted = StructureSearchResult::from_rounds(
        term0.clone(),
        rho0.clone(),
        vec![ledger_with(vec![MoveVerdict::Demoted { log_e: -1.0 }])],
    );
    assert!(
        demoted.structure_changed(),
        "a landed Demoted death folds an atom to ~0 routing ⇒ structure_changed() must be true"
    );
}

/// whose residuals carry an unexplained factor direction (a structure the
/// current dictionary does not express) HARVESTS a birth proposal — the
/// candidate atom whose held-out e-value the gate then adjudicates. This is
/// the proposal channel the production site re-enabled (`max_births > 0`);
/// without it K could never grow.
#[test]
fn residual_bearing_fit_harvests_birth_proposal() {
    // An OVERCOMPLETE dictionary (G = 8 atoms, exactly one active per token ⇒
    // L0 = 1) so the #2233 birth pre-screen the harvest channel now embeds
    // (`predicted_birth_dl_bits`) sees a real support saving log₂(G/L0) = 3
    // bits/token — the term that funds a curved birth. A single-atom dictionary
    // (G = 1, log₂(G/L0) = 0) offers ZERO overcompleteness and a lone LINEAR
    // residual direction earns no code saving either, so the pre-screen
    // correctly refuses to birth there: growth pays only once a curved atom
    // spares the extra active slots a flat span would spend.
    let n = 40usize;
    let k = 8usize;
    let active: Vec<Vec<bool>> = (0..n)
        .map(|row| (0..k).map(|atom| atom == row % k).collect())
        .collect();
    let (term, rho) = planted_term(&active);
    // Inject a genuinely CURVED (rank-2) residual: a zero-mean circle living in
    // the 2-plane spanned by two ORTHONORMAL output directions `u ⟂ v`. Its two
    // factor coordinates carry balanced, common-mode-free energy (an isotropic
    // 2-D cloud, not a dominant shared direction), so the LOCAL ambient span the
    // pre-screen measures on the candidate's own firing rows is ≈ 2 ⇒ priced as
    // a circle (d = 1, code term 0) with a positive support saving. This is the
    // curved structure a born atom absorbs — the birth the theorem admits.
    let p = term.output_dim();
    let mut residuals = Array2::<f64>::zeros((n, p));
    let u = [0.6_f64, -0.4, 0.5, -0.3];
    let v = [0.4_f64, 0.6, 0.3, 0.5]; // u · v = 0, ‖u‖ = ‖v‖
    let un: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    let vn: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for row in 0..n {
        let theta = std::f64::consts::TAU * (row as f64) / (n as f64);
        let (s, c) = theta.sin_cos();
        for out in 0..p {
            residuals[[row, out]] = 2.0 * (c * u[out] / un + s * v[out] / vn);
        }
    }
    let params = HarvestParams {
        max_fusions: 0,
        max_fissions: 0,
        // The production-enabled budget (births > 0) — the whole point of
        // #977: K can grow.
        max_births: 2,
    };
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
    let births: usize = report
        .proposals
        .iter()
        .filter(|p| matches!(p.mv, StructureMove::Birth { .. }))
        .count();
    assert!(
        births >= 1,
        "a residual-bearing fit with births enabled must harvest at least \
         one birth proposal (so K can be discovered); got {:?}",
        report.proposals.iter().map(|p| &p.mv).collect::<Vec<_>>()
    );
    assert!(
        report.births_proposed >= 1,
        "births_proposed must count the harvested births; got {}",
        report.births_proposed
    );
    assert!(
        report.birth_skipped_reason.is_none(),
        "the birth channel must run (no skip) on a non-degenerate residual; got {:?}",
        report.birth_skipped_reason
    );
}

/// #977 NULL oracle: a target the dictionary reconstructs exactly leaves
/// ZERO residual, so the birth channel finds no factor subspace and proposes
/// no birth — nothing is born under the null. (The round driver's e-gate is
/// the second line of defense; this asserts the harvest itself does not
/// manufacture growth where there is no unexplained structure.)
#[test]
fn fully_reconstructed_null_harvests_no_birth() {
    let n = 40usize;
    let active: Vec<Vec<bool>> = (0..n).map(|_| vec![true]).collect();
    let (term, rho) = planted_term(&active);
    // Residual ≡ 0: the dictionary reconstructs the target exactly, so there
    // is no unexplained factor to mine.
    let p = term.output_dim();
    let zero_residual = Array2::<f64>::zeros((n, p));
    let params = HarvestParams {
        max_fusions: 0,
        max_fissions: 0,
        max_births: 2,
    };
    let report = harvest_move_proposals(&term, &rho, zero_residual.view(), &params).unwrap();
    let births: usize = report
        .proposals
        .iter()
        .filter(|p| matches!(p.mv, StructureMove::Birth { .. }))
        .count();
    assert_eq!(
        births, 0,
        "a fully-reconstructed (zero-residual) null must harvest no birth \
         proposal; got {births} births"
    );
}

/// Oracle (#997 trigger): a planted SHATTER — two atoms with identical
/// supports (one curved family re-encoded as near-duplicate flat atoms) —
/// produces a FUSION proposal on that pair (symmetric code dependence ≈ 1),
/// and NO fission audit (asymmetry ≈ 0).
#[test]
fn planted_shatter_harvests_fusion_not_fission() {
    // Atoms 0 and 1 share support exactly (every third row); atom 2 is
    // independent. n = 30.
    let n = 30usize;
    let active: Vec<Vec<bool>> = (0..n)
        .map(|row| {
            let dup = row % 3 == 0;
            vec![dup, dup, row % 2 == 0]
        })
        .collect();
    let (term, rho) = planted_term(&active);
    let residuals = residuals_of(&term);
    let params = HarvestParams {
        max_fusions: 4,
        max_fissions: 4,
        max_births: 0,
    };
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
    let has_fusion_01 = report.proposals.iter().any(|p| {
        matches!(p.mv, StructureMove::Fusion { a, b } if (a, b) == (0, 1) || (a, b) == (1, 0))
    });
    assert!(
        has_fusion_01,
        "shattered duplicate pair (0,1) must yield a fusion proposal; got {:?}",
        report.proposals.iter().map(|p| &p.mv).collect::<Vec<_>>()
    );
    // The duplicate pair is symmetric ⇒ no absorption fission audit on it.
    let has_fission = report
        .proposals
        .iter()
        .any(|p| matches!(p.mv, StructureMove::Fission { .. }));
    assert!(
        !has_fission,
        "symmetric duplicate supports must not trigger an absorption fission audit"
    );
}

/// Oracle (#997 trigger): a planted ABSORPTION (A⊇B: B's support nests
/// inside A's) produces a FISSION audit on the parent A (high conditional
/// asymmetry, parent conditional ≈ 1). The planted atoms are 1-D `Periodic`
/// (NOT a `d = 2` product), so the #993 within-atom carve is undefined on
/// them and the candidate rides on the co-activation audit — recorded
/// loudly via `fission_carve_unavailable_count`, never silent.
#[test]
fn planted_absorption_harvests_fission_audit_with_loud_carve_skip() {
    // Atom 0 (parent) active on rows ≡ 0 mod 2 PLUS rows ≡ 1 mod 4; atom 1
    // (child) active only on rows ≡ 0 mod 4 — strictly nested in 0's
    // support ⇒ P(0|1) = 1, P(1|0) < 1. n = 40.
    let n = 40usize;
    let active: Vec<Vec<bool>> = (0..n)
        .map(|row| {
            let child = row % 4 == 0;
            let parent = row % 2 == 0 || row % 4 == 1;
            vec![parent, child, row % 5 == 0]
        })
        .collect();
    let (term, rho) = planted_term(&active);
    let residuals = residuals_of(&term);
    let params = HarvestParams {
        max_fusions: 4,
        max_fissions: 4,
        max_births: 0,
    };
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
    let fissioned_parent = report
        .proposals
        .iter()
        .any(|p| matches!(p.mv, StructureMove::Fission { atom: 0 }));
    assert!(
        fissioned_parent,
        "nested-support parent (atom 0) must be flagged for a fission audit; got {:?}",
        report.proposals.iter().map(|p| &p.mv).collect::<Vec<_>>()
    );
    assert_eq!(
        report.fission_carve_ran_count, 0,
        "1-D periodic atoms are not a product manifold; the within-atom carve cannot run"
    );
    assert!(
        report.fission_carve_unavailable_count >= 1,
        "the non-product fission candidate must be recorded as carve-unavailable, not silent"
    );
    assert!(
        report.fission_carve_results.is_empty(),
        "no carve ran, so there are no carve results to report"
    );
}

/// Oracle (#997 type-I): three INDEPENDENT planted atoms (marginal supports
/// at coprime strides) yield NO fusion proposal — the trigger does not
/// manufacture binding edges where the codes are independent, so the e-gate
/// is never even asked to reject a true null.
#[test]
fn independent_atoms_harvest_no_fusion() {
    let n = 60usize;
    let active: Vec<Vec<bool>> = (0..n)
        .map(|row| vec![row % 2 == 0, row % 3 == 0, row % 5 == 0])
        .collect();
    let (term, rho) = planted_term(&active);
    let residuals = residuals_of(&term);
    let params = HarvestParams {
        max_fusions: 4,
        max_fissions: 4,
        max_births: 0,
    };
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
    let has_fusion = report
        .proposals
        .iter()
        .any(|p| matches!(p.mv, StructureMove::Fusion { .. }));
    assert!(
        !has_fusion,
        "independent atom supports must not produce fusion proposals; got {:?}",
        report.proposals.iter().map(|p| &p.mv).collect::<Vec<_>>()
    );
}

/// #1890 verification fixture: plant ONE circle tiled into `k` arc atoms with
/// DISJOINT supports. Row `r` (of `n`) sits at circle phase `2π·r/n`; arc atom
/// `j` owns the contiguous row block `[j·n/k, (j+1)·n/k)` (routed ON there,
/// OFF elsewhere). Every atom shares the SAME periodic basis and a unit-circle
/// decoder (harmonic cols 1,2 → ambient dims 0,1) scaled by `decoder_scale[j]`,
/// so a shared scale makes all arcs decode onto ONE closed curve — the
/// over-tiling signature the co-activation fusion lane is structurally blind to
/// (disjoint supports ⇒ anti-correlated codes). A per-atom scale plants a
/// CONCENTRIC circle of a different radius (a genuinely distinct manifold —
/// the negative control).
fn tiled_circle_term(
    n: usize,
    k: usize,
    decoder_scale: &[f64],
) -> (SaeManifoldTerm, SaeManifoldRho) {
    assert_eq!(decoder_scale.len(), k, "one decoder scale per arc atom");
    let p = 4usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let mut atoms = Vec::with_capacity(k);
    let mut coord_blocks = Vec::with_capacity(k);
    for (j, &scale) in decoder_scale.iter().enumerate() {
        // Unit-circle decoder: first cos/sin harmonic → ambient {0,1}, radius
        // `scale`. Shared scale ⇒ ONE circle (glue); distinct scale ⇒ a
        // concentric distinct circle (no glue).
        let mut decoder = Array2::<f64>::zeros((m, p));
        decoder[[1, 0]] = scale;
        decoder[[2, 1]] = scale;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("arc_{j}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords.clone());
    }
    let mut logits = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        let owner = (row * k) / n; // contiguous arc ownership
        for j in 0..k {
            logits[[row, j]] = if j == owner { ON } else { OFF };
        }
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
    (term, rho)
}

/// #1890 negative control (type-I): two CONCENTRIC circles of different radii —
/// a shared plane (so the frames align and the geometric pre-screen DOES
/// nominate the pair), disjoint adjacent arc supports, but a NON-isometric
/// transition onto a genuinely distinct manifold. Because the pair is screened,
/// this exercises the seam EQUIVALENCE e-value itself, not the pre-screen: it
/// must REJECT (negative log-e), so the engine's positive-evidence e-gate never
/// accepts the glue. Clusters that DON'T glue must not be forced together.
#[test]
fn distinct_concentric_circles_do_not_glue() {
    let n = 40usize;
    let (term, rho) = tiled_circle_term(n, 2, &[1.0, 2.0]); // radius 1 vs radius 2
    let residuals = Array2::<f64>::zeros((n, 4));
    let params = HarvestParams {
        max_fusions: 4,
        max_fissions: 4,
        max_births: 0,
    };
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();

    // No co-activation fusion (disjoint supports), same as the tiling case.
    assert!(
        !report
            .proposals
            .iter()
            .any(|p| matches!(p.mv, StructureMove::Fusion { .. })),
        "disjoint supports must yield no co-activation fusion"
    );

    // The shared plane aligns the frames, so the pair IS screened — this is a
    // genuine test of the equivalence e-value, which must reject the distinct
    // manifold.
    assert!(
        report.glue_candidates_screened >= 1,
        "the shared-plane pair must be geometrically screened"
    );
    let (e_distinct, _) = unit_speed_glue_certificate(&term, residuals.view(), 0, 1)
        .expect("the aligned pair yields a seam e-value and transition certificate");
    assert!(
        e_distinct.log_e_value < 0.0,
        "distinct concentric circles must NOT glue (negative log-e), got {}",
        e_distinct.log_e_value
    );

    // Any emitted glue proposal therefore carries negative evidence — rejected
    // by the engine's e-gate (which certifies only positive log-e).
    for p in &report.proposals {
        if matches!(p.mv, StructureMove::Glue { .. }) {
            assert!(
                p.trigger < 0.0,
                "the distinct-circle glue must carry negative evidence, got {}",
                p.trigger
            );
        }
    }
}

#[test]
fn sphere_polar_factor_requires_identifiable_full_rank_alignment() {
    let rank_deficient = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
    assert!(nearest_orthogonal_3x3(rank_deficient).is_none());

    let proper = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let recovered = nearest_orthogonal_3x3(proper).unwrap();
    for row in 0..3 {
        for column in 0..3 {
            assert!(
                (recovered[row][column] - proper[row][column]).abs() <= 16.0 * f64::EPSILON
            );
        }
    }
}

#[test]
fn closed_form_transition_uses_first_nonzero_harmonic_without_scanning() {
    let mut decoder_a = Array2::<f64>::zeros((5, 3));
    decoder_a[[3, 0]] = 2.0;
    decoder_a[[4, 1]] = 1.0;
    decoder_a[[0, 2]] = 0.25;
    let decoder_b = periodic_decoder_under_transition(decoder_a.view(), -1, 0.125).unwrap();
    let (sign, offset) =
        fit_periodic_transition_from_decoders(decoder_a.view(), decoder_b.view()).unwrap();
    assert_eq!(sign, -1);
    let recovered = periodic_decoder_under_transition(decoder_a.view(), sign, offset).unwrap();
    for (actual, expected) in recovered.iter().zip(decoder_b.iter()) {
        assert!((actual - expected).abs() < 32.0 * f64::EPSILON);
    }
}

/// An orientation-reversing seam is a valid equivalence, but it is NOT a
/// license to erase either local chart.  The production proposal must select
/// the atlas-register outcome, and applying it must preserve the numerical
/// chart count and fitted image while reducing the semantic atom count.
#[test]
fn orientation_reversing_seam_registers_atlas_without_destructive_fusion() {
    let n = 40usize;
    let (mut term, rho) = tiled_circle_term(n, 2, &[1.0, 1.0]);
    // `sin` changes sign under t -> -t; `cos` does not.  B therefore traces
    // exactly A's image with the orientation-reversing transition t_A=-t_B.
    term.atoms[1].decoder_coefficients[[1, 0]] = -1.0;
    let residuals = Array2::<f64>::zeros((n, 4));
    let (transition, _) = unit_speed_glue_certificate(&term, residuals.view(), 0, 1)
        .expect("reflected charts have an exact certified seam");
    assert_eq!(transition.sign, -1);
    assert!(transition.log_e_value > 5.0);

    let report = harvest_move_proposals(
        &term,
        &rho,
        residuals.view(),
        &HarvestParams {
            max_fusions: 4,
            max_fissions: 0,
            max_births: 0,
        },
    )
    .unwrap();
    let mv = report
        .proposals
        .iter()
        .find_map(|proposal| match proposal.mv {
            StructureMove::Glue {
                a,
                b,
                outcome: ChartGlueOutcome::RegisterAtlas,
            } => Some(StructureMove::Glue {
                a,
                b,
                outcome: ChartGlueOutcome::RegisterAtlas,
            }),
            _ => None,
        })
        .expect("negative seam must propose atlas registration");

    let fitted_before = term.try_fitted().unwrap();
    let (registered, _) = apply_structure_move(&term, &rho, &mv, &[]).unwrap();
    assert_eq!(registered.k_atoms(), 2, "both local charts must survive");
    assert_eq!(registered.semantic_atom_count(), 1);
    assert_eq!(registered.chart_atlases().len(), 1);
    assert_eq!(registered.chart_atlases()[0].transitions()[0].sign, -1);
    assert_eq!(
        registered.try_fitted().unwrap(),
        fitted_before,
        "atlas registration is an image-exact quotient"
    );

    let assignments = registered.assignment.assignments();
    for row in 0..n {
        let (activation, partition) = registered
            .atlas_partition_of_unity(0, assignments.row(row))
            .unwrap();
        assert!((partition.sum() - 1.0).abs() < 8.0 * f64::EPSILON);
        for (slot, &chart) in registered.chart_atlases()[0].charts().iter().enumerate() {
            assert!(
                (activation * partition[slot] - assignments[[row, chart]]).abs()
                    < 8.0 * f64::EPSILON
            );
        }
    }
}

/// #1890 over-birth reassembly — the PHYSICAL-excision resurrection fix, on the
/// private primitives the `chart_gluing_1890.rs` integration test cannot reach.
/// A circle over-tiled into 4 disjoint arcs proposes a spanning set of glues
/// (each arc pair lies on ONE circle, so `unit_speed_glue_certificate`
/// certifies with large positive log-e) with ZERO co-activation fusions. A
/// round that accepts a matching of those glues then EXCISES the folded
/// partners for real ([`compact_glued_atoms`]): folding + demoting alone leaves
/// a zero-mass atom the #976/#1003 active-mass guard revives on the next refit,
/// so the effective count never falls; removal drops both the raw and active
/// size, and the wider-arc survivors STILL glue — the round sequence converges
/// toward the single reassembled chart K=1.
#[test]
fn over_tiling_physical_excision_reduces_k_toward_one() {
    use gam_solve::structure_search::{MoveRecord, MoveVerdict};

    let n = 32usize;
    let (mut term, mut rho) = tiled_circle_term(n, 4, &[1.0; 4]);
    assert_eq!(term.k_atoms(), 4);

    // The primitive certifies every arc pair as ONE circle (large positive
    // log-e), and the co-activation lane stays silent on the disjoint supports.
    let residuals0 = Array2::<f64>::zeros((n, 4));
    let (e_arc, _) = unit_speed_glue_certificate(&term, residuals0.view(), 0, 2)
        .expect("a d=1 aligned disjoint pair yields a certified seam e-value");
    assert!(
        e_arc.log_e_value > 5.0,
        "e_glue must certify two arcs of one circle, got {}",
        e_arc.log_e_value
    );
    let params0 = HarvestParams {
        max_fusions: 16,
        max_fissions: 0,
        max_births: 0,
    };
    let report0 = harvest_move_proposals(&term, &rho, residuals0.view(), &params0).unwrap();
    assert!(
        !report0
            .proposals
            .iter()
            .any(|p| matches!(p.mv, StructureMove::Fusion { .. })),
        "disjoint tiling must yield no co-activation fusion"
    );
    assert!(
        report0.glues_proposed >= 3,
        "a spanning set (≥ k−1 = 3 edges) must reassemble the 4 arcs, got {}",
        report0.glues_proposed
    );
    let first_epoch_glue_claims: Vec<ClaimKind> = report0
        .proposals
        .iter()
        .filter(|proposal| matches!(proposal.mv, StructureMove::Glue { .. }))
        .map(|proposal| proposal.claim.clone())
        .collect();
    assert!(!first_epoch_glue_claims.is_empty());

    // Seed old-K state that MUST NOT survive a physical dictionary resize.
    // These are all legitimate transient states immediately after a fit;
    // the compactor owns invalidating them before the reduced-K polish.
    term.assignment.frozen_logits = Some(term.assignment.logits.clone());
    term.last_frames_active = true;
    term.fixed_decoder_assembly = true;
    term.border_hbb_workspace = Array2::<f64>::ones((3, 3));
    term.decoder_repulsion_gate = Some(vec![(0, 1, 1.0)]);
    term.streaming_gates_frozen = true;
    term.expected_criterion_gauge_deflated_directions = Some(7);
    term.criterion_gauge_deflation_reanchors = 2;
    term.criterion_gauge_deflation_last_delta_sign = -1;
    term.dictionary_cocollapse_reseeds = 3;
    term.structural_cocollapse_reseeds = 4;
    let accepted_glue = |a: usize, b: usize| MoveRecord {
        mv: StructureMove::Glue {
            a,
            b,
            outcome: ChartGlueOutcome::Fuse,
        },
        trigger: 40.0,
        structure_hash: 0,
        claim: ClaimKind::Custom {
            label: format!("seam_glue:{a}:{b}"),
        },
        verdict: MoveVerdict::Accepted { log_e: 40.0 },
    };
    // A matching (shares no atom) of disjoint glues — the within-round `touched`
    // guard admits exactly this shape in one round.
    let ledger = SearchLedger {
        alpha: 0.05,
        moves: vec![accepted_glue(0, 1), accepted_glue(2, 3)],
        collapse_events: Vec::new(),
    };
    let removed =
        compact_glued_atoms(&mut term, &mut rho, &ledger, &report0.certified_glues).unwrap();
    assert_eq!(
        removed, 2,
        "both folded partners must be physically excised"
    );
    assert_eq!(
        term.k_atoms(),
        2,
        "physical excision must reduce K from 4 to 2 (no active-mass resurrection)"
    );
    assert!(
        term.assignment
            .logits
            .rows()
            .into_iter()
            .all(|row| row.as_slice().is_some()),
        "compaction must materialize a row-contiguous router for the polish refit"
    );
    assert_eq!(
        rho.log_ard.len(),
        2,
        "ρ ARD blocks must fall in lock-step with the atoms"
    );
    assert_eq!(
        rho.log_lambda_smooth.len(),
        2,
        "ρ smoothness blocks must fall in lock-step with the atoms"
    );
    assert!(
        term.assignment.frozen_logits.is_none(),
        "an old-K frozen router cannot survive compaction"
    );
    assert!(!term.last_frames_active);
    assert!(!term.fixed_decoder_assembly);
    assert_eq!(term.border_hbb_workspace.dim(), (0, 0));
    assert!(term.decoder_repulsion_gate.is_none());
    assert!(!term.streaming_gates_frozen);
    assert_eq!(term.expected_criterion_gauge_deflated_directions, None);
    assert_eq!(term.criterion_gauge_deflation_reanchors, 0);
    assert_eq!(term.criterion_gauge_deflation_last_delta_sign, 0);
    assert_eq!(term.dictionary_cocollapse_reseeds, 0);
    assert_eq!(term.structural_cocollapse_reseeds, 0);
    // The two survivors each now cover a half-circle and STILL glue — the round
    // sequence converges toward the single reassembled chart (K=1).
    let residuals = Array2::<f64>::zeros((n, 4));
    let params = HarvestParams {
        max_fusions: 4,
        max_fissions: 0,
        max_births: 0,
    };
    let report2 = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
    assert!(
        report2.glues_proposed >= 1,
        "the two reassembled half-circle survivors must still glue toward K=1"
    );
    for proposal in report2
        .proposals
        .iter()
        .filter(|proposal| matches!(proposal.mv, StructureMove::Glue { .. }))
    {
        assert!(
            !first_epoch_glue_claims.contains(&proposal.claim),
            "a reduced dictionary must not reuse old atom-index evidence: {:?}",
            proposal.claim
        );
    }
}

/// The compactor must adopt the transition measured while the retired chart
/// still had live support. Folding first demotes B and makes seam fitting
/// impossible, so the harvest certificate carries both the transition and
/// B's certified support rows to the round boundary.
#[test]
fn physical_excision_transplants_coords_from_the_live_seam() {
    use gam_solve::structure_search::{MoveRecord, MoveVerdict};

    let (mut term, mut rho) = tiled_circle_term(32, 2, &[1.0; 2]);
    assert!(term.assignment.frozen_logits.is_none());
    let seam = fit_seam_transition(&term, 0, 1).expect("live pair has a seam");
    let residuals = Array2::<f64>::zeros((32, 4));
    let (_, certificate) = unit_speed_glue_certificate(&term, residuals.view(), 0, 1)
        .expect("live pair has a harvest-time glue certificate");
    let da = term.assignment.coords[0].latent_dim();
    let db = term.assignment.coords[1].latent_dim();
    let flat_b = term.assignment.coords[1].as_flat().to_owned();
    let expected: Vec<(usize, f64)> = seam
        .rows_b
        .iter()
        .map(|&row| {
            let mapped = (seam.sign * flat_b[row * db] + seam.offset).rem_euclid(seam.period);
            (row, mapped)
        })
        .collect();
    // Poison only A's INACTIVE coordinates on B's rows.  A correct
    // transplant overwrites every sentinel through the measured seam.
    let mut flat_a = term.assignment.coords[0].as_flat().to_owned();
    for &(row, mapped) in &expected {
        flat_a[row * da] = (mapped + 0.37).rem_euclid(seam.period);
    }
    term.assignment.coords[0].set_flat(flat_a.view());
    // Simulate the scoring-refit drift that caused #1890's production red:
    // the terminal state can no longer identify a decoder transition.  The
    // accepted structural object is nevertheless fully specified by the
    // live harvest certificate above and must not be inferred again here.
    term.atoms[1].decoder_coefficients.fill(0.0);
    assert!(
        fit_seam_transition(&term, 0, 1).is_none(),
        "post-harvest fixture must make seam re-fitting impossible"
    );

    let ledger = SearchLedger {
        alpha: 0.05,
        moves: vec![MoveRecord {
            mv: StructureMove::Glue {
                a: 0,
                b: 1,
                outcome: ChartGlueOutcome::Fuse,
            },
            trigger: 40.0,
            structure_hash: 0,
            claim: ClaimKind::Custom {
                label: "test-live-seam".to_string(),
            },
            verdict: MoveVerdict::Accepted { log_e: 40.0 },
        }],
        collapse_events: Vec::new(),
    };
    compact_glued_atoms(&mut term, &mut rho, &ledger, &[certificate]).unwrap();

    assert_eq!(term.k_atoms(), 1);
    let survivor = term.assignment.coords[0].as_flat();
    for (row, mapped) in expected {
        assert!(
            (survivor[row * da] - mapped).abs() < 1.0e-12,
            "row {row}: survivor coordinate {} != seam-mapped {mapped}",
            survivor[row * da]
        );
    }
}

/// Variable-K compaction is transactional: malformed paired ρ state returns
/// an error before changing the term, instead of panicking halfway through a
/// gather and leaving atom/assignment arrays at different widths.
#[test]
fn physical_excision_validation_is_transactional() {
    use gam_solve::structure_search::{MoveRecord, MoveVerdict};

    let (mut term, mut rho) = tiled_circle_term(16, 3, &[1.0; 3]);
    let atoms_before = term.k_atoms();
    let logits_before = term.assignment.logits.clone();
    rho.log_ard.pop();
    let remove = std::collections::BTreeSet::from([1usize]);

    let err = remove_atoms(&mut term, &mut rho, &remove).unwrap_err();
    assert!(
        err.contains("rho per-atom lengths"),
        "unexpected error: {err}"
    );
    assert_eq!(term.k_atoms(), atoms_before);
    assert_eq!(term.assignment.logits, logits_before);

    // Restore ρ and feed an impossible overlapping matching.  The compactor
    // must reject the whole ledger before the first fold mutates any logit.
    rho.log_ard.push(Array1::zeros(1));
    let accepted = |a: usize, b: usize| MoveRecord {
        mv: StructureMove::Glue {
            a,
            b,
            outcome: ChartGlueOutcome::Fuse,
        },
        trigger: 40.0,
        structure_hash: 0,
        claim: ClaimKind::Custom {
            label: format!("test-glue:{a}:{b}"),
        },
        verdict: MoveVerdict::Accepted { log_e: 40.0 },
    };
    let overlapping = SearchLedger {
        alpha: 0.05,
        moves: vec![accepted(0, 1), accepted(1, 2)],
        collapse_events: Vec::new(),
    };
    let err = compact_glued_atoms(&mut term, &mut rho, &overlapping, &[]).unwrap_err();
    assert!(
        err.contains("not an atom-disjoint matching"),
        "unexpected error: {err}"
    );
    assert_eq!(term.k_atoms(), atoms_before);
    assert_eq!(term.assignment.logits, logits_before);

    // A structurally valid accepted record without its harvest certificate
    // is also rejected transactionally.  Re-fitting a replacement seam at
    // this boundary would recreate the scoring/adoption ordering bug.
    let missing_certificate = SearchLedger {
        alpha: 0.05,
        moves: vec![accepted(0, 1)],
        collapse_events: Vec::new(),
    };
    let err = compact_glued_atoms(&mut term, &mut rho, &missing_certificate, &[]).unwrap_err();
    assert!(
        err.contains("no harvest-time certificate"),
        "unexpected error: {err}"
    );
    assert_eq!(term.k_atoms(), atoms_before);
    assert_eq!(term.assignment.logits, logits_before);
}

/// Oracle (#997 death trigger): a diverged ARD precision yields a DEATH
/// proposal; a terminal collapse event yields a death even with finite ARD.
#[test]
fn diverged_ard_and_terminal_collapse_harvest_deaths() {
    let n = 20usize;
    let active: Vec<Vec<bool>> = (0..n).map(|row| vec![true, row % 2 == 0, false]).collect();
    let (mut term, mut rho) = planted_term(&active);
    // Diverge atom 2's ARD precision well past the divergence floor.
    rho.log_ard[2] = Array1::from_elem(1, ARD_DIVERGENCE_LOG_PRECISION + 5.0);
    // Inject a terminal collapse for atom 1 (finite ARD, but routing gone).
    term.record_collapse_event(CollapseEvent {
        iteration: 3,
        atom: 1,
        max_active_mass: 1e-6,
        floor: 1e-3,
        action: CollapseAction::Terminal,
    });
    let residuals = residuals_of(&term);
    let params = HarvestParams {
        max_fusions: 0,
        max_fissions: 0,
        max_births: 0,
    };
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();
    let death_atoms: Vec<usize> = report
        .proposals
        .iter()
        .filter_map(|p| match p.mv {
            StructureMove::Death { atom } => Some(atom),
            _ => None,
        })
        .collect();
    assert!(
        death_atoms.contains(&2),
        "diverged ARD on atom 2 must yield a death proposal; got {death_atoms:?}"
    );
    assert!(
        death_atoms.contains(&1),
        "terminal collapse on atom 1 must yield a death proposal; got {death_atoms:?}"
    );
}

/// Apply-move restructuring oracle: fission GROWS the dictionary by one atom
/// (child inherits parent's basis + ARD block), fusion and death keep K
/// (fold / demote), birth appends a residual-factor atom.
#[test]
fn apply_move_restructures_warm() {
    let n = 12usize;
    let active: Vec<Vec<bool>> = (0..n).map(|row| vec![true, row % 2 == 0]).collect();
    let (term, rho) = planted_term(&active);
    let k0 = term.k_atoms();

    // Fission: K grows, child ARD block inherited.
    let (fissioned, fissioned_rho) =
        apply_structure_move(&term, &rho, &StructureMove::Fission { atom: 0 }, &[]).unwrap();
    assert_eq!(fissioned.k_atoms(), k0 + 1);
    assert_eq!(fissioned_rho.log_ard.len(), k0 + 1);
    // Every length-K ρ vector the penalty assembler indexes by atom must
    // grow with K, not just `log_ard`. `log_lambda_smooth` is read as
    // `lambda_smooth[atom_idx]` in construction.rs; a stale length-K vector
    // panics out of bounds on the K-th (new) atom (#357).
    assert_eq!(
        fissioned_rho.log_lambda_smooth.len(),
        fissioned.k_atoms(),
        "fission must grow per-atom log_lambda_smooth in lockstep with K"
    );

    // Fusion: K unchanged, atom b demoted to ~0 routing.
    let (fused, _) =
        apply_structure_move(&term, &rho, &StructureMove::Fusion { a: 0, b: 1 }, &[]).unwrap();
    assert_eq!(fused.k_atoms(), k0);
    let fused_assign = fused.assignment.assignments();
    assert!(
        fused_assign.column(1).iter().all(|&m| m < 1e-6),
        "fused-away atom 1 must route to ~0 mass"
    );

    // Death: K unchanged, atom demoted.
    let (dead, _) =
        apply_structure_move(&term, &rho, &StructureMove::Death { atom: 1 }, &[]).unwrap();
    assert_eq!(dead.k_atoms(), k0);
    let dead_assign = dead.assignment.assignments();
    assert!(dead_assign.column(1).iter().all(|&m| m < 1e-6));

    // Birth: K grows, and the new atom RECONSTRUCTS the residual-factor image.
    //
    // Since the #977 topology RACE, a born atom no longer carries the raw
    // `factor_dir` coefficients verbatim: its topology is chosen by evidence
    // and its decoder is the winning basis's penalized least-squares fit to the
    // birth target `Y = Φ_template · factor_dir` (so the raw coefficient
    // `[[0,0]]` is shrunk by the fit ridge — `0.6999…`, not exactly `0.7`).
    // The structural invariant the move must preserve is therefore
    // RECONSTRUCTION PARITY, not coefficient identity: the born atom, evaluated
    // on its own coordinates with its own (raced) basis, must reproduce the
    // birth-target image to within the small fit ridge.
    let p = term.output_dim();
    let m = term.atoms[0].basis_size();
    let mut decoder = Array2::<f64>::zeros((m, p));
    decoder[[0, 0]] = 0.7;
    let birth_target = term.atoms[0].basis_values.dot(&decoder); // Φ_template · factor_dir
    let (born, born_rho) = apply_structure_move(
        &term,
        &rho,
        &StructureMove::Birth { candidate: 0 },
        &[decoder],
    )
    .unwrap();
    assert_eq!(born.k_atoms(), k0 + 1);
    assert_eq!(born_rho.log_ard.len(), k0 + 1);
    // ρ's per-atom smoothness vector must grow in step with K (the #1556
    // contract `assemble_arrow_schur` validates); a stale-length vector would
    // panic the next assemble on the per-atom `lambda_smooth[atom_idx]` index.
    assert_eq!(born_rho.log_lambda_smooth.len(), k0 + 1);
    let born_atom = &born.atoms[k0];
    let born_image = born_atom.basis_values.dot(&born_atom.decoder_coefficients);
    assert_eq!(born_image.dim(), birth_target.dim());
    let mut max_recon_err = 0.0_f64;
    for (a, b) in born_image.iter().zip(birth_target.iter()) {
        max_recon_err = max_recon_err.max((a - b).abs());
    }
    assert!(
        max_recon_err < 1e-3,
        "born atom must reconstruct the residual-factor image (penalized fit); \
         max |Φ_born·B_born − Φ_template·factor_dir| = {max_recon_err:.3e} (> 1e-3)"
    );
}

/// #357 regression: after a structure move that GROWS the atom count
/// (fission/birth), the returned ρ's per-atom `log_lambda_smooth` must be
/// length-K so the penalty assembler's `lambda_smooth[atom_idx]` read does
/// not panic out of bounds. Before the fix `duplicate_atom`/`born_atom`
/// pushed only `log_ard`, leaving `log_lambda_smooth` one short — the next
/// `assemble_arrow_schur_inner` panicked with `index out of bounds: the len
/// is K but the index is K` (construction.rs `scaled_s[[i,j]] =
/// lambda_smooth[atom_idx] * s_ij`). This drives the REAL assembly so it
/// fails on the buggy path, not just on a length assertion.
#[test]
fn grown_atom_count_assembles_without_lambda_smooth_oob_357() {
    let n = 16usize;
    let active: Vec<Vec<bool>> = (0..n).map(|row| vec![true, row % 2 == 0]).collect();
    let (term, rho) = planted_term(&active);
    let target = Array2::<f64>::from_shape_fn((n, term.output_dim()), |(row, col)| {
        0.1 * (row as f64) - 0.05 * (col as f64)
    });

    // Fission grows K by one.
    let (fissioned, fissioned_rho) =
        apply_structure_move(&term, &rho, &StructureMove::Fission { atom: 0 }, &[]).unwrap();
    assert_eq!(fissioned_rho.log_lambda_smooth.len(), fissioned.k_atoms());
    // The assembly indexes lambda_smooth[atom_idx] for every atom; on the
    // pre-fix ρ this panicked out of bounds for the new K-th atom.
    let mut fissioned = fissioned;
    fissioned
        .assemble_arrow_schur_scaled(target.view(), &fissioned_rho, None, 1.0)
        .expect("post-fission assembly must not panic or error on the grown atom set");

    // Birth grows K by one and must assemble too.
    let p = term.output_dim();
    let m = term.atoms[0].basis_size();
    let mut decoder = Array2::<f64>::zeros((m, p));
    decoder[[0, 0]] = 0.5;
    let (born, born_rho) = apply_structure_move(
        &term,
        &rho,
        &StructureMove::Birth { candidate: 0 },
        &[decoder],
    )
    .unwrap();
    assert_eq!(born_rho.log_lambda_smooth.len(), born.k_atoms());
    let mut born = born;
    born.assemble_arrow_schur_scaled(target.view(), &born_rho, None, 1.0)
        .expect("post-birth assembly must not panic or error on the grown atom set");
}

/// Ledger byte-determinism oracle (#997): two runs of the round driver over
/// the same planted shatter, with a deterministic scripted fit, serialize
/// the per-round ledgers byte-identically.
#[test]
fn round_driver_ledger_is_byte_deterministic() {
    let n = 24usize;
    let active: Vec<Vec<bool>> = (0..n)
        .map(|row| {
            let dup = row % 3 == 0;
            vec![dup, dup, row % 2 == 0]
        })
        .collect();

    let run = || {
        let (term, rho) = planted_term(&active);
        let target = Array2::<f64>::zeros((n, term.output_dim()));
        let mut ledger = gam_terms::inference::structure_evidence::StructureLedger::new();
        let budget = MoveBudget {
            max_moves: 4,
            alpha: 0.05,
        };
        let params = HarvestParams {
            max_fusions: 4,
            max_fissions: 0,
            max_births: 0,
        };
        let config = RoundDriverConfig {
            n_shards: 3,
            budget,
            harvest_params: params,
            curl: None,
        };
        // Deterministic no-op fit: the scripted gate sees the unrefit
        // candidate (the engine's determinism is what this asserts, not the
        // SAE inner solve).
        run_structure_search_rounds(
            term,
            rho,
            target.view(),
            config,
            &mut ledger,
            |t, r, _| Ok((t, r)),
            |t, r, _| Ok((t, r)),
            // No-op polish: this determinism oracle scripts the gate and
            // never runs the SAE inner solve.
            |t, r, _| Ok((t, r)),
        )
        .unwrap()
    };

    let a = run();
    let b = run();
    let sa = serde_json::to_string(&a.rounds).unwrap();
    let sb = serde_json::to_string(&b.rounds).unwrap();
    assert_eq!(
        sa, sb,
        "identical inputs must produce a byte-identical ledger"
    );
    assert_eq!(a.term.k_atoms(), b.term.k_atoms());
}

/// Estimation/eval split oracle: the split reserves estimation rows and
/// partitions the remainder into held-out shards that do NOT overlap the
/// estimation set (the universal-inference contract the gates rely on).
#[test]
fn estimation_eval_split_is_disjoint() {
    let target = Array2::<f64>::zeros((20, 3));
    let split = estimation_eval_split(target.view(), 4);
    assert!(!split.estimation_rows.is_empty());
    assert!(!split.shards.is_empty());
    let est: std::collections::HashSet<usize> = split.estimation_rows.iter().copied().collect();
    for shard in &split.shards {
        for &row in &shard.rows {
            assert!(
                !est.contains(&row),
                "eval shard row {row} must not be in the estimation set"
            );
        }
    }
}

/// #977 per-atom topology RACE oracle: two birth targets — one tracing a
/// CIRCLE in output space as the coordinate sweeps, the other a straight
/// LINE — must be assigned DIFFERENT topologies by evidence. A genuine
/// dictionary learner does not stamp every born atom with atom-0's circle
/// template: the circular residual earns a Periodic (circle) basis, the
/// straight residual a EuclideanPatch (line). This is the heterogeneous,
/// evidence-chosen dictionary the issue demands.
#[test]
fn birth_topology_race_assigns_circle_vs_line_by_evidence() {
    use std::f64::consts::TAU;

    let n = 80usize;
    // A monotone 1-D latent coordinate the residual image is parameterized by.
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);

    // CIRCLE target: γ(t) = (cos 2πt, sin 2πt) — full revolution, strong
    // turning a straight line cannot express. Two output channels carry the
    // circle; the rest are zero.
    let p = 4usize;
    let mut circle_target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let t = coords[[row, 0]];
        circle_target[[row, 0]] = (TAU * t).cos();
        circle_target[[row, 1]] = (TAU * t).sin();
    }

    // LINE target: γ(t) = t·u — a straight ray, zero turning. The circle basis
    // has no parsimony advantage; the cheaper line wins on evidence.
    let mut line_target = Array2::<f64>::zeros((n, p));
    let u = [0.7_f64, -0.4, 0.5, -0.2];
    for row in 0..n {
        let t = coords[[row, 0]];
        for c in 0..p {
            line_target[[row, c]] = t * u[c];
        }
    }

    let weights = Array1::<f64>::ones(n);

    let circle_fit =
        race_birth_topology(coords.view(), circle_target.view(), weights.view(), 1)
            .expect("circle race runs")
            .expect("circle race has a realizable candidate");
    let line_fit = race_birth_topology(coords.view(), line_target.view(), weights.view(), 1)
        .expect("line race runs")
        .expect("line race has a realizable candidate");

    assert_eq!(
        circle_fit.geometry.kind(),
        &SaeAtomBasisKind::Periodic,
        "a circular birth residual must win the circle (Periodic) topology"
    );
    assert_eq!(
        line_fit.geometry.kind(),
        &SaeAtomBasisKind::EuclideanPatch,
        "a straight birth residual must win the line (EuclideanPatch) topology"
    );
    // The crux: the two atoms get DIFFERENT topologies by evidence — the
    // dictionary is heterogeneous, not all-circle.
    assert_ne!(
        circle_fit.geometry.kind(),
        line_fit.geometry.kind(),
        "the discovery must assign DIFFERENT topologies to the circle and line \
         atoms (evidence-chosen, not inherited)"
    );
}

/// #977 d=2 topology-race COMPLETENESS: the candidate set includes the
/// Cylinder kind, and a birth target that is genuinely cylindrical — periodic
/// along one latent axis and unbounded-linear along the other — is adjudicated
/// to the Cylinder topology, not forced into a torus (which would wrap the
/// linear axis spuriously) or a flat patch (which would lose the periodicity).
/// This is the realizable d=2 race the issue demands: torus / sphere /
/// euclidean / cylinder, evidence-chosen.
#[test]
fn birth_topology_race_d2_includes_and_selects_cylinder() {
    use std::f64::consts::TAU;

    // The d=2 candidate set must literally CONTAIN the cylinder candidate.
    let n = 120usize;
    let coords = Array2::<f64>::from_shape_fn((n, 2), |(row, axis)| {
        // axis 0: a phase that completes ~2 revolutions over the rows;
        // axis 1: a monotone unbounded coordinate.
        if axis == 0 {
            (row as f64 / n as f64) * 2.0
        } else {
            (row as f64 / n as f64) * 3.0 - 1.5
        }
    });
    let specs = topology_candidates_for_dim(coords.view(), 2).expect("d=2 candidates build");
    let has_cylinder = specs
        .iter()
        .any(|s| s.geometry.kind() == &SaeAtomBasisKind::Cylinder);
    assert!(
        has_cylinder,
        "the d=2 topology-race candidate set MUST include the Cylinder kind; got {:?}",
        specs.iter().map(|s| s.geometry.kind()).collect::<Vec<_>>()
    );
    let has_torus = specs
        .iter()
        .any(|s| s.geometry.kind() == &SaeAtomBasisKind::Torus);
    let has_sphere = specs
        .iter()
        .any(|s| s.geometry.kind() == &SaeAtomBasisKind::Sphere);
    let has_patch = specs
        .iter()
        .any(|s| s.geometry.kind() == &SaeAtomBasisKind::EuclideanPatch);
    assert!(
        has_torus && has_sphere && has_patch,
        "the d=2 race must be COMPLETE (torus + sphere + euclidean + cylinder)"
    );

    // CYLINDER target: periodic along axis 0 (cos/sin of the phase) AND
    // linearly growing along axis 1 (a magnitude ramp). A torus would have to
    // wrap the magnitude axis (no periodicity there); a flat patch cannot
    // express the full revolution; the cylinder expresses both exactly.
    let p = 4usize;
    let mut cyl_target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let phase = coords[[row, 0]];
        let mag = coords[[row, 1]];
        cyl_target[[row, 0]] = (TAU * phase).cos();
        cyl_target[[row, 1]] = (TAU * phase).sin();
        // The linear-axis structure: a magnitude ramp on a third channel.
        cyl_target[[row, 2]] = mag;
    }
    let weights = Array1::<f64>::ones(n);
    let cyl_fit = race_birth_topology(coords.view(), cyl_target.view(), weights.view(), 2)
        .expect("cylinder race runs")
        .expect("cylinder race has a realizable candidate");
    assert_eq!(
        cyl_fit.geometry.kind(),
        &SaeAtomBasisKind::Cylinder,
        "a cylindrical birth residual (periodic along one axis, linear along the \
         other) must win the Cylinder topology by evidence; got {:?}",
        cyl_fit.geometry.kind()
    );
}

/// #1218 PRODUCTION-GATE wiring proof: the corrected PG gate-block
/// normalizer is consumed by the live per-shard likelihood the K-vs-(K+1)
/// birth gate forms its split-LR from — not just by the isolated unit test.
///
/// `eval_log_lik` is the exact `alternative_log_lik` / `null_sup_log_lik`
/// closure `run_atom_birth_gate` accumulates (see [`run_structure_search_rounds`]),
/// so it is the production gate's evaluation statistic. We score the SAME
/// shard under a K-atom null and a (K+1)-atom candidate and isolate the
/// gate-block contribution: growing the dictionary by one atom adds exactly
/// one gate coordinate, so the `−½·d_g·log(2π)` normalizer (the term #1218
/// fixed the sign of) does NOT cancel in the gate difference. With the
/// corrected (subtracted) sign it is an Occam PENALTY that resists the
/// extra atom; the buggy (added) sign would flip it into a spurious REWARD.
#[test]
fn production_gate_consumes_corrected_pg_normalizer() {
    let n = 32usize;
    // K=2 null and a K=3 candidate, every atom routed on every row so the
    // gate logits are well-defined and finite.
    let null_active: Vec<Vec<bool>> = (0..n).map(|_| vec![true, true]).collect();
    let cand_active: Vec<Vec<bool>> = (0..n).map(|_| vec![true, true, true]).collect();
    let (null_term, _) = planted_term(&null_active);
    let (cand_term, _) = planted_term(&cand_active);
    assert_eq!(null_term.k_atoms(), 2);
    assert_eq!(cand_term.k_atoms(), 3, "candidate grows K by one atom");

    // One held-out shard: the row block the gate accumulates evidence over.
    let p = null_term.output_dim();
    let target = Arc::new(Array2::<f64>::zeros((n, p)));
    let shard = RowBlockShard {
        target: target.clone(),
        rows: (0..n).collect(),
    };

    // The gate-block contribution alone (private helper the live
    // `eval_log_lik` adds in): the corrected normalizer is reachable here.
    let null_gate = gate_block_log_evidence(&null_term, &shard).unwrap();
    let cand_gate = gate_block_log_evidence(&cand_term, &shard).unwrap();
    assert!(
        null_gate.is_finite() && cand_gate.is_finite(),
        "gate-block evidence must be finite on a well-posed gate block"
    );

    // The Occam normalizer per added gate coordinate. The candidate carries
    // K+1 gate coordinates, the null K, so the gate-difference includes one
    // extra `−½·log(2π)` normalizer that must NOT cancel.
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let gate_delta = cand_gate - null_gate;

    // Corrected sign ⇒ the per-coordinate normalizer SUBTRACTS, so the
    // extra atom's gate-block log-evidence is pushed DOWN by ≈ ½·log(2π)
    // relative to a no-normalizer baseline. The decisive, sign-sensitive
    // assertion: the extra-coordinate normalizer is the *negative*
    // ½·log(2π) Occam term, never the positive (buggy) one. Compare against
    // the per-atom evidence WITHOUT the normalizer to isolate it.
    let per_atom_no_norm = |term: &SaeManifoldTerm| -> f64 {
        // Re-derive the gate evidence with the normalizer ADDED back (the
        // pre-fix sign) to recover the unnormalized quadratic/logdet part.
        // `gate_block_log_evidence` already SUBTRACTS ½·d_g·log(2π); adding
        // it back yields the normalizer-free score, and the difference
        // between candidate and null of THAT isolates everything except the
        // one extra normalizer.
        let dg = term.k_atoms() as f64; // one gate coordinate per atom
        gate_block_log_evidence(term, &shard).unwrap() + 0.5 * dg * log_2pi
    };
    let no_norm_delta = per_atom_no_norm(&cand_term) - per_atom_no_norm(&null_term);
    let normalizer_in_delta = gate_delta - no_norm_delta;

    // The normalizer contribution to the K→K+1 gate difference must be
    // exactly `−½·log(2π)` (one extra gate coordinate, corrected sign).
    assert!(
        (normalizer_in_delta + 0.5 * log_2pi).abs() < 1e-9,
        "the gate-block normalizer in the K→K+1 difference must be the \
         corrected −½·log(2π) Occam penalty, got {normalizer_in_delta} \
         (buggy +½·log(2π) = {})",
        0.5 * log_2pi
    );

    // And the full production statistic carries it: the gate-block evidence
    // is a real, finite addend on top of the reconstruction likelihood.
    let full = eval_log_lik(&cand_term, &shard).unwrap();
    let recon_only = {
        // Reconstruction-only baseline (what the path returned BEFORE the
        // wiring): −½·SSE over the shard rows.
        let fitted = cand_term.try_fitted().unwrap();
        let mut sse = 0.0;
        for &row in &shard.rows {
            for out in 0..p {
                let d = fitted[[row, out]] - shard.target[[row, out]];
                sse += d * d;
            }
        }
        -0.5 * sse
    };
    assert!(
        (full - (recon_only + cand_gate)).abs() < 1e-9,
        "the live per-shard likelihood must equal reconstruction + the \
         PG gate-block evidence (so the corrected normalizer reaches the gate)"
    );
}

/// Fission must BREAK the parent/child symmetry. Duplicating an atom
/// identically (same decoder, mass split 50/50) sits at a symmetric saddle of
/// the joint refit — the children's gradients are identical, so a
/// deterministic refit never separates them and the fission is a no-op the
/// e-gate rejects. The anti-symmetric perturbation makes the two children's
/// decoders genuinely differ (so the refit can separate factors) while the
/// equal-mass combined decoder `½(parent+child)` stays EXACTLY the original
/// (warm-start preserved).
#[test]
fn fission_breaks_symmetry_so_children_can_separate() {
    let (term, rho) = planted_term(&vec![vec![true]; 8]);
    assert_eq!(term.k_atoms(), 1);
    let orig = term.atoms[0].decoder_coefficients.clone();

    let (child, _child_rho) =
        apply_structure_move(&term, &rho, &StructureMove::Fission { atom: 0 }, &[]).unwrap();
    assert_eq!(child.k_atoms(), 2, "fission must add one atom");

    let d0 = &child.atoms[0].decoder_coefficients;
    let d1 = &child.atoms[1].decoder_coefficients;
    // (1) Symmetry BROKEN: the children's decoders are not identical (without
    // this the refit is stuck at the symmetric saddle and fission is a no-op).
    let sep = (d0 - d1).iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = orig.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    assert!(
        sep / scale > 1.0e-3,
        "fission children must NOT be identical (symmetric saddle); rel sep = {}",
        sep / scale
    );
    // (2) Warm-start preserved EXACTLY: the equal-mass combined decoder is the
    // original (the anti-symmetric ±ε perturbation cancels).
    let combined = (d0 + d1).mapv(|x| 0.5 * x);
    let warm_err = (&combined - &orig)
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    assert!(
        warm_err < 1.0e-12,
        "mass-split combined decoder must equal the original; err = {warm_err}"
    );
    // (3) Mass split is EVEN: the parent and child carry equal routing logits
    // on every row (each gets half the parent's softmax mass).
    for row in 0..child.assignment.logits.nrows() {
        assert!(
            (child.assignment.logits[[row, 0]] - child.assignment.logits[[row, 1]]).abs()
                < 1e-12,
            "fission must split routing mass 50/50 (equal child logits)"
        );
    }
}

/// Softmax fusion must PRESERVE the combined routing mass. Merging the two
/// constituent logits with `logsumexp` keeps `mass(fused) = mass(a)+mass(b)`;
/// the old `max` under-massed the fused atom (½ vs ⅔ on this 3-atom fixture
/// where atoms 0,1 are co-active and atom 2 competes), leaving the warm-start
/// short and risking a FALSE e-gate rejection of a good fusion under a capped
/// refit. (For ordered Beta--Bernoulli routing `max` stays correct — the gate is un-normalized.)
#[test]
fn fusion_preserves_combined_softmax_mass() {
    let (term, rho) = planted_term(&vec![vec![true, true, true]; 6]);
    let combined: Vec<f64> = (0..6)
        .map(|r| {
            let a = term.assignment.try_assignments_row(r).unwrap();
            a[0] + a[1]
        })
        .collect();
    let (fused, _) =
        apply_structure_move(&term, &rho, &StructureMove::Fusion { a: 0, b: 1 }, &[]).unwrap();
    for r in 0..6 {
        let a = fused.assignment.try_assignments_row(r).unwrap();
        assert!(
            (a[0] - combined[r]).abs() < 1e-6,
            "fused atom must carry the COMBINED softmax mass (logsumexp, not \
             max): got {}, want {} (row {r})",
            a[0],
            combined[r]
        );
        // Sanity: plain max would have given ½ here, materially short of ⅔.
        assert!(
            combined[r] > 0.6,
            "fixture must exercise a co-active pair (combined mass {} should be ~⅔)",
            combined[r]
        );
    }
}

#[test]
fn fusion_of_zero_mass_pair_yields_neg_inf_not_nan() {
    // Folding two atoms whose softmax logits are BOTH -∞ (zero routing mass on
    // a row) must give the mass-preserving combined logit -∞ (combined mass 0),
    // NOT NaN. Pre-fix, `logsumexp(-∞,-∞)` evaluated `(-∞)-(-∞)=NaN` and poisoned
    // the entire logits row.
    let (mut term, rho) = planted_term(&vec![vec![true, true, true]; 6]);
    assert!(
        matches!(term.assignment.mode, AssignmentMode::Softmax { .. }),
        "fixture must be softmax-routed to exercise the logsumexp combine"
    );
    // Zero out atoms 0 and 1 on row 0 (both -∞), leave the rest finite.
    term.assignment.logits[[0, 0]] = f64::NEG_INFINITY;
    term.assignment.logits[[0, 1]] = f64::NEG_INFINITY;
    let (fused, _) =
        apply_structure_move(&term, &rho, &StructureMove::Fusion { a: 0, b: 1 }, &[]).unwrap();
    let folded = fused.assignment.logits[[0, 0]];
    assert!(
        !folded.is_nan(),
        "fused zero-mass logit must not be NaN (got {folded})"
    );
    assert_eq!(
        folded,
        f64::NEG_INFINITY,
        "combined mass of two zero-mass atoms is zero → logit -∞"
    );
    // The whole row must stay NaN-free so softmax over it is well defined.
    for c in 0..fused.assignment.logits.ncols() {
        assert!(
            !fused.assignment.logits[[0, c]].is_nan(),
            "row 0 col {c} must not be NaN after the fold"
        );
    }
}

// =======================================================================
// Curl / flatten Phase-4 killer demo (INTEGRATION_PLAN §8 definition of
// done): plant a centered circle, let a NONNEGATIVE-gate linear dictionary
// shatter it into four rectified half-atoms (±u, ±v), and show the curl
// proposer coalesces them, recovers the circle, and would win on the
// evidence the race reads — while a Gaussian-fill plane is NOT curled, a
// diameter-collapsed circle flattens, and a healthy ring is left alone.
// =======================================================================

/// A straight-line (Linear) atom whose ambient image is `t ↦ t · dir` over
/// the supplied per-row coordinate — `Φ = [1, t]`, decoder rows
/// `[0; dir]`. This is the rectified half-atom a nonnegative gate parks on
/// one lobe of a centered signed direction.
fn linear_line_atom(name: &str, coord: &Array1<f64>, dir: &Array1<f64>) -> SaeManifoldAtom {
    let n = coord.len();
    let p = dir.len();
    let mut phi = Array2::<f64>::zeros((n, 2));
    let mut jet = ndarray::Array3::<f64>::zeros((n, 2, 1));
    for r in 0..n {
        phi[[r, 0]] = 1.0;
        phi[[r, 1]] = coord[r];
        jet[[r, 0, 0]] = 0.0;
        jet[[r, 1, 0]] = 1.0;
    }
    let mut decoder = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        decoder[[1, j]] = dir[j];
    }
    SaeManifoldAtom::new_with_provided_function_gram(
        name.to_string(),
        SaeAtomBasisKind::Linear,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(2),
    )
    .unwrap()
}

/// Build a dictionary of four rectified half-atoms `±u, ±v` parking a
/// centered feature in the `(e0, e1)` plane of `R⁴`. When `gaussian` the
/// parked feature is an isotropic 2-D Gaussian (κ ≈ 2, no curved gain);
/// otherwise a constant-radius circle (κ ≈ 1). Each half is gated on the
/// rows where its lobe is positive, so the ± gates are disjoint (the
/// coalescer's precondition) and the two signed axes co-fire on every row.
fn shattered_plane_term(gaussian: bool) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = 600usize;
    let radius = 3.0_f64;
    let u = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let v = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let neg_u = u.mapv(|x| -x);
    let neg_v = v.mapv(|x| -x);
    let mut s = 0xC0FFEE_u64;
    let lcg = |st: &mut u64| -> f64 {
        *st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    // Per-row (x, y) in-plane coordinates the four halves rectify.
    let mut xs = Array1::<f64>::zeros(n);
    let mut ys = Array1::<f64>::zeros(n);
    for r in 0..n {
        if gaussian {
            // Box–Muller isotropic Gaussian.
            let u1 = lcg(&mut s).max(1e-12);
            let u2 = lcg(&mut s);
            let g0 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            let g1 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).sin();
            xs[r] = radius * g0;
            ys[r] = radius * g1;
        } else {
            let th = std::f64::consts::TAU * (r as f64 + 0.5) / n as f64;
            xs[r] = radius * th.cos();
            ys[r] = radius * th.sin();
        }
    }
    // Rectified coordinates per half.
    let cu: Array1<f64> = xs.mapv(|x| x.max(0.0));
    let cnu: Array1<f64> = xs.mapv(|x| (-x).max(0.0));
    let cv: Array1<f64> = ys.mapv(|y| y.max(0.0));
    let cnv: Array1<f64> = ys.mapv(|y| (-y).max(0.0));
    let atoms = vec![
        linear_line_atom("half_+u", &cu, &u),
        linear_line_atom("half_-u", &cnu, &neg_u),
        linear_line_atom("half_+v", &cv, &v),
        linear_line_atom("half_-v", &cnv, &neg_v),
    ];
    let coord_blocks = vec![
        cu.clone().insert_axis(ndarray::Axis(1)),
        cnu.clone().insert_axis(ndarray::Axis(1)),
        cv.clone().insert_axis(ndarray::Axis(1)),
        cnv.clone().insert_axis(ndarray::Axis(1)),
    ];
    let k = atoms.len();
    // Gate each half on the rows where its lobe is active (coordinate > 0).
    let lobes = [&cu, &cnu, &cv, &cnv];
    let mut logits = Array2::<f64>::zeros((n, k));
    for r in 0..n {
        for (a, lobe) in lobes.iter().enumerate() {
            logits[[r, a]] = if lobe[r] > 1e-9 { ON } else { OFF };
        }
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        vec![LatentManifold::Euclidean; k],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
    (term, rho)
}

/// KILLER DEMO — the curl proposer recovers a centered circle a linear
/// dictionary shattered into four rectified halves: it coalesces the ±
/// pairs, reads κ ≈ 1 off the joint amplitude law, recommends the
/// promotion, and the seed born through the existing plumbing reconstructs
/// the planted ring.
#[test]
fn curl_recovers_shattered_centered_circle() {
    let (term, rho) = shattered_plane_term(false);
    let residuals = residuals_of(&term);
    let cfg = CurlConfig::default();
    let cands = curl_candidates(&term, residuals.view(), &cfg).unwrap();
    assert!(
        !cands.is_empty(),
        "curl must recover the shattered circle (got no candidate)"
    );
    let cand = &cands[0];
    // All four rectified halves coalesced into the two signed axes.
    let mut members = cand.members.clone();
    members.sort_unstable();
    members.dedup();
    assert_eq!(
        members,
        vec![0, 1, 2, 3],
        "the circle's donor set is all four rectified halves"
    );
    assert!(
        cand.net_evidence > 0.0,
        "net evidence must favour the circle"
    );

    // The typed seed is an atomic row-level contract. A truncated gate is
    // schema corruption, not evidence that the omitted rows are absent.
    let mut malformed_seed = cand.seed.clone();
    let BirthSeed::Circle { gate, .. } = &mut malformed_seed else {
        panic!("curl candidate must carry typed circle state");
    };
    gate.pop();
    let malformed_error = apply_structure_move_seeded(
        &term,
        &rho,
        &StructureMove::Birth { candidate: 0 },
        &[malformed_seed],
    )
    .err()
    .expect("a truncated circle gate must be rejected");
    assert!(
        malformed_error.contains("one entry per row"),
        "unexpected truncated-gate error: {malformed_error}"
    );

    // Born through the existing birth plumbing → a Periodic circle atom.
    let mv = StructureMove::Birth { candidate: 0 };
    let seeds = vec![cand.seed.clone()];
    let (born, _born_rho) = apply_structure_move_seeded(&term, &rho, &mv, &seeds).unwrap();
    let circle = born.k_atoms() - 1;
    assert_eq!(
        born.atoms[circle].basis_kind(),
        &SaeAtomBasisKind::Periodic,
        "curl births a Periodic (circle) atom"
    );
    // The born circle's own reconstruction traces the planted ring: every
    // active row sits at radius ≈ R about the centre.
    let img = atom_ambient_image(&born.atoms[circle]);
    let ncols = img.ncols();
    let mut center = Array1::<f64>::zeros(ncols);
    for r in 0..img.nrows() {
        for j in 0..ncols {
            center[j] += img[[r, j]];
        }
    }
    center.mapv_inplace(|x| x / img.nrows() as f64);
    let mut min_r = f64::INFINITY;
    let mut max_r = 0.0_f64;
    for r in 0..img.nrows() {
        let mut rr = 0.0_f64;
        for j in 0..ncols {
            let d = img[[r, j]] - center[j];
            rr += d * d;
        }
        let rr = rr.sqrt();
        min_r = min_r.min(rr);
        max_r = max_r.max(rr);
    }
    // A ring: radius nearly constant across rows (thickness ≪ radius).
    assert!(
        max_r > 0.0 && (max_r - min_r) / max_r < 0.1,
        "born circle must trace a constant-radius ring (min={min_r:.3}, max={max_r:.3})"
    );
}

/// A Gaussian-fill plane (κ ≈ 2, the zero-gain point of the coding law) is
/// NOT curled — the radius law is exactly the flat-parse null.
#[test]
fn curl_rejects_gaussian_fill_plane() {
    let (term, _rho) = shattered_plane_term(true);
    let residuals = residuals_of(&term);
    let cfg = CurlConfig::default();
    let cands = curl_candidates(&term, residuals.view(), &cfg).unwrap();
    assert!(
        cands.is_empty(),
        "a Gaussian-fill plane must not be curled (κ ≈ 2)"
    );
}

/// Build a single-Periodic-atom term whose phase coordinate takes the given
/// per-row turns; the fundamental decoder places the ring in the `(e0, e1)`
/// plane at radius `R`.
fn single_circle_term(phase_turns: &Array1<f64>) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = phase_turns.len();
    let p = 4usize;
    let radius = 3.0_f64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = phase_turns.clone().insert_axis(ndarray::Axis(1));
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[2, 0]] = radius; // cos₁ · e0
    decoder[[1, 1]] = radius; // sin₁ · e1
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let logits = Array2::<f64>::from_elem((n, 1), ON);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

/// A circle whose angular mass has collapsed to a diameter (phases at 0 and
/// ½ turn only) is flagged for flattening; a healthy full-coverage ring is
/// not.
#[test]
fn flatten_flags_diameter_and_spares_healthy_ring() {
    let n = 400usize;
    // Diameter: phases alternate 0 / ½ turn → angles {0, π}.
    let diameter_phases = Array1::from_shape_fn(n, |r| if r % 2 == 0 { 0.0 } else { 0.5 });
    let (diam_term, _) = single_circle_term(&diameter_phases);
    let flagged = flatten_candidates(&diam_term);
    assert_eq!(flagged, vec![0], "a diameter-collapsed circle must flatten");

    // Healthy ring: full angular coverage.
    let ring_phases = Array1::from_shape_fn(n, |r| r as f64 / n as f64);
    let (ring_term, _) = single_circle_term(&ring_phases);
    let flagged = flatten_candidates(&ring_term);
    assert!(
        flagged.is_empty(),
        "a healthy full-coverage ring must NOT be flattened"
    );
}

/// KILLER DEMO — end-to-end through the round driver: with curl ON the
/// shattered centered circle yields a circle Birth that certifies through
/// the same e-gate the residual births race through; with curl OFF (the
/// default) the driver is unchanged. The paired null checks pin the
/// structural boundary: Gaussian fill is not curled, and a diameter
/// collapse flattens to rank 1.
#[test]
fn curl_killer_demo_planted_circle_wins_race() {
    let (term, _rho) = shattered_plane_term(false);
    let residuals = residuals_of(&term);
    let cands = curl_candidates(&term, residuals.view(), &CurlConfig::default()).unwrap();
    assert!(
        !cands.is_empty(),
        "curl must recover the shattered circle before the race"
    );
    let mut members = cands[0].members.clone();
    members.sort_unstable();
    members.dedup();
    assert_eq!(
        members,
        vec![0, 1, 2, 3],
        "the recovered circle must claim all four rectified halves"
    );

    let budget = MoveBudget {
        max_moves: 4,
        alpha: 0.05,
    };
    let harvest_params = HarvestParams {
        max_fusions: 0,
        max_fissions: 0,
        max_births: 0,
    };
    let run = |curl: Option<CurlConfig>| -> StructureSearchResult {
        let (term, rho) = shattered_plane_term(false);
        let target = 2.0 * term.try_fitted().unwrap();
        let mut ledger = StructureLedger::new();
        let config = RoundDriverConfig {
            n_shards: 3,
            budget,
            harvest_params,
            curl,
        };
        run_structure_search_rounds(
            term,
            rho,
            target.view(),
            config,
            &mut ledger,
            |t: SaeManifoldTerm, r: SaeManifoldRho, _rows: &[usize]| Ok((t, r)),
            |t: SaeManifoldTerm, r: SaeManifoldRho, _rows: &[usize]| Ok((t, r)),
            |t: SaeManifoldTerm, r: SaeManifoldRho, _rows: &[usize]| Ok((t, r)),
        )
        .unwrap()
    };

    let off = run(None);
    let off_births = off
        .rounds
        .iter()
        .flat_map(|r| r.moves.iter())
        .filter(|m| matches!(m.mv, StructureMove::Birth { .. }))
        .count();
    assert_eq!(off_births, 0, "curl OFF (default) must inject no births");

    let on = run(Some(CurlConfig::default()));
    let accepted_curl_births = on
        .rounds
        .iter()
        .flat_map(|r| r.moves.iter())
        .filter(|m| {
            matches!(m.mv, StructureMove::Birth { .. })
                && matches!(
                    m.verdict,
                    gam_solve::structure_search::MoveVerdict::Accepted { .. }
                )
        })
        .count();
    assert_eq!(
        accepted_curl_births, 1,
        "curl ON must certify exactly one circle Birth winner"
    );
    assert_eq!(
        on.term.atoms.last().map(|a| a.basis_kind()),
        Some(&SaeAtomBasisKind::Periodic),
        "the accepted curl winner must be the recovered circle atom"
    );
    assert!(
        on.structure_changed(),
        "accepted curl winner must mutate the returned term"
    );

    let (gauss_term, _) = shattered_plane_term(true);
    let gauss_residuals = residuals_of(&gauss_term);
    let gauss_cands =
        curl_candidates(&gauss_term, gauss_residuals.view(), &CurlConfig::default()).unwrap();
    assert!(
        gauss_cands.is_empty(),
        "a Gaussian-fill plane must not be curled"
    );

    let n = 400usize;
    let radii = Array1::<f64>::from_elem(n, 3.0);
    let angles = Array1::<f64>::from_shape_fn(n, |r| {
        if r % 2 == 0 {
            0.0
        } else {
            std::f64::consts::PI
        }
    });
    let flatten = crate::manifold::flatten_verdict(radii.view(), angles.view()).unwrap();
    assert!(flatten.recommend_flatten, "diameter must flatten");
    assert_eq!(flatten.residual_rank, 1, "diameter must flatten to rank 1");
}