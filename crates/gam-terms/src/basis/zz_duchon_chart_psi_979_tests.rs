//! gam#979 — the hybrid Duchon design ψ-derivative under the kernel chart.
//!
//! In high dimension at a large spectral power the raw hybrid Duchon–Matérn
//! kernel underflows (its spectral normalization is ~1e-15 at `d = 16`,
//! `s = 9`), and the forward basis ships the kernel block multiplied by the
//! chart amplitude `α(ψ) = 1/max|K|` (`duchon_kernel_chart`). The design the
//! REML criterion is built on is therefore `α(ψ)·K(ψ)`, and its ψ-derivative
//! is `α(K_ψ + (ln α)_ψ K)`, not `K_ψ`. Before this gate existed, the
//! derivative operator formed `K_ψ` alone: at the large-scale benchmark's
//! `duchon(pc1..pc16, order=0, power=9, length_scale=1)` that is ~1e-15 of
//! the true derivative, the analytic outer gradient silently dropped every
//! κ-dependence that enters through the design, and the κ line search walked
//! uphill on every trial (the `gam fit --transformation-normal` timeout).
//!
//! The gate differences the FORWARD design the basis actually ships, through
//! its own frozen chart, against the operator's materialized first and second
//! ψ-derivatives. The low-dimensional control has `α = 1` and pins that the
//! chart is inert there; the 16-D fixture asserts `α ≠ 1` so it cannot pass
//! vacuously.

#![cfg(test)]

use ndarray::{Array2, ArrayView2};

use super::*;

/// A deterministic, non-degenerate cloud in `d` dimensions on the ±2 range of
/// a standardized coordinate (distinct irrational multipliers per axis so no
/// two axes alias).
fn standardized_cloud(n: usize, d: usize) -> Array2<f64> {
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for a in 0..d {
            let multiplier = ((a + 2) as f64 * 2.0 + 1.0).sqrt().fract();
            data[[i, a]] = 4.0 * ((i as f64 * multiplier + 0.37 * a as f64).fract() - 0.5);
        }
    }
    data
}

/// A hybrid Duchon spec at `(order, power)` with the chart frozen off one cold
/// build, so nothing but the length scale moves when ψ does — the same
/// discipline `zz_duchon_axis_psi_2735_tests` uses.
fn frozen_hybrid_fixture(
    d: usize,
    n: usize,
    centers: usize,
    order: DuchonNullspaceOrder,
    power: f64,
) -> (Array2<f64>, DuchonBasisSpec) {
    let data = standardized_cloud(n, d);
    let mut spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: centers,
        },
        length_scale: Some(1.0),
        power,
        nullspace_order: order,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let base = build_duchon_basis(data.view(), &spec).expect("cold base build");
    if let BasisMetadata::Duchon {
        centers,
        identifiability_transform,
        radial_reparam,
        ..
    } = &base.metadata
    {
        spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
        spec.radial_reparam = radial_reparam.clone();
        spec.identifiability = match identifiability_transform {
            Some(t) => SpatialIdentifiability::FrozenTransform {
                transform: t.clone(),
            },
            None => SpatialIdentifiability::None,
        };
    } else {
        panic!("expected Duchon metadata");
    }
    (data, spec)
}

fn fixture_centers(spec: &DuchonBasisSpec) -> Array2<f64> {
    match &spec.center_strategy {
        CenterStrategy::UserProvided(c) => c.clone(),
        _ => unreachable!("fixture freezes the centers"),
    }
}

/// The frozen spec with the isotropic coordinate moved to `ψ`: `ℓ = e^{−ψ}`.
fn spec_at_psi(spec: &DuchonBasisSpec, psi: f64) -> DuchonBasisSpec {
    let mut out = spec.clone();
    out.length_scale = Some((-psi).exp());
    out
}

fn frobenius(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn chart_amplification(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> f64 {
    let centers = fixture_centers(spec);
    let order = duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(order);
    let s_order = spec.power_as_usize();
    let length_scale = spec.length_scale.expect("hybrid fixture");
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    duchon_kernel_chart(
        centers.view(),
        Some(length_scale),
        p_order,
        s_order,
        data.ncols(),
        None,
        Some(&coeffs),
        None,
    )
    .amplification
}

// ---------------------------------------------------------------------------
// The latent-coordinate Jacobian under the same chart.
//
// `LatentCoordDesignDerivative::new_duchon` supplies `∂X/∂t` for the joint
// `[rho, latent]` driver. The shipped design is `α·φ(||t/σ − c||)`, and `α`
// depends on the centers and the range only, so the coordinate Jacobian is
// the raw one times `α`. The ground truth is the production rebuild
// (`build_term_collection_design` through the frozen spec), central-differenced
// in one latent coordinate — the same discipline as the #2643 frame gate.
// ---------------------------------------------------------------------------

use ndarray::s;

// ---------------------------------------------------------------------------
// The operator penalties (mass, tension) under the same chart.
//
// `duchon_operator_penalty_candidates` is the forward; its collocation
// quadratures now carry the chart amplitude `α` like the design does, and
// `build_duchon_operator_penalty_psi_derivatives` mirrors it. The gate
// differences the forward's NORMALIZED penalties along ψ against the analytic
// normalized first jets, per penalty source, at the benchmark shape (α ≫ 1)
// and at the 3-D sibling (α = 1).
// ---------------------------------------------------------------------------

fn fixture_collocation_points(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> Array2<f64> {
    match build_duchon_basis(data, spec).expect("frozen build").metadata {
        BasisMetadata::Duchon {
            operator_collocation_points: Some(points),
            ..
        } => points,
        _ => panic!("hybrid Duchon with operator penalties must realize collocation points"),
    }
}

fn forward_operator_penalties(
    collocation: &Array2<f64>,
    centers: &Array2<f64>,
    spec: &DuchonBasisSpec,
) -> Vec<(String, Array2<f64>)> {
    duchon_operator_penalty_candidates(
        collocation.view(),
        centers.view(),
        &spec.operator_penalties,
        spec.length_scale,
        spec.power,
        spec.nullspace_order,
        false,
        None,
        spec.radial_reparam.as_ref(),
        &mut BasisWorkspace::default(),
    )
    .expect("forward operator penalties")
    .into_iter()
    .map(|candidate| (format!("{:?}", candidate.source), candidate.matrix.dense().clone()))
    .collect()
}

/// The mass penalty rebuilt from first principles beside the worker: amplified
/// kernel values at (collocation, center) pairs through the frozen chart
/// `Z·V`, the polynomial block, column centering, the Gram, and the
/// normalization — with `∂φ/∂ψ = δ φ + r φ_r` from the same radial jets.
/// Returns `(S̃, ∂S̃/∂ψ)`.
fn mass_reconstruction(
    collocation: &Array2<f64>,
    centers: &Array2<f64>,
    spec: &DuchonBasisSpec,
) -> (Array2<f64>, Array2<f64>) {
    let order = duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(order);
    let s_order = spec.power_as_usize();
    let ell = spec.length_scale.expect("hybrid fixture");
    let d = centers.ncols();
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / ell);
    let amp = duchon_kernel_amplification(
        centers.view(),
        Some(ell),
        p_order,
        s_order,
        d,
        None,
        Some(&coeffs),
        None,
    );
    let mut workspace = BasisWorkspace::default();
    let z = duchon_frozen_radial_chart(
        kernel_constraint_nullspace(centers.view(), order, &mut workspace.cache)
            .expect("side-condition null space"),
        spec,
        "mass reconstruction",
    )
    .expect("frozen radial chart");
    let delta = duchon_scaling_exponent(p_order, s_order, d);
    let (m, k) = (collocation.nrows(), centers.nrows());
    let mut raw = Array2::<f64>::zeros((m, k));
    let mut raw_psi = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            let r = (0..d)
                .map(|a| (collocation[[i, a]] - centers[[j, a]]).powi(2))
                .sum::<f64>()
                .sqrt();
            let jets = duchon_radial_jets(r, ell, p_order, s_order, d, &coeffs).expect("jets");
            raw[[i, j]] = amp * jets.phi;
            raw_psi[[i, j]] = amp * (delta * jets.phi + r * jets.phi_r);
        }
    }
    let poly = polynomial_block_from_order(collocation.view(), order);
    let kernel_cols = z.ncols();
    let total = kernel_cols + poly.ncols();
    let mut d0 = Array2::<f64>::zeros((m, total));
    let mut d0_psi = Array2::<f64>::zeros((m, total));
    d0.slice_mut(s![.., ..kernel_cols]).assign(&raw.dot(&z));
    d0.slice_mut(s![.., kernel_cols..]).assign(&poly);
    d0_psi.slice_mut(s![.., ..kernel_cols]).assign(&raw_psi.dot(&z));
    let zeros = Array2::<f64>::zeros((m, total));
    let (s0, s0_psi, _) = centered_operator_gram_and_psi_derivatives(&d0, &d0_psi, &zeros);
    let (s_norm, s_norm_psi, _, _) =
        normalize_penaltywith_psi_derivatives(&s0, &s0_psi, &Array2::<f64>::zeros(s0.raw_dim()));
    (s_norm, s_norm_psi)
}

/// Per source: `(source, |fd|, best relative gap over two steps)`.
fn operator_penalty_gaps(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    label: &str,
) -> Vec<(String, f64, f64)> {
    let collocation = fixture_collocation_points(data, spec);
    let centers = fixture_centers(spec);
    let (sources, firsts, _) = build_duchon_operator_penalty_psi_derivatives(
        collocation.view(),
        centers.view(),
        spec,
        None,
        &mut BasisWorkspace::default(),
    )
    .expect("operator penalty ψ-jets");
    assert!(!sources.is_empty(), "{label}: the fixture must emit operator penalties");
    // Diagnostic (printed, not asserted): where does the mass jet disagree —
    // in the value the two sides build, or in the jet they assemble from it?
    {
        let forward_base = forward_operator_penalties(&collocation, &centers, spec);
        if let (Some((_, s_fwd)), Some(worker_idx)) = (
            forward_base.iter().find(|(n, _)| n == "OperatorMass"),
            sources.iter().position(|s| format!("{s:?}") == "OperatorMass"),
        ) {
            let (s_mine, s_mine_psi) = mass_reconstruction(&collocation, &centers, spec);
            let value_gap = frobenius(&(s_fwd - &s_mine)) / frobenius(s_fwd).max(1e-300);
            // Split the value gap: the builder's own D0 through the same centered
            // Gram + normalization, against the candidate and against the rebuild.
            let ops = build_duchon_collocation_operator_matriceswithworkspace(
                centers.view(),
                collocation.view(),
                None,
                spec.length_scale,
                spec.power,
                spec.nullspace_order,
                None,
                None,
                1,
                spec.radial_reparam.as_ref().map(|v| v.view()),
                &mut BasisWorkspace::default(),
            )
            .expect("forward collocation blocks");
            let (s_builder, _) = normalize_penalty(&symmetrize_penalty(&centered_design_gram(&ops.d0)));
            // Finer split: the builder's D0 without the frozen radial chart
            // against a chart-free rebuild (raw kernel · Z | poly), per block.
            let ops_no_v = build_duchon_collocation_operator_matriceswithworkspace(
                centers.view(),
                collocation.view(),
                None,
                spec.length_scale,
                spec.power,
                spec.nullspace_order,
                None,
                None,
                1,
                None,
                &mut BasisWorkspace::default(),
            )
            .expect("forward collocation blocks without the radial chart");
            {
                let order = duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
                let p_order = duchon_p_from_nullspace_order(order);
                let s_order = spec.power_as_usize();
                let ell = spec.length_scale.expect("hybrid fixture");
                let d = centers.ncols();
                let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / ell);
                let mut workspace = BasisWorkspace::default();
                let z = kernel_constraint_nullspace(centers.view(), order, &mut workspace.cache)
                    .expect("side-condition null space");
                let (m, k) = (collocation.nrows(), centers.nrows());
                let mut raw = Array2::<f64>::zeros((m, k));
                for i in 0..m {
                    for j in 0..k {
                        let r = (0..d)
                            .map(|a| (collocation[[i, a]] - centers[[j, a]]).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        raw[[i, j]] =
                            duchon_radial_jets(r, ell, p_order, s_order, d, &coeffs).expect("jets").phi;
                    }
                }
                let kernel_rebuilt = raw.dot(&z);
                let phi_at = |r: f64| {
                    duchon_radial_jets(r, ell, p_order, s_order, d, &coeffs)
                        .expect("jets")
                        .phi
                };
                let (phi0, phi_eps, phi_floor) = (phi_at(0.0), phi_at(1e-10), phi_at(1e-5));
                let coincident = (0..m)
                    .flat_map(|i| (0..k).map(move |j| (i, j)))
                    .filter(|&(i, j)| {
                        (0..d).all(|a| collocation[[i, a]] == centers[[j, a]])
                    })
                    .count();
                eprintln!(
                    "[{label}] MASS-RECON collision value: phi(0)={phi0:.9e} phi(1e-10)={phi_eps:.9e} \
                     phi(1e-5)={phi_floor:.9e} rel(1e-10 vs 0)={:.3e}; coincident collocation/center pairs={coincident}",
                    (phi_eps - phi0).abs() / phi0.abs().max(1e-300)
                );
                let kc = z.ncols();
                let kernel_builder = ops_no_v.d0.slice(s![.., ..kc]).to_owned();
                let poly_builder = ops_no_v.d0.slice(s![.., kc..]).to_owned();
                let poly_rebuilt = polynomial_block_from_order(collocation.view(), order);
                eprintln!(
                    "[{label}] MASS-RECON no-chart: builder D0 {}x{} (kernel_cols={}, poly_cols={}, kernel_nullspace={:?}); \
                     kernel gap={:.3e} |builder|={:.3e} |rebuilt|={:.3e}; poly gap={:.3e}; raw max|phi|={:.3e}; \
                     V={:?}",
                    ops_no_v.d0.nrows(),
                    ops_no_v.d0.ncols(),
                    kc,
                    ops_no_v.polynomial_block_cols,
                    ops_no_v.kernel_nullspace_transform.as_ref().map(|t| t.dim()),
                    frobenius(&(&kernel_builder - &kernel_rebuilt)) / frobenius(&kernel_builder).max(1e-300),
                    frobenius(&kernel_builder),
                    frobenius(&kernel_rebuilt),
                    frobenius(&(&poly_builder - &poly_rebuilt)) / frobenius(&poly_builder).max(1e-300),
                    raw.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
                    spec.radial_reparam.as_ref().map(|v| v.dim()),
                );
            }
            eprintln!(
                "[{label}] MASS-RECON builder D0 {}x{} amp={:.3e}; gap(candidate vs builder-gram)={:.3e} \
                 gap(rebuilt vs builder-gram)={:.3e} |cand|={:.6e} |builder|={:.6e} |rebuilt|={:.6e}",
                ops.d0.nrows(),
                ops.d0.ncols(),
                ops.kernel_amplification,
                frobenius(&(s_fwd - &s_builder)) / frobenius(s_fwd).max(1e-300),
                frobenius(&(&s_mine - &s_builder)) / frobenius(&s_builder).max(1e-300),
                frobenius(s_fwd),
                frobenius(&s_builder),
                frobenius(&s_mine)
            );
            let jet_gap = frobenius(&(&firsts[worker_idx] - &s_mine_psi))
                / frobenius(&s_mine_psi).max(1e-300);
            eprintln!(
                "[{label}] MASS-RECON value gap(forward vs rebuilt)={value_gap:.3e} \
                 jet gap(worker vs rebuilt)={jet_gap:.3e} |rebuilt jet|={:.6e}",
                frobenius(&s_mine_psi)
            );
        }
    }
    let mut out = Vec::new();
    for (source, analytic) in sources.iter().zip(firsts.iter()) {
        let name = format!("{source:?}");
        let mut best_gap = f64::INFINITY;
        let mut fd_norm = 0.0;
        for &h in &[2.0e-3_f64, 1.0e-3] {
            let plus = forward_operator_penalties(&collocation, &centers, &spec_at_psi(spec, h));
            let minus = forward_operator_penalties(&collocation, &centers, &spec_at_psi(spec, -h));
            let find = |list: &[(String, Array2<f64>)]| {
                list.iter()
                    .find(|(candidate, _)| *candidate == name)
                    .map(|(_, matrix)| matrix.clone())
                    .unwrap_or_else(|| panic!("{label}: forward emits no {name} penalty"))
            };
            let fd = (find(&plus) - find(&minus)) / (2.0 * h);
            let gap = frobenius(&(analytic - &fd)) / frobenius(&fd).max(1e-300);
            eprintln!(
                "[{label}] {name} h={h:.1e} |an|={:.6e} |fd|={:.6e} gap={gap:.3e}",
                frobenius(analytic),
                frobenius(&fd)
            );
            if gap < best_gap {
                best_gap = gap;
                fd_norm = frobenius(&fd);
            }
        }
        out.push((name, fd_norm, best_gap));
    }
    out
}

fn assert_operator_penalty_gaps(gaps: &[(String, f64, f64)], label: &str) {
    for (name, fd_norm, gap) in gaps {
        assert!(
            *fd_norm > 1e-6,
            "{label}: {name} does not move with ψ in this fixture (|fd| = {fd_norm:.3e}), so the gate is vacuous"
        );
        // Mass is a quadrature Gram on both sides and matches to the central
        // difference's own truncation (1e-6). Tension takes the closed-form
        // path where it converges, whose self-pair bundle is ε-regularized;
        // measured 2026-09-01 at 16-D `Linear` power 9: a step-independent
        // 3.76e-4 relative residual (1e-6 at order 0 and at 3-D). The bar
        // below is that measurement's reach, not a bound the closed form is
        // known to meet.
        let bar = if name == "OperatorTension" { 1e-3 } else { 1e-4 };
        assert!(
            *gap < bar,
            "{label}: analytic ∂S̃/∂ψ of {name} differs from the forward's central difference by {gap:.3e} (bar {bar:.0e})"
        );
    }
}

/// The benchmark's chart: mass and tension jets must be those of the shipped
/// (amplified, normalized) penalties. Before this gate they were exactly zero.
#[test]
fn duchon_operator_penalty_psi_jets_match_the_forward_16d_order0_power9() {
    let (data, spec) = frozen_hybrid_fixture(16, 120, 24, DuchonNullspaceOrder::Zero, 9.0);
    assert!(chart_amplification(data.view(), &spec) != 1.0, "the fixture must be amplified");
    let gaps = operator_penalty_gaps(data.view(), &spec, "opers_16d_order0_power9");
    assert_operator_penalty_gaps(&gaps, "opers_16d_order0_power9");
}

#[test]
fn duchon_operator_penalty_psi_jets_match_the_forward_16d_linear_power9() {
    let (data, spec) = frozen_hybrid_fixture(16, 120, 24, DuchonNullspaceOrder::Linear, 9.0);
    assert!(chart_amplification(data.view(), &spec) != 1.0, "the fixture must be amplified");
    let gaps = operator_penalty_gaps(data.view(), &spec, "opers_16d_linear_power9");
    assert_operator_penalty_gaps(&gaps, "opers_16d_linear_power9");
}

/// The un-amplified sibling: the same jets with `α = 1`, so a gap here is a
/// formula gap and not a scale one.
#[test]
fn duchon_operator_penalty_psi_jets_match_the_forward_3d_order0_power9() {
    let (data, spec) = frozen_hybrid_fixture(3, 160, 10, DuchonNullspaceOrder::Zero, 9.0);
    assert_eq!(chart_amplification(data.view(), &spec), 1.0, "3-D order-0 power-9 is not amplified");
    let gaps = operator_penalty_gaps(data.view(), &spec, "opers_3d_order0_power9");
    assert_operator_penalty_gaps(&gaps, "opers_3d_order0_power9");
}
