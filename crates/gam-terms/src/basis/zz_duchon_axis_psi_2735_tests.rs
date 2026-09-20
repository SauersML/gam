//! gam#2735 — the Duchon per-axis ψ derivative surface.
//!
//! The isotropic Duchon ψ = log κ derivative is already gated against a central
//! difference of the rebuilt penalty elsewhere in this crate. These tests pin
//! the per-axis surface against **that** rather than re-deriving it, using the
//! identity the construction is built on:
//!
//! ```text
//!     Σ_a ∂S/∂ψ_a = ∂S/∂ψ
//! ```
//!
//! — moving every raw per-axis coordinate by the same amount leaves every
//! contrast fixed and multiplies κ, so the all-ones direction of the per-axis
//! frame IS the isotropic direction. If any part of the per-axis chain is
//! wrong — the `σ_a` bookkeeping, the `δ/d` prefactor lift, the explicit
//! metric-weight factors in the operator blocks, or the collision branch —
//! that sum stops matching a derivative nobody is allowed to change.
//!
//! The absolute scale is then pinned separately by differencing the value.

#![cfg(test)]

use ndarray::{Array2, ArrayView2};

use super::*;

/// A frozen 3-D hybrid Duchon chart with an explicitly anisotropic η.
///
/// The chart (centers, radial reparam, identifiability transform) is frozen off
/// a cold build so nothing but `(ℓ, η)` moves when ψ does — the same discipline
/// the isotropic FD fixtures use, and the reason the #2444/#2445 frame-motion
/// residue cannot leak into these numbers.
fn frozen_aniso_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let n = 60usize;
    let dim = 3usize;
    let mut data = Array2::<f64>::zeros((n, dim));
    // A deterministic, non-symmetric, non-degenerate cloud: no two axes share a
    // spread, so a per-axis derivative that silently symmetrizes is visible.
    for i in 0..n {
        let u = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = (3.0 * u).sin();
        data[[i, 1]] = 0.6 * (5.0 * u + 0.4).cos();
        data[[i, 2]] = 1.4 * u - 0.7 + 0.15 * (11.0 * u).sin();
    }
    let mut spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 9 },
        length_scale: Some(crate::basis::MaternLengthScale::fixed(0.8)),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        // Explicitly non-zero so `auto_seed_aniso_contrasts` honours it
        // verbatim instead of replacing it with the knot-cloud seed.
        aniso_log_scales: Some(vec![0.35, -0.10, -0.25]),
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let base = build_duchon_basis(data.view(), &spec).expect("cold base build");
    if let BasisMetadata::Duchon {
        centers,
        identifiability_transform,
        radial_reparam,
        aniso_log_scales,
        ..
    } = &base.metadata
    {
        spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
        spec.radial_reparam = radial_reparam.clone();
        spec.aniso_log_scales = aniso_log_scales.clone();
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

/// Move the RAW per-axis coordinate `ψ_a` by `h` and re-encode the spec.
///
/// `ψ_b = ψ̄ + η_b` with `ψ̄ = −ln ℓ` and `Σ η = 0`, so the decode after a shift
/// on one axis is `ℓ' = exp(−(ψ̄ + h/d))` and `η'_b = η_b + h(δ_ab − 1/d)`.
/// Both halves move: a test that perturbed only η would be differentiating a
/// different coordinate than the optimizer owns.
fn shift_raw_psi(spec: &DuchonBasisSpec, axis: usize, h: f64) -> DuchonBasisSpec {
    let eta = spec
        .aniso_log_scales
        .clone()
        .expect("fixture is anisotropic");
    let dim = eta.len();
    let inv_d = 1.0 / dim as f64;
    let psi_bar = -spec.length_scale.and_then(|scale| scale.resolved()).expect("hybrid fixture").ln();
    let mut out = spec.clone();
    out.length_scale = Some((-(psi_bar + h * inv_d)).exp());
    out.aniso_log_scales = Some(
        eta.iter()
            .enumerate()
            .map(|(b, &value)| value + h * (f64::from(b == axis) - inv_d))
            .collect(),
    );
    out
}

fn frozen_chart(centers: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> Array2<f64> {
    let mut workspace = BasisWorkspace::default();
    let order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let mut z = kernel_constraint_nullspace(centers, order, &mut workspace.cache)
        .expect("kernel constraint nullspace");
    if let Some(v) = spec.radial_reparam.as_ref() {
        z = z.dot(v);
    }
    z
}

fn frozen_transform(spec: &DuchonBasisSpec) -> Option<Array2<f64>> {
    match &spec.identifiability {
        SpatialIdentifiability::FrozenTransform { transform } => Some(transform.clone()),
        _ => None,
    }
}

fn relative_gap(analytic: &Array2<f64>, reference: &Array2<f64>) -> f64 {
    assert_eq!(analytic.dim(), reference.dim());
    let diff = (analytic - reference)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let scale = reference
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-12);
    diff / scale
}

fn axis_directions(dim: usize) -> Vec<DuchonPsiDirection> {
    (0..dim).map(DuchonPsiDirection::Axis).collect()
}

/// The core algebra, with no fixture: the per-axis jet contracts to the
/// isotropic one along the all-ones direction, for both derivative orders.
#[test]
fn duchon_axis_psi_jet_contracts_to_the_isotropic_jet_2735() {
    // Deterministic pseudo-random probes over exponent, radius, and a share
    // vector that is genuinely uneven (a uniform σ would not detect a formula
    // that used `1/d` where it should use `σ_a`).
    let cases: [(f64, f64, f64, f64, usize); 5] = [
        (1.7, -0.9, 0.35, 0.7, 3),
        (-2.3, 4.1, -1.25, 1.9, 4),
        (0.5, 0.0, 0.0, 0.25, 2),
        (11.0, -3.5, 8.75, 2.4, 6),
        (-0.75, 1.1, -0.4, 1.0, 5),
    ];
    for (value, radial_first, radialsecond, r, dim) in cases {
        for exponent in [-4.0_f64, -1.0, 0.0, 2.0, 5.5] {
            // Shares that sum to one but are far from uniform.
            let raw: Vec<f64> = (0..dim).map(|a| 1.0 + a as f64 * 1.7).collect();
            let total: f64 = raw.iter().sum();
            let shares: Vec<f64> = raw.iter().map(|v| v / total).collect();

            let (global_first, global_second) =
                scaled_log_kappa_derivatives(value, radial_first, radialsecond, exponent, r);

            let mut summed_first = 0.0;
            let mut summed_second = 0.0;
            for a in 0..dim {
                for b in 0..dim {
                    let (first, second) = duchon_axis_log_kappa_derivatives(
                        value,
                        radial_first,
                        radialsecond,
                        exponent,
                        r,
                        dim,
                        shares[a],
                        shares[b],
                        a == b,
                    );
                    if b == 0 {
                        summed_first += first;
                    }
                    summed_second += second;
                }
            }

            let first_gap = (summed_first - global_first).abs();
            let second_gap = (summed_second - global_second).abs();
            assert!(
                first_gap <= 1e-10 * global_first.abs().max(1.0),
                "Σ_a ∂F/∂ψ_a != ∂F/∂ψ: {summed_first} vs {global_first} \
                 (exponent={exponent}, r={r}, dim={dim})"
            );
            assert!(
                second_gap <= 1e-10 * global_second.abs().max(1.0),
                "Σ_ab ∂²F/∂ψ_a∂ψ_b != ∂²F/∂ψ²: {summed_second} vs {global_second} \
                 (exponent={exponent}, r={r}, dim={dim})"
            );
        }
    }
}

/// The share convention keeps the contraction identity at collision, where
/// `σ` is otherwise `0/0`.
#[test]
fn duchon_axis_shares_are_symmetric_at_collision_and_sum_to_one_2735() {
    let shares = duchon_axis_shares(&[0.0, 0.0, 0.0], 0.0);
    assert_eq!(shares.len(), 3);
    for &share in &shares {
        assert!((share - 1.0 / 3.0).abs() < 1e-15);
    }
    let components = [0.25, 0.75, 1.0];
    let r = components.iter().sum::<f64>().sqrt();
    let shares = duchon_axis_shares(&components, r);
    assert!((shares.iter().sum::<f64>() - 1.0).abs() < 1e-14);
}

/// The low-dimensional finite-part representative has an algebraic ψ jet at
/// an exact data/center collision. This is the row topology farthest-point
/// centers guarantee: every selected center is also a data row. The ordinary
/// geometric carrier has `s_a = 0` there and therefore cannot represent the
/// shipped jet.
#[test]
fn duchon_direct_pfd_axis_collision_uses_exact_algebraic_carrier_2735() {
    let length_scale = 0.8_f64;
    let p_order = 2usize;
    let s_order = 1usize;
    let dim = 2usize;
    assert!(
        !duchon_hybrid_stable_integral_applies(p_order, s_order, dim),
        "the discriminator must exercise the direct partial-fraction route"
    );
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let radial = RadialScalarKind::Duchon {
        length_scale,
        p_order,
        s_order,
        dim,
        coeffs: coeffs.clone(),
    };
    let (value, global_first, global_second) =
        duchon_partial_fraction_kernel_psi_triplet(
            0.0,
            length_scale,
            p_order,
            s_order,
            dim,
            &coeffs,
        )
        .expect("direct collision ψ jet");
    let (encoded_value, encoded_q, encoded_t, marked) = radial
        .eval_per_axis_psi_carriers(0.0)
        .expect("per-axis collision carrier");
    assert!(marked, "the direct-PFD collision must use the algebraic carrier");
    assert_eq!(encoded_value, value);

    let c = duchon_scaling_exponent(p_order, s_order, dim) / dim as f64;
    let first = ImplicitDesignPsiDerivative::first_kernel_value(
        1.0,
        encoded_value,
        encoded_q,
        ALGEBRAIC_PER_AXIS_COMPONENT,
        c,
    );
    let second = ImplicitDesignPsiDerivative::second_kernel_value(
        1.0,
        encoded_value,
        encoded_q,
        encoded_t,
        ALGEBRAIC_PER_AXIS_COMPONENT,
        ALGEBRAIC_PER_AXIS_COMPONENT,
        0.0,
        c,
        c,
        0.0,
    );
    let expected_first = global_first / dim as f64;
    let expected_second = global_second / (dim * dim) as f64;
    assert!(
        (first - expected_first).abs() <= 1e-13 * expected_first.abs().max(1.0),
        "raw-axis collision first jet {first} != exact global share {expected_first}"
    );
    assert!(
        (second - expected_second).abs() <= 1e-13 * expected_second.abs().max(1.0),
        "raw-axis collision second jet {second} != exact global share {expected_second}"
    );

    // Prove the row is discriminating: the former homogeneous shortcut reads
    // only `c*K` because its geometric component is zero, and is not the
    // derivative of this shipped finite-part representative.
    let homogeneous_first = c * value;
    assert!(
        (homogeneous_first - expected_first).abs()
            > 1e-6 * expected_first.abs().max(value.abs()).max(1.0),
        "fixture does not distinguish the direct collision jet from c*K"
    );
}

/// Native penalty: the per-axis first derivatives sum to the isotropic one,
/// block by block.
#[test]
fn duchon_native_penalty_axis_psi_sums_to_the_isotropic_derivative_2735() {
    let (_, spec) = frozen_aniso_fixture();
    let centers = fixture_centers(&spec);
    let dim = centers.ncols();
    let transform = frozen_transform(&spec);
    let mut workspace = BasisWorkspace::default();

    let mut directions = vec![DuchonPsiDirection::Global];
    directions.extend(axis_directions(dim));
    let bundles = build_duchon_native_penalty_psi_derivatives_in_directions(
        centers.view(),
        &spec,
        transform.as_ref(),
        &mut workspace,
        &directions,
    )
    .expect("native per-direction derivatives");

    let (sources, global_first, _) = &bundles[0];
    assert!(
        !sources.is_empty(),
        "the fixture must produce at least one native penalty block"
    );
    for (block, global) in global_first.iter().enumerate() {
        let mut summed = Array2::<f64>::zeros(global.raw_dim());
        for axis in 0..dim {
            summed += &bundles[1 + axis].1[block];
        }
        let gap = relative_gap(&summed, global);
        assert!(
            gap < 1e-9,
            "native block {block} ({:?}): Σ_a ∂S/∂ψ_a differs from ∂S/∂ψ by {gap:.3e}",
            sources[block]
        );
    }
}

/// Native penalty: the per-axis first derivative IS the derivative of the
/// shipped value block along the raw ψ_a coordinate.
#[test]
fn duchon_native_penalty_axis_psi_matches_a_central_difference_of_the_value_2735() {
    let (_, spec) = frozen_aniso_fixture();
    let centers = fixture_centers(&spec);
    let dim = centers.ncols();
    let transform = frozen_transform(&spec);
    let z = frozen_chart(centers.view(), &spec);
    let order = duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let mut workspace = BasisWorkspace::default();

    let analytic = build_duchon_native_penalty_psi_derivatives_in_directions(
        centers.view(),
        &spec,
        transform.as_ref(),
        &mut workspace,
        &axis_directions(dim),
    )
    .expect("native per-axis derivatives");

    let value_blocks = |local: &DuchonBasisSpec| -> Vec<Array2<f64>> {
        let candidates = duchon_native_penalty_candidates(
            centers.view(),
            local.length_scale,
            local.power,
            order,
            local.aniso_log_scales.as_deref(),
            &z,
            transform.as_ref(),
        )
        .expect("native penalty candidates");
        filter_penalty_candidates(candidates)
            .expect("filter native candidates")
            .active
            .iter()
            .map(|penalty| penalty.matrix.to_owned())
            .collect()
    };

    let eps = 1e-5_f64;
    for axis in 0..dim {
        let plus = value_blocks(&shift_raw_psi(&spec, axis, eps));
        let minus = value_blocks(&shift_raw_psi(&spec, axis, -eps));
        assert_eq!(plus.len(), analytic[axis].1.len());
        for (block, first) in analytic[axis].1.iter().enumerate() {
            let fd = (&plus[block] - &minus[block]) / (2.0 * eps);
            let gap = relative_gap(first, &fd);
            assert!(
                gap < 5e-5,
                "native block {block}, axis {axis}: analytic ∂S/∂ψ_a differs from \
                 the central difference of the shipped value by {gap:.3e}"
            );
        }
    }
}

/// Native penalty: the per-axis second derivative is the derivative of the
/// per-axis first, along the same coordinate.
#[test]
fn duchon_native_penalty_axis_psi_second_matches_a_difference_of_the_first_2735() {
    let (_, spec) = frozen_aniso_fixture();
    let centers = fixture_centers(&spec);
    let dim = centers.ncols();
    let transform = frozen_transform(&spec);
    let mut workspace = BasisWorkspace::default();
    let directions = axis_directions(dim);

    let analytic = build_duchon_native_penalty_psi_derivatives_in_directions(
        centers.view(),
        &spec,
        transform.as_ref(),
        &mut workspace,
        &directions,
    )
    .expect("native per-axis derivatives");

    let eps = 1e-5_f64;
    for axis in 0..dim {
        let mut ws_plus = BasisWorkspace::default();
        let mut ws_minus = BasisWorkspace::default();
        let plus = build_duchon_native_penalty_psi_derivatives_in_directions(
            centers.view(),
            &shift_raw_psi(&spec, axis, eps),
            transform.as_ref(),
            &mut ws_plus,
            &[DuchonPsiDirection::Axis(axis)],
        )
        .expect("plus");
        let minus = build_duchon_native_penalty_psi_derivatives_in_directions(
            centers.view(),
            &shift_raw_psi(&spec, axis, -eps),
            transform.as_ref(),
            &mut ws_minus,
            &[DuchonPsiDirection::Axis(axis)],
        )
        .expect("minus");
        for (block, second) in analytic[axis].2.iter().enumerate() {
            let fd = (&plus[0].1[block] - &minus[0].1[block]) / (2.0 * eps);
            let gap = relative_gap(second, &fd);
            assert!(
                gap < 5e-4,
                "native block {block}, axis {axis}: analytic ∂²S/∂ψ_a² differs from \
                 the central difference of the analytic first by {gap:.3e}"
            );
        }
    }
}

/// Operator penalty: the per-axis first derivatives sum to the isotropic one.
///
/// This is the sharpest single check in the file. The operator blocks are the
/// only place the metric weights `w_b` appear as EXPLICIT factors rather than
/// only inside `r`, so their per-axis derivative carries the `2 w_b (δ_ab −
/// 1/d)` term as well as the radial chain rule — and the sum identity holds
/// only if the `δ/d` prefactor lift and that weight term are both right.
#[test]
fn duchon_operator_penalty_axis_psi_sums_to_the_isotropic_derivative_2735() {
    let (data, spec) = frozen_aniso_fixture();
    let centers = fixture_centers(&spec);
    let dim = centers.ncols();
    let transform = frozen_transform(&spec);
    let mut workspace = BasisWorkspace::default();
    let collocation = select_thin_plate_knots(
        data.view(),
        (DUCHON_COLLOCATION_OVERSAMPLE * centers.nrows()).min(data.nrows()),
    )
    .expect("collocation points");

    let mut directions = vec![DuchonPsiDirection::Global];
    directions.extend(axis_directions(dim));
    let bundles = build_duchon_operator_penalty_psi_derivatives_in_directions(
        collocation.view(),
        centers.view(),
        &spec,
        transform.as_ref(),
        &mut workspace,
        &directions,
    )
    .expect("operator per-direction derivatives");

    let (sources, global_first, _) = &bundles[0];
    assert!(
        !sources.is_empty(),
        "the fixture must produce at least one operator penalty block"
    );
    for (block, global) in global_first.iter().enumerate() {
        let mut summed = Array2::<f64>::zeros(global.raw_dim());
        for axis in 0..dim {
            summed += &bundles[1 + axis].1[block];
        }
        let gap = relative_gap(&summed, global);
        assert!(
            gap < 1e-8,
            "operator block {block} ({:?}): Σ_a ∂S/∂ψ_a differs from ∂S/∂ψ by {gap:.3e}",
            sources[block]
        );
    }
}

/// Operator penalty: the per-axis second derivative is the derivative of the
/// per-axis first.
#[test]
fn duchon_operator_penalty_axis_psi_second_matches_a_difference_of_the_first_2735() {
    let (data, spec) = frozen_aniso_fixture();
    let centers = fixture_centers(&spec);
    let dim = centers.ncols();
    let transform = frozen_transform(&spec);
    let collocation = select_thin_plate_knots(
        data.view(),
        (DUCHON_COLLOCATION_OVERSAMPLE * centers.nrows()).min(data.nrows()),
    )
    .expect("collocation points");
    let mut workspace = BasisWorkspace::default();

    let analytic = build_duchon_operator_penalty_psi_derivatives_in_directions(
        collocation.view(),
        centers.view(),
        &spec,
        transform.as_ref(),
        &mut workspace,
        &axis_directions(dim),
    )
    .expect("operator per-axis derivatives");

    let eps = 1e-5_f64;
    for axis in 0..dim {
        let mut ws_plus = BasisWorkspace::default();
        let mut ws_minus = BasisWorkspace::default();
        let plus = build_duchon_operator_penalty_psi_derivatives_in_directions(
            collocation.view(),
            centers.view(),
            &shift_raw_psi(&spec, axis, eps),
            transform.as_ref(),
            &mut ws_plus,
            &[DuchonPsiDirection::Axis(axis)],
        )
        .expect("plus");
        let minus = build_duchon_operator_penalty_psi_derivatives_in_directions(
            collocation.view(),
            centers.view(),
            &shift_raw_psi(&spec, axis, -eps),
            transform.as_ref(),
            &mut ws_minus,
            &[DuchonPsiDirection::Axis(axis)],
        )
        .expect("minus");
        for (block, second) in analytic[axis].2.iter().enumerate() {
            let fd = (&plus[0].1[block] - &minus[0].1[block]) / (2.0 * eps);
            let gap = relative_gap(second, &fd);
            assert!(
                gap < 5e-4,
                "operator block {block}, axis {axis}: analytic ∂²S/∂ψ_a² differs from \
                 the central difference of the analytic first by {gap:.3e}"
            );
        }
    }
}

/// Operator penalty: the ψ derivative is the derivative of the **shipped**
/// value blocks — the test whose absence let a value/derivative desync live.
///
/// The value side builds its collocation operators with `aniso = None` and, on
/// a `scale_dims` term, REPLACES the single isotropic tension block with `dim`
/// per-axis `OperatorRelevance` blocks. The derivative side used to do neither:
/// it differentiated an anisotropic operator penalty the design never ships,
/// and emitted one tension block against the design's `dim` relevance blocks,
/// which the consumer then zipped positionally. Two independent desyncs, both
/// invisible to a sum-identity or second-vs-first check, because those compare
/// the derivative against ITSELF.
///
/// This differences the value.
#[test]
fn duchon_operator_penalty_psi_matches_a_central_difference_of_the_value_2735() {
    let (data, spec) = frozen_aniso_fixture();
    let centers = fixture_centers(&spec);
    let dim = centers.ncols();
    let transform = frozen_transform(&spec);
    let order = duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let collocation = select_thin_plate_knots(
        data.view(),
        (DUCHON_COLLOCATION_OVERSAMPLE * centers.nrows()).min(data.nrows()),
    )
    .expect("collocation points");

    let mut directions = vec![DuchonPsiDirection::Global];
    directions.extend(axis_directions(dim));
    let mut workspace = BasisWorkspace::default();
    let analytic = build_duchon_operator_penalty_psi_derivatives_in_directions(
        collocation.view(),
        centers.view(),
        &spec,
        transform.as_ref(),
        &mut workspace,
        &directions,
    )
    .expect("operator per-direction derivatives");

    // The value, through the SAME entry the realized design uses.
    let value_blocks = |local: &DuchonBasisSpec| -> (Vec<PenaltySource>, Vec<Array2<f64>>) {
        let mut ws = BasisWorkspace::default();
        let candidates = duchon_operator_penalty_candidates(
            collocation.view(),
            centers.view(),
            &local.operator_penalties,
            local.length_scale,
            local.power,
            order,
            local.aniso_log_scales.is_some(),
            transform.as_ref(),
            local.radial_reparam.as_ref(),
            &mut ws,
        )
        .expect("operator penalty candidates");
        let filtered = filter_penalty_candidates(candidates).expect("filter operator candidates");
        (
            filtered
                .active
                .iter()
                .map(|penalty| penalty.info.source.clone())
                .collect(),
            filtered
                .active
                .iter()
                .map(|penalty| penalty.matrix.to_owned())
                .collect(),
        )
    };

    let (value_sources, _) = value_blocks(&spec);
    assert_eq!(
        analytic[0].0, value_sources,
        "the derivative's active penalty SOURCES must be the design's, in order"
    );
    assert!(
        value_sources
            .iter()
            .any(|source| matches!(source, PenaltySource::OperatorRelevance { .. })),
        "the fixture must exercise the per-axis relevance split: got {value_sources:?}"
    );

    let eps = 1e-5_f64;
    // Global: ψ = log κ, so ℓ ↦ ℓ·e^{∓h}.
    {
        let mut plus = spec.clone();
        plus.length_scale = Some(spec.length_scale.unwrap() * (-eps).exp());
        let mut minus = spec.clone();
        minus.length_scale = Some(spec.length_scale.unwrap() * eps.exp());
        let (_, blocks_plus) = value_blocks(&plus);
        let (_, blocks_minus) = value_blocks(&minus);
        for (block, first) in analytic[0].1.iter().enumerate() {
            let fd = (&blocks_plus[block] - &blocks_minus[block]) / (2.0 * eps);
            let gap = relative_gap(first, &fd);
            assert!(
                gap < 5e-5,
                "operator block {block} ({:?}): analytic ∂S/∂ψ differs from the central \
                 difference of the shipped value by {gap:.3e}",
                value_sources[block]
            );
        }
    }
    // Per axis: the raw ψ_a moves ℓ at rate 1/d and every contrast, and the
    // operator penalty ignores the contrasts — so the FD must come out at 1/d
    // of the isotropic one, which is what the analytic claims.
    for axis in 0..dim {
        let (_, blocks_plus) = value_blocks(&shift_raw_psi(&spec, axis, eps));
        let (_, blocks_minus) = value_blocks(&shift_raw_psi(&spec, axis, -eps));
        for (block, first) in analytic[1 + axis].1.iter().enumerate() {
            let fd = (&blocks_plus[block] - &blocks_minus[block]) / (2.0 * eps);
            let gap = relative_gap(first, &fd);
            assert!(
                gap < 5e-5,
                "operator block {block} ({:?}), axis {axis}: analytic ∂S/∂ψ_a differs from \
                 the central difference of the shipped value by {gap:.3e}",
                value_sources[block]
            );
        }
    }
}
