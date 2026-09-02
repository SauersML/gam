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

fn forward_design(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> Array2<f64> {
    build_duchon_basis(data, spec)
        .expect("forward design at ψ")
        .design
        .to_dense()
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

