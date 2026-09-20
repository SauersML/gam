#![cfg(test)]
//! The thin-plate side-condition frame `Z = null(P(C)ᵀ)` must be the same
//! orthonormal basis whether a caller hands in raw or mean-centered centers.
//! The dense builder factors knot-mean-centered knots and freezes its radial
//! reparam `V` in that frame; the n-free penalty re-key, the ψ / log-κ
//! derivatives and the lazy design all rebuild `Z` from the RAW frozen centers
//! and apply that `V`. With the RRQR pivots depending on the coordinate
//! location, an offset domain rotated `Z` under the frozen `V`.
use crate::basis::{
    BasisMetadata, BasisWorkspace, CenterStrategy, SpatialIdentifiability, ThinPlateBasisSpec,
    build_thin_plate_basiswithworkspace, mean_centered_centers,
    thin_plate_kernel_constraint_nullspace, thin_plate_penalties_at_length_scale,
};
use ndarray::Array2;

/// Deterministic scattered 2-D cloud on `offset + scale·[0, 1)²`.
fn scattered_cloud(n: usize, offset: f64, scale: f64) -> Array2<f64> {
    let mut values = Vec::with_capacity(2 * n);
    for i in 0..n {
        let u = ((i as f64) * 0.618_033_988_749_894_9).fract();
        let v = ((i as f64) * 0.754_877_666_246_692_7 + 0.5).fract();
        values.push(offset + scale * u);
        values.push(offset + scale * v);
    }
    Array2::from_shape_vec((n, 2), values).expect("shape")
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "shape mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn thin_plate_constraint_frame_is_translation_invariant() {
    for &(offset, scale) in &[(0.0, 1.0), (0.0, 10.0), (100.0, 1.0), (-37.5, 4.0)] {
        let centers = scattered_cloud(20, offset, scale);
        let centered = mean_centered_centers(centers.view());
        let mut ws_raw = BasisWorkspace::default();
        let mut ws_centered = BasisWorkspace::default();
        let z_raw = thin_plate_kernel_constraint_nullspace(centers.view(), &mut ws_raw.cache)
            .expect("raw-center frame");
        let z_centered =
            thin_plate_kernel_constraint_nullspace(centered.view(), &mut ws_centered.cache)
                .expect("centered-center frame");
        let diff = max_abs_diff(&z_raw, &z_centered);
        assert!(
            diff <= 1e-9,
            "offset={offset} scale={scale}: raw and centered centers must give the same Z, \
             max |ΔZ| = {diff:e}"
        );
    }
}

#[test]
fn thin_plate_n_free_rekey_matches_cold_penalty_on_offset_domain() {
    // Offset and wide domains are where the RRQR pivot order of the raw
    // `[1, x, y]` block differs from the centered one.
    for &(offset, scale) in &[(100.0, 1.0), (0.0, 10.0)] {
        let data = scattered_cloud(60, offset, scale);
        let length_scale = scale;
        let spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 14 },
            periodic: None,
            length_scale,
            identifiability: SpatialIdentifiability::OrthogonalToParametric,
            double_penalty: false,
            radial_reparam: None,
        };
        let mut workspace = BasisWorkspace::default();
        let cold = build_thin_plate_basiswithworkspace(data.view(), &spec, &mut workspace)
            .expect("cold thin-plate build");
        let BasisMetadata::ThinPlate {
            centers,
            identifiability_transform,
            radial_reparam,
            ..
        } = &cold.metadata
        else {
            panic!("thin-plate build must emit ThinPlate metadata");
        };
        let mut rekey_workspace = BasisWorkspace::default();
        let (rekeyed, _) = thin_plate_penalties_at_length_scale(
            centers.view(),
            identifiability_transform.as_ref(),
            radial_reparam.as_ref(),
            length_scale,
            false,
            &mut rekey_workspace,
        )
        .expect("n-free thin-plate re-key");
        assert_eq!(
            rekeyed.len(),
            cold.active_penalties.len(),
            "offset={offset} scale={scale}: re-key penalty count"
        );
        for (index, (fresh, built)) in rekeyed.iter().zip(cold.active_penalties.iter()).enumerate()
        {
            let peak = built
                .matrix
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()))
                .max(f64::MIN_POSITIVE);
            let diff = max_abs_diff(fresh, &built.matrix) / peak;
            assert!(
                diff <= 1e-8,
                "offset={offset} scale={scale}: re-keyed penalty {index} at the fitted length \
                 scale must equal the cold penalty, relative max |ΔS| = {diff:e}"
            );
        }
    }
}
