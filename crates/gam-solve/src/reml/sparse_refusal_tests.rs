use super::*;
use faer::sparse::{SparseColMat, Triplet};
use gam_problem::{GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};
use std::sync::{Arc, atomic::Ordering};

fn with_sparse_fixture(check: impl FnOnce(&mut RemlState<'_>, &Array1<f64>)) {
    const P: usize = 32;
    let n = 2 * P;
    let triplets: Vec<_> = (0..n).map(|row| Triplet::new(row, row % P, 1.0)).collect();
    let x = SparseColMat::try_new_from_triplets(n, P, &triplets).expect("sparse design");
    let y = Array1::from_shape_fn(n, |i| (i % P) as f64 / 8.0 + if i < P { -0.4 } else { 0.6 });
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    ));
    let config = RemlConfig::external(likelihood, 1e-8, false);
    let specs = [crate::estimate::PenaltySpec::from_blockwise(
        gam_terms::smooth::BlockwisePenalty::new(0..P, Array2::eye(P)),
    )];
    let (penalties, _) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &[0],
        P,
        "sparse refusal regression",
    )
    .expect("penalty");
    let mut state = RemlState::newwith_offset(
        y.view(),
        x,
        weights.view(),
        offset.view(),
        penalties,
        P,
        &config,
        Some(vec![0]),
        None,
        None,
    )
    .expect("state");
    check(&mut state, &y);
}

#[test]
fn sparse_gaussian_bundle_matches_diagonal_and_dense_references() {
    with_sparse_fixture(|state, y| {
        let rho = array![0.0];
        let sparse = state
            .prepare_eval_bundlewithkey(&rho, None)
            .expect("sparse bundle");
        assert!(matches!(
            sparse.geometry.geometry,
            RemlGeometry::SparseExactSpd
        ));
        assert!(matches!(
            sparse.pirls_result.coordinate_frame,
            crate::pirls::PirlsCoordinateFrame::OriginalSparseNative
        ));
        let exact = sparse.sparse_exact.as_ref().expect("sparse factor");
        assert!((exact.logdet_h - 32.0 * 3.0_f64.ln()).abs() < 1e-10);
        for i in 0..32 {
            let expected = (y[i] + y[i + 32]) / 3.0;
            assert!((sparse.pirls_result.beta_transformed.as_ref()[i] - expected).abs() < 1e-10);
        }
        let mut decision = sparse.geometry.clone();
        decision.geometry = RemlGeometry::DenseSpectral;
        let dense = state
            .prepare_dense_eval_bundlewithkey(&rho, None, BundleRows::Observed, decision)
            .expect("dense reference");
        for i in 0..32 {
            for j in 0..32 {
                let expected = if i == j { 3.0 } else { 0.0 };
                assert!((dense.h_total[[i, j]] - expected).abs() < 1e-10);
            }
        }
    });
}

#[test]
fn sparse_hessian_refusal_is_not_replaced_by_dense_bundle() {
    with_sparse_fixture(|state, _| {
        let rho = array![0.0];
        let good = state
            .prepare_eval_bundlewithkey(&rho, None)
            .expect("valid sparse mode");
        let mut corrupt = good.pirls_result.as_ref().clone();
        // Inject a concrete row-geometry failure after a certified inner solve.
        // The cached positive Hessian is intact, so the former catch-all path
        // could build a dense bundle and erase this error.
        corrupt.finalweights[3] = f64::NAN;
        corrupt.cache_compacted = false;
        let key = state.rhokey_sanitized(&rho).expect("cache key");
        state
            .cache_manager
            .pirls_cache
            .write()
            .expect("cache")
            .insert(key, Arc::new(corrupt));
        match state.prepare_eval_bundlewithkey(&rho, None) {
            Err(EstimationError::PirlsRowGeometryUnrepresentable {
                row,
                quantity,
                value,
                ..
            }) => {
                assert_eq!(row, 3);
                assert_eq!(quantity, "observed Hessian weight");
                assert!(value.is_nan());
            }
            Err(other) => panic!("original row error was replaced: {other}"),
            Ok(_) => panic!("sparse row refusal was erased by a dense bundle"),
        }
    });
}

#[test]
fn constraints_route_dense_and_sparse_builder_refuses() {
    with_sparse_fixture(|state, _| {
        let rho = array![0.0];
        let sparse_decision = state.select_reml_geometry(&rho).expect("sparse routing");
        assert!(matches!(
            sparse_decision.geometry,
            RemlGeometry::SparseExactSpd
        ));
        for linear in [false, true] {
            state.coefficient_lower_bounds = if linear {
                None
            } else {
                Some(Array1::zeros(32))
            };
            state.linear_constraints = if linear {
                Some(
                    crate::pirls::LinearInequalityConstraints::new(
                        Array2::eye(32),
                        Array1::zeros(32),
                    )
                    .expect("constraints"),
                )
            } else {
                None
            };
            let decision = state
                .select_reml_geometry(&rho)
                .expect("constrained routing");
            assert!(matches!(decision.geometry, RemlGeometry::DenseSpectral));
            assert_eq!(decision.reason, "constraints_present");
            match state.prepare_sparse_eval_bundlewithkey(
                &rho,
                None,
                BundleRows::Observed,
                sparse_decision.clone(),
            ) {
                Err(EstimationError::InvalidInput(message)) => {
                    assert!(message.contains("does not support coefficient constraints"))
                }
                Err(other) => panic!("unexpected constraint refusal: {other}"),
                Ok(_) => panic!("sparse builder accepted unsupported constraint curvature"),
            }
        }
    });
}

#[test]
fn transformed_pirls_frame_reroutes_without_a_second_inner_solve() {
    with_sparse_fixture(|state, _| {
        let rho = array![0.0];
        let good = state
            .prepare_eval_bundlewithkey(&rho, None)
            .expect("valid sparse mode");
        let mut transformed = good.pirls_result.as_ref().clone();
        // This ridge fixture's transformation is identity, so its coefficient
        // values and Hessian are also a valid transformed-coordinate carrier.
        transformed.coordinate_frame = crate::pirls::PirlsCoordinateFrame::TransformedQs;
        transformed.cache_compacted = false;
        let key = state.rhokey_sanitized(&rho).expect("cache key");
        state
            .cache_manager
            .pirls_cache
            .write()
            .expect("cache")
            .insert(key, Arc::new(transformed));
        state.last_inner_iters.store(usize::MAX, Ordering::Relaxed);
        let rerouted = state
            .prepare_eval_bundlewithkey(&rho, None)
            .expect("coordinate rerouting");
        assert!(matches!(
            rerouted.geometry.geometry,
            RemlGeometry::DenseSpectral
        ));
        assert_eq!(rerouted.geometry.reason, "pirls_frame_not_sparse_native");
        assert!(rerouted.sparse_exact.is_none());
        assert_eq!(state.last_inner_iters.load(Ordering::Relaxed), usize::MAX);
        assert_eq!(
            rerouted.pirls_result.beta_transformed.as_ref(),
            good.pirls_result.beta_transformed.as_ref()
        );
    });
}
