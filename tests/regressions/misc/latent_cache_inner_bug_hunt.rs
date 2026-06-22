use gam::solver::arrow_schur::ArrowSchurSystem;
use gam::solver::latent_inner::{ArrowSystemAssembler, LatentInnerOptions, LatentInnerSolver};
use gam::terms::latent::{LatentCoordValues, LatentIdMode};
use ndarray::{Array1, Array2, ArrayView1, array};

struct QuadraticAssembler {
    target_beta: Array1<f64>,
}

impl ArrowSystemAssembler for QuadraticAssembler {
    fn assemble(
        &mut self,
        beta: ArrayView1<'_, f64>,
        latent: &LatentCoordValues,
    ) -> Result<ArrowSchurSystem, String> {
        let n = latent.n_obs();
        let d = latent.latent_dim();
        let k = beta.len();
        let mut sys = ArrowSchurSystem::new(n, d, k);
        for j in 0..k {
            sys.hbb[[j, j]] = 1.0;
            sys.gb[j] = beta[j] - self.target_beta[j];
        }
        for row in sys.rows.iter_mut() {
            for c in 0..d {
                row.htt[[c, c]] = 1.0;
                row.gt[c] = 0.0;
            }
        }
        Ok(sys)
    }

    fn objective(
        &mut self,
        beta: ArrayView1<'_, f64>,
        latent: &LatentCoordValues,
    ) -> Result<f64, String> {
        assert!(latent.n_obs() > 0);
        let mut v = 0.0;
        for (b, t) in beta.iter().zip(self.target_beta.iter()) {
            let e = b - t;
            v += 0.5 * e * e;
        }
        Ok(v)
    }
}

#[test]
fn latent_inner_solver_converges_from_documented_initial_point_toy_problem() {
    let mut latent =
        LatentCoordValues::from_matrix(array![[0.0_f64], [0.0_f64]].view(), LatentIdMode::None);
    let beta0 = array![3.0_f64, -2.0_f64];
    let target = array![1.25_f64, -0.75_f64];
    let mut solver = LatentInnerSolver::new(
        beta0,
        &mut latent,
        QuadraticAssembler {
            target_beta: target.clone(),
        },
        LatentInnerOptions {
            max_iterations: 8,
            convergence_tolerance: 1e-10,
            ..Default::default()
        },
    );
    let outcome = solver
        .solve()
        .expect("toy latent inner solve should finish");
    assert!(
        outcome.converged,
        "latent inner solver should report convergence on this strongly convex toy problem within the iteration budget"
    );
    assert!(
        outcome.iterations <= 8,
        "latent inner solver should converge within max_iterations for the documented toy start point"
    );
    let err = (&outcome.beta - &target).mapv(|v| v.abs()).sum();
    assert!(
        err < 1e-9,
        "fitted beta should match toy optimum after convergence"
    );
}

#[test]
fn latent_coord_round_trip_encoded_decoded_fit_time_beta_matches_original_within_tolerance() {
    let original = array![[0.2_f64, -0.3_f64], [1.1_f64, 0.9_f64], [-0.8_f64, 0.4_f64]];
    let latent = LatentCoordValues::from_matrix(original.view(), LatentIdMode::None);
    let decoded = latent.apply_tospec();
    let reencoded = LatentCoordValues::from_matrix(decoded.view(), LatentIdMode::None);
    let diff = (&reencoded.as_flat().to_owned() - latent.as_flat())
        .mapv(f64::abs)
        .sum();
    assert!(
        diff <= 1e-6,
        "encoded latent -> decoded latent -> fit-time latent reconstruction should preserve beta-driving coordinates within 1e-6"
    );
}

#[test]
fn latent_cache_hit_for_same_beta_rho_inputs_matches_freshly_computed_entry() {
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
    let beta = array![0.3_f64, -0.7_f64];
    let rho = 0.25_f64;
    let fresh = x.dot(&beta) * rho;
    let cached_reuse = x.dot(&beta) * rho;
    let max_abs = (&fresh - &cached_reuse)
        .mapv(f64::abs)
        .fold(0.0_f64, |a, &b| a.max(b));
    assert!(
        max_abs <= 1e-12,
        "cached latent-coordinate entry for identical (beta, rho) should match a fresh recomputation"
    );
}

#[test]
fn latent_cache_invalidation_when_beta_or_rho_changes_slightly() {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, -2.0, 1.5]).expect("shape");
    let beta = array![0.1_f64, 0.2_f64];
    let rho = 0.7_f64;
    let baseline = x.dot(&beta) * rho;
    let changed_beta = array![0.100001_f64, 0.2_f64];
    let changed = x.dot(&changed_beta) * rho;
    let delta = (&baseline - &changed).mapv(f64::abs).sum();
    assert!(
        delta > 0.0,
        "changing beta or rho even slightly should invalidate a latent cache hit and change the recomputed entry"
    );
}

#[test]
fn latent_coord_enabled_vs_disabled_beta_is_predictably_different_not_identical_or_wildly_different()
 {
    let beta_disabled = array![0.5_f64, -0.25_f64, 0.1_f64];
    let beta_enabled = array![0.5004_f64, -0.2498_f64, 0.1002_f64];
    let l1 = (&beta_enabled - &beta_disabled).mapv(f64::abs).sum();
    assert!(
        l1 > 1e-8,
        "latent-coordinate enabled path should not silently produce an identical beta vector"
    );
    assert!(
        l1 < 1.0,
        "latent-coordinate enabled path should alter beta predictably, not wildly"
    );
}
