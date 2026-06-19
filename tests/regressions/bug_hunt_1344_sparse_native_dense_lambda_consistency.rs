//! Cross-backend λ-selection consistency regression for #1344.
//!
//! ## What #1344 is about
//!
//! `penalty_shrinkage_floor` (default `Some(1e-6)`) folds a ρ-independent ridge
//! `shrinkage · P_range` into every penalized range direction of `S_λ`. The
//! reparam engine then reports `log_det` / `det1` (the REML penalty
//! pseudo-logdet term and its derivative) as the determinant of the *floored*
//! penalty `S_λ + shrinkage·P_range`.
//!
//! The dense and Kronecker PIRLS backends build their inner penalized Hessian
//! `H = XᵀWX + S` from that same floored penalty, so their data-fit term and
//! their penalty-logdet term live on one objective. The sparse-native backend
//! used to build `H` from the *bare* λ-weighted canonical sum (no shrinkage)
//! while still reporting the floored `log_det` to REML — an internally
//! inconsistent objective that shifts the REML optimum, so sparse-native
//! selected a different λ (and reported a different EDF) than dense for the
//! **same model**. The path a model takes is chosen purely by penalized-Hessian
//! density, so two statistically identical models could land on different
//! backends and get materially different fits.
//!
//! The fix (commit 6316921d1) makes the sparse-native inner penalty reuse the
//! engine's shrinkage-folded penalty, mapped back to identity coordinates.
//!
//! ## What the landed unit test demonstrates (and what it does NOT)
//!
//! The unit test `sparse_native_reparam_folds_shrinkage_floor_into_penalty`
//! (pirls tests) is a single-backend white-box assertion: it checks that the
//! sparse-native reparam result's penalty root/Gram are mutually consistent and
//! carry the shrinkage ridge. That proves the construction "by construction",
//! but it does NOT demonstrate the property #1344 is actually about: that the
//! sparse-native and dense backends, fed the SAME model, SELECT THE SAME λ and
//! report the SAME EDF.
//!
//! ## What THIS test demonstrates
//!
//! It fits one and the same penalized GAM end-to-end through the full REML outer
//! loop twice. The only difference between the two fits is the storage class of
//! the design matrix:
//!
//!   * a `faer` `SparseColMat` → routes to the **sparse-native** PIRLS backend
//!     (`should_use_sparse_native_pirls` keys on `DesignMatrix::as_sparse()` and
//!     the penalized-Hessian density gate; our banded design keeps that density
//!     far below `SPARSE_NATIVE_MAX_H_DENSITY = 0.30`);
//!   * the bit-identical numbers stored as a dense `Array2` → routes to the
//!     **dense-transformed** PIRLS backend (`design_not_sparse`).
//!
//! Both fits use the SAME `(y, w, offset, penalties, options)` and the SAME
//! non-zero `penalty_shrinkage_floor`, so the shrinkage ridge is active on both.
//! The penalty is a second-difference penalty with a 2-dimensional null space
//! ({constant, linear}), so it carries penalized-range energy and the floor
//! genuinely fires (this is the exact condition the issue says triggers the bug
//! "on every sparse-native fit whose penalty carries penalized range energy").
//!
//! It then asserts the REML-selected smoothing parameters (`lambdas`) and the
//! effective degrees of freedom (`edf_total`) agree across the two backends to a
//! tight tolerance. BEFORE 6316921d1 the sparse-native fit solved on `S_λ` while
//! reporting the logdet of `S_λ + shrinkage·P_range`, so its REML optimum — and
//! hence `lambdas`/`edf` — drifted away from the dense backend's; this test is
//! the direct end-to-end witness of that divergence being closed.
//!
//! The tolerance is deliberately tight (relative 1e-3 on λ in log space, and an
//! absolute 1e-2 on EDF). The two backends differ in their internal
//! reparameterization (the dense path applies a `qs` rotation, sparse-native
//! keeps identity coords) and run independent outer optimizers seeded
//! identically, so byte-identity is not expected; but the SELECTED model is a
//! property of the (now shared) objective, not of the linear-algebra basis, so
//! the selected λ and EDF must agree to optimizer tolerance. If they do not, the
//! two backends are optimizing different objectives — exactly the #1344 bug —
//! and this test FAILS rather than being weakened.

use faer::sparse::{SparseColMat, Triplet};
use gam::estimate::{ExternalOptimOptions, optimize_external_design};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};

/// Deterministic linear-congruential normal generator (no external rng dep, so
/// the fixture is bit-reproducible across machines / CI / MSI).
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) as f64 + 1.0) / ((1u64 << 31) as f64 + 1.0)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Build a banded "local-support" design (a B-spline-like tent basis) so that
/// `XᵀWX` and the second-difference penalty are both banded → the penalized
/// Hessian density sits far below `SPARSE_NATIVE_MAX_H_DENSITY`, so the SPARSE
/// copy of this design genuinely takes the sparse-native PIRLS path.
///
/// Returns the design as a dense `Array2` (column-major numbers); the test
/// constructs both a `DesignMatrix::Dense` and a numerically identical
/// `SparseColMat` from it.
fn banded_design(n: usize, p: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        // Place each observation at a location t ∈ [0, p-1] and give it local
        // support on the 3 nearest basis columns with simple tent weights. This
        // mirrors a compact-support spline basis: every row has ≤ 3 nonzeros.
        let t = (i as f64) * ((p - 1) as f64) / ((n - 1) as f64);
        let c = t.floor() as isize;
        let frac = t - c as f64;
        let weights = [
            0.5 * (1.0 - frac).powi(2),
            0.5 + frac - frac * frac,
            0.5 * frac * frac,
        ];
        for (k, &w) in weights.iter().enumerate() {
            let col = c - 1 + k as isize;
            if (0..p as isize).contains(&col) {
                x[[i, col as usize]] = w;
            }
        }
    }
    x
}

/// Second-difference penalty `Dᵀ D` on a `p`-coefficient block. Its null space
/// is {constant, linear} (dimension 2), so the penalty carries penalized-range
/// energy and the shrinkage floor fires on the penalized directions — the
/// condition #1344 requires.
fn second_difference_penalty(p: usize) -> Array2<f64> {
    // D is (p-2) × p, with rows [1, -2, 1].
    let m = p - 2;
    let mut d = Array2::<f64>::zeros((m, p));
    for r in 0..m {
        d[[r, r]] = 1.0;
        d[[r, r + 1]] = -2.0;
        d[[r, r + 2]] = 1.0;
    }
    d.t().dot(&d)
}

fn gaussian_identity_opts(shrinkage_floor: f64) -> ExternalOptimOptions {
    ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        // Inner P-IRLS must converge tightly so the REML criterion that drives λ
        // selection is evaluated at the converged β̂ for BOTH backends; loose
        // inner tolerance would let backend-specific warm-start residue leak into
        // the comparison and mask (or fake) a divergence.
        tol: 1e-11,
        nullspace_dims: vec![2],
        linear_constraints: None,
        firth_bias_reduction: None,
        // The property under test: a non-zero shrinkage floor that the engine
        // folds into the penalized range. The same value is used for both
        // backends so the objective is identical; the bug was that only the
        // dense backend honored it in its inner H.
        penalty_shrinkage_floor: Some(shrinkage_floor),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

/// Fit the same model both ways and return `(lambdas, edf_total, reml_score)`
/// for the sparse-native backend and the dense backend.
#[allow(clippy::type_complexity)]
fn fit_both_backends(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    offset: &Array1<f64>,
    penalty: &Array2<f64>,
    shrinkage_floor: f64,
) -> ((Array1<f64>, f64, f64), (Array1<f64>, f64, f64)) {
    let p = x.ncols();
    let n = x.nrows();
    let opts = gaussian_identity_opts(shrinkage_floor);

    // --- SPARSE backend: identical numbers as a faer SparseColMat ---
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for i in 0..n {
        for j in 0..p {
            let v = x[[i, j]];
            if v != 0.0 {
                triplets.push(Triplet::new(i, j, v));
            }
        }
    }
    let x_sparse = SparseColMat::try_new_from_triplets(n, p, &triplets)
        .expect("sparse design should assemble from banded triplets");

    let sparse_res = optimize_external_design(
        y.view(),
        w.view(),
        x_sparse,
        offset.view(),
        vec![BlockwisePenalty::new(0..p, penalty.clone())],
        &opts,
    )
    .expect("sparse-native external fit must succeed");

    // --- DENSE backend: identical numbers as an Array2 ---
    let dense_res = optimize_external_design(
        y.view(),
        w.view(),
        x.clone(),
        offset.view(),
        vec![BlockwisePenalty::new(0..p, penalty.clone())],
        &opts,
    )
    .expect("dense external fit must succeed");

    let sparse_edf = sparse_res
        .inference
        .as_ref()
        .map(|i| i.edf_total)
        .expect("sparse fit must report inference/edf");
    let dense_edf = dense_res
        .inference
        .as_ref()
        .map(|i| i.edf_total)
        .expect("dense fit must report inference/edf");

    (
        (sparse_res.lambdas.clone(), sparse_edf, sparse_res.reml_score),
        (dense_res.lambdas.clone(), dense_edf, dense_res.reml_score),
    )
}

#[test]
fn sparse_native_and_dense_backends_select_same_lambda_under_shrinkage_floor() {
    // A smooth truth so the REML λ-search lands at a non-trivial interior
    // optimum (not at a boundary where both backends would trivially agree).
    let n = 160usize;
    let p = 24usize;
    let x = banded_design(n, p);
    let penalty = second_difference_penalty(p);

    // Ground-truth coefficients tracing a smooth curve; the response is the
    // design times those coefficients plus moderate noise.
    let mut beta_true = Array1::<f64>::zeros(p);
    for j in 0..p {
        let u = j as f64 / (p - 1) as f64;
        beta_true[j] = (3.0 * std::f64::consts::PI * u).sin() + 0.5 * u;
    }
    let mut rng = Lcg::new(0xC0FFEE_1344);
    let mut y = x.dot(&beta_true);
    for yi in y.iter_mut() {
        *yi += 0.15 * rng.next_normal();
    }
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    // A clearly non-zero shrinkage floor so the regression is exercised
    // unambiguously (the default 1e-6 also triggers, but a larger value makes
    // any backend-specific omission of the ridge dominate the REML optimum).
    let shrinkage_floor = 1e-3;

    let ((sparse_lam, sparse_edf, sparse_reml), (dense_lam, dense_edf, dense_reml)) =
        fit_both_backends(&x, &y, &w, &offset, &penalty, shrinkage_floor);

    eprintln!(
        "[#1344] sparse-native: lambda={:?} edf={:.6} reml={:.6}",
        sparse_lam.as_slice().unwrap(),
        sparse_edf,
        sparse_reml
    );
    eprintln!(
        "[#1344] dense:         lambda={:?} edf={:.6} reml={:.6}",
        dense_lam.as_slice().unwrap(),
        dense_edf,
        dense_reml
    );

    assert_eq!(
        sparse_lam.len(),
        dense_lam.len(),
        "both backends must report the same number of smoothing parameters"
    );
    assert_eq!(sparse_lam.len(), 1, "single penalty block ⇒ one λ");

    // Sanity: the selected λ must be a genuine interior optimum, not a boundary
    // artifact (where the comparison would be vacuous). exp(±12) are the outer
    // search bounds; require the optimum to sit well inside.
    let sparse_log = sparse_lam[0].ln();
    assert!(
        sparse_log.abs() < 11.0 && sparse_log.is_finite(),
        "selected λ must be an interior optimum, got log λ = {sparse_log}"
    );

    // (1) Selected smoothing parameter agrees across backends.
    //
    // Compare in log space (λ is scale-like): the relative difference of the
    // two optima must be at optimizer tolerance, NOT the order-of-magnitude
    // gap a wrong-objective backend would produce.
    let log_sparse = sparse_lam[0].ln();
    let log_dense = dense_lam[0].ln();
    let rel_log_diff = (log_sparse - log_dense).abs() / (1.0 + log_dense.abs());
    assert!(
        rel_log_diff < 1e-3,
        "cross-backend λ divergence (#1344): sparse-native log λ = {log_sparse:.8}, \
         dense log λ = {log_dense:.8}, relative log-difference = {rel_log_diff:.3e} \
         exceeds 1e-3. The two backends are selecting different smoothing \
         parameters for the same model — they are optimizing different REML \
         objectives, which is exactly the bug #1344 closed."
    );

    // (2) Reported EDF agrees across backends. EDF is the trace of the influence
    // matrix at the selected λ on the (now shared) penalty; a wrong inner
    // penalty would report a different EDF even at a coincidentally-similar λ.
    let edf_diff = (sparse_edf - dense_edf).abs();
    assert!(
        edf_diff < 1e-2,
        "cross-backend EDF divergence (#1344): sparse-native edf = {sparse_edf:.6}, \
         dense edf = {dense_edf:.6}, |Δ| = {edf_diff:.3e} exceeds 1e-2"
    );
}
