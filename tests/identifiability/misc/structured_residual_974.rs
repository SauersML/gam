//! #974 — structured-residual covariance estimator + single likelihood-whitening
//! seam: planted-truth recovery and value↔gradient consistency.
//!
//! These tests assert OBJECTIVE quality against self-constructed analytic ground
//! truth (the reference-as-truth paradigm), not gam-reproduces-a-tool output:
//!
//!   1. **Planted interference-subspace recovery.** Residuals are generated as
//!      `r_n = Λ₀ f_n + ε_n` with a known low-rank interference factor `Λ₀`; the
//!      fitted factor's range must align with `range(Λ₀)` to small principal
//!      angles.
//!
//!   2. **Smooth activity-scale recovery.** Residuals carry a planted smooth
//!      scale law `c(z) = exp(slope·z)` modulating the factor energy; the fitted
//!      per-row scale must track that law (monotone in `z`, high rank
//!      correlation).
//!
//!   3. **Topology-race bias removal.** A planted correlated-residual confound
//!      makes a SPURIOUS topology preferred under the isotropic likelihood;
//!      whitening by the estimated structured covariance REMOVES that preference
//!      and selects the true topology.
//!
//!   4. **Value↔gradient consistency through the whitening seam.** With a
//!      `WhitenedStructured` `RowMetric` installed on a real SAE term, the
//!      whitened PENALIZED-OBJECTIVE VALUE (`penalized_objective_total`, whose
//!      data-fit component is the whitened `½ rᵀMr`) finite-differences to the
//!      assembled β GRADIENT (`sys.gb`) — proving the single seam feeds both the
//!      value and the gradient of the exact objective the inner Newton step
//!      descends (the objective↔gradient-desync cure). `sys.gb` is the gradient
//!      of that full objective — data-fit PLUS the always-on collapse-prevention
//!      barriers the assembly folds in — so the FD must use the same objective,
//!      not the barrier-free `loss().total()`.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no new public knobs. Fixed seeds.

use ndarray::{Array1, Array2, Array3};

use gam::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam::inference::row_metric::{MetricProvenance, RowMetric};
use gam::solver::arrow_schur::ArrowSchurSystem;
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};

/// Deterministic standard-normal-ish draw via Box–Muller on an LCG uniform.
/// Fully reproducible (no RNG state outside the passed seed word).
fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // (0, 1) open interval.
    (((*state >> 11) as f64) + 0.5) / (1u64 << 53) as f64
}

fn lcg_normal(state: &mut u64) -> f64 {
    let u1 = lcg_uniform(state).max(1e-12);
    let u2 = lcg_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Largest principal angle (radians) between two column subspaces, via the
/// minimal singular value of `Q_aᵀ Q_b` for orthonormal bases `Q_a, Q_b`. Small
/// ⇒ the subspaces nearly coincide.
fn largest_principal_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let qa = orthonormal_basis(a);
    let qb = orthonormal_basis(b);
    // M = Q_aᵀ Q_b (ra × rb). Its singular values are the cosines of the
    // principal angles; the smallest cosine is the largest angle.
    let ra = qa.ncols();
    let rb = qb.ncols();
    let p = qa.nrows();
    let mut m = Array2::<f64>::zeros((ra, rb));
    for i in 0..ra {
        for j in 0..rb {
            let mut acc = 0.0;
            for k in 0..p {
                acc += qa[[k, i]] * qb[[k, j]];
            }
            m[[i, j]] = acc;
        }
    }
    // Smallest singular value of M via the smallest eigenvalue of MᵀM.
    let mtm = m.t().dot(&m);
    let min_eig = smallest_sym_eigenvalue(&mtm).max(0.0);
    min_eig.sqrt().clamp(-1.0, 1.0).acos()
}

/// Gram–Schmidt orthonormal basis for the column space of `a` (drops
/// numerically-zero columns).
fn orthonormal_basis(a: &Array2<f64>) -> Array2<f64> {
    let p = a.nrows();
    let mut cols: Vec<Array1<f64>> = Vec::new();
    for j in 0..a.ncols() {
        let mut v = a.column(j).to_owned();
        for q in &cols {
            let dot: f64 = v.iter().zip(q.iter()).map(|(&x, &y)| x * y).sum();
            for i in 0..p {
                v[i] -= dot * q[i];
            }
        }
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-8 {
            for i in 0..p {
                v[i] /= norm;
            }
            cols.push(v);
        }
    }
    let r = cols.len();
    let mut q = Array2::<f64>::zeros((p, r.max(1)));
    for (j, col) in cols.iter().enumerate() {
        for i in 0..p {
            q[[i, j]] = col[i];
        }
    }
    if r == 0 {
        // Degenerate: a single canonical axis avoids a zero-width basis.
        q[[0, 0]] = 1.0;
    }
    q
}

/// Smallest eigenvalue of a small symmetric matrix via the power method on a
/// shifted inverse-free deflation: for the 1×1/2×2/3×3 cases here we use a
/// robust closed path through the characteristic-polynomial-free Jacobi sweep.
fn smallest_sym_eigenvalue(m: &Array2<f64>) -> f64 {
    let n = m.nrows();
    if n == 1 {
        return m[[0, 0]];
    }
    // Symmetric power iteration on (cI − M) to find the largest eigenvalue of
    // the shift, hence the smallest of M. c = trace + 1 dominates the spectrum.
    let trace: f64 = (0..n).map(|i| m[[i, i]]).sum();
    let shift = trace.abs() + 1.0;
    let mut shifted = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            shifted[[i, j]] = -m[[i, j]];
        }
        shifted[[i, i]] += shift;
    }
    let mut v = Array1::<f64>::from_elem(n, 1.0 / (n as f64).sqrt());
    let mut lambda = 0.0;
    for _ in 0..500 {
        let w = shifted.dot(&v);
        let norm = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-300 {
            break;
        }
        for i in 0..n {
            v[i] = w[i] / norm;
        }
        lambda = norm;
    }
    // Largest eigenvalue of the shift is `shift − min_eig(M)`.
    shift - lambda
}

// ---------------------------------------------------------------------------
// 1. Planted interference-subspace recovery.
// ---------------------------------------------------------------------------

#[test]
fn fitted_factor_recovers_planted_interference_subspace() {
    let n = 4000usize;
    let p = 5usize;
    let r0 = 2usize;
    // Planted interference factor Λ₀ (p × r0): two fixed, well-separated
    // directions, scaled large so the factor dominates the idiosyncratic noise.
    let lambda0 = ndarray::array![[1.6, 0.2], [1.4, -0.3], [0.1, 1.5], [-0.2, 1.3], [0.3, 0.1],];
    let sigma_eps = 0.25_f64;
    let mut seed = 0x9E3779B97F4A7C15_u64;
    let mut residuals = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let mut f = [0.0_f64; 2];
        for fk in f.iter_mut() {
            *fk = lcg_normal(&mut seed);
        }
        for i in 0..p {
            let mut val = sigma_eps * lcg_normal(&mut seed);
            for k in 0..r0 {
                val += lambda0[[i, k]] * f[k];
            }
            residuals[[row, i]] = val;
        }
    }
    // Constant activity ⇒ homoscedastic factor model; the subspace is what we
    // test here.
    let activity = Array1::<f64>::zeros(n);
    let model = StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residuals.view(),
        activity: activity.view(),
        max_factor_rank: 3,
    })
    .expect("estimator fits");

    // The evidence ladder must recover exactly the planted rank.
    assert_eq!(
        model.factor_rank(),
        r0,
        "evidence ladder must select the planted factor rank {r0}, got {}",
        model.factor_rank()
    );

    // The fitted factor range must align with the planted interference subspace.
    let fitted = model.factor().to_owned();
    let angle = largest_principal_angle(&fitted, &lambda0);
    assert!(
        angle < 0.12,
        "largest principal angle between fitted Λ range and planted Λ₀ range must \
         be small; got {angle} rad ({} deg)",
        angle.to_degrees()
    );
}

// ---------------------------------------------------------------------------
// 2. Smooth activity-scale recovery.
// ---------------------------------------------------------------------------

#[test]
fn fitted_scale_recovers_planted_activity_law() {
    let n = 5000usize;
    let p = 4usize;
    // Single planted interference direction with a SMOOTH activity scale
    // c(z) = exp(slope·z): the factor energy grows with z. We recover c(z) up to
    // the global (mean-1) normalization the estimator applies.
    let lambda0 = ndarray::array![[1.5], [1.2], [-0.4], [0.3]];
    let sigma_eps = 0.2_f64;
    let slope = 1.3_f64;
    let mut seed = 0xD1B54A32D192ED03_u64;
    let mut residuals = Array2::<f64>::zeros((n, p));
    let mut activity = Array1::<f64>::zeros(n);
    let mut true_scale = Array1::<f64>::zeros(n);
    for row in 0..n {
        let z = (row as f64) / (n as f64 - 1.0); // z ∈ [0, 1]
        activity[row] = z;
        let c = (slope * z).exp();
        true_scale[row] = c;
        let amp = c.sqrt(); // r_n = √c·Λ₀ f + ε ⇒ Cov factor energy ∝ c
        let f = lcg_normal(&mut seed);
        for i in 0..p {
            residuals[[row, i]] = amp * lambda0[[i, 0]] * f + sigma_eps * lcg_normal(&mut seed);
        }
    }
    let model = StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residuals.view(),
        activity: activity.view(),
        max_factor_rank: 2,
    })
    .expect("estimator fits");

    assert_eq!(model.factor_rank(), 1, "single planted factor");

    // The fitted per-row scale (mean-1 normalized) must track the mean-1
    // normalized planted law. Compare via Spearman-free Pearson correlation
    // (the law is monotone, so a high positive correlation pins the shape).
    let fitted_scale = model.row_scale().to_owned();
    let mean_fit = fitted_scale.iter().sum::<f64>() / n as f64;
    let mean_true = true_scale.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_f = 0.0;
    let mut var_t = 0.0;
    for row in 0..n {
        let df = fitted_scale[row] - mean_fit;
        let dt = true_scale[row] - mean_true;
        cov += df * dt;
        var_f += df * df;
        var_t += dt * dt;
    }
    let corr = cov / (var_f.sqrt() * var_t.sqrt()).max(1e-30);
    assert!(
        corr > 0.9,
        "fitted activity scale must correlate strongly with the planted c(z)=exp({slope}·z) \
         law; got Pearson r = {corr}"
    );

    // And the scale must be genuinely increasing (the planted law is monotone):
    // the mean fitted scale in the top-z decile must exceed the bottom-z decile.
    let dec = n / 10;
    let lo: f64 = (0..dec).map(|i| fitted_scale[i]).sum::<f64>() / dec as f64;
    let hi: f64 = (n - dec..n).map(|i| fitted_scale[i]).sum::<f64>() / dec as f64;
    assert!(
        hi > 1.5 * lo,
        "fitted scale must rise with activity z (planted exp law): bottom-decile \
         mean {lo}, top-decile mean {hi}"
    );
}

// ---------------------------------------------------------------------------
// 3. Topology-race bias removal.
// ---------------------------------------------------------------------------

#[test]
fn structured_likelihood_removes_spurious_topology_preference() {
    // Setup: the residuals exhibit a strong planted interference direction `u`
    // (high-variance correlated NOISE — a nuisance subspace, not signal). We FIT
    // the structured covariance to a training residual sample carrying that
    // interference, then SCORE two candidate "topologies" by the data-fit each
    // leaves on a held-out row:
    //
    //   * SPURIOUS topology: it "uses up" capacity fitting the interference
    //     noise, so its leftover residual is ORTHOGONAL to `u` with a SMALL raw
    //     Euclidean norm — so the ISOTROPIC loss (Σ½‖r‖²) wrongly PREFERS it.
    //   * TRUE topology: it does NOT chase the noise, so its residual is the
    //     unexplained interference itself — ALIGNED with `u` and with a LARGER
    //     raw norm — which the isotropic loss wrongly disfavors.
    //
    // The estimated structured metric Σ⁻¹ DOWN-WEIGHTS the high-variance
    // interference direction (it is noise: large variance ⇒ small precision) and
    // UP-WEIGHTS the orthogonal idiosyncratic directions. So the whitened
    // Mahalanobis loss rᵀΣ⁻¹r treats the true topology's interference-aligned
    // residual as cheap (correctly: it is noise) and the spurious topology's
    // orthogonal residual as expensive — REMOVING the spurious preference.
    let n = 3000usize;
    let p = 4usize;
    // Planted interference direction (unit) and a large variance along it.
    let u = {
        let raw = ndarray::array![1.0_f64, 0.8, -0.6, 0.3];
        let norm = raw.iter().map(|&x| x * x).sum::<f64>().sqrt();
        raw.mapv(|v| v / norm)
    };
    let interference_amp = 3.0_f64;
    let idio = 0.3_f64;
    let mut seed = 0xA5A5A5A5DEADBEEF_u64;
    let mut residuals = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let g = interference_amp * lcg_normal(&mut seed);
        for i in 0..p {
            residuals[[row, i]] = g * u[i] + idio * lcg_normal(&mut seed);
        }
    }
    let activity = Array1::<f64>::zeros(n);
    // `WhitenedStructured` has a single production site: the #974 factor-analytic
    // fitter `StructuredResidualModel::fit(...).row_metric(...)` in `gam-solve`.
    // `RowMetric` lives in the lower `gam-problem` crate, so an inherent
    // `RowMetric::from_estimated_residual_covariance` constructor cannot reach the
    // fitter (it would invert the crate dependency) and would duplicate that sole
    // production site — fit through the documented API instead.
    let model = StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residuals.view(),
        activity: activity.view(),
        max_factor_rank: 2,
    })
    .expect("structured covariance fits");
    let metric = model.row_metric(n).expect("structured metric builds");
    assert!(
        metric.whitens_likelihood(),
        "the fitted structured covariance must produce a likelihood-whitening metric"
    );
    assert!(matches!(
        metric.provenance(),
        MetricProvenance::WhitenedStructured { .. }
    ));

    // Candidate residuals on a representative row (use row 0's metric).
    // An orthogonal-to-u direction via Gram–Schmidt of a canonical axis.
    let mut ortho = ndarray::array![0.0_f64, 0.0, 0.0, 1.0];
    let dot: f64 = ortho.iter().zip(u.iter()).map(|(&a, &b)| a * b).sum();
    for i in 0..p {
        ortho[i] -= dot * u[i];
    }
    let onorm = ortho.iter().map(|&x| x * x).sum::<f64>().sqrt();
    ortho.mapv_inplace(|v| v / onorm);

    // Spurious topology: leftover residual ORTHOGONAL to u, SMALL raw norm.
    let spurious = ortho.mapv(|v| 0.6 * v);
    // True topology: residual ALIGNED with the interference noise u, LARGER raw
    // norm (it leaves the nuisance subspace unexplained — correctly).
    let truth = u.mapv(|v| 1.2 * v);

    // Isotropic (RSS) loss: the spurious candidate wins (smaller raw norm).
    let iso_spurious: f64 = spurious.iter().map(|&v| v * v).sum();
    let iso_true: f64 = truth.iter().map(|&v| v * v).sum();
    assert!(
        iso_spurious < iso_true,
        "by construction the ISOTROPIC loss must PREFER the spurious topology \
         (smaller raw norm): spurious {iso_spurious} vs true {iso_true}"
    );

    // Structured (Mahalanobis) loss: the high-variance interference direction is
    // DOWN-weighted by Σ⁻¹, so the true (interference-aligned) residual is cheap
    // (it is noise) and the spurious (orthogonal, idiosyncratic) residual is
    // expensive — the spurious preference is removed.
    let maha_spurious = metric.quad_form(0, spurious.view());
    let maha_true = metric.quad_form(0, truth.view());
    assert!(
        maha_true < maha_spurious,
        "the structured likelihood must REMOVE the spurious topology preference: \
         under Σ⁻¹ the true (interference-aligned, i.e. noise) topology must be \
         cheaper; got spurious {maha_spurious} vs true {maha_true}"
    );
}

// ---------------------------------------------------------------------------
// 4. Value↔gradient consistency through the single whitening seam.
// ---------------------------------------------------------------------------

/// Build a small Euclidean-patch SAE term with deterministic decoder/basis.
fn build_sae_term(
    k_atoms: usize,
    m: usize,
    d: usize,
    n: usize,
    p: usize,
    seed: u64,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let mut rng = seed;
    let next = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let logits = Array2::from_shape_fn((n, k_atoms), |_| 0.4 * next(&mut rng));
    let target = Array2::from_shape_fn((n, p), |_| 0.5 * next(&mut rng));
    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(k_atoms);
    let mut coord_blocks: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let phi = Array2::from_shape_fn((n, m), |_| 0.2 * next(&mut rng));
        let jet = Array3::from_shape_fn((n, m, d), |_| 0.02 * next(&mut rng));
        let decoder = Array2::from_shape_fn((m, p), |_| 0.3 * next(&mut rng));
        let mut smooth = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            smooth[[i, i]] = 0.1;
        }
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("atom_{atom_idx}"),
            SaeAtomBasisKind::EuclideanPatch,
            d,
            phi,
            jet,
            decoder,
            smooth,
        )
        .expect("atom builds");
        atoms.push(atom);
        coord_blocks.push(Array2::from_shape_fn((n, d), |_| 0.5 * next(&mut rng)));
    }
    let assignment =
        SaeAssignment::from_blocks_with_mode(logits, coord_blocks, AssignmentMode::softmax(1.0))
            .expect("assignment builds");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("term builds");
    // Smoothness ρ-suppressed (very negative log-λ) so the smoothness penalty is
    // machine-negligible; the β-dependent objective is then the whitened data-fit
    // plus the always-on, data-driven collapse-prevention barriers (decoder
    // repulsion + separation/amplitude barrier) the assembly folds into `sys.gb`.
    // ARD acts on the latent coordinates, not β, so it is β-independent regardless
    // of its strength. The FD in the test uses `penalized_objective_total`, which
    // carries exactly the same terms as `sys.gb`.
    let log_ard: Vec<Array1<f64>> = (0..k_atoms).map(|_| Array1::from_elem(d, 0.0)).collect();
    let rho = SaeManifoldRho::new(0.0, -40.0, log_ard);
    (term, target, rho)
}

/// An anisotropic, row-varying per-row precision factor stack `U_n ∈ ℝ^{p×p}` so
/// `M_n = U_n U_nᵀ ≠ I_p` — a genuine `WhitenedStructured` metric whose whitening
/// is non-trivial. Deterministic (no RNG).
fn whitening_metric(n: usize, p: usize) -> RowMetric {
    let mut u = Array2::<f64>::zeros((n, p * p));
    for row in 0..n {
        for i in 0..p {
            for k in 0..p {
                let base = if i == k { 1.0 } else { 0.0 };
                u[[row, i * p + k]] =
                    base + 0.15 * ((i + 2 * k) as f64).sin() + 0.03 * (row as f64).cos();
            }
        }
    }
    RowMetric::whitened_structured(std::sync::Arc::new(u), p, p).expect("metric builds")
}

#[test]
fn whitened_data_fit_value_matches_assembled_beta_gradient_fd() {
    let (k, m, d, n, p) = (3usize, 4usize, 1usize, 12usize, 3usize);
    let (mut term, target, rho) = build_sae_term(k, m, d, n, p, 0x1234_5678_9ABC_DEF0);
    term.set_row_metric(whitening_metric(n, p))
        .expect("install WhitenedStructured metric");

    // Assemble the whitened system and pull the β data-fit gradient.
    let sys: ArrowSchurSystem = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("assemble");
    let analytic_gb = sys.gb.clone();

    // Central-difference the whitened PENALIZED-OBJECTIVE VALUE wrt each β
    // coordinate. `sys.gb` is the β gradient of the exact scalar the inner Newton
    // line search descends — `penalized_objective_total` — NOT of the bare
    // `loss().total()`. Besides the whitened data-fit (and the ρ-suppressed
    // smoothness), that objective carries the always-on, data-driven
    // collapse-prevention terms the assembly folds straight into `sys.gb`: the
    // collinearity-gated decoder repulsion (`add_sae_decoder_repulsion`, #1026)
    // and the interior-point separation/amplitude barriers
    // (`add_sae_separation_barrier`, #1522/#1610). Those are β-dependent and are
    // present in `penalized_objective_total` (via `decoder_repulsion_value` +
    // `separation_barrier_value`) but absent from `loss().total()`, so FDing the
    // bare loss would spuriously disagree with `sys.gb` at any coordinate the
    // barriers touch. FDing the SAME penalized objective the gradient is derived
    // from is the value↔gradient-desync cure the seam claims: the whitening still
    // enters through the data-fit component of that objective on BOTH sides, so a
    // genuine whitening desync is still caught.
    let beta0 = term.flatten_beta();
    let h = 1e-6;
    let mut max_rel_err = 0.0_f64;
    let mut worst = (0usize, 0.0_f64, 0.0_f64);
    // Check a representative subset of coordinates (every coordinate is the same
    // kind of decoder weight; checking all is fine at this size).
    for idx in 0..beta0.len() {
        let mut bp = beta0.clone();
        bp[idx] += h;
        term.set_flat_beta(bp.view()).expect("set +h");
        let lp = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("penalized objective +h");
        let mut bm = beta0.clone();
        bm[idx] -= h;
        term.set_flat_beta(bm.view()).expect("set -h");
        let lm = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("penalized objective -h");
        term.set_flat_beta(beta0.view()).expect("restore");
        let fd = (lp - lm) / (2.0 * h);
        let an = analytic_gb[idx];
        let denom = an.abs().max(fd.abs()).max(1e-6);
        let rel = (fd - an).abs() / denom;
        if rel > max_rel_err {
            max_rel_err = rel;
            worst = (idx, an, fd);
        }
    }
    assert!(
        max_rel_err < 1e-4,
        "whitened penalized-objective VALUE must finite-difference to the assembled β \
         GRADIENT (single-seam value↔gradient consistency; the whitening enters via the \
         data-fit component): worst coord {} analytic {} fd {} (max rel err {max_rel_err})",
        worst.0,
        worst.1,
        worst.2
    );
}

#[test]
fn euclidean_metric_leaves_assembled_gradient_bit_identical() {
    // The seam must be byte-identical to the historical isotropic path when the
    // metric does not whiten the likelihood (Euclidean / none). Assemble with no
    // metric and with a Euclidean metric installed; the β gradient must match
    // bit-for-bit.
    let (k, m, d, n, p) = (3usize, 4usize, 1usize, 12usize, 3usize);
    let (mut term_none, target, rho) = build_sae_term(k, m, d, n, p, 0x0BADF00DCAFEBABE);
    let mut term_euc = term_none.clone();

    let sys_none = term_none
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("assemble none");
    term_euc
        .set_row_metric(RowMetric::euclidean(n, p).expect("euclidean"))
        .expect("install euclidean");
    let sys_euc = term_euc
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("assemble euclidean");

    assert_eq!(
        sys_none.gb.len(),
        sys_euc.gb.len(),
        "gradient length must match"
    );
    for idx in 0..sys_none.gb.len() {
        assert_eq!(
            sys_none.gb[idx].to_bits(),
            sys_euc.gb[idx].to_bits(),
            "Euclidean RowMetric must leave the assembled β gradient BIT-IDENTICAL \
             to the no-metric path at coord {idx}"
        );
    }
    // Per-row t-block gradient must also be bit-identical.
    assert_eq!(sys_none.rows.len(), sys_euc.rows.len());
    for r in 0..sys_none.rows.len() {
        for a in 0..sys_none.rows[r].gt.len() {
            assert_eq!(
                sys_none.rows[r].gt[a].to_bits(),
                sys_euc.rows[r].gt[a].to_bits(),
                "Euclidean must leave the t-block gradient bit-identical at row {r}, axis {a}"
            );
        }
    }
}
