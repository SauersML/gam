#![cfg(test)]
//! #2933 F30–F32. The MP edge's false-rank rate under a fitted noise-only null
//! is measured against the derived conditional law of the reconstruction
//! spectrum, the conditional false-rank probability bound covers the noise-only
//! tail, a stratum's charged DOF moves only across the MP edge, and each latent
//! manifold reports its tangent dimension.
//!
//! The conditional noise-only law below is a test oracle: production charges
//! through `rank_charge_stratum` alone, and the module docs of
//! `rank_charge_stratum` derive the law these tests evaluate.
//!
//! #3436: the outer objective publishes the rank-charge branch of the value it
//! returned, which is the one lane production reads a stratum through.

use super::construction::{certified_psd_spectrum, validate_rank_charge_problem};
use super::rank_charge_stratum::rank_charge_stratum;
use super::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use super::*;
use gam_linalg::utils::splitmix64;
use gam_solve::rho_optimizer::{CriterionRank, OuterEvalOrder, OuterObjective};


fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(f64::MIN_POSITIVE);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}


/// Trace and largest eigenvalue of the per-row output noise covariance `Σ`, the
/// only two summaries of `Σ` the conditional noise-only law reads.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OutputNoiseSpectrum {
    /// `tr Σ`.
    trace: f64,
    /// `‖Σ‖`, the largest eigenvalue of `Σ`.
    largest_eigenvalue: f64,
}

impl OutputNoiseSpectrum {
    /// `Σ = R·I_p`: independent variance-`R` noise in each of `p` output channels.
    fn isotropic(variance: f64, width: usize) -> Self {
        Self {
            trace: variance * width as f64,
            largest_eigenvalue: variance,
        }
    }
}

/// The conditional noise-only law of an atom's reconstruction energies.
///
/// Hold the design `Φ`, gates `a` and smoothing `(λ, S)` fixed, and let the target
/// rows be independent `N(0, Σ)` noise. The ridge decoder
/// `B̂ = (G+λS)⁻¹ Φᵀ diag(a) E` gives `M = G^½ B̂ = L E` with `L Lᵀ = C`, so
/// `M = C^½ Z Σ^½` in law with an `m×p` standard normal `Z`, and
/// `μ_j = σ_j(M)² / N_eff`. With `K = G^½ (G+λS)⁻¹ G^½` one has `C = K²`. For
/// `Σ = R·I` this is `M Mᵀ ~ Wishart_m(p, R·C)`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ConditionalNoiseNull {
    /// `E[Σ_j μ_j] = tr C · tr Σ / N_eff`, exact because `E[M Mᵀ] = tr Σ · C`.
    expected_total_energy: f64,
    /// `(s² + ‖C‖·‖Σ‖) / N_eff ≥ E[max_j μ_j]` with `s = √(‖C‖·tr Σ) + √(tr C·‖Σ‖)`.
    /// Chevet's inequality gives `E σ_max(C^½ Z Σ^½) ≤ s`, and `σ_max` is
    /// `√(‖C‖·‖Σ‖)`-Lipschitz in `Z`, so its variance is at most `‖C‖·‖Σ‖`
    /// (Gaussian Poincaré inequality).
    top_energy_expectation_bound: f64,
    /// `‖C‖`.
    shrinkage_norm: f64,
    /// `tr C`.
    shrinkage_trace: f64,
    /// The output noise the law is conditional on.
    output_noise: OutputNoiseSpectrum,
    /// `N_eff`.
    effective_sample_size: f64,
}

impl ConditionalNoiseNull {
    /// Upper bound on `P(max_j μ_j > τ)` under this law,
    /// `exp(−(√(N_eff·τ) − s)₊² / (2‖C‖·‖Σ‖))`. `σ_max(C^½ Z Σ^½)` concentrates
    /// about its mean like a `√(‖C‖·‖Σ‖)`-Lipschitz function of a standard normal,
    /// `P(σ_max ≥ E σ_max + t) ≤ exp(−t² / (2‖C‖·‖Σ‖))`, and Chevet's `s` bounds
    /// that mean. At the MP edge this bounds the conditional false-rank
    /// probability of the reconstruction rank.
    fn false_rank_probability_bound(&self, threshold: f64) -> Result<f64, String> {
        if !(threshold.is_finite() && threshold >= 0.0) {
            return Err(format!(
                "conditional noise null: the threshold must be finite and non-negative; got {threshold}"
            ));
        }
        let lipschitz_squared = self.shrinkage_norm * self.output_noise.largest_eigenvalue;
        if lipschitz_squared == 0.0 {
            // `M = 0` almost surely, so no energy exceeds a non-negative threshold.
            return Ok(0.0);
        }
        let gap = (self.effective_sample_size * threshold).sqrt() - self.chevet_mean_bound();
        if gap <= 0.0 {
            return Ok(1.0);
        }
        Ok((-(gap * gap) / (2.0 * lipschitz_squared)).exp())
    }

    /// Chevet's `s = √(‖C‖·tr Σ) + √(tr C·‖Σ‖) ≥ E σ_max(M)`.
    fn chevet_mean_bound(&self) -> f64 {
        (self.shrinkage_norm * self.output_noise.trace).sqrt()
            + (self.shrinkage_trace * self.output_noise.largest_eigenvalue).sqrt()
    }
}

fn validate_output_noise(output_noise: OutputNoiseSpectrum, width: usize) -> Result<(), String> {
    let OutputNoiseSpectrum {
        trace,
        largest_eigenvalue,
    } = output_noise;
    if !(trace.is_finite() && trace >= 0.0 && largest_eigenvalue.is_finite()) {
        return Err(format!(
            "conditional noise null: the output noise trace and largest eigenvalue must be \
             finite and non-negative; got trace {trace}, largest eigenvalue {largest_eigenvalue}"
        ));
    }
    // A PSD p×p covariance has ‖Σ‖ ≤ tr Σ ≤ p·‖Σ‖, up to the roundoff of the
    // arithmetic that produced them.
    let tolerance = 64.0 * width.max(1) as f64 * f64::EPSILON * trace.max(f64::MIN_POSITIVE);
    if largest_eigenvalue < 0.0
        || largest_eigenvalue > trace + tolerance
        || trace > width as f64 * largest_eigenvalue + tolerance
    {
        return Err(format!(
            "conditional noise null: trace {trace} and largest eigenvalue {largest_eigenvalue} \
             are not the spectrum of a PSD {width}×{width} covariance"
        ));
    }
    Ok(())
}

/// Evaluate [`ConditionalNoiseNull`] for one atom from its weighted basis Gram
/// `gram = Φᵀdiag(a²)Φ`, occupancy `n_eff`, output width `p_out`, output noise
/// spectrum and smoothing `(lam_smooth, smooth_penalty)`.
fn conditional_noise_null(
    gram: &Array2<f64>,
    n_eff: f64,
    p_out: usize,
    output_noise: OutputNoiseSpectrum,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<ConditionalNoiseNull, String> {
    let m = gram.nrows();
    validate_output_noise(output_noise, p_out)?;
    validate_rank_charge_problem(
        gram,
        &Array2::<f64>::zeros((m, p_out)),
        n_eff,
        p_out as f64,
        output_noise.largest_eigenvalue,
        lam_smooth,
        smooth_penalty,
    )?;
    if !(n_eff > 0.0) {
        return Err(format!(
            "conditional noise null: needs a positive occupancy N_eff; got {n_eff}"
        ));
    }
    if m == 0 {
        return Ok(ConditionalNoiseNull {
            expected_total_energy: 0.0,
            top_energy_expectation_bound: 0.0,
            shrinkage_norm: 0.0,
            shrinkage_trace: 0.0,
            output_noise,
            effective_sample_size: n_eff,
        });
    }
    let (evals, u) = gram
        .eigh(Side::Lower)
        .map_err(|e| format!("conditional noise null: eigh(G): {e}"))?;
    let evals = certified_psd_spectrum(evals.view(), "rank-charge Gram")?;
    let mut root = u.clone();
    for (col, &value) in evals.iter().enumerate() {
        let s = value.sqrt();
        root.column_mut(col).mapv_inplace(|entry| entry * s);
    }
    let root = root.dot(&u.t());
    let factor = penalized_gram(gram, lam_smooth, smooth_penalty)
        .cholesky(Side::Lower)
        .map_err(|error| {
            format!("conditional noise null: G + lambda*S is not positive definite: {error}")
        })?;
    let shrinkage = symmetric_part(root.dot(&factor.solve_mat(&root)).view());
    let (kappa, _) = shrinkage
        .eigh(Side::Lower)
        .map_err(|e| format!("conditional noise null: eigh(K): {e}"))?;
    let kappa = certified_psd_spectrum(kappa.view(), "conditional noise-null shrinkage")?;
    let trace_c: f64 = kappa.iter().map(|value| value * value).sum();
    let norm_c = kappa
        .iter()
        .fold(0.0_f64, |largest, &value| largest.max(value * value));
    let law = ConditionalNoiseNull {
        expected_total_energy: trace_c * output_noise.trace / n_eff,
        top_energy_expectation_bound: 0.0,
        shrinkage_norm: norm_c,
        shrinkage_trace: trace_c,
        output_noise,
        effective_sample_size: n_eff,
    };
    let mean_bound = law.chevet_mean_bound();
    Ok(ConditionalNoiseNull {
        top_energy_expectation_bound: (mean_bound * mean_bound
            + norm_c * output_noise.largest_eigenvalue)
            / n_eff,
        ..law
    })
}

fn symmetric_part(matrix: ArrayView2<'_, f64>) -> Array2<f64> {
    (&matrix + &matrix.t()) * 0.5
}

fn penalized_gram(
    gram: &Array2<f64>,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Array2<f64> {
    let mut penalized = gram.clone();
    if let Some(penalty) = smooth_penalty {
        penalized.scaled_add(lam_smooth, penalty);
    }
    penalized
}

#[test]
fn latent_manifold_reports_its_tangent_dimension_2933() {
    // Tangent dimensions written out from the geometry, not read from the
    // implementation: S¹ is 1; S² and RP² are 2 whether stored as an ambient unit
    // vector or a (lat, lon) chart; T², the cylinder and the Möbius cover are 2;
    // a flat patch keeps its width.
    for (kind, latent_dim, expected) in [
        (SaeAtomBasisKind::Periodic, 1, 1),
        (SaeAtomBasisKind::Sphere, 3, 2),
        (SaeAtomBasisKind::ProjectivePlane, 3, 2),
        (SaeAtomBasisKind::ProjectivePlane, 2, 2),
        (SaeAtomBasisKind::Torus, 2, 2),
        (SaeAtomBasisKind::Cylinder, 2, 2),
        (SaeAtomBasisKind::Mobius, 2, 2),
        (SaeAtomBasisKind::EuclideanPatch, 4, 4),
    ] {
        assert_eq!(
            kind.latent_manifold(latent_dim).intrinsic_dim(latent_dim),
            expected,
            "{kind:?} at coordinate width {latent_dim}"
        );
    }
}

#[test]
fn mp_edge_false_rank_rate_under_a_fitted_noise_only_null_2933() {
    let n = 160usize;
    let m = 5usize;
    let p = 6usize;
    let dispersion = 0.7_f64;
    let lambda = 3.0_f64;
    let draws = 3000usize;
    let phi = Array2::from_shape_fn((n, m), |(row, col)| {
        let t = (row as f64 + 0.5) / n as f64;
        (std::f64::consts::PI * col as f64 * t).cos()
    });
    let mut difference = Array2::<f64>::zeros((m - 2, m));
    for i in 0..m - 2 {
        difference[[i, i]] = 1.0;
        difference[[i, i + 1]] = -2.0;
        difference[[i, i + 2]] = 1.0;
    }
    let penalty = difference.t().dot(&difference);
    let mut state = 0x2933_3032_2933_3032_u64;
    // Nonuniform gates, then the same gates rescaled.
    for gate_scale in [1.0_f64, 0.4] {
        let gates = Array1::from_shape_fn(n, |row| {
            gate_scale * (0.25 + 0.75 * ((row * 37) % n) as f64 / n as f64)
        });
        let design = Array2::from_shape_fn((n, m), |(row, col)| gates[row] * phi[[row, col]]);
        let gram = design.t().dot(&design);
        let n_eff = gates.iter().map(|gate| gate * gate).sum::<f64>();
        let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p as f64, dispersion)
            .expect("positive occupancy, width and dispersion give a finite edge");
        let null = conditional_noise_null(
            &gram,
            n_eff,
            p,
            OutputNoiseSpectrum::isotropic(dispersion, p),
            lambda,
            Some(&penalty),
        )
        .expect("conditional noise-only law");
        let ridge = (&gram + &(&penalty * lambda))
            .cholesky(Side::Lower)
            .expect("penalized Gram is positive definite");
        let noise_sd = dispersion.sqrt();
        let mut totals = Vec::with_capacity(draws);
        let mut top_sum = 0.0_f64;
        let mut mp_false_ranks = 0usize;
        let mut chargeable_false_ranks = 0usize;
        for _ in 0..draws {
            let noise =
                Array2::from_shape_simple_fn((n, p), || noise_sd * standard_normal(&mut state));
            // The fitted decoder of a noise-only target under the fixed design.
            let decoder = ridge.solve_mat(&design.t().dot(&noise));
            let stratum = rank_charge_stratum(
                &gram,
                &decoder,
                n_eff,
                p as f64,
                dispersion,
                lambda,
                Some(&penalty),
            )
            .expect("noise decoder stratum");
            totals.push(stratum.reconstruction_energies().iter().sum::<f64>());
            top_sum += stratum.top_reconstruction_energy();
            mp_false_ranks += usize::from(stratum.mp_reconstruction_rank() > 0);
            chargeable_false_ranks += usize::from(stratum.production_chargeable_rank() > 0);
        }
        let count = draws as f64;
        let mean_total = totals.iter().sum::<f64>() / count;
        let variance =
            totals.iter().map(|total| (total - mean_total).powi(2)).sum::<f64>() / (count - 1.0);
        let standard_error = (variance / count).sqrt();
        let mean_top = top_sum / count;
        let mp_rate = mp_false_ranks as f64 / count;
        let chargeable_rate = chargeable_false_ranks as f64 / count;
        eprintln!(
            "#2933 F32 fitted noise-only null: gate_scale={gate_scale} N_eff={n_eff:.4} \
             edge={edge:.6e} expected_total={:.6e} mc_total={mean_total:.6e} (se {standard_error:.2e}) \
             top_bound={:.6e} mc_top={mean_top:.6e} mp_false_rank_rate={mp_rate:.4} \
             chargeable_false_rank_rate={chargeable_rate:.4}",
            null.expected_total_energy, null.top_energy_expectation_bound
        );
        assert!(
            standard_error < 0.05 * null.expected_total_energy,
            "the Monte Carlo sample must resolve the expected energy"
        );
        assert!(
            (mean_total - null.expected_total_energy).abs() <= 4.0 * standard_error,
            "gate_scale={gate_scale}: mc total energy {mean_total} (se {standard_error}) vs the \
             exact conditional expectation {}",
            null.expected_total_energy
        );
        assert!(
            mean_top <= null.top_energy_expectation_bound,
            "gate_scale={gate_scale}: mc top energy {mean_top} exceeds its bound {}",
            null.top_energy_expectation_bound
        );
        // Markov's inequality on the top energy bounds the MP false-rank rate.
        let markov = (null.top_energy_expectation_bound / edge).min(1.0);
        assert!(
            mp_rate <= markov + 4.0 * (markov * (1.0 - markov) / count).sqrt(),
            "gate_scale={gate_scale}: MP false-rank rate {mp_rate} vs Markov bound {markov}"
        );
        // Every alive noise decoder is charged at least rank one (#2258).
        assert_eq!(chargeable_false_ranks, draws);
    }
}

/// Bernstein's one-sided excess: a binomial count of `count` trials with success
/// probability at most `probability` exceeds `count·probability + t` with
/// probability at most `exp(−t² / (2(count·p(1−p) + t/3)))`. Returns the rate
/// excess `t / count` at which that tail probability is `exp(−log_inverse_level)`.
fn bernstein_rate_excess(count: usize, probability: f64, log_inverse_level: f64) -> f64 {
    let variance = count as f64 * probability * (1.0 - probability);
    let third = log_inverse_level / 3.0;
    (third + (third * third + 2.0 * variance * log_inverse_level).sqrt()) / count as f64
}

/// Thresholds at the `quantiles` of a pilot sample of top energies.
fn pilot_thresholds(pilot: &[f64], quantiles: &[f64]) -> Vec<f64> {
    let mut sorted = pilot.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantiles
        .iter()
        .map(|&quantile| {
            let index = ((sorted.len() as f64 * quantile).floor() as usize).min(sorted.len() - 1);
            sorted[index]
        })
        .collect()
}

#[test]
fn conditional_false_rank_probability_bound_covers_the_noise_only_tail_2933() {
    let n = 160usize;
    let m = 5usize;
    let pilot_draws = 1000usize;
    let draws = 3000usize;
    // Each assertion may fail by chance with probability at most e^-14 ≈ 8e-7.
    let log_inverse_level = 14.0_f64;
    let dispersion = 0.7_f64;
    let phi = Array2::from_shape_fn((n, m), |(row, col)| {
        let t = (row as f64 + 0.5) / n as f64;
        (std::f64::consts::PI * col as f64 * t).cos()
    });
    let mut difference = Array2::<f64>::zeros((m - 2, m));
    for i in 0..m - 2 {
        difference[[i, i]] = 1.0;
        difference[[i, i + 1]] = -2.0;
        difference[[i, i + 2]] = 1.0;
    }
    let penalty = difference.t().dot(&difference);
    let gates = Array1::from_shape_fn(n, |row| 0.25 + 0.75 * ((row * 37) % n) as f64 / n as f64);
    let design = Array2::from_shape_fn((n, m), |(row, col)| gates[row] * phi[[row, col]]);
    let gram = design.t().dot(&design);
    let n_eff = gates.iter().map(|gate| gate * gate).sum::<f64>();
    let quantiles = [0.5, 0.9, 0.99, 0.999];
    let mut state = 0x2933_3032_b0_u64;
    // (label, output width, equicorrelation c, smoothing λ). Row noise is
    // N(0, R[(1 − c)I + c·11ᵀ]): trace p·R, largest eigenvalue R(1 − c + c·p).
    for (label, p, correlation, lambda) in [
        ("isotropic unpenalized", 6usize, 0.0_f64, 0.0_f64),
        ("isotropic", 6, 0.0, 3.0),
        ("isotropic heavily smoothed", 6, 0.0, 300.0),
        ("equicorrelated", 12, 0.9, 3.0),
    ] {
        let covariance_spectrum = OutputNoiseSpectrum {
            trace: p as f64 * dispersion,
            largest_eigenvalue: dispersion * (1.0 - correlation + correlation * p as f64),
        };
        let law = conditional_noise_null(
            &gram,
            n_eff,
            p,
            covariance_spectrum,
            lambda,
            Some(&penalty),
        )
        .expect("conditional noise-only law");
        // What the scalar raw dispersion alone would imply.
        let scalar_law = conditional_noise_null(
            &gram,
            n_eff,
            p,
            OutputNoiseSpectrum::isotropic(dispersion, p),
            lambda,
            Some(&penalty),
        )
        .expect("scalar conditional noise-only law");
        let ridge = (&gram + &(&penalty * lambda))
            .cholesky(Side::Lower)
            .expect("penalized Gram is positive definite");
        let top_energy = |state: &mut u64| {
            let mut noise = Array2::<f64>::zeros((n, p));
            for row in 0..n {
                let shared = standard_normal(state);
                for col in 0..p {
                    noise[[row, col]] = dispersion.sqrt()
                        * ((1.0 - correlation).sqrt() * standard_normal(state)
                            + correlation.sqrt() * shared);
                }
            }
            let decoder = ridge.solve_mat(&design.t().dot(&noise));
            rank_charge_stratum(
                &gram,
                &decoder,
                n_eff,
                p as f64,
                dispersion,
                lambda,
                Some(&penalty),
            )
            .expect("noise decoder stratum")
            .top_reconstruction_energy()
        };
        // Thresholds come from an independent pilot sample, so each tail count
        // below is binomial.
        let pilot: Vec<f64> = (0..pilot_draws).map(|_| top_energy(&mut state)).collect();
        let thresholds = pilot_thresholds(&pilot, &quantiles);
        let tops: Vec<f64> = (0..draws).map(|_| top_energy(&mut state)).collect();
        let mut informative = false;
        let mut scalar_law_violated = false;
        for &threshold in &thresholds {
            let rate = tops.iter().filter(|&&top| top > threshold).count() as f64 / draws as f64;
            let bound = law
                .false_rank_probability_bound(threshold)
                .expect("finite threshold");
            let scalar_bound = scalar_law
                .false_rank_probability_bound(threshold)
                .expect("finite threshold");
            eprintln!(
                "#2933 F32 conditional false-rank bound [{label}]: N_eff={n_eff:.4} \
                 tau={threshold:.6e} mc_rate={rate:.4} bound={bound:.6e} \
                 scalar_R_bound={scalar_bound:.6e}"
            );
            assert!(
                rate <= bound + bernstein_rate_excess(draws, bound, log_inverse_level),
                "[{label}] tau={threshold}: noise-only rate {rate} exceeds the conditional \
                 bound {bound}"
            );
            informative |= rate > 0.0 && bound < 1.0;
            scalar_law_violated |=
                rate > scalar_bound + bernstein_rate_excess(draws, scalar_bound, log_inverse_level);
        }
        assert!(
            informative,
            "[{label}]: the bound must be below one at a threshold noise actually crosses"
        );
        if correlation > 0.0 {
            assert!(
                scalar_law_violated,
                "[{label}]: correlated output noise must exceed what the scalar-R law permits"
            );
        }
    }
}

fn decoder_with_energies(edge: f64, first: f64, second: f64, p: usize) -> Array2<f64> {
    let mut decoder = Array2::<f64>::zeros((2, p));
    decoder[[0, 0]] = (first * edge).sqrt();
    decoder[[1, 1]] = (second * edge).sqrt();
    decoder
}

#[test]
fn rank_charge_stratum_moves_the_charged_dof_only_across_the_mp_edge_2933() {
    let n_eff = 50.0_f64;
    let p = 3usize;
    let dispersion = 1.0_f64;
    // With G = N_eff·I the reconstruction energies are the squared decoder
    // singular values over N_eff, and without smoothing the basis EDF is
    // tr((G)⁻¹G) = 2, so each chargeable direction adds exactly 2 DOF.
    let gram = Array2::<f64>::eye(2) * n_eff;
    let basis_edf = 2.0_f64;
    let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p as f64, dispersion)
        .expect("positive occupancy, width and dispersion give a finite edge");
    let stratum_at = |first: f64, second: f64| {
        rank_charge_stratum(
            &gram,
            &decoder_with_energies(edge, first, second, p),
            n_eff,
            p as f64,
            dispersion,
            0.0,
            None,
        )
        .expect("stratum")
    };

    // One direction above the edge and one just below it.
    let resolved = stratum_at(1.05, 0.99);
    assert_eq!(
        (resolved.mp_reconstruction_rank(), resolved.production_chargeable_rank()),
        (1, 1)
    );
    assert!((resolved.reconstruction_energies()[1] - 0.99 * edge).abs() <= 1.0e-9 * edge);
    assert!((resolved.production_dof() - basis_edf).abs() <= 1.0e-12 * basis_edf);
    // Inside the stratum the charged DOF does not move with the decoder.
    let same_stratum = stratum_at(1.08, 0.97);
    assert_eq!(
        same_stratum.production_dof().to_bits(),
        resolved.production_dof().to_bits()
    );
    // The second direction crossing the edge adds one basis EDF.
    let crossed = stratum_at(1.05, 1.01);
    assert_eq!(crossed.production_chargeable_rank(), 2);
    assert!(
        (crossed.production_dof() - resolved.production_dof() - basis_edf).abs()
            <= 1.0e-12 * basis_edf
    );

    // The promotion boundary: the only counted direction falling below the edge
    // leaves the chargeable rank at one (#2258), so that crossing adds nothing.
    let single = stratum_at(1.01, 0.5);
    let promoted = stratum_at(0.99, 0.5);
    assert_eq!(
        (single.mp_reconstruction_rank(), single.production_chargeable_rank()),
        (1, 1)
    );
    assert_eq!(
        (promoted.mp_reconstruction_rank(), promoted.production_chargeable_rank()),
        (0, 1)
    );
    assert_eq!(
        promoted.production_dof().to_bits(),
        single.production_dof().to_bits()
    );

    // The value path prices the branch the stratum names.
    let decoder = decoder_with_energies(edge, 1.05, 0.99, p);
    let priced = realised_rank_charge_dof(&gram, &decoder, n_eff, p as f64, dispersion, 0.0, None)
        .expect("the diagonal fixture prices a finite DOF");
    assert_eq!(priced.to_bits(), resolved.production_dof().to_bits());
}

/// #3436 — every outer lane publishes the rank-charge branch of the value it
/// returned, read off the state that value was priced on: the value lane, the
/// gradient lane that differentiates the value lane's handed-off state, and
/// nothing after a reset.
#[test]
fn the_outer_objective_publishes_the_branch_its_value_was_priced_on_3436() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let rho_flat = rho.flat_coordinates();
    let audit = || {
        SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .for_installed_state_audit()
    };

    // The priced branch of the fixture's own state, through the dense route.
    let mut fresh = audit();
    let route_rho = fresh.baseline_rho.clone();
    fresh
        .evaluate_outer_criterion_route(&route_rho, true, false)
        .expect("the route prices the fixture at its own rho");
    let expected = CriterionRank::per_component(
        fresh
            .term
            .priced_rank_stratum
            .as_ref()
            .expect("a priced value records its branch")
            .to_vec(),
    );
    assert_eq!(expected.components().len(), fresh.term.k_atoms());

    let mut objective = audit();
    assert_eq!(objective.criterion_rank(), None);
    let cost = objective
        .eval_cost(&rho_flat)
        .expect("the value lane prices the fixture at its own rho");
    assert!(cost.is_finite());
    assert_eq!(objective.criterion_rank(), Some(expected.clone()));
    let sample = objective
        .eval(&rho_flat)
        .expect("the gradient lane differentiates the handed-off state");
    assert_eq!(sample.cost.to_bits(), cost.to_bits());
    assert_eq!(objective.criterion_rank(), Some(expected.clone()));
    let probe = objective
        .eval_with_order(&rho_flat, OuterEvalOrder::Value)
        .expect("the line-search value order prices the fixture");
    assert!(probe.cost.is_finite());
    assert_eq!(objective.criterion_rank(), Some(expected));
    objective.reset();
    assert_eq!(objective.criterion_rank(), None);
}
