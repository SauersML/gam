//! #2249 / #2263 **calibration** harness — proves the *shipped*
//! [`crate::inference::steering::steer_delta`] `predicted_nats` tracks the true
//! output KL with unit slope across edit directions, THROUGH a nontrivial
//! Tier-0 per-column frame, and that the target-dose loop
//! [`crate::inference::steering::steer_to_target_nats`] lands a requested nats
//! dose in ~1 probe.
//!
//! [`crate::inference::tests_dose_units_2249`] already pins the *units* of
//! `fisher_mass` against the closed-form categorical Fisher with no term and no
//! frame in the loop. This module closes the gap that the earlier audit
//! (`ace3b9af3`) flagged as the real #2249 confound: the shipped predictor runs
//! `predicted_nats = 0.5·δ_rawᵀ M δ_raw`, where `δ_raw` is the decoder chord
//! *un-scaled back to raw activation units* by the term's Tier-0 column scale
//! `σ` and `M = U Uᵀ` is the raw-frame output-Fisher. If that σ un-scaling is
//! dropped (the pre-`ace3b9af3` bug) the dose is priced through `D⁻¹ M D⁻¹`
//! (`D = diag σ`), a per-DIRECTION mis-scale that both biases the slope and
//! collapses R². This test drives the actual `steer_delta`/`steer_to_target_nats`
//! surface on a real fitted term carrying an asymmetric σ and shows:
//!
//! 1. **In-frame calibration**: regressing the exact categorical KL of the
//!    applied `δ_raw` on the shipped `predicted_nats`, over a sweep of random
//!    edit directions and amplitudes, yields slope ≈ 1, intercept ≈ 0, R² ≈ 1.
//! 2. **The frame is load-bearing**: pricing the SAME move through the
//!    internal-frame chord `δ_int = δ_raw ⊘ σ` (i.e. `D⁻¹ M D⁻¹`, the dropped-σ
//!    bug) drives the calibration slope materially off 1 under the asymmetric σ
//!    — the confound the fix removes is real, not cosmetic.
//! 3. **Target-dose loop**: [`steer_to_target_nats`] with an exact-KL probe
//!    converges to a requested in-radius dose within tolerance in ≤ a couple of
//!    probes and records a readout-KL radius; the probe-free closed-form seed
//!    reproduces the target dose exactly in the quadratic regime.
//!
//! The readout is the same closed-form categorical softmax the units test uses
//! (logits = raw activation vector, identity Jacobian ⇒ the output-Fisher metric
//! IS the categorical Fisher `diag(p) − ppᵀ` at that row's operating point), so
//! the "true KL" is computed exactly with no model in the loop. The term is a
//! genuine `build_sae_fit_seed` product with an installed periodic evaluator, so
//! `steer_delta` exercises the real decode / tangent / dose path, not a stub.

#[cfg(test)]
mod tests {
    use crate::inference::steering::{TargetDoseConfig, steer_delta, steer_to_target_nats};
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitSeedReport, SaeFitSeedRequest,
        SaeFisherRowMetricRequest, SaeManifoldTerm, SaeMinimalSeedReport, SaeMinimalSeedRequest,
        build_sae_fit_seed, build_sae_minimal_seed,
    };
    use gam_problem::RowMetric;
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
    use ndarray::{Array1, Array2, Array3, ArrayView1};
    use std::sync::Arc;

    const N_CIRCLE: usize = 48;
    const P_OUT: usize = 6;
    const NOISE_SIGMA: f64 = 0.02;

    /// Asymmetric Tier-0 per-column scale (max/min ≈ 6×): the σ that, if not
    /// un-scaled out of the decoder chord, mis-prices the dose per direction.
    const TIER0_SCALE: [f64; P_OUT] = [0.4, 2.5, 0.6, 1.8, 0.5, 2.2];

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn lcg_normal(state: &mut u64) -> f64 {
        let u1 = lcg(state).max(1e-12);
        let u2 = lcg(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn softmax(z: ArrayView1<'_, f64>) -> Vec<f64> {
        let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = z.iter().map(|&v| (v - max_z).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&v| v / sum).collect()
    }

    fn kl(p: &[f64], q: &[f64]) -> f64 {
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| if pi > 0.0 { pi * (pi / qi).ln() } else { 0.0 })
            .sum()
    }

    /// Closed-form categorical Fisher quadratic form `δᵀ(diag(p) − ppᵀ)δ =
    /// Σ pᵢδᵢ² − (Σ pᵢδᵢ)²`, computed independently of the packed `U` factor the
    /// metric is built from — the reference the shipped `predicted_nats` and the
    /// dropped-σ mispricing are both scored against.
    fn categorical_quad_form(p_probs: &[f64], delta: &[f64]) -> f64 {
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        for (&pi, &di) in p_probs.iter().zip(delta.iter()) {
            s1 += pi * di * di;
            s2 += pi * di;
        }
        s1 - s2 * s2
    }

    /// Pack the full-rank categorical Fisher factor stack `U ∈ ℝ^{n×p×p}` for a
    /// per-row logit matrix `z_raw` (row `n` = raw operating-point logits). Column
    /// `c` of `U_n` is `√p_c·(e_c − p)`, so `U_n U_nᵀ = diag(p) − ppᵀ = F(z_n)`
    /// exactly — a genuine (untruncated) raw-frame output-Fisher metric.
    fn categorical_fisher_metric(z_raw: &Array2<f64>) -> RowMetric {
        let n = z_raw.nrows();
        let p = z_raw.ncols();
        // (n, p*rank) row-major, U_n[i, c] at flat index i*p + c (rank = p).
        let mut flat = vec![0.0_f64; n * p * p];
        for row in 0..n {
            let probs = softmax(z_raw.row(row));
            for c in 0..p {
                let sqrt_pc = probs[c].sqrt();
                for i in 0..p {
                    let e_ci = if i == c { 1.0 } else { 0.0 };
                    flat[row * p * p + i * p + c] = sqrt_pc * (e_ci - probs[i]);
                }
            }
        }
        let u = Array2::from_shape_vec((n, p * p), flat).expect("U shape");
        RowMetric::output_fisher(Arc::new(u), p, p).expect("full-rank output-Fisher metric")
    }

    /// A smooth `p`-dim periodic embedding of the circle plus tiny deterministic
    /// noise — a nontrivial target the minimal seed fits a real periodic decoder
    /// to, so `decode_at_coords` is a genuine curved map (harmonics 1..3).
    fn circle_embedding_target() -> Array2<f64> {
        let mut state = 0x2249_0000_0000_0011u64;
        Array2::from_shape_fn((N_CIRCLE, P_OUT), |(i, j)| {
            let theta = std::f64::consts::TAU * (i as f64) / (N_CIRCLE as f64);
            let harmonic = (j / 2 + 1) as f64;
            let clean = if j % 2 == 0 {
                (harmonic * theta).cos()
            } else {
                (harmonic * theta).sin()
            };
            clean + NOISE_SIGMA * lcg_normal(&mut state)
        })
    }

    /// Build a genuine K=1 periodic-atom term (installed evaluator + decoder +
    /// fitted coords), then inject the asymmetric Tier-0 σ. The row metric is
    /// replaced afterward with the full categorical Fisher at each row's RAW
    /// operating point, so the term `steer_delta` reads is fully calibrated by
    /// construction.
    fn build_calibrated_term() -> (SaeManifoldTerm, RowMetric, Array1<f64>) {
        let target = circle_embedding_target();
        let assignment_kind = SaeFitAssignmentKind::Softmax;
        let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
            target: target.view(),
            atom_basis: vec!["periodic".to_string()],
            atom_dim: vec![1],
            assignment_kind,
            alpha: 1.0,
            tau: 1.0,
            threshold: 0.0,
            top_k: None,
            random_state: 0,
            initial_logits: None,
            initial_coords: None,
        })
        .expect("minimal seed");
        let SaeMinimalSeedReport {
            atom_basis,
            effective_atom_dim,
            atom_centers,
            basis_values,
            basis_jacobian,
            basis_sizes,
            decoder_coefficients,
            smooth_penalties,
            initial_logits,
            initial_coords,
            refine_routing,
        } = minimal;

        // A placeholder rank-1 output-Fisher so the seed installs a behavioral
        // metric; it is replaced below with the full categorical Fisher.
        let dummy_u = Array3::<f64>::from_shape_fn(
            (N_CIRCLE, P_OUT, 1),
            |(_, i, _)| if i == 0 { 1.0 } else { 0.0 },
        );
        let dummy_metric =
            SaeFisherRowMetricRequest::from_tag(dummy_u.view(), N_CIRCLE, P_OUT, None, None)
                .expect("placeholder metric request");

        let registry = AnalyticPenaltyRegistry::new();
        let seed = build_sae_fit_seed(SaeFitSeedRequest {
            target: target.view(),
            atom_basis: &atom_basis,
            atom_dim: &effective_atom_dim,
            atom_centers: &atom_centers,
            basis_values: basis_values.view(),
            basis_jacobian: basis_jacobian.view(),
            basis_sizes: &basis_sizes,
            decoder_coefficients: decoder_coefficients.view(),
            smooth_penalties: smooth_penalties.view(),
            initial_logits: initial_logits.view(),
            initial_coords: initial_coords.view(),
            alpha: 1.0,
            tau: 1.0,
            learnable_alpha: false,
            assignment_kind,
            sparsity_strength: 1.0,
            smoothness: 1.0,
            max_iter: 4,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            top_k: None,
            threshold: 0.0,
            native_ard_enabled: true,
            seed_refine_routing: refine_routing,
            seed_refine_random_state: 0,
            data_row_reseed: false,
            fit_config: SaeFitConfig::default(),
            temperature_schedule: None,
            fisher_metric: Some(dummy_metric),
            row_loss_weights: None,
            registry: &registry,
        })
        .expect("fit seed");
        let SaeFitSeedReport {
            base_term: mut term,
            ..
        } = seed;

        // Inject the asymmetric Tier-0 column scale: decode_at now returns
        // g_raw = σ ⊙ g_int, exactly the standardized-fit frame the confound
        // lived in.
        let scale = Array1::from_vec(TIER0_SCALE.to_vec());
        term.set_tier0_scale(scale.clone())
            .expect("inject asymmetric tier0 scale");

        // Raw operating-point logits at each row's fitted coordinate:
        // z_raw = σ ⊙ decode_int(t_row). Build the full categorical Fisher there.
        let coords = term.assignment.coords[0].as_matrix();
        let atom = &term.atoms[0];
        let mut z_raw = Array2::<f64>::zeros((N_CIRCLE, P_OUT));
        for row in 0..N_CIRCLE {
            let t = coords.row(row).to_vec();
            let t_mat = Array2::from_shape_vec((1, t.len()), t).expect("coord row");
            let g_int = atom
                .decode_at_coords(t_mat.view())
                .expect("decode at fitted coord");
            for j in 0..P_OUT {
                z_raw[[row, j]] = scale[j] * g_int[[0, j]];
            }
        }
        let metric = categorical_fisher_metric(&z_raw);
        term.set_row_metric(metric).expect("install calibrated metric");
        let metric = term.row_metric().expect("metric installed").clone();
        let angles: Array1<f64> = Array1::from_shape_fn(N_CIRCLE, |i| coords[[i, 0]]);
        (term, metric, angles)
    }

    /// Ordinary least squares `y = slope·x + intercept` with coefficient of
    /// determination R². Returns `(slope, intercept, r2)`.
    fn regress(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
        let n = xs.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let mut sxx = 0.0_f64;
        let mut sxy = 0.0_f64;
        let mut syy = 0.0_f64;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            sxx += (x - mean_x) * (x - mean_x);
            sxy += (x - mean_x) * (y - mean_y);
            syy += (y - mean_y) * (y - mean_y);
        }
        let slope = sxy / sxx;
        let intercept = mean_y - slope * mean_x;
        let mut ss_res = 0.0_f64;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let pred = slope * x + intercept;
            ss_res += (y - pred) * (y - pred);
        }
        let r2 = 1.0 - ss_res / syy;
        (slope, intercept, r2)
    }

    /// #2249 — the shipped `predicted_nats` calibrates against exact categorical
    /// KL through the asymmetric Tier-0 frame (slope→1, R²→1), and dropping the σ
    /// un-scaling (the pre-`ace3b9af3` bug) drives the slope materially off 1.
    #[test]
    fn shipped_predicted_nats_calibrates_through_tier0_frame() {
        let (term, metric, angles) = build_calibrated_term();
        let atom = &term.atoms[0];
        let tau = std::f64::consts::TAU;

        let mut predicted = Vec::new();
        let mut true_kl = Vec::new();
        let mut internal_dose = Vec::new();
        let scale = term.tier0_scale().expect("scale set").to_owned();

        let mut state = 0x2249_C0FFEE_u64;
        for row in 0..N_CIRCLE {
            let t_from = angles[row];
            let z_from: Vec<f64> = {
                let t_mat = Array2::from_shape_vec((1, 1), vec![t_from]).unwrap();
                let g = atom.decode_at_coords(t_mat.view()).unwrap();
                (0..P_OUT).map(|j| scale[j] * g[[0, j]]).collect()
            };
            let p_from = softmax(ArrayView1::from(&z_from));
            for _ in 0..6 {
                // Small edit: random direction, small magnitude, modest amplitude
                // — inside the quadratic/in-radius regime.
                let dt = (lcg(&mut state) - 0.5) * 0.03;
                let amp = 0.4 + 0.8 * lcg(&mut state);
                let t_to = (t_from + dt).rem_euclid(tau);
                let plan = steer_delta(&term, &metric, 0, row, amp, &[t_from], &[t_to])
                    .expect("steer_delta on calibrated term");
                let pred = plan.predicted_nats.expect("behavioral dose");
                let delta_raw = plan.delta.to_vec();
                let z_to: Vec<f64> = z_from
                    .iter()
                    .zip(delta_raw.iter())
                    .map(|(&z, &d)| z + d)
                    .collect();
                let p_to = softmax(ArrayView1::from(&z_to));
                let exact = kl(&p_from, &p_to);

                // Dropped-σ mispricing: price the internal-frame chord
                // δ_int = δ_raw ⊘ σ through the SAME raw-frame Fisher — i.e.
                // δ_rawᵀ D⁻¹ M D⁻¹ δ_raw, the pre-fix confound.
                let delta_int: Vec<f64> = delta_raw
                    .iter()
                    .zip(scale.iter())
                    .map(|(&d, &s)| d / s)
                    .collect();
                let mispriced = 0.5 * categorical_quad_form(&p_from, &delta_int);

                // The shipped dose must equal the closed-form categorical
                // quadratic on δ_raw to machine precision (M = F by construction).
                let closed = 0.5 * categorical_quad_form(&p_from, &delta_raw);
                assert!(
                    (pred - closed).abs() <= 1e-9 * closed.max(1e-30) + 1e-12,
                    "shipped predicted_nats {pred} must equal the closed-form categorical \
                     quadratic {closed} on the raw chord"
                );

                predicted.push(pred);
                true_kl.push(exact);
                internal_dose.push(mispriced);
            }
        }

        let (slope, intercept, r2) = regress(&predicted, &true_kl);
        let max_true = true_kl.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            slope > 0.95 && slope < 1.05,
            "in-frame calibration slope {slope} must be ≈1 (predicted_nats tracks true KL)"
        );
        assert!(
            intercept.abs() < 0.02 * max_true,
            "in-frame calibration intercept {intercept} must be ≈0 vs max KL {max_true}"
        );
        assert!(
            r2 > 0.99,
            "in-frame calibration R² {r2} must be ≈1 across edit directions"
        );

        // The frame is load-bearing: the dropped-σ dose does NOT calibrate.
        let (slope_int, _, _) = regress(&internal_dose, &true_kl);
        assert!(
            (slope_int - 1.0).abs() > 0.1,
            "dropped-σ (internal-frame) dose slope {slope_int} must be materially off 1 under \
             the asymmetric Tier-0 scale — the frame un-scaling is load-bearing, not cosmetic"
        );
    }

    /// #2263 — [`steer_to_target_nats`] lands a requested in-radius nats dose in a
    /// couple of exact-KL probes, records a readout-KL radius, and its probe-free
    /// closed-form seed reproduces the target dose exactly in the quadratic regime.
    #[test]
    fn target_dose_loop_lands_requested_nats() {
        let (term, metric, angles) = build_calibrated_term();
        let atom = &term.atoms[0];
        let tau = std::f64::consts::TAU;
        let scale = term.tier0_scale().expect("scale set").to_owned();

        let row = 3usize;
        let t_from = angles[row];
        let t_to = (t_from + 0.02).rem_euclid(tau);
        let z_from: Vec<f64> = {
            let t_mat = Array2::from_shape_vec((1, 1), vec![t_from]).unwrap();
            let g = atom.decode_at_coords(t_mat.view()).unwrap();
            (0..P_OUT).map(|j| scale[j] * g[[0, j]]).collect()
        };
        let p_from = softmax(ArrayView1::from(&z_from));

        // Unit-amplitude raw chord: dg_raw = g_raw(t_to) − g_raw(t_from).
        let dg_raw: Vec<f64> = {
            let t_mat = Array2::from_shape_vec((1, 1), vec![t_to]).unwrap();
            let g = atom.decode_at_coords(t_mat.view()).unwrap();
            (0..P_OUT)
                .map(|j| scale[j] * g[[0, j]] - z_from[j])
                .collect()
        };
        let unit_nats = 0.5 * categorical_quad_form(&p_from, &dg_raw);
        assert!(unit_nats > 0.0, "unit chord must carry Fisher mass");

        // Target a fraction of the unit dose so the amplitude stays in-radius.
        let target_nats = 0.5 * unit_nats;

        // Probe-free closed-form seed: predicted_nats must equal the target
        // exactly (a0² · unit_nats = q*).
        let seed_plan = steer_to_target_nats(
            &term,
            &metric,
            0,
            row,
            &[t_from],
            &[t_to],
            target_nats,
            TargetDoseConfig::default(),
            None,
        )
        .expect("closed-form seed");
        assert!(
            (seed_plan.predicted_nats - target_nats).abs() <= 1e-9 * target_nats,
            "closed-form seed dose {} must equal target {target_nats}",
            seed_plan.predicted_nats
        );
        let expect_a0 = (target_nats / unit_nats).sqrt();
        assert!(
            (seed_plan.seed_amplitude - expect_a0).abs() <= 1e-9 * expect_a0,
            "seed amplitude {} must be sqrt(q*/unit_nats) = {expect_a0}",
            seed_plan.seed_amplitude
        );
        assert!(!seed_plan.converged, "probe-free seed is unvalidated by construction");

        // Model-in-the-loop probe: exact categorical KL of the applied chord.
        let z_from_probe = z_from.clone();
        let dg_probe = dg_raw.clone();
        let p_from_probe = p_from.clone();
        let mut probe = move |a: f64| -> Result<f64, String> {
            let z_to: Vec<f64> = z_from_probe
                .iter()
                .zip(dg_probe.iter())
                .map(|(&z, &d)| z + a * d)
                .collect();
            let p_to = softmax(ArrayView1::from(&z_to));
            Ok(kl(&p_from_probe, &p_to))
        };
        let plan = steer_to_target_nats(
            &term,
            &metric,
            0,
            row,
            &[t_from],
            &[t_to],
            target_nats,
            TargetDoseConfig::default(),
            Some(&mut probe),
        )
        .expect("target-dose loop with exact-KL probe");

        assert!(plan.converged, "target-dose loop must converge on the exact-KL probe");
        let measured = plan.measured_nats.expect("measured dose");
        assert!(
            (measured - target_nats).abs() / target_nats <= 2.0e-2,
            "measured KL {measured} must land within 2% of target {target_nats}"
        );
        assert!(
            plan.iterations <= 4,
            "an in-radius target must converge in a couple of probes; took {}",
            plan.iterations
        );
        assert!(
            plan.readout_kl_radius.is_some(),
            "an in-radius probe must record a readout-KL radius"
        );
    }
}
