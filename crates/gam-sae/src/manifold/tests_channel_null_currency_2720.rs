//! #2720 follow-up — the channel-null family's smoothness slope is carried by
//! the sub-floor singular value, a DATA-FIT currency the penalty never sees.
//!
//! The torus cell of the geometry sweep (PR #2772) measured the decoder
//! CHANNEL-null family carrying live smoothness slope — up to 4.47x the
//! convergence tolerance on 11 of 392 directions — while every other kind
//! measures flat. The landed #2720 resolution's own evidence table (the
//! `quotient_residual_norm_sq` doc in `fit_drivers.rs`) lists the channel-null
//! family at "—" directions: the flatness evidence never covered it, because
//! the landed sweep table could not seed its torus cell (`sphere,3` panic,
//! repaired on the #2772 branch).
//!
//! ## The two families are flat by two different arguments
//!
//! * **beta-null** (`joint_decoder_beta_null_directions`, fit_drivers.rs:2078):
//!   machine-null eigenvectors `v` of the PENALIZED Gram `G + lambda*S` with
//!   `G = D^T D` PSD. `v'(G + lambda*S)v = 0` with both terms PSD forces
//!   `v'Gv = 0` AND `v'Sv = 0`, so `Sv = 0` EXACTLY — the smoothness prior is
//!   structurally flat along the whole family, at every state, for every kind.
//!   No measurement needed; the control test below measures it anyway.
//!
//! * **channel-null** (`decoder_channel_null_directions`, fit_drivers.rs:2316):
//!   gated ONLY on the decoder's SVD — right-singular channels `c` with
//!   `sigma <= sqrt(SAE_DECODER_BETA_NULL_RELATIVE_FLOOR) * sigma_max`, i.e.
//!   `sigma <= sqrt(1e-9) * sigma_max ~ 3.16e-5 * sigma_max`. The penalty is
//!   never consulted at emission, and the outer-gradient solver's Rayleigh
//!   floor (construction_quasi_laplace.rs) — which DOES check candidates
//!   against the cached Hessian — is a different consumer: the convergence
//!   quotient (`posterior_null_quotient_basis`, fit_drivers.rs:2533) chains
//!   the RAW stream with no such screen. The smoothness value is
//!   `0.5*lambda*tr(B^T S B)` (construction.rs `decoder_smoothness_value`),
//!   so for the emitted direction `delta_beta = e_col (x) c` (channel `c` at
//!   basis row `col`, unit norm):
//!
//!   ```text
//!   d/d(eps) smoothness = lambda * (S B c)[col] = lambda * sigma * (S u)[col]
//!   ```
//!
//!   where `B c = sigma * u`. The slope is carried by the sub-floor singular
//!   value — a data-fit currency — times the roughness `S` charges on `u` — a
//!   penalty currency. It vanishes only if `sigma = 0` EXACTLY, which the
//!   floor does not require. The two currencies convert through `||S||_2`:
//!   the torus tensor-product roughness `(k^2 + l^2)^2` reaches `324` at the
//!   default per-axis order H=3, so the same sub-floor sigma can carry two
//!   orders of magnitude more slope on a torus atom than on a small-spectrum
//!   kind.
//!
//! ## What this probe pins
//!
//! 1. The measured smoothness slope along every RAW emitted channel-null
//!    direction equals the closed form `lambda * sigma * (S u)[col]` — the
//!    sub-floor singular value leaking through the smoothness Gram. Agreement
//!    is the proof; no other term contributes and nothing state-dependent is
//!    involved.
//! 2. The contrast: directions in `S`'s own nullspace — the flatness source
//!    the beta-null family inherits — measure flat through the same
//!    instrument on the same fixture.
//!
//! This probe does NOT adjudicate the fix (per-kind precondition at emission
//! vs Rayleigh floor at the quotient vs mover coverage): it pins the
//! mechanism any fix must address.

#![cfg(test)]
use super::*;

/// The closed form `lambda * sigma * (S u)[col]` for every channel-null
/// direction `e_col (x) c` the construction emits, recomputed independently
/// from the atom's own `S` and `B`. The channel IS a right-singular vector
/// with `||B c|| = sigma`, so no SVD call is needed: `u = B c / sigma`.
///
/// Returns `(col, sigma, closed_form)` in the construction's emission order.
fn channel_null_closed_forms(
    term: &SaeManifoldTerm,
    lambda_smooth: &[f64],
) -> Result<Vec<(usize, f64, f64)>, String> {
    let directions = term.decoder_channel_null_directions()?;
    let n = term.n_obs();
    let q = term.assignment.row_block_dim();
    let beta_base = n * q;
    let p = term.output_dim();
    let border_dim = term.factored_border_dim();
    let basis_rows = border_dim / p.max(1);
    let atom = &term.atoms[0];
    let b = atom.decoder_coefficients();
    let s = atom.smooth_penalty();
    let lambda = lambda_smooth
        .first()
        .copied()
        .ok_or("probe: no smooth scale")?;
    let mut out = Vec::with_capacity(directions.len());
    for direction in &directions {
        // Recover (col, c) from the direction's single populated basis row:
        // entries `beta_base + col*p .. beta_base + col*p + p`.
        let mut col = usize::MAX;
        let mut channel = Array1::<f64>::zeros(p);
        for slot in 0..basis_rows {
            let start = beta_base + slot * p;
            let block = direction.slice(s![start..start + p]);
            if block.iter().any(|&v| v != 0.0) {
                if col != usize::MAX {
                    return Err("probe: direction populates two basis rows".into());
                }
                col = slot;
                channel.assign(&block);
            }
        }
        if col == usize::MAX {
            return Err("probe: direction populates no basis row".into());
        }
        let bc = b.dot(&channel);
        let sigma = bc.dot(&bc).sqrt();
        let u = if sigma > 0.0 {
            bc / sigma
        } else {
            Array1::<f64>::zeros(bc.len())
        };
        let su = s.dot(&u);
        out.push((col, sigma, lambda * sigma * su[col]));
    }
    Ok(out)
}

/// Mechanism pin: the measured smoothness slope along every RAW channel-null
/// direction equals the closed form `lambda * sigma * (S u)[col]` — the
/// sub-floor singular value converted into penalty currency through `S`.
///
/// Fails if the closed form and the measurement disagree (the mechanism story
/// is wrong), or if no kind emits a direction with live slope (the symptom has
/// moved and this probe's subject matter is gone).
#[test]
fn channel_null_smoothness_slope_is_the_subfloor_sigma_leak_2720() {
    let z = tests_gauge_posterior_flatness_2720::planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let mut any_live_slope = false;
    let mut worst_closed_form_gap: f64 = 0.0;
    let mut compared = 0usize;
    for &(kind, latent_dim) in tests_gauge_posterior_flatness_2720::GAUGE_SWEEP_KINDS {
        let mut term =
            tests_gauge_posterior_flatness_2720::seeded_term_of_kind(z.view(), kind, latent_dim);
        let rho = SaeManifoldRho::new(
            0.0,
            0.0,
            vec![Array1::<f64>::zeros(term.assignment.coords[0].latent_dim())],
        );
        let lambda_smooth = rho
            .lambda_smooth_vec()
            .expect("the fixture rho carries one smoothing block per atom");
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        let directions = term.decoder_channel_null_directions().expect("emissions");
        let closed = channel_null_closed_forms(&term, &lambda_smooth).expect("closed forms");
        assert_eq!(
            directions.len(),
            closed.len(),
            "every emitted direction must carry a closed form"
        );
        if directions.is_empty() {
            println!("[2720-leak] kind={kind}: no channel-null direction emitted");
            continue;
        }
        let mut kind_live = 0usize;
        for (direction, &(col, sigma, closed_form)) in directions.iter().zip(closed.iter()) {
            // The emission is already unit-norm (`e_col (x) c` with `||c||=1`).
            let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
            let Some(direction) = tests_gauge_posterior_flatness_2720::unit_norm(direction.clone())
            else {
                continue;
            };
            let slope = tests_gauge_posterior_flatness_2720::directional_derivative_terms(
                &mut term,
                z.view(),
                &rho,
                &registry,
                &direction,
                1.0e-5,
            )
            .expect("directional derivative");
            compared += 1;
            let gap = (slope.smoothness - closed_form / norm).abs();
            worst_closed_form_gap = worst_closed_form_gap.max(gap);
            let live = slope.smoothness.abs() > tolerance;
            if live {
                kind_live += 1;
                any_live_slope = true;
            }
            if live || kind_live <= 3 {
                println!(
                    "[2720-leak] kind={kind} col={col} sigma={sigma:.3e} norm={norm:.3e} \
                     slope={:.6e} closed={:.6e} gap={gap:.2e} ({:.2}x tol) live={live}",
                    slope.smoothness,
                    closed_form / norm,
                    slope.smoothness.abs() / tolerance,
                );
            }
        }
        println!(
            "[2720-leak] kind={kind}: measured {} channel-null directions, \
             {kind_live} live (>{tolerance:.3e})",
            directions.len(),
        );
    }
    assert!(
        compared > 0,
        "no channel-null direction was emitted on any kind; the family this probe pins is \
         absent from the seeded fixture — re-measure before trusting the mechanism story"
    );
    assert!(
        any_live_slope,
        "no kind emitted a channel-null direction with live smoothness slope; the torus \
         symptom this probe pins has moved — re-measure before trusting the mechanism story"
    );
    // The closed form must track the measurement on every compared direction.
    // The central-difference instrument (h=1e-5 on unit directions) truncates
    // near its own roundoff; 1e-3 absolute is a generous band that still
    // rejects any competing mechanism at the 1e-3 scale.
    assert!(
        worst_closed_form_gap < 1.0e-3,
        "the closed form lambda*sigma*(S u)[col] disagrees with the measured smoothness \
         slope by {worst_closed_form_gap:.3e}; the sub-floor-sigma mechanism story is wrong"
    );
}

/// Control: directions in `S`'s own nullspace — the flatness source the
/// beta-null family inherits (`v'(G + lambda*S)v = 0` with PSD terms forces
/// `Sv = 0`) — measure flat through the same instrument on the same fixture.
///
/// Every roughness Gram in the repo annihilates at least the constant mode
/// (periodic/torus harmonic `(0, 0)`; Duchon polynomial nullspace; sphere
/// degree-0), so this control is guaranteed non-empty per kind and the
/// contrast with the channel-null leak is a FAMILY property, not a fixture
/// artifact.
#[test]
fn smoothness_gram_nullspace_measures_flat_2720() {
    let z = tests_gauge_posterior_flatness_2720::planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let mut worst = 0.0_f64;
    let mut measured = 0usize;
    for &(kind, latent_dim) in tests_gauge_posterior_flatness_2720::GAUGE_SWEEP_KINDS {
        let mut term =
            tests_gauge_posterior_flatness_2720::seeded_term_of_kind(z.view(), kind, latent_dim);
        let rho = SaeManifoldRho::new(
            0.0,
            0.0,
            vec![Array1::<f64>::zeros(term.assignment.coords[0].latent_dim())],
        );
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        let atom = &term.atoms[0];
        let s = atom.smooth_penalty();
        let m = s.nrows();
        let sym = 0.5 * (&s.to_owned() + &s.t().to_owned());
        let (evals, evecs) = sym
            .eigh(Side::Lower)
            .expect("roughness Gram eigendecomposition");
        let scale = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let nulls: Vec<usize> = (0..m)
            .filter(|&i| evals[i].abs() <= 1.0e-12 * scale.max(1.0))
            .collect();
        println!(
            "[2720-zero] kind={kind}: S is {m}x{m}, spectral scale {scale:.3e}, \
             {} null directions",
            nulls.len(),
        );
        for i in nulls {
            // Embed the null vector across every output channel, the same
            // layout `joint_decoder_beta_null_directions` uses, then measure
            // through the value function like every other arm here.
            let p = term.output_dim();
            let n = term.n_obs();
            let q = term.assignment.row_block_dim();
            let beta_base = n * q;
            let mut direction = Array1::<f64>::zeros(beta_base + term.factored_border_dim());
            for basis_col in 0..m {
                for out_col in 0..p {
                    direction[beta_base + basis_col * p + out_col] = evecs[[basis_col, i]];
                }
            }
            let Some(direction) = tests_gauge_posterior_flatness_2720::unit_norm(direction) else {
                continue;
            };
            let slope = tests_gauge_posterior_flatness_2720::directional_derivative_terms(
                &mut term,
                z.view(),
                &rho,
                &registry,
                &direction,
                1.0e-5,
            )
            .expect("directional derivative");
            measured += 1;
            worst = worst.max(slope.smoothness.abs());
            println!(
                "[2720-zero] kind={kind} null eigval={:.3e} smoothness slope {:.3e} \
                 ({:.2}x tol)",
                evals[i],
                slope.smoothness,
                slope.smoothness.abs() / tolerance,
            );
        }
    }
    assert!(
        measured > 0,
        "no roughness-Gram nullspace direction was measured"
    );
    assert!(
        worst < SAE_MANIFOLD_INNER_GRAD_REL_TOL,
        "an S-nullspace direction carries smoothness slope {worst:.3e}; the structural \
         flatness argument (S v = 0) is falsified through the value function"
    );
}
