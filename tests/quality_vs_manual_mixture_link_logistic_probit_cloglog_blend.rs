//! End-to-end quality: gam's *mixture inverse link* — a learnable convex blend
//! of the logit, probit, and cloglog inverse links — must reproduce, to
//! floating-point precision, the same blended mean that a hand-coded reference
//! computes from the three textbook link CDFs.
//!
//! No mature external tool *fits* a mixture/blend of inverse links (gamlss,
//! VGAM, mgcv each expose a single fixed link per fit), so a head-to-head fit
//! comparison does not exist. The honest, principled test of this niche
//! capability is therefore the *link-evaluation contract* itself:
//!
//!     mu(eta) = Σ_j π_j · g_j^{-1}(eta),
//!     π = softmax(ρ, last logit fixed at 0),   π_j ≥ 0,  Σ_j π_j = 1,
//!
//! with g_1^{-1}=logit, g_2^{-1}=probit, g_3^{-1}=cloglog. We evaluate gam's
//! `mixture_inverse_link_jet` over a realistic η grid (η = design·β from a cubic
//! B-spline basis on x ~ U(0,10)) and compare μ̂ against a hand-coded R
//! reference that forms the identical weighted sum of `plogis`, `pnorm`, and the
//! cloglog inverse `1 - exp(-exp(η))`. Because this is pure link arithmetic with
//! no fitting uncertainty on either side, the two must agree to ~machine
//! precision: rel_l2 < 1e-3 and pearson > 0.99999 are deliberately conservative
//! (gam achieves ~1e-15 here) yet still flag any real divergence in the blend.
//!
//! We additionally assert the simplex contract gam's softmax must satisfy
//! (π_j ≥ 0, Σπ_j = 1 to machine precision) and the identifiability convention
//! (free logits ρ recover the optimized weights: π_j/π_K = exp(ρ_j), with the
//! reference π built from the *same* ρ via plain softmax).

use gam::mixture_link::{mixture_inverse_link_jet, state_fromspec};
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::types::{LinkComponent, MixtureLinkSpec};
use ndarray::Array1;

#[test]
fn mixture_link_blend_matches_handcoded_reference() {
    // ---- target weights π = (0.40, 0.35, 0.25) over (logit, probit, cloglog) ----
    // gam's softmax fixes the LAST component's logit at 0, so the free logits are
    //   ρ_1 = ln(π_1/π_3) = ln(0.40/0.25) = ln(1.6)
    //   ρ_2 = ln(π_2/π_3) = ln(0.35/0.25) = ln(1.4)
    // which reproduce π exactly through softmax_last_fixedzero.
    let pi_target = [0.40_f64, 0.35, 0.25];
    let rho = Array1::from(vec![
        (pi_target[0] / pi_target[2]).ln(),
        (pi_target[1] / pi_target[2]).ln(),
    ]);
    let spec = MixtureLinkSpec {
        components: vec![
            LinkComponent::Logit,
            LinkComponent::Probit,
            LinkComponent::CLogLog,
        ],
        initial_rho: rho.clone(),
    };
    let state = state_fromspec(&spec).expect("construct mixture link state");

    // ---- simplex contract: π_j ≥ 0 and Σπ_j = 1 to machine precision ----------
    let pi = state.pi.to_vec();
    assert_eq!(pi.len(), 3, "three-component blend");
    let pi_sum: f64 = pi.iter().sum();
    for (j, &p) in pi.iter().enumerate() {
        assert!(p >= 0.0, "softmax weight π_{j} must be nonnegative, got {p}");
    }
    assert!(
        (pi_sum - 1.0).abs() < 1e-15,
        "softmax weights must sum to 1 to machine precision, got Σπ={pi_sum:.17e}"
    );
    // Recovered weights match the intended simplex point.
    for (j, (&got, &want)) in pi.iter().zip(pi_target.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "weight π_{j} mismatch: got {got:.17e} want {want:.17e}"
        );
    }
    // Identifiability: free logits recover the optimized weights, π_j/π_K=exp(ρ_j).
    let pi_k = pi[2];
    for (j, &r) in rho.iter().enumerate() {
        let ratio = pi[j] / pi_k;
        assert!(
            (ratio - r.exp()).abs() < 1e-12,
            "identifiability: π_{j}/π_K should equal exp(ρ_{j}); got {ratio:.17e} vs {:.17e}",
            r.exp()
        );
    }

    // ---- realistic η grid: η = design·β from a cubic B-spline basis (k=5) ------
    // x ~ U(0,10) on a deterministic grid; a fixed cubic-spline basis (5 columns)
    // with a fixed β yields a smoothly varying linear predictor that exercises
    // both tails of all three inverse links. This is identical data fed to both
    // engines: we emit η itself to R, so there is zero basis-convention drift —
    // the comparison is strictly about the link blend, not the basis.
    let n = 600usize;
    let x: Vec<f64> = (0..n).map(|i| 10.0 * (i as f64) / (n as f64 - 1.0)).collect();

    // Cubic (degree-3) B-spline-like basis with 5 columns over [0,10] using a
    // simple cardinal cubic on 5 equally spaced centers; this is only a vehicle
    // to generate a structured, wiggly η — its exact form is irrelevant because
    // η is shared verbatim with the reference.
    let centers = [1.0_f64, 3.25, 5.5, 7.75, 10.0];
    let scale = 2.25_f64;
    let beta = [1.4_f64, -2.1, 1.8, -1.6, 1.1];
    let eta: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mut e = -0.4; // intercept-like offset to spread η across (0,1) in μ
            for c in 0..5 {
                let z = (xi - centers[c]) / scale;
                // cubic radial-ish bump (cardinal cubic): max(0, 1-|z|)^3 weighting
                let t = (1.0 - z.abs()).max(0.0);
                e += beta[c] * t * t * t;
            }
            e
        })
        .collect();

    // ---- gam: μ̂ via the mixture inverse-link jet (pure link eval) -------------
    let gam_mu: Vec<f64> = eta
        .iter()
        .map(|&e| mixture_inverse_link_jet(&state, e).mu)
        .collect();

    // pointwise: a binomial mean from a convex blend of proper inverse links must
    // lie strictly in (0,1) for finite η.
    for (i, &m) in gam_mu.iter().enumerate() {
        assert!(
            m > 0.0 && m < 1.0,
            "mixture μ must lie in (0,1); μ[{i}]={m:.17e} at η={:.6}",
            eta[i]
        );
    }

    // ---- hand-coded R reference: weighted sum of three inverse-link CDFs -------
    // Feed the identical η and the identical weights; R recomputes the blend from
    // base R only (plogis, pnorm, and the closed-form cloglog inverse). No
    // packages required.
    let r = run_r(
        &[
            Column::new("eta", &eta),
            Column::new("pi1", &vec![pi[0]; n]),
            Column::new("pi2", &vec![pi[1]; n]),
            Column::new("pi3", &vec![pi[2]; n]),
        ],
        r#"
        # Inverse links on the linear predictor eta:
        #   logit^-1   = plogis(eta)        = 1/(1+exp(-eta))
        #   probit^-1  = pnorm(eta)
        #   cloglog^-1 = 1 - exp(-exp(eta))
        g_logit  <- plogis(df$eta)
        g_probit <- pnorm(df$eta)
        g_cll    <- 1 - exp(-exp(df$eta))
        mu <- df$pi1 * g_logit + df$pi2 * g_probit + df$pi3 * g_cll
        emit("mu", mu)
        emit("wsum", df$pi1[1] + df$pi2[1] + df$pi3[1])
        "#,
    );
    let ref_mu = r.vector("mu");
    let ref_wsum = r.scalar("wsum");
    assert_eq!(ref_mu.len(), n, "reference μ length mismatch");
    assert!(
        (ref_wsum - 1.0).abs() < 1e-12,
        "reference weights must form a simplex, Σπ={ref_wsum:.17e}"
    );

    // ---- compare ---------------------------------------------------------------
    let rel = relative_l2(&gam_mu, ref_mu);
    let corr = pearson(&gam_mu, ref_mu);
    eprintln!(
        "mixture-link blend (logit/probit/cloglog), n={n} π=({:.4},{:.4},{:.4}) \
         rel_l2={rel:.3e} pearson={corr:.12} μ∈[{:.4},{:.4}]",
        pi[0],
        pi[1],
        pi[2],
        gam_mu.iter().cloned().fold(f64::INFINITY, f64::min),
        gam_mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // Pure link arithmetic on both sides => agreement is limited only by f64
    // round-off (gam computes ~1e-15 here). 1e-3 / 0.99999 are far looser than
    // the achievable precision yet would catch any real blend/weight/link bug.
    assert!(
        rel < 1e-3,
        "mixture μ̂ diverges from hand-coded blend: rel_l2={rel:.3e}"
    );
    assert!(
        corr > 0.99999,
        "mixture μ̂ shape disagrees with hand-coded blend: pearson={corr:.12}"
    );
}
