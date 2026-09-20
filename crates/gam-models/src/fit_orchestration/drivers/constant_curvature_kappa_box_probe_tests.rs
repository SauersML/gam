// #2747: the curvature criterion must identify κ⋆ at a range it was NOT handed.
//
// This file replaces the #2687/#2716 probe module that mapped `V_p(κ)` across
// the admissible interval and reported it monotone. That question is answered:
// the descent was the RANGE coordinate leaking into `dκ` through the fill
// rule's slice, and both the map and the rail move with `ℓ`, not with the box.
// A probe whose question has an answer is a regression test or it is nothing,
// so the sweep it ran is now an assertion.
//
// The bar is deliberately a 3 × 3 grid rather than a single fixture. The
// pre-#2747 criterion is CORRECT on the diagonal cell where the truth's own
// radial length scale happens to equal the auto `ℓ_ref` — that is the cell the
// shipped acceptance fixture uses — so any gate built on one range cannot see
// the defect at all.
#[cfg(test)]
mod constant_curvature_kappa_range_identification_tests {
    use super::*;
    use gam_terms::basis::{
        CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    };

    fn next_unit(state: &mut u64) -> f64 {
        (gam_linalg::utils::splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_gauss(state: &mut u64) -> f64 {
        let u1 = next_unit(state).max(1.0e-12);
        let u2 = next_unit(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn spec_at(kappa: f64, centers: usize, length_scale: f64) -> ConstantCurvatureBasisSpec {
        ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint {
                num_centers: centers,
            },
            kappa,
            kappa_fixed: false,
            length_scale,
            length_scale_fixed: false,
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        }
    }

    /// `n` chart points in a radius-`radius` disk, and a response that is an
    /// exact member of the κ⋆ span AT `truth_ell` — so the truth is reachable at
    /// the planted curvature and at no other, and any failure is an estimator
    /// defect rather than misspecification.
    fn dataset_in_span(
        n: usize,
        kappa_star: f64,
        radius: f64,
        truth_ell: f64,
        centers: usize,
        noise_sd: f64,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut state = seed;
        let mut feats = Array2::<f64>::zeros((n, 2));
        let mut noise = Array1::<f64>::zeros(n);
        for i in 0..n {
            let (x1, x2) = loop {
                let a = 2.0 * next_unit(&mut state) - 1.0;
                let b = 2.0 * next_unit(&mut state) - 1.0;
                if a * a + b * b <= 1.0 {
                    break (a * radius, b * radius);
                }
            };
            feats[(i, 0)] = x1;
            feats[(i, 1)] = x2;
            noise[i] = next_gauss(&mut state);
        }
        let truth = gam_terms::basis::build_constant_curvature_basis(
            feats.view(),
            &spec_at(kappa_star, centers, truth_ell),
        )
        .expect("the planted κ⋆ geometry is inside its own chart");
        let design = truth.design.to_dense();
        let mut y = Array1::<f64>::zeros(n);
        for j in 0..design.ncols() {
            let w = 1.0 / (1.0 + j as f64);
            for i in 0..n {
                y[i] += w * design[(i, j)];
            }
        }
        let mean = y.iter().sum::<f64>() / n as f64;
        let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
        assert!(sd > 0.0, "the planted κ⋆ = {kappa_star} signal collapsed");
        for i in 0..n {
            y[i] = (y[i] - mean) / sd + noise_sd * noise[i];
        }
        (feats, y)
    }

    /// The auto range the builder picks for this cloud, and the κ box.
    fn seed_range_and_box(feats: &Array2<f64>, centers: usize) -> (f64, f64) {
        let spec = spec_at(0.0, centers, 0.0);
        let realized = gam_terms::basis::constant_curvature_realized_centers(feats.view(), &spec)
            .expect("realized centers");
        let ell = gam_terms::basis::realized_constant_curvature_length_scale(realized.view(), 0.0)
            .expect("auto range");
        let mut max_r2 = 0.0_f64;
        for row in feats.outer_iter().chain(realized.outer_iter()) {
            max_r2 = max_r2.max(row.dot(&row));
        }
        (ell, 0.5 / max_r2)
    }

    /// The bias bar, and where it comes from.
    ///
    /// `κ̂` at this fixture size is an estimate from `n = 240` rows at SNR 33
    /// through a 6-center basis, so it is not exact. Measured across the nine
    /// cells of this grid, the largest `|κ̂ − κ⋆|` is **0.19** and the median is
    /// 0.07; the realized `ℓ̂` recovers the planted range to within 3%.
    ///
    /// `0.45` is that maximum plus room for one grid step (0.116) of
    /// platform-to-platform floating-point drift in the argmin, and a little
    /// more. It is not a tolerance chosen to make the test pass — it is more
    /// than twice the observed error, because the failures this gate exists to
    /// catch are not fractions of a unit: the pre-#2747 criterion railed at
    /// `±1.41`, inverted the sign (`κ̂ = −0.35` against a planted `+1.0`), and
    /// read `∓0.94` on flat truth. The two sharp claims below — zero rails,
    /// correct sign in every cell — carry the load and admit no tolerance at
    /// all.
    const KAPPA_BIAS_BAR: f64 = 0.45;

    /// THE GATE. Three planted curvatures × three planted ranges. The
    /// range-profiled criterion must put its argmin near κ⋆ in every cell —
    /// interior, correct sign on the curved arms, and not curved at all on the
    /// flat one.
    ///
    /// Measured before #2747 on exactly this grid: the criterion was right in
    /// the `1×` column and nowhere else — railed at a box endpoint, sign
    /// inverted, or reporting a confident interior `κ̂ = ∓0.94` on genuinely
    /// FLAT data. Every one of those failures is a range error wearing a
    /// curvature's clothes.
    #[test]
    fn range_profiled_criterion_identifies_kappa_star_at_every_planted_range() {
        // Powered rather than cheap: `κ̂` at n = 120 / noise 0.10 has a sampling
        // spread of ±0.5, which would force a bar too loose to separate a fixed
        // estimator from a broken one. n = 240 at SNR 33 costs a fraction of a
        // second and brings the spread inside 0.31.
        let n = 240usize;
        let centers = 6usize;
        let radius = 0.6_f64;
        let seed = 0x5EED_2747_0000_0000_u64;
        const GRID: usize = 24;

        for &kappa_star in &[-1.0_f64, 0.0, 1.0] {
            for &range_mult in &[0.5_f64, 1.0, 2.0] {
                // The auto range is a property of the cloud, so read it from a
                // throwaway cloud drawn with the same stream before planting.
                let (probe_feats, _) = dataset_in_span(n, 0.0, radius, 1.0, centers, 0.0, seed);
                let (ell_ref, cap) = seed_range_and_box(&probe_feats, centers);
                let (feats, y) = dataset_in_span(
                    n,
                    kappa_star,
                    radius,
                    ell_ref * range_mult,
                    centers,
                    0.03,
                    seed,
                );

                let profile = ConstantCurvatureProfile::new(
                    feats.view(),
                    y.view(),
                    spec_at(0.0, centers, 0.0),
                )
                .expect("profile is constructible on the fixture");
                let mut best = (f64::INFINITY, f64::NAN);
                for i in 0..=GRID {
                    let kappa = -cap + 2.0 * cap * (i as f64) / (GRID as f64);
                    if let Ok((value, _, _)) = profile.evaluate(kappa)
                        && value < best.0
                    {
                        best = (value, kappa);
                    }
                }
                let step = 2.0 * cap / (GRID as f64);
                let (eta_hat, _, outcome) = profile
                    .minimize_over_eta(best.1)
                    .expect("the range box is searchable at the argmin");
                eprintln!(
                    "[#2747] κ⋆={kappa_star:+.2} range={range_mult}×ℓ_ref({ell_ref:.4}): \
                     κ̂={:+.4} (box ±{cap:.4}, step {step:.4}), ℓ̂={:.4} [{outcome:?}]",
                    best.1,
                    eta_hat.exp()
                );
                assert!(
                    best.1.abs() < cap * 0.999,
                    "κ⋆={kappa_star} at {range_mult}×ℓ_ref: κ̂={} is RAILED at the box endpoint \
                     ±{cap}; a railed κ̂ is a readout of the box, not of the data",
                    best.1
                );
                assert!(
                    (best.1 - kappa_star).abs() <= KAPPA_BIAS_BAR,
                    "κ⋆={kappa_star} at {range_mult}×ℓ_ref: κ̂={} is more than the derived bias \
                     bar {KAPPA_BIAS_BAR} from the planted curvature",
                    best.1
                );
                if kappa_star != 0.0 {
                    assert!(
                        best.1.signum() == kappa_star.signum(),
                        "κ⋆={kappa_star} at {range_mult}×ℓ_ref: κ̂={} has the WRONG SIGN — the \
                         geometry verdict is inverted",
                        best.1
                    );
                }
            }
        }
    }

    /// The cheap value path and the full jet must be the SAME criterion.
    ///
    /// The inner range solve brackets and line-searches on
    /// `constant_curvature_psi_profile_value` and refines on
    /// `constant_curvature_psi_profile_jet`, so the two build the bordered
    /// design/penalty pair independently. A divergence between them would not
    /// look like a failure — it would look like a line search that accepts steps
    /// the Newton's own objective rejects, i.e. a solve that wanders. Bit
    /// equality is the right bar: both call the same solver on the same
    /// matrices, so anything short of it means the matrices differ.
    #[test]
    fn the_bracket_value_and_the_jet_value_are_the_same_criterion() {
        let n = 200usize;
        let centers = 6usize;
        let seed = 0x5EED_2747_0000_0003_u64;
        let (probe_feats, _) = dataset_in_span(n, 0.0, 0.6, 1.0, centers, 0.0, seed);
        let (ell_ref, _) = seed_range_and_box(&probe_feats, centers);
        let (feats, y) = dataset_in_span(n, 0.7, 0.6, ell_ref, centers, 0.05, seed);
        for &kappa in &[-1.1_f64, -0.3, 0.0, 0.4, 1.1] {
            for mult in [0.3_f64, 1.0, 3.0] {
                let eta = (ell_ref * mult).ln();
                // A fresh profile per side, so neither answer can be served from
                // the other's cache.
                let by_value = ConstantCurvatureProfile::new(
                    feats.view(),
                    y.view(),
                    spec_at(0.0, centers, 0.0),
                )
                .expect("profile is constructible")
                .evaluate_value(kappa, eta)
                .expect("the value path evaluates");
                let by_jet = ConstantCurvatureProfile::new(
                    feats.view(),
                    y.view(),
                    spec_at(0.0, centers, 0.0),
                )
                .expect("profile is constructible")
                .evaluate_psi(kappa, eta)
                .expect("the jet path evaluates")
                .value;
                assert_eq!(
                    by_value,
                    by_jet,
                    "κ={kappa} ℓ={}: the bracket's criterion and the jet's criterion differ",
                    eta.exp()
                );
            }
        }
    }

    /// An explicit `length_scale=` PINS the range, exactly as an explicit
    /// `kappa=` pins the geometry — the same mgcv-`sp=` convention on the
    /// smooth's other coordinate.
    ///
    /// Pinned means pinned: the inner solve must report `Pinned` at every κ,
    /// hand back the user's value verbatim, and take the plain κ slice for the
    /// profile's derivatives, because a range the criterion may not move has
    /// `dη̂/dκ = 0` — by construction, not by assumption, which is exactly what
    /// distinguishes this outcome from `Uncertified` — and the envelope/Schur
    /// reduction does not apply to it. The resulting κ̂ is a conditional estimate
    /// and the report says so (`length_scale_estimated = false`).
    #[test]
    fn a_pinned_range_is_honoured_verbatim_and_never_profiled() {
        let n = 240usize;
        let centers = 6usize;
        let radius = 0.6_f64;
        let seed = 0x5EED_2747_0000_0002_u64;
        let (probe_feats, _) = dataset_in_span(n, 0.0, radius, 1.0, centers, 0.0, seed);
        let (ell_ref, _) = seed_range_and_box(&probe_feats, centers);
        let (feats, y) = dataset_in_span(n, 1.0, radius, ell_ref, centers, 0.03, seed);

        let pinned = ell_ref * 0.37;
        let mut spec = spec_at(0.0, centers, pinned);
        spec.length_scale_fixed = true;
        let profile = ConstantCurvatureProfile::new(feats.view(), y.view(), spec)
            .expect("profile is constructible with a pinned range");
        for &kappa in &[-1.0_f64, 0.0, 0.7] {
            let (eta_hat, jet, outcome) = profile
                .minimize_over_eta(kappa)
                .expect("a pinned range still evaluates");
            assert_eq!(
                outcome,
                RangeSolveOutcome::Pinned,
                "a pinned range is not an interior minimizer, and it is not a stalled solve \
                 either -- `dη̂/dκ = 0` holds by construction, which is what `Pinned` claims \
                 and `Uncertified` does not"
            );
            assert!(
                (eta_hat.exp() - pinned).abs() <= 1.0e-12 * pinned,
                "κ={kappa}: the pinned range must be honoured verbatim, got {}",
                eta_hat.exp()
            );
            let (value, first, second) = profile.evaluate(kappa).expect("profiled jet");
            assert_eq!(
                (value, first, second),
                jet.kappa_slice(),
                "κ={kappa}: a pinned range must take the plain κ slice"
            );
        }
    }

    /// The range coordinate must actually be estimated, and estimated WELL:
    /// `ℓ̂` at the criterion's own argmin must track the planted range across a
    /// factor of four, not sit at the heuristic seed.
    ///
    /// This is the half that separates "the optimizer moved η" from "η is
    /// identified". If `ℓ̂` were pinned at `ℓ_ref` the κ gate above would still
    /// be satisfiable by luck on one cell; it is not satisfiable on three.
    #[test]
    fn the_fitted_range_tracks_the_planted_range() {
        let n = 240usize;
        let centers = 6usize;
        let radius = 0.6_f64;
        let seed = 0x5EED_2747_0000_0001_u64;
        let (probe_feats, _) = dataset_in_span(n, 0.0, radius, 1.0, centers, 0.0, seed);
        let (ell_ref, _) = seed_range_and_box(&probe_feats, centers);

        let mut fitted = Vec::new();
        for &range_mult in &[0.5_f64, 1.0, 2.0] {
            let (feats, y) =
                dataset_in_span(n, 1.0, radius, ell_ref * range_mult, centers, 0.03, seed);
            let profile =
                ConstantCurvatureProfile::new(feats.view(), y.view(), spec_at(0.0, centers, 0.0))
                    .expect("profile is constructible on the fixture");
            let (eta_hat, _, _) = profile
                .minimize_over_eta(1.0)
                .expect("the range box is searchable at the planted κ");
            eprintln!(
                "[#2747 range] planted {:.4} ({range_mult}×ℓ_ref) -> ℓ̂ = {:.4}",
                ell_ref * range_mult,
                eta_hat.exp()
            );
            fitted.push(eta_hat.exp());
        }
        assert!(
            fitted[0] < fitted[1] && fitted[1] < fitted[2],
            "ℓ̂ must be strictly increasing in the planted range; got {fitted:?}"
        );
        // The realized ratio must recover the planted factor of two to within a
        // factor of two itself — a real identification claim, loose enough for
        // the n = 120 sampling error and tight enough to fail a pinned ℓ̂ (which
        // would give ratio 1.0 exactly).
        for (lo, hi) in [(0usize, 1usize), (1, 2)] {
            let ratio = fitted[hi] / fitted[lo];
            assert!(
                ratio > 1.25 && ratio < 4.0,
                "ℓ̂ ratio {ratio} across a planted factor of 2 ({:?}) does not identify the range",
                fitted
            );
        }
    }

    /// `V_p(κ)` may not depend on a range the search has ALREADY been to.
    ///
    /// The profile's box, bracket and seed are documented as DERIVED from the
    /// realized center set, and two of the three were. The seed was
    /// `spec.length_scale`, and `realized_constant_curvature_length_scale`
    /// returns an explicit positive value VERBATIM — falling back to the derived
    /// median only on the `0.0` auto sentinel. By the time the CI and the
    /// flatness LR build their profile, that field is no longer a request: both
    /// of the fit's write-backs have overwritten it with `ℓ̂`, the range this
    /// same criterion profiled to AT `κ̂`
    /// (`cc.length_scale = psi_hat.length_scale` in the free-κ enrollment, and
    /// `s.length_scale = *length_scale` in `freeze_term_collection_from_design`).
    ///
    /// `eta_seed` is pushed into the inner solve's candidate list beside the
    /// deterministic scan, so seeding it at `ℓ̂` is a warm start from ONE κ.
    /// `minimize_over_eta` states the rule it breaks: *"a profile likelihood
    /// that is not a function of its own argument cannot support an interval"*.
    /// The point estimate is the argmin of the criterion built with the auto
    /// seed; the interval and the LR are level sets of the criterion built with
    /// `ℓ̂`. Nothing makes `κ̂` stationary for the second one, and
    /// `curvature_profile_lr_endpoint` hard-errors — *"fitted curvature is not
    /// the minimum of its inference profile"* — exactly when it is not.
    ///
    /// This is the same argument the constructor already makes one field up
    /// about `identifiability`, whose frozen transform it un-freezes because it
    /// *"is the global identifiability frame realized at one particular fitted
    /// ψ"*. A fitted range is that too.
    ///
    /// The bar is BIT equality, not a tolerance. Two profiles that differ in the
    /// last place are two functions, and the failure this gate exists to catch —
    /// a search whose answer depends on where a previous search ended — shows up
    /// there first. The fixture is chosen so the injected seed is genuinely a
    /// different starting point: on hyperbolic truth at twice the auto range,
    /// `ℓ̂(κ)` moves by an order of magnitude across the κ box, so a seed pinned
    /// at one κ's answer is nowhere near the next κ's.
    #[test]
    fn the_profile_is_the_same_function_whatever_range_the_spec_arrives_carrying() {
        let n = 240usize;
        let centers = 6usize;
        let radius = 0.6_f64;
        let seed = 0x5EED_2747_0000_0004_u64;
        let (probe_feats, _) = dataset_in_span(n, 0.0, radius, 1.0, centers, 0.0, seed);
        let (ell_ref, cap) = seed_range_and_box(&probe_feats, centers);
        let (feats, y) = dataset_in_span(n, -1.0, radius, ell_ref * 2.0, centers, 0.03, seed);

        // What the FIT writes back: the range this criterion profiles to at one
        // κ. Taken at the box's hyperbolic end, which is where `ℓ̂` is furthest
        // from the auto rule.
        let derived =
            ConstantCurvatureProfile::new(feats.view(), y.view(), spec_at(0.0, centers, 0.0))
                .expect("profile is constructible from an auto-range spec");
        let anchor_kappa = -0.9 * cap;
        let (eta_at_anchor, _, _) = derived
            .minimize_over_eta(anchor_kappa)
            .expect("the range box is searchable at the anchor");
        let fitted_ell = eta_at_anchor.exp();
        assert!(
            (fitted_ell / ell_ref - 1.0).abs() > 0.1,
            "this gate needs an ℓ̂ that is NOT the auto rule to be a test at all; \
             got ℓ̂ = {fitted_ell} against ℓ_ref = {ell_ref}"
        );

        // The same spec as the fit hands to `curvature_inference_forspec`: the
        // realized range written back, `length_scale_fixed` still false because
        // the USER did not pin anything.
        let mut carried = spec_at(0.0, centers, fitted_ell);
        assert!(!carried.length_scale_fixed);
        carried.kappa = anchor_kappa;
        let replayed = ConstantCurvatureProfile::new(feats.view(), y.view(), carried)
            .expect("profile is constructible from a fitted spec");

        eprintln!(
            "[#2747 seed] ℓ_ref={ell_ref:.6} ℓ̂(κ={anchor_kappa:+.4})={fitted_ell:.6}  \
             box ±{cap:.4}"
        );
        for i in 0..=12u32 {
            let kappa = -cap + 2.0 * cap * f64::from(i) / 12.0;
            let a = derived.evaluate(kappa);
            let b = replayed.evaluate(kappa);
            match (a, b) {
                (Ok(a), Ok(b)) => assert_eq!(
                    a, b,
                    "κ={kappa}: V_p and its two derivatives moved when the spec arrived \
                     carrying ℓ̂={fitted_ell} instead of the auto sentinel — the profile is \
                     not a function of κ alone"
                ),
                (Err(a), Err(b)) => assert_eq!(
                    a.to_string(),
                    b.to_string(),
                    "κ={kappa}: the two profiles refuse for different reasons"
                ),
                (a, b) => {
                    panic!("κ={kappa}: one profile evaluated and the other refused: {a:?} vs {b:?}")
                }
            }
        }
    }

    /// The η-profile read off a jet taken AWAY from η's minimizer is the
    /// profile, not the κ slice beside it (gam#3426).
    ///
    /// The inner solve certifies η̂ by its Newton decrement against `1/(2n)`, so
    /// the jet `evaluate` reduces carries a residual `V_η ≠ 0`, and the plain
    /// envelope `V_p′ = V_κ` is then short by `V_κη·V_η/V_ηη` — measured at
    /// `1.5e-2` on the FD fixture below. On a quadratic `V` the Newton minimizer
    /// of the jet's quadratic in η IS the minimizer, so the reduction must return
    /// the closed-form profile exactly, at every η the jet is taken at.
    #[test]
    fn the_eta_profile_of_an_off_minimum_jet_is_the_profile() {
        // V(κ, η) = ½(a κ² + 2b κη + c η²) + g κ + f η, c > 0.
        let (a, b, c, g, f) = (3.0_f64, 1.7, 2.5, -0.4, 0.9);
        let v = |kappa: f64, eta: f64| {
            0.5 * (a * kappa * kappa + 2.0 * b * kappa * eta + c * eta * eta) + g * kappa + f * eta
        };
        // η*(κ) = −(bκ + f)/c, so V_p(κ) = V(κ, η*(κ)) with
        // V_p′ = (a − b²/c)κ + g − b f/c and V_p″ = a − b²/c.
        for kappa in [-0.7_f64, 0.0, 1.3] {
            let eta_star = -(b * kappa + f) / c;
            let exact = (
                v(kappa, eta_star),
                (a - b * b / c) * kappa + g - b * f / c,
                a - b * b / c,
            );
            for offset in [-0.3_f64, 0.0, 0.05, 0.8] {
                let eta = eta_star + offset;
                let jet = ProfiledRemlPsiJet {
                    value: v(kappa, eta),
                    rho_at_bound: false,
                    gradient: [a * kappa + b * eta + g, b * kappa + c * eta + f],
                    hessian: [[a, b], [b, c]],
                };
                let (value, first, second) =
                    jet.eta_profiled_kappa_jet().expect("V_ηη > 0 identifies η");
                let scale = 1.0 + exact.0.abs().max(exact.1.abs()).max(exact.2.abs());
                for (name, got, want) in [
                    ("V_p", value, exact.0),
                    ("V_p′", first, exact.1),
                    ("V_p″", second, exact.2),
                ] {
                    assert!(
                        (got - want).abs() <= 16.0 * f64::EPSILON * scale,
                        "κ={kappa}, η−η*={offset}: {name}={got:.17e} against the closed-form \
                         profile {want:.17e}"
                    );
                }
            }
        }
    }

    /// The PROFILED derivatives must be the derivatives of the PROFILED value.
    ///
    /// `constant_curvature_kappa_jet_fd_tests` differences `V(κ, η)` at fixed
    /// `η`. Nothing differences `V_p(κ) = min_η V(κ, η)`, and the reduction from
    /// one to the other is where the range coordinate's whole cost lands:
    /// `V_p′ = V_κ` by the envelope theorem (with the certified residual `V_η`
    /// carried, gam#3426), and `V_p″ = V_κκ − V_κη²/V_ηη` by one more
    /// differentiation. The Schur term is NON-POSITIVE, so a profile
    /// that fails to apply it does not produce a wrong fit — it produces an
    /// OVERSTATED curvature, which is what the outer solve's terminal
    /// stationarity certificate is denominated in (#2458). Exactly the class of
    /// defect that is invisible in the fitted numbers.
    ///
    /// And it was reachable. `minimize_over_eta` decided whether the reduction
    /// applied from a `converged` flag set by two of its several `break`s, so a
    /// backtracking search that exhausted BECAUSE the incumbent was already the
    /// minimum left the flag unset and the reduction unapplied. Six shipped
    /// fixtures were measured in that state, at `V_η` between `1.7e-6` and
    /// `1.9e-3` against `V_ηη` between `0.93` and `6.3e3` (gam#2747).
    ///
    /// So this gate asks the question in the form that does not depend on the
    /// classification at all: a central difference. It also asserts its own
    /// TEETH — that the unreduced `V_κκ` fails the same bar wherever the Schur
    /// term is material — because a gate on a correction is worthless at a
    /// fixture where the correction is zero.
    #[test]
    fn the_profiled_second_derivative_is_the_derivative_of_the_profiled_first() {
        let n = 200usize;
        let centers = 6usize;
        let radius = 0.6_f64;
        let seed = 0x5EED_2747_0000_0005_u64;
        let (probe_feats, _) = dataset_in_span(n, 0.0, radius, 1.0, centers, 0.0, seed);
        let (ell_ref, cap) = seed_range_and_box(&probe_feats, centers);
        let (feats, y) = dataset_in_span(n, 0.8, radius, ell_ref * 1.7, centers, 0.03, seed);
        let profile =
            ConstantCurvatureProfile::new(feats.view(), y.view(), spec_at(0.0, centers, 0.0))
                .expect("profile is constructible on the fixture");

        // `h` differences an ANALYTIC first derivative, whose only noise is the
        // inner solve's `η̂` wobble entering through `V_κη·δη`. A step three
        // orders inside the κ box keeps the difference in the smooth regime
        // while staying far above that floor.
        let h = 1.0e-3_f64 * cap;
        let mut interior_cells = 0usize;
        let mut toothy_cells = 0usize;
        for i in 0..=8u32 {
            let kappa = -0.8 * cap + 1.6 * cap * f64::from(i) / 8.0;
            let (_, jet, outcome) = profile
                .minimize_over_eta(kappa)
                .expect("the range box is searchable");
            if outcome != RangeSolveOutcome::InteriorMinimum {
                eprintln!("[#2747 profile-fd] κ={kappa:+.4}: {outcome:?}, not gated here");
                continue;
            }
            let (value, first, second) = profile.evaluate(kappa).expect("profiled jet");
            // The two difference points are ordinary profile evaluations and can
            // land on a κ whose own range solve is uncertified, where `evaluate`
            // refuses by design. That is not this gate's subject, so the cell is
            // skipped and the count below is what keeps skipping from becoming a
            // way to pass.
            let (Ok((plus, first_plus, _)), Ok((minus, first_minus, _))) =
                (profile.evaluate(kappa + h), profile.evaluate(kappa - h))
            else {
                eprintln!(
                    "[#2747 profile-fd] κ={kappa:+.4}: a difference point has no profile \
                     derivative, cell skipped"
                );
                continue;
            };
            interior_cells += 1;
            let fd_first = (plus - minus) / (2.0 * h);
            let fd_second = (first_plus - first_minus) / (2.0 * h);
            let schur = jet.hessian[0][1] * jet.hessian[0][1] / jet.hessian[1][1];
            let unreduced = second + schur;
            let rel_first = (first - fd_first).abs() / (1.0 + fd_first.abs());
            let rel_second = (second - fd_second).abs() / (1.0 + fd_second.abs());
            let rel_unreduced = (unreduced - fd_second).abs() / (1.0 + fd_second.abs());
            eprintln!(
                "[#2747 profile-fd] κ={kappa:+.4} V_p={value:.6} \
                 V_p′={first:+.6e} (fd {fd_first:+.6e}, rel {rel_first:.2e})  \
                 V_p″={second:+.6e} (fd {fd_second:+.6e}, rel {rel_second:.2e})  \
                 schur={schur:.6e} unreduced rel {rel_unreduced:.2e}"
            );
            assert!(
                rel_first <= 1.0e-4,
                "κ={kappa}: V_p′={first:.9e} against a central difference of V_p \
                 {fd_first:.9e} (rel {rel_first:.3e}); the envelope reduction does not \
                 differentiate the value beside it"
            );
            assert!(
                rel_second <= 1.0e-3,
                "κ={kappa}: V_p″={second:.9e} against a central difference of V_p′ \
                 {fd_second:.9e} (rel {rel_second:.3e}); the Schur reduction does not \
                 differentiate the first derivative beside it"
            );
            // Teeth. Where the Schur correction is material, the UNREDUCED
            // `V_κκ` — what a profile that misfiles an interior minimum
            // returns — must fail the same bar by an order of magnitude, so a
            // regression cannot slip through on a cell where the correction
            // happens to vanish.
            if rel_unreduced > 1.0e-2 {
                toothy_cells += 1;
            }
        }
        assert!(
            interior_cells >= 5,
            "only {interior_cells} of nine κ reached an interior range minimum; this gate \
             measures the reduction and cannot do it on a fixture that never applies one"
        );
        assert!(
            toothy_cells >= 1,
            "the Schur correction was immaterial at every one of {interior_cells} gated κ, so \
             the bars above would pass on a profile that never applied it"
        );
    }

    /// A `double_penalty=` term has no curvature estimate here, and the profile
    /// says so instead of scoring a different model.
    ///
    /// The criterion is a single-λ closed form. `double_penalty = true` makes
    /// the basis emit TWO active penalties — the RKHS Gram and a ridge `I` —
    /// which the fit gives two independent smoothing parameters. Both profile
    /// entry points forced the flag off, so `κ̂` and `ℓ̂` were selected against
    /// the one-penalty model while the fit realized the two-penalty one: an
    /// estimate for a model nobody fits, with a CI and a flatness p-value
    /// attached to it. Nothing anywhere reported the substitution.
    ///
    /// The pair below is the whole claim: the flag is the ONLY difference
    /// between the two specs, one is refused and the other is not, and the
    /// refusal names both ways out.
    #[test]
    fn a_double_penalty_term_is_refused_rather_than_scored_as_a_different_model() {
        let n = 120usize;
        let centers = 6usize;
        let seed = 0x5EED_2747_0000_0006_u64;
        let (feats, y) = dataset_in_span(n, 0.5, 0.6, 1.0, centers, 0.05, seed);

        let mut ridged = spec_at(0.0, centers, 0.0);
        ridged.double_penalty = true;
        let refusal = ConstantCurvatureProfile::new(feats.view(), y.view(), ridged)
            .expect_err("a two-penalty basis has no single-λ profile");
        let message = refusal.to_string();
        eprintln!("[#2747 double-penalty] {message}");
        // The flag that caused it, the arity that fails, and both ways out. A
        // refusal a caller cannot act on is a crash with better prose.
        for needle in ["double_penalty", "two penalties", "kappa="] {
            assert!(
                message.contains(needle),
                "the refusal must name the flag, the arity that fails, and a way out; \
                 {needle:?} is missing from {message:?}"
            );
        }

        // The control: the same spec without the flag builds, and it is the ONLY
        // difference between them, so the refusal cannot be blamed on the
        // fixture.
        ConstantCurvatureProfile::new(feats.view(), y.view(), spec_at(0.0, centers, 0.0))
            .expect("the default single-penalty term still profiles");
    }

    /// The κ route is held to the fit's own `tol`, like every other outer route.
    ///
    /// It used to declare `options.tol.max(√ε)`. The canonical fit tolerance is
    /// `1e-10`, and `√ε ≈ 1.49e-8` is larger, so the floor was always the value
    /// used: this one route quietly loosened the stationarity band by about 150x
    /// while every other route passed `tol` straight through. The exact
    /// `d²V_p/dκ²` and the declared size set what this route can resolve. A floor
    /// set by a machine constant does not.
    #[test]
    fn the_kappa_route_holds_the_fits_own_tolerance() {
        let n = 120usize;
        let centers = 6usize;
        let seed = 0x5EED_2747_0000_0007_u64;
        let (probe_feats, _) = dataset_in_span(n, 0.0, 0.6, 1.0, centers, 0.0, seed);
        let (_, cap) = seed_range_and_box(&probe_feats, centers);
        let (feats, y) = dataset_in_span(n, 0.5, 0.6, 1.0, centers, 0.05, seed);
        let profile =
            ConstantCurvatureProfile::new(feats.view(), y.view(), spec_at(0.0, centers, 0.0))
                .expect("profile is constructible on the fixture");
        for tol in [1e-10_f64, 1e-6] {
            let options = FitOptions {
                tol,
                ..FitOptions::default()
            };
            let problem = constant_curvature_kappa_problem(n, &profile, &options, -cap, cap);
            assert_eq!(
                problem.tolerance(),
                tol,
                "the κ problem must declare the fit's tol {tol:e} verbatim"
            );
        }
    }

    /// At the canonical `tol = 1e-10`, with the floor gone, the κ solve still
    /// reaches a converged optimum with an interior κ̂. The floor was not what
    /// made the route converge.
    #[test]
    fn the_kappa_solve_converges_at_the_canonical_tolerance() {
        let n = 240usize;
        let centers = 6usize;
        let radius = 0.6_f64;
        let seed = 0x5EED_2747_0000_0008_u64;
        let kappa_star = 0.7_f64;
        let (probe_feats, _) = dataset_in_span(n, 0.0, radius, 1.0, centers, 0.0, seed);
        let (ell_ref, cap) = seed_range_and_box(&probe_feats, centers);
        let (feats, y) = dataset_in_span(n, kappa_star, radius, ell_ref, centers, 0.03, seed);
        let profile =
            ConstantCurvatureProfile::new(feats.view(), y.view(), spec_at(0.0, centers, 0.0))
                .expect("profile is constructible on the fixture");
        let options = FitOptions {
            tol: 1e-10,
            ..FitOptions::default()
        };
        let optimum = solve_constant_curvature_kappa_profile(n, profile, &options, -cap, cap, 0)
            .expect("the κ solve must converge at the canonical tolerance");
        eprintln!(
            "[κ tol] planted κ⋆ = {kappa_star}, κ̂ = {:.6}, ℓ̂ = {:.6}, box = ±{cap:.4}",
            optimum.kappa, optimum.length_scale
        );
        assert!(
            optimum.kappa > -cap && optimum.kappa < cap,
            "κ̂ = {} must be interior to the box ±{cap}",
            optimum.kappa
        );
        assert!(optimum.length_scale.is_finite() && optimum.length_scale > 0.0);
    }
}
