//! Layer 3 against an independent brute force and a seeded coverage study.
//!
//! The oracle re-selects `ρ̂(z)` by scanning the closed-form REML criterion of
//! [`GaussianRemlRhoResponse`] — its own dense Cholesky per evaluation, no
//! eigenbasis, no bound — on a fine `z` grid, then refits the augmented model
//! explicitly and ranks its residuals. Test oracles may grid; production does
//! not.

use super::*;
use crate::inference::full_conformal::test_support::GaussianRemlRhoResponse;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, StudentT, Uniform};
use rayon::prelude::*;

const P: usize = 7;

fn cosine_row(t: f64) -> Array1<f64> {
    Array1::from_shape_fn(P, |j| (j as f64 * std::f64::consts::PI * t).cos())
}

/// The quartic-frequency curvature penalty on the cosine basis: the constant
/// and first harmonic are unpenalized (nullity 2), one smoothing parameter.
fn curvature_penalty() -> Array2<f64> {
    Array2::from_shape_fn((P, P), |(i, j)| {
        if i == j && i >= 2 { (i as f64).powi(4) } else { 0.0 }
    })
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    /// A step the cosine basis cannot represent, Gaussian noise.
    MisspecifiedMean,
    /// A smooth mean with Student-t noise of 2.5 degrees of freedom.
    HeavyTails,
    /// A smooth mean with noise sd rising five-fold across the design.
    Heteroscedastic,
}

impl Scenario {
    const ALL: [Scenario; 3] = [
        Scenario::MisspecifiedMean,
        Scenario::HeavyTails,
        Scenario::Heteroscedastic,
    ];

    fn draw(self, rng: &mut StdRng) -> (f64, f64) {
        let t = Uniform::new(0.0, 1.0).expect("uniform").sample(rng);
        let smooth = (2.0 * std::f64::consts::PI * t).sin();
        let y = match self {
            Scenario::MisspecifiedMean => {
                let step = if t < 0.4 { -1.0 } else { 1.5 };
                step + 0.4 * Normal::new(0.0, 1.0).expect("normal").sample(rng)
            }
            Scenario::HeavyTails => {
                smooth + 0.4 * StudentT::new(2.5).expect("student t").sample(rng)
            }
            Scenario::Heteroscedastic => {
                smooth + (0.1 + 0.5 * t) * Normal::new(0.0, 1.0).expect("normal").sample(rng)
            }
        };
        (t, y)
    }

    /// `n` training rows and one exchangeable test row `(x_*, y_*)`.
    fn sample(self, n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>, f64) {
        let (x, y, x_star, y_star, _) = self.sample_with_tie_uniform(n, seed);
        (x, y, x_star, y_star)
    }

    /// [`Self::sample`] plus the test row's smoothed-p-value uniform, drawn
    /// from the same seeded stream AFTER the rows: the rows are the draw
    /// `sample` makes, and the uniform is independent of them. Drawing it here
    /// rather than from an unseeded generator makes a replicate, and so a
    /// coverage cell, the same measurement on every run.
    fn sample_with_tie_uniform(
        self,
        n: usize,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>, f64, f64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = Array2::zeros((n, P));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let (t, yi) = self.draw(&mut rng);
            x.row_mut(i).assign(&cosine_row(t));
            y[i] = yi;
        }
        let (t_star, y_star) = self.draw(&mut rng);
        let tie_uniform = Uniform::new(0.0, 1.0).expect("uniform").sample(&mut rng);
        (x, y, cosine_row(t_star), y_star, tie_uniform)
    }
}

fn contains(set: &FullConformalSet, z: f64) -> bool {
    set.intervals.iter().any(|itv| itv.contains(z))
}

fn unit_weights(n: usize) -> Array1<f64> {
    Array1::ones(n)
}

/// The global REML minimizer of `response`'s criterion (`z = None`: the
/// training rows alone) over its closed domain, by a dense scan polished by
/// golden section. An end the criterion descends through is found like any
/// other minimizer.
fn brute_force_rho(response: &GaussianRemlRhoResponse<'_>, z: Option<f64>) -> f64 {
    let criterion = |rho: f64| {
        response
            .eval(rho, z)
            .map(|ev| ev.value)
            .unwrap_or(f64::INFINITY)
    };
    let (lo, hi) = if z.is_some() {
        response.augmented_rho_domain
    } else {
        response.rho_domain
    };
    let scan = 240usize;
    let step = (hi - lo) / scan as f64;
    let best = (0..=scan)
        .map(|k| (k, criterion(lo + step * k as f64)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("scan")
        .0;
    let (mut a, mut b) = (
        lo + step * best.saturating_sub(1) as f64,
        (lo + step * (best + 1) as f64).min(hi),
    );
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    let (mut c, mut d) = (b - ratio * (b - a), a + ratio * (b - a));
    let (mut fc, mut fd) = (criterion(c), criterion(d));
    for _ in 0..80 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - ratio * (b - a);
            fc = criterion(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + ratio * (b - a);
            fd = criterion(d);
        }
    }
    if fc < fd { c } else { d }
}

/// Brute-force membership of `z` in the honest set: the global REML minimizer
/// over the augmented domain, then the explicit augmented fit and its residual
/// rank.
fn oracle_member(
    response: &GaussianRemlRhoResponse<'_>,
    x: &Array2<f64>,
    y: &Array1<f64>,
    s_lambda: &Array2<f64>,
    x_star: &Array1<f64>,
    z: f64,
    alpha: f64,
) -> bool {
    let lambda = brute_force_rho(response, Some(z)).exp();
    let mut normal = x.t().dot(x) + s_lambda * lambda;
    for i in 0..P {
        for j in 0..P {
            normal[[i, j]] += x_star[i] * x_star[j];
        }
    }
    let rhs = x.t().dot(y) + x_star * z;
    let beta = normal
        .cholesky(Side::Lower)
        .expect("augmented normal matrix is SPD")
        .solvevec(&rhs);
    let test_score = (z - x_star.dot(&beta)).abs();
    let scores: Vec<f64> = x
        .rows()
        .into_iter()
        .zip(y.iter())
        .map(|(row, yi)| (*yi - row.dot(&beta)).abs())
        .collect();
    let greater = scores.iter().filter(|&&v| v > test_score).count();
    let tied = scores.iter().filter(|&&v| v == test_score).count();
    greater as f64 + 0.5 * (1 + tied) as f64 > alpha * (x.nrows() + 1) as f64
}

struct OracleReport {
    honest_mismatches: Vec<f64>,
    frozen_mismatches: usize,
    finite_endpoints: usize,
}

/// Compares the honest set, and the frozen Layer-1 set, with the oracle on a
/// 1601-point `z` grid covering every finite endpoint, plus far tail points.
/// Grid points within one spacing of an honest endpoint are the breakpoint
/// tolerance; no other point may disagree.
fn oracle_compare(
    x: &Array2<f64>,
    y: &Array1<f64>,
    s_lambda: &Array2<f64>,
    x_star: &Array1<f64>,
    alpha: f64,
) -> OracleReport {
    let n = x.nrows();
    let honest = honest_full_conformal_with_uniform(
        x,
        y,
        &unit_weights(n),
        s_lambda,
        Some(1),
        x_star,
        alpha,
        0.5,
    )
    .expect("honest set");
    assert_eq!(
        honest.certificate,
        ConformalCertificate::HonestRefit,
        "single-penalty Gaussian row must carry the honest certificate"
    );
    let frozen =
        ExactGaussianFullConformal::new_with_uniform(x, y, &unit_weights(n), s_lambda, x_star, 0.5)
            .expect("frozen engine")
            .prediction_set(alpha)
            .expect("prediction set");
    let response = GaussianRemlRhoResponse::new(x, y, s_lambda, x_star).expect("response");

    let center = honest.plug_in_mean;
    let endpoints: Vec<f64> = honest
        .set
        .intervals
        .iter()
        .flat_map(|itv| [itv.lo, itv.hi])
        .filter(|z| z.is_finite())
        .collect();
    let reach = endpoints
        .iter()
        .chain(frozen.intervals.iter().flat_map(|itv| [&itv.lo, &itv.hi]))
        .filter(|z| z.is_finite())
        .map(|z| (z - center).abs())
        .fold(1.0, f64::max);
    let half_width = 1.5 * reach;
    let points = 1601usize;
    let spacing = 2.0 * half_width / (points - 1) as f64;
    let mut honest_mismatches = Vec::new();
    let mut frozen_mismatches = 0;
    let grid = (0..points).map(|k| center - half_width + spacing * k as f64);
    let tails = [1.0e2, 1.0e4, 1.0e6]
        .into_iter()
        .flat_map(|scale| [center - scale * half_width, center + scale * half_width]);
    for z in grid.chain(tails) {
        let truth = oracle_member(&response, x, y, s_lambda, x_star, z, alpha);
        let near_breakpoint = endpoints.iter().any(|e| (z - e).abs() <= spacing);
        if contains(&honest.set, z) != truth && !near_breakpoint {
            honest_mismatches.push(z);
        }
        if contains(&frozen, z) != truth && !near_breakpoint {
            frozen_mismatches += 1;
        }
    }
    OracleReport {
        honest_mismatches,
        frozen_mismatches,
        finite_endpoints: endpoints.len(),
    }
}

/// The returned set equals the brute-force honest set up to breakpoint
/// resolution on every case, and the brute force separates the honest map
/// from the frozen-ρ set: the frozen set is refuted on a case where the
/// honest set is not.
#[test]
fn honest_set_matches_brute_force_reml_refits() {
    let cases: Vec<(Scenario, usize, u64, f64)> = Scenario::ALL
        .into_iter()
        .flat_map(|scenario| {
            [
                (12usize, 11u64, 0.2),
                (20, 23, 0.1),
                (30, 37, 0.1),
                (40, 41, 0.2),
            ]
            .into_iter()
            .map(move |(n, seed, alpha)| (scenario, n, seed, alpha))
        })
        .collect();
    let reports: Vec<_> = cases
        .par_iter()
        .map(|&(scenario, n, seed, alpha)| {
            let (x, y, x_star, _) = scenario.sample(n, seed);
            let report = oracle_compare(&x, &y, &curvature_penalty(), &x_star, alpha);
            (scenario, n, seed, alpha, report)
        })
        .collect();
    let mut frozen_refuted = 0;
    let mut finite = 0;
    for (scenario, n, seed, alpha, report) in &reports {
        eprintln!(
            "{scenario:?} n={n} seed={seed} α={alpha}: honest mismatches {}, frozen mismatches {}, \
             finite endpoints {}",
            report.honest_mismatches.len(),
            report.frozen_mismatches,
            report.finite_endpoints
        );
        assert!(
            report.honest_mismatches.is_empty(),
            "{scenario:?} n={n} seed={seed} α={alpha}: the honest set disagrees with brute-force \
             REML refits away from its breakpoints at z = {:?}",
            report.honest_mismatches
        );
        frozen_refuted += usize::from(report.frozen_mismatches > 0);
        finite += report.finite_endpoints;
    }
    assert!(finite > 0, "no case exercised a finite endpoint");
    assert!(
        frozen_refuted > 0,
        "the oracle never separated the honest map from the frozen-ρ set; it has no power"
    );
}

/// The stored strength is irrelevant: `ρ` is measured relative to `Sλ`, and
/// rescaling `Sλ` moves the domain with it. Exactly so in real arithmetic; in
/// `f64` a breakpoint is located only as finely as the criterion separates
/// strengths near its minimizer, `√(band/curvature)` in `ρ`, and a rescaled
/// domain bisects `ρ` at different points, so the retained cell around a
/// breakpoint may differ by that resolution — always on the side of keeping.
#[test]
fn honest_set_does_not_depend_on_the_stored_strength() {
    let (x, y, x_star, _) = Scenario::Heteroscedastic.sample(25, 5);
    let s = curvature_penalty();
    let set_at = |scale: f64| {
        honest_full_conformal_with_uniform(
            &x,
            &y,
            &unit_weights(25),
            &(&s * scale),
            Some(1),
            &x_star,
            0.1,
            0.5,
        )
        .expect("honest set")
    };
    let base = set_at(1.0);
    assert_eq!(base.certificate, ConformalCertificate::HonestRefit);
    for scale in [1.0e-3, 7.5, 1.0e4] {
        let other = set_at(scale);
        assert_eq!(other.certificate, ConformalCertificate::HonestRefit);
        assert_eq!(other.set.intervals.len(), base.set.intervals.len(), "scale {scale}");
        for (a, b) in other.set.intervals.iter().zip(&base.set.intervals) {
            for (u, v) in [(a.lo, b.lo), (a.hi, b.hi)] {
                assert!(
                    u == v || (u - v).abs() <= 1.0e-4 * (1.0 + v.abs()),
                    "scale {scale}: endpoint {u} vs {v}"
                );
            }
        }
    }
}

/// Every refusal is typed; the frozen set it returns is the Layer-1 set.
#[test]
fn unsupported_penalty_structures_are_refused_loudly() {
    let (x, y, x_star, _) = Scenario::HeavyTails.sample(20, 9);
    let s = curvature_penalty();
    let frozen =
        ExactGaussianFullConformal::new_with_uniform(&x, &y, &unit_weights(20), &s, &x_star, 0.5)
            .expect("frozen")
            .prediction_set(0.1)
            .expect("prediction set");
    for (count, expected) in [
        (
            None,
            ConformalCertificate::Refused(ConformalRefusal::UnknownPenaltyStructure),
        ),
        (
            Some(2),
            ConformalCertificate::Refused(ConformalRefusal::MultiPenalty),
        ),
        (Some(0), ConformalCertificate::ExactFrozen),
    ] {
        let row = honest_full_conformal_with_uniform(
            &x,
            &y,
            &unit_weights(20),
            &s,
            count,
            &x_star,
            0.1,
            0.5,
        )
        .expect("row");
        assert_eq!(row.certificate, expected, "penalty count {count:?}");
        assert_eq!(row.certificate.is_guaranteed(), count == Some(0));
        assert_eq!(row.set.intervals, frozen.intervals, "penalty count {count:?}");
    }
}

/// No probe-grid certificate or its wording survives in Layer 1–3's source.
#[test]
fn no_probe_grid_certificate_remains() {
    let sources = [
        ("full_conformal.rs", include_str!("../../full_conformal.rs")),
        ("honest.rs", include_str!("../honest.rs")),
    ];
    // Assembled at run time so this file does not match itself.
    let needles = [
        ["pro", "be"].concat(),
        ["grid-", "checked"].concat(),
        ["rho", "_lipschitz"].concat(),
        ["Lipschitz ", "assumption"].concat(),
        ["FrozenRho", "Certificate"].concat(),
        ["Certified", "FullConformal"].concat(),
        ["certified_", "full_conformal"].concat(),
        ["frozen_rho_", "certified"].concat(),
        ["boundary_", "margin"].concat(),
    ];
    for (name, source) in sources {
        for needle in &needles {
            assert!(
                !source.to_lowercase().contains(&needle.to_lowercase()),
                "{name} still carries the frozen-ρ certificate's `{needle}`"
            );
        }
    }
}

// ── Seeded Monte Carlo coverage ──────────────────────────────────────────

const REPS: usize = 2000;
const ALPHAS: [f64; 2] = [0.1, 0.05];

/// The coverage study's false-alarm budget: the probability that the whole
/// suite reports a failure when every Monte-Carlo-graded class covers at
/// exactly `1 − α`. Each graded tally is held to `1 − α − z·MCSE` with the
/// Bonferroni `z = Φ⁻¹(1 − δ/m)` over the `m` graded tallies, so the budget is
/// spent once across all of them rather than once per tally (#3338).
const COVERAGE_FALSE_ALARM_BUDGET: f64 = 0.01;

/// `m`: the Monte-Carlo-graded tallies in the whole study. Nine
/// `(scenario, n)` cells, both `α`, and one graded class, `honest_refit`;
/// `coverage_cell` refuses to grade any other class by Monte Carlo, so this
/// count cannot silently fall behind the classes it covers.
const MONTE_CARLO_TALLIES: usize = 9 * ALPHAS.len();

/// How a row class's coverage is graded (#3338).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Grading {
    /// A fixed map, symmetric in the augmented rows: graded exactly by
    /// [`exchangeability_identity`], with no sampling noise.
    Identity,
    /// A re-selecting map's conservative enclosure: graded by Monte Carlo
    /// against [`COVERAGE_FALSE_ALARM_BUDGET`].
    MonteCarlo,
    /// The frozen-ρ̂ set of a refused row, whose ρ̂ was selected on the
    /// training rows alone: its certificate claims no finite-sample coverage,
    /// so it is measured and reported, never held to `1 − α`.
    Reported,
}

fn grading(certificate: ConformalCertificate) -> Grading {
    match certificate {
        ConformalCertificate::ExactFrozen => Grading::Identity,
        ConformalCertificate::HonestRefit | ConformalCertificate::ConservativeFrozen => {
            Grading::MonteCarlo
        }
        ConformalCertificate::Refused(_) => Grading::Reported,
    }
}

/// One replicate's outcome for every row class at every `α`.
struct Replicate {
    /// `(certificate label, α index, covered, how the class is graded)`.
    rows: Vec<(&'static str, usize, bool, Grading)>,
    /// Per `α`: the exact-frozen map's `(covered folds, identity count)`.
    identity: [(usize, usize); 2],
    honest_extra_refits: [usize; 2],
    honest_factorizations: [usize; 2],
}

/// The exchangeability identity of a fixed penalty's map at `(α, u)`.
///
/// Hold out each of the `n + 1` augmented rows in turn and invert on the other
/// `n` with the same `u`. At its own held-out response every fold's augmented
/// data is the same `n + 1` rows, and the fixed-penalty fit is symmetric in
/// them, so every fold scores the same residuals. A fold is covered exactly
/// when its row's strict-dominator count `r` satisfies `r + u > α(n + 1)`, and
/// with distinct scores `r` takes each value in `0..=n` once. So the number of
/// covered folds is `#{r ∈ 0..=n : r + u > α(n + 1)}` exactly: the finite-sample
/// statement that `1 − α` coverage averages, checked with no sampling noise.
/// Returns `(covered folds, that count)`.
fn exchangeability_identity(
    x: &Array2<f64>,
    y: &Array1<f64>,
    x_star: &Array1<f64>,
    y_star: f64,
    penalty: &Array2<f64>,
    alpha: f64,
    tie_uniform: f64,
) -> Result<(usize, usize), String> {
    let n = x.nrows();
    let x_aug = Array2::from_shape_fn((n + 1, x.ncols()), |(i, j)| {
        if i < n { x[[i, j]] } else { x_star[j] }
    });
    let y_aug = Array1::from_shape_fn(n + 1, |i| if i < n { y[i] } else { y_star });
    let mut covered = 0;
    for fold in 0..=n {
        let keep: Vec<usize> = (0..=n).filter(|&i| i != fold).collect();
        let row = honest_full_conformal_with_uniform(
            &x_aug.select(ndarray::Axis(0), &keep),
            &y_aug.select(ndarray::Axis(0), &keep),
            &unit_weights(n),
            penalty,
            Some(0),
            &x_aug.row(fold).to_owned(),
            alpha,
            tie_uniform,
        )?;
        if row.certificate != ConformalCertificate::ExactFrozen {
            return Err(format!(
                "fold {fold}: a fixed penalty was certified {}",
                row.certificate.label()
            ));
        }
        covered += usize::from(contains(&row.set, y_aug[fold]));
    }
    let threshold = crate::inference::full_conformal::conformal_rank_threshold(alpha, n + 1);
    let count = (0..=n)
        .filter(|&r| r as f64 + tie_uniform > threshold)
        .count();
    Ok((covered, count))
}

/// The library's own failures are carried out rather than swallowed, so the run
/// that sees one names the replicate that produced it (#3394).
fn replicate(scenario: Scenario, n: usize, seed: u64) -> Result<Replicate, String> {
    let (x, y, x_star, y_star, tie_uniform) = scenario.sample_with_tie_uniform(n, seed);
    let weights = unit_weights(n);
    let s = curvature_penalty();
    // The trained fit: λ̂ by REML on the training rows alone, stored as `Sλ`.
    // Where REML prefers the penalty's null-space model, λ̂ is the domain's
    // upper end, which the criterion descends through.
    let response = GaussianRemlRhoResponse::new(&x, &y, &s, &x_star).expect("response");
    let rho_hat = brute_force_rho(&response, None);
    let s_lambda = &s * rho_hat.exp();
    let mut rows = Vec::new();
    let mut identity = [(0, 0); 2];
    let mut honest_extra_refits = [0; 2];
    let mut honest_factorizations = [0; 2];
    for (a, &alpha) in ALPHAS.iter().enumerate() {
        // K = 1: the honest map. K = 0: a fixed penalty, nothing re-selected.
        // K = 2, and a payload without the count: refused, frozen set.
        for (count, penalty) in [
            (Some(1), &s_lambda),
            (Some(0), &s),
            (Some(2), &s_lambda),
            (None, &s_lambda),
        ] {
            // One U per test row, shared by every α and penalty class, as
            // production draws it once per row.
            let row = honest_full_conformal_with_uniform(
                &x,
                &y,
                &weights,
                penalty,
                count,
                &x_star,
                alpha,
                tie_uniform,
            )
            .map_err(|error| format!("α={alpha} penalty_count={count:?}: {error}"))?;
            if count == Some(1) {
                honest_extra_refits[a] = row.cost.extra_refits;
                honest_factorizations[a] = row.cost.factorizations;
            }
            let grade = grading(row.certificate);
            if grade == Grading::Identity {
                identity[a] =
                    exchangeability_identity(&x, &y, &x_star, y_star, penalty, alpha, tie_uniform)
                        .map_err(|error| format!("α={alpha} identity: {error}"))?;
            }
            rows.push((
                row.certificate.label(),
                a,
                contains(&row.set, y_star),
                grade,
            ));
        }
    }
    Ok(Replicate {
        rows,
        identity,
        honest_extra_refits,
        honest_factorizations,
    })
}

/// One `(scenario, n)` cell of the coverage study, over the same replicates at
/// `α ∈ {0.1, 0.05}`, each class graded as [`Grading`] says (#3338): the
/// exact-frozen map by [`exchangeability_identity`] on every replicate, the
/// honest map's coverage `≥ 1 − α − z·MCSE` with `z` spending
/// [`COVERAGE_FALSE_ALARM_BUDGET`] over [`MONTE_CARLO_TALLIES`], and the
/// refused classes printed without a bar, since asserting one would assert a
/// guarantee the library disclaims. Also asserts the cost of this cell's
/// honest rows: one factorization each, and a median local-refit count of at
/// most two.
///
/// `seed_index` and `size_index` are the cell's position in
/// `Scenario::ALL × [20, 50, 200]`; they enter the seed exactly as they did
/// when the nine cells were one test, so every replicate is the same draw.
fn coverage_cell(scenario: Scenario, seed_index: usize, n: usize, size_index: usize) {
    let base = 1_000_003 * (1 + seed_index as u64) + 10_007 * (1 + size_index as u64);
    let reps: Vec<Replicate> = (0..REPS as u64)
        .into_par_iter()
        .map(|r| {
            let seed = base + r;
            replicate(scenario, n, seed)
                .map_err(|error| format!("{scenario:?} n={n} seed={seed}: {error}"))
        })
        .collect::<Result<Vec<Replicate>, String>>()
        .expect("every replicate of the coverage cell fits");
    let mut failures = Vec::new();
    let mut refits = Vec::new();
    let mut tallies: std::collections::BTreeMap<(&str, usize), (usize, usize)> = Default::default();
    let mut gradings = std::collections::BTreeMap::new();
    let mut identity_checks = 0;
    for (r, rep) in reps.iter().enumerate() {
        for (a, &(covered, count)) in rep.identity.iter().enumerate() {
            if covered != count {
                failures.push(format!(
                    "{scenario:?} n={n} α={} seed={}: exact_frozen covers {covered} of the \
                     {} held-out folds, the exchangeability identity says {count}",
                    ALPHAS[a],
                    base + r as u64,
                    n + 1
                ));
            }
        }
        for &(label, a, covered, grade) in &rep.rows {
            gradings.insert(label, grade);
            if grade == Grading::Identity {
                identity_checks += 1;
            }
            let entry = tallies.entry((label, a)).or_default();
            entry.0 += 1;
            entry.1 += usize::from(covered);
        }
        for a in 0..ALPHAS.len() {
            assert_eq!(
                rep.honest_factorizations[a], 1,
                "an honest row refactorized the normal matrix"
            );
            refits.push(rep.honest_extra_refits[a]);
        }
    }
    // The identity must have run on every replicate at both α, or the cell
    // graded nothing exactly.
    assert_eq!(
        identity_checks,
        REPS * ALPHAS.len(),
        "{scenario:?} n={n}: the fixed penalty's rows were not all exact_frozen"
    );
    let z = gam_math::probability::standard_normal_quantile(
        1.0 - COVERAGE_FALSE_ALARM_BUDGET / MONTE_CARLO_TALLIES as f64,
    )
    .expect("Bonferroni quantile");
    for (&(label, a), &(rows, covered)) in &tallies {
        let alpha = ALPHAS[a];
        let coverage = covered as f64 / rows as f64;
        let mcse = (coverage * (1.0 - coverage) / rows as f64).sqrt();
        let grade = gradings[label];
        eprintln!(
            "{scenario:?} n={n} α={alpha} {label}: coverage {coverage:.4} ± {mcse:.4} \
             over {rows} rows, graded {grade:?}"
        );
        if grade == Grading::MonteCarlo {
            assert_eq!(
                label, "honest_refit",
                "{label} is graded by Monte Carlo but MONTE_CARLO_TALLIES does not count it"
            );
            if coverage < 1.0 - alpha - z * mcse {
                failures.push(format!(
                    "{scenario:?} n={n} α={alpha} {label}: {coverage:.4} < {:.4} (z = {z:.3})",
                    1.0 - alpha - z * mcse
                ));
            }
        }
    }
    // Every K = 1 row is honest: none silently fell back. Collected over both
    // α rather than asserted inside the loop (#3394): a cell that refuses rows
    // at α = 0.1 AND at α = 0.05 used to report only the first, and the refusal
    // count is the quantity this bar is about.
    let mut refusals = Vec::new();
    for a in 0..ALPHAS.len() {
        let honest = tallies.get(&("honest_refit", a)).map_or(0, |t| t.0);
        if honest != REPS {
            refusals.push(format!(
                "{scenario:?} n={n} α={}: {} of {REPS} single-penalty rows were refused",
                ALPHAS[a],
                REPS - honest
            ));
        }
    }
    refits.sort_unstable();
    let median = refits[refits.len() / 2];
    eprintln!(
        "{scenario:?} n={n} honest rows: median extra refits {median}, max {}, one \
         factorization each",
        refits.last().copied().unwrap_or(0)
    );
    assert!(
        refusals.is_empty(),
        "single-penalty rows were refused instead of fitted: {refusals:#?}"
    );
    assert!(
        failures.is_empty(),
        "coverage failures (exchangeability identity, or below 1 − α − z·MCSE): {failures:#?}"
    );
    assert!(
        median <= 2,
        "{scenario:?} n={n}: median extra refits per honest row is {median}"
    );
}

// The study is nine independent `(scenario, n)` cells, and it is nine tests
// (#3338). As one test it was over nextest's per-test cap (600 s = slow-timeout
// 300 s × 2) from the day it landed -- PR #3213 reported about 870 s on 4 cores
// -- and a run that is killed at the cap has measured only the cells it reached:
// `direct-dy5-w2oc-3275-g1.log` got through 5 of the 9 before the kill, so four
// cells were asserting nothing at all. A per-test cap is a per-test budget, so
// the packaging is what was wrong: the cells share no state, the seeds are
// unchanged, and each one now carries its own verdict instead of being folded
// into one timeout. The refit median is also now asserted per cell, which is
// STRICTER than the single median over all nine that it replaces -- that one
// could be held under two by the cheap cells while an expensive cell drifted.
//
// This does not change the cost of an honest row. The z branch-and-bound is
// 94-97 % of the honest call at 1,000-2,100 cells and 35-121 ms per row, and
// every production row with `conformal_certificate = honest_refit` pays it;
// #3338's cell-count item is open, and the cap is not raised for it here.

/// [`exchangeability_identity`] on one replicate: the covered folds equal the
/// identity's count at both `α`, and the count moves with `α` (at `n = 20` the
/// two `α` give counts that differ for every `u`), so the equality is the
/// fold-by-fold rank statement and not a constant both sides happen to share.
#[test]
fn the_exchangeability_identity_holds_and_moves_with_alpha_3338() {
    let (x, y, x_star, y_star, tie_uniform) =
        Scenario::Heteroscedastic.sample_with_tie_uniform(20, 7);
    let s = curvature_penalty();
    let counts: Vec<usize> = ALPHAS
        .iter()
        .map(|&alpha| {
            let (covered, count) =
                exchangeability_identity(&x, &y, &x_star, y_star, &s, alpha, tie_uniform)
                    .expect("identity");
            assert_eq!(covered, count, "α={alpha}, U={tie_uniform}");
            assert!(0 < count && count <= 21, "α={alpha}: count {count}");
            count
        })
        .collect();
    assert_ne!(counts[0], counts[1], "U={tie_uniform}");
}

#[test]
fn full_conformal_coverage_misspecified_mean_n20() {
    coverage_cell(Scenario::MisspecifiedMean, 0, 20, 0);
}

#[test]
fn full_conformal_coverage_misspecified_mean_n50() {
    coverage_cell(Scenario::MisspecifiedMean, 0, 50, 1);
}

#[test]
fn full_conformal_coverage_misspecified_mean_n200() {
    coverage_cell(Scenario::MisspecifiedMean, 0, 200, 2);
}

#[test]
fn full_conformal_coverage_heavy_tails_n20() {
    coverage_cell(Scenario::HeavyTails, 1, 20, 0);
}

#[test]
fn full_conformal_coverage_heavy_tails_n50() {
    coverage_cell(Scenario::HeavyTails, 1, 50, 1);
}

#[test]
fn full_conformal_coverage_heavy_tails_n200() {
    coverage_cell(Scenario::HeavyTails, 1, 200, 2);
}

#[test]
fn full_conformal_coverage_heteroscedastic_n20() {
    coverage_cell(Scenario::Heteroscedastic, 2, 20, 0);
}

#[test]
fn full_conformal_coverage_heteroscedastic_n50() {
    coverage_cell(Scenario::Heteroscedastic, 2, 50, 1);
}

#[test]
fn full_conformal_coverage_heteroscedastic_n200() {
    coverage_cell(Scenario::Heteroscedastic, 2, 200, 2);
}

/// #3338. A comparison whose enclosure is strictly positive at one end of the
/// cell and strictly negative at the other has its crossing inside the cell,
/// and `certified_sign_change` returns that crossing.
///
/// The fixture is dyadic end to end — the line `s − ¼` over `[−1, 1]`, whose
/// endpoint values `−1¼` and `¾` and interpolation `−1 + 2·0.625` are all exact
/// in binary — so the returned root is the crossing bit for bit and the
/// assertion needs no band.
#[test]
fn a_certified_sign_change_is_returned_at_the_crossing_3338() {
    let line = Affine::exact(-0.25, 1.0);
    let (s1, s2) = (-1.0_f64, 1.0_f64);
    let root = certified_sign_change(&line, 0.1, 0.0, s1, s2)
        .expect("an enclosure that is negative at one end and positive at the other crosses");
    assert_eq!(root.to_bits(), 0.25_f64.to_bits());
    assert!(
        root > s1 && root < s2,
        "a returned split must be strictly interior, or it partitions nothing: {root}"
    );
    assert_eq!(line.value(root).to_bits(), 0.0_f64.to_bits());
}

/// #3338. Where the enclosure's own certified width reaches across an endpoint,
/// this cell has not established that the comparison changes sign in it, and the
/// rule declines rather than guessing a crossing. A same-signed enclosure
/// declines for the same reason.
#[test]
fn an_uncertified_crossing_returns_no_split_3338() {
    let line = Affine::exact(-0.25, 1.0);
    // `value(1) = 0.75` and the certified half-width is `0.8`: the enclosure
    // still contains zero at that end, so nothing is established.
    assert!(certified_sign_change(&line, 0.8, 0.0, -1.0, 1.0).is_none());
    // The evaluation's own rounding enters the same width, so a growth large
    // enough to cover the endpoint declines too.
    assert!(certified_sign_change(&line, 0.0, 0.7, -1.0, 1.0).is_none());
    // No crossing at all: both ends are strictly positive.
    let rising = Affine::exact(2.0, 1.0);
    assert!(certified_sign_change(&rising, 0.1, 0.0, -1.0, 1.0).is_none());
}

/// #3338. What the certified split buys, as arithmetic rather than as a
/// hypothesis. A midpoint descent reaches a crossing only by halving down to
/// the crossing's own resolution, which takes `log2(width / ulp(root))` cells —
/// the mantissa's 52 bits plus the cell's width in octaves — and each level
/// pays for the sibling next to the breakpoint as well. That is the
/// "about 40 halvings per breakpoint" the issue reads off the 1,000-2,100
/// cells a row examines. The certified split reaches the same point once.
#[test]
fn the_midpoint_descent_spends_one_cell_per_mantissa_bit_3338() {
    let line = Affine::exact(-0.25, 1.0);
    let (s1, s2) = (-1.0_f64, 1.0_f64);
    let root = certified_sign_change(&line, 0.1, 0.0, s1, s2).expect("a certified crossing");
    let ulp = root.next_up() - root;
    let halvings = ((s2 - s1) / ulp).log2();
    assert_eq!(
        halvings,
        (f64::MANTISSA_DIGITS - 1) as f64 + ((s2 - s1) / root.abs()).log2(),
        "the depth of a midpoint descent is the mantissa plus the cell's width in octaves"
    );
    assert!(
        halvings > (f64::MANTISSA_DIGITS - 1) as f64,
        "reaching this crossing by halving costs more than a mantissa of cells, got {halvings}"
    );
}
