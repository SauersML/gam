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
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = Array2::zeros((n, P));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let (t, yi) = self.draw(&mut rng);
            x.row_mut(i).assign(&cosine_row(t));
            y[i] = yi;
        }
        let (t_star, y_star) = self.draw(&mut rng);
        (x, y, cosine_row(t_star), y_star)
    }
}

fn contains(set: &FullConformalSet, z: f64) -> bool {
    set.intervals.iter().any(|itv| itv.lo <= z && z <= itv.hi)
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
/// over the augmented domain, then the explicit augmented fit and its smoothed
/// residual rank `#{e_i > e_*} + U·(1 + #{e_i = e_*}) > α(n + 1)`.
#[allow(clippy::too_many_arguments)]
fn oracle_member(
    response: &GaussianRemlRhoResponse<'_>,
    x: &Array2<f64>,
    y: &Array1<f64>,
    s_lambda: &Array2<f64>,
    x_star: &Array1<f64>,
    z: f64,
    alpha: f64,
    tie_uniform: f64,
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
    let greater = scores.iter().filter(|&&e| e > test_score).count();
    let tied = scores.iter().filter(|&&e| e == test_score).count();
    greater as f64 + tie_uniform * (1.0 + tied as f64) > alpha * (x.nrows() + 1) as f64
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
    let honest = honest_full_conformal(x, y, &unit_weights(n), s_lambda, Some(1), x_star, alpha)
        .expect("honest set");
    assert_eq!(
        honest.certificate,
        ConformalCertificate::HonestRefit,
        "single-penalty Gaussian row must carry the honest certificate"
    );
    let frozen_engine = ExactGaussianFullConformal::new(x, y, &unit_weights(n), s_lambda, x_star)
        .expect("frozen engine");
    let tie_uniform = frozen_engine.tie_uniform();
    let frozen = frozen_engine.prediction_set(alpha);
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
        let truth = oracle_member(&response, x, y, s_lambda, x_star, z, alpha, tie_uniform);
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
            [(12usize, 11u64, 0.2), (20, 23, 0.1), (30, 37, 0.1), (40, 41, 0.2)]
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
        honest_full_conformal(&x, &y, &unit_weights(25), &(&s * scale), Some(1), &x_star, 0.1)
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
    let frozen = ExactGaussianFullConformal::new(&x, &y, &unit_weights(20), &s, &x_star)
        .expect("frozen")
        .prediction_set(0.1);
    for (count, expected) in [
        (None, ConformalCertificate::Refused(ConformalRefusal::UnknownPenaltyStructure)),
        (Some(2), ConformalCertificate::Refused(ConformalRefusal::MultiPenalty)),
        (Some(0), ConformalCertificate::ExactFrozen),
    ] {
        let row = honest_full_conformal(&x, &y, &unit_weights(20), &s, count, &x_star, 0.1)
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

/// One replicate's outcome for every row class at every `α`.
struct Replicate {
    /// `(certificate label, α index, covered)`.
    rows: Vec<(&'static str, usize, bool)>,
    honest_extra_refits: [usize; 2],
    honest_factorizations: [usize; 2],
}

fn replicate(scenario: Scenario, n: usize, seed: u64) -> Replicate {
    let (x, y, x_star, y_star) = scenario.sample(n, seed);
    let weights = unit_weights(n);
    let s = curvature_penalty();
    // The trained fit: λ̂ by REML on the training rows alone, stored as `Sλ`.
    // Where REML prefers the penalty's null-space model, λ̂ is the domain's
    // upper end, which the criterion descends through.
    let response = GaussianRemlRhoResponse::new(&x, &y, &s, &x_star).expect("response");
    let rho_hat = brute_force_rho(&response, None);
    let s_lambda = &s * rho_hat.exp();
    let mut rows = Vec::new();
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
            let row = honest_full_conformal(&x, &y, &weights, penalty, count, &x_star, alpha)
                .expect("conformal row");
            if count == Some(1) {
                honest_extra_refits[a] = row.cost.extra_refits;
                honest_factorizations[a] = row.cost.factorizations;
            }
            rows.push((row.certificate.label(), a, contains(&row.set, y_star)));
        }
    }
    Replicate {
        rows,
        honest_extra_refits,
        honest_factorizations,
    }
}

/// Coverage of every row class, `n ∈ {20, 50, 200}`, three misspecified noise
/// laws, `α ∈ {0.1, 0.05}` from the same replicates: `≥ 1 − α − 2·MCSE`, no row
/// excluded. The symmetric maps — the honest refit and the fixed penalty — are
/// exact with the smoothed p-value, so their coverage is also `≤ 1 − α +
/// 3·MCSE`; the refused rows freeze a λ̂ that saw `y` but not `y_*`, and only
/// the lower bound is claimed for them. Also reports the cost of the honest rows: one factorization each,
/// and the median number of local refits.
#[test]
fn full_conformal_coverage_holds_for_every_row_class() {
    let mut failures = Vec::new();
    let mut all_refits = Vec::new();
    for (s_index, scenario) in Scenario::ALL.into_iter().enumerate() {
        for (n_index, n) in [20usize, 50, 200].into_iter().enumerate() {
            let base = 1_000_003 * (1 + s_index as u64) + 10_007 * (1 + n_index as u64);
            let reps: Vec<Replicate> = (0..REPS as u64)
                .into_par_iter()
                .map(|r| replicate(scenario, n, base + r))
                .collect();
            let mut tallies: std::collections::BTreeMap<(&str, usize), (usize, usize)> =
                Default::default();
            for rep in &reps {
                for &(label, a, covered) in &rep.rows {
                    let entry = tallies.entry((label, a)).or_default();
                    entry.0 += 1;
                    entry.1 += usize::from(covered);
                }
                for a in 0..ALPHAS.len() {
                    assert_eq!(
                        rep.honest_factorizations[a], 1,
                        "an honest row refactorized the normal matrix"
                    );
                    all_refits.push(rep.honest_extra_refits[a]);
                }
            }
            for (&(label, a), &(rows, covered)) in &tallies {
                let alpha = ALPHAS[a];
                let coverage = covered as f64 / rows as f64;
                let mcse = (coverage * (1.0 - coverage) / rows as f64).sqrt();
                eprintln!(
                    "{scenario:?} n={n} α={alpha} {label}: coverage {coverage:.4} ± {mcse:.4} \
                     over {rows} rows"
                );
                if coverage < 1.0 - alpha - 2.0 * mcse {
                    failures.push(format!(
                        "{scenario:?} n={n} α={alpha} {label}: {coverage:.4} < {:.4}",
                        1.0 - alpha - 2.0 * mcse
                    ));
                }
                let symmetric = label == "honest_refit" || label == "exact_frozen";
                if symmetric && coverage > 1.0 - alpha + 3.0 * mcse {
                    failures.push(format!(
                        "{scenario:?} n={n} α={alpha} {label}: {coverage:.4} > {:.4}",
                        1.0 - alpha + 3.0 * mcse
                    ));
                }
            }
            // Every K = 1 row is honest: none silently fell back.
            for a in 0..ALPHAS.len() {
                let honest = tallies.get(&("honest_refit", a)).map_or(0, |t| t.0);
                assert_eq!(
                    honest, REPS,
                    "{scenario:?} n={n}: {} of {REPS} single-penalty rows were refused",
                    REPS - honest
                );
            }
        }
    }
    all_refits.sort_unstable();
    let median = all_refits[all_refits.len() / 2];
    eprintln!(
        "honest rows: median extra refits {median}, max {}, one factorization each",
        all_refits.last().copied().unwrap_or(0)
    );
    assert!(failures.is_empty(), "coverage outside its band: {failures:#?}");
    assert!(median <= 2, "median extra refits per honest row is {median}");
}
