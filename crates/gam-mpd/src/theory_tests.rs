#![cfg(test)]
//! Machine checks of the theorems in `theory`: each test names the theorem it exercises, and
//! every tolerance is derived (a band, an exact rational sum, a stated confidence).

use super::bounds::{kl_over_logit_boxes, kl_supremum_over_logit_boxes, total_variation_over_logit_boxes};
use super::codec::{prefix_integer_len_bits, signed_prefix_integer_len_bits, subset_code_len_bits};
use super::compile::null::{NullEditOutcome, physically_null_supremum};
use super::secant::BandedMatrix;
use super::supports::EvidenceStatus;
use super::theory::{
    ClaimVerdict, CodedCandidate, IntegerCode, adversary_advantage_upper, claim_robustness, exact_classes,
    quotient_consistency,
};
use super::verify::{certified_argmax, computed_argmax};
use gam_linalg::roundoff::accumulation_growth;
use gam_math::categorical::categorical_kl_from_logits_with_error;
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// `Σ_len counts[len] 2^{−len}` as the exact numerator over `2^{top}`.
fn kraft_numerator(counts: &[u128], top: usize) -> u128 {
    counts.iter().enumerate().map(|(len, count)| count << (top - len)).sum()
}

/// Lemma 1b's component codes: each integer code of the family has Kraft sum at most 1 over the
/// integers below `2^20`, exactly (a rational sum), and Elias γ's partial sum is `1 − 2^{−20}`.
#[test]
fn the_integer_codes_satisfy_kraft() {
    let top = 96;
    for code in IntegerCode::FAMILY {
        let mut counts = vec![0u128; top + 1];
        for n in 1..(1u64 << 20) {
            counts[code.len_bits(n).expect("length") as usize] += 1;
        }
        let numerator = kraft_numerator(&counts, top);
        assert!(numerator <= 1u128 << top, "{code:?}: Kraft sum above 1");
        if code == IntegerCode::EliasGamma {
            assert_eq!(numerator, (1u128 << top) - (1u128 << (top - 20)), "γ's partial sum is 1 − 2^-20");
        }
    }
    // The family's ω is the codec's own code, signed through the codec's own map.
    for n in [1u64, 2, 3, 4, 7, 8, 16, 17, 1 << 20, u64::MAX] {
        assert_eq!(IntegerCode::EliasOmega.len_bits(n).expect("ω"), prefix_integer_len_bits(n).expect("codec"));
    }
    for v in [0i64, 1, -1, 2, -2, 8, -32768, i64::MAX] {
        assert_eq!(IntegerCode::EliasOmega.signed_len_bits(v).expect("ω"), signed_prefix_integer_len_bits(v).expect("codec"));
    }
}

/// Lemma 1b: the enumerative subset code has Kraft sum at most 1 over every subset of an
/// `n`-element universe.
#[test]
fn the_subset_code_satisfies_kraft() {
    let top = 64;
    for universe in 0..=14usize {
        let mut counts = vec![0u128; top + 1];
        let mut binomial = 1u128;
        for k in 0..=universe {
            counts[subset_code_len_bits(universe, k).expect("length") as usize] += binomial;
            binomial = binomial * (universe - k) as u128 / (k + 1) as u128;
        }
        assert!(kraft_numerator(&counts, top) <= 1u128 << top, "universe {universe}");
    }
}

/// Theorem 2e: the signed Elias δ and γ codes are strictly subadditive on nonzero pairs, and the
/// codec's signed Elias ω is not (the jump counterexamples, up to its 2-bit gap).
#[test]
fn elias_delta_and_gamma_are_subadditive_and_omega_is_not() {
    let reach = 1i64 << 10;
    for code in [IntegerCode::EliasDelta, IntegerCode::EliasGamma] {
        let length = |v: i64| code.signed_len_bits(v).expect("length");
        for a in -reach..=reach {
            for b in -reach..=reach {
                if a != 0 && b != 0 {
                    assert!(length(a + b) < length(a) + length(b), "{code:?}: {a} + {b}");
                }
            }
        }
        // The large-argument step of the proof: ℓ(4M + 1) ≤ ℓ(2M) + (3 for δ, 2 for γ).
        let step = if code == IntegerCode::EliasDelta { 3 } else { 2 };
        for shift in 0..60 {
            for m in [(1u64 << shift).max(1), (1u64 << shift) + 1, (3u64 << shift) / 2 + 1] {
                let (wide, base) = (code.len_bits(4 * m + 1).expect("ℓ"), code.len_bits(2 * m).expect("ℓ"));
                assert!(wide <= base + step, "{code:?}: M = {m}");
            }
        }
    }
    let omega = |v: i64| IntegerCode::EliasOmega.signed_len_bits(v).expect("length");
    assert_eq!((omega(8), omega(7), omega(1)), (11, 7, 3));
    assert_eq!((omega(-32768), omega(-32767), omega(-1)), (28, 23, 3));
    let worst = (-(1i64 << 16)..=(1 << 16))
        .flat_map(|a| [-1i64, 1].map(move |b| (a, b)))
        .map(|(a, b)| omega(a + b) as i64 - omega(a) as i64 - omega(b) as i64)
        .max()
        .expect("pairs");
    assert_eq!(worst, 2, "ω's worst excess over subadditivity");
    // The operator of Theorem 2e: k entries of 8 beside one 1, against its split.
    let k = 100u64;
    let merged = k * omega(8) + omega(1);
    let split = k * omega(7) + omega(1) + k * omega(1) + omega(0);
    assert_eq!((merged, split), (1103, 1004));
}

/// Theorem 6.2 and 6.3: a claim robust over a family is proven under the mixture code and under
/// every code within half its margin; a code-dependent claim is reported with its codes.
#[test]
fn a_robust_claim_survives_the_mixture_and_nearby_codes() {
    let mut rng = StdRng::seed_from_u64(29_51);
    let mut checked = [0usize; 3];
    for _ in 0..4000 {
        let (candidates, codes) = (rng.random_range(2..7usize), rng.random_range(1..5usize));
        let claim: Vec<bool> = (0..candidates).map(|_| rng.random_range(0..2) == 1).collect();
        let family: Vec<CodedCandidate> = (0..candidates)
            .map(|_| {
                let lower = rng.random_range(0.0..40.0);
                CodedCandidate {
                    program_bits: (0..codes).map(|_| rng.random_range(10..60u64)).collect(),
                    data_bits_lower: lower,
                    data_bits_upper: lower + rng.random_range(0.0..0.5),
                }
            })
            .collect();
        let verdict = claim_robustness(&family, &claim).expect("verdict");
        match verdict {
            ClaimVerdict::Robust { margin_bits } => {
                checked[0] += 1;
                // The mixture: min over codes plus the selector.
                let selector = (usize::BITS - (codes - 1).leading_zeros()) as u64;
                let mixture: Vec<CodedCandidate> = family
                    .iter()
                    .map(|c| CodedCandidate {
                        program_bits: vec![c.program_bits.iter().copied().min().expect("codes") + selector],
                        ..c.clone()
                    })
                    .collect();
                assert!(matches!(claim_robustness(&mixture, &claim).expect("mixture"), ClaimVerdict::Robust { .. }));
                // A code within less than half the margin of code 0, on every candidate.
                let shift = (margin_bits / 2.0).floor() as i64 - 1;
                if shift >= 1 {
                    let nearby: Vec<CodedCandidate> = family
                        .iter()
                        .zip(&claim)
                        .map(|(c, holds)| {
                            let moved = if *holds { c.program_bits[0] + shift as u64 } else { c.program_bits[0] - shift.min(c.program_bits[0] as i64) as u64 };
                            CodedCandidate { program_bits: vec![moved], ..c.clone() }
                        })
                        .collect();
                    assert!(matches!(claim_robustness(&nearby, &claim).expect("nearby"), ClaimVerdict::Robust { .. }));
                }
            }
            ClaimVerdict::CodeDependent { supporting, refuting, .. } => {
                checked[1] += 1;
                assert!(!supporting.is_empty() && !refuting.is_empty());
                assert!(supporting.iter().all(|c| !refuting.contains(c)));
            }
            _ => checked[2] += 1,
        }
    }
    assert!(checked.iter().all(|count| *count > 50), "every verdict kind occurs: {checked:?}");
}

/// Theorem 7 on random quotients: when the next state is a function of the class under every
/// intervention (plus rounding inside the bands) the check never refutes and the representative
/// update reproduces every row within `ε`; when one intervention reads a coordinate the summary drops,
/// the check refutes at that intervention with a same-class pair, even though the clean run passes.
#[test]
fn quotient_consistency_decides_whether_an_abstract_update_exists() {
    let mut rng = StdRng::seed_from_u64(0xb15);
    for trial in 0..200 {
        let (rows, classes_count, width) = (rng.random_range(4..40usize), rng.random_range(1..6usize), rng.random_range(1..4usize));
        let classes: Vec<usize> = (0..rows).map(|_| rng.random_range(0..classes_count)).collect();
        let hidden: Vec<f64> = (0..rows).map(|_| rng.random_range(-1.0..1.0)).collect();
        let table: Vec<Array2<f64>> =
            (0..3).map(|_| Array2::from_shape_simple_fn((classes_count, width), || rng.random_range(-2.0..2.0))).collect();
        let band = [0.0, 1e-12, 1e-6][trial % 3];
        let state = |u: usize, reads_hidden: bool, rng: &mut StdRng| BandedMatrix {
            values: Array2::from_shape_fn((rows, width), |(r, c)| {
                let jitter = if band > 0.0 { rng.random_range(-band..band) } else { 0.0 };
                table[u][[classes[r], c]] + jitter + if reads_hidden { hidden[r] } else { 0.0 }
            }),
            bands: Array2::from_elem((rows, width), band),
        };
        let consistent: Vec<BandedMatrix> = (0..3).map(|u| state(u, false, &mut rng)).collect();
        let status = quotient_consistency(&classes, &consistent).expect("checks");
        let EvidenceStatus::Exact { value, numerical_error, .. } = status else { panic!("trial {trial}: {status:?}") };
        assert!(value <= numerical_error, "trial {trial}: ε = {value} beyond its band {numerical_error}");
        // The constructive direction: a representative per class predicts every row within ε + band.
        for (u, next) in consistent.iter().enumerate() {
            for r in 0..rows {
                let representative = classes.iter().position(|c| *c == classes[r]).expect("a representative");
                for c in 0..width {
                    let gap = (next.values[[r, c]] - next.values[[representative, c]]).abs();
                    assert!(gap <= value + numerical_error, "trial {trial}, u {u}");
                }
            }
        }
        // Intervention 2 reads the dropped coordinate: refuted there, with a same-class witness.
        let mut broken = consistent.clone();
        broken[2] = state(2, true, &mut rng);
        let same_class_rows = (0..rows).any(|a| (0..rows).any(|b| a != b && classes[a] == classes[b]));
        match quotient_consistency(&classes, &broken).expect("checks") {
            EvidenceStatus::Counterexample { witness, .. } => {
                assert_eq!(witness.intervention, 2);
                assert_eq!(classes[witness.rows.0], classes[witness.rows.1]);
                assert!(quotient_consistency(&classes, &broken[..2]).expect("checks").lower_bound().is_some_and(|l| l <= 2.0 * band));
            }
            other => assert!(!same_class_rows, "trial {trial}: expected a refutation, got {other:?}"),
        }
    }
    // Classes read off an exact state; a banded one is refused.
    let exact = BandedMatrix { values: ndarray::array![[1.0, 0.0], [1.0, -0.0], [0.5, 0.0]], bands: Array2::zeros((3, 2)) };
    assert_eq!(exact_classes(&exact).expect("exact"), vec![0, 0, 1]);
    let banded = BandedMatrix { bands: Array2::from_elem((3, 2), 1e-16), ..exact };
    assert!(exact_classes(&banded).is_err());
}

/// Theorem 5.3 and its corollary: no pair of logit vectors sampled inside the two boxes has a KL
/// above the certified supremum, however large the radii.
#[test]
fn no_sample_inside_the_logit_boxes_exceeds_the_kl_supremum() {
    let mut rng = StdRng::seed_from_u64(2951);
    for _ in 0..300 {
        let classes = rng.random_range(2..9usize);
        let reference = Array1::from_shape_fn(classes, |_| rng.random_range(-4.0..4.0));
        let perturbed = &reference + &Array1::from_shape_fn(classes, |_| rng.random_range(-0.5..0.5));
        let scale = [1e-3, 0.1, 1.0][rng.random_range(0..3usize)];
        let reference_radius = Array1::from_shape_fn(classes, |_| rng.random_range(0.0..scale));
        let perturbed_radius = Array1::from_shape_fn(classes, |_| rng.random_range(0.0..scale));
        let status =
            kl_supremum_over_logit_boxes(reference.view(), reference_radius.view(), perturbed.view(), perturbed_radius.view())
                .expect("bound");
        let upper = status.upper_bound().expect("a uniform bound");
        for _ in 0..200 {
            // Vertices of the boxes as well as interior points.
            let draw = |center: &Array1<f64>, radius: &Array1<f64>, rng: &mut StdRng| {
                Array1::from_shape_fn(classes, |i| {
                    let t = if rng.random_range(0..2) == 0 { rng.random_range(-1.0..=1.0) } else { [-1.0, 1.0][rng.random_range(0..2usize)] };
                    center[i] + t * radius[i]
                })
            };
            let (p, q) = (draw(&reference, &reference_radius, &mut rng), draw(&perturbed, &perturbed_radius, &mut rng));
            let (kl, error) = categorical_kl_from_logits_with_error(&p.to_vec(), &q.to_vec()).expect("kl");
            assert!(kl - error <= upper, "sampled KL {kl} above the supremum {upper}");
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Theorems 2f and 5.2: the edit-normalized certificate.

/// Edits `D` (one row per component) and linear prediction errors `E` (one row per component);
/// `G = D Dᵀ` and `K = E Eᵀ / columns`.
fn grams(edits: &Array2<f64>, errors: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    (edits.dot(&edits.t()), errors.dot(&errors.t()) / errors.ncols() as f64)
}

fn bounded(edits: &Array2<f64>, errors: &Array2<f64>) -> (f64, f64, Vec<f64>) {
    let (g, k) = grams(edits, errors);
    match physically_null_supremum(g.view(), k.view()).expect("certificate") {
        NullEditOutcome::Bounded(EvidenceStatus::Exact { value, numerical_error, witness, .. }) => {
            (value, numerical_error, witness.map(|w| w.direction).unwrap_or_default())
        }
        other => panic!("expected an exact bound, got {other:?}"),
    }
}

/// Theorem 5.2 and its corollary: no random or ascending search over control changes beats the
/// certified supremum per unit of edit, and the witness attains it. Theorem 2f: renaming, rescaling
/// and duplicating components leave the supremum unchanged within the errors, and a duplicate whose
/// predicted change differs is refused, after which an ascent grows without bound.
#[test]
fn no_search_beats_the_edit_certificate_and_it_ignores_how_components_are_named() {
    let mut rng = StdRng::seed_from_u64(0x5ea);
    for trial in 0..20 {
        let (m, p, n) = (rng.random_range(2..6usize), 9usize, 7usize);
        let edits = Array2::from_shape_simple_fn((m, p), || rng.random_range(-1.0..1.0));
        let errors = Array2::from_shape_simple_fn((m, n), || rng.random_range(-0.5..0.5));
        let (value, error, witness) = bounded(&edits, &errors);
        let (g, k) = grams(&edits, &errors);
        let quadratic = |matrix: &Array2<f64>, u: &Array1<f64>| u.dot(&matrix.dot(u));
        let witness = Array1::from(witness);
        // Evaluating `uᵀKu` here rounds by at most `γ_{m²}` of `|u|ᵀ|K||u|`.
        let rounding = |u: &Array1<f64>| accumulation_growth(m * m + 1) * u.mapv(f64::abs).dot(&k.mapv(f64::abs).dot(&u.mapv(f64::abs)));
        assert!((quadratic(&k, &witness) - value).abs() <= error + rounding(&witness), "trial {trial}: the witness attains");
        // Random directions and a projected ascent on the Rayleigh quotient.
        for _ in 0..500 {
            let mut u = Array1::from_shape_simple_fn(m, || rng.random_range(-1.0..1.0));
            for _ in 0..20 {
                let (gu, ku) = (g.dot(&u), k.dot(&u));
                let ratio = u.dot(&ku) / u.dot(&gu);
                let slack = accumulation_growth(2 * m + 2) * (ratio.abs() + u.mapv(f64::abs).dot(&k.mapv(f64::abs).dot(&u.mapv(f64::abs))) / u.dot(&gu));
                assert!(ratio <= value + error + slack, "trial {trial}: a search found {ratio} above {value} ± {error}");
                u = &u + &((&ku - &(&gu * ratio)) * 0.1);
            }
        }
        // Renaming, rescaling and duplicating one component.
        let mut order: Vec<usize> = (0..m).collect();
        order.rotate_left(1);
        let renamed = bounded(&edits.select(Axis(0), &order), &errors.select(Axis(0), &order));
        let scales = Array1::from_shape_simple_fn(m, || rng.random_range(0.25..4.0));
        let rescaled = bounded(&(&edits * &scales.clone().insert_axis(Axis(1))), &(&errors * &scales.insert_axis(Axis(1))));
        let with_copy: Vec<usize> = (0..m).chain([0]).collect();
        let duplicated = bounded(&edits.select(Axis(0), &with_copy), &errors.select(Axis(0), &with_copy));
        for (other, other_error, _) in [renamed, rescaled, duplicated] {
            // Each certificate is exact up to its own error; the two exact values are one supremum,
            // up to the Grams' own assembly rounding, `γ_p` of each entry's magnitude.
            let assembly = accumulation_growth(p + n) * value.max(other);
            assert!((other - value).abs() <= error + other_error + assembly, "trial {trial}: {other} against {value}");
        }
        // A duplicate that predicts a different change for the same edit.
        let mut divergent = errors.select(Axis(0), &with_copy);
        divergent.row_mut(m).mapv_inplace(|v| v + 0.3);
        let (g, k) = grams(&edits.select(Axis(0), &with_copy), &divergent);
        let NullEditOutcome::Refused(refusal) = physically_null_supremum(g.view(), k.view()).expect("certificate") else {
            panic!("trial {trial}: ker G ⊄ ker K must be refused")
        };
        let direction = Array1::from(refusal.into_witness().expect("a witness").direction);
        let base = Array1::from_shape_fn(m + 1, |i| if i == 0 { 1.0 } else { 0.0 });
        let ratio = |t: f64| {
            let u = &base + &(&direction * t);
            quadratic(&k, &u) / quadratic(&g, &u)
        };
        assert!(ratio(1e3) > 1e3 * ratio(1.0).abs().max(1.0), "trial {trial}: the refused sup is unbounded");
    }
}

// ------------------------------------------------------------------------------------------------
// Small exact worlds: distributions, binomial tails and Clopper–Pearson, computed independently of
// every owner module.

fn softmax(logits: &[f64]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|z| (z - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0
}

fn kl_nats(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a / b).ln()).sum()
}

fn random_logits(rng: &mut StdRng, classes: usize, spread: f64) -> Vec<f64> {
    (0..classes).map(|_| rng.random_range(-spread..spread)).collect()
}

fn shuffle<T>(values: &mut [T], rng: &mut StdRng) {
    for i in (1..values.len()).rev() {
        values.swap(i, rng.random_range(0..=i));
    }
}

/// `ln Pr[Bin(n, p) = k]` from exact log-factorial sums.
fn log_binomial_pmf(n: u64, k: u64, p: f64) -> f64 {
    let log_choose: f64 = (1..=k).map(|i| ((n - k + i) as f64).ln() - (i as f64).ln()).sum();
    let success = if k == 0 { 0.0 } else { k as f64 * p.ln() };
    let failure = if k == n { 0.0 } else { (n - k) as f64 * (-p).ln_1p() };
    log_choose + success + failure
}

fn binomial_cdf(n: u64, k: u64, p: f64) -> f64 {
    (0..=k).map(|i| log_binomial_pmf(n, i, p).exp()).sum::<f64>().min(1.0)
}

/// Clopper–Pearson's upper end, the `p` with `Pr[Bin(n, p) ≤ k] = δ`, by bisection on the
/// decreasing CDF and returned from the covering side.
fn clopper_pearson_upper(k: u64, n: u64, delta: f64) -> f64 {
    if k == n {
        return 1.0;
    }
    let (mut low, mut high) = (k as f64 / n as f64, 1.0);
    for _ in 0..60 {
        let middle = 0.5 * (low + high);
        if binomial_cdf(n, k, middle) > delta { low = middle } else { high = middle }
    }
    high
}

/// The least `q` with `Pr[Bin(trials, rate) > q] ≤ 10⁻⁹`: a statement that fails with probability at
/// most `rate` per independent trial fails more than `q` times with probability at most `10⁻⁹`.
fn failure_ceiling(trials: u64, rate: f64) -> u64 {
    let mut tail = 1.0;
    for q in 0..=trials {
        tail -= log_binomial_pmf(trials, q, rate).exp();
        if tail <= 1e-9 {
            return q;
        }
    }
    trials
}

// ------------------------------------------------------------------------------------------------
// Theorem 1a.

/// Theorem 1a's argmax step: whenever `certified_argmax` accepts, every point of the logit box,
/// its worst vertex included, has the computed argmax.
#[test]
fn a_certified_argmax_is_the_argmax_of_every_point_of_the_box() {
    let mut rng = StdRng::seed_from_u64(0x1a);
    let mut certified = 0;
    for _ in 0..3000 {
        let classes = rng.random_range(2..8usize);
        let values = Array1::from(random_logits(&mut rng, classes, 3.0));
        let scale = [1e-3, 0.1, 1.0][rng.random_range(0..3usize)];
        let radius = Array1::from_shape_fn(classes, |_| rng.random_range(0.0..scale));
        if !certified_argmax(values.view(), radius.view()) {
            continue;
        }
        certified += 1;
        let top = computed_argmax(values.view());
        let worst = Array1::from_shape_fn(classes, |i| if i == top { values[i] - radius[i] } else { values[i] + radius[i] });
        assert_eq!(computed_argmax(worst.view()), top);
        for _ in 0..100 {
            let point = Array1::from_shape_fn(classes, |i| values[i] + rng.random_range(-1.0..=1.0) * radius[i]);
            assert_eq!(computed_argmax(point.view()), top);
        }
    }
    assert!(certified > 300, "{certified} certified rows");
}

// ------------------------------------------------------------------------------------------------
// Theorems 1c, 1d, 5d.2–3 and 6b.

/// Theorems 1c, 1d and 5d.2–3, and 6b's code-free validity: over repeated samples, Clopper–Pearson
/// covers a program fixed before the sample; the Occam bound under Elias γ and under Elias δ, on a unit
/// indicator and on a `[0, 1]`-valued unit bound ([`adversary_advantage_upper`]), covers the program
/// selected on the same sample; and Clopper–Pearson applied to the selected program does not. Each
/// covering bound fails at rate at most `δ`, so it fails more often than [`failure_ceiling`] with
/// probability at most `10⁻⁹`, and the selected Clopper–Pearson exceeds that ceiling.
#[test]
fn occam_covers_the_selected_program_and_clopper_pearson_does_not() {
    let mut rng = StdRng::seed_from_u64(0x0cca);
    let (units, candidates, n, trials, delta) = (64usize, 256usize, 200usize, 300u64, 0.05);
    // Every candidate has one population law (a permutation of one multiset over uniform units), so the
    // candidate with the least sample mean is chosen by chance alone.
    let indicator: Vec<f64> = (0..units).map(|u| if u < 19 { 1.0 } else { 0.0 }).collect();
    let graded: Vec<f64> = (0..units).map(|u| (u as f64 / units as f64).powi(2)).collect();
    let population = |values: &[f64]| values.iter().sum::<f64>() / units as f64;
    let (rate, mean) = (population(&indicator), population(&graded));
    let permuted = |base: &[f64], rng: &mut StdRng| {
        let mut values = base.to_vec();
        shuffle(&mut values, rng);
        values
    };
    let binary: Vec<Vec<f64>> = (0..candidates).map(|_| permuted(&indicator, &mut rng)).collect();
    let real: Vec<Vec<f64>> = (0..candidates).map(|_| permuted(&graded, &mut rng)).collect();
    let mut failures = [0u64; 5];
    for _ in 0..trials {
        let sample: Vec<usize> = (0..n).map(|_| rng.random_range(0..units)).collect();
        let sums = |tables: &[Vec<f64>]| -> Vec<f64> { tables.iter().map(|t| sample.iter().map(|&u| t[u]).sum()).collect() };
        let selected = |s: &[f64]| (0..candidates).min_by(|a, b| s[*a].total_cmp(&s[*b])).expect("candidates");
        let (binary_sums, real_sums) = (sums(&binary), sums(&real));
        let (chosen, chosen_real) = (selected(&binary_sums), selected(&real_sums));
        let k = binary_sums[chosen] as u64;
        failures[0] += u64::from(clopper_pearson_upper(binary_sums[0] as u64, n as u64, delta) < rate);
        failures[1] += u64::from(clopper_pearson_upper(k, n as u64, delta) < rate);
        // Candidate `j` is sent as the integer `j + 1`.
        for (slot, code) in [(2, IntegerCode::EliasGamma), (3, IntegerCode::EliasDelta)] {
            let bits = code.len_bits(chosen as u64 + 1).expect("length") as f64;
            let occam = k as f64 / n as f64 + ((bits * std::f64::consts::LN_2 + (1.0 / delta).ln()) / (2.0 * n as f64)).sqrt();
            failures[slot] += u64::from(occam < rate);
        }
        let values: Vec<f64> = sample.iter().map(|&u| real[chosen_real][u]).collect();
        let bits = IntegerCode::EliasGamma.len_bits(chosen_real as u64 + 1).expect("length");
        failures[4] += u64::from(adversary_advantage_upper(&values, bits, 1.0 - delta, 1).expect("bound") < mean);
    }
    let ceiling = failure_ceiling(trials, delta);
    for slot in [0, 2, 3, 4] {
        assert!(failures[slot] <= ceiling, "slot {slot}: {failures:?} against {ceiling}");
    }
    assert!(failures[1] > ceiling, "selection must break Clopper–Pearson: {failures:?} against {ceiling}");
}

// ------------------------------------------------------------------------------------------------
// Theorem 2a.

/// The executed logits of `second · relu(first · x)` with a rigorous per-entry forward-error radius:
/// `γ_w |first||x|` before the ReLU (which is 1-Lipschitz), then `γ_h |second||h̃| + |second| e₁`, and the
/// radius's own nonnegative evaluation inflated by `1 + γ`.
fn executed_logits(first: &Array2<f64>, second: &Array2<f64>, x: &[f64]) -> (Array1<f64>, Array1<f64>) {
    let ((hidden, width), vocab) = (first.dim(), second.nrows());
    let pre: Vec<f64> = (0..hidden).map(|i| (0..width).fold(0.0, |s, j| s + first[[i, j]] * x[j])).collect();
    let pre_error: Vec<f64> =
        (0..hidden).map(|i| accumulation_growth(width) * (0..width).map(|j| (first[[i, j]] * x[j]).abs()).sum::<f64>()).collect();
    let active: Vec<f64> = pre.iter().map(|v| v.max(0.0)).collect();
    let logits = Array1::from_shape_fn(vocab, |o| (0..hidden).fold(0.0, |s, i| s + second[[o, i]] * active[i]));
    let inflation = 1.0 + accumulation_growth(2 * (hidden + width) + 4);
    let radius = Array1::from_shape_fn(vocab, |o| {
        let rounding = accumulation_growth(hidden) * (0..hidden).map(|i| (second[[o, i]] * active[i]).abs()).sum::<f64>();
        let carried: f64 = (0..hidden).map(|i| second[[o, i]].abs() * pre_error[i]).sum();
        ((rounding + carried) * inflation).next_up()
    });
    (logits, radius)
}

/// Theorem 2a: a gauge move of the network (a permutation of hidden units with dyadic in/out scales,
/// exact on the weights) moves each row's divergence from a fixed program only within the two
/// executions' certified errors, the summed divergence within their sum, and never flips a certified
/// argmax.
#[test]
fn a_gauge_move_of_the_network_moves_the_divergence_only_within_the_bands() {
    let mut rng = StdRng::seed_from_u64(0x2a);
    let (inputs, width, hidden, vocab) = (20usize, 6usize, 8usize, 5usize);
    for trial in 0..50 {
        let first = Array2::from_shape_simple_fn((hidden, width), || rng.random_range(-1.0..1.0));
        let second = Array2::from_shape_simple_fn((vocab, hidden), || rng.random_range(-1.0..1.0));
        let mut order: Vec<usize> = (0..hidden).collect();
        shuffle(&mut order, &mut rng);
        let exponents: Vec<i32> = (0..hidden).map(|_| rng.random_range(-3..=3)).collect();
        let moved_first = Array2::from_shape_fn((hidden, width), |(i, j)| first[[order[i], j]] * 2f64.powi(exponents[i]));
        let moved_second = Array2::from_shape_fn((vocab, hidden), |(o, i)| second[[o, order[i]]] * 2f64.powi(-exponents[i]));
        let (mut total, mut moved_total, mut band) = (0.0, 0.0, 0.0);
        for _ in 0..inputs {
            let x = random_logits(&mut rng, width, 2.0);
            let (logits, radius) = executed_logits(&first, &second, &x);
            let (moved_logits, moved_radius) = executed_logits(&moved_first, &moved_second, &x);
            let program = &logits + &Array1::from(random_logits(&mut rng, vocab, 0.5));
            let exact = Array1::zeros(vocab);
            let divergence = |reference: &Array1<f64>, radius: &Array1<f64>| {
                match kl_over_logit_boxes(reference.view(), radius.view(), program.view(), exact.view()).expect("bound") {
                    EvidenceStatus::Exact { value, numerical_error, .. } => (value, numerical_error),
                    other => panic!("trial {trial}: {other:?}"),
                }
            };
            let ((value, error), (moved_value, moved_error)) = (divergence(&logits, &radius), divergence(&moved_logits, &moved_radius));
            assert!((value - moved_value).abs() <= error + moved_error, "trial {trial}: {value} ± {error} against {moved_value} ± {moved_error}");
            (total, moved_total, band) = (total + value, moved_total + moved_value, band + error + moved_error);
            if certified_argmax(logits.view(), radius.view()) && certified_argmax(moved_logits.view(), moved_radius.view()) {
                assert_eq!(computed_argmax(logits.view()), computed_argmax(moved_logits.view()));
            }
        }
        let summation = accumulation_growth(inputs) * (total + moved_total);
        assert!((total - moved_total).abs() <= band + summation, "trial {trial}: summed divergence");
    }
}

// ------------------------------------------------------------------------------------------------
// Theorem 2d and 6b.

/// Theorem 2d under every code of the family (6b): the precision codeword's saving from moving `k`
/// bits finer, `Δ = max(0, ℓ(q) − ℓ(q + k))`, stays below the `25(2^k − 1)` bits of copy overhead on
/// every `|q| ≤ 2^16`, and for `q < −k` within the proof's `step · ⌈log₂(k + 1)⌉`, where `step` is the
/// code's doubling increment `ℓ(2w) ≤ ℓ(w) + step`, checked below `2^20` and at every scale above.
#[test]
fn duplication_is_penalized_under_every_code_of_the_family() {
    for (code, step) in [(IntegerCode::EliasGamma, 2u64), (IntegerCode::EliasDelta, 3), (IntegerCode::EliasOmega, 5)] {
        let scales = (20..62).flat_map(|s| [1u64 << s, (1u64 << s) + 1, (1u64 << (s + 1)) - 1]);
        for w in (1u64..1 << 20).chain(scales) {
            assert!(code.len_bits(2 * w).expect("ℓ") <= code.len_bits(w).expect("ℓ") + step, "{code:?}: w = {w}");
        }
        let length = |v: i64| code.signed_len_bits(v).expect("length");
        for k in (1..=16i64).chain([32, 63]) {
            let doublings = u64::from(u64::BITS - (k as u64).leading_zeros());
            let overhead = 25u128 * ((1u128 << k) - 1);
            for q in -(1i64 << 16)..=(1 << 16) {
                let saving = length(q).saturating_sub(length(q + k));
                assert!(u128::from(saving) < overhead, "{code:?}: q = {q}, k = {k}");
                if q < -k {
                    assert!(saving <= step * doublings, "{code:?}: q = {q}, k = {k}");
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Theorem 3.

/// Theorem 3 on planted worlds: with the planted program exact (`D = 0`, `L* = 40` bits) against
/// shorter wrong programs and longer ones, a complete family repeated `r` times selects the planted
/// class iff `r > r₀`, and a longer program never wins; on i.i.d. rows the misidentification
/// frequency stays within the union of Hoeffding terms (at the `10⁻⁹` ceiling).
#[test]
fn a_planted_program_is_identified_exactly_past_its_threshold() {
    let mut rng = StdRng::seed_from_u64(0x3d);
    let (inputs, vocab, planted) = (8usize, 4usize, 40.0f64);
    let ln2 = std::f64::consts::LN_2;
    for trial in 0..40 {
        let native: Vec<Vec<f64>> = (0..inputs).map(|_| random_logits(&mut rng, vocab, 3.0)).collect();
        let wrong = |rng: &mut StdRng, bits: f64| {
            let scale = rng.random_range(0.3..2.0);
            let kl: Vec<f64> = native
                .iter()
                .map(|z| {
                    let moved: Vec<f64> = z.iter().map(|v| v + scale * rng.random_range(-1.0..1.0)).collect();
                    kl_nats(&softmax(z), &softmax(&moved))
                })
                .collect();
            (bits, kl)
        };
        let shorter: Vec<(f64, Vec<f64>)> = (0..30).map(|_| { let bits = rng.random_range(5..40) as f64; wrong(&mut rng, bits) }).collect();
        let longer: Vec<(f64, Vec<f64>)> = (0..10).map(|_| { let bits = rng.random_range(41..70) as f64; wrong(&mut rng, bits) }).collect();
        // `D_μ` in bits per row, `μ` uniform on the inputs.
        let divergence = |kl: &[f64]| kl.iter().sum::<f64>() / inputs as f64 / ln2;
        let threshold =
            shorter.iter().map(|(bits, kl)| (planted - bits) / (inputs as f64 * divergence(kl))).fold(0.0, f64::max);
        for r in 1..=(2.0 * threshold).ceil() as usize + 2 {
            let r = r as f64;
            if (r - threshold).abs() <= 1e-9 * threshold {
                continue;
            }
            let total = |(bits, kl): &(f64, Vec<f64>)| bits + r * kl.iter().sum::<f64>() / ln2;
            let beaten = shorter.iter().any(|candidate| total(candidate) <= planted);
            assert_eq!(!beaten, r > threshold, "trial {trial}: r = {r}, r₀ = {threshold}");
            assert!(longer.iter().all(|candidate| total(candidate) > planted), "trial {trial}");
        }
        if trial >= 6 {
            continue;
        }
        // Sampled rows: the Hoeffding union at the least `N` where it is at most 0.2.
        let most = shorter.iter().flat_map(|(_, kl)| kl.iter().copied()).fold(0.0, f64::max);
        let bound = |n: f64| -> f64 {
            shorter
                .iter()
                .map(|(bits, kl)| {
                    let gap = divergence(kl) - (planted - bits) / n;
                    if gap <= 0.0 { 1.0 } else { (-2.0 * n * gap * gap * ln2 * ln2 / (most * most)).exp() }
                })
                .sum()
        };
        let rows = (1..=4000).find(|n| bound(*n as f64) <= 0.2);
        let Some(rows) = rows else { continue };
        let trials = 200u64;
        let mut failures = 0u64;
        for _ in 0..trials {
            let sample: Vec<usize> = (0..rows).map(|_| rng.random_range(0..inputs)).collect();
            failures += u64::from(
                shorter.iter().any(|(bits, kl)| bits + sample.iter().map(|&x| kl[x]).sum::<f64>() / ln2 <= planted),
            );
        }
        assert!(failures <= failure_ceiling(trials, bound(rows as f64)), "trial {trial}: {failures} at N = {rows}");
    }
}

// ------------------------------------------------------------------------------------------------
// Theorem 4.

/// A set-type intervention atom on one target of a program's reals: a value, or the lattice of
/// `bits` fraction bits applied to the base real.
#[derive(Clone, Copy, Debug)]
enum Atom {
    Set { target: usize, value: f64 },
    Lattice { target: usize, bits: i32 },
}

impl Atom {
    fn target(self) -> usize {
        match self {
            Self::Set { target, .. } | Self::Lattice { target, .. } => target,
        }
    }
}

fn round_to_lattice(value: f64, bits: i32) -> f64 {
    (value * 2f64.powi(bits)).round_ties_even() / 2f64.powi(bits)
}

/// Applies a history to the base reals; a lattice atom rounds the base real, or, when
/// `reads_base` is false, the current one.
fn apply_history(base: &[f64], history: &[Atom], reads_base: bool) -> Vec<f64> {
    let mut state = base.to_vec();
    for atom in history {
        match *atom {
            Atom::Set { target, value } => state[target] = value,
            Atom::Lattice { target, bits } => {
                state[target] = round_to_lattice(if reads_base { base[target] } else { state[target] }, bits)
            }
        }
    }
    state
}

/// Theorem 4's intervention algebra: atoms read off the base program commute on distinct targets,
/// a later atom on one target annihilates an earlier one, and every history equals its normal form
/// `Sort(Collapse(·))`, bitwise. A lattice atom applied to already-rounded reals is a double rounding
/// and is not left-annihilative (Remark 22's failure): `0.3` to 2 bits then 1 bit is `0`, to 1 bit is `0.5`.
#[test]
fn base_read_atoms_form_an_intervention_algebra_and_double_rounding_does_not() {
    let mut rng = StdRng::seed_from_u64(0x4a);
    let targets = 5usize;
    let draw = |rng: &mut StdRng| {
        let target = rng.random_range(0..targets);
        if rng.random_range(0..2) == 0 {
            Atom::Set { target, value: rng.random_range(-64..64) as f64 / 16.0 }
        } else {
            Atom::Lattice { target, bits: rng.random_range(0..6) }
        }
    };
    let mut double_rounding_failures = 0;
    for _ in 0..2000 {
        let base: Vec<f64> = (0..targets).map(|_| rng.random_range(-4.0..4.0)).collect();
        let history: Vec<Atom> = (0..rng.random_range(0..12)).map(|_| draw(&mut rng)).collect();
        let mut last = std::collections::BTreeMap::new();
        for atom in &history {
            last.insert(atom.target(), *atom);
        }
        let normal: Vec<Atom> = last.into_values().collect();
        assert_eq!(apply_history(&base, &history, true), apply_history(&base, &normal, true));
        let (a, b) = (draw(&mut rng), draw(&mut rng));
        if a.target() != b.target() {
            assert_eq!(apply_history(&base, &[a, b], true), apply_history(&base, &[b, a], true));
        } else {
            assert_eq!(apply_history(&base, &[a, b], true), apply_history(&base, &[b], true));
            double_rounding_failures +=
                usize::from(apply_history(&base, &[a, b], false) != apply_history(&base, &[b], false));
        }
    }
    let (early, late) = (Atom::Lattice { target: 0, bits: 2 }, Atom::Lattice { target: 0, bits: 1 });
    assert_eq!(apply_history(&[0.3], &[early, late], false), vec![0.0]);
    assert_eq!(apply_history(&[0.3], &[late], false), vec![0.5]);
    assert!(double_rounding_failures > 0);
}

// ------------------------------------------------------------------------------------------------
// Theorems 5c, 5d and 5e.

/// SplitMix64, as a deterministic adaptive strategy's hash of its view.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The game world of Theorem 5d: unit weights `μ`, and per unit and choice (a row and an edit) the
/// native model's and the program's output distributions and logits.
struct GameWorld {
    weights: Vec<f64>,
    native: Vec<Vec<Vec<f64>>>,
    program: Vec<Vec<Vec<f64>>>,
    native_logits: Vec<Vec<Vec<f64>>>,
    program_logits: Vec<Vec<Vec<f64>>>,
}

/// Both worlds' probability of every complete transcript of `rounds` rounds, when the strategy picks
/// its choice as a hash of `seed`, the transcript so far and the fresh unit.
fn transcript_laws(world: &GameWorld, rounds: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let (units, choices, vocab) = (world.weights.len(), world.native[0].len(), world.native[0][0].len());
    let (mut native, mut program, mut views) = (vec![1.0], vec![1.0], vec![0u64]);
    for _ in 0..rounds {
        let (mut next_native, mut next_program, mut next_views) = (Vec::new(), Vec::new(), Vec::new());
        for (index, &view) in views.iter().enumerate() {
            for unit in 0..units {
                let choice = (mix(seed ^ mix(view.wrapping_mul(0x100).wrapping_add(unit as u64))) % choices as u64) as usize;
                for token in 0..vocab {
                    next_native.push(native[index] * world.weights[unit] * world.native[unit][choice][token]);
                    next_program.push(program[index] * world.weights[unit] * world.program[unit][choice][token]);
                    next_views.push(mix(view ^ ((unit * vocab + token) as u64 + 1)));
                }
            }
        }
        (native, program, views) = (next_native, next_program, next_views);
    }
    (native, program)
}

/// Theorem 5d.1 and 5e.1 on small worlds, for sampled and for greedy decoding: no adaptive strategy's
/// exact transcript total variation over `k ≤ 3` rounds exceeds `k E_μ[Z*]`; the one-round strategy that
/// takes each unit's worst choice attains `E_μ[Z*]`; the certified `Ẑ` of each unit bounds `Z*`; and
/// `μ{Z* > t} ≤ E_μ[Ẑ]/t`. The tolerance is the transcript sums' `γ` band.
#[test]
fn no_adaptive_adversary_beats_the_composed_total_variation_bound() {
    let mut rng = StdRng::seed_from_u64(0x5d);
    let (units, rows, edits, vocab) = (3usize, 2usize, 2usize, 3usize);
    let choices = rows * edits;
    for trial in 0..40 {
        let greedy = trial % 2 == 1;
        let raw: Vec<f64> = (0..units).map(|_| rng.random_range(0.1..1.0)).collect();
        let weights: Vec<f64> = raw.iter().map(|w| w / raw.iter().sum::<f64>()).collect();
        let native_logits: Vec<Vec<Vec<f64>>> =
            (0..units).map(|_| (0..choices).map(|_| random_logits(&mut rng, vocab, 3.0)).collect()).collect();
        let program_logits: Vec<Vec<Vec<f64>>> = native_logits
            .iter()
            .map(|unit| {
                let scale = [0.0, 0.05, 0.5, 2.0][rng.random_range(0..4usize)];
                unit.iter().map(|z| z.iter().map(|v| v + scale * rng.random_range(-1.0..1.0)).collect()).collect()
            })
            .collect();
        let decode = |z: &Vec<f64>| {
            if greedy {
                let top = computed_argmax(Array1::from(z.clone()).view());
                (0..vocab).map(|t| f64::from(u8::from(t == top))).collect()
            } else {
                softmax(z)
            }
        };
        let law = |logits: &Vec<Vec<Vec<f64>>>| -> Vec<Vec<Vec<f64>>> {
            logits.iter().map(|unit| unit.iter().map(decode).collect()).collect()
        };
        let world = GameWorld { weights, native: law(&native_logits), program: law(&program_logits), native_logits, program_logits };
        let worst: Vec<f64> = (0..units)
            .map(|u| (0..choices).map(|c| total_variation(&world.native[u][c], &world.program[u][c])).fold(0.0, f64::max))
            .collect();
        let expected: f64 = world.weights.iter().zip(&worst).map(|(w, z)| w * z).sum();
        // The certified `Ẑ`: the per-row box bound for sampled decoding, the disagreement indicator for greedy.
        let zero = Array1::zeros(vocab);
        let certified: Vec<f64> = (0..units)
            .map(|u| {
                (0..choices)
                    .map(|c| {
                        let (native, program) = (Array1::from(world.native_logits[u][c].clone()), Array1::from(world.program_logits[u][c].clone()));
                        if greedy {
                            f64::from(u8::from(computed_argmax(native.view()) != computed_argmax(program.view())))
                        } else {
                            total_variation_over_logit_boxes(native.view(), zero.view(), program.view(), zero.view())
                                .expect("bound")
                                .upper_bound()
                                .expect("a uniform bound")
                        }
                    })
                    .fold(0.0, f64::max)
            })
            .collect();
        let rounding = accumulation_growth(4 * vocab);
        for (z, bound) in worst.iter().zip(&certified) {
            assert!(*z <= bound + rounding, "trial {trial}: Z* = {z} above Ẑ = {bound}");
        }
        let certified_mean: f64 = world.weights.iter().zip(&certified).map(|(w, z)| w * z).sum();
        for t in [0.05, 0.1, 0.25, 0.5, 0.9] {
            let mass: f64 = world.weights.iter().zip(&worst).filter(|(_, z)| **z > t).map(|(w, _)| w).sum();
            assert!(mass <= certified_mean / t + rounding, "trial {trial}: Markov at {t}");
        }
        // The one-round strategy that takes each unit's worst choice: its transcript law's total variation.
        let (mut native, mut program) = (Vec::new(), Vec::new());
        for u in 0..units {
            let choice = (0..choices)
                .max_by(|a, b| {
                    total_variation(&world.native[u][*a], &world.program[u][*a])
                        .total_cmp(&total_variation(&world.native[u][*b], &world.program[u][*b]))
                })
                .expect("choices");
            native.extend(world.native[u][choice].iter().map(|p| world.weights[u] * p));
            program.extend(world.program[u][choice].iter().map(|p| world.weights[u] * p));
        }
        let attained = total_variation(&native, &program);
        assert!((attained - expected).abs() <= accumulation_growth(units * vocab + 2 * vocab) * 2.0, "trial {trial}: attained");
        for rounds in 1..=3usize {
            let transcripts = (units * vocab).pow(rounds as u32);
            let slack = accumulation_growth(2 * rounds + transcripts + 2 * vocab + units) * 4.0 * rounds as f64;
            for seed in 0..30u64 {
                let (native, program) = transcript_laws(&world, rounds, mix(seed + 1000 * trial as u64));
                let advantage = total_variation(&native, &program);
                assert!(
                    advantage <= rounds as f64 * expected + slack,
                    "trial {trial}, {rounds} rounds: advantage {advantage} above {rounds} × {expected}"
                );
            }
        }
    }
}

/// Theorem 5c: the counterexample loop on a finite `X × A` adds a new input or edit at every
/// refinement, ends within `|X| + |A|` refinements, and its final claimed grade (the maximum row KL on
/// the contract) equals the exhaustive maximum over the declared domain. A verifier that checks only
/// a few rows per round also terminates, but can stop with a grade below the exhaustive one: a search
/// that fails proves nothing.
#[test]
fn the_counterexample_loop_ends_at_the_exhaustive_certificate() {
    let mut rng = StdRng::seed_from_u64(0x5c);
    let (inputs, edits, vocab, candidates) = (6usize, 4usize, 3usize, 40usize);
    let domain = inputs * edits;
    let mut under_reported = 0;
    for trial in 0..100 {
        let native: Vec<Vec<f64>> = (0..domain).map(|_| random_logits(&mut rng, vocab, 3.0)).collect();
        // Each candidate is accurate on most rows and wrong on a few, at its own code length.
        let programs: Vec<(f64, Vec<f64>)> = (0..candidates)
            .map(|_| {
                let bits = rng.random_range(10..80) as f64;
                let kl = native
                    .iter()
                    .map(|z| {
                        let scale = if rng.random_range(0..5) == 0 { 2.0 } else { 0.05 };
                        let moved: Vec<f64> = z.iter().map(|v| v + scale * rng.random_range(-1.0..1.0)).collect();
                        kl_nats(&softmax(z), &softmax(&moved))
                    })
                    .collect();
                (bits, kl)
            })
            .collect();
        let run = |thorough: bool, rng: &mut StdRng| -> (usize, f64, usize) {
            let (mut xs, mut alphas) = (std::collections::BTreeSet::from([0usize]), std::collections::BTreeSet::from([0usize]));
            let mut refinements = 0;
            loop {
                let rows: Vec<usize> = xs.iter().flat_map(|x| alphas.iter().map(move |a| x * edits + a)).collect();
                let score = |j: usize| programs[j].0 + rows.iter().map(|&r| programs[j].1[r]).sum::<f64>() / std::f64::consts::LN_2;
                let chosen = (0..candidates).min_by(|a, b| score(*a).total_cmp(&score(*b))).expect("candidates");
                let kl = &programs[chosen].1;
                let claimed = rows.iter().map(|&r| kl[r]).fold(0.0, f64::max);
                let checked: Vec<usize> = if thorough { (0..domain).collect() } else { (0..3).map(|_| rng.random_range(0..domain)).collect() };
                let Some(&worst) = checked.iter().filter(|&&r| kl[r] > claimed).max_by(|a, b| kl[**a].total_cmp(&kl[**b])) else {
                    return (chosen, claimed, refinements);
                };
                let (x, a) = (worst / edits, worst % edits);
                assert!(!(xs.contains(&x) && alphas.contains(&a)), "trial {trial}: a refutation lies outside the contract");
                xs.insert(x);
                alphas.insert(a);
                refinements += 1;
                assert!(refinements <= inputs + edits - 2, "trial {trial}: too many refinements");
            }
        };
        let (chosen, claimed, _) = run(true, &mut rng);
        let exhaustive = programs[chosen].1.iter().copied().fold(0.0, f64::max);
        assert_eq!(claimed, exhaustive, "trial {trial}: the final grade is the exhaustive certificate");
        let (weak_chosen, weak_claimed, _) = run(false, &mut rng);
        let weak_exhaustive = programs[weak_chosen].1.iter().copied().fold(0.0, f64::max);
        assert!(weak_claimed <= weak_exhaustive);
        under_reported += usize::from(weak_claimed < weak_exhaustive);
    }
    assert!(under_reported > 0, "a weak search must sometimes stop short of the supremum");
}

// ------------------------------------------------------------------------------------------------
// Theorem 6.1, 6.4 and 6b.

/// Theorem 6.1: two codes that differ only on a field whose values are one multiset in every
/// candidate give one verdict.
#[test]
fn codes_that_differ_only_on_shared_fields_give_one_verdict() {
    let mut rng = StdRng::seed_from_u64(0x61);
    for _ in 0..2000 {
        let candidates = rng.random_range(2..7usize);
        let shared: Vec<u64> = (0..rng.random_range(1..6)).map(|_| rng.random_range(1..1u64 << 20)).collect();
        let own: Vec<Vec<u64>> =
            (0..candidates).map(|_| (0..rng.random_range(0..5)).map(|_| rng.random_range(1..1u64 << 12)).collect()).collect();
        let data: Vec<(f64, f64)> = (0..candidates)
            .map(|_| {
                let lower = rng.random_range(0.0..60.0);
                (lower, lower + rng.random_range(0.0..0.5))
            })
            .collect();
        let claim: Vec<bool> = (0..candidates).map(|_| rng.random_range(0..2) == 1).collect();
        let family = |shared_code: IntegerCode| -> Vec<CodedCandidate> {
            (0..candidates)
                .map(|c| {
                    let mut fields = shared.clone();
                    shuffle(&mut fields, &mut StdRng::seed_from_u64(c as u64));
                    let bits = fields.iter().map(|v| shared_code.len_bits(*v).expect("ℓ")).sum::<u64>()
                        + own[c].iter().map(|v| IntegerCode::EliasDelta.len_bits(*v).expect("ℓ")).sum::<u64>();
                    CodedCandidate { program_bits: vec![bits], data_bits_lower: data[c].0, data_bits_upper: data[c].1 }
                })
                .collect()
        };
        let verdicts: Vec<ClaimVerdict> =
            IntegerCode::FAMILY.iter().map(|code| claim_robustness(&family(*code), &claim).expect("verdict")).collect();
        for verdict in &verdicts[1..] {
            assert_eq!(std::mem::discriminant(verdict), std::mem::discriminant(&verdicts[0]), "{verdicts:?}");
        }
    }
}

/// 6b's magnitude trade-off and Theorem 6.4: a program that writes one large index exactly against one
/// that writes a few small indices and pays `residual` data bits per repetition is code dependent at one
/// repetition exactly when the codes disagree on the sign of `L_c(fine) − L_c(coarse) − residual`, and is
/// robust past `max_c r₀(c) + 1`; the γ–δ gap on the large index is `f − 2⌊log₂(f + 1)⌋`.
#[test]
fn a_magnitude_trade_off_is_code_dependent_until_the_data_decides() {
    let mut rng = StdRng::seed_from_u64(0x64);
    let mut dependent = 0;
    for trial in 0..500 {
        let f = rng.random_range(8..60u32);
        let big = (1u64 << f) + rng.random_range(0..1u64 << (f - 1));
        let small: Vec<u64> = (0..rng.random_range(1..30)).map(|_| rng.random_range(1..8u64)).collect();
        let residual = rng.random_range(0.01..4.0);
        let gap = IntegerCode::EliasGamma.len_bits(big).expect("ℓ") - IntegerCode::EliasDelta.len_bits(big).expect("ℓ");
        assert_eq!(gap, u64::from(f) - 2 * u64::from(u32::BITS - 1 - (f + 1).leading_zeros()));
        let fine: Vec<u64> = IntegerCode::FAMILY.iter().map(|c| c.len_bits(big).expect("ℓ")).collect();
        let coarse: Vec<u64> =
            IntegerCode::FAMILY.iter().map(|c| small.iter().map(|v| c.len_bits(*v).expect("ℓ")).sum()).collect();
        let family = |r: f64| {
            vec![
                CodedCandidate { program_bits: fine.clone(), data_bits_lower: 0.0, data_bits_upper: 0.0 },
                CodedCandidate { program_bits: coarse.clone(), data_bits_lower: r * residual, data_bits_upper: (r * residual).next_up() },
            ]
        };
        let claim = [true, false];
        let threshold = (0..fine.len()).map(|c| (fine[c] as f64 - coarse[c] as f64) / residual).fold(0.0, f64::max);
        let verdict = claim_robustness(&family(threshold.floor() + 2.0), &claim).expect("verdict");
        assert!(matches!(verdict, ClaimVerdict::Robust { .. }), "trial {trial}: {verdict:?}");
        let excess: Vec<f64> = (0..fine.len()).map(|c| fine[c] as f64 - coarse[c] as f64 - residual).collect();
        let disagree = excess.iter().any(|e| *e < 0.0) && excess.iter().any(|e| *e > 0.0);
        let verdict = claim_robustness(&family(1.0), &claim).expect("verdict");
        assert_eq!(matches!(verdict, ClaimVerdict::CodeDependent { .. }), disagree, "trial {trial}: {verdict:?} for {excess:?}");
        dependent += usize::from(disagree);
    }
    assert!(dependent > 20, "{dependent} code-dependent trade-offs");
}
