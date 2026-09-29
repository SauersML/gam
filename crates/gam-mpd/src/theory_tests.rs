#![cfg(test)]
//! Machine checks of the theorems in `theory`: each test names the theorem it exercises, and
//! every tolerance is derived (a band, an exact rational sum, a stated confidence).

use super::bounds::kl_supremum_over_logit_boxes;
use super::codec::{prefix_integer_len_bits, signed_prefix_integer_len_bits, subset_code_len_bits};
use super::secant::BandedMatrix;
use super::supports::EvidenceStatus;
use super::theory::{ClaimVerdict, CodedCandidate, IntegerCode, claim_robustness, exact_classes, quotient_consistency};
use gam_math::categorical::categorical_kl_from_logits_with_error;
use ndarray::{Array1, Array2};
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
