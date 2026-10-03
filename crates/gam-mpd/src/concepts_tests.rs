#![cfg(test)]
//! Concepts: the KT code, planted fire-together groups recovered as named concepts (a member that
//! co-fires but is not worth its program stays out), and decoding fresh words from names alone.

use super::concepts::{Model, Priced, Sets, fit, kt_bits};

fn uniform(seed: u64) -> f64 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

const PLANTED: [std::ops::Range<u32>; 3] = [0..6, 10..16, 20..25];
/// Runs with the first group but its absence costs almost nothing.
const PASSENGER: u32 = 7;
const PROGRAM: f64 = 5.0;

/// Words over 40 subcomponents: three planted groups, each needed at rate 0.2 (every member on,
/// 50 bits each if left off); the passenger on with the first group at 0.01 bits; 30..40 on at
/// random at rate 0.1 and 0.1 bits each.
fn planted(words: usize, salt: u64) -> (Priced, Vec<[bool; 3]>) {
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut missing = Vec::new();
    let mut truth = Vec::new();
    for t in 0..words as u64 {
        let draw = |j: u64, k: u64| uniform(salt ^ (t * 1_000_003 + j * 101 + k));
        let needed = [0, 1, 2].map(|g| draw(1000 + g, 7) < 0.2);
        for j in 0..40u32 {
            let price = match PLANTED.iter().position(|r| r.contains(&j)) {
                Some(g) if needed[g] => Some(50.0),
                Some(_) => None,
                None if j == PASSENGER && needed[0] => Some(0.01),
                None if j >= 30 && draw(u64::from(j), 3) < 0.1 => Some(0.1),
                None => None,
            };
            if let Some(p) = price {
                indices.push(j);
                missing.push(p);
            }
        }
        indptr.push(indices.len());
        truth.push(needed);
    }
    (Priced::new(Sets::new(40, indptr, indices).expect("sets"), missing).expect("priced"), truth)
}

#[test]
fn kt_code_lengths() {
    assert_eq!(kt_bits(0, 0), 0.0);
    assert!((kt_bits(1, 1) - 1.0).abs() < 1e-12);
    // 1 then 0: ½ then (0 + ½)/(1 + 1).
    assert!((kt_bits(1, 2) - 3.0).abs() < 1e-12);
    assert!((kt_bits(3, 10) - kt_bits(7, 10)).abs() < 1e-9);
}

#[test]
fn planted_groups_become_the_vocabulary() {
    let (priced, _) = planted(3000, 0xC0);
    let program = vec![PROGRAM; 40];
    let fitted = fit(&priced, &program, 40.0);
    let mut vocabulary: Vec<Vec<u32>> = fitted.concepts.iter().filter(|c| c.invocations() > 0).map(|c| c.members.clone()).collect();
    vocabulary.sort();
    let want: Vec<Vec<u32>> = PLANTED.iter().map(|r| r.clone().collect()).collect();
    assert_eq!(vocabulary, want, "the passenger and the cheap noise must stay out of the vocabulary");
    // Naming the groups costs less than running every word's own set as its program.
    let own: f64 = priced.program_bits(&program).iter().sum();
    assert!(fitted.total_bits() < own, "{} bits vs {own} for the words' own sets", fitted.total_bits());

    // Fresh words: the names alone decode to the needed groups.
    let model = Model::new(&fitted);
    let (fresh, truth) = planted(1000, 0x5EED);
    let mut agree = 0;
    for (t, needed) in truth.iter().enumerate() {
        let k = fresh.sets.indptr[t]..fresh.sets.indptr[t + 1];
        let (invoked, bits) = model.encode(fresh.sets.row(t), &fresh.missing[k]);
        let decoded = model.decode(&invoked);
        let want: Vec<u32> = PLANTED.iter().zip(needed).filter(|(_, n)| **n).flat_map(|(r, _)| r.clone()).collect();
        agree += usize::from(decoded == want);
        assert!(bits.error < 5.0, "word {t} leaves {} bits of error", bits.error);
    }
    assert!(agree >= 995, "{agree} of 1000 fresh words decode to exactly their needed groups");
}
