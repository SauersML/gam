#![cfg(test)]
//! Concepts: the KT code, planted co-firing groups recovered with their invocations, and no
//! concept where subcomponents run independently.

use super::concepts::{Fit, Model, Sets, fit, kt_bits};

fn uniform(seed: u64) -> f64 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

const PLANTED: [std::ops::Range<u32>; 3] = [0..6, 10..17, 20..25];

/// Words over 60 subcomponents: three planted groups each invoked at rate 0.2 (members run at
/// 0.95 when invoked, 0.01 otherwise), and 30..60 running independently at 0.1.
fn planted(words: usize, salt: u64) -> (Sets, Vec<[bool; 3]>) {
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut truth = Vec::new();
    for t in 0..words as u64 {
        let seed = |j: u64, k: u64| uniform(salt ^ (t * 1_000_003 + j * 101 + k));
        let on = [0, 1, 2].map(|g| seed(1000 + g, 7) < 0.2);
        for j in 0..60u32 {
            let p = match PLANTED.iter().position(|r| r.contains(&j)) {
                Some(g) if on[g] => 0.95,
                Some(_) => 0.01,
                None if j >= 30 => 0.1,
                None => 0.0,
            };
            if seed(u64::from(j), 3) < p {
                indices.push(j);
            }
        }
        indptr.push(indices.len());
        truth.push(on);
    }
    (Sets::new(60, indptr, indices).expect("sets"), truth)
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
fn planted_concepts_are_recovered_and_shorten_the_code() {
    let (sets, _) = planted(3000, 0xC0);
    let fitted = fit(&sets, 40.0);
    let mut concepts: Vec<Vec<u32>> = fitted.groups.iter().filter(|g| g.members.len() > 1).map(|g| g.members.clone()).collect();
    concepts.sort();
    let want: Vec<Vec<u32>> = PLANTED.iter().map(|r| r.clone().collect()).collect();
    assert_eq!(concepts, want);
    assert!(fitted.total_bits() < Fit::independent_bits(&sets));

    // Fresh words: the encoder names the invoked groups, the decoder turns their members on.
    let model = Model::new(&fitted);
    let (fresh, truth) = planted(1000, 0x5EED);
    let mut agree = 0;
    let mut saved = 0.0;
    for (t, on) in truth.iter().enumerate() {
        let (invoked, bits) = model.encode(fresh.row(t));
        saved += bits.independent - (bits.choices + bits.members + bits.alone);
        let named: Vec<bool> = (0..3).map(|g| invoked.iter().any(|c| model.concepts[*c as usize].members[0] == PLANTED[g].start)).collect();
        agree += named.iter().zip(on).filter(|(a, b)| a == b).count();
        for c in &invoked {
            assert_eq!(model.decode(&[*c]), model.concepts[*c as usize].members);
        }
    }
    assert!(agree as f64 >= 0.99 * 3000.0, "{agree} of 3000 invocations recovered");
    assert!(saved > 0.0, "held-out words cost {saved} bits more than the independent code");
}

#[test]
fn independent_subcomponents_form_no_concept() {
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    for t in 0..3000u64 {
        for j in 0..30u32 {
            if uniform(t * 31 + u64::from(j)) < 0.2 {
                indices.push(j);
            }
        }
        indptr.push(indices.len());
    }
    let sets = Sets::new(30, indptr, indices).expect("sets");
    let fitted = fit(&sets, 40.0);
    assert!(fitted.groups.iter().all(|g| g.members.len() == 1));
    assert!((fitted.total_bits() - Fit::independent_bits(&sets)).abs() < 1e-6);
}
