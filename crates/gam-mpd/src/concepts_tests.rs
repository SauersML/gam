#![cfg(test)]
//! Concepts: the KT code, planted fire-together groups recovered as named concepts (a member that
//! co-fires but is not worth its program stays out), the exact total refusing what the prices
//! wrongly propose, and coding fresh words from names alone.

use super::concepts::{Model, Oracle, Sets, fit, kt_bits};
use std::time::{Duration, Instant};

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

/// The model as an additive oracle: a word's KL is the price of every set member its program
/// leaves off (observations ln 2, so a price in nats is its bits), and the prices are exact.
struct Additive<'a> {
    sets: &'a Sets,
    price: &'a [f64],
}

impl Oracle for Additive<'_> {
    fn kl(&mut self, programs: &[Vec<u32>]) -> Result<Vec<f64>, String> {
        Ok(programs
            .iter()
            .enumerate()
            .map(|(t, p)| {
                let k = self.sets.indptr[t]..self.sets.indptr[t + 1];
                self.sets.row(t).iter().zip(&self.price[k]).filter(|(j, _)| p.binary_search(*j).is_err()).map(|(_, d)| d).sum()
            })
            .collect())
    }

    fn prices(&mut self, programs: &[Vec<u32>], sets: &Sets) -> Result<Vec<f64>, String> {
        assert_eq!(sets.indices, self.sets.indices, "priced on its own sets");
        assert_eq!(programs.len(), sets.rows(), "a program per word");
        Ok(self.price.to_vec())
    }
}

/// Words over 40 subcomponents: three planted groups, each needed at rate 0.2 (every member on,
/// 50 nats each if left off); the passenger on with the first group at 0.01; 30..40 on at random
/// at rate 0.1 and 0.1 each.
fn planted(words: usize, salt: u64) -> (Sets, Vec<f64>, Vec<[bool; 3]>) {
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut price = Vec::new();
    let mut truth = Vec::new();
    for t in 0..words as u64 {
        let draw = |j: u64, k: u64| uniform(salt ^ (t * 1_000_003 + j * 101 + k));
        let needed = [0, 1, 2].map(|g| draw(1000 + g, 7) < 0.2);
        for j in 0..40u32 {
            let p = match PLANTED.iter().position(|r| r.contains(&j)) {
                Some(g) if needed[g] => Some(50.0),
                Some(_) => None,
                None if j == PASSENGER && needed[0] => Some(0.01),
                None if j >= 30 && draw(u64::from(j), 3) < 0.1 => Some(0.1),
                None => None,
            };
            if let Some(p) = p {
                indices.push(j);
                price.push(p);
            }
        }
        indptr.push(indices.len());
        truth.push(needed);
    }
    (Sets::new(40, indptr, indices).expect("sets"), price, truth)
}

fn later() -> Instant {
    Instant::now() + Duration::from_secs(600)
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
    let (sets, price, _) = planted(3000, 0xC0);
    let program = vec![PROGRAM; 40];
    let fitted = fit(&sets, &price, &program, 40.0, std::f64::consts::LN_2, later(), &mut Additive { sets: &sets, price: &price }).expect("fit");
    let mut vocabulary: Vec<Vec<u32>> = fitted.concepts.iter().filter(|c| c.invocations() > 0).map(|c| c.members.clone()).collect();
    vocabulary.sort();
    let want: Vec<Vec<u32>> = PLANTED.iter().map(|r| r.clone().collect()).collect();
    assert_eq!(vocabulary, want, "the passenger and the cheap noise must stay out of the vocabulary");
    assert!(fitted.total_bits < fitted.start_bits, "{} bits vs {} for the words' own sets", fitted.total_bits, fitted.start_bits);

    // Fresh words: the names alone decode to the needed groups.
    let model = Model::new(&fitted);
    let (fresh, fresh_price, truth) = planted(1000, 0x5EED);
    let coded = model.encode(&fresh, &fresh_price, std::f64::consts::LN_2, later(), &mut Additive { sets: &fresh, price: &fresh_price }).expect("code");
    let mut agree = 0;
    for (t, needed) in truth.iter().enumerate() {
        let want: Vec<u32> = PLANTED.iter().zip(needed).filter(|(_, n)| **n).flat_map(|(r, _)| r.clone()).collect();
        agree += usize::from(model.decode(&coded.invoked[t]) == want);
        assert!(coded.bits[t].kl < 5.0, "word {t} leaves {} nats", coded.bits[t].kl);
    }
    assert!(agree >= 995, "{agree} of 1000 fresh words decode to exactly their needed groups");
}

/// Two subcomponents that back each other up: either alone keeps the word exact, both off cost
/// 100 nats. At both on each one's price is nothing, so the prices propose dropping both. A price
/// is a flip with the rest of the program kept.
struct Redundant;

impl Oracle for Redundant {
    fn kl(&mut self, programs: &[Vec<u32>]) -> Result<Vec<f64>, String> {
        Ok(programs.iter().map(|p| if p.is_empty() { 100.0 } else { 0.0 }).collect())
    }

    fn prices(&mut self, programs: &[Vec<u32>], sets: &Sets) -> Result<Vec<f64>, String> {
        Ok((0..sets.rows())
            .flat_map(|t| sets.row(t).iter().map(move |j| (t, *j)))
            .map(|(t, j)| if programs[t].iter().any(|o| *o != j) { 0.0 } else { 100.0 })
            .collect())
    }
}

#[test]
fn the_exact_total_refuses_dropping_both_backups() {
    let words = 200;
    let sets = Sets::new(2, (0..=words).map(|t| 2 * t).collect(), (0..words).flat_map(|_| [0, 1]).collect()).expect("sets");
    // At the sets themselves each backup's price is nothing: the other covers it.
    let fitted = fit(&sets, &vec![0.0; 2 * words], &[PROGRAM, PROGRAM], 40.0, std::f64::consts::LN_2, later(), &mut Redundant).expect("fit");
    let on: Vec<usize> = fitted.concepts.iter().map(|c| c.invocations()).collect();
    assert_eq!(on.iter().sum::<usize>(), words, "exactly one backup runs on every word: {on:?}");
    assert!(fitted.kl.iter().all(|k| *k == 0.0), "no word is left without a backup");
    assert!(fitted.total_bits < fitted.start_bits);
}
