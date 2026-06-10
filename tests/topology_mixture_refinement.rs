//! #996 — local refinement around the mixture-ladder winner.
//!
//! The coarse [`MIXTURE_K_LADDER`] = [1, 2, 3, 5, 7, 9] cannot *name* a
//! planted k = 4, 6, or 8 truth; before refinement the rung returned the
//! nearest rung instead of the true order, and the order is part of the
//! scientific claim ("7 clusters" vs "circle"). These planted tests assert:
//!
//! * an off-ladder truth (k = 4 and k = 6 blobs at the same matched SNR as
//!   the existing planted races) is recovered as EXACTLY its planted order;
//! * an in-ladder truth (k = 7) still wins, with the refinement proving its
//!   bracket (both immediate neighbours fitted and worse) rather than
//!   creeping the order;
//! * on circle truth (no cluster structure) the refinement terminates with a
//!   bracketed small-order winner instead of walking the ladder upward.

use gam::solver::evidence::GaussianMixtureConfig;
use gam::solver::topology_selector::{MIXTURE_K_LADDER, MixtureRungResult, fit_mixture_rung};
use ndarray::Array2;

// ---------------------------------------------------------------------------
// Deterministic RNG (fixed integer seed, no clock).
// ---------------------------------------------------------------------------

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-12);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Planted generators at the same matched SNR as tests/topology_mixture_rung.rs.
// ---------------------------------------------------------------------------

const SNR: f64 = 12.0;
const N_OBS: usize = 350;

/// `k_true` isotropic blobs on a unit ring, within-cluster spread set by the
/// nearest-neighbour spacing over SNR (matched-SNR convention of the existing
/// planted races).
fn sample_k_clusters(k_true: usize, seed: u64) -> Array2<f64> {
    let r_struct = 1.0_f64;
    let spacing = 2.0 * r_struct * (std::f64::consts::PI / k_true as f64).sin();
    let spread = spacing / SNR;
    let mut rng = SplitMix64::new(seed ^ 0xC1057E12_u64 ^ (k_true as u64) << 32);
    let mut centers = Vec::with_capacity(k_true);
    for c in 0..k_true {
        let phi = std::f64::consts::TAU * c as f64 / k_true as f64;
        centers.push((r_struct * phi.cos(), r_struct * phi.sin()));
    }
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    for i in 0..N_OBS {
        let c = (rng.next_u64() as usize) % k_true;
        let (cx, cy) = centers[c];
        out[[i, 0]] = cx + spread * rng.next_gaussian();
        out[[i, 1]] = cy + spread * rng.next_gaussian();
    }
    out
}

/// Unit ring with radial jitter (no cluster structure).
fn sample_circle(seed: u64) -> Array2<f64> {
    let noise = 1.0 / SNR;
    let mut rng = SplitMix64::new(seed ^ 0xC18C1E_u64);
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    for i in 0..N_OBS {
        let theta = std::f64::consts::TAU * rng.next_unit();
        out[[i, 0]] = theta.cos() + noise * rng.next_gaussian();
        out[[i, 1]] = theta.sin() + noise * rng.next_gaussian();
    }
    out
}

fn fit(data: &Array2<f64>) -> MixtureRungResult {
    fit_mixture_rung(
        data.view(),
        MIXTURE_K_LADDER,
        GaussianMixtureConfig::default(),
    )
    .expect("mixture rung must fit")
}

/// Assert the winner is BRACKETED: both immediate neighbours were fitted
/// (k−1 only when ≥ 1) and both score strictly worse. This is the structural
/// guarantee refinement adds — and the proof it terminated by bracketing,
/// not by creeping or by the probe cap.
fn assert_bracketed(rung: &MixtureRungResult, label: &str) {
    let winner = rung.winner();
    let nle_of = |k: usize| -> Option<f64> {
        rung.fits
            .iter()
            .find(|f| f.k == k)
            .map(|f| f.negative_log_evidence)
    };
    if winner.k > 1 {
        let lower = nle_of(winner.k - 1).unwrap_or_else(|| {
            panic!(
                "{label}: refinement must have fitted the lower neighbour k={}",
                winner.k - 1
            )
        });
        assert!(
            lower > winner.negative_log_evidence,
            "{label}: lower neighbour k={} must score worse ({lower:.2} vs {:.2})",
            winner.k - 1,
            winner.negative_log_evidence
        );
    }
    let upper = nle_of(winner.k + 1).unwrap_or_else(|| {
        panic!(
            "{label}: refinement must have fitted the upper neighbour k={}",
            winner.k + 1
        )
    });
    assert!(
        upper > winner.negative_log_evidence,
        "{label}: upper neighbour k={} must score worse ({upper:.2} vs {:.2})",
        winner.k + 1,
        winner.negative_log_evidence
    );
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn off_ladder_truths_k4_and_k6_are_recovered_exactly() {
    for &k_true in &[4_usize, 6] {
        for &seed in &[7_u64, 41, 97] {
            let data = sample_k_clusters(k_true, seed);
            let rung = fit(&data);
            let winner = rung.winner();
            assert_eq!(
                winner.k,
                k_true,
                "k_true={k_true} seed={seed}: the refined rung must NAME the \
                 planted off-ladder order exactly (got k={}, fitted orders: {:?})",
                winner.k,
                rung.fits.iter().map(|f| f.k).collect::<Vec<_>>(),
            );
            assert_bracketed(&rung, &format!("k_true={k_true} seed={seed}"));
        }
    }
}

#[test]
fn in_ladder_truth_k7_is_unaffected_and_bracketed() {
    for &seed in &[13_u64, 53] {
        let data = sample_k_clusters(7, seed);
        let rung = fit(&data);
        assert_eq!(
            rung.winner().k,
            7,
            "seed={seed}: an in-ladder truth must keep winning after \
             refinement (got k={})",
            rung.winner().k
        );
        assert_bracketed(&rung, &format!("k_true=7 seed={seed}"));
    }
}

#[test]
fn circle_truth_refinement_brackets_instead_of_creeping() {
    for &seed in &[11_u64, 47] {
        let data = sample_circle(seed);
        let rung = fit(&data);
        let winner_k = rung.winner().k;
        // No planted cluster structure: the in-class winner must stay within
        // the coarse ladder's neighbourhood (the parameter pricing must stop
        // the upward walk), and it must be a genuine bracketed optimum.
        let ladder_top = *MIXTURE_K_LADDER.last().unwrap();
        assert!(
            winner_k <= ladder_top + 1,
            "seed={seed}: circle truth must not let refinement creep the \
             order past the ladder top (winner k={winner_k}, top={ladder_top})"
        );
        assert_bracketed(&rung, &format!("circle seed={seed}"));
    }
}
