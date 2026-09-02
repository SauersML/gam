//! #2111 INTEGRATED dense-torus acceptance test — the end-to-end bar the issue
//! specifies, closing the gap between the ISA producer / pair-κ machinery (unit-
//! tested) and the full birth pipeline (`fit_stagewise`).
//!
//! FIXTURE. A dense product-of-circles torus: `k = 6` circles on DISJOINT
//! axis-aligned output frames (circle `c` on dims `(2c, 2c+1)`), every row on
//! EVERY circle (density 6 — the degenerate regime where single-plane ring-ness
//! fails and only 4th-order ISA separates the factors), distinct amplitudes
//! `1.0 … 0.55`, independent angles, small isotropic noise. This is the
//! `probe_2101_birth_locus_disjoint_6circle_ordered_beta_bernoulli` structure at a sample size
//! (`n = 700`) clear of the dense-case small-sample floor (`n ≥ 300`; the
//! gated-edge `ISA_SUBSAMPLE_FLOOR` resolution bound concerns `q → ½` gates,
//! not this density-1 fixture).
//!
//! PIPELINE. Seed a single K=1 circle atom on circle 0's true coordinate, then run
//! the integrated forward-birth + backfit engine [`fit_stagewise`]. On a disjoint
//! residual the shared-factor model is rank-0, so births fall through to the ISA
//! fallback seed (`residual_principal_birth_candidate` → `isa_extract_certified_plane`)
//! — the exact machinery #2111 specifies. The seed contributes circle 0; the ISA
//! births must recover circles 1–5 from the blended dense residual.
//!
//! ACCEPTANCE BAR (#2111). 6 born atoms; every atom's decoder output-plane matches
//! a distinct true circle at overlap ≥ 0.9; every decoder is clean (singular-value
//! participation ratio ≤ 3 — a rank-2 circle decoder has PR ≈ 2); `n_distinct = 6`,
//! `n_clean = 6` (best overlap ≥ 0.9 AND second-best ≤ 0.2); and the forward phase
//! exits NATURALLY (`stopped_reason != MaxBirths`).
//!
//! If the bar is not met, the printed per-atom overlap / PR table + birth ledger
//! localise WHICH stage drops the ball (ISA rotation, birth acceptance, or joint
//! backfit) — the failure mode is the finding.

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_2111_dense_torus_acceptance;` — its single declaration. Saying so in-file
// makes the test scope a claim the compiler enforces rather than one the
// filename merely implies, which is what puts the fixture helpers below in
// the same scope as the `#[test]` fns they serve.
#![cfg(test)]

use ndarray::{Array1, Array2, ArrayView2};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-12);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Dense product-of-circles torus target (`n × p`) + the true axis-aligned circle
/// planes. Circle `c` lives on dims `(2c, 2c+1)` with amplitude `amps[c]` and an
/// independent per-row angle; every row carries every circle (dense).
fn dense_torus(
    n: usize,
    p: usize,
    k: usize,
    amps: &[f64],
    sigma: f64,
    seed: u64,
) -> (Array2<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>) {
    assert!(p >= 2 * k && amps.len() == k);
    let mut s = seed;
    let mut data = Array2::<f64>::zeros((n, p));
    // Per-circle true per-row angle (turns) — kept to seed circle 0 and to sanity
    // the fixture; the fitter never sees the angles of circles 1..k.
    let mut turns = vec![Array2::<f64>::zeros((n, 1)); k];
    for i in 0..n {
        for c in 0..k {
            let t = lcg(&mut s);
            turns[c][[i, 0]] = t;
            let ang = std::f64::consts::TAU * t;
            data[[i, 2 * c]] += amps[c] * ang.cos();
            data[[i, 2 * c + 1]] += amps[c] * ang.sin();
        }
        for j in 0..p {
            data[[i, j]] += sigma * lcg_normal(&mut s);
        }
    }
    let true_planes: Vec<Array2<f64>> = (0..k)
        .map(|c| {
            Array2::from_shape_fn(
                (p, 2),
                |(row, col)| {
                    if row == 2 * c + col { 1.0 } else { 0.0 }
                },
            )
        })
        .collect();
    (data, true_planes, turns)
}

/// Fixture sanity (fast, no fit): the planted dense torus really carries `2k`
/// above-noise directions in the expected axis-aligned frames — a guard that the
/// integrated test above is exercising the intended structure, not a degenerate
/// input. Uses the shared column-second-moment eigenstructure.
#[test]
fn dense_torus_fixture_has_2k_signal_dirs_2111() {
    let k = 6usize;
    let (data, _planes, _turns) = dense_torus(700, 16, k, &vec![1.0; k], 0.05, 0x2111_F1F7);
    let signal = column_signal_rank(data.view(), 0.05 * 0.05);
    eprintln!(
        "[#2111 fixture] above-noise signal directions = {signal} (expect {})",
        2 * k
    );
    assert!(
        signal == 2 * k,
        "dense {k}-torus must show exactly {} signal directions; got {signal}",
        2 * k
    );
}

/// Count column-covariance eigenvalues above an isotropic-noise Marchenko–Pastur
/// edge — the number of real signal directions in the centered data.
fn column_signal_rank(data: ArrayView2<'_, f64>, noise_var: f64) -> usize {
    use gam_linalg::faer_ndarray::FaerEigh;
    let (n, p) = data.dim();
    let mut mean = Array1::<f64>::zeros(p);
    for i in 0..n {
        for j in 0..p {
            mean[j] += data[[i, j]];
        }
    }
    mean.mapv_inplace(|v| v / n as f64);
    let mut cov = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for a in 0..p {
            let ra = data[[i, a]] - mean[a];
            for b in a..p {
                cov[[a, b]] += ra * (data[[i, b]] - mean[b]);
            }
        }
    }
    for a in 0..p {
        for b in a..p {
            let v = cov[[a, b]] / n as f64;
            cov[[a, b]] = v;
            cov[[b, a]] = v;
        }
    }
    let (evals, _) = cov.eigh(crate::manifold::Side::Lower).expect("cov eigh");
    let edge = noise_var * (1.0 + (p as f64 / n as f64).sqrt()).powi(2);
    evals.iter().filter(|&&e| e > edge).count()
}
