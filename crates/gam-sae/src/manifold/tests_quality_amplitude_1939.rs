//! #1939 OBJECTIVE-QUALITY acceptance bar — existence/intensity DECOUPLING in the
//! physical dictionary. The current representation carries intensity directly in
//! each fitted decoder. *Existence* (does this atom explain held-out structure)
//! must be identified separately from *intensity* (how large its contribution is).
//!
//! We plant that ground truth: two live circles on DISJOINT output subspaces whose
//! amplitudes differ by ~an order of magnitude, plus a DEAD atom slot with no
//! planted signal, and fit a K=3 dictionary. The objective is truth recovery —
//! the planted amplitude ratio and the dead/alive partition — NOT reproduction of
//! any reference tool's fitted parameters.
//!
//! Intensity lives in the decoder magnitude `‖B_k‖`; the executable objective bar
//! below therefore identifies a dead atom by its held-out contribution while
//! separately checking the recovered intensity ratio of the two live atoms.

use super::tests::deterministic_circle_noise;
use super::*;

/// Two circles on disjoint output-column parities with UNEQUAL amplitudes, plus a
/// third (dead) subspace that carries no planted signal — returned UN-whitened so
/// the planted amplitudes survive into the target (a column-standardized target
/// would quotient exactly the intensity this test measures).
///
/// * circle A: even output columns, amplitude `amp_a` (the strong atom),
/// * circle B: odd output columns, amplitude `amp_b` (the weak-but-real atom),
/// * the remaining variance is small isotropic noise (no third circle).
fn two_unequal_circles_plus_dead(
    n: usize,
    p: usize,
    amp_a: f64,
    amp_b: f64,
    sigma: f64,
) -> Array2<f64> {
    let mut fa = Array2::<f64>::zeros((2, p));
    let mut fb = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        if j % 2 == 0 {
            fa[[0, j]] = deterministic_circle_noise(j, 0);
            fa[[1, j]] = deterministic_circle_noise(j, 1);
        } else {
            fb[[0, j]] = deterministic_circle_noise(j, 2);
            fb[[1, j]] = deterministic_circle_noise(j, 3);
        }
    }
    // Orthonormalize each planted 2-frame so the circle's ambient radius is 1 and
    // the decoded amplitude is exactly `amp_*` (not tangled with frame scale).
    for f in [&mut fa, &mut fb] {
        for r in 0..2 {
            let nrm = (0..p).map(|j| f[[r, j]] * f[[r, j]]).sum::<f64>().sqrt();
            for j in 0..p {
                f[[r, j]] /= nrm.max(1.0e-300);
            }
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let ta = std::f64::consts::TAU * (row as f64) / (n as f64);
        // Use frequency 3 for the weak circle. Each fitted circle basis reaches
        // only harmonic 2, so the strong frequency-1 chart cannot also absorb the
        // weak signal as its second harmonic. A frequency-2 planting would make
        // decoder ownership non-identifiable before existence/intensity is tested.
        let tb = std::f64::consts::TAU * (3.0 * row as f64 + 0.37) / (n as f64);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for j in 0..p {
            z[[row, j]] = amp_a * (ca * fa[[0, j]] + sa * fa[[1, j]])
                + amp_b * (cb * fb[[0, j]] + sb * fb[[1, j]])
                + sigma * deterministic_circle_noise(row, j + 7);
        }
    }
    z
}

/// Per-atom RMS of the GATED decoded contribution `a_ik · (Φ_k B_k)_i` over all
/// rows and output columns — the atom's physical intensity as it enters the
/// reconstruction (existence × intensity combined into an output-energy readout).
fn per_atom_contribution_rms(term: &SaeManifoldTerm) -> Vec<f64> {
    let n = term.n_obs();
    let p = term.output_dim();
    let k = term.k_atoms();
    let mut sumsq = vec![0.0_f64; k];
    let mut buf = vec![0.0_f64; p];
    for row in 0..n {
        let weights = term
            .assignment
            .try_assignments_row(row)
            .expect("row index is below n_obs and the buffer is k_atoms wide");
        for atom in 0..k {
            let a_k = weights[atom];
            term.atoms[atom].fill_decoded_row(row, &mut buf);
            for &g in buf.iter() {
                let v = a_k * g;
                sumsq[atom] += v * v;
            }
        }
    }
    sumsq
        .into_iter()
        .map(|s| (s / (n as f64 * p as f64)).sqrt())
        .collect()
}

/// Even-column energy fraction of an atom's decoder — the subspace fingerprint
/// used to match a fitted atom to circle A (even, ~1.0) vs circle B (odd, ~0.0).
fn even_energy_fraction(atom: &SaeManifoldAtom, p: usize) -> f64 {
    let b = atom.decoder_coefficients();
    let mut e_even = 0.0_f64;
    let mut e_odd = 0.0_f64;
    for col in 0..b.nrows() {
        for out in 0..p {
            let v = b[[col, out]] * b[[col, out]];
            if out % 2 == 0 {
                e_even += v;
            } else {
                e_odd += v;
            }
        }
    }
    e_even / (e_even + e_odd).max(1.0e-300)
}

