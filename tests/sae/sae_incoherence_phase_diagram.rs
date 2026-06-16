//! #1008 phase-diagram validation — the conservative curved-dictionary
//! global-optimality certificate must have **no certified-but-wrong cell, ever**
//! (the same standard as `tests/topology_race_calibration.rs`).
//!
//! The certificate certifies global optimality up to the residual gauge iff
//!
//! ```text
//!   μ̂  ≤  c0 · a_floor² · (1 − 1/SNR) · (1 − C_κ·κ̂) / K
//! ```
//!
//! with the preconditions `C_κ·κ̂ < 1` (tangent-graph validity) and `SNR > 1`.
//! This sweep plants a `(μ, κ, SNR)` grid and asserts the properties that make a
//! certified-but-wrong verdict impossible:
//!
//! 1. **Precondition safety** — a cell whose curvature voids the tangent-graph
//!    perturbation (`κ̂ ≥ 1/C_κ`) or whose SNR is at/below the noise floor
//!    (`SNR ≤ 1`) is NEVER certified. These are exactly the regimes where the
//!    linear-case recovery argument does not transfer, so certifying them would
//!    be a wrong claim.
//! 2. **Monotone certified region** — certification is monotone non-increasing
//!    in μ and κ and non-decreasing in SNR. The certified set is therefore a
//!    contiguous "benign corner" with no interior hole where a wrong cell could
//!    hide: if a cell is uncertified, every harder cell (larger μ/κ, smaller
//!    SNR) is uncertified too.
//! 3. **Subset of a conservative recovery boundary** — every certified cell lies
//!    strictly inside an independent, deliberately *loose* analytic recovery
//!    boundary (a cell is "plausibly recoverable" only if its μ is below the
//!    benign-regime incoherence bound `a_floor²·(1−1/SNR)·(1−κ)/K`, the
//!    constant-free version of the threshold). The certificate uses a strictly
//!    smaller constant `c0 < 1`, so its certified set must be a strict subset:
//!    it never certifies a cell the loose boundary already rules out.

use gam::terms::sae_manifold::{
    GlobalOptimalityVerdict, SAE_CERT_CURVATURE_CONSTANT, SAE_CERT_INCOHERENCE_BUDGET,
    curved_dictionary_global_optimality_verdict,
};

const K_ATOMS: usize = 3;
const A_FLOOR: f64 = 0.8;

/// A grid point in the `(μ, κ, SNR)` planted phase diagram.
struct Cell {
    mu: f64,
    kappa: f64,
    snr: f64,
}

fn grid() -> Vec<Cell> {
    let mus = [0.0, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0];
    let kappas = [0.0, 0.1, 0.3, 0.6, 0.9, 0.99, 1.0, 1.5, 3.0];
    let snrs = [0.5, 0.9, 1.0, 1.01, 1.5, 3.0, 10.0, 100.0];
    let mut cells = Vec::new();
    for &mu in &mus {
        for &kappa in &kappas {
            for &snr in &snrs {
                cells.push(Cell { mu, kappa, snr });
            }
        }
    }
    cells
}

fn verdict(cell: &Cell) -> GlobalOptimalityVerdict {
    curved_dictionary_global_optimality_verdict(cell.mu, cell.kappa, A_FLOOR, cell.snr, K_ATOMS)
}

#[test]
fn no_certified_cell_violates_a_precondition() {
    // Property 1: a certified cell can NEVER be in a precondition-failing regime.
    // Those are precisely the cells where the recovery theory does not apply, so
    // a certification there would be the one unforgivable error.
    let graph_validity_bound = 1.0 / SAE_CERT_CURVATURE_CONSTANT;
    for cell in grid() {
        if verdict(&cell).is_certified() {
            assert!(
                cell.kappa < graph_validity_bound,
                "certified cell with κ̂={} ≥ tangent-graph bound {} — the \
                 perturbation off the linear case is uncontrolled, this is a \
                 certified-but-wrong cell",
                cell.kappa,
                graph_validity_bound
            );
            assert!(
                cell.snr > 1.0,
                "certified cell with SNR={} ≤ 1 — signal not above noise, the \
                 recovery argument does not apply",
                cell.snr
            );
            assert!(
                verdict(&cell).margin() > 0.0,
                "a certified cell must carry a strictly positive margin"
            );
        }
    }
}

#[test]
fn certified_region_is_monotone_no_interior_hole() {
    // Property 2: certification is monotone — harder cells (↑μ, ↑κ, ↓SNR) are
    // never certified when an easier one is not. A monotone certified region has
    // no interior hole, so there is no isolated wrong cell hiding inside it.
    let mus = [0.0, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0];
    let kappas = [0.0, 0.1, 0.3, 0.6, 0.9];
    let snrs = [1.01, 1.5, 3.0, 10.0, 100.0];

    // Monotone non-increasing in μ.
    for &kappa in &kappas {
        for &snr in &snrs {
            let mut prev_certified = true;
            for &mu in &mus {
                let c = verdict(&Cell { mu, kappa, snr }).is_certified();
                assert!(
                    !(c && !prev_certified),
                    "certification re-appeared at larger μ={mu} (κ={kappa}, \
                     SNR={snr}) — region not monotone in μ"
                );
                prev_certified = c;
            }
        }
    }
    // Monotone non-increasing in κ.
    for &mu in &mus {
        for &snr in &snrs {
            let mut prev_certified = true;
            for &kappa in &kappas {
                let c = verdict(&Cell { mu, kappa, snr }).is_certified();
                assert!(
                    !(c && !prev_certified),
                    "certification re-appeared at larger κ={kappa} (μ={mu}, \
                     SNR={snr}) — region not monotone in κ"
                );
                prev_certified = c;
            }
        }
    }
    // Monotone non-decreasing in SNR (ascending SNR ⇒ certification can only turn ON).
    for &mu in &mus {
        for &kappa in &kappas {
            let mut prev_uncertified = true;
            for &snr in &snrs {
                let uncertified = !verdict(&Cell { mu, kappa, snr }).is_certified();
                assert!(
                    !(uncertified && !prev_uncertified),
                    "certification disappeared at larger SNR={snr} (μ={mu}, \
                     κ={kappa}) — region not monotone in SNR"
                );
                prev_uncertified = uncertified;
            }
        }
    }
}

#[test]
fn certified_set_is_strict_subset_of_loose_recovery_boundary() {
    // Property 3: every certified cell is inside the constant-free benign-regime
    // recovery boundary `μ ≤ a²·(1−1/SNR)·(1−κ)/K`. The certificate's own budget
    // uses `c0 = SAE_CERT_INCOHERENCE_BUDGET < 1`, so a certified cell clears the
    // loose boundary with a factor `1/c0` of room to spare — it can never
    // certify a cell the loose boundary already excludes.
    assert!(
        SAE_CERT_INCOHERENCE_BUDGET < 1.0,
        "the incoherence budget constant must be < 1 for the certificate to be a \
         strict subset of the loose boundary"
    );
    let mut certified_count = 0usize;
    for cell in grid() {
        if !verdict(&cell).is_certified() {
            continue;
        }
        certified_count += 1;
        // κ < 1 and SNR > 1 hold (property 1); the loose boundary is then well
        // defined and positive.
        let loose_boundary =
            A_FLOOR * A_FLOOR * (1.0 - 1.0 / cell.snr) * (1.0 - cell.kappa) / K_ATOMS as f64;
        assert!(
            cell.mu < loose_boundary,
            "certified cell μ={} is NOT below the loose recovery boundary {} \
             (κ={}, SNR={}) — certified-but-wrong",
            cell.mu,
            loose_boundary,
            cell.kappa,
            cell.snr
        );
        // And strictly inside by the budget factor: μ ≤ c0 · boundary.
        assert!(
            cell.mu <= SAE_CERT_INCOHERENCE_BUDGET * loose_boundary + 1.0e-12,
            "a certified cell must clear the loose boundary by the conservative \
             factor c0={}; μ={}, boundary={}",
            SAE_CERT_INCOHERENCE_BUDGET,
            cell.mu,
            loose_boundary
        );
    }
    // The grid must actually exercise the certified branch — a vacuously-passing
    // sweep would prove nothing.
    assert!(
        certified_count > 0,
        "the planted grid produced no certified cell — the sweep is vacuous"
    );
}

#[test]
fn benign_corner_certifies_and_hard_corner_does_not() {
    // A clearly-benign cell (orthogonal frames μ=0, flat atoms κ=0, high SNR)
    // must certify; a clearly-hard cell (coherent frames, curved atoms, low SNR)
    // must not. Pins both ends of the diagram so the certificate is neither
    // vacuously-always-certified nor vacuously-never.
    let benign = curved_dictionary_global_optimality_verdict(0.0, 0.0, A_FLOOR, 100.0, K_ATOMS);
    assert!(
        benign.is_certified() && benign.margin() > 0.0,
        "the benign corner (μ=0, κ=0, SNR=100) must certify globally"
    );
    let hard = curved_dictionary_global_optimality_verdict(0.8, 0.95, A_FLOOR, 1.2, K_ATOMS);
    assert!(
        !hard.is_certified(),
        "the hard corner (coherent, curved, low SNR) must stay uncertified"
    );
}
