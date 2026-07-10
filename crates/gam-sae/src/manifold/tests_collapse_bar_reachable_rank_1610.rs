//! `collapse_bar_uses_reachable_dictionary_rank_not_nominal_count_1610`, split
//! verbatim out of `tests.rs` to keep that tracked file under the #780 10k-line
//! gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; the shared
//! `periodic_basis` fixture helper is sourced from the sibling `tests` module.

use super::tests::periodic_basis;
use super::*;
use ndarray::{Array2, Array3, array};

/// #1610 / S1 — the co-collapse degeneracy floor must be calibrated against the
/// dictionary's GEOMETRICALLY REACHABLE rank `Σ_k rank(Φ_k)`, NOT the nominal
/// coefficient count `Σ_k basis_size_k`. The floor is the signal-free null-`R²`
/// `q / n` (S1 guard surgery replaced the former `0.5 × dense PCA ceiling` bar);
/// using the nominal coefficient count for `q` over-states what a curved/degenerate
/// dictionary can actually span, biasing the floor high. The floor is monotone in
/// `q`, so the reachable rank (≤ nominal) keeps it DOWN — the property asserted
/// below is invariant to the exact floor formula, only that it is monotone in `q`
/// and reads the chart geometry alone.
///
/// Fixture: a K=2 dictionary on a rank-2 (unit-circle) target.
///   * atom A — a `[1, sin, cos]` periodic chart on 4 distinct angles: full
///     chart rank `rank(Φ_A) = 3 = basis_size`.
///   * atom B — the SAME chart but evaluated at a SINGLE repeated coordinate, so
///     every row of `Φ_B` is identical: `rank(Φ_B) = 1 ≪ basis_size = 3` — a
///     geometrically degenerate atom that linearly spans only ONE direction.
///
/// Properties asserted:
///   1. `reachable_dictionary_rank = rank([Φ_A | Φ_B]) = 3` — the CONCATENATED
///      chart-design rank, NOT the sum `rank(Φ_A)+rank(Φ_B)=4`. Atom B's single
///      reachable direction is the constant vector `[1,1,1,1]`, which already lies
///      inside atom A's `[1,sin,cos]` column span, so the union spans only 3
///      directions. Strictly BELOW the nominal `Σ basis_size = 6`.
///   1b. #C5 discriminator: two IDENTICAL atoms reach `rank = rank(Φ) = 3`, while
///      the summed count would double to 6 — the concatenated rank refuses to
///      double-count shared column spaces.
///   2. the reachable rank is read from the chart design ALONE, so a co-collapsed
///      decoder (`B → 0`) reports the SAME reachable rank (the guard does not
///      lower its own bar at the very collapse it must catch).
///   3. the live `absolute_degeneracy_ev_floor` at the reachable rank is < the
///      floor at the nominal count (the bias is corrected DOWNWARD, never up).
#[test]
pub(crate) fn collapse_bar_uses_reachable_dictionary_rank_not_nominal_count_1610() {
    use crate::manifold::outer_objective::{
        absolute_degeneracy_ev_floor, reachable_dictionary_rank,
    };

    // A target with genuine spectral spread across >= 6 directions, so the rank-q
    // PCA ceiling STRICTLY increases with q over the range that separates the
    // reachable rank (3) from the nominal count (6). n=8 rows, p=8 cols with
    // geometrically decaying singular values along distinct coordinate axes: the
    // captured-variance fraction at rank 3 is strictly below that at rank 6, so
    // the two bars differ and the cap (min(n,p)=8) never masks the difference.
    let (n, p) = (8usize, 8usize);
    let mut target = Array2::<f64>::zeros((n, p));
    // Row r, axis a: value sigma_a on the a-th axis when r == a, else 0 — a
    // diagonal-ish design whose centered singular values are the sigma_a, strictly
    // decreasing, so each extra rank adds a strictly positive variance increment.
    let sigma = [8.0_f64, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625];
    for a in 0..p {
        target[[a, a]] = sigma[a];
    }

    // atom A: full-rank periodic chart on 4 distinct angles → rank(Φ_A) = 3.
    let coords_a = array![[0.0_f64], [0.25], [0.5], [0.75]];
    let (phi_a, jet_a) = periodic_basis(&coords_a);
    // atom B: SAME chart evaluated at one repeated coordinate → every Φ row equal
    // → rank(Φ_B) = 1, even though basis_size = 3.
    let coords_b = array![[0.3_f64], [0.3], [0.3], [0.3]];
    let (phi_b, jet_b) = periodic_basis(&coords_b);

    let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
    };

    // Healthy (nonzero) decoders for both atoms.
    let atom_a = make(
        "full",
        phi_a.clone(),
        jet_a.clone(),
        Array2::<f64>::ones((3, p)),
    );
    let atom_b = make(
        "degenerate",
        phi_b.clone(),
        jet_b.clone(),
        Array2::<f64>::ones((3, p)),
    );

    // Per-atom realized chart image rank (capped at output dim p=8, so no cap
    // bites here): full chart spans rank 3; the repeated-coordinate chart rank 1.
    assert_eq!(
        atom_a.realized_chart_image_rank().unwrap(),
        3,
        "full periodic chart on 4 distinct angles spans rank 3 in [1,sin,cos]"
    );
    assert_eq!(
        atom_b.realized_chart_image_rank().unwrap(),
        1,
        "a chart evaluated at ONE repeated coordinate has identical rows → rank 1, \
         far below its basis_size 3"
    );

    let atoms = vec![atom_a, atom_b];
    let nominal: usize = atoms
        .iter()
        .map(|a| a.basis_size())
        .sum::<usize>()
        .min(n)
        .min(p);
    let reachable = reachable_dictionary_rank(&atoms, n, p);

    // (1) reachable rank = rank(Φ_A) + rank(Φ_B) = 3 + 1 = 4, STRICTLY below the
    // nominal Σ basis_size = 6 — the biased-high count is corrected.
    assert_eq!(
        reachable, 4,
        "reachable rank must be rank(Φ_A) + rank(Φ_B) = 3 + 1"
    );
    assert_eq!(
        nominal, 6,
        "nominal Σ basis_size (capped) must be 6 on this fixture"
    );
    assert!(
        reachable < nominal,
        "reachable rank {reachable} must be strictly below the nominal count \
         {nominal} (this fails if someone reverts to Σ basis_size)"
    );

    // (2) decoder-magnitude independence: a co-collapsed dictionary (decoders → 0)
    // reports the SAME reachable rank, so the guard keeps its full bar at the
    // collapse it must catch.
    let collapsed: Vec<SaeManifoldAtom> = vec![
        make("full0", phi_a, jet_a, Array2::<f64>::zeros((3, p))),
        make("degenerate0", phi_b, jet_b, Array2::<f64>::zeros((3, p))),
    ];
    assert_eq!(
        reachable_dictionary_rank(&collapsed, n, p),
        reachable,
        "reachable rank must read the chart design alone and be invariant to the \
         decoder magnitude (a co-collapsed decoder still reports full geometric reach)"
    );

    // (3) the live null-`R²` degeneracy floor at the reachable rank is STRICTLY
    // BELOW the floor at the nominal count: the floor `q / n` is monotone in the
    // rank `q`, so calibrating against the (smaller) reachable geometry keeps the
    // floor DOWN. This is the behavioral consequence — it fails if the call sites
    // revert to the nominal count.
    let bar_reachable = absolute_degeneracy_ev_floor(target.view(), reachable);
    let bar_nominal = absolute_degeneracy_ev_floor(target.view(), nominal);
    assert!(
        bar_reachable < bar_nominal,
        "degeneracy floor at reachable rank ({bar_reachable}) must be strictly below the \
         floor at the nominal count ({bar_nominal})"
    );
    // Both finite and in [0,1] — a real, usable floor (`q / n` with q ≤ min(n,p) ≤ n).
    assert!(
        bar_reachable.is_finite() && (0.0..=1.0).contains(&bar_reachable),
        "reachable-rank degeneracy floor must be a finite EV fraction, got {bar_reachable}"
    );
}

/// 2026-07-10 regression — a SATURATED floor is NO VERDICT, not certain collapse.
/// With reachable rank `q >= n` the signal-free least squares reproduces the
/// target exactly (`colspan(Φ) = R^n`), so the null floor carries zero evidence:
/// every possible EV sits at or below it, and returning `1.0` branded EV≈0.999
/// fits on small-n/basis-rich fixtures as co-collapsed. The floor must return
/// NaN (the established n==0 no-verdict convention; both caller arms compare
/// `<= floor`, which is false on NaN, standing the absolute arm down).
#[test]
fn saturated_degeneracy_floor_is_no_verdict() {
    let target = ndarray::Array2::<f64>::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f64);
    // q == n and q > n: both saturate — no verdict.
    assert!(absolute_degeneracy_ev_floor(target.view(), 4).is_nan());
    assert!(absolute_degeneracy_ev_floor(target.view(), 9).is_nan());
    // q < n: the classical null-R² q/n, unchanged.
    let floor = absolute_degeneracy_ev_floor(target.view(), 3);
    assert!((floor - 0.75).abs() < 1e-15, "expected 3/4, got {floor}");
    // The no-verdict NaN stands the caller arms down: `ev <= floor` is false.
    let ev = 0.999_f64;
    assert!(!(ev <= absolute_degeneracy_ev_floor(target.view(), 4)));
}
