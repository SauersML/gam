//! #2023 — the co-collapse reseed's seed provenance, with the positive control
//! the acceptance bar has never had.
//!
//! #2023's bar is "no PC reseed events in the log". At `origin/main` before this
//! module, that bar could not fail, for four independent reasons:
//!
//!   1. `BirthSeed::PrincipalComponent` is constructed NOWHERE in the tree —
//!      every occurrence is its own definition, its own `matches!`, its own
//!      `code()` arm, and doc comments.
//!   2. `fit_drivers.rs`, which performs the live PC reseed, contained zero
//!      references to the migration ledger, so the one path that can violate the
//!      bar could not reach the counter.
//!   3. `record_search_round` stamps every accepted structure-search birth
//!      `BirthSeed::ResidualFactor` unconditionally — the seed is ASSERTED, not
//!      observed.
//!   4. `assert_no_pc_reseed`, named in the ledger's own header as the thing
//!      that "fails loudly", did not exist.
//!
//! `b0e19d6e8` fixed (4) and gave the ledger's counter its own control. This
//! module covers (2): `SaeManifoldTerm` now counts, per reseeded ATOM, which
//! branch of `reseed_atoms_onto_distinct_residual_pcs` actually ran, and the
//! tests below require BOTH branches to be observed firing.
//!
//! That two-sidedness is the point. A counter wired to a constant passes one arm
//! and fails the other; a counter incremented in both branches fails whichever it
//! does not belong to. Only a counter that tracks the branch actually taken
//! passes both — and only then does a later reading of zero mean the reseed drew
//! from data rows rather than meaning nobody was watching.

use super::tests::{small_two_atom_periodic_term, trivial_k1_euclidean_term};
use super::*;

/// A chart-kind atom takes the PRINCIPAL-COMPONENT branch even with the #2023
/// data-row lever switched ON, and the counter says so.
///
/// This is the arm that makes every downstream "no PC reseeds" reading
/// meaningful, and it also pins the audit finding that motivated it: the
/// data-row rule's `all_flat` guard admits only `EuclideanPatch | Linear`, so
/// curved atoms — the ones Tier 2 is made of — can never take it. Enabling the
/// lever and still landing on PCA is not a bug here; it is the documented
/// behaviour, and pinning it stops the lever from being mistaken for a
/// fit-wide switch.
#[test]
fn chart_atoms_reseed_from_principal_components_even_with_the_lever_on_2023() {
    let (mut term, target, rho) = small_two_atom_periodic_term();
    // The lever ON, so a counter that merely mirrored the flag would fail here.
    term.set_data_row_reseed(true);
    assert_eq!(term.pc_reseeded_atoms, 0, "#2023: fresh terms have reseeded nothing");
    assert_eq!(term.data_row_reseeded_atoms, 0);

    // A large `pc_pair_offset` also clears the exhausted-pool condition, so the
    // ONLY thing sending this to the PCA branch is the atoms' chart kind.
    term.reseed_atoms_onto_distinct_residual_pcs(&[0, 1], target.view(), &rho, 8)
        .expect("#2023: the periodic fixture must reseed onto residual PCs");

    assert_eq!(
        term.pc_reseeded_atoms, 2,
        "#2023: the PC branch must count the ATOMS it reseeded (2), not the call (1) — \
         one exhausted-pool retry that re-plants several atoms on the same leading PCs \
         is several chances to re-collapse"
    );
    assert_eq!(
        term.data_row_reseeded_atoms, 0,
        "#2023: the data-row branch did not run and must not be credited"
    );

    // Accumulates across retries: the bar is over the whole fit's history, so a
    // second reseed must add rather than replace.
    term.reseed_atoms_onto_distinct_residual_pcs(&[0], target.view(), &rho, 9)
        .expect("#2023: a second reseed must succeed");
    assert_eq!(
        term.pc_reseeded_atoms, 3,
        "#2023: seed provenance accumulates over the fit; it is never reset, so a later \
         reseed cannot launder an earlier one"
    );
}

/// A flat atom, the lever ON, and the PC pool exhausted takes the DATA-ROW
/// branch — and the counter attributes it there rather than to PCs.
///
/// Without this arm the test above is satisfied by a counter that is simply
/// incremented unconditionally.
#[test]
fn flat_atoms_past_the_exhausted_pc_pool_reseed_from_data_rows_2023() {
    let mut term = trivial_k1_euclidean_term();
    let n = term.n_obs();
    let p = term.output_dim();
    // A residual with structure in every output direction, so neither branch is
    // degenerate on this fixture.
    let target =
        Array2::<f64>::from_shape_fn((n, p), |(row, col)| ((row + 1) as f64) * 0.25 - (col as f64) * 0.1);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    term.set_data_row_reseed(true);

    // `pc_pairs = min(p, n) / 2`; an offset at or past it is the exhausted pool
    // the data-row rule exists for.
    let pc_pairs = p.min(n) / 2;
    term.reseed_atoms_onto_distinct_residual_pcs(&[0], target.view(), &rho, pc_pairs.max(1))
        .expect("#2023: the euclidean fixture must reseed from data rows");

    assert_eq!(
        term.data_row_reseeded_atoms, 1,
        "#2023: a flat atom past the exhausted PC pool with the lever on must be \
         attributed to the DATA-ROW branch"
    );
    assert_eq!(
        term.pc_reseeded_atoms, 0,
        "#2023: the PC branch did not run and must not be credited — a counter \
         incremented unconditionally fails here"
    );
}

/// The lever OFF is the shipped default, and it sends a flat atom to the PC
/// branch however exhausted the pool is.
///
/// This is the audit's headline in executable form: the #2023 replacement rule
/// is implemented but `data_row_reseed` defaults to `false`, so the PC reseed —
/// #1893's mechanism, the one this issue exists to remove — is what production
/// runs today.
#[test]
fn the_data_row_rule_is_opt_in_so_the_default_is_still_a_pc_reseed_2023() {
    let mut term = trivial_k1_euclidean_term();
    let n = term.n_obs();
    let p = term.output_dim();
    let target =
        Array2::<f64>::from_shape_fn((n, p), |(row, col)| ((row + 1) as f64) * 0.25 - (col as f64) * 0.1);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    // Deliberately NOT calling `set_data_row_reseed` — this is the default a
    // production fit gets.
    term.reseed_atoms_onto_distinct_residual_pcs(&[0], target.view(), &rho, p.min(n).max(1))
        .expect("#2023: the default path must still reseed");

    assert_eq!(
        term.pc_reseeded_atoms, 1,
        "#2023: with the lever at its DEFAULT the reseed draws from principal \
         components — this is the live #1893 mechanism, and the counter is what \
         makes its presence a measurement rather than an assertion"
    );
    assert_eq!(term.data_row_reseeded_atoms, 0);
}
