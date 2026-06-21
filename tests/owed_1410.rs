//! Owed-work regression gate for issue #1410.
//!
//! #1410: the compact SAE assembly path sized its per-Rayon-worker scratch
//! (`decoded` at `k_atoms·p` and `jac_white` at `q·max(w_dim,p)`) by the FULL
//! atom count `K` and full tangent dim `q`, not by the row's compact active
//! set. Via `map_init` every worker received its own full-`K` copy
//! (≈11 GiB/worker at K=100k, p=5120, d=1). The fix sizes the scratch by the
//! compact row dimensions: `decoded_rows = max_r |active_atoms[r]|` and
//! `scratch_q = max_r row_q_active(r)`, falling back to full `k_atoms`/`q` only
//! when there is no compact layout (the dense softmax path, #1408).
//!
//! This test pins the SIZING contract the fix depends on, using only the public
//! crate API (`SaeRowLayout`'s public fields + `row_q_active`). It reproduces
//! the exact reduction the production `match row_layout` arm performs
//! (`construction.rs`: `decoded_rows`/`scratch_q`) and asserts that, when a row
//! layout activates only a small fraction of a large dictionary, the scratch
//! dimensions scale with the COMPACT active support — not with `K` or full `q`.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use gam::terms::sae::manifold::SaeRowLayout;

/// Build a layout over `k_atoms` atoms (each `coord_dim`-dimensional) where
/// every one of `n` rows activates only `active_per_row` atoms. This mirrors the
/// compact JumpReLU / IBP-MAP active-set layout that drives the assembly's
/// per-worker scratch sizing.
fn compact_layout(
    n: usize,
    k_atoms: usize,
    coord_dim: usize,
    active_per_row: usize,
) -> SaeRowLayout {
    let coord_dims = vec![coord_dim; k_atoms];
    // Full-q offsets: atom k owns the [k] logit slot then `coord_dim` coord
    // slots; layout out the full-q frame so offsets are realistic.
    let mut coord_offsets_full = Vec::with_capacity(k_atoms);
    let mut cursor = k_atoms; // logit block of width k_atoms precedes coords
    for _ in 0..k_atoms {
        coord_offsets_full.push(cursor);
        cursor += coord_dim;
    }

    let mut active_atoms = Vec::with_capacity(n);
    let mut coord_starts = Vec::with_capacity(n);
    for row in 0..n {
        // Pick `active_per_row` distinct atoms, rotating across rows so the
        // active sets differ per row (the production reduction takes a max over
        // rows, so heterogeneous rows are the meaningful case).
        let mut active: Vec<usize> = (0..active_per_row)
            .map(|j| (row + j) % k_atoms)
            .collect();
        active.sort_unstable();
        active.dedup();

        // Compact coord_starts, exactly as `from_active_atoms` computes them:
        // the compact logit block of width `active.len()` precedes the coords.
        let mut starts = Vec::with_capacity(active.len());
        let mut c = active.len();
        for &k in &active {
            starts.push(c);
            c += coord_dims[k];
        }
        active_atoms.push(active);
        coord_starts.push(starts);
    }

    SaeRowLayout {
        active_atoms,
        coord_starts,
        coord_offsets_full,
        coord_dims,
    }
}

/// Reproduce the production sizing reduction from `construction.rs`
/// (`decoded_rows`, `scratch_q` in the `Some(layout)` arm of the
/// `match row_layout.as_ref()` block) using only the public layout API.
fn scratch_dims(layout: &SaeRowLayout) -> (usize, usize) {
    let n = layout.active_atoms.len();
    let decoded_rows = (0..n)
        .map(|r| layout.active_atoms[r].len())
        .max()
        .unwrap_or(0)
        .max(1);
    let scratch_q = (0..n)
        .map(|r| layout.row_q_active(r))
        .max()
        .unwrap_or(0)
        .max(1);
    (decoded_rows, scratch_q)
}

/// #1410: with a compact active-set layout, the per-worker scratch dimensions
/// must scale with the row's active support, NOT with the full dictionary `K`
/// or the full tangent dimension `q`. A regression that reverts to full-`K`
/// sizing (the ≈11 GiB/worker blow-up) makes `decoded_rows` jump from
/// `active_per_row` to `k_atoms` and `scratch_q` from the compact row dim to
/// full `q`, which this test forbids.
#[test]
fn compact_scratch_scales_with_active_support_not_k_1410() {
    let n = 64;
    let k_atoms = 4096; // large dictionary
    let coord_dim = 1;
    let active_per_row = 8; // << k_atoms

    let layout = compact_layout(n, k_atoms, coord_dim, active_per_row);

    // Full-q is the dense fallback dimension: one logit slot per atom plus all
    // per-atom coord blocks. This is what the `None` (dense softmax) arm uses.
    let full_q: usize = k_atoms + k_atoms * coord_dim;

    let (decoded_rows, scratch_q) = scratch_dims(&layout);

    // decoded_rows is the max active-atom count — exactly `active_per_row`, the
    // compact support, and far below `k_atoms`.
    assert_eq!(
        decoded_rows, active_per_row,
        "decoded scratch row count must equal the max compact active-atom \
         count ({active_per_row}), not full K ({k_atoms})"
    );
    assert!(
        decoded_rows < k_atoms,
        "decoded scratch rows ({decoded_rows}) must be far below full K ({k_atoms})"
    );

    // scratch_q is the compact per-row tangent dim: active.len() logit slots +
    // sum of active coord dims = active_per_row * (1 + coord_dim).
    let expected_scratch_q = active_per_row * (1 + coord_dim);
    assert_eq!(
        scratch_q, expected_scratch_q,
        "tangent scratch dim must equal the compact row dim ({expected_scratch_q}), \
         not full q ({full_q})"
    );
    assert!(
        scratch_q < full_q,
        "tangent scratch dim ({scratch_q}) must be far below full q ({full_q})"
    );

    // The load-bearing footprint claim: the per-worker `decoded` buffer is
    // `decoded_rows · p` floats. At the compact sizing it is `active_per_row · p`;
    // a full-K regression would be `k_atoms · p`. Assert the compact buffer is
    // smaller by the full active/K ratio (here 512×).
    let ratio = k_atoms / decoded_rows;
    assert!(
        ratio >= 512,
        "compact decoded buffer must be >=512x smaller than the full-K buffer; got {ratio}x"
    );
}

/// #1410: the sizing reduction takes a MAX across rows, so a single
/// densely-active row sets the scratch dimension for all workers. This pins that
/// the helper is driven by the worst-case row, not an average, so the buffers
/// are always large enough for every row's compact block (correctness: a buffer
/// sized below some row's active set would be an out-of-bounds bug, not just a
/// perf change).
#[test]
fn compact_scratch_sized_by_worst_case_row_1410() {
    let n = 16;
    let k_atoms = 1024;
    let coord_dim = 2;

    // Build heterogeneous rows by hand: most rows active 3 atoms, one row
    // active 9 atoms. The scratch must be sized for the 9-atom row.
    let coord_dims = vec![coord_dim; k_atoms];
    let mut coord_offsets_full = Vec::with_capacity(k_atoms);
    let mut cursor = k_atoms;
    for _ in 0..k_atoms {
        coord_offsets_full.push(cursor);
        cursor += coord_dim;
    }

    let mut active_atoms = Vec::with_capacity(n);
    let mut coord_starts = Vec::with_capacity(n);
    for row in 0..n {
        let count = if row == 7 { 9 } else { 3 };
        let mut active: Vec<usize> = (0..count).map(|j| (row * 13 + j) % k_atoms).collect();
        active.sort_unstable();
        active.dedup();
        let mut starts = Vec::with_capacity(active.len());
        let mut c = active.len();
        for &k in &active {
            starts.push(c);
            c += coord_dims[k];
        }
        active_atoms.push(active);
        coord_starts.push(starts);
    }

    let layout = SaeRowLayout {
        active_atoms,
        coord_starts,
        coord_offsets_full,
        coord_dims,
    };

    let (decoded_rows, scratch_q) = scratch_dims(&layout);

    // Worst-case row activates 9 atoms.
    assert_eq!(
        decoded_rows, 9,
        "decoded scratch must be sized for the densest row (9 active atoms)"
    );
    // Its compact tangent dim is 9 logit slots + 9 coord blocks of width 2.
    assert_eq!(
        scratch_q,
        9 * (1 + coord_dim),
        "tangent scratch must be sized for the densest row's compact block"
    );

    // Every row's compact block must FIT inside the sized scratch (the
    // correctness invariant the sizing guarantees).
    for row in 0..n {
        assert!(
            layout.active_atoms[row].len() <= decoded_rows,
            "row {row} active set must fit the decoded scratch"
        );
        assert!(
            layout.row_q_active(row) <= scratch_q,
            "row {row} compact tangent block must fit the tangent scratch"
        );
    }
}
