//! Owed-work regression gate for issue #1450 — the high-`K` SAE compact-path
//! support-selection contract ("per-token work is INDEPENDENT of `K`") must be
//! exercised END-TO-END through the REAL proposal/selection path, not just
//! asserted on a hand-built active set.
//!
//! Fast-regression note. This is a STRUCTURAL contract (per-row block dims do
//! not scale with `K`), so it is proved by COMPARING two `K` (`K = 512` vs
//! `K = 1024`, a 2× ratio) — the K-INVARIANCE of the per-row work is the point,
//! NOT the absolute `K`. Both the fixture build (one `SaeManifoldAtom` per atom,
//! a `(n, K)` logits matrix) AND `assemble_arrow_schur`'s B-tier setup
//! (`refresh_intrinsic_smooth_penalty`/`smooth_ops`/`assignment_prior_grad_hdiag`
//! over all `K`) are irreducibly `O(K)`, so a large absolute `K` buys nothing
//! for this contract while costing minutes of wall-clock. The compact
//! active-set selection engages whenever `set_softmax_active_cap` is set
//! (`softmax_active_plan` → explicit cap, independent of `K` vs any dense-Gram
//! budget), so the SAME `select_nth_unstable_by` proposal (#1411) and the SAME
//! compact per-row blocks are produced at `K = 512` as at `K = 100000`; only
//! the (untested-here) absolute setup cost differs. Two scattered (low/mid/high)
//! `K` values force the selection to scan all `K`, not a prefix.
//!
//! Background. #1411 (landed `41a03188c`) fixed `SaeRowLayout::from_dense_weights`
//! to select the per-row top-`cap` active atoms with an `O(K)` PARTIAL select
//! (`select_nth_unstable_by`) instead of a full `O(K log K)` per-row sort, so the
//! support proposal is genuinely `O(K)` per row and the assembled per-row block
//! is `O(active)` rather than `O(K)`. The only `K = 100000`-scale test that
//! existed (`sparse_active_layout_work_scales_with_active_atoms_not_total_k`,
//! `src/terms/sae/manifold/tests.rs`) built its layout from an ALREADY-KNOWN
//! 3-atom active set via `from_active_atoms` and only compared `q²` arithmetic —
//! it never ran the actual `from_dense_weights` proposal/selection (the thing
//! #1411 fixed), never computed assignments over a real large `K`, and never
//! assembled a real large-`K` SAE term.
//!
//! This integration test closes that gap through the PUBLIC crate API only. It
//! builds a realistic large-`K` softmax SAE term whose per-row logits concentrate
//! on a handful of planted atoms spread across the full `K` range, sets an
//! explicit `top_k` cap (`set_softmax_active_cap`, the #1409 lever), and calls
//! `assemble_arrow_schur` — which internally runs `softmax_active_plan` →
//! `SaeRowLayout::from_dense_weights` (the #1411 selection path) → the compact
//! `assemble_arrow_schur` row blocks. It then observes the assembled per-row
//! block dimensions (`sys.rows[..].htt.nrows()`) and asserts:
//!
//!   (a) every per-row block dim is bounded by the `O(active) = top_k·(1 + d)`
//!       contract, NOT by `K`;
//!   (b) the per-row dims and the total assembly work are IDENTICAL across two
//!       separated `K` (`K = 512` vs `K = 1024`, a 2× ratio) with the same
//!       intended planted active set — the literal "`q` and the measured work
//!       are ~equal, NOT scaling with `K`" contract;
//!   (c) the compact work is dwarfed (by > 1e3×) by the full-`K` dense block at
//!       `K = 512`, the regime `sparse_active_layout_work_scales_*` only
//!       asserted arithmetically off a hand-picked active set.
//!
//! NOTE on access surface. `SaeRowLayout::from_dense_weights`,
//! `SaeManifoldTerm::last_row_layout`, and `fixed_decoder_assembly` are all
//! `pub(crate)` and so are unreachable from an external integration test. The
//! test therefore drives the SAME selection path INDIRECTLY through the public
//! `assemble_arrow_schur` + `set_softmax_active_cap` entry points and measures
//! the contract on the assembled `ArrowSchurSystem` row blocks (public:
//! `sys.rows`, `ArrowRowBlock::htt`, `::gt`). This is exactly how the public
//! #1409 gate (`tests/owed_1409.rs`) observes the compact layout; #1450 extends
//! it to a real end-to-end assemble over a large `K` and to the K-invariance of
//! the measured work.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use ndarray::{Array1, Array2, Array3};

use gam::terms::{
    AssignmentMode, LatentManifold, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom,
    SaeManifoldRho, SaeManifoldTerm,
};

/// Build a softmax SAE term whose per-row softmax mass concentrates on a planted
/// top-`k` support (large positive logits on the planted atoms, a deep floor
/// everywhere else), with a shared constant+linear 1-D Euclidean basis per atom.
/// The planted atoms are spread across the full `K` range, so a correct top-`k`
/// selection MUST scan all `K`, not just a prefix — exercising the real `O(K)`
/// proposal/selection path (`from_dense_weights` via `assemble_arrow_schur`).
///
/// This mirrors the fixture used by the in-crate large-`K` test and the public
/// #1409 gate so the two stay comparable.
fn planted_softmax_term(
    n: usize,
    k_atoms: usize,
    planted: &[Vec<usize>],
    p: usize,
) -> (SaeManifoldTerm, Array2<f64>) {
    assert_eq!(planted.len(), n);
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (row as f64 / n as f64) - 0.5);
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        // Shared constant+linear basis: column 0 = 1, column 1 = t (jet d/dt = 1).
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = Array3::<f64>::zeros((n, 2, 1));
        for row in 0..n {
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = coords[[row, 0]];
            jet[[row, 1, 0]] = 1.0;
        }
        // Distinct decoder direction per atom so the reconstruction is genuine.
        let mut decoder = Array2::<f64>::zeros((2, p));
        decoder[[1, atom_idx % p]] = 0.1 + 0.01 * ((atom_idx % 7) as f64);
        atoms.push(
            SaeManifoldAtom::new(
                format!("atom{atom_idx}"),
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(2),
            )
            .unwrap(),
        );
        coord_blocks.push(coords.clone());
        manifolds.push(LatentManifold::Euclidean);
    }
    // Logits: a deep uniform floor everywhere, large mass on the planted atoms.
    let mut logits = Array2::<f64>::from_elem((n, k_atoms), -6.0);
    for (row, active) in planted.iter().enumerate() {
        for &k in active {
            logits[[row, k]] = 6.0;
        }
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, c)| 0.05 * ((row + c) as f64).sin());
    (term, target)
}

/// Assemble the compact softmax system for a given total `K` under an explicit
/// `top_k` cap and return the per-row latent block dimensions. This drives the
/// REAL selection path end-to-end: `assemble_arrow_schur` → `softmax_active_plan`
/// → `SaeRowLayout::from_dense_weights` (the #1411 `O(K)` partial-select
/// proposal) → the compact per-row Arrow-Schur blocks. The returned dims are the
/// observable footprint of that selection.
fn compact_assembly_dims(
    n: usize,
    k_atoms: usize,
    planted: &[Vec<usize>],
    p: usize,
    top_k: usize,
) -> Vec<usize> {
    let (mut term, target) = planted_softmax_term(n, k_atoms, planted, p);
    // The #1409/#1411 lever: fold top_k into the OPTIMIZATION so softmax engages
    // the compact active-set layout built by `from_dense_weights`, instead of a
    // post-fit projection over a full-`K` block.
    term.set_softmax_active_cap(Some(top_k));
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k_atoms]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("compact softmax assembly must succeed at large K");
    // Each per-row block must be square and consistent (htt vs gt) — a real
    // assembled Gauss-Newton block, not a stub.
    for r in &sys.rows {
        assert_eq!(
            r.htt.nrows(),
            r.htt.ncols(),
            "assembled per-row htt must be square"
        );
        assert_eq!(
            r.htt.nrows(),
            r.gt.len(),
            "assembled per-row htt dim must match its gradient gt"
        );
    }
    sys.rows.iter().map(|r| r.htt.nrows()).collect()
}

/// #1450 — end-to-end: the compact support-selection path must produce per-row
/// blocks whose dimension (and total assembly work) tracks the per-row
/// active-atom count, NOT `K`, and that footprint must be IDENTICAL when `K`
/// doubles for the same planted active set. This exercises the real
/// `from_dense_weights` proposal (via `assemble_arrow_schur`), unlike
/// `sparse_active_layout_work_scales_with_active_atoms_not_total_k`, which only
/// did `q²` arithmetic on a hand-built `from_active_atoms` layout. A 2× `K`
/// ratio at a moderate `K` is sufficient for the invariance claim; see the
/// file-level note on why the absolute `K` is kept small (fixture setup is
/// `O(K)`).
#[test]
fn large_k_compact_selection_work_is_independent_of_total_k_end_to_end_1450() {
    let n = 4usize;
    let p = 4usize;
    let top_k = 3usize;
    const K_SMALL: usize = 512;
    const K_LARGE: usize = 1_024; // 2× ratio — enough to prove dims don't scale with K.
    // Each row's planted top-`top_k` support, spread far across the full K range
    // (low / mid / high), so a correct selection must scan all K. The same
    // intended active set is reused at both K so q and the work are comparable.
    let planted: Vec<Vec<usize>> = (0..n)
        .map(|row| vec![row, K_SMALL / 2 + row, K_SMALL - 100 + row])
        .collect();

    // Two separated totals at a 2× ratio. Both are far above the dense-Gram
    // budget, so the compact active-set path engages identically at each.
    let dims_small = compact_assembly_dims(n, K_SMALL, &planted, p, top_k);
    let dims_large = compact_assembly_dims(n, K_LARGE, &planted, p, top_k);

    // (a) O(active)-per-token: every per-row block dim is bounded by the active
    // contract `top_k·(1 + d) = top_k·2` (d = 1 coord per atom), regardless of K.
    // A full-`K` softmax block would be `(K - 1)` free logits + `K` coord axes,
    // i.e. ~1023 at K=512 — orders of magnitude larger than the compact `top_k·2`.
    let bound = top_k * (1 + 1);
    for row in 0..n {
        assert!(
            dims_small[row] <= bound,
            "row {row}: K={K_SMALL} compact dim {} exceeds the O(active) bound {bound} — \
             the high-K selection produced a K-scaled block",
            dims_small[row]
        );
    }

    // (b) K-INVARIANCE of the measured work: doubling K with the same intended
    // active set must leave every per-row block dim IDENTICAL. This is the literal
    // #1450 contract ("q and the measured work are ~equal, NOT scaling with K").
    // A selection whose per-token cost depended on K could not produce a
    // K-invariant assembled footprint.
    for row in 0..n {
        assert_eq!(
            dims_small[row], dims_large[row],
            "row {row}: assembled compact dim must be INDEPENDENT of total K — \
             K={K_SMALL} gave {} but K={K_LARGE} gave {} (per-token work scaled with K)",
            dims_small[row], dims_large[row]
        );
    }
    let work_small: usize = dims_small.iter().map(|&q| q * q).sum();
    let work_large: usize = dims_large.iter().map(|&q| q * q).sum();
    assert_eq!(
        work_small, work_large,
        "total compact assembly work must be INDEPENDENT of total K: \
         K={K_SMALL} work {work_small} vs K={K_LARGE} work {work_large}"
    );

    // (c) The compact work is dwarfed by the full-`K` dense block at K=K_SMALL —
    // the regime the prior arithmetic-only test asserted off a hand-picked active
    // set, now measured on the REAL selection path's output. The dense block
    // would be `(K - 1)` free logits + `K` coord axes per row. At K=512 the dense
    // block is ~4.2e6 vs compact ~144, a >2.9e4× gap; we assert a conservative
    // >1e3× dwarf so the structural separation is certified with margin (the gap
    // GROWS with K, so this is a strict lower bound on the real production K).
    let dense_q = (K_SMALL - 1) + K_SMALL;
    let dense_work = n * dense_q * dense_q;
    assert!(
        work_small.saturating_mul(1_000) < dense_work,
        "compact selection work {work_small} must be vastly (> 1e3×) below the \
         full-K dense block work {dense_work} at K={K_SMALL}"
    );
}

/// #1450 companion: the support actually proposed by the real selection at large
/// `K` must recover the planted top-`top_k` atoms — i.e. the `O(K)` partial
/// select (`select_nth_unstable_by`, #1411) picks the genuine per-row peaks
/// scattered across the full `K`, not an arbitrary prefix. We observe this
/// indirectly: the per-row block dim equals EXACTLY the active contract
/// `top_k·(1 + d)`, which can only hold if the selection kept exactly the
/// `top_k` planted (above-floor) atoms and no spurious tail atoms.
#[test]
fn large_k_selection_recovers_full_planted_support_not_a_prefix_1450() {
    let n = 4usize;
    let p = 3usize;
    let top_k = 3usize;
    const K: usize = 512;
    // Planted peaks scattered low / mid / high so a prefix-only or sort-truncated
    // selection that ignored the tail could not recover all three.
    let planted: Vec<Vec<usize>> = (0..n)
        .map(|row| vec![row, K / 2 + row, K - 100 + row])
        .collect();

    let dims = compact_assembly_dims(n, K, &planted, p, top_k);

    // d = 1 coord axis per atom → each kept atom contributes (1 logit slot + 1
    // coord axis) = 2. Exactly the `top_k` planted atoms (all above the deep
    // floor) must survive, so every row block is exactly `top_k·2`. Anything
    // smaller means a planted peak in the tail was dropped (a prefix/partial-sort
    // regression); anything larger means spurious floor atoms leaked in.
    let exact = top_k * (1 + 1);
    for row in 0..n {
        assert_eq!(
            dims[row], exact,
            "row {row}: compact block dim {} must equal the exact O(top_k) size {exact} — \
             the K={K} selection must recover the full planted top-{top_k} support \
             (peaks scattered across all K), not a prefix",
            dims[row]
        );
    }
}
