//! Massive-K SAE encode — K-scaling throughput curve (issue #988).
//!
//! ## Why this test exists
//!
//! `encode_full_path_throughput` measures the certified encode against a SINGLE
//! atom. The user's target is the massive-K manifold SAE (K up to 32,000), where
//! the encode's dominant cost is per-row ATOM ROUTING over the whole dictionary.
//! The naive router (`route_exact`'s universal-bound certificate never fires for a
//! realistic dictionary, so it falls back to an O(K) full scan per row) makes the
//! whole encode O(N·K) — it blows up at K=32k.
//!
//! The production SPEED path (`amortized_encode_with_index_fast` /
//! `amortized_reconstruct_with_index_fast`) instead routes each row via the
//! sublinear LSH gather (`SaeCandidateIndex::propose`, which touches only
//! `~num_tables·bucket_occupancy = O(log K)` atoms and scores `~budget = 8·log2(K)`
//! candidates), so the routing — and therefore the whole fast encode→decode — is
//! sublinear in K. The per-atom encode is K-independent (each routed row only
//! touches its own atom's chart atlas).
//!
//! ## What it asserts
//!
//! It builds K circle atoms embedded into distinct subspaces of `R^p`, an LSH
//! index + sketch over them, and the certified [`EncodeAtlas`], then TIMES the
//! fast index-routed encode over a fixed batch of `N` rows at `K = 1024, 8192,
//! 32000` and reports rows/sec. The contract: throughput does NOT collapse
//! linearly with K — the K=32000 rate stays a large fraction of the K=1024 rate
//! (sublinear), which an O(N·K) router could never do (its rate would fall ~31×).

