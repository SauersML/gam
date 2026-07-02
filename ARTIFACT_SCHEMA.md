# Tiered dictionary artifact — schema & contract (WS-J)

**Status:** v1 draft. This document is the *contract* the other SAC workstreams
emit into and consume from. Where a producing workstream's on-disk format is not
yet frozen, the loader in `examples/tiered_artifact.py` reads defensively and the
"producer contract" note below states exactly the fields WS-J needs. Conform to
those field names and the union is free.

The artifact is the Part-4 done-state object: *one command fits the tiered
dictionary on sharded activations of a frontier model and emits a canonical,
content-hashed artifact.* It is the union

```
TieredArtifact  =  T0 stats  ∪  T1 dictionary  ∪  T2 typed atoms
                   ∪  Σ residual model  ∪  encoder  ∪  content hash
```

behind four verbs — `fit / encode / steer / diff` (`examples/dictionary_cli.py`).

Per SPEC.md, this layer is *serialization / composition glue only*: no model math
lives here (the fitters are Rust; the arc-EV ledger and canonical-hash port are
the only arithmetic, both byte-format definitions rather than statistics). It is
the legitimate Python surface — everything it serializes was produced by a Rust
fitter or a frozen manifest.

## On-disk layout

A `TieredArtifact` serializes to a directory:

```
<artifact_dir>/
  artifact.json          # manifest: version, content_hash, tier pointers, T0, atom table
  t1_decoder.npy         # (K1, P) f32 unit-norm linear decoder  (T1)
  t1_codes.npz           # optional: training routing (indices,codes) if carried
  t2_manifold.json       # ManifoldSAE.to_dict() payload           (T2)
  sigma.npz              # Σ whitening factor (residual covariance root)  [optional]
  encoder/               # distilled encoder bundle                 [optional]
    encoder.pt           #   torch state_dict (DistilledEncoder module)
    encoder_meta.json    #   scale params, k_atoms, atom_dims, fallback calibration
```

`artifact.json` is the single source of truth for the content hash; every array
file's bytes are folded into it (see "Content hash" below), so a bit-identical
directory re-hashes identically and a tampered array file changes the hash.

## Tier T0 — corpus statistics

Producer: **WS-D data plane** (`ShardWriter` manifest `stats`) and/or **WS-C**
(baked into `dictionary_artifact`). Contract (already emitted by
`examples/residual_shard_io.py::ShardWriter._build_manifest`):

```json
"stats": { "mean": [P floats], "norm": [P floats] }
"d_model": P, "total_tokens": N
```

A **finalized** WS-D manifest (`finalize_harvest.py`, `compute_t0`) carries a full
`t0` block WS-J passes through verbatim — this is the frozen contract, matched
exactly by `load_t0_from_manifest`. As harvested on node2 (Qwen3-32B, layers
24/32/40, `d_model=5120`), the massive-activation ("rogue") dims are a nested
block, not a flat list:

```json
"t0": {
  "d_model": 5120,
  "mean": [P floats], "std": [P floats], "rms": [P floats],   // per-dim
  "scale_median_std": float,      // robust central per-dim std (whitening ref)
  "scale_median_rms": float,
  "rogue_dims": {                 // massive-activation dims (~376 for Qwen3-32B L24)
    "index":          [int, ...], // the coordinates removed before geometry
    "rms":            [float,...],
    "rms_over_median":[float,...],
    "mad_z":          [float,...],
    "rule": "rms>5*median_rms OR MAD-z>8"
  }
}
```

A **provisional** manifest carries only `stats: {mean, norm}`; T0 is then built
from those with `rogue_dims`/`scale` absent (recorded, never fabricated). The
content hash folds `mean/std/rms/scale_median_*` and the `rogue_dims.index` set
(the identity of the massive-activation dims), so two harvests that agree on the
massive-activation set and per-dim stats hash their T0 identically.

## Tier T1 — linear sparse dictionary

Producer: **WS-C tier-1 at scale** (`gamfit.sparse_dictionary_fit` /
`SparseDictStreamArtifact`), exported through the Rust `dictionary_artifact`.
Contract WS-J consumes:

- `decoder`: `(K1, P)` f32, unit-norm rows (`SparseDictionaryFit.decoder` or
  `SparseDictStreamArtifact.decoder`).
- optional `indices` `(N, active)` int + `codes` `(N, active)` f32 — the training
  routing. Streamed T1 (`SparseDictStreamArtifact`) carries only the decoder; that
  is sufficient — held-out routing is recomputed by `.transform(X)`.
- optional `t0`: WS-C bakes T0 into the same artifact ("export through
  `dictionary_artifact` with T0 stats baked in"); if present it populates the T0
  tier and the loader does not require a separate manifest.

## Tier T2 — typed curved atoms

Producer: **WS-A SAC engine**. `sac_fit` / `compose_tiers` emit a set of accepted
K=1 atoms; each is a `gamfit.ManifoldSAE` (single atom) or the assembled
multi-atom `ManifoldSAE` from terminal assembly. Contract:

- Preferred: a single `ManifoldSAE.to_dict()` payload for the whole T2 tier (the
  terminal-assembly object). WS-J stores it verbatim as `t2_manifold.json` and
  reads the per-atom typed report from it: `topology`, on-manifold `Θ` /
  curvature (`.curvature()`), shape band (`.shape_uncertainty()`), the
  `hybrid_split` curved-vs-linear verdict, and `ΔEV` per birth.
- Interim (SAC prototype, no terminal assembly yet): **WS-A writes a
  `examples/sac_results/` directory** with one `atom_<k>.json`
  (`ManifoldSAE.to_dict()` of the K=1 fit) plus a `sac_meta.json`:

  ```json
  { "ev_trace": [float, ...],          // combined EV after each birth (monotone)
    "birth_log": [ {"birth":i,"delta_ev":..,"accepted":bool,"hybrid":str,"r2":..}, ... ],
    "t1_ev": float, "combined_ev": float, "k": int,
    "d_atom": int, "atom_topology": str }
  ```

  This is exactly what `examples/sac_prototype.py::SacResult` already holds; WS-A
  needs only to dump it. `tiered_artifact.load_sac_result(dir)` reads either form.

## Σ — structured residual model

Producer: **WS-A** (the `StructuredResidualModel` refit on the running residual;
`structured_residual_passes` inside `sae_manifold_fit`). The whitening factor is
the residual-covariance root Σ^{1/2} used so the curved likelihood is whitened
from atom one. Contract (optional; when absent the artifact records
`"sigma": null` and encode/steer fall back to the identity metric):

```json
sigma.npz : { "root": (P, P) f32 }   // lower-triangular Σ^{1/2}, or a low-rank (P, r) factor
```

WS-A does not yet expose Σ as a standalone Python object; until it does, WS-J
stores whatever `converged_latents`/fit payload surfaces and marks Σ `pending`.

## Encoder — amortized distilled encoder

Producer: **WS-E encoder + corpus sweep** (`gamfit.distill.distill_encoder` →
`DistilledEncoder`; `encode_with_fallback` → `EncoderFallbackStats`). Contract:

```json
encoder/encoder_meta.json : {
  "k_atoms": int, "atom_dims": [int,...], "input_dim": P,
  "scale_center": [...], "scale_spread": [...],   // the _scale() calibration
  "fallback_calibration": { "gate": float, ... }, // encode_with_fallback gate
  "distilled_from_content_hash": "sha256:..."      // T2 hash the encoder was distilled on
}
encoder/encoder.pt : torch state_dict of the DistilledEncoder module
```

`distilled_from_content_hash` binds the encoder to the exact dictionary it was
distilled on — `encode` refuses an encoder whose hash does not match the loaded
artifact (a mismatched encoder silently defines a different feature map).

## Content hash

The canonical content hash is a faithful Python mirror of the Rust
`crates/gam-sae/src/dictionary_artifact.rs` v1 hash (the same one W9's
`seed_stability.py` ported and validated), extended to cover every tier so the
whole artifact is content-addressed, not just the atoms:

```
content_hash = sha256(
    b"gam-tiered-artifact-v1"
  ⧺ T0 canonical bytes  (mean,norm,rogue_dims,scale as f64 LE)
  ⧺ T1 canonical bytes  (unit-normed, orientation-fixed decoder rows, f64 LE)
  ⧺ T2 canonical dictionary hash  (the Rust-mirrored per-atom orbit hash)
  ⧺ encoder.distilled_from_content_hash (str, or "none")
)
```

The T2 sub-hash is exactly `dictionary_artifact.rs`'s: atoms scaled to
‖B_k‖_F = 1, oriented to a finite-gauge convention, sorted by atom hash, folded
with the residual finite-gauge string and gauge certificate. So a T2 hash emitted
here equals the one the Rust `canonical_dictionary_artifact` would emit on the
same fitted atoms (WS-B's gauge quotient is what makes that equality certified
rather than byte-luck). `diff` reports per-tier hash equality and, for T2,
atom-level matching + principal-angle subspace agreement — byte inequality with
subspace agreement ≈ 1 is the "seed-unstable latents, seed-stable subspace"
signal (SAC_PLAN Part-4 stability).

## Verbs

| verb | does | reuses |
|---|---|---|
| `fit` | drive the tiered pipeline (T1 → SAC T2 → assemble) and write the artifact dir | `compose_tiers`, `sac_prototype`, `ShardReader` |
| `encode` | amortized encoder over rows, exact-solve certificate fallback, fallback rate by frequency decile | `DistilledEncoder`, `encode_with_fallback` |
| `steer` | dose-calibrated chart move on one atom; `predicted_nats` before the edit | `ManifoldSAE.steer` (W8 dose machinery) |
| `diff` | compare two artifacts: atom matching, subspace angles, per-tier hash deltas | W9 harness (`latent_match`, `union_subspace`, canonical hash) |

## Producer checklist (what each workstream must emit)

- **WS-C (tier-1):** `dictionary_artifact` dir with `decoder (K1,P)` + `t0` baked
  in. → fills T0, T1.
- **WS-A (SAC):** `examples/sac_results/` (`atom_<k>.json` + `sac_meta.json`) or a
  single assembled `ManifoldSAE.to_dict()`. → fills T2, ev_trace, Σ when exposed.
- **WS-E (encoder):** `encoder/` bundle with `distilled_from_content_hash`. →
  fills encoder tier.
- **WS-D (data):** `ShardWriter` manifest with `stats` + rogue dims/scale. →
  source for T0 when WS-C does not bake it.
