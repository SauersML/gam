# Low-rank ambient frames as the primary β parameterization for the curved stage

Status: design note fixing the architecture for the curved manifold-SAE stage at
LLM width. Companion module: `crates/gam-sae/src/manifold/inframe_curved.rs`.

## The problem this fixes

The curved manifold-SAE term parameterizes each atom's decoder as a
coefficient block `B_k` of shape `M_k × p`, where `M_k` is the atom's basis
size and `p` is the ambient (residual-stream) output dimension. The arrow-Schur
border — the dense coupling block the inner Newton solve and the Laplace
evidence log-det both run on — carries `Σ_k M_k · p` coefficients, and each
atom's posterior covariance is `(M_k · p)²`.

At the shapes the reviewer flagged this does not fit:

| quantity        | K=32k, M=8, p=4096                         |
|-----------------|--------------------------------------------|
| shared border   | `Σ M_k·p ≈ 32768·8·4096 ≈ 1.07 × 10⁹` coeffs |
| per-atom cov    | `(M_k·p)² = (8·4096)² ≈ 1.07 × 10⁹` f64 ≈ **8.6 GB / atom** |

The per-atom covariance alone is impossible at LLM width, and the border is a
billion-coefficient dense object. The `p` factor is the whole problem: the
curved engine never needs the full ambient width to describe an atom whose
decoded image lives in a handful of directions.

## The pieces that already exist (issue #972)

The **single-atom decoder frame** already factors the *linear* border. Each
atom may carry a `GrassmannFrame` `U_k` (`p × r_k`, column-orthonormal, a point
on the Stiefel manifold `St(r_k, p)` / its span a point on `Gr(r_k, p)`); the
decoder factors `B_k = C_k U_kᵀ` with the coordinate matrix `C_k` (`M_k × r_k`)
living IN the border and `U_k` profiled OUT by closed-form streaming polar
steps (`crate::frames::GrassmannFrame::polar_update`). Border becomes
`Σ_k M_k · r_k`; the frame's `r_k(p − r_k)` intrinsic degrees of freedom enter
the Laplace evidence dimension honestly (`frame_manifold_dimension`).

`maybe_activate_decoder_frame` auto-derives `r_k` from the decoder's numerical
rank and activates the frame when it shrinks the border by at least
`SAE_FRAME_ACTIVATION_MARGIN`. This is the primary path for the *linear* lane.

What was missing is the same discipline for the **curved** stage: confining the
curved REML fit itself to a learned low-rank ambient frame so the curved engine
only ever sees in-frame coordinates, and doing so as the DEFAULT flagship path
rather than an opportunistic per-atom auto-activation.

## The architecture: a three-stage cascade

```
            full corpus X  (N × p)
                  │
   ┌──────────────▼──────────────┐   Stage 1 — LINEAR / BLOCK LANE
   │ full-width linear + block    │   owns the full-p projection.
   │ sparse dictionary on ALL     │   Border here is already factored per
   │ tokens                       │   #972; residual R = X − X̂_linear.
   └──────────────┬──────────────┘
                  │  residual R  (N × p)
   ┌──────────────▼──────────────┐   Stage 2 — FRAMES OWN FULL-p
   │ per REGION learn a           │   U_g ∈ St(r, p), r ≈ 8–32, from the
   │ Grassmann frame U_g (p × r)  │   region's residual targets by the
   │ from the residual span       │   closed-form polar / thin-SVD seed.
   └──────────────┬──────────────┘   Charged ONCE, globally, against N.
                  │  in-frame coords Z_g = R_g U_g  (n_g × r)
   ┌──────────────▼──────────────┐   Stage 3 — CURVED REML, IN-FRAME
   │ curved chart fit PURELY in   │   Border = Σ M_k·r, cov = (M·r)².
   │ the r-dim frame; cross-fit   │   Each block pays ½·d_eff·log n_g.
   │ evidence gate, EBH selection │   Projected to ambient ON DEMAND.
   └─────────────────────────────┘
```

Three properties make this the default:

1. **The curved engine never sees `p`.** Everything downstream of Stage 2 — the
   chart fit, the border, the posterior covariance, the shape bands — is
   computed in the `r`-dimensional in-frame coordinate space. The only `p`-sized
   objects are the frame `U_g` itself (`p × r`, formed once outside the border by
   an `O(p r²)` step) and the lift `Ẑ U_gᵀ` used to project a prediction or a
   band back to ambient on demand.

2. **The frame cost is amortized globally, not per block.** A learned frame
   carries `r(p − r)` intrinsic Grassmann degrees of freedom. Charging that
   against every per-region curved gate would reject everything at `p = 4096`
   (`8·4088 ≈ 32k` DOF ≫ any single region's deviance gain). Instead the frame
   is a Stage-2 structural object shared across the whole corpus and across every
   atom that routes into its region: its `r(p − r) · ½ log N` charge is paid
   ONCE against the full token count `N`, and the per-block Stage-3 gate pays only
   the incremental curved DOF `½·d_eff·log n_g`. This is the amortization the
   cascade exists to expose.

3. **Parity with the joint full-p fit on the frame's span.** When the frame
   contains the truth (the planted curved structure lives in `range(U_g)`), the
   in-frame fit and a full-p fit restricted to that span agree to machine
   precision — the frame is an exact orthonormal change of coordinates, not an
   approximation. The dense joint full-p curved fit is therefore reserved for the
   *certified subset* — regions whose deviance charge justifies re-opening the
   full ambient width — while the in-frame cascade is the flagship default for
   everything else.

## Evidence law (SPEC.md: REML-only decisions, no magic constants)

Every accept/reject is an evidence comparison in the SAME single currency the
joint REML PROMOTE gate and the hybrid-split DEMOTE gate already use
(`crate::manifold::rank_charge_dof`). Concretely, per region `g`:

- **deviance gain** `Δ_g`: cross-fit (held-out) reduction in reconstruction SSE
  of the in-frame curved chart over the in-frame linear (rank-1 PCA) predictor.
  Held-out so an over-fit chart cannot buy its own charge.
- **charge** `C_g = ½ · d_eff · log n_eff`, with `d_eff` the realised curved DOF
  in the frame and `n_eff` the autocorrelation-corrected effective sample count.
- **frame charge (global, once)** `½ · r(p − r) · log N` — the Stage-2 structural
  cost, reported separately and amortized across all regions sharing the frame.
- accept when the held-out margin `Δ_g − C_g` is positive with a lower CI bound
  above zero; select the accepted set by e-BH at level `α` so the compose pass
  controls the false-discovery rate over regions.

The frame rank `r` is not a magic constant: it is the numerical rank of the
region's residual span at the relative spectral cutoff `SAE_FRAME_RANK_CUTOFF`
(the same cutoff `decoder_numerical_rank` uses), clamped to a configured band
`[r_min, r_max]`. A region whose residual is genuinely `r`-dimensional pays for
exactly `r` directions; a full-rank region declines the frame and stays on the
certified full-p path.

## Frame timescale

The frame is held FIXED within a single Stage-3 curved fit (identifiability: a
moving frame and moving in-frame coordinates are jointly unidentifiable up to an
`O(r)` rotation, and a fixed frame makes the in-frame border an exact orthonormal
restriction of the ambient border). Between passes the frame MAY be
Grassmann-refreshed by one closed-form polar step from the accumulated
decoder-target cross-moment (`GrassmannCrossMoment` → `polar_update`), and the
refresh is KEPT only if the held-out in-frame EV improves — a REML-only decision,
never an unconditional update. This is the slow-timescale option; the default
single-pass flagship path holds the frame at its Stage-2 seed.

## What is measured

`inframe_curved.rs` returns a `CascadeMemoryLedger` with the exact border and
covariance byte counts for both parameterizations, so the ordersof-magnitude
claim is asserted from measured numbers, not by fiat:

- `dense_border_coeffs = Σ M_k·p`  vs  `inframe_border_coeffs = Σ M_k·r`
- `dense_cov_bytes = Σ (M_k·p)²·8`  vs  `inframe_cov_bytes = Σ (M_k·r)²·8`

At K=32k, M=8, p=4096, r=16 the border shrinks 256× and the per-atom covariance
shrinks 65536× (8.6 GB → 131 KB), which is what moves the curved stage from
"impossible at LLM width" to "tractable banded fit".
