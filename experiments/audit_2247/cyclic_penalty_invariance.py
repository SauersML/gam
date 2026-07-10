"""#2247 blocker-5a evidence: the retired cyclic coefficient-difference penalty
is NOT the roughness integral it claimed to model.

For a uniform periodic lattice of K coefficients, both the discarded
`create_cyclic_difference_penalty_matrix` (S = DᵀD, D = 2nd difference on the
ring) and the true second-derivative roughness operator are circulant, so their
eigenvalues are known in closed form on the Fourier mode e^{i j θ}:

    difference penalty   λ_diff(j) = (2 − 2 cos(2π j / K))²           # DᵀD, raw
    continuum roughness  λ_true(j) ∝ ∫ (f'')² dθ = 2π · j⁴            # exact Gram

The plots below are computed with numpy alone (no gam build). They show the two
SPEC-5 defects of penalizing coefficients instead of the function:
  (A) the difference penalty's whole SCALE depends on knot density K, and its
      SHAPE across modes deviates from j⁴ — so λ̂ lands on a K-dependent scale;
  (B) relative to the true roughness of each Fourier mode, the difference
      penalty progressively UNDER-penalizes high-frequency wiggle, collapsing to
      ~0.16 at the Nyquist mode — a 6× error exactly where roughness matters.

The fix routes every periodic 1-D Duchon term to `build_periodic_duchon_basis_1d`,
whose penalty ω = zᵀ K_centers z is the exact function-space Gram (the Bernoulli
Green's function of (d²/dx²)^m) — knot-invariant and reparameterization-invariant.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def diff_penalty_eigs(K):
    """Eigenvalues of DᵀD (2nd cyclic difference) for K ring coefficients."""
    j = np.arange(K)
    return (2.0 - 2.0 * np.cos(2.0 * np.pi * j / K)) ** 2


def true_roughness_eigs(K):
    """Continuum ∫(f'')² per Fourier mode j, up to the shared 2π factor: j⁴."""
    j = np.arange(K)
    return (j.astype(float)) ** 4


fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ---- Panel A: spectra, normalized to the fundamental (j=1) to compare SHAPE ----
Ks = [16, 32, 64]
colors = ["#1b9e77", "#7570b3", "#d95f02"]
for K, c in zip(Ks, colors):
    j = np.arange(1, K // 2 + 1)
    ld = diff_penalty_eigs(K)[1 : K // 2 + 1]
    ld = ld / ld[0]
    axA.loglog(j, ld, "o-", color=c, ms=4, lw=1.5,
               label=f"difference penalty  DᵀD  (K={K})")

jt = np.arange(1, 33)
lt = jt.astype(float) ** 4
lt = lt / lt[0]
axA.loglog(jt, lt, "k--", lw=2.4, label="exact function roughness  ∝ j⁴")
axA.set_xlabel("Fourier mode  j")
axA.set_ylabel("penalty eigenvalue  (normalized to j = 1)")
axA.set_title("(A) Coefficient penalty ≠ roughness integral\n"
              "shape depends on knot count K; only exact Gram is ∝ j⁴")
axA.legend(fontsize=8, loc="upper left")
axA.grid(True, which="both", alpha=0.25)

# ---- Panel B: relative under-penalization vs the true continuum operator ------
for K, c in zip(Ks, colors):
    j = np.arange(1, K // 2 + 1)
    x = 2.0 * np.pi * j / K           # discrete angular frequency
    # h-normalized difference eigenvalue vs true x⁴ (continuum): ratio → 1 low-freq
    ratio = ((2.0 - 2.0 * np.cos(x)) ** 2) / (x ** 4)
    axB.plot(j / (K / 2.0), ratio, "o-", color=c, ms=4, lw=1.5,
             label=f"K = {K}")

axB.axhline(1.0, color="k", ls=":", lw=1.2, label="exact roughness (target = 1)")
nyq = 16.0 / np.pi ** 4
axB.annotate(f"Nyquist mode:\n{nyq:.3f}  (≈6× under-penalized)",
             xy=(1.0, nyq), xytext=(0.55, 0.45), fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="#555"))
axB.set_xlabel("normalized frequency  j / (K/2)   (1 = Nyquist)")
axB.set_ylabel("difference-penalty / true roughness")
axB.set_title("(B) The difference penalty under-penalizes\n"
              "high-frequency wiggle — collapse curves are K-invariant here,\n"
              "but the ABSOLUTE scale in (A) is not")
axB.set_ylim(0, 1.08)
axB.legend(fontsize=8, loc="lower left")
axB.grid(True, alpha=0.25)

fig.suptitle("#2247 · blocker 5a — retired cyclic coefficient-difference penalty vs exact function-space Gram",
             fontsize=12, y=1.01)
fig.tight_layout()
out = "/Users/user/gam/experiments/audit_2247/cyclic_penalty_invariance.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

# ---- numeric summary printed for the issue comment --------------------------
print("\nNyquist-mode penalty / true roughness (should be 1.0 for a function penalty):")
for K in Ks:
    x = np.pi  # j = K/2
    print(f"  K={K:3d}:  {((2-2*np.cos(x))**2)/(x**4):.4f}")
print("\nAbsolute fundamental-mode (j=1) difference eigenvalue — scale drifts with K:")
for K in Ks:
    print(f"  K={K:3d}:  λ_diff(1) = {diff_penalty_eigs(K)[1]:.3e}")
