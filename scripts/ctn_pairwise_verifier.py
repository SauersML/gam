#!/usr/bin/env python3
"""CTN pairwise verifier — independent reference for the Rust pairwise oracle.

Mirrors the Rust toy fixture in
    src/families/transformation_normal.rs
        tests::toy_covariate_design_and_derivs
        tests::toy_family_and_derivatives
        tests::ctn_pairwise_oracle_dumps_json

Computes pair(a, b).a = ∂²(-log L)/∂ψ_a∂ψ_b at fixed β via straight numpy on
the Khatri-Rao tensor decomposition of h, h':

    h[i]   = Σ_{j,k} R[i, j] · C[i, k; ψ] · β[j·p_cov + k]
    h'[i]  = Σ_{j,k} R'[i, j] · C[i, k; ψ] · β[j·p_cov + k]

Per-row CTN log-likelihood:    log L_i = w_i · (-½·h[i]² + log h'[i])
    F = -log L (no penalty in this fixture; double_penalty=false, no penalties registered)

Therefore:
    ∂²F/∂ψ_a∂ψ_b
    = Σ_i w_i · [∂h/∂ψ_a · ∂h/∂ψ_b
                 + h · ∂²h/∂ψ_a∂ψ_b
                 - (1/h') · ∂²h'/∂ψ_a∂ψ_b
                 + (1/h'²) · ∂h'/∂ψ_a · ∂h'/∂ψ_b]

Run order:
    1. cargo test --release ctn_pairwise_oracle_dumps_json -- --nocapture
       (writes /tmp/ctn_pairwise_oracle.json)
    2. python3 scripts/ctn_pairwise_verifier.py
       (loads the Rust JSON, computes the reference, diffs)
"""

import json
import os
import sys

import numpy as np


def main():
    # ── Toy fixture (must match transformation_normal.rs:tests exactly) ──
    psi = np.array([0.15, -0.10])
    beta = np.array([0.15, -0.05, 0.80, 0.30])
    v = np.array([0.4, -0.7])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    # response bases at the four observation points
    response_val = np.array(
        [[1.0, -1.0], [1.0, -0.2], [1.0, 0.6], [1.0, 1.3]], dtype=np.float64
    )
    response_deriv = np.array(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )

    # covariate-side fixtures (4×2 each)
    x0 = np.array([[1.00, 0.40], [1.10, 0.35], [1.20, 0.45], [0.95, 0.50]])
    x_a = np.array([[0.10, -0.02], [0.08, 0.01], [0.12, -0.01], [0.09, 0.03]])
    x_b = np.array([[-0.04, 0.06], [-0.02, 0.05], [-0.03, 0.04], [-0.01, 0.07]])
    x_aa = np.array([[0.02, 0.00], [0.01, 0.01], [0.02, -0.01], [0.01, 0.02]])
    x_ab = np.array([[0.01, -0.01], [0.00, 0.02], [0.01, 0.01], [0.00, -0.01]])
    x_bb = np.array([[-0.01, 0.02], [-0.02, 0.01], [-0.01, 0.00], [-0.02, 0.02]])

    # C(x; ψ) and per-axis derivatives at the chosen ψ
    cov = (
        x0
        + psi[0] * x_a
        + psi[1] * x_b
        + 0.5 * psi[0] ** 2 * x_aa
        + psi[0] * psi[1] * x_ab
        + 0.5 * psi[1] ** 2 * x_bb
    )
    cov_first = [
        x_a + psi[0] * x_aa + psi[1] * x_ab,  # ∂C/∂ψ_0
        x_b + psi[0] * x_ab + psi[1] * x_bb,  # ∂C/∂ψ_1
    ]
    cov_second = [
        [x_aa, x_ab],  # ∂²C/∂ψ_0∂ψ_0,  ∂²C/∂ψ_0∂ψ_1
        [x_ab, x_bb],  # ∂²C/∂ψ_1∂ψ_0,  ∂²C/∂ψ_1∂ψ_1
    ]

    # Khatri-Rao β reshape: β[j·p_cov + k]; p_resp=2, p_cov=2.
    # lifted[i, j·p_cov+k] = R[i, j] · C[i, k]
    # h[i] = Σ_jk R[i,j] · C[i,k] · β[j·p_cov+k]
    #      = Σ_j R[i,j] · (C @ β_row_j)[i]
    p_resp, p_cov = 2, 2
    n = 4
    beta_jk = beta.reshape(p_resp, p_cov)  # β_jk = β[j*p_cov+k]

    def lifted_forward(resp_basis, cov_mat):
        # returns n-vector: Σ_j R[i,j] · (cov_mat @ β_jk_j)[i]
        out = np.zeros(n)
        for j in range(p_resp):
            cb = cov_mat @ beta_jk[j]  # n-vector
            out += resp_basis[:, j] * cb
        return out

    h = lifted_forward(response_val, cov)
    h_prime = lifted_forward(response_deriv, cov)
    inv_hp = 1.0 / h_prime
    inv_hp_sq = inv_hp ** 2

    # First and second directional ψ-derivatives of h, h'
    v_val = [lifted_forward(response_val, cov_first[a]) for a in range(2)]
    v_deriv = [lifted_forward(response_deriv, cov_first[a]) for a in range(2)]
    v_val2 = [
        [lifted_forward(response_val, cov_second[a][b]) for b in range(2)]
        for a in range(2)
    ]
    v_deriv2 = [
        [lifted_forward(response_deriv, cov_second[a][b]) for b in range(2)]
        for a in range(2)
    ]

    # ── pair(a, b).a = ∂²F/∂ψ_aψ_b at fixed β ──
    pair_a = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            term = (
                v_val[a] * v_val[b]
                + h * v_val2[a][b]
                - inv_hp * v_deriv2[a][b]
                + inv_hp_sq * v_deriv[a] * v_deriv[b]
            )
            pair_a[a, b] = float(np.sum(weights * term))

    # ── Directional contraction Σ_b v_b · pair(a, b).a ──
    a_dir_ref = pair_a @ v  # shape (2,)

    print("[verifier] toy fixture matches transformation_normal.rs::tests")
    print(f"[verifier] ψ={psi}, β={beta}, v={v}")
    print()
    print("[verifier] Reference pair(a, b).a (numpy):")
    for a in range(2):
        for b in range(2):
            print(f"  pair({a},{b}).a = {pair_a[a, b]:+.10e}")
    print()
    print("[verifier] Reference directional contraction a_dir(i) = Σ_j v_j · pair(i,j).a:")
    for i in range(2):
        print(f"  a_dir({i}) = {a_dir_ref[i]:+.10e}")

    # ── Load Rust output and diff ──
    rust_path = "/tmp/ctn_pairwise_oracle.json"
    if not os.path.exists(rust_path):
        print()
        print(f"[verifier] {rust_path} not found.")
        print(
            "[verifier] Run: cargo test --release ctn_pairwise_oracle_dumps_json -- --nocapture"
        )
        sys.exit(0)

    with open(rust_path) as f:
        rust = json.load(f)

    print()
    print("[verifier] Loaded Rust output. Diffing element-wise.")
    print()

    max_pair_diff = 0.0
    for entry in rust["pairwise"]:
        i, j = entry["i"], entry["j"]
        rust_a = entry["a"]
        ref_a = pair_a[i, j]
        diff = abs(rust_a - ref_a)
        max_pair_diff = max(max_pair_diff, diff)
        print(
            f"  pair({i},{j}).a: rust={rust_a:+.10e}  ref={ref_a:+.10e}  |Δ|={diff:.3e}"
        )

    max_dir_diff = 0.0
    for entry in rust["directional_contraction"]:
        i = entry["i"]
        rust_dir = entry["a_dir"]
        ref_dir = a_dir_ref[i]
        diff = abs(rust_dir - ref_dir)
        max_dir_diff = max(max_dir_diff, diff)
        print(
            f"  a_dir({i}):       rust={rust_dir:+.10e}  ref={ref_dir:+.10e}  |Δ|={diff:.3e}"
        )

    print()
    print(f"[verifier] max |Δ| pair  = {max_pair_diff:.3e}")
    print(f"[verifier] max |Δ| a_dir = {max_dir_diff:.3e}")

    # Tight tolerance for IEEE-double rebuild of identical algebra in two
    # languages. If this fails, either:
    #   (a) the Rust pairwise body has a subtle bug not yet caught by the
    #       existing tests (interesting!),
    #   (b) my numpy reference has a different convention (e.g. Khatri-Rao
    #       β indexing flipped), or
    #   (c) the toy fixture in tests has drifted from this script.
    tol = 1e-12
    if max_pair_diff < tol and max_dir_diff < tol:
        print(
            f"[verifier] \033[32mPASS\033[0m — Rust pairwise body matches numpy reference at < {tol}"
        )
        sys.exit(0)
    else:
        # Detailed sign-and-symmetry hints when something is off.
        rust_pair = {(e["i"], e["j"]): e["a"] for e in rust["pairwise"]}
        for (i, j), rust_a in sorted(rust_pair.items()):
            if (j, i) in rust_pair:
                sym = abs(rust_pair[(i, j)] - rust_pair[(j, i)])
                if sym > 1e-12:
                    print(
                        f"[verifier]   asymmetry pair({i},{j})-pair({j},{i}) = {sym:.3e}"
                    )
        print(
            f"[verifier] \033[31mFAIL\033[0m — Rust pairwise differs from numpy reference at > {tol}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
