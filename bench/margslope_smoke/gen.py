"""Generate biobank-shape margslope CSVs at varying n.

Schema mirrors `rust_margslope_aniso_duchon16d_rigid` from biobank_scale.yml:
  phenotype (binary), sex (0/1), age_entry_std (z), pgs_ctn_z (z),
  pc1_std .. pc16_std (z, standard normal).

Truth: low-dim signal — eta = 0.5·sin(2π·age·0.3) + 0.4·sex + 0.6·PC1 + 0.3·z
Probit link → binary phenotype. Other 15 PCs are nuisance (genuine zero
true effect), so the 16D Duchon must discover that the signal lives on PC1.
"""
import math
import random
import sys


def std_normal(rng, two_pi=2.0 * math.pi):
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(two_pi * u2)


def erf_approx(x):
    a1, a2, a3, a4, a5, p = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911,
    )
    sign = -1.0 if x < 0 else 1.0
    ax = abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-ax * ax)
    return sign * y


def write_csv(n, path, seed=0x5CA1AB1E):
    rng = random.Random(seed ^ n)
    two_pi = 2.0 * math.pi
    sqrt_2 = math.sqrt(2.0)
    pc_cols = [f"pc{j}_std" for j in range(1, 17)]
    header = ["phenotype", "sex", "age_entry_std", "pgs_ctn_z"] + pc_cols
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for _ in range(n):
            sex = rng.randint(0, 1)
            age = std_normal(rng)
            z = std_normal(rng)
            pcs = [std_normal(rng) for _ in range(16)]
            eta = (
                0.5 * math.sin(two_pi * age * 0.3)
                + 0.4 * sex
                + 0.6 * pcs[0]
                + 0.3 * z
            )
            p = 0.5 * (1.0 + erf_approx(eta / sqrt_2))
            y = 1 if rng.random() < p else 0
            row = [str(y), str(sex), f"{age:.10g}", f"{z:.10g}"] + [f"{v:.10g}" for v in pcs]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1])
    out = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0x5CA1AB1E
    write_csv(n, out, seed)
    print(f"wrote n={n} to {out}", file=sys.stderr)
