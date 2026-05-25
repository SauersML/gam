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
from statistics import NormalDist


STANDARD_NORMAL = NormalDist()


def write_csv(n, path, seed=0x5CA1AB1E):
    rng = random.Random(seed ^ n)
    two_pi = 2.0 * math.pi
    pc_cols = [f"pc{j}_std" for j in range(1, 17)]
    header = ["phenotype", "sex", "age_entry_std", "pgs_ctn_z"] + pc_cols
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for _ in range(n):
            sex = rng.randint(0, 1)
            age = rng.gauss(0.0, 1.0)
            z = rng.gauss(0.0, 1.0)
            pcs = [rng.gauss(0.0, 1.0) for _ in range(16)]
            eta = (
                0.5 * math.sin(two_pi * age * 0.3)
                + 0.4 * sex
                + 0.6 * pcs[0]
                + 0.3 * z
            )
            p = STANDARD_NORMAL.cdf(eta)
            y = 1 if rng.random() < p else 0
            row = [str(y), str(sex), f"{age:.10g}", f"{z:.10g}"] + [f"{v:.10g}" for v in pcs]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1])
    out = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0x5CA1AB1E
    write_csv(n, out, seed)
    print(f"wrote n={n} to {out}", file=sys.stderr)
