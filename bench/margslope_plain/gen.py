"""Generate plain marginal-slope CSVs at varying n.

User-target schema for the plain (no scale_dims, no linkwiggle, no score-warp,
no survival) marginal-slope Duchon path:

  columns: case (0/1), sex (0/1), prs_z (std-normal), PC1..PC10 (std-normal)

Truth: probit P(Y=1|x,z) = Phi(a(x) + b(x)*z) where
  a(x)     = a0 + 0.6*PC1 + 0.4*sin(2π*PC2*0.3) + 0.5*sex
  log b(x) = b0 + 0.3*PC1 - 0.2*PC3

So the signal lives on a low-dim subspace of the 10D PC sphere, the
remaining PCs are nuisance, and prs_z carries a covariate-modulated effect.
"""
import math
import random
import sys


def probit_probability(eta: float) -> float:
    return 0.5 * (1.0 + math.erf(eta / math.sqrt(2.0)))


def write_csv(n: int, path: str, seed: int = 0xA110CA7E) -> None:
    rng = random.Random(seed ^ n)
    pc_cols = [f"PC{j}" for j in range(1, 11)]
    header = ["case", "sex", "prs_z"] + pc_cols
    a0 = -0.4
    b0 = 0.0
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for _ in range(n):
            sex = rng.randint(0, 1)
            z = rng.gauss(0.0, 1.0)
            pcs = [rng.gauss(0.0, 1.0) for _ in range(10)]
            a = a0 + 0.6 * pcs[0] + 0.4 * math.sin(math.tau * pcs[1] * 0.3) + 0.5 * sex
            log_b = b0 + 0.3 * pcs[0] - 0.2 * pcs[2]
            p_y = probit_probability(a + math.exp(log_b) * z)
            y = 1 if rng.random() < p_y else 0
            row = [str(y), str(sex), f"{z:.10g}"] + [f"{v:.10g}" for v in pcs]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1])
    out = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0xA110CA7E
    write_csv(n, out, seed)
    print(f"wrote n={n} to {out}", file=sys.stderr)
