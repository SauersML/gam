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


def write_csv(n, path, seed=0xA110CA7E):
    rng = random.Random(seed ^ n)
    two_pi = 2.0 * math.pi
    sqrt_2 = math.sqrt(2.0)
    pc_cols = [f"PC{j}" for j in range(1, 11)]
    header = ["case", "sex", "prs_z"] + pc_cols
    a0 = -0.4
    b0 = 0.0
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for _ in range(n):
            sex = rng.randint(0, 1)
            z = std_normal(rng)
            pcs = [std_normal(rng) for _ in range(10)]
            a = a0 + 0.6 * pcs[0] + 0.4 * math.sin(two_pi * pcs[1] * 0.3) + 0.5 * sex
            log_b = b0 + 0.3 * pcs[0] - 0.2 * pcs[2]
            b = math.exp(log_b)
            eta = a + b * z
            p_y = 0.5 * (1.0 + erf_approx(eta / sqrt_2))
            y = 1 if rng.random() < p_y else 0
            row = [str(y), str(sex), f"{z:.10g}"] + [f"{v:.10g}" for v in pcs]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1])
    out = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0xA110CA7E
    write_csv(n, out, seed)
    print(f"wrote n={n} to {out}", file=sys.stderr)
