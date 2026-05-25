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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gamfit._binding import rust_module
from gamfit._exceptions import map_exception


def inverse_link(values: list[float], link: str) -> list[float]:
    try:
        return [float(v) for v in rust_module().apply_inverse_link_array(values, link)]
    except Exception as exc:
        raise map_exception(exc) from exc


def write_csv(n: int, path: str, seed: int = 0xA110CA7E) -> None:
    rng = random.Random(seed ^ n)
    pc_cols = [f"PC{j}" for j in range(1, 11)]
    header = ["case", "sex", "prs_z"] + pc_cols
    a0 = -0.4
    b0 = 0.0
    pending_rows: list[tuple[int, float, list[float]]] = []
    offsets: list[float] = []
    log_slopes: list[float] = []
    for _ in range(n):
        sex = rng.randint(0, 1)
        z = rng.gauss(0.0, 1.0)
        pcs = [rng.gauss(0.0, 1.0) for _ in range(10)]
        offsets.append(a0 + 0.6 * pcs[0] + 0.4 * math.sin(math.tau * pcs[1] * 0.3) + 0.5 * sex)
        log_slopes.append(b0 + 0.3 * pcs[0] - 0.2 * pcs[2])
        pending_rows.append((sex, z, pcs))
    slopes = inverse_link(log_slopes, "log")
    probabilities = inverse_link(
        [offset + slope * z for offset, slope, (_sex, z, _pcs) in zip(offsets, slopes, pending_rows)],
        "probit",
    )
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for (sex, z, pcs), p_y in zip(pending_rows, probabilities):
            y = 1 if rng.random() < p_y else 0
            row = [str(y), str(sex), f"{z:.10g}"] + [f"{v:.10g}" for v in pcs]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1])
    out = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0xA110CA7E
    write_csv(n, out, seed)
    print(f"wrote n={n} to {out}", file=sys.stderr)
