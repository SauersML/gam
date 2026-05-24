"""Streaming Arrow-Schur synthetic K=100K demo."""

from __future__ import annotations

import resource
import sys
import time

from gamfit._binding import rust_module


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def main() -> None:
    n_obs = 10_000
    n_atoms = 100_000
    latent_dim = 2
    beta_dim = 8
    chunk_size = 4096
    baseline = rss_mb()
    t0 = time.perf_counter()

    result = rust_module().streaming_arrow_schur_synthetic_demo(
        n_obs=n_obs,
        n_atoms=n_atoms,
        latent_dim=latent_dim,
        beta_dim=beta_dim,
        chunk_size=chunk_size,
    )

    peak = rss_mb()
    print(
        f"N={n_obs:,} obs K={n_atoms:,} atoms d={latent_dim} "
        f"beta_dim={beta_dim} chunk_size={chunk_size}"
    )
    print(f"delta_beta_norm={result['delta_beta_norm']:.6e}")
    print(f"delta_t_norm={result['delta_t_norm']:.6e}")
    print(f"elapsed_sec={time.perf_counter() - t0:.3f}")
    print(f"peak_rss_mb={peak:.1f}")
    print(f"peak_rss_delta_mb={peak - baseline:.1f}")
    assert peak < 2048.0


if __name__ == "__main__":
    main()
