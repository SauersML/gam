"""Streaming Arrow-Schur synthetic K=100K demo."""

from __future__ import annotations

from gamfit._binding import rust_module


def main() -> None:
    result = rust_module().streaming_arrow_schur_synthetic_demo(
        n_obs=10_000,
        n_atoms=100_000,
        latent_dim=2,
        beta_dim=8,
        chunk_size=4096,
    )
    print(
        f"streaming_arrow_schur n_atoms={result['n_atoms']:,} "
        f"delta_beta_norm={result['delta_beta_norm']:.6e}"
    )


if __name__ == "__main__":
    main()
