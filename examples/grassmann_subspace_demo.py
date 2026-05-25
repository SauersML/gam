"""Recover a principal subspace as a point on Gr(k, n)."""

import gamfit


def main() -> None:
    print(gamfit.GrassmannManifold(k=2, n=6).to_json())


if __name__ == "__main__":
    main()
