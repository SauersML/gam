#!/usr/bin/env python3
"""LatentCoord circle retraction payload demo."""

from __future__ import annotations

import gamfit
from gamfit._api import _normalize_latents


def main() -> None:
    n = 32
    latents = {
        "t": gamfit.LatentCoord(
            n=n,
            d=1,
            init="pca",
            manifold="circle",
            retraction="circle",
        )
    }
    payload = _normalize_latents(latents)
    assert payload is not None
    assert payload["t"]["retraction"] == "circle"
    print(payload["t"])


if __name__ == "__main__":
    main()
