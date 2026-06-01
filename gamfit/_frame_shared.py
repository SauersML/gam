"""Backend-agnostic primitives shared by the per-frame adapters.

The NumPy, torch, and JAX frame adapters all need the identical
"validate a list of 1-D coordinate vectors of equal length and stack them
into a ``(B, d)`` array" loop. The only per-backend variation is *how* a
single coordinate is coerced/validated and *how* the validated list is
stacked. This module factors the loop — including the error-message text —
into one place; each adapter supplies two small callables.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


def stack_coords_generic(
    coords: Sequence[Any],
    *,
    coerce: Callable[[int, Any, int | None], tuple[Any, int]],
    stack: Callable[[list[Any]], Any],
) -> Any:
    """Validate 1-D equal-length coordinates and stack them to ``(B, d)``.

    Parameters
    ----------
    coords
        Sequence of backend-native coordinate vectors (one per intrinsic
        axis).
    coerce
        ``coerce(idx, c, ref_len) -> (value, length)``. Receives the
        coordinate index, the raw coordinate, and the reference length seen
        so far (``None`` for the first coordinate). It must coerce ``c`` to
        the backend's native float vector, raise ``ValueError`` when the
        coordinate is not 1-D, and return the validated value together with
        its length. The shared loop performs the cross-coordinate
        length-equality check, so ``coerce`` only needs to report the
        coordinate's own length.
    stack
        ``stack(values) -> (B, d)`` array, e.g. ``np.stack(.., axis=1)``.
    """
    if len(coords) == 0:
        raise ValueError("stack_coords requires at least one coordinate")
    values: list[Any] = []
    ref_len: int | None = None
    for idx, c in enumerate(coords):
        value, length = coerce(idx, c, ref_len)
        if ref_len is None:
            ref_len = length
        elif length != ref_len:
            raise ValueError(
                f"coord {idx}: length {length} does not match reference "
                f"length {ref_len}"
            )
        values.append(value)
    return stack(values)
