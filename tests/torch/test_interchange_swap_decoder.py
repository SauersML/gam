"""Tests for ``gamfit.InterchangeSwapDecoder`` (DAS interchange decoder)."""
from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
torch: Any = pytest.importorskip("torch")


def _make_decoder(D: int = 6, F: int = 4, *, seed: int = 0):
    from gamfit import InterchangeSwapDecoder

    torch.manual_seed(seed)
    return InterchangeSwapDecoder(D, F)


def test_forward_shape_and_value() -> None:
    dec = _make_decoder(D=5, F=3)
    z = torch.randn(7, 3)
    x_hat = dec(z)
    assert x_hat.shape == (7, 5)
    # Manual recomputation: (gate * z) @ W_dec^T + b
    expected = (z * dec.gate) @ dec.W_dec.t() + dec.bias
    torch.testing.assert_close(x_hat, expected)


def test_swap_decode_all_true_mask_matches_forward_za() -> None:
    dec = _make_decoder()
    z_a = torch.randn(4, dec.F)
    z_b = torch.randn(4, dec.F)
    mask = torch.ones(dec.F, dtype=torch.bool)
    out = dec.swap_decode(z_a, z_b, atom_mask=mask)
    torch.testing.assert_close(out, dec(z_a))


def test_swap_decode_all_false_mask_matches_forward_zb() -> None:
    dec = _make_decoder()
    z_a = torch.randn(4, dec.F)
    z_b = torch.randn(4, dec.F)
    mask = torch.zeros(dec.F, dtype=torch.bool)
    out = dec.swap_decode(z_a, z_b, atom_mask=mask)
    torch.testing.assert_close(out, dec(z_b))


def test_swap_decode_half_mask_composes_correctly() -> None:
    dec = _make_decoder(D=4, F=4)
    z_a = torch.randn(3, 4)
    z_b = torch.randn(3, 4)
    mask = torch.tensor([True, False, True, False])
    out = dec.swap_decode(z_a, z_b, atom_mask=mask)
    z_expected = torch.stack(
        [z_a[:, 0], z_b[:, 1], z_a[:, 2], z_b[:, 3]], dim=1
    )
    expected = dec(z_expected)
    torch.testing.assert_close(out, expected)


def test_swap_decode_is_differentiable() -> None:
    dec = _make_decoder()
    z_a = torch.randn(5, dec.F, requires_grad=True)
    z_b = torch.randn(5, dec.F, requires_grad=True)
    mask = torch.tensor([True, False, True, False])
    out = dec.swap_decode(z_a, z_b, atom_mask=mask)
    loss = (out ** 2).sum()
    loss.backward()
    # Gate is per-feature scalar; should receive grad from both halves of the
    # interchange (atoms 0,2 from z_a contribute; atoms 1,3 from z_b).
    assert dec.gate.grad is not None
    assert torch.all(torch.isfinite(dec.gate.grad))
    assert dec.W_dec.grad is not None
    assert torch.all(torch.isfinite(dec.W_dec.grad))
    # The selected halves of z_a / z_b should have non-trivial grad.
    assert z_a.grad is not None
    assert z_b.grad is not None
    # z_a gradient must be zero on the unselected atoms (False mask positions).
    torch.testing.assert_close(
        z_a.grad[:, ~mask], torch.zeros_like(z_a.grad[:, ~mask])
    )
    torch.testing.assert_close(
        z_b.grad[:, mask], torch.zeros_like(z_b.grad[:, mask])
    )


def test_swap_loss_as_training_objective() -> None:
    """A swap-R^2-style loss should be optimizable end-to-end."""
    dec = _make_decoder(D=3, F=2)
    z_a = torch.randn(8, 2)
    z_b = torch.randn(8, 2)
    mask = torch.tensor([True, False])
    target = torch.randn(8, 3)

    opt = torch.optim.SGD(dec.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        swapped = dec.swap_decode(z_a, z_b, atom_mask=mask)
        loss = ((swapped - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_unknown_swap_mode_raises() -> None:
    from gamfit import InterchangeSwapDecoder

    with pytest.raises(ValueError, match="unknown swap_mode"):
        InterchangeSwapDecoder(D=4, F=3, swap_mode="not_a_mode")


def test_mismatched_z_shapes_raise() -> None:
    dec = _make_decoder()
    z_a = torch.randn(4, dec.F)
    z_b = torch.randn(5, dec.F)
    mask = torch.zeros(dec.F, dtype=torch.bool)
    with pytest.raises(ValueError, match="same shape"):
        dec.swap_decode(z_a, z_b, atom_mask=mask)


def test_mask_must_be_bool_length_F() -> None:
    dec = _make_decoder(D=4, F=3)
    z = torch.randn(2, 3)
    with pytest.raises(TypeError, match="bool"):
        dec.swap_decode(z, z, atom_mask=torch.tensor([1, 0, 1]))
    with pytest.raises(ValueError, match="length F"):
        dec.swap_decode(z, z, atom_mask=torch.tensor([True, False]))


def test_gate_and_W_dec_are_separate_parameters() -> None:
    """Decoupling is the whole point: gate must not be a slice of W_dec."""
    dec = _make_decoder()
    param_ids = {id(p) for p in dec.parameters()}
    assert id(dec.gate) in param_ids
    assert id(dec.W_dec) in param_ids
    assert dec.gate.shape == (dec.F,)
    assert dec.W_dec.shape == (dec.D, dec.F)


def test_module_resolves_at_top_level() -> None:
    import gamfit

    assert gamfit.InterchangeSwapDecoder is not None
    dec = gamfit.InterchangeSwapDecoder(D=4, F=2)
    assert dec(torch.randn(1, 2)).shape == (1, 4)
