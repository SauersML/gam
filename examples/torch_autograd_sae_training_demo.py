"""Train one tiny SAE step with a gamfit torch isometry penalty."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gamfit.torch import IsometryPenalty


def main() -> None:
    torch.manual_seed(11)
    n, input_dim, latent_dim = 64, 8, 3
    factors = torch.randn(n, 2, dtype=torch.float64)
    x = factors @ torch.randn(2, input_dim, dtype=torch.float64)
    x = x + 0.03 * torch.randn_like(x)
    x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)

    encoder = nn.Linear(input_dim, latent_dim, dtype=torch.float64)
    decoder = nn.Linear(latent_dim, input_dim, bias=False, dtype=torch.float64)
    penalty = IsometryPenalty(weight=0.01)
    opt = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)

    opt.zero_grad(set_to_none=True)
    z = encoder(x)
    loss = F.mse_loss(decoder(z), x) + penalty(z, decoder.weight)
    loss.backward()
    opt.step()

    print(f"loss={float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
