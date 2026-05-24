"""End-to-end SAE training with gamfit torch penalties."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from gamfit.torch import ARDPenalty, IsometryPenalty


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.SiLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def synthetic_batch(n: int, input_dim: int, true_dim: int) -> torch.Tensor:
    torch.manual_seed(7)
    factors = torch.randn(n, true_dim, dtype=torch.float64)
    mixing = torch.randn(true_dim, input_dim, dtype=torch.float64)
    x = factors @ mixing + 0.05 * torch.randn(n, input_dim, dtype=torch.float64)
    return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)


def main() -> None:
    torch.manual_seed(11)
    n = 192
    input_dim = 12
    latent_dim = 5
    x = synthetic_batch(n, input_dim, true_dim=3)

    model = SparseAutoencoder(input_dim, latent_dim).to(dtype=torch.float64)
    iso = IsometryPenalty(weight=0.01)
    ard = ARDPenalty(latent_dim=latent_dim)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(ard.parameters()),
        lr=3e-3,
    )
    history: list[tuple[float, float, float]] = []

    for epoch in range(50):
        opt.zero_grad(set_to_none=True)
        recon, z = model(x)
        recon_loss = F.mse_loss(recon, x)
        iso_loss = iso(z, model.decoder.weight)
        ard_loss = 0.002 * ard(z)
        loss = recon_loss + iso_loss + ard_loss
        loss.backward()
        opt.step()
        history.append((float(loss), float(recon_loss), float(ard_loss)))
        print(
            f"epoch={epoch + 1:02d} loss={history[-1][0]:.6f} "
            f"recon={history[-1][1]:.6f} ard={history[-1][2]:.6f}"
        )

    with torch.no_grad():
        _, z = model(x)
        activeness = z.pow(2).mean(0).sqrt()
    print("final_loss", f"{history[-1][0]:.6f}")
    print("final_recon", f"{history[-1][1]:.6f}")
    print("atom_activeness", [round(v, 6) for v in activeness.tolist()])


if __name__ == "__main__":
    main()
