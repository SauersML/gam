"""A real-valued toy of the #2951 toy gate (train_toys.py's export) as budget descent's target
(bench/vpd_2951/budget_descent.py with DESCENT_TOY=DIR): the toy's maps as vpd_model Sites under
vpd_model's site names (h.{l}.attn.{q,k,v,o}_proj, h.{l}.mlp.{c_fc,down_proj}), its activation, and
its Gaussian head, whose outputs (in units of the task residual) stand in for the logits.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/toy_target.py TOY_DIR
writes TOY_DIR/tokens_descent.u16, budget descent's token file for the toy: rows of 513 tokens,
rows 0..31 the training inputs (tokens HELD_OUT on, tiled over rows 32..1023) and rows 1024..1031
the held-out inputs (tokens 0..4095), as descent reads them (DESCENT_TRAIN_ROWS=32)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vpd_2951"))
import vpd_model  # noqa: E402

KINDS = [("attn", "q_proj"), ("attn", "k_proj"), ("attn", "v_proj"), ("attn", "o_proj"), ("mlp", "c_fc"), ("mlp", "down_proj")]


def record(dir_: Path) -> dict:
    return json.loads((Path(dir_) / "export.json").read_text())


def blob(dir_: Path, name: str, rec: dict) -> np.ndarray:
    return np.fromfile(Path(dir_) / f"{name}.f64", dtype="<f8").reshape(rec["files"][name]["shape"])


def site_names(dir_):
    layers = record(dir_)["config"]["n_layers"]
    return lambda n_layer=None: [f"h.{l}.{a}.{k}" for l in range(layers) for a, k in KINDS]


def activation(dir_):
    law = record(dir_)["config"]["mlp_act"]
    return {"relu": torch.relu, "identity": lambda x: x}[law]


class ToyTarget(torch.nn.Module):
    def __init__(self, dir_, device):
        super().__init__()
        rec = record(dir_)
        c = rec["config"]
        if "real_valued" not in rec:
            raise SystemExit(f"{dir_}: not a real-valued toy (a Gaussian head)")
        t = lambda name: torch.tensor(blob(dir_, name, rec), dtype=torch.float32, device=device)  # noqa: E731
        self.n_layer, self.n_head, self.hd = c["n_layers"], c["n_heads"], c["head_dim"]
        self.wte = t("wte")
        self.sites = torch.nn.ModuleDict()
        for l in range(self.n_layer):
            for a, k in KINDS:
                W = t(f"blocks.{l}.{a}.{k}")
                if a == "attn" and torch.any(W != 0):
                    raise SystemExit(f"{dir_}: a toy's attention must be zero (no attention here)")
                self.sites[f"h-{l}-{a}-{k}"] = vpd_model.Site(W)
        self.head = t("gaussian_head")
        self.bias = t("gaussian_head.bias")[0] if "gaussian_head.bias" in rec["files"] else None
        self.relu = c["head"]["relu"]
        self.act = activation(dir_)

    def site(self, name: str) -> vpd_model.Site:
        return self.sites[name.replace(".", "-")]

    def hidden(self, ids):
        x = self.wte[ids]
        for l in range(self.n_layer):
            s = lambda k: self.site(f"h.{l}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")  # noqa: E731
            # The zero attention: every map is called (its input cached as the others'), adding nothing.
            s("q_proj")(x), s("k_proj")(x)
            x = x + s("o_proj")(s("v_proj")(x))
            x = x + s("down_proj")(self.act(s("c_fc")(x)))
        return x

    def forward(self, ids):
        y = self.hidden(ids) @ self.head.T
        if self.bias is not None:
            y = y + self.bias
        return torch.relu(y) if self.relu else y


def load(dir_, device):
    return ToyTarget(dir_, device)


def tokens(dir_: Path):
    held, rows = 4096, record(dir_)["files"]["wte"]["shape"][0]
    train = np.arange(held, rows)
    if len(train) < 32 * 512:
        raise SystemExit(f"{dir_}: {len(train)} training inputs, fewer than 32 rows of 512")
    out = np.zeros((1032, 513), dtype=np.uint16)
    out[:1024, :512] = np.resize(train[: 32 * 512].reshape(32, 512), (1024, 512))
    out[1024:, :512] = np.arange(held).reshape(8, 512)
    out[:, 512] = out[:, 0]
    out.tofile(Path(dir_) / "tokens_descent.u16")
    print(Path(dir_) / "tokens_descent.u16")


if __name__ == "__main__":
    tokens(Path(sys.argv[1]))
