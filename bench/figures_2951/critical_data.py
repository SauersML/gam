"""Per-token criticality on the rows of masks_vpd4l.npz (#2951): the target's next-token entropy and
its loss on the true next token, next to VPD's pieces switched on and VPD's per-token KL."""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.chdir(os.path.expanduser("~/mpd-data/vpd"))  # the runs are read relative to here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vpd_2951"))
from vpd_model import load_target, val_tokens  # noqa: E402

Z = np.load(Path.home() / "mpd-data/frontier/masks_vpd4l.npz")
R, S = Z["ids"].shape
ids = val_tokens(R, seq=S + 1, offset=1024)  # one extra token: the label of the last position
assert np.array_equal(ids[:, :S].numpy(), Z["ids"]), "rows differ from the mask dump"
target = load_target("mps")
ent, loss = np.zeros((R, S)), np.zeros((R, S))
with torch.no_grad():
    for i in range(0, R, 4):
        b = ids[i:i + 4].to("mps")
        lp = F.log_softmax(target(b[:, :S]).float(), -1)
        ent[i:i + 4] = -(lp.exp() * lp).sum(-1).cpu().double().numpy()
        loss[i:i + 4] = -lp.gather(-1, b[:, 1:, None])[..., 0].cpu().double().numpy()
out = Path.home() / "mpd-data/figures/data/critical.npz"
np.savez(out, pieces=np.diff(Z["vpd_indptr"]).reshape(R, S), kl=Z["kl_vpd"], entropy=ent, loss=loss)
print(out, "mean entropy", ent.mean(), "mean loss", loss.mean())
