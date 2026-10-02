"""Target reproduction: next-token CE of t-9d2b8f02 on held-out Pile val rows (reported val_loss 2.7075)."""
import sys, time, numpy as np, torch, torch.nn.functional as F
from vpd_model import load_target, DATA
n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
m = load_target("mps")
rows = torch.from_numpy(np.load(DATA)[:n].astype(np.int64))
ces = []
t0 = time.time()
with torch.no_grad():
    for i in range(0, n, 16):
        b = rows[i:i + 16].to("mps")
        lg = m(b[:, :512])
        ces.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1), reduction="none").view(b.shape[0], -1).mean(1).cpu())
ce = torch.cat(ces)
print(f"target CE over {n} rows x 512 tokens: {ce.mean():.4f} +- {ce.std() / len(ce) ** 0.5:.4f} (reported val_loss 2.7075) [{time.time() - t0:.1f}s]")
