"""Qwen3-0.6B as a target for VPD's training code (nano_param_decomp/run.py's `decompose`, and the
same semantics as the JAX trainer's qwen3_0_6b.yaml, run p-3653caa3).

Target: the Hugging Face Qwen3ForCausalLM, run on packed rows the way VPD's trainer runs them
(param_decomp/sequence.py SequenceLayout): attention never crosses a document, and positions restart
at each document and at the row's start. Documents are framed as text then <|endoftext|> (151643),
no BOS, and the text's spellings of special tokens stay text, so a row's documents are read off its
tokens: a new document starts after each 151643. `forward(input_ids)` returns logits [B, S, V], as
`decompose` requires.

Sites: per layer the q, k, v and o projections and the MLP's gate, up and down maps, at the
`nn.Linear` paths `decompose`'s `C_per_module` names (`site_paths`). A masked q or k output feeds
q_norm/k_norm, then RoPE, as in VPD's Qwen3 target.

Data: the FineWeb sample/350BT shards tokenized for Qwen3 (~/mpd-data/qwen3_fineweb/release:
training files 0-7, 200,000 documents each and 1.089B tokens in all; held-out file 284; VPD's split
trains on files [0, 284) and evaluates on file 284), unpacked at `ROOT`. A shard's
documents.u32 is its documents concatenated in order, each ending in 151643; packing is VPD's
pack_documents (param_decomp/experiments/lm/data.py): rows of `seq_len` tokens cut from the
concatenation, documents spanning rows, the incomplete last row dropped. The held-out evaluation set
is the first `HELDOUT_ROWS` packed rows of file 284.

Run `python qwen3_target.py check` for the exactness checks (float64 on the CPU against the HF
forward), `smoke` for a few steps of `decompose` on a two-layer prefix, and `cost` for VPD's
parameter, memory and FLOP counts at 0.6B.
"""

import hashlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

MODEL = "Qwen/Qwen3-0.6B"
EOT = 151643
# The release tarballs' layout (zstd -d --long=31 | tar -x): ROOT/train_shard00..07, ROOT/heldout_f284.
ROOT = Path(os.environ.get("QWEN3_FINEWEB", Path.home() / "mpd-data/qwen3_fineweb/extracted/qwen3_fineweb"))
TRAIN_DIRS = [ROOT / f"train_shard{i:02d}" for i in range(8)]
HELDOUT_DIR = ROOT / "heldout_f284"
SEQ_LEN = 1024  # VPD's docs1024
HELDOUT_ROWS = 512

# VPD's subcomponents per site for Qwen3-0.6B (qwen3_0_6b.yaml, `decomposition.sites.cs`).
VPD_C = {"q": 2048, "k": 1024, "v": 1024, "o": 2048, "gate": 3072, "up": 3072, "down": 3072}
KINDS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def load(dtype: torch.dtype = torch.float32, attn: str = "sdpa") -> nn.Module:
    """The HF model from the local snapshot, in `dtype`, in eval mode."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn, local_files_only=True)
    return model.eval()


def layout(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """VPD's SequenceLayout of packed rows: per token its document within the row (a new one after
    each 151643) and its position within that document."""
    ends = (input_ids == EOT).long()
    documents = torch.cumsum(ends, dim=1) - ends
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    starts = torch.ones_like(documents, dtype=torch.bool)
    starts[:, 1:] = documents[:, 1:] != documents[:, :-1]
    offsets = torch.cummax(torch.where(starts, positions, torch.zeros_like(positions)), dim=1).values
    return documents, positions - offsets


class Qwen3Target(nn.Module):
    """`hf` on packed rows: causal attention within each document, positions per document."""

    def __init__(self, hf: nn.Module) -> None:
        super().__init__()
        self.hf = hf

    def forward(self, input_ids: Tensor) -> Tensor:
        documents, positions = layout(input_ids)
        S = input_ids.shape[1]
        causal = torch.ones(S, S, dtype=torch.bool, device=input_ids.device).tril()
        allowed = causal[None] & (documents[:, :, None] == documents[:, None, :])
        dtype = self.hf.get_input_embeddings().weight.dtype
        mask = torch.zeros(allowed.shape, dtype=dtype, device=input_ids.device).masked_fill(~allowed, torch.finfo(dtype).min)
        return self.hf(input_ids=input_ids, attention_mask=mask[:, None], position_ids=positions, use_cache=False).logits


def site_paths(n_layers: int = 28, prefix: str = "hf.model.layers") -> dict[str, str]:
    """Per site `{layer}.{kind}` its `nn.Linear` path in a `Qwen3Target`."""
    return {f"{l}.{k}": f"{prefix}.{l}.{path}" for l in range(n_layers) for k, path in KINDS.items()}


def c_per_module(n_layers: int = 28, cs: dict[str, int] = VPD_C) -> dict[str, int]:
    """`decompose`'s `C_per_module` for a `Qwen3Target`."""
    return {path: cs[site.split(".")[1]] for site, path in site_paths(n_layers).items()}


def packed_rows(directory: Path, seq_len: int = SEQ_LEN) -> np.ndarray:
    """A shard's documents packed into rows of `seq_len` (VPD's pack_documents), a read-only view."""
    tokens = np.memmap(directory / "documents.u32", dtype=np.uint32, mode="r")
    rows = len(tokens) // seq_len
    return tokens[: rows * seq_len].reshape(rows, seq_len)


def train_loader(batch: int, seed: int = 0, seq_len: int = SEQ_LEN, dirs: list[Path] = TRAIN_DIRS) -> Iterator[Tensor]:
    """The training shards' packed rows in a fresh seeded order per pass, `batch` at a time, as
    int64 [batch, seq_len] (`decompose`'s loader)."""
    shards = [packed_rows(d, seq_len) for d in dirs]
    index = [(s, r) for s, rows in enumerate(shards) for r in range(len(rows))]
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(index))
        for start in range(0, len(order) - batch + 1, batch):
            picked = [index[i] for i in order[start : start + batch]]
            yield torch.from_numpy(np.stack([shards[s][r] for s, r in picked]).astype(np.int64))


def heldout(rows: int = HELDOUT_ROWS, seq_len: int = SEQ_LEN) -> Tensor:
    """The held-out evaluation set: the first `rows` packed rows of file 284, int64."""
    return torch.from_numpy(np.array(packed_rows(HELDOUT_DIR, seq_len)[:rows], dtype=np.int64))


def heldout_loader(batch: int, rows: int = HELDOUT_ROWS, seq_len: int = SEQ_LEN) -> Iterator[Tensor]:
    """The held-out set cycled `batch` rows at a time (`decompose`'s eval loader)."""
    held = heldout(rows, seq_len)
    while True:
        for start in range(0, rows - batch + 1, batch):
            yield held[start : start + batch]


def check() -> None:
    """Exactness against the HF forward, float64 on the CPU: (1) on one document's tokens the
    target is the plain HF forward; (2) on a packed held-out row each document's logits are the HF
    forward of that document alone; (3) with VPD's components installed in target mode, the
    logits are unchanged. Prints the largest differences and the held-out set's digest."""
    torch.manual_seed(0)
    hf = load(torch.float64, attn="eager")
    target = Qwen3Target(hf)
    held = heldout()
    print(f"held-out set: {tuple(held.shape)} tokens of file 284, sha256 {hashlib.sha256(held.numpy().tobytes()).hexdigest()}")
    # The held-out row with the most documents.
    row = held[int((held == EOT).sum(dim=1).argmax())][None]
    documents, positions = layout(row)
    bounds = torch.nonzero(torch.diff(documents[0], prepend=documents[0, :1] - 1)).flatten().tolist() + [row.shape[1]]
    print(f"row: {len(bounds) - 1} documents, positions restart at {bounds[:-1]}")
    with torch.no_grad():
        logits = target(row)
        worst = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            alone = hf(input_ids=row[:, a:b], use_cache=False).logits
            worst = max(worst, (logits[:, a:b] - alone).abs().max().item())
        print(f"(2) packed row, each document against HF on it alone: max |Δ logit| {worst:.3e} (logit scale {logits.abs().max().item():.1f})")
        a, b = max(zip(bounds[:-1], bounds[1:]), key=lambda ab: ab[1] - ab[0])
        single = row[:, a:b]
        plain = hf(input_ids=single, use_cache=False).logits
        print(f"(1) one document ({b - a} tokens), target against plain HF: max |Δ logit| {(target(single) - plain).abs().max().item():.3e}")
        sys.path.insert(0, str(Path.home() / "mpd-data/param-decomp/nano_param_decomp"))
        from run import install_components  # VPD's ComponentLinear, target mode

        # Target mode runs W_target alone, whatever C; one subcomponent per site keeps V, U small.
        install_components(target, c_per_module(hf.config.num_hidden_layers, {k: 1 for k in VPD_C}))
        print(f"(3) VPD components installed, target mode: max |Δ logit| {(target(row) - logits).abs().max().item():.3e}")


def cost(batch_rows: int = 256, seq_len: int = SEQ_LEN, n_layers: int = 28) -> None:
    """VPD's counts at Qwen3-0.6B (decompose's training step, per token), from the model's shapes:
    subcomponents and U/V parameters per site under VPD's own 0.6B C and under vpd4l's ratios of C
    to the site's rank bound; the gate (CI) network as VPD sized it for vpd4l (the global shared
    transformer: d_model 2048, 8 blocks, MLP 8192, reading every site's input concatenated) and as
    VPD sized it for 0.6B (chunkwise transformer on the block residuals: d_model 1024, 4 blocks,
    MLP 4096); the persistent PGD sources; and the step's matmul FLOPs."""
    d, q, kv, ff, vocab = 1024, 2048, 1024, 3072, 151936
    shapes = {"q": (d, q), "k": (d, kv), "v": (d, kv), "o": (q, d), "gate": (d, ff), "up": (d, ff), "down": (ff, d)}
    # vpd4l (d 768, ff 3072, 6 heads x 128): C / min(d_in, d_out) per kind; gate and up take c_fc's.
    ratios = {"q": 512 / 768, "k": 512 / 768, "v": 1024 / 768, "o": 1024 / 768, "gate": 3072 / 768, "up": 3072 / 768, "down": 3584 / 768}
    target_linear = n_layers * sum(i * o for i, o in shapes.values())
    target_all = target_linear + vocab * d  # tied embedding and head
    tokens = batch_rows * seq_len
    for name, cs in [("VPD 0.6B C", VPD_C), ("vpd4l ratios", {k: round(r * min(shapes[k])) for k, r in ratios.items()})]:
        C = n_layers * sum(cs.values())
        uv = n_layers * sum(cs[k] * (i + o) for k, (i, o) in shapes.items())
        d_in = n_layers * sum(i for i, _ in shapes.values())
        # (parameters, attention FLOPs per token over all blocks): one transformer over every site's
        # input concatenated, or one per layer over its residual stream (blocks_per_chunk 1).
        def transformer(width: int, dm: int, blocks: int, hidden: int, out: int, copies: int) -> tuple[int, float]:
            params = copies * (width * dm + dm + blocks * (4 * dm * dm + 2 * dm * hidden + hidden + dm) + dm * out + out)
            return params, copies * blocks * 2 * 2 * seq_len * dm
        gates = {"global (vpd4l-sized)": transformer(d_in, 2048, 8, 8192, C, 1), "chunkwise (VPD 0.6B)": transformer(d, 1024, 4, 4096, sum(cs.values()), n_layers)}
        # Per token, matmul FLOPs (2 per multiply-add) of one masked forward: x V and (x V m) U per
        # site, the delta's x (W - VU)^T, attention over a causal half, and the head.
        attn = n_layers * 2 * 2 * seq_len / 2 * q
        plain = 2 * target_linear + attn + 2 * vocab * d
        masked = plain + 2 * uv
        for label, (ci, ci_attn) in gates.items():
            ci_fwd = 2 * ci + ci_attn
            # decompose's step: the target forward; the CI forward and backward; 2 PPGD warmup
            # forwards with source backwards; the stochastic reconstruction forward and backward;
            # the PPGD reconstruction forward, its source backward and the main backward.
            step = plain + 3 * ci_fwd + 2 * 2 * masked + 3 * masked + 4 * masked
            trainable = uv + ci
            print(f"{name}, CI {label}: C total {C:,}; U/V {uv / 1e9:.3f}B; CI {ci / 1e9:.3f}B; target {target_all / 1e9:.3f}B ({target_linear / 1e9:.3f}B in the sites)")
            print(f"  state: params+grads+AdamW f32 {16 * trainable / 2**30:.1f} GiB; PPGD sources+Adam f32 per token {12 * (C + 7 * n_layers) / 2**20:.2f} MiB ({12 * (C + 7 * n_layers) * tokens / 2**30:.0f} GiB at {batch_rows}x{seq_len})")
            print(f"  activations kept per masked forward with grad (x V and masked, bf16): {4 * C / 2**20:.2f} MiB per token")
            print(f"  FLOPs per token per step {step / 1e9:.1f} G; per step at {batch_rows}x{seq_len}: {step * tokens / 1e15:.2f} PFLOP")


def smoke(layers: int = 2, steps: int = 3) -> None:
    """A few steps of VPD's `decompose` on the target's first `layers` layers (a small C and gate
    network, short rows), on whatever device `decompose` picks: the wiring of sites, loaders and
    the CI network's inputs."""
    sys.path.insert(0, str(Path.home() / "mpd-data/param-decomp/nano_param_decomp"))
    from run import Config, decompose

    hf = load(torch.float32)
    hf.model.layers = hf.model.layers[:layers]
    hf.config.num_hidden_layers = layers
    seq_len = 128
    cfg = Config(C_per_module=c_per_module(layers, {k: 16 for k in VPD_C}), n_steps=steps, batch_size=2, seq_len=seq_len, faithfulness_warmup_steps=2, ci_d_model=64, ci_n_blocks=1, ci_n_heads=4, ci_mlp_hidden=128, eval_freq=steps, slow_eval_freq=10**9, slow_eval_on_first_step=False, eval_batch_size=2, log_every=1)
    decompose(Qwen3Target(hf), cfg, train_loader(2, seq_len=seq_len), heldout_loader(2, rows=4, seq_len=seq_len))


if __name__ == "__main__":
    {"check": check, "cost": cost, "smoke": smoke}[sys.argv[1]]()
