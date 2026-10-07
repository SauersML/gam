"""A tiny random model and behavior for running the graph oracle's whole path in CI (#2951): the export
format of gam_mpd's test_support::tiny_export (2 heads of 4, width 8, MLP 16, vocabulary 11; f64 files
and export.json), a behavior file on it (prompts with a one-token counterfactual), and mech's shapes
entry for it, registered in-process only (shapes.json lists real models).
"""

from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mech  # noqa: E402

D, MLP, VOCAB, HEADS, HEAD_DIM = 8, 16, 11, 2, 4
SHAPES = {"layers": None, "heads": HEADS, "kv_heads": HEADS, "head_dim": HEAD_DIM, "d_model": D, "d_mlp": MLP,
          "vocab": VOCAB, "views": {"native": True, "vpd": None, "library": None, "transcoder": None},
          "source": ["bench/oracle/graph/e2e/tiny.py"]}


def register(layers: int) -> None:
    """Lets mech trace programs for model "tiny" in this process (trace_inline)."""
    mech.shapes("vpd4l")  # loads the registry
    mech._SHAPES["tiny"] = {**SHAPES, "layers": layers}
    if "tiny" not in mech.MODELS:
        mech.MODELS = mech.MODELS + ("tiny",)


def export(directory: Path, layers: int = 2, seed: int = 0) -> Path:
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}

    def write(name: str, shape: tuple[int, int], values: list[float]) -> None:
        (directory / f"{name}.f64").write_bytes(struct.pack(f"<{len(values)}d", *values))
        files[name] = {"shape": list(shape)}

    draw = lambda n, scale: [(rng.random() - 0.5) * scale for _ in range(n)]
    write("wte", (VOCAB, D), draw(VOCAB * D, 2.0))
    write("final_norm.gain", (1, D), [1 + g for g in draw(D, 0.4)])
    for l in range(layers):
        for name, shape in [("attn.q_proj", (D, D)), ("attn.k_proj", (D, D)), ("attn.v_proj", (D, D)), ("attn.o_proj", (D, D)),
                            ("mlp.c_fc", (MLP, D)), ("mlp.down_proj", (D, MLP))]:
            write(f"blocks.{l}.{name}", shape, draw(shape[0] * shape[1], 1.0))
        for g in ("rms1", "rms2"):
            write(f"blocks.{l}.{g}.gain", (1, D), [1 + v for v in draw(D, 0.4)])
    write("tokens", (6, 12), [float(rng.randrange(VOCAB)) for _ in range(72)])
    record = {"config": {"d_model": D, "n_layers": layers, "n_heads": HEADS, "n_kv_heads": HEADS, "head_dim": HEAD_DIM,
                         "d_mlp": MLP, "vocab": VOCAB, "rope_theta": 10000.0, "rope_pairing": "rotate_half", "norm_eps": 1e-6,
                         "mlp_act": "gelu_tanh", "tied_embeddings": True},
              "files": files}
    (directory / "export.json").write_text(json.dumps(record))
    return directory


def behavior(path: Path, prompts: int = 12, length: int = 10, seed: int = 1) -> Path:
    """Copying: the sequence repeats its first token at the end; the counterfactual changes that token
    (and so the answer)."""
    rng = random.Random(seed)
    items = []
    for _ in range(prompts):
        ids = [rng.randrange(VOCAB) for _ in range(length - 1)]
        ids.append(ids[0])
        other = (ids[0] + 1 + rng.randrange(VOCAB - 1)) % VOCAB
        cf = [other] + ids[1:-1] + [other]
        items.append({"text": "", "token_ids": ids, "target_positions": [length - 2],
                      "counterfactual": {"text": "", "token_ids": cf}})
    record = {"id": "tiny.copy", "model": "tiny", "family": "copy", "description": "the last token repeats the first",
              "frequency": None, "prompts": items, "split": "train", "model_accuracy": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path
