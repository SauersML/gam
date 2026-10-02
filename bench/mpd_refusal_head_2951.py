"""The frozen blocks after a decomposed window, as a head process for gam_mpd::masked::Head (#2951).

usage: MPD_MEM_GIB=4 python mpd_refusal_head_2951.py MODEL_DIR FIRST SOCKET [DEVICE]

Holds a Hugging Face causal LM's blocks FIRST.. and its readout (MODEL_DIR holds config.json and
model.safetensors), computed in fp32 on its own stored weights, and serves one client (`mpd_refusal_pieces_2951 fit`) on the Unix socket SOCKET, little-endian (a server
of its own rather than the driver's child, so each process holds its own memory reservation):

  request   u8 op (1 logits, 2 pullback, 0 quit), u32 rows,
            u32[rows] sequence, u32[rows] position, u8[rows] scored, f32[rows × d] x (the stream entering FIRST),
            and for op 2 f32[S × vocab] the cotangent at the S scored rows' logits;
  response  op 1: f32[S × vocab] the scored rows' logits;  op 2: f32[rows × d] the cotangent at x.

Each sequence (a run of equal sequence ids) runs on its own, causal over its rows at their positions.
"""

import os
import socket
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def blocks_from(model_dir, first):
    """The model's blocks `first..` (renumbered from 0), embedding, final norm and readout, streamed tensor by
    tensor from model.safetensors: linear maps and the embedding stay in their stored (bf16) values and widen
    them on the fly, so every product runs in fp32 on the exact weights while only the stored bytes are held."""
    config = AutoConfig.from_pretrained(model_dir)
    layers = range(first, config.num_hidden_layers)
    config.num_hidden_layers = len(layers)
    if getattr(config, "layer_types", None):
        config.layer_types = [config.layer_types[l] for l in layers]
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    params = dict(model.named_parameters())
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
        for key in f.keys():
            name = key
            if key.startswith("model.layers."):
                layer = int(key.split(".")[2])
                if layer < first:
                    continue
                name = key.replace(f"model.layers.{layer}.", f"model.layers.{layer - first}.", 1)
            if name in params:
                params[name].data.copy_(f.get_tensor(key))
    model.tie_weights()

    def linear(self, x):
        return F.linear(x, self.weight.float(), None if self.bias is None else self.bias.float())

    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.forward = linear.__get__(mod)
        elif not isinstance(mod, torch.nn.Embedding):
            for p in mod.parameters(recurse=False):
                p.data = p.data.float()
            for name, b in list(mod.named_buffers(recurse=False)):
                if b.is_floating_point():
                    setattr(mod, name, b.float())
    return model.eval()


def main():
    model_dir, first, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    # The CPU by default: its footprint is the weights and one request's activations (no device cache).
    device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    torch.set_num_threads(int(os.environ.get("HEAD_THREADS", "6")))
    model = blocks_from(model_dir, first).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    d, vocab = model.config.hidden_size, model.config.vocab_size
    if os.path.exists(path):
        os.remove(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    print(f"head: blocks {first}.. on {device}, serving {path}", file=sys.stderr, flush=True)
    connection = server.accept()[0]
    stdin, stdout = connection.makefile("rb"), connection.makefile("wb")

    def read(n):
        data = stdin.read(n)
        if len(data) != n:
            raise SystemExit(0)
        return data

    while True:
        op = read(1)[0]
        if op == 0:
            return
        rows = struct.unpack("<I", read(4))[0]
        sequence = np.frombuffer(read(4 * rows), dtype="<u4")
        position = np.frombuffer(read(4 * rows), dtype="<u4")
        scored = np.frombuffer(read(rows), dtype=np.uint8).astype(bool)
        x = torch.from_numpy(np.frombuffer(read(4 * rows * d), dtype="<f4").reshape(rows, d).copy()).to(device)
        cot = None
        if op == 2:
            S = int(scored.sum())
            cot = torch.from_numpy(np.frombuffer(read(4 * S * vocab), dtype="<f4").reshape(S, vocab).copy()).to(device)
        starts = [0] + [r for r in range(1, rows) if sequence[r] != sequence[r - 1]] + [rows]
        out, grads, k = [], [], 0
        for a, b in zip(starts[:-1], starts[1:]):
            keep = torch.from_numpy(np.nonzero(scored[a:b])[0]).to(device)
            pos = torch.from_numpy(position[a:b].astype(np.int64))[None].to(device)
            emb = x[a:b][None].clone().requires_grad_(op == 2)
            with torch.set_grad_enabled(op == 2):
                hidden = model.model(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state[0]
                logits = model.lm_head(hidden.index_select(0, keep))
            if op == 1:
                out.append(logits.detach())
            else:
                n = keep.numel()
                (logits * cot[k:k + n]).sum().backward()
                grads.append(emb.grad[0].detach())
                k += n
        result = torch.cat(out if op == 1 else grads).float().cpu().numpy().astype("<f4")
        stdout.write(result.tobytes())
        stdout.flush()
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
