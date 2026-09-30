"""#2951 operator-first probes: one reader for a standard decoder checkpoint, architecture read from config.

Every weight-level probe gets its operators from here, already folded, so the same code serves any
decoder in the table below and a new family is a table row, not a new code path. Tensors are read lazily
one at a time as float64 (bf16 / f32 -> f64 is exact), so a 1B-parameter model is never materialised in
float64 (the probes work one layer at a time).

Norm placement (residual stream h, RMSNorm N_* = gain * v / rms(v), rms excluded where stated):
  pre  (llama, mistral, qwen2, qwen3): h += attn(N_in(h)),        h += mlp(N_post_attn(h))
  post (olmo2):                        h += N_post_attn(attn(h)),  h += N_post_ff(mlp(h))
so in a post-norm model the blocks read the raw residual and their outputs are rescaled before the add.

q/k norm scope, read from the gain's shape: "head" (Qwen3: gain of length head_dim, one RMS per head),
"full" (OLMo 2: gain of length H * head_dim, ONE RMS over the whole projection, shared by all heads),
or absent. Rotary pairing is the HF rotate_half convention (plane j = coords (j, j + hd/2)).
"""

from __future__ import annotations

import glob
import json
import os
import struct

import numpy as np

PLACEMENT = {"llama": "pre", "mistral": "pre", "qwen2": "pre", "qwen3": "pre", "olmo2": "post"}


def compact_json(obj):
    """One top-level key per line, compact values: keeps receipts under the
    repository's tracked-file line limit (build.rs MAX_TRACKED_FILE_LINES)."""
    if isinstance(obj, dict):
        body = ",\n".join(
            json.dumps(k) + ": " + json.dumps(v, separators=(",", ":")) for k, v in obj.items()
        )
        return "{\n" + body + "\n}\n"
    return json.dumps(obj, separators=(",", ":")) + "\n"


def snapshot(model, revision="main"):
    """Local snapshot directory (downloads if needed; honours HF_HUB_CACHE, e.g. ~/mpd-data/hf)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


def placement(config):
    kind = config["model_type"] if isinstance(config, dict) else config.model_type
    if kind not in PLACEMENT:
        raise SystemExit(f"model_type {kind!r}: norm placement not declared in PLACEMENT")
    return PLACEMENT[kind]


def mlp_input_norm(layer, where):
    """HF module normalising the MLP input (pre-norm), or None when the MLP reads the raw residual."""
    return layer.post_attention_layernorm if where == "pre" else None


def mlp_output_norm(layer, where):
    """HF module applied to the MLP output before the residual add (post-norm), or None."""
    return layer.post_feedforward_layernorm if where == "post" else None


class Decoder:
    def __init__(self, model, revision="main"):
        self.model, self.revision = model, revision
        self.snap = snapshot(model, revision)
        self.index = {}
        for path in sorted(glob.glob(os.path.join(self.snap, "*.safetensors"))):
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            for name, meta in header.items():
                if name != "__metadata__":
                    self.index[name] = (path, 8 + n, meta)
        cfg = self.config = json.load(open(os.path.join(self.snap, "config.json")))
        self.where = placement(cfg)
        self.H, self.KV, self.d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
        self.L, self.n_ff = cfg["num_hidden_layers"], cfg["intermediate_size"]
        self.hd = cfg.get("head_dim") or self.d // self.H
        self.group = self.H // self.KV
        self.theta = float(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"])
        self.tied = "lm_head.weight" not in self.index
        qn = "model.layers.0.self_attn.q_norm.weight"
        if qn not in self.index:
            self.qk_norm = None
        else:
            self.qk_norm = {self.hd: "head", self.H * self.hd: "full"}[self.index[qn][2]["shape"][0]]

    def __call__(self, name):
        path, base, meta = self.index[name]
        lo, hi = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + lo)
            raw = f.read(hi - lo)
        if meta["dtype"] == "BF16":
            a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif meta["dtype"] == "F32":
            a = np.frombuffer(raw, dtype=np.float32)
        else:
            raise SystemExit(f"{name}: dtype {meta['dtype']} not handled")
        return a.astype(np.float64).reshape(meta["shape"])

    def describe(self):
        return {"model": self.model, "revision": self.revision, "snapshot": self.snap,
                "model_type": self.config["model_type"], "norm_placement": self.where, "qk_norm": self.qk_norm,
                "tied_embeddings": self.tied, "H": self.H, "KV": self.KV, "head_dim": self.hd, "d_model": self.d,
                "layers": self.L, "d_ff": self.n_ff, "rope_theta": self.theta}

    def _p(self, layer):
        return f"model.layers.{layer}."

    def attn_input_gain(self, layer):
        return self(self._p(layer) + "input_layernorm.weight") if self.where == "pre" else None

    def attn_output_gain(self, layer):
        return self(self._p(layer) + "post_attention_layernorm.weight") if self.where == "post" else None

    def mlp_input_gain(self, layer):
        return self(self._p(layer) + "post_attention_layernorm.weight") if self.where == "pre" else None

    def qk(self, layer, fold_input_norm=True):
        """Q (H, hd, d), K (KV, hd, d) with the q/k-norm gains folded (per-head or full-projection gain,
        reshaped onto its head rows) and, in a pre-norm model, the input RMSNorm gain."""
        p = self._p(layer) + "self_attn."
        Q = self(p + "q_proj.weight").reshape(self.H, self.hd, -1)
        K = self(p + "k_proj.weight").reshape(self.KV, self.hd, -1)
        if self.qk_norm is not None:
            Q = Q * self(p + "q_norm.weight").reshape(-1, self.hd)[:, :, None]
            K = K * self(p + "k_norm.weight").reshape(-1, self.hd)[:, :, None]
        gamma = self.attn_input_gain(layer)
        if fold_input_norm and gamma is not None:
            Q, K = Q * gamma, K * gamma
        return Q, K

    def qk_norm_rms(self, raw, heads, eps):
        """The excluded q/k-norm normalisers of raw projections (heads, hd), one per head: per-head RMS
        ("head"), the single full-projection RMS broadcast to every head ("full"), or 1."""
        if self.qk_norm == "head":
            return np.sqrt((raw**2).mean(-1) + eps)
        if self.qk_norm == "full":
            return np.full(heads, np.sqrt((raw**2).mean() + eps))
        return np.ones(heads)

    def ov(self, layer):
        """Per query head h, factors (Lh d x hd, Rh hd x d) with the residual write Lh Rh: W_O[:, h] W_V[g(h)]
        with the input RMSNorm gain on the right (pre) or the attention-output RMSNorm gain on the left (post);
        the RMS normaliser itself is excluded (a positive scalar per token, the same for every head)."""
        p = self._p(layer) + "self_attn."
        wv, wo = self(p + "v_proj.weight"), self(p + "o_proj.weight")
        g_in, g_out = self.attn_input_gain(layer), self.attn_output_gain(layer)
        out = []
        for h in range(self.H):
            g = h // self.group
            left = wo[:, h * self.hd:(h + 1) * self.hd]
            right = wv[g * self.hd:(g + 1) * self.hd]
            out.append((left if g_out is None else g_out[:, None] * left,
                        right if g_in is None else right * g_in[None, :]))
        return out

    def rope_inverse_frequencies(self):
        """The HF default rotary inverse frequencies theta^(-2j/hd), j < hd/2; a declared rope scaling is refused."""
        scaling = self.config.get("rope_scaling")
        if scaling and scaling.get("rope_type", scaling.get("type", "default")) != "default":
            raise SystemExit(f"rope_scaling {scaling!r} not handled")
        return (self.theta ** (-np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)).tolist()

    def attention_letter(self, layer, prefix):
        """Layer ``layer``'s attention block as the MPD surface's ``attention`` observability letter
        (joint_operators::attention_letters: heads grouped into routing laws by their score operators, one letter
        per law, its summed OV transport) and the tensors it names, ids under ``prefix``.

        A per-head q/k norm (Qwen3) is declared as the block's query_key_norm. A full-projection q/k norm (OLMo 2)
        is ONE normaliser shared by every head, so it cannot separate heads: only its gain is folded into the
        projection rows. The input RMSNorm gain (pre-norm) is the letter's input_gain, folded into the query/key and
        value reads; the attention-output RMSNorm gain (post-norm) its output_gain, on the transports' rows. The
        RMS normalisers themselves are excluded (one positive scalar per token, shared by every head)."""
        p = self._p(layer) + "self_attn."
        tensors = {}

        def put(name, array):
            tensors[prefix + name] = np.ascontiguousarray(array)
            return prefix + name

        def projection(name, gain=None):
            w = self(p + f"{name}_proj.weight")
            bias = p + f"{name}_proj.bias"
            return {"weight": put(name, w if gain is None else w * gain[:, None]),
                    "bias": put(name + "_bias", self(bias)) if bias in self.index else None}

        norm = None
        if self.qk_norm == "full":
            query = projection("q", gain=self(p + "q_norm.weight"))
            key = projection("k", gain=self(p + "k_norm.weight"))
        else:
            query, key = projection("q"), projection("k")
            if self.qk_norm == "head":
                norm = {"epsilon": float(self.config["rms_norm_eps"]), "query_gain": put("q_norm", self(p + "q_norm.weight")),
                        "key_gain": put("k_norm", self(p + "k_norm.weight"))}
        g_in, g_out = self.attn_input_gain(layer), self.attn_output_gain(layer)
        attention = {"geometry": {"model_dim": self.d, "n_heads": self.H, "n_kv_heads": self.KV, "head_dim": self.hd},
                     "rotary": {"pairing": "half_split", "inverse_frequencies": self.rope_inverse_frequencies(),
                                "attention_scaling": 1.0},
                     "score_scale": self.hd ** -0.5, "query": query, "key": key,
                     "value": projection("v"), "output": projection("o"), "query_key_norm": norm}
        letter = {"kind": "attention", "attention": attention,
                  "input_gain": None if g_in is None else put("input_gain", g_in),
                  "output_gain": None if g_out is None else put("output_gain", g_out)}
        return letter, tensors

    def mlp(self, layer):
        p = self._p(layer) + "mlp."
        return self(p + "gate_proj.weight"), self(p + "up_proj.weight"), self(p + "down_proj.weight")

    def mlp_reads(self, layer):
        """[W_gate; W_up] as residual read rows (pre-MLP RMSNorm gain folded in a pre-norm model)."""
        wg, wu, _ = self.mlp(layer)
        reads = np.vstack([wg, wu])
        gain = self.mlp_input_gain(layer)
        return reads if gain is None else reads * gain[None, :]

    def readout(self, rows=None):
        """Unembedding rows times the final RMSNorm gain (lm_head, or the tied embedding)."""
        w = self("model.embed_tokens.weight" if self.tied else "lm_head.weight")
        if rows is not None:
            w = w[rows]
        return w * self("model.norm.weight")[None, :]
