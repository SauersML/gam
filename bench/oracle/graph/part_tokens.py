"""Part tokens (#2951): every part of the target's attached view is one token of the oracle, whose input
embedding and output logit row are computed from the part's own weight vectors, so naming a part in a
question or emitting one in a program is sensing it. Shared by predict/sft.py, predict/eval_kl.py and
rl/train.py.

Tokens (mech addresses in brackets; token_of / address_of convert both ways):
  <p:L.S.I>      VPD subcomponent I of layer L's site S, S in q k v o fc down   (PD.vpd[L].v_proj[I] ...)
  <p:L.h.I>      native head I of layer L                                     (L[L].head[I])
  <p:L.m>        layer L's whole native MLP                                    (L[L].mlp[:])
  <p:L.a>        layer L's whole native attention                              (L[L].head[:])
Native heads and blocks are parts only where nothing decomposes them (the registry lists what exists).

Features. A part's feature vector is fixed by its kind:
  VPD subcomponent u v^T of a site (read v in R^d_in, write u in R^d_out):
      [v / |v| * sqrt(d_in), u / |u| * sqrt(d_out), log |v|, log |u|]
  native head, attention or MLP: predict/vectors.py's 8 directions with their log singular values, flattened.
Maps. Per kind, P_in (linear, rescaled to the RMS of the oracle's token embeddings) gives the token's input
embedding and P_out (linear) its output row: the logit of part p after hidden state h is h . P_out(f_p),
next to the base vocabulary's logits, so choosing a part is a softmax over the parts' own vectors. There
are no per-ID parameters: relabeling the parts changes nothing (tested). For many parts the softmax is
over all of them at once (38,912 VPD subcomponents of vpd4l are 0.26 of Qwen3's vocabulary); a two-level
choice (site, then part) can sit on top without changing the rows.
vLLM: materialize() writes the extended embedding and lm_head (base vocabulary plus one row per part) and the
tokenizer with the part tokens added, for the sampler to reload after each training round.

  part_tokens.py build --out REGISTRY.safetensors [--native MODEL_DIR --pieces PIECES.json] [--vpd UV.safetensors]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
SITES = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "fc", "down_proj": "down"}
CODES = {v: k for k, v in SITES.items()}


def token_of(address: str) -> str:
    if m := re.fullmatch(r"PD\.vpd\[(\d+)\]\.(\w+)\[(\d+)\]", address):
        return f"<p:{m[1]}.{SITES[m[2]]}.{m[3]}>"
    if m := re.fullmatch(r"L\[(\d+)\]\.head\[(\d+)\]", address):
        return f"<p:{m[1]}.h.{m[2]}>"
    if m := re.fullmatch(r"L\[(\d+)\]\.mlp\[:\]", address):
        return f"<p:{m[1]}.m>"
    if m := re.fullmatch(r"L\[(\d+)\]\.head\[:\]", address):
        return f"<p:{m[1]}.a>"
    raise ValueError(f"no part token for {address!r}")


def address_of(token: str) -> str:
    if m := re.fullmatch(r"<p:(\d+)\.(q|k|v|o|fc|down)\.(\d+)>", token):
        return f"PD.vpd[{m[1]}].{CODES[m[2]]}[{m[3]}]"
    if m := re.fullmatch(r"<p:(\d+)\.h\.(\d+)>", token):
        return f"L[{m[1]}].head[{m[2]}]"
    if m := re.fullmatch(r"<p:(\d+)\.m>", token):
        return f"L[{m[1]}].mlp[:]"
    if m := re.fullmatch(r"<p:(\d+)\.a>", token):
        return f"L[{m[1]}].head[:]"
    raise ValueError(f"not a part token: {token!r}")


def site_of(address: str) -> str:
    """The site a part sits in (the first level of the site -> part hierarchy): a VPD site of a layer, or a
    layer's native attention or MLP."""
    if m := re.fullmatch(r"PD\.vpd\[(\d+)\]\.(\w+)\[\d+\]", address):
        return f"{m[1]}.{SITES[m[2]]}"
    if m := re.fullmatch(r"L\[(\d+)\]\.(head|mlp)\[.*\]", address):
        return f"{m[1]}.{'attn' if m[2] == 'head' else 'mlp'}"
    raise ValueError(address)


def kind_of(address: str) -> str:
    """The feature kind: one map per kind (VPD site, native head, MLP or attention)."""
    if m := re.fullmatch(r"PD\.vpd\[\d+\]\.(\w+)\[\d+\]", address):
        return "vpd." + SITES[m[1]]
    if re.fullmatch(r"L\[\d+\]\.head\[\d+\]", address):
        return "head"
    if address.endswith(".mlp[:]"):
        return "mlp"
    return "attn"


class Registry:
    """The parts in a fixed order (the order of their token ids) and their features by kind."""

    def __init__(self, addresses: list[str], features: dict[str, torch.Tensor]):
        self.addresses = list(addresses)
        self.kinds = [kind_of(a) for a in self.addresses]
        self.features = features  # kind -> [parts of that kind in registry order, F_kind]
        self.tokens = [token_of(a) for a in self.addresses]
        self.index = {a: i for i, a in enumerate(self.addresses)}
        self.by_kind = {k: [i for i, kk in enumerate(self.kinds) if kk == k] for k in features}
        self.sites = sorted({site_of(a) for a in self.addresses})
        self.site = [self.sites.index(site_of(a)) for a in self.addresses]  # site index per part

    @staticmethod
    def load(path) -> "Registry":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            addresses = json.loads(f.metadata()["addresses"])
            features = {k[len("kind:"):]: f.get_tensor(k) for k in f.keys()}
        return Registry(addresses, features)

    def save(self, path):
        from safetensors.torch import save_file

        save_file({f"kind:{k}": v.contiguous() for k, v in self.features.items()}, str(path), metadata={"addresses": json.dumps(self.addresses)})

    def permuted(self, order: list[int]) -> "Registry":
        """The same parts relabeled (for the invariance test)."""
        addresses = [self.addresses[i] for i in order]
        features = {}
        for k in self.features:
            rows = [self.by_kind[k].index(i) for i in order if self.kinds[i] == k]
            features[k] = self.features[k][rows]
        return Registry(addresses, features)

    def rewrite(self, text: str) -> str:
        """Every address of a registry part in a text replaced by its token."""
        if not hasattr(self, "_pattern"):
            alts = sorted(self.addresses, key=len, reverse=True)
            self._pattern = re.compile("|".join(re.escape(a) for a in alts))
        return self._pattern.sub(lambda m: self.tokens[self.index[m[0]]], text)


class PartTokens(nn.Module):
    """P_in and P_out per kind; rows in registry order; embed and logits next to the base vocabulary.

    Site -> part: a part's input embedding and output row are its own map plus its site's, where a site is
    represented by the mean feature of its parts through a per-kind site map (so the rows share a site
    component and site_logits / part_logits_in give the two-level choice for views too large to score at
    once). Still permutation-invariant: a site's mean does not depend on the parts' order."""

    def __init__(self, reg: Registry, hidden: int, emb_rms: float, base_vocab: int, dev=None):
        super().__init__()
        self.reg, self.base_vocab, self.emb_rms = reg, base_vocab, emb_rms
        names = {k: k.replace(".", "_") for k in reg.features}
        self.names = names
        self.p_in = nn.ModuleDict({names[k]: nn.Linear(f.shape[1], hidden) for k, f in reg.features.items()})
        self.p_out = nn.ModuleDict({names[k]: nn.Linear(f.shape[1], hidden, bias=False) for k, f in reg.features.items()})
        self.s_in = nn.ModuleDict({names[k]: nn.Linear(f.shape[1], hidden, bias=False) for k, f in reg.features.items()})
        self.s_out = nn.ModuleDict({names[k]: nn.Linear(f.shape[1], hidden, bias=False) for k, f in reg.features.items()})
        for m in list(self.p_out.values()) + list(self.s_out.values()):
            nn.init.normal_(m.weight, std=0.02 / m.weight.shape[1] ** 0.5)  # parts start improbable
        if dev is not None:
            self.to(dev)
        self.feats = {k: f.to(dev) if dev is not None else f for k, f in reg.features.items()}
        self.idx = {k: torch.tensor(reg.by_kind[k], device=dev) for k in reg.features}
        # Per kind: each part's site (within the kind's sites) and the sites' mean features.
        self.site_local, self.site_feats, self.site_global = {}, {}, {}
        for k, f in self.feats.items():
            sites = [reg.site[i] for i in reg.by_kind[k]]
            uniq = sorted(set(sites))
            local = torch.tensor([uniq.index(x) for x in sites], device=f.device)
            self.site_local[k] = local
            self.site_feats[k] = torch.zeros(len(uniq), f.shape[1], device=f.device).index_add(0, local, f.float()) / torch.bincount(local)[:, None].float()
            self.site_global[k] = torch.tensor(uniq, device=f.device)

    def rows(self, which: str) -> torch.Tensor:
        """[parts, hidden]: input embeddings (which = "in", RMS of the token embeddings) or output rows; each
        the part's map plus its site's."""
        maps, smaps = (self.p_in, self.s_in) if which == "in" else (self.p_out, self.s_out)
        out = None
        for k, f in self.feats.items():
            y = maps[self.names[k]](f.float()) + smaps[self.names[k]](self.site_feats[k])[self.site_local[k]]
            if which == "in":
                y = self.emb_rms * y / y.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            if out is None:
                out = torch.zeros(len(self.reg.addresses), y.shape[1], device=y.device, dtype=y.dtype)
            out = out.index_copy(0, self.idx[k], y)
        return out

    def site_rows(self) -> torch.Tensor:
        """[sites, hidden]: the output rows of the sites (the first level of the two-level choice)."""
        out = torch.zeros(len(self.reg.sites), next(iter(self.s_out.values())).weight.shape[0], device=next(self.parameters()).device)
        for k in self.feats:
            out = out.index_copy(0, self.site_global[k], self.s_out[self.names[k]](self.site_feats[k]))
        return out

    def site_logits(self, h):
        """Logits of the sites after hidden state h."""
        return h.float() @ self.site_rows().T

    def part_logits_in(self, h, site: int):
        """(registry indices, logits) of the parts of one site: the second level."""
        idx = [i for i, s in enumerate(self.reg.site) if s == site]
        return idx, h.float() @ self.rows("out")[idx].T

    # The interface rl/train.py uses.
    def tokens(self) -> list[str]:
        return self.reg.tokens

    def input_rows(self) -> torch.Tensor:
        return self.rows("in")

    def output_rows(self) -> torch.Tensor:
        return self.rows("out")

    def embed(self, model, ids: torch.Tensor) -> torch.Tensor:
        base = model.get_input_embeddings()
        part = ids >= self.base_vocab
        emb = base(ids.clamp(max=self.base_vocab - 1))
        if bool(part.any()):
            emb = emb.clone()
            emb[part] = self.rows("in")[ids[part] - self.base_vocab].to(emb.dtype)
        return emb

    def logits(self, model, h: torch.Tensor) -> torch.Tensor:
        """Base vocabulary logits (rows below base_vocab) and one logit per part."""
        base = model.lm_head(h)[..., : self.base_vocab]
        return torch.cat([base.float(), h.float() @ self.rows("out").T.float()], dim=-1)

    def add_to_tokenizer(self, tok):
        """Adds the part tokens in registry order; their ids are base_vocab + registry index. Ordinary added
        tokens, not special ones: decoding with skip_special_tokens must keep them (rl/train.py does the same)."""
        added = tok.add_tokens(self.reg.tokens, special_tokens=False)
        ids = tok.convert_tokens_to_ids(self.reg.tokens)
        if ids != list(range(self.base_vocab, self.base_vocab + len(self.reg.tokens))):
            raise ValueError(f"part token ids are not contiguous from {self.base_vocab} ({added} added)")
        return tok

    @torch.no_grad()
    def materialize(self, model, tok, out_dir):
        """The extended vocabulary for a sampler: embed_tokens and lm_head with one row per part after the
        base vocabulary (safetensors, the model's dtype) and the tokenizer with the part tokens."""
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dtype = model.get_input_embeddings().weight.dtype
        emb = torch.cat([model.get_input_embeddings().weight[: self.base_vocab], self.rows("in").to(dtype)])
        head = torch.cat([model.lm_head.weight[: self.base_vocab], self.rows("out").to(dtype)])
        save_file({"model.embed_tokens.weight": emb.contiguous().cpu(), "lm_head.weight": head.contiguous().cpu()}, str(out / "part_vocab.safetensors"))
        tok.save_pretrained(str(out))
        (out / "part_tokens.json").write_text(json.dumps({"base_vocab": self.base_vocab, "parts": len(self.reg.addresses)}))
        return emb.shape[0]


def build(native: str | None, pieces: str | None, vpd: str | None) -> Registry:
    addresses, features = [], {}
    if native:
        sys.path.insert(0, str(HERE / "predict"))
        from transformers import AutoModelForCausalLM

        import vectors

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(native, dtype=torch.bfloat16).to(dev).float().eval()
        with torch.no_grad():
            for a in json.load(open(pieces)):
                features.setdefault(kind_of(a), []).append(vectors.part_vectors(model, a).flatten().cpu())
                addresses.append(a)
    if vpd:
        from safetensors.torch import load_file

        uv = load_file(vpd)
        for name in sorted({k.rsplit(".", 1)[0] for k in uv}):
            layer, site = int(name.split(".")[1]), name.split(".")[3]
            U, V = uv[f"{name}.U"].float(), uv[f"{name}.V"].float()  # [C, d_out], [d_in, C]
            nu, nv = U.norm(dim=1), V.norm(dim=0)
            f = torch.cat([(V / nv).T * V.shape[0] ** 0.5, U / nu[:, None] * U.shape[1] ** 0.5, nv.log()[:, None], nu.log()[:, None]], dim=1)
            features.setdefault(f"vpd.{SITES[site]}", []).append(f)
            addresses += [f"PD.vpd[{layer}].{site}[{c}]" for c in range(U.shape[0])]
    feats = {k: torch.cat(v) if v[0].dim() == 2 else torch.stack(v) for k, v in features.items()}
    return Registry(addresses, feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--native", default="")
    b.add_argument("--pieces", default="")
    b.add_argument("--vpd", default="")
    args = ap.parse_args()
    reg = build(args.native or None, args.pieces or None, args.vpd or None)
    reg.save(args.out)
    print(json.dumps({"parts": len(reg.addresses), "kinds": {k: list(v.shape) for k, v in reg.features.items()}}))


if __name__ == "__main__":
    main()
