"""The description score of the oracle (#2951): a description z of one of VPD's vpd4l subcomponents,
scored by how well a frozen text-only reader R, given z alone, predicts the model M's next-token
distributions under edits of the subcomponent, plus the description's own length:

  S(z) = L(z) + sum over (x, a) of KL(p_M(. | x, a) || R(. | z, x, a))   in bits,

  L(z)       -log2 of z under a frozen prior model (read as the answer to a fixed request to describe a
             component), so a long or unlikely description pays for itself;
  (x, a)     the experiments: contexts x, the subcomponent's `--top` strongest and `--others` other
             contexts of a row-edit label table (vpd_labels.py --edit row), each cut at its peak p, times
             the edits a: none, removal at p and doubling at p (M's weight edit W + (alpha - 1) u v^T at
             row p, alpha 0 and 2), so cases where the edit changes nothing count as well;
  p_M        M's next-token distribution at p under a, computed here (vpd_labels.Model);
  R(t | ...) R's probability of answering with token t's text and then ending its turn (so different
             tokens are disjoint answers), over M's `--candidates` most probable tokens S, and
             R(other) = 1 - sum over S, the rest; KL over S and the rest.
The reward of a description is -S(z); S(none) = sum KL(p_M || R(. | no description)) is reported beside
it (no description, no L), and saved_bits = S(none) - S(z).

  codelength.py score --labels ROW_TABLE --uv UV --items D.jsonl --reader-backend vllm --reader Qwen/Qwen3-8B
                      --prior-backend vllm --prior Qwen/Qwen3-1.7B --out OUT.jsonl
  codelength.py serve ... --listen HOST:PORT     ({"op": "score", "items": [...]} -> {"ok": {"results": [...]}})
D.jsonl lines {"id", "component": [layer, kind, index], "description"}.
"""

from __future__ import annotations

import argparse
import json
import math
import socketserver
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

import reader as R

LN2 = math.log(2.0)
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
EDITS = (("none", None), ("removed", 0.0), ("made 2 times stronger", 2.0))

NEXT_PROMPT = (
    "A component of a 4-layer language model is described as follows.\n\nDescription: {description}\n\n"
    "The model reads the text below{edit}. Answer with the model's most likely next token after the text, "
    "and nothing else.\n\nText: {text}"
)
PRIOR_PROMPT = "Describe what this component of a language model responds to and what it does."


class Behaviour:
    """A row-edit label table (contexts and peaks) and the target M, giving per subcomponent its
    experiments: (text up to the peak, the edit in words, M's candidate tokens' texts, their
    probabilities, the rest's probability)."""

    def __init__(self, root: Path, uv: Path, top: int, others: int, candidates: int, tokenizer: Path = TOKENIZER):
        import tokenizers
        import torch
        from vpd_labels import Model, load_uv

        self.torch = torch
        self.root = root
        self.top, self.others, self.candidates = top, others, candidates
        self.tokens = load_file(str(root / "contexts.safetensors"))["tokens"]
        self.tok = tokenizers.Tokenizer.from_file(str(tokenizer))
        self.meta = {}
        for path in root.glob("site_*.json"):
            m = json.loads(path.read_text())
            assert m.get("edit") == "row", f"{path}: a row-edit table (vpd_labels.py --edit row) is required"
            self.meta[(m["layer"], m["site"].split(".")[-1])] = (m, path.with_suffix(".safetensors"))
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(dev)
        self.uv = load_uv(dev, uv)

    @lru_cache(maxsize=4)
    def site(self, layer: int, kind: str) -> dict:
        return load_file(str(self.meta[(layer, kind)][1]))

    @lru_cache(maxsize=4096)
    def experiments(self, layer: int, kind: str, c: int) -> tuple:
        """c is the subcomponent's index in the table (its subcomponent number in `subcomponents`)."""
        torch = self.torch
        d = self.site(layer, kind)
        meta = self.meta[(layer, kind)][0]
        chosen = list(range(min(self.top, meta["top"]))) + list(range(meta["top"], meta["top"] + self.others))
        name = f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"
        U, V = self.uv[name]
        sub = int(d["subcomponents"][c])
        dev = self.model.dev
        rows = torch.tensor([int(d["contexts"][c, j]) for j in chosen], device=dev)
        pos = torch.tensor([int(d["position"][c, j]) for j in chosen], device=dev)
        ids = torch.from_numpy(self.tokens[rows.cpu().numpy()].astype(np.int64)).to(dev)
        entering, final = self.model.clean(ids)
        at = torch.arange(len(chosen), device=dev)
        out = []
        for words, alpha in EDITS:
            if alpha is None:
                lp = self.model.log_probs(final[at, pos])
            else:
                edit = (name, V[:, sub][None].expand(len(chosen), -1), U[sub][None].expand(len(chosen), -1), torch.full((len(chosen),), alpha - 1.0, device=dev), pos, pos + 1)
                h, _ = self.model.from_layer(layer, entering[layer], edit)
                lp = self.model.log_probs(h[at, pos])
            p = lp.exp().double().cpu().numpy()
            for r in range(len(chosen)):
                top = np.argsort(-p[r])[: self.candidates]
                text = self.tok.decode(ids[r, : int(pos[r]) + 1].tolist())
                said = "" if alpha is None else f", with this component {words} at its last token"
                out.append((text, said, tuple(self.tok.decode([int(t)]) for t in top), tuple(float(x) for x in p[r, top]), float(max(0.0, 1.0 - p[r, top].sum()))))
        return tuple(out)


class Prompts:
    """Token ids of the scoring prompts in one model's chat template, with the indices scored."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731

    def chat(self, user: str, answer_ids: list[int]) -> tuple[list[int], int]:
        marker = "\u0000ANSWER\u0000"
        text = self.tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": marker}], tokenize=False, enable_thinking=False)
        head, tail = text.split(marker)
        a = self.enc(head)
        return a + answer_ids + self.enc(tail), len(a)

    def answer(self, description: str, text: str, edit: str, candidate: str) -> tuple[list[int], list[int]]:
        """The candidate's text as the reader's whole answer: its tokens and the end of the turn scored."""
        answer = self.enc(candidate)
        ids, start = self.chat(NEXT_PROMPT.format(description=description or "(none)", edit=edit, text=text), answer)
        return ids, list(range(start, start + len(answer) + 1))

    def prior(self, description: str) -> tuple[list[int], list[int]]:
        answer = self.enc(description)
        ids, start = self.chat(PRIOR_PROMPT, answer)
        return ids, list(range(start, start + len(answer)))


def kl_bits(p: tuple, rest: float, log_q: list[float]) -> float:
    """KL(p || q) in bits over the candidates and the rest, q(rest) = 1 - sum q (at least 1e-9)."""
    q = np.exp(np.array(log_q))
    q_rest = max(1e-9, 1.0 - float(q.sum()))
    p_ = np.array(p)
    keep = p_ > 0
    kl = float((p_[keep] * (np.log(p_[keep]) - np.log(np.maximum(q[keep], 1e-300)))).sum())
    if rest > 0:
        kl += rest * (math.log(rest) - math.log(q_rest))
    return kl / LN2


def score(reader, prior, behaviour: Behaviour, items: list[dict]) -> list[dict]:
    """Bits per item (module note), and each component's no-description bits once per call."""
    rp, pp = Prompts(reader.tokenizer), Prompts(prior.tokenizer)
    jobs: list[tuple[object, int, list[int], list[int]]] = []
    exps: dict[tuple, tuple] = {}
    keyed = [(("nothing", tuple(c)), "", tuple(c)) for c in sorted({tuple(it["component"]) for it in items})] + [(i, it["description"], tuple(it["component"])) for i, it in enumerate(items)]
    for key, description, comp in keyed:
        exps[comp] = behaviour.experiments(int(comp[0]), str(comp[1]), int(comp[2]))
        for e, (text, edit, cands, _, _) in enumerate(exps[comp]):
            for cand in cands:
                jobs.append((key, e, *rp.answer(description, text, edit, cand)))
    lps = reader.token_log_probs([j[2] for j in jobs], [j[3] for j in jobs])
    per: dict[tuple, list[float]] = {}
    for (key, e, _, _), lp in zip(jobs, lps):
        per.setdefault((key, e), []).append(float(lp.sum()))
    prior_jobs = [(i, *pp.prior(it["description"])) for i, it in enumerate(items) if it["description"]]
    prior_lps = prior.token_log_probs([j[1] for j in prior_jobs], [j[2] for j in prior_jobs]) if prior_jobs else []
    description_bits = {i: float(-lp.sum() / LN2) for (i, _, _), lp in zip(prior_jobs, prior_lps)}

    def behaviour_bits(key, comp) -> float:
        return sum(kl_bits(p, rest, per[(key, e)]) for e, (_, _, _, p, rest) in enumerate(exps[comp]))

    out = []
    for i, it in enumerate(items):
        comp = tuple(it["component"])
        b = {"description": description_bits.get(i, 0.0), "behaviour": behaviour_bits(i, comp)}
        b["total"] = b["description"] + b["behaviour"]
        nothing = behaviour_bits(("nothing", comp), comp)
        out.append({"id": it.get("id"), "component": list(comp), "bits": b, "nothing_bits": nothing, "reward": -b["total"], "saved_bits": nothing - b["total"],
                    "experiments": len(exps[comp])})
    return out


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if not line.strip():
                continue
            start = time.time()
            try:
                request = json.loads(line)
                if request.get("op") != "score":
                    raise ValueError(f"unknown op {request.get('op')!r} (have score)")
                srv = self.server
                reply = {"ok": {"results": score(srv.reader, srv.prior, srv.labels, request["items"]), "reader": R.describe(srv.reader), "prior": R.describe(srv.prior)}}
            except Exception as e:  # the reply carries the failure; the service keeps running
                reply = {"error": f"{type(e).__name__}: {e}"}
            reply["seconds"] = time.time() - start
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


def backend(kind: str, model: str, args, share: float):
    ns = argparse.Namespace(**{**vars(args), "backend": kind, "model": model, "gpu_memory_utilization": share})
    return R.make_backend(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["score", "serve"])
    ap.add_argument("--labels", required=True, help="a row-edit label table (vpd_labels.py --edit row)")
    ap.add_argument("--uv", required=True, help="VPD's subcomponents (uv.safetensors)")
    ap.add_argument("--candidates", type=int, default=16, help="M's most probable next tokens scored one by one (the rest pooled)")
    ap.add_argument("--tokenizer", default=str(TOKENIZER), help="the target's tokenizer.json")
    ap.add_argument("--reader-backend", required=True, choices=["vllm", "transformers"])
    ap.add_argument("--reader", required=True)
    ap.add_argument("--prior-backend", required=True, choices=["vllm", "transformers"])
    ap.add_argument("--prior", required=True)
    ap.add_argument("--top", type=int, default=4, help="the strongest measured contexts read per component")
    ap.add_argument("--others", type=int, default=4, help="the other measured contexts read per component")
    ap.add_argument("--items")
    ap.add_argument("--out")
    ap.add_argument("--listen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="vllm: the reader's share of the GPU's memory")
    ap.add_argument("--prior-gpu-memory-utilization", type=float, default=0.15, help="vllm: the prior's share, when it is another model")
    args = ap.parse_args()
    reader = backend(args.reader_backend, args.reader, args, args.gpu_memory_utilization)
    prior = reader if (args.prior, args.prior_backend) == (args.reader, args.reader_backend) else backend(args.prior_backend, args.prior, args, args.prior_gpu_memory_utilization)
    behaviour = Behaviour(Path(args.labels), Path(args.uv), args.top, args.others, args.candidates, Path(args.tokenizer))
    if args.command == "serve":
        host, sep, port = args.listen.rpartition(":")
        server = socketserver.TCPServer((host, int(port)), _Handler) if sep and port.isdigit() else socketserver.UnixStreamServer(args.listen, _Handler)
        server.reader, server.prior, server.labels = reader, prior, behaviour
        print(f"code-length scorer: reader {R.describe(reader)}, prior {R.describe(prior)}, listening on {args.listen}", file=sys.stderr, flush=True)
        server.serve_forever()
    items = [json.loads(line) for line in open(args.items) if line.strip()]
    start = time.time()
    rows = score(reader, prior, behaviour, items)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "reader": R.describe(reader), "prior": R.describe(prior)}) + "\n")
    print(json.dumps({"items": len(rows), "seconds": time.time() - start, "mean_reward_bits": float(np.mean([r["reward"] for r in rows])),
                      "mean_saved_bits": float(np.mean([r["saved_bits"] for r in rows]))}))


if __name__ == "__main__":
    main()
