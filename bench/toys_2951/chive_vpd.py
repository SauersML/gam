"""Case `chive_vpd`: a small CHIVE (Karvonen et al., arXiv 2608.16747) on VPD's 4-layer Pile model.

The protocol, kept with CHIVE's template and thresholds:

* prompts are prefixes of Pile validation rows; the target is sampled 30 times per prompt at
  temperature 1 for `K` tokens;
* a behaviour is a token appearing among the `K` sampled tokens, flagged when its rate is at least
  30% (tokens among the corpus's most frequent are not behaviours of interest); the classifier is
  that exact token match, frozen;
* each behaviour gets 10 prompt edits (substitutions near the end, at random and at the
  behaviour token's own occurrences, deletions, a swap), each measured by 30 more samples;
* each edit becomes the claim "this edit changes the behavior rate by >= 30 pp", true when it moved
  the rate by at least 50 pp, false when by at most 15 pp (others make no claim); claims are
  balanced 50/50 and split by prompt into dev and held out (the held-out labels sealed).

A predictor sees the transcript (the prompt, the 30 samples, the behaviour and its rate) and the
claim, never the model on the edit. `predict` fits two predictors on the dev claims and scores the
held-out claims by AUROC: transcript-only features, and the same plus the read-only dependencies of
the original forward pass (the first-order change of the behaviour token's log-probability under
the edit, from one backward pass on the original prompt, and each edited position's direct effect
on the behaviour logit through every head's attention and output map).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vpd_2951"))
from vpd_model import DATA, TARGET_DIR, gelu_tanh, load_target, rms  # noqa: E402

from common import ROOT  # noqa: E402

K, SAMPLES, RATE, EDITS = 3, 30, 0.3, 10
TRUE_MOVE, FALSE_MOVE = 0.5, 0.15
FREQUENT = 200


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def sample(model, prompt, generator):
    """`SAMPLES` continuations of `K` tokens at temperature 1."""
    ids = torch.tensor(np.tile(prompt, (SAMPLES, 1)), device=model.wte.device)
    for _ in range(K):
        logits = model.hidden(ids)[:, -1] @ model.wte.T
        probs = torch.softmax(logits.cpu().double(), -1)
        nxt = torch.multinomial(probs, 1, generator=generator).to(ids.device)
        ids = torch.cat([ids, nxt], 1)
    return ids[:, len(prompt):].cpu().numpy()


def rate(samples, token):
    return float((samples == token).any(1).mean())


def edits_for(prompt, behaviour, rng, unigram):
    """Ten edits: substitutions near the end, at random, at the behaviour token's occurrences,
    deletions and a swap."""
    n = len(prompt)
    near = list(range(max(1, n - 8), n))
    far = list(range(1, max(2, n - 8)))
    replacement = lambda old: int(next(t for t in rng.choice(len(unigram), size=64, p=unigram) if t not in (old, behaviour)))
    out = []
    for j in rng.choice(near, size=min(3, len(near)), replace=False):
        out.append({"kind": "substitute", "position": int(j), "token": replacement(prompt[j])})
    for j in rng.choice(far, size=min(2, len(far)), replace=False):
        out.append({"kind": "substitute", "position": int(j), "token": replacement(prompt[j])})
    occurrences = [int(j) for j in np.nonzero(prompt == behaviour)[0] if j > 0]
    for j in occurrences[-2:]:
        out.append({"kind": "substitute", "position": j, "token": replacement(prompt[j])})
    while len(out) < 7:
        j = int(rng.choice(near))
        out.append({"kind": "substitute", "position": j, "token": replacement(prompt[j])})
    out.append({"kind": "delete", "position": int(rng.choice(near))})
    out.append({"kind": "delete", "position": int(rng.choice(far))})
    j = int(rng.choice(near[1:] if len(near) > 1 else near))
    out.append({"kind": "swap", "positions": [j - 1, j]})
    return out[:EDITS]


def apply(prompt, edit):
    x = prompt.copy()
    if edit["kind"] == "substitute":
        x[edit["position"]] = edit["token"]
    elif edit["kind"] == "delete":
        x = np.delete(x, edit["position"])
    else:
        i, j = edit["positions"]
        x[i], x[j] = x[j], x[i]
    return x


def describe(edit, prompt, decode):
    if edit["kind"] == "substitute":
        j = edit["position"]
        return f"replace the token {decode([prompt[j]])!r} at position {j} (of {len(prompt)}) with {decode([edit['token']])!r}"
    if edit["kind"] == "delete":
        j = edit["position"]
        return f"delete the token {decode([prompt[j]])!r} at position {j} (of {len(prompt)})"
    i, j = edit["positions"]
    return f"swap the tokens {decode([prompt[i]])!r} and {decode([prompt[j]])!r} at positions {i} and {j} (of {len(prompt)})"


def forward_from(model, x):
    """The final normed residual from embeddings `x` (`Target.hidden` after the embedding), with
    each layer's normed input and attention pattern."""
    B, T, _ = x.shape
    trace = []
    for i in range(model.n_layer):
        s = lambda k: model.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
        h = rms(x, model.norms[2 * i], model.eps)
        q = s("q_proj")(h).view(B, T, model.n_head, model.hd).transpose(1, 2)
        k = s("k_proj")(h).view(B, T, model.n_head, model.hd).transpose(1, 2)
        v = s("v_proj")(h).view(B, T, model.n_head, model.hd).transpose(1, 2)
        q, k = model._rope(q, T), model._rope(k, T)
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(model.hd)).masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf")).softmax(-1)
        trace.append((a, v))
        y = a @ v
        x = x + s("o_proj")(y.transpose(1, 2).reshape(B, T, -1))
        h = rms(x, model.norms[2 * i + 1], model.eps)
        x = x + s("down_proj")(gelu_tanh(s("c_fc")(h)))
    return x, trace


def dependencies(model, prompt, token):
    """Read-only dependencies of the behaviour token's next-position log-probability on the
    original prompt: its gradient in every position's embedding, and per position its direct
    effect on the token's logit through each head (attention weight times the head's write, read by
    the token's unembedding through the final norm at the last position)."""
    ids = torch.tensor(prompt[None], device=model.wte.device)
    x = model.wte[ids].clone().requires_grad_(True)
    final, trace = forward_from(model, x)
    normed = rms(final, model.ln_f, model.eps)
    logp = torch.log_softmax(normed[0, -1] @ model.wte.T, -1)[token]
    (grad,) = torch.autograd.grad(logp, x)
    with torch.no_grad():
        scale = model.ln_f * torch.rsqrt(final[0, -1].pow(2).mean() + model.eps)
        reader = model.wte[token] * scale
        direct = torch.zeros(len(prompt), device=x.device)
        for i, (a, v) in enumerate(trace):
            W_O = model.site(f"h.{i}.attn.o_proj").W
            for h in range(model.n_head):
                write = v[0, h] @ W_O[:, h * model.hd:(h + 1) * model.hd].T
                direct += a[0, h, -1] * (write @ reader)
    return grad[0].detach().cpu().numpy(), direct.cpu().numpy(), float(logp)


def build(root=ROOT, n_prompts=400, seed=0):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TARGET_DIR / "tokenizer.json"))
    decode = lambda ids: tokenizer.decode([int(t) for t in ids])
    model = load_target(device())
    rows = np.load(DATA, mmap_mode="r")
    counts = np.bincount(np.asarray(rows[:, :512]).ravel(), minlength=model.wte.shape[0]).astype(np.float64)
    frequent = set(np.argsort(-counts)[:FREQUENT].tolist())
    unigram = counts / counts.sum()
    out = root / "chive_vpd"
    out.mkdir(parents=True, exist_ok=True)
    behaviours, claims = [], []
    for p in range(n_prompts):
        row = int(rng.integers(len(rows)))
        end = int(rng.integers(24, 96))
        prompt = np.asarray(rows[row, :end], dtype=np.int64)
        samples = sample(model, prompt, generator)
        tokens, hits = np.unique(samples, return_counts=True)
        rates = {int(t): rate(samples, int(t)) for t in tokens if int(t) not in frequent}
        flagged = [t for t, r in rates.items() if r >= RATE]
        if not flagged:
            continue
        behaviour = max(flagged, key=rates.get)
        base = rates[behaviour]
        record = {"id": f"chive_vpd/{p:04d}", "row": row, "prompt": prompt.tolist(), "behaviour": behaviour, "rate": base,
                  "behaviour_text": decode([behaviour]), "prompt_text": decode(prompt), "samples": samples.tolist(),
                  "sample_texts": [decode(s) for s in samples]}
        behaviours.append(record)
        for e, edit in enumerate(edits_for(prompt, behaviour, rng, unigram)):
            edited = apply(prompt, edit)
            after = rate(sample(model, edited, generator), behaviour)
            touched = [edit["position"]] if "position" in edit else edit["positions"]
            stratum = ("edits the last token" if len(prompt) - 1 in touched else
                       "touches the behaviour token" if any(prompt[j] == behaviour for j in touched) else "earlier tokens")
            claims.append({"id": f"{record['id']}/{e}", "behaviour": record["id"], "edit": edit, "edit_text": describe(edit, prompt, decode), "stratum": stratum,
                           "claim": f"Applying this edit changes the rate of the behavior (the continuation contains {decode([behaviour])!r}) by at least 30 percentage points.",
                           "moved": after - base})
        print(f"prompt {p}: behaviour {decode([behaviour])!r} at {base:.2f}, {len(claims)} edits measured", flush=True)
    labelled = [c for c in claims if abs(c["moved"]) >= TRUE_MOVE or abs(c["moved"]) <= FALSE_MOVE]
    true = [c for c in labelled if abs(c["moved"]) >= TRUE_MOVE]
    false = [c for c in labelled if abs(c["moved"]) <= FALSE_MOVE]
    keep = min(len(true), len(false))
    pick = lambda group: [group[i] for i in sorted(rng.choice(len(group), size=keep, replace=False))]
    balanced = sorted(pick(true) + pick(false), key=lambda c: c["id"])
    held_behaviours = set(rng.choice([b["id"] for b in behaviours], size=len(behaviours) // 2, replace=False).tolist())
    sealed = {}
    for c in balanced:
        c["split"] = "held_out" if c["behaviour"] in held_behaviours else "dev"
        c["label"] = abs(c["moved"]) >= TRUE_MOVE
        if c["split"] == "held_out":
            sealed[c["id"]] = {"label": c.pop("label"), "moved": c.pop("moved")}
    json.dump({"case": "chive_vpd", "protocol": __doc__, "behaviours": behaviours, "claims": balanced}, open(out / "claims.json", "w"))
    (root / "sealed").mkdir(parents=True, exist_ok=True)
    json.dump({"case": "chive_vpd", "labels": sealed}, open(root / "sealed" / "chive_vpd.json", "w"))
    print(f"{len(behaviours)} behaviours, {len(claims)} edits, {len(true)} true and {len(false)} false claims; {2 * keep} kept balanced, {len(sealed)} held out -> {out}")


def features(model, behaviour, claim, cache):
    """Transcript-only features of a claim, and the read-only dependency features."""
    prompt = np.array(behaviour["prompt"])
    edit = claim["edit"]
    n = len(prompt)
    positions = [edit["position"]] if "position" in edit else edit["positions"]
    token = behaviour["behaviour"]
    old = [int(prompt[j]) for j in positions]
    transcript = [
        float(edit["kind"] == "substitute"), float(edit["kind"] == "delete"), float(edit["kind"] == "swap"),
        math.log1p(n - 1 - max(positions)), float(token in old), float(edit.get("token") == token),
        float(any(o in np.array(behaviour["samples"]) for o in old)), behaviour["rate"], math.log(n),
    ]
    if behaviour["id"] not in cache:
        cache[behaviour["id"]] = dependencies(model, prompt, token)
    grad, direct, logp = cache[behaviour["id"]]
    W = model.wte.detach().cpu().numpy()
    if edit["kind"] == "substitute":
        first = grad[positions[0]] @ (W[edit["token"]] - W[old[0]])
    elif edit["kind"] == "delete":
        first = -grad[positions[0]] @ W[old[0]]
    else:
        i, j = positions
        first = grad[i] @ (W[prompt[j]] - W[prompt[i]]) + grad[j] @ (W[prompt[i]] - W[prompt[j]])
    p = math.exp(logp)
    moved = abs(p * (math.exp(max(min(first, 20.0), -20.0)) - 1.0))
    explanation = [first, abs(first), min(moved, 1.0), float(sum(abs(direct[j]) for j in positions)), float(np.abs(direct).sum())]
    return transcript, transcript + explanation, (grad, direct, logp, first, moved)


def statement(behaviour, claim, computed, W, decode):
    """The dependencies as an explanation's text: the input tokens the behaviour token's
    probability depends on most (first-order effect of removing each), the largest head-routed
    direct effects, and the first-order change this edit makes, all from the original prompt."""
    grad, direct, logp, first, moved = computed
    prompt = np.array(behaviour["prompt"])
    removal = np.array([-grad[j] @ W[prompt[j]] for j in range(len(prompt))])
    lines = [f"Explanation of the original forward pass (read-only, from the model's weights on the original prompt). "
             f"P(next token = {behaviour['behaviour_text']!r}) = {math.exp(logp):.2f}. It depends most on these input tokens "
             f"(first-order change in its log-probability if the token were removed):"]
    for j in np.argsort(-np.abs(removal))[:6]:
        lines.append(f"  position {j} {decode([prompt[j]])!r}: {removal[j]:+.2f} nats")
    lines.append("Largest direct effects on its logit through attention heads (attention weight times the head's write):")
    for j in np.argsort(-np.abs(direct))[:3]:
        lines.append(f"  position {j} {decode([prompt[j]])!r}: {direct[j]:+.2f}")
    lines.append(f"For this edit, the first-order change in that log-probability is {first:+.2f} nats "
                 f"(a change of about {min(moved, 1.0) * 100:.0f} points in the next-token probability).")
    return "\n".join(lines)


def logistic(X, y, ridge=1.0):
    """Ridge-penalised logistic regression by Newton's method (an intercept column added)."""
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    penalty = ridge * np.eye(X.shape[1])
    penalty[-1, -1] = 0.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + penalty, X.T @ (p - y) + penalty @ w)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return lambda Z: 1.0 / (1.0 + np.exp(-np.hstack([Z, np.ones((len(Z), 1))]) @ w))


def predict(root=ROOT):
    """Both predictors fitted on the dev claims, their P(true) for every claim."""
    record = json.load(open(root / "chive_vpd" / "claims.json"))
    behaviours = {b["id"]: b for b in record["behaviours"]}
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TARGET_DIR / "tokenizer.json"))
    decode = lambda ids: tokenizer.decode([int(t) for t in ids])
    model = load_target(device())
    W = model.wte.detach().cpu().numpy()
    cache = {}
    rows = [features(model, behaviours[c["behaviour"]], c, cache) for c in record["claims"]]
    texts = {c["id"]: statement(behaviours[c["behaviour"]], c, r[2], W, decode) for c, r in zip(record["claims"], rows)}
    json.dump(texts, open(root / "chive_vpd" / "explanations.json", "w"))
    dev = [i for i, c in enumerate(record["claims"]) if c["split"] == "dev"]
    labels = np.array([float(record["claims"][i]["label"]) for i in dev])
    for name, column in (("transcript", 0), ("dependencies", 1)):
        X = np.array([r[column] for r in rows])
        mean, std = X[dev].mean(0), X[dev].std(0) + 1e-12
        fit = logistic((X[dev] - mean) / std, labels)
        p = fit((X - mean) / std)
        json.dump({c["id"]: float(q) for c, q in zip(record["claims"], p)}, open(root / "chive_vpd" / f"predictions.{name}.json", "w"))
    print("wrote predictions for", len(rows), "claims")


LLM_TASK = """You predict how a language model's sampled continuations change when its prompt is edited, without running the model.
Below is a transcript: the prompt the model read, the 30 continuations it sampled (temperature 1, 3 tokens each), and a behaviour with its measured rate.
Then a claim about one edit of the prompt. Reply with only one number: the probability that the claim is true."""


def ask(text, model="haiku"):
    import subprocess

    for _ in range(3):
        out = subprocess.run(["claude", "-p", "--model", model, text], capture_output=True, text=True, timeout=300)
        try:
            return min(max(float(out.stdout.strip().split()[-1]), 0.0), 1.0)
        except (ValueError, IndexError):
            continue
    return 0.5


def llm(root=ROOT):
    """CHIVE's predictor: a language model reading the transcript and the claim (and, in the second
    condition, the read-only explanation), answering P(true), for every held-out claim."""
    from concurrent.futures import ThreadPoolExecutor

    record = json.load(open(root / "chive_vpd" / "claims.json"))
    texts = json.load(open(root / "chive_vpd" / "explanations.json"))
    behaviours = {b["id"]: b for b in record["behaviours"]}
    claims = [c for c in record["claims"] if c["split"] == "held_out"]

    def transcript(c):
        b = behaviours[c["behaviour"]]
        samples = "\n".join(f"  {t!r}" for t in b["sample_texts"])
        return (f"{LLM_TASK}\n\nPROMPT (the model continues right after it):\n{b['prompt_text'][-1500:]}\n\nSAMPLES:\n{samples}\n\n"
                f"BEHAVIOUR: the continuation contains {b['behaviour_text']!r}; rate {b['rate']:.2f} over the 30 samples.\n\n"
                f"EDIT: {c['edit_text']}\nCLAIM: {c['claim']}")

    for name, with_explanation in (("llm_transcript", False), ("llm_dependencies", True)):
        prompts = [transcript(c) + (f"\n\n{texts[c['id']]}" if with_explanation else "") + "\n\nProbability the claim is true:" for c in claims]
        with ThreadPoolExecutor(8) as pool:
            answers = list(pool.map(ask, prompts))
        json.dump({c["id"]: p for c, p in zip(claims, answers)}, open(root / "chive_vpd" / f"predictions.{name}.json", "w"))
        print(name, "answered", len(answers), flush=True)


if __name__ == "__main__":
    {"build": build, "predict": predict, "llm": llm}[sys.argv[1]](*(Path(a) for a in sys.argv[2:]))
