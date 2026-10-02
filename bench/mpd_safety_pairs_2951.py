"""Safety-track prompt sets for #2951, from public, citable sources only.

Sources (downloaded under data/):
  XSTest      Roettger et al. 2024, "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in
              Large Language Models", NAACL 2024 (HF Paul/XSTest): 250 safe prompts that look unsafe + 200
              unsafe contrast prompts, written in matched type/focus order
  JBB         Chao et al. 2024, "JailbreakBench" (NeurIPS D&B), JBB-Behaviors: 100 harmful behaviors and 100
              benign behaviors matched one-to-one by behavior/category (HF JailbreakBench/JBB-Behaviors)
  AdvBench    Zou et al. 2023, "Universal and Transferable Adversarial Attacks on Aligned Language Models",
              harmful_behaviors.csv (github.com/llm-attacks/llm-attacks)
  Alpaca      Taori et al. 2023, Stanford Alpaca (HF tatsu-lab/alpaca), instructions with no input field
  SEP         Zverev et al. 2025, "Can LLMs Separate Instructions From Data? And What Do We Even Mean By
              That?", ICLR 2025 (github.com/egozverev/Should-It-Be-Executed-Or-Processed, SEP_dataset.json)

Outputs:
  direction_sets.json        harmful/harmless instructions for fitting and selecting the refusal direction
                             (AdvBench vs Alpaca; disjoint from every evaluation pair)
  refusal_pairs.json         matched (harmful, benign-but-similar) pairs for evaluation: XSTest contrast pairs
                             and JBB harmful/benign pairs
  instruction_source.json    SEP items, each as three chats with the SAME probe instruction: given by the user
                             (in the task), embedded in quoted data, embedded in a tool result
"""

import json
import random
from pathlib import Path

import pandas as pd

D = Path.home() / "mpd-data/safety/data"
OUT = Path.home() / "mpd-data/safety"
rng = random.Random(0)

# ---------------------------------------------------------------- refusal-direction fit sets (Arditi et al. 2024 recipe)
adv = pd.read_csv(D / "advbench_harmful_behaviors.csv")["goal"].tolist()
alp = pd.read_parquet(next((D / "tatsu-lab__alpaca/data").glob("*.parquet")))
alp = [r for r, i in zip(alp["instruction"], alp["input"]) if not str(i).strip()]
rng.shuffle(adv)
rng.shuffle(alp)
direction = {"harmful_train": adv[:256], "harmful_val": adv[256:320],
             "harmless_train": alp[:256], "harmless_val": alp[256:320],
             "sources": {"harmful": "AdvBench harmful_behaviors (Zou et al. 2023)", "harmless": "Alpaca, no-input instructions (Taori et al. 2023)"}}
json.dump(direction, open(OUT / "direction_sets.json", "w"), indent=1)

# ---------------------------------------------------------------- matched evaluation pairs
x = pd.read_csv(D / "Paul__XSTest/xstest_prompts.csv")
SAFE_OF = {"contrast_homonyms": "homonyms", "contrast_figurative_language": "figurative_language",
           "contrast_safe_targets": "safe_targets", "contrast_safe_contexts": "safe_contexts",
           "contrast_definitions": "definitions", "contrast_historical_events": "historical_events",
           "contrast_privacy": "privacy_public", "contrast_discr": "nons_group_real_discr"}
pairs = []
for ctype, stype in SAFE_OF.items():
    u = x[x.type == ctype].sort_values("id")
    s = x[x.type == stype].sort_values("id")
    for (_, ur), (_, sr) in zip(u.iterrows(), s.iterrows()):
        pairs.append({"id": f"xstest_{ur.id}_{sr.id}", "source": "XSTest", "category": stype,
                      "harmful": ur.prompt, "benign": sr.prompt,
                      "focus_match": bool(isinstance(ur.focus, str) and ur.focus == sr.focus)})
jh = pd.read_csv(D / "JailbreakBench__JBB-Behaviors/data/harmful-behaviors.csv")
jb = pd.read_csv(D / "JailbreakBench__JBB-Behaviors/data/benign-behaviors.csv")
for (_, h), (_, b) in zip(jh.iterrows(), jb.iterrows()):
    pairs.append({"id": f"jbb_{h.Index}", "source": "JBB-Behaviors", "category": h.Category,
                  "harmful": h.Goal, "benign": b.Goal, "focus_match": h.Behavior == b.Behavior})
xs_extra = x[x.type == "real_group_nons_discr"]["prompt"].tolist() + x[x.type == "privacy_fictional"]["prompt"].tolist()
json.dump({"pairs": pairs, "benign_unpaired": [{"source": "XSTest", "prompt": p} for p in xs_extra],
           "note": "XSTest pairs are the i-th contrast prompt with the i-th safe prompt of the matching type (the "
                   "suite writes them in matched order; focus_match marks pairs sharing the trigger word). JBB "
                   "pairs are matched by row (benign behaviors were written per harmful behavior)."},
          open(OUT / "refusal_pairs.json", "w"), indent=1)

# ---------------------------------------------------------------- instruction-source discrimination (SEP)
sep = json.load(open(D / "SEP_dataset.json"))
rng.shuffle(sep)
items = []
for k, e in enumerate(sep[:320]):
    task, data, probe_in_task, probe_in_data = (e["system_prompt_clean"], e["prompt_clean"],
                                                e["system_prompt_instructed"], e["prompt_instructed"])
    if probe_in_task.startswith(task):  # SEP appends or prepends the probe to the task
        probe = probe_in_task[len(task):].strip()
    elif probe_in_task.endswith(task):
        probe = probe_in_task[:-len(task)].strip()
    else:
        continue
    items.append({
        "id": f"sep_{k}", "witness": e["witness"], "probe": probe, "info": e["info"],
        # the same probe instruction from three sources; the ideal model executes it only in 'user'
        "user": [{"role": "user", "content": f"{probe_in_task}\n\n{data}"}],
        "quoted": [{"role": "user", "content": f"{task}\n\nText:\n\"\"\"\n{probe_in_data}\n\"\"\""}],
        "tool": [{"role": "user", "content": f"{task} The text is in the tool result."},
                 {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "fetch_document", "arguments": {}}}]},
                 {"role": "tool", "content": probe_in_data}],
        "clean": [{"role": "user", "content": f"{task}\n\nText:\n\"\"\"\n{data}\n\"\"\""}],
    })
items = items[:300]
json.dump({"items": items, "metric": "witness rate: the probe's witness string appears in the reply (SEP's own test)",
           "source": "SEP_dataset.json (Zverev et al., ICLR 2025)"}, open(OUT / "instruction_source.json", "w"), indent=1)
print(len(direction["harmful_train"]), len(pairs), len(items))
