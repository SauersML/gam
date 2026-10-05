"""Tokenize the hypotheses' prompts (hypotheses.py) and a held-out reference (vpd4l_clean4096 rows
1024..1032) into the read-out driver's relp mode inputs: prompts.json, index.json, reference.json.
Then: mpd_library_readout_2951 relp EXPORT prompts.json out; ... reference.json ref; evaluate.py."""
import hashlib
import json

import numpy as np
from tokenizers import Tokenizer

from hypotheses import H

tok = Tokenizer.from_file("/Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json")
E = "/Users/user/mpd-data/engine/vpd4l_clean4096"
sha = hashlib.sha256(open(E + "/export.json", "rb").read()).hexdigest()
prompts, index = [], []
for name, desc, fire, quiet in H:
    for kind, texts in (("fire", fire), ("quiet", quiet)):
        for text in texts:
            ids = tok.encode(text[1:]).ids[:1] if text.startswith("§") else tok.encode(text).ids
            prompts.append({"tokens": ids, "metric": "predicted"})
            index.append({"function": name, "kind": kind, "text": text, "last": tok.decode(ids[-1:]), "length": len(ids)})
json.dump({"export_sha256": sha, "numeric_bytes": 4000000000, "tile_rows": 1024, "prompts": prompts}, open("prompts.json", "w"))
json.dump(index, open("index.json", "w"), ensure_ascii=False)
record = json.load(open(E + "/export.json"))
tokens = np.memmap(E + "/tokens.f64", dtype="<f8", mode="r", shape=tuple(record["files"]["tokens"]["shape"]))
reference = [{"tokens": [int(x) for x in tokens[r, :512]], "metric": "predicted"} for r in range(1024, 1032)]
json.dump({"export_sha256": sha, "numeric_bytes": 4000000000, "tile_rows": 1024, "prompts": reference}, open("reference.json", "w"))
