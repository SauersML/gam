#!/usr/bin/env python3
"""A frozen oracle report on a blind organism as the organism benchmark's report (#2951).

The oracle names native components and edits in the server's terms (operators blocks.L.qH, kG, vG,
oH, c_fc, gate_proj, down_proj); the benchmark names Hugging Face tensors with rows and columns
(model.layers.L.self_attn.q_proj.weight, ...). This file maps one onto the other: a head's query map
is rows [H d_h, (H + 1) d_h) of q_proj, a key or value map rows of k_proj or v_proj, a head's output
map columns of o_proj, c_fc is up_proj; and it writes the report's edit, W(alpha) - W of every edited
operator as the server computes it, as an 'add' safetensors file in float32.

usage: MPD_MEM_GIB=1 venv python organism_report.py RUN_DIR   (the run's server must be up: investigate.py calls it)
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

HF_MLP = {"c_fc": "mlp.up_proj", "gate_proj": "mlp.gate_proj", "down_proj": "mlp.down_proj"}


def tensor_block(operator, head_width):
    """(HF tensor name, rows slice or None, cols slice or None) of a server operator."""
    _, layer, part = operator.split(".", 2)
    prefix = f"model.layers.{layer}."
    if part in HF_MLP:
        return prefix + HF_MLP[part] + ".weight", None, None
    kind, index = part[0], int(part[1:])
    block = slice(index * head_width, (index + 1) * head_width)
    if kind in "qkv":
        return prefix + f"self_attn.{kind}_proj.weight", block, None
    if kind == "o":
        return prefix + "self_attn.o_proj.weight", None, block
    raise ValueError(f"no tensor for operator {operator}")


def location_of(component, head_width):
    """The benchmark's location entries ({tensor, rows?, cols?}) a server component edits."""
    kind = component["kind"]
    if kind == "head":
        name = f"blocks.{component['layer']}.o{component['head']}"
    elif kind == "neuron":
        tensor, _, _ = tensor_block(f"blocks.{component['layer']}.down_proj", head_width)
        return [{"tensor": tensor, "cols": [component["index"]]}]
    else:
        name = component["name"]
    tensor, rows, cols = tensor_block(name, head_width)
    entry = {"tensor": tensor}
    offset_rows = range(rows.start, rows.stop) if rows else None
    offset_cols = range(cols.start, cols.stop) if cols else None
    if kind == "rows":
        offset_rows = [(rows.start if rows else 0) + r for r in component["rows"]]
    if kind == "columns":
        offset_cols = [(cols.start if cols else 0) + c for c in component["columns"]]
    if offset_rows is not None:
        entry["rows"] = list(offset_rows)
    if offset_cols is not None:
        entry["cols"] = list(offset_cols)
    return [entry]


def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    text = json.dumps(header).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for b in blobs:
            f.write(b)


def edit_file(request, model, edits, out_dir, shapes, head_width):
    """Write the report's edit as an 'add' file; returns its path, or None for no edit."""
    if not edits:
        return None
    reply = request({"op": "delta", "model": model, "edits": edits, "directory": str(out_dir / "delta")})
    tensors = {}
    for entry in reply["operators"]:
        rows_n, cols_n = entry["shape"]
        delta = np.fromfile(entry["file"], dtype="<f8").reshape(rows_n, cols_n)
        tensor, rows, cols = tensor_block(entry["operator"], head_width)
        full = tensors.setdefault(tensor, np.zeros(shapes[tensor], dtype=np.float64))
        full[rows or slice(None), cols or slice(None)] += delta
    path = out_dir / "edit.safetensors"
    write_safetensors(path, tensors)
    return path


def tensor_shapes(checkpoint):
    raw = open(Path(checkpoint) / "model.safetensors", "rb").read(8)
    (n,) = struct.unpack("<Q", raw)
    with open(Path(checkpoint) / "model.safetensors", "rb") as f:
        f.seek(8)
        header = json.loads(f.read(n))
    return {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}


def convert(run, request):
    frozen = json.load(open(run / "report.frozen.json"))
    report = frozen["report"]
    task = json.load(open(run / "task.json"))
    info = request({"op": "info"})["models"]["updated"]
    head_width = info["head_width"]
    shapes = tensor_shapes(Path(task["models"]["updated"]).expanduser())
    location = []
    for c in report["components"]:
        try:
            location.extend(location_of(c["component"], head_width))
        except (KeyError, ValueError) as e:
            print(f"component without a tensor location: {c['component']} ({e})", file=sys.stderr)
    out = {
        "schema": "mpd.organism-report/1",
        "organism": task["organism"],
        "rule": report["rule"],
        "predictor": {"python": report["predictor_python"]},
        "location": location,
        "key_terms": report.get("key_terms", []),
        "notes": "\n\n".join(f"{k.replace('_', ' ').upper()}: {report[k]}" for k in ("information_used", "mechanism", "predicted_effects")) +
                 f"\n\nEDIT, EXPECTED: {report['edit']['expected']}\n\nFrozen oracle report sha256 {frozen['sha256']}",
    }
    path = edit_file(request, "updated", report["edit"]["edits"], run, shapes, head_width)
    if path:
        out["edit"] = {"file": str(path.resolve()), "mode": "add"}
    target = run / "organism_report.json"
    target.write_text(json.dumps(out, indent=1))
    return target


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate import Client

    run = Path(sys.argv[1])
    session = json.load(open(run / "session.json"))
    print(convert(run, Client(session["socket"])))
