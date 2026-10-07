"""Checks of the prediction-training stack (#2951): the held-out piece split, the question syntax read back
by parity.py, the answer reader and KL of eval_kl.py, the LoRA adapter and its PEFT export, and (when the
checkpoints are on this machine) the executors: Qwen3's written-out decoder against Hugging Face's forward,
identity interventions, an MLP removal against a hooked Hugging Face run, and a VPD subcomponent's scale
against the native weight edit W + (a - 1) u v^T on vpd4l.

  python -m pytest bench/oracle/graph/predict/test_predict.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_kl  # noqa: E402
import generate as G  # noqa: E402
import parity  # noqa: E402
import sft  # noqa: E402

QWEN = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots").glob("*"))
VPD = Path.home() / "mpd-data/oracle/vpd/uv.safetensors"


def test_held_out_units_fixed_and_one_in_ten():
    a = G.held_out_units("neuron", 3, 3072)
    assert (a == G.held_out_units("neuron", 3, 3072)).all()
    assert 0.07 < a.mean() < 0.13
    assert not (a == G.held_out_units("neuron", 4, 3072)).all()
    assert not (a == G.held_out_units("tc", 3, 3072)).all()


@pytest.mark.parametrize("piece", [("head", 3, 5), ("neurons", 7, (1, 20, 300)), ("mlp", 2), ("attn", 9)])
def test_piece_text_read_back(piece):
    back = parity.parse_piece(G.piece_text(piece))
    assert back[0] == piece[0] and back[1] == piece[1]
    if piece[0] == "head":
        assert back[2] == piece[2]
    if piece[0] == "neurons":
        assert tuple(back[2]) == piece[2]


def test_piece_text_mech_spelling():
    assert G.piece_text(("mlp", 2)) == "L[2].mlp[:]"
    assert G.piece_text(("attn", 2)) == "L[2].head[:]"
    assert G.piece_text(("tc", 14, (5, 9))) == "PD.tc[14][5, 9]"
    assert G.piece_text(("vpd", 1, "c_fc", (7,))) == "PD.vpd[1].c_fc[7]"


def test_read_answer_and_kl():
    strs, p = [" the", " a", " an", " this", " that"], [0.5, 0.2, 0.1, 0.05, 0.05]
    exact = eval_kl.read_answer('" the" 0.50 | " a" 0.20 | " an" 0.10 | " this" 0.05 | " that" 0.05 | other 0.10')
    assert eval_kl.kl_bits(strs, p, exact) < 1e-9
    missing = eval_kl.read_answer('" a" 0.70 | other 0.30')
    assert eval_kl.kl_bits(strs, p, missing) > 5.0  # an omitted top token costs log2 of the spread
    assert eval_kl.read_answer("not an answer") is None
    assert eval_kl.kl_bits(strs, p, None) > eval_kl.kl_bits(strs, p, missing)


def test_lora_starts_at_the_base_model_and_exports_to_peft(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].q_proj = torch.nn.Linear(8, 6, bias=False)
    x = torch.randn(3, 8)
    before = model.layers[0].q_proj(x)
    adapters = sft.wrap(model, 4, 8.0)
    assert torch.equal(model.layers[0].q_proj(x), before)  # B = 0
    adapters["layers.0.q_proj"].B.data.normal_()
    after = model.layers[0].q_proj(x)
    a = adapters["layers.0.q_proj"]
    assert torch.allclose(after, before + (x @ a.A.T) @ a.B.T * 2.0, atol=1e-6)
    from safetensors.torch import load_file, save_file

    save_file({"layers.0.q_proj.A": a.A.detach(), "layers.0.q_proj.B": a.B.detach()}, str(tmp_path / "ad.safetensors"))
    sft.save_peft(tmp_path / "ad.safetensors", tmp_path / "peft", "base", 4, 8.0)
    peft = load_file(str(tmp_path / "peft/adapter_model.safetensors"))
    assert set(peft) == {"base_model.model.layers.0.q_proj.lora_A.weight", "base_model.model.layers.0.q_proj.lora_B.weight"}
    config = json.loads((tmp_path / "peft/adapter_config.json").read_text())
    assert config["r"] == 4 and config["lora_alpha"] == 8.0 and config["modules_to_save"] is None


def test_changed_marks_moved_answers():
    assert sft.changed({"numbers": {"kl_bits": 0.2}})
    assert not sft.changed({"numbers": {"kl_bits": [0.01, 0.05, 0.0, 0.02]}})
    assert sft.changed({"numbers": {"tokens_unchanged": 3, "edited_ids": [1] * 8}})


@pytest.fixture(scope="module")
def qwen():
    if not QWEN:
        pytest.skip("Qwen3-0.6B not in the Hugging Face cache")
    return G.Qwen3(str(QWEN[0]), torch.device("cpu"))


def tokens():
    return torch.tensor([[785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9856, 374]])


def test_qwen3_decoder_matches_hugging_face(qwen):
    t = tokens()
    ours = qwen.log_probs(qwen.forward(t))
    with torch.no_grad():
        ref = torch.log_softmax(qwen.model(t).logits[:, -1].double(), dim=-1)
    assert (ours - ref).abs().max().item() < 1e-4


def test_identity_interventions_change_nothing(qwen):
    t = torch.cat([tokens(), tokens().flip(1)])
    rec = {}
    clean = qwen.log_probs(qwen.forward(t, None, rec))
    iv = qwen.new(2)
    G.apply_scale(iv, 0, ("head", 3, 2), 1.0)
    G.apply_scale(iv, 1, ("neurons", 5, (1, 2, 3)), 1.0)
    assert (qwen.log_probs(qwen.forward(t, iv, None, rec)) - clean).abs().max().item() < 1e-9
    # A swap from the text itself is the clean run.
    iv = qwen.new(2)
    mask = torch.zeros(2, qwen.L, qwen.H, dtype=torch.bool)
    mask[:, 4, 7] = True
    iv.head_swap = (mask, rec["z_last"])
    assert (qwen.log_probs(qwen.forward(t, iv, None, rec)) - clean).abs().max().item() < 1e-9


def test_mlp_removal_matches_hooked_hugging_face(qwen):
    t = tokens()
    iv = qwen.new(1)
    G.apply_scale(iv, 0, ("mlp", 6), 0.0)
    ours = qwen.log_probs(qwen.forward(t, iv))
    handle = qwen.layers[6].mlp.register_forward_hook(lambda mod, inp, out: torch.zeros_like(out))
    try:
        with torch.no_grad():
            ref = torch.log_softmax(qwen.model(t).logits[:, -1].double(), dim=-1)
    finally:
        handle.remove()
    assert (ours - ref).abs().max().item() < 1e-4


def test_vpd_subcomponent_scale_is_the_weight_edit():
    if not VPD.exists():
        pytest.skip("vpd4l files absent")
    import vpd4l

    m = vpd4l.Vpd4l(torch.device("cpu"))
    t = torch.tensor([[50, 2000, 300, 4000, 500, 6000, 700, 8000]])
    iv = m.new(1)
    idx = (3, 17)
    G.apply_scale(iv, 0, ("vpd", 1, "c_fc", idx), 2.0)
    ours = m.log_probs(m.forward(t, iv))
    U, V = m.parts[(1, "c_fc")]
    site = m.t.site("h.1.mlp.c_fc")
    W = site.W.clone()
    try:
        site.W += (2.0 - 1.0) * (V[:, list(idx)] @ U[list(idx)]).T  # W + (a - 1) sum_i u_i v_i^T, W [out, in]
        ref = m.log_probs(m.forward(t))
    finally:
        site.W.copy_(W)
    assert (ours - ref).abs().max().item() < 1e-4
    assert (ours - m.log_probs(m.forward(t))).abs().max().item() > 1e-6  # the edit does something


def test_concat_matches_separate_forwards(qwen):
    t = torch.cat([tokens(), tokens().flip(1)])
    a, b = qwen.new(2), qwen.new(2)
    G.apply_scale(a, 0, ("head", 10, 3), 0.0)
    G.apply_scale(a, 1, ("neurons", 4, (7, 8)), 2.0)
    G.apply_scale(b, 0, ("mlp", 12), 0.5)
    G.apply_scale(b, 1, ("attn", 2), 0.0)
    joint = qwen.log_probs(qwen.forward(t.repeat(2, 1), G.Interventions.concat([a, b])))
    apart = torch.cat([qwen.log_probs(qwen.forward(t, a)), qwen.log_probs(qwen.forward(t, b))])
    # Equal up to float32 rounding: a batch of 4 rows may block the matmuls differently from 2 (Linux CI: 1.7e-5).
    assert (joint - apart).abs().max().item() < 1e-4


def test_attend_weights_reproduce_the_head_read(qwen):
    """The recorded attention weights of a head at the last position, applied to its values, give its read z."""
    t = tokens()
    rec = {"attend": {0: (5, 9)}}
    qwen.forward(t, None, rec)
    w = rec["attend_weights"][0]
    assert abs(w.sum().item() - 1.0) < 1e-5
    captured = {}
    layer = qwen.layers[5]
    handle = layer.input_layernorm.register_forward_hook(lambda mod, inp, out: captured.setdefault("x", out))
    try:
        qwen.forward(t, None, {})
    finally:
        handle.remove()
    pos = torch.arange(t.shape[1])[None]
    cos, sin = qwen.inner.rotary_emb(captured["x"], pos)
    _, _, v = qwen.qkv(layer, captured["x"], cos, sin)
    z_from_weights = w @ v[0, 9 // (qwen.H // qwen.KV)]
    assert (z_from_weights - rec["z_last"][0, 5, 9]).abs().max().item() < 1e-4


def test_draws_respect_the_piece_split(qwen):
    rng = np.random.default_rng(0)
    for split in ("train", "heldout"):
        d = G.Draw(qwen, rng, split)
        for _ in range(50):
            p = d.uniform()
            if p[0] == "head":
                assert G.held_out_units("head", p[1], qwen.H)[p[2]] == (split == "heldout")
            elif p[0] == "neurons":
                assert all(G.held_out_units("neuron", p[1], qwen.Fn)[i] == (split == "heldout") for i in p[2])
            else:
                assert split == "train"  # whole blocks only in training shards


def test_vpd_subcomponents_are_the_engines():
    """PD.vpd[l].site[i] means the same subcomponent here (uv.safetensors) as in the Rust engine's export and
    in mech's shapes: identical factors, same order, same counts."""
    export = Path.home() / "mpd-data/engine/vpd4l_decomposition"
    if not (VPD.exists() and export.exists()):
        pytest.skip("vpd4l files absent")
    from safetensors.numpy import load_file

    uv = load_file(str(VPD))
    files = json.loads((export / "export.json").read_text())["files"]
    shapes = json.loads((HERE.parent / "shapes.json").read_text())["vpd4l"]["views"]["vpd"]
    for name, meta in files.items():
        if name.endswith((".U", ".V")):
            a = np.fromfile(export / f"{name}.f64", dtype="<f8").reshape(meta["shape"])
            assert np.array_equal(a, uv[name].astype(np.float64)), name
            if name.endswith(".U"):
                layer, site = int(name.split(".")[1]), name.split(".")[3]
                assert shapes[layer][site] == a.shape[0], name


def test_cut_with_the_text_as_its_own_counterfactual_changes_nothing(qwen):
    """A cut delivers the writer's value on x'; with x' = x the run is the clean run."""
    t = torch.cat([tokens(), tokens().flip(1)])
    clean = qwen.log_probs(qwen.forward(t))
    iv = qwen.new(2)
    iv.cuts = {0: ("head", 3, 5, "head", 6, 2, "key"), 1: ("mlp", 4, -1, "logits", 28, -1, "")}
    rec_cf = {"write_requests": {r: c[:3] for r, c in iv.cuts.items()}}
    qwen.forward(t, None, rec_cf)
    assert (qwen.log_probs(qwen.forward(t, iv, None, rec_cf)) - clean).abs().max().item() < 1e-5
    other = t.flip(0)  # a different x' moves the prediction
    rec_cf = {"write_requests": {r: c[:3] for r, c in iv.cuts.items()}}
    qwen.forward(other, None, rec_cf)
    assert (qwen.log_probs(qwen.forward(t, iv, None, rec_cf)) - clean).abs().max().item() > 1e-4


def test_eval_kl_reads_the_targets_own_tokens():
    """vpd4l and Qwen3 tokenizers differ: eval_kl takes the target's decoded tokens and vocabulary from the record."""
    q = {"model": "vpd4l", "vocab": 50277, "numbers": {"edited": {"ids": [11, 12], "p": [0.6, 0.3], "tokens": [" the", " a"]}}}
    assert eval_kl.target_tokens(q, "edited", tok=None) == [" the", " a"]
    with pytest.raises(ValueError):
        eval_kl.target_tokens({"model": "vpd4l", "numbers": {"edited": {"ids": [11], "p": [1.0]}}}, "edited", tok=None)
    answer = eval_kl.read_answer('" the" 0.60 | " a" 0.30 | other 0.10')
    assert eval_kl.kl_bits([" the", " a"], [0.6, 0.3], answer, 50277) < 1e-9
