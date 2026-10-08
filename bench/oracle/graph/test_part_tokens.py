"""Checks of part_tokens.py (#2951): tokens round-trip to addresses, relabeling the parts changes no part's
embedding or logit, and the part tokens enter a Qwen3 tokenizer as single contiguous ids.

  python -m pytest bench/oracle/graph/test_part_tokens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import part_tokens as PT  # noqa: E402

ADDRESSES = ["PD.vpd[2].v_proj[559]", "PD.vpd[0].c_fc[7]", "PD.vpd[3].down_proj[1200]", "L[2].head[6]", "L[5].mlp[:]", "L[0].head[:]",
             "PD.vpd[1].q_proj[3]", "PD.vpd[1].o_proj[44]", "PD.vpd[2].k_proj[9]", "L[11].head[2]"]


def registry(seed=0):
    g = torch.Generator().manual_seed(seed)
    kinds = [PT.kind_of(a) for a in ADDRESSES]
    dims = {"vpd.v": 6, "vpd.fc": 7, "vpd.down": 5, "vpd.q": 6, "vpd.o": 4, "vpd.k": 6, "head": 9, "mlp": 9, "attn": 9}
    feats = {k: torch.randn(kinds.count(k), dims[k], generator=g) for k in dict.fromkeys(kinds)}
    return PT.Registry(ADDRESSES, feats)


@pytest.mark.parametrize("address", ADDRESSES)
def test_token_round_trip(address):
    assert PT.address_of(PT.token_of(address)) == address


def test_examples():
    assert PT.token_of("PD.vpd[2].v_proj[559]") == "<p:2.v.559>"
    assert PT.token_of("L[2].head[6]") == "<p:2.h.6>"
    assert PT.token_of("L[5].mlp[:]") == "<p:5.m>"


def test_relabeling_changes_nothing():
    reg = registry()
    torch.manual_seed(0)
    a = PT.PartTokens(reg, hidden=16, emb_rms=1.0, base_vocab=100)
    order = [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]
    perm = reg.permuted(order)
    b = PT.PartTokens(perm, hidden=16, emb_rms=1.0, base_vocab=100)
    b.load_state_dict(a.state_dict())
    for which in ("in", "out"):
        ra, rb = a.rows(which), b.rows(which)
        for j, i in enumerate(order):  # part i of reg is part j of perm
            assert torch.allclose(ra[i], rb[j], atol=1e-6)


def test_rewrite_uses_tokens():
    reg = registry()
    text = "scale(L[2].head[6], 0) and cut(node(L[5].mlp[:]) >> logits) but not L[2].head[61] or L[3].head[6]"
    out = reg.rewrite(text)
    assert "scale(<p:2.h.6>, 0)" in out and "node(<p:5.m>)" in out
    assert "L[2].head[61]" in out and "L[3].head[6]" in out


def test_tokenizer_ids_are_contiguous():
    snaps = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots").glob("*"))
    if not snaps:
        pytest.skip("Qwen3 tokenizer not cached")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snaps[0]))
    reg = registry()
    base = len(tok)
    m = PT.PartTokens(reg, hidden=8, emb_rms=1.0, base_vocab=base)
    m.add_to_tokenizer(tok)
    ids = tok("scale(<p:2.h.6>, 0)", add_special_tokens=False)["input_ids"]
    assert base + reg.index["L[2].head[6]"] in ids
    assert tok.decode([base + reg.index["PD.vpd[2].v_proj[559]"]]) == "<p:2.v.559>"


def test_sites_and_two_level_choice():
    reg = registry()
    m = PT.PartTokens(reg, hidden=16, emb_rms=1.0, base_vocab=100)
    assert reg.sites == sorted(set(reg.sites)) and len(reg.site) == len(ADDRESSES)
    h = torch.randn(16)
    site = reg.site[reg.index["PD.vpd[2].v_proj[559]"]]
    idx, logits = m.part_logits_in(h, site)
    assert reg.index["PD.vpd[2].v_proj[559]"] in idx and logits.shape == (len(idx),)
    assert m.site_logits(h).shape == (len(reg.sites),)
    assert m.input_rows().shape == (len(ADDRESSES), 16) and m.tokens() == reg.tokens
