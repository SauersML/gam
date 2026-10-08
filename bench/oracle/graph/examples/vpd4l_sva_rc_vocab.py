"""Behavior sva.rc (vpd4l): Subject-verb agreement after a noun followed by a relative clause with a
distractor noun; the counterfactual flips the subject's number.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(PD.vpd[0].c_fc[123, 1604, 1663, 2084], PD.vpd[0].down_proj[622, 862, 1090, 3455])
mlp1 = node(PD.vpd[1].c_fc[766, 3008], PD.vpd[1].down_proj[1077, 2735])
mlp2 = node(
    PD.vpd[2].c_fc[323, 770, 774, 1096, 1119, 1291, 1605, 1834, 2039, 2757],
    PD.vpd[2].down_proj[666, 1496, 2237, 2527, 3045, 3076]
)
h3_3 = node(L[3].head[3])

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_3,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_3,
    mlp1 >> mlp2,
    mlp1 >> h3_3,
    mlp2 >> h3_3,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_3 >> logits,
)
