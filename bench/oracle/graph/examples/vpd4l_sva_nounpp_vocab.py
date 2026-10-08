"""Behavior sva.nounpp (vpd4l): Subject-verb agreement after a noun followed by a prepositional phrase
with a distractor noun; the counterfactual flips the subject's number.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(PD.vpd[0].c_fc[53, 123, 1604, 1663, 2084], PD.vpd[0].down_proj[622, 862, 3455])
mlp1 = node(
    PD.vpd[1].c_fc[477, 766, 1003, 2179, 2610, 2828, 3008], PD.vpd[1].down_proj[44, 594, 612, 1108,
    1562, 1744, 2022, 2621, 3220]
)
mlp2 = node(
    PD.vpd[2].c_fc[323, 770, 774, 1096, 1291, 1315, 2039, 2415, 2757], PD.vpd[2].down_proj[666, 945,
    1496, 2223, 3045, 3439, 3562]
)
h3_0 = node(L[3].head[0])
h3_3 = node(L[3].head[3])
mlp3 = node(
    PD.vpd[3].c_fc[179, 220, 237, 312, 403, 903, 1777, 1844, 2628], PD.vpd[3].down_proj[152, 468,
    1254, 1501, 2244, 2740, 3291]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_3,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_3,
    mlp1 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_3,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_3 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)
