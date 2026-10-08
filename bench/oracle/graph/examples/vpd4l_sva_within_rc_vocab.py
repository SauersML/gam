"""Behavior sva.within_rc (vpd4l): Subject-verb agreement after the verb inside a relative clause
agreeing with that clause's subject; the counterfactual flips the subject's number.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(PD.vpd[0].c_fc[925, 1091, 2084], PD.vpd[0].down_proj[137, 279, 622, 862, 3126])
mlp1 = node(PD.vpd[1].c_fc[345, 687, 1073, 2202, 2662, 2828], PD.vpd[1].down_proj[1217, 3001])
mlp2 = node(
    PD.vpd[2].c_fc[323, 770, 774, 1096, 1291, 1605, 1858, 2039, 2267, 2415, 2631, 2757],
    PD.vpd[2].down_proj[252, 308, 666, 699, 857, 945, 1268, 1496, 1553, 1581, 1820, 2868, 3045,
    3049, 3076, 3219, 3240, 3330, 3404, 3493]
)
h3_0 = node(L[3].head[0])
h3_3 = node(L[3].head[3])
mlp3 = node(
    PD.vpd[3].c_fc[68, 179, 237, 403, 500, 626, 896, 995, 1032, 1622, 1720, 1777, 2313, 2423, 2548,
    2868, 2883, 2913, 2950], PD.vpd[3].down_proj[152, 367, 934, 1341, 1501, 1833, 1912, 2244, 2549,
    2740, 3277, 3291, 3532]
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
