"""Behavior syllogism.all_are (vpd4l): Syllogism: 'All X are Y. N is one of the X. So N is a' is
followed by Y; the counterfactual changes the category.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[229, 327, 363, 379, 684, 861, 1032, 1148, 1304, 1418, 1443, 1512, 1579, 1759,
    1879, 1933, 2042, 2054, 2118, 2271, 2402, 2445, 2583, 2598, 2632, 2653, 2683, 2857, 2901, 2959,
    2976, 3045], PD.vpd[0].down_proj[42, 105, 279, 406, 584, 929, 1025, 1284, 1385, 1493, 1515,
    1587, 1628, 1925, 2044, 2094, 2196, 2275, 2505, 2530, 2619, 2726, 2818, 2875, 3196, 3213, 3253,
    3289, 3335, 3393, 3455, 3473]
)
h2_3 = node(L[2].head[3])
mlp2 = node(
    PD.vpd[2].c_fc[128, 265, 483, 770, 1191, 1995, 2151], PD.vpd[2].down_proj[1058, 1134, 1553,
    2400, 2602, 2898, 3420, 3497, 3503]
)
h3_4 = node(L[3].head[4])
mlp3 = node(
    PD.vpd[3].c_fc[87, 148, 315, 326, 452, 906, 1014, 1121, 1259, 1511, 1636, 1688, 1869, 1878,
    1897, 2313, 2387, 2472, 2831], PD.vpd[3].down_proj[360, 884, 1070, 1179, 1242, 1451, 1492, 1694,
    1732, 2652, 2862, 3233, 3272]
)

edges(
    embed >> mlp0,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_4,
    embed >> mlp3,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_4,
    h2_3 >> mlp3,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)
