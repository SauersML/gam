"""Behavior list_copy.first (vpd4l): List indexing: given a list of four words, the first word; the
counterfactual swaps the first word with another, so the same words appear.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[53, 69, 127, 229, 306, 328, 332, 363, 436, 497, 582, 614, 646, 655, 668, 684,
    697, 749, 750, 805, 848, 861, 889, 895, 918, 985, 1085, 1275, 1277, 1307, 1418, 1491, 1512,
    1531, 1564, 1579, 1663, 1673, 1724, 1759, 1778, 1782, 1838, 2040, 2054, 2087, 2094, 2098, 2107,
    2118, 2266, 2271, 2315, 2376, 2471, 2632, 2645, 2653, 2662, 2683, 2857, 3013, 3045, 3065],
    PD.vpd[0].down_proj[4, 105, 123, 172, 245, 351, 363, 384, 400, 406, 430, 473, 488, 584, 648,
    750, 785, 941, 942, 946, 1025, 1139, 1363, 1385, 1434, 1464, 1467, 1515, 1525, 1567, 1587, 1597,
    1792, 2044, 2094, 2172, 2196, 2248, 2367, 2423, 2424, 2554, 2567, 2585, 2606, 2687, 2696, 2726,
    2771, 2860, 3012, 3086, 3166, 3191, 3200, 3201, 3209, 3256, 3289, 3335, 3393, 3455, 3473, 3523]
)
h2_3 = node(L[2].head[3])
h3_4 = node(L[3].head[4])
h3_5 = node(L[3].head[5])
mlp3 = node(
    PD.vpd[3].c_fc[94, 148, 153, 162, 312, 315, 325, 326, 507, 603, 724, 805, 906, 945, 1014, 1060,
    1076, 1097, 1121, 1204, 1259, 1306, 1318, 1339, 1394, 1405, 1428, 1446, 1449, 1511, 1565, 1636,
    1688, 1717, 1825, 1869, 2069, 2364, 2387, 2393, 2472, 2562, 2574, 2661, 2699, 2831, 3049],
    PD.vpd[3].down_proj[349, 801, 898, 1070, 1327, 1451, 1695, 2225, 2712, 2862, 2955, 3116, 3279,
    3445, 3467, 3480, 3565]
)

edges(
    embed >> mlp0,
    embed >> h2_3,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h2_3,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    h2_3 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)
