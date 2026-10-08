"""Behavior sva.simple (vpd4l): Subject-verb agreement after a determiner and noun; the counterfactual
flips the subject's number.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[123, 747, 1032, 1167, 1185, 1225, 1531, 1604, 1636, 1663, 1759, 1908, 1933, 2084,
    2336, 2645, 2822, 3013], PD.vpd[0].down_proj[295, 622, 657, 862, 920, 1060, 1090, 1777, 2275,
    2367, 3171, 3455, 3491, 3494]
)
mlp1 = node(
    PD.vpd[1].c_fc[157, 345, 370, 402, 570, 687, 723, 766, 1282, 1728, 1786, 1876, 1914, 2179, 2255,
    2765, 2828, 3008], PD.vpd[1].down_proj[515, 594, 612, 1202, 1217, 1403, 1472, 1521, 2245, 2621,
    3034, 3220, 3465, 3478]
)
mlp2 = node(
    PD.vpd[2].c_fc[87, 323, 445, 522, 770, 774, 1096, 1291, 1315, 2039, 2089, 2472, 2631, 2757],
    PD.vpd[2].down_proj[666, 708, 945, 1268, 1496, 1581, 1798, 1820, 2237, 3045, 3049, 3076, 3129,
    3240, 3279, 3439, 3493, 3503]
)
h3_0 = node(L[3].head[0])
h3_3 = node(L[3].head[3])
mlp3 = node(
    PD.vpd[3].c_fc[68, 179, 228, 237, 386, 403, 467, 583, 626, 645, 884, 891, 903, 909, 995, 1010,
    1032, 1035, 1192, 1243, 1444, 1454, 1481, 1673, 1681, 1720, 1777, 1844, 1983, 2057, 2387, 2423,
    2548, 2554, 2570, 2883, 2913], PD.vpd[3].down_proj[61, 152, 341, 468, 873, 1086, 1176, 1232,
    1254, 1341, 1362, 1501, 1531, 1833, 1912, 1943, 1984, 2093, 2244, 2740, 2787, 2905, 3275, 3277,
    3291, 3334, 3377]
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
