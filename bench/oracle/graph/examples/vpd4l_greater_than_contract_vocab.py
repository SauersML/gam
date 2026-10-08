"""Behavior greater_than.contract (vpd4l): Greater-than: the end year starts with the same century, so
its last two digits must exceed the start year's.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[69, 330, 332, 504, 749, 779, 805, 964, 985, 1010, 1015, 1098, 1349, 1500, 1544,
    1579, 1701, 1822, 2087, 2121, 2218, 2220, 2315, 2645, 2653, 2657, 2674, 3006, 3038, 3045],
    PD.vpd[0].down_proj[105, 148, 168, 169, 293, 351, 379, 579, 648, 925, 946, 979, 1139, 1173,
    1391, 1433, 1567, 1623, 1686, 1745, 1792, 1847, 2233, 2275, 2280, 2423, 2530, 2585, 2778, 2812,
    3086, 3200, 3257, 3523]
)
mlp1 = node(
    PD.vpd[1].c_fc[353, 811, 833, 1393, 1689, 1786, 2022, 2070, 2101, 2103, 2116, 2259, 2284, 2610,
    2650, 2765, 2922, 3067], PD.vpd[1].down_proj[34, 269, 298, 321, 672, 738, 830, 1077, 2267, 2621,
    3016, 3270, 3572, 3579]
)
h2_3 = node(L[2].head[3])
h2_4 = node(L[2].head[4])
mlp2 = node(
    PD.vpd[2].c_fc[2, 64, 87, 90, 335, 393, 411, 445, 483, 762, 774, 915, 935, 1108, 1222, 1273,
    1293, 1326, 1417, 1436, 1573, 1613, 1748, 1853, 1902, 1984, 2128, 2151, 2417, 2482, 2601, 2884,
    2978, 3015, 3056], PD.vpd[2].down_proj[65, 544, 708, 773, 778, 827, 857, 945, 1088, 1171, 1573,
    1774, 1775, 1982, 2178, 2317, 2376, 2532, 2550, 2708, 2854, 3141, 3160, 3271, 3325, 3374, 3376,
    3492, 3562]
)
h3_5 = node(L[3].head[5])
mlp3 = node(
    PD.vpd[3].c_fc[153, 312, 316, 325, 363, 497, 507, 509, 598, 945, 995, 1016, 1097, 1149, 1187,
    1195, 1204, 1318, 1394, 1423, 1446, 1454, 1493, 1755, 1828, 2028, 2091, 2158, 2162, 2193, 2313,
    2364, 2393, 2562, 2574, 2633, 2650, 2667, 2716, 2831, 3048, 3049], PD.vpd[3].down_proj[83, 300,
    324, 350, 477, 510, 577, 623, 760, 1216, 1352, 1395, 1402, 1415, 1445, 1680, 1772, 2004, 2220,
    2799, 3390, 3421]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> mlp2,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)
