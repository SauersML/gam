"""Behavior pattern_ab.alternate (vpd4l): Alternation: two words alternate several times; the next word
continues the alternation; the counterfactual starts the alternation with the other word.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

h0_4 = node(L[0].head[4])
mlp0 = node(
    PD.vpd[0].c_fc[69, 229, 328, 668, 697, 805, 889, 985, 1032, 1185, 1275, 1328, 1418, 1531, 1650,
    1724, 1759, 1778, 1957, 2094, 2471, 2598, 2634, 2822, 2857, 3013, 3038], PD.vpd[0].down_proj[42,
    172, 216, 245, 862, 920, 929, 946, 1060, 1090, 1385, 1433, 1434, 1792, 1847, 1925, 2190, 2248,
    2423, 2505, 2530, 2585, 2606, 2619, 2712, 2726, 2818, 3082, 3086, 3196, 3200, 3209, 3257, 3260,
    3382, 3473, 3523]
)
h1_1 = node(L[1].head[1])
mlp1 = node(
    PD.vpd[1].c_fc[236, 261, 687, 738, 743, 748, 799, 1608, 2103, 2179, 2493, 2662, 2698],
    PD.vpd[1].down_proj[26, 846, 926]
)
h2_3 = node(L[2].head[3])
h2_4 = node(L[2].head[4])
h3_5 = node(L[3].head[5])
mlp3 = node(
    PD.vpd[3].c_fc[11, 153, 161, 191, 258, 271, 312, 363, 398, 430, 476, 500, 589, 821, 872, 891,
    903, 906, 945, 1020, 1060, 1076, 1149, 1195, 1259, 1318, 1339, 1366, 1394, 1414, 1446, 1449,
    1454, 1460, 1511, 1673, 1697, 1699, 1720, 1755, 1767, 1807, 1828, 1876, 1890, 1897, 1924, 2057,
    2069, 2099, 2122, 2304, 2356, 2364, 2393, 2429, 2453, 2548, 2562, 2628, 2661, 2719, 2745, 2842,
    2883, 2913, 3013, 3048, 3049], PD.vpd[3].down_proj[61, 180, 219, 222, 230, 300, 377, 428, 470,
    477, 502, 650, 703, 898, 986, 989, 1034, 1126, 1127, 1312, 1352, 1402, 1408, 1428, 1438, 1439,
    1448, 1477, 1513, 1528, 1543, 1569, 1622, 1800, 1807, 2170, 2225, 2337, 2420, 2477, 2534, 2549,
    2614, 2653, 2712, 2713, 2786, 2787, 2891, 2955, 3021, 3116, 3233, 3279, 3379, 3414, 3467, 3480,
    3500]
)

edges(
    embed >> h0_4,
    embed >> mlp0,
    embed >> h1_1,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_5,
    embed >> mlp3,
    h0_4 >> mlp0,
    h0_4 >> h1_1,
    h0_4 >> mlp1,
    h0_4 >> h2_3,
    h0_4 >> h2_4,
    h0_4 >> h3_5,
    h0_4 >> mlp3,
    mlp0 >> h1_1,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h1_1 >> mlp1,
    h1_1 >> h2_3,
    h1_1 >> h2_4,
    h1_1 >> h3_5,
    h1_1 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_5 >> mlp3,
    h0_4 >> logits,
    mlp0 >> logits,
    h1_1 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)
