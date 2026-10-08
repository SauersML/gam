"""Behavior plural.regular (vpd4l): Plural of a regular noun after a count above one. Phrasings: 'I
have one {X} and you have two' | 'one {X}, two' | 'There is one {X} here and three'.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[69, 327, 328, 332, 363, 379, 497, 521, 591, 614, 652, 655, 664, 805, 861, 1098,
    1225, 1275, 1277, 1307, 1333, 1443, 1564, 1663, 1759, 1778, 1782, 1798, 1908, 1933, 1961, 2001,
    2107, 2164, 2214, 2218, 2271, 2379, 2402, 2445, 2471, 2518, 2554, 2576, 2583, 2632, 2634, 2653,
    2761, 2822, 2857, 2887, 2901, 2930, 2959, 3013, 3044, 3057, 3065], PD.vpd[0].down_proj[4, 42,
    105, 137, 146, 210, 328, 363, 406, 473, 488, 584, 606, 785, 862, 920, 942, 1025, 1196, 1284,
    1363, 1377, 1385, 1391, 1411, 1423, 1464, 1467, 1493, 1515, 1587, 1592, 1597, 1792, 1838, 1884,
    1925, 1991, 2041, 2044, 2190, 2196, 2275, 2443, 2505, 2522, 2530, 2585, 2606, 2718, 2818, 2875,
    3012, 3079, 3186, 3196, 3201, 3209, 3239, 3267, 3289, 3341, 3382, 3393, 3438, 3445, 3455, 3473,
    3523]
)
mlp1 = node(
    PD.vpd[1].c_fc[261, 477, 1526, 2103, 2179, 2610, 2828, 2922, 2992], PD.vpd[1].down_proj[515,
    806, 1709, 2497, 2740, 3396, 3572]
)
h2_1 = node(L[2].head[1])
h2_2 = node(L[2].head[2])
h2_3 = node(L[2].head[3])
h2_4 = node(L[2].head[4])
h3_0 = node(L[3].head[0])
h3_1 = node(L[3].head[1])
h3_4 = node(L[3].head[4])
h3_5 = node(L[3].head[5])
mlp3 = node(
    PD.vpd[3].c_fc[4, 87, 94, 122, 148, 153, 161, 178, 200, 220, 237, 312, 315, 452, 603, 802, 805,
    821, 876, 903, 906, 945, 1014, 1019, 1023, 1036, 1076, 1121, 1145, 1149, 1287, 1306, 1318, 1405,
    1446, 1449, 1480, 1481, 1511, 1565, 1636, 1673, 1688, 1767, 1844, 1869, 1878, 1890, 1897, 2069,
    2122, 2289, 2313, 2356, 2364, 2387, 2472, 2615, 2661, 2727, 2745, 2831, 2992, 3035, 3049],
    PD.vpd[3].down_proj[147, 180, 222, 230, 258, 360, 384, 477, 502, 706, 723, 735, 848, 934, 989,
    994, 1009, 1070, 1126, 1127, 1168, 1176, 1179, 1312, 1327, 1341, 1365, 1408, 1418, 1439, 1448,
    1451, 1453, 1513, 1583, 1685, 1694, 1732, 1790, 2093, 2248, 2381, 2520, 2534, 2549, 2583, 2598,
    2631, 2786, 2787, 2862, 2881, 3003, 3055, 3067, 3077, 3116, 3148, 3228, 3351, 3377, 3414, 3445]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_0,
    embed >> h3_1,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_0,
    mlp0 >> h3_1,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_0,
    mlp1 >> h3_1,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_1 >> h3_0,
    h2_1 >> h3_1,
    h2_1 >> h3_4,
    h2_1 >> h3_5,
    h2_1 >> mlp3,
    h2_2 >> h3_0,
    h2_2 >> h3_1,
    h2_2 >> h3_4,
    h2_2 >> h3_5,
    h2_2 >> mlp3,
    h2_3 >> h3_0,
    h2_3 >> h3_1,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_0,
    h2_4 >> h3_1,
    h2_4 >> h3_4,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_0 >> mlp3,
    h3_1 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_0 >> logits,
    h3_1 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)
