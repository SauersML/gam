"""Behavior alphabet.lower (vpd4l): Alphabet succession: a run of lowercase letters; the next letter
continues it.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[127, 436, 646, 668, 697, 698, 1040, 1085, 1275, 1307, 1349, 1376, 1418, 1477,
    1724, 1747, 1778, 2094, 2310, 2450, 2464, 3038, 3065], PD.vpd[0].down_proj[200, 245, 351, 418,
    606, 897, 942, 1025, 1433, 1525, 1567, 1584, 1597, 1847, 2079, 2094, 2134, 2172, 2190, 2248,
    2264, 2419, 2423, 2505, 2656, 2712, 2726, 2771, 3012, 3059, 3082, 3086, 3191, 3196, 3200, 3257,
    3341, 3401, 3413, 3473, 3523]
)
mlp1 = node(
    PD.vpd[1].c_fc[261, 345, 477, 570, 1096, 1200, 1578, 1615, 1689, 1728, 1839, 2022, 2086, 2179,
    2828, 2922, 2992, 3059, 3067], PD.vpd[1].down_proj[405, 547, 830, 950, 1108, 1333, 1595, 1775,
    2245, 2267, 2893, 3357, 3444]
)
h2_2 = node(L[2].head[2])
h2_4 = node(L[2].head[4])
mlp2 = node(
    PD.vpd[2].c_fc[9, 67, 87, 411, 472, 483, 503, 568, 578, 656, 711, 747, 814, 815, 818, 853, 962,
    1030, 1108, 1119, 1169, 1191, 1222, 1417, 1523, 1626, 1914, 2415, 2482, 2488, 2510, 2536, 2794,
    2886, 2916, 2962], PD.vpd[2].down_proj[13, 65, 166, 419, 523, 615, 673, 827, 857, 1013, 1532,
    1586, 1775, 2144, 2154, 2159, 2317, 2400, 2501, 2550, 2554, 3160, 3232, 3279, 3343, 3403, 3441,
    3493]
)
h3_0 = node(L[3].head[0])
mlp3 = node(
    PD.vpd[3].c_fc[72, 153, 200, 220, 271, 363, 403, 603, 843, 876, 881, 896, 1145, 1192, 1280,
    1314, 1382, 1481, 1565, 1717, 1767, 1940, 1995, 2051, 2313, 2364, 2393, 2424, 2574, 2716, 2719,
    2831, 2970, 3013], PD.vpd[3].down_proj[130, 220, 254, 350, 502, 650, 723, 765, 1040, 1127, 1216,
    1293, 1311, 1327, 1402, 2026, 2181, 2420, 2614, 2665, 2787, 2933, 3021, 3048, 3183, 3328, 3351,
    3467, 3480, 3532]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_2,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_0,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_2,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> mlp3,
    mlp1 >> h2_2,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> mlp3,
    h2_2 >> mlp2,
    h2_2 >> h3_0,
    h2_2 >> mlp3,
    h2_4 >> mlp2,
    h2_4 >> h3_0,
    h2_4 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_2 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    mlp3 >> logits,
)
