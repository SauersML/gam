"""Behavior word_successor.ordinal (vpd4l): Ordinal succession: a run of ordinal words; the next word
continues the order.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[69, 127, 268, 330, 363, 381, 497, 504, 521, 779, 843, 1098, 1131, 1500, 1747,
    1758, 1822, 1838, 1929, 1933, 2061, 2173, 2266, 2315, 2445, 2468, 2565, 2861, 3006, 3045],
    PD.vpd[0].down_proj[4, 123, 146, 331, 342, 677, 925, 946, 1003, 1139, 1149, 1376, 1377, 1515,
    1525, 1745, 2001, 2280, 2530, 2967, 2994, 3086, 3094, 3126, 3128, 3133, 3191, 3213, 3256, 3266,
    3275, 3393, 3455, 3506]
)
mlp1 = node(
    PD.vpd[1].c_fc[145, 282, 477, 687, 851, 1108, 2022, 2739, 2828, 3008, 3067],
    PD.vpd[1].down_proj[100, 321, 515, 1217, 1855]
)
h2_1 = node(L[2].head[1])
h2_2 = node(L[2].head[2])
h2_3 = node(L[2].head[3])
h2_4 = node(L[2].head[4])
h3_0 = node(L[3].head[0])
h3_5 = node(L[3].head[5])
mlp3 = node(
    PD.vpd[3].c_fc[148, 153, 271, 312, 315, 598, 843, 906, 945, 978, 1019, 1053, 1187, 1195, 1306,
    1365, 1428, 1481, 1517, 1565, 1657, 1767, 1851, 1861, 1876, 2193, 2387, 2490, 2661, 3048],
    PD.vpd[3].down_proj[280, 300, 461, 510, 544, 570, 623, 986, 1075, 1086, 1248, 1275, 1395, 1402,
    1439, 1445, 1448, 1453, 2147, 2431, 2470, 2486, 2614, 2665, 2763, 3018, 3021, 3055, 3193, 3421,
    3467, 3480, 3484, 3554]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_0,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_0,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_0,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_1 >> h3_0,
    h2_1 >> h3_5,
    h2_1 >> mlp3,
    h2_2 >> h3_0,
    h2_2 >> h3_5,
    h2_2 >> mlp3,
    h2_3 >> h3_0,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_0,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_0 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_0 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)
