"""Behavior roman_numerals.sequence (vpd4l): Roman numeral succession: a run of Roman numerals; the
next numeral continues it.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[69, 120, 176, 229, 243, 330, 379, 497, 504, 515, 527, 589, 646, 652, 664, 684,
    779, 787, 843, 914, 925, 985, 1015, 1035, 1040, 1131, 1268, 1277, 1307, 1357, 1376, 1378, 1477,
    1554, 1564, 1604, 1629, 1636, 1650, 1663, 1683, 1724, 1747, 1758, 1797, 1822, 1853, 1891, 2054,
    2061, 2094, 2114, 2254, 2266, 2285, 2352, 2382, 2402, 2464, 2471, 2482, 2549, 2557, 2576, 2598,
    2683, 2887, 2961, 3044, 3057], PD.vpd[0].down_proj[4, 16, 51, 95, 245, 331, 351, 473, 614, 750,
    786, 920, 946, 1025, 1060, 1139, 1148, 1149, 1196, 1363, 1464, 1515, 1525, 1597, 1807, 1847,
    1919, 2044, 2094, 2134, 2275, 2280, 2354, 2367, 2423, 2522, 2530, 2567, 2732, 2817, 2875, 3012,
    3082, 3133, 3184, 3196, 3200, 3201, 3209, 3210, 3253, 3256, 3275, 3341, 3382, 3401, 3445, 3473]
)
mlp1 = node(
    PD.vpd[1].c_fc[458, 477, 1526, 1786, 1911, 2017, 2022, 2103, 2179, 2298, 2327, 2420, 2584, 2645,
    2698, 2828, 2992, 3067], PD.vpd[1].down_proj[26, 321, 515, 830, 836, 980, 1108, 1217, 2022,
    2740, 3001, 3016, 3220, 3478]
)
h2_2 = node(L[2].head[2])
h2_3 = node(L[2].head[3])
h2_4 = node(L[2].head[4])
h3_0 = node(L[3].head[0])
h3_5 = node(L[3].head[5])
mlp3 = node(
    PD.vpd[3].c_fc[258, 312, 315, 363, 507, 599, 843, 945, 978, 1016, 1035, 1382, 1565, 1636, 1661,
    1765, 1767, 1876, 2069, 2304, 2364, 2387, 2716, 2719, 2971, 3047], PD.vpd[3].down_proj[187, 201,
    216, 280, 423, 477, 544, 570, 706, 723, 799, 986, 1049, 1402, 1528, 1673, 1675, 1732, 1802,
    1803, 1807, 1984, 2486, 2544, 2545, 2549, 2670, 2710, 2763, 2778, 2787, 2799, 2880, 2909, 3018,
    3390, 3467, 3554]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_0,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_0,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_0,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
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
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_0 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)
