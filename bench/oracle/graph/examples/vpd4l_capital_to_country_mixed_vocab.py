"""Behavior capital_to_country.mixed (vpd4l): Factual recall in reverse: the country whose capital is
the named city. Phrasings: '{X} is the capital city of' | 'Q: Which country has {X} as its
capital?\nA:'.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[53, 243, 332, 363, 545, 614, 646, 684, 698, 750, 852, 889, 925, 1035, 1040, 1085,
    1091, 1275, 1323, 1349, 1435, 1443, 1500, 1531, 1554, 1564, 1579, 1663, 1683, 1724, 1747, 1759,
    1778, 1817, 1879, 1891, 1908, 1933, 1957, 2040, 2042, 2087, 2114, 2214, 2235, 2266, 2554, 2565,
    2583, 2632, 2634, 2645, 2653, 2683, 2761, 2887, 2910, 2959, 3013, 3038, 3045],
    PD.vpd[0].down_proj[4, 16, 168, 279, 326, 351, 406, 499, 648, 750, 785, 911, 929, 1025, 1060,
    1269, 1377, 1385, 1390, 1433, 1464, 1467, 1515, 1587, 1597, 2041, 2044, 2065, 2151, 2172, 2190,
    2275, 2313, 2344, 2367, 2530, 2567, 2647, 2687, 2696, 2712, 2718, 2771, 2860, 2875, 2903, 2967,
    2994, 3062, 3079, 3094, 3126, 3166, 3196, 3200, 3201, 3209, 3253, 3256, 3257, 3260, 3267, 3289,
    3335, 3341, 3473, 3523]
)
mlp1 = node(
    PD.vpd[1].c_fc[98, 329, 345, 477, 1578, 1728, 2103, 2179, 2493, 2828], PD.vpd[1].down_proj[806,
    1217, 1926, 3220, 3396, 3478]
)
h2_3 = node(L[2].head[3])
mlp2 = node(
    PD.vpd[2].c_fc[146, 318, 323, 483, 656, 726, 747, 759, 820, 853, 915, 1191, 1227, 1290, 1315,
    1340, 1417, 1523, 1573, 1748, 1866, 1888, 1914, 1984, 2267, 2415, 2482, 2510, 2588, 2735, 3004],
    PD.vpd[2].down_proj[65, 308, 379, 770, 791, 945, 1038, 1058, 1134, 1197, 1268, 1287, 1559, 1566,
    1586, 1763, 1775, 1948, 2341, 2457, 2886, 2926, 2973, 3040, 3160, 3219, 3271, 3279, 3359, 3383,
    3404, 3546, 3581]
)
h3_0 = node(L[3].head[0])
h3_4 = node(L[3].head[4])
h3_5 = node(L[3].head[5])
mlp3 = node(
    PD.vpd[3].c_fc[72, 191, 200, 250, 312, 439, 476, 603, 722, 843, 980, 1035, 1036, 1076, 1145,
    1149, 1259, 1320, 1339, 1449, 1470, 1481, 1512, 1666, 1686, 1807, 1851, 1869, 1876, 2133, 2143,
    2239, 2304, 2387, 2661, 2699, 2844, 2920], PD.vpd[3].down_proj[258, 349, 430, 898, 1028, 1078,
    1203, 1365, 1380, 1458, 1497, 1780, 1782, 1846, 1984, 2225, 2325, 2520, 2582, 2712, 2787, 2880,
    3116, 3301, 3377, 3502]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_0,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_4,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)
