"""Behavior pronoun_gender.thinks (vpd4l; M's top token is right on 0% of the targets): Gendered
pronoun: the next word is the pronoun for the named person; the counterfactual swaps the name's
gender.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.09 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 1.09, L3.H4 1.00, layer 2 MLP 0.57, L3.H2 0.39, layer 3 MLP 0.35, layer 1
MLP 0.20 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 1.09 bits; these 128 together leave 0.22 of 1.09)
mlp0 = node(L[0].mlp[
    49, 158, 177, 180, 190, 247, 265, 278, 292, 366, 393, 405, 477, 503, 510, 614, 615, 661, 726,
    729, 730, 737, 749, 751, 759, 800, 802, 888, 894, 897, 918, 944, 978, 995, 1005, 1029, 1052,
    1062, 1069, 1074, 1096, 1101, 1117, 1128, 1130, 1139, 1169, 1191, 1201, 1217, 1221, 1222, 1223,
    1255, 1258, 1291, 1330, 1405, 1428, 1444, 1470, 1482, 1483, 1489, 1493, 1525, 1548, 1588, 1617,
    1633, 1648, 1693, 1744, 1779, 1785, 1786, 1805, 1854, 1902, 1905, 1906, 1958, 1987, 1989, 2055,
    2083, 2127, 2145, 2159, 2200, 2256, 2265, 2380, 2397, 2411, 2415, 2433, 2436, 2468, 2516, 2545,
    2616, 2628, 2630, 2633, 2635, 2640, 2646, 2673, 2687, 2729, 2735, 2759, 2767, 2810, 2825, 2831,
    2859, 2896, 2921, 2922, 2940, 2993, 2995, 2999, 3010, 3031, 3039
])
# layer 1's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.20 bits; these 128 together leave 0.73 of 1.09)
mlp1 = node(L[1].mlp[
    0, 55, 69, 114, 137, 189, 204, 228, 229, 322, 338, 340, 375, 414, 416, 422, 437, 452, 464, 470,
    523, 525, 546, 569, 583, 595, 686, 732, 747, 763, 774, 818, 827, 852, 880, 934, 943, 954, 965,
    989, 1012, 1054, 1058, 1153, 1193, 1250, 1254, 1265, 1290, 1306, 1347, 1367, 1396, 1398, 1402,
    1427, 1437, 1467, 1514, 1519, 1529, 1570, 1588, 1597, 1600, 1631, 1653, 1656, 1659, 1665, 1676,
    1718, 1722, 1729, 1733, 1750, 1780, 1798, 1821, 1851, 1899, 1956, 1978, 2005, 2014, 2023, 2039,
    2047, 2124, 2128, 2167, 2185, 2269, 2286, 2300, 2307, 2327, 2332, 2342, 2482, 2486, 2487, 2493,
    2534, 2538, 2552, 2608, 2647, 2711, 2714, 2737, 2741, 2846, 2847, 2849, 2864, 2877, 2885, 2890,
    2944, 2956, 2960, 2964, 2968, 3004, 3035, 3045, 3047
])
# layer 2's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.57 bits; these 128 together leave 0.40 of 1.09)
mlp2 = node(L[2].mlp[
    17, 51, 67, 72, 108, 127, 130, 158, 164, 170, 177, 218, 264, 265, 278, 319, 365, 382, 390, 408,
    459, 499, 504, 518, 543, 549, 563, 582, 595, 600, 606, 642, 660, 674, 690, 700, 718, 736, 750,
    774, 834, 865, 867, 916, 936, 954, 983, 1006, 1010, 1014, 1046, 1089, 1135, 1157, 1215, 1234,
    1255, 1271, 1272, 1273, 1300, 1334, 1357, 1364, 1380, 1386, 1407, 1410, 1429, 1435, 1440, 1444,
    1481, 1543, 1548, 1589, 1596, 1603, 1641, 1661, 1701, 1728, 1730, 1763, 1774, 1793, 1817, 1843,
    1868, 1880, 1885, 1898, 1946, 1976, 2003, 2049, 2063, 2065, 2081, 2086, 2173, 2174, 2220, 2231,
    2233, 2293, 2301, 2350, 2362, 2367, 2510, 2677, 2680, 2696, 2701, 2705, 2717, 2738, 2758, 2759,
    2801, 2846, 2904, 2911, 2917, 2920, 2931, 3044
])
h3_2 = node(L[3].head[2])  # recovers 0.39 bits
h3_4 = node(L[3].head[4])  # recovers 1.00 bits
# layer 3's MLP: the 32 neurons that recover the most when patched alone (the whole MLP recovers 0.35 bits; these 32 together leave 0.50 of 1.09)
mlp3 = node(L[3].mlp[
    141, 227, 360, 491, 616, 708, 894, 1026, 1179, 1210, 1282, 1428, 1523, 1704, 1756, 1799, 1910,
    1923, 1967, 2170, 2174, 2385, 2451, 2603, 2667, 2762, 2782, 2825, 2899, 2923, 2970, 3060
])

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_2,
    embed >> h3_4,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_2,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    mlp1 >> mlp2,
    mlp1 >> h3_2,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    mlp2 >> h3_2,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_2 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)

