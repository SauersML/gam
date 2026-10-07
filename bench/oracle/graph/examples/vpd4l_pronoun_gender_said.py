"""Behavior pronoun_gender.said (vpd4l; M's top token is right on 98% of the targets): Gendered
pronoun: the next word is the pronoun for the named person; the counterfactual swaps the name's
gender.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.00 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 0.99, L3.H4 0.83, layer 2 MLP 0.45, layer 3 MLP 0.32, layer 1 MLP 0.22 bits.
The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.99 bits; these 128 together leave 0.28 of 1.00)
mlp0 = node(L[0].mlp[
    6, 35, 49, 136, 158, 180, 190, 247, 257, 278, 292, 354, 366, 393, 405, 420, 430, 477, 493, 503,
    504, 510, 614, 661, 683, 709, 710, 726, 749, 751, 759, 792, 826, 857, 897, 978, 993, 1005, 1052,
    1062, 1074, 1101, 1112, 1117, 1139, 1191, 1221, 1222, 1258, 1291, 1304, 1428, 1442, 1464, 1470,
    1482, 1489, 1491, 1493, 1525, 1548, 1550, 1588, 1594, 1624, 1633, 1634, 1648, 1693, 1726, 1759,
    1761, 1778, 1779, 1781, 1785, 1786, 1817, 1902, 1958, 1983, 1987, 1989, 2055, 2083, 2116, 2145,
    2159, 2160, 2200, 2212, 2265, 2277, 2321, 2332, 2352, 2372, 2397, 2411, 2433, 2435, 2436, 2516,
    2544, 2545, 2616, 2628, 2630, 2633, 2635, 2646, 2673, 2687, 2703, 2729, 2759, 2767, 2825, 2831,
    2886, 2900, 2919, 2921, 2940, 2970, 2993, 2995, 3010
])
# layer 1's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.22 bits; these 128 together leave 0.73 of 1.00)
mlp1 = node(L[1].mlp[
    114, 148, 189, 204, 222, 228, 229, 256, 267, 340, 357, 372, 411, 414, 421, 422, 425, 436, 437,
    448, 452, 499, 500, 546, 583, 620, 665, 672, 681, 686, 730, 763, 818, 824, 852, 880, 926, 934,
    954, 964, 987, 1012, 1017, 1088, 1114, 1162, 1171, 1221, 1234, 1254, 1265, 1290, 1321, 1327,
    1339, 1396, 1402, 1427, 1450, 1458, 1467, 1514, 1519, 1561, 1570, 1588, 1596, 1597, 1616, 1659,
    1660, 1671, 1674, 1676, 1718, 1723, 1729, 1750, 1766, 1782, 1792, 1821, 1978, 2005, 2023, 2078,
    2124, 2128, 2131, 2148, 2166, 2193, 2212, 2257, 2264, 2286, 2300, 2306, 2326, 2327, 2367, 2389,
    2440, 2486, 2487, 2493, 2529, 2534, 2552, 2555, 2573, 2584, 2647, 2711, 2714, 2737, 2806, 2864,
    2890, 2914, 2939, 2944, 2960, 2968, 2991, 2995, 3035, 3045
])
# layer 2's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.45 bits; these 128 together leave 0.45 of 1.00)
mlp2 = node(L[2].mlp[
    17, 72, 74, 104, 108, 130, 164, 170, 218, 286, 301, 319, 330, 331, 354, 362, 365, 408, 459, 467,
    504, 520, 529, 559, 563, 582, 595, 600, 641, 642, 646, 660, 674, 761, 769, 861, 916, 942, 983,
    986, 1003, 1006, 1010, 1089, 1115, 1135, 1150, 1157, 1192, 1215, 1228, 1245, 1255, 1271, 1273,
    1291, 1300, 1308, 1357, 1364, 1380, 1386, 1429, 1435, 1444, 1451, 1461, 1481, 1543, 1591, 1596,
    1603, 1607, 1641, 1701, 1724, 1728, 1755, 1763, 1774, 1793, 1796, 1817, 1843, 1868, 1871, 1898,
    1935, 2003, 2023, 2049, 2063, 2065, 2125, 2128, 2136, 2173, 2174, 2233, 2259, 2278, 2301, 2340,
    2350, 2356, 2362, 2367, 2374, 2486, 2504, 2510, 2547, 2680, 2705, 2717, 2743, 2759, 2801, 2814,
    2872, 2897, 2904, 2956, 2972, 2973, 3012, 3025, 3044
])
h3_4 = node(L[3].head[4])  # recovers 0.83 bits
# layer 3's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.32 bits; these 64 together leave 0.59 of 1.00)
mlp3 = node(L[3].mlp[
    34, 141, 160, 187, 253, 291, 387, 497, 572, 695, 708, 750, 765, 894, 1026, 1098, 1125, 1179,
    1210, 1217, 1227, 1235, 1275, 1279, 1351, 1354, 1358, 1428, 1466, 1497, 1542, 1553, 1554, 1569,
    1581, 1692, 1704, 1742, 1756, 1799, 1923, 2047, 2118, 2126, 2174, 2219, 2292, 2310, 2335, 2379,
    2385, 2451, 2484, 2497, 2574, 2626, 2761, 2845, 2882, 2899, 2970, 2991, 2997, 3060
])

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_4,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    mlp1 >> mlp2,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)

