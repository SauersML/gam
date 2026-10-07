"""Behavior past_tense.irregular (vpd4l; M's top token is right on 27% of the targets): Past tense of
an irregular verb. Phrasings: 'Every day I {X}. Yesterday I' | 'Today they {X}. Last week they' | 'I
usually {X} in the morning, but last night I'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 0.95 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 0.94, L3.H4 0.70, layer 3 MLP 0.43, L2.H4 0.39, L2.H3 0.33, layer 1 MLP 0.17
bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 256 neurons that recover the most when patched alone (the whole MLP recovers 0.94 bits; these 256 together leave 0.49 of 0.95)
mlp0 = node(L[0].mlp[
    5, 6, 9, 18, 58, 75, 89, 128, 139, 143, 144, 155, 169, 179, 201, 215, 219, 239, 241, 243, 247,
    259, 264, 266, 278, 288, 321, 325, 328, 333, 338, 358, 363, 386, 409, 426, 429, 447, 488, 491,
    504, 509, 510, 519, 528, 550, 574, 582, 604, 606, 608, 621, 661, 683, 700, 714, 718, 729, 749,
    761, 762, 791, 797, 806, 812, 846, 850, 858, 860, 873, 887, 899, 918, 949, 952, 955, 956, 968,
    977, 979, 1001, 1003, 1008, 1017, 1028, 1043, 1081, 1094, 1097, 1102, 1112, 1132, 1136, 1175,
    1188, 1201, 1214, 1246, 1268, 1275, 1279, 1294, 1297, 1299, 1307, 1317, 1331, 1337, 1349, 1358,
    1367, 1370, 1389, 1417, 1442, 1448, 1458, 1459, 1461, 1482, 1484, 1502, 1540, 1542, 1565, 1573,
    1583, 1587, 1607, 1610, 1626, 1648, 1650, 1656, 1665, 1684, 1709, 1732, 1747, 1748, 1763, 1778,
    1786, 1792, 1793, 1804, 1818, 1835, 1844, 1856, 1859, 1861, 1884, 1899, 1900, 1902, 1907, 1933,
    1935, 1944, 1954, 1965, 1970, 1971, 1972, 1975, 1986, 1998, 2023, 2040, 2064, 2075, 2087, 2093,
    2094, 2131, 2143, 2150, 2153, 2160, 2174, 2194, 2208, 2229, 2257, 2271, 2280, 2292, 2297, 2299,
    2318, 2324, 2333, 2338, 2369, 2374, 2396, 2409, 2413, 2418, 2422, 2423, 2445, 2460, 2464, 2477,
    2512, 2518, 2574, 2578, 2600, 2609, 2618, 2624, 2646, 2650, 2656, 2660, 2667, 2676, 2694, 2712,
    2733, 2762, 2769, 2770, 2784, 2790, 2791, 2805, 2819, 2825, 2828, 2840, 2845, 2849, 2855, 2856,
    2858, 2859, 2867, 2880, 2881, 2882, 2883, 2911, 2920, 2927, 2949, 2963, 2972, 2978, 2999, 3014,
    3019, 3035
])
# layer 1's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.17 bits; these 64 together leave 0.83 of 0.95)
mlp1 = node(L[1].mlp[
    6, 158, 166, 185, 267, 278, 294, 324, 583, 707, 708, 735, 790, 833, 871, 894, 906, 1097, 1103,
    1126, 1152, 1162, 1214, 1228, 1237, 1252, 1270, 1276, 1436, 1467, 1468, 1516, 1629, 1687, 1693,
    1728, 1733, 1773, 1942, 1954, 1962, 2035, 2109, 2115, 2128, 2186, 2193, 2290, 2313, 2329, 2427,
    2471, 2513, 2529, 2554, 2591, 2592, 2616, 2671, 2694, 2968, 2977, 3037, 3069
])
h2_3 = node(L[2].head[3])  # recovers 0.33 bits
h2_4 = node(L[2].head[4])  # recovers 0.39 bits
h3_4 = node(L[3].head[4])  # recovers 0.70 bits
# layer 3's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.43 bits; these 128 together leave 0.62 of 0.95)
mlp3 = node(L[3].mlp[
    2, 19, 21, 25, 31, 40, 78, 87, 106, 163, 257, 260, 286, 298, 307, 329, 353, 372, 391, 404, 413,
    426, 435, 517, 522, 535, 552, 592, 617, 618, 656, 661, 679, 682, 692, 699, 710, 791, 823, 852,
    879, 882, 949, 965, 966, 1082, 1096, 1100, 1149, 1179, 1187, 1246, 1314, 1320, 1329, 1331, 1339,
    1490, 1508, 1580, 1593, 1598, 1630, 1696, 1738, 1746, 1775, 1822, 1826, 1833, 1842, 1863, 1910,
    1937, 1939, 1951, 1957, 1969, 1984, 2010, 2012, 2047, 2052, 2072, 2080, 2149, 2162, 2168, 2201,
    2203, 2219, 2244, 2253, 2263, 2279, 2310, 2313, 2350, 2364, 2421, 2449, 2535, 2538, 2581, 2591,
    2603, 2621, 2633, 2663, 2666, 2677, 2679, 2683, 2750, 2751, 2761, 2762, 2773, 2790, 2893, 2895,
    2907, 2911, 2942, 2958, 3037, 3039, 3058
])

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_4,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    h2_3 >> h3_4,
    h2_3 >> mlp3,
    h2_4 >> h3_4,
    h2_4 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)

