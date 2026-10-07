"""Behavior past_tense.regular (vpd4l; M's top token is right on 22% of the targets): Past tense of a
regular verb. Phrasings: 'Every day I {X}. Yesterday I' | 'Today they {X}. Last week they' | 'I
usually {X} in the morning, but last night I'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 0.99 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 0.99, L3.H4 0.76, layer 3 MLP 0.49, L2.H4 0.35, L2.H3 0.29, layer 1 MLP
0.15, layer 2 MLP 0.13 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 256 neurons that recover the most when patched alone (the whole MLP recovers 0.99 bits; these 256 together leave 0.49 of 0.99)
mlp0 = node(L[0].mlp[
    6, 18, 29, 35, 62, 71, 83, 93, 102, 108, 114, 119, 157, 176, 187, 189, 195, 199, 203, 206, 207,
    219, 232, 247, 249, 251, 256, 259, 260, 270, 283, 287, 293, 309, 312, 315, 324, 329, 330, 353,
    366, 386, 392, 394, 426, 441, 447, 454, 459, 473, 479, 493, 499, 525, 528, 530, 568, 606, 609,
    610, 611, 616, 644, 647, 651, 660, 666, 681, 690, 697, 703, 732, 745, 757, 766, 767, 770, 787,
    796, 800, 802, 803, 810, 812, 846, 849, 873, 890, 914, 918, 977, 983, 996, 1030, 1056, 1064,
    1072, 1074, 1097, 1101, 1112, 1131, 1134, 1135, 1142, 1143, 1153, 1189, 1190, 1279, 1299, 1310,
    1313, 1352, 1358, 1365, 1370, 1384, 1405, 1415, 1442, 1444, 1446, 1456, 1499, 1502, 1518, 1522,
    1524, 1545, 1551, 1559, 1567, 1581, 1584, 1597, 1605, 1607, 1613, 1614, 1621, 1638, 1648, 1654,
    1683, 1684, 1687, 1705, 1726, 1733, 1747, 1783, 1793, 1814, 1844, 1846, 1860, 1878, 1893, 1900,
    1902, 1907, 1921, 1934, 1946, 1954, 1960, 1970, 1984, 1986, 2013, 2018, 2020, 2040, 2064, 2070,
    2071, 2083, 2113, 2121, 2127, 2131, 2161, 2164, 2172, 2179, 2180, 2185, 2196, 2211, 2218, 2257,
    2261, 2292, 2297, 2299, 2308, 2311, 2325, 2326, 2329, 2332, 2333, 2344, 2395, 2396, 2403, 2418,
    2423, 2425, 2436, 2463, 2464, 2472, 2483, 2518, 2523, 2534, 2574, 2578, 2600, 2608, 2618, 2620,
    2624, 2653, 2667, 2678, 2705, 2715, 2748, 2754, 2794, 2820, 2849, 2858, 2859, 2868, 2880, 2890,
    2900, 2919, 2920, 2921, 2927, 2954, 2958, 2965, 2977, 2980, 2994, 2999, 3020, 3037, 3046, 3069
])
# layer 1's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.15 bits; these 64 together leave 0.88 of 0.99)
mlp1 = node(L[1].mlp[
    121, 132, 137, 166, 194, 212, 228, 243, 397, 401, 407, 458, 465, 488, 609, 633, 707, 726, 774,
    904, 925, 971, 1048, 1051, 1076, 1139, 1162, 1187, 1254, 1262, 1270, 1388, 1515, 1554, 1728,
    1764, 1779, 1811, 2008, 2038, 2103, 2119, 2128, 2183, 2242, 2263, 2268, 2312, 2372, 2420, 2490,
    2545, 2573, 2592, 2671, 2694, 2732, 2766, 2798, 2835, 2837, 3028, 3036, 3037
])
h2_3 = node(L[2].head[3])  # recovers 0.29 bits
h2_4 = node(L[2].head[4])  # recovers 0.35 bits
# layer 2's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.13 bits; these 64 together leave 0.84 of 0.99)
mlp2 = node(L[2].mlp[
    8, 120, 197, 311, 320, 422, 467, 495, 517, 566, 567, 631, 761, 918, 979, 991, 1000, 1050, 1054,
    1066, 1069, 1121, 1229, 1231, 1295, 1358, 1361, 1447, 1481, 1492, 1556, 1600, 1740, 1743, 1853,
    1881, 1900, 1969, 2068, 2091, 2128, 2200, 2259, 2284, 2367, 2389, 2421, 2423, 2471, 2473, 2475,
    2511, 2547, 2565, 2567, 2666, 2678, 2746, 2769, 2901, 2920, 2946, 2963, 2981
])
h3_4 = node(L[3].head[4])  # recovers 0.76 bits
# layer 3's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.49 bits; these 128 together leave 0.65 of 0.99)
mlp3 = node(L[3].mlp[
    2, 13, 32, 37, 40, 50, 87, 106, 110, 122, 153, 262, 271, 307, 329, 372, 413, 417, 441, 456, 499,
    535, 552, 579, 617, 650, 661, 688, 692, 709, 745, 757, 791, 801, 879, 882, 940, 949, 955, 966,
    991, 993, 994, 1016, 1026, 1072, 1096, 1100, 1101, 1141, 1150, 1162, 1179, 1245, 1329, 1331,
    1339, 1454, 1466, 1490, 1566, 1580, 1581, 1593, 1598, 1615, 1617, 1676, 1746, 1756, 1809, 1822,
    1850, 1863, 1869, 1906, 1951, 1963, 2010, 2012, 2030, 2037, 2047, 2048, 2052, 2060, 2092, 2093,
    2101, 2176, 2201, 2219, 2244, 2253, 2263, 2327, 2338, 2350, 2364, 2385, 2406, 2421, 2484, 2535,
    2538, 2562, 2581, 2586, 2628, 2633, 2640, 2663, 2666, 2679, 2683, 2718, 2748, 2750, 2751, 2761,
    2762, 2839, 2845, 2893, 3002, 3006, 3026, 3049
])

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_4,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_4,
    h2_3 >> mlp3,
    h2_4 >> mlp2,
    h2_4 >> h3_4,
    h2_4 >> mlp3,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)

