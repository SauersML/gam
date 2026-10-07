"""Behavior quote_close.said (vpd4l; M's top token is right on 9% of the targets): Quote closing: after
a quoted sentence ends with a period, the next token closes the quotation; the counterfactual opens
a parenthesis instead.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.40 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 3.39, L2.H1 3.09, L3.H3 2.12, layer 1 MLP 2.08, layer 2 MLP 1.29, layer 3
MLP 1.06, L0.H1 0.82, L1.H4 0.76, L3.H1 0.62, L1.H5 0.49 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_1 = node(L[0].head[1])  # recovers 0.82 bits
# layer 0's MLP: the 32 neurons that recover the most when patched alone (the whole MLP recovers 3.39 bits; these 32 together leave 0.11 of 3.40)
mlp0 = node(L[0].mlp[
    60, 72, 380, 496, 637, 646, 767, 796, 992, 999, 1067, 1078, 1174, 1270, 1283, 1297, 1334, 1721,
    1782, 2239, 2316, 2339, 2351, 2402, 2413, 2484, 2570, 2690, 2926, 2959, 3002, 3014
])
h1_4 = node(L[1].head[4])  # recovers 0.76 bits
h1_5 = node(L[1].head[5])  # recovers 0.49 bits
# layer 1's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 2.08 bits; these 64 together leave 0.81 of 3.40)
mlp1 = node(L[1].mlp[
    0, 57, 77, 99, 133, 174, 183, 275, 465, 468, 524, 561, 565, 595, 735, 819, 897, 917, 998, 999,
    1172, 1223, 1272, 1287, 1292, 1317, 1338, 1411, 1426, 1492, 1528, 1576, 1637, 1656, 1658, 1735,
    1808, 1809, 1835, 1843, 1931, 1939, 1983, 2169, 2246, 2303, 2391, 2395, 2399, 2431, 2473, 2488,
    2503, 2523, 2571, 2647, 2731, 2778, 2789, 2796, 2867, 2973, 3048, 3057
])
h2_1 = node(L[2].head[1])  # recovers 3.09 bits
# layer 2's MLP: the 256 neurons that recover the most when patched alone (the whole MLP recovers 1.29 bits; these 256 together leave 1.18 of 3.40)
mlp2 = node(L[2].mlp[
    6, 8, 9, 44, 45, 70, 74, 99, 116, 129, 140, 158, 198, 199, 215, 236, 240, 252, 253, 255, 265,
    274, 280, 289, 297, 320, 333, 341, 350, 367, 371, 380, 390, 391, 398, 401, 453, 457, 466, 475,
    491, 507, 511, 514, 518, 543, 545, 564, 574, 583, 586, 606, 611, 613, 616, 625, 636, 637, 643,
    671, 674, 686, 709, 710, 711, 729, 752, 789, 832, 845, 846, 851, 859, 860, 862, 864, 880, 898,
    905, 906, 910, 929, 937, 945, 957, 982, 983, 988, 999, 1011, 1019, 1021, 1026, 1049, 1064, 1075,
    1089, 1092, 1102, 1105, 1114, 1119, 1120, 1121, 1127, 1137, 1176, 1177, 1178, 1188, 1189, 1210,
    1229, 1262, 1266, 1282, 1292, 1370, 1400, 1407, 1451, 1461, 1464, 1472, 1479, 1480, 1549, 1562,
    1566, 1577, 1581, 1588, 1590, 1604, 1606, 1612, 1615, 1646, 1656, 1695, 1718, 1720, 1734, 1750,
    1757, 1761, 1778, 1781, 1782, 1786, 1787, 1790, 1796, 1827, 1842, 1845, 1863, 1888, 1893, 1905,
    1918, 1922, 1937, 1967, 1972, 1982, 1983, 1996, 1999, 2006, 2038, 2072, 2090, 2091, 2092, 2095,
    2098, 2104, 2117, 2129, 2134, 2140, 2141, 2148, 2149, 2152, 2199, 2209, 2225, 2231, 2264, 2265,
    2299, 2306, 2309, 2310, 2325, 2331, 2340, 2348, 2379, 2380, 2434, 2465, 2473, 2475, 2504, 2514,
    2565, 2571, 2578, 2579, 2610, 2625, 2628, 2644, 2646, 2664, 2667, 2671, 2705, 2709, 2712, 2718,
    2723, 2725, 2738, 2739, 2749, 2752, 2758, 2787, 2795, 2808, 2814, 2841, 2845, 2851, 2862, 2881,
    2882, 2886, 2920, 2924, 2926, 2928, 2937, 2957, 2959, 2961, 2963, 2974, 2975, 3003, 3050, 3058
])
h3_1 = node(L[3].head[1])  # recovers 0.62 bits
h3_3 = node(L[3].head[3])  # recovers 2.12 bits
# layer 3's MLP: the 8 neurons that recover the most when patched alone (the whole MLP recovers 1.06 bits; these 8 together leave 1.37 of 3.40)
mlp3 = node(L[3].mlp[
    82, 502, 1220, 1640, 1717, 2505, 2917, 2974
])

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> h1_4,
    embed >> h1_5,
    embed >> mlp1,
    embed >> h2_1,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_3,
    embed >> mlp3,
    h0_1 >> mlp0,
    h0_1 >> h1_4,
    h0_1 >> h1_5,
    h0_1 >> mlp1,
    h0_1 >> h2_1,
    h0_1 >> mlp2,
    h0_1 >> h3_1,
    h0_1 >> h3_3,
    h0_1 >> mlp3,
    mlp0 >> h1_4,
    mlp0 >> h1_5,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    h1_4 >> mlp1,
    h1_4 >> h2_1,
    h1_4 >> mlp2,
    h1_4 >> h3_1,
    h1_4 >> h3_3,
    h1_4 >> mlp3,
    h1_5 >> mlp1,
    h1_5 >> h2_1,
    h1_5 >> mlp2,
    h1_5 >> h3_1,
    h1_5 >> h3_3,
    h1_5 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> mlp2,
    mlp1 >> h3_1,
    mlp1 >> h3_3,
    mlp1 >> mlp3,
    h2_1 >> mlp2,
    h2_1 >> h3_1,
    h2_1 >> h3_3,
    h2_1 >> mlp3,
    mlp2 >> h3_1,
    mlp2 >> h3_3,
    mlp2 >> mlp3,
    h3_1 >> mlp3,
    h3_3 >> mlp3,
    h0_1 >> logits,
    mlp0 >> logits,
    h1_4 >> logits,
    h1_5 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)

