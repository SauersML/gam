"""Behavior bracket_close.call (vpd4l; M's top token is right on 50% of the targets): Bracket closing
in function calls: after the last argument the next token closes the open calls.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.58 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 3.54, L2.H1 2.61, L0.H1 1.75, L1.H0 1.59, L1.H4 1.34, layer 2 MLP 1.18,
L3.H3 1.17, L1.H1 1.03, L3.H1 0.81, L2.H0 0.55, L1.H3 0.50 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_1 = node(L[0].head[1])  # recovers 1.75 bits
# layer 0's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 3.54 bits; these 64 together leave 0.22 of 3.58)
mlp0 = node(L[0].mlp[
    1, 14, 82, 159, 161, 165, 183, 235, 378, 416, 490, 587, 592, 616, 673, 702, 712, 752, 783, 808,
    852, 886, 994, 1121, 1170, 1181, 1217, 1220, 1274, 1278, 1336, 1447, 1472, 1512, 1579, 1587,
    1682, 1694, 1751, 1792, 1872, 1893, 1924, 1941, 1949, 1981, 2065, 2122, 2177, 2221, 2365, 2402,
    2589, 2650, 2773, 2802, 2825, 2851, 2867, 2881, 2904, 2929, 3010, 3027
])
h1_0 = node(L[1].head[0])  # recovers 1.59 bits
h1_1 = node(L[1].head[1])  # recovers 1.03 bits
h1_3 = node(L[1].head[3])  # recovers 0.50 bits
h1_4 = node(L[1].head[4])  # recovers 1.34 bits
h2_0 = node(L[2].head[0])  # recovers 0.55 bits
h2_1 = node(L[2].head[1])  # recovers 2.61 bits
# layer 2's MLP: the 256 neurons that recover the most when patched alone (the whole MLP recovers 1.18 bits; these 256 together leave 0.85 of 3.58)
mlp2 = node(L[2].mlp[
    6, 12, 33, 58, 67, 80, 81, 92, 109, 118, 125, 135, 141, 150, 159, 177, 189, 212, 215, 229, 238,
    250, 274, 283, 287, 289, 297, 300, 314, 318, 325, 327, 331, 342, 351, 360, 363, 395, 410, 415,
    430, 452, 468, 472, 478, 479, 480, 482, 490, 517, 532, 555, 578, 588, 590, 657, 672, 687, 698,
    717, 718, 719, 722, 725, 726, 744, 753, 764, 775, 798, 800, 814, 831, 835, 839, 847, 855, 858,
    864, 875, 885, 915, 917, 927, 941, 983, 1036, 1042, 1051, 1059, 1068, 1080, 1098, 1101, 1106,
    1117, 1131, 1141, 1143, 1156, 1169, 1172, 1179, 1202, 1203, 1204, 1212, 1215, 1222, 1254, 1282,
    1286, 1298, 1328, 1332, 1350, 1372, 1373, 1376, 1401, 1411, 1414, 1453, 1466, 1471, 1487, 1502,
    1507, 1532, 1545, 1574, 1600, 1613, 1616, 1617, 1633, 1643, 1647, 1655, 1660, 1663, 1674, 1700,
    1731, 1734, 1739, 1740, 1750, 1760, 1777, 1785, 1814, 1866, 1868, 1879, 1880, 1893, 1897, 1898,
    1914, 1922, 1939, 1944, 1947, 1951, 1964, 1971, 1974, 1995, 2006, 2033, 2038, 2050, 2057, 2066,
    2087, 2107, 2111, 2120, 2121, 2129, 2135, 2152, 2157, 2169, 2188, 2192, 2205, 2209, 2210, 2226,
    2229, 2244, 2250, 2254, 2255, 2280, 2286, 2294, 2296, 2324, 2347, 2359, 2365, 2372, 2388, 2411,
    2412, 2415, 2435, 2442, 2447, 2448, 2456, 2498, 2499, 2502, 2510, 2516, 2561, 2562, 2565, 2597,
    2600, 2619, 2648, 2652, 2683, 2715, 2717, 2750, 2756, 2758, 2830, 2838, 2861, 2879, 2886, 2892,
    2894, 2945, 2953, 2967, 2972, 2975, 2977, 2980, 2992, 2997, 3009, 3037, 3038, 3056, 3060, 3067,
    3069
])
h3_1 = node(L[3].head[1])  # recovers 0.81 bits
h3_3 = node(L[3].head[3])  # recovers 1.17 bits

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> h1_0,
    embed >> h1_1,
    embed >> h1_3,
    embed >> h1_4,
    embed >> h2_0,
    embed >> h2_1,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_3,
    h0_1 >> mlp0,
    h0_1 >> h1_0,
    h0_1 >> h1_1,
    h0_1 >> h1_3,
    h0_1 >> h1_4,
    h0_1 >> h2_0,
    h0_1 >> h2_1,
    h0_1 >> mlp2,
    h0_1 >> h3_1,
    h0_1 >> h3_3,
    mlp0 >> h1_0,
    mlp0 >> h1_1,
    mlp0 >> h1_3,
    mlp0 >> h1_4,
    mlp0 >> h2_0,
    mlp0 >> h2_1,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    h1_0 >> h2_0,
    h1_0 >> h2_1,
    h1_0 >> mlp2,
    h1_0 >> h3_1,
    h1_0 >> h3_3,
    h1_1 >> h2_0,
    h1_1 >> h2_1,
    h1_1 >> mlp2,
    h1_1 >> h3_1,
    h1_1 >> h3_3,
    h1_3 >> h2_0,
    h1_3 >> h2_1,
    h1_3 >> mlp2,
    h1_3 >> h3_1,
    h1_3 >> h3_3,
    h1_4 >> h2_0,
    h1_4 >> h2_1,
    h1_4 >> mlp2,
    h1_4 >> h3_1,
    h1_4 >> h3_3,
    h2_0 >> mlp2,
    h2_0 >> h3_1,
    h2_0 >> h3_3,
    h2_1 >> mlp2,
    h2_1 >> h3_1,
    h2_1 >> h3_3,
    mlp2 >> h3_1,
    mlp2 >> h3_3,
    h0_1 >> logits,
    mlp0 >> logits,
    h1_0 >> logits,
    h1_1 >> logits,
    h1_3 >> logits,
    h1_4 >> logits,
    h2_0 >> logits,
    h2_1 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
)

