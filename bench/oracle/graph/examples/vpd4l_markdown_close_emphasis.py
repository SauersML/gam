"""Behavior markdown_close.emphasis (vpd4l; M's top token is right on 84% of the targets): Markdown
closing: after a word opened with ** (bold) or _ (italic), the next token closes the same marker;
the counterfactual opens the other marker.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 4.88 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 4.87, L2.H1 3.42, layer 1 MLP 1.80, L3.H3 1.65, L2.H2 1.42, L3.H0 1.19,
layer 2 MLP 1.17, L1.H4 0.66, L0.H3 0.54 bits. The program lets every write among them reach every
later read.
"""
from mech import node, edges, L, embed, logits

h0_3 = node(L[0].head[3])  # recovers 0.54 bits
# layer 0's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 4.87 bits; these 64 together leave 2.43 of 4.88)
mlp0 = node(L[0].mlp[
    15, 66, 96, 170, 172, 197, 204, 335, 367, 497, 498, 603, 650, 675, 697, 713, 760, 767, 911, 938,
    939, 1025, 1069, 1083, 1188, 1313, 1320, 1328, 1329, 1330, 1429, 1451, 1543, 1737, 1738, 1743,
    1827, 1898, 1997, 2064, 2105, 2110, 2196, 2212, 2221, 2255, 2287, 2387, 2408, 2421, 2433, 2503,
    2520, 2553, 2606, 2636, 2739, 2743, 2784, 2881, 2926, 2943, 2977, 3052
])
h1_4 = node(L[1].head[4])  # recovers 0.66 bits
# layer 1's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 1.80 bits; these 64 together leave 2.85 of 4.88)
mlp1 = node(L[1].mlp[
    213, 529, 545, 559, 590, 637, 730, 734, 805, 830, 839, 843, 854, 875, 890, 910, 920, 970, 1011,
    1026, 1067, 1135, 1219, 1240, 1348, 1365, 1366, 1450, 1567, 1613, 1685, 1690, 1722, 1788, 1790,
    1794, 1802, 1916, 1993, 1998, 2066, 2111, 2136, 2143, 2262, 2319, 2328, 2340, 2399, 2454, 2457,
    2503, 2546, 2701, 2796, 2811, 2898, 2905, 2941, 2944, 2960, 2971, 3057, 3068
])
h2_1 = node(L[2].head[1])  # recovers 3.42 bits
h2_2 = node(L[2].head[2])  # recovers 1.42 bits
# layer 2's MLP: the 256 neurons that recover the most when patched alone (the whole MLP recovers 1.17 bits; these 256 together leave 2.58 of 4.88)
mlp2 = node(L[2].mlp[
    3, 4, 11, 24, 28, 30, 60, 77, 79, 93, 96, 115, 134, 135, 161, 197, 209, 212, 218, 223, 238, 255,
    278, 301, 308, 319, 320, 335, 366, 367, 368, 384, 390, 399, 427, 430, 437, 453, 463, 464, 466,
    489, 491, 509, 511, 512, 545, 548, 558, 568, 572, 576, 595, 598, 618, 621, 629, 630, 641, 643,
    655, 656, 672, 710, 735, 738, 747, 752, 761, 762, 788, 814, 817, 826, 834, 835, 859, 864, 899,
    931, 957, 964, 965, 968, 970, 987, 1000, 1009, 1030, 1046, 1114, 1115, 1127, 1132, 1183, 1189,
    1196, 1214, 1245, 1247, 1251, 1260, 1266, 1286, 1291, 1304, 1309, 1329, 1330, 1346, 1354, 1356,
    1362, 1367, 1406, 1415, 1421, 1429, 1438, 1444, 1464, 1468, 1470, 1471, 1488, 1489, 1502, 1510,
    1542, 1556, 1560, 1562, 1572, 1587, 1588, 1594, 1612, 1642, 1644, 1673, 1683, 1685, 1686, 1706,
    1728, 1761, 1781, 1782, 1789, 1795, 1796, 1807, 1812, 1816, 1826, 1830, 1832, 1836, 1857, 1866,
    1882, 1901, 1910, 1918, 1930, 1933, 1940, 1950, 1954, 1957, 1959, 1981, 1987, 1992, 1995, 1999,
    2068, 2087, 2093, 2099, 2118, 2125, 2131, 2135, 2139, 2143, 2144, 2161, 2164, 2166, 2175, 2176,
    2180, 2182, 2217, 2218, 2235, 2239, 2243, 2246, 2247, 2251, 2254, 2259, 2317, 2335, 2346, 2358,
    2382, 2399, 2409, 2412, 2429, 2432, 2456, 2465, 2485, 2501, 2553, 2554, 2557, 2609, 2611, 2621,
    2653, 2662, 2667, 2700, 2709, 2714, 2725, 2778, 2786, 2787, 2844, 2862, 2868, 2871, 2886, 2902,
    2907, 2920, 2927, 2930, 2932, 2941, 2949, 2951, 2960, 2967, 2973, 3000, 3037, 3044, 3061, 3062
])
h3_0 = node(L[3].head[0])  # recovers 1.19 bits
h3_3 = node(L[3].head[3])  # recovers 1.65 bits

edges(
    embed >> h0_3,
    embed >> mlp0,
    embed >> h1_4,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_3,
    h0_3 >> mlp0,
    h0_3 >> h1_4,
    h0_3 >> mlp1,
    h0_3 >> h2_1,
    h0_3 >> h2_2,
    h0_3 >> mlp2,
    h0_3 >> h3_0,
    h0_3 >> h3_3,
    mlp0 >> h1_4,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_3,
    h1_4 >> mlp1,
    h1_4 >> h2_1,
    h1_4 >> h2_2,
    h1_4 >> mlp2,
    h1_4 >> h3_0,
    h1_4 >> h3_3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_3,
    h2_1 >> mlp2,
    h2_1 >> h3_0,
    h2_1 >> h3_3,
    h2_2 >> mlp2,
    h2_2 >> h3_0,
    h2_2 >> h3_3,
    mlp2 >> h3_0,
    mlp2 >> h3_3,
    h0_3 >> logits,
    mlp0 >> logits,
    h1_4 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_3 >> logits,
)

