"""Behavior syllogism.all_are (vpd4l; M's top token is right on 10% of the targets): Syllogism: 'All X
are Y. N is one of the X. So N is a' is followed by Y; the counterfactual changes the category.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.31 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 1.30, L2.H3 0.88, L3.H4 0.83, layer 3 MLP 0.66, layer 2 MLP 0.14 bits. The
program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 256 neurons that recover the most when patched alone (the whole MLP recovers 1.30 bits; these 256 together leave 0.54 of 1.31)
mlp0 = node(L[0].mlp[
    5, 8, 32, 35, 43, 45, 51, 66, 72, 102, 131, 139, 143, 144, 172, 177, 182, 194, 195, 229, 244,
    245, 247, 251, 260, 263, 276, 314, 330, 335, 354, 364, 366, 378, 388, 399, 420, 429, 439, 440,
    479, 481, 485, 492, 499, 509, 528, 536, 556, 571, 574, 580, 584, 585, 606, 609, 614, 617, 622,
    623, 638, 650, 666, 676, 688, 730, 735, 748, 767, 770, 800, 807, 813, 842, 849, 852, 853, 855,
    858, 884, 888, 901, 918, 919, 923, 932, 962, 979, 993, 1001, 1003, 1025, 1027, 1039, 1041, 1072,
    1095, 1106, 1110, 1111, 1113, 1129, 1131, 1136, 1154, 1159, 1171, 1179, 1186, 1189, 1190, 1215,
    1246, 1250, 1257, 1275, 1281, 1297, 1329, 1343, 1346, 1349, 1350, 1395, 1413, 1425, 1429, 1464,
    1465, 1484, 1495, 1506, 1516, 1524, 1550, 1551, 1554, 1558, 1585, 1591, 1593, 1594, 1604, 1605,
    1633, 1655, 1675, 1686, 1699, 1701, 1705, 1710, 1732, 1733, 1754, 1764, 1774, 1786, 1809, 1812,
    1829, 1906, 1934, 1944, 1963, 1983, 1989, 1990, 1997, 1999, 2002, 2007, 2008, 2021, 2023, 2036,
    2042, 2052, 2082, 2083, 2133, 2134, 2140, 2143, 2162, 2185, 2192, 2195, 2204, 2217, 2234, 2243,
    2246, 2257, 2260, 2266, 2302, 2304, 2314, 2339, 2343, 2365, 2375, 2381, 2409, 2411, 2422, 2435,
    2442, 2443, 2448, 2452, 2475, 2491, 2537, 2544, 2549, 2559, 2571, 2592, 2611, 2653, 2660, 2667,
    2672, 2712, 2719, 2723, 2760, 2776, 2778, 2782, 2784, 2796, 2807, 2829, 2851, 2856, 2867, 2872,
    2888, 2889, 2890, 2914, 2917, 2924, 2930, 2935, 2965, 2967, 2971, 2995, 3031, 3063, 3064, 3070
])
h2_3 = node(L[2].head[3])  # recovers 0.88 bits
# layer 2's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.14 bits; these 64 together leave 1.03 of 1.31)
mlp2 = node(L[2].mlp[
    21, 99, 104, 106, 184, 193, 319, 330, 385, 454, 493, 601, 660, 943, 956, 984, 994, 1042, 1234,
    1274, 1347, 1406, 1407, 1438, 1479, 1491, 1662, 1677, 1715, 1749, 1797, 1803, 1851, 1900, 1966,
    1969, 2064, 2077, 2091, 2103, 2109, 2122, 2128, 2143, 2259, 2264, 2362, 2384, 2417, 2454, 2550,
    2557, 2565, 2678, 2774, 2794, 2804, 2811, 2821, 2865, 2935, 2949, 3024, 3059
])
h3_4 = node(L[3].head[4])  # recovers 0.83 bits
# layer 3's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.66 bits; these 128 together leave 0.81 of 1.31)
mlp3 = node(L[3].mlp[
    40, 69, 78, 83, 110, 130, 145, 171, 186, 208, 212, 219, 262, 268, 297, 317, 349, 353, 360, 381,
    400, 404, 434, 457, 466, 480, 499, 538, 601, 617, 649, 660, 671, 682, 708, 722, 760, 815, 827,
    879, 907, 932, 945, 974, 983, 1012, 1125, 1141, 1145, 1174, 1182, 1241, 1259, 1276, 1286, 1327,
    1334, 1344, 1354, 1417, 1419, 1438, 1456, 1534, 1562, 1585, 1593, 1609, 1646, 1650, 1799, 1800,
    1815, 1835, 1836, 1856, 1863, 1883, 1937, 1939, 1963, 1994, 2034, 2037, 2040, 2045, 2047, 2052,
    2080, 2097, 2101, 2199, 2231, 2288, 2293, 2346, 2350, 2356, 2385, 2430, 2449, 2484, 2530, 2538,
    2572, 2602, 2621, 2649, 2663, 2667, 2684, 2692, 2727, 2750, 2751, 2761, 2763, 2768, 2819, 2825,
    2850, 2893, 2909, 2920, 2946, 2952, 2994, 3060
])

edges(
    embed >> mlp0,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_4,
    embed >> mlp3,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_4,
    h2_3 >> mlp3,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)

