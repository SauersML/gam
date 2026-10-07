"""Behavior list_copy.first (vpd4l; M's top token is right on 27% of the targets): List indexing: given
a list of four words, the first word; the counterfactual swaps the first word with another, so the
same words appear.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.56 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 1.55, L3.H4 1.14, L2.H3 0.93, layer 3 MLP 0.81, L3.H5 0.64 bits. The program
lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 256 neurons that recover the most when patched alone (the whole MLP recovers 1.55 bits; these 256 together leave 0.79 of 1.56)
mlp0 = node(L[0].mlp[
    2, 22, 39, 47, 49, 87, 89, 129, 139, 141, 155, 185, 188, 203, 204, 229, 231, 232, 234, 242, 268,
    291, 297, 312, 319, 332, 346, 348, 359, 361, 386, 393, 400, 406, 407, 421, 426, 455, 472, 481,
    485, 491, 496, 501, 509, 510, 517, 521, 543, 547, 584, 585, 595, 601, 611, 613, 626, 639, 642,
    648, 652, 660, 662, 676, 715, 716, 718, 787, 793, 796, 799, 810, 820, 827, 847, 859, 872, 884,
    892, 893, 907, 911, 913, 920, 923, 938, 955, 969, 1024, 1037, 1080, 1083, 1087, 1109, 1122,
    1123, 1130, 1138, 1140, 1157, 1179, 1183, 1185, 1201, 1204, 1212, 1217, 1235, 1239, 1244, 1264,
    1282, 1288, 1301, 1315, 1323, 1326, 1355, 1366, 1379, 1422, 1445, 1456, 1457, 1489, 1504, 1554,
    1592, 1593, 1594, 1601, 1613, 1633, 1642, 1654, 1657, 1689, 1705, 1714, 1721, 1730, 1761, 1768,
    1772, 1778, 1802, 1811, 1832, 1861, 1882, 1905, 1916, 1926, 1957, 1971, 1984, 2010, 2017, 2023,
    2058, 2091, 2104, 2118, 2130, 2137, 2160, 2169, 2199, 2211, 2212, 2229, 2243, 2257, 2260, 2284,
    2292, 2327, 2363, 2369, 2374, 2378, 2411, 2429, 2431, 2434, 2440, 2448, 2482, 2497, 2501, 2504,
    2505, 2508, 2509, 2518, 2525, 2534, 2550, 2554, 2562, 2563, 2570, 2579, 2581, 2583, 2585, 2594,
    2627, 2633, 2640, 2645, 2650, 2653, 2660, 2662, 2671, 2673, 2678, 2680, 2681, 2688, 2701, 2723,
    2765, 2769, 2779, 2784, 2800, 2815, 2823, 2825, 2826, 2839, 2840, 2846, 2847, 2849, 2860, 2864,
    2865, 2878, 2884, 2898, 2909, 2922, 2928, 2929, 2944, 2948, 2992, 2994, 2997, 2999, 3002, 3039,
    3060
])
h2_3 = node(L[2].head[3])  # recovers 0.93 bits
h3_4 = node(L[3].head[4])  # recovers 1.14 bits
h3_5 = node(L[3].head[5])  # recovers 0.64 bits
# layer 3's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.81 bits; these 128 together leave 0.91 of 1.56)
mlp3 = node(L[3].mlp[
    21, 78, 83, 87, 106, 110, 136, 208, 244, 246, 262, 275, 286, 302, 307, 329, 353, 363, 383, 400,
    404, 475, 499, 523, 532, 584, 592, 618, 638, 650, 661, 692, 708, 709, 710, 736, 757, 779, 826,
    882, 907, 916, 945, 1003, 1053, 1058, 1066, 1096, 1116, 1125, 1126, 1141, 1154, 1182, 1191,
    1229, 1254, 1274, 1329, 1339, 1369, 1417, 1437, 1580, 1582, 1593, 1605, 1618, 1731, 1733, 1738,
    1746, 1769, 1772, 1805, 1809, 1835, 1861, 1863, 1910, 1915, 1939, 1946, 1994, 2010, 2047, 2052,
    2065, 2072, 2092, 2098, 2181, 2206, 2263, 2293, 2310, 2342, 2350, 2421, 2441, 2449, 2484, 2530,
    2535, 2537, 2538, 2591, 2628, 2663, 2674, 2677, 2683, 2692, 2711, 2717, 2747, 2751, 2766, 2788,
    2852, 2859, 2883, 2888, 2946, 2955, 3007, 3026, 3049
])

edges(
    embed >> mlp0,
    embed >> h2_3,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h2_3,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    h2_3 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

