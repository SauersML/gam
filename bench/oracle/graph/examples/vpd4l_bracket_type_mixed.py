"""Behavior bracket_type.mixed (vpd4l; M's top token is right on 55% of the targets): Bracket type
matching: after a run of opened brackets of mixed types and one inner pair closed, the next token
closes the innermost open bracket with its own type; the counterfactual changes that bracket's type.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 5.31 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 5.29, L2.H1 4.15, L3.H3 3.52, layer 1 MLP 2.57, L3.H1 2.15, layer 3 MLP
2.00, L0.H1 1.31, layer 2 MLP 0.90 bits. The program lets every write among them reach every later
read.
"""
from mech import node, edges, L, embed, logits

h0_1 = node(L[0].head[1])  # recovers 1.31 bits
# layer 0's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 5.29 bits; these 64 together leave 1.06 of 5.31)
mlp0 = node(L[0].mlp[
    40, 41, 68, 82, 91, 114, 237, 253, 258, 303, 356, 381, 406, 413, 415, 441, 480, 483, 521, 632,
    729, 811, 822, 860, 864, 935, 955, 1040, 1162, 1199, 1217, 1295, 1350, 1398, 1424, 1475, 1489,
    1535, 1555, 1597, 1602, 1616, 1693, 1890, 1951, 1975, 1980, 1989, 2083, 2160, 2200, 2212, 2255,
    2273, 2354, 2402, 2573, 2825, 2907, 2919, 2941, 2948, 2959, 2985
])
# layer 1's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 2.57 bits; these 64 together leave 2.00 of 5.31)
mlp1 = node(L[1].mlp[
    4, 37, 57, 69, 88, 108, 209, 234, 316, 355, 384, 390, 425, 461, 549, 576, 587, 595, 604, 611,
    626, 674, 779, 794, 839, 899, 917, 935, 1018, 1287, 1369, 1395, 1413, 1416, 1488, 1576, 1639,
    1643, 1750, 1823, 1845, 1861, 1939, 1943, 1944, 2016, 2077, 2155, 2167, 2223, 2334, 2399, 2465,
    2613, 2630, 2681, 2796, 2799, 2863, 2941, 2955, 2997, 3033, 3038
])
h2_1 = node(L[2].head[1])  # recovers 4.15 bits
# layer 2's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.90 bits; these 128 together leave 2.20 of 5.31)
mlp2 = node(L[2].mlp[
    22, 93, 185, 242, 289, 299, 316, 336, 343, 390, 407, 415, 428, 440, 452, 481, 515, 623, 654,
    657, 709, 714, 725, 775, 832, 888, 927, 936, 955, 968, 978, 988, 1026, 1034, 1045, 1059, 1068,
    1073, 1085, 1138, 1151, 1202, 1208, 1242, 1260, 1264, 1309, 1338, 1339, 1354, 1362, 1414, 1416,
    1429, 1433, 1435, 1459, 1525, 1526, 1536, 1621, 1705, 1723, 1750, 1806, 1808, 1811, 1833, 1853,
    1889, 1893, 1894, 1934, 1941, 1981, 2010, 2026, 2049, 2051, 2057, 2092, 2148, 2156, 2191, 2209,
    2226, 2275, 2299, 2335, 2348, 2360, 2374, 2380, 2381, 2390, 2410, 2465, 2517, 2519, 2542, 2558,
    2604, 2611, 2619, 2622, 2642, 2643, 2644, 2645, 2671, 2697, 2761, 2786, 2813, 2843, 2871, 2912,
    2934, 2947, 2967, 2972, 2988, 2992, 2993, 3005, 3025, 3033, 3038
])
h3_1 = node(L[3].head[1])  # recovers 2.15 bits
h3_3 = node(L[3].head[3])  # recovers 3.52 bits
# layer 3's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 2.00 bits; these 128 together leave 1.07 of 5.31)
mlp3 = node(L[3].mlp[
    12, 26, 53, 57, 60, 82, 88, 91, 113, 133, 138, 233, 235, 251, 256, 259, 292, 333, 335, 338, 345,
    367, 428, 470, 483, 556, 648, 669, 752, 764, 775, 843, 844, 880, 1019, 1054, 1103, 1150, 1151,
    1171, 1172, 1193, 1196, 1220, 1285, 1309, 1350, 1359, 1367, 1382, 1406, 1419, 1422, 1432, 1434,
    1451, 1494, 1510, 1514, 1528, 1562, 1573, 1577, 1584, 1602, 1620, 1640, 1689, 1717, 1741, 1768,
    1785, 1801, 1808, 1825, 1845, 1867, 1871, 1885, 1900, 1948, 1961, 2061, 2109, 2116, 2132, 2188,
    2235, 2237, 2287, 2290, 2297, 2337, 2363, 2381, 2419, 2453, 2471, 2486, 2487, 2492, 2503, 2513,
    2526, 2559, 2564, 2598, 2646, 2697, 2698, 2699, 2709, 2721, 2759, 2795, 2799, 2849, 2867, 2914,
    2917, 2951, 2974, 2997, 3001, 3017, 3028, 3046, 3057
])

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_3,
    embed >> mlp3,
    h0_1 >> mlp0,
    h0_1 >> mlp1,
    h0_1 >> h2_1,
    h0_1 >> mlp2,
    h0_1 >> h3_1,
    h0_1 >> h3_3,
    h0_1 >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
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
    mlp1 >> logits,
    h2_1 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)

