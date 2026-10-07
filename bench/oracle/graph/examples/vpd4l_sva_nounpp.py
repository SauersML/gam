"""Behavior sva.nounpp (vpd4l; M's top token is right on 10% of the targets): Subject-verb agreement
after a noun followed by a prepositional phrase with a distractor noun; the counterfactual flips the
subject's number.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.31 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 1.30, L3.H3 1.19, layer 2 MLP 0.99, layer 1 MLP 0.49, L3.H0 0.29, layer 3
MLP 0.21 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 1.30 bits; these 64 together leave 0.14 of 1.31)
mlp0 = node(L[0].mlp[
    30, 45, 61, 113, 190, 287, 325, 394, 530, 561, 645, 673, 676, 724, 734, 795, 869, 983, 1013,
    1081, 1211, 1219, 1242, 1261, 1291, 1343, 1360, 1380, 1385, 1387, 1395, 1404, 1426, 1467, 1480,
    1495, 1499, 1514, 1614, 1628, 1663, 1738, 1825, 1845, 1892, 1902, 1960, 1993, 2151, 2167, 2196,
    2205, 2228, 2266, 2374, 2397, 2398, 2475, 2606, 2646, 2677, 2949, 2996, 3028
])
# layer 1's MLP: the 256 neurons that recover the most when patched alone (the whole MLP recovers 0.49 bits; these 256 together leave 0.46 of 1.31)
mlp1 = node(L[1].mlp[
    0, 4, 16, 28, 44, 68, 74, 83, 87, 100, 115, 121, 130, 182, 185, 187, 189, 195, 217, 222, 224,
    260, 263, 285, 296, 299, 311, 319, 322, 333, 338, 340, 347, 355, 357, 362, 378, 385, 389, 403,
    410, 425, 469, 477, 481, 507, 508, 512, 522, 542, 553, 554, 556, 558, 567, 583, 630, 650, 680,
    686, 703, 744, 757, 758, 783, 797, 818, 830, 831, 833, 845, 846, 864, 867, 869, 883, 934, 943,
    944, 990, 1016, 1067, 1070, 1073, 1074, 1079, 1088, 1101, 1106, 1112, 1118, 1131, 1153, 1184,
    1200, 1227, 1229, 1235, 1245, 1247, 1268, 1289, 1292, 1296, 1298, 1311, 1319, 1332, 1337, 1357,
    1358, 1361, 1364, 1373, 1402, 1419, 1422, 1425, 1477, 1480, 1484, 1492, 1496, 1501, 1519, 1536,
    1548, 1572, 1581, 1582, 1596, 1601, 1610, 1614, 1660, 1662, 1691, 1693, 1702, 1720, 1739, 1749,
    1752, 1762, 1783, 1788, 1821, 1841, 1843, 1844, 1872, 1879, 1887, 1925, 1958, 1959, 1962, 1967,
    1970, 1978, 1983, 1990, 1999, 2000, 2001, 2005, 2013, 2022, 2035, 2049, 2054, 2061, 2063, 2064,
    2095, 2101, 2117, 2124, 2128, 2139, 2144, 2162, 2175, 2183, 2188, 2207, 2233, 2238, 2247, 2248,
    2261, 2275, 2289, 2290, 2298, 2310, 2328, 2334, 2393, 2407, 2420, 2431, 2440, 2449, 2450, 2453,
    2457, 2458, 2471, 2487, 2490, 2502, 2518, 2548, 2554, 2569, 2593, 2594, 2602, 2612, 2618, 2628,
    2653, 2662, 2678, 2679, 2705, 2721, 2744, 2746, 2758, 2766, 2801, 2808, 2812, 2817, 2836, 2839,
    2849, 2853, 2875, 2908, 2919, 2921, 2930, 2931, 2937, 2946, 2977, 2984, 2989, 3019, 3044, 3049,
    3051, 3071
])
# layer 2's MLP: the 32 neurons that recover the most when patched alone (the whole MLP recovers 0.99 bits; these 32 together leave 0.15 of 1.31)
mlp2 = node(L[2].mlp[
    236, 330, 365, 412, 477, 491, 500, 578, 662, 862, 865, 895, 1055, 1086, 1331, 1409, 1444, 1447,
    1449, 1634, 1683, 1802, 1817, 1837, 1931, 1984, 2180, 2225, 2420, 2547, 2760, 2838
])
h3_0 = node(L[3].head[0])  # recovers 0.29 bits
h3_3 = node(L[3].head[3])  # recovers 1.19 bits
# layer 3's MLP: the 32 neurons that recover the most when patched alone (the whole MLP recovers 0.21 bits; these 32 together leave 0.59 of 1.31)
mlp3 = node(L[3].mlp[
    17, 171, 178, 317, 392, 714, 815, 830, 911, 1045, 1112, 1394, 1659, 1689, 1786, 1911, 1951,
    1958, 1995, 2046, 2114, 2126, 2204, 2217, 2259, 2267, 2276, 2309, 2427, 2474, 2716, 3022
])

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_3,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_3,
    mlp1 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_3,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_3 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)

