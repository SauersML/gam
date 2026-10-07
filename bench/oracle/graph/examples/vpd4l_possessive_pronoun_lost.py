"""Behavior possessive_pronoun.lost (vpd4l; M's top token is right on 30% of the targets): Possessive
pronoun: after a named person loses something, 'her' or 'his' follows by the name's usual gender;
the counterfactual swaps the name's gender.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 0.79 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 0.79, L3.H4 0.72, layer 2 MLP 0.42, layer 1 MLP 0.16, layer 3 MLP 0.10 bits.
The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.79 bits; these 128 together leave 0.19 of 0.79)
mlp0 = node(L[0].mlp[
    2, 6, 49, 93, 158, 167, 180, 190, 247, 278, 292, 328, 344, 346, 366, 393, 405, 439, 449, 477,
    503, 510, 517, 569, 614, 661, 666, 726, 749, 751, 759, 826, 832, 852, 857, 888, 894, 897, 978,
    995, 1052, 1062, 1074, 1096, 1101, 1110, 1112, 1117, 1128, 1139, 1154, 1169, 1198, 1217, 1222,
    1223, 1225, 1230, 1258, 1291, 1307, 1330, 1348, 1382, 1428, 1470, 1482, 1489, 1491, 1493, 1525,
    1548, 1588, 1593, 1594, 1633, 1634, 1638, 1693, 1744, 1779, 1785, 1817, 1896, 1903, 1906, 1983,
    1987, 2055, 2083, 2134, 2145, 2159, 2160, 2200, 2265, 2277, 2321, 2397, 2411, 2415, 2434, 2436,
    2545, 2628, 2630, 2633, 2635, 2646, 2722, 2729, 2759, 2767, 2790, 2810, 2825, 2831, 2886, 2896,
    2921, 2922, 2940, 2977, 2993, 2995, 3010, 3031, 3039
])
# layer 1's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.16 bits; these 64 together leave 0.61 of 0.79)
mlp1 = node(L[1].mlp[
    166, 189, 204, 212, 228, 340, 411, 414, 428, 437, 452, 470, 523, 546, 686, 827, 852, 880, 934,
    954, 1150, 1254, 1442, 1467, 1514, 1519, 1581, 1588, 1597, 1624, 1659, 1676, 1722, 1729, 1750,
    1821, 1948, 2005, 2028, 2084, 2124, 2128, 2193, 2269, 2286, 2332, 2337, 2427, 2446, 2487, 2552,
    2682, 2711, 2714, 2737, 2773, 2806, 2846, 2890, 2944, 2968, 2977, 3034, 3035
])
# layer 2's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.42 bits; these 64 together leave 0.39 of 0.79)
mlp2 = node(L[2].mlp[
    17, 67, 108, 158, 164, 170, 177, 264, 277, 319, 330, 331, 365, 408, 504, 600, 630, 642, 660,
    674, 954, 979, 1010, 1014, 1046, 1157, 1215, 1231, 1245, 1271, 1300, 1328, 1357, 1386, 1417,
    1435, 1440, 1444, 1481, 1603, 1641, 1701, 1719, 1729, 1774, 1793, 1817, 1843, 1844, 1871, 1900,
    1919, 1935, 2003, 2049, 2063, 2110, 2165, 2233, 2350, 2362, 2677, 2801, 3044
])
h3_4 = node(L[3].head[4])  # recovers 0.72 bits
# layer 3's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.10 bits; these 64 together leave 0.43 of 0.79)
mlp3 = node(L[3].mlp[
    54, 141, 145, 371, 387, 499, 512, 552, 586, 654, 695, 708, 850, 861, 894, 960, 982, 1017, 1026,
    1050, 1101, 1125, 1141, 1161, 1179, 1210, 1241, 1259, 1275, 1407, 1428, 1551, 1617, 1648, 1703,
    1715, 1717, 1756, 1780, 1799, 1837, 1910, 1937, 1965, 1967, 2023, 2060, 2236, 2296, 2320, 2351,
    2385, 2407, 2438, 2442, 2451, 2454, 2484, 2544, 2576, 2762, 2881, 3049, 3060
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

