"""Behavior sva.rc (vpd4l; M's top token is right on 3% of the targets): Subject-verb agreement after a
noun followed by a relative clause with a distractor noun; the counterfactual flips the subject's
number.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 0.80 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 0.79, layer 2 MLP 0.66, L3.H3 0.62, layer 1 MLP 0.39 bits. The program lets
every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.79 bits; these 64 together leave 0.11 of 0.80)
mlp0 = node(L[0].mlp[
    30, 45, 190, 287, 325, 358, 463, 615, 622, 645, 724, 734, 795, 832, 857, 869, 1081, 1163, 1202,
    1207, 1211, 1219, 1242, 1261, 1268, 1313, 1343, 1380, 1385, 1404, 1426, 1467, 1480, 1499, 1626,
    1628, 1639, 1817, 1825, 1845, 1892, 1902, 1960, 1966, 1993, 2137, 2167, 2196, 2199, 2266, 2374,
    2397, 2398, 2413, 2423, 2428, 2475, 2606, 2677, 2823, 2848, 3028, 3045, 3063
])
# layer 1's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.39 bits; these 64 together leave 0.30 of 0.80)
mlp1 = node(L[1].mlp[
    74, 100, 115, 185, 189, 217, 227, 260, 319, 322, 333, 340, 347, 355, 362, 385, 462, 508, 512,
    583, 708, 783, 830, 831, 864, 867, 934, 977, 1016, 1073, 1118, 1200, 1227, 1229, 1331, 1332,
    1358, 1376, 1412, 1512, 1536, 1691, 1693, 1879, 1925, 1967, 1978, 1990, 1999, 2000, 2054, 2061,
    2188, 2264, 2298, 2442, 2457, 2466, 2490, 2502, 2678, 2853, 2931, 2946
])
# layer 2's MLP: the 16 neurons that recover the most when patched alone (the whole MLP recovers 0.66 bits; these 16 together leave 0.10 of 0.80)
mlp2 = node(L[2].mlp[
    330, 365, 412, 662, 895, 1331, 1444, 1449, 1634, 1663, 1683, 1802, 1817, 1837, 1933, 1984
])
h3_3 = node(L[3].head[3])  # recovers 0.62 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_3,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_3,
    mlp1 >> mlp2,
    mlp1 >> h3_3,
    mlp2 >> h3_3,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_3 >> logits,
)

