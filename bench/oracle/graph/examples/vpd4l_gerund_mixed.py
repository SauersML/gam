"""Behavior gerund.mixed (vpd4l; M's top token is right on 22% of the targets): Morphology: the -ing
form of a verb, with its spelling changes. Phrasings: 'go: going\nsit: sitting\n{X}:' | 'I like to
{X}. Right now I am'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.06 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 3.38 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: layer 0 MLP 1.05, L3.H4 0.65, layer 3 MLP 0.58, layer 1 MLP 0.42, layer 2 MLP 0.36 bits.
The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

# layer 0's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 1.05 bits; these 64 together leave 0.69 of 1.06)
mlp0 = node(L[0].mlp[
    17, 39, 66, 88, 98, 135, 156, 172, 255, 352, 369, 462, 606, 609, 636, 688, 711, 787, 807, 862,
    956, 1171, 1222, 1226, 1240, 1241, 1279, 1392, 1584, 1616, 1692, 1709, 1723, 1729, 1733, 1778,
    1792, 1800, 1828, 1849, 1878, 1926, 1960, 1961, 2102, 2211, 2309, 2316, 2333, 2355, 2358, 2368,
    2420, 2437, 2468, 2490, 2562, 2650, 2745, 2775, 2784, 2825, 2894, 2998
])
# layer 1's MLP: the 32 neurons that recover the most when patched alone (the whole MLP recovers 0.42 bits; these 32 together leave 0.77 of 1.06)
mlp1 = node(L[1].mlp[
    105, 114, 198, 257, 291, 330, 618, 661, 729, 755, 876, 966, 1123, 1165, 1193, 1221, 1303, 1598,
    1680, 1685, 1760, 1894, 1977, 2140, 2193, 2276, 2438, 2517, 2555, 2587, 2878, 2980
])
# layer 2's MLP: the 64 neurons that recover the most when patched alone (the whole MLP recovers 0.36 bits; these 64 together leave 0.75 of 1.06)
mlp2 = node(L[2].mlp[
    21, 39, 83, 105, 147, 228, 319, 330, 400, 449, 462, 481, 585, 615, 631, 635, 725, 727, 728, 786,
    827, 834, 858, 872, 883, 902, 910, 957, 989, 995, 999, 1033, 1044, 1050, 1137, 1234, 1329, 1358,
    1447, 1459, 1492, 1718, 1725, 1842, 1931, 2110, 2148, 2206, 2307, 2341, 2355, 2362, 2466, 2473,
    2538, 2539, 2560, 2646, 2661, 2667, 2769, 2895, 2963, 3059
])
h3_4 = node(L[3].head[4])  # recovers 0.65 bits
# layer 3's MLP: the 128 neurons that recover the most when patched alone (the whole MLP recovers 0.58 bits; these 128 together leave 0.65 of 1.06)
mlp3 = node(L[3].mlp[
    10, 37, 80, 100, 111, 171, 182, 216, 243, 244, 245, 255, 257, 262, 297, 313, 340, 363, 383, 404,
    411, 421, 426, 427, 450, 469, 480, 487, 521, 535, 539, 561, 608, 617, 672, 675, 707, 709, 743,
    747, 767, 829, 837, 857, 860, 870, 879, 888, 890, 949, 998, 1082, 1096, 1109, 1145, 1150, 1152,
    1162, 1196, 1235, 1276, 1331, 1346, 1508, 1519, 1566, 1593, 1606, 1620, 1640, 1648, 1649, 1680,
    1693, 1741, 1789, 1817, 1846, 1863, 1892, 1895, 1957, 1963, 1973, 2007, 2010, 2052, 2072, 2090,
    2093, 2128, 2179, 2196, 2237, 2238, 2253, 2263, 2307, 2310, 2320, 2403, 2409, 2412, 2421, 2432,
    2464, 2484, 2505, 2591, 2602, 2618, 2678, 2679, 2718, 2726, 2728, 2797, 2845, 2849, 2859, 2871,
    2924, 2939, 2945, 2955, 2968, 2982, 3046
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

