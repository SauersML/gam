"""Behavior pronoun_gender.because (vpd4l): Gendered pronoun: the next word is the pronoun for the
named person; the counterfactual swaps the name's gender.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[123, 226, 275, 327, 328, 332, 354, 379, 591, 985, 1085, 1275, 1297, 1308, 1443,
    1512, 1531, 1604, 1629, 1724, 1838, 1973, 2040, 2042, 2054, 2079, 2214, 2266, 2554, 2683, 2822,
    2901, 2961], PD.vpd[0].down_proj[137, 146, 200, 245, 406, 785, 862, 1036, 1196, 1411, 1991,
    2041, 2044, 2095, 2172, 2252, 2275, 2359, 2419, 2424, 2492, 2643, 2696, 2726, 2860, 3196, 3201,
    3382, 3455, 3473, 3494]
)
mlp1 = node(PD.vpd[1].down_proj[1217, 3478])
mlp2 = node(
    PD.vpd[2].c_fc[64, 67, 82, 108, 178, 259, 323, 401, 606, 747, 762, 770, 962, 1108, 1191, 1293,
    1340, 1449, 1613, 1691, 1858, 1914, 2151, 2244, 2276, 2415, 2482, 2646, 2757, 2786, 2794, 2868,
    3048], PD.vpd[2].down_proj[40, 324, 350, 827, 849, 1058, 1134, 1496, 1559, 1560, 1566, 1652,
    1664, 1750, 1798, 1862, 2237, 2292, 2341, 2400, 2412, 2550, 2951, 3040, 3150, 3160, 3214, 3271,
    3404, 3503, 3581]
)
h3_2 = node(L[3].head[2])
h3_4 = node(L[3].head[4])

edges(
    embed >> mlp0,
    embed >> mlp2,
    embed >> h3_2,
    embed >> h3_4,
    mlp0 >> mlp2,
    mlp0 >> h3_2,
    mlp0 >> h3_4,
    mlp1 >> mlp2,
    mlp1 >> h3_2,
    mlp1 >> h3_4,
    mlp2 >> h3_2,
    mlp2 >> h3_4,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_2 >> logits,
    h3_4 >> logits,
)
