"""Behavior bracket_close.call (vpd4l): Bracket closing in function calls: after the last argument the
next token closes the open calls.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

h0_1 = node(L[0].head[1])
mlp0 = node(PD.vpd[0].c_fc[1133, 1153], PD.vpd[0].down_proj[1231, 2984])
h1_0 = node(L[1].head[0])
h1_1 = node(L[1].head[1])
h1_3 = node(L[1].head[3])
h1_4 = node(L[1].head[4])
h2_0 = node(L[2].head[0])
h2_1 = node(L[2].head[1])
mlp2 = node(
    PD.vpd[2].c_fc[67, 87, 108, 232, 317, 401, 483, 574, 679, 774, 820, 853, 879, 925, 935, 1096,
    1108, 1169, 1171, 1191, 1206, 1227, 1246, 1273, 1326, 1362, 1383, 1386, 1525, 1605, 1626, 1666,
    1669, 1748, 1765, 1866, 1888, 2010, 2151, 2244, 2267, 2347, 2348, 2353, 2397, 2415, 2417, 2451,
    2461, 2472, 2482, 2583, 2599, 2618, 2662, 2675, 2806, 2847, 2916, 2972, 3033, 3041, 3044],
    PD.vpd[2].down_proj[143, 146, 174, 238, 468, 489, 523, 615, 626, 645, 666, 699, 741, 790, 904,
    1063, 1088, 1089, 1140, 1173, 1197, 1395, 1521, 1560, 1573, 1589, 1671, 1763, 1843, 1885, 1919,
    1939, 1990, 2070, 2223, 2236, 2271, 2293, 2314, 2317, 2341, 2400, 2428, 2512, 2604, 2888, 2926,
    3040, 3184, 3212, 3255, 3261, 3271, 3301, 3309, 3325, 3334, 3359, 3376, 3403, 3404, 3493, 3494,
    3535, 3546]
)
h3_1 = node(L[3].head[1])
h3_3 = node(L[3].head[3])

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
