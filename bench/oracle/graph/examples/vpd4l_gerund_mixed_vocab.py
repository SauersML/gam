"""Behavior gerund.mixed (vpd4l): Morphology: the -ing form of a verb, with its spelling changes.
Phrasings: 'go: going\nsit: sitting\n{X}:' | 'I like to {X}. Right now I am'.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[53, 116, 328, 354, 521, 614, 664, 852, 895, 1131, 1308, 1500, 1657, 1747, 1759,
    1778, 1797, 1933, 1973, 2218, 2237, 2402, 2445, 2471, 2554, 2559, 2576, 2683, 2901, 3057],
    PD.vpd[0].down_proj[42, 172, 210, 295, 328, 363, 516, 584, 1092, 1145, 1284, 1423, 1493, 1567,
    1847, 1891, 1925, 1990, 2172, 2248, 2275, 2443, 2571, 2718, 2752, 2819, 2875, 3006, 3196, 3210,
    3256, 3257, 3289, 3455]
)
mlp1 = node(
    PD.vpd[1].c_fc[329, 402, 477, 1028, 2086, 2103, 2493, 2698, 2828, 2898],
    PD.vpd[1].down_proj[515, 672, 926, 950, 1217, 1817]
)
mlp2 = node(PD.vpd[2].c_fc[483, 1108, 1206, 2978], PD.vpd[2].down_proj[65, 2400, 3376, 3384])
h3_4 = node(L[3].head[4])
mlp3 = node(
    PD.vpd[3].c_fc[90, 153, 312, 363, 452, 718, 945, 1145, 1204, 1287, 1511, 1869, 1890, 1904, 2044,
    2099, 2304], PD.vpd[3].down_proj[502, 680, 934, 1049, 1458, 1492, 1685, 1728, 2278, 2486, 2545,
    2710, 2787, 2862, 3377]
)

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
