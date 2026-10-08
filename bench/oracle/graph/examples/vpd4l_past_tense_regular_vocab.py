"""Behavior past_tense.regular (vpd4l): Past tense of a regular verb. Phrasings: 'Every day I {X}.
Yesterday I' | 'Today they {X}. Last week they' | 'I usually {X} in the morning, but last night I'.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[53, 229, 328, 379, 561, 614, 861, 925, 1032, 1275, 1443, 1531, 1564, 1650, 1933,
    2266, 2271, 2445, 2569, 2744, 2761, 2857, 2887, 2901, 2976, 3057], PD.vpd[0].down_proj[42, 75,
    584, 928, 1139, 1148, 1385, 1391, 1411, 1467, 1493, 1541, 1587, 1660, 1925, 1990, 1991, 2041,
    2190, 2196, 2275, 2423, 2424, 2585, 2620, 2696, 2718, 2722, 2818, 2875, 3196, 3210, 3256, 3289,
    3382, 3443, 3455, 3473]
)
mlp1 = node(PD.vpd[1].c_fc[261, 329, 477, 738, 2103, 2179], PD.vpd[1].down_proj[1217, 3478])
h2_3 = node(L[2].head[3])
h2_4 = node(L[2].head[4])
mlp2 = node(PD.vpd[2].c_fc[1108, 1887], PD.vpd[2].down_proj[773, 1403, 1906, 2387, 2874, 3219])
h3_4 = node(L[3].head[4])
mlp3 = node(
    PD.vpd[3].c_fc[87, 161, 200, 315, 843, 903, 906, 945, 1019, 1121, 1145, 1287, 1394, 1511, 1688,
    1869, 1890, 2387, 2831], PD.vpd[3].down_proj[301, 360, 502, 934, 1176, 1362, 1617, 1795, 1943,
    2248, 2862, 3077, 3271]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_4,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_4,
    h2_3 >> mlp3,
    h2_4 >> mlp2,
    h2_4 >> h3_4,
    h2_4 >> mlp3,
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)
