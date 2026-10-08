"""Behavior past_tense.irregular (vpd4l): Past tense of an irregular verb. Phrasings: 'Every day I {X}.
Yesterday I' | 'Today they {X}. Last week they' | 'I usually {X} in the morning, but last night I'.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[62, 66, 209, 229, 328, 507, 614, 664, 1032, 1225, 1443, 1564, 1650, 1933, 2266,
    2271, 2371, 2402, 2445, 2518, 2559, 2583, 2901, 3057], PD.vpd[0].down_proj[4, 42, 75, 105, 210,
    313, 370, 584, 928, 1015, 1284, 1385, 1397, 1411, 1438, 1467, 1493, 1541, 1587, 1681, 1925,
    1991, 2041, 2065, 2196, 2275, 2424, 2530, 2585, 2687, 2818, 2875, 2942, 3012, 3196, 3210, 3256,
    3289, 3393, 3443]
)
mlp1 = node(PD.vpd[1].c_fc[2179], PD.vpd[1].down_proj[1217])
h2_3 = node(L[2].head[3])
h2_4 = node(L[2].head[4])
h3_4 = node(L[3].head[4])
mlp3 = node(
    PD.vpd[3].c_fc[161, 906, 1014, 1019, 1121, 1145, 1287, 1688, 1869, 1878, 2387, 2831],
    PD.vpd[3].down_proj[934, 1362, 2862, 3271]
)

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_4,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_4,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_4,
    mlp1 >> mlp3,
    h2_3 >> h3_4,
    h2_3 >> mlp3,
    h2_4 >> h3_4,
    h2_4 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)
