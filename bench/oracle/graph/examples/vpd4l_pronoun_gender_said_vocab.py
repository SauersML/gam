"""Behavior pronoun_gender.said (vpd4l): Gendered pronoun: the next word is the pronoun for the named
person; the counterfactual swaps the name's gender.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[226, 327, 1443, 2042, 2822], PD.vpd[0].down_proj[406, 862, 1036, 2095, 2424,
    2696, 2860, 3196, 3455, 3473, 3494]
)
mlp1 = node(PD.vpd[1].c_fc[2828])
mlp2 = node(
    PD.vpd[2].c_fc[323, 770, 1096, 1108, 1914, 2267, 2482, 2646, 2757], PD.vpd[2].down_proj[65, 324,
    1134, 1559, 2237, 2951, 3160]
)
h3_4 = node(L[3].head[4])
mlp3 = node(
    PD.vpd[3].c_fc[161, 163, 220, 326, 1121, 1869, 2364, 2387], PD.vpd[3].down_proj[300, 301, 1750,
    1782, 2017, 2133, 2862, 2933]
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
    mlp2 >> h3_4,
    mlp2 >> mlp3,
    h3_4 >> mlp3,
    mlp0 >> logits,
    mlp2 >> logits,
    h3_4 >> logits,
    mlp3 >> logits,
)
