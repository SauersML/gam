"""Behavior possessive_pronoun.lost (vpd4l): Possessive pronoun: after a named person loses something,
'her' or 'his' follows by the name's usual gender; the counterfactual swaps the name's gender.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[226, 327, 1443, 1604, 2042, 2822], PD.vpd[0].down_proj[406, 862, 1036, 2095,
    2424, 2860, 3196, 3455, 3473, 3494]
)
mlp1 = node(PD.vpd[1].c_fc[1728, 2103, 2828], PD.vpd[1].down_proj[3478])
mlp2 = node(
    PD.vpd[2].c_fc[323, 770, 1108, 1206, 1887, 1914, 2415, 2482, 2646, 2757],
    PD.vpd[2].down_proj[65, 324, 1134, 1496, 2237, 2951]
)
h3_4 = node(L[3].head[4])
mlp3 = node(
    PD.vpd[3].c_fc[87, 161, 163, 220, 452, 482, 903, 1825, 1869, 2313, 2372, 2387, 2719],
    PD.vpd[3].down_proj[220, 1782, 3351]
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
