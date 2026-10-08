"""Behavior quote_close.said (vpd4l): Quote closing: after a quoted sentence ends with a period, the
next token closes the quotation; the counterfactual opens a parenthesis instead.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

h0_1 = node(L[0].head[1])
mlp0 = node(
    PD.vpd[0].c_fc[225, 234, 1115, 1553, 2226, 2863], PD.vpd[0].down_proj[302, 663, 1224, 1743,
    1966, 2571, 2610, 2666, 2727, 3220]
)
h1_4 = node(L[1].head[4])
h1_5 = node(L[1].head[5])
mlp1 = node(PD.vpd[1].c_fc[353, 3039])
h2_1 = node(L[2].head[1])
mlp2 = node(
    PD.vpd[2].c_fc[87, 108, 128, 232, 853, 925, 1048, 1096, 1490, 1888, 2128, 2244, 2355, 2451,
    3014, 3033], PD.vpd[2].down_proj[65, 693, 857, 1188, 1195, 1496, 2604, 2868, 2898, 2942, 3040,
    3219, 3261, 3330, 3384, 3403]
)
h3_1 = node(L[3].head[1])
h3_3 = node(L[3].head[3])
mlp3 = node(PD.vpd[3].c_fc[1013])

edges(
    embed >> h0_1,
    embed >> mlp0,
    embed >> h1_4,
    embed >> h1_5,
    embed >> mlp1,
    embed >> h2_1,
    embed >> mlp2,
    embed >> h3_1,
    embed >> h3_3,
    embed >> mlp3,
    h0_1 >> mlp0,
    h0_1 >> h1_4,
    h0_1 >> h1_5,
    h0_1 >> mlp1,
    h0_1 >> h2_1,
    h0_1 >> mlp2,
    h0_1 >> h3_1,
    h0_1 >> h3_3,
    h0_1 >> mlp3,
    mlp0 >> h1_4,
    mlp0 >> h1_5,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> mlp2,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    h1_4 >> mlp1,
    h1_4 >> h2_1,
    h1_4 >> mlp2,
    h1_4 >> h3_1,
    h1_4 >> h3_3,
    h1_4 >> mlp3,
    h1_5 >> mlp1,
    h1_5 >> h2_1,
    h1_5 >> mlp2,
    h1_5 >> h3_1,
    h1_5 >> h3_3,
    h1_5 >> mlp3,
    h2_1 >> mlp2,
    h2_1 >> h3_1,
    h2_1 >> h3_3,
    h2_1 >> mlp3,
    mlp2 >> h3_1,
    mlp2 >> h3_3,
    mlp2 >> mlp3,
    h3_1 >> mlp3,
    h3_3 >> mlp3,
    h0_1 >> logits,
    mlp0 >> logits,
    h1_4 >> logits,
    h1_5 >> logits,
    h2_1 >> logits,
    mlp2 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
)
