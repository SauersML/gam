"""Behavior pronoun_gender.said (vpd4l, VPD's subcomponents; M's top token is right on 98% of the
targets): Gendered pronoun: the next word is the pronoun for the named person; the counterfactual
swaps the name's gender.

Nodes: per layer's attention and MLP, the VPD subcomponents whose clean contribution, patched into
the counterfactual run, recovers the most of the 1.00 bits per target between the clean and
counterfactual answers, as many as pay for their opaque price (1/2 log2 N bits per weight, N =
2^24): layer 0 attn: 128 subcomponents (23 q_proj, 23 k_proj, 33 v_proj, 49 o_proj), leaving 0.76
bits when patched together; layer 0 mlp: 16 subcomponents (5 c_fc, 11 down_proj), leaving 0.23 bits
when patched together; layer 1 attn: 64 subcomponents (4 q_proj, 3 k_proj, 24 v_proj, 33 o_proj),
leaving 0.90 bits when patched together; layer 1 mlp: 1 subcomponents (1 c_fc), leaving 0.97 bits
when patched together; layer 2 attn: 64 subcomponents (1 q_proj, 2 k_proj, 39 v_proj, 22 o_proj),
leaving 0.75 bits when patched together; layer 2 mlp: 16 subcomponents (9 c_fc, 7 down_proj),
leaving 0.86 bits when patched together; layer 3 attn: 8 subcomponents (3 v_proj, 5 o_proj), leaving
0.14 bits when patched together; layer 3 mlp: 16 subcomponents (8 c_fc, 8 down_proj), leaving 0.81
bits when patched together. The program lets every write among them reach every later read.
"""
from mech import node, edges, PD, embed, logits

attn0 = node(
    PD.vpd[0].q_proj[8, 16, 37, 57, 83, 94, 120, 179, 196, 256, 263, 266, 292, 330, 347, 357, 358,
    374, 457, 470, 472, 489, 503], PD.vpd[0].k_proj[15, 16, 24, 54, 81, 99, 127, 132, 138, 184, 196,
    207, 301, 309, 316, 326, 371, 399, 418, 461, 488, 494, 506], PD.vpd[0].v_proj[82, 99, 160, 204,
    205, 286, 304, 345, 358, 370, 430, 442, 478, 491, 495, 546, 550, 573, 585, 586, 597, 635, 668,
    703, 712, 750, 837, 868, 869, 870, 937, 948, 1003], PD.vpd[0].o_proj[8, 17, 27, 40, 51, 57, 69,
    73, 119, 210, 232, 284, 328, 329, 402, 408, 411, 441, 453, 471, 493, 536, 614, 632, 635, 650,
    662, 677, 687, 705, 734, 755, 757, 841, 844, 883, 908, 937, 960, 965, 984, 986, 992, 999, 1004,
    1007, 1013, 1016, 1018]
)
mlp0 = node(
    PD.vpd[0].c_fc[226, 327, 1443, 2042, 2822], PD.vpd[0].down_proj[406, 862, 1036, 2095, 2424,
    2696, 2860, 3196, 3455, 3473, 3494]
)
attn1 = node(
    PD.vpd[1].q_proj[243, 268, 356, 497], PD.vpd[1].k_proj[147, 215, 495], PD.vpd[1].v_proj[11, 43,
    56, 72, 102, 115, 127, 136, 179, 296, 389, 471, 543, 568, 592, 648, 674, 725, 745, 840, 859,
    908, 984, 1000], PD.vpd[1].o_proj[37, 79, 163, 187, 219, 220, 255, 260, 292, 300, 311, 323, 336,
    338, 344, 374, 383, 417, 482, 546, 573, 578, 626, 676, 807, 818, 860, 865, 877, 902, 928, 1004,
    1022]
)
mlp1 = node(
    PD.vpd[1].c_fc[2828]
)
attn2 = node(
    PD.vpd[2].q_proj[279], PD.vpd[2].k_proj[1, 224], PD.vpd[2].v_proj[11, 34, 49, 130, 151, 172,
    218, 288, 311, 345, 346, 363, 391, 420, 446, 473, 475, 499, 507, 526, 531, 537, 632, 660, 730,
    750, 762, 769, 781, 844, 854, 866, 882, 893, 941, 991, 999, 1012, 1017], PD.vpd[2].o_proj[4, 19,
    32, 88, 146, 171, 183, 213, 234, 268, 350, 367, 404, 424, 448, 686, 784, 962, 984, 990, 995,
    1015]
)
mlp2 = node(
    PD.vpd[2].c_fc[323, 770, 1096, 1108, 1914, 2267, 2482, 2646, 2757], PD.vpd[2].down_proj[65, 324,
    1134, 1559, 2237, 2951, 3160]
)
attn3 = node(
    PD.vpd[3].v_proj[526, 676, 1010], PD.vpd[3].o_proj[205, 281, 574, 594, 776]
)
mlp3 = node(
    PD.vpd[3].c_fc[161, 163, 220, 326, 1121, 1869, 2364, 2387], PD.vpd[3].down_proj[300, 301, 1750,
    1782, 2017, 2133, 2862, 2933]
)

edges(
    embed >> attn0,
    embed >> mlp0,
    embed >> attn1,
    embed >> mlp1,
    embed >> attn2,
    embed >> mlp2,
    embed >> attn3,
    embed >> mlp3,
    attn0 >> mlp0,
    attn0 >> attn1,
    attn0 >> mlp1,
    attn0 >> attn2,
    attn0 >> mlp2,
    attn0 >> attn3,
    attn0 >> mlp3,
    mlp0 >> attn1,
    mlp0 >> mlp1,
    mlp0 >> attn2,
    mlp0 >> mlp2,
    mlp0 >> attn3,
    mlp0 >> mlp3,
    attn1 >> mlp1,
    attn1 >> attn2,
    attn1 >> mlp2,
    attn1 >> attn3,
    attn1 >> mlp3,
    attn2 >> mlp2,
    attn2 >> attn3,
    attn2 >> mlp3,
    mlp2 >> attn3,
    mlp2 >> mlp3,
    attn3 >> mlp3,
    attn0 >> logits,
    mlp0 >> logits,
    attn1 >> logits,
    attn2 >> logits,
    mlp2 >> logits,
    attn3 >> logits,
    mlp3 >> logits,
)
