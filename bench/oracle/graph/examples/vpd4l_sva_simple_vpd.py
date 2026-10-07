"""Behavior sva.simple (vpd4l, VPD's subcomponents; M's top token is right on 7% of the targets):
Subject-verb agreement after a determiner and noun; the counterfactual flips the subject's number.

Nodes: per layer's attention and MLP, the VPD subcomponents whose clean contribution, patched into
the counterfactual run, recovers the most of the 2.52 bits per target between the clean and
counterfactual answers, as many as pay for their opaque price (1/2 log2 N bits per weight, N =
2^24): layer 0 attn: 64 subcomponents (19 q_proj, 7 k_proj, 13 v_proj, 25 o_proj), leaving 2.40 bits
when patched together; layer 0 mlp: 32 subcomponents (18 c_fc, 14 down_proj), leaving 0.38 bits when
patched together; layer 1 attn: 32 subcomponents (4 q_proj, 4 k_proj, 9 v_proj, 15 o_proj), leaving
2.45 bits when patched together; layer 1 mlp: 32 subcomponents (18 c_fc, 14 down_proj), leaving 2.35
bits when patched together; layer 2 attn: 128 subcomponents (10 q_proj, 2 k_proj, 55 v_proj, 61
o_proj), leaving 2.14 bits when patched together; layer 2 mlp: 32 subcomponents (14 c_fc, 18
down_proj), leaving 0.64 bits when patched together; layer 3 attn: 128 subcomponents (4 q_proj, 54
v_proj, 70 o_proj), leaving 0.88 bits when patched together; layer 3 mlp: 64 subcomponents (37 c_fc,
27 down_proj), leaving 1.14 bits when patched together. The program lets every write among them
reach every later read.
"""
from mech import node, edges, PD, embed, logits

attn0 = node(
    PD.vpd[0].q_proj[27, 28, 38, 42, 74, 98, 148, 156, 195, 230, 237, 268, 276, 277, 302, 317, 413,
    491, 502], PD.vpd[0].k_proj[59, 126, 214, 309, 357, 453, 463], PD.vpd[0].v_proj[64, 82, 200,
    253, 298, 308, 544, 815, 898, 917, 921, 926, 982], PD.vpd[0].o_proj[26, 73, 201, 268, 275, 329,
    351, 367, 386, 481, 489, 493, 531, 536, 588, 616, 628, 632, 663, 773, 808, 841, 957, 965, 971]
)
mlp0 = node(
    PD.vpd[0].c_fc[123, 747, 1032, 1167, 1185, 1225, 1531, 1604, 1636, 1663, 1759, 1908, 1933, 2084,
    2336, 2645, 2822, 3013], PD.vpd[0].down_proj[295, 622, 657, 862, 920, 1060, 1090, 1777, 2275,
    2367, 3171, 3455, 3491, 3494]
)
attn1 = node(
    PD.vpd[1].q_proj[83, 131, 284, 470], PD.vpd[1].k_proj[104, 112, 423, 480], PD.vpd[1].v_proj[72,
    188, 228, 315, 550, 724, 806, 871, 984], PD.vpd[1].o_proj[113, 189, 191, 262, 273, 319, 386,
    411, 454, 498, 569, 573, 756, 781, 997]
)
mlp1 = node(
    PD.vpd[1].c_fc[157, 345, 370, 402, 570, 687, 723, 766, 1282, 1728, 1786, 1876, 1914, 2179, 2255,
    2765, 2828, 3008], PD.vpd[1].down_proj[515, 594, 612, 1202, 1217, 1403, 1472, 1521, 2245, 2621,
    3034, 3220, 3465, 3478]
)
attn2 = node(
    PD.vpd[2].q_proj[50, 65, 132, 146, 195, 202, 270, 279, 300, 337], PD.vpd[2].k_proj[1, 327],
    PD.vpd[2].v_proj[17, 22, 23, 57, 102, 128, 169, 172, 174, 180, 222, 259, 260, 264, 276, 283,
    294, 318, 326, 341, 345, 346, 352, 376, 377, 397, 404, 417, 436, 453, 473, 491, 572, 581, 631,
    656, 660, 665, 693, 740, 750, 753, 757, 792, 802, 910, 925, 928, 933, 937, 941, 953, 999, 1012,
    1013], PD.vpd[2].o_proj[17, 46, 88, 149, 171, 186, 224, 227, 234, 246, 253, 260, 288, 320, 334,
    337, 338, 378, 415, 428, 437, 443, 449, 458, 476, 501, 509, 523, 546, 577, 587, 598, 609, 626,
    628, 630, 638, 656, 698, 735, 739, 746, 788, 800, 801, 806, 809, 813, 826, 845, 866, 888, 890,
    892, 949, 983, 984, 998, 1000, 1004, 1015]
)
mlp2 = node(
    PD.vpd[2].c_fc[87, 323, 445, 522, 770, 774, 1096, 1291, 1315, 2039, 2089, 2472, 2631, 2757],
    PD.vpd[2].down_proj[666, 708, 945, 1268, 1496, 1581, 1798, 1820, 2237, 3045, 3049, 3076, 3129,
    3240, 3279, 3439, 3493, 3503]
)
attn3 = node(
    PD.vpd[3].q_proj[60, 219, 277, 502], PD.vpd[3].v_proj[44, 61, 90, 135, 214, 228, 274, 286, 299,
    302, 339, 388, 392, 401, 403, 405, 446, 471, 486, 500, 513, 526, 547, 550, 562, 566, 595, 605,
    609, 610, 676, 677, 732, 733, 744, 752, 783, 818, 821, 823, 853, 856, 863, 877, 879, 911, 925,
    958, 960, 982, 985, 996, 1010, 1018], PD.vpd[3].o_proj[0, 1, 9, 16, 27, 28, 38, 53, 68, 74, 81,
    86, 92, 102, 139, 178, 181, 190, 205, 212, 251, 272, 281, 338, 359, 382, 398, 399, 403, 445,
    447, 499, 502, 514, 520, 562, 566, 571, 584, 590, 597, 637, 650, 656, 673, 692, 704, 712, 716,
    720, 731, 734, 749, 756, 766, 776, 782, 785, 802, 805, 848, 855, 863, 882, 938, 941, 948, 969,
    1001, 1016]
)
mlp3 = node(
    PD.vpd[3].c_fc[68, 179, 228, 237, 386, 403, 467, 583, 626, 645, 884, 891, 903, 909, 995, 1010,
    1032, 1035, 1192, 1243, 1444, 1454, 1481, 1673, 1681, 1720, 1777, 1844, 1983, 2057, 2387, 2423,
    2548, 2554, 2570, 2883, 2913], PD.vpd[3].down_proj[61, 152, 341, 468, 873, 1086, 1176, 1232,
    1254, 1341, 1362, 1501, 1531, 1833, 1912, 1943, 1984, 2093, 2244, 2740, 2787, 2905, 3275, 3277,
    3291, 3334, 3377]
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
    mlp1 >> attn2,
    mlp1 >> mlp2,
    mlp1 >> attn3,
    mlp1 >> mlp3,
    attn2 >> mlp2,
    attn2 >> attn3,
    attn2 >> mlp3,
    mlp2 >> attn3,
    mlp2 >> mlp3,
    attn3 >> mlp3,
    attn0 >> logits,
    mlp0 >> logits,
    attn1 >> logits,
    mlp1 >> logits,
    attn2 >> logits,
    mlp2 >> logits,
    attn3 >> logits,
    mlp3 >> logits,
)
