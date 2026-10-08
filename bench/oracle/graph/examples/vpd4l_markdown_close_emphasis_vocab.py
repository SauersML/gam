"""Behavior markdown_close.emphasis (vpd4l): Markdown closing: after a word opened with ** (bold) or _
(italic), the next token closes the same marker; the counterfactual opens the other marker.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

h0_3 = node(L[0].head[3])
mlp0 = node(
    PD.vpd[0].c_fc[253, 392, 545, 761, 877, 924, 1330, 2103, 2414], PD.vpd[0].down_proj[652, 872,
    1580, 2191, 2525, 2929, 3531]
)
h1_4 = node(L[1].head[4])
mlp1 = node(
    PD.vpd[1].c_fc[6, 13, 28, 83, 100, 109, 139, 149, 197, 199, 222, 334, 345, 358, 429, 431, 442,
    477, 511, 537, 550, 623, 751, 766, 890, 963, 990, 995, 1003, 1011, 1041, 1084, 1086, 1089, 1157,
    1186, 1191, 1224, 1243, 1251, 1272, 1296, 1331, 1372, 1378, 1522, 1548, 1595, 1689, 1752, 1783,
    1796, 1904, 2017, 2069, 2100, 2103, 2116, 2190, 2370, 2392, 2459, 2492, 2639, 2645, 2671, 2717,
    2792, 2833, 2877, 2898, 2915, 2944, 2983, 2992, 3006, 3007, 3039, 3067], PD.vpd[1].down_proj[44,
    55, 87, 100, 258, 269, 312, 317, 503, 515, 516, 672, 697, 856, 887, 913, 972, 1043, 1108, 1151,
    1202, 1269, 1329, 1390, 1651, 1652, 1667, 1855, 1926, 2239, 2245, 2413, 2418, 2497, 2546, 2740,
    2793, 2888, 2976, 3001, 3108, 3220, 3416, 3427, 3478, 3487, 3491, 3505, 3567]
)
h2_1 = node(L[2].head[1])
h2_2 = node(L[2].head[2])
mlp2 = node(
    PD.vpd[2].c_fc[9, 64, 87, 216, 232, 258, 318, 483, 522, 626, 695, 705, 815, 879, 941, 992, 1030,
    1048, 1059, 1096, 1108, 1119, 1173, 1191, 1275, 1315, 1361, 1490, 1497, 1523, 1566, 1605, 1613,
    1626, 1630, 1651, 1666, 1748, 1777, 1785, 1853, 1858, 1866, 1887, 1902, 1915, 2233, 2267, 2417,
    2451, 2467, 2583, 2595, 2624, 2631, 2916, 2935, 3014, 3016, 3048], PD.vpd[2].down_proj[55, 65,
    133, 170, 188, 244, 486, 491, 699, 827, 854, 857, 883, 945, 946, 1004, 1063, 1131, 1140, 1195,
    1268, 1287, 1340, 1395, 1470, 1566, 1609, 1772, 1939, 1982, 1990, 2186, 2219, 2223, 2271, 2292,
    2293, 2376, 2387, 2400, 2428, 2457, 2501, 2527, 2604, 2720, 2865, 2898, 2951, 2973, 3045, 3049,
    3053, 3066, 3129, 3184, 3186, 3219, 3221, 3240, 3279, 3309, 3334, 3404, 3492, 3503, 3546, 3576]
)
h3_0 = node(L[3].head[0])
h3_3 = node(L[3].head[3])

edges(
    embed >> h0_3,
    embed >> mlp0,
    embed >> h1_4,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_3,
    h0_3 >> mlp0,
    h0_3 >> h1_4,
    h0_3 >> mlp1,
    h0_3 >> h2_1,
    h0_3 >> h2_2,
    h0_3 >> mlp2,
    h0_3 >> h3_0,
    h0_3 >> h3_3,
    mlp0 >> h1_4,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_3,
    h1_4 >> mlp1,
    h1_4 >> h2_1,
    h1_4 >> h2_2,
    h1_4 >> mlp2,
    h1_4 >> h3_0,
    h1_4 >> h3_3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_3,
    h2_1 >> mlp2,
    h2_1 >> h3_0,
    h2_1 >> h3_3,
    h2_2 >> mlp2,
    h2_2 >> h3_0,
    h2_2 >> h3_3,
    mlp2 >> h3_0,
    mlp2 >> h3_3,
    h0_3 >> logits,
    mlp0 >> logits,
    h1_4 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_3 >> logits,
)
