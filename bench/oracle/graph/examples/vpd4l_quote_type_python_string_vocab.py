"""Behavior quote_type.python_string (vpd4l): Quote matching: a Python string opened with a single or a
double quote is closed with the same mark; the counterfactual opens with the other mark.

Facts measured on the behavior's 0 target tokens (clean prompts; the answer is the next token): what
removing each node does to the answer, and what its own write does to the logits through the direct
path only (no later layers). The edges are this program's claim; the checker tests them.
"""
from mech import node, edges, L, PD, embed, logits

mlp0 = node(
    PD.vpd[0].c_fc[162, 234, 239, 306, 545, 749, 852, 914, 918, 1001, 1115, 1308, 2226, 2494, 2603,
    2816, 2845, 2889, 3013], PD.vpd[0].down_proj[446, 516, 652, 1030, 1348, 1613, 1966, 2233, 2522,
    2670, 3094, 3320, 3371]
)
mlp1 = node(
    PD.vpd[1].c_fc[47, 57, 127, 137, 138, 170, 173, 237, 249, 363, 377, 379, 386, 515, 516, 537,
    545, 570, 593, 622, 633, 672, 725, 766, 817, 923, 1040, 1060, 1191, 1200, 1204, 1404, 1407,
    1414, 1422, 1429, 1446, 1786, 1857, 1873, 1880, 1902, 1960, 2103, 2141, 2179, 2314, 2418, 2443,
    2454, 2459, 2492, 2584, 2587, 2643, 2721, 2767, 2828, 2933, 2941, 2942, 2958, 2997, 3029, 3059],
    PD.vpd[1].down_proj[0, 21, 100, 145, 172, 201, 307, 321, 380, 471, 502, 638, 739, 819, 867, 869,
    896, 913, 950, 1114, 1149, 1151, 1156, 1202, 1204, 1217, 1320, 1325, 1354, 1418, 1436, 1471,
    1504, 1523, 1529, 1595, 1599, 1709, 1755, 1776, 1783, 1926, 2022, 2025, 2187, 2229, 2253, 2295,
    2649, 2699, 2740, 2799, 2824, 2958, 2977, 2978, 3012, 3016, 3080, 3108, 3220, 3402, 3555]
)
h2_1 = node(L[2].head[1])
h2_2 = node(L[2].head[2])
h3_0 = node(L[3].head[0])
h3_1 = node(L[3].head[1])
h3_3 = node(L[3].head[3])
mlp3 = node(PD.vpd[3].c_fc[1657], PD.vpd[3].down_proj[3224])

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> h3_0,
    embed >> h3_1,
    embed >> h3_3,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> h3_0,
    mlp0 >> h3_1,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> h3_0,
    mlp1 >> h3_1,
    mlp1 >> h3_3,
    mlp1 >> mlp3,
    h2_1 >> h3_0,
    h2_1 >> h3_1,
    h2_1 >> h3_3,
    h2_1 >> mlp3,
    h2_2 >> h3_0,
    h2_2 >> h3_1,
    h2_2 >> h3_3,
    h2_2 >> mlp3,
    h3_0 >> mlp3,
    h3_1 >> mlp3,
    h3_3 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    h3_0 >> logits,
    h3_1 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)
