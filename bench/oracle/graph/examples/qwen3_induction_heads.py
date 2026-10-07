"""Induction on Qwen3-0.6B (behavior induction_random.words8): a list of 8 random words is repeated,
and at a point in the repeat the next word is the one that followed the same word in the first copy.
M gets 96% of the targets right.

Heads were chosen by measurements on the behavior's 96 prompts: each head's attention from the target
to the token after the earlier copy of the current word (induction), from every position to the
previous one, and the cost of removing the head (its output columns zeroed), KL(M || M_e) in bits at
the target. The program claims two stages, each a previous-token head feeding an induction head's key
(not measured here; the checker's edge cuts test it):
- early: L1.H3 and L2.H12 put 86% and 76% of their attention on the previous token; L3.H10 puts 83%
  on the token after the earlier copy (removal 0.10 bits);
- late: L15.H3 and L20.H0 attend to the previous token (84%, 61%; removing L20.H0 costs 0.26 bits);
  L16.H14 and L21.H8 attend to the token after the earlier copy (85%, 71%); L21.H8 matters most of
  any induction head (0.61 bits).
Left to their averages: L1.H5, L2.H6 and the MLPs of layers 0-2, whose removal costs 3.9 to 11.5
bits here; the two heads show neither attention pattern. The claim is that their write barely
depends on the prompt.
"""
from mech import node, edges, L, embed, logits

prev_early = node(L[1].head[3], L[2].head[12])   # write the previous token at each position
induction_early = node(L[3].head[10])            # finds the position after the earlier copy, copies it
prev_late = node(L[15].head[3], L[20].head[0])
induction_late = node(L[16].head[14], L[21].head[8])

edges(
    embed >> prev_early.value,              # the token to move forward one position
    prev_early >> induction_early.key,      # key at j+1: "the word before me was A"
    embed >> induction_early.query,         # query at t: "I am A"
    embed >> induction_early.value,         # value at j+1: B, the word to copy
    embed >> prev_late.value,
    prev_late >> induction_late.key,
    embed >> induction_late.value,
    induction_early >> logits,
    induction_late >> logits,               # B
)
