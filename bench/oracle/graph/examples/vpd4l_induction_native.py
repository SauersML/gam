"""Induction on vpd4l: when the current token appeared earlier in the context, predict the token that
followed it there ("A B ... A -> B").

The heads are the repo's induction circuit (mpd_library_contracts_2951.rs, measured on vpd4l):
a previous-token head (L1.H1) writes, at every position, the token before it; the induction head
(L2.H4) reads that write through its key, so its query for the current token A matches the position
right after the earlier A, and it copies the token there (B); a copy head (L3.H5) reads the
induction head through its value and writes B again.

The other nodes come from patching on induction_random.words8 (the counterfactual changes the word
that follows A in the first copy; the clean and counterfactual answers differ by 7.4 bits per target):
running the counterfactual with one component's output taken from the clean prompt recovers 7.4 bits
for layer 0's MLP, which acts as part of the token embedding, 6.3 for L2.H4, 3.1 for L2.H3, 2.8 for
L3.H5, 1.7 for L3.H4 and 4.4 for layer 3's MLP. Patched alone, L1.H1 recovers 0.05 bits; the program
claims it matters through the induction heads' keys (its write differs from the counterfactual's
only at the position after the edited word).
"""
from mech import node, edges, L, embed, logits

tokens = node(L[0].mlp)          # the token's own features: layer 0's MLP extends the embedding
prev = node(L[1].head[1])        # previous-token head: attends from t to t-1, writes x[t-1]
induction = node(L[2].head[4])   # attends from the current A to the position after the earlier A
induction2 = node(L[2].head[3])  # a second head of the same kind (recovers 3.1 bits alone)
copy = node(L[3].head[5])        # reads the induction heads through its value, writes B again
copy2 = node(L[3].head[4])
out = node(L[3].mlp)             # turns the copied features into B's logit

edges(
    embed >> tokens,
    embed >> prev, tokens >> prev,               # the token moved forward one position
    prev >> induction.key, prev >> induction2.key,  # key at j+1: "the token before me was A"
    tokens >> induction.query, tokens >> induction2.query,  # query at t: "I am A"
    tokens >> induction.value, tokens >> induction2.value,  # value at j+1: B, the token to copy
    induction >> copy.value, induction2 >> copy.value,
    induction >> copy2.value, induction2 >> copy2.value,
    induction >> out, induction2 >> out, copy >> out, copy2 >> out,
    induction >> logits, induction2 >> logits,    # B, written directly
    copy >> logits, copy2 >> logits, out >> logits,
)
