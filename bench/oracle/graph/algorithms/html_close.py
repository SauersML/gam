"""HTML tag closing: after "</" the next tag closes the innermost element still open."""
from mech import bind, claim


def open_tags(tokens):
    # the stack of open elements: a tag name after "<" opens, after "</" closes
    out, stack = [], []
    for t, tok in enumerate(tokens):
        before = tokens[t - 1] if t else ""
        if before.endswith("</"):
            if stack and stack[-1] == tok.strip():
                stack.pop()
        elif before.endswith("<") and tok.strip().isalpha():
            stack.append(tok.strip())
        out.append(list(stack))
    return out


def answer(tokens, open_tags):
    # after "</", the innermost open element
    return [open_tags[t][-1] if tokens[t].endswith("</") and open_tags[t] else None for t in range(len(tokens))]
