"""HTML tag closing: after "</" the next tag closes the innermost element still open."""
from mech import bind, claim


def open_tags(tokens):
    # the stack of open elements in the text: "<name" opens, "</name" closes
    out = []
    for t in range(len(tokens)):
        stack = []
        for piece in "".join(tokens[: t + 1]).split("<")[1:]:
            name = ""
            for c in piece.lstrip("/"):
                if not c.isalnum():
                    break
                name += c
            if piece.startswith("/"):
                if stack and stack[-1] == name:
                    stack.pop()
            elif name:
                stack.append(name)
        out.append(stack)
    return out


def answer(tokens, open_tags):
    # after "</", the innermost open element
    return [s[-1] if "".join(tokens[: t + 1]).endswith("</") and s else None for t, s in enumerate(open_tags)]
