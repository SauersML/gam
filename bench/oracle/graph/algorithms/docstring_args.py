"""Docstring arguments: after documenting the first arguments of a function, the next documented line names
the next argument of its signature."""


def params(tokens):
    # the signature's arguments, in order
    out = []
    for t in range(len(tokens)):
        text = "".join(tokens[: t + 1])
        head = text.split("def ", 1)[1] if "def " in text else ""
        inner = head.split("(", 1)[1].split(")", 1)[0] if "(" in head and ")" in head else ""
        out.append([p.strip() for p in inner.split(",") if p.strip()])
    return out


def documented(tokens):
    # the documented lines after "Args:": (indentation, name)
    out = []
    for t in range(len(tokens)):
        text = "".join(tokens[: t + 1])
        lines = text.split("Args:", 1)[1].split("\n")[1:-1] if "Args:" in text else []
        out.append([(line[: len(line) - len(line.lstrip())], line.split(":")[0].strip()) for line in lines if ":" in line])
    return out


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, params, documented):
    # a new line at the same indentation, naming the first argument not documented yet
    out = []
    for t in range(len(tokens)):
        names = [name for _, name in documented[t]]
        left = [p for p in params[t] if p not in names]
        if not documented[t] or not left:
            out.append(None)
        else:
            out.append(rest("\n" + documented[t][0][0] + left[0], "".join(tokens[: t + 1])))
    return out
