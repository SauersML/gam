"""Possessive pronoun: after the named person loses something, her or his by the name's gender. The names' genders are what the model recalls."""
from mech import bind, claim

FEMALE = {
    'Alice', 'Amy', 'Anna', 'Claire', 'Diana', 'Emily', 'Emma', 'Grace', 'Helen', 'Jane', 'Julia',
    'Karen', 'Kate', 'Laura', 'Linda', 'Lisa', 'Mary', 'Nancy', 'Olivia', 'Rachel', 'Rose',
    'Sarah', 'Sophie', 'Susan',
}
MALE = {
    'Adam', 'Andrew', 'Brian', 'Chris', 'Daniel', 'David', 'Eric', 'Frank', 'George', 'Henry',
    'Jack', 'James', 'John', 'Kevin', 'Mark', 'Michael', 'Paul', 'Peter', 'Richard', 'Robert',
    'Ryan', 'Scott', 'Steven', 'Tom',
}


def female(tokens):
    # whether the latest name is a woman's (None before any name)
    out, last = [], None
    for tok in tokens:
        if tok.strip() in FEMALE:
            last = True
        elif tok.strip() in MALE:
            last = False
        out.append(last)
    return out


def answer(tokens, female):
    # the pronoun for that person
    return [None if f is None else ' her' if f else ' his' for f in female]
