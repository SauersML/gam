"""Behavior families for the graph oracle (MPD #2951): templates with exact answers.

A family returns variants; a variant is a list of items. An item is a prompt prefix, its answer, and a
counterfactual (a minimal edit of the prefix that changes the answer). An item without a counterfactual
carries a `tmpl` key, and build.py pairs it with another item of the same variant and template whose
prefix and answer have the same token lengths (fact families: swap the entity).

Every family is generated for every model; build.py measures the model and keeps a behavior only when the
answer is the model's top-1 token on most targets. Generators receive `tok`, which answers tokenizer
questions (`tok.single(" word")`: one token), so word lists can be filtered per model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SVA_DIR = Path.home() / "mpd-data/circuits/sva"


@dataclass
class Item:
    prefix: str
    answer: str
    cf_prefix: str | None = None
    cf_answer: str | None = None
    tmpl: int = 0
    accept: list[str] | None = None  # other correct answers (greater-than, rhyme); scoring uses their first token
    cf_accept: list[str] | None = None


@dataclass
class Variant:
    name: str
    description: str
    items: list[Item] = field(default_factory=list)


FAMILIES: dict[str, callable] = {}


def family(fn):
    FAMILIES[fn.__name__] = fn
    return fn


def words(tok, ws, n=1):
    """Keep the words that are `n` tokens with a leading space."""
    return [w for w in ws if tok.count(" " + w) == n]


NAMES_F = "Mary Anna Sarah Emma Lisa Laura Kate Julia Rachel Linda Susan Alice Grace Helen Emily Rose Amy Jane Claire Diana Nancy Karen Sophie Olivia".split()
NAMES_M = "John David Michael James Robert Tom Peter Paul Mark Daniel Kevin George Steven Richard Brian Eric Frank Jack Henry Adam Ryan Chris Scott Andrew".split()
PLACES = "store park school office market station hospital library beach restaurant garden museum".split()
OBJECTS = "drink book ball letter key gift ring pen bag cake flower hat".split()
NOUNS = ("apple chair river lamp window garden pencil mountain bottle candle doctor table spoon forest kitten bridge "
         "rocket guitar island blanket camera dragon ladder mirror pillow tunnel violin wallet basket button cookie "
         "engine feather hammer jacket ocean planet rabbit saddle tiger tomato bucket carpet desert dolphin anchor "
         "barrel castle cherry circus coffee cotton crystal diamond falcon finger garlic glove helmet honey jungle "
         "lemon magnet marble needle orange parrot pepper pirate puzzle rainbow ribbon robot salmon shadow silver "
         "spider sugar summer temple thunder ticket tower velvet wagon whale winter wizard zebra").split()


@family
def ioi(tok, rng):
    """Indirect object identification: the name that appeared once is the answer."""
    names = words(tok, NAMES_F + NAMES_M)
    templates = [
        ("store", "When {A} and {B} went to the {place}, {S} gave a {obj} to"),
        ("argument", "Then, {A} and {B} had a long argument. Afterwards {S} said to"),
        ("work", "Today {A} and {B} were working at the {place}. {S} decided to give a {obj} to"),
    ]
    out = []
    for vname, t in templates:
        v = Variant(f"{vname}", f"IOI ({vname} template): after two names, the second mention of one name (the subject) is followed by the other name (the indirect object).")
        for _ in range(96):
            a, b = rng.sample(names, 2)
            s = rng.choice([a, b])
            io = b if s == a else a
            kw = dict(A=a, B=b, place=rng.choice(PLACES), obj=rng.choice(OBJECTS))
            v.items.append(Item(t.format(S=s, **kw), " " + io, t.format(S=io, **kw), " " + s))
        out.append(v)
    return out


INDUCTION_WORDS = NOUNS + ("blue green quick slow happy angry silent bright golden purple frozen hidden wooden ancient "
                           "gentle bitter").split()


@family
def induction_random(tok, rng):
    """Induction on a repeated list of random words: after the repeat starts, continue the copy."""
    ws = words(tok, INDUCTION_WORDS)
    out = []
    for n in (8, 16):
        v = Variant(f"words{n}", f"Induction: a list of {n} random words is repeated; at a point in the repeat the next word is the one that followed the same word in the first copy.")
        for _ in range(96):
            seq = rng.sample(ws, n + 1)
            first, spare = seq[:n], seq[n]
            k = rng.randrange(2, n - 1)
            alt = list(first)
            alt[k] = spare
            v.items.append(Item(" " + " ".join(first) + " ." + " " + " ".join(first[:k]), " " + first[k],
                                " " + " ".join(alt) + " ." + " " + " ".join(alt[:k]), " " + spare))
        out.append(v)
    return out


PHRASES = ["Yesterday the {a} found a {b} under the old {c} near the river.",
           "My aunt keeps a {a} and a {b} in the {c} behind her house.",
           "The museum shows a {a} made of {b} next to a broken {c}.",
           "Every winter we bring the {a} inside and cover the {b} with a {c}.",
           "He painted the {a} blue and hung a {b} above the {c}."]
FILLERS = ["Nobody knew why.", "It rained all day.", "Then the bell rang.", "We talked about it for hours.", "Later that week, people repeated the story."]


@family
def induction_phrase(tok, rng):
    """Induction on a repeated natural sentence whose slots hold random nouns: the repeat copies the noun."""
    ws = words(tok, NOUNS)
    v = Variant("sentence", "Induction on a repeated sentence: a sentence with random nouns in its slots is repeated after a filler sentence; at a slot the next word is the noun from the first copy.")
    for _ in range(160):
        t = rng.choice(PHRASES)
        a, b, c, d = rng.sample(ws, 4)
        slot = rng.choice(["b", "c"])
        fill = dict(a=a, b=b, c=c)
        alt = dict(fill, **{slot: d})
        cut = lambda f: t[:t.index("{" + slot + "}")].format(**f)
        filler = rng.choice(FILLERS)
        v.items.append(Item(t.format(**fill) + " " + filler + " " + cut(fill).rstrip(), " " + fill[slot],
                            t.format(**alt) + " " + filler + " " + cut(alt).rstrip(), " " + d))
    return [v]


SURNAMES = ("Zalinski Montague Okafor Brennan Castellano Whitfield Abernathy Kowalczyk Delacroix Featherstone Hargreaves "
            "Lindqvist Nakamura Pemberton Quarrington Rasmussen Thornbury Vasquez Wainwright Yarborough Fitzgerald "
            "Galloway Holloway Kingsley Mendoza Ferreira Oyelaran Petrakis Szymanski Tremblay").split()


@family
def name_repeat(tok, rng):
    """Copy a two-part name: a first name seen before is followed by the same surname."""
    firsts = words(tok, NAMES_F + NAMES_M)
    cities = words(tok, "Boston Paris London Chicago Berlin Tokyo Madrid Denver Austin Dublin".split())
    out = []
    v1 = Variant("record", "Name repetition in a record: a first name that appeared earlier with a surname is followed by that surname.")
    v2 = Variant("story", "Name repetition in prose: a person introduced by full name is later mentioned by first name and the surname follows.")
    while len(v1.items) < 128:
        f = rng.choice(firsts)
        s1, s2 = rng.sample(SURNAMES, 2)
        if tok.count(" " + s1) != tok.count(" " + s2):
            continue
        age, city = rng.randrange(21, 79), rng.choice(cities)
        t = "Name: {F} {S}\nAge: {age}\nCity: {city}\n\nName: {F}"
        v1.items.append(Item(t.format(F=f, S=s1, age=age, city=city), " " + s1, t.format(F=f, S=s2, age=age, city=city), " " + s2))
        t2 = "Our new neighbor is {F} {S}. Everyone on the street likes {F}"
        v2.items.append(Item(t2.format(F=f, S=s1), " " + s1, t2.format(F=f, S=s2), " " + s2))
    out += [v1, v2]
    return out


@family
def sva(tok, rng):
    """Subject-verb agreement (Marks et al. pairs): the verb agrees in number with the subject."""
    out = []
    for name, desc in [("simple", "a determiner and noun"), ("nounpp", "a noun followed by a prepositional phrase with a distractor noun"),
                       ("rc", "a noun followed by a relative clause with a distractor noun"), ("within_rc", "the verb inside a relative clause agreeing with that clause's subject")]:
        v = Variant(name, f"Subject-verb agreement after {desc}; the counterfactual flips the subject's number.")
        rows = [json.loads(l) for f in ("train", "test") for l in (SVA_DIR / f"{name}_{f}.json").read_text().splitlines()]
        rng.shuffle(rows)
        seen = set()
        for r in rows:
            if r["clean_prefix"] + r["clean_answer"] in seen:
                continue
            seen.add(r["clean_prefix"] + r["clean_answer"])
            v.items.append(Item(r["clean_prefix"], r["clean_answer"], r["patch_prefix"], r["patch_answer"]))
            if len(v.items) >= 128:
                break
        out.append(v)
    return out


@family
def greater_than(tok, rng):
    """Greater-than on years: the end year of a span must be later than the start year."""
    out = []
    for vname, t in [("war", "The war lasted from the year {C}{Y:02d} to the year {C}"),
                     ("contract", "The contract was valid from {C}{Y:02d} until {C}"),
                     ("reign", "The king ruled from {C}{Y:02d} to {C}")]:
        v = Variant(vname, "Greater-than: the end year starts with the same century, so its last two digits must exceed the start year's.")
        for _ in range(96):
            c = rng.choice([15, 16, 17, 18])
            y, y2 = rng.sample(range(2, 90), 2)
            ok = [f"{z:02d}" for z in range(y + 1, 100)]
            ok2 = [f"{z:02d}" for z in range(y2 + 1, 100)]
            v.items.append(Item(t.format(C=c, Y=y), rng.choice(ok), t.format(C=c, Y=y2), rng.choice(ok2), accept=ok, cf_accept=ok2))
        out.append(v)
    return out


COUNTRIES = [  # country, capital, language, continent, demonym adjective
    ("France", "Paris", "French", "Europe", "French"), ("Germany", "Berlin", "German", "Europe", "German"),
    ("Italy", "Rome", "Italian", "Europe", "Italian"), ("Spain", "Madrid", "Spanish", "Europe", "Spanish"),
    ("Japan", "Tokyo", "Japanese", "Asia", "Japanese"), ("China", "Beijing", "Chinese", "Asia", "Chinese"),
    ("Russia", "Moscow", "Russian", "Europe", "Russian"), ("Egypt", "Cairo", "Arabic", "Africa", "Egyptian"),
    ("Greece", "Athens", "Greek", "Europe", "Greek"), ("Portugal", "Lisbon", "Portuguese", "Europe", "Portuguese"),
    ("Poland", "Warsaw", "Polish", "Europe", "Polish"), ("Austria", "Vienna", "German", "Europe", "Austrian"),
    ("Hungary", "Budapest", "Hungarian", "Europe", "Hungarian"), ("Sweden", "Stockholm", "Swedish", "Europe", "Swedish"),
    ("Norway", "Oslo", "Norwegian", "Europe", "Norwegian"), ("Finland", "Helsinki", "Finnish", "Europe", "Finnish"),
    ("Denmark", "Copenhagen", "Danish", "Europe", "Danish"), ("Ireland", "Dublin", "English", "Europe", "Irish"),
    ("Turkey", "Ankara", "Turkish", "Asia", "Turkish"), ("Iran", "Tehran", "Persian", "Asia", "Iranian"),
    ("Iraq", "Baghdad", "Arabic", "Asia", "Iraqi"), ("India", "Delhi", "Hindi", "Asia", "Indian"),
    ("Thailand", "Bangkok", "Thai", "Asia", "Thai"), ("Vietnam", "Hanoi", "Vietnamese", "Asia", "Vietnamese"),
    ("Indonesia", "Jakarta", "Indonesian", "Asia", "Indonesian"), ("Korea", "Seoul", "Korean", "Asia", "Korean"),
    ("Kenya", "Nairobi", "Swahili", "Africa", "Kenyan"), ("Nigeria", "Abuja", "English", "Africa", "Nigerian"),
    ("Ethiopia", "Addis", "Amharic", "Africa", "Ethiopian"), ("Morocco", "Rabat", "Arabic", "Africa", "Moroccan"),
    ("Peru", "Lima", "Spanish", "South America", "Peruvian"), ("Chile", "Santiago", "Spanish", "South America", "Chilean"),
    ("Argentina", "Buenos", "Spanish", "South America", "Argentine"), ("Brazil", "Brasilia", "Portuguese", "South America", "Brazilian"),
    ("Colombia", "Bogota", "Spanish", "South America", "Colombian"), ("Venezuela", "Caracas", "Spanish", "South America", "Venezuelan"),
    ("Mexico", "Mexico", "Spanish", "North America", "Mexican"), ("Canada", "Ottawa", "English", "North America", "Canadian"),
    ("Cuba", "Havana", "Spanish", "North America", "Cuban"), ("Australia", "Canberra", "English", "Australia", "Australian"),
    ("Belgium", "Brussels", "Dutch", "Europe", "Belgian"), ("Netherlands", "Amsterdam", "Dutch", "Europe", "Dutch"),
    ("Switzerland", "Bern", "German", "Europe", "Swiss"), ("Ukraine", "Kyiv", "Ukrainian", "Europe", "Ukrainian"),
    ("Romania", "Bucharest", "Romanian", "Europe", "Romanian"), ("Bulgaria", "Sofia", "Bulgarian", "Europe", "Bulgarian"),
    ("Serbia", "Belgrade", "Serbian", "Europe", "Serbian"), ("Croatia", "Zagreb", "Croatian", "Europe", "Croatian"),
    ("Pakistan", "Islamabad", "Urdu", "Asia", "Pakistani"), ("Afghanistan", "Kabul", "Pashto", "Asia", "Afghan"),
    ("Syria", "Damascus", "Arabic", "Asia", "Syrian"), ("Lebanon", "Beirut", "Arabic", "Asia", "Lebanese"),
    ("Israel", "Jerusalem", "Hebrew", "Asia", "Israeli"), ("Philippines", "Manila", "Filipino", "Asia", "Filipino"),
    ("Malaysia", "Kuala", "Malay", "Asia", "Malaysian"), ("Mongolia", "Ulaanbaatar", "Mongolian", "Asia", "Mongolian"),
    ("Nepal", "Kathmandu", "Nepali", "Asia", "Nepalese"), ("Ghana", "Accra", "English", "Africa", "Ghanaian"),
    ("Senegal", "Dakar", "French", "Africa", "Senegalese"), ("Uganda", "Kampala", "English", "Africa", "Ugandan"),
    ("Tanzania", "Dodoma", "Swahili", "Africa", "Tanzanian"), ("Iceland", "Reykjavik", "Icelandic", "Europe", "Icelandic"),
    ("Scotland", "Edinburgh", "English", "Europe", "Scottish"), ("Wales", "Cardiff", "Welsh", "Europe", "Welsh"),
    ("Czechia", "Prague", "Czech", "Europe", "Czech"), ("Slovakia", "Bratislava", "Slovak", "Europe", "Slovak"),
    ("Ecuador", "Quito", "Spanish", "South America", "Ecuadorian"), ("Bolivia", "Sucre", "Spanish", "South America", "Bolivian"),
    ("Uruguay", "Montevideo", "Spanish", "South America", "Uruguayan"), ("Jamaica", "Kingston", "English", "North America", "Jamaican"),
    ("Libya", "Tripoli", "Arabic", "Africa", "Libyan"), ("Sudan", "Khartoum", "Arabic", "Africa", "Sudanese"),
]


MIN_PROMPTS = 64  # the suite's floor of prompts per behavior


def fact_variants(rows, templates, desc, name="mixed"):
    """One variant holding every template's phrasing of the relation. Items carry `tmpl` so build.py pairs
    entities only within a template and with equal token lengths."""
    v = Variant(name, desc + " Phrasings: " + " | ".join(repr(t) for _, t in templates) + ".")
    for i, (_, t) in enumerate(templates):
        for subj, ans in rows:
            v.items.append(Item(t.format(X=subj), " " + ans, tmpl=i))
    return [v]


@family
def capital(tok, rng):
    """Country capitals."""
    rows = [(c[0], c[1]) for c in COUNTRIES]
    return fact_variants(rows, [("plain", "The capital of {X} is"), ("qa", "Q: What is the capital of {X}?\nA:"),
                                ("city", "{X}'s capital city is")], "Factual recall: the capital city of a named country.")


@family
def country_language(tok, rng):
    """The language spoken in a country."""
    rows = [(c[0], c[2]) for c in COUNTRIES]
    return fact_variants(rows, [("speak", "In {X}, most people speak"), ("official", "The official language of {X} is")],
                         "Factual recall: the main language of a named country.")


@family
def country_continent(tok, rng):
    """The continent of a country."""
    rows = [(c[0], c[3]) for c in COUNTRIES]
    return fact_variants(rows, [("located", "{X} is a country located in"), ("continent", "The continent where {X} lies is")],
                         "Factual recall: the continent of a named country.")


@family
def demonym(tok, rng):
    """The adjective for a person from a country."""
    rows = [(c[0], c[4]) for c in COUNTRIES]
    return fact_variants(rows, [("born", "She was born and raised in {X}, so she is"), ("citizen", "A citizen of {X} is called")],
                         "Factual recall: the nationality adjective of a named country.")


LANDMARKS = [("the Eiffel Tower", "Paris"), ("the Colosseum", "Rome"), ("Big Ben", "London"), ("the Kremlin", "Moscow"),
             ("the Statue of Liberty", "New"), ("the Brandenburg Gate", "Berlin"), ("the Acropolis", "Athens"),
             ("the Louvre", "Paris"), ("the Sagrada Familia", "Barcelona"), ("the Golden Gate Bridge", "San"),
             ("the Space Needle", "Seattle"), ("the Forbidden City", "Beijing"), ("the Taj Mahal", "Agra"),
             ("the Vatican", "Rome"), ("Buckingham Palace", "London"), ("the Opera House", "Sydney"),
             ("the Hollywood sign", "Los"), ("the Gateway Arch", "St"), ("the Tower of London", "London"),
             ("the Burj Khalifa", "Dubai"), ("Times Square", "New"), ("the Alhambra", "Granada"),
             ("the Little Mermaid statue", "Copenhagen"), ("Red Square", "Moscow"), ("the Parthenon", "Athens"),
             ("the Pantheon", "Rome"), ("the Trevi Fountain", "Rome"), ("Notre Dame Cathedral", "Paris"),
             ("the Prado Museum", "Madrid"), ("the Rijksmuseum", "Amsterdam"), ("the Brooklyn Bridge", "New"),
             ("the White House", "Washington"), ("the Hagia Sophia", "Istanbul"), ("the Blue Mosque", "Istanbul"),
             ("the Uffizi Gallery", "Florence"), ("the Leaning Tower", "Pisa"), ("Tower Bridge", "London"),
             ("the Arc de Triomphe", "Paris"), ("the CN Tower", "Toronto"), ("Wrigley Field", "Chicago"),
             ("Charles Bridge", "Prague"), ("the Petronas Towers", "Kuala"), ("Table Mountain", "Cape"),
             ("the Christ the Redeemer statue", "Rio"), ("the Bund", "Shanghai"), ("Tokyo Tower", "Tokyo"),
             ("the Grand Bazaar", "Istanbul"), ("the Atomium", "Brussels"), ("Schonbrunn Palace", "Vienna"),
             ("the Sydney Harbour Bridge", "Sydney"), ("the Alamo", "San"), ("the Liberty Bell", "Philadelphia"),
             ("the Hermitage Museum", "Saint"), ("Edinburgh Castle", "Edinburgh"), ("the Duomo", "Florence"),
             ("Westminster Abbey", "London"), ("the Smithsonian", "Washington"), ("the Empire State Building", "New"),
             ("the Rialto Bridge", "Venice"), ("St Mark's Basilica", "Venice"), ("the Mona Lisa", "Paris"),
             ("the Reichstag", "Berlin"), ("the Sistine Chapel", "Rome"), ("the Gherkin", "London")]


@family
def landmark_city(tok, rng):
    """The city of a landmark."""
    rows = [(x[0].upper() + x[1:], c) for x, c in LANDMARKS]
    return fact_variants(rows, [("located", "{X} is located in the city of"), ("visit", "Tourists who want to see {X} travel to")],
                         "Factual recall: the city where a named landmark stands.")


@family
def pronoun_gender(tok, rng):
    """A pronoun referring to a named person matches the name's usual gender."""
    f, m = words(tok, NAMES_F), words(tok, NAMES_M)
    out = []
    # names never open the text: a text-initial name lacks the leading space and splits differently
    for vname, t in [("because", "Yesterday {N} went to the {place} because"), ("said", "After {N} finished the work at the {place},"),
                     ("thinks", "That day {N} looked tired at the {place}, so I asked if")]:
        v = Variant(vname, "Gendered pronoun: the next word is the pronoun for the named person; the counterfactual swaps the name's gender.")
        for _ in range(200):
            a, b = rng.choice(f), rng.choice(m)
            p = rng.choice(PLACES)
            if rng.random() < 0.5:
                v.items.append(Item(t.format(N=a, place=p), " she", t.format(N=b, place=p), " he"))
            else:
                v.items.append(Item(t.format(N=b, place=p), " he", t.format(N=a, place=p), " she"))
        out.append(v)
    return out


@family
def bracket_close(tok, rng):
    """Close the innermost open bracket."""
    vs = list("abcdxyzmn")
    out = []
    v1 = Variant("paren", "Bracket closing in arithmetic: after a nested parenthesized expression the next token closes the open parentheses.")
    v2 = Variant("list", "Bracket closing in nested lists: after the last element the next token closes the open lists.")
    v3 = Variant("call", "Bracket closing in function calls: after the last argument the next token closes the open calls.")
    for _ in range(160):
        a, b, c = rng.sample(vs, 3)
        op1, op2 = rng.choice("+-*"), rng.choice("+-*")
        # counterfactual: the inner group closed earlier, so one parenthesis remains open
        # counterfactual: a minus sign replaces the inner opening bracket, so one bracket fewer is open
        v1.items.append(Item(f"y = ({a} {op1} ({b} {op2} {c}", "))", f"y = ({a} {op1} -{b} {op2} {c}", ")"))
        n = [rng.randrange(1, 10) for _ in range(4)]
        v2.items.append(Item(f"x = [{n[0]}, {n[1]}, [{n[2]}, {n[3]}", "]]", f"x = [{n[0]}, {n[1]}, -{n[2]}, {n[3]}", "]"))
        f1, f2 = rng.sample(["max", "min", "abs", "len", "sum", "int", "str", "round"], 2)
        v3.items.append(Item(f"value = {f1}({f2}({a}, {b}", "))", f"value = {f1}({f2}({a}), {b}", ")"))
    return [v1, v2, v3]


@family
def quote_close(tok, rng):
    """Close an open quotation."""
    subj = words(tok, NAMES_F + NAMES_M)
    phrases = ["I will be there soon", "this is not what I ordered", "the train leaves at noon", "we should go home",
               "please close the door", "the answer is simple", "nobody told me", "it is going to rain",
               "I love this song", "the meeting is cancelled", "we won the game", "dinner is ready"]
    v = Variant("said", "Quote closing: after a quoted sentence ends with a period, the next token closes the quotation; the counterfactual opens a parenthesis instead.")
    for _ in range(96):
        n, p = rng.choice(subj), rng.choice(phrases)
        # counterfactual: a parenthesis opens the aside instead of a quotation mark
        v.items.append(Item(f'{n} looked up and said, "{p.capitalize()}.', '"', f'{n} looked up and said, ({p.capitalize()}.', ")"))
    return [v]


@family
def code_variable(tok, rng):
    """Python: reuse the variable that the code defined."""
    names = words(tok, "item value total count result name entry record score price node user word line".split())
    out = []
    v1 = Variant("loop", "Python variable reuse: in a for loop body, the loop variable is printed.")
    v2 = Variant("return", "Python variable reuse: a function returns the local variable it just computed.")
    v3 = Variant("argument", "Python variable reuse: a function body uses its only argument.")
    for _ in range(200):
        a, b = rng.sample(names, 2)
        seq = rng.choice(["data", "rows", "items", "values", "entries"])
        v1.items.append(Item(f"for {a} in {seq}:\n    print(", a, f"for {b} in {seq}:\n    print(", b))
        f = rng.choice(["compute", "combine", "merge", "update", "measure"])
        v2.items.append(Item(f"def {f}(x, y):\n    {a} = x + y\n    return", " " + a, f"def {f}(x, y):\n    {b} = x + y\n    return", " " + b))
        v3.items.append(Item(f"def {f}({a}):\n    if {a} is None:\n        return None\n    return len(", a,
                             f"def {f}({b}):\n    if {b} is None:\n        return None\n    return len(", b))
    return [v1, v2, v3]


@family
def docstring_args(tok, rng):
    """Python: the docstring's Args section lists the signature's arguments in order."""
    names = words(tok, "alpha beta gamma delta width height depth start stop step size count limit offset scale label".split())
    v = Variant("args", "Docstring argument recall: after two documented arguments, the next documented name is the third argument of the signature.")
    for _ in range(96):
        a, b, c, d = rng.sample(names, 4)
        t = 'def run({a}, {b}, {c}):\n    """Run the job.\n\n    Args:\n        {a}: the first setting.\n        {b}: the second setting.\n       '
        v.items.append(Item(t.format(a=a, b=b, c=c), " " + c, t.format(a=a, b=b, c=d), " " + d))
    return [v]


PLURALS_REG = "cat dog car book tree house bird chair pen cup shoe star boat lamp door key bell coin hat bag apple flower river ring desk girl boy friend road farm".split()
PLURALS_IRR = [("mouse", "mice"), ("child", "children"), ("man", "men"), ("woman", "women"), ("foot", "feet"), ("tooth", "teeth"),
               ("goose", "geese"), ("person", "people"), ("knife", "knives"), ("wolf", "wolves"), ("leaf", "leaves"),
               ("wife", "wives"), ("life", "lives"), ("half", "halves"), ("ox", "oxen"), ("cactus", "cacti"), ("box", "boxes"),
               ("bus", "buses"), ("city", "cities"), ("baby", "babies"), ("party", "parties"), ("story", "stories"),
               ("glass", "glasses"), ("church", "churches"), ("dish", "dishes"), ("fox", "foxes"), ("lady", "ladies"),
               ("puppy", "puppies"), ("hero", "heroes"), ("potato", "potatoes"), ("tomato", "tomatoes"), ("loaf", "loaves")]


@family
def plural(tok, rng):
    """Plural nouns after a number."""
    reg = [(w, w + "s") for w in PLURALS_REG]
    ts = [("have", "I have one {X} and you have two"), ("list", "one {X}, two"), ("there", "There is one {X} here and three")]
    return (fact_variants(reg, ts, "Plural of a regular noun after a count above one.", "regular") +
            fact_variants(PLURALS_IRR, ts, "Plural of an irregular or spelling-changing noun after a count above one.", "irregular"))


PAST_REG = "walk talk play jump cook clean paint watch call open help work visit wash climb dance laugh listen pull push start finish travel kick".split()
PAST_IRR = [("go", "went"), ("eat", "ate"), ("see", "saw"), ("run", "ran"), ("write", "wrote"), ("drink", "drank"), ("swim", "swam"),
            ("sing", "sang"), ("drive", "drove"), ("buy", "bought"), ("bring", "brought"), ("think", "thought"), ("teach", "taught"),
            ("catch", "caught"), ("fly", "flew"), ("speak", "spoke"), ("break", "broke"), ("take", "took"), ("give", "gave"),
            ("make", "made"), ("sleep", "slept"), ("leave", "left"), ("feel", "felt"), ("sit", "sat"), ("stand", "stood"),
            ("win", "won"), ("lose", "lost"), ("find", "found"), ("tell", "told"), ("sell", "sold"), ("ride", "rode")]


@family
def past_tense(tok, rng):
    """Past tense after a habitual present."""
    reg = [(w, w + "ed") for w in PAST_REG]
    ts = [("yesterday", "Every day I {X}. Yesterday I"), ("last_week", "Today they {X}. Last week they"),
          ("last_night", "I usually {X} in the morning, but last night I")]
    return (fact_variants(reg, ts, "Past tense of a regular verb.", "regular") + fact_variants(PAST_IRR, ts, "Past tense of an irregular verb.", "irregular"))


ANTONYMS = [("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"), ("up", "down"), ("light", "dark"),
            ("old", "young"), ("rich", "poor"), ("hard", "soft"), ("high", "low"), ("long", "short"), ("open", "closed"),
            ("early", "late"), ("full", "empty"), ("wet", "dry"), ("strong", "weak"), ("clean", "dirty"), ("heavy", "light"),
            ("thick", "thin"), ("loud", "quiet"), ("true", "false"), ("good", "bad"), ("love", "hate"), ("win", "lose"),
            ("buy", "sell"), ("push", "pull"), ("give", "take"), ("inside", "outside"), ("left", "right"), ("north", "south"),
            ("east", "west"), ("day", "night"), ("black", "white"), ("begin", "end"), ("first", "last"), ("cheap", "expensive"),
            ("safe", "dangerous"), ("easy", "difficult"), ("wide", "narrow"), ("deep", "shallow"), ("sweet", "sour"),
            ("smooth", "rough"), ("brave", "cowardly"), ("friend", "enemy"), ("question", "answer"), ("arrive", "leave"),
            ("accept", "reject"), ("increase", "decrease"), ("maximum", "minimum"), ("above", "below"), ("before", "after"),
            ("always", "never"), ("male", "female"), ("tall", "short"), ("near", "far"), ("alive", "dead"), ("asleep", "awake"),
            ("polite", "rude"), ("noisy", "quiet"), ("ancient", "modern"), ("upper", "lower"), ("major", "minor"),
            ("positive", "negative"), ("visible", "invisible"), ("possible", "impossible")]


@family
def antonym(tok, rng):
    """The opposite of a word."""
    return fact_variants(ANTONYMS, [("zero_shot", "The opposite of {X} is"),
                                    ("few_shot", "hot: cold\ntall: short\nyes: no\n{X}:")], "Antonym of a common word.")


@family
def number_successor(tok, rng):
    """Continue an arithmetic sequence of integers."""
    out = []
    for vname, step, desc in [("plus1", 1, "counting up by one"), ("plus2", 2, "counting up by two"), ("minus1", -1, "counting down by one"),
                              ("plus10", 10, "counting up by ten")]:
        v = Variant(vname, f"Number succession: a comma-separated sequence {desc}; the next number continues it.")
        while len(v.items) < 160:
            digits = rng.choice([2, 3])
            s, s2 = (rng.randrange(10 ** (digits - 1), 10 ** digits) for _ in range(2))
            seq = [s + step * i for i in range(5)]
            seq2 = [s2 + step * i for i in range(5)]
            if s == s2 or any(len(str(x)) != digits for x in seq + seq2):
                continue
            v.items.append(Item(", ".join(map(str, seq[:4])) + ",", " " + str(seq[4]), ", ".join(map(str, seq2[:4])) + ",", " " + str(seq2[4])))
        out.append(v)
    return out


NUMWORDS = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split()  # later ordinals split into several tokens


LIST_TEMPLATES = ["{L},", "Here is the order: {L},", "She said them in order: {L},", "Sequence: {L},"]


def sequence_items(lst, rng, cyclic, after_templates=(), n=200):
    """Runs of 2-4 consecutive items in list templates, plus 'the one after X' templates; the counterfactual
    starts the run elsewhere in the same template."""
    items, L = [], len(lst)
    for _ in range(n):
        if after_templates and rng.random() < 0.3:
            t = rng.choice(after_templates)
            i, j = rng.sample(range(L if cyclic else L - 1), 2)
            items.append(Item(t.format(X=lst[i]), " " + lst[(i + 1) % L], t.format(X=lst[j]), " " + lst[(j + 1) % L]))
            continue
        k = rng.randrange(2, 5)
        t = rng.choice(LIST_TEMPLATES)
        span = L if cyclic else L - k
        i, j = rng.sample(range(span), 2)
        run = lambda a: ", ".join(lst[(a + q) % L] for q in range(k))
        items.append(Item(t.format(L=run(i)), " " + lst[(i + k) % L], t.format(L=run(j)), " " + lst[(j + k) % L]))
    return items


@family
def word_successor(tok, rng):
    """Continue a sequence of number words or ordinals."""
    return [Variant("cardinal", "Number-word succession: a run of number words; the next word continues the count.",
                    sequence_items(NUMWORDS, rng, False, ["The number after {X} is", "Count on from {X}: the next one is"])),
            Variant("ordinal", "Ordinal succession: a run of ordinal words; the next word continues the order.",
                    sequence_items(ORDINALS, rng, False, ["The one after the {X} is the"], n=600))]


DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()


@family
def day_successor(tok, rng):
    """The next day of the week."""
    return [Variant("mixed", "Weekday succession: the day that follows in a run of weekdays, or the day after a named one.",
                    sequence_items(DAYS, rng, True, ["The day after {X} is", "If today is {X}, tomorrow is", "Every {X} is followed by"]))]


@family
def month_successor(tok, rng):
    """The next month."""
    return [Variant("mixed", "Month succession: the month that follows in a run of months, or the month after a named one.",
                    sequence_items(MONTHS, rng, True, ["The month after {X} is", "If this month is {X}, next month is", "{X} is followed by"]))]


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@family
def alphabet(tok, rng):
    """The next letter of the alphabet."""
    return [Variant("upper", "Alphabet succession: a run of capital letters; the next letter continues it.",
                    sequence_items(list(LETTERS), rng, False, ["The letter after {X} is"])),
            Variant("lower", "Alphabet succession: a run of lowercase letters; the next letter continues it.",
                    sequence_items(list(LETTERS.lower()), rng, False, ["The letter after {X} is"]))]


ACRO_WORDS = ("Federal National Central Global United Royal Digital Public Advanced Applied Bureau Council Office Agency Institute "
              "Research Science Energy Health Market Network Security Training Systems Planning Development Transport Water "
              "Environment Finance Education Language Medical Ocean Space Trade Union Kingdom Investment Labor Product").split()


@family
def acronym(tok, rng):
    """The acronym of a capitalized name, inside parentheses."""
    ws = words(tok, ACRO_WORDS)
    v = Variant("three_word", "Acronym: after a three-word capitalized name and an open parenthesis, the next tokens spell its initials.")
    for _ in range(128):
        a, b, c = rng.sample(ws, 3)
        b2 = rng.choice([w for w in ws if w[0] != b[0] and w not in (a, c)])
        v.items.append(Item(f"The {a} {b} {c} (", a[0] + b[0] + c[0], f"The {a} {b2} {c} (", a[0] + b2[0] + c[0]))
    return [v]


MC_FACTS = [("What color is the sky on a clear day?", "blue", ["green", "red", "purple"]),
            ("How many legs does a dog have?", "four", ["two", "six", "eight"]),
            ("What is the capital of France?", "Paris", ["London", "Berlin", "Madrid"]),
            ("Which animal says meow?", "cat", ["cow", "dog", "duck"]),
            ("What is frozen water called?", "ice", ["steam", "sand", "glass"]),
            ("Which planet do we live on?", "Earth", ["Mars", "Venus", "Jupiter"]),
            ("What do bees make?", "honey", ["milk", "silk", "wool"]),
            ("How many days are in a week?", "seven", ["five", "six", "ten"]),
            ("What color is grass?", "green", ["blue", "orange", "pink"]),
            ("Which season is the coldest?", "winter", ["summer", "spring", "autumn"]),
            ("What is the opposite of hot?", "cold", ["warm", "big", "fast"]),
            ("Which animal is the largest?", "whale", ["mouse", "cat", "frog"]),
            ("What do cows drink?", "water", ["milk", "oil", "juice"]),
            ("What is 2 plus 2?", "4", ["3", "5", "6"]),
            ("Which fruit is yellow and long?", "banana", ["apple", "grape", "cherry"]),
            ("What do we use to write on paper?", "pen", ["spoon", "fork", "shoe"])]


@family
def mc_letter(tok, rng):
    """Multiple-choice answering with a letter."""
    v = Variant("abcd", "Multiple-choice letter: a question with four lettered options; the answer is the letter of the correct option. The counterfactual moves the correct option to another letter.")
    for _ in range(128):
        q, right, wrong = rng.choice(MC_FACTS)
        opts = [right] + list(wrong)
        rng.shuffle(opts)
        i = opts.index(right)
        j = rng.choice([k for k in range(4) if k != i])
        alt = list(opts)
        alt[i], alt[j] = alt[j], alt[i]
        fmt = lambda o: f"Question: {q}\n" + "".join(f"{L}. {x}\n" for L, x in zip("ABCD", o)) + "Answer:"
        v.items.append(Item(fmt(opts), " " + "ABCD"[i], fmt(alt), " " + "ABCD"[j]))
    return [v]


@family
def arithmetic(tok, rng):
    """Single-digit arithmetic in a few-shot format."""
    out = []
    for vname, sym, fn in [("add", "+", lambda a, b: a + b), ("sub", "-", lambda a, b: a - b), ("mul", "*", lambda a, b: a * b)]:
        v = Variant(vname, f"Arithmetic: after two worked examples, the result of a {vname} problem on small numbers.")
        for _ in range(300):
            x1, y1, x2, y2 = (rng.randrange(2, 10) for _ in range(4))
            if vname == "sub":
                x1, y1, x2, y2 = max(x1, y1), min(x1, y1), max(x2, y2), min(x2, y2)
            shots = f"{x1} {sym} {y1} = {fn(x1, y1)}\n{x2} {sym} {y2} = {fn(x2, y2)}\n"
            a, b = rng.randrange(2, 10), rng.randrange(1, 10)
            if vname == "sub" and b > a:
                a, b = b, a
            b2 = rng.choice([x for x in range(1, 10) if x != b and (vname != "sub" or x <= a)])
            r, r2 = fn(a, b), fn(a, b2)
            if len(str(r)) != len(str(r2)):
                continue
            v.items.append(Item(shots + f"{a} {sym} {b} =", f" {r}", shots + f"{a} {sym} {b2} =", f" {r2}"))
        out.append(v)
    return out


OBJ_COLORS = [("banana", "yellow"), ("lemon", "yellow"), ("snow", "white"), ("milk", "white"), ("grass", "green"), ("leaf", "green"),
              ("sky", "blue"), ("ocean", "blue"), ("blood", "red"), ("tomato", "red"), ("strawberry", "red"), ("coal", "black"),
              ("crow", "black"), ("carrot", "orange"), ("pumpkin", "orange"), ("orange", "orange"), ("lime", "green"),
              ("cucumber", "green"), ("cherry", "red"), ("sun", "yellow"), ("chocolate", "brown"), ("coffee", "brown"),
              ("eggplant", "purple"), ("grape", "purple"), ("flamingo", "pink"), ("frog", "green"), ("cloud", "white"),
              ("tire", "black"), ("fire truck", "red"), ("polar bear", "white"), ("emerald", "green"), ("ruby", "red"),
              ("sapphire", "blue"), ("lavender", "purple"), ("pig", "pink"), ("bark", "brown")]


@family
def object_color(tok, rng):
    """The typical color of an object."""
    return fact_variants(OBJ_COLORS, [("color_of", "The color of a {X} is usually"), ("qa", "Q: What color is a {X}?\nA: It is")],
                         "Commonsense recall: the typical color of a named object.")


@family
def list_copy(tok, rng):
    """Copy a named position from a short list."""
    ws = words(tok, NOUNS)
    v1 = Variant("first", "List indexing: given a list of four words, the first word; the counterfactual swaps the first word with another, so the same words appear.")
    v2 = Variant("last", "List indexing: given a list of four words, the last word; the counterfactual swaps the last word with another, so the same words appear.")
    for _ in range(160):
        a = rng.sample(ws, 4)
        k = rng.randrange(1, 4)
        b1 = list(a)
        b1[0], b1[k] = b1[k], b1[0]
        b2 = list(a)
        b2[3], b2[k - 1] = b2[k - 1], b2[3]
        f = lambda xs, pos: f"List: {', '.join(xs)}.\nThe {pos} word in the list is"
        v1.items.append(Item(f(a, "first"), " " + a[0], f(b1, "first"), " " + b1[0]))
        v2.items.append(Item(f(a, "last"), " " + a[3], f(b2, "last"), " " + b2[3]))
    return [v1, v2]


TRANSLATIONS = {
    "french": [("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"), ("bread", "pain"), ("book", "livre"),
               ("car", "voiture"), ("sun", "soleil"), ("moon", "lune"), ("tree", "arbre"), ("apple", "pomme"), ("friend", "ami"),
               ("day", "jour"), ("night", "nuit"), ("red", "rouge"), ("green", "vert"), ("black", "noir"), ("white", "blanc"),
               ("man", "homme"), ("woman", "femme"), ("child", "enfant"), ("city", "ville"), ("school", "école"),
               ("cheese", "fromage"), ("milk", "lait"), ("fish", "poisson"), ("bird", "oiseau"), ("horse", "cheval"),
               ("door", "porte"), ("window", "fenêtre"), ("sea", "mer"), ("king", "roi"), ("time", "temps"), ("love", "amour")],
    "spanish": [("dog", "perro"), ("cat", "gato"), ("house", "casa"), ("water", "agua"), ("bread", "pan"), ("book", "libro"),
                ("car", "coche"), ("sun", "sol"), ("moon", "luna"), ("tree", "árbol"), ("apple", "manzana"), ("friend", "amigo"),
                ("day", "día"), ("night", "noche"), ("red", "rojo"), ("green", "verde"), ("black", "negro"), ("white", "blanco"),
                ("man", "hombre"), ("woman", "mujer"), ("child", "niño"), ("city", "ciudad"), ("school", "escuela"),
                ("cheese", "queso"), ("milk", "leche"), ("fish", "pescado"), ("bird", "pájaro"), ("horse", "caballo"),
                ("door", "puerta"), ("window", "ventana"), ("sea", "mar"), ("king", "rey"), ("time", "tiempo"), ("love", "amor")],
    "german": [("dog", "Hund"), ("cat", "Katze"), ("house", "Haus"), ("water", "Wasser"), ("bread", "Brot"), ("book", "Buch"),
               ("car", "Auto"), ("sun", "Sonne"), ("moon", "Mond"), ("tree", "Baum"), ("apple", "Apfel"), ("friend", "Freund"),
               ("day", "Tag"), ("night", "Nacht"), ("red", "rot"), ("green", "grün"), ("black", "schwarz"), ("white", "weiß"),
               ("man", "Mann"), ("woman", "Frau"), ("child", "Kind"), ("city", "Stadt"), ("school", "Schule"),
               ("cheese", "Käse"), ("milk", "Milch"), ("fish", "Fisch"), ("bird", "Vogel"), ("horse", "Pferd"),
               ("door", "Tür"), ("window", "Fenster"), ("sea", "Meer"), ("king", "König"), ("time", "Zeit"), ("love", "Liebe")],
}


@family
def translation(tok, rng):
    """Translate an English word in a few-shot list."""
    out = []
    for lang, pairs in TRANSLATIONS.items():
        v = Variant(lang, f"Word translation English to {lang.capitalize()}: after two example pairs, the translation of a new word.")
        for _ in range(400):
            (e1, f1), (e2, f2), (e, f), (e_, f_) = rng.sample(pairs, 4)
            t = "English: {e1}\n{L}: {f1}\n\nEnglish: {e2}\n{L}: {f2}\n\nEnglish: {e}\n{L}:"
            v.items.append(Item(t.format(e1=e1, f1=f1, e2=e2, f2=f2, e=e, L=lang.capitalize()), " " + f,
                                t.format(e1=e1, f1=f1, e2=e2, f2=f2, e=e_, L=lang.capitalize()), " " + f_))
        out.append(v)
    return out


POS_ADJ = "wonderful excellent amazing delightful fantastic superb lovely brilliant great perfect".split()
NEG_ADJ = "terrible awful horrible dreadful disappointing boring bad poor disgusting mediocre".split()


@family
def sentiment(tok, rng):
    """Few-shot sentiment labels."""
    out = []
    for vname, things in [("movie", ["movie", "film", "plot", "acting", "ending"]), ("restaurant", ["food", "service", "soup", "pizza", "dessert"])]:
        v = Variant(vname, f"Sentiment classification ({vname} reviews): after two labeled reviews, the label of a new review; the counterfactual swaps its adjective's polarity.")
        for _ in range(128):
            t1, t2, t3 = rng.sample(things, 3)
            shots = f"Review: The {t1} was {rng.choice(POS_ADJ)}.\nSentiment: positive\n\nReview: The {t2} was {rng.choice(NEG_ADJ)}.\nSentiment: negative\n\n"
            p, n = rng.choice(POS_ADJ), rng.choice(NEG_ADJ)
            if tok.count(" " + p) != tok.count(" " + n):
                continue
            a = Item(shots + f"Review: The {t3} was {p}.\nSentiment:", " positive", shots + f"Review: The {t3} was {n}.\nSentiment:", " negative")
            if rng.random() < 0.5:
                a = Item(a.cf_prefix, a.cf_answer, a.prefix, a.answer)
            v.items.append(a)
        out.append(v)
    return out


RHYMES = [["cat", "hat", "bat", "mat", "rat", "sat", "fat", "flat"], ["ring", "king", "sing", "wing", "thing", "bring", "spring"],
          ["light", "night", "bright", "fight", "sight", "right", "kite", "white"], ["day", "way", "play", "say", "stay", "gray", "may"],
          ["cake", "lake", "make", "bake", "take", "snake", "wake"], ["tree", "bee", "free", "see", "sea", "key", "knee"],
          ["moon", "soon", "spoon", "tune", "noon", "June"], ["rain", "train", "brain", "chain", "pain", "plain", "main"],
          ["book", "look", "cook", "hook", "took", "shook"], ["ball", "call", "fall", "tall", "wall", "hall", "small"],
          ["dog", "frog", "log", "fog", "jog"], ["bed", "red", "head", "bread", "said", "fed"], ["hand", "sand", "band", "land", "stand"],
          ["star", "car", "far", "jar", "bar"], ["fun", "sun", "run", "bun", "one", "done"], ["boat", "coat", "goat", "note", "float"]]


@family
def rhyme(tok, rng):
    """A rhyming word in a few-shot list."""
    v = Variant("pairs", "Rhyme: after example rhyming pairs, a word that rhymes with the given word (any member of its rhyme set counts).")
    for _ in range(128):
        g1, g2, g3, g4 = rng.sample(RHYMES, 4)
        shots = f"Words that rhyme:\n{g1[0]}: {g1[1]}\n{g2[0]}: {g2[1]}\n"
        w, w2 = g3[0], g4[0]
        ok, ok2 = [" " + x for x in g3[1:]], [" " + x for x in g4[1:]]
        v.items.append(Item(shots + f"{w}:", ok[0], shots + f"{w2}:", ok2[0], accept=ok, cf_accept=ok2))
    return [v]


GENDER_PAIRS = [("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"), ("brother", "sister"), ("son", "daughter"),
                ("uncle", "aunt"), ("husband", "wife"), ("prince", "princess"), ("actor", "actress"), ("nephew", "niece"),
                ("grandfather", "grandmother"), ("waiter", "waitress"), ("lord", "lady"), ("gentleman", "lady"), ("he", "she"),
                ("him", "her"), ("his", "her"), ("mr", "mrs"), ("bull", "cow"), ("rooster", "hen"), ("emperor", "empress"),
                ("god", "goddess"), ("hero", "heroine"), ("groom", "bride"), ("monk", "nun"), ("wizard", "witch"), ("duke", "duchess"),
                ("stepfather", "stepmother"), ("boyfriend", "girlfriend"), ("male", "female"), ("dad", "mom"), ("grandson", "granddaughter"),
                ("sir", "madam"), ("stallion", "mare"), ("host", "hostess")]


@family
def gender_analogy(tok, rng):
    """The female counterpart of a male word."""
    return fact_variants(GENDER_PAIRS, [("few_shot", "brother: sister\nfather: mother\n{X}:"),
                                        ("analogy", "Man is to woman as {X} is to")], "Analogy: the female counterpart of a male word.")


COMPARATIVES = [("big", "bigger"), ("small", "smaller"), ("fast", "faster"), ("slow", "slower"), ("tall", "taller"), ("short", "shorter"),
                ("old", "older"), ("young", "younger"), ("hot", "hotter"), ("cold", "colder"), ("long", "longer"), ("strong", "stronger"),
                ("weak", "weaker"), ("rich", "richer"), ("poor", "poorer"), ("good", "better"), ("bad", "worse"), ("happy", "happier"),
                ("easy", "easier"), ("heavy", "heavier"), ("light", "lighter"), ("dark", "darker"), ("high", "higher"), ("low", "lower"),
                ("wide", "wider"), ("deep", "deeper"), ("cheap", "cheaper"), ("clean", "cleaner"), ("loud", "louder"), ("quiet", "quieter"),
                ("near", "nearer"), ("far", "farther"), ("smart", "smarter"), ("hard", "harder"), ("soft", "softer"), ("sweet", "sweeter"),
                ("warm", "warmer"), ("cool", "cooler"), ("thin", "thinner"), ("thick", "thicker"), ("busy", "busier"), ("funny", "funnier"),
                ("large", "larger"), ("nice", "nicer"), ("safe", "safer"), ("brave", "braver"), ("late", "later"), ("early", "earlier")]


@family
def comparative(tok, rng):
    """The comparative form of an adjective."""
    return fact_variants(COMPARATIVES, [("few_shot", "tall: taller\nfast: faster\n{X}:"),
                                        ("sentence", "This box is {X}, but that box is even")], "Morphology: the comparative form of an adjective.")


CATEGORIES = [("rose", "flower"), ("tulip", "flower"), ("daisy", "flower"), ("apple", "fruit"), ("banana", "fruit"), ("mango", "fruit"),
              ("carrot", "vegetable"), ("potato", "vegetable"), ("onion", "vegetable"), ("dog", "animal"), ("lion", "animal"),
              ("elephant", "animal"), ("eagle", "bird"), ("sparrow", "bird"), ("parrot", "bird"), ("salmon", "fish"), ("shark", "fish"),
              ("trout", "fish"), ("oak", "tree"), ("pine", "tree"), ("maple", "tree"), ("piano", "instrument"), ("violin", "instrument"),
              ("guitar", "instrument"), ("hammer", "tool"), ("wrench", "tool"), ("saw", "tool"), ("car", "vehicle"), ("truck", "vehicle"),
              ("bus", "vehicle"), ("red", "color"), ("blue", "color"), ("green", "color"), ("soccer", "sport"), ("tennis", "sport"),
              ("basketball", "sport"), ("iron", "metal"), ("copper", "metal"), ("gold", "metal"), ("shirt", "clothing"),
              ("jacket", "clothing"), ("sock", "clothing"), ("Mars", "planet"), ("Jupiter", "planet"), ("Venus", "planet"),
              ("chess", "game"), ("poker", "game"), ("python", "snake"), ("cobra", "snake"), ("ant", "insect"), ("beetle", "insect"),
              ("bee", "insect"), ("Paris", "city"), ("Tokyo", "city"), ("London", "city"), ("whale", "mammal"), ("dolphin", "mammal"),
              ("cow", "mammal"), ("English", "language"), ("French", "language"), ("Spanish", "language"), ("Monday", "day"),
              ("Tuesday", "day"), ("January", "month"), ("March", "month")]


@family
def category(tok, rng):
    """The category of an object."""
    return fact_variants(CATEGORIES, [("type_of", "The {X} is a type of"), ("few_shot", "cat: animal\nhammer: tool\n{X}:")],
                         "Hypernym: the category a named thing belongs to.")


@family
def uppercase(tok, rng):
    """Uppercase a word in a few-shot list."""
    ws = words(tok, NOUNS)
    v = Variant("few_shot", "Case conversion: after two examples, the uppercase form of a new word.")
    for _ in range(600):
        a, b, c, d = rng.sample(ws, 4)
        t = "{a} -> {A}\n{b} -> {B}\n{c} ->"
        v.items.append(Item(t.format(a=a, A=a.upper(), b=b, B=b.upper(), c=c), " " + c.upper(), t.format(a=a, A=a.upper(), b=b, B=b.upper(), c=d), " " + d.upper()))
    return [v]


BIGRAMS = [("New", "York"), ("United", "States"), ("Los", "Angeles"), ("Hong", "Kong"), ("Sri", "Lanka"), ("Buenos", "Aires"),
           ("Saudi", "Arabia"), ("Puerto", "Rico"), ("Las", "Vegas"), ("San", "Francisco"), ("Costa", "Rica"), ("Prime", "Minister"),
           ("Vice", "President"), ("Supreme", "Court"), ("World", "War"), ("Middle", "East"), ("Wall", "Street"), ("White", "House"),
           ("Barack", "Obama"), ("Donald", "Trump"), ("Hillary", "Clinton"), ("Harry", "Potter"), ("Star", "Wars"), ("Tel", "Aviv"),
           ("Kuala", "Lumpur"), ("Real", "Madrid"), ("Manchester", "United"), ("Notre", "Dame"), ("Pearl", "Harbor"), ("Silicon", "Valley"),
           ("Abu", "Dhabi"), ("Rio", "de"), ("Pacific", "Ocean"), ("Atlantic", "Ocean"), ("Mount", "Everest"), ("Elon", "Musk"),
           ("Steve", "Jobs"), ("Bill", "Gates"), ("Vladimir", "Putin"), ("Angela", "Merkel"), ("Albert", "Einstein"),
           ("Isaac", "Newton"), ("Charles", "Darwin"), ("William", "Shakespeare"), ("Martin", "Luther"), ("Abraham", "Lincoln"),
           ("George", "Washington"), ("Winston", "Churchill"), ("Nelson", "Mandela"), ("Mother", "Teresa"), ("Wikipedia", "article"),
           ("Air", "Force"), ("Bank", "of"), ("Red", "Cross"), ("Coca", "Cola"), ("Burger", "King"), ("Taylor", "Swift"),
           ("Lady", "Gaga"), ("Michael", "Jackson"), ("Elvis", "Presley"), ("Bob", "Dylan"), ("Pink", "Floyd"), ("Rolling", "Stones")]


@family
def bigram(tok, rng):
    """Complete a frequent two-word name."""
    return fact_variants(BIGRAMS, [("sentence", "Last year I read a long article about {X}"), ("list", "Topics: weather, sports, {X}")],
                         "Frequent bigram: the second word of a common multiword name follows its first word.")


IDIOMS = [("as well", "as"), ("in order", "to"), ("on the other", "hand"), ("at the same", "time"), ("in addition", "to"),
          ("for the first", "time"), ("as a matter of", "fact"), ("in front", "of"), ("as soon", "as"), ("in spite", "of"),
          ("at least", "one"), ("by the", "way"), ("in terms", "of"), ("due", "to"), ("according", "to"), ("rather", "than"),
          ("each", "other"), ("so far", "as"), ("as long", "as"), ("on behalf", "of"), ("in case", "of"), ("instead", "of"),
          ("a lot", "of"), ("in the middle", "of"), ("at the end of the", "day"), ("once upon a", "time"), ("better late than", "never"),
          ("last but not", "least"), ("the United", "States"), ("ladies and", "gentlemen"), ("salt and", "pepper"),
          ("bread and", "butter"), ("rock and", "roll"), ("black and", "white"), ("pros and", "cons"), ("up and", "down"),
          ("back and", "forth"), ("more or", "less"), ("sooner or", "later"), ("trial and", "error"), ("safe and", "sound"),
          ("first come, first", "served"), ("over and", "over"), ("give or", "take"), ("now and", "then"), ("here and", "there"),
          ("peace and", "quiet"), ("law and", "order"), ("time and", "again"), ("sick and", "tired"), ("again and", "again"),
          ("by and", "large"), ("odds and", "ends"), ("all of a", "sudden"), ("in the long", "run"), ("for the time", "being"),
          ("on the", "contrary"), ("with respect", "to"), ("in accordance", "with"), ("as far", "as"), ("in charge", "of"),
          ("on top", "of"), ("by means", "of"), ("in favor", "of")]


@family
def idiom(tok, rng):
    """Complete a fixed multiword expression."""
    return fact_variants(IDIOMS, [("sentence", "She told me that, {X}"), ("start", "It was, {X}")],
                         "Fixed expression: the last word of a frequent multiword expression.")


HTML_TAGS = "div span p table li ul ol strong em section header footer button form label td tr h1 h2 nav".split()


@family
def html_close(tok, rng):
    """Close the innermost open HTML tag."""
    v1 = Variant("nested", "HTML tag closing: after an inner element is closed, the next closing tag names the outer element.")
    v2 = Variant("text", "HTML tag closing: after text inside an element and an opening '</', the tag name of that element.")
    for _ in range(200):
        a, b, a2 = rng.sample(HTML_TAGS, 3)
        txt = rng.choice(["Hello", "Contact us", "Read more", "Home", "Price", "Next page", "Login"])
        v1.items.append(Item(f'<{a} class="box"><{b}>{txt}</{b}></', a, f'<{a2} class="box"><{b}>{txt}</{b}></', a2))
        v2.items.append(Item(f'<{a}>{txt}</', a, f'<{a2}>{txt}</', a2))
    return [v1, v2]


BOX_ITEMS = "apple pen ball key coin ring cup book hat shoe watch card".split()


@family
def entity_binding(tok, rng):
    """Entity tracking: recall which container an object was put in."""
    items = words(tok, BOX_ITEMS)
    out = []
    for n in (2, 3):
        v = Variant(f"boxes{n}", f"Entity binding: {n} objects are each placed in a lettered box; asked for one object's box, the answer is its letter. The counterfactual swaps the boxes of the queried object and another.")
        for _ in range(200):
            objs = rng.sample(items, n)
            boxes = rng.sample("ABCDEFG", n)
            q = rng.randrange(n)
            o = rng.choice([i for i in range(n) if i != q])
            alt = list(boxes)
            alt[q], alt[o] = alt[o], alt[q]
            f = lambda bs: " ".join(f"The {x} is in box {b}." for x, b in zip(objs, bs)) + f" The {objs[q]} is in box"
            v.items.append(Item(f(boxes), " " + boxes[q], f(alt), " " + alt[q]))
        out.append(v)
    return out


@family
def first_letter(tok, rng):
    """The first letter of a word."""
    ws = words(tok, NOUNS)
    v = Variant("starts_with", "Spelling: the first letter of a quoted word, as a capital; the counterfactual quotes a word with another first letter.")
    for _ in range(400):
        a, b = rng.sample(ws, 2)
        if a[0] == b[0]:
            continue
        t = rng.choice(['The word "{w}" starts with the letter', 'The first letter of the word "{w}" is the letter'])
        v.items.append(Item(t.format(w=a), " " + a[0].upper(), t.format(w=b), " " + b[0].upper()))  # the models name letters in capitals
    return [v]


ELEMENTS = [("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("copper", "Cu"), ("sodium", "Na"), ("potassium", "K"), ("lead", "Pb"),
            ("tin", "Sn"), ("mercury", "Hg"), ("oxygen", "O"), ("hydrogen", "H"), ("carbon", "C"), ("nitrogen", "N"), ("helium", "He"),
            ("neon", "Ne"), ("calcium", "Ca"), ("magnesium", "Mg"), ("zinc", "Zn"), ("chlorine", "Cl"), ("sulfur", "S"),
            ("phosphorus", "P"), ("aluminum", "Al"), ("silicon", "Si"), ("uranium", "U"), ("platinum", "Pt"), ("nickel", "Ni"),
            ("lithium", "Li"), ("fluorine", "F"), ("argon", "Ar"), ("iodine", "I"), ("titanium", "Ti"), ("cobalt", "Co"),
            ("tungsten", "W"), ("chromium", "Cr"), ("manganese", "Mn"), ("boron", "B")]


@family
def chemical_symbol(tok, rng):
    """The chemical symbol of an element."""
    return fact_variants(ELEMENTS, [("symbol", "The chemical symbol for {X} is"), ("few_shot", "oxygen: O\ncarbon: C\n{X}:")],
                         "Factual recall: the chemical symbol of a named element.")


CURRENCIES = [("Japan", "yen"), ("India", "rupee"), ("Russia", "ruble"), ("China", "yuan"), ("Mexico", "peso"), ("Brazil", "real"),
              ("Britain", "pound"), ("Korea", "won"), ("Thailand", "baht"), ("Poland", "zloty"), ("Sweden", "krona"),
              ("Switzerland", "franc"), ("Israel", "shekel"), ("Turkey", "lira"), ("Vietnam", "dong"), ("Germany", "euro"),
              ("France", "euro"), ("Italy", "euro"), ("Spain", "euro"), ("Argentina", "peso"), ("Chile", "peso"),
              ("Colombia", "peso"), ("Canada", "dollar"), ("Australia", "dollar"), ("Egypt", "pound"), ("Indonesia", "rupiah"),
              ("Denmark", "krone"), ("Norway", "krone"), ("Hungary", "forint"), ("Ukraine", "hryvnia"), ("Malaysia", "ringgit"),
              ("Philippines", "peso"), ("Pakistan", "rupee"), ("Iran", "rial"), ("Kenya", "shilling"), ("Nigeria", "naira")]


@family
def currency(tok, rng):
    """The currency of a country."""
    return fact_variants(CURRENCIES, [("currency", "The currency of {X} is the"), ("pay", "When shopping in {X}, you pay with the local")],
                         "Factual recall: the currency of a named country.")


# Families written for this suite that I do not know from published interpretability work; build.py holds them all out.
NOVEL = {"bracket_type", "clock_add", "json_value", "python_list_index", "roman_numerals", "pattern_ab", "alphabet_skip",
         "month_number", "last_letter", "email_domain"}


@family
def bracket_type(tok, rng):
    """Close the innermost open bracket with the matching bracket type."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    v = Variant("mixed", "Bracket type matching: after a run of opened brackets of mixed types and one inner pair closed, the next token closes the innermost open bracket with its own type; the counterfactual changes that bracket's type.")
    for _ in range(300):
        a, b = rng.sample(list(pairs), 2)
        x, y = rng.sample("abcdxyz", 2)
        c = rng.choice(list(pairs))
        t = "f = {A} {x} + {C} {y} {Cc} +"
        v.items.append(Item(t.format(A=a, x=x, C=c, y=y, Cc=pairs[c]) + f" 1 ", pairs[a],
                            t.format(A=b, x=x, C=c, y=y, Cc=pairs[c]) + f" 1 ", pairs[b]))
    return [v]


@family
def clock_add(tok, rng):
    """Hours on a 12-hour clock."""
    v = Variant("hours_later", "Clock arithmetic: the hour on a 12-hour clock a few hours after a given hour (wrapping past 12), in two phrasings.")
    for _ in range(600):
        h, d = rng.randrange(1, 13), rng.randrange(1, 6)
        h2 = rng.choice([x for x in range(1, 13) if x != h])
        r, r2 = (h + d - 1) % 12 + 1, (h2 + d - 1) % 12 + 1
        t = rng.choice(["If it is {h} o'clock now, then {d} hours later it will be",
                        "The meeting starts at {h} o'clock and lasts {d} hours, so it ends at"])
        v.items.append(Item(t.format(h=h, d=d), f" {r}", t.format(h=h2, d=d), f" {r2}"))
    return [v]


JSON_KEYS = ["name", "city", "color", "pet", "food", "team", "brand", "song"]
JSON_VALS = ["Alice", "Boston", "green", "parrot", "pizza", "Tigers", "Nike", "Yesterday", "Oslo", "violet", "hamster", "sushi", "Lions", "Sony"]


@family
def json_value(tok, rng):
    """Look up a key's value in a JSON object."""
    vals = words(tok, JSON_VALS)
    v = Variant("lookup", "JSON lookup: given a JSON object with three keys, the value of the queried key; the counterfactual queries another key.")
    for _ in range(300):
        ks = rng.sample(JSON_KEYS, 3)
        vs = rng.sample(vals, 3)
        q, q2 = rng.sample(range(3), 2)
        obj = "{" + ", ".join(f'"{k}": "{x}"' for k, x in zip(ks, vs)) + "}"
        t = 'data = {obj}\nprint(data["{k}"])  # prints'
        v.items.append(Item(t.format(obj=obj, k=ks[q]), " " + vs[q], t.format(obj=obj, k=ks[q2]), " " + vs[q2]))
    return [v]


@family
def python_list_index(tok, rng):
    """Index into a Python list literal."""
    v = Variant("index", "Python list indexing: the element at a given index of a four-element list literal; the counterfactual asks for another index.")
    for _ in range(300):
        xs = rng.sample(range(10, 100), 4)
        i, j = rng.sample(range(4), 2)
        t = "x = [{xs}]\nassert x[{i}] =="
        v.items.append(Item(t.format(xs=", ".join(map(str, xs)), i=i), f" {xs[i]}", t.format(xs=", ".join(map(str, xs)), i=j), f" {xs[j]}"))
    return [v]


ROMAN = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX".split()


@family
def roman_numerals(tok, rng):
    """Continue a sequence of Roman numerals."""
    return [Variant("sequence", "Roman numeral succession: a run of Roman numerals; the next numeral continues it.",
                    sequence_items(ROMAN, rng, False, ["The Roman numeral after {X} is"], n=600))]


@family
def pattern_ab(tok, rng):
    """Continue an alternating pattern of two words."""
    ws = words(tok, NOUNS)
    v = Variant("alternate", "Alternation: two words alternate several times; the next word continues the alternation; the counterfactual starts the alternation with the other word.")
    for _ in range(300):
        a, b = rng.sample(ws, 2)
        n = rng.randrange(3, 6)
        v.items.append(Item(" ".join([a, b] * n), " " + a, " ".join([b, a] * n), " " + b))
    return [v]


@family
def alphabet_skip(tok, rng):
    """Continue a run of letters that skips one letter each step."""
    v = Variant("every_other", "Alphabet with a stride of two or three: a run of capital or lowercase letters at a fixed stride; the next letter continues the stride.")
    for _ in range(600):
        st, k = rng.choice([2, 3]), rng.randrange(3, 5)
        al = rng.choice([LETTERS, LETTERS.lower()])
        i, j = rng.sample(range(0, 26 - st * k), 2)
        f = lambda s: ", ".join(al[s + st * q] for q in range(k)) + ","
        v.items.append(Item(f(i), " " + al[i + st * k], f(j), " " + al[j + st * k]))
    return [v]


@family
def month_number(tok, rng):
    """The number of a month."""
    rows = [(m, str(i + 1)) for i, m in enumerate(MONTHS)]
    v = Variant("number", "Month numbering in both directions: the number of a named month, or the month with a given number, in several phrasings.")
    to_num = ["In the calendar, {m} is month number", "Months: January = 1, February = 2. {m} =", "{m} is the month numbered"]
    to_name = ["In the calendar, month number {n} is", "Months: 1 = January, 2 = February. {n} =", "The month numbered {n} is"]
    for _ in range(600):
        i, j = rng.sample(range(12), 2)
        if rng.random() < 0.5:
            t = rng.choice(to_num)
            v.items.append(Item(t.format(m=rows[i][0]), " " + rows[i][1], t.format(m=rows[j][0]), " " + rows[j][1]))
        else:
            t = rng.choice(to_name)
            v.items.append(Item(t.format(n=rows[i][1]), " " + rows[i][0], t.format(n=rows[j][1]), " " + rows[j][0]))
    return [v]


@family
def last_letter(tok, rng):
    """The last letter of a word."""
    ws = words(tok, NOUNS)
    v = Variant("ends_with", "Spelling: after two worked examples, the last letter of a word as a capital; the counterfactual asks about a word with another last letter.")
    for _ in range(400):
        a, b = rng.sample(ws, 2)
        if a[-1] == b[-1]:
            continue
        t = "Last letter of each word:\ncat: T\nlamp: P\n{w}:"
        v.items.append(Item(t.format(w=a), " " + a[-1].upper(), t.format(w=b), " " + b[-1].upper()))
    return [v]


DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "proton", "icloud"]


@family
def email_domain(tok, rng):
    """Copy the domain of a person's e-mail address."""
    names = words(tok, NAMES_F + NAMES_M)
    v = Variant("domain", "Copy from a contact record: the domain of the named person's e-mail address among two contacts; the counterfactual asks for the other contact.")
    for _ in range(800):
        a, b = rng.sample(names, 2)
        da, db = rng.sample(DOMAINS, 2)
        t = "Contacts:\n{a}: {la}@{da}.com\n{b}: {lb}@{db}.com\n\n{q}'s e-mail provider is"
        kw = dict(a=a, b=b, la=a.lower(), lb=b.lower(), da=da, db=db)
        v.items.append(Item(t.format(q=a, **kw), " " + da, t.format(q=b, **kw), " " + db))
    return [v]


@family
def key_value_lookup(tok, rng):
    """Associative recall of a code paired with a word."""
    ws = words(tok, NOUNS)
    v = Variant("codes", "Associative recall: three word=number pairs, then a word; the next token is its number; the counterfactual queries another word.")
    for _ in range(300):
        ks = rng.sample(ws, 3)
        vs = rng.sample(range(10, 100), 3)
        i, j = rng.sample(range(3), 2)
        pre = ", ".join(f"{k}={x}" for k, x in zip(ks, vs)) + ". "
        v.items.append(Item(pre + ks[i] + "=", str(vs[i]), pre + ks[j] + "=", str(vs[j])))
    return [v]


@family
def compare_numbers(tok, rng):
    """The larger of two numbers."""
    v = Variant("larger", "Number comparison: the larger of two two-digit numbers; the counterfactual swaps which one is larger by changing one number.")
    for _ in range(300):
        x, y = rng.sample(range(10, 100), 2)
        lo, hi = min(x, y), max(x, y)
        y2 = rng.randrange(10, lo) if lo > 10 else None
        if y2 is None:
            continue
        t = "Which number is larger, {a} or {b}? Answer:"
        # counterfactual: the larger number is replaced by one below the smaller
        if rng.random() < 0.5:
            v.items.append(Item(t.format(a=lo, b=hi), f" {hi}", t.format(a=lo, b=y2), f" {lo}"))
        else:
            v.items.append(Item(t.format(a=hi, b=lo), f" {hi}", t.format(a=y2, b=lo), f" {lo}"))
    return [v]


@family
def parity(tok, rng):
    """Whether a number is even or odd."""
    v = Variant("even_odd", "Parity: after two worked examples, whether a number is even or odd; the counterfactual changes the last digit's parity.")
    for _ in range(300):
        n = rng.randrange(10, 1000)
        m = n + rng.choice([-1, 1])
        t = "Number: 7 -> odd\nNumber: 10 -> even\nNumber: {n} ->"
        v.items.append(Item(t.format(n=n), " even" if n % 2 == 0 else " odd", t.format(n=m), " even" if m % 2 == 0 else " odd"))
    return [v]


@family
def possessive_pronoun(tok, rng):
    """The possessive pronoun for a named person."""
    f, m = words(tok, NAMES_F), words(tok, NAMES_M)
    objs = ["keys", "phone", "wallet", "umbrella", "notebook", "glasses", "jacket"]
    v = Variant("lost", "Possessive pronoun: after a named person loses something, 'her' or 'his' follows by the name's usual gender; the counterfactual swaps the name's gender.")
    for _ in range(400):
        a, b, o = rng.choice(f), rng.choice(m), rng.choice(objs)
        t = rng.choice(["This morning", "Yesterday", "On Monday", "After lunch", "Last night", "Before the trip"]) + " {n} could not find"
        if rng.random() < 0.5:
            v.items.append(Item(t.format(n=a), " her", t.format(n=b), " his"))
        else:
            v.items.append(Item(t.format(n=b), " his", t.format(n=a), " her"))
    return [v]


@family
def capital_to_country(tok, rng):
    """The country of a capital city (the reverse of capital)."""
    rows = [(c[1], c[0]) for c in COUNTRIES if c[1] not in ("Mexico", "Buenos", "Kuala", "Addis")]
    return fact_variants(rows, [("capital_of", "{X} is the capital city of"), ("qa", "Q: Which country has {X} as its capital?\nA:")],
                         "Factual recall in reverse: the country whose capital is the named city.")


ANIMAL_SOUNDS = [("cow", "moo"), ("dog", "bark"), ("cat", "meow"), ("duck", "quack"), ("sheep", "baa"), ("lion", "roar"),
                 ("pig", "oink"), ("horse", "neigh"), ("owl", "hoot"), ("snake", "hiss"), ("bee", "buzz"), ("frog", "croak"),
                 ("wolf", "howl"), ("mouse", "squeak"), ("rooster", "crow"), ("donkey", "bray"), ("bird", "chirp"), ("goat", "bleat"),
                 ("crow", "caw"), ("dove", "coo"), ("elephant", "trumpet"), ("hyena", "laugh"), ("turkey", "gobble"), ("chick", "peep")]
ANIMAL_BABIES = [("cat", "kitten"), ("dog", "puppy"), ("cow", "calf"), ("sheep", "lamb"), ("horse", "foal"), ("goat", "kid"),
                 ("pig", "piglet"), ("duck", "duckling"), ("frog", "tadpole"), ("bear", "cub"), ("lion", "cub"), ("kangaroo", "joey"),
                 ("deer", "fawn"), ("goose", "gosling"), ("swan", "cygnet"), ("owl", "owlet"), ("eagle", "eaglet"), ("hen", "chick"),
                 ("butterfly", "caterpillar"), ("seal", "pup"), ("whale", "calf"), ("rabbit", "bunny"), ("fox", "kit"), ("tiger", "cub")]


@family
def animal_sound(tok, rng):
    """The sound an animal makes."""
    return fact_variants(ANIMAL_SOUNDS, [("says", "Children learn that the {X} says"), ("few_shot", "dog: woof\ncat: meow\n{X}:"),
                                         ("sound", "The sound a {X} makes is called a")], "Commonsense recall: the sound a named animal makes.")


@family
def animal_baby(tok, rng):
    """The name of an animal's young."""
    return fact_variants(ANIMAL_BABIES, [("called", "A baby {X} is called a"), ("few_shot", "dog: puppy\ncat: kitten\n{X}:"),
                                         ("young", "The young of a {X} is known as a")], "Commonsense recall: the word for a named animal's young.")


TOOLS = [("cut paper", "scissors"), ("drive a nail", "hammer"), ("tell the time", "watch"), ("eat soup", "spoon"), ("write a letter", "pen"),
         ("dig a hole", "shovel"), ("sweep the floor", "broom"), ("unlock a door", "key"), ("see in the dark", "flashlight"),
         ("boil water", "kettle"), ("dry your hair", "towel"), ("brush your teeth", "toothbrush"), ("comb your hair", "comb"),
         ("take a photo", "camera"), ("measure length", "ruler"), ("tighten a bolt", "wrench"), ("stay dry in the rain", "umbrella"),
         ("call a friend", "phone"), ("cut wood", "saw"), ("paint a wall", "brush"), ("water the plants", "hose"), ("open a can", "opener"),
         ("light a candle", "match"), ("climb onto the roof", "ladder")]


@family
def tool_use(tok, rng):
    """The tool for a task."""
    return fact_variants(TOOLS, [("use", "To {X}, you use a"), ("need", "I need to {X}, so please hand me the"),
                                 ("qa", "Q: What do you use to {X}?\nA: A")], "Commonsense recall: the everyday tool used for a named task.")


YES_NO = [("Is the sun hot?", "Yes"), ("Is ice cold?", "Yes"), ("Can fish swim?", "Yes"), ("Do birds have feathers?", "Yes"),
          ("Is water wet?", "Yes"), ("Do cats bark?", "No"), ("Can pigs fly?", "No"), ("Is snow black?", "No"), ("Is fire cold?", "No"),
          ("Do trees have roots?", "Yes"), ("Can a car swim?", "No"), ("Is the moon made of cheese?", "No"), ("Do cows give milk?", "Yes"),
          ("Is grass blue?", "No"), ("Do humans breathe air?", "Yes"), ("Can rocks talk?", "No"), ("Is a banana a fruit?", "Yes"),
          ("Is a whale a fish?", "No"), ("Does the sun rise in the east?", "Yes"), ("Is two larger than five?", "No"),
          ("Do spiders have eight legs?", "Yes"), ("Is Paris in Italy?", "No"), ("Is a tomato blue?", "No"), ("Can dogs read books?", "No"),
          ("Is honey sweet?", "Yes"), ("Do snakes have legs?", "No"), ("Is lemon juice sour?", "Yes"), ("Can babies drive cars?", "No"),
          ("Is the ocean salty?", "Yes"), ("Do plants need light?", "Yes"), ("Is a mouse bigger than an elephant?", "No"),
          ("Is ten more than three?", "Yes")]


@family
def yes_no_facts(tok, rng):
    """Answer a yes/no commonsense question."""
    v = Variant("qa", "Yes/no questions about everyday facts after two answered examples; the counterfactual asks a question with the other answer after the same examples.")
    yes = [q for q, a in YES_NO if a == "Yes"]
    no = [q for q, a in YES_NO if a == "No"]
    for k in range(12):  # each pair of worked examples is one template; build.py pairs questions within it
        sy, sn = rng.choice(yes), rng.choice(no)
        shots = f"Q: {sy}\nA: Yes\nQ: {sn}\nA: No\n" if rng.random() < 0.5 else f"Q: {sn}\nA: No\nQ: {sy}\nA: Yes\n"
        for q, ans in YES_NO:
            if q not in (sy, sn):
                v.items.append(Item(shots + f"Q: {q}\nA:", " " + ans, tmpl=k))
    return [v]


@family
def lowercase(tok, rng):
    """Lowercase a word in a few-shot list."""
    ws = words(tok, NOUNS)
    v = Variant("few_shot", "Case conversion: after two examples, the lowercase form of a new uppercase word.")
    for _ in range(600):
        a, b, c, d = rng.sample(ws, 4)
        t = "{A} -> {a}\n{B} -> {b}\n{C} ->"
        v.items.append(Item(t.format(A=a.upper(), a=a, B=b.upper(), b=b, C=c.upper()), " " + c,
                            t.format(A=a.upper(), a=a, B=b.upper(), b=b, C=d.upper()), " " + d))
    return [v]


@family
def number_pattern(tok, rng):
    """Continue the square numbers or the Fibonacci numbers."""
    sq = [n * n for n in range(1, 16)]
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    tri = [n * (n + 1) // 2 for n in range(1, 16)]
    out = []
    for name, seq, desc in [("squares", sq, "square numbers"), ("fibonacci", fib, "Fibonacci numbers"), ("triangular", tri, "triangular numbers")]:
        v = Variant(name, f"Number patterns: a run of consecutive {desc}; the next number continues it.",
                    sequence_items([str(x) for x in seq], rng, False, [], n=400))
        out.append(v)
    return out


NOVEL |= {"musical_notes", "list_reverse", "markdown_close", "quote_type", "un_prefix", "count_repeats", "planet_order"}
PLANETS = "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune".split()
SOLFEGE = "do re mi fa sol la ti".split()


@family
def planet_order(tok, rng):
    """The next planet outward from the Sun."""
    return [Variant("order", "Planet order: a run of planets outward from the Sun, or the planet after a named one; the next planet continues the order.",
                    sequence_items(PLANETS, rng, False, ["The planet after {X}, going outward from the Sun, is", "Moving outward from {X}, the next planet is",
                                                         "Counting outward from the Sun, the planet that follows {X} is", "Next after {X} in the solar system comes"], n=1000))]


@family
def musical_notes(tok, rng):
    """The next solfege syllable."""
    return [Variant("solfege", "Solfege succession: a run of solfege syllables (do re mi ...); the next syllable continues the scale, wrapping to do.",
                    sequence_items(SOLFEGE, rng, True, ["In the scale, the note after {X} is"], n=400))]


@family
def list_reverse(tok, rng):
    """Write a list in reverse order."""
    ws = words(tok, NOUNS)
    v = Variant("reverse", "List reversal: a four-word list is being written in reverse; after three reversed words the next is the list's first word; the counterfactual reverses a list whose first two words are swapped.")
    for _ in range(300):
        a = rng.sample(ws, 4)
        b = [a[1], a[0]] + a[2:]
        f = lambda xs: f"List: {', '.join(xs)}\nReversed: {', '.join(xs[::-1][:2])},"
        # counterfactual: swapping the first two words changes the next reversed word
        v.items.append(Item(f(a), " " + a[1], f(b), " " + b[1]))
    return [v]


@family
def min_of_list(tok, rng):
    """The smallest number in a short list."""
    v = Variant("smallest", "Minimum: the smallest of three two-digit numbers; the counterfactual lowers another number below it.")
    for _ in range(300):
        xs = rng.sample(range(20, 100), 3)
        i = xs.index(min(xs))
        j = rng.choice([k for k in range(3) if k != i])
        ys = list(xs)
        ys[j] = rng.randrange(10, min(xs))
        t = "The smallest of {a}, {b} and {c} is"
        v.items.append(Item(t.format(a=xs[0], b=xs[1], c=xs[2]), f" {min(xs)}", t.format(a=ys[0], b=ys[1], c=ys[2]), f" {min(ys)}"))
    return [v]


UN_WORDS = "happy kind fair able known usual likely lucky safe true tidy wise clear common even fit".split()


@family
def un_prefix(tok, rng):
    """Negate an adjective with the prefix un-."""
    ws = words(tok, UN_WORDS)
    v = Variant("few_shot", "Morphology: after two examples, the adjective with the prefix un-.")
    for _ in range(400):
        a, b, c, d = rng.sample(ws, 4)
        t = "{a} -> un{a}\n{b} -> un{b}\n{c} ->"
        v.items.append(Item(t.format(a=a, b=b, c=c), " un" + c, t.format(a=a, b=b, c=d), " un" + d))
    return [v]


ELEMENT_NUMBERS = [("hydrogen", 1), ("helium", 2), ("lithium", 3), ("carbon", 6), ("nitrogen", 7), ("oxygen", 8), ("fluorine", 9),
                   ("neon", 10), ("sodium", 11), ("magnesium", 12), ("aluminum", 13), ("silicon", 14), ("sulfur", 16), ("chlorine", 17),
                   ("argon", 18), ("potassium", 19), ("calcium", 20), ("iron", 26), ("copper", 29), ("zinc", 30), ("silver", 47),
                   ("gold", 79), ("mercury", 80), ("lead", 82), ("uranium", 92), ("beryllium", 4), ("boron", 5), ("phosphorus", 15),
                   ("nickel", 28), ("tin", 50), ("iodine", 53), ("platinum", 78)]


@family
def element_number(tok, rng):
    """The atomic number of an element."""
    return fact_variants([(e, str(n)) for e, n in ELEMENT_NUMBERS],
                         [("number", "The atomic number of {X} is"), ("few_shot", "hydrogen: 1\ncarbon: 6\n{X}:"),
                          ("element", "In the periodic table, {X} is element number")],
                         "Factual recall: the atomic number of a named element.")


STATES = [("California", "CA"), ("Texas", "TX"), ("Florida", "FL"), ("Ohio", "OH"), ("Georgia", "GA"), ("Michigan", "MI"),
          ("Virginia", "VA"), ("Washington", "WA"), ("Arizona", "AZ"), ("Colorado", "CO"), ("Oregon", "OR"), ("Nevada", "NV"),
          ("Illinois", "IL"), ("Alabama", "AL"), ("Alaska", "AK"), ("Hawaii", "HI"), ("Idaho", "ID"), ("Iowa", "IA"),
          ("Kansas", "KS"), ("Kentucky", "KY"), ("Louisiana", "LA"), ("Maine", "ME"), ("Maryland", "MD"), ("Minnesota", "MN"),
          ("Missouri", "MO"), ("Montana", "MT"), ("Nebraska", "NE"), ("Utah", "UT"), ("Vermont", "VT"), ("Wisconsin", "WI"),
          ("Wyoming", "WY"), ("Tennessee", "TN"), ("Indiana", "IN"), ("Oklahoma", "OK"), ("Arkansas", "AR"), ("Mississippi", "MS")]


@family
def state_abbreviation(tok, rng):
    """The postal abbreviation of a US state."""
    return fact_variants(STATES, [("list", "Texas: TX\nOhio: OH\n{X}:"), ("code", "The two-letter postal code for {X} is")],
                         "Factual recall: the postal abbreviation of a named US state.")


@family
def markdown_close(tok, rng):
    """Close an open Markdown emphasis marker."""
    ws = words(tok, NOUNS)
    v = Variant("emphasis", "Markdown closing: after a word opened with ** (bold) or _ (italic), the next token closes the same marker; the counterfactual opens the other marker.")
    for _ in range(300):
        a, b = rng.sample(ws, 2)
        t = "Remember to bring the {m}{a}"
        v.items.append(Item(t.format(m="**", a=a), "**", t.format(m="_", a=a), "_"))
    return [v]


@family
def quote_type(tok, rng):
    """Close a quotation with the same quote mark that opened it."""
    ws = words(tok, NOUNS)
    v = Variant("python_string", "Quote matching: a Python string opened with a single or a double quote is closed with the same mark; the counterfactual opens with the other mark.")
    for _ in range(300):
        a, b = rng.sample(ws, 2)
        t = "name = {q}{a}_{b}"
        v.items.append(Item(t.format(q="'", a=a, b=b), "'", t.format(q='"', a=a, b=b), '"'))
    return [v]


CONTRACTIONS = [("do not", "don't"), ("can not", "can't"), ("will not", "won't"), ("is not", "isn't"), ("are not", "aren't"),
                ("was not", "wasn't"), ("were not", "weren't"), ("does not", "doesn't"), ("did not", "didn't"), ("have not", "haven't"),
                ("has not", "hasn't"), ("had not", "hadn't"), ("would not", "wouldn't"), ("should not", "shouldn't"),
                ("could not", "couldn't"), ("I am", "I'm"), ("you are", "you're"), ("they are", "they're"), ("we are", "we're"),
                ("it is", "it's"), ("I will", "I'll"), ("you will", "you'll"), ("I have", "I've"), ("we have", "we've"),
                ("they have", "they've"), ("I would", "I'd"), ("she is", "she's"), ("he is", "he's"), ("let us", "let's"),
                ("must not", "mustn't")]


@family
def contraction(tok, rng):
    """The contracted form of a phrase."""
    return fact_variants(CONTRACTIONS, [("arrow", "cannot -> can't\nI am -> I'm\n{X} ->"), ("short", "The short form of \"{X}\" is \"")],
                         "Morphology: the English contraction of a two-word phrase.")


SUPERLATIVES = [(a, b[:-2] + "est" if b.endswith("er") else b) for a, b in COMPARATIVES if a not in ("good", "bad", "far")] + \
               [("good", "best"), ("bad", "worst"), ("far", "farthest")]


@family
def superlative(tok, rng):
    """The superlative form of an adjective."""
    return fact_variants(SUPERLATIVES, [("few_shot", "tall: tallest\nfast: fastest\n{X}:"), ("sentence", "This box is {X}, but that box is the")],
                         "Morphology: the superlative form of an adjective.")


GERUNDS = [("run", "running"), ("swim", "swimming"), ("sit", "sitting"), ("make", "making"), ("write", "writing"), ("read", "reading"),
           ("play", "playing"), ("sing", "singing"), ("dance", "dancing"), ("cook", "cooking"), ("stop", "stopping"), ("plan", "planning"),
           ("drive", "driving"), ("ride", "riding"), ("walk", "walking"), ("jump", "jumping"), ("eat", "eating"), ("sleep", "sleeping"),
           ("lie", "lying"), ("die", "dying"), ("begin", "beginning"), ("forget", "forgetting"), ("hope", "hoping"), ("come", "coming"),
           ("take", "taking"), ("give", "giving"), ("shop", "shopping"), ("hit", "hitting"), ("cut", "cutting"), ("win", "winning"),
           ("study", "studying"), ("try", "trying"), ("fly", "flying"), ("open", "opening"), ("listen", "listening"), ("travel", "traveling")]


@family
def gerund(tok, rng):
    """The -ing form of a verb."""
    return fact_variants(GERUNDS, [("few_shot", "go: going\nsit: sitting\n{X}:"), ("sentence", "I like to {X}. Right now I am")],
                         "Morphology: the -ing form of a verb, with its spelling changes.")


@family
def count_repeats(tok, rng):
    """Count how many times a word is repeated."""
    ws = words(tok, NOUNS)
    nums = ["two", "three", "four", "five"]
    v = Variant("how_many", "Counting: a word appears two to five times among six words (the rest one other word); the next word is its count; the counterfactual turns one occurrence into the other word.")
    for _ in range(400):
        a, b = rng.sample(ws, 2)
        n = rng.randrange(3, 6)
        pos = sorted(rng.sample(range(6), n))
        seq = [a if i in pos else b for i in range(6)]
        alt = list(seq)
        alt[rng.choice(pos)] = b
        t = "Words: {s}.\nHow many times does {a} appear? Answer:"
        v.items.append(Item(t.format(s=" ".join(seq), a=a), " " + nums[n - 2], t.format(s=" ".join(alt), a=a), " " + nums[n - 3]))
    return [v]


@family
def syllogism(tok, rng):
    """Conclude a categorical syllogism."""
    cats = [("cats", "animal", "an"), ("roses", "flower", "a"), ("oaks", "tree", "a"), ("trucks", "vehicle", "a"),
            ("hammers", "tool", "a"), ("violins", "instrument", "an"), ("apples", "fruit", "a"), ("sparrows", "bird", "a"),
            ("salmon", "fish", "a"), ("carrots", "vegetable", "a")]
    names = words(tok, NAMES_F + NAMES_M + ["Rex", "Fluffy", "Max", "Bella", "Spot"])
    v = Variant("all_are", "Syllogism: 'All X are Y. N is one of the X. So N is a' is followed by Y; the counterfactual changes the category.")
    for _ in range(1200):
        (x, y, art), (x2, y2, art2) = rng.sample(cats, 2)
        if art != art2:
            continue
        n = rng.choice(names)
        t = "All {x} are {y}s. {n} is one of the {x}. Therefore {n} is {art}"
        v.items.append(Item(t.format(x=x, y=y, n=n, art=art), " " + y, t.format(x=x2, y=y2, n=n, art=art), " " + y2))
    return [v]


@family
def transitive_compare(tok, rng):
    """The extreme of a chain of comparisons."""
    names = words(tok, NAMES_F + NAMES_M)
    v = Variant("tallest", "Transitive comparison: two 'taller than' statements chain three people; the tallest is the one at the top of the chain; the counterfactual reverses the chain.")
    for _ in range(400):
        a, b, c = rng.sample(names, 3)
        t = "{p} is taller than {q}. {q} is taller than {r}. The tallest of the three is"
        v.items.append(Item(t.format(p=a, q=b, r=c), " " + a, t.format(p=c, q=b, r=a), " " + c))
    return [v]


@family
def variable_assignment(tok, rng):
    """Follow a chain of Python assignments."""
    names = list("abcdxyzpqr")
    v = Variant("chain", "Variable binding: a value is assigned and copied along a chain of variables; the printed value is the original; the counterfactual changes the value.")
    for _ in range(400):
        a, b, c = rng.sample(names, 3)
        x, y = rng.sample(range(10, 100), 2)
        t = "{a} = {x}\n{b} = {a}\n{c} = {b}\nprint({c})  # prints"
        v.items.append(Item(t.format(a=a, b=b, c=c, x=x), f" {x}", t.format(a=a, b=b, c=c, x=y), f" {y}"))
    return [v]


@family
def two_digit_add(tok, rng):
    """Two-digit addition without carrying into a third digit."""
    v = Variant("add", "Two-digit addition after two worked examples; the counterfactual changes one operand by one.")
    for _ in range(400):
        a, b = rng.randrange(10, 50), rng.randrange(10, 50)
        b2 = b + 1
        shots = "12 + 31 = 43\n25 + 14 = 39\n"
        v.items.append(Item(shots + f"{a} + {b} =", f" {a + b}", shots + f"{a} + {b2} =", f" {a + b2}"))
    return [v]


@family
def object_location(tok, rng):
    """Where an object was put."""
    names = words(tok, NAMES_F + NAMES_M)
    places = ["box", "drawer", "basket", "bag", "cupboard", "closet", "fridge", "garage"]
    objs = ["ball", "book", "key", "apple", "phone", "hat", "cup", "pen"]
    v = Variant("put", "Object tracking: two objects are put in two places; asked where one object is, the answer is its place; the counterfactual swaps the two places.")
    for _ in range(400):
        n = rng.choice(names)
        o1, o2 = rng.sample(objs, 2)
        p1, p2 = rng.sample(places, 2)
        t = "{n} put the {o1} in the {p1} and the {o2} in the {p2}. The {o1} is in the"
        v.items.append(Item(t.format(n=n, o1=o1, o2=o2, p1=p1, p2=p2), " " + p1, t.format(n=n, o1=o1, o2=o2, p1=p2, p2=p1), " " + p2))
    return [v]


IMPORTS = [("numpy", "np"), ("pandas", "pd"), ("matplotlib.pyplot", "plt"), ("tensorflow", "tf"), ("seaborn", "sns"),
           ("networkx", "nx"), ("scipy.stats", "stats"), ("torch.nn", "nn"), ("datetime", "dt"), ("plotly.express", "px"),
           ("polars", "pl"), ("jax.numpy", "jnp"), ("statsmodels.api", "sm"), ("torch.nn.functional", "F"), ("xarray", "xr"),
           ("dask.dataframe", "dd"), ("geopandas", "gpd"), ("plotly.graph_objects", "go"), ("tkinter", "tk"),
           ("multiprocessing", "mp"), ("numpy.typing", "npt"), ("altair", "alt")]


@family
def code_import_alias(tok, rng):
    """The conventional alias of a Python import."""
    rows = IMPORTS
    v = Variant("alias", "Code convention: the conventional alias of an imported Python module, and its later use; several phrasings.")
    ts = ["import {m} as", "import os\nimport {m} as", "# setup\nimport sys\nimport {m} as", "import json\nimport {m} as",
          "from pathlib import Path\nimport {m} as", "import re\nimport {m} as", "#!/usr/bin/env python\nimport {m} as"]
    for i, t in enumerate(ts):
        for m, al in rows:
            v.items.append(Item(t.format(m=m), " " + al, tmpl=i))
    return [v]


@family
def unit_conversion(tok, rng):
    """Convert a quantity between units by a factor of ten, a hundred or a thousand."""
    units = [("meters", "centimeters", 100), ("kilometers", "meters", 1000), ("kilograms", "grams", 1000), ("liters", "milliliters", 1000),
             ("centimeters", "millimeters", 10), ("dollars", "cents", 100), ("hours", "minutes", 60), ("minutes", "seconds", 60),
             ("days", "hours", 24), ("weeks", "days", 7), ("feet", "inches", 12), ("dozens", "items", 12)]
    v = Variant("scale", "Unit conversion: after one worked example of the same units, a small quantity converted to the smaller unit; the counterfactual changes the quantity.")
    for _ in range(800):
        big, small, f = rng.choice(units)
        q, q2 = rng.sample(range(2, 10), 2) if rng.random() < 0.5 else rng.sample(range(11, 20), 2)
        t = "1 {b} = {f} {s}\n{q} {b} ="
        v.items.append(Item(t.format(b=big, f=f, s=small, q=q), f" {q * f}", t.format(b=big, f=f, s=small, q=q2), f" {q2 * f}"))
    return [v]


AUTHORS = [("Hamlet", "Shakespeare"), ("Macbeth", "Shakespeare"), ("Pride and Prejudice", "Austen"), ("Emma", "Austen"),
           ("Oliver Twist", "Dickens"), ("Great Expectations", "Dickens"), ("War and Peace", "Tolstoy"), ("Anna Karenina", "Tolstoy"),
           ("1984", "Orwell"), ("Animal Farm", "Orwell"), ("The Odyssey", "Homer"), ("The Iliad", "Homer"), ("Ulysses", "Joyce"),
           ("Moby-Dick", "Melville"), ("Frankenstein", "Shelley"), ("Dracula", "Stoker"), ("Don Quixote", "Cervantes"),
           ("The Divine Comedy", "Dante"), ("Faust", "Goethe"), ("Crime and Punishment", "Dostoevsky"), ("The Raven", "Poe"),
           ("Leaves of Grass", "Whitman"), ("The Old Man and the Sea", "Hemingway"), ("The Great Gatsby", "Fitzgerald"),
           ("Brave New World", "Huxley"), ("Lolita", "Nabokov"), ("Les Miserables", "Hugo"), ("The Hobbit", "Tolkien"),
           ("Jane Eyre", "Bronte"), ("Walden", "Thoreau"), ("Candide", "Voltaire"), ("Beloved", "Morrison")]


@family
def author_of(tok, rng):
    """The author of a famous work."""
    return fact_variants(AUTHORS, [("written_by", "{X} was written by the author whose surname is"), ("few_shot", "Hamlet: Shakespeare\nEmma: Austen\n{X}:"),
                                   ("qa", "Q: Who wrote {X}? Give the surname.\nA:")], "Factual recall: the surname of the author of a famous work.")
