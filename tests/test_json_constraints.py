from stolcke_pcfg import PCFG, StolckeParser


def build_json_grammar() -> PCFG:
    return PCFG(
        [
            ("S", ["Obj"], 1.0),
            ("Obj", ["{", "Members", "}"], 1.0),

            ("Members", ["Pair"], 0.5),
            ("Members", ["Pair", ",", "Members"], 0.5),

            ("Pair", ['"name"', ":", "Name"], 1 / 3),
            ("Pair", ['"age"', ":", "Age"], 1 / 3),
            ("Pair", ['"tags"', ":", "Tags"], 1 / 3),

            ("Name", ['"Alice"'], 0.5),
            ("Name", ['"Bob"'], 0.5),

            ("Age", ["0"], 1 / 3),
            ("Age", ["1"], 1 / 3),
            ("Age", ["2"], 1 / 3),

            ("Tags", ["[", "TagList", "]"], 1.0),

            ("TagList", ["NameTag"], 0.5),
            ("TagList", ["NameTag", ",", "TagList"], 0.5),

            ("NameTag", ['"x"'], 0.5),
            ("NameTag", ['"y"'], 0.5),
        ]
    )


def test_json_schema_example_accepts_sample():
    G = build_json_grammar()
    P = StolckeParser(G, "S")
    tokens = [
        "{",
        '"name"', ":", '"Alice"', ",",
        '"age"', ":", "2", ",",
        '"tags"', ":", "[", '"x"', ",", '"y"', "]",
        "}",
    ]
    for t in tokens:
        assert P.step(t)
    assert P.accepted()

