from stolcke_pcfg import PCFG, StolckeParser


def build_json_grammar() -> PCFG:
    # A small JSON-like grammar enforcing a fixed schema:
    # {"name": Name, "age": Age, "tags": Tags}
    # Name in {"Alice", "Bob"}
    # Age in {0,1,2}
    # Tags is [ one-or-more of {"x","y"} separated by commas ]

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


def run_demo():
    G = build_json_grammar()
    P = StolckeParser(G, "S")

    tokens = [
        "{",
        '"name"', ":", '"Alice"', ",",
        '"age"', ":", "2", ",",
        '"tags"', ":", "[", '"x"', ",", '"y"', "]",
        "}",
    ]
    print("Allowed at start:", sorted(P.allowed_terminals()))
    for t in tokens:
        ok = P.step(t)
        print(f"step({t!r}) -> {ok}; prefix_logprob={P.prefix_logprob()}")
        if not ok:
            break
    print("accepted:", P.accepted())
    if P.accepted():
        print("sentence_logprob:", P.sentence_logprob())


if __name__ == "__main__":
    run_demo()

