import argparse
import sys

from . import PCFG, StolckeParser


def _demo_parser() -> StolckeParser:
    # Simple left-recursive grammar: S -> S 'a' | 'a'
    g = PCFG([
        ("S", ["S", "a"], 0.4),
        ("S", ["a"], 0.6),
    ])
    return StolckeParser(g, "S")


def run_sequence(tokens: list[str], *, eliminate_units: bool = True) -> int:
    p = StolckeParser(_demo_parser().G, "S", eliminate_units=eliminate_units)
    print(f"Allowed at start: {sorted(p.allowed_terminals())}")
    for t in tokens:
        ok = p.step(t)
        print(f"step('{t}') -> {ok} prefix_logprob={p.prefix_logprob()}")
        if not ok:
            print("Token not allowed at this position; stopping.")
            break
    print("accepted?", p.accepted())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stolcke-parser",
        description=(
            "Demo CLI for the stolcke_pcfg package. By default runs a simple "
            "left-recursive grammar S -> S 'a' | 'a' over an input sequence."
        ),
    )
    parser.add_argument(
        "tokens",
        nargs="*",
        help="Space-separated input tokens (default: 'a a a').",
    )
    parser.add_argument(
        "--no-unit-elim",
        action="store_true",
        help="Disable unit-production elimination (not recommended).",
    )
    args = parser.parse_args(argv)
    tokens = args.tokens or ["a", "a", "a"]
    return run_sequence(tokens, eliminate_units=not args.no_unit_elim)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
