from __future__ import annotations

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


def run_sequence(tokens: list[str]) -> int:
    p = _demo_parser()
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
    args = parser.parse_args(argv)
    tokens = args.tokens or ["a", "a", "a"]
    return run_sequence(tokens)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
