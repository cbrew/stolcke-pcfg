import numpy as np

from stolcke_pcfg import PCFG, ConstrainedDecoderAdapter, StolckeParser
from stolcke_pcfg.hf_logits import GrammarConstrainedLogitsProcessor


def build_simple_grammar() -> PCFG:
    # S -> 'a' | 'b'
    return PCFG([
        ("S", ["a"], 0.5),
        ("S", ["b"], 0.5),
    ])


def test_processor_masks_numpy_array_batch():
    G = build_simple_grammar()
    P = StolckeParser(G, "S")

    # Fake tokenizer bridges: 'a'->1, 'b'->2; others unmapped
    def id2s(i: int) -> str:
        table = {1: "a", 2: "b"}
        return table.get(i, "?")

    def s2id(s: str):
        table = {"a": 1, "b": 2}
        return table.get(s)

    adapter = ConstrainedDecoderAdapter(P, id2s, s2id)
    proc = GrammarConstrainedLogitsProcessor(adapter)

    # Two-batch scores, vocab size 5
    scores = np.zeros((2, 5), dtype=float)
    out = proc(None, scores)

    # At start, allowed {'a','b'} => ids {1,2} should remain 0; others -inf-ish
    assert np.isfinite(out[:, 1]).all()
    assert np.isfinite(out[:, 2]).all()
    assert (out[:, 0] < -1e20).all()
    assert (out[:, 3] < -1e20).all()
    assert (out[:, 4] < -1e20).all()


def test_processor_respects_parser_progress():
    # Grammar: S -> 'a' 'b' (single derivation)
    G = PCFG([
        ("S", ["a", "b"], 1.0),
    ])
    P = StolckeParser(G, "S")

    def id2s(i: int) -> str:
        return {1: "a", 2: "b"}.get(i, "?")

    def s2id(s: str):
        return {"a": 1, "b": 2}.get(s)

    adapter = ConstrainedDecoderAdapter(P, id2s, s2id)
    proc = GrammarConstrainedLogitsProcessor(adapter)

    scores = np.zeros((1, 5), dtype=float)
    # Step 0: only 'a' (id=1) allowed
    out0 = proc(None, scores)
    assert np.isfinite(out0[0, 1]) and (out0[0, [0, 2, 3, 4]] < -1e20).all()

    # After consuming 'a', only 'b' (id=2) allowed
    assert adapter.step_with_token(1)
    out1 = proc(None, scores)
    assert np.isfinite(out1[0, 2]) and (out1[0, [0, 1, 3, 4]] < -1e20).all()
